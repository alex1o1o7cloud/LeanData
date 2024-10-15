import Mathlib

namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3132_313273

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3132_313273


namespace NUMINAMATH_CALUDE_square_perimeter_l3132_313232

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 200) (h2 : side^2 = area) :
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3132_313232


namespace NUMINAMATH_CALUDE_age_problem_l3132_313250

/-- Theorem: Given the age relationships and total age, prove b's age --/
theorem age_problem (a b c d e : ℝ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  e = d / 2 + 3 →
  a + b + c + d + e = 70 →
  b = 16.625 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3132_313250


namespace NUMINAMATH_CALUDE_lower_limit_of_set_D_l3132_313270

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetD : Set ℕ := {n : ℕ | isPrime n ∧ n ≤ 25}

theorem lower_limit_of_set_D (rangeD : ℕ) (h_range : rangeD = 12) :
  ∃ (lower : ℕ), lower = 13 ∧ 
    (∀ n ∈ SetD, n ≥ lower) ∧
    (∃ m ∈ SetD, m = lower) ∧
    (∃ max ∈ SetD, max - lower = rangeD) :=
sorry

end NUMINAMATH_CALUDE_lower_limit_of_set_D_l3132_313270


namespace NUMINAMATH_CALUDE_square_of_product_l3132_313294

theorem square_of_product (p q : ℝ) : (-3 * p * q)^2 = 9 * p^2 * q^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l3132_313294


namespace NUMINAMATH_CALUDE_functional_equation_l3132_313236

theorem functional_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ x : ℝ, f x = 1/2 - x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_l3132_313236


namespace NUMINAMATH_CALUDE_square_differences_sum_l3132_313235

theorem square_differences_sum : 1010^2 - 990^2 - 1005^2 + 995^2 - 1002^2 + 998^2 = 28000 := by
  sorry

end NUMINAMATH_CALUDE_square_differences_sum_l3132_313235


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3132_313215

theorem hyperbola_eccentricity :
  let hyperbola := fun (x y : ℝ) => x^2 / 5 - y^2 / 4 = 1
  ∃ (e : ℝ), e = (3 * Real.sqrt 5) / 5 ∧
    ∀ (x y : ℝ), hyperbola x y → 
      e = Real.sqrt ((x^2 / 5) + (y^2 / 4)) / Real.sqrt (x^2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3132_313215


namespace NUMINAMATH_CALUDE_bugs_farthest_apart_l3132_313272

/-- Two circles with a common point and bugs moving on them -/
structure TwoCirclesWithBugs where
  /-- Diameter of the larger circle in cm -/
  d_large : ℝ
  /-- Diameter of the smaller circle in cm -/
  d_small : ℝ
  /-- The two circles have exactly one common point -/
  common_point : Prop
  /-- Bugs start at the common point and move at the same speed -/
  bugs_same_speed : Prop

/-- The number of laps completed by the bug on the smaller circle when the bugs are farthest apart -/
def farthest_apart_laps (circles : TwoCirclesWithBugs) : ℕ :=
  4

/-- Theorem stating that the bugs are farthest apart after 4 laps on the smaller circle -/
theorem bugs_farthest_apart (circles : TwoCirclesWithBugs) 
    (h1 : circles.d_large = 48) 
    (h2 : circles.d_small = 30) : 
  farthest_apart_laps circles = 4 := by
  sorry

end NUMINAMATH_CALUDE_bugs_farthest_apart_l3132_313272


namespace NUMINAMATH_CALUDE_john_total_distance_l3132_313220

/-- Calculates the total distance driven given two separate trips with different speeds and durations. -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that John's total driving distance is 235 miles. -/
theorem john_total_distance :
  total_distance 35 2 55 3 = 235 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l3132_313220


namespace NUMINAMATH_CALUDE_divisibility_by_1961_l3132_313248

theorem divisibility_by_1961 (n : ℕ) : ∃ k : ℤ, 5^(2*n) * 3^(4*n) - 2^(6*n) = k * 1961 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1961_l3132_313248


namespace NUMINAMATH_CALUDE_correct_average_unchanged_l3132_313214

theorem correct_average_unchanged (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 18 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = (n : ℚ) * initial_avg :=
by sorry

end NUMINAMATH_CALUDE_correct_average_unchanged_l3132_313214


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l3132_313227

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 6 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l3132_313227


namespace NUMINAMATH_CALUDE_two_valid_colorings_l3132_313293

/-- Represents the three possible colors for a hexagon. -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a position in the hexagonal grid. -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents the hexagonal grid. -/
def HexGrid := Position → Color

/-- Checks if two positions are adjacent in the hexagonal grid. -/
def are_adjacent (p1 p2 : Position) : Bool :=
  sorry

/-- Checks if a coloring of the hexagonal grid is valid. -/
def is_valid_coloring (grid : HexGrid) : Prop :=
  (grid ⟨1, 1⟩ = Color.Red) ∧
  (∀ p1 p2, are_adjacent p1 p2 → grid p1 ≠ grid p2)

/-- The number of valid colorings for the hexagonal grid. -/
def num_valid_colorings : ℕ :=
  sorry

/-- Theorem stating that there are exactly 2 valid colorings of the hexagonal grid. -/
theorem two_valid_colorings : num_valid_colorings = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_colorings_l3132_313293


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3132_313282

/-- The product of fractions in the sequence -/
def fraction_product : ℕ → ℚ
| 0 => 3 / 1
| n + 1 => fraction_product n * ((3 * (n + 1) + 6) / (3 * (n + 1)))

/-- The last term in the sequence -/
def last_term : ℚ := 3003 / 2997

/-- The number of terms in the sequence -/
def num_terms : ℕ := 999

theorem fraction_product_simplification :
  fraction_product num_terms * last_term = 1001 := by
  sorry


end NUMINAMATH_CALUDE_fraction_product_simplification_l3132_313282


namespace NUMINAMATH_CALUDE_kyle_bottles_l3132_313245

theorem kyle_bottles (bottle_capacity : ℕ) (additional_bottles : ℕ) (total_stars : ℕ) :
  bottle_capacity = 15 →
  additional_bottles = 3 →
  total_stars = 75 →
  ∃ (initial_bottles : ℕ), initial_bottles = 2 ∧ 
    (initial_bottles + additional_bottles) * bottle_capacity = total_stars :=
by sorry

end NUMINAMATH_CALUDE_kyle_bottles_l3132_313245


namespace NUMINAMATH_CALUDE_smallest_w_proof_l3132_313210

def smallest_w : ℕ := 79092

theorem smallest_w_proof :
  ∀ w : ℕ,
  w > 0 →
  (∃ k : ℕ, 1452 * w = 2^4 * 3^3 * 13^3 * k) →
  w ≥ smallest_w :=
by
  sorry

#check smallest_w_proof

end NUMINAMATH_CALUDE_smallest_w_proof_l3132_313210


namespace NUMINAMATH_CALUDE_part_one_part_two_l3132_313276

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one (m : ℝ) :
  (∀ x : ℝ, f m x < 0) → -4 < m ∧ m ≤ 0 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → f m x > -m + x - 1) → m > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3132_313276


namespace NUMINAMATH_CALUDE_min_value_theorem_l3132_313271

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.rpow 2 (x - 3) = Real.rpow (1 / 2) y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.rpow 2 (a - 3) = Real.rpow (1 / 2) b → 
    1 / x + 4 / y ≤ 1 / a + 4 / b) ∧ 1 / x + 4 / y = 3 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l3132_313271


namespace NUMINAMATH_CALUDE_product_even_odd_is_odd_l3132_313238

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem product_even_odd_is_odd (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_product_even_odd_is_odd_l3132_313238


namespace NUMINAMATH_CALUDE_not_on_inverse_proportion_graph_l3132_313212

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

def point_on_graph (p : ℝ × ℝ) : Prop :=
  inverse_proportion p.1 p.2

theorem not_on_inverse_proportion_graph :
  point_on_graph (-2, -3) ∧
  point_on_graph (-3, -2) ∧
  ¬point_on_graph (1, 5) ∧
  point_on_graph (4, 1.5) :=
sorry

end NUMINAMATH_CALUDE_not_on_inverse_proportion_graph_l3132_313212


namespace NUMINAMATH_CALUDE_integer_k_not_dividing_binomial_coefficient_l3132_313241

theorem integer_k_not_dividing_binomial_coefficient (k : ℤ) : 
  k ≠ 1 ↔ ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ n ∈ S, ¬(n + k : ℤ) ∣ (Nat.choose (2 * n) n : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_integer_k_not_dividing_binomial_coefficient_l3132_313241


namespace NUMINAMATH_CALUDE_solution_set_equals_open_interval_l3132_313206

def solution_set : Set ℝ := {x | x^2 - 9*x + 14 < 0 ∧ 2*x + 3 > 0}

theorem solution_set_equals_open_interval :
  solution_set = Set.Ioo 2 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_open_interval_l3132_313206


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3132_313228

theorem gcd_of_three_numbers : Nat.gcd 12222 (Nat.gcd 18333 36666) = 6111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3132_313228


namespace NUMINAMATH_CALUDE_triangle_side_range_l3132_313244

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def hasTwoSolutions (t : Triangle) : Prop :=
  t.b * Real.sin t.A < t.a ∧ t.a < t.b

-- Theorem statement
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.A = π / 6)
  (h3 : hasTwoSolutions t) :
  1 < t.a ∧ t.a < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3132_313244


namespace NUMINAMATH_CALUDE_binomial_expansion_5_plus_4_cubed_l3132_313207

theorem binomial_expansion_5_plus_4_cubed : (5 + 4)^3 = 729 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_5_plus_4_cubed_l3132_313207


namespace NUMINAMATH_CALUDE_sum_of_divisors_57_l3132_313246

/-- The sum of all positive divisors of 57 is 80. -/
theorem sum_of_divisors_57 : (Finset.filter (λ x => 57 % x = 0) (Finset.range 58)).sum id = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_57_l3132_313246


namespace NUMINAMATH_CALUDE_intersection_M_N_l3132_313265

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3132_313265


namespace NUMINAMATH_CALUDE_least_common_multiple_3_4_6_7_8_l3132_313264

theorem least_common_multiple_3_4_6_7_8 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) ∧ (8 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_3_4_6_7_8_l3132_313264


namespace NUMINAMATH_CALUDE_convex_polygon_partition_l3132_313275

/-- A convex polygon represented by its side lengths -/
structure ConvexPolygon where
  sides : List ℝ
  sides_positive : ∀ s ∈ sides, s > 0
  convexity : ∀ s ∈ sides, s ≤ (sides.sum / 2)

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := p.sides.sum

/-- A partition of the sides of a polygon into two sets -/
structure Partition (p : ConvexPolygon) where
  set1 : List ℝ
  set2 : List ℝ
  partition_complete : set1 ∪ set2 = p.sides
  partition_disjoint : set1 ∩ set2 = ∅

theorem convex_polygon_partition (p : ConvexPolygon) :
  ∃ (part : Partition p), |part.set1.sum - part.set2.sum| ≤ (perimeter p) / 3 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_partition_l3132_313275


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l3132_313208

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) →
  (¬ (d ≤ 7)) →
  (¬ (d ≤ 6)) →
  (d ≠ 9) →
  7 < d ∧ d < 8 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l3132_313208


namespace NUMINAMATH_CALUDE_first_load_theorem_l3132_313239

/-- Calculates the number of pieces of clothing in the first load -/
def first_load_pieces (total_pieces : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  total_pieces - (num_equal_loads * pieces_per_equal_load)

/-- Theorem stating that given 59 total pieces of clothing, with 9 equal loads of 3 pieces each,
    the number of pieces in the first load is 32. -/
theorem first_load_theorem :
  first_load_pieces 59 9 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_load_theorem_l3132_313239


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_parallel_result_point_coordinates_l3132_313217

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_result :
  {x : ℝ | f' x = 4} = {1, -1} :=
by sorry

theorem point_coordinates :
  {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = f x ∧ f' x = 4} = {(1, 0), (-1, -4)} :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_parallel_result_point_coordinates_l3132_313217


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l3132_313240

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the given binary number is equal to 51 in decimal -/
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l3132_313240


namespace NUMINAMATH_CALUDE_opposite_signs_inequality_l3132_313219

theorem opposite_signs_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_inequality_l3132_313219


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_l3132_313263

theorem sum_of_max_min_a (a b : ℝ) (h : a - 2*a*b + 2*a*b^2 + 4 = 0) :
  ∃ (a_max a_min : ℝ),
    (∀ x : ℝ, (∃ y : ℝ, x - 2*x*y + 2*x*y^2 + 4 = 0) → x ≤ a_max ∧ x ≥ a_min) ∧
    a_max + a_min = -8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_l3132_313263


namespace NUMINAMATH_CALUDE_mural_hourly_rate_l3132_313242

/-- Calculates the hourly rate for painting a mural given its dimensions, painting rate, and total charge -/
theorem mural_hourly_rate (length width : ℝ) (paint_rate : ℝ) (total_charge : ℝ) :
  length = 20 ∧ width = 15 ∧ paint_rate = 20 ∧ total_charge = 15000 →
  total_charge / (length * width * paint_rate / 60) = 150 := by
  sorry

#check mural_hourly_rate

end NUMINAMATH_CALUDE_mural_hourly_rate_l3132_313242


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l3132_313254

theorem avery_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : 
  num_chickens = 20 →
  eggs_per_chicken = 6 →
  eggs_per_carton = 12 →
  (num_chickens * eggs_per_chicken) / eggs_per_carton = 10 := by
sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l3132_313254


namespace NUMINAMATH_CALUDE_triangular_sum_congruence_l3132_313260

theorem triangular_sum_congruence (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_sum_congruence_l3132_313260


namespace NUMINAMATH_CALUDE_sky_diving_company_total_amount_l3132_313287

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem sky_diving_company_total_amount :
  individual_bookings + group_bookings - refunds = 26400 := by
  sorry

end NUMINAMATH_CALUDE_sky_diving_company_total_amount_l3132_313287


namespace NUMINAMATH_CALUDE_square_difference_divided_l3132_313204

theorem square_difference_divided : (180^2 - 150^2) / 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l3132_313204


namespace NUMINAMATH_CALUDE_greatest_n_perfect_square_l3132_313218

/-- Sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Product of sum of squares -/
def product_sum_squares (n : ℕ) : ℕ :=
  (sum_squares n) * (sum_squares (2 * n) - sum_squares n)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Main theorem -/
theorem greatest_n_perfect_square :
  (∀ k : ℕ, k ≤ 2023 → is_perfect_square (product_sum_squares k) → k ≤ 1921) ∧
  is_perfect_square (product_sum_squares 1921) := by sorry

end NUMINAMATH_CALUDE_greatest_n_perfect_square_l3132_313218


namespace NUMINAMATH_CALUDE_father_age_problem_l3132_313257

/-- The age problem -/
theorem father_age_problem (father_age son_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 10 →
  father_age = 33 := by
sorry

end NUMINAMATH_CALUDE_father_age_problem_l3132_313257


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l3132_313222

theorem certain_number_subtraction (x : ℝ) (y : ℝ) : 
  (3 * x = (y - x) + 4) → (x = 5) → (y = 16) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l3132_313222


namespace NUMINAMATH_CALUDE_debate_participants_l3132_313226

theorem debate_participants (third_school : ℕ) 
  (h1 : third_school + (third_school + 40) + 2 * (third_school + 40) = 920) : 
  third_school = 200 := by
sorry

end NUMINAMATH_CALUDE_debate_participants_l3132_313226


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3132_313258

theorem multiplication_addition_equality : 45 * 52 + 78 * 45 = 5850 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3132_313258


namespace NUMINAMATH_CALUDE_saplings_in_park_l3132_313262

theorem saplings_in_park (total_trees : ℕ) (ancient_oaks : ℕ) (fir_trees : ℕ) : 
  total_trees = 96 → ancient_oaks = 15 → fir_trees = 23 → 
  total_trees - (ancient_oaks + fir_trees) = 58 := by
  sorry

end NUMINAMATH_CALUDE_saplings_in_park_l3132_313262


namespace NUMINAMATH_CALUDE_intersection_M_N_l3132_313277

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def N : Set ℝ := {x | 3 * x + 2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3132_313277


namespace NUMINAMATH_CALUDE_candy_to_drink_ratio_l3132_313213

def deal_price : ℚ := 20
def ticket_price : ℚ := 8
def popcorn_price : ℚ := ticket_price - 3
def drink_price : ℚ := popcorn_price + 1
def savings : ℚ := 2

def normal_total : ℚ := deal_price + savings
def candy_price : ℚ := normal_total - (ticket_price + popcorn_price + drink_price)

theorem candy_to_drink_ratio : candy_price / drink_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_to_drink_ratio_l3132_313213


namespace NUMINAMATH_CALUDE_next_two_terms_of_sequence_l3132_313266

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ - d * (n - 1)

theorem next_two_terms_of_sequence :
  let a₁ := 19.8
  let d := 1.2
  (arithmetic_sequence a₁ d 4 = 16.2) ∧ (arithmetic_sequence a₁ d 5 = 15) := by
  sorry

end NUMINAMATH_CALUDE_next_two_terms_of_sequence_l3132_313266


namespace NUMINAMATH_CALUDE_even_function_inequality_l3132_313267

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_even : is_even_function f)
  (h_increasing : increasing_on_negative f)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : 0 < x₂)
  (h_sum_pos : 0 < x₁ + x₂) :
  f (-x₁) > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3132_313267


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l3132_313221

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 35 →
  editors > 38 →
  writers + editors + x = total →
  x ≤ 26 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l3132_313221


namespace NUMINAMATH_CALUDE_peters_nickels_problem_l3132_313281

theorem peters_nickels_problem :
  ∃! n : ℕ, 40 < n ∧ n < 400 ∧ 
    n % 4 = 2 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n = 142 := by
  sorry

end NUMINAMATH_CALUDE_peters_nickels_problem_l3132_313281


namespace NUMINAMATH_CALUDE_square_area_increase_l3132_313216

theorem square_area_increase (x : ℝ) : (x + 3)^2 - x^2 = 45 → x^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3132_313216


namespace NUMINAMATH_CALUDE_soda_consumption_proof_l3132_313231

/-- The number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := 360

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The number of soda bottles Debby drank per day -/
def soda_bottles_per_day : ℕ := total_soda_bottles / days_lasted

theorem soda_consumption_proof : soda_bottles_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_soda_consumption_proof_l3132_313231


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3132_313296

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i^3 / (1 + i) = -1/2 - 1/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3132_313296


namespace NUMINAMATH_CALUDE_factor_expression_l3132_313200

theorem factor_expression (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3132_313200


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_is_180_l3132_313211

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_first_two : a 1 + a 2 = 20
  sum_third_fourth : a 3 + a 4 = 60

/-- The sum of the fifth and sixth terms of the geometric sequence is 180 -/
theorem sum_fifth_sixth_is_180 (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_is_180_l3132_313211


namespace NUMINAMATH_CALUDE_geometric_sequence_divisibility_l3132_313259

theorem geometric_sequence_divisibility (a₁ a₂ : ℚ) (n : ℕ) : 
  a₁ = 5/8 → a₂ = 25 → 
  (∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, m < k → ¬(∃ q : ℚ, a₁ * (a₂ / a₁)^(m-1) = 2000000 * q)) ∧
              (∃ q : ℚ, a₁ * (a₂ / a₁)^(k-1) = 2000000 * q)) → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_divisibility_l3132_313259


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3132_313249

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 5 * c) : 
  a * b * c = 22500 / 343 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3132_313249


namespace NUMINAMATH_CALUDE_sqrt_cube_root_equality_l3132_313286

theorem sqrt_cube_root_equality (a : ℝ) (h : a > 0) : 
  Real.sqrt (a * Real.rpow a (1/3)) = Real.rpow a (2/3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_equality_l3132_313286


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3132_313274

def A : Set ℤ := {-1, 3}
def B : Set ℤ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3132_313274


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3132_313223

/-- Calculate the profit percentage given the cost price and selling price -/
theorem profit_percentage_calculation (cost_price selling_price : ℚ) :
  cost_price = 60 →
  selling_price = 78 →
  (selling_price - cost_price) / cost_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3132_313223


namespace NUMINAMATH_CALUDE_time_to_work_l3132_313225

-- Define the variables
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120
def total_time : ℝ := 3

-- Define the theorem
theorem time_to_work : 
  ∃ (distance : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    (distance / speed_to_work) * 60 = 108 := by
  sorry

end NUMINAMATH_CALUDE_time_to_work_l3132_313225


namespace NUMINAMATH_CALUDE_angle_expression_value_l3132_313290

theorem angle_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3132_313290


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l3132_313279

theorem average_of_combined_sets (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) 
  (h1 : n1 = 60) (h2 : n2 = 40) (h3 : avg1 = 40) (h4 : avg2 = 60) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l3132_313279


namespace NUMINAMATH_CALUDE_integer_expression_is_integer_l3132_313201

theorem integer_expression_is_integer (n : ℤ) : ∃ m : ℤ, (n / 3 + n^2 / 2 + n^3 / 6 : ℚ) = m := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_is_integer_l3132_313201


namespace NUMINAMATH_CALUDE_fifth_power_last_digit_l3132_313299

theorem fifth_power_last_digit (n : ℕ) : n % 10 = (n^5) % 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_last_digit_l3132_313299


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3132_313202

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, f x a ≥ m) → a = -6 ∨ a = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3132_313202


namespace NUMINAMATH_CALUDE_inequality_proof_l3132_313291

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(a+c)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3132_313291


namespace NUMINAMATH_CALUDE_unique_campers_difference_l3132_313252

def rowing_problem (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ) : Prop :=
  let total_afternoon := morning_and_afternoon + afternoon_and_evening + (afternoon - morning_and_afternoon - afternoon_and_evening)
  let total_evening := afternoon_and_evening + evening_only
  morning = 33 ∧
  morning_and_afternoon = 11 ∧
  afternoon = 34 ∧
  afternoon_and_evening = 20 ∧
  evening_only = 10 ∧
  total_afternoon - total_evening = 4

theorem unique_campers_difference :
  ∃ (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ),
    rowing_problem morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only :=
by
  sorry

end NUMINAMATH_CALUDE_unique_campers_difference_l3132_313252


namespace NUMINAMATH_CALUDE_intersection_sum_coordinates_l3132_313297

/-- The quartic equation -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x + 1

/-- The linear equation -/
def g (x y : ℝ) : ℝ := 2*x - 3*y - 6

/-- The intersection points of f and g -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2 = 0}

theorem intersection_sum_coordinates :
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    x₁ + x₂ + x₃ + x₄ = 3 ∧
    y₁ + y₂ + y₃ + y₄ = -6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_coordinates_l3132_313297


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3132_313234

/-- Parabola P₁ -/
def P₁ (x y : ℚ) : Prop := y = x^2 + 52/25

/-- Parabola P₂ -/
def P₂ (x y : ℚ) : Prop := x = y^2 + 81/16

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

/-- L has rational slope -/
def rational_slope (a b : ℕ) : Prop := ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (a : ℚ) / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∀ x y : ℚ, P₁ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₁ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  (∀ x y : ℚ, P₂ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₂ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  rational_slope a b →
  a > 0 → b > 0 → c > 0 →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 168 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l3132_313234


namespace NUMINAMATH_CALUDE_center_of_specific_pyramid_l3132_313269

/-- The center of the circumscribed sphere of a triangular pyramid -/
def center_of_circumscribed_sphere (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The center of the circumscribed sphere of a triangular pyramid with
    vertices at (1,0,1), (1,1,0), (0,1,1), and (0,0,0) has coordinates (1/2, 1/2, 1/2) -/
theorem center_of_specific_pyramid :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (1, 1, 0)
  let C : ℝ × ℝ × ℝ := (0, 1, 1)
  let D : ℝ × ℝ × ℝ := (0, 0, 0)
  center_of_circumscribed_sphere A B C D = (1/2, 1/2, 1/2) := by sorry

end NUMINAMATH_CALUDE_center_of_specific_pyramid_l3132_313269


namespace NUMINAMATH_CALUDE_competition_results_l3132_313233

/-- Represents the weights of lifts for an athlete -/
structure AthleteLifts where
  first : ℕ
  second : ℕ

/-- The competition results -/
def Competition : Type :=
  AthleteLifts × AthleteLifts × AthleteLifts

def joe_total (c : Competition) : ℕ := c.1.first + c.1.second
def mike_total (c : Competition) : ℕ := c.2.1.first + c.2.1.second
def lisa_total (c : Competition) : ℕ := c.2.2.first + c.2.2.second

def joe_condition (c : Competition) : Prop :=
  2 * c.1.first = c.1.second + 300

def mike_condition (c : Competition) : Prop :=
  c.2.1.second = c.2.1.first + 200

def lisa_condition (c : Competition) : Prop :=
  c.2.2.first = 3 * c.2.2.second

theorem competition_results (c : Competition) 
  (h1 : joe_total c = 900)
  (h2 : mike_total c = 1100)
  (h3 : lisa_total c = 1000)
  (h4 : joe_condition c)
  (h5 : mike_condition c)
  (h6 : lisa_condition c) :
  c.1.first = 400 ∧ c.2.1.first = 450 ∧ c.2.2.second = 250 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l3132_313233


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3132_313298

theorem smaller_number_in_ratio (a b : ℕ) : 
  a > 0 → b > 0 → a * 3 = b * 2 → lcm a b = 120 → a = 80 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3132_313298


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3132_313203

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 - 3 * y^2 + 6 * x - x^2 + 3 * y^2 = x^2 + 6 * x :=
by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) :
  4 * m^2 + 1 + 2 * m - 3 * (2 + m - m^2) = 7 * m^2 - m - 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3132_313203


namespace NUMINAMATH_CALUDE_total_people_is_123_l3132_313255

/-- Calculates the total number of people on a bus given the number of boys and additional information. -/
def total_people_on_bus (num_boys : ℕ) : ℕ :=
  let num_girls := num_boys + (2 * num_boys) / 5
  let num_students := num_boys + num_girls
  let num_adults := 3  -- driver, assistant, and teacher
  num_students + num_adults

/-- Theorem stating that given the conditions, the total number of people on the bus is 123. -/
theorem total_people_is_123 : total_people_on_bus 50 = 123 := by
  sorry

#eval total_people_on_bus 50

end NUMINAMATH_CALUDE_total_people_is_123_l3132_313255


namespace NUMINAMATH_CALUDE_meaningful_expression_l3132_313283

/-- The expression sqrt(a+1)/(a-2) is meaningful iff a ≥ -1 and a ≠ 2 -/
theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ∧ (a ≠ 2) ↔ a ≥ -1 ∧ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3132_313283


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3132_313289

/-- The coefficient of x³ in the expansion of (3x³ + 2x² + 4x + 5)(4x³ + 3x² + 5x + 6) is 32 -/
theorem x_cubed_coefficient (x : ℝ) : 
  (3*x^3 + 2*x^2 + 4*x + 5) * (4*x^3 + 3*x^2 + 5*x + 6) = 
  32*x^3 + (12*x^5 + 15*x^4 + 23*x^2 + 34*x + 30) := by
sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3132_313289


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l3132_313230

theorem chess_club_mixed_groups 
  (total_children : ℕ) 
  (total_groups : ℕ) 
  (group_size : ℕ) 
  (boy_boy_games : ℕ) 
  (girl_girl_games : ℕ) : 
  total_children = 90 →
  total_groups = 30 →
  group_size = 3 →
  boy_boy_games = 30 →
  girl_girl_games = 14 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 23 ∧ 
    mixed_groups + (boy_boy_games / 3) + (girl_girl_games / 3) = total_groups) :=
by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l3132_313230


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3132_313224

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that ¬q is a necessary but not sufficient condition for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3132_313224


namespace NUMINAMATH_CALUDE_quadratic_sum_l3132_313288

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c (-3) = 0) →
  (QuadraticFunction a b c 5 = 0) →
  (∀ x, QuadraticFunction a b c x ≥ -36) →
  (∃ x, QuadraticFunction a b c x = -36) →
  a + b + c = -36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3132_313288


namespace NUMINAMATH_CALUDE_dans_remaining_cards_l3132_313285

/-- Given Dan's initial number of baseball cards, the number of torn cards,
    and the number of cards sold to Sam, prove that Dan now has 82 baseball cards. -/
theorem dans_remaining_cards
  (initial_cards : ℕ)
  (torn_cards : ℕ)
  (sold_cards : ℕ)
  (h1 : initial_cards = 97)
  (h2 : torn_cards = 8)
  (h3 : sold_cards = 15) :
  initial_cards - sold_cards = 82 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_cards_l3132_313285


namespace NUMINAMATH_CALUDE_base_conversion_equivalence_l3132_313284

theorem base_conversion_equivalence :
  ∀ (C B : ℕ),
    C < 9 →
    B < 6 →
    9 * C + B = 6 * B + C →
    C = 0 ∧ B = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_equivalence_l3132_313284


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3132_313205

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3132_313205


namespace NUMINAMATH_CALUDE_tuesday_sales_fifty_l3132_313251

/-- Calculates the number of books sold on Tuesday given the initial stock,
    sales on other days, and the percentage of unsold books. -/
def books_sold_tuesday (initial_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ)
    (unsold_percentage : ℚ) : ℕ :=
  let unsold_books := (initial_stock : ℚ) * unsold_percentage / 100
  let other_days_sales := monday_sales + wednesday_sales + thursday_sales + friday_sales
  initial_stock - (other_days_sales + unsold_books.ceil.toNat)

/-- Theorem stating that the number of books sold on Tuesday is 50. -/
theorem tuesday_sales_fifty :
  books_sold_tuesday 1100 75 64 78 135 (63945/1000) = 50 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_sales_fifty_l3132_313251


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l3132_313256

theorem unique_solution_of_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔
  x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l3132_313256


namespace NUMINAMATH_CALUDE_car_travel_inequality_l3132_313280

/-- Represents the daily distance traveled by a car -/
def daily_distance : ℝ → ℝ
| x => x + 19

/-- Represents the total distance traveled in 8 days -/
def total_distance (x : ℝ) : ℝ := 8 * (daily_distance x)

/-- Theorem stating the inequality representing the car's travel -/
theorem car_travel_inequality (x : ℝ) :
  total_distance x > 2200 ↔ 8 * (x + 19) > 2200 := by sorry

end NUMINAMATH_CALUDE_car_travel_inequality_l3132_313280


namespace NUMINAMATH_CALUDE_rotation_result_l3132_313247

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a vector 180° about the y-axis -/
def rotateY180 (v : Vector3D) : Vector3D :=
  { x := -v.x, y := v.y, z := -v.z }

/-- The given vector -/
def givenVector : Vector3D :=
  { x := 2, y := -1, z := 1 }

/-- The expected result after rotation -/
def expectedResult : Vector3D :=
  { x := -2, y := -1, z := -1 }

theorem rotation_result :
  rotateY180 givenVector = expectedResult := by sorry

end NUMINAMATH_CALUDE_rotation_result_l3132_313247


namespace NUMINAMATH_CALUDE_don_max_bottles_l3132_313292

/-- The number of bottles Shop A sells to Don -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Shop B sells to Don -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Shop C sells to Don -/
def shop_c_bottles : ℕ := 220

/-- The maximum number of bottles Don can buy -/
def max_bottles : ℕ := shop_a_bottles + shop_b_bottles + shop_c_bottles

theorem don_max_bottles : max_bottles = 550 := by sorry

end NUMINAMATH_CALUDE_don_max_bottles_l3132_313292


namespace NUMINAMATH_CALUDE_speed_limit_inequality_l3132_313253

/-- Given a speed limit of 40 km/h, prove that it can be expressed as v ≤ 40, where v is the speed of a vehicle. -/
theorem speed_limit_inequality (v : ℝ) (speed_limit : ℝ) (h : speed_limit = 40) :
  v ≤ speed_limit ↔ v ≤ 40 := by sorry

end NUMINAMATH_CALUDE_speed_limit_inequality_l3132_313253


namespace NUMINAMATH_CALUDE_ball_count_l3132_313243

/-- Given a box of balls with specific properties, prove the total number of balls -/
theorem ball_count (orange purple yellow total : ℕ) : 
  orange + purple + yellow = total →  -- Total balls
  orange = 2 * n →  -- Ratio condition for orange
  purple = 3 * n →  -- Ratio condition for purple
  yellow = 4 * n →  -- Ratio condition for yellow
  yellow = 32 →     -- Given number of yellow balls
  total = 72 :=     -- Prove total number of balls
by sorry

end NUMINAMATH_CALUDE_ball_count_l3132_313243


namespace NUMINAMATH_CALUDE_max_ab_value_l3132_313229

/-- The maximum value of ab given the conditions -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a * 2 - b * (-1) = 2) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → (x - 2)^2 + (y + 1)^2 = 4) :
  a * b ≤ 1/2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * 2 - b * (-1) = 2 ∧ a * b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l3132_313229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2018_l3132_313237

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = -2018
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

/-- The main theorem -/
theorem arithmetic_sequence_sum_2018 (seq : ArithmeticSequence) :
  (sum_n seq 2015 / 2015) - (sum_n seq 2013 / 2013) = 2 →
  sum_n seq 2018 = -2018 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2018_l3132_313237


namespace NUMINAMATH_CALUDE_faster_train_speed_l3132_313261

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (overtake_time : ℝ)
  (h1 : train_length = 50)
  (h2 : slower_speed = 36)
  (h3 : overtake_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l3132_313261


namespace NUMINAMATH_CALUDE_campground_distance_l3132_313268

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l3132_313268


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3132_313278

/-- The total cost of buying pens and exercise books -/
def total_cost (m n : ℝ) : ℝ := 2 * m + 3 * n

/-- Theorem: The total cost of 2 pens at m yuan each and 3 exercise books at n yuan each is 2m + 3n yuan -/
theorem total_cost_calculation (m n : ℝ) :
  total_cost m n = 2 * m + 3 * n :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3132_313278


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l3132_313295

theorem power_fraction_evaluation (a b : ℕ) : 
  (2^a : ℕ) ∣ 360 ∧ 
  (3^b : ℕ) ∣ 360 ∧ 
  ∀ k > a, ¬((2^k : ℕ) ∣ 360) ∧ 
  ∀ l > b, ¬((3^l : ℕ) ∣ 360) →
  ((1/4 : ℚ) ^ (b - a) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l3132_313295


namespace NUMINAMATH_CALUDE_trapezoid_existence_l3132_313209

-- Define the trapezoid structure
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ

-- Define the existence theorem
theorem trapezoid_existence (a b c α β : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) : 
  ∃ t : Trapezoid, 
    t.a = a ∧ t.b = b ∧ t.c = c ∧ t.α = α ∧ t.β = β :=
sorry


end NUMINAMATH_CALUDE_trapezoid_existence_l3132_313209
