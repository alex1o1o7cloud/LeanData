import Mathlib

namespace NUMINAMATH_CALUDE_vector_operation_l3926_392691

/-- Given vectors a and b in R², prove that 2a - b equals (-1, 0) --/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 4)) :
  2 • a - b = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3926_392691


namespace NUMINAMATH_CALUDE_investment_final_value_l3926_392608

def investment_value (initial : ℝ) (w1 w2 w3 w4 w5 w6 : ℝ) : ℝ :=
  initial * (1 + w1) * (1 + w2) * (1 - w3) * (1 + w4) * (1 + w5) * (1 - w6)

theorem investment_final_value :
  let initial : ℝ := 400
  let week1_gain : ℝ := 0.25
  let week2_gain : ℝ := 0.50
  let week3_loss : ℝ := 0.10
  let week4_gain : ℝ := 0.20
  let week5_gain : ℝ := 0.05
  let week6_loss : ℝ := 0.15
  investment_value initial week1_gain week2_gain week3_loss week4_gain week5_gain week6_loss = 722.925 := by
  sorry

end NUMINAMATH_CALUDE_investment_final_value_l3926_392608


namespace NUMINAMATH_CALUDE_midway_point_distance_l3926_392629

/-- The distance from Yooseon's house to the midway point of her path to school -/
def midway_distance (house_to_hospital : ℕ) (hospital_to_school : ℕ) : ℕ :=
  (house_to_hospital + hospital_to_school) / 2

theorem midway_point_distance :
  let house_to_hospital := 1700
  let hospital_to_school := 900
  midway_distance house_to_hospital hospital_to_school = 1300 := by
  sorry

end NUMINAMATH_CALUDE_midway_point_distance_l3926_392629


namespace NUMINAMATH_CALUDE_circular_permutations_2a2b2c_l3926_392649

/-- The number of first-type circular permutations for a multiset with given element counts -/
def circularPermutations (counts : List Nat) : Nat :=
  sorry

/-- Theorem: The number of first-type circular permutations for 2 a's, 2 b's, and 2 c's is 16 -/
theorem circular_permutations_2a2b2c :
  circularPermutations [2, 2, 2] = 16 := by
  sorry

end NUMINAMATH_CALUDE_circular_permutations_2a2b2c_l3926_392649


namespace NUMINAMATH_CALUDE_triangle_side_range_l3926_392670

theorem triangle_side_range :
  ∀ x : ℝ, 
    (∃ t : Set (ℝ × ℝ × ℝ), 
      t.Nonempty ∧ 
      (∀ s ∈ t, s.1 = 3 ∧ s.2.1 = 6 ∧ s.2.2 = x) ∧
      (∀ s ∈ t, s.1 + s.2.1 > s.2.2 ∧ s.1 + s.2.2 > s.2.1 ∧ s.2.1 + s.2.2 > s.1)) →
    3 < x ∧ x < 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3926_392670


namespace NUMINAMATH_CALUDE_three_points_distance_is_four_l3926_392686

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- A point (x, y) is on the graph of f if y = f(x) -/
def on_graph (x y : ℝ) : Prop := y = f x

/-- The distance of a point (x, y) from the x-axis is the absolute value of y -/
def distance_from_x_axis (y : ℝ) : ℝ := |y|

/-- There exist exactly three points on the graph of f with distance m from the x-axis -/
def three_points_with_distance (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    on_graph x₁ y₁ ∧ on_graph x₂ y₂ ∧ on_graph x₃ y₃ ∧
    distance_from_x_axis y₁ = m ∧
    distance_from_x_axis y₂ = m ∧
    distance_from_x_axis y₃ = m ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    ∀ x y : ℝ, on_graph x y → distance_from_x_axis y = m → (x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem three_points_distance_is_four :
  ∀ m : ℝ, three_points_with_distance m → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_three_points_distance_is_four_l3926_392686


namespace NUMINAMATH_CALUDE_S_and_S_l3926_392665

-- Define the systems S and S'
def S (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ x * (x^4 - y^2 + x^2) = 1

def S' (x y : ℝ) : Prop :=
  y * (x^4 - y^2 + x^2) = x ∧ y = x^2

-- Theorem stating that S and S' do not have the same set of solutions
theorem S_and_S'_different_solutions :
  ¬(∀ x y : ℝ, S x y ↔ S' x y) :=
sorry

end NUMINAMATH_CALUDE_S_and_S_l3926_392665


namespace NUMINAMATH_CALUDE_friendly_snakes_not_green_l3926_392648

structure Snake where
  friendly : Bool
  green : Bool
  can_multiply : Bool
  can_divide : Bool

def Tom_snakes : Finset Snake := sorry

theorem friendly_snakes_not_green :
  ∀ s ∈ Tom_snakes,
  (s.friendly → s.can_multiply) ∧
  (s.green → ¬s.can_divide) ∧
  (¬s.can_divide → ¬s.can_multiply) →
  (s.friendly → ¬s.green) :=
by sorry

end NUMINAMATH_CALUDE_friendly_snakes_not_green_l3926_392648


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3926_392634

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p → p ∣ (2^14 + 7^9) → p > 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3926_392634


namespace NUMINAMATH_CALUDE_min_tiles_cover_floor_l3926_392617

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the tile dimensions -/
def tile : Rectangle := { length := 3, width := 4 }

/-- Represents the floor dimensions -/
def floor : Rectangle := { length := 36, width := 60 }

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded (t : Rectangle) (f : Rectangle) : ℕ :=
  (area f) / (area t)

theorem min_tiles_cover_floor :
  tilesNeeded tile floor = 180 := by sorry

end NUMINAMATH_CALUDE_min_tiles_cover_floor_l3926_392617


namespace NUMINAMATH_CALUDE_part_one_part_two_l3926_392622

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.A = -4/5

-- Part I
theorem part_one (t : Triangle) (h : triangle_conditions t) (ha : t.a = 4) :
  Real.sin t.B = 3/10 := by sorry

-- Part II
theorem part_two (t : Triangle) (h : triangle_conditions t) (hs : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 2 * Real.sqrt 34 ∧ t.c = 10 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3926_392622


namespace NUMINAMATH_CALUDE_theta_value_l3926_392635

theorem theta_value : ∃! θ : ℕ, θ ∈ Finset.range 10 ∧ θ ≠ 0 ∧ 294 / θ = 30 + 4 * θ := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l3926_392635


namespace NUMINAMATH_CALUDE_puppies_brought_in_l3926_392601

-- Define the given conditions
def initial_puppies : ℕ := 5
def adopted_per_day : ℕ := 8
def days_to_adopt_all : ℕ := 5

-- Define the theorem
theorem puppies_brought_in :
  ∃ (brought_in : ℕ), 
    initial_puppies + brought_in = adopted_per_day * days_to_adopt_all :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_puppies_brought_in_l3926_392601


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l3926_392624

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the interval [2, 3]
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem max_min_values_on_interval :
  (∀ x ∈ interval, f x ≤ f 2) ∧
  (∀ x ∈ interval, f 3 ≤ f x) ∧
  (f 2 = -2) ∧
  (f 3 = -18) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l3926_392624


namespace NUMINAMATH_CALUDE_sleepy_squirrel_stockpile_l3926_392628

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by each busy squirrel -/
def nuts_per_busy_squirrel : ℕ := 30

/-- The number of days the squirrels have been stockpiling -/
def days_stockpiling : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts : ℕ := 20

theorem sleepy_squirrel_stockpile :
  busy_squirrels * nuts_per_busy_squirrel * days_stockpiling + 
  sleepy_squirrel_nuts * days_stockpiling = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_sleepy_squirrel_stockpile_l3926_392628


namespace NUMINAMATH_CALUDE_reader_group_total_l3926_392687

/-- Represents the number of readers in a group reading different types of books. -/
structure ReaderGroup where
  sci_fi : ℕ     -- Number of readers who read science fiction
  literary : ℕ   -- Number of readers who read literary works
  both : ℕ       -- Number of readers who read both

/-- Calculates the total number of readers in the group. -/
def total_readers (g : ReaderGroup) : ℕ :=
  g.sci_fi + g.literary - g.both

/-- Theorem stating that for the given reader numbers, the total is 650. -/
theorem reader_group_total :
  ∃ (g : ReaderGroup), g.sci_fi = 250 ∧ g.literary = 550 ∧ g.both = 150 ∧ total_readers g = 650 :=
by
  sorry

#check reader_group_total

end NUMINAMATH_CALUDE_reader_group_total_l3926_392687


namespace NUMINAMATH_CALUDE_a_2010_at_1_l3926_392681

def a : ℕ → (ℝ → ℝ)
  | 0 => λ x => 1
  | 1 => λ x => x^2 + x + 1
  | (n+2) => λ x => (x^(n+2) + 1) * a (n+1) x - a n x

theorem a_2010_at_1 : a 2010 1 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_a_2010_at_1_l3926_392681


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3926_392654

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ),
  initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.4)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.46 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3926_392654


namespace NUMINAMATH_CALUDE_g_sum_equal_164_l3926_392663

def g (x : ℝ) : ℝ := 2 * x^6 - 5 * x^4 + 7 * x^2 + 6

theorem g_sum_equal_164 (h : g 15 = 82) : g 15 + g (-15) = 164 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_equal_164_l3926_392663


namespace NUMINAMATH_CALUDE_bag_of_balls_l3926_392631

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 20)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 37)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l3926_392631


namespace NUMINAMATH_CALUDE_greatest_mean_Y_Z_l3926_392689

-- Define the piles of rocks
variable (X Y Z : Set ℝ)

-- Define the mean weight functions
variable (mean : Set ℝ → ℝ)

-- Define the conditions
variable (h1 : mean X = 30)
variable (h2 : mean Y = 70)
variable (h3 : mean (X ∪ Y) = 50)
variable (h4 : mean (X ∪ Z) = 40)

-- Define the function to calculate the mean of Y and Z
def mean_Y_Z : ℝ := mean (Y ∪ Z)

-- Theorem statement
theorem greatest_mean_Y_Z : 
  ∀ n : ℕ, mean_Y_Z ≤ 70 ∧ (mean_Y_Z > 69 → mean_Y_Z = 70) :=
sorry

end NUMINAMATH_CALUDE_greatest_mean_Y_Z_l3926_392689


namespace NUMINAMATH_CALUDE_younger_brother_height_l3926_392640

theorem younger_brother_height (h1 h2 : ℝ) (h1_positive : 0 < h1) (h2_positive : 0 < h2) 
  (height_difference : h2 - h1 = 12) (height_sum : h1 + h2 = 308) (h1_smaller : h1 < h2) : h1 = 148 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_height_l3926_392640


namespace NUMINAMATH_CALUDE_bankers_discount_l3926_392676

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℚ) (time : ℕ) (rate : ℚ) : 
  bankers_gain = 360 ∧ time = 3 ∧ rate = 12/100 → 
  ∃ (bankers_discount : ℚ), bankers_discount = 5625/10 :=
by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l3926_392676


namespace NUMINAMATH_CALUDE_sarah_cupcake_count_l3926_392688

def is_valid_cupcake_count (c : ℕ) : Prop :=
  ∃ (k : ℕ), 
    c + k = 6 ∧ 
    (90 * c + 40 * k) % 100 = 0

theorem sarah_cupcake_count :
  ∀ c : ℕ, is_valid_cupcake_count c → c = 4 ∨ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cupcake_count_l3926_392688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3926_392657

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 4 150

theorem arithmetic_sequence_150th_term :
  term_150 = 599 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3926_392657


namespace NUMINAMATH_CALUDE_range_of_a_l3926_392625

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, x - a ≤ 0) → a ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3926_392625


namespace NUMINAMATH_CALUDE_billboard_perimeter_l3926_392673

theorem billboard_perimeter (area : ℝ) (short_side : ℝ) (perimeter : ℝ) : 
  area = 104 → 
  short_side = 8 → 
  perimeter = 2 * (area / short_side + short_side) →
  perimeter = 42 := by
sorry


end NUMINAMATH_CALUDE_billboard_perimeter_l3926_392673


namespace NUMINAMATH_CALUDE_laptop_price_increase_l3926_392659

theorem laptop_price_increase (P₀ : ℝ) : 
  let P₂ := P₀ * (1 + 0.06)^2
  P₂ > 56358 :=
by
  sorry

#check laptop_price_increase

end NUMINAMATH_CALUDE_laptop_price_increase_l3926_392659


namespace NUMINAMATH_CALUDE_smallest_common_factor_40_90_l3926_392602

theorem smallest_common_factor_40_90 : 
  ∃ (a : ℕ), a > 0 ∧ Nat.gcd a 40 > 1 ∧ Nat.gcd a 90 > 1 ∧ 
  ∀ (b : ℕ), b > 0 → Nat.gcd b 40 > 1 → Nat.gcd b 90 > 1 → a ≤ b :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_40_90_l3926_392602


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3926_392623

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 3 ∧
    f x_min = -17 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3926_392623


namespace NUMINAMATH_CALUDE_julie_total_earnings_l3926_392684

/-- Calculates Julie's total earnings for September and October based on her landscaping business rates and hours worked. -/
theorem julie_total_earnings (
  -- September hours
  small_lawn_sept : ℕ) (large_lawn_sept : ℕ) (simple_garden_sept : ℕ) (complex_garden_sept : ℕ)
  (small_tree_sept : ℕ) (large_tree_sept : ℕ) (mulch_sept : ℕ)
  -- Rates
  (small_lawn_rate : ℕ) (large_lawn_rate : ℕ) (simple_garden_rate : ℕ) (complex_garden_rate : ℕ)
  (small_tree_rate : ℕ) (large_tree_rate : ℕ) (mulch_rate : ℕ)
  -- Given conditions
  (h1 : small_lawn_sept = 10) (h2 : large_lawn_sept = 15) (h3 : simple_garden_sept = 2)
  (h4 : complex_garden_sept = 1) (h5 : small_tree_sept = 5) (h6 : large_tree_sept = 5)
  (h7 : mulch_sept = 5)
  (h8 : small_lawn_rate = 4) (h9 : large_lawn_rate = 6) (h10 : simple_garden_rate = 8)
  (h11 : complex_garden_rate = 10) (h12 : small_tree_rate = 10) (h13 : large_tree_rate = 15)
  (h14 : mulch_rate = 12) :
  -- Theorem statement
  (small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
   simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
   small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
   mulch_rate * mulch_sept) +
  ((small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
    simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
    small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
    mulch_rate * mulch_sept) * 3 / 2) = 8525/10 := by
  sorry

end NUMINAMATH_CALUDE_julie_total_earnings_l3926_392684


namespace NUMINAMATH_CALUDE_two_digit_sum_reverse_cube_l3926_392600

/-- A function that reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- The main theorem -/
theorem two_digit_sum_reverse_cube :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ isPerfectCube (n + reverseDigits n) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_reverse_cube_l3926_392600


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3926_392652

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x+y)*(x-y) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3926_392652


namespace NUMINAMATH_CALUDE_h_of_3_eq_3_l3926_392639

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  if x = 1 then 0  -- Handle the case when x = 1 separately
  else ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^4 + 1) * (x^5 + 1) * (x^6 + 1) * (x^7 + 1) * (x^8 + 1) * (x^9 + 1) - 1) / (x^26 - 1)

-- Theorem statement
theorem h_of_3_eq_3 : h 3 = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_3_eq_3_l3926_392639


namespace NUMINAMATH_CALUDE_plane_at_distance_from_point_and_through_axis_l3926_392683

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Sphere where
  center : Point
  radius : ℝ

def ProjectionAxis : Set Point := sorry

-- Define the distance between a point and a plane
def distancePointPlane (p : Point) (plane : Plane) : ℝ := sorry

-- Define a predicate for a plane passing through the projection axis
def passesThroughProjectionAxis (plane : Plane) : Prop := sorry

-- Define a predicate for a plane being tangent to a sphere
def isTangentTo (plane : Plane) (sphere : Sphere) : Prop := sorry

-- The main theorem
theorem plane_at_distance_from_point_and_through_axis
  (A : Point) (d : ℝ) (P : Plane) :
  (distancePointPlane A P = d ∧ passesThroughProjectionAxis P) ↔
  (isTangentTo P (Sphere.mk A d) ∧ passesThroughProjectionAxis P) := by
  sorry

end NUMINAMATH_CALUDE_plane_at_distance_from_point_and_through_axis_l3926_392683


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3926_392651

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 17 → 
  b = 2 * a → 
  (a + b) / 2 = 33.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3926_392651


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l3926_392692

theorem inscribed_cylinder_height (r_cylinder : ℝ) (r_sphere : ℝ) :
  r_cylinder = 3 →
  r_sphere = 7 →
  let h := 2 * (2 * Real.sqrt 10)
  h = 2 * Real.sqrt (r_sphere^2 - r_cylinder^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l3926_392692


namespace NUMINAMATH_CALUDE_original_number_proof_l3926_392614

theorem original_number_proof (x : ℚ) :
  1 + (1 / x) = 5 / 2 → x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3926_392614


namespace NUMINAMATH_CALUDE_bananas_per_friend_l3926_392607

def virginia_bananas : ℕ := 40
def virginia_marbles : ℕ := 4
def number_of_friends : ℕ := 40

theorem bananas_per_friend :
  virginia_bananas / number_of_friends = 1 :=
sorry

end NUMINAMATH_CALUDE_bananas_per_friend_l3926_392607


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l3926_392630

/-- Represents a modified cube with smaller cubes removed from corners and sliced by a plane -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ
  slicePlane : Bool

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube with side length 5, smaller cubes of side length 2 removed from corners,
    and sliced by a plane, has 40 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := {
    originalSideLength := 5,
    removedCubeSideLength := 2,
    slicePlane := true
  }
  edgeCount cube = 40 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l3926_392630


namespace NUMINAMATH_CALUDE_function_properties_l3926_392656

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (5 - x) = f (5 + x))
    (h3 : ¬ (∀ x y, f x = f y)) :
    IsEven f ∧ IsPeriodic f 10 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3926_392656


namespace NUMINAMATH_CALUDE_linear_function_m_value_l3926_392653

theorem linear_function_m_value :
  ∃! m : ℝ, m ≠ 0 ∧ (∀ x y : ℝ, y = m * x^(|m + 1|) - 2 → ∃ a b : ℝ, y = a * x + b) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l3926_392653


namespace NUMINAMATH_CALUDE_rational_sum_of_three_cubes_l3926_392606

theorem rational_sum_of_three_cubes (t : ℚ) : 
  ∃ (x y z : ℚ), t = x^3 + y^3 + z^3 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_of_three_cubes_l3926_392606


namespace NUMINAMATH_CALUDE_john_foundation_homes_l3926_392650

/-- Represents the dimensions of a concrete slab for a home foundation -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the number of homes given foundation parameters -/
def calculateHomes (slab : SlabDimensions) (concreteDensity : Float) (concreteCostPerPound : Float) (totalFoundationCost : Float) : Float :=
  let slabVolume := slab.length * slab.width * slab.height
  let concreteWeight := slabVolume * concreteDensity
  let costPerHome := concreteWeight * concreteCostPerPound
  totalFoundationCost / costPerHome

/-- Proves that John is laying the foundation for 3 homes -/
theorem john_foundation_homes :
  let slab : SlabDimensions := { length := 100, width := 100, height := 0.5 }
  let concreteDensity : Float := 150
  let concreteCostPerPound : Float := 0.02
  let totalFoundationCost : Float := 45000
  calculateHomes slab concreteDensity concreteCostPerPound totalFoundationCost = 3 := by
  sorry


end NUMINAMATH_CALUDE_john_foundation_homes_l3926_392650


namespace NUMINAMATH_CALUDE_age_of_other_man_l3926_392675

/-- Given a group of men where two are replaced by women, prove the age of a specific man. -/
theorem age_of_other_man
  (n : ℕ)  -- Total number of people
  (m : ℕ)  -- Number of men initially
  (w : ℕ)  -- Number of women replacing men
  (age_increase : ℝ)  -- Increase in average age
  (known_man_age : ℝ)  -- Age of the known man
  (women_avg_age : ℝ)  -- Average age of the women
  (h1 : n = 8)  -- Total number of people is 8
  (h2 : m = 8)  -- Initial number of men is 8
  (h3 : w = 2)  -- Number of women replacing men is 2
  (h4 : age_increase = 2)  -- Average age increases by 2 years
  (h5 : known_man_age = 24)  -- One man is 24 years old
  (h6 : women_avg_age = 30)  -- Average age of women is 30 years
  : ∃ (other_man_age : ℝ), other_man_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l3926_392675


namespace NUMINAMATH_CALUDE_at_most_one_super_plus_good_l3926_392667

/-- Represents an 8x8 chessboard with numbers 1 to 64 --/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- A number is super-plus-good if it's the largest in its row and smallest in its column --/
def is_super_plus_good (board : Chessboard) (row col : Fin 8) : Prop :=
  (∀ c : Fin 8, board row c ≤ board row col) ∧
  (∀ r : Fin 8, board row col ≤ board r col)

/-- The arrangement is valid if each number appears exactly once --/
def is_valid_arrangement (board : Chessboard) : Prop :=
  ∀ n : Fin 64, ∃! (row col : Fin 8), board row col = n

theorem at_most_one_super_plus_good (board : Chessboard) 
  (h : is_valid_arrangement board) :
  ∃! (row col : Fin 8), is_super_plus_good board row col :=
sorry

end NUMINAMATH_CALUDE_at_most_one_super_plus_good_l3926_392667


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_l3926_392646

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

theorem circle_equation_with_diameter (x y : ℝ) :
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16 →
  (x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ↔
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_l3926_392646


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3926_392662

-- Define the parabola
def parabola (a x : ℝ) : ℝ := x^3 - 3*a*x + a^3

-- Define the line
def line (a x : ℝ) : ℝ := x + 2*a

-- Define the derivative of the parabola with respect to x
def parabola_derivative (a x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem line_through_parabola_vertex :
  ∃! (a : ℝ), ∃ (x : ℝ),
    (parabola_derivative a x = 0) ∧
    (line a x = parabola a x) :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3926_392662


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3926_392658

theorem reciprocal_of_negative_2023 :
  (1 : ℚ) / (-2023 : ℚ) = -(1 / 2023) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3926_392658


namespace NUMINAMATH_CALUDE_remainder_squared_pred_l3926_392693

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_pred_l3926_392693


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_neg_one_l3926_392677

theorem sum_of_roots_eq_neg_one (m n : ℝ) : 
  m ≠ 0 → 
  n ≠ 0 → 
  (∀ x : ℝ, x ≠ 0 → 1 / x^2 + m / x + n = 0) → 
  m * n = 1 → 
  m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_neg_one_l3926_392677


namespace NUMINAMATH_CALUDE_consignment_total_items_l3926_392644

/-- Represents the price and quantity of items in a consignment shop. -/
structure ConsignmentItems where
  camera_price : ℕ
  clock_price : ℕ
  pen_price : ℕ
  receiver_price : ℕ
  camera_quantity : ℕ

/-- Conditions for the consignment shop problem -/
def ConsignmentConditions (items : ConsignmentItems) : Prop :=
  -- Total value of all items is 240 rubles
  (3 * items.camera_quantity * items.pen_price + 
   items.camera_quantity * items.clock_price + 
   items.camera_quantity * items.receiver_price + 
   items.camera_quantity * items.camera_price = 240) ∧
  -- Sum of receiver and clock prices is 4 rubles more than sum of camera and pen prices
  (items.receiver_price + items.clock_price = items.camera_price + items.pen_price + 4) ∧
  -- Sum of clock and pen prices is 24 rubles less than sum of camera and receiver prices
  (items.clock_price + items.pen_price + 24 = items.camera_price + items.receiver_price) ∧
  -- Pen price is an integer not exceeding 6 rubles
  (items.pen_price ≤ 6) ∧
  -- Number of cameras equals camera price divided by 10
  (items.camera_quantity = items.camera_price / 10) ∧
  -- Number of clocks equals number of receivers and number of cameras
  (items.camera_quantity = items.camera_quantity) ∧
  -- Number of pens is three times the number of cameras
  (3 * items.camera_quantity = 3 * items.camera_quantity)

/-- The theorem stating that under the given conditions, the total number of items is 18 -/
theorem consignment_total_items (items : ConsignmentItems) 
  (h : ConsignmentConditions items) : 
  (6 * items.camera_quantity = 18) := by
  sorry


end NUMINAMATH_CALUDE_consignment_total_items_l3926_392644


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3926_392679

/-- Given a geometric sequence {a_n} with a₁ = 1 and a₄ = 8, prove that a₇ = 64 -/
theorem geometric_sequence_seventh_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                                -- First term condition
  a 4 = 8 →                                -- Fourth term condition
  a 7 = 64 :=                              -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3926_392679


namespace NUMINAMATH_CALUDE_preimage_of_two_zero_l3926_392638

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (1, 1) is the preimage of (2, 0) under f -/
theorem preimage_of_two_zero :
  f (1, 1) = (2, 0) ∧ ∀ p : ℝ × ℝ, f p = (2, 0) → p = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_zero_l3926_392638


namespace NUMINAMATH_CALUDE_camel_cannot_end_adjacent_l3926_392610

/-- Represents a hexagonal board with side length m -/
structure HexBoard where
  m : ℕ

/-- The total number of fields on a hexagonal board -/
def HexBoard.total_fields (board : HexBoard) : ℕ :=
  3 * board.m^2 - 3 * board.m + 1

/-- The number of moves a camel makes on the board -/
def HexBoard.camel_moves (board : HexBoard) : ℕ :=
  board.total_fields - 1

/-- Theorem stating that a camel cannot end on an adjacent field to its starting position -/
theorem camel_cannot_end_adjacent (board : HexBoard) :
  ∃ (start finish : ℕ), start ≠ finish ∧ 
  finish ≠ (start + 1) ∧ finish ≠ (start - 1) ∧
  finish = (start + board.camel_moves) % board.total_fields :=
sorry

end NUMINAMATH_CALUDE_camel_cannot_end_adjacent_l3926_392610


namespace NUMINAMATH_CALUDE_more_boys_probability_l3926_392672

-- Define the possible number of children
inductive ChildCount : Type
  | zero : ChildCount
  | one : ChildCount
  | two : ChildCount
  | three : ChildCount

-- Define the probability distribution for the number of children
def childCountProb : ChildCount → ℚ
  | ChildCount.zero => 1/15
  | ChildCount.one => 6/15
  | ChildCount.two => 6/15
  | ChildCount.three => 2/15

-- Define the probability of a child being a boy
def boyProb : ℚ := 1/2

-- Define the event of having more boys than girls
def moreBoysEvent : ChildCount → ℚ
  | ChildCount.zero => 0
  | ChildCount.one => 1/2
  | ChildCount.two => 1/4
  | ChildCount.three => 1/2

-- State the theorem
theorem more_boys_probability :
  (moreBoysEvent ChildCount.zero * childCountProb ChildCount.zero +
   moreBoysEvent ChildCount.one * childCountProb ChildCount.one +
   moreBoysEvent ChildCount.two * childCountProb ChildCount.two +
   moreBoysEvent ChildCount.three * childCountProb ChildCount.three) = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_probability_l3926_392672


namespace NUMINAMATH_CALUDE_fourth_selection_is_65_l3926_392643

/-- Systematic sampling function -/
def systematicSample (totalParts : ℕ) (sampleSize : ℕ) (firstSelection : ℕ) (selectionNumber : ℕ) : ℕ :=
  let samplingInterval := totalParts / sampleSize
  firstSelection + (selectionNumber - 1) * samplingInterval

/-- Theorem: In the given systematic sampling scenario, the fourth selection is part number 65 -/
theorem fourth_selection_is_65 :
  let totalParts := 200
  let sampleSize := 10
  let firstSelection := 5
  let fourthSelection := 4
  systematicSample totalParts sampleSize firstSelection fourthSelection = 65 := by
  sorry

#eval systematicSample 200 10 5 4  -- Should output 65

end NUMINAMATH_CALUDE_fourth_selection_is_65_l3926_392643


namespace NUMINAMATH_CALUDE_grid_bottom_right_value_l3926_392645

/-- Represents a 3x3 grid with some known values -/
structure Grid :=
  (a b c d e f g h i : ℕ)
  (b_eq_6 : b = 6)
  (c_eq_3 : c = 3)
  (h_eq_2 : h = 2)
  (all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0)

/-- The product of each row, column, and diagonal is the same -/
def grid_property (grid : Grid) : Prop :=
  let p := grid.a * grid.b * grid.c
  p = grid.d * grid.e * grid.f ∧
  p = grid.g * grid.h * grid.i ∧
  p = grid.a * grid.d * grid.g ∧
  p = grid.b * grid.e * grid.h ∧
  p = grid.c * grid.f * grid.i ∧
  p = grid.a * grid.e * grid.i ∧
  p = grid.c * grid.e * grid.g

theorem grid_bottom_right_value (grid : Grid) (h : grid_property grid) : grid.i = 36 := by
  sorry

end NUMINAMATH_CALUDE_grid_bottom_right_value_l3926_392645


namespace NUMINAMATH_CALUDE_circle_area_doubling_l3926_392603

theorem circle_area_doubling (r : ℝ) (h : r > 0) : 
  π * (2 * r)^2 = 4 * (π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_doubling_l3926_392603


namespace NUMINAMATH_CALUDE_inverse_composition_l3926_392609

/-- A function g with specific values --/
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 3

/-- The inverse of g --/
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 5
| 4 => 1
| 5 => 2

/-- g is bijective --/
axiom g_bijective : Function.Bijective g

/-- g_inv is indeed the inverse of g --/
axiom g_inv_is_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

/-- The main theorem --/
theorem inverse_composition : g_inv (g_inv (g_inv 5)) = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l3926_392609


namespace NUMINAMATH_CALUDE_two_quadratic_solving_algorithms_l3926_392674

/-- A quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An algorithm to solve a quadratic equation -/
structure QuadraticSolver where
  solve : QuadraticEquation → Set ℝ

/-- The specific quadratic equation x^2 - 5x + 6 = 0 -/
def specificEquation : QuadraticEquation :=
  { a := 1, b := -5, c := 6 }

theorem two_quadratic_solving_algorithms :
  ∃ (algo1 algo2 : QuadraticSolver), algo1 ≠ algo2 ∧
    algo1.solve specificEquation = algo2.solve specificEquation :=
sorry

end NUMINAMATH_CALUDE_two_quadratic_solving_algorithms_l3926_392674


namespace NUMINAMATH_CALUDE_coast_guard_overtakes_at_2_15pm_l3926_392647

/-- Represents the time of day in hours and minutes -/
structure TimeOfDay where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Represents the chase scenario -/
structure ChaseScenario where
  initialDistance : ℝ
  initialTime : TimeOfDay
  smugglerInitialSpeed : ℝ
  coastGuardSpeed : ℝ
  smugglerReducedSpeed : ℝ
  malfunctionTime : ℝ

/-- Calculates the time when the coast guard overtakes the smuggler -/
def overtakeTime (scenario : ChaseScenario) : TimeOfDay :=
  sorry

/-- The main theorem to prove -/
theorem coast_guard_overtakes_at_2_15pm
  (scenario : ChaseScenario)
  (h1 : scenario.initialDistance = 15)
  (h2 : scenario.initialTime = ⟨10, 0, sorry⟩)
  (h3 : scenario.smugglerInitialSpeed = 18)
  (h4 : scenario.coastGuardSpeed = 20)
  (h5 : scenario.smugglerReducedSpeed = 16)
  (h6 : scenario.malfunctionTime = 1) :
  overtakeTime scenario = ⟨14, 15, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_coast_guard_overtakes_at_2_15pm_l3926_392647


namespace NUMINAMATH_CALUDE_failed_students_l3926_392655

/-- The number of students who failed an examination, given the total number of students and the percentage who passed. -/
theorem failed_students (total : ℕ) (pass_percent : ℚ) : 
  total = 700 → pass_percent = 35 / 100 → 
  (total : ℚ) * (1 - pass_percent) = 455 := by
  sorry

end NUMINAMATH_CALUDE_failed_students_l3926_392655


namespace NUMINAMATH_CALUDE_equation_solution_l3926_392642

theorem equation_solution :
  ∃! x : ℚ, (x^2 + 2*x + 2) / (x + 2) = x + 3 :=
by
  use (-4/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3926_392642


namespace NUMINAMATH_CALUDE_pizza_three_toppings_l3926_392671

/-- Represents a pizza with 24 slices and three toppings -/
structure Pizza :=
  (pepperoni : Finset Nat)
  (mushrooms : Finset Nat)
  (olives : Finset Nat)
  (h1 : pepperoni ∪ mushrooms ∪ olives = Finset.range 24)
  (h2 : pepperoni.card = 15)
  (h3 : mushrooms.card = 14)
  (h4 : olives.card = 12)
  (h5 : (pepperoni ∩ mushrooms).card = 6)
  (h6 : (mushrooms ∩ olives).card = 5)
  (h7 : (pepperoni ∩ olives).card = 4)

theorem pizza_three_toppings (p : Pizza) : (p.pepperoni ∩ p.mushrooms ∩ p.olives).card = 0 := by
  sorry

end NUMINAMATH_CALUDE_pizza_three_toppings_l3926_392671


namespace NUMINAMATH_CALUDE_lucy_trip_graph_properties_l3926_392685

/-- Represents a segment of Lucy's trip -/
inductive TripSegment
  | CityTraffic
  | Highway
  | Roadwork
  | Shopping

/-- Represents the slope of a graph segment -/
inductive Slope
  | Flat
  | Gradual
  | Steep

/-- Represents Lucy's entire trip -/
def LucyTrip := List TripSegment

/-- Defines Lucy's trip based on the problem description -/
def lucyTripSegments : LucyTrip :=
  [TripSegment.CityTraffic, TripSegment.Highway, TripSegment.Roadwork, 
   TripSegment.Shopping,
   TripSegment.Roadwork, TripSegment.Highway, TripSegment.CityTraffic]

/-- Maps a trip segment to its corresponding slope -/
def segmentToSlope (segment : TripSegment) : Slope :=
  match segment with
  | TripSegment.CityTraffic => Slope.Gradual
  | TripSegment.Highway => Slope.Steep
  | TripSegment.Roadwork => Slope.Gradual
  | TripSegment.Shopping => Slope.Flat

/-- Theorem stating that Lucy's trip graph has varying slopes and includes a flat part -/
theorem lucy_trip_graph_properties (trip : LucyTrip) : 
  trip = lucyTripSegments → 
  (∃ (s1 s2 : Slope), s1 ≠ s2 ∧ s1 ∈ (trip.map segmentToSlope) ∧ s2 ∈ (trip.map segmentToSlope)) ∧
  (Slope.Flat ∈ (trip.map segmentToSlope)) := by
  sorry


end NUMINAMATH_CALUDE_lucy_trip_graph_properties_l3926_392685


namespace NUMINAMATH_CALUDE_bus_rental_equation_l3926_392680

theorem bus_rental_equation (x : ℝ) (h : x > 2) :
  180 / x - 180 / (x + 2) = 3 :=
by sorry


end NUMINAMATH_CALUDE_bus_rental_equation_l3926_392680


namespace NUMINAMATH_CALUDE_infinite_pairs_with_same_prime_factors_l3926_392626

theorem infinite_pairs_with_same_prime_factors :
  ∀ k : ℕ, k > 1 →
  ∃ m n : ℕ, m ≠ n ∧ m > 0 ∧ n > 0 ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1))) ∧
  m = 2^k - 2 ∧
  n = 2^k * (2^k - 2) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_same_prime_factors_l3926_392626


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_l3926_392641

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_not_divisible : ∃ (n : ℕ), n = 124 ∧ 
  ¬(factorial 1999 ∣ 34^n * factorial n) ∧ 
  ∀ (m : ℕ), m < n → (factorial 1999 ∣ 34^m * factorial m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_l3926_392641


namespace NUMINAMATH_CALUDE_product_inspection_probabilities_l3926_392636

/-- Given a total of 10 products with 8 first-grade and 2 second-grade products,
    calculate probabilities when 2 products are randomly inspected. -/
theorem product_inspection_probabilities :
  let total_products : ℕ := 10
  let first_grade_products : ℕ := 8
  let second_grade_products : ℕ := 2
  let inspected_products : ℕ := 2

  -- Probability that both products are first-grade
  (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 28/45 ∧
  
  -- Probability that at least one product is second-grade
  1 - (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 17/45 :=
by
  sorry


end NUMINAMATH_CALUDE_product_inspection_probabilities_l3926_392636


namespace NUMINAMATH_CALUDE_function_properties_l3926_392619

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * Real.sin (ω * x / 2) ^ 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem function_properties
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : is_periodic (f ω) (3 * Real.pi))
  (h_min_period : ∀ T, 0 < T ∧ T < 3 * Real.pi → ¬ is_periodic (f ω) T)
  (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_f_C : f ω C = 1)
  (h_trig_eq : 2 * Real.sin (2 * B) = Real.cos B + Real.cos (A - C)) :
  (∃ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), ∀ y ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f ω x ≤ f ω y) ∧
  f ω (Real.pi / 2) = Real.sqrt 3 - 1 ∧
  Real.sin A = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3926_392619


namespace NUMINAMATH_CALUDE_expression_equals_25_l3926_392696

theorem expression_equals_25 (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 1) :
  x + y = 25 := by sorry

end NUMINAMATH_CALUDE_expression_equals_25_l3926_392696


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3926_392661

open Real

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), 
    a = 3 * x₁ + 3 ∧ 
    a = 2 * x₂ + log x₂ ∧
    ∀ (y₁ y₂ : ℝ), 
      (a = 3 * y₁ + 3 ∧ a = 2 * y₂ + log y₂) → 
      |x₂ - x₁| ≤ |y₂ - y₁| ∧
      |x₂ - x₁| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3926_392661


namespace NUMINAMATH_CALUDE_simplify_fraction_l3926_392669

theorem simplify_fraction : (210 : ℚ) / 315 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3926_392669


namespace NUMINAMATH_CALUDE_unique_even_solution_l3926_392627

def f (n : ℤ) : ℤ :=
  if n < 0 then n^2 + 4*n + 4 else 3*n - 15

theorem unique_even_solution :
  ∃! a : ℤ, Even a ∧ f (-3) + f 3 + f a = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_even_solution_l3926_392627


namespace NUMINAMATH_CALUDE_median_intersection_locus_l3926_392660

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

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  edge1 : Point3D → Prop
  edge2 : Point3D → Prop
  edge3 : Point3D → Prop

/-- The locus of median intersections in a trihedral angle -/
def medianIntersectionLocus (angle : TrihedralAngle) (A : Point3D) : Plane3D :=
  sorry

/-- Main theorem: The locus of median intersections is a plane parallel to OBC and 1/3 away from A -/
theorem median_intersection_locus 
  (angle : TrihedralAngle) 
  (A : Point3D) 
  (h1 : angle.edge1 A) :
  ∃ (plane : Plane3D),
    (medianIntersectionLocus angle A = plane) ∧ 
    (∃ (B C : Point3D), 
      angle.edge2 B ∧ 
      angle.edge3 C ∧ 
      (plane.a * B.x + plane.b * B.y + plane.c * B.z + plane.d = 0) ∧
      (plane.a * C.x + plane.b * C.y + plane.c * C.z + plane.d = 0)) ∧
    (∃ (k : ℝ), k = 1/3 ∧ 
      (plane.a * A.x + plane.b * A.y + plane.c * A.z + plane.d = k * 
       (plane.a * angle.vertex.x + plane.b * angle.vertex.y + plane.c * angle.vertex.z + plane.d))) :=
by sorry

end NUMINAMATH_CALUDE_median_intersection_locus_l3926_392660


namespace NUMINAMATH_CALUDE_pyramid_volume_l3926_392605

/-- The volume of a pyramid with a right triangular base of side length 2 and height 2 is 4/3 -/
theorem pyramid_volume (s h : ℝ) (hs : s = 2) (hh : h = 2) :
  (1 / 3 : ℝ) * (1 / 2 * s * s) * h = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3926_392605


namespace NUMINAMATH_CALUDE_complex_modulus_and_argument_l3926_392620

open Complex

theorem complex_modulus_and_argument : 
  let z : ℂ := -Complex.sin (π/8) - Complex.I * Complex.cos (π/8)
  (abs z = 1) ∧ (arg z = -5*π/8) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_and_argument_l3926_392620


namespace NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l3926_392637

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l3926_392637


namespace NUMINAMATH_CALUDE_equation_solution_l3926_392604

theorem equation_solution (x : ℝ) :
  (5.31 * Real.tan (6 * x) * Real.cos (2 * x) - Real.sin (2 * x) - 2 * Real.sin (4 * x) = 0) ↔
  (∃ k : ℤ, x = k * π / 2) ∨ (∃ k : ℤ, x = π / 18 * (6 * k + 1) ∨ x = π / 18 * (6 * k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3926_392604


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_attained_l3926_392682

theorem quadratic_minimum (x : ℝ) : 2 * x^2 + 16 * x + 40 ≥ 8 := by sorry

theorem quadratic_minimum_attained : ∃ x : ℝ, 2 * x^2 + 16 * x + 40 = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_attained_l3926_392682


namespace NUMINAMATH_CALUDE_last_season_episodes_l3926_392668

/-- The number of seasons before the announcement -/
def previous_seasons : ℕ := 9

/-- The number of episodes in each regular season -/
def episodes_per_season : ℕ := 22

/-- The duration of each episode in hours -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all seasons in hours -/
def total_watch_time : ℚ := 112

/-- The additional episodes in the last season compared to regular seasons -/
def additional_episodes : ℕ := 4

theorem last_season_episodes (last_season_episodes : ℕ) :
  last_season_episodes = episodes_per_season + additional_episodes ∧
  (previous_seasons * episodes_per_season + last_season_episodes) * episode_duration = total_watch_time :=
by sorry

end NUMINAMATH_CALUDE_last_season_episodes_l3926_392668


namespace NUMINAMATH_CALUDE_glenn_spends_35_dollars_l3926_392666

/-- The cost of a movie ticket on Monday -/
def monday_price : ℕ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℕ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℕ := 5 * monday_price

/-- The total amount Glenn spends on movie tickets -/
def glenn_total_spent : ℕ := wednesday_price + saturday_price

/-- Theorem stating that Glenn spends $35 on movie tickets -/
theorem glenn_spends_35_dollars : glenn_total_spent = 35 := by
  sorry

end NUMINAMATH_CALUDE_glenn_spends_35_dollars_l3926_392666


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3926_392632

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), a^2 - 1 + (a + 1) * Complex.I = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3926_392632


namespace NUMINAMATH_CALUDE_field_trip_students_l3926_392698

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) :
  van_capacity = 7 →
  num_vans = 6 →
  num_adults = 9 →
  (van_capacity * num_vans - num_adults : ℕ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l3926_392698


namespace NUMINAMATH_CALUDE_loan_duration_proof_l3926_392699

/-- Represents the annual interest rate as a decimal -/
def interest_rate (percent : ℚ) : ℚ := percent / 100

/-- Calculates the annual interest given a principal and an interest rate -/
def annual_interest (principal : ℚ) (rate : ℚ) : ℚ := principal * rate

/-- Calculates the gain over a period of time -/
def gain (annual_gain : ℚ) (years : ℚ) : ℚ := annual_gain * years

theorem loan_duration_proof (principal : ℚ) (rate_A_to_B rate_B_to_C total_gain : ℚ) :
  principal = 2000 →
  rate_A_to_B = interest_rate 10 →
  rate_B_to_C = interest_rate 11.5 →
  total_gain = 90 →
  gain (annual_interest principal rate_B_to_C - annual_interest principal rate_A_to_B) 3 = total_gain :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_proof_l3926_392699


namespace NUMINAMATH_CALUDE_apple_lovers_problem_l3926_392613

theorem apple_lovers_problem (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) 
  (h1 : total_apples = 1430)
  (h2 : initial_per_person = 22)
  (h3 : decrease = 9) :
  ∃ (initial_people new_people : ℕ),
    initial_people * initial_per_person = total_apples ∧
    (initial_people + new_people) * (initial_per_person - decrease) = total_apples ∧
    new_people = 45 := by
  sorry

end NUMINAMATH_CALUDE_apple_lovers_problem_l3926_392613


namespace NUMINAMATH_CALUDE_inequality_proof_l3926_392621

theorem inequality_proof (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3926_392621


namespace NUMINAMATH_CALUDE_oranges_taken_away_l3926_392694

/-- Represents the number of fruits in Tina's bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits Tina took away -/
structure FruitsTakenAway where
  oranges : Nat
  tangerines : Nat

def initial_bag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def fruits_taken : FruitsTakenAway := { oranges := 2, tangerines := 10 }

theorem oranges_taken_away (bag : FruitBag) (taken : FruitsTakenAway) : 
  taken.oranges = 2 ↔ 
    (bag.tangerines - taken.tangerines = (bag.oranges - taken.oranges) + 4) ∧
    (taken.tangerines = 10) ∧
    (bag = initial_bag) := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_away_l3926_392694


namespace NUMINAMATH_CALUDE_inequality_proof_l3926_392615

theorem inequality_proof (a b c : ℝ) : 
  a = 4/5 → b = Real.sin (2/3) → c = Real.cos (1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3926_392615


namespace NUMINAMATH_CALUDE_cara_seating_arrangement_l3926_392695

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem cara_seating_arrangement :
  choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangement_l3926_392695


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3926_392664

theorem arithmetic_computation : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3926_392664


namespace NUMINAMATH_CALUDE_sine_function_period_l3926_392690

/-- Given a sinusoidal function with angular frequency ω > 0 and smallest positive period 2π/3, prove that ω = 3. -/
theorem sine_function_period (ω : ℝ) : ω > 0 → (∀ x, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + 2 * π / 3) + π / 6)) → ω = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_period_l3926_392690


namespace NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l3926_392612

/-- Given two equilateral triangles, one inside the other, this theorem calculates
    the area of one of the three congruent trapezoids formed between them. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ)
  (inner_area : ℝ)
  (h_outer : outer_area = 36)
  (h_inner : inner_area = 4)
  (h_positive : 0 < inner_area ∧ inner_area < outer_area) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l3926_392612


namespace NUMINAMATH_CALUDE_intersection_of_isosceles_and_right_angled_l3926_392633

-- Define the set of all triangles
def Triangle : Type := sorry

-- Define the property of being isosceles
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the property of being right-angled
def IsRightAngled (t : Triangle) : Prop := sorry

-- Define the set of isosceles triangles
def M : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the set of right-angled triangles
def N : Set Triangle := {t : Triangle | IsRightAngled t}

-- Define the property of being both isosceles and right-angled
def IsIsoscelesRightAngled (t : Triangle) : Prop := IsIsosceles t ∧ IsRightAngled t

-- Theorem statement
theorem intersection_of_isosceles_and_right_angled :
  M ∩ N = {t : Triangle | IsIsoscelesRightAngled t} := by sorry

end NUMINAMATH_CALUDE_intersection_of_isosceles_and_right_angled_l3926_392633


namespace NUMINAMATH_CALUDE_A_older_than_B_by_two_l3926_392611

-- Define the ages of A, B, and C
def B : ℕ := 14
def C : ℕ := B / 2
def A : ℕ := 37 - B - C

-- Theorem statement
theorem A_older_than_B_by_two : A = B + 2 := by
  sorry

end NUMINAMATH_CALUDE_A_older_than_B_by_two_l3926_392611


namespace NUMINAMATH_CALUDE_hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l3926_392697

def H (n : ℕ+) : ℕ := 2 * n.val ^ 2 - n.val

theorem hexagonal_gcd_bound (n : ℕ+) : Nat.gcd (3 * H n) (n.val + 1) ≤ 12 :=
sorry

theorem hexagonal_gcd_achieves_bound : ∃ n : ℕ+, Nat.gcd (3 * H n) (n.val + 1) = 12 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l3926_392697


namespace NUMINAMATH_CALUDE_cylinder_radius_l3926_392616

theorem cylinder_radius (length width : Real) (h1 : length = 3 * Real.pi) (h2 : width = Real.pi) :
  ∃ (r : Real), (r = 3/2 ∨ r = 1/2) ∧ 
  (2 * Real.pi * r = length ∨ 2 * Real.pi * r = width) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l3926_392616


namespace NUMINAMATH_CALUDE_two_scoop_sundaes_l3926_392678

theorem two_scoop_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_scoop_sundaes_l3926_392678


namespace NUMINAMATH_CALUDE_no_real_solutions_l3926_392618

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
  (∀ x : ℝ, a * x^2 + a * x + a ≠ b) ↔ (a = 0 ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3926_392618
