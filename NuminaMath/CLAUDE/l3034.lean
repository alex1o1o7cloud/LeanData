import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_width_l3034_303497

/-- Given a rectangle with length 4 times its width and area 196 square inches, 
    prove that its width is 7 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : w * (4 * w) = 196) : w = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3034_303497


namespace NUMINAMATH_CALUDE_power_of_five_equality_l3034_303484

theorem power_of_five_equality (k : ℕ) : 5^k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_equality_l3034_303484


namespace NUMINAMATH_CALUDE_minimum_value_of_fraction_l3034_303461

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_of_fraction (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 9 / n ≥ 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_fraction_l3034_303461


namespace NUMINAMATH_CALUDE_trig_function_slope_angle_l3034_303488

/-- Given a trigonometric function f(x) = a*sin(x) - b*cos(x) with the property
    that f(π/4 - x) = f(π/4 + x) for all x, prove that the slope angle of the line
    ax - by + c = 0 is 3π/4. -/
theorem trig_function_slope_angle (a b c : ℝ) :
  (∀ x, a * Real.sin x - b * Real.cos x = a * Real.sin (Real.pi/4 - x) - b * Real.cos (Real.pi/4 - x)) →
  (∃ k : ℝ, k > 0 ∧ a = k ∧ b = k) →
  Real.arctan (a / b) = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_trig_function_slope_angle_l3034_303488


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l3034_303470

theorem smallest_number_with_remainder_two : ∃ n : ℕ,
  n > 2 ∧
  n % 9 = 2 ∧
  n % 10 = 2 ∧
  n % 11 = 2 ∧
  (∀ m : ℕ, m > 2 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ n) ∧
  n = 992 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l3034_303470


namespace NUMINAMATH_CALUDE_inequality_condition_l3034_303400

theorem inequality_condition (a b c : ℝ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b → (a + Real.sqrt (b + c) > b + Real.sqrt (a + c))) ↔ c > (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3034_303400


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3034_303471

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 2.5) : x^2 + (1 / x^2) = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3034_303471


namespace NUMINAMATH_CALUDE_largest_package_size_l3034_303413

theorem largest_package_size (lucy_markers emma_markers : ℕ) 
  (h1 : lucy_markers = 54)
  (h2 : emma_markers = 36) :
  Nat.gcd lucy_markers emma_markers = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3034_303413


namespace NUMINAMATH_CALUDE_negative_solutions_count_l3034_303499

def f (x : ℤ) : ℤ := x^6 - 75*x^4 + 1000*x^2 - 6000

theorem negative_solutions_count :
  ∃! (S : Finset ℤ), (∀ x ∈ S, f x < 0) ∧ (∀ x ∉ S, f x ≥ 0) ∧ Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_solutions_count_l3034_303499


namespace NUMINAMATH_CALUDE_triangle_equilateral_from_cosine_product_l3034_303439

theorem triangle_equilateral_from_cosine_product (A B C : ℝ) 
  (triangle_condition : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (angle_sum : A + B + C = π) 
  (cosine_product : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_from_cosine_product_l3034_303439


namespace NUMINAMATH_CALUDE_coin_difference_l3034_303469

/-- Represents the number of coins of a specific denomination a person has -/
structure CoinCount where
  fiveRuble : ℕ
  twoRuble : ℕ

/-- Calculates the total value in rubles for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  5 * coins.fiveRuble + 2 * coins.twoRuble

/-- Represents the coin counts for Petya and Vanya -/
structure CoinDistribution where
  petya : CoinCount
  vanya : CoinCount

/-- Checks if the coin distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  dist.vanya.fiveRuble = dist.petya.twoRuble ∧
  dist.vanya.twoRuble = dist.petya.fiveRuble ∧
  totalValue dist.petya = totalValue dist.vanya + 60

theorem coin_difference (dist : CoinDistribution) 
  (h : isValidDistribution dist) : 
  dist.petya.fiveRuble - dist.petya.twoRuble = 20 :=
sorry

end NUMINAMATH_CALUDE_coin_difference_l3034_303469


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3034_303468

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  x^2 + y^2 = 13 := by sorry

theorem problem_2 (a : ℝ) (h : (4 - a)^2 + (a + 3)^2 = 7) :
  (4 - a) * (a + 3) = 21 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3034_303468


namespace NUMINAMATH_CALUDE_jeffrey_bottle_caps_l3034_303462

/-- 
Given that Jeffrey can create 6 groups of bottle caps with 2 bottle caps in each group,
prove that the total number of bottle caps is 12.
-/
theorem jeffrey_bottle_caps : 
  let groups : ℕ := 6
  let caps_per_group : ℕ := 2
  groups * caps_per_group = 12 := by sorry

end NUMINAMATH_CALUDE_jeffrey_bottle_caps_l3034_303462


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3034_303490

theorem solve_linear_equation (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3034_303490


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l3034_303486

/-- Given that i is the imaginary unit and |((a+i)/i)| = 2, prove that a = √3 where a is a positive real number. -/
theorem imaginary_unit_equation (i : ℂ) (a : ℝ) (h1 : i * i = -1) (h2 : a > 0) :
  Complex.abs ((a + i) / i) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l3034_303486


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3034_303419

/-- Represents a school population with teachers and students. -/
structure SchoolPopulation where
  teachers : ℕ
  maleStudents : ℕ
  femaleStudents : ℕ

/-- Represents a stratified sample from the school population. -/
structure StratifiedSample where
  totalSize : ℕ
  femalesSampled : ℕ

/-- Theorem: Given the school population and number of females sampled, 
    the total sample size is 192. -/
theorem stratified_sample_size 
  (school : SchoolPopulation)
  (sample : StratifiedSample)
  (h1 : school.teachers = 200)
  (h2 : school.maleStudents = 1200)
  (h3 : school.femaleStudents = 1000)
  (h4 : sample.femalesSampled = 80) :
  sample.totalSize = 192 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3034_303419


namespace NUMINAMATH_CALUDE_households_with_only_bike_l3034_303416

-- Define the total number of households
def total_households : ℕ := 90

-- Define the number of households without car or bike
def households_without_car_or_bike : ℕ := 11

-- Define the number of households with both car and bike
def households_with_both : ℕ := 16

-- Define the number of households with a car
def households_with_car : ℕ := 44

-- Theorem to prove
theorem households_with_only_bike : 
  total_households - households_without_car_or_bike - households_with_car + households_with_both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l3034_303416


namespace NUMINAMATH_CALUDE_range_of_special_set_l3034_303424

def is_valid_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ c = 10 ∧ (a + b + c) / 3 = 6 ∧ b = 6

theorem range_of_special_set :
  ∀ a b c : ℝ, is_valid_set a b c → c - a = 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_special_set_l3034_303424


namespace NUMINAMATH_CALUDE_pet_store_birds_l3034_303489

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℝ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3034_303489


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3034_303436

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3034_303436


namespace NUMINAMATH_CALUDE_girls_count_l3034_303466

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 402

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys + girl_boy_difference

theorem girls_count : num_girls = 739 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l3034_303466


namespace NUMINAMATH_CALUDE_harry_bought_apples_l3034_303433

/-- The number of apples Harry initially had -/
def initial_apples : ℕ := 79

/-- The number of apples Harry ended up with -/
def final_apples : ℕ := 84

/-- The number of apples Harry bought -/
def bought_apples : ℕ := final_apples - initial_apples

theorem harry_bought_apples :
  bought_apples = final_apples - initial_apples :=
by sorry

end NUMINAMATH_CALUDE_harry_bought_apples_l3034_303433


namespace NUMINAMATH_CALUDE_unique_x_with_square_property_l3034_303407

theorem unique_x_with_square_property : ∃! x : ℕ+, 
  (∃ k : ℕ, (2 * x.val + 1 : ℕ) = k^2) ∧ 
  (∀ y : ℕ, (2 * x.val + 2 : ℕ) ≤ y ∧ y ≤ (3 * x.val + 2) → ¬∃ k : ℕ, y = k^2) ∧
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_square_property_l3034_303407


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3034_303493

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3034_303493


namespace NUMINAMATH_CALUDE_reservoir_capacity_shortage_l3034_303481

/-- Proves that the normal level of a reservoir is 7 million gallons short of total capacity
    given specific conditions about the current amount and capacity. -/
theorem reservoir_capacity_shortage :
  ∀ (current_amount normal_level total_capacity : ℝ),
  current_amount = 6 →
  current_amount = 2 * normal_level →
  current_amount = 0.6 * total_capacity →
  total_capacity - normal_level = 7 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_shortage_l3034_303481


namespace NUMINAMATH_CALUDE_a_value_is_two_l3034_303431

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x + 6

/-- The condition that f(a, x) > 0 only when x ∈ (-∞, -2) ∪ (3, ∞) -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x > 0 ↔ (x < -2 ∨ x > 3)

/-- The theorem stating that under the given condition, a = 2 -/
theorem a_value_is_two :
  ∃ a : ℝ, condition a ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_a_value_is_two_l3034_303431


namespace NUMINAMATH_CALUDE_three_points_with_midpoint_l3034_303414

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a line
structure Point where
  position : ℝ
  color : Color

-- Define the theorem
theorem three_points_with_midpoint
  (line : Set Point)
  (h_nonempty : Set.Nonempty line)
  (h_two_colors : ∃ p q : Point, p ∈ line ∧ q ∈ line ∧ p.color ≠ q.color)
  (h_one_color : ∀ p : Point, p ∈ line → (p.color = Color.Red ∨ p.color = Color.Blue)) :
  ∃ p q r : Point,
    p ∈ line ∧ q ∈ line ∧ r ∈ line ∧
    p.color = q.color ∧ q.color = r.color ∧
    q.position = (p.position + r.position) / 2 :=
sorry

end NUMINAMATH_CALUDE_three_points_with_midpoint_l3034_303414


namespace NUMINAMATH_CALUDE_common_chord_equation_l3034_303487

/-- Given two circles in the xy-plane, this theorem states the equation of the line
    on which their common chord lies. -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 10*x - 10*y = 0) →
  (x^2 + y^2 + 6*x - 2*y - 40 = 0) →
  (2*x + y - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3034_303487


namespace NUMINAMATH_CALUDE_inequalities_proof_l3034_303494

theorem inequalities_proof (a b r s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0) (hs : s > 0) 
  (hrs : 1/r + 1/s = 1) : 
  (a^2 * b ≤ 4 * ((a + b) / 3)^3) ∧ 
  ((a^r / r) + (b^s / s) ≥ a * b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3034_303494


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l3034_303411

theorem fraction_product_theorem : 
  (5 / 4 : ℚ) * (8 / 16 : ℚ) * (20 / 12 : ℚ) * (32 / 64 : ℚ) * 
  (50 / 20 : ℚ) * (40 / 80 : ℚ) * (70 / 28 : ℚ) * (48 / 96 : ℚ) = 625 / 768 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l3034_303411


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l3034_303412

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 4)^2 + (y - 5)^2 + (z - 6)^2 = 0 → x + y + z = 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l3034_303412


namespace NUMINAMATH_CALUDE_surface_area_difference_after_cube_removal_l3034_303450

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the surface area difference after cube removal -/
def surfaceAreaDifference (length width height cubeEdge : ℝ) : ℝ :=
  let originalArea := surfaceArea length width height
  let removedArea := 3 * cubeEdge ^ 2
  let addedArea := cubeEdge ^ 2
  originalArea - removedArea + addedArea - originalArea

theorem surface_area_difference_after_cube_removal :
  surfaceAreaDifference 5 4 3 2 = -8 := by sorry

end NUMINAMATH_CALUDE_surface_area_difference_after_cube_removal_l3034_303450


namespace NUMINAMATH_CALUDE_no_rational_square_in_sequence_l3034_303482

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_square_in_sequence :
  ∀ n : ℕ, ¬ ∃ r : ℚ, sequence_a n = r ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_square_in_sequence_l3034_303482


namespace NUMINAMATH_CALUDE_imo_2007_hktst_1_problem_6_l3034_303438

theorem imo_2007_hktst_1_problem_6 :
  ∀ x y : ℕ+, 
    (∃ k : ℕ+, x = 11 * k^2 ∧ y = 11 * k) ↔ 
    ∃ n : ℤ, (x.val^2 * y.val + x.val + y.val : ℤ) = n * (x.val * y.val^2 + y.val + 11) := by
  sorry

end NUMINAMATH_CALUDE_imo_2007_hktst_1_problem_6_l3034_303438


namespace NUMINAMATH_CALUDE_isosceles_triangle_construction_uniqueness_l3034_303465

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  radius : ℝ
  altitude : ℝ
  orthocenter : ℝ
  is_positive : base > 0 ∧ radius > 0 ∧ altitude > 0
  bisects_altitude : orthocenter = altitude / 2

/-- Theorem stating that an isosceles triangle can be uniquely constructed given the base, radius, and orthocenter condition -/
theorem isosceles_triangle_construction_uniqueness 
  (b r : ℝ) 
  (hb : b > 0) 
  (hr : r > 0) : 
  ∃! t : IsoscelesTriangle, t.base = b ∧ t.radius = r :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_construction_uniqueness_l3034_303465


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l3034_303426

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l3034_303426


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l3034_303496

/-- Represents the number of fruits in a salad -/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Conditions for the fruit salad problem -/
def validFruitSalad (s : FruitSalad) : Prop :=
  s.blueberries + s.raspberries + s.grapes + s.cherries = 350 ∧
  s.raspberries = 3 * s.blueberries ∧
  s.grapes = 4 * s.cherries ∧
  s.cherries = 5 * s.raspberries

/-- Theorem stating that a valid fruit salad has 66 cherries -/
theorem fruit_salad_cherries (s : FruitSalad) (h : validFruitSalad s) : s.cherries = 66 := by
  sorry

#check fruit_salad_cherries

end NUMINAMATH_CALUDE_fruit_salad_cherries_l3034_303496


namespace NUMINAMATH_CALUDE_sandy_watermelons_count_l3034_303435

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := total_watermelons - jason_watermelons

theorem sandy_watermelons_count : sandy_watermelons = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandy_watermelons_count_l3034_303435


namespace NUMINAMATH_CALUDE_ellipse_foci_l3034_303432

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  ∀ x y : ℝ, ellipse_equation x y → is_focus x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3034_303432


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3034_303401

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 13

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = {-13, -1, 1, 13} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3034_303401


namespace NUMINAMATH_CALUDE_jacobs_age_multiple_l3034_303405

/-- Proves that Jacob's age will be 3 times his son's age in five years -/
theorem jacobs_age_multiple (jacob_age son_age : ℕ) : 
  jacob_age = 40 →
  son_age = 10 →
  jacob_age - 5 = 7 * (son_age - 5) →
  (jacob_age + 5) = 3 * (son_age + 5) := by
  sorry

end NUMINAMATH_CALUDE_jacobs_age_multiple_l3034_303405


namespace NUMINAMATH_CALUDE_delivery_fee_percentage_l3034_303472

def toy_organizer_cost : ℝ := 78
def gaming_chair_cost : ℝ := 83
def toy_organizer_sets : ℕ := 3
def gaming_chairs : ℕ := 2
def total_paid : ℝ := 420

def total_before_fee : ℝ := toy_organizer_cost * toy_organizer_sets + gaming_chair_cost * gaming_chairs

def delivery_fee : ℝ := total_paid - total_before_fee

theorem delivery_fee_percentage : (delivery_fee / total_before_fee) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_delivery_fee_percentage_l3034_303472


namespace NUMINAMATH_CALUDE_first_oil_price_first_oil_price_is_40_l3034_303447

/-- Given two varieties of oil mixed together, calculate the price of the first variety. -/
theorem first_oil_price 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ) 
  (first_oil_volume : ℝ) : ℝ :=
  let total_volume := first_oil_volume + second_oil_volume
  let second_oil_total_cost := second_oil_volume * second_oil_price
  let mixture_total_cost := total_volume * mixture_price
  let first_oil_total_cost := mixture_total_cost - second_oil_total_cost
  first_oil_total_cost / first_oil_volume

/-- The price of the first variety of oil is 40, given the specified conditions. -/
theorem first_oil_price_is_40 : 
  first_oil_price 240 60 52 160 = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_oil_price_first_oil_price_is_40_l3034_303447


namespace NUMINAMATH_CALUDE_system_solution_l3034_303464

/-- Given a system of equations with parameters a, b, and c, prove that the solutions for x, y, and z are as stated. -/
theorem system_solution (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ (x y z : ℝ),
    x ≠ y ∧
    (x - y) / (x + z) = a ∧
    (x^2 - y^2) / (x + z) = b ∧
    (x^3 + x^2*y - x*y^2 - y^3) / (x + z)^2 = b^2 / (a^2 * c) ∧
    x = (a^3 * c + b) / (2 * a) ∧
    y = (b - a^3 * c) / (2 * a) ∧
    z = (2 * a^2 * c - a^3 * c - b) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3034_303464


namespace NUMINAMATH_CALUDE_vector_ratio_implies_k_l3034_303420

/-- Given vectors a and b in ℝ², if (a + 2b) / (3a - b) exists, then k = -6 -/
theorem vector_ratio_implies_k (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (-2, k))
  (h3 : ∃ (r : ℝ), r • (3 • a - b) = a + 2 • b) :
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_implies_k_l3034_303420


namespace NUMINAMATH_CALUDE_field_trip_students_l3034_303429

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def number_of_buses : ℕ := 7

/-- The total number of students going on the field trip -/
def total_students : ℕ := seats_per_bus * number_of_buses

theorem field_trip_students : total_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l3034_303429


namespace NUMINAMATH_CALUDE_yoga_time_l3034_303406

/-- Mandy's exercise routine -/
def exercise_routine (gym bicycle yoga : ℕ) : Prop :=
  -- Gym to bicycle ratio is 2:3
  3 * gym = 2 * bicycle ∧
  -- Yoga to total exercise ratio is 2:3
  3 * yoga = 2 * (gym + bicycle) ∧
  -- Mandy spends 30 minutes doing yoga
  yoga = 30

/-- Theorem stating that given the exercise routine, yoga time is 30 minutes -/
theorem yoga_time (gym bicycle yoga : ℕ) :
  exercise_routine gym bicycle yoga → yoga = 30 := by
  sorry

end NUMINAMATH_CALUDE_yoga_time_l3034_303406


namespace NUMINAMATH_CALUDE_cookies_sold_l3034_303402

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def lottery_win : ℕ := 500
def sister_gift : ℕ := 500
def remaining_needed : ℕ := 3214

theorem cookies_sold :
  ∃ (n : ℕ), n * cookie_price = 
    trip_cost - 
    (hourly_wage * hours_worked + 
     lottery_win + 
     2 * sister_gift + 
     remaining_needed) :=
by sorry

end NUMINAMATH_CALUDE_cookies_sold_l3034_303402


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3034_303403

theorem quadratic_roots_transformation (D E F : ℝ) (α β : ℝ) (h1 : D ≠ 0) :
  (D * α^2 + E * α + F = 0) →
  (D * β^2 + E * β + F = 0) →
  ∃ (p q : ℝ), (α^2 + 1)^2 + p * (α^2 + 1) + q = 0 ∧
                (β^2 + 1)^2 + p * (β^2 + 1) + q = 0 ∧
                p = (2 * D * F - E^2 - 2 * D^2) / D^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3034_303403


namespace NUMINAMATH_CALUDE_x_minus_y_range_l3034_303418

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the range of x - y
def range_x_minus_y (x y : ℝ) : Prop :=
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10

-- Theorem statement
theorem x_minus_y_range :
  ∀ (x y ρ θ : ℝ), C ρ θ → x = ρ * Real.cos θ → y = ρ * Real.sin θ → range_x_minus_y x y :=
sorry

end NUMINAMATH_CALUDE_x_minus_y_range_l3034_303418


namespace NUMINAMATH_CALUDE_dorothy_profit_l3034_303422

/-- Dorothy's doughnut business profit calculation -/
theorem dorothy_profit (ingredients_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) :
  ingredients_cost = 53 →
  num_doughnuts = 25 →
  price_per_doughnut = 3 →
  num_doughnuts * price_per_doughnut - ingredients_cost = 22 :=
by
  sorry

#check dorothy_profit

end NUMINAMATH_CALUDE_dorothy_profit_l3034_303422


namespace NUMINAMATH_CALUDE_flowerbed_perimeter_l3034_303404

/-- The perimeter of a rectangular flowerbed with given dimensions -/
theorem flowerbed_perimeter : 
  let width : ℝ := 4
  let length : ℝ := 2 * width - 1
  2 * (length + width) = 22 := by sorry

end NUMINAMATH_CALUDE_flowerbed_perimeter_l3034_303404


namespace NUMINAMATH_CALUDE_min_value_fourth_root_plus_reciprocal_l3034_303477

theorem min_value_fourth_root_plus_reciprocal (x : ℝ) (hx : x > 0) :
  2 * x^(1/4) + 1/x ≥ 3 ∧ (2 * x^(1/4) + 1/x = 3 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fourth_root_plus_reciprocal_l3034_303477


namespace NUMINAMATH_CALUDE_find_x_l3034_303409

theorem find_x (y z : ℝ) (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
                         (h2 : y + z = 110) : x = 106 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3034_303409


namespace NUMINAMATH_CALUDE_legs_heads_difference_l3034_303449

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- The main theorem about the difference between legs and twice the heads -/
theorem legs_heads_difference (group : AnimalGroup) 
    (h : group.cows = 18) : 
    totalLegs group - 2 * totalHeads group = 36 := by
  sorry


end NUMINAMATH_CALUDE_legs_heads_difference_l3034_303449


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3034_303452

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3034_303452


namespace NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l3034_303443

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 34:15 -/
theorem speed_ratio_eddy_freddy :
  let eddy_distance : ℝ := 510  -- km
  let eddy_time : ℝ := 3        -- hours
  let freddy_distance : ℝ := 300  -- km
  let freddy_time : ℝ := 4        -- hours
  let eddy_speed : ℝ := eddy_distance / eddy_time
  let freddy_speed : ℝ := freddy_distance / freddy_time
  (eddy_speed / freddy_speed) = 34 / 15 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l3034_303443


namespace NUMINAMATH_CALUDE_unique_valid_integer_l3034_303485

-- Define a type for 10-digit integers
def TenDigitInteger := Fin 10 → Fin 10

-- Define a property for strictly increasing sequence
def StrictlyIncreasing (n : TenDigitInteger) : Prop :=
  ∀ i j : Fin 10, i < j → n i < n j

-- Define a property for using each digit exactly once
def UsesEachDigitOnce (n : TenDigitInteger) : Prop :=
  ∀ d : Fin 10, ∃! i : Fin 10, n i = d

-- Define the set of valid integers
def ValidIntegers : Set TenDigitInteger :=
  {n | n 0 ≠ 0 ∧ StrictlyIncreasing n ∧ UsesEachDigitOnce n}

-- Theorem statement
theorem unique_valid_integer : ∃! n : TenDigitInteger, n ∈ ValidIntegers := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_integer_l3034_303485


namespace NUMINAMATH_CALUDE_tina_pens_theorem_l3034_303467

def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def yellow_pens : ℕ := pink_pens + green_pens - 5
def pens_used_per_day : ℕ := 4

theorem tina_pens_theorem :
  let total_pens := pink_pens + green_pens + blue_pens + yellow_pens
  let days_to_use_pink := (pink_pens + pens_used_per_day - 1) / pens_used_per_day
  total_pens = 46 ∧ days_to_use_pink = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_pens_theorem_l3034_303467


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l3034_303478

theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∃ (r : ℝ × ℝ), r ∈ line ∧
        dist p r ≥ dist q r) ∧
    (∃ (s : ℝ × ℝ), s ∈ line ∧ dist p s = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l3034_303478


namespace NUMINAMATH_CALUDE_sector_central_angle_l3034_303457

theorem sector_central_angle (area : Real) (radius : Real) (h1 : area = 3 / 8 * Real.pi) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 / 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3034_303457


namespace NUMINAMATH_CALUDE_binomial_variance_l3034_303479

variable (p : ℝ)

-- Define the random variable X
def X : ℕ → ℝ
| 0 => 1 - p
| 1 => p
| _ => 0

-- Conditions
axiom p_range : 0 < p ∧ p < 1

-- Define the probability mass function
def pmf (k : ℕ) : ℝ := X p k

-- Define the expected value
def expectation : ℝ := p

-- Define the variance
def variance : ℝ := p * (1 - p)

-- Theorem statement
theorem binomial_variance : 
  ∀ (p : ℝ), 0 < p ∧ p < 1 → variance p = p * (1 - p) :=
by sorry

end NUMINAMATH_CALUDE_binomial_variance_l3034_303479


namespace NUMINAMATH_CALUDE_function_symmetry_l3034_303480

/-- Given a real-valued function f(x) = x³ + sin(x) + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem function_symmetry (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3 + Real.sin x + 1) 
    (h2 : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3034_303480


namespace NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l3034_303442

theorem max_geometric_mean_of_sequence (A : ℝ) (a : Fin 6 → ℝ) :
  (∃ i, a i = 1) →
  (∀ i, i < 4 → (a i + a (i + 1) + a (i + 2)) / 3 = (a (i + 1) + a (i + 2) + a (i + 3)) / 3) →
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 = A →
  (∃ i, i < 4 → 
    ∀ j, j < 4 → 
      (a j * a (j + 1) * a (j + 2)) ^ (1/3 : ℝ) ≤ ((3 * A - 1) ^ 2 / 4) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l3034_303442


namespace NUMINAMATH_CALUDE_colored_balls_theorem_l3034_303410

/-- Represents a box of colored balls -/
structure ColoredBalls where
  total : ℕ
  colors : ℕ
  min_same_color : ℕ → ℕ

/-- The problem statement -/
theorem colored_balls_theorem (box : ColoredBalls) 
  (h_total : box.total = 100)
  (h_colors : box.colors = 3)
  (h_min_same_color : box.min_same_color 26 ≥ 10) :
  box.min_same_color 66 ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_theorem_l3034_303410


namespace NUMINAMATH_CALUDE_max_min_distance_difference_l3034_303437

/-- Two unit squares with horizontal and vertical sides -/
structure UnitSquare where
  bottomLeft : ℝ × ℝ

/-- The minimum distance between two points -/
def minDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- The maximum distance between two points -/
def maxDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- Theorem: The difference between max and min possible y values is 5 - 3√2 -/
theorem max_min_distance_difference (s1 s2 : UnitSquare) 
  (h : minDistance s1 s2 = 5) :
  ∃ (yMin yMax : ℝ),
    yMin ≤ maxDistance s1 s2 ∧ 
    maxDistance s1 s2 ≤ yMax ∧
    yMax - yMin = 5 - 3 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_max_min_distance_difference_l3034_303437


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l3034_303460

theorem fruit_drink_volume (orange_percent : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  orange_percent = 0.15 →
  watermelon_percent = 0.60 →
  grape_ounces = 30 →
  ∃ total_ounces : ℝ,
    total_ounces = 120 ∧
    orange_percent * total_ounces + watermelon_percent * total_ounces + grape_ounces = total_ounces :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l3034_303460


namespace NUMINAMATH_CALUDE_circle_radius_l3034_303434

/-- Circle with center (3, -5) and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 5)^2 = r^2}

/-- Line 4x - 3y - 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 - 2 = 0}

/-- The shortest distance from a point on the circle to the line is 1 -/
def ShortestDistance (r : ℝ) : Prop :=
  ∃ p ∈ Circle r, ∀ q ∈ Circle r, ∀ l ∈ Line,
    dist p l ≤ dist q l ∧ dist p l = 1

theorem circle_radius (r : ℝ) :
  ShortestDistance r → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3034_303434


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_l3034_303440

-- Part 1
theorem part_1 (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3 ∧ x₁ + p / x₁ = q ∧ x₂ + p / x₂ = q) →
  p = -6 ∧ q = 1 := by sorry

-- Part 2
theorem part_2 :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ + 7 / x₁ = 8 ∧ x₂ + 7 / x₂ = 8) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ ∀ y : ℝ, y + 7 / y = 8 → y ≤ x) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ x = 7) := by sorry

-- Part 3
theorem part_3 (n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n) →
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n ∧
    (2 * x₁ - 1) / (2 * x₂) = (n - 1) / (n + 1)) := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_l3034_303440


namespace NUMINAMATH_CALUDE_tower_count_remainder_l3034_303425

/-- Represents a cube with an edge length --/
structure Cube where
  edge_length : ℕ

/-- Represents a tower of cubes --/
inductive Tower : Type
  | empty : Tower
  | cons : Cube → Tower → Tower

/-- Checks if a tower is valid according to the rules --/
def is_valid_tower : Tower → Bool
  | Tower.empty => true
  | Tower.cons c Tower.empty => true
  | Tower.cons c1 (Tower.cons c2 t) =>
    c1.edge_length ≤ c2.edge_length + 3 && is_valid_tower (Tower.cons c2 t)

/-- The set of cubes with edge lengths from 1 to 10 --/
def cube_set : List Cube :=
  List.map (λ k => ⟨k⟩) (List.range 10)

/-- Counts the number of valid towers that can be constructed --/
def count_valid_towers (cubes : List Cube) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem --/
theorem tower_count_remainder (U : ℕ) :
  U = count_valid_towers cube_set →
  U % 1000 = 536 :=
sorry

end NUMINAMATH_CALUDE_tower_count_remainder_l3034_303425


namespace NUMINAMATH_CALUDE_baker_eggs_theorem_l3034_303463

/-- Calculates the number of eggs needed for a given amount of flour, based on a recipe ratio. -/
def eggs_needed (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) : ℚ :=
  (available_flour / recipe_flour) * recipe_eggs

theorem baker_eggs_theorem (recipe_flour : ℚ) (recipe_eggs : ℚ) (available_flour : ℚ) 
  (h1 : recipe_flour = 2)
  (h2 : recipe_eggs = 3)
  (h3 : available_flour = 6) :
  eggs_needed recipe_flour recipe_eggs available_flour = 9 := by
  sorry

#eval eggs_needed 2 3 6

end NUMINAMATH_CALUDE_baker_eggs_theorem_l3034_303463


namespace NUMINAMATH_CALUDE_circle_equation_implies_value_l3034_303473

theorem circle_equation_implies_value (x y : ℝ) : 
  x^2 + y^2 - 12*x + 16*y + 100 = 0 → (x - 7)^(-y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_value_l3034_303473


namespace NUMINAMATH_CALUDE_ratio_problem_l3034_303454

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) :
  (x + y) / (y + z) = 26 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3034_303454


namespace NUMINAMATH_CALUDE_factorial_ratio_l3034_303458

theorem factorial_ratio : Nat.factorial 5 / Nat.factorial (5 - 3) = 60 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3034_303458


namespace NUMINAMATH_CALUDE_exists_fib_divisible_l3034_303483

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- For any natural number m, there exists a Fibonacci number divisible by m -/
theorem exists_fib_divisible (m : ℕ) : ∃ n : ℕ, n ≥ 1 ∧ m ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_l3034_303483


namespace NUMINAMATH_CALUDE_hawks_score_l3034_303495

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points per touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

theorem hawks_score :
  total_points = 21 :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l3034_303495


namespace NUMINAMATH_CALUDE_car_replacement_cost_l3034_303421

/-- Given an old car worth $20,000 sold at 80% of its value and a new car with
    a sticker price of $30,000 bought at 90% of its value, prove that the
    difference in cost (out of pocket) is $11,000. -/
theorem car_replacement_cost (old_car_value : ℝ) (new_car_price : ℝ)
    (old_car_sale_percentage : ℝ) (new_car_buy_percentage : ℝ)
    (h1 : old_car_value = 20000)
    (h2 : new_car_price = 30000)
    (h3 : old_car_sale_percentage = 0.8)
    (h4 : new_car_buy_percentage = 0.9) :
    new_car_buy_percentage * new_car_price - old_car_sale_percentage * old_car_value = 11000 :=
by sorry

end NUMINAMATH_CALUDE_car_replacement_cost_l3034_303421


namespace NUMINAMATH_CALUDE_distance_sum_inequality_l3034_303417

theorem distance_sum_inequality (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 := by sorry

end NUMINAMATH_CALUDE_distance_sum_inequality_l3034_303417


namespace NUMINAMATH_CALUDE_container_capacity_proof_l3034_303476

theorem container_capacity_proof :
  ∀ (C : ℝ),
    (C > 0) →
    (0.3 * C + 27 = 0.75 * C) →
    C = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_proof_l3034_303476


namespace NUMINAMATH_CALUDE_spiral_config_399_400_401_l3034_303423

/-- A function representing the spiral number sequence -/
def spiral_sequence : ℕ → ℕ := sorry

/-- Perfect squares are positioned at the center-bottom of their spiral layers -/
axiom perfect_square_position (n : ℕ) :
  ∃ (k : ℕ), k^2 = n → spiral_sequence n = spiral_sequence (n-1) + 1

/-- The vertical configuration of three consecutive numbers -/
def vertical_config (a b c : ℕ) : Prop :=
  spiral_sequence b = spiral_sequence a + 1 ∧
  spiral_sequence c = spiral_sequence b + 1

/-- Theorem stating the configuration of 399, 400, and 401 in the spiral -/
theorem spiral_config_399_400_401 :
  vertical_config 399 400 401 := by sorry

end NUMINAMATH_CALUDE_spiral_config_399_400_401_l3034_303423


namespace NUMINAMATH_CALUDE_optimal_robot_purchase_l3034_303474

/-- Represents the robot purchase problem -/
structure RobotPurchase where
  cost_A : ℕ  -- Cost of A robot in yuan
  cost_B : ℕ  -- Cost of B robot in yuan
  capacity_A : ℕ  -- Daily capacity of A robot in tons
  capacity_B : ℕ  -- Daily capacity of B robot in tons
  total_robots : ℕ  -- Total number of robots to purchase
  min_capacity : ℕ  -- Minimum daily capacity required

/-- The optimal solution minimizes the total cost -/
def optimal_solution (rp : RobotPurchase) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_robot_purchase :
  let rp : RobotPurchase := {
    cost_A := 12000,
    cost_B := 20000,
    capacity_A := 90,
    capacity_B := 100,
    total_robots := 30,
    min_capacity := 2830
  }
  let (num_A, num_B, total_cost) := optimal_solution rp
  num_A = 17 ∧ num_B = 13 ∧ total_cost = 464000 :=
by sorry

end NUMINAMATH_CALUDE_optimal_robot_purchase_l3034_303474


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3034_303455

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3034_303455


namespace NUMINAMATH_CALUDE_complex_number_location_l3034_303453

theorem complex_number_location (z : ℂ) : 
  z * Complex.I = 2015 - Complex.I → 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3034_303453


namespace NUMINAMATH_CALUDE_bees_after_14_days_l3034_303475

/-- Calculates the total number of bees in a hive after a given number of days -/
def totalBeesAfterDays (initialBees : ℕ) (beesHatchedPerDay : ℕ) (beesLostPerDay : ℕ) (days : ℕ) : ℕ :=
  initialBees + (beesHatchedPerDay - beesLostPerDay) * days + 1

/-- Theorem: Given the specified conditions, the total number of bees after 14 days is 64801 -/
theorem bees_after_14_days :
  totalBeesAfterDays 20000 5000 1800 14 = 64801 := by
  sorry

#eval totalBeesAfterDays 20000 5000 1800 14

end NUMINAMATH_CALUDE_bees_after_14_days_l3034_303475


namespace NUMINAMATH_CALUDE_non_self_intersecting_chains_count_l3034_303456

/-- Represents a point on a circle -/
structure CirclePoint where
  label : ℕ

/-- Represents a polygonal chain on a circle -/
structure PolygonalChain where
  points : List CirclePoint
  is_non_self_intersecting : Bool

/-- The number of ways to form a non-self-intersecting polygonal chain -/
def count_non_self_intersecting_chains (n : ℕ) : ℕ :=
  n * 2^(n-2)

/-- Theorem stating the number of ways to form a non-self-intersecting polygonal chain -/
theorem non_self_intersecting_chains_count 
  (n : ℕ) 
  (h : n > 1) :
  (∀ (chain : PolygonalChain), 
    chain.points.length = n ∧ 
    chain.is_non_self_intersecting = true) →
  (∃! count : ℕ, count = count_non_self_intersecting_chains n) :=
sorry

end NUMINAMATH_CALUDE_non_self_intersecting_chains_count_l3034_303456


namespace NUMINAMATH_CALUDE_extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l3034_303445

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

-- Theorem for the extreme values when a = -3
theorem extreme_values_when_a_neg_three :
  (∃ x₁ x₂ : ℝ, f (-3) x₁ = 5 ∧ f (-3) x₂ = -6 ∧
    ∀ x : ℝ, f (-3) x ≤ 5 ∧ f (-3) x ≥ -6) :=
sorry

-- Theorem for the intersection with x-axis when a ≥ 1
theorem one_intersection_when_a_ge_one :
  ∀ a : ℝ, a ≥ 1 →
    ∃! x : ℝ, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l3034_303445


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3034_303498

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 + 5*x - 14 < 0 ↔ -7 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3034_303498


namespace NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3034_303446

theorem tens_digit_of_N_power_20 (N : ℕ) (h1 : Even N) (h2 : ¬ (10 ∣ N)) :
  (N^20 % 100) / 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3034_303446


namespace NUMINAMATH_CALUDE_customer_ratio_l3034_303491

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def predicted_saturday_customers : ℕ := 574

/-- The total number of customers on Friday -/
def friday_customers : ℕ := breakfast_customers + lunch_customers + dinner_customers

/-- The theorem stating the ratio of predicted Saturday customers to Friday customers -/
theorem customer_ratio : 
  (predicted_saturday_customers : ℚ) / (friday_customers : ℚ) = 574 / 287 := by
  sorry


end NUMINAMATH_CALUDE_customer_ratio_l3034_303491


namespace NUMINAMATH_CALUDE_min_triangle_area_on_unit_grid_l3034_303492

/-- The area of a triangle given three points on a 2D grid -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- The minimum area of a triangle on a unit grid -/
theorem min_triangle_area_on_unit_grid : 
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    triangleArea x1 y1 x2 y2 x3 y3 = (1 / 2 : ℚ) ∧ 
    (∀ (a1 b1 a2 b2 a3 b3 : ℤ), triangleArea a1 b1 a2 b2 a3 b3 ≥ (1 / 2 : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_on_unit_grid_l3034_303492


namespace NUMINAMATH_CALUDE_f_greater_than_g_l3034_303408

def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_g_l3034_303408


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3034_303428

/-- Given the recess breaks and total time outside of class, prove the lunch break duration. -/
theorem lunch_break_duration 
  (recess1 recess2 recess3 total_outside : ℕ)
  (h1 : recess1 = 15)
  (h2 : recess2 = 15)
  (h3 : recess3 = 20)
  (h4 : total_outside = 80) :
  total_outside - (recess1 + recess2 + recess3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l3034_303428


namespace NUMINAMATH_CALUDE_first_valid_year_is_2913_l3034_303444

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2100 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2913 :
  (∀ y, 2100 < y ∧ y < 2913 → sum_of_digits y ≠ 15) ∧
  is_valid_year 2913 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2913_l3034_303444


namespace NUMINAMATH_CALUDE_at_least_two_equal_l3034_303441

theorem at_least_two_equal (x y z : ℝ) : 
  (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0 →
  (x = y ∨ y = z ∨ z = x) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l3034_303441


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l3034_303430

/-- Given a circle and rectangle with specific properties, prove the length of the rectangle's longer side --/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) (shorter_side longer_side : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  circle_area = π * r^2 →  -- Area of the circle
  rectangle_area = 3 * circle_area →  -- Rectangle area is three times circle area
  shorter_side = 2 * r →  -- Shorter side is twice the radius
  rectangle_area = shorter_side * longer_side →  -- Rectangle area formula
  longer_side = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l3034_303430


namespace NUMINAMATH_CALUDE_count_squares_with_six_black_l3034_303415

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  x : Nat
  y : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 6 black squares -/
def containsSixBlackSquares (s : Square) : Bool :=
  if s.size ≥ 5 then true
  else if s.size = 4 then (s.x + s.y) % 2 = 0
  else false

/-- Counts the number of squares containing at least 6 black squares -/
def countSquaresWithSixBlack : Nat :=
  let fourByFour := (boardSize - 3) * (boardSize - 3) / 2
  let fiveByFive := (boardSize - 4) * (boardSize - 4)
  let sixBySix := (boardSize - 5) * (boardSize - 5)
  let sevenBySeven := (boardSize - 6) * (boardSize - 6)
  let eightByEight := (boardSize - 7) * (boardSize - 7)
  let nineByNine := (boardSize - 8) * (boardSize - 8)
  let tenByTen := 1
  fourByFour + fiveByFive + sixBySix + sevenBySeven + eightByEight + nineByNine + tenByTen

theorem count_squares_with_six_black :
  countSquaresWithSixBlack = 115 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_with_six_black_l3034_303415


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3034_303459

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the interval [-2, 4]
def interval : Set ℝ := Set.Icc (-2) 4

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3034_303459


namespace NUMINAMATH_CALUDE_lower_bound_k_squared_l3034_303427

theorem lower_bound_k_squared (k : ℤ) (V : ℤ) (h1 : k^2 > V) (h2 : k^2 < 225) 
  (h3 : ∃ (S : Finset ℤ), S.card ≤ 6 ∧ ∀ x, x ∈ S ↔ x^2 > V ∧ x^2 < 225) :
  81 ≤ k^2 := by
  sorry

end NUMINAMATH_CALUDE_lower_bound_k_squared_l3034_303427


namespace NUMINAMATH_CALUDE_pregnant_fish_count_l3034_303451

theorem pregnant_fish_count (tanks : ℕ) (young_per_fish : ℕ) (total_young : ℕ) :
  tanks = 3 →
  young_per_fish = 20 →
  total_young = 240 →
  ∃ fish_per_tank : ℕ, fish_per_tank * tanks * young_per_fish = total_young ∧ fish_per_tank = 4 :=
by sorry

end NUMINAMATH_CALUDE_pregnant_fish_count_l3034_303451


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3034_303448

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that (m + n)^2023 = -1 --/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (m = 3 ∧ n = -4) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3034_303448
