import Mathlib

namespace NUMINAMATH_CALUDE_exponent_calculation_l3578_357849

theorem exponent_calculation : 3^3 * 5^3 * 3^5 * 5^5 = 15^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3578_357849


namespace NUMINAMATH_CALUDE_tin_silver_ratio_l3578_357839

/-- Represents the composition of a metal bar made of tin and silver -/
structure MetalBar where
  tin : ℝ
  silver : ℝ

/-- Properties of the metal bar -/
def bar_properties (bar : MetalBar) : Prop :=
  bar.tin + bar.silver = 40 ∧
  0.1375 * bar.tin + 0.075 * bar.silver = 4

/-- The ratio of tin to silver in the bar is 2:3 -/
theorem tin_silver_ratio (bar : MetalBar) :
  bar_properties bar → bar.tin / bar.silver = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tin_silver_ratio_l3578_357839


namespace NUMINAMATH_CALUDE_books_together_l3578_357898

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem books_together : total_books 20 21 = 41 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l3578_357898


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3578_357891

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3578_357891


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l3578_357834

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 8
  let b : ℤ := a - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l3578_357834


namespace NUMINAMATH_CALUDE_medication_mixture_volume_l3578_357826

/-- Given a mixture of two medications A and B, where:
    - Medication A contains 40% pain killer
    - Medication B contains 20% pain killer
    - The patient receives exactly 215 milliliters of pain killer daily
    - There are 425 milliliters of medication B in the mixture
    Prove that the total volume of the mixture given to the patient daily is 750 milliliters. -/
theorem medication_mixture_volume :
  let medication_a_percentage : ℝ := 0.40
  let medication_b_percentage : ℝ := 0.20
  let total_painkiller : ℝ := 215
  let medication_b_volume : ℝ := 425
  let total_mixture_volume : ℝ := (total_painkiller - medication_b_percentage * medication_b_volume) / 
                                  (medication_a_percentage - medication_b_percentage) + medication_b_volume
  total_mixture_volume = 750 :=
by sorry

end NUMINAMATH_CALUDE_medication_mixture_volume_l3578_357826


namespace NUMINAMATH_CALUDE_basis_properties_l3578_357817

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (S : Set V) : Prop :=
  Submodule.span ℝ S = ⊤ ∧ LinearIndependent ℝ (fun x => x : S → V)

theorem basis_properties {a b c : V} (h : is_basis {a, b, c}) :
  is_basis {a + b, b + c, c + a} ∧
  ∀ p : V, ∃ x y z : ℝ, p = x • a + y • b + z • c :=
sorry

end NUMINAMATH_CALUDE_basis_properties_l3578_357817


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3578_357887

theorem min_value_abs_sum (x : ℝ) : 
  |x + 1| + |x - 2| + |x - 3| ≥ 4 ∧ ∃ y : ℝ, |y + 1| + |y - 2| + |y - 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3578_357887


namespace NUMINAMATH_CALUDE_sin_cos_transformation_l3578_357846

/-- The transformation between sin and cos functions -/
theorem sin_cos_transformation (f g : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  (∀ x, g x = Real.cos (2 * x)) →
  (∀ θ, Real.sin θ = Real.cos (θ - π / 2)) →
  f x = g (x + 3 * π / 8) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_l3578_357846


namespace NUMINAMATH_CALUDE_ribbon_triangle_to_pentagon_l3578_357858

theorem ribbon_triangle_to_pentagon (triangle_side : ℝ) (pentagon_side : ℝ) : 
  triangle_side = 20 / 9 → pentagon_side = (3 * triangle_side) / 5 → pentagon_side = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_triangle_to_pentagon_l3578_357858


namespace NUMINAMATH_CALUDE_half_percent_of_160_l3578_357897

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_160_l3578_357897


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l3578_357877

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part I
theorem intersection_complement_when_a_is_two :
  M ∩ (Set.univ \ N 2) = {x | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem union_equality_iff_a_leq_two (a : ℝ) :
  M ∪ N a = M ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l3578_357877


namespace NUMINAMATH_CALUDE_distance_difference_l3578_357860

/-- The distance Aleena biked in 5 hours -/
def aleena_distance : ℕ := 75

/-- The distance Bob biked in 5 hours -/
def bob_distance : ℕ := 60

/-- Theorem stating the difference between Aleena's and Bob's distances after 5 hours -/
theorem distance_difference : aleena_distance - bob_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3578_357860


namespace NUMINAMATH_CALUDE_square_of_3y_plus_4_when_y_is_neg_2_l3578_357809

theorem square_of_3y_plus_4_when_y_is_neg_2 :
  let y : ℤ := -2
  (3 * y + 4)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_square_of_3y_plus_4_when_y_is_neg_2_l3578_357809


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3578_357880

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (531 * m) % 24 = (1067 * m) % 24 → m ≥ n) ∧
  (531 * n) % 24 = (1067 * n) % 24 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3578_357880


namespace NUMINAMATH_CALUDE_second_applicant_revenue_l3578_357868

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The theorem to prove -/
theorem second_applicant_revenue
  (first : Applicant)
  (second : Applicant)
  (h1 : first.salary = 42000)
  (h2 : first.revenue = 93000)
  (h3 : first.trainingMonths = 3)
  (h4 : first.trainingCostPerMonth = 1200)
  (h5 : first.hiringBonusPercent = 0)
  (h6 : second.salary = 45000)
  (h7 : second.trainingMonths = 0)
  (h8 : second.trainingCostPerMonth = 0)
  (h9 : second.hiringBonusPercent = 1)
  (h10 : netGain second = netGain first + 850) :
  second.revenue = 93700 := by
  sorry

end NUMINAMATH_CALUDE_second_applicant_revenue_l3578_357868


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3578_357804

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i + 2) / i = 1 - 2 * i := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3578_357804


namespace NUMINAMATH_CALUDE_chord_length_l3578_357899

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3578_357899


namespace NUMINAMATH_CALUDE_concert_audience_fraction_l3578_357848

theorem concert_audience_fraction (total_audience : ℕ) 
  (second_band_fraction : ℚ) (h1 : total_audience = 150) 
  (h2 : second_band_fraction = 2/3) : 
  1 - second_band_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_concert_audience_fraction_l3578_357848


namespace NUMINAMATH_CALUDE_school_distance_l3578_357886

/-- The distance between a girl's house and school, given her travel speeds and total round trip time. -/
theorem school_distance (speed_to_school speed_from_school : ℝ) (total_time : ℝ) : 
  speed_to_school = 6 →
  speed_from_school = 4 →
  total_time = 10 →
  (1 / speed_to_school + 1 / speed_from_school) * (speed_to_school * speed_from_school / (speed_to_school + speed_from_school)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_l3578_357886


namespace NUMINAMATH_CALUDE_triangle_theorem_l3578_357833

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  t.C = π / 3 ∧ 
  (t.a = 5 → t.b = 8 → t.c = 7) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_theorem_l3578_357833


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l3578_357884

theorem x_plus_y_equals_four (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 16) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_four_l3578_357884


namespace NUMINAMATH_CALUDE_blind_box_equations_l3578_357873

/-- Represents the blind box production scenario -/
structure BlindBoxProduction where
  total_fabric : ℝ
  fabric_for_a : ℝ
  fabric_for_b : ℝ

/-- Conditions for the blind box production -/
def valid_production (p : BlindBoxProduction) : Prop :=
  p.total_fabric = 135 ∧
  p.fabric_for_a + p.fabric_for_b = p.total_fabric ∧
  2 * p.fabric_for_a = 3 * p.fabric_for_b

/-- Theorem stating the correct system of equations for the blind box production -/
theorem blind_box_equations (p : BlindBoxProduction) :
  valid_production p →
  p.fabric_for_a + p.fabric_for_b = 135 ∧ 2 * p.fabric_for_a = 3 * p.fabric_for_b := by
  sorry

end NUMINAMATH_CALUDE_blind_box_equations_l3578_357873


namespace NUMINAMATH_CALUDE_block_height_is_75_l3578_357843

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the properties of the cubes cut from the block -/
structure CubeProperties where
  sideLength : ℝ
  count : ℕ

/-- Checks if the given dimensions and cube properties satisfy the problem conditions -/
def satisfiesConditions (block : BlockDimensions) (cube : CubeProperties) : Prop :=
  block.length = 15 ∧
  block.width = 30 ∧
  cube.count = 10 ∧
  (cube.sideLength ∣ block.length) ∧
  (cube.sideLength ∣ block.width) ∧
  (cube.sideLength ∣ block.height) ∧
  block.length * block.width * block.height = cube.sideLength ^ 3 * cube.count

theorem block_height_is_75 (block : BlockDimensions) (cube : CubeProperties) :
  satisfiesConditions block cube → block.height = 75 := by
  sorry

end NUMINAMATH_CALUDE_block_height_is_75_l3578_357843


namespace NUMINAMATH_CALUDE_saree_discount_problem_l3578_357869

/-- Proves that the first discount percentage is 10% given the conditions of the saree pricing problem -/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 600 →
  second_discount = 5 →
  final_price = 513 →
  ∃ (first_discount : ℝ),
    first_discount = 10 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l3578_357869


namespace NUMINAMATH_CALUDE_power_value_from_equation_l3578_357874

theorem power_value_from_equation (x y : ℝ) 
  (h : |x - 2| + Real.sqrt (y + 3) = 0) : 
  y ^ x = 9 := by sorry

end NUMINAMATH_CALUDE_power_value_from_equation_l3578_357874


namespace NUMINAMATH_CALUDE_deposit_calculation_l3578_357871

theorem deposit_calculation (remaining : ℝ) (deposit_rate : ℝ) : 
  remaining = 950 →
  deposit_rate = 0.05 →
  (deposit_rate * (remaining / (1 - deposit_rate))) = 50 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3578_357871


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_reciprocals_l3578_357832

theorem quadratic_roots_negative_reciprocals (k : ℝ) : 
  (∃ α : ℝ, α ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = α ∨ x = -1/α))) →
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_reciprocals_l3578_357832


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3578_357863

/-- 
Given a selling price and a profit percentage, calculate the cost price.
-/
theorem cost_price_calculation 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : selling_price = 1800) 
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3578_357863


namespace NUMINAMATH_CALUDE_complex_power_approximation_l3578_357841

/-- The complex number (2 + i)/(2 - i) raised to the power of 600 is approximately equal to -0.982 - 0.189i -/
theorem complex_power_approximation :
  let z : ℂ := (2 + Complex.I) / (2 - Complex.I)
  ∃ (ε : ℝ) (hε : ε > 0), Complex.abs (z^600 - (-0.982 - 0.189 * Complex.I)) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_power_approximation_l3578_357841


namespace NUMINAMATH_CALUDE_line_through_three_points_l3578_357825

/-- A line contains the points (-2, 7), (7, k), and (21, 4). This theorem proves that k = 134/23. -/
theorem line_through_three_points (k : ℚ) : 
  (∃ (m b : ℚ), 
    (7 : ℚ) = m * (-2 : ℚ) + b ∧ 
    k = m * (7 : ℚ) + b ∧ 
    (4 : ℚ) = m * (21 : ℚ) + b) → 
  k = 134 / 23 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l3578_357825


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_attained_min_value_is_1215_l3578_357831

theorem min_value_sum (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c :=
by sorry

theorem min_value_attained (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 ∧ a + 2*b + 3*c = 1215 :=
by sorry

theorem min_value_is_1215 :
  ∃ (x y z : ℕ+), x^3 + y^3 + z^3 - 3*x*y*z = 607 ∧
  (∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c) ∧
  x + 2*y + 3*z = 1215 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_attained_min_value_is_1215_l3578_357831


namespace NUMINAMATH_CALUDE_A_subset_A_inter_B_iff_l3578_357844

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem A_subset_A_inter_B_iff (a : ℝ) : 
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by sorry

end NUMINAMATH_CALUDE_A_subset_A_inter_B_iff_l3578_357844


namespace NUMINAMATH_CALUDE_gas_consumption_reduction_l3578_357837

theorem gas_consumption_reduction (initial_price : ℝ) (initial_consumption : ℝ) 
  (h1 : initial_price > 0) (h2 : initial_consumption > 0) :
  let price_after_increases := initial_price * 1.3 * 1.2
  let new_consumption := initial_consumption * initial_price / price_after_increases
  let reduction_percentage := (initial_consumption - new_consumption) / initial_consumption * 100
  reduction_percentage = (1 - 1 / (1.3 * 1.2)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gas_consumption_reduction_l3578_357837


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3578_357836

theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3578_357836


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3578_357808

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3578_357808


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l3578_357816

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  line1 : Line
  line2 : Line
  line3 : Line

/-- Calculate the area of a triangle given its three bounding lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { line1 := { slope := 2, intercept := 0 }
  , line2 := { slope := -2, intercept := 0 }
  , line3 := { slope := 0, intercept := 4 }
  }

theorem triangle_area_is_eight :
  triangleArea problemTriangle = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l3578_357816


namespace NUMINAMATH_CALUDE_polar_bear_trout_consumption_l3578_357815

/-- The daily fish consumption of the polar bear in buckets -/
def total_fish : ℝ := 0.6

/-- The daily salmon consumption of the polar bear in buckets -/
def salmon : ℝ := 0.4

/-- The daily trout consumption of the polar bear in buckets -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption :
  trout = 0.2 := by sorry

end NUMINAMATH_CALUDE_polar_bear_trout_consumption_l3578_357815


namespace NUMINAMATH_CALUDE_smallest_good_is_correct_l3578_357855

/-- The operation described in the problem -/
def operation (n : ℕ) : ℕ :=
  (n / 10) + 2 * (n % 10)

/-- A number is 'good' if it's unchanged by the operation -/
def is_good (n : ℕ) : Prop :=
  operation n = n

/-- The smallest 'good' number -/
def smallest_good : ℕ :=
  10^99 + 1

theorem smallest_good_is_correct :
  is_good smallest_good ∧ ∀ m : ℕ, m < smallest_good → ¬ is_good m :=
sorry

end NUMINAMATH_CALUDE_smallest_good_is_correct_l3578_357855


namespace NUMINAMATH_CALUDE_bottle_ratio_is_half_l3578_357859

/-- Represents the distribution of bottles in a delivery van -/
structure BottleDistribution where
  total : ℕ
  cider : ℕ
  beer : ℕ
  mixed : ℕ
  first_house : ℕ

/-- The ratio of bottles given to the first house to the total number of bottles -/
def bottle_ratio (d : BottleDistribution) : ℚ :=
  d.first_house / d.total

/-- Theorem stating the ratio of bottles given to the first house to the total number of bottles -/
theorem bottle_ratio_is_half (d : BottleDistribution) 
    (h1 : d.total = 180)
    (h2 : d.cider = 40)
    (h3 : d.beer = 80)
    (h4 : d.mixed = d.total - d.cider - d.beer)
    (h5 : d.first_house = 90) : 
  bottle_ratio d = 1/2 := by
  sorry

#eval bottle_ratio { total := 180, cider := 40, beer := 80, mixed := 60, first_house := 90 }

end NUMINAMATH_CALUDE_bottle_ratio_is_half_l3578_357859


namespace NUMINAMATH_CALUDE_jessica_allowance_l3578_357819

def weekly_allowance : ℝ := 26.67

theorem jessica_allowance (allowance : ℝ) 
  (h1 : 0.45 * allowance + 17 = 29) : 
  allowance = weekly_allowance := by
  sorry

#check jessica_allowance

end NUMINAMATH_CALUDE_jessica_allowance_l3578_357819


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3578_357878

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3578_357878


namespace NUMINAMATH_CALUDE_maria_bike_purchase_l3578_357803

/-- The amount Maria needs to earn to buy a bike -/
def amount_to_earn (retail_price savings mother_contribution : ℕ) : ℕ :=
  retail_price - (savings + mother_contribution)

/-- Theorem: Maria needs to earn $230 to buy the bike -/
theorem maria_bike_purchase (retail_price savings mother_contribution : ℕ)
  (h1 : retail_price = 600)
  (h2 : savings = 120)
  (h3 : mother_contribution = 250) :
  amount_to_earn retail_price savings mother_contribution = 230 := by
  sorry

end NUMINAMATH_CALUDE_maria_bike_purchase_l3578_357803


namespace NUMINAMATH_CALUDE_sin_cos_shift_l3578_357896

/-- Given two functions f and g defined on real numbers,
    prove that they are equivalent up to a horizontal shift. -/
theorem sin_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x + π / 3)
  let g : ℝ → ℝ := λ x ↦ Real.cos (2 * x)
  f x = g (x - π / 12) := by
  sorry


end NUMINAMATH_CALUDE_sin_cos_shift_l3578_357896


namespace NUMINAMATH_CALUDE_friday_earnings_calculation_l3578_357866

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

end NUMINAMATH_CALUDE_friday_earnings_calculation_l3578_357866


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l3578_357892

/-- The floor number on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  1 + vasya_steps / (petya_steps / 2)

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

#eval vasya_floor 36 72

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l3578_357892


namespace NUMINAMATH_CALUDE_room_occupancy_l3578_357824

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (2 : ℚ) / 3 * people = (3 : ℚ) / 4 * chairs ∧ 
  chairs - (3 : ℚ) / 4 * chairs = 6 →
  people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l3578_357824


namespace NUMINAMATH_CALUDE_two_person_subcommittees_of_six_l3578_357852

/-- The number of two-person sub-committees from a six-person committee -/
def two_person_subcommittees (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The number of two-person sub-committees from a six-person committee is 15 -/
theorem two_person_subcommittees_of_six :
  two_person_subcommittees 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_of_six_l3578_357852


namespace NUMINAMATH_CALUDE_election_total_votes_l3578_357889

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  invalid_percent : ℚ
  b_votes : ℕ
  a_excess_percent : ℚ

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.invalid_percent = 1/5 ∧
  e.a_excess_percent = 3/20 ∧
  e.b_votes = 2184 ∧
  (e.total_votes : ℚ) * (1 - e.invalid_percent) = 
    (e.b_votes : ℚ) + (e.b_votes : ℚ) + e.total_votes * e.a_excess_percent

theorem election_total_votes (e : Election) (h : valid_election e) : 
  e.total_votes = 6720 := by
  sorry

#check election_total_votes

end NUMINAMATH_CALUDE_election_total_votes_l3578_357889


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3578_357893

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3578_357893


namespace NUMINAMATH_CALUDE_unique_number_l3578_357872

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def starts_with_six (n : ℕ) : Prop := ∃ k, n = 6000 + k ∧ 0 ≤ k ∧ k < 1000

def move_six_to_end (n : ℕ) : ℕ :=
  let k := n - 6000
  1000 * (k / 100) + 10 * (k % 100) + 6

theorem unique_number :
  ∃! n : ℕ, is_four_digit n ∧ 
            starts_with_six n ∧ 
            move_six_to_end n = n - 1152 ∧
            n = 6538 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l3578_357872


namespace NUMINAMATH_CALUDE_rebus_solution_l3578_357840

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) ∧
    100 * A + 10 * C + C = 1416 :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_l3578_357840


namespace NUMINAMATH_CALUDE_pie_eating_problem_l3578_357895

theorem pie_eating_problem (initial_stock : ℕ) (daily_portion : ℕ) (day : ℕ) :
  initial_stock = 340 →
  daily_portion > 0 →
  day > 0 →
  initial_stock = day * daily_portion + daily_portion / 4 →
  (day = 5 ∨ day = 21) :=
sorry

end NUMINAMATH_CALUDE_pie_eating_problem_l3578_357895


namespace NUMINAMATH_CALUDE_two_digit_multiplication_trick_l3578_357882

theorem two_digit_multiplication_trick (a b c : ℕ) 
  (h1 : b + c = 10) 
  (h2 : 0 ≤ a ∧ a ≤ 9) 
  (h3 : 0 ≤ b ∧ b ≤ 9) 
  (h4 : 0 ≤ c ∧ c ≤ 9) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_trick_l3578_357882


namespace NUMINAMATH_CALUDE_greenhouse_path_area_l3578_357813

/-- Calculates the total area of paths in Joanna's greenhouse --/
theorem greenhouse_path_area :
  let num_rows : ℕ := 5
  let beds_per_row : ℕ := 3
  let bed_width : ℕ := 4
  let bed_height : ℕ := 3
  let path_width : ℕ := 2
  
  let total_width : ℕ := beds_per_row * bed_width + (beds_per_row + 1) * path_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * path_width
  
  let total_area : ℕ := total_width * total_height
  let bed_area : ℕ := num_rows * beds_per_row * bed_width * bed_height
  
  total_area - bed_area = 360 :=
by sorry


end NUMINAMATH_CALUDE_greenhouse_path_area_l3578_357813


namespace NUMINAMATH_CALUDE_largest_number_value_l3578_357820

theorem largest_number_value (a b c : ℕ) : 
  a < b ∧ b < c ∧
  a + b + c = 80 ∧
  c = b + 9 ∧
  b = a + 4 ∧
  a * b = 525 →
  c = 34 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l3578_357820


namespace NUMINAMATH_CALUDE_acme_vowel_soup_words_l3578_357864

/-- The number of different letters available -/
def num_letters : ℕ := 5

/-- The number of times each letter appears -/
def letter_count : ℕ := 5

/-- The length of words to be formed -/
def word_length : ℕ := 5

/-- The total number of words that can be formed -/
def total_words : ℕ := num_letters ^ word_length

theorem acme_vowel_soup_words : total_words = 3125 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_words_l3578_357864


namespace NUMINAMATH_CALUDE_car_distance_l3578_357806

/-- Proves that a car traveling 2/3 as fast as a train going 90 miles per hour will cover 40 miles in 40 minutes -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2/3 →
  travel_time_minutes = 40 →
  (car_speed_ratio * train_speed) * (travel_time_minutes / 60) = 40 := by
  sorry

#check car_distance

end NUMINAMATH_CALUDE_car_distance_l3578_357806


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l3578_357856

theorem two_digit_number_interchange (a b k : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≤ 9) 
  (h3 : 10 * a + b = k * (a + b)) :
  10 * b + a = (11 - k) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l3578_357856


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_solutions_specific_equation_l3578_357847

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -48
  let b : ℝ := 66
  let c : ℝ := 195
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 11/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_solutions_specific_equation_l3578_357847


namespace NUMINAMATH_CALUDE_bicycle_price_reduction_l3578_357827

theorem bicycle_price_reduction (initial_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  initial_price = 200 →
  discount1 = 0.3 →
  discount2 = 0.4 →
  discount3 = 0.1 →
  initial_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 75.60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_reduction_l3578_357827


namespace NUMINAMATH_CALUDE_cassidy_grades_below_B_l3578_357821

/-- The number of grades below B that Cassidy received -/
def grades_below_B : ℕ := sorry

/-- The base grounding period in days -/
def base_grounding : ℕ := 14

/-- The additional grounding days for each grade below B -/
def extra_days_per_grade : ℕ := 3

/-- The total grounding period in days -/
def total_grounding : ℕ := 26

theorem cassidy_grades_below_B :
  grades_below_B * extra_days_per_grade + base_grounding = total_grounding ∧
  grades_below_B = 4 := by sorry

end NUMINAMATH_CALUDE_cassidy_grades_below_B_l3578_357821


namespace NUMINAMATH_CALUDE_bruce_mangoes_purchase_l3578_357822

/-- The amount of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 1110

/-- The amount of mangoes purchased in kg -/
def mangoes_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mangoes_price

theorem bruce_mangoes_purchase :
  mangoes_kg = 10 := by sorry

end NUMINAMATH_CALUDE_bruce_mangoes_purchase_l3578_357822


namespace NUMINAMATH_CALUDE_points_on_line_l3578_357801

/-- Given a line defined by x = (y^2 / 3) - (2 / 5), if three points (m, n), (m + p, n + 9), and (m + q, n + 18) lie on this line, then p = 6n + 27 and q = 12n + 108 -/
theorem points_on_line (m n p q : ℝ) : 
  (m = n^2 / 3 - 2 / 5) →
  (m + p = (n + 9)^2 / 3 - 2 / 5) →
  (m + q = (n + 18)^2 / 3 - 2 / 5) →
  (p = 6 * n + 27 ∧ q = 12 * n + 108) := by
sorry

end NUMINAMATH_CALUDE_points_on_line_l3578_357801


namespace NUMINAMATH_CALUDE_complex_product_l3578_357850

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 3)
  (h3 : 3 * z₁ - 2 * z₂ = (3 / 2 : ℂ) - Complex.I) :
  z₁ * z₂ = -(30 / 13 : ℂ) + (72 / 13 : ℂ) * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_l3578_357850


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l3578_357854

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1/a + 1/b + 1/c) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l3578_357854


namespace NUMINAMATH_CALUDE_expand_polynomial_l3578_357851

/-- Proves the expansion of (12x^2 + 5x - 3) * (3x^3 + 2) -/
theorem expand_polynomial (x : ℝ) :
  (12 * x^2 + 5 * x - 3) * (3 * x^3 + 2) =
  36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3578_357851


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_384_l3578_357812

/-- Represents the number of executives -/
def num_executives : ℕ := 5

/-- Represents the total number of people (executives + partners) -/
def total_people : ℕ := 2 * num_executives

/-- Calculates the number of distinct seating arrangements -/
def seating_arrangements : ℕ :=
  (List.range num_executives).foldl (λ acc i => acc * (total_people - 2 * i)) 1 / total_people

/-- Theorem stating that the number of distinct seating arrangements is 384 -/
theorem seating_arrangements_eq_384 : seating_arrangements = 384 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_384_l3578_357812


namespace NUMINAMATH_CALUDE_divisibility_of_20_pow_15_minus_1_l3578_357883

theorem divisibility_of_20_pow_15_minus_1 :
  (11 : ℕ) * 31 * 61 ∣ 20^15 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_20_pow_15_minus_1_l3578_357883


namespace NUMINAMATH_CALUDE_sin_product_equals_neg_two_fifths_l3578_357802

theorem sin_product_equals_neg_two_fifths (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_neg_two_fifths_l3578_357802


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_f_geq_3_l3578_357890

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Theorem 1
theorem min_value_when_a_is_one :
  ∀ x ∈ Set.Ioo 0 (Real.exp 1), f 1 x ≥ f 1 1 ∧ f 1 1 = 1 := by sorry

-- Theorem 2
theorem range_of_a_when_f_geq_3 :
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f a x ≥ 3) → a ≥ Real.exp 2 := by sorry

end

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_f_geq_3_l3578_357890


namespace NUMINAMATH_CALUDE_onion_basket_change_l3578_357857

theorem onion_basket_change (initial : ℝ) : 
  let added_by_sara := 4.5
  let removed_by_sally := 5.25
  let added_by_fred := 9.75
  (initial + added_by_sara - removed_by_sally + added_by_fred) - initial = 9 :=
by sorry

end NUMINAMATH_CALUDE_onion_basket_change_l3578_357857


namespace NUMINAMATH_CALUDE_binomial_distribution_p_value_l3578_357875

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_p_value 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 300)
  (h_var : variance X = 200) :
  X.p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_binomial_distribution_p_value_l3578_357875


namespace NUMINAMATH_CALUDE_g_value_l3578_357845

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_value (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2) (h4 : f 1 + g (-1) = 4) : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_value_l3578_357845


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3578_357807

theorem trigonometric_identity (α : Real) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan ((15 * α) / 2) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3578_357807


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l3578_357829

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l3578_357829


namespace NUMINAMATH_CALUDE_max_dots_on_surface_l3578_357838

/-- The sum of dots on a standard die -/
def standardDieSum : ℕ := 21

/-- The maximum number of dots visible on a die with 5 visible faces -/
def maxDotsOn5Faces : ℕ := 20

/-- The number of dots visible on a die with 4 visible faces -/
def dotsOn4Faces : ℕ := 14

/-- The maximum number of dots visible on a die with 2 visible faces -/
def maxDotsOn2Faces : ℕ := 11

/-- The number of dice with 5 visible faces -/
def numDice5Faces : ℕ := 6

/-- The number of dice with 4 visible faces -/
def numDice4Faces : ℕ := 5

/-- The number of dice with 2 visible faces -/
def numDice2Faces : ℕ := 2

theorem max_dots_on_surface :
  numDice5Faces * maxDotsOn5Faces +
  numDice4Faces * dotsOn4Faces +
  numDice2Faces * maxDotsOn2Faces = 212 :=
by sorry

end NUMINAMATH_CALUDE_max_dots_on_surface_l3578_357838


namespace NUMINAMATH_CALUDE_equidecomposable_transitivity_l3578_357867

-- Define the concept of a polygon
def Polygon : Type := sorry

-- Define the concept of equidecomposability between two polygons
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equidecomposable_transitivity (P Q R : Polygon) :
  equidecomposable P R → equidecomposable Q R → equidecomposable P Q := by
  sorry

end NUMINAMATH_CALUDE_equidecomposable_transitivity_l3578_357867


namespace NUMINAMATH_CALUDE_marble_probability_l3578_357865

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 150) (h2 : blue = 24) (h3 : red = 37) :
  let white := total - blue - red
  (red + white : ℚ) / total = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l3578_357865


namespace NUMINAMATH_CALUDE_computer_games_count_l3578_357828

def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def computer_game_price : ℕ := 90
def polo_shirt_count : ℕ := 3
def necklace_count : ℕ := 2
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

theorem computer_games_count :
  ∃ (n : ℕ), 
    n * computer_game_price + 
    polo_shirt_count * polo_shirt_price + 
    necklace_count * necklace_price - 
    rebate = total_cost_after_rebate ∧ 
    n = 1 := by sorry

end NUMINAMATH_CALUDE_computer_games_count_l3578_357828


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l3578_357862

/-- 
For a parabola with equation x^2 = ay and latus rectum y = 2, 
the value of a is -8.
-/
theorem parabola_latus_rectum (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- equation of parabola
  (∃ x : ℝ, x^2 = 2*a) →    -- latus rectum condition
  a = -8 := by
sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l3578_357862


namespace NUMINAMATH_CALUDE_tree_distance_l3578_357800

/-- Given 6 equally spaced trees along a straight road, where the distance between
    the first and fourth tree is 60 feet, the distance between the first and last
    tree is 100 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * d / 3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l3578_357800


namespace NUMINAMATH_CALUDE_a_minus_b_equals_seven_l3578_357842

theorem a_minus_b_equals_seven (a b : ℝ) 
  (ha : a^2 = 9)
  (hb : |b| = 4)
  (hgt : a > b) : 
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_seven_l3578_357842


namespace NUMINAMATH_CALUDE_sue_age_l3578_357888

theorem sue_age (total_age kate_age maggie_age sue_age : ℕ) : 
  total_age = 48 → kate_age = 19 → maggie_age = 17 → 
  total_age = kate_age + maggie_age + sue_age →
  sue_age = 12 := by
sorry

end NUMINAMATH_CALUDE_sue_age_l3578_357888


namespace NUMINAMATH_CALUDE_simplify_expression_find_expression_value_evaluate_expression_l3578_357894

-- Part 1
theorem simplify_expression (a b : ℝ) :
  10 * (a - b)^4 - 25 * (a - b)^4 + 5 * (a - b)^4 = -10 * (a - b)^4 := by sorry

-- Part 2
theorem find_expression_value (x y : ℝ) (h : 2 * x^2 - 3 * y = 8) :
  4 * x^2 - 6 * y - 32 = -16 := by sorry

-- Part 3
theorem evaluate_expression (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) (h2 : a * b - 2 * b^2 = -3) :
  3 * a^2 + 4 * a * b + 4 * b^2 = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_expression_value_evaluate_expression_l3578_357894


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3578_357835

/-- Given a line ax - 2by = 2 (where a > 0 and b > 0) passing through the center of the circle 
    x² + y² - 4x + 2y + 1 = 0, the minimum value of 4/(a+2) + 1/(b+1) is 9/4. -/
theorem min_value_of_expression (a b : ℝ) : a > 0 → b > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ a'*x' - 2*b'*y' = 2) → 
    4/(a+2) + 1/(b+1) ≤ 4/(a'+2) + 1/(b'+1)) → 
  4/(a+2) + 1/(b+1) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3578_357835


namespace NUMINAMATH_CALUDE_sinks_per_house_l3578_357818

/-- Given that a carpenter bought 266 sinks to cover 44 houses,
    prove that the number of sinks needed for each house is 6. -/
theorem sinks_per_house (total_sinks : ℕ) (num_houses : ℕ) 
  (h1 : total_sinks = 266) (h2 : num_houses = 44) :
  total_sinks / num_houses = 6 := by
  sorry

#check sinks_per_house

end NUMINAMATH_CALUDE_sinks_per_house_l3578_357818


namespace NUMINAMATH_CALUDE_q_polynomial_form_l3578_357823

def q (x : ℝ) : ℝ := sorry

theorem q_polynomial_form :
  ∀ x, q x + (2*x^6 + 5*x^4 + 10*x^2) = (9*x^4 + 30*x^3 + 40*x^2 + 5*x + 3) →
  q x = -2*x^6 + 4*x^4 + 30*x^3 + 30*x^2 + 5*x + 3 :=
by sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l3578_357823


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_optimization_l3578_357853

/-- Given a positive real number S, this theorem states that for any rectangle with area S and perimeter p,
    the expression S / (2S + p + 2) is maximized when the rectangle is a square, 
    and the maximum value is S / (2(√S + 1)²). -/
theorem rectangle_area_perimeter_optimization (S : ℝ) (hS : S > 0) :
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = S →
    S / (2 * S + 2 * (a + b) + 2) ≤ S / (2 * (Real.sqrt S + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_optimization_l3578_357853


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3578_357881

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3578_357881


namespace NUMINAMATH_CALUDE_volume_relationship_l3578_357879

/-- Given a right circular cone, cylinder, and sphere with specific properties, 
    prove the relationship between their volumes. -/
theorem volume_relationship (h r : ℝ) (A M C : ℝ) : 
  h > 0 → r > 0 →
  A = (1/3) * π * r^2 * h →
  M = π * r^2 * (2*h) →
  C = (4/3) * π * h^3 →
  A + M - C = π * h^3 := by
  sorry


end NUMINAMATH_CALUDE_volume_relationship_l3578_357879


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3578_357805

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 2) ^ 6) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3578_357805


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3578_357876

theorem arithmetic_calculations : 
  (8 / (-2) - (-4) * (-3) = 8) ∧ 
  ((-2)^3 / 4 * (5 - (-3)^2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3578_357876


namespace NUMINAMATH_CALUDE_jellybean_count_l3578_357814

/-- The number of jellybeans in a dozen -/
def jellybeans_per_dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * jellybeans_per_dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have together -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l3578_357814


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_prob_l3578_357861

/-- Represents a set of symmetrical dice -/
structure DiceSet where
  /-- The number of dice in the set -/
  num_dice : ℕ
  /-- The maximum number of points on each die -/
  max_points : ℕ
  /-- The probability of getting a sum of 2022 -/
  prob_2022 : ℝ
  /-- Assumption that the probability is positive -/
  pos_prob : prob_2022 > 0
  /-- Assumption that 2022 is achievable with these dice -/
  sum_2022 : num_dice * max_points = 2022

/-- 
Theorem: Given a set of symmetrical dice where a sum of 2022 is possible 
with probability p > 0, the smallest sum possible with the same probability p is 337.
-/
theorem smallest_sum_with_same_prob (d : DiceSet) : 
  d.num_dice = 337 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_prob_l3578_357861


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3578_357810

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def alternating_arrangements (n : ℕ) : ℕ := 2 * factorial n * factorial n

theorem johnson_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) :
  total_arrangements (boys + girls) - alternating_arrangements boys =
  39168 := by sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3578_357810


namespace NUMINAMATH_CALUDE_total_remaining_students_l3578_357830

def calculate_remaining_students (initial_a initial_b initial_c new_a new_b new_c transfer_rate_a transfer_rate_b transfer_rate_c : ℕ) : ℕ :=
  let total_a := initial_a + new_a
  let total_b := initial_b + new_b
  let total_c := initial_c + new_c
  let remaining_a := total_a - (total_a * transfer_rate_a / 100)
  let remaining_b := total_b - (total_b * transfer_rate_b / 100)
  let remaining_c := total_c - (total_c * transfer_rate_c / 100)
  remaining_a + remaining_b + remaining_c

theorem total_remaining_students :
  calculate_remaining_students 160 145 130 20 25 15 30 25 20 = 369 :=
by sorry

end NUMINAMATH_CALUDE_total_remaining_students_l3578_357830


namespace NUMINAMATH_CALUDE_senior_ticket_price_l3578_357870

theorem senior_ticket_price 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_senior_ticket_price_l3578_357870


namespace NUMINAMATH_CALUDE_line_passes_first_third_quadrants_iff_positive_slope_l3578_357811

/-- A line passes through the first and third quadrants if and only if its slope is positive -/
theorem line_passes_first_third_quadrants_iff_positive_slope (k : ℝ) :
  (k ≠ 0 ∧ ∀ x y : ℝ, y = k * x → (x > 0 → y > 0) ∧ (x < 0 → y < 0)) ↔ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_first_third_quadrants_iff_positive_slope_l3578_357811


namespace NUMINAMATH_CALUDE_son_age_proof_l3578_357885

theorem son_age_proof (father_age son_age : ℝ) : 
  father_age = son_age + 35 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 12.5 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l3578_357885
