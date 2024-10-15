import Mathlib

namespace NUMINAMATH_CALUDE_f_of_5_eq_2515_l2662_266257

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 27*x^3 - 20*x^2 - 72*x + 40

/-- Theorem: f(5) equals 2515 -/
theorem f_of_5_eq_2515 : f 5 = 2515 := by sorry

end NUMINAMATH_CALUDE_f_of_5_eq_2515_l2662_266257


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2662_266255

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles on a 9x9 chessboard with 10 horizontal and 10 vertical lines -/
theorem square_to_rectangle_ratio : 
  (num_squares : ℚ) / num_rectangles = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2662_266255


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2662_266281

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2662_266281


namespace NUMINAMATH_CALUDE_train_speed_l2662_266230

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 750)
  (h3 : crossing_time = 80) :
  (train_length + bridge_length) / crossing_time = 12.375 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2662_266230


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2662_266295

theorem power_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2662_266295


namespace NUMINAMATH_CALUDE_haleys_marbles_l2662_266206

theorem haleys_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (num_boys : ℕ) 
  (h1 : total_marbles = 28)
  (h2 : marbles_per_boy = 2)
  (h3 : total_marbles = num_boys * marbles_per_boy) :
  num_boys = 14 := by
  sorry

end NUMINAMATH_CALUDE_haleys_marbles_l2662_266206


namespace NUMINAMATH_CALUDE_jim_distance_l2662_266283

/-- Represents the distance covered by a person in a certain number of steps -/
structure StepDistance where
  steps : ℕ
  distance : ℝ

/-- Carly's step distance -/
def carly_step : ℝ := 0.5

/-- The relationship between Carly's and Jim's steps for the same distance -/
def step_ratio : ℚ := 3 / 4

/-- Number of Jim's steps we want to calculate the distance for -/
def jim_steps : ℕ := 24

/-- Theorem stating that Jim travels 9 metres in 24 steps -/
theorem jim_distance : 
  ∀ (carly : StepDistance) (jim : StepDistance),
  carly.steps = 3 ∧ 
  jim.steps = 4 ∧
  carly.distance = jim.distance ∧
  carly.distance = carly_step * carly.steps →
  (jim_steps : ℝ) * jim.distance / jim.steps = 9 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_l2662_266283


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l2662_266242

theorem quadratic_root_implies_u (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 469) / 12)^2 + 25 * ((-25 - Real.sqrt 469) / 12) + u = 0) → 
  u = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l2662_266242


namespace NUMINAMATH_CALUDE_roses_to_tulips_ratio_l2662_266235

/-- Represents the number of flowers of each type in the shop -/
structure FlowerShop where
  carnations : ℕ
  violets : ℕ
  tulips : ℕ
  roses : ℕ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 4 ∧
  shop.carnations = 2 * (shop.carnations + shop.violets + shop.tulips + shop.roses) / 3

/-- Theorem stating that in a valid flower shop, the ratio of roses to tulips is 1:1 -/
theorem roses_to_tulips_ratio (shop : FlowerShop) (h : validFlowerShop shop) :
  shop.roses = shop.tulips := by
  sorry

#check roses_to_tulips_ratio

end NUMINAMATH_CALUDE_roses_to_tulips_ratio_l2662_266235


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2662_266261

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l2662_266261


namespace NUMINAMATH_CALUDE_triangle_side_length_l2662_266276

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if a = √5, c = 2, and cos(A) = 2/3, then b = 3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2662_266276


namespace NUMINAMATH_CALUDE_expansion_equality_l2662_266215

theorem expansion_equality (x : ℝ) : 24 * (x + 3) * (2 * x - 4) = 48 * x^2 + 48 * x - 288 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2662_266215


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2662_266240

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2662_266240


namespace NUMINAMATH_CALUDE_onions_sum_to_285_l2662_266219

/-- The total number of onions grown by Sara, Sally, Fred, Amy, and Matthew -/
def total_onions (sara sally fred amy matthew : ℕ) : ℕ :=
  sara + sally + fred + amy + matthew

/-- Theorem stating that the total number of onions grown is 285 -/
theorem onions_sum_to_285 :
  total_onions 40 55 90 25 75 = 285 := by
  sorry

end NUMINAMATH_CALUDE_onions_sum_to_285_l2662_266219


namespace NUMINAMATH_CALUDE_inequality_proof_l2662_266223

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2662_266223


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2662_266285

theorem sunzi_wood_measurement_problem (x y : ℝ) :
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔
  (x - y = 4.5 ∧ ∃ (z : ℝ), z = x/2 ∧ z + 1 = y ∧ x - (z + 1) = 4.5) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2662_266285


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2662_266296

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2662_266296


namespace NUMINAMATH_CALUDE_sequence_property_l2662_266217

/-- Given a sequence where the nth term is of the form 32000+n + m/n = (2000+n) 3(m/n),
    prove that when n = 2016, (n³)/(n²) = 2016 -/
theorem sequence_property : 
  ∀ n : ℕ, n = 2016 → (n^3 : ℚ) / (n^2 : ℚ) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2662_266217


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l2662_266290

/-- Prove that the polar curve equation ρ = √2 cos(θ - π/4) is equivalent to the rectangular coordinate equation (x - 1/2)² + (y - 1/2)² = 1/2 -/
theorem polar_to_rectangular_equivalence (x y ρ θ : ℝ) :
  (ρ = Real.sqrt 2 * Real.cos (θ - π / 4)) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) →
  (x - 1 / 2) ^ 2 + (y - 1 / 2) ^ 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l2662_266290


namespace NUMINAMATH_CALUDE_square_cut_from_rectangle_l2662_266288

theorem square_cut_from_rectangle (a b x : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) (h4 : x ≤ min a b) :
  (2 * (a + b) + 2 * x = a * b) ∧ (a * b - x^2 = 2 * (a + b)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_from_rectangle_l2662_266288


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2662_266202

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) → 
  x ≥ (5 - Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2662_266202


namespace NUMINAMATH_CALUDE_polynomial_sum_l2662_266270

theorem polynomial_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 262.88 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2662_266270


namespace NUMINAMATH_CALUDE_cos_2theta_plus_pi_3_l2662_266244

theorem cos_2theta_plus_pi_3 (θ : Real) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.cos (2 * θ + π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_plus_pi_3_l2662_266244


namespace NUMINAMATH_CALUDE_largest_integer_with_gcd_18_6_l2662_266229

theorem largest_integer_with_gcd_18_6 :
  ∀ n : ℕ, n < 150 → n > 138 → Nat.gcd n 18 ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_gcd_18_6_l2662_266229


namespace NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_thirteen_l2662_266232

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- The minimum number of tries required to guarantee obtaining one blue and one yellow ball -/
def minTriesForBlueAndYellow (counts : BallCounts) : Nat :=
  counts.purple + counts.blue + 1

theorem min_tries_for_blue_and_yellow_is_thirteen :
  let counts : BallCounts := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow counts = 13 := by sorry

end NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_thirteen_l2662_266232


namespace NUMINAMATH_CALUDE_polynomial_roots_and_product_l2662_266238

/-- Given a polynomial p(x) = x³ + (3/2)(1-a)x² - 3ax + b where a and b are real numbers,
    and |p(x)| ≤ 1 for all x in [0, √3], prove that p(x) = 0 has three real roots
    and calculate a specific product of these roots. -/
theorem polynomial_roots_and_product (a b : ℝ) 
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → 
      |x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b| ≤ 1) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁^2 - 2 - x₂) * (x₂^2 - 2 - x₃) * (x₃^2 - 2 - x₁) = -9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_product_l2662_266238


namespace NUMINAMATH_CALUDE_toothpick_pattern_l2662_266289

/-- 
Given an arithmetic sequence where:
- The first term is 5
- The common difference is 4
Prove that the 250th term is 1001
-/
theorem toothpick_pattern (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) :
  n = 250 → a₁ = 5 → d = 4 → aₙ = a₁ + (n - 1) * d → aₙ = 1001 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l2662_266289


namespace NUMINAMATH_CALUDE_mary_towel_count_l2662_266227

/-- Proves that Mary has 4 towels given the conditions of the problem --/
theorem mary_towel_count :
  ∀ (mary_towel_count frances_towel_count : ℕ)
    (total_weight mary_towel_weight frances_towel_weight : ℚ),
  mary_towel_count = 4 * frances_towel_count →
  total_weight = 60 →
  frances_towel_weight = 128 / 16 →
  total_weight = mary_towel_weight + frances_towel_weight →
  mary_towel_weight = mary_towel_count * (frances_towel_weight / frances_towel_count) →
  mary_towel_count = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_mary_towel_count_l2662_266227


namespace NUMINAMATH_CALUDE_percent_of_sixty_l2662_266274

theorem percent_of_sixty : (25 : ℚ) / 100 * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_sixty_l2662_266274


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l2662_266225

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_ways :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) *
  (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l2662_266225


namespace NUMINAMATH_CALUDE_sum_of_squares_l2662_266253

theorem sum_of_squares (x y : ℝ) (h1 : x * (x + y) = 35) (h2 : y * (x + y) = 77) :
  (x + y)^2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2662_266253


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2662_266298

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 3 * x^3 - x^2 - 20 * x + 27
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2662_266298


namespace NUMINAMATH_CALUDE_max_value_abcd_l2662_266268

theorem max_value_abcd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (sum_eq_3 : a + b + c + d = 3) :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end NUMINAMATH_CALUDE_max_value_abcd_l2662_266268


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l2662_266218

theorem intersection_equality_implies_m_value (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l2662_266218


namespace NUMINAMATH_CALUDE_smallest_sum_of_bases_l2662_266297

theorem smallest_sum_of_bases : ∃ (a b : ℕ), 
  (a > 6 ∧ b > 6) ∧ 
  (6 * a + 2 = 2 * b + 6) ∧ 
  (∀ (a' b' : ℕ), (a' > 6 ∧ b' > 6) → (6 * a' + 2 = 2 * b' + 6) → a + b ≤ a' + b') ∧
  a + b = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_bases_l2662_266297


namespace NUMINAMATH_CALUDE_tank_emptying_time_l2662_266248

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFullness : ℚ
  pipeARatePerMinute : ℚ
  pipeBRatePerMinute : ℚ
  pipeCRatePerMinute : ℚ

/-- Calculates the time to empty or fill the tank given its properties -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFullness / (tank.pipeARatePerMinute + tank.pipeBRatePerMinute + tank.pipeCRatePerMinute)

/-- Theorem stating the time to empty the specific tank configuration -/
theorem tank_emptying_time :
  let tank : WaterTank := {
    initialFullness := 7/11,
    pipeARatePerMinute := 1/15,
    pipeBRatePerMinute := -1/8,
    pipeCRatePerMinute := 1/20
  }
  timeToEmptyOrFill tank = 840/11 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l2662_266248


namespace NUMINAMATH_CALUDE_disk_intersection_theorem_l2662_266211

-- Define a type for colors
inductive Color
  | Red
  | White
  | Green

-- Define a type for disks
structure Disk where
  color : Color
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two disks intersect
def intersect (d1 d2 : Disk) : Prop :=
  let (x1, y1) := d1.center
  let (x2, y2) := d2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 ≤ (d1.radius + d2.radius) ^ 2

-- Define a function to check if three disks have a common point
def commonPoint (d1 d2 d3 : Disk) : Prop :=
  ∃ (x y : ℝ), 
    (x - d1.center.1) ^ 2 + (y - d1.center.2) ^ 2 ≤ d1.radius ^ 2 ∧
    (x - d2.center.1) ^ 2 + (y - d2.center.2) ^ 2 ≤ d2.radius ^ 2 ∧
    (x - d3.center.1) ^ 2 + (y - d3.center.2) ^ 2 ≤ d3.radius ^ 2

-- State the theorem
theorem disk_intersection_theorem (disks : Finset Disk) :
  (disks.card = 6) →
  (∃ (r1 r2 w1 w2 g1 g2 : Disk), 
    r1 ∈ disks ∧ r2 ∈ disks ∧ w1 ∈ disks ∧ w2 ∈ disks ∧ g1 ∈ disks ∧ g2 ∈ disks ∧
    r1.color = Color.Red ∧ r2.color = Color.Red ∧
    w1.color = Color.White ∧ w2.color = Color.White ∧
    g1.color = Color.Green ∧ g2.color = Color.Green) →
  (∀ (r w g : Disk), r ∈ disks → w ∈ disks → g ∈ disks →
    r.color = Color.Red → w.color = Color.White → g.color = Color.Green →
    commonPoint r w g) →
  (∃ (c : Color), ∃ (d1 d2 : Disk), d1 ∈ disks ∧ d2 ∈ disks ∧
    d1.color = c ∧ d2.color = c ∧ intersect d1 d2) :=
by sorry

end NUMINAMATH_CALUDE_disk_intersection_theorem_l2662_266211


namespace NUMINAMATH_CALUDE_completing_square_result_l2662_266267

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2662_266267


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2662_266262

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.625)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.125 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2662_266262


namespace NUMINAMATH_CALUDE_composite_shape_area_l2662_266224

/-- The area of a rectangle -/
def rectangleArea (length width : ℕ) : ℕ := length * width

/-- The total area of the composite shape -/
def totalArea (a b c : ℕ × ℕ) : ℕ :=
  rectangleArea a.1 a.2 + rectangleArea b.1 b.2 + rectangleArea c.1 c.2

/-- Theorem stating that the total area of the given composite shape is 83 square units -/
theorem composite_shape_area :
  totalArea (8, 6) (5, 4) (3, 5) = 83 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l2662_266224


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2662_266212

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 / (4 * Real.pi * r₂^2) = 1 / 9) :
  (4 / 3) * Real.pi * r₁^3 / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2662_266212


namespace NUMINAMATH_CALUDE_quadratic_linear_common_solution_l2662_266236

theorem quadratic_linear_common_solution
  (a d : ℝ) (x₁ x₂ e : ℝ) 
  (ha : a ≠ 0)
  (hd : d ≠ 0)
  (hx : x₁ ≠ x₂)
  (h_common : d * x₁ + e = 0)
  (h_unique : ∃! x, a * (x - x₁) * (x - x₂) + d * x + e = 0) :
  a * (x₂ - x₁) = d := by
sorry

end NUMINAMATH_CALUDE_quadratic_linear_common_solution_l2662_266236


namespace NUMINAMATH_CALUDE_jack_remaining_gift_card_value_jack_gift_card_return_l2662_266239

/-- Calculates the remaining value of gift cards Jack can return after sending some to a scammer. -/
theorem jack_remaining_gift_card_value 
  (bb_count : ℕ) (bb_value : ℕ) (wm_count : ℕ) (wm_value : ℕ) 
  (bb_sent : ℕ) (wm_sent : ℕ) : ℕ :=
  let total_bb := bb_count * bb_value
  let total_wm := wm_count * wm_value
  let sent_bb := bb_sent * bb_value
  let sent_wm := wm_sent * wm_value
  let remaining_bb := total_bb - sent_bb
  let remaining_wm := total_wm - sent_wm
  remaining_bb + remaining_wm

/-- Proves that Jack can return gift cards worth $3900. -/
theorem jack_gift_card_return : 
  jack_remaining_gift_card_value 6 500 9 200 1 2 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_gift_card_value_jack_gift_card_return_l2662_266239


namespace NUMINAMATH_CALUDE_equation_solutions_l2662_266271

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 7

theorem equation_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 5 ∧ ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2662_266271


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l2662_266241

theorem modular_congruence_existence (a c : ℕ) (b : ℤ) :
  ∃ x : ℕ, (a ^ x + x : ℤ) ≡ b [ZMOD c] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l2662_266241


namespace NUMINAMATH_CALUDE_philosophers_more_numerous_than_mathematicians_l2662_266220

theorem philosophers_more_numerous_than_mathematicians 
  (x : ℕ) -- x represents the number of people who are both mathematicians and philosophers
  (h_positive : x > 0) -- assumption that at least one person belongs to either group
  : 9 * x > 7 * x := by
  sorry

end NUMINAMATH_CALUDE_philosophers_more_numerous_than_mathematicians_l2662_266220


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2662_266279

-- Define the total number of chocolate bars
def total_chocolate_bars : ℕ := 400

-- Define the number of small boxes
def num_small_boxes : ℕ := 16

-- Define the number of chocolate bars in each small box
def bars_per_small_box : ℕ := total_chocolate_bars / num_small_boxes

-- Theorem to prove
theorem chocolate_distribution :
  bars_per_small_box = 25 :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2662_266279


namespace NUMINAMATH_CALUDE_prob_at_least_one_prob_exactly_one_l2662_266249

/-- Probability of event A occurring -/
def probA : ℚ := 4/5

/-- Probability of event B occurring -/
def probB : ℚ := 3/5

/-- Probability of event C occurring -/
def probC : ℚ := 2/5

/-- Events A, B, and C are independent -/
axiom independence : True

/-- Probability of at least one event occurring -/
theorem prob_at_least_one : 
  1 - (1 - probA) * (1 - probB) * (1 - probC) = 119/125 := by sorry

/-- Probability of exactly one event occurring -/
theorem prob_exactly_one :
  probA * (1 - probB) * (1 - probC) + 
  (1 - probA) * probB * (1 - probC) + 
  (1 - probA) * (1 - probB) * probC = 37/125 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_prob_exactly_one_l2662_266249


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2662_266201

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x < -1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2662_266201


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2662_266263

/-- The number of ways to arrange 9 distinct objects in a row with specific conditions -/
def arrangement_count : ℕ := 2880

/-- The total number of objects -/
def total_objects : ℕ := 9

/-- The number of objects that must be at the ends -/
def end_objects : ℕ := 2

/-- The number of objects that must be adjacent -/
def adjacent_objects : ℕ := 2

/-- The number of remaining objects -/
def remaining_objects : ℕ := total_objects - end_objects - adjacent_objects

theorem arrangement_theorem :
  arrangement_count = 
    2 * -- ways to arrange end objects
    (remaining_objects + 1) * -- ways to place adjacent objects
    2 * -- ways to arrange adjacent objects
    remaining_objects! -- ways to arrange remaining objects
  := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2662_266263


namespace NUMINAMATH_CALUDE_exists_multiple_2020_with_sum_digits_multiple_2020_l2662_266214

/-- Given a natural number n, returns the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a natural number that is a multiple of 2020
    and has a sum of digits that is also a multiple of 2020 -/
theorem exists_multiple_2020_with_sum_digits_multiple_2020 :
  ∃ n : ℕ, 2020 ∣ n ∧ 2020 ∣ sumOfDigits n := by sorry

end NUMINAMATH_CALUDE_exists_multiple_2020_with_sum_digits_multiple_2020_l2662_266214


namespace NUMINAMATH_CALUDE_calculation_proof_l2662_266258

theorem calculation_proof : 2^2 - Real.tan (60 * π / 180) + |Real.sqrt 3 - 1| - (3 - Real.pi)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2662_266258


namespace NUMINAMATH_CALUDE_matthew_egg_rolls_l2662_266208

/-- Given the egg roll consumption of Alvin, Patrick, and Matthew, prove that Matthew ate 6 egg rolls. -/
theorem matthew_egg_rolls (alvin patrick matthew : ℕ) : 
  alvin = 4 →
  patrick = alvin / 2 →
  matthew = 3 * patrick →
  matthew = 6 := by
sorry

end NUMINAMATH_CALUDE_matthew_egg_rolls_l2662_266208


namespace NUMINAMATH_CALUDE_pencil_distribution_l2662_266266

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 891 →
  max_students = 81 →
  total_pens % max_students = 0 →
  total_pencils % max_students = 0 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2662_266266


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2662_266286

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2662_266286


namespace NUMINAMATH_CALUDE_gcd_123456_789012_l2662_266291

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_123456_789012_l2662_266291


namespace NUMINAMATH_CALUDE_number_of_trucks_l2662_266275

/-- The number of trucks used in transportation -/
def x : ℕ := sorry

/-- The total profit from Qingxi to Shenzhen in yuan -/
def total_profit : ℕ := 11560

/-- The profit per truck from Qingxi to Guangzhou in yuan -/
def profit_qingxi_guangzhou : ℕ := 480

/-- The initial profit per truck from Guangzhou to Shenzhen in yuan -/
def initial_profit_guangzhou_shenzhen : ℕ := 520

/-- The decrease in profit for each additional truck in yuan -/
def profit_decrease : ℕ := 20

/-- The profit from Guangzhou to Shenzhen as a function of the number of trucks -/
def profit_guangzhou_shenzhen (n : ℕ) : ℤ :=
  initial_profit_guangzhou_shenzhen * n - profit_decrease * (n - 1)

theorem number_of_trucks : x = 10 := by
  have h1 : profit_qingxi_guangzhou * x + profit_guangzhou_shenzhen x = total_profit := by sorry
  sorry

end NUMINAMATH_CALUDE_number_of_trucks_l2662_266275


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2662_266293

theorem line_inclination_angle (x y : ℝ) :
  x + y - Real.sqrt 3 = 0 → ∃ θ : ℝ, θ = 135 * π / 180 ∧ Real.tan θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2662_266293


namespace NUMINAMATH_CALUDE_gina_collected_two_bags_l2662_266246

/-- The number of bags Gina collected by herself -/
def gina_bags : ℕ := 2

/-- The number of bags collected by the rest of the neighborhood -/
def neighborhood_bags : ℕ := 82 * gina_bags

/-- The weight of each bag in pounds -/
def bag_weight : ℕ := 4

/-- The total weight of litter collected in pounds -/
def total_weight : ℕ := 664

/-- Theorem stating that Gina collected 2 bags of litter -/
theorem gina_collected_two_bags :
  gina_bags = 2 ∧
  neighborhood_bags = 82 * gina_bags ∧
  bag_weight = 4 ∧
  total_weight = 664 ∧
  total_weight = bag_weight * (gina_bags + neighborhood_bags) :=
by sorry

end NUMINAMATH_CALUDE_gina_collected_two_bags_l2662_266246


namespace NUMINAMATH_CALUDE_savings_equality_l2662_266254

/-- Proves that A's savings equal B's savings given the problem conditions --/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 3000)
  (h2 : a_salary = 2250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l2662_266254


namespace NUMINAMATH_CALUDE_pencil_length_theorem_l2662_266245

def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

theorem pencil_length_theorem (original_length sharpened_off : ℕ) 
  (h1 : original_length = 31)
  (h2 : sharpened_off = 17) :
  pencil_length_after_sharpening original_length sharpened_off = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_theorem_l2662_266245


namespace NUMINAMATH_CALUDE_remainder_of_five_n_mod_eleven_l2662_266272

theorem remainder_of_five_n_mod_eleven (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_five_n_mod_eleven_l2662_266272


namespace NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2662_266277

def probability_four_threes (n m k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / m) ^ k * ((m - 1) / m) ^ (n - k)

theorem probability_four_threes_eight_dice :
  probability_four_threes 8 6 4 = 43750 / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2662_266277


namespace NUMINAMATH_CALUDE_ab_value_l2662_266233

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + Real.sqrt (b + 3) = 0) : a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2662_266233


namespace NUMINAMATH_CALUDE_base9_726_to_base3_l2662_266252

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  ((d / 3), (d % 3))

/-- Converts a base-9 number to its base-3 representation -/
def base9ToBase3 (n : Nat) : List Nat :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (a, b) := base9ToBase3Digit d; [a, b]))

theorem base9_726_to_base3 :
  base9ToBase3 726 = [2, 1, 0, 2, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base9_726_to_base3_l2662_266252


namespace NUMINAMATH_CALUDE_difference_A_B_l2662_266204

def A : ℕ → ℕ
  | 0 => 41
  | n + 1 => (2*n + 1) * (2*n + 2) + A n

def B : ℕ → ℕ
  | 0 => 1
  | n + 1 => (2*n) * (2*n + 1) + B n

theorem difference_A_B : A 20 - B 20 = 380 := by
  sorry

end NUMINAMATH_CALUDE_difference_A_B_l2662_266204


namespace NUMINAMATH_CALUDE_current_velocity_l2662_266209

-- Define the rowing speeds
def downstream_speed (v c : ℝ) : ℝ := v + c
def upstream_speed (v c : ℝ) : ℝ := v - c

-- Define the conditions of the problem
def downstream_distance : ℝ := 32
def upstream_distance : ℝ := 14
def trip_time : ℝ := 6

-- Theorem statement
theorem current_velocity :
  ∃ (v c : ℝ),
    downstream_speed v c * trip_time = downstream_distance ∧
    upstream_speed v c * trip_time = upstream_distance ∧
    c = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_l2662_266209


namespace NUMINAMATH_CALUDE_original_average_age_is_40_l2662_266207

/-- Proves that the original average age of a class is 40 years given specific conditions. -/
theorem original_average_age_is_40 
  (N : ℕ) -- Original number of students
  (A : ℝ) -- Original average age
  (new_students : ℕ) -- Number of new students
  (new_age : ℝ) -- Average age of new students
  (age_decrease : ℝ) -- Decrease in average age after new students join
  (h1 : N = 2) -- Original number of students is 2
  (h2 : new_students = 2) -- 2 new students join
  (h3 : new_age = 32) -- Average age of new students is 32
  (h4 : age_decrease = 4) -- Average age decreases by 4
  (h5 : (A * N + new_age * new_students) / (N + new_students) = A - age_decrease) -- New average age equation
  : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_age_is_40_l2662_266207


namespace NUMINAMATH_CALUDE_product_remainder_ten_l2662_266234

theorem product_remainder_ten (a b c d : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 7) (hc : c % 10 = 5) (hd : d % 10 = 3) :
  (a * b * c * d) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_ten_l2662_266234


namespace NUMINAMATH_CALUDE_sum_angles_regular_star_5_l2662_266265

/-- A regular 5-pointed star inscribed in a circle -/
structure RegularStar5 where
  /-- The angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points in the star -/
  num_points : ℕ
  /-- The number of points is 5 -/
  h_num_points : num_points = 5

/-- The sum of angles at the tips of a regular 5-pointed star is 540° -/
theorem sum_angles_regular_star_5 (star : RegularStar5) : 
  star.num_points * star.tip_angle = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_angles_regular_star_5_l2662_266265


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_focal_length_specific_hyperbola_l2662_266226

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2c, where c² = a² + b² -/
theorem hyperbola_focal_length (a b c : ℝ) (h : a > 0) (h' : b > 0) :
  (a^2 = 10) → (b^2 = 2) → (c^2 = a^2 + b^2) →
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem focal_length_specific_hyperbola :
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧
  (a^2 = 10) ∧ (b^2 = 2) ∧ (c^2 = a^2 + b^2) ∧
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_focal_length_specific_hyperbola_l2662_266226


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l2662_266259

/-- Proves that the cost of each chocolate bar is $3 -/
theorem chocolate_bar_cost (initial_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℚ) : 
  initial_bars = 7 → unsold_bars = 4 → total_revenue = 9 → 
  (total_revenue / (initial_bars - unsold_bars : ℚ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l2662_266259


namespace NUMINAMATH_CALUDE_rosemary_leaves_count_rosemary_leaves_solution_l2662_266287

theorem rosemary_leaves_count : ℕ → Prop :=
  fun r : ℕ =>
    let basil_pots : ℕ := 3
    let rosemary_pots : ℕ := 9
    let thyme_pots : ℕ := 6
    let basil_leaves_per_plant : ℕ := 4
    let thyme_leaves_per_plant : ℕ := 30
    let total_leaves : ℕ := 354
    
    basil_pots * basil_leaves_per_plant + 
    rosemary_pots * r + 
    thyme_pots * thyme_leaves_per_plant = total_leaves →
    r = 18

theorem rosemary_leaves_solution : rosemary_leaves_count 18 := by
  sorry

end NUMINAMATH_CALUDE_rosemary_leaves_count_rosemary_leaves_solution_l2662_266287


namespace NUMINAMATH_CALUDE_lighting_power_increase_l2662_266237

/-- Proves that the increase in lighting power is 60 BT given the initial and final power values. -/
theorem lighting_power_increase (N_before N_after : ℝ) 
  (h1 : N_before = 240)
  (h2 : N_after = 300) :
  N_after - N_before = 60 := by
  sorry

end NUMINAMATH_CALUDE_lighting_power_increase_l2662_266237


namespace NUMINAMATH_CALUDE_car_speeds_problem_l2662_266213

/-- Proves that given the problem conditions, the speeds of the two cars are 60 km/h and 90 km/h -/
theorem car_speeds_problem (total_distance : ℝ) (meeting_distance : ℝ) (speed_difference : ℝ)
  (h1 : total_distance = 200)
  (h2 : meeting_distance = 80)
  (h3 : speed_difference = 30)
  (h4 : meeting_distance / speed_a = (total_distance - meeting_distance) / (speed_a + speed_difference))
  (speed_a : ℝ)
  (speed_b : ℝ)
  (h5 : speed_b = speed_a + speed_difference) :
  speed_a = 60 ∧ speed_b = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speeds_problem_l2662_266213


namespace NUMINAMATH_CALUDE_power_equation_l2662_266260

theorem power_equation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^n = 6) : a^(2*m - n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l2662_266260


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l2662_266250

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  short_side : ℝ
  long_side : ℝ
  perimeter : ℝ
  cost_per_foot : ℝ
  long_side_relation : long_side = 3 * short_side
  perimeter_equation : perimeter = 2 * short_side + 2 * long_side

/-- The cost to replace one short side of the fence -/
def replacement_cost (plot : RectangularPlot) : ℝ :=
  plot.short_side * plot.cost_per_foot

/-- Theorem stating the replacement cost for the given conditions -/
theorem replacement_cost_theorem (plot : RectangularPlot) 
  (h_perimeter : plot.perimeter = 640)
  (h_cost : plot.cost_per_foot = 5) :
  replacement_cost plot = 400 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_theorem_l2662_266250


namespace NUMINAMATH_CALUDE_watch_cost_price_l2662_266216

/-- Proves that the cost price of a watch is 280 Rs. given specific selling conditions -/
theorem watch_cost_price (selling_price : ℝ) : 
  (selling_price = 0.54 * 280) →  -- Sold at 46% loss
  (selling_price + 140 = 1.04 * 280) →  -- If sold for 140 more, 4% gain
  280 = 280 := by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2662_266216


namespace NUMINAMATH_CALUDE_georgia_has_24_students_l2662_266231

/-- Represents the number of students Georgia has, given her muffin-making habits. -/
def georgia_students : ℕ :=
  let batches : ℕ := 36
  let muffins_per_batch : ℕ := 6
  let months : ℕ := 9
  let total_muffins : ℕ := batches * muffins_per_batch
  total_muffins / months

/-- Proves that Georgia has 24 students based on her muffin-making habits. -/
theorem georgia_has_24_students : georgia_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_georgia_has_24_students_l2662_266231


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l2662_266221

/-- The inverse proportion function passing through (2, -1) has k = -2 --/
theorem inverse_proportion_through_point (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → k / x = -1 / 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l2662_266221


namespace NUMINAMATH_CALUDE_range_of_m_l2662_266284

theorem range_of_m (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 1) 
  (hab : a + b = 2) 
  (h_ineq : ∀ m, (4/a) + (1/(b-1)) > m^2 + 8*m) :
  -9 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2662_266284


namespace NUMINAMATH_CALUDE_factorization_proof_l2662_266210

theorem factorization_proof (a b c : ℝ) : 
  (a^2 + 2*b^2 - 2*c^2 + 3*a*b + a*c = (a + b - c)*(a + 2*b + 2*c)) ∧
  (a^2 - 2*b^2 - 2*c^2 - a*b + 5*b*c - a*c = (a - 2*b + c)*(a + b - 2*c)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2662_266210


namespace NUMINAMATH_CALUDE_sons_age_l2662_266269

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 18 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2662_266269


namespace NUMINAMATH_CALUDE_apple_sales_proof_l2662_266264

/-- The number of apples sold by Reginald --/
def apples_sold : ℕ := 20

/-- The price of each apple in dollars --/
def apple_price : ℚ := 1.25

/-- The cost of Reginald's bike in dollars --/
def bike_cost : ℚ := 80

/-- The fraction of the bike cost that the repairs cost --/
def repair_cost_fraction : ℚ := 1/4

/-- The fraction of earnings remaining after repairs --/
def remaining_earnings_fraction : ℚ := 1/5

theorem apple_sales_proof :
  apples_sold = 20 ∧
  apple_price = 1.25 ∧
  bike_cost = 80 ∧
  repair_cost_fraction = 1/4 ∧
  remaining_earnings_fraction = 1/5 ∧
  (apples_sold : ℚ) * apple_price - bike_cost * repair_cost_fraction = 
    remaining_earnings_fraction * ((apples_sold : ℚ) * apple_price) :=
by sorry

end NUMINAMATH_CALUDE_apple_sales_proof_l2662_266264


namespace NUMINAMATH_CALUDE_z_percent_of_x_l2662_266280

theorem z_percent_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end NUMINAMATH_CALUDE_z_percent_of_x_l2662_266280


namespace NUMINAMATH_CALUDE_house_transaction_loss_l2662_266200

/-- Proves that given a house initially valued at $12000, after selling it at a 15% loss
and buying it back at a 20% gain, the original owner loses $240. -/
theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h1 : initial_value = 12000)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.20) :
  initial_value - (initial_value * (1 - loss_percent) * (1 + gain_percent)) = -240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l2662_266200


namespace NUMINAMATH_CALUDE_largest_integer_with_three_digit_square_in_base_7_l2662_266205

theorem largest_integer_with_three_digit_square_in_base_7 :
  ∃ M : ℕ, 
    (∀ n : ℕ, n^2 ≥ 7^2 ∧ n^2 < 7^3 → n ≤ M) ∧
    M^2 ≥ 7^2 ∧ M^2 < 7^3 ∧
    M = 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_three_digit_square_in_base_7_l2662_266205


namespace NUMINAMATH_CALUDE_prob_rain_A_given_B_l2662_266251

/-- The probability of rain in city A given rain in city B -/
theorem prob_rain_A_given_B (pA pB pAB : ℝ) : 
  pA = 0.2 → pB = 0.18 → pAB = 0.12 → pAB / pB = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_A_given_B_l2662_266251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2662_266282

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) : 
  ∀ n : ℕ, n > 0 → a n = -2 * n + 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2662_266282


namespace NUMINAMATH_CALUDE_trig_identities_l2662_266243

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α - Real.pi/4) = 1/3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 80/37) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l2662_266243


namespace NUMINAMATH_CALUDE_alternate_angle_measure_l2662_266256

-- Define the angle measures as real numbers
def angle_A : ℝ := 0
def angle_B : ℝ := 0
def angle_C : ℝ := 0

-- State the theorem
theorem alternate_angle_measure :
  -- Conditions
  (angle_A = (1/4) * angle_B) →  -- ∠A is 1/4 of ∠B
  (angle_C = angle_A) →          -- ∠C and ∠A are alternate angles (due to parallel lines)
  (angle_B + angle_C = 180) →    -- ∠B and ∠C form a straight line
  -- Conclusion
  (angle_C = 36) := by
  sorry

end NUMINAMATH_CALUDE_alternate_angle_measure_l2662_266256


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2662_266273

theorem complex_number_modulus (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i * (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2662_266273


namespace NUMINAMATH_CALUDE_other_interest_rate_is_sixteen_percent_l2662_266247

/-- Proves that given the investment conditions, the other interest rate is 16% -/
theorem other_interest_rate_is_sixteen_percent
  (investment_difference : ℝ)
  (higher_rate_investment : ℝ)
  (higher_rate : ℝ)
  (h1 : investment_difference = 1260)
  (h2 : higher_rate_investment = 2520)
  (h3 : higher_rate = 0.08)
  (h4 : higher_rate_investment = (higher_rate_investment - investment_difference) + investment_difference)
  (h5 : higher_rate_investment * higher_rate = (higher_rate_investment - investment_difference) * (16 / 100)) :
  ∃ (other_rate : ℝ), other_rate = 16 / 100 :=
by
  sorry

#check other_interest_rate_is_sixteen_percent

end NUMINAMATH_CALUDE_other_interest_rate_is_sixteen_percent_l2662_266247


namespace NUMINAMATH_CALUDE_initial_trees_count_l2662_266292

/-- The number of walnut trees in the park after planting -/
def total_trees : ℕ := 55

/-- The number of walnut trees planted today -/
def planted_trees : ℕ := 33

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_trees_count : initial_trees = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_count_l2662_266292


namespace NUMINAMATH_CALUDE_fraction_simplification_l2662_266222

theorem fraction_simplification :
  let x : ℚ := 1/3
  1 / (1 / x^1 + 1 / x^2 + 1 / x^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2662_266222


namespace NUMINAMATH_CALUDE_equation_solution_l2662_266203

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 ∧ x₂ = (-1 - Real.sqrt 17) / 2) ∧
  (∀ x : ℝ, x^2 - |x - 1| - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2662_266203


namespace NUMINAMATH_CALUDE_max_pie_pieces_l2662_266278

theorem max_pie_pieces : 
  (∃ (n : ℕ), n > 0 ∧ 
    ∃ (A B : ℕ), 
      10000 ≤ A ∧ A < 100000 ∧ 
      10000 ≤ B ∧ B < 100000 ∧ 
      A = B * n ∧ 
      (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10)) ∧
    ∀ (m : ℕ), m > n → 
      ¬(∃ (C D : ℕ), 
        10000 ≤ C ∧ C < 100000 ∧ 
        10000 ≤ D ∧ D < 100000 ∧ 
        C = D * m ∧ 
        (∀ (i j : Fin 5), i ≠ j → (C / 10^i.val % 10) ≠ (C / 10^j.val % 10)))) ∧
  (∃ (A B : ℕ), 
    10000 ≤ A ∧ A < 100000 ∧ 
    10000 ≤ B ∧ B < 100000 ∧ 
    A = B * 7 ∧ 
    (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pie_pieces_l2662_266278


namespace NUMINAMATH_CALUDE_mangoes_per_kilogram_l2662_266294

theorem mangoes_per_kilogram (total_harvest : ℕ) (sold_to_market : ℕ) (remaining_mangoes : ℕ) :
  total_harvest = 60 →
  sold_to_market = 20 →
  remaining_mangoes = 160 →
  ∃ (sold_to_community : ℕ),
    sold_to_community = (total_harvest - sold_to_market) / 2 ∧
    remaining_mangoes = (total_harvest - sold_to_market - sold_to_community) * 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_kilogram_l2662_266294


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2662_266299

theorem consecutive_integers_sum (p q r s : ℤ) : 
  (q = p + 1 ∧ r = p + 2 ∧ s = p + 3) →  -- consecutive integers condition
  (p + s = 109) →                        -- given sum condition
  (q + r = 109) :=                       -- theorem to prove
by
  sorry


end NUMINAMATH_CALUDE_consecutive_integers_sum_l2662_266299


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2662_266228

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2662_266228
