import Mathlib

namespace NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l3402_340231

theorem floor_equality_iff_in_interval (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l3402_340231


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3402_340293

def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ 2/x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3402_340293


namespace NUMINAMATH_CALUDE_m_range_l3402_340232

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3402_340232


namespace NUMINAMATH_CALUDE_milk_for_nine_cookies_l3402_340233

-- Define the relationship between cookies and quarts of milk
def milk_for_cookies (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 18

-- Define the conversion from quarts to pints
def quarts_to_pints (quarts : ℚ) : ℚ :=
  2 * quarts

theorem milk_for_nine_cookies :
  quarts_to_pints (milk_for_cookies 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_nine_cookies_l3402_340233


namespace NUMINAMATH_CALUDE_total_jumps_in_3min_l3402_340210

/-- The number of jumps Jung-min can do in 4 minutes -/
def jung_min_jumps_4min : ℕ := 256

/-- The number of jumps Jimin can do in 3 minutes -/
def jimin_jumps_3min : ℕ := 111

/-- The duration we want to calculate the total jumps for (in minutes) -/
def duration : ℕ := 3

/-- Theorem stating that the sum of Jung-min's and Jimin's jumps in 3 minutes is 303 -/
theorem total_jumps_in_3min :
  (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min = 303 := by
  sorry

#eval (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min

end NUMINAMATH_CALUDE_total_jumps_in_3min_l3402_340210


namespace NUMINAMATH_CALUDE_coefficient_sum_equality_l3402_340236

theorem coefficient_sum_equality (n : ℕ) (h : n ≥ 5) :
  (Finset.range (n - 4)).sum (λ k => Nat.choose (k + 5) 5) = Nat.choose (n + 1) 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equality_l3402_340236


namespace NUMINAMATH_CALUDE_five_integers_average_l3402_340221

theorem five_integers_average (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₁ + a₅) + 
  (a₂ + a₃) + (a₂ + a₄) + (a₂ + a₅) + 
  (a₃ + a₄) + (a₃ + a₅) + 
  (a₄ + a₅) = 2020 →
  (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 101 := by
sorry

#eval (2020 : ℤ) / 4  -- To verify that 2020 / 4 = 505

end NUMINAMATH_CALUDE_five_integers_average_l3402_340221


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3402_340257

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : geometric_sequence a r)
  (h_roots : a 2 * a 6 = 64) :
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3402_340257


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l3402_340237

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a line passes through the third quadrant -/
def passesThroughThirdQuadrant (l : Line) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_not_in_third_quadrant 
  (a b : ℝ) 
  (h_first_quadrant : isInFirstQuadrant ⟨a*b, a+b⟩) :
  ¬passesThroughThirdQuadrant ⟨b, a, -a*b⟩ :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l3402_340237


namespace NUMINAMATH_CALUDE_bakers_cakes_l3402_340296

theorem bakers_cakes (initial_cakes : ℕ) : 
  initial_cakes - 105 + 170 = 186 → initial_cakes = 121 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3402_340296


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3402_340274

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem cross_number_puzzle : 
  ∃! d : ℕ, d < 10 ∧ 
  (∃ m : ℕ, is_three_digit (3^m) ∧ second_digit (3^m) = d) ∧
  (∃ n : ℕ, is_three_digit (7^n) ∧ second_digit (7^n) = d) ∧
  d = 4 :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3402_340274


namespace NUMINAMATH_CALUDE_pentagon_sum_edges_vertices_l3402_340250

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  edges : ℕ
  vertices : ℕ
  is_pentagon : edges = 5 ∧ vertices = 5

/-- The sum of edges and vertices in a pentagon is 10 -/
theorem pentagon_sum_edges_vertices (p : Pentagon) : p.edges + p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_edges_vertices_l3402_340250


namespace NUMINAMATH_CALUDE_cake_box_theorem_l3402_340228

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting cake boxes into a carton -/
structure CakeBoxProblem where
  carton : BoxDimensions
  cakeBox : BoxDimensions

/-- Calculates the maximum number of cake boxes that can fit in a carton -/
def maxCakeBoxes (p : CakeBoxProblem) : ℕ :=
  (boxVolume p.carton) / (boxVolume p.cakeBox)

/-- The main theorem stating the maximum number of cake boxes that can fit in the given carton -/
theorem cake_box_theorem (p : CakeBoxProblem) 
  (h_carton : p.carton = ⟨25, 42, 60⟩) 
  (h_cake_box : p.cakeBox = ⟨8, 7, 5⟩) : 
  maxCakeBoxes p = 225 := by
  sorry

#eval maxCakeBoxes ⟨⟨25, 42, 60⟩, ⟨8, 7, 5⟩⟩

end NUMINAMATH_CALUDE_cake_box_theorem_l3402_340228


namespace NUMINAMATH_CALUDE_simple_interest_rate_proof_l3402_340255

/-- Given a principal amount and a simple interest rate, 
    if the amount becomes 7/6 of itself after 4 years, 
    then the rate is 1/24 -/
theorem simple_interest_rate_proof 
  (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 4 * R) = 7/6 * P → R = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_proof_l3402_340255


namespace NUMINAMATH_CALUDE_bridge_length_l3402_340223

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3402_340223


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l3402_340206

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total fencing cost for a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 180 ∧
    plot.fencingCostPerMeter = 6.5 ∧
    totalFencingCost plot = 1170 := by
  sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l3402_340206


namespace NUMINAMATH_CALUDE_expression_evaluation_l3402_340202

theorem expression_evaluation : 
  |-2| + (1/3)⁻¹ - Real.sqrt 9 + (Real.sin (45 * π / 180) - 1)^0 - (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3402_340202


namespace NUMINAMATH_CALUDE_two_digit_cube_l3402_340265

theorem two_digit_cube (x : ℕ) : x = 93 ↔ 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 101010 * x + 1 = n^3) ∧
  (101010 * x + 1 ≥ 1000000 ∧ 101010 * x + 1 < 10000000) := by
sorry

end NUMINAMATH_CALUDE_two_digit_cube_l3402_340265


namespace NUMINAMATH_CALUDE_crickets_found_later_l3402_340282

theorem crickets_found_later (initial : ℝ) (final : ℕ) : initial = 7.0 → final = 18 → final - initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_crickets_found_later_l3402_340282


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3402_340270

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_same_remainder (a b n : ℕ) : Prop :=
  a % n = b % n

theorem greatest_common_divisor_with_digit_sum (a b : ℕ) :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ (a - b) ∧
  has_same_remainder a b n ∧
  sum_of_digits n = 4 ∧
  (∀ m : ℕ, m > n → m ∣ (a - b) → sum_of_digits m ≠ 4) →
  1120 ∣ n ∧ ∀ k : ℕ, k < 1120 → ¬(n ∣ k) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3402_340270


namespace NUMINAMATH_CALUDE_gordon_jamie_maine_coon_difference_total_cats_verification_l3402_340240

/-- The number of Persian cats Jamie owns -/
def jamie_persians : ℕ := 4

/-- The number of Maine Coons Jamie owns -/
def jamie_maine_coons : ℕ := 2

/-- The number of Persian cats Gordon owns -/
def gordon_persians : ℕ := jamie_persians / 2

/-- The total number of cats -/
def total_cats : ℕ := 13

/-- The number of Maine Coons Gordon owns -/
def gordon_maine_coons : ℕ := 3

theorem gordon_jamie_maine_coon_difference :
  gordon_maine_coons - jamie_maine_coons = 1 :=
by
  sorry

/-- The number of Maine Coons Hawkeye owns -/
def hawkeye_maine_coons : ℕ := gordon_maine_coons - 1

/-- Verification that the total number of cats is correct -/
theorem total_cats_verification :
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons +
  hawkeye_maine_coons = total_cats :=
by
  sorry

end NUMINAMATH_CALUDE_gordon_jamie_maine_coon_difference_total_cats_verification_l3402_340240


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3402_340204

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 9996 ∧ 
  n % 17 = 0 ∧ 
  n ≤ 9999 ∧ 
  ∀ m : ℕ, m % 17 = 0 ∧ m ≤ 9999 → m ≤ n := by
sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3402_340204


namespace NUMINAMATH_CALUDE_sin_squared_alpha_plus_pi_fourth_l3402_340216

theorem sin_squared_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.cos (2*α) = 4/5) : 
  Real.sin (α + π/4)^2 = 4/5 := by sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_plus_pi_fourth_l3402_340216


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3402_340262

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^3 / (2*i - 1)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3402_340262


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3402_340243

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3402_340243


namespace NUMINAMATH_CALUDE_complex_equations_solutions_l3402_340213

theorem complex_equations_solutions :
  let x₁ : ℚ := -7/5
  let y₁ : ℚ := 5
  let x₂ : ℚ := 5
  let y₂ : ℚ := -1
  (3 * y₁ : ℂ) + (5 * x₁ * I) = 15 - 7 * I ∧
  (2 * x₂ + 3 * y₂ : ℂ) + ((x₂ - y₂) * I) = 7 + 6 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_equations_solutions_l3402_340213


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_4_and_9_l3402_340241

theorem greatest_three_digit_divisible_by_4_and_9 : 
  ∃ n : ℕ, n = 972 ∧ 
  n < 1000 ∧ 
  n ≥ 100 ∧
  n % 4 = 0 ∧ 
  n % 9 = 0 ∧
  ∀ m : ℕ, m < 1000 → m ≥ 100 → m % 4 = 0 → m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_4_and_9_l3402_340241


namespace NUMINAMATH_CALUDE_product_equals_power_of_three_l3402_340201

theorem product_equals_power_of_three : 25 * 15 * 9 * 5.4 * 3.24 = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_power_of_three_l3402_340201


namespace NUMINAMATH_CALUDE_surface_area_of_specific_block_l3402_340218

/-- Represents a rectangular solid block made of unit cubes -/
structure RectangularBlock where
  length : Nat
  width : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the surface area of a rectangular block -/
def surface_area (block : RectangularBlock) : Nat :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- Theorem stating that the surface area of the specific block is 66 square units -/
theorem surface_area_of_specific_block :
  ∃ (block : RectangularBlock),
    block.length = 5 ∧
    block.width = 3 ∧
    block.height = 1 ∧
    block.total_cubes = 15 ∧
    surface_area block = 66 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_block_l3402_340218


namespace NUMINAMATH_CALUDE_factorial_20_19_div_5_is_perfect_square_l3402_340212

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_20_19_div_5_is_perfect_square :
  is_perfect_square ((factorial 20 * factorial 19) / 5) := by
  sorry

end NUMINAMATH_CALUDE_factorial_20_19_div_5_is_perfect_square_l3402_340212


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3402_340203

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3402_340203


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l3402_340260

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents the company's sales outlet distribution -/
structure CompanyDistribution where
  regions : List Region
  totalOutlets : Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : CompanyDistribution
  sampleSize : Nat
  hasDistinctSubgroups : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasDistinctSubgroups then
    SamplingMethod.StratifiedSampling
  else
    SamplingMethod.SimpleRandomSampling

/-- Theorem: Stratified sampling is more appropriate for populations with distinct subgroups -/
theorem stratified_sampling_appropriate_for_subgroups 
  (scenario : SamplingScenario) 
  (h : scenario.hasDistinctSubgroups = true) : 
  appropriateSamplingMethod scenario = SamplingMethod.StratifiedSampling :=
sorry

/-- Company distribution for the given problem -/
def companyDistribution : CompanyDistribution :=
  { regions := [
      { name := "A", outlets := 150 },
      { name := "B", outlets := 120 },
      { name := "C", outlets := 180 },
      { name := "D", outlets := 150 }
    ],
    totalOutlets := 600
  }

/-- Sampling scenario for investigation (1) -/
def investigation1 : SamplingScenario :=
  { population := companyDistribution,
    sampleSize := 100,
    hasDistinctSubgroups := true
  }

/-- Sampling scenario for investigation (2) -/
def investigation2 : SamplingScenario :=
  { population := { regions := [{ name := "C_large", outlets := 20 }], totalOutlets := 20 },
    sampleSize := 7,
    hasDistinctSubgroups := false
  }

/-- Theorem: The appropriate sampling method for investigation (1) is Stratified Sampling -/
theorem investigation1_uses_stratified_sampling :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling :=
sorry

/-- Theorem: The appropriate sampling method for investigation (2) is Simple Random Sampling -/
theorem investigation2_uses_simple_random_sampling :
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l3402_340260


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3402_340295

theorem polynomial_division_theorem (x : ℝ) : 
  (9*x^3 + 32*x^2 + 89*x + 271)*(x - 3) + 801 = 9*x^4 + 5*x^3 - 7*x^2 + 4*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3402_340295


namespace NUMINAMATH_CALUDE_quilt_transformation_l3402_340211

theorem quilt_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
  sorry

end NUMINAMATH_CALUDE_quilt_transformation_l3402_340211


namespace NUMINAMATH_CALUDE_fraction_problem_l3402_340217

theorem fraction_problem :
  ∃ (n d : ℚ), n + d = 5.25 ∧ (n + 3) / (2 * d) = 1/3 ∧ n/d = 2/33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3402_340217


namespace NUMINAMATH_CALUDE_mcq_options_count_l3402_340230

theorem mcq_options_count (p_all_correct : ℚ) (p_tf_correct : ℚ) (n : ℕ) : 
  p_all_correct = 1 / 12 →
  p_tf_correct = 1 / 2 →
  (1 / n : ℚ) * p_tf_correct * p_tf_correct = p_all_correct →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mcq_options_count_l3402_340230


namespace NUMINAMATH_CALUDE_marts_income_percentage_l3402_340238

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.3)) :
  mart / juan = 0.78 := by
sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l3402_340238


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3402_340289

theorem book_arrangement_count :
  let n_math_books : ℕ := 4
  let n_english_books : ℕ := 6
  let all_math_books_together : Prop := True
  let all_english_books_together : Prop := True
  let specific_english_book_at_left : Prop := True
  let all_books_distinct : Prop := True
  (n_math_books.factorial * (n_english_books - 1).factorial) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3402_340289


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3402_340276

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies : number_of_pies 51 41 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3402_340276


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l3402_340226

theorem second_pipe_fill_time (pipe1_time pipe2_time outlet_time all_pipes_time : ℝ) 
  (h1 : pipe1_time = 18)
  (h2 : outlet_time = 45)
  (h3 : all_pipes_time = 0.08333333333333333)
  (h4 : 1 / pipe1_time + 1 / pipe2_time - 1 / outlet_time = 1 / all_pipes_time) :
  pipe2_time = 20 := by sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l3402_340226


namespace NUMINAMATH_CALUDE_x_and_y_negative_l3402_340278

theorem x_and_y_negative (x y : ℝ) 
  (h1 : 2 * x - 3 * y > x) 
  (h2 : x + 4 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_and_y_negative_l3402_340278


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3402_340258

/-- A quadratic equation x^2 + x + a = 0 has one positive root and one negative root -/
def has_one_positive_one_negative_root (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + x + a = 0 ∧ y^2 + y + a = 0

/-- The condition a < -1 is sufficient but not necessary for x^2 + x + a = 0 
    to have one positive and one negative root -/
theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a < -1 → has_one_positive_one_negative_root a) ∧
  (∃ a : ℝ, -1 ≤ a ∧ a < 0 ∧ has_one_positive_one_negative_root a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3402_340258


namespace NUMINAMATH_CALUDE_exists_determining_question_l3402_340287

/-- Represents the type of being (Human or Zombie) --/
inductive Being
| Human
| Zombie

/-- Represents a possible response to a question --/
inductive Response
| Bal
| Yes
| No

/-- Represents a question that can be asked --/
def Question := Being → Response

/-- A function that determines the type of being based on a response --/
def DetermineBeing := Response → Being

/-- Humans always tell the truth, zombies always lie --/
axiom truth_telling (q : Question) :
  ∀ (b : Being), 
    (b = Being.Human → q b = Response.Bal) ∧
    (b = Being.Zombie → q b ≠ Response.Bal)

/-- There exists a question that can determine the type of being --/
theorem exists_determining_question :
  ∃ (q : Question) (d : DetermineBeing),
    ∀ (b : Being), d (q b) = b :=
sorry

end NUMINAMATH_CALUDE_exists_determining_question_l3402_340287


namespace NUMINAMATH_CALUDE_percent_of_percent_l3402_340283

theorem percent_of_percent (a b : ℝ) (ha : a = 20) (hb : b = 25) :
  (a / 100) * (b / 100) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3402_340283


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l3402_340205

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, a^2 * x + 2 * y + 1 = 0 → x - a * y - 2 = 0 → 
   (a^2 * 1 + 2 * (-a) = 0)) → 
  (a = 2 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l3402_340205


namespace NUMINAMATH_CALUDE_log_equation_solution_l3402_340268

theorem log_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log 5
  ∀ x : ℝ, f (x^2 - 25*x) = 3 ↔ x = 5*(5 + 3*Real.sqrt 5) ∨ x = 5*(5 - 3*Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3402_340268


namespace NUMINAMATH_CALUDE_investor_share_calculation_l3402_340261

/-- Calculates the share of profit for an investor given the investments and time periods. -/
theorem investor_share_calculation
  (investment_a investment_b total_profit : ℚ)
  (time_a time_b total_time : ℕ)
  (h1 : investment_a = 150)
  (h2 : investment_b = 200)
  (h3 : total_profit = 100)
  (h4 : time_a = 12)
  (h5 : time_b = 6)
  (h6 : total_time = 12)
  : (investment_a * time_a) / ((investment_a * time_a) + (investment_b * time_b)) * total_profit = 60 := by
  sorry

#check investor_share_calculation

end NUMINAMATH_CALUDE_investor_share_calculation_l3402_340261


namespace NUMINAMATH_CALUDE_remainder_problem_l3402_340251

theorem remainder_problem (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3402_340251


namespace NUMINAMATH_CALUDE_max_tan_b_in_triangle_l3402_340235

/-- Given a triangle ABC with AB = 25 and BC = 15, the maximum value of tan B is 4/3 -/
theorem max_tan_b_in_triangle (A B C : ℝ × ℝ) :
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 25 →
  d B C = 15 →
  ∀ C' : ℝ × ℝ, d A C' ≥ d A C → d B C' = 15 →
  Real.tan (Real.arccos ((d A B)^2 + (d B C)^2 - (d A C)^2) / (2 * d A B * d B C)) ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_tan_b_in_triangle_l3402_340235


namespace NUMINAMATH_CALUDE_inequality_proof_l3402_340242

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3402_340242


namespace NUMINAMATH_CALUDE_parabola_vertex_l3402_340245

/-- The vertex of the parabola y = x^2 - 1 has coordinates (0, -1) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := fun x ↦ x^2 - 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = -1 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3402_340245


namespace NUMINAMATH_CALUDE_solution_to_equation_l3402_340224

theorem solution_to_equation : ∃ x y : ℤ, 5 * x + 4 * y = 14 ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3402_340224


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_range_l3402_340259

theorem quadratic_inequality_and_range (a b : ℝ) (k : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_range_l3402_340259


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3402_340269

theorem angle_measure_in_triangle (P Q R : ℝ) 
  (h1 : P = 75)
  (h2 : Q = 2 * R - 15)
  (h3 : P + Q + R = 180) :
  R = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3402_340269


namespace NUMINAMATH_CALUDE_oscar_review_questions_l3402_340208

/-- The number of questions Professor Oscar must review -/
def total_questions (num_classes : ℕ) (students_per_class : ℕ) (questions_per_exam : ℕ) : ℕ :=
  num_classes * students_per_class * questions_per_exam

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 5 35 10 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_oscar_review_questions_l3402_340208


namespace NUMINAMATH_CALUDE_set_operations_l3402_340263

def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x ≤ 1}) ∧
  (B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3402_340263


namespace NUMINAMATH_CALUDE_domain_of_composed_function_l3402_340280

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ Set.Icc 0 3, f (x + 1) ∈ Set.range f) →
  {x : ℝ | f (2^x) ∈ Set.range f} = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_l3402_340280


namespace NUMINAMATH_CALUDE_jean_carter_books_l3402_340256

/-- Prove that given 12 total volumes, paperback price of $18, hardcover price of $30, 
    and total spent of $312, the number of hardcover volumes bought is 8. -/
theorem jean_carter_books 
  (total_volumes : ℕ) 
  (paperback_price hardcover_price : ℚ) 
  (total_spent : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 18)
  (hh : hardcover_price = 30)
  (hs : total_spent = 312) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_price + (total_volumes - hardcover_count) * paperback_price = total_spent ∧ 
    hardcover_count = 8 :=
by sorry

end NUMINAMATH_CALUDE_jean_carter_books_l3402_340256


namespace NUMINAMATH_CALUDE_computer_off_time_l3402_340246

/-- Represents days of the week -/
inductive Day
  | Friday
  | Saturday

/-- Represents time of day in hours (0-23) -/
def Time := Fin 24

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

/-- Adds hours to a given moment, wrapping to the next day if necessary -/
def addHours (m : Moment) (h : Nat) : Moment :=
  let totalHours := m.time.val + h
  let newDay := if totalHours ≥ 24 then Day.Saturday else m.day
  let newTime := Fin.ofNat (totalHours % 24)
  { day := newDay, time := newTime }

theorem computer_off_time 
  (start : Moment) 
  (h : Nat) 
  (start_day : start.day = Day.Friday)
  (start_time : start.time = ⟨14, sorry⟩)
  (duration : h = 30) :
  addHours start h = { day := Day.Saturday, time := ⟨20, sorry⟩ } := by
  sorry

#check computer_off_time

end NUMINAMATH_CALUDE_computer_off_time_l3402_340246


namespace NUMINAMATH_CALUDE_cube_split_2017_l3402_340267

theorem cube_split_2017 (m : ℕ) (h1 : m > 1) : 
  (m^3 = (m - 1)*(m^2 + m + 1) + (m - 1)^2 + (m - 1)^2 + 1) → 
  ((m - 1)*(m^2 + m + 1) = 2017 ∨ (m - 1)^2 = 2017 ∨ (m - 1)^2 + 2 = 2017) → 
  m = 46 := by
sorry

end NUMINAMATH_CALUDE_cube_split_2017_l3402_340267


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3402_340239

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3402_340239


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l3402_340214

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value 
  (x y : ℝ) -- The coordinates of the intersection point
  (h1 : y = 3 * x + 6) -- First line equation
  (h2 : y = -4 * x - 20) -- Second line equation
  (h3 : y = 2 * x + k) -- Third line equation
  : k = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l3402_340214


namespace NUMINAMATH_CALUDE_kays_aerobics_time_l3402_340227

/-- Given Kay's weekly exercise routine, calculate the time spent on aerobics. -/
theorem kays_aerobics_time (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  (aerobics_ratio * total_time) / (aerobics_ratio + weight_ratio) = 150 := by
  sorry

#check kays_aerobics_time

end NUMINAMATH_CALUDE_kays_aerobics_time_l3402_340227


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3402_340252

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3402_340252


namespace NUMINAMATH_CALUDE_machine_probabilities_theorem_l3402_340222

/-- Machine processing probabilities -/
structure MachineProbabilities where
  A : ℝ  -- Probability for machine A
  B : ℝ  -- Probability for machine B
  C : ℝ  -- Probability for machine C

/-- Given conditions -/
def conditions (p : MachineProbabilities) : Prop :=
  p.A * (1 - p.B) = 1/4 ∧
  p.B * (1 - p.C) = 1/12 ∧
  p.A * p.C = 2/9

/-- Theorem statement -/
theorem machine_probabilities_theorem (p : MachineProbabilities) 
  (h : conditions p) :
  p.A = 1/3 ∧ p.B = 1/4 ∧ p.C = 2/3 ∧
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_machine_probabilities_theorem_l3402_340222


namespace NUMINAMATH_CALUDE_delta_4_zero_delta_3_nonzero_l3402_340275

def u (n : ℕ) : ℤ := n^3 + n

def delta_1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | 1 => delta_1 u
  | k + 1 => delta_1 (delta k u)

theorem delta_4_zero_delta_3_nonzero :
  (∀ n, delta 4 u n = 0) ∧ (∃ n, delta 3 u n ≠ 0) := by sorry

end NUMINAMATH_CALUDE_delta_4_zero_delta_3_nonzero_l3402_340275


namespace NUMINAMATH_CALUDE_fraction_simplification_l3402_340253

theorem fraction_simplification : 
  ((2^1004)^2 - (2^1002)^2) / ((2^1003)^2 - (2^1001)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3402_340253


namespace NUMINAMATH_CALUDE_no_valid_n_for_ap_l3402_340292

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  180 % n = 0 ∧ 
  ∃ (k : ℕ), k^2 = (180 / n : ℚ) - (3/2 : ℚ) * n + (3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_for_ap_l3402_340292


namespace NUMINAMATH_CALUDE_chord_midpoint_theorem_l3402_340234

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Theorem statement
theorem chord_midpoint_theorem :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →
  (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2 →
  ∀ (x y : ℝ), chord_equation x y ↔ ∃ t : ℝ, (x, y) = (1 - t, 1 + t/4) :=
sorry

end NUMINAMATH_CALUDE_chord_midpoint_theorem_l3402_340234


namespace NUMINAMATH_CALUDE_max_valid_coloring_size_l3402_340284

/-- A type representing the color of a square on the board -/
inductive Color
| Black
| White

/-- A function type representing a coloring of an n × n board -/
def BoardColoring (n : ℕ) := Fin n → Fin n → Color

/-- Predicate to check if a board coloring satisfies the condition -/
def ValidColoring (n : ℕ) (coloring : BoardColoring n) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 → c1 ≠ c2 → 
    (coloring r1 c1 = coloring r1 c2 → coloring r2 c1 ≠ coloring r2 c2) ∧
    (coloring r1 c1 = coloring r2 c1 → coloring r1 c2 ≠ coloring r2 c2)

/-- Theorem stating that 4 is the maximum value of n for which a valid coloring exists -/
theorem max_valid_coloring_size :
  (∃ (coloring : BoardColoring 4), ValidColoring 4 coloring) ∧
  (∀ n : ℕ, n > 4 → ¬∃ (coloring : BoardColoring n), ValidColoring n coloring) :=
sorry

end NUMINAMATH_CALUDE_max_valid_coloring_size_l3402_340284


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_m_eq_one_l3402_340244

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line x - y = 0 -/
def slope1 : ℝ := 1

/-- The slope of the line x + my = 0 -/
def slope2 (m : ℝ) : ℝ := -m

/-- Theorem: The lines x - y = 0 and x + my = 0 are perpendicular if and only if m = 1 -/
theorem lines_perpendicular_iff_m_eq_one (m : ℝ) :
  perpendicular slope1 (slope2 m) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_m_eq_one_l3402_340244


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l3402_340290

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with given dimensions, 
    the radius of its circumscribed circle is 10.625 -/
theorem circumscribed_circle_radius_of_specific_trapezoid : 
  let t : IsoscelesTrapezoid := { base1 := 9, base2 := 21, height := 8 }
  circumscribedCircleRadius t = 10.625 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l3402_340290


namespace NUMINAMATH_CALUDE_score_well_defined_and_nonnegative_l3402_340220

theorem score_well_defined_and_nonnegative (N C : ℕ+) 
  (h1 : N ≤ 20) (h2 : C ≥ 1) : 
  ∃ (score : ℕ), score = ⌊(N : ℝ) / (C : ℝ)⌋ ∧ score ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_score_well_defined_and_nonnegative_l3402_340220


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3402_340281

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 2) : 
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 27/8 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 2 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 27/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3402_340281


namespace NUMINAMATH_CALUDE_tulips_to_add_l3402_340277

def tulip_to_daisy_ratio : ℚ := 3 / 4
def initial_daisies : ℕ := 32
def added_daisies : ℕ := 24

theorem tulips_to_add (tulips_added : ℕ) : 
  (tulip_to_daisy_ratio * (initial_daisies + added_daisies : ℚ)).num = 
  (tulip_to_daisy_ratio * initial_daisies).num + tulips_added → 
  tulips_added = 18 :=
by sorry

end NUMINAMATH_CALUDE_tulips_to_add_l3402_340277


namespace NUMINAMATH_CALUDE_factorization_equality_l3402_340279

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3402_340279


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3402_340209

theorem coin_flip_probability (p_tails : ℝ) (p_sequence : ℝ) : 
  p_tails = 1/2 → 
  p_sequence = 0.0625 →
  (1 - p_tails) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3402_340209


namespace NUMINAMATH_CALUDE_operation_state_theorem_l3402_340299

/-- Represents the state of a student operating a computer -/
def OperationState := Fin 5 → Fin 5 → Bool

/-- The given condition that the product of diagonal elements is 0 -/
def DiagonalProductZero (a : OperationState) : Prop :=
  (a 0 0) && (a 1 1) && (a 2 2) && (a 3 3) && (a 4 4) = false

/-- At least one student is not operating their own computer -/
def AtLeastOneNotOwnComputer (a : OperationState) : Prop :=
  ∃ i : Fin 5, a i i = false

/-- 
If the product of diagonal elements in the operation state matrix is 0,
then at least one student is not operating their own computer.
-/
theorem operation_state_theorem (a : OperationState) :
  DiagonalProductZero a → AtLeastOneNotOwnComputer a := by
  sorry

end NUMINAMATH_CALUDE_operation_state_theorem_l3402_340299


namespace NUMINAMATH_CALUDE_frog_max_hop_sum_l3402_340229

/-- The maximum sum of hop lengths for a frog hopping on integers -/
theorem frog_max_hop_sum (n : ℕ+) : 
  ∃ (S : ℕ), S = (4^n.val - 1) / 3 ∧ 
  ∀ (hop_lengths : List ℕ), 
    (∀ l ∈ hop_lengths, ∃ k : ℕ, l = 2^k) →
    (∀ p ∈ List.range (2^n.val), List.count p (List.scanl (λ acc x => (acc + x) % (2^n.val)) 0 hop_lengths) ≤ 1) →
    List.sum hop_lengths ≤ S :=
sorry

end NUMINAMATH_CALUDE_frog_max_hop_sum_l3402_340229


namespace NUMINAMATH_CALUDE_negative_three_squared_l3402_340288

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l3402_340288


namespace NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3402_340271

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3402_340271


namespace NUMINAMATH_CALUDE_percentage_problem_l3402_340254

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * N = (3 / 5) * N - 10 → 
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3402_340254


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3402_340272

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1101

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1101

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- The theorem stating that the absolute difference between P_s and P_d is 1/2201 -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 2201 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3402_340272


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l3402_340286

theorem triangle_angle_difference (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b = a * Real.tan B →
  A > π / 2 →
  A - B = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l3402_340286


namespace NUMINAMATH_CALUDE_max_digit_sum_for_special_fraction_l3402_340219

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc where a, b, c are digits -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem max_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ+),
    DecimalABC a b c = (1 : ℚ) / y ∧
    y ≤ 12 ∧
    ∀ (a' b' c' : Digit) (y' : ℕ+),
      DecimalABC a' b' c' = (1 : ℚ) / y' →
      y' ≤ 12 →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 8 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_for_special_fraction_l3402_340219


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l3402_340273

def point_to_x_axis_distance (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_P_to_x_axis :
  let P : ℝ × ℝ := (-3, 2)
  point_to_x_axis_distance P = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l3402_340273


namespace NUMINAMATH_CALUDE_sum_greater_than_double_l3402_340266

theorem sum_greater_than_double (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2*b := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_double_l3402_340266


namespace NUMINAMATH_CALUDE_lip_gloss_coverage_l3402_340291

theorem lip_gloss_coverage 
  (num_tubs : ℕ) 
  (tubes_per_tub : ℕ) 
  (total_people : ℕ) 
  (h1 : num_tubs = 6) 
  (h2 : tubes_per_tub = 2) 
  (h3 : total_people = 36) : 
  total_people / (num_tubs * tubes_per_tub) = 3 := by
sorry

end NUMINAMATH_CALUDE_lip_gloss_coverage_l3402_340291


namespace NUMINAMATH_CALUDE_tangent_line_through_origin_l3402_340294

/-- The tangent line to y = e^x passing through the origin -/
theorem tangent_line_through_origin :
  ∃! (a b : ℝ), 
    b = Real.exp a ∧ 
    0 = b - (Real.exp a) * a ∧ 
    a = 1 ∧ 
    b = Real.exp 1 ∧
    Real.exp a = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_through_origin_l3402_340294


namespace NUMINAMATH_CALUDE_simple_interest_difference_l3402_340200

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal rate time : ℕ) 
  (h_principal : principal = 2800)
  (h_rate : rate = 4)
  (h_time : time = 5) : 
  principal - (principal * rate * time) / 100 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l3402_340200


namespace NUMINAMATH_CALUDE_family_size_theorem_l3402_340298

def family_size_problem (fathers_side : ℕ) (total : ℕ) : Prop :=
  let mothers_side := total - fathers_side
  let difference := mothers_side - fathers_side
  let percentage := (difference : ℚ) / fathers_side * 100
  fathers_side = 10 ∧ total = 23 → percentage = 30

theorem family_size_theorem :
  family_size_problem 10 23 := by
  sorry

end NUMINAMATH_CALUDE_family_size_theorem_l3402_340298


namespace NUMINAMATH_CALUDE_equation_equivalence_l3402_340285

theorem equation_equivalence (x : ℝ) :
  (x - 2)^5 + (x - 6)^5 = 32 →
  let z := x - 4
  z^5 + 40*z^3 + 80*z - 32 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3402_340285


namespace NUMINAMATH_CALUDE_not_divisible_by_59_l3402_340297

theorem not_divisible_by_59 (x y : ℕ) 
  (hx : ¬ 59 ∣ x) 
  (hy : ¬ 59 ∣ y) 
  (h_sum : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_59_l3402_340297


namespace NUMINAMATH_CALUDE_garden_playground_area_equality_l3402_340215

theorem garden_playground_area_equality (garden_width garden_length playground_width : ℝ) :
  garden_width = 8 →
  2 * (garden_width + garden_length) = 64 →
  garden_width * garden_length = 16 * playground_width :=
by
  sorry

end NUMINAMATH_CALUDE_garden_playground_area_equality_l3402_340215


namespace NUMINAMATH_CALUDE_domain_of_sqrt_2cos_minus_1_l3402_340247

/-- The domain of f(x) = √(2cos(x) - 1) -/
theorem domain_of_sqrt_2cos_minus_1 (x : ℝ) : 
  (∃ f : ℝ → ℝ, f x = Real.sqrt (2 * Real.cos x - 1)) ↔ 
  (∃ k : ℤ, 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_2cos_minus_1_l3402_340247


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3402_340248

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (k : ℝ), m^2 + m - 2 + (m^2 - 1) * I = k * I) → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3402_340248


namespace NUMINAMATH_CALUDE_expression_evaluation_l3402_340225

theorem expression_evaluation (x y : ℝ) 
  (h : |x + 1/2| + (y - 1)^2 = 0) : 
  5 * x^2 * y - (6 * x * y - 2 * (x * y - 2 * x^2 * y) - x * y^2) + 4 * x * y = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3402_340225


namespace NUMINAMATH_CALUDE_peanuts_added_l3402_340264

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 12) : 
  final_peanuts - initial_peanuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l3402_340264


namespace NUMINAMATH_CALUDE_min_height_of_box_l3402_340207

/-- Represents a rectangular box with square bases -/
structure Box where
  base : ℝ  -- side length of the square base
  height : ℝ -- height of the box
  h_positive : 0 < height
  b_positive : 0 < base

/-- The surface area of a box -/
def surface_area (box : Box) : ℝ :=
  2 * box.base^2 + 4 * box.base * box.height

/-- The constraint that the height is 5 units greater than the base -/
def height_constraint (box : Box) : Prop :=
  box.height = box.base + 5

theorem min_height_of_box (box : Box) 
  (h_constraint : height_constraint box)
  (h_surface_area : surface_area box ≥ 150) :
  box.height ≥ 10 ∧ ∃ (b : Box), height_constraint b ∧ surface_area b ≥ 150 ∧ b.height = 10 :=
sorry

end NUMINAMATH_CALUDE_min_height_of_box_l3402_340207


namespace NUMINAMATH_CALUDE_shelter_dogs_l3402_340249

theorem shelter_dogs (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 16) = 15 / 11 →
  dogs = 60 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l3402_340249
