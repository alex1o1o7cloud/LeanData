import Mathlib

namespace NUMINAMATH_CALUDE_largest_number_l270_27051

theorem largest_number : 
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  b > a ∧ b > c ∧ b > d := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l270_27051


namespace NUMINAMATH_CALUDE_chicken_feed_bag_weight_l270_27054

-- Define the constants from the problem
def chicken_price : ℚ := 3/2
def feed_bag_cost : ℚ := 2
def feed_per_chicken : ℚ := 2
def num_chickens : ℕ := 50
def total_profit : ℚ := 65

-- Define the theorem
theorem chicken_feed_bag_weight :
  ∃ (bag_weight : ℚ),
    bag_weight * (feed_bag_cost / feed_per_chicken) = num_chickens * chicken_price - total_profit ∧
    bag_weight > 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chicken_feed_bag_weight_l270_27054


namespace NUMINAMATH_CALUDE_range_of_f_l270_27073

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l270_27073


namespace NUMINAMATH_CALUDE_triple_transformation_to_zero_l270_27001

/-- Represents a transformation on a triple of integers -/
inductive Transform : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop
  | xy : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (min (2*x) (y-x), max (2*x) (y-x), z)
  | xz : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (min (2*x) (z-x), y, max (2*x) (z-x))
  | yz : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (x, min (2*y) (z-y), max (2*y) (z-y))

/-- Represents a sequence of transformations -/
def TransformSeq : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Relation.ReflTransGen Transform

/-- The main theorem to be proved -/
theorem triple_transformation_to_zero :
  ∀ x y z : ℕ, x ≤ y → y ≤ z → ∃ a b c : ℕ, TransformSeq (x, y, z) (a, b, c) ∧ (a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end NUMINAMATH_CALUDE_triple_transformation_to_zero_l270_27001


namespace NUMINAMATH_CALUDE_no_real_solutions_l270_27014

theorem no_real_solutions : 
  ¬∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l270_27014


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l270_27035

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l270_27035


namespace NUMINAMATH_CALUDE_train_crossing_time_l270_27009

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) :
  train_length = 300 →
  signal_pole_time = 24 →
  platform_length = 187.5 →
  (train_length + platform_length) / (train_length / signal_pole_time) = 39 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l270_27009


namespace NUMINAMATH_CALUDE_candy_bar_cost_l270_27070

/-- The cost of a candy bar given initial and final amounts --/
theorem candy_bar_cost (initial : ℕ) (final : ℕ) (h : initial = 4) (h' : final = 3) :
  initial - final = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l270_27070


namespace NUMINAMATH_CALUDE_f_pi_third_eq_half_l270_27082

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi / 2 + α)) / 
  (Real.cos (-Real.pi / 2 + α) * Real.tan (Real.pi + α))

theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_f_pi_third_eq_half_l270_27082


namespace NUMINAMATH_CALUDE_intersection_distance_l270_27020

theorem intersection_distance (n d k : ℝ) (h1 : d ≠ 0) (h2 : n * 0 + d = 3) :
  let f (x : ℝ) := x^2 + 4*x + 3
  let g (x : ℝ) := n*x + d
  let c := |f k - g k|
  (∃! k, f k = g k) → c = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_distance_l270_27020


namespace NUMINAMATH_CALUDE_quadratic_root_property_l270_27028

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ + m = 0) →
  (2 * x₂^2 + 4 * x₂ + m = 0) →
  (x₁^2 + x₂^2 + 2*x₁*x₂ - x₁^2*x₂^2 = 0) →
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l270_27028


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l270_27043

theorem infinitely_many_perfect_squares (k : ℕ+) :
  ∃ f : ℕ → ℕ+, Monotone f ∧ ∀ i : ℕ, ∃ m : ℕ+, (f i : ℕ) * 2^(k : ℕ) - 7 = m^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l270_27043


namespace NUMINAMATH_CALUDE_triangular_pyramid_distance_sum_l270_27095

/-- A triangular pyramid with volume V, face areas (S₁, S₂, S₃, S₄), and distances (H₁, H₂, H₃, H₄) from any internal point Q to each face. -/
structure TriangularPyramid where
  V : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  K : ℝ
  volume_positive : V > 0
  areas_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0
  distances_positive : H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0
  K_positive : K > 0
  area_ratios : S₁ = K ∧ S₂ = 2*K ∧ S₃ = 3*K ∧ S₄ = 4*K

/-- The theorem stating the relationship between distances, volume, and K for a triangular pyramid. -/
theorem triangular_pyramid_distance_sum (p : TriangularPyramid) :
  p.H₁ + 2*p.H₂ + 3*p.H₃ + 4*p.H₄ = 3*p.V/p.K :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_distance_sum_l270_27095


namespace NUMINAMATH_CALUDE_zoo_rabbits_count_l270_27060

theorem zoo_rabbits_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 60)
  (h2 : total_legs = 192) : 
  ∃ (rabbits peacocks : ℕ), 
    rabbits + peacocks = total_heads ∧ 
    4 * rabbits + 2 * peacocks = total_legs ∧ 
    rabbits = 36 := by
  sorry

end NUMINAMATH_CALUDE_zoo_rabbits_count_l270_27060


namespace NUMINAMATH_CALUDE_class_size_l270_27019

/-- The number of people who like both baseball and football -/
def both : ℕ := 5

/-- The number of people who only like baseball -/
def only_baseball : ℕ := 2

/-- The number of people who only like football -/
def only_football : ℕ := 3

/-- The number of people who like neither baseball nor football -/
def neither : ℕ := 6

/-- The total number of people in the class -/
def total : ℕ := both + only_baseball + only_football + neither

theorem class_size : total = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_l270_27019


namespace NUMINAMATH_CALUDE_amanda_final_pay_l270_27004

/-- Calculate Amanda's final pay after deductions and penalties --/
theorem amanda_final_pay 
  (regular_wage : ℝ) 
  (regular_hours : ℝ) 
  (overtime_rate : ℝ) 
  (overtime_hours : ℝ) 
  (commission : ℝ) 
  (tax_rate : ℝ) 
  (insurance_rate : ℝ) 
  (other_expenses : ℝ) 
  (penalty_rate : ℝ) 
  (h1 : regular_wage = 50)
  (h2 : regular_hours = 8)
  (h3 : overtime_rate = 1.5)
  (h4 : overtime_hours = 2)
  (h5 : commission = 150)
  (h6 : tax_rate = 0.15)
  (h7 : insurance_rate = 0.05)
  (h8 : other_expenses = 40)
  (h9 : penalty_rate = 0.2) :
  let total_earnings := regular_wage * regular_hours + 
                        regular_wage * overtime_rate * overtime_hours + 
                        commission
  let deductions := total_earnings * tax_rate + 
                    total_earnings * insurance_rate + 
                    other_expenses
  let earnings_after_deductions := total_earnings - deductions
  let penalty := earnings_after_deductions * penalty_rate
  let final_pay := earnings_after_deductions - penalty
  final_pay = 416 := by sorry

end NUMINAMATH_CALUDE_amanda_final_pay_l270_27004


namespace NUMINAMATH_CALUDE_jogger_speed_l270_27016

theorem jogger_speed (usual_distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  usual_distance = 30 →
  faster_speed = 16 →
  extra_distance = 10 →
  ∃ (usual_speed : ℝ),
    usual_speed * (usual_distance + extra_distance) / faster_speed = usual_distance ∧
    usual_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_jogger_speed_l270_27016


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l270_27074

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 5500

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  coefficient := 5.5
  exponent := 3
  h_coefficient := by sorry
}

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct : 
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l270_27074


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_from_parabola_focus_l270_27053

/-- Given a parabola and a hyperbola with shared focus, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes_from_parabola_focus 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (b : ℝ) :
  (∀ x y, parabola x y ↔ y^2 = 16*x) →
  (∀ x y, hyperbola x y ↔ x^2/12 - y^2/b^2 = 1) →
  (∃ x₀, x₀ = 4 ∧ parabola x₀ 0 ∧ ∀ y, hyperbola x₀ y → y = 0) →
  (∀ x y, hyperbola x y → y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_from_parabola_focus_l270_27053


namespace NUMINAMATH_CALUDE_max_value_when_h_3_h_values_when_max_negative_one_l270_27006

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x - h)^2

-- Define the range of x
def x_range (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5

-- Part 1: Maximum value when h = 3
theorem max_value_when_h_3 :
  ∀ x, x_range x → f 3 x ≤ 0 ∧ ∃ x₀, x_range x₀ ∧ f 3 x₀ = 0 :=
sorry

-- Part 2: Values of h when maximum is -1
theorem h_values_when_max_negative_one :
  (∀ x, x_range x → f h x ≤ -1 ∧ ∃ x₀, x_range x₀ ∧ f h x₀ = -1) →
  h = 6 ∨ h = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_h_3_h_values_when_max_negative_one_l270_27006


namespace NUMINAMATH_CALUDE_log_product_equals_two_l270_27093

theorem log_product_equals_two (y : ℝ) (h : y > 0) : 
  (Real.log y / Real.log 3) * (Real.log 9 / Real.log y) = 2 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_two_l270_27093


namespace NUMINAMATH_CALUDE_integral_equals_22_over_3_l270_27097

theorem integral_equals_22_over_3 : ∫ x in (1 : ℝ)..3, (2 * x - 1 / x^2) = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_22_over_3_l270_27097


namespace NUMINAMATH_CALUDE_insulation_minimum_cost_l270_27050

/-- Represents the total cost function over 20 years for insulation thickness x (in cm) -/
def f (x : ℝ) : ℝ := 800 - 74 * x

/-- The domain of the function f -/
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 10

theorem insulation_minimum_cost :
  ∃ (x : ℝ), domain x ∧ f x = 700 ∧ ∀ (y : ℝ), domain y → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_insulation_minimum_cost_l270_27050


namespace NUMINAMATH_CALUDE_sum_of_extremes_in_third_row_l270_27098

/-- Represents a position in the grid -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 17

/-- The total number of cells in the grid -/
def totalCells : ℕ := gridSize * gridSize

/-- The center position of the grid -/
def centerPosition : Position :=
  ⟨gridSize / 2, gridSize / 2⟩

/-- Creates a spiral grid with numbers from 1 to totalCells -/
def createSpiralGrid : SpiralGrid := sorry

/-- Gets the number at a specific position in the grid -/
def getNumber (grid : SpiralGrid) (pos : Position) : ℕ := sorry

/-- Finds the smallest number in the third row -/
def smallestInThirdRow (grid : SpiralGrid) : ℕ := sorry

/-- Finds the largest number in the third row -/
def largestInThirdRow (grid : SpiralGrid) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_extremes_in_third_row :
  let grid := createSpiralGrid
  smallestInThirdRow grid + largestInThirdRow grid = 544 := by sorry

end NUMINAMATH_CALUDE_sum_of_extremes_in_third_row_l270_27098


namespace NUMINAMATH_CALUDE_desired_average_l270_27037

theorem desired_average (numbers : List ℕ) (h1 : numbers = [6, 16, 8, 22]) : 
  (numbers.sum / numbers.length : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_l270_27037


namespace NUMINAMATH_CALUDE_inequality_solution_l270_27026

theorem inequality_solution (x : ℝ) :
  0 < x ∧ x < Real.pi →
  ((8 / (3 * Real.sin x - Real.sin (3 * x))) + 3 * (Real.sin x)^2 ≤ 5) ↔
  x = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l270_27026


namespace NUMINAMATH_CALUDE_binomial_variance_calculation_l270_27003

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial variable -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial variable -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_calculation (ξ : BinomialVariable) 
  (h_n : ξ.n = 36) 
  (h_exp : expectedValue ξ = 12) : 
  variance ξ = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_calculation_l270_27003


namespace NUMINAMATH_CALUDE_fourth_week_distance_l270_27048

def running_schedule (week1_distance : ℝ) : ℕ → ℝ
  | 1 => week1_distance * 7
  | 2 => (2 * week1_distance + 3) * 7
  | 3 => (2 * week1_distance + 3) * 9
  | 4 => (2 * week1_distance + 3) * 9 * 0.9 * 0.5 * 5
  | _ => 0

theorem fourth_week_distance :
  running_schedule 2 4 = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_distance_l270_27048


namespace NUMINAMATH_CALUDE_average_rounds_is_four_l270_27041

/-- Represents the distribution of golf rounds played by members -/
structure GolfRoundsDistribution where
  rounds : Fin 6 → ℕ
  members : Fin 6 → ℕ

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRoundsRounded (dist : GolfRoundsDistribution) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => dist.rounds i * dist.members i)
  let totalMembers := (Finset.range 6).sum (λ i => dist.members i)
  (totalRounds + totalMembers / 2) / totalMembers

/-- The specific distribution given in the problem -/
def givenDistribution : GolfRoundsDistribution where
  rounds := λ i => i.val + 1
  members := ![4, 3, 5, 6, 2, 7]

theorem average_rounds_is_four :
  averageRoundsRounded givenDistribution = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_four_l270_27041


namespace NUMINAMATH_CALUDE_opposite_of_2023_l270_27049

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l270_27049


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_1181_over_20_l270_27036

theorem sqrt_fraction_sum_equals_sqrt_1181_over_20 :
  Real.sqrt (16/25 + 9/4 + 1/16) = Real.sqrt 1181 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_1181_over_20_l270_27036


namespace NUMINAMATH_CALUDE_min_sum_polygon_sides_and_count_l270_27025

theorem min_sum_polygon_sides_and_count (m n : ℕ) : 
  m > 0 → 
  n ≥ 3 → 
  (180 * m * n - 360 * m) % 8 = 0 → 
  ∃ (m' n' : ℕ), m' > 0 ∧ n' ≥ 3 ∧ (180 * m' * n' - 360 * m') % 8 = 0 ∧ m' + n' = 5 ∧ ∀ (k l : ℕ), k > 0 → l ≥ 3 → (180 * k * l - 360 * k) % 8 = 0 → k + l ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_polygon_sides_and_count_l270_27025


namespace NUMINAMATH_CALUDE_total_profit_is_6300_l270_27065

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit based on the given profit sharing scenario -/
def calculate_total_profit (ps : ProfitSharing) : ℕ :=
  let tom_investment_months := ps.tom_investment * ps.tom_months
  let jose_investment_months := ps.jose_investment * ps.jose_months
  let ratio_denominator := tom_investment_months + jose_investment_months
  let tom_profit := (tom_investment_months * ps.jose_profit) / jose_investment_months
  tom_profit + ps.jose_profit

/-- Theorem stating that the total profit for the given scenario is 6300 -/
theorem total_profit_is_6300 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000) 
  (h2 : ps.tom_months = 12) 
  (h3 : ps.jose_investment = 4500) 
  (h4 : ps.jose_months = 10) 
  (h5 : ps.jose_profit = 3500) : 
  calculate_total_profit ps = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_6300_l270_27065


namespace NUMINAMATH_CALUDE_individual_can_cost_l270_27034

-- Define the cost of a 12-pack of soft drinks
def pack_cost : ℚ := 299 / 100

-- Define the number of cans in a pack
def cans_per_pack : ℕ := 12

-- Define the function to calculate the cost per can
def cost_per_can : ℚ := pack_cost / cans_per_pack

-- Theorem to prove
theorem individual_can_cost : 
  (round (cost_per_can * 100) / 100 : ℚ) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_individual_can_cost_l270_27034


namespace NUMINAMATH_CALUDE_fraction_difference_l270_27086

theorem fraction_difference (r s : ℕ+) : 
  (5 : ℚ) / 11 < (r : ℚ) / s ∧ 
  (r : ℚ) / s < 4 / 9 ∧ 
  (∀ (r' s' : ℕ+), (5 : ℚ) / 11 < (r' : ℚ) / s' ∧ (r' : ℚ) / s' < 4 / 9 → s ≤ s') →
  s - r = 11 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l270_27086


namespace NUMINAMATH_CALUDE_sin_cos_properties_l270_27058

open Real

theorem sin_cos_properties : ¬(
  (∃ (T : ℝ), T > 0 ∧ T = π/2 ∧ ∀ (x : ℝ), sin (2*x) = sin (2*(x + T))) ∧
  (∀ (x : ℝ), cos x = cos (π - x))
) := by sorry

end NUMINAMATH_CALUDE_sin_cos_properties_l270_27058


namespace NUMINAMATH_CALUDE_waiter_remaining_customers_l270_27005

/-- Calculates the number of remaining customers after some customers leave. -/
def remainingCustomers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem waiter_remaining_customers :
  remainingCustomers 21 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_waiter_remaining_customers_l270_27005


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l270_27042

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x, x^2 - 3*x + 2 < 0 ∧ ¬(x < 2)) = False ∧
  (∃ x, x < 2 ∧ x^2 - 3*x + 2 ≥ 0) = True :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l270_27042


namespace NUMINAMATH_CALUDE_least_common_denominator_l270_27089

theorem least_common_denominator (a b c d e : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 8) (he : e = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 72 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l270_27089


namespace NUMINAMATH_CALUDE_line_through_points_with_slope_l270_27085

theorem line_through_points_with_slope (k : ℝ) : 
  (∃ (m : ℝ), m = (3 * k - (-9)) / (7 - k) ∧ m = 2 * k) → 
  k = 9 / 2 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_with_slope_l270_27085


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l270_27013

theorem quartic_equation_roots (a b : ℝ) :
  let x : ℝ → Prop := λ x => x^4 - 2*a*x^2 + b^2 = 0
  ∃ (ε₁ ε₂ : {r : ℝ // r = 1 ∨ r = -1}),
    x (ε₁ * (Real.sqrt ((a + b)/2) + ε₂ * Real.sqrt ((a - b)/2))) :=
by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l270_27013


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_97_l270_27017

theorem largest_five_digit_divisible_by_97 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 97 = 0 → n ≤ 99930 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_97_l270_27017


namespace NUMINAMATH_CALUDE_news_spread_theorem_l270_27080

/-- Represents a village with residents and their acquaintance relationships -/
structure Village where
  residents : Finset Nat
  acquaintances : Nat → Nat → Prop

/-- Represents the spread of news in the village over time -/
def news_spread (v : Village) (initial : Finset Nat) (t : Nat) : Finset Nat :=
  sorry

/-- The theorem stating that there exists a subset of 90 residents that can spread news to all residents within 10 days -/
theorem news_spread_theorem (v : Village) : 
  v.residents.card = 1000 → 
  ∃ (subset : Finset Nat), 
    subset.card = 90 ∧ 
    subset ⊆ v.residents ∧
    news_spread v subset 10 = v.residents :=
  sorry

end NUMINAMATH_CALUDE_news_spread_theorem_l270_27080


namespace NUMINAMATH_CALUDE_range_of_5m_minus_n_l270_27011

-- Define a decreasing and odd function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_decreasing : ∀ x y, x < y → f x > f y)
variable (h_odd : ∀ x, f (-x) = -f x)

-- Define the conditions on m and n
variable (m n : ℝ)
variable (h_cond1 : f m + f (n - 2) ≤ 0)
variable (h_cond2 : f (m - n - 1) ≤ 0)

-- Theorem statement
theorem range_of_5m_minus_n : 5 * m - n ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_5m_minus_n_l270_27011


namespace NUMINAMATH_CALUDE_twenty_is_least_pieces_l270_27033

/-- The number of expected guests -/
def expected_guests : Set Nat := {10, 11}

/-- A function to check if a number of pieces can be equally divided among a given number of guests -/
def can_divide_equally (pieces : Nat) (guests : Nat) : Prop :=
  ∃ (share : Nat), pieces = guests * share

/-- The proposition that a given number of pieces is the least number that can be equally divided among either 10 or 11 guests -/
def is_least_pieces (pieces : Nat) : Prop :=
  (∀ g ∈ expected_guests, can_divide_equally pieces g) ∧
  (∀ p < pieces, ∃ g ∈ expected_guests, ¬can_divide_equally p g)

/-- Theorem stating that 20 is the least number of pieces that can be equally divided among either 10 or 11 guests -/
theorem twenty_is_least_pieces : is_least_pieces 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_least_pieces_l270_27033


namespace NUMINAMATH_CALUDE_smallest_gregory_bottles_l270_27075

/-- The number of bottles Paul drinks -/
def paul_bottles : ℕ → ℕ := fun p => p

/-- The number of bottles Donald drinks -/
def donald_bottles : ℕ → ℕ := fun p => 2 * paul_bottles p + 3

/-- The number of bottles Gregory drinks -/
def gregory_bottles : ℕ → ℕ := fun p => 3 * donald_bottles p - 5

theorem smallest_gregory_bottles :
  ∀ p : ℕ, p ≥ 1 → gregory_bottles p ≥ 10 ∧ gregory_bottles 1 = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_gregory_bottles_l270_27075


namespace NUMINAMATH_CALUDE_power_equation_solution_l270_27002

theorem power_equation_solution :
  ∃ x : ℝ, (5 : ℝ)^(x + 2) = 625 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l270_27002


namespace NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l270_27069

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp (total : ℕ) (stay_home : ℕ) : ℕ :=
  total - stay_home

/-- Proof that 610769 kids from Lawrence county go to camp -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1201565 590796 = 610769 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_camp_attendance_l270_27069


namespace NUMINAMATH_CALUDE_balls_removed_by_other_students_l270_27091

theorem balls_removed_by_other_students (tennis_balls soccer_balls baskets students_removed_8 remaining_balls : ℕ) 
  (h1 : tennis_balls = 15)
  (h2 : soccer_balls = 5)
  (h3 : baskets = 5)
  (h4 : students_removed_8 = 3)
  (h5 : remaining_balls = 56) : 
  ((baskets * (tennis_balls + soccer_balls)) - (students_removed_8 * 8) - remaining_balls) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_balls_removed_by_other_students_l270_27091


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l270_27024

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
theorem man_speed_in_still_water 
  (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 34)
  (h2 : downstream_speed = 48) :
  (upstream_speed + downstream_speed) / 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l270_27024


namespace NUMINAMATH_CALUDE_stone_breadth_proof_l270_27021

/-- Given a hall and stones with specified dimensions, prove the breadth of each stone -/
theorem stone_breadth_proof (hall_length hall_width : ℝ) (stone_length : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.3)
  (h4 : num_stones = 3600) :
  ∃ (stone_breadth : ℝ), 
    stone_breadth = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_length * stone_breadth * num_stones) :=
by sorry

end NUMINAMATH_CALUDE_stone_breadth_proof_l270_27021


namespace NUMINAMATH_CALUDE_blood_expiration_date_l270_27055

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the number of days in January -/
def days_in_january : ℕ := 31

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

/-- Represents the expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 10

theorem blood_expiration_date :
  blood_expiration_time / seconds_per_day = days_in_january + 11 :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_date_l270_27055


namespace NUMINAMATH_CALUDE_children_in_milburg_l270_27032

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The total population of Milburg -/
def total_population : ℕ := 8243

/-- Theorem: The number of children in Milburg is 2987 -/
theorem children_in_milburg : total_population - grown_ups = 2987 := by
  sorry

end NUMINAMATH_CALUDE_children_in_milburg_l270_27032


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l270_27000

/-- The side length of the nth square in the sequence -/
def square_side (n : ℕ) : ℕ := 2 * n - 1

/-- The number of tiles in the nth square -/
def square_tiles (n : ℕ) : ℕ := (square_side n) ^ 2

/-- The difference in tiles between the 9th and 8th squares -/
def tile_difference : ℕ := square_tiles 9 - square_tiles 8

theorem ninth_minus_eighth_square_tiles : tile_difference = 64 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l270_27000


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l270_27076

/-- Given an equation (x^2 - dx)/(cx - k) = (m-2)/(m+2) where c, d, and k are constants,
    prove that when m = 2(c - d)/(c + d), the equation has roots which are numerically
    equal but of opposite signs. -/
theorem roots_opposite_signs (c d k : ℝ) :
  let m := 2 * (c - d) / (c + d)
  let f := fun x => (x^2 - d*x) / (c*x - k) - (m - 2) / (m + 2)
  ∃ (r : ℝ), f r = 0 ∧ f (-r) = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l270_27076


namespace NUMINAMATH_CALUDE_stirling_second_kind_l270_27084

/-- Stirling number of the second kind -/
def S (n k : ℕ) : ℚ :=
  sorry

/-- Main theorem for Stirling numbers of the second kind -/
theorem stirling_second_kind (n : ℕ) (h : n ≥ 2) :
  (∀ k, k ≥ 2 → S n k = k * S (n-1) k + S (n-1) (k-1)) ∧
  S n 1 = 1 ∧
  S n 2 = 2^(n-1) - 1 ∧
  S n 3 = (1/6) * 3^n - (1/2) * 2^n + 1/2 ∧
  S n 4 = (1/24) * 4^n - (1/6) * 3^n + (1/4) * 2^n - 1/6 :=
by
  sorry

end NUMINAMATH_CALUDE_stirling_second_kind_l270_27084


namespace NUMINAMATH_CALUDE_xyz_inequality_l270_27092

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l270_27092


namespace NUMINAMATH_CALUDE_uncle_height_l270_27015

/-- Represents the heights of James and his uncle before and after James' growth spurt -/
structure HeightScenario where
  james_initial : ℝ
  uncle : ℝ
  james_growth : ℝ
  height_diff_after : ℝ

/-- The conditions of the problem -/
def problem_conditions (h : HeightScenario) : Prop :=
  h.james_initial = (2/3) * h.uncle ∧
  h.james_growth = 10 ∧
  h.height_diff_after = 14 ∧
  h.uncle = (h.james_initial + h.james_growth + h.height_diff_after)

/-- The theorem stating that given the problem conditions, the uncle's height is 72 inches -/
theorem uncle_height (h : HeightScenario) : 
  problem_conditions h → h.uncle = 72 := by sorry

end NUMINAMATH_CALUDE_uncle_height_l270_27015


namespace NUMINAMATH_CALUDE_counters_ratio_l270_27063

/-- Represents a person with counters and marbles -/
structure Person where
  counters : ℕ
  marbles : ℕ

/-- The problem setup -/
def problem : Prop :=
  ∃ (reina kevin : Person),
    kevin.counters = 40 ∧
    kevin.marbles = 50 ∧
    reina.marbles = 4 * kevin.marbles ∧
    reina.counters + reina.marbles = 320 ∧
    reina.counters * 1 = kevin.counters * 3

/-- The theorem stating that the ratio of Reina's counters to Kevin's counters is 3:1 -/
theorem counters_ratio : problem := by
  sorry

end NUMINAMATH_CALUDE_counters_ratio_l270_27063


namespace NUMINAMATH_CALUDE_bezout_identity_solutions_l270_27062

theorem bezout_identity_solutions (a b d u v : ℤ) 
  (h_gcd : d = Int.gcd a b) 
  (h_bezout : a * u + b * v = d) : 
  (∀ x y : ℤ, a * x + b * y = d ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a) ∧
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by sorry

end NUMINAMATH_CALUDE_bezout_identity_solutions_l270_27062


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l270_27079

-- Define the initial purchase price
def initialPrice : ℝ := 8000

-- Define the depreciation rate (20% = 0.20)
def depreciationRate : ℝ := 0.20

-- Define the time period in years
def timePeriod : ℕ := 2

-- Function to calculate the market value after a given number of years
def marketValue (years : ℕ) : ℝ :=
  initialPrice * (1 - depreciationRate) ^ years

-- Theorem statement
theorem machine_value_after_two_years :
  marketValue timePeriod = 5120 := by
  sorry


end NUMINAMATH_CALUDE_machine_value_after_two_years_l270_27079


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l270_27056

theorem rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 24)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 10) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    a * b * c = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l270_27056


namespace NUMINAMATH_CALUDE_line_slope_l270_27012

theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_slope_l270_27012


namespace NUMINAMATH_CALUDE_hotel_towels_l270_27059

theorem hotel_towels (num_rooms : ℕ) (people_per_room : ℕ) (total_towels : ℕ) : 
  num_rooms = 10 →
  people_per_room = 3 →
  total_towels = 60 →
  total_towels / (num_rooms * people_per_room) = 2 := by
sorry

end NUMINAMATH_CALUDE_hotel_towels_l270_27059


namespace NUMINAMATH_CALUDE_exam_students_count_l270_27022

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  T = 88 * N ∧ 
  T - 8 * 50 = 92 * (N - 8) ∧ 
  T - 8 * 50 - 100 = 92 * (N - 9) → 
  N = 84 := by
sorry

end NUMINAMATH_CALUDE_exam_students_count_l270_27022


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_minus_x_minus_one_geq_zero_l270_27030

theorem negation_of_forall_exp_minus_x_minus_one_geq_zero :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_minus_x_minus_one_geq_zero_l270_27030


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l270_27078

theorem product_from_lcm_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.lcm a b = 60) (h4 : Nat.gcd a b = 5) : a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l270_27078


namespace NUMINAMATH_CALUDE_range_of_a_l270_27094

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a+2)*x + b

-- Define the theorem
theorem range_of_a (a b : ℝ) :
  (f a b (-1) = -2) →
  (∀ x, ∃ y, y = Real.log (f a b x + 3)) →
  -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l270_27094


namespace NUMINAMATH_CALUDE_sqrt_16_div_2_l270_27029

theorem sqrt_16_div_2 : Real.sqrt 16 / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_div_2_l270_27029


namespace NUMINAMATH_CALUDE_subtracted_number_for_perfect_square_l270_27046

theorem subtracted_number_for_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_for_perfect_square_l270_27046


namespace NUMINAMATH_CALUDE_faye_coloring_books_l270_27040

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

theorem faye_coloring_books : 
  let initial_books : ℕ := 34
  let books_bought : ℕ := 48
  let final_books : ℕ := 79
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l270_27040


namespace NUMINAMATH_CALUDE_chocolates_distribution_l270_27081

/-- Given a large box containing small boxes and chocolate bars, 
    calculate the number of chocolate bars in each small box. -/
def chocolates_per_small_box (total_chocolates : ℕ) (num_small_boxes : ℕ) : ℕ :=
  total_chocolates / num_small_boxes

/-- Theorem: In a large box with 15 small boxes and 300 chocolate bars,
    each small box contains 20 chocolate bars. -/
theorem chocolates_distribution :
  chocolates_per_small_box 300 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l270_27081


namespace NUMINAMATH_CALUDE_xy_plus_2y_value_l270_27083

theorem xy_plus_2y_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x*y + 2*y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_2y_value_l270_27083


namespace NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l270_27027

theorem six_digit_divisibility_by_seven (a b c d e f : ℕ) :
  (0 < a) →
  (a < 10) →
  (b < 10) →
  (c < 10) →
  (d < 10) →
  (e < 10) →
  (f < 10) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 7 = 0 →
  (100000 * f + 10000 * a + 1000 * b + 100 * c + 10 * d + e) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l270_27027


namespace NUMINAMATH_CALUDE_number_of_triceratopses_l270_27010

/-- Represents the number of rhinoceroses -/
def r : ℕ := sorry

/-- Represents the number of triceratopses -/
def t : ℕ := sorry

/-- The total number of horns -/
def total_horns : ℕ := 31

/-- The total number of legs -/
def total_legs : ℕ := 48

/-- Theorem stating that the number of triceratopses is 7 -/
theorem number_of_triceratopses : t = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_triceratopses_l270_27010


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_three_l270_27008

/-- Given that x = 2 and y = 1 is a solution to the equation x + my = 5, prove that m = 3 -/
theorem solution_implies_m_equals_three (x y m : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : x + m * y = 5) : 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_three_l270_27008


namespace NUMINAMATH_CALUDE_sum_special_integers_sum_special_integers_proof_l270_27064

theorem sum_special_integers : ℕ → ℤ → ℤ → Prop :=
  fun a b c =>
    (∀ n : ℕ, a ≤ n) →  -- a is the smallest natural number
    (0 < b ∧ ∀ m : ℤ, 0 < m → b ≤ m) →  -- b is the smallest positive integer
    (c < 0 ∧ ∀ k : ℤ, k < 0 → k ≤ c) →  -- c is the largest negative integer
    a + b + c = 0

-- The proof of this theorem is omitted
theorem sum_special_integers_proof : ∃ a : ℕ, ∃ b c : ℤ, sum_special_integers a b c :=
  sorry

end NUMINAMATH_CALUDE_sum_special_integers_sum_special_integers_proof_l270_27064


namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l270_27007

theorem cube_sum_minus_triple_product (p : ℕ) : 
  Prime p → 
  ({(x, y) : ℕ × ℕ | x^3 + y^3 - 3*x*y = p - 1} = 
    if p = 2 then {(1, 0), (0, 1)} 
    else if p = 5 then {(2, 2)} 
    else ∅) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l270_27007


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_real_implies_x_geq_5_l270_27077

theorem sqrt_x_minus_5_real_implies_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_real_implies_x_geq_5_l270_27077


namespace NUMINAMATH_CALUDE_perpendicular_vector_proof_l270_27090

/-- Given two parallel lines with direction vector (5, 4), prove that the vector (v₁, v₂) 
    perpendicular to (5, 4) satisfying v₁ + v₂ = 7 is (-28, 35). -/
theorem perpendicular_vector_proof (v₁ v₂ : ℝ) : 
  (5 * 4 + 4 * (-5) = 0) →  -- Lines are parallel with direction vector (5, 4)
  (5 * v₁ + 4 * v₂ = 0) →   -- (v₁, v₂) is perpendicular to (5, 4)
  (v₁ + v₂ = 7) →           -- Sum of v₁ and v₂ is 7
  (v₁ = -28 ∧ v₂ = 35) :=   -- Conclusion: v₁ = -28 and v₂ = 35
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vector_proof_l270_27090


namespace NUMINAMATH_CALUDE_rowing_time_calculation_l270_27045

-- Define the given constants
def man_speed : ℝ := 6
def river_speed : ℝ := 3
def total_distance : ℝ := 4.5

-- Define the theorem
theorem rowing_time_calculation :
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := total_distance / 2
  let upstream_time := one_way_distance / upstream_speed
  let downstream_time := one_way_distance / downstream_speed
  let total_time := upstream_time + downstream_time
  total_time = 1 := by sorry

end NUMINAMATH_CALUDE_rowing_time_calculation_l270_27045


namespace NUMINAMATH_CALUDE_convention_handshakes_l270_27031

/-- Represents the Annual Mischief Convention --/
structure Convention where
  num_gremlins : ℕ
  num_imps : ℕ
  num_antisocial_gremlins : ℕ

/-- Calculates the number of handshakes at the convention --/
def count_handshakes (c : Convention) : ℕ :=
  let social_gremlins := c.num_gremlins - c.num_antisocial_gremlins
  let gremlin_handshakes := social_gremlins * (social_gremlins - 1) / 2
  let imp_gremlin_handshakes := c.num_imps * c.num_gremlins
  gremlin_handshakes + imp_gremlin_handshakes

/-- The main theorem stating the number of handshakes at the convention --/
theorem convention_handshakes :
  let c := Convention.mk 25 18 5
  count_handshakes c = 640 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l270_27031


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l270_27072

/-- Given a geometric sequence with positive terms and a specific arithmetic sequence condition, prove the common ratio. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_positive : ∀ n, a n > 0)
  (h_arith : a 1 + 2 * a 2 = a 3) : 
  q = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l270_27072


namespace NUMINAMATH_CALUDE_point_on_y_axis_l270_27038

/-- A point M with coordinates (t-3, 5-t) is on the y-axis if and only if its coordinates are (0, 2) -/
theorem point_on_y_axis (t : ℝ) :
  (t - 3 = 0 ∧ (t - 3, 5 - t) = (0, 2)) ↔ (t - 3, 5 - t).1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l270_27038


namespace NUMINAMATH_CALUDE_only_statement4_correct_l270_27057

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetryYOZPlane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetryYAxis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetryOrigin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

-- Define the statements
def statement1 (p : Point3D) : Prop := symmetryXAxis p = ⟨p.x, -p.y, p.z⟩
def statement2 (p : Point3D) : Prop := symmetryYOZPlane p = ⟨p.x, -p.y, -p.z⟩
def statement3 (p : Point3D) : Prop := symmetryYAxis p = ⟨-p.x, p.y, p.z⟩
def statement4 (p : Point3D) : Prop := symmetryOrigin p = ⟨-p.x, -p.y, -p.z⟩

-- Theorem to prove
theorem only_statement4_correct (p : Point3D) :
  ¬(statement1 p) ∧ ¬(statement2 p) ∧ ¬(statement3 p) ∧ (statement4 p) :=
sorry

end NUMINAMATH_CALUDE_only_statement4_correct_l270_27057


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l270_27096

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l270_27096


namespace NUMINAMATH_CALUDE_power_sum_fifth_l270_27068

/-- Given real numbers a, b, x, y satisfying certain conditions, 
    prove that ax^5 + by^5 = 180.36 -/
theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 56) :
  a * x^5 + b * y^5 = 180.36 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l270_27068


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l270_27066

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7 where a, b, c are real constants,
    if f(-2011) = -17, then f(2011) = 31 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := λ x : ℝ => a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l270_27066


namespace NUMINAMATH_CALUDE_bell_rings_count_l270_27088

/-- Represents a school day with a given number of classes -/
structure SchoolDay where
  num_classes : ℕ
  current_class : ℕ

/-- Calculates the number of times the bell has rung -/
def bell_rings (day : SchoolDay) : ℕ :=
  2 * (day.current_class - 1) + 1

/-- Theorem stating that the bell has rung 15 times -/
theorem bell_rings_count (day : SchoolDay) 
  (h1 : day.num_classes = 8) 
  (h2 : day.current_class = day.num_classes) : 
  bell_rings day = 15 := by
  sorry

#check bell_rings_count

end NUMINAMATH_CALUDE_bell_rings_count_l270_27088


namespace NUMINAMATH_CALUDE_completing_square_transformation_l270_27061

theorem completing_square_transformation (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l270_27061


namespace NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l270_27023

/-- A twelve-sided die with faces numbered from 1 to 12 -/
structure TwelveSidedDie :=
  (faces : Finset ℕ)
  (face_count : faces.card = 12)
  (face_range : ∀ n, n ∈ faces ↔ 1 ≤ n ∧ n ≤ 12)

/-- The expected value of a roll of a twelve-sided die -/
def expected_value (d : TwelveSidedDie) : ℚ :=
  (d.faces.sum id) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die :
  ∀ d : TwelveSidedDie, expected_value d = 13/2 :=
sorry

end NUMINAMATH_CALUDE_expected_value_twelve_sided_die_l270_27023


namespace NUMINAMATH_CALUDE_factors_of_sixty_l270_27099

theorem factors_of_sixty : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_sixty_l270_27099


namespace NUMINAMATH_CALUDE_mary_regular_rate_l270_27087

/-- Represents Mary's work schedule and pay structure --/
structure MaryPayStructure where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  maxEarnings : ℚ

/-- Calculates Mary's regular hourly rate --/
def regularHourlyRate (m : MaryPayStructure) : ℚ :=
  let totalRegularHours := m.regularHours
  let totalOvertimeHours := m.maxHours - m.regularHours
  let overtimeMultiplier := 1 + m.overtimeRate
  m.maxEarnings / (totalRegularHours + overtimeMultiplier * totalOvertimeHours)

/-- Theorem stating that Mary's regular hourly rate is $8 --/
theorem mary_regular_rate :
  let m : MaryPayStructure := {
    maxHours := 70,
    regularHours := 20,
    overtimeRate := 1/4,
    maxEarnings := 660
  }
  regularHourlyRate m = 8 := by sorry

end NUMINAMATH_CALUDE_mary_regular_rate_l270_27087


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l270_27047

-- Define set A
def A (a : ℝ) : Set ℝ :=
  {y | y^2 - (a^2 + a + 1)*y + a*(a^2 + 1) > 0}

-- Define set B
def B : Set ℝ :=
  {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - x + 1}

-- Theorem 1
theorem intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ → 1 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem complement_intersection_when_a_minimum :
  let a : ℝ := -2
  (∀ x : ℝ, x^2 + 1 ≥ a*x) →
  (Set.compl (A a) ∩ B) = {y : ℝ | 2 ≤ y ∧ y ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l270_27047


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l270_27071

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 2 / 5) :
  a / c = 25 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l270_27071


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l270_27044

theorem complex_arithmetic_equality : 10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l270_27044


namespace NUMINAMATH_CALUDE_cubes_to_add_l270_27039

theorem cubes_to_add (small_cube_side : ℕ) (large_cube_side : ℕ) (add_cube_side : ℕ) : 
  small_cube_side = 8 →
  large_cube_side = 12 →
  add_cube_side = 2 →
  (large_cube_side^3 - small_cube_side^3) / add_cube_side^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_cubes_to_add_l270_27039


namespace NUMINAMATH_CALUDE_complex_equation_solution_l270_27067

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l270_27067


namespace NUMINAMATH_CALUDE_car_speed_proof_l270_27052

/-- Proves that a car's speed is 600 km/h given the problem conditions -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v - 1 / 900) * 3600 = 2 ↔ v = 600 := by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l270_27052


namespace NUMINAMATH_CALUDE_complex_second_quadrant_l270_27018

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a
  (z.re < 0 ∧ z.im > 0) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_l270_27018
