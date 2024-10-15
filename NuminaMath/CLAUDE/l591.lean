import Mathlib

namespace NUMINAMATH_CALUDE_bees_on_first_day_l591_59104

/-- Given that Mrs. Hilt saw some bees on the first day and 3 times as many on the second day,
    counting 432 bees on the second day, prove that she saw 144 bees on the first day. -/
theorem bees_on_first_day (first_day : ℕ) (second_day : ℕ) : 
  second_day = 3 * first_day → second_day = 432 → first_day = 144 := by
  sorry

end NUMINAMATH_CALUDE_bees_on_first_day_l591_59104


namespace NUMINAMATH_CALUDE_div_power_eq_reciprocal_power_l591_59157

/-- Division power operation for rational numbers -/
def div_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1 / a^(n - 1)

/-- Theorem: Division power equals reciprocal of power with exponent decreased by 2 -/
theorem div_power_eq_reciprocal_power (a : ℚ) (n : ℕ) (h : a ≠ 0) (hn : n ≥ 2) :
  div_power a n = 1 / a^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_div_power_eq_reciprocal_power_l591_59157


namespace NUMINAMATH_CALUDE_nested_fraction_equals_21_55_l591_59156

theorem nested_fraction_equals_21_55 :
  1 / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_21_55_l591_59156


namespace NUMINAMATH_CALUDE_range_f_a_2_range_a_two_zeros_l591_59103

-- Define the function f(x) = x^2 - ax - a + 3
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a + 3

-- Part 1: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_f_a_2 :
  ∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc 0 3, f 2 x = y :=
sorry

-- Part 2: Range of a when f(x) has two zeros x₁ and x₂ with x₁x₂ > 0
theorem range_a_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ * x₂ > 0) →
  a ∈ Set.Ioi (-6) ∪ Set.Ioo 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_f_a_2_range_a_two_zeros_l591_59103


namespace NUMINAMATH_CALUDE_peters_remaining_money_l591_59164

/-- Calculates Peter's remaining money after market purchases -/
theorem peters_remaining_money
  (initial_amount : ℕ)
  (potato_quantity potato_price : ℕ)
  (tomato_quantity tomato_price : ℕ)
  (cucumber_quantity cucumber_price : ℕ)
  (banana_quantity banana_price : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_quantity = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_quantity = 9)
  (h5 : tomato_price = 3)
  (h6 : cucumber_quantity = 5)
  (h7 : cucumber_price = 4)
  (h8 : banana_quantity = 3)
  (h9 : banana_price = 5) :
  initial_amount - (potato_quantity * potato_price +
                    tomato_quantity * tomato_price +
                    cucumber_quantity * cucumber_price +
                    banana_quantity * banana_price) = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l591_59164


namespace NUMINAMATH_CALUDE_increasing_odd_function_bound_l591_59141

/-- A function f: ℝ → ℝ is a "k-type increasing function" if for all x, f(x + k) > f(x) -/
def is_k_type_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (x + k) > f x

theorem increasing_odd_function_bound (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : ∀ x, f (-x) = -f x)
    (h_pos : ∀ x > 0, f x = |x - a| - 2*a)
    (h_inc : is_k_type_increasing f 2017) :
    a < 2017/6 := by
  sorry

end NUMINAMATH_CALUDE_increasing_odd_function_bound_l591_59141


namespace NUMINAMATH_CALUDE_hex_fraction_sum_max_l591_59152

theorem hex_fraction_sum_max (a b c : ℕ) (y : ℕ) (h1 : a ≤ 15) (h2 : b ≤ 15) (h3 : c ≤ 15)
  (h4 : (a * 256 + b * 16 + c : ℕ) = 4096 / y) (h5 : 0 < y) (h6 : y ≤ 16) :
  a + b + c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_hex_fraction_sum_max_l591_59152


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l591_59153

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l591_59153


namespace NUMINAMATH_CALUDE_sqrt_neg_nine_squared_l591_59139

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_nine_squared_l591_59139


namespace NUMINAMATH_CALUDE_donut_ratio_l591_59137

/-- Given a total of 40 donuts shared among Delta, Beta, and Gamma,
    where Delta takes 8 donuts and Gamma takes 8 donuts,
    prove that the ratio of Beta's donuts to Gamma's donuts is 3:1. -/
theorem donut_ratio :
  ∀ (total delta gamma beta : ℕ),
    total = 40 →
    delta = 8 →
    gamma = 8 →
    beta = total - delta - gamma →
    beta / gamma = 3 := by
  sorry

end NUMINAMATH_CALUDE_donut_ratio_l591_59137


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l591_59149

theorem fraction_sum_equals_one (x : ℝ) : x / (x + 1) + 1 / (x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l591_59149


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l591_59182

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2004 + b^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l591_59182


namespace NUMINAMATH_CALUDE_share_distribution_l591_59129

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 595 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 70 := by sorry

end NUMINAMATH_CALUDE_share_distribution_l591_59129


namespace NUMINAMATH_CALUDE_kvass_price_after_increases_l591_59100

theorem kvass_price_after_increases (x y : ℝ) : 
  x + y = 1 →
  1.2 * (0.5 * x + y) = 1 →
  1.44 * y < 1 :=
by sorry

end NUMINAMATH_CALUDE_kvass_price_after_increases_l591_59100


namespace NUMINAMATH_CALUDE_original_average_l591_59140

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 15) (h2 : (n * (a + 12)) / n = 52) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l591_59140


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l591_59166

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l591_59166


namespace NUMINAMATH_CALUDE_not_always_externally_tangent_l591_59181

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the positional relationship between two circles
inductive CircleRelationship
  | Disjoint
  | ExternallyTangent
  | Intersecting
  | InternallyTangent
  | Contained

-- Define a function to determine if two circles have no intersection
def noIntersection (c1 c2 : Circle) : Prop :=
  sorry

-- Define a function to determine the relationship between two circles
def circleRelationship (c1 c2 : Circle) : CircleRelationship :=
  sorry

-- Theorem statement
theorem not_always_externally_tangent (c1 c2 : Circle) :
  ¬(noIntersection c1 c2 → circleRelationship c1 c2 = CircleRelationship.ExternallyTangent) :=
sorry

end NUMINAMATH_CALUDE_not_always_externally_tangent_l591_59181


namespace NUMINAMATH_CALUDE_most_cost_effective_boat_rental_l591_59161

/-- Represents the cost and capacity of a boat type -/
structure BoatType where
  capacity : Nat
  cost : Nat

/-- Represents a combination of boats -/
structure BoatCombination where
  largeboats : Nat
  smallboats : Nat

def totalPeople (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.capacity + b.smallboats * small.capacity

def totalCost (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.cost + b.smallboats * small.cost

def isSufficient (b : BoatCombination) (large : BoatType) (small : BoatType) (people : Nat) : Prop :=
  totalPeople b large small ≥ people

def isMoreCostEffective (b1 b2 : BoatCombination) (large : BoatType) (small : BoatType) : Prop :=
  totalCost b1 large small < totalCost b2 large small

theorem most_cost_effective_boat_rental :
  let large : BoatType := { capacity := 6, cost := 24 }
  let small : BoatType := { capacity := 4, cost := 20 }
  let people : Nat := 46
  let optimal : BoatCombination := { largeboats := 7, smallboats := 1 }
  (isSufficient optimal large small people) ∧
  (∀ b : BoatCombination, 
    isSufficient b large small people → 
    totalCost optimal large small ≤ totalCost b large small) := by
  sorry

#check most_cost_effective_boat_rental

end NUMINAMATH_CALUDE_most_cost_effective_boat_rental_l591_59161


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l591_59172

theorem imaginary_part_of_complex_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l591_59172


namespace NUMINAMATH_CALUDE_max_value_of_sum_l591_59108

theorem max_value_of_sum (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (a + b + c + d : ℤ) ≤ -5 ∧ ∃ (a₀ c₀ d₀ : ℤ) (b₀ : ℕ+), 
    a₀ + b₀ = c₀ ∧ b₀ + c₀ = d₀ ∧ c₀ + d₀ = a₀ ∧ a₀ + b₀ + c₀ + d₀ = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l591_59108


namespace NUMINAMATH_CALUDE_jeans_price_proof_l591_59154

/-- The original price of one pair of jeans -/
def original_price : ℝ := 40

/-- The discounted price for two pairs of jeans -/
def discounted_price (p : ℝ) : ℝ := 2 * p * 0.9

/-- The total price for three pairs of jeans -/
def total_price (p : ℝ) : ℝ := discounted_price p + p

theorem jeans_price_proof :
  total_price original_price = 112 :=
sorry

end NUMINAMATH_CALUDE_jeans_price_proof_l591_59154


namespace NUMINAMATH_CALUDE_certain_number_equation_l591_59133

theorem certain_number_equation : ∃ x : ℝ, 0.6 * 50 = 0.45 * x + 16.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l591_59133


namespace NUMINAMATH_CALUDE_divisors_of_16n4_l591_59127

theorem divisors_of_16n4 (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 13) :
  (Nat.divisors (16 * n^4)).card = 245 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_16n4_l591_59127


namespace NUMINAMATH_CALUDE_larry_channels_l591_59148

/-- Calculates the final number of channels Larry has after a series of changes. -/
def final_channels (initial : ℕ) (removed1 : ℕ) (added1 : ℕ) (removed2 : ℕ) (added2 : ℕ) (added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that given the specific changes to Larry's channel package, he ends up with 147 channels. -/
theorem larry_channels : final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l591_59148


namespace NUMINAMATH_CALUDE_train_crossing_time_l591_59187

/-- Proves that a train of given length crossing a bridge of given length in a given time will take 40 seconds to cross a signal post. -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 9000 →
  bridge_crossing_time = 600 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l591_59187


namespace NUMINAMATH_CALUDE_final_value_is_990_l591_59116

def loop_calculation (s i : ℕ) : ℕ :=
  if i ≥ 9 then loop_calculation (s * i) (i - 1)
  else s

theorem final_value_is_990 : loop_calculation 1 11 = 990 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_990_l591_59116


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l591_59197

/-- The volume of a rectangular prism with face areas √3, √5, and √15 is √15 -/
theorem rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 3)
  (h2 : x * z = Real.sqrt 5)
  (h3 : y * z = Real.sqrt 15) :
  x * y * z = Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l591_59197


namespace NUMINAMATH_CALUDE_simplify_fraction_l591_59199

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (3 / (x - 1)) + ((x - 3) / (1 - x^2)) = (2*x + 6) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l591_59199


namespace NUMINAMATH_CALUDE_square_sum_problem_l591_59168

theorem square_sum_problem (a b c d : ℤ) (h : (a^2 + b^2) * (c^2 + d^2) = 1993) : a^2 + b^2 + c^2 + d^2 = 1994 := by
  sorry

-- Define 1993 as a prime number
def p : ℕ := 1993

axiom p_prime : Nat.Prime p

end NUMINAMATH_CALUDE_square_sum_problem_l591_59168


namespace NUMINAMATH_CALUDE_simplified_rational_expression_l591_59196

theorem simplified_rational_expression (x : ℝ) 
  (h1 : x^2 - 5*x + 6 ≠ 0) 
  (h2 : x^2 - 7*x + 12 ≠ 0) 
  (h3 : x^2 - 5*x + 4 ≠ 0) : 
  (x^2 - 3*x + 2) / (x^2 - 5*x + 6) / ((x^2 - 5*x + 4) / (x^2 - 7*x + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_rational_expression_l591_59196


namespace NUMINAMATH_CALUDE_tile_arrangement_probability_l591_59190

theorem tile_arrangement_probability : 
  let total_tiles : ℕ := 7
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 3
  let favorable_arrangements : ℕ := Nat.choose 4 2
  let total_arrangements : ℕ := Nat.choose total_tiles x_tiles
  (favorable_arrangements : ℚ) / total_arrangements = 6 / 35 := by
sorry

end NUMINAMATH_CALUDE_tile_arrangement_probability_l591_59190


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l591_59193

/-- An ellipse containing two specific circles has a minimum area of (3√3/2)π -/
theorem ellipse_minimum_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l591_59193


namespace NUMINAMATH_CALUDE_intersection_point_determines_d_l591_59160

theorem intersection_point_determines_d : ∀ d : ℝ,
  (∃ x y : ℝ, 3 * x - 4 * y = d ∧ 6 * x + 8 * y = -d ∧ x = 2 ∧ y = -3) →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_determines_d_l591_59160


namespace NUMINAMATH_CALUDE_arithmetic_equality_l591_59130

theorem arithmetic_equality : (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l591_59130


namespace NUMINAMATH_CALUDE_set_relations_l591_59102

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 2 * x + 6 * k < 0 ∧ k ≠ 0}

theorem set_relations (k : ℝ) :
  (A k ⊆ Set.Ioo 2 3 → k ≥ 2/5) ∧
  (Set.Ioo 2 3 ⊆ A k → k ≤ 2/5) ∧
  (Set.inter (A k) (Set.Ioo 2 3) ≠ ∅ → k < Real.sqrt 6 / 6) :=
sorry

end NUMINAMATH_CALUDE_set_relations_l591_59102


namespace NUMINAMATH_CALUDE_vector_expression_evaluation_l591_59101

/-- Prove that the vector expression evaluates to the given result -/
theorem vector_expression_evaluation :
  (⟨3, -8⟩ : ℝ × ℝ) - 5 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-7, 12⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_evaluation_l591_59101


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l591_59195

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 6)
  (h_S : S = 42)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 35 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l591_59195


namespace NUMINAMATH_CALUDE_no_rearrangement_sum_999999999_l591_59117

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define a predicate to check if a number is a digit rearrangement of another
def isDigitRearrangement (n m : ℕ) : Prop :=
  sumOfDigits n = sumOfDigits m

theorem no_rearrangement_sum_999999999 (n : ℕ) :
  ¬∃ m : ℕ, isDigitRearrangement n m ∧ m + n = 999999999 :=
sorry

end NUMINAMATH_CALUDE_no_rearrangement_sum_999999999_l591_59117


namespace NUMINAMATH_CALUDE_correct_initial_chips_l591_59146

/-- The number of chips Marnie ate initially to see if she likes them -/
def initial_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := 100

/-- The number of chips Marnie eats per day starting from the second day -/
def daily_chips : ℕ := 10

/-- The total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the initial number of chips Marnie ate is correct -/
theorem correct_initial_chips :
  2 * initial_chips + (total_days - 1) * daily_chips = total_chips :=
by sorry

end NUMINAMATH_CALUDE_correct_initial_chips_l591_59146


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l591_59115

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, x₀ - 2 > Real.log x₀) ↔ (∀ x : ℝ, x - 2 ≤ Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l591_59115


namespace NUMINAMATH_CALUDE_candies_per_packet_is_18_l591_59120

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby has -/
def num_packets : ℕ := 2

/-- The number of days Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating that the number of candies in each packet is 18 -/
theorem candies_per_packet_is_18 :
  candies_per_packet * num_packets = 
    (days_eating_two * 2 + days_eating_one) * weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_candies_per_packet_is_18_l591_59120


namespace NUMINAMATH_CALUDE_max_surface_area_increase_l591_59171

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the original small cuboid -/
def originalCuboid : CuboidDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem stating the maximum increase in surface area -/
theorem max_surface_area_increase :
  ∃ (finalCuboid : CuboidDimensions),
    surfaceArea finalCuboid - surfaceArea originalCuboid ≤ 10 ∧
    ∀ (otherCuboid : CuboidDimensions),
      surfaceArea otherCuboid - surfaceArea originalCuboid ≤
        surfaceArea finalCuboid - surfaceArea originalCuboid :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_increase_l591_59171


namespace NUMINAMATH_CALUDE_second_term_value_l591_59185

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the Δ operator
def delta (A : Sequence) : Sequence :=
  λ n => A (n + 1) - A n

-- Main theorem
theorem second_term_value (A : Sequence) 
  (h1 : ∀ n, delta (delta A) n = 1)
  (h2 : A 12 = 0)
  (h3 : A 22 = 0) : 
  A 2 = 100 := by
sorry


end NUMINAMATH_CALUDE_second_term_value_l591_59185


namespace NUMINAMATH_CALUDE_father_speed_is_60kmh_l591_59158

/-- Misha's father's driving speed in km/h -/
def father_speed : ℝ := 60

/-- Distance Misha walked in km -/
def misha_walk_distance : ℝ := 5

/-- Time saved in minutes -/
def time_saved : ℝ := 10

/-- Proves that Misha's father's driving speed is 60 km/h given the conditions -/
theorem father_speed_is_60kmh :
  father_speed = 60 ∧
  misha_walk_distance = 5 ∧
  time_saved = 10 →
  father_speed = 60 := by
  sorry

#check father_speed_is_60kmh

end NUMINAMATH_CALUDE_father_speed_is_60kmh_l591_59158


namespace NUMINAMATH_CALUDE_farthest_point_is_two_zero_l591_59147

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the tangency conditions
def externally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle1 x' y' → ((x - x')^2 + (y - y')^2 = (1 + r)^2)

def internally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle2 x' y' → ((x - x')^2 + (y - y')^2 = (3 - r)^2)

-- Define the farthest point condition
def is_farthest_point (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y ∧
  ∀ (x' y' : ℝ), externally_tangent x' y' → internally_tangent x' y' → 
    (x^2 + y^2 ≥ x'^2 + y'^2)

-- Theorem statement
theorem farthest_point_is_two_zero : is_farthest_point 2 0 := by sorry

end NUMINAMATH_CALUDE_farthest_point_is_two_zero_l591_59147


namespace NUMINAMATH_CALUDE_triangle_max_area_l591_59107

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  (Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) →
  (a + b + c = 12) →
  (∀ a' b' c' : ℝ, a' + b' + c' = 12 → 
    a' * b' * Real.sin C / 2 ≤ 36 * (3 - 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l591_59107


namespace NUMINAMATH_CALUDE_mischievous_quadratic_min_root_product_l591_59124

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := x^2 - (r + s) * x + r * s

/-- A polynomial is mischievous if p(p(x)) = 0 has exactly four real roots -/
def IsMischievous (p : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, p (p x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem stating that the mischievous quadratic polynomial with minimized root product evaluates to 1 at x = 1 -/
theorem mischievous_quadratic_min_root_product (r s : ℝ) :
  IsMischievous (QuadraticPolynomial r s) →
  (∀ r' s' : ℝ, IsMischievous (QuadraticPolynomial r' s') → r * s ≤ r' * s') →
  QuadraticPolynomial r s 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mischievous_quadratic_min_root_product_l591_59124


namespace NUMINAMATH_CALUDE_worst_player_is_daughter_l591_59179

-- Define the set of players
inductive Player
| Father
| Sister
| Daughter
| Son

-- Define the gender type
inductive Gender
| Male
| Female

-- Define the generation type
inductive Generation
| Older
| Younger

-- Function to get the gender of a player
def gender : Player → Gender
| Player.Father => Gender.Male
| Player.Sister => Gender.Female
| Player.Daughter => Gender.Female
| Player.Son => Gender.Male

-- Function to get the generation of a player
def generation : Player → Generation
| Player.Father => Generation.Older
| Player.Sister => Generation.Older
| Player.Daughter => Generation.Younger
| Player.Son => Generation.Younger

-- Function to determine if two players could be twins
def couldBeTwins : Player → Player → Prop
| Player.Daughter, Player.Son => True
| Player.Son, Player.Daughter => True
| _, _ => False

-- Theorem statement
theorem worst_player_is_daughter :
  ∀ (worst best : Player),
    (∃ twin : Player, couldBeTwins worst twin ∧ gender twin = gender best) →
    generation worst ≠ generation best →
    worst = Player.Daughter :=
sorry

end NUMINAMATH_CALUDE_worst_player_is_daughter_l591_59179


namespace NUMINAMATH_CALUDE_expression_value_l591_59132

theorem expression_value (x y : ℚ) (h : 12 * x = 4 * y + 2) :
  6 * y - 18 * x + 7 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_value_l591_59132


namespace NUMINAMATH_CALUDE_kale_spring_mowings_l591_59159

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowings : ℕ := sorry

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowings : ℕ := 5

/-- The difference between spring and summer mowings -/
def mowing_difference : ℕ := 3

/-- Theorem stating that Kale mowed his lawn 8 times in the spring -/
theorem kale_spring_mowings :
  spring_mowings = 8 ∧
  summer_mowings = 5 ∧
  spring_mowings - summer_mowings = mowing_difference :=
sorry

end NUMINAMATH_CALUDE_kale_spring_mowings_l591_59159


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l591_59123

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 3

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

def distinguishable_arrangements : ℕ := total_tiles.factorial / (brown_tiles.factorial * purple_tiles.factorial * green_tiles.factorial * yellow_tiles.factorial)

theorem tile_arrangement_count : distinguishable_arrangements = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l591_59123


namespace NUMINAMATH_CALUDE_train_crossing_time_l591_59151

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 180 →
  train_speed_kmh = 72 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l591_59151


namespace NUMINAMATH_CALUDE_no_real_solutions_l591_59131

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 20 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l591_59131


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_intersects_l_l591_59118

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Define the line l
def l (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Define the line we found
def found_line (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem line_passes_through_P_and_intersects_l :
  -- The line passes through P
  found_line P.1 P.2 ∧
  -- The line is not parallel to l (they intersect)
  ∃ x y, found_line x y ∧ l x y ∧ (x, y) ≠ P :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_intersects_l_l591_59118


namespace NUMINAMATH_CALUDE_total_pies_sold_l591_59136

/-- Represents a type of pie --/
inductive PieType
| Shepherds
| ChickenPot
| VegetablePot
| BeefPot

/-- Represents the size of a pie --/
inductive PieSize
| Small
| Large

/-- Represents the number of pieces a pie is cut into --/
def pieceCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 4
  | PieType.Shepherds, PieSize.Large => 8
  | PieType.ChickenPot, PieSize.Small => 5
  | PieType.ChickenPot, PieSize.Large => 10
  | PieType.VegetablePot, PieSize.Small => 6
  | PieType.VegetablePot, PieSize.Large => 12
  | PieType.BeefPot, PieSize.Small => 7
  | PieType.BeefPot, PieSize.Large => 14

/-- Represents the number of customers who ordered each type and size of pie --/
def customerCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 52
  | PieType.Shepherds, PieSize.Large => 76
  | PieType.ChickenPot, PieSize.Small => 80
  | PieType.ChickenPot, PieSize.Large => 130
  | PieType.VegetablePot, PieSize.Small => 42
  | PieType.VegetablePot, PieSize.Large => 96
  | PieType.BeefPot, PieSize.Small => 35
  | PieType.BeefPot, PieSize.Large => 105

/-- Calculates the number of pies sold for a given type and size --/
def piesSold (t : PieType) (s : PieSize) : ℕ :=
  (customerCount t s + pieceCount t s - 1) / pieceCount t s

/-- Theorem: The total number of pies sold is 80 --/
theorem total_pies_sold :
  (piesSold PieType.Shepherds PieSize.Small +
   piesSold PieType.Shepherds PieSize.Large +
   piesSold PieType.ChickenPot PieSize.Small +
   piesSold PieType.ChickenPot PieSize.Large +
   piesSold PieType.VegetablePot PieSize.Small +
   piesSold PieType.VegetablePot PieSize.Large +
   piesSold PieType.BeefPot PieSize.Small +
   piesSold PieType.BeefPot PieSize.Large) = 80 :=
by sorry

end NUMINAMATH_CALUDE_total_pies_sold_l591_59136


namespace NUMINAMATH_CALUDE_eggs_division_l591_59145

theorem eggs_division (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) :
  total_eggs = 15 →
  num_groups = 3 →
  eggs_per_group * num_groups = total_eggs →
  eggs_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_division_l591_59145


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l591_59138

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root --/
axiom tangent_iff_double_root (a b c : ℝ) : 
  (∃ k, a * k^2 + b * k + c = 0 ∧ b^2 - 4*a*c = 0) ↔ 
  (∃! x y : ℝ, a * x^2 + b * x + c = 0 ∧ y^2 = 4 * a * x)

/-- The main theorem: if the line 4x + 7y + k = 0 is tangent to the parabola y^2 = 16x, then k = 49 --/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 7*y + k = 0 → y^2 = 16*x) →
  (∃! x y : ℝ, 4*x + 7*y + k = 0 ∧ y^2 = 16*x) →
  k = 49 := by
  sorry


end NUMINAMATH_CALUDE_line_tangent_to_parabola_l591_59138


namespace NUMINAMATH_CALUDE_tangent_lines_slope_4_tangent_line_at_point_2_neg6_l591_59174

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_slope_4 (x y : ℝ) :
  (4 * x - y - 18 = 0 ∨ 4 * x - y - 14 = 0) →
  ∃ x₀ : ℝ, f' x₀ = 4 ∧ y = f x₀ + 4 * (x - x₀) :=
sorry

theorem tangent_line_at_point_2_neg6 (x y : ℝ) :
  13 * x - y - 32 = 0 →
  y = f 2 + f' 2 * (x - 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_slope_4_tangent_line_at_point_2_neg6_l591_59174


namespace NUMINAMATH_CALUDE_total_balls_l591_59144

def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := 30

theorem total_balls : 
  soccer_balls + basketballs + tennis_balls + baseballs + volleyballs = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l591_59144


namespace NUMINAMATH_CALUDE_divisibility_implication_l591_59114

theorem divisibility_implication (x y : ℤ) : 
  (∃ k : ℤ, 4*x - y = 3*k) → (∃ m : ℤ, 4*x^2 + 7*x*y - 2*y^2 = 9*m) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l591_59114


namespace NUMINAMATH_CALUDE_truck_loading_time_l591_59105

theorem truck_loading_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 5) :
  1 / (rate1 + rate2) = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_time_l591_59105


namespace NUMINAMATH_CALUDE_park_area_theorem_l591_59142

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- Calculates the area of a rectangular park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- Calculates the perimeter of a rectangular park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Calculates the fencing cost for a rectangular park -/
def fencingCost (park : RectangularPark) (costPerMeter : ℝ) : ℝ :=
  perimeter park * costPerMeter

theorem park_area_theorem (park : RectangularPark) :
  fencingCost park 0.5 = 155 → area park = 5766 := by
  sorry

end NUMINAMATH_CALUDE_park_area_theorem_l591_59142


namespace NUMINAMATH_CALUDE_same_height_siblings_l591_59194

-- Define the number of siblings
def num_siblings : ℕ := 5

-- Define the total height of all siblings
def total_height : ℕ := 330

-- Define the height of one sibling
def one_sibling_height : ℕ := 60

-- Define Eliza's height
def eliza_height : ℕ := 68

-- Define the height difference between Eliza and one sibling
def height_difference : ℕ := 2

-- Theorem to prove
theorem same_height_siblings (h : ℕ) : 
  h * 2 + one_sibling_height + eliza_height + (eliza_height + height_difference) = total_height →
  h = 66 := by
  sorry


end NUMINAMATH_CALUDE_same_height_siblings_l591_59194


namespace NUMINAMATH_CALUDE_smallest_single_discount_l591_59110

theorem smallest_single_discount (m : ℕ) : m = 29 ↔ 
  (∀ k : ℕ, k < m → 
    ((1 - k / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.08)^3 ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.12)^2)) ∧
  ((1 - m / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - m / 100 : ℝ) < (1 - 0.08)^3 ∧
   (1 - m / 100 : ℝ) < (1 - 0.12)^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_single_discount_l591_59110


namespace NUMINAMATH_CALUDE_hard_candy_colouring_amount_l591_59175

/-- Represents the candy store's daily production and food colouring usage --/
structure CandyStore where
  lollipop_colouring : ℕ  -- ml of food colouring per lollipop
  lollipops_made : ℕ      -- number of lollipops made
  hard_candies_made : ℕ   -- number of hard candies made
  total_colouring : ℕ     -- total ml of food colouring used

/-- Calculates the amount of food colouring needed for each hard candy --/
def hard_candy_colouring (store : CandyStore) : ℕ :=
  (store.total_colouring - store.lollipop_colouring * store.lollipops_made) / store.hard_candies_made

/-- Theorem stating the amount of food colouring needed for each hard candy --/
theorem hard_candy_colouring_amount (store : CandyStore)
  (h1 : store.lollipop_colouring = 5)
  (h2 : store.lollipops_made = 100)
  (h3 : store.hard_candies_made = 5)
  (h4 : store.total_colouring = 600) :
  hard_candy_colouring store = 20 := by
  sorry

end NUMINAMATH_CALUDE_hard_candy_colouring_amount_l591_59175


namespace NUMINAMATH_CALUDE_interval_length_theorem_l591_59192

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x_min x_max : ℝ), 
    (∀ x : ℝ, c ≤ 3*x + 4 ∧ 3*x + 4 ≤ d ↔ x_min ≤ x ∧ x ≤ x_max) ∧
    x_max - x_min = 15) →
  d - c = 45 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l591_59192


namespace NUMINAMATH_CALUDE_prob_at_least_one_diamond_l591_59184

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of diamond cards in a standard deck -/
def diamondCardCount : ℕ := 13

/-- Probability of drawing at least one diamond when drawing two cards without replacement -/
def probAtLeastOneDiamond : ℚ :=
  1 - (standardDeckSize - diamondCardCount) * (standardDeckSize - diamondCardCount - 1) /
      (standardDeckSize * (standardDeckSize - 1))

theorem prob_at_least_one_diamond :
  probAtLeastOneDiamond = 15 / 34 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_diamond_l591_59184


namespace NUMINAMATH_CALUDE_quadratic_solutions_second_eq_solutions_l591_59183

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 + x - 3 = 0

-- Define the second equation
def second_eq (x : ℝ) : Prop := (2*x + 1)^2 = 3*(2*x + 1)

-- Theorem for the first equation
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_eq x1 ∧ 
    quadratic_eq x2 ∧ 
    x1 = (-1 + Real.sqrt 13) / 2 ∧ 
    x2 = (-1 - Real.sqrt 13) / 2 :=
sorry

-- Theorem for the second equation
theorem second_eq_solutions :
  ∃ x1 x2 : ℝ, 
    second_eq x1 ∧ 
    second_eq x2 ∧ 
    x1 = -1/2 ∧ 
    x2 = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_second_eq_solutions_l591_59183


namespace NUMINAMATH_CALUDE_cards_in_hospital_eq_403_l591_59188

/-- The number of cards Mariela received while in the hospital -/
def cards_in_hospital : ℕ := 690 - 287

/-- Theorem stating that Mariela received 403 cards while in the hospital -/
theorem cards_in_hospital_eq_403 : cards_in_hospital = 403 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_hospital_eq_403_l591_59188


namespace NUMINAMATH_CALUDE_complex_equation_solution_l591_59113

theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) + 2 * Complex.I * z = (2 : ℂ) - 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l591_59113


namespace NUMINAMATH_CALUDE_x_minus_y_values_l591_59125

theorem x_minus_y_values (x y : ℝ) (h1 : |x + 1| = 4) (h2 : (y + 2)^2 = 0) :
  x - y = 5 ∨ x - y = -3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l591_59125


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l591_59186

/-- Represents an isosceles triangle DEF -/
structure IsoscelesTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of one of the congruent sides -/
  side : ℝ
  /-- Assertion that the base is positive -/
  base_pos : base > 0
  /-- Assertion that the area is positive -/
  area_pos : area > 0
  /-- Assertion that the side is positive -/
  side_pos : side > 0

/-- Theorem stating the relationship between the base, area, and side length of an isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.area = 75) : 
  t.side = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l591_59186


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_asymptotes_and_point_l591_59162

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 3 - y^2 / 12 = 1

/-- Theorem: Given a hyperbola with asymptotes y = ±2x and passing through (2, 2),
    its equation is x²/3 - y²/12 = 1 -/
theorem hyperbola_equation_from_asymptotes_and_point :
  ∀ (h : Hyperbola), h.asymptote_slope = 2 → h.point = (2, 2) →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_asymptotes_and_point_l591_59162


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_eq_neg_e_l591_59165

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x
  else if x = 0 then 0
  else Real.exp (-x) - a * x

-- State the theorem
theorem three_zeros_implies_a_eq_neg_e (a : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a = -Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_three_zeros_implies_a_eq_neg_e_l591_59165


namespace NUMINAMATH_CALUDE_base5_to_base8_conversion_l591_59189

/-- Converts a base-5 number to base-10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 -/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem base5_to_base8_conversion :
  base10_to_base8 (base5_to_base10 1234) = 302 := by sorry

end NUMINAMATH_CALUDE_base5_to_base8_conversion_l591_59189


namespace NUMINAMATH_CALUDE_triangle_min_value_l591_59134

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB - bcosA = c/3, then the minimum value of (acosA + bcosB) / (acosB) is √2. -/
theorem triangle_min_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin B →
  a * Real.cos B - b * Real.cos A = c / 3 →
  ∃ (x : ℝ), x = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) ∧
    x ≥ Real.sqrt 2 ∧
    ∀ (y : ℝ), y = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_min_value_l591_59134


namespace NUMINAMATH_CALUDE_jean_grandchildren_gift_l591_59126

/-- Calculates the total amount given to grandchildren per year -/
def total_given_to_grandchildren (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proves that Jean gives $480 to her grandchildren per year -/
theorem jean_grandchildren_gift :
  total_given_to_grandchildren 3 2 80 = 480 := by
  sorry

#eval total_given_to_grandchildren 3 2 80

end NUMINAMATH_CALUDE_jean_grandchildren_gift_l591_59126


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l591_59106

def is_relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem smallest_sum_with_conditions :
  ∃ (a b c d e : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ¬(is_relatively_prime a b) ∧
    ¬(is_relatively_prime b c) ∧
    ¬(is_relatively_prime c d) ∧
    ¬(is_relatively_prime d e) ∧
    is_relatively_prime a c ∧
    is_relatively_prime a d ∧
    is_relatively_prime a e ∧
    is_relatively_prime b d ∧
    is_relatively_prime b e ∧
    is_relatively_prime c e ∧
    a + b + c + d + e = 75 ∧
    (∀ (a' b' c' d' e' : ℕ),
      a' > 0 → b' > 0 → c' > 0 → d' > 0 → e' > 0 →
      ¬(is_relatively_prime a' b') →
      ¬(is_relatively_prime b' c') →
      ¬(is_relatively_prime c' d') →
      ¬(is_relatively_prime d' e') →
      is_relatively_prime a' c' →
      is_relatively_prime a' d' →
      is_relatively_prime a' e' →
      is_relatively_prime b' d' →
      is_relatively_prime b' e' →
      is_relatively_prime c' e' →
      a' + b' + c' + d' + e' ≥ 75) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l591_59106


namespace NUMINAMATH_CALUDE_expression_simplification_l591_59150

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 27*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l591_59150


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l591_59191

/-- Given an arithmetic sequence {a_n} with a₁ = 1, common difference d ≠ 0,
    and a₁, a₂, and a₅ forming a geometric sequence, 
    prove that the sum of the first 8 terms (S₈) is equal to 64. -/
theorem arithmetic_geometric_sum (d : ℝ) (h1 : d ≠ 0) : 
  let a : ℕ → ℝ := fun n => 1 + (n - 1) * d
  let S : ℕ → ℝ := fun n => (n * (2 + (n - 1) * d)) / 2
  (a 2)^2 = (a 1) * (a 5) → S 8 = 64 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l591_59191


namespace NUMINAMATH_CALUDE_square_side_length_difference_l591_59155

/-- Given two squares with side lengths x and y, where the perimeter of the smaller square
    is 20 cm less than the perimeter of the larger square, prove that the side length of
    the larger square is 5 cm more than the side length of the smaller square. -/
theorem square_side_length_difference (x y : ℝ) (h : 4 * x + 20 = 4 * y) : y = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l591_59155


namespace NUMINAMATH_CALUDE_shop_inventory_l591_59109

theorem shop_inventory (large : ℕ) (medium : ℕ) (sold : ℕ) (left : ℕ) (small : ℕ) :
  large = 22 →
  medium = 50 →
  sold = 83 →
  left = 13 →
  large + medium + small = sold + left →
  small = 24 := by
sorry

end NUMINAMATH_CALUDE_shop_inventory_l591_59109


namespace NUMINAMATH_CALUDE_cory_chairs_proof_l591_59112

/-- The number of chairs Cory bought -/
def num_chairs : ℕ := 4

/-- The cost of the patio table -/
def table_cost : ℕ := 55

/-- The cost of each chair -/
def chair_cost : ℕ := 20

/-- The total cost of the table and chairs -/
def total_cost : ℕ := 135

theorem cory_chairs_proof :
  num_chairs * chair_cost + table_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cory_chairs_proof_l591_59112


namespace NUMINAMATH_CALUDE_jakes_drink_volume_l591_59121

/-- Represents the composition of a drink in parts -/
structure DrinkComposition :=
  (coke : ℕ)
  (sprite : ℕ)
  (mountainDew : ℕ)

/-- Calculates the total volume of a drink given its composition and the volume of Coke -/
def totalVolume (composition : DrinkComposition) (cokeVolume : ℚ) : ℚ :=
  let totalParts := composition.coke + composition.sprite + composition.mountainDew
  let volumePerPart := cokeVolume / composition.coke
  totalParts * volumePerPart

/-- Theorem: The total volume of Jake's drink is 18 ounces -/
theorem jakes_drink_volume :
  let composition : DrinkComposition := ⟨2, 1, 3⟩
  let cokeVolume : ℚ := 6
  totalVolume composition cokeVolume = 18 := by
  sorry

end NUMINAMATH_CALUDE_jakes_drink_volume_l591_59121


namespace NUMINAMATH_CALUDE_powder_division_theorem_l591_59128

/-- Represents the measurements and properties of the magical powder division problem. -/
structure PowderDivision where
  total_measured : ℝ
  remaining_measured : ℝ
  removed_measured : ℝ
  error : ℝ

/-- The actual weights of the two portions of the magical powder. -/
def actual_weights (pd : PowderDivision) : ℝ × ℝ :=
  (pd.remaining_measured - pd.error, pd.removed_measured - pd.error)

/-- Theorem stating that given the measurements and assuming a consistent error,
    the actual weights of the two portions are 4 and 3 zolotniks. -/
theorem powder_division_theorem (pd : PowderDivision) 
  (h1 : pd.total_measured = 6)
  (h2 : pd.remaining_measured = 3)
  (h3 : pd.removed_measured = 2)
  (h4 : pd.total_measured = pd.remaining_measured + pd.removed_measured - pd.error) :
  actual_weights pd = (4, 3) := by
  sorry

#eval actual_weights { total_measured := 6, remaining_measured := 3, removed_measured := 2, error := -1 }

end NUMINAMATH_CALUDE_powder_division_theorem_l591_59128


namespace NUMINAMATH_CALUDE_angle_coincidence_l591_59173

def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem angle_coincidence (α : ℝ) 
  (h1 : is_obtuse_angle α) 
  (h2 : (4 * α) % 360 = α % 360) : 
  α = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_coincidence_l591_59173


namespace NUMINAMATH_CALUDE_largest_sum_is_185_l591_59135

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of two two-digit numbers formed by three digits -/
def sum_XYZ (X Y Z : Digit) : ℕ := 10 * X.val + 11 * Y.val + Z.val

/-- The largest possible sum given the constraints -/
def largest_sum : ℕ := 185

/-- Theorem stating that 185 is the largest possible sum -/
theorem largest_sum_is_185 :
  ∀ X Y Z : Digit,
    X.val > Y.val →
    Y.val > Z.val →
    X ≠ Y →
    Y ≠ Z →
    X ≠ Z →
    sum_XYZ X Y Z ≤ largest_sum :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_185_l591_59135


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l591_59176

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, y^3 = x^3 + 8*x^2 - 6*x + 8 ↔ (x = 0 ∧ y = 2) ∨ (x = 9 ∧ y = 11) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l591_59176


namespace NUMINAMATH_CALUDE_shoe_pairs_l591_59163

theorem shoe_pairs (ellie riley : ℕ) : 
  ellie = riley + 3 →
  ellie + riley = 13 →
  ellie = 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_pairs_l591_59163


namespace NUMINAMATH_CALUDE_james_units_per_semester_l591_59169

/-- Given that James pays $2000 for 2 semesters and each unit costs $50,
    prove that he takes 20 units per semester. -/
theorem james_units_per_semester 
  (total_cost : ℕ) 
  (num_semesters : ℕ) 
  (unit_cost : ℕ) 
  (h1 : total_cost = 2000) 
  (h2 : num_semesters = 2) 
  (h3 : unit_cost = 50) : 
  (total_cost / num_semesters) / unit_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_units_per_semester_l591_59169


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l591_59167

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/a + 4/b ≥ 9/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 4/b = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l591_59167


namespace NUMINAMATH_CALUDE_no_integer_solution_l591_59111

def is_all_twos (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2

theorem no_integer_solution : ¬ ∃ (N : ℤ), is_all_twos (2008 * N.natAbs) :=
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l591_59111


namespace NUMINAMATH_CALUDE_max_children_correct_l591_59122

/-- Represents the types of buses available --/
inductive BusType
| A
| B
| C
| D

/-- Calculates the total number of seats for a given bus type --/
def totalSeats (t : BusType) : ℕ :=
  match t with
  | BusType.A => 36
  | BusType.B => 54
  | BusType.C => 36
  | BusType.D => 36

/-- Represents the safety regulation for maximum number of children per bus type --/
def safetyRegulation (t : BusType) : ℕ :=
  match t with
  | BusType.A => 40
  | BusType.B => 50
  | BusType.C => 35
  | BusType.D => 30

/-- Calculates the maximum number of children that can be accommodated on a given bus type --/
def maxChildren (t : BusType) : ℕ :=
  min (totalSeats t) (safetyRegulation t)

theorem max_children_correct :
  (maxChildren BusType.A = 36) ∧
  (maxChildren BusType.B = 50) ∧
  (maxChildren BusType.C = 35) ∧
  (maxChildren BusType.D = 30) :=
by
  sorry


end NUMINAMATH_CALUDE_max_children_correct_l591_59122


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l591_59143

/-- The probability of drawing two red shoes from a set of 6 red shoes and 4 green shoes is 1/3. -/
theorem probability_two_red_shoes (total_shoes : ℕ) (red_shoes : ℕ) (green_shoes : ℕ) :
  total_shoes = red_shoes + green_shoes →
  red_shoes = 6 →
  green_shoes = 4 →
  (red_shoes : ℚ) / total_shoes * ((red_shoes - 1) : ℚ) / (total_shoes - 1) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l591_59143


namespace NUMINAMATH_CALUDE_colored_segment_existence_l591_59119

/-- A color type with exactly 4 colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A point on a line with a color -/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The theorem statement -/
theorem colored_segment_existence 
  (n : ℕ) 
  (points : Fin n → ColoredPoint) 
  (h_n : n ≥ 4) 
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position) 
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (i j : Fin n), i < j ∧
    (∃ (c₁ c₂ c₃ c₄ : Color), 
      c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄ ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₁) ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₂) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₃) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₄)) :=
by
  sorry

end NUMINAMATH_CALUDE_colored_segment_existence_l591_59119


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l591_59198

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a q)
  (h_condition : a 5 ^ 2 = 2 * a 3 * a 9) :
  q = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l591_59198


namespace NUMINAMATH_CALUDE_jogging_problem_l591_59170

/-- Jogging problem -/
theorem jogging_problem (total_distance : ℝ) (total_time : ℝ) (halfway_point : ℝ) :
  total_distance = 3 →
  total_time = 24 →
  halfway_point = total_distance / 2 →
  (halfway_point / total_distance) * total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_jogging_problem_l591_59170


namespace NUMINAMATH_CALUDE_smallest_multiple_one_to_five_l591_59177

theorem smallest_multiple_one_to_five : ∃ n : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_one_to_five_l591_59177


namespace NUMINAMATH_CALUDE_red_balls_count_l591_59178

theorem red_balls_count (total_balls : ℕ) (black_balls : ℕ) (prob_black : ℚ) : 
  black_balls = 5 → 
  prob_black = 1/4 → 
  total_balls = black_balls + (total_balls - black_balls) →
  (total_balls - black_balls) = 15 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l591_59178


namespace NUMINAMATH_CALUDE_smallest_power_of_ten_minus_one_divisible_by_37_l591_59180

theorem smallest_power_of_ten_minus_one_divisible_by_37 :
  (∃ n : ℕ, 10^n - 1 ≡ 0 [MOD 37]) ∧
  (∀ k : ℕ, k < 3 → ¬(10^k - 1 ≡ 0 [MOD 37])) ∧
  (10^3 - 1 ≡ 0 [MOD 37]) :=
sorry

end NUMINAMATH_CALUDE_smallest_power_of_ten_minus_one_divisible_by_37_l591_59180
