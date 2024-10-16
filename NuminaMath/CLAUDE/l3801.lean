import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3801_380121

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3801_380121


namespace NUMINAMATH_CALUDE_hydrangea_price_l3801_380171

def pansy_price : ℝ := 2.50
def petunia_price : ℝ := 1.00
def num_pansies : ℕ := 5
def num_petunias : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00
def change_received : ℝ := 23.00

theorem hydrangea_price (hydrangea_cost : ℝ) : hydrangea_cost = 12.50 := by
  sorry

#check hydrangea_price

end NUMINAMATH_CALUDE_hydrangea_price_l3801_380171


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3801_380129

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3801_380129


namespace NUMINAMATH_CALUDE_min_cards_is_smallest_l3801_380180

/-- The smallest number of cards needed to represent all integers from 1 to n! as sums of factorials -/
def min_cards (n : ℕ+) : ℕ :=
  n.val * (n.val + 1) / 2 + 1

/-- Theorem stating that min_cards gives the smallest possible number of cards needed -/
theorem min_cards_is_smallest (n : ℕ+) :
  ∀ (t : ℕ), t ≤ n.val.factorial →
  ∃ (S : Finset ℕ),
    (∀ m ∈ S, ∃ k : ℕ+, m = k.val.factorial) ∧
    (S.card ≤ min_cards n) ∧
    (t = S.sum id) :=
sorry

end NUMINAMATH_CALUDE_min_cards_is_smallest_l3801_380180


namespace NUMINAMATH_CALUDE_probability_of_specific_three_card_arrangement_l3801_380157

/-- The number of possible arrangements of n distinct objects -/
def numberOfArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The probability of a specific arrangement given n distinct objects -/
def probabilityOfSpecificArrangement (n : ℕ) : ℚ :=
  1 / (numberOfArrangements n)

theorem probability_of_specific_three_card_arrangement :
  probabilityOfSpecificArrangement 3 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_three_card_arrangement_l3801_380157


namespace NUMINAMATH_CALUDE_square_of_binomial_l3801_380159

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 20 * x + 9 = (r * x + s)^2) → 
  a = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3801_380159


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3801_380132

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given vectors a and b, with a parallel to b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
    (ha : a = (2, 3)) 
    (hb : b = (4, -1 + y)) 
    (h_parallel : parallel a b) : 
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3801_380132


namespace NUMINAMATH_CALUDE_min_inverse_sum_min_inverse_sum_achieved_l3801_380141

theorem min_inverse_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  (1 / x + 1 / y) ≥ 3 / 5 := by
  sorry

theorem min_inverse_sum_achieved (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 12 ∧ x * y = 20 ∧ 1 / x + 1 / y = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_inverse_sum_min_inverse_sum_achieved_l3801_380141


namespace NUMINAMATH_CALUDE_fraction_inequality_l3801_380151

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3801_380151


namespace NUMINAMATH_CALUDE_container_capacity_l3801_380153

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 45 →
  final_fill = 0.75 →
  ∃ (capacity : Real), capacity = 100 ∧
    final_fill * capacity = initial_fill * capacity + added_water :=
by sorry

end NUMINAMATH_CALUDE_container_capacity_l3801_380153


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3801_380107

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is equal to 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ ↦ 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ, r θ * Real.sin θ) ∈ Metric.sphere center radius) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3801_380107


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3801_380166

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : ArithmeticSequence b)
  (ha1 : a 1 = 3)
  (hb1 : b 1 = -3)
  (h19 : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3801_380166


namespace NUMINAMATH_CALUDE_smallest_a1_l3801_380122

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 9 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : SequenceA a) :
  ∀ a1 : ℝ, a 1 ≥ a1 → a1 ≥ 19/36 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l3801_380122


namespace NUMINAMATH_CALUDE_two_cos_sixty_degrees_equals_one_l3801_380168

theorem two_cos_sixty_degrees_equals_one : 2 * Real.cos (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_sixty_degrees_equals_one_l3801_380168


namespace NUMINAMATH_CALUDE_waiter_customers_l3801_380111

theorem waiter_customers (initial new_customers customers_left : ℕ) :
  initial ≥ customers_left →
  (initial - customers_left + new_customers : ℕ) = initial - customers_left + new_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l3801_380111


namespace NUMINAMATH_CALUDE_packaging_cost_per_bar_l3801_380133

/-- Proves that the cost of packaging material per bar is $2 -/
theorem packaging_cost_per_bar
  (num_bars : ℕ)
  (cost_per_bar : ℚ)
  (total_selling_price : ℚ)
  (total_profit : ℚ)
  (h1 : num_bars = 5)
  (h2 : cost_per_bar = 5)
  (h3 : total_selling_price = 90)
  (h4 : total_profit = 55) :
  (total_selling_price - total_profit - (↑num_bars * cost_per_bar)) / ↑num_bars = 2 := by
  sorry

#check packaging_cost_per_bar

end NUMINAMATH_CALUDE_packaging_cost_per_bar_l3801_380133


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_exists_x0_negation_is_false_l3801_380108

-- Define the necessary condition
def necessary_condition (a b : ℝ) : Prop := a + b > 4

-- Define the stronger condition
def stronger_condition (a b : ℝ) : Prop := a > 2 ∧ b > 2

-- Statement 1: Necessary but not sufficient condition
theorem necessary_not_sufficient :
  (∀ a b : ℝ, stronger_condition a b → necessary_condition a b) ∧
  (∃ a b : ℝ, necessary_condition a b ∧ ¬stronger_condition a b) := by sorry

-- Statement 2: Existence of x₀
theorem exists_x0 : ∃ x₀ : ℝ, x₀^2 - x₀ > 0 := by sorry

-- Statement 3: Negation is false
theorem negation_is_false : ¬(∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_exists_x0_negation_is_false_l3801_380108


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3801_380187

theorem quadratic_inequality_equivalence (x : ℝ) : 
  x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3801_380187


namespace NUMINAMATH_CALUDE_simplify_fraction_l3801_380160

theorem simplify_fraction : 15 * (16 / 9) * (-45 / 32) = -25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3801_380160


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3801_380135

theorem gcd_of_three_numbers : Nat.gcd 17934 (Nat.gcd 23526 51774) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3801_380135


namespace NUMINAMATH_CALUDE_sugar_packs_theorem_l3801_380196

/-- Given the total amount of sugar, weight per pack, and leftover sugar, 
    calculate the number of packs. -/
def calculate_packs (total_sugar : ℕ) (weight_per_pack : ℕ) (leftover_sugar : ℕ) : ℕ :=
  (total_sugar - leftover_sugar) / weight_per_pack

/-- Theorem stating that given the specific conditions, 
    the number of packs is 12. -/
theorem sugar_packs_theorem (total_sugar weight_per_pack leftover_sugar : ℕ) 
  (h1 : total_sugar = 3020)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  calculate_packs total_sugar weight_per_pack leftover_sugar = 12 := by
  sorry

#eval calculate_packs 3020 250 20

end NUMINAMATH_CALUDE_sugar_packs_theorem_l3801_380196


namespace NUMINAMATH_CALUDE_g_range_l3801_380120

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 3 * Real.cos x ^ 2 - 4 * Real.cos x + 5 * Real.sin x ^ 2 - 7) / (Real.cos x - 2)

theorem g_range : 
  ∀ y : ℝ, (∃ x : ℝ, Real.cos x ≠ 2 ∧ g x = y) ↔ 1 ≤ y ∧ y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_g_range_l3801_380120


namespace NUMINAMATH_CALUDE_max_cuttable_strings_l3801_380123

/-- Represents a volleyball net as a graph --/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of nodes in the net --/
def VolleyballNet.nodeCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1)

/-- Calculates the total number of strings in the net --/
def VolleyballNet.stringCount (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + (net.rows + 1) * net.cols

/-- Theorem: Maximum number of cuttable strings in a 10x100 volleyball net --/
theorem max_cuttable_strings (net : VolleyballNet) 
  (h_rows : net.rows = 10) (h_cols : net.cols = 100) : 
  net.stringCount - (net.nodeCount - 1) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_max_cuttable_strings_l3801_380123


namespace NUMINAMATH_CALUDE_indians_invented_arabic_numerals_l3801_380167

/-- Represents a numerical system -/
structure NumericalSystem where
  digits : Set Nat
  name : String
  isUniversal : Bool

/-- The civilization that invented a numerical system -/
inductive Civilization
  | Indians
  | Chinese
  | Babylonians
  | Arabs

/-- Arabic numerals as defined in the problem -/
def arabicNumerals : NumericalSystem :=
  { digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    name := "Arabic numerals",
    isUniversal := true }

/-- The theorem stating that ancient Indians invented Arabic numerals -/
theorem indians_invented_arabic_numerals :
  ∃ (inventor : Civilization), inventor = Civilization.Indians ∧
  (arabicNumerals.digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
   arabicNumerals.name = "Arabic numerals" ∧
   arabicNumerals.isUniversal = true) :=
by sorry

end NUMINAMATH_CALUDE_indians_invented_arabic_numerals_l3801_380167


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l3801_380170

theorem fourth_term_of_sequence (x : ℤ) : 
  x^2 - 2*x - 3 < 0 → 
  ∃ (a : ℕ → ℤ), (∀ n, a (n+1) - a n = a 1 - a 0) ∧ 
                 (∀ n, a n = x → x^2 - 2*x - 3 < 0) ∧
                 (a 3 = 3 ∨ a 3 = -1) :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l3801_380170


namespace NUMINAMATH_CALUDE_carl_watermelon_price_l3801_380189

/-- Calculates the price per watermelon given initial count, remaining count, and total profit -/
def price_per_watermelon (initial_count : ℕ) (remaining_count : ℕ) (total_profit : ℕ) : ℚ :=
  total_profit / (initial_count - remaining_count)

/-- Theorem: Given Carl's watermelon sales data, prove the price per watermelon is $3 -/
theorem carl_watermelon_price :
  price_per_watermelon 53 18 105 = 3 := by
  sorry

end NUMINAMATH_CALUDE_carl_watermelon_price_l3801_380189


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_problem_l3801_380112

/-- The number of leaps required for a dog to catch a rabbit -/
def dog_catches_rabbit (initial_distance : ℕ) (dog_leap : ℕ) (rabbit_jump : ℕ) : ℕ :=
  initial_distance / (dog_leap - rabbit_jump)

theorem dog_catches_rabbit_problem :
  dog_catches_rabbit 150 9 7 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_problem_l3801_380112


namespace NUMINAMATH_CALUDE_similar_rectangle_ratio_l3801_380128

/-- Given a rectangle with length 40 meters and width 20 meters, 
    prove that a similar smaller rectangle with an area of 200 square meters 
    has dimensions that are 1/2 of the larger rectangle's dimensions. -/
theorem similar_rectangle_ratio (big_length big_width small_area : ℝ) 
  (h1 : big_length = 40)
  (h2 : big_width = 20)
  (h3 : small_area = 200)
  (h4 : small_area = (big_length * r) * (big_width * r)) 
  (r : ℝ) : r = 1 / 2 := by
  sorry

#check similar_rectangle_ratio

end NUMINAMATH_CALUDE_similar_rectangle_ratio_l3801_380128


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3801_380164

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3801_380164


namespace NUMINAMATH_CALUDE_not_juggling_sequence_l3801_380142

/-- Definition of the juggling sequence -/
def j : ℕ → ℕ
| 0 => 5
| 1 => 7
| 2 => 2
| n + 3 => j n

/-- Function f that calculates the time when a ball will be caught -/
def f (t : ℕ) : ℕ := t + j (t % 3)

/-- Theorem stating that 572 is not a juggling sequence -/
theorem not_juggling_sequence : ¬ (∀ n m : ℕ, n < 3 → m < 3 → n ≠ m → f n ≠ f m) := by
  sorry

end NUMINAMATH_CALUDE_not_juggling_sequence_l3801_380142


namespace NUMINAMATH_CALUDE_f_extremum_f_range_of_a_l3801_380117

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * (x - 1) + b * Real.exp x) / Real.exp x

-- Part 1
theorem f_extremum :
  let a : ℝ := -1
  let b : ℝ := 0
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≥ f a b x) ∧
  (∀ x : ℝ, f a b x ≥ -1 / Real.exp 2) ∧
  (¬ ∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M) := by sorry

-- Part 2
theorem f_range_of_a :
  let b : ℝ := 1
  (∀ a : ℝ, (∀ x : ℝ, f a b x ≠ 0) → a ∈ Set.Ioo (-Real.exp 2) 0) ∧
  (∀ a : ℝ, a ∈ Set.Ioo (-Real.exp 2) 0 → (∀ x : ℝ, f a b x ≠ 0)) := by sorry

end

end NUMINAMATH_CALUDE_f_extremum_f_range_of_a_l3801_380117


namespace NUMINAMATH_CALUDE_max_value_of_f_one_l3801_380195

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4, 
    the maximum value of f(1) is 7. -/
theorem max_value_of_f_one (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*b*x + a + 2*b
  (f 0 = 4) → (∀ y : ℝ, f 1 ≤ 7) ∧ (∃ y : ℝ, f 1 = 7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_one_l3801_380195


namespace NUMINAMATH_CALUDE_builder_wage_is_100_l3801_380138

/-- The daily wage of a builder given the construction rates and total cost -/
def builder_daily_wage (builders_per_floor : ℕ) (days_per_floor : ℕ) 
  (total_builders : ℕ) (total_houses : ℕ) (floors_per_house : ℕ) 
  (total_cost : ℕ) : ℚ :=
  (total_cost : ℚ) / (total_builders * total_houses * floors_per_house * days_per_floor : ℚ)

theorem builder_wage_is_100 :
  builder_daily_wage 3 30 6 5 6 270000 = 100 := by sorry

end NUMINAMATH_CALUDE_builder_wage_is_100_l3801_380138


namespace NUMINAMATH_CALUDE_removed_triangles_area_l3801_380173

theorem removed_triangles_area (s : ℝ) (h1 : s > 0) : 
  let x := (s - 8) / 2
  4 * (1/2 * x^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l3801_380173


namespace NUMINAMATH_CALUDE_gcd_85_100_l3801_380109

theorem gcd_85_100 : Nat.gcd 85 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_85_100_l3801_380109


namespace NUMINAMATH_CALUDE_mystery_number_addition_l3801_380162

theorem mystery_number_addition (mystery_number certain_number : ℕ) : 
  mystery_number = 47 → 
  mystery_number + certain_number = 92 → 
  certain_number = 45 := by
sorry

end NUMINAMATH_CALUDE_mystery_number_addition_l3801_380162


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3801_380103

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 567 [ZMOD 9]) → n ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3801_380103


namespace NUMINAMATH_CALUDE_triangle_side_length_l3801_380148

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.y = 7)
  (h2 : t.z = 6)
  (h3 : Real.cos (t.Y - t.Z) = 17/18) :
  t.x = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3801_380148


namespace NUMINAMATH_CALUDE_smallest_yellow_candy_count_l3801_380147

/-- The cost of a piece of yellow candy in cents -/
def yellow_candy_cost : ℕ := 15

/-- The number of red candies Joe can buy -/
def red_candy_count : ℕ := 10

/-- The number of green candies Joe can buy -/
def green_candy_count : ℕ := 16

/-- The number of blue candies Joe can buy -/
def blue_candy_count : ℕ := 18

theorem smallest_yellow_candy_count :
  ∃ n : ℕ, n > 0 ∧
  (yellow_candy_cost * n) % red_candy_count = 0 ∧
  (yellow_candy_cost * n) % green_candy_count = 0 ∧
  (yellow_candy_cost * n) % blue_candy_count = 0 ∧
  (∀ m : ℕ, m > 0 →
    (yellow_candy_cost * m) % red_candy_count = 0 →
    (yellow_candy_cost * m) % green_candy_count = 0 →
    (yellow_candy_cost * m) % blue_candy_count = 0 →
    m ≥ n) ∧
  n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_yellow_candy_count_l3801_380147


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3801_380193

theorem max_value_of_expression (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (h1 : x^2 + y^2 - x*y/2 = 36)
  (h2 : w^2 + z^2 + w*z/2 = 36)
  (h3 : x*z + y*w = 30) :
  (x*y + w*z)^2 ≤ 960 ∧ ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^2 + b^2 - a*b/2 = 36 ∧
    d^2 + c^2 + d*c/2 = 36 ∧
    a*c + b*d = 30 ∧
    (a*b + d*c)^2 = 960 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3801_380193


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3801_380191

/-- Polar to Cartesian Coordinate Conversion Theorem -/
theorem polar_to_cartesian_conversion (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) ∧
  (ρ^2 = x^2 + y^2) →
  (x^2 + (y - 2)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3801_380191


namespace NUMINAMATH_CALUDE_cortland_apples_l3801_380184

theorem cortland_apples (total : ℝ) (golden : ℝ) (macintosh : ℝ) 
  (h1 : total = 0.67)
  (h2 : golden = 0.17)
  (h3 : macintosh = 0.17) :
  total - (golden + macintosh) = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_cortland_apples_l3801_380184


namespace NUMINAMATH_CALUDE_max_value_of_a_l3801_380163

theorem max_value_of_a : 
  (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → 
  ∃ a_max : ℝ, a_max = 4 ∧ ∀ a : ℝ, (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3801_380163


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3801_380172

theorem shaded_area_theorem (square_side : ℝ) (total_beans : ℕ) (shaded_beans : ℕ) :
  square_side = 2 →
  total_beans = 200 →
  shaded_beans = 120 →
  (shaded_beans : ℝ) / (total_beans : ℝ) * (square_side ^ 2) = 12 / 5 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l3801_380172


namespace NUMINAMATH_CALUDE_cheezits_calorie_count_l3801_380139

/-- The number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := sorry

/-- The number of bags of Cheezits James ate -/
def bags_eaten : ℕ := 3

/-- The number of ounces per bag of Cheezits -/
def ounces_per_bag : ℕ := 2

/-- The number of minutes James ran -/
def minutes_run : ℕ := 40

/-- The number of calories James burned per minute of running -/
def calories_burned_per_minute : ℕ := 12

/-- The number of excess calories James had after eating and running -/
def excess_calories : ℕ := 420

theorem cheezits_calorie_count :
  calories_per_ounce = 150 ∧
  bags_eaten * ounces_per_bag * calories_per_ounce - minutes_run * calories_burned_per_minute = excess_calories :=
by sorry

end NUMINAMATH_CALUDE_cheezits_calorie_count_l3801_380139


namespace NUMINAMATH_CALUDE_inequality_proof_l3801_380127

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3801_380127


namespace NUMINAMATH_CALUDE_x_equation_result_l3801_380114

theorem x_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 3*x^5 + x^2 = -5*x + 4*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_result_l3801_380114


namespace NUMINAMATH_CALUDE_time_after_1875_minutes_l3801_380178

/-- Represents time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1875_minutes : 
  let start_time := Time.mk 15 15  -- 3:15 p.m.
  let end_time := Time.mk 10 30    -- 10:30 a.m.
  addMinutes start_time 1875 = end_time :=
by sorry

end NUMINAMATH_CALUDE_time_after_1875_minutes_l3801_380178


namespace NUMINAMATH_CALUDE_quarters_needed_for_final_soda_l3801_380130

theorem quarters_needed_for_final_soda (total_quarters : ℕ) (soda_cost : ℕ) : 
  total_quarters = 855 → soda_cost = 7 → 
  (soda_cost - (total_quarters % soda_cost)) = 6 := by
sorry

end NUMINAMATH_CALUDE_quarters_needed_for_final_soda_l3801_380130


namespace NUMINAMATH_CALUDE_impossible_circle_arrangement_l3801_380131

theorem impossible_circle_arrangement : ¬ ∃ (a : Fin 7 → ℕ),
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 1) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 2) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 3) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 4) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 5) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 6) ∧
  (∃ i, a i + a ((i + 1) % 7) + a ((i + 2) % 7) = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_circle_arrangement_l3801_380131


namespace NUMINAMATH_CALUDE_function_expression_l3801_380192

theorem function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x + 1) :
  ∀ x : ℝ, f x = (1/2) * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_function_expression_l3801_380192


namespace NUMINAMATH_CALUDE_group_dynamics_index_difference_l3801_380198

theorem group_dynamics_index_difference (n : ℕ) (k_f : ℕ) (h1 : n = 25) (h2 : k_f = 8) :
  let k_m := n - k_f
  let index_female := (n - k_f) / n
  let index_male := (n - k_m) / n
  index_female - index_male = 9 / 25 := by
sorry

end NUMINAMATH_CALUDE_group_dynamics_index_difference_l3801_380198


namespace NUMINAMATH_CALUDE_parallel_vectors_k_eq_two_l3801_380119

/-- Two vectors in ℝ² are parallel if and only if their components are proportional -/
axiom vector_parallel_iff_proportional {a b : ℝ × ℝ} :
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) ↔ ∃ (s : ℝ), a.1 * b.2 = s * a.2 * b.1

/-- Given vectors a = (k, 2) and b = (1, 1), if a is parallel to b, then k = 2 -/
theorem parallel_vectors_k_eq_two (k : ℝ) :
  let a : ℝ × ℝ := (k, 2)
  let b : ℝ × ℝ := (1, 1)
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_eq_two_l3801_380119


namespace NUMINAMATH_CALUDE_lecture_hall_tables_l3801_380152

theorem lecture_hall_tables (total_legs : ℕ) (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) :
  total_legs = 680 →
  stools_per_table = 8 →
  stool_legs = 4 →
  table_legs = 4 →
  (total_legs : ℚ) / ((stools_per_table * stool_legs + table_legs) : ℚ) = 680 / 36 :=
by sorry

end NUMINAMATH_CALUDE_lecture_hall_tables_l3801_380152


namespace NUMINAMATH_CALUDE_composition_ratio_l3801_380165

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3801_380165


namespace NUMINAMATH_CALUDE_stating_assignment_methods_eq_36_l3801_380104

/-- Represents the number of workshops --/
def num_workshops : ℕ := 3

/-- Represents the total number of employees --/
def total_employees : ℕ := 5

/-- Represents the number of employees that must be assigned together --/
def paired_employees : ℕ := 2

/-- Represents the number of remaining employees after considering the paired employees --/
def remaining_employees : ℕ := total_employees - paired_employees

/-- 
  Calculates the number of ways to assign employees to workshops
  given the constraints mentioned in the problem
--/
def assignment_methods : ℕ := 
  num_workshops * (remaining_employees.factorial + remaining_employees.choose 2 * (num_workshops - 1))

/-- 
  Theorem stating that the number of assignment methods
  satisfying the given conditions is 36
--/
theorem assignment_methods_eq_36 : assignment_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_stating_assignment_methods_eq_36_l3801_380104


namespace NUMINAMATH_CALUDE_shelves_per_case_l3801_380126

theorem shelves_per_case (num_cases : ℕ) (records_per_shelf : ℕ) (ridges_per_record : ℕ) 
  (shelf_fullness : ℚ) (total_ridges : ℕ) : ℕ :=
  let shelves_per_case := (total_ridges / (shelf_fullness * records_per_shelf * ridges_per_record)) / num_cases
  3

#check shelves_per_case 4 20 60 (3/5) 8640

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_shelves_per_case_l3801_380126


namespace NUMINAMATH_CALUDE_disconnected_circuit_scenarios_l3801_380169

/-- Represents a circuit with solder points -/
structure Circuit where
  total_points : ℕ
  is_disconnected : Bool

/-- Calculates the number of scenarios where solder points can fall off -/
def scenarios_with_fallen_points (c : Circuit) : ℕ :=
  2^c.total_points - 1

/-- Theorem: For a disconnected circuit with 6 solder points, there are 63 scenarios of fallen points -/
theorem disconnected_circuit_scenarios :
  ∀ (c : Circuit), c.total_points = 6 → c.is_disconnected = true →
  scenarios_with_fallen_points c = 63 := by
  sorry

#check disconnected_circuit_scenarios

end NUMINAMATH_CALUDE_disconnected_circuit_scenarios_l3801_380169


namespace NUMINAMATH_CALUDE_triangle_problem_l3801_380161

theorem triangle_problem (A B C : Real) (BC AB AC : Real) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ Real.cos A = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3801_380161


namespace NUMINAMATH_CALUDE_minutes_to_skate_on_ninth_day_l3801_380186

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 60

/-- The number of days Gage skated for 60 minutes -/
def days_skating_60_min : ℕ := 6

/-- The number of minutes Gage skated each day for the next 2 days -/
def minutes_per_day_next_2 : ℕ := 120

/-- The number of days Gage skated for 120 minutes -/
def days_skating_120_min : ℕ := 2

/-- The target average number of minutes per day for all 9 days -/
def target_average_minutes : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- Theorem stating the number of minutes Gage needs to skate on the 9th day -/
theorem minutes_to_skate_on_ninth_day :
  target_average_minutes * total_days -
  (minutes_per_day_first_6 * days_skating_60_min +
   minutes_per_day_next_2 * days_skating_120_min) = 300 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_skate_on_ninth_day_l3801_380186


namespace NUMINAMATH_CALUDE_thomas_worked_four_weeks_l3801_380182

/-- The number of whole weeks Thomas worked given his weekly rate and total amount paid -/
def weeks_worked (weekly_rate : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / weekly_rate : ℕ)

/-- Theorem stating that Thomas worked for 4 weeks -/
theorem thomas_worked_four_weeks :
  weeks_worked 4550 19500 = 4 := by
  sorry

end NUMINAMATH_CALUDE_thomas_worked_four_weeks_l3801_380182


namespace NUMINAMATH_CALUDE_profit_distribution_l3801_380101

/-- Profit distribution in a business partnership --/
theorem profit_distribution (a b c : ℕ) (profit_b : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → profit_b = 4000 →
  ∃ (profit_a profit_c : ℕ),
    profit_a * b = profit_b * a ∧
    profit_c * b = profit_b * c ∧
    profit_c - profit_a = 1600 :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_l3801_380101


namespace NUMINAMATH_CALUDE_negation_and_range_of_a_l3801_380145

def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0

theorem negation_and_range_of_a :
  (∀ a : ℝ, ¬(proposition_p a) ↔ ∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_negation_and_range_of_a_l3801_380145


namespace NUMINAMATH_CALUDE_log_stack_total_l3801_380125

/-- The sum of an arithmetic sequence with 15 terms, starting at 15 and ending at 1 -/
def log_stack_sum : ℕ := 
  let first_term := 15
  let last_term := 1
  let num_terms := 15
  (num_terms * (first_term + last_term)) / 2

/-- The total number of logs in the stack is 120 -/
theorem log_stack_total : log_stack_sum = 120 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_total_l3801_380125


namespace NUMINAMATH_CALUDE_three_five_two_takes_five_steps_l3801_380143

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of the process: reverse, add 3, then add to original -/
def step (n : ℕ) : ℕ := n + (reverseNum n + 3)

/-- Counts the number of steps to reach a palindrome -/
def stepsToBecomePalindrome (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem three_five_two_takes_five_steps :
  352 ≥ 100 ∧ 352 ≤ 400 ∧ 
  ¬isPalindrome 352 ∧
  stepsToBecomePalindrome 352 = 5 := by sorry

end NUMINAMATH_CALUDE_three_five_two_takes_five_steps_l3801_380143


namespace NUMINAMATH_CALUDE_right_triangle_cone_rotation_l3801_380110

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 800π cm³ and rotating about leg b produces a cone with volume 1920π cm³,
    then the hypotenuse length is 26 cm. -/
theorem right_triangle_cone_rotation (a b : ℝ) :
  a > 0 ∧ b > 0 →
  (1 / 3 : ℝ) * Real.pi * a * b^2 = 800 * Real.pi →
  (1 / 3 : ℝ) * Real.pi * b * a^2 = 1920 * Real.pi →
  Real.sqrt (a^2 + b^2) = 26 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_cone_rotation_l3801_380110


namespace NUMINAMATH_CALUDE_end_of_year_dance_attendance_l3801_380188

theorem end_of_year_dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ boys = (girls - 1) + 8 → boys = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_end_of_year_dance_attendance_l3801_380188


namespace NUMINAMATH_CALUDE_factorization_equality_l3801_380124

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3801_380124


namespace NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l3801_380181

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l3801_380181


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3801_380154

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 5| + |x + 3|

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -4 ∨ x ≥ 6}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 10} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3801_380154


namespace NUMINAMATH_CALUDE_g_at_negative_three_l3801_380146

theorem g_at_negative_three (g : ℝ → ℝ) :
  (∀ x, g x = 10 * x^3 - 7 * x^2 - 5 * x + 6) →
  g (-3) = -312 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l3801_380146


namespace NUMINAMATH_CALUDE_only_nice_number_l3801_380197

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def is_valid_sequence (s : ℕ → ℕ × ℕ) : Prop :=
  s 1 = (1, 3) ∧
  ∀ k, (s (k + 1) = (P (s k).1, Q (s k).2)) ∨ (s (k + 1) = (Q (s k).1, P (s k).2))

def is_nice (n : ℕ) : Prop :=
  ∃ s, is_valid_sequence s ∧ (s n).1 = (s n).2

theorem only_nice_number : ∀ n : ℕ, is_nice n ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_only_nice_number_l3801_380197


namespace NUMINAMATH_CALUDE_rabbit_distribution_count_l3801_380102

/-- Represents the number of stores -/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits -/
def num_parents : ℕ := 2

/-- Represents the number of child rabbits -/
def num_children : ℕ := 4

/-- Represents the total number of rabbits -/
def total_rabbits : ℕ := num_parents + num_children

/-- 
Represents the number of ways to distribute rabbits to stores 
such that no store has both a parent and a child 
-/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_count : distribution_ways = 380 := by sorry

end NUMINAMATH_CALUDE_rabbit_distribution_count_l3801_380102


namespace NUMINAMATH_CALUDE_election_votes_l3801_380179

theorem election_votes (candidate1_percentage : ℚ) (candidate2_votes : ℕ) :
  candidate1_percentage = 60 / 100 →
  candidate2_votes = 240 →
  ∃ total_votes : ℕ,
    candidate1_percentage * total_votes = total_votes - candidate2_votes ∧
    total_votes = 600 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l3801_380179


namespace NUMINAMATH_CALUDE_radius_of_special_polygon_l3801_380115

/-- A regular polygon with the given properties -/
structure RegularPolygon where
  side_length : ℝ
  interior_angle_sum : ℝ
  exterior_angle_sum : ℝ

/-- The radius of a regular polygon -/
def radius (p : RegularPolygon) : ℝ := sorry

/-- The theorem to be proved -/
theorem radius_of_special_polygon :
  ∀ (p : RegularPolygon),
    p.side_length = 2 →
    p.interior_angle_sum = 2 * p.exterior_angle_sum →
    radius p = 2 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_special_polygon_l3801_380115


namespace NUMINAMATH_CALUDE_one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l3801_380176

/-- The Cantor set constructed by repeatedly removing the middle third of each interval --/
def CantorSet : Set ℝ :=
  sorry

/-- The nth iteration in the Cantor set construction --/
def CantorIteration (n : ℕ) : Set (Set ℝ) :=
  sorry

/-- The length of the nth iteration in the Cantor set construction --/
def CantorIterationLength (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating that 1/3 and 3/81 belong to the Cantor set --/
theorem one_third_and_three_eightyone_in_cantor :
  (1/3 : ℝ) ∈ CantorSet ∧ (3/81 : ℝ) ∈ CantorSet :=
sorry

/-- Theorem stating the length of the nth iteration in the Cantor set construction --/
theorem cantor_iteration_length (n : ℕ) :
  CantorIterationLength n = (2/3 : ℝ) ^ (n - 1) :=
sorry

end NUMINAMATH_CALUDE_one_third_and_three_eightyone_in_cantor_cantor_iteration_length_l3801_380176


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3801_380194

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3801_380194


namespace NUMINAMATH_CALUDE_walking_speed_l3801_380144

/-- 
Given that:
- Jack's speed is (x^2 - 13x - 30) miles per hour
- Jill covers (x^2 - 6x - 91) miles in (x + 7) hours
- Jack and Jill walk at the same rate

Prove that their speed is 4 miles per hour
-/
theorem walking_speed (x : ℝ) 
  (h1 : x ≠ -7)  -- Assumption to avoid division by zero
  (h2 : x > 0)   -- Assumption for positive speed
  (h3 : (x^2 - 6*x - 91) / (x + 7) = x^2 - 13*x - 30) :  -- Jack and Jill walk at the same rate
  x^2 - 13*x - 30 = 4 := by sorry

end NUMINAMATH_CALUDE_walking_speed_l3801_380144


namespace NUMINAMATH_CALUDE_inscribed_cube_diagonal_l3801_380177

/-- The diagonal length of a cube inscribed in a sphere of radius R is 2R -/
theorem inscribed_cube_diagonal (R : ℝ) (R_pos : R > 0) :
  ∃ (cube : Set (Fin 3 → ℝ)), 
    (∀ p ∈ cube, ‖p‖ = R) ∧ 
    (∃ (d : Fin 3 → ℝ), d ∈ cube ∧ ‖d‖ = 2*R) :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_diagonal_l3801_380177


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3801_380149

theorem consecutive_even_integers_sum (x : ℝ) :
  (x - 2) * x * (x + 2) = 48 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 6 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3801_380149


namespace NUMINAMATH_CALUDE_midpoints_form_equilateral_triangle_l3801_380190

/-- A hexagon inscribed in a unit circle with alternate sides of length 1 -/
structure InscribedHexagon where
  /-- The vertices of the hexagon -/
  vertices : Fin 6 → ℝ × ℝ
  /-- The hexagon is inscribed in a unit circle -/
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1
  /-- Alternate sides have length 1 -/
  alt_sides_length : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = 1 ∨ 
                           dist (vertices ((i + 1) % 6)) (vertices ((i + 2) % 6)) = 1

/-- The midpoints of the three sides that don't have length 1 -/
def midpoints (h : InscribedHexagon) : Fin 3 → ℝ × ℝ := sorry

/-- The theorem statement -/
theorem midpoints_form_equilateral_triangle (h : InscribedHexagon) : 
  ∀ i j, dist (midpoints h i) (midpoints h j) = dist (midpoints h 0) (midpoints h 1) :=
sorry

end NUMINAMATH_CALUDE_midpoints_form_equilateral_triangle_l3801_380190


namespace NUMINAMATH_CALUDE_inequality_solution_l3801_380183

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 < x → x < y → f y < f x
axiom f_at_neg_three : f (-3) = 1

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 3}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x < 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3801_380183


namespace NUMINAMATH_CALUDE_files_per_folder_l3801_380118

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 43 →
  deleted_files = 31 →
  num_folders = 2 →
  num_folders > 0 →
  ∃ (files_per_folder : ℕ),
    files_per_folder * num_folders = initial_files - deleted_files ∧
    files_per_folder = 6 :=
by sorry

end NUMINAMATH_CALUDE_files_per_folder_l3801_380118


namespace NUMINAMATH_CALUDE_terms_before_four_l3801_380185

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem terms_before_four (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 100 ∧ d = -6 ∧ arithmetic_sequence a₁ d n = 4 → n - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_four_l3801_380185


namespace NUMINAMATH_CALUDE_square_not_partitionable_into_10deg_isosceles_triangles_l3801_380134

-- Define a square
def Square : Type := Unit

-- Define an isosceles triangle with a 10° vertex angle
def IsoscelesTriangle10Deg : Type := Unit

-- Define a partition of a square
def Partition (s : Square) : Type := List IsoscelesTriangle10Deg

-- Theorem statement
theorem square_not_partitionable_into_10deg_isosceles_triangles :
  ¬∃ (s : Square) (p : Partition s), p.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_not_partitionable_into_10deg_isosceles_triangles_l3801_380134


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l3801_380106

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l3801_380106


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_monic_cubic_integer_coeffs_l3801_380140

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 13 = 0 := by sorry

theorem monic_cubic_integer_coeffs :
  ∃ (a b c : ℤ), ∀ (x : ℝ), x^3 - 6*x^2 + 12*x - 13 = x^3 + a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_monic_cubic_integer_coeffs_l3801_380140


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3801_380199

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → a^2 + b^2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3801_380199


namespace NUMINAMATH_CALUDE_average_speed_three_sections_l3801_380175

/-- The average speed of a person traveling on a 1 km street divided into three equal sections,
    with speeds of 4 km/h, 10 km/h, and 6 km/h in each section respectively. -/
theorem average_speed_three_sections (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 1 →
  speed1 = 4 →
  speed2 = 10 →
  speed3 = 6 →
  let section_distance := total_distance / 3
  let time1 := section_distance / speed1
  let time2 := section_distance / speed2
  let time3 := section_distance / speed3
  let total_time := time1 + time2 + time3
  total_distance / total_time = 180 / 31 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_three_sections_l3801_380175


namespace NUMINAMATH_CALUDE_tetrahedron_special_points_l3801_380105

-- Define the tetrahedron P-ABC
structure Tetrahedron :=
  (P A B C : EuclideanSpace ℝ (Fin 3))

-- Define the projection O of P onto the base plane ABC
def projection (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the property of equal angles between lateral edges and base plane
def equal_lateral_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the property of mutually perpendicular lateral edges
def perpendicular_lateral_edges (t : Tetrahedron) : Prop := sorry

-- Define the property of equal angles between side faces and base plane
def equal_face_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the circumcenter of a triangle
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the orthocenter of a triangle
def is_orthocenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the incenter of a triangle
def is_incenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Theorem statements
theorem tetrahedron_special_points (t : Tetrahedron) :
  (equal_lateral_base_angles t → is_circumcenter (projection t) t.A t.B t.C) ∧
  (perpendicular_lateral_edges t → is_orthocenter (projection t) t.A t.B t.C) ∧
  (equal_face_base_angles t → is_incenter (projection t) t.A t.B t.C) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_special_points_l3801_380105


namespace NUMINAMATH_CALUDE_sphere_only_identical_views_l3801_380158

-- Define the possible geometric bodies
inductive GeometricBody
  | Sphere
  | Cube
  | RegularTetrahedron

-- Define a function that checks if all views are identical
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_identical_views_l3801_380158


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_product_60_l3801_380100

/-- The number of distinct prime factors of the product of divisors of 60 -/
theorem distinct_prime_factors_of_divisor_product_60 : ∃ (B : ℕ), 
  (∀ d : ℕ, d ∣ 60 → d ∣ B) ∧ 
  (∀ n : ℕ, (∀ d : ℕ, d ∣ 60 → d ∣ n) → B ∣ n) ∧
  (Nat.card {p : ℕ | Nat.Prime p ∧ p ∣ B} = 3) :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_product_60_l3801_380100


namespace NUMINAMATH_CALUDE_function_and_range_proof_l3801_380150

-- Define the function f
def f (x : ℝ) (b c : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem function_and_range_proof :
  ∀ b c : ℝ,
  (∀ x : ℝ, f x b c < 0 ↔ 1 < x ∧ x < 5) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → ∃ t : ℝ, f x b c ≤ 2 + t) →
  (∀ x : ℝ, f x b c = 2 * x^2 - 12 * x + 10) ∧
  (∀ t : ℝ, t ≥ -10 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ f x b c ≤ 2 + t) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_proof_l3801_380150


namespace NUMINAMATH_CALUDE_limo_cost_per_hour_l3801_380116

/-- Calculates the cost of a limo per hour given prom expenses -/
theorem limo_cost_per_hour 
  (ticket_cost : ℝ) 
  (dinner_cost : ℝ) 
  (tip_percentage : ℝ) 
  (limo_hours : ℝ) 
  (total_cost : ℝ) 
  (h1 : ticket_cost = 100)
  (h2 : dinner_cost = 120)
  (h3 : tip_percentage = 0.3)
  (h4 : limo_hours = 6)
  (h5 : total_cost = 836) :
  (total_cost - (2 * ticket_cost + dinner_cost + tip_percentage * dinner_cost)) / limo_hours = 80 :=
by sorry

end NUMINAMATH_CALUDE_limo_cost_per_hour_l3801_380116


namespace NUMINAMATH_CALUDE_distance_difference_l3801_380156

/-- The width of the streets in Tranquility Town -/
def street_width : ℝ := 30

/-- The length of the rectangular block -/
def block_length : ℝ := 500

/-- The width of the rectangular block -/
def block_width : ℝ := 300

/-- The perimeter of Alice's path -/
def alice_perimeter : ℝ := 2 * ((block_length + street_width) + (block_width + street_width))

/-- The perimeter of Bob's path -/
def bob_perimeter : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- The theorem stating the difference in distance walked -/
theorem distance_difference : bob_perimeter - alice_perimeter = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3801_380156


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3801_380174

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) :
  (x - Int.floor x) + (Int.floor x + 1) + x = 3 * ((Int.floor x + 1)) →
  x = -2 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3801_380174


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l3801_380155

theorem units_digit_sum_powers : (19^89 + 89^19) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l3801_380155


namespace NUMINAMATH_CALUDE_gcd_lcm_pairs_l3801_380137

theorem gcd_lcm_pairs :
  (Nat.gcd 6 12 = 6 ∧ Nat.lcm 6 12 = 12) ∧
  (Nat.gcd 7 8 = 1 ∧ Nat.lcm 7 8 = 56) ∧
  (Nat.gcd 15 20 = 5 ∧ Nat.lcm 15 20 = 60) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_pairs_l3801_380137


namespace NUMINAMATH_CALUDE_expression_evaluation_l3801_380136

theorem expression_evaluation (x y z : ℝ) : 
  (x - (y + z)) - ((x + y) - 2*z) = -2*y - 3*z := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3801_380136


namespace NUMINAMATH_CALUDE_fruit_juice_volume_l3801_380113

/-- Proves that the volume of fruit juice in Carrie's punch is 40 oz -/
theorem fruit_juice_volume (total_punch : ℕ) (mountain_dew : ℕ) (ice : ℕ) :
  total_punch = 140 ∧ mountain_dew = 72 ∧ ice = 28 →
  ∃ (fruit_juice : ℕ), total_punch = mountain_dew + ice + fruit_juice ∧ fruit_juice = 40 := by
sorry

end NUMINAMATH_CALUDE_fruit_juice_volume_l3801_380113
