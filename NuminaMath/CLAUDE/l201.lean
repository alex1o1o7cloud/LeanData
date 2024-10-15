import Mathlib

namespace NUMINAMATH_CALUDE_top_triangle_number_l201_20120

/-- Represents the shape of a cell in the diagram -/
inductive Shape
| Circle
| Triangle
| Hexagon

/-- The sum of numbers in each shape -/
def sum_of_shape (s : Shape) : ℕ :=
  match s with
  | Shape.Circle => 10
  | Shape.Triangle => 15
  | Shape.Hexagon => 30

/-- The total number of cells in the diagram -/
def total_cells : ℕ := 9

/-- The set of numbers used in the diagram -/
def number_set : Finset ℕ := Finset.range 9

/-- The theorem stating the possible numbers in the top triangle -/
theorem top_triangle_number :
  ∃ (n : ℕ), n ∈ number_set ∧ n ≥ 8 ∧ n ≤ 9 ∧
  (∃ (a b : ℕ), a ∈ number_set ∧ b ∈ number_set ∧ a + b + n = sum_of_shape Shape.Triangle) :=
sorry

end NUMINAMATH_CALUDE_top_triangle_number_l201_20120


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l201_20152

/-- Given a bag marked at $240 with a 50% discount, prove that the discounted price is $120. -/
theorem discounted_price_calculation (marked_price : ℝ) (discount_rate : ℝ) :
  marked_price = 240 →
  discount_rate = 0.5 →
  marked_price * (1 - discount_rate) = 120 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l201_20152


namespace NUMINAMATH_CALUDE_equation_solution_l201_20155

theorem equation_solution (t : ℝ) : 
  (5 * 3^t + Real.sqrt (25 * 9^t) = 50) ↔ (t = Real.log 5 / Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l201_20155


namespace NUMINAMATH_CALUDE_non_positive_sequence_l201_20190

theorem non_positive_sequence (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_ineq : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) :
  ∀ i : ℕ, i ≤ n → a i ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_non_positive_sequence_l201_20190


namespace NUMINAMATH_CALUDE_simplify_expression_l201_20143

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l201_20143


namespace NUMINAMATH_CALUDE_eating_relationship_l201_20139

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x | a * x^2 = 1}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem eating_relationship (a : ℝ) : 
  (a ≥ 0) → (full_eating A (B a) ∨ partial_eating A (B a)) ↔ a ∈ ({0, 1, 4} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_eating_relationship_l201_20139


namespace NUMINAMATH_CALUDE_nancy_shoe_count_nancy_has_168_shoes_l201_20179

/-- Calculates the total number of individual shoes Nancy has given her shoe collection. -/
theorem nancy_shoe_count (boots : ℕ) (slippers : ℕ) (heels : ℕ) : ℕ :=
  let total_pairs := boots + slippers + heels
  2 * total_pairs

/-- Proves that Nancy has 168 individual shoes given the conditions of her shoe collection. -/
theorem nancy_has_168_shoes : nancy_shoe_count 6 15 63 = 168 := by
  sorry

#check nancy_has_168_shoes

end NUMINAMATH_CALUDE_nancy_shoe_count_nancy_has_168_shoes_l201_20179


namespace NUMINAMATH_CALUDE_karen_start_time_l201_20131

/-- Proves that Karen starts 4 minutes late in the car race --/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) 
  (h1 : karen_speed = 60)
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) :
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_karen_start_time_l201_20131


namespace NUMINAMATH_CALUDE_noah_sticker_count_l201_20150

/-- Given the number of stickers for Kristoff, calculate the number of stickers Noah has -/
def noahs_stickers (kristoff : ℕ) : ℕ :=
  let riku : ℕ := 25 * kristoff
  let lila : ℕ := 2 * (kristoff + riku)
  kristoff * lila - 3

theorem noah_sticker_count : noahs_stickers 85 = 375697 := by
  sorry

end NUMINAMATH_CALUDE_noah_sticker_count_l201_20150


namespace NUMINAMATH_CALUDE_intersecting_chords_probability_2023_l201_20114

/-- Given a circle with 2023 evenly spaced points, this function calculates
    the probability that when selecting four distinct points A, B, C, and D randomly,
    chord AB intersects chord CD and chord AC intersects chord BD. -/
def intersecting_chords_probability (n : ℕ) : ℚ :=
  if n = 2023 then 1/6 else 0

/-- Theorem stating that the probability of the specific chord intersection
    scenario for 2023 points is 1/6. -/
theorem intersecting_chords_probability_2023 :
  intersecting_chords_probability 2023 = 1/6 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_probability_2023_l201_20114


namespace NUMINAMATH_CALUDE_largest_n_two_solutions_exceed_two_l201_20141

/-- The cubic polynomial in question -/
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^3 - (n + 9 : ℝ) * x^2 + (2 * n^2 - 3 * n - 34 : ℝ) * x + 2 * (n - 4) * (n + 3 : ℝ)

/-- The statement that 8 is the largest integer for which the equation has two solutions > 2 -/
theorem largest_n_two_solutions_exceed_two :
  ∀ n : ℤ, (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ x ≠ y ∧ f n x = 0 ∧ f n y = 0) → n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_n_two_solutions_exceed_two_l201_20141


namespace NUMINAMATH_CALUDE_twins_age_problem_l201_20160

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 9 → age = 4 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l201_20160


namespace NUMINAMATH_CALUDE_youtube_video_dislikes_l201_20100

theorem youtube_video_dislikes :
  let initial_likes : ℕ := 5000
  let initial_dislikes : ℕ := (initial_likes / 3) + 50
  let likes_increase : ℕ := 2000
  let dislikes_increase : ℕ := 400
  let new_likes : ℕ := initial_likes + likes_increase
  let new_dislikes : ℕ := initial_dislikes + dislikes_increase
  let doubled_new_likes : ℕ := 2 * new_likes
  doubled_new_likes - new_dislikes = 11983 ∧ new_dislikes = 2017 :=
by sorry


end NUMINAMATH_CALUDE_youtube_video_dislikes_l201_20100


namespace NUMINAMATH_CALUDE_sequence_equals_index_l201_20108

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a n > 0) ∧
  (∀ n m : ℕ, n ≥ 1 → m ≥ 1 → n < m → a n < a m) ∧
  (∀ n : ℕ, n ≥ 1 → a (2*n) = a n + n) ∧
  (∀ n : ℕ, n ≥ 1 → is_prime (a n) → is_prime n)

theorem sequence_equals_index (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_equals_index_l201_20108


namespace NUMINAMATH_CALUDE_second_quadrant_fraction_negative_l201_20145

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that for a point in the second quadrant, a/b < 0 -/
theorem second_quadrant_fraction_negative (p : Point) :
  is_in_second_quadrant p → p.x / p.y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_second_quadrant_fraction_negative_l201_20145


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_t_value_l201_20140

theorem polynomial_factor_implies_t_value :
  ∀ t : ℤ,
  (∃ a b : ℤ, ∀ x : ℤ, x^3 - x^2 - 7*x + t = (x + 1) * (x^2 + a*x + b)) →
  t = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_t_value_l201_20140


namespace NUMINAMATH_CALUDE_inverse_proportion_function_l201_20165

/-- 
If the inverse proportion function y = m/x passes through the point (m, m/8),
then the function can be expressed as y = 8/x.
-/
theorem inverse_proportion_function (m : ℝ) (h : m ≠ 0) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = m / x) ∧ f m = m / 8) → 
  (∃ (g : ℝ → ℝ), ∀ x, x ≠ 0 → g x = 8 / x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_function_l201_20165


namespace NUMINAMATH_CALUDE_initial_apples_in_pile_l201_20192

def apple_pile (initial : ℕ) (added : ℕ) (final : ℕ) : Prop :=
  initial + added = final

def package_size : ℕ := 11

theorem initial_apples_in_pile : 
  ∃ (initial : ℕ), apple_pile initial 5 13 ∧ initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_in_pile_l201_20192


namespace NUMINAMATH_CALUDE_total_campers_rowing_l201_20116

theorem total_campers_rowing (morning afternoon evening : ℕ) 
  (h1 : morning = 36) 
  (h2 : afternoon = 13) 
  (h3 : evening = 49) : 
  morning + afternoon + evening = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l201_20116


namespace NUMINAMATH_CALUDE_class_average_problem_l201_20105

theorem class_average_problem (percent_high : Real) (percent_mid : Real) (percent_low : Real)
  (score_high : Real) (score_low : Real) (overall_average : Real) :
  percent_high = 15 →
  percent_mid = 50 →
  percent_low = 35 →
  score_high = 100 →
  score_low = 63 →
  overall_average = 76.05 →
  (percent_high * score_high + percent_mid * ((percent_high * score_high + percent_mid * X + percent_low * score_low) / 100) + percent_low * score_low) / 100 = overall_average →
  X = 78 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l201_20105


namespace NUMINAMATH_CALUDE_farm_animals_after_addition_l201_20122

/-- Represents the farm with its animals -/
structure Farm :=
  (cows : ℕ)
  (pigs : ℕ)
  (goats : ℕ)

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.cows + f.pigs + f.goats

/-- Adds new animals to the farm -/
def Farm.add (f : Farm) (new_cows new_pigs new_goats : ℕ) : Farm :=
  { cows := f.cows + new_cows,
    pigs := f.pigs + new_pigs,
    goats := f.goats + new_goats }

/-- Theorem: The farm will have 21 animals after adding the new ones -/
theorem farm_animals_after_addition :
  let initial_farm := Farm.mk 2 3 6
  let final_farm := initial_farm.add 3 5 2
  final_farm.total = 21 := by sorry

end NUMINAMATH_CALUDE_farm_animals_after_addition_l201_20122


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l201_20111

/-- The sum of consecutive integers from a to (a + n - 1) -/
def sumConsecutive (a : ℤ) (n : ℕ) : ℤ := n * (2 * a + n - 1) / 2

/-- The target sum we want to achieve -/
def targetSum : ℤ := 2015

/-- Theorem stating that the maximum number of consecutive integers summing to 2015 is 4030 -/
theorem max_consecutive_sum :
  (∃ a : ℤ, sumConsecutive a 4030 = targetSum) ∧
  (∀ n : ℕ, n > 4030 → ∀ a : ℤ, sumConsecutive a n ≠ targetSum) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l201_20111


namespace NUMINAMATH_CALUDE_binomial_expansion_largest_coefficient_l201_20188

theorem binomial_expansion_largest_coefficient (n : ℕ) : 
  (∃ k, k = 5 ∧ 
    (∀ j, j ≠ k → Nat.choose n k > Nat.choose n j) ∧
    (∀ j, j < k → Nat.choose n j < Nat.choose n (j+1)) ∧
    (∀ j, k < j ∧ j ≤ n → Nat.choose n j < Nat.choose n (j-1))) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_largest_coefficient_l201_20188


namespace NUMINAMATH_CALUDE_second_polygon_sides_l201_20135

/-- 
Given two regular polygons with the same perimeter, where the first has 50 sides 
and a side length three times as long as the second, prove that the second polygon has 150 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l201_20135


namespace NUMINAMATH_CALUDE_meal_pass_cost_sally_meal_pass_cost_l201_20162

/-- Calculates the cost of a meal pass for Sally's trip to Sea World --/
theorem meal_pass_cost (savings : ℕ) (parking : ℕ) (entrance : ℕ) (distance : ℕ) (mpg : ℕ) 
  (gas_price : ℕ) (additional_savings : ℕ) : ℕ :=
  let round_trip := 2 * distance
  let gas_needed := round_trip / mpg
  let gas_cost := gas_needed * gas_price
  let known_costs := parking + entrance + gas_cost
  let remaining_costs := known_costs - savings
  additional_savings - remaining_costs

/-- The meal pass for Sally's trip to Sea World costs $25 --/
theorem sally_meal_pass_cost : 
  meal_pass_cost 28 10 55 165 30 3 95 = 25 := by
  sorry

end NUMINAMATH_CALUDE_meal_pass_cost_sally_meal_pass_cost_l201_20162


namespace NUMINAMATH_CALUDE_spring_sales_five_million_l201_20183

/-- Represents the annual pizza sales of a restaurant in millions -/
def annual_sales : ℝ := 20

/-- Represents the winter pizza sales of the restaurant in millions -/
def winter_sales : ℝ := 4

/-- Represents the percentage of annual sales that occur in winter -/
def winter_percentage : ℝ := 0.20

/-- Represents the percentage of annual sales that occur in summer -/
def summer_percentage : ℝ := 0.30

/-- Represents the percentage of annual sales that occur in fall -/
def fall_percentage : ℝ := 0.25

/-- Theorem stating that spring sales are 5 million pizzas -/
theorem spring_sales_five_million :
  winter_sales = winter_percentage * annual_sales →
  ∃ (spring_percentage : ℝ),
    spring_percentage + winter_percentage + summer_percentage + fall_percentage = 1 ∧
    spring_percentage * annual_sales = 5 := by
  sorry

end NUMINAMATH_CALUDE_spring_sales_five_million_l201_20183


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l201_20156

def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / m - y^2 / (m + 1) = 1 → 
    (m > 0 ∧ m + 1 > 0) ∨ (m < 0 ∧ m + 1 < 0)

theorem hyperbola_m_range :
  {m : ℝ | is_hyperbola m} = {m | m < -1 ∨ m > 0} := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l201_20156


namespace NUMINAMATH_CALUDE_linear_function_fixed_point_l201_20180

theorem linear_function_fixed_point (k : ℝ) : (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_fixed_point_l201_20180


namespace NUMINAMATH_CALUDE_remainder_50_power_50_mod_7_l201_20176

theorem remainder_50_power_50_mod_7 : 50^50 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_50_power_50_mod_7_l201_20176


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l201_20101

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x y : ℝ,
    (|x - 3| = 15 ∧ |y - 3| = 15) ∧
    x ≠ y ∧
    d = |x - y| ∧
    d = 30

-- The proof is omitted
theorem absolute_value_equation_solution_difference_proof :
  ∃ d : ℝ, absolute_value_equation_solution_difference d :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l201_20101


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l201_20173

theorem arcsin_sin_eq_x_div_3 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 * Real.pi / 2) (3 * Real.pi / 2) ∧ 
    Real.arcsin (Real.sin x) = x / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l201_20173


namespace NUMINAMATH_CALUDE_quadratic_root_on_line_l201_20107

/-- A root of a quadratic equation lies on a corresponding line in the p-q plane. -/
theorem quadratic_root_on_line (p q x₀ : ℝ) : 
  x₀^2 + p * x₀ + q = 0 → q = -x₀ * p - x₀^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_on_line_l201_20107


namespace NUMINAMATH_CALUDE_complementary_sets_count_l201_20137

/-- Represents a card with four attributes -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  pattern : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if three cards form a complementary set -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem complementary_sets_count :
  deck.card = 81 ∧ (∀ c1 c2 : Card, c1 ∈ deck → c2 ∈ deck → c1 = c2 ∨ c1.shape ≠ c2.shape ∨ c1.color ≠ c2.color ∨ c1.shade ≠ c2.shade ∨ c1.pattern ≠ c2.pattern) →
  complementarySets.card = 5400 := by
  sorry

end NUMINAMATH_CALUDE_complementary_sets_count_l201_20137


namespace NUMINAMATH_CALUDE_jason_toys_count_l201_20121

theorem jason_toys_count :
  ∀ (rachel_toys john_toys jason_toys : ℕ),
    rachel_toys = 1 →
    john_toys = rachel_toys + 6 →
    jason_toys = 3 * john_toys →
    jason_toys = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_jason_toys_count_l201_20121


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l201_20146

theorem algebraic_expression_value (x : ℝ) : 
  4 * x^2 - 2 * x + 3 = 11 → 2 * x^2 - x - 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l201_20146


namespace NUMINAMATH_CALUDE_domain_of_f_squared_l201_20199

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_of_f_squared :
  {x : ℝ | ∃ y ∈ dom_f, x^2 = y} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_squared_l201_20199


namespace NUMINAMATH_CALUDE_quadratic_composition_theorem_l201_20186

/-- A unitary quadratic trinomial -/
structure UnitaryQuadratic where
  b : ℝ
  c : ℝ

/-- Evaluate a unitary quadratic trinomial at a point -/
def evaluate (f : UnitaryQuadratic) (x : ℝ) : ℝ :=
  x^2 + f.b * x + f.c

/-- Composition of two unitary quadratic trinomials -/
def compose (f g : UnitaryQuadratic) : UnitaryQuadratic :=
  { b := g.b^2 + f.b * (1 + g.b) + g.c * f.b
    c := g.c^2 + f.b * g.c + f.c }

/-- A polynomial has no real roots -/
def hasNoRealRoots (f : UnitaryQuadratic) : Prop :=
  ∀ x : ℝ, evaluate f x ≠ 0

theorem quadratic_composition_theorem (f g : UnitaryQuadratic) 
    (h1 : hasNoRealRoots (compose f g))
    (h2 : hasNoRealRoots (compose g f)) :
    hasNoRealRoots (compose f f) ∨ hasNoRealRoots (compose g g) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_composition_theorem_l201_20186


namespace NUMINAMATH_CALUDE_range_of_a_l201_20147

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l201_20147


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l201_20170

/-- The maximum marks of an exam, given the conditions from the problem -/
def maximum_marks : ℕ :=
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  500

/-- Theorem stating that the maximum marks of the exam is 500 -/
theorem exam_maximum_marks :
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  maximum_marks = 500 := by
  sorry

#check exam_maximum_marks

end NUMINAMATH_CALUDE_exam_maximum_marks_l201_20170


namespace NUMINAMATH_CALUDE_modifiedLucas_100th_term_mod_10_l201_20113

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => (modifiedLucas n + modifiedLucas (n + 1)) % 10

theorem modifiedLucas_100th_term_mod_10 :
  modifiedLucas 99 % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modifiedLucas_100th_term_mod_10_l201_20113


namespace NUMINAMATH_CALUDE_least_possible_x_l201_20112

theorem least_possible_x (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m : ℤ, y = 2 * m + 1) →  -- y is odd
  (∃ n : ℤ, z = 2 * n + 1) →  -- z is odd
  y - x > 5 →
  z - x ≥ 9 →
  (∀ w : ℤ, (∃ j : ℤ, w = 2 * j) → w ≥ x) →
  x = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_x_l201_20112


namespace NUMINAMATH_CALUDE_total_weight_is_103_2_l201_20195

/-- The total weight of all books owned by Sandy, Benny, and Tim -/
def total_weight : ℝ :=
  let sandy_books := 10
  let sandy_weight := 1.5
  let benny_books := 24
  let benny_weight := 1.2
  let tim_books := 33
  let tim_weight := 1.8
  sandy_books * sandy_weight + benny_books * benny_weight + tim_books * tim_weight

/-- Theorem stating that the total weight of all books is 103.2 pounds -/
theorem total_weight_is_103_2 : total_weight = 103.2 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_103_2_l201_20195


namespace NUMINAMATH_CALUDE_laptop_price_l201_20106

theorem laptop_price : ∃ (S : ℝ), S = 733 ∧ 
  (0.80 * S - 120 = 0.65 * S - 10) := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l201_20106


namespace NUMINAMATH_CALUDE_constant_term_proof_l201_20197

/-- The constant term in the expansion of (x^2 + 3)(x - 2/x)^6 -/
def constantTerm : ℤ := -240

/-- The expression (x^2 + 3)(x - 2/x)^6 -/
def expression (x : ℚ) : ℚ := (x^2 + 3) * (x - 2/x)^6

theorem constant_term_proof :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expression x) ∧
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c : ℤ) = constantTerm :=
sorry

end NUMINAMATH_CALUDE_constant_term_proof_l201_20197


namespace NUMINAMATH_CALUDE_nikolai_faster_l201_20196

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- The race parameters -/
def turning_point : ℕ := 2000

/-- Gennady's characteristics -/
def gennady : Goat := ⟨"Gennady", 6⟩

/-- Nikolai's characteristics -/
def nikolai : Goat := ⟨"Nikolai", 4⟩

/-- Calculates the number of jumps needed to reach the turning point -/
def jumps_to_turning_point (g : Goat) : ℕ :=
  (turning_point + g.jump_distance - 1) / g.jump_distance

/-- Calculates the total distance traveled to the turning point -/
def distance_to_turning_point (g : Goat) : ℕ :=
  (jumps_to_turning_point g) * g.jump_distance

/-- Theorem stating that Nikolai completes the journey faster -/
theorem nikolai_faster : 
  distance_to_turning_point nikolai < distance_to_turning_point gennady :=
sorry

end NUMINAMATH_CALUDE_nikolai_faster_l201_20196


namespace NUMINAMATH_CALUDE_problem_statement_l201_20110

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1/a^2 + 1/b^2 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l201_20110


namespace NUMINAMATH_CALUDE_subset_implies_a_zero_l201_20184

theorem subset_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a = 0 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_zero_l201_20184


namespace NUMINAMATH_CALUDE_cartesian_equation_chord_length_l201_20157

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 4 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of curve C
theorem cartesian_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 4*x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁ t₂, line_equation t₁ A.1 A.2 ∧ line_equation t₂ B.1 B.2 ∧
   A.2^2 = 4*A.1 ∧ B.2^2 = 4*B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 7 / 3 :=
sorry

end NUMINAMATH_CALUDE_cartesian_equation_chord_length_l201_20157


namespace NUMINAMATH_CALUDE_ellipse_intersection_max_y_intercept_l201_20198

/-- An ellipse with major axis 2√2 times the minor axis, passing through (2, √2/2) --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a = 2 * Real.sqrt 2 * b
  h4 : (2 / a)^2 + ((Real.sqrt 2 / 2) / b)^2 = 1

/-- A line intersecting the ellipse at two points --/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h1 : k ≠ 0  -- Line is not parallel to coordinate axes

/-- The distance between intersection points is 2√2 --/
def intersection_distance (e : Ellipse) (l : IntersectingLine e) : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1^2 / e.a^2) + ((l.k * x1 + l.m)^2 / e.b^2) = 1 ∧
    (x2^2 / e.a^2) + ((l.k * x2 + l.m)^2 / e.b^2) = 1 ∧
    (x2 - x1)^2 + (l.k * (x2 - x1))^2 = 8

/-- The theorem to be proved --/
theorem ellipse_intersection_max_y_intercept (e : Ellipse) :
  ∃ (max_m : ℝ), max_m = Real.sqrt 14 - Real.sqrt 7 ∧
  ∀ (l : IntersectingLine e), intersection_distance e l →
    l.m ≤ max_m ∧
    ∃ (l' : IntersectingLine e), intersection_distance e l' ∧ l'.m = max_m :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_max_y_intercept_l201_20198


namespace NUMINAMATH_CALUDE_inequality_solution_set_l201_20164

-- Define the inequality
def inequality (x : ℝ) : Prop := abs ((2 - x) / x) > (x - 2) / x

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l201_20164


namespace NUMINAMATH_CALUDE_garrett_roses_count_l201_20109

/-- Mrs. Santiago's red roses -/
def santiago_roses : ℕ := 58

/-- The difference between Mrs. Santiago's and Mrs. Garrett's red roses -/
def rose_difference : ℕ := 34

/-- Mrs. Garrett's red roses -/
def garrett_roses : ℕ := santiago_roses - rose_difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_roses_count_l201_20109


namespace NUMINAMATH_CALUDE_both_are_dwarves_l201_20126

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| AboutGold : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define the truth value of a statement given the speaker and the statement type
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- Define A's statement
def A_statement : Statement := Statement.AboutGold

-- Define B's statement about A
def B_statement (A_type : Inhabitant) : Statement :=
  match A_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (A_type B_type : Inhabitant),
    A_type = Inhabitant.Dwarf ∧
    B_type = Inhabitant.Dwarf ∧
    isTruthful A_type A_statement = False ∧
    isTruthful B_type (B_statement A_type) = True :=
  sorry


end NUMINAMATH_CALUDE_both_are_dwarves_l201_20126


namespace NUMINAMATH_CALUDE_amandas_car_round_trip_time_l201_20168

/-- Given that:
    1. The bus takes 40 minutes to drive 80 miles to the beach.
    2. Amanda's car takes five fewer minutes than the bus for the same trip.
    Prove that Amanda's car takes 70 minutes to make a round trip to the beach. -/
theorem amandas_car_round_trip_time :
  let bus_time : ℕ := 40
  let car_time_difference : ℕ := 5
  let car_one_way_time : ℕ := bus_time - car_time_difference
  car_one_way_time * 2 = 70 := by sorry

end NUMINAMATH_CALUDE_amandas_car_round_trip_time_l201_20168


namespace NUMINAMATH_CALUDE_emilio_gifts_l201_20134

theorem emilio_gifts (total gifts_from_jorge gifts_from_pedro : ℕ) 
  (h1 : gifts_from_jorge = 6)
  (h2 : gifts_from_pedro = 4)
  (h3 : total = 21) :
  total - gifts_from_jorge - gifts_from_pedro = 11 := by
  sorry

end NUMINAMATH_CALUDE_emilio_gifts_l201_20134


namespace NUMINAMATH_CALUDE_tan_225_degrees_equals_one_l201_20133

theorem tan_225_degrees_equals_one : Real.tan (225 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_225_degrees_equals_one_l201_20133


namespace NUMINAMATH_CALUDE_number_problem_l201_20123

theorem number_problem (N : ℝ) : 
  (1/6 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/7 : ℝ) * N = 25 → 
  (60/100 : ℝ) * N = 252 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l201_20123


namespace NUMINAMATH_CALUDE_self_employed_tax_calculation_l201_20181

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- The problem statement --/
theorem self_employed_tax_calculation :
  let income : ℝ := 350000
  let tax_rate : ℝ := 0.06
  calculate_tax_amount income tax_rate = 21000 := by
  sorry

end NUMINAMATH_CALUDE_self_employed_tax_calculation_l201_20181


namespace NUMINAMATH_CALUDE_subtraction_addition_equality_l201_20161

theorem subtraction_addition_equality : -32 - (-14) + 4 = -14 := by sorry

end NUMINAMATH_CALUDE_subtraction_addition_equality_l201_20161


namespace NUMINAMATH_CALUDE_inequality_proof_l201_20138

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ ((a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l201_20138


namespace NUMINAMATH_CALUDE_sum_of_four_squares_of_five_l201_20153

theorem sum_of_four_squares_of_five : 5^2 + 5^2 + 5^2 + 5^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_of_five_l201_20153


namespace NUMINAMATH_CALUDE_slope_of_line_slope_to_angle_l201_20148

/-- The slope of the line x + √3 * y - 1 = 0 is -√3/3 -/
theorem slope_of_line (x y : ℝ) : 
  (x + Real.sqrt 3 * y - 1 = 0) → (y = -(1 / Real.sqrt 3) * x + 1 / Real.sqrt 3) :=
by sorry

/-- The slope -√3/3 corresponds to an angle of 150° -/
theorem slope_to_angle (θ : ℝ) :
  Real.tan θ = -(Real.sqrt 3 / 3) → θ = 150 * (Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_slope_to_angle_l201_20148


namespace NUMINAMATH_CALUDE_profit_percentage_problem_l201_20171

/-- Given that the cost price of 20 articles equals the selling price of x articles,
    and the profit percentage is 25%, prove that x equals 16. -/
theorem profit_percentage_problem (x : ℝ) 
  (h1 : 20 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_problem_l201_20171


namespace NUMINAMATH_CALUDE_negative_point_two_fifth_times_five_fifth_equals_negative_one_l201_20189

theorem negative_point_two_fifth_times_five_fifth_equals_negative_one :
  (-0.2)^5 * 5^5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_point_two_fifth_times_five_fifth_equals_negative_one_l201_20189


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l201_20127

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : building_height = 24)
  (h3 : building_shadow = 60)
  (h4 : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  flagpole_shadow = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_flagpole_shadow_length_l201_20127


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_AOB_l201_20125

/-- The minimum perimeter of triangle AOB given the conditions -/
theorem min_perimeter_triangle_AOB :
  let P : ℝ × ℝ := (4, 2)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (P ∈ l) ∧
    (A.1 > 0 ∧ A.2 = 0) ∧
    (B.1 = 0 ∧ B.2 > 0) ∧
    (A ∈ l) ∧ (B ∈ l) ∧
    (∀ (A' B' : ℝ × ℝ) (l' : Set (ℝ × ℝ)),
      (P ∈ l') ∧
      (A'.1 > 0 ∧ A'.2 = 0) ∧
      (B'.1 = 0 ∧ B'.2 > 0) ∧
      (A' ∈ l') ∧ (B' ∈ l') →
      dist O A + dist O B + dist A B ≤ dist O A' + dist O B' + dist A' B') ∧
    (dist O A + dist O B + dist A B = 20) :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_triangle_AOB_l201_20125


namespace NUMINAMATH_CALUDE_gravitational_force_calculation_l201_20142

/-- Gravitational force calculation -/
theorem gravitational_force_calculation
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ = 8000)
  (h₂ : d₂ = 320000)
  (h₃ : f₁ = 150)
  (h₄ : k = f₁ * d₁^2) -- Inverse square law
  : ∃ f₂ : ℝ, f₂ = k / d₂^2 ∧ f₂ = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_calculation_l201_20142


namespace NUMINAMATH_CALUDE_sally_dozens_of_eggs_l201_20158

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Sally bought -/
def total_eggs : ℕ := 48

/-- The number of dozens of eggs Sally bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem sally_dozens_of_eggs : dozens_bought = 4 := by
  sorry

end NUMINAMATH_CALUDE_sally_dozens_of_eggs_l201_20158


namespace NUMINAMATH_CALUDE_apple_distribution_result_l201_20166

/-- Represents the apple distribution problem --/
def apple_distribution (jim jane jerry jack jill jasmine jacob : ℕ) : ℚ :=
  let jack_to_jill := jack / 4
  let jasmine_jacob_shared := jasmine + jacob
  let jim_final := jim + (jasmine_jacob_shared / 10)
  let total_apples := jim_final + jane + jerry + (jack - jack_to_jill) + 
                      (jill + jack_to_jill) + (jasmine_jacob_shared / 2) + 
                      (jasmine_jacob_shared / 2)
  let average_apples := total_apples / 7
  average_apples / jim_final

/-- Theorem stating the result of the apple distribution problem --/
theorem apple_distribution_result : 
  ∃ ε > 0, |apple_distribution 20 60 40 80 50 30 90 - 1.705| < ε :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_result_l201_20166


namespace NUMINAMATH_CALUDE_power_product_simplification_l201_20132

theorem power_product_simplification (x y : ℝ) :
  (x^3 * y^2)^2 * (x / y^3) = x^7 * y := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l201_20132


namespace NUMINAMATH_CALUDE_circle_area_from_parallel_chords_l201_20104

-- Define the circle C
def C : Real → Real → Prop := sorry

-- Define the two lines
def line1 (x y : Real) : Prop := x - y - 1 = 0
def line2 (x y : Real) : Prop := x - y - 5 = 0

-- Define the chord length
def chord_length : Real := 10

-- Theorem statement
theorem circle_area_from_parallel_chords 
  (h1 : ∃ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2)
  (h2 : ∃ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4)
  (h3 : ∀ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2 → 
        Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length)
  (h4 : ∀ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4 → 
        Real.sqrt ((x3 - x4)^2 + (y3 - y4)^2) = chord_length) :
  (∃ (r : Real), ∀ (x y : Real), C x y ↔ (x - 0)^2 + (y - 0)^2 = r^2) ∧ 
  (∃ (area : Real), area = 27 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_from_parallel_chords_l201_20104


namespace NUMINAMATH_CALUDE_unique_solution_l201_20115

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y * z) = f x * f y * f z - 6 * x * y * z

/-- The main theorem stating that the only function satisfying the equation is f(x) = 2x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) : 
  ∀ x : ℝ, f x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l201_20115


namespace NUMINAMATH_CALUDE_limit_sum_geometric_sequence_l201_20167

def geometricSequence (n : ℕ) : ℚ := (1/2) * (1/2)^(n-1)

def sumGeometricSequence (n : ℕ) : ℚ := 
  (1/2) * (1 - (1/2)^n) / (1 - 1/2)

theorem limit_sum_geometric_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sumGeometricSequence n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sum_geometric_sequence_l201_20167


namespace NUMINAMATH_CALUDE_no_solution_to_double_inequality_l201_20118

theorem no_solution_to_double_inequality :
  ¬∃ (x : ℝ), (3 * x + 2 < (x + 2)^2) ∧ ((x + 2)^2 < 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_double_inequality_l201_20118


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l201_20149

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 2 * b + 3 = n^2) → b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l201_20149


namespace NUMINAMATH_CALUDE_max_value_inequality_l201_20130

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l201_20130


namespace NUMINAMATH_CALUDE_f_derivative_at_two_l201_20185

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : deriv (f a b) 1 = 0) :
  deriv (f a b) 2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_at_two_l201_20185


namespace NUMINAMATH_CALUDE_topsoil_cost_calculation_l201_20117

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yard_to_cubic_foot : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def topsoil_volume_cubic_yards : ℝ := 8

/-- Calculate the cost of topsoil given its volume in cubic yards -/
def topsoil_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yard_to_cubic_foot * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation :
  topsoil_cost topsoil_volume_cubic_yards = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_calculation_l201_20117


namespace NUMINAMATH_CALUDE_sets_equality_implies_sum_l201_20119

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {x, y/x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x+y, 0}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2014 + y^2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_sum_l201_20119


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l201_20178

/-- Given a triangle PQR where ∠P is thrice ∠R and ∠Q is equal to ∠R, 
    the measure of ∠Q is 36°. -/
theorem angle_measure_in_special_triangle (P Q R : ℝ) : 
  P + Q + R = 180 →  -- sum of angles in a triangle
  P = 3 * R →        -- ∠P is thrice ∠R
  Q = R →            -- ∠Q is equal to ∠R
  Q = 36 :=          -- measure of ∠Q is 36°
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l201_20178


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l201_20182

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (win_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * win_tails

theorem biased_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let win_tails : ℚ := -4
  coin_flip_expected_value p_heads p_tails win_heads win_tails = -2/5 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l201_20182


namespace NUMINAMATH_CALUDE_union_A_B_l201_20102

def A : Set ℤ := {0, 1}

def B : Set ℤ := {x | (x + 2) * (x - 1) < 0}

theorem union_A_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l201_20102


namespace NUMINAMATH_CALUDE_parallelogram_inscribed_in_circle_is_rectangle_l201_20175

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a parallelogram
def isParallelogram (q : Quadrilateral) : Prop := sorry

-- Define an inscribed quadrilateral
def isInscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def isRectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem parallelogram_inscribed_in_circle_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  isParallelogram q → isInscribed q c → isRectangle q := by sorry

end NUMINAMATH_CALUDE_parallelogram_inscribed_in_circle_is_rectangle_l201_20175


namespace NUMINAMATH_CALUDE_factorization_equality_l201_20154

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 6 * x = 3 * x * (x * y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l201_20154


namespace NUMINAMATH_CALUDE_park_creatures_l201_20187

theorem park_creatures (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 686) : ∃ (birds mammals imaginary : ℕ),
  birds + mammals + imaginary = total_heads ∧
  2 * birds + 4 * mammals + 3 * imaginary = total_legs ∧
  birds = 214 := by
  sorry

end NUMINAMATH_CALUDE_park_creatures_l201_20187


namespace NUMINAMATH_CALUDE_prob_at_least_three_heads_is_half_l201_20177

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least three heads when flipping five coins -/
def prob_at_least_three_heads : ℚ := 1/2

/-- Theorem stating that the probability of getting at least three heads 
    when flipping five coins simultaneously is 1/2 -/
theorem prob_at_least_three_heads_is_half : 
  prob_at_least_three_heads = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_three_heads_is_half_l201_20177


namespace NUMINAMATH_CALUDE_inequality_proof_l201_20174

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^4 ≥ x^2*y^2 + y^2*z^2 + z^2*x^2 ∧ 
  x^2*y^2 + y^2*z^2 + z^2*x^2 ≥ x*y*z*(x+y+z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l201_20174


namespace NUMINAMATH_CALUDE_quadratic_solution_unique_l201_20151

theorem quadratic_solution_unique (x : ℝ) :
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0 → x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_unique_l201_20151


namespace NUMINAMATH_CALUDE_even_function_properties_l201_20128

-- Define the function f
def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

-- State the theorem
theorem even_function_properties (m : ℝ) :
  (∀ x, f x m = f (-x) m) →
  (m = 1 ∧ (∀ x, f x m = 0 ↔ x = 1 ∨ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l201_20128


namespace NUMINAMATH_CALUDE_product_equals_four_l201_20163

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ (a b c : ℝ), a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_four_l201_20163


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l201_20129

theorem rectangle_side_ratio (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : (a / b) ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l201_20129


namespace NUMINAMATH_CALUDE_equation_solution_and_difference_l201_20193

theorem equation_solution_and_difference :
  (∃ x : ℚ, 11 * x + 4 = 7) ∧
  (let x : ℚ := 3 / 11; 11 * x + 4 = 7) ∧
  (12 / 11 - 3 / 11 = 9 / 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_difference_l201_20193


namespace NUMINAMATH_CALUDE_triangular_cross_section_solids_l201_20172

/-- Enumeration of geometric solids -/
inductive GeometricSolid
  | Cube
  | Cylinder
  | Cone
  | RegularTriangularPrism

/-- Predicate to determine if a geometric solid can have a triangular cross-section -/
def has_triangular_cross_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cube => True
  | GeometricSolid.Cylinder => False
  | GeometricSolid.Cone => True
  | GeometricSolid.RegularTriangularPrism => True

/-- Theorem stating which geometric solids can have a triangular cross-section -/
theorem triangular_cross_section_solids :
  ∀ (solid : GeometricSolid),
    has_triangular_cross_section solid ↔
      (solid = GeometricSolid.Cube ∨
       solid = GeometricSolid.Cone ∨
       solid = GeometricSolid.RegularTriangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_triangular_cross_section_solids_l201_20172


namespace NUMINAMATH_CALUDE_min_value_sqrt_x_squared_plus_two_l201_20124

theorem min_value_sqrt_x_squared_plus_two (x : ℝ) :
  Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_x_squared_plus_two_l201_20124


namespace NUMINAMATH_CALUDE_max_value_of_function_l201_20136

/-- The function f(x) = x^2(1-3x) has a maximum value of 1/12 in the interval (0, 1/3) -/
theorem max_value_of_function : 
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (1/3) ∧ 
  (∀ x, x ∈ Set.Ioo 0 (1/3) → x^2 * (1 - 3*x) ≤ c) ∧
  c = 1/12 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l201_20136


namespace NUMINAMATH_CALUDE_prob_end_after_two_draw_prob_exactly_two_white_l201_20159

/-- Represents the color of a ping-pong ball -/
inductive BallColor
  | Red
  | White
  | Blue

/-- Represents the box of ping-pong balls -/
structure Box where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The probability of drawing a specific color ball from the box -/
def drawProbability (box : Box) (color : BallColor) : Rat :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Blue => box.blue / box.total

/-- The box configuration as per the problem -/
def problemBox : Box := {
  total := 10
  red := 5
  white := 3
  blue := 2
}

/-- The probability of the process ending after two draws -/
def probEndAfterTwoDraw (box : Box) : Rat :=
  (1 - drawProbability box BallColor.Blue) * drawProbability box BallColor.Blue

/-- The probability of exactly drawing 2 white balls -/
def probExactlyTwoWhite (box : Box) : Rat :=
  3 * drawProbability box BallColor.Red * (drawProbability box BallColor.White)^2 +
  drawProbability box BallColor.White * drawProbability box BallColor.White * drawProbability box BallColor.Blue

theorem prob_end_after_two_draw :
  probEndAfterTwoDraw problemBox = 4 / 25 := by sorry

theorem prob_exactly_two_white :
  probExactlyTwoWhite problemBox = 153 / 1000 := by sorry

end NUMINAMATH_CALUDE_prob_end_after_two_draw_prob_exactly_two_white_l201_20159


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l201_20103

theorem soccer_ball_cost (football_cost soccer_cost : ℚ) : 
  (3 * football_cost + soccer_cost = 155) →
  (2 * football_cost + 3 * soccer_cost = 220) →
  soccer_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l201_20103


namespace NUMINAMATH_CALUDE_m_range_l201_20191

def y₁ (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 2)
def y₂ (x : ℝ) : ℝ := x - 1

theorem m_range :
  (∀ m : ℝ,
    (∀ x : ℝ, y₁ m x < 0 ∨ y₂ x < 0) ∧
    (∃ x : ℝ, x < -3 ∧ y₁ m x * y₂ x < 0)) ↔
  (∀ m : ℝ, -4 < m ∧ m < -3/2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l201_20191


namespace NUMINAMATH_CALUDE_repeating_not_necessarily_periodic_l201_20194

/-- Definition of the sequence property --/
def has_repeating_property (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, t > 0 ∧ ∀ n : ℕ, a k = a (k + n * t)

/-- Definition of periodicity --/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a k = a (k + T)

/-- Theorem stating that a sequence with the repeating property is not necessarily periodic --/
theorem repeating_not_necessarily_periodic :
  ∃ a : ℕ → ℕ, has_repeating_property a ∧ ¬ is_periodic a := by
  sorry

end NUMINAMATH_CALUDE_repeating_not_necessarily_periodic_l201_20194


namespace NUMINAMATH_CALUDE_circle_ratio_l201_20169

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (hab : a > b) (hb : b > 0) 
  (h : R^2 = (a/b) * (R^2 - r^2)) : 
  R/r = Real.sqrt (a/(a-b)) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l201_20169


namespace NUMINAMATH_CALUDE_congruence_solution_count_l201_20144

theorem congruence_solution_count :
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 45 = 70 % 45 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_count_l201_20144
