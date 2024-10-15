import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_cubes_l131_13117

theorem sum_of_cubes (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x + y + x^2*y + x*y^2 = 24) : 
  x^3 + y^3 = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l131_13117


namespace NUMINAMATH_CALUDE_inverse_g_84_l131_13147

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l131_13147


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l131_13155

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + x + b

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 ↔ quadratic_function a b x > 0) →
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l131_13155


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l131_13126

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b ∧ b > 0
  ecc : c / a = Real.sqrt 2 / 2
  perimeter : ℝ
  h_perimeter : perimeter = 4

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_m : m ≠ 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_line : ∀ x y, y = k * x + m
  h_intersect : A.1^2 / (E.b^2) + A.2^2 / (E.a^2) = 1 ∧
                B.1^2 / (E.b^2) + B.2^2 / (E.a^2) = 1
  h_relation : A.1 + 3 * B.1 = 4 * P.1 ∧ A.2 + 3 * B.2 = 4 * P.2

/-- The main theorem -/
theorem ellipse_and_line_properties (E : Ellipse) (L : IntersectingLine E) :
  (E.a = 1 ∧ E.b = Real.sqrt 2 / 2) ∧
  (L.m ∈ Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioo (1/2 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l131_13126


namespace NUMINAMATH_CALUDE_kaleb_summer_earnings_l131_13160

/-- Kaleb's lawn mowing business earnings --/
theorem kaleb_summer_earnings 
  (spring_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (total_amount : ℕ) 
  (h1 : spring_earnings = 4)
  (h2 : supplies_cost = 4)
  (h3 : total_amount = 50)
  : ℕ := by
  sorry

#check kaleb_summer_earnings

end NUMINAMATH_CALUDE_kaleb_summer_earnings_l131_13160


namespace NUMINAMATH_CALUDE_ursula_shopping_cost_l131_13122

/-- Represents the prices of items in Ursula's shopping trip -/
structure ShoppingPrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ
  eggs : ℝ
  honey : ℝ

/-- Calculates the total cost of all items -/
def totalCost (prices : ShoppingPrices) : ℝ :=
  prices.butter + prices.bread + prices.cheese + prices.tea + prices.eggs + prices.honey

/-- Theorem stating the conditions and the result of Ursula's shopping trip -/
theorem ursula_shopping_cost (prices : ShoppingPrices) : 
  prices.bread = prices.butter / 2 →
  prices.butter = 0.8 * prices.cheese →
  prices.tea = 1.5 * (prices.bread + prices.butter + prices.cheese) →
  prices.tea = 10 →
  prices.eggs = prices.bread / 2 →
  prices.honey = prices.eggs + 3 →
  abs (totalCost prices - 20.87) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ursula_shopping_cost_l131_13122


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l131_13134

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l131_13134


namespace NUMINAMATH_CALUDE_binary_1101100_equals_108_l131_13182

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101100_equals_108 :
  binary_to_decimal [false, false, true, true, false, true, true] = 108 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101100_equals_108_l131_13182


namespace NUMINAMATH_CALUDE_sugar_amount_in_recipe_l131_13192

/-- A recipe with specified amounts of ingredients -/
structure Recipe where
  flour : ℕ
  salt : ℕ
  sugar : ℕ

/-- The condition that sugar is one more cup than salt -/
def sugar_salt_relation (r : Recipe) : Prop :=
  r.sugar = r.salt + 1

theorem sugar_amount_in_recipe (r : Recipe) 
  (h1 : r.flour = 6)
  (h2 : r.salt = 7)
  (h3 : sugar_salt_relation r) :
  r.sugar = 8 := by
sorry

end NUMINAMATH_CALUDE_sugar_amount_in_recipe_l131_13192


namespace NUMINAMATH_CALUDE_min_coefficient_value_l131_13101

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ min_box : ℤ, (min_box = 34 ∧ box ≥ min_box) := by
  sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l131_13101


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l131_13121

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : α > 0) (h5 : β > 0) (h6 : γ > 0)
  (h7 : α = 2 * Real.sqrt (b * c))
  (h8 : β = 2 * Real.sqrt (c * a))
  (h9 : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l131_13121


namespace NUMINAMATH_CALUDE_square_area_error_l131_13171

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x * 1.12
  let actual_area := x^2
  let calculated_area := measured_side^2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 25.44 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l131_13171


namespace NUMINAMATH_CALUDE_infinite_omega_increasing_sequence_l131_13188

/-- The number of distinct prime divisors of a positive integer -/
def omega (n : ℕ) : ℕ := sorry

/-- The set of integers n > 1 satisfying ω(n) < ω(n+1) < ω(n+2) is infinite -/
theorem infinite_omega_increasing_sequence :
  Set.Infinite {n : ℕ | n > 1 ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2)} :=
sorry

end NUMINAMATH_CALUDE_infinite_omega_increasing_sequence_l131_13188


namespace NUMINAMATH_CALUDE_complement_of_equal_angles_is_proposition_l131_13132

-- Define what a proposition is in this context
def is_proposition (statement : String) : Prop :=
  -- A statement is a proposition if it can be true or false
  ∃ (truth_value : Bool), (truth_value = true ∨ truth_value = false)

-- The statement we want to prove is a proposition
def complement_of_equal_angles_statement : String :=
  "The complement of equal angles are equal"

-- Theorem stating that the given statement is a proposition
theorem complement_of_equal_angles_is_proposition :
  is_proposition complement_of_equal_angles_statement :=
sorry

end NUMINAMATH_CALUDE_complement_of_equal_angles_is_proposition_l131_13132


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l131_13185

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (size : PackSize) : ℕ :=
  match size with
  | .small => 8
  | .medium => 16
  | .large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (combo : PackCombination) : ℕ :=
  combo.small * cansInPack PackSize.small +
  combo.medium * cansInPack PackSize.medium +
  combo.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (combo : PackCombination) : ℕ :=
  combo.small + combo.medium + combo.large

/-- Checks if a pack combination is valid for the given total cans -/
def isValidCombination (combo : PackCombination) (totalCansNeeded : ℕ) : Prop :=
  totalCans combo = totalCansNeeded

/-- Theorem: The minimum number of packs needed to buy exactly 120 cans of soda is 5 -/
theorem min_packs_for_120_cans :
  ∃ (minCombo : PackCombination),
    isValidCombination minCombo 120 ∧
    totalPacks minCombo = 5 ∧
    ∀ (combo : PackCombination),
      isValidCombination combo 120 → totalPacks combo ≥ totalPacks minCombo := by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l131_13185


namespace NUMINAMATH_CALUDE_stone_heap_theorem_l131_13178

/-- 
Given k ≥ 3 heaps of stones with 1, 2, ..., k stones respectively,
after merging heaps, the final number of stones p is given by
p = (k + 1) * (3k - 1) / 8.
This function returns p given k.
-/
def final_stones (k : ℕ) : ℚ :=
  (k + 1) * (3 * k - 1) / 8

/-- 
This theorem states that for k ≥ 3, the final number of stones p
is a perfect square if and only if both 2k + 2 and 3k + 1 are perfect squares,
and that the least k satisfying this condition is 161.
-/
theorem stone_heap_theorem (k : ℕ) (h : k ≥ 3) :
  (∃ n : ℕ, final_stones k = n^2) ↔ 
  (∃ x y : ℕ, 2*k + 2 = x^2 ∧ 3*k + 1 = y^2) ∧
  (∀ m : ℕ, m < 161 → ¬(∃ x y : ℕ, 2*m + 2 = x^2 ∧ 3*m + 1 = y^2)) :=
sorry

end NUMINAMATH_CALUDE_stone_heap_theorem_l131_13178


namespace NUMINAMATH_CALUDE_largest_sum_simplification_l131_13120

theorem largest_sum_simplification :
  let sums := [1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/5, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/5) ∧ (1/3 + 1/5 = 8/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_simplification_l131_13120


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l131_13166

/-- Represents the revenue from ticket sales -/
def TicketRevenue (f h d : ℕ) (p : ℚ) : ℚ :=
  f * p + h * (p / 2) + d * (2 * p)

theorem charity_ticket_revenue :
  ∃ (f h d : ℕ) (p : ℚ),
    f + h + d = 200 ∧
    TicketRevenue f h d p = 5000 ∧
    f * p = 4500 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l131_13166


namespace NUMINAMATH_CALUDE_combine_expression_l131_13169

theorem combine_expression (a b : ℝ) : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end NUMINAMATH_CALUDE_combine_expression_l131_13169


namespace NUMINAMATH_CALUDE_sequence_sum_eq_square_l131_13119

def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (List.range n).sum

theorem sequence_sum_eq_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_eq_square_l131_13119


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l131_13165

theorem shoe_selection_theorem (n : ℕ) (m : ℕ) (h : n = 5 ∧ m = 4) :
  (Nat.choose n 1) * (Nat.choose (n - 1) (m - 2)) * (Nat.choose 2 1) * (Nat.choose 2 1) = 120 :=
sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l131_13165


namespace NUMINAMATH_CALUDE_vector_operation_result_l131_13150

/-- Prove that the vector operation (3, -6) - 5(1, -9) + (-1, 4) results in (-3, 43) -/
theorem vector_operation_result : 
  (⟨3, -6⟩ : ℝ × ℝ) - 5 • ⟨1, -9⟩ + ⟨-1, 4⟩ = ⟨-3, 43⟩ := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l131_13150


namespace NUMINAMATH_CALUDE_expansion_and_binomial_coeff_l131_13127

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the sum of binomial coefficients for (a + b)^n
def sumBinomialCoeff (n : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (a + b)^n
def thirdTermCoeff (n : ℕ) : ℕ := sorry

theorem expansion_and_binomial_coeff :
  -- Part I: The term containing 1/x^2 in (2x^2 + 1/x)^5
  (binomial 5 4) * 2 = 10 ∧
  -- Part II: If sum of binomial coefficients in (2x^2 + 1/x)^5 is 28 less than
  -- the coefficient of the third term in (√x + 2/x)^n, then n = 6
  ∃ n : ℕ, sumBinomialCoeff 5 = thirdTermCoeff n - 28 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_expansion_and_binomial_coeff_l131_13127


namespace NUMINAMATH_CALUDE_rocky_fights_l131_13154

/-- Represents the number of fights Rocky boxed in his career. -/
def total_fights : ℕ := sorry

/-- The fraction of fights that were knockouts. -/
def knockout_fraction : ℚ := 1/2

/-- The fraction of knockouts that were in the first round. -/
def first_round_knockout_fraction : ℚ := 1/5

/-- The number of knockouts in the first round. -/
def first_round_knockouts : ℕ := 19

theorem rocky_fights : 
  total_fights = 190 ∧ 
  (knockout_fraction * first_round_knockout_fraction * total_fights : ℚ) = first_round_knockouts := by
  sorry

end NUMINAMATH_CALUDE_rocky_fights_l131_13154


namespace NUMINAMATH_CALUDE_x_value_when_y_is_half_l131_13130

theorem x_value_when_y_is_half :
  ∀ x y : ℚ, y = 2 / (4 * x + 2) → y = 1 / 2 → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_half_l131_13130


namespace NUMINAMATH_CALUDE_card_sum_theorem_l131_13183

theorem card_sum_theorem (n : ℕ) (m : ℕ) (h1 : n ≥ 3) (h2 : m = n * (n - 1) / 2) (h3 : Odd m) :
  ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l131_13183


namespace NUMINAMATH_CALUDE_max_cuts_length_30x30_225pieces_l131_13180

/-- Represents a square board with cuts along grid lines -/
structure Board where
  size : ℕ
  pieces : ℕ
  cuts_length : ℕ

/-- The maximum possible total length of cuts for a given board configuration -/
def max_cuts_length (b : Board) : ℕ :=
  (b.pieces * 10 - 4 * b.size) / 2

/-- Theorem stating the maximum possible total length of cuts for the given board -/
theorem max_cuts_length_30x30_225pieces :
  ∃ (b : Board), b.size = 30 ∧ b.pieces = 225 ∧ max_cuts_length b = 1065 := by
  sorry

end NUMINAMATH_CALUDE_max_cuts_length_30x30_225pieces_l131_13180


namespace NUMINAMATH_CALUDE_orange_seller_gain_l131_13139

/-- The percentage gain a man wants to achieve when selling oranges -/
def desired_gain (initial_rate : ℚ) (loss_percent : ℚ) (new_rate : ℚ) : ℚ :=
  let cost_price := 1 / (initial_rate * (1 - loss_percent / 100))
  let new_price := 1 / new_rate
  (new_price / cost_price - 1) * 100

/-- Theorem stating the desired gain for specific selling rates and loss percentage -/
theorem orange_seller_gain :
  desired_gain 18 8 (11420689655172414 / 1000000000000000) = 45 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_gain_l131_13139


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l131_13125

noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

def tangent_line_at_2 (x y : ℝ) : Prop := y = x - 4

def tangent_lines_through_A (x y : ℝ) : Prop := y = x - 4 ∨ y = -2

theorem tangent_lines_theorem :
  (∀ x y : ℝ, y = f x → tangent_line_at_2 x y ↔ x = 2) ∧
  (∀ x y : ℝ, y = f x → tangent_lines_through_A x y ↔ (x = 2 ∧ y = -2) ∨ (x = 1 ∧ y = -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l131_13125


namespace NUMINAMATH_CALUDE_parallel_lines_plane_count_l131_13137

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the specifics of a line for this problem

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to count the number of planes determined by three lines -/
def count_planes (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem: The number of planes determined by three mutually parallel lines is either 1 or 3 -/
theorem parallel_lines_plane_count (l1 l2 l3 : Line3D) 
  (h1 : are_parallel l1 l2) 
  (h2 : are_parallel l2 l3) 
  (h3 : are_parallel l1 l3) : 
  count_planes l1 l2 l3 = 1 ∨ count_planes l1 l2 l3 = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_plane_count_l131_13137


namespace NUMINAMATH_CALUDE_power_evaluation_l131_13198

theorem power_evaluation : (2 ^ 2) ^ (2 ^ (2 + 1)) = 65536 := by sorry

end NUMINAMATH_CALUDE_power_evaluation_l131_13198


namespace NUMINAMATH_CALUDE_root_between_roots_l131_13123

theorem root_between_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + b = 0)
  (h2 : ∃ y : ℝ, y^2 - a*y + b = 0) :
  ∃ (x_1 y_1 z : ℝ), x_1^2 + a*x_1 + b = 0 ∧
                     y_1^2 - a*y_1 + b = 0 ∧
                     z^2 + 2*a*z + 2*b = 0 ∧
                     ((x_1 < z ∧ z < y_1) ∨ (y_1 < z ∧ z < x_1)) :=
by sorry

end NUMINAMATH_CALUDE_root_between_roots_l131_13123


namespace NUMINAMATH_CALUDE_odd_not_even_function_implication_l131_13175

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_not_even_function_implication (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∃ x, f a x ≠ f a (-x)) →
  (∃ x, f a x ≠ 0) →
  (a + 1)^2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_not_even_function_implication_l131_13175


namespace NUMINAMATH_CALUDE_sqrt_four_minus_one_l131_13100

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_one_l131_13100


namespace NUMINAMATH_CALUDE_cherries_cost_correct_l131_13114

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℚ := 985 / 100

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℚ := 1208 / 100

/-- The total amount Alyssa spent -/
def total_spent : ℚ := 2193 / 100

/-- Theorem stating that the amount Alyssa paid for cherries is correct -/
theorem cherries_cost_correct : cherries_cost = total_spent - grapes_cost := by
  sorry

end NUMINAMATH_CALUDE_cherries_cost_correct_l131_13114


namespace NUMINAMATH_CALUDE_basketball_scores_l131_13194

theorem basketball_scores (total_players : ℕ) (less_than_yoongi : ℕ) (h1 : total_players = 21) (h2 : less_than_yoongi = 11) :
  total_players - less_than_yoongi - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_scores_l131_13194


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_l131_13164

theorem modulus_of_specific_complex : let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ‖z‖ = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_l131_13164


namespace NUMINAMATH_CALUDE_lowest_price_correct_l131_13141

/-- Calculates the lowest price per unit to sell electronic components without making a loss. -/
def lowest_price_per_unit (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_units)

theorem lowest_price_correct (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) :
  lowest_price_per_unit production_cost shipping_cost fixed_costs num_units =
  (production_cost * num_units + shipping_cost * num_units + fixed_costs) / num_units :=
by sorry

#eval lowest_price_per_unit 120 10 25000 100

end NUMINAMATH_CALUDE_lowest_price_correct_l131_13141


namespace NUMINAMATH_CALUDE_distance_to_line_l131_13163

/-- The distance from a point on the line y = ax - 2a + 5 to the line x - 2y + 3 = 0 is √5 -/
theorem distance_to_line : ∀ (a : ℝ), ∃ (A : ℝ × ℝ),
  (A.2 = a * A.1 - 2 * a + 5) ∧ 
  (|A.1 - 2 * A.2 + 3| / Real.sqrt (1 + 4) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_l131_13163


namespace NUMINAMATH_CALUDE_number_problem_l131_13170

theorem number_problem (x : ℝ) : (x / 4 + 15 = 27) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l131_13170


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l131_13196

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_x_plus_y_less_than_4 :
  probability givenSquare condition = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l131_13196


namespace NUMINAMATH_CALUDE_theater_attendance_l131_13102

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) : 
  ∃ (num_children : ℕ), 
    num_children ≤ total_people ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue ∧
    num_children = 18 := by
  sorry

#check theater_attendance

end NUMINAMATH_CALUDE_theater_attendance_l131_13102


namespace NUMINAMATH_CALUDE_jasons_textbooks_l131_13176

/-- Represents the problem of determining the number of textbooks Jason has. -/
theorem jasons_textbooks :
  let bookcase_limit : ℕ := 80  -- Maximum weight the bookcase can hold in pounds
  let hardcover_count : ℕ := 70  -- Number of hardcover books
  let hardcover_weight : ℚ := 1/2  -- Weight of each hardcover book in pounds
  let textbook_weight : ℕ := 2  -- Weight of each textbook in pounds
  let knickknack_count : ℕ := 3  -- Number of knick-knacks
  let knickknack_weight : ℕ := 6  -- Weight of each knick-knack in pounds
  let over_limit : ℕ := 33  -- Amount the total collection is over the weight limit in pounds

  let total_weight := bookcase_limit + over_limit
  let hardcover_total_weight := hardcover_count * hardcover_weight
  let knickknack_total_weight := knickknack_count * knickknack_weight
  let textbook_total_weight := total_weight - (hardcover_total_weight + knickknack_total_weight)

  textbook_total_weight / textbook_weight = 30 := by
  sorry

end NUMINAMATH_CALUDE_jasons_textbooks_l131_13176


namespace NUMINAMATH_CALUDE_wheel_speed_problem_l131_13173

theorem wheel_speed_problem (circumference : ℝ) (time_decrease : ℝ) (speed_increase : ℝ) :
  circumference = 15 →
  time_decrease = 1 / 3 →
  speed_increase = 10 →
  ∃ (original_speed : ℝ),
    original_speed * (circumference / 5280) = circumference / 5280 ∧
    (original_speed + speed_increase) * ((circumference / (5280 * original_speed)) - time_decrease / 3600) = circumference / 5280 ∧
    original_speed = 15 :=
by sorry

end NUMINAMATH_CALUDE_wheel_speed_problem_l131_13173


namespace NUMINAMATH_CALUDE_g_neg_one_eq_neg_one_l131_13112

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_eq_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
    sorry

end NUMINAMATH_CALUDE_g_neg_one_eq_neg_one_l131_13112


namespace NUMINAMATH_CALUDE_fourth_root_of_x_sqrt_x_squared_l131_13189

theorem fourth_root_of_x_sqrt_x_squared (x : ℝ) (hx : x > 0) : 
  (((x * Real.sqrt x) ^ 2) ^ (1/4 : ℝ)) = x ^ (3/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_of_x_sqrt_x_squared_l131_13189


namespace NUMINAMATH_CALUDE_escalator_length_is_126_l131_13193

/-- Calculates the length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Proves that the length of the escalator is 126 feet under the given conditions. -/
theorem escalator_length_is_126 :
  escalator_length 11 3 9 = 126 := by
  sorry

#eval escalator_length 11 3 9

end NUMINAMATH_CALUDE_escalator_length_is_126_l131_13193


namespace NUMINAMATH_CALUDE_quadratic_coefficients_unique_l131_13156

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficients_unique :
  ∀ a b c : ℝ,
    (∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c (-0.75)) ∧
    QuadraticFunction a b c (-0.75) = 3.25 ∧
    QuadraticFunction a b c 0 = 1 →
    a = -4 ∧ b = -6 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_unique_l131_13156


namespace NUMINAMATH_CALUDE_quadratic_minimum_l131_13172

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l131_13172


namespace NUMINAMATH_CALUDE_quadratic_equiv_abs_value_l131_13152

theorem quadratic_equiv_abs_value : ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ |x - 8| = 3) ↔ (b = -16 ∧ c = 55) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equiv_abs_value_l131_13152


namespace NUMINAMATH_CALUDE_set_union_problem_l131_13106

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {3^a, b}
  A ∪ B = {-1, 0, 1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l131_13106


namespace NUMINAMATH_CALUDE_find_number_l131_13118

theorem find_number : ∃ x : ℝ, x - 2.95 - 2.95 = 9.28 ∧ x = 15.18 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l131_13118


namespace NUMINAMATH_CALUDE_message_encoding_l131_13140

-- Define the encoding functions
def oldEncode (s : String) : String := sorry

def newEncode (s : String) : String := sorry

-- Define the decoding function
def decode (s : String) : String := sorry

-- Theorem statement
theorem message_encoding :
  let originalMessage := "011011010011"
  let decodedMessage := decode originalMessage
  newEncode decodedMessage = "211221121" := by sorry

end NUMINAMATH_CALUDE_message_encoding_l131_13140


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_3_l131_13131

theorem binomial_coefficient_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_3_l131_13131


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l131_13161

theorem fraction_ratio_equality (x : ℚ) : (2 / 3 : ℚ) / x = (3 / 5 : ℚ) / (7 / 15 : ℚ) → x = 14 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l131_13161


namespace NUMINAMATH_CALUDE_mode_of_data_set_l131_13124

def data_set : List ℕ := [5, 4, 4, 3, 6, 2]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 4 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l131_13124


namespace NUMINAMATH_CALUDE_fraction_power_four_l131_13199

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_four_l131_13199


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l131_13107

/-- Given a right triangle DEF with hypotenuse DE = 15, DF = 9, and EF = 12,
    the distance from F to the midpoint of DE is 7.5 -/
theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l131_13107


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l131_13191

/-- The ratio of the area of a square with side length 30 cm to the area of a rectangle with dimensions 28 cm by 45 cm is 5/7. -/
theorem square_to_rectangle_area_ratio : 
  let square_side : ℝ := 30
  let rect_length : ℝ := 28
  let rect_width : ℝ := 45
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area / rect_area = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_ratio_l131_13191


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l131_13115

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 450 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 450 → m ≥ n :=
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l131_13115


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l131_13103

/-- Represents a wheel with spokes -/
structure Wheel :=
  (spokes : ℕ)
  (spokes_le_three : spokes ≤ 3)

/-- Represents a configuration of wheels -/
def WheelConfiguration := List Wheel

/-- The total number of spokes in a configuration -/
def total_spokes (config : WheelConfiguration) : ℕ :=
  config.map Wheel.spokes |>.sum

/-- Theorem stating that 3 wheels are possible and 2 wheels are not possible -/
theorem wheel_configuration_theorem 
  (config : WheelConfiguration) 
  (total_spokes_ge_seven : total_spokes config ≥ 7) : 
  (∃ (three_wheel_config : WheelConfiguration), three_wheel_config.length = 3 ∧ total_spokes three_wheel_config ≥ 7) ∧
  (¬ ∃ (two_wheel_config : WheelConfiguration), two_wheel_config.length = 2 ∧ total_spokes two_wheel_config ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l131_13103


namespace NUMINAMATH_CALUDE_unique_root_condition_l131_13144

theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  a = (Real.log 6 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l131_13144


namespace NUMINAMATH_CALUDE_seventeenth_group_number_l131_13111

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroup : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  firstGroup + (groupNumber - 1) * interval

/-- Theorem: The 17th group number in the given systematic sampling is 264 -/
theorem seventeenth_group_number :
  systematicSample 800 50 8 17 = 264 := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_group_number_l131_13111


namespace NUMINAMATH_CALUDE_equal_distribution_of_cards_l131_13149

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_cards_l131_13149


namespace NUMINAMATH_CALUDE_cherry_pitting_time_l131_13186

/-- Proves that it takes 2 hours to pit cherries for a pie given the specified conditions -/
theorem cherry_pitting_time :
  ∀ (pounds_needed : ℕ) 
    (cherries_per_pound : ℕ) 
    (cherries_per_set : ℕ) 
    (minutes_per_set : ℕ),
  pounds_needed = 3 →
  cherries_per_pound = 80 →
  cherries_per_set = 20 →
  minutes_per_set = 10 →
  (pounds_needed * cherries_per_pound * minutes_per_set) / 
  (cherries_per_set * 60) = 2 := by
sorry

end NUMINAMATH_CALUDE_cherry_pitting_time_l131_13186


namespace NUMINAMATH_CALUDE_count_nines_in_range_l131_13104

/-- The number of occurrences of the digit 9 in all integers from 1 to 1000 (inclusive) -/
def count_nines : ℕ := sorry

/-- The range of integers we're considering -/
def range_start : ℕ := 1
def range_end : ℕ := 1000

theorem count_nines_in_range : count_nines = 300 := by sorry

end NUMINAMATH_CALUDE_count_nines_in_range_l131_13104


namespace NUMINAMATH_CALUDE_unique_triple_solution_l131_13190

theorem unique_triple_solution (a b p : ℕ) (h_prime : Nat.Prime p) :
  (a + b)^p = p^a + p^b ↔ a = 1 ∧ b = 1 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l131_13190


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l131_13177

/-- Represents a hyperbola with parameter m -/
structure Hyperbola (m : ℝ) where
  eq : ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3

/-- The distance from the center to a focus of the hyperbola -/
def focal_distance (h : Hyperbola m) : ℝ := 2

theorem hyperbola_m_value (h : Hyperbola m) 
  (focus : focal_distance h = 2) : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l131_13177


namespace NUMINAMATH_CALUDE_expression_evaluation_l131_13162

theorem expression_evaluation (x c : ℝ) (hx : x = 3) (hc : c = 2) :
  (x^2 + c)^2 - (x^2 - c)^2 = 72 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l131_13162


namespace NUMINAMATH_CALUDE_triangular_array_theorem_l131_13145

/-- Represents the elements of a triangular array -/
def a (i j : ℕ) : ℚ :=
  sorry

/-- The common ratio for geometric sequences in rows -/
def common_ratio : ℚ := 1 / 2

/-- The common difference for the arithmetic sequence in the first column -/
def common_diff : ℚ := 1 / 4

theorem triangular_array_theorem (n : ℕ) (h : n > 0) :
  ∀ (i j : ℕ), i ≥ j → i > 0 → j > 0 →
  (∀ k, k > 0 → a k 1 - a (k-1) 1 = common_diff) →
  (∀ k l, k > 2 → l > 0 → a k (l+1) / a k l = common_ratio) →
  a n 3 = n / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_theorem_l131_13145


namespace NUMINAMATH_CALUDE_expression_simplification_l131_13142

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (1 / (x - 1) + 1 / (x + 1)) / (x^2 / (3 * x^2 - 3)) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l131_13142


namespace NUMINAMATH_CALUDE_abc_system_property_l131_13116

theorem abc_system_property (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + a = b^2)
  (eq2 : b^2 + b = c^2)
  (eq3 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_system_property_l131_13116


namespace NUMINAMATH_CALUDE_correct_factorization_l131_13151

theorem correct_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

#check correct_factorization

end NUMINAMATH_CALUDE_correct_factorization_l131_13151


namespace NUMINAMATH_CALUDE_smallest_w_l131_13129

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^7) (936 * w))
  (h2 : is_factor (3^4) (936 * w))
  (h3 : is_factor (5^3) (936 * w))
  (h4 : is_factor (7^2) (936 * w))
  (h5 : is_factor (11^2) (936 * w)) :
  w ≥ 320166000 ∧ 
  (∀ v : ℕ, v > 0 → 
    is_factor (2^7) (936 * v) → 
    is_factor (3^4) (936 * v) → 
    is_factor (5^3) (936 * v) → 
    is_factor (7^2) (936 * v) → 
    is_factor (11^2) (936 * v) → 
    v ≥ w) :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l131_13129


namespace NUMINAMATH_CALUDE_election_total_votes_l131_13105

/-- Represents an election with two candidates -/
structure Election where
  totalValidVotes : ℕ
  invalidVotes : ℕ
  losingCandidatePercentage : ℚ
  voteDifference : ℕ

/-- The total number of polled votes in the election -/
def totalPolledVotes (e : Election) : ℕ :=
  e.totalValidVotes + e.invalidVotes

/-- Theorem stating the total polled votes for the given election scenario -/
theorem election_total_votes (e : Election) 
  (h1 : e.losingCandidatePercentage = 1/5) 
  (h2 : e.voteDifference = 500) 
  (h3 : e.invalidVotes = 10) :
  totalPolledVotes e = 843 := by
  sorry

end NUMINAMATH_CALUDE_election_total_votes_l131_13105


namespace NUMINAMATH_CALUDE_book_sale_profit_percentage_l131_13159

/-- Calculates the profit percentage for a book sale with given parameters. -/
theorem book_sale_profit_percentage
  (purchase_price : ℝ)
  (purchase_tax_rate : ℝ)
  (shipping_fee : ℝ)
  (selling_price : ℝ)
  (trading_tax_rate : ℝ)
  (h1 : purchase_price = 32)
  (h2 : purchase_tax_rate = 0.05)
  (h3 : shipping_fee = 2.5)
  (h4 : selling_price = 56)
  (h5 : trading_tax_rate = 0.07)
  : ∃ (profit_percentage : ℝ), abs (profit_percentage - 44.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_profit_percentage_l131_13159


namespace NUMINAMATH_CALUDE_A_equals_B_l131_13108

-- Define set A
def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}

-- Define set B
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l131_13108


namespace NUMINAMATH_CALUDE_geometry_theorem_l131_13179

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the perpendicular relation between a line and a plane
def perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relation between lines
def parallelLines (l1 l2 : Line) : Prop := sorry

-- Define skew relation between lines
def skewLines (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between planes
def parallelPlanes (p1 p2 : Plane) : Prop := sorry

-- Define intersection relation between planes
def planesIntersect (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between lines
def perpendicularLines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (h1 : perpendicularToPlane m α) 
  (h2 : perpendicularToPlane n β) :
  (parallelLines m n → parallelPlanes α β) ∧ 
  (skewLines m n → planesIntersect α β) ∧
  (perpendicularPlanes α β → perpendicularLines m n) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l131_13179


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l131_13109

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l131_13109


namespace NUMINAMATH_CALUDE_congruent_triangles_exist_l131_13184

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_4 : n ≥ 4)

/-- A subset of vertices of a regular polygon -/
structure VertexSubset (n : ℕ) where
  (polygon : RegularPolygon n)
  (r : ℕ)
  (vertices : Finset (Fin n))
  (subset_size : vertices.card = r)

/-- Two triangles in a regular polygon -/
structure PolygonTrianglePair (n : ℕ) where
  (polygon : RegularPolygon n)
  (t1 t2 : Fin n → Fin n → Fin n → Prop)

/-- Congruence of two triangles in a regular polygon -/
def CongruentTriangles (n : ℕ) (pair : PolygonTrianglePair n) : Prop :=
  sorry

/-- The main theorem -/
theorem congruent_triangles_exist (n : ℕ) (V : VertexSubset n) 
  (h : V.r * (V.r - 3) ≥ n) : 
  ∃ (pair : PolygonTrianglePair n), 
    (∀ i j k, pair.t1 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    (∀ i j k, pair.t2 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    CongruentTriangles n pair :=
sorry

end NUMINAMATH_CALUDE_congruent_triangles_exist_l131_13184


namespace NUMINAMATH_CALUDE_f_sum_property_l131_13174

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_property_l131_13174


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_one_squared_leq_one_l131_13113

theorem negation_of_exists_lt_one_squared_leq_one :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_one_squared_leq_one_l131_13113


namespace NUMINAMATH_CALUDE_smallest_x_value_l131_13195

theorem smallest_x_value (x y z : ℝ) 
  (sum_condition : x + y + z = 6)
  (product_condition : x * y + x * z + y * z = 10) :
  ∀ x' : ℝ, (∃ y' z' : ℝ, x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10) → x' ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l131_13195


namespace NUMINAMATH_CALUDE_log_ratio_identity_l131_13168

theorem log_ratio_identity (a b x : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hb : b > 0) (hx : x > 0) :
  (Real.log x / Real.log a) / (Real.log x / Real.log (a * b)) = 1 + Real.log b / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_identity_l131_13168


namespace NUMINAMATH_CALUDE_fraction_equality_l131_13146

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (1 / a + 1 / b = 1 / 2) → (a * b / (a + b) = 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l131_13146


namespace NUMINAMATH_CALUDE_rattlesnake_count_l131_13133

theorem rattlesnake_count (total : ℕ) (pythons boa_constrictors rattlesnakes vipers : ℕ) :
  total = 350 ∧
  total = pythons + boa_constrictors + rattlesnakes + vipers ∧
  pythons = 2 * boa_constrictors ∧
  vipers = rattlesnakes / 2 ∧
  boa_constrictors = 60 ∧
  pythons + vipers = (40 * total) / 100 →
  rattlesnakes = 40 := by
sorry

end NUMINAMATH_CALUDE_rattlesnake_count_l131_13133


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l131_13136

/-- A regular polygon with exterior angles of 45 degrees has interior angle sum of 1080 degrees -/
theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ), 
  n > 2 →
  (360 : ℝ) / n = 45 →
  (n - 2) * 180 = 1080 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l131_13136


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l131_13157

/-- Geometric sequence with sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ S n = a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  S 4 = 1 →
  S 8 = 3 →
  a 17 + a 18 + a 19 + a 20 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l131_13157


namespace NUMINAMATH_CALUDE_sleep_ratio_theorem_l131_13158

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Theorem stating the ratio of the fourth night's sleep to the third night's sleep -/
theorem sleep_ratio_theorem (s : SleepPattern) 
  (h1 : s.first_night = 6)
  (h2 : s.second_night = s.first_night + 2)
  (h3 : s.third_night = s.second_night / 2)
  (h4 : s.fourth_night = s.third_night * (s.fourth_night / s.third_night))
  (h5 : s.first_night + s.second_night + s.third_night + s.fourth_night = 30) :
  s.fourth_night / s.third_night = 3 := by
  sorry

end NUMINAMATH_CALUDE_sleep_ratio_theorem_l131_13158


namespace NUMINAMATH_CALUDE_sampling_is_systematic_l131_13153

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with rows and seats --/
structure Auditorium where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  auditorium : Auditorium
  seatNumberSelected : Nat

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.seatNumberSelected > 0 ∧ 
  strategy.seatNumberSelected ≤ strategy.auditorium.seatsPerRow ∧
  strategy.seatNumberSelected = strategy.seatNumberSelected

/-- Theorem stating that the given sampling strategy is systematic --/
theorem sampling_is_systematic (a : Auditorium) (s : SamplingStrategy) :
  a.rows = 25 → 
  a.seatsPerRow = 20 → 
  s.auditorium = a → 
  s.seatNumberSelected = 15 → 
  isSystematicSampling s := by
  sorry

#check sampling_is_systematic

end NUMINAMATH_CALUDE_sampling_is_systematic_l131_13153


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l131_13197

theorem right_triangle_consecutive_legs (a : ℕ) :
  let b := a + 1
  let c := Real.sqrt (a^2 + b^2)
  c^2 = 2*a^2 + 2*a + 1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l131_13197


namespace NUMINAMATH_CALUDE_angle_bisector_coefficient_sum_l131_13110

/-- Given a triangle ABC with vertices A = (-3, 2), B = (4, -1), and C = (-1, -5),
    the equation of the angle bisector of ∠A in the form dx + 2y + e = 0
    has coefficients d and e such that d + e equals a specific value. -/
theorem angle_bisector_coefficient_sum (d e : ℝ) : 
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ × ℝ := (-1, -5)
  ∃ (k : ℝ), d * A.1 + 2 * A.2 + e = k ∧
             d * B.1 + 2 * B.2 + e = 0 ∧
             d * C.1 + 2 * C.2 + e = 0 →
  d + e = sorry -- The exact value would be calculated here
:= by sorry


end NUMINAMATH_CALUDE_angle_bisector_coefficient_sum_l131_13110


namespace NUMINAMATH_CALUDE_concert_attendance_l131_13148

/-- Proves that given the initial ratio of women to men is 1:2, and after 12 women and 29 men left
    the ratio became 1:3, the original number of people at the concert was 21. -/
theorem concert_attendance (w m : ℕ) : 
  w / m = 1 / 2 →  -- Initial ratio of women to men
  (w - 12) / (m - 29) = 1 / 3 →  -- Ratio after some people left
  w + m = 21  -- Total number of people initially
  := by sorry

end NUMINAMATH_CALUDE_concert_attendance_l131_13148


namespace NUMINAMATH_CALUDE_eighth_term_value_l131_13128

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The 8th term of the geometric sequence is either 128 or -128 -/
theorem eighth_term_value (seq : GeometricSequence) :
  seq.a 8 = 128 ∨ seq.a 8 = -128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l131_13128


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l131_13138

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l131_13138


namespace NUMINAMATH_CALUDE_solve_system_l131_13181

theorem solve_system (a b : ℝ) 
  (eq1 : 2020*a + 2030*b = 2050)
  (eq2 : 2030*a + 2040*b = 2060) : 
  a - b = -5 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l131_13181


namespace NUMINAMATH_CALUDE_total_candy_count_l131_13187

theorem total_candy_count (chocolate_boxes : ℕ) (caramel_boxes : ℕ) (mint_boxes : ℕ) (berry_boxes : ℕ)
  (chocolate_caramel_pieces_per_box : ℕ) (mint_pieces_per_box : ℕ) (berry_pieces_per_box : ℕ)
  (h1 : chocolate_boxes = 7)
  (h2 : caramel_boxes = 3)
  (h3 : mint_boxes = 5)
  (h4 : berry_boxes = 4)
  (h5 : chocolate_caramel_pieces_per_box = 8)
  (h6 : mint_pieces_per_box = 10)
  (h7 : berry_pieces_per_box = 12) :
  chocolate_boxes * chocolate_caramel_pieces_per_box +
  caramel_boxes * chocolate_caramel_pieces_per_box +
  mint_boxes * mint_pieces_per_box +
  berry_boxes * berry_pieces_per_box = 178 := by
sorry

end NUMINAMATH_CALUDE_total_candy_count_l131_13187


namespace NUMINAMATH_CALUDE_line_properties_l131_13135

/-- Given a line passing through two points and a direction vector format, prove the value of 'a' and the x-intercept. -/
theorem line_properties (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 7) →
  p2 = (2, -2) →
  (∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -1)) →
  a = 5/9 ∧ 
  (∃ x : ℝ, x = 4 ∧ 0 = -x + 4) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l131_13135


namespace NUMINAMATH_CALUDE_election_percentages_correct_l131_13167

def votes : List Nat := [1136, 7636, 10628, 8562, 6490]

def total_votes : Nat := votes.sum

def percentage (votes : Nat) (total : Nat) : Float :=
  (votes.toFloat / total.toFloat) * 100

def percentages : List Float :=
  votes.map (λ v => percentage v total_votes)

theorem election_percentages_correct :
  percentages ≈ [3.20, 21.54, 29.98, 24.15, 18.30] := by
  sorry

end NUMINAMATH_CALUDE_election_percentages_correct_l131_13167


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l131_13143

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = 3 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l131_13143
