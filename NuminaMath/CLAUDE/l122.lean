import Mathlib

namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l122_12214

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)}

theorem integer_fraction_pairs :
  {p : ℕ+ × ℕ+ | is_integer_fraction p.1 p.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l122_12214


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l122_12276

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  12 = Nat.gcd n 72 ∧ ∀ m : ℕ, m ∣ n → m ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l122_12276


namespace NUMINAMATH_CALUDE_new_mixture_ratio_l122_12225

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℚ
  water : ℚ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratioAlcoholToWater (m : Mixture) : ℚ := m.alcohol / m.water

/-- First jar with 3:1 ratio and 4 liters total -/
def jar1 : Mixture := { alcohol := 3, water := 1 }

/-- Second jar with 2:1 ratio and 6 liters total -/
def jar2 : Mixture := { alcohol := 4, water := 2 }

/-- Amount taken from first jar -/
def amount1 : ℚ := 1

/-- Amount taken from second jar -/
def amount2 : ℚ := 2

/-- New mixture created from combining portions of jar1 and jar2 -/
def newMixture : Mixture := {
  alcohol := amount1 * (jar1.alcohol / (jar1.alcohol + jar1.water)) + 
             amount2 * (jar2.alcohol / (jar2.alcohol + jar2.water)),
  water := amount1 * (jar1.water / (jar1.alcohol + jar1.water)) + 
           amount2 * (jar2.water / (jar2.alcohol + jar2.water))
}

theorem new_mixture_ratio : 
  ratioAlcoholToWater newMixture = 41 / 19 := by sorry

end NUMINAMATH_CALUDE_new_mixture_ratio_l122_12225


namespace NUMINAMATH_CALUDE_complex_equation_solution_l122_12299

theorem complex_equation_solution :
  ∃ x : ℂ, (5 : ℂ) - 3 * Complex.I * x = (7 : ℂ) - Complex.I * x ∧ x = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l122_12299


namespace NUMINAMATH_CALUDE_parabola_vertex_l122_12287

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 3)

/-- Theorem: The vertex of the parabola y = -(x-1)^2 + 3 is (1, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≤ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l122_12287


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l122_12282

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  lastInningsScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns) / (stats.innings)

/-- Theorem stating the batsman's average after 12 innings -/
theorem batsman_average_after_12_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 12)
  (h2 : stats.lastInningsScore = 48)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateAverage stats = calculateAverage { stats with 
    innings := stats.innings - 1, 
    totalRuns := stats.totalRuns - stats.lastInningsScore 
  } + stats.averageIncrease) :
  calculateAverage stats = 26 := by
  sorry

#check batsman_average_after_12_innings

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l122_12282


namespace NUMINAMATH_CALUDE_park_outer_diameter_l122_12226

/-- Represents the dimensions of a circular park -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer boundary diameter is 60 feet -/
theorem park_outer_diameter (park : CircularPark) 
  (h1 : park.pond_diameter = 16)
  (h2 : park.garden_width = 12)
  (h3 : park.path_width = 10) : 
  outer_boundary_diameter park = 60 := by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l122_12226


namespace NUMINAMATH_CALUDE_curve_translation_l122_12205

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.sin x - 2 * y + 3 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (1 + y) * Real.cos x - 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ (x y : ℝ),
  original_curve (x + π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l122_12205


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l122_12255

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (x^2 - 5*x + 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) ∧
    A = -6 ∧ B = 7 ∧ C = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l122_12255


namespace NUMINAMATH_CALUDE_vector_equality_l122_12293

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

theorem vector_equality : c = a - b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l122_12293


namespace NUMINAMATH_CALUDE_distribute_6_3_l122_12201

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_6_3 : distribute 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_3_l122_12201


namespace NUMINAMATH_CALUDE_kyler_won_one_game_l122_12200

/-- Represents a chess tournament between Peter, Emma, and Kyler -/
structure ChessTournament where
  total_games : ℕ
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

/-- Calculates Kyler's wins in the chess tournament -/
def kyler_wins (t : ChessTournament) : ℕ :=
  t.total_games - (t.peter_wins + t.peter_losses + t.emma_wins + t.emma_losses + t.kyler_losses)

/-- Theorem stating that Kyler won 1 game in the given tournament conditions -/
theorem kyler_won_one_game (t : ChessTournament) 
  (h1 : t.total_games = 15)
  (h2 : t.peter_wins = 5)
  (h3 : t.peter_losses = 3)
  (h4 : t.emma_wins = 2)
  (h5 : t.emma_losses = 4)
  (h6 : t.kyler_losses = 4) :
  kyler_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_kyler_won_one_game_l122_12200


namespace NUMINAMATH_CALUDE_total_gross_profit_after_discounts_l122_12258

/-- Calculate the total gross profit for three items after discounts --/
theorem total_gross_profit_after_discounts
  (price_A price_B price_C : ℝ)
  (gross_profit_percentage : ℝ)
  (discount_A discount_B discount_C : ℝ)
  (h1 : price_A = 91)
  (h2 : price_B = 110)
  (h3 : price_C = 240)
  (h4 : gross_profit_percentage = 1.60)
  (h5 : discount_A = 0.10)
  (h6 : discount_B = 0.05)
  (h7 : discount_C = 0.12) :
  let cost_A := price_A / (1 + gross_profit_percentage)
  let cost_B := price_B / (1 + gross_profit_percentage)
  let cost_C := price_C / (1 + gross_profit_percentage)
  let discounted_price_A := price_A * (1 - discount_A)
  let discounted_price_B := price_B * (1 - discount_B)
  let discounted_price_C := price_C * (1 - discount_C)
  let gross_profit_A := discounted_price_A - cost_A
  let gross_profit_B := discounted_price_B - cost_B
  let gross_profit_C := discounted_price_C - cost_C
  let total_gross_profit := gross_profit_A + gross_profit_B + gross_profit_C
  ∃ ε > 0, |total_gross_profit - 227.98| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_gross_profit_after_discounts_l122_12258


namespace NUMINAMATH_CALUDE_trapezoid_AB_length_l122_12213

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  AB : ℝ
  -- Length of side CD
  CD : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Condition: The ratio of areas is 5:2
  area_ratio_condition : area_ratio = 5 / 2
  -- Condition: The sum of AB and CD is 280
  sum_sides : AB + CD = 280

/-- Theorem stating that under given conditions, AB = 200 -/
theorem trapezoid_AB_length (t : Trapezoid) : t.AB = 200 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_AB_length_l122_12213


namespace NUMINAMATH_CALUDE_bakery_pie_division_l122_12270

theorem bakery_pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → total_pie / num_people = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l122_12270


namespace NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l122_12256

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ∧
  Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ≤ 4 :=
by sorry

theorem bounds_are_tight : 
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 2 * Real.sqrt 2 ∧
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l122_12256


namespace NUMINAMATH_CALUDE_larger_cube_volume_l122_12216

theorem larger_cube_volume (original_volume : ℝ) (scale_factor : ℝ) :
  original_volume = 216 →
  scale_factor = 2.5 →
  (scale_factor * (original_volume ^ (1/3 : ℝ)))^3 = 3375 := by
sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l122_12216


namespace NUMINAMATH_CALUDE_min_value_fraction_l122_12246

theorem min_value_fraction (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l122_12246


namespace NUMINAMATH_CALUDE_box_height_is_55cm_l122_12221

/-- The height of the box Bob needs to reach the light fixture -/
def box_height (ceiling_height light_fixture_distance bob_height bob_reach : ℝ) : ℝ :=
  ceiling_height - light_fixture_distance - (bob_height + bob_reach)

/-- Theorem stating the height of the box Bob needs -/
theorem box_height_is_55cm :
  let ceiling_height : ℝ := 300
  let light_fixture_distance : ℝ := 15
  let bob_height : ℝ := 180
  let bob_reach : ℝ := 50
  box_height ceiling_height light_fixture_distance bob_height bob_reach = 55 := by
  sorry

#eval box_height 300 15 180 50

end NUMINAMATH_CALUDE_box_height_is_55cm_l122_12221


namespace NUMINAMATH_CALUDE_computer_speed_significant_figures_l122_12211

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (n : ScientificNotation) : ℕ :=
  sorry

/-- The given computer speed in scientific notation -/
def computerSpeed : ScientificNotation :=
  { coefficient := 2.09
    exponent := 10 }

/-- Theorem stating that the computer speed has 3 significant figures -/
theorem computer_speed_significant_figures :
  countSignificantFigures computerSpeed = 3 := by
  sorry

end NUMINAMATH_CALUDE_computer_speed_significant_figures_l122_12211


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l122_12260

theorem least_addition_for_divisibility : 
  (∃ x : ℕ, x ≥ 0 ∧ (228712 + x) % (2 * 3 * 5) = 0) ∧ 
  (∀ y : ℕ, y ≥ 0 ∧ (228712 + y) % (2 * 3 * 5) = 0 → y ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l122_12260


namespace NUMINAMATH_CALUDE_booklet_sheets_theorem_l122_12203

/-- Given a stack of sheets folded into a booklet, this function calculates
    the number of sheets in the original stack based on the sum of page numbers on one sheet. -/
def calculate_original_sheets (sum_of_page_numbers : ℕ) : ℕ :=
  (sum_of_page_numbers - 2) / 4

/-- Theorem stating that if the sum of page numbers on one sheet is 74,
    then the original stack contained 9 sheets. -/
theorem booklet_sheets_theorem (sum_is_74 : calculate_original_sheets 74 = 9) :
  calculate_original_sheets 74 = 9 := by
  sorry

#eval calculate_original_sheets 74  -- Should output 9

end NUMINAMATH_CALUDE_booklet_sheets_theorem_l122_12203


namespace NUMINAMATH_CALUDE_circle_through_M_same_center_as_C_l122_12220

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 11 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 13

-- State the theorem
theorem circle_through_M_same_center_as_C :
  ∀ (x y : ℝ),
  (∃ (h k r : ℝ), ∀ (u v : ℝ), circle_C u v ↔ (u - h)^2 + (v - k)^2 = r^2) →
  circle_equation point_M.1 point_M.2 ∧
  (∀ (u v : ℝ), circle_C u v ↔ circle_equation u v) :=
sorry

end NUMINAMATH_CALUDE_circle_through_M_same_center_as_C_l122_12220


namespace NUMINAMATH_CALUDE_adjacent_difference_at_least_six_l122_12230

/-- A 9x9 table containing integers from 1 to 81 -/
def Table : Type := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they share a side -/
def adjacent (a b : Fin 9 × Fin 9) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- The table contains all integers from 1 to 81 exactly once -/
def valid_table (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

theorem adjacent_difference_at_least_six (t : Table) (h : valid_table t) :
  ∃ (a b : Fin 9 × Fin 9), adjacent a b ∧ 
    ((t a.1 a.2).val + 6 ≤ (t b.1 b.2).val ∨ (t b.1 b.2).val + 6 ≤ (t a.1 a.2).val) :=
sorry

end NUMINAMATH_CALUDE_adjacent_difference_at_least_six_l122_12230


namespace NUMINAMATH_CALUDE_count_valid_sequences_l122_12280

/-- The set of digits to be used -/
def Digits : Finset Nat := {0, 1, 2, 3, 4}

/-- A function to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function to check if a digit sequence satisfies the condition -/
def validSequence (seq : List Nat) : Bool :=
  seq.length = 5 ∧ 
  seq.toFinset = Digits ∧
  ∃ i, i ∈ [1, 2, 3] ∧ 
    isEven (seq.nthLe i sorry) ∧ 
    ¬isEven (seq.nthLe (i-1) sorry) ∧ 
    ¬isEven (seq.nthLe (i+1) sorry)

/-- The main theorem -/
theorem count_valid_sequences : 
  (Digits.toList.permutations.filter validSequence).length = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l122_12280


namespace NUMINAMATH_CALUDE_special_arrangement_count_l122_12245

/-- The number of permutations of n distinct objects taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 people in a row with specific conditions -/
def special_arrangement : ℕ :=
  permutations 2 2 * permutations 4 4

theorem special_arrangement_count :
  special_arrangement = 48 :=
sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l122_12245


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l122_12228

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 30 * π / 180)  -- A = 30°
  (h2 : t.a = Real.sqrt 3)   -- a = √3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle is π
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ratio_theorem_l122_12228


namespace NUMINAMATH_CALUDE_sweater_selling_price_l122_12267

/-- The selling price of a sweater given the cost of materials and total gain -/
theorem sweater_selling_price 
  (balls_per_sweater : ℕ) 
  (cost_per_ball : ℕ) 
  (total_gain : ℕ) 
  (num_sweaters : ℕ) : 
  balls_per_sweater = 4 → 
  cost_per_ball = 6 → 
  total_gain = 308 → 
  num_sweaters = 28 → 
  (balls_per_sweater * cost_per_ball * num_sweaters + total_gain) / num_sweaters = 35 := by
  sorry

#check sweater_selling_price

end NUMINAMATH_CALUDE_sweater_selling_price_l122_12267


namespace NUMINAMATH_CALUDE_problem_proof_l122_12233

theorem problem_proof : -1^2023 + (-8) / (-4) - |(-5)| = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l122_12233


namespace NUMINAMATH_CALUDE_production_quantities_max_type_A_for_school_l122_12250

-- Define the parameters
def total_production : ℕ := 400000
def cost_A : ℚ := 1.2
def cost_B : ℚ := 0.4
def price_A : ℚ := 1.6
def price_B : ℚ := 0.6
def profit : ℚ := 110000
def school_budget : ℚ := 7680
def discount_A : ℚ := 0.1
def school_purchase : ℕ := 10000

-- Part 1: Production quantities
theorem production_quantities :
  ∃ (x y : ℕ),
    x + y = total_production ∧
    (price_A - cost_A) * x + (price_B - cost_B) * y = profit ∧
    x = 15000 ∧
    y = 25000 :=
sorry

-- Part 2: Maximum type A books for school
theorem max_type_A_for_school :
  ∃ (m : ℕ),
    m ≤ school_purchase ∧
    price_A * (1 - discount_A) * m + price_B * (school_purchase - m) ≤ school_budget ∧
    m = 2000 ∧
    ∀ n, n > m → 
      price_A * (1 - discount_A) * n + price_B * (school_purchase - n) > school_budget :=
sorry

end NUMINAMATH_CALUDE_production_quantities_max_type_A_for_school_l122_12250


namespace NUMINAMATH_CALUDE_minimal_intercept_line_properties_l122_12218

/-- A line that passes through (1, 4) with positive intercepts and minimal sum of intercepts -/
def minimal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem minimal_intercept_line_properties :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  minimal_intercept_line 1 4 ∧
  (∀ x y, minimal_intercept_line x y → x = 0 ∨ y = 0 → x = a ∨ y = b) ∧
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ x y, x + y = c + d ∧ (x = 0 ∨ y = 0)) →
    a + b ≤ c + d) :=
by sorry

end NUMINAMATH_CALUDE_minimal_intercept_line_properties_l122_12218


namespace NUMINAMATH_CALUDE_yard_sale_problem_l122_12275

theorem yard_sale_problem (total_items video_games dvds books working_video_games working_dvds : ℕ) 
  (h1 : total_items = 56)
  (h2 : video_games = 30)
  (h3 : dvds = 15)
  (h4 : books = total_items - video_games - dvds)
  (h5 : working_video_games = 20)
  (h6 : working_dvds = 10) :
  (video_games - working_video_games) + (dvds - working_dvds) = 15 := by
  sorry

end NUMINAMATH_CALUDE_yard_sale_problem_l122_12275


namespace NUMINAMATH_CALUDE_rectangle_area_l122_12271

/-- The area of a rectangle with perimeter 60 and width 10 is 200 -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 60) (h2 : width = 10) :
  2 * (perimeter / 2 - width) * width = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l122_12271


namespace NUMINAMATH_CALUDE_company_employees_l122_12249

/-- Proves that if a company had 15% more employees in December than in January,
    and it had 500 employees in December, then it had 435 employees in January. -/
theorem company_employees (january_employees : ℕ) (december_employees : ℕ) : 
  december_employees = 500 →
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 435 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l122_12249


namespace NUMINAMATH_CALUDE_inequality_proof_l122_12289

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ∧
  ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ≥ (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l122_12289


namespace NUMINAMATH_CALUDE_point_b_value_l122_12217

/-- Given a point A representing 3 on the number line, moving 3 units from A to reach point B 
    results in B representing either 0 or 6. -/
theorem point_b_value (A B : ℝ) : 
  A = 3 → (B - A = 3 ∨ A - B = 3) → (B = 0 ∨ B = 6) := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l122_12217


namespace NUMINAMATH_CALUDE_remainder_theorem_l122_12234

-- Define the polynomial q(x)
def q (x : ℝ) (D : ℝ) : ℝ := 2 * x^6 - 3 * x^4 + D * x^2 + 6

-- State the theorem
theorem remainder_theorem (D : ℝ) :
  q 2 D = 14 → q (-2) D = 158 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l122_12234


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l122_12281

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_increasing : increasing_on_nonneg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l122_12281


namespace NUMINAMATH_CALUDE_min_value_and_t_value_l122_12268

theorem min_value_and_t_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 2/b = 2) :
  (∃ (min : ℝ), min = 4 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 2 → 2*x + y ≥ min) ∧
  (∃ (t : ℝ), t = 6 ∧ 4^a = t ∧ 3^b = t) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_t_value_l122_12268


namespace NUMINAMATH_CALUDE_sequence_first_term_l122_12208

theorem sequence_first_term (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = 1 / (1 - a n)) →
  a 2 = 2 →
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_first_term_l122_12208


namespace NUMINAMATH_CALUDE_matrix_rank_theorem_l122_12261

theorem matrix_rank_theorem (m n : ℕ) (A : Matrix (Fin m) (Fin n) ℚ) 
  (h : ∃ (S : Finset ℕ), S.card ≥ m + n ∧ 
    (∀ p ∈ S, Nat.Prime p ∧ ∃ (i : Fin m) (j : Fin n), |A i j| = p)) : 
  Matrix.rank A ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_rank_theorem_l122_12261


namespace NUMINAMATH_CALUDE_common_points_on_line_l122_12222

-- Define the circles and line
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + (y - 1)^2 = a^2 ∧ a > 0
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def line (x y : ℝ) : Prop := y = 2*x

-- Define the theorem
theorem common_points_on_line (a : ℝ) : 
  (∀ x y : ℝ, circle1 a x y ∧ circle2 x y → line x y) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_points_on_line_l122_12222


namespace NUMINAMATH_CALUDE_log_inequality_l122_12248

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l122_12248


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l122_12253

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l122_12253


namespace NUMINAMATH_CALUDE_line_l_equation_circle_M_equations_l122_12251

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define perpendicularity of lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define the equation of line l
def l (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the equations of circle M
def M₁ (x y : ℝ) : Prop := (x + 5/7)^2 + (y + 10/7)^2 = 25/49
def M₂ (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∀ x y : ℝ, l x y ↔ (∃ m : ℝ, perpendicular m 2 ∧ y - P.2 = m * (x - P.1)) :=
sorry

-- Theorem for the equations of circle M
theorem circle_M_equations :
  ∀ x y : ℝ, 
    (∃ a b r : ℝ, 
      l₁ a b ∧ 
      (∀ t : ℝ, (t - a)^2 + b^2 = r^2 → t = 0) ∧ 
      ((a + b + 2)^2 / 2 + 1/2 = r^2) ∧
      ((x - a)^2 + (y - b)^2 = r^2)) 
    ↔ (M₁ x y ∨ M₂ x y) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_circle_M_equations_l122_12251


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l122_12237

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l122_12237


namespace NUMINAMATH_CALUDE_enter_exit_ways_count_l122_12279

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of ways to enter and exit the room -/
def ways_to_enter_and_exit : ℕ := num_doors * num_doors

/-- Theorem: The number of different ways to enter and exit a room with four doors is 64 -/
theorem enter_exit_ways_count : ways_to_enter_and_exit = 64 := by
  sorry

end NUMINAMATH_CALUDE_enter_exit_ways_count_l122_12279


namespace NUMINAMATH_CALUDE_bracelet_selling_price_l122_12262

def number_of_bracelets : ℕ := 12
def cost_per_bracelet : ℚ := 1
def cost_of_cookies : ℚ := 3
def money_left : ℚ := 3

def total_cost : ℚ := number_of_bracelets * cost_per_bracelet
def total_revenue : ℚ := cost_of_cookies + money_left + total_cost

theorem bracelet_selling_price :
  (total_revenue / number_of_bracelets : ℚ) = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_selling_price_l122_12262


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l122_12235

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property we want to prove
theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l122_12235


namespace NUMINAMATH_CALUDE_subway_construction_equation_l122_12277

/-- Represents the subway construction scenario -/
structure SubwayConstruction where
  total_length : ℝ
  extra_meters_per_day : ℝ
  days_saved : ℝ
  original_plan : ℝ

/-- The equation holds for the given subway construction scenario -/
def equation_holds (sc : SubwayConstruction) : Prop :=
  sc.total_length / sc.original_plan - sc.total_length / (sc.original_plan + sc.extra_meters_per_day) = sc.days_saved

/-- Theorem stating that the equation holds for the specific scenario described in the problem -/
theorem subway_construction_equation :
  ∀ (sc : SubwayConstruction),
    sc.total_length = 120 ∧
    sc.extra_meters_per_day = 5 ∧
    sc.days_saved = 4 →
    equation_holds sc :=
by
  sorry

#check subway_construction_equation

end NUMINAMATH_CALUDE_subway_construction_equation_l122_12277


namespace NUMINAMATH_CALUDE_max_profit_price_l122_12286

/-- Represents the profit function for a store selling items -/
def profit_function (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - purchase_price) * (base_quantity - price_sensitivity * (x - base_price))

theorem max_profit_price (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) 
    (h1 : purchase_price = 20)
    (h2 : base_price = 30)
    (h3 : base_quantity = 400)
    (h4 : price_sensitivity = 20) : 
  ∃ (max_price : ℝ), max_price = 35 ∧ 
    ∀ (x : ℝ), profit_function purchase_price base_price base_quantity price_sensitivity x ≤ 
               profit_function purchase_price base_price base_quantity price_sensitivity max_price :=
by sorry

#check max_profit_price

end NUMINAMATH_CALUDE_max_profit_price_l122_12286


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l122_12232

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, x^2 + 6*x*y + 9*y^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l122_12232


namespace NUMINAMATH_CALUDE_slope_of_intersection_line_l122_12291

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem slope_of_intersection_line (C D : ℝ × ℝ) (h : intersection C D) : 
  (D.2 - C.2) / (D.1 - C.1) = 11/6 := by sorry

end NUMINAMATH_CALUDE_slope_of_intersection_line_l122_12291


namespace NUMINAMATH_CALUDE_cyclist_round_trip_l122_12252

/-- A cyclist's round trip with given conditions -/
theorem cyclist_round_trip (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (distance1 : ℝ) (distance2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 12)
  (h3 : distance2 = 24)
  (h4 : speed1 = 8)
  (h5 : speed2 = 12)
  (h6 : total_time = 7.5) :
  (2 * total_distance) / (total_time - (distance1 / speed1 + distance2 / speed2)) = 9 := by
sorry

end NUMINAMATH_CALUDE_cyclist_round_trip_l122_12252


namespace NUMINAMATH_CALUDE_product_list_price_l122_12269

/-- Given a product with the following properties:
  - Sold at 90% of its list price
  - Earns a profit of 20%
  - Has a cost price of 21 yuan
  Prove that its list price is 28 yuan. -/
theorem product_list_price (list_price : ℝ) : 
  (0.9 * list_price - 21 = 21 * 0.2) → list_price = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_list_price_l122_12269


namespace NUMINAMATH_CALUDE_existence_of_special_set_l122_12241

theorem existence_of_special_set (n : ℕ) (hn : n ≥ 3) :
  ∃ (S : Finset ℕ),
    (Finset.card S = 2 * n) ∧
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n →
      ∃ (A : Finset ℕ),
        A ⊆ S ∧
        Finset.card A = m ∧
        2 * (A.sum id) = S.sum id) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l122_12241


namespace NUMINAMATH_CALUDE_well_digging_cost_l122_12223

/-- The cost of digging a cylindrical well -/
theorem well_digging_cost (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : 
  depth = 14 → diameter = 3 → cost_per_cubic_meter = 16 →
  ∃ (total_cost : ℝ), abs (total_cost - 1584.24) < 0.01 ∧ 
  total_cost = cost_per_cubic_meter * Real.pi * (diameter / 2)^2 * depth := by
sorry

end NUMINAMATH_CALUDE_well_digging_cost_l122_12223


namespace NUMINAMATH_CALUDE_sum_of_xyz_l122_12212

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3*x + 2*y - z = 12) : 
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l122_12212


namespace NUMINAMATH_CALUDE_max_distance_to_c_l122_12266

/-- The maximum distance from the origin to point C in an equilateral triangle ABC, 
    where A is on the unit circle and B is at (3,0) -/
theorem max_distance_to_c (A B C : ℝ × ℝ) : 
  (A.1^2 + A.2^2 = 1) →  -- A is on the unit circle
  (B = (3, 0)) →         -- B is at (3,0)
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- ABC is equilateral
  (∃ (D : ℝ × ℝ), (D.1^2 + D.2^2 = 1) ∧  -- D is another point on the unit circle
    (dist D B = dist B C ∧ dist B C = dist C D) →  -- DBC is also equilateral
    dist (0, 0) C ≤ 4) :=  -- The distance from O to C is at most 4
by sorry

end NUMINAMATH_CALUDE_max_distance_to_c_l122_12266


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l122_12259

/-- Calculates the total sleep for a week given specific sleep patterns -/
def totalSleepInWeek (daysInWeek : ℕ) (lowSleepDays : ℕ) (lowSleepHours : ℝ) 
                     (recommendedSleep : ℝ) (percentNormalSleep : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - lowSleepDays
  let normalSleepHours := recommendedSleep * percentNormalSleep
  lowSleepDays * lowSleepHours + normalSleepDays * normalSleepHours

/-- Proves that John's total sleep for the week is 30 hours -/
theorem johns_weekly_sleep : 
  totalSleepInWeek 7 2 3 8 0.6 = 30 := by
  sorry


end NUMINAMATH_CALUDE_johns_weekly_sleep_l122_12259


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l122_12202

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    the first term of the sequence is 768/81. -/
theorem geometric_sequence_first_term :
  ∀ (a r : ℚ),
    a * r^4 = 48 →
    a * r^5 = 72 →
    a = 768/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l122_12202


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_power_greater_than_100_l122_12265

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n := by sorry

theorem negation_of_power_greater_than_100 :
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_power_greater_than_100_l122_12265


namespace NUMINAMATH_CALUDE_daniel_noodles_l122_12292

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℝ := 54.0

/-- The number of noodles Daniel gave away -/
def given_away : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def remaining_noodles : ℝ := initial_noodles - given_away

theorem daniel_noodles : remaining_noodles = 42.0 := by sorry

end NUMINAMATH_CALUDE_daniel_noodles_l122_12292


namespace NUMINAMATH_CALUDE_base_4_of_185_l122_12229

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c d : ℕ) : ℕ :=
  a * (4^3) + b * (4^2) + c * (4^1) + d * (4^0)

/-- The base 4 representation of 185 (base 10) is 2321 --/
theorem base_4_of_185 : base4ToBase10 2 3 2 1 = 185 := by
  sorry

end NUMINAMATH_CALUDE_base_4_of_185_l122_12229


namespace NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l122_12274

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are strictly increasing -/
def FirstThreeIncreasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  FirstThreeIncreasing a ↔ MonotonicallyIncreasing a :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l122_12274


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l122_12207

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    is_valid_number n ∧
    digit_sum n = 14 →
    n ≤ 333322 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l122_12207


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l122_12215

/-- The equation of a circle symmetric to x^2 + y^2 = 1 with respect to the line x + y = 1 -/
theorem symmetric_circle_equation :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}
  let symmetric_circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}
  (∀ p ∈ C, ∃ q ∈ symmetric_circle, (q.1 + p.1)/2 = (q.2 + p.2)/2 ∧ (q.1 + p.1)/2 + (q.2 + p.2)/2 = 1) ∧
  (∀ q ∈ symmetric_circle, ∃ p ∈ C, (q.1 + p.1)/2 = (q.2 + p.2)/2 ∧ (q.1 + p.1)/2 + (q.2 + p.2)/2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l122_12215


namespace NUMINAMATH_CALUDE_factors_of_210_l122_12288

theorem factors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l122_12288


namespace NUMINAMATH_CALUDE_trajectory_of_Q_equation_l122_12244

/-- The trajectory of point Q given the conditions in the problem -/
def trajectory_of_Q (x y : ℝ) : Prop :=
  2 * x - y + 5 = 0

/-- The line on which point P moves -/
def line_of_P (x y : ℝ) : Prop :=
  2 * x - y + 3 = 0

/-- Point M is fixed at (-1, 2) -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Q is on the extension line of PM and PM = MQ -/
def Q_condition (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q.1 - point_M.1 = t * (point_M.1 - P.1) ∧ 
                    Q.2 - point_M.2 = t * (point_M.2 - P.2)

theorem trajectory_of_Q_equation :
  ∀ x y : ℝ, 
    (∃ P : ℝ × ℝ, line_of_P P.1 P.2 ∧ Q_condition P (x, y)) →
    trajectory_of_Q x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_equation_l122_12244


namespace NUMINAMATH_CALUDE_bluray_price_l122_12240

/-- The price of a Blu-ray movie given the following conditions:
  * 8 DVDs cost $12 each
  * There are 4 Blu-ray movies
  * The average price of all 12 movies is $14
-/
theorem bluray_price :
  ∀ (x : ℝ),
  (8 * 12 + 4 * x) / 12 = 14 →
  x = 18 :=
by sorry

end NUMINAMATH_CALUDE_bluray_price_l122_12240


namespace NUMINAMATH_CALUDE_solution_set_f_non_empty_solution_set_l122_12219

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 7| + 3*m

-- Theorem 1: Solution set of f(x) + x^2 - 4 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 := by sorry

-- Theorem 2: Condition for non-empty solution set of f(x) < g(x)
theorem non_empty_solution_set (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_non_empty_solution_set_l122_12219


namespace NUMINAMATH_CALUDE_lottery_probability_l122_12295

/-- A lottery with probabilities for certain number ranges -/
structure Lottery where
  prob_1_to_45 : ℚ
  prob_1_or_larger : ℚ
  prob_1_to_45_is_valid : prob_1_to_45 = 7/15
  prob_1_or_larger_is_valid : prob_1_or_larger = 14/15

/-- The probability of drawing a number less than or equal to 45 in the lottery -/
def prob_le_45 (l : Lottery) : ℚ := l.prob_1_to_45

theorem lottery_probability (l : Lottery) :
  prob_le_45 l = l.prob_1_to_45 := by sorry

end NUMINAMATH_CALUDE_lottery_probability_l122_12295


namespace NUMINAMATH_CALUDE_tony_initial_money_l122_12283

/-- Given Tony's expenses and remaining money, prove his initial amount --/
theorem tony_initial_money :
  ∀ (initial spent_ticket spent_hotdog remaining : ℕ),
    spent_ticket = 8 →
    spent_hotdog = 3 →
    remaining = 9 →
    initial = spent_ticket + spent_hotdog + remaining →
    initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_tony_initial_money_l122_12283


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l122_12296

theorem least_n_satisfying_inequality : ∀ n : ℕ, n > 0 → 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l122_12296


namespace NUMINAMATH_CALUDE_parabola_c_value_l122_12298

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  (p.x_coord 4 = 5) →  -- vertex at (5,4)
  (p.x_coord 6 = 1) →  -- passes through (1,6)
  (p.x_coord 0 = -27) →  -- passes through (-27,0)
  p.c = -27 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l122_12298


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l122_12294

theorem absolute_value_calculation : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l122_12294


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l122_12272

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l122_12272


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l122_12238

theorem fraction_sum_zero (a b c : ℤ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (avg : b = (a + c) / 2)
  (sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l122_12238


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l122_12278

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = Complex.I) : 
  z.im = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l122_12278


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_length_l122_12206

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Determines if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_segment_length 
  (PQRS : Quadrilateral) 
  (T : Point) :
  isConvex PQRS →
  distance PQRS.P PQRS.Q = 15 →
  distance PQRS.R PQRS.S = 20 →
  distance PQRS.P PQRS.R = 25 →
  T = intersection PQRS.P PQRS.R PQRS.Q PQRS.S →
  triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S →
  distance PQRS.P T = 75 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_length_l122_12206


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l122_12239

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_geometric : geometric_sequence b)
  (h_a_sum : a 1001 + a 1015 = Real.pi)
  (h_b_prod : b 6 * b 9 = 2) :
  Real.tan ((a 1 + a 2015) / (1 + b 7 * b 8)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l122_12239


namespace NUMINAMATH_CALUDE_line_point_z_coordinate_l122_12257

/-- Given a line passing through two points in 3D space, 
    find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) 
  (h1 : p1 = (1, 3, 2)) 
  (h2 : p2 = (4, 2, -1)) 
  (h3 : x = 7) : 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z)) ∧ z = -4 :=
sorry

end NUMINAMATH_CALUDE_line_point_z_coordinate_l122_12257


namespace NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l122_12297

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: Given the conditions, prove that the new average is 92 -/
theorem batsman_average_after_20th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 19)
  (h2 : newAverage stats 130 = stats.average + 2)
  : newAverage stats 130 = 92 := by
  sorry

#check batsman_average_after_20th_innings

end NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l122_12297


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l122_12254

theorem triangle_expression_simplification (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l122_12254


namespace NUMINAMATH_CALUDE_angle_CAD_is_15_degrees_l122_12264

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a square in 2D space -/
structure Square where
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- Calculates the angle between three points in degrees -/
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def isSquare (s : Square) : Prop := sorry

/-- Theorem: In a coplanar configuration where ABC is an equilateral triangle 
    and BCDE is a square, the measure of angle CAD is 15 degrees -/
theorem angle_CAD_is_15_degrees 
  (A B C D E : Point2D) 
  (triangle : Triangle) 
  (square : Square) : 
  triangle.A = A ∧ triangle.B = B ∧ triangle.C = C ∧
  square.B = B ∧ square.C = C ∧ square.D = D ∧ square.E = E ∧
  isEquilateral triangle ∧ 
  isSquare square → 
  angle C A D = 15 := by
  sorry

end NUMINAMATH_CALUDE_angle_CAD_is_15_degrees_l122_12264


namespace NUMINAMATH_CALUDE_fayes_initial_money_l122_12224

/-- Proves that Faye's initial amount of money was $20 --/
theorem fayes_initial_money :
  ∀ (X : ℝ),
  (X + 2*X - (10*1.5 + 5*3) = 30) →
  X = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fayes_initial_money_l122_12224


namespace NUMINAMATH_CALUDE_fred_total_cards_l122_12243

def initial_cards : ℕ := 26
def cards_given_away : ℕ := 18
def new_cards_found : ℕ := 40

theorem fred_total_cards : 
  initial_cards - cards_given_away + new_cards_found = 48 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_cards_l122_12243


namespace NUMINAMATH_CALUDE_factorial_15_base_18_trailing_zeros_l122_12204

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def countTrailingZerosBase18 (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

theorem factorial_15_base_18_trailing_zeros :
  countTrailingZerosBase18 (factorial 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_15_base_18_trailing_zeros_l122_12204


namespace NUMINAMATH_CALUDE_g_four_to_four_l122_12242

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^4 = 16 -/
theorem g_four_to_four (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 16 = 16) : 
  (g 4)^4 = 16 := by
sorry

end NUMINAMATH_CALUDE_g_four_to_four_l122_12242


namespace NUMINAMATH_CALUDE_square_difference_l122_12231

theorem square_difference (n : ℕ) (h : n = 50) : n^2 - (n-1)^2 = 2*n - 1 := by
  sorry

#check square_difference

end NUMINAMATH_CALUDE_square_difference_l122_12231


namespace NUMINAMATH_CALUDE_eleven_power_2023_mod_5_l122_12247

theorem eleven_power_2023_mod_5 : 11^2023 % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_eleven_power_2023_mod_5_l122_12247


namespace NUMINAMATH_CALUDE_pascal_triangle_row_34_l122_12290

theorem pascal_triangle_row_34 : 
  let row_34 := List.range 35
  let nth_elem (n : ℕ) := Nat.choose 34 n
  (nth_elem 29 = 278256) ∧ (nth_elem 30 = 46376) := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_34_l122_12290


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l122_12284

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to a natural number -/
def from_binary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem base_2_representation_of_123 :
  to_binary 123 = [true, true, true, true, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l122_12284


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_11_pow_2002_l122_12236

theorem sum_of_last_two_digits_of_11_pow_2002 : 
  ∃ (n : ℕ), 11^2002 = n * 100 + 21 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_11_pow_2002_l122_12236


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l122_12210

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ x/4 + 3/(4*x) = 1 ∧ ∀ (y : ℝ), y > 0 ∧ y/4 + 3/(4*y) = 1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l122_12210


namespace NUMINAMATH_CALUDE_consecutive_ones_count_l122_12273

def a : ℕ → ℕ
  | 0 => 1  -- We define a(0) = 1 to simplify the recursion
  | 1 => 2
  | 2 => 3
  | (n + 3) => a (n + 2) + a (n + 1)

theorem consecutive_ones_count : 
  (2^8 : ℕ) - a 8 = 201 :=
sorry

end NUMINAMATH_CALUDE_consecutive_ones_count_l122_12273


namespace NUMINAMATH_CALUDE_square_remainder_l122_12285

theorem square_remainder (n x : ℤ) (h : n % x = 3) : (n^2) % x = 9 % x := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l122_12285


namespace NUMINAMATH_CALUDE_min_points_in_S_l122_12227

-- Define a point in the xy-plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the set S
def S : Set Point := sorry

-- Define symmetry conditions
def symmetric_origin (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) (-p.y) ∈ s

def symmetric_x_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.x (-p.y) ∈ s

def symmetric_y_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) p.y ∈ s

def symmetric_y_eq_x (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.y p.x ∈ s

-- Theorem statement
theorem min_points_in_S :
  symmetric_origin S ∧
  symmetric_x_axis S ∧
  symmetric_y_axis S ∧
  symmetric_y_eq_x S ∧
  Point.mk 2 3 ∈ S →
  ∃ (points : Finset Point), points.card = 8 ∧ ↑points ⊆ S ∧
    (∀ (subset : Finset Point), ↑subset ⊆ S → subset.card < 8 → subset ≠ points) :=
sorry

end NUMINAMATH_CALUDE_min_points_in_S_l122_12227


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l122_12263

/-- A rectangular plot with given perimeter and short side length has a specific ratio of long to short sides -/
theorem rectangular_plot_ratio (perimeter : ℝ) (short_side : ℝ) 
  (h_perimeter : perimeter = 640) 
  (h_short_side : short_side = 80) : 
  (perimeter / 2 - short_side) / short_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l122_12263


namespace NUMINAMATH_CALUDE_encryption_decryption_l122_12209

/-- Given an encryption formula y = a^x - 2, prove that when a^3 - 2 = 6 and y = 14, x = 4 --/
theorem encryption_decryption (a : ℝ) (h1 : a^3 - 2 = 6) (y : ℝ) (h2 : y = 14) :
  ∃ x : ℝ, a^x - 2 = y ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_encryption_decryption_l122_12209
