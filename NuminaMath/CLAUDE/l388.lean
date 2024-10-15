import Mathlib

namespace NUMINAMATH_CALUDE_melinda_doughnuts_count_l388_38894

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of coffees Melinda bought -/
def melinda_coffees : ℕ := 6

/-- The number of doughnuts Melinda bought -/
def melinda_doughnuts : ℕ := 5

theorem melinda_doughnuts_count : 
  ∃ (coffee_cost : ℚ), 
    (harold_doughnuts : ℚ) * doughnut_cost + (harold_coffees : ℚ) * coffee_cost = harold_total ∧
    (melinda_doughnuts : ℚ) * doughnut_cost + (melinda_coffees : ℚ) * coffee_cost = melinda_total :=
by sorry

end NUMINAMATH_CALUDE_melinda_doughnuts_count_l388_38894


namespace NUMINAMATH_CALUDE_cosine_sum_squared_l388_38802

theorem cosine_sum_squared : 
  (Real.cos (42 * π / 180) + Real.cos (102 * π / 180) + 
   Real.cos (114 * π / 180) + Real.cos (174 * π / 180))^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_squared_l388_38802


namespace NUMINAMATH_CALUDE_watermelons_left_after_sales_l388_38832

def initial_watermelons : ℕ := 10 * 12

def yesterday_sale_percentage : ℚ := 40 / 100

def today_sale_fraction : ℚ := 1 / 4

def tomorrow_sale_multiplier : ℚ := 3 / 2

def discount_threshold : ℕ := 10

theorem watermelons_left_after_sales : 
  let yesterday_sale := initial_watermelons * yesterday_sale_percentage
  let after_yesterday := initial_watermelons - yesterday_sale
  let today_sale := after_yesterday * today_sale_fraction
  let after_today := after_yesterday - today_sale
  let tomorrow_sale := today_sale * tomorrow_sale_multiplier
  after_today - tomorrow_sale = 27 := by sorry

end NUMINAMATH_CALUDE_watermelons_left_after_sales_l388_38832


namespace NUMINAMATH_CALUDE_quadratic_sum_l388_38809

/-- A quadratic function with vertex at (2, 5) passing through (3, 2) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ :=
  fun x ↦ d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 2 = 5) →
  (QuadraticFunction d e f 3 = 2) →
  d + e + 2*f = -5 := by
    sorry

end NUMINAMATH_CALUDE_quadratic_sum_l388_38809


namespace NUMINAMATH_CALUDE_bob_yogurt_order_l388_38814

-- Define the problem parameters
def expired_percentage : ℚ := 40 / 100
def pack_cost : ℚ := 12
def total_refund : ℚ := 384

-- State the theorem
theorem bob_yogurt_order :
  ∃ (total_packs : ℚ),
    total_packs * expired_percentage * pack_cost = total_refund ∧
    total_packs = 80 := by
  sorry

end NUMINAMATH_CALUDE_bob_yogurt_order_l388_38814


namespace NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l388_38848

/-- The weight of each hamburger in ounces -/
def hamburger_weight : ℕ := 4

/-- The total weight in ounces eaten by last year's winner -/
def last_year_winner_weight : ℕ := 84

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record : ℕ := 
  (last_year_winner_weight / hamburger_weight) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's record -/
theorem tonya_needs_22_hamburgers : 
  hamburgers_to_beat_record = 22 := by sorry

end NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l388_38848


namespace NUMINAMATH_CALUDE_janelle_marbles_l388_38868

/-- Calculates the total number of marbles Janelle has after a series of transactions. -/
def total_marbles (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (gifted_red : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (gift_red : ℕ) 
  (returned_blue : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let remaining_green := initial_green - gift_green
  let remaining_blue := total_blue - gift_blue + returned_blue
  let remaining_red := gifted_red - gift_red
  remaining_green + remaining_blue + remaining_red

/-- Proves that Janelle ends up with 197 marbles given the initial conditions and transactions. -/
theorem janelle_marbles : 
  total_marbles 26 12 15 7 9 12 3 8 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janelle_marbles_l388_38868


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l388_38867

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a + 3)*x - 4*a + 3 else a^x

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l388_38867


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l388_38827

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l388_38827


namespace NUMINAMATH_CALUDE_reassignment_count_l388_38818

/-- The number of people and jobs -/
def n : ℕ := 5

/-- The number of ways to reassign n jobs to n people such that at least 2 people change jobs -/
def reassignments (n : ℕ) : ℕ := n.factorial - 1

/-- Theorem: The number of ways to reassign 5 jobs to 5 people, 
    such that at least 2 people change jobs from their initial assignment, is 5! - 1 -/
theorem reassignment_count : reassignments n = 119 := by
  sorry

end NUMINAMATH_CALUDE_reassignment_count_l388_38818


namespace NUMINAMATH_CALUDE_inequality_for_positive_product_l388_38835

theorem inequality_for_positive_product (a b : ℝ) (h : a * b > 0) :
  b / a + a / b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_product_l388_38835


namespace NUMINAMATH_CALUDE_parabola_segment_length_l388_38839

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Length of PQ on a parabola -/
theorem parabola_segment_length
  (p : ℝ)
  (hp : p > 0)
  (P Q : ParabolaPoint)
  (hP : P.y^2 = 2*p*P.x)
  (hQ : Q.y^2 = 2*p*Q.x)
  (h_sum : P.x + Q.x = 3*p) :
  |P.x - Q.x| + p = 4*p :=
by sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l388_38839


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l388_38875

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l388_38875


namespace NUMINAMATH_CALUDE_same_color_probability_l388_38856

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 30

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 8

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 9

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 10

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 2

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating that the probability of rolling the same color on both dice is 25/90 -/
theorem same_color_probability :
  (purpleSides^2 + orangeSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 25 / 90 := by
  sorry


end NUMINAMATH_CALUDE_same_color_probability_l388_38856


namespace NUMINAMATH_CALUDE_solve_for_B_l388_38807

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 29) ∧ (B = 7) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_B_l388_38807


namespace NUMINAMATH_CALUDE_a_oxen_count_l388_38823

/-- Represents the number of oxen and months for each person renting the pasture -/
structure Grazing where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent based on oxen and months -/
def calculateShare (g : Grazing) (totalRent : ℚ) (totalProduct : ℕ) : ℚ :=
  totalRent * (g.oxen * g.months : ℚ) / totalProduct

theorem a_oxen_count (a : Grazing) (b c : Grazing) 
    (h1 : b.oxen = 12 ∧ b.months = 5)
    (h2 : c.oxen = 15 ∧ c.months = 3)
    (h3 : a.months = 7)
    (h4 : calculateShare c 245 (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) = 63) :
  a.oxen = 17 := by
  sorry

#check a_oxen_count

end NUMINAMATH_CALUDE_a_oxen_count_l388_38823


namespace NUMINAMATH_CALUDE_min_median_length_l388_38869

/-- In a right triangle with height h dropped onto the hypotenuse,
    the minimum length of the median that bisects the longer leg is (3/2) * h. -/
theorem min_median_length (h : ℝ) (h_pos : h > 0) :
  ∃ (m : ℝ), m ≥ (3/2) * h ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = h^2 →
    ((x/2 + y)^2 + (h/2)^2).sqrt ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_median_length_l388_38869


namespace NUMINAMATH_CALUDE_mean_equality_implies_sum_l388_38884

theorem mean_equality_implies_sum (x y : ℝ) : 
  (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3 → x + y = 26.5 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_sum_l388_38884


namespace NUMINAMATH_CALUDE_negative_510_in_third_quadrant_l388_38854

-- Define a function to normalize an angle to the range [-360°, 0°)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360 - 360

-- Define a function to determine the quadrant of an angle
def quadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if -270 < normalizedAngle && normalizedAngle ≤ -180 then 3
  else if -180 < normalizedAngle && normalizedAngle ≤ -90 then 2
  else if -90 < normalizedAngle && normalizedAngle ≤ 0 then 1
  else 4

-- Theorem: -510° is in the third quadrant
theorem negative_510_in_third_quadrant : quadrant (-510) = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_510_in_third_quadrant_l388_38854


namespace NUMINAMATH_CALUDE_tournament_winning_group_exists_l388_38810

/-- A directed graph representing a tournament. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- The property that player i wins against player j. -/
def Wins (t : Tournament n) (i j : Fin n) : Prop := t i j

/-- An ordered group of four players satisfying the winning condition. -/
def WinningGroup (t : Tournament n) (a₁ a₂ a₃ a₄ : Fin n) : Prop :=
  Wins t a₁ a₂ ∧ Wins t a₁ a₃ ∧ Wins t a₁ a₄ ∧
  Wins t a₂ a₃ ∧ Wins t a₂ a₄ ∧
  Wins t a₃ a₄

/-- The main theorem: For n = 8, every tournament has a winning group,
    and this property does not hold for n < 8. -/
theorem tournament_winning_group_exists :
  (∀ (t : Tournament 8), ∃ a₁ a₂ a₃ a₄, WinningGroup t a₁ a₂ a₃ a₄) ∧
  (∀ n < 8, ∃ (t : Tournament n), ∀ a₁ a₂ a₃ a₄, ¬WinningGroup t a₁ a₂ a₃ a₄) :=
sorry

end NUMINAMATH_CALUDE_tournament_winning_group_exists_l388_38810


namespace NUMINAMATH_CALUDE_eggs_left_for_breakfast_l388_38846

def total_eggs : ℕ := 36

def eggs_for_crepes : ℕ := (2 * total_eggs) / 5

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (3 * eggs_after_crepes) / 7

def eggs_after_cupcakes : ℕ := eggs_after_crepes - eggs_for_cupcakes

def eggs_for_quiche : ℕ := eggs_after_cupcakes / 2

def eggs_left : ℕ := eggs_after_cupcakes - eggs_for_quiche

theorem eggs_left_for_breakfast : eggs_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_for_breakfast_l388_38846


namespace NUMINAMATH_CALUDE_line_equation_problem_l388_38878

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m

/-- Point in ℝ² -/
def Point := ℝ × ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (line : Set Point) : Point := sorry

/-- The problem statement -/
theorem line_equation_problem (lines : TwoLines) 
  (h1 : (0, 0) ∈ lines.ℓ ∩ lines.m)
  (h2 : ∀ x y, (x, y) ∈ lines.ℓ ↔ 3 * x + 4 * y = 0)
  (h3 : reflect (-2, 3) lines.ℓ = reflect (3, -2) lines.m) :
  ∀ x y, (x, y) ∈ lines.m ↔ 7 * x - 25 * y = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_problem_l388_38878


namespace NUMINAMATH_CALUDE_dans_gift_l388_38860

-- Define the number of cards Sally sold
def cards_sold : ℕ := 27

-- Define the number of cards Sally bought
def cards_bought : ℕ := 20

-- Define the total number of cards Sally has now
def total_cards : ℕ := 34

-- Theorem to prove
theorem dans_gift (cards_from_dan : ℕ) : 
  cards_from_dan = total_cards - cards_bought := by
  sorry

#check dans_gift

end NUMINAMATH_CALUDE_dans_gift_l388_38860


namespace NUMINAMATH_CALUDE_boa_constrictors_count_l388_38883

/-- The number of boa constrictors in the park -/
def num_boa : ℕ := sorry

/-- The number of pythons in the park -/
def num_python : ℕ := sorry

/-- The number of rattlesnakes in the park -/
def num_rattlesnake : ℕ := 40

/-- The total number of snakes in the park -/
def total_snakes : ℕ := 200

theorem boa_constrictors_count :
  (num_boa + num_python + num_rattlesnake = total_snakes) →
  (num_python = 3 * num_boa) →
  (num_boa = 40) :=
by sorry

end NUMINAMATH_CALUDE_boa_constrictors_count_l388_38883


namespace NUMINAMATH_CALUDE_lcm_36_132_l388_38886

theorem lcm_36_132 : Nat.lcm 36 132 = 396 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_132_l388_38886


namespace NUMINAMATH_CALUDE_greatest_square_power_of_three_under_200_l388_38803

theorem greatest_square_power_of_three_under_200 : ∃ n : ℕ, 
  n < 200 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  (∃ k : ℕ, n = 3^k) ∧
  (∀ x : ℕ, x < 200 → (∃ y : ℕ, x = y^2) → (∃ z : ℕ, x = 3^z) → x ≤ n) ∧
  n = 81 :=
by sorry

end NUMINAMATH_CALUDE_greatest_square_power_of_three_under_200_l388_38803


namespace NUMINAMATH_CALUDE_arithmetic_progression_cos_sum_l388_38881

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_cos_sum (a : ℕ → ℝ) :
  is_arithmetic_progression a →
  a 1 + a 7 + a 13 = 4 * Real.pi →
  Real.cos (a 2 + a 12) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cos_sum_l388_38881


namespace NUMINAMATH_CALUDE_small_pizza_has_eight_slices_l388_38838

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of people -/
def num_people : ℕ := 3

/-- The number of slices each person can eat -/
def slices_per_person : ℕ := 12

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 14

/-- The number of small pizzas ordered -/
def num_small_pizzas : ℕ := 1

/-- The number of large pizzas ordered -/
def num_large_pizzas : ℕ := 2

theorem small_pizza_has_eight_slices :
  small_pizza_slices = 8 ∧
  num_people * slices_per_person ≤ 
    num_small_pizzas * small_pizza_slices + num_large_pizzas * large_pizza_slices :=
by sorry

end NUMINAMATH_CALUDE_small_pizza_has_eight_slices_l388_38838


namespace NUMINAMATH_CALUDE_ball_size_ratio_l388_38842

/-- Given three balls A, B, and C with different sizes, where A is three times bigger than B,
    and B is half the size of C, prove that A is 1.5 times the size of C. -/
theorem ball_size_ratio :
  ∀ (size_A size_B size_C : ℝ),
  size_A > 0 → size_B > 0 → size_C > 0 →
  size_A = 3 * size_B →
  size_B = (1 / 2) * size_C →
  size_A = (3 / 2) * size_C :=
by
  sorry

end NUMINAMATH_CALUDE_ball_size_ratio_l388_38842


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l388_38800

/-- A geometric sequence with first term 1024 and sixth term 125 has its fourth term equal to 2000 -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ), 
  (∃ r : ℝ, ∀ n : ℕ, a n = 1024 * r ^ (n - 1)) →  -- Geometric sequence definition
  a 1 = 1024 →                                   -- First term condition
  a 6 = 125 →                                    -- Sixth term condition
  a 4 = 2000 :=                                  -- Fourth term (to prove)
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l388_38800


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l388_38866

theorem contrapositive_equivalence (x : ℝ) :
  (¬ (-2 < x ∧ x < 2) → ¬ (x^2 < 4)) ↔ ((x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l388_38866


namespace NUMINAMATH_CALUDE_alex_cell_phone_cost_l388_38893

/-- Calculates the total cost of a cell phone plan -/
def calculate_total_cost (base_cost : ℚ) (cost_per_text : ℚ) (cost_per_extra_minute : ℚ) 
  (included_hours : ℕ) (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_cost := cost_per_text * texts_sent
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost := cost_per_extra_minute * extra_minutes
  base_cost + text_cost + extra_minutes_cost

/-- The total cost of Alex's cell phone plan is $48.50 -/
theorem alex_cell_phone_cost : 
  calculate_total_cost 20 (7/100) (15/100) 30 150 32 = 485/10 := by
  sorry

end NUMINAMATH_CALUDE_alex_cell_phone_cost_l388_38893


namespace NUMINAMATH_CALUDE_second_hour_billboards_l388_38815

/-- The number of billboards counted in the second hour -/
def billboards_second_hour (first_hour : ℕ) (third_hour : ℕ) (total_hours : ℕ) (average : ℕ) : ℕ :=
  average * total_hours - first_hour - third_hour

theorem second_hour_billboards :
  billboards_second_hour 17 23 3 20 = 20 := by
  sorry

#eval billboards_second_hour 17 23 3 20

end NUMINAMATH_CALUDE_second_hour_billboards_l388_38815


namespace NUMINAMATH_CALUDE_algal_bloom_characteristic_l388_38892

/-- Represents the characteristics of algal population growth --/
inductive AlgalGrowthCharacteristic
  | IrregularFluctuations
  | UnevenDistribution
  | RapidGrowthShortPeriod
  | SeasonalGrowthDecline

/-- Represents the nutrient level in a water body --/
inductive NutrientLevel
  | Oligotrophic
  | Mesotrophic
  | Eutrophic

/-- Represents an algal bloom event --/
structure AlgalBloom where
  nutrientLevel : NutrientLevel
  growthCharacteristic : AlgalGrowthCharacteristic

/-- Theorem stating that algal blooms in eutrophic water bodies are characterized by rapid growth in a short period --/
theorem algal_bloom_characteristic (bloom : AlgalBloom) 
  (h : bloom.nutrientLevel = NutrientLevel.Eutrophic) : 
  bloom.growthCharacteristic = AlgalGrowthCharacteristic.RapidGrowthShortPeriod := by
  sorry

end NUMINAMATH_CALUDE_algal_bloom_characteristic_l388_38892


namespace NUMINAMATH_CALUDE_hadley_books_l388_38837

theorem hadley_books (initial_books : ℕ) 
  (h1 : initial_books - 50 + 40 - 30 = 60) : initial_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_hadley_books_l388_38837


namespace NUMINAMATH_CALUDE_peach_basket_ratios_and_percentages_l388_38876

/-- Represents the number of peaches of each color in the basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of peaches in the basket -/
def totalPeaches (basket : PeachBasket) : ℕ :=
  basket.red + basket.yellow + basket.green + basket.orange

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of a specific color to the total -/
def colorRatio (count : ℕ) (total : ℕ) : Ratio :=
  let gcd := Nat.gcd count total
  { numerator := count / gcd, denominator := total / gcd }

/-- Calculates the percentage of a specific color -/
def colorPercentage (count : ℕ) (total : ℕ) : Float :=
  (count.toFloat / total.toFloat) * 100

theorem peach_basket_ratios_and_percentages
  (basket : PeachBasket)
  (h_red : basket.red = 8)
  (h_yellow : basket.yellow = 14)
  (h_green : basket.green = 6)
  (h_orange : basket.orange = 4) :
  let total := totalPeaches basket
  (colorRatio basket.green total = Ratio.mk 3 16) ∧
  (colorRatio basket.yellow total = Ratio.mk 7 16) ∧
  (colorPercentage basket.green total = 18.75) ∧
  (colorPercentage basket.yellow total = 43.75) := by
  sorry


end NUMINAMATH_CALUDE_peach_basket_ratios_and_percentages_l388_38876


namespace NUMINAMATH_CALUDE_time_per_check_is_two_minutes_l388_38833

/-- The time per check for lice checks at an elementary school -/
def time_per_check : ℕ :=
  let kindergarteners : ℕ := 26
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let total_students : ℕ := kindergarteners + first_graders + second_graders + third_graders
  let total_time_hours : ℕ := 3
  let total_time_minutes : ℕ := total_time_hours * 60
  total_time_minutes / total_students

/-- Theorem stating that the time per check is 2 minutes -/
theorem time_per_check_is_two_minutes : time_per_check = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_per_check_is_two_minutes_l388_38833


namespace NUMINAMATH_CALUDE_sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l388_38844

theorem sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1 :
  0 < Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) ∧
  Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l388_38844


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l388_38805

def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -4 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l388_38805


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l388_38811

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 12 → x = 128 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l388_38811


namespace NUMINAMATH_CALUDE_expression_exists_l388_38885

/-- Represents an expression formed by ones and operators --/
inductive Expression
  | One : Expression
  | Add : Expression → Expression → Expression
  | Mul : Expression → Expression → Expression

/-- Evaluates the expression --/
def evaluate : Expression → ℕ
  | Expression.One => 1
  | Expression.Add e1 e2 => evaluate e1 + evaluate e2
  | Expression.Mul e1 e2 => evaluate e1 * evaluate e2

/-- Swaps the operators in the expression --/
def swap_operators : Expression → Expression
  | Expression.One => Expression.One
  | Expression.Add e1 e2 => Expression.Mul (swap_operators e1) (swap_operators e2)
  | Expression.Mul e1 e2 => Expression.Add (swap_operators e1) (swap_operators e2)

/-- Theorem stating the existence of the required expression --/
theorem expression_exists : ∃ (e : Expression), 
  evaluate e = 2014 ∧ evaluate (swap_operators e) = 2014 := by
  sorry


end NUMINAMATH_CALUDE_expression_exists_l388_38885


namespace NUMINAMATH_CALUDE_interest_rate_is_five_paise_l388_38891

/-- Calculates the interest rate in paise per rupee per month given the principal, time, and simple interest -/
def interest_rate_paise (principal : ℚ) (time_months : ℚ) (simple_interest : ℚ) : ℚ :=
  (simple_interest / (principal * time_months)) * 100

/-- Theorem stating that for the given conditions, the interest rate is 5 paise per rupee per month -/
theorem interest_rate_is_five_paise 
  (principal : ℚ) 
  (time_months : ℚ) 
  (simple_interest : ℚ) 
  (h1 : principal = 20)
  (h2 : time_months = 6)
  (h3 : simple_interest = 6) :
  interest_rate_paise principal time_months simple_interest = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_five_paise_l388_38891


namespace NUMINAMATH_CALUDE_min_days_for_eleven_groups_l388_38813

/-- Represents a festival schedule --/
structure FestivalSchedule where
  days : ℕ
  groups : ℕ
  performing : Fin days → Finset (Fin groups)
  watching : Fin days → Finset (Fin groups)

/-- Checks if a festival schedule is valid --/
def isValidSchedule (s : FestivalSchedule) : Prop :=
  (∀ d, s.performing d ∩ s.watching d = ∅) ∧
  (∀ d, s.performing d ∪ s.watching d = Finset.univ) ∧
  (∀ g₁ g₂, g₁ ≠ g₂ → ∃ d, g₁ ∈ s.watching d ∧ g₂ ∈ s.performing d)

/-- The main theorem --/
theorem min_days_for_eleven_groups :
  ∃ (s : FestivalSchedule), s.groups = 11 ∧ s.days = 6 ∧ isValidSchedule s ∧
  ∀ (s' : FestivalSchedule), s'.groups = 11 → isValidSchedule s' → s'.days ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_min_days_for_eleven_groups_l388_38813


namespace NUMINAMATH_CALUDE_percentage_problem_l388_38887

theorem percentage_problem (P : ℝ) : 25 = (P / 100) * 25 + 21 → P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l388_38887


namespace NUMINAMATH_CALUDE_max_value_polynomial_l388_38864

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 656^2 / 18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l388_38864


namespace NUMINAMATH_CALUDE_toothpick_20th_stage_l388_38804

def toothpick_sequence (n : ℕ) : ℕ := 5 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 62 := by
sorry

end NUMINAMATH_CALUDE_toothpick_20th_stage_l388_38804


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l388_38841

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l388_38841


namespace NUMINAMATH_CALUDE_maggie_yellow_packs_l388_38821

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs (red_packs green_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : ℕ :=
  (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

/-- Theorem stating that Maggie bought 8 packs of yellow bouncy balls -/
theorem maggie_yellow_packs : yellow_packs 4 4 10 160 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maggie_yellow_packs_l388_38821


namespace NUMINAMATH_CALUDE_final_x_value_l388_38857

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (x : ℕ)
  (S : ℕ)

/-- The initial state of the program -/
def initial_state : ProgramState :=
  { x := 3, S := 0 }

/-- Updates the program state for one iteration -/
def update_state (state : ProgramState) : ProgramState :=
  { x := state.x + 2, S := state.S + (state.x + 2) }

/-- Predicate to check if the loop should continue -/
def continue_loop (state : ProgramState) : Prop :=
  state.S < 10000

/-- The final state of the program after all iterations -/
noncomputable def final_state : ProgramState :=
  sorry  -- The actual computation of the final state

/-- Theorem stating that the final x value is 201 -/
theorem final_x_value :
  (final_state.x = 201) ∧ (final_state.S ≥ 10000) ∧ 
  (update_state final_state).S > 10000 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l388_38857


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_1502_l388_38801

/-- The sum of the tens digit and the units digit in the decimal representation of 8^1502 is 10 -/
theorem sum_of_digits_8_pow_1502 : ∃ n : ℕ, 8^1502 = 100 * n + 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_1502_l388_38801


namespace NUMINAMATH_CALUDE_expansion_of_5000_power_150_l388_38895

theorem expansion_of_5000_power_150 :
  ∃ (n : ℕ), 5000^150 = n * 10^450 ∧ 1 ≤ n ∧ n < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_expansion_of_5000_power_150_l388_38895


namespace NUMINAMATH_CALUDE_april_roses_problem_l388_38824

theorem april_roses_problem (price : ℕ) (leftover : ℕ) (earnings : ℕ) (initial : ℕ) : 
  price = 7 → 
  leftover = 4 → 
  earnings = 35 → 
  price * (initial - leftover) = earnings → 
  initial = 9 := by
sorry

end NUMINAMATH_CALUDE_april_roses_problem_l388_38824


namespace NUMINAMATH_CALUDE_hunter_frog_count_l388_38806

/-- The number of frogs Hunter saw in the pond -/
def total_frogs (lily_pad_frogs log_frogs baby_frogs : ℕ) : ℕ :=
  lily_pad_frogs + log_frogs + baby_frogs

/-- Two dozen -/
def two_dozen : ℕ := 2 * 12

theorem hunter_frog_count :
  total_frogs 5 3 two_dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_hunter_frog_count_l388_38806


namespace NUMINAMATH_CALUDE_ellipse_equation_l388_38888

/-- An ellipse with foci and points satisfying certain conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : A.1^2 / a^2 + A.2^2 / b^2 = 1  -- A is on the ellipse
  h₄ : B.1^2 / a^2 + B.2^2 / b^2 = 1  -- B is on the ellipse
  h₅ : (A.1 - B.1) * (F₁.1 - F₂.1) + (A.2 - B.2) * (F₁.2 - F₂.2) = 0  -- AB ⟂ F₁F₂
  h₆ : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16  -- |AB| = 4
  h₇ : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 12  -- |F₁F₂| = 2√3

/-- The equation of the ellipse is x²/9 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l388_38888


namespace NUMINAMATH_CALUDE_rationalize_denominator_l388_38851

theorem rationalize_denominator :
  36 / Real.sqrt 7 = (36 * Real.sqrt 7) / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l388_38851


namespace NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l388_38849

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_fill : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank completely -/
def time_to_complete (tank : WaterTank) : ℚ :=
  let combined_rate := tank.fill_rate - tank.empty_rate
  let amount_to_change := 1 - tank.initial_fill
  amount_to_change / (-combined_rate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes :
  let tank := WaterTank.mk (1/5) (1/15) (1/6)
  time_to_complete tank = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l388_38849


namespace NUMINAMATH_CALUDE_abs_x_eq_3_system_solution_2_system_solution_3_l388_38879

-- Part 1
theorem abs_x_eq_3 (x : ℝ) : |x| = 3 ↔ x = 3 ∨ x = -3 := by sorry

-- Part 2
theorem system_solution_2 (x y : ℝ) : 
  y * (x - 1) = 0 ∧ 2 * x + 5 * y = 7 ↔ 
  (x = 7/2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := by sorry

-- Part 3
theorem system_solution_3 (x y : ℝ) :
  x * y - 2 * x - y + 2 = 0 ∧ x + 6 * y = 3 ∧ 3 * x + y = 8 ↔
  (x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_abs_x_eq_3_system_solution_2_system_solution_3_l388_38879


namespace NUMINAMATH_CALUDE_cabbage_count_this_year_l388_38862

/-- Represents the number of cabbages in a square garden --/
def CabbageCount (side : ℕ) : ℕ := side * side

/-- Theorem stating the number of cabbages this year given the conditions --/
theorem cabbage_count_this_year :
  ∀ (last_year_side : ℕ),
  (CabbageCount (last_year_side + 1) - CabbageCount last_year_side = 197) →
  (CabbageCount (last_year_side + 1) = 9801) :=
by
  sorry

end NUMINAMATH_CALUDE_cabbage_count_this_year_l388_38862


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l388_38873

theorem sum_of_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  ∃ (c : ℝ), c = 7 ∧ 
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ c) ∧
  (∀ (c' : ℝ), c' < c → 
    ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 
      Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) > c') :=
by sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l388_38873


namespace NUMINAMATH_CALUDE_max_value_expression_l388_38880

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-9.5) 9.5)
  (hb : b ∈ Set.Icc (-9.5) 9.5)
  (hc : c ∈ Set.Icc (-9.5) 9.5)
  (hd : d ∈ Set.Icc (-9.5) 9.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 380 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l388_38880


namespace NUMINAMATH_CALUDE_functional_equation_solution_l388_38828

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + x*y) = f x * f y + y * f x + x * f (x + y)) :
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l388_38828


namespace NUMINAMATH_CALUDE_correct_monthly_repayment_l388_38812

/-- Calculates the monthly repayment amount for a loan -/
def calculate_monthly_repayment (loan_amount : ℝ) (monthly_interest_rate : ℝ) (loan_term_months : ℕ) : ℝ :=
  sorry

/-- Theorem stating the correct monthly repayment amount -/
theorem correct_monthly_repayment :
  let loan_amount : ℝ := 500000
  let monthly_interest_rate : ℝ := 0.005
  let loan_term_months : ℕ := 360
  abs (calculate_monthly_repayment loan_amount monthly_interest_rate loan_term_months - 2997.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_correct_monthly_repayment_l388_38812


namespace NUMINAMATH_CALUDE_necessary_condition_equality_l388_38829

theorem necessary_condition_equality (a b c : ℝ) (h : c ≠ 0) :
  a = b → a * c = b * c :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_equality_l388_38829


namespace NUMINAMATH_CALUDE_probability_equals_frequency_l388_38861

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of yellow balls in the bag -/
def yellow_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + yellow_balls

/-- Represents the observed limiting frequency from the experiment -/
def observed_frequency : ℚ := 2/5

/-- Theorem stating that the probability of selecting a red ball equals the observed frequency -/
theorem probability_equals_frequency : 
  (red_balls : ℚ) / (total_balls : ℚ) = observed_frequency :=
sorry

end NUMINAMATH_CALUDE_probability_equals_frequency_l388_38861


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l388_38859

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / b → x * y = c → 
  x < y ∧ x = Real.sqrt (a * c / b) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l388_38859


namespace NUMINAMATH_CALUDE_solution_set_inequality_l388_38898

theorem solution_set_inequality (x : ℝ) :
  (x - 5) * (x + 1) > 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l388_38898


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l388_38850

def M : Set ℕ := {x : ℕ | 0 < x ∧ x < 4}
def N : Set ℕ := {x : ℕ | 1 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l388_38850


namespace NUMINAMATH_CALUDE_calculate_dime_piles_l388_38855

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in cents -/
def total_value : ℕ := 2100

/-- Calculates the number of piles of dimes given the conditions -/
theorem calculate_dime_piles : 
  ∃ (dime_piles : ℕ),
    dime_piles * coins_per_pile * dime_value + 
    quarter_piles * coins_per_pile * quarter_value +
    nickel_piles * coins_per_pile * nickel_value +
    penny_piles * coins_per_pile * penny_value = total_value ∧
    dime_piles = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_dime_piles_l388_38855


namespace NUMINAMATH_CALUDE_negation_of_inequality_l388_38817

theorem negation_of_inequality (x : Real) : 
  (¬ ∀ x ∈ Set.Ioo 0 (π/2), x > Real.sin x) ↔ 
  (∃ x ∈ Set.Ioo 0 (π/2), x ≤ Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_negation_of_inequality_l388_38817


namespace NUMINAMATH_CALUDE_perfect_square_problem_l388_38825

theorem perfect_square_problem :
  (∃ (x : ℕ), 7^2040 = x^2) ∧
  (∀ (x : ℕ), 8^2041 ≠ x^2) ∧
  (∃ (x : ℕ), 9^2042 = x^2) ∧
  (∃ (x : ℕ), 10^2043 = x^2) ∧
  (∃ (x : ℕ), 11^2044 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l388_38825


namespace NUMINAMATH_CALUDE_union_M_N_equals_geq_one_l388_38896

-- Define set M
def M : Set ℝ := {x | x - 2 > 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- Theorem statement
theorem union_M_N_equals_geq_one : M ∪ N = {x | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_geq_one_l388_38896


namespace NUMINAMATH_CALUDE_function_max_at_pi_third_l388_38872

/-- The function f(x) reaches its maximum value at x₀ = π/3 --/
theorem function_max_at_pi_third (x : ℝ) : 
  let f := λ x : ℝ => (Real.sqrt 3 / 2) * Real.sin x + (1 / 2) * Real.cos x
  ∃ (x₀ : ℝ), x₀ = π/3 ∧ ∀ y, f y ≤ f x₀ :=
by sorry

end NUMINAMATH_CALUDE_function_max_at_pi_third_l388_38872


namespace NUMINAMATH_CALUDE_sand_received_by_city_c_l388_38865

/-- The amount of sand received by City C given the total sand and amounts received by other cities -/
theorem sand_received_by_city_c 
  (total : ℝ) 
  (city_a : ℝ) 
  (city_b : ℝ) 
  (city_d : ℝ) 
  (h_total : total = 95) 
  (h_city_a : city_a = 16.5) 
  (h_city_b : city_b = 26) 
  (h_city_d : city_d = 28) : 
  total - (city_a + city_b + city_d) = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sand_received_by_city_c_l388_38865


namespace NUMINAMATH_CALUDE_shooting_match_sequences_l388_38822

/-- Represents the number of targets in each column --/
structure TargetArrangement where
  columnA : Nat
  columnB : Nat
  columnC : Nat

/-- Calculates the number of valid sequences for breaking targets --/
def validSequences (arrangement : TargetArrangement) : Nat :=
  (Nat.factorial 4 / Nat.factorial 1 / Nat.factorial 3) *
  (Nat.factorial 6 / Nat.factorial 3 / Nat.factorial 3)

/-- Theorem statement for the shooting match problem --/
theorem shooting_match_sequences (arrangement : TargetArrangement)
  (h1 : arrangement.columnA = 4)
  (h2 : arrangement.columnB = 3)
  (h3 : arrangement.columnC = 3) :
  validSequences arrangement = 80 := by
  sorry

end NUMINAMATH_CALUDE_shooting_match_sequences_l388_38822


namespace NUMINAMATH_CALUDE_spacefarer_resources_sum_l388_38871

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem spacefarer_resources_sum :
  let crystal := base3_to_base10 [0, 2, 1, 2]
  let rare_metals := base3_to_base10 [2, 0, 1, 2]
  let alien_tech := base3_to_base10 [2, 0, 1]
  crystal + rare_metals + alien_tech = 145 := by
sorry

end NUMINAMATH_CALUDE_spacefarer_resources_sum_l388_38871


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l388_38863

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we're working with (1.12 million) -/
def number : ℝ := 1.12e6

/-- The claimed scientific notation of the number -/
def claimed_notation : ScientificNotation := {
  coefficient := 1.12,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the claimed notation is correct for the given number -/
theorem scientific_notation_correct : 
  number = claimed_notation.coefficient * (10 : ℝ) ^ claimed_notation.exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l388_38863


namespace NUMINAMATH_CALUDE_equation_solution_l388_38852

theorem equation_solution (α β : ℝ) : 
  (∀ x : ℝ, x ≠ -β → x ≠ 30 → x ≠ 70 → 
    (x - α) / (x + β) = (x^2 + 120*x + 1575) / (x^2 - 144*x + 1050)) →
  α + β = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l388_38852


namespace NUMINAMATH_CALUDE_work_completion_time_l388_38889

/-- The time taken to complete a work given two workers with different rates and a partial work completion scenario -/
theorem work_completion_time
  (amit_rate : ℚ)
  (ananthu_rate : ℚ)
  (amit_days : ℕ)
  (h_amit_rate : amit_rate = 1 / 15)
  (h_ananthu_rate : ananthu_rate = 1 / 45)
  (h_amit_days : amit_days = 3)
  : ∃ (total_days : ℕ), total_days = amit_days + ⌈(1 - amit_rate * amit_days) / ananthu_rate⌉ ∧ total_days = 39 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l388_38889


namespace NUMINAMATH_CALUDE_a2016_equals_2025_l388_38836

/-- An arithmetic sequence with common difference 2 and a2007 = 2007 -/
def arithmetic_seq (n : ℕ) : ℕ :=
  2007 + 2 * (n - 2007)

/-- Theorem stating that the 2016th term of the sequence is 2025 -/
theorem a2016_equals_2025 : arithmetic_seq 2016 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_a2016_equals_2025_l388_38836


namespace NUMINAMATH_CALUDE_kristin_bell_peppers_count_l388_38847

/-- The number of bell peppers Kristin has -/
def kristin_bell_peppers : ℕ := 2

/-- The number of carrots Jaylen has -/
def jaylen_carrots : ℕ := 5

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers : ℕ := 2

/-- The number of green beans Kristin has -/
def kristin_green_beans : ℕ := 20

/-- The total number of vegetables Jaylen has -/
def jaylen_total_vegetables : ℕ := 18

theorem kristin_bell_peppers_count :
  (jaylen_carrots + jaylen_cucumbers + 
   (kristin_green_beans / 2 - 3) + 
   (2 * kristin_bell_peppers) = jaylen_total_vegetables) →
  kristin_bell_peppers = 2 := by
  sorry

end NUMINAMATH_CALUDE_kristin_bell_peppers_count_l388_38847


namespace NUMINAMATH_CALUDE_bicycle_journey_l388_38882

theorem bicycle_journey (t₅ t₁₅ : ℝ) (h_positive : t₅ > 0 ∧ t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 → t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_journey_l388_38882


namespace NUMINAMATH_CALUDE_blue_red_face_ratio_l388_38820

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_red_face_ratio_l388_38820


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l388_38840

/-- The number of minutes in a day -/
def minutes_per_day : ℚ := 24 * 60

/-- The number of minutes the watch loses per day -/
def minutes_lost_per_day : ℚ := 5/2

/-- The number of days between March 15 at 1 PM and March 21 at 9 AM -/
def days_elapsed : ℚ := 5 + 5/6

/-- The correct additional minutes to set the watch -/
def n : ℚ := 14 + 14/23

theorem watch_correction_theorem :
  n = (minutes_per_day / (minutes_per_day - minutes_lost_per_day) - 1) * (days_elapsed * minutes_per_day) :=
by sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l388_38840


namespace NUMINAMATH_CALUDE_barry_vitamin_d3_days_l388_38899

/-- Calculates the number of days Barry was told to take vitamin D3 -/
def vitaminD3Days (capsules_per_bottle : ℕ) (capsules_per_day : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / capsules_per_day) * bottles_needed

theorem barry_vitamin_d3_days :
  vitaminD3Days 60 2 6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_barry_vitamin_d3_days_l388_38899


namespace NUMINAMATH_CALUDE_clock_angle_at_3_15_l388_38870

/-- The angle in degrees that the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle in degrees that the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The starting position of the hour hand at 3:00 in degrees -/
def hour_hand_start : ℝ := 90

/-- The number of minutes past 3:00 -/
def minutes_past : ℝ := 15

/-- Calculates the position of the hour hand at 3:15 -/
def hour_hand_position : ℝ := hour_hand_start + hour_hand_speed * minutes_past

/-- Calculates the position of the minute hand at 3:15 -/
def minute_hand_position : ℝ := minute_hand_speed * minutes_past

/-- The acute angle between the hour hand and minute hand at 3:15 -/
def clock_angle : ℝ := |hour_hand_position - minute_hand_position|

theorem clock_angle_at_3_15 : clock_angle = 7.5 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_15_l388_38870


namespace NUMINAMATH_CALUDE_chord_distance_half_arc_l388_38831

/-- Given a circle with radius R and a chord at distance d from the center,
    the distance of the chord corresponding to an arc half as long is √(R(R+d)/2). -/
theorem chord_distance_half_arc (R d : ℝ) (h₁ : R > 0) (h₂ : 0 ≤ d) (h₃ : d < R) :
  let distance_half_arc := Real.sqrt (R * (R + d) / 2)
  distance_half_arc > 0 ∧ distance_half_arc < R :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_half_arc_l388_38831


namespace NUMINAMATH_CALUDE_solve_equation_l388_38874

theorem solve_equation (a : ℚ) : a + a / 3 = 8 / 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l388_38874


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l388_38890

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | (3 - x) / (x + 1) > 0}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.compl N) = {x : ℝ | x < -2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l388_38890


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l388_38808

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem for proposition ①
theorem prop_1 (b : ℝ) : 
  ∀ x, f x b 0 = -f (-x) b 0 := by sorry

-- Theorem for proposition ②
theorem prop_2 (c : ℝ) (h : c > 0) :
  ∃! x, f x 0 c = 0 := by sorry

-- Theorem for proposition ③
theorem prop_3 (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c + 2 * c := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l388_38808


namespace NUMINAMATH_CALUDE_hall_volume_l388_38877

theorem hall_volume (length width : ℝ) (h : ℝ) : 
  length = 6 ∧ width = 6 ∧ 2 * (length * width) = 2 * (length * h) + 2 * (width * h) → 
  length * width * h = 108 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_l388_38877


namespace NUMINAMATH_CALUDE_shark_stingray_ratio_l388_38819

theorem shark_stingray_ratio :
  ∀ (total_fish sharks stingrays : ℕ),
    total_fish = 84 →
    stingrays = 28 →
    sharks + stingrays = total_fish →
    sharks / stingrays = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_shark_stingray_ratio_l388_38819


namespace NUMINAMATH_CALUDE_sum_fraction_equality_l388_38858

theorem sum_fraction_equality (x y z : ℝ) (h : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_equality_l388_38858


namespace NUMINAMATH_CALUDE_divisors_of_180_l388_38845

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

def count_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_180 :
  (largest_prime_factor (sum_of_divisors 180) = 13) ∧
  (count_divisors 180 = 18) := by sorry

end NUMINAMATH_CALUDE_divisors_of_180_l388_38845


namespace NUMINAMATH_CALUDE_remainder_27_power_27_plus_27_mod_28_l388_38826

theorem remainder_27_power_27_plus_27_mod_28 :
  (27^27 + 27) % 28 = 26 := by
  sorry

end NUMINAMATH_CALUDE_remainder_27_power_27_plus_27_mod_28_l388_38826


namespace NUMINAMATH_CALUDE_second_year_associates_percentage_l388_38853

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  not_first_year : ℝ
  more_than_two_years : ℝ

/-- The theorem stating that the percentage of second-year associates is 30% -/
theorem second_year_associates_percentage
  (ap : AssociatePercentages)
  (h1 : ap.not_first_year = 60)
  (h2 : ap.more_than_two_years = 30) :
  ap.not_first_year - ap.more_than_two_years = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_year_associates_percentage_l388_38853


namespace NUMINAMATH_CALUDE_cosine_expression_simplification_fraction_simplification_l388_38843

-- Part 1
theorem cosine_expression_simplification :
  2 * Real.cos (45 * π / 180) - (-2 * Real.sqrt 3) ^ 0 + 1 / (Real.sqrt 2 + 1) - Real.sqrt 8 = -2 := by
  sorry

-- Part 2
theorem fraction_simplification (x : ℝ) (h : x = -Real.sqrt 2) :
  (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_expression_simplification_fraction_simplification_l388_38843


namespace NUMINAMATH_CALUDE_divisibility_by_27_l388_38816

theorem divisibility_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_27_l388_38816


namespace NUMINAMATH_CALUDE_series_sum_equals_81_and_two_fifths_l388_38834

def series_sum : ℚ :=
  1 + 3 * (1/6) + 5 * (1/12) + 7 * (1/20) + 9 * (1/30) + 11 * (1/42) + 
  13 * (1/56) + 15 * (1/72) + 17 * (1/90)

theorem series_sum_equals_81_and_two_fifths : 
  series_sum = 81 + 2/5 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_81_and_two_fifths_l388_38834


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l388_38830

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = (1 - Complex.I)^2 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l388_38830


namespace NUMINAMATH_CALUDE_bob_hair_growth_time_l388_38897

/-- Represents the growth of Bob's hair over time -/
def hair_growth (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * time

/-- Theorem stating the time it takes for Bob's hair to grow from 6 inches to 36 inches -/
theorem bob_hair_growth_time :
  let initial_length : ℝ := 6
  let final_length : ℝ := 36
  let monthly_growth_rate : ℝ := 0.5
  let years : ℝ := 5
  hair_growth initial_length (monthly_growth_rate * 12) years = final_length := by
  sorry

#check bob_hair_growth_time

end NUMINAMATH_CALUDE_bob_hair_growth_time_l388_38897
