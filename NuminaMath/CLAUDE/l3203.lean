import Mathlib

namespace NUMINAMATH_CALUDE_special_triangle_angle_exists_l3203_320356

/-- A triangle with a circumcircle where one altitude is tangent to the circumcircle -/
structure SpecialTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The circumcircle of the triangle -/
  circumcircle : Set (ℝ × ℝ)
  /-- An altitude of the triangle -/
  altitude : Set (ℝ × ℝ)
  /-- The altitude is tangent to the circumcircle -/
  is_tangent : altitude ∩ circumcircle ≠ ∅

/-- The angles of a triangle -/
def angles (t : SpecialTriangle) : Set ℝ := sorry

/-- Theorem: In a SpecialTriangle, there exists an angle greater than 90° and less than 135° -/
theorem special_triangle_angle_exists (t : SpecialTriangle) :
  ∃ θ ∈ angles t, 90 < θ ∧ θ < 135 := by sorry

end NUMINAMATH_CALUDE_special_triangle_angle_exists_l3203_320356


namespace NUMINAMATH_CALUDE_child_cost_age_18_l3203_320375

/-- Represents the cost of raising a child --/
structure ChildCost where
  initialYearlyCost : ℕ
  initialYears : ℕ
  laterYearlyCost : ℕ
  tuitionCost : ℕ
  totalCost : ℕ

/-- Calculates the age at which the child stops incurring yearly cost --/
def ageStopCost (c : ChildCost) : ℕ :=
  let initialCost := c.initialYears * c.initialYearlyCost
  let laterYears := (c.totalCost - initialCost - c.tuitionCost) / c.laterYearlyCost
  c.initialYears + laterYears

/-- Theorem stating that given the specific costs, the child stops incurring yearly cost at age 18 --/
theorem child_cost_age_18 :
  let c := ChildCost.mk 5000 8 10000 125000 265000
  ageStopCost c = 18 := by
  sorry

#eval ageStopCost (ChildCost.mk 5000 8 10000 125000 265000)

end NUMINAMATH_CALUDE_child_cost_age_18_l3203_320375


namespace NUMINAMATH_CALUDE_sum_always_negative_l3203_320350

def f (x : ℝ) : ℝ := -x - x^3

theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_negative_l3203_320350


namespace NUMINAMATH_CALUDE_complex_symmetry_quotient_l3203_320337

theorem complex_symmetry_quotient (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → z₁ = 1 + I → z₁ / z₂ = I := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_quotient_l3203_320337


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3203_320335

-- Define the conic section equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-4)^2 + (y-3)^2) = 8

-- Define the focal points
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (4, 3)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focal_point1.1 + focal_point2.1) / 2)^2 / a^2 +
    (y - (focal_point1.2 + focal_point2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3203_320335


namespace NUMINAMATH_CALUDE_jills_salary_l3203_320321

theorem jills_salary (net_salary : ℝ) 
  (h1 : net_salary > 0)
  (discretionary_income : ℝ)
  (h2 : discretionary_income = net_salary / 5)
  (vacation_fund : ℝ)
  (h3 : vacation_fund = 0.3 * discretionary_income)
  (savings : ℝ)
  (h4 : savings = 0.2 * discretionary_income)
  (socializing : ℝ)
  (h5 : socializing = 0.35 * discretionary_income)
  (remaining : ℝ)
  (h6 : remaining = discretionary_income - vacation_fund - savings - socializing)
  (h7 : remaining = 105) :
  net_salary = 3500 := by
sorry

end NUMINAMATH_CALUDE_jills_salary_l3203_320321


namespace NUMINAMATH_CALUDE_recipe_total_cups_l3203_320373

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given ratio and flour amount, the total cups is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  let flourCups : ℕ := 15
  totalCups ratio flourCups = 30 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l3203_320373


namespace NUMINAMATH_CALUDE_existence_of_point_N_l3203_320351

theorem existence_of_point_N (a m : ℝ) (ha : a > 0) (hm : m ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioi 0)) :
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = a^2 ∧ |y₀| = (|m| * a) / Real.sqrt (1 + m) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_N_l3203_320351


namespace NUMINAMATH_CALUDE_assignment_theorem_l3203_320353

/-- The number of ways to assign 4 distinct objects to 3 distinct groups, 
    with at least one object in each group -/
def assignment_ways : ℕ := 36

/-- The number of ways to choose 2 objects from 4 distinct objects -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 distinct objects -/
def arrange_three : ℕ := Nat.factorial 3

theorem assignment_theorem : 
  assignment_ways = choose_two_from_four * arrange_three := by
  sorry

end NUMINAMATH_CALUDE_assignment_theorem_l3203_320353


namespace NUMINAMATH_CALUDE_emmy_and_rosa_ipods_l3203_320343

/-- 
Given that Emmy originally had 14 iPods, lost 6, and has twice as many as Rosa,
prove that Emmy and Rosa have 12 iPods together.
-/
theorem emmy_and_rosa_ipods :
  ∀ (emmy_original emmy_lost emmy_current rosa : ℕ),
  emmy_original = 14 →
  emmy_lost = 6 →
  emmy_current = emmy_original - emmy_lost →
  emmy_current = 2 * rosa →
  emmy_current + rosa = 12 := by
  sorry

end NUMINAMATH_CALUDE_emmy_and_rosa_ipods_l3203_320343


namespace NUMINAMATH_CALUDE_ordering_of_distinct_positives_l3203_320322

theorem ordering_of_distinct_positives (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (heq : a^2 + c^2 = 2*b*c) :
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_ordering_of_distinct_positives_l3203_320322


namespace NUMINAMATH_CALUDE_trig_system_relation_l3203_320305

/-- Given a system of trigonometric equations, prove the relationship between a, b, and c -/
theorem trig_system_relation (x y a b c : ℝ) 
  (h1 : Real.sin x + Real.sin y = 2 * a)
  (h2 : Real.cos x + Real.cos y = 2 * b)
  (h3 : Real.tan x + Real.tan y = 2 * c) :
  a * (b + a * c) = c * (a^2 + b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_system_relation_l3203_320305


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l3203_320382

/-- The number of different positive six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, 8 -/
def sixDigitIntegersCount : ℕ := 60

/-- The multiset of digits used to form the integers -/
def digits : Multiset ℕ := {1, 1, 3, 3, 3, 8}

theorem count_six_digit_integers : 
  (Multiset.card digits = 6) → 
  (Multiset.count 1 digits = 2) → 
  (Multiset.count 3 digits = 3) → 
  (Multiset.count 8 digits = 1) → 
  sixDigitIntegersCount = 60 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l3203_320382


namespace NUMINAMATH_CALUDE_range_of_a_for_monotone_decreasing_f_l3203_320376

/-- A piecewise function f(x) defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x^2 - 6*x + a^2 + 1
  else x^(5 - 2*a)

/-- The theorem stating the range of a for which f is monotonically decreasing -/
theorem range_of_a_for_monotone_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Ioo (5/2) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotone_decreasing_f_l3203_320376


namespace NUMINAMATH_CALUDE_corn_spacing_theorem_l3203_320306

/-- Calculates the space required for each seed in a row of corn. -/
def space_per_seed (row_length_feet : ℕ) (seeds_per_row : ℕ) : ℕ :=
  (row_length_feet * 12) / seeds_per_row

/-- Theorem: Given a row length of 120 feet and 80 seeds per row, 
    the space required for each seed is 18 inches. -/
theorem corn_spacing_theorem : space_per_seed 120 80 = 18 := by
  sorry

end NUMINAMATH_CALUDE_corn_spacing_theorem_l3203_320306


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l3203_320374

theorem ceiling_neg_sqrt_64_over_9 : 
  ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l3203_320374


namespace NUMINAMATH_CALUDE_paint_usage_l3203_320320

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) 
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1/3)
  (h3 : second_week_fraction = 1/5) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 168 := by
sorry


end NUMINAMATH_CALUDE_paint_usage_l3203_320320


namespace NUMINAMATH_CALUDE_select_three_from_fifteen_l3203_320395

theorem select_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_fifteen_l3203_320395


namespace NUMINAMATH_CALUDE_population_ratio_theorem_l3203_320348

/-- Represents the population ratio in a town --/
structure PopulationRatio where
  men : ℝ
  women : ℝ
  children : ℝ
  elderly : ℝ

/-- The population ratio satisfies the given conditions --/
def satisfiesConditions (p : PopulationRatio) : Prop :=
  p.women = 0.9 * p.men ∧
  p.children = 0.6 * (p.men + p.women) ∧
  p.elderly = 0.25 * (p.women + p.children)

/-- The theorem stating the ratio of men to the combined population of others --/
theorem population_ratio_theorem (p : PopulationRatio) 
  (h : satisfiesConditions p) : 
  p.men / (p.women + p.children + p.elderly) = 1 / 2.55 := by
  sorry

#check population_ratio_theorem

end NUMINAMATH_CALUDE_population_ratio_theorem_l3203_320348


namespace NUMINAMATH_CALUDE_ribbon_length_l3203_320392

/-- The original length of two ribbons with specific cutting conditions -/
theorem ribbon_length : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (x - 12 = 2 * (x - 18)) ∧ 
  (x = 24) := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_l3203_320392


namespace NUMINAMATH_CALUDE_gerald_toy_cars_l3203_320342

theorem gerald_toy_cars (initial_cars : ℕ) (donation_fraction : ℚ) (remaining_cars : ℕ) :
  initial_cars = 20 →
  donation_fraction = 1 / 4 →
  remaining_cars = initial_cars - (initial_cars * donation_fraction).floor →
  remaining_cars = 15 := by
  sorry

end NUMINAMATH_CALUDE_gerald_toy_cars_l3203_320342


namespace NUMINAMATH_CALUDE_four_thirds_is_36_l3203_320347

theorem four_thirds_is_36 (x : ℚ) : (4 : ℚ) / 3 * x = 36 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_is_36_l3203_320347


namespace NUMINAMATH_CALUDE_building_dimension_difference_l3203_320389

-- Define the building structure
structure Building where
  floor1_length : ℝ
  floor1_width : ℝ
  floor2_length : ℝ
  floor2_width : ℝ

-- Define the conditions
def building_conditions (b : Building) : Prop :=
  b.floor1_width = (1/2) * b.floor1_length ∧
  b.floor1_length * b.floor1_width = 578 ∧
  b.floor2_width = (1/3) * b.floor2_length ∧
  b.floor2_length * b.floor2_width = 450

-- Define the combined length and width
def combined_length (b : Building) : ℝ := b.floor1_length + b.floor2_length
def combined_width (b : Building) : ℝ := b.floor1_width + b.floor2_width

-- Theorem statement
theorem building_dimension_difference (b : Building) 
  (h : building_conditions b) : 
  ∃ ε > 0, |combined_length b - combined_width b - 41.494| < ε :=
sorry

end NUMINAMATH_CALUDE_building_dimension_difference_l3203_320389


namespace NUMINAMATH_CALUDE_child_share_proof_l3203_320360

theorem child_share_proof (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) 
  (h1 : total_amount = 2700)
  (h2 : ratio_a = 2)
  (h3 : ratio_b = 3)
  (h4 : ratio_c = 4) : 
  (total_amount * ratio_b) / (ratio_a + ratio_b + ratio_c) = 900 := by
  sorry

end NUMINAMATH_CALUDE_child_share_proof_l3203_320360


namespace NUMINAMATH_CALUDE_sum_of_digits_of_M_l3203_320383

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition M^2 = 36^49 * 49^36
axiom M_squared : (M : ℕ)^2 = 36^49 * 49^36

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_M : sum_of_digits (M : ℕ) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_M_l3203_320383


namespace NUMINAMATH_CALUDE_sara_peaches_l3203_320377

theorem sara_peaches (initial_peaches additional_peaches : ℝ) 
  (h1 : initial_peaches = 61.0) 
  (h2 : additional_peaches = 24.0) : 
  initial_peaches + additional_peaches = 85.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_l3203_320377


namespace NUMINAMATH_CALUDE_eleven_times_digit_sum_l3203_320327

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem eleven_times_digit_sum :
  ∀ n : ℕ, n = 11 * sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end NUMINAMATH_CALUDE_eleven_times_digit_sum_l3203_320327


namespace NUMINAMATH_CALUDE_overall_loss_is_184_76_l3203_320398

-- Define the structure for an item
structure Item where
  name : String
  price : ℝ
  currency : String
  tax_rate : ℝ
  discount_rate : ℝ
  profit_loss_rate : ℝ

-- Define the currency conversion rates
def conversion_rates : List (String × ℝ) :=
  [("USD", 75), ("EUR", 80), ("GBP", 100), ("JPY", 0.7)]

-- Define the items
def items : List Item :=
  [{ name := "grinder", price := 150, currency := "USD", tax_rate := 0.1, discount_rate := 0, profit_loss_rate := -0.04 },
   { name := "mobile_phone", price := 100, currency := "EUR", tax_rate := 0.15, discount_rate := 0.05, profit_loss_rate := 0.1 },
   { name := "laptop", price := 200, currency := "GBP", tax_rate := 0.08, discount_rate := 0, profit_loss_rate := -0.08 },
   { name := "camera", price := 12000, currency := "JPY", tax_rate := 0.05, discount_rate := 0.12, profit_loss_rate := 0.15 }]

-- Function to calculate the final price of an item in INR
def calculate_final_price (item : Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Function to calculate the overall profit or loss
def calculate_overall_profit_loss (items : List Item) (conversion_rates : List (String × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem overall_loss_is_184_76 :
  calculate_overall_profit_loss items conversion_rates = -184.76 := by sorry

end NUMINAMATH_CALUDE_overall_loss_is_184_76_l3203_320398


namespace NUMINAMATH_CALUDE_rabbits_ate_four_potatoes_l3203_320352

/-- The number of potatoes eaten by rabbits -/
def potatoes_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of potatoes eaten by rabbits is 4 -/
theorem rabbits_ate_four_potatoes (h1 : initial = 7) (h2 : remaining = 3) :
  potatoes_eaten initial remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_four_potatoes_l3203_320352


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l3203_320345

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 44

/-- The number of teeth a horse has -/
def horse_teeth : ℕ := 40

/-- The number of teeth a rabbit has -/
def rabbit_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 7

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 12

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 9

/-- The number of horses Vann will clean -/
def num_horses : ℕ := 4

/-- The number of rabbits Vann will clean -/
def num_rabbits : ℕ := 15

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 
  num_dogs * dog_teeth + 
  num_cats * cat_teeth + 
  num_pigs * pig_teeth + 
  num_horses * horse_teeth + 
  num_rabbits * rabbit_teeth

theorem vann_teeth_cleaning : total_teeth = 1630 := by
  sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l3203_320345


namespace NUMINAMATH_CALUDE_game_result_l3203_320355

-- Define the point function
def g (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

-- Define Allie's rolls
def allie_rolls : List Nat := [3, 6, 5, 2, 4]

-- Define Betty's rolls
def betty_rolls : List Nat := [2, 6, 1, 4]

-- Calculate total points for a list of rolls
def total_points (rolls : List Nat) : Nat :=
  rolls.map g |>.sum

-- Theorem statement
theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 308 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l3203_320355


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3203_320340

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : seq.a 2 / seq.a 4 = 7 / 6) :
  S seq 7 / S seq 3 = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3203_320340


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3203_320384

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cylinder : ℝ) 
  (h_cone : ℝ) 
  (h_cylinder_eq : h_cylinder = 30) 
  (h_cone_eq : h_cone = 15) 
  (r_eq : r = 10) : 
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3203_320384


namespace NUMINAMATH_CALUDE_odd_product_probability_zero_l3203_320319

-- Define the type for our grid
def Grid := Fin 3 → Fin 3 → Fin 9

-- Define a function to check if a number is odd
def isOdd (n : Fin 9) : Prop := n.val % 2 = 1

-- Define a function to check if a row has all odd numbers
def rowAllOdd (g : Grid) (row : Fin 3) : Prop :=
  ∀ col : Fin 3, isOdd (g row col)

-- Define our main theorem
theorem odd_product_probability_zero :
  ∀ g : Grid, (∀ row : Fin 3, rowAllOdd g row) → False :=
sorry

end NUMINAMATH_CALUDE_odd_product_probability_zero_l3203_320319


namespace NUMINAMATH_CALUDE_expression_equals_6500_l3203_320349

theorem expression_equals_6500 : (2015 / 1 + 2015 / 0.31) / (1 + 0.31) = 6500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_6500_l3203_320349


namespace NUMINAMATH_CALUDE_remaining_average_of_prime_numbers_l3203_320330

theorem remaining_average_of_prime_numbers 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℚ) 
  (subset_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 10) 
  (h3 : total_average = 95) 
  (h4 : subset_average = 85) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 105 := by
sorry

end NUMINAMATH_CALUDE_remaining_average_of_prime_numbers_l3203_320330


namespace NUMINAMATH_CALUDE_quadratic_term_coefficient_and_constant_term_l3203_320385

/-- Represents a quadratic equation in the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation -3x² - 2x = 0 -/
def givenEquation : QuadraticEquation :=
  { a := -3, b := -2, c := 0 }

theorem quadratic_term_coefficient_and_constant_term :
  (givenEquation.a = -3) ∧ (givenEquation.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_term_coefficient_and_constant_term_l3203_320385


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3203_320354

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 4) + Real.log (a 7) + Real.log (a 10) = 3 →
  a 1 * a 13 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3203_320354


namespace NUMINAMATH_CALUDE_inequality_transformation_l3203_320371

theorem inequality_transformation (a b : ℝ) : a ≤ b → -a/2 ≥ -b/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l3203_320371


namespace NUMINAMATH_CALUDE_subtracted_number_l3203_320303

theorem subtracted_number (x : ℝ) (y : ℝ) : 
  x = 62.5 → ((x + 5) * 2 / 5 - y = 22) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3203_320303


namespace NUMINAMATH_CALUDE_mrs_sheridan_initial_fish_l3203_320317

/-- The number of fish Mrs. Sheridan received from her sister -/
def fish_from_sister : ℕ := 47

/-- The total number of fish Mrs. Sheridan has after receiving fish from her sister -/
def total_fish : ℕ := 69

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℕ := total_fish - fish_from_sister

theorem mrs_sheridan_initial_fish :
  initial_fish = 22 :=
sorry

end NUMINAMATH_CALUDE_mrs_sheridan_initial_fish_l3203_320317


namespace NUMINAMATH_CALUDE_lucas_speed_equals_miguel_speed_l3203_320301

/-- Given the relative speeds of Miguel, Sophie, and Lucas, prove that Lucas's speed equals Miguel's speed. -/
theorem lucas_speed_equals_miguel_speed (miguel_speed : ℝ) (sophie_speed : ℝ) (lucas_speed : ℝ)
  (h1 : miguel_speed = 6)
  (h2 : sophie_speed = 3/4 * miguel_speed)
  (h3 : lucas_speed = 4/3 * sophie_speed) :
  lucas_speed = miguel_speed :=
by sorry

end NUMINAMATH_CALUDE_lucas_speed_equals_miguel_speed_l3203_320301


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3203_320331

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0) : 
  x₁^3 + x₂^3 = 18 ∧ x₂/x₁ + x₁/x₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3203_320331


namespace NUMINAMATH_CALUDE_lenny_pens_boxes_l3203_320341

theorem lenny_pens_boxes : ∀ (total_pens : ℕ) (pens_per_box : ℕ),
  pens_per_box = 5 →
  (total_pens : ℚ) * (3 / 5 : ℚ) * (3 / 4 : ℚ) = 45 →
  total_pens / pens_per_box = 20 :=
by
  sorry

#check lenny_pens_boxes

end NUMINAMATH_CALUDE_lenny_pens_boxes_l3203_320341


namespace NUMINAMATH_CALUDE_gold_weight_is_ten_l3203_320310

def weights : List ℕ := List.range 19

theorem gold_weight_is_ten (iron_weights bronze_weights : List ℕ) 
  (h1 : iron_weights.length = 9)
  (h2 : bronze_weights.length = 9)
  (h3 : iron_weights ⊆ weights)
  (h4 : bronze_weights ⊆ weights)
  (h5 : (iron_weights.sum - bronze_weights.sum) = 90)
  (h6 : iron_weights ∩ bronze_weights = [])
  : weights.sum - iron_weights.sum - bronze_weights.sum = 10 := by
  sorry

#check gold_weight_is_ten

end NUMINAMATH_CALUDE_gold_weight_is_ten_l3203_320310


namespace NUMINAMATH_CALUDE_car_distance_proof_l3203_320397

theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 30 →
  (∃ (initial_speed : ℝ), 
    initial_speed * initial_time = new_speed * (initial_time * (2/3))) →
  (∃ (distance : ℝ), distance = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3203_320397


namespace NUMINAMATH_CALUDE_exist_common_members_l3203_320300

/-- A structure representing a parliament with committees -/
structure Parliament :=
  (members : Finset ℕ)
  (committees : Finset (Finset ℕ))
  (h_member_count : members.card = 1600)
  (h_committee_count : committees.card = 16000)
  (h_committee_size : ∀ c ∈ committees, c.card = 80)
  (h_committees_subset : ∀ c ∈ committees, c ⊆ members)

/-- Theorem stating that there exist at least two committees with at least 4 common members -/
theorem exist_common_members (p : Parliament) :
  ∃ c1 c2 : Finset ℕ, c1 ∈ p.committees ∧ c2 ∈ p.committees ∧ c1 ≠ c2 ∧ (c1 ∩ c2).card ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_exist_common_members_l3203_320300


namespace NUMINAMATH_CALUDE_min_value_inequality_l3203_320379

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((1 / (x + y)) + (1 / (y + z)) + (1 / (z + x))) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3203_320379


namespace NUMINAMATH_CALUDE_white_balls_count_l3203_320302

theorem white_balls_count (red_balls : ℕ) (ratio_red : ℕ) (ratio_white : ℕ) : 
  red_balls = 16 → ratio_red = 4 → ratio_white = 5 → 
  (red_balls * ratio_white) / ratio_red = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3203_320302


namespace NUMINAMATH_CALUDE_book_cost_is_16_l3203_320329

/-- Represents the cost of Léa's purchases -/
def total_cost : ℕ := 28

/-- Represents the number of binders Léa bought -/
def num_binders : ℕ := 3

/-- Represents the cost of each binder -/
def binder_cost : ℕ := 2

/-- Represents the number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- Represents the cost of each notebook -/
def notebook_cost : ℕ := 1

/-- Proves that the cost of the book is $16 -/
theorem book_cost_is_16 : 
  total_cost - (num_binders * binder_cost + num_notebooks * notebook_cost) = 16 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_is_16_l3203_320329


namespace NUMINAMATH_CALUDE_charlotte_theorem_l3203_320313

/-- Represents the state of boxes after each step of adding marbles --/
def BoxState (n : ℕ) := ℕ → ℕ

/-- Initial state of boxes --/
def initial_state (n : ℕ) : BoxState n :=
  λ i => if i ≤ n ∧ i > 0 then i else 0

/-- Add a marble to each box --/
def add_to_all (state : BoxState n) : BoxState n :=
  λ i => state i + 1

/-- Add a marble to boxes divisible by k --/
def add_to_divisible (k : ℕ) (state : BoxState n) : BoxState n :=
  λ i => if state i % k = 0 then state i + 1 else state i

/-- Perform Charlotte's procedure --/
def charlotte_procedure (n : ℕ) : BoxState n :=
  let initial := initial_state n
  let after_first_step := add_to_all initial
  (List.range n).foldl (λ state k => add_to_divisible (k + 2) state) after_first_step

/-- Check if all boxes have exactly n+1 marbles --/
def all_boxes_have_n_plus_one (n : ℕ) (state : BoxState n) : Prop :=
  ∀ i, i > 0 → i ≤ n → state i = n + 1

/-- The main theorem --/
theorem charlotte_theorem (n : ℕ) :
  all_boxes_have_n_plus_one n (charlotte_procedure n) ↔ Nat.Prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_charlotte_theorem_l3203_320313


namespace NUMINAMATH_CALUDE_no_preimage_iff_k_less_than_neg_two_l3203_320334

/-- The function f: ℝ → ℝ defined by f(x) = x² - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that there is no real solution to f(x) = k if and only if k < -2 -/
theorem no_preimage_iff_k_less_than_neg_two :
  ∀ k : ℝ, (¬∃ x : ℝ, f x = k) ↔ k < -2 := by sorry

end NUMINAMATH_CALUDE_no_preimage_iff_k_less_than_neg_two_l3203_320334


namespace NUMINAMATH_CALUDE_equation_result_is_55_l3203_320346

/-- The result of 4 times a number plus 7 times the same number, given the number is 5.0 -/
def equation_result (n : ℝ) : ℝ := 4 * n + 7 * n

/-- Theorem stating that the result of the equation is 55.0 when the number is 5.0 -/
theorem equation_result_is_55 : equation_result 5.0 = 55.0 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_is_55_l3203_320346


namespace NUMINAMATH_CALUDE_no_14_consecutive_integers_exists_21_consecutive_integers_l3203_320396

-- Define the set of primes for part 1
def primes1 : Set ℕ := {2, 3, 5, 7, 11}

-- Define the set of primes for part 2
def primes2 : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to check if a number is divisible by any prime in a given set
def divisibleByAnyPrime (n : ℕ) (primes : Set ℕ) : Prop :=
  ∃ p ∈ primes, n % p = 0

-- Part 1: No set of 14 consecutive integers satisfies the condition
theorem no_14_consecutive_integers : 
  ¬∃ n : ℕ, ∀ k ∈ Finset.range 14, divisibleByAnyPrime (n + k) primes1 := by
sorry

-- Part 2: There exists a set of 21 consecutive integers that satisfies the condition
theorem exists_21_consecutive_integers : 
  ∃ n : ℕ, ∀ k ∈ Finset.range 21, divisibleByAnyPrime (n + k) primes2 := by
sorry

end NUMINAMATH_CALUDE_no_14_consecutive_integers_exists_21_consecutive_integers_l3203_320396


namespace NUMINAMATH_CALUDE_intersection_x_sum_zero_l3203_320381

theorem intersection_x_sum_zero (x₁ x₂ : ℝ) : 
  x₁^2 + 9^2 = 169 → x₂^2 + 9^2 = 169 → x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_sum_zero_l3203_320381


namespace NUMINAMATH_CALUDE_equation_solution_l3203_320339

theorem equation_solution : 
  ∃! x : ℚ, (3 - 2*x) / (x + 2) + (3*x - 6) / (3 - 2*x) = 2 ∧ x = -3/5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3203_320339


namespace NUMINAMATH_CALUDE_zero_has_square_root_l3203_320325

theorem zero_has_square_root : ∃ x : ℝ, x^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_has_square_root_l3203_320325


namespace NUMINAMATH_CALUDE_exactly_one_valid_set_l3203_320391

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A set of consecutive integers is valid if it contains at least two integers and sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive a n = 18

theorem exactly_one_valid_set :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_valid_set_l3203_320391


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_5_sqrt_2_l3203_320370

/-- A rectangular prism with dimensions length, width, and height -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A quadrilateral formed by four points in 3D space -/
structure Quadrilateral3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The area of a quadrilateral formed by the intersection of a plane with a rectangular prism -/
def quadrilateral_area (prism : RectangularPrism) (quad : Quadrilateral3D) : ℝ := sorry

/-- The main theorem stating the area of the quadrilateral ABCD -/
theorem quadrilateral_area_is_5_sqrt_2 (prism : RectangularPrism) (quad : Quadrilateral3D) :
  prism.length = 2 ∧ prism.width = 1 ∧ prism.height = 3 →
  quad.A = ⟨0, 0, 0⟩ ∧ quad.C = ⟨2, 1, 3⟩ →
  quad.B.x = 1 ∧ quad.B.y = 1 ∧ quad.B.z = 0 →
  quad.D.x = 1 ∧ quad.D.y = 0 ∧ quad.D.z = 3 →
  quadrilateral_area prism quad = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_5_sqrt_2_l3203_320370


namespace NUMINAMATH_CALUDE_rice_mixture_price_l3203_320368

/-- Proves that mixing rice at given prices in a specific ratio results in the desired mixture price -/
theorem rice_mixture_price (price1 price2 mixture_price : ℚ) (ratio1 ratio2 : ℕ) : 
  price1 = 31/10 ∧ price2 = 36/10 ∧ mixture_price = 13/4 ∧ ratio1 = 3 ∧ ratio2 = 7 →
  (ratio1 : ℚ) * price1 + (ratio2 : ℚ) * price2 = (ratio1 + ratio2 : ℚ) * mixture_price :=
by
  sorry

#check rice_mixture_price

end NUMINAMATH_CALUDE_rice_mixture_price_l3203_320368


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l3203_320365

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l3203_320365


namespace NUMINAMATH_CALUDE_songs_per_album_l3203_320323

/-- Given that Rachel bought 8 albums and a total of 16 songs, prove that each album has 2 songs. -/
theorem songs_per_album (num_albums : ℕ) (total_songs : ℕ) (h1 : num_albums = 8) (h2 : total_songs = 16) :
  total_songs / num_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_songs_per_album_l3203_320323


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3203_320332

/-- Given vectors a and b with specified properties, prove that |a + 2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) (α : ℝ) :
  (a.fst^2 + a.snd^2 = 4) →
  (b.fst = Real.cos α ∧ b.snd = Real.sin α) →
  (a.fst * b.fst + a.snd * b.snd = Real.cos (π / 3) * 2) →
  ((a.fst + 2 * b.fst)^2 + (a.snd + 2 * b.snd)^2 = 12) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3203_320332


namespace NUMINAMATH_CALUDE_range_of_f_l3203_320315

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3203_320315


namespace NUMINAMATH_CALUDE_roots_sum_quotient_and_reciprocal_l3203_320307

theorem roots_sum_quotient_and_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a ≠ 0) → 
  (b ≠ 0) → 
  a/b + b/a = 18 := by sorry

end NUMINAMATH_CALUDE_roots_sum_quotient_and_reciprocal_l3203_320307


namespace NUMINAMATH_CALUDE_cut_cube_edge_count_l3203_320336

/-- A cube with corners cut off -/
structure CutCube where
  /-- The number of vertices in the original cube -/
  original_vertices : Nat
  /-- The number of edges in the original cube -/
  original_edges : Nat
  /-- The number of new edges created by each vertex cut -/
  new_edges_per_vertex : Nat

/-- The theorem stating that a cube with corners cut off has 56 edges -/
theorem cut_cube_edge_count (c : CutCube) 
  (h1 : c.original_vertices = 8)
  (h2 : c.original_edges = 12)
  (h3 : c.new_edges_per_vertex = 4) :
  c.new_edges_per_vertex * c.original_vertices + 2 * c.original_edges = 56 := by
  sorry

#check cut_cube_edge_count

end NUMINAMATH_CALUDE_cut_cube_edge_count_l3203_320336


namespace NUMINAMATH_CALUDE_daltons_uncle_gift_l3203_320388

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost ball_cost savings needed_more : ℕ) : ℕ :=
  (jump_rope_cost + board_game_cost + ball_cost) - savings - needed_more

/-- Proof that Dalton's uncle gave him $13 -/
theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_daltons_uncle_gift_l3203_320388


namespace NUMINAMATH_CALUDE_painting_time_equation_l3203_320367

/-- The time it takes Sarah to paint the room alone (in hours) -/
def sarah_time : ℝ := 4

/-- The time it takes Tom to paint the room alone (in hours) -/
def tom_time : ℝ := 6

/-- The duration of the break (in hours) -/
def break_time : ℝ := 2

/-- The total time it takes Sarah and Tom to paint the room together, including the break (in hours) -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating the equation that the total time satisfies -/
theorem painting_time_equation :
  (1 / sarah_time + 1 / tom_time) * (total_time - break_time) = 1 := by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l3203_320367


namespace NUMINAMATH_CALUDE_geometric_sequence_product_bound_l3203_320386

theorem geometric_sequence_product_bound (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_seq : (4 * a^2 + b^2)^2 = a * b) : a * b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_bound_l3203_320386


namespace NUMINAMATH_CALUDE_job_application_ratio_l3203_320338

theorem job_application_ratio (total_applications in_state_applications : ℕ) 
  (h1 : total_applications = 600)
  (h2 : in_state_applications = 200) :
  (total_applications - in_state_applications) / in_state_applications = 2 := by
sorry

end NUMINAMATH_CALUDE_job_application_ratio_l3203_320338


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3203_320326

open Set
open Function
open Real

def f (x : ℝ) := 3 * x^2 - 12 * x + 9

theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = Iio 1 ∪ Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3203_320326


namespace NUMINAMATH_CALUDE_class_average_problem_l3203_320357

theorem class_average_problem (n1 n2 : ℕ) (avg2 avg_all : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
  (h3 : avg2 = 60) (h4 : avg_all = 56.25) : 
  (n1 + n2 : ℝ) * avg_all = n1 * ((n1 + n2 : ℝ) * avg_all - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

#check class_average_problem

end NUMINAMATH_CALUDE_class_average_problem_l3203_320357


namespace NUMINAMATH_CALUDE_stratified_sampling_used_l3203_320328

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | SamplingByLot
  | RandomNumberTable
  | Stratified

/-- Represents a population with two strata -/
structure Population where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Represents a sample from a population with two strata -/
structure Sample where
  total : Nat
  stratum1 : Nat
  stratum2 : Nat
  h_sum : stratum1 + stratum2 = total

/-- Determines if the sampling method is stratified based on population and sample data -/
def isStratifiedSampling (pop : Population) (sample : Sample) : Prop :=
  (pop.stratum1 : Rat) / pop.total = (sample.stratum1 : Rat) / sample.total ∧
  (pop.stratum2 : Rat) / pop.total = (sample.stratum2 : Rat) / sample.total

/-- The theorem to be proved -/
theorem stratified_sampling_used
  (pop : Population)
  (sample : Sample)
  (h_pop : pop = { total := 900, stratum1 := 500, stratum2 := 400, h_sum := rfl })
  (h_sample : sample = { total := 45, stratum1 := 25, stratum2 := 20, h_sum := rfl }) :
  isStratifiedSampling pop sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_used_l3203_320328


namespace NUMINAMATH_CALUDE_room_with_193_black_tiles_has_1089_total_tiles_l3203_320363

/-- Represents a square room with tiled floor -/
structure TiledRoom where
  side_length : ℕ
  black_tile_count : ℕ

/-- Calculates the number of black tiles in a square room with given side length -/
def black_tiles (s : ℕ) : ℕ := 6 * s - 5

/-- Calculates the total number of tiles in a square room with given side length -/
def total_tiles (s : ℕ) : ℕ := s * s

/-- Theorem stating that a square room with 193 black tiles has 1089 total tiles -/
theorem room_with_193_black_tiles_has_1089_total_tiles :
  ∃ (room : TiledRoom), room.black_tile_count = 193 ∧ total_tiles room.side_length = 1089 :=
by sorry

end NUMINAMATH_CALUDE_room_with_193_black_tiles_has_1089_total_tiles_l3203_320363


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3203_320372

/-- A geometric sequence with the given first four terms -/
def geometric_sequence : Fin 5 → ℝ
  | 0 => 10
  | 1 => -15
  | 2 => 22.5
  | 3 => -33.75
  | 4 => 50.625

/-- The common ratio of the geometric sequence -/
def common_ratio : ℝ := -1.5

theorem geometric_sequence_properties :
  (∀ n : Fin 3, geometric_sequence (n + 1) = geometric_sequence n * common_ratio) ∧
  geometric_sequence 4 = geometric_sequence 3 * common_ratio :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3203_320372


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l3203_320314

theorem sum_of_solutions_equation (x : ℝ) :
  (x ≠ 1 ∧ x ≠ -1) →
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 8 / (x - 1)) →
  ∃ (y : ℝ), (y ≠ 1 ∧ y ≠ -1) ∧
    ((-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 8 / (y - 1)) ∧
    (x + y = 10 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l3203_320314


namespace NUMINAMATH_CALUDE_unique_set_l3203_320364

def is_valid_set (A : Finset ℤ) : Prop :=
  A.card = 4 ∧
  ∀ (subset : Finset ℤ), subset ⊆ A → subset.card = 3 →
    (subset.sum id) ∈ ({-1, 5, 3, 8} : Finset ℤ)

theorem unique_set :
  ∃! (A : Finset ℤ), is_valid_set A ∧ A = {-3, 0, 2, 6} :=
sorry

end NUMINAMATH_CALUDE_unique_set_l3203_320364


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l3203_320399

theorem smallest_interesting_number :
  ∃ (n : ℕ), n = 1800 ∧
  (∀ m : ℕ, m < n →
    ¬(∃ k : ℕ, 2 * m = k ^ 2) ∨
    ¬(∃ l : ℕ, 15 * m = l ^ 3)) ∧
  (∃ k : ℕ, 2 * n = k ^ 2) ∧
  (∃ l : ℕ, 15 * n = l ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l3203_320399


namespace NUMINAMATH_CALUDE_kimberly_skittles_l3203_320362

/-- The number of Skittles Kimberly initially had -/
def initial_skittles : ℕ := 5

/-- The number of Skittles Kimberly bought -/
def bought_skittles : ℕ := 7

/-- The total number of Skittles Kimberly has after buying more -/
def total_skittles : ℕ := 12

/-- Theorem stating that the initial number of Skittles plus the bought Skittles equals the total Skittles -/
theorem kimberly_skittles : initial_skittles + bought_skittles = total_skittles := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l3203_320362


namespace NUMINAMATH_CALUDE_exponential_function_determination_l3203_320344

theorem exponential_function_determination (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 2 = 4) :
  ∀ x, f x = 2^x :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_determination_l3203_320344


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3203_320369

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3203_320369


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3203_320316

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3203_320316


namespace NUMINAMATH_CALUDE_eagles_score_l3203_320311

/-- Given the total points and margin of victory in a basketball game, prove the losing team's score. -/
theorem eagles_score (total_points margin : ℕ) (h1 : total_points = 82) (h2 : margin = 18) :
  (total_points - margin) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_eagles_score_l3203_320311


namespace NUMINAMATH_CALUDE_seventh_root_unity_sum_l3203_320309

theorem seventh_root_unity_sum (q : ℂ) (h : q^7 = 1) :
  q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6) = 
    if q = 1 then (3 : ℂ) / 2 else -2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_sum_l3203_320309


namespace NUMINAMATH_CALUDE_jessica_pie_count_l3203_320358

theorem jessica_pie_count (apples_per_serving : ℝ) (num_guests : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ)
  (h1 : apples_per_serving = 1.5)
  (h2 : num_guests = 12)
  (h3 : servings_per_pie = 8)
  (h4 : apples_per_guest = 3) :
  (num_guests * apples_per_guest / apples_per_serving) / servings_per_pie = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_pie_count_l3203_320358


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3203_320312

theorem greatest_divisor_with_remainders : 
  let a := 690
  let b := 875
  let r₁ := 10
  let r₂ := 25
  Int.gcd (a - r₁) (b - r₂) = 170 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3203_320312


namespace NUMINAMATH_CALUDE_ellipse_t_range_l3203_320333

def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (10 - t) + y^2 / (t - 4) = 1

theorem ellipse_t_range :
  {t : ℝ | is_ellipse t} = {t | t ∈ (Set.Ioo 4 7) ∪ (Set.Ioo 7 10)} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_t_range_l3203_320333


namespace NUMINAMATH_CALUDE_parabola_circle_problem_l3203_320380

/-- Parabola in the Cartesian coordinate system -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Circle in the Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The setup of the problem -/
structure ParabolaCircleSetup where
  C : Parabola
  Q : Circle
  h_Q_passes_O : Q.center.1^2 + Q.center.2^2 = Q.radius^2
  h_Q_passes_F : (Q.center.1 - C.p/2)^2 + Q.center.2^2 = Q.radius^2
  h_Q_center_directrix : Q.center.1 + C.p/2 = 3/2

/-- The theorem to be proved -/
theorem parabola_circle_problem (setup : ParabolaCircleSetup) :
  -- 1. The equation of parabola C is y^2 = 4x
  setup.C.p = 2 ∧
  -- 2. For any point M(t, 4) on C and chords MD and ME with MD ⊥ ME,
  --    the line DE passes through the fixed point (8, -4)
  ∀ t : ℝ, setup.C.eq t 4 →
    ∀ D E : ℝ × ℝ, setup.C.eq D.1 D.2 → setup.C.eq E.1 E.2 →
      (t - D.1) * (t - E.1) + (4 - D.2) * (4 - E.2) = 0 →
        ∃ m : ℝ, (D.1 = m * (D.2 + 4) + 8 ∧ E.1 = m * (E.2 + 4) + 8) ∨
                 (D.1 = m * (D.2 - 4) + 4 ∧ E.1 = m * (E.2 - 4) + 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_problem_l3203_320380


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3203_320318

/-- Represents a co-ed softball team with different skill levels -/
structure SoftballTeam where
  beginnerMen : ℕ
  beginnerWomen : ℕ
  intermediateMen : ℕ
  intermediateWomen : ℕ
  advancedMen : ℕ
  advancedWomen : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.beginnerMen = 2 ∧ 
  team.beginnerWomen = 4 ∧
  team.intermediateMen = 3 ∧
  team.intermediateWomen = 5 ∧
  team.advancedMen = 1 ∧
  team.advancedWomen = 3 →
  (team.beginnerMen + team.intermediateMen + team.advancedMen) * 2 = 
  (team.beginnerWomen + team.intermediateWomen + team.advancedWomen) := by
  sorry

#check softball_team_ratio

end NUMINAMATH_CALUDE_softball_team_ratio_l3203_320318


namespace NUMINAMATH_CALUDE_smallest_even_three_digit_multiple_of_17_l3203_320304

theorem smallest_even_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 
    n % 17 = 0 ∧ 
    n % 2 = 0 ∧ 
    100 ≤ n ∧ n ≤ 999 → 
    n ≥ 136 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_three_digit_multiple_of_17_l3203_320304


namespace NUMINAMATH_CALUDE_misread_weight_l3203_320394

theorem misread_weight (class_size : ℕ) (incorrect_avg : ℚ) (correct_avg : ℚ) (correct_weight : ℚ) (x : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  correct_avg = 58.85 →
  correct_weight = 65 →
  class_size * correct_avg = class_size * incorrect_avg - x + correct_weight →
  x = 56 := by
sorry

end NUMINAMATH_CALUDE_misread_weight_l3203_320394


namespace NUMINAMATH_CALUDE_list_fraction_problem_l3203_320366

theorem list_fraction_problem (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6) * l.sum :=
sorry

end NUMINAMATH_CALUDE_list_fraction_problem_l3203_320366


namespace NUMINAMATH_CALUDE_mark_cereal_boxes_l3203_320390

def soup_cost : ℕ := 6 * 2
def bread_cost : ℕ := 2 * 5
def milk_cost : ℕ := 2 * 4
def cereal_cost : ℕ := 3
def total_payment : ℕ := 4 * 10

def cereal_boxes : ℕ := (total_payment - (soup_cost + bread_cost + milk_cost)) / cereal_cost

theorem mark_cereal_boxes : cereal_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_cereal_boxes_l3203_320390


namespace NUMINAMATH_CALUDE_notebook_pencil_cost_l3203_320359

/-- Given the prices of notebooks and pencils in two scenarios, prove the cost of one notebook and one pencil. -/
theorem notebook_pencil_cost
  (scenario1 : 6 * notebook_price + 4 * pencil_price = 9.2)
  (scenario2 : 3 * notebook_price + pencil_price = 3.8)
  : notebook_price + pencil_price = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pencil_cost_l3203_320359


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l3203_320324

/-- Represents the number of balloons a person has for each color -/
structure BalloonCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The total number of balloons for all people -/
def totalBalloons (people : List BalloonCount) : BalloonCount :=
  { blue := people.foldl (fun acc p => acc + p.blue) 0,
    red := people.foldl (fun acc p => acc + p.red) 0,
    green := people.foldl (fun acc p => acc + p.green) 0,
    yellow := people.foldl (fun acc p => acc + p.yellow) 0 }

theorem balloon_count_theorem (joan melanie eric : BalloonCount)
  (h_joan : joan = { blue := 40, red := 30, green := 0, yellow := 0 })
  (h_melanie : melanie = { blue := 41, red := 0, green := 20, yellow := 0 })
  (h_eric : eric = { blue := 0, red := 25, green := 0, yellow := 15 }) :
  totalBalloons [joan, melanie, eric] = { blue := 81, red := 55, green := 20, yellow := 15 } := by
  sorry

#check balloon_count_theorem

end NUMINAMATH_CALUDE_balloon_count_theorem_l3203_320324


namespace NUMINAMATH_CALUDE_special_function_monotonicity_l3203_320378

/-- A function f: ℝ → ℝ satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 - x) = f x) ∧
  (∀ x, (x - 1/2) * (deriv^[2] f x) > 0)

/-- Theorem stating the monotonicity property of the special function -/
theorem special_function_monotonicity 
  (f : ℝ → ℝ) (hf : SpecialFunction f) (x₁ x₂ : ℝ) 
  (h_order : x₁ < x₂) (h_sum : x₁ + x₂ > 1) : 
  f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_special_function_monotonicity_l3203_320378


namespace NUMINAMATH_CALUDE_multiplier_is_three_l3203_320387

theorem multiplier_is_three :
  ∃ (x : ℤ), 
    (3 * x = (62 - x) + 26) ∧ 
    (x = 22) → 
    3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l3203_320387


namespace NUMINAMATH_CALUDE_john_steve_race_l3203_320393

theorem john_steve_race (john_speed steve_speed : ℝ) (final_push_time : ℝ) (finish_ahead : ℝ) :
  john_speed = 4.2 →
  steve_speed = 3.8 →
  final_push_time = 42.5 →
  finish_ahead = 2 →
  john_speed * final_push_time - steve_speed * final_push_time - finish_ahead = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_steve_race_l3203_320393


namespace NUMINAMATH_CALUDE_john_ate_ten_chips_l3203_320308

/-- Calculates the number of potato chips eaten given the total calories and cheezit information. -/
def potato_chips_eaten (total_chip_calories : ℕ) (num_cheezits : ℕ) (total_calories : ℕ) : ℕ :=
  let chip_calories := total_chip_calories / (total_calories - total_chip_calories - num_cheezits * 4 / 3 * (total_chip_calories / (total_calories - total_chip_calories)))
  total_chip_calories / chip_calories

/-- Theorem stating that John ate 10 potato chips given the problem conditions. -/
theorem john_ate_ten_chips :
  potato_chips_eaten 60 6 108 = 10 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_ten_chips_l3203_320308


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l3203_320361

theorem max_value_x_sqrt_3_minus_x_squared (x : ℝ) (h1 : 0 < x) (h2 : x < Real.sqrt 3) :
  (∀ y, 0 < y ∧ y < Real.sqrt 3 → x * Real.sqrt (3 - x^2) ≥ y * Real.sqrt (3 - y^2)) →
  x * Real.sqrt (3 - x^2) = 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l3203_320361
