import Mathlib

namespace NUMINAMATH_CALUDE_mothers_day_discount_l3508_350835

theorem mothers_day_discount (original_price : ℝ) : 
  (original_price * 0.9 * 0.96 = 108) → original_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l3508_350835


namespace NUMINAMATH_CALUDE_product_of_roots_l3508_350898

theorem product_of_roots : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 + 18*r₁ + 30 = 2 * Real.sqrt (r₁^2 + 18*r₁ + 45)) ∧
  (r₂^2 + 18*r₂ + 30 = 2 * Real.sqrt (r₂^2 + 18*r₂ + 45)) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3508_350898


namespace NUMINAMATH_CALUDE_first_sales_amount_l3508_350804

/-- The amount of the first sales in millions of dollars -/
def first_sales : ℝ := sorry

/-- The profit on the first sales in millions of dollars -/
def first_profit : ℝ := 5

/-- The profit on the next $30 million in sales in millions of dollars -/
def second_profit : ℝ := 12

/-- The amount of the second sales in millions of dollars -/
def second_sales : ℝ := 30

/-- The increase in profit ratio from the first to the second sales -/
def profit_ratio_increase : ℝ := 0.2000000000000001

theorem first_sales_amount :
  (first_profit / first_sales) * (1 + profit_ratio_increase) = second_profit / second_sales ∧
  first_sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_sales_amount_l3508_350804


namespace NUMINAMATH_CALUDE_some_number_calculation_l3508_350865

theorem some_number_calculation (X : ℝ) : 
  2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002 → X = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_calculation_l3508_350865


namespace NUMINAMATH_CALUDE_triangle_formation_l3508_350849

-- Define the triangle formation condition
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_formation (a : ℝ) : 
  can_form_triangle 5 a 9 ↔ a = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3508_350849


namespace NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l3508_350864

def Universe := 200
def Mozart := 150
def Bach := 120
def Beethoven := 90

theorem min_mozart_bach_not_beethoven :
  ∃ (m b e mb mbe : ℕ),
    m ≤ Mozart ∧
    b ≤ Bach ∧
    e ≤ Beethoven ∧
    mb ≤ m ∧
    mb ≤ b ∧
    mbe ≤ mb ∧
    mbe ≤ e ∧
    m + b - mb ≤ Universe ∧
    m + b + e - mb - mbe ≤ Universe ∧
    mb - mbe ≥ 10 :=
  sorry

end NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l3508_350864


namespace NUMINAMATH_CALUDE_factorial_calculation_l3508_350876

theorem factorial_calculation : 
  Nat.factorial 8 - 7 * Nat.factorial 7 - 2 * Nat.factorial 6 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l3508_350876


namespace NUMINAMATH_CALUDE_converse_correct_l3508_350838

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Define the converse proposition
def converse_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Theorem stating that the converse_proposition is indeed the converse of the original_proposition
theorem converse_correct :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, converse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_converse_correct_l3508_350838


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3508_350880

theorem algebraic_expression_equality (y : ℝ) : 
  3 * y^2 - 2 * y + 6 = 8 → (3/2) * y^2 - y + 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3508_350880


namespace NUMINAMATH_CALUDE_chocolate_count_l3508_350863

/-- The number of large boxes in the massive crate -/
def large_boxes : ℕ := 54

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 24

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 37

/-- The total number of chocolate bars in the massive crate -/
def total_chocolates : ℕ := large_boxes * small_boxes_per_large * chocolates_per_small

theorem chocolate_count : total_chocolates = 47952 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l3508_350863


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3508_350874

-- Define the polynomial
def f (x : ℝ) := x^3 - x + 2

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f a = 0 ∧ f b = 0 ∧ f c = 0 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3508_350874


namespace NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l3508_350824

/-- Represents the population of Smithtown -/
structure Population where
  right_handed : ℕ
  left_handed : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for the Smithtown population -/
def is_valid_smithtown_population (p : Population) : Prop :=
  p.right_handed = 3 * p.left_handed ∧
  3 * p.women = 2 * p.men ∧
  p.men ≤ p.right_handed

/-- Theorem stating that in a valid Smithtown population, 
    25% of the total population are left-handed women -/
theorem smithtown_left_handed_women_percentage 
  (p : Population) (h : is_valid_smithtown_population p) : 
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

#check smithtown_left_handed_women_percentage

end NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l3508_350824


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3508_350887

theorem inserted_numbers_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ a = 4 * r ∧ b = 4 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) →           -- Arithmetic progression condition
  b = a + 4 →                                   -- Difference condition
  a + b = 8 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3508_350887


namespace NUMINAMATH_CALUDE_gate_buyers_count_l3508_350854

/-- The number of people who pre-bought tickets -/
def preBuyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def prePrice : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gatePrice : ℕ := 200

/-- The additional amount paid by gate buyers compared to pre-buyers -/
def additionalAmount : ℕ := 2900

/-- The number of people who bought tickets at the gate -/
def gateBuyers : ℕ := 30

theorem gate_buyers_count :
  gateBuyers * gatePrice = preBuyers * prePrice + additionalAmount := by
  sorry

end NUMINAMATH_CALUDE_gate_buyers_count_l3508_350854


namespace NUMINAMATH_CALUDE_base5_44_equals_binary_10111_l3508_350858

-- Define a function to convert a base-5 number to decimal
def base5ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Theorem stating that (44)₅ in base-5 is equal to (10111)₂ in binary
theorem base5_44_equals_binary_10111 :
  decimalToBinary (base5ToDecimal 44) = [1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base5_44_equals_binary_10111_l3508_350858


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3508_350878

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 2 * gridSize + 2

/-- The total number of ways to choose 4 dots from 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : ℚ := collinearWays / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 12 / 12650 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3508_350878


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3508_350870

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
  a₁ = 16 → 
  aₙ = 32 → 
  d = 1/3 → 
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 1176 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3508_350870


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l3508_350856

theorem quadratic_equation_sum (a b : ℝ) : 
  a^2 - 2*a + 1 = 49 →
  b^2 - 2*b + 1 = 49 →
  a ≥ b →
  3*a + 2*b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l3508_350856


namespace NUMINAMATH_CALUDE_zero_in_interval_l3508_350815

noncomputable def f (x : ℝ) : ℝ := 2^x - 6 - Real.log x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3508_350815


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3508_350821

/-- A rectangle with integer dimensions and perimeter 200 has a minimum area of 99 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter condition
  (l * w ≥ 99) :=          -- minimum area
by sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3508_350821


namespace NUMINAMATH_CALUDE_length_of_AB_l3508_350882

-- Define the points
variable (A B C D E F G : ℝ)

-- Define the conditions
variable (h1 : C = (A + B) / 2)
variable (h2 : D = (A + C) / 2)
variable (h3 : E = (A + D) / 2)
variable (h4 : F = (A + E) / 2)
variable (h5 : G = (A + F) / 2)
variable (h6 : G - A = 1)

-- State the theorem
theorem length_of_AB : B - A = 32 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_l3508_350882


namespace NUMINAMATH_CALUDE_families_with_car_or_ebike_l3508_350885

theorem families_with_car_or_ebike (total_car : ℕ) (total_ebike : ℕ) (both : ℕ) :
  total_car = 35 → total_ebike = 65 → both = 20 →
  total_car + total_ebike - both = 80 := by
  sorry

end NUMINAMATH_CALUDE_families_with_car_or_ebike_l3508_350885


namespace NUMINAMATH_CALUDE_range_interval_length_l3508_350847

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℚ := 3/2 * (-1/2)^(n-1)
def S (n : ℕ) : ℚ := 1 - (-1/2)^n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := S n + 1 / S n

-- State the theorem
theorem range_interval_length :
  (∀ n : ℕ, n > 0 → -2 * S 2 + 4 * S 4 = 2 * S 3) →
  (∃ L : ℚ, L > 0 ∧ ∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L ∧ b n ∈ Set.Icc x y) ∧
  (∀ L' : ℚ, L' > 0 → (∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L' ∧ b n ∈ Set.Icc x y) → L' ≥ 1/6) :=
sorry

end NUMINAMATH_CALUDE_range_interval_length_l3508_350847


namespace NUMINAMATH_CALUDE_candy_packing_problem_l3508_350894

theorem candy_packing_problem (a : ℕ) : 
  (a % 10 = 6) ∧ 
  (a % 15 = 11) ∧ 
  (200 ≤ a) ∧ 
  (a ≤ 250) ↔ 
  (a = 206 ∨ a = 236) :=
sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l3508_350894


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3508_350800

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3508_350800


namespace NUMINAMATH_CALUDE_circle_equation_condition_l3508_350814

/-- The equation x^2 + y^2 - x + y + m = 0 represents a circle if and only if m < 1/2 -/
theorem circle_equation_condition (x y m : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) ↔ 
  m < (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_circle_equation_condition_l3508_350814


namespace NUMINAMATH_CALUDE_park_visitors_l3508_350875

theorem park_visitors (hikers bike_riders total : ℕ) : 
  hikers = 427 →
  hikers = bike_riders + 178 →
  total = hikers + bike_riders →
  total = 676 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l3508_350875


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3508_350889

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x^(1/4) : ℝ) - 15 / (8 - (x^(1/4) : ℝ))
  ∀ x : ℝ, f x = 0 ↔ x = 81 ∨ x = 625 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3508_350889


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l3508_350801

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 9 * p - 15 = 0) → 
  (3 * q^2 - 9 * q - 15 = 0) → 
  (3 * p - 5) * (6 * q - 10) = -130 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l3508_350801


namespace NUMINAMATH_CALUDE_expr_D_not_fraction_l3508_350846

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (f g : ℚ → ℚ), ∀ x, expr x = (f x) / (g x) ∧ g x ≠ 0

-- Define the expressions
def expr_A (x : ℚ) : ℚ := 1 / (x^2)
def expr_B (a b : ℚ) : ℚ := (b + 3) / a
def expr_C (x : ℚ) : ℚ := (x^2 - 1) / (x + 1)
def expr_D (a : ℚ) : ℚ := (2 / 7) * a

-- Theorem stating that expr_D is not a fraction
theorem expr_D_not_fraction : ¬ is_fraction expr_D :=
sorry

end NUMINAMATH_CALUDE_expr_D_not_fraction_l3508_350846


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l3508_350841

theorem fraction_is_positive_integer (p : ℕ+) :
  (∃ k : ℕ+, (5 * p + 15 : ℚ) / (3 * p - 9 : ℚ) = k) ↔ 4 ≤ p ∧ p ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l3508_350841


namespace NUMINAMATH_CALUDE_pete_walked_3350_miles_l3508_350899

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer :=
  (max_reading : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (resets : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total miles walked based on the yearly walking data --/
def total_miles_walked (data : YearlyWalkingData) : ℕ :=
  ((data.resets * (data.pedometer.max_reading + 1) + data.final_reading) / data.steps_per_mile)

/-- Theorem stating that Pete walked 3350 miles given the problem conditions --/
theorem pete_walked_3350_miles :
  let petes_pedometer : Pedometer := ⟨99999⟩
  let petes_data : YearlyWalkingData := ⟨petes_pedometer, 50, 25000, 1500⟩
  total_miles_walked petes_data = 3350 := by
  sorry


end NUMINAMATH_CALUDE_pete_walked_3350_miles_l3508_350899


namespace NUMINAMATH_CALUDE_science_marks_calculation_l3508_350806

/-- Calculates the marks scored in science given the total marks and marks in other subjects. -/
def science_marks (total : ℕ) (music : ℕ) (social_studies : ℕ) : ℕ :=
  total - (music + social_studies + music / 2)

/-- Theorem stating that given the specific marks, the science marks must be 70. -/
theorem science_marks_calculation :
  science_marks 275 80 85 = 70 := by
  sorry

end NUMINAMATH_CALUDE_science_marks_calculation_l3508_350806


namespace NUMINAMATH_CALUDE_equation_solution_l3508_350877

theorem equation_solution : 
  ∃! x : ℚ, x ≠ 3 ∧ (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3508_350877


namespace NUMINAMATH_CALUDE_cupcake_net_profit_l3508_350808

/-- Calculates the net profit from selling cupcakes given the specified conditions -/
theorem cupcake_net_profit :
  let cupcake_cost : ℚ := 0.75
  let burnt_cupcakes : ℕ := 24
  let first_batch : ℕ := 24
  let second_batch : ℕ := 24
  let eaten_immediately : ℕ := 5
  let eaten_later : ℕ := 4
  let selling_price : ℚ := 2

  let total_cupcakes : ℕ := burnt_cupcakes + first_batch + second_batch
  let total_cost : ℚ := cupcake_cost * total_cupcakes
  let cupcakes_to_sell : ℕ := total_cupcakes - burnt_cupcakes - eaten_immediately - eaten_later
  let revenue : ℚ := selling_price * cupcakes_to_sell
  let net_profit : ℚ := revenue - total_cost

  net_profit = 72 := by sorry

end NUMINAMATH_CALUDE_cupcake_net_profit_l3508_350808


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3508_350828

/-- Given a geometric sequence {a_n} with common ratio q = 2 and sum of first n terms S_n,
    prove that S_4 / a_4 = 15/8 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →
  S 4 / a 4 = 15/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3508_350828


namespace NUMINAMATH_CALUDE_cone_slant_height_is_10_l3508_350884

/-- The slant height of a cone, given its base radius and that its lateral surface unfolds into a semicircle. -/
def slant_height (base_radius : ℝ) : ℝ :=
  2 * base_radius

theorem cone_slant_height_is_10 :
  let base_radius : ℝ := 5
  slant_height base_radius = 10 :=
by sorry

end NUMINAMATH_CALUDE_cone_slant_height_is_10_l3508_350884


namespace NUMINAMATH_CALUDE_f_properties_l3508_350848

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi/6) + 1

theorem f_properties :
  let T := Real.pi
  let interval := Set.Icc (Real.pi/4) ((2*Real.pi)/3)
  (∀ x, f (x + T) = f x) ∧  -- Smallest positive period
  (∀ x ∈ interval, f x ≤ 2) ∧  -- Maximum value
  (∃ x ∈ interval, f x = 2) ∧  -- Maximum value is attained
  (∀ x ∈ interval, f x ≥ -1) ∧  -- Minimum value
  (∃ x ∈ interval, f x = -1) :=  -- Minimum value is attained
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3508_350848


namespace NUMINAMATH_CALUDE_highest_uniquely_identifiable_score_l3508_350820

/-- The AHSME scoring system -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- The maximum number of questions in AHSME -/
def max_questions : ℕ := 30

/-- Predicate to check if a score is uniquely identifiable -/
def is_uniquely_identifiable (s : ℕ) : Prop :=
  ∃! (c w : ℕ), c ≤ max_questions ∧ w ≤ max_questions ∧ s = score c w

/-- Theorem stating that 130 is the highest possible uniquely identifiable score over 100 -/
theorem highest_uniquely_identifiable_score :
  (∀ s : ℕ, s > 130 → ¬(is_uniquely_identifiable s)) ∧
  (is_uniquely_identifiable 130) ∧
  (130 > 100) :=
sorry

end NUMINAMATH_CALUDE_highest_uniquely_identifiable_score_l3508_350820


namespace NUMINAMATH_CALUDE_mexican_restaurant_bill_solution_l3508_350802

/-- Represents the cost of items at a Mexican restaurant -/
structure MexicanRestaurantCosts where
  T : ℝ  -- Cost of a taco
  E : ℝ  -- Cost of an enchilada
  B : ℝ  -- Cost of a burrito

/-- The bills for three friends at a Mexican restaurant -/
def friend_bills (c : MexicanRestaurantCosts) : Prop :=
  2 * c.T + 3 * c.E = 7.80 ∧
  3 * c.T + 5 * c.E = 12.70 ∧
  4 * c.T + 2 * c.E + c.B = 15.40

/-- The theorem stating the unique solution for the Mexican restaurant bill problem -/
theorem mexican_restaurant_bill_solution :
  ∃! c : MexicanRestaurantCosts, friend_bills c ∧ c.T = 0.90 ∧ c.E = 2.00 ∧ c.B = 7.80 :=
by sorry

end NUMINAMATH_CALUDE_mexican_restaurant_bill_solution_l3508_350802


namespace NUMINAMATH_CALUDE_range_of_a_l3508_350886

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a : 
  (∀ x, p x → (∀ a, q x a)) ∧ 
  (∃ x a, ¬(p x) ∧ q x a) → 
  ∀ a, 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3508_350886


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3508_350868

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define condition p
def condition_p (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

-- Define condition q
def condition_q (t : Triangle) : Prop :=
  t.A - t.B = t.B - t.C

-- Theorem stating p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ t : Triangle, condition_q t → condition_p t) ∧
  ¬(∀ t : Triangle, condition_p t → condition_q t) := by
  sorry


end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3508_350868


namespace NUMINAMATH_CALUDE_room_area_square_inches_l3508_350822

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Define the length of the room in feet
def room_length_feet : ℕ := 10

-- Theorem: The area of the room in square inches is 14400
theorem room_area_square_inches :
  (room_length_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_room_area_square_inches_l3508_350822


namespace NUMINAMATH_CALUDE_a_10_value_l3508_350851

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 5 ∧ a 7 = -7 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In the given arithmetic sequence, a_10 = -16 -/
theorem a_10_value (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 10 = -16 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l3508_350851


namespace NUMINAMATH_CALUDE_tau_prime_factors_divide_l3508_350823

/-- The number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- For positive integers a and b, if σ(a^n) divides σ(b^n) for all n ∈ ℕ,
    then each prime factor of τ(a) divides τ(b) -/
theorem tau_prime_factors_divide (a b : ℕ+) 
  (h : ∀ n : ℕ, (sigma (a^n) : ℕ) ∣ (sigma (b^n) : ℕ)) :
  ∀ p : ℕ, Prime p → p ∣ tau a → p ∣ tau b := by
  sorry

end NUMINAMATH_CALUDE_tau_prime_factors_divide_l3508_350823


namespace NUMINAMATH_CALUDE_kindergarten_pet_distribution_l3508_350855

/-- Represents the kindergarten pet distribution problem -/
theorem kindergarten_pet_distribution 
  (total_children : ℕ) 
  (children_with_both : ℕ) 
  (children_with_cats : ℕ) 
  (h1 : total_children = 30)
  (h2 : children_with_both = 6)
  (h3 : children_with_cats = 12)
  : total_children - children_with_cats = 18 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_pet_distribution_l3508_350855


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3508_350807

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 162 ∧ x = 13.5 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3508_350807


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3508_350819

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 - 7 * x > 6) ↔ (x < -2/3 ∨ x > 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3508_350819


namespace NUMINAMATH_CALUDE_summer_camp_selection_probability_l3508_350890

theorem summer_camp_selection_probability :
  let total_students : ℕ := 9
  let male_students : ℕ := 5
  let female_students : ℕ := 4
  let selected_students : ℕ := 5
  let min_per_gender : ℕ := 2

  let total_combinations := Nat.choose total_students selected_students
  let valid_combinations := Nat.choose male_students min_per_gender * Nat.choose female_students (selected_students - min_per_gender) +
                            Nat.choose male_students (selected_students - min_per_gender) * Nat.choose female_students min_per_gender

  (valid_combinations : ℚ) / total_combinations = 50 / 63 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_selection_probability_l3508_350890


namespace NUMINAMATH_CALUDE_motel_total_rent_l3508_350816

/-- Represents the total rent charged by a motel on a Saturday night. -/
def total_rent (r40 r60 : ℕ) : ℕ := 40 * r40 + 60 * r60

/-- The condition that changing 10 rooms from $60 to $40 reduces the total rent by 50%. -/
def rent_reduction_condition (r40 r60 : ℕ) : Prop :=
  total_rent (r40 + 10) (r60 - 10) = (total_rent r40 r60) / 2

/-- The theorem stating that the total rent charged by the motel is $800. -/
theorem motel_total_rent :
  ∃ (r40 r60 : ℕ), rent_reduction_condition r40 r60 ∧ total_rent r40 r60 = 800 :=
sorry

end NUMINAMATH_CALUDE_motel_total_rent_l3508_350816


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l3508_350892

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The reverse of a two-digit number. -/
def reverse (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- The property that a number, when added to its reverse, sums to 144. -/
def hasProperty (n : ℕ) : Prop := n + reverse n = 144

/-- The main theorem stating that there are exactly 6 two-digit numbers satisfying the property. -/
theorem count_numbers_with_property : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ hasProperty n) ∧ Finset.card s = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l3508_350892


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l3508_350817

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 390 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  c = 119 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_cherries_l3508_350817


namespace NUMINAMATH_CALUDE_complex_multiplication_l3508_350871

theorem complex_multiplication (z : ℂ) : z = 2 - I → I^3 * z = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3508_350871


namespace NUMINAMATH_CALUDE_inequality_proof_l3508_350843

theorem inequality_proof (n : ℕ) (h : n > 2) :
  (2*n - 1)^n + (2*n)^n < (2*n + 1)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3508_350843


namespace NUMINAMATH_CALUDE_transform_sine_function_l3508_350862

/-- Given a function f and its transformed version g, 
    where g is obtained by shortening the abscissas of f to half their original length 
    and then shifting the resulting curve to the right by π/3 units,
    prove that f(x) = sin(x/2 + π/12) if g(x) = sin(x - π/4) -/
theorem transform_sine_function (f g : ℝ → ℝ) :
  (∀ x, g x = f ((x - π/3) / 2)) →
  (∀ x, g x = Real.sin (x - π/4)) →
  ∀ x, f x = Real.sin (x/2 + π/12) := by
sorry

end NUMINAMATH_CALUDE_transform_sine_function_l3508_350862


namespace NUMINAMATH_CALUDE_g_five_equals_one_l3508_350891

theorem g_five_equals_one (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x * g y) 
  (h2 : ∀ x : ℝ, g x ≠ 0) : 
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l3508_350891


namespace NUMINAMATH_CALUDE_pictures_per_album_l3508_350831

/-- Proves that the number of pictures in each album is the total number of pictures divided by the number of albums --/
theorem pictures_per_album 
  (phone_pics camera_pics total_pics num_albums : ℕ) 
  (h1 : total_pics = phone_pics + camera_pics)
  (h2 : num_albums = 3)
  (h3 : total_pics % num_albums = 0) : 
  total_pics / num_albums = total_pics / num_albums :=
by sorry

end NUMINAMATH_CALUDE_pictures_per_album_l3508_350831


namespace NUMINAMATH_CALUDE_f_is_even_l3508_350893

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := |h (x^5)|

-- Theorem statement
theorem f_is_even (h : ℝ → ℝ) (h_even : IsEven h) : IsEven (f h) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3508_350893


namespace NUMINAMATH_CALUDE_special_polynomial_value_at_zero_l3508_350825

/-- A polynomial of degree 6 satisfying p(3^n) = 1/(3^n) for n = 0, 1, 2, ..., 6 -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d e f g : ℝ, ∀ x, p x = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g) ∧
  (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n))

theorem special_polynomial_value_at_zero 
  (p : ℝ → ℝ) (h : special_polynomial p) : p 0 = 2186 / 729 :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_value_at_zero_l3508_350825


namespace NUMINAMATH_CALUDE_two_solutions_l3508_350895

-- Define the equation
def equation (x a : ℝ) : Prop := abs (x - 3) = a * x - 1

-- Define the condition for two solutions
theorem two_solutions (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ a ∧ equation x₂ a) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l3508_350895


namespace NUMINAMATH_CALUDE_patrick_caught_eight_l3508_350850

/-- The number of fish caught by each person -/
structure FishCaught where
  patrick : ℕ
  angus : ℕ
  ollie : ℕ

/-- The conditions of the fishing problem -/
def fishing_conditions (fc : FishCaught) : Prop :=
  fc.angus = fc.patrick + 4 ∧
  fc.ollie = fc.angus - 7 ∧
  fc.ollie = 5

/-- Theorem: Given the fishing conditions, Patrick caught 8 fish -/
theorem patrick_caught_eight (fc : FishCaught) 
  (h : fishing_conditions fc) : fc.patrick = 8 := by
  sorry

end NUMINAMATH_CALUDE_patrick_caught_eight_l3508_350850


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l3508_350867

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes

/-- Calculates the total minutes in a workday -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of workday spent in meetings is 40% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l3508_350867


namespace NUMINAMATH_CALUDE_cubic_polynomial_q_value_l3508_350810

/-- A cubic polynomial Q(x) = x^3 + px^2 + qx + d -/
def cubicPolynomial (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

theorem cubic_polynomial_q_value 
  (p q d : ℝ) 
  (h1 : -(p/3) = q) -- mean of zeros equals product of zeros taken two at a time
  (h2 : q = 1 + p + q + d) -- product of zeros taken two at a time equals sum of coefficients
  (h3 : d = 7) -- y-intercept is 7
  : q = 8/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_q_value_l3508_350810


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l3508_350869

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (total_subjects : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 89)
  (h3 : physics = 82)
  (h4 : biology = 81)
  (h5 : average = 85)
  (h6 : total_subjects = 5) :
  let total_marks := average * total_subjects
  let known_subjects_marks := english + mathematics + physics + biology
  let chemistry := total_marks - known_subjects_marks
  chemistry = 87 := by
    sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l3508_350869


namespace NUMINAMATH_CALUDE_emily_beads_count_l3508_350834

/-- Given that Emily makes necklaces where each necklace requires 12 beads,
    and she made 7 necklaces, prove that the total number of beads she had is 84. -/
theorem emily_beads_count :
  let beads_per_necklace : ℕ := 12
  let necklaces_made : ℕ := 7
  beads_per_necklace * necklaces_made = 84 :=
by sorry

end NUMINAMATH_CALUDE_emily_beads_count_l3508_350834


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3508_350866

theorem triangle_perimeter_bound (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  ConvexOn ℝ (Set.Ioo 0 π) Real.sin →
  2 * R * (Real.sin A + Real.sin B + Real.sin C) ≤ 3 * Real.sqrt 3 * R :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3508_350866


namespace NUMINAMATH_CALUDE_chocolate_savings_theorem_l3508_350852

/-- Represents the cost and packaging details of a chocolate store -/
structure ChocolateStore where
  cost_per_chocolate : ℚ
  pack_size : ℕ

/-- Calculates the cost for a given number of weeks at a store -/
def calculate_cost (store : ChocolateStore) (weeks : ℕ) : ℚ :=
  let chocolates_needed := 2 * weeks
  let packs_needed := (chocolates_needed + store.pack_size - 1) / store.pack_size
  ↑packs_needed * store.pack_size * store.cost_per_chocolate

/-- The problem statement -/
theorem chocolate_savings_theorem :
  let local_store := ChocolateStore.mk 3 1
  let store_a := ChocolateStore.mk 2 5
  let store_b := ChocolateStore.mk (5/2) 1
  let store_c := ChocolateStore.mk (9/5) 10
  let weeks := 13
  let local_cost := calculate_cost local_store weeks
  let cost_a := calculate_cost store_a weeks
  let cost_b := calculate_cost store_b weeks
  let cost_c := calculate_cost store_c weeks
  let savings_a := local_cost - cost_a
  let savings_b := local_cost - cost_b
  let savings_c := local_cost - cost_c
  let max_savings := max savings_a (max savings_b savings_c)
  max_savings = 28 := by sorry

end NUMINAMATH_CALUDE_chocolate_savings_theorem_l3508_350852


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3508_350840

theorem quadratic_equation_value (x : ℝ) (h : x^2 - 3*x = 4) : 2*x^2 - 6*x - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3508_350840


namespace NUMINAMATH_CALUDE_coffee_mixture_proof_l3508_350859

/-- The cost of Colombian coffee beans per pound -/
def colombian_cost : ℝ := 5.50

/-- The cost of Peruvian coffee beans per pound -/
def peruvian_cost : ℝ := 4.25

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 40

/-- The desired cost per pound of the mixture -/
def mixture_cost : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mixture -/
def colombian_amount : ℝ := 11.2

theorem coffee_mixture_proof :
  colombian_amount * colombian_cost + (total_weight - colombian_amount) * peruvian_cost = 
  mixture_cost * total_weight :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_proof_l3508_350859


namespace NUMINAMATH_CALUDE_unique_valid_tournament_l3508_350833

/-- Represents the result of a chess game -/
inductive GameResult
  | Win
  | Draw
  | Loss

/-- Represents a player in the chess tournament -/
structure Player where
  id : Fin 5
  score : Rat

/-- Represents the result of a game between two players -/
structure GameOutcome where
  player1 : Fin 5
  player2 : Fin 5
  result : GameResult

/-- Represents the chess tournament -/
structure ChessTournament where
  players : Fin 5 → Player
  games : List GameOutcome

def ChessTournament.isValid (t : ChessTournament) : Prop :=
  -- Each player played exactly once with each other
  (t.games.length = 10) ∧
  -- First-place winner had no draws
  (¬ ∃ g ∈ t.games, g.player1 = 0 ∧ g.result = GameResult.Draw) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 0 ∧ g.result = GameResult.Draw) ∧
  -- Second-place winner did not lose any game
  (¬ ∃ g ∈ t.games, g.player1 = 1 ∧ g.result = GameResult.Loss) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 1 ∧ g.result = GameResult.Win) ∧
  -- Fourth-place player did not win any game
  (¬ ∃ g ∈ t.games, g.player1 = 3 ∧ g.result = GameResult.Win) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 3 ∧ g.result = GameResult.Loss) ∧
  -- Scores of all participants were different
  (∀ i j : Fin 5, i ≠ j → (t.players i).score ≠ (t.players j).score)

/-- The unique valid tournament configuration -/
def uniqueTournament : ChessTournament := sorry

theorem unique_valid_tournament :
  ∀ t : ChessTournament, t.isValid → t = uniqueTournament := by sorry

end NUMINAMATH_CALUDE_unique_valid_tournament_l3508_350833


namespace NUMINAMATH_CALUDE_problem_solution_l3508_350812

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / (a * x)

theorem problem_solution (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, ∀ y > 0, x < Real.exp 1 → y > Real.exp 1 → f a x < f a y) ∧
  (∀ x > 0, f a x ≤ x - 1/a → a ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → x₂ * Real.log x₁ + x₁ * Real.log x₂ = 0 → x₁ + x₂ > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3508_350812


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l3508_350888

theorem pigeonhole_on_permutation_sums (n : ℕ) :
  ∀ (p : Fin (2 * n) → Fin (2 * n)),
  Function.Bijective p →
  ∃ i j : Fin (2 * n), i ≠ j ∧ 
    (p i + i.val + 1) % (2 * n) = (p j + j.val + 1) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l3508_350888


namespace NUMINAMATH_CALUDE_factorization_difference_l3508_350879

theorem factorization_difference (c d : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d)) → c - d = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_l3508_350879


namespace NUMINAMATH_CALUDE_alice_prob_after_three_turns_l3508_350857

/-- Represents the player who has the ball -/
inductive Player
  | Alice
  | Bob

/-- The game state after each turn -/
def GameState := List Player

/-- The probability of Alice keeping the ball on her turn -/
def aliceKeepProb : ℚ := 2/3

/-- The probability of Bob keeping the ball on his turn -/
def bobKeepProb : ℚ := 2/3

/-- The initial game state with Alice having the ball -/
def initialState : GameState := [Player.Alice]

/-- Calculates the probability of a specific game state after three turns -/
def stateProb (state : GameState) : ℚ :=
  match state with
  | [Player.Alice, Player.Alice, Player.Alice, Player.Alice] => aliceKeepProb * aliceKeepProb * aliceKeepProb
  | [Player.Alice, Player.Alice, Player.Bob, Player.Alice] => aliceKeepProb * (1 - aliceKeepProb) * (1 - bobKeepProb)
  | [Player.Alice, Player.Bob, Player.Alice, Player.Alice] => (1 - aliceKeepProb) * (1 - bobKeepProb) * aliceKeepProb
  | [Player.Alice, Player.Bob, Player.Bob, Player.Alice] => (1 - aliceKeepProb) * bobKeepProb * (1 - bobKeepProb)
  | _ => 0

/-- All possible game states after three turns where Alice ends up with the ball -/
def validStates : List GameState := [
  [Player.Alice, Player.Alice, Player.Alice, Player.Alice],
  [Player.Alice, Player.Alice, Player.Bob, Player.Alice],
  [Player.Alice, Player.Bob, Player.Alice, Player.Alice],
  [Player.Alice, Player.Bob, Player.Bob, Player.Alice]
]

/-- The main theorem: probability of Alice having the ball after three turns is 14/27 -/
theorem alice_prob_after_three_turns :
  (validStates.map stateProb).sum = 14/27 := by
  sorry


end NUMINAMATH_CALUDE_alice_prob_after_three_turns_l3508_350857


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l3508_350832

theorem mod_equivalence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 28726 ≡ n [ZMOD 17] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l3508_350832


namespace NUMINAMATH_CALUDE_magic_square_sum_l3508_350896

theorem magic_square_sum (a b c d e f g : ℕ) : 
  (a + 13 + 12 + 1 = 34) →
  (g + 13 + 2 + 16 = 34) →
  (f + 16 + 9 + 4 = 34) →
  (c + 1 + 15 + 4 = 34) →
  (b + 12 + 7 + 9 = 34) →
  (d + 15 + 6 + 3 = 34) →
  (e + 2 + 7 + 14 = 34) →
  a - b - c + d + e + f - g = 11 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l3508_350896


namespace NUMINAMATH_CALUDE_equation_solution_l3508_350883

theorem equation_solution : ∃! x : ℝ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3508_350883


namespace NUMINAMATH_CALUDE_max_digits_after_subtraction_l3508_350805

theorem max_digits_after_subtraction :
  ∀ (a b c : ℕ),
  10000 ≤ a ∧ a ≤ 99999 →
  1000 ≤ b ∧ b ≤ 9999 →
  0 ≤ c ∧ c ≤ 9 →
  (Nat.digits 10 (a * b - c)).length ≤ 9 ∧
  ∃ (x y z : ℕ),
    10000 ≤ x ∧ x ≤ 99999 ∧
    1000 ≤ y ∧ y ≤ 9999 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (Nat.digits 10 (x * y - z)).length = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_after_subtraction_l3508_350805


namespace NUMINAMATH_CALUDE_volunteers_arrangement_count_l3508_350853

/-- The number of ways to arrange volunteers for tasks. -/
def arrangeVolunteers (volunteers : ℕ) (tasks : ℕ) : ℕ :=
  (tasks - 1).choose (volunteers - 1) * volunteers.factorial

/-- Theorem stating the number of arrangements for 4 volunteers and 5 tasks. -/
theorem volunteers_arrangement_count :
  arrangeVolunteers 4 5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_arrangement_count_l3508_350853


namespace NUMINAMATH_CALUDE_recurrence_initial_values_l3508_350803

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = (x n ^ 2 + 10) / 7

/-- The property of being bounded above -/
def BoundedAbove (x : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, x n ≤ M

/-- The set of possible initial values for bounded sequences satisfying the recurrence -/
def PossibleInitialValues : Set ℝ :=
  {x₀ : ℝ | ∃ x : ℤ → ℝ, RecurrenceSequence x ∧ BoundedAbove x ∧ x 0 = x₀}

theorem recurrence_initial_values :
    PossibleInitialValues = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_initial_values_l3508_350803


namespace NUMINAMATH_CALUDE_simplify_expression_l3508_350829

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3508_350829


namespace NUMINAMATH_CALUDE_dart_partitions_l3508_350872

def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem dart_partitions :
  partition_count 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dart_partitions_l3508_350872


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3508_350827

def henry_present_age : ℕ := 27
def jill_present_age : ℕ := 16

theorem age_ratio_proof :
  (henry_present_age - 5) / (jill_present_age - 5) = 2 ∧
  henry_present_age + jill_present_age = 43 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3508_350827


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3508_350844

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4*z) = 7 ∧ z = -23/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3508_350844


namespace NUMINAMATH_CALUDE_exists_angle_sum_with_adjacent_le_180_l3508_350881

/-- A convex quadrilateral is a quadrilateral where each interior angle is less than 180 degrees. -/
structure ConvexQuadrilateral where
  angles : Fin 4 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  all_angles_less_than_180 : ∀ i, angles i < 180

/-- 
In any convex quadrilateral, there exists an angle such that the sum of 
this angle with each of its adjacent angles does not exceed 180°.
-/
theorem exists_angle_sum_with_adjacent_le_180 (q : ConvexQuadrilateral) : 
  ∃ i : Fin 4, (q.angles i + q.angles ((i + 1) % 4) ≤ 180) ∧ 
                (q.angles i + q.angles ((i + 3) % 4) ≤ 180) := by
  sorry


end NUMINAMATH_CALUDE_exists_angle_sum_with_adjacent_le_180_l3508_350881


namespace NUMINAMATH_CALUDE_fence_cost_l3508_350897

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 :=
by sorry

end NUMINAMATH_CALUDE_fence_cost_l3508_350897


namespace NUMINAMATH_CALUDE_monitor_horizontal_length_l3508_350839

/-- Given a rectangle with a 16:9 aspect ratio and a diagonal of 32 inches,
    prove that the horizontal length is (16 * 32) / sqrt(337) --/
theorem monitor_horizontal_length (h w d : ℝ) : 
  h / w = 9 / 16 → 
  h^2 + w^2 = d^2 → 
  d = 32 → 
  w = (16 * 32) / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_monitor_horizontal_length_l3508_350839


namespace NUMINAMATH_CALUDE_chickens_in_coop_l3508_350873

theorem chickens_in_coop (coop run free_range : ℕ) : 
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 →
  coop = 14 := by
sorry

end NUMINAMATH_CALUDE_chickens_in_coop_l3508_350873


namespace NUMINAMATH_CALUDE_function_composition_equality_l3508_350826

theorem function_composition_equality (f : ℝ → ℝ) :
  (∀ x, f (2 * x - 1) = x^2 - x) →
  (∀ x, f x = (1/4) * (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3508_350826


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l3508_350830

/-- Represents the possible outcomes of rolling an eight-sided die -/
inductive DieOutcome
  | One
  | Prime
  | Composite
  | Eight

/-- The probability distribution of the die outcomes -/
def dieProbability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One => 1/8
  | DieOutcome.Prime => 1/2
  | DieOutcome.Composite => 1/4
  | DieOutcome.Eight => 1/8

/-- The expected number of rolls per day -/
def expectedRollsPerDay : ℚ := 7/5

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- Theorem: The expected number of die rolls in a non-leap year is 511 -/
theorem expected_rolls_in_year :
  expectedRollsPerDay * daysInNonLeapYear = 511 := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l3508_350830


namespace NUMINAMATH_CALUDE_waysToSelectIs186_l3508_350861

/-- The number of ways to select 5 balls from a bag containing 4 red balls and 6 white balls,
    such that the total score is at least 7 points (where red balls score 2 points and white balls score 1 point). -/
def waysToSelect : ℕ :=
  Nat.choose 4 4 * Nat.choose 6 1 +
  Nat.choose 4 3 * Nat.choose 6 2 +
  Nat.choose 4 2 * Nat.choose 6 3

/-- The theorem stating that the number of ways to select the balls is 186. -/
theorem waysToSelectIs186 : waysToSelect = 186 := by
  sorry

end NUMINAMATH_CALUDE_waysToSelectIs186_l3508_350861


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3508_350837

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 + 3 * Complex.I → z = -1/2 + 5/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3508_350837


namespace NUMINAMATH_CALUDE_calculate_expressions_l3508_350809

theorem calculate_expressions : 
  (1 - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  ((-1/2 : ℝ) * (-2)^2 - (-1/8 : ℝ)^(1/3) + ((-1/2 : ℝ)^2)^(1/2) = -1) := by
  sorry

end NUMINAMATH_CALUDE_calculate_expressions_l3508_350809


namespace NUMINAMATH_CALUDE_first_covering_triangular_number_l3508_350845

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def covers_all_columns (n : ℕ) : Prop :=
  ∀ k : Fin 10, ∃ m : ℕ, m ≤ n ∧ triangular_number m % 10 = k

theorem first_covering_triangular_number :
  (covers_all_columns 29) ∧ (∀ k < 29, ¬ covers_all_columns k) :=
sorry

end NUMINAMATH_CALUDE_first_covering_triangular_number_l3508_350845


namespace NUMINAMATH_CALUDE_base_four_20314_equals_568_l3508_350842

def base_four_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_four_20314_equals_568 :
  base_four_to_decimal [4, 1, 3, 0, 2] = 568 := by
  sorry

end NUMINAMATH_CALUDE_base_four_20314_equals_568_l3508_350842


namespace NUMINAMATH_CALUDE_greatest_k_value_l3508_350860

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) → 
  k ≤ Real.sqrt 117 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l3508_350860


namespace NUMINAMATH_CALUDE_range_of_m_l3508_350836

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (m + 2)*x - 1 < (m + 2)*y - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3508_350836


namespace NUMINAMATH_CALUDE_max_t_value_l3508_350813

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  r ≤ 13 →
  t ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_max_t_value_l3508_350813


namespace NUMINAMATH_CALUDE_answer_key_combinations_l3508_350818

/-- Represents the number of possible answers for a true-false question -/
def true_false_options : ℕ := 2

/-- Represents the number of possible answers for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions in the quiz -/
def num_true_false : ℕ := 3

/-- Represents the number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 3

/-- Calculates the number of ways to arrange true-false answers where all answers cannot be the same -/
def true_false_combinations : ℕ := true_false_options ^ num_true_false - 2

/-- Calculates the number of ways to arrange multiple-choice answers -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- Theorem stating that the total number of ways to create an answer key is 384 -/
theorem answer_key_combinations : 
  true_false_combinations * multiple_choice_combinations = 384 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l3508_350818


namespace NUMINAMATH_CALUDE_quadratic_linear_system_solution_l3508_350811

theorem quadratic_linear_system_solution :
  ∀ x y : ℝ,
  (x^2 - 6*x + 8 = 0) ∧ (y + 2*x = 12) →
  ((x = 4 ∧ y = 4) ∨ (x = 2 ∧ y = 8)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_linear_system_solution_l3508_350811
