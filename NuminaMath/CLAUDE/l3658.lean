import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3658_365893

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from its right vertex to one of its asymptotes is b/2,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (a * b) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3658_365893


namespace NUMINAMATH_CALUDE_parabola_properties_l3658_365808

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (1, -3)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 9*x

theorem parabola_properties :
  (∀ x y, parabola_equation x y → parabola_equation x (-y)) ∧ 
  parabola_equation 0 0 ∧
  parabola_equation (circle_center.1) (circle_center.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3658_365808


namespace NUMINAMATH_CALUDE_bus_full_after_twelve_stops_l3658_365882

/-- The number of seats in the bus -/
def bus_seats : ℕ := 78

/-- The function representing the total number of passengers after n stops -/
def total_passengers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that 12 is the smallest positive integer that fills the bus -/
theorem bus_full_after_twelve_stops :
  (∀ k : ℕ, k > 0 → k < 12 → total_passengers k < bus_seats) ∧
  total_passengers 12 = bus_seats :=
sorry

end NUMINAMATH_CALUDE_bus_full_after_twelve_stops_l3658_365882


namespace NUMINAMATH_CALUDE_expression_simplification_l3658_365848

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -3) :
  (a^2 - 9) / (a^2 + 6*a + 9) / ((a - 3) / (a^2 + 3*a)) - (a - a^2) / (a - 1) = 2*a :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3658_365848


namespace NUMINAMATH_CALUDE_max_tasty_compote_weight_l3658_365895

theorem max_tasty_compote_weight 
  (fresh_apples : ℝ) 
  (dried_apples : ℝ) 
  (fresh_water_content : ℝ) 
  (dried_water_content : ℝ) 
  (max_water_content : ℝ) :
  fresh_apples = 4 →
  dried_apples = 1 →
  fresh_water_content = 0.9 →
  dried_water_content = 0.12 →
  max_water_content = 0.95 →
  ∃ (max_compote : ℝ),
    max_compote = 25.6 ∧
    ∀ (added_water : ℝ),
      (fresh_apples * fresh_water_content + 
       dried_apples * dried_water_content + 
       added_water) / 
      (fresh_apples + dried_apples + added_water) ≤ max_water_content →
      fresh_apples + dried_apples + added_water ≤ max_compote :=
by sorry

end NUMINAMATH_CALUDE_max_tasty_compote_weight_l3658_365895


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l3658_365820

theorem polynomial_product_sum (p q : ℚ) : 
  (∀ x, (4 * x^2 - 5 * x + p) * (6 * x^2 + q * x - 12) = 
   24 * x^4 - 62 * x^3 - 69 * x^2 + 94 * x - 36) → 
  p + q = 43 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l3658_365820


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15_to_100_l3658_365851

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceTerms (a : ℕ) (d : ℕ) (lastTerm : ℕ) : ℕ :=
  (lastTerm - a) / d + 1

/-- Theorem: The arithmetic sequence with first term 15, last term 100, and common difference 5 has 18 terms -/
theorem arithmetic_sequence_15_to_100 :
  arithmeticSequenceTerms 15 5 100 = 18 := by
  sorry

#eval arithmeticSequenceTerms 15 5 100

end NUMINAMATH_CALUDE_arithmetic_sequence_15_to_100_l3658_365851


namespace NUMINAMATH_CALUDE_division_reduction_l3658_365888

theorem division_reduction (x : ℝ) : 
  (63 / x = 63 - 42) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l3658_365888


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3658_365811

theorem basketball_free_throws (deshawn kayla annieka : ℕ) : 
  deshawn = 12 →
  annieka = 14 →
  annieka = kayla - 4 →
  (kayla - deshawn) / deshawn * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3658_365811


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_progressions_l3658_365883

/-- The sum of the first 40 terms of an arithmetic progression -/
def S (p : ℕ) : ℕ :=
  let a := p  -- first term
  let d := 2 * p + 2  -- common difference
  let n := 40  -- number of terms
  n * (2 * a + (n - 1) * d) / 2

/-- The sum of S_p for p from 1 to 10 -/
def total_sum : ℕ :=
  (Finset.range 10).sum (fun i => S (i + 1))

theorem sum_of_arithmetic_progressions : total_sum = 103600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_progressions_l3658_365883


namespace NUMINAMATH_CALUDE_banana_pear_difference_l3658_365843

/-- Represents a bowl of fruit with apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Properties of the fruit bowl -/
def is_valid_fruit_bowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas > bowl.pears ∧
  bowl.apples + bowl.pears + bowl.bananas = 19 ∧
  bowl.bananas = 9

theorem banana_pear_difference (bowl : FruitBowl) 
  (h : is_valid_fruit_bowl bowl) : 
  bowl.bananas - bowl.pears = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_difference_l3658_365843


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l3658_365830

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k = (Nat.gcd (15^4 - 9^4) (2^32)) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_of_fourth_powers_l3658_365830


namespace NUMINAMATH_CALUDE_dam_building_problem_l3658_365852

/-- Represents the work rate of beavers building a dam -/
def work_rate (beavers : ℕ) (hours : ℝ) : ℝ := beavers * hours

/-- The number of beavers in the second group -/
def second_group_beavers : ℕ := 12

theorem dam_building_problem :
  let first_group_beavers : ℕ := 20
  let first_group_hours : ℝ := 3
  let second_group_hours : ℝ := 5
  work_rate first_group_beavers first_group_hours = work_rate second_group_beavers second_group_hours :=
by
  sorry

end NUMINAMATH_CALUDE_dam_building_problem_l3658_365852


namespace NUMINAMATH_CALUDE_skew_lines_definition_l3658_365861

-- Define a type for lines in 3D space
def Line3D : Type := ℝ × ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line3D) : Prop :=
  ¬(parallel l1 l2) ∧ ¬(intersect l1 l2)

-- Theorem stating the definition of skew lines
theorem skew_lines_definition (l1 l2 : Line3D) :
  skew l1 l2 ↔ (¬(parallel l1 l2) ∧ ¬(intersect l1 l2)) := by sorry

end NUMINAMATH_CALUDE_skew_lines_definition_l3658_365861


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3658_365832

-- Part 1
theorem problem_1 : 2023^2 - 2022 * 2024 = 1 := by sorry

-- Part 2
theorem problem_2 (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -1) (h3 : m ≠ 0) :
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3658_365832


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3658_365879

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 7) ∧ 
   (x - 2)^2 + (y - 3)^2 = (10 - 2)^2 + (7 - 3)^2 ∧
   (y - 3) = m * (x - 2) ∧
   y = m * x + b) →
  m + b = 15 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3658_365879


namespace NUMINAMATH_CALUDE_knights_on_red_chairs_l3658_365823

/-- Represents the type of chair occupant -/
inductive Occupant
| Knight
| Liar

/-- Represents the color of a chair -/
inductive ChairColor
| Blue
| Red

/-- Represents the state of the room -/
structure RoomState where
  totalChairs : ℕ
  knights : ℕ
  liars : ℕ
  knightsOnRed : ℕ
  liarsOnBlue : ℕ

/-- The initial state of the room -/
def initialState : RoomState :=
  { totalChairs := 20
  , knights := 20 - (20 : ℕ) / 2  -- Arbitrary split between knights and liars
  , liars := (20 : ℕ) / 2
  , knightsOnRed := 0
  , liarsOnBlue := 0 }

/-- The state of the room after rearrangement -/
def finalState (initial : RoomState) : RoomState :=
  { totalChairs := initial.totalChairs
  , knights := initial.knights
  , liars := initial.liars
  , knightsOnRed := (initial.totalChairs : ℕ) / 2 - initial.liars
  , liarsOnBlue := (initial.totalChairs : ℕ) / 2 - (initial.knights - ((initial.totalChairs : ℕ) / 2 - initial.liars)) }

theorem knights_on_red_chairs (initial : RoomState) :
  (finalState initial).knightsOnRed = 5 := by
  sorry

end NUMINAMATH_CALUDE_knights_on_red_chairs_l3658_365823


namespace NUMINAMATH_CALUDE_tank_filling_time_l3658_365831

/-- Given a tank and three hoses X, Y, and Z, prove that they together fill the tank in 24/13 hours. -/
theorem tank_filling_time (T X Y Z : ℝ) (hxy : T = 2 * (X + Y)) (hxz : T = 3 * (X + Z)) (hyz : T = 4 * (Y + Z)) :
  T / (X + Y + Z) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3658_365831


namespace NUMINAMATH_CALUDE_sarahs_pastry_flour_l3658_365858

def rye_flour : ℕ := 5
def wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def total_flour : ℕ := 20

def pastry_flour : ℕ := total_flour - (rye_flour + wheat_bread_flour + chickpea_flour)

theorem sarahs_pastry_flour : pastry_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_pastry_flour_l3658_365858


namespace NUMINAMATH_CALUDE_gloria_purchase_l3658_365891

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions from the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.pencil + p.eraser = 45 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem gloria_purchase (p : StorePrices) : 
  store_conditions p → p.notebook + p.eraser = 85 := by
  sorry

end NUMINAMATH_CALUDE_gloria_purchase_l3658_365891


namespace NUMINAMATH_CALUDE_tan_x_equals_negative_seven_l3658_365892

theorem tan_x_equals_negative_seven (x : ℝ) 
  (h1 : Real.sin (x + π/4) = 3/5)
  (h2 : Real.sin (x - π/4) = 4/5) : 
  Real.tan x = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_equals_negative_seven_l3658_365892


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3658_365863

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, 5 * x^2 + n * x + 48 = (5 * x + a) * (x + b)) → 
  n ≤ 241 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3658_365863


namespace NUMINAMATH_CALUDE_line_through_point_and_intersection_l3658_365878

/-- The line passing through P(2, -3) and the intersection of two given lines -/
theorem line_through_point_and_intersection :
  let P : ℝ × ℝ := (2, -3)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 2 * y - 4
  let line2 : ℝ → ℝ → ℝ := λ x y => x - y + 5
  let result_line : ℝ → ℝ → ℝ := λ x y => 3.4 * x + 1.6 * y - 2
  -- The result line passes through P
  (result_line P.1 P.2 = 0) ∧
  -- The result line passes through the intersection point of line1 and line2
  (∃ x y : ℝ, line1 x y = 0 ∧ line2 x y = 0 ∧ result_line x y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_point_and_intersection_l3658_365878


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3658_365828

-- Define Rahul's and Deepak's ages
def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 10
def deepak_current_age : ℕ := 12

-- Define the ratio we want to prove
def target_ratio : Rat := 4 / 3

-- Theorem statement
theorem age_ratio_proof :
  (rahul_future_age - years_to_future : ℚ) / deepak_current_age = target_ratio := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3658_365828


namespace NUMINAMATH_CALUDE_midpoint_linear_combination_l3658_365821

/-- Given two points A and B in the plane, prove that for their midpoint C,
    a specific linear combination of C's coordinates equals -28. -/
theorem midpoint_linear_combination (A B : ℝ × ℝ) (h : A = (10, 15) ∧ B = (-2, 3)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = -28 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_linear_combination_l3658_365821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3658_365809

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term 
  (a : ℕ → ℝ) 
  (h_sum : a 1 + a 2 + a 3 = 6) 
  (h_5th : a 5 = 8) 
  (h_arith : arithmetic_sequence a) : 
  a 20 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3658_365809


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l3658_365877

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l3658_365877


namespace NUMINAMATH_CALUDE_sqrt_equation_root_l3658_365840

theorem sqrt_equation_root :
  ∃ x : ℝ, (Real.sqrt x + Real.sqrt (x + 2) = 12) ∧ (x = 5041 / 144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_root_l3658_365840


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3658_365872

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3658_365872


namespace NUMINAMATH_CALUDE_moles_of_H2O_formed_l3658_365868

-- Define the chemical reaction
structure ChemicalReaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio : ℕ → ℕ → ℕ

-- Define the problem setup
def reaction : ChemicalReaction := {
  reactant1 := "NaHCO3"
  reactant2 := "HC2H3O2"
  product1 := "NaC2H3O2"
  product2 := "CO2"
  product3 := "H2O"
  ratio := λ x y => min x y
}

def initial_moles_NaHCO3 : ℕ := 3
def initial_moles_HC2H3O2 : ℕ := 3

-- State the theorem
theorem moles_of_H2O_formed (r : ChemicalReaction) 
  (h1 : r = reaction) 
  (h2 : initial_moles_NaHCO3 = 3) 
  (h3 : initial_moles_HC2H3O2 = 3) : 
  r.ratio initial_moles_NaHCO3 initial_moles_HC2H3O2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_H2O_formed_l3658_365868


namespace NUMINAMATH_CALUDE_total_balls_is_six_l3658_365884

/-- The number of balls in each box -/
def balls_per_box : ℕ := 3

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The total number of balls -/
def total_balls : ℕ := balls_per_box * num_boxes

theorem total_balls_is_six : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_is_six_l3658_365884


namespace NUMINAMATH_CALUDE_prime_sum_divisibility_l3658_365887

theorem prime_sum_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisibility_l3658_365887


namespace NUMINAMATH_CALUDE_impossibleIdenticalLongNumbers_l3658_365804

/-- Represents a long number formed by concatenating integers -/
def LongNumber := List Nat

/-- Checks if a number is in the valid range [0, 999] -/
def isValidNumber (n : Nat) : Prop := n ≤ 999

/-- Splits a list of numbers into two groups -/
def split (numbers : List Nat) : Prop := ∃ (group1 group2 : List Nat), 
  (group1 ++ group2).Perm numbers ∧ group1 ≠ [] ∧ group2 ≠ []

theorem impossibleIdenticalLongNumbers : 
  ¬∃ (numbers : List Nat), 
    (∀ n ∈ numbers, isValidNumber n) ∧ 
    (∀ n, isValidNumber n → n ∈ numbers) ∧
    (∃ (group1 group2 : LongNumber), 
      split numbers ∧ 
      group1.toString = group2.toString) := by
  sorry

end NUMINAMATH_CALUDE_impossibleIdenticalLongNumbers_l3658_365804


namespace NUMINAMATH_CALUDE_binomial_probability_equals_eight_twentyseven_l3658_365860

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (dist : BinomialDistribution) (k : ℕ) : ℝ :=
  (dist.n.choose k) * (dist.p ^ k) * ((1 - dist.p) ^ (dist.n - k))

theorem binomial_probability_equals_eight_twentyseven :
  let ξ : BinomialDistribution := ⟨4, 1/3, by norm_num, by norm_num⟩
  binomialPMF ξ 2 = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_equals_eight_twentyseven_l3658_365860


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_2549_l3658_365814

theorem sqrt_product_plus_one_equals_2549 :
  Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_2549_l3658_365814


namespace NUMINAMATH_CALUDE_work_duration_l3658_365847

/-- Given two workers p and q, where p can complete a job in 15 days and q in 20 days,
    this theorem proves that if 0.5333333333333333 of the job remains after they work
    together for d days, then d must equal 4. -/
theorem work_duration (p q d : ℝ) : 
  p = 1 / 15 →
  q = 1 / 20 →
  1 - (p + q) * d = 0.5333333333333333 →
  d = 4 := by
  sorry


end NUMINAMATH_CALUDE_work_duration_l3658_365847


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l3658_365862

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 8

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 3

/-- The probability that a specific ball is chosen in any swap -/
def prob_chosen : ℚ := 1 / 4

/-- The probability that a ball is not chosen in a single swap -/
def prob_not_chosen : ℚ := 1 - prob_chosen

/-- The probability that a ball is in its original position after all transpositions -/
def prob_original_position : ℚ := prob_not_chosen ^ num_transpositions + 
  num_transpositions * prob_chosen ^ 2 * prob_not_chosen

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_balls_in_original_position :
  expected_original_positions = 9 / 2 := by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l3658_365862


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3658_365867

theorem trigonometric_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  Real.cos (α - β) = 4/5 ∧ Real.cos α = 3/5 ∧ Real.cos β = 24/25 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3658_365867


namespace NUMINAMATH_CALUDE_normal_block_volume_l3658_365801

/-- The volume of a normal block of cheese -/
def normal_volume : ℝ := sorry

/-- The volume of a large block of cheese -/
def large_volume : ℝ := 36

/-- The relationship between large and normal block volumes -/
axiom volume_relationship : large_volume = 12 * normal_volume

theorem normal_block_volume : normal_volume = 3 := by sorry

end NUMINAMATH_CALUDE_normal_block_volume_l3658_365801


namespace NUMINAMATH_CALUDE_recipe_total_cups_l3658_365845

theorem recipe_total_cups (butter baking_soda flour sugar : ℚ)
  (ratio : butter = 1 ∧ baking_soda = 2 ∧ flour = 5 ∧ sugar = 3)
  (flour_cups : flour * 3 = 15) :
  butter * 3 + baking_soda * 3 + 15 + sugar * 3 = 33 := by
sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l3658_365845


namespace NUMINAMATH_CALUDE_insert_zeros_is_perfect_cube_l3658_365853

/-- Given a non-negative integer n, the function calculates the number
    obtained by inserting n zeros between each digit of 1331. -/
def insert_zeros (n : ℕ) : ℕ :=
  10^(3*n+3) + 3 * 10^(2*n+2) + 3 * 10^(n+1) + 1

/-- Theorem stating that for any non-negative integer n,
    the number obtained by inserting n zeros between each digit of 1331
    is equal to (10^(n+1) + 1)^3. -/
theorem insert_zeros_is_perfect_cube (n : ℕ) :
  insert_zeros n = (10^(n+1) + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_insert_zeros_is_perfect_cube_l3658_365853


namespace NUMINAMATH_CALUDE_triangle_inequality_l3658_365841

theorem triangle_inequality (a b c m : ℝ) (γ : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < m) (h5 : 0 < γ) (h6 : γ < π) : 
  a + b + m ≤ ((2 + Real.cos (γ / 2)) / (2 * Real.sin (γ / 2))) * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3658_365841


namespace NUMINAMATH_CALUDE_fishing_problem_l3658_365880

/-- Represents the number of fish caught by each person --/
structure FishCaught where
  jason : ℕ
  ryan : ℕ
  jeffery : ℕ

/-- The fishing problem statement --/
theorem fishing_problem (f : FishCaught) 
  (h1 : f.jason + f.ryan + f.jeffery = 100)
  (h2 : f.jeffery = 2 * f.ryan)
  (h3 : f.jeffery = 60) : 
  f.ryan = 30 := by
sorry


end NUMINAMATH_CALUDE_fishing_problem_l3658_365880


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l3658_365856

/-- Represents a parabola in the form y^2 = 4px --/
structure Parabola where
  p : ℝ

/-- The directrix of a parabola --/
def directrix (para : Parabola) : ℝ := -para.p

theorem parabola_directrix_equation :
  let para : Parabola := ⟨1⟩
  directrix para = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l3658_365856


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l3658_365870

/-- Given a geometric sequence with first term 3 and second term 9/2,
    prove that the eighth term is 6561/128 -/
theorem eighth_term_of_geometric_sequence (a : ℕ → ℚ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 9/2)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 2 / a 1)) :
  a 8 = 6561/128 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l3658_365870


namespace NUMINAMATH_CALUDE_fraction_product_l3658_365826

theorem fraction_product : (4/5 : ℚ) * (5/6 : ℚ) * (6/7 : ℚ) * (7/8 : ℚ) * (8/9 : ℚ) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3658_365826


namespace NUMINAMATH_CALUDE_polar_eq_of_cartesian_line_l3658_365875

/-- The polar coordinate equation ρ cos θ = 1 represents the line x = 1 in Cartesian coordinates -/
theorem polar_eq_of_cartesian_line (ρ θ : ℝ) :
  (ρ * Real.cos θ = 1) ↔ (ρ * Real.cos θ = 1) :=
by sorry

end NUMINAMATH_CALUDE_polar_eq_of_cartesian_line_l3658_365875


namespace NUMINAMATH_CALUDE_midpoint_value_l3658_365874

/-- Given two distinct points (m, n) and (p, q) on the curve x^2 - 5xy + 2y^2 + 7x - 6y + 3 = 0,
    where (m + 2, n + k) is the midpoint of the line segment connecting (m, n) and (p, q),
    and the line passing through (m, n) and (p, q) has the equation x - 5y + 1 = 0,
    prove that k = 2/5. -/
theorem midpoint_value (m n p q k : ℝ) : 
  (m ≠ p ∨ n ≠ q) →
  m^2 - 5*m*n + 2*n^2 + 7*m - 6*n + 3 = 0 →
  p^2 - 5*p*q + 2*q^2 + 7*p - 6*q + 3 = 0 →
  m + 2 = (m + p) / 2 →
  n + k = (n + q) / 2 →
  m - 5*n + 1 = 0 →
  p - 5*q + 1 = 0 →
  k = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_value_l3658_365874


namespace NUMINAMATH_CALUDE_tissue_packs_per_box_l3658_365839

/-- Proves that the number of packs in each box is 20 given the specified conditions -/
theorem tissue_packs_per_box :
  ∀ (total_boxes : ℕ) 
    (tissues_per_pack : ℕ) 
    (cost_per_tissue : ℚ) 
    (total_cost : ℚ),
  total_boxes = 10 →
  tissues_per_pack = 100 →
  cost_per_tissue = 5 / 100 →
  total_cost = 1000 →
  (total_cost / total_boxes) / (tissues_per_pack * cost_per_tissue) = 20 := by
sorry

end NUMINAMATH_CALUDE_tissue_packs_per_box_l3658_365839


namespace NUMINAMATH_CALUDE_minimum_value_implies_k_l3658_365825

/-- Given that k is a positive constant and the minimum value of the function
    y = x^2 + k/x (where x > 0) is 3, prove that k = 2. -/
theorem minimum_value_implies_k (k : ℝ) (h1 : k > 0) :
  (∀ x > 0, x^2 + k/x ≥ 3) ∧ (∃ x > 0, x^2 + k/x = 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_k_l3658_365825


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3658_365818

theorem quadratic_equation_rewrite :
  ∀ x : ℝ, (-5 * x^2 = 2 * x + 10) ↔ (x^2 + (2/5) * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3658_365818


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3658_365876

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 4

def ring_arrangements (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k * Nat.choose (k + number_of_fingers) number_of_fingers

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange = 31752000 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3658_365876


namespace NUMINAMATH_CALUDE_right_triangle_area_l3658_365802

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let angle_30 : ℝ := 30 * π / 180
  let angle_60 : ℝ := 60 * π / 180
  let angle_90 : ℝ := 90 * π / 180
  h = 4 →
  (1/2) * (h * (2 * h / Real.sqrt 3)) * (h * Real.sqrt 3) = (16 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3658_365802


namespace NUMINAMATH_CALUDE_probability_sum_5_l3658_365834

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (fun p => p.1 < p.2)

def sum_5_pairs : Finset (ℕ × ℕ) := valid_pairs.filter (fun p => p.1 + p.2 = 5)

theorem probability_sum_5 : 
  (sum_5_pairs.card : ℚ) / valid_pairs.card = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_sum_5_l3658_365834


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3658_365815

theorem least_positive_integer_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (528 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 → (528 + m) % 5 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l3658_365815


namespace NUMINAMATH_CALUDE_sports_meeting_score_l3658_365869

/-- Represents the score for a single placement --/
inductive Placement
| first
| second
| third

/-- Calculates the score for a given placement --/
def score (p : Placement) : Nat :=
  match p with
  | .first => 5
  | .second => 3
  | .third => 1

/-- Represents the placements of a class --/
structure ClassPlacements where
  first : Nat
  second : Nat
  third : Nat

/-- Calculates the total score for a class given its placements --/
def totalScore (cp : ClassPlacements) : Nat :=
  cp.first * score Placement.first +
  cp.second * score Placement.second +
  cp.third * score Placement.third

/-- Calculates the total number of placements for a class --/
def totalPlacements (cp : ClassPlacements) : Nat :=
  cp.first + cp.second + cp.third

theorem sports_meeting_score (class1 class2 : ClassPlacements) :
  totalPlacements class1 = 2 →
  totalPlacements class2 = 4 →
  totalScore class1 = totalScore class2 →
  totalScore class1 + totalScore class2 + 7 = 27 :=
by sorry

end NUMINAMATH_CALUDE_sports_meeting_score_l3658_365869


namespace NUMINAMATH_CALUDE_total_price_increase_l3658_365890

-- Define the sequence of price increases
def price_increases : List Real := [0.375, 0.31, 0.427, 0.523, 0.272]

-- Function to calculate the total price increase factor
def total_increase_factor (increases : List Real) : Real :=
  List.foldl (fun acc x => acc * (1 + x)) 1 increases

-- Theorem stating the total equivalent percent increase
theorem total_price_increase : 
  ∀ ε > 0, 
  |total_increase_factor price_increases - 1 - 3.9799| < ε := by
  sorry

end NUMINAMATH_CALUDE_total_price_increase_l3658_365890


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3658_365846

-- Define the sets M and N
def M : Set ℝ := {x | ∃ t : ℝ, x = 2^t}
def N : Set ℝ := {x | ∃ t : ℝ, x = Real.sin t}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3658_365846


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3658_365894

theorem unique_solution_equation (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3658_365894


namespace NUMINAMATH_CALUDE_care_package_weight_l3658_365816

def final_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_chocolate := initial_weight * 1.4
  let weight_after_snacks := weight_after_chocolate + 0.6 - 0.35 + 0.85
  let weight_after_cookies := weight_after_snacks * 1.6
  let weight_after_brownie_removal := weight_after_cookies - 0.45
  5 * initial_weight

theorem care_package_weight :
  let initial_weight := 1.25 + 0.75 + 1.5
  final_weight initial_weight = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l3658_365816


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3658_365881

-- Problem 1
theorem factorization_problem_1 (a b x : ℝ) :
  x^2 * (a - b) + 4 * (b - a) = (a - b) * (x + 2) * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  -a^3 + 6 * a^2 * b - 9 * a * b^2 = -a * (a - 3 * b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3658_365881


namespace NUMINAMATH_CALUDE_bacteria_growth_l3658_365873

theorem bacteria_growth (n : ℕ) : 
  (∀ t : ℕ, t ≤ 10 → n * (4 ^ t) = n * 4 ^ t) →
  n * 4 ^ 10 = 1048576 ↔ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l3658_365873


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_l3658_365803

theorem sqrt_12_minus_sqrt_3 : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_l3658_365803


namespace NUMINAMATH_CALUDE_basketball_team_score_l3658_365805

theorem basketball_team_score :
  ∀ (tobee jay sean remy alex : ℕ),
  tobee = 4 →
  jay = 2 * tobee + 6 →
  sean = jay / 2 →
  remy = tobee + jay - 3 →
  alex = sean + remy + 4 →
  tobee + jay + sean + remy + alex = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_score_l3658_365805


namespace NUMINAMATH_CALUDE_expression_value_l3658_365886

theorem expression_value : (3^2 - 5*3 + 6) / (3 - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3658_365886


namespace NUMINAMATH_CALUDE_infinitely_many_special_n_l3658_365855

theorem infinitely_many_special_n : ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
  (∃ m : ℕ, n * m = 2^(2^n + 1) + 1) ∧ 
  (∀ m : ℕ, n * m ≠ 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_n_l3658_365855


namespace NUMINAMATH_CALUDE_distance_blown_westward_is_200km_l3658_365857

/-- Represents the journey of a ship -/
structure ShipJourney where
  speed : ℝ
  travelTime : ℝ
  totalDistance : ℝ
  finalPosition : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  journey.speed * journey.travelTime - journey.finalPosition

/-- Theorem stating the distance blown westward is 200 km -/
theorem distance_blown_westward_is_200km (journey : ShipJourney) 
  (h1 : journey.speed = 30)
  (h2 : journey.travelTime = 20)
  (h3 : journey.speed * journey.travelTime = journey.totalDistance / 2)
  (h4 : journey.finalPosition = journey.totalDistance / 3) :
  distanceBlownWestward journey = 200 := by
  sorry

#check distance_blown_westward_is_200km

end NUMINAMATH_CALUDE_distance_blown_westward_is_200km_l3658_365857


namespace NUMINAMATH_CALUDE_chocolate_comparison_l3658_365897

theorem chocolate_comparison 
  (robert_chocolates : ℕ)
  (robert_price : ℚ)
  (nickel_chocolates : ℕ)
  (nickel_discount : ℚ)
  (h1 : robert_chocolates = 7)
  (h2 : robert_price = 2)
  (h3 : nickel_chocolates = 5)
  (h4 : nickel_discount = 1.5)
  (h5 : robert_chocolates * robert_price = nickel_chocolates * (robert_price - nickel_discount)) :
  ∃ (n : ℕ), (robert_price * robert_chocolates) / (robert_price - nickel_discount) - robert_chocolates = n ∧ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_comparison_l3658_365897


namespace NUMINAMATH_CALUDE_fern_leaves_count_l3658_365819

/-- The number of leaves on all ferns -/
def total_leaves (num_ferns : ℕ) (fronds_per_fern : ℕ) (leaves_per_frond : ℕ) : ℕ :=
  num_ferns * fronds_per_fern * leaves_per_frond

/-- Theorem stating the total number of leaves on all ferns -/
theorem fern_leaves_count :
  total_leaves 6 7 30 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fern_leaves_count_l3658_365819


namespace NUMINAMATH_CALUDE_sixth_term_value_l3658_365806

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l3658_365806


namespace NUMINAMATH_CALUDE_black_pens_count_l3658_365859

theorem black_pens_count (total_pens blue_pens : ℕ) 
  (h1 : total_pens = 8)
  (h2 : blue_pens = 4) :
  total_pens - blue_pens = 4 := by
  sorry

end NUMINAMATH_CALUDE_black_pens_count_l3658_365859


namespace NUMINAMATH_CALUDE_expected_value_of_new_balls_l3658_365810

/-- Represents the outcome of drawing balls in a ping pong match -/
inductive BallDraw
  | zero
  | one
  | two

/-- The probability mass function for the number of new balls drawn -/
def prob_new_balls (draw : BallDraw) : ℚ :=
  match draw with
  | BallDraw.zero => 37/100
  | BallDraw.one  => 54/100
  | BallDraw.two  => 9/100

/-- The number of new balls for each outcome -/
def num_new_balls (draw : BallDraw) : ℕ :=
  match draw with
  | BallDraw.zero => 0
  | BallDraw.one  => 1
  | BallDraw.two  => 2

/-- The expected value of new balls in the second draw -/
def expected_value : ℚ :=
  (prob_new_balls BallDraw.zero * num_new_balls BallDraw.zero) +
  (prob_new_balls BallDraw.one  * num_new_balls BallDraw.one)  +
  (prob_new_balls BallDraw.two  * num_new_balls BallDraw.two)

theorem expected_value_of_new_balls :
  expected_value = 18/25 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_new_balls_l3658_365810


namespace NUMINAMATH_CALUDE_f_values_l3658_365835

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_f_values_l3658_365835


namespace NUMINAMATH_CALUDE_factorization_equality_l3658_365836

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3658_365836


namespace NUMINAMATH_CALUDE_compute_expression_l3658_365864

theorem compute_expression : 20 * (180 / 3 + 40 / 5 + 16 / 32 + 2) = 1410 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3658_365864


namespace NUMINAMATH_CALUDE_zero_subset_integers_negation_squared_positive_l3658_365865

-- Define the set containing only 0
def zero_set : Set ℤ := {0}

-- Statement 1: {0} is a subset of ℤ
theorem zero_subset_integers : zero_set ⊆ Set.univ := by sorry

-- Statement 2: Negation of "for all x in ℤ, x² > 0" is "there exists x in ℤ such that x² ≤ 0"
theorem negation_squared_positive :
  (¬ ∀ x : ℤ, x^2 > 0) ↔ (∃ x : ℤ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_zero_subset_integers_negation_squared_positive_l3658_365865


namespace NUMINAMATH_CALUDE_west_movement_negative_l3658_365817

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (d : Direction) (distance : ℝ) : ℝ :=
  match d with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_movement_negative (distance : ℝ) :
  movement Direction.East distance = distance →
  movement Direction.West distance = -distance :=
by
  sorry

end NUMINAMATH_CALUDE_west_movement_negative_l3658_365817


namespace NUMINAMATH_CALUDE_rooster_to_hen_ratio_l3658_365889

/-- Given a chicken farm with roosters and hens, prove the ratio of roosters to hens. -/
theorem rooster_to_hen_ratio 
  (total_chickens : ℕ) 
  (roosters : ℕ) 
  (h_total : total_chickens = 9000)
  (h_roosters : roosters = 6000) : 
  (roosters : ℚ) / (total_chickens - roosters) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rooster_to_hen_ratio_l3658_365889


namespace NUMINAMATH_CALUDE_grasshopper_theorem_l3658_365898

/-- Represents the order of grasshoppers -/
inductive GrasshopperOrder
  | Even
  | Odd

/-- Represents a single jump of a grasshopper -/
def jump (order : GrasshopperOrder) : GrasshopperOrder :=
  match order with
  | GrasshopperOrder.Even => GrasshopperOrder.Odd
  | GrasshopperOrder.Odd => GrasshopperOrder.Even

/-- Represents multiple jumps of grasshoppers -/
def multipleJumps (initialOrder : GrasshopperOrder) (n : Nat) : GrasshopperOrder :=
  match n with
  | 0 => initialOrder
  | Nat.succ m => jump (multipleJumps initialOrder m)

theorem grasshopper_theorem :
  multipleJumps GrasshopperOrder.Even 1999 = GrasshopperOrder.Odd :=
by sorry

end NUMINAMATH_CALUDE_grasshopper_theorem_l3658_365898


namespace NUMINAMATH_CALUDE_other_candidate_votes_l3658_365813

theorem other_candidate_votes
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (winning_candidate_percentage : ℚ)
  (h_total : total_votes = 7500)
  (h_invalid : invalid_percentage = 1/5)
  (h_winning : winning_candidate_percentage = 11/20) :
  ⌊(1 - invalid_percentage) * (1 - winning_candidate_percentage) * total_votes⌋ = 2700 :=
sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l3658_365813


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l3658_365842

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 20 → lost_shoes = 9 → max_pairs = 11 →
  max_pairs = initial_pairs - lost_shoes ∧ 
  max_pairs * 2 + lost_shoes ≤ initial_pairs * 2 :=
by sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l3658_365842


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l3658_365885

/-- Proves that for x = (3 + √5)^20, n = ⌊x⌋, and f = x - n, x(1 - f) = 1 -/
theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 5) ^ 20
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l3658_365885


namespace NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l3658_365827

/-- A function to check if a triple of integers forms a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of numbers -/
def setA : List ℤ := [5, 12, 13]
def setB : List ℤ := [7, 9, 11]
def setC : List ℤ := [6, 9, 12]
def setD : List ℚ := [3/10, 4/10, 5/10]

/-- Theorem stating that only setA is a Pythagorean triple -/
theorem only_setA_is_pythagorean_triple :
  (isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (¬ isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (∀ (a b c : ℚ), a ∈ setD → b ∈ setD → c ∈ setD → ¬ isPythagoreanTriple a.num b.num c.num) :=
by sorry


end NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l3658_365827


namespace NUMINAMATH_CALUDE_rip3_properties_l3658_365844

-- Define the basic concepts
def Cell : Type := sorry
def RIP3 : Type := sorry
def Gene : Type := sorry

-- Define the properties and relationships
def can_convert_apoptosis_to_necrosis (r : RIP3) : Prop := sorry
def controls_synthesis_of (g : Gene) (r : RIP3) : Prop := sorry
def exists_in_human_body (r : RIP3) : Prop := sorry
def can_regulate_cell_death_mode (r : RIP3) : Prop := sorry
def has_gene (c : Cell) (g : Gene) : Prop := sorry

-- State the theorem
theorem rip3_properties :
  ∃ (r : RIP3) (g : Gene),
    exists_in_human_body r ∧
    can_convert_apoptosis_to_necrosis r ∧
    controls_synthesis_of g r ∧
    can_regulate_cell_death_mode r ∧
    ∀ (c : Cell), has_gene c g :=
sorry

-- Note: This theorem encapsulates the main points about RIP3 from the problem statement,
-- without making claims about the correctness or incorrectness of the given statements.

end NUMINAMATH_CALUDE_rip3_properties_l3658_365844


namespace NUMINAMATH_CALUDE_members_playing_both_sports_l3658_365854

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def both_sports (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given club -/
theorem members_playing_both_sports (club : SportsClub)
  (h1 : club.total = 30)
  (h2 : club.badminton = 16)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  both_sports club = 7 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_both_sports_l3658_365854


namespace NUMINAMATH_CALUDE_earning_goal_proof_l3658_365807

/-- Calculates the total earnings for a salesperson given fixed earnings, commission rate, and sales amount. -/
def totalEarnings (fixedEarnings : ℝ) (commissionRate : ℝ) (sales : ℝ) : ℝ :=
  fixedEarnings + commissionRate * sales

/-- Proves that the earning goal is $500 given the specified conditions. -/
theorem earning_goal_proof :
  let fixedEarnings : ℝ := 190
  let commissionRate : ℝ := 0.04
  let minSales : ℝ := 7750
  totalEarnings fixedEarnings commissionRate minSales = 500 := by
  sorry

#eval totalEarnings 190 0.04 7750

end NUMINAMATH_CALUDE_earning_goal_proof_l3658_365807


namespace NUMINAMATH_CALUDE_long_tennis_players_l3658_365899

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 35 →
  football = 26 →
  both = 17 →
  neither = 6 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ 
    long_tennis = total - (football - both) - neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l3658_365899


namespace NUMINAMATH_CALUDE_polynomial_sum_l3658_365800

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3658_365800


namespace NUMINAMATH_CALUDE_journalism_club_arrangement_l3658_365850

/-- The number of students in the arrangement -/
def num_students : ℕ := 5

/-- The number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- The number of possible positions for the teacher pair -/
def teacher_pair_positions : ℕ := num_students - 1

/-- The total number of arrangements -/
def total_arrangements : ℕ := num_students.factorial * (teacher_pair_positions * num_teachers.factorial)

/-- Theorem stating that the total number of arrangements is 960 -/
theorem journalism_club_arrangement :
  total_arrangements = 960 := by sorry

end NUMINAMATH_CALUDE_journalism_club_arrangement_l3658_365850


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3658_365829

/-- Given a circle with equation x^2 - 8x + y^2 - 4y = -4, prove its center and radius -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, 2) ∧
    radius = 4 ∧
    ∀ (x y : ℝ), x^2 - 8*x + y^2 - 4*y = -4 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3658_365829


namespace NUMINAMATH_CALUDE_silk_order_total_l3658_365866

/-- The number of yards of green silk dyed by the factory -/
def green_silk : ℕ := 61921

/-- The number of yards of pink silk dyed by the factory -/
def pink_silk : ℕ := 49500

/-- The total number of yards of silk dyed by the factory -/
def total_silk : ℕ := green_silk + pink_silk

theorem silk_order_total :
  total_silk = 111421 :=
by sorry

end NUMINAMATH_CALUDE_silk_order_total_l3658_365866


namespace NUMINAMATH_CALUDE_optimal_warehouse_location_l3658_365824

/-- The optimal warehouse location problem -/
theorem optimal_warehouse_location 
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) (h₁ : ∀ x > 0, y₁ x = k₁ / x) 
  (h₂ : ∀ x > 0, y₂ x = k₂ * x) (h₃ : k₁ > 0) (h₄ : k₂ > 0)
  (h₅ : y₁ 10 = 4) (h₆ : y₂ 10 = 16) :
  ∃ x₀ > 0, ∀ x > 0, y₁ x + y₂ x ≥ y₁ x₀ + y₂ x₀ ∧ x₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_optimal_warehouse_location_l3658_365824


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3658_365838

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3658_365838


namespace NUMINAMATH_CALUDE_puzzle_arrangement_count_l3658_365896

/-- The number of letters in the word "puzzle" -/
def n : ℕ := 6

/-- The number of times the letter "z" appears in "puzzle" -/
def z_count : ℕ := 2

/-- The number of distinct arrangements of the letters in "puzzle" -/
def puzzle_arrangements : ℕ := n.factorial / z_count.factorial

theorem puzzle_arrangement_count : puzzle_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_arrangement_count_l3658_365896


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l3658_365822

theorem cube_root_three_equation : 
  (1 : ℝ) / (2 - Real.rpow 3 (1/3)) = (2 + Real.rpow 3 (1/3)) * (2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l3658_365822


namespace NUMINAMATH_CALUDE_halfway_fraction_l3658_365849

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3658_365849


namespace NUMINAMATH_CALUDE_square_equation_solve_l3658_365837

theorem square_equation_solve (x y : ℝ) (h1 : x^2 = y + 4) (h2 : x = 7) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solve_l3658_365837


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3658_365812

theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r > 0 → 
  ratio > 0 → 
  let width := 2 * r
  let length := ratio * width
  let area := length * width
  r = 8 ∧ ratio = 3 → area = 768 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3658_365812


namespace NUMINAMATH_CALUDE_mollys_current_age_l3658_365833

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old after 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : sandy_future_age ages) : 
  ages.molly = 27 := by
  sorry

end NUMINAMATH_CALUDE_mollys_current_age_l3658_365833


namespace NUMINAMATH_CALUDE_vehicle_license_count_l3658_365871

/-- The number of possible letters for a license -/
def num_letters : ℕ := 3

/-- The number of possible digits for each position in a license -/
def num_digits : ℕ := 10

/-- The number of digit positions in a license -/
def num_digit_positions : ℕ := 6

/-- The total number of possible vehicle licenses -/
def total_licenses : ℕ := num_letters * (num_digits ^ num_digit_positions)

theorem vehicle_license_count :
  total_licenses = 3000000 := by sorry

end NUMINAMATH_CALUDE_vehicle_license_count_l3658_365871
