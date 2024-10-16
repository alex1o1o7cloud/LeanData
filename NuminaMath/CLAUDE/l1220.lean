import Mathlib

namespace NUMINAMATH_CALUDE_casper_enter_exit_ways_l1220_122043

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways Casper can enter and exit the castle -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Casper can enter and exit is 56 -/
theorem casper_enter_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_casper_enter_exit_ways_l1220_122043


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l1220_122088

theorem gwens_birthday_money (mom_gift dad_gift : ℕ) 
  (h1 : mom_gift = 8)
  (h2 : dad_gift = 5) :
  mom_gift - dad_gift = 3 := by
  sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l1220_122088


namespace NUMINAMATH_CALUDE_speaker_sale_profit_l1220_122021

theorem speaker_sale_profit (selling_price : ℝ) 
  (profit_percentage : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1.44 →
  profit_percentage = 0.2 →
  loss_percentage = 0.1 →
  let cost_price_1 := selling_price / (1 + profit_percentage)
  let cost_price_2 := selling_price / (1 - loss_percentage)
  let total_cost := cost_price_1 + cost_price_2
  let total_revenue := 2 * selling_price
  total_revenue - total_cost = 0.08 := by
sorry

end NUMINAMATH_CALUDE_speaker_sale_profit_l1220_122021


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1220_122097

theorem partial_fraction_decomposition (x : ℝ) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (x^2 - 10*x + 16) / ((x - 2) * (x - 3) * (x - 4)) =
  2 / (x - 2) + 5 / (x - 3) + 0 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1220_122097


namespace NUMINAMATH_CALUDE_range_of_m_l1220_122085

/-- Proposition p: The equation x²/(2m) - y²/(m-1) = 1 represents an ellipse with foci on the y-axis -/
def prop_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

/-- Proposition q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is in the interval (1,2) -/
def prop_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1220_122085


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l1220_122096

def repeating_decimal : ℚ := 33 / 99999

theorem repeating_decimal_value : 
  (10^5 - 10^3 : ℚ) * repeating_decimal = 32.67 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l1220_122096


namespace NUMINAMATH_CALUDE_essay_section_length_l1220_122041

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_length : ℕ) 
  (body_sections : ℕ) 
  (total_length : ℕ) :
  intro_length = 450 →
  conclusion_length = 3 * intro_length →
  body_sections = 4 →
  total_length = 5000 →
  (total_length - (intro_length + conclusion_length)) / body_sections = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_essay_section_length_l1220_122041


namespace NUMINAMATH_CALUDE_function_property_l1220_122094

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1220_122094


namespace NUMINAMATH_CALUDE_four_hundred_billion_scientific_notation_l1220_122001

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Definition of a billion -/
def billion : ℕ := 10^9

/-- Theorem: 400 billion in scientific notation is 4 × 10^11 -/
theorem four_hundred_billion_scientific_notation :
  ∃ (sn : ScientificNotation), (400 * billion : ℝ) = sn.coefficient * (10 : ℝ)^sn.exponent ∧ 
  sn.coefficient = 4 ∧ sn.exponent = 11 := by
  sorry

end NUMINAMATH_CALUDE_four_hundred_billion_scientific_notation_l1220_122001


namespace NUMINAMATH_CALUDE_min_a_value_l1220_122076

open Real

-- Define the function f(x) = x/ln(x) - 1/(4x)
noncomputable def f (x : ℝ) : ℝ := x / log x - 1 / (4 * x)

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x ∈ Set.Icc (exp 1) (exp 2), x / log x ≤ 1/4 + a*x) ↔ 
  a ≥ 1/2 - 1/(4 * (exp 2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l1220_122076


namespace NUMINAMATH_CALUDE_complex_sum_example_l1220_122058

theorem complex_sum_example : (2 : ℂ) + 5*I + (3 : ℂ) - 7*I = 5 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_example_l1220_122058


namespace NUMINAMATH_CALUDE_fraction_problem_l1220_122050

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1220_122050


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1220_122083

theorem polynomial_remainder (x : ℝ) : 
  (5 * x^3 - 9 * x^2 + 3 * x + 17) % (x - 2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1220_122083


namespace NUMINAMATH_CALUDE_cubic_roots_product_l1220_122092

theorem cubic_roots_product (r s t : ℝ) : 
  (r^3 - 20*r^2 + 18*r - 7 = 0) ∧ 
  (s^3 - 20*s^2 + 18*s - 7 = 0) ∧ 
  (t^3 - 20*t^2 + 18*t - 7 = 0) →
  (1 + r) * (1 + s) * (1 + t) = 46 := by
sorry


end NUMINAMATH_CALUDE_cubic_roots_product_l1220_122092


namespace NUMINAMATH_CALUDE_todd_ate_eight_cupcakes_l1220_122099

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proof that Todd ate 8 cupcakes -/
theorem todd_ate_eight_cupcakes :
  cupcakes_eaten 18 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_todd_ate_eight_cupcakes_l1220_122099


namespace NUMINAMATH_CALUDE_no_valid_coloring_l1220_122045

/-- Represents a coloring of a rectangular grid --/
def GridColoring (m n : ℕ) := Fin m → Fin n → Bool

/-- Checks if the number of white cells equals the number of black cells --/
def equalColors (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 1 else 0)) =
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 0 else 1))

/-- Checks if more than 3/4 of cells in each row are of the same color --/
def rowColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ i, (4 * (Finset.univ.sum fun j => if coloring i j then 1 else 0) > 3 * n) ∨
       (4 * (Finset.univ.sum fun j => if coloring i j then 0 else 1) > 3 * n)

/-- Checks if more than 3/4 of cells in each column are of the same color --/
def columnColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ j, (4 * (Finset.univ.sum fun i => if coloring i j then 1 else 0) > 3 * m) ∨
       (4 * (Finset.univ.sum fun i => if coloring i j then 0 else 1) > 3 * m)

/-- The main theorem stating that no valid coloring exists --/
theorem no_valid_coloring (m n : ℕ) : ¬∃ (coloring : GridColoring m n),
  equalColors m n coloring ∧ rowColorDominance m n coloring ∧ columnColorDominance m n coloring :=
sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l1220_122045


namespace NUMINAMATH_CALUDE_tangent_properties_l1220_122005

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ is also the tangent line of g -/
def tangent_condition (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

theorem tangent_properties :
  (tangent_condition 3 (-1)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition a x₁) → a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_properties_l1220_122005


namespace NUMINAMATH_CALUDE_island_subdivision_theorem_l1220_122000

/-- Represents a country on the island -/
structure Country where
  id : Nat

/-- Represents the map of the island -/
structure IslandMap where
  countries : List Country
  borders : List (Country × Country)

/-- Represents a coloring of the map -/
def Coloring := Country → Bool

/-- Check if a coloring is valid (adjacent countries have different colors) -/
def is_valid_coloring (map : IslandMap) (coloring : Coloring) : Prop :=
  ∀ c1 c2, (c1, c2) ∈ map.borders → coloring c1 ≠ coloring c2

/-- A subdivision of the map -/
def Subdivision := IslandMap → IslandMap

/-- The main theorem: There exists a subdivision that allows for a valid two-coloring -/
theorem island_subdivision_theorem (map : IslandMap) :
  ∃ (sub : Subdivision) (coloring : Coloring),
    is_valid_coloring (sub map) coloring :=
  sorry


end NUMINAMATH_CALUDE_island_subdivision_theorem_l1220_122000


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1220_122065

-- Define the divisibility relation
def divides (m n : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- Define an infinite set of natural numbers
def InfiniteSet (S : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ m ∈ S, m > n

theorem divisibility_implies_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, InfiniteSet S ∧ ∀ n ∈ S, divides (a^n + b^n) (a^(n+1) + b^(n+1))) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1220_122065


namespace NUMINAMATH_CALUDE_intersection_condition_max_area_rhombus_condition_l1220_122054

-- Define the lines and ellipse
def l₁ (k x : ℝ) : ℝ := k * x + 2
def l₂ (k x : ℝ) : ℝ := k * x - 2
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop := ∃ A B C D : ℝ × ℝ,
  ellipse A.1 A.2 ∧ (A.2 = l₁ k A.1 ∨ A.2 = l₂ k A.1) ∧
  ellipse B.1 B.2 ∧ (B.2 = l₁ k B.1 ∨ B.2 = l₂ k B.1) ∧
  ellipse C.1 C.2 ∧ (C.2 = l₁ k C.1 ∨ C.2 = l₂ k C.1) ∧
  ellipse D.1 D.2 ∧ (D.2 = l₁ k D.1 ∨ D.2 = l₂ k D.1)

-- Define the area of the quadrilateral
def area (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define the rhombus condition
def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

theorem intersection_condition (k : ℝ) :
  intersection_points k ↔ abs k > Real.sqrt 3 / 3 := sorry

theorem max_area {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, area A B C D ≤ 4 * Real.sqrt 3 := sorry

theorem rhombus_condition {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, is_rhombus A B C D → k = Real.sqrt 15 / 3 ∨ k = -Real.sqrt 15 / 3 := sorry

end NUMINAMATH_CALUDE_intersection_condition_max_area_rhombus_condition_l1220_122054


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1220_122062

theorem arithmetic_calculation : 1 + 2 * 3 - 4 + 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1220_122062


namespace NUMINAMATH_CALUDE_pauls_journey_time_l1220_122019

theorem pauls_journey_time (paul_time : ℝ) : 
  (paul_time + 7 * (paul_time + 2) = 46) → paul_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_pauls_journey_time_l1220_122019


namespace NUMINAMATH_CALUDE_pechkin_calculation_error_l1220_122012

/-- Represents Pechkin's journey --/
structure PechkinJourney where
  totalDistance : ℝ
  walkingSpeed : ℝ
  cyclingSpeed : ℝ
  walkingDistance : ℝ
  cyclingTime : ℝ
  totalTime : ℝ

/-- Conditions of Pechkin's journey --/
def journeyConditions (j : PechkinJourney) : Prop :=
  j.walkingSpeed = 5 ∧
  j.cyclingSpeed = 12 ∧
  j.walkingDistance = j.totalDistance / 2 ∧
  j.cyclingTime = j.totalTime / 3

/-- Theorem stating that Pechkin's calculations are inconsistent --/
theorem pechkin_calculation_error (j : PechkinJourney) 
  (h : journeyConditions j) : 
  j.cyclingSpeed * j.cyclingTime ≠ j.totalDistance - j.walkingDistance :=
sorry

end NUMINAMATH_CALUDE_pechkin_calculation_error_l1220_122012


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l1220_122014

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem smaller_cuboid_width :
  let large_cuboid := CuboidDimensions.mk 12 14 10
  let num_smaller_cuboids : ℕ := 56
  let smaller_cuboid_length : ℝ := 5
  let smaller_cuboid_height : ℝ := 2
  let large_volume := cuboidVolume large_cuboid
  let smaller_volume := large_volume / num_smaller_cuboids
  smaller_volume / (smaller_cuboid_length * smaller_cuboid_height) = 3 := by
  sorry

#check smaller_cuboid_width

end NUMINAMATH_CALUDE_smaller_cuboid_width_l1220_122014


namespace NUMINAMATH_CALUDE_floor_abs_sum_l1220_122063

theorem floor_abs_sum : ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l1220_122063


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l1220_122046

theorem least_product_of_primes_above_30 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 30 → s > 30 → r ≠ s → r * s ≥ 1147 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l1220_122046


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1220_122034

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (1 - a) * x^2 - 2 * x + 1 < 0}
  if a > 1 then
    S = {x : ℝ | x < (1 - Real.sqrt a) / (a - 1) ∨ x > (1 + Real.sqrt a) / (a - 1)}
  else if a = 1 then
    S = {x : ℝ | x > 1 / 2}
  else if 0 < a ∧ a < 1 then
    S = {x : ℝ | (1 - Real.sqrt a) / (1 - a) < x ∧ x < (1 + Real.sqrt a) / (1 - a)}
  else
    S = ∅ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1220_122034


namespace NUMINAMATH_CALUDE_candy_count_is_twelve_l1220_122037

/-- The total number of candy pieces Wendy and her brother have -/
def total_candy (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  brother_candy + wendy_boxes * pieces_per_box

/-- Theorem: The total number of candy pieces Wendy and her brother have is 12 -/
theorem candy_count_is_twelve :
  total_candy 6 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_is_twelve_l1220_122037


namespace NUMINAMATH_CALUDE_power_division_rule_l1220_122056

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1220_122056


namespace NUMINAMATH_CALUDE_field_trip_adults_l1220_122018

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 25 →
  num_vans = 6 →
  ∃ (num_adults : ℕ), num_adults = num_vans * van_capacity - num_students ∧ num_adults = 5 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1220_122018


namespace NUMINAMATH_CALUDE_class_size_problem_l1220_122079

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (size_B : ℕ) (size_A : ℕ) (size_C : ℕ)
  (h1 : size_A = 2 * size_B)
  (h2 : size_A = size_C / 3)
  (h3 : size_B = 20) :
  size_C = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l1220_122079


namespace NUMINAMATH_CALUDE_equation_equivalence_l1220_122067

theorem equation_equivalence (x : ℝ) : 6 - (x - 2) / 2 = x ↔ 12 - x + 2 = 2 * x :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1220_122067


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1220_122051

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1220_122051


namespace NUMINAMATH_CALUDE_speech_contest_probability_l1220_122061

/-- Represents the number of participants in the speech contest -/
def total_participants : ℕ := 10

/-- Represents the number of participants from Class 1 -/
def class1_participants : ℕ := 3

/-- Represents the number of participants from Class 2 -/
def class2_participants : ℕ := 2

/-- Represents the number of participants from other classes -/
def other_participants : ℕ := 5

/-- Calculates the probability of Class 1 students being consecutive and Class 2 students not being consecutive -/
def probability_class1_consecutive_class2_not : ℚ :=
  1 / 20

/-- Theorem stating the probability of the given event -/
theorem speech_contest_probability :
  probability_class1_consecutive_class2_not = 1 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_speech_contest_probability_l1220_122061


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1220_122023

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/82 - 1) * 100 := by
  sorry

#eval (100/82 - 1) * 100 -- This will output approximately 21.95

end NUMINAMATH_CALUDE_profit_percent_calculation_l1220_122023


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l1220_122031

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_percentage : ℚ)
  (total_red_percentage : ℚ)
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : honda_red_percentage = 90 / 100)
  (h4 : total_red_percentage = 60 / 100)
  : (((total_red_percentage * total_cars) - (honda_red_percentage * honda_cars)) /
     (total_cars - honda_cars) : ℚ) = 225 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l1220_122031


namespace NUMINAMATH_CALUDE_greatest_equal_distribution_l1220_122030

theorem greatest_equal_distribution (a b c : ℕ) (ha : a = 1050) (hb : b = 1260) (hc : c = 210) :
  Nat.gcd a (Nat.gcd b c) = 210 := by
  sorry

end NUMINAMATH_CALUDE_greatest_equal_distribution_l1220_122030


namespace NUMINAMATH_CALUDE_sadie_homework_problem_l1220_122028

/-- The total number of math homework problems Sadie has for the week. -/
def total_problems : ℕ := 140

/-- The number of solving linear equations problems Sadie has. -/
def linear_equations_problems : ℕ := 28

/-- Theorem stating that the total number of math homework problems is 140,
    given the conditions from the problem. -/
theorem sadie_homework_problem :
  (total_problems : ℝ) * 0.4 * 0.5 = linear_equations_problems :=
by sorry

end NUMINAMATH_CALUDE_sadie_homework_problem_l1220_122028


namespace NUMINAMATH_CALUDE_zero_not_in_positive_integers_l1220_122060

theorem zero_not_in_positive_integers : 0 ∉ {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_positive_integers_l1220_122060


namespace NUMINAMATH_CALUDE_simplify_expression_l1220_122003

theorem simplify_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + 2 * Real.sqrt (x * y) + y) / (Real.sqrt x + Real.sqrt y) - 
  (Real.sqrt (x * y) + Real.sqrt x) * Real.sqrt (1 / x) = Real.sqrt x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1220_122003


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1220_122087

theorem polynomial_simplification (x : ℝ) : 
  (x^5 + x^4 + x + 10) - (x^5 + 2*x^4 - x^3 + 12) = -x^4 + x^3 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1220_122087


namespace NUMINAMATH_CALUDE_total_profit_is_36000_l1220_122072

/-- The total subscription amount in rupees -/
def total_subscription : ℕ := 50000

/-- The amount a receives from the profit in rupees -/
def a_profit : ℕ := 15120

/-- The difference between a's and b's subscriptions in rupees -/
def a_b_diff : ℕ := 4000

/-- The difference between b's and c's subscriptions in rupees -/
def b_c_diff : ℕ := 5000

/-- Theorem stating that the total profit is 36000 rupees -/
theorem total_profit_is_36000 :
  ∃ (a b c : ℕ) (total_profit : ℕ),
    a = b + a_b_diff ∧
    b = c + b_c_diff ∧
    a + b + c = total_subscription ∧
    a_profit * total_subscription = a * total_profit ∧
    total_profit = 36000 :=
sorry

end NUMINAMATH_CALUDE_total_profit_is_36000_l1220_122072


namespace NUMINAMATH_CALUDE_square_area_17m_l1220_122073

theorem square_area_17m (side_length : ℝ) (h : side_length = 17) :
  side_length * side_length = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_area_17m_l1220_122073


namespace NUMINAMATH_CALUDE_student_arrangements_l1220_122029

/-- The number of students in the row -/
def n : ℕ := 7

/-- Calculate the number of arrangements where two students are adjacent -/
def arrangements_two_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are adjacent -/
def arrangements_three_adjacent (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where two students are adjacent and one student is not at either end -/
def arrangements_two_adjacent_one_not_end (n : ℕ) : ℕ := sorry

/-- Calculate the number of arrangements where three students are together and the other four are together -/
def arrangements_two_groups (n : ℕ) : ℕ := sorry

theorem student_arrangements :
  arrangements_two_adjacent n = 1440 ∧
  arrangements_three_adjacent n = 720 ∧
  arrangements_two_adjacent_one_not_end n = 960 ∧
  arrangements_two_groups n = 288 := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l1220_122029


namespace NUMINAMATH_CALUDE_one_large_pizza_sufficient_l1220_122086

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSizes where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of pizzas ordered for each dietary restriction --/
structure PizzaOrder where
  gluten_free_small : Nat
  dairy_free_medium : Nat
  large : Nat

/-- Calculates if the pizza order is sufficient for both brothers --/
def is_sufficient_order (sizes : PizzaSizes) (order : PizzaOrder) : Prop :=
  let gluten_free_slices := order.gluten_free_small * sizes.small + order.large * sizes.large
  let dairy_free_slices := order.dairy_free_medium * sizes.medium
  gluten_free_slices ≥ 15 ∧ dairy_free_slices ≥ 15

/-- Theorem stating that ordering 1 large pizza is sufficient --/
theorem one_large_pizza_sufficient 
  (sizes : PizzaSizes)
  (h_large : sizes.large = 14)
  (h_medium : sizes.medium = 10)
  (h_small : sizes.small = 8) :
  is_sufficient_order sizes { gluten_free_small := 1, dairy_free_medium := 2, large := 1 } :=
by sorry


end NUMINAMATH_CALUDE_one_large_pizza_sufficient_l1220_122086


namespace NUMINAMATH_CALUDE_teacher_age_l1220_122038

theorem teacher_age (num_students : ℕ) (avg_age_students : ℕ) (avg_increase : ℕ) : 
  num_students = 22 →
  avg_age_students = 21 →
  avg_increase = 1 →
  (num_students * avg_age_students + 44) / (num_students + 1) = avg_age_students + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l1220_122038


namespace NUMINAMATH_CALUDE_amelia_half_money_left_l1220_122081

/-- Represents the fraction of money Amelia has left after buying all books -/
def amelia_money_left (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) : ℝ :=
  total_money - (book_cost * num_books)

/-- Theorem stating that Amelia will have half of her money left after buying all books -/
theorem amelia_half_money_left 
  (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) 
  (h1 : total_money > 0) 
  (h2 : book_cost > 0) 
  (h3 : num_books > 0)
  (h4 : (1/4) * total_money = (1/2) * (book_cost * num_books)) :
  amelia_money_left total_money book_cost num_books = (1/2) * total_money := by
  sorry

#check amelia_half_money_left

end NUMINAMATH_CALUDE_amelia_half_money_left_l1220_122081


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1220_122024

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 5 * x - 2 * y = 10}
  let original_slope : ℝ := 5 / 2
  let perpendicular_slope : ℝ := -1 / original_slope
  perpendicular_slope = -2 / 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1220_122024


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l1220_122009

/-- A polynomial is a perfect square binomial if it can be expressed as (px + q)^2 for some real p and q -/
def is_perfect_square_binomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If ax^2 + 21x + 9 is the square of a binomial, then a = 49/4 -/
theorem square_binomial_coefficient (a : ℝ) :
  is_perfect_square_binomial a 21 9 → a = 49 / 4 := by
  sorry


end NUMINAMATH_CALUDE_square_binomial_coefficient_l1220_122009


namespace NUMINAMATH_CALUDE_total_points_scored_l1220_122048

/-- Given a player who played 10.0 games and scored 12 points in each game,
    the total points scored is 120. -/
theorem total_points_scored (games : ℝ) (points_per_game : ℕ) : 
  games = 10.0 → points_per_game = 12 → games * (points_per_game : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l1220_122048


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1220_122077

theorem triangle_angle_problem (A B C : ℝ) : 
  A = 32 ∧ 
  C = 2 * A - 12 ∧ 
  B = 3 * A ∧ 
  A + B + C = 180 → 
  B = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1220_122077


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1220_122055

/-- Given a quadratic equation x^2 - 4x = 5, prove that its standard form coefficients are 1, -4, and -5 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 4*x = 5 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -4 ∧ c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1220_122055


namespace NUMINAMATH_CALUDE_paint_per_statue_l1220_122008

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) : 
  total_paint = 7/8 ∧ num_statues = 14 → 
  total_paint / num_statues = 7/112 := by
sorry

end NUMINAMATH_CALUDE_paint_per_statue_l1220_122008


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1220_122032

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - 1)

theorem f_max_min_on_interval :
  let a : ℝ := -3 * Real.pi / 4
  let b : ℝ := 3 * Real.pi / 4
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 0 ∧
    min = -(Real.sqrt 2 / 2) * Real.exp (3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1220_122032


namespace NUMINAMATH_CALUDE_paperback_cost_is_twelve_l1220_122082

/-- Represents the book club's annual fee collection --/
structure BookClub where
  members : ℕ
  snackFeePerMember : ℕ
  hardcoverBooksPerMember : ℕ
  hardcoverBookPrice : ℕ
  paperbackBooksPerMember : ℕ
  totalCollected : ℕ

/-- Calculates the cost per paperback book --/
def costPerPaperback (club : BookClub) : ℚ :=
  let snackTotal := club.members * club.snackFeePerMember
  let hardcoverTotal := club.members * club.hardcoverBooksPerMember * club.hardcoverBookPrice
  let paperbackTotal := club.totalCollected - snackTotal - hardcoverTotal
  paperbackTotal / (club.members * club.paperbackBooksPerMember)

/-- Theorem stating that the cost per paperback book is $12 --/
theorem paperback_cost_is_twelve (club : BookClub) 
    (h1 : club.members = 6)
    (h2 : club.snackFeePerMember = 150)
    (h3 : club.hardcoverBooksPerMember = 6)
    (h4 : club.hardcoverBookPrice = 30)
    (h5 : club.paperbackBooksPerMember = 6)
    (h6 : club.totalCollected = 2412) :
    costPerPaperback club = 12 := by
  sorry


end NUMINAMATH_CALUDE_paperback_cost_is_twelve_l1220_122082


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l1220_122033

theorem infinitely_many_primes_6n_plus_5 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p = 6 * n + 5} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l1220_122033


namespace NUMINAMATH_CALUDE_v_closed_under_cube_l1220_122057

-- Define the set v as the set of fourth powers of positive integers
def v : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^4}

-- Theorem statement
theorem v_closed_under_cube (x : ℕ) (hx : x ∈ v) : x^3 ∈ v := by
  sorry

end NUMINAMATH_CALUDE_v_closed_under_cube_l1220_122057


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1220_122004

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 8 ∧ ∀ m : ℤ, |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1220_122004


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_neg_one_l1220_122011

/-- Given a quadratic function f(x) = x^2 - 2mx + 6 that is decreasing on (-∞, -1],
    prove that m ≥ -1 -/
theorem quadratic_decreasing_implies_m_geq_neg_one (m : ℝ) : 
  (∀ x ≤ -1, ∀ y ≤ -1, x < y → (x^2 - 2*m*x + 6) > (y^2 - 2*m*y + 6)) → 
  m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_neg_one_l1220_122011


namespace NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1220_122064

theorem cylinder_in_sphere_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 6) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * (r_sphere ^ 2 - r_cylinder ^ 2).sqrt
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  (v_sphere - v_cylinder) / π = 288 - 64 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1220_122064


namespace NUMINAMATH_CALUDE_joe_age_proof_l1220_122095

theorem joe_age_proof (joe james : ℕ) : 
  joe = james + 10 →
  2 * (joe + 8) = 3 * (james + 8) →
  joe = 22 := by
sorry

end NUMINAMATH_CALUDE_joe_age_proof_l1220_122095


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l1220_122049

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem k_range_when_intersection_nonempty :
  ∀ k : ℝ, (A ∩ B k).Nonempty → k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l1220_122049


namespace NUMINAMATH_CALUDE_card_game_properties_l1220_122074

/-- A card collection game with 3 colors -/
structure CardGame where
  colors : Nat
  cards_per_color : Nat

/-- The probability of not collecting 3 cards of the same color after 4 purchases -/
def prob_not_three_same (game : CardGame) : ℚ :=
  2 / 3

/-- The distribution of X (number of purchases before collecting 3 cards of the same color) -/
def distribution_X (game : CardGame) (x : Nat) : ℚ :=
  match x with
  | 3 => 1 / 9
  | 4 => 2 / 9
  | 5 => 8 / 27
  | 6 => 20 / 81
  | 7 => 10 / 81
  | _ => 0

/-- The expectation of X -/
def expectation_X (game : CardGame) : ℚ :=
  409 / 81

/-- Main theorem about the card collection game -/
theorem card_game_properties (game : CardGame) 
    (h1 : game.colors = 3) 
    (h2 : game.cards_per_color = 3) : 
  prob_not_three_same game = 2 / 3 ∧ 
  (∀ x, distribution_X game x = match x with
                                | 3 => 1 / 9
                                | 4 => 2 / 9
                                | 5 => 8 / 27
                                | 6 => 20 / 81
                                | 7 => 10 / 81
                                | _ => 0) ∧
  expectation_X game = 409 / 81 := by
  sorry

end NUMINAMATH_CALUDE_card_game_properties_l1220_122074


namespace NUMINAMATH_CALUDE_set_operations_l1220_122022

def U : Set Nat := {1,2,3,4,5,6,7,8,9,10,11,13}
def A : Set Nat := {2,4,6,8}
def B : Set Nat := {3,4,5,6,8,9,11}

theorem set_operations :
  (A ∪ B = {2,3,4,5,6,8,9,11}) ∧
  (U \ A = {1,3,5,7,9,10,11,13}) ∧
  (U \ (A ∩ B) = {1,2,3,5,7,9,10,11,13}) ∧
  (A ∪ (U \ B) = {1,2,4,6,7,8,10,13}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1220_122022


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1220_122068

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 32 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    a' * b' * c' * d' = 8 ∧ 
    e' * f' * g' * h' = 16 ∧ 
    (a' * e')^2 + (b' * f')^2 + (c' * g')^2 + (d' * h')^2 = 32 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1220_122068


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l1220_122078

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 10) 
  (h2 : current_speed = 2.5) : 
  speed_against_current + 2 * current_speed = 15 :=
by
  sorry

#check mans_speed_with_current

end NUMINAMATH_CALUDE_mans_speed_with_current_l1220_122078


namespace NUMINAMATH_CALUDE_problem_statement_l1220_122047

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1220_122047


namespace NUMINAMATH_CALUDE_absolute_value_subtraction_l1220_122020

theorem absolute_value_subtraction : 2 - |(-3)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_subtraction_l1220_122020


namespace NUMINAMATH_CALUDE_cookies_needed_to_fill_bags_l1220_122093

/-- Represents the number of cookies needed to fill a bag completely -/
def bagCapacity : ℕ := 16

/-- Represents the total number of cookies Edgar bought -/
def totalCookies : ℕ := 292

/-- Represents the number of chocolate chip cookies Edgar bought -/
def chocolateChipCookies : ℕ := 154

/-- Represents the number of oatmeal raisin cookies Edgar bought -/
def oatmealRaisinCookies : ℕ := 86

/-- Represents the number of sugar cookies Edgar bought -/
def sugarCookies : ℕ := 52

/-- Calculates the number of additional cookies needed to fill the last bag completely -/
def additionalCookiesNeeded (cookieCount : ℕ) : ℕ :=
  bagCapacity - (cookieCount % bagCapacity)

theorem cookies_needed_to_fill_bags :
  additionalCookiesNeeded chocolateChipCookies = 6 ∧
  additionalCookiesNeeded oatmealRaisinCookies = 10 ∧
  additionalCookiesNeeded sugarCookies = 12 :=
by
  sorry

#check cookies_needed_to_fill_bags

end NUMINAMATH_CALUDE_cookies_needed_to_fill_bags_l1220_122093


namespace NUMINAMATH_CALUDE_complex_quadrant_l1220_122002

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the conditions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Define the equation
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

-- Theorem statement
theorem complex_quadrant (z a : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : equation z a) : 
  (a + z).re > 0 ∧ (a + z).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1220_122002


namespace NUMINAMATH_CALUDE_expansion_no_constant_term_l1220_122006

def has_no_constant_term (n : ℕ+) : Prop :=
  ∀ k : ℕ, k ≤ n → (1 + k - 4 * (k / 4) ≠ 0 ∧ 2 + k - 4 * (k / 4) ≠ 0)

theorem expansion_no_constant_term (n : ℕ+) (h : 2 ≤ n ∧ n ≤ 7) :
  has_no_constant_term n ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_no_constant_term_l1220_122006


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1220_122039

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2 × 2 × 2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The main theorem -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 972 := by
  sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1220_122039


namespace NUMINAMATH_CALUDE_numerals_with_prime_first_digit_l1220_122044

/-- The set of prime digits less than 10 -/
def primedigits : Finset ℕ := {2, 3, 5, 7}

/-- The number of numerals with prime first digit -/
def num_numerals : ℕ := 400

/-- The number of digits in the numerals -/
def num_digits : ℕ := 3

theorem numerals_with_prime_first_digit :
  (primedigits.card : ℝ) * (10 ^ (num_digits - 1)) = num_numerals := by sorry

end NUMINAMATH_CALUDE_numerals_with_prime_first_digit_l1220_122044


namespace NUMINAMATH_CALUDE_max_profit_at_180_l1220_122015

-- Define the selling price x and daily sales y
variable (x y : ℝ)

-- Define the cost price
def cost_price : ℝ := 80

-- Define the range of selling price
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the relationship between y and x
def sales_function (x : ℝ) : ℝ := -0.5 * x + 160

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (sales_function x)

-- Theorem statement
theorem max_profit_at_180 :
  ∀ x, selling_price_range x →
    profit_function x ≤ profit_function 180 ∧
    profit_function 180 = 7000 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_180_l1220_122015


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1220_122052

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log (2^x) > 1)) ∧
  (∀ x : ℝ, Real.log (2^x) > 1 → x > 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1220_122052


namespace NUMINAMATH_CALUDE_intersection_points_are_two_and_eight_l1220_122090

/-- The set of k values for which |z - 4| = 3|z + 4| intersects |z| = k at exactly one point -/
def intersection_points : Set ℝ :=
  {k : ℝ | ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_points set contains only 2 and 8 -/
theorem intersection_points_are_two_and_eight :
  intersection_points = {2, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_are_two_and_eight_l1220_122090


namespace NUMINAMATH_CALUDE_angle_function_equality_l1220_122035

/-- Given an angle α in the third quadrant, if cos(α - 3π/2) = 1/5, then
    (sin(α - π/2) * cos(3π/2 + α) * tan(π - α)) / (tan(-α - π) * sin(-α - π)) = 2√6/5 -/
theorem angle_function_equality (α : Real) 
    (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
    (h2 : Real.cos (α - 3*π/2) = 1/5) :
    (Real.sin (α - π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)) / 
    (Real.tan (-α - π) * Real.sin (-α - π)) = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_equality_l1220_122035


namespace NUMINAMATH_CALUDE_book_ratio_is_four_to_one_l1220_122013

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The total number of books Zig and Flo wrote together -/
def total_books : ℕ := 75

/-- The number of books Flo wrote -/
def flo_books : ℕ := total_books - zig_books

/-- The ratio of books written by Zig to books written by Flo -/
def book_ratio : ℚ := zig_books / flo_books

theorem book_ratio_is_four_to_one :
  book_ratio = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_book_ratio_is_four_to_one_l1220_122013


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1220_122027

theorem rectangle_area_equals_perimeter (b : ℝ) (h1 : b > 0) :
  let l := 3 * b
  let area := l * b
  let perimeter := 2 * (l + b)
  area = perimeter → b = 8/3 ∧ l = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1220_122027


namespace NUMINAMATH_CALUDE_triangle_type_l1220_122075

theorem triangle_type (A B C : ℝ) (a b c : ℝ) 
  (h : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.sin C) :
  A = π / 4 ∧ B = π / 4 ∧ C = π / 2 := by
  sorry

#check triangle_type

end NUMINAMATH_CALUDE_triangle_type_l1220_122075


namespace NUMINAMATH_CALUDE_no_double_composition_square_minus_two_l1220_122098

theorem no_double_composition_square_minus_two :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_double_composition_square_minus_two_l1220_122098


namespace NUMINAMATH_CALUDE_x_satisfies_quadratic_l1220_122089

theorem x_satisfies_quadratic (x y : ℝ) 
  (h1 : x^2 - y = 10) 
  (h2 : x + y = 14) : 
  x^2 + x - 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_satisfies_quadratic_l1220_122089


namespace NUMINAMATH_CALUDE_danny_found_eighteen_caps_l1220_122069

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initial : ℕ) (total : ℕ) : ℕ := total - initial

/-- Theorem: Danny found 18 bottle caps at the park -/
theorem danny_found_eighteen_caps : 
  let initial := 37
  let total := 55
  bottleCapsFound initial total = 18 := by
sorry

end NUMINAMATH_CALUDE_danny_found_eighteen_caps_l1220_122069


namespace NUMINAMATH_CALUDE_largest_common_term_proof_l1220_122059

def arithmetic_progression (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def is_common_term (t : ℕ) : Prop :=
  ∃ n m : ℕ, arithmetic_progression 7 7 n = t ∧ arithmetic_progression 5 10 m = t

def largest_common_term_below_300 : ℕ := 265

theorem largest_common_term_proof :
  (is_common_term largest_common_term_below_300) ∧
  (∀ t : ℕ, t < 300 → is_common_term t → t ≤ largest_common_term_below_300) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_proof_l1220_122059


namespace NUMINAMATH_CALUDE_total_pancakes_l1220_122070

/-- Represents the number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- Represents the number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- Represents the number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- Represents the number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- Theorem stating the total number of pancakes Hank needs to make -/
theorem total_pancakes : 
  short_stack_orders * short_stack + big_stack_orders * big_stack = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_pancakes_l1220_122070


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1220_122010

def has_90_divisors (n : ℕ) : Prop :=
  (Finset.card (Nat.divisors n) = 90)

def ends_with_8_zeros (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^8 * k

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    ends_with_8_zeros a ∧
    ends_with_8_zeros b ∧
    has_90_divisors a ∧
    has_90_divisors b ∧
    a + b = 700000000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1220_122010


namespace NUMINAMATH_CALUDE_star_running_back_yards_l1220_122080

/-- Represents the yardage statistics for a football player -/
structure PlayerStats where
  total_yards : ℕ
  pass_yards : ℕ
  run_yards : ℕ

/-- Calculates the running yards for a player given total yards and pass yards -/
def calculate_run_yards (total : ℕ) (pass : ℕ) : ℕ :=
  total - pass

/-- Theorem stating that the star running back's running yards is 90 -/
theorem star_running_back_yards (player : PlayerStats)
    (h1 : player.total_yards = 150)
    (h2 : player.pass_yards = 60)
    (h3 : player.run_yards = calculate_run_yards player.total_yards player.pass_yards) :
    player.run_yards = 90 := by
  sorry

end NUMINAMATH_CALUDE_star_running_back_yards_l1220_122080


namespace NUMINAMATH_CALUDE_work_time_ratio_l1220_122084

theorem work_time_ratio (a b : ℝ) (h1 : b = 18) (h2 : 1/a + 1/b = 1/3) :
  a / b = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_work_time_ratio_l1220_122084


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l1220_122025

/-- Represents the ticket tiers --/
inductive TicketTier
  | Standard
  | Premium
  | VIP

/-- Represents the ticket types for the second show --/
inductive TicketType
  | Regular
  | Student
  | Senior

/-- Ticket prices for the first show --/
def firstShowPrice (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 25
  | TicketTier.Premium => 40
  | TicketTier.VIP => 60

/-- Ticket quantities for the first show --/
def firstShowQuantity (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 120
  | TicketTier.Premium => 60
  | TicketTier.VIP => 20

/-- Discount rates for the second show --/
def discountRate (type : TicketType) : ℚ :=
  match type with
  | TicketType.Regular => 0
  | TicketType.Student => 0.1
  | TicketType.Senior => 0.15

/-- Ticket quantities for the second show --/
def secondShowQuantity (tier : TicketTier) (type : TicketType) : ℕ :=
  match tier, type with
  | TicketTier.Standard, TicketType.Student => 240
  | TicketTier.Standard, TicketType.Senior => 120
  | TicketTier.Premium, TicketType.Student => 120
  | TicketTier.Premium, TicketType.Senior => 60
  | TicketTier.VIP, TicketType.Student => 40
  | TicketTier.VIP, TicketType.Senior => 20
  | _, TicketType.Regular => 0

/-- Calculate the earnings from the first show --/
def firstShowEarnings : ℕ :=
  (firstShowQuantity TicketTier.Standard * firstShowPrice TicketTier.Standard) +
  (firstShowQuantity TicketTier.Premium * firstShowPrice TicketTier.Premium) +
  (firstShowQuantity TicketTier.VIP * firstShowPrice TicketTier.VIP)

/-- Calculate the discounted price for the second show --/
def secondShowPrice (tier : TicketTier) (type : TicketType) : ℚ :=
  (firstShowPrice tier : ℚ) * (1 - discountRate type)

/-- Calculate the earnings from the second show --/
def secondShowEarnings : ℚ :=
  (secondShowQuantity TicketTier.Standard TicketType.Student * secondShowPrice TicketTier.Standard TicketType.Student) +
  (secondShowQuantity TicketTier.Standard TicketType.Senior * secondShowPrice TicketTier.Standard TicketType.Senior) +
  (secondShowQuantity TicketTier.Premium TicketType.Student * secondShowPrice TicketTier.Premium TicketType.Student) +
  (secondShowQuantity TicketTier.Premium TicketType.Senior * secondShowPrice TicketTier.Premium TicketType.Senior) +
  (secondShowQuantity TicketTier.VIP TicketType.Student * secondShowPrice TicketTier.VIP TicketType.Student) +
  (secondShowQuantity TicketTier.VIP TicketType.Senior * secondShowPrice TicketTier.VIP TicketType.Senior)

/-- The main theorem stating that the total earnings from both shows equal $24,090 --/
theorem total_earnings_theorem :
  (firstShowEarnings : ℚ) + secondShowEarnings = 24090 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l1220_122025


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_product_l1220_122007

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has unique digits -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem smallest_sum_of_digits_product :
  ∃ (x y : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    has_unique_digits (x * 100 + y) ∧
    (x * y ≥ 1000) ∧
    (x * y < 10000) ∧
    sum_of_digits (x * y) = 12 ∧
    ∀ (a b : ℕ),
      is_two_digit a →
      is_two_digit b →
      has_unique_digits (a * 100 + b) →
      (a * b ≥ 1000) →
      (a * b < 10000) →
      sum_of_digits (a * b) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_product_l1220_122007


namespace NUMINAMATH_CALUDE_jar_marbles_l1220_122017

theorem jar_marbles (a b c : ℕ) : 
  b = a + 12 →
  c = 2 * b →
  a + b + c = 148 →
  a = 28 := by sorry

end NUMINAMATH_CALUDE_jar_marbles_l1220_122017


namespace NUMINAMATH_CALUDE_regular_discount_is_30_percent_l1220_122066

/-- The regular discount range for pet food at a store -/
def regular_discount_range : ℝ := sorry

/-- The additional sale discount percentage -/
def additional_sale_discount : ℝ := 0.20

/-- The manufacturer's suggested retail price (MSRP) for a container of pet food -/
def msrp : ℝ := 35.00

/-- The lowest possible price after the additional sale discount -/
def lowest_sale_price : ℝ := 19.60

/-- Theorem stating that the regular discount range is 30% -/
theorem regular_discount_is_30_percent :
  regular_discount_range = 0.30 := by sorry

end NUMINAMATH_CALUDE_regular_discount_is_30_percent_l1220_122066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1220_122036

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d < 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 2 * a 4 = 12
  h4 : a 2 + a 4 = 8

/-- The theorem stating the existence of a unique solution and sum of first 10 terms -/
theorem arithmetic_sequence_solution (seq : ArithmeticSequence) :
  ∃! (a₁ : ℝ), 
    (seq.a 1 = a₁) ∧
    (∃! (d : ℝ), d = seq.d) ∧
    (∃ (S₁₀ : ℝ), S₁₀ = (10 * seq.a 1) + (10 * 9 / 2 * seq.d)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1220_122036


namespace NUMINAMATH_CALUDE_solve_for_a_l1220_122053

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 8*a^2

-- Define the theorem
theorem solve_for_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0)
  (h₂ : ∀ x, f a x < 0 ↔ x₁ < x ∧ x < x₂)
  (h₃ : x₂ - x₁ = 15) :
  a = 5/2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l1220_122053


namespace NUMINAMATH_CALUDE_volume_S_polynomial_bc_over_ad_value_l1220_122026

/-- A right rectangular prism with edge lengths 2, 4, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- The coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_S_polynomial (B : RectPrism) :
  ∃ coeffs : VolumeCoeffs,
    ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d :=
  sorry

theorem bc_over_ad_value (B : RectPrism) (coeffs : VolumeCoeffs)
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.675 :=
  sorry

end NUMINAMATH_CALUDE_volume_S_polynomial_bc_over_ad_value_l1220_122026


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1220_122040

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1220_122040


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l1220_122071

/-- Proves that the first interest rate is 10% given the problem conditions --/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (total_profit : ℕ)
  (second_rate : ℚ)
  (h1 : total_amount = 50000)
  (h2 : first_part = 30000)
  (h3 : second_part = total_amount - first_part)
  (h4 : total_profit = 7000)
  (h5 : second_rate = 20 / 100)
  : ∃ (r : ℚ), r = 10 / 100 ∧ 
    total_profit = (first_part * r).floor + (second_part * second_rate).floor :=
by sorry


end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l1220_122071


namespace NUMINAMATH_CALUDE_speed_to_achieve_average_l1220_122016

/-- Given a person driving at two different speeds over two time periods, 
    this theorem proves the required speed for the second period to achieve a specific average speed. -/
theorem speed_to_achieve_average 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (additional_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 3) 
  (h3 : additional_time = 2) 
  (h4 : average_speed = 70) : 
  ∃ x : ℝ, 
    (initial_speed * initial_time + x * additional_time) / (initial_time + additional_time) = average_speed 
    ∧ x = 85 := by
  sorry

end NUMINAMATH_CALUDE_speed_to_achieve_average_l1220_122016


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1220_122091

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -8
def c : ℝ := -7

-- Define the condition for m (not divisible by the square of any prime)
def is_square_free (m : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ m) → False

-- Define the theorem
theorem quadratic_root_difference (m n : ℕ) (h1 : is_square_free m) (h2 : n > 0) :
  (((b^2 - 4*a*c).sqrt / (2*a)) = (m.sqrt / n)) → m + n = 56 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1220_122091


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1220_122042

theorem polynomial_factorization (x y : ℝ) : 
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2)*(x^2 - 3*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1220_122042
