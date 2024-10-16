import Mathlib

namespace NUMINAMATH_CALUDE_smallest_integer_for_inequality_l2858_285854

theorem smallest_integer_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 ≤ 3*(a^2*x^4 + b^2*y^4 + c^2*z^4)) ∧
  (∀ n : ℕ, n < 3 → ∃ x y z : ℝ, (a*x^2 + b*y^2 + c*z^2)^2 > n*(a^2*x^4 + b^2*y^4 + c^2*z^4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_for_inequality_l2858_285854


namespace NUMINAMATH_CALUDE_birdseed_mixture_cost_per_pound_l2858_285895

theorem birdseed_mixture_cost_per_pound
  (millet_weight : ℝ)
  (millet_cost_per_lb : ℝ)
  (sunflower_weight : ℝ)
  (sunflower_cost_per_lb : ℝ)
  (h1 : millet_weight = 100)
  (h2 : millet_cost_per_lb = 0.60)
  (h3 : sunflower_weight = 25)
  (h4 : sunflower_cost_per_lb = 1.10) :
  (millet_weight * millet_cost_per_lb + sunflower_weight * sunflower_cost_per_lb) /
  (millet_weight + sunflower_weight) = 0.70 := by
  sorry

#check birdseed_mixture_cost_per_pound

end NUMINAMATH_CALUDE_birdseed_mixture_cost_per_pound_l2858_285895


namespace NUMINAMATH_CALUDE_tommy_wheels_count_l2858_285890

/-- The number of wheels Tommy saw during his run -/
def total_wheels (truck_wheels car_wheels bicycle_wheels bus_wheels : ℕ)
                 (num_trucks num_cars num_bicycles num_buses : ℕ) : ℕ :=
  truck_wheels * num_trucks + car_wheels * num_cars +
  bicycle_wheels * num_bicycles + bus_wheels * num_buses

theorem tommy_wheels_count :
  total_wheels 4 4 2 6 12 13 8 3 = 134 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheels_count_l2858_285890


namespace NUMINAMATH_CALUDE_planted_fraction_is_seven_tenths_l2858_285887

/-- Represents a right triangle with a square at the right angle -/
structure RightTriangleWithSquare where
  leg1 : ℝ
  leg2 : ℝ
  squareDistance : ℝ

/-- Calculates the fraction of the area not covered by the square -/
def plantedFraction (t : RightTriangleWithSquare) : ℝ :=
  sorry

/-- Theorem statement for the specific problem -/
theorem planted_fraction_is_seven_tenths :
  let t : RightTriangleWithSquare := {
    leg1 := 5,
    leg2 := 12,
    squareDistance := 3
  }
  plantedFraction t = 7/10 :=
sorry

end NUMINAMATH_CALUDE_planted_fraction_is_seven_tenths_l2858_285887


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l2858_285875

/-- Given a car that uses 6.5 gallons of gasoline to travel 130 kilometers,
    prove that its fuel efficiency is 20 kilometers per gallon. -/
theorem car_fuel_efficiency :
  ∀ (distance : ℝ) (fuel : ℝ),
    distance = 130 →
    fuel = 6.5 →
    distance / fuel = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l2858_285875


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2858_285858

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 2 + a 6 = 6
  h2 : (5 * (a 1 + a 5)) / 2 = 35 / 3

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = (2 / 3) * n + 1 / 3) ∧
  (∀ n : ℕ, S seq n ≥ 1) ∧
  (S seq 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2858_285858


namespace NUMINAMATH_CALUDE_perpendicular_lines_direction_vectors_l2858_285846

theorem perpendicular_lines_direction_vectors (b : ℝ) :
  let v1 : Fin 2 → ℝ := ![- 5, 11]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, (v1 • v2) = 0) → b = 33 / 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_direction_vectors_l2858_285846


namespace NUMINAMATH_CALUDE_apples_given_away_l2858_285865

/-- Given that Joan picked a certain number of apples and now has fewer,
    prove that the number of apples she gave away is the difference between
    the initial and current number of apples. -/
theorem apples_given_away (initial current : ℕ) (h : current ≤ initial) :
  initial - current = initial - current := by sorry

end NUMINAMATH_CALUDE_apples_given_away_l2858_285865


namespace NUMINAMATH_CALUDE_function_composition_result_l2858_285834

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_result (a b : ℝ) :
  (∀ x, h a b x = x + 9) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l2858_285834


namespace NUMINAMATH_CALUDE_polynomial_expression_and_coefficient_sum_l2858_285814

theorem polynomial_expression_and_coefficient_sum :
  ∀ d : ℝ, d ≠ 0 →
  ∃ (a b c : ℤ),
    (10 * d + 17 + 12 * d^2) + (6 * d + 3) = a * d + b + c * d^2 ∧
    a = 16 ∧ b = 20 ∧ c = 12 ∧
    a + b + c = 48 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expression_and_coefficient_sum_l2858_285814


namespace NUMINAMATH_CALUDE_daniels_age_l2858_285821

/-- Given the ages of Uncle Ben, Edward, and Daniel, prove Daniel's age --/
theorem daniels_age (uncle_ben_age : ℚ) (edward_age : ℚ) (daniel_age : ℚ) : 
  uncle_ben_age = 50 →
  edward_age = 2/3 * uncle_ben_age →
  daniel_age = edward_age - 7 →
  daniel_age = 79/3 := by sorry

end NUMINAMATH_CALUDE_daniels_age_l2858_285821


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2858_285844

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2858_285844


namespace NUMINAMATH_CALUDE_horner_v3_value_l2858_285800

def horner_step (x : ℝ) (v : ℝ) (a : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (horner_step x) 0

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^3 + 5 * x^2 - 4

theorem horner_v3_value :
  let coeffs := [2, 0, -3, 5, 0, -4]
  let x := 2
  let v3 := (horner_method (coeffs.take 4) x)
  v3 = 15 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l2858_285800


namespace NUMINAMATH_CALUDE_triangle_area_l2858_285815

/-- Given vectors m and n, and function f, prove the area of triangle ABC -/
theorem triangle_area (x : ℝ) :
  let m : ℝ × ℝ := (Real.sqrt 3 * Real.sin x - Real.cos x, 1)
  let n : ℝ × ℝ := (Real.cos x, 1/2)
  let f : ℝ → ℝ := λ x => m.1 * n.1 + m.2 * n.2
  let a : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 4
  ∀ A : ℝ, f A = 1 →
    ∃ b : ℝ, 
      let s := (a + b + c) / 2
      2 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2858_285815


namespace NUMINAMATH_CALUDE_badminton_players_count_l2858_285849

theorem badminton_players_count (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_tennis : tennis = 18)
  (h_neither : neither = 5)
  (h_both : both = 3) :
  total = tennis + (total - tennis - neither) - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_count_l2858_285849


namespace NUMINAMATH_CALUDE_complex_fraction_modulus_l2858_285837

theorem complex_fraction_modulus (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i) / (a + b*i) = 2 - i → Complex.abs (a - b*i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_modulus_l2858_285837


namespace NUMINAMATH_CALUDE_order_of_abc_l2858_285811

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := 1 / Real.sin 1

theorem order_of_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2858_285811


namespace NUMINAMATH_CALUDE_g_of_5_l2858_285869

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_of_5 : g 5 = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l2858_285869


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2858_285878

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- a_n represents the nth term of the arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_problem (h : S 9 a = 45) : a 5 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2858_285878


namespace NUMINAMATH_CALUDE_loan_B_is_5000_l2858_285828

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  rate : ℚ  -- Interest rate per annum
  time_B : ℕ  -- Time for loan B in years
  time_C : ℕ  -- Time for loan C in years
  amount_C : ℚ  -- Amount lent to C
  total_interest : ℚ  -- Total interest received from both loans

/-- Calculates the amount lent to B given the loan details --/
def calculate_loan_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - loan.amount_C * loan.rate * loan.time_C) / (loan.rate * loan.time_B)

/-- Theorem stating that the amount lent to B is 5000 --/
theorem loan_B_is_5000 (loan : LoanDetails) 
  (h1 : loan.rate = 9 / 100)
  (h2 : loan.time_B = 2)
  (h3 : loan.time_C = 4)
  (h4 : loan.amount_C = 3000)
  (h5 : loan.total_interest = 1980) :
  calculate_loan_B loan = 5000 := by
  sorry

#eval calculate_loan_B { rate := 9/100, time_B := 2, time_C := 4, amount_C := 3000, total_interest := 1980 }

end NUMINAMATH_CALUDE_loan_B_is_5000_l2858_285828


namespace NUMINAMATH_CALUDE_total_quarters_l2858_285822

def initial_quarters : ℕ := 8
def additional_quarters : ℕ := 3

theorem total_quarters : initial_quarters + additional_quarters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_quarters_l2858_285822


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l2858_285850

theorem complex_root_magnitude (z : ℂ) : z^2 + z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l2858_285850


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_product_of_fractions_l2858_285885

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem product_of_fractions :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_product_of_fractions_l2858_285885


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2858_285805

/-- Given a geometric sequence {a_n} with sum S_n = 2010^n + t, prove a_1 = 2009 -/
theorem geometric_sequence_first_term (n : ℕ) (t : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ k, S k = 2010^k + t) →
  (a 1 * a 3 = (a 2)^2) →
  a 1 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2858_285805


namespace NUMINAMATH_CALUDE_power_of_power_l2858_285859

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2858_285859


namespace NUMINAMATH_CALUDE_sports_club_non_players_l2858_285863

theorem sports_club_non_players (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_non_players_l2858_285863


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2858_285861

theorem complex_division_simplification :
  (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2858_285861


namespace NUMINAMATH_CALUDE_sum_remainder_is_two_l2858_285804

theorem sum_remainder_is_two (n : ℤ) : (8 - n + (n + 4)) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_is_two_l2858_285804


namespace NUMINAMATH_CALUDE_division_multiplication_negatives_l2858_285818

theorem division_multiplication_negatives : (-100) / (-25) * (-6) = -24 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_negatives_l2858_285818


namespace NUMINAMATH_CALUDE_set_relationships_evaluation_l2858_285872

theorem set_relationships_evaluation :
  let s1 : Set (Set ℕ) := {{0}, {2, 3, 4}}
  let s2 : Set ℕ := {0}
  let s3 : Set ℤ := {-1, 0, 1}
  let s4 : Set ℤ := {0, -1, 1}
  ({0} ∈ s1) = false ∧
  (∅ ⊆ s2) = true ∧
  (s3 = s4) = true ∧
  (0 ∈ (∅ : Set ℕ)) = false :=
by sorry

end NUMINAMATH_CALUDE_set_relationships_evaluation_l2858_285872


namespace NUMINAMATH_CALUDE_hexagon_quadrilateral_areas_l2858_285876

/-- The area of a regular hexagon -/
def hexagon_area : ℝ := 156

/-- The number of distinct quadrilateral shapes possible -/
def num_distinct_quadrilaterals : ℕ := 3

/-- The areas of the distinct quadrilaterals -/
def quadrilateral_areas : Set ℝ := {78, 104}

/-- Theorem: Given a regular hexagon with area 156 cm², the areas of all possible
    distinct quadrilaterals formed by its vertices are 78 cm² and 104 cm² -/
theorem hexagon_quadrilateral_areas :
  ∀ (area : ℝ), area ∈ quadrilateral_areas →
  ∃ (vertices : Finset (Fin 6)), vertices.card = 4 ∧
  (area = hexagon_area / 2 ∨ area = hexagon_area * 2 / 3) :=
sorry

end NUMINAMATH_CALUDE_hexagon_quadrilateral_areas_l2858_285876


namespace NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l2858_285868

theorem absolute_difference_of_product_and_sum (m n : ℝ) 
  (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l2858_285868


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2858_285826

/-- A piecewise function f: ℝ → ℝ defined by two parts -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (2*a - 1)*x + 7*a - 2 else a^x

/-- Theorem stating the condition for f to be monotonically decreasing -/
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ 3/8 ≤ a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2858_285826


namespace NUMINAMATH_CALUDE_largest_B_term_l2858_285883

def B (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ j ∈ Finset.range 2001, B 181 ≥ B j :=
sorry

end NUMINAMATH_CALUDE_largest_B_term_l2858_285883


namespace NUMINAMATH_CALUDE_quadratic_reducible_conditions_l2858_285848

def is_quadratic_or_reducible (a b : ℚ) : Prop :=
  ∃ (p q r : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
    (a / (1 - x) - 2 / (2 - x) + 3 / (3 - x) - 4 / (4 - x) + b / (5 - x) = 0) ↔
    (p * x^2 + q * x + r = 0)

theorem quadratic_reducible_conditions :
  ∀ a b : ℚ, is_quadratic_or_reducible a b ↔
    ((a, b) = (1, 2) ∨
     (a, b) = (13/48, 178/48) ∨
     (a, b) = (9/14, 5/2) ∨
     (a, b) = (1/2, 5/2) ∨
     (a, b) = (0, 0)) := by sorry

end NUMINAMATH_CALUDE_quadratic_reducible_conditions_l2858_285848


namespace NUMINAMATH_CALUDE_divisibility_pairs_l2858_285873

def satisfies_condition (a b : ℕ) : Prop :=
  (a + 1) % b = 0 ∧ (b + 1) % a = 0

theorem divisibility_pairs :
  ∀ a b : ℕ, satisfies_condition a b ↔ ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l2858_285873


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2858_285847

theorem sum_of_x_and_y (x y : ℝ) : 
  |x - 2*y - 3| + (y - 2*x)^2 = 0 → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2858_285847


namespace NUMINAMATH_CALUDE_faucet_filling_time_l2858_285832

/-- Given that five faucets fill a 150-gallon tub in 9 minutes,
    prove that ten faucets will fill a 75-gallon tub in 135 seconds. -/
theorem faucet_filling_time 
  (initial_faucets : ℕ) 
  (initial_volume : ℝ) 
  (initial_time : ℝ) 
  (target_faucets : ℕ) 
  (target_volume : ℝ) 
  (h1 : initial_faucets = 5) 
  (h2 : initial_volume = 150) 
  (h3 : initial_time = 9) 
  (h4 : target_faucets = 10) 
  (h5 : target_volume = 75) : 
  (target_volume / target_faucets) * (initial_time / (initial_volume / initial_faucets)) * 60 = 135 := by
  sorry

#check faucet_filling_time

end NUMINAMATH_CALUDE_faucet_filling_time_l2858_285832


namespace NUMINAMATH_CALUDE_A_profit_share_l2858_285862

-- Define the investments and profit shares
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100
def total_profit : ℕ := 12200

-- Theorem to prove A's share of the profit
theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end NUMINAMATH_CALUDE_A_profit_share_l2858_285862


namespace NUMINAMATH_CALUDE_page_number_added_twice_l2858_285899

theorem page_number_added_twice (m : ℕ) (p : ℕ) : 
  m = 71 → 
  1 ≤ p → 
  p ≤ m → 
  (m * (m + 1)) / 2 + p = 2550 → 
  p = 6 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l2858_285899


namespace NUMINAMATH_CALUDE_plate_cup_cost_l2858_285833

/-- Given that 100 plates and 200 cups cost $7.50, prove that 20 plates and 40 cups cost $1.50 -/
theorem plate_cup_cost (plate_rate cup_rate : ℚ) : 
  100 * plate_rate + 200 * cup_rate = (7.5 : ℚ) → 
  20 * plate_rate + 40 * cup_rate = (1.5 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_plate_cup_cost_l2858_285833


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2858_285810

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  h : y = x^2 - 2*x - 3

/-- Predicate for a circle being tangent to x-axis or y-axis -/
def is_tangent_to_axis (p : ParabolaPoint) : Prop :=
  (p.y = 2 ∨ p.y = -2) ∨ (p.x = 2 ∨ p.x = -2)

/-- The set of points where the circle is tangent to an axis -/
def tangent_points : Set ParabolaPoint :=
  { p | is_tangent_to_axis p }

/-- Theorem stating the coordinates of tangent points -/
theorem tangent_point_coordinates :
  ∀ p ∈ tangent_points,
    (p.x = 1 + Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 - Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 + Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 1 - Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 2 ∧ p.y = -3) ∨
    (p.x = -2 ∧ p.y = 5) :=
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2858_285810


namespace NUMINAMATH_CALUDE_class_size_proof_l2858_285835

theorem class_size_proof (total : ℕ) : 
  (1 / 4 : ℚ) * total = total - ((3 / 4 : ℚ) * total) →
  (1 / 3 : ℚ) * ((3 / 4 : ℚ) * total) = ((3 / 4 : ℚ) * total) - 10 →
  10 = (2 / 3 : ℚ) * ((3 / 4 : ℚ) * total) →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l2858_285835


namespace NUMINAMATH_CALUDE_fraction_equality_l2858_285845

theorem fraction_equality : (1 : ℚ) / 2 + (1 : ℚ) / 4 = 9 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2858_285845


namespace NUMINAMATH_CALUDE_expression_simplification_l2858_285817

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  a^2 / (a^2 + 2*a) - (a^2 - 2*a + 1) / (a + 2) / ((a^2 - 1) / (a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2858_285817


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2858_285889

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 8 = 0 ∧ 
    x₂^2 + a*x₂ + 8 = 0 ∧ 
    x₁ - 64/(17*x₂^3) = x₂ - 64/(17*x₁^3)) 
  → a = 12 ∨ a = -12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2858_285889


namespace NUMINAMATH_CALUDE_airplane_speed_l2858_285888

/-- Given a distance and flight times with and against wind, calculate the average speed without wind -/
theorem airplane_speed (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ)
  (h1 : distance = 9360)
  (h2 : time_with_wind = 12)
  (h3 : time_against_wind = 13) :
  ∃ (speed_no_wind : ℝ) (wind_speed : ℝ),
    speed_no_wind = 750 ∧
    time_with_wind * (speed_no_wind + wind_speed) = distance ∧
    time_against_wind * (speed_no_wind - wind_speed) = distance :=
by sorry

end NUMINAMATH_CALUDE_airplane_speed_l2858_285888


namespace NUMINAMATH_CALUDE_final_rope_length_l2858_285893

/-- Represents the weekly rope transactions in feet -/
def weekly_transactions : List ℝ :=
  [6, 18, 14, -9, 8, -1, 3, -10]

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Calculates the total rope length in inches after all transactions -/
def total_rope_length : ℝ :=
  (weekly_transactions.sum * feet_to_inches)

theorem final_rope_length :
  total_rope_length = 348 := by sorry

end NUMINAMATH_CALUDE_final_rope_length_l2858_285893


namespace NUMINAMATH_CALUDE_milk_water_mixture_l2858_285824

theorem milk_water_mixture (total_weight : ℝ) (added_water : ℝ) (new_ratio : ℝ) :
  total_weight = 85 →
  added_water = 5 →
  new_ratio = 3 →
  let initial_water := (total_weight - new_ratio * added_water) / (new_ratio + 1)
  let initial_milk := total_weight - initial_water
  (initial_milk / initial_water) = 27 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l2858_285824


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l2858_285825

theorem power_equality_implies_exponent (q : ℕ) : 27^8 = 9^q → q = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l2858_285825


namespace NUMINAMATH_CALUDE_faulty_key_is_seven_or_nine_l2858_285809

/-- Represents a digit key on a keypad -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a key press was registered or not -/
inductive KeyPress
| registered
| notRegistered

/-- Represents a sequence of ten attempted key presses -/
def AttemptedSequence := Vector Digit 10

/-- Represents the actual registered sequence after pressing keys -/
def RegisteredSequence := Vector Digit 7

/-- Checks if a digit appears at least five times in a sequence -/
def appearsAtLeastFiveTimes (d : Digit) (s : AttemptedSequence) : Prop := sorry

/-- Checks if the registration pattern of a digit matches the faulty key pattern -/
def matchesFaultyPattern (d : Digit) (s : AttemptedSequence) (r : RegisteredSequence) : Prop := sorry

/-- The main theorem stating that the faulty key must be either 7 or 9 -/
theorem faulty_key_is_seven_or_nine
  (attempted : AttemptedSequence)
  (registered : RegisteredSequence)
  (h1 : ∃ (d : Digit), appearsAtLeastFiveTimes d attempted)
  (h2 : ∀ (d : Digit), appearsAtLeastFiveTimes d attempted → matchesFaultyPattern d attempted registered) :
  ∃ (d : Digit), d = Digit.seven ∨ d = Digit.nine :=
sorry

end NUMINAMATH_CALUDE_faulty_key_is_seven_or_nine_l2858_285809


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2858_285816

/-- A cube with volume 5x and surface area x has x equal to 5400 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 5*x ∧ 6*s^2 = x) → x = 5400 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2858_285816


namespace NUMINAMATH_CALUDE_second_player_strategy_exists_first_player_strategy_exists_l2858_285857

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 4-digit number -/
def FourDigitNumber := Fin 10000

/-- The game state, representing the current partially filled subtraction problem -/
structure GameState where
  minuend : FourDigitNumber
  subtrahend : FourDigitNumber

/-- A player's move, either calling out a digit or placing a digit -/
inductive Move
  | CallDigit : Digit → Move
  | PlaceDigit : Digit → Nat → Move

/-- The result of the game -/
def gameResult (finalState : GameState) : Int :=
  (finalState.minuend.val : Int) - (finalState.subtrahend.val : Int)

/-- A strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: There exists a strategy for the second player to keep the difference ≤ 4000 -/
theorem second_player_strategy_exists : 
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≤ 4000 := by sorry

/-- Theorem: There exists a strategy for the first player to keep the difference ≥ 4000 -/
theorem first_player_strategy_exists :
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≥ 4000 := by sorry

end NUMINAMATH_CALUDE_second_player_strategy_exists_first_player_strategy_exists_l2858_285857


namespace NUMINAMATH_CALUDE_english_homework_time_l2858_285829

def total_time : ℕ := 180
def math_time : ℕ := 45
def science_time : ℕ := 50
def history_time : ℕ := 25
def project_time : ℕ := 30

theorem english_homework_time :
  total_time - (math_time + science_time + history_time + project_time) = 30 := by
sorry

end NUMINAMATH_CALUDE_english_homework_time_l2858_285829


namespace NUMINAMATH_CALUDE_equation_solutions_l2858_285860

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧
    3*x₁*(x₁-1) = 2*x₁-2 ∧ 3*x₂*(x₂-1) = 2*x₂-2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2858_285860


namespace NUMINAMATH_CALUDE_exists_distribution_prob_white_gt_two_thirds_l2858_285852

/-- Represents a distribution of balls in two boxes -/
structure BallDistribution :=
  (white_box1 : ℕ)
  (black_box1 : ℕ)
  (white_box2 : ℕ)
  (black_box2 : ℕ)

/-- The total number of white balls -/
def total_white : ℕ := 8

/-- The total number of black balls -/
def total_black : ℕ := 8

/-- Calculates the probability of drawing a white ball given a distribution -/
def prob_white (d : BallDistribution) : ℚ :=
  let p_box1 := (d.white_box1 : ℚ) / (d.white_box1 + d.black_box1 : ℚ)
  let p_box2 := (d.white_box2 : ℚ) / (d.white_box2 + d.black_box2 : ℚ)
  (1/2 : ℚ) * p_box1 + (1/2 : ℚ) * p_box2

/-- Theorem stating that there exists a distribution where the probability of drawing a white ball is greater than 2/3 -/
theorem exists_distribution_prob_white_gt_two_thirds :
  ∃ (d : BallDistribution),
    d.white_box1 + d.white_box2 = total_white ∧
    d.black_box1 + d.black_box2 = total_black ∧
    prob_white d > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_exists_distribution_prob_white_gt_two_thirds_l2858_285852


namespace NUMINAMATH_CALUDE_beehives_for_candles_l2858_285894

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem beehives_for_candles : 
  (3 : ℚ) * 96 / 12 = 24 := by sorry

end NUMINAMATH_CALUDE_beehives_for_candles_l2858_285894


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l2858_285898

/-- Given a square carpet with the following properties:
  * Side length of the carpet is 12 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * S is the side length of the large shaded square
  * T is the side length of each smaller shaded square
  * The ratio 12:S is 4
  * The ratio S:T is 4
  Prove that the total shaded area is 15.75 square feet -/
theorem shaded_area_of_carpet (S T : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 4)
  : S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l2858_285898


namespace NUMINAMATH_CALUDE_prove_a_equals_two_l2858_285870

/-- Given a > 1 and f(x) = a^x + 1, prove that a = 2 if f(2) - f(1) = 2 -/
theorem prove_a_equals_two (a : ℝ) (h1 : a > 1) : 
  (fun x => a^x + 1) 2 - (fun x => a^x + 1) 1 = 2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_prove_a_equals_two_l2858_285870


namespace NUMINAMATH_CALUDE_distinct_roots_rectangle_perimeter_l2858_285823

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + 4*k - 3

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(4*k - 3)

-- Statement 1: The equation always has two distinct real roots
theorem distinct_roots (k : ℝ) : discriminant k > 0 := by sorry

-- Define the sum and product of roots
def sum_of_roots (k : ℝ) : ℝ := 2*k + 1
def product_of_roots (k : ℝ) : ℝ := 4*k - 3

-- Statement 2: When roots represent rectangle sides with diagonal √31, perimeter is 14
theorem rectangle_perimeter (k : ℝ) 
  (h1 : sum_of_roots k^2 + product_of_roots k = 31) 
  (h2 : k > 0) : 
  2 * sum_of_roots k = 14 := by sorry

end NUMINAMATH_CALUDE_distinct_roots_rectangle_perimeter_l2858_285823


namespace NUMINAMATH_CALUDE_black_larger_than_gray_l2858_285891

/-- The gray area of the rectangles -/
def gray_area (a b c : ℝ) : ℝ := (10 - a) + (7 - b) - c

/-- The black area of the rectangles -/
def black_area (a b c : ℝ) : ℝ := (13 - a) - b + (5 - c)

/-- Theorem stating that the black area is larger than the gray area by 1 square unit -/
theorem black_larger_than_gray (a b c : ℝ) : 
  black_area a b c - gray_area a b c = 1 := by sorry

end NUMINAMATH_CALUDE_black_larger_than_gray_l2858_285891


namespace NUMINAMATH_CALUDE_prob_more_ones_than_eights_l2858_285880

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling more 1's than 8's when rolling five fair eight-sided dice -/
def probMoreOnesThanEights : ℚ := 14026 / 32768

/-- Theorem stating that the probability of rolling more 1's than 8's is correct -/
theorem prob_more_ones_than_eights :
  let totalOutcomes : ℕ := numSides ^ numDice
  let probEqualOnesAndEights : ℚ := 4716 / totalOutcomes
  probMoreOnesThanEights = (1 - probEqualOnesAndEights) / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_eights_l2858_285880


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l2858_285855

theorem nested_fraction_simplification :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l2858_285855


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l2858_285820

theorem number_satisfying_equation : ∃ x : ℝ, (0.08 * x) + (0.10 * 40) = 5.92 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l2858_285820


namespace NUMINAMATH_CALUDE_sum_of_y_coefficients_correct_expressions_equal_l2858_285830

/-- The sum of coefficients of terms containing y in (5x+3y+2)(2x+5y+3) -/
def sum_of_y_coefficients : ℤ := 65

/-- The original expression -/
def original_expression (x y : ℚ) : ℚ := (5*x + 3*y + 2) * (2*x + 5*y + 3)

/-- Expanded form of the original expression -/
def expanded_expression (x y : ℚ) : ℚ := 
  10*x^2 + 31*x*y + 19*x + 15*y^2 + 19*y + 6

/-- Theorem stating that the sum of coefficients of terms containing y 
    in the expanded expression is equal to sum_of_y_coefficients -/
theorem sum_of_y_coefficients_correct : 
  (31 : ℤ) + 15 + 19 = sum_of_y_coefficients := by sorry

/-- Theorem stating that the original expression and expanded expression are equal -/
theorem expressions_equal (x y : ℚ) : 
  original_expression x y = expanded_expression x y := by sorry

end NUMINAMATH_CALUDE_sum_of_y_coefficients_correct_expressions_equal_l2858_285830


namespace NUMINAMATH_CALUDE_fifth_decimal_place_of_1_0025_pow_10_l2858_285882

theorem fifth_decimal_place_of_1_0025_pow_10 :
  ∃ (n : ℕ) (r : ℚ), 
    (1 + 1/400)^10 = n + r ∧ 
    n < (1 + 1/400)^10 ∧
    (1 + 1/400)^10 < n + 1 ∧
    (r * 100000).floor = 8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_decimal_place_of_1_0025_pow_10_l2858_285882


namespace NUMINAMATH_CALUDE_population_reproduction_after_development_l2858_285802

/-- Represents the types of population reproduction --/
inductive PopulationReproductionType
  | Primitive
  | Traditional
  | TransitionToModern
  | Modern

/-- Represents the state of society after a major development of productive forces --/
structure SocietyState where
  productiveForcesDeveloped : Bool
  materialWealthIncreased : Bool
  populationGrowthRapid : Bool
  healthCareImproved : Bool
  mortalityRatesDecreased : Bool

/-- Determines the type of population reproduction based on the society state --/
def determinePopulationReproductionType (state : SocietyState) : PopulationReproductionType :=
  if state.productiveForcesDeveloped ∧
     state.materialWealthIncreased ∧
     state.populationGrowthRapid ∧
     state.healthCareImproved ∧
     state.mortalityRatesDecreased
  then PopulationReproductionType.Traditional
  else PopulationReproductionType.Primitive

/-- Theorem stating that after the first major development of productive forces, 
    the population reproduction type was Traditional --/
theorem population_reproduction_after_development 
  (state : SocietyState) 
  (h1 : state.productiveForcesDeveloped)
  (h2 : state.materialWealthIncreased)
  (h3 : state.populationGrowthRapid)
  (h4 : state.healthCareImproved)
  (h5 : state.mortalityRatesDecreased) :
  determinePopulationReproductionType state = PopulationReproductionType.Traditional := by
  sorry

end NUMINAMATH_CALUDE_population_reproduction_after_development_l2858_285802


namespace NUMINAMATH_CALUDE_cat_food_finished_l2858_285867

def daily_consumption : ℚ := 1/4 + 1/6

def total_cans : ℕ := 10

def days_to_finish : ℕ := 15

theorem cat_food_finished :
  (daily_consumption * days_to_finish : ℚ) ≥ total_cans ∧
  (daily_consumption * (days_to_finish - 1) : ℚ) < total_cans := by
  sorry

end NUMINAMATH_CALUDE_cat_food_finished_l2858_285867


namespace NUMINAMATH_CALUDE_unique_solution_m_l2858_285801

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
axiom quadratic_one_solution (a b c : ℝ) : 
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0

/-- The value of m for which 3x^2 - 7x + m = 0 has exactly one solution -/
theorem unique_solution_m : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_m_l2858_285801


namespace NUMINAMATH_CALUDE_f_at_one_l2858_285803

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_at_one_l2858_285803


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2858_285884

/-- Given a cube with surface area 864 square centimeters, its volume is 1728 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (a : ℝ), 
  (6 * a^2 = 864) →  -- Surface area of cube is 864 sq cm
  (a^3 = 1728)       -- Volume of cube is 1728 cubic cm
:= by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2858_285884


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2858_285827

theorem pure_imaginary_product (m : ℝ) : 
  (∃ (z : ℂ), z * z = -1 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).re = 0 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).im ≠ 0) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2858_285827


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2858_285819

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the ternary number 102₃
def ternary_num : ℕ := 11

-- Theorem statement
theorem product_of_binary_and_ternary :
  binary_num * ternary_num = 143 := by sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2858_285819


namespace NUMINAMATH_CALUDE_candy_count_l2858_285896

theorem candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409) 
  (h2 : red = 145) 
  (h3 : total = red + blue) : blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l2858_285896


namespace NUMINAMATH_CALUDE_lcm_of_25_35_50_l2858_285874

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_25_35_50_l2858_285874


namespace NUMINAMATH_CALUDE_math_test_blank_questions_l2858_285866

theorem math_test_blank_questions 
  (total_questions : ℕ) 
  (word_problems : ℕ) 
  (addition_subtraction_problems : ℕ)
  (questions_answered : ℕ) 
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : addition_subtraction_problems = 28)
  (h4 : questions_answered = 38)
  (h5 : word_problems + addition_subtraction_problems = total_questions) :
  total_questions - questions_answered = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_test_blank_questions_l2858_285866


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_l2858_285806

def num_sides : ℕ := 8
def num_dice : ℕ := 5

theorem probability_not_all_same_dice :
  1 - (num_sides : ℚ) / (num_sides ^ num_dice) = 4095 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_all_same_dice_l2858_285806


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2858_285808

theorem bowling_ball_weight (canoe_weight : ℝ) (bowling_ball_weight : ℝ) : 
  canoe_weight = 36 →
  6 * bowling_ball_weight = 4 * canoe_weight →
  bowling_ball_weight = 24 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2858_285808


namespace NUMINAMATH_CALUDE_inequality_solution_l2858_285881

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1/2 ↔ x ∈ Set.Ioc (-8) (-2) ∪ Set.Icc 6 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2858_285881


namespace NUMINAMATH_CALUDE_cut_to_square_iff_perfect_square_l2858_285864

/-- Represents a figure on a grid -/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure -/
inductive Cut
  | Line : Cut

/-- Represents the result of cutting the figure -/
structure CutResult where
  parts : Fin 3 → GridFigure

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Can form a square from the cut parts -/
def can_form_square (cr : CutResult) : Prop :=
  ∃ side : ℕ, (cr.parts 0).area + (cr.parts 1).area + (cr.parts 2).area = side * side

/-- The main theorem: a figure can be cut into three parts to form a square
    if and only if its area is a perfect square -/
theorem cut_to_square_iff_perfect_square (f : GridFigure) :
  (∃ cuts : List Cut, ∃ cr : CutResult, can_form_square cr) ↔ is_perfect_square f.area :=
sorry

end NUMINAMATH_CALUDE_cut_to_square_iff_perfect_square_l2858_285864


namespace NUMINAMATH_CALUDE_sine_shift_right_l2858_285839

/-- The equation of a sine function shifted to the right -/
theorem sine_shift_right (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t)
  let shift : ℝ := π / 3
  let g (t : ℝ) := f (t - shift)
  g x = Real.sin (2 * x - 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_right_l2858_285839


namespace NUMINAMATH_CALUDE_road_length_l2858_285897

theorem road_length (trees : ℕ) (tree_space : ℕ) (between_space : ℕ) : 
  trees = 13 → tree_space = 1 → between_space = 12 → 
  trees * tree_space + (trees - 1) * between_space = 157 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l2858_285897


namespace NUMINAMATH_CALUDE_number_of_boys_who_love_marbles_l2858_285838

def total_marbles : ℕ := 35
def marbles_per_boy : ℕ := 7

theorem number_of_boys_who_love_marbles : 
  total_marbles / marbles_per_boy = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_who_love_marbles_l2858_285838


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2858_285840

theorem smallest_lcm_with_gcd_5 (p q : ℕ) : 
  1000 ≤ p ∧ p < 10000 ∧ 
  1000 ≤ q ∧ q < 10000 ∧ 
  Nat.gcd p q = 5 →
  201000 ≤ Nat.lcm p q ∧ 
  ∃ (p' q' : ℕ), 1000 ≤ p' ∧ p' < 10000 ∧ 
                 1000 ≤ q' ∧ q' < 10000 ∧ 
                 Nat.gcd p' q' = 5 ∧
                 Nat.lcm p' q' = 201000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2858_285840


namespace NUMINAMATH_CALUDE_root_ratio_implies_k_value_l2858_285812

theorem root_ratio_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + k = 0 ∧ s^2 - 4*s + k = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_implies_k_value_l2858_285812


namespace NUMINAMATH_CALUDE_nina_savings_time_l2858_285843

theorem nina_savings_time (video_game_cost : ℝ) (headset_cost : ℝ) (sales_tax_rate : ℝ) 
  (weekly_allowance : ℝ) (savings_rate : ℝ) :
  video_game_cost = 50 →
  headset_cost = 70 →
  sales_tax_rate = 0.12 →
  weekly_allowance = 10 →
  savings_rate = 0.40 →
  ⌈(((video_game_cost + headset_cost) * (1 + sales_tax_rate)) / 
    (weekly_allowance * savings_rate))⌉ = 34 := by
  sorry

end NUMINAMATH_CALUDE_nina_savings_time_l2858_285843


namespace NUMINAMATH_CALUDE_carpenter_logs_l2858_285871

/-- Proves that the carpenter currently has 8 logs given the conditions of the problem -/
theorem carpenter_logs :
  ∀ (total_woodblocks : ℕ) (woodblocks_per_log : ℕ) (additional_logs_needed : ℕ),
    total_woodblocks = 80 →
    woodblocks_per_log = 5 →
    additional_logs_needed = 8 →
    (total_woodblocks - additional_logs_needed * woodblocks_per_log) / woodblocks_per_log = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_carpenter_logs_l2858_285871


namespace NUMINAMATH_CALUDE_number_of_friends_l2858_285842

/-- Given that Mary, Sam, Keith, and Alyssa each have 6 baseball cards,
    prove that the number of friends is 4. -/
theorem number_of_friends : ℕ :=
  let mary_cards := 6
  let sam_cards := 6
  let keith_cards := 6
  let alyssa_cards := 6
  4

#check number_of_friends

end NUMINAMATH_CALUDE_number_of_friends_l2858_285842


namespace NUMINAMATH_CALUDE_function_two_zeros_implies_a_range_l2858_285807

/-- If the function y = x + a/x + 1 has two zeros, then a ∈ (-∞, 1/4) -/
theorem function_two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + a / x₁ + 1 = 0 ∧ x₂ + a / x₂ + 1 = 0) →
  a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_function_two_zeros_implies_a_range_l2858_285807


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2858_285836

theorem quadratic_roots_property (r s : ℝ) : 
  (3 * r^2 - 5 * r - 7 = 0) → 
  (3 * s^2 - 5 * s - 7 = 0) → 
  (r ≠ s) →
  (4 * r^2 - 4 * s^2) / (r - s) = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2858_285836


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l2858_285879

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l2858_285879


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l2858_285841

/-- The number of teams in the conference -/
def num_teams : ℕ := 12

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

/-- Theorem stating the total number of games in a season -/
theorem high_school_twelve_games :
  total_games = 204 :=
sorry

end NUMINAMATH_CALUDE_high_school_twelve_games_l2858_285841


namespace NUMINAMATH_CALUDE_projection_vector_l2858_285853

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj := (a • b) / (a • a) • a
  proj = ![0, 1/2, 1/2] := by sorry

end NUMINAMATH_CALUDE_projection_vector_l2858_285853


namespace NUMINAMATH_CALUDE_train_passing_platform_l2858_285851

/-- Given a train of length 250 meters passing a pole in 10 seconds,
    prove that it takes 60 seconds to pass a platform of length 1250 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_length = 1250) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 60 := by
  sorry

#check train_passing_platform

end NUMINAMATH_CALUDE_train_passing_platform_l2858_285851


namespace NUMINAMATH_CALUDE_book_difference_l2858_285877

theorem book_difference (total : ℕ) (fiction : ℕ) (picture : ℕ)
  (h_total : total = 35)
  (h_fiction : fiction = 5)
  (h_picture : picture = 11)
  (h_autobio : ∃ autobio : ℕ, autobio = 2 * fiction) :
  ∃ nonfiction : ℕ, nonfiction - fiction = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_difference_l2858_285877


namespace NUMINAMATH_CALUDE_largest_expression_l2858_285813

theorem largest_expression : 
  let a := (1 : ℚ) / 2
  let b := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let c := (1 : ℚ) / 4 + (1 : ℚ) / 5 + (1 : ℚ) / 6
  let d := (1 : ℚ) / 5 + (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8
  let e := (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8 + (1 : ℚ) / 9 + (1 : ℚ) / 10
  e > a ∧ e > b ∧ e > c ∧ e > d := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2858_285813


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2858_285892

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2858_285892


namespace NUMINAMATH_CALUDE_angle5_measure_l2858_285886

-- Define the angle measures as real numbers
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom angle1_fraction : angle1 = (1/4) * angle2
axiom supplementary : angle2 + angle5 = 180

-- State the theorem
theorem angle5_measure : angle5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle5_measure_l2858_285886


namespace NUMINAMATH_CALUDE_base4_calculation_l2858_285831

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Multiplication operation for base 4 numbers --/
def mul_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a * to_decimal b)

/-- Division operation for base 4 numbers --/
def div_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a / to_decimal b)

theorem base4_calculation : 
  mul_base4 (div_base4 (to_base4 210) (to_base4 3)) (to_base4 21) = to_base4 1102 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l2858_285831


namespace NUMINAMATH_CALUDE_intersecting_triangles_circumcircle_containment_l2858_285856

/-- A triangle in a plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- Two triangles intersect if they have a common point -/
def intersect (t1 t2 : Triangle) : Prop :=
  sorry

/-- A point is inside or on a circle -/
def inside_or_on_circle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem intersecting_triangles_circumcircle_containment 
  (t1 t2 : Triangle) (h : intersect t1 t2) :
  ∃ (i : Fin 3), inside_or_on_circle (t1.vertices i) (circumcircle t2) ∨
                 inside_or_on_circle (t2.vertices i) (circumcircle t1) :=
sorry

end NUMINAMATH_CALUDE_intersecting_triangles_circumcircle_containment_l2858_285856
