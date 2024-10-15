import Mathlib

namespace NUMINAMATH_CALUDE_girls_average_age_l3597_359780

/-- Proves that the average age of girls is 11 years given the school's statistics -/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 604 →
  boys_avg_age = 12 →
  school_avg_age = 47/4 →
  num_girls = 151 →
  (total_students * school_avg_age - (total_students - num_girls) * boys_avg_age) / num_girls = 11 := by
  sorry


end NUMINAMATH_CALUDE_girls_average_age_l3597_359780


namespace NUMINAMATH_CALUDE_equation_solution_l3597_359702

theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 1) = 3 / (4 - x)) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3597_359702


namespace NUMINAMATH_CALUDE_call_center_ratio_l3597_359755

theorem call_center_ratio (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (6 / 5 : ℚ) * a * b = (3 / 4 : ℚ) * b * b → a / b = (5 / 8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_call_center_ratio_l3597_359755


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l3597_359736

theorem floor_of_expression_equals_eight :
  ⌊(1005^3 : ℝ) / (1003 * 1004) - (1003^3 : ℝ) / (1004 * 1005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l3597_359736


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3597_359700

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3597_359700


namespace NUMINAMATH_CALUDE_twentieth_term_is_96_l3597_359783

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
theorem twentieth_term_is_96 :
  arithmeticSequenceTerm 1 5 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_96_l3597_359783


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3597_359709

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 40 → x * y ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3597_359709


namespace NUMINAMATH_CALUDE_johns_car_efficiency_l3597_359724

/-- Calculates the miles per gallon (MPG) of John's car based on his weekly driving habits. -/
def johns_car_mpg (work_miles_one_way : ℕ) (work_days : ℕ) (leisure_miles : ℕ) (gas_used : ℕ) : ℚ :=
  let total_miles := 2 * work_miles_one_way * work_days + leisure_miles
  total_miles / gas_used

/-- Proves that John's car gets 30 miles per gallon based on his weekly driving habits. -/
theorem johns_car_efficiency :
  johns_car_mpg 20 5 40 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_efficiency_l3597_359724


namespace NUMINAMATH_CALUDE_subset_sum_exists_l3597_359705

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l3597_359705


namespace NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l3597_359787

theorem diophantine_equation_prime_divisor (x y n : ℕ) 
  (h1 : x ≥ 3) (h2 : n ≥ 2) (h3 : x^2 + 5 = y^n) :
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≡ 1 [MOD 4] := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l3597_359787


namespace NUMINAMATH_CALUDE_match_problem_solution_l3597_359751

/-- Represents the number of matches in each pile -/
structure MatchPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the described operations on the piles -/
def performOperations (piles : MatchPiles) : MatchPiles :=
  let step1 := MatchPiles.mk
    (piles.first - piles.second)
    (piles.second + piles.second)
    piles.third
  let step2 := MatchPiles.mk
    step1.first
    (step1.second - step1.third)
    (step1.third + step1.third)
  MatchPiles.mk
    (step2.first + step2.third)
    step2.second
    (step2.third - step2.first)

/-- Theorem stating the solution to the match problem -/
theorem match_problem_solution (piles : MatchPiles) :
  piles.first + piles.second + piles.third = 96 →
  let final := performOperations piles
  final.first = final.second ∧ final.second = final.third →
  piles = MatchPiles.mk 44 28 24 := by
  sorry

end NUMINAMATH_CALUDE_match_problem_solution_l3597_359751


namespace NUMINAMATH_CALUDE_probability_different_specialties_l3597_359760

def total_students : ℕ := 50
def art_students : ℕ := 15
def dance_students : ℕ := 35

theorem probability_different_specialties :
  let total_combinations := total_students.choose 2
  let different_specialty_combinations := art_students * dance_students
  (different_specialty_combinations : ℚ) / total_combinations = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_different_specialties_l3597_359760


namespace NUMINAMATH_CALUDE_f_local_minimum_at_2_l3597_359741

def f (x : ℝ) := x^3 - 3*x^2 + 1

theorem f_local_minimum_at_2 :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_f_local_minimum_at_2_l3597_359741


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3597_359779

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -9/2 < x ∧ x < 7/2 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3597_359779


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l3597_359772

open Set

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def slope_condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 1

theorem odd_function_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_slope : slope_condition f) 
  (h_f1 : f 1 = 1) :
  {x : ℝ | f x - x > 0} = Iio (-1) ∪ Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l3597_359772


namespace NUMINAMATH_CALUDE_star_sqrt_eleven_l3597_359738

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem star_sqrt_eleven : star (Real.sqrt 11) (Real.sqrt 11) = 44 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt_eleven_l3597_359738


namespace NUMINAMATH_CALUDE_negative_difference_equality_l3597_359784

theorem negative_difference_equality (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_equality_l3597_359784


namespace NUMINAMATH_CALUDE_parabola_ellipse_shared_focus_l3597_359714

/-- Given a parabola and an ellipse with shared focus, prove p = 8 -/
theorem parabola_ellipse_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ x y, y^2 = 2*p*x) →  -- parabola equation
  (∃ x y, x^2/(3*p) + y^2/p = 1) →  -- ellipse equation
  (∃ x, x = p/2 ∧ x^2 = p^2/4) →  -- focus of parabola
  (∃ x, x^2 = 3*p^2/4) →  -- focus of ellipse
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_shared_focus_l3597_359714


namespace NUMINAMATH_CALUDE_geometric_sequence_k_value_l3597_359763

/-- A geometric sequence with sum S_n = 3 * 2^n + k -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  k : ℝ
  sum_formula : ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

/-- The value of k in the geometric sequence sum formula -/
theorem geometric_sequence_k_value (seq : GeometricSequence) : seq.k = -3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_k_value_l3597_359763


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3597_359718

/-- The equation of the tangent line to y = ln x + x^2 at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log t + t^2
  let f' : ℝ → ℝ := λ t => 1/t + 2*t
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, 1)
  3*x - y - 2 = 0 ↔ y - point.2 = slope * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3597_359718


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3597_359719

theorem simplify_fraction_product : (320 : ℚ) / 18 * 9 / 144 * 4 / 5 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3597_359719


namespace NUMINAMATH_CALUDE_problem_statement_l3597_359762

def x : ℕ := 18
def y : ℕ := 8
def z : ℕ := 2

theorem problem_statement :
  -- (A) The arithmetic mean of x and y is greater than their geometric mean
  (x + y) / 2 > Real.sqrt (x * y) ∧
  -- (B) The sum of x and z is greater than their product divided by the sum of x and y
  (x + z : ℝ) > (x * z : ℝ) / (x + y) ∧
  -- (C) If the product of x and z is fixed, their sum can be made arbitrarily large
  (∀ ε > 0, ∃ k > 0, k + (x * z : ℝ) / k > 1 / ε) ∧
  -- (D) The arithmetic mean of x, y, and z is NOT greater than the sum of their squares divided by their sum
  ¬((x + y + z : ℝ) / 3 > (x^2 + y^2 + z^2 : ℝ) / (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3597_359762


namespace NUMINAMATH_CALUDE_coca_cola_purchase_l3597_359747

/-- The number of bottles of Coca-Cola to be purchased. -/
def num_bottles : ℕ := 40

/-- The price of each bottle of Coca-Cola in yuan. -/
def price_per_bottle : ℚ := 28/10

/-- The denomination of the banknotes in yuan. -/
def banknote_value : ℕ := 20

/-- The minimum number of banknotes needed to cover the total cost. -/
def min_banknotes : ℕ := 6

theorem coca_cola_purchase (n : ℕ) (p : ℚ) (b : ℕ) :
  n = num_bottles → p = price_per_bottle → b = banknote_value →
  min_banknotes = (n * p / b).ceil := by sorry

end NUMINAMATH_CALUDE_coca_cola_purchase_l3597_359747


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3597_359730

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3597_359730


namespace NUMINAMATH_CALUDE_octal_subtraction_l3597_359748

/-- Converts a base-8 number to base-10 --/
def octalToDecimal (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a base-10 number to base-8 --/
def decimalToOctal (n : ℕ) : ℕ :=
  let tens := n / 8
  let ones := n % 8
  tens * 10 + ones

/-- Proves that 346₈ - 255₈ = 71₈ --/
theorem octal_subtraction : decimalToOctal (octalToDecimal 346 - octalToDecimal 255) = 71 := by
  sorry


end NUMINAMATH_CALUDE_octal_subtraction_l3597_359748


namespace NUMINAMATH_CALUDE_susan_remaining_money_l3597_359790

def susan_fair_spending (initial_amount food_cost : ℕ) : ℕ :=
  let game_cost := 3 * food_cost
  let total_spent := food_cost + game_cost
  initial_amount - total_spent

theorem susan_remaining_money :
  susan_fair_spending 90 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_money_l3597_359790


namespace NUMINAMATH_CALUDE_shift_right_result_l3597_359771

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem shift_right_result :
  let original := LinearFunction.mk 2 4
  let shifted := shift_right original 2
  shifted = LinearFunction.mk 2 0 := by sorry

end NUMINAMATH_CALUDE_shift_right_result_l3597_359771


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3597_359795

/-- Given a line with slope 2 and y-intercept -3, its equation is 2x - y - 3 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (x y : ℝ), 
    (∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ y = m * x + b) →
    2 * x - y - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3597_359795


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l3597_359769

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l3597_359769


namespace NUMINAMATH_CALUDE_trig_identity_l3597_359713

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3597_359713


namespace NUMINAMATH_CALUDE_simplify_fraction_l3597_359745

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3597_359745


namespace NUMINAMATH_CALUDE_polynomial_not_factorable_l3597_359774

theorem polynomial_not_factorable : ¬ ∃ (a b c d : ℤ),
  ∀ (x : ℝ), x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_factorable_l3597_359774


namespace NUMINAMATH_CALUDE_original_average_age_proof_l3597_359788

theorem original_average_age_proof (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 ∧
  new_students = 12 ∧
  new_avg = 32 ∧
  avg_decrease = 6 →
  original_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l3597_359788


namespace NUMINAMATH_CALUDE_parcel_weight_proof_l3597_359740

theorem parcel_weight_proof (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 145)
  (h3 : z + x = 150) :
  x + y + z = 213.5 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_proof_l3597_359740


namespace NUMINAMATH_CALUDE_stephanie_silverware_l3597_359776

/-- The number of types of silverware Stephanie needs to buy -/
def numTypes : ℕ := 4

/-- The initial number of pieces Stephanie plans to buy for each type -/
def initialPlan : ℕ := 5 + 10

/-- The reduction in the number of spoons and butter knives -/
def reductionSpoonsButter : ℕ := 4

/-- The reduction in the number of steak knives -/
def reductionSteak : ℕ := 5

/-- The reduction in the number of forks -/
def reductionForks : ℕ := 3

/-- The total number of silverware pieces Stephanie will buy -/
def totalSilverware : ℕ := 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSteak) + 
  (initialPlan - reductionForks)

theorem stephanie_silverware : totalSilverware = 44 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_silverware_l3597_359776


namespace NUMINAMATH_CALUDE_mat_equation_solution_l3597_359789

theorem mat_equation_solution :
  ∃! x : ℝ, (589 + x) + (544 - x) + 80 * x = 2013 := by
  sorry

end NUMINAMATH_CALUDE_mat_equation_solution_l3597_359789


namespace NUMINAMATH_CALUDE_regular_nonagon_angle_l3597_359756

/-- A regular nonagon inscribed in a circle -/
structure RegularNonagon :=
  (vertices : Fin 9 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 9, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (is_inscribed : ∃ center : ℝ × ℝ, ∀ i : Fin 9, dist center (vertices i) = dist center (vertices 0))

/-- The angle measure between three consecutive vertices of a regular nonagon -/
def angle_measure (n : RegularNonagon) (i : Fin 9) : ℝ :=
  sorry

/-- Theorem: The angle measure between three consecutive vertices of a regular nonagon is 40 degrees -/
theorem regular_nonagon_angle (n : RegularNonagon) (i : Fin 9) :
  angle_measure n i = 40 := by sorry

end NUMINAMATH_CALUDE_regular_nonagon_angle_l3597_359756


namespace NUMINAMATH_CALUDE_min_n_for_integer_sqrt_l3597_359723

theorem min_n_for_integer_sqrt (n : ℕ+) : 
  (∃ k : ℕ, k^2 = 51 + n) → (∀ m : ℕ+, m < n → ¬∃ k : ℕ, k^2 = 51 + m) → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_integer_sqrt_l3597_359723


namespace NUMINAMATH_CALUDE_consecutive_points_length_l3597_359743

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 7) →            -- de = 7
  (c - a = 11) →           -- ac = 11
  (e - a = 20) →           -- ae = 20
  (b - a = 5) :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l3597_359743


namespace NUMINAMATH_CALUDE_large_bulb_cost_l3597_359707

def prove_large_bulb_cost (small_bulbs : ℕ) (large_bulbs : ℕ) (initial_amount : ℕ) (small_bulb_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  small_bulbs = 3 →
  large_bulbs = 1 →
  initial_amount = 60 →
  small_bulb_cost = 8 →
  remaining_amount = 24 →
  (initial_amount - remaining_amount - small_bulbs * small_bulb_cost) / large_bulbs = 12

theorem large_bulb_cost : prove_large_bulb_cost 3 1 60 8 24 := by
  sorry

end NUMINAMATH_CALUDE_large_bulb_cost_l3597_359707


namespace NUMINAMATH_CALUDE_min_sides_rotatable_polygon_l3597_359742

theorem min_sides_rotatable_polygon (n : ℕ) (angle : ℚ) : 
  n > 0 ∧ 
  angle = 50 ∧ 
  (360 / n : ℚ) ∣ angle →
  n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_min_sides_rotatable_polygon_l3597_359742


namespace NUMINAMATH_CALUDE_prime_seven_mod_eight_not_sum_three_squares_l3597_359773

theorem prime_seven_mod_eight_not_sum_three_squares (p : ℕ) (hp : Nat.Prime p) (hm : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a * a + b * b + c * c : ℤ) = p := by
  sorry

end NUMINAMATH_CALUDE_prime_seven_mod_eight_not_sum_three_squares_l3597_359773


namespace NUMINAMATH_CALUDE_adjacent_angles_l3597_359726

theorem adjacent_angles (α β : ℝ) : 
  α + β = 180 →  -- sum of adjacent angles is 180°
  α = β + 30 →   -- one angle is 30° larger than the other
  (α = 105 ∧ β = 75) ∨ (α = 75 ∧ β = 105) := by
sorry

end NUMINAMATH_CALUDE_adjacent_angles_l3597_359726


namespace NUMINAMATH_CALUDE_expression_simplification_l3597_359759

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3597_359759


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l3597_359791

theorem sequence_gcd_property :
  (¬∃(a : ℕ → ℕ), ∀i j, i < j → Nat.gcd (a i + j) (a j + i) = 1) ∧
  (∀p, Prime p ∧ Odd p → ∃(a : ℕ → ℕ), ∀i j, i < j → ¬(p ∣ Nat.gcd (a i + j) (a j + i))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l3597_359791


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3597_359799

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -4 ∧ 
  c = Real.sqrt 53 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 → 
  h + k + a + b = 3 + Real.sqrt 37 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3597_359799


namespace NUMINAMATH_CALUDE_average_headcount_is_11600_l3597_359701

def fall_02_03_headcount : ℕ := 11700
def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600

def average_headcount : ℚ :=
  (fall_02_03_headcount + fall_03_04_headcount + fall_04_05_headcount) / 3

theorem average_headcount_is_11600 :
  average_headcount = 11600 := by sorry

end NUMINAMATH_CALUDE_average_headcount_is_11600_l3597_359701


namespace NUMINAMATH_CALUDE_problem_statement_l3597_359715

theorem problem_statement (θ : ℝ) 
  (h : Real.sin (π / 4 - θ) + Real.cos (π / 4 - θ) = 1 / 5) :
  Real.cos (2 * θ) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3597_359715


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3597_359733

theorem sum_reciprocals_bound (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  9 ≤ (1/a + 1/b + 1/c) ∧ 
  ∀ M : ℝ, ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 1/x + 1/y + 1/z > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3597_359733


namespace NUMINAMATH_CALUDE_altitude_construction_possible_l3597_359794

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a triangle in a plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a construction step -/
inductive ConstructionStep
  | DrawLine (p1 p2 : Point)
  | DrawCircle (center : Point) (through : Point)
  | MarkPoints (points : List Point)

/-- Represents an erasing step -/
structure EraseStep :=
  (points : List Point)

/-- Function to check if a triangle is acute-angled and non-equilateral -/
def isAcuteNonEquilateral (t : Triangle) : Prop := sorry

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Function to construct an altitude of a triangle -/
def constructAltitude (t : Triangle) (v : Point) : Line := sorry

/-- Theorem stating that altitudes can be constructed despite point erasure -/
theorem altitude_construction_possible 
  (t : Triangle) 
  (h_acute : isAcuteNonEquilateral t) :
  ∃ (steps : List ConstructionStep),
    ∀ (erases : List EraseStep),
      (∀ e ∈ erases, e.points.length ≤ 3) →
      ∃ (a b c : Line),
        isPointOnLine t.a a ∧ 
        isPointOnLine t.b b ∧ 
        isPointOnLine t.c c ∧
        a = constructAltitude t t.a ∧
        b = constructAltitude t t.b ∧
        c = constructAltitude t t.c :=
sorry

end NUMINAMATH_CALUDE_altitude_construction_possible_l3597_359794


namespace NUMINAMATH_CALUDE_flour_bags_theorem_l3597_359792

def measurements : List Int := [3, 1, 0, 2, 6, -1, 2, 1, -4, 1]

def standard_weight : Int := 100

theorem flour_bags_theorem (measurements : List Int) (standard_weight : Int) :
  measurements = [3, 1, 0, 2, 6, -1, 2, 1, -4, 1] →
  standard_weight = 100 →
  (∀ m ∈ measurements, |0| ≤ |m|) ∧
  (measurements.sum = 11) ∧
  (measurements.length * standard_weight + measurements.sum = 1011) :=
by sorry

end NUMINAMATH_CALUDE_flour_bags_theorem_l3597_359792


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3597_359770

/-- The polynomial P(x) -/
def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

/-- The derivative of P(x) -/
def P' (x : ℝ) : ℝ := 6*x^5 - 24*x^3 - 12*x^2 + 18*x + 12

/-- The greatest common divisor of P(x) and P'(x) -/
noncomputable def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

/-- The resulting polynomial R(x) -/
def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3597_359770


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3597_359785

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (2 * l + 2 * w = 24) →  -- Perimeter of rectangle
  (l = 2 * w) →  -- Length is twice the width
  (t / w = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3597_359785


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l3597_359749

theorem complex_modulus_sqrt_5 (a b : ℝ) (z : ℂ) : 
  (a + Complex.I)^2 = b * Complex.I → z = a + b * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l3597_359749


namespace NUMINAMATH_CALUDE_sixth_group_52_implies_m_7_l3597_359708

/-- Represents a systematic sampling scheme with the given conditions -/
structure SystematicSampling where
  population : ℕ
  groups : ℕ
  sample_size : ℕ
  first_group_range : Set ℕ
  offset_rule : ℕ → ℕ → ℕ

/-- The specific systematic sampling scheme from the problem -/
def problem_sampling : SystematicSampling :=
  { population := 100
  , groups := 10
  , sample_size := 10
  , first_group_range := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  , offset_rule := λ m k => if m + k < 11 then (m + k - 1) % 10 else (m + k - 11) % 10
  }

/-- The theorem to be proved -/
theorem sixth_group_52_implies_m_7 (s : SystematicSampling) (h : s = problem_sampling) :
  ∃ (m : ℕ), m ∈ s.first_group_range ∧ s.offset_rule m 6 = 2 → m = 7 :=
sorry

end NUMINAMATH_CALUDE_sixth_group_52_implies_m_7_l3597_359708


namespace NUMINAMATH_CALUDE_min_value_theorem_l3597_359725

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 / x + 4 / y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3597_359725


namespace NUMINAMATH_CALUDE_population_change_l3597_359734

theorem population_change (P : ℝ) : 
  P > 0 →
  P * 0.9 * 1.1 * 0.9 = 4455 →
  P = 5000 := by
sorry

end NUMINAMATH_CALUDE_population_change_l3597_359734


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3597_359758

theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3)) →
  (∀ x y : ℝ, x ≥ 1 → x + y ≤ 3 → y ≥ a * (x - 3) → 2 * x + y ≥ 1) →
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧ 2 * x + y = 1) →
  a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3597_359758


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l3597_359764

theorem right_triangle_acute_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α = 90 →           -- one angle is 90° (right angle)
  β = 20 →           -- given angle is 20°
  γ = 70 :=          -- prove that the other acute angle is 70°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l3597_359764


namespace NUMINAMATH_CALUDE_smallest_digit_not_in_odd_units_l3597_359753

def odd_units_digits : Set Nat := {1, 3, 5, 7, 9}

def is_digit (n : Nat) : Prop := n < 10

theorem smallest_digit_not_in_odd_units : 
  (∀ d, is_digit d → d ∉ odd_units_digits → d ≥ 0) ∧ 
  (0 ∉ odd_units_digits) ∧ 
  is_digit 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_not_in_odd_units_l3597_359753


namespace NUMINAMATH_CALUDE_z_values_l3597_359739

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let z := (x - 3)^2 * (x + 4) / (3 * x - 4)
  z = 0 ∨ z = 192 := by sorry

end NUMINAMATH_CALUDE_z_values_l3597_359739


namespace NUMINAMATH_CALUDE_only_t_squared_valid_l3597_359722

-- Define a type for programming statements
inductive ProgramStatement
  | Input (var : String) (value : String)
  | Assignment (var : String) (expr : String)
  | Print (var : String) (value : String)

-- Define a function to check if a statement is valid
def isValidStatement : ProgramStatement → Bool
  | ProgramStatement.Input var value => false  -- INPUT x=3 is not valid
  | ProgramStatement.Assignment "T" "T*T" => true  -- T=T*T is valid
  | ProgramStatement.Assignment var1 var2 => false  -- A=B=2 is not valid
  | ProgramStatement.Print var value => false  -- PRINT A=4 is not valid

-- Theorem stating that only T=T*T is valid among the given statements
theorem only_t_squared_valid :
  (isValidStatement (ProgramStatement.Input "x" "3") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "A" "B=2") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "T" "T*T") = true) ∧
  (isValidStatement (ProgramStatement.Print "A" "4") = false) := by
  sorry


end NUMINAMATH_CALUDE_only_t_squared_valid_l3597_359722


namespace NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l3597_359706

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, 7 * |x - 4*a| + |x - a^2| + 6*x - 3*a ≠ 0) ↔ a < -17 ∨ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l3597_359706


namespace NUMINAMATH_CALUDE_road_length_16_trees_l3597_359778

/-- Calculates the length of a road given the number of trees, space per tree, and space between trees. -/
def roadLength (numTrees : ℕ) (spacePerTree : ℕ) (spaceBetweenTrees : ℕ) : ℕ :=
  numTrees * spacePerTree + (numTrees - 1) * spaceBetweenTrees

/-- Proves that the length of the road with 16 trees, 1 foot per tree, and 9 feet between trees is 151 feet. -/
theorem road_length_16_trees : roadLength 16 1 9 = 151 := by
  sorry

end NUMINAMATH_CALUDE_road_length_16_trees_l3597_359778


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l3597_359735

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l3597_359735


namespace NUMINAMATH_CALUDE_tan_pi_fourth_equals_one_l3597_359765

theorem tan_pi_fourth_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_fourth_equals_one_l3597_359765


namespace NUMINAMATH_CALUDE_lucy_current_fish_l3597_359731

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_current_fish : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_current_fish_l3597_359731


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3597_359796

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3597_359796


namespace NUMINAMATH_CALUDE_alternating_sum_coefficients_l3597_359717

theorem alternating_sum_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^2 * (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -31 :=
by sorry

end NUMINAMATH_CALUDE_alternating_sum_coefficients_l3597_359717


namespace NUMINAMATH_CALUDE_productivity_increase_l3597_359729

theorem productivity_increase (original_hours new_hours : ℝ) 
  (wage_increase : ℝ) (productivity_increase : ℝ) : 
  original_hours = 8 → 
  new_hours = 7 → 
  wage_increase = 0.05 →
  (new_hours / original_hours) * (1 + productivity_increase) = 1 + wage_increase →
  productivity_increase = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_productivity_increase_l3597_359729


namespace NUMINAMATH_CALUDE_ball_probabilities_l3597_359750

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black = p_yellow + 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3597_359750


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3597_359781

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) ∧
  (3 + 1 - 3 / (3 - 1)) / ((3^2 - 4*3 + 4) / (3 - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3597_359781


namespace NUMINAMATH_CALUDE_radio_listening_time_l3597_359793

/-- Calculates the time spent listening to the radio during a flight --/
theorem radio_listening_time (total_flight_time reading_time movie_time dinner_time game_time nap_time : ℕ) :
  total_flight_time = 680 →
  reading_time = 120 →
  movie_time = 240 →
  dinner_time = 30 →
  game_time = 70 →
  nap_time = 180 →
  total_flight_time - (reading_time + movie_time + dinner_time + game_time + nap_time) = 40 :=
by sorry

end NUMINAMATH_CALUDE_radio_listening_time_l3597_359793


namespace NUMINAMATH_CALUDE_beyonce_album_songs_l3597_359744

theorem beyonce_album_songs (singles : ℕ) (albums : ℕ) (songs_per_album : ℕ) (total_songs : ℕ) : 
  singles = 5 → albums = 2 → songs_per_album = 15 → total_songs = 55 → 
  total_songs - (singles + albums * songs_per_album) = 20 := by
sorry

end NUMINAMATH_CALUDE_beyonce_album_songs_l3597_359744


namespace NUMINAMATH_CALUDE_broken_clock_theorem_l3597_359737

/-- Represents the time shown on a clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ

/-- Calculates the time shown on the broken clock after a given number of real minutes --/
def brokenClockTime (startTime : ClockTime) (realMinutes : ℕ) : ClockTime :=
  let totalMinutes := startTime.hours * 60 + startTime.minutes + realMinutes * 5 / 4
  { hours := totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem broken_clock_theorem :
  let startTime := ClockTime.mk 14 0
  let realMinutes := 40
  brokenClockTime startTime realMinutes = ClockTime.mk 14 50 :=
by sorry

end NUMINAMATH_CALUDE_broken_clock_theorem_l3597_359737


namespace NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l3597_359782

theorem outlet_pipe_emptying_time 
  (fill_time_1 : ℝ) 
  (fill_time_2 : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : fill_time_1 = 18) 
  (h2 : fill_time_2 = 30) 
  (h3 : combined_fill_time = 0.06666666666666665) :
  let fill_rate_1 := 1 / fill_time_1
  let fill_rate_2 := 1 / fill_time_2
  let combined_fill_rate := 1 / combined_fill_time
  ∃ (empty_time : ℝ), 
    fill_rate_1 + fill_rate_2 - (1 / empty_time) = combined_fill_rate ∧ 
    empty_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_outlet_pipe_emptying_time_l3597_359782


namespace NUMINAMATH_CALUDE_agri_product_sales_model_l3597_359767

/-- Agricultural product sales model -/
structure AgriProduct where
  cost_price : ℝ
  sales_quantity : ℝ → ℝ
  max_price : ℝ

/-- Daily sales profit function -/
def daily_profit (p : AgriProduct) (x : ℝ) : ℝ :=
  x * (p.sales_quantity x) - p.cost_price * (p.sales_quantity x)

/-- Theorem stating the properties of the agricultural product sales model -/
theorem agri_product_sales_model (p : AgriProduct) 
  (h_cost : p.cost_price = 20)
  (h_quantity : ∀ x, p.sales_quantity x = -2 * x + 80)
  (h_max_price : p.max_price = 30) :
  (∀ x, daily_profit p x = -2 * x^2 + 120 * x - 1600) ∧
  (∃ x, x ≤ p.max_price ∧ daily_profit p x = 150 ∧ x = 25) :=
sorry

end NUMINAMATH_CALUDE_agri_product_sales_model_l3597_359767


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l3597_359704

/-- Given two circles, where the diameter of the first increases by 2π,
    the proportional increase in the circumference of the second is 2π² -/
theorem circle_circumference_increase (d₁ d₂ : ℝ) : 
  let increase_diameter : ℝ := 2 * Real.pi
  let increase_circumference : ℝ → ℝ := λ x => Real.pi * x
  increase_circumference increase_diameter = 2 * Real.pi^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l3597_359704


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3597_359732

/-- Given that y varies inversely as x, prove that if y = 6 when x = 3, then y = 3/2 when x = 12 -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- y varies inversely as x
  y 3 = 6 →                     -- y = 6 when x = 3
  y 12 = 3 / 2 :=               -- y = 3/2 when x = 12
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l3597_359732


namespace NUMINAMATH_CALUDE_martha_blue_butterflies_l3597_359752

/-- Proves that Martha has 4 blue butterflies given the conditions of the problem -/
theorem martha_blue_butterflies :
  ∀ (total blue yellow black : ℕ),
    total = 11 →
    black = 5 →
    blue = 2 * yellow →
    total = blue + yellow + black →
    blue = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_blue_butterflies_l3597_359752


namespace NUMINAMATH_CALUDE_no_intersection_l3597_359711

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no intersection points
theorem no_intersection :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l3597_359711


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3597_359746

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (x₀ y₀ x y : ℝ) : Prop := x₀ * x + y₀ * y = 4

def PointOutsideCircle (x₀ y₀ : ℝ) : Prop := x₀^2 + y₀^2 > 4

theorem line_intersects_circle (x₀ y₀ : ℝ) 
  (h1 : PointOutsideCircle x₀ y₀) :
  ∃ x y : ℝ, Circle x y ∧ Line x₀ y₀ x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3597_359746


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3597_359712

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (height : ℝ) 
  (stripe_width : ℝ) 
  (h_diameter : diameter = 30) 
  (h_height : height = 80) 
  (h_stripe_width : stripe_width = 3) :
  stripe_width * height = 240 := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3597_359712


namespace NUMINAMATH_CALUDE_triangle_theorem_l3597_359766

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A > Real.sin t.C)
  (h2 : t.a * t.c * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3597_359766


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3597_359721

/-- The curve defined by r = 8 tan(θ)cos(θ) in polar coordinates is a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∃ (x₀ y₀ R : ℝ), ∀ (θ : ℝ) (r : ℝ),
    r = 8 * Real.tan θ * Real.cos θ →
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = R^2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3597_359721


namespace NUMINAMATH_CALUDE_equation_root_interval_l3597_359761

-- Define the function f(x) = lg(x+1) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2 + x - 3

-- State the theorem
theorem equation_root_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 ∧
  ∀ (k : ℤ), (∃ (y : ℝ), y ∈ Set.Ioo (k : ℝ) (k + 1) ∧ f y = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_interval_l3597_359761


namespace NUMINAMATH_CALUDE_max_value_expression_l3597_359777

theorem max_value_expression (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (a - b^2) * (b - a^2) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3597_359777


namespace NUMINAMATH_CALUDE_tiling_condition_l3597_359754

/-- A tile is represented by its dimensions -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- A grid is represented by its side length -/
structure Grid :=
  (side : ℕ)

/-- Predicate to check if a grid can be tiled by a given tile -/
def can_be_tiled (g : Grid) (t : Tile) : Prop :=
  ∃ (k : ℕ), g.side = k * t.length ∧ g.side * g.side = k * k * (t.length * t.width)

/-- The main theorem stating the condition for tiling an n×n grid with 4×1 tiles -/
theorem tiling_condition (n : ℕ) :
  (∃ (g : Grid) (t : Tile), g.side = n ∧ t.length = 4 ∧ t.width = 1 ∧ can_be_tiled g t) ↔ 
  (∃ (k : ℕ), n = 4 * k) :=
sorry

end NUMINAMATH_CALUDE_tiling_condition_l3597_359754


namespace NUMINAMATH_CALUDE_sleeves_weight_addition_l3597_359775

theorem sleeves_weight_addition (raw_squat : ℝ) (wrap_percentage : ℝ) (wrap_sleeve_difference : ℝ) 
  (h1 : raw_squat = 600)
  (h2 : wrap_percentage = 0.25)
  (h3 : wrap_sleeve_difference = 120) :
  let squat_with_wraps := raw_squat + wrap_percentage * raw_squat
  let squat_with_sleeves := squat_with_wraps - wrap_sleeve_difference
  squat_with_sleeves - raw_squat = 30 := by
sorry

end NUMINAMATH_CALUDE_sleeves_weight_addition_l3597_359775


namespace NUMINAMATH_CALUDE_total_students_surveyed_l3597_359728

theorem total_students_surveyed :
  let french_and_english : ℕ := 25
  let french_not_english : ℕ := 65
  let percent_not_french : ℚ := 55/100
  let total_students : ℕ := 200
  (french_and_english + french_not_english : ℚ) / total_students = 1 - percent_not_french :=
by sorry

end NUMINAMATH_CALUDE_total_students_surveyed_l3597_359728


namespace NUMINAMATH_CALUDE_vehicle_speeds_l3597_359727

/-- Proves that given the conditions, the bus speed is 20 km/h and the car speed is 60 km/h. -/
theorem vehicle_speeds 
  (distance : ℝ) 
  (bus_delay : ℝ) 
  (car_arrival_delay : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : distance = 70) 
  (h2 : bus_delay = 3) 
  (h3 : car_arrival_delay = 2/3) 
  (h4 : speed_ratio = 3) : 
  ∃ (bus_speed car_speed : ℝ), 
    bus_speed = 20 ∧ 
    car_speed = 60 ∧ 
    distance / bus_speed = distance / car_speed + bus_delay - car_arrival_delay ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l3597_359727


namespace NUMINAMATH_CALUDE_product_evaluation_l3597_359716

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3597_359716


namespace NUMINAMATH_CALUDE_water_usage_median_and_mode_l3597_359703

def water_usage : List ℝ := [7, 5, 6, 8, 9, 9, 10]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem water_usage_median_and_mode :
  median water_usage = 8 ∧ mode water_usage = 9 := by sorry

end NUMINAMATH_CALUDE_water_usage_median_and_mode_l3597_359703


namespace NUMINAMATH_CALUDE_mancino_garden_length_l3597_359757

theorem mancino_garden_length :
  ∀ (L : ℝ),
  (3 * L * 5 + 2 * 8 * 4 = 304) →
  L = 16 := by
sorry

end NUMINAMATH_CALUDE_mancino_garden_length_l3597_359757


namespace NUMINAMATH_CALUDE_parabola_directrix_l3597_359768

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 4x + 4) / 8, its directrix is y = -2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) : 
  p.a = 1/8 ∧ p.b = -1/2 ∧ p.c = 1/2 → d.y = -2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_directrix_l3597_359768


namespace NUMINAMATH_CALUDE_disrespectful_quadratic_max_sum_at_one_l3597_359786

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  t : ℝ
  k : ℝ

/-- The value of the polynomial at x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 - p.t * x + p.k

/-- The composition of the polynomial with itself -/
def QuadraticPolynomial.compose (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.eval (p.eval x)

/-- A quadratic polynomial is disrespectful if p(p(x)) = 0 has exactly four real solutions -/
def QuadraticPolynomial.isDisrespectful (p : QuadraticPolynomial) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (∀ x : ℝ, p.compose x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The sum of coefficients of a quadratic polynomial -/
def QuadraticPolynomial.sumCoefficients (p : QuadraticPolynomial) : ℝ :=
  1 - p.t + p.k

/-- The theorem to be proved -/
theorem disrespectful_quadratic_max_sum_at_one :
  ∃ (p : QuadraticPolynomial),
    p.isDisrespectful ∧
    (∀ q : QuadraticPolynomial, q.isDisrespectful → p.sumCoefficients ≥ q.sumCoefficients) ∧
    p.eval 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_disrespectful_quadratic_max_sum_at_one_l3597_359786


namespace NUMINAMATH_CALUDE_hyperbola_center_correct_l3597_359798

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y + 8)^2 / 16^2 - (5 * x - 15)^2 / 9^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, -2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_correct_l3597_359798


namespace NUMINAMATH_CALUDE_chip_division_percentage_l3597_359797

theorem chip_division_percentage (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) 
  (h_total : total_chips = 100)
  (h_ratio : ratio_small + ratio_large = 10)
  (h_ratio_order : ratio_large > ratio_small)
  (h_ratio_large : ratio_large = 6) :
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chip_division_percentage_l3597_359797


namespace NUMINAMATH_CALUDE_probability_is_four_twentyfirsts_l3597_359710

/-- Represents a person with a unique age --/
structure Person :=
  (age : ℕ)

/-- The set of all possible orderings of people leaving the meeting --/
def Orderings : Type := List Person

/-- Checks if the youngest person leaves before the oldest in a given ordering --/
def youngest_before_oldest (ordering : Orderings) : Prop :=
  sorry

/-- Checks if the 3rd, 4th, and 5th people in the ordering are in ascending age order --/
def middle_three_ascending (ordering : Orderings) : Prop :=
  sorry

/-- The set of all valid orderings (where youngest leaves before oldest) --/
def valid_orderings (people : Finset Person) : Finset Orderings :=
  sorry

/-- The probability of the event occurring --/
def probability (people : Finset Person) : ℚ :=
  sorry

theorem probability_is_four_twentyfirsts 
  (people : Finset Person) 
  (h1 : people.card = 7) 
  (h2 : ∀ p q : Person, p ∈ people → q ∈ people → p ≠ q → p.age ≠ q.age) : 
  probability people = 4 / 21 :=
sorry

end NUMINAMATH_CALUDE_probability_is_four_twentyfirsts_l3597_359710


namespace NUMINAMATH_CALUDE_optimal_allocation_l3597_359720

/-- Represents the advertising problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents an advertising allocation --/
structure Allocation where
  timeA : ℝ
  timeB : ℝ

/-- Calculates the total revenue for a given allocation --/
def totalRevenue (p : AdvertisingProblem) (a : Allocation) : ℝ :=
  p.revenueA * a.timeA + p.revenueB * a.timeB

/-- Checks if an allocation is valid given the problem constraints --/
def isValidAllocation (p : AdvertisingProblem) (a : Allocation) : Prop :=
  a.timeA ≥ 0 ∧ a.timeB ≥ 0 ∧
  a.timeA + a.timeB ≤ p.totalTime ∧
  p.rateA * a.timeA + p.rateB * a.timeB ≤ p.totalBudget

/-- The main theorem stating that the given allocation maximizes revenue --/
theorem optimal_allocation (p : AdvertisingProblem) 
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (a : Allocation),
    isValidAllocation p a ∧
    totalRevenue p a = 70 ∧
    ∀ (b : Allocation), isValidAllocation p b → totalRevenue p b ≤ totalRevenue p a :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l3597_359720
