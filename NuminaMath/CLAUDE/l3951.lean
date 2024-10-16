import Mathlib

namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l3951_395185

theorem average_of_quadratic_solutions (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (x₁ + x₂) / 2 = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l3951_395185


namespace NUMINAMATH_CALUDE_sample_survey_suitability_l3951_395116

-- Define the set of all surveys
def Surveys : Set Nat := {1, 2, 3, 4}

-- Define the characteristics of each survey
def is_destructive_testing (s : Nat) : Prop :=
  s = 1 ∨ s = 4

def has_large_scope (s : Nat) : Prop :=
  s = 2

def has_small_scope (s : Nat) : Prop :=
  s = 3

-- Define what makes a survey suitable for sampling
def suitable_for_sampling (s : Nat) : Prop :=
  is_destructive_testing s ∨ has_large_scope s

-- Theorem to prove
theorem sample_survey_suitability :
  {s ∈ Surveys | suitable_for_sampling s} = {1, 2, 4} := by
  sorry


end NUMINAMATH_CALUDE_sample_survey_suitability_l3951_395116


namespace NUMINAMATH_CALUDE_unique_rearrangement_difference_l3951_395119

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    max a (max b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + min a (min b c) -
    (min a (min b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + max a (max b c)) = n

theorem unique_rearrangement_difference :
  ∃! n : ℕ, is_valid_number n :=
by sorry

end NUMINAMATH_CALUDE_unique_rearrangement_difference_l3951_395119


namespace NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3951_395160

theorem original_price_of_discounted_shoes 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : purchase_price = 51)
  (h2 : discount_percentage = 75) : 
  purchase_price / (1 - discount_percentage / 100) = 204 := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3951_395160


namespace NUMINAMATH_CALUDE_xyz_absolute_value_l3951_395132

theorem xyz_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by sorry

end NUMINAMATH_CALUDE_xyz_absolute_value_l3951_395132


namespace NUMINAMATH_CALUDE_largest_last_digit_is_two_l3951_395103

/-- A string of digits satisfying the given conditions -/
structure SpecialString :=
  (digits : Fin 1003 → Nat)
  (first_digit : digits 0 = 2)
  (consecutive_divisible : ∀ i : Fin 1002, 
    (digits i * 10 + digits (i.succ)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i.succ)) % 23 = 0)

/-- The largest possible last digit in the special string -/
def largest_last_digit : Nat := 2

/-- Theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two :
  ∀ s : SpecialString, s.digits 1002 ≤ largest_last_digit :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_is_two_l3951_395103


namespace NUMINAMATH_CALUDE_min_value_theorem_l3951_395162

/-- Given positive real numbers a, b, c, and a function f with minimum value 2,
    prove that a + b + c = 2 and the minimum value of 1/a + 1/b + 1/c is 9/2 -/
theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 2) :
  (a + b + c = 2) ∧ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → 1/x + 1/y + 1/z ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3951_395162


namespace NUMINAMATH_CALUDE_polygon_exterior_interior_angles_equal_l3951_395187

theorem polygon_exterior_interior_angles_equal (n : ℕ) : 
  (n ≥ 3) → (360 = (n - 2) * 180) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_interior_angles_equal_l3951_395187


namespace NUMINAMATH_CALUDE_unoccupied_volume_correct_l3951_395152

/-- Represents the dimensions of a cube in inches -/
structure CubeDimensions where
  side : ℝ

/-- Calculates the volume of a cube given its dimensions -/
def cubeVolume (d : CubeDimensions) : ℝ := d.side ^ 3

/-- Represents the container and its contents -/
structure Container where
  dimensions : CubeDimensions
  waterFillRatio : ℝ
  iceCubes : ℕ
  iceCubeDimensions : CubeDimensions

/-- Calculates the unoccupied volume in the container -/
def unoccupiedVolume (c : Container) : ℝ :=
  let containerVolume := cubeVolume c.dimensions
  let waterVolume := c.waterFillRatio * containerVolume
  let iceCubeVolume := cubeVolume c.iceCubeDimensions
  let totalIceVolume := c.iceCubes * iceCubeVolume
  containerVolume - waterVolume - totalIceVolume

/-- The main theorem to prove -/
theorem unoccupied_volume_correct (c : Container) : 
  c.dimensions.side = 12 ∧ 
  c.waterFillRatio = 3/4 ∧ 
  c.iceCubes = 6 ∧ 
  c.iceCubeDimensions.side = 1.5 → 
  unoccupiedVolume c = 411.75 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_volume_correct_l3951_395152


namespace NUMINAMATH_CALUDE_intersection_angle_implies_ratio_l3951_395198

-- Define the ellipse and hyperbola
def is_on_ellipse (x y a₁ b₁ : ℝ) : Prop := x^2 / a₁^2 + y^2 / b₁^2 = 1
def is_on_hyperbola (x y a₂ b₂ : ℝ) : Prop := x^2 / a₂^2 - y^2 / b₂^2 = 1

-- Define the common foci
def are_common_foci (F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop := 
  ∃ c : ℝ, c^2 = a₁^2 - b₁^2 ∧ c^2 = a₂^2 + b₂^2 ∧
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the angle between foci
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem intersection_angle_implies_ratio 
  (P F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  is_on_ellipse P.1 P.2 a₁ b₁ →
  is_on_hyperbola P.1 P.2 a₂ b₂ →
  are_common_foci F₁ F₂ a₁ b₁ a₂ b₂ →
  angle_F₁PF₂ P F₁ F₂ = π / 3 →
  b₁ / b₂ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_angle_implies_ratio_l3951_395198


namespace NUMINAMATH_CALUDE_candy_cost_theorem_l3951_395177

def candy_problem (caramel_price : ℚ) : Prop :=
  let candy_bar_price := 2 * caramel_price
  let cotton_candy_price := 2 * candy_bar_price
  6 * candy_bar_price + 3 * caramel_price + cotton_candy_price = 57

theorem candy_cost_theorem : candy_problem 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_theorem_l3951_395177


namespace NUMINAMATH_CALUDE_triangle_inequality_l3951_395145

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3951_395145


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3951_395107

theorem quadratic_solution_sum (a b : ℕ+) : 
  (∃ x : ℝ, x^2 + 16*x = 96 ∧ x = Real.sqrt a - b) → a + b = 168 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3951_395107


namespace NUMINAMATH_CALUDE_probability_two_dice_rolls_l3951_395166

-- Define the number of sides on each die
def sides : ℕ := 8

-- Define the favorable outcomes for the first die (numbers less than 4)
def favorable_first : ℕ := 3

-- Define the favorable outcomes for the second die (numbers greater than 5)
def favorable_second : ℕ := 3

-- State the theorem
theorem probability_two_dice_rolls : 
  (favorable_first / sides) * (favorable_second / sides) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_dice_rolls_l3951_395166


namespace NUMINAMATH_CALUDE_satisfactory_grades_fraction_l3951_395101

theorem satisfactory_grades_fraction :
  let grades := [3, 7, 4, 2, 4]  -- A, B, C, D, E+F
  let satisfactory := 4  -- Number of satisfactory grade categories (A, B, C, D)
  let total_students := grades.sum
  let satisfactory_students := (grades.take satisfactory).sum
  (satisfactory_students : ℚ) / total_students = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_grades_fraction_l3951_395101


namespace NUMINAMATH_CALUDE_remainders_of_65_powers_l3951_395140

theorem remainders_of_65_powers (n : ℕ) : 
  (65^(6*n) % 9 = 1) ∧ 
  (65^(6*n + 1) % 9 = 2) ∧ 
  (65^(6*n + 2) % 9 = 4) ∧ 
  (65^(6*n + 3) % 9 = 8) := by
sorry

end NUMINAMATH_CALUDE_remainders_of_65_powers_l3951_395140


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l3951_395178

def is_valid_digit (d : ℕ) : Prop := d ∈ ({2, 3, 4, 5, 6} : Set ℕ)

def digits_to_number (p q r s t : ℕ) : ℕ := p * 10000 + q * 1000 + r * 100 + s * 10 + t

theorem five_digit_divisibility (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t →
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
  (p * 100 + q * 10 + r) % 6 = 0 →
  (q * 100 + r * 10 + s) % 3 = 0 →
  (r * 100 + s * 10 + t) % 9 = 0 →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l3951_395178


namespace NUMINAMATH_CALUDE_solve_for_q_l3951_395120

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m - n)/66)
  (h3 : 5/6 = (q - m)/150) : 
  q = 230 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3951_395120


namespace NUMINAMATH_CALUDE_apartments_on_more_floors_proof_l3951_395193

/-- Represents the number of apartments on a floor with more apartments -/
def apartments_on_more_floors : ℕ := 6

/-- Represents the total number of floors in the building -/
def total_floors : ℕ := 12

/-- Represents the number of apartments on floors with fewer apartments -/
def apartments_on_fewer_floors : ℕ := 5

/-- Represents the maximum number of residents per apartment -/
def max_residents_per_apartment : ℕ := 4

/-- Represents the maximum total number of residents in the building -/
def max_total_residents : ℕ := 264

theorem apartments_on_more_floors_proof :
  let floors_with_more := total_floors / 2
  let floors_with_fewer := total_floors / 2
  let total_apartments_fewer := floors_with_fewer * apartments_on_fewer_floors
  let total_apartments := max_total_residents / max_residents_per_apartment
  let apartments_on_more_total := total_apartments - total_apartments_fewer
  apartments_on_more_floors = apartments_on_more_total / floors_with_more :=
by
  sorry

#check apartments_on_more_floors_proof

end NUMINAMATH_CALUDE_apartments_on_more_floors_proof_l3951_395193


namespace NUMINAMATH_CALUDE_evaluate_expression_l3951_395168

theorem evaluate_expression : (728 * 728) - (727 * 729) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3951_395168


namespace NUMINAMATH_CALUDE_least_distinct_values_l3951_395157

/-- Given a list of 2023 positive integers with a unique mode occurring exactly 15 times,
    the least number of distinct values in the list is 145. -/
theorem least_distinct_values (l : List ℕ+) 
  (h_length : l.length = 2023)
  (h_unique_mode : ∃! m : ℕ+, l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) :
  (l.toFinset.card : ℕ) = 145 ∧ 
  ∀ k : ℕ, k < 145 → ¬∃ l' : List ℕ+, 
    l'.length = 2023 ∧ 
    (∃! m : ℕ+, l'.count m = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15) ∧
    (l'.toFinset.card : ℕ) = k :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3951_395157


namespace NUMINAMATH_CALUDE_divya_age_l3951_395169

theorem divya_age (divya_age nacho_age : ℝ) : 
  nacho_age + 5 = 3 * (divya_age + 5) →
  nacho_age + divya_age = 40 →
  divya_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_divya_age_l3951_395169


namespace NUMINAMATH_CALUDE_triangle_identity_l3951_395126

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Ensure positive side lengths
  ha : a > 0
  hb : b > 0
  hc : c > 0
  -- Ensure angles are in (0, π)
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Ensure angles sum to π
  hsum : A + B + C = π

-- State the theorem
theorem triangle_identity (t : Triangle) :
  t.a^2 * Real.sin (2 * t.B) + t.b^2 * Real.sin (2 * t.A) = 2 * t.a * t.b * Real.sin t.C :=
by sorry

end NUMINAMATH_CALUDE_triangle_identity_l3951_395126


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3951_395173

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3951_395173


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3951_395194

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3951_395194


namespace NUMINAMATH_CALUDE_mystery_discount_rate_l3951_395137

theorem mystery_discount_rate 
  (biography_price : ℝ) 
  (mystery_price : ℝ) 
  (biography_count : ℕ) 
  (mystery_count : ℕ) 
  (total_savings : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : biography_count = 5)
  (h4 : mystery_count = 3)
  (h5 : total_savings = 19)
  (h6 : total_discount_rate = 0.43)
  : ∃ (biography_discount : ℝ) (mystery_discount : ℝ),
    biography_discount + mystery_discount = total_discount_rate ∧ 
    mystery_discount = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_mystery_discount_rate_l3951_395137


namespace NUMINAMATH_CALUDE_total_run_duration_l3951_395171

/-- Calculates the total duration of a run given two segments with different speeds and distances. -/
theorem total_run_duration 
  (speed1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : speed1 = 18) 
  (h2 : time1 = 3) 
  (h3 : distance2 = 70) 
  (h4 : speed2 = 14) : 
  time1 + distance2 / speed2 = 8 := by
  sorry

#check total_run_duration

end NUMINAMATH_CALUDE_total_run_duration_l3951_395171


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l3951_395197

/-- Given a manuscript typing service with the following conditions:
    - 100 total pages
    - 30 pages revised once
    - 20 pages revised twice
    - $5 per page for initial typing
    - $780 total cost
    Prove that the cost per page for each revision is $4. -/
theorem manuscript_revision_cost :
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let initial_cost_per_page : ℚ := 5
  let total_cost : ℚ := 780
  let revision_cost_per_page : ℚ := 4
  (total_pages * initial_cost_per_page + 
   (pages_revised_once * revision_cost_per_page + 
    pages_revised_twice * 2 * revision_cost_per_page) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l3951_395197


namespace NUMINAMATH_CALUDE_find_missing_number_l3951_395167

theorem find_missing_number (x : ℕ) : 
  (55 + 48 + x + 2 + 684 + 42) / 6 = 223 → x = 507 := by
  sorry

end NUMINAMATH_CALUDE_find_missing_number_l3951_395167


namespace NUMINAMATH_CALUDE_large_envelopes_count_l3951_395153

theorem large_envelopes_count (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ) : 
  total_letters = 80 →
  small_envelope_letters = 20 →
  letters_per_large_envelope = 2 →
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 := by
sorry

end NUMINAMATH_CALUDE_large_envelopes_count_l3951_395153


namespace NUMINAMATH_CALUDE_logic_statements_correctness_l3951_395130

theorem logic_statements_correctness :
  ∃! (n : Nat), n = 2 ∧
  (((∀ p q, p ∧ q → p ∨ q) ∧ (∃ p q, p ∨ q ∧ ¬(p ∧ q))) ∧
   ((∃ p q, ¬(p ∧ q) ∧ ¬(p ∨ q)) ∨ (∀ p q, p ∨ q → ¬(p ∧ q))) ∧
   ((∀ p q, ¬p → p ∨ q) ∧ (∃ p q, p ∨ q ∧ p)) ∧
   ((∀ p q, ¬p → ¬(p ∧ q)) ∧ (∃ p q, ¬(p ∧ q) ∧ p))) :=
by sorry

end NUMINAMATH_CALUDE_logic_statements_correctness_l3951_395130


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3951_395110

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4)
  (h2 : (2*m + n) / 2 = 5) :
  (m + n) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3951_395110


namespace NUMINAMATH_CALUDE_probability_three_defective_l3951_395143

/-- Represents the probability of selecting a defective smartphone from a category. -/
structure CategoryProbability where
  total : ℕ
  defective : ℕ
  probability : ℚ
  valid : probability = defective / total

/-- Represents the data for the smartphone shipment. -/
structure ShipmentData where
  premium : CategoryProbability
  standard : CategoryProbability
  basic : CategoryProbability

/-- The probability of selecting three defective smartphones, one from each category. -/
def probabilityAllDefective (data : ShipmentData) : ℚ :=
  data.premium.probability * data.standard.probability * data.basic.probability

/-- The given shipment data. -/
def givenShipment : ShipmentData := {
  premium := { total := 120, defective := 26, probability := 26 / 120, valid := by norm_num }
  standard := { total := 160, defective := 68, probability := 68 / 160, valid := by norm_num }
  basic := { total := 60, defective := 30, probability := 30 / 60, valid := by norm_num }
}

/-- Theorem stating that the probability of selecting three defective smartphones
    is equal to 221 / 4800 for the given shipment data. -/
theorem probability_three_defective :
  probabilityAllDefective givenShipment = 221 / 4800 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_defective_l3951_395143


namespace NUMINAMATH_CALUDE_red_lettuce_cost_l3951_395106

/-- The amount spent on red lettuce given the total cost and cost of green lettuce -/
def amount_spent_on_red_lettuce (total_cost green_cost : ℕ) : ℕ :=
  total_cost - green_cost

/-- Proof that the amount spent on red lettuce is $6 -/
theorem red_lettuce_cost : amount_spent_on_red_lettuce 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_lettuce_cost_l3951_395106


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3951_395147

theorem quadratic_inequality_solution_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3951_395147


namespace NUMINAMATH_CALUDE_power_of_negative_product_l3951_395142

theorem power_of_negative_product (a b : ℝ) : (-a^3 * b^5)^2 = a^6 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l3951_395142


namespace NUMINAMATH_CALUDE_olivias_cookie_baggies_l3951_395175

def cookies_per_baggie : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

theorem olivias_cookie_baggies :
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cookie_baggies_l3951_395175


namespace NUMINAMATH_CALUDE_average_speed_round_trip_budapest_debrecen_average_speed_l3951_395155

/-- The average speed of a round trip between two cities, given the speeds for each direction. -/
theorem average_speed_round_trip (s : ℝ) (v1 v2 : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) :
  let t1 := s / v1
  let t2 := s / v2
  let total_time := t1 + t2
  let total_distance := 2 * s
  total_distance / total_time = 2 * v1 * v2 / (v1 + v2) :=
by sorry

/-- The average speed of a car traveling between Budapest and Debrecen. -/
theorem budapest_debrecen_average_speed :
  let v1 := 56 -- km/h
  let v2 := 72 -- km/h
  let avg_speed := 2 * v1 * v2 / (v1 + v2)
  avg_speed = 63 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_budapest_debrecen_average_speed_l3951_395155


namespace NUMINAMATH_CALUDE_digits_of_3_pow_10_times_5_pow_6_l3951_395199

theorem digits_of_3_pow_10_times_5_pow_6 :
  (Nat.digits 10 (3^10 * 5^6)).length = 9 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_10_times_5_pow_6_l3951_395199


namespace NUMINAMATH_CALUDE_angle_conversion_and_coterminal_l3951_395127

-- Define α in degrees
def α : ℝ := 1680

-- Theorem statement
theorem angle_conversion_and_coterminal (α : ℝ) :
  ∃ (k : ℤ) (β : ℝ), 
    (α * π / 180 = 2 * k * π + β) ∧ 
    (0 ≤ β) ∧ (β < 2 * π) ∧
    (∃ (θ : ℝ), 
      (θ = -8 * π / 3) ∧ 
      (-4 * π < θ) ∧ (θ < -2 * π) ∧
      (∃ (m : ℤ), θ = 2 * m * π + β)) := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_and_coterminal_l3951_395127


namespace NUMINAMATH_CALUDE_equation_identity_l3951_395112

theorem equation_identity (x : ℝ) : (3*x - 2)*(2*x + 5) - x = 6*x^2 + 2*(5*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l3951_395112


namespace NUMINAMATH_CALUDE_euler_line_parallel_l3951_395161

/-- Triangle ABC with vertices A(-3,0), B(3,0), and C(3,3) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨-3, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩}

/-- The Euler line of a triangle -/
def euler_line (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- A line with equation ax + (a^2 - 3)y - 9 = 0 -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + (a^2 - 3) * p.2 - 9 = 0}

/-- Two lines are parallel -/
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem euler_line_parallel :
  ∀ a : ℝ, parallel (line_l a) (euler_line triangle_ABC) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_euler_line_parallel_l3951_395161


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3951_395122

theorem quadratic_equation_roots
  (a b c : ℝ)
  (h : (a - c)^2 > a^2 + c^2) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3951_395122


namespace NUMINAMATH_CALUDE_joes_journey_time_l3951_395115

/-- Represents the problem of Joe's journey to school -/
theorem joes_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
  r_w > 0 →
  3 * r_w = 3 * d / 4 →
  (3 + 1 / 4 : ℝ) = 3 + (d / 4) / (4 * r_w) :=
by sorry

end NUMINAMATH_CALUDE_joes_journey_time_l3951_395115


namespace NUMINAMATH_CALUDE_no_solutions_prime_factorial_inequality_l3951_395114

theorem no_solutions_prime_factorial_inequality :
  ¬ ∃ (n k : ℕ), Prime n ∧ n ≤ n! - k^n ∧ n! - k^n ≤ k * n :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_prime_factorial_inequality_l3951_395114


namespace NUMINAMATH_CALUDE_determinant_max_value_l3951_395134

theorem determinant_max_value :
  let f : ℝ → ℝ := λ θ => 2 * Real.sqrt 2 * Real.cos θ + Real.cos (2 * θ)
  ∃ (θ : ℝ), f θ = 2 * Real.sqrt 2 + 1 ∧ ∀ (φ : ℝ), f φ ≤ 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_max_value_l3951_395134


namespace NUMINAMATH_CALUDE_Michael_birth_year_l3951_395118

def IMO_start_year : ℕ := 1959

def Michael_age_at_10th_IMO : ℕ := 15

def IMO_held_annually : Prop := ∀ n : ℕ, n ≥ IMO_start_year → ∃ m : ℕ, m = n - IMO_start_year + 1

theorem Michael_birth_year :
  IMO_held_annually →
  ∃ year : ℕ, year = IMO_start_year + 9 - Michael_age_at_10th_IMO ∧ year = 1953 :=
by sorry

end NUMINAMATH_CALUDE_Michael_birth_year_l3951_395118


namespace NUMINAMATH_CALUDE_sandwich_combinations_count_l3951_395188

/-- Represents the number of toppings available -/
def num_toppings : ℕ := 7

/-- Represents the number of bread types available -/
def num_bread_types : ℕ := 3

/-- Represents the number of filling types available -/
def num_filling_types : ℕ := 3

/-- Represents the maximum number of filling layers -/
def max_filling_layers : ℕ := 2

/-- Calculates the total number of sandwich combinations -/
def total_sandwich_combinations : ℕ :=
  (2^num_toppings) * num_bread_types * (num_filling_types + num_filling_types^2)

/-- Theorem stating that the total number of sandwich combinations is 4608 -/
theorem sandwich_combinations_count :
  total_sandwich_combinations = 4608 := by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_count_l3951_395188


namespace NUMINAMATH_CALUDE_trig_invariant_poly_characterization_l3951_395105

/-- A real polynomial that satisfies P(cos x) = P(sin x) for all real x -/
def TrigInvariantPoly (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (Real.cos x) = P (Real.sin x)

/-- The main theorem stating the existence of Q for a trig-invariant polynomial P -/
theorem trig_invariant_poly_characterization
  (P : ℝ → ℝ) (hP : TrigInvariantPoly P) :
  ∃ Q : ℝ → ℝ, ∀ X : ℝ, P X = Q (X^4 - X^2) := by
  sorry

end NUMINAMATH_CALUDE_trig_invariant_poly_characterization_l3951_395105


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3951_395159

theorem complex_magnitude_fourth_power : 
  Complex.abs ((1 + Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3951_395159


namespace NUMINAMATH_CALUDE_smaller_angle_is_55_degrees_l3951_395144

/-- A parallelogram with specific angle properties -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The length of the parallelogram -/
  length : ℝ
  /-- The width of the parallelogram -/
  width : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- Consecutive angles in a parallelogram are supplementary -/
  supplementary : smaller_angle + larger_angle = 180
  /-- The length is three times the width -/
  length_width_ratio : length = 3 * width

/-- Theorem: In a parallelogram where one angle exceeds the other by 70 degrees,
    the measure of the smaller angle is 55 degrees -/
theorem smaller_angle_is_55_degrees (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_is_55_degrees_l3951_395144


namespace NUMINAMATH_CALUDE_circle_E_equation_l3951_395104

-- Define the circle E
def circle_E (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the condition that E passes through A(0,0) and B(1,1)
def passes_through_A_and_B (E : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ E ∧ (1, 1) ∈ E

-- Define the three additional conditions
def condition_1 (E : Set (ℝ × ℝ)) : Prop :=
  (2, 0) ∈ E

def condition_2 (E : Set (ℝ × ℝ)) : Prop :=
  ∀ m : ℝ, ∃ p q : ℝ × ℝ, p ∈ E ∧ q ∈ E ∧
    p.2 = m * (p.1 - 1) ∧ q.2 = m * (q.1 - 1) ∧
    p ≠ q

def condition_3 (E : Set (ℝ × ℝ)) : Prop :=
  ∃ y : ℝ, (0, y) ∈ E ∧ ∀ t : ℝ, t ≠ y → (0, t) ∉ E

-- The main theorem
theorem circle_E_equation :
  ∀ E : Set (ℝ × ℝ),
  passes_through_A_and_B E →
  (condition_1 E ∨ condition_2 E ∨ condition_3 E) →
  E = circle_E (1, 0) 1 :=
sorry

end NUMINAMATH_CALUDE_circle_E_equation_l3951_395104


namespace NUMINAMATH_CALUDE_jogger_difference_l3951_395108

/-- The number of joggers bought by each person -/
structure Joggers where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the problem -/
def jogger_problem (j : Joggers) : Prop :=
  j.christopher = 20 * j.tyson ∧
  j.christopher = 80 ∧
  j.alexander = j.tyson + 22

/-- The theorem to prove -/
theorem jogger_difference (j : Joggers) (h : jogger_problem j) :
  j.christopher - j.alexander = 54 := by
  sorry

end NUMINAMATH_CALUDE_jogger_difference_l3951_395108


namespace NUMINAMATH_CALUDE_total_insects_eaten_l3951_395146

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by all animals -/
def total_insects : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_eaten :
  total_insects = 66 := by sorry

end NUMINAMATH_CALUDE_total_insects_eaten_l3951_395146


namespace NUMINAMATH_CALUDE_tourists_escape_theorem_l3951_395102

/-- Represents the color of a hat -/
inductive HatColor
  | Black
  | White

/-- Represents a tourist in the line -/
structure Tourist where
  position : Nat
  hatColor : HatColor

/-- Represents the line of tourists -/
def TouristLine := List Tourist

/-- A strategy is a function that takes the visible hats and previous guesses
    and returns a guess for the current tourist's hat color -/
def Strategy := (visibleHats : List HatColor) → (previousGuesses : List HatColor) → HatColor

/-- Applies the strategy to a line of tourists and returns the number of correct guesses -/
def applyStrategy (line : TouristLine) (strategy : Strategy) : Nat :=
  sorry

/-- There exists a strategy that guarantees at least 9 out of 10 tourists can correctly guess their hat color -/
theorem tourists_escape_theorem :
  ∃ (strategy : Strategy),
    ∀ (line : TouristLine),
      line.length = 10 →
      applyStrategy line strategy ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_tourists_escape_theorem_l3951_395102


namespace NUMINAMATH_CALUDE_complex_power_difference_zero_l3951_395189

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero : (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_zero_l3951_395189


namespace NUMINAMATH_CALUDE_correct_calculation_l3951_395109

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3951_395109


namespace NUMINAMATH_CALUDE_shirt_sales_price_solution_l3951_395192

/-- Represents the selling price and profit calculation for shirts -/
def ShirtSales (x : ℝ) : Prop :=
  let cost_price : ℝ := 80
  let initial_daily_sales : ℝ := 30
  let price_reduction : ℝ := 130 - x
  let additional_sales : ℝ := 2 * price_reduction
  let total_daily_sales : ℝ := initial_daily_sales + additional_sales
  let profit_per_shirt : ℝ := x - cost_price
  let daily_profit : ℝ := profit_per_shirt * total_daily_sales
  daily_profit = 2000

theorem shirt_sales_price_solution :
  ∃ x : ℝ, ShirtSales x ∧ (x = 105 ∨ x = 120) :=
sorry

end NUMINAMATH_CALUDE_shirt_sales_price_solution_l3951_395192


namespace NUMINAMATH_CALUDE_students_left_l3951_395139

theorem students_left (total : ℕ) (checked_out : ℕ) (h1 : total = 124) (h2 : checked_out = 93) :
  total - checked_out = 31 := by
  sorry

end NUMINAMATH_CALUDE_students_left_l3951_395139


namespace NUMINAMATH_CALUDE_ant_return_probability_l3951_395121

/-- Represents the probability of an ant returning to its starting vertex
    after n steps on a regular tetrahedron. -/
def P (n : ℕ) : ℚ :=
  1/4 - 1/4 * (-1/3)^(n-1)

/-- The probability of an ant returning to its starting vertex
    after 6 steps on a regular tetrahedron with edge length 1. -/
theorem ant_return_probability :
  P 6 = 61/243 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l3951_395121


namespace NUMINAMATH_CALUDE_gumball_probability_l3951_395124

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  blue_prob + pink_prob = 1 →
  blue_prob * blue_prob = 16 / 36 →
  pink_prob = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_gumball_probability_l3951_395124


namespace NUMINAMATH_CALUDE_lightbulb_most_suitable_l3951_395125

/-- Represents a survey option --/
inductive SurveyOption
  | SecurityCheck
  | ClassmateExercise
  | JobInterview
  | LightbulbLifespan

/-- Defines what makes a survey suitable for sampling --/
def suitableForSampling (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.SecurityCheck => false
  | SurveyOption.ClassmateExercise => false
  | SurveyOption.JobInterview => false
  | SurveyOption.LightbulbLifespan => true

/-- Theorem stating that the lightbulb lifespan survey is most suitable for sampling --/
theorem lightbulb_most_suitable :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.LightbulbLifespan →
    suitableForSampling SurveyOption.LightbulbLifespan ∧
    ¬(suitableForSampling option) :=
by
  sorry

#check lightbulb_most_suitable

end NUMINAMATH_CALUDE_lightbulb_most_suitable_l3951_395125


namespace NUMINAMATH_CALUDE_percentage_increase_l3951_395113

theorem percentage_increase (x y z : ℝ) (h1 : y = 0.5 * z) (h2 : x = 0.65 * z) :
  (x - y) / y * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3951_395113


namespace NUMINAMATH_CALUDE_geometric_sequence_max_first_term_l3951_395123

theorem geometric_sequence_max_first_term 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 ≥ 1)
  (h_a2 : a 2 ≤ 2)
  (h_a3 : a 3 ≥ 3) :
  a 1 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_first_term_l3951_395123


namespace NUMINAMATH_CALUDE_coin_exchange_impossibility_l3951_395128

theorem coin_exchange_impossibility : ¬ ∃ n : ℕ, 1 + 4 * n = 26 := by
  sorry

end NUMINAMATH_CALUDE_coin_exchange_impossibility_l3951_395128


namespace NUMINAMATH_CALUDE_gcd_1887_2091_l3951_395138

theorem gcd_1887_2091 : Nat.gcd 1887 2091 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1887_2091_l3951_395138


namespace NUMINAMATH_CALUDE_sphere_radius_in_tetrahedron_l3951_395164

/-- A regular tetrahedron with side length 1 containing four spheres --/
structure TetrahedronWithSpheres where
  /-- The side length of the regular tetrahedron --/
  sideLength : ℝ
  /-- The radius of each sphere --/
  sphereRadius : ℝ
  /-- The number of spheres --/
  numSpheres : ℕ
  /-- Each sphere is tangent to three faces of the tetrahedron --/
  tangentToFaces : Prop
  /-- Each sphere is tangent to the other three spheres --/
  tangentToOtherSpheres : Prop
  /-- The side length of the tetrahedron is 1 --/
  sideLength_eq_one : sideLength = 1
  /-- There are exactly four spheres --/
  numSpheres_eq_four : numSpheres = 4

/-- The theorem stating the radius of the spheres in the tetrahedron --/
theorem sphere_radius_in_tetrahedron (t : TetrahedronWithSpheres) :
  t.sphereRadius = (Real.sqrt 6 - 1) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_tetrahedron_l3951_395164


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3951_395195

theorem rational_inequality_solution (x : ℝ) : 
  (1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔ 
  (x < -3 ∨ (-1 < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3951_395195


namespace NUMINAMATH_CALUDE_jane_mean_score_l3951_395184

def jane_scores : List ℝ := [96, 95, 90, 87, 91, 75]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℝ) = 89 := by
  sorry

end NUMINAMATH_CALUDE_jane_mean_score_l3951_395184


namespace NUMINAMATH_CALUDE_train_length_calculation_l3951_395196

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 46.5) 
  (h2 : man_speed = 2.5) 
  (h3 : passing_time = 62.994960403167745) : 
  ∃ (length : ℝ), abs (length - 770) < 0.1 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3951_395196


namespace NUMINAMATH_CALUDE_olivers_card_collection_l3951_395179

/-- Oliver's card collection problem -/
theorem olivers_card_collection :
  ∀ (alien_baseball monster_club battle_gremlins : ℕ),
  monster_club = 2 * alien_baseball →
  battle_gremlins = 48 →
  battle_gremlins = 3 * alien_baseball →
  monster_club = 32 := by
sorry

end NUMINAMATH_CALUDE_olivers_card_collection_l3951_395179


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l3951_395129

/-- The sticker price of a laptop. -/
def stickerPrice : ℝ := 1100

/-- The price at store C after discount and rebate. -/
def storeCPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at store D after discount. -/
def storeDPrice (price : ℝ) : ℝ := 0.7 * price

theorem laptop_price_theorem : 
  storeCPrice stickerPrice = storeDPrice stickerPrice - 10 := by sorry

end NUMINAMATH_CALUDE_laptop_price_theorem_l3951_395129


namespace NUMINAMATH_CALUDE_num_university_students_is_11_l3951_395172

/-- Represents a chess tournament with university students and two Level 3 students -/
structure ChessTournament where
  num_university_students : ℕ
  university_student_score : ℚ
  level3_total_score : ℚ

/-- The total number of games played in the tournament -/
def total_games (t : ChessTournament) : ℚ :=
  (t.num_university_students + 2) * (t.num_university_students + 1) / 2

/-- The total score of all participants -/
def total_score (t : ChessTournament) : ℚ :=
  t.num_university_students * t.university_student_score + t.level3_total_score

/-- Theorem stating the number of university students in the tournament -/
theorem num_university_students_is_11 (t : ChessTournament) 
  (h1 : t.level3_total_score = 13/2)
  (h2 : total_score t = total_games t)
  (h3 : t.num_university_students > 0) :
  t.num_university_students = 11 := by
  sorry

end NUMINAMATH_CALUDE_num_university_students_is_11_l3951_395172


namespace NUMINAMATH_CALUDE_candy_bar_sales_ratio_l3951_395170

theorem candy_bar_sales_ratio :
  ∀ (price : ℚ) (marvin_sales : ℕ) (tina_extra_earnings : ℚ),
    price = 2 →
    marvin_sales = 35 →
    tina_extra_earnings = 140 →
    ∃ (tina_sales : ℕ),
      tina_sales * price = marvin_sales * price + tina_extra_earnings ∧
      tina_sales = 3 * marvin_sales :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_sales_ratio_l3951_395170


namespace NUMINAMATH_CALUDE_inequality_range_l3951_395154

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3951_395154


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3951_395174

/-- Calculate the simple interest rate given principal, final amount, and time -/
theorem simple_interest_rate (principal final_amount time : ℝ) 
  (h_principal : principal > 0)
  (h_final : final_amount > principal)
  (h_time : time > 0) :
  (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3951_395174


namespace NUMINAMATH_CALUDE_intersection_nonempty_a_subset_b_l3951_395148

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / (x - 3) < -1}
def B (a : ℝ) : Set ℝ := {x | (x - (a^2 + 2)) / (x - a) < 0}

-- Part 1: Intersection is non-empty
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ a < 0 ∨ (0 < a ∧ a < 3) :=
sorry

-- Part 2: A is a subset of B
theorem a_subset_b (a : ℝ) :
  A ⊆ B a ↔ a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_a_subset_b_l3951_395148


namespace NUMINAMATH_CALUDE_symmetry_implies_exponent_l3951_395131

theorem symmetry_implies_exponent (a b : ℝ) : 
  (2 * a + 1 = 1 ∧ -3 * a = -(3 - b)) → b^a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_exponent_l3951_395131


namespace NUMINAMATH_CALUDE_min_cost_notebooks_l3951_395176

/-- Represents the unit price of type A notebooks -/
def price_A : ℝ := 11

/-- Represents the unit price of type B notebooks -/
def price_B : ℝ := price_A + 1

/-- Represents the total number of notebooks to be purchased -/
def total_notebooks : ℕ := 100

/-- Represents the constraint on the quantity of type B notebooks -/
def type_B_constraint (a : ℕ) : Prop := total_notebooks - a ≤ 3 * a

/-- Represents the cost function for purchasing notebooks -/
def cost_function (a : ℕ) : ℝ := price_A * a + price_B * (total_notebooks - a)

/-- Theorem stating that the minimum cost for purchasing 100 notebooks is $1100 -/
theorem min_cost_notebooks : 
  ∃ (a : ℕ), a ≤ total_notebooks ∧ 
  type_B_constraint a ∧ 
  (∀ (b : ℕ), b ≤ total_notebooks → type_B_constraint b → cost_function a ≤ cost_function b) ∧
  cost_function a = 1100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_notebooks_l3951_395176


namespace NUMINAMATH_CALUDE_tangent_and_common_point_l3951_395111

/-- The line l: y = kx - 3k + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k + 2

/-- The curve C: (x-1)² + (y+1)² = 4 where -1 ≤ x ≤ 1 -/
def curve (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 4 ∧ -1 ≤ x ∧ x ≤ 1

theorem tangent_and_common_point (k : ℝ) :
  (∃ x, curve x (line k x) ∧
    ∀ x', x' ≠ x → ¬ curve x' (line k x')) ↔
  k = 5/12 ∨ (1/2 < k ∧ k ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_common_point_l3951_395111


namespace NUMINAMATH_CALUDE_altitude_length_l3951_395180

/-- An isosceles triangle with given side lengths and altitude --/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Altitude
  ad : ℝ
  -- Altitude meets base at midpoint
  isMidpoint : bd = bc / 2

/-- The theorem stating the length of the altitude in the given isosceles triangle --/
theorem altitude_length (t : IsoscelesTriangle) 
  (h1 : t.ab = 10) 
  (h2 : t.bc = 16) : 
  t.ad = 6 := by
  sorry

#check altitude_length

end NUMINAMATH_CALUDE_altitude_length_l3951_395180


namespace NUMINAMATH_CALUDE_friday_first_day_over_200_l3951_395100

/-- Represents the days of the week -/
inductive Day
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the number of days after Monday -/
def daysAfterMonday (d : Day) : Nat :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- Calculates the number of paperclips on a given day -/
def paperclipsOn (d : Day) : Nat :=
  4 * (3 ^ (daysAfterMonday d))

/-- Theorem: Friday is the first day with more than 200 paperclips -/
theorem friday_first_day_over_200 :
  (∀ d : Day, d ≠ Day.friday → paperclipsOn d ≤ 200) ∧
  paperclipsOn Day.friday > 200 :=
sorry

end NUMINAMATH_CALUDE_friday_first_day_over_200_l3951_395100


namespace NUMINAMATH_CALUDE_math_competition_score_l3951_395182

theorem math_competition_score (total_questions n_correct n_wrong n_unanswered : ℕ) 
  (new_score old_score : ℕ) :
  total_questions = 50 ∧ 
  new_score = 150 ∧ 
  old_score = 118 ∧ 
  new_score = 6 * n_correct + 3 * n_unanswered ∧ 
  old_score = 40 + 5 * n_correct - 2 * n_wrong ∧ 
  total_questions = n_correct + n_wrong + n_unanswered →
  n_unanswered = 16 := by
sorry

end NUMINAMATH_CALUDE_math_competition_score_l3951_395182


namespace NUMINAMATH_CALUDE_math_book_cost_l3951_395156

theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) :
  total_books = 80 →
  math_books = 32 →
  history_book_cost = 5 →
  total_cost = 368 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧
    math_book_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_book_cost_l3951_395156


namespace NUMINAMATH_CALUDE_round_trip_percentage_l3951_395191

/-- Proves that 80% of passengers held round-trip tickets given the conditions -/
theorem round_trip_percentage (total_passengers : ℕ) 
  (h1 : (40 : ℝ) / 100 * total_passengers = (passengers_with_car : ℝ))
  (h2 : (50 : ℝ) / 100 * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_car) :
  (80 : ℝ) / 100 * total_passengers = (passengers_with_roundtrip : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l3951_395191


namespace NUMINAMATH_CALUDE_irrational_partner_is_one_l3951_395135

theorem irrational_partner_is_one (a b : ℝ) : 
  (∃ (q : ℚ), a ≠ (q : ℝ)) → -- a is irrational
  (a * b - a - b + 1 = 0) →  -- given equation
  b = 1 :=                   -- conclusion: b equals 1
by sorry

end NUMINAMATH_CALUDE_irrational_partner_is_one_l3951_395135


namespace NUMINAMATH_CALUDE_teacher_age_l3951_395183

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students : ℝ) * student_avg_age + 36 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l3951_395183


namespace NUMINAMATH_CALUDE_rectangle_covers_curve_l3951_395181

/-- A plane curve is a continuous function from a closed interval to ℝ² -/
def PlaneCurve := Set.Icc 0 1 → ℝ × ℝ

/-- The length of a plane curve -/
def curveLength (γ : PlaneCurve) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle covers a curve -/
def covers (r : Rectangle) (γ : PlaneCurve) : Prop := sorry

/-- Main theorem: For any plane curve of length 1, there exists a rectangle of area 1/4 that covers it -/
theorem rectangle_covers_curve (γ : PlaneCurve) (h : curveLength γ = 1) :
  ∃ r : Rectangle, r.area = 1/4 ∧ covers r γ := by sorry

end NUMINAMATH_CALUDE_rectangle_covers_curve_l3951_395181


namespace NUMINAMATH_CALUDE_typing_service_problem_l3951_395149

/-- Represents the typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (total_cost : ℕ) 
  (first_typing_cost : ℕ) 
  (revision_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_revised_twice = 30)
  (h3 : total_cost = 1400)
  (h4 : first_typing_cost = 10)
  (h5 : revision_cost = 5) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 20 ∧
    total_cost = 
      first_typing_cost * total_pages + 
      revision_cost * pages_revised_once + 
      2 * revision_cost * pages_revised_twice :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l3951_395149


namespace NUMINAMATH_CALUDE_ellipse_tangent_to_lines_l3951_395158

/-- The first line tangent to the ellipse -/
def line1 (x y : ℝ) : Prop := x + 2*y = 27

/-- The second line tangent to the ellipse -/
def line2 (x y : ℝ) : Prop := 7*x + 4*y = 81

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 162*x^2 + 81*y^2 = 13122

/-- Theorem stating that the given ellipse equation is tangent to both lines -/
theorem ellipse_tangent_to_lines :
  ∀ x y : ℝ, line1 x y ∨ line2 x y → ellipse_equation x y := by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_to_lines_l3951_395158


namespace NUMINAMATH_CALUDE_install_time_proof_l3951_395190

/-- The time required to install the remaining windows -/
def time_to_install_remaining (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) : ℕ :=
  (total_windows - installed_windows) * time_per_window

/-- Proof that the time to install remaining windows is 36 hours -/
theorem install_time_proof (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) :
  time_to_install_remaining total_windows installed_windows time_per_window = 36 :=
by sorry

end NUMINAMATH_CALUDE_install_time_proof_l3951_395190


namespace NUMINAMATH_CALUDE_minimum_triangle_area_l3951_395165

/-- Given a line passing through (2, 1) and intersecting the positive x and y axes at points A and B
    respectively, with O as the origin, the minimum area of triangle AOB is 4. -/
theorem minimum_triangle_area (k : ℝ) (h : k < 0) :
  let xA := 2 - 1 / k
  let yB := 1 - 2 * k
  let area := (1 / 2) * xA * yB
  4 ≤ area :=
by sorry

end NUMINAMATH_CALUDE_minimum_triangle_area_l3951_395165


namespace NUMINAMATH_CALUDE_only_zero_solution_l3951_395141

theorem only_zero_solution (m n : ℤ) : 231 * m^2 = 130 * n^2 → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_l3951_395141


namespace NUMINAMATH_CALUDE_fox_max_berries_l3951_395117

/-- The number of bear cubs -/
def num_cubs : ℕ := 100

/-- The total number of berries initially -/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat -/
def max_fox_berries : ℕ := 1

/-- Theorem stating the maximum number of berries the fox can eat -/
theorem fox_max_berries :
  max_fox_berries = (total_berries % num_cubs) :=
by sorry

end NUMINAMATH_CALUDE_fox_max_berries_l3951_395117


namespace NUMINAMATH_CALUDE_largest_angle_in_hexagon_l3951_395186

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Sum of angles in a hexagon is 720°
  A + B + C + D + E + F = 720 ∧
  -- Given conditions
  A = 90 ∧
  B = 120 ∧
  C = 95 ∧
  D = E ∧
  F = 2 * D + 25

-- Theorem statement
theorem largest_angle_in_hexagon (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) : 
  max A (max B (max C (max D (max E F)))) = 220 := by
  sorry


end NUMINAMATH_CALUDE_largest_angle_in_hexagon_l3951_395186


namespace NUMINAMATH_CALUDE_kelly_games_left_l3951_395163

theorem kelly_games_left (initial_games : ℝ) (games_given_away : ℕ) : 
  initial_games = 121.0 →
  games_given_away = 99 →
  initial_games - games_given_away = 22.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l3951_395163


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3951_395136

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n) →
  a 1 + a 2 + a 3 = 2 →
  a 3 + a 4 + a 5 = 8 →
  a 4 + a 5 + a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3951_395136


namespace NUMINAMATH_CALUDE_exists_infinite_ap_not_in_polynomial_image_l3951_395133

/-- A polynomial of degree 10 with integer coefficients -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
    ∀ x, P x = a₁₀ * x^10 + a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + 
             a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

/-- An infinite arithmetic progression -/
def InfiniteArithmeticProgression (a d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = a + k * d}

/-- The main theorem -/
theorem exists_infinite_ap_not_in_polynomial_image (P : ℤ → ℤ) 
    (h : IntPolynomial P) :
  ∃ (a d : ℤ), d ≠ 0 ∧ 
    ∀ n ∈ InfiniteArithmeticProgression a d, 
      ∀ k : ℤ, P k ≠ n :=
by sorry

end NUMINAMATH_CALUDE_exists_infinite_ap_not_in_polynomial_image_l3951_395133


namespace NUMINAMATH_CALUDE_equation_roots_right_triangle_l3951_395150

-- Define the equation
def equation (x a b : ℝ) : Prop := |x^2 - 2*a*x + b| = 8

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2

-- Theorem statement
theorem equation_roots_right_triangle (a b : ℝ) :
  (∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation x a b ∧ equation y a b ∧ equation z a b ∧
    is_right_triangle x y z ∧
    (∀ w : ℝ, equation w a b → w = x ∨ w = y ∨ w = z)) →
  a + b = 264 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_right_triangle_l3951_395150


namespace NUMINAMATH_CALUDE_stream_speed_l3951_395151

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 10 →
  distance = 90 →
  time = 5 →
  ∃ (stream_speed : ℝ), 
    distance = (rowing_speed + stream_speed) * time ∧
    stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3951_395151
