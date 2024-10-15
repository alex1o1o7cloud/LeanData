import Mathlib

namespace NUMINAMATH_CALUDE_pascal_triangle_value_l1042_104293

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 51

/-- The position of the number we're looking for in the row (1-indexed) -/
def position : ℕ := 43

/-- The value we want to prove is correct -/
def target_value : ℕ := 10272278170

theorem pascal_triangle_value :
  binomial (row_length - 1) (position - 1) = target_value := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_value_l1042_104293


namespace NUMINAMATH_CALUDE_k_range_given_a_n_geq_a_3_l1042_104206

/-- A sequence where a_n = n^2 - k*n -/
def a (n : ℕ+) (k : ℝ) : ℝ := n.val^2 - k * n.val

/-- The theorem stating that if a_n ≥ a_3 for all positive integers n, then k is in [5, 7] -/
theorem k_range_given_a_n_geq_a_3 :
  (∀ n : ℕ+, a n k ≥ a 3 k) → k ∈ Set.Icc (5 : ℝ) 7 := by
  sorry

end NUMINAMATH_CALUDE_k_range_given_a_n_geq_a_3_l1042_104206


namespace NUMINAMATH_CALUDE_nested_radical_value_l1042_104257

def nested_radical (x : ℝ) : Prop := x = Real.sqrt (20 + x)

theorem nested_radical_value :
  ∃ x : ℝ, nested_radical x ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1042_104257


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l1042_104242

theorem brand_preference_ratio (total_respondents : ℕ) (brand_x_preference : ℕ) 
  (h1 : total_respondents = 80)
  (h2 : brand_x_preference = 60)
  (h3 : brand_x_preference < total_respondents) :
  (brand_x_preference : ℚ) / (total_respondents - brand_x_preference : ℚ) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l1042_104242


namespace NUMINAMATH_CALUDE_expression_values_l1042_104226

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (k : ℤ), k ∈ ({5, 2, 1, -2, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = k) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1042_104226


namespace NUMINAMATH_CALUDE_construction_team_equation_l1042_104265

/-- Represents the equation for a construction team's road-laying project -/
theorem construction_team_equation (x : ℝ) (h : x > 0) :
  let total_length : ℝ := 480
  let efficiency_increase : ℝ := 0.5
  let days_ahead : ℝ := 4
  (total_length / x) - (total_length / ((1 + efficiency_increase) * x)) = days_ahead :=
by sorry

end NUMINAMATH_CALUDE_construction_team_equation_l1042_104265


namespace NUMINAMATH_CALUDE_sally_next_birthday_l1042_104202

theorem sally_next_birthday (adam mary sally danielle : ℝ) 
  (h1 : adam = 1.3 * mary)
  (h2 : mary = 0.75 * sally)
  (h3 : sally = 0.8 * danielle)
  (h4 : adam + mary + sally + danielle = 60) :
  ⌈sally⌉ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sally_next_birthday_l1042_104202


namespace NUMINAMATH_CALUDE_inequality_proof_l1042_104296

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1042_104296


namespace NUMINAMATH_CALUDE_calculator_cost_l1042_104274

/-- Given information about calculator purchases, prove the cost of each graphing calculator. -/
theorem calculator_cost (total_cost : ℕ) (total_calculators : ℕ) (scientific_cost : ℕ)
  (scientific_count : ℕ) (graphing_count : ℕ)
  (h1 : total_cost = 1625)
  (h2 : total_calculators = 45)
  (h3 : scientific_cost = 10)
  (h4 : scientific_count = 20)
  (h5 : graphing_count = 25)
  (h6 : total_calculators = scientific_count + graphing_count) :
  (total_cost - scientific_cost * scientific_count) / graphing_count = 57 := by
  sorry

#eval (1625 - 10 * 20) / 25  -- Should output 57

end NUMINAMATH_CALUDE_calculator_cost_l1042_104274


namespace NUMINAMATH_CALUDE_duck_flying_ratio_l1042_104246

/-- Represents the flying time of a duck during different seasons -/
structure DuckFlyingTime where
  total : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the ratio of north flying time to south flying time -/
def northToSouthRatio (d : DuckFlyingTime) : ℚ :=
  let north := d.total - d.south - d.east
  (north : ℚ) / d.south

/-- Theorem stating that the ratio of north to south flying time is 2:1 -/
theorem duck_flying_ratio :
  ∀ d : DuckFlyingTime,
  d.total = 180 ∧ d.south = 40 ∧ d.east = 60 →
  northToSouthRatio d = 2 := by
  sorry


end NUMINAMATH_CALUDE_duck_flying_ratio_l1042_104246


namespace NUMINAMATH_CALUDE_johns_donation_l1042_104253

/-- Given 6 initial contributions and a new contribution that increases the average by 50% to $75, prove that the new contribution is $225. -/
theorem johns_donation (initial_contributions : ℕ) (new_average : ℚ) : 
  initial_contributions = 6 ∧ 
  new_average = 75 ∧ 
  new_average = (3/2) * (300 / initial_contributions) →
  ∃ (johns_contribution : ℚ), 
    johns_contribution = 225 ∧
    new_average = (300 + johns_contribution) / (initial_contributions + 1) :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_l1042_104253


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1042_104281

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 3 / 5 → x + y + 10 = 50 → min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1042_104281


namespace NUMINAMATH_CALUDE_women_decrease_l1042_104259

theorem women_decrease (initial_men initial_women final_men final_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  final_men = initial_men + 2 →
  24 = initial_women - 3 →
  final_men = 14 →
  final_women = 24 →
  initial_women - final_women = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_women_decrease_l1042_104259


namespace NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l1042_104233

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  ∃ (m : ℝ), m = (6 - 2*Real.sqrt 3) / 3 ∧ ∀ (a b : ℝ), a^2 - a*b + b^2 = 1 → x^2 + 2*y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l1042_104233


namespace NUMINAMATH_CALUDE_syllogism_flaw_l1042_104211

theorem syllogism_flaw : ¬(∀ a : ℝ, a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_syllogism_flaw_l1042_104211


namespace NUMINAMATH_CALUDE_equal_numbers_sum_l1042_104237

theorem equal_numbers_sum (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d + e = 45 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_sum_l1042_104237


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l1042_104271

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (a+1)x^2 + (a-2)x + a^2 - a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a+1)*x^2 + (a-2)*x + a^2 - a - 2

theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l1042_104271


namespace NUMINAMATH_CALUDE_johns_salary_increase_l1042_104225

theorem johns_salary_increase (original_salary new_salary : ℝ) 
  (h1 : new_salary = 110)
  (h2 : new_salary = original_salary * (1 + 0.8333333333333334)) : 
  original_salary = 60 := by
sorry

end NUMINAMATH_CALUDE_johns_salary_increase_l1042_104225


namespace NUMINAMATH_CALUDE_power_product_rule_l1042_104224

theorem power_product_rule (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l1042_104224


namespace NUMINAMATH_CALUDE_product_zero_l1042_104228

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l1042_104228


namespace NUMINAMATH_CALUDE_value_of_b_l1042_104286

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1042_104286


namespace NUMINAMATH_CALUDE_journey_equation_correct_l1042_104294

/-- Represents a car journey with a stop -/
structure Journey where
  initial_speed : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ
  stop_duration : ℝ

/-- Theorem stating that the given equation correctly represents the total distance traveled -/
theorem journey_equation_correct (j : Journey) 
  (h1 : j.initial_speed = 90)
  (h2 : j.final_speed = 110)
  (h3 : j.total_distance = 300)
  (h4 : j.total_time = 3.5)
  (h5 : j.stop_duration = 0.5) :
  ∃ t : ℝ, j.initial_speed * t + j.final_speed * (j.total_time - j.stop_duration - t) = j.total_distance :=
sorry

end NUMINAMATH_CALUDE_journey_equation_correct_l1042_104294


namespace NUMINAMATH_CALUDE_square_difference_l1042_104223

theorem square_difference (x y : ℚ) 
  (h1 : (x + y)^2 = 49/144) 
  (h2 : (x - y)^2 = 1/144) : 
  x^2 - y^2 = 7/144 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1042_104223


namespace NUMINAMATH_CALUDE_box_tape_relation_l1042_104207

def tape_needed (long_side short_side : ℝ) (num_boxes : ℕ) : ℝ :=
  num_boxes * (long_side + 2 * short_side)

theorem box_tape_relation (L S : ℝ) :
  tape_needed L S 5 + tape_needed 40 40 2 = 540 →
  L = 60 - 2 * S :=
by
  sorry

end NUMINAMATH_CALUDE_box_tape_relation_l1042_104207


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l1042_104261

theorem smallest_k_for_64_power_gt_4_16 : ∃ k : ℕ, k = 6 ∧ 64^k > 4^16 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l1042_104261


namespace NUMINAMATH_CALUDE_power_multiplication_l1042_104277

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1042_104277


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l1042_104272

theorem sum_of_distinct_roots_is_zero 
  (a b c x y : ℝ) 
  (ha : a^3 + a*x + y = 0)
  (hb : b^3 + b*x + y = 0)
  (hc : c^3 + c*x + y = 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l1042_104272


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1042_104222

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x - 6)^2 - 9

-- Define the square
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ

-- Predicate to check if a square is inscribed in the region
def isInscribed (square : InscribedSquare) : Prop :=
  let halfSide := square.sideLength / 2
  let leftX := square.center - halfSide
  let rightX := square.center + halfSide
  leftX ≥ 0 ∧ 
  rightX ≥ 0 ∧ 
  parabola rightX = -square.sideLength

-- Theorem statement
theorem inscribed_square_area :
  ∃ (square : InscribedSquare), 
    isInscribed square ∧ 
    square.sideLength^2 = 40 - 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1042_104222


namespace NUMINAMATH_CALUDE_growth_rate_is_ten_percent_l1042_104268

def turnover_may : ℝ := 1
def turnover_july : ℝ := 1.21

def growth_rate (r : ℝ) : Prop :=
  turnover_may * (1 + r)^2 = turnover_july

theorem growth_rate_is_ten_percent :
  ∃ (r : ℝ), growth_rate r ∧ r = 0.1 :=
sorry

end NUMINAMATH_CALUDE_growth_rate_is_ten_percent_l1042_104268


namespace NUMINAMATH_CALUDE_complex_magnitude_l1042_104298

theorem complex_magnitude (z : ℂ) (h : 3 * z + Complex.I = 1 - 4 * Complex.I * z) :
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1042_104298


namespace NUMINAMATH_CALUDE_scientific_notation_of_1500000_l1042_104217

theorem scientific_notation_of_1500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500000 = a * (10 : ℝ) ^ n ∧ a = 1.5 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1500000_l1042_104217


namespace NUMINAMATH_CALUDE_four_number_sequence_l1042_104297

theorem four_number_sequence : ∃ (a₁ a₂ a₃ a₄ : ℝ),
  (a₂^2 = a₁ * a₃) ∧
  (2 * a₃ = a₂ + a₄) ∧
  (a₁ + a₄ = 21) ∧
  (a₂ + a₃ = 18) ∧
  ((a₁ = 3 ∧ a₂ = 6 ∧ a₃ = 12 ∧ a₄ = 18) ∨
   (a₁ = 18.75 ∧ a₂ = 11.25 ∧ a₃ = 6.75 ∧ a₄ = 2.25)) :=
by
  sorry


end NUMINAMATH_CALUDE_four_number_sequence_l1042_104297


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1042_104216

/-- An isosceles triangle with height 8 and perimeter 32 has an area of 48 -/
theorem isosceles_triangle_area (b : ℝ) (l : ℝ) : 
  b > 0 → l > 0 →
  (2 * l + b = 32) →  -- perimeter condition
  (l^2 = (b/2)^2 + 8^2) →  -- Pythagorean theorem for height
  (1/2 * b * 8 = 48) :=  -- area formula
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1042_104216


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l1042_104203

def isSystematicSample (sample : List Nat) (totalItems : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  ∀ i, i ∈ sample → i ≤ totalItems ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → ∃ k, j - i = k * ((totalItems - 1) / (sampleSize - 1))

theorem systematic_sampling_example :
  isSystematicSample [3, 13, 23, 33, 43] 50 5 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_example_l1042_104203


namespace NUMINAMATH_CALUDE_jelly_cost_l1042_104229

/-- Proof of the cost of jelly given bread, peanut butter, and leftover money -/
theorem jelly_cost (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) 
  (total_money : ℚ) (leftover_money : ℚ) :
  bread_price = 2.25 →
  bread_quantity = 3 →
  peanut_butter_price = 2 →
  total_money = 14 →
  leftover_money = 5.25 →
  leftover_money = total_money - (bread_price * bread_quantity + peanut_butter_price) :=
by
  sorry

#check jelly_cost

end NUMINAMATH_CALUDE_jelly_cost_l1042_104229


namespace NUMINAMATH_CALUDE_cubic_roots_reciprocal_squares_sum_l1042_104230

theorem cubic_roots_reciprocal_squares_sum : 
  ∀ a b c : ℝ, 
  (a^3 - 8*a^2 + 15*a - 7 = 0) → 
  (b^3 - 8*b^2 + 15*b - 7 = 0) → 
  (c^3 - 8*c^2 + 15*c - 7 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 113/49) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_reciprocal_squares_sum_l1042_104230


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_quadrilateral_perimeter_proof_l1042_104295

/-- The perimeter of a quadrilateral with vertices A(0,0), B(0,10), C(8,10), and D(8,0) is 36 -/
theorem quadrilateral_perimeter : ℝ → Prop :=
  fun perimeter =>
    let A : ℝ × ℝ := (0, 0)
    let B : ℝ × ℝ := (0, 10)
    let C : ℝ × ℝ := (8, 10)
    let D : ℝ × ℝ := (8, 0)
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
    let DA := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
    perimeter = AB + BC + CD + DA ∧ perimeter = 36

/-- Proof of the theorem -/
theorem quadrilateral_perimeter_proof : quadrilateral_perimeter 36 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_quadrilateral_perimeter_proof_l1042_104295


namespace NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l1042_104292

-- Problem 1
theorem expand_and_simplify (x : ℝ) :
  (2*x - 1)*(2*x - 3) - (1 - 2*x)*(2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_complex_fraction (a : ℝ) (ha : a ≠ 0) (ha1 : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l1042_104292


namespace NUMINAMATH_CALUDE_sum_of_squares_difference_l1042_104254

theorem sum_of_squares_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_difference_l1042_104254


namespace NUMINAMATH_CALUDE_ratio_problem_l1042_104260

theorem ratio_problem (a b x m : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a / b = 4 / 5 → x = a * (1 + 1/4) → m = b * (1 - 4/5) → m / x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1042_104260


namespace NUMINAMATH_CALUDE_island_puzzle_l1042_104299

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The statement made by a person about the number of liars -/
def Statement := Fin 5 → ℕ

/-- Checks if a person's statement is truthful given the actual number of liars -/
def isStatementTruthful (p : Person) (s : ℕ) (actualLiars : ℕ) : Prop :=
  match p with
  | Person.Knight => s = actualLiars
  | Person.Liar => s ≠ actualLiars

/-- The main theorem to prove -/
theorem island_puzzle :
  ∀ (people : Fin 5 → Person) (statements : Statement),
  (∀ i j : Fin 5, i ≠ j → statements i ≠ statements j) →
  (∀ i : Fin 5, statements i = i.val + 1) →
  (∃! i : Fin 5, people i = Person.Knight) →
  (∀ i : Fin 5, isStatementTruthful (people i) (statements i) 4) :=
sorry

end NUMINAMATH_CALUDE_island_puzzle_l1042_104299


namespace NUMINAMATH_CALUDE_tire_change_problem_l1042_104264

theorem tire_change_problem (total_cars : ℕ) (tires_per_car : ℕ) (half_change_cars : ℕ) (tires_left : ℕ) : 
  total_cars = 10 →
  tires_per_car = 4 →
  half_change_cars = 2 →
  tires_left = 20 →
  ∃ (no_change_cars : ℕ), 
    no_change_cars = total_cars - (half_change_cars + (total_cars * tires_per_car - tires_left - half_change_cars * (tires_per_car / 2)) / tires_per_car) ∧
    no_change_cars = 4 :=
by sorry

end NUMINAMATH_CALUDE_tire_change_problem_l1042_104264


namespace NUMINAMATH_CALUDE_larger_number_proof_l1042_104248

theorem larger_number_proof (x y : ℤ) (h1 : x - y = 5) (h2 : x + y = 37) :
  max x y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1042_104248


namespace NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l1042_104231

theorem x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l1042_104231


namespace NUMINAMATH_CALUDE_water_difference_before_exchange_l1042_104283

/-- The difference in water amounts before the exchange, given the conditions of the problem -/
theorem water_difference_before_exchange 
  (S H : ℝ) -- S and H represent the initial amounts of water for Seungmin and Hyoju
  (h1 : S > H) -- Seungmin has more water than Hyoju
  (h2 : S - 0.43 - (H + 0.43) = 0.88) -- Difference after exchange
  : S - H = 1.74 := by sorry

end NUMINAMATH_CALUDE_water_difference_before_exchange_l1042_104283


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1042_104256

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

theorem min_distance_to_line (m n : ℝ) 
  (h : (a.1 - m) * (b.1 - m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), line_equation x y → 
    d ≤ Real.sqrt ((x - m)^2 + (y - n)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1042_104256


namespace NUMINAMATH_CALUDE_nick_money_value_l1042_104288

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels Nick has -/
def num_nickels : ℕ := 6

/-- The number of dimes Nick has -/
def num_dimes : ℕ := 2

/-- The number of quarters Nick has -/
def num_quarters : ℕ := 1

/-- The total value of Nick's coins in cents -/
def total_value : ℕ := num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value

theorem nick_money_value : total_value = 75 := by
  sorry

end NUMINAMATH_CALUDE_nick_money_value_l1042_104288


namespace NUMINAMATH_CALUDE_largest_difference_l1042_104284

def Digits : Finset ℕ := {1, 3, 7, 8, 9}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 5) ∧
  (∀ d ∈ Digits, (d ∈ Finset.filter (λ x => x ∈ Digits) (Finset.range 10)))

theorem largest_difference :
  ∃ (a b : ℕ), is_valid_pair a b ∧
    ∀ (x y : ℕ), is_valid_pair x y → (a - b ≥ x - y) ∧ (a - b = 9868) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l1042_104284


namespace NUMINAMATH_CALUDE_classroom_difference_maple_leaf_elementary_l1042_104247

theorem classroom_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, guinea_pigs_per_class =>
  let total_students := num_classrooms * students_per_class
  let total_pets := num_classrooms * (rabbits_per_class + guinea_pigs_per_class)
  total_students - total_pets

theorem maple_leaf_elementary :
  classroom_difference 6 15 1 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_classroom_difference_maple_leaf_elementary_l1042_104247


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l1042_104236

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l1042_104236


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_seating_satisfies_spacing_l1042_104201

/-- The number of ways to seat people on chairs with spacing requirements -/
def seating_arrangements (people chairs : ℕ) : ℕ :=
  if people = 3 ∧ chairs = 8 then 36 else 0

/-- Theorem stating the correct number of seating arrangements -/
theorem correct_seating_arrangements :
  seating_arrangements 3 8 = 36 := by
  sorry

/-- Theorem proving the seating arrangement satisfies the spacing requirement -/
theorem seating_satisfies_spacing (arrangement : Fin 8 → Option (Fin 3)) :
  seating_arrangements 3 8 = 36 →
  (∀ i j : Fin 3, i ≠ j →
    ∀ s t : Fin 8, arrangement s = some i ∧ arrangement t = some j →
      (s : ℕ) + 1 < (t : ℕ) ∨ (t : ℕ) + 1 < (s : ℕ)) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_seating_satisfies_spacing_l1042_104201


namespace NUMINAMATH_CALUDE_range_of_a_l1042_104279

theorem range_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_one : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1042_104279


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l1042_104290

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 8
  | n + 1 => 2 * club_size n - 2

/-- Theorem stating that the club size after 4 years is 98 -/
theorem club_size_after_four_years :
  club_size 4 = 98 := by
  sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l1042_104290


namespace NUMINAMATH_CALUDE_reasoning_method_is_inductive_l1042_104208

-- Define the set of animals
inductive Animal : Type
| Ape : Animal
| Cat : Animal
| Elephant : Animal
| OtherMammal : Animal

-- Define the breathing method
inductive BreathingMethod : Type
| Lungs : BreathingMethod

-- Define the reasoning method
inductive ReasoningMethod : Type
| Inductive : ReasoningMethod
| Deductive : ReasoningMethod
| Analogical : ReasoningMethod
| CompleteInductive : ReasoningMethod

-- Define a function that represents breathing for specific animals
def breathes : Animal → BreathingMethod
| Animal.Ape => BreathingMethod.Lungs
| Animal.Cat => BreathingMethod.Lungs
| Animal.Elephant => BreathingMethod.Lungs
| Animal.OtherMammal => BreathingMethod.Lungs

-- Define a predicate for reasoning from specific to general
def reasonsFromSpecificToGeneral (method : ReasoningMethod) : Prop :=
  method = ReasoningMethod.Inductive

-- Theorem statement
theorem reasoning_method_is_inductive :
  (∀ a : Animal, breathes a = BreathingMethod.Lungs) →
  (reasonsFromSpecificToGeneral ReasoningMethod.Inductive) :=
by sorry

end NUMINAMATH_CALUDE_reasoning_method_is_inductive_l1042_104208


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l1042_104227

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l1042_104227


namespace NUMINAMATH_CALUDE_paving_stone_width_l1042_104251

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Given a courtyard and paving stone specifications, proves that the width of each paving stone is 2 meters -/
theorem paving_stone_width
  (courtyard : Courtyard)
  (stone_count : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard.length = 30)
  (h2 : courtyard.width = 33/2)
  (h3 : stone_count = 99)
  (h4 : stone_length = 5/2) :
  ∃ (stone : PavingStone), stone.length = stone_length ∧ stone.width = 2 :=
sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1042_104251


namespace NUMINAMATH_CALUDE_lock_combination_l1042_104252

-- Define the base
def base : ℕ := 11

-- Define the function to convert from base 11 to decimal
def toDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (λ acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

-- Define the equation in base 11
axiom equation_holds : ∃ (S T A R E B : ℕ),
  S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
  S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
  T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
  A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
  R ≠ E ∧ R ≠ B ∧
  E ≠ B ∧
  (S * base^3 + T * base^2 + A * base + R) +
  (T * base^3 + A * base^2 + R * base + S) +
  (R * base^3 + E * base^2 + S * base + T) +
  (R * base^3 + A * base^2 + R * base + E) +
  (B * base^3 + E * base^2 + A * base + R) =
  (B * base^3 + E * base^2 + S * base + T)

-- Theorem to prove
theorem lock_combination : 
  ∃ (S T A R : ℕ), toDecimal [S, T, A, R] = 7639 ∧
  (∃ (E B : ℕ), 
    S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
    S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
    T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
    A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
    R ≠ E ∧ R ≠ B ∧
    E ≠ B ∧
    (S * base^3 + T * base^2 + A * base + R) +
    (T * base^3 + A * base^2 + R * base + S) +
    (R * base^3 + E * base^2 + S * base + T) +
    (R * base^3 + A * base^2 + R * base + E) +
    (B * base^3 + E * base^2 + A * base + R) =
    (B * base^3 + E * base^2 + S * base + T)) :=
by sorry

end NUMINAMATH_CALUDE_lock_combination_l1042_104252


namespace NUMINAMATH_CALUDE_triangle_determines_plane_l1042_104220

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3.x = p1.x + t * (p2.x - p1.x) ∧
               p3.y = p1.y + t * (p2.y - p1.y) ∧
               p3.z = p1.z + t * (p2.z - p1.z)

/-- A plane contains a point if the point satisfies the plane equation -/
def PlaneContainsPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Theorem: Three non-collinear points uniquely determine a plane -/
theorem triangle_determines_plane (p1 p2 p3 : Point3D) 
  (h : NonCollinear p1 p2 p3) : 
  ∃! (plane : Plane), PlaneContainsPoint plane p1 ∧ 
                      PlaneContainsPoint plane p2 ∧ 
                      PlaneContainsPoint plane p3 :=
sorry

end NUMINAMATH_CALUDE_triangle_determines_plane_l1042_104220


namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l1042_104214

-- Define the number of internal medicine doctors and surgeons
def num_internal_med : ℕ := 12
def num_surgeons : ℕ := 8

-- Define the number of doctors needed for the team
def team_size : ℕ := 5

-- Theorem for part (1)
theorem disaster_relief_team_selection_part1 :
  (Nat.choose (num_internal_med + num_surgeons - 2) (team_size - 1)) = 3060 :=
sorry

-- Theorem for part (2)
theorem disaster_relief_team_selection_part2 :
  (Nat.choose (num_internal_med + num_surgeons) team_size) -
  (Nat.choose num_surgeons team_size) -
  (Nat.choose num_internal_med team_size) = 14656 :=
sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l1042_104214


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l1042_104258

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y - 2 - a = 0 ∧ x = y) ↔ (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l1042_104258


namespace NUMINAMATH_CALUDE_coffee_package_size_l1042_104240

theorem coffee_package_size (total_coffee : ℕ) (large_package_size : ℕ) (large_package_count : ℕ) (small_package_count_diff : ℕ) :
  total_coffee = 55 →
  large_package_size = 10 →
  large_package_count = 3 →
  small_package_count_diff = 2 →
  ∃ (small_package_size : ℕ),
    small_package_size * (large_package_count + small_package_count_diff) +
    large_package_size * large_package_count = total_coffee ∧
    small_package_size = 5 :=
by sorry

end NUMINAMATH_CALUDE_coffee_package_size_l1042_104240


namespace NUMINAMATH_CALUDE_min_value_condition_l1042_104282

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then |x + a| + |x - 2|
  else x^2 - a*x + (1/2)*a + 1

theorem min_value_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2*a) ∧ (∃ x : ℝ, f a x = 2*a) ↔ a = -Real.sqrt 13 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_condition_l1042_104282


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1042_104244

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2 * x₁^2 + (2*k - 1) * x₁ + 1 = 0 ∧ k^2 * x₂^2 + (2*k - 1) * x₂ + 1 = 0) →
  (k < 1/4 ∧ k ≠ 0) ∧
  ¬∃ (k : ℝ), ∃ (x : ℝ), k^2 * x^2 + (2*k - 1) * x + 1 = 0 ∧ k^2 * (-x)^2 + (2*k - 1) * (-x) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1042_104244


namespace NUMINAMATH_CALUDE_smiles_cookies_leftover_l1042_104291

theorem smiles_cookies_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_smiles_cookies_leftover_l1042_104291


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1042_104200

theorem expression_simplification_and_evaluation :
  let f (a : ℝ) := a / (a - 1) + (a + 1) / (a^2 - 1)
  let g (a : ℝ) := (a + 1) / (a - 1)
  ∀ a : ℝ, a^2 - 1 ≠ 0 →
    f a = g a ∧
    (a = 0 → g a = -1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1042_104200


namespace NUMINAMATH_CALUDE_reciprocal_opposite_equation_l1042_104270

theorem reciprocal_opposite_equation (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (a * b) ^ 4 - 3 * (c + d) ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_equation_l1042_104270


namespace NUMINAMATH_CALUDE_greg_needs_61_60_l1042_104250

/-- Calculates the additional amount Greg needs to buy a scooter, helmet, and lock -/
def additional_amount_needed (scooter_price helmet_price lock_price discount_rate tax_rate gift_card savings : ℚ) : ℚ :=
  let discounted_scooter := scooter_price * (1 - discount_rate)
  let subtotal := discounted_scooter + helmet_price + lock_price
  let total_with_tax := subtotal * (1 + tax_rate)
  let final_price := total_with_tax - gift_card
  final_price - savings

/-- Theorem stating that Greg needs $61.60 more -/
theorem greg_needs_61_60 :
  additional_amount_needed 90 30 15 0.1 0.1 20 57 = 61.6 := by
  sorry

end NUMINAMATH_CALUDE_greg_needs_61_60_l1042_104250


namespace NUMINAMATH_CALUDE_number_puzzle_2016_l1042_104255

theorem number_puzzle_2016 : ∃ (x y : ℕ), ∃ (z : ℕ), 
  x + y = 2016 ∧ 
  x = 10 * y + z ∧ 
  z < 10 ∧
  x = 1833 ∧ 
  y = 183 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_2016_l1042_104255


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l1042_104238

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + 1) / (2*x*(1-x))

theorem f_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) + f (1-x) = x :=
by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l1042_104238


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1042_104209

/-- An arithmetic sequence is a sequence where the difference between 
    each consecutive term is constant. -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_eighth_term 
  (seq : ArithmeticSequence) 
  (h4 : seq.nthTerm 4 = 23) 
  (h6 : seq.nthTerm 6 = 47) : 
  seq.nthTerm 8 = 71 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1042_104209


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1042_104275

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧
  b % 4 = 2 ∧
  b % 3 = 2 ∧
  b % 5 = 3 ∧
  (∀ (x : ℕ), x > 0 ∧ x % 4 = 2 ∧ x % 3 = 2 ∧ x % 5 = 3 → x ≥ b) ∧
  b = 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1042_104275


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1042_104241

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of basic flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops in each new flavor -/
def num_scoops : ℕ := 5

theorem ice_cream_flavors :
  distribute num_scoops num_flavors = Nat.choose (num_scoops + num_flavors - 1) (num_flavors - 1) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1042_104241


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1042_104263

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1042_104263


namespace NUMINAMATH_CALUDE_triangle_side_product_l1042_104232

theorem triangle_side_product (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = pi) →
  ((a + b)^2 - c^2 = 4) →
  (C = pi / 3) →
  (a * b = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_product_l1042_104232


namespace NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1042_104289

theorem possible_values_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 8) 
  (hb : |b| = 6) 
  (hab : |a + b| = a + b) : 
  a - b = 2 ∨ a - b = 14 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1042_104289


namespace NUMINAMATH_CALUDE_john_finish_time_l1042_104243

/-- The time it takes for John to finish the job by himself -/
def john_time : ℝ := 1.5

/-- The time it takes for David to finish the job by himself -/
def david_time : ℝ := 2 * john_time

/-- The time it takes for John and David to finish the job together -/
def combined_time : ℝ := 1

theorem john_finish_time :
  (1 / john_time + 1 / david_time) * combined_time = 1 ∧ david_time = 2 * john_time → john_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_john_finish_time_l1042_104243


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l1042_104249

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = t.b + 1 ∧ t.c = t.b - 1 ∧ t.A = 2 * t.C

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The main theorem to prove -/
theorem special_triangle_perimeter :
  ∀ t : Triangle, SpecialTriangle t → perimeter t = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_perimeter_l1042_104249


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1042_104213

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) 
  (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) : 
  12 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1042_104213


namespace NUMINAMATH_CALUDE_simplify_expression_l1042_104234

theorem simplify_expression (s t : ℝ) : 105 * s - 37 * s + 18 * t = 68 * s + 18 * t := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1042_104234


namespace NUMINAMATH_CALUDE_missing_number_solution_l1042_104285

theorem missing_number_solution : ∃ x : ℤ, 10111 - 10 * x * 5 = 10011 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_missing_number_solution_l1042_104285


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1042_104267

theorem smaller_circle_radius (r_large : ℝ) (r_small : ℝ) : 
  r_large = 4 →
  π * r_small^2 = (1/2) * π * r_large^2 →
  (π * r_small^2) + (π * r_large^2 - π * r_small^2) = 2 * (π * r_large^2 - π * r_small^2) →
  r_small = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1042_104267


namespace NUMINAMATH_CALUDE_shelf_filling_l1042_104204

theorem shelf_filling (A H C S M N E : ℕ) (x y z : ℝ) (l : ℝ) 
  (hA : A > 0) (hH : H > 0) (hC : C > 0) (hS : S > 0) (hM : M > 0) (hN : N > 0) (hE : E > 0)
  (hDistinct : A ≠ H ∧ A ≠ C ∧ A ≠ S ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧
               H ≠ C ∧ H ≠ S ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧
               C ≠ S ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧
               S ≠ M ∧ S ≠ N ∧ S ≠ E ∧
               M ≠ N ∧ M ≠ E ∧
               N ≠ E)
  (hThickness : 0 < x ∧ x < y ∧ x < z)
  (hFill1 : A * x + H * y + C * z = l)
  (hFill2 : S * x + M * y + N * z = l)
  (hFill3 : E * x = l) :
  E = (A * M + C * N - S * H - N * H) / (M + N - H) :=
sorry

end NUMINAMATH_CALUDE_shelf_filling_l1042_104204


namespace NUMINAMATH_CALUDE_four_birdhouses_built_l1042_104278

/-- The number of birdhouses that can be built with a given budget -/
def num_birdhouses (plank_cost nail_cost planks_per_house nails_per_house budget : ℚ) : ℚ :=
  budget / (plank_cost * planks_per_house + nail_cost * nails_per_house)

/-- Theorem stating that 4 birdhouses can be built with $88 given the specified costs and materials -/
theorem four_birdhouses_built :
  num_birdhouses 3 0.05 7 20 88 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_birdhouses_built_l1042_104278


namespace NUMINAMATH_CALUDE_fathers_age_multiple_l1042_104239

theorem fathers_age_multiple (son_age : ℕ) (father_age : ℕ) (k : ℕ) : 
  father_age = 27 →
  father_age = k * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_multiple_l1042_104239


namespace NUMINAMATH_CALUDE_office_to_bedroom_ratio_l1042_104273

/-- Represents the energy consumption of lights in watts per hour -/
structure LightEnergy where
  bedroom : ℝ
  office : ℝ
  livingRoom : ℝ

/-- Calculates the total energy used over a given number of hours -/
def totalEnergyUsed (l : LightEnergy) (hours : ℝ) : ℝ :=
  (l.bedroom + l.office + l.livingRoom) * hours

/-- Theorem stating the ratio of office light energy to bedroom light energy -/
theorem office_to_bedroom_ratio (l : LightEnergy) :
  l.bedroom = 6 →
  l.livingRoom = 4 * l.bedroom →
  totalEnergyUsed l 2 = 96 →
  l.office / l.bedroom = 3 := by
sorry

end NUMINAMATH_CALUDE_office_to_bedroom_ratio_l1042_104273


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1042_104266

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1042_104266


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1042_104215

/-- The line 4x + 3y + 9 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 9 = 0 ∧ y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1042_104215


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1042_104218

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (427751 - k) % 10 = 0 ∧ 
  ∀ (m : ℕ), m < k → (427751 - m) % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1042_104218


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l1042_104205

def p (x : ℝ) : ℝ := -3 * x^3 - 4 * x^2 - 8 * x + 2
def q (x : ℝ) : ℝ := -2 * x^2 - 7 * x + 3

theorem coefficient_x_squared_in_product :
  ∃ (a b c d e : ℝ), p x * q x = a * x^4 + b * x^3 + 40 * x^2 + d * x + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l1042_104205


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l1042_104221

/-- The radius of a cylinder with given conditions -/
def cylinder_radius : ℝ := 12

theorem cylinder_radius_proof (h : ℝ) (r : ℝ) :
  h = 4 →
  (π * (r + 4)^2 * h = π * r^2 * (h + 4)) →
  r = cylinder_radius := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l1042_104221


namespace NUMINAMATH_CALUDE_hypotenuse_square_l1042_104235

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z^3 + pz + q,
    and satisfy |a|^2 + |b|^2 + |c|^2 = 360, if the points corresponding to a, b, and c
    form a right triangle with the right angle at a, then the square of the length of
    the hypotenuse is 432. -/
theorem hypotenuse_square (a b c : ℂ) (p q : ℂ) :
  (a^3 + p*a + q = 0) →
  (b^3 + p*b + q = 0) →
  (c^3 + p*c + q = 0) →
  (Complex.abs a)^2 + (Complex.abs b)^2 + (Complex.abs c)^2 = 360 →
  (b - a) • (c - a) = 0 →  -- Right angle at a
  (Complex.abs (b - c))^2 = 432 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_square_l1042_104235


namespace NUMINAMATH_CALUDE_z_max_min_l1042_104262

def z (x y : ℝ) : ℝ := 2 * x + y

theorem z_max_min (x y : ℝ) (h1 : x + y ≤ 2) (h2 : x ≥ 1) (h3 : y ≥ 0) :
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≤ 4) ∧
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≥ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 4) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 2) :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_l1042_104262


namespace NUMINAMATH_CALUDE_probability_three_students_same_canteen_l1042_104276

/-- The probability of all three students going to the same canteen -/
def probability_same_canteen (num_canteens : ℕ) (num_students : ℕ) : ℚ :=
  if num_canteens = 2 ∧ num_students = 3 then
    1 / 4
  else
    0

/-- Theorem: The probability of all three students going to the same canteen is 1/4 -/
theorem probability_three_students_same_canteen :
  probability_same_canteen 2 3 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_students_same_canteen_l1042_104276


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1042_104212

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep / horses = 4 / 7 →
    horses * 230 = 12880 →
    sheep = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1042_104212


namespace NUMINAMATH_CALUDE_circle_properties_l1042_104269

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y + 4 = 0) :
  (∃ (k : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≤ k ∧ k = 4/3) ∧
  (∃ (m : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≥ m ∧ m = 0) ∧
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → x' + y' ≤ M ∧ M = 3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1042_104269


namespace NUMINAMATH_CALUDE_lottery_investment_ratio_l1042_104210

def lottery_winnings : ℕ := 12006
def savings_amount : ℕ := 1000
def fun_money : ℕ := 2802

theorem lottery_investment_ratio :
  let after_tax := lottery_winnings / 2
  let after_loans := after_tax - (after_tax / 3)
  let after_savings := after_loans - savings_amount
  let stock_investment := after_savings - fun_money
  (stock_investment : ℚ) / savings_amount = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_lottery_investment_ratio_l1042_104210


namespace NUMINAMATH_CALUDE_cubic_derivative_odd_implies_nonzero_l1042_104219

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem cubic_derivative_odd_implies_nonzero (a b c : ℝ) :
  is_odd (f' a b c) → a^2 + c^2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_cubic_derivative_odd_implies_nonzero_l1042_104219


namespace NUMINAMATH_CALUDE_distance_to_cegled_l1042_104245

/-- The problem setup for calculating the distance to Cegléd -/
structure TravelProblem where
  s : ℝ  -- Total distance from home to Cegléd
  v : ℝ  -- Planned speed for both Antal and Béla
  t : ℝ  -- Planned travel time
  s₁ : ℝ  -- Béla's travel distance when alone

/-- The conditions of the problem -/
def problem_conditions (p : TravelProblem) : Prop :=
  p.t = p.s / p.v ∧  -- Planned time
  p.s₁ = 4 * p.s / 5 ∧  -- Béla's solo distance
  p.s / 5 = 48 * (1 / 6) ∧  -- Final section travel time
  (4 * p.s₁) / (3 * p.v) = (4 * (p.s₁ + 2 * p.s / 5)) / (5 * p.v)  -- Time equivalence for travel

/-- The theorem stating that the total distance is 40 km -/
theorem distance_to_cegled (p : TravelProblem) 
  (h : problem_conditions p) : p.s = 40 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_cegled_l1042_104245


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1042_104287

theorem quadratic_equation_distinct_roots (k : ℝ) :
  k = 1 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - k = 0 ∧ 2 * x₂^2 - k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1042_104287


namespace NUMINAMATH_CALUDE_expression_value_l1042_104280

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 1 = 0) :
  (3*x + 2) * (3*x - 2) + x * (x - 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1042_104280
