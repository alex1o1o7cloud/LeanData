import Mathlib

namespace NUMINAMATH_CALUDE_specific_cone_properties_l302_30210

/-- Represents a cone with given height and slant height -/
structure Cone where
  height : ℝ
  slant_height : ℝ

/-- The central angle (in degrees) of the unfolded lateral surface of a cone -/
def central_angle (c : Cone) : ℝ := sorry

/-- The lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the properties of a specific cone -/
theorem specific_cone_properties :
  let c := Cone.mk (2 * Real.sqrt 2) 3
  central_angle c = 120 ∧ lateral_surface_area c = 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_cone_properties_l302_30210


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l302_30247

def M : Set ℝ := {2, 4, 6, 8}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l302_30247


namespace NUMINAMATH_CALUDE_chocolate_cost_in_dollars_l302_30228

/-- The cost of the chocolate in cents -/
def chocolate_cost (money_in_pocket : ℕ) (borrowed : ℕ) (needed : ℕ) : ℕ :=
  money_in_pocket * 100 + borrowed + needed

theorem chocolate_cost_in_dollars :
  let money_in_pocket : ℕ := 4
  let borrowed : ℕ := 59
  let needed : ℕ := 41
  (chocolate_cost money_in_pocket borrowed needed) / 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_in_dollars_l302_30228


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l302_30241

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ 
  Real.sqrt 50 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l302_30241


namespace NUMINAMATH_CALUDE_red_cards_probability_l302_30231

/-- The probability of drawing three red cards in succession from a deck of 60 cards,
    where 30 cards are red and 30 are black, is equal to 29/247. -/
theorem red_cards_probability (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 60) 
  (h2 : red_cards = 30) :
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 29 / 247 := by
  sorry

#eval (30 * 29 * 28) / (60 * 59 * 58)

end NUMINAMATH_CALUDE_red_cards_probability_l302_30231


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_two_l302_30243

def point_on_number_line (x : ℝ) := True

theorem point_three_units_from_negative_two (A : ℝ) :
  point_on_number_line A →
  |A - (-2)| = 3 →
  A = -5 ∨ A = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_two_l302_30243


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l302_30269

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of times 'A' appears in "BANANA" -/
def a_count : ℕ := 3

/-- The number of times 'N' appears in "BANANA" -/
def n_count : ℕ := 2

/-- The number of times 'B' appears in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l302_30269


namespace NUMINAMATH_CALUDE_josephs_birth_year_l302_30296

-- Define the year of the first revised AMC 8
def first_revised_amc8_year : ℕ := 1987

-- Define Joseph's age when he took the seventh AMC 8
def josephs_age_at_seventh_amc8 : ℕ := 15

-- Define the number of years between the first and seventh AMC 8
def years_between_first_and_seventh_amc8 : ℕ := 6

-- Theorem to prove Joseph's birth year
theorem josephs_birth_year : 
  first_revised_amc8_year + years_between_first_and_seventh_amc8 - josephs_age_at_seventh_amc8 = 1978 := by
  sorry

end NUMINAMATH_CALUDE_josephs_birth_year_l302_30296


namespace NUMINAMATH_CALUDE_stratified_sample_B_size_l302_30285

/-- Represents the number of individuals in each level -/
structure PopulationLevels where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the total population -/
def total_population (p : PopulationLevels) : ℕ := p.A + p.B + p.C

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_sample : ℕ
  population : PopulationLevels

/-- Calculates the number of individuals to be sampled from a specific level -/
def sample_size_for_level (s : StratifiedSample) (level_size : ℕ) : ℕ :=
  (s.total_sample * level_size) / (total_population s.population)

theorem stratified_sample_B_size 
  (sample : StratifiedSample) 
  (h1 : sample.population.A = 5 * n)
  (h2 : sample.population.B = 3 * n)
  (h3 : sample.population.C = 2 * n)
  (h4 : sample.total_sample = 150)
  (n : ℕ) :
  sample_size_for_level sample sample.population.B = 45 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_B_size_l302_30285


namespace NUMINAMATH_CALUDE_simplify_expression_l302_30226

theorem simplify_expression (b c : ℝ) : 
  (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) * (7 * c^2) = 5040 * b^10 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l302_30226


namespace NUMINAMATH_CALUDE_min_value_of_s_l302_30229

theorem min_value_of_s (a b : ℤ) :
  let s := a^3 + b^3 - 60*a*b*(a + b)
  s ≥ 2012 → s ≥ 2015 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_s_l302_30229


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l302_30286

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 72 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l302_30286


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l302_30217

/-- Calculates the principal amount given the interest rate, time, and total interest --/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given a loan with 12% per annum simple interest rate, 
    if the interest after 3 years is 4320, then the principal amount borrowed was 12000 --/
theorem loan_principal_calculation :
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 4320
  calculate_principal rate time interest = 12000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l302_30217


namespace NUMINAMATH_CALUDE_don_earlier_rum_amount_l302_30295

/-- The amount of rum Don had earlier in the day -/
def donEarlierRum (pancakeRum : ℝ) (maxMultiplier : ℝ) (remainingRum : ℝ) : ℝ :=
  maxMultiplier * pancakeRum - (pancakeRum + remainingRum)

/-- Theorem stating that Don had 12oz of rum earlier that day -/
theorem don_earlier_rum_amount :
  donEarlierRum 10 3 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_don_earlier_rum_amount_l302_30295


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_inscribed_circle_l302_30291

/-- 
Given a right triangle with an inscribed circle, where the point of tangency 
divides one of the legs into segments of lengths m and n (m < n), 
the hypotenuse of the triangle is (m^2 + n^2) / (n - m).
-/
theorem hypotenuse_of_right_triangle_with_inscribed_circle 
  (m n : ℝ) (h : m < n) : ∃ (x : ℝ), 
  x > 0 ∧ 
  x = (m^2 + n^2) / (n - m) ∧
  x^2 = (x - n + m)^2 + (m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_inscribed_circle_l302_30291


namespace NUMINAMATH_CALUDE_cubic_monomial_properties_l302_30273

/-- A cubic monomial with coefficient -2 using only variables x and y -/
def cubic_monomial (x y : ℝ) : ℝ := -2 * x^2 * y

theorem cubic_monomial_properties (x y : ℝ) :
  ∃ (a b c : ℕ), a + b + c = 3 ∧ cubic_monomial x y = -2 * x^a * y^b := by
  sorry

end NUMINAMATH_CALUDE_cubic_monomial_properties_l302_30273


namespace NUMINAMATH_CALUDE_last_painted_cell_l302_30280

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def is_painted (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

def covers_all_columns (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 → i ≤ 8 → ∃ k : ℕ, k ≤ n ∧ is_painted k ∧ k % 8 = i

theorem last_painted_cell :
  ∃ n : ℕ, n = 120 ∧ is_painted n ∧ covers_all_columns n ∧
  ∀ m : ℕ, m < n → ¬(covers_all_columns m) :=
sorry

end NUMINAMATH_CALUDE_last_painted_cell_l302_30280


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l302_30206

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (2 - x) - 3 = x / (x - 2) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l302_30206


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_l302_30288

theorem pizza_slices_per_pizza (num_people : ℕ) (slices_per_person : ℕ) (num_pizzas : ℕ) : 
  num_people = 18 → slices_per_person = 3 → num_pizzas = 6 →
  (num_people * slices_per_person) / num_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_l302_30288


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l302_30238

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l302_30238


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l302_30218

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fifth_term_of_sequence (y : ℝ) :
  let a₁ := 3
  let r := 3 * y
  geometric_sequence a₁ r 5 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l302_30218


namespace NUMINAMATH_CALUDE_triangle_inequality_l302_30277

theorem triangle_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2)^2 ≥ 4 * Real.tan (A / 2) * (Real.tan (A / 2) * Real.tan (C / 2) - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l302_30277


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l302_30272

/-- Given a box with 10 balls, where 1 is yellow, 3 are green, and the rest are red,
    the probability of randomly drawing a red ball is 3/5. -/
theorem probability_of_red_ball (total_balls : ℕ) (yellow_balls : ℕ) (green_balls : ℕ) :
  total_balls = 10 →
  yellow_balls = 1 →
  green_balls = 3 →
  (total_balls - yellow_balls - green_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

#check probability_of_red_ball

end NUMINAMATH_CALUDE_probability_of_red_ball_l302_30272


namespace NUMINAMATH_CALUDE_fourth_power_sum_of_roots_l302_30289

theorem fourth_power_sum_of_roots (r₁ r₂ r₃ r₄ : ℝ) : 
  (r₁^4 - r₁ - 504 = 0) → 
  (r₂^4 - r₂ - 504 = 0) → 
  (r₃^4 - r₃ - 504 = 0) → 
  (r₄^4 - r₄ - 504 = 0) → 
  r₁^4 + r₂^4 + r₃^4 + r₄^4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_of_roots_l302_30289


namespace NUMINAMATH_CALUDE_complex_addition_subtraction_l302_30239

theorem complex_addition_subtraction : 
  (1 : ℂ) * (5 - 6 * I) + (-2 - 2 * I) - (3 + 3 * I) = -11 * I := by sorry

end NUMINAMATH_CALUDE_complex_addition_subtraction_l302_30239


namespace NUMINAMATH_CALUDE_circle_segment_area_l302_30212

theorem circle_segment_area (R : ℝ) (R_pos : R > 0) : 
  let circle_area := π * R^2
  let square_side := R * Real.sqrt 2
  let square_area := square_side^2
  let segment_area := (circle_area - square_area) / 4
  segment_area = R^2 * (π - 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_segment_area_l302_30212


namespace NUMINAMATH_CALUDE_purchases_total_price_l302_30227

/-- The total price of a refrigerator and a washing machine -/
def total_price (refrigerator_price washing_machine_price : ℕ) : ℕ :=
  refrigerator_price + washing_machine_price

/-- Theorem: The total price of the purchases is $7060 -/
theorem purchases_total_price :
  let refrigerator_price : ℕ := 4275
  let washing_machine_price : ℕ := refrigerator_price - 1490
  total_price refrigerator_price washing_machine_price = 7060 := by
sorry

end NUMINAMATH_CALUDE_purchases_total_price_l302_30227


namespace NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l302_30216

/-- Given three consecutive even integers whose squares sum to 2930, 
    prove that the sum of their cubes is 81720 -/
theorem consecutive_even_integers_cube_sum (n : ℤ) : 
  (∃ (n : ℤ), 
    (n^2 + (n+2)^2 + (n+4)^2 = 2930) ∧ 
    (∃ (k : ℤ), n = 2*k)) →
  n^3 + (n+2)^3 + (n+4)^3 = 81720 :=
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l302_30216


namespace NUMINAMATH_CALUDE_largest_root_ratio_l302_30274

def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

def largest_root (p : ℝ → ℝ) : ℝ := sorry

theorem largest_root_ratio : 
  largest_root g / largest_root f = 2 := by sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l302_30274


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l302_30237

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l302_30237


namespace NUMINAMATH_CALUDE_area_of_K_l302_30287

/-- The set K in the plane Cartesian coordinate system xOy -/
def K : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

/-- The area of set K -/
theorem area_of_K : MeasureTheory.volume K = 24 := by
  sorry

end NUMINAMATH_CALUDE_area_of_K_l302_30287


namespace NUMINAMATH_CALUDE_lcm_54_75_l302_30244

theorem lcm_54_75 : Nat.lcm 54 75 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_54_75_l302_30244


namespace NUMINAMATH_CALUDE_total_shoes_needed_l302_30246

def num_dogs : ℕ := 3
def num_cats : ℕ := 2
def num_ferrets : ℕ := 1
def paws_per_animal : ℕ := 4

theorem total_shoes_needed : 
  (num_dogs + num_cats + num_ferrets) * paws_per_animal = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_needed_l302_30246


namespace NUMINAMATH_CALUDE_tan_a_value_l302_30251

theorem tan_a_value (a : Real) (h : Real.tan (a + π/4) = 1/7) : 
  Real.tan a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_value_l302_30251


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l302_30255

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 22 + 5 + y) / 5 = 12 → y = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l302_30255


namespace NUMINAMATH_CALUDE_completing_square_result_l302_30281

theorem completing_square_result (x : ℝ) : 
  x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l302_30281


namespace NUMINAMATH_CALUDE_cylinder_volume_with_inscribed_sphere_l302_30294

/-- The volume of a cylinder with an inscribed sphere (tangent to top, bottom, and side) is 2π, 
    given that the volume of the inscribed sphere is 4π/3. -/
theorem cylinder_volume_with_inscribed_sphere (r : ℝ) (h : ℝ) :
  (4 / 3 * π * r^3 = 4 * π / 3) →
  (π * r^2 * h = 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_with_inscribed_sphere_l302_30294


namespace NUMINAMATH_CALUDE_irrationality_of_cube_plus_sqrt_two_l302_30253

theorem irrationality_of_cube_plus_sqrt_two (t : ℝ) :
  (∃ (r : ℚ), t + Real.sqrt 2 = r) → ¬ (∃ (s : ℚ), t^3 + Real.sqrt 2 = s) := by
sorry

end NUMINAMATH_CALUDE_irrationality_of_cube_plus_sqrt_two_l302_30253


namespace NUMINAMATH_CALUDE_no_common_solution_exists_l302_30263

/-- A_{n}^k denotes the number of k-permutations of n elements -/
def A (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- C_{n}^k denotes the number of k-combinations of n elements -/
def C (n k : ℕ) : ℕ := Nat.choose n k

theorem no_common_solution_exists : ¬ ∃ (n : ℕ), n ≥ 3 ∧ 
  A (2*n) 3 = 2 * A (n+1) 4 ∧ 
  C (n+2) (n-2) + C (n+2) (n-3) = (A (n+3) 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_exists_l302_30263


namespace NUMINAMATH_CALUDE_chimps_in_old_cage_l302_30245

/-- The number of chimps staying in the old cage is equal to the total number of chimps minus the number of chimps being moved. -/
theorem chimps_in_old_cage (total_chimps moving_chimps : ℕ) :
  total_chimps ≥ moving_chimps →
  total_chimps - moving_chimps = total_chimps - moving_chimps :=
by
  sorry

#check chimps_in_old_cage 45 18

end NUMINAMATH_CALUDE_chimps_in_old_cage_l302_30245


namespace NUMINAMATH_CALUDE_apple_ratio_simplification_l302_30258

def sarah_apples : ℕ := 630
def brother_apples : ℕ := 270
def cousin_apples : ℕ := 540

theorem apple_ratio_simplification :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    sarah_apples / k = 7 ∧ 
    brother_apples / k = 3 ∧ 
    cousin_apples / k = 6 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_simplification_l302_30258


namespace NUMINAMATH_CALUDE_rectangle_45_odd_intersections_impossible_l302_30283

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a grid line -/
def isOnGridLine (p : Point) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if two line segments are at 45° angle to each other -/
def isAt45Degree (p1 p2 q1 q2 : Point) : Prop :=
  |p2.x - p1.x| = |p2.y - p1.y| ∧ |q2.x - q1.x| = |q2.y - q1.y|

/-- Counts the number of grid line intersections for a line segment -/
def gridIntersections (p1 p2 : Point) : ℕ :=
  sorry

/-- Main theorem: It's impossible for all sides of a 45° rectangle to intersect an odd number of grid lines -/
theorem rectangle_45_odd_intersections_impossible (rect : Rectangle) :
  (¬ isOnGridLine rect.A) →
  (¬ isOnGridLine rect.B) →
  (¬ isOnGridLine rect.C) →
  (¬ isOnGridLine rect.D) →
  isAt45Degree rect.A rect.B rect.B rect.C →
  ¬ (Odd (gridIntersections rect.A rect.B) ∧
     Odd (gridIntersections rect.B rect.C) ∧
     Odd (gridIntersections rect.C rect.D) ∧
     Odd (gridIntersections rect.D rect.A)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_45_odd_intersections_impossible_l302_30283


namespace NUMINAMATH_CALUDE_january_oil_bill_l302_30271

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 30) / january_bill = 5 / 3) →
  january_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l302_30271


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_is_11_l302_30233

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def is_simplest_quadratic_radical (a : ℚ) : Prop :=
  a > 0 ∧ ¬(is_perfect_square a) ∧
  ∀ b c : ℚ, (b > 1 ∧ c > 0 ∧ a = b * c) → ¬(is_perfect_square b)

-- Theorem statement
theorem simplest_quadratic_radical_is_11 :
  is_simplest_quadratic_radical 11 ∧
  ¬(is_simplest_quadratic_radical (5/2)) ∧
  ¬(is_simplest_quadratic_radical 12) ∧
  ¬(is_simplest_quadratic_radical (1/3)) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_is_11_l302_30233


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l302_30250

theorem definite_integral_sin_plus_one :
  ∫ x in (-1)..(1), (Real.sin x + 1) = 2 - 2 * Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l302_30250


namespace NUMINAMATH_CALUDE_ellipse_equation_l302_30290

/-- Given an ellipse with the endpoint of its short axis at (3, 0) and focal distance 4,
    prove that its equation is (y²/25) + (x²/9) = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let short_axis_endpoint : ℝ × ℝ := (3, 0)
  let focal_distance : ℝ := 4
  (y^2 / 25) + (x^2 / 9) = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_equation_l302_30290


namespace NUMINAMATH_CALUDE_set_size_comparison_l302_30260

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_set_size_comparison_l302_30260


namespace NUMINAMATH_CALUDE_new_person_weight_l302_30266

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 100 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l302_30266


namespace NUMINAMATH_CALUDE_fraction_product_l302_30207

theorem fraction_product : (5/8 : ℚ) * (7/9 : ℚ) * (11/13 : ℚ) * (3/5 : ℚ) * (17/19 : ℚ) * (8/15 : ℚ) = 14280/1107000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l302_30207


namespace NUMINAMATH_CALUDE_janet_total_earnings_l302_30219

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℝ) (sculpture_rate : ℝ) (hours_worked : ℝ) 
                   (sculpture1_weight : ℝ) (sculpture2_weight : ℝ) : ℝ :=
  exterminator_rate * hours_worked + 
  sculpture_rate * (sculpture1_weight + sculpture2_weight)

/-- Proves that Janet's total earnings are $1640 given the specified conditions -/
theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_earnings_l302_30219


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l302_30221

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∀ n : ℕ, n > 0 → 
  (¬ is_prime n) → 
  (¬ is_square n) → 
  (smallest_prime_factor n ≥ 60) → 
  n ≥ 4087 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l302_30221


namespace NUMINAMATH_CALUDE_gemma_tip_calculation_l302_30222

/-- Calculates the tip given to a delivery person based on the number of pizzas ordered,
    the cost per pizza, the amount paid, and the change received. -/
def calculate_tip (num_pizzas : ℕ) (cost_per_pizza : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  amount_paid - change - (num_pizzas * cost_per_pizza)

/-- Proves that given the specified conditions, the tip Gemma gave to the delivery person was $5. -/
theorem gemma_tip_calculation :
  let num_pizzas : ℕ := 4
  let cost_per_pizza : ℕ := 10
  let amount_paid : ℕ := 50
  let change : ℕ := 5
  calculate_tip num_pizzas cost_per_pizza amount_paid change = 5 := by
  sorry

#eval calculate_tip 4 10 50 5

end NUMINAMATH_CALUDE_gemma_tip_calculation_l302_30222


namespace NUMINAMATH_CALUDE_salary_increase_l302_30264

theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) : 
  new_salary = 120 ∧ increase_percentage = 100 → old_salary = 60 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l302_30264


namespace NUMINAMATH_CALUDE_exists_alpha_congruence_l302_30205

theorem exists_alpha_congruence : ∃ α : ℤ, α ^ 2 ≡ 2 [ZMOD 7^3] ∧ α ≡ 3 [ZMOD 7] :=
sorry

end NUMINAMATH_CALUDE_exists_alpha_congruence_l302_30205


namespace NUMINAMATH_CALUDE_max_value_theorem_l302_30225

theorem max_value_theorem (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 200) :
  ∃ (max_value : ℝ), max_value = 30000 ∧ 
  ∀ (x y z w : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → 
  x + y + z + w = 200 → 2*x*y + 3*y*z + 4*z*w ≤ max_value :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l302_30225


namespace NUMINAMATH_CALUDE_complement_and_intersection_l302_30282

def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}
def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem complement_and_intersection :
  (U \ A = {8, 10}) ∧ (A ∩ (U \ B) = {4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_complement_and_intersection_l302_30282


namespace NUMINAMATH_CALUDE_original_paint_intensity_l302_30223

/-- Given a paint mixture scenario, prove that the original paint intensity was 50%. -/
theorem original_paint_intensity
  (replacement_intensity : ℝ)
  (final_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h1 : replacement_intensity = 25)
  (h2 : final_intensity = 30)
  (h3 : replaced_fraction = 0.8)
  : (1 - replaced_fraction) * 50 + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#check original_paint_intensity

end NUMINAMATH_CALUDE_original_paint_intensity_l302_30223


namespace NUMINAMATH_CALUDE_fold_line_equation_l302_30298

/-- The perpendicular bisector of the line segment joining two points (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem fold_line_equation :
  perpendicular_bisector 5 3 1 (-1) = {p : ℝ × ℝ | p.2 = -p.1 + 4} := by
  sorry

end NUMINAMATH_CALUDE_fold_line_equation_l302_30298


namespace NUMINAMATH_CALUDE_pqr_product_l302_30240

theorem pqr_product (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → 
  p + q + r = 27 → 
  1 / p + 1 / q + 1 / r + 432 / (p * q * r) = 1 → 
  p * q * r = 1380 := by
sorry

end NUMINAMATH_CALUDE_pqr_product_l302_30240


namespace NUMINAMATH_CALUDE_problem_solution_l302_30254

theorem problem_solution : 
  (∃ x : ℚ, x - 2/11 = -1/3 ∧ x = -5/33) ∧ 
  (-2 - (-1/3 + 1/2) = -13/6) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l302_30254


namespace NUMINAMATH_CALUDE_equalize_payment_is_two_l302_30265

/-- The amount B should give A to equalize payments when buying basketballs -/
def equalize_payment (n : ℕ+) : ℚ :=
  let total_cost := n.val * n.val
  let full_payments := total_cost / 10
  let remainder := total_cost % 10
  if remainder = 0 then 0
  else (10 - remainder) / 2

theorem equalize_payment_is_two (n : ℕ+) : equalize_payment n = 2 := by
  sorry


end NUMINAMATH_CALUDE_equalize_payment_is_two_l302_30265


namespace NUMINAMATH_CALUDE_four_digit_sum_27_eq_3276_l302_30232

/-- The number of four-digit whole numbers whose digits sum to 27 -/
def four_digit_sum_27 : ℕ :=
  (Finset.range 10).sum (fun a =>
    (Finset.range 10).sum (fun b =>
      (Finset.range 10).sum (fun c =>
        (Finset.range 10).sum (fun d =>
          if a ≥ 1 ∧ a + b + c + d = 27 then 1 else 0))))

theorem four_digit_sum_27_eq_3276 : four_digit_sum_27 = 3276 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_27_eq_3276_l302_30232


namespace NUMINAMATH_CALUDE_max_students_distribution_l302_30293

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1008) (h2 : pencils = 928) :
  (Nat.gcd pens pencils : ℕ) = 16 := by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l302_30293


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_28_l302_30259

theorem largest_five_digit_congruent_to_17_mod_28 : ∃ x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  x ≡ 17 [MOD 28] ∧
  (∀ y : ℕ, (y ≥ 10000 ∧ y < 100000) ∧ y ≡ 17 [MOD 28] → y ≤ x) ∧
  x = 99947 := by
sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_28_l302_30259


namespace NUMINAMATH_CALUDE_trip_average_speed_l302_30209

/-- Calculates the average speed given three segments of a trip -/
def average_speed (d1 d2 d3 t1 t2 t3 : ℚ) : ℚ :=
  (d1 + d2 + d3) / (t1 + t2 + t3)

/-- Theorem: The average speed for the given trip is 1200/18 miles per hour -/
theorem trip_average_speed :
  average_speed 420 480 300 6 7 5 = 1200 / 18 := by
  sorry

end NUMINAMATH_CALUDE_trip_average_speed_l302_30209


namespace NUMINAMATH_CALUDE_problem_statement_l302_30220

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : b - c = 2) :
  (a - c)^2 + 3*a + 1 - 3*c = 41 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l302_30220


namespace NUMINAMATH_CALUDE_calcium_atomic_weight_l302_30284

/-- The atomic weight of Oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Calcium Oxide (CaO) -/
def molecular_weight_CaO : ℝ := 56

/-- The atomic weight of Calcium -/
def atomic_weight_Ca : ℝ := molecular_weight_CaO - atomic_weight_O

/-- Theorem stating that the atomic weight of Calcium is 40 -/
theorem calcium_atomic_weight :
  atomic_weight_Ca = 40 := by sorry

end NUMINAMATH_CALUDE_calcium_atomic_weight_l302_30284


namespace NUMINAMATH_CALUDE_vacation_cost_theorem_l302_30201

/-- Calculates the total cost of a vacation in USD given specific expenses and exchange rates -/
def vacation_cost (num_people : ℕ) 
                  (rent_per_person : ℝ) 
                  (transport_per_person : ℝ) 
                  (food_per_person : ℝ) 
                  (activities_per_person : ℝ) 
                  (euro_to_usd : ℝ) 
                  (pound_to_usd : ℝ) 
                  (yen_to_usd : ℝ) : ℝ :=
  let total_rent := num_people * rent_per_person * euro_to_usd
  let total_transport := num_people * transport_per_person
  let total_food := num_people * food_per_person * pound_to_usd
  let total_activities := num_people * activities_per_person * yen_to_usd
  total_rent + total_transport + total_food + total_activities

/-- The total cost of the vacation is $1384.25 -/
theorem vacation_cost_theorem : 
  vacation_cost 7 65 25 50 2750 1.2 1.4 0.009 = 1384.25 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_theorem_l302_30201


namespace NUMINAMATH_CALUDE_solution_equals_expected_l302_30297

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points satisfying x ⋆ y = y ⋆ x
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the union of x-axis, y-axis, and lines y = x and y = -x
def expected_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem solution_equals_expected : solution_set = expected_set := by
  sorry

end NUMINAMATH_CALUDE_solution_equals_expected_l302_30297


namespace NUMINAMATH_CALUDE_divisibility_by_24_l302_30299

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l302_30299


namespace NUMINAMATH_CALUDE_p_bounds_l302_30214

/-- Represents the minimum number of reconstructions needed to transform
    one triangulation into another for a convex n-gon. -/
def p (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds on p(n) for convex n-gons. -/
theorem p_bounds (n : ℕ) : 
  n ≥ 3 → 
  p n ≥ n - 3 ∧ 
  p n ≤ 2*n - 7 ∧ 
  (n ≥ 13 → p n ≤ 2*n - 10) := by sorry


end NUMINAMATH_CALUDE_p_bounds_l302_30214


namespace NUMINAMATH_CALUDE_zoom_download_time_ratio_l302_30248

/-- Prove that the ratio of Windows download time to Mac download time is 3:1 -/
theorem zoom_download_time_ratio :
  let total_time := 82
  let mac_download_time := 10
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let glitch_time := audio_glitch_time + video_glitch_time
  let no_glitch_time := 2 * glitch_time
  let windows_download_time := total_time - (mac_download_time + glitch_time + no_glitch_time)
  (windows_download_time : ℚ) / mac_download_time = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_zoom_download_time_ratio_l302_30248


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l302_30268

/-- Given that θ is an internal angle of a triangle and sin θ + cos θ = 1/2,
    prove that x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an internal angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 1/2) :
  ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    ∀ (x y : Real), 
      x^2 * Real.sin θ - y^2 * Real.cos θ = 1 ↔ 
      (x^2 / a^2) + (y^2 / b^2) = 1 ∧
      a < b :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l302_30268


namespace NUMINAMATH_CALUDE_target_line_is_perpendicular_l302_30252

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x y : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    (A * 3 + B * 4 + C = 0) ∧ 
    (A * B = -1) ∧
    (A * 2 + B * (-1) = 0)

/-- The specific line we're looking for -/
def target_line (x y : ℝ) : Prop :=
  x + 2 * y - 11 = 0

/-- Theorem stating that the target line satisfies the conditions -/
theorem target_line_is_perpendicular : 
  perpendicular_line 3 4 ↔ target_line 3 4 :=
sorry

end NUMINAMATH_CALUDE_target_line_is_perpendicular_l302_30252


namespace NUMINAMATH_CALUDE_equation_solution_l302_30235

theorem equation_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 20) = (x^2 - 3*x - 18) / (x^2 - 2*x - 35) ↔ 
  x = 4 + Real.sqrt 21 ∨ x = 4 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l302_30235


namespace NUMINAMATH_CALUDE_total_weight_of_goods_l302_30276

theorem total_weight_of_goods (x : ℝ) 
  (h1 : (x - 10) / 7 = (x + 5) / 8) : x = 115 := by
  sorry

#check total_weight_of_goods

end NUMINAMATH_CALUDE_total_weight_of_goods_l302_30276


namespace NUMINAMATH_CALUDE_parallelogram_x_value_l302_30234

/-- A parallelogram ABCD with specific properties -/
structure Parallelogram where
  x : ℝ
  area : ℝ
  h : x > 0
  angle : ℝ
  h_angle : angle = 30 * π / 180
  h_area : area = 35

/-- The theorem stating that x = 14 for the given parallelogram -/
theorem parallelogram_x_value (p : Parallelogram) : p.x = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_x_value_l302_30234


namespace NUMINAMATH_CALUDE_square_difference_hundred_l302_30202

theorem square_difference_hundred : ∃ x y : ℤ, x^2 - y^2 = 100 ∧ x = 26 ∧ y = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_hundred_l302_30202


namespace NUMINAMATH_CALUDE_system_of_inequalities_l302_30215

theorem system_of_inequalities (x : ℝ) :
  3 * (x + 1) > 5 * x + 4 ∧ (x - 1) / 2 ≤ (2 * x - 1) / 3 → -1 ≤ x ∧ x < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l302_30215


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_is_nine_fourteenths_l302_30256

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℚ :=
  let total_hours_per_week : ℕ := 7 * 24
  let weekday_reduced_hours : ℕ := 5 * 12
  let weekend_reduced_hours : ℕ := 2 * 24
  let total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours
  ↑total_reduced_hours / ↑total_hours_per_week

/-- Proof that the reduced rate fraction is 9/14 -/
theorem reduced_rate_fraction_is_nine_fourteenths :
  reduced_rate_fraction = 9 / 14 :=
by sorry

end NUMINAMATH_CALUDE_reduced_rate_fraction_is_nine_fourteenths_l302_30256


namespace NUMINAMATH_CALUDE_notebooks_divisible_by_three_l302_30200

/-- A family is preparing backpacks with school supplies. -/
structure SchoolSupplies where
  pencils : ℕ
  notebooks : ℕ
  backpacks : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : SchoolSupplies) : Prop :=
  s.pencils = 9 ∧
  s.backpacks = 3 ∧
  s.pencils % s.backpacks = 0 ∧
  s.notebooks % s.backpacks = 0

/-- Theorem stating that the number of notebooks must be divisible by 3 -/
theorem notebooks_divisible_by_three (s : SchoolSupplies) 
  (h : problem_conditions s) : 
  s.notebooks % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_divisible_by_three_l302_30200


namespace NUMINAMATH_CALUDE_gumball_problem_l302_30292

theorem gumball_problem (alicia_gumballs : ℕ) : 
  alicia_gumballs = 20 →
  let pedro_gumballs := alicia_gumballs + (alicia_gumballs * 3 / 2)
  let maria_gumballs := pedro_gumballs / 2
  let alicia_eaten := alicia_gumballs / 3
  let pedro_eaten := pedro_gumballs / 3
  let maria_eaten := maria_gumballs / 3
  (alicia_gumballs - alicia_eaten) + (pedro_gumballs - pedro_eaten) + (maria_gumballs - maria_eaten) = 65 := by
sorry

end NUMINAMATH_CALUDE_gumball_problem_l302_30292


namespace NUMINAMATH_CALUDE_two_integers_sum_l302_30204

theorem two_integers_sum (x y : ℤ) (h1 : x - y = 1) (h2 : x = -4) (h3 : y = -5) : x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l302_30204


namespace NUMINAMATH_CALUDE_right_triangle_count_l302_30203

/-- Represents a right triangle with integer vertices and right angle at the origin -/
structure RightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ

/-- Checks if a point is the incenter of a right triangle -/
def is_incenter (t : RightTriangle) (m : ℚ × ℚ) : Prop :=
  sorry

/-- Counts the number of right triangles with given incenter -/
def count_triangles (p : ℕ) : ℕ :=
  sorry

theorem right_triangle_count (p : ℕ) (h : Nat.Prime p) :
  count_triangles p = 108 ∨ count_triangles p = 42 ∨ count_triangles p = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_count_l302_30203


namespace NUMINAMATH_CALUDE_product_correction_l302_30270

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem product_correction (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 → -- a is a three-digit number
  (reverseDigits a) * b = 468 → -- incorrect calculation
  a * b = 1116 := by sorry

end NUMINAMATH_CALUDE_product_correction_l302_30270


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l302_30261

theorem quadratic_roots_range (k : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + k*x₁ - k = 0 ∧ x₂^2 + k*x₂ - k = 0) →
  (1 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3) →
  -9/2 < k ∧ k < -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l302_30261


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l302_30275

/-- Given a sequence a_n where a_1 = 2 and {1+a_n} is a geometric sequence
    with common ratio 3, prove that a_4 = 80 -/
theorem sequence_fourth_term (a : ℕ → ℝ) : 
  a 1 = 2 ∧ 
  (∀ n : ℕ, (1 + a (n + 1)) = 3 * (1 + a n)) →
  a 4 = 80 := by
sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l302_30275


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l302_30224

theorem termite_ridden_not_collapsing (total_homes : ℕ) 
  (termite_ridden : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden = 1 / 3 →
  collapsing_ratio = 7 / 10 →
  (termite_ridden - termite_ridden * collapsing_ratio) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l302_30224


namespace NUMINAMATH_CALUDE_secret_spread_reaches_target_l302_30262

/-- Represents the number of students who know the secret on a given day -/
def secret_spread : ℕ → ℕ
| 0 => 4  -- Monday (day 0): Jessica + 3 friends
| 1 => 10 -- Tuesday (day 1): Previous + 2 * 3 new
| 2 => 22 -- Wednesday (day 2): Previous + 3 * 3 + 3 new
| n + 3 => secret_spread (n + 2) + 3 * (secret_spread (n + 2) - secret_spread (n + 1))

/-- The day when the secret reaches at least 7280 students -/
def target_day : ℕ := 9

theorem secret_spread_reaches_target :
  secret_spread target_day ≥ 7280 := by
  sorry


end NUMINAMATH_CALUDE_secret_spread_reaches_target_l302_30262


namespace NUMINAMATH_CALUDE_rectangle_area_l302_30257

/-- Given a rectangle with diagonal length y and length three times its width, 
    prove that its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (h : y > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l302_30257


namespace NUMINAMATH_CALUDE_family_members_count_l302_30211

def num_birds : ℕ := 4
def num_dogs : ℕ := 3
def num_cats : ℕ := 18

def bird_feet : ℕ := 2
def dog_feet : ℕ := 4
def cat_feet : ℕ := 4

def animal_heads : ℕ := num_birds + num_dogs + num_cats

def animal_feet : ℕ := num_birds * bird_feet + num_dogs * dog_feet + num_cats * cat_feet

def human_feet : ℕ := 2
def human_head : ℕ := 1

theorem family_members_count :
  ∃ (F : ℕ), animal_feet + F * human_feet = animal_heads + F * human_head + 74 ∧ F = 7 := by
  sorry

end NUMINAMATH_CALUDE_family_members_count_l302_30211


namespace NUMINAMATH_CALUDE_friends_total_skittles_l302_30242

/-- Given a person who gives a certain number of Skittles to each of their friends,
    calculate the total number of Skittles their friends have. -/
def total_skittles (skittles_per_friend : ℝ) (num_friends : ℝ) : ℝ :=
  skittles_per_friend * num_friends

/-- Theorem stating that if a person gives 40.0 Skittles to each of their 5.0 friends,
    the total number of Skittles their friends have is 200.0. -/
theorem friends_total_skittles :
  total_skittles 40.0 5.0 = 200.0 := by
  sorry

end NUMINAMATH_CALUDE_friends_total_skittles_l302_30242


namespace NUMINAMATH_CALUDE_cheerleader_size6_count_l302_30213

/-- Represents the number of cheerleaders needing each uniform size -/
structure CheerleaderSizes where
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The conditions of the cheerleader uniform problem -/
def cheerleader_uniform_problem (s : CheerleaderSizes) : Prop :=
  s.size2 = 4 ∧
  s.size12 * 2 = s.size6 ∧
  s.size2 + s.size6 + s.size12 = 19

/-- The theorem stating the solution to the cheerleader uniform problem -/
theorem cheerleader_size6_count :
  ∃ s : CheerleaderSizes, cheerleader_uniform_problem s ∧ s.size6 = 10 :=
sorry

end NUMINAMATH_CALUDE_cheerleader_size6_count_l302_30213


namespace NUMINAMATH_CALUDE_congruence_problem_l302_30249

theorem congruence_problem (c d m : ℤ) : 
  c ≡ 25 [ZMOD 53] →
  d ≡ 98 [ZMOD 53] →
  m ∈ Finset.Icc 150 200 →
  (c - d ≡ m [ZMOD 53] ↔ m = 192) := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l302_30249


namespace NUMINAMATH_CALUDE_no_rational_roots_for_all_quadratics_l302_30208

/-- The largest known prime number -/
def p : ℕ := 2^24036583 - 1

/-- Theorem stating that there are no positive integers c such that
    both p^2 - 4c and p^2 + 4c are perfect squares -/
theorem no_rational_roots_for_all_quadratics :
  ¬∃ c : ℕ+, ∃ a b : ℕ, (p^2 - 4*c.val = a^2) ∧ (p^2 + 4*c.val = b^2) :=
sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_all_quadratics_l302_30208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l302_30279

theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                            -- first term is 3
    a 1 = 7 →                            -- second term is 7
    a 19 = 79 :=                         -- 20th term (index 19) is 79
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l302_30279


namespace NUMINAMATH_CALUDE_milk_replacement_theorem_l302_30278

/-- The fraction of original substance remaining after one replacement operation -/
def replacement_fraction : ℝ := 0.8

/-- The number of replacement operations -/
def num_operations : ℕ := 3

/-- The percentage of original substance remaining after multiple replacement operations -/
def remaining_percentage (f : ℝ) (n : ℕ) : ℝ := 100 * f^n

theorem milk_replacement_theorem :
  remaining_percentage replacement_fraction num_operations = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_theorem_l302_30278


namespace NUMINAMATH_CALUDE_jars_left_unpacked_eighty_jars_left_l302_30267

/-- The number of jars left unpacked given the packing configuration and total jars --/
theorem jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
  (jars_per_box2 : ℕ) (num_boxes2 : ℕ) (total_jars : ℕ) : ℕ :=
  by
  have h1 : jars_per_box1 = 12 := by sorry
  have h2 : num_boxes1 = 10 := by sorry
  have h3 : jars_per_box2 = 10 := by sorry
  have h4 : num_boxes2 = 30 := by sorry
  have h5 : total_jars = 500 := by sorry
  
  let packed_jars := jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2
  
  have packed_eq : packed_jars = 420 := by sorry
  
  exact total_jars - packed_jars

/-- The main theorem stating that 80 jars will be left unpacked --/
theorem eighty_jars_left : jars_left_unpacked 12 10 10 30 500 = 80 := by sorry

end NUMINAMATH_CALUDE_jars_left_unpacked_eighty_jars_left_l302_30267


namespace NUMINAMATH_CALUDE_triple_base_double_exponent_l302_30230

theorem triple_base_double_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end NUMINAMATH_CALUDE_triple_base_double_exponent_l302_30230


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l302_30236

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
  v₁ > v₂ →
  v₁ + v₂ = 20 →
  v₁ - v₂ = 5 →
  v₁ / v₂ = 5 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l302_30236
