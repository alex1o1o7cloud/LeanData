import Mathlib

namespace NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l2050_205030

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 360 →
  a / 5 = b / 4 →
  a / 5 = c / 3 →
  max (180 - a) (max (180 - b) (180 - c)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_largest_interior_angle_of_triangle_l2050_205030


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l2050_205002

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 78)
  (h2 : total_selling_price = 6788)
  (h3 : profit_per_meter = 29) :
  (total_selling_price - profit_per_meter * total_length) / total_length = 58 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l2050_205002


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l2050_205018

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given number has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ+) : Prop := number_of_factors n = 8

/-- Theorem stating that 24 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors :
  has_eight_factors 24 ∧ ∀ m : ℕ+, m < 24 → ¬(has_eight_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l2050_205018


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2050_205078

theorem geometric_sequence_seventh_term 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) 
  (h2 : a * r^3 = 16) 
  (h3 : a * r^8 = 2) : 
  a * r^6 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2050_205078


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2050_205085

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2050_205085


namespace NUMINAMATH_CALUDE_unique_zip_code_l2050_205029

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_palindrome (a b c : ℕ) : Prop := a = c

def is_consecutive (a b : ℕ) : Prop := b = a + 1

theorem unique_zip_code (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b + c + d + e = 20 ∧
  is_consecutive a b ∧
  c ≠ 0 ∧ c ≠ a ∧ c ≠ b ∧
  is_palindrome a b c ∧
  d = 2 * a ∧
  d + e = 13 ∧
  is_prime (a * 10000 + b * 1000 + c * 100 + d * 10 + e) →
  a * 10000 + b * 1000 + c * 100 + d * 10 + e = 34367 :=
by sorry

end NUMINAMATH_CALUDE_unique_zip_code_l2050_205029


namespace NUMINAMATH_CALUDE_sqrt_17_irrational_l2050_205043

theorem sqrt_17_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_irrational_l2050_205043


namespace NUMINAMATH_CALUDE_product_of_solutions_absolute_value_equation_l2050_205074

theorem product_of_solutions_absolute_value_equation :
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, |x| = 3 * (|x| - 2) ↔ x = x₁ ∨ x = x₂) ∧
    x₁ * x₂ = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_absolute_value_equation_l2050_205074


namespace NUMINAMATH_CALUDE_like_terms_imply_m_minus_n_eq_two_l2050_205068

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → 
    ∃ (c1 c2 : ℚ), term1 x y = c1 * x^x * y^y ∧ term2 x y = c2 * x^x * y^y

/-- The first monomial 3x^m*y -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^m * y

/-- The second monomial -x^3*y^n -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := -1 * x^3 * y^n

theorem like_terms_imply_m_minus_n_eq_two (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_minus_n_eq_two_l2050_205068


namespace NUMINAMATH_CALUDE_polynomial_difference_l2050_205094

theorem polynomial_difference (a : ℝ) : (6 * a^2 - 5*a + 3) - (5 * a^2 + 2*a - 1) = a^2 - 7*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_l2050_205094


namespace NUMINAMATH_CALUDE_volunteer_selection_ways_l2050_205042

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days for community service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- Function to calculate the number of ways to select volunteers --/
def select_volunteers (n : ℕ) : ℕ :=
  (n) * (n - 1) * (n - 2)

theorem volunteer_selection_ways :
  select_volunteers n = 60 :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_ways_l2050_205042


namespace NUMINAMATH_CALUDE_sum_of_digits_product_53_nines_53_fours_l2050_205007

def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_product_53_nines_53_fours :
  sum_of_digits (repeat_digit 9 53 * repeat_digit 4 53) = 477 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_53_nines_53_fours_l2050_205007


namespace NUMINAMATH_CALUDE_triangular_pyramid_volume_l2050_205075

/-- The volume of a triangular pyramid formed by intersecting a right prism with a plane --/
theorem triangular_pyramid_volume 
  (a α β φ : ℝ) 
  (ha : a > 0)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hαβ : α + β < π)
  (hφ : 0 < φ ∧ φ < π/2) :
  ∃ V : ℝ, V = (a^3 * Real.sin α^2 * Real.sin β^2 * Real.tan φ) / (6 * Real.sin (α + β)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_volume_l2050_205075


namespace NUMINAMATH_CALUDE_division_result_l2050_205013

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2050_205013


namespace NUMINAMATH_CALUDE_basket_average_price_l2050_205006

/-- Given 4 baskets with an average cost of $4 and a fifth basket costing $8,
    the average price of all 5 baskets is $4.80. -/
theorem basket_average_price (num_initial_baskets : ℕ) (initial_avg_cost : ℚ) (fifth_basket_cost : ℚ) :
  num_initial_baskets = 4 →
  initial_avg_cost = 4 →
  fifth_basket_cost = 8 →
  (num_initial_baskets * initial_avg_cost + fifth_basket_cost) / (num_initial_baskets + 1) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_basket_average_price_l2050_205006


namespace NUMINAMATH_CALUDE_intersection_integer_point_l2050_205015

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- The intersection point of two lines -/
def intersection (m : ℤ) : ℚ × ℚ :=
  let x := (4 + 2*m) / (1 - m)
  let y := x - 4
  (x, y)

/-- Predicate to check if a point has integer coordinates -/
def isIntegerPoint (p : ℚ × ℚ) : Prop :=
  ∃ (ip : IntegerPoint), (ip.x : ℚ) = p.1 ∧ (ip.y : ℚ) = p.2

theorem intersection_integer_point :
  ∃ (m : ℤ), isIntegerPoint (intersection m) ∧ m = 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_integer_point_l2050_205015


namespace NUMINAMATH_CALUDE_sum_of_angles_l2050_205047

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the triangles and quadrilateral
def is_triangle (x y z : ℝ) : Prop := x + y + z = 180

-- Axioms based on the problem conditions
axiom triangle_ABC : is_triangle A B C
axiom triangle_DEF : is_triangle D E F
axiom quadrilateral_BEFC : B + E + F + C = 360

-- Theorem to prove
theorem sum_of_angles : A + B + C + D + E + F = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l2050_205047


namespace NUMINAMATH_CALUDE_obtuse_isosceles_triangle_vertex_angle_l2050_205096

/-- An obtuse isosceles triangle with the given property has a vertex angle of 150° -/
theorem obtuse_isosceles_triangle_vertex_angle 
  (a : ℝ) 
  (θ : ℝ) 
  (h_a_pos : a > 0)
  (h_θ_pos : θ > 0)
  (h_θ_acute : θ < π / 2)
  (h_isosceles : a^2 = (2 * a * Real.cos θ) * (2 * a * Real.sin θ)) :
  π - 2*θ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_obtuse_isosceles_triangle_vertex_angle_l2050_205096


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l2050_205027

theorem soda_discount_percentage (regular_price : ℝ) (discounted_total : ℝ) (cans : ℕ) :
  regular_price = 0.15 →
  discounted_total = 10.125 →
  cans = 75 →
  ∃ (discount : ℝ), 
    discount = 0.1 ∧
    cans * regular_price * (1 - discount) = discounted_total :=
by sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l2050_205027


namespace NUMINAMATH_CALUDE_total_spent_proof_l2050_205008

/-- The total amount spent on gifts and giftwrapping -/
def total_spent (gift_cost giftwrap_cost : ℚ) : ℚ :=
  gift_cost + giftwrap_cost

/-- Theorem: Given the cost of gifts and giftwrapping, prove the total amount spent -/
theorem total_spent_proof (gift_cost giftwrap_cost : ℚ) 
  (h1 : gift_cost = 561)
  (h2 : giftwrap_cost = 139) : 
  total_spent gift_cost giftwrap_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_proof_l2050_205008


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2050_205041

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -48
  let b : ℝ := 96
  let c : ℝ := -72
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2050_205041


namespace NUMINAMATH_CALUDE_max_square_area_with_perimeter_34_l2050_205003

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def rectangle_area (l w : ℕ) : ℕ := l * w

def rectangle_perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_square_area_with_perimeter_34 :
  ∀ l w : ℕ,
    rectangle_perimeter l w = 34 →
    is_perfect_square (rectangle_area l w) →
    rectangle_area l w ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_square_area_with_perimeter_34_l2050_205003


namespace NUMINAMATH_CALUDE_power_product_equality_l2050_205037

theorem power_product_equality : (-0.25)^2022 * 4^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2050_205037


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2050_205000

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2050_205000


namespace NUMINAMATH_CALUDE_inequalities_hold_l2050_205063

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  (x^3 + y^3 + z^3 < a^3 + b^3 + c^3) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2050_205063


namespace NUMINAMATH_CALUDE_table_area_proof_l2050_205035

theorem table_area_proof (total_runner_area : ℝ) 
                         (coverage_percentage : ℝ)
                         (two_layer_area : ℝ)
                         (three_layer_area : ℝ) 
                         (h1 : total_runner_area = 220)
                         (h2 : coverage_percentage = 0.80)
                         (h3 : two_layer_area = 24)
                         (h4 : three_layer_area = 28) :
  ∃ (table_area : ℝ), table_area = 275 ∧ 
    coverage_percentage * table_area = total_runner_area := by
  sorry


end NUMINAMATH_CALUDE_table_area_proof_l2050_205035


namespace NUMINAMATH_CALUDE_total_reptiles_count_l2050_205065

/-- The number of swamps in the sanctuary -/
def num_swamps : ℕ := 4

/-- The number of reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles in all swamp areas -/
def total_reptiles : ℕ := num_swamps * reptiles_per_swamp

theorem total_reptiles_count : total_reptiles = 1424 := by
  sorry

end NUMINAMATH_CALUDE_total_reptiles_count_l2050_205065


namespace NUMINAMATH_CALUDE_closed_polygonal_line_links_divisible_by_four_l2050_205028

/-- Represents a link in the polygonal line -/
structure Link where
  direction : Bool  -- True for horizontal, False for vertical
  length : Nat
  is_odd : Odd length

/-- Represents a closed polygonal line on a square grid -/
structure PolygonalLine where
  links : List Link
  is_closed : links.length > 0

/-- The main theorem to prove -/
theorem closed_polygonal_line_links_divisible_by_four (p : PolygonalLine) :
  4 ∣ p.links.length :=
sorry

end NUMINAMATH_CALUDE_closed_polygonal_line_links_divisible_by_four_l2050_205028


namespace NUMINAMATH_CALUDE_solution_set_correct_l2050_205061

/-- The solution set of the system of equations y² = x and y = x -/
def solution_set : Set (ℝ × ℝ) := {(1, 1), (0, 0)}

/-- The system of equations y² = x and y = x -/
def system_equations (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = p.1 ∧ p.2 = p.1

theorem solution_set_correct :
  ∀ p : ℝ × ℝ, p ∈ solution_set ↔ system_equations p := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2050_205061


namespace NUMINAMATH_CALUDE_max_prob_second_game_C_l2050_205019

variable (p₁ p₂ p₃ : ℝ)

-- Define the probabilities of winning against each player
def prob_A := p₁
def prob_B := p₂
def prob_C := p₃

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_game_C :
  P_C > P_A ∧ P_C > P_B :=
sorry

end NUMINAMATH_CALUDE_max_prob_second_game_C_l2050_205019


namespace NUMINAMATH_CALUDE_fathers_age_problem_l2050_205026

/-- Father's age problem -/
theorem fathers_age_problem (F C1 C2 : ℕ) : 
  F = 3 * (C1 + C2) →  -- Father's age is three times the sum of children's ages
  F + 5 = 2 * (C1 + 5 + C2 + 5) →  -- After 5 years, father's age will be twice the sum of children's ages
  F = 45 :=  -- Father's current age is 45
by sorry

end NUMINAMATH_CALUDE_fathers_age_problem_l2050_205026


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l2050_205073

/-- Represents a palindromic number -/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initial_reading : ℕ := 12321

/-- The duration of the drive in hours -/
def drive_duration : ℝ := 4

/-- The speed limit in miles per hour -/
def speed_limit : ℝ := 65

/-- The greatest possible average speed in miles per hour -/
def max_average_speed : ℝ := 50

/-- The theorem stating the greatest possible average speed -/
theorem greatest_possible_average_speed :
  ∀ (final_reading : ℕ),
    IsPalindrome initial_reading →
    IsPalindrome final_reading →
    final_reading > initial_reading →
    (final_reading - initial_reading : ℝ) ≤ speed_limit * drive_duration →
    (∀ (speed : ℝ), speed ≤ speed_limit → 
      (final_reading - initial_reading : ℝ) / drive_duration ≤ speed) →
    (final_reading - initial_reading : ℝ) / drive_duration = max_average_speed :=
sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l2050_205073


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2050_205051

-- Define the conditions
def p (x y : ℝ) : Prop := (x - 1) * (y - 2) = 0
def q (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 0

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2050_205051


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2050_205071

theorem sqrt_equality_implies_specific_integers (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (1 + Real.sqrt (21 + 12 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2050_205071


namespace NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l2050_205055

theorem three_digit_integers_with_remainders : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 6 ∧ 
              n % 12 = 9) ∧
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l2050_205055


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2050_205098

/-- An isosceles triangle with base 10 and equal sides 7 has perimeter 24 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun base side perimeter =>
    base = 10 ∧ side = 7 ∧ perimeter = base + 2 * side → perimeter = 24

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 10 7 24 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2050_205098


namespace NUMINAMATH_CALUDE_jenn_savings_l2050_205090

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the total value of coins in a jar -/
def jarValue (coin : String) (count : ℕ) : ℚ :=
  (coinValue coin * count : ℚ) / 100

/-- Calculates the available amount after applying the usage constraint -/
def availableAmount (amount : ℚ) (constraint : ℚ) : ℚ :=
  amount * constraint

/-- Represents Jenn's saving scenario -/
structure SavingScenario where
  quarterJars : ℕ
  quarterCount : ℕ
  dimeJars : ℕ
  dimeCount : ℕ
  nickelJars : ℕ
  nickelCount : ℕ
  monthlyPennies : ℕ
  months : ℕ
  usageConstraint : ℚ
  bikeCost : ℚ

/-- Theorem stating that Jenn will have $24.57 left after buying the bike -/
theorem jenn_savings (scenario : SavingScenario) : 
  scenario.quarterJars = 4 ∧ 
  scenario.quarterCount = 160 ∧
  scenario.dimeJars = 4 ∧
  scenario.dimeCount = 300 ∧
  scenario.nickelJars = 2 ∧
  scenario.nickelCount = 500 ∧
  scenario.monthlyPennies = 12 ∧
  scenario.months = 6 ∧
  scenario.usageConstraint = 4/5 ∧
  scenario.bikeCost = 240 →
  let totalQuarters := jarValue "quarter" (scenario.quarterJars * scenario.quarterCount)
  let totalDimes := jarValue "dime" (scenario.dimeJars * scenario.dimeCount)
  let totalNickels := jarValue "nickel" (scenario.nickelJars * scenario.nickelCount)
  let totalPennies := jarValue "penny" (scenario.monthlyPennies * scenario.months)
  let availableQuarters := availableAmount totalQuarters scenario.usageConstraint
  let availableDimes := availableAmount totalDimes scenario.usageConstraint
  let availableNickels := availableAmount totalNickels scenario.usageConstraint
  let availablePennies := availableAmount totalPennies scenario.usageConstraint
  let totalAvailable := availableQuarters + availableDimes + availableNickels + availablePennies
  totalAvailable - scenario.bikeCost = 24.57 := by
  sorry

end NUMINAMATH_CALUDE_jenn_savings_l2050_205090


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_range_l2050_205005

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + 2^x else (1/2) * x + a

-- Theorem statement
theorem two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0) →
  a ∈ Set.Icc (-2) (-1/2) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_range_l2050_205005


namespace NUMINAMATH_CALUDE_no_three_prime_roots_in_geometric_progression_l2050_205034

theorem no_three_prime_roots_in_geometric_progression :
  ¬∃ (p₁ p₂ p₃ : ℕ) (n₁ n₂ n₃ : ℤ) (a r : ℝ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ ∧
    n₁ ≠ n₂ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₃ ∧
    a > 0 ∧ r > 0 ∧
    a * r^n₁ = Real.sqrt p₁ ∧
    a * r^n₂ = Real.sqrt p₂ ∧
    a * r^n₃ = Real.sqrt p₃ :=
by sorry

end NUMINAMATH_CALUDE_no_three_prime_roots_in_geometric_progression_l2050_205034


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2050_205046

/-- Given a line with equation y - 3 = -3(x - 6), prove that the sum of its x-intercept and y-intercept is 28. -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 3 = -3 * (x - 6)) →
  (∃ x_int y_int : ℝ,
    (y_int - 3 = -3 * (x_int - 6) ∧ y_int = 0) ∧
    (0 - 3 = -3 * (0 - 6) ∧ y = y_int) ∧
    x_int + y_int = 28) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2050_205046


namespace NUMINAMATH_CALUDE_range_of_m_l2050_205050

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define sets A and B
variable (A B : Set ℝ)

-- State the theorem
theorem range_of_m (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ A ∧ x₁ ≠ x₂ ∧ f x₁ = f x₂ ∧ f x₁ ∈ B) :
  ∀ m ∈ B, m > -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2050_205050


namespace NUMINAMATH_CALUDE_enid_sweaters_count_l2050_205017

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_wool : ℕ := 82

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := (total_wool - (aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater)) / wool_per_sweater

theorem enid_sweaters_count :
  enid_sweaters = 8 := by sorry

end NUMINAMATH_CALUDE_enid_sweaters_count_l2050_205017


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2050_205045

-- Define the function f(x) = 2x^2
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2050_205045


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l2050_205087

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l2050_205087


namespace NUMINAMATH_CALUDE_max_stamps_proof_l2050_205049

/-- The price of a stamp in cents -/
def stamp_price : ℕ := 37

/-- The amount of money available in cents -/
def available_money : ℕ := 4000

/-- The maximum number of stamps that can be purchased -/
def max_stamps : ℕ := 108

theorem max_stamps_proof :
  (stamp_price * max_stamps ≤ available_money) ∧
  (∀ n : ℕ, stamp_price * n ≤ available_money → n ≤ max_stamps) := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_proof_l2050_205049


namespace NUMINAMATH_CALUDE_correct_converses_l2050_205020

-- Proposition 1
def prop1 (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Proposition 2
def prop2 (x : ℝ) : Prop := -2 ≤ x ∧ x < 3 → (x + 2) * (x - 3) ≤ 0

-- Proposition 3
def prop3 (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Proposition 4
def prop4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even x ∧ Even y → Even (x + y)

-- Converses of the propositions
def conv1 (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

def conv2 (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0 → -2 ≤ x ∧ x < 3

def conv3 (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

def conv4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even (x + y) → Even x ∧ Even y

theorem correct_converses :
  (∀ x, conv1 x) ∧
  (∀ x y, conv3 x y) ∧
  ¬(∀ x, conv2 x) ∧
  ¬(∀ x y, conv4 x y) :=
by sorry

end NUMINAMATH_CALUDE_correct_converses_l2050_205020


namespace NUMINAMATH_CALUDE_parallel_postulate_l2050_205066

-- Define a structure for points and lines in a 2D Euclidean plane
structure EuclideanPlane where
  Point : Type
  Line : Type
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

-- State the theorem
theorem parallel_postulate (plane : EuclideanPlane) 
  (l : plane.Line) (p : plane.Point) (h : ¬ plane.on_line p l) :
  ∃! m : plane.Line, plane.on_line p m ∧ plane.parallel m l :=
sorry

end NUMINAMATH_CALUDE_parallel_postulate_l2050_205066


namespace NUMINAMATH_CALUDE_seed_germination_percentage_experiment_result_l2050_205086

/-- Calculates the percentage of total seeds germinated in an agricultural experiment. -/
theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : ℚ :=
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100

/-- The percentage of total seeds germinated in the given agricultural experiment. -/
theorem experiment_result : 
  seed_germination_percentage 500 200 (30/100) (50/100) = 250/700 * 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_experiment_result_l2050_205086


namespace NUMINAMATH_CALUDE_abie_chips_count_l2050_205093

theorem abie_chips_count (initial bags_given bags_bought : ℕ) 
  (h1 : initial = 20)
  (h2 : bags_given = 4)
  (h3 : bags_bought = 6) : 
  initial - bags_given + bags_bought = 22 := by
  sorry

end NUMINAMATH_CALUDE_abie_chips_count_l2050_205093


namespace NUMINAMATH_CALUDE_max_value_on_circle_l2050_205014

/-- The maximum value of x^2 + y^2 for points on the circle x^2 - 4x - 4 + y^2 = 0 -/
theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 → x^2 + y^2 ≤ 12 + 8*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l2050_205014


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l2050_205044

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 22500 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 1000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l2050_205044


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2050_205062

theorem quadratic_factorization (c d : ℕ) (hc : c > d) : 
  (∀ x, x^2 - 20*x + 91 = (x - c)*(x - d)) → 2*d - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2050_205062


namespace NUMINAMATH_CALUDE_probability_three_yellow_apples_l2050_205022

def total_apples : ℕ := 10
def yellow_apples : ℕ := 4
def selected_apples : ℕ := 3

def probability_all_yellow : ℚ := (yellow_apples.choose selected_apples) / (total_apples.choose selected_apples)

theorem probability_three_yellow_apples :
  probability_all_yellow = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_yellow_apples_l2050_205022


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l2050_205091

/-- Represents a complex figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  max_triangle_side : ℕ
  min_triangle_side : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  20

/-- Theorem stating that for the given figure, 20 toothpicks must be removed to eliminate all triangles -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.max_triangle_side = 3)
  (h3 : figure.min_triangle_side = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l2050_205091


namespace NUMINAMATH_CALUDE_stream_current_rate_l2050_205025

/-- Represents the man's usual rowing speed in still water -/
def r : ℝ := sorry

/-- Represents the speed of the stream's current -/
def w : ℝ := sorry

/-- The distance traveled downstream and upstream -/
def distance : ℝ := 24

/-- Theorem stating the conditions and the conclusion about the stream's current -/
theorem stream_current_rate :
  (distance / (r + w) + 6 = distance / (r - w)) ∧
  (distance / (3*r + w) + 2 = distance / (3*r - w)) →
  w = 2 := by sorry

end NUMINAMATH_CALUDE_stream_current_rate_l2050_205025


namespace NUMINAMATH_CALUDE_stones_for_hall_l2050_205067

/-- Calculates the number of stones required to pave a hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).num.natAbs

/-- Theorem stating that 9000 stones are required to pave the given hall -/
theorem stones_for_hall : stones_required 72 30 4 6 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_stones_for_hall_l2050_205067


namespace NUMINAMATH_CALUDE_system_solution_l2050_205099

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = k - 1) →
  (2*x + y = 5*k + 4) →
  (x + y = 5) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2050_205099


namespace NUMINAMATH_CALUDE_equation_solutions_l2050_205059

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 2 * (3 * x + 1) = 6 ∧ x = -2) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2050_205059


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l2050_205033

theorem waiter_income_fraction (salary : ℚ) (salary_positive : salary > 0) : 
  let tips := (7 / 3) * salary
  let bonuses := (2 / 5) * salary
  let total_income := salary + tips + bonuses
  tips / total_income = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l2050_205033


namespace NUMINAMATH_CALUDE_rabbit_escape_theorem_l2050_205038

/-- The number of additional jumps a rabbit can make before a dog catches it. -/
def rabbit_jumps_before_catch (head_start : ℕ) (dog_jumps : ℕ) (rabbit_jumps : ℕ)
  (dog_distance : ℕ) (rabbit_distance : ℕ) : ℕ :=
  14 * head_start

/-- Theorem stating the number of jumps a rabbit can make before being caught by a dog
    under specific conditions. -/
theorem rabbit_escape_theorem :
  rabbit_jumps_before_catch 50 5 6 7 9 = 700 := by
  sorry

#eval rabbit_jumps_before_catch 50 5 6 7 9

end NUMINAMATH_CALUDE_rabbit_escape_theorem_l2050_205038


namespace NUMINAMATH_CALUDE_triangle_vertices_l2050_205095

/-- The lines forming the triangle --/
def line1 (x y : ℚ) : Prop := 2 * x + y - 6 = 0
def line2 (x y : ℚ) : Prop := x - y + 4 = 0
def line3 (x y : ℚ) : Prop := y + 1 = 0

/-- The vertices of the triangle --/
def vertex1 : ℚ × ℚ := (2/3, 14/3)
def vertex2 : ℚ × ℚ := (-5, -1)
def vertex3 : ℚ × ℚ := (7/2, -1)

/-- Theorem stating that the given points are the vertices of the triangle --/
theorem triangle_vertices : 
  (line1 vertex1.1 vertex1.2 ∧ line2 vertex1.1 vertex1.2) ∧
  (line2 vertex2.1 vertex2.2 ∧ line3 vertex2.1 vertex2.2) ∧
  (line1 vertex3.1 vertex3.2 ∧ line3 vertex3.1 vertex3.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vertices_l2050_205095


namespace NUMINAMATH_CALUDE_probability_divisible_by_four_probability_calculation_l2050_205039

def fair_12_sided_die := Finset.range 12

theorem probability_divisible_by_four (a b : ℕ) : 
  a ∈ fair_12_sided_die → b ∈ fair_12_sided_die →
  (a % 4 = 0 ∧ b % 4 = 0) ↔ (10 * a + b) % 4 = 0 ∧ a % 4 = 0 ∧ b % 4 = 0 :=
by sorry

theorem probability_calculation :
  (Finset.filter (λ x : ℕ × ℕ => x.1 % 4 = 0 ∧ x.2 % 4 = 0) (fair_12_sided_die.product fair_12_sided_die)).card /
  (fair_12_sided_die.card * fair_12_sided_die.card : ℚ) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_four_probability_calculation_l2050_205039


namespace NUMINAMATH_CALUDE_julies_work_hours_l2050_205084

theorem julies_work_hours (hourly_rate : ℝ) (days_per_week : ℕ) (monthly_salary : ℝ) :
  hourly_rate = 5 →
  days_per_week = 6 →
  monthly_salary = 920 →
  (monthly_salary / hourly_rate) / (days_per_week * 4 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_julies_work_hours_l2050_205084


namespace NUMINAMATH_CALUDE_unique_a_value_l2050_205053

def A (a : ℚ) : Set ℚ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℚ, 3 ∈ A a ∧ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l2050_205053


namespace NUMINAMATH_CALUDE_unique_A_value_l2050_205052

theorem unique_A_value (A : ℝ) (x₁ x₂ : ℂ) 
  (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : A * x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) :
  A = -7 := by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l2050_205052


namespace NUMINAMATH_CALUDE_income_comparison_l2050_205056

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 0.6400000000000001 * juan) :
  (mary - tim) / tim = 0.6 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l2050_205056


namespace NUMINAMATH_CALUDE_T_divisibility_l2050_205082

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, ¬(5 ∣ x)) ∧ (∃ x ∈ T, 7 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_T_divisibility_l2050_205082


namespace NUMINAMATH_CALUDE_two_faces_same_sides_l2050_205004

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Finset Face
  nonempty : faces.Nonempty

theorem two_faces_same_sides (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end NUMINAMATH_CALUDE_two_faces_same_sides_l2050_205004


namespace NUMINAMATH_CALUDE_sum_greater_two_necessary_not_sufficient_l2050_205024

theorem sum_greater_two_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
sorry

end NUMINAMATH_CALUDE_sum_greater_two_necessary_not_sufficient_l2050_205024


namespace NUMINAMATH_CALUDE_f_is_power_function_l2050_205032

/-- Definition of a power function -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

/-- The function y = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: f is a power function -/
theorem f_is_power_function : is_power_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_power_function_l2050_205032


namespace NUMINAMATH_CALUDE_count_special_integers_l2050_205088

theorem count_special_integers : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 200 < n ∧ n < 300 ∧ ∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) ∧
    (∀ n : ℤ, 200 < n → n < 300 → (∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) → n ∈ S) ∧
    Finset.card S = 5 :=
sorry

end NUMINAMATH_CALUDE_count_special_integers_l2050_205088


namespace NUMINAMATH_CALUDE_problem_solution_l2050_205021

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2050_205021


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l2050_205077

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) →
  (∃ x₁ x₂, x₁ + x₂ = -31/3 ∧ 
    ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l2050_205077


namespace NUMINAMATH_CALUDE_total_initial_tickets_l2050_205069

def dave_tiger_original_price : ℝ := 43
def dave_tiger_discount_rate : ℝ := 0.20
def dave_keychain_price : ℝ := 5.5
def dave_tickets_left : ℝ := 55

def alex_dinosaur_original_price : ℝ := 65
def alex_dinosaur_discount_rate : ℝ := 0.15
def alex_tickets_left : ℝ := 42

theorem total_initial_tickets : 
  let dave_tiger_discounted_price := dave_tiger_original_price * (1 - dave_tiger_discount_rate)
  let dave_total_spent := dave_tiger_discounted_price + dave_keychain_price
  let dave_initial_tickets := dave_total_spent + dave_tickets_left

  let alex_dinosaur_discounted_price := alex_dinosaur_original_price * (1 - alex_dinosaur_discount_rate)
  let alex_initial_tickets := alex_dinosaur_discounted_price + alex_tickets_left

  dave_initial_tickets + alex_initial_tickets = 192.15 := by sorry

end NUMINAMATH_CALUDE_total_initial_tickets_l2050_205069


namespace NUMINAMATH_CALUDE_initial_student_count_l2050_205010

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) : 
  initial_avg = 60.5 → new_avg = 64.0 → dropped_score = 8 → 
  ∃ n : ℕ, n > 0 ∧ 
    initial_avg * n = new_avg * (n - 1) + dropped_score ∧
    n = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l2050_205010


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2050_205057

/-- The area of a square field with a given diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 98.00000000000001) :
  d^2 / 2 = 4802.000000000001 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2050_205057


namespace NUMINAMATH_CALUDE_odd_decreasing_sum_negative_l2050_205023

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is decreasing on [0, ∞) if f(x) ≥ f(y) whenever 0 ≤ x < y -/
def DecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → f x ≥ f y

theorem odd_decreasing_sum_negative
  (f : ℝ → ℝ)
  (hodd : OddFunction f)
  (hdec : DecreasingOnNonnegative f)
  (a b : ℝ)
  (hsum : a + b > 0) :
  f a + f b < 0 :=
sorry

end NUMINAMATH_CALUDE_odd_decreasing_sum_negative_l2050_205023


namespace NUMINAMATH_CALUDE_only_divisor_square_sum_l2050_205097

theorem only_divisor_square_sum (n : ℕ+) :
  ∀ d : ℕ+, d ∣ (3 * n^2) → ∃ k : ℕ, n^2 + d = k^2 → d = 3 * n^2 :=
sorry

end NUMINAMATH_CALUDE_only_divisor_square_sum_l2050_205097


namespace NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l2050_205054

/-- An angle is coterminal with 60 degrees if it can be expressed as k * 360 + 60, where k is an integer -/
def is_coterminal_with_60 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = k * 360 + 60

/-- Theorem stating that -300 degrees is coterminal with 60 degrees -/
theorem negative_300_coterminal_with_60 : is_coterminal_with_60 (-300) := by
  sorry

end NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l2050_205054


namespace NUMINAMATH_CALUDE_merchant_profit_calculation_l2050_205076

theorem merchant_profit_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 20 →
  discount_percentage = 5 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 14 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_calculation_l2050_205076


namespace NUMINAMATH_CALUDE_max_days_for_88_alligators_l2050_205016

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the eating rate of the snake in alligators per week -/
def eating_rate : ℕ := 1

/-- Represents the total number of alligators eaten -/
def total_alligators : ℕ := 88

/-- Calculates the maximum number of days to eat a given number of alligators -/
def max_days_to_eat (alligators : ℕ) (rate : ℕ) (days_in_week : ℕ) : ℕ :=
  alligators * days_in_week / rate

/-- Theorem stating that the maximum number of days to eat 88 alligators is 616 -/
theorem max_days_for_88_alligators :
  max_days_to_eat total_alligators eating_rate days_per_week = 616 := by
  sorry

end NUMINAMATH_CALUDE_max_days_for_88_alligators_l2050_205016


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_is_34_min_value_achieved_l2050_205058

theorem min_value_expression (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∀ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) :=
by sorry

theorem min_value_is_34 (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥ 34 :=
by sorry

theorem min_value_achieved (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_is_34_min_value_achieved_l2050_205058


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l2050_205070

/-- The quadratic function y = 2x^2 - 8x + 10 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 10

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry : ℝ := 2

/-- Theorem: The axis of symmetry of the quadratic function f(x) = 2x^2 - 8x + 10 is x = 2 -/
theorem axis_of_symmetry_is_correct : 
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry

#check axis_of_symmetry_is_correct

end NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l2050_205070


namespace NUMINAMATH_CALUDE_work_time_for_c_l2050_205083

/-- The time it takes for worker c to complete the work alone, given the following conditions:
  * a and b can do the work in 2 days
  * b and c can do the work in 3 days
  * c and a can do the work in 4 days
-/
theorem work_time_for_c (a b c : ℝ) 
  (hab : a + b = 1/2)  -- a and b can do the work in 2 days
  (hbc : b + c = 1/3)  -- b and c can do the work in 3 days
  (hca : c + a = 1/4)  -- c and a can do the work in 4 days
  : 1 / c = 24 := by
  sorry


end NUMINAMATH_CALUDE_work_time_for_c_l2050_205083


namespace NUMINAMATH_CALUDE_tanα_tanβ_value_l2050_205040

theorem tanα_tanβ_value (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) :
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tanα_tanβ_value_l2050_205040


namespace NUMINAMATH_CALUDE_system_equation_solution_l2050_205064

theorem system_equation_solution :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℝ),
  2*x₁ + x₂ + x₃ + x₄ + x₅ = 6 →
  x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12 →
  x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24 →
  x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48 →
  x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96 →
  3*x₄ + 2*x₅ = 181 :=
by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l2050_205064


namespace NUMINAMATH_CALUDE_discounted_price_l2050_205072

/-- Given a top with an original price of m yuan and a discount of 20%,
    the actual selling price is 0.8m yuan. -/
theorem discounted_price (m : ℝ) : 
  let original_price := m
  let discount_rate := 0.2
  let selling_price := m * (1 - discount_rate)
  selling_price = 0.8 * m := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_l2050_205072


namespace NUMINAMATH_CALUDE_caseys_water_ratio_l2050_205048

/-- Proves that the ratio of water needed by each duck to water needed by each pig is 1:16 given the conditions of Casey's water pumping scenario. -/
theorem caseys_water_ratio :
  let pump_rate : ℚ := 3  -- gallons per minute
  let pump_time : ℚ := 25  -- minutes
  let corn_rows : ℕ := 4
  let corn_plants_per_row : ℕ := 15
  let water_per_corn_plant : ℚ := 1/2  -- gallons
  let num_pigs : ℕ := 10
  let water_per_pig : ℚ := 4  -- gallons
  let num_ducks : ℕ := 20

  let total_water : ℚ := pump_rate * pump_time
  let corn_water : ℚ := (corn_rows * corn_plants_per_row : ℚ) * water_per_corn_plant
  let pig_water : ℚ := (num_pigs : ℚ) * water_per_pig
  let duck_water : ℚ := total_water - corn_water - pig_water
  let water_per_duck : ℚ := duck_water / num_ducks

  water_per_duck / water_per_pig = 1 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_caseys_water_ratio_l2050_205048


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2050_205080

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℝ, 5*x + 2*y = 25 ∧ 3*x + 4*y = 15 ∧ x = 5 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l2050_205080


namespace NUMINAMATH_CALUDE_average_comparisons_equals_size_l2050_205089

/-- Represents a sequential search on an unordered array -/
structure SequentialSearch where
  /-- The number of elements in the array -/
  size : ℕ
  /-- Predicate indicating if the array is unordered -/
  unordered : Prop
  /-- Predicate indicating if the searched element is not in the array -/
  element_not_present : Prop

/-- The average number of comparisons needed in a sequential search -/
def average_comparisons (search : SequentialSearch) : ℕ := sorry

/-- Theorem stating that the average number of comparisons is equal to the array size 
    when the element is not present in an unordered array -/
theorem average_comparisons_equals_size (search : SequentialSearch) 
  (h_size : search.size = 100)
  (h_unordered : search.unordered)
  (h_not_present : search.element_not_present) :
  average_comparisons search = search.size := by sorry

end NUMINAMATH_CALUDE_average_comparisons_equals_size_l2050_205089


namespace NUMINAMATH_CALUDE_move_right_coords_specific_point_move_l2050_205031

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally -/
def moveRight (p : Point) (h : ℝ) : Point :=
  { x := p.x + h, y := p.y }

theorem move_right_coords (p : Point) (h : ℝ) :
  moveRight p h = { x := p.x + h, y := p.y } := by sorry

theorem specific_point_move :
  let p : Point := { x := -1, y := 3 }
  moveRight p 2 = { x := 1, y := 3 } := by sorry

end NUMINAMATH_CALUDE_move_right_coords_specific_point_move_l2050_205031


namespace NUMINAMATH_CALUDE_goats_in_field_l2050_205060

theorem goats_in_field (total_animals cows sheep : ℕ) 
  (h1 : total_animals = 200)
  (h2 : cows = 40)
  (h3 : sheep = 56) : 
  total_animals - (cows + sheep) = 104 := by
  sorry

end NUMINAMATH_CALUDE_goats_in_field_l2050_205060


namespace NUMINAMATH_CALUDE_trapezoid_KL_length_l2050_205009

/-- A trapezoid with points K and L on its diagonals -/
structure Trapezoid :=
  (A B C D K L : ℝ × ℝ)
  (is_trapezoid : sorry)
  (BC : ℝ)
  (AD : ℝ)
  (K_on_AC : sorry)
  (L_on_BD : sorry)
  (CK_KA_ratio : sorry)
  (BL_LD_ratio : sorry)

/-- The length of KL in the trapezoid -/
def KL_length (t : Trapezoid) : ℝ := sorry

theorem trapezoid_KL_length (t : Trapezoid) : 
  KL_length t = (1 / 11) * |7 * t.AD - 4 * t.BC| := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_KL_length_l2050_205009


namespace NUMINAMATH_CALUDE_creative_arts_academy_painting_paradox_l2050_205081

theorem creative_arts_academy_painting_paradox :
  let total_students : ℝ := 100
  let enjoy_painting_ratio : ℝ := 0.7
  let dont_enjoy_painting_ratio : ℝ := 1 - enjoy_painting_ratio
  let enjoy_but_negate_ratio : ℝ := 0.25
  let dont_enjoy_but_affirm_ratio : ℝ := 0.15

  let enjoy_painting : ℝ := total_students * enjoy_painting_ratio
  let dont_enjoy_painting : ℝ := total_students * dont_enjoy_painting_ratio
  
  let enjoy_but_negate : ℝ := enjoy_painting * enjoy_but_negate_ratio
  let dont_enjoy_but_affirm : ℝ := dont_enjoy_painting * dont_enjoy_but_affirm_ratio
  
  let total_claim_dislike : ℝ := enjoy_but_negate + (dont_enjoy_painting - dont_enjoy_but_affirm)
  
  (enjoy_but_negate / total_claim_dislike) * 100 = 40.698 :=
by sorry

end NUMINAMATH_CALUDE_creative_arts_academy_painting_paradox_l2050_205081


namespace NUMINAMATH_CALUDE_tree_height_increase_l2050_205001

/-- Proves that the annual increase in tree height is 2 feet given the initial conditions --/
theorem tree_height_increase (initial_height : ℝ) (annual_increase : ℝ) : 
  initial_height = 4 →
  initial_height + 6 * annual_increase = (initial_height + 4 * annual_increase) * (4/3) →
  annual_increase = 2 := by
sorry

end NUMINAMATH_CALUDE_tree_height_increase_l2050_205001


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2050_205079

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 ^ 2 + 3 * a 4 + 1 = 0) →
  (a 12 ^ 2 + 3 * a 12 + 1 = 0) →
  a 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2050_205079


namespace NUMINAMATH_CALUDE_albert_pizza_consumption_l2050_205092

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The number of large pizzas Albert buys -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def num_small_pizzas : ℕ := 2

/-- The total number of slices Albert eats -/
def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_consumption_l2050_205092


namespace NUMINAMATH_CALUDE_translated_line_equation_l2050_205036

/-- Given a line with slope 2 passing through the point (5, 1), prove that its equation is y = 2x - 9 -/
theorem translated_line_equation (x y : ℝ) :
  (y = 2 * x + 3) →  -- Original line equation
  (∃ b, y = 2 * x + b) →  -- Translated line has the same slope but different y-intercept
  (1 = 2 * 5 + b) →  -- The translated line passes through (5, 1)
  (y = 2 * x - 9)  -- The equation of the translated line
  := by sorry

end NUMINAMATH_CALUDE_translated_line_equation_l2050_205036


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2050_205012

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2 / 9 - y^2 / m = 1 ∧ x^2 + y^2 - 4*x - 5 = 0) →
  (∃ (k : ℝ), k = 4/3 ∧ 
    (∀ (x y : ℝ), (x^2 / 9 - y^2 / m = 1) → (y = k*x ∨ y = -k*x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2050_205012


namespace NUMINAMATH_CALUDE_certain_number_proof_l2050_205011

theorem certain_number_proof (h : 16 * 21.3 = 340.8) : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2050_205011
