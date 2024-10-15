import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l3912_391252

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3912_391252


namespace NUMINAMATH_CALUDE_height_percentage_difference_l3912_391254

theorem height_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_height_percentage_difference_l3912_391254


namespace NUMINAMATH_CALUDE_football_game_spectators_l3912_391286

/-- Represents the number of spectators at a football game --/
structure Spectators :=
  (adults : ℕ)
  (children : ℕ)
  (vips : ℕ)

/-- Conditions of the football game spectator problem --/
def football_game_conditions (s : Spectators) : Prop :=
  s.vips = 20 ∧
  s.children = s.adults / 2 ∧
  2 * s.adults + 2 * s.children + 2 * s.vips = 310

/-- Theorem stating the correct number of spectators --/
theorem football_game_spectators :
  ∃ (s : Spectators), football_game_conditions s ∧
    s.adults = 90 ∧ s.children = 45 ∧ s.vips = 20 ∧
    s.adults + s.children + s.vips = 155 :=
sorry


end NUMINAMATH_CALUDE_football_game_spectators_l3912_391286


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3912_391283

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3912_391283


namespace NUMINAMATH_CALUDE_sqrt_cube_equivalence_l3912_391208

theorem sqrt_cube_equivalence (x : ℝ) (h : x ≤ 0) :
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_equivalence_l3912_391208


namespace NUMINAMATH_CALUDE_track_meet_adults_l3912_391243

theorem track_meet_adults (children : ℕ) (total_seats : ℕ) (empty_seats : ℕ) 
  (h1 : children = 52)
  (h2 : total_seats = 95)
  (h3 : empty_seats = 14) :
  total_seats - empty_seats - children = 29 := by
  sorry

end NUMINAMATH_CALUDE_track_meet_adults_l3912_391243


namespace NUMINAMATH_CALUDE_valid_team_combinations_l3912_391299

/-- The number of ways to select a team of size k from n people -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of guests -/
def total_guests : ℕ := 5

/-- The number of male guests -/
def male_guests : ℕ := 3

/-- The number of female guests -/
def female_guests : ℕ := 2

/-- The required team size -/
def team_size : ℕ := 3

/-- The number of valid team combinations -/
def valid_combinations : ℕ := 
  choose male_guests 1 * choose female_guests 2 + 
  choose male_guests 2 * choose female_guests 1

theorem valid_team_combinations : valid_combinations = 9 := by sorry

end NUMINAMATH_CALUDE_valid_team_combinations_l3912_391299


namespace NUMINAMATH_CALUDE_lanas_bouquets_l3912_391236

theorem lanas_bouquets (tulips roses extra : ℕ) : 
  tulips = 36 → roses = 37 → extra = 3 → 
  tulips + roses + extra = 76 := by
sorry

end NUMINAMATH_CALUDE_lanas_bouquets_l3912_391236


namespace NUMINAMATH_CALUDE_expected_difference_coffee_tea_l3912_391235

-- Define the die sides
def dieSides : Nat := 8

-- Define perfect squares and primes up to 8
def perfectSquares : List Nat := [1, 4]
def primes : List Nat := [2, 3, 5, 7]

-- Define probabilities
def probCoffee : ℚ := 1 / 4
def probTea : ℚ := 1 / 2

-- Define number of days in a non-leap year
def daysInYear : Nat := 365

-- State the theorem
theorem expected_difference_coffee_tea :
  (probCoffee * daysInYear : ℚ) - (probTea * daysInYear : ℚ) = -91.25 := by
  sorry

end NUMINAMATH_CALUDE_expected_difference_coffee_tea_l3912_391235


namespace NUMINAMATH_CALUDE_pencil_weight_l3912_391225

/-- Given that 5 pencils weigh 141.5 grams, prove that one pencil weighs 28.3 grams. -/
theorem pencil_weight (total_weight : ℝ) (num_pencils : ℕ) (h1 : total_weight = 141.5) (h2 : num_pencils = 5) :
  total_weight / num_pencils = 28.3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_weight_l3912_391225


namespace NUMINAMATH_CALUDE_xy_equal_three_l3912_391210

theorem xy_equal_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equal_three_l3912_391210


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3912_391215

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3912_391215


namespace NUMINAMATH_CALUDE_order_of_numbers_l3912_391265

def Ψ : ℤ := -1006

def Ω : ℤ := -1007

def Θ : ℤ := -1008

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3912_391265


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l3912_391242

/-- Given 20 carrots weighing 3.64 kg, if 4 carrots are removed and the average weight
    of the remaining 16 carrots is 180 grams, then the average weight of the 4 removed
    carrots is 190 grams. -/
theorem carrot_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 3.64 →
  remaining_avg = 180 →
  let removed := 4
  let remaining := 20 - removed
  let removed_weight := total_weight * 1000 - remaining * remaining_avg
  removed_weight / removed = 190 := by
  sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l3912_391242


namespace NUMINAMATH_CALUDE_domain_of_f_l3912_391253

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/3) + (x - 7) ^ (1/4)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ 7}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ ∃ y : ℝ, f x = y :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l3912_391253


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3912_391200

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (-6 + t * Complex.I) = 3 * Real.sqrt 10 → t = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3912_391200


namespace NUMINAMATH_CALUDE_complex_division_example_l3912_391226

theorem complex_division_example : (1 - 3*I) / (1 + I) = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_example_l3912_391226


namespace NUMINAMATH_CALUDE_binomial_factorial_l3912_391294

theorem binomial_factorial : Nat.factorial (Nat.choose 8 5) = Nat.factorial 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_l3912_391294


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3912_391213

theorem necessary_but_not_sufficient_condition :
  ∃ (x : ℝ), (x < 0 ∨ x > 2) ∧ (2*x^2 - 5*x - 3 < 0) ∧
  ∀ (y : ℝ), (2*y^2 - 5*y - 3 ≥ 0) → (y < 0 ∨ y > 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3912_391213


namespace NUMINAMATH_CALUDE_cubic_inequality_l3912_391281

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3912_391281


namespace NUMINAMATH_CALUDE_card_sum_theorem_l3912_391269

theorem card_sum_theorem (a b c d e f g h : ℕ) : 
  (a + b) * (c + d) * (e + f) * (g + h) = 330 → 
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l3912_391269


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_sevenths_l3912_391272

theorem opposite_of_negative_three_sevenths :
  let x : ℚ := -3/7
  let y : ℚ := 3/7
  (∀ a b : ℚ, (a + b = 0 ↔ b = -a)) →
  y = -x :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_sevenths_l3912_391272


namespace NUMINAMATH_CALUDE_cost_of_3000_pencils_l3912_391211

def pencil_cost (quantity : ℕ) : ℚ :=
  let base_price := 36 / 120
  let discount_threshold := 2000
  let discount_factor := 0.9
  if quantity > discount_threshold
  then (quantity : ℚ) * base_price * discount_factor
  else (quantity : ℚ) * base_price

theorem cost_of_3000_pencils :
  pencil_cost 3000 = 810 := by sorry

end NUMINAMATH_CALUDE_cost_of_3000_pencils_l3912_391211


namespace NUMINAMATH_CALUDE_circle_equation_radius_l3912_391216

theorem circle_equation_radius (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y + c = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  c = -16 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l3912_391216


namespace NUMINAMATH_CALUDE_hamburger_sales_solution_l3912_391271

/-- Represents the hamburger sales problem. -/
def HamburgerSales (total_goal : ℕ) (price : ℕ) (first_group : ℕ) (remaining : ℕ) : Prop :=
  let total_hamburgers := total_goal / price
  let accounted_for := first_group + remaining
  total_hamburgers - accounted_for = 2

/-- Theorem stating the solution to the hamburger sales problem. -/
theorem hamburger_sales_solution :
  HamburgerSales 50 5 4 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_sales_solution_l3912_391271


namespace NUMINAMATH_CALUDE_solve_equations_l3912_391278

theorem solve_equations :
  (∀ x : ℝ, 4 * x = 20 → x = 5) ∧
  (∀ x : ℝ, x - 18 = 40 → x = 58) ∧
  (∀ x : ℝ, x / 7 = 12 → x = 84) ∧
  (∀ n : ℝ, 8 * n / 2 = 15 → n = 15 / 4) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3912_391278


namespace NUMINAMATH_CALUDE_det_dilation_matrix_3d_l3912_391255

/-- A matrix representing a dilation centered at the origin with scale factor 4 -/
def dilation_matrix (n : ℕ) (k : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (λ _ => k)

theorem det_dilation_matrix_3d :
  let E := dilation_matrix 3 4
  Matrix.det E = 64 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_3d_l3912_391255


namespace NUMINAMATH_CALUDE_sum_of_powers_of_fifth_root_of_unity_l3912_391298

theorem sum_of_powers_of_fifth_root_of_unity (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_fifth_root_of_unity_l3912_391298


namespace NUMINAMATH_CALUDE_specific_pyramid_surface_area_l3912_391267

/-- A right rectangular pyramid with square bases -/
structure RightRectangularPyramid where
  upperBaseEdge : ℝ
  lowerBaseEdge : ℝ
  sideEdge : ℝ

/-- Calculate the surface area of a right rectangular pyramid -/
def surfaceArea (p : RightRectangularPyramid) : ℝ :=
  -- Surface area calculation
  sorry

/-- The theorem stating the surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : RightRectangularPyramid := {
    upperBaseEdge := 2,
    lowerBaseEdge := 4,
    sideEdge := 2
  }
  surfaceArea p = 10 * Real.sqrt 3 + 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_surface_area_l3912_391267


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_inequality_l3912_391240

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 6} = Set.Ioo (-8/3) (4/3) := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_inequality_l3912_391240


namespace NUMINAMATH_CALUDE_train_cars_count_l3912_391203

theorem train_cars_count (cars_per_15s : ℕ) (total_time : ℕ) : 
  cars_per_15s = 10 → total_time = 210 → (total_time * cars_per_15s) / 15 = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l3912_391203


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_1000_l3912_391279

theorem units_digit_of_7_to_1000 : (7^1000 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_1000_l3912_391279


namespace NUMINAMATH_CALUDE_system_solution_l3912_391228

theorem system_solution (a b c x y z : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : x * y = a) (h2 : y * z = b) (h3 : z * x = c) :
  (x = Real.sqrt (a * c / b) ∨ x = -Real.sqrt (a * c / b)) ∧
  (y = Real.sqrt (a * b / c) ∨ y = -Real.sqrt (a * b / c)) ∧
  (z = Real.sqrt (b * c / a) ∨ z = -Real.sqrt (b * c / a)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3912_391228


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3912_391249

theorem quadratic_roots_properties (a b : ℝ) : 
  a^2 + 5*a + 2 = 0 → 
  b^2 + 5*b + 2 = 0 → 
  a ≠ b →
  (1/a + 1/b = -5/2) ∧ ((a^2 + 7*a) * (b^2 + 7*b) = 32) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3912_391249


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l3912_391202

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 30 → Nat.lcm a b = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l3912_391202


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l3912_391214

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l3912_391214


namespace NUMINAMATH_CALUDE_mark_solutions_mark_coefficients_l3912_391261

/-- Lauren's equation solutions -/
def lauren_solutions : Set ℝ := {x | |x - 6| = 3}

/-- Mark's equation -/
def mark_equation (b c : ℝ) (x : ℝ) : Prop := x^2 + b*x + c = 0

/-- Mark's equation has Lauren's solutions plus x = -2 -/
theorem mark_solutions (b c : ℝ) : 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

/-- The values of b and c in Mark's equation -/
theorem mark_coefficients : 
  ∃ b c : ℝ, (b = -12 ∧ c = 27) ∧ 
  (∀ x ∈ lauren_solutions, mark_equation b c x) ∧ 
  mark_equation b c (-2) :=
sorry

end NUMINAMATH_CALUDE_mark_solutions_mark_coefficients_l3912_391261


namespace NUMINAMATH_CALUDE_complex_number_existence_l3912_391237

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = -Complex.im (z + 3)) ∧
  ((z = -1 - 2*Complex.I) ∨ (z = -2 - Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l3912_391237


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3912_391246

/-- The perimeter of a region formed by three identical touching circles -/
theorem shaded_region_perimeter (c : ℝ) (θ : ℝ) : 
  c > 0 → θ > 0 → θ < 2 * Real.pi →
  let r := c / (2 * Real.pi)
  let arc_length := θ / (2 * Real.pi) * c
  3 * arc_length = c →
  c = 48 → θ = 2 * Real.pi / 3 →
  3 * arc_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3912_391246


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3912_391233

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → 0 < x^2 ∧ x^2 < 1) ∧
  (∃ x, 0 < x^2 ∧ x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3912_391233


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3912_391292

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    a * x₁^2 + (a + 2) * x₁ + 9 * a = 0 ∧
    a * x₂^2 + (a + 2) * x₂ + 9 * a = 0 ∧
    x₁ < 2 ∧ 2 < x₂) →
  -4/15 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3912_391292


namespace NUMINAMATH_CALUDE_parabola_intersection_probability_l3912_391229

/-- Represents the outcome of rolling a fair six-sided die -/
inductive SixSidedDie : Type
  | one | two | three | four | five | six

/-- Represents the outcome of rolling a fair four-sided die (2 to 5) -/
inductive FourSidedDie : Type
  | two | three | four | five

/-- Represents a parabola of the form y = x^2 + ax + b -/
structure Parabola1 where
  a : SixSidedDie
  b : SixSidedDie

/-- Represents a parabola of the form y = x^2 + px^2 + cx + d -/
structure Parabola2 where
  p : FourSidedDie
  c : SixSidedDie
  d : SixSidedDie

/-- Returns true if two parabolas intersect -/
def intersect (p1 : Parabola1) (p2 : Parabola2) : Bool :=
  sorry

/-- Probability that two randomly chosen parabolas intersect -/
def intersection_probability : ℚ :=
  sorry

theorem parabola_intersection_probability :
  intersection_probability = 209 / 216 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_probability_l3912_391229


namespace NUMINAMATH_CALUDE_product_of_sums_l3912_391223

theorem product_of_sums (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b + a + b = 3) (hbc : b * c + b + c = 3) (hac : a * c + a + c = 3) :
  (a + 1) * (b + 1) * (c + 1) = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l3912_391223


namespace NUMINAMATH_CALUDE_neither_question_correct_percentage_l3912_391231

theorem neither_question_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.50)
  (h3 : p_both = 0.33)
  : 1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_neither_question_correct_percentage_l3912_391231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l3912_391256

/-- Given an arithmetic sequence with 5 terms, first term 8, and common difference 8,
    prove that the average (mean) of the sequence is 24. -/
theorem arithmetic_sequence_average (a : Fin 5 → ℕ) 
  (h1 : a 0 = 8)
  (h2 : ∀ i : Fin 4, a (i + 1) = a i + 8) :
  (Finset.sum Finset.univ a) / 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l3912_391256


namespace NUMINAMATH_CALUDE_james_work_hours_james_work_hours_proof_l3912_391217

/-- Calculates the number of hours James needs to work to pay for food waste and janitorial costs -/
theorem james_work_hours (james_wage : ℝ) (meat_cost meat_wasted : ℝ) 
  (fruit_veg_cost fruit_veg_wasted : ℝ) (bread_cost bread_wasted : ℝ)
  (janitor_wage janitor_hours : ℝ) : ℝ :=
  let total_cost := meat_cost * meat_wasted + fruit_veg_cost * fruit_veg_wasted + 
                    bread_cost * bread_wasted + janitor_wage * 1.5 * janitor_hours
  total_cost / james_wage

/-- Proves that James needs to work 50 hours given the specific conditions -/
theorem james_work_hours_proof : 
  james_work_hours 8 5 20 4 15 1.5 60 10 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_work_hours_james_work_hours_proof_l3912_391217


namespace NUMINAMATH_CALUDE_incorrect_derivation_l3912_391244

theorem incorrect_derivation : ¬ (∀ (a b c : ℝ), c > 0 → c / a > c / b → a < b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_derivation_l3912_391244


namespace NUMINAMATH_CALUDE_imaginary_unit_expression_l3912_391277

theorem imaginary_unit_expression : Complex.I^7 - 2 / Complex.I = Complex.I := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_expression_l3912_391277


namespace NUMINAMATH_CALUDE_rabbit_count_prove_rabbit_count_l3912_391280

theorem rabbit_count : ℕ → ℕ → Prop :=
  fun total_white total_gray =>
    (∃ (caged_white : ℕ), caged_white = 6 ∧ total_white = caged_white + 9) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray) ∧
    (∃ (caged_white : ℕ), caged_white = 9 ∧ total_white = caged_white) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray + 16) →
    total_white + total_gray = 159

theorem prove_rabbit_count : ∃ (total_white total_gray : ℕ), rabbit_count total_white total_gray :=
  sorry

end NUMINAMATH_CALUDE_rabbit_count_prove_rabbit_count_l3912_391280


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3912_391282

/-- The sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (m : ℕ) : ℕ := 5 * m - 20

theorem largest_of_five_consecutive_even : 
  ∃ m : ℕ, sum_first_n_even 30 = sum_five_consecutive_even m ∧ m = 190 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3912_391282


namespace NUMINAMATH_CALUDE_max_product_constrained_l3912_391224

theorem max_product_constrained (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x/3 + y/4 = 1) : 
  x * y ≤ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀/3 + y₀/4 = 1 ∧ x₀ * y₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l3912_391224


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3912_391245

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_of_M_and_N :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3912_391245


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l3912_391260

theorem fraction_to_zero_power (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l3912_391260


namespace NUMINAMATH_CALUDE_undeclared_major_fraction_l3912_391221

/-- The fraction of students who have not declared a major among second- and third-year students -/
theorem undeclared_major_fraction :
  let total_students : ℚ := 1
  let first_year_students : ℚ := 1/3
  let second_year_students : ℚ := 1/3
  let third_year_students : ℚ := 1/3
  let first_year_undeclared : ℚ := 4/5 * first_year_students
  let second_year_declared : ℚ := 1/2 * (first_year_students - first_year_undeclared)
  let second_year_undeclared : ℚ := second_year_students - second_year_declared
  let third_year_undeclared : ℚ := 1/4 * third_year_students
  (second_year_undeclared + third_year_undeclared) / total_students = 23/60 := by
  sorry

end NUMINAMATH_CALUDE_undeclared_major_fraction_l3912_391221


namespace NUMINAMATH_CALUDE_video_difference_l3912_391247

/-- The number of videos watched by three friends -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Ekon -/
def ekon_videos : ℕ := kelsey_videos - 43

/-- The number of videos watched by Uma -/
def uma_videos : ℕ := total_videos - kelsey_videos - ekon_videos

/-- Ekon watched fewer videos than Uma -/
axiom ekon_less_than_uma : ekon_videos < uma_videos

theorem video_difference : uma_videos - ekon_videos = 17 := by
  sorry

end NUMINAMATH_CALUDE_video_difference_l3912_391247


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3912_391274

theorem rational_inequality_solution (x : ℝ) :
  (x ≠ -1 ∧ x ≠ 2) →
  ((x^2 + 3*x - 4) / (x^2 - x - 2) > 0 ↔ x > 2 ∨ x < -4) :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3912_391274


namespace NUMINAMATH_CALUDE_prism_diagonals_l3912_391291

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (p : RectangularPrism) : ℕ :=
  12 + 4  -- 12 face diagonals + 4 space diagonals

/-- Theorem: A rectangular prism with dimensions 4, 3, and 5 has 16 diagonals -/
theorem prism_diagonals :
  let p : RectangularPrism := ⟨4, 3, 5⟩
  num_diagonals p = 16 := by
  sorry

end NUMINAMATH_CALUDE_prism_diagonals_l3912_391291


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3912_391285

/-- If a point P with coordinates (m-3, 2+m) lies on the x-axis, then its coordinates are (-5, 0). -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P.2 = 0) →
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P = (-5, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3912_391285


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l3912_391209

theorem arithmetic_geometric_harmonic_means (a b c : ℝ) :
  (a + b + c) / 3 = 9 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l3912_391209


namespace NUMINAMATH_CALUDE_common_root_divisibility_l3912_391266

theorem common_root_divisibility (a b c : ℤ) (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) : 
  3 ∣ (a + b + 2*c) := by
sorry

end NUMINAMATH_CALUDE_common_root_divisibility_l3912_391266


namespace NUMINAMATH_CALUDE_triangle_area_l3912_391227

/-- Given a triangle with perimeter 36 and inradius 2.5, prove its area is 45 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 36) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : 
  area = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3912_391227


namespace NUMINAMATH_CALUDE_amoeba_growth_30_minutes_l3912_391273

/-- The number of amoebas after a given time interval, given an initial population and growth rate. -/
def amoeba_population (initial : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial * growth_factor ^ intervals

/-- Theorem stating that given the initial conditions, the final amoeba population after 30 minutes is 36450. -/
theorem amoeba_growth_30_minutes :
  let initial_population : ℕ := 50
  let growth_factor : ℕ := 3
  let interval_duration : ℕ := 5
  let total_duration : ℕ := 30
  let num_intervals : ℕ := total_duration / interval_duration
  amoeba_population initial_population growth_factor num_intervals = 36450 := by
  sorry

#eval amoeba_population 50 3 6

end NUMINAMATH_CALUDE_amoeba_growth_30_minutes_l3912_391273


namespace NUMINAMATH_CALUDE_f_even_k_value_g_f_common_point_a_range_l3912_391234

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The logarithm base 4 -/
noncomputable def log4 (x : ℝ) : ℝ := (Real.log x) / (Real.log 4)

/-- The function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log4 (4^x + 1) + k * x

/-- The function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log4 (a * 2^x - 4/3 * a)

/-- The number of common points between f and g -/
def CommonPoints (f g : ℝ → ℝ) : Prop := ∃! x, f x = g x

theorem f_even_k_value :
  IsEven (f k) → k = -1/2 :=
sorry

theorem g_f_common_point_a_range :
  CommonPoints (f (-1/2)) (g a) → (a > 1 ∨ a = -3) :=
sorry

end NUMINAMATH_CALUDE_f_even_k_value_g_f_common_point_a_range_l3912_391234


namespace NUMINAMATH_CALUDE_b_speed_is_13_l3912_391270

-- Define the walking scenario
def walking_scenario (speed_A speed_B initial_distance meeting_time : ℝ) : Prop :=
  speed_A > 0 ∧ speed_B > 0 ∧ initial_distance > 0 ∧ meeting_time > 0 ∧
  speed_A * meeting_time + speed_B * meeting_time = initial_distance

-- Theorem statement
theorem b_speed_is_13 :
  ∀ (speed_B : ℝ),
    walking_scenario 12 speed_B 25 1 →
    speed_B = 13 := by
  sorry

end NUMINAMATH_CALUDE_b_speed_is_13_l3912_391270


namespace NUMINAMATH_CALUDE_tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l3912_391297

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def isScalene (t : Triangle) : Prop := sorry

-- Define the tasty pair property
def isTastyPair (t : Triangle) (P Q : Point) : Prop := sorry

-- Define isogonal conjugates
def isIsogonalConjugate (t : Triangle) (P Q : Point) : Prop := sorry

-- Define the reflection of a triangle about its circumcenter
def reflectTriangle (t : Triangle) : Triangle := sorry

-- Main theorem
theorem tasty_pair_iff_isogonal_conjugate (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∀ P Q : Point, isTastyPair t P Q ↔ isIsogonalConjugate t P Q :=
sorry

-- Additional theorem
theorem exists_tasty_pair_for_both_triangles (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∃ P Q : Point, isTastyPair t P Q ∧ isTastyPair (reflectTriangle t) P Q :=
sorry

end NUMINAMATH_CALUDE_tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l3912_391297


namespace NUMINAMATH_CALUDE_math_competition_problem_l3912_391257

theorem math_competition_problem (a b : ℝ) 
  (ha : 4 / a^4 - 2 / a^2 - 3 = 0) 
  (hb : b^4 + b^2 - 3 = 0) : 
  (a^4 * b^4 + 4) / a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l3912_391257


namespace NUMINAMATH_CALUDE_four_digit_sum_3333_l3912_391238

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Converts a FourDigitNumber to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  1000 * n.1 + 100 * n.2.1 + 10 * n.2.2.1 + n.2.2.2

/-- Rearranges a FourDigitNumber by moving the last digit to the front -/
def rearrange (n : FourDigitNumber) : FourDigitNumber :=
  (n.2.2.2, n.1, n.2.1, n.2.2.1)

/-- Checks if a FourDigitNumber contains zero -/
def containsZero (n : FourDigitNumber) : Bool :=
  n.1 = 0 || n.2.1 = 0 || n.2.2.1 = 0 || n.2.2.2 = 0

theorem four_digit_sum_3333 (n : FourDigitNumber) :
  ¬containsZero n →
  toNumber n + toNumber (rearrange n) = 3333 →
  n = (1, 2, 1, 2) ∨ n = (2, 1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_3333_l3912_391238


namespace NUMINAMATH_CALUDE_simplify_fraction_l3912_391232

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) : 
  (a^2 / (a - 1)) - ((1 - 2*a) / (1 - a)) = a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3912_391232


namespace NUMINAMATH_CALUDE_pen_collection_problem_l3912_391230

/-- Represents the pen collection problem --/
theorem pen_collection_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 25)
  (h2 : final_pens = 75)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_problem_l3912_391230


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3912_391248

/-- Proves that the initial number of bananas per child is 2 -/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) 
  (extra_bananas : ℕ) (h1 : total_children = 610) (h2 : absent_children = 305) 
  (h3 : extra_bananas = 2) : 
  (total_children : ℚ) * (total_children - absent_children) = 
  (total_children - absent_children) * ((total_children - absent_children) + extra_bananas) :=
by sorry

#check initial_bananas_per_child

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3912_391248


namespace NUMINAMATH_CALUDE_ratio_problem_l3912_391276

theorem ratio_problem (a b c x y : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4)
  (h4 : x = 1) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3912_391276


namespace NUMINAMATH_CALUDE_baker_problem_l3912_391218

/-- The number of cakes that can be made given the available ingredients and recipe requirements. -/
def num_cakes : ℕ := 49

/-- The number of loaves of bread that can be made given the available ingredients and recipe requirements. -/
def num_bread : ℕ := 30

/-- The amount of flour available (in cups). -/
def flour_available : ℕ := 188

/-- The amount of sugar available (in cups). -/
def sugar_available : ℕ := 113

/-- The amount of flour required for one loaf of bread (in cups). -/
def flour_per_bread : ℕ := 3

/-- The amount of sugar required for one loaf of bread (in cups). -/
def sugar_per_bread : ℚ := 1/2

/-- The amount of flour required for one cake (in cups). -/
def flour_per_cake : ℕ := 2

/-- The amount of sugar required for one cake (in cups). -/
def sugar_per_cake : ℕ := 2

theorem baker_problem :
  (num_bread * flour_per_bread + num_cakes * flour_per_cake = flour_available) ∧
  (num_bread * sugar_per_bread + num_cakes * sugar_per_cake = sugar_available) :=
by sorry

end NUMINAMATH_CALUDE_baker_problem_l3912_391218


namespace NUMINAMATH_CALUDE_select_two_from_nine_l3912_391289

theorem select_two_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 2 → Nat.choose n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_nine_l3912_391289


namespace NUMINAMATH_CALUDE_equation_solution_l3912_391250

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 6 / (y / 3) → y = 18 ∨ y = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3912_391250


namespace NUMINAMATH_CALUDE_sandals_sold_l3912_391241

theorem sandals_sold (sneakers boots total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : boots = 11)
  (h3 : total = 17)
  (h4 : ∃ sandals : ℕ, total = sneakers + sandals + boots) :
  ∃ sandals : ℕ, sandals = 4 ∧ total = sneakers + sandals + boots :=
by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_l3912_391241


namespace NUMINAMATH_CALUDE_bus_problem_l3912_391219

theorem bus_problem (initial : ℕ) (first_on : ℕ) (second_off : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) :
  initial = 18 →
  first_on = 5 →
  second_off = 4 →
  third_off = 3 →
  third_on = 5 →
  final = 25 →
  ∃ (second_on : ℕ), 
    final = initial + first_on + second_on - second_off - third_off + third_on ∧
    second_on = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l3912_391219


namespace NUMINAMATH_CALUDE_equalize_foma_ierema_l3912_391290

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equalize_foma_ierema (w : MerchantWealth) 
  (h : problem_conditions w) : 
  ∃ (x : ℕ), w.foma - x = w.ierema + x ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_equalize_foma_ierema_l3912_391290


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l3912_391288

theorem estimate_sqrt_expression :
  6 < (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) ∧
  (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l3912_391288


namespace NUMINAMATH_CALUDE_triangle_problem_l3912_391284

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the angle B and area of the triangle under specific conditions. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.sin A = Real.sqrt 3 / 3 →
  B = π / 3 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3912_391284


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3912_391262

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3912_391262


namespace NUMINAMATH_CALUDE_amy_yard_area_l3912_391295

theorem amy_yard_area :
  ∀ (short_posts long_posts : ℕ) 
    (post_distance : ℝ) 
    (total_posts : ℕ),
  short_posts > 1 →
  long_posts > 1 →
  post_distance > 0 →
  total_posts = 24 →
  long_posts = (3 * short_posts) / 2 →
  total_posts = 2 * short_posts + 2 * long_posts - 4 →
  post_distance = 3 →
  (short_posts - 1 : ℝ) * post_distance * ((long_posts - 1 : ℝ) * post_distance) = 189 :=
by sorry

end NUMINAMATH_CALUDE_amy_yard_area_l3912_391295


namespace NUMINAMATH_CALUDE_clerical_to_total_ratio_l3912_391204

def total_employees : ℕ := 3600

def clerical_ratio (c : ℕ) : Prop :=
  (c / 2 : ℚ) = 0.2 * (total_employees - c / 2 : ℚ)

theorem clerical_to_total_ratio :
  ∃ c : ℕ, clerical_ratio c ∧ c * 3 = total_employees :=
sorry

end NUMINAMATH_CALUDE_clerical_to_total_ratio_l3912_391204


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_39_l3912_391287

/-- The equation of the graph -/
def graph_equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, graph_equation p.1 p.2 d

/-- If the graph of 3x^2 + y^2 + 6x - 12y + d = 0 consists of a single point, then d = 39 -/
theorem graph_single_point_implies_d_eq_39 : ∀ d : ℝ, single_point d → d = 39 := by
  sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_39_l3912_391287


namespace NUMINAMATH_CALUDE_expand_product_l3912_391205

theorem expand_product (x : ℝ) : (x + 3) * (x + 7) = x^2 + 10*x + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3912_391205


namespace NUMINAMATH_CALUDE_sum_g_11_and_neg_11_l3912_391239

/-- Given a function g(x) = px^8 + qx^6 - rx^4 + sx^2 + 5, 
    if g(11) = 7, then g(11) + g(-11) = 14 -/
theorem sum_g_11_and_neg_11 (p q r s : ℝ) : 
  let g : ℝ → ℝ := λ x => p * x^8 + q * x^6 - r * x^4 + s * x^2 + 5
  g 11 = 7 → g 11 + g (-11) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_11_and_neg_11_l3912_391239


namespace NUMINAMATH_CALUDE_complement_of_union_is_singleton_one_l3912_391206

-- Define the universal set I
def I : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem complement_of_union_is_singleton_one :
  (M ∪ N)ᶜ = {1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_singleton_one_l3912_391206


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l3912_391275

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d  -- Four different integers
  → d = 90  -- Largest integer is 90
  → (a + b + c + d) / 4 = 68  -- Average is 68
  → a ≥ 5  -- Smallest integer is at least 5
:= by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l3912_391275


namespace NUMINAMATH_CALUDE_paper_distribution_l3912_391264

theorem paper_distribution (total_students : ℕ) (total_sheets : ℕ) (leftover_sheets : ℕ)
  (h1 : total_students = 24)
  (h2 : total_sheets = 50)
  (h3 : leftover_sheets = 2)
  (h4 : ∃ (girls : ℕ), girls * 3 = total_students) :
  ∃ (girls : ℕ), girls * 3 = total_students ∧ 
    (total_sheets - leftover_sheets) / girls = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l3912_391264


namespace NUMINAMATH_CALUDE_distance_P_to_xoy_is_3_l3912_391293

/-- The distance from a point to the xOy plane in 3D Cartesian coordinates --/
def distance_to_xoy_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.2.2|

/-- The point P with coordinates (1, -2, 3) --/
def P : ℝ × ℝ × ℝ := (1, -2, 3)

/-- Theorem: The distance from point P(1,-2,3) to the xOy plane is 3 --/
theorem distance_P_to_xoy_is_3 : distance_to_xoy_plane P = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xoy_is_3_l3912_391293


namespace NUMINAMATH_CALUDE_only_B_and_C_valid_l3912_391296

-- Define the set of individuals
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define a type for the selection of individuals
def Selection := Person → Prop

-- Define the conditions
def condition1 (s : Selection) : Prop := s Person.A → s Person.B
def condition2 (s : Selection) : Prop := ¬(s Person.C) → ¬(s Person.B)
def condition3 (s : Selection) : Prop := s Person.C → ¬(s Person.D)

-- Define that exactly two individuals are selected
def exactlyTwo (s : Selection) : Prop :=
  (∃ (p1 p2 : Person), p1 ≠ p2 ∧ s p1 ∧ s p2 ∧ ∀ (p : Person), s p → (p = p1 ∨ p = p2))

-- State the theorem
theorem only_B_and_C_valid :
  ∀ (s : Selection),
    condition1 s →
    condition2 s →
    condition3 s →
    exactlyTwo s →
    s Person.B ∧ s Person.C ∧ ¬(s Person.A) ∧ ¬(s Person.D) :=
by
  sorry


end NUMINAMATH_CALUDE_only_B_and_C_valid_l3912_391296


namespace NUMINAMATH_CALUDE_katy_brownies_theorem_l3912_391259

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end NUMINAMATH_CALUDE_katy_brownies_theorem_l3912_391259


namespace NUMINAMATH_CALUDE_bike_distance_proof_l3912_391251

theorem bike_distance_proof (x t : ℝ) 
  (h1 : (x + 1) * (3 * t / 4) = x * t)
  (h2 : (x - 1) * (t + 3) = x * t) :
  x * t = 36 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l3912_391251


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3912_391268

theorem unique_prime_triple : 
  ∃! (x y z : ℕ), 
    (Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z) ∧ 
    (x > y ∧ y > z) ∧
    (Nat.Prime (x - y) ∧ Nat.Prime (y - z) ∧ Nat.Prime (x - z)) ∧
    (x = 7 ∧ y = 5 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3912_391268


namespace NUMINAMATH_CALUDE_rhombus_area_l3912_391207

/-- The area of a rhombus with side length 4 and a 45-degree angle between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → angle = π / 4 → 
  let area := side * side * Real.sin angle
  area = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3912_391207


namespace NUMINAMATH_CALUDE_cos_sum_equality_l3912_391201

theorem cos_sum_equality (x : Real) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.cos x + Real.cos (π / 3 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equality_l3912_391201


namespace NUMINAMATH_CALUDE_boat_men_count_l3912_391220

/-- The number of men in the boat -/
def n : ℕ := 8

/-- The weight of the man being replaced -/
def old_weight : ℕ := 60

/-- The weight of the new man -/
def new_weight : ℕ := 68

/-- The increase in average weight after replacement -/
def avg_increase : ℕ := 1

theorem boat_men_count :
  ∀ W : ℕ,
  (W + (new_weight - old_weight)) / n = W / n + avg_increase →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_boat_men_count_l3912_391220


namespace NUMINAMATH_CALUDE_max_d_value_l3912_391222

def is_valid_number (d f : ℕ) : Prop :=
  d < 10 ∧ f < 10 ∧ (636330 + 100000 * d + f) % 33 = 0

theorem max_d_value :
  (∃ d f : ℕ, is_valid_number d f) →
  (∀ d f : ℕ, is_valid_number d f → d ≤ 9) ∧
  (∃ f : ℕ, is_valid_number 9 f) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3912_391222


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_is_rectangle_l3912_391263

-- Define a circle
def Circle : Type := Unit

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of being inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem inscribed_quadrilateral_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  inscribed_in_circle q c → is_rectangle q := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_is_rectangle_l3912_391263


namespace NUMINAMATH_CALUDE_profit_reached_l3912_391258

/-- The number of disks in a buying pack -/
def buying_pack : ℕ := 5

/-- The cost of a buying pack in dollars -/
def buying_cost : ℚ := 8

/-- The number of disks in a selling pack -/
def selling_pack : ℕ := 4

/-- The price of a selling pack in dollars -/
def selling_price : ℚ := 10

/-- The target profit in dollars -/
def target_profit : ℚ := 120

/-- The minimum number of disks that must be sold to reach the target profit -/
def disks_to_sell : ℕ := 134

theorem profit_reached :
  let cost_per_disk : ℚ := buying_cost / buying_pack
  let price_per_disk : ℚ := selling_price / selling_pack
  let profit_per_disk : ℚ := price_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk ≥ target_profit → n ≥ disks_to_sell :=
by sorry

end NUMINAMATH_CALUDE_profit_reached_l3912_391258


namespace NUMINAMATH_CALUDE_alternating_odd_sum_equals_21_l3912_391212

/-- Calculates the sum of the alternating series of odd numbers from 1 to 41 -/
def alternating_odd_sum : ℤ :=
  let n := 20  -- Number of pairs (1-3), (5-7), etc.
  41 - 2 * n

/-- The sum of the series 1-3+5-7+9-11+13-...-39+41 equals 21 -/
theorem alternating_odd_sum_equals_21 : alternating_odd_sum = 21 := by
  sorry

#eval alternating_odd_sum  -- To check the result

end NUMINAMATH_CALUDE_alternating_odd_sum_equals_21_l3912_391212
