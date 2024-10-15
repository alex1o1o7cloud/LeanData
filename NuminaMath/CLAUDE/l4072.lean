import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4072_407276

/-- The equation of a line passing through (-2, 3) with an inclination angle of 45° is x - y + 5 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (∃ (m : ℝ), m = Real.tan (π / 4) ∧ 
    y - 3 = m * (x - (-2))) ↔ 
  x - y + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4072_407276


namespace NUMINAMATH_CALUDE_angle_cosine_equivalence_l4072_407256

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the theorem
theorem angle_cosine_equivalence (t : Triangle) :
  (t.A > t.B ↔ Real.cos t.A < Real.cos t.B) :=
by sorry

end NUMINAMATH_CALUDE_angle_cosine_equivalence_l4072_407256


namespace NUMINAMATH_CALUDE_fishing_line_length_l4072_407259

/-- The original length of a fishing line can be calculated from its current length. -/
theorem fishing_line_length (current_length : ℝ) (h : current_length = 8.9) :
  (current_length + 3.1) * 3.1 * 2.1 = 78.12 := by
  sorry

#check fishing_line_length

end NUMINAMATH_CALUDE_fishing_line_length_l4072_407259


namespace NUMINAMATH_CALUDE_max_known_cards_l4072_407263

/-- A strategy for selecting cards and receiving information -/
structure CardStrategy where
  selectCards : Fin 2013 → Finset (Fin 2013)
  receiveNumber : Finset (Fin 2013) → Fin 2013

/-- The set of cards for which we know the numbers after applying a strategy -/
def knownCards (s : CardStrategy) : Finset (Fin 2013) :=
  sorry

/-- The theorem stating that 1986 is the maximum number of cards we can guarantee to know -/
theorem max_known_cards :
  (∃ (s : CardStrategy), (knownCards s).card = 1986) ∧
  (∀ (s : CardStrategy), (knownCards s).card ≤ 1986) :=
sorry

end NUMINAMATH_CALUDE_max_known_cards_l4072_407263


namespace NUMINAMATH_CALUDE_certain_number_problem_l4072_407268

theorem certain_number_problem : ∃ x : ℚ, (24 : ℚ) = (4/5) * x + 4 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4072_407268


namespace NUMINAMATH_CALUDE_circle_center_height_l4072_407244

/-- Represents a circle inside a parabola y = 2x^2, tangent at two points -/
structure CircleInParabola where
  /-- x-coordinate of one tangency point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- Condition: The circle is tangent to the parabola -/
  tangent : (a^2 + (2*a^2 - b)^2 = r^2) ∧ ((-a)^2 + (2*(-a)^2 - b)^2 = r^2)
  /-- Condition: The circle's center is on the y-axis -/
  center_on_y_axis : True

/-- Theorem: The y-coordinate of the circle's center equals the y-coordinate of the tangency points -/
theorem circle_center_height (c : CircleInParabola) : c.b = 2 * c.a^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_height_l4072_407244


namespace NUMINAMATH_CALUDE_parabola_y_comparison_l4072_407298

/-- Given a parabola y = -x² + 4x + c, prove that the y-coordinate of the point (-1, y₁) 
    is less than the y-coordinate of the point (1, y₂) on this parabola. -/
theorem parabola_y_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -(-1)^2 + 4*(-1) + c) 
  (h₂ : y₂ = -(1)^2 + 4*(1) + c) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_comparison_l4072_407298


namespace NUMINAMATH_CALUDE_x_value_proof_l4072_407273

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l4072_407273


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l4072_407294

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p → Prime q → p ≤ q → (p * q) ∣ ((5^p - 2^p) * (7^q - 2^q)) →
  ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 3) ∨ (p = 5 ∧ q = 37) ∨ (p = 5 ∧ q = 83)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l4072_407294


namespace NUMINAMATH_CALUDE_new_variance_after_adding_datapoint_l4072_407299

/-- Given a sample with size 7, average 5, and variance 2, adding a new data point of 5 results in a new variance of 7/4 -/
theorem new_variance_after_adding_datapoint
  (sample_size : ℕ)
  (original_avg : ℝ)
  (original_var : ℝ)
  (new_datapoint : ℝ)
  (h1 : sample_size = 7)
  (h2 : original_avg = 5)
  (h3 : original_var = 2)
  (h4 : new_datapoint = 5) :
  let new_sample_size : ℕ := sample_size + 1
  let new_avg : ℝ := (sample_size * original_avg + new_datapoint) / new_sample_size
  let new_var : ℝ := (sample_size * original_var + sample_size * (new_avg - original_avg)^2) / new_sample_size
  new_var = 7/4 := by sorry

end NUMINAMATH_CALUDE_new_variance_after_adding_datapoint_l4072_407299


namespace NUMINAMATH_CALUDE_sequence_arrangement_count_l4072_407235

theorem sequence_arrangement_count : ℕ :=
  let n : ℕ := 40
  let k : ℕ := 31
  let m : ℕ := 20
  Nat.choose n (n - k) * Nat.factorial (k - 2) * Nat.factorial (n - k)

#check sequence_arrangement_count

end NUMINAMATH_CALUDE_sequence_arrangement_count_l4072_407235


namespace NUMINAMATH_CALUDE_purple_four_leaved_clovers_l4072_407225

theorem purple_four_leaved_clovers (total_clovers : ℕ) (four_leaf_percentage : ℚ) (purple_fraction : ℚ) : 
  total_clovers = 500 →
  four_leaf_percentage = 1/5 →
  purple_fraction = 1/4 →
  (total_clovers : ℚ) * four_leaf_percentage * purple_fraction = 25 := by
sorry

end NUMINAMATH_CALUDE_purple_four_leaved_clovers_l4072_407225


namespace NUMINAMATH_CALUDE_product_equals_3408_l4072_407223

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3408_l4072_407223


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l4072_407230

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 3^x

theorem f_composition_negative_two : f (f (-2)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l4072_407230


namespace NUMINAMATH_CALUDE_expand_and_simplify_l4072_407218

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l4072_407218


namespace NUMINAMATH_CALUDE_power_of_m_divisible_by_24_l4072_407238

theorem power_of_m_divisible_by_24 (m : ℕ+) 
  (h1 : ∃ k : ℕ, (24 : ℕ) ∣ m^k)
  (h2 : (8 : ℕ) = Nat.gcd m ((Finset.range m).sup (λ i => Nat.gcd m i))) : 
  (∀ k < 1, ¬((24 : ℕ) ∣ m^k)) ∧ ((24 : ℕ) ∣ m^1) :=
sorry

end NUMINAMATH_CALUDE_power_of_m_divisible_by_24_l4072_407238


namespace NUMINAMATH_CALUDE_money_fraction_after_two_years_l4072_407248

/-- The simple interest rate per annum as a decimal -/
def interest_rate : ℝ := 0.08333333333333337

/-- The time period in years -/
def time_period : ℝ := 2

/-- The fraction of the sum of money after the given time period -/
def money_fraction : ℝ := 1 + interest_rate * time_period

theorem money_fraction_after_two_years :
  money_fraction = 1.1666666666666667 := by sorry

end NUMINAMATH_CALUDE_money_fraction_after_two_years_l4072_407248


namespace NUMINAMATH_CALUDE_quadratic_root_implies_q_value_l4072_407253

theorem quadratic_root_implies_q_value (p q : ℝ) (h : Complex.I ^ 2 = -1) :
  (3 * (1 + 4 * Complex.I) ^ 2 + p * (1 + 4 * Complex.I) + q = 0) → q = 51 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_q_value_l4072_407253


namespace NUMINAMATH_CALUDE_f_neg_two_eq_four_l4072_407216

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem f_neg_two_eq_four 
  (f : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : 
  f (-2) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_four_l4072_407216


namespace NUMINAMATH_CALUDE_money_transfer_problem_l4072_407272

/-- Represents the money transfer problem between Marco and Mary -/
theorem money_transfer_problem (marco_initial : ℕ) (mary_initial : ℕ) (mary_spends : ℕ) :
  marco_initial = 24 →
  mary_initial = 15 →
  mary_spends = 5 →
  let marco_gives := marco_initial / 2
  let mary_final := mary_initial + marco_gives - mary_spends
  let marco_final := marco_initial - marco_gives
  mary_final - marco_final = 10 := by sorry

end NUMINAMATH_CALUDE_money_transfer_problem_l4072_407272


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4072_407286

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  X^4 + 3 * X^3 = (X^2 - 3 * X + 2) * q + r ∧ 
  r = 36 * X - 32 ∧ 
  r.degree < (X^2 - 3 * X + 2).degree := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4072_407286


namespace NUMINAMATH_CALUDE_travel_ways_l4072_407255

theorem travel_ways (ways_AB ways_BC : ℕ) (h1 : ways_AB = 3) (h2 : ways_BC = 2) : 
  ways_AB * ways_BC = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_l4072_407255


namespace NUMINAMATH_CALUDE_distance_to_y_axis_reflection_distance_specific_point_l4072_407285

/-- The distance between a point and its reflection over the y-axis --/
theorem distance_to_y_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - (-x))^2 + (y - y)^2) = 2 * |x| :=
sorry

/-- The distance between (2, -4) and its reflection over the y-axis is 4 --/
theorem distance_specific_point : 
  Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_reflection_distance_specific_point_l4072_407285


namespace NUMINAMATH_CALUDE_alcohol_solution_concentration_l4072_407237

theorem alcohol_solution_concentration 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_percentage = 0.4)
  (h3 : added_alcohol = 1.2) :
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_concentration_l4072_407237


namespace NUMINAMATH_CALUDE_parabola_point_range_l4072_407254

/-- Parabola type representing y² = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle intersects a line -/
def circle_intersects_line (c : Circle) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

/-- Main theorem -/
theorem parabola_point_range (p : Parabola) (m : PointOnParabola p) :
  let c : Circle := { center := p.focus, radius := Real.sqrt ((m.x - p.focus.1)^2 + (m.y - p.focus.2)^2) }
  circle_intersects_line c p.directrix → m.x > 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_point_range_l4072_407254


namespace NUMINAMATH_CALUDE_min_value_arithmetic_progression_l4072_407266

/-- Given real numbers x, y, z in [0, 4] where x^2, y^2, z^2 form an arithmetic progression 
    with common difference 2, the minimum value of |x-y|+|y-z| is 4 - 2√3 -/
theorem min_value_arithmetic_progression (x y z : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 4) 
  (h2 : 0 ≤ y ∧ y ≤ 4) 
  (h3 : 0 ≤ z ∧ z ≤ 4) 
  (h4 : y^2 - x^2 = z^2 - y^2) 
  (h5 : y^2 - x^2 = 2) : 
  ∃ (m : ℝ), m = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ 4 → 0 ≤ y' ∧ y' ≤ 4 → 0 ≤ z' ∧ z' ≤ 4 → 
  y'^2 - x'^2 = z'^2 - y'^2 → y'^2 - x'^2 = 2 → 
  m ≤ |x' - y'| + |y' - z'| :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_progression_l4072_407266


namespace NUMINAMATH_CALUDE_sandy_age_l4072_407243

/-- Given that Molly is 20 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 70 years old. -/
theorem sandy_age (sandy molly : ℕ) 
  (h1 : molly = sandy + 20) 
  (h2 : sandy * 9 = molly * 7) : 
  sandy = 70 := by sorry

end NUMINAMATH_CALUDE_sandy_age_l4072_407243


namespace NUMINAMATH_CALUDE_class_size_l4072_407211

/-- The number of boys in the class -/
def n : ℕ := sorry

/-- The initial (incorrect) average weight -/
def initial_avg : ℚ := 584/10

/-- The correct average weight -/
def correct_avg : ℚ := 587/10

/-- The difference between the correct and misread weight -/
def weight_diff : ℚ := 62 - 56

theorem class_size :
  (n : ℚ) * initial_avg + weight_diff = n * correct_avg ∧ n = 20 := by sorry

end NUMINAMATH_CALUDE_class_size_l4072_407211


namespace NUMINAMATH_CALUDE_smallest_third_term_l4072_407265

/-- An arithmetic sequence of five positive integers with sum 80 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_80 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 80

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem stating that the smallest possible third term is 16 -/
theorem smallest_third_term :
  ∀ seq : ArithmeticSequence, third_term seq ≥ 16 := by
  sorry

#check smallest_third_term

end NUMINAMATH_CALUDE_smallest_third_term_l4072_407265


namespace NUMINAMATH_CALUDE_flag_distribution_l4072_407297

theorem flag_distribution (F : ℕ) (blue_percent red_percent : ℚ) :
  F % 2 = 0 →
  blue_percent = 60 / 100 →
  red_percent = 65 / 100 →
  blue_percent + red_percent - 1 = 25 / 100 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l4072_407297


namespace NUMINAMATH_CALUDE_least_8bit_number_proof_l4072_407208

/-- The least positive base-10 number requiring 8 binary digits -/
def least_8bit_number : ℕ := 128

/-- Convert a natural number to its binary representation -/
def to_binary (n : ℕ) : List Bool := sorry

/-- Count the number of digits in a binary representation -/
def binary_digit_count (n : ℕ) : ℕ := (to_binary n).length

theorem least_8bit_number_proof :
  (∀ m : ℕ, m < least_8bit_number → binary_digit_count m < 8) ∧
  binary_digit_count least_8bit_number = 8 := by sorry

end NUMINAMATH_CALUDE_least_8bit_number_proof_l4072_407208


namespace NUMINAMATH_CALUDE_two_correct_statements_l4072_407212

theorem two_correct_statements (a b : ℝ) 
  (h : (a - Real.sqrt (a^2 - 1)) * (b - Real.sqrt (b^2 - 1)) = 1) :
  (a = b ∧ a * b = 1) ∧ 
  (a + b ≠ 0 ∧ a * b ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_statements_l4072_407212


namespace NUMINAMATH_CALUDE_diana_garden_area_l4072_407239

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_distance : ℝ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the garden --/
def garden_area (g : Garden) : ℝ :=
  (g.short_side_posts - 1) * g.post_distance * (g.long_side_posts - 1) * g.post_distance

/-- Theorem stating the area of Diana's garden --/
theorem diana_garden_area :
  ∀ g : Garden,
  g.total_posts = 24 ∧
  g.post_distance = 3 ∧
  g.long_side_posts = (3 * g.short_side_posts + 1) / 2 ∧
  2 * g.short_side_posts + 2 * g.long_side_posts - 4 = g.total_posts →
  garden_area g = 135 := by
  sorry

end NUMINAMATH_CALUDE_diana_garden_area_l4072_407239


namespace NUMINAMATH_CALUDE_trig_inequality_and_equality_condition_l4072_407292

theorem trig_inequality_and_equality_condition (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9) ∧
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) = 9 ↔ 
    α = Real.arctan (Real.sqrt 2) ∧ β = π/4) :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_and_equality_condition_l4072_407292


namespace NUMINAMATH_CALUDE_square_field_problem_l4072_407226

theorem square_field_problem (a p : ℝ) (x : ℝ) : 
  p = 36 →                           -- perimeter is 36 feet
  a = (p / 4) ^ 2 →                  -- area formula for square
  6 * a = 6 * (2 * p + x) →          -- given equation
  x = 9 := by sorry                  -- prove x = 9

end NUMINAMATH_CALUDE_square_field_problem_l4072_407226


namespace NUMINAMATH_CALUDE_exists_unstudied_planet_l4072_407222

/-- Represents a planet in the solar system -/
structure Planet where
  id : ℕ

/-- Represents the solar system with its properties -/
structure SolarSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  closest_planet : Planet → Planet
  odd_num_planets : Odd planets.card
  distinct_distances : ∀ p1 p2 p3 p4 : Planet, p1 ≠ p2 → p3 ≠ p4 → (p1, p2) ≠ (p3, p4) → distance p1 p2 ≠ distance p3 p4
  closest_is_closest : ∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 (closest_planet p1) ≤ distance p1 p2
  not_self_study : ∀ p : Planet, closest_planet p ≠ p

/-- There exists a planet not being studied by any astronomer -/
theorem exists_unstudied_planet (s : SolarSystem) : 
  ∃ p : Planet, p ∈ s.planets ∧ ∀ q : Planet, q ∈ s.planets → s.closest_planet q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unstudied_planet_l4072_407222


namespace NUMINAMATH_CALUDE_solve_equation_l4072_407249

theorem solve_equation (x : ℝ) : (3 - 5 + 7 = 6 - x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4072_407249


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l4072_407252

theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) : 
  (3*x - 5 : ℚ) = (7*x - 17) - ((7*x - 17) - (3*x - 5)) → 
  (7*x - 17 : ℚ) = (4*x + 3) - ((4*x + 3) - (7*x - 17)) → 
  (∃ a d : ℚ, a = 3*x - 5 ∧ d = (7*x - 17) - (3*x - 5) ∧ 
    a + (n - 1) * d = 4033) → 
  n = 641 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l4072_407252


namespace NUMINAMATH_CALUDE_min_value_expression_l4072_407224

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 9 ∧
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4072_407224


namespace NUMINAMATH_CALUDE_point_on_line_l4072_407227

/-- For any point (m,n) on the line y = 2x + 1, 2m - n = -1 -/
theorem point_on_line (m n : ℝ) : n = 2 * m + 1 → 2 * m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l4072_407227


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l4072_407274

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l4072_407274


namespace NUMINAMATH_CALUDE_remainder_proof_l4072_407207

theorem remainder_proof : ∃ (q : ℕ), 4351 = 101 * q + 8 :=
by
  -- We define the greatest common divisor G as 101
  let G : ℕ := 101

  -- We define the condition that G divides 5161 with remainder 10
  have h1 : ∃ (q : ℕ), 5161 = G * q + 10 := by sorry

  -- We prove that 4351 divided by G has remainder 8
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l4072_407207


namespace NUMINAMATH_CALUDE_largest_two_decimal_rounding_to_five_l4072_407233

-- Define a two-decimal number that rounds to 5.0
def is_valid_number (x : ℚ) : Prop :=
  (x ≥ 4.95) ∧ (x < 5.05) ∧ (∃ n : ℤ, x = n / 100)

-- Define the largest possible value
def largest_value : ℚ := 5.04

-- Theorem statement
theorem largest_two_decimal_rounding_to_five :
  ∀ x : ℚ, is_valid_number x → x ≤ largest_value :=
by sorry

end NUMINAMATH_CALUDE_largest_two_decimal_rounding_to_five_l4072_407233


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4072_407240

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4072_407240


namespace NUMINAMATH_CALUDE_simplify_fraction_l4072_407287

theorem simplify_fraction : (36 : ℚ) / 54 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4072_407287


namespace NUMINAMATH_CALUDE_min_distance_to_tangent_point_l4072_407231

/-- The minimum distance from a point on the line y = x + 1 to a tangent point 
    on the circle (x - 3)² + y² = 1 is √7. -/
theorem min_distance_to_tangent_point : 
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (P.2 = P.1 + 1) ∧ 
    ((T.1 - 3)^2 + T.2^2 = 1) ∧
    (∀ (Q : ℝ × ℝ), (Q.2 = Q.1 + 1) → (Q.1 - 3)^2 + Q.2^2 = 1 → 
      dist P T ≤ dist Q T) ∧
    dist P T = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_tangent_point_l4072_407231


namespace NUMINAMATH_CALUDE_no_consecutive_squares_equal_consecutive_fourth_powers_l4072_407281

theorem no_consecutive_squares_equal_consecutive_fourth_powers :
  ¬ ∃ (m n : ℕ), m^2 + (m+1)^2 = n^4 + (n+1)^4 := by
sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_equal_consecutive_fourth_powers_l4072_407281


namespace NUMINAMATH_CALUDE_spinner_probability_l4072_407280

/-- Represents an equilateral triangle dissected by its altitudes -/
structure DissectedTriangle where
  regions : ℕ
  shaded_regions : ℕ

/-- The probability of a spinner landing in a shaded region -/
def landing_probability (t : DissectedTriangle) : ℚ :=
  t.shaded_regions / t.regions

/-- Theorem stating the probability of landing in a shaded region -/
theorem spinner_probability (t : DissectedTriangle) 
  (h1 : t.regions = 6)
  (h2 : t.shaded_regions = 3) : 
  landing_probability t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l4072_407280


namespace NUMINAMATH_CALUDE_circle_line_intersection_l4072_407278

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 15 = 0

/-- The line equation -/
def line_eq (x y m : ℝ) : Prop := (1 + 3*m)*x + (3 - 2*m)*y + 4*m - 17 = 0

/-- The theorem stating that the circle and line always intersect at two points -/
theorem circle_line_intersection :
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    (∀ m : ℝ, circle_eq p.1 p.2 ∧ line_eq p.1 p.2 m) ∧
    (∀ m : ℝ, circle_eq q.1 q.2 ∧ line_eq q.1 q.2 m) ∧
    (∀ r : ℝ × ℝ, (∀ m : ℝ, circle_eq r.1 r.2 ∧ line_eq r.1 r.2 m) → r = p ∨ r = q) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l4072_407278


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l4072_407290

theorem maximum_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (maximum_marks : ℝ) :
  percentage = 90 →
  scored_marks = 405 →
  percentage / 100 * maximum_marks = scored_marks →
  maximum_marks = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l4072_407290


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l4072_407202

theorem factorial_difference_quotient (n : ℕ) (h : n ≥ 8) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial n = n^2 + 3*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l4072_407202


namespace NUMINAMATH_CALUDE_mans_rate_l4072_407229

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l4072_407229


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l4072_407217

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l4072_407217


namespace NUMINAMATH_CALUDE_karen_graded_eight_tests_l4072_407236

/-- Represents the bonus calculation for a teacher based on test scores. -/
def bonus_calculation (n : ℕ) : Prop :=
  let base_bonus := 500
  let extra_bonus_per_point := 10
  let base_threshold := 75
  let max_score := 150
  let current_average := 70
  let last_two_tests_score := 290
  let target_bonus := 600
  let total_current_points := n * current_average
  let total_points_after := total_current_points + last_two_tests_score
  let final_average := total_points_after / (n + 2)
  (final_average > base_threshold) ∧
  (target_bonus = base_bonus + (final_average - base_threshold) * extra_bonus_per_point) ∧
  (∀ m : ℕ, m ≤ n + 2 → m * max_score ≥ total_points_after)

/-- Theorem stating that Karen has graded 8 tests. -/
theorem karen_graded_eight_tests : ∃ (n : ℕ), bonus_calculation n ∧ n = 8 :=
  sorry

end NUMINAMATH_CALUDE_karen_graded_eight_tests_l4072_407236


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l4072_407296

/-- Given three distinct collinear points A, B, and C with coordinates relative to point O,
    prove that the value of k is -1/4. -/
theorem collinear_points_k_value
  (k : ℝ)
  (A B C : ℝ × ℝ)
  (hA : A = (k, 2))
  (hB : B = (1, 2*k))
  (hC : C = (1-k, -1))
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hCollinear : ∃ (t : ℝ), C - A = t • (B - A)) :
  k = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l4072_407296


namespace NUMINAMATH_CALUDE_bisection_next_step_l4072_407270

/-- The bisection method's next step for a function with given properties -/
theorem bisection_next_step (f : ℝ → ℝ) (h1 : f 1 < 0) (h2 : f 1.5 > 0) :
  let x₀ : ℝ := (1 + 1.5) / 2
  x₀ = 1.25 := by sorry

end NUMINAMATH_CALUDE_bisection_next_step_l4072_407270


namespace NUMINAMATH_CALUDE_sum_c_d_equals_three_l4072_407206

theorem sum_c_d_equals_three (a b c d : ℝ)
  (h1 : a + b = 12)
  (h2 : b + c = 9)
  (h3 : a + d = 6) :
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_c_d_equals_three_l4072_407206


namespace NUMINAMATH_CALUDE_compute_expression_l4072_407269

theorem compute_expression : 4 * 4^3 - 16^60 / 16^57 = -3840 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l4072_407269


namespace NUMINAMATH_CALUDE_company_survey_l4072_407289

/-- The number of employees who do not use social networks -/
def non_users : ℕ := 40

/-- The fraction of social network users who use VKontakte -/
def vk_users : ℚ := 3/4

/-- The fraction of social network users who use both VKontakte and Odnoklassniki -/
def both_users : ℚ := 13/20

/-- The fraction of total employees who use Odnoklassniki -/
def ok_users : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_users : ℚ) * (vk_users + (1 - vk_users)) = N * ok_users :=
by sorry

end NUMINAMATH_CALUDE_company_survey_l4072_407289


namespace NUMINAMATH_CALUDE_jessica_driving_days_l4072_407275

/-- Calculates the number of days needed to meet a driving hour requirement -/
def daysToMeetRequirement (requiredHours : ℕ) (minutesPerTrip : ℕ) : ℕ :=
  let requiredMinutes := requiredHours * 60
  let minutesPerDay := minutesPerTrip * 2
  requiredMinutes / minutesPerDay

theorem jessica_driving_days :
  daysToMeetRequirement 50 20 = 75 := by
  sorry

#eval daysToMeetRequirement 50 20

end NUMINAMATH_CALUDE_jessica_driving_days_l4072_407275


namespace NUMINAMATH_CALUDE_dividend_proof_l4072_407214

theorem dividend_proof : (10918788 : ℕ) / 12 = 909899 := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l4072_407214


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l4072_407258

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.47

/-- The fraction representation of the repeating decimal -/
def fraction : ℚ := 47 / 99

/-- Theorem stating that the repeating decimal equals the fraction -/
theorem decimal_equals_fraction : repeating_decimal = fraction := by sorry

/-- Theorem stating that the fraction is in lowest terms -/
theorem fraction_is_lowest_terms : 
  ∀ (a b : ℕ), a / b = fraction → b ≠ 0 → a.gcd b = 1 := by sorry

/-- The main theorem to prove -/
theorem sum_of_numerator_and_denominator : 
  ∃ (n d : ℕ), n / d = fraction ∧ n.gcd d = 1 ∧ n + d = 146 := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l4072_407258


namespace NUMINAMATH_CALUDE_steves_oranges_l4072_407260

/-- Steve's orange sharing problem -/
theorem steves_oranges (initial_oranges shared_oranges : ℕ) :
  initial_oranges = 46 →
  shared_oranges = 4 →
  initial_oranges - shared_oranges = 42 := by
  sorry

end NUMINAMATH_CALUDE_steves_oranges_l4072_407260


namespace NUMINAMATH_CALUDE_trapezium_side_length_l4072_407221

theorem trapezium_side_length (a b h area : ℝ) : 
  b = 20 → h = 12 → area = 228 → area = (a + b) * h / 2 → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l4072_407221


namespace NUMINAMATH_CALUDE_vector_cosine_and_projection_l4072_407241

/-- Given vectors a and b with their components, prove the cosine of the angle between them
    and the scalar projection of a onto b. -/
theorem vector_cosine_and_projection (a b : ℝ × ℝ) (h : a = (3, 1) ∧ b = (-2, 4)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ = -Real.sqrt 2 / 10) ∧
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_cosine_and_projection_l4072_407241


namespace NUMINAMATH_CALUDE_race_runners_count_l4072_407271

theorem race_runners_count :
  ∀ (total_runners : ℕ) (ammar_position : ℕ) (julia_position : ℕ),
  ammar_position > 0 →
  julia_position > ammar_position →
  ammar_position - 1 = (total_runners - ammar_position) / 2 →
  julia_position = ammar_position + 10 →
  julia_position - 1 = 2 * (total_runners - julia_position) →
  total_runners = 31 :=
by
  sorry

#check race_runners_count

end NUMINAMATH_CALUDE_race_runners_count_l4072_407271


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4072_407264

/-- The equation (3x+7)(x-5) = -27 + kx has exactly one real solution if and only if k = -8 + 4√6 or k = -8 - 4√6 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+7)*(x-5) = -27 + k*x) ↔ 
  (k = -8 + 4*Real.sqrt 6 ∨ k = -8 - 4*Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4072_407264


namespace NUMINAMATH_CALUDE_probability_one_white_two_red_l4072_407262

def white_balls : ℕ := 4
def red_balls : ℕ := 5
def total_balls : ℕ := white_balls + red_balls
def drawn_balls : ℕ := 3

def favorable_outcomes : ℕ := (Nat.choose white_balls 1) * (Nat.choose red_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_one_white_two_red : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_two_red_l4072_407262


namespace NUMINAMATH_CALUDE_apples_picked_theorem_l4072_407250

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem apples_picked_theorem : total_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_theorem_l4072_407250


namespace NUMINAMATH_CALUDE_pot_stacking_l4072_407295

theorem pot_stacking (total_pots : ℕ) (vertical_stack : ℕ) (shelves : ℕ) 
  (h1 : total_pots = 60)
  (h2 : vertical_stack = 5)
  (h3 : shelves = 4) :
  (total_pots / vertical_stack) / shelves = 3 := by
  sorry

end NUMINAMATH_CALUDE_pot_stacking_l4072_407295


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4072_407228

theorem inequality_and_equality_condition (n : ℕ) (hn : n ≥ 1) :
  (1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) ≥ (n.factorial : ℝ)^((2 : ℝ) / n) ∧
  ((1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) = (n.factorial : ℝ)^((2 : ℝ) / n) ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4072_407228


namespace NUMINAMATH_CALUDE_sum_base4_equals_1232_l4072_407200

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- The sum of 111₄, 323₄, and 132₄ is equal to 1232₄ in base 4 -/
theorem sum_base4_equals_1232 :
  let a := base4ToDecimal [1, 1, 1]
  let b := base4ToDecimal [3, 2, 3]
  let c := base4ToDecimal [1, 3, 2]
  let sum := base4ToDecimal [1, 2, 3, 2]
  a + b + c = sum := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_1232_l4072_407200


namespace NUMINAMATH_CALUDE_david_current_age_l4072_407247

/-- David's current age -/
def david_age : ℕ := sorry

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Number of years until David's age is twice his daughter's -/
def years_until_double : ℕ := 16

theorem david_current_age :
  david_age = 40 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_david_current_age_l4072_407247


namespace NUMINAMATH_CALUDE_rectangle_area_l4072_407209

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + 4 * W = 34) (h2 : 4 * L + 2 * W = 38) :
  L * W = 35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l4072_407209


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l4072_407283

/-- The length of the cycle of the last two digits of 8^n -/
def cycle_length : ℕ := 20

/-- The last two digits of 8^3 -/
def last_two_digits_8_cubed : ℕ := 12

/-- The exponent we're interested in -/
def target_exponent : ℕ := 2023

theorem tens_digit_of_8_pow_2023 : 
  (target_exponent % cycle_length = 3) → 
  (last_two_digits_8_cubed / 10 = 1) → 
  (8^target_exponent / 10 % 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l4072_407283


namespace NUMINAMATH_CALUDE_product_of_numbers_l4072_407267

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x^2 + y^2 = 157) : 
  x * y = 22 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l4072_407267


namespace NUMINAMATH_CALUDE_library_visitors_average_l4072_407234

theorem library_visitors_average (monday_visitors : ℕ) (total_visitors : ℕ) :
  monday_visitors = 50 →
  total_visitors = 250 →
  (total_visitors - (monday_visitors + 2 * monday_visitors)) / 5 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l4072_407234


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4072_407204

/-- If the equation (m+1)x^2 + 2mx + 1 = 0 is linear with respect to x, then its solution is 1/2. -/
theorem linear_equation_solution (m : ℝ) : 
  (m + 1 = 0) → (2*m ≠ 0) → 
  ∃ (x : ℝ), ((m + 1) * x^2 + 2*m*x + 1 = 0) ∧ (x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4072_407204


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l4072_407246

theorem sum_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_l4072_407246


namespace NUMINAMATH_CALUDE_circle_radius_l4072_407213

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 8 = 2*x + 4*y) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l4072_407213


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l4072_407282

/-- Definition of the bow-tie operation -/
noncomputable def bowtie (p q : ℝ) : ℝ := p + Real.sqrt (q + Real.sqrt (q + Real.sqrt (q + Real.sqrt q)))

/-- Theorem: If 5 bow-tie q equals 13, then q equals 56 -/
theorem bowtie_equation_solution (q : ℝ) : bowtie 5 q = 13 → q = 56 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l4072_407282


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4072_407203

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 4) (a - 2)).im ≠ 0 ∧ (Complex.mk (a^2 - 4) (a - 2)).re = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4072_407203


namespace NUMINAMATH_CALUDE_chocolate_difference_l4072_407245

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l4072_407245


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l4072_407251

/-- Volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a b : ℝ) (h_positive : 0 < a ∧ 0 < b) (h_height_constraint : a * Real.sqrt 2 / 2 ≤ b ∧ b < a * Real.sqrt 3 / 2) :
  ∃ V : ℝ, V = (a^3 * b) / (12 * Real.sqrt (3 * a^2 - 4 * b^2)) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l4072_407251


namespace NUMINAMATH_CALUDE_new_person_weight_l4072_407257

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the new person weighs 65 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  weight_replaced = 45 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 65 := by
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l4072_407257


namespace NUMINAMATH_CALUDE_chris_birthday_money_l4072_407242

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l4072_407242


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4072_407220

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first 5 terms of a sequence. -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumFirstFive a = 20) : 
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4072_407220


namespace NUMINAMATH_CALUDE_joshua_bottles_count_l4072_407232

theorem joshua_bottles_count (bottles_per_crate : ℕ) (num_crates : ℕ) (extra_bottles : ℕ) : 
  bottles_per_crate = 12 → 
  num_crates = 10 → 
  extra_bottles = 10 → 
  bottles_per_crate * num_crates + extra_bottles = 130 := by
sorry

end NUMINAMATH_CALUDE_joshua_bottles_count_l4072_407232


namespace NUMINAMATH_CALUDE_divisible_by_seven_l4072_407261

theorem divisible_by_seven (x y : ℕ+) (a b : ℕ) 
  (h1 : 3 * x.val + 4 * y.val = a ^ 2)
  (h2 : 4 * x.val + 3 * y.val = b ^ 2) : 
  7 ∣ x.val ∧ 7 ∣ y.val := by
sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l4072_407261


namespace NUMINAMATH_CALUDE_olives_per_jar_l4072_407219

/-- Proves that the number of olives in a jar is 20 given the problem conditions --/
theorem olives_per_jar (
  total_money : ℝ)
  (olives_needed : ℕ)
  (jar_cost : ℝ)
  (change : ℝ)
  (h1 : total_money = 10)
  (h2 : olives_needed = 80)
  (h3 : jar_cost = 1.5)
  (h4 : change = 4)
  : (olives_needed : ℝ) / ((total_money - change) / jar_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_olives_per_jar_l4072_407219


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l4072_407201

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add necessary fields if needed

/-- A diagonal in a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add necessary fields if needed

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The set of all diagonals in a hexagon -/
def all_diagonals (H : ConvexHexagon) : Set (Diagonal H) :=
  sorry

/-- The number of diagonals in a hexagon -/
def num_diagonals (H : ConvexHexagon) : ℕ :=
  9

/-- The number of pairs of diagonals that intersect inside the hexagon -/
def num_intersecting_pairs (H : ConvexHexagon) : ℕ :=
  15

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def prob_intersect (H : ConvexHexagon) : ℚ :=
  15 / 36

theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  prob_intersect H = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l4072_407201


namespace NUMINAMATH_CALUDE_division_problem_l4072_407277

theorem division_problem : ∃ (D : ℕ+) (N : ℤ), 
  N = 5 * D.val ∧ N % 11 = 2 ∧ D = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4072_407277


namespace NUMINAMATH_CALUDE_special_function_sum_l4072_407279

/-- A function satisfying f(p+q) = f(p)f(q) for all p and q, and f(1) = 3 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ p q : ℝ, f (p + q) = f p * f q) ∧ (f 1 = 3)

/-- The main theorem to prove -/
theorem special_function_sum (f : ℝ → ℝ) (h : special_function f) :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 +
  (f 4^2 + f 8) / f 7 + (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_l4072_407279


namespace NUMINAMATH_CALUDE_pyramid_volume_from_star_figure_l4072_407291

/-- The volume of a pyramid formed by folding a star figure cut from a square --/
theorem pyramid_volume_from_star_figure (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 40)
  (h_inner : inner_side = 15) :
  let base_area := inner_side ^ 2
  let midpoint_to_center := outer_side / 2
  let center_to_inner_side := inner_side / 2
  let triangle_height := midpoint_to_center - center_to_inner_side
  let pyramid_height := Real.sqrt (triangle_height ^ 2 - (inner_side / 2) ^ 2)
  let volume := (1 / 3) * base_area * pyramid_height
  volume = 750 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_from_star_figure_l4072_407291


namespace NUMINAMATH_CALUDE_odd_square_mod_eight_l4072_407293

theorem odd_square_mod_eight (k : ℤ) : ∃ m : ℤ, (2 * k + 1)^2 = 8 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_mod_eight_l4072_407293


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l4072_407284

theorem unique_magnitude_of_quadratic_roots (w : ℂ) :
  w^2 - 6*w + 40 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l4072_407284


namespace NUMINAMATH_CALUDE_cube_root_two_identity_l4072_407288

theorem cube_root_two_identity (s : ℝ) : s = 1 / (1 - Real.rpow 2 (1/3)) → s = -(1 + Real.rpow 2 (1/3) + Real.rpow 2 (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_identity_l4072_407288


namespace NUMINAMATH_CALUDE_gcd_315_2016_l4072_407215

theorem gcd_315_2016 : Nat.gcd 315 2016 = 63 := by sorry

end NUMINAMATH_CALUDE_gcd_315_2016_l4072_407215


namespace NUMINAMATH_CALUDE_contest_result_l4072_407205

/-- The total number of baskets made by Alex, Sandra, and Hector -/
def totalBaskets (alex sandra hector : ℕ) : ℕ := alex + sandra + hector

/-- Theorem: Given the conditions, the total number of baskets is 80 -/
theorem contest_result : ∃ (sandra hector : ℕ),
  sandra = 3 * 8 ∧ 
  hector = 2 * sandra ∧
  totalBaskets 8 sandra hector = 80 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l4072_407205


namespace NUMINAMATH_CALUDE_average_score_is_correct_rounded_average_score_l4072_407210

/-- Represents the score distribution for a class test --/
structure ScoreDistribution where
  score_100 : Nat
  score_95  : Nat
  score_85  : Nat
  score_75  : Nat
  score_65  : Nat
  score_55  : Nat
  score_45  : Nat

/-- Calculates the average score given a score distribution --/
def calculateAverageScore (dist : ScoreDistribution) : Rat :=
  let totalStudents := dist.score_100 + dist.score_95 + dist.score_85 + 
                       dist.score_75 + dist.score_65 + dist.score_55 + dist.score_45
  let totalScore := 100 * dist.score_100 + 95 * dist.score_95 + 85 * dist.score_85 +
                    75 * dist.score_75 + 65 * dist.score_65 + 55 * dist.score_55 +
                    45 * dist.score_45
  totalScore / totalStudents

/-- The main theorem stating that the average score is approximately 76.3333 --/
theorem average_score_is_correct (dist : ScoreDistribution) 
  (h1 : dist.score_100 = 10)
  (h2 : dist.score_95 = 20)
  (h3 : dist.score_85 = 40)
  (h4 : dist.score_75 = 30)
  (h5 : dist.score_65 = 25)
  (h6 : dist.score_55 = 15)
  (h7 : dist.score_45 = 10) :
  calculateAverageScore dist = 11450 / 150 := by
  sorry

/-- The rounded average score is 76 --/
theorem rounded_average_score (dist : ScoreDistribution)
  (h : calculateAverageScore dist = 11450 / 150) :
  Int.floor (calculateAverageScore dist + 1/2) = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_correct_rounded_average_score_l4072_407210
