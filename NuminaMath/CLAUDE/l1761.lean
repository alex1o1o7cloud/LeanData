import Mathlib

namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l1761_176135

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ)
  (h1 : alpha_day1_score = 160)
  (h2 : alpha_day1_total = 300)
  (h3 : alpha_day2_score = 140)
  (h4 : alpha_day2_total = 200)
  (h5 : beta_day1_total + beta_day2_total = 500)
  (h6 : beta_day1_total ≠ 300)
  (h7 : beta_day1_score > 0)
  (h8 : beta_day2_score > 0)
  (h9 : (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h10 : (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5) :
  (beta_day1_score + beta_day2_score : ℚ) / (beta_day1_total + beta_day2_total) ≤ 349/500 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l1761_176135


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l1761_176170

theorem simplify_and_sum_exponents (a b d : ℝ) : 
  ∃ (k : ℝ), (54 * a^5 * b^9 * d^14)^(1/3) = 3 * a * b^3 * d^4 * k ∧ 1 + 3 + 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l1761_176170


namespace NUMINAMATH_CALUDE_square_2209_product_l1761_176193

theorem square_2209_product (x : ℤ) (h : x^2 = 2209) : (x + 2) * (x - 2) = 2205 := by
  sorry

end NUMINAMATH_CALUDE_square_2209_product_l1761_176193


namespace NUMINAMATH_CALUDE_commute_distance_is_21_l1761_176178

/-- Represents the carpool scenario with given parameters -/
structure Carpool where
  friends : ℕ := 5
  gas_price : ℚ := 5/2
  car_efficiency : ℚ := 30
  commute_days_per_week : ℕ := 5
  commute_weeks_per_month : ℕ := 4
  individual_payment : ℚ := 14

/-- Calculates the one-way commute distance given a Carpool scenario -/
def calculate_commute_distance (c : Carpool) : ℚ :=
  (c.individual_payment * c.friends * c.car_efficiency) / 
  (2 * c.gas_price * c.commute_days_per_week * c.commute_weeks_per_month)

/-- Theorem stating that the one-way commute distance is 21 miles -/
theorem commute_distance_is_21 (c : Carpool) : 
  calculate_commute_distance c = 21 := by
  sorry

end NUMINAMATH_CALUDE_commute_distance_is_21_l1761_176178


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l1761_176145

theorem alcohol_water_ratio (mixture : ℝ) (alcohol water : ℝ) 
  (h1 : alcohol = (1 : ℝ) / 7 * mixture) 
  (h2 : water = (2 : ℝ) / 7 * mixture) : 
  alcohol / water = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l1761_176145


namespace NUMINAMATH_CALUDE_mojave_population_in_five_years_l1761_176109

/-- Calculates the future population of Mojave --/
def future_population (initial_population : ℕ) (growth_factor : ℕ) (percentage_increase : ℕ) : ℕ :=
  (initial_population * growth_factor) * (100 + percentage_increase) / 100

/-- Theorem stating the future population of Mojave --/
theorem mojave_population_in_five_years 
  (initial_population : ℕ) 
  (growth_factor : ℕ) 
  (percentage_increase : ℕ) 
  (h1 : initial_population = 4000)
  (h2 : growth_factor = 3)
  (h3 : percentage_increase = 40) :
  future_population initial_population growth_factor percentage_increase = 16800 := by
  sorry

end NUMINAMATH_CALUDE_mojave_population_in_five_years_l1761_176109


namespace NUMINAMATH_CALUDE_tan_A_gt_1_necessary_not_sufficient_l1761_176133

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_180 : A + B + C = π

-- Define the property "A is not the smallest angle"
def not_smallest_angle (t : AcuteTriangle) : Prop :=
  t.A ≥ t.B ∨ t.A ≥ t.C

-- Theorem statement
theorem tan_A_gt_1_necessary_not_sufficient (t : AcuteTriangle) :
  (¬(not_smallest_angle t) → ¬(Real.tan t.A > 1)) ∧
  ¬(Real.tan t.A > 1 → not_smallest_angle t) :=
sorry

end NUMINAMATH_CALUDE_tan_A_gt_1_necessary_not_sufficient_l1761_176133


namespace NUMINAMATH_CALUDE_isabel_finished_problems_l1761_176186

/-- Calculates the number of finished homework problems given the initial total,
    remaining pages, and problems per page. -/
def finished_problems (initial : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial - (remaining_pages * problems_per_page)

/-- Proves that Isabel finished 32 problems given the initial conditions. -/
theorem isabel_finished_problems :
  finished_problems 72 5 8 = 32 := by
  sorry


end NUMINAMATH_CALUDE_isabel_finished_problems_l1761_176186


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1761_176156

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1761_176156


namespace NUMINAMATH_CALUDE_ralph_wild_animal_pictures_l1761_176102

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference between Derrick's and Ralph's picture count -/
def picture_difference : ℕ := 8

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := derrick_pictures - picture_difference

theorem ralph_wild_animal_pictures : ralph_pictures = 26 := by sorry

end NUMINAMATH_CALUDE_ralph_wild_animal_pictures_l1761_176102


namespace NUMINAMATH_CALUDE_point_outside_circle_l1761_176117

theorem point_outside_circle (a : ℝ) : 
  let P : ℝ × ℝ := (a, 10)
  let C : ℝ × ℝ := (1, 1)
  let r : ℝ := Real.sqrt 2
  let d : ℝ := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  d > r := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1761_176117


namespace NUMINAMATH_CALUDE_sharona_bought_four_more_pencils_l1761_176195

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 11

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 13

/-- The number of pencils Sharona bought -/
def sharona_pencils : ℕ := 17

/-- The amount Jamar paid in cents -/
def jamar_paid : ℕ := 143

/-- The amount Sharona paid in cents -/
def sharona_paid : ℕ := 187

theorem sharona_bought_four_more_pencils :
  pencil_price > 1 ∧
  pencil_price * jamar_pencils = jamar_paid ∧
  pencil_price * sharona_pencils = sharona_paid →
  sharona_pencils - jamar_pencils = 4 :=
by sorry

end NUMINAMATH_CALUDE_sharona_bought_four_more_pencils_l1761_176195


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1761_176144

/-- Given z = 1 + i and (z^2 + az + b) / (z^2 - z + 1) = 1 - i, where a and b are real numbers, 
    then a = -1 and b = 2. -/
theorem complex_fraction_equality (a b : ℝ) : 
  let z : ℂ := 1 + I
  ((z^2 + a*z + b) / (z^2 - z + 1) = 1 - I) → (a = -1 ∧ b = 2) := by
sorry


end NUMINAMATH_CALUDE_complex_fraction_equality_l1761_176144


namespace NUMINAMATH_CALUDE_even_odd_property_l1761_176166

theorem even_odd_property (a b : ℤ) : 
  (Even (a - b) ∧ Odd (a + b + 1)) ∨ (Odd (a - b) ∧ Even (a + b + 1)) := by
sorry

end NUMINAMATH_CALUDE_even_odd_property_l1761_176166


namespace NUMINAMATH_CALUDE_apple_price_36kg_l1761_176164

/-- The price of apples for a given weight --/
def apple_price (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

theorem apple_price_36kg (l q : ℚ) : 
  (apple_price l q 20 = 100) → 
  (apple_price l q 33 = 168) → 
  (apple_price l q 36 = 186) := by
  sorry

#check apple_price_36kg

end NUMINAMATH_CALUDE_apple_price_36kg_l1761_176164


namespace NUMINAMATH_CALUDE_derivative_limit_relation_l1761_176165

theorem derivative_limit_relation (f : ℝ → ℝ) (x₀ : ℝ) (h : HasDerivAt f 2 x₀) :
  Filter.Tendsto (fun k => (f (x₀ - k) - f x₀) / (2 * k)) (Filter.atTop.comap (fun k => 1 / k)) (nhds (-1)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_limit_relation_l1761_176165


namespace NUMINAMATH_CALUDE_g_max_min_sum_l1761_176158

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + x

theorem g_max_min_sum :
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 7 :=
by sorry

end NUMINAMATH_CALUDE_g_max_min_sum_l1761_176158


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l1761_176143

/-- The total sleep John got in a week given specific sleep patterns --/
def totalSleepInWeek (daysWithLowSleep : ℕ) (hoursLowSleep : ℕ) 
  (recommendedSleep : ℕ) (percentageNormalSleep : ℚ) : ℚ :=
  (daysWithLowSleep * hoursLowSleep : ℚ) + 
  ((7 - daysWithLowSleep) * (recommendedSleep * percentageNormalSleep))

/-- Theorem stating that John's total sleep for the week is 30 hours --/
theorem johns_weekly_sleep : 
  totalSleepInWeek 2 3 8 (60 / 100) = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_sleep_l1761_176143


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l1761_176189

/-- The ratio of radii of two spheres given their volumes -/
theorem sphere_radii_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) 
  (h2 : V_small = 0.275 * V_large) : 
  (V_small / V_large)^(1/3 : ℝ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l1761_176189


namespace NUMINAMATH_CALUDE_right_triangle_area_l1761_176113

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) (h_hypotenuse : c = 10) : 
  (1 / 2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1761_176113


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l1761_176190

theorem hyperbola_line_intersection (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ P Q : ℝ × ℝ,
    (P.1^2 / a - P.2^2 / b = 1) ∧
    (Q.1^2 / a - Q.2^2 / b = 1) ∧
    (P.1 + P.2 = 1) ∧
    (Q.1 + Q.2 = 1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    1 / a - 1 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l1761_176190


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1761_176121

theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^10 + 11 * x^9) + (5 * x^12 + 3 * x^10 + x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10) =
  20 * x^12 + 11 * x^10 + 12 * x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1761_176121


namespace NUMINAMATH_CALUDE_square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l1761_176107

theorem square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven
  (a b : ℝ) (h : a^2 - 3*b = 5) : 2*a^2 - 6*b + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l1761_176107


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_exterior_angle_ratio_l1761_176199

theorem isosceles_triangle_with_exterior_angle_ratio (α β γ : ℝ) : 
  -- The triangle is isosceles
  β = γ →
  -- Two exterior angles are in the ratio of 1:4
  ∃ (x : ℝ), (180 - α = x ∧ 180 - β = 4*x) ∨ (180 - β = x ∧ 180 - α = 4*x) →
  -- The sum of interior angles is 180°
  α + β + γ = 180 →
  -- The interior angles are 140°, 20°, and 20°
  α = 140 ∧ β = 20 ∧ γ = 20 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_exterior_angle_ratio_l1761_176199


namespace NUMINAMATH_CALUDE_ned_second_table_trays_l1761_176184

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def total_trips : ℕ := 4

/-- The number of trays Ned picked up from the first table -/
def trays_first_table : ℕ := 27

/-- The number of trays Ned picked up from the second table -/
def trays_second_table : ℕ := total_trips * trays_per_trip - trays_first_table

theorem ned_second_table_trays : trays_second_table = 5 := by
  sorry

end NUMINAMATH_CALUDE_ned_second_table_trays_l1761_176184


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l1761_176162

/-- Proves that a tax reduction of 20% results in a 4% revenue decrease when consumption increases by 20% -/
theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : x > 0)
  (h2 : T > 0)
  (h3 : C > 0)
  (h4 : (T - x / 100 * T) * (C + 20 / 100 * C) = 0.96 * T * C) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l1761_176162


namespace NUMINAMATH_CALUDE_function_property_implies_range_l1761_176163

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem function_property_implies_range (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_even_function f) 
  (h2 : decreasing_on_nonnegative f) 
  (h3 : f (a + 2) > f (a - 3)) : 
  a < 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_property_implies_range_l1761_176163


namespace NUMINAMATH_CALUDE_audiobook_completion_time_l1761_176179

/-- Calculates the time to finish audiobooks given the number of books, length per book, and daily listening time. -/
def timeToFinishAudiobooks (numBooks : ℕ) (hoursPerBook : ℕ) (hoursPerDay : ℕ) : ℕ :=
  numBooks * (hoursPerBook / hoursPerDay)

/-- Proves that under the given conditions, it takes 90 days to finish the audiobooks. -/
theorem audiobook_completion_time :
  timeToFinishAudiobooks 6 30 2 = 90 :=
by
  sorry

#eval timeToFinishAudiobooks 6 30 2

end NUMINAMATH_CALUDE_audiobook_completion_time_l1761_176179


namespace NUMINAMATH_CALUDE_triangle_area_l1761_176132

/-- Given a triangle ABC with the following properties:
  * A = 60°
  * a = √3
  * sin B + sin C = 6√2 * sin B * sin C
  Prove that the area of the triangle is √3/8 -/
theorem triangle_area (B C : ℝ) (b c : ℝ) : 
  let A : ℝ := π / 3
  let a : ℝ := Real.sqrt 3
  Real.sin B + Real.sin C = 6 * Real.sqrt 2 * Real.sin B * Real.sin C →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1761_176132


namespace NUMINAMATH_CALUDE_cosine_value_implies_expression_value_l1761_176129

theorem cosine_value_implies_expression_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos (π/2 + x) = 4/5) : 
  (Real.sin (2*x) - 2 * (Real.sin x)^2) / (1 + Real.tan x) = -168/25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_implies_expression_value_l1761_176129


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1761_176188

theorem sixth_term_of_geometric_sequence (a₁ a₉ : ℝ) (h₁ : a₁ = 12) (h₂ : a₉ = 31104) :
  let r := (a₉ / a₁) ^ (1 / 8)
  let a₆ := a₁ * r ^ 5
  a₆ = 93312 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1761_176188


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1761_176124

theorem max_value_trig_expression :
  ∀ θ φ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 0 ≤ φ ∧ φ ≤ π/2 →
  3 * Real.sin θ * Real.cos φ + 2 * (Real.sin φ)^2 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1761_176124


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1761_176101

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1761_176101


namespace NUMINAMATH_CALUDE_notebook_purchase_difference_l1761_176122

theorem notebook_purchase_difference (price : ℚ) (marie_count jake_count : ℕ) : 
  price > (1/4 : ℚ) →
  price * marie_count = (15/4 : ℚ) →
  price * jake_count = 5 →
  jake_count - marie_count = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_difference_l1761_176122


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l1761_176173

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l1761_176173


namespace NUMINAMATH_CALUDE_find_5b_l1761_176108

theorem find_5b (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_5b_l1761_176108


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l1761_176172

theorem max_distinct_pairs (n : ℕ) (h : n = 3010) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1201 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3005) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ), m > k →
      ¬∃ (pairs' : Finset (ℕ × ℕ)),
        pairs'.card = m ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
        (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 3005) ∧
        (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l1761_176172


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1761_176187

theorem expansion_coefficient (n : ℕ) : 
  ((-2:ℤ)^n + n * (-2:ℤ)^(n-1) = -128) → n = 6 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1761_176187


namespace NUMINAMATH_CALUDE_dog_human_years_ratio_l1761_176141

theorem dog_human_years_ratio : 
  (∀ (dog_age human_age : ℝ), dog_age = 7 * human_age) → 
  (∃ (x : ℝ), x * 3 = 21 ∧ 7 / x = 7 / 6) :=
by sorry

end NUMINAMATH_CALUDE_dog_human_years_ratio_l1761_176141


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1761_176130

/-- Represents a cylindrical fuel tank -/
structure FuelTank where
  capacity : ℝ
  initial_percentage : ℝ
  initial_volume : ℝ

/-- Theorem stating the capacity of the fuel tank -/
theorem fuel_tank_capacity (tank : FuelTank)
  (h1 : tank.initial_percentage = 0.25)
  (h2 : tank.initial_volume = 60)
  : tank.capacity = 240 := by
  sorry

#check fuel_tank_capacity

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1761_176130


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l1761_176180

theorem geometric_sequence_proof (m : ℝ) :
  (4 / 1 = (2 * m + 8) / 4) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l1761_176180


namespace NUMINAMATH_CALUDE_average_carnations_value_l1761_176176

/-- The average number of carnations in Trevor's bouquets -/
def average_carnations : ℚ :=
  let bouquets : List ℕ := [9, 23, 13, 36, 28, 45]
  (bouquets.sum : ℚ) / bouquets.length

/-- Proof that the average number of carnations is 25.67 -/
theorem average_carnations_value :
  average_carnations = 25.67 := by
  sorry

end NUMINAMATH_CALUDE_average_carnations_value_l1761_176176


namespace NUMINAMATH_CALUDE_floor_expression_equals_two_l1761_176161

theorem floor_expression_equals_two :
  ⌊(2012^3 : ℝ) / (2010 * 2011) + (2010^3 : ℝ) / (2011 * 2012)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_two_l1761_176161


namespace NUMINAMATH_CALUDE_parabola_transformation_l1761_176148

/-- A parabola is above a line if it opens upwards and doesn't intersect the line. -/
def parabola_above_line (a b c : ℝ) : Prop :=
  a > 0 ∧ (b - c)^2 < 4*a*c

theorem parabola_transformation (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (above : parabola_above_line a b c) : 
  parabola_above_line c (-b) a :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1761_176148


namespace NUMINAMATH_CALUDE_fraction_equality_l1761_176137

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem fraction_equality : 
  let a : ℝ := 3
  let b : ℝ := 2
  (at_op a b) / (hash_op a b) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1761_176137


namespace NUMINAMATH_CALUDE_door_replacement_cost_l1761_176140

/-- The total cost of replacing doors given the number of bedroom and outside doors,
    the cost of outside doors, and that bedroom doors cost half as much as outside doors. -/
def total_door_cost (num_bedroom_doors num_outside_doors outside_door_cost : ℕ) : ℕ :=
  num_outside_doors * outside_door_cost +
  num_bedroom_doors * (outside_door_cost / 2)

/-- Theorem stating that the total cost for replacing 3 bedroom doors and 2 outside doors
    is $70, given that outside doors cost $20 each and bedroom doors cost half as much. -/
theorem door_replacement_cost :
  total_door_cost 3 2 20 = 70 := by
  sorry


end NUMINAMATH_CALUDE_door_replacement_cost_l1761_176140


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1761_176154

/-- A Mersenne number is of the form 2^n - 1 where n is a positive integer. -/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A Mersenne prime is a Mersenne number that is also prime. -/
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = mersenne_number n ∧ Prime p

/-- The largest Mersenne prime less than 500 is 127. -/
theorem largest_mersenne_prime_under_500 :
  (∀ p : ℕ, p < 500 → is_mersenne_prime p → p ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1761_176154


namespace NUMINAMATH_CALUDE_max_carps_eaten_l1761_176155

/-- Represents the eating behavior of pikes in a pond -/
structure PikePond where
  initialPikes : ℕ
  pikesForFull : ℕ
  carpPerFull : ℕ

/-- Calculates the maximum number of full pikes -/
def maxFullPikes (pond : PikePond) : ℕ :=
  (pond.initialPikes - 1) / pond.pikesForFull

/-- Theorem: The maximum number of crucian carps eaten is 9 given the initial conditions -/
theorem max_carps_eaten (pond : PikePond) 
  (h1 : pond.initialPikes = 30)
  (h2 : pond.pikesForFull = 3)
  (h3 : pond.carpPerFull = 1) : 
  maxFullPikes pond * pond.carpPerFull = 9 := by
  sorry

#eval maxFullPikes { initialPikes := 30, pikesForFull := 3, carpPerFull := 1 }

end NUMINAMATH_CALUDE_max_carps_eaten_l1761_176155


namespace NUMINAMATH_CALUDE_prob_basket_A_given_white_l1761_176100

/-- Represents a basket with white and black balls -/
structure Basket where
  white : ℕ
  black : ℕ

/-- The probability of choosing a specific basket -/
def choose_probability : ℚ := 1/2

/-- Calculates the probability of picking a white ball from a given basket -/
def white_probability (b : Basket) : ℚ :=
  b.white / (b.white + b.black)

/-- Theorem: Probability of choosing Basket A given a white ball was picked -/
theorem prob_basket_A_given_white 
  (basket_A basket_B : Basket)
  (h_A : basket_A = ⟨2, 3⟩)
  (h_B : basket_B = ⟨1, 3⟩) :
  let p_A := choose_probability
  let p_B := choose_probability
  let p_W_A := white_probability basket_A
  let p_W_B := white_probability basket_B
  let p_W := p_A * p_W_A + p_B * p_W_B
  p_A * p_W_A / p_W = 8/13 := by
    sorry

end NUMINAMATH_CALUDE_prob_basket_A_given_white_l1761_176100


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l1761_176192

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 10 (2*x) - Nat.choose 10 (x+1) = 0) → (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l1761_176192


namespace NUMINAMATH_CALUDE_odd_function_derivative_even_l1761_176127

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_derivative_even
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hodd : odd_function f) :
  even_function (deriv f) :=
sorry

end NUMINAMATH_CALUDE_odd_function_derivative_even_l1761_176127


namespace NUMINAMATH_CALUDE_wang_yue_more_stable_l1761_176168

def li_na_scores : List ℝ := [80, 70, 90, 70]
def wang_yue_scores (a : ℝ) : List ℝ := [80, a, 70, 90]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem wang_yue_more_stable (a : ℝ) :
  a ≥ 70 →
  median li_na_scores + 5 = median (wang_yue_scores a) →
  variance (wang_yue_scores a) < variance li_na_scores :=
sorry

end NUMINAMATH_CALUDE_wang_yue_more_stable_l1761_176168


namespace NUMINAMATH_CALUDE_second_student_male_probability_l1761_176181

/-- The probability that the second student to leave is male, given 2 male and 2 female students -/
def probability_second_male (num_male num_female : ℕ) : ℚ :=
  if num_male = 2 ∧ num_female = 2 then 1/6 else 0

/-- Theorem stating that the probability of the second student to leave being male is 1/6 -/
theorem second_student_male_probability :
  probability_second_male 2 2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_second_student_male_probability_l1761_176181


namespace NUMINAMATH_CALUDE_mask_digit_correct_l1761_176125

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Associates each mask with a digit -/
def mask_digit : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- The theorem to be proved -/
theorem mask_digit_correct :
  (mask_digit Mask.elephant) * (mask_digit Mask.elephant) = 36 ∧
  (mask_digit Mask.mouse) * (mask_digit Mask.mouse) = 16 ∧
  (mask_digit Mask.pig) * (mask_digit Mask.pig) = 64 ∧
  (mask_digit Mask.panda) * (mask_digit Mask.panda) = 1 ∧
  (∀ m1 m2 : Mask, m1 ≠ m2 → mask_digit m1 ≠ mask_digit m2) :=
by sorry

#check mask_digit_correct

end NUMINAMATH_CALUDE_mask_digit_correct_l1761_176125


namespace NUMINAMATH_CALUDE_jims_journey_distance_l1761_176149

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ := driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 642 558 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l1761_176149


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l1761_176157

theorem cistern_emptying_time (fill_time : ℝ) (combined_fill_time : ℝ) (empty_time : ℝ) : 
  fill_time = 2 → 
  combined_fill_time = 2.571428571428571 →
  (1 / fill_time) - (1 / empty_time) = (1 / combined_fill_time) →
  empty_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l1761_176157


namespace NUMINAMATH_CALUDE_stratified_sampling_boys_l1761_176119

theorem stratified_sampling_boys (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) :
  total_boys = 48 →
  total_girls = 36 →
  sample_size = 21 →
  (total_boys * sample_size) / (total_boys + total_girls) = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_boys_l1761_176119


namespace NUMINAMATH_CALUDE_one_zero_quadratic_l1761_176136

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem one_zero_quadratic (a : ℝ) :
  (∃! x, f a x = 0) → (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_one_zero_quadratic_l1761_176136


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1761_176183

/-- Given two runners a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.05 part of the race length as a head start to finish at the same time,
    prove that the ratio of a's speed to b's speed is 1/0.95 -/
theorem race_speed_ratio (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : ∀ L : ℝ, L > 0 → L / v_a = (L - 0.05 * L) / v_b) : 
  v_a / v_b = 1 / 0.95 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1761_176183


namespace NUMINAMATH_CALUDE_cross_pentominoes_fit_on_chessboard_l1761_176106

/-- A "cross" pentomino consists of 5 unit squares -/
def cross_pentomino_area : ℝ := 5

/-- The chessboard is 8x8 units -/
def chessboard_side : ℝ := 8

/-- The number of cross pentominoes to be cut -/
def num_crosses : ℕ := 9

/-- The area of half-rectangles between crosses -/
def half_rectangle_area : ℝ := 1

/-- The number of half-rectangles -/
def num_half_rectangles : ℕ := 8

/-- The maximum area of corner pieces -/
def max_corner_piece_area : ℝ := 1.5

/-- The number of corner pieces -/
def num_corner_pieces : ℕ := 4

theorem cross_pentominoes_fit_on_chessboard :
  (num_crosses : ℝ) * cross_pentomino_area +
  (num_half_rectangles : ℝ) * half_rectangle_area +
  (num_corner_pieces : ℝ) * max_corner_piece_area ≤ chessboard_side ^ 2 :=
sorry

end NUMINAMATH_CALUDE_cross_pentominoes_fit_on_chessboard_l1761_176106


namespace NUMINAMATH_CALUDE_least_common_denominator_l1761_176134

theorem least_common_denominator : 
  Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 8))))) = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l1761_176134


namespace NUMINAMATH_CALUDE_sqrt_47_minus_2_range_l1761_176169

theorem sqrt_47_minus_2_range : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_47_minus_2_range_l1761_176169


namespace NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l1761_176110

theorem min_colors_for_distribution_centers (n : ℕ) : n ≥ 5 ↔ n + n.choose 2 ≥ 12 := by
  sorry

#check min_colors_for_distribution_centers

end NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l1761_176110


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1761_176182

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1761_176182


namespace NUMINAMATH_CALUDE_triangle_property_l1761_176198

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

/-- The main theorem stating the conditions and conclusion about the triangle. -/
theorem triangle_property (t : Triangle) 
  (h1 : 2*t.a*(Real.sin t.A) = (2*t.b + t.c)*(Real.sin t.B) + (2*t.c + t.b)*(Real.sin t.C))
  (h2 : Real.sin t.B + Real.sin t.C = 1) :
  t.A = 2*π/3 ∧ t.B = π/6 ∧ t.C = π/6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1761_176198


namespace NUMINAMATH_CALUDE_product_of_surds_l1761_176152

theorem product_of_surds : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_surds_l1761_176152


namespace NUMINAMATH_CALUDE_harmony_sum_l1761_176196

def alphabet_value (n : ℕ) : ℤ :=
  match n % 13 with
  | 0 => -3
  | 1 => -2
  | 2 => -1
  | 3 => 0
  | 4 => 1
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 1
  | 9 => 0
  | 10 => -1
  | 11 => -2
  | 12 => -3
  | _ => 0  -- This case should never occur due to the modulo operation

theorem harmony_sum : 
  alphabet_value 8 + alphabet_value 1 + alphabet_value 18 + 
  alphabet_value 13 + alphabet_value 15 + alphabet_value 14 + 
  alphabet_value 25 = -7 := by
sorry

end NUMINAMATH_CALUDE_harmony_sum_l1761_176196


namespace NUMINAMATH_CALUDE_acme_soup_words_count_l1761_176151

/-- The number of possible words of length n formed from a set of k distinct letters,
    where each letter appears at least n times. -/
def word_count (n k : ℕ) : ℕ := k^n

/-- The specific case for 6-letter words formed from 6 distinct letters. -/
def acme_soup_words : ℕ := word_count 6 6

theorem acme_soup_words_count :
  acme_soup_words = 46656 := by
  sorry

end NUMINAMATH_CALUDE_acme_soup_words_count_l1761_176151


namespace NUMINAMATH_CALUDE_genuine_product_probability_l1761_176142

theorem genuine_product_probability 
  (p_second : ℝ) 
  (p_third : ℝ) 
  (h1 : p_second = 0.03) 
  (h2 : p_third = 0.01) 
  : 1 - (p_second + p_third) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_genuine_product_probability_l1761_176142


namespace NUMINAMATH_CALUDE_twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l1761_176131

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to form
    a square without breaking any sticks, it returns 0. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 12 sticks, we need to break 2 sticks to form a square -/
theorem twelve_sticks_need_two_breaks : minSticksToBreak 12 = 2 :=
  sorry

/-- Theorem stating that for 15 sticks, we can form a square without breaking any sticks -/
theorem fifteen_sticks_no_breaks : minSticksToBreak 15 = 0 :=
  sorry

end NUMINAMATH_CALUDE_twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l1761_176131


namespace NUMINAMATH_CALUDE_kabadi_players_count_l1761_176139

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := total_players - kho_kho_only + both_games

theorem kabadi_players_count : kabadi_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_kabadi_players_count_l1761_176139


namespace NUMINAMATH_CALUDE_sum_of_multiples_l1761_176167

theorem sum_of_multiples (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l1761_176167


namespace NUMINAMATH_CALUDE_flowmaster_pump_l1761_176123

/-- The FlowMaster pump problem -/
theorem flowmaster_pump (pump_rate : ℝ) (time : ℝ) (h1 : pump_rate = 600) (h2 : time = 0.5) :
  pump_rate * time = 300 := by
  sorry

end NUMINAMATH_CALUDE_flowmaster_pump_l1761_176123


namespace NUMINAMATH_CALUDE_all_divisors_end_in_one_l1761_176114

theorem all_divisors_end_in_one (n : ℕ+) :
  ∀ d : ℕ, d > 0 → d ∣ ((10^(5^n.val) - 1) / 9) → d % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_divisors_end_in_one_l1761_176114


namespace NUMINAMATH_CALUDE_tile_arrangement_exists_l1761_176147

/-- Represents a 2x1 tile with a diagonal -/
structure Tile :=
  (position : Fin 6 × Fin 6)  -- Top-left corner position in the 6x6 grid
  (orientation : Bool)        -- True for horizontal, False for vertical
  (diagonal : Bool)           -- True for one diagonal direction, False for the other

/-- Represents the 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Option Tile

/-- Check if a tile placement is valid -/
def valid_placement (grid : Grid) (tile : Tile) : Prop :=
  -- Add conditions to check if the tile fits within the grid
  -- and doesn't overlap with other tiles
  sorry

/-- Check if diagonal endpoints don't coincide -/
def no_coinciding_diagonals (grid : Grid) : Prop :=
  -- Add conditions to check that no diagonal endpoints coincide
  sorry

theorem tile_arrangement_exists : ∃ (grid : Grid),
  (∃ (tiles : Finset Tile), tiles.card = 18 ∧ 
    (∀ t ∈ tiles, valid_placement grid t)) ∧
  no_coinciding_diagonals grid :=
sorry

end NUMINAMATH_CALUDE_tile_arrangement_exists_l1761_176147


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1761_176177

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * (2 - z) = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1761_176177


namespace NUMINAMATH_CALUDE_exam_pass_count_l1761_176159

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), 
    passed ≤ total ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg ∧
    passed = 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_count_l1761_176159


namespace NUMINAMATH_CALUDE_sarah_eli_age_ratio_l1761_176103

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, 
    prove that the ratio of Sarah's age to Eli's age is 2:1 -/
theorem sarah_eli_age_ratio :
  ∀ (kaylin_age sarah_age eli_age freyja_age : ℕ),
    kaylin_age = 33 →
    freyja_age = 10 →
    sarah_age = kaylin_age + 5 →
    eli_age = freyja_age + 9 →
    ∃ (n : ℕ), sarah_age = n * eli_age →
    sarah_age / eli_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_eli_age_ratio_l1761_176103


namespace NUMINAMATH_CALUDE_solve_star_equation_l1761_176138

-- Define the custom operation ※
def star (a b : ℚ) : ℚ := a + b

-- State the theorem
theorem solve_star_equation :
  ∃ x : ℚ, star 4 (star x 3) = 1 ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1761_176138


namespace NUMINAMATH_CALUDE_village_households_l1761_176191

/-- The number of households in a village where:
    1. Each household requires 20 litres of water per month
    2. 2000 litres of water lasts for 10 months for all households -/
def number_of_households : ℕ := 10

/-- The amount of water required per household per month (in litres) -/
def water_per_household_per_month : ℕ := 20

/-- The total amount of water available (in litres) -/
def total_water : ℕ := 2000

/-- The number of months the water supply lasts -/
def months_supply : ℕ := 10

theorem village_households :
  number_of_households * water_per_household_per_month * months_supply = total_water :=
by sorry

end NUMINAMATH_CALUDE_village_households_l1761_176191


namespace NUMINAMATH_CALUDE_equation_equivalence_l1761_176185

theorem equation_equivalence (x y z : ℝ) :
  (x - z)^2 - 4*(x - y)*(y - z) = 0 → z + x - 2*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1761_176185


namespace NUMINAMATH_CALUDE_speed_equivalence_l1761_176194

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 18.334799999999998

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 66.00528

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l1761_176194


namespace NUMINAMATH_CALUDE_max_subsets_of_N_l1761_176115

/-- The set M -/
def M : Finset ℕ := {0, 2, 3, 7}

/-- The set N -/
def N : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 * p.2) (M.product M)

/-- Theorem: The maximum number of subsets of N is 128 -/
theorem max_subsets_of_N : Finset.card (Finset.powerset N) = 128 := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_of_N_l1761_176115


namespace NUMINAMATH_CALUDE_tom_payment_multiple_l1761_176105

def original_price : ℝ := 3.00
def tom_payment : ℝ := 9.00

theorem tom_payment_multiple : tom_payment / original_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_multiple_l1761_176105


namespace NUMINAMATH_CALUDE_ice_pop_probability_l1761_176171

def total_ice_pops : ℕ := 17
def cherry_ice_pops : ℕ := 5
def children : ℕ := 5

theorem ice_pop_probability :
  1 - (Nat.factorial cherry_ice_pops : ℚ) / (Nat.factorial total_ice_pops / Nat.factorial (total_ice_pops - children)) = 1 - 1 / 4762 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_probability_l1761_176171


namespace NUMINAMATH_CALUDE_max_base8_digit_sum_l1761_176146

-- Define a function to convert a natural number to its base-8 representation
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to sum the digits of a number in its base-8 representation
def sumBase8Digits (n : ℕ) : ℕ :=
  (toBase8 n).sum

-- Theorem statement
theorem max_base8_digit_sum :
  ∃ (m : ℕ), m < 5000 ∧ 
  (∀ (n : ℕ), n < 5000 → sumBase8Digits n ≤ sumBase8Digits m) ∧
  sumBase8Digits m = 28 :=
sorry

end NUMINAMATH_CALUDE_max_base8_digit_sum_l1761_176146


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1761_176120

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : (ℝ × ℝ)) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

theorem parallel_vectors_k_value
  (a b : ℝ × ℝ)  -- a and b are plane vectors
  (h_not_collinear : ¬ are_parallel a b)  -- a and b are non-collinear
  (m : ℝ × ℝ)
  (h_m : m = (a.1 - 2 * b.1, a.2 - 2 * b.2))  -- m = a - 2b
  (k : ℝ)
  (n : ℝ × ℝ)
  (h_n : n = (3 * a.1 + k * b.1, 3 * a.2 + k * b.2))  -- n = 3a + kb
  (h_parallel : are_parallel m n)  -- m is parallel to n
  : k = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1761_176120


namespace NUMINAMATH_CALUDE_log3_negative_implies_x_negative_but_not_conversely_l1761_176153

-- Define the logarithm function with base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Theorem statement
theorem log3_negative_implies_x_negative_but_not_conversely :
  (∀ x : ℝ, log3 (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ log3 (x + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_log3_negative_implies_x_negative_but_not_conversely_l1761_176153


namespace NUMINAMATH_CALUDE_initial_interest_rate_is_45_percent_l1761_176104

/-- Given an initial deposit amount and two interest scenarios, 
    prove that the initial interest rate is 45% --/
theorem initial_interest_rate_is_45_percent 
  (P : ℝ) -- Principal amount (initial deposit)
  (r : ℝ) -- Initial interest rate (as a percentage)
  (h1 : P * r / 100 = 405) -- Interest at initial rate is 405
  (h2 : P * (r + 5) / 100 = 450) -- Interest at (r + 5)% is 450
  : r = 45 := by
sorry

end NUMINAMATH_CALUDE_initial_interest_rate_is_45_percent_l1761_176104


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1761_176150

theorem rectangle_dimension_change (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let A := L * W
  let W' := 0.4 * W
  let A' := 1.36 * A
  ∃ L', L' = 3.4 * L ∧ A' = L' * W' :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1761_176150


namespace NUMINAMATH_CALUDE_yang_hui_field_theorem_l1761_176160

/-- Represents a rectangular field with given area and perimeter --/
structure RectangularField where
  area : ℕ
  perimeter : ℕ

/-- Calculates the difference between length and width of a rectangular field --/
def lengthWidthDifference (field : RectangularField) : ℕ :=
  let length := (field.perimeter + (field.perimeter^2 - 16 * field.area).sqrt) / 4
  let width := field.perimeter / 2 - length
  length - width

/-- Theorem stating the difference between length and width for the specific field --/
theorem yang_hui_field_theorem : 
  ∀ (field : RectangularField), 
  field.area = 864 ∧ field.perimeter = 120 → lengthWidthDifference field = 12 := by
  sorry

end NUMINAMATH_CALUDE_yang_hui_field_theorem_l1761_176160


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1761_176111

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a4a5 : a 4 * a 5 = 1) 
  (h_a8a9 : a 8 * a 9 = 16) : 
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1761_176111


namespace NUMINAMATH_CALUDE_tangent_line_at_one_zero_l1761_176126

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem tangent_line_at_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_zero_l1761_176126


namespace NUMINAMATH_CALUDE_ship_lock_weight_scientific_notation_l1761_176197

theorem ship_lock_weight_scientific_notation :
  867000 = 8.67 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_ship_lock_weight_scientific_notation_l1761_176197


namespace NUMINAMATH_CALUDE_factorization_equality_l1761_176112

theorem factorization_equality (a b : ℝ) : a^3 - 9*a*b^2 = a*(a+3*b)*(a-3*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1761_176112


namespace NUMINAMATH_CALUDE_mikes_weekly_pullups_l1761_176116

/-- Calculates the number of pull-ups Mike does in a week -/
theorem mikes_weekly_pullups 
  (pullups_per_visit : ℕ) 
  (office_visits_per_day : ℕ) 
  (days_in_week : ℕ) 
  (h1 : pullups_per_visit = 2) 
  (h2 : office_visits_per_day = 5) 
  (h3 : days_in_week = 7) : 
  pullups_per_visit * office_visits_per_day * days_in_week = 70 := by
  sorry

#check mikes_weekly_pullups

end NUMINAMATH_CALUDE_mikes_weekly_pullups_l1761_176116


namespace NUMINAMATH_CALUDE_lydia_plant_count_l1761_176128

/-- Represents the total number of plants Lydia has -/
def total_plants : ℕ := sorry

/-- Represents the number of flowering plants Lydia has -/
def flowering_plants : ℕ := sorry

/-- Represents the number of flowering plants on the porch -/
def porch_plants : ℕ := sorry

/-- The percentage of flowering plants among all plants -/
def flowering_percentage : ℚ := 2/5

/-- The fraction of flowering plants on the porch -/
def porch_fraction : ℚ := 1/4

/-- The number of flowers each flowering plant produces -/
def flowers_per_plant : ℕ := 5

/-- The total number of flowers on the porch -/
def total_porch_flowers : ℕ := 40

theorem lydia_plant_count :
  (flowering_plants = flowering_percentage * total_plants) ∧
  (porch_plants = porch_fraction * flowering_plants) ∧
  (total_porch_flowers = porch_plants * flowers_per_plant) →
  total_plants = 80 := by sorry

end NUMINAMATH_CALUDE_lydia_plant_count_l1761_176128


namespace NUMINAMATH_CALUDE_conjecture_proof_l1761_176175

theorem conjecture_proof (n : ℕ) (h : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_conjecture_proof_l1761_176175


namespace NUMINAMATH_CALUDE_remainder_mod_48_l1761_176118

theorem remainder_mod_48 (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + x) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + x) % (6^3) = 6^2 % (6^3)) :
  x % 48 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_mod_48_l1761_176118


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1761_176174

def f (x : ℝ) := x^2 - 4*x + 3

theorem f_decreasing_on_interval :
  ∀ x y : ℝ, x < y → y ≤ 2 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1761_176174
