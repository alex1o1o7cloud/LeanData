import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_inequality_l442_44251

theorem sin_cos_inequality (x : ℝ) : -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l442_44251


namespace NUMINAMATH_CALUDE_interval_equinumerosity_l442_44220

theorem interval_equinumerosity (a : ℝ) (ha : a > 0) :
  ∃ f : Set.Icc 0 1 → Set.Icc 0 a, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_interval_equinumerosity_l442_44220


namespace NUMINAMATH_CALUDE_quadratic_value_at_negative_two_l442_44230

theorem quadratic_value_at_negative_two (a b : ℝ) :
  (2 * a * 1^2 + b * 1 = 3) → (a * (-2)^2 - b * (-2) = 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_negative_two_l442_44230


namespace NUMINAMATH_CALUDE_polynomial_roots_l442_44235

theorem polynomial_roots : ∃ (a b c : ℝ), 
  (a = -1 ∧ b = Real.sqrt 6 ∧ c = -Real.sqrt 6) ∧
  (∀ x : ℝ, x^3 + x^2 - 6*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l442_44235


namespace NUMINAMATH_CALUDE_parade_vehicles_l442_44257

theorem parade_vehicles (b t q : ℕ) : 
  b + t + q = 12 →
  2*b + 3*t + 4*q = 35 →
  q = 5 :=
by sorry

end NUMINAMATH_CALUDE_parade_vehicles_l442_44257


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l442_44286

theorem linear_systems_solutions :
  -- System 1
  (2 : ℝ) + 2 * (1 : ℝ) = (4 : ℝ) ∧
  (2 : ℝ) + 3 * (1 : ℝ) = (5 : ℝ) ∧
  -- System 2
  2 * (2 : ℝ) - 5 * (5 : ℝ) = (-21 : ℝ) ∧
  4 * (2 : ℝ) + 3 * (5 : ℝ) = (23 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l442_44286


namespace NUMINAMATH_CALUDE_overlook_distance_proof_l442_44283

/-- The distance to Mount Overlook in miles -/
def distance_to_overlook : ℝ := 12

/-- Jeannie's hiking speed to Mount Overlook in miles per hour -/
def speed_to_overlook : ℝ := 4

/-- Jeannie's hiking speed from Mount Overlook in miles per hour -/
def speed_from_overlook : ℝ := 6

/-- Total time of the hike in hours -/
def total_time : ℝ := 5

theorem overlook_distance_proof :
  distance_to_overlook = 12 ∧
  (distance_to_overlook / speed_to_overlook + distance_to_overlook / speed_from_overlook = total_time) :=
by sorry

end NUMINAMATH_CALUDE_overlook_distance_proof_l442_44283


namespace NUMINAMATH_CALUDE_gcd_lcm_multiple_relationship_l442_44204

theorem gcd_lcm_multiple_relationship (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 6) :
  Nat.gcd a b = b ∧ Nat.lcm a b = a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_multiple_relationship_l442_44204


namespace NUMINAMATH_CALUDE_final_orchid_count_l442_44252

/-- The number of orchids in a vase after adding more -/
def orchids_in_vase (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

theorem final_orchid_count : orchids_in_vase 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_final_orchid_count_l442_44252


namespace NUMINAMATH_CALUDE_fraction_subtraction_l442_44237

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l442_44237


namespace NUMINAMATH_CALUDE_matrix_solution_l442_44265

def determinant (a : ℝ) (x : ℝ) : ℝ :=
  (2*x + a) * ((x + a)^2 - x^2) - x * (x*(x + a) - x^2) + x * (x^2 - x*(x + a))

theorem matrix_solution (a : ℝ) (ha : a ≠ 0) :
  {x : ℝ | determinant a x = 0} = {-a/2, a/Real.sqrt 2, -a/Real.sqrt 2} :=
sorry

end NUMINAMATH_CALUDE_matrix_solution_l442_44265


namespace NUMINAMATH_CALUDE_x_plus_y_equals_1003_l442_44259

theorem x_plus_y_equals_1003 
  (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 1003 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_1003_l442_44259


namespace NUMINAMATH_CALUDE_projection_matrix_values_l442_44222

-- Define the matrix P
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49],
    ![c, 29/49]]

-- Define the property of being a projection matrix
def is_projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

-- Theorem statement
theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (P a c) → a = 41/49 ∧ c = 204/1225 :=
by sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l442_44222


namespace NUMINAMATH_CALUDE_inequality_proof_l442_44250

theorem inequality_proof (a b : ℝ) : 
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l442_44250


namespace NUMINAMATH_CALUDE_sqrt_two_decomposition_l442_44288

theorem sqrt_two_decomposition :
  ∃ (a : ℤ) (b : ℝ), 
    (Real.sqrt 2 = a + b) ∧ 
    (0 ≤ b) ∧ 
    (b < 1) ∧ 
    (a = 1) ∧ 
    (1 / b = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_decomposition_l442_44288


namespace NUMINAMATH_CALUDE_sum_of_x_sixth_powers_l442_44219

theorem sum_of_x_sixth_powers (x : ℕ) (b : ℕ) :
  (x : ℝ) * (x : ℝ)^6 = (x : ℝ)^b → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_sixth_powers_l442_44219


namespace NUMINAMATH_CALUDE_original_price_from_profit_and_selling_price_l442_44258

/-- Given an article sold at a 10% profit with a selling price of 550, 
    the original price of the article is 500. -/
theorem original_price_from_profit_and_selling_price :
  ∀ (original_price selling_price : ℝ),
    selling_price = 550 →
    selling_price = original_price * 1.1 →
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_profit_and_selling_price_l442_44258


namespace NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_l442_44232

theorem least_number_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_l442_44232


namespace NUMINAMATH_CALUDE_ronald_laundry_proof_l442_44255

/-- The number of days between Tim's laundry sessions -/
def tim_laundry_interval : ℕ := 9

/-- The number of days until Ronald and Tim do laundry on the same day again -/
def next_common_laundry_day : ℕ := 18

/-- The number of days between Ronald's laundry sessions -/
def ronald_laundry_interval : ℕ := 3

theorem ronald_laundry_proof :
  (tim_laundry_interval ∣ next_common_laundry_day) ∧
  (ronald_laundry_interval ∣ next_common_laundry_day) ∧
  (∀ n : ℕ, n ∣ next_common_laundry_day → n ≤ ronald_laundry_interval ∨ ronald_laundry_interval < n) →
  ronald_laundry_interval = 3 :=
by sorry

end NUMINAMATH_CALUDE_ronald_laundry_proof_l442_44255


namespace NUMINAMATH_CALUDE_johnny_works_four_hours_on_third_job_l442_44225

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hours_job1 : ℕ
  rate_job1 : ℕ
  hours_job2 : ℕ
  rate_job2 : ℕ
  rate_job3 : ℕ
  days : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked on the third job each day --/
def hours_job3_per_day (w : WorkSchedule) : ℕ :=
  let daily_earnings_job12 := w.hours_job1 * w.rate_job1 + w.hours_job2 * w.rate_job2
  let total_earnings_job12 := daily_earnings_job12 * w.days
  let total_earnings_job3 := w.total_earnings - total_earnings_job12
  total_earnings_job3 / (w.rate_job3 * w.days)

/-- Theorem stating that given Johnny's work schedule, he works 4 hours on the third job each day --/
theorem johnny_works_four_hours_on_third_job (w : WorkSchedule)
  (h1 : w.hours_job1 = 3)
  (h2 : w.rate_job1 = 7)
  (h3 : w.hours_job2 = 2)
  (h4 : w.rate_job2 = 10)
  (h5 : w.rate_job3 = 12)
  (h6 : w.days = 5)
  (h7 : w.total_earnings = 445) :
  hours_job3_per_day w = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_works_four_hours_on_third_job_l442_44225


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l442_44289

theorem inequality_solution_implies_m_range (m : ℝ) : 
  (∀ x, (m - 1) * x > m - 1 ↔ x < 1) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l442_44289


namespace NUMINAMATH_CALUDE_unique_positive_number_sum_with_square_l442_44209

theorem unique_positive_number_sum_with_square : ∃! x : ℝ, x > 0 ∧ x^2 + x = 156 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_sum_with_square_l442_44209


namespace NUMINAMATH_CALUDE_roots_derivative_sum_negative_l442_44253

open Real

theorem roots_derivative_sum_negative (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
  (a * x₁ - log x₁ = 0) → (a * x₂ - log x₂ = 0) →
  (a - 1 / x₁) + (a - 1 / x₂) < 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_derivative_sum_negative_l442_44253


namespace NUMINAMATH_CALUDE_parabola_directrix_l442_44234

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (a : ℝ), a = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = a)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l442_44234


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l442_44205

def fruit_arrangements (total : ℕ) (apples oranges bananas : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas)

theorem cory_fruit_arrangements :
  fruit_arrangements 9 4 2 2 = 3780 :=
by sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l442_44205


namespace NUMINAMATH_CALUDE_annes_cats_weight_l442_44248

/-- The total weight of Anne's four cats -/
def total_weight (first_female_weight : ℝ) : ℝ :=
  let second_female_weight := 1.5 * first_female_weight
  let first_male_weight := 2 * first_female_weight
  let second_male_weight := first_female_weight + second_female_weight
  first_female_weight + second_female_weight + first_male_weight + second_male_weight

/-- Theorem stating that the total weight of Anne's four cats is 14 kilograms -/
theorem annes_cats_weight : total_weight 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_annes_cats_weight_l442_44248


namespace NUMINAMATH_CALUDE_product_357_sum_28_l442_44294

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 357 →
  (a : ℕ) + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_357_sum_28_l442_44294


namespace NUMINAMATH_CALUDE_computer_sticker_price_l442_44233

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
  (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
  sticker_price = 750 := by
sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l442_44233


namespace NUMINAMATH_CALUDE_shortest_chord_length_l442_44238

/-- The shortest chord length of the intersection between a line and a circle -/
theorem shortest_chord_length (m : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | 2 * m * x - y - 8 * m - 3 = 0}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6 * x + 12 * y + 20 = 0}
  ∃ (chord_length : ℝ), 
    chord_length = 2 * Real.sqrt 15 ∧ 
    ∀ (other_length : ℝ), 
      (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ 
        other_length = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) →
      other_length ≥ chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l442_44238


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l442_44297

/-- Shopping mall product purchase problem -/
theorem shopping_mall_problem 
  (cost_price_A cost_price_B : ℚ)
  (quantity_A quantity_B : ℕ)
  (selling_price_A selling_price_B : ℚ) :
  cost_price_A = cost_price_B - 2 →
  80 / cost_price_A = 100 / cost_price_B →
  quantity_A = 3 * quantity_B - 5 →
  quantity_A + quantity_B ≤ 95 →
  selling_price_A = 12 →
  selling_price_B = 15 →
  (selling_price_A - cost_price_A) * quantity_A + 
  (selling_price_B - cost_price_B) * quantity_B > 380 →
  (cost_price_A = 8 ∧ cost_price_B = 10) ∧
  (∀ n : ℕ, n ≤ quantity_B → n ≤ 25) ∧
  ((quantity_A = 67 ∧ quantity_B = 24) ∨ 
   (quantity_A = 70 ∧ quantity_B = 25)) := by
sorry


end NUMINAMATH_CALUDE_shopping_mall_problem_l442_44297


namespace NUMINAMATH_CALUDE_cosine_sum_upper_bound_l442_44264

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_upper_bound_l442_44264


namespace NUMINAMATH_CALUDE_nathans_earnings_186_l442_44291

/-- Calculates the total earnings from Nathan's harvest --/
def nathans_earnings (strawberry_plants : ℕ) (tomato_plants : ℕ) 
  (strawberries_per_plant : ℕ) (tomatoes_per_plant : ℕ) 
  (fruits_per_basket : ℕ) (strawberry_basket_price : ℕ) (tomato_basket_price : ℕ) : ℕ :=
  let total_strawberries := strawberry_plants * strawberries_per_plant
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let strawberry_baskets := total_strawberries / fruits_per_basket
  let tomato_baskets := total_tomatoes / fruits_per_basket
  let strawberry_earnings := strawberry_baskets * strawberry_basket_price
  let tomato_earnings := tomato_baskets * tomato_basket_price
  strawberry_earnings + tomato_earnings

/-- Theorem stating that Nathan's earnings from his harvest equal $186 --/
theorem nathans_earnings_186 :
  nathans_earnings 5 7 14 16 7 9 6 = 186 := by
  sorry

end NUMINAMATH_CALUDE_nathans_earnings_186_l442_44291


namespace NUMINAMATH_CALUDE_first_chapter_pages_l442_44216

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem stating that for a book with 93 total pages and 33 pages in the second chapter,
    the first chapter has 60 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 93 → b.chapter2_pages = 33 → pages_in_chapter1 b = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l442_44216


namespace NUMINAMATH_CALUDE_tangent_line_circle_l442_44202

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

/-- The circle equation -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 2 - a = 0

/-- The theorem statement -/
theorem tangent_line_circle (a : ℝ) :
  (∃ x y : ℝ, line_equation x y ∧ circle_equation x y a ∧
    ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a → (x = x' ∧ y = y')) →
  a = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l442_44202


namespace NUMINAMATH_CALUDE_some_number_value_l442_44263

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (x / 3) = 41 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l442_44263


namespace NUMINAMATH_CALUDE_laws_in_concept_l442_44201

/-- The probability that exactly M laws are included in the Concept -/
def prob_exactly_M (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of exactly M laws being included and the expected number of laws -/
theorem laws_in_concept (K N M : ℕ) (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : M ≤ K) :
  (prob_exactly_M K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws K N p = K * (1 - (1 - p)^N)) := by
  sorry

#check laws_in_concept

end NUMINAMATH_CALUDE_laws_in_concept_l442_44201


namespace NUMINAMATH_CALUDE_cubic_root_sum_l442_44268

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 158 / 73 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l442_44268


namespace NUMINAMATH_CALUDE_function_form_l442_44279

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating the form of the function satisfying the equation -/
theorem function_form (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_function_form_l442_44279


namespace NUMINAMATH_CALUDE_higher_speed_is_two_l442_44256

-- Define the runners
structure Runner :=
  (blocks : ℕ)
  (minutes : ℕ)

-- Define the speed calculation function
def speed (r : Runner) : ℚ :=
  r.blocks / r.minutes

-- Define Tiffany and Moses
def tiffany : Runner := ⟨6, 3⟩
def moses : Runner := ⟨12, 8⟩

-- Theorem: The higher average speed is 2 blocks per minute
theorem higher_speed_is_two :
  max (speed tiffany) (speed moses) = 2 := by
  sorry

end NUMINAMATH_CALUDE_higher_speed_is_two_l442_44256


namespace NUMINAMATH_CALUDE_area_of_region_l442_44242

/-- The area of the region defined by x^2 + y^2 + 8x - 18y = 0 is 97π -/
theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 97 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y - 9)^2)) ^ 2 ∧
   x^2 + y^2 + 8*x - 18*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l442_44242


namespace NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l442_44213

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (n : ℕ+), (3 * p + 25 : ℚ) / (2 * p - 5) = n) ↔ p ∈ ({3, 5, 9, 35} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l442_44213


namespace NUMINAMATH_CALUDE_number_added_before_division_l442_44249

theorem number_added_before_division (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) → 
  (∃ n : ℤ, ∃ m : ℤ, x + n = 41 * m + 18) → 
  (∃ n : ℤ, x + n ≡ 18 [ZMOD 41] ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_number_added_before_division_l442_44249


namespace NUMINAMATH_CALUDE_expression_evaluation_l442_44224

theorem expression_evaluation : 
  Real.sqrt ((16^10 + 8^10 + 2^30) / (16^4 + 8^11 + 2^20)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l442_44224


namespace NUMINAMATH_CALUDE_games_missed_l442_44240

/-- Given that Benny's high school played 39 baseball games and he attended 14 games,
    prove that the number of games Benny missed is 25. -/
theorem games_missed (total_games : ℕ) (games_attended : ℕ) (h1 : total_games = 39) (h2 : games_attended = 14) :
  total_games - games_attended = 25 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l442_44240


namespace NUMINAMATH_CALUDE_digit_B_is_three_l442_44267

/-- Represents a digit from 1 to 7 -/
def Digit := Fin 7

/-- Represents the set of points A, B, C, D, E, F -/
structure Points where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
             B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
             C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
             D ≠ E ∧ D ≠ F ∧
             E ≠ F

/-- The sum of digits along each line -/
def lineSums (p : Points) : ℕ :=
  (p.A.val + p.B.val + p.C.val + 1) +
  (p.A.val + p.E.val + p.F.val + 1) +
  (p.C.val + p.D.val + p.E.val + 1) +
  (p.B.val + p.D.val + 1) +
  (p.B.val + p.F.val + 1)

theorem digit_B_is_three (p : Points) (h : lineSums p = 51) : p.B.val + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_B_is_three_l442_44267


namespace NUMINAMATH_CALUDE_cos_two_beta_equals_one_l442_44292

theorem cos_two_beta_equals_one (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 0) : 
  Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_beta_equals_one_l442_44292


namespace NUMINAMATH_CALUDE_sin_2x_value_l442_44228

theorem sin_2x_value (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : 
  Real.sin (2*x) = -7/25 := by sorry

end NUMINAMATH_CALUDE_sin_2x_value_l442_44228


namespace NUMINAMATH_CALUDE_original_selling_price_l442_44284

/-- The original selling price given the profit rates and price difference -/
theorem original_selling_price 
  (original_profit_rate : ℝ)
  (reduced_purchase_rate : ℝ)
  (new_profit_rate : ℝ)
  (price_difference : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : reduced_purchase_rate = 0.1)
  (h3 : new_profit_rate = 0.3)
  (h4 : price_difference = 49) :
  ∃ (purchase_price : ℝ),
    (1 + original_profit_rate) * purchase_price = 770 ∧
    ((1 - reduced_purchase_rate) * (1 + new_profit_rate) - (1 + original_profit_rate)) * purchase_price = price_difference :=
by sorry

end NUMINAMATH_CALUDE_original_selling_price_l442_44284


namespace NUMINAMATH_CALUDE_circle_extrema_l442_44295

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 + 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 - 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≤ Real.sqrt 10 + Real.sqrt 6) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 - Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≥ Real.sqrt 10 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_extrema_l442_44295


namespace NUMINAMATH_CALUDE_skate_cost_is_65_l442_44298

/-- The cost of renting skates for one visit -/
def rental_cost : ℚ := 2.5

/-- The number of visits needed to justify buying skates -/
def visits : ℕ := 26

/-- The cost of a new pair of skates -/
def skate_cost : ℚ := rental_cost * visits

/-- Theorem stating that the cost of a new pair of skates is $65 -/
theorem skate_cost_is_65 : skate_cost = 65 := by sorry

end NUMINAMATH_CALUDE_skate_cost_is_65_l442_44298


namespace NUMINAMATH_CALUDE_initial_girls_count_l442_44200

theorem initial_girls_count (total : ℕ) : 
  (total ≠ 0) →
  (total / 2 : ℚ) = (total / 2 : ℕ) →
  ((total / 2 : ℕ) - 5 : ℚ) / total = 2 / 5 →
  total / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l442_44200


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_15_l442_44261

theorem binomial_coefficient_21_15 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 20 14 = 38760)
  (h3 : Nat.choose 22 15 = 203490) :
  Nat.choose 21 15 = 87210 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_15_l442_44261


namespace NUMINAMATH_CALUDE_f_pi_8_equals_sqrt_2_l442_44272

noncomputable def f (x : ℝ) : ℝ := 
  1 / (2 * Real.tan x) + (Real.sin (x/2) * Real.cos (x/2)) / (2 * Real.cos (x/2)^2 - 1)

theorem f_pi_8_equals_sqrt_2 : f (π/8) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_8_equals_sqrt_2_l442_44272


namespace NUMINAMATH_CALUDE_remainder_theorem_l442_44260

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = q * (3^151 + 3^75 + 1) + 294 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l442_44260


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l442_44271

/-- Given a right triangle, if rotating it about one leg produces a cone of volume 972π cm³
    and rotating it about the other leg produces a cone of volume 1458π cm³,
    then the length of the hypotenuse is 12√5 cm. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/3) * π * a * b^2 = 972 * π →
  (1/3) * π * b * a^2 = 1458 * π →
  c = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l442_44271


namespace NUMINAMATH_CALUDE_or_implies_and_implies_not_equivalent_l442_44254

theorem or_implies_and_implies_not_equivalent :
  ¬(∀ (A B C : Prop), ((A ∨ B) → C) ↔ ((A ∧ B) → C)) := by
sorry

end NUMINAMATH_CALUDE_or_implies_and_implies_not_equivalent_l442_44254


namespace NUMINAMATH_CALUDE_f_equality_iff_a_half_l442_44243

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (4 : ℝ) ^ x else (2 : ℝ) ^ (a - x)

theorem f_equality_iff_a_half (a : ℝ) (h : a ≠ 1) :
  f a (1 - a) = f a (a - 1) ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_equality_iff_a_half_l442_44243


namespace NUMINAMATH_CALUDE_mentor_fraction_l442_44281

theorem mentor_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  n = 2 * s / 3 → (n / 2 + s / 3) / (n + s) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mentor_fraction_l442_44281


namespace NUMINAMATH_CALUDE_trees_on_rectangular_plot_l442_44290

/-- The number of trees planted on a rectangular plot -/
def num_trees (length width spacing : ℕ) : ℕ :=
  ((length / spacing) + 1) * ((width / spacing) + 1)

/-- Theorem: The number of trees planted at a five-foot distance from each other
    on a rectangular plot of land with sides 120 feet and 70 feet is 375 -/
theorem trees_on_rectangular_plot :
  num_trees 120 70 5 = 375 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_rectangular_plot_l442_44290


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_is_one_l442_44274

theorem zeroth_power_of_nonzero_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_is_one_l442_44274


namespace NUMINAMATH_CALUDE_initial_tax_rate_proof_l442_44227

def annual_income : ℝ := 48000
def new_tax_rate : ℝ := 30
def tax_savings : ℝ := 7200

theorem initial_tax_rate_proof :
  ∃ (initial_rate : ℝ),
    initial_rate > 0 ∧
    initial_rate < 100 ∧
    (initial_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = tax_savings ∧
    initial_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_tax_rate_proof_l442_44227


namespace NUMINAMATH_CALUDE_no_solution_exists_l442_44244

theorem no_solution_exists : ¬∃ x : ℝ, (|x^2 - 14*x + 40| = 3) ∧ (x^2 - 14*x + 45 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l442_44244


namespace NUMINAMATH_CALUDE_triangle_area_l442_44299

theorem triangle_area (a b c : ℝ) (α : ℝ) (h1 : a = 14)
  (h2 : α = Real.pi / 3) (h3 : b / c = 8 / 5) :
  (1 / 2) * b * c * Real.sin α = 40 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l442_44299


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l442_44282

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {y | ∃ a ∈ P, y = 2*a - 1}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l442_44282


namespace NUMINAMATH_CALUDE_highway_on_map_l442_44239

/-- Represents the scale of a map as a ratio -/
structure MapScale where
  numerator : ℕ
  denominator : ℕ

/-- Converts kilometers to centimeters -/
def km_to_cm (km : ℕ) : ℕ := km * 100000

/-- Calculates the length on a map given the actual length and map scale -/
def length_on_map (actual_length_km : ℕ) (scale : MapScale) : ℕ :=
  (km_to_cm actual_length_km) * scale.numerator / scale.denominator

/-- Theorem stating that a 155 km highway on a 1:500000 scale map is 31 cm long -/
theorem highway_on_map :
  let actual_length_km : ℕ := 155
  let scale : MapScale := ⟨1, 500000⟩
  length_on_map actual_length_km scale = 31 := by sorry

end NUMINAMATH_CALUDE_highway_on_map_l442_44239


namespace NUMINAMATH_CALUDE_fairCoinDifference_l442_44208

def fairCoinProbability : ℚ := 1 / 2

def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def probabilityThreeHeads : ℚ :=
  binomialProbability 4 3 fairCoinProbability

def probabilityFourHeads : ℚ :=
  fairCoinProbability^4

theorem fairCoinDifference :
  probabilityThreeHeads - probabilityFourHeads = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fairCoinDifference_l442_44208


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l442_44296

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l442_44296


namespace NUMINAMATH_CALUDE_no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l442_44270

/-- Defines a function that checks if a number is divisible by any prime in a given range -/
def divisible_by_prime_in_range (n : ℕ) (lower upper : ℕ) : Prop :=
  ∃ p, Prime p ∧ lower ≤ p ∧ p ≤ upper ∧ p ∣ n

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 11 -/
theorem no_fourteen_consecutive_integers : ¬ ∃ start : ℕ, ∀ k : ℕ, k < 14 →
  divisible_by_prime_in_range (start + k) 2 11 := by sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 13 -/
theorem exist_twentyone_consecutive_integers : ∃ start : ℕ, ∀ k : ℕ, k < 21 →
  divisible_by_prime_in_range (start + k) 2 13 := by sorry

end NUMINAMATH_CALUDE_no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l442_44270


namespace NUMINAMATH_CALUDE_jeremy_age_l442_44217

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) :
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l442_44217


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l442_44215

theorem concentric_circles_ratio 
  (r R : ℝ) 
  (a b c : ℝ) 
  (h_positive : 0 < r ∧ 0 < R ∧ 0 < a ∧ 0 < b ∧ 0 < c)
  (h_r_less_R : r < R)
  (h_area_ratio : (π * R^2 - π * r^2) / (π * R^2) = a / (b + c)) :
  R / r = Real.sqrt a / Real.sqrt (b + c - a) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l442_44215


namespace NUMINAMATH_CALUDE_specific_pentagon_perimeter_l442_44266

/-- Pentagon ABCDE with specific side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- Theorem: The perimeter of the specific pentagon is 12 -/
theorem specific_pentagon_perimeter :
  ∃ (p : Pentagon),
    p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 2 ∧ p.DE = 2 ∧
    p.AE ^ 2 = (p.AB + p.BC) ^ 2 + (p.CD + p.DE) ^ 2 ∧
    perimeter p = 12 := by
  sorry


end NUMINAMATH_CALUDE_specific_pentagon_perimeter_l442_44266


namespace NUMINAMATH_CALUDE_u_value_l442_44247

/-- A line passing through points (2, 8), (4, 14), (6, 20), and (18, u) -/
structure Line where
  -- Define the slope of the line
  slope : ℝ
  -- Define the y-intercept of the line
  intercept : ℝ
  -- Ensure the line passes through (2, 8)
  point1 : 8 = slope * 2 + intercept
  -- Ensure the line passes through (4, 14)
  point2 : 14 = slope * 4 + intercept
  -- Ensure the line passes through (6, 20)
  point3 : 20 = slope * 6 + intercept

/-- The u-coordinate of the point (18, u) on the line -/
def u (l : Line) : ℝ := l.slope * 18 + l.intercept

/-- Theorem stating that u = 56 for the given line -/
theorem u_value (l : Line) : u l = 56 := by
  sorry

end NUMINAMATH_CALUDE_u_value_l442_44247


namespace NUMINAMATH_CALUDE_triangle_side_equations_l442_44275

theorem triangle_side_equations (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬(∃ x y z : ℝ, x^2 - 2*b*x + 2*a*c = 0 ∧ y^2 - 2*c*y + 2*a*b = 0 ∧ z^2 - 2*a*z + 2*b*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l442_44275


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l442_44218

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), (228712 + m) % 9 = 0 → m ≥ n) ∧
  (228712 + n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l442_44218


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_3_l442_44231

theorem complex_exp_13pi_over_3 : Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_3_l442_44231


namespace NUMINAMATH_CALUDE_intersection_M_N_l442_44262

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | x^2 - 25 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l442_44262


namespace NUMINAMATH_CALUDE_gcd_7584_18027_l442_44293

theorem gcd_7584_18027 : Nat.gcd 7584 18027 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7584_18027_l442_44293


namespace NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_proof_l442_44285

theorem rectangle_area_difference : ℕ → Prop :=
  fun d => ∀ l w : ℕ,
    (l + w = 30) →  -- Perimeter condition: 2l + 2w = 60 simplified
    (∃ l' w' : ℕ, l' + w' = 30 ∧ l' * w' = l * w + d) →  -- Larger area exists
    (∀ l'' w'' : ℕ, l'' + w'' = 30 → l'' * w'' ≤ l * w + d) →  -- No larger area exists
    d = 196

-- The proof goes here
theorem rectangle_area_difference_proof : rectangle_area_difference 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_proof_l442_44285


namespace NUMINAMATH_CALUDE_roots_sum_absolute_value_l442_44276

theorem roots_sum_absolute_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + x + m = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    |x₁| + |x₂| = 3) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_absolute_value_l442_44276


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l442_44229

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given dimensions satisfy the problem conditions -/
def satisfiesConditions (dim : RectDimensions) : Prop :=
  dim.length + dim.width = 11 ∧
  (dim.length = 5 ∧ dim.width = 6) ∨
  (dim.length = 8 ∧ dim.width = 3) ∨
  (dim.length = 4 ∧ dim.width = 7)

theorem rectangle_dimensions :
  ∀ (dim : RectDimensions),
    (2 * (dim.length + dim.width) = 22) →
    (∃ (subRect : RectDimensions),
      subRect.length = 2 ∧ subRect.width = 6 ∧
      subRect.length ≤ dim.length ∧ subRect.width ≤ dim.width) →
    satisfiesConditions dim :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l442_44229


namespace NUMINAMATH_CALUDE_austonHeightCm_l442_44280

/-- Converts inches to centimeters -/
def inchesToCm (inches : ℝ) : ℝ := inches * 2.54

/-- Auston's height in inches -/
def austonHeightInches : ℝ := 60

/-- Theorem stating Auston's height in centimeters -/
theorem austonHeightCm : inchesToCm austonHeightInches = 152.4 := by
  sorry

end NUMINAMATH_CALUDE_austonHeightCm_l442_44280


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l442_44210

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | |x - 2| ≥ 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l442_44210


namespace NUMINAMATH_CALUDE_linear_function_point_l442_44226

/-- Given a linear function y = x - 1 that passes through the point (m, 2), prove that m = 3 -/
theorem linear_function_point (m : ℝ) : (2 : ℝ) = m - 1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_l442_44226


namespace NUMINAMATH_CALUDE_expansion_equality_l442_44223

theorem expansion_equality (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l442_44223


namespace NUMINAMATH_CALUDE_shopper_receive_amount_l442_44246

/-- The amount of money each person has and donates --/
def problem (isabella sam giselle valentina ethan : ℚ) : Prop :=
  isabella = giselle + 15 ∧
  isabella = sam + 45 ∧
  giselle = 120 ∧
  valentina = 2 * sam ∧
  ethan = isabella - 75

/-- The total donation amount --/
def total_donation (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  0.2 * isabella + 0.15 * sam + 0.1 * giselle + 0.25 * valentina + 0.3 * ethan

/-- The amount each shopper receives after equal distribution --/
def shopper_receive (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  (total_donation isabella sam giselle valentina ethan) / 4

/-- Theorem stating the amount each shopper receives --/
theorem shopper_receive_amount :
  ∀ isabella sam giselle valentina ethan,
  problem isabella sam giselle valentina ethan →
  shopper_receive isabella sam giselle valentina ethan = 28.875 :=
by sorry

end NUMINAMATH_CALUDE_shopper_receive_amount_l442_44246


namespace NUMINAMATH_CALUDE_algebraic_expressions_l442_44245

variable (a x : ℝ)

theorem algebraic_expressions :
  ((-3 * a^2)^3 - 4 * a^2 * a^4 + 5 * a^9 / a^3 = -26 * a^6) ∧
  (((x + 1) * (x + 2) + 2 * (x - 1)) / x = x + 5) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expressions_l442_44245


namespace NUMINAMATH_CALUDE_clock_hands_coincidence_time_l442_44241

/-- Represents the state of a clock's hands -/
structure ClockState where
  minute_angle : ℝ
  hour_angle : ℝ

/-- Represents the movement rates of clock hands -/
structure ClockRates where
  minute_rate : ℝ
  hour_rate : ℝ

/-- Calculates the time taken for clock hands to move from one state to another -/
def time_between_states (initial : ClockState) (final : ClockState) (rates : ClockRates) : ℝ :=
  sorry

theorem clock_hands_coincidence_time :
  let initial_state : ClockState := { minute_angle := 0, hour_angle := 180 }
  let final_state : ClockState := { minute_angle := 0, hour_angle := 0 }
  let rates : ClockRates := { minute_rate := 6, hour_rate := 0.5 }
  let time := time_between_states initial_state final_state rates
  time = 360 ∧ time < 12 * 60 := by sorry

end NUMINAMATH_CALUDE_clock_hands_coincidence_time_l442_44241


namespace NUMINAMATH_CALUDE_fraction_of_product_l442_44277

theorem fraction_of_product (total : ℝ) (result : ℝ) : 
  total = 5020 →
  (3/4 : ℝ) * (1/2 : ℝ) * total = (3/4 : ℝ) * (1/2 : ℝ) * 5020 →
  result = 753.0000000000001 →
  (result / ((3/4 : ℝ) * (1/2 : ℝ) * total) : ℝ) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_product_l442_44277


namespace NUMINAMATH_CALUDE_parabola_equation_l442_44221

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line x - y + 2 = 0
def focus_line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0
  -- Axis of symmetry is a coordinate axis
  ∧ (∀ x y : ℝ, p.equation x y → p.equation x (-y) ∨ p.equation (-x) y)
  -- Focus on the line x - y + 2 = 0
  ∧ ∃ fx fy : ℝ, focus_line fx fy ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x + fx)^2 + (y + fy)^2)
    ∨ (∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x - fx)^2 + (y + fy)^2))

-- Theorem statement
theorem parabola_equation (p : Parabola) (h : parabola_conditions p) :
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -8*x) ∨ (∀ x y : ℝ, p.equation x y ↔ x^2 = 8*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l442_44221


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l442_44236

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3 * (i + 1)) / (i - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l442_44236


namespace NUMINAMATH_CALUDE_rectangle_area_l442_44211

/-- The area of a rectangle with perimeter 40 and length twice its width -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l442_44211


namespace NUMINAMATH_CALUDE_min_value_greater_than_five_l442_44212

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |x + a - 1| + (a + 1)^2

-- State the theorem
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f x a > 5) ↔ a < (-1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_greater_than_five_l442_44212


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l442_44214

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defective units that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of Type B defective units that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The total percentage of defective units (Type A or B) that are shipped for sale -/
def total_defective_shipped_rate : ℝ :=
  type_a_defect_rate * type_a_ship_rate + type_b_defect_rate * type_b_ship_rate

theorem defective_shipped_percentage :
  total_defective_shipped_rate = 0.0069 := by sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l442_44214


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l442_44203

theorem compare_negative_fractions : -2/3 > -5/7 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l442_44203


namespace NUMINAMATH_CALUDE_function_value_given_cube_l442_44206

theorem function_value_given_cube (x : ℝ) (h : x^3 = 8) :
  (x - 1) * (x + 1) * (x^2 + x + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_function_value_given_cube_l442_44206


namespace NUMINAMATH_CALUDE_centroid_tetrahedron_volume_ratio_l442_44278

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInterior (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Main theorem: volume ratio of centroids' tetrahedron to original tetrahedron -/
theorem centroid_tetrahedron_volume_ratio 
  (ABCD : Tetrahedron) (P : Point3D) 
  (h : isInterior P ABCD) : 
  let G1 := centroid ⟨P, ABCD.A, ABCD.B, ABCD.C⟩
  let G2 := centroid ⟨P, ABCD.B, ABCD.C, ABCD.D⟩
  let G3 := centroid ⟨P, ABCD.C, ABCD.D, ABCD.A⟩
  let G4 := centroid ⟨P, ABCD.D, ABCD.A, ABCD.B⟩
  volume ⟨G1, G2, G3, G4⟩ / volume ABCD = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_centroid_tetrahedron_volume_ratio_l442_44278


namespace NUMINAMATH_CALUDE_triangle_inequality_l442_44287

/-- Prove that for positive integers x, y, z and angles α, β, γ in [0, π) where any two angles 
    sum to more than the third, the following inequality holds:
    √(x²+y²-2xy cos α) + √(y²+z²-2yz cos β) ≥ √(z²+x²-2zx cos γ) -/
theorem triangle_inequality (x y z : ℕ+) (α β γ : ℝ)
  (h_α : 0 ≤ α ∧ α < π)
  (h_β : 0 ≤ β ∧ β < π)
  (h_γ : 0 ≤ γ ∧ γ < π)
  (h_sum1 : α + β > γ)
  (h_sum2 : β + γ > α)
  (h_sum3 : γ + α > β) :
  Real.sqrt (x.val^2 + y.val^2 - 2*x.val*y.val*(Real.cos α)) + 
  Real.sqrt (y.val^2 + z.val^2 - 2*y.val*z.val*(Real.cos β)) ≥
  Real.sqrt (z.val^2 + x.val^2 - 2*z.val*x.val*(Real.cos γ)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l442_44287


namespace NUMINAMATH_CALUDE_micah_ate_six_strawberries_l442_44269

/-- The number of strawberries in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of strawberries Micah picked -/
def dozens_picked : ℕ := 2

/-- The number of strawberries Micah saved for his mom -/
def saved_for_mom : ℕ := 18

/-- The total number of strawberries Micah picked -/
def total_picked : ℕ := dozens_picked * dozen

/-- The number of strawberries Micah ate -/
def eaten_by_micah : ℕ := total_picked - saved_for_mom

theorem micah_ate_six_strawberries : eaten_by_micah = 6 := by
  sorry

end NUMINAMATH_CALUDE_micah_ate_six_strawberries_l442_44269


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l442_44207

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a : Line) (α β : Plane)
  (h1 : planeParallel α β)
  (h2 : lineInPlane a α) :
  lineParallelPlane a β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l442_44207


namespace NUMINAMATH_CALUDE_sqrt_x_plus_5_real_l442_44273

theorem sqrt_x_plus_5_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 5) ↔ x ≥ -5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_5_real_l442_44273
