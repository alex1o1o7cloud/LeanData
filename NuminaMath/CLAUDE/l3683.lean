import Mathlib

namespace NUMINAMATH_CALUDE_empty_boxes_count_l3683_368330

/-- The number of empty boxes after n operations, where n is the number of non-empty boxes. -/
def empty_boxes (n : ℕ) : ℤ :=
  -1 + 6 * n

/-- The theorem stating that when there are 34 non-empty boxes, there are 203 empty boxes. -/
theorem empty_boxes_count : empty_boxes 34 = 203 := by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l3683_368330


namespace NUMINAMATH_CALUDE_expenditure_difference_l3683_368368

theorem expenditure_difference
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percent : ℝ)
  (quantity_purchased_percent : ℝ)
  (h1 : price_increase_percent = 25)
  (h2 : quantity_purchased_percent = 70)
  : abs (original_price * original_quantity * (1 + price_increase_percent / 100) * (quantity_purchased_percent / 100) - original_price * original_quantity) / (original_price * original_quantity) = 0.125 := by
sorry

end NUMINAMATH_CALUDE_expenditure_difference_l3683_368368


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3683_368337

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 0, a}
  let B : Set ℝ := {0, Real.sqrt a}
  B ⊆ A → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3683_368337


namespace NUMINAMATH_CALUDE_g_one_value_l3683_368389

-- Define the polynomial f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the conditions
structure Conditions (a b c : ℝ) : Prop :=
  (a_lt_b : a < b)
  (b_lt_c : b < c)
  (one_lt_a : 1 < a)

-- Define the theorem
theorem g_one_value (a b c : ℝ) (h : Conditions a b c) :
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 ↔ ∃ y, f a b c y = 0 ∧ x = 1 / y) →
    (∃ k, k ≠ 0 ∧ ∀ x, g x = k * (x^3 + (c/k)*x^2 + (b/k)*x + a/k)) →
    g 1 = (1 + a + b + c) / c :=
sorry

end NUMINAMATH_CALUDE_g_one_value_l3683_368389


namespace NUMINAMATH_CALUDE_unique_solution_x4_y2_71_l3683_368382

theorem unique_solution_x4_y2_71 :
  ∀ x y : ℕ+, x^4 = y^2 + 71 → x = 6 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x4_y2_71_l3683_368382


namespace NUMINAMATH_CALUDE_summer_performs_1300_salutations_l3683_368388

/-- The number of sun salutations Summer performs throughout an entire year. -/
def summer_sun_salutations : ℕ :=
  let poses_per_day : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year. -/
theorem summer_performs_1300_salutations : summer_sun_salutations = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_performs_1300_salutations_l3683_368388


namespace NUMINAMATH_CALUDE_problem_equivalence_l3683_368303

theorem problem_equivalence :
  (3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2) ∧
  (|Real.sqrt 3 - Real.sqrt 2| + |Real.sqrt 3 - 2| + Real.sqrt ((-2)^2) = 4 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalence_l3683_368303


namespace NUMINAMATH_CALUDE_man_to_son_age_ratio_l3683_368392

def son_age : ℕ := 20
def age_difference : ℕ := 22

def man_age : ℕ := son_age + age_difference

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem man_to_son_age_ratio :
  man_age_in_two_years / son_age_in_two_years = 2 ∧
  man_age_in_two_years % son_age_in_two_years = 0 := by
  sorry

#eval man_age_in_two_years / son_age_in_two_years

end NUMINAMATH_CALUDE_man_to_son_age_ratio_l3683_368392


namespace NUMINAMATH_CALUDE_count_non_consecutive_digits_999999_l3683_368346

/-- Counts integers from 0 to n without consecutive identical digits -/
def countNonConsecutiveDigits (n : ℕ) : ℕ :=
  sorry

/-- The sum of geometric series 9^1 + 9^2 + ... + 9^6 -/
def geometricSum : ℕ :=
  sorry

theorem count_non_consecutive_digits_999999 :
  countNonConsecutiveDigits 999999 = 597880 := by
  sorry

end NUMINAMATH_CALUDE_count_non_consecutive_digits_999999_l3683_368346


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3683_368326

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3683_368326


namespace NUMINAMATH_CALUDE_small_cakes_per_hour_l3683_368364

-- Define the variables
def helpers : ℕ := 10
def hours : ℕ := 3
def large_cakes_needed : ℕ := 20
def small_cakes_needed : ℕ := 700
def large_cakes_per_hour : ℕ := 2

-- Define the theorem
theorem small_cakes_per_hour :
  ∃ (s : ℕ), 
    s * helpers * (hours - (large_cakes_needed / large_cakes_per_hour)) = small_cakes_needed ∧
    s = 35 := by
  sorry

end NUMINAMATH_CALUDE_small_cakes_per_hour_l3683_368364


namespace NUMINAMATH_CALUDE_gcd_of_180_270_450_l3683_368365

theorem gcd_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_270_450_l3683_368365


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3683_368323

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = ((n - 2) * 180 : ℝ) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3683_368323


namespace NUMINAMATH_CALUDE_harmonic_subsequence_existence_l3683_368354

/-- The harmonic sequence -/
def harmonic_seq : ℕ → ℚ
  | n => 1 / n

/-- A subsequence of the harmonic sequence -/
def subseq (f : ℕ → ℕ) : ℕ → ℚ := λ n => harmonic_seq (f n)

/-- The property that each term, starting from the third, is the difference of the two preceding terms -/
def has_difference_property (s : ℕ → ℚ) : Prop :=
  ∀ n ≥ 3, s n = s (n - 1) - s (n - 2)

theorem harmonic_subsequence_existence :
  ∃ f : ℕ → ℕ, (∀ n m, n < m → f n < f m) ∧ 
              has_difference_property (subseq f) ∧
              (∃ N : ℕ, N ≥ 100) :=
sorry

end NUMINAMATH_CALUDE_harmonic_subsequence_existence_l3683_368354


namespace NUMINAMATH_CALUDE_equilateral_triangle_complex_plane_l3683_368391

theorem equilateral_triangle_complex_plane (z : ℂ) (μ : ℝ) : 
  Complex.abs z = 3 →
  μ > 2 →
  (Complex.abs (z^3 - z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (z^3 - z)) →
  μ = 1 + Real.sqrt 82 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_complex_plane_l3683_368391


namespace NUMINAMATH_CALUDE_quadratic_points_property_l3683_368381

/-- Represents a quadratic function y = ax² - 4ax + c, where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h_a_pos : a > 0

/-- Represents the y-coordinates of the four points on the quadratic function -/
structure FourPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ

/-- 
  Theorem: For a quadratic function y = ax² - 4ax + c (a > 0) passing through points 
  A(-2, y₁), B(0, y₂), C(3, y₃), D(5, y₄), if y₂y₄ < 0, then y₁y₃ < 0.
-/
theorem quadratic_points_property (f : QuadraticFunction) (p : FourPoints) :
  (p.y₂ * p.y₄ < 0) → (p.y₁ * p.y₃ < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_property_l3683_368381


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3683_368376

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  let asymptote_slope := b / a
  (e = 2 * asymptote_slope) → e = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3683_368376


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3683_368342

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3683_368342


namespace NUMINAMATH_CALUDE_coins_missing_l3683_368358

theorem coins_missing (total : ℚ) : 
  let lost := (1 : ℚ) / 3 * total
  let found := (3 : ℚ) / 4 * lost
  let remaining := total - lost + found
  total - remaining = (1 : ℚ) / 12 * total := by
sorry

end NUMINAMATH_CALUDE_coins_missing_l3683_368358


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3683_368383

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x + (5/8) * a - (3/2)

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a x = 2) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3683_368383


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3683_368315

theorem arithmetic_square_root_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3683_368315


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3683_368357

theorem base_conversion_problem (a b c : ℕ) (h1 : a ≤ 6) (h2 : b ≤ 6) (h3 : c ≤ 6) 
  (h4 : a ≤ 8) (h5 : b ≤ 8) (h6 : c ≤ 8) :
  (49 * a + 7 * b + c = 81 * c + 9 * b + a) → (49 * a + 7 * b + c = 248) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3683_368357


namespace NUMINAMATH_CALUDE_divisibility_by_240_l3683_368373

theorem divisibility_by_240 (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) : 
  240 ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l3683_368373


namespace NUMINAMATH_CALUDE_zero_not_in_2_16_l3683_368375

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having only one zero
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a zero being within an interval
def zero_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_not_in_2_16 (h1 : has_unique_zero f)
  (h2 : zero_in_interval f 0 16)
  (h3 : zero_in_interval f 0 8)
  (h4 : zero_in_interval f 0 4)
  (h5 : zero_in_interval f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_2_16_l3683_368375


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3683_368386

theorem min_value_of_sum_of_squares (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3683_368386


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3683_368344

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3683_368344


namespace NUMINAMATH_CALUDE_min_sum_mutually_exclusive_events_l3683_368363

theorem min_sum_mutually_exclusive_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hA : ℝ) (hB : ℝ) (h_mutually_exclusive : hA + hB = 1) 
  (h_prob_A : hA = 1 / y) (h_prob_B : hB = 4 / x) : 
  x + y ≥ 9 ∧ ∃ x y, x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_min_sum_mutually_exclusive_events_l3683_368363


namespace NUMINAMATH_CALUDE_initial_number_of_men_l3683_368384

/-- Given a group of men where replacing two men (aged 20 and 22) with two women (average age 29)
    increases the average age by 2 years, prove that the initial number of men is 8. -/
theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * 29 - (20 + 22) : ℝ) = 2 * M → M = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_men_l3683_368384


namespace NUMINAMATH_CALUDE_surface_area_of_cube_with_holes_l3683_368320

/-- Represents a cube with square holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the surface area of a cube with square holes cut through each face -/
def surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let inner_surface_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + inner_surface_area

/-- Theorem stating that the surface area of the specified cube with holes is 168 square meters -/
theorem surface_area_of_cube_with_holes :
  let cube := CubeWithHoles.mk 4 2
  surface_area cube = 168 := by
  sorry


end NUMINAMATH_CALUDE_surface_area_of_cube_with_holes_l3683_368320


namespace NUMINAMATH_CALUDE_mildred_total_oranges_l3683_368302

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := 77

/-- The number of oranges Mildred's father gave her -/
def additional_oranges : ℕ := 2

/-- Theorem: Mildred's total number of oranges is 79 -/
theorem mildred_total_oranges : 
  initial_oranges + additional_oranges = 79 := by
  sorry

end NUMINAMATH_CALUDE_mildred_total_oranges_l3683_368302


namespace NUMINAMATH_CALUDE_toy_car_spending_l3683_368327

theorem toy_car_spending
  (A B C D E F G H : ℝ)
  (last_month : ℝ := A + B + C + D + E)
  (this_month_new : ℝ := F + G + H)
  (discount : ℝ := 0.2)
  (total_before_discount : ℝ := 2 * last_month + this_month_new)
  (total_after_discount : ℝ := (1 - discount) * total_before_discount) :
  total_after_discount = 1.6 * A + 1.6 * B + 1.6 * C + 1.6 * D + 1.6 * E + 0.8 * F + 0.8 * G + 0.8 * H :=
by sorry

end NUMINAMATH_CALUDE_toy_car_spending_l3683_368327


namespace NUMINAMATH_CALUDE_divisibility_condition_l3683_368304

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3683_368304


namespace NUMINAMATH_CALUDE_animal_shelter_problem_l3683_368314

/-- Represents the animal shelter problem --/
theorem animal_shelter_problem
  (initial_dogs : ℕ) (initial_cats : ℕ) (new_pets : ℕ) (total_after_month : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (h1 : initial_dogs = 30)
  (h2 : initial_cats = 28)
  (h3 : new_pets = 13)
  (h4 : total_after_month = 65)
  (h5 : dog_adoption_rate = 1/2)
  (h6 : cat_adoption_rate = 1/4)
  (h7 : lizard_adoption_rate = 1/5) :
  ∃ (initial_lizards : ℕ),
    initial_lizards = 20 ∧
    (↑initial_dogs * (1 - dog_adoption_rate) +
     ↑initial_cats * (1 - cat_adoption_rate) +
     ↑initial_lizards * (1 - lizard_adoption_rate) +
     ↑new_pets : ℚ) = total_after_month :=
by sorry

end NUMINAMATH_CALUDE_animal_shelter_problem_l3683_368314


namespace NUMINAMATH_CALUDE_polynomial_root_l3683_368380

/-- Given a polynomial g(x) = 3x^4 - 2x^3 + x^2 + 4x + s, 
    prove that s = -2 when g(-1) = 0 -/
theorem polynomial_root (s : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + s) (-1) = 0 ↔ s = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_l3683_368380


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_five_l3683_368321

theorem sum_of_squares_divisible_by_five (x y : ℤ) :
  (∃ n : ℤ, (x^2 + y^2) = 5*n) →
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_five_l3683_368321


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3683_368399

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3683_368399


namespace NUMINAMATH_CALUDE_intersection_point_l3683_368329

theorem intersection_point (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l3683_368329


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l3683_368360

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 + Real.sqrt 50 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l3683_368360


namespace NUMINAMATH_CALUDE_smallest_n_square_and_fifth_power_l3683_368349

theorem smallest_n_square_and_fifth_power :
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (l : ℕ), 5 * n = l^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (l : ℕ), 5 * m = l^5) → 
    m ≥ 625) ∧
  n = 625 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_fifth_power_l3683_368349


namespace NUMINAMATH_CALUDE_sum_of_roots_l3683_368395

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3683_368395


namespace NUMINAMATH_CALUDE_social_media_earnings_per_hour_l3683_368307

/-- Calculates the earnings per hour for checking social media posts -/
theorem social_media_earnings_per_hour 
  (payment_per_post : ℝ) 
  (time_per_post : ℝ) 
  (seconds_per_hour : ℝ) 
  (h1 : payment_per_post = 0.25)
  (h2 : time_per_post = 10)
  (h3 : seconds_per_hour = 3600) :
  (payment_per_post * (seconds_per_hour / time_per_post)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_social_media_earnings_per_hour_l3683_368307


namespace NUMINAMATH_CALUDE_consecutive_sum_39_largest_l3683_368362

theorem consecutive_sum_39_largest (n m : ℕ) : 
  n + 1 = m → n + m = 39 → m = 20 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_largest_l3683_368362


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l3683_368359

theorem equation_has_four_real_solutions :
  let f : ℝ → ℝ := λ x => 6*x/(x^2 + x + 1) + 7*x/(x^2 - 7*x + 2) + 5/2
  (∃ (a b c d : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (∀ (w x y z : ℝ), (∀ r, f r = 0 ↔ r = w ∨ r = x ∨ r = y ∨ r = z) →
    w = a ∧ x = b ∧ y = c ∧ z = d) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l3683_368359


namespace NUMINAMATH_CALUDE_birds_meeting_point_l3683_368339

/-- Theorem: Meeting point of two birds flying towards each other --/
theorem birds_meeting_point 
  (total_distance : ℝ) 
  (speed_bird1 : ℝ) 
  (speed_bird2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed_bird1 = 4)
  (h3 : speed_bird2 = 1) :
  (speed_bird1 * total_distance) / (speed_bird1 + speed_bird2) = 16 := by
  sorry

#check birds_meeting_point

end NUMINAMATH_CALUDE_birds_meeting_point_l3683_368339


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3683_368306

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 2

-- Define the point of tangency
def x₀ : ℝ := 1
def y₀ : ℝ := 6

-- Define the slope of the tangent line
def m : ℝ := 9

-- Statement to prove
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (9*x - y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3683_368306


namespace NUMINAMATH_CALUDE_square_side_length_l3683_368318

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 289 ∧ area = side * side → side = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3683_368318


namespace NUMINAMATH_CALUDE_inequality_solution_l3683_368361

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 1) ≥ x^2 / (x - 1) + 7/6) ↔ 
  (x < (-1 - Real.sqrt 5) / 2 ∨ 
   (-1 < x ∧ x < 1) ∨ 
   x > (-1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3683_368361


namespace NUMINAMATH_CALUDE_bridge_length_l3683_368313

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3683_368313


namespace NUMINAMATH_CALUDE_B_set_given_A_l3683_368378

def f (a b x : ℝ) : ℝ := x^2 + a*x + b

def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

theorem B_set_given_A (a b : ℝ) :
  A a b = {-1, 3} → B a b = {-1, Real.sqrt 3, -Real.sqrt 3, 3} := by
  sorry

end NUMINAMATH_CALUDE_B_set_given_A_l3683_368378


namespace NUMINAMATH_CALUDE_dave_phone_difference_l3683_368331

theorem dave_phone_difference (initial_apps initial_files final_apps final_files : ℕ) : 
  initial_apps = 11 → 
  initial_files = 3 → 
  final_apps = 2 → 
  final_files = 24 → 
  final_files - final_apps = 22 := by
  sorry

end NUMINAMATH_CALUDE_dave_phone_difference_l3683_368331


namespace NUMINAMATH_CALUDE_buoy_radius_l3683_368348

/-- The radius of a spherical buoy given the dimensions of the hole it leaves in ice --/
theorem buoy_radius (hole_diameter : ℝ) (hole_depth : ℝ) (buoy_radius : ℝ) : 
  hole_diameter = 30 → hole_depth = 12 → buoy_radius = 15.375 := by
  sorry

#check buoy_radius

end NUMINAMATH_CALUDE_buoy_radius_l3683_368348


namespace NUMINAMATH_CALUDE_vector_problem_l3683_368372

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem vector_problem (a : ℝ × ℝ) :
  collinear a (1, -2) →
  a.1 * 1 + a.2 * (-2) = -10 →
  a = (-2, 4) ∧ Real.sqrt ((a.1 + 6)^2 + (a.2 - 7)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3683_368372


namespace NUMINAMATH_CALUDE_fixed_term_deposit_result_l3683_368317

/-- Calculates the total amount after a fixed term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that the total amount after the fixed term deposit is 21998 yuan -/
theorem fixed_term_deposit_result : 
  let principal : ℝ := 20000
  let rate : ℝ := 0.0333
  let time : ℝ := 3
  totalAmount principal rate time = 21998 := by
sorry


end NUMINAMATH_CALUDE_fixed_term_deposit_result_l3683_368317


namespace NUMINAMATH_CALUDE_equation_solution_l3683_368301

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 7/2 ∧ 
  (∀ x : ℝ, 4 * (1 - x)^2 = 25 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3683_368301


namespace NUMINAMATH_CALUDE_floor_ceil_abs_difference_l3683_368369

theorem floor_ceil_abs_difference : |⌊(-5.67 : ℝ)⌋| - ⌈(42.1 : ℝ)⌉ = -37 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_abs_difference_l3683_368369


namespace NUMINAMATH_CALUDE_mixture_volume_calculation_l3683_368340

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem mixture_volume_calculation (initial_volume : ℝ) : 
  (0.20 * initial_volume + 8.333333333333334) / (initial_volume + 8.333333333333334) = 0.25 →
  initial_volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_calculation_l3683_368340


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l3683_368341

theorem geometric_progression_terms (a q : ℝ) : 
  a + a * q = 20 → a * q^2 + a * q^3 = 20/9 →
  ((a = 15 ∧ q = 1/3) ∨ (a = 30 ∧ q = -1/3)) ∧
  (∃ (terms : Fin 4 → ℝ), 
    (terms 0 = a ∧ terms 1 = a * q ∧ terms 2 = a * q^2 ∧ terms 3 = a * q^3) ∧
    ((terms 0 = 15 ∧ terms 1 = 5 ∧ terms 2 = 5/3 ∧ terms 3 = 5/9) ∨
     (terms 0 = 30 ∧ terms 1 = -10 ∧ terms 2 = 10/3 ∧ terms 3 = -10/9))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_terms_l3683_368341


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3683_368332

theorem basketball_tryouts (girls boys callback : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : callback = 10) :
  girls + boys - callback = 39 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3683_368332


namespace NUMINAMATH_CALUDE_parallelepiped_skew_lines_l3683_368308

/-- A parallelepiped is a three-dimensional figure formed by six parallelograms. -/
structure Parallelepiped :=
  (vertices : Finset (Fin 8))

/-- A line in the parallelepiped is defined by two distinct vertices. -/
def Line (p : Parallelepiped) := {l : Fin 8 × Fin 8 // l.1 ≠ l.2}

/-- Two lines are skew if they are not coplanar and do not intersect. -/
def areSkew (p : Parallelepiped) (l1 l2 : Line p) : Prop := sorry

/-- The set of all lines in the parallelepiped. -/
def allLines (p : Parallelepiped) : Finset (Line p) := sorry

/-- The set of all pairs of skew lines in the parallelepiped. -/
def skewLinePairs (p : Parallelepiped) : Finset (Line p × Line p) := sorry

theorem parallelepiped_skew_lines (p : Parallelepiped) :
  (allLines p).card = 28 → (skewLinePairs p).card = 174 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_skew_lines_l3683_368308


namespace NUMINAMATH_CALUDE_system_solution_unique_l3683_368397

theorem system_solution_unique (x y : ℝ) : 
  2 * x - 5 * y = 2 ∧ x + 3 * y = 12 ↔ x = 6 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3683_368397


namespace NUMINAMATH_CALUDE_smallest_number_l3683_368370

theorem smallest_number : min (-5 : ℝ) (min (-0.8) (min 0 (abs (-6)))) = -5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3683_368370


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3683_368310

theorem sqrt_equation_solution (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (8 - s) ^ (1/4) → s = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3683_368310


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3683_368316

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 4 ∧ ∀ x ∈ Set.Icc (-1) 2, f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3683_368316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3683_368324

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 8th term is 23. -/
theorem arithmetic_sequence_8th_term :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 8 = 23 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3683_368324


namespace NUMINAMATH_CALUDE_area_polygon1_area_polygon2_area_polygon3_l3683_368356

-- Define the polygons
def polygon1 := {(x, y) : ℝ × ℝ | |x| ≤ 1 ∧ |y| ≤ 1}
def polygon2 := {(x, y) : ℝ × ℝ | |x| + |y| ≤ 10}
def polygon3 := {(x, y) : ℝ × ℝ | |x| + |y| + |x+y| ≤ 2020}

-- Define the areas
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statements
theorem area_polygon1 : area polygon1 = 4 := by sorry

theorem area_polygon2 : area polygon2 = 200 := by sorry

theorem area_polygon3 : area polygon3 = 3060300 := by sorry

end NUMINAMATH_CALUDE_area_polygon1_area_polygon2_area_polygon3_l3683_368356


namespace NUMINAMATH_CALUDE_pears_eaten_by_mike_l3683_368374

theorem pears_eaten_by_mike (jason_pears keith_pears remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : remaining_pears = 81) :
  jason_pears + keith_pears - remaining_pears = 12 := by
  sorry

end NUMINAMATH_CALUDE_pears_eaten_by_mike_l3683_368374


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3683_368336

/-- The probability of selecting two red balls from a bag containing 6 red, 5 blue, and 2 green balls, when 2 balls are picked at random. -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 6) (h2 : blue = 5) (h3 : green = 2) :
  let total := red + blue + green
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3683_368336


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3683_368385

def p (x y : ℝ) : Prop := (x - 2) * (y - 5) ≠ 0

def q (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 5

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3683_368385


namespace NUMINAMATH_CALUDE_product_equals_120_l3683_368305

theorem product_equals_120 (n : ℕ) (h : n = 3) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_120_l3683_368305


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l3683_368352

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of overlap between two squares -/
def overlapArea (s1 s2 : Square) : ℝ :=
  sorry

/-- Calculates the total area covered by two squares -/
def totalCoveredArea (s1 s2 : Square) : ℝ :=
  sorry

theorem area_of_overlapping_squares :
  let s1 : Square := { sideLength := 20, center := (0, 0) }
  let s2 : Square := { sideLength := 20, center := (10, 0) }
  totalCoveredArea s1 s2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l3683_368352


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3683_368300

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.09)
  (h3 : time = 1) :
  principal * rate * time = 900 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3683_368300


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l3683_368396

theorem complex_multiplication_division (P F G : ℂ) :
  P = 3 + 4 * Complex.I ∧
  F = -Complex.I ∧
  G = 3 - 4 * Complex.I →
  (P * F * G) / (-3 * Complex.I) = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l3683_368396


namespace NUMINAMATH_CALUDE_box_height_is_five_l3683_368309

-- Define the box dimensions and cube properties
def box_length : ℝ := 8
def box_width : ℝ := 15
def cube_volume : ℝ := 10
def min_cubes : ℕ := 60

-- Define the theorem
theorem box_height_is_five :
  let total_volume := (min_cubes : ℝ) * cube_volume
  let height := total_volume / (box_length * box_width)
  height = 5 := by sorry

end NUMINAMATH_CALUDE_box_height_is_five_l3683_368309


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3683_368367

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, 3)
  let m : ℝ := f' P.1
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0
  tangent_eq P.1 P.2 ∧ ∀ x y, tangent_eq x y ↔ y - P.2 = m * (x - P.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3683_368367


namespace NUMINAMATH_CALUDE_two_common_points_l3683_368322

/-- Two curves in the xy-plane -/
structure Curves where
  curve1 : ℝ → ℝ → Prop
  curve2 : ℝ → ℝ → Prop

/-- The specific curves from the problem -/
def problem_curves : Curves where
  curve1 := λ x y => x^2 + 9*y^2 = 9
  curve2 := λ x y => 9*x^2 + y^2 = 1

/-- A point that satisfies both curves -/
def is_common_point (c : Curves) (x y : ℝ) : Prop :=
  c.curve1 x y ∧ c.curve2 x y

/-- The set of all common points -/
def common_points (c : Curves) : Set (ℝ × ℝ) :=
  {p | is_common_point c p.1 p.2}

/-- The theorem stating that there are exactly two common points -/
theorem two_common_points :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
  common_points problem_curves = {p1, p2} :=
sorry

end NUMINAMATH_CALUDE_two_common_points_l3683_368322


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l3683_368377

open Real
open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_range (a b : V) 
  (h1 : ‖b‖ = 2) 
  (h2 : ‖a‖ = 2 * ‖b - a‖) : 
  4/3 ≤ ‖a‖ ∧ ‖a‖ ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l3683_368377


namespace NUMINAMATH_CALUDE_convex_separation_equivalence_l3683_368311

-- Define the type for a compact convex set in ℝ²
def CompactConvexSet : Type := Set (Real × Real)

-- Define the property of a set being compact and convex
def is_compact_convex (S : CompactConvexSet) : Prop := sorry

-- Define the convex hull operation
def conv_hull (S T : CompactConvexSet) : CompactConvexSet := sorry

-- Define the property of a line separating two sets
def separates (L : Set (Real × Real)) (S T : CompactConvexSet) : Prop := sorry

-- Define the property of a line intersecting a set
def intersects (L : Set (Real × Real)) (S : CompactConvexSet) : Prop := sorry

-- The main theorem
theorem convex_separation_equivalence 
  (A B C : CompactConvexSet) 
  (hA : is_compact_convex A) 
  (hB : is_compact_convex B) 
  (hC : is_compact_convex C) : 
  (∀ L : Set (Real × Real), ¬(intersects L A ∧ intersects L B ∧ intersects L C)) ↔ 
  (∃ LA LB LC : Set (Real × Real), 
    separates LA A (conv_hull B C) ∧ 
    separates LB B (conv_hull A C) ∧ 
    separates LC C (conv_hull A B)) := by
  sorry

end NUMINAMATH_CALUDE_convex_separation_equivalence_l3683_368311


namespace NUMINAMATH_CALUDE_sector_arc_length_l3683_368345

/-- Given a circular sector with a central angle of 150° and a radius of 6 cm,
    the arc length is 5π cm. -/
theorem sector_arc_length :
  let θ : ℝ := 150  -- Central angle in degrees
  let r : ℝ := 6    -- Radius in cm
  let L : ℝ := (θ / 360) * (2 * Real.pi * r)  -- Arc length formula
  L = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3683_368345


namespace NUMINAMATH_CALUDE_show_episodes_per_week_l3683_368371

/-- Calculates the number of episodes shown per week given the episode length,
    filming time multiplier, and total filming time for a certain number of weeks. -/
def episodes_per_week (episode_length : ℕ) (filming_multiplier : ℚ) (total_filming_time : ℕ) (num_weeks : ℕ) : ℚ :=
  let filming_time_per_episode : ℚ := episode_length * filming_multiplier
  let total_minutes : ℕ := total_filming_time * 60
  let total_episodes : ℚ := total_minutes / filming_time_per_episode
  total_episodes / num_weeks

/-- Proves that the number of episodes shown each week is 5 under the given conditions. -/
theorem show_episodes_per_week :
  episodes_per_week 20 (3/2) 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_per_week_l3683_368371


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3683_368379

/-- Given a right triangle with sides 5, 12, and 13, let x be the side length of a square
    inscribed with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on a leg of the triangle. Then x/y = 20/17. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  x^2 + (12 - x)^2 = 13^2 ∧
  x^2 + (5 - x)^2 = 12^2 ∧
  y^2 + (5 - y)^2 = (12 - y)^2 →
  x / y = 20 / 17 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3683_368379


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l3683_368394

theorem greatest_prime_factor_of_5_pow_5_plus_12_pow_3 :
  (Nat.factors (5^5 + 12^3)).maximum? = some 19 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l3683_368394


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l3683_368390

theorem johns_candy_store_spending (allowance : ℚ) (arcade_fraction : ℚ) (toy_fraction : ℚ) :
  allowance = 3.60 ∧ 
  arcade_fraction = 3/5 ∧ 
  toy_fraction = 1/3 →
  allowance * (1 - arcade_fraction) * (1 - toy_fraction) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l3683_368390


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l3683_368398

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x * Real.log x + x^2 - 2 * a * x + a^2

-- Define the derivative of f as g
def g (a : ℝ) (x : ℝ) : ℝ := -2 * (1 + Real.log x) + 2 * x - 2 * a

theorem tangent_perpendicular_implies_a (a : ℝ) :
  (g a 1 = -1) → a = -1/2 := by sorry

theorem solutions_of_g_eq_zero (a : ℝ) :
  (a < 0 → ∀ x, g a x ≠ 0) ∧
  (a = 0 → ∃! x, g a x = 0) ∧
  (a > 0 → ∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0) := by sorry

end

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_solutions_of_g_eq_zero_l3683_368398


namespace NUMINAMATH_CALUDE_ratio_proof_l3683_368343

theorem ratio_proof (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.gcd a.val b.val = 5) (h3 : Nat.lcm a.val b.val = 60) : a.val * 4 = b.val * 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l3683_368343


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_minimum_l3683_368350

theorem arithmetic_geometric_sequence_minimum (n : ℕ) (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d > 0 →
  (∀ k, a k = a 1 + (k - 1) * d) →
  a 1 = 5 →
  (a 5 - 1)^2 = a 2 * a 10 →
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * d)) →
  (∀ k, (2 * S k + k + 32) / (a k + 1) ≥ 20 / 3) ∧
  (∃ k, (2 * S k + k + 32) / (a k + 1) = 20 / 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_minimum_l3683_368350


namespace NUMINAMATH_CALUDE_ludvik_favorite_number_l3683_368325

/-- Ludvík's favorite number problem -/
theorem ludvik_favorite_number 
  (a b : ℝ) -- original dividend and divisor
  (h1 : (2 * a) / (b + 12) = (a - 42) / (b / 2)) -- equality of the two scenarios
  (h2 : (2 * a) / (b + 12) > 0) -- ensure the result is positive
  : (2 * a) / (b + 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ludvik_favorite_number_l3683_368325


namespace NUMINAMATH_CALUDE_right_angled_triangle_x_values_l3683_368353

def triangle_ABC (x : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let AB := (2, -1)
    let AC := (x, 3)
    let BC := (x - 2, 4)
    (AB.1 * AC.1 + AB.2 * AC.2 = 0) ∨ 
    (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∨
    (AC.1 * BC.1 + AC.2 * BC.2 = 0)

theorem right_angled_triangle_x_values :
  ∀ x : ℝ, triangle_ABC x → x = 3/2 ∨ x = 4 :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_x_values_l3683_368353


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3683_368312

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 + (1/6)^2 = (37 * x / 85) * ((1/5)^2 + (1/7)^2 + (1/8)^2) * y →
  Real.sqrt x / Real.sqrt y = 1737 / 857 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3683_368312


namespace NUMINAMATH_CALUDE_parabola_equation_and_max_area_l3683_368328

-- Define the parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

-- Define a point on the parabola
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : c.equation x y

-- Define the vector from focus to a point
def vector_from_focus (c : Parabola) (p : PointOnParabola c) : ℝ × ℝ :=
  (p.x - c.focus.1, p.y - c.focus.2)

theorem parabola_equation_and_max_area 
  (c : Parabola)
  (h_focus : c.focus = (0, 1))
  (h_equation : ∀ x y, c.equation x y ↔ x^2 = 2 * c.p * y)
  (A B C : PointOnParabola c)
  (h_vector_sum : vector_from_focus c A + vector_from_focus c B + vector_from_focus c C = (0, 0)) :
  (∀ x y, c.equation x y ↔ x^2 = 4 * y) ∧
  (∃ (max_area : ℝ), max_area = (3 * Real.sqrt 6) / 2 ∧
    ∀ (area : ℝ), area = abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2 →
      area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_and_max_area_l3683_368328


namespace NUMINAMATH_CALUDE_expression_simplification_l3683_368333

theorem expression_simplification (a : ℝ) (h : a = 2) : 
  (1 / (a + 1) - (a + 2) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 4)) * (a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3683_368333


namespace NUMINAMATH_CALUDE_distinct_roots_sum_abs_gt_six_l3683_368319

theorem distinct_roots_sum_abs_gt_six (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 9 = 0 → 
  r₂^2 + p*r₂ + 9 = 0 → 
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_abs_gt_six_l3683_368319


namespace NUMINAMATH_CALUDE_students_in_both_games_l3683_368334

theorem students_in_both_games (total : ℕ) (game_a : ℕ) (game_b : ℕ) 
  (h_total : total = 55) (h_game_a : game_a = 38) (h_game_b : game_b = 42) :
  ∃ x : ℕ, x = game_a + game_b - total ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_games_l3683_368334


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3683_368387

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 8 * x - 3 > 0 ↔ x < -1/3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3683_368387


namespace NUMINAMATH_CALUDE_sequence_relation_l3683_368347

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * b (n + 1) - b n

theorem sequence_relation (n : ℕ) : (b n)^2 = 3 * (a n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l3683_368347


namespace NUMINAMATH_CALUDE_sum_N_equals_2250_l3683_368335

def N : ℕ → ℤ
  | 0 => 0
  | n + 1 => let k := 40 - n
              if n % 2 = 0 then
                N n + (3*k)^2 + (3*k-1)^2 + (3*k-2)^2
              else
                N n - (3*k)^2 - (3*k-1)^2 - (3*k-2)^2

theorem sum_N_equals_2250 : N 40 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_sum_N_equals_2250_l3683_368335


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l3683_368338

/-- For a normal distribution with given properties, prove the standard deviation --/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 17.5) (h2 : μ - 2 * σ = 12.5) : σ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l3683_368338


namespace NUMINAMATH_CALUDE_signal_arrangements_l3683_368355

def num_red_flags : ℕ := 3
def num_white_flags : ℕ := 2
def total_flags : ℕ := num_red_flags + num_white_flags

theorem signal_arrangements : (total_flags.choose num_red_flags) = 10 := by
  sorry

end NUMINAMATH_CALUDE_signal_arrangements_l3683_368355


namespace NUMINAMATH_CALUDE_decoration_nail_count_l3683_368393

theorem decoration_nail_count :
  ∀ D : ℕ,
  (D : ℚ) * (21/80) = 20 →
  ⌊(D : ℚ) * (5/8)⌋ = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_decoration_nail_count_l3683_368393


namespace NUMINAMATH_CALUDE_probability_of_six_l3683_368351

/-- A fair die with 6 faces -/
structure FairDie :=
  (faces : Nat)
  (is_fair : Bool)
  (h_faces : faces = 6)
  (h_fair : is_fair = true)

/-- The probability of getting a specific face on a fair die -/
def probability_of_face (d : FairDie) : ℚ :=
  1 / d.faces

/-- Theorem: The probability of getting any specific face on a fair 6-faced die is 1/6 -/
theorem probability_of_six (d : FairDie) : probability_of_face d = 1 / 6 := by
  sorry

#eval (1 : ℚ) / 6  -- To show that 1/6 ≈ 0.17

end NUMINAMATH_CALUDE_probability_of_six_l3683_368351


namespace NUMINAMATH_CALUDE_min_value_of_a_l3683_368366

theorem min_value_of_a (a b c : ℝ) : 
  a + b + c = 3 → 
  a ≥ b → 
  b ≥ c → 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 → 
  a ≥ 4/3 ∧ ∀ a' : ℝ, (∃ b' c' : ℝ, 
    a' + b' + c' = 3 ∧ 
    a' ≥ b' ∧ 
    b' ≥ c' ∧ 
    (∃ x : ℝ, a' * x^2 + b' * x + c' = 0)) → 
  a' ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3683_368366
