import Mathlib

namespace NUMINAMATH_CALUDE_product_of_reversed_digits_l2083_208330

theorem product_of_reversed_digits (A B : ℕ) (k : ℕ) : 
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 →
  (10 * A + B) * (10 * B + A) = k →
  (k + 1) % 101 = 0 →
  k = 403 := by
sorry

end NUMINAMATH_CALUDE_product_of_reversed_digits_l2083_208330


namespace NUMINAMATH_CALUDE_triangle_side_length_l2083_208351

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π/3 →
  Real.cos B = (2 * Real.sqrt 7) / 7 →
  b = 3 →
  a = (3 * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2083_208351


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2083_208346

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ r : ℝ, r > 0 ∧ c = 2 * r) → c / 2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2083_208346


namespace NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l2083_208384

theorem derivative_zero_at_negative_one (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 - 4) * (x - t)
  let f' : ℝ → ℝ := λ x ↦ 2*x*(x - t) + (x^2 - 4)
  f' (-1) = 0 → t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l2083_208384


namespace NUMINAMATH_CALUDE_coin_collection_problem_l2083_208371

theorem coin_collection_problem (n d q : ℕ) : 
  n + d + q = 23 →
  5 * n + 10 * d + 25 * q = 320 →
  d = n + 3 →
  q - n = 2 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l2083_208371


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2083_208369

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

theorem quadratic_function_value (d e f : ℝ) :
  (∀ x, QuadraticFunction d e f x = d * x^2 + e * x + f) →
  QuadraticFunction d e f 0 = 2 →
  (∀ x, QuadraticFunction d e f (3.5 + x) = QuadraticFunction d e f (3.5 - x)) →
  ∃ n : ℤ, QuadraticFunction d e f 10 = n →
  QuadraticFunction d e f 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2083_208369


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l2083_208386

def probability_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem mayor_approval_probability : 
  probability_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l2083_208386


namespace NUMINAMATH_CALUDE_book_pages_theorem_l2083_208352

/-- Represents the number of pages read each night --/
structure ReadingPattern :=
  (night1 : ℕ)
  (night2 : ℕ)
  (night3 : ℕ)
  (night4 : ℕ)

/-- Calculates the total number of pages in the book --/
def totalPages (rp : ReadingPattern) : ℕ :=
  rp.night1 + rp.night2 + rp.night3 + rp.night4

/-- Theorem: The book has 100 pages in total --/
theorem book_pages_theorem (rp : ReadingPattern) 
  (h1 : rp.night1 = 15)
  (h2 : rp.night2 = 2 * rp.night1)
  (h3 : rp.night3 = rp.night2 + 5)
  (h4 : rp.night4 = 20) : 
  totalPages rp = 100 := by
  sorry


end NUMINAMATH_CALUDE_book_pages_theorem_l2083_208352


namespace NUMINAMATH_CALUDE_sequence_modulo_l2083_208337

/-- Given a prime number p > 3, we define a sequence a_n as follows:
    a_n = n for n ∈ {0, 1, ..., p-1}
    a_n = a_{n-1} + a_{n-p} for n ≥ p
    This theorem states that a_{p^3} ≡ p-1 (mod p) -/
theorem sequence_modulo (p : ℕ) (hp : p.Prime ∧ p > 3) : 
  ∃ a : ℕ → ℕ, 
    (∀ n < p, a n = n) ∧ 
    (∀ n ≥ p, a n = a (n-1) + a (n-p)) ∧ 
    a (p^3) ≡ p-1 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_sequence_modulo_l2083_208337


namespace NUMINAMATH_CALUDE_symmetry_propositions_l2083_208398

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Proposition ①
def prop1 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x - 1) = f (x + 1)) →
  (∀ x y : ℝ, f (1 + (x - 1)) = f (1 - (x - 1)) ∧ y = f (x - 1) ↔ y = f (2 - x))

-- Proposition ②
def prop2 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = -f (-x)) →
  (∀ x y : ℝ, y = f (x - 1) ↔ -y = f (2 - x))

-- Proposition ③
def prop3 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 1) + f (1 - x) = 0) →
  (∀ x y : ℝ, y = f x ↔ -y = f (2 - x))

-- Proposition ④
def prop4 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y = f (x - 1) ↔ y = f (1 - x)

-- Theorem stating which propositions are correct
theorem symmetry_propositions :
  ¬ (∀ f : ℝ → ℝ, prop1 f) ∧
  (∀ f : ℝ → ℝ, prop2 f) ∧
  (∀ f : ℝ → ℝ, prop3 f) ∧
  ¬ (∀ f : ℝ → ℝ, prop4 f) :=
sorry

end NUMINAMATH_CALUDE_symmetry_propositions_l2083_208398


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l2083_208367

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/3))^(1/4))^3 * (((b^16)^(1/4))^(1/3))^3 = b^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l2083_208367


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2083_208366

/-- The cost relationship between bananas, apples, and oranges -/
structure FruitCosts where
  banana_apple_ratio : ℚ  -- 5 bananas cost as much as 3 apples
  apple_orange_ratio : ℚ  -- 9 apples cost as much as 6 oranges

/-- The theorem stating the cost equivalence of 30 bananas and 12 oranges -/
theorem banana_orange_equivalence (fc : FruitCosts) 
  (h1 : fc.banana_apple_ratio = 5 / 3)
  (h2 : fc.apple_orange_ratio = 9 / 6) : 
  (30 : ℚ) / fc.banana_apple_ratio * fc.apple_orange_ratio = 12 := by
  sorry

#check banana_orange_equivalence

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2083_208366


namespace NUMINAMATH_CALUDE_min_value_of_f_l2083_208381

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2083_208381


namespace NUMINAMATH_CALUDE_dan_gave_fourteen_marbles_l2083_208380

/-- The number of marbles Dan gave to Mary -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Dan gave 14 marbles to Mary -/
theorem dan_gave_fourteen_marbles :
  let initial := 64
  let remaining := 50
  marbles_given initial remaining = 14 := by
sorry

end NUMINAMATH_CALUDE_dan_gave_fourteen_marbles_l2083_208380


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_achieved_l2083_208387

theorem max_value_of_sum_of_roots (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  (Real.sqrt (x + 20) + Real.sqrt (20 - x) + Real.sqrt (2 * x) + Real.sqrt (30 - x)) ≤ Real.sqrt 630 :=
by sorry

theorem max_value_achieved (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  ∃ y, 0 ≤ y ∧ y ≤ 20 ∧
    (Real.sqrt (y + 20) + Real.sqrt (20 - y) + Real.sqrt (2 * y) + Real.sqrt (30 - y)) = Real.sqrt 630 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_achieved_l2083_208387


namespace NUMINAMATH_CALUDE_book_reading_percentage_l2083_208372

theorem book_reading_percentage (total_pages : ℕ) (remaining_pages : ℕ) : 
  total_pages = 400 → remaining_pages = 320 → 
  (((total_pages - remaining_pages) : ℚ) / total_pages) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_book_reading_percentage_l2083_208372


namespace NUMINAMATH_CALUDE_three_false_propositions_l2083_208304

theorem three_false_propositions :
  (¬ ∀ a b : ℝ, (1 / a < 1 / b) → (a > b)) ∧
  (¬ ∀ a b c : ℝ, (a > b ∧ b > c) → (a * |c| > b * |c|)) ∧
  (¬ ∃ x₀ : ℝ, ∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_three_false_propositions_l2083_208304


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l2083_208373

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci property
def foci_property (a b : ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the complementary angle property
def complementary_angles (k : ℝ) : Prop :=
  ∃ (xC yC xD yD : ℝ),
    yC - 1 = k * (xC - 1) ∧
    yD - 1 = -k * (xD - 1)

theorem ellipse_slope_theorem (a b : ℝ) (F₁ F₂ : ℝ × ℝ) (k : ℝ) :
  a > b ∧ b > 0 →
  is_on_ellipse 1 1 a b →
  foci_property a b F₁ F₂ →
  Real.sqrt ((1 - F₁.1)^2 + (1 - F₁.2)^2) + Real.sqrt ((1 - F₂.1)^2 + (1 - F₂.2)^2) = 4 →
  complementary_angles k →
  ∃ (xC yC xD yD : ℝ),
    is_on_ellipse xC yC a b ∧
    is_on_ellipse xD yD a b ∧
    (yD - yC) / (xD - xC) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l2083_208373


namespace NUMINAMATH_CALUDE_complex_modulus_l2083_208350

theorem complex_modulus (z : ℂ) (h : z = Complex.I / (1 + Complex.I)) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2083_208350


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2083_208382

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 1 / 3 →    -- Given ratio of sides
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- Segments r and s form the hypotenuse
  r / s = 1 / 9 :=   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2083_208382


namespace NUMINAMATH_CALUDE_farmer_earnings_l2083_208395

/-- Calculates the total earnings from selling potatoes and carrots -/
def total_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                   (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) : ℚ :=
  let potato_bundles := potato_count / potato_bundle_size
  let carrot_bundles := carrot_count / carrot_bundle_size
  potato_bundles * potato_bundle_price + carrot_bundles * carrot_bundle_price

/-- The farmer's earnings from selling all harvested crops -/
theorem farmer_earnings : 
  total_earnings 250 25 1.9 320 20 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_l2083_208395


namespace NUMINAMATH_CALUDE_max_g_6_l2083_208365

/-- A polynomial of degree 2 with real, nonnegative coefficients -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the maximum value of g(6) given the conditions -/
theorem max_g_6 (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h1 : g a b c 3 = 3) (h2 : g a b c 9 = 243) :
  ∀ a' b' c', a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → 
  g a' b' c' 3 = 3 → g a' b' c' 9 = 243 → 
  g a' b' c' 6 ≤ 6 :=
by sorry

#check max_g_6

end NUMINAMATH_CALUDE_max_g_6_l2083_208365


namespace NUMINAMATH_CALUDE_product_variation_l2083_208339

theorem product_variation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (5 * a) * b = 5 * (a * b) := by sorry

end NUMINAMATH_CALUDE_product_variation_l2083_208339


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_unique_perpendicular_plane_l2083_208306

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (passes_through : Plane → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular_line_plane b α →
  perpendicular_line_line a b :=
sorry

-- Theorem 2
theorem unique_perpendicular_plane
  (a b : Line) :
  perpendicular_line_line a b →
  ∃! p : Plane, passes_through p b ∧ perpendicular_line_plane a p :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_unique_perpendicular_plane_l2083_208306


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_roots_l2083_208318

theorem unique_magnitude_quadratic_roots : ∃! m : ℝ, ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_roots_l2083_208318


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2083_208358

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 19881 → n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2083_208358


namespace NUMINAMATH_CALUDE_average_speed_theorem_l2083_208391

theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end NUMINAMATH_CALUDE_average_speed_theorem_l2083_208391


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2083_208378

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2083_208378


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2083_208353

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (m : ℝ) :
  let z : ℂ := Complex.mk (m + 1) (m - 1)
  is_purely_imaginary z → m = -1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2083_208353


namespace NUMINAMATH_CALUDE_jessica_cut_two_roses_l2083_208377

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses given the initial and final flower counts -/
theorem jessica_cut_two_roses :
  roses_cut 15 62 17 96 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_two_roses_l2083_208377


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_9_l2083_208374

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def five_digit_number (d : ℕ) : ℕ := 56780 + d

theorem five_digit_multiple_of_9 (d : ℕ) : 
  d < 10 → (is_multiple_of_9 (five_digit_number d) ↔ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_9_l2083_208374


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2083_208305

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2083_208305


namespace NUMINAMATH_CALUDE_find_b_l2083_208385

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2083_208385


namespace NUMINAMATH_CALUDE_largest_invertible_interval_for_g_l2083_208355

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the theorem
theorem largest_invertible_interval_for_g :
  ∃ (a : ℝ), 
    (∀ (I : Set ℝ), (2 ∈ I) → (∀ (x y : ℝ), x ∈ I → y ∈ I → x ≠ y → g x ≠ g y) → 
      I ⊆ {x : ℝ | a ≤ x}) ∧
    ({x : ℝ | a ≤ x} ⊆ {x : ℝ | ∀ (y : ℝ), y ∈ {x : ℝ | a ≤ x} → y ≠ x → g y ≠ g x}) ∧
    a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_for_g_l2083_208355


namespace NUMINAMATH_CALUDE_toys_per_week_l2083_208326

/-- The number of days worked per week -/
def days_per_week : ℕ := 3

/-- The number of toys produced per day -/
def toys_per_day : ℝ := 2133.3333333333335

/-- Theorem: The number of toys produced per week is 6400 -/
theorem toys_per_week : ℕ := by
  sorry

end NUMINAMATH_CALUDE_toys_per_week_l2083_208326


namespace NUMINAMATH_CALUDE_remainder_sum_l2083_208319

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53)
  (hd : d % 45 = 28) : 
  (c + d) % 15 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2083_208319


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_l2083_208345

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 → a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_l2083_208345


namespace NUMINAMATH_CALUDE_wang_parts_processed_l2083_208394

/-- Represents the number of parts processed by a worker in a given time -/
def parts_processed (rate : ℕ) (time : ℕ) : ℕ := rate * time

/-- Represents Xiao Wang's work cycle -/
def wang_cycle (total_time : ℕ) : ℕ :=
  parts_processed 15 (2 * (total_time / 3))

/-- Represents Xiao Li's work -/
def li_work (total_time : ℕ) : ℕ :=
  parts_processed 12 total_time

theorem wang_parts_processed :
  ∃ (t : ℕ), t > 0 ∧ wang_cycle t = li_work t ∧ wang_cycle t = 60 :=
sorry

end NUMINAMATH_CALUDE_wang_parts_processed_l2083_208394


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l2083_208349

theorem max_sum_under_constraint (a b c : ℝ) :
  a^2 + 4*b^2 + 9*c^2 - 2*a - 12*b + 6*c + 2 = 0 →
  a + b + c ≤ 17/3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l2083_208349


namespace NUMINAMATH_CALUDE_fred_initial_sheets_l2083_208317

def initial_sheets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x received given final =>
    x + received - given = final

theorem fred_initial_sheets :
  ∃ x : ℕ, initial_sheets x 307 156 363 ∧ x = 212 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_initial_sheets_l2083_208317


namespace NUMINAMATH_CALUDE_daisy_exchange_impossible_l2083_208364

/-- Represents the number of girls in the row -/
def n : ℕ := 33

/-- Represents the number of places each girl passes her daisy -/
def pass_distance : ℕ := 2

/-- Predicate that checks if a girl at position i receives a daisy -/
def receives_daisy (i : ℕ) : Prop :=
  ∃ j : ℕ, j ≤ n ∧ (i = j + pass_distance ∨ i = j - pass_distance)

/-- Theorem stating it's impossible for every girl to end up with exactly one daisy -/
theorem daisy_exchange_impossible : ¬(∀ i : ℕ, i ≤ n → ∃! j : ℕ, receives_daisy j ∧ i = j) :=
sorry

end NUMINAMATH_CALUDE_daisy_exchange_impossible_l2083_208364


namespace NUMINAMATH_CALUDE_hare_jumps_to_12th_cell_l2083_208323

/-- The number of ways a hare can reach the nth cell -/
def hare_jumps : ℕ → ℕ
| 0 => 0  -- No ways to reach the 0th cell (not part of the strip)
| 1 => 1  -- One way to be at the 1st cell (starting position)
| 2 => 1  -- One way to reach the 2nd cell (single jump from 1st)
| (n + 3) => hare_jumps (n + 2) + hare_jumps (n + 1)

/-- The number of ways a hare can jump from the 1st cell to the 12th cell is 144 -/
theorem hare_jumps_to_12th_cell : hare_jumps 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_hare_jumps_to_12th_cell_l2083_208323


namespace NUMINAMATH_CALUDE_cyclist_travel_time_is_40_l2083_208376

/-- Represents the tram schedule and cyclist's journey -/
structure TramSchedule where
  /-- Interval between tram departures from Station A (in minutes) -/
  departure_interval : ℕ
  /-- Time for a tram to travel from Station A to Station B (in minutes) -/
  journey_time : ℕ
  /-- Number of trams encountered by the cyclist -/
  trams_encountered : ℕ

/-- Calculates the cyclist's travel time -/
def cyclist_travel_time (schedule : TramSchedule) : ℕ :=
  (schedule.trams_encountered + 2) * schedule.departure_interval

/-- Theorem stating the cyclist's travel time is 40 minutes -/
theorem cyclist_travel_time_is_40 (schedule : TramSchedule)
  (h1 : schedule.departure_interval = 5)
  (h2 : schedule.journey_time = 15)
  (h3 : schedule.trams_encountered = 10) :
  cyclist_travel_time schedule = 40 := by
  sorry

#eval cyclist_travel_time { departure_interval := 5, journey_time := 15, trams_encountered := 10 }

end NUMINAMATH_CALUDE_cyclist_travel_time_is_40_l2083_208376


namespace NUMINAMATH_CALUDE_morning_faces_l2083_208332

/-- Represents a cuboid room -/
structure CuboidRoom where
  totalFaces : Nat
  eveningFaces : Nat

/-- Theorem: The number of faces Samuel painted in the morning is 3 -/
theorem morning_faces (room : CuboidRoom) 
  (h1 : room.totalFaces = 6)
  (h2 : room.eveningFaces = 3) : 
  room.totalFaces - room.eveningFaces = 3 := by
  sorry

#check morning_faces

end NUMINAMATH_CALUDE_morning_faces_l2083_208332


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l2083_208354

/-- The cost of a large monkey doll satisfies the given conditions --/
theorem large_monkey_doll_cost : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (300 / (L - 2) = 300 / L + 25) ∧ 
  (L = 6) := by
sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l2083_208354


namespace NUMINAMATH_CALUDE_percentage_of_chemical_b_in_solution_x_l2083_208363

-- Define the solutions and mixture
def solution_x (a b : ℝ) : Prop := a + b = 1 ∧ a = 0.1
def solution_y : Prop := 0.2 + 0.8 = 1
def mixture (x y : ℝ) : Prop := x + y = 1 ∧ x = 0.8

-- Define the chemical compositions
def chemical_a_in_mixture : ℝ := 0.12
def chemical_b_in_solution_x : ℝ := 0.9

-- State the theorem
theorem percentage_of_chemical_b_in_solution_x 
  (a b x y : ℝ) 
  (hx : solution_x a b) 
  (hy : solution_y)
  (hm : mixture x y)
  (ha : x * a + y * 0.2 = chemical_a_in_mixture) :
  b = chemical_b_in_solution_x :=
sorry

end NUMINAMATH_CALUDE_percentage_of_chemical_b_in_solution_x_l2083_208363


namespace NUMINAMATH_CALUDE_ellipse_foci_l2083_208356

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 25 = 1

/-- The coordinates of a focus of the ellipse -/
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 3 ∨ y = -3)

/-- Theorem: The foci of the given ellipse are at (0, ±3) -/
theorem ellipse_foci :
  ∀ x y : ℝ, is_ellipse x y → ∃ fx fy : ℝ, is_focus fx fy :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2083_208356


namespace NUMINAMATH_CALUDE_range_of_a_l2083_208303

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -3 ∨ x > 1
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x, ¬(p x) → ¬(q x a)) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2083_208303


namespace NUMINAMATH_CALUDE_smallest_c_is_correct_l2083_208307

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive real value c such that d(n) ≤ c * √n for all positive integers n -/
noncomputable def smallest_c : ℝ := Real.sqrt 3

theorem smallest_c_is_correct :
  (∀ n : ℕ+, (num_divisors n : ℝ) ≤ smallest_c * Real.sqrt n) ∧
  (∀ c : ℝ, 0 < c → c < smallest_c →
    ∃ n : ℕ+, (num_divisors n : ℝ) > c * Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_is_correct_l2083_208307


namespace NUMINAMATH_CALUDE_expression_equals_one_l2083_208334

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b*c) * (b^2 - a*c))) +
  (a^2 * c^2 / ((a^2 - b*c) * (c^2 - a*b))) +
  (b^2 * c^2 / ((b^2 - a*c) * (c^2 - a*b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2083_208334


namespace NUMINAMATH_CALUDE_starting_player_wins_l2083_208396

/-- A game state representing the cards held by each player -/
structure GameState :=
  (player_cards : List Nat)
  (opponent_cards : List Nat)

/-- Check if a list of digits can form a number divisible by 17 -/
def can_form_divisible_by_17 (digits : List Nat) : Bool :=
  sorry

/-- The optimal strategy for the starting player -/
def optimal_strategy (state : GameState) : Option Nat :=
  sorry

/-- Theorem stating that the starting player wins with optimal play -/
theorem starting_player_wins :
  ∀ (initial_cards : List Nat),
    initial_cards.length = 7 ∧
    (∀ n, n ∈ initial_cards → n ≥ 0 ∧ n ≤ 6) →
    ∃ (final_state : GameState),
      final_state.player_cards ⊆ initial_cards ∧
      final_state.opponent_cards ⊆ initial_cards ∧
      final_state.player_cards.length + final_state.opponent_cards.length = 7 ∧
      can_form_divisible_by_17 final_state.player_cards ∧
      ¬can_form_divisible_by_17 final_state.opponent_cards :=
  sorry

end NUMINAMATH_CALUDE_starting_player_wins_l2083_208396


namespace NUMINAMATH_CALUDE_ali_flower_sales_l2083_208338

/-- Calculates the total number of flowers sold by Ali -/
def total_flowers_sold (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday + 2 * monday

theorem ali_flower_sales : total_flowers_sold 4 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l2083_208338


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l2083_208325

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l2083_208325


namespace NUMINAMATH_CALUDE_syllogism_arrangement_l2083_208361

-- Define the property of being divisible by 2
def divisible_by_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define odd numbers
def odd (n : ℕ) : Prop := ¬(divisible_by_2 n)

-- State the theorem
theorem syllogism_arrangement :
  (∀ n : ℕ, odd n → ¬(divisible_by_2 n)) →  -- Statement ②
  (odd 2013) →                              -- Statement ③
  ¬(divisible_by_2 2013)                    -- Statement ①
  := by sorry

end NUMINAMATH_CALUDE_syllogism_arrangement_l2083_208361


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_l2083_208314

/-- Given a square with side length 2a centered at the origin, intersected by the line y = x/3,
    the perimeter of one resulting quadrilateral divided by a equals (14 + 2√10) / 3. -/
theorem square_intersection_perimeter (a : ℝ) (h : a > 0) :
  let square := {p : ℝ × ℝ | max (|p.1|) (|p.2|) = a}
  let line := {p : ℝ × ℝ | p.2 = p.1 / 3}
  let intersection := square ∩ line
  let quadrilateral_perimeter := 
    2 * (a - a / 3) +  -- vertical sides
    2 * a +            -- horizontal side
    Real.sqrt ((2*a)^2 + (2*a/3)^2) -- diagonal
  quadrilateral_perimeter / a = (14 + 2 * Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_l2083_208314


namespace NUMINAMATH_CALUDE_jakes_weight_l2083_208392

theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra) 
  (h2 : jake + kendra = 287) : 
  jake = 194 := by sorry

end NUMINAMATH_CALUDE_jakes_weight_l2083_208392


namespace NUMINAMATH_CALUDE_solution_satisfies_system_and_initial_conditions_l2083_208379

noncomputable def y₁ (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def y₂ (x : ℝ) : ℝ := Real.exp (-x) + Real.exp (2 * x)
noncomputable def y₃ (x : ℝ) : ℝ := -Real.exp (-x) + Real.exp (2 * x)

theorem solution_satisfies_system_and_initial_conditions :
  (∀ x, (deriv y₁) x = y₂ x + y₃ x) ∧
  (∀ x, (deriv y₂) x = y₁ x + y₃ x) ∧
  (∀ x, (deriv y₃) x = y₁ x + y₂ x) ∧
  y₁ 0 = 1 ∧
  y₂ 0 = 2 ∧
  y₃ 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_and_initial_conditions_l2083_208379


namespace NUMINAMATH_CALUDE_toothpick_problem_l2083_208347

theorem toothpick_problem (n : ℕ) : 
  n > 5000 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 →
  n = 5039 :=
by sorry

end NUMINAMATH_CALUDE_toothpick_problem_l2083_208347


namespace NUMINAMATH_CALUDE_salt_production_january_l2083_208315

/-- The salt production problem -/
theorem salt_production_january (
  monthly_increase : ℕ → ℝ)
  (average_daily_production : ℝ)
  (h1 : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → monthly_increase n = 100)
  (h2 : average_daily_production = 100.27397260273973)
  (h3 : ∃ january_production : ℝ,
    (january_production +
      (january_production + monthly_increase 1) +
      (january_production + monthly_increase 1 + monthly_increase 2) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10 + monthly_increase 11)) / 365 = average_daily_production) :
  ∃ january_production : ℝ, january_production = 2500 :=
by sorry

end NUMINAMATH_CALUDE_salt_production_january_l2083_208315


namespace NUMINAMATH_CALUDE_table_sum_theorem_l2083_208321

-- Define a 3x3 table as a function from (Fin 3 × Fin 3) to ℕ
def Table := Fin 3 → Fin 3 → ℕ

-- Define the property that the table contains numbers from 1 to 9
def containsOneToNine (t : Table) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → ∃ i j : Fin 3, t i j = n

-- Define the sum of a diagonal
def diagonalSum (t : Table) (d : Bool) : ℕ :=
  if d then t 0 0 + t 1 1 + t 2 2
  else t 0 2 + t 1 1 + t 2 0

-- Define the sum of five specific cells
def fiveCellSum (t : Table) : ℕ :=
  t 0 1 + t 1 0 + t 1 1 + t 1 2 + t 2 1

theorem table_sum_theorem (t : Table) 
  (h1 : containsOneToNine t)
  (h2 : diagonalSum t true = 7)
  (h3 : diagonalSum t false = 21) :
  fiveCellSum t = 25 := by
  sorry


end NUMINAMATH_CALUDE_table_sum_theorem_l2083_208321


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2083_208308

/-- The angle between two planar vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : a.1^2 + a.2^2 = 4)           -- |a| = 2 condition
  (h3 : b.1^2 + b.2^2 = 1)           -- |b| = 1 condition
  : angle_between a b = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2083_208308


namespace NUMINAMATH_CALUDE_number_tower_pattern_l2083_208390

theorem number_tower_pattern (n : ℕ) : (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_number_tower_pattern_l2083_208390


namespace NUMINAMATH_CALUDE_binomial_coefficient_1999000_l2083_208383

theorem binomial_coefficient_1999000 :
  ∀ x : ℕ+, (∃ y : ℕ+, Nat.choose x.val y.val = 1999000) ↔ (x.val = 1999000 ∨ x.val = 2000) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1999000_l2083_208383


namespace NUMINAMATH_CALUDE_playground_route_combinations_l2083_208389

theorem playground_route_combinations : 
  ∀ (n : ℕ) (k : ℕ), n = 2 ∧ k = 3 → n ^ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_playground_route_combinations_l2083_208389


namespace NUMINAMATH_CALUDE_y_satisfies_differential_equation_l2083_208301

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt ((Real.log ((1 + Real.exp x) / 2))^2 + 1)

-- State the theorem
theorem y_satisfies_differential_equation (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_differential_equation_l2083_208301


namespace NUMINAMATH_CALUDE_players_in_both_games_l2083_208309

theorem players_in_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : indoor = 110) : 
  outdoor + indoor - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_players_in_both_games_l2083_208309


namespace NUMINAMATH_CALUDE_marbles_remaining_l2083_208322

def initial_marbles : ℕ := 47
def shared_marbles : ℕ := 42

theorem marbles_remaining : initial_marbles - shared_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l2083_208322


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l2083_208399

theorem isosceles_right_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ z₂ →
  (z₂ - 0) • (z₁ - 0) = 0 →  -- Perpendicular condition
  Complex.abs (z₂ - 0) = Complex.abs (z₁ - z₂) →  -- Isosceles condition
  a^2 / b = 2*Real.sqrt 2 + 2*Complex.I*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l2083_208399


namespace NUMINAMATH_CALUDE_packet_weight_difference_l2083_208336

theorem packet_weight_difference (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + e) / 4 = 79 →
  a = 75 →
  e - d = 3 :=
by sorry

end NUMINAMATH_CALUDE_packet_weight_difference_l2083_208336


namespace NUMINAMATH_CALUDE_angle_DEB_is_165_l2083_208375

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angleACB : ℝ
  angleABC : ℝ
  -- Other angles
  angleADE : ℝ
  angleCDE : ℝ
  -- AEB is a straight angle
  angleAEB : ℝ

-- Define the theorem
theorem angle_DEB_is_165 (config : GeometricConfiguration) 
  (h1 : config.angleACB = 90)
  (h2 : config.angleABC = 55)
  (h3 : config.angleADE = 130)
  (h4 : config.angleCDE = 50)
  (h5 : config.angleAEB = 180) :
  ∃ (angleDEB : ℝ), angleDEB = 165 := by
    sorry

end NUMINAMATH_CALUDE_angle_DEB_is_165_l2083_208375


namespace NUMINAMATH_CALUDE_math_city_intersections_l2083_208329

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool
  no_three_streets_intersect : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem: A city with 12 streets, no parallel streets, and no three streets intersecting at a point has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel_streets = true → c.no_three_streets_intersect = true →
  max_intersections c = 66 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l2083_208329


namespace NUMINAMATH_CALUDE_counterexample_exists_l2083_208312

/-- A function that returns the sum of digits of a natural number in base 4038 -/
def sumOfDigitsBase4038 (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is "good" (sum of digits in base 4038 is divisible by 2019) -/
def isGood (n : ℕ) : Prop := sumOfDigitsBase4038 n % 2019 = 0

theorem counterexample_exists (a : ℝ) : a ≥ 2019 →
  ∃ (seq : ℕ → ℕ), 
    (∀ m n : ℕ, m ≠ n → seq m ≠ seq n) ∧ 
    (∀ n : ℕ, (seq n : ℝ) ≤ a * n) ∧
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → isGood (seq n)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2083_208312


namespace NUMINAMATH_CALUDE_larger_number_problem_l2083_208368

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 27) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2083_208368


namespace NUMINAMATH_CALUDE_race_distance_l2083_208302

/-- Represents the race scenario where p runs x% faster than q, q has a head start, and the race ends in a tie -/
def race_scenario (x y : ℝ) : Prop :=
  ∀ (vq : ℝ), vq > 0 →
    let vp := vq * (1 + x / 100)
    let head_start := (x / 10) * y
    let dq := 1000 * y / x
    let dp := dq + head_start
    dq / vq = dp / vp

/-- The theorem stating that both runners cover the same distance in the given race scenario -/
theorem race_distance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  race_scenario x y →
  ∃ (d : ℝ), d = 1000 * y / x ∧ 
    (∀ (vq : ℝ), vq > 0 →
      let vp := vq * (1 + x / 100)
      let head_start := (x / 10) * y
      d = 1000 * y / x ∧ d + head_start = (10000 * y + x * y^2) / (10 * x)) :=
sorry

end NUMINAMATH_CALUDE_race_distance_l2083_208302


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2083_208343

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2083_208343


namespace NUMINAMATH_CALUDE_tangent_to_ln_curve_l2083_208310

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ (∀ y : ℝ, y > 0 → k * y ≤ Real.log y)) → 
  k = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_to_ln_curve_l2083_208310


namespace NUMINAMATH_CALUDE_remainder_problem_l2083_208328

theorem remainder_problem (n : ℤ) (h : 2 * n % 15 = 2) : n % 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2083_208328


namespace NUMINAMATH_CALUDE_count_valid_bouquets_l2083_208316

/-- The number of valid bouquet combinations -/
def num_bouquets : ℕ := 11

/-- Represents a bouquet with roses and carnations -/
structure Bouquet where
  roses : ℕ
  carnations : ℕ

/-- The cost of a single rose -/
def rose_cost : ℕ := 4

/-- The cost of a single carnation -/
def carnation_cost : ℕ := 2

/-- The total budget for the bouquet -/
def total_budget : ℕ := 60

/-- Checks if a bouquet is valid according to the problem constraints -/
def is_valid_bouquet (b : Bouquet) : Prop :=
  b.roses ≥ 5 ∧
  b.roses * rose_cost + b.carnations * carnation_cost = total_budget

/-- The main theorem stating that there are exactly 11 valid bouquet combinations -/
theorem count_valid_bouquets :
  (∃ (bouquets : Finset Bouquet),
    bouquets.card = num_bouquets ∧
    (∀ b ∈ bouquets, is_valid_bouquet b) ∧
    (∀ b : Bouquet, is_valid_bouquet b → b ∈ bouquets)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_bouquets_l2083_208316


namespace NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l2083_208370

theorem tan_30_plus_4cos_30 :
  Real.tan (30 * π / 180) + 4 * Real.cos (30 * π / 180) = 7 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_30_plus_4cos_30_l2083_208370


namespace NUMINAMATH_CALUDE_largest_good_and_smallest_bad_l2083_208360

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_and_smallest_bad :
  (is_good_number 576) ∧
  (∀ M : ℕ, M ≥ 577 → ¬(is_good_number M)) ∧
  (¬(is_good_number 443)) ∧
  (∀ M : ℕ, 288 < M ∧ M ≤ 442 → is_good_number M) :=
by sorry

end NUMINAMATH_CALUDE_largest_good_and_smallest_bad_l2083_208360


namespace NUMINAMATH_CALUDE_profit_revenue_relationship_l2083_208324

/-- Represents the financial data of a company over two years -/
structure CompanyFinancials where
  prevRevenue : ℝ
  prevProfit : ℝ
  currRevenue : ℝ
  currProfit : ℝ

/-- The theorem stating the relationship between profits and revenues -/
theorem profit_revenue_relationship (c : CompanyFinancials)
  (h1 : c.currRevenue = 0.8 * c.prevRevenue)
  (h2 : c.currProfit = 0.2 * c.currRevenue)
  (h3 : c.currProfit = 1.6000000000000003 * c.prevProfit) :
  c.prevProfit / c.prevRevenue = 0.1 := by
  sorry

#check profit_revenue_relationship

end NUMINAMATH_CALUDE_profit_revenue_relationship_l2083_208324


namespace NUMINAMATH_CALUDE_square_area_equals_side_perimeter_l2083_208333

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_area_equals_side_perimeter :
  ∀ s : ℝ, s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_side_perimeter_l2083_208333


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2083_208342

theorem square_sum_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2083_208342


namespace NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l2083_208388

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon 
  (ABCDEFGH : RegularOctagon) 
  (A E C : ℝ × ℝ) 
  (hA : A = ABCDEFGH.vertices 0)
  (hE : E = ABCDEFGH.vertices 4)
  (hC : C = ABCDEFGH.vertices 2) :
  angle_measure A E C = 112.5 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l2083_208388


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l2083_208300

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The smallest positive integer k(m) satisfying the Fibonacci periodicity modulo m -/
def k_m (m : ℕ) : ℕ := sorry

theorem fibonacci_periodicity (m : ℕ) (h : m > 0) :
  (∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ m^2 ∧ fib i % m = fib j % m ∧ fib (i + 1) % m = fib (j + 1) % m) ∧
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, fib (n + k) % m = fib n % m) ∧
  (∀ n : ℕ, fib (n + k_m m) % m = fib n % m) ∧
  (fib (k_m m) % m = 0 ∧ fib (k_m m + 1) % m = 1) ∧
  (∀ k : ℕ, k > 0 → (∀ n : ℕ, fib (n + k) % m = fib n % m) ↔ k_m m ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l2083_208300


namespace NUMINAMATH_CALUDE_price_adjustment_l2083_208362

theorem price_adjustment (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_l2083_208362


namespace NUMINAMATH_CALUDE_coin_box_problem_l2083_208313

theorem coin_box_problem :
  ∃ (N B : ℕ), 
    N = 9 * (B - 2) ∧
    N - 6 * (B - 3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_coin_box_problem_l2083_208313


namespace NUMINAMATH_CALUDE_ahmed_has_13_goats_l2083_208344

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_has_13_goats : ahmed_goats = 13 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_has_13_goats_l2083_208344


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2083_208393

theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -1]
  5 • v1 - 3 • v2 + v3 = ![20, -44] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2083_208393


namespace NUMINAMATH_CALUDE_problem_solution_l2083_208311

theorem problem_solution (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 5 = 103 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2083_208311


namespace NUMINAMATH_CALUDE_inequality_proof_l2083_208357

theorem inequality_proof (a b x y : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2083_208357


namespace NUMINAMATH_CALUDE_parabola_focus_l2083_208340

/-- The focus of a parabola with equation x = 4y^2 is at (1/16, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (x = 4 * y^2) → (∃ p : ℝ, p > 0 ∧ x = y^2 / (4 * p) ∧ (1 / (16 : ℝ), 0) = (p, 0)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2083_208340


namespace NUMINAMATH_CALUDE_marks_increase_ratio_class_marks_double_l2083_208320

/-- Given a class of students, prove that if their marks are increased by a certain ratio,
    the ratio of new marks to original marks can be determined by the new and old averages. -/
theorem marks_increase_ratio (n : ℕ) (old_avg new_avg : ℚ) :
  n > 0 →
  old_avg > 0 →
  new_avg > old_avg →
  (n * new_avg) / (n * old_avg) = new_avg / old_avg := by sorry

/-- In a class of 11 students, if the average marks increase from 36 to 72,
    prove that the ratio of new marks to original marks is 2. -/
theorem class_marks_double :
  let n : ℕ := 11
  let old_avg : ℚ := 36
  let new_avg : ℚ := 72
  (n * new_avg) / (n * old_avg) = 2 := by sorry

end NUMINAMATH_CALUDE_marks_increase_ratio_class_marks_double_l2083_208320


namespace NUMINAMATH_CALUDE_choir_group_division_l2083_208327

theorem choir_group_division (sopranos altos tenors basses : ℕ) 
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18) :
  ∃ (n : ℕ), n = 3 ∧ 
  n > 0 ∧
  sopranos % n = 0 ∧ 
  altos % n = 0 ∧ 
  tenors % n = 0 ∧ 
  basses % n = 0 ∧
  sopranos / n < (altos + tenors + basses) / n ∧
  ∀ m : ℕ, m > n → 
    (sopranos % m ≠ 0 ∨ 
     altos % m ≠ 0 ∨ 
     tenors % m ≠ 0 ∨ 
     basses % m ≠ 0 ∨
     sopranos / m ≥ (altos + tenors + basses) / m) :=
by sorry

end NUMINAMATH_CALUDE_choir_group_division_l2083_208327


namespace NUMINAMATH_CALUDE_smallest_k_is_two_l2083_208331

/-- A five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Predicate to check if a number has digits in non-decreasing order -/
def hasNonDecreasingDigits (n : FiveDigitNumber) : Prop := sorry

/-- Predicate to check if two numbers have at least one digit in common -/
def hasCommonDigit (n m : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers satisfying the problem conditions -/
def SpecialSet (k : ℕ) : Set FiveDigitNumber := sorry

theorem smallest_k_is_two :
  ∀ k : ℕ,
    (∀ n : FiveDigitNumber, hasNonDecreasingDigits n →
      ∃ m ∈ SpecialSet k, hasCommonDigit n m) →
    k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_two_l2083_208331


namespace NUMINAMATH_CALUDE_abs_inequality_l2083_208341

theorem abs_inequality (a b : ℝ) (h : a^2 + b^2 ≤ 4) :
  |3 * a^2 - 8 * a * b - 3 * b^2| ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l2083_208341


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2083_208335

def A : Set ℝ := {x | |x| * (x^2 - 4*x + 3) < 0}

def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2083_208335


namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l2083_208397

theorem polar_to_rectangular_coordinates (r : ℝ) (θ : ℝ) :
  r = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l2083_208397


namespace NUMINAMATH_CALUDE_video_views_proof_l2083_208348

/-- Calculates the total views of a video given initial views and subsequent increases -/
def total_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) : ℕ :=
  initial_views + increase_factor * initial_views + additional_views

/-- Theorem stating that given the specific conditions, the total views equal 94000 -/
theorem video_views_proof :
  let initial_views : ℕ := 4000
  let increase_factor : ℕ := 10
  let additional_views : ℕ := 50000
  total_views initial_views increase_factor additional_views = 94000 := by
  sorry

#eval total_views 4000 10 50000

end NUMINAMATH_CALUDE_video_views_proof_l2083_208348


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_300_l2083_208359

theorem rectangle_area_with_inscribed_circle : ℝ → ℝ → ℝ → Prop :=
  fun radius ratio area =>
    let width := 2 * radius
    let length := ratio * width
    area = length * width

theorem rectangle_area_is_300 :
  ∃ (radius ratio : ℝ),
    radius = 5 ∧
    ratio = 3 ∧
    rectangle_area_with_inscribed_circle radius ratio 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_300_l2083_208359
