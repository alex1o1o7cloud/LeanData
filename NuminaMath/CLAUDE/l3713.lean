import Mathlib

namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l3713_371320

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 140 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l3713_371320


namespace NUMINAMATH_CALUDE_largest_square_area_in_isosceles_triangle_l3713_371309

/-- The area of the largest square that can be cut from an isosceles triangle -/
theorem largest_square_area_in_isosceles_triangle 
  (base height : ℝ) 
  (h_base : base = 2) 
  (h_height : height = 3) :
  let side_length := (2 * base * height) / (base + height)
  (side_length^2 : ℝ) = 36 / 25 := by sorry

end NUMINAMATH_CALUDE_largest_square_area_in_isosceles_triangle_l3713_371309


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3713_371323

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = -2 + 3*Complex.I) → k = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3713_371323


namespace NUMINAMATH_CALUDE_point_coordinates_l3713_371337

def is_valid_point (x y : ℝ) : Prop :=
  |y| = 1 ∧ |x| = 2

theorem point_coordinates :
  ∀ x y : ℝ, is_valid_point x y ↔ (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3713_371337


namespace NUMINAMATH_CALUDE_recurrence_sequence_eventually_periodic_l3713_371397

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → u n = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

/-- A sequence is bounded if there exist m and M such that m ≤ u_n ≤ M for all n -/
def IsBounded (u : ℕ → ℤ) : Prop :=
  ∃ m M : ℤ, ∀ n : ℕ, m ≤ u n ∧ u n ≤ M

/-- A sequence is eventually periodic if there exist N and p such that u_{n+p} = u_n for all n ≥ N -/
def EventuallyPeriodic (u : ℕ → ℤ) : Prop :=
  ∃ N p : ℕ, p > 0 ∧ ∀ n : ℕ, n ≥ N → u (n + p) = u n

/-- The main theorem: a bounded recurrence sequence is eventually periodic -/
theorem recurrence_sequence_eventually_periodic (u : ℕ → ℤ) 
  (h_recurrence : RecurrenceSequence u) (h_bounded : IsBounded u) : 
  EventuallyPeriodic u :=
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_eventually_periodic_l3713_371397


namespace NUMINAMATH_CALUDE_library_books_calculation_l3713_371341

theorem library_books_calculation (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 60 → 
  return_rate = 7/10 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 57 := by
sorry

end NUMINAMATH_CALUDE_library_books_calculation_l3713_371341


namespace NUMINAMATH_CALUDE_store_posters_l3713_371353

theorem store_posters (P : ℕ) : 
  (2 : ℚ) / 5 * P + (1 : ℚ) / 2 * P + 5 = P → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_store_posters_l3713_371353


namespace NUMINAMATH_CALUDE_hash_four_six_l3713_371354

-- Define the operation #
def hash (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem hash_four_six : hash 4 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_six_l3713_371354


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_l3713_371347

theorem binomial_coefficient_1000 : 
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_l3713_371347


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3713_371329

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3713_371329


namespace NUMINAMATH_CALUDE_soda_cost_l3713_371382

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℚ
  soda : ℚ
  fries : ℚ

/-- The total cost of Keegan's purchase in cents -/
def keegan_total (c : Cost) : ℚ := 3 * c.burger + 2 * c.soda + c.fries

/-- The total cost of Alex's purchase in cents -/
def alex_total (c : Cost) : ℚ := 2 * c.burger + 3 * c.soda + c.fries

theorem soda_cost :
  ∃ (c : Cost),
    keegan_total c = 975 ∧
    alex_total c = 900 ∧
    c.soda = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l3713_371382


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_minus_one_l3713_371359

theorem solution_set_x_abs_x_minus_one (x : ℝ) :
  {x : ℝ | x * |x - 1| > 0} = {x : ℝ | 0 < x ∧ x ≠ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_minus_one_l3713_371359


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l3713_371314

/-- Given the coordinates of point P(x, y) satisfying the following conditions,
    prove that the locus of P is a hyperbola. -/
theorem locus_is_hyperbola
  (a c : ℝ)
  (x y θ₁ θ₂ : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (h4 : c > 1) :
  ∃ (k m n : ℝ), y^2 = k * x^2 + m * x + n ∧ k > 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l3713_371314


namespace NUMINAMATH_CALUDE_inequality_proof_l3713_371312

theorem inequality_proof (a r : ℝ) (n : ℕ) 
  (ha : a ≥ -2) (hr : r ≥ 0) (hn : n ≥ 1) :
  r^(2*n) + a*r^n + 1 ≥ (1 - r)^(2*n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3713_371312


namespace NUMINAMATH_CALUDE_max_value_expression_l3713_371388

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 2*a*b*c + 1) :
  (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b) ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3713_371388


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3713_371306

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p p' : Point2D) : Prop :=
  p'.x = p.x ∧ p'.y = -p.y

/-- Theorem: If P(-3, 2) is symmetric to P' with respect to the x-axis,
    then P' has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let P : Point2D := ⟨-3, 2⟩
  let P' : Point2D := ⟨-3, -2⟩
  symmetricXAxis P P' → P' = ⟨-3, -2⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3713_371306


namespace NUMINAMATH_CALUDE_college_students_count_l3713_371345

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) :
  boys + girls = 1040 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3713_371345


namespace NUMINAMATH_CALUDE_not_hearing_favorite_song_probability_l3713_371396

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Calculates the duration of the nth song in the sequence -/
def nthSongDuration (n : ℕ) : SongDuration :=
  45 + 15 * n

/-- Generates a playlist of 12 songs with increasing durations -/
def generatePlaylist : Playlist :=
  List.range 12 |>.map nthSongDuration

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total duration we're interested in (5 minutes in seconds) -/
def totalDuration : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song 
    within the first 5 minutes of a random playlist -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) (favoriteDuration : SongDuration) (totalDuration : SongDuration) : ℚ :=
  sorry

theorem not_hearing_favorite_song_probability :
  probabilityNotHearingFavoriteSong generatePlaylist favoriteSongDuration totalDuration = 65 / 66 := by
  sorry

end NUMINAMATH_CALUDE_not_hearing_favorite_song_probability_l3713_371396


namespace NUMINAMATH_CALUDE_womens_average_age_l3713_371334

theorem womens_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) : 
  n = 6 ∧ 
  n * A - 10 - 12 + W₁ + W₂ = n * (A + 2) → 
  (W₁ + W₂) / 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_womens_average_age_l3713_371334


namespace NUMINAMATH_CALUDE_movie_attendance_l3713_371378

theorem movie_attendance (total_cost concession_cost child_ticket adult_ticket : ℕ)
  (num_children : ℕ) (h1 : total_cost = 76) (h2 : concession_cost = 12)
  (h3 : child_ticket = 7) (h4 : adult_ticket = 10) (h5 : num_children = 2) :
  (total_cost - concession_cost - num_children * child_ticket) / adult_ticket = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_attendance_l3713_371378


namespace NUMINAMATH_CALUDE_tangent_circle_existence_l3713_371331

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line as a pair of points
structure Line where
  point1 : Point
  point2 : Point

-- Define tangency between two circles at a point
def CircleTangentToCircle (S S' : Circle) (A : Point) : Prop :=
  -- The centers of S and S' and point A are collinear
  sorry

-- Define tangency between a circle and a line at a point
def CircleTangentToLine (S : Circle) (l : Line) (B : Point) : Prop :=
  -- The radius of S at B is perpendicular to l
  sorry

-- The main theorem
theorem tangent_circle_existence 
  (S : Circle) (A : Point) (l : Line) : 
  ∃ (S' : Circle) (B : Point), 
    CircleTangentToCircle S S' A ∧ 
    CircleTangentToLine S' l B :=
  sorry

end NUMINAMATH_CALUDE_tangent_circle_existence_l3713_371331


namespace NUMINAMATH_CALUDE_pam_total_fruits_l3713_371302

-- Define the given conditions
def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4
def gerald_apple_bags : ℕ := 5
def gerald_orange_bags : ℕ := 4
def gerald_apples_per_bag : ℕ := 30
def gerald_oranges_per_bag : ℕ := 25
def pam_apple_ratio : ℕ := 3
def pam_orange_ratio : ℕ := 2

-- Theorem to prove
theorem pam_total_fruits :
  pam_apple_bags * (pam_apple_ratio * gerald_apples_per_bag) +
  pam_orange_bags * (pam_orange_ratio * gerald_oranges_per_bag) = 740 := by
  sorry


end NUMINAMATH_CALUDE_pam_total_fruits_l3713_371302


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3713_371338

-- Define the sets
def set1 (a b : ℝ) : Set ℝ := {1, a, b/a}
def set2 (a b : ℝ) : Set ℝ := {0, a^2, a+b}

-- State the theorem
theorem set_equality_implies_sum (a b : ℝ) :
  set1 a b = set2 a b → a^2013 + b^2012 = -1 :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3713_371338


namespace NUMINAMATH_CALUDE_least_period_is_30_l3713_371387

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def least_common_positive_period (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
  ∀ q : ℝ, 0 < q ∧ q < p → ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬is_period f q

theorem least_period_is_30 :
  least_common_positive_period 30 := by sorry

end NUMINAMATH_CALUDE_least_period_is_30_l3713_371387


namespace NUMINAMATH_CALUDE_nancy_savings_l3713_371369

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (quarters : ℕ) : 
  quarters = dozen → (quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l3713_371369


namespace NUMINAMATH_CALUDE_exterior_angle_parallel_lines_l3713_371370

theorem exterior_angle_parallel_lines (α β γ δ : ℝ) : 
  α = 40 → β = 40 → γ + δ = 180 → α + β + γ = 180 → δ = 80 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_parallel_lines_l3713_371370


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l3713_371310

theorem cubic_polynomial_roots (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1 ∧ x ≠ 2 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l3713_371310


namespace NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3713_371373

theorem x_minus_p_in_terms_of_p (x p : ℝ) : 
  (|x - 3| = p + 1) → (x < 3) → (x - p = 2 - 2*p) := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3713_371373


namespace NUMINAMATH_CALUDE_no_xy_term_when_k_is_3_l3713_371330

/-- The polynomial that we're analyzing -/
def polynomial (x y k : ℝ) : ℝ := -x^2 - 3*k*x*y - 3*y^2 + 9*x*y - 8

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := -3*k + 9

theorem no_xy_term_when_k_is_3 :
  ∃ (k : ℝ), xy_coefficient k = 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_no_xy_term_when_k_is_3_l3713_371330


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l3713_371307

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 →
  not_enrolled = 528 →
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l3713_371307


namespace NUMINAMATH_CALUDE_combination_ratio_problem_l3713_371322

theorem combination_ratio_problem (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_combination_ratio_problem_l3713_371322


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3713_371372

-- Define atomic weights
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_C : ℝ := 12.011

-- Define number of atoms for each element
def num_H : ℕ := 4
def num_Cr : ℕ := 2
def num_O : ℕ := 4
def num_N : ℕ := 3
def num_C : ℕ := 5

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_Cr : ℝ) * atomic_weight_Cr +
  (num_O : ℝ) * atomic_weight_O +
  (num_N : ℝ) * atomic_weight_N +
  (num_C : ℝ) * atomic_weight_C

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight = 274.096 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3713_371372


namespace NUMINAMATH_CALUDE_randy_bats_count_l3713_371350

theorem randy_bats_count :
  ∀ (gloves bats : ℕ),
    gloves = 29 →
    gloves = 7 * bats + 1 →
    bats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_randy_bats_count_l3713_371350


namespace NUMINAMATH_CALUDE_min_value_alpha_gamma_l3713_371335

open Complex

theorem min_value_alpha_gamma (f : ℂ → ℂ) (α γ : ℂ) :
  (∀ z, f z = (4 + I) * z^2 + α * z + γ) →
  (f 1).im = 0 →
  (f I).im = 0 →
  ∃ (α₀ γ₀ : ℂ), abs α₀ + abs γ₀ = Real.sqrt 2 ∧ 
    ∀ (α' γ' : ℂ), (∀ z, f z = (4 + I) * z^2 + α' * z + γ') →
      (f 1).im = 0 → (f I).im = 0 → abs α' + abs γ' ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_alpha_gamma_l3713_371335


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3713_371376

theorem sum_of_coefficients (a b : ℝ) : 
  (Nat.choose 6 4) * a^4 * b^2 = 135 →
  (Nat.choose 6 5) * a^5 * b = -18 →
  (a + b)^6 = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3713_371376


namespace NUMINAMATH_CALUDE_company_price_ratio_l3713_371304

/-- Given companies A, B, and KW where:
    - KW's price is 30% more than A's assets
    - KW's price is 100% more than B's assets
    Prove that KW's price is approximately 78.79% of A and B's combined assets -/
theorem company_price_ratio (P A B : ℝ) 
  (h1 : P = A + 0.3 * A) 
  (h2 : P = B + B) : 
  ∃ ε > 0, abs (P / (A + B) - 0.7879) < ε :=
sorry

end NUMINAMATH_CALUDE_company_price_ratio_l3713_371304


namespace NUMINAMATH_CALUDE_original_triangle_area_l3713_371368

theorem original_triangle_area
  (original : Real)  -- Area of the original triangle
  (new : Real)       -- Area of the new triangle
  (h1 : new = 256)   -- The area of the new triangle is 256 square feet
  (h2 : new = 16 * original)  -- The new triangle's area is 16 times the original
  : original = 16 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3713_371368


namespace NUMINAMATH_CALUDE_function_properties_l3713_371361

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - 0 < φ < π
    - The distance between two adjacent zeros of f(x) is π/2
    - g(x) is f(x) shifted left by π/6 units
    - g(x) is an even function
    
    This theorem states that:
    1. f(x) = sin(2x + π/6)
    2. The axis of symmetry is x = kπ/2 + π/6 for k ∈ ℤ
    3. The interval of monotonic increase is [kπ - π/3, kπ + π/6] for k ∈ ℤ -/
theorem function_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (f : ℝ → ℝ) (hf : f = fun x ↦ Real.sin (ω * x + φ))
  (h_zeros : ∀ x₁ x₂, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → |x₁ - x₂| = π / 2)
  (g : ℝ → ℝ) (hg : g = fun x ↦ f (x + π / 6))
  (h_even : ∀ x, g x = g (-x)) :
  (f = fun x ↦ Real.sin (2 * x + π / 6)) ∧
  (∀ k : ℤ, ∃ x, x = k * π / 2 + π / 6 ∧ ∀ y, f (2 * x - y) = f (2 * x + y)) ∧
  (∀ k : ℤ, ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → Monotone (f ∘ (fun y ↦ y + x))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3713_371361


namespace NUMINAMATH_CALUDE_count_integers_in_range_l3713_371356

theorem count_integers_in_range : 
  (Finset.range 1001).card = (Finset.Ico 1000 2001).card := by sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l3713_371356


namespace NUMINAMATH_CALUDE_complex_number_equalities_l3713_371364

/-- Prove complex number equalities -/
theorem complex_number_equalities :
  let i : ℂ := Complex.I
  let z₁ : ℂ := (1 + 2*i)^2 + 3*(1 - i)
  let z₂ : ℂ := 2 + i
  let z₃ : ℂ := 1 - i
  let z₄ : ℂ := 1 + i
  let z₅ : ℂ := 1 - Complex.I * Real.sqrt 3
  let z₆ : ℂ := Complex.I * Real.sqrt 3 + i
  (z₁ / z₂ = 1/5 + 2/5*i) ∧
  (z₃ / z₄^2 + z₄ / z₃^2 = -1) ∧
  (z₅ / z₆^2 = -1/4 - (Real.sqrt 3)/4*i) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equalities_l3713_371364


namespace NUMINAMATH_CALUDE_star_3_2_l3713_371392

-- Define the ★ operation
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

-- Theorem statement
theorem star_3_2 : star 3 2 = 125 := by sorry

end NUMINAMATH_CALUDE_star_3_2_l3713_371392


namespace NUMINAMATH_CALUDE_library_book_redistribution_l3713_371357

theorem library_book_redistribution (total_boxes : Nat) (books_per_box : Nat) (new_box_capacity : Nat) :
  total_boxes = 1421 →
  books_per_box = 27 →
  new_box_capacity = 35 →
  (total_boxes * books_per_box) % new_box_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_library_book_redistribution_l3713_371357


namespace NUMINAMATH_CALUDE_additional_rook_possible_l3713_371349

/-- Represents a 10x10 chessboard -/
def Board := Fin 10 → Fin 10 → Bool

/-- Checks if a rook at position (x, y) attacks another rook at position (x', y') -/
def attacks (x y x' y' : Fin 10) : Prop :=
  x = x' ∨ y = y'

/-- Represents a valid rook placement on the board -/
def ValidPlacement (b : Board) : Prop :=
  ∃ (n : Nat) (positions : Fin n → Fin 10 × Fin 10),
    n ≤ 8 ∧
    (∀ i j, i ≠ j → ¬attacks (positions i).1 (positions i).2 (positions j).1 (positions j).2) ∧
    (∀ i, b (positions i).1 (positions i).2 = true) ∧
    (∃ (blackCount whiteCount : Nat),
      blackCount = whiteCount ∧
      blackCount + whiteCount = n ∧
      (∀ i, (((positions i).1 + (positions i).2) % 2 = 0) = (i < blackCount)))

/-- The main theorem stating that an additional rook can be placed -/
theorem additional_rook_possible (b : Board) (h : ValidPlacement b) :
  ∃ (x y : Fin 10), b x y = false ∧ ∀ (x' y' : Fin 10), b x' y' = true → ¬attacks x y x' y' := by
  sorry

end NUMINAMATH_CALUDE_additional_rook_possible_l3713_371349


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l3713_371358

theorem raffle_ticket_sales (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) 
  (total_tickets : ℕ) (female_tickets : ℕ) :
  total_members > 0 →
  male_members > 0 →
  female_members = 2 * male_members →
  total_members = male_members + female_members →
  (total_tickets : ℚ) / total_members = 66 →
  (female_tickets : ℚ) / female_members = 70 →
  (total_tickets - female_tickets : ℚ) / male_members = 66 :=
by sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l3713_371358


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3713_371342

theorem hotel_room_charges (G R P : ℝ) 
  (h1 : R = G * (1 + 0.60))
  (h2 : P = R * (1 - 0.50)) :
  P = G * (1 - 0.20) := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3713_371342


namespace NUMINAMATH_CALUDE_absolute_value_zero_l3713_371390

theorem absolute_value_zero (x : ℚ) : |4*x + 6| = 0 ↔ x = -3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_zero_l3713_371390


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_to_direction_l3713_371365

/-- A line parameterized by x = 3t + 3, y = 2t + 3 -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3*t + 3, 2*t + 3)

/-- The vector we want to prove is correct -/
def vector : ℝ × ℝ := (9, 6)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 2)

theorem vector_to_line_parallel_to_direction :
  ∃ (t : ℝ), parametric_line t = vector ∧ 
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_to_direction_l3713_371365


namespace NUMINAMATH_CALUDE_value_range_cos_x_tan_x_l3713_371363

-- Define the function f(x) = cos x tan x
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.tan x

-- Theorem statement
theorem value_range_cos_x_tan_x :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ Real.pi / 2 + Real.pi * ↑k ∧ f x = y) ↔ -1 < y ∧ y < 1 :=
by sorry

end NUMINAMATH_CALUDE_value_range_cos_x_tan_x_l3713_371363


namespace NUMINAMATH_CALUDE_distance_to_school_l3713_371332

def walking_speed : ℝ := 80
def travel_time : ℝ := 28

theorem distance_to_school :
  walking_speed * travel_time = 2240 := by sorry

end NUMINAMATH_CALUDE_distance_to_school_l3713_371332


namespace NUMINAMATH_CALUDE_investment_interest_rate_l3713_371399

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part : ℝ) 
  (second_rate : ℝ) 
  (total_interest : ℝ) :
  total_investment = 3000 →
  first_part = 300 →
  second_part = total_investment - first_part →
  second_rate = 5 →
  total_interest = 144 →
  total_interest = (first_part * (3 : ℝ) / 100) + (second_part * second_rate / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l3713_371399


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3713_371391

theorem inscribed_circle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 6) (h₃ : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3713_371391


namespace NUMINAMATH_CALUDE_element_uniquely_identified_l3713_371321

/-- Represents a 6x6 grid of distinct elements -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- The column of an element in the original grid -/
def OriginalColumn := Fin 6

/-- The column of an element in the new grid -/
def NewColumn := Fin 6

/-- Given a grid, an original column, and a new column, 
    returns the unique position of the element in both grids -/
def findElement (g : Grid) (oc : OriginalColumn) (nc : NewColumn) : 
  (Fin 6 × Fin 6) × (Fin 6 × Fin 6) :=
sorry

theorem element_uniquely_identified (g : Grid) (oc : OriginalColumn) (nc : NewColumn) :
  ∃! (p₁ p₂ : Fin 6 × Fin 6), 
    (findElement g oc nc).1 = p₁ ∧ 
    (findElement g oc nc).2 = p₂ ∧
    g p₁.1 p₁.2 = g p₂.2 p₂.1 :=
sorry

end NUMINAMATH_CALUDE_element_uniquely_identified_l3713_371321


namespace NUMINAMATH_CALUDE_triangle_side_length_l3713_371383

theorem triangle_side_length (n : ℕ) : 
  (7 + 11 + n > 35) ∧ 
  (7 + 11 > n) ∧ 
  (7 + n > 11) ∧ 
  (11 + n > 7) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3713_371383


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3713_371398

-- Define repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum : 
  repeating_234 - repeating_567 + repeating_891 = 186 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3713_371398


namespace NUMINAMATH_CALUDE_quarter_orbit_distance_l3713_371384

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ  -- Distance of nearest point from focus
  apogee : ℝ   -- Distance of farthest point from focus

/-- Calculates the distance from a point on the orbit to the focus -/
def distance_to_focus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  sorry

theorem quarter_orbit_distance (orbit : EllipticalOrbit) 
  (h1 : orbit.perigee = 3)
  (h2 : orbit.apogee = 15) :
  distance_to_focus orbit 0.25 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_quarter_orbit_distance_l3713_371384


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3713_371326

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^2 - 3*x + 1) - 4*(2*x^2 - 3*x + 5) = 8*x^3 - 14*x^2 + 14*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3713_371326


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3713_371385

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 1}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem complement_union_theorem : (U \ A) ∪ B = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3713_371385


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3713_371375

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation --/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Main theorem --/
theorem compound_interest_problem (principal : ℝ) :
  compound_interest principal 0.1 2 = 420 →
  total_amount principal 420 = 2420 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3713_371375


namespace NUMINAMATH_CALUDE_graduation_chairs_l3713_371386

/-- Calculates the total number of chairs needed for a graduation ceremony. -/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  graduates + (graduates * parents_per_graduate) + teachers + (teachers / 2)

/-- Proves that 180 chairs are needed for the given graduation ceremony. -/
theorem graduation_chairs : chairs_needed 50 2 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_graduation_chairs_l3713_371386


namespace NUMINAMATH_CALUDE_triangle_inequality_l3713_371324

theorem triangle_inequality (a b c : ℝ) : |a - c| ≤ |a - b| + |b - c| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3713_371324


namespace NUMINAMATH_CALUDE_negative_1234_mod_9_l3713_371316

theorem negative_1234_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_1234_mod_9_l3713_371316


namespace NUMINAMATH_CALUDE_geometric_series_sum_first_six_terms_l3713_371348

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_six_terms :
  let a : ℚ := 3
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometric_series_sum a r n = 364/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_first_six_terms_l3713_371348


namespace NUMINAMATH_CALUDE_inequality_problem_l3713_371339

theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) → 
  a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3713_371339


namespace NUMINAMATH_CALUDE_ratio_q_p_l3713_371317

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2 * (distinct_numbers - 2) : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem ratio_q_p : q / p = 360 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l3713_371317


namespace NUMINAMATH_CALUDE_matching_socks_probability_l3713_371367

/-- The number of gray-bottomed socks -/
def gray_socks : ℕ := 12

/-- The number of white-bottomed socks -/
def white_socks : ℕ := 10

/-- The total number of socks -/
def total_socks : ℕ := gray_socks + white_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of selecting a matching pair of socks -/
theorem matching_socks_probability :
  (choose_two gray_socks + choose_two white_socks : ℚ) / choose_two total_socks = 111 / 231 := by
  sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l3713_371367


namespace NUMINAMATH_CALUDE_concert_ticket_ratio_l3713_371394

theorem concert_ticket_ratio (initial_amount : ℚ) (motorcycle_cost : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 5000)
  (h2 : motorcycle_cost = 2800)
  (h3 : final_amount = 825)
  (h4 : ∃ (concert_cost : ℚ),
    final_amount = (initial_amount - motorcycle_cost - concert_cost) * (3/4)) :
  ∃ (concert_cost : ℚ),
    concert_cost / (initial_amount - motorcycle_cost) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_ratio_l3713_371394


namespace NUMINAMATH_CALUDE_inequality_proof_l3713_371395

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) :
  a * B + b * C + c * A ≤ k^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3713_371395


namespace NUMINAMATH_CALUDE_equal_principal_repayment_formula_l3713_371371

/-- Repayment amount for the nth month -/
def repayment_amount (n : ℕ) : ℚ :=
  3928 - 8 * n

/-- Properties of the loan -/
def loan_amount : ℚ := 480000
def repayment_years : ℕ := 20
def monthly_interest_rate : ℚ := 4 / 1000

theorem equal_principal_repayment_formula :
  ∀ n : ℕ, n > 0 → n ≤ repayment_years * 12 →
  repayment_amount n =
    loan_amount / (repayment_years * 12) +
    (loan_amount - (n - 1) * (loan_amount / (repayment_years * 12))) * monthly_interest_rate :=
by sorry

end NUMINAMATH_CALUDE_equal_principal_repayment_formula_l3713_371371


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l3713_371377

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a + b + c = 90 →         -- sum is 90
  b = 3 * a ∧ c = 5 * a →  -- ratio 2:3:5
  a = 10 :=                -- smallest integer is 10
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l3713_371377


namespace NUMINAMATH_CALUDE_initial_men_count_l3713_371319

/-- The number of men working initially -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the new group -/
def new_men : ℕ := 6

/-- The number of hours worked per day by the new group -/
def new_hours_per_day : ℕ := 20

/-- The number of days worked by the new group -/
def new_days : ℕ := 8

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  new_men * new_hours_per_day * new_days :=
by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l3713_371319


namespace NUMINAMATH_CALUDE_pencil_length_l3713_371343

/-- The total length of a pencil with purple, black, and blue sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h1 : purple_length = 3)
  (h2 : black_length = 2)
  (h3 : blue_length = 1) :
  purple_length + black_length + blue_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3713_371343


namespace NUMINAMATH_CALUDE_typistSalary_l3713_371346

/-- Calculates the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

/-- Theorem stating that the typist's final salary is 6270 Rs -/
theorem typistSalary :
  finalSalary 6000 10 5 = 6270 := by
  sorry

#eval finalSalary 6000 10 5

end NUMINAMATH_CALUDE_typistSalary_l3713_371346


namespace NUMINAMATH_CALUDE_sarahs_age_l3713_371303

theorem sarahs_age :
  ∀ (s : ℚ), -- Sarah's age
  (∃ (g : ℚ), -- Grandmother's age
    g = 10 * s ∧ -- Grandmother is ten times Sarah's age
    g - s = 60) -- Grandmother was 60 when Sarah was born
  → s = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l3713_371303


namespace NUMINAMATH_CALUDE_inner_triangle_area_l3713_371389

/-- Given a triangle ABC with sides a, b, c, and lines parallel to the sides drawn at a distance d from them,
    the area of the resulting inner triangle is (t - ds)^2 / t, where t is the area of ABC and s is its semi-perimeter. -/
theorem inner_triangle_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let s := (a + b + c) / 2
  let t := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inner_area := (t - d * s)^2 / t
  ∃ (inner_triangle_area : ℝ), inner_triangle_area = inner_area :=
by sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l3713_371389


namespace NUMINAMATH_CALUDE_profit_equation_correct_l3713_371381

/-- Represents the profit calculation for a bicycle sale --/
def profit_equation (x : ℝ) : Prop :=
  0.8 * (1 + 0.45) * x - x = 50

/-- Theorem stating that the profit equation correctly represents the given scenario --/
theorem profit_equation_correct (x : ℝ) : profit_equation x ↔ 
  (∃ (markup discount profit : ℝ),
    markup = 0.45 ∧
    discount = 0.2 ∧
    profit = 50 ∧
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_correct_l3713_371381


namespace NUMINAMATH_CALUDE_cyclists_speed_l3713_371311

/-- Proves that the cyclist's speed is 11 miles per hour given the problem conditions --/
theorem cyclists_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 13.75 / 60 →
  ∃ (cyclist_speed : ℝ), cyclist_speed = 11 :=
by
  sorry

#check cyclists_speed

end NUMINAMATH_CALUDE_cyclists_speed_l3713_371311


namespace NUMINAMATH_CALUDE_passing_marks_proof_l3713_371352

/-- Given an exam with total marks T and passing marks P, prove that P = 120 -/
theorem passing_marks_proof (T : ℝ) (P : ℝ) : 
  (0.30 * T = P - 30) →
  (0.45 * T = P + 15) →
  P = 120 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_proof_l3713_371352


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l3713_371313

/-- Calculates the average speed of a car trip given the following conditions:
  * The total trip duration is 6 hours
  * The car travels at an average speed of 75 mph for the first 4 hours
  * The car travels at an average speed of 60 mph for the remaining hours
-/
theorem car_trip_average_speed : 
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let second_part_time : ℝ := total_time - first_part_time
  let first_part_speed : ℝ := 75
  let second_part_speed : ℝ := 60
  let total_distance : ℝ := first_part_speed * first_part_time + second_part_speed * second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 70 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l3713_371313


namespace NUMINAMATH_CALUDE_plant_original_price_l3713_371340

/-- Given a 10% discount on a plant and a final price of $9, prove that the original price was $10. -/
theorem plant_original_price (discount_percentage : ℚ) (discounted_price : ℚ) : 
  discount_percentage = 10 →
  discounted_price = 9 →
  (1 - discount_percentage / 100) * 10 = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_plant_original_price_l3713_371340


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3713_371360

theorem bridget_apples : ℕ → Prop :=
  fun x =>
    let remaining_after_ann := x / 3
    let remaining_after_cassie := remaining_after_ann - 5
    let remaining_after_found := remaining_after_cassie + 3
    remaining_after_found = 6 → x = 24

-- Proof
theorem bridget_apples_proof : ∃ x : ℕ, bridget_apples x := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3713_371360


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3713_371393

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3713_371393


namespace NUMINAMATH_CALUDE_message_difference_l3713_371355

/-- 
Given:
- Alina sent fewer messages than Lucia on the first day
- Lucia sent 120 messages on the first day
- On the second day, Lucia sent 1/3 of her first day's messages
- On the second day, Alina doubled her first day's messages
- On the third day, they both sent the same number of messages as the first day
- The total number of messages over three days is 680

Prove that the difference between Lucia's and Alina's messages on the first day is 20.
-/
theorem message_difference (alina_day1 : ℕ) : 
  alina_day1 < 120 →
  alina_day1 + 120 + 2 * alina_day1 + 40 + alina_day1 + 120 = 680 →
  120 - alina_day1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_message_difference_l3713_371355


namespace NUMINAMATH_CALUDE_valid_y_characterization_l3713_371333

/-- The set of y values in [0, 2π] for which sin(x+y) ≥ sin(x) - sin(y) holds for all x in [0, 2π] -/
def valid_y_set : Set ℝ :=
  {y | 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.sin (x + y) ≥ Real.sin x - Real.sin y}

theorem valid_y_characterization :
  valid_y_set = {0, 2 * Real.pi} := by sorry

end NUMINAMATH_CALUDE_valid_y_characterization_l3713_371333


namespace NUMINAMATH_CALUDE_converse_of_square_equals_one_l3713_371305

theorem converse_of_square_equals_one (a : ℝ) : 
  (∀ a, a = 1 → a^2 = 1) → (∀ a, a^2 = 1 → a = 1) := by
  sorry

end NUMINAMATH_CALUDE_converse_of_square_equals_one_l3713_371305


namespace NUMINAMATH_CALUDE_fraction_less_than_one_necessary_not_sufficient_l3713_371318

theorem fraction_less_than_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ (∃ a, 1 / a < 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_necessary_not_sufficient_l3713_371318


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l3713_371300

theorem product_of_five_consecutive_not_square :
  ∀ a : ℕ, a > 0 →
    ¬∃ n : ℕ, a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l3713_371300


namespace NUMINAMATH_CALUDE_cube_opposite_sum_l3713_371351

/-- Represents the numbering of a cube's faces -/
structure CubeNumbering where
  numbers : Fin 6 → ℕ
  consecutive : ∀ i j, i < j → numbers j = numbers i + (j - i)
  opposite_sum_equal : ∃ s, ∀ i, numbers i + numbers (5 - i) = s

/-- The common sum of opposite faces for a valid cube numbering is 81 -/
theorem cube_opposite_sum (c : CubeNumbering) : 
  ∃ s, (∀ i, c.numbers i + c.numbers (5 - i) = s) ∧ s = 81 := by
  sorry

end NUMINAMATH_CALUDE_cube_opposite_sum_l3713_371351


namespace NUMINAMATH_CALUDE_twenty_percent_value_l3713_371336

theorem twenty_percent_value (x : ℝ) : 1.2 * x = 600 → 0.2 * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_value_l3713_371336


namespace NUMINAMATH_CALUDE_high_octane_half_cost_l3713_371327

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane : ℚ
  regular_octane : ℚ
  cost_ratio : ℚ
  total : ℚ

/-- Calculates the fraction of total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  (fuel.high_octane * fuel.cost_ratio) / ((fuel.high_octane * fuel.cost_ratio) + fuel.regular_octane)

/-- Theorem: For a fuel mixture with 15 parts high octane and 45 parts regular octane,
    where high octane costs 3 times as much as regular octane,
    the fraction of the total cost due to high octane is 1/2 -/
theorem high_octane_half_cost (fuel : FuelMixture)
  (h1 : fuel.high_octane = 15)
  (h2 : fuel.regular_octane = 45)
  (h3 : fuel.cost_ratio = 3)
  (h4 : fuel.total = fuel.high_octane + fuel.regular_octane) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_high_octane_half_cost_l3713_371327


namespace NUMINAMATH_CALUDE_expression_value_l3713_371325

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) :
  8 - 6 * a + 9 * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3713_371325


namespace NUMINAMATH_CALUDE_brother_ages_l3713_371308

theorem brother_ages (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 := by
  sorry

end NUMINAMATH_CALUDE_brother_ages_l3713_371308


namespace NUMINAMATH_CALUDE_complement_of_union_l3713_371379

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3713_371379


namespace NUMINAMATH_CALUDE_parabola_point_position_l3713_371380

theorem parabola_point_position 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_below : 2 < a + b + c) : 
  2 < c + b + a := by sorry

end NUMINAMATH_CALUDE_parabola_point_position_l3713_371380


namespace NUMINAMATH_CALUDE_no_standard_operation_satisfies_equation_l3713_371328

theorem no_standard_operation_satisfies_equation : ¬∃ (op : ℝ → ℝ → ℝ), 
  (op = (·+·) ∨ op = (·-·) ∨ op = (·*·) ∨ op = (·/·)) ∧ 
  (op 12 4) - 3 + (6 - 2) = 7 := by
sorry

end NUMINAMATH_CALUDE_no_standard_operation_satisfies_equation_l3713_371328


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3713_371315

/-- Given a quadratic inequality ax^2 + (a-1)x - 1 > 0 with solution set (-1, -1/2), prove that a = -2 -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + (a-1)*x - 1 > 0 ↔ -1 < x ∧ x < -1/2) → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3713_371315


namespace NUMINAMATH_CALUDE_negation_of_zero_product_implication_l3713_371301

theorem negation_of_zero_product_implication :
  (∀ x y : ℝ, xy = 0 → x = 0 ∨ y = 0) ↔
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_implication_l3713_371301


namespace NUMINAMATH_CALUDE_triangle_max_area_l3713_371374

/-- Given a triangle ABC with area S, prove that the maximum value of S is √3/4
    when 2S + √3(AB · AC) = 0 and |BC| = √3 -/
theorem triangle_max_area (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  2 * S + Real.sqrt 3 * (AB.1 * AC.1 + AB.2 * AC.2) = 0 →
  BC.1^2 + BC.2^2 = 3 →
  S ≤ Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3713_371374


namespace NUMINAMATH_CALUDE_max_sin_cos_product_l3713_371362

theorem max_sin_cos_product (x : Real) : 
  (∃ (y : Real), y = Real.sin x * Real.cos x ∧ ∀ (z : Real), z = Real.sin x * Real.cos x → z ≤ y) → 
  (∃ (max : Real), max = (1 : Real) / 2 ∧ ∀ (y : Real), y = Real.sin x * Real.cos x → y ≤ max) :=
sorry

end NUMINAMATH_CALUDE_max_sin_cos_product_l3713_371362


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_cubed_l3713_371366

theorem ceiling_negative_seven_fourths_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_cubed_l3713_371366


namespace NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3713_371344

/-- The number of days it takes for all pollywogs to disappear from the pond -/
def days_to_disappear (initial_pollywogs : ℕ) (maturation_rate : ℕ) (catching_rate : ℕ) (catching_duration : ℕ) : ℕ :=
  let combined_rate := maturation_rate + catching_rate
  let pollywogs_after_catching := initial_pollywogs - combined_rate * catching_duration
  let remaining_days := pollywogs_after_catching / maturation_rate
  catching_duration + remaining_days

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from the pond -/
theorem pollywogs_disappear_in_44_days :
  days_to_disappear 2400 50 10 20 = 44 := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l3713_371344
