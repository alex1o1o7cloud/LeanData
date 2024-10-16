import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_foci_l23_2365

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- The coordinates of a focus -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem stating that the foci of the given ellipse are at (±4, 0) -/
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_l23_2365


namespace NUMINAMATH_CALUDE_angle_inequality_l23_2334

theorem angle_inequality (α β γ : Real) 
  (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : γ < π) :
  Real.sin (α / 2) + Real.sin (β / 2) > Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l23_2334


namespace NUMINAMATH_CALUDE_fourth_plus_fifth_sum_l23_2341

/-- A geometric sequence with a negative common ratio satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  q_neg : q < 0
  second_term : a 2 = 1 - a 1
  fourth_term : a 4 = 4 - a 3
  geom_seq : ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the fourth and fifth terms of the geometric sequence is -8 -/
theorem fourth_plus_fifth_sum (seq : GeometricSequence) : seq.a 4 + seq.a 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_plus_fifth_sum_l23_2341


namespace NUMINAMATH_CALUDE_divisor_difference_greater_than_sqrt_l23_2396

theorem divisor_difference_greater_than_sqrt (A B : ℕ) 
  (h1 : A > 1) 
  (h2 : B ∣ A^2 + 1) 
  (h3 : B > A) : 
  B - A > Real.sqrt A :=
sorry

end NUMINAMATH_CALUDE_divisor_difference_greater_than_sqrt_l23_2396


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l23_2363

/-- Given vectors a and b in ℝ², if a + b is perpendicular to b, then the second component of a is 9. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (5, m)
  let b : ℝ × ℝ := (2, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 9 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l23_2363


namespace NUMINAMATH_CALUDE_typists_pages_time_relation_l23_2302

/-- Given that 10 typists can type 25 pages in 5 minutes, 
    prove that 2 typists can type 2 pages in 2 minutes. -/
theorem typists_pages_time_relation : 
  ∀ (n : ℕ), 
    (10 : ℝ) * (25 : ℝ) / (5 : ℝ) = n * (2 : ℝ) / (2 : ℝ) → 
    n = 2 :=
by sorry

end NUMINAMATH_CALUDE_typists_pages_time_relation_l23_2302


namespace NUMINAMATH_CALUDE_curve_transformation_l23_2399

theorem curve_transformation (x y : ℝ) :
  x^2 + y^2 = 1 → 4 * (x/2)^2 + (2*y)^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l23_2399


namespace NUMINAMATH_CALUDE_account_balance_after_transfer_l23_2300

/-- Given an initial account balance and an amount transferred out, 
    calculate the final account balance. -/
def final_balance (initial : ℕ) (transferred : ℕ) : ℕ :=
  initial - transferred

theorem account_balance_after_transfer :
  final_balance 27004 69 = 26935 := by
  sorry

end NUMINAMATH_CALUDE_account_balance_after_transfer_l23_2300


namespace NUMINAMATH_CALUDE_logarithmic_identity_l23_2313

theorem logarithmic_identity (a b : ℝ) (h1 : a^2 + b^2 = 7*a*b) (h2 : a*b ≠ 0) :
  Real.log (|a + b| / 3) = (1/2) * (Real.log |a| + Real.log |b|) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_identity_l23_2313


namespace NUMINAMATH_CALUDE_checkerboard_probability_l23_2388

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 9

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def prob_non_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability :
  prob_non_perimeter = 49 / 81 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l23_2388


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l23_2320

-- Define the complex number z
variable (z : ℂ)

-- Define the equation (4+3i)z = |3-4i|
def equation : Prop := (4 + 3*Complex.I) * z = Complex.abs (3 - 4*Complex.I)

-- Theorem statement
theorem imaginary_part_of_z (h : equation z) : z.im = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l23_2320


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l23_2354

theorem sum_of_reciprocals_of_roots (a b c d : ℝ) (z₁ z₂ z₃ z₄ : ℂ) : 
  z₁^4 + a*z₁^3 + b*z₁^2 + c*z₁ + d = 0 ∧
  z₂^4 + a*z₂^3 + b*z₂^2 + c*z₂ + d = 0 ∧
  z₃^4 + a*z₃^3 + b*z₃^2 + c*z₃ + d = 0 ∧
  z₄^4 + a*z₄^3 + b*z₄^2 + c*z₄ + d = 0 ∧
  Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1 ∧ Complex.abs z₃ = 1 ∧ Complex.abs z₄ = 1 →
  1/z₁ + 1/z₂ + 1/z₃ + 1/z₄ = -a := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l23_2354


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l23_2392

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 3 = 3 →
  a 4 + a 6 = 6 →
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l23_2392


namespace NUMINAMATH_CALUDE_reachable_points_characterization_l23_2339

-- Define the road as a line
def Road : Type := ℝ

-- Define a point in the 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the tourist's starting point A
def A : Point := ⟨0, 0⟩

-- Define the tourist's speed on the road
def roadSpeed : ℝ := 6

-- Define the tourist's speed on the field
def fieldSpeed : ℝ := 3

-- Define the time limit
def timeLimit : ℝ := 1

-- Define the set of reachable points
def ReachablePoints : Set Point :=
  {p : Point | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ timeLimit ∧
    (p.x^2 / roadSpeed^2 + p.y^2 / fieldSpeed^2 ≤ t^2)}

-- Define the line segment on the road
def RoadSegment : Set Point :=
  {p : Point | p.y = 0 ∧ |p.x| ≤ roadSpeed * timeLimit}

-- Define the semicircles
def Semicircles : Set Point :=
  {p : Point | ∃ (c : ℝ), 
    c = roadSpeed * timeLimit ∧
    ((p.x - c)^2 + p.y^2 ≤ (fieldSpeed * timeLimit)^2 ∨
     (p.x + c)^2 + p.y^2 ≤ (fieldSpeed * timeLimit)^2) ∧
    p.y ≥ 0}

-- Theorem statement
theorem reachable_points_characterization :
  ReachablePoints = RoadSegment ∪ Semicircles :=
sorry

end NUMINAMATH_CALUDE_reachable_points_characterization_l23_2339


namespace NUMINAMATH_CALUDE_triangle_angle_bisectors_l23_2306

/-- Given a triangle ABC with sides a, b, and c, this theorem proves the formulas for the lengths of its angle bisectors. -/
theorem triangle_angle_bisectors 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (a₁ b₁ cc₁ : ℝ),
    a₁ = a * c / (a + b) ∧
    b₁ = b * c / (a + b) ∧
    cc₁^2 = a * b * (1 - c^2 / (a + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bisectors_l23_2306


namespace NUMINAMATH_CALUDE_empty_subset_of_disjoint_nonempty_l23_2325

theorem empty_subset_of_disjoint_nonempty (A B : Set α) :
  A ≠ ∅ → A ∩ B = ∅ → ∅ ⊆ B := by sorry

end NUMINAMATH_CALUDE_empty_subset_of_disjoint_nonempty_l23_2325


namespace NUMINAMATH_CALUDE_function_symmetry_property_l23_2393

open Real

/-- Given a function f(x) = a cos(x) + bx² + 2, prove that
    f(2016) - f(-2016) + f''(2017) + f''(-2017) = 0 for any real a and b -/
theorem function_symmetry_property (a b : ℝ) :
  let f := fun x => a * cos x + b * x^2 + 2
  let f'' := fun x => -a * cos x + 2 * b
  f 2016 - f (-2016) + f'' 2017 + f'' (-2017) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_property_l23_2393


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l23_2321

/-- Calculates the intensity of a paint mixture when a fraction of the original paint is replaced with a different paint. -/
def mixedPaintIntensity (originalIntensity addedIntensity : ℝ) (fractionReplaced : ℝ) : ℝ :=
  originalIntensity * (1 - fractionReplaced) + addedIntensity * fractionReplaced

/-- Theorem stating that mixing 45% intensity paint with 25% intensity paint in a 3:1 ratio results in 40% intensity paint. -/
theorem paint_mixture_intensity :
  let originalIntensity : ℝ := 0.45
  let addedIntensity : ℝ := 0.25
  let fractionReplaced : ℝ := 0.25
  mixedPaintIntensity originalIntensity addedIntensity fractionReplaced = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_intensity_l23_2321


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l23_2308

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (7 * x + 2) * (4 : ℝ) ^ (2 * x + 5) = (8 : ℝ) ^ (5 * x + 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l23_2308


namespace NUMINAMATH_CALUDE_smallest_n_for_factorial_sum_l23_2333

def lastFourDigits (n : ℕ) : ℕ := n % 10000

def isValidSequence (seq : List ℕ) : Prop :=
  ∀ x ∈ seq, x ≤ 15 ∧ x > 0

theorem smallest_n_for_factorial_sum : 
  (∃ (seq : List ℕ), 
    seq.length = 3 ∧ 
    isValidSequence seq ∧ 
    lastFourDigits (seq.map Nat.factorial).sum = 2001) ∧ 
  (∀ (n : ℕ) (seq : List ℕ), 
    n < 3 → 
    seq.length = n → 
    isValidSequence seq → 
    lastFourDigits (seq.map Nat.factorial).sum ≠ 2001) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorial_sum_l23_2333


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l23_2385

theorem consecutive_page_numbers : ∃ (n : ℕ), 
  n > 0 ∧ 
  n * (n + 1) = 20412 ∧ 
  n + (n + 1) = 283 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l23_2385


namespace NUMINAMATH_CALUDE_percentage_left_approx_20_l23_2383

-- Define the initial population
def initial_population : ℕ := 4675

-- Define the percentage of people who died
def death_percentage : ℚ := 5 / 100

-- Define the final population
def final_population : ℕ := 3553

-- Define the function to calculate the percentage who left
def percentage_left (init : ℕ) (death_perc : ℚ) (final : ℕ) : ℚ :=
  let remaining := init - (init * death_perc).floor
  ((remaining - final : ℚ) / remaining) * 100

-- Theorem statement
theorem percentage_left_approx_20 :
  ∃ ε > 0, abs (percentage_left initial_population death_percentage final_population - 20) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_left_approx_20_l23_2383


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l23_2345

theorem quadrilateral_inequality (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : a + b > e) (h2 : c + d > e) (h3 : a + d > f) (h4 : b + c > f) :
  (a + b + c + d) * (e + f) > 2 * (e^2 + f^2) := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l23_2345


namespace NUMINAMATH_CALUDE_trigonometric_identity_l23_2338

theorem trigonometric_identity (θ : Real) (h : Real.tan θ = 3) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l23_2338


namespace NUMINAMATH_CALUDE_min_floor_sum_l23_2387

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  34 ≤ ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l23_2387


namespace NUMINAMATH_CALUDE_unique_prime_solution_l23_2311

theorem unique_prime_solution :
  ∃! (p q : ℕ) (n : ℕ), 
    Prime p ∧ Prime q ∧ n > 1 ∧
    (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) ∧
    p = 2 ∧ q = 5 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l23_2311


namespace NUMINAMATH_CALUDE_intersection_condition_m_values_l23_2395

theorem intersection_condition_m_values (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - x - 6 = 0}
  let B : Set ℝ := {x | x * m - 1 = 0}
  (A ∩ B = B) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_m_values_l23_2395


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l23_2323

theorem quadratic_inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, 2*x^2 + b*x + a > 0 ↔ x < -1 ∨ x > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l23_2323


namespace NUMINAMATH_CALUDE_completing_square_l23_2371

theorem completing_square (x : ℝ) : x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l23_2371


namespace NUMINAMATH_CALUDE_expression_evaluation_l23_2359

theorem expression_evaluation : 
  (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 
  (Real.log (Real.sqrt 2) / Real.log 10 + Real.log (Real.sqrt 5) / Real.log 10) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l23_2359


namespace NUMINAMATH_CALUDE_binomial_product_l23_2351

theorem binomial_product (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l23_2351


namespace NUMINAMATH_CALUDE_concrete_pillars_amount_l23_2352

/-- Calculates the concrete needed for supporting pillars with environmental factors --/
def concrete_for_pillars (total_concrete : ℝ) (roadway_concrete : ℝ) (anchor_concrete : ℝ) (env_factor : ℝ) : ℝ :=
  let total_anchor_concrete := 2 * anchor_concrete
  let initial_pillar_concrete := total_concrete - roadway_concrete - total_anchor_concrete
  let pillar_increase := initial_pillar_concrete * env_factor
  initial_pillar_concrete + pillar_increase

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem concrete_pillars_amount : 
  concrete_for_pillars 4800 1600 700 0.05 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_concrete_pillars_amount_l23_2352


namespace NUMINAMATH_CALUDE_division_remainder_problem_l23_2335

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1620)
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l23_2335


namespace NUMINAMATH_CALUDE_share_distribution_l23_2361

theorem share_distribution (a b c : ℕ) : 
  a + b + c = 1010 →
  (a - 25) * 2 = (b - 10) * 3 →
  (a - 25) * 5 = (c - 15) * 3 →
  c = 495 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l23_2361


namespace NUMINAMATH_CALUDE_complex_real_condition_l23_2346

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m + 2*I) / (3 - 4*I)
  (∃ (x : ℝ), z = x) → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l23_2346


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l23_2337

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l23_2337


namespace NUMINAMATH_CALUDE_divisible_by_2520_l23_2386

theorem divisible_by_2520 (n : ℕ) : ∃ k : ℤ, (n^7 : ℤ) - 14*(n^5 : ℤ) + 49*(n^3 : ℤ) - 36*(n : ℤ) = 2520 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2520_l23_2386


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l23_2367

/-- A circle with diameter endpoints C(2, 3) and D(8, 9) on a coordinate plane -/
structure Circle where
  C : ℝ × ℝ
  D : ℝ × ℝ
  h1 : C = (2, 3)
  h2 : D = (8, 9)

/-- The area of the circle -/
def area (c : Circle) : ℝ := sorry

/-- The circumference of the circle -/
def circumference (c : Circle) : ℝ := sorry

theorem circle_area_and_circumference (c : Circle) : 
  area c = 18 * Real.pi ∧ circumference c = 6 * Real.sqrt 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_and_circumference_l23_2367


namespace NUMINAMATH_CALUDE_copper_part_mass_l23_2348

/-- Given two parts with equal volume, one made of aluminum and one made of copper,
    prove that the mass of the copper part is approximately 0.086 kg. -/
theorem copper_part_mass
  (ρ_A : Real) -- density of aluminum
  (ρ_M : Real) -- density of copper
  (Δm : Real)  -- mass difference between parts
  (h1 : ρ_A = 2700) -- density of aluminum in kg/m³
  (h2 : ρ_M = 8900) -- density of copper in kg/m³
  (h3 : Δm = 0.06)  -- mass difference in kg
  : ∃ (m_M : Real), abs (m_M - 0.086) < 0.001 ∧ 
    ∃ (V : Real), V > 0 ∧ V = m_M / ρ_M ∧ V = (m_M - Δm) / ρ_A :=
by sorry

end NUMINAMATH_CALUDE_copper_part_mass_l23_2348


namespace NUMINAMATH_CALUDE_individual_egg_price_is_50_l23_2304

/-- The price per individual egg in cents -/
def individual_egg_price : ℕ := sorry

/-- The number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- The price of a tray of eggs in cents -/
def tray_price : ℕ := 1200

/-- The savings per egg when buying a tray, in cents -/
def savings_per_egg : ℕ := 10

theorem individual_egg_price_is_50 : 
  individual_egg_price = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_individual_egg_price_is_50_l23_2304


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l23_2327

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) := x * Real.log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
  (∀ y : ℝ, y > Real.exp (-1) → ¬ StrictMonoOn f (Set.Ioo 0 y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l23_2327


namespace NUMINAMATH_CALUDE_journey_time_difference_l23_2342

theorem journey_time_difference 
  (speed : ℝ) 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_difference_l23_2342


namespace NUMINAMATH_CALUDE_g_of_3_eq_neg_1_l23_2384

def g (x : ℝ) : ℝ := 2 * (x - 2)^2 - 3 * (x - 2)

theorem g_of_3_eq_neg_1 : g 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_neg_1_l23_2384


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l23_2391

theorem opposite_of_negative_2023 : ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l23_2391


namespace NUMINAMATH_CALUDE_root_difference_of_arithmetic_progression_l23_2397

-- Define the polynomial coefficients
def a : ℝ := 81
def b : ℝ := -171
def c : ℝ := 107
def d : ℝ := -18

-- Define the polynomial
def p (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_difference_of_arithmetic_progression :
  ∃ (r₁ r₂ r₃ : ℝ),
    -- The roots satisfy the polynomial equation
    p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧
    -- The roots are in arithmetic progression
    r₂ - r₁ = r₃ - r₂ ∧
    -- The difference between the largest and smallest roots is approximately 1.66
    abs (max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) - 1.66) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_root_difference_of_arithmetic_progression_l23_2397


namespace NUMINAMATH_CALUDE_stock_price_increase_l23_2316

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 25 / 100) * (1 + 30 / 100) = 117 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l23_2316


namespace NUMINAMATH_CALUDE_roberta_record_listening_time_l23_2319

theorem roberta_record_listening_time :
  let x : ℕ := 8  -- initial number of records
  let y : ℕ := 12 -- additional records received
  let z : ℕ := 30 -- records bought
  let t : ℕ := 2  -- time needed to listen to each record in days
  (x + y + z) * t = 100 := by sorry

end NUMINAMATH_CALUDE_roberta_record_listening_time_l23_2319


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l23_2324

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 1} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l23_2324


namespace NUMINAMATH_CALUDE_factorization_proof_l23_2317

theorem factorization_proof (m n : ℝ) : 4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l23_2317


namespace NUMINAMATH_CALUDE_number_problem_l23_2349

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l23_2349


namespace NUMINAMATH_CALUDE_largest_k_for_2_pow_15_l23_2309

/-- The sum of k consecutive odd integers starting from 2m + 1 -/
def sumConsecutiveOdds (m k : ℕ) : ℕ := k * (2 * m + k)

/-- Proposition: The largest value of k for which 2^15 is expressible as the sum of k consecutive odd integers is 128 -/
theorem largest_k_for_2_pow_15 : 
  (∃ (m : ℕ), sumConsecutiveOdds m 128 = 2^15) ∧ 
  (∀ (k : ℕ), k > 128 → ¬∃ (m : ℕ), sumConsecutiveOdds m k = 2^15) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_for_2_pow_15_l23_2309


namespace NUMINAMATH_CALUDE_magician_works_two_weeks_l23_2379

/-- Calculates the number of weeks a magician works given their hourly rate, daily hours, and total payment. -/
def magician_weeks_worked (hourly_rate : ℚ) (daily_hours : ℚ) (total_payment : ℚ) : ℚ :=
  total_payment / (hourly_rate * daily_hours * 7)

/-- Theorem stating that a magician charging $60 per hour, working 3 hours per day, and receiving $2520 in total works for 2 weeks. -/
theorem magician_works_two_weeks :
  magician_weeks_worked 60 3 2520 = 2 := by
  sorry

end NUMINAMATH_CALUDE_magician_works_two_weeks_l23_2379


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l23_2350

-- Define the total number of male and female students
def total_male : ℕ := 500
def total_female : ℕ := 400

-- Define the number of male students selected
def selected_male : ℕ := 25

-- Define the function to calculate the number of female students to be selected
def female_to_select : ℕ := (selected_male * total_female) / total_male

-- Theorem statement
theorem stratified_sampling_ratio :
  female_to_select = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l23_2350


namespace NUMINAMATH_CALUDE_product_of_roots_l23_2344

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → ∃ y : ℝ, (x + 3) * (x - 5) = 22 ∧ (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l23_2344


namespace NUMINAMATH_CALUDE_existence_of_hundredth_square_l23_2307

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a square that can be cut out from the grid -/
structure Square :=
  (size : ℕ)
  (position : ℕ × ℕ)

/-- The total number of 2×2 squares that can fit in a grid -/
def total_squares (g : Grid) : ℕ :=
  (g.size - 1) * (g.size - 1)

/-- Predicate to check if a square can be cut out from the grid -/
def can_cut_square (g : Grid) (s : Square) : Prop :=
  s.size = 2 ∧ 
  s.position.1 ≤ g.size - 1 ∧ 
  s.position.2 ≤ g.size - 1

theorem existence_of_hundredth_square (g : Grid) (cut_squares : Finset Square) :
  g.size = 29 →
  cut_squares.card = 99 →
  (∀ s ∈ cut_squares, can_cut_square g s) →
  ∃ s : Square, can_cut_square g s ∧ s ∉ cut_squares :=
sorry

end NUMINAMATH_CALUDE_existence_of_hundredth_square_l23_2307


namespace NUMINAMATH_CALUDE_expression_values_l23_2378

theorem expression_values (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (gcd_ab : Nat.gcd a b = 1) (gcd_bc : Nat.gcd b c = 1) (gcd_ca : Nat.gcd c a = 1) :
  (a + b) / c + (b + c) / a + (c + a) / b = 7 ∨ (a + b) / c + (b + c) / a + (c + a) / b = 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l23_2378


namespace NUMINAMATH_CALUDE_ac_equals_twelve_l23_2312

theorem ac_equals_twelve (a b c d : ℝ) 
  (h1 : a = 2 * b)
  (h2 : c = d * b)
  (h3 : d + d = b * c)
  (h4 : d = 3) : 
  a * c = 12 := by
sorry

end NUMINAMATH_CALUDE_ac_equals_twelve_l23_2312


namespace NUMINAMATH_CALUDE_cube_expansion_equals_3375000_l23_2357

theorem cube_expansion_equals_3375000 (y : ℝ) (h : y = 50) : 
  y^3 + 3*y^2*(2*y) + 3*y*(2*y)^2 + (2*y)^3 = 3375000 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_equals_3375000_l23_2357


namespace NUMINAMATH_CALUDE_candies_remaining_l23_2390

/-- The number of candies remaining after Carlos ate all the yellow candies -/
def remaining_candies (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ :=
  red + blue

/-- Theorem stating the number of remaining candies given the problem conditions -/
theorem candies_remaining :
  ∀ (red : ℕ) (yellow : ℕ) (blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  remaining_candies red yellow blue = 90 := by
sorry

end NUMINAMATH_CALUDE_candies_remaining_l23_2390


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l23_2380

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - min initial_pairs shoes_lost

/-- Theorem stating that with 27 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 18. -/
theorem marcella_shoes_theorem :
  max_remaining_pairs 27 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l23_2380


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l23_2336

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l23_2336


namespace NUMINAMATH_CALUDE_ducks_joining_l23_2303

theorem ducks_joining (original : ℕ) (total : ℕ) (joined : ℕ) : 
  original = 13 → total = 33 → joined = total - original → joined = 20 := by
sorry

end NUMINAMATH_CALUDE_ducks_joining_l23_2303


namespace NUMINAMATH_CALUDE_arc_length_calculation_l23_2364

theorem arc_length_calculation (r θ : Real) (h1 : r = 2) (h2 : θ = π/3) :
  r * θ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l23_2364


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l23_2373

theorem sqrt_sum_simplification : Real.sqrt 3600 + Real.sqrt 1600 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l23_2373


namespace NUMINAMATH_CALUDE_face_covers_are_squares_and_rectangles_l23_2310

/-- A parallelogram covering a face of a unit cube -/
structure FaceCover where
  -- The parallelogram's area
  area : ℝ
  -- The parallelogram is a square
  is_square : Prop
  -- The parallelogram is a rectangle
  is_rectangle : Prop

/-- A cube with edge length 1 covered by six identical parallelograms -/
structure CoveredCube where
  -- The edge length of the cube
  edge_length : ℝ
  -- The six identical parallelograms covering the cube
  face_covers : Fin 6 → FaceCover
  -- All face covers are identical
  covers_identical : ∀ (i j : Fin 6), face_covers i = face_covers j
  -- The edge length is 1
  edge_is_unit : edge_length = 1
  -- Each face cover has an area of 1
  cover_area_is_unit : ∀ (i : Fin 6), (face_covers i).area = 1

/-- Theorem: All face covers of a unit cube are squares and rectangles -/
theorem face_covers_are_squares_and_rectangles (cube : CoveredCube) :
  (∀ (i : Fin 6), (cube.face_covers i).is_square) ∧
  (∀ (i : Fin 6), (cube.face_covers i).is_rectangle) := by
  sorry


end NUMINAMATH_CALUDE_face_covers_are_squares_and_rectangles_l23_2310


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l23_2377

theorem unique_number_with_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3) ∧
  n = 62 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l23_2377


namespace NUMINAMATH_CALUDE_certain_number_problem_l23_2374

theorem certain_number_problem (x : ℕ) (n : ℕ) : x = 4 → 3 * x + n = 48 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l23_2374


namespace NUMINAMATH_CALUDE_fraction_power_equality_l23_2375

theorem fraction_power_equality (x y : ℚ) 
  (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3 : ℚ) * x^7 * y^8 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l23_2375


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l23_2370

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h_conv : abs r < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → (r = 1/4 ∨ r = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l23_2370


namespace NUMINAMATH_CALUDE_power_of_i_product_l23_2362

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_product : i^45 * i^105 = -1 := by sorry

end NUMINAMATH_CALUDE_power_of_i_product_l23_2362


namespace NUMINAMATH_CALUDE_y_coordinate_of_point_p_l23_2314

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def foci_distance : ℝ := 6

-- Define the sum of distances from P to foci
def sum_distances_to_foci : ℝ := 10

-- Define the radius of the inscribed circle
def inscribed_circle_radius : ℝ := 1

-- Main theorem
theorem y_coordinate_of_point_p :
  ∀ x y : ℝ,
  is_on_ellipse x y →
  x ≥ 0 →
  y > 0 →
  y = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_point_p_l23_2314


namespace NUMINAMATH_CALUDE_unique_n_value_l23_2372

/-- Represents a round-robin golf tournament with the given conditions -/
structure GolfTournament where
  /-- Total number of players -/
  T : ℕ
  /-- Number of points scored by each player other than Simon and Garfunkle -/
  n : ℕ
  /-- Condition: Total number of matches equals total points distributed -/
  matches_eq_points : T * (T - 1) / 2 = 16 + n * (T - 2)
  /-- Condition: Tournament has at least 3 players -/
  min_players : T ≥ 3

/-- Theorem stating that the only possible value for n is 17 -/
theorem unique_n_value (tournament : GolfTournament) : tournament.n = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_value_l23_2372


namespace NUMINAMATH_CALUDE_min_rows_correct_l23_2356

/-- The minimum number of rows required to seat students under given conditions -/
def min_rows (total_students : ℕ) (max_per_school : ℕ) (seats_per_row : ℕ) : ℕ :=
  -- Definition to be proved
  15

theorem min_rows_correct (total_students max_per_school seats_per_row : ℕ) 
  (h1 : total_students = 2016)
  (h2 : max_per_school = 40)
  (h3 : seats_per_row = 168)
  (h4 : ∀ (school_size : ℕ), school_size ≤ max_per_school → school_size ≤ seats_per_row) :
  min_rows total_students max_per_school seats_per_row = 15 := by
  sorry

#eval min_rows 2016 40 168

end NUMINAMATH_CALUDE_min_rows_correct_l23_2356


namespace NUMINAMATH_CALUDE_high_jump_probabilities_l23_2301

/-- The probability of clearing the height in a single jump -/
def p : ℝ := 0.8

/-- The probability of clearing the height on two consecutive jumps -/
def prob_two_consecutive : ℝ := p * p

/-- The probability of clearing the height for the first time on the third attempt -/
def prob_third_attempt : ℝ := (1 - p) * (1 - p) * p

/-- The minimum number of attempts required to clear the height with a 99% probability -/
def min_attempts : ℕ := 3

/-- Theorem stating the probabilities and minimum attempts -/
theorem high_jump_probabilities :
  prob_two_consecutive = 0.64 ∧
  prob_third_attempt = 0.032 ∧
  min_attempts = 3 ∧
  (1 - (1 - p) ^ min_attempts ≥ 0.99) :=
by sorry

end NUMINAMATH_CALUDE_high_jump_probabilities_l23_2301


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l23_2340

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ  -- x^2 coefficient
  b : ℝ  -- y^2 coefficient
  c : ℝ  -- x coefficient
  d : ℝ  -- y coefficient
  e : ℝ  -- constant term

/-- Represents a line in the 2D plane -/
structure Line where
  a : ℝ  -- x coefficient
  b : ℝ  -- y coefficient
  c : ℝ  -- constant term

/-- Definition of the first circle -/
def circle1 : Circle := { a := 1, b := 1, c := -2, d := 0, e := -4 }

/-- Definition of the second circle -/
def circle2 : Circle := { a := 1, b := 1, c := 0, d := 2, e := -6 }

/-- The common chord line -/
def commonChord : Line := { a := 1, b := 1, c := -1 }

/-- Theorem: The given line is the common chord of the two circles -/
theorem common_chord_of_circles :
  commonChord = Line.mk 1 1 (-1) ∧
  (∀ x y : ℝ, x + y - 1 = 0 →
    (x^2 + y^2 - 2*x - 4 = 0 ↔ x^2 + y^2 + 2*y - 6 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l23_2340


namespace NUMINAMATH_CALUDE_smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l23_2330

theorem smallest_a_for_sqrt_12a_integer (a : ℕ) : 
  (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

theorem three_satisfies_condition : 
  ∃ (n : ℕ), n > 0 ∧ n^2 = 12*3 :=
sorry

theorem three_is_smallest : 
  ∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l23_2330


namespace NUMINAMATH_CALUDE_prob_one_success_value_min_institutes_l23_2318

-- Define the probabilities of success for each institute
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

-- Define the probability of exactly one institute succeeding
def prob_one_success : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

-- Define the function to calculate the probability of at least one success
-- given n institutes with probability p
def prob_at_least_one (n : ℕ) (p : ℚ) : ℚ := 1 - (1 - p)^n

-- Theorem 1: The probability of exactly one institute succeeding is 11/24
theorem prob_one_success_value : prob_one_success = 11/24 := by sorry

-- Theorem 2: The minimum number of institutes with success probability 1/3
-- needed to achieve at least 99/100 overall success probability is 12
theorem min_institutes : 
  (∀ n < 12, prob_at_least_one n (1/3) < 99/100) ∧ 
  prob_at_least_one 12 (1/3) ≥ 99/100 := by sorry

end NUMINAMATH_CALUDE_prob_one_success_value_min_institutes_l23_2318


namespace NUMINAMATH_CALUDE_opposite_is_five_l23_2322

theorem opposite_is_five (x : ℝ) : -x = 5 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_is_five_l23_2322


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l23_2343

/-- The number of handshakes exchanged at a gathering of couples -/
def handshakes_at_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that for 6 couples, the number of handshakes is 54 -/
theorem six_couples_handshakes :
  handshakes_at_gathering 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l23_2343


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l23_2355

theorem isosceles_triangle_area : 
  ∀ a b : ℕ,
  a > 0 ∧ b > 0 →
  2 * a + b = 12 →
  (a + a > b ∧ a + b > a) →
  (∃ (s : ℝ), s * s = (a * a : ℝ) - (b * b / 4 : ℝ)) →
  (a * s / 2 : ℝ) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l23_2355


namespace NUMINAMATH_CALUDE_zeros_of_f_l23_2398

def f (x : ℝ) : ℝ := (x^2 - 3*x) * (x + 4)

theorem zeros_of_f : {x : ℝ | f x = 0} = {0, 3, -4} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l23_2398


namespace NUMINAMATH_CALUDE_car_speed_problem_l23_2329

/-- Proves that a car traveling for two hours with an average speed of 80 km/h
    and a second hour speed of 90 km/h must have a first hour speed of 70 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 90 →
  average_speed = 80 →
  average_speed = (first_hour_speed + second_hour_speed) / 2 →
  first_hour_speed = 70 := by
sorry


end NUMINAMATH_CALUDE_car_speed_problem_l23_2329


namespace NUMINAMATH_CALUDE_manny_has_more_ten_bills_l23_2328

-- Define the number of bills each person has
def mandy_twenty_bills : ℕ := 3
def manny_fifty_bills : ℕ := 2

-- Define the value of each bill type
def twenty_bill_value : ℕ := 20
def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10

-- Calculate the total value for each person
def mandy_total : ℕ := mandy_twenty_bills * twenty_bill_value
def manny_total : ℕ := manny_fifty_bills * fifty_bill_value

-- Calculate the number of $10 bills each person can get
def mandy_ten_bills : ℕ := mandy_total / ten_bill_value
def manny_ten_bills : ℕ := manny_total / ten_bill_value

-- State the theorem
theorem manny_has_more_ten_bills : manny_ten_bills - mandy_ten_bills = 4 := by
  sorry

end NUMINAMATH_CALUDE_manny_has_more_ten_bills_l23_2328


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l23_2315

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  (max (abs a) (abs b) = abs (max a b)) :=
sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l23_2315


namespace NUMINAMATH_CALUDE_factor_4t_squared_minus_64_l23_2358

theorem factor_4t_squared_minus_64 (t : ℝ) : 4 * t^2 - 64 = 4 * (t - 4) * (t + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_4t_squared_minus_64_l23_2358


namespace NUMINAMATH_CALUDE_sum_equation_proof_l23_2326

theorem sum_equation_proof (N : ℕ) : 
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999 = 8000 - N → N = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_proof_l23_2326


namespace NUMINAMATH_CALUDE_new_tram_properties_l23_2382

/-- Represents the properties of a tram journey -/
structure TramJourney where
  distance : ℝ
  old_time : ℝ
  new_time : ℝ
  old_speed : ℝ
  new_speed : ℝ

/-- Theorem stating the properties of the new tram journey -/
theorem new_tram_properties (j : TramJourney) 
  (h1 : j.distance = 20)
  (h2 : j.new_time = j.old_time - 1/5)
  (h3 : j.new_speed = j.old_speed + 5)
  (h4 : j.distance = j.old_speed * j.old_time)
  (h5 : j.distance = j.new_speed * j.new_time) :
  j.new_time = 4/5 ∧ j.new_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_new_tram_properties_l23_2382


namespace NUMINAMATH_CALUDE_bridget_apples_l23_2353

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 4 + 4 = x → x = 22 :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_l23_2353


namespace NUMINAMATH_CALUDE_adjacent_sum_negative_total_sum_positive_l23_2347

theorem adjacent_sum_negative_total_sum_positive :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ),
    (a₁ + a₂ < 0) ∧
    (a₂ + a₃ < 0) ∧
    (a₃ + a₄ < 0) ∧
    (a₄ + a₅ < 0) ∧
    (a₅ + a₁ < 0) ∧
    (a₁ + a₂ + a₃ + a₄ + a₅ > 0) :=
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_negative_total_sum_positive_l23_2347


namespace NUMINAMATH_CALUDE_population_growth_factors_l23_2331

/-- Represents a population of organisms -/
structure Population where
  density : ℝ
  genotypeFrequency : ℝ
  kValue : ℝ

/-- Factors affecting population growth -/
inductive GrowthFactor
  | BirthRate
  | DeathRate
  | CarryingCapacity

/-- Represents ideal conditions for population growth -/
def idealConditions : Prop := sorry

/-- Main factors affecting population growth under ideal conditions -/
def mainFactors : Set GrowthFactor := sorry

theorem population_growth_factors :
  idealConditions →
  mainFactors = {GrowthFactor.BirthRate, GrowthFactor.DeathRate} ∧
  GrowthFactor.CarryingCapacity ∉ mainFactors :=
sorry

end NUMINAMATH_CALUDE_population_growth_factors_l23_2331


namespace NUMINAMATH_CALUDE_ios_department_larger_l23_2360

theorem ios_department_larger (n m : ℕ) : 
  (7 * n + 15 * m = 15 * n + 9 * m) → m > n := by
  sorry

end NUMINAMATH_CALUDE_ios_department_larger_l23_2360


namespace NUMINAMATH_CALUDE_circle_area_through_points_l23_2332

/-- The area of a circle with center P(2, -1) passing through Q(-4, 6) is 85π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 85 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l23_2332


namespace NUMINAMATH_CALUDE_sector_max_area_l23_2394

/-- 
Given a sector with circumference c, this theorem states that:
1. The maximum area of the sector is c^2/16
2. The maximum area occurs when the arc length is c/2
-/
theorem sector_max_area (c : ℝ) (h : c > 0) :
  ∃ (max_area arc_length : ℝ),
    max_area = c^2 / 16 ∧
    arc_length = c / 2 ∧
    ∀ (area : ℝ), area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l23_2394


namespace NUMINAMATH_CALUDE_last_term_of_gp_l23_2369

theorem last_term_of_gp (a : ℝ) (r : ℝ) (S : ℝ) (n : ℕ) :
  a = 9 →
  r = 1/3 →
  S = 40/3 →
  S = a * (1 - r^n) / (1 - r) →
  a * r^(n-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_last_term_of_gp_l23_2369


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4519_l23_2381

theorem largest_prime_factor_of_4519 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4519 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4519 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4519_l23_2381


namespace NUMINAMATH_CALUDE_vector_magnitude_l23_2305

def a : Fin 2 → ℝ := ![(-2 : ℝ), 1]
def b (k : ℝ) : Fin 2 → ℝ := ![k, -3]
def c : Fin 2 → ℝ := ![1, 2]

theorem vector_magnitude (k : ℝ) :
  (∀ i : Fin 2, (a i - 2 * b k i) * c i = 0) →
  Real.sqrt ((b k 0)^2 + (b k 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l23_2305


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_one_l23_2389

/-- An arithmetic sequence with specific first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + 2*y
  | 1 => x - y
  | 2 => 2*x*y
  | 3 => x / (2*y)
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the specific arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one :
  let x : ℚ := 4
  let y : ℚ := 1
  arithmetic_sequence x y 4 = -1 := by sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_one_l23_2389


namespace NUMINAMATH_CALUDE_power_inequality_l23_2376

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^b * b^c * c^a ≤ a^a * b^b * c^c := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l23_2376


namespace NUMINAMATH_CALUDE_least_bench_sections_l23_2368

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that the least positive integer N such that N bench sections can hold
    an equal number of adults and children is 3, given that one bench section
    holds 8 adults or 12 children. -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h1 : capacity.adults = 8)
    (h2 : capacity.children = 12) :
    ∃ N : Nat, N > 0 ∧ N * capacity.adults = N * capacity.children ∧
    ∀ M : Nat, M > 0 → M * capacity.adults = M * capacity.children → N ≤ M :=
  by sorry

end NUMINAMATH_CALUDE_least_bench_sections_l23_2368


namespace NUMINAMATH_CALUDE_spinner_probability_l23_2366

theorem spinner_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l23_2366
