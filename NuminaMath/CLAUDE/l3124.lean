import Mathlib

namespace NUMINAMATH_CALUDE_tangent_sum_over_cosine_l3124_312403

theorem tangent_sum_over_cosine (x : Real) :
  let a := x * π / 180  -- Convert degrees to radians
  (Real.tan a + Real.tan (2*a) + Real.tan (7*a) + Real.tan (8*a)) / Real.cos a = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_over_cosine_l3124_312403


namespace NUMINAMATH_CALUDE_lower_bound_of_fraction_l3124_312446

theorem lower_bound_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (3 * a) + 3 / b ≥ 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_lower_bound_of_fraction_l3124_312446


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l3124_312459

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.8 + 0.9 * total_land * 0.1 + 90 = 0.9 * total_land) →
  total_land = 1000 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l3124_312459


namespace NUMINAMATH_CALUDE_minimal_sum_distances_l3124_312419

noncomputable section

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Inverse point with respect to a circle -/
def inverse_point (c : Circle) (p : Point) : Point := sorry

/-- Line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Perpendicular bisector of two points -/
def perpendicular_bisector (p q : Point) : Line := sorry

/-- Intersection point of a line and a circle -/
def line_circle_intersection (l : Line) (c : Circle) : Option Point := sorry

/-- Theorem: Minimal sum of distances from two fixed points to a point on a circle -/
theorem minimal_sum_distances (c : Circle) (p q : Point) 
  (h1 : distance c.center p = distance c.center q) 
  (h2 : distance c.center p < c.radius ∧ distance c.center q < c.radius) :
  ∃ z : Point, 
    (distance c.center z = c.radius) ∧ 
    (∀ w : Point, distance c.center w = c.radius → 
      distance p z + distance q z ≤ distance p w + distance q w) :=
sorry

end

end NUMINAMATH_CALUDE_minimal_sum_distances_l3124_312419


namespace NUMINAMATH_CALUDE_bronze_to_silver_ratio_l3124_312410

def total_watches : ℕ := 88
def silver_watches : ℕ := 20
def gold_watches : ℕ := 9

def bronze_watches : ℕ := total_watches - silver_watches - gold_watches

theorem bronze_to_silver_ratio :
  bronze_watches * 20 = silver_watches * 59 := by sorry

end NUMINAMATH_CALUDE_bronze_to_silver_ratio_l3124_312410


namespace NUMINAMATH_CALUDE_percentage_problem_l3124_312474

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 130 = 65 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3124_312474


namespace NUMINAMATH_CALUDE_quadratic_value_l3124_312461

/-- A quadratic function with axis of symmetry at x = 3.5 and p(-6) = 0 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_value (d e f : ℝ) :
  (∀ x : ℝ, p d e f x = p d e f (7 - x)) →  -- Axis of symmetry at x = 3.5
  p d e f (-6) = 0 →                        -- p(-6) = 0
  ∃ n : ℤ, p d e f 13 = n →                 -- p(13) is an integer
  p d e f 13 = 0 :=                         -- Conclusion: p(13) = 0
by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l3124_312461


namespace NUMINAMATH_CALUDE_sum_of_roots_l3124_312435

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 18*a^2 + 75*a - 200 = 0)
  (hb : 8*b^3 - 72*b^2 - 350*b + 3200 = 0) : 
  a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3124_312435


namespace NUMINAMATH_CALUDE_arbitrarily_large_N_exists_l3124_312423

def is_increasing_seq (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x n < x (n + 1)

def limit_zero (x : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n / n| < ε

theorem arbitrarily_large_N_exists (x : ℕ → ℝ) 
  (h_pos : ∀ n, x n > 0)
  (h_inc : is_increasing_seq x)
  (h_lim : limit_zero x) :
  ∀ M : ℕ, ∃ N > M, ∀ i : ℕ, 1 ≤ i → i < N → x i + x (2*N - i) < 2 * x N :=
sorry

end NUMINAMATH_CALUDE_arbitrarily_large_N_exists_l3124_312423


namespace NUMINAMATH_CALUDE_r_geq_one_l3124_312480

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x

def g (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m

def r (m : ℝ) (x : ℝ) : ℝ := 1 / f x + (4 * m * x) / g m x

theorem r_geq_one (h1 : m > 0) (h2 : x ≥ 0) : r m x ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_r_geq_one_l3124_312480


namespace NUMINAMATH_CALUDE_smallest_nonzero_y_value_l3124_312421

theorem smallest_nonzero_y_value (y : ℝ) : 
  y > 0 ∧ Real.sqrt (6 * y + 3) = 3 * y + 1 → y ≥ Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonzero_y_value_l3124_312421


namespace NUMINAMATH_CALUDE_polynomial_arrangement_l3124_312494

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 3*x^2 - x + x^3 - 1

-- Define the arranged polynomial
def arranged_polynomial (x : ℝ) : ℝ := -1 - x + 3*x^2 + x^3

-- Theorem stating that the original polynomial is equal to the arranged polynomial
theorem polynomial_arrangement :
  ∀ x : ℝ, original_polynomial x = arranged_polynomial x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_arrangement_l3124_312494


namespace NUMINAMATH_CALUDE_history_books_count_l3124_312439

theorem history_books_count (total : ℕ) (reading_fraction : ℚ) (math_fraction : ℚ) :
  total = 10 →
  reading_fraction = 2 / 5 →
  math_fraction = 3 / 10 →
  let reading := (reading_fraction * total).floor
  let math := (math_fraction * total).floor
  let science := math - 1
  let non_history := reading + math + science
  let history := total - non_history
  history = 1 := by sorry

end NUMINAMATH_CALUDE_history_books_count_l3124_312439


namespace NUMINAMATH_CALUDE_chipmunk_acorns_count_l3124_312455

/-- The number of acorns hidden by the chipmunk in each hole -/
def chipmunk_acorns_per_hole : ℕ := 3

/-- The number of acorns hidden by the squirrel in each hole -/
def squirrel_acorns_per_hole : ℕ := 4

/-- The number of holes dug by the chipmunk -/
def chipmunk_holes : ℕ := 16

/-- The number of holes dug by the squirrel -/
def squirrel_holes : ℕ := chipmunk_holes - 4

/-- The total number of acorns hidden by the chipmunk -/
def chipmunk_total_acorns : ℕ := chipmunk_acorns_per_hole * chipmunk_holes

/-- The total number of acorns hidden by the squirrel -/
def squirrel_total_acorns : ℕ := squirrel_acorns_per_hole * squirrel_holes

theorem chipmunk_acorns_count : chipmunk_total_acorns = 48 ∧ chipmunk_total_acorns = squirrel_total_acorns :=
by sorry

end NUMINAMATH_CALUDE_chipmunk_acorns_count_l3124_312455


namespace NUMINAMATH_CALUDE_james_soda_consumption_l3124_312443

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week (7 days)
-/
theorem james_soda_consumption 
  (packs : ℕ) 
  (sodas_per_pack : ℕ) 
  (initial_sodas : ℕ) 
  (days_to_finish : ℕ) 
  (h1 : packs = 5)
  (h2 : sodas_per_pack = 12)
  (h3 : initial_sodas = 10)
  (h4 : days_to_finish = 7) :
  (packs * sodas_per_pack + initial_sodas) / days_to_finish = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_soda_consumption_l3124_312443


namespace NUMINAMATH_CALUDE_min_illuminated_points_l3124_312445

/-- Calculates the number of illuminated points for a given angle -/
def illuminatedPoints (angle : ℕ) : ℕ :=
  180 / Nat.gcd 180 angle

/-- Represents the setup of the laser pointers in the circular room -/
structure LaserSetup where
  n : ℕ
  n_less_than_90 : n < 90

/-- Calculates the total number of distinct illuminated points for a given setup -/
def totalIlluminatedPoints (setup : LaserSetup) : ℕ :=
  illuminatedPoints setup.n + illuminatedPoints (setup.n + 1) - 1

/-- The theorem stating the minimum number of illuminated points -/
theorem min_illuminated_points :
  ∃ (setup : LaserSetup), ∀ (other : LaserSetup),
    totalIlluminatedPoints setup ≤ totalIlluminatedPoints other ∧
    totalIlluminatedPoints setup = 28 :=
  sorry

end NUMINAMATH_CALUDE_min_illuminated_points_l3124_312445


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3124_312471

/-- The imaginary part of (1-i)/(1+3i) is -2/5 -/
theorem imaginary_part_of_complex_fraction :
  Complex.im ((1 - Complex.I) / (1 + 3 * Complex.I)) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3124_312471


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l3124_312402

/-- The cost per page for manuscript revision --/
def revision_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (initial_cost_per_page : ℚ) (total_cost : ℚ) : ℚ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_cost_per_page
  let revision_pages := pages_revised_once + 2 * pages_revised_twice
  (total_cost - initial_typing_cost) / revision_pages

theorem manuscript_revision_cost :
  revision_cost 100 20 30 10 1400 = 5 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l3124_312402


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3124_312490

/-- Represents a triangle with side lengths and an angle -/
structure Triangle :=
  (sideAB : ℝ)
  (sideAC : ℝ)
  (sideBC : ℝ)
  (angleBAC : ℝ)

/-- Represents a circle with a radius -/
structure Circle :=
  (radius : ℝ)

/-- Calculates the area of two shaded regions in a specific geometric configuration -/
def shadedArea (t : Triangle) (c : Circle) : ℝ :=
  sorry

theorem shaded_area_calculation (t : Triangle) (c : Circle) :
  t.sideAB = 16 ∧ t.sideAC = 16 ∧ t.sideBC = c.radius * 2 ∧ t.angleBAC = 120 * π / 180 →
  shadedArea t c = 43 * π - 128 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3124_312490


namespace NUMINAMATH_CALUDE_total_seeds_after_trading_is_2340_l3124_312426

/-- Represents the number of watermelon seeds each person has -/
structure SeedCount where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Calculates the total number of seeds after trading -/
def totalSeedsAfterTrading (initial : SeedCount) : ℕ :=
  let bomAfter := initial.bom - 50
  let gwiAfter := initial.gwi + (initial.yeon * 20 / 100)
  let yeonAfter := initial.yeon - (initial.yeon * 20 / 100)
  let eunAfter := initial.eun + 50
  bomAfter + gwiAfter + yeonAfter + eunAfter

/-- Theorem stating that the total number of seeds after trading is 2340 -/
theorem total_seeds_after_trading_is_2340 (initial : SeedCount) 
  (h1 : initial.yeon = 3 * initial.gwi)
  (h2 : initial.gwi = initial.bom + 40)
  (h3 : initial.eun = 2 * initial.gwi)
  (h4 : initial.bom = 300) :
  totalSeedsAfterTrading initial = 2340 := by
  sorry

#eval totalSeedsAfterTrading { bom := 300, gwi := 340, yeon := 1020, eun := 680 }

end NUMINAMATH_CALUDE_total_seeds_after_trading_is_2340_l3124_312426


namespace NUMINAMATH_CALUDE_pool_filling_time_l3124_312437

/-- Proves that filling a pool of 15,000 gallons with four hoses (two at 2 gal/min, two at 3 gal/min) takes 25 hours -/
theorem pool_filling_time : 
  let pool_volume : ℝ := 15000
  let hose_rate_1 : ℝ := 2
  let hose_rate_2 : ℝ := 3
  let num_hoses_1 : ℕ := 2
  let num_hoses_2 : ℕ := 2
  let total_rate : ℝ := hose_rate_1 * num_hoses_1 + hose_rate_2 * num_hoses_2
  let fill_time_minutes : ℝ := pool_volume / total_rate
  let fill_time_hours : ℝ := fill_time_minutes / 60
  fill_time_hours = 25 := by
sorry


end NUMINAMATH_CALUDE_pool_filling_time_l3124_312437


namespace NUMINAMATH_CALUDE_sarees_with_six_shirts_l3124_312465

/-- The price of a saree in dollars -/
def saree_price : ℕ := sorry

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := sorry

/-- The number of sarees bought with 6 shirts -/
def num_sarees : ℕ := sorry

theorem sarees_with_six_shirts :
  (2 * saree_price + 4 * shirt_price = 1600) →
  (12 * shirt_price = 2400) →
  (num_sarees * saree_price + 6 * shirt_price = 1600) →
  num_sarees = 1 := by
  sorry

end NUMINAMATH_CALUDE_sarees_with_six_shirts_l3124_312465


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3124_312460

theorem sum_with_radical_conjugate : 
  let a : ℝ := 15
  let b : ℝ := Real.sqrt 500
  (a - b) + (a + b) = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3124_312460


namespace NUMINAMATH_CALUDE_quadratic_inequality_impossibility_l3124_312404

/-- Given a quadratic function f(x) = ax^2 + 2ax + 1 where a ≠ 0,
    it is impossible for f(-2) > f(-1) > f(0) to be true. -/
theorem quadratic_inequality_impossibility (a : ℝ) (h : a ≠ 0) :
  ¬∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 + 2 * a * x + 1) ∧ 
  (f (-2) > f (-1) ∧ f (-1) > f 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_impossibility_l3124_312404


namespace NUMINAMATH_CALUDE_max_value_constraint_l3124_312408

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * a * c ≤ 8/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3124_312408


namespace NUMINAMATH_CALUDE_early_registration_percentage_l3124_312453

/-- The percentage of attendees who registered at least two weeks in advance and paid in full -/
def early_reg_and_paid : ℝ := 78

/-- The percentage of attendees who paid in full but did not register early -/
def paid_not_early : ℝ := 10

/-- Proves that the percentage of attendees who registered at least two weeks in advance is 78% -/
theorem early_registration_percentage : ℝ := by
  sorry

end NUMINAMATH_CALUDE_early_registration_percentage_l3124_312453


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3124_312496

/-- A positive geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ q : ℝ, q > 0 ∧ q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (∀ n, a (n + 1) < a n) ∧
  (a 2 * a 8 = 6) ∧
  (a 4 + a 6 = 5)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 3 / a 7 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3124_312496


namespace NUMINAMATH_CALUDE_expand_product_l3124_312488

theorem expand_product (x : ℝ) : (5*x^2 + 7) * (3*x^3 + 4*x + 1) = 15*x^5 + 41*x^3 + 5*x^2 + 28*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3124_312488


namespace NUMINAMATH_CALUDE_range_of_sum_l3124_312495

theorem range_of_sum (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) :
  2 < x + y ∧ x + y < 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3124_312495


namespace NUMINAMATH_CALUDE_intersection_range_m_l3124_312472

theorem intersection_range_m (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| + |x + 1| ≤ 3}
  let B : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}
  (∃ x, x ∈ A ∩ B) → m > -5/2 ∧ m < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_m_l3124_312472


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3124_312401

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis
    of the ellipse 4(x-2)^2 + 16(y-3)^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoints_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * (y - 3)^2 = 64 ↔ 
      (x - 2)^2 / 16 + (y - 3)^2 / 4 = 1) → 
    (C.1 - 2)^2 / 16 + (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + (D.2 - 3)^2 / 4 = 1 →
    (C.1 - 2)^2 / 16 = 1 ∨ (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 = 1 ∨ (D.2 - 3)^2 / 4 = 1 →
    ((C.1 - 2)^2 / 16 = 1 ∧ (D.2 - 3)^2 / 4 = 1) ∨
    ((C.2 - 3)^2 / 4 = 1 ∧ (D.1 - 2)^2 / 16 = 1) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3124_312401


namespace NUMINAMATH_CALUDE_sum_of_distances_l3124_312431

/-- Given two line segments AB and A'B', with points D and D' on them respectively,
    and a point P on AB, prove that the sum of PD and P'D' is 10/3 units. -/
theorem sum_of_distances (AB A'B' AD A'D' PD : ℝ) (h1 : AB = 8)
    (h2 : A'B' = 6) (h3 : AD = 3) (h4 : A'D' = 1) (h5 : PD = 2)
    (h6 : PD / P'D' = 3 / 2) : PD + P'D' = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l3124_312431


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3124_312438

/-- Given a geometric sequence with first term 32 and common ratio 1/4,
    prove that the 15th term is 1/8388608 -/
theorem fifteenth_term_of_geometric_sequence :
  let a₁ : ℚ := 32  -- First term
  let r : ℚ := 1/4  -- Common ratio
  let n : ℕ := 15   -- Term number we're looking for
  let aₙ : ℚ := a₁ * r^(n-1)  -- General term formula
  aₙ = 1/8388608 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3124_312438


namespace NUMINAMATH_CALUDE_k_range_l3124_312434

theorem k_range (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x - 1 < 0) → -4 < k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l3124_312434


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3124_312440

theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x - Real.sqrt 3 * y + m = 0 → 
    (x^2 + y^2 - 2*y - 2 = 0 → 
      ∃ x' y' : ℝ, x' - Real.sqrt 3 * y' + m = 0 ∧ 
        x'^2 + y'^2 - 2*y' - 2 = 0 ∧ 
        ∀ x'' y'' : ℝ, x'' - Real.sqrt 3 * y'' + m = 0 → 
          x''^2 + y''^2 - 2*y'' - 2 ≤ 0)) ↔ 
  (m = -Real.sqrt 3 ∨ m = 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3124_312440


namespace NUMINAMATH_CALUDE_range_of_a_l3124_312441

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 / (x - 1) < 1
def q (x a : ℝ) : Prop := x^2 + (a - 1) * x - a > 0

-- Define the property that p is sufficient but not necessary for q
def p_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, p_sufficient_not_necessary a ↔ -2 < a ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3124_312441


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3124_312478

theorem sum_of_fractions : (2 : ℚ) / 5 + 3 / 8 + 1 / 4 = 41 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3124_312478


namespace NUMINAMATH_CALUDE_opposite_absolute_square_l3124_312422

theorem opposite_absolute_square (x y : ℝ) : 
  (|x - 2| = -(y + 7)^2 ∨ -(x - 2) = (y + 7)^2) → y^x = 49 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_square_l3124_312422


namespace NUMINAMATH_CALUDE_common_tangents_M_N_l3124_312475

/-- Circle M defined by the equation x^2 + y^2 - 4y = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- Circle N defined by the equation (x-1)^2 + (y-1)^2 = 1 -/
def circle_N (x y : ℝ) : Prop := (x-1)^2 + (y-1)^2 = 1

/-- The number of common tangent lines between two circles -/
def num_common_tangents (M N : (ℝ → ℝ → Prop)) : ℕ := sorry

/-- Theorem stating that the number of common tangent lines between circles M and N is 2 -/
theorem common_tangents_M_N : num_common_tangents circle_M circle_N = 2 := by sorry

end NUMINAMATH_CALUDE_common_tangents_M_N_l3124_312475


namespace NUMINAMATH_CALUDE_perimeter_area_sum_l3124_312447

/-- A parallelogram with vertices at (2,3), (2,8), (9,8), and (9,3) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 3)
  v2 : ℝ × ℝ := (2, 8)
  v3 : ℝ × ℝ := (9, 8)
  v4 : ℝ × ℝ := (9, 3)

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  2 * (abs (p.v3.1 - p.v1.1) + abs (p.v2.2 - p.v1.2))

/-- Calculate the area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  abs (p.v3.1 - p.v1.1) * abs (p.v2.2 - p.v1.2)

/-- The sum of the perimeter and area of the parallelogram is 59 -/
theorem perimeter_area_sum (p : Parallelogram) : perimeter p + area p = 59 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_area_sum_l3124_312447


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l3124_312415

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l3124_312415


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l3124_312491

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_students = 1200) 
  (h2 : sample_size = 30) 
  (h3 : female_in_sample = 14) :
  let male_in_sample := sample_size - female_in_sample
  let male_in_grade := (male_in_sample : ℚ) / sample_size * total_students
  male_in_grade = 640 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l3124_312491


namespace NUMINAMATH_CALUDE_lottery_ambo_probability_l3124_312487

theorem lottery_ambo_probability (n : ℕ) : 
  (n ≥ 5) →
  (Nat.choose 5 2 : ℚ) / (Nat.choose n 2 : ℚ) = 5 / 473 →
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_lottery_ambo_probability_l3124_312487


namespace NUMINAMATH_CALUDE_belize_homes_without_features_l3124_312432

/-- The town of Belize with its home characteristics -/
structure BelizeTown where
  total_homes : ℕ
  white_homes : ℕ
  non_white_homes : ℕ
  non_white_with_fireplace : ℕ
  non_white_with_fireplace_and_basement : ℕ
  non_white_without_fireplace : ℕ
  non_white_without_fireplace_with_garden : ℕ

/-- Properties of the Belize town -/
def belize_properties (t : BelizeTown) : Prop :=
  t.total_homes = 400 ∧
  t.white_homes = t.total_homes / 4 ∧
  t.non_white_homes = t.total_homes - t.white_homes ∧
  t.non_white_with_fireplace = t.non_white_homes / 5 ∧
  t.non_white_with_fireplace_and_basement = t.non_white_with_fireplace / 3 ∧
  t.non_white_without_fireplace = t.non_white_homes - t.non_white_with_fireplace ∧
  t.non_white_without_fireplace_with_garden = t.non_white_without_fireplace / 2

/-- Theorem: The number of non-white homes without fireplace, basement, or garden is 120 -/
theorem belize_homes_without_features (t : BelizeTown) 
  (h : belize_properties t) : 
  t.non_white_without_fireplace - t.non_white_without_fireplace_with_garden = 120 := by
  sorry

end NUMINAMATH_CALUDE_belize_homes_without_features_l3124_312432


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3124_312428

/-- A 2-digit positive integer is represented by its tens and ones digits -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a 2-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem unique_two_digit_number :
  ∃! (c : TwoDigitNumber), 
    c.tens + c.ones = 10 ∧ 
    c.tens * c.ones = 25 ∧ 
    c.value = 55 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3124_312428


namespace NUMINAMATH_CALUDE_range_of_a_l3124_312493

-- Define set A
def A : Set ℝ := {x | 1 < |x - 2| ∧ |x - 2| < 2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 1) * x + a < 0}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ A ∩ B a) ↔ a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3124_312493


namespace NUMINAMATH_CALUDE_mexica_numbers_less_than_2019_l3124_312444

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- A natural number is mexica if it's of the form n^(d(n)) -/
def is_mexica (m : ℕ) : Prop :=
  ∃ n : ℕ+, m = n.val ^ (d n)

/-- The set of mexica numbers less than 2019 -/
def mexica_set : Finset ℕ :=
  {1, 4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 64, 1296}

theorem mexica_numbers_less_than_2019 :
  {m : ℕ | is_mexica m ∧ m < 2019} = mexica_set := by sorry

end NUMINAMATH_CALUDE_mexica_numbers_less_than_2019_l3124_312444


namespace NUMINAMATH_CALUDE_abs_5e_minus_15_l3124_312436

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem abs_5e_minus_15 : |5 * e - 15| = 1.4086 := by sorry

end NUMINAMATH_CALUDE_abs_5e_minus_15_l3124_312436


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3124_312457

theorem absolute_value_sum (a : ℝ) (h1 : -2 < a) (h2 : a < 0) :
  |a| + |a + 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3124_312457


namespace NUMINAMATH_CALUDE_hash_example_l3124_312456

def hash (a b c d : ℝ) : ℝ := d * b^2 - 4 * a * c

theorem hash_example : hash 2 3 1 (1/2) = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_hash_example_l3124_312456


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l3124_312424

/-- A convex hexagon -/
structure ConvexHexagon where
  -- We don't need to define the structure of a hexagon for this problem

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the structure of a diagonal for this problem

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : ConvexHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (h : ConvexHexagon) : ℚ :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (h : ConvexHexagon) :
  intersection_probability h = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l3124_312424


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l3124_312469

theorem stratified_sampling_probability (students teachers support_staff sample_size : ℕ) 
  (h1 : students = 2500)
  (h2 : teachers = 350)
  (h3 : support_staff = 150)
  (h4 : sample_size = 300) :
  (sample_size * students) / ((students + teachers + support_staff) * students) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l3124_312469


namespace NUMINAMATH_CALUDE_railroad_grade_reduction_l3124_312449

theorem railroad_grade_reduction (rise : ℝ) (initial_grade : ℝ) (reduced_grade : ℝ) :
  rise = 800 →
  initial_grade = 0.04 →
  reduced_grade = 0.03 →
  ⌊(rise / reduced_grade - rise / initial_grade)⌋ = 6667 := by
  sorry

end NUMINAMATH_CALUDE_railroad_grade_reduction_l3124_312449


namespace NUMINAMATH_CALUDE_exactlyOneOfThreeCount_l3124_312400

/-- The number of math majors taking exactly one of Galois Theory, Hyperbolic Geometry, or Topology -/
def exactlyOneOfThree (total : ℕ) (noElective : ℕ) (ant_gt : ℕ) (gt_hg : ℕ) (hg_cry : ℕ) (cry_top : ℕ) (top_ant : ℕ) (ant_or_cry : ℕ) : ℕ :=
  total - noElective - ant_gt - gt_hg - hg_cry - cry_top - top_ant - ant_or_cry

theorem exactlyOneOfThreeCount :
  exactlyOneOfThree 100 22 7 12 3 15 8 16 = 17 :=
sorry

end NUMINAMATH_CALUDE_exactlyOneOfThreeCount_l3124_312400


namespace NUMINAMATH_CALUDE_circle_equation_l3124_312492

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersectsXAxisAt (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = 0 ∧ p2.2 = 0 ∧
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2

def isTangentToLine (c : Circle) : Prop :=
  let d := |c.center.1 - c.center.2 + 1| / Real.sqrt 2
  d = c.radius

-- Theorem statement
theorem circle_equation (c : Circle) :
  isInFirstQuadrant c.center →
  intersectsXAxisAt c (1, 0) (3, 0) →
  isTangentToLine c →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3124_312492


namespace NUMINAMATH_CALUDE_min_value_expression_l3124_312466

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (x + 2) * (2 * y + 1) / (x * y) ≥ 19 + 4 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3124_312466


namespace NUMINAMATH_CALUDE_zoo_ratio_l3124_312464

theorem zoo_ratio :
  let lions : ℕ := 30
  let penguins : ℕ := lions + 82
  (lions : ℚ) / penguins = 15 / 56 :=
by sorry

end NUMINAMATH_CALUDE_zoo_ratio_l3124_312464


namespace NUMINAMATH_CALUDE_solution_set_when_m_neg_one_m_range_for_subset_condition_l3124_312482

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem m_range_for_subset_condition :
  {m : ℝ | ∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2 * x + 1|} = Set.Icc (-11/4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_neg_one_m_range_for_subset_condition_l3124_312482


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3124_312433

theorem sum_of_decimals : 
  let addend1 : ℚ := 57/100
  let addend2 : ℚ := 23/100
  addend1 + addend2 = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3124_312433


namespace NUMINAMATH_CALUDE_tv_price_change_l3124_312407

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.45) = P * 1.16 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l3124_312407


namespace NUMINAMATH_CALUDE_vector_subtraction_l3124_312430

/-- Given two vectors OA and OB in 2D space, prove that the vector AB is their difference -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  OB - OA = (-9, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3124_312430


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l3124_312479

/-- Given two points on opposite sides of a line, prove the range of the line's constant term -/
theorem opposite_sides_line_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6) ∧ 
    ((3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0)) →
  (-7 < m ∧ m < 24) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l3124_312479


namespace NUMINAMATH_CALUDE_sum_greater_than_four_probability_l3124_312462

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of outcomes where the sum is 4 or less -/
def outcomesLessThanOrEqualToFour : ℕ := 6

/-- The probability that the sum of two dice is greater than four -/
def probabilityGreaterThanFour : ℚ := 1 - (outcomesLessThanOrEqualToFour : ℚ) / (totalOutcomes : ℚ)

theorem sum_greater_than_four_probability :
  probabilityGreaterThanFour = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_probability_l3124_312462


namespace NUMINAMATH_CALUDE_max_m_inequality_l3124_312477

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 1 / b = 1 / 4) : 
  (∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*m) ∧ 
               (∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 / x + 1 / y = 1 / 4 ∧ 2*x + y < 4*(m + ε))) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*n) → n ≤ m) ∧
  m = 9 :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l3124_312477


namespace NUMINAMATH_CALUDE_all_functions_increasing_l3124_312451

-- Define the functions
def f₁ (x : ℝ) : ℝ := 2 * x
def f₂ (x : ℝ) : ℝ := x^2 + 2*x - 1
def f₃ (x : ℝ) : ℝ := abs (x + 2)
def f₄ (x : ℝ) : ℝ := abs x + 2

-- Define the interval [0, +∞)
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Theorem statement
theorem all_functions_increasing :
  (∀ x y, nonnegative x → nonnegative y → x < y → f₁ x < f₁ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₂ x < f₂ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₃ x < f₃ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₄ x < f₄ y) :=
by sorry

end NUMINAMATH_CALUDE_all_functions_increasing_l3124_312451


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l3124_312412

theorem sqrt_sum_equals_2sqrt14 :
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l3124_312412


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3124_312418

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 9 → x = -77 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3124_312418


namespace NUMINAMATH_CALUDE_stating_initial_order_correct_l3124_312484

/-- Represents the colors of the notebooks -/
inductive Color
  | Blue
  | Grey
  | Brown
  | Red
  | Yellow

/-- Represents a stack of notebooks -/
def Stack := List Color

/-- The first arrangement of notebooks -/
def first_arrangement : (Stack × Stack) :=
  ([Color.Red, Color.Yellow, Color.Grey], [Color.Brown, Color.Blue])

/-- The second arrangement of notebooks -/
def second_arrangement : (Stack × Stack) :=
  ([Color.Brown, Color.Red], [Color.Yellow, Color.Grey, Color.Blue])

/-- The hypothesized initial order of notebooks -/
def initial_order : Stack :=
  [Color.Brown, Color.Red, Color.Yellow, Color.Grey, Color.Blue]

/-- 
Theorem stating that the initial_order is correct given the two arrangements
-/
theorem initial_order_correct :
  ∃ (process : Stack → (Stack × Stack)),
    process initial_order = first_arrangement ∧
    process (initial_order.reverse.reverse) = second_arrangement :=
sorry

end NUMINAMATH_CALUDE_stating_initial_order_correct_l3124_312484


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3124_312497

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3124_312497


namespace NUMINAMATH_CALUDE_joe_toy_cars_l3124_312463

theorem joe_toy_cars (initial_cars : ℕ) (grandmother_multiplier : ℕ) : 
  initial_cars = 50 → grandmother_multiplier = 3 → 
  initial_cars + grandmother_multiplier * initial_cars = 200 := by
sorry

end NUMINAMATH_CALUDE_joe_toy_cars_l3124_312463


namespace NUMINAMATH_CALUDE_min_value_theorem_l3124_312429

theorem min_value_theorem (x y : ℝ) (h1 : x * y = 1/2) (h2 : 0 < x ∧ x < 1) (h3 : 0 < y ∧ y < 1) :
  (2 / (1 - x)) + (1 / (1 - y)) ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3124_312429


namespace NUMINAMATH_CALUDE_find_g_of_x_l3124_312416

theorem find_g_of_x (x : ℝ) : 
  let g := fun (x : ℝ) ↦ -4*x^4 + 5*x^3 - 2*x^2 + 7*x + 2
  4*x^4 + 2*x^2 - 7*x + g x = 5*x^3 - 4*x + 2 := by sorry

end NUMINAMATH_CALUDE_find_g_of_x_l3124_312416


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l3124_312417

/-- Represents a cube cake -/
structure CubeCake where
  size : Nat
  top_iced : Bool
  front_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two sides -/
def count_double_iced_cubes (cake : CubeCake) : Nat :=
  if cake.top_iced && cake.front_iced then
    cake.size - 1
  else
    0

/-- Theorem: A 3x3x3 cake with top and front face iced has 2 subcubes with icing on two sides -/
theorem double_iced_cubes_count :
  let cake : CubeCake := { size := 3, top_iced := true, front_iced := true }
  count_double_iced_cubes cake = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_iced_cubes_count_l3124_312417


namespace NUMINAMATH_CALUDE_equation_solution_l3124_312411

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem equation_solution :
  ∃ (z : ℂ), (1 - i * z = -1 + i * z) ∧ (z = -i) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3124_312411


namespace NUMINAMATH_CALUDE_laura_debt_after_one_year_l3124_312413

/-- Calculates the total amount owed after applying simple interest -/
def totalAmountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Laura's debt after one year -/
theorem laura_debt_after_one_year :
  totalAmountOwed 35 0.05 1 = 36.75 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_after_one_year_l3124_312413


namespace NUMINAMATH_CALUDE_expression_evaluation_l3124_312489

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7 * 3) = -1379 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3124_312489


namespace NUMINAMATH_CALUDE_books_rearrangement_l3124_312485

theorem books_rearrangement (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 42 → 
  books_per_new_box = 45 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 0 :=
by sorry

end NUMINAMATH_CALUDE_books_rearrangement_l3124_312485


namespace NUMINAMATH_CALUDE_tangent_line_curve_equivalence_l3124_312467

theorem tangent_line_curve_equivalence 
  (α β m n : ℝ) 
  (h_pos_α : α > 0) 
  (h_pos_β : β > 0) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_relation : 1 / α + 1 / β = 1) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    (m * x + n * y = 1) ∧ 
    (x ^ α + y ^ α = 1) ∧
    (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' ^ α + y' ^ α = 1 → m * x' + n * y' ≥ 1))
  ↔ 
  (m ^ β + n ^ β = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_equivalence_l3124_312467


namespace NUMINAMATH_CALUDE_incorrect_games_proportion_l3124_312470

/-- Represents a chess tournament -/
structure ChessTournament where
  N : ℕ  -- number of players
  incorrect_games : ℕ  -- number of incorrect games

/-- Definition of a round-robin tournament -/
def is_round_robin (t : ChessTournament) : Prop :=
  t.incorrect_games ≤ t.N * (t.N - 1) / 2

/-- The main theorem: incorrect games are less than 75% of total games -/
theorem incorrect_games_proportion (t : ChessTournament) 
  (h : is_round_robin t) : 
  (4 * t.incorrect_games : ℚ) < (3 * t.N * (t.N - 1) : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_incorrect_games_proportion_l3124_312470


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3124_312448

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3124_312448


namespace NUMINAMATH_CALUDE_max_value_abc_l3124_312498

theorem max_value_abc (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 3 ∧
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3124_312498


namespace NUMINAMATH_CALUDE_no_additional_cocoa_needed_l3124_312468

/-- Represents the chocolate cake recipe and baking scenario. -/
structure ChocolateCakeScenario where
  recipe_ratio : Real  -- Amount of cocoa powder per pound of cake batter
  cake_weight : Real   -- Total weight of the cake to be made
  given_cocoa : Real   -- Amount of cocoa powder already provided

/-- Calculates if additional cocoa powder is needed for the chocolate cake. -/
def additional_cocoa_needed (scenario : ChocolateCakeScenario) : Real :=
  scenario.recipe_ratio * scenario.cake_weight - scenario.given_cocoa

/-- Proves that no additional cocoa powder is needed in the given scenario. -/
theorem no_additional_cocoa_needed (scenario : ChocolateCakeScenario) 
  (h1 : scenario.recipe_ratio = 0.4)
  (h2 : scenario.cake_weight = 450)
  (h3 : scenario.given_cocoa = 259) : 
  additional_cocoa_needed scenario ≤ 0 := by
  sorry

#eval additional_cocoa_needed { recipe_ratio := 0.4, cake_weight := 450, given_cocoa := 259 }

end NUMINAMATH_CALUDE_no_additional_cocoa_needed_l3124_312468


namespace NUMINAMATH_CALUDE_postcard_selection_ways_l3124_312452

theorem postcard_selection_ways : 
  let total_teachers : ℕ := 4
  let type_a_cards : ℕ := 2
  let type_b_cards : ℕ := 3
  let total_cards_to_select : ℕ := 4
  ∃ (ways : ℕ), ways = 10 ∧ 
    ways = (Nat.choose total_teachers type_a_cards) + 
           (Nat.choose total_teachers (total_teachers - type_a_cards)) :=
by sorry

end NUMINAMATH_CALUDE_postcard_selection_ways_l3124_312452


namespace NUMINAMATH_CALUDE_point_coordinates_l3124_312458

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (P : Point) 
  (h1 : is_in_second_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 3) :
  P.x = -3 ∧ P.y = 2 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3124_312458


namespace NUMINAMATH_CALUDE_vertex_coordinates_l3124_312454

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -1

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x has coordinates (1, -1) -/
theorem vertex_coordinates :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l3124_312454


namespace NUMINAMATH_CALUDE_semicircle_radius_l3124_312476

theorem semicircle_radius (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * Real.pi * (a/2)^2 = 8 * Real.pi →
  Real.pi * (b/2) = 8.5 * Real.pi →
  c/2 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3124_312476


namespace NUMINAMATH_CALUDE_monica_study_ratio_l3124_312414

/-- Monica's study schedule problem -/
theorem monica_study_ratio : 
  ∀ (thursday_hours : ℝ),
  thursday_hours > 0 →
  2 + thursday_hours + (thursday_hours / 2) + (2 + thursday_hours + (thursday_hours / 2)) = 22 →
  thursday_hours / 2 = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_monica_study_ratio_l3124_312414


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3124_312405

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → (m-1)^2 - (m-3)*(m+3) - (m-1)*(m-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3124_312405


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l3124_312483

theorem ceiling_floor_calculation : ⌈(15 / 8) * (-35 / 4)⌉ - ⌊(15 / 8) * ⌊(-35 / 4) + (1 / 4)⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l3124_312483


namespace NUMINAMATH_CALUDE_jennifer_fish_count_l3124_312425

/-- The number of tanks Jennifer has already built -/
def built_tanks : ℕ := 3

/-- The number of fish each built tank can hold -/
def fish_per_built_tank : ℕ := 15

/-- The number of tanks Jennifer plans to build -/
def planned_tanks : ℕ := 3

/-- The number of fish each planned tank can hold -/
def fish_per_planned_tank : ℕ := 10

/-- The total number of fish Jennifer wants to house -/
def total_fish : ℕ := built_tanks * fish_per_built_tank + planned_tanks * fish_per_planned_tank

theorem jennifer_fish_count : total_fish = 75 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fish_count_l3124_312425


namespace NUMINAMATH_CALUDE_function_intersection_condition_l3124_312481

/-- The function f(x) = (k+1)x^2 - 2x + 1 has intersections with the x-axis
    if and only if k ≤ 0. -/
theorem function_intersection_condition (k : ℝ) :
  (∃ x, (k + 1) * x^2 - 2 * x + 1 = 0) ↔ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_condition_l3124_312481


namespace NUMINAMATH_CALUDE_average_speed_is_55_l3124_312450

-- Define the problem parameters
def initial_reading : ℕ := 2332
def final_reading : ℕ := 2772
def total_time : ℕ := 8

-- Define the average speed calculation
def average_speed : ℚ := (final_reading - initial_reading : ℚ) / total_time

-- Theorem statement
theorem average_speed_is_55 : average_speed = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_is_55_l3124_312450


namespace NUMINAMATH_CALUDE_line_through_points_l3124_312499

/-- Given a line y = ax + b passing through points (3, 2) and (7, 14), prove that 2a - b = 13 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  2 * a - b = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3124_312499


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3124_312486

theorem base_10_to_base_7 : ∃ (a b c d : ℕ), 
  803 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
  a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3124_312486


namespace NUMINAMATH_CALUDE_cube_surface_area_l3124_312442

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3124_312442


namespace NUMINAMATH_CALUDE_dans_car_efficiency_l3124_312409

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def miles_per_gallon : ℝ := 32

/-- Represents the cost of gas in dollars per gallon. -/
def gas_cost_per_gallon : ℝ := 4

/-- Represents the distance Dan's car can travel in miles. -/
def distance_traveled : ℝ := 368

/-- Represents the total cost of gas in dollars. -/
def total_gas_cost : ℝ := 46

/-- Proves that Dan's car gets 32 miles per gallon given the conditions. -/
theorem dans_car_efficiency :
  miles_per_gallon = distance_traveled / (total_gas_cost / gas_cost_per_gallon) := by
  sorry


end NUMINAMATH_CALUDE_dans_car_efficiency_l3124_312409


namespace NUMINAMATH_CALUDE_min_value_theorem_l3124_312473

/-- Two lines are perpendicular if the sum of products of their coefficients is zero -/
def perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * a₂ + b₁ * b₂ = 0

/-- Definition of the first line l₁: (a-1)x + y - 1 = 0 -/
def line1 (a x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0

/-- Definition of the second line l₂: x + 2by + 1 = 0 -/
def line2 (b x y : ℝ) : Prop := x + 2 * b * y + 1 = 0

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : perpendicular (a - 1) 1 1 (2 * b)) :
  (∀ a' b', a' > 0 → b' > 0 → perpendicular (a' - 1) 1 1 (2 * b') → 2 / a + 1 / b ≤ 2 / a' + 1 / b') ∧ 
  2 / a + 1 / b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3124_312473


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3124_312427

theorem quadratic_equation_solutions (a b m : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧ ∀ x : ℝ, a * (x + m)^2 + b = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 0 ∧ ∀ x : ℝ, a * (x - m + 2)^2 + b = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3124_312427


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l3124_312406

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

/-- The common ratio of the geometric sequence -/
def commonRatio : ℚ := 25 / (5/6)

/-- The nth term of the specific geometric sequence -/
def nthTerm (n : ℕ) : ℚ := geometricSequence (5/6) commonRatio n

/-- Predicate to check if a rational number is divisible by 2,000,000 -/
def divisibleByTwoMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (2000000 : ℚ) * k

/-- Statement: 8 is the smallest positive integer n such that the nth term 
    of the geometric sequence is divisible by 2,000,000 -/
theorem smallest_divisible_term : 
  (∀ m : ℕ, m < 8 → ¬(divisibleByTwoMillion (nthTerm m))) ∧ 
  (divisibleByTwoMillion (nthTerm 8)) := by sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l3124_312406


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3124_312420

/-- Given a geometric progression with the first three terms 2^(1/4), 2^(1/8), and 2^(1/16),
    the fourth term is 2^(1/32). -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ a₄ : ℝ) : 
  a₁ = 2^(1/4) → a₂ = 2^(1/8) → a₃ = 2^(1/16) → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) →
  a₄ = 2^(1/32) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3124_312420
