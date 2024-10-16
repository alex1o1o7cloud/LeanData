import Mathlib

namespace NUMINAMATH_CALUDE_smallest_integer_problem_l490_49052

theorem smallest_integer_problem (a b c : ℕ+) : 
  (a : ℝ) + b + c = 90 ∧ 
  2 * a = 3 * b ∧ 
  2 * a = 5 * c ∧ 
  (a : ℝ) * b * c < 22000 → 
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_problem_l490_49052


namespace NUMINAMATH_CALUDE_percentage_sixth_graders_combined_l490_49095

theorem percentage_sixth_graders_combined (annville_total : ℕ) (cleona_total : ℕ)
  (annville_sixth_percent : ℚ) (cleona_sixth_percent : ℚ) :
  annville_total = 100 →
  cleona_total = 200 →
  annville_sixth_percent = 11 / 100 →
  cleona_sixth_percent = 17 / 100 →
  let annville_sixth := (annville_sixth_percent * annville_total : ℚ).floor
  let cleona_sixth := (cleona_sixth_percent * cleona_total : ℚ).floor
  let total_sixth := annville_sixth + cleona_sixth
  let total_students := annville_total + cleona_total
  (total_sixth : ℚ) / total_students = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_sixth_graders_combined_l490_49095


namespace NUMINAMATH_CALUDE_factorization_equality_l490_49013

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l490_49013


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l490_49020

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (2 : ℚ) / (n + 2) + (n + 1) / (n + 2) = 3 ∧ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l490_49020


namespace NUMINAMATH_CALUDE_reconstruct_pentagon_l490_49051

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A pentagon inscribed in a circle --/
structure InscribedPentagon where
  circle : Circle
  vertices : Fin 5 → Point

/-- Check if a point lies on a circle --/
def isOnCircle (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a point is the midpoint of two other points --/
def isMidpoint (p : Point) (a : Point) (b : Point) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

/-- The main theorem --/
theorem reconstruct_pentagon 
  (c : Circle) 
  (m₁ m₂ m₃ m₄ : Point) 
  (h₁ : isOnCircle c m₁) 
  (h₂ : isOnCircle c m₂) 
  (h₃ : isOnCircle c m₃) 
  (h₄ : isOnCircle c m₄) :
  ∃! (p : InscribedPentagon), 
    p.circle = c ∧ 
    (∃ (i₁ i₂ i₃ i₄ : Fin 5), 
      i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧
      isMidpoint m₁ (p.vertices i₁) (p.vertices (i₁ + 1)) ∧
      isMidpoint m₂ (p.vertices i₂) (p.vertices (i₂ + 1)) ∧
      isMidpoint m₃ (p.vertices i₃) (p.vertices (i₃ + 1)) ∧
      isMidpoint m₄ (p.vertices i₄) (p.vertices (i₄ + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_reconstruct_pentagon_l490_49051


namespace NUMINAMATH_CALUDE_hyperbola_equation_l490_49037

/-- Given a hyperbola with the following properties:
  1. Its equation is of the form x²/a² - y²/b² = 1 where a > 0 and b > 0
  2. It has an asymptote parallel to the line x + 2y + 5 = 0
  3. One of its foci lies on the line x + 2y + 5 = 0
  Prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ k, k ≠ 0 ∧ b / a = 1 / 2 * k) -- Asymptote parallel condition
  (h4 : ∃ x y, x + 2*y + 5 = 0 ∧ (x - a)^2 / a^2 + y^2 / b^2 = 1) -- Focus on line condition
  : a^2 = 20 ∧ b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l490_49037


namespace NUMINAMATH_CALUDE_trigonometric_identity_l490_49018

theorem trigonometric_identity : 
  2 * (Real.sin (35 * π / 180) * Real.cos (25 * π / 180) + 
       Real.cos (35 * π / 180) * Real.cos (65 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l490_49018


namespace NUMINAMATH_CALUDE_fifth_term_value_l490_49059

def sequence_term (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => 2^i + 5)

theorem fifth_term_value : sequence_term 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l490_49059


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l490_49026

theorem roots_of_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) →
  a = -6 ∧ b = -1 ∧ c = -4 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l490_49026


namespace NUMINAMATH_CALUDE_power_sum_div_diff_equals_17_15_l490_49033

theorem power_sum_div_diff_equals_17_15 :
  (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_div_diff_equals_17_15_l490_49033


namespace NUMINAMATH_CALUDE_area_XPQ_is_435_div_48_l490_49077

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side XZ -/
  xz : ℝ
  /-- Distance XP on side XY -/
  xp : ℝ
  /-- Distance XQ on side XZ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than or equal to xy -/
  xp_bounds : 0 < xp ∧ xp ≤ xy
  /-- xq is positive and less than or equal to xz -/
  xq_bounds : 0 < xq ∧ xq ≤ xz

/-- The area of triangle XPQ in the given configuration -/
def areaXPQ (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the area of triangle XPQ is 435/48 for the given configuration -/
theorem area_XPQ_is_435_div_48 (t : TriangleXYZ) 
    (h_xy : t.xy = 8) 
    (h_yz : t.yz = 9) 
    (h_xz : t.xz = 10) 
    (h_xp : t.xp = 3) 
    (h_xq : t.xq = 6) : 
  areaXPQ t = 435 / 48 := by
  sorry

end NUMINAMATH_CALUDE_area_XPQ_is_435_div_48_l490_49077


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l490_49032

theorem inequality_solution_implies_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ -1/2 * x^2 + 2*x > m*x) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l490_49032


namespace NUMINAMATH_CALUDE_min_roots_in_interval_l490_49070

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g (3 + x) = g (3 - x)) ∧ (∀ x, g (5 + x) = g (5 - x))

/-- The theorem stating the minimum number of roots in the given interval -/
theorem min_roots_in_interval
  (g : ℝ → ℝ)
  (h_symmetric : SymmetricFunction g)
  (h_g1_zero : g 1 = 0) :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, g x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    roots.card ≥ 250 :=
  sorry

end NUMINAMATH_CALUDE_min_roots_in_interval_l490_49070


namespace NUMINAMATH_CALUDE_chicken_count_l490_49010

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l490_49010


namespace NUMINAMATH_CALUDE_quadrilateral_property_l490_49019

theorem quadrilateral_property (α β γ δ : Real) 
  (convex : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0)
  (sum_angles : α + β + γ + δ = 2 * π)
  (sum_cosines : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) :
  (α + β = π ∨ γ + δ = π) ∨ (α + γ = β + δ) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l490_49019


namespace NUMINAMATH_CALUDE_chess_team_selection_l490_49048

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_team_selection :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 10
  let boys_to_select : ℕ := 5
  let girls_to_select : ℕ := 3
  (choose total_boys boys_to_select) * (choose total_girls girls_to_select) = 6720 := by
sorry

end NUMINAMATH_CALUDE_chess_team_selection_l490_49048


namespace NUMINAMATH_CALUDE_triangle_formation_l490_49089

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 2 4) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 3 4 8) ∧
  ¬(can_form_triangle 4 5 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l490_49089


namespace NUMINAMATH_CALUDE_inequality_proof_l490_49016

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l490_49016


namespace NUMINAMATH_CALUDE_equation_solution_l490_49098

theorem equation_solution (x : ℚ) (h : x ≠ -2) :
  (4 * x / (x + 2) - 2 / (x + 2) = 3 / (x + 2)) → x = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l490_49098


namespace NUMINAMATH_CALUDE_wendy_photo_albums_l490_49042

theorem wendy_photo_albums 
  (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 22)
  (h2 : camera_pics = 2)
  (h3 : pics_per_album = 6)
  : (phone_pics + camera_pics) / pics_per_album = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_photo_albums_l490_49042


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_40_squared_l490_49012

theorem largest_prime_divisor_of_17_squared_plus_40_squared :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 40^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (17^2 + 40^2) → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_40_squared_l490_49012


namespace NUMINAMATH_CALUDE_first_brand_price_l490_49061

/-- The regular price of pony jeans -/
def pony_price : ℝ := 18

/-- The total savings on 5 pairs of jeans -/
def total_savings : ℝ := 8.55

/-- The sum of the two discount rates -/
def sum_discount_rates : ℝ := 0.22

/-- The discount rate on pony jeans -/
def pony_discount_rate : ℝ := 0.15

/-- The number of pairs of the first brand of jeans -/
def num_first_brand : ℕ := 3

/-- The number of pairs of pony jeans -/
def num_pony : ℕ := 2

/-- Theorem stating that the regular price of the first brand of jeans is $15 -/
theorem first_brand_price : ∃ (price : ℝ),
  price = 15 ∧
  (price * num_first_brand * (sum_discount_rates - pony_discount_rate) +
   pony_price * num_pony * pony_discount_rate = total_savings) :=
sorry

end NUMINAMATH_CALUDE_first_brand_price_l490_49061


namespace NUMINAMATH_CALUDE_area_of_r3_l490_49040

/-- Given a square R1 with area 36, R2 formed by connecting midpoints of R1's sides,
    and R3 formed by moving R2's corners halfway to its center, prove R3's area is 4.5 -/
theorem area_of_r3 (r1 r2 r3 : Real) : 
  r1^2 = 36 → 
  r2 = r1 * Real.sqrt 2 / 2 → 
  r3 = r2 / 2 → 
  r3^2 = 4.5 := by sorry

end NUMINAMATH_CALUDE_area_of_r3_l490_49040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l490_49011

/-- An arithmetic sequence {aₙ} with a₅ = 9 and a₁ + a₇ = 14 has the general formula aₙ = 2n - 1 -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l490_49011


namespace NUMINAMATH_CALUDE_urn_has_eleven_marbles_l490_49014

/-- Represents the number of marbles in an urn -/
structure Urn where
  green : ℕ
  yellow : ℕ

/-- The conditions of the marble problem -/
def satisfies_conditions (u : Urn) : Prop :=
  (4 * (u.green - 3) = u.green + u.yellow - 3) ∧
  (3 * u.green = u.green + u.yellow - 4)

/-- The theorem stating that an urn satisfying the conditions has 11 marbles -/
theorem urn_has_eleven_marbles (u : Urn) 
  (h : satisfies_conditions u) : u.green + u.yellow = 11 := by
  sorry

#check urn_has_eleven_marbles

end NUMINAMATH_CALUDE_urn_has_eleven_marbles_l490_49014


namespace NUMINAMATH_CALUDE_line_parametric_equations_l490_49022

/-- Parametric equations of a line passing through M(1,5) with inclination angle 2π/3 -/
theorem line_parametric_equations (t : ℝ) : 
  let M : ℝ × ℝ := (1, 5)
  let angle : ℝ := 2 * Real.pi / 3
  let P : ℝ × ℝ := (1 - (1/2) * t, 5 + (Real.sqrt 3 / 2) * t)
  (P.1 - M.1 = t * Real.cos angle) ∧ (P.2 - M.2 = t * Real.sin angle) := by
  sorry

end NUMINAMATH_CALUDE_line_parametric_equations_l490_49022


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l490_49039

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |((y^(1/2) - 2/y)^3 - (x^(1/2) - 2/x)^3) - c| < ε) ∧ c = -6 :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l490_49039


namespace NUMINAMATH_CALUDE_kevin_cards_l490_49000

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) : 
  initial_cards - lost_cards = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l490_49000


namespace NUMINAMATH_CALUDE_six_good_points_l490_49092

/-- A lattice point on a 9x9 grid -/
structure LatticePoint where
  x : Fin 9
  y : Fin 9

/-- A triangle defined by three lattice points -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Calculates the area of a triangle given three lattice points -/
def triangleArea (P Q R : LatticePoint) : ℚ :=
  sorry

/-- Checks if a point is a "good point" for a given triangle -/
def isGoodPoint (T : Triangle) (P : LatticePoint) : Prop :=
  triangleArea P T.A T.B = triangleArea P T.A T.C

/-- The main theorem stating that there are exactly 6 "good points" -/
theorem six_good_points (T : Triangle) : 
  ∃! (goodPoints : Finset LatticePoint), 
    (∀ P ∈ goodPoints, isGoodPoint T P) ∧ 
    goodPoints.card = 6 :=
  sorry

end NUMINAMATH_CALUDE_six_good_points_l490_49092


namespace NUMINAMATH_CALUDE_set_intersection_equality_l490_49067

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : M ∩ N = intersection := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l490_49067


namespace NUMINAMATH_CALUDE_correct_ring_arrangements_l490_49094

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_arrange : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_arrange *
  Nat.choose (rings_to_arrange + fingers - 1) (fingers - 1) *
  Nat.factorial rings_to_arrange

/-- Theorem stating the correct number of ring arrangements -/
theorem correct_ring_arrangements :
  ring_arrangements 7 6 4 = 423360 := by
  sorry

end NUMINAMATH_CALUDE_correct_ring_arrangements_l490_49094


namespace NUMINAMATH_CALUDE_sqrt_product_l490_49078

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l490_49078


namespace NUMINAMATH_CALUDE_cookies_prepared_l490_49068

theorem cookies_prepared (cookies_per_guest : ℕ) (num_guests : ℕ) (total_cookies : ℕ) :
  cookies_per_guest = 19 →
  num_guests = 2 →
  total_cookies = cookies_per_guest * num_guests →
  total_cookies = 38 := by
sorry

end NUMINAMATH_CALUDE_cookies_prepared_l490_49068


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_l490_49065

/-- Given a triangle with sides a, b, and c satisfying certain equations, 
    prove that the triangle is isosceles. -/
theorem triangle_is_isosceles 
  (a b c : ℝ) 
  (eq1 : a^2 - 4*b = 7) 
  (eq2 : b^2 - 4*c = -6) 
  (eq3 : c^2 - 6*a = -18) : 
  (b = c ∨ a = b ∨ a = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_l490_49065


namespace NUMINAMATH_CALUDE_ladder_slide_l490_49046

theorem ladder_slide (ladder_length : Real) (initial_distance : Real) (top_slip : Real) (foot_slide : Real) : 
  ladder_length = 30 ∧ 
  initial_distance = 8 ∧ 
  top_slip = 4 ∧ 
  foot_slide = 2 →
  (ladder_length ^ 2 = initial_distance ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2)) ^ 2) ∧
  (ladder_length ^ 2 = (initial_distance + foot_slide) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2) - top_slip) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ladder_slide_l490_49046


namespace NUMINAMATH_CALUDE_bartender_cheating_l490_49047

theorem bartender_cheating (total_cost : ℚ) (whiskey_cost pipe_cost : ℕ) : 
  total_cost = 11.80 ∧ whiskey_cost = 3 ∧ pipe_cost = 6 → ¬(∃ n : ℕ, total_cost = n * 3) :=
by sorry

end NUMINAMATH_CALUDE_bartender_cheating_l490_49047


namespace NUMINAMATH_CALUDE_two_std_dev_below_value_l490_49082

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevBelow (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

theorem two_std_dev_below_value :
  let nd : NormalDistribution := { μ := 14.0, σ := 1.5 }
  twoStdDevBelow nd = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_value_l490_49082


namespace NUMINAMATH_CALUDE_square_number_placement_l490_49075

theorem square_number_placement :
  ∃ (a b c d e : ℕ),
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (Nat.gcd a b > 1 ∧ Nat.gcd b c > 1 ∧ Nat.gcd c d > 1 ∧ Nat.gcd d a > 1) ∧
    (Nat.gcd a e > 1 ∧ Nat.gcd b e > 1 ∧ Nat.gcd c e > 1 ∧ Nat.gcd d e > 1) ∧
    (Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_number_placement_l490_49075


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l490_49038

/-- Proves that for a downward-opening parabola passing through (-1, y₁) and (4, y₂), y₁ > y₂ -/
theorem parabola_point_comparison (a c y₁ y₂ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-1 - 1)^2 + c)
  (h_y₂ : y₂ = a * (4 - 1)^2 + c) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l490_49038


namespace NUMINAMATH_CALUDE_olivine_stones_difference_l490_49035

theorem olivine_stones_difference (agate_stones olivine_stones diamond_stones : ℕ) : 
  agate_stones = 30 →
  olivine_stones > agate_stones →
  diamond_stones = olivine_stones + 11 →
  agate_stones + olivine_stones + diamond_stones = 111 →
  olivine_stones = agate_stones + 5 := by
sorry

end NUMINAMATH_CALUDE_olivine_stones_difference_l490_49035


namespace NUMINAMATH_CALUDE_triangle_properties_l490_49066

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * B →
  Real.cos B = 2/3 →
  a * b * Real.cos C = 88 →
  Real.cos C = 22/27 ∧ a + b + c = 28 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l490_49066


namespace NUMINAMATH_CALUDE_stating_escalator_steps_l490_49024

/-- Represents the total number of steps on an escalator -/
def total_steps : ℕ := 40

/-- Represents the number of steps I ascend on the moving escalator -/
def my_steps : ℕ := 20

/-- Represents the time I take to ascend the escalator in seconds -/
def my_time : ℕ := 60

/-- Represents the number of steps my wife ascends on the moving escalator -/
def wife_steps : ℕ := 16

/-- Represents the time my wife takes to ascend the escalator in seconds -/
def wife_time : ℕ := 72

/-- 
Theorem stating that the total number of steps on the escalator is 40,
given the conditions about my ascent and my wife's ascent.
-/
theorem escalator_steps : 
  (total_steps - my_steps) / my_time = (total_steps - wife_steps) / wife_time :=
sorry

end NUMINAMATH_CALUDE_stating_escalator_steps_l490_49024


namespace NUMINAMATH_CALUDE_extreme_points_range_l490_49081

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

theorem extreme_points_range (a : ℝ) (x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂)
  (h₃ : ∀ x, f a x = 0 → x = x₁ ∨ x = x₂)
  (h₄ : ∀ m : ℝ, m * x₁ - f a x₂ / Real.exp x₁ > 0) :
  ∀ m : ℝ, m ≥ 2 ↔ m * x₁ - f a x₂ / Real.exp x₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_extreme_points_range_l490_49081


namespace NUMINAMATH_CALUDE_equation_solution_l490_49063

theorem equation_solution : 
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l490_49063


namespace NUMINAMATH_CALUDE_remainder_problem_l490_49054

theorem remainder_problem (n : ℤ) : 
  (n % 4 = 3) → (n % 9 = 5) → (n % 36 = 23) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l490_49054


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l490_49096

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2

/-- Given parallel vectors (2,3) and (x,-6), x equals -4 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, 3) (x, -6) → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l490_49096


namespace NUMINAMATH_CALUDE_initial_watermelons_l490_49023

theorem initial_watermelons (eaten : ℕ) (left : ℕ) (initial : ℕ) : 
  eaten = 3 → left = 1 → initial = eaten + left → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_watermelons_l490_49023


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l490_49091

theorem fraction_sum_theorem : (1/2 : ℚ) * (1/3 : ℚ) + (1/3 : ℚ) * (1/4 : ℚ) + (1/4 : ℚ) * (1/5 : ℚ) = (3/10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l490_49091


namespace NUMINAMATH_CALUDE_axis_triangle_line_equation_l490_49069

/-- A line passing through a point and forming a triangle with the axes --/
structure AxisTriangleLine where
  /-- The slope of the line --/
  k : ℝ
  /-- The line passes through the point (1, 2) --/
  passes_through : k * (1 - 0) = 2 - 0
  /-- The slope is negative --/
  negative_slope : k < 0
  /-- The area of the triangle formed with the axes is 4 --/
  triangle_area : (1/2) * (2 - k) * (1 - 2/k) = 4

/-- The equation of the line is 2x + y - 4 = 0 --/
theorem axis_triangle_line_equation (l : AxisTriangleLine) : 
  ∃ (a b c : ℝ), a * 1 + b * 2 + c = 0 ∧ 
                  ∀ x y, a * x + b * y + c = 0 ↔ y - 2 = l.k * (x - 1) :=
sorry

end NUMINAMATH_CALUDE_axis_triangle_line_equation_l490_49069


namespace NUMINAMATH_CALUDE_floating_time_calculation_l490_49029

/-- Floating time calculation -/
theorem floating_time_calculation
  (boat_speed_with_current : ℝ)
  (boat_speed_against_current : ℝ)
  (distance_floated : ℝ)
  (h1 : boat_speed_with_current = 28)
  (h2 : boat_speed_against_current = 24)
  (h3 : distance_floated = 20) :
  (distance_floated / ((boat_speed_with_current - boat_speed_against_current) / 2)) = 10 := by
  sorry

#check floating_time_calculation

end NUMINAMATH_CALUDE_floating_time_calculation_l490_49029


namespace NUMINAMATH_CALUDE_inequality_solution_set_l490_49057

theorem inequality_solution_set (x : ℝ) : 
  (3/20 : ℝ) + |x - 9/40| + |x + 1/8| < (1/2 : ℝ) ↔ -3/40 < x ∧ x < 11/40 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l490_49057


namespace NUMINAMATH_CALUDE_number_of_selection_schemes_l490_49003

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of cities to visit -/
def total_cities : ℕ := 4

/-- The number of people who cannot visit a specific city -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for cities with restrictions -/
def selection_schemes (n m r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - 2 * ((n - 1).factorial / (n - m).factorial)

/-- The main theorem stating the number of selection schemes -/
theorem number_of_selection_schemes :
  selection_schemes total_people total_cities restricted_people = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_of_selection_schemes_l490_49003


namespace NUMINAMATH_CALUDE_smallest_total_is_47_l490_49076

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.seventh = 2 * counts.ninth ∧
  7 * counts.sixth = 4 * counts.ninth

/-- The smallest possible total number of students --/
def smallestTotal : ℕ := 47

/-- Theorem stating that the smallest possible total number of students is 47 --/
theorem smallest_total_is_47 :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    counts.ninth + counts.seventh + counts.sixth = smallestTotal ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other →
      other.ninth + other.seventh + other.sixth ≥ smallestTotal) :=
  sorry

end NUMINAMATH_CALUDE_smallest_total_is_47_l490_49076


namespace NUMINAMATH_CALUDE_linear_program_unbounded_l490_49045

def objective_function (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + 2*x₃ - x₄

def constraint1 (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = 1
def constraint2 (x₂ x₃ x₄ : ℝ) : Prop := x₂ + x₃ - x₄ = 1
def non_negative (x : ℝ) : Prop := x ≥ 0

theorem linear_program_unbounded :
  ∀ M : ℝ, ∃ x₁ x₂ x₃ x₄ : ℝ,
    constraint1 x₁ x₂ ∧
    constraint2 x₂ x₃ x₄ ∧
    non_negative x₁ ∧
    non_negative x₂ ∧
    non_negative x₃ ∧
    non_negative x₄ ∧
    objective_function x₁ x₂ x₃ x₄ > M :=
by
  sorry


end NUMINAMATH_CALUDE_linear_program_unbounded_l490_49045


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l490_49086

def expand_polynomial (x : ℝ) := x * (x - 1) * (x + 1)^4

theorem x_squared_coefficient :
  ∃ (a b c d e : ℝ),
    expand_polynomial x = a*x^5 + b*x^4 + c*x^3 + 5*x^2 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l490_49086


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l490_49043

theorem smallest_upper_bound (a b : ℤ) (h1 : a > 6) (h2 : ∀ (x y : ℤ), x > 6 → x - y ≥ 4) : 
  ∃ N : ℤ, (a + b < N) ∧ (∀ M : ℤ, M < N → ¬(a + b < M)) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l490_49043


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l490_49002

theorem committee_meeting_attendance :
  ∀ (assoc_prof asst_prof : ℕ),
  2 * assoc_prof + asst_prof = 11 →
  assoc_prof + 2 * asst_prof = 16 →
  assoc_prof + asst_prof = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l490_49002


namespace NUMINAMATH_CALUDE_distance_calculation_l490_49036

/-- The speed of light in km/s -/
def speed_of_light : ℝ := 3 * 10^5

/-- The time it takes for light to reach Earth from Proxima Centauri in years -/
def travel_time : ℝ := 4

/-- The number of seconds in a year -/
def seconds_per_year : ℝ := 3 * 10^7

/-- The distance from Proxima Centauri to Earth in km -/
def distance_to_proxima_centauri : ℝ := speed_of_light * travel_time * seconds_per_year

theorem distance_calculation :
  distance_to_proxima_centauri = 3.6 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l490_49036


namespace NUMINAMATH_CALUDE_ellipse_k_range_l490_49021

/-- Represents an ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- Foci are on the y-axis if the equation is in the form x^2/b^2 + y^2/a^2 = 1 with a > b -/
def foci_on_y_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- The main theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ∧ foci_on_y_axis k → 0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l490_49021


namespace NUMINAMATH_CALUDE_no_solution_quadratic_l490_49079

theorem no_solution_quadratic (p q r s : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + r*x + s ≠ 0) :
  ∀ x : ℝ, 2017*x^2 + (1009*p + 1008*r)*x + 1009*q + 1008*s ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_l490_49079


namespace NUMINAMATH_CALUDE_men_who_left_l490_49027

/-- Given a hostel with provisions for a certain number of men and days,
    calculate the number of men who left if the provisions last longer. -/
theorem men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) :
  initial_men = 250 →
  initial_days = 32 →
  new_days = 40 →
  ∃ (men_left : ℕ),
    men_left = 50 ∧
    initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end NUMINAMATH_CALUDE_men_who_left_l490_49027


namespace NUMINAMATH_CALUDE_hyperbola_equation_l490_49097

/-- Given a hyperbola with equation x^2/a^2 - y^2/4 = 1 and an asymptote y = (1/2)x,
    prove that the equation of the hyperbola is x^2/16 - y^2/4 = 1 -/
theorem hyperbola_equation (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/4 = 1 → ∃ t : ℝ, y = (1/2) * x * t) →
  (∀ x y : ℝ, x^2/16 - y^2/4 = 1 ↔ x^2/a^2 - y^2/4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l490_49097


namespace NUMINAMATH_CALUDE_trigonometric_identity_l490_49031

theorem trigonometric_identity (x : ℝ) : 
  (4 * Real.sin x ^ 3 * Real.cos (3 * x) + 4 * Real.cos x ^ 3 * Real.sin (3 * x) = 3 * Real.sin (2 * x)) ↔ 
  (∃ n : ℤ, x = π / 6 * (2 * ↑n + 1)) ∨ (∃ k : ℤ, x = π * ↑k) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l490_49031


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l490_49083

/-- Given a right triangle with one leg of 15 inches and the angle opposite to that leg being 30°,
    the length of the hypotenuse is 30 inches. -/
theorem right_triangle_hypotenuse (a b c : ℝ) (θ : ℝ) : 
  a = 15 →  -- One leg is 15 inches
  θ = 30 * π / 180 →  -- Angle opposite to that leg is 30° (converted to radians)
  θ = Real.arcsin (a / c) →  -- Sine of the angle is opposite over hypotenuse
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 30 :=  -- Hypotenuse is 30 inches
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l490_49083


namespace NUMINAMATH_CALUDE_soda_packs_minimum_l490_49085

def min_packs (total : ℕ) (pack_sizes : List ℕ) : ℕ :=
  sorry

theorem soda_packs_minimum :
  min_packs 120 [8, 15, 30] = 4 :=
sorry

end NUMINAMATH_CALUDE_soda_packs_minimum_l490_49085


namespace NUMINAMATH_CALUDE_percentage_relation_l490_49084

theorem percentage_relation (x y z : ℝ) : 
  x = 1.2 * y ∧ y = 0.5 * z → x = 0.6 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l490_49084


namespace NUMINAMATH_CALUDE_book_cost_l490_49080

theorem book_cost (initial_money : ℕ) (notebooks : ℕ) (notebook_cost : ℕ) (books : ℕ) (money_left : ℕ) : 
  initial_money = 56 →
  notebooks = 7 →
  notebook_cost = 4 →
  books = 2 →
  money_left = 14 →
  (initial_money - money_left - notebooks * notebook_cost) / books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l490_49080


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l490_49062

theorem trigonometric_simplification :
  (Real.sin (11 * π / 180) * Real.cos (15 * π / 180) + 
   Real.sin (15 * π / 180) * Real.cos (11 * π / 180)) / 
  (Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
   Real.sin (12 * π / 180) * Real.cos (18 * π / 180)) = 
  2 * Real.sin (26 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l490_49062


namespace NUMINAMATH_CALUDE_quadratic_function_ordering_l490_49007

/-- A quadratic function with the given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x : ℝ, a * (2 + x)^2 + b * (2 + x) + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the ordering of function values -/
theorem quadratic_function_ordering (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * 1^2 + f.b * 1 + f.c ∧
  f.a * 1^2 + f.b * 1 + f.c < f.a * 4^2 + f.b * 4 + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_ordering_l490_49007


namespace NUMINAMATH_CALUDE_candy_sharing_theorem_l490_49008

/-- Represents the amount of candy each person has initially -/
structure CandyDistribution where
  hugh : ℕ
  tommy : ℕ
  melany : ℕ

/-- Calculates the amount of candy each person gets when shared equally -/
def equalShare (dist : CandyDistribution) : ℕ :=
  (dist.hugh + dist.tommy + dist.melany) / 3

/-- Theorem: When Hugh has 8 pounds, Tommy has 6 pounds, and Melany has 7 pounds of candy,
    sharing equally results in each person having 7 pounds of candy -/
theorem candy_sharing_theorem (dist : CandyDistribution) 
  (h1 : dist.hugh = 8) 
  (h2 : dist.tommy = 6) 
  (h3 : dist.melany = 7) : 
  equalShare dist = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_theorem_l490_49008


namespace NUMINAMATH_CALUDE_triangle_exists_l490_49087

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_exists : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l490_49087


namespace NUMINAMATH_CALUDE_sequence_sum_l490_49055

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 21 ∧ 
  Q + R + S = 21 ∧ 
  R + S + T = 21 ∧ 
  S + T + U = 21 ∧ 
  T + U + V = 21 → 
  P + V = 14 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l490_49055


namespace NUMINAMATH_CALUDE_exponent_division_l490_49073

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l490_49073


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l490_49074

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) 
  (h1 : total_time = 3)
  (h2 : swim_distance = 0.5)
  (h3 : swim_speed = 1)
  (h4 : run_distance = 5)
  (h5 : run_speed = 5)
  (h6 : bike_distance = 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l490_49074


namespace NUMINAMATH_CALUDE_annies_initial_apples_l490_49034

theorem annies_initial_apples (initial_apples total_apples apples_from_nathan : ℕ) :
  total_apples = initial_apples + apples_from_nathan →
  apples_from_nathan = 6 →
  total_apples = 12 →
  initial_apples = 6 := by
sorry

end NUMINAMATH_CALUDE_annies_initial_apples_l490_49034


namespace NUMINAMATH_CALUDE_g_fixed_points_l490_49053

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points (x : ℝ) : g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_fixed_points_l490_49053


namespace NUMINAMATH_CALUDE_intercept_sum_l490_49001

/-- The modulus of the congruence -/
def m : ℕ := 17

/-- The congruence relation -/
def congruence (x y : ℕ) : Prop :=
  (5 * x) % m = (3 * y + 2) % m

/-- Definition of x-intercept -/
def x_intercept (x₀ : ℕ) : Prop :=
  x₀ < m ∧ congruence x₀ 0

/-- Definition of y-intercept -/
def y_intercept (y₀ : ℕ) : Prop :=
  y₀ < m ∧ congruence 0 y₀

/-- The main theorem -/
theorem intercept_sum :
  ∀ x₀ y₀ : ℕ, x_intercept x₀ → y_intercept y₀ → x₀ + y₀ = 19 :=
by sorry

end NUMINAMATH_CALUDE_intercept_sum_l490_49001


namespace NUMINAMATH_CALUDE_chloe_trivia_points_l490_49044

/-- Chloe's trivia game points calculation -/
theorem chloe_trivia_points 
  (first_round : ℕ) 
  (last_round_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ (second_round : ℕ), 
    first_round + second_round - last_round_loss = total_points ∧ 
    second_round = 50 := by
sorry

end NUMINAMATH_CALUDE_chloe_trivia_points_l490_49044


namespace NUMINAMATH_CALUDE_min_value_of_expression_l490_49056

theorem min_value_of_expression (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 ∧ (1 / m + 2 / n = 4 ↔ n = 2 * m ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l490_49056


namespace NUMINAMATH_CALUDE_min_product_of_tangent_line_to_unit_circle_l490_49009

theorem min_product_of_tangent_line_to_unit_circle (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x^2 + y^2 = 1 ∧ x/a + y/b = 1) → a * b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_tangent_line_to_unit_circle_l490_49009


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l490_49058

theorem multiply_mixed_number : 7 * (9 + 2/5) = 329/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l490_49058


namespace NUMINAMATH_CALUDE_expression_value_at_negative_one_l490_49006

theorem expression_value_at_negative_one :
  let x : ℤ := -1
  (x^2 + 5*x - 6) = -10 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_one_l490_49006


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l490_49099

open Real

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) (h : f = λ x => sin x - cos x) (h1 : f α = 1) : sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l490_49099


namespace NUMINAMATH_CALUDE_appetizers_needed_l490_49088

/-- Represents the number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- Represents the number of guests -/
def number_of_guests : ℕ := 30

/-- Represents the number of dozens of deviled eggs prepared -/
def dozens_deviled_eggs : ℕ := 3

/-- Represents the number of dozens of pigs in a blanket prepared -/
def dozens_pigs_in_blanket : ℕ := 2

/-- Represents the number of dozens of kebabs prepared -/
def dozens_kebabs : ℕ := 2

/-- Represents the number of items in a dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem stating that Patsy needs to make 8 more dozen appetizers -/
theorem appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (dozens_deviled_eggs + dozens_pigs_in_blanket + dozens_kebabs) * items_per_dozen) / 
  items_per_dozen = 8 := by
  sorry

end NUMINAMATH_CALUDE_appetizers_needed_l490_49088


namespace NUMINAMATH_CALUDE_dolphin_shark_ratio_l490_49005

/-- The ratio of buckets fed to dolphins compared to sharks -/
def R : ℚ := 1 / 2

/-- The number of buckets fed to sharks daily -/
def shark_buckets : ℕ := 4

/-- The number of days in 3 weeks -/
def days : ℕ := 21

/-- The total number of buckets lasting 3 weeks -/
def total_buckets : ℕ := 546

theorem dolphin_shark_ratio :
  R * shark_buckets * days +
  shark_buckets * days +
  (5 * shark_buckets) * days = total_buckets := by sorry

end NUMINAMATH_CALUDE_dolphin_shark_ratio_l490_49005


namespace NUMINAMATH_CALUDE_x_geq_y_l490_49015

theorem x_geq_y (a : ℝ) : 2 * a * (a + 3) ≥ (a - 3) * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_x_geq_y_l490_49015


namespace NUMINAMATH_CALUDE_water_left_for_fourth_neighborhood_l490_49071

-- Define the total capacity of the water tower
def total_capacity : ℕ := 1200

-- Define the water usage of the first neighborhood
def first_neighborhood_usage : ℕ := 150

-- Define the water usage of the second neighborhood
def second_neighborhood_usage : ℕ := 2 * first_neighborhood_usage

-- Define the water usage of the third neighborhood
def third_neighborhood_usage : ℕ := second_neighborhood_usage + 100

-- Define the total usage of the first three neighborhoods
def total_usage : ℕ := first_neighborhood_usage + second_neighborhood_usage + third_neighborhood_usage

-- Theorem to prove
theorem water_left_for_fourth_neighborhood :
  total_capacity - total_usage = 350 := by sorry

end NUMINAMATH_CALUDE_water_left_for_fourth_neighborhood_l490_49071


namespace NUMINAMATH_CALUDE_puzzle_sum_l490_49050

def is_valid_puzzle (a b c d e f g h i : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem puzzle_sum (a b c d e f g h i : ℕ) :
  is_valid_puzzle a b c d e f g h i →
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) + (100 * g + 10 * h + i) = 1665 →
  b + e + h = 15 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l490_49050


namespace NUMINAMATH_CALUDE_sixPeopleRoundTable_l490_49072

/-- Number of distinct seating arrangements for n people around a round table,
    considering rotational symmetry -/
def roundTableSeating (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of distinct seating arrangements for 6 people around a round table,
    considering rotational symmetry, is 120 -/
theorem sixPeopleRoundTable : roundTableSeating 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sixPeopleRoundTable_l490_49072


namespace NUMINAMATH_CALUDE_binomial_60_3_l490_49028

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l490_49028


namespace NUMINAMATH_CALUDE_convex_ngon_non_acute_side_l490_49049

/-- A convex n-gon is a polygon with n sides and n vertices, where all internal angles are less than 180 degrees. -/
def ConvexNGon (n : ℕ) : Type := sorry

/-- An angle is acute if it is less than 90 degrees. -/
def IsAcute (angle : ℝ) : Prop := angle < 90

/-- Given a convex n-gon and a side, returns the two angles at the endpoints of that side. -/
def EndpointAngles (polygon : ConvexNGon n) (side : Fin n) : ℝ × ℝ := sorry

theorem convex_ngon_non_acute_side (n : ℕ) (hn : n ≥ 7) :
  ∀ (polygon : ConvexNGon n), ∃ (side : Fin n),
    let (angle1, angle2) := EndpointAngles polygon side
    ¬(IsAcute angle1 ∨ IsAcute angle2) :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_non_acute_side_l490_49049


namespace NUMINAMATH_CALUDE_x_is_28_percent_greater_than_150_l490_49041

theorem x_is_28_percent_greater_than_150 :
  ∀ x : ℝ, x = 150 * (1 + 28/100) → x = 192 := by
  sorry

end NUMINAMATH_CALUDE_x_is_28_percent_greater_than_150_l490_49041


namespace NUMINAMATH_CALUDE_tan_ratio_sum_l490_49060

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_sum_l490_49060


namespace NUMINAMATH_CALUDE_range_of_a_l490_49090

def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

theorem range_of_a (a : ℝ) : (A ∪ B a = Set.univ) → a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l490_49090


namespace NUMINAMATH_CALUDE_inequality_and_min_value_l490_49093

theorem inequality_and_min_value (a b x y : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : x > 0) (h5 : y > 0) :
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
  (∀ x ∈ Set.Ioo 0 (1/2), 2/x + 9/(1-2*x) ≥ 25) ∧
  (2/(1/5) + 9/(1-2*(1/5)) = 25) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_min_value_l490_49093


namespace NUMINAMATH_CALUDE_f_properties_l490_49064

-- Define the function f(x) = -x|x| + 2x
def f (x : ℝ) : ℝ := -x * abs x + 2 * x

-- State the theorem
theorem f_properties :
  -- f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- f is monotonically decreasing on (-∞, -1)
  (∀ x y, x < y → y < -1 → f y < f x) ∧
  -- f is monotonically decreasing on (1, +∞)
  (∀ x y, 1 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l490_49064


namespace NUMINAMATH_CALUDE_point_b_value_l490_49030

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) :
  a.value = 1 → distance a b = 3 → b.value = 4 ∨ b.value = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l490_49030


namespace NUMINAMATH_CALUDE_blackboard_problem_l490_49004

/-- Represents the state of the blackboard -/
structure BoardState where
  ones : ℕ
  twos : ℕ
  threes : ℕ
  fours : ℕ

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours + 2 }
  | Operation.erase_124_add_3 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes + 2, fours := state.fours - 1 }
  | Operation.erase_134_add_2 => 
      { ones := state.ones - 1, twos := state.twos + 2, 
        threes := state.threes - 1, fours := state.fours - 1 }
  | Operation.erase_234_add_1 => 
      { ones := state.ones + 2, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours - 1 }

/-- Checks if the board state is in a final state (only three numbers remain) -/
def isFinalState (state : BoardState) : Bool :=
  (state.ones + state.twos + state.threes + state.fours) = 3

/-- Calculates the product of the remaining numbers -/
def productOfRemaining (state : BoardState) : ℕ :=
  (if state.ones > 0 then 1^state.ones else 1) *
  (if state.twos > 0 then 2^state.twos else 1) *
  (if state.threes > 0 then 3^state.threes else 1) *
  (if state.fours > 0 then 4^state.fours else 1)

/-- The main theorem to prove -/
theorem blackboard_problem :
  ∃ (operations : List Operation),
    let initialState : BoardState := { ones := 11, twos := 22, threes := 33, fours := 44 }
    let finalState := operations.foldl applyOperation initialState
    isFinalState finalState ∧ productOfRemaining finalState = 12 := by
  sorry


end NUMINAMATH_CALUDE_blackboard_problem_l490_49004


namespace NUMINAMATH_CALUDE_sum_of_multiples_l490_49017

def largest_two_digit_multiple_of_5 : ℕ :=
  95

def smallest_three_digit_multiple_of_7 : ℕ :=
  105

theorem sum_of_multiples : 
  largest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l490_49017


namespace NUMINAMATH_CALUDE_largest_square_with_four_lattice_points_l490_49025

/-- A point (x, y) is a lattice point if both x and y are integers. -/
def isLatticePoint (p : ℝ × ℝ) : Prop :=
  Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square contains exactly four lattice points in its interior. -/
def squareContainsFourLatticePoints (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  isLatticePoint p₁ ∧ isLatticePoint p₂ ∧ isLatticePoint p₃ ∧ isLatticePoint p₄ ∧
  (∀ p ∈ s, isLatticePoint p → p = p₁ ∨ p = p₂ ∨ p = p₃ ∨ p = p₄)

/-- The theorem statement -/
theorem largest_square_with_four_lattice_points :
  ∃ (s : Set (ℝ × ℝ)), squareContainsFourLatticePoints s ∧
  (∀ (t : Set (ℝ × ℝ)), squareContainsFourLatticePoints t → MeasureTheory.volume s ≥ MeasureTheory.volume t) ∧
  MeasureTheory.volume s = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_square_with_four_lattice_points_l490_49025
