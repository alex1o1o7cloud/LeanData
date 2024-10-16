import Mathlib

namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l642_64253

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l642_64253


namespace NUMINAMATH_CALUDE_T_2023_mod_10_l642_64258

/-- Represents a sequence of C's and D's -/
inductive Sequence : Type
| C : Sequence
| D : Sequence
| cons : Sequence → Sequence → Sequence

/-- Checks if a sequence is valid (no more than two consecutive C's or D's) -/
def isValid : Sequence → Bool
| Sequence.C => true
| Sequence.D => true
| Sequence.cons s₁ s₂ => sorry  -- Implementation details omitted

/-- Counts the number of valid sequences of length n -/
def T (n : ℕ+) : ℕ :=
  (List.map (fun s => if isValid s then 1 else 0) (sorry : List Sequence)).sum
  -- Implementation details omitted

/-- Main theorem: T(2023) is congruent to 6 modulo 10 -/
theorem T_2023_mod_10 : T 2023 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_T_2023_mod_10_l642_64258


namespace NUMINAMATH_CALUDE_field_fencing_l642_64232

/-- Proves that a rectangular field with area 80 sq. feet and one side 20 feet requires 28 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length * width = 80 → 
  length = 20 → 
  length + 2 * width = 28 := by sorry

end NUMINAMATH_CALUDE_field_fencing_l642_64232


namespace NUMINAMATH_CALUDE_track_duration_in_seconds_l642_64237

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℚ) : ℚ := minutes * 60

/-- The duration of the music track in minutes -/
def trackDurationMinutes : ℚ := 12.5

/-- Theorem: A music track playing for 12.5 minutes lasts 750 seconds -/
theorem track_duration_in_seconds : 
  minutesToSeconds trackDurationMinutes = 750 := by sorry

end NUMINAMATH_CALUDE_track_duration_in_seconds_l642_64237


namespace NUMINAMATH_CALUDE_m_minus_n_eq_neg_reals_l642_64274

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_eq_neg_reals : 
  setDifference M N = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_m_minus_n_eq_neg_reals_l642_64274


namespace NUMINAMATH_CALUDE_triangle_ratio_l642_64262

theorem triangle_ratio (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l642_64262


namespace NUMINAMATH_CALUDE_problem_solution_l642_64222

theorem problem_solution (x : ℝ) (hx_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l642_64222


namespace NUMINAMATH_CALUDE_water_left_over_l642_64297

theorem water_left_over (players : ℕ) (initial_water : ℕ) (water_per_player : ℕ) (spilled_water : ℕ) :
  players = 30 →
  initial_water = 8000 →
  water_per_player = 200 →
  spilled_water = 250 →
  initial_water - (players * water_per_player + spilled_water) = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_water_left_over_l642_64297


namespace NUMINAMATH_CALUDE_xy_product_range_l642_64289

theorem xy_product_range (x y : ℝ) : 
  x^2 * y^2 + x^2 - 10*x*y - 8*x + 16 = 0 → 0 ≤ x*y ∧ x*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_range_l642_64289


namespace NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_range_l642_64254

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  9 / (x + 4) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici (1/2) :=
sorry

-- Part 2
theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → 
  k > Real.sqrt 2 ∨ k < -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_range_l642_64254


namespace NUMINAMATH_CALUDE_tie_cost_l642_64269

theorem tie_cost (pants_cost shirt_cost paid change : ℕ) 
  (h1 : pants_cost = 140)
  (h2 : shirt_cost = 43)
  (h3 : paid = 200)
  (h4 : change = 2) :
  paid - change - (pants_cost + shirt_cost) = 15 := by
sorry

end NUMINAMATH_CALUDE_tie_cost_l642_64269


namespace NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l642_64272

-- Define the triangles and their properties
def triangle_ABC (a : ℝ) := {BC : ℝ // BC = 2 ∧ ∃ (AC : ℝ), AC = a}
def triangle_ABD := {AD : ℝ // AD = 3}

-- Define the theorem
theorem right_triangles_common_hypotenuse (a : ℝ) 
  (h : a ≥ Real.sqrt 5) -- Ensure BD is real
  (ABC : triangle_ABC a) (ABD : triangle_ABD) :
  ∃ (BD : ℝ), BD = Real.sqrt (a^2 - 5) :=
sorry

end NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l642_64272


namespace NUMINAMATH_CALUDE_taller_tree_is_84_feet_l642_64286

def taller_tree_height (h1 h2 : ℝ) : Prop :=
  h1 > h2 ∧ h1 - h2 = 24 ∧ h2 / h1 = 5 / 7

theorem taller_tree_is_84_feet :
  ∃ (h1 h2 : ℝ), taller_tree_height h1 h2 ∧ h1 = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_taller_tree_is_84_feet_l642_64286


namespace NUMINAMATH_CALUDE_square_circumscribed_circle_radius_l642_64287

/-- Given a square with perimeter x and circumscribed circle radius y, prove that y = (√2 / 8) * x -/
theorem square_circumscribed_circle_radius (x y : ℝ) 
  (h_perimeter : x > 0) -- Ensure positive perimeter
  (h_square : ∃ (s : ℝ), s > 0 ∧ 4 * s = x) -- Existence of square side length
  (h_circumscribed : y > 0) -- Ensure positive radius
  : y = (Real.sqrt 2 / 8) * x := by
  sorry

end NUMINAMATH_CALUDE_square_circumscribed_circle_radius_l642_64287


namespace NUMINAMATH_CALUDE_investment_years_equals_three_l642_64252

/-- Calculates the number of years for which a principal is invested, given the interest rate,
    principal amount, and the difference between the principal and interest. -/
def calculate_investment_years (rate : ℚ) (principal : ℚ) (principal_minus_interest : ℚ) : ℚ :=
  (principal - principal_minus_interest) / (principal * rate / 100)

theorem investment_years_equals_three :
  let rate : ℚ := 12
  let principal : ℚ := 9200
  let principal_minus_interest : ℚ := 5888
  calculate_investment_years rate principal principal_minus_interest = 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_years_equals_three_l642_64252


namespace NUMINAMATH_CALUDE_hash_property_l642_64204

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem stating that if a # b = 100, then (a + b) + 6 = 11 -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l642_64204


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l642_64210

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 3 = 0
    passes through the point (1, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- L1 equation
  (2*1 + 1 - 3 = 0) ∧  -- L2 passes through (1, 1)
  (1 * 2 + (-2) * 1 = 0)  -- L1 and L2 are perpendicular (slope product = -1)
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l642_64210


namespace NUMINAMATH_CALUDE_percentage_theorem_l642_64284

theorem percentage_theorem (y x z : ℝ) (h : y * x^2 + 3 * z - 6 > 0) :
  ((2 * (y * x^2 + 3 * z - 6)) / 5 + (3 * (y * x^2 + 3 * z - 6)) / 10) / (y * x^2 + 3 * z - 6) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_theorem_l642_64284


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l642_64291

theorem largest_angle_of_triangle (y : ℝ) : 
  45 + 60 + y = 180 →
  max (max 45 60) y = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l642_64291


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l642_64225

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a ≠ 0 → 
  |r| < 1 → 
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum : 
  (∑' n, (1 : ℝ) * (1/4 : ℝ)^n) = 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l642_64225


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l642_64217

theorem no_solution_to_equation :
  ¬∃ x : ℝ, (1 / (x + 8) + 1 / (x + 5) + 1 / (x + 1) = 1 / (x + 11) + 1 / (x + 2) + 1 / (x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l642_64217


namespace NUMINAMATH_CALUDE_mary_card_count_l642_64255

/-- The number of Pokemon cards Mary has after receiving gifts from Sam and Alex -/
def final_card_count (initial_cards torn_cards sam_gift alex_gift : ℕ) : ℕ :=
  initial_cards - torn_cards + sam_gift + alex_gift

/-- Theorem stating that Mary has 196 Pokemon cards after the events described -/
theorem mary_card_count : 
  final_card_count 123 18 56 35 = 196 := by
  sorry

end NUMINAMATH_CALUDE_mary_card_count_l642_64255


namespace NUMINAMATH_CALUDE_no_solution_exists_l642_64267

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a^2 + 3*b^2 + 2 = 3*a*b := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l642_64267


namespace NUMINAMATH_CALUDE_margo_pairing_probability_l642_64215

/-- The probability of a specific pairing in a class with random pairings -/
def pairingProbability (totalStudents : ℕ) (favorableOutcomes : ℕ) : ℚ :=
  favorableOutcomes / (totalStudents - 1)

/-- Theorem: The probability of Margo being paired with either Irma or Julia -/
theorem margo_pairing_probability :
  let totalStudents : ℕ := 32
  let favorableOutcomes : ℕ := 2
  pairingProbability totalStudents favorableOutcomes = 2 / 31 := by
  sorry

end NUMINAMATH_CALUDE_margo_pairing_probability_l642_64215


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l642_64233

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, 4 * x^2 - 40 * x + 48 = (a * x + b)^2 + c) ∧
    a * b = -20 ∧
    c = -52 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l642_64233


namespace NUMINAMATH_CALUDE_polynomial_simplification_l642_64243

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 3 * x^4 - 5 * x^3 + 2 * x^2 - 10 * x + 8) + 
  (-3 * x^5 - x^4 + 4 * x^3 - 2 * x^2 + 15 * x - 12) = 
  -x^5 + 2 * x^4 - x^3 + 5 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l642_64243


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_always_power_of_two_l642_64231

theorem no_quadratic_trinomial_always_power_of_two : 
  ¬ ∃ (a b c : ℤ), ∀ (x : ℕ), ∃ (n : ℕ), a * x^2 + b * x + c = 2^n := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_always_power_of_two_l642_64231


namespace NUMINAMATH_CALUDE_bug_meeting_point_l642_64240

theorem bug_meeting_point (PQ QR RP : ℝ) (h1 : PQ = 7) (h2 : QR = 8) (h3 : RP = 9) :
  let perimeter := PQ + QR + RP
  let distance_traveled := 10
  let QS := distance_traveled - PQ
  QS = 3 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l642_64240


namespace NUMINAMATH_CALUDE_square_overlap_area_l642_64295

theorem square_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.cos β = 3/5) :
  let square_side : Real := 2
  let overlap_area : Real := 
    2 * (square_side * (1 - Real.tan (β/2)) / (1 + Real.tan (β/2))) * square_side / 2
  overlap_area = 4/3 := by sorry

end NUMINAMATH_CALUDE_square_overlap_area_l642_64295


namespace NUMINAMATH_CALUDE_f_increasing_neg_f_max_neg_l642_64288

/-- An odd function that is increasing on [3, 7] with minimum value 5 -/
def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f is increasing on [3, 7] -/
axiom f_increasing_pos : ∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 7 → f x < f y

/-- The minimum value of f on [3, 7] is 5 -/
axiom f_min_pos : ∃ x₀, 3 ≤ x₀ ∧ x₀ ≤ 7 ∧ f x₀ = 5 ∧ ∀ x, 3 ≤ x ∧ x ≤ 7 → f x₀ ≤ f x

/-- f is increasing on [-7, -3] -/
theorem f_increasing_neg : ∀ x y, -7 ≤ x ∧ x < y ∧ y ≤ -3 → f x < f y :=
sorry

/-- The maximum value of f on [-7, -3] is -5 -/
theorem f_max_neg : ∃ x₀, -7 ≤ x₀ ∧ x₀ ≤ -3 ∧ f x₀ = -5 ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ f x₀ :=
sorry

end NUMINAMATH_CALUDE_f_increasing_neg_f_max_neg_l642_64288


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l642_64214

theorem absolute_value_equation_solution : 
  ∃! x : ℝ, |x - 30| + |x - 25| = |2*x - 50| + 5 ∧ x = 32.5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l642_64214


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l642_64296

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l642_64296


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l642_64235

/-- The height of a cylinder with radius 12 inches that has the same volume as a sphere with radius 3 inches is 1/4 inch. -/
theorem melted_ice_cream_height : 
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h →
  h = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_melted_ice_cream_height_l642_64235


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l642_64228

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_unique_parameters :
  ∀ ξ : BinomialRV, expectation ξ = 12 → variance ξ = 2.4 → ξ.n = 15 ∧ ξ.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l642_64228


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l642_64201

theorem buckingham_palace_visitors :
  let current_day_visitors : ℕ := 132
  let previous_day_visitors : ℕ := 274
  let total_visitors : ℕ := 406
  let days_considered : ℕ := 2
  current_day_visitors + previous_day_visitors = total_visitors →
  days_considered = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l642_64201


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l642_64281

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (3 * x^2 + 2 * x + 1) = 3} = {4/3, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l642_64281


namespace NUMINAMATH_CALUDE_angle_c_measure_l642_64200

theorem angle_c_measure (A B C : ℝ) : 
  A = 86 →
  B = 3 * C + 22 →
  A + B + C = 180 →
  C = 18 := by
sorry

end NUMINAMATH_CALUDE_angle_c_measure_l642_64200


namespace NUMINAMATH_CALUDE_base_equality_implies_three_l642_64257

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from an arbitrary base to base 10 -/
def baseNToBase10 (n b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem base_equality_implies_three :
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 35 = baseNToBase10 132 b :=
by
  sorry

end NUMINAMATH_CALUDE_base_equality_implies_three_l642_64257


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l642_64275

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    its center coordinates are (-2, 1) and its radius is 3 -/
theorem circle_center_and_radius : 
  ∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  ∃ (h k r : ℝ), h = -2 ∧ k = 1 ∧ r = 3 ∧
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l642_64275


namespace NUMINAMATH_CALUDE_sandbox_side_length_l642_64273

/-- Represents the properties of a square sandbox. -/
structure Sandbox where
  sandPerArea : Real  -- Pounds of sand per square inch
  totalSand : Real    -- Total pounds of sand needed
  sideLength : Real   -- Length of each side in inches

/-- 
Theorem: Given a square sandbox where 30 pounds of sand fills 80 square inches,
and 600 pounds of sand fills the entire sandbox, the length of each side is 40 inches.
-/
theorem sandbox_side_length (sb : Sandbox)
  (h1 : sb.sandPerArea = 30 / 80)
  (h2 : sb.totalSand = 600) :
  sb.sideLength = 40 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_side_length_l642_64273


namespace NUMINAMATH_CALUDE_complex_equality_proof_l642_64213

theorem complex_equality_proof (n : ℤ) (h : 0 ≤ n ∧ n ≤ 13) : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.cos (2 * n * π / 14) + Complex.I * Complex.sin (2 * n * π / 14) → n = 5 :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_proof_l642_64213


namespace NUMINAMATH_CALUDE_percentage_loss_l642_64278

theorem percentage_loss (cost_price selling_price : ℝ) : 
  cost_price = 750 → 
  selling_price = 675 → 
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_l642_64278


namespace NUMINAMATH_CALUDE_rectangle_square_division_l642_64229

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ 
    (∃ (p q : ℕ), p > q ∧ 
      (a * b / n).sqrt / (a * b / (n + 76)).sqrt = p / q)) → 
  n = 324 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_division_l642_64229


namespace NUMINAMATH_CALUDE_right_triangle_area_l642_64283

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by x-axis, y-axis, and a line -/
structure RightTriangle where
  line : Line

/-- Calculate the area of a right triangle -/
def area (t : RightTriangle) : ℝ :=
  sorry

theorem right_triangle_area :
  let l := Line.mk (-4, 8) (-8, 4)
  let t := RightTriangle.mk l
  area t = 72 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l642_64283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l642_64208

theorem arithmetic_sequence_inequality (a₁ : ℝ) (d : ℝ) :
  (∀ n : Fin 8, a₁ + (n : ℕ) * d > 0) →
  d ≠ 0 →
  (a₁ * (a₁ + 7 * d)) < ((a₁ + 3 * d) * (a₁ + 4 * d)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l642_64208


namespace NUMINAMATH_CALUDE_parallel_lines_circle_chord_l642_64268

/-- Given three equally spaced parallel lines intersecting a circle, creating chords of lengths 38, 38, and 34, the distance between two adjacent parallel lines is 6. -/
theorem parallel_lines_circle_chord (r : ℝ) : 
  let chord1 : ℝ := 38
  let chord2 : ℝ := 38
  let chord3 : ℝ := 34
  let d : ℝ := 6
  38 * r^2 = 722 + (19/4) * d^2 ∧ 
  34 * r^2 = 578 + (153/4) * d^2 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_chord_l642_64268


namespace NUMINAMATH_CALUDE_range_of_a_min_value_expression_equality_condition_l642_64223

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| + |x - 20|

-- Define the property that the solution set is not empty
def solution_set_nonempty (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < 10 * a + 10

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : solution_set_nonempty a → a > 0 := by sorry

-- Theorem for the minimum value of a + 4/a^2
theorem min_value_expression (a : ℝ) (h : a > 0) :
  a + 4 / a^2 ≥ 3 := by sorry

-- Theorem for the equality condition
theorem equality_condition (a : ℝ) (h : a > 0) :
  a + 4 / a^2 = 3 ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_expression_equality_condition_l642_64223


namespace NUMINAMATH_CALUDE_log_product_equals_four_l642_64241

-- Define a variable k that is positive and not equal to 1
variable (k : ℝ) (hk : k > 0 ∧ k ≠ 1)

-- Define x as a positive real number
variable (x : ℝ) (hx : x > 0)

-- State the theorem
theorem log_product_equals_four (h : Real.log x / Real.log k * Real.log k / Real.log 7 = 4) : 
  x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_l642_64241


namespace NUMINAMATH_CALUDE_prime_sum_product_l642_64290

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l642_64290


namespace NUMINAMATH_CALUDE_vector_calculation_l642_64218

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_calculation :
  (1/3 : ℝ) • a - (4/3 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l642_64218


namespace NUMINAMATH_CALUDE_gear_teeth_problem_l642_64249

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_teeth_problem_l642_64249


namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l642_64251

theorem cubic_sum_of_quadratic_roots : 
  ∀ a b : ℝ, 
  (3 * a^2 - 5 * a + 7 = 0) → 
  (3 * b^2 - 5 * b + 7 = 0) → 
  (a ≠ b) →
  (a^3 / b^3 + b^3 / a^3 = -190 / 343) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l642_64251


namespace NUMINAMATH_CALUDE_percentage_of_75_to_125_l642_64279

theorem percentage_of_75_to_125 : ∀ (x : ℝ), x = (75 : ℝ) / (125 : ℝ) * 100 → x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_75_to_125_l642_64279


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l642_64259

/-- Given that the solution set of ax² + bx + 1 > 0 is {x | -1 < x < 1/3}, prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l642_64259


namespace NUMINAMATH_CALUDE_characterize_inequality_l642_64245

theorem characterize_inequality (x y : ℝ) :
  x^2 * y - y ≥ 0 ↔ (y ≥ 0 ∧ abs x ≥ 1) ∨ (y ≤ 0 ∧ abs x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_characterize_inequality_l642_64245


namespace NUMINAMATH_CALUDE_polynomial_factorization_l642_64202

theorem polynomial_factorization (x : ℝ) : x^4 + 256 = (x^2 - 8*x + 16) * (x^2 + 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l642_64202


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_diameter_l642_64244

/-- Given a right triangle inscribed in a circle with legs of lengths 6 and 8,
    the diameter of the circle is 10. -/
theorem inscribed_right_triangle_diameter :
  ∀ (circle : Real → Real → Prop) (triangle : Real → Real → Real → Prop),
    (∃ (x y z : Real), triangle x y z ∧ x^2 + y^2 = z^2) →  -- Right triangle condition
    (∃ (a b : Real), triangle 6 8 a) →  -- Leg lengths condition
    (∀ (p q r : Real), triangle p q r → circle p q) →  -- Triangle inscribed in circle
    (∃ (d : Real), d = 10 ∧ ∀ (p q : Real), circle p q → (p - q)^2 ≤ d^2) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_diameter_l642_64244


namespace NUMINAMATH_CALUDE_elmer_pond_maturation_rate_l642_64263

/-- The rate at which pollywogs mature and leave the pond -/
def maturation_rate (
  initial_pollywogs : ℕ
  ) (
  days_to_disappear : ℕ
  ) (
  catch_rate : ℕ
  ) (
  catch_days : ℕ
  ) : ℚ :=
  (initial_pollywogs - catch_rate * catch_days) / days_to_disappear

/-- Theorem stating the maturation rate of pollywogs in Elmer's pond -/
theorem elmer_pond_maturation_rate :
  maturation_rate 2400 44 10 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_elmer_pond_maturation_rate_l642_64263


namespace NUMINAMATH_CALUDE_fraction_simplification_l642_64270

theorem fraction_simplification : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l642_64270


namespace NUMINAMATH_CALUDE_puppy_sleeps_16_hours_l642_64292

def connor_sleep_time : ℕ := 6

def luke_sleep_time (connor_sleep_time : ℕ) : ℕ := connor_sleep_time + 2

def puppy_sleep_time (luke_sleep_time : ℕ) : ℕ := 2 * luke_sleep_time

theorem puppy_sleeps_16_hours :
  puppy_sleep_time (luke_sleep_time connor_sleep_time) = 16 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sleeps_16_hours_l642_64292


namespace NUMINAMATH_CALUDE_triangle_inequality_l642_64205

/-- Theorem: In a triangle with two sides of lengths 3 and 8, the third side is between 5 and 11 -/
theorem triangle_inequality (a b c : ℝ) : a = 3 ∧ b = 8 → 5 < c ∧ c < 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l642_64205


namespace NUMINAMATH_CALUDE_inequality_proof_l642_64298

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) 
  (h5 : a + b + c + d = 8) : 
  (a^3 / (a^2 + b + c)) + (b^3 / (b^2 + c + d)) + 
  (c^3 / (c^2 + d + a)) + (d^3 / (d^2 + a + b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l642_64298


namespace NUMINAMATH_CALUDE_four_distinct_three_digit_numbers_with_sum_divisibility_l642_64234

theorem four_distinct_three_digit_numbers_with_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (b + c + d) % a = 0 ∧
    (a + c + d) % b = 0 ∧
    (a + b + d) % c = 0 ∧
    (a + b + c) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_three_digit_numbers_with_sum_divisibility_l642_64234


namespace NUMINAMATH_CALUDE_numbers_with_seven_in_range_l642_64277

/-- The count of natural numbers from 1 to 800 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := 152

/-- The total count of numbers from 1 to 800 -/
def total_count : ℕ := 800

/-- The count of numbers from 1 to 800 without the digit 7 -/
def count_without_seven : ℕ := 648

theorem numbers_with_seven_in_range :
  count_numbers_with_seven = total_count - count_without_seven :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_seven_in_range_l642_64277


namespace NUMINAMATH_CALUDE_x_equals_two_l642_64239

-- Define the * operation
def star (a b : ℕ) : ℕ := 
  Finset.sum (Finset.range b) (λ i => a + i)

-- State the theorem
theorem x_equals_two : 
  ∃ x : ℕ, star x 10 = 65 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_x_equals_two_l642_64239


namespace NUMINAMATH_CALUDE_volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l642_64212

-- Define conversion factors
def m_to_dm : ℝ := 10
def L_to_mL : ℝ := 1000
def dm_to_cm : ℝ := 10

-- Theorem statements
theorem volume_conversion_m_to_dm : 
  20 * (m_to_dm ^ 3) = 20000 := by sorry

theorem volume_conversion_mL_to_L : 
  15 / L_to_mL = 0.015 := by sorry

theorem volume_conversion_cm_to_dm : 
  1200 / (dm_to_cm ^ 3) = 1.2 := by sorry

end NUMINAMATH_CALUDE_volume_conversion_m_to_dm_volume_conversion_mL_to_L_volume_conversion_cm_to_dm_l642_64212


namespace NUMINAMATH_CALUDE_product_of_x_values_l642_64207

theorem product_of_x_values (x : ℝ) : 
  (|18 / x - 6| = 3) → (∃ y : ℝ, y ≠ x ∧ |18 / y - 6| = 3 ∧ x * y = 12) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l642_64207


namespace NUMINAMATH_CALUDE_joan_makes_ten_ham_sandwiches_l642_64250

/-- Represents the number of slices of cheese required for each type of sandwich -/
structure SandwichRecipe where
  ham_cheese_slices : ℕ
  grilled_cheese_slices : ℕ

/-- Represents the sandwich making scenario -/
structure SandwichScenario where
  recipe : SandwichRecipe
  total_cheese_slices : ℕ
  grilled_cheese_count : ℕ

/-- Calculates the number of ham sandwiches made -/
def ham_sandwiches_made (scenario : SandwichScenario) : ℕ :=
  (scenario.total_cheese_slices - scenario.grilled_cheese_count * scenario.recipe.grilled_cheese_slices) / scenario.recipe.ham_cheese_slices

/-- Theorem stating that Joan makes 10 ham sandwiches -/
theorem joan_makes_ten_ham_sandwiches (scenario : SandwichScenario) 
  (h1 : scenario.recipe.ham_cheese_slices = 2)
  (h2 : scenario.recipe.grilled_cheese_slices = 3)
  (h3 : scenario.total_cheese_slices = 50)
  (h4 : scenario.grilled_cheese_count = 10) :
  ham_sandwiches_made scenario = 10 := by
  sorry

#eval ham_sandwiches_made { 
  recipe := { ham_cheese_slices := 2, grilled_cheese_slices := 3 },
  total_cheese_slices := 50,
  grilled_cheese_count := 10
}

end NUMINAMATH_CALUDE_joan_makes_ten_ham_sandwiches_l642_64250


namespace NUMINAMATH_CALUDE_no_isosceles_triangle_36_degree_l642_64230

theorem no_isosceles_triangle_36_degree (a b : ℕ+) : ¬ ∃ θ : ℝ,
  θ = 36 * π / 180 ∧
  (a : ℝ) * ((5 : ℝ).sqrt - 1) / 2 = b :=
sorry

end NUMINAMATH_CALUDE_no_isosceles_triangle_36_degree_l642_64230


namespace NUMINAMATH_CALUDE_fib_arithmetic_seq_solution_l642_64280

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property of three consecutive Fibonacci numbers forming an arithmetic sequence -/
def is_fib_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

theorem fib_arithmetic_seq_solution :
  ∃ a b c : ℕ, is_fib_arithmetic_seq a b c ∧ a + b + c = 3000 ∧ a = 998 := by
  sorry


end NUMINAMATH_CALUDE_fib_arithmetic_seq_solution_l642_64280


namespace NUMINAMATH_CALUDE_eric_park_time_ratio_l642_64203

/-- The ratio of Eric's return time to his time to reach the park is 3:1 -/
theorem eric_park_time_ratio :
  let time_to_park : ℕ := 20 + 10  -- Time to reach the park (running + jogging)
  let time_to_return : ℕ := 90     -- Time to return home
  (time_to_return : ℚ) / time_to_park = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_eric_park_time_ratio_l642_64203


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l642_64216

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3 →
  a 1 * a 11 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l642_64216


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l642_64260

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) / (1 + Complex.I) = (a : ℂ) + (b : ℂ) * Complex.I → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l642_64260


namespace NUMINAMATH_CALUDE_force_on_smooth_surface_with_pulleys_l642_64264

/-- The force required to move a mass on a smooth horizontal surface using a pulley system -/
theorem force_on_smooth_surface_with_pulleys 
  (m : ℝ) -- mass in kg
  (g : ℝ) -- acceleration due to gravity in m/s²
  (h_m_pos : m > 0)
  (h_g_pos : g > 0) :
  ∃ F : ℝ, F = 4 * m * g :=
by sorry

end NUMINAMATH_CALUDE_force_on_smooth_surface_with_pulleys_l642_64264


namespace NUMINAMATH_CALUDE_solid_is_cone_l642_64226

-- Define the properties of the solid
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

-- Define what it means for a view to be an isosceles triangle
def is_isosceles_triangle (view : Type) : Prop := sorry

-- Define what it means for a view to be a circle
def is_circle (view : Type) : Prop := sorry

-- Define what it means for a solid to be a cone
def is_cone (s : Solid) : Prop := sorry

-- State the theorem
theorem solid_is_cone (s : Solid) 
  (h1 : is_isosceles_triangle s.front_view)
  (h2 : is_isosceles_triangle s.side_view)
  (h3 : is_circle s.top_view) :
  is_cone s := by sorry

end NUMINAMATH_CALUDE_solid_is_cone_l642_64226


namespace NUMINAMATH_CALUDE_tangent_identity_l642_64206

theorem tangent_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) ≠ 0) (h2 : Real.tan (α - β) ≠ 0) :
  (Real.tan α + Real.tan β) / Real.tan (α + β) + 
  (Real.tan α - Real.tan β) / Real.tan (α - β) + 
  2 * (Real.tan α)^2 = 2 / (Real.cos α)^2 := by
sorry

end NUMINAMATH_CALUDE_tangent_identity_l642_64206


namespace NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_all_reals_l642_64282

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | f 0 x > 0} = {x : ℝ | -2 < x ∧ x < 1} :=
sorry

-- Part 2: Range of m when solution set is ℝ
theorem solution_set_all_reals (m : ℝ) :
  ({x : ℝ | f m x > 0} = Set.univ) ↔ (1 < m ∧ m < 9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_solution_set_all_reals_l642_64282


namespace NUMINAMATH_CALUDE_game_draw_probability_l642_64293

theorem game_draw_probability (amy_win lily_win eve_win draw : ℚ) : 
  amy_win = 2/5 → lily_win = 1/5 → eve_win = 1/10 → 
  amy_win + lily_win + eve_win + draw = 1 →
  draw = 3/10 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l642_64293


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l642_64221

theorem cubic_equation_solutions :
  ∀ x y : ℤ, x^3 = y^3 + 2*y^2 + 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l642_64221


namespace NUMINAMATH_CALUDE_linearDependence_l642_64276

/-- Two 2D vectors -/
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

/-- The set of vectors is linearly dependent iff there exist non-zero scalars a and b
    such that a * v1 + b * v2 = 0 -/
def isLinearlyDependent (k : ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * v1 i + b * v2 k i = 0)

theorem linearDependence (k : ℝ) : isLinearlyDependent k ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_linearDependence_l642_64276


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l642_64265

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l642_64265


namespace NUMINAMATH_CALUDE_tom_profit_l642_64266

/-- Calculates the profit from a stock transaction -/
def calculate_profit (
  initial_shares : ℕ
  ) (initial_price : ℚ
  ) (sold_shares : ℕ
  ) (selling_price : ℚ
  ) (remaining_shares_value_multiplier : ℚ
  ) : ℚ :=
  let total_cost := initial_shares * initial_price
  let revenue_from_sold := sold_shares * selling_price
  let revenue_from_remaining := (initial_shares - sold_shares) * (initial_price * remaining_shares_value_multiplier)
  let total_revenue := revenue_from_sold + revenue_from_remaining
  total_revenue - total_cost

/-- Tom's stock transaction profit is $40 -/
theorem tom_profit : 
  calculate_profit 20 3 10 4 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_l642_64266


namespace NUMINAMATH_CALUDE_henrys_scores_l642_64220

theorem henrys_scores (G M : ℝ) (h1 : G + M + 66 + (G + M + 66) / 3 = 248) : G + M = 120 := by
  sorry

end NUMINAMATH_CALUDE_henrys_scores_l642_64220


namespace NUMINAMATH_CALUDE_chairs_for_play_l642_64209

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end NUMINAMATH_CALUDE_chairs_for_play_l642_64209


namespace NUMINAMATH_CALUDE_complex_expression_value_l642_64238

theorem complex_expression_value : 
  let expr := (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) * Real.exp 3.5 + Real.log (Real.sin 0.785)
  ∃ ε > 0, |expr - 15563.91492641| < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_value_l642_64238


namespace NUMINAMATH_CALUDE_trains_meet_time_trains_meet_time_approx_l642_64242

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
theorem trains_meet_time (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := initial_distance + length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  total_distance / relative_speed

/-- The time for two trains to meet is approximately 6.69 seconds. -/
theorem trains_meet_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |trains_meet_time 90 95 250 64 92 - 6.69| < ε :=
sorry

end NUMINAMATH_CALUDE_trains_meet_time_trains_meet_time_approx_l642_64242


namespace NUMINAMATH_CALUDE_cafeteria_apples_l642_64294

theorem cafeteria_apples (apples_to_students : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_to_students = 42 →
  num_pies = 9 →
  apples_per_pie = 6 →
  apples_to_students + num_pies * apples_per_pie = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l642_64294


namespace NUMINAMATH_CALUDE_special_polygon_properties_l642_64248

/-- A polygon where the sum of interior angles is 1/4 more than the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  h : (n - 2) * 180 = 360 + (1/4) * 360

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 12 ∧ num_diagonals p.n = 54 := by
  sorry

#check special_polygon_properties

end NUMINAMATH_CALUDE_special_polygon_properties_l642_64248


namespace NUMINAMATH_CALUDE_smallest_n_for_odd_digits_of_9997n_l642_64227

def all_digits_odd (m : ℕ) : Prop :=
  ∀ d, d ∈ m.digits 10 → d % 2 = 1

theorem smallest_n_for_odd_digits_of_9997n :
  ∀ n : ℕ, n > 1 →
    (all_digits_odd (9997 * n) ↔ n ≥ 3335) ∧
    all_digits_odd (9997 * 3335) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_odd_digits_of_9997n_l642_64227


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l642_64246

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  groupSize : ℕ
  numberFrom16thGroup : ℕ

/-- The number drawn from the first group in a systematic sampling scheme -/
def numberFromFirstGroup (s : SystematicSampling) : ℕ :=
  (s.numberFrom16thGroup - 1) % s.groupSize + 1

/-- Theorem stating the result of the systematic sampling problem -/
theorem systematic_sampling_result (s : SystematicSampling) 
  (h1 : s.totalStudents = 160)
  (h2 : s.sampleSize = 20)
  (h3 : s.groupSize = 8)
  (h4 : s.numberFrom16thGroup = 126) :
  numberFromFirstGroup s = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l642_64246


namespace NUMINAMATH_CALUDE_xyz_value_l642_64224

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 80 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l642_64224


namespace NUMINAMATH_CALUDE_ratio_evaluation_and_closest_integer_l642_64219

theorem ratio_evaluation_and_closest_integer : 
  let r := (2^3000 + 2^3003) / (2^3001 + 2^3002)
  r = 3/2 ∧ ∀ n : ℤ, |r - 2| ≤ |r - n| :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_and_closest_integer_l642_64219


namespace NUMINAMATH_CALUDE_penguin_fish_consumption_l642_64247

theorem penguin_fish_consumption (initial_size : ℕ) 
  (h1 : initial_size * 2 * 3 + 129 = 1077) 
  (h2 : 237 / initial_size = 3 / 2) : 
  ∃ (fish_per_penguin : ℚ), fish_per_penguin = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_penguin_fish_consumption_l642_64247


namespace NUMINAMATH_CALUDE_ratio_HC_JE_l642_64285

-- Define the points
variable (A B C D E F G H J K : ℝ × ℝ)

-- Define the conditions
axiom points_on_line : ∃ (t : ℝ), A = (0, 0) ∧ B = (1, 0) ∧ C = (3, 0) ∧ D = (4, 0) ∧ E = (5, 0) ∧ F = (7, 0)
axiom G_off_line : G.2 ≠ 0
axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)
axiom J_on_GF : ∃ (t : ℝ), J = G + t • (F - G)
axiom K_on_GB : ∃ (t : ℝ), K = G + t • (B - G)
axiom parallel_lines : ∃ (k : ℝ), 
  H - C = k • (G - A) ∧ 
  J - E = k • (G - A) ∧ 
  K - B = k • (G - A)

-- Define the theorem
theorem ratio_HC_JE : 
  (H.1 - C.1) / (J.1 - E.1) = 7/8 :=
sorry

end NUMINAMATH_CALUDE_ratio_HC_JE_l642_64285


namespace NUMINAMATH_CALUDE_unique_intersection_l642_64271

/-- The quadratic function f(x) = bx^2 + bx + 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + b * x + 2

/-- The linear function g(x) = 2x + 4 -/
def g (x : ℝ) : ℝ := 2 * x + 4

/-- The discriminant of the quadratic equation resulting from equating f and g -/
def discriminant (b : ℝ) : ℝ := (b - 2)^2 + 8 * b

theorem unique_intersection (b : ℝ) : 
  (∃! x, f b x = g x) ↔ b = -2 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l642_64271


namespace NUMINAMATH_CALUDE_pamela_skittles_l642_64211

theorem pamela_skittles (initial_skittles : ℕ) (given_skittles : ℕ) : 
  initial_skittles = 50 → given_skittles = 7 → initial_skittles - given_skittles = 43 := by
sorry

end NUMINAMATH_CALUDE_pamela_skittles_l642_64211


namespace NUMINAMATH_CALUDE_floor_product_equals_17_l642_64256

def solution_set : Set ℝ := Set.Ici 4.25 ∩ Set.Iio 4.5

theorem floor_product_equals_17 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 17 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_17_l642_64256


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l642_64299

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 4 * X^2 + 5 * X - 3
  let p₂ : Polynomial ℤ := 6 * X^2 - 5 * X + 1
  (p₁ * p₂).coeff 2 = -39 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l642_64299


namespace NUMINAMATH_CALUDE_min_apples_collected_l642_64236

theorem min_apples_collected (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ (p1 p2 p3 p4 p5 : ℕ), 
    p1 + p2 + p3 + p4 + p5 = 100 ∧ 
    0 < p1 ∧ p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧
    (∀ i ∈ [p1, p2, p3, p4], (i * (n * 7 / 10) % 100 = 0)))
  (h3 : ∀ m : ℕ, m < n → 
    ¬(∃ (q1 q2 q3 q4 q5 : ℕ), 
      q1 + q2 + q3 + q4 + q5 = 100 ∧ 
      0 < q1 ∧ q1 < q2 ∧ q2 < q3 ∧ q3 < q4 ∧ q4 < q5 ∧
      (∀ i ∈ [q1, q2, q3, q4], (i * (m * 7 / 10) % 100 = 0))))
  : n = 20 :=
sorry

end NUMINAMATH_CALUDE_min_apples_collected_l642_64236


namespace NUMINAMATH_CALUDE_village_new_average_age_l642_64261

/-- Represents the population data of a village --/
structure VillagePopulation where
  men_ratio : ℚ
  women_ratio : ℚ
  men_increase : ℚ
  men_avg_age : ℚ
  women_avg_age : ℚ

/-- Calculates the new average age of the population after men's population increase --/
def new_average_age (v : VillagePopulation) : ℚ :=
  let new_men_ratio := v.men_ratio * (1 + v.men_increase)
  let total_population := new_men_ratio + v.women_ratio
  let total_age := new_men_ratio * v.men_avg_age + v.women_ratio * v.women_avg_age
  total_age / total_population

/-- Theorem stating that the new average age is approximately 37.3 years --/
theorem village_new_average_age :
  let v : VillagePopulation := {
    men_ratio := 3,
    women_ratio := 4,
    men_increase := 1/10,
    men_avg_age := 40,
    women_avg_age := 35
  }
  ∃ ε > 0, |new_average_age v - 37.3| < ε :=
sorry

end NUMINAMATH_CALUDE_village_new_average_age_l642_64261
