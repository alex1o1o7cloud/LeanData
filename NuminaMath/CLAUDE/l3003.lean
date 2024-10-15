import Mathlib

namespace NUMINAMATH_CALUDE_range_of_b_length_of_AB_l3003_300362

-- Define the line and ellipse
def line (x b : ℝ) : ℝ := x + b
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the intersection condition
def intersects (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  ellipse x₁ (line x₁ b) ∧ 
  ellipse x₂ (line x₂ b)

-- Theorem for the range of b
theorem range_of_b :
  ∀ b : ℝ, intersects b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

-- Theorem for the length of AB when b = 1
theorem length_of_AB :
  intersects 1 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧
    y₁ = line x₁ 1 ∧
    y₂ = line x₂ 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_length_of_AB_l3003_300362


namespace NUMINAMATH_CALUDE_solution_exists_l3003_300384

/-- The number of primes less than or equal to n -/
def ν (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem solution_exists (m : ℕ) (hm : m > 2) :
  (∃ n : ℕ, n > 1 ∧ n / ν n = m) → (∃ n : ℕ, n > 1 ∧ n / ν n = m - 1) :=
sorry

end NUMINAMATH_CALUDE_solution_exists_l3003_300384


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3003_300398

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- right side of square is at most the x-intercept
  h3 : parabola (center + side/2) = side -- top-right corner lies on the parabola

theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3003_300398


namespace NUMINAMATH_CALUDE_problem_statements_l3003_300395

theorem problem_statements :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) ∧
  (∀ P Q : Set ℝ, ∀ a : ℝ, a ∈ P ∩ Q → a ∈ P) ∧
  (∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)) ∧
  (∀ a b c : ℝ, (1 : ℝ) = 0 ↔ a + b + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3003_300395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3003_300342

/-- Proves the general term of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h3 : ∃ m : ℝ, ∀ n : ℕ, Real.sqrt (8 * S n + 2 * n) = m + (n - 1) * d) :
  ∀ n : ℕ, a n = 4 * n - 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3003_300342


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3003_300323

/-- A quadratic function with axis of symmetry at x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : -b / (2 * a) = 1

/-- Theorem: For a quadratic function with axis of symmetry at x = 1, c < 2b -/
theorem quadratic_inequality (f : QuadraticFunction) : f.c < 2 * f.b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3003_300323


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l3003_300367

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2/12 + y^2/3 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 8*y^2 = 8

-- Define the line (l)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x+3)

-- Define the circle with PQ as diameter passing through origin
def circle_PQ_through_origin (P Q : ℝ × ℝ) : Prop :=
  (P.1 * Q.1 + P.2 * Q.2 = 0)

theorem ellipse_and_line_theorem :
  -- The ellipse passes through (-2,√2)
  ellipse (-2) (Real.sqrt 2) →
  -- The ellipse and hyperbola share the same foci
  (∀ x y, hyperbola x y ↔ x^2/8 - y^2 = 1) →
  -- For any k, if the line intersects the ellipse at P and Q
  -- and the circle with PQ as diameter passes through origin
  (∀ k P Q, 
    line k P.1 P.2 → 
    line k Q.1 Q.2 → 
    ellipse P.1 P.2 → 
    ellipse Q.1 Q.2 → 
    circle_PQ_through_origin P Q →
    -- Then k must be ±(2√11/11)
    (k = 2 * Real.sqrt 11 / 11 ∨ k = -2 * Real.sqrt 11 / 11)) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l3003_300367


namespace NUMINAMATH_CALUDE_number_ratio_l3003_300356

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3003_300356


namespace NUMINAMATH_CALUDE_number_is_composite_l3003_300360

theorem number_is_composite (n : ℕ) (h : n = 2^1000) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^n + 1 = a * b :=
sorry

end NUMINAMATH_CALUDE_number_is_composite_l3003_300360


namespace NUMINAMATH_CALUDE_cube_sum_product_l3003_300364

theorem cube_sum_product : (3^3 * 4^3) + (3^3 * 2^3) = 1944 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l3003_300364


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3003_300313

/-- In a convex hexagon ABCDEF, prove that the measure of angle D is 145 degrees
    given the following conditions:
    - Angles A, B, and C are congruent
    - Angles D, E, and F are congruent
    - Angle A is 50 degrees less than angle D -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧  -- Angles A, B, and C are congruent
  D = E ∧ E = F ∧  -- Angles D, E, and F are congruent
  A + 50 = D ∧     -- Angle A is 50 degrees less than angle D
  A + B + C + D + E + F = 720  -- Sum of angles in a hexagon
  → D = 145 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3003_300313


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3003_300376

/-- A line passing through (1,2) with equal intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → (x = 1 ∧ y = 2)) →  -- Line passes through (1,2)
    (∃ k : ℝ, k ≠ 0 ∧ a = k ∧ b = k) →                      -- Equal intercepts condition
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3003_300376


namespace NUMINAMATH_CALUDE_equation_solution_l3003_300373

theorem equation_solution : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 5) * (x - 2)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3003_300373


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3003_300338

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin α < 0)
  (h3 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3003_300338


namespace NUMINAMATH_CALUDE_tens_digit_of_17_to_1993_l3003_300368

theorem tens_digit_of_17_to_1993 : ∃ n : ℕ, 17^1993 ≡ 30 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_17_to_1993_l3003_300368


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3003_300371

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h_sum_ratio : ∀ n : ℕ, S n / T n = 3 * n / (2 * n + 1)

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences) : 
  (seq.a 1 + seq.a 2 + seq.a 14 + seq.a 19) / 
  (seq.b 1 + seq.b 3 + seq.b 17 + seq.b 19) = 17 / 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3003_300371


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3003_300351

/-- Ellipse with foci on y-axis and center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line intersecting the ellipse -/
structure Line where
  k : ℝ
  m : ℝ

/-- Definition of the problem setup -/
def EllipseProblem (E : Ellipse) (l : Line) : Prop :=
  -- Eccentricity is √3/2
  E.a / (E.a ^ 2 - E.b ^ 2).sqrt = 2 / Real.sqrt 3 ∧
  -- Perimeter of quadrilateral is 4√5
  4 * (E.a ^ 2 + E.b ^ 2).sqrt = 4 * Real.sqrt 5 ∧
  -- Line intersects ellipse at two points
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (l.k * x₁ + l.m) ^ 2 / (4 : ℝ) + x₁ ^ 2 = 1 ∧
    (l.k * x₂ + l.m) ^ 2 / (4 : ℝ) + x₂ ^ 2 = 1 ∧
  -- AP = 3PB condition
  x₁ = -3 * x₂

/-- Main theorem to prove -/
theorem ellipse_intersection_theorem (E : Ellipse) (l : Line) 
  (h : EllipseProblem E l) : 
  1 < l.m ^ 2 ∧ l.m ^ 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3003_300351


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l3003_300381

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧ 
  (abs b > 1000000) ∧ 
  (abs c > 1000000) ∧ 
  (abs d > 1000000) ∧ 
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l3003_300381


namespace NUMINAMATH_CALUDE_shelby_rainy_driving_time_l3003_300312

/-- Represents the driving scenario of Shelby --/
structure DrivingScenario where
  sunny_speed : ℝ  -- Speed in sunny conditions (mph)
  rainy_speed : ℝ  -- Speed in rainy conditions (mph)
  total_distance : ℝ  -- Total distance covered (miles)
  total_time : ℝ  -- Total time of travel (minutes)

/-- Calculates the time spent driving in rainy conditions --/
def rainy_time (scenario : DrivingScenario) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specific conditions, the rainy driving time is 40 minutes --/
theorem shelby_rainy_driving_time :
  let scenario : DrivingScenario := {
    sunny_speed := 35,
    rainy_speed := 25,
    total_distance := 22.5,
    total_time := 50
  }
  rainy_time scenario = 40 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rainy_driving_time_l3003_300312


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3003_300358

/-- A triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  /-- The base of the triangle -/
  base : ℝ
  /-- The altitude of the triangle -/
  altitude : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square's side is parallel to and lies on the triangle's base -/
  square_on_base : square_side ≤ base

/-- The area of the inscribed square -/
def square_area (t : TriangleWithInscribedSquare) : ℝ :=
  t.square_side ^ 2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area
  (t : TriangleWithInscribedSquare)
  (h1 : t.base = 12)
  (h2 : t.altitude = 7) :
  square_area t = 36 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3003_300358


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3003_300383

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)  -- Condition for ellipse foci
  (h2 : a^2 + b^2 = 49)  -- Condition for hyperbola foci
  : a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l3003_300383


namespace NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_exact_l3003_300302

/-- Represents the number of friends -/
def num_friends : ℕ := 4

/-- Represents the number of blocks each friend has -/
def num_blocks : ℕ := 6

/-- Represents the number of boxes -/
def num_boxes : ℕ := 6

/-- Represents the probability of a specific color being placed in a specific box by one friend -/
def prob_color_in_box : ℚ := 1 / num_blocks

/-- Represents the probability of three friends placing the same color in a specific box -/
def prob_three_same_color : ℚ := prob_color_in_box ^ 3

/-- Represents the probability of at least one box having all blocks of the same color -/
def prob_at_least_one_box_same_color : ℚ := 1 - (1 - num_blocks * prob_three_same_color) ^ num_boxes

theorem prob_at_least_one_box_same_color_exact : 
  prob_at_least_one_box_same_color = 517 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_exact_l3003_300302


namespace NUMINAMATH_CALUDE_inequality_solution_l3003_300363

theorem inequality_solution (x : ℝ) : 
  x > 0 → |5 - 2*x| ≤ 8 → 0 ≤ x ∧ x ≤ 6.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3003_300363


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3003_300357

/-- Given a hyperbola with equation x²-y²/b²=1 and b > 0, 
    prove that if one of its asymptote lines is 2x-y=0, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 - y^2/b^2 = 1 → (2*x - y = 0 → b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3003_300357


namespace NUMINAMATH_CALUDE_pension_formula_l3003_300397

/-- Represents the annual pension function based on years of service -/
def annual_pension (k : ℝ) (x : ℝ) : ℝ := k * x^2

/-- The pension increase after 4 additional years of service -/
def increase_4_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 4) - annual_pension k x

/-- The pension increase after 9 additional years of service -/
def increase_9_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 9) - annual_pension k x

theorem pension_formula (k : ℝ) (x : ℝ) :
  (increase_4_years k x = 144) ∧ 
  (increase_9_years k x = 324) →
  annual_pension k x = (Real.sqrt 171 / 5) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_pension_formula_l3003_300397


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_mean_equality_proof_l3003_300393

theorem mean_equality_implies_x_value : ℝ → Prop :=
  fun x =>
    (11 + 14 + 25) / 3 = (18 + x + 4) / 3 → x = 28

-- Proof
theorem mean_equality_proof : mean_equality_implies_x_value 28 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_mean_equality_proof_l3003_300393


namespace NUMINAMATH_CALUDE_permutation_remainder_l3003_300374

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of the 18-character string -/
def N : ℕ := sorry

/-- The sum of valid permutations for different arrangements -/
def permutation_sum : ℕ :=
  (choose 5 0) * (choose 5 0) * (choose 5 1) +
  (choose 5 1) * (choose 5 1) * (choose 5 2) +
  (choose 5 2) * (choose 5 2) * (choose 5 3) +
  (choose 5 3) * (choose 5 3) * (choose 5 4)

theorem permutation_remainder :
  N ≡ 755 [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_permutation_remainder_l3003_300374


namespace NUMINAMATH_CALUDE_amount_with_r_l3003_300378

theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l3003_300378


namespace NUMINAMATH_CALUDE_laborer_income_l3003_300320

/-- Represents the financial situation of a laborer over a 10-month period -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  initial_months : ℕ
  reduced_expenditure : ℝ
  reduced_months : ℕ
  savings : ℝ

/-- The theorem stating the laborer's monthly income given the conditions -/
theorem laborer_income (lf : LaborerFinances) 
  (h1 : lf.initial_expenditure = 85)
  (h2 : lf.initial_months = 6)
  (h3 : lf.reduced_expenditure = 60)
  (h4 : lf.reduced_months = 4)
  (h5 : lf.savings = 30)
  (h6 : ∃ d : ℝ, d > 0 ∧ 
        lf.monthly_income * lf.initial_months = lf.initial_expenditure * lf.initial_months - d ∧
        lf.monthly_income * lf.reduced_months = lf.reduced_expenditure * lf.reduced_months + d + lf.savings) :
  lf.monthly_income = 78 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_l3003_300320


namespace NUMINAMATH_CALUDE_tetrahedron_division_l3003_300347

-- Define a regular tetrahedron
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_positive : edge_length > 0)

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the planes drawn through division points
structure DivisionPlanes :=
  (tetrahedron : RegularTetrahedron)
  (num_divisions : ℕ)
  (parallel_to_faces : Bool)

-- Define the number of parts the tetrahedron is divided into
def num_parts (t : RegularTetrahedron) (d : DivisionPlanes) : ℕ := 15

-- Theorem statement
theorem tetrahedron_division (t : RegularTetrahedron) :
  let d := DivisionPlanes.mk t (divide_edges t) true
  num_parts t d = 15 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l3003_300347


namespace NUMINAMATH_CALUDE_smallest_three_digit_power_of_two_plus_one_multiple_of_five_l3003_300305

theorem smallest_three_digit_power_of_two_plus_one_multiple_of_five :
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N ≤ 999) ∧ 
    (2^N + 1) % 5 = 0 ∧
    (∀ (M : ℕ), (100 ≤ M ∧ M < N) → (2^M + 1) % 5 ≠ 0) ∧
    N = 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_power_of_two_plus_one_multiple_of_five_l3003_300305


namespace NUMINAMATH_CALUDE_circus_tent_sections_l3003_300325

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) (h1 : section_capacity = 246) (h2 : total_capacity = 984) :
  total_capacity / section_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_sections_l3003_300325


namespace NUMINAMATH_CALUDE_find_K_l3003_300324

theorem find_K : ∃ K : ℕ, 32^5 * 4^5 = 2^K ∧ K = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_K_l3003_300324


namespace NUMINAMATH_CALUDE_mozzarella_amount_proof_l3003_300326

/-- The cost of the special blend cheese in dollars per kilogram -/
def special_blend_cost : ℝ := 696.05

/-- The cost of mozzarella cheese in dollars per kilogram -/
def mozzarella_cost : ℝ := 504.35

/-- The cost of romano cheese in dollars per kilogram -/
def romano_cost : ℝ := 887.75

/-- The amount of romano cheese used in kilograms -/
def romano_amount : ℝ := 18.999999999999986

/-- The amount of mozzarella cheese used in kilograms -/
def mozzarella_amount : ℝ := 19

theorem mozzarella_amount_proof :
  ∃ (m : ℝ), abs (m - mozzarella_amount) < 0.1 ∧
  m * mozzarella_cost + romano_amount * romano_cost =
  (m + romano_amount) * special_blend_cost :=
sorry

end NUMINAMATH_CALUDE_mozzarella_amount_proof_l3003_300326


namespace NUMINAMATH_CALUDE_exam_average_theorem_l3003_300370

def average_percentage (group1_count : ℕ) (group1_avg : ℚ) (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_points : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_points / total_count

theorem exam_average_theorem :
  average_percentage 15 (80/100) 10 (90/100) = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_theorem_l3003_300370


namespace NUMINAMATH_CALUDE_apples_eaten_by_two_children_l3003_300307

/-- Proves that given 5 children who each collected 15 apples, if one child sold 7 apples
    and they had 60 apples left when they got home, then two children ate a total of 8 apples. -/
theorem apples_eaten_by_two_children
  (num_children : Nat)
  (apples_per_child : Nat)
  (apples_sold : Nat)
  (apples_left : Nat)
  (h1 : num_children = 5)
  (h2 : apples_per_child = 15)
  (h3 : apples_sold = 7)
  (h4 : apples_left = 60) :
  ∃ (eaten_by_two : Nat), eaten_by_two = 8 ∧
    num_children * apples_per_child = apples_left + apples_sold + eaten_by_two :=
by sorry


end NUMINAMATH_CALUDE_apples_eaten_by_two_children_l3003_300307


namespace NUMINAMATH_CALUDE_journey_distance_proof_l3003_300319

def total_journey_time : Real := 2.5
def first_segment_time : Real := 0.5
def first_segment_speed : Real := 20
def break_time : Real := 0.25
def second_segment_time : Real := 1
def second_segment_speed : Real := 30
def third_segment_speed : Real := 15

theorem journey_distance_proof :
  let first_segment_distance := first_segment_time * first_segment_speed
  let second_segment_distance := second_segment_time * second_segment_speed
  let third_segment_time := total_journey_time - (first_segment_time + break_time + second_segment_time)
  let third_segment_distance := third_segment_time * third_segment_speed
  let total_distance := first_segment_distance + second_segment_distance + third_segment_distance
  total_distance = 51.25 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_proof_l3003_300319


namespace NUMINAMATH_CALUDE_sum_difference_1500_l3003_300391

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 : sumDifference 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_1500_l3003_300391


namespace NUMINAMATH_CALUDE_composite_ratio_l3003_300328

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_list first_six_composites : Rat) / (product_list next_six_composites) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l3003_300328


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3003_300336

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 5 * π / 12
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * (Real.sqrt 12 + 2) / 8, 3 * (Real.sqrt 12 - 2) / 8, 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3003_300336


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l3003_300365

theorem same_solution_implies_a_equals_seven (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 3 - (a - x) / 3 = 1) →
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l3003_300365


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3003_300344

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 8 * X^3 - 27 * X^2 - 32 * X + 52 = 
  (X^2 + 5 * X + 2) * q + (52 * X + 80) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3003_300344


namespace NUMINAMATH_CALUDE_equation_root_implies_z_value_l3003_300361

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) (a : ℝ) : Prop :=
  x^2 + (4 + i) * x + (4 : ℂ) + a * i = 0

-- Define the complex number z
def z (a b : ℝ) : ℂ := a + b * i

-- Theorem statement
theorem equation_root_implies_z_value (a b : ℝ) :
  equation b a → z a b = 2 - 2 * i :=
by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_z_value_l3003_300361


namespace NUMINAMATH_CALUDE_last_digit_of_322_power_111569_last_digit_is_two_l3003_300377

theorem last_digit_of_322_power_111569 : ℕ → Prop :=
  fun n => (322^111569 : ℕ) % 10 = n

theorem last_digit_is_two : last_digit_of_322_power_111569 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_322_power_111569_last_digit_is_two_l3003_300377


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3003_300327

theorem arithmetic_square_root_of_sqrt_16 :
  Real.sqrt (Real.sqrt 16) = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3003_300327


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l3003_300343

-- Define the slopes and y-intercept of the lines
def m_s : ℚ := 4/3
def b_s : ℚ := -100
def m_t : ℚ := -3/4

-- Define the lines
def line_s (x : ℚ) : ℚ := m_s * x + b_s
def line_t (x : ℚ) : ℚ := m_t * x

-- Define the intersection point
def intersection_x : ℚ := 48
def intersection_y : ℚ := -36

theorem perpendicular_lines_and_intersection :
  -- Line t is perpendicular to line s
  m_s * m_t = -1 ∧
  -- Line t passes through (0, 0)
  line_t 0 = 0 ∧
  -- The intersection point satisfies both line equations
  line_s intersection_x = intersection_y ∧
  line_t intersection_x = intersection_y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l3003_300343


namespace NUMINAMATH_CALUDE_leo_score_in_blackjack_l3003_300353

/-- In a blackjack game, given the scores of Caroline and Anthony, 
    and the fact that Leo is the winner with the winning score, 
    prove that Leo's score is 21. -/
theorem leo_score_in_blackjack 
  (caroline_score : ℕ) 
  (anthony_score : ℕ) 
  (winning_score : ℕ) 
  (leo_is_winner : Bool) : ℕ :=
by
  -- Define the given conditions
  have h1 : caroline_score = 13 := by sorry
  have h2 : anthony_score = 19 := by sorry
  have h3 : winning_score = 21 := by sorry
  have h4 : leo_is_winner = true := by sorry

  -- Prove that Leo's score is equal to the winning score
  sorry

#check leo_score_in_blackjack

end NUMINAMATH_CALUDE_leo_score_in_blackjack_l3003_300353


namespace NUMINAMATH_CALUDE_symmetric_trig_function_property_l3003_300315

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where a and b are real numbers,
    ab ≠ 0, and f is symmetric about x = π/6, prove that a = √3 * b. -/
theorem symmetric_trig_function_property (a b : ℝ) (h1 : a * b ≠ 0) :
  (∀ x, a * Real.sin (2 * x) + b * Real.cos (2 * x) = 
        a * Real.sin (2 * (Real.pi / 6 - x)) + b * Real.cos (2 * (Real.pi / 6 - x))) →
  a = Real.sqrt 3 * b := by
  sorry

end NUMINAMATH_CALUDE_symmetric_trig_function_property_l3003_300315


namespace NUMINAMATH_CALUDE_second_train_length_l3003_300354

/-- Calculates the length of the second train given the parameters of two trains approaching each other -/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (clear_time : ℝ) 
  (h1 : length_train1 = 100) 
  (h2 : speed_train1 = 42) 
  (h3 : speed_train2 = 30) 
  (h4 : clear_time = 12.998960083193344) :
  ∃ length_train2 : ℝ, abs (length_train2 - 159.98) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_l3003_300354


namespace NUMINAMATH_CALUDE_cost_per_credit_l3003_300392

/-- Calculates the cost per credit given college expenses -/
theorem cost_per_credit
  (total_credits : ℕ)
  (cost_per_textbook : ℕ)
  (num_textbooks : ℕ)
  (facilities_fee : ℕ)
  (total_expenses : ℕ)
  (h1 : total_credits = 14)
  (h2 : cost_per_textbook = 120)
  (h3 : num_textbooks = 5)
  (h4 : facilities_fee = 200)
  (h5 : total_expenses = 7100) :
  (total_expenses - (cost_per_textbook * num_textbooks + facilities_fee)) / total_credits = 450 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_credit_l3003_300392


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3003_300310

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 1) :
  2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3003_300310


namespace NUMINAMATH_CALUDE_men_entered_room_l3003_300329

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered women_left final_men : ℕ) :
  initial_men / initial_women = 4 / 5 →
  women_left = 3 →
  2 * (initial_women - women_left) = final_men →
  final_men = 14 →
  initial_men + men_entered = final_men →
  men_entered = 6 := by
sorry

end NUMINAMATH_CALUDE_men_entered_room_l3003_300329


namespace NUMINAMATH_CALUDE_room_height_proof_l3003_300304

theorem room_height_proof (length breadth diagonal : ℝ) (h : ℝ) :
  length = 12 →
  breadth = 8 →
  diagonal = 17 →
  diagonal^2 = length^2 + breadth^2 + h^2 →
  h = 9 := by
sorry

end NUMINAMATH_CALUDE_room_height_proof_l3003_300304


namespace NUMINAMATH_CALUDE_sams_books_l3003_300335

theorem sams_books (tim_books sam_books total_books : ℕ) : 
  tim_books = 44 → 
  total_books = 96 → 
  total_books = tim_books + sam_books → 
  sam_books = 52 := by
sorry

end NUMINAMATH_CALUDE_sams_books_l3003_300335


namespace NUMINAMATH_CALUDE_bugs_and_flowers_l3003_300300

/-- Given that 2.0 bugs ate 3.0 flowers in total, prove that each bug ate 1.5 flowers. -/
theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_bugs_and_flowers_l3003_300300


namespace NUMINAMATH_CALUDE_travel_allowance_percentage_l3003_300394

theorem travel_allowance_percentage
  (total_employees : ℕ)
  (salary_increase_percentage : ℚ)
  (no_increase : ℕ)
  (h1 : total_employees = 480)
  (h2 : salary_increase_percentage = 1/10)
  (h3 : no_increase = 336) :
  (total_employees - (salary_increase_percentage * total_employees + no_increase : ℚ)) / total_employees = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_travel_allowance_percentage_l3003_300394


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l3003_300306

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the theorem
theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ →
  f (-1) = y₂ →
  f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l3003_300306


namespace NUMINAMATH_CALUDE_solution_product_l3003_300380

theorem solution_product (r s : ℝ) : 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 63 →
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 63 →
  r ≠ s →
  (r + 4) * (s + 4) = -66 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l3003_300380


namespace NUMINAMATH_CALUDE_max_shadow_area_l3003_300331

/-- Regular tetrahedron with edge length a -/
structure Tetrahedron where
  a : ℝ
  a_pos : a > 0

/-- Cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- The maximum shadow area of a regular tetrahedron and a cube -/
theorem max_shadow_area 
  (t : Tetrahedron) 
  (c : Cube) 
  (light_zenith : True) -- Light source is directly above
  : 
  (∃ (shadow_area_tetra : ℝ), 
    shadow_area_tetra ≤ t.a^2 / 2 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_tetra) ∧
  (∃ (shadow_area_cube : ℝ), 
    shadow_area_cube ≤ c.a^2 * Real.sqrt 3 / 3 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_cube) :=
sorry

end NUMINAMATH_CALUDE_max_shadow_area_l3003_300331


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_l3003_300349

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_l3003_300349


namespace NUMINAMATH_CALUDE_initial_kibble_amount_l3003_300317

/-- The amount of kibble Luna is supposed to eat daily -/
def daily_kibble : ℕ := 2

/-- The amount of kibble Mary gave Luna in the morning -/
def mary_morning : ℕ := 1

/-- The amount of kibble Mary gave Luna in the evening -/
def mary_evening : ℕ := 1

/-- The amount of kibble Frank gave Luna in the afternoon -/
def frank_afternoon : ℕ := 1

/-- The amount of kibble remaining in the bag the next morning -/
def remaining_kibble : ℕ := 7

/-- The theorem stating the initial amount of kibble in the bag -/
theorem initial_kibble_amount : 
  mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon + remaining_kibble = 12 := by
  sorry

#check initial_kibble_amount

end NUMINAMATH_CALUDE_initial_kibble_amount_l3003_300317


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3003_300385

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) : 
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 1 / 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3003_300385


namespace NUMINAMATH_CALUDE_negative_product_inequality_l3003_300388

theorem negative_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_inequality_l3003_300388


namespace NUMINAMATH_CALUDE_max_value_x2_plus_y2_l3003_300399

theorem max_value_x2_plus_y2 (x y : ℝ) :
  5 * x^2 - 10 * x + 4 * y^2 = 0 →
  x^2 + y^2 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x2_plus_y2_l3003_300399


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l3003_300372

theorem height_on_hypotenuse (a b h : ℝ) : 
  a = 3 → b = 6 → a^2 + b^2 = (a*b/h)^2 → h = (6 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l3003_300372


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3003_300318

theorem inequality_equivalence (x : ℝ) : (x - 8) / (x^2 - 4*x + 13) ≥ 0 ↔ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3003_300318


namespace NUMINAMATH_CALUDE_cab_speed_fraction_l3003_300311

theorem cab_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 30 →
  delay = 6 →
  (usual_time / (usual_time + delay)) = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_cab_speed_fraction_l3003_300311


namespace NUMINAMATH_CALUDE_total_votes_l3003_300330

theorem total_votes (votes_for votes_against total_votes : ℕ) : 
  votes_for = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 290 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_l3003_300330


namespace NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_l3003_300322

/-- Given a point P₀ and a normal vector u⃗, prove that the equation
    ax + by + cz + d = 0 represents the plane passing through P₀ with normal vector u⃗. -/
theorem plane_equation_from_point_and_normal (P₀ : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ) 
  (a b c d : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a', b', c') := u
  (a = 2 ∧ b = -1 ∧ c = -3 ∧ d = 3) →
  (x₀ = 1 ∧ y₀ = 2 ∧ z₀ = 1) →
  (a' = -2 ∧ b' = 1 ∧ c' = 3) →
  ∀ (x y z : ℝ), a*x + b*y + c*z + d = 0 ↔ 
    a'*(x - x₀) + b'*(y - y₀) + c'*(z - z₀) = 0 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_l3003_300322


namespace NUMINAMATH_CALUDE_monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l3003_300350

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (1 - x) * Real.exp x + a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.exp (2 * x) + (1 - x) * Real.exp x - Real.exp x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f' a x * Real.exp (2 - x)

-- Statement 1
theorem monotonicity_of_g :
  let a := Real.exp (-2) / 2
  ∀ x y, x < 2 → y > 2 → g a x > g a 2 ∧ g a 2 < g a y := by sorry

-- Statement 2
theorem range_of_a_for_two_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) := by sorry

-- Statement 3
theorem inequality_for_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  x₁ + 2 * x₂ > 3 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l3003_300350


namespace NUMINAMATH_CALUDE_percentage_equivalence_l3003_300352

theorem percentage_equivalence : ∀ x : ℚ,
  (60 / 100) * 600 = (x / 100) * 720 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l3003_300352


namespace NUMINAMATH_CALUDE_constant_k_value_l3003_300341

theorem constant_k_value (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
sorry

end NUMINAMATH_CALUDE_constant_k_value_l3003_300341


namespace NUMINAMATH_CALUDE_fourth_number_in_second_set_l3003_300332

theorem fourth_number_in_second_set (x y : ℝ) : 
  ((28 + x + 42 + 78 + 104) / 5 = 90) →
  ((128 + 255 + 511 + y + x) / 5 = 423) →
  y = 1023 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_second_set_l3003_300332


namespace NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l3003_300366

/-- A machine that fills boxes at a constant rate -/
structure BoxFillingMachine where
  boxes_per_hour : ℚ

/-- Given a machine that fills 24 boxes in 60 minutes, prove it fills 2 boxes in 5 minutes -/
theorem boxes_filled_in_five_minutes 
  (machine : BoxFillingMachine) 
  (h : machine.boxes_per_hour = 24 / 1) : 
  (machine.boxes_per_hour * 5 / 60 : ℚ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l3003_300366


namespace NUMINAMATH_CALUDE_water_bottles_total_l3003_300333

/-- Represents the number of water bottles filled for each team --/
structure TeamBottles where
  football : ℕ
  soccer : ℕ
  lacrosse : ℕ
  rugby : ℕ

/-- Calculate the total number of water bottles filled for all teams --/
def total_bottles (t : TeamBottles) : ℕ :=
  t.football + t.soccer + t.lacrosse + t.rugby

/-- Theorem stating the total number of water bottles filled for the teams --/
theorem water_bottles_total :
  ∃ (t : TeamBottles),
    t.football = 11 * 6 ∧
    t.soccer = 53 ∧
    t.lacrosse = t.football + 12 ∧
    t.rugby = 49 ∧
    total_bottles t = 246 :=
by
  sorry


end NUMINAMATH_CALUDE_water_bottles_total_l3003_300333


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l3003_300309

theorem book_arrangement_proof :
  let total_books : ℕ := 8
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 3
  Nat.choose total_books geometry_books = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l3003_300309


namespace NUMINAMATH_CALUDE_adult_price_calculation_l3003_300382

/-- The daily price for adults at a public swimming pool -/
def adult_price (total_people : ℕ) (child_price : ℚ) (total_receipts : ℚ) (num_children : ℕ) : ℚ :=
  let num_adults : ℕ := total_people - num_children
  (total_receipts - (num_children : ℚ) * child_price) / (num_adults : ℚ)

/-- Theorem stating the adult price calculation for the given scenario -/
theorem adult_price_calculation :
  adult_price 754 (3/2) 1422 388 = 840 / 366 := by
  sorry

end NUMINAMATH_CALUDE_adult_price_calculation_l3003_300382


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3003_300340

theorem exponent_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3003_300340


namespace NUMINAMATH_CALUDE_abs_eq_self_implies_nonnegative_l3003_300301

theorem abs_eq_self_implies_nonnegative (a : ℝ) : |a| = a → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_implies_nonnegative_l3003_300301


namespace NUMINAMATH_CALUDE_coin_toss_total_l3003_300375

theorem coin_toss_total (head_count tail_count : ℕ) :
  let total_tosses := head_count + tail_count
  total_tosses = head_count + tail_count := by
  sorry

#check coin_toss_total 3 7

end NUMINAMATH_CALUDE_coin_toss_total_l3003_300375


namespace NUMINAMATH_CALUDE_min_value_theorem_l3003_300379

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_line : 3*m + n = 1) : 
  (∃ (x : ℝ), ∀ (m n : ℝ), m > 0 → n > 0 → 3*m + n = 1 → 3/m + 1/n ≥ x) ∧ 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ 3*m + n = 1 ∧ 3/m + 1/n = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3003_300379


namespace NUMINAMATH_CALUDE_units_digit_of_power_l3003_300346

theorem units_digit_of_power (n : ℕ) : n % 10 = 7 → (n^1997 % 10)^2999 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l3003_300346


namespace NUMINAMATH_CALUDE_house_number_problem_l3003_300359

theorem house_number_problem (numbers : List Nat) 
  (h_numbers : numbers = [1, 3, 4, 6, 8, 9, 11, 12, 16]) 
  (h_total : numbers.sum = 70) 
  (vova_sum dima_sum : Nat) 
  (h_vova_dima : vova_sum = 3 * dima_sum) 
  (h_sum_relation : vova_sum + dima_sum + house_number = 70) 
  (h_house_mod : house_number % 4 = 2) : 
  house_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_house_number_problem_l3003_300359


namespace NUMINAMATH_CALUDE_total_salmon_count_l3003_300369

/-- Represents the count of male and female salmon for a species -/
structure SalmonCount where
  males : Nat
  females : Nat

/-- Calculates the total number of salmon for a given species -/
def totalForSpecies (count : SalmonCount) : Nat :=
  count.males + count.females

/-- The counts for each salmon species -/
def chinookCount : SalmonCount := { males := 451228, females := 164225 }
def sockeyeCount : SalmonCount := { males := 212001, females := 76914 }
def cohoCount : SalmonCount := { males := 301008, females := 111873 }
def pinkCount : SalmonCount := { males := 518001, females := 182945 }
def chumCount : SalmonCount := { males := 230023, females := 81321 }

/-- Theorem stating that the total number of salmon across all species is 2,329,539 -/
theorem total_salmon_count : 
  totalForSpecies chinookCount + 
  totalForSpecies sockeyeCount + 
  totalForSpecies cohoCount + 
  totalForSpecies pinkCount + 
  totalForSpecies chumCount = 2329539 := by
  sorry

end NUMINAMATH_CALUDE_total_salmon_count_l3003_300369


namespace NUMINAMATH_CALUDE_divisors_of_expression_l3003_300321

theorem divisors_of_expression (n : ℕ+) : 
  ∃ (d : Finset ℕ+), 
    (∀ k : ℕ+, k ∈ d ↔ ∀ m : ℕ+, k ∣ (m * (m^2 - 1) * (m^2 + 3) * (m^2 + 5))) ∧
    d.card = 16 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_expression_l3003_300321


namespace NUMINAMATH_CALUDE_total_earrings_l3003_300355

/-- Proves that the total number of earrings for Bella, Monica, and Rachel is 70 -/
theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ)
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) :
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

#check total_earrings

end NUMINAMATH_CALUDE_total_earrings_l3003_300355


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3003_300314

/-- Given two lines that are parallel, prove that the value of 'a' is 1/2 -/
theorem parallel_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (2 * a * y - 1 = 0 ↔ y = 1 / (2 * a))) →
  (∀ x y : ℝ, ((3 * a - 1) * x + y - 1 = 0 ↔ y = -(3 * a - 1) * x + 1)) →
  (∀ x y : ℝ, 2 * a * y - 1 = 0 → (3 * a - 1) * x + y - 1 = 0 → x = 0) →
  a = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_a_value_l3003_300314


namespace NUMINAMATH_CALUDE_magic_box_result_l3003_300308

def magic_box (a b : ℝ) : ℝ := a^2 + b + 1

theorem magic_box_result : 
  let m := magic_box (-2) 3
  magic_box m 1 = 66 := by sorry

end NUMINAMATH_CALUDE_magic_box_result_l3003_300308


namespace NUMINAMATH_CALUDE_composition_of_even_function_is_even_l3003_300348

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem composition_of_even_function_is_even (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (fun x ↦ g (g (g x))) := by sorry

end NUMINAMATH_CALUDE_composition_of_even_function_is_even_l3003_300348


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3003_300389

theorem nested_fraction_evaluation :
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 ∧ 8 / 21 ≠ 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3003_300389


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l3003_300303

/-- The probability of getting exactly three tails and one head when four fair coins are tossed simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when four fair coins are tossed simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l3003_300303


namespace NUMINAMATH_CALUDE_lecture_arrangements_l3003_300339

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid arrangements for k lecturers with ordering constraints --/
def valid_arrangements (k : ℕ) : ℕ :=
  (k - 1) * k / 2

/-- Calculates the number of ways to arrange the remaining lecturers --/
def remaining_arrangements (n k : ℕ) : ℕ :=
  Nat.factorial (n - k)

/-- Theorem stating the total number of possible lecture arrangements --/
theorem lecture_arrangements :
  valid_arrangements k * remaining_arrangements n k = 240 :=
sorry

end NUMINAMATH_CALUDE_lecture_arrangements_l3003_300339


namespace NUMINAMATH_CALUDE_factorization_2x_minus_x_squared_l3003_300390

theorem factorization_2x_minus_x_squared (x : ℝ) : 2*x - x^2 = x*(2-x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_minus_x_squared_l3003_300390


namespace NUMINAMATH_CALUDE_cookies_difference_l3003_300386

def initial_cookies : ℕ := 41
def cookies_given : ℕ := 9
def cookies_eaten : ℕ := 18

theorem cookies_difference : cookies_eaten - cookies_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l3003_300386


namespace NUMINAMATH_CALUDE_number_of_boys_l3003_300387

theorem number_of_boys (num_vans : ℕ) (students_per_van : ℕ) (num_girls : ℕ) : 
  num_vans = 5 → students_per_van = 28 → num_girls = 80 → 
  num_vans * students_per_van - num_girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3003_300387


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l3003_300334

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon ABCDE is convex (sum of angles is 540°)
  a + b + c + d + e = 540 →
  -- Angle D is 30° more than angle A
  d = a + 30 →
  -- Angle E is 50° more than angle A
  e = a + 50 →
  -- Angles B and C are equal
  b = c →
  -- Angle A is 45° less than angle B
  a + 45 = b →
  -- Conclusion: Angle D measures 104°
  d = 104 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l3003_300334


namespace NUMINAMATH_CALUDE_pizza_eaten_fraction_l3003_300396

theorem pizza_eaten_fraction (n : Nat) : 
  let r : ℚ := 1/3
  let sum : ℚ := (1 - r^n) / (1 - r)
  n = 6 → sum = 364/729 := by
sorry

end NUMINAMATH_CALUDE_pizza_eaten_fraction_l3003_300396


namespace NUMINAMATH_CALUDE_circle_center_l3003_300316

/-- The center of the circle with equation x^2 + y^2 + 4x - 6y + 9 = 0 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 6*y + 9 = 0) ↔ ((x + 2)^2 + (y - 3)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_l3003_300316


namespace NUMINAMATH_CALUDE_inverse_composition_value_l3003_300337

open Function

-- Define the functions h and k
variable (h k : ℝ → ℝ)

-- Define the condition given in the problem
axiom h_k_relation : ∀ x, h⁻¹ (k x) = 3 * x - 4

-- State the theorem to be proved
theorem inverse_composition_value : k⁻¹ (h 5) = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_value_l3003_300337


namespace NUMINAMATH_CALUDE_team_selection_ways_l3003_300345

def number_of_combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem team_selection_ways (total_boys total_girls team_boys team_girls : ℕ) 
  (h1 : total_boys = 7)
  (h2 : total_girls = 9)
  (h3 : team_boys = 3)
  (h4 : team_girls = 3) :
  (number_of_combinations total_boys team_boys) * (number_of_combinations total_girls team_girls) = 2940 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l3003_300345
