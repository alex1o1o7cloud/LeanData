import Mathlib

namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_value_l2181_218121

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

-- State the theorem
theorem f_monotonicity_and_max_value :
  -- Part 1: Monotonicity
  (∀ x y : ℝ, x < y ∧ y < 3/4 → f x > f y) ∧
  (∀ x y : ℝ, 3/4 < x ∧ x < y → f x < f y) ∧
  -- Part 2: Maximum value on [2, 4]
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → f x ≤ 42) ∧
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = 42) :=
by sorry


end NUMINAMATH_CALUDE_f_monotonicity_and_max_value_l2181_218121


namespace NUMINAMATH_CALUDE_vector_operation_l2181_218139

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  2 • a - b = (7, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2181_218139


namespace NUMINAMATH_CALUDE_distinct_colorings_l2181_218190

/-- Represents the symmetry group of a regular decagon -/
def DecagonSymmetryGroup : Type := Unit

/-- The order of the decagon symmetry group -/
def decagon_symmetry_order : ℕ := 10

/-- The number of disks in the decagon -/
def total_disks : ℕ := 10

/-- The number of disks to be colored -/
def colored_disks : ℕ := 8

/-- The number of blue disks -/
def blue_disks : ℕ := 4

/-- The number of red disks -/
def red_disks : ℕ := 3

/-- The number of green disks -/
def green_disks : ℕ := 2

/-- The number of yellow disks -/
def yellow_disks : ℕ := 1

/-- The total number of colorings without considering symmetry -/
def total_colorings : ℕ := (total_disks.choose blue_disks) * 
                           ((total_disks - blue_disks).choose red_disks) * 
                           ((total_disks - blue_disks - red_disks).choose green_disks) * 
                           ((total_disks - blue_disks - red_disks - green_disks).choose yellow_disks)

/-- The number of distinct colorings considering symmetry -/
theorem distinct_colorings : 
  (total_colorings / decagon_symmetry_order : ℚ) = 1260 := by sorry

end NUMINAMATH_CALUDE_distinct_colorings_l2181_218190


namespace NUMINAMATH_CALUDE_apple_distribution_l2181_218182

theorem apple_distribution (total_apples : ℕ) (num_babies : ℕ) (min_apples : ℕ) (max_apples : ℕ) :
  total_apples = 30 →
  num_babies = 7 →
  min_apples = 3 →
  max_apples = 6 →
  ∃ (removed : ℕ), 
    (total_apples - removed) % num_babies = 0 ∧
    (total_apples - removed) / num_babies ≥ min_apples ∧
    (total_apples - removed) / num_babies ≤ max_apples ∧
    removed = 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2181_218182


namespace NUMINAMATH_CALUDE_horner_v1_for_f_at_neg_two_l2181_218169

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : List ℝ := [2, 0.3, 1, 0, 6, -5, 1]

/-- Theorem: v1 in Horner's method for f(x) at x = -2 is -7 -/
theorem horner_v1_for_f_at_neg_two :
  (horner (f.tail) (-2) : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_horner_v1_for_f_at_neg_two_l2181_218169


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2181_218173

theorem quadratic_roots_relation (k n p : ℝ) (hk : k ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -p ∧ s₁ * s₂ = k) ∧
               (3*s₁ + 3*s₂ = -k ∧ 9*s₁*s₂ = n)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2181_218173


namespace NUMINAMATH_CALUDE_seven_digit_palindromes_count_l2181_218187

/-- Represents a multiset of digits --/
def DigitMultiset := Multiset Nat

/-- Checks if a number is a palindrome --/
def isPalindrome (n : Nat) : Bool := sorry

/-- Counts the number of 7-digit palindromes that can be formed from a given multiset of digits --/
def countSevenDigitPalindromes (digits : DigitMultiset) : Nat := sorry

/-- The specific multiset of digits given in the problem --/
def givenDigits : DigitMultiset := sorry

theorem seven_digit_palindromes_count :
  countSevenDigitPalindromes givenDigits = 18 := by sorry

end NUMINAMATH_CALUDE_seven_digit_palindromes_count_l2181_218187


namespace NUMINAMATH_CALUDE_circle_polar_equation_l2181_218125

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  a : ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a given polar equation represents the specified circle -/
def is_correct_polar_equation (circle : PolarCircle) (equation : ℝ → ℝ → Prop) : Prop :=
  circle.center = (circle.a / 2, Real.pi / 2) ∧
  circle.radius = circle.a / 2 ∧
  ∀ θ ρ, equation ρ θ ↔ ρ = circle.a * Real.sin θ

theorem circle_polar_equation (a : ℝ) (h : a > 0) :
  let circle : PolarCircle := ⟨a, (a / 2, Real.pi / 2), a / 2⟩
  is_correct_polar_equation circle (fun ρ θ ↦ ρ = a * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l2181_218125


namespace NUMINAMATH_CALUDE_fire_water_requirement_l2181_218106

theorem fire_water_requirement 
  (flow_rate : ℝ) 
  (num_firefighters : ℕ) 
  (time_taken : ℝ) 
  (h1 : flow_rate = 20) 
  (h2 : num_firefighters = 5) 
  (h3 : time_taken = 40) : 
  flow_rate * num_firefighters * time_taken = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_fire_water_requirement_l2181_218106


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l2181_218136

theorem rectangular_plot_ratio (length breadth area : ℝ) : 
  breadth = 14 →
  area = 588 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l2181_218136


namespace NUMINAMATH_CALUDE_wreath_distribution_l2181_218102

/-- The number of wreaths each Greek initially had -/
def wreaths_per_greek (m : ℕ) : ℕ := 4 * m

/-- The number of Greeks -/
def num_greeks : ℕ := 3

/-- The number of Muses -/
def num_muses : ℕ := 9

/-- The total number of people (Greeks and Muses) -/
def total_people : ℕ := num_greeks + num_muses

theorem wreath_distribution (m : ℕ) (h : m > 0) :
  ∃ (initial_wreaths : ℕ),
    initial_wreaths = wreaths_per_greek m ∧
    (initial_wreaths * num_greeks) % total_people = 0 ∧
    ∀ (final_wreaths : ℕ),
      final_wreaths * total_people = initial_wreaths * num_greeks →
      final_wreaths = m :=
by sorry

end NUMINAMATH_CALUDE_wreath_distribution_l2181_218102


namespace NUMINAMATH_CALUDE_exists_specific_polyhedron_l2181_218160

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A polyhedron -/
structure Polyhedron where
  faces : List Face

/-- Counts the number of faces with a given number of sides -/
def countFaces (p : Polyhedron) (n : ℕ) : ℕ :=
  p.faces.filter (λ f => f.sides = n) |>.length

/-- Theorem: There exists a polyhedron with exactly 6 faces,
    where 2 faces are triangles, 2 faces are quadrilaterals, and 2 faces are pentagons -/
theorem exists_specific_polyhedron :
  ∃ p : Polyhedron,
    p.faces.length = 6 ∧
    countFaces p 3 = 2 ∧
    countFaces p 4 = 2 ∧
    countFaces p 5 = 2 :=
  sorry

end NUMINAMATH_CALUDE_exists_specific_polyhedron_l2181_218160


namespace NUMINAMATH_CALUDE_q_investment_proof_l2181_218110

/-- Calculates the investment of Q given the investment of P, total profit, and Q's profit share --/
def calculate_q_investment (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) : ℚ :=
  (p_investment * q_profit_share) / (total_profit - q_profit_share)

theorem q_investment_proof (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) 
  (h1 : p_investment = 54000)
  (h2 : total_profit = 18000)
  (h3 : q_profit_share = 6001.89) :
  calculate_q_investment p_investment total_profit q_profit_share = 27010 := by
  sorry

#eval calculate_q_investment 54000 18000 6001.89

end NUMINAMATH_CALUDE_q_investment_proof_l2181_218110


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l2181_218149

/-- The number of players on the Spurs basketball team -/
def num_players : ℕ := 22

/-- The number of basketballs each player has -/
def balls_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := num_players * balls_per_player

theorem spurs_basketball_count : total_basketballs = 242 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l2181_218149


namespace NUMINAMATH_CALUDE_equation_solution_l2181_218171

theorem equation_solution : 
  ∀ x : ℝ, (x - 1)^2 = 64 ↔ x = 9 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2181_218171


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2181_218148

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 5/4 * (12/5) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l2181_218148


namespace NUMINAMATH_CALUDE_high_octane_half_cost_l2181_218180

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane_units : ℕ
  regular_octane_units : ℕ
  high_octane_cost_multiplier : ℕ

/-- Calculates the fraction of the total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  let high_octane_cost := fuel.high_octane_units * fuel.high_octane_cost_multiplier
  let regular_octane_cost := fuel.regular_octane_units
  let total_cost := high_octane_cost + regular_octane_cost
  high_octane_cost / total_cost

/-- Theorem: The fraction of the cost due to high octane is 1/2 for the given fuel mixture -/
theorem high_octane_half_cost (fuel : FuelMixture) 
    (h1 : fuel.high_octane_units = 1515)
    (h2 : fuel.regular_octane_units = 4545)
    (h3 : fuel.high_octane_cost_multiplier = 3) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_high_octane_half_cost_l2181_218180


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2181_218154

/-- A quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given x -/
def QuadraticPolynomial.evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial satisfies the given condition if
    f(a) = a, f(b) = b, and f(c) = c -/
def satisfies_condition (p : QuadraticPolynomial) : Prop :=
  p.evaluate p.a = p.a ∧
  p.evaluate p.b = p.b ∧
  p.evaluate p.c = p.c

/-- The theorem stating that only x^2 + x - 1 and x - 2 satisfy the condition -/
theorem quadratic_polynomial_condition :
  ∀ p : QuadraticPolynomial,
    satisfies_condition p →
      (p = ⟨1, 1, -1⟩ ∨ p = ⟨0, 1, -2⟩) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2181_218154


namespace NUMINAMATH_CALUDE_circle_equation_l2181_218158

/-- A circle C with center (0, a) -/
structure Circle (a : ℝ) where
  center : ℝ × ℝ := (0, a)

/-- The equation of a circle with center (0, a) and radius r -/
def circleEquation (c : Circle a) (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = r^2

/-- The circle passes through the point (1, 0) -/
def passesThrough (c : Circle a) (r : ℝ) : Prop :=
  circleEquation c r 1 0

/-- The circle is divided by the x-axis into two arcs with length ratio 1:2 -/
def arcRatio (c : Circle a) : Prop :=
  abs (a / 1) = Real.sqrt 3

theorem circle_equation (a : ℝ) (c : Circle a) (h1 : passesThrough c (Real.sqrt (4/3)))
    (h2 : arcRatio c) :
    ∀ x y : ℝ, circleEquation c (Real.sqrt (4/3)) x y ↔ 
      x^2 + (y - Real.sqrt 3 / 3)^2 = 4/3 ∨ x^2 + (y + Real.sqrt 3 / 3)^2 = 4/3 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2181_218158


namespace NUMINAMATH_CALUDE_percentage_problem_l2181_218103

theorem percentage_problem : (45 * 7) / 900 * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2181_218103


namespace NUMINAMATH_CALUDE_octagon_perimeter_l2181_218159

/-- The perimeter of an octagon with alternating side lengths -/
theorem octagon_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 2) :
  4 * a + 4 * b = 12 + 8 * Real.sqrt 2 := by
  sorry

#check octagon_perimeter

end NUMINAMATH_CALUDE_octagon_perimeter_l2181_218159


namespace NUMINAMATH_CALUDE_dividend_calculation_l2181_218129

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 35.8)
  (h2 : quotient = 21.65)
  (h3 : remainder = 11.3) :
  divisor * quotient + remainder = 786.47 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2181_218129


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2181_218196

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ a = 25 * r ∧ 7/9 = a * r) : 
  a = 5 * Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2181_218196


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l2181_218172

/-- Definition of scientific notation -/
def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

/-- The number to be expressed in scientific notation -/
def number : ℝ := 123000

/-- Theorem stating that 123000 can be expressed as 1.23 × 10^5 in scientific notation -/
theorem number_in_scientific_notation :
  scientific_notation number 1.23 5 :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l2181_218172


namespace NUMINAMATH_CALUDE_height_lateral_edge_ratio_is_correct_l2181_218168

/-- Regular quadrilateral pyramid with vertex P and square base ABCD -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : RegularQuadPyramid

/-- The ratio of height to lateral edge in a regular quadrilateral pyramid
    where the intersecting plane creates a cross-section with half the area of the base -/
def height_to_lateral_edge_ratio (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

/-- Theorem stating the ratio of height to lateral edge -/
theorem height_lateral_edge_ratio_is_correct (p : RegularQuadPyramid) (plane : IntersectingPlane) :
  height_to_lateral_edge_ratio p plane = (1 + Real.sqrt 33) / 8 :=
sorry

end NUMINAMATH_CALUDE_height_lateral_edge_ratio_is_correct_l2181_218168


namespace NUMINAMATH_CALUDE_bucket_radius_l2181_218197

/-- Proves that a cylindrical bucket with height 36 cm, when emptied to form a conical heap
    of height 12 cm and base radius 63 cm, has a radius of 21 cm. -/
theorem bucket_radius (h_cylinder h_cone r_cone : ℝ) 
    (h_cylinder_val : h_cylinder = 36)
    (h_cone_val : h_cone = 12)
    (r_cone_val : r_cone = 63)
    (volume_eq : π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :
  r_cylinder = 21 :=
sorry

end NUMINAMATH_CALUDE_bucket_radius_l2181_218197


namespace NUMINAMATH_CALUDE_simplest_fraction_C_l2181_218195

def is_simplest_fraction (num : ℚ → ℚ) (denom : ℚ → ℚ) : Prop :=
  ∀ a : ℚ, ∀ k : ℚ, k ≠ 0 → num a / denom a = (k * num a) / (k * denom a) → k = 1 ∨ k = -1

theorem simplest_fraction_C :
  is_simplest_fraction (λ a => 2 * a) (λ a => 2 - a) :=
sorry

end NUMINAMATH_CALUDE_simplest_fraction_C_l2181_218195


namespace NUMINAMATH_CALUDE_degree_of_divisor_l2181_218167

/-- Given a polynomial f of degree 15 and another polynomial d, 
    if f divided by d results in a quotient of degree 7 and a remainder of degree 4, 
    then the degree of d is 8. -/
theorem degree_of_divisor (f d : Polynomial ℝ) (q : Polynomial ℝ) (r : Polynomial ℝ) :
  Polynomial.degree f = 15 →
  f = d * q + r →
  Polynomial.degree q = 7 →
  Polynomial.degree r = 4 →
  Polynomial.degree d = 8 := by
  sorry


end NUMINAMATH_CALUDE_degree_of_divisor_l2181_218167


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2181_218105

theorem fraction_evaluation (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2181_218105


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2181_218153

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (9 * x) - Real.cos (5 * x) - Real.sqrt 2 * Real.cos (4 * x) + Real.sin (9 * x) + Real.sin (5 * x) = 0) →
  (∃ k : ℤ, x = π / 8 + π * k / 2 ∨ x = π / 20 + 2 * π * k / 5 ∨ x = π / 12 + 2 * π * k / 9) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2181_218153


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2181_218166

theorem positive_real_inequalities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (|a^2 - b^2| = 1 → |a - b| < 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2181_218166


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l2181_218134

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l2181_218134


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2181_218116

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2181_218116


namespace NUMINAMATH_CALUDE_charles_watercolor_pictures_after_work_l2181_218131

/-- Represents the number of pictures drawn on a specific type of paper -/
structure PictureCount where
  regular : ℕ
  watercolor : ℕ

/-- Represents the initial paper count and pictures drawn on different occasions -/
structure DrawingData where
  initialRegular : ℕ
  initialWatercolor : ℕ
  todayPictures : PictureCount
  yesterdayBeforeWork : ℕ
  remainingRegular : ℕ

/-- Calculates the number of watercolor pictures drawn after work yesterday -/
def watercolorPicturesAfterWork (data : DrawingData) : ℕ :=
  data.initialWatercolor - data.todayPictures.watercolor -
  (data.yesterdayBeforeWork - (data.initialRegular - data.todayPictures.regular - data.remainingRegular))

/-- Theorem stating that Charles drew 6 watercolor pictures after work yesterday -/
theorem charles_watercolor_pictures_after_work :
  let data : DrawingData := {
    initialRegular := 10,
    initialWatercolor := 10,
    todayPictures := { regular := 4, watercolor := 2 },
    yesterdayBeforeWork := 6,
    remainingRegular := 2
  }
  watercolorPicturesAfterWork data = 6 := by sorry

end NUMINAMATH_CALUDE_charles_watercolor_pictures_after_work_l2181_218131


namespace NUMINAMATH_CALUDE_candy_mix_equations_correct_l2181_218138

/-- Represents the candy mixing problem -/
structure CandyMix where
  x : ℝ  -- quantity of 36 yuan/kg candy
  y : ℝ  -- quantity of 20 yuan/kg candy
  total_weight : ℝ  -- total weight of mixed candy
  mixed_price : ℝ  -- price of mixed candy per kg
  high_price : ℝ  -- price of more expensive candy per kg
  low_price : ℝ  -- price of less expensive candy per kg

/-- The system of equations correctly describes the candy mixing problem -/
theorem candy_mix_equations_correct (mix : CandyMix) 
  (h1 : mix.total_weight = 100)
  (h2 : mix.mixed_price = 28)
  (h3 : mix.high_price = 36)
  (h4 : mix.low_price = 20) :
  (mix.x + mix.y = mix.total_weight) ∧ 
  (mix.high_price * mix.x + mix.low_price * mix.y = mix.mixed_price * mix.total_weight) :=
sorry

end NUMINAMATH_CALUDE_candy_mix_equations_correct_l2181_218138


namespace NUMINAMATH_CALUDE_not_always_same_direction_for_parallel_vectors_l2181_218157

-- Define a vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define parallel vectors
def parallel (u v : V) : Prop :=
  ∃ k : ℝ, v = k • u

-- Theorem statement
theorem not_always_same_direction_for_parallel_vectors :
  ¬ ∀ (u v : V), parallel u v → (∃ k : ℝ, k > 0 ∧ v = k • u) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_direction_for_parallel_vectors_l2181_218157


namespace NUMINAMATH_CALUDE_total_coins_is_twelve_l2181_218188

def coins_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x / 2)

theorem total_coins_is_twelve :
  ∃ x : ℕ, 
    x > 0 ∧ 
    let (pete_coins, paul_coins) := coins_distribution x
    pete_coins = 5 * paul_coins ∧
    pete_coins + paul_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_twelve_l2181_218188


namespace NUMINAMATH_CALUDE_homothety_composition_l2181_218113

open Complex

def H_i_squared (i z : ℂ) : ℂ := 2 * (z - i) + i

def T (i z : ℂ) : ℂ := z + i

def H_0_squared (z : ℂ) : ℂ := 2 * z

theorem homothety_composition (i z : ℂ) : H_i_squared i z = (T i ∘ H_0_squared) (z - i) := by sorry

end NUMINAMATH_CALUDE_homothety_composition_l2181_218113


namespace NUMINAMATH_CALUDE_percentage_of_fraction_equals_value_l2181_218112

theorem percentage_of_fraction_equals_value : 
  let number : ℝ := 70.58823529411765
  let fraction : ℝ := 3 / 5
  let percentage : ℝ := 85 / 100
  percentage * (fraction * number) = 36 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_fraction_equals_value_l2181_218112


namespace NUMINAMATH_CALUDE_N_subset_M_l2181_218184

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l2181_218184


namespace NUMINAMATH_CALUDE_turquoise_more_green_count_l2181_218115

/-- Represents the survey results about the perception of turquoise color --/
structure TurquoiseSurvey where
  total : Nat
  more_blue : Nat
  both : Nat
  neither : Nat

/-- Calculates the number of people who believe turquoise is "more green" --/
def more_green (survey : TurquoiseSurvey) : Nat :=
  survey.total - (survey.more_blue - survey.both) - survey.neither

/-- Theorem stating that given the survey conditions, 80 people believe turquoise is "more green" --/
theorem turquoise_more_green_count :
  ∀ (survey : TurquoiseSurvey),
  survey.total = 150 →
  survey.more_blue = 90 →
  survey.both = 40 →
  survey.neither = 20 →
  more_green survey = 80 := by
  sorry


end NUMINAMATH_CALUDE_turquoise_more_green_count_l2181_218115


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_three_l2181_218143

theorem exists_irrational_less_than_three : ∃ x : ℝ, Irrational x ∧ |x| < 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_three_l2181_218143


namespace NUMINAMATH_CALUDE_angle_bisector_intersection_ratio_l2181_218174

/-- Given a triangle PQR with points M on PQ and N on PR such that
    PM:MQ = 2:6 and PN:NR = 3:9, if PS is the angle bisector of angle P
    intersecting MN at L, then PL:PS = 1:4 -/
theorem angle_bisector_intersection_ratio (P Q R M N S L : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 2 * t = 6 * (1 - t)) →
  (∃ u : ℝ, N = (1 - u) • P + u • R ∧ 3 * u = 9 * (1 - u)) →
  (∃ v : ℝ, S = (1 - v) • P + v • Q ∧ 
            ∃ w : ℝ, S = (1 - w) • P + w • R ∧
            v / (1 - v) = w / (1 - w)) →
  (∃ k : ℝ, L = (1 - k) • M + k • N) →
  (∃ r : ℝ, L = (1 - r) • P + r • S ∧ r = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_intersection_ratio_l2181_218174


namespace NUMINAMATH_CALUDE_central_sum_theorem_l2181_218191

/-- Represents a 4x4 matrix of integers -/
def Matrix4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the matrix are adjacent -/
def isAdjacent (a b : Fin 4 × Fin 4) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- Checks if the matrix contains all numbers from 1 to 16 -/
def containsAllNumbers (m : Matrix4x4) : Prop :=
  ∀ n : Fin 16, ∃ i j : Fin 4, m i j = n.val + 1

/-- Checks if consecutive numbers are adjacent in the matrix -/
def consecutiveAdjacent (m : Matrix4x4) : Prop :=
  ∀ n : Fin 15, ∃ i₁ j₁ i₂ j₂ : Fin 4,
    m i₁ j₁ = n.val + 1 ∧ m i₂ j₂ = n.val + 2 ∧ isAdjacent (i₁, j₁) (i₂, j₂)

/-- Calculates the sum of corner numbers in the matrix -/
def cornerSum (m : Matrix4x4) : ℕ :=
  m 0 0 + m 0 3 + m 3 0 + m 3 3

/-- Calculates the sum of central numbers in the matrix -/
def centerSum (m : Matrix4x4) : ℕ :=
  m 1 1 + m 1 2 + m 2 1 + m 2 2

theorem central_sum_theorem (m : Matrix4x4)
  (h1 : containsAllNumbers m)
  (h2 : consecutiveAdjacent m)
  (h3 : cornerSum m = 34) :
  centerSum m = 34 := by
  sorry

end NUMINAMATH_CALUDE_central_sum_theorem_l2181_218191


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2181_218135

theorem fraction_equals_zero (x : ℝ) : (x - 1) / (3 * x + 1) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2181_218135


namespace NUMINAMATH_CALUDE_fraction_of_ivys_collectors_dolls_l2181_218111

/-- The number of dolls Dina has -/
def dinas_dolls : ℕ := 60

/-- The number of collectors edition dolls Ivy has -/
def ivys_collectors_dolls : ℕ := 20

/-- The number of dolls Ivy has -/
def ivys_dolls : ℕ := dinas_dolls / 2

theorem fraction_of_ivys_collectors_dolls : 
  (ivys_collectors_dolls : ℚ) / (ivys_dolls : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_of_ivys_collectors_dolls_l2181_218111


namespace NUMINAMATH_CALUDE_sheep_sheepdog_distance_l2181_218155

/-- The initial distance between a sheep and a sheepdog -/
def initial_distance (sheep_speed sheepdog_speed : ℝ) (catch_time : ℝ) : ℝ :=
  sheepdog_speed * catch_time - sheep_speed * catch_time

/-- Theorem stating the initial distance between the sheep and sheepdog -/
theorem sheep_sheepdog_distance :
  initial_distance 12 20 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_sheep_sheepdog_distance_l2181_218155


namespace NUMINAMATH_CALUDE_escalator_steps_l2181_218165

/-- The number of steps Xiaolong takes to go down the escalator -/
def steps_down : ℕ := 30

/-- The number of steps Xiaolong takes to go up the escalator -/
def steps_up : ℕ := 90

/-- The ratio of Xiaolong's speed going up compared to going down -/
def speed_ratio : ℕ := 3

/-- The total number of visible steps on the escalator -/
def total_steps : ℕ := 60

theorem escalator_steps :
  ∃ (x : ℚ),
    (steps_down : ℚ) + (steps_down : ℚ) * x = (steps_up : ℚ) - (steps_up : ℚ) / speed_ratio * x ∧
    x = 1 ∧
    total_steps = steps_down + steps_down := by sorry

end NUMINAMATH_CALUDE_escalator_steps_l2181_218165


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2181_218117

/-- Given a total sum and a second part, calculates the interest rate of the first part
    such that the interest on the first part for 8 years equals the interest on the second part for 3 years at 5% --/
def calculate_interest_rate (total_sum : ℚ) (second_part : ℚ) : ℚ :=
  let first_part := total_sum - second_part
  let second_part_interest := second_part * 5 * 3 / 100
  second_part_interest * 100 / (first_part * 8)

theorem interest_rate_problem (total_sum : ℚ) (second_part : ℚ) 
  (h1 : total_sum = 2769)
  (h2 : second_part = 1704) :
  calculate_interest_rate total_sum second_part = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2181_218117


namespace NUMINAMATH_CALUDE_strawberry_yield_per_row_l2181_218147

theorem strawberry_yield_per_row :
  let total_rows : ℕ := 7
  let total_yield : ℕ := 1876
  let yield_per_row : ℕ := total_yield / total_rows
  yield_per_row = 268 := by sorry

end NUMINAMATH_CALUDE_strawberry_yield_per_row_l2181_218147


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2181_218104

theorem unique_solution_to_equation :
  ∃! (x y z : ℝ), 2*x^4 + 2*y^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + 7*y^2 + 7*z^2 - 14*y*z - 70*y + 70*z + 175 = 0 ∧
                   x = 0 ∧ y = 0 ∧ z = -5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2181_218104


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2181_218124

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 0 + a 1 + a 2 = 3 * a 0) 
  (h_nonzero : a 0 ≠ 0) : 
  q = -2 ∨ q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2181_218124


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2181_218183

theorem complex_modulus_problem (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) - i) / (a + i)).im ≠ 0 ∧ (((1 : ℂ) - i) / (a + i)).re = 0 →
  Complex.abs ((2 * a + 1 : ℂ) + Complex.I * Real.sqrt 2) = Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2181_218183


namespace NUMINAMATH_CALUDE_car_speed_relationship_l2181_218156

/-- Represents the relationship between the speeds and travel times of two cars -/
theorem car_speed_relationship (x : ℝ) : x > 0 →
  (80 / x - 2 = 80 / (3 * x) + 2 / 3) ↔
  (80 / x = 80 / (3 * x) + 2 + 2 / 3 ∧
   80 = x * (80 / (3 * x) + 2 + 2 / 3) ∧
   80 = 3 * x * (80 / (3 * x) + 2 / 3)) := by
  sorry

#check car_speed_relationship

end NUMINAMATH_CALUDE_car_speed_relationship_l2181_218156


namespace NUMINAMATH_CALUDE_jaden_toy_cars_l2181_218133

theorem jaden_toy_cars (initial_cars birthday_cars sister_cars friend_cars final_cars : ℕ) :
  initial_cars = 14 →
  birthday_cars = 12 →
  sister_cars = 8 →
  friend_cars = 3 →
  final_cars = 43 →
  ∃ (bought_cars : ℕ), 
    initial_cars + birthday_cars + bought_cars - sister_cars - friend_cars = final_cars ∧
    bought_cars = 28 :=
by sorry

end NUMINAMATH_CALUDE_jaden_toy_cars_l2181_218133


namespace NUMINAMATH_CALUDE_point_on_line_l2181_218151

theorem point_on_line (m : ℝ) : (5 : ℝ) = 2 * m + 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2181_218151


namespace NUMINAMATH_CALUDE_orangeade_price_theorem_l2181_218140

/-- Represents the price of orangeade per glass -/
@[ext] structure OrangeadePrice where
  price : ℚ

/-- Represents the composition of orangeade -/
@[ext] structure OrangeadeComposition where
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade -/
def total_volume (c : OrangeadeComposition) : ℚ :=
  c.orange_juice + c.water

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : ℚ) : ℚ :=
  price.price * volume

theorem orangeade_price_theorem 
  (day1_comp : OrangeadeComposition)
  (day2_comp : OrangeadeComposition)
  (day2_price : OrangeadePrice)
  (h1 : day1_comp.orange_juice = day1_comp.water)
  (h2 : day2_comp.orange_juice = day1_comp.orange_juice)
  (h3 : day2_comp.water = 2 * day2_comp.orange_juice)
  (h4 : day2_price.price = 32/100)
  (h5 : ∃ (day1_price : OrangeadePrice), 
        revenue day1_price (total_volume day1_comp) = 
        revenue day2_price (total_volume day2_comp)) :
  ∃ (day1_price : OrangeadePrice), day1_price.price = 48/100 := by
sorry


end NUMINAMATH_CALUDE_orangeade_price_theorem_l2181_218140


namespace NUMINAMATH_CALUDE_triangle_theorem_l2181_218176

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (a, b + c)
  let n : ℝ × ℝ := (1, Real.cos C + Real.sqrt 3 * Real.sin C)
  (∀ k : ℝ, m = k • n) ∧  -- m is parallel to n
  3 * b * c = 16 - a^2 ∧
  A = Real.pi / 3 ∧
  (∀ S : ℝ, S = 1/2 * b * c * Real.sin A → S ≤ Real.sqrt 3)

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  triangle_problem a b c A B C → 
    A = Real.pi / 3 ∧
    (∃ S : ℝ, S = 1/2 * b * c * Real.sin A ∧ S = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2181_218176


namespace NUMINAMATH_CALUDE_total_photos_lisa_robert_l2181_218179

def claire_photos : ℕ := 8
def lisa_photos : ℕ := 3 * claire_photos
def robert_photos : ℕ := claire_photos + 16

theorem total_photos_lisa_robert : lisa_photos + robert_photos = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_lisa_robert_l2181_218179


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2181_218137

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_properties (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3 → increasing_sequence a) ∧
  (increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  (a 1 ≥ a 2 ∧ a 2 ≥ a 3 → ¬increasing_sequence a) ∧
  (¬increasing_sequence a → a 1 ≥ a 2 ∧ a 2 ≥ a 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2181_218137


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2181_218132

theorem inequality_solution_sets (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 3) →
  (∀ x : ℝ, (b*x^2 + a) / (x + 1) > 0 ↔ x < -1 ∨ (-1/3 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2181_218132


namespace NUMINAMATH_CALUDE_equation_solution_l2181_218164

theorem equation_solution : ∃ x : ℚ, 25 - 8 = 3 * x + 1 ∧ x = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2181_218164


namespace NUMINAMATH_CALUDE_not_all_primes_from_cards_l2181_218199

/-- A card with two digits -/
structure Card :=
  (front : Nat)
  (back : Nat)

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Generate all two-digit numbers from two cards -/
def twoDigitNumbers (card1 card2 : Card) : List Nat :=
  [
    10 * card1.front + card2.front,
    10 * card1.front + card2.back,
    10 * card1.back + card2.front,
    10 * card1.back + card2.back,
    10 * card2.front + card1.front,
    10 * card2.front + card1.back,
    10 * card2.back + card1.front,
    10 * card2.back + card1.back
  ]

/-- Main theorem -/
theorem not_all_primes_from_cards :
  ∀ (card1 card2 : Card),
    card1.front ≠ card1.back ∧
    card2.front ≠ card2.back ∧
    card1.front ≠ card2.front ∧
    card1.front ≠ card2.back ∧
    card1.back ≠ card2.front ∧
    card1.back ≠ card2.back ∧
    card1.front < 10 ∧ card1.back < 10 ∧ card2.front < 10 ∧ card2.back < 10 →
    ∃ (n : Nat), n ∈ twoDigitNumbers card1 card2 ∧ ¬isPrime n :=
by sorry

end NUMINAMATH_CALUDE_not_all_primes_from_cards_l2181_218199


namespace NUMINAMATH_CALUDE_sum_of_product_sequence_l2181_218152

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ 2 * (a 3) = a 2

def arithmetic_sequence (b : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  b 1 = 1 ∧ S 3 = b 2 + 4

theorem sum_of_product_sequence
  (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (T : ℕ → ℚ) :
  geometric_sequence a →
  arithmetic_sequence b S →
  (∀ n : ℕ, T n = (a n) * (b n)) →
  ∀ n : ℕ, T n = 2 - (n + 2) * (1/2)^n :=
by sorry

end NUMINAMATH_CALUDE_sum_of_product_sequence_l2181_218152


namespace NUMINAMATH_CALUDE_exponential_comparison_l2181_218141

theorem exponential_comparison (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a^(-1 : ℝ) > a^(2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l2181_218141


namespace NUMINAMATH_CALUDE_train_probabilities_l2181_218118

/-- Three independent events with given probabilities -/
structure ThreeIndependentEvents where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  p1_in_range : 0 ≤ p1 ∧ p1 ≤ 1
  p2_in_range : 0 ≤ p2 ∧ p2 ≤ 1
  p3_in_range : 0 ≤ p3 ∧ p3 ≤ 1

/-- The probability of exactly two events occurring -/
def prob_exactly_two (e : ThreeIndependentEvents) : ℝ :=
  e.p1 * e.p2 * (1 - e.p3) + e.p1 * (1 - e.p2) * e.p3 + (1 - e.p1) * e.p2 * e.p3

/-- The probability of at least one event occurring -/
def prob_at_least_one (e : ThreeIndependentEvents) : ℝ :=
  1 - (1 - e.p1) * (1 - e.p2) * (1 - e.p3)

/-- Theorem stating the probabilities for the given scenario -/
theorem train_probabilities (e : ThreeIndependentEvents) 
  (h1 : e.p1 = 0.8) (h2 : e.p2 = 0.7) (h3 : e.p3 = 0.9) : 
  prob_exactly_two e = 0.398 ∧ prob_at_least_one e = 0.994 := by
  sorry

end NUMINAMATH_CALUDE_train_probabilities_l2181_218118


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2181_218114

theorem remainder_divisibility (n : ℤ) : 
  ∃ k : ℤ, n = 125 * k + 40 → ∃ m : ℤ, n = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2181_218114


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_63_and_151_l2181_218128

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let first := (lower + 3) / 4 * 4
  let last := upper / 4 * 4
  let n := (last - first) / 4 + 1
  n * (first + last) / 2

theorem sum_of_multiples_of_4_between_63_and_151 :
  sumOfMultiplesOf4 63 151 = 2332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_63_and_151_l2181_218128


namespace NUMINAMATH_CALUDE_second_duck_bread_pieces_l2181_218130

theorem second_duck_bread_pieces : 
  ∀ (total_bread pieces_left first_duck_fraction last_duck_pieces : ℕ),
  total_bread = 100 →
  pieces_left = 30 →
  first_duck_fraction = 2 →  -- Represents 1/2
  last_duck_pieces = 7 →
  ∃ (second_duck_pieces : ℕ),
    second_duck_pieces = total_bread - pieces_left - (total_bread / first_duck_fraction) - last_duck_pieces ∧
    second_duck_pieces = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_duck_bread_pieces_l2181_218130


namespace NUMINAMATH_CALUDE_fishing_moratorium_purpose_l2181_218194

/-- Represents a fishing moratorium period -/
structure FishingMoratorium where
  start_date : Nat
  end_date : Nat
  regulations : String

/-- Represents the purpose of a fishing moratorium -/
inductive MoratoriumPurpose
  | ProtectEndangeredSpecies
  | ReducePollution
  | ProtectFishermen
  | SustainableUse

/-- The main purpose of the fishing moratorium -/
def main_purpose (moratorium : FishingMoratorium) : MoratoriumPurpose := sorry

/-- Theorem stating the main purpose of the fishing moratorium -/
theorem fishing_moratorium_purpose 
  (moratorium : FishingMoratorium)
  (h1 : moratorium.start_date = 20150516)
  (h2 : moratorium.end_date = 20150801)
  (h3 : moratorium.regulations = "Ministry of Agriculture regulations") :
  main_purpose moratorium = MoratoriumPurpose.SustainableUse := by sorry

end NUMINAMATH_CALUDE_fishing_moratorium_purpose_l2181_218194


namespace NUMINAMATH_CALUDE_cow_chicken_goat_problem_l2181_218163

theorem cow_chicken_goat_problem (cows chickens goats : ℕ) : 
  cows + chickens + goats = 12 →
  4 * cows + 2 * chickens + 4 * goats = 18 + 2 * (cows + chickens + goats) →
  cows + goats = 9 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_goat_problem_l2181_218163


namespace NUMINAMATH_CALUDE_pizza_distribution_l2181_218107

theorem pizza_distribution (num_students : ℕ) (pieces_per_pizza : ℕ) (total_pieces : ℕ) :
  num_students = 10 →
  pieces_per_pizza = 6 →
  total_pieces = 1200 →
  (total_pieces / pieces_per_pizza) / num_students = 20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_distribution_l2181_218107


namespace NUMINAMATH_CALUDE_goat_redistribution_impossibility_l2181_218150

theorem goat_redistribution_impossibility :
  ¬ ∃ (n m : ℕ), n + 7 * m = 150 ∧ 7 * n + m = 150 :=
by sorry

end NUMINAMATH_CALUDE_goat_redistribution_impossibility_l2181_218150


namespace NUMINAMATH_CALUDE_total_harvest_l2181_218162

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem: The total number of sacks of oranges harvested after 6 days is 498 -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l2181_218162


namespace NUMINAMATH_CALUDE_range_of_m_l2181_218181

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the necessary condition
def necessary_condition (m : ℝ) (x : ℝ) : Prop :=
  x < m - 1 ∨ x > m + 1

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x : ℝ, f x > 0 → necessary_condition m x) ∧
    (∃ x : ℝ, necessary_condition m x ∧ f x ≤ 0) →
    0 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2181_218181


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l2181_218101

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*x + 2 = 0 ∧ y^2 - 3*y + 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.tan (α + β) = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l2181_218101


namespace NUMINAMATH_CALUDE_cookie_count_theorem_l2181_218146

/-- Represents a pack of cookies with a specific number of cookies -/
structure CookiePack where
  cookies : ℕ

/-- Represents a person's purchase of cookie packs -/
structure Purchase where
  packA : ℕ
  packB : ℕ
  packC : ℕ
  packD : ℕ

def packA : CookiePack := ⟨15⟩
def packB : CookiePack := ⟨30⟩
def packC : CookiePack := ⟨45⟩
def packD : CookiePack := ⟨60⟩

def paulPurchase : Purchase := ⟨1, 2, 0, 0⟩
def paulaPurchase : Purchase := ⟨1, 0, 1, 0⟩

def totalCookies (p : Purchase) : ℕ :=
  p.packA * packA.cookies + p.packB * packB.cookies + p.packC * packC.cookies + p.packD * packD.cookies

theorem cookie_count_theorem :
  totalCookies paulPurchase + totalCookies paulaPurchase = 135 := by
  sorry


end NUMINAMATH_CALUDE_cookie_count_theorem_l2181_218146


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2181_218185

theorem unique_integer_solution : 
  ∀ n : ℤ, (⌊(n^2 / 4 : ℚ) + n⌋ - ⌊n / 2⌋^2 = 5) ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2181_218185


namespace NUMINAMATH_CALUDE_expression_evaluation_l2181_218186

theorem expression_evaluation : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2181_218186


namespace NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l2181_218145

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  height : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (m : SubmergedMountain) : ℝ :=
  m.height * (1 - (m.aboveWaterVolumeFraction ^ (1/3)))

/-- The theorem stating the ocean depth for a specific mountain -/
theorem ocean_depth_for_specific_mountain : 
  let m : SubmergedMountain := { height := 12000, aboveWaterVolumeFraction := 1/5 }
  oceanDepth m = 864 := by
  sorry

end NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l2181_218145


namespace NUMINAMATH_CALUDE_circle_rectangle_area_relation_l2181_218193

theorem circle_rectangle_area_relation (x : ℝ) :
  let circle_radius : ℝ := x - 2
  let rectangle_length : ℝ := x - 3
  let rectangle_width : ℝ := x + 4
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  rectangle_area = 3 * circle_area →
  (12 * π + 1) / (3 * π - 1) = x + (-(12 * π + 1) / (2 * (1 - 3 * π)) + (12 * π + 1) / (2 * (1 - 3 * π))) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_rectangle_area_relation_l2181_218193


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2181_218178

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) := by
  sorry

#check symmetric_point_coordinates

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2181_218178


namespace NUMINAMATH_CALUDE_no_integer_cube_equal_3n2_plus_3n_plus_7_l2181_218177

theorem no_integer_cube_equal_3n2_plus_3n_plus_7 :
  ¬ ∃ (n m : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_cube_equal_3n2_plus_3n_plus_7_l2181_218177


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2181_218119

theorem power_mod_eleven : 6^305 % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2181_218119


namespace NUMINAMATH_CALUDE_new_average_is_34_l2181_218175

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverage (performance : BatsmanPerformance) : ℕ :=
  performance.lastInningScore + (performance.innings - 1) * (performance.lastInningScore / performance.innings + performance.averageIncrease - 3)

/-- Theorem stating that the new average is 34 for the given conditions -/
theorem new_average_is_34 (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningScore = 82)
  (h3 : performance.averageIncrease = 3) :
  newAverage performance = 34 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_34_l2181_218175


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2181_218198

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  1/m + 1/n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2181_218198


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2181_218126

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) = s n + d

/-- The last term of a finite arithmetic sequence. -/
def last_term (s : ℕ → ℤ) (n : ℕ) : ℤ := s (n - 1)

theorem arithmetic_sequence_length :
  ∀ s : ℕ → ℤ,
  is_arithmetic_sequence s →
  s 0 = -3 →
  last_term s 13 = 45 →
  ∃ n : ℕ, n = 13 ∧ last_term s n = 45 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2181_218126


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2181_218142

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * x * y) + f (f (x + y)) = x * f y + y * f x + f (x + y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = 2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2181_218142


namespace NUMINAMATH_CALUDE_alex_mean_score_l2181_218170

def scores : List ℝ := [86, 88, 90, 91, 95, 99]

def jane_score_count : ℕ := 2
def alex_score_count : ℕ := 4
def jane_mean_score : ℝ := 93

theorem alex_mean_score : 
  (scores.sum - jane_score_count * jane_mean_score) / alex_score_count = 90.75 := by
  sorry

end NUMINAMATH_CALUDE_alex_mean_score_l2181_218170


namespace NUMINAMATH_CALUDE_range_of_a_l2181_218122

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2181_218122


namespace NUMINAMATH_CALUDE_sqrt_8_div_7_same_type_as_sqrt_2_l2181_218192

-- Define what it means for two quadratic radicals to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_div_7_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8 / 7) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt (1/3)) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 12) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_8_div_7_same_type_as_sqrt_2_l2181_218192


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2181_218123

/-- The equation x^2 - 18y^2 - 6x + 4y + 9 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), x^2 - 18*y^2 - 6*x + 4*y + 9 = 0 ↔
  ((x - c) / a)^2 - ((y - d) / b)^2 = 1 ∧
  e = 1 := by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2181_218123


namespace NUMINAMATH_CALUDE_comic_book_pages_l2181_218108

/-- Given that Trevor drew 220 pages in total over three months,
    and the third month's issue was four pages longer than the others,
    prove that the first issue had 72 pages. -/
theorem comic_book_pages :
  ∀ (x : ℕ),
  (x + x + (x + 4) = 220) →
  (x = 72) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_pages_l2181_218108


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2181_218120

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2181_218120


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2181_218127

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ = -14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2181_218127


namespace NUMINAMATH_CALUDE_soldier_average_score_l2181_218189

theorem soldier_average_score : 
  let shots : List ℕ := List.replicate 6 10 ++ [9] ++ List.replicate 3 8
  (shots.sum : ℚ) / shots.length = 93/10 := by
  sorry

end NUMINAMATH_CALUDE_soldier_average_score_l2181_218189


namespace NUMINAMATH_CALUDE_jump_distance_difference_l2181_218144

theorem jump_distance_difference (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 13)
  (h2 : frog_jump = 11) :
  grasshopper_jump - frog_jump = 2 := by
  sorry

end NUMINAMATH_CALUDE_jump_distance_difference_l2181_218144


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2181_218100

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + π / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2181_218100


namespace NUMINAMATH_CALUDE_second_round_score_l2181_218109

/-- Represents the number of darts thrown in each round -/
def darts_per_round : ℕ := 8

/-- Represents the minimum points per dart -/
def min_points_per_dart : ℕ := 3

/-- Represents the maximum points per dart -/
def max_points_per_dart : ℕ := 9

/-- Represents the points scored in the first round -/
def first_round_points : ℕ := 24

/-- Represents the ratio of points scored in the second round compared to the first round -/
def second_round_ratio : ℚ := 2

/-- Represents the ratio of points scored in the third round compared to the second round -/
def third_round_ratio : ℚ := (3/2 : ℚ)

/-- Theorem stating that Misha scored 48 points in the second round -/
theorem second_round_score : 
  first_round_points * second_round_ratio = 48 := by sorry

end NUMINAMATH_CALUDE_second_round_score_l2181_218109


namespace NUMINAMATH_CALUDE_union_A_B_when_m_half_B_subset_A_iff_m_range_l2181_218161

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + m) * (x - 2*m - 1) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Statement 1
theorem union_A_B_when_m_half : 
  A (1/2) ∪ B = {x | -2 < x ∧ x < 2} := by sorry

-- Statement 2
theorem B_subset_A_iff_m_range :
  ∀ m : ℝ, B ⊆ A m ↔ m ≤ -3/2 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_half_B_subset_A_iff_m_range_l2181_218161
