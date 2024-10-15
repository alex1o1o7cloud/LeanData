import Mathlib

namespace NUMINAMATH_CALUDE_elastic_collision_momentum_exchange_l1791_179192

/-- Represents a particle with mass and velocity -/
structure Particle where
  mass : ℝ
  velocity : ℝ

/-- Calculates the momentum of a particle -/
def momentum (p : Particle) : ℝ := p.mass * p.velocity

/-- Represents the state of two particles before and after a collision -/
structure CollisionState where
  particle1 : Particle
  particle2 : Particle

/-- Defines an elastic head-on collision between two identical particles -/
def elasticCollision (initial : CollisionState) (final : CollisionState) : Prop :=
  initial.particle1.mass = initial.particle2.mass ∧
  initial.particle1.mass = final.particle1.mass ∧
  initial.particle2.velocity = 0 ∧
  momentum initial.particle1 + momentum initial.particle2 = momentum final.particle1 + momentum final.particle2 ∧
  (momentum initial.particle1)^2 + (momentum initial.particle2)^2 = (momentum final.particle1)^2 + (momentum final.particle2)^2

theorem elastic_collision_momentum_exchange 
  (initial final : CollisionState)
  (h_elastic : elasticCollision initial final)
  (h_initial_momentum : momentum initial.particle1 = p ∧ momentum initial.particle2 = 0) :
  momentum final.particle1 = 0 ∧ momentum final.particle2 = p := by
  sorry

end NUMINAMATH_CALUDE_elastic_collision_momentum_exchange_l1791_179192


namespace NUMINAMATH_CALUDE_sequence_less_than_two_l1791_179160

theorem sequence_less_than_two (a : ℕ → ℝ) :
  (∀ n, a n < 2) ↔ ¬(∃ k, a k ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_less_than_two_l1791_179160


namespace NUMINAMATH_CALUDE_intersection_product_range_l1791_179158

open Real

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := ∃ φ, x = 2 * cos φ ∧ y = sin φ
def l (θ ρ : ℝ) (α : ℝ) : Prop := θ = α ∧ ρ > 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ π/4

-- Define the polar equations
def C₁_polar (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
def C₂_polar (ρ θ : ℝ) : Prop := ρ^2 = 4 / (1 + 3 * sin θ^2)

-- Define the intersection points
def M (ρ_M : ℝ) (α : ℝ) : Prop := C₁_polar ρ_M α ∧ l α ρ_M α
def N (ρ_N : ℝ) (α : ℝ) : Prop := C₂_polar ρ_N α ∧ l α ρ_N α

-- State the theorem
theorem intersection_product_range :
  ∀ α ρ_M ρ_N, α_range α → M ρ_M α → N ρ_N α → ρ_M ≠ 0 → ρ_N ≠ 0 →
  (8 * sqrt 5 / 5) ≤ ρ_M * ρ_N ∧ ρ_M * ρ_N ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_range_l1791_179158


namespace NUMINAMATH_CALUDE_sequence_properties_l1791_179173

def arithmetic_seq (a b n : ℕ) : ℕ := a + (n - 1) * b

def geometric_seq (b a n : ℕ) : ℕ := b * a^(n - 1)

def c_seq (a b n : ℕ) : ℚ := (arithmetic_seq a b n - 8) / (geometric_seq b a n)

theorem sequence_properties (a b : ℕ) :
  (a > 0) →
  (b > 0) →
  (arithmetic_seq a b 1 < geometric_seq b a 1) →
  (geometric_seq b a 1 < arithmetic_seq a b 2) →
  (arithmetic_seq a b 2 < geometric_seq b a 2) →
  (geometric_seq b a 2 < arithmetic_seq a b 3) →
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ arithmetic_seq a b m + 1 = geometric_seq b a n) →
  (a = 2 ∧ b = 3 ∧ ∃ k : ℕ, ∀ n : ℕ, n > 0 → c_seq a b n ≤ c_seq a b k ∧ c_seq a b k = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1791_179173


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1791_179191

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  expectation : ℝ
  variance : ℝ
  h1 : 0 < p ∧ p < 1
  h2 : expectation = n * p
  h3 : variance = n * p * (1 - p)

/-- Theorem stating that a binomial distribution with given expectation and variance has specific n and p values -/
theorem binomial_distribution_unique_parameters
  (ξ : BinomialDistribution)
  (h_expectation : ξ.expectation = 2.4)
  (h_variance : ξ.variance = 1.44) :
  ξ.n = 6 ∧ ξ.p = 0.4 :=
sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1791_179191


namespace NUMINAMATH_CALUDE_zachary_did_19_pushups_l1791_179190

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := 58

/-- The difference between David's and Zachary's push-ups -/
def difference : ℕ := 39

/-- Theorem stating that Zachary did 19 push-ups given the conditions -/
theorem zachary_did_19_pushups : zachary_pushups = 19 := by sorry

end NUMINAMATH_CALUDE_zachary_did_19_pushups_l1791_179190


namespace NUMINAMATH_CALUDE_chess_match_outcomes_count_l1791_179166

/-- The number of different possible outcomes for a chess match draw -/
def chessMatchOutcomes : ℕ :=
  2^8 * Nat.factorial 8

/-- Theorem stating the number of different possible outcomes for a chess match draw -/
theorem chess_match_outcomes_count :
  chessMatchOutcomes = 2^8 * Nat.factorial 8 := by
  sorry

#eval chessMatchOutcomes

end NUMINAMATH_CALUDE_chess_match_outcomes_count_l1791_179166


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1791_179156

-- Define the quadratic function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
def F (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := F x - k * x

theorem quadratic_function_theorem (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x : ℝ, F x = f a b x) ∧ 
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g k)) ↔ (k ≤ -2 ∨ k ≥ 6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1791_179156


namespace NUMINAMATH_CALUDE_cherry_tomatoes_per_jar_l1791_179193

theorem cherry_tomatoes_per_jar 
  (total_tomatoes : ℕ) 
  (num_jars : ℕ) 
  (h1 : total_tomatoes = 56) 
  (h2 : num_jars = 7) : 
  total_tomatoes / num_jars = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_per_jar_l1791_179193


namespace NUMINAMATH_CALUDE_equal_roots_implies_c_value_l1791_179199

-- Define the quadratic equation
def quadratic (x c : ℝ) : ℝ := x^2 + 6*x - c

-- Define the discriminant of the quadratic equation
def discriminant (c : ℝ) : ℝ := 6^2 - 4*(1)*(-c)

-- Theorem statement
theorem equal_roots_implies_c_value :
  (∃ x : ℝ, quadratic x c = 0 ∧ 
    ∀ y : ℝ, quadratic y c = 0 → y = x) →
  c = -9 := by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_c_value_l1791_179199


namespace NUMINAMATH_CALUDE_laundry_time_calculation_l1791_179120

theorem laundry_time_calculation (loads : ℕ) (wash_time : ℕ) (dry_time : ℕ) :
  loads = 8 ∧ wash_time = 45 ∧ dry_time = 60 →
  (loads * (wash_time + dry_time)) / 60 = 14 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_calculation_l1791_179120


namespace NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l1791_179175

/-- A triangle with sides a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The condition R(b+c) = a√(bc) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.R * (t.b + t.c) = t.a * Real.sqrt (t.b * t.c)

/-- Definition of an isosceles right triangle -/
def isIsoscelesRight (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a * t.a + t.b * t.b = t.c * t.c

/-- The main theorem -/
theorem triangle_condition_implies_isosceles_right (t : Triangle) :
  satisfiesCondition t → isIsoscelesRight t :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l1791_179175


namespace NUMINAMATH_CALUDE_last_duck_bread_pieces_l1791_179164

theorem last_duck_bread_pieces (total : ℕ) (left : ℕ) (first_duck : ℕ) (second_duck : ℕ) :
  total = 100 →
  left = 30 →
  first_duck = total / 2 →
  second_duck = 13 →
  total - left - first_duck - second_duck = 7 :=
by sorry

end NUMINAMATH_CALUDE_last_duck_bread_pieces_l1791_179164


namespace NUMINAMATH_CALUDE_composite_face_dots_l1791_179149

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four faces of interest in the composite figure -/
inductive CompositeFace
  | A
  | B
  | C
  | D

/-- A function that returns the number of dots on a die face -/
def dots_on_face (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- A function that maps a composite face to its corresponding die face -/
def composite_to_die_face (face : CompositeFace) : DieFace :=
  match face with
  | CompositeFace.A => DieFace.three
  | CompositeFace.B => DieFace.five
  | CompositeFace.C => DieFace.six
  | CompositeFace.D => DieFace.five

/-- Theorem stating the number of dots on each composite face -/
theorem composite_face_dots (face : CompositeFace) :
  dots_on_face (composite_to_die_face face) =
    match face with
    | CompositeFace.A => 3
    | CompositeFace.B => 5
    | CompositeFace.C => 6
    | CompositeFace.D => 5 := by
  sorry

end NUMINAMATH_CALUDE_composite_face_dots_l1791_179149


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1791_179198

theorem right_triangle_side_length : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 13 → a = 12 →
  c^2 = a^2 + b^2 →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1791_179198


namespace NUMINAMATH_CALUDE_triangle_property_l1791_179104

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - 5)^2 + |t.b - 12| + (t.c - 13)^2 = 0

-- Define what it means to be a right triangle with c as hypotenuse
def is_right_triangle_with_c_hypotenuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  is_right_triangle_with_c_hypotenuse t :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l1791_179104


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_l1791_179187

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a^2 * b^2 * c^2)^(1/3) + ((1 - a^2) * (1 - b^2) * (1 - c^2))^(1/3) ≤ 1 ∧
  ∃ x y z, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧
    (x^2 * y^2 * z^2)^(1/3) + ((1 - x^2) * (1 - y^2) * (1 - z^2))^(1/3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_l1791_179187


namespace NUMINAMATH_CALUDE_water_moles_equal_cao_moles_l1791_179110

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product : String

-- Define the molar quantities
structure MolarQuantities where
  cao_moles : ℝ
  h2o_moles : ℝ
  caoh2_moles : ℝ

-- Define the problem parameters
def cao_mass : ℝ := 168
def cao_molar_mass : ℝ := 56.08
def target_caoh2_moles : ℝ := 3

-- Define the reaction
def calcium_hydroxide_reaction : Reaction :=
  { reactant1 := "CaO", reactant2 := "H2O", product := "Ca(OH)2" }

-- Theorem statement
theorem water_moles_equal_cao_moles 
  (reaction : Reaction) 
  (quantities : MolarQuantities) :
  reaction = calcium_hydroxide_reaction →
  quantities.caoh2_moles = target_caoh2_moles →
  quantities.cao_moles = cao_mass / cao_molar_mass →
  quantities.h2o_moles = quantities.cao_moles :=
by sorry

end NUMINAMATH_CALUDE_water_moles_equal_cao_moles_l1791_179110


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1791_179169

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧
  (x*y/z + y*z/x + z*x/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1791_179169


namespace NUMINAMATH_CALUDE_certain_number_power_l1791_179181

theorem certain_number_power (m : ℤ) (a : ℝ) : 
  (-2 : ℝ)^(2*m) = a^(21-m) → m = 7 → a = -2 := by sorry

end NUMINAMATH_CALUDE_certain_number_power_l1791_179181


namespace NUMINAMATH_CALUDE_trajectory_of_point_m_l1791_179161

/-- The trajectory of point M on a line segment AB with given conditions -/
theorem trajectory_of_point_m (a : ℝ) (x y : ℝ) :
  (∃ (m b : ℝ),
    -- AB has length 2a
    m^2 + b^2 = (2*a)^2 ∧
    -- M divides AB in ratio 1:2
    x = (2/3) * m ∧
    y = (1/3) * b) →
  x^2 / ((4/3 * a)^2) + y^2 / ((2/3 * a)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_m_l1791_179161


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1791_179155

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, eccentricity e, and length of real axis 2a,
    prove that the distance from the focus to the asymptote line is √3 when e = 2 and 2a = 2. -/
theorem hyperbola_focus_asymptote_distance
  (a b c : ℝ)
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = 2)
  (h_real_axis : 2 * a = 2) :
  (b * c) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1791_179155


namespace NUMINAMATH_CALUDE_circle_polygons_l1791_179150

theorem circle_polygons (n : ℕ) (h : n = 15) :
  let quadrilaterals := Nat.choose n 4
  let triangles := Nat.choose n 3
  quadrilaterals + triangles = 1820 := by
  sorry

end NUMINAMATH_CALUDE_circle_polygons_l1791_179150


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1791_179144

theorem systematic_sampling_probability (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 121) (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_students = 20 / 121 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1791_179144


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l1791_179195

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l1791_179195


namespace NUMINAMATH_CALUDE_min_distance_between_line_and_curve_l1791_179142

/-- The minimum distance between a point on y = 2x + 1 and a point on y = x + ln x -/
theorem min_distance_between_line_and_curve : ∃ (d : ℝ), d = (2 * Real.sqrt 5) / 5 ∧
  ∀ (P Q : ℝ × ℝ),
    (P.2 = 2 * P.1 + 1) →
    (Q.2 = Q.1 + Real.log Q.1) →
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_line_and_curve_l1791_179142


namespace NUMINAMATH_CALUDE_original_average_proof_l1791_179132

theorem original_average_proof (n : ℕ) (original_average : ℚ) : 
  n = 12 → 
  (2 * original_average * n) / n = 100 →
  original_average = 50 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l1791_179132


namespace NUMINAMATH_CALUDE_complex_roots_of_quadratic_l1791_179188

theorem complex_roots_of_quadratic : 
  let z₁ : ℂ := -1 + Real.sqrt 2 - Complex.I * Real.sqrt 2
  let z₂ : ℂ := -1 - Real.sqrt 2 + Complex.I * Real.sqrt 2
  (z₁^2 + 2*z₁ = 3 - 4*Complex.I) ∧ (z₂^2 + 2*z₂ = 3 - 4*Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_complex_roots_of_quadratic_l1791_179188


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1791_179118

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n! + 2 < p ∧ p < n! + n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1791_179118


namespace NUMINAMATH_CALUDE_max_students_distribution_l1791_179189

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 2500) (h2 : pencils = 1575) : 
  Nat.gcd pens pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1791_179189


namespace NUMINAMATH_CALUDE_probability_not_pair_is_four_fifths_l1791_179127

def num_pairs : ℕ := 3
def num_shoes : ℕ := 2 * num_pairs

def probability_not_pair : ℚ :=
  1 - (num_pairs * 1 : ℚ) / (num_shoes.choose 2 : ℚ)

theorem probability_not_pair_is_four_fifths :
  probability_not_pair = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_pair_is_four_fifths_l1791_179127


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l1791_179147

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 - years_since_built →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary = 95 := by
  sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l1791_179147


namespace NUMINAMATH_CALUDE_root_value_theorem_l1791_179163

theorem root_value_theorem (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 2026 - m^2 + 2*m = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1791_179163


namespace NUMINAMATH_CALUDE_triangle_properties_l1791_179112

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (BD : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin ((A + C) / 2) = b * Real.sin C ∧
  BD = 1 ∧
  b = Real.sqrt 3 ∧
  BD * (a * Real.sin C) = b * c * Real.sin (π / 2) →
  B = π / 3 ∧ 
  a + b + c = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1791_179112


namespace NUMINAMATH_CALUDE_hexagon_circumscribable_l1791_179133

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon defined by six points -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Checks if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def equal_length (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a circle can be circumscribed around a set of points -/
def can_circumscribe (points : List Point) : Prop := sorry

/-- Theorem: A circle can be circumscribed around a hexagon with the given properties -/
theorem hexagon_circumscribable (h : Hexagon) :
  parallel h.A h.B h.D h.E →
  parallel h.B h.C h.E h.F →
  parallel h.C h.D h.F h.A →
  equal_length h.A h.D h.B h.E →
  equal_length h.A h.D h.C h.F →
  can_circumscribe [h.A, h.B, h.C, h.D, h.E, h.F] := by
  sorry

end NUMINAMATH_CALUDE_hexagon_circumscribable_l1791_179133


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1791_179116

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1791_179116


namespace NUMINAMATH_CALUDE_existence_of_x0_l1791_179145

theorem existence_of_x0 (a : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1) 1 ∧ x₀^3 < -x₀^2 + x₀ - a) ↔
  (a > 5/27) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_x0_l1791_179145


namespace NUMINAMATH_CALUDE_distributive_property_fraction_l1791_179114

theorem distributive_property_fraction (x y : ℝ) :
  (x + y) / 2 = x / 2 + y / 2 := by sorry

end NUMINAMATH_CALUDE_distributive_property_fraction_l1791_179114


namespace NUMINAMATH_CALUDE_non_zero_digits_after_decimal_l1791_179119

theorem non_zero_digits_after_decimal (n : ℕ) (d : ℕ) : 
  (720 : ℚ) / (2^5 * 5^9) = n / (10^d) ∧ 
  n % 10 ≠ 0 ∧
  n < 10^4 ∧ 
  n ≥ 10^3 →
  d = 8 :=
sorry

end NUMINAMATH_CALUDE_non_zero_digits_after_decimal_l1791_179119


namespace NUMINAMATH_CALUDE_point_transformation_final_coordinates_l1791_179174

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to find the symmetric point about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Function to move a point to the left -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  { x := p.x - units, y := p.y }

/-- Theorem stating that the given transformations result in the expected point -/
theorem point_transformation (initialPoint : Point) :
  initialPoint.x = -2 ∧ initialPoint.y = 3 →
  (moveLeft (symmetricAboutOrigin initialPoint) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin initialPoint) 2).y = -3 := by
  sorry

/-- Main theorem proving the final coordinates -/
theorem final_coordinates : ∃ (p : Point),
  p.x = -2 ∧ p.y = 3 ∧
  (moveLeft (symmetricAboutOrigin p) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin p) 2).y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_final_coordinates_l1791_179174


namespace NUMINAMATH_CALUDE_sin_difference_of_inverse_trig_functions_l1791_179194

theorem sin_difference_of_inverse_trig_functions :
  Real.sin (Real.arcsin (3/5) - Real.arctan (1/2)) = 2 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_of_inverse_trig_functions_l1791_179194


namespace NUMINAMATH_CALUDE_number_ratio_l1791_179146

theorem number_ratio (f s t : ℝ) : 
  t = 2 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  s / f = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1791_179146


namespace NUMINAMATH_CALUDE_locus_of_nine_point_center_on_BC_l1791_179182

/-- Triangle ABC with fixed vertices B and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (1, 0)

/-- The nine-point center of a triangle -/
def ninePointCenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is on a line segment -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- The locus of point A -/
def locusOfA (x y : ℝ) : Prop := x^2 - y^2 = 1

theorem locus_of_nine_point_center_on_BC (t : Triangle) :
  isOnSegment (ninePointCenter t) t.B t.C ↔ locusOfA t.A.1 t.A.2 := by sorry

end NUMINAMATH_CALUDE_locus_of_nine_point_center_on_BC_l1791_179182


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l1791_179139

theorem polynomial_root_problem (d : ℚ) :
  (∃ (x : ℝ), x^3 + 4*x + d = 0 ∧ x = 2 + Real.sqrt 5) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0 ∧ n = -4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l1791_179139


namespace NUMINAMATH_CALUDE_greatest_k_value_l1791_179130

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = 10) →
  k ≤ 2 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1791_179130


namespace NUMINAMATH_CALUDE_function_inequality_l1791_179177

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

-- State the theorem
theorem function_inequality :
  (∀ x, -8 < x → x < 8 → f x ≠ 0) →  -- f is defined on (-8, 8)
  is_even f →
  is_monotonic_on f 0 8 →
  f (-3) < f 2 →
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1791_179177


namespace NUMINAMATH_CALUDE_price_increase_calculation_l1791_179123

/-- Represents the ticket pricing model for an airline -/
structure TicketPricing where
  basePrice : ℝ
  daysBeforeDeparture : ℕ
  dailyIncreaseRate : ℝ

/-- Calculates the price increase for buying a ticket one day later -/
def priceIncrease (pricing : TicketPricing) : ℝ :=
  pricing.basePrice * pricing.dailyIncreaseRate

/-- Theorem: The price increase for buying a ticket one day later is $52.50 -/
theorem price_increase_calculation (pricing : TicketPricing)
  (h1 : pricing.basePrice = 1050)
  (h2 : pricing.daysBeforeDeparture = 14)
  (h3 : pricing.dailyIncreaseRate = 0.05) :
  priceIncrease pricing = 52.50 := by
  sorry

#eval priceIncrease { basePrice := 1050, daysBeforeDeparture := 14, dailyIncreaseRate := 0.05 }

end NUMINAMATH_CALUDE_price_increase_calculation_l1791_179123


namespace NUMINAMATH_CALUDE_square_root_of_16_l1791_179165

theorem square_root_of_16 : Real.sqrt 16 = 4 ∧ Real.sqrt 16 = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_16_l1791_179165


namespace NUMINAMATH_CALUDE_polar_to_cartesian_x_plus_y_bounds_l1791_179186

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the circle in Cartesian coordinates
def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Theorem stating the equivalence of polar and Cartesian equations
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    polar_circle ρ θ ↔ cartesian_circle x y :=
sorry

-- Theorem for the bounds of x + y
theorem x_plus_y_bounds :
  ∀ (x y : ℝ), cartesian_circle x y → 2 ≤ x + y ∧ x + y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_x_plus_y_bounds_l1791_179186


namespace NUMINAMATH_CALUDE_first_lift_weight_l1791_179134

/-- Given two lifts with a total weight of 600 pounds, where twice the weight of the first lift
    is 300 pounds more than the weight of the second lift, prove that the weight of the first lift
    is 300 pounds. -/
theorem first_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 600)
  (lift_relation : 2 * first_lift = second_lift + 300) : 
  first_lift = 300 := by
  sorry

end NUMINAMATH_CALUDE_first_lift_weight_l1791_179134


namespace NUMINAMATH_CALUDE_quartic_sum_l1791_179108

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 3 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- The sum of the polynomial evaluated at 3 and -3 equals 407m -/
theorem quartic_sum (m : ℝ) (P : QuarticPolynomial m) : P.Q 3 + P.Q (-3) = 407 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_l1791_179108


namespace NUMINAMATH_CALUDE_days_to_reach_goal_chris_breath_holding_days_l1791_179162

/-- Given Chris's breath-holding capacity and improvement rate, calculate the number of days to reach his goal. -/
theorem days_to_reach_goal (start_capacity : ℕ) (daily_improvement : ℕ) (goal : ℕ) : ℕ :=
  let days := (goal - start_capacity) / daily_improvement
  days

/-- Prove that Chris needs 6 more days to reach his goal. -/
theorem chris_breath_holding_days : days_to_reach_goal 30 10 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_days_to_reach_goal_chris_breath_holding_days_l1791_179162


namespace NUMINAMATH_CALUDE_point_coordinates_l1791_179168

/-- Given a point A(-m, √m) in the Cartesian coordinate system,
    prove that its coordinates are (-16, 4) if its distance to the x-axis is 4. -/
theorem point_coordinates (m : ℝ) :
  (∃ A : ℝ × ℝ, A = (-m, Real.sqrt m) ∧ |A.2| = 4) →
  (∃ A : ℝ × ℝ, A = (-16, 4)) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1791_179168


namespace NUMINAMATH_CALUDE_marcus_pebble_ratio_l1791_179117

def pebble_ratio (initial : ℕ) (received : ℕ) (final : ℕ) : Prop :=
  let skipped := initial + received - final
  (2 * skipped = initial) ∧ (skipped ≠ 0)

theorem marcus_pebble_ratio :
  pebble_ratio 18 30 39 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebble_ratio_l1791_179117


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1791_179157

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1791_179157


namespace NUMINAMATH_CALUDE_factorization_cubic_factorization_fifth_power_l1791_179100

-- We don't need to prove the first part as no specific factorization was provided

-- Prove the factorization of x^3 + 2x^2 + 4x + 3
theorem factorization_cubic (x : ℝ) : 
  x^3 + 2*x^2 + 4*x + 3 = (x + 1) * (x^2 + x + 3) := by
sorry

-- Prove the factorization of x^5 - 1
theorem factorization_fifth_power (x : ℝ) : 
  x^5 - 1 = (x - 1) * (x^4 + x^3 + x^2 + x + 1) := by
sorry

end NUMINAMATH_CALUDE_factorization_cubic_factorization_fifth_power_l1791_179100


namespace NUMINAMATH_CALUDE_square_sum_xy_l1791_179197

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1791_179197


namespace NUMINAMATH_CALUDE_ellipse_min_distance_sum_l1791_179105

theorem ellipse_min_distance_sum (x y : ℝ) : 
  (x^2 / 2 + y^2 = 1) →  -- Point (x, y) is on the ellipse
  (∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 / 2 + y'^2 = 1 → 
    (x'^2 + y'^2) + ((x' + 1)^2 + y'^2) ≥ min) ∧ 
    min = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_min_distance_sum_l1791_179105


namespace NUMINAMATH_CALUDE_painting_wall_percentage_l1791_179183

/-- Calculates the percentage of a wall taken up by a painting -/
theorem painting_wall_percentage 
  (painting_width : ℝ) 
  (painting_height : ℝ) 
  (wall_width : ℝ) 
  (wall_height : ℝ) 
  (h1 : painting_width = 2) 
  (h2 : painting_height = 4) 
  (h3 : wall_width = 5) 
  (h4 : wall_height = 10) : 
  (painting_width * painting_height) / (wall_width * wall_height) * 100 = 16 := by
  sorry

#check painting_wall_percentage

end NUMINAMATH_CALUDE_painting_wall_percentage_l1791_179183


namespace NUMINAMATH_CALUDE_f_increasing_and_no_negative_roots_l1791_179129

noncomputable section

variable (a : ℝ) (h : a > 1)

def f (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_increasing_and_no_negative_roots :
  (∀ x y, -1 < x ∧ x < y → f a x < f a y) ∧
  (∀ x, x < 0 → f a x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_f_increasing_and_no_negative_roots_l1791_179129


namespace NUMINAMATH_CALUDE_common_chord_equation_l1791_179126

/-- The equation of the line containing the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → x + 2*y = 0 := by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1791_179126


namespace NUMINAMATH_CALUDE_salary_and_new_savings_l1791_179167

/-- Represents expenses as percentages of salary -/
structure Expenses where
  food : ℚ
  rent : ℚ
  entertainment : ℚ
  conveyance : ℚ
  utilities : ℚ
  miscellaneous : ℚ

/-- Calculates the total expenses as a percentage -/
def totalExpenses (e : Expenses) : ℚ :=
  e.food + e.rent + e.entertainment + e.conveyance + e.utilities + e.miscellaneous

/-- Calculates the savings percentage -/
def savingsPercentage (e : Expenses) : ℚ :=
  1 - totalExpenses e

/-- Theorem: Given the initial expenses and savings, prove the monthly salary and new savings percentage -/
theorem salary_and_new_savings 
  (initial_expenses : Expenses)
  (initial_savings : ℚ)
  (salary : ℚ)
  (new_entertainment : ℚ)
  (new_conveyance : ℚ)
  (h1 : initial_expenses.food = 0.30)
  (h2 : initial_expenses.rent = 0.25)
  (h3 : initial_expenses.entertainment = 0.15)
  (h4 : initial_expenses.conveyance = 0.10)
  (h5 : initial_expenses.utilities = 0.05)
  (h6 : initial_expenses.miscellaneous = 0.05)
  (h7 : initial_savings = 1500)
  (h8 : savingsPercentage initial_expenses * salary = initial_savings)
  (h9 : new_entertainment = initial_expenses.entertainment + 0.05)
  (h10 : new_conveyance = initial_expenses.conveyance - 0.03)
  : salary = 15000 ∧ 
    savingsPercentage { initial_expenses with 
      entertainment := new_entertainment,
      conveyance := new_conveyance 
    } = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_salary_and_new_savings_l1791_179167


namespace NUMINAMATH_CALUDE_average_ticket_cost_l1791_179124

/-- Calculates the average cost of tickets per person given the specified conditions --/
theorem average_ticket_cost (full_price : ℕ) (total_people : ℕ) (half_price_tickets : ℕ) (free_tickets : ℕ) (full_price_tickets : ℕ) :
  full_price = 150 →
  total_people = 5 →
  half_price_tickets = 2 →
  free_tickets = 1 →
  full_price_tickets = 2 →
  (full_price * full_price_tickets + (full_price / 2) * half_price_tickets) / total_people = 90 :=
by sorry

end NUMINAMATH_CALUDE_average_ticket_cost_l1791_179124


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l1791_179154

theorem xyz_product_magnitude (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  x + 1/y = y + 1/z → y + 1/z = z + 1/x + 1 →
  |x*y*z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l1791_179154


namespace NUMINAMATH_CALUDE_calvins_collection_size_l1791_179107

/-- Calculates the total number of insects in Calvin's collection. -/
def calvinsTotalInsects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvins_collection_size :
  calvinsTotalInsects 12 3 = 27 := by
  sorry

#eval calvinsTotalInsects 12 3

end NUMINAMATH_CALUDE_calvins_collection_size_l1791_179107


namespace NUMINAMATH_CALUDE_roses_cut_l1791_179138

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1791_179138


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1791_179196

/-- Represents a population with possible strata --/
structure Population where
  total : Nat
  strata : List Nat

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified

/-- Represents a sampling problem --/
structure SamplingProblem where
  population : Population
  sampleSize : Nat

/-- Determines the appropriate sampling method for a given problem --/
def appropriateSamplingMethod (problem : SamplingProblem) : SamplingMethod :=
  sorry

theorem correct_sampling_methods
  (collegeProblem : SamplingProblem)
  (workshopProblem : SamplingProblem)
  (h1 : collegeProblem.population = { total := 300, strata := [150, 150] })
  (h2 : collegeProblem.sampleSize = 100)
  (h3 : workshopProblem.population = { total := 100, strata := [] })
  (h4 : workshopProblem.sampleSize = 10) :
  appropriateSamplingMethod collegeProblem = SamplingMethod.Stratified ∧
  appropriateSamplingMethod workshopProblem = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1791_179196


namespace NUMINAMATH_CALUDE_cube_roots_of_primes_not_in_arithmetic_progression_l1791_179153

theorem cube_roots_of_primes_not_in_arithmetic_progression 
  (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ¬∃ (a d : ℝ), {(p : ℝ)^(1/3), (q : ℝ)^(1/3), (r : ℝ)^(1/3)} ⊆ {a + n * d | n : ℤ} :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_primes_not_in_arithmetic_progression_l1791_179153


namespace NUMINAMATH_CALUDE_new_car_cost_proof_l1791_179143

/-- The monthly cost of renting a car -/
def rental_cost : ℕ := 20

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total difference in cost over a year between renting and buying -/
def total_difference : ℕ := 120

/-- The monthly cost of the new car -/
def new_car_cost : ℕ := 30

theorem new_car_cost_proof : 
  new_car_cost * months_in_year - rental_cost * months_in_year = total_difference := by
  sorry

end NUMINAMATH_CALUDE_new_car_cost_proof_l1791_179143


namespace NUMINAMATH_CALUDE_cathys_final_balance_l1791_179185

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_final_balance :
  cathys_money 12 25 = 87 :=
by sorry

end NUMINAMATH_CALUDE_cathys_final_balance_l1791_179185


namespace NUMINAMATH_CALUDE_min_value_problem_l1791_179135

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∀ x y, x > 0 ∧ y > 1 ∧ x + y = 2 → (4 / a + 1 / (b - 1) ≤ 4 / x + 1 / (y - 1)) ∧
  (∃ x y, x > 0 ∧ y > 1 ∧ x + y = 2 ∧ 4 / x + 1 / (y - 1) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1791_179135


namespace NUMINAMATH_CALUDE_race_distance_l1791_179179

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 20 →
  b_time = 25 →
  beat_distance = 26 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 104 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l1791_179179


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l1791_179159

theorem unique_digit_divisibility : 
  ∃! n : ℕ, 0 < n ∧ n ≤ 9 ∧ 
  100 ≤ 25 * n ∧ 25 * n ≤ 999 ∧ 
  (25 * n) % n = 0 ∧ (25 * n) % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l1791_179159


namespace NUMINAMATH_CALUDE_gas_tank_cost_l1791_179148

theorem gas_tank_cost (initial_fullness : ℚ) (after_adding_fullness : ℚ) 
  (added_amount : ℚ) (gas_price : ℚ) : 
  initial_fullness = 1/8 →
  after_adding_fullness = 3/4 →
  added_amount = 30 →
  gas_price = 138/100 →
  (1 - after_adding_fullness) * 
    (added_amount / (after_adding_fullness - initial_fullness)) * 
    gas_price = 1656/100 := by
  sorry

#eval (1 : ℚ) - 3/4  -- Expected: 1/4
#eval 30 / (3/4 - 1/8)  -- Expected: 48
#eval 1/4 * 48  -- Expected: 12
#eval 12 * 138/100  -- Expected: 16.56

end NUMINAMATH_CALUDE_gas_tank_cost_l1791_179148


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1791_179136

theorem imaginary_part_of_complex_number :
  let z : ℂ := 3 - 2 * I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1791_179136


namespace NUMINAMATH_CALUDE_expression_evaluation_l1791_179131

theorem expression_evaluation (x y z : ℚ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  (1/y + 1/z) / (1/x) = 35/12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1791_179131


namespace NUMINAMATH_CALUDE_spheres_radius_l1791_179103

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The radius of each sphere -/
  radius : ℝ
  /-- The number of spheres is 8 -/
  num_spheres : Nat
  num_spheres_eq : num_spheres = 8
  /-- The cube is a unit cube -/
  cube_edge : ℝ
  cube_edge_eq : cube_edge = 1
  /-- Each sphere touches three adjacent spheres -/
  touches_adjacent : True
  /-- Spheres are inscribed in trihedral angles -/
  inscribed_in_angles : True

/-- The radius of spheres in the specific configuration is 1/4 -/
theorem spheres_radius (config : SpheresInCube) : config.radius = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_l1791_179103


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1791_179140

def largest_prime_factor (n : ℕ) : ℕ := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem sum_largest_smallest_prime_factors_546 :
  largest_prime_factor 546 + smallest_prime_factor 546 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1791_179140


namespace NUMINAMATH_CALUDE_fixed_point_of_arcsin_function_l1791_179102

theorem fixed_point_of_arcsin_function (m : ℝ) :
  ∃ (P : ℝ × ℝ), P = (0, -1) ∧ ∀ x : ℝ, m * Real.arcsin x - 1 = P.2 ↔ x = P.1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_arcsin_function_l1791_179102


namespace NUMINAMATH_CALUDE_rhombus_height_is_half_side_l1791_179137

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  side_positive : 0 < side
  diag1_positive : 0 < diag1
  diag2_positive : 0 < diag2
  geometric_mean : side ^ 2 = diag1 * diag2

/-- The height of a rhombus with side s that is the geometric mean of its diagonals is s/2 -/
theorem rhombus_height_is_half_side (r : Rhombus) : 
  r.side / 2 = (r.diag1 * r.diag2) / (4 * r.side) := by
  sorry

#check rhombus_height_is_half_side

end NUMINAMATH_CALUDE_rhombus_height_is_half_side_l1791_179137


namespace NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l1791_179141

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_prime_arithmetic_sequence (p q r s : ℕ) : 
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 4 →
  r = q + 4 →
  s = r + 4 →
  ones_digit p = 9 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l1791_179141


namespace NUMINAMATH_CALUDE_train_length_l1791_179113

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1791_179113


namespace NUMINAMATH_CALUDE_refrigerator_price_l1791_179125

theorem refrigerator_price (P : ℝ) 
  (selling_price : P + 0.1 * P = 23100)
  (discount : ℝ := 0.2)
  (transport_cost : ℝ := 125)
  (installation_cost : ℝ := 250) :
  P * (1 - discount) + transport_cost + installation_cost = 17175 := by
sorry

end NUMINAMATH_CALUDE_refrigerator_price_l1791_179125


namespace NUMINAMATH_CALUDE_simplify_expression_l1791_179115

theorem simplify_expression (x : ℝ) : 3*x + 2*x^2 + 5*x - x^2 + 7 = x^2 + 8*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1791_179115


namespace NUMINAMATH_CALUDE_least_n_multiple_of_1000_l1791_179121

theorem least_n_multiple_of_1000 : ∃ (n : ℕ), n > 0 ∧ n = 797 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1000 ∣ (2^m + 5^m - m))) ∧ 
  (1000 ∣ (2^n + 5^n - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_n_multiple_of_1000_l1791_179121


namespace NUMINAMATH_CALUDE_train_length_proof_l1791_179180

/-- The length of a train that passes a pole in 15 seconds and a 100-meter platform in 40 seconds -/
def train_length : ℝ := 60

theorem train_length_proof (t : ℝ) (h1 : t > 0) :
  (t / 15 = (t + 100) / 40) → t = train_length :=
by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l1791_179180


namespace NUMINAMATH_CALUDE_flashlight_distance_ratio_l1791_179122

/-- Proves that the ratio of Freddie's flashlight distance to Veronica's is 3:1 --/
theorem flashlight_distance_ratio :
  ∀ (V F : ℕ),
  V = 1000 →
  F > V →
  ∃ (D : ℕ), D = 5 * F - 2000 →
  D = V + 12000 →
  F / V = 3 :=
by sorry

end NUMINAMATH_CALUDE_flashlight_distance_ratio_l1791_179122


namespace NUMINAMATH_CALUDE_max_distance_to_point_l1791_179128

theorem max_distance_to_point (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (w : ℂ), Complex.abs w = 1 ∧ Complex.abs (w - (1 + Complex.I)) = Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_point_l1791_179128


namespace NUMINAMATH_CALUDE_nancy_finished_problems_l1791_179184

/-- Given that Nancy had 101 homework problems initially, still has 6 pages of problems to do,
    and each page has 9 problems, prove that she finished 47 problems. -/
theorem nancy_finished_problems (total_problems : ℕ) (pages_left : ℕ) (problems_per_page : ℕ)
    (h1 : total_problems = 101)
    (h2 : pages_left = 6)
    (h3 : problems_per_page = 9) :
    total_problems - (pages_left * problems_per_page) = 47 := by
  sorry


end NUMINAMATH_CALUDE_nancy_finished_problems_l1791_179184


namespace NUMINAMATH_CALUDE_product_of_distinct_non_trivial_primes_last_digit_l1791_179106

def is_non_trivial_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def last_digit (n : ℕ) : ℕ :=
  n % 10

theorem product_of_distinct_non_trivial_primes_last_digit 
  (p q : ℕ) (hp : is_non_trivial_prime p) (hq : is_non_trivial_prime q) (hpq : p ≠ q) :
  ∃ d : ℕ, d ∈ [1, 3, 7, 9] ∧ last_digit (p * q) = d :=
sorry

end NUMINAMATH_CALUDE_product_of_distinct_non_trivial_primes_last_digit_l1791_179106


namespace NUMINAMATH_CALUDE_allocation_theorem_l1791_179178

/-- The number of ways to allocate employees to departments -/
def allocation_count (total_employees : ℕ) (num_departments : ℕ) : ℕ :=
  sorry

/-- Two employees are considered as one unit -/
def combined_employees : ℕ := 4

/-- Number of ways to distribute combined employees into departments -/
def distribution_ways : ℕ := sorry

/-- Number of ways to assign groups to departments -/
def assignment_ways : ℕ := sorry

theorem allocation_theorem :
  allocation_count 5 3 = distribution_ways * assignment_ways ∧
  distribution_ways = 6 ∧
  assignment_ways = 6 ∧
  allocation_count 5 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_allocation_theorem_l1791_179178


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l1791_179152

/-- Represents a triangular grid figure made of toothpicks -/
structure TriangularGrid where
  total_toothpicks : ℕ
  total_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : TriangularGrid) : ℕ :=
  grid.horizontal_toothpicks

/-- Theorem: For a specific triangular grid, the minimum number of toothpicks 
    to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (grid : TriangularGrid) 
    (h1 : grid.total_toothpicks = 40)
    (h2 : grid.total_triangles > 35)
    (h3 : grid.horizontal_toothpicks = 15) : 
  min_toothpicks_to_remove grid = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l1791_179152


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1791_179176

-- Define the quadratic equation
def quadratic (x p : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition that the roots differ by 2
def roots_differ_by_two (p : ℤ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ quadratic x p = 0 ∧ quadratic y p = 0 ∧ |x - y| = 2

-- The theorem to prove
theorem quadratic_roots_difference (p : ℤ) :
  roots_differ_by_two p → p = 7 ∨ p = -7 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_difference_l1791_179176


namespace NUMINAMATH_CALUDE_three_digit_equation_solutions_l1791_179101

theorem three_digit_equation_solutions :
  ∀ x y z : ℕ,
  (100 ≤ x ∧ x ≤ 999) ∧
  (100 ≤ y ∧ y ≤ 999) ∧
  (100 ≤ z ∧ z ≤ 999) ∧
  (17 * x + 15 * y - 28 * z = 61) ∧
  (19 * x - 25 * y + 12 * z = 31) →
  ((x = 265 ∧ y = 372 ∧ z = 358) ∨
   (x = 525 ∧ y = 740 ∧ z = 713)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_equation_solutions_l1791_179101


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1791_179171

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given lengths satisfy the triangle inequality for external diagonals -/
def satisfies_triangle_inequality (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.a^2 + d.c^2 > d.b^2

/-- Theorem stating that {5, 6, 8} cannot be the lengths of external diagonals of a right regular prism -/
theorem invalid_external_diagonals : 
  ¬(satisfies_triangle_inequality ⟨5, 6, 8⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1791_179171


namespace NUMINAMATH_CALUDE_attic_items_count_l1791_179111

theorem attic_items_count (total : ℝ) (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (junk_count : ℝ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  junk_count = 28 →
  junk_percent * total = junk_count →
  useful_percent * total = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_attic_items_count_l1791_179111


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l1791_179109

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16246775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l1791_179109


namespace NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1791_179170

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (9 * x)^18 = (18 * x)^9 ↔ x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1791_179170


namespace NUMINAMATH_CALUDE_gym_attendance_l1791_179151

theorem gym_attendance (initial : ℕ) 
  (h1 : initial + 5 - 2 = 19) : initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_l1791_179151


namespace NUMINAMATH_CALUDE_longest_working_secretary_time_l1791_179172

/-- Proves that given three secretaries whose working times are in the ratio of 2:3:5 
    and who worked a combined total of 110 hours, the secretary who worked the longest 
    spent 55 hours on the project. -/
theorem longest_working_secretary_time (a b c : ℕ) : 
  a + b + c = 110 →
  2 * a = 3 * b →
  2 * a = 5 * c →
  c = 55 := by
  sorry

end NUMINAMATH_CALUDE_longest_working_secretary_time_l1791_179172
