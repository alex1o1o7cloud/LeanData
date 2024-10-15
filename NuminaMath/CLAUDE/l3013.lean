import Mathlib

namespace NUMINAMATH_CALUDE_wedge_volume_l3013_301305

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h : ℝ) (θ : ℝ) : 
  d = 20 → θ = 30 → (π * (d/2)^2 * h * θ) / 360 = (500/3) * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3013_301305


namespace NUMINAMATH_CALUDE_hyperbola_intersection_slopes_product_l3013_301327

/-- Hyperbola C with asymptotic line equation y = ±√3x and point P(2,3) on it -/
structure Hyperbola :=
  (asymptote : ℝ → ℝ)
  (point : ℝ × ℝ)
  (h_asymptote : ∀ x, asymptote x = Real.sqrt 3 * x ∨ asymptote x = -Real.sqrt 3 * x)
  (h_point : point = (2, 3))

/-- Line l: y = kx + m -/
structure Line :=
  (k m : ℝ)

/-- Intersection points A and B of line l with hyperbola C -/
structure Intersection :=
  (A B : ℝ × ℝ)
  (k₁ k₂ : ℝ)

/-- The theorem to be proved -/
theorem hyperbola_intersection_slopes_product
  (C : Hyperbola) (l : Line) (I : Intersection) :
  ∃ (k m : ℝ), l.k = -3/2 ∧ I.k₁ * I.k₂ = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_slopes_product_l3013_301327


namespace NUMINAMATH_CALUDE_vowel_count_l3013_301317

theorem vowel_count (num_vowels : ℕ) (total_written : ℕ) : 
  num_vowels = 5 → total_written = 15 → (total_written / num_vowels : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_vowel_count_l3013_301317


namespace NUMINAMATH_CALUDE_anns_total_blocks_l3013_301359

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem: Ann's total number of blocks after finding more -/
theorem anns_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end NUMINAMATH_CALUDE_anns_total_blocks_l3013_301359


namespace NUMINAMATH_CALUDE_haley_concert_spending_l3013_301364

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l3013_301364


namespace NUMINAMATH_CALUDE_tangent_intersection_theorem_l3013_301393

/-- The x-coordinate of the point where a line tangent to two circles intersects the x-axis -/
def tangent_intersection_x : ℝ := 4.5

/-- The radius of the first circle -/
def r1 : ℝ := 3

/-- The radius of the second circle -/
def r2 : ℝ := 5

/-- The x-coordinate of the center of the second circle -/
def c2_x : ℝ := 12

theorem tangent_intersection_theorem :
  let x := tangent_intersection_x
  x > 0 ∧ 
  x / (c2_x - x) = r1 / r2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_theorem_l3013_301393


namespace NUMINAMATH_CALUDE_quadratic_function_equal_values_l3013_301357

theorem quadratic_function_equal_values (a m n : ℝ) (h1 : a ≠ 0) (h2 : m ≠ n) :
  (a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_equal_values_l3013_301357


namespace NUMINAMATH_CALUDE_triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l3013_301348

/- Define the necessary types and structures -/
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Point where
  x : ℝ
  y : ℝ

/- Define the given information -/
def nagel_point (t : Triangle) : Point := sorry
def altitude_foot (t : Triangle) (v : Point) : Point := sorry

/- State the theorem -/
theorem triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot 
  (N : Point) (B : Point) (E : Point) :
  ∃! (t : Triangle), 
    B = t.B ∧ 
    N = nagel_point t ∧ 
    E = altitude_foot t B := by
  sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_from_nagel_point_vertex_and_altitude_foot_l3013_301348


namespace NUMINAMATH_CALUDE_train_passing_time_l3013_301316

theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 350)
  (h2 : length2 = 450)
  (h3 : speed1 = 63 * 1000 / 3600)
  (h4 : speed2 = 81 * 1000 / 3600)
  (h5 : speed2 > speed1) :
  (length1 + length2) / (speed2 - speed1) = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3013_301316


namespace NUMINAMATH_CALUDE_distance_to_origin_l3013_301323

/-- The distance from the point corresponding to the complex number 2i/(1-i) to the origin in the complex plane is √2. -/
theorem distance_to_origin : Complex.abs (2 * Complex.I / (1 - Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3013_301323


namespace NUMINAMATH_CALUDE_range_of_expression_l3013_301325

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  ∃ (x : ℝ), 0 < x ∧ x < 8 ∧ x = (a - b) * c^2 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l3013_301325


namespace NUMINAMATH_CALUDE_two_x_minus_y_value_l3013_301334

theorem two_x_minus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  2 * x - y = 4 ∨ 2 * x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_two_x_minus_y_value_l3013_301334


namespace NUMINAMATH_CALUDE_square_side_length_l3013_301314

theorem square_side_length (x : ℝ) (h : x > 0) : 
  x^2 = x * 2 / 2 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3013_301314


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3013_301307

theorem sufficient_not_necessary (x : ℝ) :
  (x > (1/2 : ℝ) → 2*x^2 + x - 1 > 0) ∧
  ¬(2*x^2 + x - 1 > 0 → x > (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3013_301307


namespace NUMINAMATH_CALUDE_a_10_equals_21_l3013_301373

def arithmetic_sequence (b : ℕ+ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ+, b (n + 1) - b n = d

theorem a_10_equals_21
  (a : ℕ+ → ℚ)
  (b : ℕ+ → ℚ)
  (h1 : a 1 = 3)
  (h2 : arithmetic_sequence b)
  (h3 : ∀ n : ℕ+, b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 10 = 21 := by
sorry

end NUMINAMATH_CALUDE_a_10_equals_21_l3013_301373


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l3013_301337

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 180 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ bc : BookCollection, is_valid_collection bc → bc.hardcover_nonfiction = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l3013_301337


namespace NUMINAMATH_CALUDE_principal_calculation_l3013_301343

theorem principal_calculation (P r : ℝ) : 
  P * r * 2 = 10200 →
  P * ((1 + r)^2 - 1) = 11730 →
  P = 17000 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l3013_301343


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3013_301353

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 6)^2 + (z - 8)^2 = 0 → 2*x + 2*y + 2*z = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3013_301353


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_l3013_301396

theorem no_perfect_square_sum (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) :
  ¬ ∃ (a : ℤ), x + y + z = a^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_l3013_301396


namespace NUMINAMATH_CALUDE_problem_solution_l3013_301313

theorem problem_solution : 2^(0^(1^9)) + ((2^0)^1)^9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3013_301313


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l3013_301392

/-- The number of handshakes in a gathering of couples -/
def num_handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands
    with everyone except their spouse and one other person, 
    the total number of handshakes is 54. -/
theorem six_couples_handshakes :
  num_handshakes 6 = 54 := by
  sorry

#eval num_handshakes 6  -- Should output 54

end NUMINAMATH_CALUDE_six_couples_handshakes_l3013_301392


namespace NUMINAMATH_CALUDE_number_added_to_x_l3013_301319

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 5 → x = 4 → some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_x_l3013_301319


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3013_301339

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → 1/x' + 1/y' ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3013_301339


namespace NUMINAMATH_CALUDE_gnome_with_shoes_weighs_34_l3013_301338

/-- The weight of a gnome without shoes -/
def gnome_weight : ℝ := sorry

/-- The weight of a gnome with shoes -/
def gnome_with_shoes_weight : ℝ := sorry

/-- The difference in weight between a gnome with shoes and without shoes -/
def shoe_weight_difference : ℝ := 2

/-- The total weight of five gnomes with shoes and five gnomes without shoes -/
def total_weight : ℝ := 330

/-- Theorem stating that a gnome with shoes weighs 34 kg -/
theorem gnome_with_shoes_weighs_34 :
  gnome_with_shoes_weight = 34 :=
by
  sorry

/-- Axiom: A gnome with shoes weighs 2 kg more than a gnome without shoes -/
axiom shoe_weight_relation :
  gnome_with_shoes_weight = gnome_weight + shoe_weight_difference

/-- Axiom: The total weight of five gnomes with shoes and five gnomes without shoes is 330 kg -/
axiom total_weight_relation :
  5 * gnome_with_shoes_weight + 5 * gnome_weight = total_weight

end NUMINAMATH_CALUDE_gnome_with_shoes_weighs_34_l3013_301338


namespace NUMINAMATH_CALUDE_solution_characterization_l3013_301365

def divides (x y : ℤ) : Prop := ∃ k : ℤ, y = k * x

def is_solution (a b : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ 
  divides (2 * a + 1) (3 * b - 1) ∧
  divides (2 * b + 1) (3 * a - 1)

theorem solution_characterization :
  ∀ a b : ℕ, is_solution a b ↔ ((a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l3013_301365


namespace NUMINAMATH_CALUDE_exists_78_lines_1992_intersections_l3013_301380

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ

/-- Theorem: There exists a configuration of 78 lines with exactly 1992 intersection points -/
theorem exists_78_lines_1992_intersections :
  ∃ (config : LineConfiguration), config.num_lines = 78 ∧ config.num_intersections = 1992 :=
sorry

end NUMINAMATH_CALUDE_exists_78_lines_1992_intersections_l3013_301380


namespace NUMINAMATH_CALUDE_custom_deck_probability_l3013_301326

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (new_ranks : ℕ)

/-- The probability of drawing a specific type of card -/
def draw_probability (d : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / d.total_cards

/-- Our specific deck configuration -/
def custom_deck : Deck :=
  { total_cards := 60
  , ranks := 15
  , suits := 4
  , cards_per_suit := 15
  , new_ranks := 2 }

theorem custom_deck_probability :
  let d := custom_deck
  let diamond_cards := d.cards_per_suit
  let new_rank_cards := d.new_ranks * d.suits
  let favorable_cards := diamond_cards + new_rank_cards - d.new_ranks
  draw_probability d favorable_cards = 7 / 20 := by
  sorry


end NUMINAMATH_CALUDE_custom_deck_probability_l3013_301326


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3013_301346

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  k : ℝ
  m : ℝ

-- Define the properties of the ellipse
def is_valid_ellipse (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b^2 = 3 ∧ C.a > C.b ∧ C.b > 0

-- Define the intersection of line and ellipse
def line_intersects_ellipse (l : Line) (C : Ellipse) : Prop :=
  ∃ x y, (x^2 / C.a^2) + (y^2 / C.b^2) = 1 ∧ y = l.k * x + l.m

-- Define the condition for the circle passing through origin
def circle_passes_through_origin (l : Line) (C : Ellipse) : Prop :=
  ∃ x₁ y₁ x₂ y₂, 
    line_intersects_ellipse l C ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m

-- Main theorem
theorem ellipse_and_line_properties (C : Ellipse) (l : Line) :
  is_valid_ellipse C →
  circle_passes_through_origin l C →
  (C.a^2 = 4 ∧ C.b^2 = 3) ∧
  (l.m < -2 * Real.sqrt 21 / 7 ∨ l.m > 2 * Real.sqrt 21 / 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3013_301346


namespace NUMINAMATH_CALUDE_system_one_solution_l3013_301333

theorem system_one_solution (x y : ℝ) : 
  x + 3 * y = 3 ∧ x - y = 1 → x = (3 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_system_one_solution_l3013_301333


namespace NUMINAMATH_CALUDE_line_through_point_l3013_301350

theorem line_through_point (k : ℝ) : (2 * k * 3 - 1 = 5) ↔ (k = 1) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l3013_301350


namespace NUMINAMATH_CALUDE_chord_distance_l3013_301389

/-- Given a circle intersected by three equally spaced parallel lines resulting in chords of lengths 38, 38, and 34, the distance between two adjacent parallel chords is 6. -/
theorem chord_distance (r : ℝ) (d : ℝ) : 
  d > 0 ∧ 
  r^2 = d^2 + 19^2 ∧ 
  r^2 = (3*d)^2 + 17^2 →
  2*d = 6 :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_l3013_301389


namespace NUMINAMATH_CALUDE_cube_root_64_equals_2_power_m_l3013_301397

theorem cube_root_64_equals_2_power_m (m : ℝ) : (64 : ℝ)^(1/3) = 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_64_equals_2_power_m_l3013_301397


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3013_301361

theorem roots_of_polynomial : ∀ x : ℝ,
  (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) →
  8 * x^5 - 45 * x^4 + 84 * x^3 - 84 * x^2 + 45 * x - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3013_301361


namespace NUMINAMATH_CALUDE_angle_equality_l3013_301303

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sqrt 2 * Real.sin (π/6) = Real.cos θ - Real.sin θ) : 
  θ = π/12 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3013_301303


namespace NUMINAMATH_CALUDE_share_calculation_l3013_301318

theorem share_calculation (total A B C : ℝ) 
  (h1 : total = 1800)
  (h2 : A = (2/5) * (B + C))
  (h3 : B = (1/5) * (A + C))
  (h4 : A + B + C = total) :
  A = 3600/7 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l3013_301318


namespace NUMINAMATH_CALUDE_triangle_theorem_l3013_301315

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : A > 0 ∧ B > 0 ∧ C > 0
  h_angle_sum : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Define the main theorem
theorem triangle_theorem (t : Triangle) :
  (3 * t.a^2 - 4 * Real.sqrt 3 * t.S = 3 * t.b^2 + 3 * t.c^2) →
  (t.A = 2 * π / 3) ∧
  (t.a = 3 → 6 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3013_301315


namespace NUMINAMATH_CALUDE_function_decomposition_l3013_301363

/-- A non-negative function defined on [-3, 3] -/
def NonNegativeFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 0}

/-- An even function defined on [-3, 3] -/
def EvenFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = f (-x)}

/-- An odd function defined on [-3, 3] -/
def OddFunction := {f : ℝ → ℝ // ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x = -f (-x)}

theorem function_decomposition
  (f : EvenFunction) (g : OddFunction)
  (h : ∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x + g.val x ≥ 2007 * x * Real.sqrt (9 - x^2) + x^2006) :
  ∃ p : NonNegativeFunction,
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f.val x = x^2006 + (p.val x + p.val (-x)) / 2) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, g.val x = 2007 * x * Real.sqrt (9 - x^2) + (p.val x - p.val (-x)) / 2) :=
sorry

end NUMINAMATH_CALUDE_function_decomposition_l3013_301363


namespace NUMINAMATH_CALUDE_sally_earnings_l3013_301321

def earnings_per_house : ℕ := 25
def houses_cleaned : ℕ := 96

theorem sally_earnings :
  (earnings_per_house * houses_cleaned) / 12 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sally_earnings_l3013_301321


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3013_301322

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a < 9) (hb : 0 < b ∧ b < 9) :
  ∃ (m : ℤ), m = -5 ∧ ∀ (x y : ℕ), (0 < x ∧ x < 9) → (0 < y ∧ y < 9) → m ≤ (3 * x^2 - x * y : ℤ) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3013_301322


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l3013_301320

theorem no_prime_satisfies_equation : 
  ¬ ∃ (q : ℕ), Nat.Prime q ∧ 
  (1 * q^3 + 0 * q^2 + 1 * q + 2) + 
  (3 * q^2 + 0 * q + 7) + 
  (1 * q^2 + 1 * q + 4) + 
  (1 * q^2 + 2 * q + 6) + 
  7 = 
  (1 * q^2 + 4 * q + 3) + 
  (2 * q^2 + 7 * q + 2) + 
  (3 * q^2 + 6 * q + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l3013_301320


namespace NUMINAMATH_CALUDE_max_collisions_l3013_301354

/-- Represents an ant with a position and velocity -/
structure Ant where
  position : ℝ
  velocity : ℝ

/-- The configuration of n ants on a line -/
def AntConfiguration (n : ℕ) := Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (config : AntConfiguration n) : Prop := sorry

/-- The number of collisions that occur in a given configuration -/
def NumberOfCollisions (config : AntConfiguration n) : ℕ := sorry

/-- Theorem stating the maximum number of collisions for n ants -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (config : AntConfiguration n),
    HasFiniteCollisions config ∧
    NumberOfCollisions config = n * (n - 1) / 2 ∧
    ∀ (other_config : AntConfiguration n),
      HasFiniteCollisions other_config →
      NumberOfCollisions other_config ≤ n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_collisions_l3013_301354


namespace NUMINAMATH_CALUDE_min_abs_diff_bound_l3013_301391

theorem min_abs_diff_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  min (min (|a - b|) (|b - c|)) (|c - a|) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_bound_l3013_301391


namespace NUMINAMATH_CALUDE_prop_values_l3013_301309

theorem prop_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prop_values_l3013_301309


namespace NUMINAMATH_CALUDE_circle_radius_in_triangle_l3013_301360

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Determines if a circle is tangent to two sides of a triangle -/
def is_tangent_to_sides (c : Circle) (t : Triangle) : Prop := sorry

/-- Determines if a circle lies entirely within a triangle -/
def lies_within_triangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem statement -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) : 
  t.a = 120 → t.b = 120 → t.c = 70 →
  r.radius = 20 →
  is_tangent_to_sides r t →
  are_externally_tangent r s →
  is_tangent_to_sides s t →
  lies_within_triangle s t →
  s.radius = 54 - 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_triangle_l3013_301360


namespace NUMINAMATH_CALUDE_min_value_of_f_l3013_301301

/-- The function f(x) = 3/x + 1/(1-3x) has a minimum value of 16 on the interval (0, 1/3) -/
theorem min_value_of_f (x : ℝ) (hx : 0 < x ∧ x < 1/3) : 3/x + 1/(1-3*x) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3013_301301


namespace NUMINAMATH_CALUDE_unique_number_l3013_301386

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  digit_sum n = 12 ∧ 
  reverse_digits (n + 36) = n :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l3013_301386


namespace NUMINAMATH_CALUDE_investment_percentage_l3013_301371

/-- Proves that given the investment conditions, the unknown percentage is 4% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (unknown_rate : ℝ) 
  (total_interest : ℝ) (amount_at_unknown_rate : ℝ) :
  total_investment = 17000 →
  known_rate = 18 →
  total_interest = 1380 →
  amount_at_unknown_rate = 12000 →
  (amount_at_unknown_rate * unknown_rate / 100 + 
   (total_investment - amount_at_unknown_rate) * known_rate / 100 = total_interest) →
  unknown_rate = 4 := by
sorry


end NUMINAMATH_CALUDE_investment_percentage_l3013_301371


namespace NUMINAMATH_CALUDE_correct_operation_result_l3013_301381

theorem correct_operation_result (x : ℝ) : 
  ((x / 8) ^ 2 = 49) → ((x * 8) * 2 = 896) := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_result_l3013_301381


namespace NUMINAMATH_CALUDE_watch_selling_prices_l3013_301394

/-- Calculates the selling price given the cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

theorem watch_selling_prices :
  let watch1CP : ℚ := 1400
  let watch1Profit : ℚ := 5
  let watch2CP : ℚ := 1800
  let watch2Profit : ℚ := 8
  let watch3CP : ℚ := 2500
  let watch3Profit : ℚ := 12
  (sellingPrice watch1CP watch1Profit = 1470) ∧
  (sellingPrice watch2CP watch2Profit = 1944) ∧
  (sellingPrice watch3CP watch3Profit = 2800) :=
by sorry

end NUMINAMATH_CALUDE_watch_selling_prices_l3013_301394


namespace NUMINAMATH_CALUDE_log_sum_equality_l3013_301384

-- Define the problem
theorem log_sum_equality : Real.log 50 + Real.log 20 + Real.log 4 = 3.60206 := by
  sorry

#check log_sum_equality

end NUMINAMATH_CALUDE_log_sum_equality_l3013_301384


namespace NUMINAMATH_CALUDE_special_bet_cost_l3013_301347

def lottery_numbers : ℕ := 36
def numbers_per_bet : ℕ := 7
def cost_per_bet : ℕ := 2

def consecutive_numbers_01_to_10 : ℕ := 3
def consecutive_numbers_11_to_20 : ℕ := 2
def single_number_21_to_30 : ℕ := 1
def single_number_31_to_36 : ℕ := 1

def ways_01_to_10 : ℕ := 10 - consecutive_numbers_01_to_10 + 1
def ways_11_to_20 : ℕ := 10 - consecutive_numbers_11_to_20 + 1
def ways_21_to_30 : ℕ := 10
def ways_31_to_36 : ℕ := 6

theorem special_bet_cost (total_combinations : ℕ) (total_cost : ℕ) :
  total_combinations = ways_01_to_10 * ways_11_to_20 * ways_21_to_30 * ways_31_to_36 ∧
  total_cost = total_combinations * cost_per_bet ∧
  total_cost = 8640 := by
  sorry

end NUMINAMATH_CALUDE_special_bet_cost_l3013_301347


namespace NUMINAMATH_CALUDE_subset_P_l3013_301399

-- Define the set P
def P : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l3013_301399


namespace NUMINAMATH_CALUDE_no_real_d_for_two_distinct_roots_l3013_301341

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that there are no real values of d such that g(g(x)) has exactly 2 distinct real roots -/
theorem no_real_d_for_two_distinct_roots :
  ¬ ∃ d : ℝ, ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ :=
sorry

end NUMINAMATH_CALUDE_no_real_d_for_two_distinct_roots_l3013_301341


namespace NUMINAMATH_CALUDE_five_mondays_in_september_l3013_301342

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  numDays : Nat

def August : Month := sorry
def September : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- Determines the day of the week for the first day of the next month -/
def nextMonthFirstDay (m : Month) : DayOfWeek := sorry

theorem five_mondays_in_september 
  (h1 : August.numDays = 31)
  (h2 : September.numDays = 30)
  (h3 : countDayOccurrences August DayOfWeek.Sunday = 5) :
  countDayOccurrences September DayOfWeek.Monday = 5 := by sorry

end NUMINAMATH_CALUDE_five_mondays_in_september_l3013_301342


namespace NUMINAMATH_CALUDE_inequality_always_holds_l3013_301306

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l3013_301306


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3013_301377

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : (a^6 + b^6) / (a + b)^6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3013_301377


namespace NUMINAMATH_CALUDE_smallest_and_largest_with_digit_sum_17_l3013_301375

def digit_sum (n : ℕ) : ℕ := sorry

def all_digits_different (n : ℕ) : Prop := sorry

theorem smallest_and_largest_with_digit_sum_17 :
  ∃ (smallest largest : ℕ),
    (∀ n : ℕ, digit_sum n = 17 → all_digits_different n →
      smallest ≤ n ∧ n ≤ largest) ∧
    digit_sum smallest = 17 ∧
    all_digits_different smallest ∧
    digit_sum largest = 17 ∧
    all_digits_different largest ∧
    smallest = 89 ∧
    largest = 743210 :=
sorry

end NUMINAMATH_CALUDE_smallest_and_largest_with_digit_sum_17_l3013_301375


namespace NUMINAMATH_CALUDE_equation_system_solutions_l3013_301331

theorem equation_system_solutions :
  ∀ (x y z : ℝ),
  (x = (2 * z^2) / (1 + z^2)) ∧
  (y = (2 * x^2) / (1 + x^2)) ∧
  (z = (2 * y^2) / (1 + y^2)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l3013_301331


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3013_301335

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z ≠ 0) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3013_301335


namespace NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l3013_301308

theorem comic_book_stacking_arrangements :
  let hulk_comics : ℕ := 8
  let ironman_comics : ℕ := 7
  let wolverine_comics : ℕ := 6
  let total_comics : ℕ := hulk_comics + ironman_comics + wolverine_comics
  let arrange_hulk : ℕ := Nat.factorial hulk_comics
  let arrange_ironman : ℕ := Nat.factorial ironman_comics
  let arrange_wolverine : ℕ := Nat.factorial wolverine_comics
  let arrange_within_groups : ℕ := arrange_hulk * arrange_ironman * arrange_wolverine
  let arrange_groups : ℕ := Nat.factorial 3
  arrange_within_groups * arrange_groups = 69657088000 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_arrangements_l3013_301308


namespace NUMINAMATH_CALUDE_fencing_theorem_l3013_301324

/-- Represents a rectangular field with given dimensions -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the fencing required for three sides of a rectangular field -/
def fencing_required (field : RectangularField) : ℝ :=
  2 * field.width + field.length

theorem fencing_theorem (field : RectangularField) 
  (h1 : field.area = 600)
  (h2 : field.uncovered_side = 30)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  fencing_required field = 70 := by
  sorry

#check fencing_theorem

end NUMINAMATH_CALUDE_fencing_theorem_l3013_301324


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3013_301330

theorem lcm_gcd_product : Nat.lcm 6 (Nat.lcm 8 12) * Nat.gcd 6 (Nat.gcd 8 12) = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3013_301330


namespace NUMINAMATH_CALUDE_rachel_essay_time_l3013_301358

/-- Calculates the total time spent on an essay in hours -/
def total_essay_time (pages_written : ℕ) (writing_rate : ℚ) (research_time : ℕ) (editing_time : ℕ) : ℚ :=
  let writing_time : ℚ := pages_written * writing_rate
  let total_minutes : ℚ := research_time + writing_time + editing_time
  total_minutes / 60

/-- Theorem: Rachel spends 5 hours completing the essay -/
theorem rachel_essay_time : 
  total_essay_time 6 (30 : ℚ) 45 75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_essay_time_l3013_301358


namespace NUMINAMATH_CALUDE_x_power_ten_plus_inverse_l3013_301311

theorem x_power_ten_plus_inverse (x : ℝ) (h : x + 1/x = 5) : x^10 + 1/x^10 = 6430223 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ten_plus_inverse_l3013_301311


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l3013_301390

theorem cubic_expression_evaluation :
  1001^3 - 1000 * 1001^2 - 1000^2 * 1001 + 1000^3 = 2001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l3013_301390


namespace NUMINAMATH_CALUDE_marble_distribution_l3013_301352

def jasmine_initial : ℕ := 120
def lola_initial : ℕ := 15
def marbles_given : ℕ := 19

theorem marble_distribution :
  (jasmine_initial - marbles_given) = 3 * (lola_initial + marbles_given) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3013_301352


namespace NUMINAMATH_CALUDE_hotdog_ratio_l3013_301378

/-- Represents the number of hotdogs for each person -/
structure Hotdogs where
  ella : ℕ
  emma : ℕ
  luke : ℕ
  hunter : ℕ

/-- Given conditions for the hotdog problem -/
def hotdog_problem (h : Hotdogs) : Prop :=
  h.ella = 2 ∧
  h.emma = 2 ∧
  h.luke = 2 * (h.ella + h.emma) ∧
  h.ella + h.emma + h.luke + h.hunter = 14

/-- Theorem stating the ratio of Hunter's hotdogs to his sisters' total hotdogs -/
theorem hotdog_ratio (h : Hotdogs) (hcond : hotdog_problem h) :
  h.hunter / (h.ella + h.emma) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hotdog_ratio_l3013_301378


namespace NUMINAMATH_CALUDE_max_quotient_value_l3013_301312

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 900 ≤ b ∧ b ≤ 1800) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 900 ≤ y ∧ y ≤ 1800 → x / y ≤ a / b) →
  a / b = 5 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l3013_301312


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3013_301383

/-- An ellipse with center at the origin, foci on the x-axis, 
    minor axis length 8√2, and eccentricity 1/3 -/
structure Ellipse where
  b : ℝ
  e : ℝ
  minor_axis : b = 4 * Real.sqrt 2
  eccentricity : e = 1/3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_theorem (E : Ellipse) (x y : ℝ) :
  ellipse_equation x y := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l3013_301383


namespace NUMINAMATH_CALUDE_stacy_height_proof_l3013_301300

/-- Calculates Stacy's current height given her previous height, James' growth, and the difference between their growth. -/
def stacys_current_height (stacy_previous_height james_growth growth_difference : ℕ) : ℕ :=
  stacy_previous_height + james_growth + growth_difference

/-- Proves that Stacy's current height is 57 inches. -/
theorem stacy_height_proof :
  stacys_current_height 50 1 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_proof_l3013_301300


namespace NUMINAMATH_CALUDE_square_garden_area_l3013_301329

theorem square_garden_area (p : ℝ) (s : ℝ) : 
  p = 28 →                   -- The perimeter is 28 feet
  p = 4 * s →                -- Perimeter of a square is 4 times the side length
  s^2 = p + 21 →             -- Area is equal to perimeter plus 21
  s^2 = 49 :=                -- The area of the garden is 49 square feet
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l3013_301329


namespace NUMINAMATH_CALUDE_mrs_brown_payment_l3013_301376

/-- Calculates the final price after applying multiple discounts --/
def calculate_final_price (base_price : ℝ) (mother_discount : ℝ) (child_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  let price_after_mother := base_price * (1 - mother_discount)
  let price_after_child := price_after_mother * (1 - child_discount)
  price_after_child * (1 - vip_discount)

/-- Theorem stating that Mrs. Brown's final payment amount is $201.10 --/
theorem mrs_brown_payment : 
  let shoes_price : ℝ := 125
  let handbag_price : ℝ := 75
  let scarf_price : ℝ := 45
  let total_price : ℝ := shoes_price + handbag_price + scarf_price
  let mother_discount : ℝ := 0.10
  let child_discount : ℝ := 0.04
  let vip_discount : ℝ := 0.05
  calculate_final_price total_price mother_discount child_discount vip_discount = 201.10 := by
  sorry


end NUMINAMATH_CALUDE_mrs_brown_payment_l3013_301376


namespace NUMINAMATH_CALUDE_f_increasing_interval_l3013_301362

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x

-- State the theorem
theorem f_increasing_interval :
  ∀ x y : ℝ, x ≥ 3 → y > x → f y > f x :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l3013_301362


namespace NUMINAMATH_CALUDE_infinite_fibonacci_divisible_l3013_301370

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For any positive integer N, there are infinitely many Fibonacci numbers divisible by N -/
theorem infinite_fibonacci_divisible (N : ℕ) (hN : N > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, N ∣ fib k := by
  sorry

end NUMINAMATH_CALUDE_infinite_fibonacci_divisible_l3013_301370


namespace NUMINAMATH_CALUDE_pages_left_to_read_l3013_301328

/-- Calculates the number of pages left to be read in a book --/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (daily_reading : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : daily_reading = 20) 
  (h4 : days = 7) :
  total_pages - (pages_read + daily_reading * days) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l3013_301328


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3013_301398

theorem inequality_equivalence (x : ℝ) : -4 * x - 8 > 0 ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3013_301398


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3013_301367

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) : 
  deficient_containers = 3 →
  (15 * total_containers - deficient_containers > 150) →
  (∀ n : ℕ, 15 * n - deficient_containers > 150 → n ≥ total_containers) →
  15 * total_containers - deficient_containers = 162 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3013_301367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3013_301302

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of an arithmetic sequence with first term a₁ and common difference d. -/
def arithmetic_sequence_term (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  ∀ n : ℕ, a n = -2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3013_301302


namespace NUMINAMATH_CALUDE_smallest_m_for_meaningful_sqrt_l3013_301344

theorem smallest_m_for_meaningful_sqrt (m : ℤ) : 
  (∀ k : ℤ, k < m → ¬(2*k + 1 ≥ 0)) → (2*m + 1 ≥ 0) → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_meaningful_sqrt_l3013_301344


namespace NUMINAMATH_CALUDE_johns_drive_speed_l3013_301379

/-- Proves that given the conditions of John's drive, his average speed during the last 40 minutes was 70 mph -/
theorem johns_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed_first_40 : ℝ) (speed_next_40 : ℝ)
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed_first_40 = 50)
  (h4 : speed_next_40 = 60) :
  let time_segment := total_time / 3
  let distance_first_40 := speed_first_40 * time_segment
  let distance_next_40 := speed_next_40 * time_segment
  let distance_last_40 := total_distance - (distance_first_40 + distance_next_40)
  distance_last_40 / time_segment = 70 := by
  sorry

end NUMINAMATH_CALUDE_johns_drive_speed_l3013_301379


namespace NUMINAMATH_CALUDE_soccer_balls_count_initial_balls_count_l3013_301372

/-- The initial number of soccer balls in the bag -/
def initial_balls : ℕ := sorry

/-- The number of additional balls added to the bag -/
def added_balls : ℕ := 18

/-- The final number of balls in the bag -/
def final_balls : ℕ := 24

/-- Theorem stating that the initial number of balls plus the added balls equals the final number of balls -/
theorem soccer_balls_count : initial_balls + added_balls = final_balls := by sorry

/-- Theorem proving that the initial number of balls is 6 -/
theorem initial_balls_count : initial_balls = 6 := by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_initial_balls_count_l3013_301372


namespace NUMINAMATH_CALUDE_simplify_fraction_l3013_301345

theorem simplify_fraction : (15^30) / (45^15) = 5^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3013_301345


namespace NUMINAMATH_CALUDE_equation_solutions_l3013_301304

theorem equation_solutions (x : ℝ) : 
  (8 / (Real.sqrt (x - 9) - 10) + 2 / (Real.sqrt (x - 9) - 5) + 
   9 / (Real.sqrt (x - 9) + 5) + 15 / (Real.sqrt (x - 9) + 10) = 0) ↔ 
  (x = (70/23)^2 + 9 ∨ x = (25/11)^2 + 9 ∨ x = 575/34 + 9) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3013_301304


namespace NUMINAMATH_CALUDE_derivative_bound_l3013_301332

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem derivative_bound
  (h_cont : ContDiff ℝ 3 f)
  (h_pos : ∀ x, f x > 0 ∧ (deriv f) x > 0 ∧ (deriv^[2] f) x > 0 ∧ (deriv^[3] f) x > 0)
  (h_bound : ∀ x, (deriv^[3] f) x ≤ f x) :
  ∀ x, (deriv f) x < 2 * f x :=
sorry

end NUMINAMATH_CALUDE_derivative_bound_l3013_301332


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l3013_301310

/-- Given an angle θ in the Cartesian coordinate system with vertex at the origin,
    initial side on the positive x-axis, and terminal side on the line y = 3x,
    prove that sin 2θ = 3/5 -/
theorem sin_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = 3 * x ∧ x > 0 ∧ y > 0 ∧ (θ = Real.arctan (y / x))) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l3013_301310


namespace NUMINAMATH_CALUDE_min_value_expression_l3013_301388

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  b / a^2 + 4 / b + a / 2 ≥ 2 * Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ b₀ / a₀^2 + 4 / b₀ + a₀ / 2 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3013_301388


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l3013_301374

-- Define the inequality
def inequality (x a : ℝ) : Prop := (x + a) / (x^2 + 4*x + 3) > 0

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-3 < x ∧ x < -1) ∨ x > 2

-- Theorem statement
theorem inequality_solution_implies_a_value :
  (∀ x : ℝ, inequality x a ↔ solution_set x) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l3013_301374


namespace NUMINAMATH_CALUDE_range_of_a_l3013_301385

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B (a : ℝ) : Set ℝ := {x | (2*x - a) / (x + 1) > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (A ⊂ B a ∧ A ≠ B a) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3013_301385


namespace NUMINAMATH_CALUDE_finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l3013_301368

-- Define the number of divisors function
def d (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define highly divisible property
def is_highly_divisible (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → d m < d n

-- Define consecutive highly divisible property
def consecutive_highly_divisible (m n : ℕ) : Prop :=
  is_highly_divisible m ∧ is_highly_divisible n ∧ m < n ∧
  ∀ s : ℕ, m < s → s < n → ¬is_highly_divisible s

-- Theorem for part (a)
theorem finite_consecutive_divisible_pairs :
  {p : ℕ × ℕ | consecutive_highly_divisible p.1 p.2 ∧ p.1 ∣ p.2}.Finite :=
sorry

-- Theorem for part (b)
theorem infinite_highly_divisible_multiples (p : ℕ) (hp : Nat.Prime p) :
  {r : ℕ | is_highly_divisible r ∧ is_highly_divisible (p * r)}.Infinite :=
sorry

end NUMINAMATH_CALUDE_finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l3013_301368


namespace NUMINAMATH_CALUDE_largest_number_l3013_301351

theorem largest_number (S : Set ℝ) (hS : S = {1/2, 0, 1, -9}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3013_301351


namespace NUMINAMATH_CALUDE_no_two_right_angles_l3013_301356

-- Define a triangle as a structure with three angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_is_180 : A + B + C = 180

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : ¬(t.A = 90 ∧ t.B = 90 ∨ t.A = 90 ∧ t.C = 90 ∨ t.B = 90 ∧ t.C = 90) := by
  sorry


end NUMINAMATH_CALUDE_no_two_right_angles_l3013_301356


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_parallel_planes_l3013_301355

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contained_in b β)
  (h3 : parallel α β) :
  line_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_parallel_planes_l3013_301355


namespace NUMINAMATH_CALUDE_only_5_12_13_is_pythagorean_triple_l3013_301387

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem only_5_12_13_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 1 3 5 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_only_5_12_13_is_pythagorean_triple_l3013_301387


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_vector_expression_equality_l3013_301369

-- Part 1
theorem trigonometric_expression_equality :
  Real.cos (25 * Real.pi / 3) + Real.tan (-15 * Real.pi / 4) = 3/2 := by sorry

-- Part 2
theorem vector_expression_equality {n : Type*} [NormedAddCommGroup n] :
  ∀ (a b : n), 2 • (a - b) - (2 • a + b) + 3 • b = 0 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_vector_expression_equality_l3013_301369


namespace NUMINAMATH_CALUDE_x_value_is_three_l3013_301366

theorem x_value_is_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + 6 * x * y = x^3 + x * y^2) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_three_l3013_301366


namespace NUMINAMATH_CALUDE_square_root_of_four_l3013_301382

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3013_301382


namespace NUMINAMATH_CALUDE_car_original_price_l3013_301349

/-- Proves the original price of a car given repair cost, selling price, and profit percentage -/
theorem car_original_price (repair_cost selling_price : ℝ) (profit_percentage : ℝ) :
  repair_cost = 12000 →
  selling_price = 80000 →
  profit_percentage = 40.35 →
  ∃ (original_price : ℝ),
    (selling_price - (original_price + repair_cost)) / original_price * 100 = profit_percentage ∧
    abs (original_price - 48425.44) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_l3013_301349


namespace NUMINAMATH_CALUDE_starting_number_with_20_multiples_of_5_l3013_301340

theorem starting_number_with_20_multiples_of_5 :
  (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) ∧
  (∀ n : ℕ, (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) → n = 10) :=
by sorry

end NUMINAMATH_CALUDE_starting_number_with_20_multiples_of_5_l3013_301340


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3013_301395

theorem min_value_expression (x y : ℝ) : x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 ≥ -16 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + 8*x*Real.sin y - 16*(Real.cos y)^2 = -16 := by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3013_301395


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3013_301336

theorem quadratic_roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 2*m - 1 = 0) → (n^2 - 2*n - 1 = 0) → m^2 + n^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3013_301336
