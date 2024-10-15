import Mathlib

namespace NUMINAMATH_CALUDE_fiona_peeled_22_l146_14608

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  martin_rate : ℕ
  fiona_rate : ℕ
  fiona_join_time : ℕ

/-- Calculates the number of potatoes Fiona peeled -/
def fiona_peeled (scenario : PotatoPeeling) : ℕ :=
  let martin_peeled := scenario.martin_rate * scenario.fiona_join_time
  let remaining := scenario.total_potatoes - martin_peeled
  let combined_rate := scenario.martin_rate + scenario.fiona_rate
  let combined_time := (remaining + combined_rate - 1) / combined_rate -- Ceiling division
  scenario.fiona_rate * combined_time

/-- Theorem stating that Fiona peeled 22 potatoes -/
theorem fiona_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.martin_rate = 4)
  (h3 : scenario.fiona_rate = 6)
  (h4 : scenario.fiona_join_time = 6) :
  fiona_peeled scenario = 22 := by
  sorry

#eval fiona_peeled { total_potatoes := 60, martin_rate := 4, fiona_rate := 6, fiona_join_time := 6 }

end NUMINAMATH_CALUDE_fiona_peeled_22_l146_14608


namespace NUMINAMATH_CALUDE_cubic_root_series_sum_l146_14669

/-- Given a positive real number s satisfying s³ + (1/4)s - 1 = 0,
    the series s² + 2s⁵ + 3s⁸ + 4s¹¹ + ... converges to 16 -/
theorem cubic_root_series_sum (s : ℝ) (hs : 0 < s) (heq : s^3 + (1/4) * s - 1 = 0) :
  ∑' n, (n + 1) * s^(3*n + 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_series_sum_l146_14669


namespace NUMINAMATH_CALUDE_cube_edge_length_equality_l146_14615

theorem cube_edge_length_equality (a : ℝ) : 
  let parallelepiped_volume : ℝ := 2 * 3 * 6
  let parallelepiped_surface_area : ℝ := 2 * (2 * 3 + 3 * 6 + 2 * 6)
  let cube_volume : ℝ := a^3
  let cube_surface_area : ℝ := 6 * a^2
  (parallelepiped_volume / cube_volume = parallelepiped_surface_area / cube_surface_area) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_equality_l146_14615


namespace NUMINAMATH_CALUDE_equation_solutions_l146_14600

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 = 16 ↔ x = 1 ∨ x = 9) ∧
  (∀ x : ℝ, 2*x^2 - 1 = -4*x ↔ x = -1 + Real.sqrt 6 / 2 ∨ x = -1 - Real.sqrt 6 / 2) ∧
  (∀ x : ℝ, 5*x*(x+1) = 2*(x+1) ↔ x = -1 ∨ x = 2/5) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l146_14600


namespace NUMINAMATH_CALUDE_complex_equation_solution_l146_14689

theorem complex_equation_solution (z : ℂ) : (1 + 2 * Complex.I) * z = 3 - Complex.I → z = 1/5 - 7/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l146_14689


namespace NUMINAMATH_CALUDE_isabel_total_songs_l146_14620

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of jazz albums Isabel bought -/
def jazz_albums : ℕ := 4

/-- The number of rock albums Isabel bought -/
def rock_albums : ℕ := 3

/-- The number of songs in each country album -/
def songs_per_country_album : ℕ := 9

/-- The number of songs in each pop album -/
def songs_per_pop_album : ℕ := 9

/-- The number of songs in each jazz album -/
def songs_per_jazz_album : ℕ := 12

/-- The number of songs in each rock album -/
def songs_per_rock_album : ℕ := 14

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 
  country_albums * songs_per_country_album +
  pop_albums * songs_per_pop_album +
  jazz_albums * songs_per_jazz_album +
  rock_albums * songs_per_rock_album

theorem isabel_total_songs : total_songs = 162 := by
  sorry

end NUMINAMATH_CALUDE_isabel_total_songs_l146_14620


namespace NUMINAMATH_CALUDE_zero_point_location_l146_14624

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log (1/2)

-- Define the theorem
theorem zero_point_location 
  (a b c x₀ : ℝ) 
  (h1 : f a * f b * f c < 0)
  (h2 : 0 < a) (h3 : a < b) (h4 : b < c)
  (h5 : f x₀ = 0) : 
  x₀ > a := by
  sorry

end NUMINAMATH_CALUDE_zero_point_location_l146_14624


namespace NUMINAMATH_CALUDE_total_marks_proof_l146_14664

theorem total_marks_proof (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  larry_score = 3 * keith_score →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 := by
sorry

end NUMINAMATH_CALUDE_total_marks_proof_l146_14664


namespace NUMINAMATH_CALUDE_track_length_is_24_l146_14638

/-- Represents a circular ski track -/
structure SkiTrack where
  length : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : SkiTrack) : Prop :=
  track.downhill_speed = 4 * track.uphill_speed ∧
  track.length > 0 ∧
  ∃ (min_distance max_distance : ℝ),
    min_distance = 4 ∧
    max_distance = 13 ∧
    max_distance - min_distance = 9

/-- The theorem to be proved -/
theorem track_length_is_24 (track : SkiTrack) :
  problem_conditions track → track.length = 24 := by
  sorry

end NUMINAMATH_CALUDE_track_length_is_24_l146_14638


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l146_14659

/-- Given a function f(x) = ax^2 / (x+1), prove that if the slope of the tangent line
    at the point (1, f(1)) is 1, then a = 4/3 -/
theorem tangent_slope_implies_a (a : ℝ) :
  let f := fun x : ℝ => (a * x^2) / (x + 1)
  let f' := fun x : ℝ => ((a * x^2 + 2 * a * x) / (x + 1)^2)
  f' 1 = 1 → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l146_14659


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l146_14649

theorem tennis_tournament_matches (n : ℕ) (b : ℕ) (h1 : n = 120) (h2 : b = 40) :
  let total_matches := n - 1
  total_matches = 119 ∧ total_matches % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l146_14649


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l146_14685

/-- Given an ellipse with point P and foci F₁ and F₂, if ∠PF₁F₂ = 60° and |PF₂| = √3|PF₁|,
    then the eccentricity of the ellipse is √3 - 1. -/
theorem ellipse_eccentricity (P F₁ F₂ : ℝ × ℝ) (a c : ℝ) :
  let e := c / a
  let angle_PF₁F₂ := Real.pi / 3  -- 60° in radians
  let dist_PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let dist_PF₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let dist_F₁F₂ := 2 * c
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 4 * a^2 →  -- P is on the ellipse
  dist_F₁F₂^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 →  -- Definition of distance between foci
  Real.cos angle_PF₁F₂ = (dist_PF₁^2 + dist_F₁F₂^2 - dist_PF₂^2) / (2 * dist_PF₁ * dist_F₁F₂) →  -- Cosine rule
  dist_PF₂ = Real.sqrt 3 * dist_PF₁ →
  e = Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l146_14685


namespace NUMINAMATH_CALUDE_stratified_sampling_company_a_l146_14629

def total_representatives : ℕ := 100
def company_a_representatives : ℕ := 40
def company_b_representatives : ℕ := 60
def total_sample_size : ℕ := 10

theorem stratified_sampling_company_a :
  (company_a_representatives * total_sample_size) / total_representatives = 4 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_company_a_l146_14629


namespace NUMINAMATH_CALUDE_cube_corners_equivalence_l146_14631

/-- A corner piece consists of three 1x1x1 cubes -/
def corner_piece : ℕ := 3

/-- The dimensions of the cube -/
def cube_dimension : ℕ := 3

/-- The number of corner pieces -/
def num_corners : ℕ := 9

/-- Theorem: The total number of 1x1x1 cubes in a 3x3x3 cube 
    is equal to the total number of 1x1x1 cubes in 9 corner pieces -/
theorem cube_corners_equivalence : 
  cube_dimension ^ 3 = num_corners * corner_piece := by
  sorry

end NUMINAMATH_CALUDE_cube_corners_equivalence_l146_14631


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l146_14644

-- Define the types for lines and planes
def Line : Type := Real × Real × Real → Prop
def Plane : Type := Real × Real × Real → Prop

-- Define the relations
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : parallel m n) : 
  perpendicular_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l146_14644


namespace NUMINAMATH_CALUDE_problem_solution_l146_14616

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -5) :
  x + x^3 / y^2 + y^3 / x^2 + y = 285 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l146_14616


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l146_14662

/-- Given that N(3,5) is the midpoint of line segment CD and C has coordinates (1,10),
    prove that the sum of the coordinates of point D is 5. -/
theorem sum_coordinates_of_D (C D N : ℝ × ℝ) : 
  C = (1, 10) →
  N = (3, 5) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l146_14662


namespace NUMINAMATH_CALUDE_susan_remaining_spaces_l146_14623

/-- Represents a board game with a given number of spaces and a player's movements. -/
structure BoardGame where
  total_spaces : ℕ
  movements : List ℤ

/-- Calculates the remaining spaces to the end of the game. -/
def remaining_spaces (game : BoardGame) : ℕ :=
  game.total_spaces - (game.movements.sum.toNat)

/-- Susan's board game scenario -/
def susan_game : BoardGame :=
  { total_spaces := 48,
    movements := [8, -3, 6] }

/-- Theorem stating that Susan needs to move 37 spaces to reach the end -/
theorem susan_remaining_spaces :
  remaining_spaces susan_game = 37 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_spaces_l146_14623


namespace NUMINAMATH_CALUDE_mayo_bottle_size_l146_14645

/-- Proves the size of a mayo bottle at a normal store given bulk pricing information -/
theorem mayo_bottle_size 
  (costco_price : ℝ) 
  (normal_store_price : ℝ) 
  (savings : ℝ) 
  (gallon_in_ounces : ℝ) 
  (h1 : costco_price = 8) 
  (h2 : normal_store_price = 3) 
  (h3 : savings = 16) 
  (h4 : gallon_in_ounces = 128) : 
  (gallon_in_ounces / ((savings + costco_price) / normal_store_price)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_mayo_bottle_size_l146_14645


namespace NUMINAMATH_CALUDE_shaded_square_ratio_l146_14693

/-- The ratio of the area of a shaded square to the area of a large square in a specific grid configuration -/
theorem shaded_square_ratio : 
  ∀ (n : ℕ) (large_square_area shaded_square_area : ℝ),
  n = 5 →
  large_square_area = n^2 →
  shaded_square_area = 4 * (1/2) →
  shaded_square_area / large_square_area = 2/25 := by
sorry

end NUMINAMATH_CALUDE_shaded_square_ratio_l146_14693


namespace NUMINAMATH_CALUDE_probability_select_from_both_sets_l146_14637

/-- The probability of selecting one card from each of two sets when drawing two cards at random without replacement, given a total of 13 cards where one set has 6 cards and the other has 7 cards. -/
theorem probability_select_from_both_sets : 
  ∀ (total : ℕ) (set1 : ℕ) (set2 : ℕ),
  total = 13 → set1 = 6 → set2 = 7 →
  (set1 / total * set2 / (total - 1) + set2 / total * set1 / (total - 1) : ℚ) = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_select_from_both_sets_l146_14637


namespace NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l146_14633

def parallelVectors (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_acute_angle (x : ℝ) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : parallelVectors (Real.sin x, 1) (1/2, Real.cos x)) : 
  x = π/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l146_14633


namespace NUMINAMATH_CALUDE_parentheses_removal_equivalence_l146_14635

theorem parentheses_removal_equivalence (x : ℝ) : 
  (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equivalence_l146_14635


namespace NUMINAMATH_CALUDE_simplify_expression_l146_14698

theorem simplify_expression (a b : ℝ) : 
  (30*a + 70*b) + (15*a + 40*b) - (12*a + 55*b) + (5*a - 10*b) = 38*a + 45*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l146_14698


namespace NUMINAMATH_CALUDE_parabola_translation_l146_14667

/-- The equation of a parabola after vertical translation -/
def translated_parabola (original : ℝ → ℝ) (translation : ℝ) : ℝ → ℝ :=
  fun x => original x + translation

/-- Theorem: Moving y = x^2 up 3 units results in y = x^2 + 3 -/
theorem parabola_translation :
  let original := fun x : ℝ => x^2
  translated_parabola original 3 = fun x => x^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l146_14667


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l146_14643

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l146_14643


namespace NUMINAMATH_CALUDE_intersection_is_open_interval_l146_14627

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the complement of M relative to ℝ
def M_complement : Set ℝ := {y | y ∉ M}

-- Define the intersection of M_complement and N
def intersection : Set ℝ := M_complement ∩ N

-- Theorem statement
theorem intersection_is_open_interval :
  intersection = {x | -Real.sqrt 3 < x ∧ x < -1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_is_open_interval_l146_14627


namespace NUMINAMATH_CALUDE_abs_sum_inequality_sum_bound_from_square_sum_l146_14605

-- Part I
theorem abs_sum_inequality (x a : ℝ) (ha : a > 0) :
  |x - 1/a| + |x + a| ≥ 2 := by sorry

-- Part II
theorem sum_bound_from_square_sum (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 3) :
  |x + 2*y + z| ≤ 3 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_sum_bound_from_square_sum_l146_14605


namespace NUMINAMATH_CALUDE_square_greater_than_abs_square_l146_14642

theorem square_greater_than_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_square_l146_14642


namespace NUMINAMATH_CALUDE_three_A_minus_two_B_three_A_minus_two_B_special_case_l146_14625

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2*x^2 + 3*x*y - 2*x - 1
def B (x y : ℝ) : ℝ := -x^2 + x*y - 1

-- Theorem for the general case
theorem three_A_minus_two_B (x y : ℝ) :
  3 * A x y - 2 * B x y = 8*x^2 + 7*x*y - 6*x - 1 :=
by sorry

-- Theorem for the specific case when |x+2| + (y-1)^2 = 0
theorem three_A_minus_two_B_special_case (x y : ℝ) 
  (h : |x + 2| + (y - 1)^2 = 0) :
  3 * A x y - 2 * B x y = 29 :=
by sorry

end NUMINAMATH_CALUDE_three_A_minus_two_B_three_A_minus_two_B_special_case_l146_14625


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l146_14672

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 512 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l146_14672


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l146_14622

theorem integer_solutions_of_system : 
  ∀ (x y z t : ℤ), 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l146_14622


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l146_14684

/-- The number of 6-digit numbers with at least two zeros -/
def six_digit_numbers_with_at_least_two_zeros : ℕ :=
  900000 - (9^6 + 5 * 9^5)

/-- Proof that the number of 6-digit numbers with at least two zeros is 73,314 -/
theorem count_six_digit_numbers_with_at_least_two_zeros :
  six_digit_numbers_with_at_least_two_zeros = 73314 := by
  sorry

#eval six_digit_numbers_with_at_least_two_zeros

end NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l146_14684


namespace NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l146_14671

/-- An infinite geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are in ascending order -/
def FirstThreeAscending (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity (a : ℕ → ℝ) 
  (h : GeometricSequence a) : 
  MonotonicallyIncreasing a ↔ FirstThreeAscending a := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_monotonicity_l146_14671


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l146_14640

/-- Given a circle with circumference 72 meters and an arc subtended by a 45° central angle,
    the length of the arc is 9 meters. -/
theorem arc_length_45_degrees (D : ℝ) (EF : ℝ) : 
  D = 72 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length as a fraction of the circumference
  EF = 9 := by sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l146_14640


namespace NUMINAMATH_CALUDE_last_s_replacement_l146_14680

/-- Represents the rules of the cryptographic code --/
structure CryptoRules where
  firstShift : ℕ
  vowels : List Char
  vowelSequence : List ℕ

/-- Counts the occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : ℕ := sorry

/-- Calculates the triangular number for a given n --/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Applies the shift to a character based on the rules --/
def applyShift (c : Char) (count : ℕ) (rules : CryptoRules) : Char := sorry

/-- Main theorem to prove --/
theorem last_s_replacement (message : String) (rules : CryptoRules) :
  let lastSCount := countOccurrences 's' message
  let shift := triangularNumber lastSCount % 26
  let newPos := (('s'.toNat - 'a'.toNat + 1 + shift) % 26) + 'a'.toNat - 1
  Char.ofNat newPos = 'g' := by sorry

end NUMINAMATH_CALUDE_last_s_replacement_l146_14680


namespace NUMINAMATH_CALUDE_factor_sum_l146_14695

/-- If x^2 + 3x + 4 is a factor of x^4 + Px^2 + Q, then P + Q = 15 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l146_14695


namespace NUMINAMATH_CALUDE_years_ago_p_half_q_l146_14652

/-- The number of years ago when p was half of q's age, given their current ages' ratio and sum. -/
theorem years_ago_p_half_q (p q : ℕ) (h1 : p * 4 = q * 3) (h2 : p + q = 28) : 
  ∃ y : ℕ, p - y = (q - y) / 2 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_years_ago_p_half_q_l146_14652


namespace NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l146_14653

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 5 and a triangular section DBE cut from it
    with DB = EB = 2, the perimeter of the remaining quadrilateral ACED is 13 -/
theorem remaining_quadrilateral_perimeter :
  ∀ (abc : Triangle) (dbe : Triangle) (aced : Quadrilateral),
    abc.a = 5 ∧ abc.b = 5 ∧ abc.c = 5 →  -- ABC is equilateral with side length 5
    dbe.a = 2 ∧ dbe.b = 2 ∧ dbe.c = 2 →  -- DBE is equilateral with side length 2
    aced.a = 5 ∧                         -- AC remains untouched
    aced.b = abc.b - dbe.b ∧             -- CE = AB - DB
    aced.c = dbe.c ∧                     -- ED is a side of DBE
    aced.d = abc.c - dbe.c →             -- DA = BC - BE
    aced.perimeter = 13 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l146_14653


namespace NUMINAMATH_CALUDE_divisor_counts_of_N_l146_14658

def N : ℕ := 10^40

/-- The number of natural divisors of N that are neither perfect squares nor perfect cubes -/
def count_non_square_non_cube_divisors (n : ℕ) : ℕ := sorry

/-- The number of natural divisors of N that cannot be represented as m^n where m and n are natural numbers and n > 1 -/
def count_non_power_divisors (n : ℕ) : ℕ := sorry

theorem divisor_counts_of_N :
  (count_non_square_non_cube_divisors N = 1093) ∧
  (count_non_power_divisors N = 981) := by sorry

end NUMINAMATH_CALUDE_divisor_counts_of_N_l146_14658


namespace NUMINAMATH_CALUDE_unique_m_value_l146_14606

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {m, 1}
  let B : Set ℝ := {m^2, -1}
  A = B → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l146_14606


namespace NUMINAMATH_CALUDE_equal_ratios_imply_k_value_l146_14610

theorem equal_ratios_imply_k_value (x y z k : ℝ) 
  (h1 : 12 / (x + z) = k / (z - y))
  (h2 : k / (z - y) = 5 / (y - x))
  (h3 : y = 0) : k = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_imply_k_value_l146_14610


namespace NUMINAMATH_CALUDE_product_tens_digit_is_nine_l146_14678

theorem product_tens_digit_is_nine (x : ℤ) : 
  0 ≤ x ∧ x ≤ 9 → 
  ((200 + 10 * x + 7) * 39 ≡ 90 [ZMOD 100] ↔ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_product_tens_digit_is_nine_l146_14678


namespace NUMINAMATH_CALUDE_sum_of_first_50_terms_l146_14683

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |> List.sum

theorem sum_of_first_50_terms (a : ℕ → ℕ) :
  a 1 = 7 ∧ (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 20) →
  sequence_sum a 50 = 500 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_50_terms_l146_14683


namespace NUMINAMATH_CALUDE_sum_of_first_three_terms_l146_14603

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define S_3 as the sum of the first three terms
def S3 : ℚ := a 1 + a 2 + a 3

-- Theorem statement
theorem sum_of_first_three_terms : S3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_three_terms_l146_14603


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l146_14618

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l146_14618


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l146_14690

/-- The number of egg cartons that can be filled given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def egg_cartons_filled (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Theorem stating that with 20 chickens, each laying 6 eggs, and egg cartons
    that hold 12 eggs each, the number of egg cartons that can be filled is 10. -/
theorem avery_egg_cartons :
  egg_cartons_filled 20 6 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l146_14690


namespace NUMINAMATH_CALUDE_mathematician_daily_questions_l146_14646

theorem mathematician_daily_questions 
  (project1_questions : ℕ) 
  (project2_questions : ℕ) 
  (days_in_week : ℕ) 
  (h1 : project1_questions = 518) 
  (h2 : project2_questions = 476) 
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 :=
by sorry

end NUMINAMATH_CALUDE_mathematician_daily_questions_l146_14646


namespace NUMINAMATH_CALUDE_beanie_tickets_l146_14628

def arcade_tickets (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem beanie_tickets : arcade_tickets 11 10 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_beanie_tickets_l146_14628


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l146_14692

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = 17 / Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l146_14692


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l146_14691

theorem cubic_sum_theorem (p q r : ℝ) (hp : p ≠ q) (hq : q ≠ r) (hr : r ≠ p)
  (h : (p^3 + 8) / p = (q^3 + 8) / q ∧ (q^3 + 8) / q = (r^3 + 8) / r) :
  p^3 + q^3 + r^3 = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l146_14691


namespace NUMINAMATH_CALUDE_divisibility_problem_l146_14656

theorem divisibility_problem (n a b c d : ℤ) 
  (hn : n > 0) 
  (h1 : n ∣ (a + b + c + d)) 
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) : 
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l146_14656


namespace NUMINAMATH_CALUDE_parking_lot_cars_l146_14613

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l146_14613


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l146_14682

-- Define the sets S and T
def S : Set ℝ := {0, 1, 2, 3}
def T : Set ℝ := {x | |x - 1| ≤ 1}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l146_14682


namespace NUMINAMATH_CALUDE_base_conversion_sum_l146_14650

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : Nat) (c d : Nat) : Nat :=
  sorry

theorem base_conversion_sum :
  let c : Nat := 12
  let d : Nat := 13
  base8ToBase10 356 + base14ToBase10 4 c d = 1203 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l146_14650


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l146_14673

theorem complex_modulus_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l146_14673


namespace NUMINAMATH_CALUDE_find_d_l146_14666

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l146_14666


namespace NUMINAMATH_CALUDE_decagon_triangles_l146_14632

/-- The number of vertices in a decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem decagon_triangles : choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l146_14632


namespace NUMINAMATH_CALUDE_system_no_solution_l146_14679

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, nx + y + z ≠ 2 ∨ x + ny + z ≠ 2 ∨ x + y + nz ≠ 2) ↔ n = -1 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l146_14679


namespace NUMINAMATH_CALUDE_abcd_mod_11_l146_14660

theorem abcd_mod_11 (a b c d : ℕ) : 
  a < 11 → b < 11 → c < 11 → d < 11 →
  (a + 3*b + 4*c + 2*d) % 11 = 3 →
  (3*a + b + 2*c + d) % 11 = 5 →
  (2*a + 4*b + c + 3*d) % 11 = 7 →
  (a + b + c + d) % 11 = 2 →
  (a * b * c * d) % 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_mod_11_l146_14660


namespace NUMINAMATH_CALUDE_calculation_proof_l146_14648

theorem calculation_proof : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l146_14648


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_triangle_l146_14697

/-- 
Given an isosceles trapezoid with parallel sides a and c, non-parallel sides (legs) b, 
and diagonals e, prove that e² = b² + ac, which implies that the triangle formed by 
e, b, and √(ac) is a right triangle.
-/
theorem isosceles_trapezoid_right_triangle 
  (a c b e : ℝ) 
  (h_positive : a > 0 ∧ c > 0 ∧ b > 0 ∧ e > 0)
  (h_isosceles : ∃ m : ℝ, b^2 = ((a - c)/2)^2 + m^2 ∧ e^2 = ((a + c)/2)^2 + m^2) :
  e^2 = b^2 + a*c :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_triangle_l146_14697


namespace NUMINAMATH_CALUDE_zoo_ratio_l146_14617

theorem zoo_ratio (sea_lions penguins : ℕ) : 
  sea_lions = 48 →
  penguins = sea_lions + 84 →
  (sea_lions : ℚ) / penguins = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ratio_l146_14617


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l146_14663

theorem quadratic_root_difference (b : ℝ) : 
  (∃ (x y : ℝ), 2 * x^2 + b * x = 12 ∧ 
                 2 * y^2 + b * y = 12 ∧ 
                 y - x = 5.5 ∧ 
                 (∀ z : ℝ, 2 * z^2 + b * z = 12 → (z = x ∨ z = y))) →
  b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l146_14663


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l146_14676

theorem product_from_sum_and_difference (a b : ℝ) : 
  a + b = 60 ∧ a - b = 10 → a * b = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l146_14676


namespace NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l146_14699

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the sine of an angle in a triangle
def sine (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Define whether a point lies on the circumcircle of a triangle
def onCircumcircle (t : Triangle) (p : Point) : Prop := sorry

theorem triangle_inequality (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A ≤ distance p t.B * sine t t.B + distance p t.C * sine t t.C :=
sorry

theorem equality_condition (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A = distance p t.B * sine t t.B + distance p t.C * sine t t.C ↔
  onCircumcircle t p :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l146_14699


namespace NUMINAMATH_CALUDE_binary_11001_is_25_l146_14604

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end NUMINAMATH_CALUDE_binary_11001_is_25_l146_14604


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_negative_three_l146_14607

theorem sum_of_fractions_equals_negative_three 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a + b + c = 3) :
  1 / (b^2 + c^2 - 3*a^2) + 1 / (a^2 + c^2 - 3*b^2) + 1 / (a^2 + b^2 - 3*c^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_negative_three_l146_14607


namespace NUMINAMATH_CALUDE_monotone_decreasing_range_l146_14688

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else -x^2 + a

theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_range_l146_14688


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l146_14647

theorem ratio_x_to_y (x y : ℝ) (h : (14*x - 5*y) / (17*x - 3*y) = 4/6) : x/y = 1/23 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l146_14647


namespace NUMINAMATH_CALUDE_smallest_number_l146_14665

theorem smallest_number (s : Set ℤ) (hs : s = {-2, 0, -1, 3}) : 
  ∃ m ∈ s, ∀ x ∈ s, m ≤ x ∧ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l146_14665


namespace NUMINAMATH_CALUDE_inequality_solution_set_l146_14657

theorem inequality_solution_set (x : ℝ) :
  (2 * x) / (x - 2) ≤ 1 ↔ x ∈ Set.Icc (-2) 2 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l146_14657


namespace NUMINAMATH_CALUDE_society_officer_selection_l146_14641

theorem society_officer_selection (n : ℕ) (k : ℕ) : n = 12 ∧ k = 5 →
  (n.factorial / (n - k).factorial) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_society_officer_selection_l146_14641


namespace NUMINAMATH_CALUDE_group_dance_arrangements_l146_14612

/-- The number of boys in the group dance -/
def num_boys : ℕ := 10

/-- The number of girls in the group dance -/
def num_girls : ℕ := 10

/-- The total number of people in the group dance -/
def total_people : ℕ := num_boys + num_girls

/-- The number of columns in the group dance -/
def num_columns : ℕ := 2

/-- The number of arrangements when boys and girls are in separate columns -/
def separate_columns_arrangements : ℕ := 2 * (Nat.factorial num_boys)^2

/-- The number of arrangements when boys and girls can stand in any column -/
def mixed_columns_arrangements : ℕ := Nat.factorial total_people

/-- The number of pairings when boys and girls are in separate columns and internal order doesn't matter -/
def pairings_separate_columns : ℕ := 2 * Nat.factorial num_boys

theorem group_dance_arrangements :
  (separate_columns_arrangements = 2 * (Nat.factorial num_boys)^2) ∧
  (mixed_columns_arrangements = Nat.factorial total_people) ∧
  (pairings_separate_columns = 2 * Nat.factorial num_boys) :=
sorry

end NUMINAMATH_CALUDE_group_dance_arrangements_l146_14612


namespace NUMINAMATH_CALUDE_morgan_change_l146_14661

/-- Calculates the change received from a purchase given item costs and amount paid --/
def calculate_change (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) : ℕ :=
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change --/
theorem morgan_change : calculate_change 4 2 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_morgan_change_l146_14661


namespace NUMINAMATH_CALUDE_juice_remaining_l146_14630

theorem juice_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 5 → given = 18/7 → remaining = initial - given → remaining = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_juice_remaining_l146_14630


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l146_14614

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l146_14614


namespace NUMINAMATH_CALUDE_supremum_of_expression_l146_14651

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9/2 ∧ 
  ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1 ∧ -1 / (2 * a') - 2 / b' = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_supremum_of_expression_l146_14651


namespace NUMINAMATH_CALUDE_a_plus_b_value_l146_14675

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = univ) → (A ∩ B a b = Ioc 3 4) → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l146_14675


namespace NUMINAMATH_CALUDE_probability_one_defective_l146_14668

def total_items : ℕ := 6
def good_items : ℕ := 4
def defective_items : ℕ := 2
def selected_items : ℕ := 3

theorem probability_one_defective :
  (Nat.choose good_items (selected_items - 1) * Nat.choose defective_items 1) /
  Nat.choose total_items selected_items = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_defective_l146_14668


namespace NUMINAMATH_CALUDE_potatoes_remaining_l146_14626

/-- Calculates the number of potatoes left after distribution -/
def potatoes_left (total : ℕ) (to_gina : ℕ) : ℕ :=
  let to_tom := 2 * to_gina
  let to_anne := to_tom / 3
  total - (to_gina + to_tom + to_anne)

/-- Theorem stating that 47 potatoes are left after distribution -/
theorem potatoes_remaining : potatoes_left 300 69 = 47 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_remaining_l146_14626


namespace NUMINAMATH_CALUDE_highest_common_factor_l146_14639

/- Define the polynomials f and g -/
def f (n : ℕ) (x : ℝ) : ℝ := n * x^(n+1) - (n+1) * x^n + 1

def g (n : ℕ) (x : ℝ) : ℝ := x^n - n*x + n - 1

/- State the theorem -/
theorem highest_common_factor (n : ℕ) (h : n ≥ 2) :
  ∃ (p q : ℝ → ℝ), 
    (∀ x, f n x = (x - 1)^2 * p x) ∧ 
    (∀ x, g n x = (x - 1) * q x) ∧
    (∀ r : ℝ → ℝ, (∀ x, f n x = r x * (p x)) → (∀ x, g n x = r x * (q x)) → 
      ∃ (s : ℝ → ℝ), ∀ x, r x = (x - 1)^2 * s x) :=
sorry

end NUMINAMATH_CALUDE_highest_common_factor_l146_14639


namespace NUMINAMATH_CALUDE_ceil_sum_sqrt_l146_14621

theorem ceil_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 35⌉ + ⌈Real.sqrt 350⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceil_sum_sqrt_l146_14621


namespace NUMINAMATH_CALUDE_root_implies_k_value_l146_14674

theorem root_implies_k_value (k : ℚ) : 
  (∃ x : ℚ, x^2 - 2*x + 2*k = 0) ∧ (1^2 - 2*1 + 2*k = 0) → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l146_14674


namespace NUMINAMATH_CALUDE_equality_of_ratios_l146_14655

theorem equality_of_ratios (a b c d : ℕ) 
  (h1 : a / c = b / d) 
  (h2 : a / c = (a * b + 1) / (c * d + 1)) 
  (h3 : b / d = (a * b + 1) / (c * d + 1)) : 
  a = c ∧ b = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_ratios_l146_14655


namespace NUMINAMATH_CALUDE_jade_handled_80_transactions_l146_14677

/-- Calculates the number of transactions Jade handled given the conditions of the problem. -/
def jade_transactions (mabel_transactions : ℕ) : ℕ :=
  let anthony_transactions := mabel_transactions + mabel_transactions / 10
  let cal_transactions := anthony_transactions * 2 / 3
  cal_transactions + 14

/-- Theorem stating that Jade handled 80 transactions given the conditions of the problem. -/
theorem jade_handled_80_transactions : jade_transactions 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_handled_80_transactions_l146_14677


namespace NUMINAMATH_CALUDE_hawks_score_l146_14636

/-- 
Given:
- The total points scored by both teams is 50
- The Eagles won by a margin of 18 points

Prove that the Hawks scored 16 points
-/
theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 50 →
  eagles_points = hawks_points + 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 16 := by
sorry

end NUMINAMATH_CALUDE_hawks_score_l146_14636


namespace NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2025_l146_14609

def x (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 :
  ∀ k : ℕ, ∃ i : Fin 5, ¬(2025 ∣ x (k + i.val)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2025_l146_14609


namespace NUMINAMATH_CALUDE_circle_C_properties_l146_14601

def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 3

def point_on_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = 2 + Real.sqrt 5 * Real.cos θ ∧ y = 2 + Real.sqrt 5 * Real.sin θ

theorem circle_C_properties :
  -- 1. Parametric equations
  (∀ x y θ : ℝ, point_on_C x y ↔ 
    x = 2 + Real.sqrt 5 * Real.cos θ ∧ 
    y = 2 + Real.sqrt 5 * Real.sin θ) ∧
  -- 2. Maximum value of x + 2y
  (∀ x y : ℝ, point_on_C x y → x + 2*y ≤ 11) ∧
  -- 3. Coordinates at maximum
  (point_on_C 3 4 ∧ 3 + 2*4 = 11) :=
by sorry

end NUMINAMATH_CALUDE_circle_C_properties_l146_14601


namespace NUMINAMATH_CALUDE_newspaper_photos_newspaper_photos_proof_l146_14694

theorem newspaper_photos : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (pages_with_two_photos : ℕ) 
      (photos_per_page_first : ℕ) 
      (pages_with_three_photos : ℕ) 
      (photos_per_page_second : ℕ) 
      (total_photos : ℕ) =>
    pages_with_two_photos = 12 ∧ 
    photos_per_page_first = 2 ∧
    pages_with_three_photos = 9 ∧ 
    photos_per_page_second = 3 →
    total_photos = pages_with_two_photos * photos_per_page_first + 
                   pages_with_three_photos * photos_per_page_second ∧
    total_photos = 51

theorem newspaper_photos_proof : newspaper_photos 12 2 9 3 51 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_newspaper_photos_proof_l146_14694


namespace NUMINAMATH_CALUDE_min_sum_of_powers_with_same_last_four_digits_l146_14611

theorem min_sum_of_powers_with_same_last_four_digits :
  ∀ m n : ℕ+,
    m ≠ n →
    (10000 : ℤ) ∣ (2019^(m.val) - 2019^(n.val)) →
    ∀ k l : ℕ+,
      k ≠ l →
      (10000 : ℤ) ∣ (2019^(k.val) - 2019^(l.val)) →
      m.val + n.val ≤ k.val + l.val →
      m.val + n.val = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_powers_with_same_last_four_digits_l146_14611


namespace NUMINAMATH_CALUDE_cookie_theorem_l146_14619

/-- The number of combinations when selecting 8 cookies from 4 types, with at least one of each type -/
def cookieCombinations : ℕ := 46

/-- The function that calculates the number of combinations -/
def calculateCombinations (totalCookies : ℕ) (cookieTypes : ℕ) : ℕ :=
  sorry

theorem cookie_theorem :
  calculateCombinations 8 4 = cookieCombinations :=
by sorry

end NUMINAMATH_CALUDE_cookie_theorem_l146_14619


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l146_14634

/-- Given a hyperbola with equation x²/25 - y²/16 = 1, prove that the positive value m
    such that y = ±mx represents the asymptotes is 4/5 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  x^2 / 25 - y^2 / 16 = 1 →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 4/5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l146_14634


namespace NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l146_14681

theorem arc_length_for_36_degree_angle (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 → θ_deg = 36 → l = (θ_deg * π / 180) * (d / 2) → l = 2 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l146_14681


namespace NUMINAMATH_CALUDE_parabola_c_value_l146_14687

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value :
  ∀ p : Parabola,
    p.x_coord (-3) = 2 →  -- vertex at (2, -3)
    p.x_coord (-1) = 7 →  -- passes through (7, -1)
    p.c = 53/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l146_14687


namespace NUMINAMATH_CALUDE_special_function_at_one_fifth_l146_14654

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y ∨ f x > f y) ∧
  (∀ x, 0 < x → f (f x - 1/x) = 2)

/-- The value of f(1/5) for a special function f -/
theorem special_function_at_one_fifth
    (f : ℝ → ℝ) (h : special_function f) : f (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_fifth_l146_14654


namespace NUMINAMATH_CALUDE_K_is_perfect_square_l146_14686

def K (n : ℕ) : ℚ :=
  (4 * (10^(2*n) - 1) / 9) - (8 * (10^n - 1) / 9)

theorem K_is_perfect_square (n : ℕ) :
  ∃ (m : ℚ), K n = m^2 := by
sorry

end NUMINAMATH_CALUDE_K_is_perfect_square_l146_14686


namespace NUMINAMATH_CALUDE_not_all_same_probability_l146_14670

def roll_five_eight_sided_dice : ℕ := 8^5

def same_number_outcomes : ℕ := 8

theorem not_all_same_probability :
  (roll_five_eight_sided_dice - same_number_outcomes) / roll_five_eight_sided_dice = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_not_all_same_probability_l146_14670


namespace NUMINAMATH_CALUDE_escalator_speed_l146_14602

/-- The speed of an escalator given its length, a person's walking speed, and the time taken to cover the entire length. -/
theorem escalator_speed (escalator_length : ℝ) (walking_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 126 ∧ walking_speed = 3 ∧ time_taken = 9 →
  ∃ (escalator_speed : ℝ), 
    escalator_speed = 11 ∧ 
    (escalator_speed + walking_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l146_14602


namespace NUMINAMATH_CALUDE_tom_carrot_consumption_l146_14696

/-- Proves that Tom ate 1 pound of carrots given the conditions of the problem -/
theorem tom_carrot_consumption (C : ℝ) : 
  C > 0 →  -- Assuming C is positive (implicit in the original problem)
  51 * C + 2 * C * (51 / 3) = 85 →
  C = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_carrot_consumption_l146_14696
