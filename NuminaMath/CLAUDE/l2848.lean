import Mathlib

namespace NUMINAMATH_CALUDE_square_area_from_vertices_l2848_284816

/-- The area of a square with adjacent vertices at (1,3) and (-2,7) is 25 -/
theorem square_area_from_vertices :
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-2, 7)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l2848_284816


namespace NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l2848_284881

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_aces := 4,
    num_kings := 4 }

theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l2848_284881


namespace NUMINAMATH_CALUDE_unique_integer_pair_solution_l2848_284820

theorem unique_integer_pair_solution : 
  ∃! (x y : ℤ), Real.sqrt (x - Real.sqrt (x + 23)) = 2 * Real.sqrt 2 - y := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_solution_l2848_284820


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_side_lengths_l2848_284839

theorem rectangular_parallelepiped_side_lengths 
  (x y z : ℝ) 
  (sum_eq : x + y + z = 17)
  (area_eq : 2*x*y + 2*y*z + 2*z*x = 180)
  (sq_sum_eq : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_side_lengths_l2848_284839


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l2848_284833

theorem tennis_ball_ratio : 
  let total_ordered : ℕ := 64
  let extra_yellow : ℕ := 20
  let white_balls : ℕ := total_ordered / 2
  let yellow_balls : ℕ := total_ordered / 2 + extra_yellow
  let gcd : ℕ := Nat.gcd white_balls yellow_balls
  (white_balls / gcd : ℕ) = 8 ∧ (yellow_balls / gcd : ℕ) = 13 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l2848_284833


namespace NUMINAMATH_CALUDE_sequences_not_periodic_l2848_284890

/-- Sequence A constructed by writing slices of increasing lengths from 1,0,0,0,... -/
def sequence_A : ℕ → ℕ := sorry

/-- Sequence B constructed by writing slices of two, four, six, etc., elements from 1,2,3,1,2,3,... -/
def sequence_B : ℕ → ℕ := sorry

/-- Sequence C formed by adding the corresponding elements of A and B -/
def sequence_C (n : ℕ) : ℕ := sequence_A n + sequence_B n

/-- A sequence is periodic if there exists a positive integer k such that
    for all n ≥ some fixed N, a(n+k) = a(n) -/
def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ) (h : k > 0), ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → a (n + k) = a n

theorem sequences_not_periodic :
  ¬(is_periodic sequence_A) ∧ ¬(is_periodic sequence_B) ∧ ¬(is_periodic sequence_C) := by sorry

end NUMINAMATH_CALUDE_sequences_not_periodic_l2848_284890


namespace NUMINAMATH_CALUDE_binomial_15_12_l2848_284888

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l2848_284888


namespace NUMINAMATH_CALUDE_inequality_proof_l2848_284824

theorem inequality_proof (a b c : ℕ+) (h : c ≥ b) :
  (a ^ b.val) * ((a + b) ^ c.val) > (c ^ b.val) * (a ^ c.val) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2848_284824


namespace NUMINAMATH_CALUDE_elberta_has_41_dollars_l2848_284810

def granny_smith_amount : ℕ := 72

def anjou_amount (granny_smith : ℕ) : ℕ :=
  granny_smith / 4

def elberta_amount (anjou : ℕ) : ℕ :=
  2 * anjou + 5

theorem elberta_has_41_dollars : 
  elberta_amount (anjou_amount granny_smith_amount) = 41 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_41_dollars_l2848_284810


namespace NUMINAMATH_CALUDE_fraction_product_equality_l2848_284803

theorem fraction_product_equality : (2 : ℚ) / 8 * (6 : ℚ) / 9 = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l2848_284803


namespace NUMINAMATH_CALUDE_mem_not_zeige_l2848_284870

-- Define our universes
variable (U : Type)

-- Define our sets
variable (Mem Enform Zeige : Set U)

-- State our premises
variable (h1 : Mem ⊆ Enform)
variable (h2 : Enform ∩ Zeige = ∅)

-- State our theorem
theorem mem_not_zeige :
  (∀ x, x ∈ Mem → x ∉ Zeige) ∧
  (Mem ∩ Zeige = ∅) :=
sorry

end NUMINAMATH_CALUDE_mem_not_zeige_l2848_284870


namespace NUMINAMATH_CALUDE_frustum_cut_off_height_l2848_284879

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  originalHeight : ℝ
  frustumHeight : ℝ
  upperRadius : ℝ
  lowerRadius : ℝ

/-- Calculates the height of the smaller cone cut off from the original cone -/
def cutOffHeight (f : Frustum) : ℝ :=
  f.originalHeight - f.frustumHeight

theorem frustum_cut_off_height (f : Frustum) 
  (h1 : f.originalHeight = 30)
  (h2 : f.frustumHeight = 18)
  (h3 : f.upperRadius = 6)
  (h4 : f.lowerRadius = 10) :
  cutOffHeight f = 12 := by
sorry

end NUMINAMATH_CALUDE_frustum_cut_off_height_l2848_284879


namespace NUMINAMATH_CALUDE_students_with_both_calculation_l2848_284860

/-- The number of students who brought both apples and bananas -/
def students_with_both : ℕ := sorry

/-- The number of students who brought apples -/
def students_with_apples : ℕ := 12

/-- The number of students who brought bananas -/
def students_with_bananas : ℕ := 8

/-- The number of students who brought only one type of fruit -/
def students_with_one_fruit : ℕ := 10

theorem students_with_both_calculation : 
  students_with_both = students_with_apples + students_with_bananas - students_with_one_fruit :=
by sorry

end NUMINAMATH_CALUDE_students_with_both_calculation_l2848_284860


namespace NUMINAMATH_CALUDE_polar_cartesian_equivalence_l2848_284819

/-- The curve C in polar coordinates -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

/-- The curve C in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

/-- Theorem stating the equivalence of polar and Cartesian equations for curve C -/
theorem polar_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by
  sorry

end NUMINAMATH_CALUDE_polar_cartesian_equivalence_l2848_284819


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l2848_284818

theorem fixed_point_parabola :
  ∀ (k : ℝ), 225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l2848_284818


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2848_284843

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2848_284843


namespace NUMINAMATH_CALUDE_point_line_plane_relations_l2848_284852

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (lies_in_line : Line → Plane → Prop)
variable (lies_in_point : Point → Plane → Prop)

-- State the theorem
theorem point_line_plane_relations 
  (A : Point) (a : Line) (α : Plane) (B : Point) :
  lies_on A a → lies_in_line a α → lies_in_point B α →
  (A ∈ {x : Point | lies_on x a}) ∧ 
  ({x : Point | lies_on x a} ⊆ {x : Point | lies_in_point x α}) ∧ 
  (B ∈ {x : Point | lies_in_point x α}) :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_relations_l2848_284852


namespace NUMINAMATH_CALUDE_negation_of_implication_l2848_284841

theorem negation_of_implication (x y : ℝ) : 
  ¬(x = 0 ∧ y = 0 → x * y = 0) ↔ (¬(x = 0 ∧ y = 0) → x * y ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2848_284841


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2848_284846

theorem prime_equation_solution (p : ℕ) (x y : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : x * (y^2 - p) + y * (x^2 - p) = 5 * p) : 
  p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2848_284846


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2848_284828

/-- The common ratio of the geometric series 4/5 - 5/12 + 25/72 - ... is -25/48 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/5
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/72
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₂ = r * a₁ ∧ a₃ = r * a₂) →
  r = -25/48 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2848_284828


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2848_284876

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 7) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2848_284876


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2848_284815

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y + 1 = 0 -/
theorem intersection_line_equation 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁ * 2 + b₁ * 3 + 1 = 0)
  (h₂ : a₂ * 2 + b₂ * 3 + 1 = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧ 2 * a₁ + 3 * b₁ + k = 0 ∧ 2 * a₂ + 3 * b₂ + k = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2848_284815


namespace NUMINAMATH_CALUDE_calculator_squaring_l2848_284859

theorem calculator_squaring (n : ℕ) : (1 : ℝ) ^ (2^n) ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_calculator_squaring_l2848_284859


namespace NUMINAMATH_CALUDE_triangle_height_l2848_284812

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 8 → area = 16 → area = (base * height) / 2 → height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2848_284812


namespace NUMINAMATH_CALUDE_kenny_contribution_percentage_l2848_284897

/-- Represents the contributions and total cost for house painting -/
structure PaintingContributions where
  total_cost : ℕ
  judson_contribution : ℕ
  kenny_contribution : ℕ
  camilo_contribution : ℕ

/-- Defines the conditions for the house painting contributions -/
def valid_contributions (c : PaintingContributions) : Prop :=
  c.total_cost = 1900 ∧
  c.judson_contribution = 500 ∧
  c.kenny_contribution > c.judson_contribution ∧
  c.camilo_contribution = c.kenny_contribution + 200 ∧
  c.judson_contribution + c.kenny_contribution + c.camilo_contribution = c.total_cost

/-- Calculates the percentage difference between Kenny's and Judson's contributions -/
def percentage_difference (c : PaintingContributions) : ℚ :=
  (c.kenny_contribution - c.judson_contribution : ℚ) / c.judson_contribution * 100

/-- Theorem stating that Kenny contributed 20% more than Judson -/
theorem kenny_contribution_percentage (c : PaintingContributions)
  (h : valid_contributions c) : percentage_difference c = 20 := by
  sorry


end NUMINAMATH_CALUDE_kenny_contribution_percentage_l2848_284897


namespace NUMINAMATH_CALUDE_total_snowfall_l2848_284873

theorem total_snowfall (morning_snow afternoon_snow : ℝ) 
  (h1 : morning_snow = 0.125)
  (h2 : afternoon_snow = 0.5) :
  morning_snow + afternoon_snow = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_total_snowfall_l2848_284873


namespace NUMINAMATH_CALUDE_part_one_part_two_l2848_284853

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- Part 1
theorem part_one :
  let a : ℝ := -2
  (B a ∩ A = {x | 1 ≤ x ∧ x < 4}) ∧
  (B a ∩ (Set.univ \ A) = {x | (-4 ≤ x ∧ x < 1) ∨ (4 ≤ x ∧ x < 5)}) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2848_284853


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l2848_284809

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 36 games, there are 9 teams -/
theorem round_robin_tournament_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 36 → n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l2848_284809


namespace NUMINAMATH_CALUDE_bank_deposit_problem_l2848_284842

theorem bank_deposit_problem (P : ℝ) : 
  P > 0 →
  (P * 0.15 * 5 - P * 0.15 * 3.5 = 144) →
  P = 640 := by
sorry

end NUMINAMATH_CALUDE_bank_deposit_problem_l2848_284842


namespace NUMINAMATH_CALUDE_direct_proportion_constant_zero_l2848_284802

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- Given that y = x + b is a direct proportion function, b must be zero -/
theorem direct_proportion_constant_zero (b : ℝ) :
  IsDirectProportionFunction (fun x ↦ x + b) → b = 0 := by
  sorry


end NUMINAMATH_CALUDE_direct_proportion_constant_zero_l2848_284802


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2848_284865

/-- Represents a cube arrangement with specific properties -/
structure CubeArrangement where
  num_cubes : ℕ
  central_cube_exposed_faces : ℕ
  surrounding_cubes_exposed_faces : ℕ
  extending_cube_exposed_faces : ℕ

/-- Calculate the volume of the cube arrangement -/
def volume (c : CubeArrangement) : ℕ :=
  c.num_cubes

/-- Calculate the surface area of the cube arrangement -/
def surface_area (c : CubeArrangement) : ℕ :=
  c.surrounding_cubes_exposed_faces * 5 + c.central_cube_exposed_faces + c.extending_cube_exposed_faces

/-- The specific cube arrangement described in the problem -/
def special_arrangement : CubeArrangement :=
  { num_cubes := 8,
    central_cube_exposed_faces := 1,
    surrounding_cubes_exposed_faces := 5,
    extending_cube_exposed_faces := 3 }

/-- Theorem stating the ratio of volume to surface area for the special arrangement -/
theorem volume_to_surface_area_ratio :
  (volume special_arrangement : ℚ) / (surface_area special_arrangement : ℚ) = 8 / 29 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2848_284865


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_l2848_284851

/-- Given a rectangle JKLM and a square NOPQ, if 30% of JKLM's area overlaps with NOPQ,
    and 40% of NOPQ's area overlaps with JKLM, then the ratio of JKLM's length to its width is 4/3. -/
theorem rectangle_square_overlap (j l m n : ℝ) :
  j > 0 → l > 0 → m > 0 → n > 0 →
  0.3 * (j * l) = 0.4 * (n * n) →
  j * l = m * n →
  j / m = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_l2848_284851


namespace NUMINAMATH_CALUDE_sin_product_equality_l2848_284835

theorem sin_product_equality : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  (Real.cos (20 * π / 180) - 1/2) / 8 := by sorry

end NUMINAMATH_CALUDE_sin_product_equality_l2848_284835


namespace NUMINAMATH_CALUDE_vector_BC_l2848_284871

/-- Given two vectors AB and AC in 2D space, prove that the vector BC is their difference -/
theorem vector_BC (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, -1]) (h2 : AC = ![-4, 1]) :
  AC - AB = ![-6, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l2848_284871


namespace NUMINAMATH_CALUDE_nine_twelve_fifteen_pythagorean_triple_l2848_284892

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- Prove that (9, 12, 15) is a Pythagorean triple --/
theorem nine_twelve_fifteen_pythagorean_triple : is_pythagorean_triple 9 12 15 := by
  sorry

end NUMINAMATH_CALUDE_nine_twelve_fifteen_pythagorean_triple_l2848_284892


namespace NUMINAMATH_CALUDE_closest_ratio_is_27_25_l2848_284895

def adult_fee : ℕ := 25
def child_fee : ℕ := 12
def total_fee : ℕ := 1950

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 1 ∧ children ≥ 1 ∧ adult_fee * adults + child_fee * children = total_fee

def ratio_difference_from_one (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_27_25 :
  ∃ (adults children : ℕ),
    is_valid_combination adults children ∧
    (∀ (a c : ℕ), is_valid_combination a c →
      ratio_difference_from_one adults children ≤ ratio_difference_from_one a c) ∧
    (adults : ℚ) / (children : ℚ) = 27 / 25 := by
  sorry

end NUMINAMATH_CALUDE_closest_ratio_is_27_25_l2848_284895


namespace NUMINAMATH_CALUDE_T_is_four_sided_polygon_l2848_284830

-- Define the set T
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a ≤ x ∧ x ≤ 3*a ∧
    a ≤ y ∧ y ≤ 3*a ∧
    x + y ≥ 2*a ∧
    x + 2*a ≥ y ∧
    y + 2*a ≥ x ∧
    x + y ≤ 4*a}

-- Theorem statement
theorem T_is_four_sided_polygon (a : ℝ) (h : a > 0) :
  ∃ (v1 v2 v3 v4 : ℝ × ℝ),
    v1 ∈ T a ∧ v2 ∈ T a ∧ v3 ∈ T a ∧ v4 ∈ T a ∧
    (∀ p ∈ T a, p = v1 ∨ p = v2 ∨ p = v3 ∨ p = v4 ∨
      (∃ t : ℝ, 0 < t ∧ t < 1 ∧
        (p = (1 - t) • v1 + t • v2 ∨
         p = (1 - t) • v2 + t • v3 ∨
         p = (1 - t) • v3 + t • v4 ∨
         p = (1 - t) • v4 + t • v1))) :=
by sorry

end NUMINAMATH_CALUDE_T_is_four_sided_polygon_l2848_284830


namespace NUMINAMATH_CALUDE_max_product_decomposition_l2848_284832

theorem max_product_decomposition (n k : ℕ) (h : k ≤ n) :
  ∃ (decomp : List ℕ),
    (decomp.sum = n) ∧
    (decomp.length = k) ∧
    (∀ (other_decomp : List ℕ),
      (other_decomp.sum = n) ∧ (other_decomp.length = k) →
      decomp.prod ≥ other_decomp.prod) ∧
    (decomp = List.replicate (n - n / k * k) (n / k + 1) ++ List.replicate (k - (n - n / k * k)) (n / k)) :=
  sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l2848_284832


namespace NUMINAMATH_CALUDE_dave_tickets_l2848_284868

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 18 tickets -/
theorem dave_tickets : final_tickets 25 22 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2848_284868


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2848_284898

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 3*m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2848_284898


namespace NUMINAMATH_CALUDE_roses_in_bouquet_l2848_284893

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_table_decorations : ℕ := 7

/-- The number of white roses in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed -/
def total_roses : ℕ := 109

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / num_bouquets

theorem roses_in_bouquet :
  roses_per_bouquet = 5 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_bouquet_l2848_284893


namespace NUMINAMATH_CALUDE_range_of_a_l2848_284878

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2848_284878


namespace NUMINAMATH_CALUDE_monthly_bill_increase_l2848_284813

theorem monthly_bill_increase (original_bill : ℝ) (increase_percentage : ℝ) : 
  original_bill = 60 →
  increase_percentage = 0.30 →
  original_bill + (increase_percentage * original_bill) = 78 := by
  sorry

end NUMINAMATH_CALUDE_monthly_bill_increase_l2848_284813


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2848_284829

/-- Represents a tile arrangement -/
structure TileArrangement where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to an arrangement -/
def add_tiles (initial : TileArrangement) (added_tiles : ℕ) : TileArrangement :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter }  -- Placeholder, actual calculation depends on arrangement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles 
  (initial : TileArrangement) 
  (h1 : initial.num_tiles = 10) 
  (h2 : initial.perimeter = 20) :
  ∃ (final : TileArrangement), 
    final = add_tiles initial 2 ∧ 
    final.perimeter = 19 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2848_284829


namespace NUMINAMATH_CALUDE_man_mass_and_pressure_l2848_284885

/-- Given a boat with specified dimensions and conditions, prove the mass and pressure exerted by a man --/
theorem man_mass_and_pressure (boat_length boat_breadth sink_depth : Real)
  (supplies_mass : Real) (water_density : Real) (gravity : Real)
  (h1 : boat_length = 6)
  (h2 : boat_breadth = 3)
  (h3 : sink_depth = 0.01)
  (h4 : supplies_mass = 15)
  (h5 : water_density = 1000)
  (h6 : gravity = 9.81) :
  ∃ (man_mass : Real) (pressure : Real),
    man_mass = 165 ∧
    pressure = 89.925 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_and_pressure_l2848_284885


namespace NUMINAMATH_CALUDE_min_value_of_f_l2848_284875

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 13/5 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2848_284875


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l2848_284848

/-- A point inside a circle that is not the center of the circle -/
structure PointInsideCircle (a : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  inside : x₀^2 + y₀^2 < a^2
  not_center : (x₀, y₀) ≠ (0, 0)

/-- The line determined by the point inside the circle -/
def line_equation (a : ℝ) (p : PointInsideCircle a) (x y : ℝ) : Prop :=
  p.x₀ * x + p.y₀ * y = a^2

/-- The circle equation -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

theorem line_separate_from_circle (a : ℝ) (ha : a > 0) (p : PointInsideCircle a) :
  ∀ x y : ℝ, line_equation a p x y → ¬circle_equation a x y :=
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l2848_284848


namespace NUMINAMATH_CALUDE_same_terminal_side_470_110_l2848_284850

/-- Two angles have the same terminal side if their difference is a multiple of 360° --/
def same_terminal_side (α β : ℝ) : Prop := ∃ k : ℤ, α = β + k * 360

/-- The theorem states that 470° has the same terminal side as 110° --/
theorem same_terminal_side_470_110 : same_terminal_side 470 110 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_470_110_l2848_284850


namespace NUMINAMATH_CALUDE_periodic_function_proof_l2848_284864

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b

theorem periodic_function_proof (f : ℝ → ℝ) (c : ℝ) 
    (h1 : FunctionalEquation f) 
    (h2 : c > 0) 
    (h3 : f (c / 2) = 0) :
    ∀ x : ℝ, f (x + 2 * c) = f x := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_proof_l2848_284864


namespace NUMINAMATH_CALUDE_polygon_sides_l2848_284822

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 for a polygon
  ((n - 2) * 180 = 2 * 360) →         -- sum of interior angles = twice sum of exterior angles
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l2848_284822


namespace NUMINAMATH_CALUDE_A_eq_set_zero_one_two_l2848_284857

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_eq_set_zero_one_two : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_eq_set_zero_one_two_l2848_284857


namespace NUMINAMATH_CALUDE_breakfast_calories_l2848_284858

def total_calories : ℕ := 2200
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130
def dinner_calories : ℕ := 832

theorem breakfast_calories : 
  total_calories - lunch_calories - snack_calories - dinner_calories = 353 := by
sorry

end NUMINAMATH_CALUDE_breakfast_calories_l2848_284858


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2848_284837

/-- Given a polynomial function f(x) = ax^7 - bx^5 + cx^3 + 2, 
    prove that f(5) + f(-5) = 4 -/
theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f 5 + f (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2848_284837


namespace NUMINAMATH_CALUDE_triangle_property_l2848_284866

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.b * Real.tan t.B = Real.sqrt 3 * (t.a * Real.cos t.C + t.c * Real.cos t.A))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π / 3 ∧ t.a + t.c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2848_284866


namespace NUMINAMATH_CALUDE_complete_square_sum_l2848_284823

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℚ, 25 * x^2 + 30 * x - 45 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2848_284823


namespace NUMINAMATH_CALUDE_data_grouping_l2848_284805

theorem data_grouping (data : Set ℤ) (max_val min_val class_interval : ℤ) :
  max_val = 42 →
  min_val = 8 →
  class_interval = 5 →
  ∀ x ∈ data, min_val ≤ x ∧ x ≤ max_val →
  (max_val - min_val) / class_interval + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_data_grouping_l2848_284805


namespace NUMINAMATH_CALUDE_abs_diff_eq_one_point_one_l2848_284862

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - Int.floor x

/-- Theorem: Given the conditions, |x - y| = 1.1 -/
theorem abs_diff_eq_one_point_one (x y : ℝ) 
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.6) : 
  |x - y| = 1.1 := by
sorry

end NUMINAMATH_CALUDE_abs_diff_eq_one_point_one_l2848_284862


namespace NUMINAMATH_CALUDE_power_relation_l2848_284838

theorem power_relation (a b : ℕ) : 2^a = 8^(b+1) → 3^a / 27^b = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2848_284838


namespace NUMINAMATH_CALUDE_ten_people_prob_l2848_284854

/-- Represents the number of valid arrangements where no two adjacent people are standing
    for n people around a circular table. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing when n people
    each flip a fair coin around a circular table. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n)

/-- The main theorem stating the probability for 10 people. -/
theorem ten_people_prob : noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ten_people_prob_l2848_284854


namespace NUMINAMATH_CALUDE_inequality_proof_l2848_284827

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / (1 + x + y) < x / (1 + x) + y / (1 + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2848_284827


namespace NUMINAMATH_CALUDE_f_max_min_range_l2848_284821

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x, f a x ≤ f a x₁) ∧
  (∀ x, f a x ≥ f a x₂)

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem f_max_min_range :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l2848_284821


namespace NUMINAMATH_CALUDE_joeys_route_length_l2848_284801

theorem joeys_route_length 
  (time_one_way : ℝ) 
  (avg_speed : ℝ) 
  (return_speed : ℝ) 
  (h1 : time_one_way = 1)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 12) : 
  ∃ (route_length : ℝ), route_length = 6 ∧ 
    route_length / return_speed + time_one_way = 2 * route_length / avg_speed :=
by sorry

end NUMINAMATH_CALUDE_joeys_route_length_l2848_284801


namespace NUMINAMATH_CALUDE_afternoon_emails_l2848_284804

theorem afternoon_emails (morning evening afternoon : ℕ) : 
  morning = 5 →
  morning = afternoon + 2 →
  afternoon = 7 := by sorry

end NUMINAMATH_CALUDE_afternoon_emails_l2848_284804


namespace NUMINAMATH_CALUDE_robot_competition_max_weight_l2848_284883

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The minimum additional weight above the standard robot weight -/
def min_additional_weight : ℝ := 5

/-- The minimum weight of a robot in the competition -/
def min_robot_weight : ℝ := standard_robot_weight + min_additional_weight

/-- The maximum weight multiplier relative to the minimum weight -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition -/
def max_robot_weight : ℝ := max_weight_multiplier * min_robot_weight

theorem robot_competition_max_weight :
  max_robot_weight = 210 := by sorry

end NUMINAMATH_CALUDE_robot_competition_max_weight_l2848_284883


namespace NUMINAMATH_CALUDE_total_hours_theorem_hours_breakdown_theorem_l2848_284847

/-- The number of hours Sangita is required to fly to earn an airplane pilot certificate -/
def required_hours : ℕ := 1320

/-- The number of hours Sangita has already completed -/
def completed_hours : ℕ := 50 + 9 + 121

/-- The number of months Sangita needs to complete her goal -/
def months : ℕ := 6

/-- The number of hours Sangita must fly per month -/
def hours_per_month : ℕ := 220

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the product of months and hours per month -/
theorem total_hours_theorem :
  required_hours = months * hours_per_month :=
by sorry

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the sum of completed hours and remaining hours -/
theorem hours_breakdown_theorem :
  required_hours = completed_hours + (required_hours - completed_hours) :=
by sorry

end NUMINAMATH_CALUDE_total_hours_theorem_hours_breakdown_theorem_l2848_284847


namespace NUMINAMATH_CALUDE_fundraiser_total_l2848_284894

theorem fundraiser_total (brownie_students : Nat) (brownie_per_student : Nat) (brownie_price : Real)
                         (cookie_students : Nat) (cookie_per_student : Nat) (cookie_price : Real)
                         (donut_students : Nat) (donut_per_student : Nat) (donut_price : Real)
                         (cupcake_students : Nat) (cupcake_per_student : Nat) (cupcake_price : Real) :
  brownie_students = 70 ∧ brownie_per_student = 20 ∧ brownie_price = 1.50 ∧
  cookie_students = 40 ∧ cookie_per_student = 30 ∧ cookie_price = 2.25 ∧
  donut_students = 35 ∧ donut_per_student = 18 ∧ donut_price = 3.00 ∧
  cupcake_students = 25 ∧ cupcake_per_student = 12 ∧ cupcake_price = 2.50 →
  (brownie_students * brownie_per_student * brownie_price +
   cookie_students * cookie_per_student * cookie_price +
   donut_students * donut_per_student * donut_price +
   cupcake_students * cupcake_per_student * cupcake_price) = 7440 :=
by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l2848_284894


namespace NUMINAMATH_CALUDE_simplify_expression_l2848_284825

theorem simplify_expression (x : ℝ) : (3*x)^4 - (4*x^2)*(2*x^3) + 5*x^4 = 86*x^4 - 8*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2848_284825


namespace NUMINAMATH_CALUDE_sum_of_dimensions_for_specific_box_l2848_284817

/-- A rectangular box with dimensions A, B, and C -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sum of dimensions of a rectangular box -/
def sum_of_dimensions (box : RectangularBox) : ℝ :=
  box.A + box.B + box.C

/-- Theorem: For a rectangular box with given surface areas, the sum of its dimensions is 27.67 -/
theorem sum_of_dimensions_for_specific_box :
  ∃ (box : RectangularBox),
    box.A * box.B = 40 ∧
    box.A * box.C = 90 ∧
    box.B * box.C = 100 ∧
    sum_of_dimensions box = 27.67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_for_specific_box_l2848_284817


namespace NUMINAMATH_CALUDE_reverse_digits_difference_reverse_digits_difference_proof_l2848_284872

def is_valid_k (k : ℕ) : Prop :=
  100 < k ∧ k < 1000

def reverse_digits (k : ℕ) : ℕ :=
  let h := k / 100
  let t := (k / 10) % 10
  let u := k % 10
  100 * u + 10 * t + h

theorem reverse_digits_difference (n : ℕ) : Prop :=
  ∃ (ks : Finset ℕ), 
    ks.card = 80 ∧ 
    (∀ k ∈ ks, is_valid_k k) ∧
    (∀ k ∈ ks, reverse_digits k = k + n) →
    n = 99

-- The proof goes here
theorem reverse_digits_difference_proof : reverse_digits_difference 99 := by
  sorry

end NUMINAMATH_CALUDE_reverse_digits_difference_reverse_digits_difference_proof_l2848_284872


namespace NUMINAMATH_CALUDE_intersection_distance_through_focus_m_value_for_perpendicular_intersections_l2848_284887

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line x y m ∧ p ≠ (0, 0)}

-- Theorem 1: Distance between intersection points when line passes through focus
theorem intersection_distance_through_focus :
  ∀ m : ℝ, line 2 0 m →
  ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 2 :=
sorry

-- Theorem 2: Value of m when intersection points form right angle with origin
theorem m_value_for_perpendicular_intersections :
  ∃ m : ℝ, ∀ A B : ℝ × ℝ,
  A ∈ intersection_points m → B ∈ intersection_points m → A ≠ B →
  A.1 * B.1 + A.2 * B.2 = 0 → m = -8 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_through_focus_m_value_for_perpendicular_intersections_l2848_284887


namespace NUMINAMATH_CALUDE_absent_fraction_l2848_284861

theorem absent_fraction (total : ℕ) (present : ℕ) 
  (h1 : total = 28) 
  (h2 : present = 20) : 
  (total - present : ℚ) / total = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_absent_fraction_l2848_284861


namespace NUMINAMATH_CALUDE_no_solution_condition_l2848_284882

theorem no_solution_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔ 
  (0 < b ∧ b < 4) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2848_284882


namespace NUMINAMATH_CALUDE_sector_area_l2848_284836

/-- Given a sector with perimeter 8 and central angle 2 radians, its area is 4 -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) (arc_length : ℝ) :
  perimeter = 8 →
  central_angle = 2 →
  perimeter = arc_length + 2 * radius →
  arc_length = radius * central_angle →
  (1 / 2) * radius * arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2848_284836


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2848_284886

theorem container_volume_ratio : 
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  (8/9 : ℝ) * A = (7/9 : ℝ) * B →
  (7/9 : ℝ) * B + (1/2 : ℝ) * C = C →
  A / C = 63 / 112 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2848_284886


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_eight_cubes_l2848_284834

/-- A shape created by joining unit cubes in a line -/
structure LineCubes where
  num_cubes : ℕ

/-- Calculate the volume of the shape -/
def volume (shape : LineCubes) : ℕ :=
  shape.num_cubes

/-- Calculate the surface area of the shape -/
def surface_area (shape : LineCubes) : ℕ :=
  2 * shape.num_cubes + 2 * 4

/-- The ratio of volume to surface area for a shape with 8 unit cubes -/
theorem volume_surface_area_ratio_eight_cubes :
  let shape : LineCubes := { num_cubes := 8 }
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_volume_surface_area_ratio_eight_cubes_l2848_284834


namespace NUMINAMATH_CALUDE_foci_coordinates_l2848_284831

/-- Given that m is the geometric mean of 2 and 8, prove that the foci of x^2 + y^2/m = 1 are at (0, ±√3) -/
theorem foci_coordinates (m : ℝ) (hm_pos : m > 0) (hm_mean : m^2 = 2 * 8) :
  let equation := fun (x y : ℝ) ↦ x^2 + y^2 / m = 1
  ∃ c : ℝ, c^2 = 3 ∧ 
    (∀ x y : ℝ, equation x y ↔ equation x (-y)) ∧
    equation 0 c ∧ equation 0 (-c) :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l2848_284831


namespace NUMINAMATH_CALUDE_square_area_ratio_l2848_284877

theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2848_284877


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l2848_284808

/-- Properties of an acute triangle ABC -/
structure AcuteTriangle where
  -- Sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  side_angle_relation : Real.sqrt 3 * Real.sin C - Real.cos B = Real.cos (A - C)
  side_a : a = 2 * Real.sqrt 3
  area : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3

/-- Theorem about the properties of the specified acute triangle -/
theorem acute_triangle_properties (t : AcuteTriangle) : 
  t.A = π/3 ∧ t.b + t.c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l2848_284808


namespace NUMINAMATH_CALUDE_streetlight_problem_l2848_284884

/-- The number of streetlights --/
def n : ℕ := 2020

/-- The number of lights to be turned off --/
def k : ℕ := 300

/-- The number of ways to select k non-adjacent positions from n-2 positions --/
def non_adjacent_selections (n k : ℕ) : ℕ := Nat.choose (n - k - 1) k

theorem streetlight_problem :
  non_adjacent_selections n k = Nat.choose 1710 300 :=
sorry

end NUMINAMATH_CALUDE_streetlight_problem_l2848_284884


namespace NUMINAMATH_CALUDE_hcl_moles_equal_one_l2848_284845

-- Define the chemical reaction
structure Reaction where
  naoh : ℕ  -- moles of Sodium hydroxide
  hcl : ℕ   -- moles of Hydrochloric acid
  h2o : ℕ   -- moles of Water produced

-- Define the balanced reaction
def balanced_reaction (r : Reaction) : Prop :=
  r.naoh = r.hcl ∧ r.naoh = r.h2o

-- Theorem statement
theorem hcl_moles_equal_one (r : Reaction) 
  (h1 : r.naoh = 1)  -- 1 mole of Sodium hydroxide is used
  (h2 : r.h2o = 1)   -- The reaction produces 1 mole of Water
  (h3 : balanced_reaction r) : -- The reaction is balanced
  r.hcl = 1 := by sorry

end NUMINAMATH_CALUDE_hcl_moles_equal_one_l2848_284845


namespace NUMINAMATH_CALUDE_vector_subtraction_l2848_284874

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![2, -x]
def b : Fin 2 → ℝ := ![-1, 3]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem vector_subtraction (x : ℝ) : 
  dot_product (a x) b = 4 → 
  (λ i : Fin 2 => (a x) i - 2 * (b i)) = ![4, -4] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2848_284874


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_existence_and_uniqueness_l2848_284880

/-- Proves the existence and uniqueness of k that satisfies the given conditions --/
theorem arithmetic_sequence_squares_existence_and_uniqueness :
  ∃! k : ℤ, ∃ n d : ℤ,
    (n - d)^2 = 36 + k ∧
    n^2 = 300 + k ∧
    (n + d)^2 = 596 + k ∧
    k = 925 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_existence_and_uniqueness_l2848_284880


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l2848_284826

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def is_triple_digit (n : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 100 * a + 10 * a + a

theorem three_digit_sum_property :
  {n : ℕ | is_three_digit n ∧ is_triple_digit (n + sum_of_digits n)} =
  {105, 324, 429, 543, 648, 762, 867, 981} := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l2848_284826


namespace NUMINAMATH_CALUDE_calculator_game_sum_l2848_284807

/-- Represents the state of the three calculators -/
structure CalculatorState where
  first : ℕ
  second : ℕ
  third : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { first := state.first ^ 2,
    second := state.second ^ 3,
    third := -state.third }

/-- Applies the operations n times to the initial state -/
def nOperations (n : ℕ) : CalculatorState :=
  match n with
  | 0 => { first := 2, second := 1, third := -2 }
  | n + 1 => applyOperations (nOperations n)

theorem calculator_game_sum (N : ℕ) :
  ∃ (n : ℕ), n > 0 ∧ 
    let finalState := nOperations 50
    finalState.first = N ∧ 
    finalState.second = 1 ∧ 
    finalState.third = -2 ∧
    (finalState.first : ℤ) + finalState.second + finalState.third = N - 1 :=
  sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l2848_284807


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l2848_284899

/-- Proves that the capacity of a fuel tank is 212 gallons given specific conditions about fuel composition and volume. -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 98 + 0.16 * (C - 98) = 30) ∧ 
  C = 212 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l2848_284899


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2848_284856

theorem cubic_roots_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  (k + m = 27 ∨ k + m = 31) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2848_284856


namespace NUMINAMATH_CALUDE_factorizations_of_2079_l2848_284811

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2079

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem factorizations_of_2079 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
sorry

end NUMINAMATH_CALUDE_factorizations_of_2079_l2848_284811


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l2848_284869

theorem polynomial_perfect_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l2848_284869


namespace NUMINAMATH_CALUDE_number_division_problem_l2848_284814

theorem number_division_problem : ∃ (n : ℕ), 
  n = 220025 ∧ 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 25) := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2848_284814


namespace NUMINAMATH_CALUDE_arthur_walked_five_and_half_miles_l2848_284844

/-- The distance Arthur walked in miles -/
def arthurs_distance (east west north : ℕ) (block_length : ℚ) : ℚ :=
  (east + west + north : ℚ) * block_length

/-- Proof that Arthur walked 5.5 miles -/
theorem arthur_walked_five_and_half_miles :
  arthurs_distance 8 4 10 (1/4) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_five_and_half_miles_l2848_284844


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l2848_284806

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Define the slope of the asymptote
def AsymptopeSlope (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the eccentricity
def Eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_minimum_value (a b e : ℝ) :
  Hyperbola a b →
  AsymptopeSlope a b →
  Eccentricity a b e →
  a > 0 →
  b > 0 →
  (∀ a' b' e' : ℝ, Hyperbola a' b' → AsymptopeSlope a' b' → Eccentricity a' b' e' →
    a' > 0 → b' > 0 → (a'^2 + e') / b' ≥ (a^2 + e) / b) →
  (a^2 + e) / b = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l2848_284806


namespace NUMINAMATH_CALUDE_locus_of_Q_l2848_284889

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the points and their properties
def SymmetricalPoints (O A B : ℝ × ℝ) := A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

-- Define the perpendicular chord
def PerpendicularChord (P P' A : ℝ × ℝ) := 
  (P'.1 - P.1) * (A.1 - P.1) + (P'.2 - P.2) * (A.2 - P.2) = 0

-- Define the symmetric point C
def SymmetricPoint (B C PP' : ℝ × ℝ) := 
  (C.1 - PP'.1) = (PP'.1 - B.1) ∧ (C.2 - PP'.2) = (PP'.2 - B.2)

-- Define the intersection point Q
def IntersectionPoint (Q PP' A C : ℝ × ℝ) := 
  (Q.1 - PP'.1) * (C.2 - A.2) = (Q.2 - PP'.2) * (C.1 - A.1) ∧
  (Q.1 - A.1) * (C.2 - A.2) = (Q.2 - A.2) * (C.1 - A.1)

-- Theorem statement
theorem locus_of_Q (O : ℝ × ℝ) (r : ℝ) (A B P P' C Q : ℝ × ℝ) :
  P ∈ Circle O r →
  SymmetricalPoints O A B →
  PerpendicularChord P P' A →
  SymmetricPoint B C P →
  IntersectionPoint Q P A C →
  ∃ (a b : ℝ), 
    (Q.1 / a)^2 + (Q.2 / b)^2 = 1 ∧
    a^2 - b^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧
    a = r :=
by sorry

end NUMINAMATH_CALUDE_locus_of_Q_l2848_284889


namespace NUMINAMATH_CALUDE_abc_system_solution_l2848_284800

theorem abc_system_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 3 * (a + b))
  (hbc : b * c = 4 * (b + c))
  (hac : a * c = 5 * (a + c)) :
  a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_abc_system_solution_l2848_284800


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2848_284891

theorem unique_solution_quadratic_system (y : ℚ) 
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1/9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2848_284891


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2848_284849

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 98 / Real.sqrt 32) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2848_284849


namespace NUMINAMATH_CALUDE_leading_zeros_of_fraction_l2848_284855

/-- The number of leading zeros in the decimal representation of a fraction -/
def leadingZeros (n d : ℕ) : ℕ :=
  sorry

theorem leading_zeros_of_fraction :
  leadingZeros 1 (2^3 * 5^5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_leading_zeros_of_fraction_l2848_284855


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2848_284896

/-- The roots of the quadratic equation x^2 - 7x + 10 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 10 = 0}

/-- An isosceles triangle with two sides equal to the roots of x^2 - 7x + 10 = 0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∨ b = c ∨ a = c
  rootSides : a ∈ roots ∧ b ∈ roots

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of the isosceles triangle is 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 12 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2848_284896


namespace NUMINAMATH_CALUDE_garden_perimeter_l2848_284863

/-- The perimeter of a rectangular garden with length 205 m and breadth 95 m is 600 m. -/
theorem garden_perimeter : 
  ∀ (perimeter length breadth : ℕ), 
    length = 205 → 
    breadth = 95 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 600 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2848_284863


namespace NUMINAMATH_CALUDE_printing_completion_time_l2848_284867

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight
def quarter_completion_time : ℕ := 12 * 60 + 30  -- 12:30 PM in minutes since midnight

-- Time taken to complete one-fourth of the job in minutes
def quarter_job_duration : ℕ := quarter_completion_time - start_time

-- Total time required to complete the entire job in minutes
def total_job_duration : ℕ := 4 * quarter_job_duration

-- Completion time in minutes since midnight
def completion_time : ℕ := start_time + total_job_duration

theorem printing_completion_time :
  completion_time = 23 * 60 := by sorry

end NUMINAMATH_CALUDE_printing_completion_time_l2848_284867


namespace NUMINAMATH_CALUDE_equation_implies_ratio_one_third_l2848_284840

theorem equation_implies_ratio_one_third 
  (a x y : ℝ) 
  (h_distinct : a ≠ x ∧ x ≠ y ∧ a ≠ y) 
  (h_eq : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_ratio_one_third_l2848_284840
