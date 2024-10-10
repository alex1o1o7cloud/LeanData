import Mathlib

namespace intersection_empty_implies_a_values_solution_set_correct_l1303_130308

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values (a : ℝ) :
  M ∩ N a = ∅ → a = -6 ∨ a = -2 := by
  sorry

-- Define the solution set
def solution_set : Set ℝ := {-6, -2}

-- State the theorem for the solution set
theorem solution_set_correct :
  ∀ a : ℝ, (M ∩ N a = ∅) → a ∈ solution_set := by
  sorry

end intersection_empty_implies_a_values_solution_set_correct_l1303_130308


namespace expressions_correctness_l1303_130304

theorem expressions_correctness (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) :
  (∃ x : ℝ, x * x = a / b) ∧ 
  (∃ y : ℝ, y * y = b / a) ∧
  (∃ z : ℝ, z * z = a * b) ∧
  (∃ w : ℝ, w * w = a / b) ∧
  (Real.sqrt (a / b) * Real.sqrt (b / a) = 1) ∧
  (Real.sqrt (a * b) / Real.sqrt (a / b) = -b) := by
  sorry

end expressions_correctness_l1303_130304


namespace line_parameterization_l1303_130337

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) → 
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end line_parameterization_l1303_130337


namespace total_exercise_time_l1303_130329

def natasha_daily_exercise : ℕ := 30
def natasha_days : ℕ := 7
def esteban_daily_exercise : ℕ := 10
def esteban_days : ℕ := 9
def minutes_per_hour : ℕ := 60

theorem total_exercise_time :
  (natasha_daily_exercise * natasha_days + esteban_daily_exercise * esteban_days) / minutes_per_hour = 5 := by
  sorry

end total_exercise_time_l1303_130329


namespace five_player_tournament_l1303_130393

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players where each player plays against every other player
    exactly once, the total number of games played is 10. -/
theorem five_player_tournament : tournament_games 5 = 10 := by
  sorry

end five_player_tournament_l1303_130393


namespace parabola_coefficients_l1303_130323

/-- A parabola with equation y = ax^2 + bx + c, vertex at (5, -1), 
    vertical axis of symmetry, and passing through (2, 8) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (a * 5^2 + b * 5 + c = -1) ∧
  (∀ x : ℝ, a * (x - 5)^2 + (a * 5^2 + b * 5 + c) = a * x^2 + b * x + c) ∧
  (a * 2^2 + b * 2 + c = 8)

/-- The values of a, b, and c for the given parabola are 1, -10, and 24 respectively -/
theorem parabola_coefficients : 
  ∃ a b c : ℝ, Parabola a b c ∧ a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end parabola_coefficients_l1303_130323


namespace beta_conditions_l1303_130368

theorem beta_conditions (β : ℂ) (h1 : β ≠ -1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  Complex.abs (β^3 + 1) = 3 ∧ Complex.abs (β^6 + 1) = 3 := by
  sorry

end beta_conditions_l1303_130368


namespace expression_equality_l1303_130311

theorem expression_equality : 
  4 * (Real.sin (π / 3)) + (1 / 2)⁻¹ - Real.sqrt 12 + |(-3)| = 5 := by
  sorry

end expression_equality_l1303_130311


namespace geometric_sequence_common_ratio_l1303_130387

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_a4 : a 4 = 4)
  (h_a6 : a 6 = 16) :
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ+, a (n + 1) = a n * q := by
  sorry

end geometric_sequence_common_ratio_l1303_130387


namespace mollys_gift_cost_l1303_130347

/-- Represents the cost and family structure for Molly's gift-sending problem -/
structure GiftSendingProblem where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_sisters : ℕ
  children_per_brother : ℕ
  children_of_sister : ℕ
  num_grandparents : ℕ
  num_cousins : ℕ

/-- Calculates the total number of packages to be sent -/
def total_packages (p : GiftSendingProblem) : ℕ :=
  p.num_parents + p.num_brothers + p.num_sisters +
  (p.num_brothers * p.children_per_brother) +
  p.children_of_sister + p.num_grandparents + p.num_cousins

/-- Calculates the total cost of sending all packages -/
def total_cost (p : GiftSendingProblem) : ℕ :=
  p.cost_per_package * total_packages p

/-- Theorem stating that the total cost for Molly's specific situation is $182 -/
theorem mollys_gift_cost :
  let p : GiftSendingProblem := {
    cost_per_package := 7,
    num_parents := 2,
    num_brothers := 4,
    num_sisters := 1,
    children_per_brother := 3,
    children_of_sister := 2,
    num_grandparents := 2,
    num_cousins := 3
  }
  total_cost p = 182 := by sorry

end mollys_gift_cost_l1303_130347


namespace other_solution_quadratic_l1303_130367

theorem other_solution_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 25 = 77 * (3/4) + 4) → 
  (48 * x^2 + 25 = 77 * x + 4) → 
  x = 3/4 ∨ x = 7/12 := by
sorry

end other_solution_quadratic_l1303_130367


namespace inequalities_solution_l1303_130317

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x - 3 * (x - 2) > 4
def inequality2 (x : ℝ) : Prop := (2 * x - 1) / 3 ≤ (x + 1) / 2

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem inequalities_solution :
  ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ solution_set x :=
by sorry

end inequalities_solution_l1303_130317


namespace other_person_age_l1303_130343

/-- Given two people where one (Marco) is 1 year older than twice the age of the other,
    and the sum of their ages is 37, prove that the younger person is 12 years old. -/
theorem other_person_age (x : ℕ) : x + (2 * x + 1) = 37 → x = 12 := by
  sorry

end other_person_age_l1303_130343


namespace min_value_sum_of_products_l1303_130355

theorem min_value_sum_of_products (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 :=
by sorry

end min_value_sum_of_products_l1303_130355


namespace a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1303_130324

theorem a_gt_one_sufficient_not_necessary_for_a_gt_zero :
  (∃ a : ℝ, a > 0 ∧ ¬(a > 1)) ∧
  (∀ a : ℝ, a > 1 → a > 0) :=
sorry

end a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1303_130324


namespace equation_positive_root_m_value_l1303_130348

theorem equation_positive_root_m_value (m x : ℝ) : 
  (m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) → 
  (x > 0) → 
  (m = 6 ∨ m = 12) :=
by sorry

end equation_positive_root_m_value_l1303_130348


namespace map_distance_conversion_l1303_130383

/-- Calculates the actual distance given map distance and scale --/
def actual_distance (map_distance : ℝ) (map_scale : ℝ) : ℝ :=
  map_distance * map_scale

/-- Theorem: Given a map scale where 312 inches represent 136 km,
    a distance of 25 inches on the map corresponds to approximately 10.897425 km
    in actual distance. --/
theorem map_distance_conversion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (actual_dist : ℝ),
    abs (actual_distance 25 (136 / 312) - actual_dist) < ε ∧
    abs (actual_dist - 10.897425) < ε :=
by sorry

end map_distance_conversion_l1303_130383


namespace unique_cube_labeling_l1303_130316

/-- A cube labeling is a function from vertices to integers -/
def CubeLabeling := Fin 8 → Fin 8

/-- A face of the cube is a set of four vertices -/
def CubeFace := Finset (Fin 8)

/-- The set of all faces of a cube -/
def allFaces : Finset CubeFace := sorry

/-- A labeling is valid if it's a bijection (each number used once) -/
def isValidLabeling (l : CubeLabeling) : Prop :=
  Function.Bijective l

/-- The sum of labels on a face equals 22 -/
def faceSum22 (l : CubeLabeling) (face : CubeFace) : Prop :=
  (face.sum (λ v => (l v).val + 1) : ℕ) = 22

/-- All faces of a labeling sum to 22 -/
def allFacesSum22 (l : CubeLabeling) : Prop :=
  ∀ face ∈ allFaces, faceSum22 l face

/-- Two labelings are equivalent if they can be obtained by flipping the cube -/
def equivalentLabelings (l₁ l₂ : CubeLabeling) : Prop := sorry

/-- The main theorem: there is only one unique labeling up to equivalence -/
theorem unique_cube_labeling :
  ∃! l : CubeLabeling, isValidLabeling l ∧ allFacesSum22 l := by sorry

end unique_cube_labeling_l1303_130316


namespace quadratic_equation_roots_l1303_130356

theorem quadratic_equation_roots (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let sum_roots := -b / a
  let prod_roots := c / a
  let new_sum := sum_roots + prod_roots
  let new_prod := sum_roots * prod_roots
  f 0 = 0 →
  (∃ x y : ℝ, x + y = new_sum ∧ x * y = new_prod) →
  ∃ k : ℝ, k ≠ 0 ∧ f = λ x => k * (x^2 - new_sum * x + new_prod) :=
by sorry

end quadratic_equation_roots_l1303_130356


namespace max_servings_emily_l1303_130381

/-- Represents the recipe for the smoothie --/
structure Recipe :=
  (servings : ℕ)
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)

def recipe : Recipe :=
  { servings := 8
  , bananas := 3
  , strawberries := 2
  , yogurt := 1
  , honey := 4 }

def emily : Available :=
  { bananas := 9
  , strawberries := 8
  , yogurt := 3 }

/-- Calculates the maximum number of servings that can be made --/
def maxServings (r : Recipe) (a : Available) : ℕ :=
  min (a.bananas * r.servings / r.bananas)
      (min (a.strawberries * r.servings / r.strawberries)
           (a.yogurt * r.servings / r.yogurt))

theorem max_servings_emily :
  maxServings recipe emily = 24 := by
  sorry

end max_servings_emily_l1303_130381


namespace quadratic_always_positive_l1303_130359

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end quadratic_always_positive_l1303_130359


namespace divisibility_conditions_l1303_130318

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, (a^3 * b - 1) = k * (a + 1)) ∧ 
  (∃ m : ℤ, (a * b^3 + 1) = m * (b - 1)) ↔ 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
sorry

end divisibility_conditions_l1303_130318


namespace surface_area_increase_after_cube_removal_l1303_130341

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Theorem: Removing a 1-foot cube from a 4×3×5 feet rectangular solid increases surface area by 2 sq ft -/
theorem surface_area_increase_after_cube_removal 
  (original : RectangularSolid) 
  (removed : Cube) 
  (h1 : original.length = 4)
  (h2 : original.width = 3)
  (h3 : original.height = 5)
  (h4 : removed.side = 1)
  (h5 : removed.side < original.length ∧ removed.side < original.width ∧ removed.side < original.height) :
  surfaceArea original + 2 = surfaceArea original + 
    (removed.side * removed.side + 2 * removed.side * removed.side) - removed.side * removed.side := by
  sorry

end surface_area_increase_after_cube_removal_l1303_130341


namespace polar_to_cartesian_line_l1303_130369

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (Real.sin θ + Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    line_equation x y :=
by sorry

end polar_to_cartesian_line_l1303_130369


namespace cos_value_for_special_angle_l1303_130398

theorem cos_value_for_special_angle (θ : Real) 
  (h1 : 6 * Real.tan θ = 2 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = -1 := by
  sorry

end cos_value_for_special_angle_l1303_130398


namespace subset_star_inclusion_l1303_130388

/-- Given non-empty sets of real numbers M and P, where M ⊆ P, prove that P* ⊆ M* -/
theorem subset_star_inclusion {M P : Set ℝ} (hM : M.Nonempty) (hP : P.Nonempty) (h_subset : M ⊆ P) :
  {y : ℝ | ∀ x ∈ P, y ≥ x} ⊆ {y : ℝ | ∀ x ∈ M, y ≥ x} := by
  sorry

end subset_star_inclusion_l1303_130388


namespace middle_number_is_four_or_five_l1303_130373

/-- Represents a triple of positive integers -/
structure IntTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if the triple satisfies all given conditions -/
def satisfiesConditions (t : IntTriple) : Prop :=
  t.a < t.b ∧ t.b < t.c ∧ t.a + t.b + t.c = 15

/-- Represents the set of all possible triples satisfying the conditions -/
def possibleTriples : Set IntTriple :=
  {t : IntTriple | satisfiesConditions t}

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.a = t.a ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.c = t.c ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.b = t.b ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 or 5 -/
theorem middle_number_is_four_or_five :
  ∀ t ∈ possibleTriples,
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.b = 4 ∨ t.b = 5 :=
sorry

end middle_number_is_four_or_five_l1303_130373


namespace candy_bars_per_box_l1303_130314

/-- Proves that the number of candy bars in each box is 10 given the specified conditions --/
theorem candy_bars_per_box 
  (num_boxes : ℕ) 
  (selling_price buying_price : ℚ)
  (total_profit : ℚ)
  (h1 : num_boxes = 5)
  (h2 : selling_price = 3/2)
  (h3 : buying_price = 1)
  (h4 : total_profit = 25) :
  (total_profit / (num_boxes * (selling_price - buying_price))) = 10 := by
  sorry


end candy_bars_per_box_l1303_130314


namespace min_value_of_sequence_l1303_130328

/-- A positive arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem min_value_of_sequence (a : ℕ → ℝ) (m n : ℕ) :
  ArithmeticGeometricSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end min_value_of_sequence_l1303_130328


namespace repeating_decimal_equals_fraction_l1303_130301

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatingLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / ((10 ^ x.repeatingLength - 1) : ℚ)

/-- The repeating decimal 7.036036036... -/
def number : RepeatingDecimal :=
  { integerPart := 7
    repeatingPart := 36
    repeatingLength := 3 }

theorem repeating_decimal_equals_fraction :
  toRational number = 781 / 111 := by
  sorry

end repeating_decimal_equals_fraction_l1303_130301


namespace toy_box_paths_l1303_130386

/-- Represents a rectangular grid --/
structure Grid :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of paths in a grid from one corner to the opposite corner,
    moving only right and up, covering a specific total distance --/
def numPaths (g : Grid) (totalDistance : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for a 50x40 grid with total distance 90,
    there are 12 possible paths --/
theorem toy_box_paths :
  let g : Grid := { length := 50, width := 40 }
  numPaths g 90 = 12 := by
  sorry

end toy_box_paths_l1303_130386


namespace seashell_difference_l1303_130363

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of cracked seashells -/
def cracked_seashells : ℕ := 29

/-- Theorem stating the difference between Fred's and Tom's seashell counts -/
theorem seashell_difference : fred_seashells - tom_seashells = 28 := by
  sorry

end seashell_difference_l1303_130363


namespace max_value_a4a7_l1303_130339

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The maximum value of a_4 * a_7 in an arithmetic sequence where a_6 = 4 -/
theorem max_value_a4a7 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h6 : a 6 = 4) :
  (∀ d : ℝ, a 4 * a 7 ≤ 18) ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end max_value_a4a7_l1303_130339


namespace rectangle_exists_l1303_130340

/-- A list of the given square side lengths -/
def square_sides : List ℕ := [2, 5, 7, 9, 16, 25, 28, 33, 36]

/-- The total area covered by all squares -/
def total_area : ℕ := (square_sides.map (λ x => x * x)).sum

/-- Proposition: There exists a rectangle with integer dimensions that can be tiled by the given squares -/
theorem rectangle_exists : ∃ (length width : ℕ), 
  length * width = total_area ∧ 
  length > 0 ∧ 
  width > 0 :=
sorry

end rectangle_exists_l1303_130340


namespace angle_ABC_measure_l1303_130376

theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  angle_CBD = 90 →
  angle_ABC + angle_ABD + angle_CBD = 270 →
  angle_ABD = 100 →
  angle_ABC = 80 := by
sorry

end angle_ABC_measure_l1303_130376


namespace one_fourth_more_than_32_5_l1303_130310

theorem one_fourth_more_than_32_5 : (1 / 4 : ℚ) + 32.5 = 32.75 := by
  sorry

end one_fourth_more_than_32_5_l1303_130310


namespace r_fourth_plus_inverse_r_fourth_l1303_130327

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end r_fourth_plus_inverse_r_fourth_l1303_130327


namespace eulers_formula_l1303_130375

/-- A polyhedron with S vertices, A edges, and F faces, where no four vertices are coplanar. -/
structure Polyhedron where
  S : ℕ  -- number of vertices
  A : ℕ  -- number of edges
  F : ℕ  -- number of faces
  no_four_coplanar : True  -- represents the condition that no four vertices are coplanar

/-- Euler's formula for polyhedra -/
theorem eulers_formula (p : Polyhedron) : p.S + p.F = p.A + 2 := by
  sorry

end eulers_formula_l1303_130375


namespace housing_price_growth_equation_l1303_130352

/-- Proves that the equation for average annual growth rate of housing prices is correct -/
theorem housing_price_growth_equation (initial_price final_price : ℝ) (growth_rate : ℝ) 
  (h1 : initial_price = 8100)
  (h2 : final_price = 12500)
  (h3 : growth_rate ≥ 0)
  (h4 : growth_rate < 1) :
  initial_price * (1 + growth_rate)^2 = final_price := by
  sorry

end housing_price_growth_equation_l1303_130352


namespace x_eq_one_sufficient_not_necessary_for_x_gt_zero_l1303_130371

theorem x_eq_one_sufficient_not_necessary_for_x_gt_zero :
  (∃ x : ℝ, x = 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ x ≠ 1) := by
  sorry

end x_eq_one_sufficient_not_necessary_for_x_gt_zero_l1303_130371


namespace quadratic_rewrite_l1303_130354

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 4) → 
  b = 8 * Real.sqrt 2 := by
sorry

end quadratic_rewrite_l1303_130354


namespace cube_sum_magnitude_l1303_130395

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 14)
  (h3 : Complex.abs (w - 2*z) = 2) :
  Complex.abs (w^3 + z^3) = 38 := by
  sorry

end cube_sum_magnitude_l1303_130395


namespace S_has_maximum_l1303_130396

def S (n : ℕ+) : ℤ := -2 * n.val ^ 3 + 21 * n.val ^ 2 + 23 * n.val

theorem S_has_maximum : ∃ (m : ℕ+), ∀ (n : ℕ+), S n ≤ S m ∧ S m = 504 := by
  sorry

end S_has_maximum_l1303_130396


namespace paper_cutting_game_l1303_130380

theorem paper_cutting_game (n : ℕ) : 
  (8 * n + 1 = 2009) ↔ (n = 251) :=
by sorry

#check paper_cutting_game

end paper_cutting_game_l1303_130380


namespace upstream_speed_calculation_l1303_130377

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 40) 
  (h2 : s.downstream = 60) : 
  upstreamSpeed s = 20 := by
  sorry

#check upstream_speed_calculation

end upstream_speed_calculation_l1303_130377


namespace sqrt_x_div_sqrt_y_l1303_130370

theorem sqrt_x_div_sqrt_y (x y : ℝ) : 
  (((1/3)^2 + (1/4)^2) / ((1/5)^2 + (1/6)^2) = 25*x/(61*y)) → 
  Real.sqrt x / Real.sqrt y = 5/2 := by
sorry

end sqrt_x_div_sqrt_y_l1303_130370


namespace draw_probability_l1303_130346

/-- The probability of player A winning a chess game -/
def prob_A_wins : ℝ := 0.4

/-- The probability that player A does not lose a chess game -/
def prob_A_not_lose : ℝ := 0.9

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem draw_probability :
  prob_draw = 0.5 :=
sorry

end draw_probability_l1303_130346


namespace mystery_book_shelves_l1303_130385

theorem mystery_book_shelves (total_books : ℕ) (books_per_shelf : ℕ) (picture_shelves : ℕ) :
  total_books = 72 →
  books_per_shelf = 9 →
  picture_shelves = 5 →
  (total_books - picture_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end mystery_book_shelves_l1303_130385


namespace coffee_drinkers_possible_values_l1303_130335

def round_table_coffee_problem (n : ℕ) (coffee_drinkers : ℕ) : Prop :=
  n = 14 ∧
  0 < coffee_drinkers ∧
  coffee_drinkers < n ∧
  ∃ (k : ℕ), k > 0 ∧ k < n/2 ∧ coffee_drinkers = n - 2*k

theorem coffee_drinkers_possible_values :
  ∀ (n : ℕ) (coffee_drinkers : ℕ),
    round_table_coffee_problem n coffee_drinkers →
    coffee_drinkers = 6 ∨ coffee_drinkers = 8 ∨ coffee_drinkers = 10 ∨ coffee_drinkers = 12 :=
by sorry

end coffee_drinkers_possible_values_l1303_130335


namespace smallest_m_for_candies_l1303_130390

theorem smallest_m_for_candies : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬(10 ∣ 15*k ∧ 18 ∣ 15*k ∧ 20 ∣ 15*k)) ∧
  (10 ∣ 15*m ∧ 18 ∣ 15*m ∧ 20 ∣ 15*m) ∧ m = 12 := by
  sorry

end smallest_m_for_candies_l1303_130390


namespace smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l1303_130344

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 400 * x % 576 = 0 → x ≥ 36 :=
  sorry

theorem thirty_six_satisfies : 400 * 36 % 576 = 0 :=
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 400 * x % 576 = 0 ∧ x = 36 :=
  sorry

end smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l1303_130344


namespace parallel_vectors_x_value_l1303_130313

/-- Given two parallel 2D vectors a and b, where a = (2, 3) and b = (x, 6), prove that x = 4. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 4 := by
sorry

end parallel_vectors_x_value_l1303_130313


namespace neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l1303_130379

-- Define the color type
inductive Color : Type
  | Black : Color
  | Red : Color
  | White : Color

-- Define the coloring function
def coloring : ℤ → Color := sorry

-- Define the coloring rules
axiom neg_black_is_white : ∀ n : ℤ, coloring n = Color.Black → coloring (-n) = Color.White
axiom sum_white_is_black : ∀ a b : ℤ, coloring a = Color.White → coloring b = Color.White → coloring (a + b) = Color.Black

-- Theorems to prove
theorem neg_white_is_black : ∀ n : ℤ, coloring n = Color.White → coloring (-n) = Color.Black := sorry

theorem sum_black_is_white : ∀ a b : ℤ, coloring a = Color.Black → coloring b = Color.Black → coloring (a + b) = Color.White := sorry

theorem zero_is_red : coloring 0 = Color.Red := sorry

theorem nonzero_black_or_white : ∀ n : ℤ, n ≠ 0 → (coloring n = Color.Black ∨ coloring n = Color.White) := sorry

theorem neg_opposite_color : ∀ n : ℤ, n ≠ 0 → 
  (coloring n = Color.Black → coloring (-n) = Color.White) ∧ 
  (coloring n = Color.White → coloring (-n) = Color.Black) := sorry

end neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l1303_130379


namespace fathers_age_when_sum_is_100_l1303_130330

/-- Given a mother aged 42 and a father aged 44, prove that the father will be 51 years old when the sum of their ages is 100. -/
theorem fathers_age_when_sum_is_100 (mother_age father_age : ℕ) 
  (h1 : mother_age = 42) 
  (h2 : father_age = 44) : 
  ∃ (years : ℕ), mother_age + years + (father_age + years) = 100 ∧ father_age + years = 51 := by
  sorry

end fathers_age_when_sum_is_100_l1303_130330


namespace cookie_theorem_l1303_130399

def cookie_problem (initial_cookies eaten_cookies bought_cookies : ℕ) : Prop :=
  eaten_cookies - bought_cookies = 2

theorem cookie_theorem (initial_cookies : ℕ) : 
  cookie_problem initial_cookies 5 3 :=
by
  sorry

end cookie_theorem_l1303_130399


namespace robbery_participants_l1303_130391

-- Define the suspects
variable (Alexey Boris Veniamin Grigory : Prop)

-- Define the conditions
axiom condition1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom condition2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom condition3 : Grigory → Boris
axiom condition4 : Boris → (Alexey ∨ Veniamin)

-- Theorem to prove
theorem robbery_participants :
  Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin :=
sorry

end robbery_participants_l1303_130391


namespace four_bb_two_divisible_by_nine_l1303_130397

theorem four_bb_two_divisible_by_nine :
  ∃! (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end four_bb_two_divisible_by_nine_l1303_130397


namespace cabbage_area_l1303_130338

theorem cabbage_area (garden_area_this_year garden_area_last_year : ℝ) 
  (cabbages_this_year cabbages_last_year : ℕ) :
  (garden_area_this_year = cabbages_this_year) →
  (garden_area_this_year = garden_area_last_year + 199) →
  (cabbages_this_year = 10000) →
  (∃ x y : ℝ, garden_area_last_year = x^2 ∧ garden_area_this_year = y^2) →
  (garden_area_this_year / cabbages_this_year = 1) :=
by
  sorry

end cabbage_area_l1303_130338


namespace smallest_four_digit_divisible_by_square_digits_l1303_130300

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by the square of each of its digits -/
def divisible_by_square_of_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % (d * d) = 0

theorem smallest_four_digit_divisible_by_square_digits :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    has_different_digits n →
    divisible_by_square_of_digits n →
    2268 ≤ n :=
sorry

end smallest_four_digit_divisible_by_square_digits_l1303_130300


namespace total_slices_is_seven_l1303_130312

/-- The number of slices of pie sold yesterday -/
def slices_yesterday : ℕ := 5

/-- The number of slices of pie served today -/
def slices_today : ℕ := 2

/-- The total number of slices of pie sold -/
def total_slices : ℕ := slices_yesterday + slices_today

theorem total_slices_is_seven : total_slices = 7 := by
  sorry

end total_slices_is_seven_l1303_130312


namespace regular_nonagon_perimeter_l1303_130378

/-- A regular polygon with 9 sides, each 2 centimeters long -/
structure RegularNonagon where
  side_length : ℝ
  num_sides : ℕ
  h1 : side_length = 2
  h2 : num_sides = 9

/-- The perimeter of a regular nonagon -/
def perimeter (n : RegularNonagon) : ℝ :=
  n.side_length * n.num_sides

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem regular_nonagon_perimeter (n : RegularNonagon) : perimeter n = 18 := by
  sorry

#check regular_nonagon_perimeter

end regular_nonagon_perimeter_l1303_130378


namespace certain_number_l1303_130306

theorem certain_number (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 := by
  sorry

end certain_number_l1303_130306


namespace class_ratio_proof_l1303_130382

theorem class_ratio_proof (B G : ℝ) 
  (h1 : B > 0) 
  (h2 : G > 0) 
  (h3 : 0.80 * B + 0.75 * G = 0.78 * (B + G)) : 
  B / G = 3 / 2 := by
  sorry

end class_ratio_proof_l1303_130382


namespace number_of_fives_l1303_130362

theorem number_of_fives (x y : ℕ) : 
  x + y = 20 →
  3 * x + 5 * y = 94 →
  y = 17 := by
sorry

end number_of_fives_l1303_130362


namespace carries_profit_l1303_130302

/-- Carrie's profit from making and decorating a wedding cake -/
theorem carries_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  supply_cost = 54 →
  (hours_per_day * days_worked * hourly_rate - supply_cost : ℕ) = 122 := by
  sorry

end carries_profit_l1303_130302


namespace size_and_precision_difference_l1303_130332

/-- Represents the precision of a number -/
inductive Precision
  | Ones
  | Tenths

/-- Represents a number with its value and precision -/
structure NumberWithPrecision where
  value : ℝ
  precision : Precision

/-- The statement that the size and precision of 3.0 and 3 are the same is false -/
theorem size_and_precision_difference : ∃ (a b : NumberWithPrecision), 
  a.value = b.value ∧ a.precision ≠ b.precision := by
  sorry

/-- The numerical value of 3.0 equals 3 -/
axiom value_equality : ∃ (a b : NumberWithPrecision), 
  a.value = 3 ∧ b.value = 3 ∧ a.value = b.value

/-- The precision of 3.0 is to the tenth -/
axiom precision_three_point_zero : ∃ (a : NumberWithPrecision), 
  a.value = 3 ∧ a.precision = Precision.Tenths

/-- The precision of 3 is to 1 -/
axiom precision_three : ∃ (b : NumberWithPrecision), 
  b.value = 3 ∧ b.precision = Precision.Ones

end size_and_precision_difference_l1303_130332


namespace team_selection_proof_l1303_130319

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 5 players from a team of 9 players, 
    where 2 seeded players must be included -/
def teamSelection : ℕ := sorry

theorem team_selection_proof :
  let totalPlayers : ℕ := 9
  let seededPlayers : ℕ := 2
  let selectCount : ℕ := 5
  teamSelection = choose (totalPlayers - seededPlayers) (selectCount - seededPlayers) := by
  sorry

end team_selection_proof_l1303_130319


namespace trigonometric_equation_equivalence_l1303_130309

theorem trigonometric_equation_equivalence (α : ℝ) : 
  (1 - 2 * (Real.cos α) ^ 2) / (2 * Real.tan (2 * α - π / 4) * (Real.sin (π / 4 + 2 * α)) ^ 2) = 
  -(Real.cos (2 * α)) / ((Real.cos (2 * α - π / 4) + Real.sin (2 * α - π / 4)) ^ 2) := by
sorry

end trigonometric_equation_equivalence_l1303_130309


namespace smallest_number_divisibility_l1303_130336

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 3668 → 
    ¬((y + 7) % 25 = 0 ∧ (y + 7) % 49 = 0 ∧ (y + 7) % 15 = 0 ∧ (y + 7) % 21 = 0)) ∧
  ((3668 + 7) % 25 = 0 ∧ (3668 + 7) % 49 = 0 ∧ (3668 + 7) % 15 = 0 ∧ (3668 + 7) % 21 = 0) :=
by sorry

end smallest_number_divisibility_l1303_130336


namespace divides_n_squared_plus_2n_plus_27_l1303_130345

theorem divides_n_squared_plus_2n_plus_27 (n : ℕ) :
  n ∣ (n^2 + 2*n + 27) ↔ n = 1 ∨ n = 3 ∨ n = 9 ∨ n = 27 := by
  sorry

end divides_n_squared_plus_2n_plus_27_l1303_130345


namespace f_range_implies_a_range_l1303_130360

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x + 1| - |4*x + a|

-- State the theorem
theorem f_range_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ -5) → a ∈ Set.Iic (-4) ∪ Set.Ici 6 :=
by sorry

end f_range_implies_a_range_l1303_130360


namespace min_cuts_to_touch_coin_l1303_130334

/-- Represents a circular object with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a straight cut on the pancake -/
structure Cut where
  width : ℝ

/-- The pancake -/
def pancake : Circle := { radius := 10 }

/-- The coin -/
def coin : Circle := { radius := 1 }

/-- The width of the area covered by a single cut -/
def cut_width : ℝ := 2

/-- The minimum number of cuts needed -/
def min_cuts : ℕ := 10

theorem min_cuts_to_touch_coin : 
  ∀ (cuts : ℕ), 
    cuts < min_cuts → 
    ∃ (coin_position : ℝ × ℝ), 
      coin_position.1^2 + coin_position.2^2 ≤ pancake.radius^2 ∧ 
      ∀ (cut : Cut), cut.width = cut_width → 
        ∃ (d : ℝ), d > coin.radius ∧ 
          ∀ (p : ℝ × ℝ), p.1^2 + p.2^2 ≤ coin.radius^2 → 
            (p.1 - coin_position.1)^2 + (p.2 - coin_position.2)^2 ≤ d^2 := by
  sorry

#check min_cuts_to_touch_coin

end min_cuts_to_touch_coin_l1303_130334


namespace find_k_value_l1303_130389

/-- Given two functions f and g, prove that if f(5) - g(5) = 12, then k = -53/5 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 8)
  (hg : ∀ x, g x = x^2 - k * x + 3)
  (h_diff : f 5 - g 5 = 12) : 
  k = -53/5 := by sorry

end find_k_value_l1303_130389


namespace wire_length_difference_l1303_130331

theorem wire_length_difference (total_length piece1 piece2 : ℝ) : 
  total_length = 30 →
  piece1 = 14 →
  piece2 = 16 →
  |piece2 - piece1| = 2 := by sorry

end wire_length_difference_l1303_130331


namespace algebraic_expression_equality_l1303_130307

theorem algebraic_expression_equality (x : ℝ) (h : x = 5) :
  3 / (x - 4) - 24 / (x^2 - 16) = 1 / 3 := by
  sorry

end algebraic_expression_equality_l1303_130307


namespace xy_power_2023_l1303_130358

theorem xy_power_2023 (x y : ℝ) (h : |x + 1| + Real.sqrt (y - 1) = 0) : 
  (x * y) ^ 2023 = -1 := by
  sorry

end xy_power_2023_l1303_130358


namespace sufficient_not_necessary_l1303_130384

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x + y = 1 → x * y ≤ 1/4) ∧
  (∃ x y : ℝ, x * y ≤ 1/4 ∧ x + y ≠ 1) := by
  sorry

end sufficient_not_necessary_l1303_130384


namespace petya_winning_strategy_l1303_130322

/-- Represents the state of cups on a 2n-gon -/
def CupState (n : ℕ) := Fin (2 * n) → Bool

/-- Checks if two positions are adjacent on a 2n-gon -/
def adjacent (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + 1) % (2 * n) = j.val ∨ (j.val + 1) % (2 * n) = i.val

/-- Checks if two positions are symmetric with respect to the center of a 2n-gon -/
def symmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + n) % (2 * n) = j.val

/-- Checks if a move is valid in the tea-pouring game -/
def valid_move (n : ℕ) (state : CupState n) (i j : Fin (2 * n)) : Prop :=
  ¬state i ∧ ¬state j ∧ (adjacent n i j ∨ symmetric n i j)

/-- Represents a winning strategy for Petya in the tea-pouring game -/
def petya_wins (n : ℕ) : Prop :=
  ∀ (state : CupState n),
    (∃ (i j : Fin (2 * n)), valid_move n state i j) →
    ∃ (i j : Fin (2 * n)), valid_move n state i j ∧
      ¬(∃ (k l : Fin (2 * n)), valid_move n (Function.update (Function.update state i true) j true) k l)

/-- The main theorem: Petya has a winning strategy if and only if n is odd -/
theorem petya_winning_strategy (n : ℕ) : petya_wins n ↔ Odd n := by
  sorry

end petya_winning_strategy_l1303_130322


namespace rainfall_problem_l1303_130326

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end rainfall_problem_l1303_130326


namespace perpendicular_vectors_x_value_l1303_130392

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b in R², if a is perpendicular to b, 
    and a = (1, 2) and b = (x, 1), then x = -2 -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  perpendicular a b → x = -2 :=
by
  sorry

end perpendicular_vectors_x_value_l1303_130392


namespace product_equals_sum_and_difference_l1303_130364

theorem product_equals_sum_and_difference :
  ∀ a b : ℤ, (a * b = a + b ∧ a * b = a - b) → (a = 0 ∧ b = 0) :=
by sorry

end product_equals_sum_and_difference_l1303_130364


namespace intersection_theorem_l1303_130350

def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | |x| > 2}

theorem intersection_theorem : A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_theorem_l1303_130350


namespace expression_value_l1303_130351

theorem expression_value (x y z : ℝ) : 
  (abs (x - 2) + (y + 3)^2 = 0) → 
  (z = -1) → 
  (2 * (x^2 * y + x * y * z) - 3 * (x^2 * y - x * y * z) - 4 * x^2 * y = 90) :=
by sorry


end expression_value_l1303_130351


namespace line_obtuse_angle_a_range_l1303_130372

/-- Given a line passing through points K(1-a, 1+a) and Q(3, 2a),
    if the line forms an obtuse angle, then a is in the open interval (-2, 1). -/
theorem line_obtuse_angle_a_range (a : ℝ) :
  let K : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let m : ℝ := (Q.2 - K.2) / (Q.1 - K.1)
  (m < 0) → a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end line_obtuse_angle_a_range_l1303_130372


namespace correlation_relationships_l1303_130357

-- Define the types of relationships
inductive Relationship
  | PointCoordinate
  | AppleYieldClimate
  | TreeDiameterHeight
  | StudentID

-- Define a function to determine if a relationship involves correlation
def involvesCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleYieldClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

-- Theorem statement
theorem correlation_relationships :
  (involvesCorrelation Relationship.PointCoordinate = False) ∧
  (involvesCorrelation Relationship.AppleYieldClimate = True) ∧
  (involvesCorrelation Relationship.TreeDiameterHeight = True) ∧
  (involvesCorrelation Relationship.StudentID = False) :=
sorry

end correlation_relationships_l1303_130357


namespace minimum_excellence_rate_l1303_130366

theorem minimum_excellence_rate (total : ℕ) (math_rate : ℚ) (chinese_rate : ℚ) 
  (h_math : math_rate = 70 / 100)
  (h_chinese : chinese_rate = 75 / 100)
  (h_total : total > 0) :
  ∃ (both_rate : ℚ), 
    both_rate ≥ 45 / 100 ∧ 
    both_rate * total ≤ math_rate * total ∧ 
    both_rate * total ≤ chinese_rate * total :=
sorry

end minimum_excellence_rate_l1303_130366


namespace total_football_games_l1303_130321

/-- Calculates the total number of football games in a season -/
theorem total_football_games 
  (games_per_month : ℝ) 
  (season_duration : ℝ) 
  (h1 : games_per_month = 323.0)
  (h2 : season_duration = 17.0) :
  games_per_month * season_duration = 5491.0 := by
  sorry

end total_football_games_l1303_130321


namespace wendy_uses_six_products_l1303_130353

/-- The number of facial products Wendy uses -/
def num_products : ℕ := sorry

/-- The time Wendy waits between each product (in minutes) -/
def wait_time : ℕ := 5

/-- The additional time Wendy spends on make-up (in minutes) -/
def makeup_time : ℕ := 30

/-- The total time for Wendy's "full face" routine (in minutes) -/
def total_time : ℕ := 55

/-- Theorem stating that Wendy uses 6 facial products -/
theorem wendy_uses_six_products : num_products = 6 :=
  by sorry

end wendy_uses_six_products_l1303_130353


namespace derivative_not_always_constant_l1303_130315

-- Define a real-valued function
def f : ℝ → ℝ := sorry

-- Define the derivative of f at a point x
def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

-- Theorem stating that the derivative is not always a constant
theorem derivative_not_always_constant :
  ∃ (f : ℝ → ℝ) (x y : ℝ), x ≠ y → derivative_at f x ≠ derivative_at f y :=
sorry

end derivative_not_always_constant_l1303_130315


namespace greatest_integer_satisfying_inequality_l1303_130394

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ 7 - 5*(-4) > 22 :=
by
  sorry

end greatest_integer_satisfying_inequality_l1303_130394


namespace incorrect_calculation_l1303_130349

theorem incorrect_calculation (h1 : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6)
  (h2 : Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3)
  (h3 : (-Real.sqrt 2)^2 = 2) :
  Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end incorrect_calculation_l1303_130349


namespace degree_to_radian_conversion_l1303_130365

theorem degree_to_radian_conversion (angle_deg : ℝ) : 
  angle_deg * (π / 180) = -5 * π / 3 ↔ angle_deg = -300 :=
sorry

end degree_to_radian_conversion_l1303_130365


namespace circle_equal_circumference_area_l1303_130303

theorem circle_equal_circumference_area (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by
  sorry

end circle_equal_circumference_area_l1303_130303


namespace opposite_sides_inequality_l1303_130325

/-- Given that point P(x₀, y₀) and point A(1, 2) are on opposite sides of the line 3x + 2y - 8 = 0,
    then 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (∃ (ε : ℝ), (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) = -ε ∧ ε > 0) →
  3*x₀ + 2*y₀ > 8 :=
by sorry

end opposite_sides_inequality_l1303_130325


namespace tan_5460_deg_equals_sqrt_3_l1303_130305

theorem tan_5460_deg_equals_sqrt_3 : Real.tan (5460 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_5460_deg_equals_sqrt_3_l1303_130305


namespace yolandas_walking_rate_l1303_130333

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bob_start_delay : ℝ)
  (bob_rate : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 52)
  (h2 : bob_start_delay = 1)
  (h3 : bob_rate = 4)
  (h4 : bob_distance = 28) :
  ∃ (yolanda_rate : ℝ),
    yolanda_rate = 3 ∧
    yolanda_rate * (bob_distance / bob_rate + bob_start_delay) + bob_distance = total_distance :=
by sorry

end yolandas_walking_rate_l1303_130333


namespace mean_temperature_l1303_130320

theorem mean_temperature (temperatures : List ℝ) : 
  temperatures = [75, 77, 76, 80, 82] → 
  (temperatures.sum / temperatures.length : ℝ) = 78 := by
  sorry

end mean_temperature_l1303_130320


namespace perpendicular_lines_parallel_l1303_130361

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b :=
sorry

end perpendicular_lines_parallel_l1303_130361


namespace fraction_equality_l1303_130374

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) :
  m / q = 3 / 14 := by
sorry

end fraction_equality_l1303_130374


namespace square_perimeter_l1303_130342

/-- Given a square with area 720 square meters, its perimeter is 48√5 meters. -/
theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 720 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 48 * Real.sqrt 5 := by
  sorry

end square_perimeter_l1303_130342
