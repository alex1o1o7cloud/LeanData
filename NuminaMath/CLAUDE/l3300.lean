import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3300_330015

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3300_330015


namespace NUMINAMATH_CALUDE_f_4_solutions_l3300_330068

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the composite function f^4
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

-- Theorem statement
theorem f_4_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, f_4 c = 3) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_f_4_solutions_l3300_330068


namespace NUMINAMATH_CALUDE_cody_initial_money_l3300_330067

theorem cody_initial_money : 
  ∀ (initial : ℕ), 
  (initial + 9 - 19 = 35) → 
  initial = 45 := by
sorry

end NUMINAMATH_CALUDE_cody_initial_money_l3300_330067


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l3300_330008

theorem reciprocal_of_negative_fraction (a b : ℤ) (hb : b ≠ 0) :
  ((-1 : ℚ) / (a : ℚ) / (b : ℚ))⁻¹ = -((b : ℚ) / (a : ℚ)) :=
by sorry

theorem reciprocal_of_negative_one_thirteenth :
  ((-1 : ℚ) / 13)⁻¹ = -13 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l3300_330008


namespace NUMINAMATH_CALUDE_distance_traveled_l3300_330079

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man hears the second blast, in minutes -/
def time_between_blasts : ℝ := 30.25

/-- The time between the first and second blasts, in minutes -/
def actual_time_between_blasts : ℝ := 30

/-- Theorem: The distance the man traveled when he heard the second blast is 4950 meters -/
theorem distance_traveled : ℝ := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3300_330079


namespace NUMINAMATH_CALUDE_problem_statement_l3300_330084

theorem problem_statement (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 15) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3300_330084


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3300_330049

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3300_330049


namespace NUMINAMATH_CALUDE_continuous_function_with_property_l3300_330094

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ (n : ℤ) (x : ℝ), n ≠ 0 → f (x + 1 / (n : ℝ)) ≤ f x + 1 / (n : ℝ)

-- State the theorem
theorem continuous_function_with_property (f : ℝ → ℝ) 
  (hf : Continuous f) (hprop : has_property f) :
  ∃ (a : ℝ), ∀ x, f x = x + a := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_with_property_l3300_330094


namespace NUMINAMATH_CALUDE_john_mary_distance_difference_l3300_330053

/-- The width of the streets in feet -/
def street_width : ℕ := 15

/-- The side length of a block in feet -/
def block_side_length : ℕ := 300

/-- The perimeter of a square -/
def square_perimeter (side_length : ℕ) : ℕ := 4 * side_length

theorem john_mary_distance_difference :
  square_perimeter (block_side_length + 2 * street_width) - square_perimeter block_side_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_mary_distance_difference_l3300_330053


namespace NUMINAMATH_CALUDE_potatoes_for_salads_correct_l3300_330097

/-- Given the total number of potatoes, the number used for mashed potatoes,
    and the number of leftover potatoes, calculate the number of potatoes
    used for salads. -/
def potatoes_for_salads (total mashed leftover : ℕ) : ℕ :=
  total - mashed - leftover

/-- Theorem stating that the number of potatoes used for salads is correct. -/
theorem potatoes_for_salads_correct
  (total mashed leftover salads : ℕ)
  (h_total : total = 52)
  (h_mashed : mashed = 24)
  (h_leftover : leftover = 13)
  (h_salads : salads = potatoes_for_salads total mashed leftover) :
  salads = 15 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_salads_correct_l3300_330097


namespace NUMINAMATH_CALUDE_first_three_seeds_l3300_330046

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeedNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 850

/-- Extracts numbers from the random number table --/
def extractNumbers (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (count : Nat) : List Nat :=
  sorry

/-- Selects valid seed numbers from a list of numbers --/
def selectValidSeedNumbers (numbers : List Nat) (count : Nat) : List Nat :=
  sorry

theorem first_three_seeds (table : RandomNumberTable) :
  let extractedNumbers := extractNumbers table 8 7 10
  let selectedSeeds := selectValidSeedNumbers extractedNumbers 3
  selectedSeeds = [785, 567, 199] := by
  sorry

end NUMINAMATH_CALUDE_first_three_seeds_l3300_330046


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_constrained_numbers_l3300_330088

theorem sum_reciprocals_of_constrained_numbers (m n : ℕ+) : 
  Nat.gcd m n = 6 → 
  Nat.lcm m n = 210 → 
  m + n = 72 → 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 17.5 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_constrained_numbers_l3300_330088


namespace NUMINAMATH_CALUDE_geometric_relations_l3300_330047

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (m l : Line) (α β : Plane) : 
  (line_perp_plane l α ∧ line_parallel_plane m α → perpendicular l m) ∧
  ¬(parallel m l ∧ line_in_plane m α → line_parallel_plane l α) ∧
  ¬(plane_perp_plane α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
  ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → plane_perp_plane α β) :=
by sorry

end NUMINAMATH_CALUDE_geometric_relations_l3300_330047


namespace NUMINAMATH_CALUDE_expression_necessarily_negative_l3300_330098

theorem expression_necessarily_negative (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_necessarily_negative_l3300_330098


namespace NUMINAMATH_CALUDE_expression_evaluation_l3300_330029

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3300_330029


namespace NUMINAMATH_CALUDE_tangent_slope_implies_trig_ratio_triangle_perimeter_range_l3300_330075

-- Problem 1
theorem tangent_slope_implies_trig_ratio 
  (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 2*x + 2*Real.sin x + Real.cos x) 
  (h2 : HasDerivAt f 2 α) : 
  (Real.sin (π - α) + Real.cos (-α)) / (2 * Real.cos (π/2 - α) + Real.cos (2*π - α)) = 3/5 := 
sorry

-- Problem 2
theorem triangle_perimeter_range 
  (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h2 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h3 : a = 1) 
  (h4 : a * Real.cos C + c/2 = b) : 
  ∃ l, l = a + b + c ∧ 2 < l ∧ l ≤ 3 := 
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_trig_ratio_triangle_perimeter_range_l3300_330075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3300_330039

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x < -1 → 2 * x^2 + x - 1 > 0) ∧
  (∃ x, 2 * x^2 + x - 1 > 0 ∧ x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3300_330039


namespace NUMINAMATH_CALUDE_geometry_exam_average_score_l3300_330061

/-- Represents a student in the geometry exam -/
structure Student where
  name : String
  mistakes : ℕ
  score : ℚ

/-- Represents the geometry exam -/
structure GeometryExam where
  totalProblems : ℕ
  firstSectionProblems : ℕ
  firstSectionPoints : ℕ
  secondSectionPoints : ℕ
  firstSectionDeduction : ℕ
  secondSectionDeduction : ℕ

theorem geometry_exam_average_score 
  (exam : GeometryExam)
  (madeline leo brent nicholas : Student)
  (h_exam : exam.totalProblems = 15 ∧ 
            exam.firstSectionProblems = 5 ∧ 
            exam.firstSectionPoints = 3 ∧ 
            exam.secondSectionPoints = 1 ∧
            exam.firstSectionDeduction = 2 ∧
            exam.secondSectionDeduction = 1)
  (h_madeline : madeline.mistakes = 2)
  (h_leo : leo.mistakes = 2 * madeline.mistakes)
  (h_brent : brent.score = 25 ∧ brent.mistakes = leo.mistakes + 1)
  (h_nicholas : nicholas.mistakes = 3 * madeline.mistakes ∧ 
                nicholas.score = brent.score - 5) :
  (madeline.score + leo.score + brent.score + nicholas.score) / 4 = 22.25 := by
  sorry

end NUMINAMATH_CALUDE_geometry_exam_average_score_l3300_330061


namespace NUMINAMATH_CALUDE_markup_rate_l3300_330037

theorem markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) : 
  selling_price = 5 → 
  profit_rate = 0.1 → 
  expense_rate = 0.15 → 
  (selling_price / (selling_price * (1 - profit_rate - expense_rate)) - 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_markup_rate_l3300_330037


namespace NUMINAMATH_CALUDE_polar_coordinates_not_bijective_l3300_330091

-- Define the types for different coordinate systems
def CartesianPoint := ℝ × ℝ
def ComplexPoint := ℂ
def PolarPoint := ℝ × ℝ  -- (r, θ)
def Vector2D := ℝ × ℝ

-- Define the bijection property
def IsBijective (f : α → β) : Prop :=
  Function.Injective f ∧ Function.Surjective f

-- State the theorem
theorem polar_coordinates_not_bijective :
  ∃ (f : CartesianPoint → ℝ × ℝ), IsBijective f ∧
  ∃ (g : ComplexPoint → ℝ × ℝ), IsBijective g ∧
  ∃ (h : Vector2D → ℝ × ℝ), IsBijective h ∧
  ¬∃ (k : PolarPoint → ℝ × ℝ), IsBijective k :=
sorry

end NUMINAMATH_CALUDE_polar_coordinates_not_bijective_l3300_330091


namespace NUMINAMATH_CALUDE_cube_edge_assignment_impossibility_l3300_330035

/-- Represents the assignment of 1 or -1 to each edge of a cube -/
def CubeAssignment := Fin 12 → Int

/-- The set of possible sums for a face given an assignment -/
def possibleSums : Finset Int := {-4, -2, 0, 2, 4}

/-- The number of faces on a cube -/
def numFaces : Nat := 6

/-- Given a cube assignment, computes the sum for a specific face -/
def faceSum (assignment : CubeAssignment) (face : Fin 6) : Int :=
  sorry -- Implementation details omitted

theorem cube_edge_assignment_impossibility :
  ¬∃ (assignment : CubeAssignment),
    (∀ (i : Fin 12), assignment i = 1 ∨ assignment i = -1) ∧
    (∀ (face1 face2 : Fin 6), face1 ≠ face2 → faceSum assignment face1 ≠ faceSum assignment face2) :=
  sorry

#check cube_edge_assignment_impossibility

end NUMINAMATH_CALUDE_cube_edge_assignment_impossibility_l3300_330035


namespace NUMINAMATH_CALUDE_max_path_length_rectangular_prism_l3300_330099

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a path through all corners of a rectangular prism -/
def CornerPath (p : RectangularPrism) : Type :=
  List (Fin 2 × Fin 2 × Fin 2)

/-- Calculates the length of a given path in a rectangular prism -/
def pathLength (p : RectangularPrism) (path : CornerPath p) : ℝ :=
  sorry

/-- Checks if a path visits all corners exactly once and returns to start -/
def isValidPath (p : RectangularPrism) (path : CornerPath p) : Prop :=
  sorry

/-- The maximum possible path length for a given rectangular prism -/
def maxPathLength (p : RectangularPrism) : ℝ :=
  sorry

theorem max_path_length_rectangular_prism :
  ∃ (k : ℝ),
    maxPathLength ⟨3, 4, 5⟩ = 4 * Real.sqrt 50 + k ∧
    k > 0 ∧ k < 2 * Real.sqrt 50 :=
  sorry

end NUMINAMATH_CALUDE_max_path_length_rectangular_prism_l3300_330099


namespace NUMINAMATH_CALUDE_ap_num_terms_l3300_330004

/-- The number of terms in an arithmetic progression -/
def num_terms_ap (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: In an arithmetic progression with first term 2, last term 62,
    and common difference 2, the number of terms is 31 -/
theorem ap_num_terms :
  num_terms_ap 2 62 2 = 31 := by
  sorry

#eval num_terms_ap 2 62 2

end NUMINAMATH_CALUDE_ap_num_terms_l3300_330004


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3300_330055

/-- The probability of picking two red balls from a bag containing 3 red, 4 blue, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h_total : total_balls = red_balls + blue_balls + green_balls)
  (h_red : red_balls = 3)
  (h_blue : blue_balls = 4)
  (h_green : green_balls = 4) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3300_330055


namespace NUMINAMATH_CALUDE_infinite_solutions_of_diophantine_equation_l3300_330027

theorem infinite_solutions_of_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z t : ℕ), (x, y, z, t) ∈ S → x^2 + y^2 = 5*(z^2 + t^2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_of_diophantine_equation_l3300_330027


namespace NUMINAMATH_CALUDE_orange_boxes_theorem_l3300_330005

/-- Given 56 oranges that need to be stored in boxes, with each box containing 7 oranges,
    prove that the number of boxes required is 8. -/
theorem orange_boxes_theorem (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 56) (h2 : oranges_per_box = 7) :
  total_oranges / oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_theorem_l3300_330005


namespace NUMINAMATH_CALUDE_f_is_even_l3300_330020

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^3)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_odd g) : ∀ x, f g (-x) = f g x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l3300_330020


namespace NUMINAMATH_CALUDE_rectangle_diagonal_distance_sum_equal_l3300_330025

-- Define a rectangle in 2D space
structure Rectangle where
  a : ℝ  -- half-width
  b : ℝ  -- half-height

-- Define points A, B, C, D of the rectangle
def cornerA (r : Rectangle) : ℝ × ℝ := (-r.a, -r.b)
def cornerB (r : Rectangle) : ℝ × ℝ := (r.a, -r.b)
def cornerC (r : Rectangle) : ℝ × ℝ := (r.a, r.b)
def cornerD (r : Rectangle) : ℝ × ℝ := (-r.a, r.b)

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem rectangle_diagonal_distance_sum_equal (r : Rectangle) (p : ℝ × ℝ) :
  distanceSquared p (cornerA r) + distanceSquared p (cornerC r) =
  distanceSquared p (cornerB r) + distanceSquared p (cornerD r) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_distance_sum_equal_l3300_330025


namespace NUMINAMATH_CALUDE_derivative_of_f_l3300_330045

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 + Real.cos x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3300_330045


namespace NUMINAMATH_CALUDE_sine_product_rational_l3300_330087

theorem sine_product_rational : 
  66 * Real.sin (π / 18) * Real.sin (3 * π / 18) * Real.sin (5 * π / 18) * 
  Real.sin (7 * π / 18) * Real.sin (9 * π / 18) = 33 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_rational_l3300_330087


namespace NUMINAMATH_CALUDE_larger_smaller_division_l3300_330031

theorem larger_smaller_division (L S Q : ℕ) : 
  L - S = 1311 → 
  L = 1430 → 
  L = S * Q + 11 → 
  Q = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_smaller_division_l3300_330031


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330006

-- Define the sets A and B
def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {x | x > -4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330006


namespace NUMINAMATH_CALUDE_square_area_ratio_l3300_330019

/-- Given three squares where each square's side is the diagonal of the next,
    the ratio of the largest square's area to the smallest square's area is 4 -/
theorem square_area_ratio (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = s₂ * Real.sqrt 2) (h₂ : s₂ = s₃ * Real.sqrt 2) :
  s₁^2 / s₃^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3300_330019


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l3300_330022

/-- The probability of finding treasure without traps on an island -/
def p_treasure_only : ℚ := 1 / 5

/-- The probability of finding neither treasure nor traps on an island -/
def p_neither : ℚ := 3 / 5

/-- The number of islands -/
def n_islands : ℕ := 7

/-- The number of islands with treasure only -/
def n_treasure_only : ℕ := 3

/-- The number of islands with neither treasure nor traps -/
def n_neither : ℕ := 4

theorem pirate_treasure_probability : 
  (Nat.choose n_islands n_treasure_only : ℚ) * 
  (p_treasure_only ^ n_treasure_only) * 
  (p_neither ^ n_neither) = 81 / 2225 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l3300_330022


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3300_330072

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 1/2 ∨ x = -3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ :=
  {y | y = 3/4 ∨ y = 20}

/-- First parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Second parabola function -/
def g (x : ℝ) : ℝ := 3*x^2 + 2*x - 1

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = g x ∧ y = f x) ↔ (x ∈ intersection_x ∧ y ∈ intersection_y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3300_330072


namespace NUMINAMATH_CALUDE_inequality_proof_l3300_330062

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) ≥ 2 ∧
  ((1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3300_330062


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_condition_l3300_330013

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_condition
  (α β : Plane) (m n l₁ l₂ : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α)
  (h_l₁_in_β : line_in_plane l₁ β)
  (h_l₂_in_β : line_in_plane l₂ β)
  (h_l₁_l₂_intersect : line_intersect l₁ l₂)
  (h_m_parallel_l₁ : line_parallel m l₁)
  (h_n_parallel_l₂ : line_parallel n l₂) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_condition_l3300_330013


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3300_330085

theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (Complex.I / (1 + Complex.I)) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3300_330085


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3300_330023

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 5 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3300_330023


namespace NUMINAMATH_CALUDE_goldbach_138_largest_diff_l3300_330082

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_largest_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → 
      (max r s - min r s) ≤ (max p q - min p q) ∧
    (max p q - min p q) = 124 :=
sorry

end NUMINAMATH_CALUDE_goldbach_138_largest_diff_l3300_330082


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3300_330002

/-- Proves that the first discount percentage is 10% given the conditions of the problem -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = list_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3300_330002


namespace NUMINAMATH_CALUDE_apple_difference_l3300_330056

theorem apple_difference (jackie_apples adam_apples : ℕ) 
  (h1 : jackie_apples = 10) (h2 : adam_apples = 8) : 
  jackie_apples - adam_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l3300_330056


namespace NUMINAMATH_CALUDE_dave_derek_money_difference_l3300_330070

theorem dave_derek_money_difference :
  let derek_initial : ℕ := 40
  let derek_lunch1 : ℕ := 14
  let derek_dad_lunch : ℕ := 11
  let derek_lunch2 : ℕ := 5
  let dave_initial : ℕ := 50
  let dave_mom_lunch : ℕ := 7
  let derek_remaining : ℕ := derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2
  let dave_remaining : ℕ := dave_initial - dave_mom_lunch
  dave_remaining - derek_remaining = 33 :=
by sorry

end NUMINAMATH_CALUDE_dave_derek_money_difference_l3300_330070


namespace NUMINAMATH_CALUDE_next_shared_meeting_l3300_330036

/-- The number of days between meetings for the drama club -/
def drama_interval : ℕ := 3

/-- The number of days between meetings for the choir -/
def choir_interval : ℕ := 5

/-- The number of days between meetings for the debate team -/
def debate_interval : ℕ := 7

/-- The theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  Nat.lcm (Nat.lcm drama_interval choir_interval) debate_interval = 105 := by
  sorry

end NUMINAMATH_CALUDE_next_shared_meeting_l3300_330036


namespace NUMINAMATH_CALUDE_circular_film_diameter_l3300_330063

theorem circular_film_diameter 
  (volume : ℝ) 
  (thickness : ℝ) 
  (π : ℝ) 
  (h1 : volume = 576) 
  (h2 : thickness = 0.2) 
  (h3 : π = Real.pi) : 
  let radius := Real.sqrt (volume / (thickness * π))
  2 * radius = 2 * Real.sqrt (2880 / π) :=
sorry

end NUMINAMATH_CALUDE_circular_film_diameter_l3300_330063


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3300_330028

/-- The center of a circle that is tangent to two parallel lines and lies on a third line -/
theorem circle_center_coordinates (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    ((x - 20)^2 + (y - 10)^2 = r^2) ∧ 
    ((x - 40/3)^2 + y^2 = r^2) ∧ 
    x^2 + y^2 = r^2) → 
  (3*x - 4*y = 20 ∧ x - 2*y = 0) → 
  x = 20 ∧ y = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3300_330028


namespace NUMINAMATH_CALUDE_square_area_is_eight_l3300_330040

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 36

-- Define the property of being inscribed in a square with side parallel to x-axis
def inscribed_in_square (c : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (center_x center_y radius : ℝ),
    ∀ (x y : ℝ), c x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- State the theorem
theorem square_area_is_eight :
  inscribed_in_square circle_equation →
  (∃ (side : ℝ), side^2 = 8 ∧
    ∀ (x y : ℝ), circle_equation x y →
      x ≥ -side/2 ∧ x ≤ side/2 ∧ y ≥ -side/2 ∧ y ≤ side/2) :=
sorry

end NUMINAMATH_CALUDE_square_area_is_eight_l3300_330040


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_Q_perfect_square_l3300_330076

/-- The polynomial Q as a function of x -/
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

/-- Theorem stating that there are no integer solutions for x such that Q(x) is a perfect square -/
theorem no_integer_solutions_for_Q_perfect_square :
  ∀ x : ℤ, ¬∃ k : ℤ, Q x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_Q_perfect_square_l3300_330076


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3300_330021

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 2 = 4) 
  (h_sum2 : a 3 + a 4 = 16) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3300_330021


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330017

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330017


namespace NUMINAMATH_CALUDE_rook_configuration_exists_iff_even_l3300_330073

/-- A configuration of rooks on an n×n board. -/
def RookConfiguration (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a rook configuration is valid (no two rooks attack each other). -/
def is_valid_configuration (n : ℕ) (config : RookConfiguration n) : Prop :=
  ∀ i j : Fin n, i ≠ j → config i ≠ config j ∧ i ≠ config j

/-- Predicate to check if two positions on the board are adjacent. -/
def are_adjacent (n : ℕ) (p1 p2 : Fin n × Fin n) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Theorem stating that a valid rook configuration with a valid move exists if and only if n is even. -/
theorem rook_configuration_exists_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial final : RookConfiguration n),
    is_valid_configuration n initial ∧
    is_valid_configuration n final ∧
    (∀ i : Fin n, are_adjacent n (i, initial i) (i, final i))) ↔
  Even n :=
sorry

end NUMINAMATH_CALUDE_rook_configuration_exists_iff_even_l3300_330073


namespace NUMINAMATH_CALUDE_distribution_theorem_l3300_330083

/-- The number of ways to distribute 6 volunteers into 4 groups and assign to 4 pavilions -/
def distribution_schemes : ℕ := 1080

/-- The number of volunteers -/
def num_volunteers : ℕ := 6

/-- The number of pavilions -/
def num_pavilions : ℕ := 4

/-- The number of groups with 2 people -/
def num_pairs : ℕ := 2

/-- The number of groups with 1 person -/
def num_singles : ℕ := 2

theorem distribution_theorem :
  (num_volunteers = 6) →
  (num_pavilions = 4) →
  (num_pairs = 2) →
  (num_singles = 2) →
  (num_pairs + num_singles = num_pavilions) →
  (2 * num_pairs + num_singles = num_volunteers) →
  distribution_schemes = 1080 := by
  sorry

#eval distribution_schemes

end NUMINAMATH_CALUDE_distribution_theorem_l3300_330083


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3300_330054

theorem complex_equation_solution (z : ℂ) :
  (3 - 4 * Complex.I) * z = 5 → z = 3/5 + 4/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3300_330054


namespace NUMINAMATH_CALUDE_limit_of_a_l3300_330059

def a (n : ℕ) : ℚ := (2 * n + 1) / (5 * n - 1)

theorem limit_of_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 2/5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_l3300_330059


namespace NUMINAMATH_CALUDE_salaries_degrees_l3300_330000

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ

/-- The total budget percentage should sum to 100% -/
axiom budget_sum (b : BudgetAllocation) : 
  b.transportation + b.research_development + b.utilities + b.equipment + b.supplies + b.salaries = 100

/-- The given budget allocation -/
def company_budget : BudgetAllocation where
  transportation := 20
  research_development := 9
  utilities := 5
  equipment := 4
  supplies := 2
  salaries := 100 - (20 + 9 + 5 + 4 + 2)

/-- The number of degrees in a full circle -/
def full_circle : ℝ := 360

/-- Theorem: The number of degrees representing salaries in the circle graph is 216 -/
theorem salaries_degrees : 
  (company_budget.salaries / 100) * full_circle = 216 := by sorry

end NUMINAMATH_CALUDE_salaries_degrees_l3300_330000


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3300_330042

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3300_330042


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3300_330014

theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h1 : equilateral_perimeter = 60)
  (h2 : isosceles_base = 10)
  : ℝ := by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3300_330014


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3300_330050

def initial_center : ℝ × ℝ := (8, -3)
def reflection_line (x y : ℝ) : Prop := y = x
def translation_vector : ℝ × ℝ := (2, -5)

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translate_point (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem circle_center_transformation :
  let reflected := reflect_point initial_center
  let final := translate_point reflected translation_vector
  final = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3300_330050


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3300_330038

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2^5 * 3^2 * 7) :
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!)) →
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!) ∧ m = 334) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3300_330038


namespace NUMINAMATH_CALUDE_power_product_equality_l3300_330032

theorem power_product_equality : (3^5 * 4^5) = 248832 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l3300_330032


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_475_l3300_330095

theorem least_multiple_of_25_greater_than_475 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 475 → n ≥ 500 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_475_l3300_330095


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3300_330078

/-- Hyperbola with given asymptotes and passing point -/
structure Hyperbola where
  -- Asymptotes are y = ±√2x
  asymptote_slope : ℝ
  asymptote_slope_sq : asymptote_slope^2 = 2
  -- Passes through (3, -2√3)
  passes_through : (3 : ℝ)^2 / 3 - (-2 * Real.sqrt 3)^2 / 6 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 / 6 = 1

/-- Point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola_equation h x y

/-- Foci of the hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Angle between foci and point on hyperbola -/
def angle_F₁PF₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the angle

/-- Area of triangle formed by foci and point on hyperbola -/
def area_PF₁F₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the area

/-- Main theorem -/
theorem hyperbola_theorem (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) :
  hyperbola_equation h p.x p.y ∧
  (angle_F₁PF₂ h f p = π / 3 → area_PF₁F₂ h f p = 6 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3300_330078


namespace NUMINAMATH_CALUDE_hike_weight_after_six_hours_l3300_330033

/-- Calculates the remaining weight after a hike given initial weights and consumption rates -/
def remaining_weight (initial_water : ℝ) (initial_food : ℝ) (initial_gear : ℝ) 
                     (water_rate : ℝ) (food_rate : ℝ) (hours : ℝ) : ℝ :=
  let remaining_water := initial_water - water_rate * hours
  let remaining_food := initial_food - food_rate * hours
  remaining_water + remaining_food + initial_gear

/-- Theorem: The remaining weight after 6 hours of hiking is 34 pounds -/
theorem hike_weight_after_six_hours :
  remaining_weight 20 10 20 2 (2/3) 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_hike_weight_after_six_hours_l3300_330033


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3300_330034

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) :
  ∃ (min_dist : ℝ), min_dist = 120 / 17 ∧ 
    ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3300_330034


namespace NUMINAMATH_CALUDE_tom_battery_usage_l3300_330074

/-- Calculates the total number of batteries used by Tom -/
def total_batteries (flashlights : ℕ) (flashlight_batteries : ℕ) 
                    (toys : ℕ) (toy_batteries : ℕ)
                    (controllers : ℕ) (controller_batteries : ℕ) : ℕ :=
  flashlights * flashlight_batteries + 
  toys * toy_batteries + 
  controllers * controller_batteries

/-- Proves that Tom used 38 batteries in total -/
theorem tom_battery_usage : 
  total_batteries 3 2 5 4 6 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_tom_battery_usage_l3300_330074


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l3300_330003

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_one :
  (A ∩ B 1 = {x | x < -2}) ∧ (A ∪ B 1 = {x | x > 2 ∨ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_a_when_complement_A_subset_B :
  ∀ a : ℝ, (Set.univ \ A : Set ℝ) ⊆ B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l3300_330003


namespace NUMINAMATH_CALUDE_odd_function_condition_l3300_330012

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2x^3 + ax^2 + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  2 * x^3 + a * x^2 + b - 1

/-- If f(x) = 2x^3 + ax^2 + b - 1 is an odd function, then a - b = -1 -/
theorem odd_function_condition (a b : ℝ) :
  IsOdd (f a b) → a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_condition_l3300_330012


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3300_330089

theorem quadratic_roots_difference_squared :
  ∀ p q : ℝ, (2 * p^2 - 9 * p + 7 = 0) → (2 * q^2 - 9 * q + 7 = 0) → (p - q)^2 = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3300_330089


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3300_330010

/-- For any positive real number a, the function f(x) = a^(x-1) + 2 always passes through the point (1, 3). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3300_330010


namespace NUMINAMATH_CALUDE_min_pieces_to_find_both_l3300_330007

-- Define the grid
def Grid := Fin 8 → Fin 8 → Bool

-- Define the properties of the grid
def has_fish (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_sausage (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_both (g : Grid) (i j : Fin 8) : Prop := has_fish g i j ∧ has_sausage g i j

-- Define the conditions
def valid_grid (g : Grid) : Prop :=
  (∃ i j k l m n : Fin 8, has_fish g i j ∧ has_fish g k l ∧ has_fish g m n ∧ 
    ¬(i = k ∧ j = l) ∧ ¬(i = m ∧ j = n) ∧ ¬(k = m ∧ l = n)) ∧
  (∃ i j k l : Fin 8, has_sausage g i j ∧ has_sausage g k l ∧ ¬(i = k ∧ j = l)) ∧
  (∃! i j : Fin 8, has_both g i j) ∧
  (∀ i j : Fin 6, ∃ k l m n : Fin 8, k ≥ i ∧ k < i + 6 ∧ l ≥ j ∧ l < j + 6 ∧
    m ≥ i ∧ m < i + 6 ∧ n ≥ j ∧ n < j + 6 ∧ has_fish g k l ∧ has_fish g m n ∧ ¬(k = m ∧ l = n)) ∧
  (∀ i j : Fin 6, ∃! k l : Fin 8, k ≥ i ∧ k < i + 3 ∧ l ≥ j ∧ l < j + 3 ∧ has_sausage g k l)

-- Define the theorem
theorem min_pieces_to_find_both (g : Grid) (h : valid_grid g) :
  ∃ s : Finset (Fin 8 × Fin 8), s.card = 5 ∧
    (∀ t : Finset (Fin 8 × Fin 8), t.card < 5 → 
      ∃ i j : Fin 8, has_both g i j ∧ (i, j) ∉ t) ∧
    (∀ i j : Fin 8, has_both g i j → (i, j) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_min_pieces_to_find_both_l3300_330007


namespace NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l3300_330096

/-- Represents the number of containers of blueberries per bush -/
def blueberries_per_bush : ℕ := 12

/-- Represents the number of containers of blueberries that can be traded for pumpkins -/
def blueberries_for_pumpkins : ℕ := 4

/-- Represents the number of pumpkins received when trading blueberries -/
def pumpkins_from_blueberries : ℕ := 3

/-- Represents the number of pumpkins that can be traded for zucchinis -/
def pumpkins_for_zucchinis : ℕ := 6

/-- Represents the number of zucchinis received when trading pumpkins -/
def zucchinis_from_pumpkins : ℕ := 5

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

theorem bushes_needed_for_zucchinis :
  ∃ (bushes : ℕ), 
    bushes * blueberries_per_bush * pumpkins_from_blueberries * zucchinis_from_pumpkins = 
    target_zucchinis * blueberries_for_pumpkins * pumpkins_for_zucchinis ∧ 
    bushes = 8 := by
  sorry

end NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l3300_330096


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l3300_330077

variable (x : Fin 10 → ℝ)

def variance (data : Fin 10 → ℝ) : ℝ := sorry

def transform (data : Fin 10 → ℝ) : Fin 10 → ℝ := 
  fun i => 2 * data i - 1

theorem variance_of_transformed_data 
  (h : variance x = 8) : 
  variance (transform x) = 32 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l3300_330077


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3300_330069

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ),
  (6 * x^2 - 12 * x + 4 = a * (x - h)^2 + k) ∧ (a + h + k = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3300_330069


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l3300_330018

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n : ℝ) : ℝ × ℝ := (m + 4, n + 2)

-- Theorem statement
theorem x_coordinate_of_first_point (m n : ℝ) :
  line_equation m n ∧ line_equation (m + 4) (n + 2) → m = 2 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l3300_330018


namespace NUMINAMATH_CALUDE_color_film_fraction_l3300_330058

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 4 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l3300_330058


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330081

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3300_330081


namespace NUMINAMATH_CALUDE_spherical_coordinate_reflection_l3300_330051

/-- Given a point with rectangular coordinates (3, 8, -6) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ, -φ) has rectangular coordinates (-3, -8, -6). -/
theorem spherical_coordinate_reflection (ρ θ φ : ℝ) : 
  (ρ * Real.sin φ * Real.cos θ = 3 ∧ 
   ρ * Real.sin φ * Real.sin θ = 8 ∧ 
   ρ * Real.cos φ = -6) → 
  (ρ * Real.sin (-φ) * Real.cos θ = -3 ∧ 
   ρ * Real.sin (-φ) * Real.sin θ = -8 ∧ 
   ρ * Real.cos (-φ) = -6) := by
  sorry

end NUMINAMATH_CALUDE_spherical_coordinate_reflection_l3300_330051


namespace NUMINAMATH_CALUDE_veranda_area_l3300_330026

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) :
  room_length = 20 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 144 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l3300_330026


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3300_330066

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 < 0 ↔ 2 < y ∧ y < 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3300_330066


namespace NUMINAMATH_CALUDE_beth_winning_strategy_l3300_330030

/-- Represents a wall of bricks in the game --/
structure Wall :=
  (size : Nat)

/-- Represents a game state with multiple walls --/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall --/
def nimValue (w : Wall) : Nat :=
  sorry

/-- Calculates the nim-value of a game state --/
def gameNimValue (state : GameState) : Nat :=
  sorry

/-- Checks if a game state is a losing position for the current player --/
def isLosingPosition (state : GameState) : Prop :=
  gameNimValue state = 0

/-- The main theorem to prove --/
theorem beth_winning_strategy (startState : GameState) :
  startState.walls = [Wall.mk 6, Wall.mk 2, Wall.mk 1] →
  isLosingPosition startState :=
sorry

end NUMINAMATH_CALUDE_beth_winning_strategy_l3300_330030


namespace NUMINAMATH_CALUDE_custom_op_example_l3300_330092

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := a^2 - b

-- State the theorem
theorem custom_op_example : custom_op (custom_op 1 2) 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3300_330092


namespace NUMINAMATH_CALUDE_x_over_y_equals_negative_one_fourth_l3300_330057

theorem x_over_y_equals_negative_one_fourth (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y)^5 + x^5 + 4 * x + y = 0) : x / y = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_negative_one_fourth_l3300_330057


namespace NUMINAMATH_CALUDE_unique_digit_multiple_6_and_9_l3300_330001

def is_multiple_of_6_and_9 (n : ℕ) : Prop :=
  n % 6 = 0 ∧ n % 9 = 0

def five_digit_number (d : ℕ) : ℕ :=
  74820 + d

theorem unique_digit_multiple_6_and_9 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6_and_9 (five_digit_number d) :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_multiple_6_and_9_l3300_330001


namespace NUMINAMATH_CALUDE_bread_sharing_theorem_l3300_330016

/-- Calculates the number of slices each friend eats when sharing bread equally -/
def slices_per_friend (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

/-- Proves that under the given conditions, each friend eats 6 slices of bread -/
theorem bread_sharing_theorem :
  let slices_per_loaf : ℕ := 15
  let num_friends : ℕ := 10
  let num_loaves : ℕ := 4
  slices_per_friend slices_per_loaf num_friends num_loaves = 6 := by
  sorry

end NUMINAMATH_CALUDE_bread_sharing_theorem_l3300_330016


namespace NUMINAMATH_CALUDE_jellybean_difference_l3300_330093

theorem jellybean_difference (total : ℕ) (black : ℕ) (green : ℕ) (orange : ℕ) : 
  total = 27 →
  black = 8 →
  orange = green - 1 →
  total = black + green + orange →
  green - black = 2 := by
sorry

end NUMINAMATH_CALUDE_jellybean_difference_l3300_330093


namespace NUMINAMATH_CALUDE_polynomial_composition_factorization_l3300_330041

theorem polynomial_composition_factorization :
  ∀ (p : Polynomial ℤ),
  (Polynomial.degree p ≥ 1) →
  ∃ (q f g : Polynomial ℤ),
    (Polynomial.degree f ≥ 1) ∧
    (Polynomial.degree g ≥ 1) ∧
    (p.comp q = f * g) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_composition_factorization_l3300_330041


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l3300_330065

/-- A parabola y = ax^2 + bx - 4 is tangent to the line y = 2x + 3 if and only if
    a = -(b-2)^2 / 28 and b ≠ 2 -/
theorem parabola_tangent_to_line (a b : ℝ) :
  (∃ x y : ℝ, y = a * x^2 + b * x - 4 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, x' ≠ x → a * x'^2 + b * x' - 4 ≠ 2 * x' + 3) ↔
  (a = -(b-2)^2 / 28 ∧ b ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l3300_330065


namespace NUMINAMATH_CALUDE_apples_left_l3300_330048

def initial_apples : ℕ := 150
def sold_percentage : ℚ := 30 / 100
def given_percentage : ℚ := 20 / 100
def donated_apples : ℕ := 2

theorem apples_left : 
  let remaining_after_sale := initial_apples - (↑initial_apples * sold_percentage).floor
  let remaining_after_given := remaining_after_sale - (↑remaining_after_sale * given_percentage).floor
  remaining_after_given - donated_apples = 82 := by sorry

end NUMINAMATH_CALUDE_apples_left_l3300_330048


namespace NUMINAMATH_CALUDE_shells_calculation_l3300_330011

/-- Given an initial amount of shells and an additional amount of shells,
    calculate the total amount of shells. -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 5 pounds of shells initially and 23 pounds added,
    the total is 28 pounds. -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shells_calculation_l3300_330011


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3300_330086

def complex_multiply (a b : ℂ) : ℂ := a * b

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z (z : ℂ) :
  complex_multiply (1 + 3*Complex.I) z = 10 →
  imaginary_part z = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3300_330086


namespace NUMINAMATH_CALUDE_nosuch_junction_population_l3300_330071

theorem nosuch_junction_population : ∃ (a b c : ℕ+), 
  (a.val^2 + 100 = b.val^2 + 1) ∧ 
  (b.val^2 + 101 = c.val^2) ∧ 
  (7 ∣ a.val^2) := by
  sorry

end NUMINAMATH_CALUDE_nosuch_junction_population_l3300_330071


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l3300_330060

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (f m + f n) = m + n) → (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l3300_330060


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l3300_330044

universe u

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l3300_330044


namespace NUMINAMATH_CALUDE_stationery_cost_l3300_330024

/-- The cost of items at a stationery store -/
theorem stationery_cost (x y z : ℝ) 
  (h1 : 4 * x + y + 10 * z = 11) 
  (h2 : 3 * x + y + 7 * z = 8.9) : 
  x + y + z = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l3300_330024


namespace NUMINAMATH_CALUDE_soda_price_proof_l3300_330052

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 18.36

theorem soda_price_proof :
  (72 * discounted_price = total_price) →
  regular_price = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_proof_l3300_330052


namespace NUMINAMATH_CALUDE_min_product_of_three_l3300_330080

def S : Finset ℤ := {-10, -7, -5, -3, 0, 2, 4, 6, 8}

theorem min_product_of_three (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≥ -480 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3300_330080


namespace NUMINAMATH_CALUDE_cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l3300_330090

-- Define the wire length and cube edge length
def wire_length : ℝ := 120
def cube_edge_length : ℝ := 10

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Define the number of vertices in a cube
def cube_vertices : ℕ := 8

-- Define the number of edges meeting at each vertex of a cube
def edges_per_vertex : ℕ := 3

-- Theorem 1: It's impossible to create the cube frame without cuts
theorem cube_frame_impossible_without_cuts :
  ¬ ∃ (path : List ℝ), 
    (path.length = cube_edges) ∧ 
    (path.sum = wire_length) ∧
    (∀ edge ∈ path, edge = cube_edge_length) :=
sorry

-- Theorem 2: The minimum number of cuts required is 3
theorem minimum_cuts_for_cube_frame :
  (cube_vertices / 2 : ℕ) - 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l3300_330090


namespace NUMINAMATH_CALUDE_at_least_one_product_contains_seven_l3300_330064

def containsSeven (m : Nat) : Bool :=
  let digits := m.digits 10
  7 ∈ digits

theorem at_least_one_product_contains_seven (n : Nat) (hn : n > 0) :
  ∃ k : Nat, k ≤ 35 ∧ k > 0 ∧ containsSeven (k * n) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_product_contains_seven_l3300_330064


namespace NUMINAMATH_CALUDE_tan_difference_sum_l3300_330009

theorem tan_difference_sum (α β γ : Real) 
  (h1 : Real.tan α = 5)
  (h2 : Real.tan β = 2)
  (h3 : Real.tan γ = 3) :
  Real.tan (α - β + γ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_sum_l3300_330009


namespace NUMINAMATH_CALUDE_initial_tangerines_count_l3300_330043

/-- The number of tangerines initially in the basket -/
def initial_tangerines : ℕ := sorry

/-- The number of tangerines Eunji ate -/
def eaten_tangerines : ℕ := 9

/-- The number of tangerines mother added -/
def added_tangerines : ℕ := 5

/-- The final number of tangerines in the basket -/
def final_tangerines : ℕ := 20

/-- Theorem stating that the initial number of tangerines was 24 -/
theorem initial_tangerines_count : initial_tangerines = 24 :=
by
  have h : initial_tangerines - eaten_tangerines + added_tangerines = final_tangerines := sorry
  sorry


end NUMINAMATH_CALUDE_initial_tangerines_count_l3300_330043
