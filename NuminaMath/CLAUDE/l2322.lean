import Mathlib

namespace NUMINAMATH_CALUDE_extremum_and_nonnegative_conditions_l2322_232291

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) - Real.log x

theorem extremum_and_nonnegative_conditions (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ∈ Set.Ioo (1 - ε) (1 + ε) → f a x ≥ f a 1) →
  a = 1 ∧
  (∀ (x : ℝ), x ≥ 1 → f a x ≥ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_nonnegative_conditions_l2322_232291


namespace NUMINAMATH_CALUDE_xyz_product_l2322_232260

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 120 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2322_232260


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l2322_232277

theorem chess_team_boys_count (total_members : ℕ) (meeting_attendees : ℕ) : 
  total_members = 30 →
  meeting_attendees = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_members ∧
    (2 * girls / 3 : ℚ) + boys = meeting_attendees ∧
    boys = 0 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l2322_232277


namespace NUMINAMATH_CALUDE_zero_existence_l2322_232242

theorem zero_existence (f : ℝ → ℝ) (hf : Continuous f) 
  (h0 : f 0 = -3) (h1 : f 1 = 6) (h3 : f 3 = -5) :
  ∃ x₁ ∈ Set.Ioo 0 1, f x₁ = 0 ∧ ∃ x₂ ∈ Set.Ioo 1 3, f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_existence_l2322_232242


namespace NUMINAMATH_CALUDE_balloon_difference_l2322_232203

theorem balloon_difference (x y z : ℚ) 
  (eq1 : x = 3 * z - 2)
  (eq2 : y = z / 4 + 5)
  (eq3 : z = y + 3) :
  x + y - z = 27 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2322_232203


namespace NUMINAMATH_CALUDE_game_prime_exists_l2322_232221

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem game_prime_exists : 
  ∃ p : ℕ, 
    is_prime p ∧ 
    ∃ (a b c d : ℕ), 
      p = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ ({4, 7, 8} : Set ℕ) ∧
      b ∈ ({4, 5, 9} : Set ℕ) ∧
      c ∈ ({1, 2, 3} : Set ℕ) ∧
      d < 10 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      p = 8923 :=
by
  sorry

end NUMINAMATH_CALUDE_game_prime_exists_l2322_232221


namespace NUMINAMATH_CALUDE_largest_root_ratio_l2322_232204

-- Define the polynomials
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

-- Define x₁ as the largest root of f
def x₁ : ℝ := sorry

-- Define x₂ as the largest root of g
def x₂ : ℝ := sorry

-- Theorem statement
theorem largest_root_ratio :
  x₂ / x₁ = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l2322_232204


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2322_232216

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z := (4 : ℂ) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2322_232216


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2322_232234

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  (x^2 - 26*x + 81 = (x + b)^2 + c) ∧ (b + c = -101) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2322_232234


namespace NUMINAMATH_CALUDE_function_is_constant_l2322_232219

/-- A function f: ℚ → ℝ satisfying |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℚ is constant. -/
theorem function_is_constant (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℚ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l2322_232219


namespace NUMINAMATH_CALUDE_poster_cost_l2322_232272

theorem poster_cost (initial_money : ℕ) (book1_cost : ℕ) (book2_cost : ℕ) (num_posters : ℕ) :
  initial_money = 20 →
  book1_cost = 8 →
  book2_cost = 4 →
  num_posters = 2 →
  initial_money - (book1_cost + book2_cost) = num_posters * (initial_money - (book1_cost + book2_cost)) / num_posters :=
by
  sorry

end NUMINAMATH_CALUDE_poster_cost_l2322_232272


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l2322_232283

def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def sale_value : ℕ := 1500
def commission_rate : ℚ := 15 / 100

theorem min_sales_to_break_even : 
  ∃ (n : ℕ), n = 200 ∧ 
  (n : ℚ) * commission_rate * sale_value + new_base_salary = current_salary ∧
  ∀ (m : ℕ), m < n → (m : ℚ) * commission_rate * sale_value + new_base_salary < current_salary :=
sorry

end NUMINAMATH_CALUDE_min_sales_to_break_even_l2322_232283


namespace NUMINAMATH_CALUDE_workshop_workers_l2322_232206

theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (non_technician_salary : ℕ) : 
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 14000 →
  non_technician_salary = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * average_salary = technician_count * technician_salary + (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 28 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2322_232206


namespace NUMINAMATH_CALUDE_cube_face_sum_l2322_232286

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels given the face values -/
def vertexSum (faces : CubeFaces) : ℕ :=
  (faces.a * faces.b * faces.c) + (faces.a * faces.e * faces.c) +
  (faces.a * faces.b * faces.f) + (faces.a * faces.e * faces.f) +
  (faces.d * faces.b * faces.c) + (faces.d * faces.e * faces.c) +
  (faces.d * faces.b * faces.f) + (faces.d * faces.e * faces.f)

/-- Calculates the sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

/-- Theorem: If the vertex sum is 1452, then the face sum is 47 -/
theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1452 → faceSum faces = 47 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_l2322_232286


namespace NUMINAMATH_CALUDE_problem_solution_l2322_232252

theorem problem_solution : (2010^2 - 2010 + 1) / (2010 + 1) = 4040091 / 2011 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2322_232252


namespace NUMINAMATH_CALUDE_expression_value_l2322_232220

theorem expression_value : 
  let a : ℚ := 1/2
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2322_232220


namespace NUMINAMATH_CALUDE_common_root_theorem_l2322_232298

theorem common_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0) ∧
  (∃ y : ℝ, b * y^11 + c * y^4 + a = 0 ∧ c * y^11 + a * y^4 + b = 0) ∧
  (∃ z : ℝ, c * z^11 + a * z^4 + b = 0 ∧ a * z^11 + b * z^4 + c = 0) →
  ∃ w : ℝ, a * w^11 + b * w^4 + c = 0 ∧
           b * w^11 + c * w^4 + a = 0 ∧
           c * w^11 + a * w^4 + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_root_theorem_l2322_232298


namespace NUMINAMATH_CALUDE_pyramid_height_specific_l2322_232281

/-- Represents a pyramid with a square base and identical triangular faces. -/
structure Pyramid where
  base_area : ℝ
  face_area : ℝ

/-- The height of a pyramid given its base area and face area. -/
def pyramid_height (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with base area 1440 and face area 840 has height 40. -/
theorem pyramid_height_specific : 
  ∀ (p : Pyramid), p.base_area = 1440 ∧ p.face_area = 840 → pyramid_height p = 40 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_specific_l2322_232281


namespace NUMINAMATH_CALUDE_small_hotdogs_count_l2322_232263

theorem small_hotdogs_count (total : ℕ) (large : ℕ) (h1 : total = 79) (h2 : large = 21) :
  total - large = 58 := by
  sorry

end NUMINAMATH_CALUDE_small_hotdogs_count_l2322_232263


namespace NUMINAMATH_CALUDE_prime_sum_special_equation_l2322_232213

theorem prime_sum_special_equation (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_special_equation_l2322_232213


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2322_232273

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 players, where each player plays every other player
    exactly once, and each game involves two players, the total number of games played is 190. --/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2322_232273


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_twelve_after_subtracting_seven_l2322_232245

theorem smallest_number_divisible_by_twelve_after_subtracting_seven : 
  ∃ N : ℕ, N > 0 ∧ (N - 7) % 12 = 0 ∧ ∀ m : ℕ, m > 0 → (m - 7) % 12 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_twelve_after_subtracting_seven_l2322_232245


namespace NUMINAMATH_CALUDE_total_wheat_mass_l2322_232258

def wheat_weights : List Float := [90, 91, 91.5, 89, 91.2, 91.3, 89.7, 88.8, 91.8, 91.1]

theorem total_wheat_mass :
  wheat_weights.sum = 905.4 := by
  sorry

end NUMINAMATH_CALUDE_total_wheat_mass_l2322_232258


namespace NUMINAMATH_CALUDE_second_die_sides_l2322_232207

theorem second_die_sides (n : ℕ) (h : n > 0) :
  (1 / 2) * ((n - 1) / (2 * n)) = 21428571428571427 / 100000000000000000 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_second_die_sides_l2322_232207


namespace NUMINAMATH_CALUDE_triangulated_square_interior_points_l2322_232261

/-- Represents a square divided into triangles -/
structure TriangulatedSquare where
  /-- The number of triangles in the square -/
  num_triangles : ℕ
  /-- The number of interior points (vertices of triangles) -/
  num_interior_points : ℕ
  /-- Condition: No vertex lies on sides or inside other triangles -/
  no_overlap : Prop
  /-- Condition: Sides of square are sides of some triangles -/
  square_sides_are_triangle_sides : Prop

/-- Theorem: A square divided into 2016 triangles has 1007 interior points -/
theorem triangulated_square_interior_points
  (ts : TriangulatedSquare)
  (h_num_triangles : ts.num_triangles = 2016) :
  ts.num_interior_points = 1007 := by
  sorry

#check triangulated_square_interior_points

end NUMINAMATH_CALUDE_triangulated_square_interior_points_l2322_232261


namespace NUMINAMATH_CALUDE_cost_of_field_trip_l2322_232237

def field_trip_cost (grandma_contribution : ℝ) (candy_bar_price : ℝ) (candy_bars_to_sell : ℕ) : ℝ :=
  grandma_contribution + candy_bar_price * (candy_bars_to_sell : ℝ)

theorem cost_of_field_trip :
  field_trip_cost 250 1.25 188 = 485 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_field_trip_l2322_232237


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2322_232297

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2322_232297


namespace NUMINAMATH_CALUDE_probability_is_half_l2322_232268

/-- Represents a game board as described in the problem -/
structure GameBoard where
  total_regions : ℕ
  shaded_regions : ℕ
  h_total : total_regions = 8
  h_shaded : shaded_regions = 4

/-- The probability of landing in a shaded region on the game board -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem probability_is_half (board : GameBoard) : probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l2322_232268


namespace NUMINAMATH_CALUDE_joan_apple_picking_l2322_232248

/-- Given that Joan gave 27 apples to Melanie and now has 16 apples,
    prove that she picked 43 apples from the orchard. -/
theorem joan_apple_picking (apples_given : ℕ) (apples_left : ℕ) 
  (h1 : apples_given = 27) (h2 : apples_left = 16) :
  apples_given + apples_left = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_apple_picking_l2322_232248


namespace NUMINAMATH_CALUDE_expected_defective_theorem_l2322_232202

/-- The expected number of defective products drawn before a genuine product is drawn -/
def expected_defective_drawn (genuine : ℕ) (defective : ℕ) : ℚ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem expected_defective_theorem :
  expected_defective_drawn 9 3 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_theorem_l2322_232202


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2322_232295

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 && y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 3 ∧ θ = 5 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2322_232295


namespace NUMINAMATH_CALUDE_tangent_line_parabola_hyperbola_eccentricity_l2322_232251

/-- Given a line y = kx - 1 tangent to the parabola x² = 8y, 
    the eccentricity of the hyperbola x² - k²y² = 1 is equal to √3 -/
theorem tangent_line_parabola_hyperbola_eccentricity :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 = 8 * y ∧ 
   ∀ x' y' : ℝ, y' = k * x' - 1 → x'^2 ≠ 8 * y' ∨ (x' = x ∧ y' = y)) →
  Real.sqrt 3 = (Real.sqrt (1 + (1 / k^2))) / 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_hyperbola_eccentricity_l2322_232251


namespace NUMINAMATH_CALUDE_square_formation_total_l2322_232294

/-- Given a square formation of people where one person is the 5th from each side,
    prove that the total number of people is 81. -/
theorem square_formation_total (n : ℕ) (h : n = 5) :
  (2 * n - 1) * (2 * n - 1) = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_formation_total_l2322_232294


namespace NUMINAMATH_CALUDE_nested_radicals_solution_l2322_232209

-- Define the left-hand side of the equation
noncomputable def leftSide (x : ℝ) : ℝ := 
  Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- Define the right-hand side of the equation
noncomputable def rightSide (x : ℝ) : ℝ := 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

-- State the theorem
theorem nested_radicals_solution :
  ∃! (x : ℝ), x > 0 ∧ leftSide x = rightSide x :=
by
  -- The unique solution is 2
  use 2
  sorry

end NUMINAMATH_CALUDE_nested_radicals_solution_l2322_232209


namespace NUMINAMATH_CALUDE_katies_flour_amount_l2322_232233

theorem katies_flour_amount (katie_flour : ℕ) (sheila_flour : ℕ) : 
  sheila_flour = katie_flour + 2 →
  katie_flour + sheila_flour = 8 →
  katie_flour = 3 := by
sorry

end NUMINAMATH_CALUDE_katies_flour_amount_l2322_232233


namespace NUMINAMATH_CALUDE_parabola_vertex_l2322_232257

/-- The parabola defined by the equation y = (3x-1)^2 + 2 has vertex (1/3, 2) -/
theorem parabola_vertex (x y : ℝ) :
  y = (3*x - 1)^2 + 2 →
  (∃ a h k : ℝ, a ≠ 0 ∧ y = a*(x - h)^2 + k ∧ h = 1/3 ∧ k = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2322_232257


namespace NUMINAMATH_CALUDE_negation_of_forall_implication_l2322_232243

theorem negation_of_forall_implication (A B : Set α) :
  (¬ (∀ x, x ∈ A → x ∈ B)) ↔ (∃ x, x ∈ A ∧ x ∉ B) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_implication_l2322_232243


namespace NUMINAMATH_CALUDE_locus_of_D_l2322_232262

-- Define the basic structure for points in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a function to calculate the area of a triangle
def triangleArea (A B C : Point) : ℝ := sorry

-- Define a function to calculate the area of a quadrilateral
def quadArea (A B C D : Point) : ℝ := sorry

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop := sorry

-- Define a function to calculate the distance from a point to a line
def distanceToLine (P : Point) (A B : Point) : ℝ := sorry

-- Define a function to check if a point is on a line
def onLine (P : Point) (A B : Point) : Prop := sorry

-- Theorem statement
theorem locus_of_D (A B C D : Point) :
  ¬collinear A B C →
  quadArea A B C D = 3 * triangleArea A B C →
  ∃ (k : ℝ), distanceToLine D A C = 4 * distanceToLine B A C ∧
             ¬onLine D A B ∧
             ¬onLine D B C :=
sorry

end NUMINAMATH_CALUDE_locus_of_D_l2322_232262


namespace NUMINAMATH_CALUDE_b_most_suitable_l2322_232266

/-- Represents a candidate in the competition -/
structure Candidate where
  name : String
  average_score : ℝ
  variance : ℝ

/-- The set of all candidates -/
def candidates : Set Candidate :=
  { ⟨"A", 92.5, 3.4⟩, ⟨"B", 92.5, 2.1⟩, ⟨"C", 92.5, 2.5⟩, ⟨"D", 92.5, 2.7⟩ }

/-- Definition of the most suitable candidate -/
def most_suitable (c : Candidate) : Prop :=
  c ∈ candidates ∧
  ∀ d ∈ candidates, c.variance ≤ d.variance

/-- Theorem stating that B is the most suitable candidate -/
theorem b_most_suitable :
  ∃ c ∈ candidates, c.name = "B" ∧ most_suitable c := by
  sorry

end NUMINAMATH_CALUDE_b_most_suitable_l2322_232266


namespace NUMINAMATH_CALUDE_count_odd_numbers_between_150_and_350_l2322_232287

theorem count_odd_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 100 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_numbers_between_150_and_350_l2322_232287


namespace NUMINAMATH_CALUDE_inequality_proof_l2322_232282

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) 
  (h5 : a + b + c + d = 8) : 
  (a^3 / (a^2 + b + c)) + (b^3 / (b^2 + c + d)) + 
  (c^3 / (c^2 + d + a)) + (d^3 / (d^2 + a + b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2322_232282


namespace NUMINAMATH_CALUDE_arrange_six_books_three_identical_l2322_232225

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem arrange_six_books_three_identical :
  arrange_books 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_six_books_three_identical_l2322_232225


namespace NUMINAMATH_CALUDE_min_cubes_required_l2322_232299

-- Define the dimensions of the box
def box_length : ℕ := 9
def box_width : ℕ := 12
def box_height : ℕ := 3

-- Define the volume of a single cube
def cube_volume : ℕ := 3

-- Theorem: The minimum number of cubes required is 108
theorem min_cubes_required : 
  (box_length * box_width * box_height) / cube_volume = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_required_l2322_232299


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_500_l2322_232218

def multiple_of_15 (n : ℤ) : ℤ := 15 * n

theorem closest_multiple_of_15_to_500 :
  ∀ k : ℤ, k ≠ 33 → |500 - multiple_of_15 33| ≤ |500 - multiple_of_15 k| :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_500_l2322_232218


namespace NUMINAMATH_CALUDE_simplify_expression_l2322_232250

theorem simplify_expression (a : ℝ) : a + 1 + a - 2 + a + 3 + a - 4 = 4*a - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2322_232250


namespace NUMINAMATH_CALUDE_total_savings_l2322_232279

/-- Represents the savings of Anne and Katherine -/
structure Savings where
  anne : ℝ
  katherine : ℝ

/-- The conditions of the savings problem -/
def SavingsConditions (s : Savings) : Prop :=
  (s.anne - 150 = (1 / 3) * s.katherine) ∧
  (2 * s.katherine = 3 * s.anne)

/-- Theorem stating that under the given conditions, the total savings is $750 -/
theorem total_savings (s : Savings) (h : SavingsConditions s) : 
  s.anne + s.katherine = 750 := by
  sorry

#check total_savings

end NUMINAMATH_CALUDE_total_savings_l2322_232279


namespace NUMINAMATH_CALUDE_last_year_production_l2322_232215

/-- The number of eggs produced by farms in Douglas County --/
structure EggProduction where
  lastYear : ℕ
  thisYear : ℕ
  increase : ℕ

/-- Theorem stating the relationship between this year's and last year's egg production --/
theorem last_year_production (e : EggProduction) 
  (h1 : e.thisYear = 4636)
  (h2 : e.increase = 3220)
  (h3 : e.thisYear = e.lastYear + e.increase) :
  e.lastYear = 1416 := by
  sorry

end NUMINAMATH_CALUDE_last_year_production_l2322_232215


namespace NUMINAMATH_CALUDE_f_negative_two_eq_one_l2322_232241

/-- The function f(x) defined as x^5 + ax^3 + x^2 + bx + 2 -/
noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + x^2 + b*x + 2

/-- Theorem: If f(2) = 3, then f(-2) = 1 -/
theorem f_negative_two_eq_one (a b : ℝ) (h : f a b 2 = 3) : f a b (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_two_eq_one_l2322_232241


namespace NUMINAMATH_CALUDE_race_theorem_l2322_232223

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance_run : ℝ → ℝ

/-- The race setup -/
structure Race where
  petya : Runner
  kolya : Runner
  vasya : Runner
  race_distance : ℝ

def Race.valid (r : Race) : Prop :=
  r.race_distance = 100 ∧
  r.petya.speed > 0 ∧ r.kolya.speed > 0 ∧ r.vasya.speed > 0 ∧
  r.petya.distance_run 0 = 0 ∧ r.kolya.distance_run 0 = 0 ∧ r.vasya.distance_run 0 = 0 ∧
  ∀ t, r.petya.distance_run t = r.petya.speed * t ∧
       r.kolya.distance_run t = r.kolya.speed * t ∧
       r.vasya.distance_run t = r.vasya.speed * t

def Race.petya_finishes_first (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance ∧
       r.kolya.distance_run t < r.race_distance ∧
       r.vasya.distance_run t < r.race_distance

def Race.half_distance_condition (r : Race) : Prop :=
  ∃ t, r.petya.distance_run t = r.race_distance / 2 ∧
       r.kolya.distance_run t + r.vasya.distance_run t = 85

theorem race_theorem (r : Race) (h_valid : r.valid) (h_first : r.petya_finishes_first)
    (h_half : r.half_distance_condition) :
    ∃ t, r.petya.distance_run t = r.race_distance ∧
         2 * r.race_distance - (r.kolya.distance_run t + r.vasya.distance_run t) = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l2322_232223


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2322_232275

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x < 2}
  let B : Set ℝ := {x | 3 - 2*x > 0}
  A ∩ B = {x | x < 3/2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2322_232275


namespace NUMINAMATH_CALUDE_walnut_count_l2322_232211

theorem walnut_count (total : ℕ) (p a c w : ℕ) : 
  total = 150 →
  p + a + c + w = total →
  a = p / 2 →
  c = 4 * a →
  w = 3 * c →
  w = 96 := by
  sorry

end NUMINAMATH_CALUDE_walnut_count_l2322_232211


namespace NUMINAMATH_CALUDE_cosine_value_from_sine_l2322_232229

theorem cosine_value_from_sine (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2 + π / 6) = 3 / 5) : 
  Real.cos (θ + 5 * π / 6) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_sine_l2322_232229


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l2322_232208

/-- The total time spent recovering escaped animals from a zoo -/
def total_recovery_time (num_lions num_rhinos num_giraffes num_gorillas : ℕ) 
  (time_per_lion time_per_rhino time_per_giraffe time_per_gorilla : ℝ) : ℝ :=
  (num_lions : ℝ) * time_per_lion + 
  (num_rhinos : ℝ) * time_per_rhino + 
  (num_giraffes : ℝ) * time_per_giraffe + 
  (num_gorillas : ℝ) * time_per_gorilla

/-- Theorem stating that the total recovery time for the given scenario is 33 hours -/
theorem zoo_recovery_time : 
  total_recovery_time 5 3 2 4 2 3 4 1.5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l2322_232208


namespace NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l2322_232265

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "within" relation for a line in a plane
variable (within : Line → Plane → Prop)

theorem perp_necessary_not_sufficient
  (l m : Line) (α : Plane)
  (h_diff : l ≠ m)
  (h_within : within m α) :
  (∀ x : Line, within x α → perp_line_plane l α → perp_line_line l x) ∧
  (∃ β : Plane, perp_line_line l m ∧ ¬perp_line_plane l β ∧ within m β) :=
sorry

end NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l2322_232265


namespace NUMINAMATH_CALUDE_real_roots_range_l2322_232246

theorem real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) ↔ 
  (m ≤ -3/2 ∨ m ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_real_roots_range_l2322_232246


namespace NUMINAMATH_CALUDE_complement_of_angle_l2322_232231

theorem complement_of_angle (A : ℝ) (h : A = 35) : 90 - A = 55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_l2322_232231


namespace NUMINAMATH_CALUDE_modified_geometric_series_sum_l2322_232288

/-- The sum of a modified geometric series -/
theorem modified_geometric_series_sum 
  (a r : ℝ) 
  (h_r : -1 < r ∧ r < 1) :
  let series_sum : ℕ → ℝ := λ n => a^2 * r^(3*n)
  ∑' n, series_sum n = a^2 / (1 - r^3) := by
  sorry

end NUMINAMATH_CALUDE_modified_geometric_series_sum_l2322_232288


namespace NUMINAMATH_CALUDE_tom_candy_left_l2322_232240

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10

def total_candy : ℕ := initial_candy + friend_candy + bought_candy

def candy_left : ℕ := total_candy - (total_candy / 2)

theorem tom_candy_left : candy_left = 10 := by sorry

end NUMINAMATH_CALUDE_tom_candy_left_l2322_232240


namespace NUMINAMATH_CALUDE_inequality_proof_l2322_232278

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2322_232278


namespace NUMINAMATH_CALUDE_zoe_winter_clothing_l2322_232289

/-- The number of boxes of winter clothing Zoe has. -/
def num_boxes : ℕ := 8

/-- The number of scarves in each box. -/
def scarves_per_box : ℕ := 4

/-- The number of mittens in each box. -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Zoe has. -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem zoe_winter_clothing :
  total_pieces = 80 :=
by sorry

end NUMINAMATH_CALUDE_zoe_winter_clothing_l2322_232289


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2322_232226

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- arithmetic sequence
  (p q : ℕ)    -- indices
  (h1 : a p = 4)
  (h2 : a q = 2)
  (h3 : p = 4 + q)
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) :  -- arithmetic sequence property
  ∃ d : ℝ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2322_232226


namespace NUMINAMATH_CALUDE_average_customer_donation_l2322_232238

/-- Given a restaurant fundraiser where:
    1. The restaurant's donation is 1/5 of the total customer donation.
    2. There are 40 customers.
    3. The restaurant's total donation is $24.
    Prove that the average customer donation is $3. -/
theorem average_customer_donation (restaurant_ratio : ℚ) (num_customers : ℕ) (restaurant_donation : ℚ) :
  restaurant_ratio = 1 / 5 →
  num_customers = 40 →
  restaurant_donation = 24 →
  (restaurant_donation / restaurant_ratio) / num_customers = 3 := by
sorry

end NUMINAMATH_CALUDE_average_customer_donation_l2322_232238


namespace NUMINAMATH_CALUDE_range_of_f_l2322_232236

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x + 2)

theorem range_of_f :
  Set.range f = Set.Ioo 0 (1/2) ∪ {1/2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2322_232236


namespace NUMINAMATH_CALUDE_complex_inequality_l2322_232284

theorem complex_inequality (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  Complex.abs (z - w) ≥ (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ∧
  (Complex.abs (z - w) = (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ↔
   (z / w).re < 0 ∨ Complex.abs z = Complex.abs w) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l2322_232284


namespace NUMINAMATH_CALUDE_sibling_pair_probability_l2322_232222

theorem sibling_pair_probability (business_students : ℕ) (law_students : ℕ) (sibling_pairs : ℕ) : 
  business_students = 500 →
  law_students = 800 →
  sibling_pairs = 30 →
  (sibling_pairs : ℚ) / (business_students * law_students) = 0.000075 := by
  sorry

end NUMINAMATH_CALUDE_sibling_pair_probability_l2322_232222


namespace NUMINAMATH_CALUDE_tim_final_coin_count_l2322_232232

/-- Represents the count of different types of coins -/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ
  pennies : ℕ

/-- Represents a transaction that modifies the coin count -/
inductive Transaction
  | DadGift : Transaction
  | DadExchange : Transaction
  | PaySister : Transaction
  | BuySnack : Transaction
  | ExchangeQuarter : Transaction

def initial_coins : CoinCount :=
  { quarters := 7, nickels := 9, dimes := 12, pennies := 5 }

def apply_transaction (coins : CoinCount) (t : Transaction) : CoinCount :=
  match t with
  | Transaction.DadGift => 
      { quarters := coins.quarters + 2,
        nickels := coins.nickels + 3,
        dimes := coins.dimes,
        pennies := coins.pennies + 5 }
  | Transaction.DadExchange => 
      { quarters := coins.quarters + 4,
        nickels := coins.nickels,
        dimes := coins.dimes - 10,
        pennies := coins.pennies }
  | Transaction.PaySister => 
      { quarters := coins.quarters,
        nickels := coins.nickels - 5,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.BuySnack => 
      { quarters := coins.quarters - 2,
        nickels := coins.nickels - 4,
        dimes := coins.dimes,
        pennies := coins.pennies }
  | Transaction.ExchangeQuarter => 
      { quarters := coins.quarters - 1,
        nickels := coins.nickels + 5,
        dimes := coins.dimes,
        pennies := coins.pennies }

def final_coins : CoinCount :=
  apply_transaction
    (apply_transaction
      (apply_transaction
        (apply_transaction
          (apply_transaction initial_coins Transaction.DadGift)
          Transaction.DadExchange)
        Transaction.PaySister)
      Transaction.BuySnack)
    Transaction.ExchangeQuarter

theorem tim_final_coin_count :
  final_coins = { quarters := 10, nickels := 8, dimes := 2, pennies := 10 } :=
by sorry

end NUMINAMATH_CALUDE_tim_final_coin_count_l2322_232232


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l2322_232296

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 10}

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def B : Set ℕ := {x | isPrime x}

theorem intersection_complement_equals_set : A ∩ (I \ B) = {4, 6, 8, 9, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l2322_232296


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2322_232264

theorem necessary_and_sufficient_condition (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2322_232264


namespace NUMINAMATH_CALUDE_gregs_dog_walking_rate_l2322_232290

/-- Greg's dog walking earnings problem -/
theorem gregs_dog_walking_rate :
  ∀ (rate : ℚ),
  (20 + 10 * rate) +   -- One dog for 10 minutes
  2 * (20 + 7 * rate) +  -- Two dogs for 7 minutes each
  3 * (20 + 9 * rate) = 171  -- Three dogs for 9 minutes each
  →
  rate = 1 := by
sorry

end NUMINAMATH_CALUDE_gregs_dog_walking_rate_l2322_232290


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2322_232205

theorem binomial_expansion_sum (a₁ a₂ : ℕ) : 
  (∀ k : ℕ, k ≤ 10 → a₁ = 20 ∧ a₂ = 180) → 
  a₁ + a₂ = 200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2322_232205


namespace NUMINAMATH_CALUDE_brick_length_l2322_232235

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The length of a brick with given dimensions and surface area -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 2) (hSA : SA = 112) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 8 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_l2322_232235


namespace NUMINAMATH_CALUDE_zero_of_composite_f_l2322_232201

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

-- State the theorem
theorem zero_of_composite_f :
  ∃ (x : ℝ), f (f x) = 0 ∧ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_composite_f_l2322_232201


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2322_232271

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  MonicQuarticPolynomial p →
  p 2 = 7 →
  p 3 = 12 →
  p 4 = 19 →
  p 5 = 28 →
  p 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2322_232271


namespace NUMINAMATH_CALUDE_venerts_in_45_degrees_l2322_232274

/-- Converts degrees to venerts given the number of venerts in a full circle -/
def degrees_to_venerts (full_circle_venerts : ℚ) (degrees : ℚ) : ℚ :=
  (degrees * full_circle_venerts) / 360

/-- Theorem: Given 600 venerts in a full circle, 45° is equivalent to 75 venerts -/
theorem venerts_in_45_degrees :
  degrees_to_venerts 600 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_venerts_in_45_degrees_l2322_232274


namespace NUMINAMATH_CALUDE_smallest_portion_l2322_232285

def bread_distribution (a : ℚ) (d : ℚ) : Prop :=
  -- Total sum is 100
  5 * a + 10 * d = 100 ∧
  -- Sum of largest three portions is 1/7 of sum of smaller two
  (3 * a + 6 * d) = (1/7) * (2 * a + d)

theorem smallest_portion : 
  ∃ (a d : ℚ), bread_distribution a d ∧ a = 5/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_portion_l2322_232285


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l2322_232276

/-- Given the cost of pens and pencils, prove the cost of one dozen pens -/
theorem cost_of_dozen_pens 
  (cost_3pens_5pencils : ℕ) 
  (ratio : ℚ) 
  (cost_dozen_pens : ℕ) 
  (h1 : cost_3pens_5pencils = 100)
  (h2 : ratio > 0)
  (h3 : cost_dozen_pens = 300) :
  cost_dozen_pens = 300 := by
  sorry

#check cost_of_dozen_pens

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l2322_232276


namespace NUMINAMATH_CALUDE_dawn_time_verify_solution_l2322_232210

/-- Represents the time in hours from dawn to noon -/
def time_dawn_to_noon : ℝ := sorry

/-- Represents the time in hours from noon to 4 PM -/
def time_noon_to_4pm : ℝ := 4

/-- Represents the time in hours from noon to 9 PM -/
def time_noon_to_9pm : ℝ := 9

/-- The theorem stating that the time from dawn to noon is 6 hours -/
theorem dawn_time : time_dawn_to_noon = 6 := by
  sorry

/-- Verification of the solution using speed ratios -/
theorem verify_solution :
  time_dawn_to_noon / time_noon_to_4pm = time_noon_to_9pm / time_dawn_to_noon := by
  sorry

end NUMINAMATH_CALUDE_dawn_time_verify_solution_l2322_232210


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_four_l2322_232259

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem subset_implies_m_equals_four (m : ℝ) : B ⊆ A m → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_four_l2322_232259


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2322_232200

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K * Real.sqrt (x + y + z)) ∧
  (∀ (L : ℝ), 
    (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
      Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ L * Real.sqrt (x + y + z)) →
    L ≤ K) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2322_232200


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2322_232255

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0)
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2322_232255


namespace NUMINAMATH_CALUDE_helga_wrote_250_articles_l2322_232254

/-- Represents Helga's work schedule and article production --/
structure HelgaWork where
  articles_per_30min : ℕ := 5
  usual_hours_per_day : ℕ := 4
  usual_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (h : HelgaWork) : ℕ :=
  let articles_per_hour := h.articles_per_30min * 2
  let usual_articles_per_day := articles_per_hour * h.usual_hours_per_day
  let usual_articles_per_week := usual_articles_per_day * h.usual_days_per_week
  let extra_articles_thursday := articles_per_hour * h.extra_hours_thursday
  let extra_articles_friday := articles_per_hour * h.extra_hours_friday
  usual_articles_per_week + extra_articles_thursday + extra_articles_friday

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (h : HelgaWork), total_articles_in_week h = 250 := by
  sorry

end NUMINAMATH_CALUDE_helga_wrote_250_articles_l2322_232254


namespace NUMINAMATH_CALUDE_min_product_of_three_positive_reals_l2322_232247

theorem min_product_of_three_positive_reals (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 1/18 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_positive_reals_l2322_232247


namespace NUMINAMATH_CALUDE_dragon_rope_problem_l2322_232230

-- Define the constants
def rope_length : ℝ := 25
def castle_radius : ℝ := 5
def dragon_height : ℝ := 3
def rope_end_distance : ℝ := 3

-- Define the variables
variable (p q r : ℕ)

-- Define the conditions
axiom p_positive : p > 0
axiom q_positive : q > 0
axiom r_positive : r > 0
axiom r_prime : Nat.Prime r

-- Define the relationship between p, q, r and the rope length touching the castle
axiom rope_touching_castle : (p - Real.sqrt q) / r = (75 - Real.sqrt 450) / 3

-- Theorem to prove
theorem dragon_rope_problem : p + q + r = 528 := by sorry

end NUMINAMATH_CALUDE_dragon_rope_problem_l2322_232230


namespace NUMINAMATH_CALUDE_hot_dog_contest_l2322_232217

/-- Hot dog eating contest problem -/
theorem hot_dog_contest (first_competitor second_competitor third_competitor : ℕ) : 
  first_competitor = 12 →
  second_competitor = 2 * first_competitor →
  third_competitor = second_competitor - (second_competitor / 4) →
  third_competitor = 18 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_contest_l2322_232217


namespace NUMINAMATH_CALUDE_bug_probability_7_l2322_232269

/-- Probability of a bug being at the starting vertex of a regular tetrahedron after n steps -/
def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => (1 / 3) * (1 - bug_probability m)

/-- The probability of the bug being at the starting vertex after 7 steps is 182/729 -/
theorem bug_probability_7 : bug_probability 7 = 182 / 729 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_7_l2322_232269


namespace NUMINAMATH_CALUDE_product_loop_result_l2322_232244

def product_loop (i : ℕ) : ℕ :=
  if i < 11 then 1 else i * product_loop (i - 1)

theorem product_loop_result :
  product_loop 12 = 132 :=
by sorry

end NUMINAMATH_CALUDE_product_loop_result_l2322_232244


namespace NUMINAMATH_CALUDE_match_triangle_formation_l2322_232214

theorem match_triangle_formation (n : ℕ) : 
  (n = 100 → ¬(3 ∣ (n * (n + 1) / 2))) ∧ 
  (n = 99 → (3 ∣ (n * (n + 1) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_match_triangle_formation_l2322_232214


namespace NUMINAMATH_CALUDE_real_number_inequalities_l2322_232293

theorem real_number_inequalities (a b c : ℝ) : 
  (a > b → a > (a + b) / 2 ∧ (a + b) / 2 > b) ∧
  (a > b ∧ b > 0 → a > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > b) ∧
  (a > b ∧ b > 0 ∧ c > 0 → (b + c) / (a + c) > b / a) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l2322_232293


namespace NUMINAMATH_CALUDE_painting_selection_theorem_l2322_232267

/-- Number of traditional Chinese paintings -/
def traditional_paintings : ℕ := 5

/-- Number of oil paintings -/
def oil_paintings : ℕ := 2

/-- Number of watercolor paintings -/
def watercolor_paintings : ℕ := 7

/-- Number of ways to choose one painting from each category -/
def one_from_each : ℕ := traditional_paintings * oil_paintings * watercolor_paintings

/-- Number of ways to choose two paintings of different types -/
def two_different_types : ℕ := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem : 
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end NUMINAMATH_CALUDE_painting_selection_theorem_l2322_232267


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l2322_232280

/-- Two externally tangent circles C₁ and C₂ -/
structure TangentCircles where
  /-- Equation of C₁: (x+2)² + (y-m)² = 9 -/
  c1 : ∀ (x y : ℝ), (x + 2)^2 + (y - m)^2 = 9
  /-- Equation of C₂: (x-m)² + (y+1)² = 4 -/
  c2 : ∀ (x y : ℝ), (x - m)^2 + (y + 1)^2 = 4
  /-- m is a real number -/
  m : ℝ

/-- The value of m for externally tangent circles C₁ and C₂ is either 2 or -5 -/
theorem tangent_circles_m_value (tc : TangentCircles) : tc.m = 2 ∨ tc.m = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l2322_232280


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l2322_232212

/-- Represents the payment for Scheme 1 -/
def scheme1_payment (x : ℕ) : ℚ :=
  90 * (1 - 30/100) * x + 100 * (1 - 15/100) * (2*x + 1)

/-- Represents the payment for Scheme 2 -/
def scheme2_payment (x : ℕ) : ℚ :=
  (90*x + 100*(2*x + 1)) * (1 - 20/100)

/-- Theorem stating that Scheme 2 is more cost-effective for x ≥ 33 -/
theorem scheme2_more_cost_effective (x : ℕ) (h : x ≥ 33) :
  scheme2_payment x < scheme1_payment x :=
sorry

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l2322_232212


namespace NUMINAMATH_CALUDE_calculate_expression_l2322_232228

theorem calculate_expression : 4 + (-2)^2 * 2 + (-36) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2322_232228


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2322_232270

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = m ∧
    r₁ / 2 + r₂ / 2 = -m ∧ (r₁ / 2) * (r₂ / 2) = n) →
  n / p = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2322_232270


namespace NUMINAMATH_CALUDE_balloon_ratio_l2322_232224

theorem balloon_ratio (sally_balloons fred_balloons : ℕ) 
  (h1 : sally_balloons = 6) 
  (h2 : fred_balloons = 18) : 
  (fred_balloons : ℚ) / sally_balloons = 3 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_l2322_232224


namespace NUMINAMATH_CALUDE_aeroplane_speed_l2322_232292

theorem aeroplane_speed (distance : ℝ) (time1 : ℝ) (time2 : ℝ) (speed2 : ℝ) :
  time1 = 6 →
  time2 = 14 / 3 →
  speed2 = 540 →
  distance = speed2 * time2 →
  distance = (distance / time1) * time1 →
  distance / time1 = 420 := by
sorry

end NUMINAMATH_CALUDE_aeroplane_speed_l2322_232292


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2322_232253

/-- Given a total sum split into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, then the second part is 1664 Rs. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first second : ℝ) :
  total = 2704 →
  first + second = total →
  (first * 3 * 8) / 100 = (second * 5 * 3) / 100 →
  second = 1664 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l2322_232253


namespace NUMINAMATH_CALUDE_gymnasium_doubles_players_l2322_232227

theorem gymnasium_doubles_players (total_tables : ℕ) 
  (h1 : total_tables = 13) 
  (h2 : ∀ x y : ℕ, x + y = total_tables → 4 * x - 2 * y = 4 → 4 * x = 20) :
  ∃ x y : ℕ, x + y = total_tables ∧ 4 * x - 2 * y = 4 ∧ 4 * x = 20 :=
sorry

end NUMINAMATH_CALUDE_gymnasium_doubles_players_l2322_232227


namespace NUMINAMATH_CALUDE_caitlin_age_is_24_l2322_232239

/-- The age of Aunt Anna in years -/
def aunt_anna_age : ℕ := 45

/-- The age of Brianna in years -/
def brianna_age : ℕ := (2 * aunt_anna_age) / 3

/-- The age difference between Brianna and Caitlin in years -/
def age_difference : ℕ := 6

/-- The age of Caitlin in years -/
def caitlin_age : ℕ := brianna_age - age_difference

/-- Theorem stating Caitlin's age -/
theorem caitlin_age_is_24 : caitlin_age = 24 := by sorry

end NUMINAMATH_CALUDE_caitlin_age_is_24_l2322_232239


namespace NUMINAMATH_CALUDE_concatenation_problem_l2322_232249

theorem concatenation_problem :
  ∃ (a b : ℕ),
    100 ≤ a ∧ a ≤ 999 ∧
    1000 ≤ b ∧ b ≤ 9999 ∧
    10000 * a + b = 11 * a * b ∧
    a + b = 1093 := by
  sorry

end NUMINAMATH_CALUDE_concatenation_problem_l2322_232249


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l2322_232256

/-- The gain percent when a cycle is bought for 450 Rs and sold for 520 Rs -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 15.56% -/
theorem cycle_gain_percent : 
  gain_percent 450 520 = 15.56 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l2322_232256
