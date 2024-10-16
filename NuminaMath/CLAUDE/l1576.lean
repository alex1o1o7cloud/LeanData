import Mathlib

namespace NUMINAMATH_CALUDE_particle_position_after_1989_minutes_l1576_157648

/-- Represents the position of a particle in 2D space -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Calculates the time taken to enclose n squares -/
def timeForSquares (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the position of the particle after 1989 minutes -/
theorem particle_position_after_1989_minutes :
  particlePosition 1989 = Position.mk 44 35 := by sorry

end NUMINAMATH_CALUDE_particle_position_after_1989_minutes_l1576_157648


namespace NUMINAMATH_CALUDE_unique_f_zero_unique_solution_l1576_157646

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(0) = 1 is the only valid solution -/
theorem unique_f_zero (f : ℝ → ℝ) (h : FunctionalEq f) : f 0 = 1 := by
  sorry

/-- The theorem stating that f(x) = x² + 1 is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEq f) : 
  ∀ x : ℝ, f x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_zero_unique_solution_l1576_157646


namespace NUMINAMATH_CALUDE_nth_term_equation_l1576_157641

theorem nth_term_equation (n : ℕ) : 
  Real.sqrt ((2 * n^2 : ℝ) / (2 * n + 1) - (n - 1)) = Real.sqrt ((n + 1) * (2 * n + 1)) / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_term_equation_l1576_157641


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l1576_157665

theorem absolute_value_equation_solution_product : 
  ∃ (x₁ x₂ : ℝ), 
    (|18 / x₁ + 4| = 3) ∧ 
    (|18 / x₂ + 4| = 3) ∧ 
    (x₁ ≠ x₂) ∧ 
    (x₁ * x₂ = 324 / 7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l1576_157665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1576_157688

/-- An arithmetic sequence {a_n} with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 4 + seq.a 5 = 24) 
  (h2 : seq.S 6 = 48) : 
  common_difference seq = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1576_157688


namespace NUMINAMATH_CALUDE_lcm_max_value_l1576_157695

theorem lcm_max_value : 
  let lcm_values := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10, Nat.lcm 15 18]
  List.maximum lcm_values = some 90 := by
sorry

end NUMINAMATH_CALUDE_lcm_max_value_l1576_157695


namespace NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l1576_157645

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Theorem: x^2 - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l1576_157645


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l1576_157676

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Hyperbola type -/
structure Hyperbola where
  equation : ℝ → ℝ → ℝ → Prop
  a : ℝ

/-- Theorem: Eccentricity of hyperbola given specific conditions -/
theorem hyperbola_eccentricity_sqrt_6
  (p : Parabola)
  (h : Hyperbola)
  (A B : ℝ × ℝ)
  (h_parabola : p.equation = fun x y ↦ y^2 = 4*x)
  (h_hyperbola : h.equation = fun x y a ↦ x^2/a^2 - y^2 = 1)
  (h_a_pos : h.a > 0)
  (h_intersection : p.equation A.1 A.2 ∧ h.equation A.1 A.2 h.a ∧
                    p.equation B.1 B.2 ∧ h.equation B.1 B.2 h.a)
  (h_right_angle : (A.1 - p.focus.1) * (B.1 - p.focus.1) +
                   (A.2 - p.focus.2) * (B.2 - p.focus.2) = 0) :
  ∃ (e : ℝ), e^2 = 6 ∧ e = (Real.sqrt ((h.a^2 + 1) / h.a^2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_6_l1576_157676


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1576_157629

theorem quadratic_solution_sum (a b c : ℝ) : a ≠ 0 → (∀ x, a * x^2 + b * x + c = 0 ↔ x = 1) → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1576_157629


namespace NUMINAMATH_CALUDE_synthetic_analytic_direct_l1576_157692

-- Define proof methods
structure ProofMethod where
  name : String
  direction : String
  isDirect : Bool

-- Define synthetic and analytic methods
def synthetic : ProofMethod := {
  name := "Synthetic",
  direction := "cause to effect",
  isDirect := true
}

def analytic : ProofMethod := {
  name := "Analytic",
  direction := "effect to cause",
  isDirect := true
}

-- Theorem statement
theorem synthetic_analytic_direct :
  synthetic.isDirect ∧ analytic.isDirect :=
sorry

end NUMINAMATH_CALUDE_synthetic_analytic_direct_l1576_157692


namespace NUMINAMATH_CALUDE_restaurant_students_l1576_157684

theorem restaurant_students (burger_orders : ℕ) (hotdog_orders : ℕ) : 
  burger_orders = 30 →
  burger_orders = 2 * hotdog_orders →
  burger_orders + hotdog_orders = 45 := by
sorry

end NUMINAMATH_CALUDE_restaurant_students_l1576_157684


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1576_157628

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1576_157628


namespace NUMINAMATH_CALUDE_solution_is_83_l1576_157605

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (x : ℝ) : Prop :=
  log 3 (x^2 - 3) + log 9 (x - 2) + log (1/3) (x^2 - 3) = 2

-- Theorem statement
theorem solution_is_83 :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ x = 83 :=
by sorry

end NUMINAMATH_CALUDE_solution_is_83_l1576_157605


namespace NUMINAMATH_CALUDE_tetrahedron_volume_with_inscribed_sphere_l1576_157606

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume_with_inscribed_sphere
  (R : ℝ)  -- Radius of the inscribed sphere
  (S₁ S₂ S₃ S₄ : ℝ)  -- Areas of the four faces of the tetrahedron
  (h₁ : R > 0)  -- Radius is positive
  (h₂ : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)  -- Face areas are positive
  : ∃ V : ℝ, V = R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_with_inscribed_sphere_l1576_157606


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1576_157677

theorem cost_price_percentage (selling_price cost_price : ℝ) 
  (h_profit_percent : (selling_price - cost_price) / cost_price = 1/3) :
  cost_price / selling_price = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1576_157677


namespace NUMINAMATH_CALUDE_rectangle_length_equals_6_3_l1576_157680

-- Define the parameters
def triangle_base : ℝ := 7.2
def triangle_height : ℝ := 7
def rectangle_width : ℝ := 4

-- Define the theorem
theorem rectangle_length_equals_6_3 :
  let triangle_area := (triangle_base * triangle_height) / 2
  let rectangle_length := triangle_area / rectangle_width
  rectangle_length = 6.3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_6_3_l1576_157680


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l1576_157661

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem cosine_value_in_triangle (t : Triangle) 
  (hm : Vector2D := ⟨Real.sqrt 3 * t.b - t.c, Real.cos t.C⟩)
  (hn : Vector2D := ⟨t.a, Real.cos t.A⟩)
  (h_parallel : parallel hm hn) :
  Real.cos t.A = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_value_in_triangle_l1576_157661


namespace NUMINAMATH_CALUDE_probability_square_or_triangle_l1576_157615

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- Theorem statement
theorem probability_square_or_triangle :
  (num_triangles + num_squares : ℚ) / total_figures = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_triangle_l1576_157615


namespace NUMINAMATH_CALUDE_aj_has_370_stamps_l1576_157669

/-- The number of stamps each person has -/
structure Stamps where
  aj : ℕ
  kj : ℕ
  cj : ℕ

/-- The conditions of the stamp collection problem -/
def stamp_problem (s : Stamps) : Prop :=
  s.cj = 5 + 2 * s.kj ∧
  s.kj = s.aj / 2 ∧
  s.aj + s.kj + s.cj = 930

/-- The theorem stating that AJ has 370 stamps -/
theorem aj_has_370_stamps :
  ∃ (s : Stamps), stamp_problem s ∧ s.aj = 370 :=
by
  sorry


end NUMINAMATH_CALUDE_aj_has_370_stamps_l1576_157669


namespace NUMINAMATH_CALUDE_worker_travel_time_l1576_157697

/-- Proves that the usual travel time is 40 minutes given the conditions -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) 
  (h2 : normal_time > 0) 
  (h3 : normal_speed * normal_time = (4/5 * normal_speed) * (normal_time + 10)) : 
  normal_time = 40 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1576_157697


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1576_157602

-- Problem 1
theorem problem_one : 2 * Real.cos (π / 4) + |1 - Real.sqrt 2| + (-2) ^ 0 = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two (a : ℝ) : 3 * a + 2 * a * (a - 1) = 2 * a ^ 2 + a := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1576_157602


namespace NUMINAMATH_CALUDE_equation_value_l1576_157608

theorem equation_value (x y : ℝ) (h : x - 3*y = 4) : 
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l1576_157608


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1576_157614

/-- Given a triangle ABC with sides a = 7, b = 5, and c = 3, 
    the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) :
  let a := ‖B - C‖
  let b := ‖A - C‖
  let c := ‖A - B‖
  a = 7 ∧ b = 5 ∧ c = 3 →
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) * (180 / Real.pi) = 120 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_measure_l1576_157614


namespace NUMINAMATH_CALUDE_third_player_games_l1576_157630

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules of the game ensure that the total games is the sum of games played by any two players -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games + g.player2_games ∧
  g.total_games = g.player1_games + g.player3_games ∧
  g.total_games = g.player2_games + g.player3_games

/-- The theorem to be proved -/
theorem third_player_games (g : TableTennisGame) 
  (h1 : g.player1_games = 10)
  (h2 : g.player2_games = 21)
  (h3 : valid_game g) :
  g.player3_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_third_player_games_l1576_157630


namespace NUMINAMATH_CALUDE_days_of_sending_roses_l1576_157609

def roses_per_day : ℕ := 24  -- 2 dozen roses per day
def total_roses : ℕ := 168   -- total number of roses sent

theorem days_of_sending_roses : 
  total_roses / roses_per_day = 7 :=
sorry

end NUMINAMATH_CALUDE_days_of_sending_roses_l1576_157609


namespace NUMINAMATH_CALUDE_cos_squared_half_diff_l1576_157622

theorem cos_squared_half_diff (α β : Real) 
  (h1 : Real.sin α + Real.sin β = Real.sqrt 6 / 3)
  (h2 : Real.cos α + Real.cos β = Real.sqrt 3 / 3) : 
  (Real.cos ((α - β) / 2))^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_half_diff_l1576_157622


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1576_157662

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car traveling 40 km in the first hour and increasing speed by 2 km/h
    every hour will travel 600 km in 12 hours. -/
theorem car_distance_theorem : total_distance 40 2 12 = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1576_157662


namespace NUMINAMATH_CALUDE_correct_transformation_l1576_157668

theorem correct_transformation (x : ℝ) : 3 * x + 5 = 4 * x → 3 * x - 4 * x = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1576_157668


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1576_157652

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 1734 → s^3 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1576_157652


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1576_157607

theorem complex_equation_sum (x y : ℝ) :
  (x - Complex.I) * Complex.I = y + 2 * Complex.I →
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1576_157607


namespace NUMINAMATH_CALUDE_two_removable_cells_exist_l1576_157621

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Bool

-- Define a cell position
structure CellPosition where
  row : Fin 4
  col : Fin 4

-- Define a function to remove a cell from the grid
def removeCell (g : Grid) (pos : CellPosition) : Grid :=
  fun r c => if r = pos.row ∧ c = pos.col then false else g r c

-- Define congruence between two parts of the grid
def isCongruent (part1 part2 : Set CellPosition) : Prop := sorry

-- Define a function to check if a grid can be divided into three congruent parts
def canDivideIntoThreeCongruentParts (g : Grid) : Prop := sorry

-- Theorem statement
theorem two_removable_cells_exist :
  ∃ (pos1 pos2 : CellPosition),
    pos1 ≠ pos2 ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos1) ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos2) := by
  sorry


end NUMINAMATH_CALUDE_two_removable_cells_exist_l1576_157621


namespace NUMINAMATH_CALUDE_rhombus_min_rotation_l1576_157675

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The minimum rotation angle for a rhombus to coincide with its original position -/
def min_rotation_angle (r : Rhombus) : ℝ := 180

/-- Theorem: The minimum rotation angle for a rhombus with a 60° angle to coincide with its original position is 180° -/
theorem rhombus_min_rotation (r : Rhombus) (angle : ℝ) (h : angle = 60) :
  min_rotation_angle r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_min_rotation_l1576_157675


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1576_157689

/-- The range of c for which there are four points on the circle x^2 + y^2 = 4
    at a distance of 1 from the line 12x - 5y + c = 0 is (-13, 13) -/
theorem circle_line_distance_range :
  ∀ c : ℝ,
  (∃ (points : Finset (ℝ × ℝ)),
    points.card = 4 ∧
    (∀ (x y : ℝ), (x, y) ∈ points →
      x^2 + y^2 = 4 ∧
      (|12*x - 5*y + c| / Real.sqrt (12^2 + (-5)^2) = 1))) ↔
  -13 < c ∧ c < 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l1576_157689


namespace NUMINAMATH_CALUDE_remainder_after_adding_2040_l1576_157682

theorem remainder_after_adding_2040 (n : ℤ) (h : n % 8 = 3) : (n + 2040) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2040_l1576_157682


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1576_157685

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 :=
sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1576_157685


namespace NUMINAMATH_CALUDE_marys_marbles_l1576_157644

/-- Given that Mary has 9.0 yellow marbles initially and gives 3.0 yellow marbles to Joan,
    prove that Mary will have 6.0 yellow marbles left. -/
theorem marys_marbles (initial : ℝ) (given : ℝ) (left : ℝ) 
    (h1 : initial = 9.0) 
    (h2 : given = 3.0) 
    (h3 : left = initial - given) : 
  left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_marys_marbles_l1576_157644


namespace NUMINAMATH_CALUDE_red_square_density_l1576_157640

/-- A standard rectangle is a rectangle on the coordinate plane with vertices at integer points and edges parallel to the coordinate axes. -/
def StandardRectangle (w h : ℕ) : Prop := sorry

/-- A unit square is a standard rectangle with an area of 1. -/
def UnitSquare : Prop := StandardRectangle 1 1

/-- A coloring of unit squares on the coordinate plane. -/
def Coloring := ℕ → ℕ → Bool

/-- The number of red squares in a standard rectangle. -/
def RedSquares (c : Coloring) (x y w h : ℕ) : ℕ := sorry

theorem red_square_density (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a) 
  (c : Coloring) (h4 : ∀ x y, RedSquares c x y a b + RedSquares c x y b a > 0) :
  ∀ N : ℕ, ∃ x y : ℕ, RedSquares c x y N N ≥ N * N * N / (N - 1) := by sorry

end NUMINAMATH_CALUDE_red_square_density_l1576_157640


namespace NUMINAMATH_CALUDE_product_abcd_l1576_157667

theorem product_abcd (a b c d : ℚ) : 
  3*a + 2*b + 4*c + 6*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  2*c - 2 = d →
  a * b * c * d = -58735360 / 81450625 :=
by sorry

end NUMINAMATH_CALUDE_product_abcd_l1576_157667


namespace NUMINAMATH_CALUDE_basketball_probability_l1576_157678

theorem basketball_probability (p : ℝ) (n : ℕ) (h1 : p = 1/3) (h2 : n = 3) :
  (1 - p)^n + n * p * (1 - p)^(n-1) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l1576_157678


namespace NUMINAMATH_CALUDE_valid_colorings_l1576_157673

-- Define a color type
inductive Color
| A
| B
| C

-- Define a coloring function type
def Coloring := ℕ → Color

-- Define the condition for a valid coloring
def ValidColoring (f : Coloring) : Prop :=
  ∀ a b c : ℕ, 2000 * (a + b) = c →
    (f a = f b ∧ f b = f c) ∨
    (f a ≠ f b ∧ f b ≠ f c ∧ f a ≠ f c)

-- Define the two valid colorings
def AllSameColor : Coloring :=
  λ _ => Color.A

def ModuloThreeColoring : Coloring :=
  λ n => match n % 3 with
    | 1 => Color.A
    | 2 => Color.B
    | 0 => Color.C
    | _ => Color.A  -- This case is unreachable, but needed for exhaustiveness

-- State the theorem
theorem valid_colorings (f : Coloring) :
  ValidColoring f ↔ (f = AllSameColor ∨ f = ModuloThreeColoring) :=
sorry

end NUMINAMATH_CALUDE_valid_colorings_l1576_157673


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1576_157674

theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 80 →
  num_boys * 13 = num_girls * 5 →
  num_girls > num_boys →
  num_girls - num_boys = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1576_157674


namespace NUMINAMATH_CALUDE_interview_scores_properties_l1576_157626

def scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

def sixtieth_percentile (l : List ℝ) : ℝ := sorry

def average (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def remove_extremes (l : List ℝ) : List ℝ := sorry

theorem interview_scores_properties :
  let s := scores
  let s_without_extremes := remove_extremes s
  (sixtieth_percentile s = 91) ∧
  (average s_without_extremes > average s) ∧
  (variance s_without_extremes < variance s) ∧
  (¬ (9 / 45 = 1 / 10)) ∧
  (¬ (average s > median s)) := by sorry

end NUMINAMATH_CALUDE_interview_scores_properties_l1576_157626


namespace NUMINAMATH_CALUDE_austin_weeks_to_buy_bicycle_l1576_157633

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Theorem: Austin needs 6 weeks to buy the bicycle -/
theorem austin_weeks_to_buy_bicycle :
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_austin_weeks_to_buy_bicycle_l1576_157633


namespace NUMINAMATH_CALUDE_circle_equation_l1576_157672

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, 
  (circle1 x y ∧ circle2 x y) → 
  (∃ a b : ℝ, centerLine a b ∧ 
    ((x - a)^2 + (y - b)^2 = (x + 1)^2 + (y - 1)^2)) → 
  requiredCircle x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1576_157672


namespace NUMINAMATH_CALUDE_circle_circumference_from_chord_l1576_157681

/-- Given a circular path with 8 evenly spaced trees, where the direct distance
    between two trees separated by 3 intervals is 100 feet, the total
    circumference of the circle is 175 feet. -/
theorem circle_circumference_from_chord (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let interval := d / 4
  let circumference := interval * 7
  circumference = 175 := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_from_chord_l1576_157681


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l1576_157696

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def squareArea (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given width -/
def canFitSideBySide (r1 r2 : Rectangle) (width : ℕ) : Prop :=
  r1.width + r2.width ≤ width

/-- Checks if two rectangles can fit one above the other within a given height -/
def canFitStackedVertically (r1 r2 : Rectangle) (height : ℕ) : Prop :=
  r1.height + r2.height ≤ height

/-- The main theorem stating the smallest possible area of the square -/
theorem smallest_square_area_for_rectangles : 
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  let minSideLength : ℕ := max (r1.width + r2.width) (r1.height + r2.height)
  ∃ (side : ℕ), 
    side ≥ minSideLength ∧
    canFitSideBySide r1 r2 side ∧
    canFitStackedVertically r1 r2 side ∧
    squareArea side = 81 ∧
    ∀ (s : ℕ), s < side → ¬(canFitSideBySide r1 r2 s ∧ canFitStackedVertically r1 r2 s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l1576_157696


namespace NUMINAMATH_CALUDE_perfect_squares_divisibility_l1576_157635

theorem perfect_squares_divisibility (n : ℤ) 
  (h1 : ∃ k : ℤ, 2 * n + 1 = k ^ 2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m ^ 2) :
  40 ∣ n := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_divisibility_l1576_157635


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1576_157618

theorem quadratic_factorization :
  ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1576_157618


namespace NUMINAMATH_CALUDE_only_324_and_648_have_property_l1576_157699

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasProperty (x : ℕ) : Prop :=
  x = 36 * sumOfDigits x

-- State the theorem
theorem only_324_and_648_have_property :
  ∀ x : ℕ, hasProperty x ↔ x = 324 ∨ x = 648 :=
sorry

end NUMINAMATH_CALUDE_only_324_and_648_have_property_l1576_157699


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l1576_157679

theorem bicycle_price_problem (cost_price_A : ℝ) : 
  let selling_price_B := cost_price_A * 1.25
  let selling_price_C := selling_price_B * 1.5
  selling_price_C = 225 → cost_price_A = 120 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l1576_157679


namespace NUMINAMATH_CALUDE_circle_chord_intersection_theorem_l1576_157637

noncomputable def circle_chord_intersection_problem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) : Prop :=
  let midpoint_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let midpoint_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  radius = 25 ∧
  chord_AB_length = 30 ∧
  chord_CD_length = 14 ∧
  midpoint_distance = 12 ∧
  (∀ X : ℝ × ℝ, (X.1 - O.1)^2 + (X.2 - O.2)^2 = radius^2 → 
    ((X = A ∨ X = B ∨ X = C ∨ X = D) ∨ 
     ((X.1 - A.1)^2 + (X.2 - A.2)^2) * ((X.1 - B.1)^2 + (X.2 - B.2)^2) > chord_AB_length^2 ∧
     ((X.1 - C.1)^2 + (X.2 - C.2)^2) * ((X.1 - D.1)^2 + (X.2 - D.2)^2) > chord_CD_length^2)) ∧
  (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 
    (P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2) ∧
  (P.1 - C.1) * (D.1 - C.1) + (P.2 - C.2) * (D.2 - C.2) = 
    (P.1 - D.1) * (C.1 - D.1) + (P.2 - D.2) * (C.2 - D.2) ∧
  (midpoint_AB.1 - midpoint_CD.1)^2 + (midpoint_AB.2 - midpoint_CD.2)^2 = midpoint_distance^2 →
  (P.1 - O.1)^2 + (P.2 - O.2)^2 = 4050 / 7

theorem circle_chord_intersection_theorem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) :
  circle_chord_intersection_problem O A B C D P radius chord_AB_length chord_CD_length midpoint_distance :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_theorem_l1576_157637


namespace NUMINAMATH_CALUDE_marble_problem_l1576_157663

theorem marble_problem (a : ℚ) : 
  (∃ (brian caden daryl : ℚ),
    brian = 3 * a ∧
    caden = 4 * brian ∧
    daryl = 2 * caden ∧
    (caden - 10) + (daryl + 10) = 190) →
  a = 95 / 18 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l1576_157663


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l1576_157657

/-- Given a tank that can be filled by either 18 buckets of 60 liters each or 216 buckets of unknown capacity, 
    prove that the capacity of each bucket in the second case is 5 liters. -/
theorem bucket_capacity_problem (tank_capacity : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
  (bucket_capacity_1 : ℝ) (bucket_capacity_2 : ℝ) 
  (h1 : tank_capacity = bucket_count_1 * bucket_capacity_1)
  (h2 : tank_capacity = bucket_count_2 * bucket_capacity_2)
  (h3 : bucket_count_1 = 18)
  (h4 : bucket_capacity_1 = 60)
  (h5 : bucket_count_2 = 216) :
  bucket_capacity_2 = 5 := by
  sorry

#check bucket_capacity_problem

end NUMINAMATH_CALUDE_bucket_capacity_problem_l1576_157657


namespace NUMINAMATH_CALUDE_polar_to_parabola_l1576_157670

/-- The polar equation r = 1 / (1 - sin θ) represents a parabola -/
theorem polar_to_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 1 + 2*y := by
  sorry

end NUMINAMATH_CALUDE_polar_to_parabola_l1576_157670


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1576_157666

theorem fraction_equivalence (x y : ℝ) (h1 : y ≠ 0) (h2 : x + 2*y ≠ 0) :
  (x + y) / (x + 2*y) = y / (2*y) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1576_157666


namespace NUMINAMATH_CALUDE_vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l1576_157698

/-- The cost of vanilla syrup in a coffee order -/
theorem vanilla_syrup_cost : ℝ :=
  let drip_coffee_cost : ℝ := 2.25
  let drip_coffee_quantity : ℕ := 2
  let espresso_cost : ℝ := 3.50
  let espresso_quantity : ℕ := 1
  let latte_cost : ℝ := 4.00
  let latte_quantity : ℕ := 2
  let cold_brew_cost : ℝ := 2.50
  let cold_brew_quantity : ℕ := 2
  let cappuccino_cost : ℝ := 3.50
  let cappuccino_quantity : ℕ := 1
  let total_order_cost : ℝ := 25.00

  have h1 : ℝ := drip_coffee_cost * drip_coffee_quantity +
                 espresso_cost * espresso_quantity +
                 latte_cost * latte_quantity +
                 cold_brew_cost * cold_brew_quantity +
                 cappuccino_cost * cappuccino_quantity

  have h2 : ℝ := total_order_cost - h1

  h2

theorem vanilla_syrup_cost_is_correct : vanilla_syrup_cost = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l1576_157698


namespace NUMINAMATH_CALUDE_parallelogram_area_is_41_l1576_157687

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

/-- Given vectors and their components -/
def v : ℝ × ℝ := (8, -5)
def w : ℝ × ℝ := (13, -3)

/-- Theorem: The area of the parallelogram formed by v and w is 41 -/
theorem parallelogram_area_is_41 : parallelogramArea v w = 41 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_41_l1576_157687


namespace NUMINAMATH_CALUDE_odd_function_value_l1576_157617

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = -x^3 + (a-2)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  -x^3 + (a-2)*x^2 + x

theorem odd_function_value (a : ℝ) :
  IsOdd (f a) → f a a = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l1576_157617


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1576_157647

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 + 49) = Real.sqrt 130 + Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1576_157647


namespace NUMINAMATH_CALUDE_fraction_sum_l1576_157619

theorem fraction_sum (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1576_157619


namespace NUMINAMATH_CALUDE_trains_return_to_start_l1576_157694

/-- Represents a metro line with a specific travel time -/
structure MetroLine where
  travelTime : Nat

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Calculates the period after which a train returns to its starting position -/
def returnPeriod (line : MetroLine) : Nat :=
  2 * line.travelTime

/-- The metro system of city N -/
def cityNMetro : MetroSystem :=
  { redLine := { travelTime := 7 },
    blueLine := { travelTime := 8 },
    greenLine := { travelTime := 9 } }

/-- Theorem stating that after 2016 minutes, all trains return to their initial positions -/
theorem trains_return_to_start (metro : MetroSystem := cityNMetro) :
  (2016 % returnPeriod metro.redLine = 0) ∧
  (2016 % returnPeriod metro.blueLine = 0) ∧
  (2016 % returnPeriod metro.greenLine = 0) :=
by sorry

end NUMINAMATH_CALUDE_trains_return_to_start_l1576_157694


namespace NUMINAMATH_CALUDE_colin_speed_l1576_157656

/-- Proves that Colin's speed is 4 mph given the relationships between speeds of Bruce, Tony, Brandon, and Colin -/
theorem colin_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = tony_speed / 3 →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_l1576_157656


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1576_157642

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_avg_salary : ℚ)
  (other_avg_salary : ℚ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : technician_avg_salary = 12000)
  (h4 : other_avg_salary = 6000) :
  (technicians * technician_avg_salary + (total_workers - technicians) * other_avg_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1576_157642


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l1576_157693

def star_list : List Nat := List.range 50 |>.map (· + 1)

def emilio_transform (n : Nat) : Nat :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "2"
  s'.toNat!

def emilio_list : List Nat := star_list.map emilio_transform

theorem star_emilio_sum_difference : 
  star_list.sum - emilio_list.sum = 210 := by
  sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l1576_157693


namespace NUMINAMATH_CALUDE_solution_of_functional_equation_l1576_157649

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, x + f x = f (f x)

/-- The theorem stating that the only solution to f(f(x)) = 0 is x = 0 -/
theorem solution_of_functional_equation (f : ℝ → ℝ) (h : FunctionalEquation f) :
  {x : ℝ | f (f x) = 0} = {0} := by
  sorry

end NUMINAMATH_CALUDE_solution_of_functional_equation_l1576_157649


namespace NUMINAMATH_CALUDE_coronavirus_infections_l1576_157634

theorem coronavirus_infections (initial_rate : ℕ) (increase_factor : ℕ) (days : ℕ) : 
  initial_rate = 300 → increase_factor = 4 → days = 14 →
  (initial_rate + initial_rate * increase_factor) * days = 21000 := by
sorry

end NUMINAMATH_CALUDE_coronavirus_infections_l1576_157634


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1576_157611

/-- Proves that given a simple interest of 4052.25, an annual interest rate of 9%,
    and a time period of 5 years, the principal sum is 9005. -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) 
    (h1 : interest = 4052.25)
    (h2 : rate = 9)
    (h3 : time = 5)
    (h4 : principal = interest / (rate * time / 100)) :
  principal = 9005 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1576_157611


namespace NUMINAMATH_CALUDE_next_perfect_cube_l1576_157655

/-- Given a perfect cube x, the next larger perfect cube is x + 3(∛x)² + 3∛x + 1 -/
theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧ y = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_cube_l1576_157655


namespace NUMINAMATH_CALUDE_notebooks_given_to_mike_l1576_157601

theorem notebooks_given_to_mike (jack_original : ℕ) (gerald : ℕ) (to_paula : ℕ) (jack_final : ℕ) : 
  jack_original = gerald + 13 →
  gerald = 8 →
  to_paula = 5 →
  jack_final = 10 →
  jack_original - to_paula - jack_final = 6 := by
sorry

end NUMINAMATH_CALUDE_notebooks_given_to_mike_l1576_157601


namespace NUMINAMATH_CALUDE_ellipse_and_line_problem_l1576_157631

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ

def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + 1}

theorem ellipse_and_line_problem (C : Ellipse) (L : ℝ → Set (ℝ × ℝ)) :
  C.c = 4 * Real.sqrt 3 →
  C.e = Real.sqrt 3 / 2 →
  (∀ x y, (x, y) ∈ {p : ℝ × ℝ | x^2 / C.a^2 + y^2 / C.b^2 = 1} ↔ (x, y) ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∃ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k ∧ A ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1} ∧ B ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∀ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k → A.1 = -2 * B.1) →
  (k = Real.sqrt 15 / 10 ∨ k = -Real.sqrt 15 / 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_problem_l1576_157631


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1576_157625

theorem order_of_logarithmic_expressions :
  let a := 2 * Real.log 0.99
  let b := Real.log 0.98
  let c := Real.sqrt 0.96 - 1
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1576_157625


namespace NUMINAMATH_CALUDE_probability_of_selecting_boy_l1576_157660

/-- Given a class with 60 students where 24 are girls, the probability of selecting a boy is 0.6 -/
theorem probability_of_selecting_boy (total_students : ℕ) (num_girls : ℕ) 
  (h1 : total_students = 60) 
  (h2 : num_girls = 24) : 
  (total_students - num_girls : ℚ) / total_students = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_boy_l1576_157660


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_sqrt_5_12_l1576_157632

theorem rationalize_and_simplify_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_sqrt_5_12_l1576_157632


namespace NUMINAMATH_CALUDE_sum_of_squares_and_minimum_l1576_157658

/-- Given an equation and Vieta's formulas, prove the sum of squares and its minimum value -/
theorem sum_of_squares_and_minimum (m : ℝ) (x₁ x₂ : ℝ) 
  (eq : x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂)
  (vieta1 : x₁ + x₂ = -(m + 1))
  (vieta2 : x₁ * x₂ = 2*m - 2)
  (D_nonneg : (m + 3)^2 ≥ 0) :
  (x₁^2 + x₂^2 = (m - 1)^2 + 4) ∧ 
  (∀ m', (m' - 1)^2 + 4 ≥ 4) ∧
  (∃ m₀, (m₀ - 1)^2 + 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_minimum_l1576_157658


namespace NUMINAMATH_CALUDE_special_function_at_65_l1576_157664

/-- A function satisfying f(xy) = xf(y) for all real x and y, with f(1) = 40 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 40)

/-- Theorem: If f is a special function, then f(65) = 2600 -/
theorem special_function_at_65 (f : ℝ → ℝ) (h : special_function f) : f 65 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_65_l1576_157664


namespace NUMINAMATH_CALUDE_sallys_pens_l1576_157650

theorem sallys_pens (students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) :
  students = 44 →
  pens_per_student = 7 →
  pens_taken_home = 17 →
  ∃ (initial_pens : ℕ),
    initial_pens = 342 ∧
    pens_taken_home = (initial_pens - students * pens_per_student) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_pens_l1576_157650


namespace NUMINAMATH_CALUDE_g_negative_one_eq_three_l1576_157638

/-- A polynomial function of degree 9 -/
noncomputable def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

/-- Theorem: If g(1) = -1, then g(-1) = 3 -/
theorem g_negative_one_eq_three {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_one_eq_three_l1576_157638


namespace NUMINAMATH_CALUDE_smallest_prime_between_squares_l1576_157610

theorem smallest_prime_between_squares : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (n : ℕ), p = n^2 + 6) ∧ 
  (∃ (m : ℕ), p = (m+1)^2 - 9) ∧
  (∀ (q : ℕ), q < p → 
    (Prime q → ¬(∃ (k : ℕ), q = k^2 + 6 ∧ q = (k+1)^2 - 9))) ∧
  p = 127 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_between_squares_l1576_157610


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l1576_157659

theorem probability_of_black_ball 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (prob_white : ℚ) :
  total_balls = 100 →
  red_balls = 45 →
  prob_white = 23/100 →
  (total_balls - red_balls - (total_balls * prob_white).floor) / total_balls = 32/100 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l1576_157659


namespace NUMINAMATH_CALUDE_matrix_power_10_l1576_157612

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_10 : A ^ 10 = !![512, 512; 512, 512] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_10_l1576_157612


namespace NUMINAMATH_CALUDE_crocus_to_daffodil_ratio_l1576_157639

/-- Represents the number of flower bulbs of each type planted by Jane. -/
structure FlowerBulbs where
  tulips : ℕ
  irises : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs. -/
def earnings (bulbs : FlowerBulbs) : ℚ :=
  0.5 * (bulbs.tulips + bulbs.irises + bulbs.daffodils + bulbs.crocus)

/-- Proves that given the conditions, the ratio of crocus bulbs to daffodil bulbs is 3:1. -/
theorem crocus_to_daffodil_ratio 
  (bulbs : FlowerBulbs)
  (h1 : bulbs.tulips = 20)
  (h2 : bulbs.irises = bulbs.tulips / 2)
  (h3 : bulbs.daffodils = 30)
  (h4 : earnings bulbs = 75) :
  bulbs.crocus / bulbs.daffodils = 3 := by
  sorry


end NUMINAMATH_CALUDE_crocus_to_daffodil_ratio_l1576_157639


namespace NUMINAMATH_CALUDE_basketball_points_distribution_l1576_157613

theorem basketball_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x →
  y ≤ 15 →
  (∀ i ∈ Finset.range 5, (y : ℝ) / 5 ≤ 3) →
  y = 14 :=
by sorry

end NUMINAMATH_CALUDE_basketball_points_distribution_l1576_157613


namespace NUMINAMATH_CALUDE_spade_equation_solution_l1576_157604

def spade (A B : ℝ) : ℝ := A^2 + 2*A*B + 3*B + 7

theorem spade_equation_solution :
  ∃ A : ℝ, spade A 5 = 97 ∧ (A = 5 ∨ A = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l1576_157604


namespace NUMINAMATH_CALUDE_green_sweets_count_l1576_157600

/-- Given the number of blue and yellow sweets, and the total number of sweets,
    calculate the number of green sweets. -/
theorem green_sweets_count 
  (blue_sweets : ℕ) 
  (yellow_sweets : ℕ) 
  (total_sweets : ℕ) 
  (h1 : blue_sweets = 310) 
  (h2 : yellow_sweets = 502) 
  (h3 : total_sweets = 1024) : 
  total_sweets - (blue_sweets + yellow_sweets) = 212 := by
sorry

#eval 1024 - (310 + 502)  -- This should output 212

end NUMINAMATH_CALUDE_green_sweets_count_l1576_157600


namespace NUMINAMATH_CALUDE_wage_decrease_increase_l1576_157691

theorem wage_decrease_increase (initial_wage : ℝ) :
  let decreased_wage := initial_wage * (1 - 0.5)
  let final_wage := decreased_wage * (1 + 0.5)
  final_wage = initial_wage * 0.75 ∧ (initial_wage - final_wage) / initial_wage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_increase_l1576_157691


namespace NUMINAMATH_CALUDE_expressions_not_always_equal_l1576_157623

theorem expressions_not_always_equal :
  ∃ (a b c : ℝ), a + b + c = 0 ∧ a + b * c ≠ (a + b) * (a + c) := by
  sorry

end NUMINAMATH_CALUDE_expressions_not_always_equal_l1576_157623


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l1576_157603

/-- The general equation of a line passing through two given points. -/
theorem line_equation_through_two_points :
  ∀ (x y : ℝ), 
  (∃ (t : ℝ), x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 1 * t) ↔ 
  x - 2*y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l1576_157603


namespace NUMINAMATH_CALUDE_joshua_justin_ratio_l1576_157690

def total_amount : ℝ := 40
def joshua_share : ℝ := 30

theorem joshua_justin_ratio :
  ∃ (k : ℝ), k > 0 ∧ joshua_share = k * (total_amount - joshua_share) →
  joshua_share / (total_amount - joshua_share) = 3 := by
  sorry

end NUMINAMATH_CALUDE_joshua_justin_ratio_l1576_157690


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l1576_157627

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_intersection_equals_set : (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l1576_157627


namespace NUMINAMATH_CALUDE_carrot_price_l1576_157683

/-- Calculates the price of a carrot given the number of tomatoes and carrots,
    the price of a tomato, and the total revenue from selling all produce. -/
theorem carrot_price
  (num_tomatoes : ℕ)
  (num_carrots : ℕ)
  (tomato_price : ℚ)
  (total_revenue : ℚ)
  (h1 : num_tomatoes = 200)
  (h2 : num_carrots = 350)
  (h3 : tomato_price = 1)
  (h4 : total_revenue = 725) :
  (total_revenue - num_tomatoes * tomato_price) / num_carrots = 3/2 := by
  sorry

#eval (725 : ℚ) - 200 * 1
#eval ((725 : ℚ) - 200 * 1) / 350

end NUMINAMATH_CALUDE_carrot_price_l1576_157683


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l1576_157671

/-- Given two points A and B, if the line AB is parallel to the x-axis, then m = -1 --/
theorem line_parallel_to_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (m + 1, -2)
  let B : ℝ × ℝ := (3, m - 1)
  (A.2 = B.2) → m = -1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l1576_157671


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l1576_157616

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2)*(x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l1576_157616


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_parabola_equation_l1576_157624

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Check if two line segments are perpendicular -/
def perpendicular (a b c d : Point) : Prop :=
  (b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y) = 0

/-- Theorem: If a parabola y² = 2px intersects the line x = 2 at points D and E,
    and OD ⊥ OE where O is the origin, then p = 1 -/
theorem parabola_intersection_theorem (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧                        -- D and E are on the line x = 2
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧  -- D and E are on the parabola
  perpendicular origin D origin E            -- OD ⊥ OE
  → C.p = 1 := by sorry

/-- Corollary: Under the conditions of the theorem, the parabola's equation is y² = 2x -/
theorem parabola_equation (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧
  perpendicular origin D origin E
  → ∀ x y : ℝ, y^2 = 2 * x ↔ y^2 = 2 * C.p * x := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_parabola_equation_l1576_157624


namespace NUMINAMATH_CALUDE_smallest_X_value_l1576_157651

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def T : ℕ := 1110

/-- X is defined as T divided by 15 -/
def X : ℕ := T / 15

theorem smallest_X_value :
  (onlyZerosAndOnes T) ∧ 
  (T % 15 = 0) ∧
  (∀ n : ℕ, n < T → ¬(onlyZerosAndOnes n ∧ n % 15 = 0)) →
  X = 74 := by sorry

end NUMINAMATH_CALUDE_smallest_X_value_l1576_157651


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l1576_157620

/-- The cost of a first-class postage stamp in pence -/
def first_class_cost : ℕ := 85

/-- The cost of a second-class postage stamp in pence -/
def second_class_cost : ℕ := 66

/-- The number of pence in a pound -/
def pence_per_pound : ℕ := 100

/-- The proposition that (r, s) is a valid solution to the stamp problem -/
def is_valid_solution (r s : ℕ) : Prop :=
  r ≥ 1 ∧ s ≥ 1 ∧ ∃ t : ℕ, t > 0 ∧ first_class_cost * r + second_class_cost * s = pence_per_pound * t

/-- The proposition that (r, s) is the optimal solution to the stamp problem -/
def is_optimal_solution (r s : ℕ) : Prop :=
  is_valid_solution r s ∧ ∀ r' s' : ℕ, is_valid_solution r' s' → r + s ≤ r' + s'

/-- The theorem stating the optimal solution to the stamp problem -/
theorem stamp_problem_solution :
  is_optimal_solution 2 5 ∧ 2 + 5 = 7 := by sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l1576_157620


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1576_157686

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
  (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
  (total_distance / (first_part_distance / first_part_speed + 
  (total_distance - first_part_distance) / second_part_speed)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1576_157686


namespace NUMINAMATH_CALUDE_constant_phi_forms_cone_l1576_157643

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  ∃ (cone : Set SphericalPoint), ConstantPhiSet d = cone :=
sorry

end NUMINAMATH_CALUDE_constant_phi_forms_cone_l1576_157643


namespace NUMINAMATH_CALUDE_friend_game_l1576_157654

theorem friend_game (a b c d : ℕ) : 
  3^a * 7^b = 3 * 7 ∧ 3^c * 7^d = 3 * 7 → (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by sorry

end NUMINAMATH_CALUDE_friend_game_l1576_157654


namespace NUMINAMATH_CALUDE_value_of_a_l1576_157653

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = 2 * c - 1)
  (eq2 : b + c = 7)
  (eq3 : c = 4) : 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1576_157653


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_14_vertices_l1576_157636

/-- A prism is a polyhedron with two congruent and parallel bases -/
structure Prism where
  base_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism is three times the number of edges in its base -/
axiom prism_edge_count (p : Prism) : p.total_edges = 3 * p.base_edges

/-- The number of vertices in a prism is twice the number of edges in its base -/
def prism_vertex_count (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem: A prism with 21 edges has 14 vertices -/
theorem prism_with_21_edges_has_14_vertices (p : Prism) (h : p.total_edges = 21) : 
  prism_vertex_count p = 14 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_14_vertices_l1576_157636
