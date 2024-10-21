import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_mixture_l492_49273

/-- The number of cans of blue paint needed for a mixture -/
def blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : ℕ :=
  Int.toNat (Int.ceil ((blue_ratio : ℚ) / (blue_ratio + green_ratio : ℚ) * total_cans))

/-- Theorem: Given a mixture with a ratio of blue to green paint of 4:3 and a total of 40 cans,
    the number of cans of blue paint needed is 23. -/
theorem blue_paint_mixture : blue_paint_cans 40 4 3 = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_mixture_l492_49273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_182_l492_49271

def sum_of_integers_satisfying_inequality : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (3/2 : ℚ) * (n + 1 : ℚ) - 3 > 15/2 ∧ n + 1 ≤ 20 
             then n + 1 + sum_of_integers_satisfying_inequality n 
             else sum_of_integers_satisfying_inequality n

theorem sum_equals_182 : sum_of_integers_satisfying_inequality 20 = 182 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_182_l492_49271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_evaluation_l492_49215

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^3 - 3 * Real.sqrt x

-- State the theorem
theorem g_evaluation : g 3 * g 1 - g 9 = -774 + 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_evaluation_l492_49215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_properties_l492_49297

/-- Represents a cylindrical water tank -/
structure WaterTank where
  r : ℝ  -- radius
  h : ℝ  -- height

/-- The volume of the water tank as a function of radius -/
noncomputable def volume (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- The construction cost of the water tank -/
noncomputable def constructionCost (tank : WaterTank) : ℝ :=
  200 * Real.pi * tank.r * tank.h + 160 * Real.pi * tank.r^2

theorem water_tank_properties :
  ∃ (tank : WaterTank),
    constructionCost tank = 12000 * Real.pi ∧
    tank.r > 0 ∧
    tank.h > 0 ∧
    tank.r < 5 * Real.sqrt 3 ∧
    volume tank.r = (Real.pi / 5) * (300 * tank.r - 4 * tank.r^3) ∧
    (∀ r : ℝ, r > 0 ∧ r < 5 * Real.sqrt 3 → volume r ≤ volume tank.r) ∧
    tank.r = 5 ∧
    tank.h = 8 ∧
    volume tank.r = 200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_properties_l492_49297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_DNC_value_l492_49283

/-- A rhombus with specific properties -/
structure SpecialRhombus where
  /-- The side length of the rhombus -/
  side : ℝ
  /-- The angle at vertex A is 60° -/
  angle_A : ℝ
  /-- Point N divides side AB in the ratio 2:1 -/
  N_ratio : ℝ
  /-- Assumptions about the rhombus -/
  h_angle : angle_A = 60
  h_ratio : N_ratio = 2/1

/-- The tangent of angle DNC in the special rhombus -/
noncomputable def tan_DNC (r : SpecialRhombus) : ℝ :=
  Real.sqrt (243 / 289)

/-- The main theorem stating that the tangent of angle DNC is √(243/289) -/
theorem tan_DNC_value (r : SpecialRhombus) : 
  tan_DNC r = Real.sqrt (243 / 289) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_DNC_value_l492_49283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac2_gt_bc2_sufficient_not_necessary_l492_49235

theorem ac2_gt_bc2_sufficient_not_necessary (a b : ℝ) :
  (∀ c : ℝ, c^2 * a > c^2 * b → a > b) ∧ 
  ¬(∀ c : ℝ, a > b → c^2 * a > c^2 * b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac2_gt_bc2_sufficient_not_necessary_l492_49235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_value_1_l492_49254

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 5) / (2*x - 4)

-- State the theorem
theorem f_has_minimum_value_1 :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), x ≥ 5/2 → f x ≥ m :=
by
  -- We'll use 1 as our minimum value
  use 1
  constructor
  -- Prove that m = 1
  · rfl
  -- Prove that for all x ≥ 5/2, f x ≥ 1
  · intro x hx
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_value_1_l492_49254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_a_range_l492_49263

/-- The function f(x) for a given a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (2*a + 1) * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1/x + 2*a*x - (2*a + 1)

theorem local_min_implies_a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, f_deriv a x = 0 → x = 1) : a > 1/2 := by
  sorry

#check local_min_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_a_range_l492_49263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_draws_l492_49253

/-- Represents a football team in the tournament -/
structure Team where
  score : ℕ

/-- Represents the tournament results -/
structure TournamentResult where
  teams : Finset Team
  num_draws : ℕ

/-- The rules and conditions of the tournament -/
def ValidTournament (result : TournamentResult) : Prop :=
  -- Four teams in the tournament
  result.teams.card = 4 ∧
  -- Total number of matches
  (result.teams.card.choose 2 : ℕ) = 6 ∧
  -- No team won all their matches
  ∀ t ∈ result.teams, t.score ≤ 7 ∧
  -- All teams have different scores
  ∀ t1 t2, t1 ∈ result.teams → t2 ∈ result.teams → t1 ≠ t2 → t1.score ≠ t2.score ∧
  -- Total score is consistent with the number of draws
  (result.teams.sum (fun t => t.score) : ℕ) = 18 - result.num_draws

/-- The main theorem -/
theorem min_draws (result : TournamentResult) :
  ValidTournament result → result.num_draws ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_draws_l492_49253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_17_equals_13_l492_49221

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (4 * sequenceA n + 3) / 4

theorem sequence_17_equals_13 : sequenceA 16 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_17_equals_13_l492_49221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l492_49257

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l492_49257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l492_49219

theorem triangle_side_length (a b c : ℝ) : 
  a^2 - 5*a + 2 = 0 →
  b^2 - 5*b + 2 = 0 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos (π/3)) →
  c = Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l492_49219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_5_l492_49298

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def sequence_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem min_sum_at_5 (a₁ d : ℝ) :
  (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 10 = 9) →
  (sequence_sum a₁ d 14 - sequence_sum a₁ d 3 = 77) →
  (∀ n : ℕ, sequence_sum a₁ d 5 ≤ sequence_sum a₁ d n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_5_l492_49298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_3x5_rectangle_l492_49225

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a path in the rectangle -/
structure RectPath where
  length : ℕ
  connects_opposite_vertices : Bool
  no_side_traversed_twice : Bool

/-- The maximum path length in a rectangle -/
def max_path_length (r : Rectangle) : ℕ :=
  2 * (r.width + r.height)

theorem max_path_length_3x5_rectangle :
  let r : Rectangle := { width := 5, height := 3 }
  let max_path : RectPath := { length := max_path_length r,
                               connects_opposite_vertices := true,
                               no_side_traversed_twice := true }
  max_path.length = 30 := by
  sorry

#check max_path_length_3x5_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_3x5_rectangle_l492_49225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l492_49214

/-- The area of a triangle with vertices at (3, -2), (12, 5), and (3, 8) is 45 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 45 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (3, -2)
  let B : ℝ × ℝ := (12, 5)
  let C : ℝ × ℝ := (3, 8)

  -- Calculate the area of the triangle
  let area := (1/2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) : ℝ)

  -- Prove that the area is equal to 45
  have h : area = 45 := by
    -- Actual computation here
    sorry

  -- Return the existence of the area and its value
  exact ⟨area, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l492_49214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_adults_attending_l492_49213

/-- The minimum number of adults attending an event where adults are seated in groups of 17,
    children in groups of 15, and there are an equal number of adults and children. -/
theorem min_adults_attending (adult_group_size children_group_size : ℕ) 
    (adult_group_size_eq : adult_group_size = 17)
    (children_group_size_eq : children_group_size = 15) : ℕ := by
  -- Define the minimum number of adults
  let min_adults : ℕ := 15
  
  -- The proof goes here
  sorry

-- Remove the #eval line as it's causing the compilation error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_adults_attending_l492_49213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_B_is_better_l492_49285

/-- Represents a chicken leg factory -/
structure Factory where
  name : String
  samples : List ℚ
  mean : ℚ
  median : ℚ
  mode : ℚ
  variance : ℚ

/-- Calculates the proportion of samples equal to a given weight -/
def proportionEqualTo (f : Factory) (weight : ℚ) : ℚ :=
  (f.samples.filter (· = weight)).length / f.samples.length

/-- Determines if one factory is better than another based on statistics -/
def isBetterFactory (f1 f2 : Factory) : Prop :=
  f1.mean = 75 ∧ 
  f2.mean = 75 ∧
  f1.median = 75 ∧ 
  f1.mode = 75 ∧
  f1.variance < f2.variance

/-- The main theorem stating that Factory B is the better choice -/
theorem factory_B_is_better (A B : Factory) 
  (hA : A.name = "A" ∧ A.mean = 75 ∧ A.median = 74.5 ∧ A.mode = 74 ∧ A.variance = 3.4)
  (hB : B.name = "B" ∧ B.mean = 75 ∧ B.median = 75 ∧ B.mode = 75 ∧ B.variance = 2)
  (hSamples : B.samples = [78, 74, 77, 73, 75, 75, 74, 74, 75, 75]) :
  isBetterFactory B A ∧ proportionEqualTo B 75 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_B_is_better_l492_49285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_values_and_extrema_l492_49296

-- Define the function f(x)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 4 * Real.log x

-- State the theorem
theorem extreme_points_imply_values_and_extrema
  (a b : ℝ)
  (h1 : ∀ x > 0, ∃ y, f a b x = y)  -- f is defined for x > 0
  (h2 : ∃ x1 x2 : ℝ, x1 = 1 ∧ x2 = 2 ∧ x1 ≠ x2 ∧
        (∀ x > 0, HasDerivAt (f a b) 0 x → x = x1 ∨ x = x2)) :
  a = 1 ∧ b = -6 ∧
  (∀ x > 0, f a b x ≤ f a b 1) ∧
  (∀ x > 0, f a b x ≥ f a b 2) ∧
  f a b 1 = -5 ∧
  f a b 2 = -8 + 4 * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_values_and_extrema_l492_49296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l492_49202

theorem y_squared_range (y : ℝ) (h : (y + 12) ^ (1/3) - (y - 12) ^ (1/3) = 4) : 
  105 < y^2 ∧ y^2 < 115 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l492_49202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andi_at_most_twice_valuable_l492_49223

/-- Represents a stamp collection -/
structure StampCollection where
  stamps : List ℚ
  deriving Repr

/-- The total value of a stamp collection -/
def totalValue (collection : StampCollection) : ℚ :=
  collection.stamps.sum

/-- The number of stamps above a given price in a collection -/
def stampsAbovePrice (collection : StampCollection) (price : ℚ) : ℕ :=
  (collection.stamps.filter (λ stamp => stamp > price)).length

/-- Andi's collection satisfies the condition relative to Bandi's -/
def satisfiesCondition (andi : StampCollection) (bandi : StampCollection) : Prop :=
  ∀ price, stampsAbovePrice andi price ≤ 2 * stampsAbovePrice bandi price

theorem andi_at_most_twice_valuable (andi bandi : StampCollection) 
  (h : satisfiesCondition andi bandi) : 
  totalValue andi ≤ 2 * totalValue bandi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andi_at_most_twice_valuable_l492_49223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_neg_two_l492_49200

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + (12 / a) * Real.log x

-- Define the derivative of f at x = 1
noncomputable def f_prime_at_1 (a : ℝ) : ℝ := 3 * a + 12 / a

-- Theorem statement
theorem min_value_implies_a_eq_neg_two (a : ℝ) (ha : a < 0) :
  (∀ b, f_prime_at_1 a ≤ f_prime_at_1 b) → f_prime_at_1 a = -12 → a = -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_eq_neg_two_l492_49200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l492_49236

noncomputable def plane (x y z : ℝ) : Prop := 2 * x - 3 * y + 4 * z = 20

noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

noncomputable def closest_point : ℝ × ℝ × ℝ := (54/29, -80/29, 83/29)

theorem closest_point_on_plane :
  plane closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  ∀ x y z, plane x y z →
    distance x y z 0 1 (-1) ≥ distance closest_point.1 closest_point.2.1 closest_point.2.2 0 1 (-1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l492_49236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_ten_l492_49265

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 16x -/
def Parabola := {p : Point | p.y^2 = 16 * p.x}

/-- The focus of the parabola y² = 16x -/
def focus : Point := { x := 4, y := 0 }

/-- A point on the parabola with x-coordinate 6 -/
def M : Point := { x := 6, y := 8 }

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem distance_to_focus_is_ten :
  M ∈ Parabola → distance M focus = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_ten_l492_49265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_range_l492_49244

/-- A function f(x) = -1/2 * x^2 + b * ln(x+2) that is decreasing on (-1, +∞) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + b * Real.log (x + 2)

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := -x + b / (x + 2)

/-- The theorem stating that if f is decreasing on (-1, +∞), then b ∈ (-∞, -1] -/
theorem decreasing_f_implies_b_range (b : ℝ) :
  (∀ x > -1, f_deriv b x ≤ 0) → b ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_b_range_l492_49244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_2_length_l492_49258

/-- Represents a rectangular figure with multiple horizontal segments -/
structure RectangularFigure where
  vertical_length : ℝ
  horizontal_segments : List ℝ

/-- Creates Figure 1 with the given dimensions -/
def create_figure_1 : RectangularFigure :=
  { vertical_length := 10,
    horizontal_segments := [3, 4, 2] }

/-- Removes three sides from the original figure to create Figure 2 -/
def create_figure_2 (fig : RectangularFigure) : RectangularFigure :=
  { vertical_length := fig.vertical_length,
    horizontal_segments := match fig.horizontal_segments with
      | _ :: x :: _ => [x]
      | _ => [] }

/-- Calculates the total length of segments in a figure -/
def total_length (fig : RectangularFigure) : ℝ :=
  fig.vertical_length + (fig.horizontal_segments.foldl (·+·) 0)

/-- Theorem: The total length of segments in Figure 2 is 14 units -/
theorem figure_2_length : total_length (create_figure_2 create_figure_1) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_2_length_l492_49258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l492_49233

/-- Function f with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + Real.pi / 3)

/-- Triangle ABC -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem circumcircle_area (ω : ℝ) (ABC : Triangle) :
  ω > 0 →
  (∀ x, f ω (x + Real.pi) = f ω x) →
  f ω ABC.A = -1/2 →
  ABC.c = 3 →
  1/2 * ABC.b * ABC.c * Real.sin ABC.A = 6 * Real.sqrt 3 →
  ABC.A < Real.pi/2 ∧ ABC.B < Real.pi/2 ∧ ABC.C < Real.pi/2 →
  (Real.pi * (ABC.a / (2 * Real.sin ABC.A))^2 : ℝ) = 49 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l492_49233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l492_49249

/-- The number of possible integer side lengths for the third side of a triangle -/
def count_possible_sides (a b : ℕ) : ℕ :=
  (Finset.range (a + b) \ Finset.range (Int.natAbs (a - b) + 1)).card

/-- Theorem: The number of possible integer side lengths for the third side of a triangle
    with sides 8 and 5 is 9 -/
theorem triangle_side_count :
  count_possible_sides 8 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l492_49249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_203_l492_49226

/-- The number of pairs of positive integers (m,n) satisfying m^2 + n < 50 -/
def count_pairs : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2 < 50)
    (Finset.product (Finset.range 50) (Finset.range 50)))

/-- Theorem stating that the count of pairs is 203 -/
theorem count_pairs_eq_203 : count_pairs = 203 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_203_l492_49226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l492_49292

-- Define the function f(x) = x^3 - 1/x^3
noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

-- Theorem stating that f is an odd function and monotonically increasing on (0, +∞)
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l492_49292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l492_49275

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the chord length
def chord_length : ℝ := 6

-- Define the midpoint of the chord
def is_midpoint (x y : ℝ) : Prop := ∃ (a b c d : ℝ), 
  is_on_circle a b ∧ is_on_circle c d ∧ 
  (c - a)^2 + (d - b)^2 = chord_length^2 ∧
  x = (a + c) / 2 ∧ y = (b + d) / 2

-- Statement to prove
theorem midpoint_trajectory : 
  ∀ (x y : ℝ), is_midpoint x y → x^2 + y^2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l492_49275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_l492_49282

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point E on side AC
variable (E : ℝ × ℝ)

-- Define points D and F
variable (D F : ℝ × ℝ)

-- Assume E is on AC
axiom E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C

-- Assume DE is parallel to BC
axiom DE_parallel_BC : ∃ k : ℝ, D - E = k • (B - C)

-- Assume EF is parallel to AB
axiom EF_parallel_AB : ∃ m : ℝ, F - E = m • (A - B)

-- Assume D is on BC
axiom D_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ D = (1 - s) • B + s • C

-- Assume F is on AB
axiom F_on_AB : ∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ F = (1 - r) • A + r • B

-- Define the areas of the triangles and quadrilateral
noncomputable def S_ADE (A D E : ℝ × ℝ) : ℝ := sorry
noncomputable def S_EFC (E F C : ℝ × ℝ) : ℝ := sorry
noncomputable def S_BDEF (B D E F : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_relation : 
  S_BDEF B D E F = 2 * Real.sqrt (S_ADE A D E * S_EFC E F C) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_l492_49282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_P_k_formula_l492_49243

/-- Given a positive integer n ≥ 3, P_k is defined as the sum of all elements
    in all k-element subsets of the set P = {1, 2, ..., n} -/
def P_k (n k : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (i + 1) * (Finset.range n).card.choose k)

/-- The sum of all P_k from k=1 to n -/
def sum_P_k (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => P_k n (k + 1))

theorem sum_P_k_formula (n : ℕ) (h : n ≥ 3) :
  sum_P_k n = n * (n + 1) * 2^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_P_k_formula_l492_49243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l492_49247

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2) ^ n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2) ^ n

theorem g_difference (n : ℕ) :
  g (n + 2) - g n = (-1 + 4 * Real.sqrt 3) / 8 * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l492_49247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_properties_l492_49248

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The equation of the directrix -/
def directrix (p : Point) : Prop := p.x = -1

/-- The midpoint of chord AB -/
def midpointAB : Point := ⟨2, 1⟩

/-- Vector dot product -/
def dot (p q : Point) : ℝ := p.x * q.x + p.y * q.y

/-- Vector magnitude -/
noncomputable def mag (p : Point) : ℝ := Real.sqrt (p.x^2 + p.y^2)

theorem parabola_chord_properties :
  ∀ A B : Point,
  A ∈ Parabola → B ∈ Parabola →
  (A.x + B.x) / 2 = midpointAB.x →
  (A.y + B.y) / 2 = midpointAB.y →
  dot A B = -15/4 ∧ mag (Point.mk (B.x - A.x) (B.y - A.y)) = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_properties_l492_49248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l492_49268

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ :=
  (-h.a * eccentricity h, 0)

/-- The right vertex of a hyperbola -/
def right_vertex (h : Hyperbola) : ℝ × ℝ :=
  (h.a, 0)

/-- Predicate to check if a triangle is acute -/
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- The intersection points of the hyperbola with the line x = -c -/
noncomputable def intersection_points (h : Hyperbola) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem hyperbola_eccentricity_range (h : Hyperbola) :
  let F := left_focus h
  let E := right_vertex h
  let (A, B) := intersection_points h
  is_acute_triangle A B E →
  1 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l492_49268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt_defined_l492_49291

theorem log_sqrt_defined (x : ℝ) : 
  (∃ y : ℝ, y = (Real.log (5 - 2*x)) / Real.sqrt (2*x - 3)) ↔ 3/2 < x ∧ x < 5/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt_defined_l492_49291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_plus_t_values_l492_49206

-- Define the set A
noncomputable def A (s t : ℝ) : Set ℝ :=
  {x | (s ≤ x ∧ x ≤ s + 1/6) ∨ (t ≤ x ∧ x ≤ t + 1)}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

-- Main theorem
theorem s_plus_t_values (s t : ℝ) :
  (1 ∉ A s t) →
  (s + 1/6 < t) →
  (∀ x ∈ A s t, f x ∈ A s t) →
  (s + t = 11/2 ∨ s + t = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_plus_t_values_l492_49206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l492_49269

/-- The radius of the inscribed circle of an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let s := (2 * a + b) / 2
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  area / s = 38 / 21 :=
by
  -- Assume the given values
  have ha : a = 8 := by sorry
  have hb : b = 5 := by sorry
  
  -- Calculate s
  let s := (2 * a + b) / 2
  
  -- Calculate area
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  
  -- Prove that area / s = 38 / 21
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l492_49269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_squared_value_l492_49216

/-- An isosceles trapezoid with a circle inscribed -/
structure IsoscelesTrapezoidWithCircle where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Length of sides AD and BC -/
  x : ℝ
  /-- AB is greater than CD -/
  ab_gt_cd : ab > cd
  /-- The trapezoid is isosceles -/
  isosceles : Prop
  /-- A circle is inscribed, centered on AB and tangent to AD and BC -/
  circle_inscribed : Prop

/-- The minimum value of x^2 in the isosceles trapezoid with an inscribed circle -/
noncomputable def min_x_squared (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  (t.ab / 2) * ((t.ab - t.cd) / 2)

/-- Theorem stating that the minimum value of x^2 is 1679 for the given trapezoid -/
theorem min_x_squared_value (t : IsoscelesTrapezoidWithCircle) 
  (h1 : t.ab = 92) (h2 : t.cd = 19) : min_x_squared t = 1679 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_squared_value_l492_49216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l492_49237

theorem range_of_sum (x y : ℝ) (h : x^2 + 2*x*y - 1 = 0) :
  x + y ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l492_49237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_vector_difference_magnitude_bounds_l492_49240

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))
noncomputable def c : ℝ × ℝ := (Real.sqrt 3, -1)

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem vector_perpendicular_condition (x : ℝ) :
  perpendicular (a x) (b x) ↔ ∃ k : ℤ, x = k * π / 2 + π / 4 := by sorry

theorem vector_difference_magnitude_bounds (x : ℝ) :
  1 ≤ Real.sqrt ((a x).1 - c.1)^2 + ((a x).2 - c.2)^2 ∧
  Real.sqrt ((a x).1 - c.1)^2 + ((a x).2 - c.2)^2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_vector_difference_magnitude_bounds_l492_49240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exists_shortest_distance_unique_l492_49204

/-- The curve function f(x) = ln(2x-1) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

/-- The distance function from a point (x, f(x)) to the line 2x-y+3=0 -/
noncomputable def distance (x : ℝ) : ℝ := |2 * x - f x + 3| / Real.sqrt 5

/-- Theorem stating that there exists a point with the shortest distance -/
theorem shortest_distance_exists :
  ∃ (x : ℝ), ∀ (y : ℝ), distance x ≤ distance y := by
  sorry

/-- Theorem stating that the shortest distance is unique -/
theorem shortest_distance_unique :
  ∃! (x : ℝ), ∀ (y : ℝ), distance x ≤ distance y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_exists_shortest_distance_unique_l492_49204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49246

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

theorem problem_solution :
  ∀ (m k : ℝ),
  (∀ x y, 0 < x ∧ x < y → f m x < f m y) →
  (∃ (m : ℝ), m = 0 ∧
    ∀ x ∈ Set.Icc 1 2,
    ∃ y ∈ Set.Icc 1 2,
    f m x = g k y →
    0 ≤ k ∧ k ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l492_49201

/-- Represents the efficiency of a worker relative to Sakshi -/
structure Efficiency where
  value : ℝ
  pos : value > 0

/-- Represents the number of days taken to complete the work -/
structure Days where
  value : ℝ
  pos : value > 0

/-- The efficiency of Tanya relative to Sakshi -/
def tanya_efficiency : Efficiency := ⟨1.2, by norm_num⟩

/-- The number of days Tanya takes to complete the work -/
def tanya_days : Days := ⟨10, by norm_num⟩

/-- The number of days Sakshi takes to complete the work -/
def sakshi_days : Days := ⟨12, by norm_num⟩

theorem work_completion_time (sakshi_days : Days) (tanya_days : Days) (tanya_efficiency : Efficiency) :
  tanya_efficiency.value = 1.2 →
  tanya_days.value = 10 →
  sakshi_days.value = 12 →
  (1 : ℝ) / tanya_days.value = tanya_efficiency.value * ((1 : ℝ) / sakshi_days.value) := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l492_49201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l492_49279

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Helper function to calculate triangle area
noncomputable def triangle_area (t : Triangle) : ℝ := 
  (1 / 2) * t.a * t.b * Real.sin t.C

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a - (Real.sqrt 2 / 2) * t.c = t.b * Real.cos t.C) :
  t.B = π / 4 ∧ 
  (t.a = 4 ∧ Real.cos t.C = 7 * Real.sqrt 2 / 10 → triangle_area t = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l492_49279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formula_l492_49286

/-- Represents a quadrilateral with sides a, b, c, d, area S, semiperimeter p, and angles A and C. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  S : ℝ
  p : ℝ
  A : ℝ
  C : ℝ

/-- The semiperimeter of a quadrilateral is half the sum of its sides. -/
noncomputable def semiperimeter (q : Quadrilateral) : ℝ := (q.a + q.b + q.c + q.d) / 2

/-- Theorem stating the area formula for a quadrilateral. -/
theorem quadrilateral_area_formula (q : Quadrilateral) 
  (h_semi : q.p = semiperimeter q) :
  (16 * q.S^2 = 4 * (q.b^2 * q.c^2 + q.a^2 * q.d^2) - 8 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) - (q.b^2 + q.c^2 - q.a^2 - q.d^2)^2) ∧
  (q.S = Real.sqrt ((q.p - q.a) * (q.p - q.b) * (q.p - q.c) * (q.p - q.d) - q.a * q.b * q.c * q.d * (Real.cos ((q.A + q.C) / 2))^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formula_l492_49286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_price_calculation_l492_49232

noncomputable def item_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage)

noncomputable def average_cost_price (cp_a cp_b cp_c : ℝ) : ℝ :=
  (cp_a + cp_b + cp_c) / 3

theorem average_cost_price_calculation 
  (selling_price_a selling_price_b selling_price_c : ℝ)
  (profit_percentage_a profit_percentage_b profit_percentage_c : ℝ)
  (h1 : selling_price_a = 600)
  (h2 : selling_price_b = 800)
  (h3 : selling_price_c = 900)
  (h4 : profit_percentage_a = 0.6)
  (h5 : profit_percentage_b = 0.4)
  (h6 : profit_percentage_c = 0.5) :
  ∃ ε > 0, |average_cost_price 
    (item_cost_price selling_price_a profit_percentage_a)
    (item_cost_price selling_price_b profit_percentage_b)
    (item_cost_price selling_price_c profit_percentage_c) - 515.48| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_price_calculation_l492_49232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_centroid_projections_approx_l492_49238

/-- Definition of a triangle with side lengths 6, 8, and 10 -/
structure RightTriangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_eq : x = 6
  y_eq : y = 8
  z_eq : z = 10
  right_angle : x^2 + y^2 = z^2

/-- Definition of the centroid of a triangle -/
noncomputable def centroid (t : RightTriangle) : ℝ × ℝ := sorry

/-- Distance from centroid to its projection on a side -/
noncomputable def centroid_to_side_projection (t : RightTriangle) (side : ℝ) : ℝ := 
  (2 * t.x * t.y) / (3 * side)

/-- Sum of distances from centroid to its projections on all sides -/
noncomputable def sum_centroid_projections (t : RightTriangle) : ℝ :=
  centroid_to_side_projection t t.z + 
  centroid_to_side_projection t t.x + 
  centroid_to_side_projection t t.y

/-- Theorem: The sum of distances from the centroid to its projections is approximately 6.27 -/
theorem sum_centroid_projections_approx (t : RightTriangle) : 
  abs (sum_centroid_projections t - 6.27) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_centroid_projections_approx_l492_49238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_operations_l492_49208

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 3^x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

-- Define the theorem
theorem sets_operations :
  (A ∪ B = Set.Ioc 0 4) ∧ (A ∩ (Set.univ \ B) = Set.Ioo 0 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_operations_l492_49208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_sqrt_7_l492_49290

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem magnitude_a_plus_2b_equals_sqrt_7 :
  Real.sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_sqrt_7_l492_49290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l492_49276

/-- The rational function g(x) with parameter k -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - 5*x + 6)

/-- The numerator of g(x) -/
noncomputable def numerator (k : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + k

/-- The denominator of g(x) -/
noncomputable def denominator (x : ℝ) : ℝ := x^2 - 5*x + 6

/-- Theorem: g(x) has exactly one vertical asymptote if and only if k = -10 -/
theorem one_vertical_asymptote (k : ℝ) : 
  (∃! x, denominator x = 0 ∧ numerator k x ≠ 0) ↔ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l492_49276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_theorem_l492_49251

/-- A trapezoid with specific side lengths and inscribed circles --/
structure Trapezoid :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (circleA_radius : ℝ)
  (circleB_radius : ℝ)
  (circleC_radius : ℝ)
  (circleD_radius : ℝ)

/-- The radius of the circle that fits inside the trapezoid touching all four circles --/
noncomputable def inner_circle_radius (t : Trapezoid) : ℝ := (-88 + 56 * Real.sqrt 6) / 26

/-- Theorem stating the radius of the inner circle in the specific trapezoid --/
theorem inner_circle_radius_theorem (t : Trapezoid) 
  (h_AB : t.AB = 8)
  (h_BC : t.BC = 7)
  (h_CD : t.CD = 6)
  (h_DA : t.DA = 7)
  (h_circleA : t.circleA_radius = 4)
  (h_circleB : t.circleB_radius = 4)
  (h_circleC : t.circleC_radius = 3)
  (h_circleD : t.circleD_radius = 3) :
  inner_circle_radius t = (-88 + 56 * Real.sqrt 6) / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_theorem_l492_49251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l492_49293

theorem angle_equality (α β γ θ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (h4 : 0 < θ ∧ θ < π)
  (h5 : Real.sin α / Real.sin β = Real.sin γ / Real.sin θ)
  (h6 : Real.sin α / Real.sin β = Real.sin (α - γ) / Real.sin (β - θ))
  : α = β ∧ γ = θ := by
  sorry

#check angle_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l492_49293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_arrival_theorem_l492_49227

/-- The time interval between boat arrivals at Sovunya's house -/
noncomputable def boat_arrival_interval (losyash_speed : ℝ) (launch_interval : ℝ) (boat_speed : ℝ) : ℝ :=
  (boat_speed - losyash_speed) * launch_interval / boat_speed

theorem boat_arrival_theorem (losyash_speed : ℝ) (launch_interval : ℝ) (boat_speed : ℝ)
  (h1 : losyash_speed = 4)
  (h2 : launch_interval = 0.5)
  (h3 : boat_speed = 10) :
  boat_arrival_interval losyash_speed launch_interval boat_speed = 0.3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval boat_arrival_interval 4 0.5 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_arrival_theorem_l492_49227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_fit_without_overlap_l492_49229

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Represents a placement of a square within a larger square -/
structure Placement where
  x : ℝ
  y : ℝ
  square : Square

/-- Checks if two placements overlap -/
def overlaps (p1 p2 : Placement) : Prop :=
  ∃ (x y : ℝ),
    p1.x < x ∧ x < p1.x + p1.square.side ∧
    p1.y < y ∧ y < p1.y + p1.square.side ∧
    p2.x < x ∧ x < p2.x + p2.square.side ∧
    p2.y < y ∧ y < p2.y + p2.square.side

/-- Theorem stating that it's possible to fit all squares without overlapping -/
theorem squares_fit_without_overlap :
  ∃ (placements : ℕ → Placement),
    (∀ i : ℕ, (placements i).square.side = 1 / (i + 1 : ℝ)) ∧
    (∀ i j : ℕ, i ≠ j → ¬ overlaps (placements i) (placements j)) ∧
    (∀ i : ℕ, (placements i).x ≥ 0 ∧ (placements i).x + (placements i).square.side ≤ 3/2 ∧
              (placements i).y ≥ 0 ∧ (placements i).y + (placements i).square.side ≤ 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_fit_without_overlap_l492_49229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l492_49262

/-- Definition of the sequence satisfying the given conditions -/
def my_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 1) = 2^n

/-- Theorem stating the general term of the sequence -/
theorem sequence_general_term (a : ℕ → ℝ) (h : my_sequence a) :
  ∀ n : ℕ, n ≥ 1 → 
    a n = if n % 2 = 0 
      then 2^((n / 2) - 1) 
      else 2^((n + 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l492_49262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_norms_l492_49261

open Real

variable (a b m : ℝ × ℝ)

def is_midpoint (m a b : ℝ × ℝ) : Prop := m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem sum_of_squared_norms (h1 : is_midpoint m a b) 
                             (h2 : m = (4, 5)) 
                             (h3 : a.1 * b.1 + a.2 * b.2 = 10) : 
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_norms_l492_49261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_equals_50_over_23_l492_49281

-- Define L(x) = x - x^2/2
noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

-- Define b_n as n iterations of L applied to 25/n
noncomputable def b (n : ℕ) : ℝ := (L^[n]) (25 / n)

-- Theorem statement
theorem limit_n_b_n_equals_50_over_23 :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * b n - 50 / 23| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_equals_50_over_23_l492_49281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_discount_proof_l492_49224

/-- Proves that the discount on shoes is 40% given the shopping scenario --/
theorem shoe_discount_proof (original_shoe_price original_dress_price dress_discount total_spent : ℝ)
  (h1 : original_shoe_price = 100)
  (h2 : original_dress_price = 100)
  (h3 : dress_discount = 20)
  (h4 : total_spent = 140) :
  ∃ (shoe_discount : ℝ), shoe_discount = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_discount_proof_l492_49224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l492_49220

/-- Parabola defined by y² = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Slope of a line -/
noncomputable def lineSlope (l : Line) : ℝ := sorry

/-- Check if a point is on a parabola -/
def isOnParabola (p : ℝ × ℝ) (c : Parabola) : Prop := sorry

/-- Check if a line passes through a point -/
def passesThrough (l : Line) (p : ℝ × ℝ) : Prop := sorry

theorem parabola_properties (c : Parabola) (l : Line) :
  c.equation = (fun x y => y^2 = 2*x) →
  c.focus = (1/2, 0) →
  (∃ A B : ℝ × ℝ, isOnParabola A c ∧ isOnParabola B c ∧
    passesThrough (Line.mk A B) c.focus ∧ lineSlope (Line.mk A B) = 2 →
    distance A B = 5/2) ∧
  (∀ A : ℝ × ℝ, isOnParabola A c →
    distanceToLine A (Line.mk (-4, 0) (0, 4)) ≥ 7*Real.sqrt 2/4) ∧
  (∃ A : ℝ × ℝ, isOnParabola A c ∧
    distanceToLine A (Line.mk (-4, 0) (0, 4)) = 7*Real.sqrt 2/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l492_49220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_order_l492_49270

noncomputable section

/-- The inverse proportion function f(x) = -2/x -/
def f (x : ℝ) : ℝ := -2 / x

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (2, f 2)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (3, f 3)

/-- y₁ is the y-coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ is the y-coordinate of point B -/
def y₂ : ℝ := B.2

/-- y₃ is the y-coordinate of point C -/
def y₃ : ℝ := C.2

theorem inverse_proportion_order : y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_order_l492_49270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_ellipse_l492_49260

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (Real.cos t) ^ 2
noncomputable def y (t : ℝ) : ℝ := (Real.cos t) * (Real.sin t)

-- State the theorem
theorem points_form_ellipse :
  ∃ a b h k : ℝ, ∀ t : ℝ,
    ((x t - h)^2) / (a^2) + ((y t - k)^2) / (b^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_ellipse_l492_49260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49252

theorem problem_solution (a b c : ℝ) (m n : ℕ+) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 5 / a = b + c) (h5 : 10 / b = c + a) (h6 : 13 / c = a + b)
  (h7 : a + b + c = m / n) (h8 : Nat.Coprime m.val n.val) : m + n = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l492_49212

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem work_completion_time 
  (a_days : ℝ) 
  (b_days : ℝ) 
  (c_days : ℝ) 
  (h1 : b_days = 6) 
  (h2 : c_days = 12) 
  (h3 : work_rate a_days + work_rate b_days + work_rate c_days = work_rate (24/7)) :
  a_days = 24 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l492_49212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_frequency_l492_49280

def GradeSet : Set ℕ := {2, 3, 4, 5}

theorem grade_frequency (grades : List ℕ) :
  grades.length = 13 ∧
  (∀ g ∈ grades, g ∈ GradeSet) ∧
  (grades.sum / grades.length : ℚ).num = grades.sum →
  ∃ g ∈ GradeSet, (grades.count g) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_frequency_l492_49280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_square_functions_with_range_one_four_square_functions_finite_eq_set_l492_49234

def is_square_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = x^2

def has_range_one_four (f : ℝ → ℝ) : Prop :=
  ∀ y, f ⁻¹' {y} ≠ ∅ → y = 1 ∨ y = 4

def square_functions_with_range_one_four : Set (ℝ → ℝ) :=
  {f | is_square_function f ∧ has_range_one_four f}

-- We need to make this noncomputable because we're working with real numbers
noncomputable def square_functions_finite : Finset (ℝ → ℝ) :=
  -- We'll define this later in the proof
  sorry

theorem count_square_functions_with_range_one_four :
  Finset.card square_functions_finite = 9 :=
sorry

theorem square_functions_finite_eq_set :
  ↑square_functions_finite = square_functions_with_range_one_four :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_square_functions_with_range_one_four_square_functions_finite_eq_set_l492_49234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_group_fraction_l492_49222

theorem age_group_fraction (total_students : ℕ) 
  (under_11_fraction : ℚ) (above_13_count : ℕ) : 
  total_students = 45 → 
  under_11_fraction = 1/3 → 
  above_13_count = 12 → 
  (total_students - (under_11_fraction * ↑total_students).num - above_13_count : ℚ) / total_students = 2/5 := by
  sorry

#check age_group_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_group_fraction_l492_49222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l492_49242

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 3 0.4) 
  (hb : b = Real.log 2) 
  (hc : c = Real.log 0.7 / Real.log 2) : 
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l492_49242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_less_than_half_karlsson_will_become_plump_l492_49267

/-- A process of repeatedly cutting a triangle along an angle bisector and removing one part -/
noncomputable def triangleBisectionProcess (initialArea : ℝ) : ℕ → ℝ
| 0 => initialArea
| n + 1 => 
  let remainingArea := triangleBisectionProcess initialArea n
  let cutArea := remainingArea * (1 / 2)  -- simplified model of the cut
  remainingArea - cutArea

/-- The theorem stating that the remaining area will eventually become less than half of the original area -/
theorem eventually_less_than_half (initialArea : ℝ) (h : initialArea > 0) :
  ∃ n : ℕ, triangleBisectionProcess initialArea n < initialArea / 2 := by
  sorry

/-- The main theorem corresponding to the original problem -/
theorem karlsson_will_become_plump (initialCakeArea : ℝ) (h : initialCakeArea > 0) :
  ∃ n : ℕ, triangleBisectionProcess initialCakeArea n < initialCakeArea / 2 := by
  exact eventually_less_than_half initialCakeArea h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_less_than_half_karlsson_will_become_plump_l492_49267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l492_49295

/-- A power function passing through (2, √2/2) with f(m) = 2 has m = 1/4 -/
theorem power_function_m_value (f : ℝ → ℝ) (a m : ℝ) :
  (∀ x, f x = x ^ a) →  -- f is a power function
  f 2 = Real.sqrt 2 / 2 →        -- f passes through (2, √2/2)
  f m = 2 →             -- f(m) = 2
  m = 1 / 4 :=          -- prove that m = 1/4
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l492_49295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_equality_l492_49284

theorem positive_real_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → b = 9*a → a = (3 : ℝ) ^ (1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_equality_l492_49284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49207

-- Define the lines
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (a - 1) * x - 4 * y = 1
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (a + 1) * x + 3 * y = 2
def l₃ : ℝ → ℝ → Prop := λ x y ↦ x - 2 * y = 3

-- Define the slope of a line (as a noncomputable function)
noncomputable def slopeOf (f : ℝ → ℝ → Prop) : ℝ := sorry

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop := sorry

-- Define the slope angle (as a noncomputable function)
noncomputable def slopeAngle (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem problem_solution :
  (∀ a : ℝ, slopeAngle (l₁ a) = 135 → a = -3) ∧
  (∀ a : ℝ, parallel (l₂ a) l₃ → a = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l492_49287

theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (Real.log 16 / Real.log x = Real.log 2 / Real.log 8) → x = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l492_49287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_composite_l492_49264

theorem expression_is_composite (n : ℕ) (h : n ≥ 9) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (Nat.factorial (n + 2) + Nat.factorial (n + 1)) / Nat.factorial n + 4 = a * b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_composite_l492_49264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_from_pine_high_school_l492_49245

theorem boys_from_pine_high_school :
  let total_students : ℕ := 150
  let total_boys : ℕ := 90
  let total_girls : ℕ := 60
  let maple_students : ℕ := 70
  let pine_students : ℕ := 80
  let oak_girls : ℕ := 20
  let maple_girls : ℕ := 30
  let pine_boys : ℕ := pine_students - (total_girls - maple_girls - oak_girls)
  pine_boys = 70 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_from_pine_high_school_l492_49245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l492_49255

/-- The function g as defined in the problem -/
noncomputable def g (x y : ℝ) (θ : ℝ) : ℝ :=
  x / (x + y) + y / (y + Real.cos θ ^ 2) + Real.cos θ ^ 2 / (Real.cos θ ^ 2 + x)

/-- The theorem stating the bounds of g -/
theorem g_bounds (x y : ℝ) (θ : ℝ) (hx : x > 0) (hy : y > 0) :
  0 < g x y θ ∧ g x y θ < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l492_49255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_subset_partition_of_positive_integers_l492_49294

theorem no_three_subset_partition_of_positive_integers : 
  ¬ ∃ (A B C : Set ℕ+), 
    (A ∪ B ∪ C = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x y, (x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ C) ∨ (x ∈ C ∧ y ∈ A) →
      x^2 - x*y + y^2 ∈ (A ∪ B ∪ C) \ ({x, y} ∩ (A ∪ B ∪ C))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_subset_partition_of_positive_integers_l492_49294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_closest_to_21_percent_l492_49277

noncomputable def item_prices : List ℝ := [13.24, 7.95, 3.75, 10.99, 3.45]
def paid_amount : ℝ := 50.00

noncomputable def total_cost : ℝ := item_prices.sum
noncomputable def change : ℝ := paid_amount - total_cost
noncomputable def change_percentage : ℝ := (change / paid_amount) * 100

theorem change_closest_to_21_percent :
  ∀ (x : ℝ), x ∈ [20, 21, 22, 23, 25] →
  |change_percentage - 21| ≤ |change_percentage - x| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_closest_to_21_percent_l492_49277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l492_49256

open Real

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : sin A / a = sin B / b
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*cos C

/-- Helper function for triangle area -/
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * sin t.B

theorem triangle_properties (t : Triangle) 
  (h1 : t.b * sin t.A + t.a * cos t.B = 0)
  (h2 : t.b = 2) :
  t.B = 3*π/4 ∧ 
  (∀ (t' : Triangle), t'.b = 2 → area t' ≤ 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l492_49256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_standard_deviation_l492_49259

noncomputable def data_set : List ℝ := [2, 3, 4, 6, 10]

/-- The average of the data set -/
noncomputable def average : ℝ := 5

/-- The standard deviation of the data set -/
noncomputable def standard_deviation : ℝ := 2 * Real.sqrt 2

theorem data_set_standard_deviation : 
  let n := data_set.length
  let squared_diff_sum := (data_set.map (λ x => (x - average)^2)).sum
  Real.sqrt (squared_diff_sum / n) = standard_deviation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_standard_deviation_l492_49259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l492_49210

noncomputable def complex_sum : ℂ :=
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (23 * Real.pi * Complex.I / 60) +
  Complex.exp (35 * Real.pi * Complex.I / 60) +
  Complex.exp (47 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60)

theorem complex_sum_argument :
  Complex.arg complex_sum = 7 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l492_49210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_l492_49278

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  sorry

theorem repeating_decimal_fraction :
  let d1 := RepeatingDecimal.mk 0 72
  let d2 := RepeatingDecimal.mk 1 81
  (toRational d1) / (toRational d2) = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_l492_49278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicolai_ate_six_pounds_of_peaches_l492_49288

/-- Represents the amount of fruit eaten by each friend in ounces -/
structure FruitEaten where
  mario : ℚ
  lydia : ℚ
  nicolai : ℚ

/-- Converts pounds to ounces -/
def poundsToOunces (pounds : ℚ) : ℚ := pounds * 16

/-- Converts ounces to pounds -/
def ouncesToPounds (ounces : ℚ) : ℚ := ounces / 16

/-- Theorem stating that Nicolai ate 6 pounds of peaches -/
theorem nicolai_ate_six_pounds_of_peaches 
  (total_fruit : ℚ) 
  (fruit_eaten : FruitEaten) 
  (h1 : total_fruit = 8) 
  (h2 : fruit_eaten.mario = 8) 
  (h3 : fruit_eaten.lydia = 24) 
  (h4 : poundsToOunces total_fruit = fruit_eaten.mario + fruit_eaten.lydia + fruit_eaten.nicolai) : 
  ouncesToPounds fruit_eaten.nicolai = 6 := by
  sorry

#eval poundsToOunces 8
#eval ouncesToPounds 96

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicolai_ate_six_pounds_of_peaches_l492_49288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l492_49250

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.cos x

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) 0 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.pi / 2) 0 → f y ≥ f x) ∧
  f x = -Real.pi / 6 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l492_49250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_l492_49211

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (x - π/3)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

-- Theorem for the range of f(x) when x ∈ [0, π/2]
theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (0 : ℝ) (π/2) = Set.Icc (0 : ℝ) ((2 + sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_l492_49211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_inequality_l492_49289

/-- A convex decagon -/
structure ConvexDecagon where
  sides : Fin 10 → ℝ
  diagonals : Fin 35 → ℝ
  convex : Convex ℝ (Set.range (λ i => (i : ℝ × ℝ)))

/-- The sum of side lengths of a convex decagon -/
def sumSides (d : ConvexDecagon) : ℝ :=
  Finset.univ.sum d.sides

/-- The sum of diagonal lengths of a convex decagon -/
def sumDiagonals (d : ConvexDecagon) : ℝ :=
  Finset.univ.sum d.diagonals

/-- The number of sides in a decagon -/
def numSides : ℕ := 10

/-- The number of diagonals in a decagon -/
def numDiagonals : ℕ := 35

theorem decagon_inequality (d : ConvexDecagon) :
  (sumSides d) * numDiagonals < (sumDiagonals d) * numSides := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_inequality_l492_49289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_one_half_l492_49272

/-- Represents the financial situation of a person who borrowed money -/
structure BorrowSituation where
  monthsSinceBorrowing : ℕ
  monthlyReturn : ℕ
  remainingDebtAfterMonths : ℕ
  remainingDebt : ℕ

/-- Calculates the ratio of returned money to borrowed money -/
def returnedToBorrowedRatio (s : BorrowSituation) : ℚ :=
  let totalReturned := s.monthsSinceBorrowing * s.monthlyReturn
  let futureReturns := s.remainingDebtAfterMonths * s.monthlyReturn
  let totalBorrowed := totalReturned + futureReturns + s.remainingDebt
  ↑totalReturned / ↑totalBorrowed

/-- Theorem stating that for the given situation, the ratio is 1:2 -/
theorem ratio_is_one_half :
  let s : BorrowSituation := {
    monthsSinceBorrowing := 6,
    monthlyReturn := 10,
    remainingDebtAfterMonths := 4,
    remainingDebt := 20
  }
  returnedToBorrowedRatio s = 1 / 2 := by
  sorry

#eval returnedToBorrowedRatio {
  monthsSinceBorrowing := 6,
  monthlyReturn := 10,
  remainingDebtAfterMonths := 4,
  remainingDebt := 20
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_one_half_l492_49272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49218

/-- The function h(x) = (m^2 - 5m + 1)x^(m+1) -/
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m + 1) * x^(m+1)

/-- The function g(x) = x + √(1 - 2x) -/
noncomputable def g (x : ℝ) : ℝ := x + Real.sqrt (1 - 2*x)

theorem problem_solution :
  (∀ x, h 0 x = x) ∧  -- h is a power function when m = 0
  (∀ x, h 0 (-x) = -(h 0 x)) ∧  -- h is an odd function when m = 0
  (∀ x ∈ Set.Icc 0 (1/2), g x ∈ Set.Icc (1/2) 1) ∧  -- range of g(x) for x ∈ [0, 1/2]
  (∃ x ∈ Set.Icc 0 (1/2), g x = 1/2) ∧  -- minimum value of g(x)
  (∃ x ∈ Set.Icc 0 (1/2), g x = 1)  -- maximum value of g(x)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l492_49218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quotient_digits_l492_49205

def quotient : Nat := 665655 / 5

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem sum_of_quotient_digits :
  sum_of_digits quotient = 12 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quotient_digits_l492_49205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_reciprocal_x_plus_one_l492_49299

/-- For any real number x > 1, the sum of the infinite series ∑(1 / (x^(2^n) + x^(-2^n))) from n = 0 to infinity is equal to 1 / (x + 1). -/
theorem infinite_sum_equals_reciprocal_x_plus_one (x : ℝ) (hx : x > 1) :
  ∑' n, 1 / (x^(2^n) + (1/x)^(2^n)) = 1 / (x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_reciprocal_x_plus_one_l492_49299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisects_arc_l492_49203

/-- Two circles touching internally with a tangent to the smaller circle -/
structure TangentCircles where
  larger_circle : Set (EuclideanSpace ℝ (Fin 2))
  smaller_circle : Set (EuclideanSpace ℝ (Fin 2))
  C : EuclideanSpace ℝ (Fin 2)  -- Point of internal tangency
  P : EuclideanSpace ℝ (Fin 2)  -- Point on smaller circle where tangent is drawn
  A : EuclideanSpace ℝ (Fin 2)  -- Intersection point of tangent with larger circle
  B : EuclideanSpace ℝ (Fin 2)  -- Other intersection point of tangent with larger circle
  is_internal_tangent : smaller_circle ⊆ larger_circle ∧ C ∈ smaller_circle ∩ larger_circle
  is_tangent_point : P ∈ smaller_circle
  tangent_intersects : A ∈ larger_circle ∧ B ∈ larger_circle
  
/-- The midpoint of an arc -/
noncomputable def midpoint_of_arc (circle : Set (EuclideanSpace ℝ (Fin 2))) (A B : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

/-- The line through two points -/
def Line.throughPoints (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- The theorem stating that CP intersects arc AB at its midpoint -/
theorem tangent_bisects_arc (tc : TangentCircles) :
  ∃ Q : EuclideanSpace ℝ (Fin 2), Q ∈ tc.larger_circle ∧ 
           Q ∈ (Line.throughPoints tc.C tc.P) ∧
           Q = midpoint_of_arc tc.larger_circle tc.A tc.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisects_arc_l492_49203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l492_49274

theorem calculation_proof : 2⁻¹ + |(-1)| - (2 + Real.pi)^0 = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l492_49274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_wrap_tetrahedron_l492_49217

/-- The minimum side length of a square paper that can completely wrap a regular tetrahedron -/
noncomputable def min_square_side_length (a : ℝ) : ℝ := (Real.sqrt 2 + Real.sqrt 6) / 2 * a

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_positive : edge_length > 0

/-- A square wrapping paper -/
structure SquareWrappingPaper where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- Predicate to check if a square paper can wrap a tetrahedron -/
def can_wrap (paper : SquareWrappingPaper) (tetra : RegularTetrahedron) : Prop :=
  paper.side_length ≥ min_square_side_length tetra.edge_length

theorem min_square_wrap_tetrahedron (a : ℝ) (ha : a > 0) :
  ∀ (paper : SquareWrappingPaper),
    can_wrap paper { edge_length := a, edge_length_positive := ha } →
    paper.side_length ≥ min_square_side_length a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_wrap_tetrahedron_l492_49217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l492_49241

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_abc_theorem (t : Triangle) 
  (h1 : t.A > 0) (h2 : t.B > 0) (h3 : t.C > 0)
  (h4 : t.A + t.B + t.C = Real.pi)
  (h5 : t.a > 0) (h6 : t.b > 0) (h7 : t.c > 0)
  (h8 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h9 : t.b / Real.sin t.B = t.c / Real.sin t.C)
  (h10 : Real.sin t.C / Real.cos t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B))
  (h11 : Real.sin (t.B - t.A) + Real.cos (t.A + t.B) = 0)
  (h12 : 1/2 * t.a * t.c * Real.sin t.B = 3 + Real.sqrt 3) :
  Real.sin t.B = (Real.sqrt 2 + Real.sqrt 6) / 4 ∧ t.a = 2 * Real.sqrt 2 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l492_49241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_of_specific_rhombus_l492_49230

/-- Represents a rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ
  h_area_positive : 0 < area
  h_ratio_positive : 0 < diagonal_ratio

/-- The length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (2 * r.area * r.diagonal_ratio / (1 + r.diagonal_ratio))

/-- Theorem: For a rhombus with area 200 and diagonal ratio 4:1, the longest diagonal is 40 units -/
theorem longest_diagonal_of_specific_rhombus :
  let r : Rhombus := {
    area := 200,
    diagonal_ratio := 4,
    h_area_positive := by norm_num,
    h_ratio_positive := by norm_num
  }
  longest_diagonal r = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_of_specific_rhombus_l492_49230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_exists_l492_49228

/-- The function representing the intersection of the two curves -/
noncomputable def f (x : ℝ) : ℝ := 10 / (x^2 + 4) - (3 - x)

/-- Theorem stating that there exists a solution close to -2.8475 -/
theorem intersection_exists : ∃ x : ℝ, ‖x + 2.8475‖ < 0.0001 ∧ f x = 0 := by
  sorry

#check intersection_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_exists_l492_49228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_one_parallel_pair_l492_49209

/-- A regular convex polygon with 2m sides -/
structure RegularConvexPolygon (n : ℕ) where
  vertices : Fin (2 * n) → ℝ × ℝ
  is_regular : sorry
  is_convex : sorry

/-- A polygon with the same vertices as another polygon, but possibly in a different order -/
structure SameVerticesPolygon {n : ℕ} (P : RegularConvexPolygon n) where
  vertices : Fin (2 * n) → ℝ × ℝ
  perm : Equiv.Perm (Fin (2 * n))
  same_vertices : ∀ i, vertices i = P.vertices (perm i)

/-- Two sides of a polygon are parallel -/
def parallel_sides {n : ℕ} (P : RegularConvexPolygon n) (π : SameVerticesPolygon P) (i j : Fin (2 * n)) : Prop :=
  sorry

/-- The number of pairs of parallel sides in a polygon -/
def num_parallel_pairs {n : ℕ} (P : RegularConvexPolygon n) (π : SameVerticesPolygon P) : ℕ :=
  sorry

/-- Main theorem: For any regular convex 2m-sided polygon, there exists a polygon
    with the same vertices and exactly one pair of parallel sides -/
theorem exists_one_parallel_pair (n : ℕ) (P : RegularConvexPolygon n) :
  ∃ π : SameVerticesPolygon P, num_parallel_pairs P π = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_one_parallel_pair_l492_49209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l492_49239

theorem remainder_problem (k : ℕ) 
  (h1 : k % 6 = 5)
  (h2 : k % 7 = 3)
  (h3 : k < 42) :
  k % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l492_49239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_with_dog_l492_49231

noncomputable def bath_time : ℝ := 20
noncomputable def blow_dry_time : ℝ := bath_time / 2
noncomputable def fetch_time : ℝ := 15
noncomputable def training_time : ℝ := 10
noncomputable def trail_length : ℝ := 4
noncomputable def flat_speed : ℝ := 6
noncomputable def uphill_speed : ℝ := 4
noncomputable def downhill_speed : ℝ := 8
noncomputable def sandy_speed : ℝ := 3

noncomputable def flat_time : ℝ := (trail_length / 4) / flat_speed * 60
noncomputable def uphill_time : ℝ := (trail_length / 4) / uphill_speed * 60
noncomputable def downhill_time : ℝ := (trail_length / 4) / downhill_speed * 60
noncomputable def sandy_time : ℝ := (trail_length / 4) / sandy_speed * 60

theorem total_time_with_dog :
  bath_time + blow_dry_time + fetch_time + training_time +
  flat_time + uphill_time + downhill_time + sandy_time = 107.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_with_dog_l492_49231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circle_center_locus_l492_49266

/-- The locus of points with a constant difference of distances from two fixed points -/
def HyperbolaBranch (f₁ f₂ : ℝ × ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f₁ - dist p f₂ = k}

/-- The centers of the given circles -/
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (4, 3)

/-- The constant difference in distances -/
def K : ℝ := 1

/-- Definition of a hyperbola branch -/
def IsHyperbolaBranch (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (f₁ f₂ : ℝ × ℝ) (a : ℝ), a > 0 ∧
    s = {p : ℝ × ℝ | |dist p f₁ - dist p f₂| = 2*a ∧ dist p f₁ > dist p f₂}

theorem external_tangent_circle_center_locus :
  ∃ (h : Set (ℝ × ℝ)), h = HyperbolaBranch F₁ F₂ K ∧ IsHyperbolaBranch h :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circle_center_locus_l492_49266
