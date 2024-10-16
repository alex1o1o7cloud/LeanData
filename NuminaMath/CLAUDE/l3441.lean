import Mathlib

namespace NUMINAMATH_CALUDE_vector_collinearity_l3441_344166

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![-1, x]
  let sum : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]
  let diff : Fin 2 → ℝ := ![a 0 - b 0, a 1 - b 1]
  (sum 0 * diff 0 + sum 1 * diff 1 = 0) → (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3441_344166


namespace NUMINAMATH_CALUDE_percent_decrease_l3441_344193

theorem percent_decrease (X Y : ℝ) (h : Y = 1.2 * X) :
  X = Y * (1 - 1/6) :=
by sorry

end NUMINAMATH_CALUDE_percent_decrease_l3441_344193


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3441_344198

theorem absolute_value_equation_solutions :
  let S : Set ℝ := {x | |x + 1| * |x - 2| * |x + 3| * |x - 4| = |x - 1| * |x + 2| * |x - 3| * |x + 4|}
  S = {0, Real.sqrt 7, -Real.sqrt 7, 
       Real.sqrt ((13 + Real.sqrt 73) / 2), -Real.sqrt ((13 + Real.sqrt 73) / 2),
       Real.sqrt ((13 - Real.sqrt 73) / 2), -Real.sqrt ((13 - Real.sqrt 73) / 2)} := by
  sorry


end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3441_344198


namespace NUMINAMATH_CALUDE_unique_a_value_l3441_344199

def A (a : ℝ) : Set ℝ := {a - 2, 2 * a^2 + 5 * a, 12}

theorem unique_a_value : ∀ a : ℝ, -3 ∈ A a ↔ a = -3/2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3441_344199


namespace NUMINAMATH_CALUDE_tiling_condition_l3441_344179

/-- Represents a tile type with its dimensions -/
inductive TileType
  | square : TileType  -- 2 × 2 tile
  | rectangle : TileType  -- 3 × 1 tile

/-- Calculates the area of a tile -/
def tileArea (t : TileType) : ℕ :=
  match t with
  | TileType.square => 4
  | TileType.rectangle => 3

/-- Represents a floor tiling with square and rectangle tiles -/
structure Tiling (n : ℕ) where
  numTiles : ℕ
  complete : n * n = numTiles * (tileArea TileType.square + tileArea TileType.rectangle)

/-- Theorem: A square floor of size n × n can be tiled with an equal number of 2 × 2 and 3 × 1 tiles
    if and only if n is divisible by 7 -/
theorem tiling_condition (n : ℕ) :
  (∃ t : Tiling n, True) ↔ ∃ k : ℕ, n = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_tiling_condition_l3441_344179


namespace NUMINAMATH_CALUDE_max_profit_at_36_l3441_344167

/-- Represents the daily sales quantity of product A in kg -/
def y (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the daily profit in yuan -/
def w (x : ℝ) : ℝ := -2 * x^2 + 160 * x - 2760

/-- The cost of product A in yuan per kg -/
def cost_A : ℝ := 20

/-- The maximum allowed price of product A (180% of cost) -/
def max_price_A : ℝ := cost_A * 1.8

theorem max_profit_at_36 :
  ∀ x : ℝ, cost_A ≤ x ∧ x ≤ max_price_A →
  w x ≤ w 36 ∧ w 36 = 408 := by
  sorry

#eval w 36

end NUMINAMATH_CALUDE_max_profit_at_36_l3441_344167


namespace NUMINAMATH_CALUDE_rectangle_garden_length_l3441_344111

/-- Represents the perimeter of a rectangle. -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with perimeter 950 m and breadth 100 m has a length of 375 m. -/
theorem rectangle_garden_length :
  ∀ (length : ℝ),
  perimeter length 100 = 950 →
  length = 375 := by
sorry

end NUMINAMATH_CALUDE_rectangle_garden_length_l3441_344111


namespace NUMINAMATH_CALUDE_slope_equation_l3441_344175

theorem slope_equation (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 3) / (1 - m) = m) : m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_equation_l3441_344175


namespace NUMINAMATH_CALUDE_nancy_keeps_ten_l3441_344176

def nancy_chips : ℕ := 22
def brother_chips : ℕ := 7
def sister_chips : ℕ := 5

theorem nancy_keeps_ten : 
  nancy_chips - (brother_chips + sister_chips) = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_keeps_ten_l3441_344176


namespace NUMINAMATH_CALUDE_complex_equation_difference_l3441_344171

theorem complex_equation_difference (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  (m + 2 * i) / i = n + i → n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l3441_344171


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l3441_344192

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (x : ℝ), f x d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l3441_344192


namespace NUMINAMATH_CALUDE_cubeRoot_of_negative_eight_eq_negative_two_l3441_344130

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_eight_eq_negative_two :
  cubeRoot (-8) = -2 := by sorry

end NUMINAMATH_CALUDE_cubeRoot_of_negative_eight_eq_negative_two_l3441_344130


namespace NUMINAMATH_CALUDE_surface_area_difference_l3441_344194

/-- The difference between the sum of surface areas of smaller cubes and the surface area of a larger cube -/
theorem surface_area_difference (larger_cube_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_cube_volume : ℝ) : 
  larger_cube_volume = 343 →
  num_smaller_cubes = 343 →
  smaller_cube_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_cube_volume ^ (2/3)) - 6 * larger_cube_volume ^ (2/3) = 1764 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l3441_344194


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l3441_344177

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 - 3*x > 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Theorem for the complement of A
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x | x ≤ -3 ∨ x ≥ 0} := by sorry

-- Theorem for the range of m when A is a subset of B
theorem range_of_m_for_subset : 
  ∀ m : ℝ, A ⊆ B m → m ≥ 0 := by sorry

-- Theorem for the range of m when A and B are disjoint
theorem range_of_m_for_disjoint : 
  ∀ m : ℝ, A ∩ B m = ∅ → m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_m_for_subset_range_of_m_for_disjoint_l3441_344177


namespace NUMINAMATH_CALUDE_gcd_b_n_b_n_plus_one_is_one_l3441_344165

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem gcd_b_n_b_n_plus_one_is_one (n : ℕ) : 
  Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_n_b_n_plus_one_is_one_l3441_344165


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3441_344156

theorem cube_root_equation_solution (x : ℝ) :
  (5 + 2 / x) ^ (1/3 : ℝ) = -3 → x = -(1/16) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3441_344156


namespace NUMINAMATH_CALUDE_tom_helicopter_rental_days_l3441_344190

/-- Calculates the number of days a helicopter was rented given the rental conditions and total payment -/
def helicopter_rental_days (hours_per_day : ℕ) (cost_per_hour : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid / (hours_per_day * cost_per_hour)

/-- Theorem: Given Tom's helicopter rental conditions, he rented it for 3 days -/
theorem tom_helicopter_rental_days :
  helicopter_rental_days 2 75 450 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_helicopter_rental_days_l3441_344190


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematics_l3441_344139

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

theorem probability_letter_in_mathematics :
  let unique_letters := mathematics.toList.toFinset
  (unique_letters.card : ℚ) / (alphabet.card : ℚ) = 8 / 26 := by sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematics_l3441_344139


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3441_344188

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3441_344188


namespace NUMINAMATH_CALUDE_bc_is_one_eighth_of_ad_l3441_344147

/-- Given a line segment AD with points B and C on it,
    prove that BC is 1/8 of AD if AB is 3 times BD and AC is 7 times CD -/
theorem bc_is_one_eighth_of_ad 
  (A B C D : EuclideanSpace ℝ (Fin 1))
  (h_B_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = (1 - t) • A + t • D)
  (h_C_on_AD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ C = (1 - s) • A + s • D)
  (h_AB_3BD : dist A B = 3 * dist B D)
  (h_AC_7CD : dist A C = 7 * dist C D) :
  dist B C = (1 / 8 : ℝ) * dist A D :=
sorry

end NUMINAMATH_CALUDE_bc_is_one_eighth_of_ad_l3441_344147


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l3441_344120

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x - π / 3)
  let h (x : ℝ) := g (-x)
  h x = Real.sin (-2 * x - 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l3441_344120


namespace NUMINAMATH_CALUDE_cantaloupes_left_l3441_344152

/-- Represents the number of melons and their prices --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  remaining_honeydews : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of cantaloupes left at the end of the day --/
theorem cantaloupes_left (s : MelonSales)
    (h1 : s.cantaloupe_price = 2)
    (h2 : s.honeydew_price = 3)
    (h3 : s.initial_cantaloupes = 30)
    (h4 : s.initial_honeydews = 27)
    (h5 : s.dropped_cantaloupes = 2)
    (h6 : s.rotten_honeydews = 3)
    (h7 : s.remaining_honeydews = 9)
    (h8 : s.total_revenue = 85) :
    s.initial_cantaloupes - s.dropped_cantaloupes -
    ((s.total_revenue - (s.honeydew_price * (s.initial_honeydews - s.rotten_honeydews - s.remaining_honeydews))) / s.cantaloupe_price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_l3441_344152


namespace NUMINAMATH_CALUDE_lunch_theorem_l3441_344100

def lunch_problem (total_spent friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 1

theorem lunch_theorem :
  lunch_problem 15 8 := by
  sorry

end NUMINAMATH_CALUDE_lunch_theorem_l3441_344100


namespace NUMINAMATH_CALUDE_petya_win_probability_l3441_344153

/-- The "Pile of Stones" game --/
structure PileOfStones :=
  (initial_stones : ℕ)
  (min_take : ℕ)
  (max_take : ℕ)

/-- A player in the "Pile of Stones" game --/
inductive Player
| Petya
| Computer

/-- The strategy of a player --/
def Strategy := ℕ → ℕ

/-- The optimal strategy for the second player --/
def optimal_strategy : Strategy := sorry

/-- A random strategy that always takes between min_take and max_take stones --/
def random_strategy (game : PileOfStones) : Strategy := sorry

/-- The probability of winning for a player given their strategy and the opponent's strategy --/
def win_probability (game : PileOfStones) (player : Player) (player_strategy : Strategy) (opponent_strategy : Strategy) : ℚ := sorry

/-- The main theorem: Petya's probability of winning is 1/256 --/
theorem petya_win_probability :
  let game : PileOfStones := ⟨16, 1, 4⟩
  win_probability game Player.Petya (random_strategy game) optimal_strategy = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_l3441_344153


namespace NUMINAMATH_CALUDE_two_n_is_good_pair_exists_good_pair_greater_than_two_l3441_344142

/-- A pair (m,n) is good if, when erasing every m-th and then every n-th number, 
    and separately erasing every n-th and then every m-th number, 
    any number k that occurs in both resulting lists appears at the same position in both lists -/
def is_good_pair (m n : ℕ) : Prop :=
  ∀ k : ℕ, 
    let pos1 := (k - k / n) - (k - k / n) / m
    let pos2 := k / m - (k / m) / n
    (pos1 ≠ 0 ∧ pos2 ≠ 0) → pos1 = pos2

/-- For any positive integer n, (2,n) is a good pair -/
theorem two_n_is_good_pair : ∀ n : ℕ, n > 0 → is_good_pair 2 n := by sorry

/-- There exists a pair of positive integers (m,n) such that 2 < m < n and (m,n) is a good pair -/
theorem exists_good_pair_greater_than_two : 
  ∃ m n : ℕ, 2 < m ∧ m < n ∧ is_good_pair m n := by sorry

end NUMINAMATH_CALUDE_two_n_is_good_pair_exists_good_pair_greater_than_two_l3441_344142


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3441_344172

theorem tan_alpha_value (α : Real) :
  (∃ x y : Real, x = -1 ∧ y = Real.sqrt 3 ∧ 
   (Real.cos α * x - Real.sin α * y = 0)) →
  Real.tan α = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3441_344172


namespace NUMINAMATH_CALUDE_M_is_line_segment_l3441_344169

-- Define the set of points M(x,y) satisfying the equation
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt ((p.1 - 1)^2 + p.2^2) + Real.sqrt ((p.1 + 1)^2 + p.2^2) = 2}

-- Define the line segment between (-1,0) and (1,0)
def lineSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2*t - 1, 0)}

-- Theorem stating that M is equal to the line segment
theorem M_is_line_segment : M = lineSegment := by sorry

end NUMINAMATH_CALUDE_M_is_line_segment_l3441_344169


namespace NUMINAMATH_CALUDE_remainder_18273_mod_9_l3441_344197

theorem remainder_18273_mod_9 : 18273 % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_18273_mod_9_l3441_344197


namespace NUMINAMATH_CALUDE_specific_figure_triangles_l3441_344191

/-- Represents a triangular figure composed of smaller equilateral triangles. -/
structure TriangularFigure where
  row1 : Nat -- Number of triangles in the first row
  row2 : Nat -- Number of triangles in the second row
  row3 : Nat -- Number of triangles in the third row
  has_outer_triangle : Bool -- Whether there's a large triangle spanning all smaller triangles
  has_diagonal_cut : Bool -- Whether there's a diagonal cut over the bottom two rows

/-- Calculates the total number of triangles in the figure. -/
def total_triangles (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating that for the specific triangular figure described,
    the total number of triangles is 11. -/
theorem specific_figure_triangles :
  let figure : TriangularFigure := {
    row1 := 3,
    row2 := 2,
    row3 := 1,
    has_outer_triangle := true,
    has_diagonal_cut := true
  }
  total_triangles figure = 11 := by sorry

end NUMINAMATH_CALUDE_specific_figure_triangles_l3441_344191


namespace NUMINAMATH_CALUDE_family_reunion_soda_cost_l3441_344185

-- Define the given conditions
def people_attending : ℕ := 5 * 12
def cans_per_box : ℕ := 10
def cost_per_box : ℚ := 2
def cans_per_person : ℕ := 2
def family_members : ℕ := 6

-- Define the theorem
theorem family_reunion_soda_cost :
  (people_attending * cans_per_person / cans_per_box * cost_per_box) / family_members = 4 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_soda_cost_l3441_344185


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3441_344113

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x^2 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ (x y : ℝ), x < y → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3441_344113


namespace NUMINAMATH_CALUDE_positive_number_equality_l3441_344158

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = (2/9) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l3441_344158


namespace NUMINAMATH_CALUDE_triangle_inequality_1_triangle_inequality_2_l3441_344143

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_A : A > 0
  pos_B : B > 0
  pos_C : C > 0
  angle_sum : A + B + C = π

-- State the theorems
theorem triangle_inequality_1 (t : Triangle) :
  1 / t.a^3 + 1 / t.b^3 + 1 / t.c^3 + t.a * t.b * t.c ≥ 2 * Real.sqrt 3 := by
  sorry

theorem triangle_inequality_2 (t : Triangle) :
  1 / t.A + 1 / t.B + 1 / t.C ≥ 9 / π := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_1_triangle_inequality_2_l3441_344143


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l3441_344115

theorem ed_doug_marble_difference :
  ∀ (ed_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) (doug : ℕ),
    ed_initial > doug →
    ed_lost = 20 →
    ed_current = 17 →
    doug = 5 →
    ed_initial = ed_current + ed_lost →
    ed_initial - doug = 32 := by
  sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l3441_344115


namespace NUMINAMATH_CALUDE_second_number_value_l3441_344127

theorem second_number_value (x y z : ℝ) (h1 : x + y + z = 120) (h2 : x / y = 3 / 4) (h3 : y / z = 7 / 9) :
  ∃ (n : ℕ), n = 40 ∧ abs (y - n) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l3441_344127


namespace NUMINAMATH_CALUDE_square_nine_on_top_l3441_344129

-- Define the grid of squares
def Grid := Fin 4 → Fin 4 → Fin 16

-- Define the initial configuration of the grid
def initial_grid : Grid :=
  fun i j => i * 4 + j + 1

-- Define the folding operations
def fold_top_over_bottom (g : Grid) : Grid :=
  fun i j => g (3 - i) j

def fold_bottom_over_top (g : Grid) : Grid :=
  fun i j => g i j

def fold_right_over_left (g : Grid) : Grid :=
  fun i j => g i (3 - j)

def fold_left_over_right (g : Grid) : Grid :=
  fun i j => g i j

-- Define the complete folding sequence
def fold_sequence (g : Grid) : Grid :=
  fold_left_over_right ∘ fold_right_over_left ∘ fold_bottom_over_top ∘ fold_top_over_bottom $ g

-- Theorem stating that after the folding sequence, square 9 is on top
theorem square_nine_on_top :
  (fold_sequence initial_grid) 0 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_nine_on_top_l3441_344129


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l3441_344170

theorem quadratic_roots_theorem (r₁ r₂ : ℝ) (p q : ℝ) : 
  (r₁^2 - 5*r₁ + 6 = 0) →
  (r₂^2 - 5*r₂ + 6 = 0) →
  (r₁^2 + p*r₁^2 + q = 0) →
  (r₂^2 + p*r₂^2 + q = 0) →
  p = -13 ∧ q = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l3441_344170


namespace NUMINAMATH_CALUDE_total_distance_walked_l3441_344195

/-- Represents the hiking trail with flat and uphill sections -/
structure HikingTrail where
  flat_distance : ℝ  -- Distance of flat section (P to Q)
  uphill_distance : ℝ  -- Distance of uphill section (Q to R)

/-- Represents the hiker's journey -/
structure HikerJourney where
  trail : HikingTrail
  flat_speed : ℝ  -- Speed on flat sections
  uphill_speed : ℝ  -- Speed going uphill
  downhill_speed : ℝ  -- Speed going downhill
  total_time : ℝ  -- Total time of the journey in hours
  rest_time : ℝ  -- Time spent resting at point R

/-- Theorem stating the total distance walked by the hiker -/
theorem total_distance_walked (journey : HikerJourney) 
  (h1 : journey.flat_speed = 4)
  (h2 : journey.uphill_speed = 3)
  (h3 : journey.downhill_speed = 6)
  (h4 : journey.total_time = 7)
  (h5 : journey.rest_time = 1)
  (h6 : journey.flat_speed * (journey.total_time - journey.rest_time) / 2 + 
        journey.trail.uphill_distance * (1 / journey.uphill_speed + 1 / journey.downhill_speed) = 
        journey.total_time - journey.rest_time) :
  2 * (journey.trail.flat_distance + journey.trail.uphill_distance) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3441_344195


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_radius_of_circle_l3441_344108

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*x + a = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 8*x - 15*y + 43 = 0

-- Theorem for part (1)
theorem tangent_lines_to_circle (x y : ℝ) :
  circle_M (-8) x y →
  (∃ (t : ℝ), (x = t * (point_P.1 - x) + x ∧ y = t * (point_P.2 - y) + y)) →
  (tangent_line_1 x ∨ tangent_line_2 x y) :=
sorry

-- Theorem for part (2)
theorem radius_of_circle (a : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  circle_M a x₁ y₁ →
  circle_M a x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = -6 →
  ∃ (r : ℝ), r^2 = 7 ∧ 
    ∀ (x y : ℝ), circle_M a x y → (x - 1)^2 + y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_radius_of_circle_l3441_344108


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l3441_344137

theorem floor_sqrt_equality (n : ℕ) : ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l3441_344137


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_l3441_344155

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 80000 + 83961

theorem smallest_B_for_divisibility :
  ∀ B : ℕ, B < 10 →
    (is_divisible_by_4 (seven_digit_number B) → B ≥ 0) ∧
    is_divisible_by_4 (seven_digit_number 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_l3441_344155


namespace NUMINAMATH_CALUDE_tan_negative_585_deg_l3441_344168

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_negative_585_deg : tan_deg (-585) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_585_deg_l3441_344168


namespace NUMINAMATH_CALUDE_height_relation_l3441_344149

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry


end NUMINAMATH_CALUDE_height_relation_l3441_344149


namespace NUMINAMATH_CALUDE_email_sending_ways_l3441_344133

/-- The number of ways to send emails given the number of email addresses and the number of emails to be sent. -/
def number_of_ways (num_addresses : ℕ) (num_emails : ℕ) : ℕ :=
  num_addresses ^ num_emails

/-- Theorem stating that the number of ways to send 5 emails using 3 email addresses is 3^5. -/
theorem email_sending_ways : number_of_ways 3 5 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_email_sending_ways_l3441_344133


namespace NUMINAMATH_CALUDE_general_rule_l3441_344138

theorem general_rule (n : ℕ+) :
  (n + 1 : ℚ) / n + (n + 1 : ℚ) = (n + 2 : ℚ) + 1 / n := by sorry

end NUMINAMATH_CALUDE_general_rule_l3441_344138


namespace NUMINAMATH_CALUDE_solve_equation_l3441_344126

theorem solve_equation : ∃ x : ℝ, (3/2 : ℝ) * x - 3 = 15 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3441_344126


namespace NUMINAMATH_CALUDE_gcd_35_and_number_between_65_and_75_l3441_344189

theorem gcd_35_and_number_between_65_and_75 :
  ∃! n : ℕ, 65 < n ∧ n < 75 ∧ Nat.gcd 35 n = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_35_and_number_between_65_and_75_l3441_344189


namespace NUMINAMATH_CALUDE_bruce_total_payment_l3441_344184

/-- Calculates the total amount Bruce paid for fruits -/
def total_amount_paid (grape_quantity grape_price mango_quantity mango_price
                       orange_quantity orange_price apple_quantity apple_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price +
  orange_quantity * orange_price + apple_quantity * apple_price

/-- Theorem: Bruce paid $1480 for the fruits -/
theorem bruce_total_payment :
  total_amount_paid 9 70 7 55 5 45 3 80 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_bruce_total_payment_l3441_344184


namespace NUMINAMATH_CALUDE_system_solutions_correct_l3441_344106

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, x - 3 * y = -10 ∧ x + y = 6 ∧ x = 2 ∧ y = 4) ∧
  -- System 2
  (∃ x y : ℝ, x / 2 - (y - 1) / 3 = 1 ∧ 4 * x - y = 8 ∧ x = 12 / 5 ∧ y = 8 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_system_solutions_correct_l3441_344106


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3441_344164

theorem min_value_of_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (2/x) + (1/y) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ (2/x₀) + (1/y₀) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3441_344164


namespace NUMINAMATH_CALUDE_fish_difference_l3441_344135

-- Define the sizes of the tanks
def first_tank_size : ℕ := 48
def second_tank_size : ℕ := first_tank_size / 2

-- Define the fish sizes
def first_tank_fish_size : ℕ := 3
def second_tank_fish_size : ℕ := 2

-- Calculate the number of fish in each tank
def fish_in_first_tank : ℕ := first_tank_size / first_tank_fish_size
def fish_in_second_tank : ℕ := second_tank_size / second_tank_fish_size

-- Calculate the number of fish in the first tank after one is eaten
def fish_in_first_tank_after_eating : ℕ := fish_in_first_tank - 1

-- Theorem to prove
theorem fish_difference :
  fish_in_first_tank_after_eating - fish_in_second_tank = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_difference_l3441_344135


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3441_344119

/-- The line 4x + 3y + 9 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 9 = 0 ∧ y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3441_344119


namespace NUMINAMATH_CALUDE_talking_birds_l3441_344131

theorem talking_birds (total : ℕ) (non_talking : ℕ) (talking : ℕ) : 
  total = 77 → non_talking = 13 → talking = total - non_talking → talking = 64 := by
sorry

end NUMINAMATH_CALUDE_talking_birds_l3441_344131


namespace NUMINAMATH_CALUDE_f_range_l3441_344132

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  ∃ x ∈ Set.Icc (0 : ℝ) 3,
  f x = y ∧ -18 ≤ y ∧ y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l3441_344132


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l3441_344102

/-- Given two rectangles of equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (area : ℝ) (width : ℝ) :
  area = 4 * 30 →
  width = 24 →
  area = width * 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l3441_344102


namespace NUMINAMATH_CALUDE_P_properties_l3441_344110

/-- P k l n denotes the number of partitions of n into no more than k summands, 
    each not exceeding l -/
def P (k l n : ℕ) : ℕ :=
  sorry

/-- The four properties of P as stated in the problem -/
theorem P_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l - n)) :=
by sorry

end NUMINAMATH_CALUDE_P_properties_l3441_344110


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l3441_344196

-- Define the trade ratios
def fish_to_bread : ℚ := 4 / 5
def bread_to_rice : ℚ := 6
def fish_to_rice : ℚ := 8 / 3

-- Theorem to prove
theorem fish_value_in_rice : fish_to_rice = 8 / 3 := by
  sorry

#eval fish_to_rice

end NUMINAMATH_CALUDE_fish_value_in_rice_l3441_344196


namespace NUMINAMATH_CALUDE_set_B_characterization_l3441_344122

def U : Set ℕ := {x | x > 0 ∧ Real.log x < 1}

def A : Set ℕ := {x | x ∈ U ∧ ∃ n : ℕ, n ≤ 4 ∧ x = 2*n + 1}

def B : Set ℕ := {x | x ∈ U ∧ x % 2 = 0}

theorem set_B_characterization :
  B = {2, 4, 6, 8} :=
sorry

end NUMINAMATH_CALUDE_set_B_characterization_l3441_344122


namespace NUMINAMATH_CALUDE_pentagon_probability_l3441_344182

/-- A type representing the points on the pentagon --/
inductive PentagonPoint
| Vertex : Fin 5 → PentagonPoint
| Midpoint : Fin 5 → PentagonPoint

/-- The total number of points on the pentagon --/
def total_points : ℕ := 10

/-- A function to determine if two points are exactly one side apart --/
def one_side_apart (p q : PentagonPoint) : Prop :=
  match p, q with
  | PentagonPoint.Vertex i, PentagonPoint.Vertex j => (j - i) % 5 = 2 ∨ (i - j) % 5 = 2
  | _, _ => False

/-- The number of ways to choose 2 points from the total points --/
def total_choices : ℕ := (total_points.choose 2)

/-- The number of point pairs that are one side apart --/
def favorable_choices : ℕ := 10

theorem pentagon_probability :
  (favorable_choices : ℚ) / total_choices = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_pentagon_probability_l3441_344182


namespace NUMINAMATH_CALUDE_blue_eyed_students_l3441_344178

theorem blue_eyed_students (total_students : ℕ) (blond_blue : ℕ) (neither : ℕ) :
  total_students = 30 →
  blond_blue = 6 →
  neither = 3 →
  ∃ (blue_eyes : ℕ),
    blue_eyes = 11 ∧
    2 * blue_eyes + (blue_eyes - blond_blue) + neither = total_students :=
by sorry

end NUMINAMATH_CALUDE_blue_eyed_students_l3441_344178


namespace NUMINAMATH_CALUDE_radical_equation_condition_l3441_344157

theorem radical_equation_condition (x y : ℝ) : 
  xy ≠ 0 → (Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_radical_equation_condition_l3441_344157


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3441_344109

theorem fraction_equivalence : 
  let original := 8 / 9
  let target := 4 / 5
  let subtracted := 4
  (8 - subtracted) / (9 - subtracted) = target := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3441_344109


namespace NUMINAMATH_CALUDE_parabola_translation_l3441_344128

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 6 0 0
  let translated := translate original 2 3
  y = 6 * x^2 → y = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3441_344128


namespace NUMINAMATH_CALUDE_marble_fraction_l3441_344161

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := (1/3) * total
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_l3441_344161


namespace NUMINAMATH_CALUDE_weight_of_B_l3441_344163

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 43)
  (h2 : (A + B) / 2 = 48)
  (h3 : (B + C) / 2 = 42) :
  B = 51 := by sorry

end NUMINAMATH_CALUDE_weight_of_B_l3441_344163


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l3441_344187

/-- The number of sections created by n line segments in a rectangle, 
    where each new line intersects all previous lines -/
def maxSections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else maxSections (n - 1) + n

/-- Theorem stating that 5 line segments can create at most 16 sections in a rectangle -/
theorem max_sections_five_lines :
  maxSections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l3441_344187


namespace NUMINAMATH_CALUDE_total_paths_a_to_d_l3441_344124

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- Theorem: The total number of paths from A to D is 9 -/
theorem total_paths_a_to_d : 
  paths_between_adjacent^3 + direct_paths = 9 := by sorry

end NUMINAMATH_CALUDE_total_paths_a_to_d_l3441_344124


namespace NUMINAMATH_CALUDE_f_not_bounded_on_neg_reals_a_range_when_f_bounded_l3441_344118

-- Define the function f(x) = 1 + x + ax^2
def f (a : ℝ) (x : ℝ) : ℝ := 1 + x + a * x^2

-- Part 1: f(x) is not bounded on (-∞, 0) when a = -1
theorem f_not_bounded_on_neg_reals :
  ¬ ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → |f (-1) x| ≤ M :=
sorry

-- Part 2: If |f(x)| ≤ 3 for all x ∈ [1, 4], then a ∈ [-1/2, -1/8]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x, x ∈ Set.Icc 1 4 → |f a x| ≤ 3) →
  a ∈ Set.Icc (-1/2) (-1/8) :=
sorry

end NUMINAMATH_CALUDE_f_not_bounded_on_neg_reals_a_range_when_f_bounded_l3441_344118


namespace NUMINAMATH_CALUDE_jump_rope_cost_is_seven_l3441_344123

/-- The cost of Dalton's jump rope --/
def jump_rope_cost (board_game_cost playground_ball_cost allowance_savings uncle_gift additional_needed : ℕ) : ℕ :=
  (allowance_savings + uncle_gift + additional_needed) - (board_game_cost + playground_ball_cost)

/-- Theorem stating that the jump rope costs $7 --/
theorem jump_rope_cost_is_seven :
  jump_rope_cost 12 4 6 13 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_cost_is_seven_l3441_344123


namespace NUMINAMATH_CALUDE_ratio_ties_to_losses_l3441_344117

def total_games : ℕ := 56
def losses : ℕ := 12
def wins : ℕ := 38

def ties : ℕ := total_games - (losses + wins)

theorem ratio_ties_to_losses :
  (ties : ℚ) / losses = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_ties_to_losses_l3441_344117


namespace NUMINAMATH_CALUDE_initial_rope_length_l3441_344121

theorem initial_rope_length 
  (r_initial : ℝ) 
  (h1 : r_initial > 0) 
  (h2 : π * (21^2 - r_initial^2) = 933.4285714285714) : 
  r_initial = 12 := by
sorry

end NUMINAMATH_CALUDE_initial_rope_length_l3441_344121


namespace NUMINAMATH_CALUDE_remainder_theorem_l3441_344181

theorem remainder_theorem (n : ℕ) (h : n % 7 = 5) : (3 * n + 2)^2 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3441_344181


namespace NUMINAMATH_CALUDE_fruit_bags_weight_l3441_344186

theorem fruit_bags_weight (x y z : ℝ) 
  (h1 : x + y = 90) 
  (h2 : y + z = 100) 
  (h3 : z + x = 110) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) : 
  x + y + z = 150 := by
sorry

end NUMINAMATH_CALUDE_fruit_bags_weight_l3441_344186


namespace NUMINAMATH_CALUDE_mowing_problem_l3441_344144

/-- Represents the time it takes to mow a lawn together -/
def mowing_time (mary_rate tom_rate : ℚ) (tom_alone_time : ℚ) : ℚ :=
  let remaining_lawn := 1 - tom_rate * tom_alone_time
  remaining_lawn / (mary_rate + tom_rate)

theorem mowing_problem :
  let mary_rate : ℚ := 1 / 3  -- Mary's mowing rate (lawn per hour)
  let tom_rate : ℚ := 1 / 6   -- Tom's mowing rate (lawn per hour)
  let tom_alone_time : ℚ := 2 -- Time Tom mows alone (hours)
  mowing_time mary_rate tom_rate tom_alone_time = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_mowing_problem_l3441_344144


namespace NUMINAMATH_CALUDE_salary_for_may_l3441_344146

/-- Proves that the salary for May is 3600, given the average salaries for two sets of four months and the salary for January. -/
theorem salary_for_may (jan feb mar apr may : ℝ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8900 →
  jan = 2900 →
  may = 3600 := by
  sorry

end NUMINAMATH_CALUDE_salary_for_may_l3441_344146


namespace NUMINAMATH_CALUDE_tickets_bought_l3441_344162

theorem tickets_bought (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 := by
sorry

end NUMINAMATH_CALUDE_tickets_bought_l3441_344162


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3441_344136

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b = 2*a) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3441_344136


namespace NUMINAMATH_CALUDE_enrollment_change_l3441_344159

theorem enrollment_change (E : ℝ) (E_1992 E_1993 E_1994 E_1995 : ℝ) : 
  E_1992 = 1.20 * E →
  E_1993 = 1.15 * E_1992 →
  E_1994 = 0.90 * E_1993 →
  E_1995 = 1.25 * E_1994 →
  (E_1995 - E) / E = 0.5525 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_change_l3441_344159


namespace NUMINAMATH_CALUDE_domain_of_g_l3441_344150

-- Define the function f with domain (2, 4)
def f : Set ℝ → Set ℝ := λ S => {x | 2 < x ∧ x < 4}

-- Define the function g(x) = f(log₂ x)
def g (f : Set ℝ → Set ℝ) : Set ℝ := λ x => ∃ y ∈ f (Set.univ), x = 2^y

-- Theorem statement
theorem domain_of_g (f : Set ℝ → Set ℝ) :
  (∀ x ∈ f (Set.univ), 2 < x ∧ x < 4) →
  (∀ x ∈ g f, 4 < x ∧ x < 16) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3441_344150


namespace NUMINAMATH_CALUDE_complement_N_intersect_M_l3441_344104

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Define set N
def N : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_N_intersect_M :
  (U \ N) ∩ M = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_N_intersect_M_l3441_344104


namespace NUMINAMATH_CALUDE_equation_solution_l3441_344141

theorem equation_solution (a b : ℝ) (h : a - b = 0) : 
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3441_344141


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3441_344107

theorem diophantine_equation_solution (t : ℤ) :
  ∀ x y : ℤ, x^4 + 2*x^3 + 8*x - 35*y + 9 = 0 ↔
  (x = 35*t + 6 ∨ x = 35*t - 4 ∨ x = 35*t - 9 ∨ x = 35*t - 16 ∨ x = 35*t - 1 ∨ x = 35*t - 11) ∧
  y = (x^4 + 2*x^3 + 8*x + 9) / 35 :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3441_344107


namespace NUMINAMATH_CALUDE_fraction_valid_for_all_reals_l3441_344112

theorem fraction_valid_for_all_reals :
  ∀ x : ℝ, (x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_valid_for_all_reals_l3441_344112


namespace NUMINAMATH_CALUDE_notebook_cost_l3441_344101

/-- Represents the problem of determining the cost of notebooks --/
theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (cost_per_notebook : ℕ) 
  (total_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  cost_per_notebook > notebooks_per_buyer →
  buyers * notebooks_per_buyer * cost_per_notebook = total_cost →
  total_cost = 2275 →
  cost_per_notebook = 13 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l3441_344101


namespace NUMINAMATH_CALUDE_vector_operation_l3441_344145

/-- Given vectors a and b, prove that -3a - 2b equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (-1, 2)) :
  (-3 : ℝ) • a - (2 : ℝ) • b = (-7, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3441_344145


namespace NUMINAMATH_CALUDE_carl_savings_problem_l3441_344174

theorem carl_savings_problem (weekly_savings : ℚ) : 
  (4 * weekly_savings + 70 = 170) → weekly_savings = 25 := by
  sorry

#check carl_savings_problem

end NUMINAMATH_CALUDE_carl_savings_problem_l3441_344174


namespace NUMINAMATH_CALUDE_new_crew_member_weight_l3441_344103

/-- Given a crew of oarsmen, prove that replacing a crew member results in a specific weight for the new crew member. -/
theorem new_crew_member_weight
  (n : ℕ) -- Number of oarsmen
  (avg_increase : ℝ) -- Increase in average weight
  (old_weight : ℝ) -- Weight of the replaced crew member
  (h1 : n = 20) -- There are 20 oarsmen
  (h2 : avg_increase = 2) -- Average weight increases by 2 kg
  (h3 : old_weight = 40) -- The replaced crew member weighs 40 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by sorry

end NUMINAMATH_CALUDE_new_crew_member_weight_l3441_344103


namespace NUMINAMATH_CALUDE_rose_cost_l3441_344114

/-- The cost of each red rose, given the conditions of Jezebel's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_sunflowers : ℕ) (sunflower_cost : ℚ) (total_cost : ℚ) :
  num_roses = 24 →
  num_sunflowers = 3 →
  sunflower_cost = 3 →
  total_cost = 45 →
  (total_cost - num_sunflowers * sunflower_cost) / num_roses = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_l3441_344114


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l3441_344105

/-- The perimeter of a pentagon with side lengths 2, √8, √18, √32, and √62 is 2 + 9√2 + √62 -/
theorem pentagon_perimeter : 
  let side1 : ℝ := 2
  let side2 : ℝ := Real.sqrt 8
  let side3 : ℝ := Real.sqrt 18
  let side4 : ℝ := Real.sqrt 32
  let side5 : ℝ := Real.sqrt 62
  side1 + side2 + side3 + side4 + side5 = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l3441_344105


namespace NUMINAMATH_CALUDE_exam_time_allocation_l3441_344140

/-- Represents the time allocation for an examination with two types of problems. -/
structure ExamTimeAllocation where
  totalQuestions : ℕ
  examDurationHours : ℕ
  typeAQuestions : ℕ
  typeATimeFactor : ℕ

/-- Calculates the time spent on Type A problems in minutes. -/
def timeSpentOnTypeA (e : ExamTimeAllocation) : ℕ :=
  let totalMinutes := e.examDurationHours * 60
  let typeBQuestions := e.totalQuestions - e.typeAQuestions
  let x := totalMinutes / (e.typeAQuestions * e.typeATimeFactor + typeBQuestions)
  e.typeAQuestions * (x * e.typeATimeFactor)

/-- Theorem stating that for the given exam parameters, 120 minutes should be spent on Type A problems. -/
theorem exam_time_allocation :
  let e : ExamTimeAllocation := {
    totalQuestions := 200,
    examDurationHours := 3,
    typeAQuestions := 100,
    typeATimeFactor := 2
  }
  timeSpentOnTypeA e = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l3441_344140


namespace NUMINAMATH_CALUDE_divisor_square_equals_one_l3441_344173

/-- d(n) represents the number of positive divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem stating that if n equals the square of its number of divisors, then n must be 1 -/
theorem divisor_square_equals_one (n : ℕ+) : n = (num_divisors n)^2 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_square_equals_one_l3441_344173


namespace NUMINAMATH_CALUDE_inequality_proof_l3441_344125

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 1 + Real.sqrt (2 * (1 - a) * (1 - b) * (1 - c))) :
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3441_344125


namespace NUMINAMATH_CALUDE_sequence_sum_equals_321_64_l3441_344160

def sequence_term (n : ℕ) : ℚ := (2^n - 1) / 2^n

def sum_of_terms (n : ℕ) : ℚ := n - 1 + 1 / 2^(n+1)

theorem sequence_sum_equals_321_64 :
  ∃ n : ℕ, sum_of_terms n = 321 / 64 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_321_64_l3441_344160


namespace NUMINAMATH_CALUDE_alyssa_future_games_l3441_344148

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem stating that Alyssa plans to attend 15 soccer games next year -/
theorem alyssa_future_games : 
  games_next_year 11 13 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_future_games_l3441_344148


namespace NUMINAMATH_CALUDE_birds_joined_l3441_344151

theorem birds_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) :
  initial_birds = 3 →
  initial_storks = 2 →
  final_total = 10 →
  final_total - (initial_birds + initial_storks) = 5 := by
  sorry

end NUMINAMATH_CALUDE_birds_joined_l3441_344151


namespace NUMINAMATH_CALUDE_inequality_proof_l3441_344180

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3441_344180


namespace NUMINAMATH_CALUDE_dinosaur_weight_theorem_l3441_344154

/-- The combined weight of Barney, five regular dinosaurs, and their food -/
def total_weight (regular_weight food_weight : ℕ) : ℕ :=
  let regular_combined := 5 * regular_weight
  let barney_weight := regular_combined + 1500
  barney_weight + regular_combined + food_weight

/-- Theorem stating the total weight of the dinosaurs and their food -/
theorem dinosaur_weight_theorem (X : ℕ) :
  total_weight 800 X = 9500 + X :=
by
  sorry

end NUMINAMATH_CALUDE_dinosaur_weight_theorem_l3441_344154


namespace NUMINAMATH_CALUDE_pumps_to_fill_tires_l3441_344116

/-- Represents the capacity of a single tire in cubic inches -/
def tireCapacity : ℝ := 500

/-- Represents the amount of air injected per pump in cubic inches -/
def airPerPump : ℝ := 50

/-- Calculates the total air needed to fill all tires -/
def totalAirNeeded : ℝ :=
  2 * tireCapacity +  -- Two flat tires
  0.6 * tireCapacity +  -- Tire that's 40% full needs 60% more
  0.3 * tireCapacity  -- Tire that's 70% full needs 30% more

/-- Theorem: The number of pumps required to fill all tires is 29 -/
theorem pumps_to_fill_tires : 
  ⌈totalAirNeeded / airPerPump⌉ = 29 := by sorry

end NUMINAMATH_CALUDE_pumps_to_fill_tires_l3441_344116


namespace NUMINAMATH_CALUDE_half_of_expression_l3441_344134

theorem half_of_expression : (2^12 + 3 * 2^10) / 2 = 2^9 * 7 := by sorry

end NUMINAMATH_CALUDE_half_of_expression_l3441_344134


namespace NUMINAMATH_CALUDE_barbara_paper_problem_l3441_344183

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles Barbara found -/
def num_bundles : ℕ := 3

/-- The number of bunches Barbara found -/
def num_bunches : ℕ := 2

/-- The number of heaps Barbara found -/
def num_heaps : ℕ := 5

/-- The total number of sheets Barbara removed -/
def total_sheets : ℕ := 114

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

theorem barbara_paper_problem :
  sheets_per_bunch * num_bunches + sheets_per_bundle * num_bundles + sheets_per_heap * num_heaps = total_sheets :=
by sorry

end NUMINAMATH_CALUDE_barbara_paper_problem_l3441_344183
