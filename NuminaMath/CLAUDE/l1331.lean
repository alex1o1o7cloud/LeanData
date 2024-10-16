import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l1331_133123

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  -a - b^2 + a*b = -16 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1331_133123


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1331_133131

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) (y : ℝ) : 
  y = 5 → 
  x₁^2 + y^2 = 169 → 
  x₂^2 + y^2 = 169 → 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1331_133131


namespace NUMINAMATH_CALUDE_max_value_theorem_l1331_133196

theorem max_value_theorem (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ a^2 + c^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = a^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1331_133196


namespace NUMINAMATH_CALUDE_solve_system_l1331_133133

theorem solve_system (x y z : ℚ) 
  (eq1 : x - y - z = 8)
  (eq2 : x + y + z = 20)
  (eq3 : x - y + 2*z = 16) :
  z = 8/3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1331_133133


namespace NUMINAMATH_CALUDE_colored_spanning_tree_existence_l1331_133179

/-- A colored edge in a graph -/
inductive ColoredEdge
  | Red
  | Green
  | Blue

/-- A graph with colored edges -/
structure ColoredGraph (V : Type) where
  edges : V → V → Option ColoredEdge

/-- A spanning tree of a graph -/
def SpanningTree (V : Type) := V → V → Prop

/-- Count the number of edges of a specific color in a spanning tree -/
def CountEdges (V : Type) (t : SpanningTree V) (c : ColoredEdge) : ℕ := sorry

/-- The main theorem -/
theorem colored_spanning_tree_existence
  (V : Type)
  (G : ColoredGraph V)
  (n : ℕ)
  (r v b : ℕ)
  (h_connected : sorry)  -- G is connected
  (h_vertex_count : sorry)  -- G has n+1 vertices
  (h_sum : r + v + b = n)
  (h_red_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Red = r)
  (h_green_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Green = v)
  (h_blue_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Blue = b) :
  ∃ t : SpanningTree V,
    CountEdges V t ColoredEdge.Red = r ∧
    CountEdges V t ColoredEdge.Green = v ∧
    CountEdges V t ColoredEdge.Blue = b :=
  sorry

end NUMINAMATH_CALUDE_colored_spanning_tree_existence_l1331_133179


namespace NUMINAMATH_CALUDE_simultaneous_equations_solutions_l1331_133193

theorem simultaneous_equations_solutions :
  let eq1 (x y : ℝ) := x^2 + 3*y = 10
  let eq2 (x y : ℝ) := 3 + y = 10/x
  (eq1 (-5) (-5) ∧ eq2 (-5) (-5)) ∧
  (eq1 2 2 ∧ eq2 2 2) ∧
  (eq1 3 (1/3) ∧ eq2 3 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solutions_l1331_133193


namespace NUMINAMATH_CALUDE_zephyrian_word_count_l1331_133104

/-- The number of letters in the Zephyrian alphabet -/
def zephyrian_alphabet_size : ℕ := 8

/-- The maximum word length in the Zephyrian language -/
def max_word_length : ℕ := 3

/-- Calculate the number of possible words in the Zephyrian language -/
def count_zephyrian_words : ℕ :=
  zephyrian_alphabet_size +
  zephyrian_alphabet_size ^ 2 +
  zephyrian_alphabet_size ^ 3

theorem zephyrian_word_count :
  count_zephyrian_words = 584 :=
sorry

end NUMINAMATH_CALUDE_zephyrian_word_count_l1331_133104


namespace NUMINAMATH_CALUDE_smallest_n_for_logarithm_sum_l1331_133136

theorem smallest_n_for_logarithm_sum : ∃ (n : ℕ), n = 3 ∧ 
  (∀ m : ℕ, m < n → 2^(2^(m+1)) < 512) ∧ 
  2^(2^(n+1)) ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_logarithm_sum_l1331_133136


namespace NUMINAMATH_CALUDE_implicit_function_derivative_l1331_133141

/-- Given an implicitly defined function y²(x) + x² - 1 = 0,
    prove that the derivative of y with respect to x is -x / y(x) -/
theorem implicit_function_derivative 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x ^ 2 + x ^ 2 - 1 = 0) :
  ∀ x, HasDerivAt y (-(x / y x)) x :=
sorry

end NUMINAMATH_CALUDE_implicit_function_derivative_l1331_133141


namespace NUMINAMATH_CALUDE_nonagon_triangles_l1331_133162

/-- The number of triangles formed by vertices of a regular nonagon -/
def triangles_in_nonagon : ℕ := Nat.choose 9 3

/-- Theorem stating that the number of triangles in a regular nonagon is 84 -/
theorem nonagon_triangles : triangles_in_nonagon = 84 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_triangles_l1331_133162


namespace NUMINAMATH_CALUDE_multiply_101_by_101_l1331_133159

theorem multiply_101_by_101 : 101 * 101 = 10201 := by sorry

end NUMINAMATH_CALUDE_multiply_101_by_101_l1331_133159


namespace NUMINAMATH_CALUDE_sand_heap_radius_l1331_133116

/-- Given a cylindrical bucket of sand and a conical heap formed from it, 
    prove that the radius of the heap's base is 63 cm. -/
theorem sand_heap_radius : 
  ∀ (h_cylinder r_cylinder h_cone r_cone : ℝ),
  h_cylinder = 36 ∧ 
  r_cylinder = 21 ∧ 
  h_cone = 12 ∧
  π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone →
  r_cone = 63 := by
  sorry

end NUMINAMATH_CALUDE_sand_heap_radius_l1331_133116


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1331_133137

theorem possible_values_of_a (a : ℝ) : 
  let P : Set ℝ := {-1, 2*a+1, a^2-1}
  0 ∈ P → a = -1/2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1331_133137


namespace NUMINAMATH_CALUDE_team_C_most_uniform_l1331_133112

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the variance for each team
def variance : Team → ℝ
| Team.A => 0.13
| Team.B => 0.11
| Team.C => 0.09
| Team.D => 0.15

-- Define a function to determine if a team has the most uniform height
def has_most_uniform_height (t : Team) : Prop :=
  ∀ other : Team, variance t ≤ variance other

-- Theorem: Team C has the most uniform height
theorem team_C_most_uniform : has_most_uniform_height Team.C := by
  sorry


end NUMINAMATH_CALUDE_team_C_most_uniform_l1331_133112


namespace NUMINAMATH_CALUDE_perimeter_of_figure_l1331_133198

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the figure ABCDEFG -/
structure Figure where
  abc : Triangle
  ade : Triangle
  efg : Triangle

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The perimeter of the figure ABCDEFG -/
def Figure.perimeter (f : Figure) : ℝ :=
  f.abc.perimeter + f.ade.c + f.efg.b + f.efg.c - f.abc.c - f.ade.a

theorem perimeter_of_figure (f : Figure) :
  f.abc.a = 5 ∧
  f.abc.a = f.abc.b ∧ f.abc.b = f.abc.c ∧  -- ABC is equilateral
  f.ade.a = f.ade.b ∧ f.ade.b = f.ade.c ∧  -- ADE is equilateral
  f.efg.a = f.efg.c ∧  -- EFG is isosceles
  f.efg.b = 2 * f.efg.a ∧  -- EG is twice EF
  f.abc.c = 2 * f.ade.a ∧  -- D is midpoint of AC
  f.ade.a = 2 * f.efg.a  -- G is midpoint of AE
  →
  f.perimeter = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_l1331_133198


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1331_133128

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1331_133128


namespace NUMINAMATH_CALUDE_distance_to_origin_l1331_133169

/-- The distance between point P(3,1) and the origin (0,0) in the Cartesian coordinate system is √10. -/
theorem distance_to_origin : Real.sqrt ((3 : ℝ) ^ 2 + (1 : ℝ) ^ 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1331_133169


namespace NUMINAMATH_CALUDE_average_books_read_rounded_l1331_133110

/-- Represents the number of books read by each category of members -/
def books_read : List Nat := [1, 2, 3, 4, 5]

/-- Represents the number of members in each category -/
def members : List Nat := [3, 4, 1, 6, 2]

/-- Calculates the total number of books read -/
def total_books : Nat := (List.zip books_read members).map (fun (b, m) => b * m) |>.sum

/-- Calculates the total number of members -/
def total_members : Nat := members.sum

/-- Calculates the average number of books read per member -/
def average : Rat := total_books / total_members

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  ⌊x + 1/2⌋

/-- Main theorem: The average number of books read, rounded to the nearest whole number, is 3 -/
theorem average_books_read_rounded : round_to_nearest average = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_books_read_rounded_l1331_133110


namespace NUMINAMATH_CALUDE_tile_coverage_l1331_133119

theorem tile_coverage (original_count : ℕ) (original_side : ℝ) (new_side : ℝ) :
  original_count = 96 →
  original_side = 3 →
  new_side = 2 →
  (original_count * original_side * original_side) / (new_side * new_side) = 216 := by
  sorry

end NUMINAMATH_CALUDE_tile_coverage_l1331_133119


namespace NUMINAMATH_CALUDE_min_distance_sum_l1331_133170

/-- A line in 2D space passing through (1,4) and intersecting positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through_point : 1 / a + 4 / b = 1

/-- The sum of distances from origin to intersection points is at least 9 -/
theorem min_distance_sum (l : IntersectingLine) :
  l.a + l.b ≥ 9 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_min_distance_sum_l1331_133170


namespace NUMINAMATH_CALUDE_remaining_money_l1331_133114

/-- Calculates the remaining money after purchases and discount --/
theorem remaining_money (initial_amount purchases discount_rate : ℚ) : 
  initial_amount = 10 ∧ 
  purchases = 3 + 2 + 1.5 + 0.75 ∧ 
  discount_rate = 0.05 → 
  initial_amount - (purchases - purchases * discount_rate) = 311/100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1331_133114


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l1331_133183

/-- The set of all letters available --/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants --/
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels --/
def vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering --/
def word_length : Nat := 5

/-- A function that returns the number of words with at least one consonant --/
def words_with_consonant : Nat :=
  letters.card ^ word_length - vowels.card ^ word_length

theorem words_with_consonant_count :
  words_with_consonant = 7744 := by sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l1331_133183


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1331_133160

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1331_133160


namespace NUMINAMATH_CALUDE_solve_system_for_x_l1331_133125

theorem solve_system_for_x (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y + z = 8) 
  (eq2 : x + 3 * y - 2 * z = 2) : 
  x = 58 / 21 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l1331_133125


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1331_133194

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/2 < 0) ↔ k ∈ Set.Ioc (-12) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1331_133194


namespace NUMINAMATH_CALUDE_bisection_method_next_interval_l1331_133188

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a < 0 ∧ f b > 0 ∧ f x₀ > 0 →
  ∃ x ∈ Set.Ioo a x₀, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_bisection_method_next_interval_l1331_133188


namespace NUMINAMATH_CALUDE_cuboid_first_edge_length_l1331_133175

/-- The length of the first edge of a cuboid with volume 30 cm³ and other edges 5 cm and 3 cm -/
def first_edge_length : ℝ := 2

/-- The volume of the cuboid -/
def cuboid_volume : ℝ := 30

/-- The width of the cuboid -/
def cuboid_width : ℝ := 5

/-- The height of the cuboid -/
def cuboid_height : ℝ := 3

theorem cuboid_first_edge_length :
  first_edge_length * cuboid_width * cuboid_height = cuboid_volume :=
by sorry

end NUMINAMATH_CALUDE_cuboid_first_edge_length_l1331_133175


namespace NUMINAMATH_CALUDE_train_length_proof_l1331_133135

/-- Given a train with constant speed that crosses two platforms of different lengths,
    prove that the length of the train is 110 meters. -/
theorem train_length_proof (speed : ℝ) (length : ℝ) :
  speed > 0 →
  speed * 15 = length + 160 →
  speed * 20 = length + 250 →
  length = 110 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l1331_133135


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l1331_133174

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def ways_to_select_chords : ℕ := total_chords.choose k

/-- The number of convex pentagons that can be formed -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / ways_to_select_chords

theorem convex_pentagon_probability : probability = 1 / 1755 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l1331_133174


namespace NUMINAMATH_CALUDE_circle_contains_at_least_250_points_l1331_133191

/-- A circle on a grid --/
structure GridCircle where
  radius : ℝ
  gridSize : ℝ

/-- The number of grid points inside a circle --/
def gridPointsInside (c : GridCircle) : ℕ :=
  sorry

/-- Theorem: A circle with radius 10 on a unit grid contains at least 250 grid points --/
theorem circle_contains_at_least_250_points (c : GridCircle) 
  (h1 : c.radius = 10)
  (h2 : c.gridSize = 1) : 
  gridPointsInside c ≥ 250 := by
  sorry

end NUMINAMATH_CALUDE_circle_contains_at_least_250_points_l1331_133191


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1331_133154

theorem fraction_to_decimal : (31 : ℚ) / (2 * 5^6) = 0.000992 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1331_133154


namespace NUMINAMATH_CALUDE_det_2CD_l1331_133185

theorem det_2CD (C D : Matrix (Fin 3) (Fin 3) ℝ) 
  (hC : Matrix.det C = 3)
  (hD : Matrix.det D = 8) :
  Matrix.det (2 • (C * D)) = 192 := by
  sorry

end NUMINAMATH_CALUDE_det_2CD_l1331_133185


namespace NUMINAMATH_CALUDE_lottery_investment_ratio_l1331_133181

def lottery_winnings : ℕ := 12006
def savings_amount : ℕ := 1000
def fun_money : ℕ := 2802

theorem lottery_investment_ratio :
  let after_tax := lottery_winnings / 2
  let after_loans := after_tax - (after_tax / 3)
  let after_savings := after_loans - savings_amount
  let stock_investment := after_savings - fun_money
  (stock_investment : ℚ) / savings_amount = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_lottery_investment_ratio_l1331_133181


namespace NUMINAMATH_CALUDE_inequality_proof_l1331_133139

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1331_133139


namespace NUMINAMATH_CALUDE_xy_values_l1331_133121

theorem xy_values (x y : ℝ) : (x + y + 2) * (x + y - 1) = 0 → x + y = -2 ∨ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l1331_133121


namespace NUMINAMATH_CALUDE_stamps_on_last_page_is_four_l1331_133132

/-- The number of stamps on the last page of Jenny's seventh book after reorganization --/
def stamps_on_last_page (
  initial_books : ℕ)
  (pages_per_book : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (full_books_after_reorg : ℕ)
  (full_pages_in_last_book : ℕ) : ℕ :=
  let total_stamps := initial_books * pages_per_book * initial_stamps_per_page
  let stamps_in_full_books := full_books_after_reorg * pages_per_book * new_stamps_per_page
  let stamps_in_full_pages_of_last_book := full_pages_in_last_book * new_stamps_per_page
  total_stamps - stamps_in_full_books - stamps_in_full_pages_of_last_book

/-- Theorem stating that under the given conditions, there are 4 stamps on the last page --/
theorem stamps_on_last_page_is_four :
  stamps_on_last_page 10 50 8 12 6 37 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stamps_on_last_page_is_four_l1331_133132


namespace NUMINAMATH_CALUDE_contrapositive_example_l1331_133111

theorem contrapositive_example : 
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1331_133111


namespace NUMINAMATH_CALUDE_existence_of_close_multiple_l1331_133168

theorem existence_of_close_multiple (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_multiple_l1331_133168


namespace NUMINAMATH_CALUDE_random_walk_properties_l1331_133129

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by sorry

end NUMINAMATH_CALUDE_random_walk_properties_l1331_133129


namespace NUMINAMATH_CALUDE_sqrt5_irrational_and_greater_than_sqrt3_l1331_133145

theorem sqrt5_irrational_and_greater_than_sqrt3 : 
  Irrational (Real.sqrt 5) ∧ Real.sqrt 5 > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_irrational_and_greater_than_sqrt3_l1331_133145


namespace NUMINAMATH_CALUDE_airline_rows_calculation_l1331_133166

/-- Represents an airline company with a fleet of airplanes -/
structure AirlineCompany where
  num_airplanes : ℕ
  rows_per_airplane : ℕ
  seats_per_row : ℕ
  flights_per_day : ℕ
  total_daily_capacity : ℕ

/-- Theorem: Given the airline company's specifications, prove that each airplane has 20 rows -/
theorem airline_rows_calculation (airline : AirlineCompany)
    (h1 : airline.num_airplanes = 5)
    (h2 : airline.seats_per_row = 7)
    (h3 : airline.flights_per_day = 2)
    (h4 : airline.total_daily_capacity = 1400) :
    airline.rows_per_airplane = 20 := by
  sorry

/-- Example airline company satisfying the given conditions -/
def example_airline : AirlineCompany :=
  { num_airplanes := 5
    rows_per_airplane := 20  -- This is what we're proving
    seats_per_row := 7
    flights_per_day := 2
    total_daily_capacity := 1400 }

end NUMINAMATH_CALUDE_airline_rows_calculation_l1331_133166


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1331_133149

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ (1/2 : ℝ) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1331_133149


namespace NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l1331_133122

/-- The probability that all 3 girls select the same colored marble from a bag with 3 white and 3 black marbles -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 6
  let white_marbles : ℕ := 3
  let black_marbles : ℕ := 3
  let girls : ℕ := 3
  let prob_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_all_black : ℚ := (black_marbles / total_marbles) * ((black_marbles - 1) / (total_marbles - 1)) * ((black_marbles - 2) / (total_marbles - 2))
  prob_all_white + prob_all_black

theorem same_color_marble_probability_is_one_twentieth : same_color_marble_probability = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l1331_133122


namespace NUMINAMATH_CALUDE_business_value_calculation_l1331_133158

theorem business_value_calculation (man_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_portion = 3/5 →
  sale_price = 2000 →
  ∃ (total_value : ℕ), total_value = 10000 ∧ 
    (sold_portion * man_share * total_value : ℚ) = sale_price := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l1331_133158


namespace NUMINAMATH_CALUDE_star_operations_l1331_133106

-- Define the new operation
def star (x y : ℚ) : ℚ := x * y + |x - y| - 2

-- Theorem statement
theorem star_operations :
  (star 3 (-2) = -3) ∧ (star (star 2 5) (-4) = -31) := by
  sorry

end NUMINAMATH_CALUDE_star_operations_l1331_133106


namespace NUMINAMATH_CALUDE_triangle_area_from_rectangle_l1331_133124

/-- The area of one right triangle formed by cutting a rectangle diagonally --/
theorem triangle_area_from_rectangle (length width : Real) (h_length : length = 0.5) (h_width : width = 0.3) :
  (length * width) / 2 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_rectangle_l1331_133124


namespace NUMINAMATH_CALUDE_total_seashells_is_58_l1331_133157

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found -/
def total_seashells : ℕ := tom_seashells + fred_seashells

/-- Theorem stating that the total number of seashells found is 58 -/
theorem total_seashells_is_58 : total_seashells = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_58_l1331_133157


namespace NUMINAMATH_CALUDE_prism_volume_l1331_133192

/-- 
  Given a right prism with an isosceles triangle base ABC, where:
  - AB = AC
  - ∠BAC = α
  - A line segment of length l from the upper vertex A₁ to the center of 
    the circumscribed circle of ABC makes an angle β with the base plane
  
  The volume of the prism is l³ sin(2β) cos(β) sin(α) cos²(α/2)
-/
theorem prism_volume 
  (α β l : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_β : 0 < β ∧ β < π/2) 
  (h_l : l > 0) : 
  ∃ (V : ℝ), V = l^3 * Real.sin (2*β) * Real.cos β * Real.sin α * (Real.cos (α/2))^2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1331_133192


namespace NUMINAMATH_CALUDE_second_triangle_invalid_l1331_133100

-- Define the sides of the triangle
def a : ℝ := 15
def b : ℝ := 15
def c : ℝ := 30

-- Define the condition for a valid triangle (triangle inequality)
def is_valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem second_triangle_invalid :
  ¬(is_valid_triangle a b c) :=
sorry

end NUMINAMATH_CALUDE_second_triangle_invalid_l1331_133100


namespace NUMINAMATH_CALUDE_average_decrease_l1331_133190

theorem average_decrease (initial_count : ℕ) (initial_avg : ℚ) (new_obs : ℚ) :
  initial_count = 6 →
  initial_avg = 13 →
  new_obs = 6 →
  let total_sum := initial_count * initial_avg
  let new_sum := total_sum + new_obs
  let new_count := initial_count + 1
  let new_avg := new_sum / new_count
  initial_avg - new_avg = 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l1331_133190


namespace NUMINAMATH_CALUDE_larger_number_proof_l1331_133148

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1331_133148


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1331_133167

def alice_number : ℕ := 30

def has_all_prime_factors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : has_all_prime_factors alice_number bob_number)
  (h2 : 5 ∣ bob_number) :
  bob_number ≥ 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1331_133167


namespace NUMINAMATH_CALUDE_ab_minus_c_equals_six_l1331_133173

theorem ab_minus_c_equals_six (a b c : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c^2 = a*b + b - 9) : 
  a*b - c = 6 := by
sorry

end NUMINAMATH_CALUDE_ab_minus_c_equals_six_l1331_133173


namespace NUMINAMATH_CALUDE_cyclists_problem_l1331_133155

/-- The problem of two cyclists traveling between Huntington and Montauk -/
theorem cyclists_problem (x y : ℝ) : 
  (y = x + 6) →                   -- Y is 6 mph faster than X
  (80 / x = (80 + 16) / y) →      -- Time taken by X equals time taken by Y
  (x = 12) :=                     -- X's speed is 12 mph
by sorry

end NUMINAMATH_CALUDE_cyclists_problem_l1331_133155


namespace NUMINAMATH_CALUDE_power_function_through_point_l1331_133118

theorem power_function_through_point (α : ℝ) : 2^α = Real.sqrt 2 → α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1331_133118


namespace NUMINAMATH_CALUDE_probability_less_than_20_l1331_133107

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l1331_133107


namespace NUMINAMATH_CALUDE_cornelia_countries_l1331_133180

/-- The number of countries Cornelia visited in Europe -/
def europe_countries : ℕ := 20

/-- The number of countries Cornelia visited in South America -/
def south_america_countries : ℕ := 10

/-- The number of countries Cornelia visited in Asia -/
def asia_countries : ℕ := 6

/-- The total number of countries Cornelia visited -/
def total_countries : ℕ := europe_countries + south_america_countries + 2 * asia_countries

theorem cornelia_countries : total_countries = 42 := by
  sorry

end NUMINAMATH_CALUDE_cornelia_countries_l1331_133180


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1331_133177

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 →
  biology_percentage = 35 / 100 →
  total_students - (biology_percentage * total_students).floor = 572 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1331_133177


namespace NUMINAMATH_CALUDE_expression_greater_than_e_l1331_133172

theorem expression_greater_than_e (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  Real.exp y - 8/x > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_expression_greater_than_e_l1331_133172


namespace NUMINAMATH_CALUDE_dance_students_l1331_133150

/-- Represents the number of students taking each elective in a school -/
structure SchoolElectives where
  total : ℕ
  art : ℕ
  music : ℕ
  dance : ℕ

/-- The properties of the school electives -/
def valid_electives (s : SchoolElectives) : Prop :=
  s.total = 400 ∧
  s.art = 200 ∧
  s.music = s.total / 5 ∧
  s.total = s.art + s.music + s.dance

/-- Theorem stating that the number of students taking dance is 120 -/
theorem dance_students (s : SchoolElectives) (h : valid_electives s) : s.dance = 120 := by
  sorry

end NUMINAMATH_CALUDE_dance_students_l1331_133150


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1331_133115

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1331_133115


namespace NUMINAMATH_CALUDE_no_solution_to_system_l1331_133189

theorem no_solution_to_system :
  ¬∃ (x y z : ℝ), (x^2 - 2*y + 2 = 0) ∧ (y^2 - 4*z + 3 = 0) ∧ (z^2 + 4*x + 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l1331_133189


namespace NUMINAMATH_CALUDE_percentage_of_whole_l1331_133151

theorem percentage_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 40.25 ↔ part = 193.2 ∧ whole = 480 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_whole_l1331_133151


namespace NUMINAMATH_CALUDE_number_with_given_division_l1331_133146

theorem number_with_given_division : ∃ n : ℕ, n = 100 ∧ n / 11 = 9 ∧ n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_with_given_division_l1331_133146


namespace NUMINAMATH_CALUDE_geometric_progression_solutions_l1331_133130

theorem geometric_progression_solutions : 
  ∃ (x₁ x₂ a₁ a₂ : ℝ), 
    (x₁ = 2 ∧ a₁ = 3 ∧ 3 * |x₁| * Real.sqrt (x₁ + 2) = 5 * x₁ + 2) ∧
    (x₂ = -2/9 ∧ a₂ = 1/2 ∧ 3 * |x₂| * Real.sqrt (x₂ + 2) = 5 * x₂ + 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solutions_l1331_133130


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l1331_133102

theorem division_multiplication_equality : (1100 / 25) * 4 / 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l1331_133102


namespace NUMINAMATH_CALUDE_complex_equation_circle_l1331_133105

/-- The set of complex numbers z satisfying |z|^2 + |z| = 2 forms a circle in the complex plane. -/
theorem complex_equation_circle : 
  {z : ℂ | Complex.abs z ^ 2 + Complex.abs z = 2} = {z : ℂ | Complex.abs z = 1} := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_circle_l1331_133105


namespace NUMINAMATH_CALUDE_minimize_y_l1331_133182

variable (a b c x : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*c*x

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x ≥ y x_min ∧ x_min = (a + b - c) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l1331_133182


namespace NUMINAMATH_CALUDE_jack_book_loss_l1331_133138

/-- Calculates the amount of money Jack lost in a year buying and selling books. -/
theorem jack_book_loss (books_per_month : ℕ) (book_cost : ℕ) (selling_price : ℕ) (months_per_year : ℕ) : 
  books_per_month = 3 →
  book_cost = 20 →
  selling_price = 500 →
  months_per_year = 12 →
  (books_per_month * months_per_year * book_cost) - selling_price = 220 := by
sorry

end NUMINAMATH_CALUDE_jack_book_loss_l1331_133138


namespace NUMINAMATH_CALUDE_curtis_farm_chickens_l1331_133161

/-- The number of chickens on Mr. Curtis's farm -/
theorem curtis_farm_chickens :
  let roosters : ℕ := 28
  let non_egg_laying_hens : ℕ := 20
  let egg_laying_hens : ℕ := 277
  roosters + non_egg_laying_hens + egg_laying_hens = 325 :=
by sorry

end NUMINAMATH_CALUDE_curtis_farm_chickens_l1331_133161


namespace NUMINAMATH_CALUDE_imaginary_part_sum_of_fractions_l1331_133113

theorem imaginary_part_sum_of_fractions :
  Complex.im (1 / (Complex.ofReal (-2) + Complex.I) + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_of_fractions_l1331_133113


namespace NUMINAMATH_CALUDE_friends_signed_up_first_day_l1331_133152

/-- The number of friends who signed up on the first day -/
def friends_first_day : ℕ := sorry

/-- The total number of friends who signed up (including first day and rest of the week) -/
def total_friends : ℕ := friends_first_day + 7

/-- The total money earned by Katrina and her friends -/
def total_money : ℕ := 125

theorem friends_signed_up_first_day : 
  5 + 5 * total_friends + 5 * total_friends = total_money ∧ friends_first_day = 5 := by sorry

end NUMINAMATH_CALUDE_friends_signed_up_first_day_l1331_133152


namespace NUMINAMATH_CALUDE_oil_water_ratio_l1331_133156

/-- Represents the capacity and contents of a bottle -/
structure Bottle where
  capacity : ℝ
  oil : ℝ
  water : ℝ

/-- The problem setup -/
def bottleProblem (C_A : ℝ) : Prop :=
  ∃ (A B C D : Bottle),
    A.capacity = C_A ∧
    A.oil = C_A / 2 ∧
    A.water = C_A / 2 ∧
    B.capacity = 2 * C_A ∧
    B.oil = C_A / 2 ∧
    B.water = 3 * C_A / 2 ∧
    C.capacity = 3 * C_A ∧
    C.oil = C_A ∧
    C.water = 2 * C_A ∧
    D.capacity = 4 * C_A ∧
    D.oil = 0 ∧
    D.water = 0

/-- The theorem to prove -/
theorem oil_water_ratio (C_A : ℝ) (h : C_A > 0) :
  bottleProblem C_A →
  ∃ (D_final : Bottle),
    D_final.capacity = 4 * C_A ∧
    D_final.oil = 2 * C_A ∧
    D_final.water = 3.7 * C_A :=
by
  sorry

#check oil_water_ratio

end NUMINAMATH_CALUDE_oil_water_ratio_l1331_133156


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1331_133126

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 160 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1331_133126


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1331_133144

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 > x

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1331_133144


namespace NUMINAMATH_CALUDE_johns_salary_increase_l1331_133165

theorem johns_salary_increase (original_salary new_salary : ℝ) 
  (h1 : new_salary = 110)
  (h2 : new_salary = original_salary * (1 + 0.8333333333333334)) : 
  original_salary = 60 := by
sorry

end NUMINAMATH_CALUDE_johns_salary_increase_l1331_133165


namespace NUMINAMATH_CALUDE_solve_equation_l1331_133197

theorem solve_equation (n m x : ℚ) 
  (h1 : (7 : ℚ) / 8 = n / 96)
  (h2 : (7 : ℚ) / 8 = (m + n) / 112)
  (h3 : (7 : ℚ) / 8 = (x - m) / 144) : 
  x = 140 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1331_133197


namespace NUMINAMATH_CALUDE_printer_price_ratio_l1331_133140

/-- Given the price of a basic computer and printer setup, prove the ratio of the printer price
    to the total price of an enhanced computer and printer setup. -/
theorem printer_price_ratio (basic_computer_price printer_price enhanced_computer_price : ℕ) : 
  basic_computer_price + printer_price = 2500 →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2125 →
  printer_price / (enhanced_computer_price + printer_price) = 1 / 8 := by
  sorry

#check printer_price_ratio

end NUMINAMATH_CALUDE_printer_price_ratio_l1331_133140


namespace NUMINAMATH_CALUDE_linear_function_composition_l1331_133195

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 4 : ℝ) = 4 * x + 5) →
  a + b = 13/3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l1331_133195


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1331_133101

theorem fraction_evaluation : (36 - 12) / (12 - 4) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1331_133101


namespace NUMINAMATH_CALUDE_train_length_calculation_l1331_133127

/-- The length of a train given jogger and train speeds, initial distance, and time to pass. -/
theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (time_to_pass : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  time_to_pass = 39 →
  (train_speed - jogger_speed) * time_to_pass - initial_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1331_133127


namespace NUMINAMATH_CALUDE_carls_weight_l1331_133147

/-- Given the weights of Al, Ben, Carl, and Ed, prove Carl's weight -/
theorem carls_weight (Al Ben Carl Ed : ℕ) 
  (h1 : Al = Ben + 25)
  (h2 : Ben = Carl - 16)
  (h3 : Ed = 146)
  (h4 : Al = Ed + 38) :
  Carl = 175 := by
  sorry

end NUMINAMATH_CALUDE_carls_weight_l1331_133147


namespace NUMINAMATH_CALUDE_power_product_rule_l1331_133164

theorem power_product_rule (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l1331_133164


namespace NUMINAMATH_CALUDE_bug_total_distance_l1331_133108

def bug_journey (start end1 end2 end3 : ℝ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_total_distance :
  bug_journey 0 4 (-3) 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l1331_133108


namespace NUMINAMATH_CALUDE_green_to_blue_ratio_is_four_to_one_l1331_133117

/-- Represents a gumball machine with red, green, and blue gumballs. -/
structure GumballMachine where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The ratio of two natural numbers as a pair of integers. -/
def Ratio := ℤ × ℤ

/-- Calculates the ratio of green to blue gumballs. -/
def greenToBlueRatio (m : GumballMachine) : Ratio :=
  (m.green, m.blue)

/-- Theorem stating the ratio of green to blue gumballs is 4:1 under given conditions. -/
theorem green_to_blue_ratio_is_four_to_one 
  (m : GumballMachine) 
  (h1 : m.red = 16)
  (h2 : m.blue = m.red / 2)
  (h3 : m.red + m.green + m.blue = 56) :
  greenToBlueRatio m = (4, 1) := by
  sorry

#check green_to_blue_ratio_is_four_to_one

end NUMINAMATH_CALUDE_green_to_blue_ratio_is_four_to_one_l1331_133117


namespace NUMINAMATH_CALUDE_intersection_point_l1331_133120

/-- A quadratic function of the form y = x^2 + px + q where 3p + q = 2023 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : 3 * p + q = 2023

/-- The point (3, 2032) lies on all quadratic functions satisfying the given condition -/
theorem intersection_point (f : QuadraticFunction) : 
  3^2 + f.p * 3 + f.q = 2032 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1331_133120


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1331_133187

/-- Given a right triangle ABC where:
    - The altitude from C to AB is 12 km
    - The sum of all sides (AB + BC + AC) is 60 km
    Prove that the length of AB is 22.5 km -/
theorem right_triangle_side_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (altitude : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2 - 
    (((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 / 
    ((B.1 - A.1)^2 + (B.2 - A.2)^2))) = 12)
  (perimeter : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) + 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 60) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l1331_133187


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1331_133186

theorem intersection_point_sum (c d : ℝ) :
  (∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) →
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) →
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1331_133186


namespace NUMINAMATH_CALUDE_range_of_a_l1331_133143

-- Define the conditions α and β
def α (x : ℝ) : Prop := x ≤ -1 ∨ x > 3
def β (a x : ℝ) : Prop := a - 1 ≤ x ∧ x < a + 2

-- State the theorem
theorem range_of_a :
  (∀ x, β a x → α x) ∧ 
  (∃ x, α x ∧ ¬β a x) →
  a ≤ -3 ∨ a > 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1331_133143


namespace NUMINAMATH_CALUDE_jelly_cost_l1331_133176

/-- Proof of the cost of jelly given bread, peanut butter, and leftover money -/
theorem jelly_cost (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) 
  (total_money : ℚ) (leftover_money : ℚ) :
  bread_price = 2.25 →
  bread_quantity = 3 →
  peanut_butter_price = 2 →
  total_money = 14 →
  leftover_money = 5.25 →
  leftover_money = total_money - (bread_price * bread_quantity + peanut_butter_price) :=
by
  sorry

#check jelly_cost

end NUMINAMATH_CALUDE_jelly_cost_l1331_133176


namespace NUMINAMATH_CALUDE_product_of_decimals_l1331_133142

theorem product_of_decimals : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1331_133142


namespace NUMINAMATH_CALUDE_integral_identity_l1331_133109

theorem integral_identity : ∫ x in (2 * Real.arctan 2)..(2 * Real.arctan 3), 
  1 / (Real.cos x * (1 - Real.cos x)) = 1/6 + Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_identity_l1331_133109


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l1331_133184

/-- The speed of the man rowing in still water (in kmph) -/
def still_water_speed : ℝ := 36

/-- The ratio of time taken to row upstream vs downstream -/
def upstream_downstream_ratio : ℝ := 3

/-- The rate of the stream (in kmph) -/
def stream_rate : ℝ := 18

theorem stream_rate_calculation :
  let d : ℝ := 1  -- Arbitrary distance
  let downstream_time := d / (still_water_speed + stream_rate)
  let upstream_time := d / (still_water_speed - stream_rate)
  upstream_time = upstream_downstream_ratio * downstream_time →
  stream_rate = 18 := by
sorry

end NUMINAMATH_CALUDE_stream_rate_calculation_l1331_133184


namespace NUMINAMATH_CALUDE_three_quarters_of_48_minus_12_l1331_133163

theorem three_quarters_of_48_minus_12 : (3 / 4 : ℚ) * 48 - 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_of_48_minus_12_l1331_133163


namespace NUMINAMATH_CALUDE_second_month_sale_l1331_133134

def sales_data : List ℕ := [8435, 8855, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem second_month_sale :
  let total_sale := average_sale * num_months
  let known_sales_sum := sales_data.sum
  let second_month_sale := total_sale - known_sales_sum
  second_month_sale = 8927 := by sorry

end NUMINAMATH_CALUDE_second_month_sale_l1331_133134


namespace NUMINAMATH_CALUDE_right_triangle_sum_of_squares_l1331_133178

theorem right_triangle_sum_of_squares (A B C : ℝ × ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 1 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_of_squares_l1331_133178


namespace NUMINAMATH_CALUDE_sequence_properties_l1331_133199

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a : ℕ → ℝ := λ n => 6 * n - 5

theorem sequence_properties :
  (∀ n, S n = 3 * n^2 - 2 * n) →
  (∀ n, a n = 6 * n - 5) ∧
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → a n - a (n-1) = 6) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1331_133199


namespace NUMINAMATH_CALUDE_kennel_problem_l1331_133103

/-- Represents the number of dogs in a kennel that don't like either watermelon or salmon. -/
def dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ) : ℕ :=
  total - (watermelon + salmon - both)

/-- Theorem stating that in a kennel of 60 dogs, where 9 like watermelon, 
    48 like salmon, and 5 like both, 8 dogs don't like either. -/
theorem kennel_problem : dogs_not_liking_either 60 9 48 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_kennel_problem_l1331_133103


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1331_133171

open Real

theorem function_inequality_implies_a_range (a b : ℝ) :
  (∀ x ∈ Set.Ioo (Real.exp 1) ((Real.exp 1) ^ 2),
    ∀ b ≤ 0,
      a * log x - b * x^2 ≥ x) →
  a ≥ (Real.exp 1)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1331_133171


namespace NUMINAMATH_CALUDE_B_and_C_complementary_l1331_133153

-- Define the sample space (outcomes of rolling a fair die)
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event B (up-facing side's number is no more than 3)
def B : Finset Nat := {1, 2, 3}

-- Define event C (up-facing side's number is at least 4)
def C : Finset Nat := {4, 5, 6}

-- Theorem stating that B and C are complementary
theorem B_and_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry


end NUMINAMATH_CALUDE_B_and_C_complementary_l1331_133153
