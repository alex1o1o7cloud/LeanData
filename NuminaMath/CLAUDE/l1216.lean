import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_three_l1216_121641

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Represents a point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1
  on_right_branch : x > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Main theorem: If a line through the right focus intersects the right branch
    at two points forming an equilateral triangle with the left focus,
    then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity_sqrt_three (h : Hyperbola a b)
  (M N : RightBranchPoint h)
  (h_line : ∃ (t : ℝ), (M.x, M.y) = right_focus h + t • ((N.x, N.y) - right_focus h))
  (h_equilateral : is_equilateral_triangle (M.x, M.y) (N.x, N.y) (left_focus h)) :
  eccentricity h = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_three_l1216_121641


namespace NUMINAMATH_CALUDE_intersection_midpoint_theorem_l1216_121622

-- Define the curve
def curve (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 2

-- Define the line
def line (x : ℝ) : ℝ := -2 * x

-- Theorem statement
theorem intersection_midpoint_theorem (s : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ = line x₁ ∧ 
    curve x₂ = line x₂ ∧ 
    (curve x₁ + curve x₂) / 2 = 7 / s) →
  s = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_theorem_l1216_121622


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1216_121608

/-- Given a rectangular plot with specified dimensions and total fencing cost,
    calculate the cost per meter of fencing. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1216_121608


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1216_121606

theorem geometric_sequence_minimum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = a n * r) (h_prod : a 5 * a 6 = 16) :
  (∀ x, a 2 + a 9 ≥ x) → x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1216_121606


namespace NUMINAMATH_CALUDE_chapters_per_day_l1216_121684

/-- Given a book with the specified properties, prove the number of chapters read per day -/
theorem chapters_per_day (total_pages : ℕ) (total_chapters : ℕ) (days_to_read : ℕ) 
  (h1 : total_pages = 193) 
  (h2 : total_chapters = 15) 
  (h3 : days_to_read = 660) :
  (total_chapters : ℚ) / days_to_read = 15 / 660 := by
  sorry

end NUMINAMATH_CALUDE_chapters_per_day_l1216_121684


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1216_121687

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1216_121687


namespace NUMINAMATH_CALUDE_solution_set_range_l1216_121603

theorem solution_set_range (t : ℝ) : 
  let A := {x : ℝ | x^2 - 4*x + t ≤ 0}
  (∃ x ∈ Set.Iic t, x ∈ A) → t ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_range_l1216_121603


namespace NUMINAMATH_CALUDE_negation_of_square_non_negative_l1216_121691

theorem negation_of_square_non_negative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_square_non_negative_l1216_121691


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_of_dimensions_l1216_121601

theorem rectangular_prism_sum_of_dimensions 
  (α β γ : ℝ) 
  (h1 : α * β = 18) 
  (h2 : α * γ = 36) 
  (h3 : β * γ = 72) : 
  α + β + γ = 21 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_of_dimensions_l1216_121601


namespace NUMINAMATH_CALUDE_special_triangle_b_value_l1216_121627

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = 4 ∧ t.b + t.c = 6 ∧ t.b < t.c ∧ Real.cos t.A = 1/2

theorem special_triangle_b_value (t : Triangle) (h : SpecialTriangle t) : t.b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_b_value_l1216_121627


namespace NUMINAMATH_CALUDE_min_original_edges_for_connected_net_l1216_121685

/-- Represents a cube with two opposite corner triangular pyramids removed -/
structure ModifiedCube where
  edge_length : ℝ
  pyramid_height : ℝ

/-- Represents an edge of the cube -/
inductive CubeEdge
  | Original
  | FaceDiagonal
  | SpaceDiagonal

/-- Represents a net of the modified cube -/
structure Net where
  edges : List CubeEdge

/-- Function to calculate the total length of edges in a net -/
def total_edge_length (c : ModifiedCube) (n : Net) : ℝ :=
  sorry

/-- Function to check if a net is connected -/
def is_connected (n : Net) : Prop :=
  sorry

/-- Function to count the number of original cube edges in a net -/
def count_original_edges (n : Net) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of original cube edges
    needed to form a connected net is 7 -/
theorem min_original_edges_for_connected_net (c : ModifiedCube) :
  ∃ (n : Net), is_connected n ∧
    (∀ (m : Net), is_connected m →
      count_original_edges n ≤ count_original_edges m) ∧
    count_original_edges n = 7 :=
  sorry

end NUMINAMATH_CALUDE_min_original_edges_for_connected_net_l1216_121685


namespace NUMINAMATH_CALUDE_uniform_purchase_solution_l1216_121630

theorem uniform_purchase_solution :
  ∃! (x y : ℕ), 5 * x - 3 * y = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_uniform_purchase_solution_l1216_121630


namespace NUMINAMATH_CALUDE_sum_of_ages_l1216_121629

theorem sum_of_ages (age_older : ℕ) (age_diff : ℕ) : age_older = 38 → age_diff = 2 → age_older + (age_older - age_diff) = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1216_121629


namespace NUMINAMATH_CALUDE_shaded_area_9x7_grid_l1216_121689

/-- Represents a grid with 2x2 squares, where alternate squares are split and shaded -/
structure ShadedGrid :=
  (width : ℕ)
  (height : ℕ)
  (square_size : ℕ)

/-- Calculates the area of the shaded region in the grid -/
def shaded_area (grid : ShadedGrid) : ℕ :=
  let horizontal_squares := grid.width / grid.square_size
  let vertical_squares := grid.height / grid.square_size
  let total_squares := horizontal_squares * vertical_squares
  let shaded_triangle_area := (grid.square_size * grid.square_size) / 2
  total_squares * shaded_triangle_area

/-- Theorem: The shaded area in a 9x7 grid with 2x2 squares is 24 square units -/
theorem shaded_area_9x7_grid :
  let grid : ShadedGrid := ⟨9, 7, 2⟩
  shaded_area grid = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_9x7_grid_l1216_121689


namespace NUMINAMATH_CALUDE_g_has_no_zeros_l1216_121623

noncomputable section

open Real

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x - a * log x - x + exp (x - 1)

-- State the theorem
theorem g_has_no_zeros (a : ℝ) (h : 0 ≤ a ∧ a ≤ exp 1) :
  ∀ x > 0, g a x ≠ 0 := by
  sorry

end

end NUMINAMATH_CALUDE_g_has_no_zeros_l1216_121623


namespace NUMINAMATH_CALUDE_zero_full_crates_l1216_121624

/-- Represents the number of berries picked for each type -/
structure BerriesPicked where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Represents the fraction of rotten berries for each type -/
structure RottenFractions where
  blueberries : ℚ
  cranberries : ℚ
  raspberries : ℚ
  gooseberries : ℚ
  strawberries : ℚ

/-- Represents the number of berries required to fill one crate for each type -/
structure CrateCapacity where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Calculates the number of full crates that can be sold -/
def calculateFullCrates (picked : BerriesPicked) (rotten : RottenFractions) (capacity : CrateCapacity) : ℕ :=
  sorry

/-- Theorem stating that the number of full crates that can be sold is 0 -/
theorem zero_full_crates : 
  let picked : BerriesPicked := ⟨30, 20, 10, 15, 25⟩
  let rotten : RottenFractions := ⟨1/3, 1/4, 1/5, 1/6, 1/7⟩
  let capacity : CrateCapacity := ⟨40, 50, 30, 60, 70⟩
  calculateFullCrates picked rotten capacity = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_full_crates_l1216_121624


namespace NUMINAMATH_CALUDE_zero_location_l1216_121654

theorem zero_location (x y : ℝ) 
  (h1 : x^5 < y^8) 
  (h2 : y^8 < y^3) 
  (h3 : y^3 < x^6)
  (h4 : x < 0)
  (h5 : 0 < y)
  (h6 : y < 1) : 
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end NUMINAMATH_CALUDE_zero_location_l1216_121654


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1216_121647

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
  n > 0 → 
  (n - 1) * n * (n + 1) = 8 * (3 * n) → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 405 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1216_121647


namespace NUMINAMATH_CALUDE_women_at_gathering_l1216_121679

/-- Represents a social gathering with men and women dancing -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  man_partners : ℕ
  woman_partners : ℕ

/-- Calculates the total number of dance pairs -/
def total_pairs (g : SocialGathering) : ℕ := g.men * g.man_partners

/-- Theorem: In a social gathering where 15 men attended, each man danced with 4 women,
    and each woman danced with 3 men, the number of women who attended is 20. -/
theorem women_at_gathering (g : SocialGathering) 
  (h1 : g.men = 15)
  (h2 : g.man_partners = 4)
  (h3 : g.woman_partners = 3)
  (h4 : total_pairs g = g.women * g.woman_partners) :
  g.women = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_at_gathering_l1216_121679


namespace NUMINAMATH_CALUDE_rectangle_area_l1216_121607

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * (1 + ratio) = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1216_121607


namespace NUMINAMATH_CALUDE_percentage_of_boys_l1216_121686

theorem percentage_of_boys (total_students : ℕ) (boys : ℕ) (percentage : ℚ) : 
  total_students = 220 →
  242 = (220 / 100) * boys →
  percentage = (boys / total_students) * 100 →
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_boys_l1216_121686


namespace NUMINAMATH_CALUDE_union_equals_A_l1216_121692

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l1216_121692


namespace NUMINAMATH_CALUDE_sequence_sum_l1216_121611

theorem sequence_sum (A B C D E F G H I : ℝ) : 
  D = 8 →
  A + B + C + D = 50 →
  B + C + D + E = 50 →
  C + D + E + F = 50 →
  D + E + F + G = 50 →
  E + F + G + H = 50 →
  F + G + H + I = 50 →
  A + I = 92 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1216_121611


namespace NUMINAMATH_CALUDE_bella_steps_l1216_121613

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- Bella's step length in feet -/
def step_length : ℝ := 2.5

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 5

/-- Calculates the number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := sorry

/-- Theorem stating that Bella takes 704 steps before meeting Ella -/
theorem bella_steps : steps_taken = 704 := by sorry

end NUMINAMATH_CALUDE_bella_steps_l1216_121613


namespace NUMINAMATH_CALUDE_leftover_value_is_230_l1216_121665

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a collection of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat

def roll_size : RollSize := { quarters := 25, dimes := 40 }

def john_coins : Coins := { quarters := 47, dimes := 71 }
def mark_coins : Coins := { quarters := 78, dimes := 132 }

def combine_coins (c1 c2 : Coins) : Coins :=
  { quarters := c1.quarters + c2.quarters,
    dimes := c1.dimes + c2.dimes }

def leftover_coins (c : Coins) (r : RollSize) : Coins :=
  { quarters := c.quarters % r.quarters,
    dimes := c.dimes % r.dimes }

def coin_value (c : Coins) : Rat :=
  (c.quarters : Rat) * (1/4) + (c.dimes : Rat) * (1/10)

theorem leftover_value_is_230 :
  let combined := combine_coins john_coins mark_coins
  let leftover := leftover_coins combined roll_size
  coin_value leftover = 23/10 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_230_l1216_121665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1216_121636

/-- Given a positive arithmetic sequence {a_n} satisfying certain conditions, prove a_10 = 21 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (∀ n, a n > 0) →  -- positive sequence
  a 1 + a 2 + a 3 = 15 →  -- sum condition
  (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13) →  -- geometric sequence condition
  a 10 = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1216_121636


namespace NUMINAMATH_CALUDE_area_difference_square_rectangle_l1216_121695

/-- Given a square and a rectangle with the same perimeter, this theorem proves
    the difference between their areas when specific dimensions are provided. -/
theorem area_difference_square_rectangle (square_perimeter : ℝ) (rect_perimeter : ℝ) (rect_length : ℝ)
  (h1 : square_perimeter = 52)
  (h2 : rect_perimeter = 52)
  (h3 : rect_length = 15) :
  (square_perimeter / 4) ^ 2 - rect_length * ((rect_perimeter / 2) - rect_length) = 4 :=
by sorry


end NUMINAMATH_CALUDE_area_difference_square_rectangle_l1216_121695


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1216_121676

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1216_121676


namespace NUMINAMATH_CALUDE_cistern_leak_time_l1216_121637

/-- Given a cistern with two pipes A and B, this theorem proves the time it takes for pipe B to leak out the full cistern. -/
theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (h1 : fill_time_A = 10) 
  (h2 : fill_time_both = 59.999999999999964) : 
  ∃ (leak_time_B : ℝ), leak_time_B = 12 ∧ 
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l1216_121637


namespace NUMINAMATH_CALUDE_button_difference_l1216_121646

theorem button_difference (sue_buttons kendra_buttons mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 64 →
  mari_buttons - 5 * kendra_buttons = 4 := by
  sorry

end NUMINAMATH_CALUDE_button_difference_l1216_121646


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l1216_121602

theorem equation_two_distinct_roots (k : ℂ) : 
  (∃ (x y : ℂ), x ≠ y ∧ 
    (∀ z : ℂ, z / (z + 3) + z / (z - 1) = k * z ↔ z = x ∨ z = y)) ↔ 
  k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l1216_121602


namespace NUMINAMATH_CALUDE_number_comparison_l1216_121669

theorem number_comparison : 0.6^7 < 0.7^6 ∧ 0.7^6 < 6^0.7 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l1216_121669


namespace NUMINAMATH_CALUDE_range_of_H_l1216_121693

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l1216_121693


namespace NUMINAMATH_CALUDE_river_objects_l1216_121675

/-- The number of objects Bill tossed into the river -/
def bill_objects (ted_sticks ted_rocks bill_sticks bill_rocks : ℕ) : ℕ :=
  bill_sticks + bill_rocks

/-- The problem statement -/
theorem river_objects 
  (ted_sticks ted_rocks : ℕ) 
  (h1 : ted_sticks = 10)
  (h2 : ted_rocks = 10)
  (h3 : ∃ bill_sticks : ℕ, bill_sticks = ted_sticks + 6)
  (h4 : ∃ bill_rocks : ℕ, ted_rocks = 2 * bill_rocks) :
  ∃ bill_sticks bill_rocks : ℕ, 
    bill_objects ted_sticks ted_rocks bill_sticks bill_rocks = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_river_objects_l1216_121675


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l1216_121651

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : ConeConfiguration :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  , sphereRadius := 0  -- To be determined
  }

/-- The theorem statement -/
theorem max_sphere_radius_squared (config : ConeConfiguration) :
  config = problemConfig →
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (s : ℝ), s > 0 → 
      (∃ (c : ConeConfiguration), c.cone1 = config.cone1 ∧ 
                                  c.cone2 = config.cone2 ∧ 
                                  c.intersectionDistance = config.intersectionDistance ∧
                                  c.sphereRadius = s) →
      s^2 ≤ r^2) ∧
    r^2 = 8704 / 29 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l1216_121651


namespace NUMINAMATH_CALUDE_parabola_directrix_l1216_121662

/-- Given a parabola y² = 2px and a point M(1, m) on it, 
    if the distance from M to its focus is 5, 
    then the equation of its directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) :
  m^2 = 2*p  -- Point M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 5^2  -- Distance from M to focus is 5
  → (-p/2 : ℝ) = -4  -- Equation of directrix is x = -4
:= by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1216_121662


namespace NUMINAMATH_CALUDE_odd_function_properties_l1216_121661

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) (h : IsOdd f) :
  (f 0 = 0) ∧
  (∀ x ≥ 0, f x ≥ -1) →
  (∃ x ≥ 0, f x = -1) →
  (∀ x ≤ 0, f x ≤ 1) ∧
  (∃ x ≤ 0, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1216_121661


namespace NUMINAMATH_CALUDE_jaden_final_cars_l1216_121696

/-- The number of toy cars Jaden has after various changes --/
def final_car_count (initial : ℕ) (bought : ℕ) (birthday : ℕ) (sister : ℕ) (friend : ℕ) : ℕ :=
  initial + bought + birthday - sister - friend

/-- Theorem stating that Jaden's final car count is 43 --/
theorem jaden_final_cars : 
  final_car_count 14 28 12 8 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_jaden_final_cars_l1216_121696


namespace NUMINAMATH_CALUDE_cos_A_in_third_quadrant_l1216_121668

theorem cos_A_in_third_quadrant (A : Real) :
  (A > π ∧ A < 3*π/2) →  -- Angle A is in the third quadrant
  (Real.sin A = -1/3) →  -- sin A = -1/3
  (Real.cos A = -2*Real.sqrt 2/3) :=  -- cos A = -2√2/3
by sorry

end NUMINAMATH_CALUDE_cos_A_in_third_quadrant_l1216_121668


namespace NUMINAMATH_CALUDE_birthday_stickers_l1216_121671

theorem birthday_stickers (initial_stickers total_stickers : ℕ) 
  (h1 : initial_stickers = 269)
  (h2 : total_stickers = 423) : 
  total_stickers - initial_stickers = 154 := by
  sorry

end NUMINAMATH_CALUDE_birthday_stickers_l1216_121671


namespace NUMINAMATH_CALUDE_min_stamps_for_37_cents_l1216_121628

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Finds the minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem min_stamps_for_37_cents :
  minCoins 37 [5, 7] = 7 := by sorry

end NUMINAMATH_CALUDE_min_stamps_for_37_cents_l1216_121628


namespace NUMINAMATH_CALUDE_solve_for_y_l1216_121616

theorem solve_for_y (x y : ℝ) (h1 : x^(y+1) = 16) (h2 : x = 8) : y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1216_121616


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1216_121645

/-- A box containing white and red balls -/
structure Box where
  white : Nat
  red : Nat

/-- The number of balls drawn from the box -/
def drawn : Nat := 3

/-- Event A: Exactly one red ball is drawn -/
def eventA (box : Box) : Prop :=
  ∃ (r w : Nat), r = 1 ∧ w = drawn - r ∧ r ≤ box.red ∧ w ≤ box.white

/-- Event B: Exactly one white ball is drawn -/
def eventB (box : Box) : Prop :=
  ∃ (w r : Nat), w = 1 ∧ r = drawn - w ∧ w ≤ box.white ∧ r ≤ box.red

/-- The box in the problem -/
def problemBox : Box := ⟨4, 3⟩

theorem events_mutually_exclusive_but_not_complementary :
  (¬ ∃ (outcome : Nat × Nat), eventA problemBox ∧ eventB problemBox) ∧
  (∃ (outcome : Nat × Nat), ¬(eventA problemBox ∨ eventB problemBox)) :=
by sorry


end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1216_121645


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1216_121610

/-- The repeating decimal 0.4444... expressed as a real number -/
def repeating_decimal : ℚ := 0.4444444444

/-- The theorem states that the repeating decimal 0.4444... is equal to 4/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1216_121610


namespace NUMINAMATH_CALUDE_heavy_wash_water_usage_l1216_121672

/-- Represents the amount of water used for different types of washes -/
structure WashingMachine where
  heavy_wash : ℚ
  regular_wash : ℚ
  light_wash : ℚ

/-- Calculates the total water usage for a given washing machine and set of loads -/
def total_water_usage (wm : WashingMachine) (heavy_loads bleach_loads : ℕ) : ℚ :=
  wm.heavy_wash * heavy_loads +
  wm.regular_wash * 3 +
  wm.light_wash * (1 + bleach_loads)

/-- Theorem stating that the heavy wash uses 20 gallons of water -/
theorem heavy_wash_water_usage :
  ∃ (wm : WashingMachine),
    wm.regular_wash = 10 ∧
    wm.light_wash = 2 ∧
    total_water_usage wm 2 2 = 76 ∧
    wm.heavy_wash = 20 := by
  sorry

end NUMINAMATH_CALUDE_heavy_wash_water_usage_l1216_121672


namespace NUMINAMATH_CALUDE_first_group_size_l1216_121656

/-- Represents the work done by a group of workers -/
structure Work where
  workers : ℕ
  days : ℕ
  hectares : ℕ

/-- The work principle: workers * days is proportional to hectares for any two Work instances -/
axiom work_principle {w1 w2 : Work} : 
  w1.workers * w1.days * w2.hectares = w2.workers * w2.days * w1.hectares

/-- The first group's work -/
def first_group : Work := { workers := 0, days := 24, hectares := 80 }

/-- The second group's work -/
def second_group : Work := { workers := 36, days := 30, hectares := 400 }

/-- The theorem to prove -/
theorem first_group_size : first_group.workers = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1216_121656


namespace NUMINAMATH_CALUDE_correct_sample_size_l1216_121625

/-- Represents the sampling strategy for a company's employee health survey. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  middle_aged_employees : ℕ
  elderly_employees : ℕ
  young_in_sample : ℕ

/-- The sample size for the company's health survey. -/
def sample_size (cs : CompanySampling) : ℕ := 15

theorem correct_sample_size (cs : CompanySampling) 
  (h1 : cs.total_employees = 750)
  (h2 : cs.young_employees = 350)
  (h3 : cs.middle_aged_employees = 250)
  (h4 : cs.elderly_employees = 150)
  (h5 : cs.young_in_sample = 7) :
  sample_size cs = 15 := by
  sorry

#check correct_sample_size

end NUMINAMATH_CALUDE_correct_sample_size_l1216_121625


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_factors_12_l1216_121659

/-- The set of natural-number factors of 12 -/
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}

/-- The sum of the reciprocals of the natural-number factors of 12 -/
def sum_reciprocals : ℚ := (factors_of_12.sum fun n => (1 : ℚ) / n)

/-- Theorem: The sum of the reciprocals of the natural-number factors of 12 is equal to 7/3 -/
theorem sum_reciprocals_of_factors_12 : sum_reciprocals = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_factors_12_l1216_121659


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l1216_121612

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: Given a hyperbola with one asymptote y = 2x and foci x-coordinate 4,
    the other asymptote has equation y = -0.5x + 10 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x => 2 * x) 
    (h2 : h.foci_x = 4) : 
    ∃ asymptote2 : ℝ → ℝ, asymptote2 = fun x => -0.5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l1216_121612


namespace NUMINAMATH_CALUDE_factorization_equality_l1216_121655

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1216_121655


namespace NUMINAMATH_CALUDE_jorge_total_goals_l1216_121620

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l1216_121620


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1216_121604

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1216_121604


namespace NUMINAMATH_CALUDE_ryan_study_difference_l1216_121664

/-- Ryan's daily study hours for different languages -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- The difference in study hours between Chinese and Spanish -/
def chineseSpanishDifference (h : StudyHours) : ℤ :=
  h.chinese - h.spanish

/-- Theorem stating the difference in study hours between Chinese and Spanish -/
theorem ryan_study_difference :
  ∀ (h : StudyHours),
    h.english = 2 → h.chinese = 5 → h.spanish = 4 →
    chineseSpanishDifference h = 1 := by
  sorry

end NUMINAMATH_CALUDE_ryan_study_difference_l1216_121664


namespace NUMINAMATH_CALUDE_meat_cost_per_pound_l1216_121697

/-- The cost of meat per pound given the total cost, rice quantity, rice price, and meat quantity -/
theorem meat_cost_per_pound 
  (total_cost : ℝ)
  (rice_quantity : ℝ)
  (rice_price_per_kg : ℝ)
  (meat_quantity : ℝ)
  (h1 : total_cost = 25)
  (h2 : rice_quantity = 5)
  (h3 : rice_price_per_kg = 2)
  (h4 : meat_quantity = 3)
  : (total_cost - rice_quantity * rice_price_per_kg) / meat_quantity = 5 := by
  sorry

end NUMINAMATH_CALUDE_meat_cost_per_pound_l1216_121697


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l1216_121660

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, x > 0 ∧ y = 2^x}
def N : Set ℝ := {x : ℝ | 2*x - x^2 > 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l1216_121660


namespace NUMINAMATH_CALUDE_certain_number_problem_l1216_121678

theorem certain_number_problem (h : 2994 / 14.5 = 173) : 
  ∃ x : ℝ, x / 1.45 = 17.3 ∧ x = 25.085 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1216_121678


namespace NUMINAMATH_CALUDE_product_one_sum_square_and_products_geq_ten_l1216_121626

theorem product_one_sum_square_and_products_geq_ten 
  (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_square_and_products_geq_ten_l1216_121626


namespace NUMINAMATH_CALUDE_cube_volume_l1216_121648

theorem cube_volume (cube_diagonal : ℝ) (h : cube_diagonal = 6 * Real.sqrt 2) :
  ∃ (volume : ℝ), volume = 216 ∧ volume = (cube_diagonal / Real.sqrt 2) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l1216_121648


namespace NUMINAMATH_CALUDE_limit_exponential_arcsin_ratio_l1216_121644

open Real

theorem limit_exponential_arcsin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((exp (3 * x) - exp (-2 * x)) / (2 * arcsin x - sin x)) - 5| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_exponential_arcsin_ratio_l1216_121644


namespace NUMINAMATH_CALUDE_turtle_count_relationship_lonely_island_turtle_count_l1216_121674

/-- The number of turtles on Happy Island -/
def happy_turtles : ℕ := 60

/-- The number of turtles on Lonely Island -/
def lonely_turtles : ℕ := 25

/-- Theorem stating the relationship between turtles on Happy and Lonely Islands -/
theorem turtle_count_relationship : happy_turtles = 2 * lonely_turtles + 10 := by
  sorry

/-- Theorem proving the number of turtles on Lonely Island -/
theorem lonely_island_turtle_count : lonely_turtles = 25 := by
  sorry

end NUMINAMATH_CALUDE_turtle_count_relationship_lonely_island_turtle_count_l1216_121674


namespace NUMINAMATH_CALUDE_sin_2023pi_over_3_l1216_121605

theorem sin_2023pi_over_3 : Real.sin (2023 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2023pi_over_3_l1216_121605


namespace NUMINAMATH_CALUDE_six_player_tournament_games_l1216_121649

/-- The number of games in a chess tournament where each player plays twice with every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 6 players, where each player plays twice with every other player, the total number of games played is 60. -/
theorem six_player_tournament_games :
  tournament_games 6 = 30 ∧ 2 * tournament_games 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_player_tournament_games_l1216_121649


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1216_121615

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ x^2 - 2*x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1216_121615


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1216_121653

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x - 1, 2⟩
  let b : Vector2D := ⟨2, 1⟩
  parallel a b → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1216_121653


namespace NUMINAMATH_CALUDE_arithmetic_properties_l1216_121640

variable (a : ℤ)

theorem arithmetic_properties :
  (216 + 35 + 84 = 35 + (216 + 84)) ∧
  (298 - 35 - 165 = 298 - (35 + 165)) ∧
  (400 / 25 / 4 = 400 / (25 * 4)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_properties_l1216_121640


namespace NUMINAMATH_CALUDE_walter_gets_49_bananas_l1216_121614

/-- Calculates the number of bananas Walter gets when sharing with Jefferson -/
def walters_bananas (jeffersons_bananas : ℕ) : ℕ :=
  let walters_fewer := jeffersons_bananas / 4
  let walters_original := jeffersons_bananas - walters_fewer
  let total_bananas := jeffersons_bananas + walters_original
  total_bananas / 2

/-- Proves that Walter gets 49 bananas when sharing with Jefferson -/
theorem walter_gets_49_bananas :
  walters_bananas 56 = 49 := by
  sorry

end NUMINAMATH_CALUDE_walter_gets_49_bananas_l1216_121614


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l1216_121680

/-- Represents the number of hits for each type of hit in baseball --/
structure HitCounts where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles given the hit counts --/
def percentageSingles (hits : HitCounts) : ℚ :=
  let singles := hits.total - (hits.homeRuns + hits.triples + hits.doubles)
  (singles : ℚ) / hits.total * 100

/-- Carlos's hit counts for the baseball season --/
def carlosHits : HitCounts :=
  { total := 50
  , homeRuns := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of Carlos's hits that were singles is 74% --/
theorem carlos_singles_percentage :
  percentageSingles carlosHits = 74 := by
  sorry


end NUMINAMATH_CALUDE_carlos_singles_percentage_l1216_121680


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1216_121634

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1216_121634


namespace NUMINAMATH_CALUDE_fraction_equality_l1216_121699

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1216_121699


namespace NUMINAMATH_CALUDE_longest_side_range_l1216_121681

/-- An obtuse triangle with sides a, b, and c, where c is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  c_longest : c ≥ max a b
  obtuse : c^2 > a^2 + b^2

/-- The theorem stating the range of the longest side in a specific obtuse triangle -/
theorem longest_side_range (t : ObtuseTriangle) (ha : t.a = 1) (hb : t.b = 2) :
  Real.sqrt 5 < t.c ∧ t.c < 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_range_l1216_121681


namespace NUMINAMATH_CALUDE_circle_sector_radius_l1216_121642

theorem circle_sector_radius 
  (area : ℝ) 
  (arc_length : ℝ) 
  (h1 : area = 8.75) 
  (h2 : arc_length = 3.5) : 
  ∃ (radius : ℝ), radius = 5 ∧ area = (1/2) * radius * arc_length :=
by
  sorry

end NUMINAMATH_CALUDE_circle_sector_radius_l1216_121642


namespace NUMINAMATH_CALUDE_sector_area_l1216_121657

theorem sector_area (circle_area : ℝ) (sector_angle : ℝ) : 
  circle_area = 9 * Real.pi →
  sector_angle = 120 →
  (sector_angle / 360) * circle_area = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1216_121657


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1216_121621

/-- The sum of the infinite series Σ(k^2 / 3^k) from k=1 to infinity is equal to 4 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = 4 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1216_121621


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1216_121652

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z + 2)) + Real.sqrt (y / (x + z + 2)) + Real.sqrt (z / (x + y + 2)) >
  (4 / Real.sqrt 3) * Real.cos (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1216_121652


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1216_121609

theorem sqrt_equation_solution (a b : ℕ+) (h : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 14 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b ↔ 
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1216_121609


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_is_exists_leq_l1216_121639

theorem negation_of_forall_gt_is_exists_leq :
  (¬ ∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_is_exists_leq_l1216_121639


namespace NUMINAMATH_CALUDE_gcd_of_256_162_450_l1216_121673

theorem gcd_of_256_162_450 : Nat.gcd 256 (Nat.gcd 162 450) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_162_450_l1216_121673


namespace NUMINAMATH_CALUDE_tree_cutting_theorem_l1216_121635

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def solo_days : ℕ := 2

/-- The number of days James works with his brothers -/
def team_days : ℕ := 3

/-- The number of brothers helping James -/
def num_brothers : ℕ := 2

/-- The percentage of trees each brother cuts compared to James -/
def brother_efficiency : ℚ := 4/5

/-- The total number of trees cut down -/
def total_trees : ℕ := 196

theorem tree_cutting_theorem :
  james_trees_per_day * solo_days + 
  (james_trees_per_day + (james_trees_per_day * brother_efficiency).floor * num_brothers) * team_days = 
  total_trees :=
sorry

end NUMINAMATH_CALUDE_tree_cutting_theorem_l1216_121635


namespace NUMINAMATH_CALUDE_next_common_term_l1216_121617

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem next_common_term
  (a₁ b₁ d₁ d₂ : ℤ)
  (h₁ : a₁ = 3)
  (h₂ : b₁ = 16)
  (h₃ : d₁ = 17)
  (h₄ : d₂ = 11)
  (h₅ : ∃ (n m : ℕ), arithmetic_sequence a₁ d₁ n = 71 ∧ arithmetic_sequence b₁ d₂ m = 71)
  : ∃ (k l : ℕ), 
    arithmetic_sequence a₁ d₁ k = arithmetic_sequence b₁ d₂ l ∧
    arithmetic_sequence a₁ d₁ k > 71 ∧
    arithmetic_sequence a₁ d₁ k = 258 :=
sorry

end NUMINAMATH_CALUDE_next_common_term_l1216_121617


namespace NUMINAMATH_CALUDE_second_number_in_sequence_l1216_121600

theorem second_number_in_sequence (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_in_sequence_l1216_121600


namespace NUMINAMATH_CALUDE_min_disks_is_ten_l1216_121666

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Rat
  files_06MB : Nat
  files_10MB : Nat
  files_03MB : Nat

/-- Calculates the minimum number of disks needed for the given storage problem. -/
def min_disks_needed (problem : StorageProblem) : Nat :=
  sorry

/-- Theorem stating that the minimum number of disks needed is 10 for the given problem. -/
theorem min_disks_is_ten (problem : StorageProblem) 
  (h1 : problem.total_files = 25)
  (h2 : problem.disk_capacity = 2)
  (h3 : problem.files_06MB = 5)
  (h4 : problem.files_10MB = 10)
  (h5 : problem.files_03MB = 10) :
  min_disks_needed problem = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_is_ten_l1216_121666


namespace NUMINAMATH_CALUDE_store_inventory_sale_l1216_121619

theorem store_inventory_sale (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (debt : ℝ) (leftover : ℝ) :
  total_items = 2000 →
  original_price = 50 →
  discount_percent = 80 →
  debt = 15000 →
  leftover = 3000 →
  (((debt + leftover) / (original_price * (1 - discount_percent / 100))) / total_items) * 100 = 90 := by
  sorry


end NUMINAMATH_CALUDE_store_inventory_sale_l1216_121619


namespace NUMINAMATH_CALUDE_circle_radius_through_three_points_l1216_121683

/-- The radius of the circle passing through three given points is 5 -/
theorem circle_radius_through_three_points : ∃ (center : ℝ × ℝ) (r : ℝ),
  r = 5 ∧
  (center.1 - 1)^2 + (center.2 - 3)^2 = r^2 ∧
  (center.1 - 4)^2 + (center.2 - 2)^2 = r^2 ∧
  (center.1 - 1)^2 + (center.2 - (-7))^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_through_three_points_l1216_121683


namespace NUMINAMATH_CALUDE_furniture_making_l1216_121688

theorem furniture_making (total_wood pieces_per_table pieces_per_chair chairs_made : ℕ) 
  (h1 : total_wood = 672)
  (h2 : pieces_per_table = 12)
  (h3 : pieces_per_chair = 8)
  (h4 : chairs_made = 48) :
  (total_wood - chairs_made * pieces_per_chair) / pieces_per_table = 24 := by
  sorry

end NUMINAMATH_CALUDE_furniture_making_l1216_121688


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1216_121631

theorem rationalize_denominator : (45 : ℝ) / Real.sqrt 45 = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1216_121631


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l1216_121690

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 5 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -16 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l1216_121690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1216_121677

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h4 : seq.S 4 = -4)
    (h6 : seq.S 6 = 6) :
    seq.S 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1216_121677


namespace NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_rounded_l1216_121633

/-- The minimum distance a runner must travel in a race with specific conditions -/
theorem minimum_race_distance : ℝ :=
  let wall_distance : ℝ := 1500
  let a_to_first_wall : ℝ := 400
  let b_to_second_wall : ℝ := 600
  let total_vertical_distance : ℝ := a_to_first_wall + wall_distance + b_to_second_wall
  let minimum_distance : ℝ := (wall_distance ^ 2 + total_vertical_distance ^ 2).sqrt
  ⌊minimum_distance + 0.5⌋

/-- The minimum distance rounded to the nearest meter is 2915 -/
theorem minimum_race_distance_rounded : 
  ⌊minimum_race_distance + 0.5⌋ = 2915 := by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_rounded_l1216_121633


namespace NUMINAMATH_CALUDE_lakeisha_lawn_size_l1216_121698

/-- The size of each lawn LaKeisha has already mowed -/
def lawn_size : ℝ := sorry

/-- LaKeisha's charge per square foot -/
def charge_per_sqft : ℝ := 0.10

/-- Cost of the book set -/
def book_cost : ℝ := 150

/-- Number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- Additional square feet to mow -/
def additional_sqft : ℝ := 600

theorem lakeisha_lawn_size :
  lawn_size = 300 ∧
  charge_per_sqft * (lawns_mowed * lawn_size + additional_sqft) = book_cost :=
sorry

end NUMINAMATH_CALUDE_lakeisha_lawn_size_l1216_121698


namespace NUMINAMATH_CALUDE_expansion_contains_constant_term_l1216_121667

/-- The expansion of (√x - 2/x)^n contains a constant term for some positive integer n -/
theorem expansion_contains_constant_term : ∃ (n : ℕ+), 
  ∃ (r : ℕ), n = 3 * r := by
  sorry

end NUMINAMATH_CALUDE_expansion_contains_constant_term_l1216_121667


namespace NUMINAMATH_CALUDE_star_polygon_points_l1216_121638

/-- A regular star polygon with n points, where each point has two types of angles -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : 0 < n

/-- The sum of all exterior angles in a polygon is 360° -/
axiom sum_of_exterior_angles : ∀ (p : StarPolygon), p.n * (p.angle_B - p.angle_A) = 360

/-- The number of points in the star polygon is 24 -/
theorem star_polygon_points (p : StarPolygon) : p.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_points_l1216_121638


namespace NUMINAMATH_CALUDE_circle_theorem_l1216_121663

/-- Represents the type of person in the circle -/
inductive PersonType
| Knight
| Liar

/-- Checks if a given number is a valid k value -/
def is_valid_k (k : ℕ) : Prop :=
  k < 100 ∧ ∃ (m : ℕ), 100 = m * (k + 1)

/-- The set of all valid k values -/
def valid_k_set : Set ℕ :=
  {1, 3, 4, 9, 19, 24, 49, 99}

/-- A circle of 100 people -/
def Circle := Fin 100 → PersonType

theorem circle_theorem (circle : Circle) :
  ∃ (k : ℕ), is_valid_k k ∧
  (∀ (i : Fin 100),
    (circle i = PersonType.Knight →
      ∀ (j : Fin 100), j < k → circle ((i + j + 1) % 100) = PersonType.Liar) ∧
    (circle i = PersonType.Liar →
      ∃ (j : Fin 100), j < k ∧ circle ((i + j + 1) % 100) = PersonType.Knight)) ↔
  ∃ (k : ℕ), k ∈ valid_k_set :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l1216_121663


namespace NUMINAMATH_CALUDE_average_marathons_rounded_l1216_121658

def marathons : List ℕ := [1, 2, 3, 4, 5]
def members : List ℕ := [6, 5, 3, 2, 3]

def total_marathons : ℕ := (List.zip marathons members).map (λ (m, n) => m * n) |>.sum
def total_members : ℕ := members.sum

def average : ℚ := total_marathons / total_members

theorem average_marathons_rounded :
  (average + 1/2).floor = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_marathons_rounded_l1216_121658


namespace NUMINAMATH_CALUDE_reflection_line_property_l1216_121643

/-- A line that reflects a point (x₁, y₁) to (x₂, y₂) -/
structure ReflectionLine where
  m : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  reflects : ((x₂ - x₁) * m + y₁ + y₂) / 2 = m * ((x₂ + x₁) / 2) + b

/-- The theorem stating that for a line y = mx + b that reflects (1, -4) to (7, 2), 3m + 2b = 3 -/
theorem reflection_line_property (line : ReflectionLine) 
    (h1 : line.x₁ = 1) (h2 : line.y₁ = -4) (h3 : line.x₂ = 7) (h4 : line.y₂ = 2) : 
    3 * line.m + 2 * line.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_property_l1216_121643


namespace NUMINAMATH_CALUDE_parkway_fifth_grade_count_l1216_121670

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def num_boys : ℕ := 312

/-- The number of students playing soccer -/
def num_soccer : ℕ := 250

/-- The proportion of boys among students playing soccer -/
def prop_boys_soccer : ℚ := 78 / 100

/-- The number of girls not playing soccer -/
def num_girls_not_soccer : ℕ := 53

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end NUMINAMATH_CALUDE_parkway_fifth_grade_count_l1216_121670


namespace NUMINAMATH_CALUDE_train_speed_l1216_121694

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 45) :
  (train_length + bridge_length) / crossing_time = 13.33333333333333 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1216_121694


namespace NUMINAMATH_CALUDE_convex_quad_probability_l1216_121632

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quad_probability_l1216_121632


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l1216_121682

theorem polar_coordinates_of_point (x y : ℝ) (r θ : ℝ) :
  x = -Real.sqrt 3 ∧ y = -1 →
  r = 2 ∧ θ = 7 * Real.pi / 6 →
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l1216_121682


namespace NUMINAMATH_CALUDE_stating_sum_of_nth_group_is_cube_l1216_121650

/-- 
Given a grouping of consecutive odd numbers as follows:
1; (3,5); (7,9,11); (13, 15, 17, 19); ...
This function represents the sum of the numbers in the n-th group.
-/
def sumOfNthGroup (n : ℕ) : ℕ :=
  n^3

/-- 
Theorem stating that the sum of the numbers in the n-th group
of the described sequence is equal to n^3.
-/
theorem sum_of_nth_group_is_cube (n : ℕ) :
  sumOfNthGroup n = n^3 := by
  sorry

end NUMINAMATH_CALUDE_stating_sum_of_nth_group_is_cube_l1216_121650


namespace NUMINAMATH_CALUDE_zoo_rabbits_l1216_121618

/-- Given a zoo with parrots and rabbits, where the ratio of parrots to rabbits
    is 3:4 and there are 21 parrots, prove that there are 28 rabbits. -/
theorem zoo_rabbits (parrots : ℕ) (rabbits : ℕ) : 
  parrots = 21 → 3 * rabbits = 4 * parrots → rabbits = 28 := by
  sorry

end NUMINAMATH_CALUDE_zoo_rabbits_l1216_121618
