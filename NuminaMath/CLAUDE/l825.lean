import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_total_l825_82507

theorem stratified_sampling_total (senior junior freshman sampled_freshman : ℕ) 
  (h1 : senior = 1000)
  (h2 : junior = 1200)
  (h3 : freshman = 1500)
  (h4 : sampled_freshman = 75) :
  (senior + junior + freshman) * sampled_freshman / freshman = 185 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_l825_82507


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l825_82518

def initial_trees : ℕ := 3
def dead_trees : ℕ := 13

theorem typhoon_tree_difference : dead_trees - (initial_trees - dead_trees) = 13 := by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l825_82518


namespace NUMINAMATH_CALUDE_count_students_in_line_l825_82599

/-- The number of students standing in a line, given that Yoojeong is at the back and there are some people in front of her. -/
def studentsInLine (peopleInFront : ℕ) : ℕ :=
  peopleInFront + 1

/-- Theorem stating that the number of students in the line is equal to the number of people in front of Yoojeong plus one. -/
theorem count_students_in_line (peopleInFront : ℕ) :
  studentsInLine peopleInFront = peopleInFront + 1 := by
  sorry

end NUMINAMATH_CALUDE_count_students_in_line_l825_82599


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l825_82567

/-- The nth positive odd multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive odd multiple of 5 is odd -/
theorem fifteenth_odd_multiple_of_5_is_odd : Odd (oddMultipleOf5 15) := by
  sorry

/-- The 15th positive odd multiple of 5 is a multiple of 5 -/
theorem fifteenth_odd_multiple_of_5_is_multiple_of_5 : ∃ k : ℕ, oddMultipleOf5 15 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l825_82567


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l825_82503

/-- Given a point with rectangular coordinates (-3, -4, 5) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (3, 4, 5). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = -3 →
  ρ * Real.sin φ * Real.sin θ = -4 →
  ρ * Real.cos φ = 5 →
  ρ * Real.sin (-φ) * Real.cos (θ + π) = 3 ∧
  ρ * Real.sin (-φ) * Real.sin (θ + π) = 4 ∧
  ρ * Real.cos (-φ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l825_82503


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l825_82561

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F are on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l825_82561


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l825_82558

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture : Mixture := sorry

/-- The final mixture after adding water -/
def final_mixture : Mixture := sorry

theorem initial_ratio_is_four_to_one :
  -- Initial mixture volume is 45 litres
  initial_mixture.milk + initial_mixture.water = 45 →
  -- 9 litres of water added
  final_mixture.water = initial_mixture.water + 9 →
  -- Final ratio of milk to water is 2:1
  final_mixture.milk / final_mixture.water = 2 →
  -- Prove that the initial ratio of milk to water was 4:1
  initial_mixture.milk / initial_mixture.water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l825_82558


namespace NUMINAMATH_CALUDE_shorter_base_length_l825_82583

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The length of the line segment joining the midpoints of the diagonals
    is half the difference between the lengths of the bases -/
axiom trapezoid_midpoint_segment (t : Trapezoid) :
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.longer_base = 115)
  (h2 : t.midpoint_segment = 5) :
  t.shorter_base = 105 := by
  sorry

#check shorter_base_length

end NUMINAMATH_CALUDE_shorter_base_length_l825_82583


namespace NUMINAMATH_CALUDE_solve_wardrobe_problem_l825_82510

def wardrobe_problem (socks shoes tshirts new_socks : ℕ) : Prop :=
  ∃ pants : ℕ,
    let current_items := 2 * socks + 2 * shoes + tshirts + pants
    current_items + 2 * new_socks = 2 * current_items ∧
    pants = 5

theorem solve_wardrobe_problem :
  wardrobe_problem 20 5 10 35 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_wardrobe_problem_l825_82510


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l825_82538

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| > 1} = Set.Iio (-4) ∪ Set.Ioi (-2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l825_82538


namespace NUMINAMATH_CALUDE_lisa_borrowed_chairs_l825_82571

/-- The number of chairs Lisa borrowed from Rodrigo's classroom -/
def chairs_borrowed (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ) : ℕ :=
  total_chairs_before - total_chairs_after

/-- Theorem stating the number of chairs Lisa borrowed -/
theorem lisa_borrowed_chairs : 
  ∀ (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ),
  red_chairs = 4 →
  yellow_chairs = 2 * red_chairs →
  blue_chairs = yellow_chairs - 2 →
  total_chairs_before = red_chairs + yellow_chairs + blue_chairs →
  total_chairs_after = 15 →
  chairs_borrowed red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after = 3 :=
by sorry

end NUMINAMATH_CALUDE_lisa_borrowed_chairs_l825_82571


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l825_82578

theorem square_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 2) :
  x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l825_82578


namespace NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l825_82512

theorem cylinder_surface_area_ratio (a : ℝ) (h : a > 0) :
  let r := a / (2 * Real.pi)
  let side_area := a^2
  let base_area := Real.pi * r^2
  let total_area := 2 * base_area + side_area
  (total_area / side_area) = (1 + 2 * Real.pi) / (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l825_82512


namespace NUMINAMATH_CALUDE_pizza_distribution_l825_82563

/-- Given 12 coworkers sharing 3 pizzas with 8 slices each, prove that each person gets 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : coworkers = 12)
  (h2 : pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l825_82563


namespace NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l825_82536

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  longerLeg : ℝ
  shorterLeg : ℝ
  hyp_longer_ratio : longerLeg = hypotenuse * Real.sqrt 3 / 2
  hyp_shorter_ratio : shorterLeg = hypotenuse / 2

/-- Represents a sequence of four 30-60-90 triangles -/
def TriangleSequence (t₁ t₂ t₃ t₄ : Triangle30_60_90) : Prop :=
  t₁.hypotenuse = 8 ∧
  t₂.hypotenuse = t₁.longerLeg ∧
  t₃.hypotenuse = t₂.longerLeg ∧
  t₄.hypotenuse = t₃.longerLeg

theorem smallest_triangle_longer_leg 
  (t₁ t₂ t₃ t₄ : Triangle30_60_90)
  (h : TriangleSequence t₁ t₂ t₃ t₄) :
  t₄.longerLeg = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l825_82536


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l825_82521

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 65 * ((n1 : ℚ) + (n2 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l825_82521


namespace NUMINAMATH_CALUDE_d_value_l825_82502

-- Define the function f(x) = x⋅(4x-3)
def f (x : ℝ) : ℝ := x * (4 * x - 3)

-- Define the interval (-9/4, 3/2)
def interval : Set ℝ := { x | -9/4 < x ∧ x < 3/2 }

-- State the theorem
theorem d_value : 
  ∃ d : ℝ, (∀ x : ℝ, f x < d ↔ x ∈ interval) → d = 27/2 := by sorry

end NUMINAMATH_CALUDE_d_value_l825_82502


namespace NUMINAMATH_CALUDE_oblique_drawing_parallelogram_oblique_drawing_other_shapes_l825_82573

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle
  | Parallelogram
  | Square
  | Rhombus

/-- Represents the result of applying the oblique drawing method to a shape -/
def obliqueDrawing (s : Shape) : Shape :=
  match s with
  | Shape.Parallelogram => Shape.Parallelogram
  | _ => Shape.Parallelogram  -- Simplified for this problem

/-- Theorem stating that the oblique drawing of a parallelogram is always a parallelogram -/
theorem oblique_drawing_parallelogram :
  ∀ s : Shape, s = Shape.Parallelogram → obliqueDrawing s = Shape.Parallelogram :=
by sorry

/-- Theorem stating that the oblique drawing of non-parallelogram shapes may not preserve the original shape -/
theorem oblique_drawing_other_shapes :
  ∃ s : Shape, s ≠ Shape.Parallelogram ∧ obliqueDrawing s ≠ s :=
by sorry

end NUMINAMATH_CALUDE_oblique_drawing_parallelogram_oblique_drawing_other_shapes_l825_82573


namespace NUMINAMATH_CALUDE_problem_statement_l825_82539

theorem problem_statement (x y : ℝ) (h : x + 2 * y - 3 = 0) : 2 * x * 4 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l825_82539


namespace NUMINAMATH_CALUDE_x_greater_than_y_l825_82528

theorem x_greater_than_y (a b x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : x = a + 1/a) 
  (h4 : y = b + 1/b) : 
  x > y := by
sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l825_82528


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l825_82516

/-- An isosceles triangle with two sides measuring 4 and 7 has a perimeter of either 15 or 18. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∧ b = 4 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 15 ∨ a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l825_82516


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l825_82598

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 90 := by sorry

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l825_82598


namespace NUMINAMATH_CALUDE_select_five_from_eight_l825_82513

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l825_82513


namespace NUMINAMATH_CALUDE_max_x_value_l825_82594

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 10) :
  x ≤ 2 ∧ ∃ y z, x = 2 ∧ y + z = 4 ∧ x + y + z = 6 ∧ x*y + x*z + y*z = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l825_82594


namespace NUMINAMATH_CALUDE_part_a_part_b_part_c_l825_82523

-- Define what it means for a number to be TOP
def is_TOP (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000) * (n % 10) = ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10)

-- Part a
theorem part_a : is_TOP 23498 := by sorry

-- Part b
theorem part_b : ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2) ∧ 
  (∀ n, is_TOP n ∧ n / 10000 = 1 ∧ n % 10 = 2 → n ∈ s) ∧
  Finset.card s = 6 := by sorry

-- Part c
theorem part_c : ∃ (s : Finset ℕ),
  (∀ n ∈ s, is_TOP n ∧ n / 10000 = 9) ∧
  (∀ n, is_TOP n ∧ n / 10000 = 9 → n ∈ s) ∧
  Finset.card s = 112 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_c_l825_82523


namespace NUMINAMATH_CALUDE_gcd_143_117_l825_82552

theorem gcd_143_117 : Nat.gcd 143 117 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_117_l825_82552


namespace NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l825_82548

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 > 0 ∧ s2 > 0) 
  (h_area_ratio : s1^2 / s2^2 = 49 / 64) : 
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_from_area_ratio_l825_82548


namespace NUMINAMATH_CALUDE_second_player_wins_l825_82532

/-- Represents the game board -/
def Board := Fin 3 → Fin 101 → Bool

/-- The initial state of the board with the central cell crossed out -/
def initialBoard : Board :=
  fun i j => i = 1 && j = 50

/-- A move in the game -/
structure Move where
  start_row : Fin 3
  start_col : Fin 101
  length : Fin 4
  direction : Bool  -- true for down-right, false for down-left

/-- Checks if a move is valid on the given board -/
def isValidMove (b : Board) (m : Move) : Bool :=
  sorry

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (b : Board) : Bool :=
  sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Board → Move),
    ∀ (game : List Move),
      game.length % 2 = 0 →
      let final_board := game.foldl applyMove initialBoard
      isGameOver final_board →
      ∃ (m : Move), isValidMove final_board m :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l825_82532


namespace NUMINAMATH_CALUDE_line_parameterization_l825_82501

/-- Given a line y = 3x + 2 parameterized as (x, y) = (5, r) + t(m, 6),
    prove that r = 17 and m = 2 -/
theorem line_parameterization (r m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, 
    (x = 5 + t * m ∧ y = r + t * 6) → y = 3 * x + 2) →
  r = 17 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l825_82501


namespace NUMINAMATH_CALUDE_video_game_cost_l825_82533

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def amount_left : ℕ := 41

def total_savings : ℕ := september_savings + october_savings + november_savings

theorem video_game_cost : total_savings - amount_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l825_82533


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l825_82547

/-- Prove that if a point (-4, a) lies on the terminal side of an angle measuring 600°, then a = -4√3. -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l825_82547


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l825_82569

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  /-- Angle between faces PQR and QRS -/
  angle : ℝ
  /-- Area of face PQR -/
  area_PQR : ℝ
  /-- Area of face QRS -/
  area_QRS : ℝ
  /-- Length of edge QR -/
  length_QR : ℝ

/-- The volume of a tetrahedron with the given properties -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.angle = 45 * π / 180)
  (h2 : t.area_PQR = 150)
  (h3 : t.area_QRS = 50)
  (h4 : t.length_QR = 10) : 
  volume t = 250 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l825_82569


namespace NUMINAMATH_CALUDE_min_value_z_l825_82549

theorem min_value_z (x y : ℝ) :
  let z := 3 * x^2 + 4 * y^2 + 8 * x - 6 * y + 30
  ∀ a b : ℝ, z ≥ 3 * a^2 + 4 * b^2 + 8 * a - 6 * b + 30 → z ≥ 24.1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l825_82549


namespace NUMINAMATH_CALUDE_circle_condition_l825_82511

theorem circle_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - k*x + 2*y + k^2 - 2 = 0 ↔ ∃ r > 0, ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) ↔
  -2 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l825_82511


namespace NUMINAMATH_CALUDE_box_volume_theorem_l825_82520

/-- Represents the possible volumes of the box -/
def PossibleVolumes : Set ℕ := {80, 100, 120, 150, 200}

/-- Theorem: Given a rectangular box with integer side lengths in the ratio 1:2:5,
    the only possible volume from the set of possible volumes is 80 -/
theorem box_volume_theorem (x : ℕ) (hx : x > 0) :
  (∃ (v : ℕ), v ∈ PossibleVolumes ∧ v = x * (2 * x) * (5 * x)) ↔ (x * (2 * x) * (5 * x) = 80) :=
by sorry

end NUMINAMATH_CALUDE_box_volume_theorem_l825_82520


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l825_82596

theorem linear_equation_equivalence (x y : ℚ) :
  2 * x + 3 * y - 4 = 0 →
  (y = (4 - 2 * x) / 3 ∧ x = (4 - 3 * y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l825_82596


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l825_82515

/-- The probability of no two adjacent people rolling the same number on an 8-sided die
    when 5 people sit around a circular table. -/
theorem probability_no_adjacent_same_roll : ℚ :=
  let num_people : ℕ := 5
  let die_sides : ℕ := 8
  let prob_same : ℚ := 1 / die_sides
  let prob_diff : ℚ := 1 - prob_same
  let prob_case1 : ℚ := prob_same * prob_diff^2 * (die_sides - 2) / die_sides
  let prob_case2 : ℚ := prob_diff^3 * (die_sides - 2) / die_sides
  302 / 512

/-- Proof of the theorem -/
theorem probability_no_adjacent_same_roll_proof :
  probability_no_adjacent_same_roll = 302 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l825_82515


namespace NUMINAMATH_CALUDE_expression_evaluation_l825_82534

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  (3 * x + 2 * y) - (3 * x - 2 * y) = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l825_82534


namespace NUMINAMATH_CALUDE_sin_360_degrees_l825_82530

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_l825_82530


namespace NUMINAMATH_CALUDE_ancient_market_prices_l825_82565

/-- The cost of animals in an ancient market --/
theorem ancient_market_prices :
  -- Define the costs of animals
  ∀ (camel_cost horse_cost ox_cost elephant_cost : ℚ),
  -- Conditions from the problem
  (10 * camel_cost = 24 * horse_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (6 * ox_cost = 4 * elephant_cost) →
  (10 * elephant_cost = 110000) →
  -- Conclusion: the cost of one camel is 4400
  camel_cost = 4400 := by
  sorry

end NUMINAMATH_CALUDE_ancient_market_prices_l825_82565


namespace NUMINAMATH_CALUDE_bicycle_price_last_year_l825_82517

theorem bicycle_price_last_year (total_sales_last_year : ℝ) (price_decrease : ℝ) 
  (sales_quantity : ℝ) (decrease_percentage : ℝ) :
  total_sales_last_year = 80000 →
  price_decrease = 200 →
  decrease_percentage = 0.1 →
  total_sales_last_year * (1 - decrease_percentage) = 
    sales_quantity * (total_sales_last_year / sales_quantity - price_decrease) →
  total_sales_last_year / sales_quantity = 2000 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_last_year_l825_82517


namespace NUMINAMATH_CALUDE_max_log_sum_l825_82564

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 5 → Real.log a + Real.log b ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l825_82564


namespace NUMINAMATH_CALUDE_g_zero_values_l825_82559

theorem g_zero_values (g : ℝ → ℝ) (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) :
  g 0 = 0 ∨ g 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_zero_values_l825_82559


namespace NUMINAMATH_CALUDE_final_rope_length_l825_82584

/-- Given a rope of length 100 feet, prove that after the described cutting process,
    the length of the final piece is 100 / (3 * 2 * 3 * 4 * 5 * 6) feet. -/
theorem final_rope_length (initial_length : ℝ := 100) : 
  initial_length / (3 * 2 * 3 * 4 * 5 * 6) = initial_length / 360 :=
by sorry

end NUMINAMATH_CALUDE_final_rope_length_l825_82584


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l825_82540

theorem sine_cosine_identity : Real.sin (20 * π / 180) * Real.cos (110 * π / 180) + Real.cos (160 * π / 180) * Real.sin (70 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l825_82540


namespace NUMINAMATH_CALUDE_total_students_correct_l825_82568

/-- The total number of students in Misha's grade -/
def total_students : ℕ := 69

/-- Misha's position from the top -/
def position_from_top : ℕ := 30

/-- Misha's position from the bottom -/
def position_from_bottom : ℕ := 40

/-- Theorem stating that the total number of students is correct given Misha's positions -/
theorem total_students_correct :
  total_students = position_from_top + position_from_bottom - 1 :=
by sorry

end NUMINAMATH_CALUDE_total_students_correct_l825_82568


namespace NUMINAMATH_CALUDE_systematic_sampling_probabilities_l825_82506

theorem systematic_sampling_probabilities 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (excluded_count : ℕ) 
  (h1 : total_students = 1005)
  (h2 : sample_size = 50)
  (h3 : excluded_count = 5) :
  (excluded_count : ℚ) / total_students = 5 / 1005 ∧
  (sample_size : ℚ) / total_students = 50 / 1005 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probabilities_l825_82506


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l825_82529

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ 
  ¬(∀ x : ℝ, x^2 > 9 → x > 3) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l825_82529


namespace NUMINAMATH_CALUDE_triangle_third_side_l825_82508

theorem triangle_third_side (a b c : ℝ) : 
  a = 1 → b = 5 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l825_82508


namespace NUMINAMATH_CALUDE_min_value_theorem_l825_82524

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  ∃ (min : ℝ), min = 5 ∧ ∀ y, y > 1 → x + 4 / (x - 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l825_82524


namespace NUMINAMATH_CALUDE_x_approximates_one_l825_82527

/-- The polynomial function P(x) = x^4 - 4x^3 + 4x^2 + 4 -/
def P (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x^2 + 4

/-- A small positive real number representing the tolerance for approximation -/
def ε : ℝ := 0.000000000000001

theorem x_approximates_one :
  ∃ x : ℝ, abs (P x - 4.999999999999999) < ε ∧ abs (x - 1) < ε :=
sorry

end NUMINAMATH_CALUDE_x_approximates_one_l825_82527


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l825_82514

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f (1 / x + 2) x) ∧
  (m * 1 - f 1 + b = 0) ∧
  (m = 3) ∧ (b = -2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l825_82514


namespace NUMINAMATH_CALUDE_razorback_total_profit_l825_82526

/-- The Razorback shop's sales during the Arkansas and Texas Tech game -/
def razorback_sales : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun tshirt_profit jersey_profit hat_profit keychain_profit
      tshirts_sold jerseys_sold hats_sold keychains_sold =>
    tshirt_profit * tshirts_sold +
    jersey_profit * jerseys_sold +
    hat_profit * hats_sold +
    keychain_profit * keychains_sold

theorem razorback_total_profit :
  razorback_sales 62 99 45 25 183 31 142 215 = 26180 := by
  sorry

end NUMINAMATH_CALUDE_razorback_total_profit_l825_82526


namespace NUMINAMATH_CALUDE_max_small_packages_with_nine_large_l825_82591

/-- Represents the weight capacity of a service lift -/
structure LiftCapacity where
  large_packages : ℕ
  small_packages : ℕ

/-- Calculates the maximum number of small packages that can be carried alongside a given number of large packages -/
def max_small_packages (capacity : LiftCapacity) (large_count : ℕ) : ℕ :=
  let large_weight := capacity.small_packages / capacity.large_packages
  let remaining_capacity := capacity.small_packages - large_count * large_weight
  remaining_capacity

/-- Theorem: Given a lift with capacity of 12 large packages or 20 small packages,
    the maximum number of small packages that can be carried alongside 9 large packages is 5 -/
theorem max_small_packages_with_nine_large :
  let capacity := LiftCapacity.mk 12 20
  max_small_packages capacity 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_small_packages_with_nine_large_l825_82591


namespace NUMINAMATH_CALUDE_certain_fraction_problem_l825_82592

theorem certain_fraction_problem (a b x y : ℚ) : 
  (a / b) / (2 / 5) = (3 / 8) / (x / y) →
  a / b = 3 / 4 →
  x / y = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_problem_l825_82592


namespace NUMINAMATH_CALUDE_largest_common_term_largest_common_term_exists_l825_82542

theorem largest_common_term (n : ℕ) : n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) →
  n ≤ 194 := by
  sorry

theorem largest_common_term_exists : 
  ∃ n : ℕ, n = 194 ∧ n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_largest_common_term_exists_l825_82542


namespace NUMINAMATH_CALUDE_banana_bunches_l825_82588

theorem banana_bunches (total_bananas : ℕ) (bunches_of_seven : ℕ) (bananas_per_bunch_of_seven : ℕ) (bananas_per_bunch_of_eight : ℕ) :
  total_bananas = 83 →
  bunches_of_seven = 5 →
  bananas_per_bunch_of_seven = 7 →
  bananas_per_bunch_of_eight = 8 →
  ∃ (bunches_of_eight : ℕ), 
    total_bananas = bunches_of_seven * bananas_per_bunch_of_seven + bunches_of_eight * bananas_per_bunch_of_eight ∧
    bunches_of_eight = 6 :=
by sorry

end NUMINAMATH_CALUDE_banana_bunches_l825_82588


namespace NUMINAMATH_CALUDE_parabola_directrix_l825_82585

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the parabola g(x) = 2px^2
def g (p x : ℝ) : ℝ := 2 * p * x^2

-- Define the tangent line to f(x) at x = -1
def tangent_line (x : ℝ) : ℝ := 2 * x + 4

-- Theorem statement
theorem parabola_directrix :
  ∃ (p : ℝ),
    (∀ x, tangent_line x = g p x) ∧
    (∃ x, f x = tangent_line x ∧ x = -1) →
    (∀ y, y = 1 ↔ ∃ x, x^2 = -4 * y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l825_82585


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l825_82537

/-- The ratio of the volume of a cube with edge length 8 inches to the volume of a sphere with diameter 12 inches is 16 / (9 * π) -/
theorem cube_sphere_volume_ratio :
  let cube_edge : ℝ := 8
  let sphere_diameter : ℝ := 12
  let cube_volume := cube_edge ^ 3
  let sphere_radius := sphere_diameter / 2
  let sphere_volume := (4 / 3) * π * sphere_radius ^ 3
  cube_volume / sphere_volume = 16 / (9 * π) := by
sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l825_82537


namespace NUMINAMATH_CALUDE_square_root_divided_by_six_l825_82579

theorem square_root_divided_by_six (x : ℝ) : x > 0 ∧ Real.sqrt x / 6 = 1 ↔ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_six_l825_82579


namespace NUMINAMATH_CALUDE_expression_evaluation_l825_82535

theorem expression_evaluation :
  Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l825_82535


namespace NUMINAMATH_CALUDE_divisible_by_five_l825_82500

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (k = 2^n - 1 ∨ k = 2^n + 1 ∨ k = 2^(2*n) + 1) ∧ k % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l825_82500


namespace NUMINAMATH_CALUDE_equation_solutions_l825_82597

theorem equation_solutions :
  (∃ x : ℝ, (x + 2)^3 + 1 = 0 ↔ x = -3) ∧
  (∃ x : ℝ, (3*x - 2)^2 = 64 ↔ x = 10/3 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l825_82597


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l825_82555

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l825_82555


namespace NUMINAMATH_CALUDE_quadratic_no_roots_positive_c_l825_82546

/-- A quadratic polynomial with no real roots and positive sum of coefficients has a positive constant term. -/
theorem quadratic_no_roots_positive_c (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≠ 0) →  -- no real roots
  a + b + c > 0 →                   -- sum of coefficients is positive
  c > 0 :=                          -- constant term is positive
by sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_positive_c_l825_82546


namespace NUMINAMATH_CALUDE_intersection_point_l825_82545

theorem intersection_point (x y : ℚ) : 
  (x = 40/17 ∧ y = 21/17) ↔ (3*x + 4*y = 12 ∧ 7*x - 2*y = 14) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l825_82545


namespace NUMINAMATH_CALUDE_right_triangle_leg_l825_82553

theorem right_triangle_leg (h : Real) (angle : Real) :
  angle = Real.pi / 4 →
  h = 10 * Real.sqrt 2 →
  h * Real.sin angle = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_l825_82553


namespace NUMINAMATH_CALUDE_system_of_equations_l825_82575

theorem system_of_equations (x y : ℝ) 
  (eq1 : 2 * x + 4 * y = 5)
  (eq2 : x - y = 10) : 
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_l825_82575


namespace NUMINAMATH_CALUDE_angle_A_value_l825_82593

theorem angle_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_l825_82593


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l825_82531

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  point_at : ℝ → (ℝ × ℝ × ℝ)

/-- The vector at a given t value -/
def vector_at (line : ParametricLine) (t : ℝ) : (ℝ × ℝ × ℝ) :=
  line.point_at t

theorem vector_at_negative_one
  (line : ParametricLine)
  (h0 : vector_at line 0 = (2, 1, 5))
  (h1 : vector_at line 1 = (5, 0, 2)) :
  vector_at line (-1) = (-1, 2, 8) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l825_82531


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l825_82590

/-- A line passing through the point (2, 1) with a slope of 2 has the equation 2x - y - 3 = 0. -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (2 : ℝ) * x - y - 3 = 0 ↔ (y - 1 = 2 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l825_82590


namespace NUMINAMATH_CALUDE_symmetric_point_tanDoubleAngle_l825_82566

/-- Given a line l in the Cartesian plane defined by the equation 2x*tan(α) + y - 1 = 0,
    and that the symmetric point of the origin (0,0) with respect to l is (1,1),
    prove that tan(2α) = 4/3. -/
theorem symmetric_point_tanDoubleAngle (α : ℝ) : 
  (∀ x y : ℝ, 2 * x * Real.tan α + y - 1 = 0 → 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) → 
  Real.tan (2 * α) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_tanDoubleAngle_l825_82566


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l825_82554

theorem right_triangle_to_square (longer_leg : ℝ) (shorter_leg : ℝ) (square_side : ℝ) : 
  longer_leg = 10 →
  longer_leg = 2 * square_side →
  shorter_leg = square_side →
  shorter_leg = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l825_82554


namespace NUMINAMATH_CALUDE_power_inequality_l825_82560

theorem power_inequality (a b : ℝ) : a^6 + b^6 ≥ a^4*b^2 + a^2*b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l825_82560


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l825_82582

theorem circle_area_from_polar_equation :
  ∀ (r : ℝ → ℝ) (θ : ℝ),
    (r θ = 3 * Real.cos θ - 4 * Real.sin θ) →
    (∃ (c : ℝ × ℝ) (R : ℝ), ∀ (x y : ℝ),
      (x - c.1)^2 + (y - c.2)^2 = R^2 ↔ 
      ∃ θ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) →
    (π * (5/2)^2 = 25*π/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l825_82582


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l825_82581

/-- Given a > 0 and the coefficient of the 1/x term in the expansion of (a√x - 1/√x)^6 is 135, prove that a = 3 -/
theorem binomial_expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (Nat.choose 6 4) * a^2 = 135) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l825_82581


namespace NUMINAMATH_CALUDE_max_loss_is_9_l825_82586

/-- Represents the ratio of money for each person --/
structure MoneyRatio :=
  (cara : ℕ)
  (janet : ℕ)
  (jerry : ℕ)
  (linda : ℕ)

/-- Represents the price range for oranges --/
structure PriceRange :=
  (min : ℚ)
  (max : ℚ)

/-- Calculates the maximum loss for Cara and Janet --/
def calculate_max_loss (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) : ℚ :=
  sorry

theorem max_loss_is_9 (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) :
  ratio.cara = 4 ∧ ratio.janet = 5 ∧ ratio.jerry = 6 ∧ ratio.linda = 7 ∧
  total_money = 110 ∧
  price_range.min = 1/2 ∧ price_range.max = 3/2 ∧
  sell_percentage = 4/5 →
  calculate_max_loss ratio total_money price_range sell_percentage = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_loss_is_9_l825_82586


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l825_82522

/-- Given two vectors a and b in R², where a = (4,8) and b = (x,4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l825_82522


namespace NUMINAMATH_CALUDE_square_roots_sum_equals_ten_l825_82519

theorem square_roots_sum_equals_ten :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_equals_ten_l825_82519


namespace NUMINAMATH_CALUDE_craft_corner_sales_l825_82572

/-- The percentage of sales that are neither brushes nor paints -/
def other_sales_percentage (total : ℝ) (brushes : ℝ) (paints : ℝ) : ℝ :=
  total - (brushes + paints)

/-- Theorem stating that given the total sales is 100%, and brushes and paints account for 45% and 28% of sales respectively, the percentage of sales that are neither brushes nor paints is 27% -/
theorem craft_corner_sales :
  let total := 100
  let brushes := 45
  let paints := 28
  other_sales_percentage total brushes paints = 27 := by
sorry

end NUMINAMATH_CALUDE_craft_corner_sales_l825_82572


namespace NUMINAMATH_CALUDE_even_result_more_likely_l825_82543

/-- Represents a calculator operation -/
inductive Operation
  | Add : Nat → Operation
  | Subtract : Nat → Operation
  | Multiply : Nat → Operation

/-- Represents a sequence of calculator operations -/
def OperationSequence := List Operation

/-- Applies a single operation to a number -/
def applyOperation (n : Int) (op : Operation) : Int :=
  match op with
  | Operation.Add m => n + m
  | Operation.Subtract m => n - m
  | Operation.Multiply m => n * m

/-- Applies a sequence of operations to an initial number -/
def applySequence (initial : Int) (seq : OperationSequence) : Int :=
  seq.foldl applyOperation initial

/-- Probability of getting an even result from a random operation sequence -/
noncomputable def probEvenResult (seqLength : Nat) : Real :=
  sorry

theorem even_result_more_likely (seqLength : Nat) :
  probEvenResult seqLength > 1 / 2 := by sorry

end NUMINAMATH_CALUDE_even_result_more_likely_l825_82543


namespace NUMINAMATH_CALUDE_total_percentage_increase_approx_l825_82580

/-- Calculates the total percentage increase in USD for a purchase of three items with given initial and final prices in different currencies. -/
theorem total_percentage_increase_approx (book_initial book_final : ℝ)
                                         (album_initial album_final : ℝ)
                                         (poster_initial poster_final : ℝ)
                                         (usd_to_eur usd_to_gbp : ℝ)
                                         (h1 : book_initial = 300)
                                         (h2 : book_final = 480)
                                         (h3 : album_initial = 15)
                                         (h4 : album_final = 20)
                                         (h5 : poster_initial = 5)
                                         (h6 : poster_final = 10)
                                         (h7 : usd_to_eur = 0.85)
                                         (h8 : usd_to_gbp = 0.75) :
  ∃ ε > 0, abs (((book_final - book_initial + 
                 (album_final - album_initial) / usd_to_eur + 
                 (poster_final - poster_initial) / usd_to_gbp) / 
                (book_initial + album_initial / usd_to_eur + 
                 poster_initial / usd_to_gbp)) - 0.5937) < ε :=
by sorry


end NUMINAMATH_CALUDE_total_percentage_increase_approx_l825_82580


namespace NUMINAMATH_CALUDE_circle_intersection_angle_l825_82576

theorem circle_intersection_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 → r₂ = 3 → r₃ = 2 →
  shaded_ratio = 5 / 11 →
  ∃ θ : ℝ,
    θ > 0 ∧ θ < π / 2 ∧
    (θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2) / (π * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = π / 176 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_l825_82576


namespace NUMINAMATH_CALUDE_four_person_greeting_card_distribution_l825_82557

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * greeting_card_distribution (n - 1)

theorem four_person_greeting_card_distribution :
  greeting_card_distribution 4 = 9 :=
sorry

end NUMINAMATH_CALUDE_four_person_greeting_card_distribution_l825_82557


namespace NUMINAMATH_CALUDE_angle_DEB_value_l825_82574

-- Define the geometric configuration
structure GeometricConfig where
  -- Triangle ABC
  angleABC : ℝ
  angleACB : ℝ
  -- Other angles
  angleCDE : ℝ
  -- Straight line and angle conditions
  angleADC_straight : angleADC = 180
  angleAEB_straight : angleAEB = 180
  -- Given conditions
  h1 : angleABC = 72
  h2 : angleACB = 90
  h3 : angleCDE = 36

-- Theorem statement
theorem angle_DEB_value (config : GeometricConfig) : ∃ (angleDEB : ℝ), angleDEB = 162 := by
  sorry


end NUMINAMATH_CALUDE_angle_DEB_value_l825_82574


namespace NUMINAMATH_CALUDE_product_abcd_equals_162_over_185_l825_82589

theorem product_abcd_equals_162_over_185 
  (a b c d : ℚ) 
  (eq1 : 3*a + 4*b + 6*c + 9*d = 45)
  (eq2 : 4*(d+c) = b + 1)
  (eq3 : 4*b + 2*c = a)
  (eq4 : 2*c - 2 = d) :
  a * b * c * d = 162 / 185 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_equals_162_over_185_l825_82589


namespace NUMINAMATH_CALUDE_sum_remainder_l825_82525

theorem sum_remainder (x y : ℤ) (hx : x % 72 = 19) (hy : y % 50 = 6) : 
  (x + y) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l825_82525


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l825_82551

theorem fraction_product_simplification :
  (21 : ℚ) / 16 * 48 / 35 * 80 / 63 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l825_82551


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l825_82577

open Real

theorem min_shift_for_symmetry (φ : ℝ) : 
  φ > 0 ∧ 
  (∀ x, sin (2 * (x - φ)) = sin (2 * (π / 3 - x))) →
  φ ≥ 5 * π / 12 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l825_82577


namespace NUMINAMATH_CALUDE_nes_sale_price_l825_82587

/-- The sale price of the NES given the trade-in value of SNES, additional payment, change, and game value. -/
theorem nes_sale_price
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_payment : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_payment = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_payment - change - game_value = 160 :=
sorry


end NUMINAMATH_CALUDE_nes_sale_price_l825_82587


namespace NUMINAMATH_CALUDE_simplify_square_roots_l825_82504

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 1457 / 500 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l825_82504


namespace NUMINAMATH_CALUDE_closest_irrational_to_four_l825_82562

theorem closest_irrational_to_four :
  let options : List ℝ := [Real.sqrt 11, Real.sqrt 13, Real.sqrt 17, Real.sqrt 19]
  let four : ℝ := Real.sqrt 16
  ∀ x ∈ options, x ≠ Real.sqrt 17 →
    |Real.sqrt 17 - four| < |x - four| :=
by sorry

end NUMINAMATH_CALUDE_closest_irrational_to_four_l825_82562


namespace NUMINAMATH_CALUDE_total_pears_picked_l825_82509

/-- Represents a person who picks pears -/
structure PearPicker where
  name : String
  morning : Bool

/-- Calculates the number of pears picked on Day 2 -/
def day2Amount (day1 : ℕ) (morning : Bool) : ℕ :=
  if morning then day1 / 2 else day1 * 2

/-- Calculates the number of pears picked on Day 3 -/
def day3Amount (day1 day2 : ℕ) : ℕ :=
  (day1 + day2 + 1) / 2  -- Adding 1 for rounding up

/-- Calculates the total pears picked by a person over three days -/
def totalPears (day1 : ℕ) (morning : Bool) : ℕ :=
  let day2 := day2Amount day1 morning
  let day3 := day3Amount day1 day2
  day1 + day2 + day3

/-- The main theorem stating the total number of pears picked -/
theorem total_pears_picked : 
  let jason := PearPicker.mk "Jason" true
  let keith := PearPicker.mk "Keith" true
  let mike := PearPicker.mk "Mike" true
  let alicia := PearPicker.mk "Alicia" false
  let tina := PearPicker.mk "Tina" false
  let nicola := PearPicker.mk "Nicola" false
  totalPears 46 jason.morning +
  totalPears 47 keith.morning +
  totalPears 12 mike.morning +
  totalPears 28 alicia.morning +
  totalPears 33 tina.morning +
  totalPears 52 nicola.morning = 747 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l825_82509


namespace NUMINAMATH_CALUDE_specific_field_area_l825_82541

/-- Represents a rectangular field with partial fencing -/
structure PartiallyFencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ

/-- Calculates the area of a rectangular field -/
def field_area (f : PartiallyFencedField) : ℝ :=
  f.length * f.width

/-- Theorem: The area of a specific partially fenced field is 390 square feet -/
theorem specific_field_area :
  ∃ (f : PartiallyFencedField),
    f.uncovered_side = 20 ∧
    f.fencing = 59 ∧
    f.length = f.uncovered_side ∧
    2 * f.width + f.uncovered_side = f.fencing ∧
    field_area f = 390 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_area_l825_82541


namespace NUMINAMATH_CALUDE_september_reading_plan_l825_82595

theorem september_reading_plan (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (flight_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  flight_pages = 100 →
  ∃ (standard_pages : ℕ),
    standard_pages * (total_days - busy_days - 1) + flight_pages = total_pages ∧
    standard_pages = 20 := by
  sorry

end NUMINAMATH_CALUDE_september_reading_plan_l825_82595


namespace NUMINAMATH_CALUDE_intersection_distance_l825_82570

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : ℝ  -- x-coordinate of the vertical directrix

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if a point is on the parabola -/
def isOnParabola (p : Point) (pa : Parabola) : Prop :=
  p.x = 2 * (5 + 2 * Real.sqrt 3) * p.y^2 + (5 + 2 * Real.sqrt 3) / 2

/-- The main theorem -/
theorem intersection_distance (e : Ellipse) (pa : Parabola) 
    (p1 p2 : Point) : 
    e.center = Point.mk 0 0 →
    e.a = 4 →
    e.b = 2 →
    pa.directrix = 5 →
    pa.focus = Point.mk (2 * Real.sqrt 3) 0 →
    isOnEllipse p1 e →
    isOnEllipse p2 e →
    isOnParabola p1 pa →
    isOnParabola p2 pa →
    p1 ≠ p2 →
    ∃ (d : ℝ), d = 2 * |p1.y - p2.y| ∧ 
               d = Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) :=
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l825_82570


namespace NUMINAMATH_CALUDE_fraction_sum_evaluation_l825_82544

theorem fraction_sum_evaluation (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 9) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_evaluation_l825_82544


namespace NUMINAMATH_CALUDE_third_vertex_x_coord_l825_82505

/-- An equilateral triangle with two vertices at (5, 0) and (5, 8) -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ := (5, 0)
  v2 : ℝ × ℝ := (5, 8)
  v3 : ℝ × ℝ
  equilateral : sorry
  v3_in_first_quadrant : v3.1 > 0 ∧ v3.2 > 0

/-- The x-coordinate of the third vertex is 5 + 4√3 -/
theorem third_vertex_x_coord (t : EquilateralTriangle) : t.v3.1 = 5 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_x_coord_l825_82505


namespace NUMINAMATH_CALUDE_quadratic_property_contradiction_l825_82556

/-- Represents a quadratic function of the form y = ax² + bx - 6 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Properties of the quadratic function --/
def QuadraticProperties (f : QuadraticFunction) : Prop :=
  ∃ (x_sym : ℝ) (y_min : ℝ),
    -- Axis of symmetry is x = 1
    x_sym = 1 ∧
    -- Minimum value is -8
    y_min = -8 ∧
    -- x = 3 is a root
    f.a * 3^2 + f.b * 3 - 6 = 0

/-- The main theorem to prove --/
theorem quadratic_property_contradiction (f : QuadraticFunction) 
  (h : QuadraticProperties f) : 
  f.a * 3^2 + f.b * 3 - 6 ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_contradiction_l825_82556


namespace NUMINAMATH_CALUDE_hexagram_ratio_is_three_l825_82550

/-- A hexagram formed by overlapping two equilateral triangles -/
structure Hexagram where
  /-- The hexagram's vertices coincide with those of a regular hexagon -/
  vertices_coincide : Bool
  /-- The number of smaller triangles in the shaded region -/
  shaded_triangles : Nat
  /-- The number of smaller triangles in the unshaded region -/
  unshaded_triangles : Nat

/-- The ratio of shaded to unshaded area in a hexagram -/
def shaded_unshaded_ratio (h : Hexagram) : ℚ :=
  h.shaded_triangles / h.unshaded_triangles

/-- Theorem: The ratio of shaded to unshaded area in the specified hexagram is 3 -/
theorem hexagram_ratio_is_three (h : Hexagram) 
  (h_vertices : h.vertices_coincide = true)
  (h_shaded : h.shaded_triangles = 18)
  (h_unshaded : h.unshaded_triangles = 6) : 
  shaded_unshaded_ratio h = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagram_ratio_is_three_l825_82550
