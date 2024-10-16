import Mathlib

namespace NUMINAMATH_CALUDE_walk_ratio_l3878_387861

def distance_first_hour : ℝ := 2
def total_distance : ℝ := 6

def distance_second_hour : ℝ := total_distance - distance_first_hour

theorem walk_ratio :
  distance_second_hour / distance_first_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_walk_ratio_l3878_387861


namespace NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l3878_387858

/-- Represents the duration of a workday in hours -/
def workday_hours : ℕ := 9

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Calculates the total workday time in minutes -/
def total_workday_minutes : ℕ := workday_hours * 60

/-- Theorem stating that the percentage of the workday spent in meetings is 25% -/
theorem percentage_of_workday_in_meetings : 
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l3878_387858


namespace NUMINAMATH_CALUDE_p_amount_l3878_387822

theorem p_amount (p q r : ℚ) 
  (h1 : p = (1/6 * p + 1/6 * p) + 32) : p = 48 := by
  sorry

end NUMINAMATH_CALUDE_p_amount_l3878_387822


namespace NUMINAMATH_CALUDE_max_regions_five_lines_l3878_387890

/-- The maximum number of regions a rectangle can be divided into by n line segments -/
def maxRegions (n : ℕ) : ℕ :=
  if n = 0 then 1 else maxRegions (n - 1) + n

/-- Theorem: The maximum number of regions a rectangle can be divided into by 5 line segments is 16 -/
theorem max_regions_five_lines :
  maxRegions 5 = 16 := by sorry

end NUMINAMATH_CALUDE_max_regions_five_lines_l3878_387890


namespace NUMINAMATH_CALUDE_string_length_problem_l3878_387876

/-- Given three strings A, B, and C, where the length of A is 6 times the length of C
    and 5 times the length of B, and the length of B is 12 meters,
    prove that the length of C is 10 meters. -/
theorem string_length_problem (A B C : ℝ) 
    (h1 : A = 6 * C) 
    (h2 : A = 5 * B) 
    (h3 : B = 12) : 
  C = 10 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l3878_387876


namespace NUMINAMATH_CALUDE_all_propositions_imply_target_l3878_387805

theorem all_propositions_imply_target : ∀ (p q r : Prop),
  (p ∧ q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ q ∧ ¬r → (p → q) ∨ r) ∧
  (p ∧ ¬q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ ¬q ∧ ¬r → (p → q) ∨ r) :=
by sorry

#check all_propositions_imply_target

end NUMINAMATH_CALUDE_all_propositions_imply_target_l3878_387805


namespace NUMINAMATH_CALUDE_two_hearts_three_different_probability_l3878_387829

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (other_suits : Nat)
  (cards_eq : cards = 52)
  (hearts_eq : hearts = 13)
  (other_suits_eq : other_suits = 39)

/-- The probability of the specified event -/
def probability_two_hearts_three_different (d : Deck) : ℚ :=
  135 / 1024

/-- Theorem statement -/
theorem two_hearts_three_different_probability (d : Deck) :
  probability_two_hearts_three_different d = 135 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_two_hearts_three_different_probability_l3878_387829


namespace NUMINAMATH_CALUDE_special_arrangement_count_l3878_387896

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in a row -/
def linearArrangements (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange 5 people in a row, where 2 specific people
    must be adjacent and in a specific order -/
def specialArrangement : ℕ := linearArrangements 4

theorem special_arrangement_count :
  specialArrangement = 24 :=
sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l3878_387896


namespace NUMINAMATH_CALUDE_individuals_from_c_is_twenty_l3878_387871

/-- Represents the ratio of individuals in strata A, B, and C -/
structure StrataRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the number of individuals to be drawn from stratum C -/
def individualsFromC (ratio : StrataRatio) (sampleSize : ℕ) : ℕ :=
  (ratio.c * sampleSize) / (ratio.a + ratio.b + ratio.c)

/-- Theorem: Given the specified ratio and sample size, 20 individuals should be drawn from C -/
theorem individuals_from_c_is_twenty :
  let ratio := StrataRatio.mk 5 3 2
  let sampleSize := 100
  individualsFromC ratio sampleSize = 20 := by
  sorry

end NUMINAMATH_CALUDE_individuals_from_c_is_twenty_l3878_387871


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3878_387838

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3878_387838


namespace NUMINAMATH_CALUDE_pebble_difference_l3878_387856

theorem pebble_difference (candy_pebbles : ℕ) (lance_multiplier : ℕ) : 
  candy_pebbles = 4 →
  lance_multiplier = 3 →
  lance_multiplier * candy_pebbles - candy_pebbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_pebble_difference_l3878_387856


namespace NUMINAMATH_CALUDE_base_7_addition_sum_l3878_387855

-- Define a function to convert a base 7 number to base 10
def to_base_10 (x : ℕ) (y : ℕ) (z : ℕ) : ℕ := x * 49 + y * 7 + z

-- Define the addition problem in base 7
def addition_problem (X Y : ℕ) : Prop :=
  to_base_10 2 X Y + to_base_10 0 5 2 = to_base_10 3 1 X

-- Define the condition that X and Y are single digits in base 7
def single_digit_base_7 (n : ℕ) : Prop := n < 7

theorem base_7_addition_sum :
  ∀ X Y : ℕ,
    addition_problem X Y →
    single_digit_base_7 X →
    single_digit_base_7 Y →
    X + Y = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_7_addition_sum_l3878_387855


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3878_387857

theorem root_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ Real.exp x = 1/x := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l3878_387857


namespace NUMINAMATH_CALUDE_cat_path_tiles_l3878_387875

def garden_width : ℕ := 12
def garden_length : ℕ := 20
def tile_size : ℕ := 2
def tiles_width : ℕ := garden_width / tile_size
def tiles_length : ℕ := garden_length / tile_size

theorem cat_path_tiles : 
  tiles_width + tiles_length - Nat.gcd tiles_width tiles_length - 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_path_tiles_l3878_387875


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3878_387847

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 6) + 18 + 3*x + 12 + (x + 9) + (3*x - 5)) / 6 = 19 → x = 37/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3878_387847


namespace NUMINAMATH_CALUDE_function_composition_l3878_387878

-- Define the function f
def f : ℝ → ℝ := fun x => 3 * (x + 1) - 1

-- State the theorem
theorem function_composition (x : ℝ) : f x = 3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l3878_387878


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3878_387817

/-- Represents a cube structure with removals -/
structure CubeStructure where
  size : Nat
  smallCubeSize : Nat
  centerRemoved : Bool
  faceCentersRemoved : Bool
  outerLayerReduced : Bool

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (c : CubeStructure) : Nat :=
  sorry

/-- The main theorem stating the surface area of the specific cube structure -/
theorem modified_cube_surface_area :
  let c : CubeStructure := {
    size := 12,
    smallCubeSize := 2,
    centerRemoved := true,
    faceCentersRemoved := true,
    outerLayerReduced := true
  }
  surfaceArea c = 3304 := by sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3878_387817


namespace NUMINAMATH_CALUDE_trivia_team_selection_l3878_387870

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 36)
  (h2 : num_groups = 3)
  (h3 : students_per_group = 9) :
  total_students - (num_groups * students_per_group) = 9 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l3878_387870


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3878_387832

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3878_387832


namespace NUMINAMATH_CALUDE_one_book_selection_ways_l3878_387894

/-- The number of ways to take one book from a shelf with Chinese, math, and English books. -/
def ways_to_take_one_book (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books + math_books + english_books

/-- Theorem: There are 37 ways to take one book from a shelf with 12 Chinese books, 14 math books, and 11 English books. -/
theorem one_book_selection_ways :
  ways_to_take_one_book 12 14 11 = 37 := by
  sorry

end NUMINAMATH_CALUDE_one_book_selection_ways_l3878_387894


namespace NUMINAMATH_CALUDE_games_won_l3878_387888

/-- Proves that the number of games won is 8, given the total games and lost games. -/
theorem games_won (total_games lost_games : ℕ) 
  (h1 : total_games = 12) 
  (h2 : lost_games = 4) : 
  total_games - lost_games = 8 := by
  sorry

#check games_won

end NUMINAMATH_CALUDE_games_won_l3878_387888


namespace NUMINAMATH_CALUDE_cookie_average_l3878_387831

theorem cookie_average : 
  let cookie_counts : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]
  (cookie_counts.sum : ℚ) / cookie_counts.length = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_average_l3878_387831


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l3878_387844

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_difference_quotient : (factorial 13 - factorial 12) / factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l3878_387844


namespace NUMINAMATH_CALUDE_lowest_possible_score_l3878_387819

/-- Represents a set of test scores -/
structure TestScores where
  scores : List Nat
  all_valid : ∀ s ∈ scores, s ≤ 100

/-- Calculates the average of a list of scores -/
def average (ts : TestScores) : Rat :=
  (ts.scores.sum : Rat) / ts.scores.length

/-- The problem statement -/
theorem lowest_possible_score 
  (first_two : TestScores)
  (h1 : first_two.scores = [82, 75])
  (h2 : first_two.scores.length = 2)
  : ∃ (last_two : TestScores),
    last_two.scores.length = 2 ∧ 
    (∃ (s : Nat), s ∈ last_two.scores ∧ s = 83) ∧
    average (TestScores.mk (first_two.scores ++ last_two.scores) sorry) = 85 ∧
    (∀ (other_last_two : TestScores),
      other_last_two.scores.length = 2 →
      average (TestScores.mk (first_two.scores ++ other_last_two.scores) sorry) = 85 →
      ∀ (s : Nat), s ∈ other_last_two.scores → s ≥ 83) := by
  sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l3878_387819


namespace NUMINAMATH_CALUDE_initial_orange_balloons_l3878_387825

theorem initial_orange_balloons (blue_balloons : ℕ) (lost_orange_balloons : ℕ) (remaining_orange_balloons : ℕ) : 
  blue_balloons = 4 → 
  lost_orange_balloons = 2 → 
  remaining_orange_balloons = 7 → 
  remaining_orange_balloons + lost_orange_balloons = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_orange_balloons_l3878_387825


namespace NUMINAMATH_CALUDE_chocolate_boxes_pieces_per_box_l3878_387821

theorem chocolate_boxes_pieces_per_box 
  (initial_boxes : ℕ) 
  (given_away_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_boxes = 14)
  (h2 : given_away_boxes = 5)
  (h3 : remaining_pieces = 54)
  (h4 : initial_boxes > given_away_boxes) :
  (remaining_pieces / (initial_boxes - given_away_boxes) = 6) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_boxes_pieces_per_box_l3878_387821


namespace NUMINAMATH_CALUDE_diameter_endpoint_theorem_l3878_387848

/-- A circle in a 2D coordinate plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space. -/
def Point := ℝ × ℝ

/-- Given a circle and one endpoint of its diameter, compute the other endpoint. -/
def otherDiameterEndpoint (c : Circle) (p : Point) : Point :=
  (2 * c.center.1 - p.1, 2 * c.center.2 - p.2)

theorem diameter_endpoint_theorem (c : Circle) (p : Point) :
  c.center = (3, 4) → p = (1, -2) → otherDiameterEndpoint c p = (5, 10) := by
  sorry

#check diameter_endpoint_theorem

end NUMINAMATH_CALUDE_diameter_endpoint_theorem_l3878_387848


namespace NUMINAMATH_CALUDE_quadratic_derivative_bound_l3878_387865

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The derivative of a quadratic polynomial at x = 0 -/
def QuadraticPolynomial.deriv_at_zero (p : QuadraticPolynomial) : ℝ := p.b

/-- A quadratic polynomial is bounded by 1 on [0, 1] -/
def is_bounded_by_one (p : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |p.eval x| ≤ 1

theorem quadratic_derivative_bound (p : QuadraticPolynomial) 
  (h : is_bounded_by_one p) : 
  |p.deriv_at_zero| ≤ 8 ∧ 
  ∀ α : ℝ, (∀ q : QuadraticPolynomial, is_bounded_by_one q → |q.deriv_at_zero| ≤ α) → 
  8 ≤ α := by
  sorry

end NUMINAMATH_CALUDE_quadratic_derivative_bound_l3878_387865


namespace NUMINAMATH_CALUDE_jeff_running_schedule_l3878_387824

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule 
  (weekday_run : ℕ) -- Planned running time per weekday in minutes
  (thursday_cut : ℕ) -- Minutes cut from Thursday's run
  (total_time : ℕ) -- Total running time for the week in minutes
  (h1 : weekday_run = 60)
  (h2 : thursday_cut = 20)
  (h3 : total_time = 290) :
  total_time - (4 * weekday_run + (weekday_run - thursday_cut)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_jeff_running_schedule_l3878_387824


namespace NUMINAMATH_CALUDE_limit_of_a_l3878_387891

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_a_l3878_387891


namespace NUMINAMATH_CALUDE_unique_n_for_prime_ones_and_seven_l3878_387866

def has_n_minus_one_ones_and_one_seven (n : ℕ) (x : ℕ) : Prop :=
  ∃ k : ℕ, k < n ∧ x = (10^n - 1) / 9 + 6 * 10^k

theorem unique_n_for_prime_ones_and_seven :
  ∃! n : ℕ, n > 0 ∧ ∀ x : ℕ, has_n_minus_one_ones_and_one_seven n x → Nat.Prime x :=
by sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_ones_and_seven_l3878_387866


namespace NUMINAMATH_CALUDE_susan_spending_equals_2000_l3878_387823

/-- The cost of a single pencil in cents -/
def pencil_cost : ℕ := 25

/-- The cost of a single pen in cents -/
def pen_cost : ℕ := 80

/-- The total number of items (pens and pencils) Susan bought -/
def total_items : ℕ := 36

/-- The number of pencils Susan bought -/
def pencils_bought : ℕ := 16

/-- Calculate Susan's total spending in cents -/
def susan_spending : ℕ := pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)

/-- Theorem: Susan's total spending equals $20.00 -/
theorem susan_spending_equals_2000 : susan_spending = 2000 := by sorry

end NUMINAMATH_CALUDE_susan_spending_equals_2000_l3878_387823


namespace NUMINAMATH_CALUDE_uncle_james_height_difference_l3878_387873

theorem uncle_james_height_difference :
  ∀ (james_original_height james_new_height uncle_height : ℝ),
  uncle_height = 72 →
  james_original_height = (2/3) * uncle_height →
  james_new_height = james_original_height + 10 →
  uncle_height - james_new_height = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_james_height_difference_l3878_387873


namespace NUMINAMATH_CALUDE_direct_variation_problem_l3878_387800

/-- A function representing direct variation between z and w -/
def directVariation (k : ℝ) (w : ℝ) : ℝ := k * w

theorem direct_variation_problem (k : ℝ) :
  (directVariation k 5 = 10) →
  (directVariation k 15 = 30) :=
by
  sorry

#check direct_variation_problem

end NUMINAMATH_CALUDE_direct_variation_problem_l3878_387800


namespace NUMINAMATH_CALUDE_parabola_focus_l3878_387827

/-- The parabola with equation y² = -8x has its focus at (-2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3878_387827


namespace NUMINAMATH_CALUDE_complex_equation_product_l3878_387841

theorem complex_equation_product (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l3878_387841


namespace NUMINAMATH_CALUDE_point_transformation_l3878_387801

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180° clockwise around (2,3) -/
def rotate180 (p : Point) : Point :=
  { x := 4 - p.x, y := 6 - p.y }

/-- Reflects a point about the line y = x -/
def reflectAboutYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Translates a point by the vector (4, -2) -/
def translate (p : Point) : Point :=
  { x := p.x + 4, y := p.y - 2 }

/-- The main theorem -/
theorem point_transformation (Q : Point) :
  (translate (reflectAboutYEqualsX (rotate180 Q)) = Point.mk 1 6) →
  (Q.y - Q.x = 13) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3878_387801


namespace NUMINAMATH_CALUDE_banana_cost_l3878_387804

/-- The cost of a bunch of bananas can be expressed as $5 minus the cost of a dozen apples -/
theorem banana_cost (apple_cost banana_cost : ℝ) : 
  apple_cost + banana_cost = 5 → banana_cost = 5 - apple_cost := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l3878_387804


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3878_387818

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3878_387818


namespace NUMINAMATH_CALUDE_specific_composite_square_perimeter_l3878_387899

/-- Represents a square composed of four rectangles and an inner square -/
structure CompositeSquare where
  /-- Total area of the four rectangles -/
  rectangle_area : ℝ
  /-- Area of the square formed by the inner vertices of the rectangles -/
  inner_square_area : ℝ

/-- Calculates the total perimeter of the four rectangles in a CompositeSquare -/
def total_perimeter (cs : CompositeSquare) : ℝ :=
  sorry

/-- Theorem stating that for a specific CompositeSquare, the total perimeter is 48 -/
theorem specific_composite_square_perimeter :
  ∃ (cs : CompositeSquare),
    cs.rectangle_area = 32 ∧
    cs.inner_square_area = 20 ∧
    total_perimeter cs = 48 :=
  sorry

end NUMINAMATH_CALUDE_specific_composite_square_perimeter_l3878_387899


namespace NUMINAMATH_CALUDE_supplement_of_supplement_58_l3878_387833

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_supplement_58 :
  supplement (supplement 58) = 58 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_supplement_58_l3878_387833


namespace NUMINAMATH_CALUDE_no_solution_condition_l3878_387887

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (k * x) / (x - 1) - (2 * k - 1) / (1 - x) ≠ 2) ↔ 
  (k = 1/3 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3878_387887


namespace NUMINAMATH_CALUDE_triangle_properties_l3878_387880

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a^2 + t.b^2 - t.c^2) * Real.tan t.C = Real.sqrt 2 * t.a * t.b) :
  (t.C = π/4 ∨ t.C = 3*π/4) ∧ 
  (t.c = 2 ∧ t.b = 2 * Real.sqrt 2 → 
    1/2 * t.a * t.b * Real.sin t.C = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3878_387880


namespace NUMINAMATH_CALUDE_joshua_bottles_count_l3878_387809

theorem joshua_bottles_count (bottles_per_crate : ℕ) (num_crates : ℕ) (extra_bottles : ℕ) : 
  bottles_per_crate = 12 → 
  num_crates = 10 → 
  extra_bottles = 10 → 
  bottles_per_crate * num_crates + extra_bottles = 130 := by
sorry

end NUMINAMATH_CALUDE_joshua_bottles_count_l3878_387809


namespace NUMINAMATH_CALUDE_number_of_boys_l3878_387836

theorem number_of_boys (total_kids : ℕ) (girls : ℕ) (boys : ℕ) :
  total_kids = 9 →
  girls = 3 →
  total_kids = girls + boys →
  boys = 6 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3878_387836


namespace NUMINAMATH_CALUDE_parallelogram_height_l3878_387813

/-- Theorem: Height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = 448 ∧ base = 32 ∧ area = base * height → height = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3878_387813


namespace NUMINAMATH_CALUDE_cos_half_times_one_plus_sin_max_value_l3878_387886

theorem cos_half_times_one_plus_sin_max_value :
  ∀ θ : Real, 0 ≤ θ ∧ θ ≤ π / 2 →
    (∀ φ : Real, 0 ≤ φ ∧ φ ≤ π / 2 →
      Real.cos (θ / 2) * (1 + Real.sin θ) ≤ Real.cos (φ / 2) * (1 + Real.sin φ)) →
    Real.cos (θ / 2) * (1 + Real.sin θ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cos_half_times_one_plus_sin_max_value_l3878_387886


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l3878_387881

theorem smallest_m_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ (m : Nat), m > 0 ∧ (∀ (q : Nat), Nat.Prime q → q > 3 → 105 ∣ 9^(q^2) - 29^q + m) ∧
  (∀ (k : Nat), k > 0 → k < m → ∃ (r : Nat), Nat.Prime r → r > 3 → ¬(105 ∣ 9^(r^2) - 29^r + k)) ∧
  m = 95 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l3878_387881


namespace NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l3878_387828

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  Complex.abs z₁ = Complex.abs (2*z₂) → 
  a^2 / b = 4.5 := by sorry

end NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l3878_387828


namespace NUMINAMATH_CALUDE_profit_reduction_theorem_l3878_387884

/-- Initial daily sales -/
def initial_sales : ℕ := 30

/-- Initial profit per unit in yuan -/
def initial_profit_per_unit : ℕ := 50

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℕ := 2

/-- Calculate daily profit based on price reduction -/
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit_per_unit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

/-- Price reduction needed for a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20  -- This is the value we want to prove

/-- Price reduction for maximum profit -/
def price_reduction_for_max_profit : ℝ :=
  17.5  -- This is the value we want to prove

theorem profit_reduction_theorem :
  daily_profit (price_reduction_for_profit 2100) = 2100 ∧
  ∀ x, daily_profit x ≤ daily_profit price_reduction_for_max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_reduction_theorem_l3878_387884


namespace NUMINAMATH_CALUDE_purple_four_leaved_clovers_l3878_387839

theorem purple_four_leaved_clovers (total_clovers : ℕ) (four_leaf_percentage : ℚ) (purple_fraction : ℚ) : 
  total_clovers = 500 →
  four_leaf_percentage = 1/5 →
  purple_fraction = 1/4 →
  (total_clovers : ℚ) * four_leaf_percentage * purple_fraction = 25 := by
sorry

end NUMINAMATH_CALUDE_purple_four_leaved_clovers_l3878_387839


namespace NUMINAMATH_CALUDE_stopping_time_maximizes_distance_l3878_387840

/-- The distance function representing the distance traveled by a car after braking. -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time at which the distance function reaches its maximum value. -/
def stopping_time : ℝ := 3

/-- Theorem stating that the stopping time maximizes the distance function. -/
theorem stopping_time_maximizes_distance :
  ∀ t : ℝ, S t ≤ S stopping_time :=
sorry

end NUMINAMATH_CALUDE_stopping_time_maximizes_distance_l3878_387840


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3878_387837

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f x * f y = f x + f y - f (x * y))
  (h2 : ∀ x y : ℚ, 1 + f (x + y) = f (x * y) + f x * f y) :
  (∀ x : ℚ, f x = 1) ∨ (∀ x : ℚ, f x = 1 - x) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3878_387837


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l3878_387882

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3) →
  (∃ x : ℝ, 2 - (a - x) / 3 = 1) →
  (∀ x : ℝ, 2 * x + 1 = 3 ↔ 2 - (a - x) / 3 = 1) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l3878_387882


namespace NUMINAMATH_CALUDE_correct_calculation_l3878_387853

theorem correct_calculation (x : ℤ) : x + 392 = 541 → x + 293 = 442 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3878_387853


namespace NUMINAMATH_CALUDE_largest_two_decimal_rounding_to_five_l3878_387810

-- Define a two-decimal number that rounds to 5.0
def is_valid_number (x : ℚ) : Prop :=
  (x ≥ 4.95) ∧ (x < 5.05) ∧ (∃ n : ℤ, x = n / 100)

-- Define the largest possible value
def largest_value : ℚ := 5.04

-- Theorem statement
theorem largest_two_decimal_rounding_to_five :
  ∀ x : ℚ, is_valid_number x → x ≤ largest_value :=
by sorry

end NUMINAMATH_CALUDE_largest_two_decimal_rounding_to_five_l3878_387810


namespace NUMINAMATH_CALUDE_sum_x_y_equals_ten_l3878_387812

theorem sum_x_y_equals_ten (x y : ℝ) 
  (h1 : |x| - x + y = 6)
  (h2 : x + |y| + y = 16) :
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_ten_l3878_387812


namespace NUMINAMATH_CALUDE_derivative_at_one_equals_three_l3878_387852

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

-- State the theorem
theorem derivative_at_one_equals_three :
  deriv f 1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_derivative_at_one_equals_three_l3878_387852


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3878_387862

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/11 < (359 : ℚ)/(359 + k) ∧ (359 : ℚ)/(359 + k) < (6 : ℚ)/11 ∧
  (∀ (n : ℕ) (k₁ k₂ : ℤ), n > 359 →
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11) →
    k₁ = k₂) →
  (∃ (k₁ k₂ : ℤ), k₁ ≠ k₂ ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3878_387862


namespace NUMINAMATH_CALUDE_sum_and_diff_expectations_l3878_387842

variable (X Y : ℝ → ℝ)

-- Define the expectation operator
def expectation (Z : ℝ → ℝ) : ℝ := sorry

-- Given conditions
axiom X_expectation : expectation X = 3
axiom Y_expectation : expectation Y = 2

-- Linearity of expectation
axiom expectation_sum (Z W : ℝ → ℝ) : expectation (Z + W) = expectation Z + expectation W
axiom expectation_diff (Z W : ℝ → ℝ) : expectation (Z - W) = expectation Z - expectation W

-- Theorem to prove
theorem sum_and_diff_expectations :
  expectation (X + Y) = 5 ∧ expectation (X - Y) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_and_diff_expectations_l3878_387842


namespace NUMINAMATH_CALUDE_average_weight_increase_l3878_387895

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 4 →
  old_weight = 95 →
  new_weight = 129 →
  (new_weight - old_weight) / initial_count = 8.5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3878_387895


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3878_387889

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 1 * a 9 = 16 → a 2 * a 5 * a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3878_387889


namespace NUMINAMATH_CALUDE_jills_peaches_l3878_387843

/-- Given that Steven has 19 peaches and 13 more peaches than Jill,
    prove that Jill has 6 peaches. -/
theorem jills_peaches (steven_peaches : ℕ) (steven_jill_diff : ℕ) 
  (h1 : steven_peaches = 19)
  (h2 : steven_peaches = steven_jill_diff + jill_peaches) :
  jill_peaches = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jills_peaches_l3878_387843


namespace NUMINAMATH_CALUDE_field_trip_buses_l3878_387892

theorem field_trip_buses (total_classrooms : ℕ) (freshmen_classrooms : ℕ) (sophomore_classrooms : ℕ)
  (freshmen_per_room : ℕ) (sophomores_per_room : ℕ) (bus_capacity : ℕ) (teachers_per_room : ℕ)
  (bus_drivers : ℕ) :
  total_classrooms = 95 →
  freshmen_classrooms = 45 →
  sophomore_classrooms = 50 →
  freshmen_per_room = 58 →
  sophomores_per_room = 47 →
  bus_capacity = 40 →
  teachers_per_room = 2 →
  bus_drivers = 15 →
  ∃ (buses : ℕ), buses = 130 ∧ 
    buses * bus_capacity ≥ 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers ∧
    (buses - 1) * bus_capacity < 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers :=
by
  sorry


end NUMINAMATH_CALUDE_field_trip_buses_l3878_387892


namespace NUMINAMATH_CALUDE_cookie_box_cost_cookie_box_cost_is_three_l3878_387874

/-- The cost of the box of cookies given bracelet-making conditions --/
theorem cookie_box_cost 
  (cost_per_bracelet : ℝ) 
  (selling_price : ℝ) 
  (num_bracelets : ℕ) 
  (money_left : ℝ) : ℝ :=
  let profit_per_bracelet := selling_price - cost_per_bracelet
  let total_profit := profit_per_bracelet * num_bracelets
  let cookie_box_cost := total_profit - money_left
  cookie_box_cost

/-- Proof that the cost of the box of cookies is $3 --/
theorem cookie_box_cost_is_three :
  cookie_box_cost 1 1.5 12 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_cost_cookie_box_cost_is_three_l3878_387874


namespace NUMINAMATH_CALUDE_nancy_limes_l3878_387834

def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def total_limes : ℕ := 103

theorem nancy_limes : total_limes - (fred_limes + alyssa_limes) = 35 := by
  sorry

end NUMINAMATH_CALUDE_nancy_limes_l3878_387834


namespace NUMINAMATH_CALUDE_car_rental_problem_l3878_387898

/-- Represents the characteristics of a car type -/
structure CarType where
  capacity : ℕ
  rentalFee : ℕ

/-- Represents a rental option -/
structure RentalOption where
  typeACars : ℕ
  typeBCars : ℕ

/-- Checks if a rental option is valid given the constraints -/
def isValidRental (opt : RentalOption) (typeA typeB : CarType) (totalCars maxCost totalPeople : ℕ) : Prop :=
  opt.typeACars + opt.typeBCars = totalCars ∧
  opt.typeACars > 0 ∧
  opt.typeBCars > 0 ∧
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee ≤ maxCost ∧
  opt.typeACars * typeA.capacity + opt.typeBCars * typeB.capacity ≥ totalPeople

/-- Calculates the total cost of a rental option -/
def rentalCost (opt : RentalOption) (typeA typeB : CarType) : ℕ :=
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee

theorem car_rental_problem (typeA typeB : CarType) 
    (h_typeA_capacity : typeA.capacity = 50)
    (h_typeA_fee : typeA.rentalFee = 400)
    (h_typeB_capacity : typeB.capacity = 30)
    (h_typeB_fee : typeB.rentalFee = 280)
    (totalCars : ℕ) (h_totalCars : totalCars = 10)
    (maxCost : ℕ) (h_maxCost : maxCost = 3500)
    (totalPeople : ℕ) (h_totalPeople : totalPeople = 360) :
  (∃ (opt : RentalOption), isValidRental opt typeA typeB totalCars maxCost totalPeople ∧ 
    opt.typeACars = 5 ∧ 
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      opt'.typeACars ≤ opt.typeACars)) ∧
  (∃ (optCostEffective : RentalOption), 
    isValidRental optCostEffective typeA typeB totalCars maxCost totalPeople ∧
    optCostEffective.typeACars = 3 ∧ 
    optCostEffective.typeBCars = 7 ∧
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      rentalCost optCostEffective typeA typeB ≤ rentalCost opt' typeA typeB)) := by
  sorry

end NUMINAMATH_CALUDE_car_rental_problem_l3878_387898


namespace NUMINAMATH_CALUDE_fraction_inequality_l3878_387885

theorem fraction_inequality (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3878_387885


namespace NUMINAMATH_CALUDE_lcm_problem_l3878_387893

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3878_387893


namespace NUMINAMATH_CALUDE_product_equality_l3878_387808

theorem product_equality : 375680169467 * 4565579427629 = 1715110767607750737263 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3878_387808


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3878_387859

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 = 4*y
def C2 (x y : ℝ) : Prop := x + y = 5

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem intersection_point_sum : 
  1 / distance P A + 1 / distance P B = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3878_387859


namespace NUMINAMATH_CALUDE_marble_probability_l3878_387850

/-- The probability of drawing 1 blue marble and 2 black marbles from a basket -/
theorem marble_probability (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 6)
  (h_black : black = 7) :
  let total := blue + yellow + black
  (blue : ℚ) / total * (black * (black - 1) : ℚ) / ((total - 1) * (total - 2)) = 7 / 170 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3878_387850


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3878_387802

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x > 0}
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3878_387802


namespace NUMINAMATH_CALUDE_milk_liters_bought_l3878_387820

/-- Given the costs of ingredients and the total cost, prove the number of liters of milk bought. -/
theorem milk_liters_bought (flour_boxes : ℕ) (flour_cost : ℕ) (egg_trays : ℕ) (egg_cost : ℕ)
  (milk_cost : ℕ) (soda_boxes : ℕ) (soda_cost : ℕ) (total_cost : ℕ)
  (h1 : flour_boxes = 3) (h2 : flour_cost = 3) (h3 : egg_trays = 3) (h4 : egg_cost = 10)
  (h5 : milk_cost = 5) (h6 : soda_boxes = 2) (h7 : soda_cost = 3) (h8 : total_cost = 80) :
  (total_cost - (flour_boxes * flour_cost + egg_trays * egg_cost + soda_boxes * soda_cost)) / milk_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_liters_bought_l3878_387820


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3878_387807

/-- Given a point A with coordinates (-1, 2) in the plane rectangular coordinate system xOy,
    prove that its coordinates with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (A : ℝ × ℝ) (h : A = (-1, 2)) :
  A = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3878_387807


namespace NUMINAMATH_CALUDE_system_solution_l3878_387816

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x - 1 / ((x - y)^2) + y = -10
def equation2 (x y : ℝ) : Prop := x * y = 20

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(-4, -5), (-5, -4), (-2.7972, -7.15), (-7.15, -2.7972), (4.5884, 4.3588), (4.3588, 4.5884)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solution_l3878_387816


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3878_387814

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, -2) and (-4, 8) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, -2)
  let p2 : ℝ × ℝ := (-4, 8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3878_387814


namespace NUMINAMATH_CALUDE_arrangementsWithRestrictionFor6_l3878_387868

/-- The number of ways to arrange n people in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person
    cannot be placed on either end -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * linearArrangements (n - 1)

/-- Theorem stating that the number of ways to arrange 6 people in a line,
    where one specific person cannot be placed on either end, is 480 -/
theorem arrangementsWithRestrictionFor6 :
    arrangementsWithRestriction 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangementsWithRestrictionFor6_l3878_387868


namespace NUMINAMATH_CALUDE_building_area_theorem_l3878_387845

/-- Represents a rectangular building with three floors -/
structure Building where
  breadth : ℝ
  length : ℝ
  area_per_floor : ℝ

/-- Calculates the total painting cost for the building -/
def total_painting_cost (b : Building) : ℝ :=
  b.area_per_floor * (3 + 4 + 5)

/-- Theorem: If the length is 200% more than the breadth and the total painting cost is 3160,
    then the total area of the building is 790 sq m -/
theorem building_area_theorem (b : Building) :
  b.length = 3 * b.breadth →
  total_painting_cost b = 3160 →
  3 * b.area_per_floor = 790 :=
by
  sorry

#check building_area_theorem

end NUMINAMATH_CALUDE_building_area_theorem_l3878_387845


namespace NUMINAMATH_CALUDE_exists_unstudied_planet_l3878_387806

/-- Represents a planet in the solar system -/
structure Planet where
  id : ℕ

/-- Represents the solar system with its properties -/
structure SolarSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  closest_planet : Planet → Planet
  odd_num_planets : Odd planets.card
  distinct_distances : ∀ p1 p2 p3 p4 : Planet, p1 ≠ p2 → p3 ≠ p4 → (p1, p2) ≠ (p3, p4) → distance p1 p2 ≠ distance p3 p4
  closest_is_closest : ∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 (closest_planet p1) ≤ distance p1 p2
  not_self_study : ∀ p : Planet, closest_planet p ≠ p

/-- There exists a planet not being studied by any astronomer -/
theorem exists_unstudied_planet (s : SolarSystem) : 
  ∃ p : Planet, p ∈ s.planets ∧ ∀ q : Planet, q ∈ s.planets → s.closest_planet q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unstudied_planet_l3878_387806


namespace NUMINAMATH_CALUDE_matrix_operation_result_l3878_387815

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-7, 8; 3, -5]

theorem matrix_operation_result : 
  2 • A + B = !![1, 2; 15, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l3878_387815


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3878_387849

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℚ := (-1/2)^r * binomial 6 r

-- Define the constant term as the term where the power of x is zero
def constantTerm : ℚ := generalTerm 4

-- Theorem statement
theorem constant_term_expansion :
  constantTerm = 15/16 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3878_387849


namespace NUMINAMATH_CALUDE_sum_of_naturals_equals_1035_l3878_387897

theorem sum_of_naturals_equals_1035 (n : ℕ) : (n * (n + 1)) / 2 = 1035 → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_naturals_equals_1035_l3878_387897


namespace NUMINAMATH_CALUDE_phone_sale_problem_l3878_387863

theorem phone_sale_problem (total : ℕ) (defective : ℕ) (customer_a : ℕ) (customer_b : ℕ) 
  (h_total : total = 20)
  (h_defective : defective = 5)
  (h_customer_a : customer_a = 3)
  (h_customer_b : customer_b = 5)
  (h_all_sold : total - defective = customer_a + customer_b + (total - defective - customer_a - customer_b)) :
  total - defective - customer_a - customer_b = 7 := by
  sorry

end NUMINAMATH_CALUDE_phone_sale_problem_l3878_387863


namespace NUMINAMATH_CALUDE_madison_gardensquare_distance_l3878_387846

/-- The distance between Madison and Gardensquare on the map -/
def map_distance (travel_time : ℝ) (average_speed : ℝ) (map_scale : ℝ) : ℝ :=
  travel_time * average_speed * map_scale

/-- Theorem: The distance between Madison and Gardensquare on the map is 5 inches -/
theorem madison_gardensquare_distance :
  map_distance 1.5 60 0.05555555555555555 = 5 := by
  sorry

end NUMINAMATH_CALUDE_madison_gardensquare_distance_l3878_387846


namespace NUMINAMATH_CALUDE_representative_selection_count_l3878_387864

def male_count : ℕ := 5
def female_count : ℕ := 4
def total_representatives : ℕ := 4
def min_male : ℕ := 2
def min_female : ℕ := 1

theorem representative_selection_count : 
  (Nat.choose male_count 2 * Nat.choose female_count 2) + 
  (Nat.choose male_count 3 * Nat.choose female_count 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l3878_387864


namespace NUMINAMATH_CALUDE_valentines_remaining_l3878_387883

theorem valentines_remaining (initial : ℕ) (children neighbors coworkers : ℕ) :
  initial ≥ children + neighbors + coworkers →
  initial - (children + neighbors + coworkers) =
  initial - children - neighbors - coworkers :=
by sorry

end NUMINAMATH_CALUDE_valentines_remaining_l3878_387883


namespace NUMINAMATH_CALUDE_average_student_headcount_theorem_l3878_387879

/-- Represents the student headcount for a specific academic year --/
structure StudentCount where
  year : String
  count : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (q : ℚ) : ℤ :=
  (q + 1/2).floor

theorem average_student_headcount_theorem 
  (headcounts : List StudentCount)
  (error_margin : ℕ)
  (h1 : headcounts.length = 3)
  (h2 : error_margin = 50)
  (h3 : ∀ sc ∈ headcounts, sc.count ≥ 10000 ∧ sc.count ≤ 12000) :
  roundToNearest (average (headcounts.map (λ sc ↦ sc.count))) = 10833 := by
sorry

end NUMINAMATH_CALUDE_average_student_headcount_theorem_l3878_387879


namespace NUMINAMATH_CALUDE_chocolate_heart_bags_l3878_387869

theorem chocolate_heart_bags (total_candy : ℕ) (total_bags : ℕ) (kisses_bags : ℕ) (non_chocolate_pieces : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : kisses_bags = 3)
  (h4 : non_chocolate_pieces = 28)
  (h5 : total_candy % total_bags = 0) -- Ensure equal division
  : (total_bags - kisses_bags - (non_chocolate_pieces / (total_candy / total_bags))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_heart_bags_l3878_387869


namespace NUMINAMATH_CALUDE_sequence_not_ap_gp_l3878_387835

theorem sequence_not_ap_gp (a b c d n : ℝ) : 
  a < b ∧ b < c ∧ a > 1 ∧ 
  b = a + d ∧ c = a + 2*d ∧ 
  n > 1 →
  ¬(∃r : ℝ, (Real.log n / Real.log b - Real.log n / Real.log a = r) ∧ 
             (Real.log n / Real.log c - Real.log n / Real.log b = r)) ∧
  ¬(∃r : ℝ, (Real.log n / Real.log b) / (Real.log n / Real.log a) = r ∧ 
             (Real.log n / Real.log c) / (Real.log n / Real.log b) = r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_ap_gp_l3878_387835


namespace NUMINAMATH_CALUDE_phantoms_initial_money_l3878_387860

def black_ink_cost : ℕ := 11
def red_ink_cost : ℕ := 15
def yellow_ink_cost : ℕ := 13
def black_ink_quantity : ℕ := 2
def red_ink_quantity : ℕ := 3
def yellow_ink_quantity : ℕ := 2
def additional_amount_needed : ℕ := 43

theorem phantoms_initial_money :
  black_ink_quantity * black_ink_cost +
  red_ink_quantity * red_ink_cost +
  yellow_ink_quantity * yellow_ink_cost -
  additional_amount_needed = 50 := by
    sorry

end NUMINAMATH_CALUDE_phantoms_initial_money_l3878_387860


namespace NUMINAMATH_CALUDE_painters_work_days_l3878_387867

theorem painters_work_days (painters_initial : ℕ) (painters_new : ℕ) (days_initial : ℚ) : 
  painters_initial = 5 → 
  painters_new = 4 → 
  days_initial = 3/2 → 
  ∃ (days_new : ℚ), days_new = 15/8 ∧ 
    painters_initial * days_initial = painters_new * days_new :=
by sorry

end NUMINAMATH_CALUDE_painters_work_days_l3878_387867


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l3878_387811

/-- The number of copies Sarah needs to make for each person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in the contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

/-- Theorem stating that the total number of pages Sarah will copy is 360 -/
theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l3878_387811


namespace NUMINAMATH_CALUDE_problem_solution_l3878_387803

theorem problem_solution : 
  let x := 0.47 * 1442 - 0.36 * 1412
  ∃ y, x + y = 3 ∧ y = -166.42 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3878_387803


namespace NUMINAMATH_CALUDE_binary_representation_of_51_l3878_387877

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 51 is 110011 -/
theorem binary_representation_of_51 :
  toBinary 51 = [true, true, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_51_l3878_387877


namespace NUMINAMATH_CALUDE_aubrey_gum_count_l3878_387826

theorem aubrey_gum_count (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : john_gum + cole_gum + aubrey_gum = 33 * 3) :
  aubrey_gum = 0 := by
sorry

end NUMINAMATH_CALUDE_aubrey_gum_count_l3878_387826


namespace NUMINAMATH_CALUDE_monochromatic_triangle_K17_l3878_387854

/-- A coloring of edges in a complete graph --/
def EdgeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph has a monochromatic triangle if there exist three distinct vertices
    such that all edges between them have the same color --/
def has_monochromatic_triangle (n : ℕ) (coloring : EdgeColoring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    coloring i j = coloring j k ∧ coloring j k = coloring i k

/-- Main theorem: Any 3-coloring of the edges of a complete graph with 17 vertices
    contains a monochromatic triangle --/
theorem monochromatic_triangle_K17 :
  ∀ (coloring : EdgeColoring 17), has_monochromatic_triangle 17 coloring :=
sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_K17_l3878_387854


namespace NUMINAMATH_CALUDE_train_length_l3878_387851

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 127 → time = 17 → ∃ (length : ℝ), abs (length - 599.76) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3878_387851


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3878_387872

theorem set_intersection_problem (M N : Set ℤ) : 
  M = {-1, 0, 1} → N = {0, 1, 2} → M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3878_387872


namespace NUMINAMATH_CALUDE_printer_paper_pairs_l3878_387830

theorem printer_paper_pairs : 
  ∀ x y : ℕ+, x * y = 221 ↔ (x = 1 ∧ y = 221) ∨ (x = 221 ∧ y = 1) ∨ (x = 13 ∧ y = 17) ∨ (x = 17 ∧ y = 13) :=
by sorry

end NUMINAMATH_CALUDE_printer_paper_pairs_l3878_387830
