import Mathlib

namespace thirtieth_term_of_sequence_l4098_409813

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end thirtieth_term_of_sequence_l4098_409813


namespace system_solution_ratio_l4098_409819

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y - z = 0 →
  4 * x + 2 * k * y + 3 * z = 0 →
  3 * x + 6 * y + 2 * z = 0 →
  x * z / (y^2) = 1368 / 25 := by
  sorry

end system_solution_ratio_l4098_409819


namespace girls_to_boys_ratio_l4098_409891

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 →
  difference = 6 →
  ∃ (girls boys : ℕ),
    girls = boys + difference ∧
    girls + boys = total ∧
    girls * 5 = boys * 7 :=
by sorry

end girls_to_boys_ratio_l4098_409891


namespace obtuse_triangle_one_obtuse_angle_l4098_409863

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > 90

-- Define an obtuse angle
def ObtuseAngle (angle : ℝ) : Prop := angle > 90

-- Theorem: An obtuse triangle has exactly one obtuse interior angle
theorem obtuse_triangle_one_obtuse_angle (t : Triangle) (h : ObtuseTriangle t) :
  ∃! i : Fin 3, ObtuseAngle (t.angles i) :=
sorry

end obtuse_triangle_one_obtuse_angle_l4098_409863


namespace point_conversion_value_l4098_409878

/-- Calculates the value of each point conversion in James' football season. -/
theorem point_conversion_value
  (touchdowns_per_game : ℕ)
  (points_per_touchdown : ℕ)
  (num_games : ℕ)
  (num_conversions : ℕ)
  (old_record : ℕ)
  (points_above_record : ℕ)
  (h1 : touchdowns_per_game = 4)
  (h2 : points_per_touchdown = 6)
  (h3 : num_games = 15)
  (h4 : num_conversions = 6)
  (h5 : old_record = 300)
  (h6 : points_above_record = 72) :
  (old_record + points_above_record - touchdowns_per_game * points_per_touchdown * num_games) / num_conversions = 2 := by
  sorry

end point_conversion_value_l4098_409878


namespace division_remainder_proof_l4098_409817

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (h1 : dividend = 729) (h2 : divisor = 38) (h3 : quotient = 19) :
  dividend - divisor * quotient = 7 := by
  sorry

end division_remainder_proof_l4098_409817


namespace theater_eye_colors_l4098_409881

theorem theater_eye_colors (total : ℕ) (blue brown black green : ℕ) : 
  total = 100 →
  blue = 19 →
  brown = total / 2 →
  black = total / 4 →
  green = total - (blue + brown + black) →
  green = 6 := by
sorry

end theater_eye_colors_l4098_409881


namespace marble_comparison_l4098_409872

theorem marble_comparison (katrina mabel amanda carlos diana : ℕ) : 
  mabel = 5 * katrina →
  amanda + 12 = 2 * katrina →
  carlos = 3 * katrina →
  diana = 2 * katrina + (katrina / 2) →
  mabel = 85 →
  mabel = amanda + carlos + diana - 30 :=
by
  sorry

end marble_comparison_l4098_409872


namespace square_sum_given_sum_and_product_l4098_409853

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end square_sum_given_sum_and_product_l4098_409853


namespace estate_distribution_l4098_409818

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) 
  (h1 : ∃ (d s w c : ℕ), d + s + w + c = E) 
  (h2 : ∃ (d s : ℕ), d + s = E / 2) 
  (h3 : ∃ (d s : ℕ), 3 * s = 2 * d) 
  (h4 : ∃ (d w : ℕ), w = 3 * d) 
  (h5 : ∃ (c : ℕ), c = 800) :
  E = 2000 := by
  sorry

end estate_distribution_l4098_409818


namespace two_extremum_function_properties_l4098_409834

/-- A function with two distinct extremum points -/
structure TwoExtremumFunction where
  f : ℝ → ℝ
  a : ℝ
  x1 : ℝ
  x2 : ℝ
  h_def : ∀ x, f x = x^2 + a * Real.log (x + 1)
  h_extremum : x1 < x2
  h_distinct : ∃ y, x1 < y ∧ y < x2

/-- The main theorem about the properties of the function -/
theorem two_extremum_function_properties (g : TwoExtremumFunction) :
  0 < g.a ∧ g.a < 1/2 ∧ 0 < g.f g.x2 / g.x1 ∧ g.f g.x2 / g.x1 < -1/2 + Real.log 2 := by
  sorry

end two_extremum_function_properties_l4098_409834


namespace complex_power_simplification_l4098_409807

theorem complex_power_simplification :
  (3 * (Complex.cos (30 * π / 180)) + 3 * Complex.I * (Complex.sin (30 * π / 180)))^4 =
  Complex.mk (-81/2) ((81 * Real.sqrt 3)/2) := by
sorry

end complex_power_simplification_l4098_409807


namespace student_number_problem_l4098_409838

theorem student_number_problem (x : ℝ) : 8 * x - 138 = 102 → x = 30 := by
  sorry

end student_number_problem_l4098_409838


namespace total_barking_dogs_l4098_409871

theorem total_barking_dogs (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 → additional_dogs = 10 → initial_dogs + additional_dogs = 40 := by
  sorry

end total_barking_dogs_l4098_409871


namespace perimeter_after_adding_tiles_l4098_409882

/-- Represents a square tile with unit length sides -/
structure UnitTile where
  x : ℕ
  y : ℕ

/-- Represents a figure made of unit tiles -/
structure TileFigure where
  tiles : List UnitTile

/-- Calculates the perimeter of a figure made of unit tiles -/
def perimeter (figure : TileFigure) : ℕ :=
  sorry

/-- Checks if a tile is adjacent to any tile in the figure -/
def isAdjacent (tile : UnitTile) (figure : TileFigure) : Bool :=
  sorry

theorem perimeter_after_adding_tiles :
  ∃ (original : TileFigure) (new1 new2 : UnitTile),
    (original.tiles.length = 16) ∧
    (∀ t ∈ original.tiles, t.x < 4 ∧ t.y < 4) ∧
    (isAdjacent new1 original) ∧
    (isAdjacent new2 original) ∧
    (perimeter (TileFigure.mk (new1 :: new2 :: original.tiles)) = 18) :=
  sorry

end perimeter_after_adding_tiles_l4098_409882


namespace total_new_people_count_l4098_409860

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_count : total_new_people = 106491 := by
  sorry

end total_new_people_count_l4098_409860


namespace work_left_fraction_l4098_409831

/-- The fraction of work left after two workers work together for a given number of days -/
def fraction_left (days_a : ℕ) (days_b : ℕ) (days_together : ℕ) : ℚ :=
  1 - (days_together : ℚ) * ((1 : ℚ) / days_a + (1 : ℚ) / days_b)

/-- Theorem: Given A can do a job in 15 days and B in 20 days, if they work together for 3 days, 
    the fraction of work left is 13/20 -/
theorem work_left_fraction : fraction_left 15 20 3 = 13 / 20 := by
  sorry

end work_left_fraction_l4098_409831


namespace quadratic_discriminant_l4098_409806

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-9) (-7) = 221 := by sorry

end quadratic_discriminant_l4098_409806


namespace A_3_2_equals_5_l4098_409820

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 2
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_5 : A 3 2 = 5 := by
  sorry

end A_3_2_equals_5_l4098_409820


namespace white_balls_count_l4098_409842

theorem white_balls_count (red_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  red_balls = 12 → 
  prob_white = 2/3 → 
  (white_balls : ℚ) / (white_balls + red_balls) = prob_white →
  white_balls = 24 := by
sorry

end white_balls_count_l4098_409842


namespace problem_1_l4098_409805

theorem problem_1 : (-16) + 28 + (-128) - (-66) = -50 := by
  sorry

end problem_1_l4098_409805


namespace max_rectangular_pen_area_l4098_409849

theorem max_rectangular_pen_area (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (width height : ℝ), 
    width > 0 ∧ height > 0 ∧
    2 * (width + height) = perimeter ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → 2 * (w + h) = perimeter → w * h ≤ width * height ∧
    width * height = 225 :=
by sorry

end max_rectangular_pen_area_l4098_409849


namespace complex_division_l4098_409841

theorem complex_division : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_division_l4098_409841


namespace number_of_arrangements_l4098_409866

-- Define the number of male volunteers
def num_male : Nat := 4

-- Define the number of female volunteers
def num_female : Nat := 2

-- Define the number of elderly people
def num_elderly : Nat := 2

-- Define the total number of people
def total_people : Nat := num_male + num_female + num_elderly

-- Define the function to calculate the number of arrangements
def calculate_arrangements (n_male : Nat) (n_female : Nat) (n_elderly : Nat) : Nat :=
  -- Treat elderly people as one unit
  let n_units := n_male + 1
  -- Calculate arrangements of units
  let unit_arrangements := Nat.factorial n_units
  -- Calculate arrangements of elderly people themselves
  let elderly_arrangements := Nat.factorial n_elderly
  -- Calculate arrangements of female volunteers in the spaces between and around other people
  let female_arrangements := (n_units + 1) * n_units
  unit_arrangements * elderly_arrangements * female_arrangements

-- Theorem statement
theorem number_of_arrangements :
  calculate_arrangements num_male num_female num_elderly = 7200 := by
  sorry

end number_of_arrangements_l4098_409866


namespace smallest_argument_in_circle_l4098_409832

open Complex

theorem smallest_argument_in_circle (p : ℂ) : 
  abs (p - 25 * I) ≤ 15 → arg p ≥ arg (12 + 16 * I) := by
  sorry

end smallest_argument_in_circle_l4098_409832


namespace no_isosceles_right_triangle_with_perimeter_60_l4098_409883

theorem no_isosceles_right_triangle_with_perimeter_60 :
  ¬ ∃ (a c : ℕ), 
    a > 0 ∧ 
    c > 0 ∧ 
    c * c = 2 * a * a ∧  -- Pythagorean theorem for isosceles right triangle
    2 * a + c = 60 :=    -- Perimeter condition
by sorry

end no_isosceles_right_triangle_with_perimeter_60_l4098_409883


namespace smallest_x_value_l4098_409855

theorem smallest_x_value (x : ℝ) : 
  (((14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x) = (7 * x - 2)) → x ≥ 4/5 :=
by sorry

end smallest_x_value_l4098_409855


namespace original_paper_sheets_l4098_409803

-- Define the number of sheets per book
def sheets_per_book : ℕ := sorry

-- Define the total number of sheets
def total_sheets : ℕ := 18000

-- Theorem statement
theorem original_paper_sheets :
  (120 * sheets_per_book = (60 : ℕ) * total_sheets / 100) ∧
  (185 * sheets_per_book + 1350 = total_sheets) :=
by sorry

end original_paper_sheets_l4098_409803


namespace square_side_length_l4098_409869

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = Real.sqrt 2 := by
  sorry

end square_side_length_l4098_409869


namespace sqrt_450_simplification_l4098_409843

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l4098_409843


namespace droid_coffee_usage_l4098_409889

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage (morning_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  daily_usage * days_per_week

/-- Theorem stating that Droid uses 126 bags of coffee beans per week -/
theorem droid_coffee_usage :
  weekly_coffee_usage 3 7 = 126 := by
  sorry

end droid_coffee_usage_l4098_409889


namespace parallelograms_in_hexagon_l4098_409893

/-- A regular hexagon -/
structure RegularHexagon where
  /-- The number of sides in a regular hexagon -/
  sides : Nat
  /-- The property that a regular hexagon has 6 sides -/
  has_six_sides : sides = 6

/-- A parallelogram formed by two adjacent equilateral triangles in a regular hexagon -/
structure Parallelogram (h : RegularHexagon) where

/-- The number of parallelograms in a regular hexagon -/
def num_parallelograms (h : RegularHexagon) : Nat :=
  h.sides

/-- Theorem: The number of parallelograms in a regular hexagon is 6 -/
theorem parallelograms_in_hexagon (h : RegularHexagon) : 
  num_parallelograms h = 6 := by
  sorry

end parallelograms_in_hexagon_l4098_409893


namespace hexagon_pattern_triangle_area_l4098_409894

/-- The area of a triangle formed by centers of alternate hexagons in a hexagonal pattern -/
theorem hexagon_pattern_triangle_area :
  ∀ (hexagon_side_length : ℝ) (triangle_side_length : ℝ),
    hexagon_side_length = 1 →
    triangle_side_length = 3 * hexagon_side_length →
    ∃ (triangle_area : ℝ),
      triangle_area = (9 * Real.sqrt 3) / 4 ∧
      triangle_area = (Real.sqrt 3 / 4) * triangle_side_length^2 :=
by sorry

end hexagon_pattern_triangle_area_l4098_409894


namespace board_cutting_theorem_l4098_409845

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end board_cutting_theorem_l4098_409845


namespace linear_function_composition_l4098_409809

/-- Given f(x) = x^2 - 2x + 1 and g(x) is a linear function such that f[g(x)] = 4x^2,
    prove that g(x) = 2x + 1 or g(x) = -2x + 1 -/
theorem linear_function_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2 - 2*x + 1) →
  (∃ a b : ℝ, ∀ x, g x = a*x + b) →
  (∀ x, f (g x) = 4 * x^2) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x + 1) := by
  sorry

end linear_function_composition_l4098_409809


namespace floor_equality_iff_interval_l4098_409899

theorem floor_equality_iff_interval (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end floor_equality_iff_interval_l4098_409899


namespace small_boxes_count_l4098_409810

theorem small_boxes_count (total_bars : ℕ) (bars_per_box : ℕ) (h1 : total_bars = 640) (h2 : bars_per_box = 32) :
  total_bars / bars_per_box = 20 := by
  sorry

end small_boxes_count_l4098_409810


namespace select_five_from_ten_l4098_409870

theorem select_five_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → Nat.choose n k = 252 := by
  sorry

end select_five_from_ten_l4098_409870


namespace equation_unique_solution_l4098_409867

/-- The function representing the left-hand side of the equation -/
def f (y : ℝ) : ℝ := (30 * y + (30 * y + 25) ^ (1/3)) ^ (1/3)

/-- The theorem stating that the equation has a unique solution -/
theorem equation_unique_solution :
  ∃! y : ℝ, f y = 15 ∧ y = 335/3 := by sorry

end equation_unique_solution_l4098_409867


namespace coin_identification_strategy_exists_l4098_409875

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
(id : Nat)
(is_genuine : Bool)

/-- Represents the scale used for weighing -/
def Scale := List Coin → List Coin → WeighingResult

/-- Represents the strategy for identifying genuine coins -/
def IdentificationStrategy := Scale → List Coin → List Coin

theorem coin_identification_strategy_exists :
  ∀ (coins : List Coin),
    coins.length = 8 →
    (coins.filter (λ c => c.is_genuine)).length = 7 →
    ∃ (strategy : IdentificationStrategy),
      ∀ (scale : Scale),
        let identified := strategy scale coins
        identified.length ≥ 5 ∧
        ∀ c ∈ identified, c.is_genuine ∧
        ∀ c ∈ identified, c ∉ (coins.filter (λ c => ¬c.is_genuine)) :=
by sorry

end coin_identification_strategy_exists_l4098_409875


namespace complex_fraction_simplification_l4098_409861

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + 5*i) = (1:ℂ) / 29 - (17:ℂ) / 29 * i :=
by sorry

end complex_fraction_simplification_l4098_409861


namespace odd_numbers_product_equality_l4098_409877

theorem odd_numbers_product_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end odd_numbers_product_equality_l4098_409877


namespace paulines_garden_tomato_kinds_l4098_409840

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  remaining_spaces : ℕ

/-- Theorem representing the problem -/
theorem paulines_garden_tomato_kinds (g : Garden) 
  (h1 : g.rows = 10)
  (h2 : g.spaces_per_row = 15)
  (h3 : g.tomatoes_per_kind = 5)
  (h4 : g.cucumber_kinds = 5)
  (h5 : g.cucumbers_per_kind = 4)
  (h6 : g.potatoes = 30)
  (h7 : g.remaining_spaces = 85)
  (h8 : g.rows * g.spaces_per_row = 
        g.tomato_kinds * g.tomatoes_per_kind + 
        g.cucumber_kinds * g.cucumbers_per_kind + 
        g.potatoes + g.remaining_spaces) : 
  g.tomato_kinds = 3 := by
  sorry

end paulines_garden_tomato_kinds_l4098_409840


namespace largest_common_divisor_540_315_l4098_409822

theorem largest_common_divisor_540_315 : Nat.gcd 540 315 = 45 := by
  sorry

end largest_common_divisor_540_315_l4098_409822


namespace certain_number_proof_l4098_409850

theorem certain_number_proof (n x : ℝ) (h1 : n = -4.5) (h2 : 10 * n = x - 2 * n) : x = -54 := by
  sorry

end certain_number_proof_l4098_409850


namespace teal_greenish_count_teal_greenish_proof_l4098_409844

def total_surveyed : ℕ := 120
def kinda_blue : ℕ := 70
def both : ℕ := 35
def neither : ℕ := 20

theorem teal_greenish_count : ℕ :=
  total_surveyed - (kinda_blue - both) - both - neither
  
theorem teal_greenish_proof : teal_greenish_count = 65 := by
  sorry

end teal_greenish_count_teal_greenish_proof_l4098_409844


namespace limit_equals_six_l4098_409830

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_six 
  (h : deriv f 2 = 3) : 
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ ≠ 0, |x₀| < δ → 
    |((f (2 + 2*x₀) - f 2) / x₀) - 6| < ε :=
sorry

end limit_equals_six_l4098_409830


namespace chess_tournament_games_l4098_409880

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 8 players, where each player plays every other player once,
    the total number of games played is 28. -/
theorem chess_tournament_games :
  gamesPlayed 8 = 28 := by
  sorry

#eval gamesPlayed 8  -- This should output 28

end chess_tournament_games_l4098_409880


namespace ratio_comparison_correct_l4098_409814

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_water_ratio : ℚ := 1 / 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 60

/-- The ratio of (flavoring to corn syrup in sport formulation) to (flavoring to corn syrup in standard formulation) -/
def ratio_comparison : ℚ := 3

/-- Theorem stating that the ratio comparison is correct given the problem conditions -/
theorem ratio_comparison_correct : 
  let standard_flavoring_to_corn := standard_ratio 0 / standard_ratio 1
  let sport_flavoring := sport_water * (sport_water_ratio * (standard_ratio 0 / standard_ratio 2))
  let sport_flavoring_to_corn := sport_flavoring / sport_corn_syrup
  (sport_flavoring_to_corn / standard_flavoring_to_corn) = ratio_comparison := by
  sorry

end ratio_comparison_correct_l4098_409814


namespace male_average_grade_l4098_409829

/-- Proves that the average grade of male students is 87 given the conditions of the problem -/
theorem male_average_grade (total_average : ℝ) (female_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : female_average = 92)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 87 := by
  sorry

end male_average_grade_l4098_409829


namespace square_real_implies_a_zero_l4098_409868

theorem square_real_implies_a_zero (a : ℝ) : 
  (Complex.I * a + 2) ^ 2 ∈ Set.range Complex.ofReal → a = 0 := by
  sorry

end square_real_implies_a_zero_l4098_409868


namespace min_additional_squares_for_symmetry_l4098_409837

-- Define the grid dimensions
def grid_width : Nat := 4
def grid_height : Nat := 5

-- Define a type for grid positions
structure GridPosition where
  x : Nat
  y : Nat

-- Define the initially shaded squares
def initial_shaded : List GridPosition := [
  { x := 1, y := 4 },
  { x := 4, y := 1 }
]

-- Define a function to check if a position is within the grid
def is_valid_position (pos : GridPosition) : Prop :=
  pos.x ≥ 0 ∧ pos.x < grid_width ∧ pos.y ≥ 0 ∧ pos.y < grid_height

-- Define a function to check if a list of positions creates horizontal and vertical symmetry
def is_symmetric (shaded : List GridPosition) : Prop :=
  ∀ pos : GridPosition, is_valid_position pos →
    (pos ∈ shaded ↔ 
     { x := grid_width - 1 - pos.x, y := pos.y } ∈ shaded ∧
     { x := pos.x, y := grid_height - 1 - pos.y } ∈ shaded)

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additional : List GridPosition),
    (∀ pos ∈ additional, pos ∉ initial_shaded) ∧
    is_symmetric (initial_shaded ++ additional) ∧
    additional.length = 6 ∧
    (∀ (other : List GridPosition),
      (∀ pos ∈ other, pos ∉ initial_shaded) →
      is_symmetric (initial_shaded ++ other) →
      other.length ≥ 6) :=
sorry

end min_additional_squares_for_symmetry_l4098_409837


namespace two_consecutive_sets_sum_100_l4098_409890

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_100 : start * length + (length * (length - 1)) / 2 = 100
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly two sets of consecutive positive integers
    whose sum is 100 and contain at least two integers -/
theorem two_consecutive_sets_sum_100 :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 100) :=
sorry

end two_consecutive_sets_sum_100_l4098_409890


namespace diego_extra_cans_l4098_409846

theorem diego_extra_cans (martha_cans : ℕ) (total_cans : ℕ) (diego_cans : ℕ) : 
  martha_cans = 90 →
  total_cans = 145 →
  diego_cans = total_cans - martha_cans →
  diego_cans - martha_cans / 2 = 10 :=
by sorry

end diego_extra_cans_l4098_409846


namespace investment_interest_rate_l4098_409824

/-- Proves that the interest rate for the first part of an investment is 3% given the specified conditions --/
theorem investment_interest_rate : 
  ∀ (total_amount first_part second_part first_rate second_rate total_interest : ℚ),
  total_amount = 4000 →
  first_part = 2800 →
  second_part = total_amount - first_part →
  second_rate = 5 →
  (first_part * first_rate / 100 + second_part * second_rate / 100) = total_interest →
  total_interest = 144 →
  first_rate = 3 := by
sorry

end investment_interest_rate_l4098_409824


namespace donny_money_left_l4098_409833

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem donny_money_left : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end donny_money_left_l4098_409833


namespace a_5_equals_20_l4098_409800

def S (n : ℕ) : ℕ := 2 * n * (n + 1)

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_5_equals_20 : a 5 = 20 := by
  sorry

end a_5_equals_20_l4098_409800


namespace contrapositive_example_l4098_409815

theorem contrapositive_example (a b : ℝ) :
  (∀ a b, a > b → a - 5 > b - 5) ↔ (∀ a b, a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_example_l4098_409815


namespace infinite_pairs_exist_l4098_409858

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Predicate to check if m divides n^2 + 1 and n divides m^2 + 1 -/
def satisfies_condition (m n : ℕ) : Prop :=
  (n^2 + 1) % m = 0 ∧ (m^2 + 1) % n = 0

theorem infinite_pairs_exist :
  ∀ k : ℕ, ∃ m n : ℕ,
    m > k ∧
    n > k ∧
    satisfies_condition m n ∧
    m = fib (2 * n + 1) ∧
    n = fib (2 * n - 1) :=
sorry

end infinite_pairs_exist_l4098_409858


namespace tangent_line_equation_l4098_409836

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1

theorem tangent_line_equation (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f) x₀ = 2 →
  ∃ y₀ : ℝ, y₀ = f x₀ ∧ 2 * x - y - Real.exp + 1 = 0 :=
by sorry

end tangent_line_equation_l4098_409836


namespace triangle_horizontal_line_l4098_409884

/-- Given two intersecting lines and the area of the triangle they form with the x-axis,
    prove the equation of the horizontal line that completes this triangle. -/
theorem triangle_horizontal_line
  (line1 : ℝ → ℝ)
  (line2 : ℝ)
  (area : ℝ)
  (h1 : ∀ x, line1 x = x)
  (h2 : line2 = -9)
  (h3 : area = 40.5)
  : ∃ y : ℝ, y = 9 ∧ 
    (1/2 : ℝ) * |line2| * y = area ∧
    (line1 (-line2) = y) :=
by sorry

end triangle_horizontal_line_l4098_409884


namespace arithmetic_sequence_length_l4098_409812

theorem arithmetic_sequence_length (a₁ l d : ℕ) (h : l = a₁ + (n - 1) * d) :
  a₁ = 4 → l = 205 → d = 3 → n = 68 := by
  sorry

end arithmetic_sequence_length_l4098_409812


namespace multiply_ones_seven_l4098_409826

theorem multiply_ones_seven : 1111111 * 1111111 = 1234567654321 := by
  sorry

end multiply_ones_seven_l4098_409826


namespace smallest_dual_base_representation_l4098_409879

/-- Represents a number in base 6 as XX₆ -/
def base6 (x : ℕ) : ℕ := 6 * x + x

/-- Represents a number in base 8 as YY₈ -/
def base8 (y : ℕ) : ℕ := 8 * y + y

/-- Checks if a digit is valid in base 6 -/
def validBase6Digit (x : ℕ) : Prop := x ≤ 5

/-- Checks if a digit is valid in base 8 -/
def validBase8Digit (y : ℕ) : Prop := y ≤ 7

theorem smallest_dual_base_representation :
  ∃ (x y : ℕ), validBase6Digit x ∧ validBase8Digit y ∧
    base6 x = base8 y ∧
    base6 x = 63 ∧
    (∀ (x' y' : ℕ), validBase6Digit x' → validBase8Digit y' →
      base6 x' = base8 y' → base6 x' ≥ 63) :=
sorry

end smallest_dual_base_representation_l4098_409879


namespace fixed_point_of_exponential_function_l4098_409852

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) - 3
  f 2 = -2 := by sorry

end fixed_point_of_exponential_function_l4098_409852


namespace pirate_treasure_probability_l4098_409888

theorem pirate_treasure_probability :
  let n_islands : ℕ := 8
  let n_treasure : ℕ := 4
  let p_treasure : ℚ := 1/3
  let p_traps : ℚ := 1/6
  let p_neither : ℚ := 1/2
  let choose := fun (n k : ℕ) => (Nat.choose n k : ℚ)
  
  (choose n_islands n_treasure) * p_treasure^n_treasure * p_neither^(n_islands - n_treasure) = 35/648 :=
by sorry

end pirate_treasure_probability_l4098_409888


namespace ratio_sum_theorem_l4098_409873

theorem ratio_sum_theorem (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end ratio_sum_theorem_l4098_409873


namespace max_students_planting_trees_l4098_409801

theorem max_students_planting_trees (a b : ℕ) : 
  3 * a + 5 * b = 115 → a + b ≤ 37 := by
  sorry

end max_students_planting_trees_l4098_409801


namespace sum_256_125_base5_l4098_409886

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_256_125_base5 :
  addBase5 (toBase5 256) (toBase5 125) = [3, 0, 1, 1] :=
sorry

end sum_256_125_base5_l4098_409886


namespace total_oil_needed_l4098_409827

def oil_for_wheels : ℕ := 2 * 15
def oil_for_chain : ℕ := 10
def oil_for_pedals : ℕ := 5
def oil_for_brakes : ℕ := 8

theorem total_oil_needed : 
  oil_for_wheels + oil_for_chain + oil_for_pedals + oil_for_brakes = 53 := by
  sorry

end total_oil_needed_l4098_409827


namespace sum_of_integers_l4098_409848

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
  (x : ℝ) + y = Real.sqrt 565 := by
  sorry

end sum_of_integers_l4098_409848


namespace year_spans_53_or_54_weeks_l4098_409821

/-- A year is either common (365 days) or leap (366 days) -/
inductive Year
  | Common
  | Leap

/-- Definition of how many days are in a year -/
def daysInYear (y : Year) : ℕ :=
  match y with
  | Year.Common => 365
  | Year.Leap => 366

/-- Definition of when a year covers a week -/
def yearCoversWeek (daysInYear : ℕ) (weekStartDay : ℕ) : Prop :=
  daysInYear - weekStartDay ≥ 6

/-- Theorem stating that a year can span either 53 or 54 weeks -/
theorem year_spans_53_or_54_weeks (y : Year) :
  ∃ (n : ℕ), (n = 53 ∨ n = 54) ∧
    (∀ (w : ℕ), w ≤ n → yearCoversWeek (daysInYear y) ((w - 1) * 7)) ∧
    (∀ (w : ℕ), w > n → ¬yearCoversWeek (daysInYear y) ((w - 1) * 7)) :=
  sorry

end year_spans_53_or_54_weeks_l4098_409821


namespace two_digit_number_problem_l4098_409854

theorem two_digit_number_problem : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n % 10 = n / 10 + 4) ∧ 
  (n * (n / 10 + n % 10) = 208) ∧
  n = 26 := by
  sorry

end two_digit_number_problem_l4098_409854


namespace sum_first_15_odd_integers_l4098_409864

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun i => 2 * i + 1) = 225 := by
  sorry

end sum_first_15_odd_integers_l4098_409864


namespace hyperbola_asymptotes_l4098_409897

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2/a^2 - y^2/3 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- State that the focus of the parabola is the right focus of the hyperbola
axiom focus_equality : ∃ a : ℝ, hyperbola (parabola_focus.1) (parabola_focus.2) a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y a : ℝ, parabola x y → hyperbola x y a → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l4098_409897


namespace right_triangle_inequality_l4098_409856

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_prop : h * c = a * b  -- property of altitude in right triangle

-- State the theorem
theorem right_triangle_inequality (t : RightTriangle) : t.a + t.b < t.c + t.h := by
  sorry

end right_triangle_inequality_l4098_409856


namespace vet_recommendation_difference_l4098_409865

/-- Given a total number of vets and percentages recommending two different brands of dog food,
    prove that the difference in the number of vets recommending each brand is as expected. -/
theorem vet_recommendation_difference
  (total_vets : ℕ)
  (puppy_kibble_percent : ℚ)
  (yummy_dog_kibble_percent : ℚ)
  (h_total : total_vets = 1000)
  (h_puppy : puppy_kibble_percent = 1/5)
  (h_yummy : yummy_dog_kibble_percent = 3/10) :
  (total_vets : ℚ) * yummy_dog_kibble_percent - (total_vets : ℚ) * puppy_kibble_percent = 100 :=
by sorry


end vet_recommendation_difference_l4098_409865


namespace john_donation_increases_average_l4098_409892

/-- Represents the donation amounts of Alice, Bob, and Carol -/
structure Donations where
  alice : ℝ
  bob : ℝ
  carol : ℝ

/-- The conditions of the problem -/
def donation_conditions (d : Donations) : Prop :=
  d.alice > 0 ∧ d.bob > 0 ∧ d.carol > 0 ∧  -- Each student donated a positive amount
  d.alice ≠ d.bob ∧ d.alice ≠ d.carol ∧ d.bob ≠ d.carol ∧  -- Each student donated a different amount
  d.alice / d.bob = 3 / 2 ∧  -- Ratio of Alice's to Bob's donation is 3:2
  d.carol / d.bob = 5 / 2 ∧  -- Ratio of Carol's to Bob's donation is 5:2
  d.alice + d.bob = 120  -- Sum of Alice's and Bob's donations is $120

/-- John's donation -/
def john_donation (d : Donations) : ℝ :=
  240

/-- The theorem to be proved -/
theorem john_donation_increases_average (d : Donations) 
  (h : donation_conditions d) : 
  (d.alice + d.bob + d.carol + john_donation d) / 4 = 
  1.5 * (d.alice + d.bob + d.carol) / 3 := by
  sorry

end john_donation_increases_average_l4098_409892


namespace hide_and_seek_players_l4098_409816

structure Friends where
  Andrew : Prop
  Boris : Prop
  Vasya : Prop
  Gena : Prop
  Denis : Prop

def consistent (f : Friends) : Prop :=
  (f.Andrew → (f.Boris ∧ ¬f.Vasya)) ∧
  (f.Boris → (f.Gena ∨ f.Denis)) ∧
  (¬f.Vasya → (¬f.Boris ∧ ¬f.Denis)) ∧
  (¬f.Andrew → (f.Boris ∧ ¬f.Gena))

theorem hide_and_seek_players :
  ∀ f : Friends, consistent f → (f.Boris ∧ f.Vasya ∧ f.Denis ∧ ¬f.Andrew ∧ ¬f.Gena) :=
by sorry

end hide_and_seek_players_l4098_409816


namespace right_triangle_product_divisible_by_60_l4098_409876

theorem right_triangle_product_divisible_by_60 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  60 ∣ (a * b * c) :=
sorry

end right_triangle_product_divisible_by_60_l4098_409876


namespace quadratic_degeneracy_l4098_409808

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of an equation -/
inductive Root
  | Finite (x : ℝ)
  | Infinity

/-- 
Given a quadratic equation ax² + bx + c = 0 where a = 0,
prove that it has one finite root -c/b and one root at infinity.
-/
theorem quadratic_degeneracy (eq : QuadraticEquation) (h : eq.a = 0) :
  ∃ (r₁ r₂ : Root), 
    r₁ = Root.Finite (-eq.c / eq.b) ∧ 
    r₂ = Root.Infinity ∧
    eq.b * (-eq.c / eq.b) + eq.c = 0 := by
  sorry

end quadratic_degeneracy_l4098_409808


namespace symmetry_y_axis_coordinates_l4098_409802

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetry_y_axis_coordinates :
  let A : Point := { x := -1, y := 2 }
  let B : Point := symmetricYAxis A
  B.x = 1 ∧ B.y = 2 := by
  sorry

end symmetry_y_axis_coordinates_l4098_409802


namespace school_population_proof_l4098_409895

theorem school_population_proof :
  ∀ (n : ℕ) (senior_class : ℕ) (total_selected : ℕ) (other_selected : ℕ),
  senior_class = 900 →
  total_selected = 20 →
  other_selected = 14 →
  (total_selected - other_selected : ℚ) / senior_class = total_selected / n →
  n = 3000 := by
sorry

end school_population_proof_l4098_409895


namespace sweets_remaining_problem_l4098_409825

/-- The number of sweets remaining in a packet after some are eaten and given away -/
def sweets_remaining (cherry strawberry pineapple : ℕ) : ℕ :=
  let total := cherry + strawberry + pineapple
  let eaten := (cherry / 2) + (strawberry / 2) + (pineapple / 2)
  let given_away := 5
  total - eaten - given_away

/-- Theorem stating the number of sweets remaining in the packet -/
theorem sweets_remaining_problem :
  sweets_remaining 30 40 50 = 55 := by
  sorry

end sweets_remaining_problem_l4098_409825


namespace binomial_coefficient_equality_l4098_409857

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end binomial_coefficient_equality_l4098_409857


namespace max_parallelograms_in_hexagon_l4098_409896

-- Define the regular hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the parallelogram
def parallelogram (side1 : ℝ) (side2 : ℝ) (angle1 : ℝ) (angle2 : ℝ) : Set (ℝ × ℝ) := sorry

-- Define a function to count non-overlapping parallelograms in a hexagon
def count_parallelograms (h : Set (ℝ × ℝ)) (p : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem max_parallelograms_in_hexagon :
  let h := regular_hexagon 3
  let p := parallelogram 1 2 (π/3) (2*π/3)
  count_parallelograms h p = 12 := by sorry

end max_parallelograms_in_hexagon_l4098_409896


namespace geometric_sequence_a9_l4098_409804

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a9 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 6 = 9 → a 9 = 27 := by
  sorry

end geometric_sequence_a9_l4098_409804


namespace relay_race_arrangements_l4098_409898

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem relay_race_arrangements : permutations 4 = 24 := by
  sorry

end relay_race_arrangements_l4098_409898


namespace prob_no_defective_bulbs_l4098_409851

/-- The probability of selecting 4 non-defective bulbs out of 10 bulbs, where 4 are defective -/
theorem prob_no_defective_bulbs (total : ℕ) (defective : ℕ) (select : ℕ) :
  total = 10 →
  defective = 4 →
  select = 4 →
  (Nat.choose (total - defective) select : ℚ) / (Nat.choose total select : ℚ) = 1 / 14 := by
  sorry

end prob_no_defective_bulbs_l4098_409851


namespace water_jar_problem_l4098_409885

theorem water_jar_problem (small_jar large_jar : ℝ) 
  (h1 : small_jar > 0) 
  (h2 : large_jar > 0) 
  (h3 : small_jar * (1/4) = large_jar * (1/5)) : 
  (1/5) * small_jar + (1/4) * large_jar = (1/2) * large_jar := by
  sorry

end water_jar_problem_l4098_409885


namespace absolute_value_equality_l4098_409823

theorem absolute_value_equality (x : ℝ) : |x + 6| = -(x + 6) ↔ x ≤ -6 := by
  sorry

end absolute_value_equality_l4098_409823


namespace no_perfect_square_203_base_n_l4098_409835

theorem no_perfect_square_203_base_n : 
  ¬ ∃ n : ℤ, 4 ≤ n ∧ n ≤ 18 ∧ ∃ k : ℤ, 2 * n^2 + 3 = k^2 :=
sorry

end no_perfect_square_203_base_n_l4098_409835


namespace ring_endomorphism_properties_division_ring_commutativity_l4098_409874

structure RingWithEndomorphism (R : Type) [Ring R] :=
  (f : R → R)
  (f_surjective : Function.Surjective f)
  (f_hom : ∀ x y, f (x + y) = f x + f y)
  (f_hom_mul : ∀ x y, f (x * y) = f x * f y)
  (f_commutes : ∀ x, x * f x = f x * x)

theorem ring_endomorphism_properties {R : Type} [Ring R] (S : RingWithEndomorphism R) :
  (∀ x y : R, x * S.f y - S.f y * x = S.f x * y - y * S.f x) ∧
  (∀ x y : R, x * (x * y - y * x) = S.f x * (x * y - y * x)) :=
sorry

theorem division_ring_commutativity {R : Type} [DivisionRing R] (S : RingWithEndomorphism R) :
  (∃ x : R, S.f x ≠ x) → (∀ a b : R, a * b = b * a) :=
sorry

end ring_endomorphism_properties_division_ring_commutativity_l4098_409874


namespace work_completion_time_l4098_409839

/-- Given that:
    1. A can complete the work in 15 days
    2. A and B working together for 5 days complete 0.5833333333333334 of the work
    Prove that B can complete the work alone in 20 days -/
theorem work_completion_time (a_time : ℝ) (b_time : ℝ) 
  (h1 : a_time = 15)
  (h2 : 5 * (1 / a_time + 1 / b_time) = 0.5833333333333334) :
  b_time = 20 := by
  sorry

end work_completion_time_l4098_409839


namespace amazon_pack_price_is_correct_l4098_409859

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 1.75

/-- The number of lighters Amanda wants to buy -/
def num_lighters : ℕ := 24

/-- The amount Amanda saves by buying online in dollars -/
def savings : ℚ := 32

/-- The cost of a pack of twelve lighters on Amazon in dollars -/
def amazon_pack_price : ℚ := 5

theorem amazon_pack_price_is_correct :
  amazon_pack_price = 5 ∧
  2 * amazon_pack_price = num_lighters * gas_station_price - savings :=
by sorry

end amazon_pack_price_is_correct_l4098_409859


namespace lcm_problem_l4098_409828

theorem lcm_problem (e n : ℕ) : 
  e > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm e n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ e) →
  n = 230 := by
sorry

end lcm_problem_l4098_409828


namespace arithmetic_sequence_cosine_l4098_409811

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
sorry

end arithmetic_sequence_cosine_l4098_409811


namespace optimal_rectangle_area_l4098_409847

theorem optimal_rectangle_area 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 360) 
  (h_min_length : min_length = 90) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    length * width = 8100 ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      l * w ≤ 8100 :=
sorry

end optimal_rectangle_area_l4098_409847


namespace cow_husk_consumption_l4098_409862

/-- Given that 26 cows eat 26 bags of husk in 26 days, prove that one cow will eat one bag of husk in 26 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 26 ∧ bags = 26 ∧ days = 26) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end cow_husk_consumption_l4098_409862


namespace direct_variation_with_constant_l4098_409887

/-- A function that varies directly as x plus a constant -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(5) = 10 and f(1) = 6, then f(7) = 12 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 5 = 10) 
  (h2 : f k c 1 = 6) : 
  f k c 7 = 12 := by
  sorry

#check direct_variation_with_constant

end direct_variation_with_constant_l4098_409887
