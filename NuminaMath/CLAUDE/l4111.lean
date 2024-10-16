import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_friends_l4111_411193

theorem restaurant_friends (initial_wings : ℕ) (cooked_wings : ℕ) (wings_per_person : ℕ) : 
  initial_wings = 9 →
  cooked_wings = 7 →
  wings_per_person = 4 →
  (initial_wings + cooked_wings) / wings_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_friends_l4111_411193


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4111_411113

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 + Complex.I) → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4111_411113


namespace NUMINAMATH_CALUDE_pipe_fill_time_l4111_411165

/-- Given three pipes P, Q, and R that can fill a tank, this theorem proves
    that if P fills the tank in 2 hours, Q in 4 hours, and all pipes together
    in 1.2 hours, then R fills the tank in 12 hours. -/
theorem pipe_fill_time (fill_rate_P fill_rate_Q fill_rate_R : ℝ) : 
  fill_rate_P = 1 / 2 →
  fill_rate_Q = 1 / 4 →
  fill_rate_P + fill_rate_Q + fill_rate_R = 1 / 1.2 →
  fill_rate_R = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l4111_411165


namespace NUMINAMATH_CALUDE_cube_edge_length_is_sqrt_3_l4111_411172

/-- The edge length of a cube inscribed in a sphere with volume 9π/2 --/
def cube_edge_length (s : Real) (c : Real) : Prop :=
  -- All vertices of the cube are on the surface of the sphere
  -- The volume of the sphere is 9π/2
  (4 / 3 * Real.pi * s^3 = 9 * Real.pi / 2) ∧
  -- The space diagonal of the cube is the diameter of the sphere
  (Real.sqrt 3 * c = 2 * s) →
  c = Real.sqrt 3

/-- Theorem stating that the edge length of the cube is √3 --/
theorem cube_edge_length_is_sqrt_3 :
  ∃ (s : Real) (c : Real), cube_edge_length s c :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_sqrt_3_l4111_411172


namespace NUMINAMATH_CALUDE_sales_solution_l4111_411101

def sales_problem (sales : List ℕ) (average : ℕ) : Prop :=
  sales.length = 4 ∧ 
  (sales.sum + (average * 5 - sales.sum)) / 5 = average

theorem sales_solution (sales : List ℕ) (average : ℕ) 
  (h : sales_problem sales average) : 
  average * 5 - sales.sum = (average * 5 - sales.sum) := by sorry

end NUMINAMATH_CALUDE_sales_solution_l4111_411101


namespace NUMINAMATH_CALUDE_rectangle_hall_dimension_difference_l4111_411187

/-- Given a rectangular hall with area 750 m² and length 30 m, 
    prove that the length is 5 m more than the breadth. -/
theorem rectangle_hall_dimension_difference : 
  ∀ (breadth : ℝ),
    breadth > 0 →
    30 * breadth = 750 →
    30 - breadth = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hall_dimension_difference_l4111_411187


namespace NUMINAMATH_CALUDE_expression_value_l4111_411118

theorem expression_value (a : ℝ) (h : a ≠ 0) : (20 * a^5) * (8 * a^4) * (1 / (4 * a^3)^3) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4111_411118


namespace NUMINAMATH_CALUDE_four_char_word_count_l4111_411117

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (excluding 'Y') -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of four-character words formed by arranging two consonants and two vowels
    in the order consonant-vowel-vowel-consonant -/
def word_count : ℕ := consonant_count * vowel_count * vowel_count * consonant_count

theorem four_char_word_count : word_count = 11025 := by
  sorry

end NUMINAMATH_CALUDE_four_char_word_count_l4111_411117


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l4111_411159

/-- 
Given an isosceles triangle with height H that is twice as long as 
its projection on the lateral side, prove that its area is H^2 * √3.
-/
theorem isosceles_triangle_area (H : ℝ) (h : H > 0) : 
  let projection := H / 2
  let base := 2 * H * Real.sqrt 3
  let area := (1 / 2) * base * H
  area = H^2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l4111_411159


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4111_411148

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 1} = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4111_411148


namespace NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l4111_411142

/-- A hexagon with angles in arithmetic progression has one angle of 114 degrees. -/
theorem hexagon_arithmetic_progression_angle (a d : ℝ) : 
  (6 * a + 15 * d = 720) →  -- Sum of angles in hexagon
  (∃ k : ℕ, k < 6 ∧ a + k * d = 114) :=  -- One angle is 114 degrees
by sorry

end NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l4111_411142


namespace NUMINAMATH_CALUDE_rectangle_area_with_squares_l4111_411141

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_squares (s : ℝ) (h : s > 0) : 
  let small_square_area := s^2
  let large_square_area := (3*s)^2
  let total_area := 2 * small_square_area + large_square_area
  total_area = 11 * s^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_squares_l4111_411141


namespace NUMINAMATH_CALUDE_ball_probabilities_l4111_411189

/-- A bag containing balls of two colors -/
structure BallBag where
  total : ℕ
  red : ℕ
  black : ℕ
  h_sum : red + black = total

/-- The probability of drawing a ball of a specific color -/
def prob_draw_color (bag : BallBag) (color : ℕ) : ℚ :=
  color / bag.total

/-- The probability of drawing two balls of a specific color without replacement -/
def prob_draw_two_color (bag : BallBag) (color : ℕ) : ℚ :=
  (color * (color - 1)) / (bag.total * (bag.total - 1))

theorem ball_probabilities (bag : BallBag) 
  (h_total : bag.total = 5)
  (h_red : bag.red = 2)
  (h_black : bag.black = 3) :
  prob_draw_color bag bag.red = 2/5 ∧ 
  prob_draw_two_color bag bag.black = 3/10 := by
  sorry

#check ball_probabilities

end NUMINAMATH_CALUDE_ball_probabilities_l4111_411189


namespace NUMINAMATH_CALUDE_max_pencils_l4111_411198

theorem max_pencils (red_cost blue_cost max_amount : ℕ) 
  (h_red_cost : red_cost = 27)
  (h_blue_cost : blue_cost = 23)
  (h_max_amount : max_amount = 940) :
  ∃ (red blue : ℕ), 
    red_cost * red + blue_cost * blue ≤ max_amount ∧ 
    blue - red ≤ 10 ∧
    red + blue = 38 ∧
    red = 14 ∧ 
    blue = 24 ∧
    ∀ (r b : ℕ), 
      red_cost * r + blue_cost * b ≤ max_amount → 
      b - r ≤ 10 → 
      r + b ≤ red + blue :=
by
  sorry

end NUMINAMATH_CALUDE_max_pencils_l4111_411198


namespace NUMINAMATH_CALUDE_part_one_part_two_l4111_411107

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, g b x₁ = f 3 x₂) →
  b ∈ Set.Icc (-1) (1/2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4111_411107


namespace NUMINAMATH_CALUDE_investment_interest_rate_calculation_l4111_411116

theorem investment_interest_rate_calculation 
  (total_investment : ℝ) 
  (known_rate : ℝ) 
  (unknown_investment : ℝ) 
  (income_difference : ℝ) :
  let known_investment := total_investment - unknown_investment
  let unknown_rate := (known_investment * known_rate - income_difference) / unknown_investment
  total_investment = 2000 ∧ 
  known_rate = 0.10 ∧ 
  unknown_investment = 800 ∧ 
  income_difference = 56 → 
  unknown_rate = 0.08 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_calculation_l4111_411116


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4111_411140

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) (h1 : i ^ 2 = -1) (h2 : (1 + i) * z = i) : 
  z.im = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4111_411140


namespace NUMINAMATH_CALUDE_parallel_line_intersection_parallel_planes_intersection_l4111_411105

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line)

-- Theorem 1
theorem parallel_line_intersection 
  (l m : Line) (α β : Plane) :
  parallel_line_plane l α →
  subset l β →
  intersect α β = m →
  parallel l m := by sorry

-- Theorem 2
theorem parallel_planes_intersection 
  (l m : Line) (α β γ : Plane) :
  parallel_plane α β →
  intersect α γ = l →
  intersect β γ = m →
  parallel l m := by sorry

end NUMINAMATH_CALUDE_parallel_line_intersection_parallel_planes_intersection_l4111_411105


namespace NUMINAMATH_CALUDE_regular_polygon_with_120_degree_interior_angle_l4111_411154

theorem regular_polygon_with_120_degree_interior_angle :
  ∀ n : ℕ, n ≥ 3 →
  (180 * (n - 2) / n : ℚ) = 120 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_120_degree_interior_angle_l4111_411154


namespace NUMINAMATH_CALUDE_reflection_line_equation_l4111_411136

/-- The line of reflection for a triangle --/
structure ReflectionLine where
  equation : ℝ → Prop

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points --/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Reflects a point about a horizontal line --/
def reflect (p : Point) (y : ℝ) : Point :=
  { x := p.x, y := 2 * y - p.y }

/-- The theorem stating the equation of the reflection line --/
theorem reflection_line_equation 
  (t : Triangle)
  (t' : Triangle)
  (h1 : t.p = Point.mk 1 4)
  (h2 : t.q = Point.mk 8 9)
  (h3 : t.r = Point.mk (-3) 7)
  (h4 : t'.p = Point.mk 1 (-6))
  (h5 : t'.q = Point.mk 8 (-11))
  (h6 : t'.r = Point.mk (-3) (-9))
  (h7 : ∃ (y : ℝ), t'.p = reflect t.p y ∧ 
                   t'.q = reflect t.q y ∧ 
                   t'.r = reflect t.r y) :
  ∃ (m : ReflectionLine), m.equation = λ y => y = -1 :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l4111_411136


namespace NUMINAMATH_CALUDE_fred_age_difference_l4111_411175

theorem fred_age_difference (jim fred sam : ℕ) : 
  jim = 2 * fred →
  jim = 46 →
  jim - 6 = 5 * (sam - 6) →
  fred - sam = 9 := by sorry

end NUMINAMATH_CALUDE_fred_age_difference_l4111_411175


namespace NUMINAMATH_CALUDE_probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l4111_411155

/-- The probability that at least 2 out of 3 friends are in the same class, given 10 classes -/
theorem probability_at_least_two_in_same_class : ℝ :=
  let total_classes := 10
  let total_friends := 3
  let prob_all_different := (total_classes * (total_classes - 1) * (total_classes - 2)) / (total_classes ^ total_friends)
  1 - prob_all_different

/-- The probability is equal to 7/25 -/
theorem probability_equals_seven_twentyfifths : probability_at_least_two_in_same_class = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_in_same_class_probability_equals_seven_twentyfifths_l4111_411155


namespace NUMINAMATH_CALUDE_area_of_S₃_l4111_411135

/-- Given a square S₁ with area 25, S₂ is constructed by bisecting the sides of S₁,
    and S₃ is constructed by bisecting the sides of S₂. -/
def square_construction (S₁ S₂ S₃ : Real → Real → Prop) : Prop :=
  (∀ x y, S₁ x y ↔ x^2 + y^2 = 25) ∧
  (∀ x y, S₂ x y ↔ ∃ a b, S₁ a b ∧ x = a/2 ∧ y = b/2) ∧
  (∀ x y, S₃ x y ↔ ∃ a b, S₂ a b ∧ x = a/2 ∧ y = b/2)

/-- The area of S₃ is 6.25 -/
theorem area_of_S₃ (S₁ S₂ S₃ : Real → Real → Prop) :
  square_construction S₁ S₂ S₃ →
  (∃ x y, S₃ x y ∧ x^2 + y^2 = 6.25) :=
sorry

end NUMINAMATH_CALUDE_area_of_S₃_l4111_411135


namespace NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_l4111_411161

/-- Represents the possible outcomes of throwing a fair regular hexahedral die -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Defines event A: "the number is odd" -/
def eventA (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.one ∨ outcome = DieOutcome.three ∨ outcome = DieOutcome.five

/-- Defines event B: "the number is 4" -/
def eventB (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.four

/-- Theorem stating that events A and B are mutually exclusive -/
theorem events_A_B_mutually_exclusive :
  ∀ (outcome : DieOutcome), ¬(eventA outcome ∧ eventB outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_l4111_411161


namespace NUMINAMATH_CALUDE_novel_pages_count_prove_novel_pages_l4111_411163

theorem novel_pages_count : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 80 ∧ total_pages = 252

theorem prove_novel_pages : novel_pages_count 252 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_count_prove_novel_pages_l4111_411163


namespace NUMINAMATH_CALUDE_people_who_left_line_fair_line_problem_l4111_411190

theorem people_who_left_line (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) : ℕ :=
  let people_who_left := initial_people + joined - final_people
  people_who_left

theorem fair_line_problem : people_who_left_line 12 15 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_line_fair_line_problem_l4111_411190


namespace NUMINAMATH_CALUDE_basketball_team_average_weight_l4111_411179

/-- The average weight of a basketball team after adding new players -/
theorem basketball_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1 : ℝ) 
  (new_player2 : ℝ) 
  (new_player3 : ℝ) 
  (new_player4 : ℝ) 
  (h1 : original_players = 8) 
  (h2 : original_average = 105.5) 
  (h3 : new_player1 = 110.3) 
  (h4 : new_player2 = 99.7) 
  (h5 : new_player3 = 103.2) 
  (h6 : new_player4 = 115.4) : 
  (original_players * original_average + new_player1 + new_player2 + new_player3 + new_player4) / (original_players + 4) = 106.05 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_average_weight_l4111_411179


namespace NUMINAMATH_CALUDE_tournament_sequences_l4111_411196

theorem tournament_sequences (n : ℕ) : (2 * n).choose n = 3432 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_tournament_sequences_l4111_411196


namespace NUMINAMATH_CALUDE_remainder_problem_l4111_411147

theorem remainder_problem (x : ℤ) : x % 63 = 27 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4111_411147


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l4111_411174

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation_theorem :
  let original := Parabola.mk 1 0 (-2)
  let translated := translate_parabola original 3 1
  translated = Parabola.mk 1 (-6) (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l4111_411174


namespace NUMINAMATH_CALUDE_smallest_battleship_board_l4111_411151

/-- Represents the types of ships in Battleship -/
inductive ShipType
  | OneByFour
  | OneByThree
  | OneByTwo
  | OneByOne

/-- The set of ships in a standard Battleship game -/
def battleshipSet : List ShipType :=
  [ShipType.OneByFour] ++
  List.replicate 2 ShipType.OneByThree ++
  List.replicate 3 ShipType.OneByTwo ++
  List.replicate 4 ShipType.OneByOne

/-- Calculates the number of nodes a ship occupies, including its surrounding space -/
def nodesOccupied (ship : ShipType) : Nat :=
  match ship with
  | ShipType.OneByFour => 10
  | ShipType.OneByThree => 8
  | ShipType.OneByTwo => 6
  | ShipType.OneByOne => 4

/-- The smallest square board size for Battleship -/
def smallestBoardSize : Nat := 7

theorem smallest_battleship_board :
  (∀ n : Nat, n < smallestBoardSize → 
    (List.sum (List.map nodesOccupied battleshipSet) > (n + 1)^2)) ∧
  (List.sum (List.map nodesOccupied battleshipSet) ≤ (smallestBoardSize + 1)^2) := by
  sorry

#eval smallestBoardSize  -- Should output 7

end NUMINAMATH_CALUDE_smallest_battleship_board_l4111_411151


namespace NUMINAMATH_CALUDE_curvilinear_triangle_area_half_triangle_area_l4111_411170

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of a curvilinear triangle formed by three circles
def curvilinearTriangleArea (c1 c2 c3 : Circle) : ℝ := sorry

-- Theorem statement
theorem curvilinear_triangle_area_half_triangle_area 
  (c1 c2 c3 : Circle) 
  (t : Triangle) 
  (h1 : c1.radius = c2.radius ∧ c2.radius = c3.radius)
  (h2 : c1.center = t.a ∧ c2.center = t.b ∧ c3.center = t.c) :
  curvilinearTriangleArea c1 c2 c3 = (1/2) * triangleArea t := by sorry

end NUMINAMATH_CALUDE_curvilinear_triangle_area_half_triangle_area_l4111_411170


namespace NUMINAMATH_CALUDE_sisters_age_difference_l4111_411180

/-- The age difference between two sisters, Denise and Diane -/
def ageDifference (deniseFutureAge deniseFutureYears dianeFutureAge dianeFutureYears : ℕ) : ℕ :=
  (deniseFutureAge - deniseFutureYears) - (dianeFutureAge - dianeFutureYears)

/-- Theorem stating that the age difference between Denise and Diane is 4 years -/
theorem sisters_age_difference :
  ageDifference 25 2 25 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sisters_age_difference_l4111_411180


namespace NUMINAMATH_CALUDE_production_days_calculation_l4111_411109

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of past n days
  (h2 : ((n * 50 + 95 : ℝ) / (n + 1) = 55)) : -- New average including today
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l4111_411109


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l4111_411158

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l4111_411158


namespace NUMINAMATH_CALUDE_last_two_digits_of_17_power_17_l4111_411110

theorem last_two_digits_of_17_power_17 : 17^17 ≡ 77 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_17_power_17_l4111_411110


namespace NUMINAMATH_CALUDE_songs_on_mp3_player_l4111_411126

theorem songs_on_mp3_player (initial : ℕ) (deleted : ℕ) (added : ℕ) :
  initial ≥ deleted →
  (initial - deleted + added : ℕ) = initial - deleted + added :=
by sorry

end NUMINAMATH_CALUDE_songs_on_mp3_player_l4111_411126


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l4111_411188

/-- The number of squares in the nth ring around a 2x3 rectangular block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 6

/-- The 50th ring contains 406 squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 406 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l4111_411188


namespace NUMINAMATH_CALUDE_no_real_solutions_l4111_411171

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4111_411171


namespace NUMINAMATH_CALUDE_equation_equivalence_l4111_411153

theorem equation_equivalence (x : ℚ) : (x - 1) / 2 - x / 5 = 1 ↔ 5 * (x - 1) - 2 * x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4111_411153


namespace NUMINAMATH_CALUDE_hall_covering_expenditure_l4111_411125

/-- Calculates the total expenditure for covering the interior of a rectangular hall with a mat -/
def calculate_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the expenditure for covering a specific hall with a mat is 19000 -/
theorem hall_covering_expenditure :
  calculate_expenditure 20 15 5 20 = 19000 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_expenditure_l4111_411125


namespace NUMINAMATH_CALUDE_pages_left_to_read_l4111_411160

/-- Given a book with a total number of pages and a number of pages already read,
    calculate the number of pages left to read. -/
theorem pages_left_to_read (total_pages pages_read : ℕ) 
    (h1 : total_pages = 563)
    (h2 : pages_read = 147) :
    total_pages - pages_read = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l4111_411160


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4111_411106

/-- The decimal representation of x as 0.36̅ -/
def x : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.36̅ -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal :
  (1 : ℚ) / x = reciprocal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4111_411106


namespace NUMINAMATH_CALUDE_income_mean_difference_l4111_411199

def num_families : ℕ := 1500

def correct_largest_income_1 : ℝ := 150000
def correct_largest_income_2 : ℝ := 148000
def incorrect_largest_income : ℝ := 1500000

def sum_other_incomes : ℝ := sorry  -- This represents the sum S in the solution

theorem income_mean_difference :
  let actual_mean := (sum_other_incomes + correct_largest_income_1 + correct_largest_income_2) / num_families
  let incorrect_mean := (sum_other_incomes + 2 * incorrect_largest_income) / num_families
  incorrect_mean - actual_mean = 1801.33 := by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l4111_411199


namespace NUMINAMATH_CALUDE_total_camp_attendance_l4111_411143

def lawrence_county_camp : ℕ := 34044
def lawrence_county_home : ℕ := 134867
def outside_county_camp : ℕ := 424944

theorem total_camp_attendance : 
  lawrence_county_camp + outside_county_camp = 459988 := by
  sorry

end NUMINAMATH_CALUDE_total_camp_attendance_l4111_411143


namespace NUMINAMATH_CALUDE_robin_gum_total_l4111_411157

theorem robin_gum_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 18 → additional = 26 → total = initial + additional → total = 44 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_total_l4111_411157


namespace NUMINAMATH_CALUDE_percentage_in_at_least_two_trips_l4111_411112

/-- Represents the percentage of students who went on a specific trip -/
structure TripParticipation where
  threeDay : Rat
  twoDay : Rat
  oneDay : Rat

/-- Represents the percentage of students who participated in multiple trips -/
structure MultipleTrips where
  threeDayAndOneDay : Rat
  twoDayAndOther : Rat

/-- Calculates the percentage of students who participated in at least two trips -/
def percentageInAtLeastTwoTrips (tp : TripParticipation) (mt : MultipleTrips) : Rat :=
  mt.threeDayAndOneDay + mt.twoDayAndOther

/-- Main theorem: The percentage of students who participated in at least two trips is 22% -/
theorem percentage_in_at_least_two_trips :
  ∀ (tp : TripParticipation) (mt : MultipleTrips),
  tp.threeDay = 25/100 ∧
  tp.twoDay = 10/100 ∧
  mt.threeDayAndOneDay = 65/100 * tp.threeDay ∧
  mt.twoDayAndOther = 60/100 * tp.twoDay →
  percentageInAtLeastTwoTrips tp mt = 22/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_at_least_two_trips_l4111_411112


namespace NUMINAMATH_CALUDE_total_games_cost_is_13800_l4111_411178

/-- Calculates the total cost of games owned by Katie and her friends -/
def totalGamesCost (katieGames : ℕ) (newFriends oldFriends : ℕ) (newFriendGames oldFriendGames : ℕ) (costPerGame : ℕ) : ℕ :=
  let totalGames := katieGames + newFriends * newFriendGames + oldFriends * oldFriendGames
  totalGames * costPerGame

/-- Theorem stating that the total cost of games is $13,800 -/
theorem total_games_cost_is_13800 :
  totalGamesCost 91 5 3 88 53 20 = 13800 := by
  sorry

#eval totalGamesCost 91 5 3 88 53 20

end NUMINAMATH_CALUDE_total_games_cost_is_13800_l4111_411178


namespace NUMINAMATH_CALUDE_frank_weed_eating_earnings_l4111_411191

/-- Calculates the amount Frank made weed eating given his lawn mowing earnings, weekly spending, and duration of savings. -/
def weed_eating_earnings (lawn_mowing_earnings weekly_spending duration_weeks : ℕ) : ℕ :=
  weekly_spending * duration_weeks - lawn_mowing_earnings

theorem frank_weed_eating_earnings :
  weed_eating_earnings 5 7 9 = 58 := by
  sorry

end NUMINAMATH_CALUDE_frank_weed_eating_earnings_l4111_411191


namespace NUMINAMATH_CALUDE_no_consecutive_red_probability_l4111_411149

def num_lights : ℕ := 8
def red_prob : ℝ := 0.4
def green_prob : ℝ := 1 - red_prob

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def prob_no_consecutive_red : ℝ :=
  (green_prob ^ num_lights) * (binomial (num_lights + 1 - 0) 0) +
  (green_prob ^ 7) * red_prob * (binomial (num_lights + 1 - 1) 1) +
  (green_prob ^ 6) * (red_prob ^ 2) * (binomial (num_lights + 1 - 2) 2) +
  (green_prob ^ 5) * (red_prob ^ 3) * (binomial (num_lights + 1 - 3) 3) +
  (green_prob ^ 4) * (red_prob ^ 4) * (binomial (num_lights + 1 - 4) 4)

theorem no_consecutive_red_probability :
  prob_no_consecutive_red = 0.3499456 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_red_probability_l4111_411149


namespace NUMINAMATH_CALUDE_total_wheels_is_142_l4111_411169

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of wheels on a unicycle -/
def unicycle_wheels : ℕ := 1

/-- The number of wheels on a four-wheeler -/
def four_wheeler_wheels : ℕ := 4

/-- The number of bicycles in Storage Area A -/
def bicycles_A : ℕ := 16

/-- The number of tricycles in Storage Area A -/
def tricycles_A : ℕ := 7

/-- The number of unicycles in Storage Area A -/
def unicycles_A : ℕ := 10

/-- The number of four-wheelers in Storage Area A -/
def four_wheelers_A : ℕ := 5

/-- The number of bicycles in Storage Area B -/
def bicycles_B : ℕ := 12

/-- The number of tricycles in Storage Area B -/
def tricycles_B : ℕ := 5

/-- The number of unicycles in Storage Area B -/
def unicycles_B : ℕ := 8

/-- The number of four-wheelers in Storage Area B -/
def four_wheelers_B : ℕ := 3

/-- The total number of wheels in both storage areas -/
def total_wheels : ℕ := 
  (bicycles_A * bicycle_wheels + tricycles_A * tricycle_wheels + unicycles_A * unicycle_wheels + four_wheelers_A * four_wheeler_wheels) +
  (bicycles_B * bicycle_wheels + tricycles_B * tricycle_wheels + unicycles_B * unicycle_wheels + four_wheelers_B * four_wheeler_wheels)

theorem total_wheels_is_142 : total_wheels = 142 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_142_l4111_411169


namespace NUMINAMATH_CALUDE_subset_A_l4111_411111

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_A_l4111_411111


namespace NUMINAMATH_CALUDE_parallelogram_has_multiple_altitudes_l4111_411146

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An altitude of a parallelogram is a line segment from a vertex perpendicular to the opposite side or its extension. -/
structure Altitude (p : Parallelogram) where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A parallelogram has more than one altitude. -/
theorem parallelogram_has_multiple_altitudes (p : Parallelogram) : ∃ (a b : Altitude p), a ≠ b := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_has_multiple_altitudes_l4111_411146


namespace NUMINAMATH_CALUDE_complex_subtraction_l4111_411182

theorem complex_subtraction (z : ℂ) : (5 - 3*I - z = -1 + 4*I) → z = 6 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l4111_411182


namespace NUMINAMATH_CALUDE_fraction_powers_sum_l4111_411134

theorem fraction_powers_sum : 
  (8/9 : ℚ)^3 * (3/4 : ℚ)^3 + (1/2 : ℚ)^3 = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_sum_l4111_411134


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l4111_411164

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ∧ y = m₂ * x + b₂) → m₁ = m₂

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) (x y : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- The second line equation: y - 2 = (b + 9)x -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y - 2 = (b + 9) * x

theorem parallel_lines_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, line1 b x y ∧ line2 b x y) → b = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l4111_411164


namespace NUMINAMATH_CALUDE_factory_earnings_l4111_411197

-- Define the number of machines and their working hours
def num_machines_1 : ℕ := 3
def hours_per_day_1 : ℕ := 23
def num_machines_2 : ℕ := 1
def hours_per_day_2 : ℕ := 12

-- Define production rate and selling price
def production_rate : ℕ := 2  -- kg per hour
def selling_price : ℕ := 50   -- dollars per kg

-- Calculate daily earnings
def daily_earnings : ℕ :=
  (num_machines_1 * hours_per_day_1 + num_machines_2 * hours_per_day_2) *
  production_rate * selling_price

-- Theorem to prove
theorem factory_earnings :
  daily_earnings = 8100 := by
  sorry

end NUMINAMATH_CALUDE_factory_earnings_l4111_411197


namespace NUMINAMATH_CALUDE_max_intersection_points_four_spheres_l4111_411181

/-- A sphere in three-dimensional space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A line in three-dimensional space -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- The number of intersection points between a line and a sphere -/
def intersectionPoints (l : Line) (s : Sphere) : ℕ := sorry

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points_four_spheres (s₁ s₂ s₃ s₄ : Sphere) :
  ∃ (l : Line), (intersectionPoints l s₁) + (intersectionPoints l s₂) +
                (intersectionPoints l s₃) + (intersectionPoints l s₄) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_four_spheres_l4111_411181


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l4111_411195

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l4111_411195


namespace NUMINAMATH_CALUDE_crimson_valley_skirts_l4111_411115

/-- The number of skirts in each valley -/
structure ValleySkirts where
  ember : ℕ
  azure : ℕ
  seafoam : ℕ
  purple : ℕ
  crimson : ℕ

/-- The conditions for the valley skirts problem -/
def valley_conditions (v : ValleySkirts) : Prop :=
  v.crimson = v.purple / 3 ∧
  v.purple = v.seafoam / 4 ∧
  v.seafoam = v.azure * 3 / 5 ∧
  v.azure = v.ember * 2 ∧
  v.ember = 120

/-- Theorem stating that given the conditions, Crimson Valley has 12 skirts -/
theorem crimson_valley_skirts (v : ValleySkirts) 
  (h : valley_conditions v) : v.crimson = 12 := by
  sorry

end NUMINAMATH_CALUDE_crimson_valley_skirts_l4111_411115


namespace NUMINAMATH_CALUDE_larger_number_proof_l4111_411121

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4111_411121


namespace NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l4111_411183

theorem tens_digit_of_19_pow_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l4111_411183


namespace NUMINAMATH_CALUDE_sin_4theta_l4111_411192

theorem sin_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (4 + Complex.I * Real.sqrt 3) / 5) : 
  Real.sin (4 * θ) = 208 * Real.sqrt 3 / 625 := by
  sorry

end NUMINAMATH_CALUDE_sin_4theta_l4111_411192


namespace NUMINAMATH_CALUDE_eccentricity_difference_l4111_411108

/-- Given an ellipse and a hyperbola sharing the same foci, prove that
    the difference of their eccentricities is √2 under certain conditions. -/
theorem eccentricity_difference (a b m n : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  let C₁ := {(x, y) : ℝ × ℝ | x^2/a^2 + y^2/b^2 = 1}
  let C₂ := {(x, y) : ℝ × ℝ | x^2/m^2 - y^2/n^2 = 1}
  let c := Real.sqrt (a^2 - b^2)
  let f := Real.sqrt (m^2 + n^2)
  let e₁ := c / a
  let e₂ := f / m
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∃ P ∈ C₁ ∩ C₂, P.1 > 0 ∧ P.2 > 0 ∧ 
  c = f ∧ 
  dist P F₁ = dist P F₂ ∧
  dist P F₁ = dist F₁ F₂ →
  e₂ - e₁ = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_eccentricity_difference_l4111_411108


namespace NUMINAMATH_CALUDE_negative_x_power_seven_divided_by_negative_x_l4111_411102

theorem negative_x_power_seven_divided_by_negative_x (x : ℝ) :
  ((-x)^7) / (-x) = x^6 := by sorry

end NUMINAMATH_CALUDE_negative_x_power_seven_divided_by_negative_x_l4111_411102


namespace NUMINAMATH_CALUDE_vector_c_value_l4111_411185

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 5)

theorem vector_c_value (c : ℝ × ℝ) : 
  3 • a + (4 • b - a) + 2 • c = (0, 0) → c = (4, -9) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l4111_411185


namespace NUMINAMATH_CALUDE_min_l_trominos_count_l4111_411103

/-- Represents a tile type -/
inductive TileType
| LTromino
| STetromino

/-- Represents the grid -/
def Grid := Fin 2020 × Fin 2021

/-- A tiling is a function that assigns a tile type to each grid position -/
def Tiling := Grid → Option TileType

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def is_valid_tiling (t : Tiling) : Prop := sorry

/-- Counts the number of L-Trominos in a tiling -/
def count_l_trominos (t : Tiling) : Nat := sorry

/-- Theorem: The minimum number of L-Trominos in a valid tiling is 1010 -/
theorem min_l_trominos_count :
  ∃ (t : Tiling), is_valid_tiling t ∧
    ∀ (t' : Tiling), is_valid_tiling t' →
      count_l_trominos t ≤ count_l_trominos t' ∧
      count_l_trominos t = 1010 :=
sorry

end NUMINAMATH_CALUDE_min_l_trominos_count_l4111_411103


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l4111_411156

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 11/17) (h2 : x - y = 1/119) : 
  x^2 - y^2 = 11/2003 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l4111_411156


namespace NUMINAMATH_CALUDE_complex_product_real_l4111_411166

theorem complex_product_real (z₁ z₂ : ℂ) :
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 ↔ z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_real_l4111_411166


namespace NUMINAMATH_CALUDE_coordinate_difference_of_P_l4111_411150

/-- Triangle ABC with vertices A(0,10), B(4,0), C(12,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 10⟩, ⟨4, 0⟩, ⟨12, 0⟩}

/-- Point P on line AC -/
def P : ℝ × ℝ := ⟨6, 5⟩

/-- Point Q on line BC -/
def Q : ℝ × ℝ := ⟨6, 0⟩

/-- Area of triangle PQC -/
def area_PQC : ℝ := 16

/-- Theorem: The positive difference between x and y coordinates of P is 1 -/
theorem coordinate_difference_of_P :
  P ∈ Set.Icc (0 : ℝ) 12 ×ˢ Set.Icc (0 : ℝ) 10 →
  Q.1 = P.1 →
  Q.2 = 0 →
  area_PQC = 16 →
  |P.1 - P.2| = 1 := by sorry

end NUMINAMATH_CALUDE_coordinate_difference_of_P_l4111_411150


namespace NUMINAMATH_CALUDE_star_arrangement_count_l4111_411133

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct shells to be placed -/
def num_shells : ℕ := 12

/-- The number of distinct arrangements of shells on a regular six-pointed star,
    considering rotational and reflectional symmetries -/
def distinct_arrangements : ℕ := Nat.factorial num_shells / star_symmetries

theorem star_arrangement_count :
  distinct_arrangements = 39916800 := by sorry

end NUMINAMATH_CALUDE_star_arrangement_count_l4111_411133


namespace NUMINAMATH_CALUDE_solution_set_inequality_l4111_411122

theorem solution_set_inequality (x : ℝ) :
  (x - 1) * |x + 2| ≥ 0 ↔ x ≥ 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l4111_411122


namespace NUMINAMATH_CALUDE_tank_A_height_l4111_411173

-- Define the tanks
structure Tank where
  circumference : ℝ
  height : ℝ

-- Define the problem parameters
def tank_A : Tank := { circumference := 8, height := 10 }
def tank_B : Tank := { circumference := 10, height := 8 }

-- Define the capacity ratio
def capacity_ratio : ℝ := 0.8000000000000001

-- Theorem statement
theorem tank_A_height :
  tank_A.height = 10 ∧
  tank_A.circumference = 8 ∧
  tank_B.circumference = 10 ∧
  tank_B.height = 8 ∧
  (tank_A.circumference * tank_A.height) / (tank_B.circumference * tank_B.height) = capacity_ratio :=
by sorry

end NUMINAMATH_CALUDE_tank_A_height_l4111_411173


namespace NUMINAMATH_CALUDE_trig_equality_l4111_411177

theorem trig_equality : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l4111_411177


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l4111_411144

open Complex

theorem modulus_of_specific_complex_number :
  let z : ℂ := (2 - I) / (2 + I)
  ‖z‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l4111_411144


namespace NUMINAMATH_CALUDE_equation_graph_is_x_axis_l4111_411162

theorem equation_graph_is_x_axis : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 - 2*x*y ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_graph_is_x_axis_l4111_411162


namespace NUMINAMATH_CALUDE_investment_increase_l4111_411176

/-- Represents the broker's investments over three years -/
def investment_change (S R : ℝ) : ℝ := by
  -- Define the changes for each year
  let year1_stock := S * 1.5
  let year1_real_estate := R * 1.2
  
  let year2_stock := year1_stock * 0.7
  let year2_real_estate := year1_real_estate * 1.1
  
  let year3_stock_initial := year2_stock + 0.5 * S
  let year3_real_estate_initial := year2_real_estate - 0.2 * R
  
  let year3_stock_final := year3_stock_initial * 1.25
  let year3_real_estate_final := year3_real_estate_initial * 0.95
  
  -- Calculate the net change
  let net_change := (year3_stock_final + year3_real_estate_final) - (S + R)
  
  exact net_change

/-- Theorem stating the net increase in investment wealth -/
theorem investment_increase (S R : ℝ) : 
  investment_change S R = 0.9375 * S + 0.064 * R := by
  sorry

end NUMINAMATH_CALUDE_investment_increase_l4111_411176


namespace NUMINAMATH_CALUDE_f_2_eq_137_60_l4111_411139

def f (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range (2 * n + 2)) (fun i => 1 / (i + 1 : ℚ))

theorem f_2_eq_137_60 : f 2 = 137 / 60 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_137_60_l4111_411139


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l4111_411145

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := 5 * a + 6 * b

/-- Theorem stating that 4 ⊕ 2 = 32 -/
theorem circle_plus_four_two : circle_plus 4 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l4111_411145


namespace NUMINAMATH_CALUDE_f_properties_l4111_411119

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a)^2 * (x + b) * Real.exp x

-- Define the first derivative of f
def f' (a b x : ℝ) : ℝ := Real.exp x * (x - a) * (x^2 + (3 - a + b) * x + 2 * b - a * b - a)

-- State the theorem
theorem f_properties (a b : ℝ) :
  (∀ x, f' a b x ≤ f' a b a) →
  ((a = 0 → b < 0) ∧
   (∃ x₄, (x₄ = a + 2 * Real.sqrt 6 ∨ x₄ = a - 2 * Real.sqrt 6) ∧ b = -a - 3) ∨
   (∃ x₄, x₄ = a + (1 + Real.sqrt 13) / 2 ∧ b = -a - (7 + Real.sqrt 13) / 2) ∨
   (∃ x₄, x₄ = a + (1 - Real.sqrt 13) / 2 ∧ b = -a - (7 - Real.sqrt 13) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l4111_411119


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l4111_411184

theorem sqrt_meaningful_iff_geq_neg_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l4111_411184


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l4111_411168

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0 → x + y + z = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l4111_411168


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l4111_411100

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Finite E] [Fact (finrank ℝ E = 3)]

-- Define unit vectors
variable (a b c d : E)

-- State the theorem
theorem max_sum_squared_distances (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l4111_411100


namespace NUMINAMATH_CALUDE_not_perfect_cube_1967_l4111_411152

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem not_perfect_cube_1967 :
  ∀ (p : Fin 1967 → Fin 1967), Function.Bijective p →
    ¬ (∃ (k : ℕ), sum_of_first_n 1967 = k^3) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_cube_1967_l4111_411152


namespace NUMINAMATH_CALUDE_minimum_duty_days_l4111_411186

theorem minimum_duty_days (total_members : ℕ) (duty_size_1 duty_size_2 : ℕ) :
  total_members = 33 →
  duty_size_1 = 9 →
  duty_size_2 = 10 →
  ∃ (k n m : ℕ), 
    k + n = 7 ∧ 
    duty_size_1 * k + duty_size_2 * n = total_members * m ∧
    ∀ (k' n' m' : ℕ), 
      k' + n' < 7 → 
      duty_size_1 * k' + duty_size_2 * n' ≠ total_members * m' :=
by sorry

end NUMINAMATH_CALUDE_minimum_duty_days_l4111_411186


namespace NUMINAMATH_CALUDE_john_travel_time_l4111_411130

/-- Proves that given a distance of 24 km and a normal travel time of 44 minutes,
    if a speed of 40 kmph results in arriving 8 minutes early,
    then a speed of 30 kmph will result in arriving 4 minutes late. -/
theorem john_travel_time (distance : ℝ) (normal_time : ℝ) (early_speed : ℝ) (late_speed : ℝ) :
  distance = 24 →
  normal_time = 44 / 60 →
  early_speed = 40 →
  late_speed = 30 →
  distance / early_speed = normal_time - 8 / 60 →
  distance / late_speed = normal_time + 4 / 60 :=
by sorry

end NUMINAMATH_CALUDE_john_travel_time_l4111_411130


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l4111_411137

theorem square_sum_given_product_and_sum (r s : ℝ) 
  (h1 : r * s = 16) 
  (h2 : r + s = 8) : 
  r^2 + s^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l4111_411137


namespace NUMINAMATH_CALUDE_increased_amount_proof_l4111_411124

theorem increased_amount_proof (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 3) :
  x + y = 60 * (1 / x) → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_increased_amount_proof_l4111_411124


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4111_411132

theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4111_411132


namespace NUMINAMATH_CALUDE_exam_score_proof_l4111_411123

theorem exam_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 70 → score2 = 80 → score3 = 90 → 
  (score1 + score2 + score3 + score4) / 4 = 70 → 
  score4 = 40 := by
sorry

end NUMINAMATH_CALUDE_exam_score_proof_l4111_411123


namespace NUMINAMATH_CALUDE_pentagon_fifth_angle_l4111_411104

/-- The sum of angles in a pentagon is 540 degrees -/
def pentagon_angle_sum : ℝ := 540

/-- The known angles of the pentagon -/
def known_angles : List ℝ := [130, 80, 105, 110]

/-- The measure of the unknown angle Q -/
def angle_q : ℝ := 115

/-- Theorem: In a pentagon with four known angles measuring 130°, 80°, 105°, and 110°, 
    the measure of the fifth angle is 115°. -/
theorem pentagon_fifth_angle :
  pentagon_angle_sum = (known_angles.sum + angle_q) :=
by sorry

end NUMINAMATH_CALUDE_pentagon_fifth_angle_l4111_411104


namespace NUMINAMATH_CALUDE_periodic_function_l4111_411127

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ)

theorem periodic_function (φ : ℝ) :
  ∃ t : ℝ, t ≠ 0 ∧ ∀ x : ℝ, f (x + t) φ = f x φ :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_periodic_function_l4111_411127


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l4111_411129

theorem smallest_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → 
  21 * x * y + 2 * y * z + 8 * z * x ≤ 12 → 
  1 / a + 2 / b + 3 / c ≤ 1 / x + 2 / y + 3 / z :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l4111_411129


namespace NUMINAMATH_CALUDE_carwash_problem_l4111_411138

/-- Represents the number of vehicles of each type washed --/
structure VehicleCounts where
  cars : ℕ
  trucks : ℕ
  suvs : ℕ

/-- Represents the prices for washing each type of vehicle --/
structure WashPrices where
  car : ℕ
  truck : ℕ
  suv : ℕ

/-- Calculates the total amount raised from a car wash --/
def totalRaised (counts : VehicleCounts) (prices : WashPrices) : ℕ :=
  counts.cars * prices.car + counts.trucks * prices.truck + counts.suvs * prices.suv

/-- The main theorem to prove --/
theorem carwash_problem (prices : WashPrices) 
    (h_car_price : prices.car = 5)
    (h_truck_price : prices.truck = 6)
    (h_suv_price : prices.suv = 7)
    (h_total : totalRaised { cars := 7, trucks := 5, suvs := 5 } prices = 100) :
  ∃ (n : ℕ), n = 7 ∧ 
    totalRaised { cars := n, trucks := 5, suvs := 5 } prices = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_carwash_problem_l4111_411138


namespace NUMINAMATH_CALUDE_solution_is_twelve_l4111_411167

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

/-- Theorem stating that 12 is the solution to the equation -/
theorem solution_is_twelve :
  ∃ (x : ℝ), custom_mul 3 (custom_mul 6 x) = 12 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_twelve_l4111_411167


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_div_7_l4111_411114

theorem tan_alpha_3_implies_fraction_eq_5_div_7 (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_div_7_l4111_411114


namespace NUMINAMATH_CALUDE_eight_digit_numbers_with_consecutive_digits_l4111_411120

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Total number of 8-digit numbers with digits 1 or 2 -/
def total_numbers : ℕ := 2^8

/-- Number of 8-digit numbers with no consecutive same digits -/
def numbers_without_consecutive : ℕ := 2 * fib 7

theorem eight_digit_numbers_with_consecutive_digits : 
  total_numbers - numbers_without_consecutive = 230 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_with_consecutive_digits_l4111_411120


namespace NUMINAMATH_CALUDE_water_trough_problem_l4111_411194

/-- Calculates the remaining water volume in a trough after a given number of days,
    given an initial volume and a constant daily evaporation rate. -/
def remaining_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_volume - evaporation_rate * days

/-- Proves that given an initial water volume of 300 gallons, with a constant evaporation rate
    of 1 gallon per day over 45 days and no additional water added or removed,
    the final water volume will be 255 gallons. -/
theorem water_trough_problem :
  remaining_water_volume 300 1 45 = 255 := by
  sorry


end NUMINAMATH_CALUDE_water_trough_problem_l4111_411194


namespace NUMINAMATH_CALUDE_milk_bottles_remaining_l4111_411128

/-- Calculates the number of milk bottles remaining after purchases. -/
def remaining_bottles (initial : ℕ) (jason : ℕ) (harry_more : ℕ) : ℕ :=
  initial - (jason + (jason + harry_more))

/-- Theorem stating the number of remaining bottles in the given scenario. -/
theorem milk_bottles_remaining : remaining_bottles 35 5 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottles_remaining_l4111_411128


namespace NUMINAMATH_CALUDE_counterfeit_coin_findable_l4111_411131

/-- Represents the type of scale: regular or magical -/
inductive ScaleType
| Regular
| Magical

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a coin -/
structure Coin := (id : Nat)

/-- Represents a weighing action -/
structure Weighing := 
  (left : List Coin)
  (right : List Coin)

/-- Represents the state of the problem -/
structure ProblemState :=
  (coins : List Coin)
  (counterfeitCoin : Coin)
  (scaleType : ScaleType)

/-- Function to perform a weighing -/
def performWeighing (state : ProblemState) (w : Weighing) : WeighingResult :=
  sorry

/-- Function representing a strategy to find the counterfeit coin -/
def findCounterfeitStrategy : ProblemState → List Weighing → Option Coin :=
  sorry

/-- Theorem stating that it's possible to find the counterfeit coin in 3 weighings -/
theorem counterfeit_coin_findable :
  ∀ (coins : List Coin) (counterfeitCoin : Coin) (scaleType : ScaleType),
    coins.length = 12 →
    counterfeitCoin ∈ coins →
    ∃ (strategy : List Weighing),
      strategy.length ≤ 3 ∧
      (findCounterfeitStrategy ⟨coins, counterfeitCoin, scaleType⟩ strategy = some counterfeitCoin) :=
sorry

end NUMINAMATH_CALUDE_counterfeit_coin_findable_l4111_411131
