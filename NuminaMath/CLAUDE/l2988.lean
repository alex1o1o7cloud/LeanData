import Mathlib

namespace NUMINAMATH_CALUDE_jane_calculation_l2988_298801

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by sorry

end NUMINAMATH_CALUDE_jane_calculation_l2988_298801


namespace NUMINAMATH_CALUDE_find_added_number_l2988_298888

theorem find_added_number (x y : ℤ) : 
  x % 82 = 5 → (x + y) % 41 = 12 → y = 7 := by sorry

end NUMINAMATH_CALUDE_find_added_number_l2988_298888


namespace NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l2988_298828

-- Definition of even number
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Definition of rational number
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Theorem 1: If p^2 is even, then p is even
theorem square_even_implies_even (p : ℤ) : is_even (p^2) → is_even p := by sorry

-- Theorem 2: √2 is irrational
theorem sqrt_2_irrational : ¬ is_rational (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l2988_298828


namespace NUMINAMATH_CALUDE_second_player_cannot_win_l2988_298884

-- Define the game of tic-tac-toe
structure TicTacToe :=
  (board : Matrix (Fin 3) (Fin 3) (Option Bool))
  (current_player : Bool)

-- Define optimal play
def optimal_play (game : TicTacToe) : Bool := sorry

-- Define the winning condition
def is_win (game : TicTacToe) (player : Bool) : Prop := sorry

-- Define the draw condition
def is_draw (game : TicTacToe) : Prop := sorry

-- Theorem: If the first player plays optimally, the second player cannot win
theorem second_player_cannot_win (game : TicTacToe) :
  optimal_play game → ¬(is_win game false) :=
by sorry

end NUMINAMATH_CALUDE_second_player_cannot_win_l2988_298884


namespace NUMINAMATH_CALUDE_garden_perimeter_l2988_298896

theorem garden_perimeter : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  w * l = 200 →
  w^2 + l^2 = 30^2 →
  l = w + 4 →
  2 * (w + l) = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2988_298896


namespace NUMINAMATH_CALUDE_janice_age_problem_l2988_298814

theorem janice_age_problem :
  ∀ x : ℕ,
  (x + 12 = 8 * (x - 2)) → x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_janice_age_problem_l2988_298814


namespace NUMINAMATH_CALUDE_team_formation_with_girls_l2988_298810

theorem team_formation_with_girls (total : Nat) (boys : Nat) (girls : Nat) (team_size : Nat) :
  total = boys + girls → boys = 5 → girls = 5 → team_size = 3 →
  (Nat.choose total team_size) - (Nat.choose boys team_size) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_with_girls_l2988_298810


namespace NUMINAMATH_CALUDE_second_wheat_rate_l2988_298886

-- Define the quantities and rates
def wheat1_quantity : ℝ := 30
def wheat1_rate : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def profit_percentage : ℝ := 0.10
def mixture_sell_rate : ℝ := 13.86

-- Define the theorem
theorem second_wheat_rate (wheat2_rate : ℝ) : 
  wheat1_quantity * wheat1_rate + wheat2_quantity * wheat2_rate = 
  (wheat1_quantity + wheat2_quantity) * mixture_sell_rate / (1 + profit_percentage) →
  wheat2_rate = 14.25 := by
sorry

end NUMINAMATH_CALUDE_second_wheat_rate_l2988_298886


namespace NUMINAMATH_CALUDE_janet_ticket_problem_l2988_298898

/-- The number of tickets needed for one ride on the roller coaster -/
def roller_coaster_tickets : ℕ := 5

/-- The total number of tickets needed for 7 rides on the roller coaster and 4 rides on the giant slide -/
def total_tickets : ℕ := 47

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 7

/-- The number of giant slide rides -/
def giant_slide_rides : ℕ := 4

/-- The number of tickets needed for one ride on the giant slide -/
def giant_slide_tickets : ℕ := 3

theorem janet_ticket_problem :
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets :=
sorry

end NUMINAMATH_CALUDE_janet_ticket_problem_l2988_298898


namespace NUMINAMATH_CALUDE_a_investment_l2988_298897

theorem a_investment (b_investment c_investment total_profit a_profit_share : ℕ) 
  (hb : b_investment = 7200)
  (hc : c_investment = 9600)
  (hp : total_profit = 9000)
  (ha : a_profit_share = 1125) : 
  ∃ a_investment : ℕ, 
    a_investment = 2400 ∧ 
    a_profit_share * (a_investment + b_investment + c_investment) = a_investment * total_profit :=
by sorry

end NUMINAMATH_CALUDE_a_investment_l2988_298897


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2988_298803

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = -3 + 4*I → z = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2988_298803


namespace NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l2988_298867

theorem sin_cos_difference_36_degrees : 
  Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l2988_298867


namespace NUMINAMATH_CALUDE_union_complement_equality_l2988_298845

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l2988_298845


namespace NUMINAMATH_CALUDE_foreign_language_selection_l2988_298865

theorem foreign_language_selection (total : ℕ) (english_speakers : ℕ) (japanese_speakers : ℕ) :
  total = 9 ∧ english_speakers = 5 ∧ japanese_speakers = 4 →
  english_speakers * japanese_speakers = 20 := by
sorry

end NUMINAMATH_CALUDE_foreign_language_selection_l2988_298865


namespace NUMINAMATH_CALUDE_problem_solution_l2988_298861

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2988_298861


namespace NUMINAMATH_CALUDE_inscribed_square_pyramid_dimensions_l2988_298824

/-- Regular pentagonal pyramid with square pyramid inscribed -/
structure PentagonalPyramidWithInscribedSquare where
  a : ℝ  -- side length of pentagonal base
  e : ℝ  -- height of pentagonal pyramid
  x : ℝ  -- side length of inscribed square base

/-- Theorem about the dimensions of the inscribed square pyramid -/
theorem inscribed_square_pyramid_dimensions
  (P : PentagonalPyramidWithInscribedSquare)
  (h_a_pos : P.a > 0)
  (h_e_pos : P.e > 0) :
  P.x = P.a / (2 * Real.sin (18 * π / 180) + Real.tan (18 * π / 180)) ∧
  ∃ (SR₁ SR₃ : ℝ),
    SR₁^2 = (P.a * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.cos (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) ∧
    SR₃^2 = (P.a * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)))^2 +
            P.e^2 - P.a^2 * Real.sin (36 * π / 180) / (Real.sin (36 * π / 180) + Real.sin (18 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_pyramid_dimensions_l2988_298824


namespace NUMINAMATH_CALUDE_good_array_probability_l2988_298835

def is_good_array (a b c d : Int) : Prop :=
  a ∈ ({-1, 0, 1} : Set Int) ∧
  b ∈ ({-1, 0, 1} : Set Int) ∧
  c ∈ ({-1, 0, 1} : Set Int) ∧
  d ∈ ({-1, 0, 1} : Set Int) ∧
  a + b ≠ c + d ∧
  a + b ≠ a + c ∧
  a + b ≠ b + d ∧
  c + d ≠ a + c ∧
  c + d ≠ b + d ∧
  a + c ≠ b + d

def total_arrays : Nat := 3^4

def good_arrays : Nat := 16

theorem good_array_probability :
  (good_arrays : ℚ) / total_arrays = 16 / 81 :=
sorry

end NUMINAMATH_CALUDE_good_array_probability_l2988_298835


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2988_298892

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 128)
  (h3 : time = 4)
  (h4 : simple_interest principal (4 : ℚ) time = interest) : 
  ∃ (rate : ℚ), rate = 4 ∧ simple_interest principal rate time = interest := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2988_298892


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2988_298853

/-- 
Given a quadratic equation (k-1)x^2 - 2x + 1 = 0, 
this theorem states the conditions on k for the equation to have two distinct real roots.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (k - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2988_298853


namespace NUMINAMATH_CALUDE_exponent_division_l2988_298891

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^8 = 1 / x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2988_298891


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l2988_298868

theorem p_or_q_is_true (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l2988_298868


namespace NUMINAMATH_CALUDE_regular_pyramid_cross_section_l2988_298822

/-- Regular pyramid with inscribed cross-section --/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Ratio of edge division by plane
  edge_ratio : ℝ × ℝ
  -- Ratio of volumes divided by plane
  volume_ratio : ℝ × ℝ
  -- Distance from sphere center to plane
  sphere_center_distance : ℝ
  -- Perimeter of cross-section
  cross_section_perimeter : ℝ

/-- Theorem about regular pyramid with specific cross-section --/
theorem regular_pyramid_cross_section 
  (p : RegularPyramid) 
  (h_base : p.base_side = 2) 
  (h_perimeter : p.cross_section_perimeter = 32/5) :
  p.edge_ratio = (2, 3) ∧ 
  p.volume_ratio = (26, 9) ∧ 
  p.sphere_center_distance = (22 * Real.sqrt 14) / (35 * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_cross_section_l2988_298822


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l2988_298809

/-- Given a rectangular metallic sheet from which squares are cut at corners to form a box,
    this theorem proves the length of the original sheet. -/
theorem metallic_sheet_length
  (square_side : ℝ)
  (sheet_width : ℝ)
  (box_volume : ℝ)
  (h_square : square_side = 6)
  (h_width : sheet_width = 36)
  (h_volume : box_volume = 5184)
  (h_box : box_volume = (sheet_length - 2 * square_side) * (sheet_width - 2 * square_side) * square_side) :
  sheet_length = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_metallic_sheet_length_l2988_298809


namespace NUMINAMATH_CALUDE_hockey_arena_seating_l2988_298838

/-- The minimum number of rows required to seat students in a hockey arena --/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given conditions --/
theorem hockey_arena_seating 
  (seats_per_row : ℕ) 
  (total_students : ℕ) 
  (max_students_per_school : ℕ) 
  (h1 : seats_per_row = 168)
  (h2 : total_students = 2016)
  (h3 : max_students_per_school = 45)
  (h4 : ∀ (school : ℕ), school ≤ total_students → school ≤ max_students_per_school) :
  min_rows seats_per_row total_students max_students_per_school = 16 :=
sorry

end NUMINAMATH_CALUDE_hockey_arena_seating_l2988_298838


namespace NUMINAMATH_CALUDE_jason_has_more_blue_marbles_l2988_298855

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue - tom_blue

/-- Theorem stating that Jason has 20 more blue marbles than Tom -/
theorem jason_has_more_blue_marbles : blue_marble_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_more_blue_marbles_l2988_298855


namespace NUMINAMATH_CALUDE_lattice_triangle_area_bound_l2988_298882

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is on the edge of a triangle -/
def isOnEdge (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Counts the number of lattice points inside a triangle -/
def interiorPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Counts the number of lattice points on the edges of a triangle -/
def boundaryPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : LatticeTriangle) : ℚ := sorry

/-- Theorem: The area of a lattice triangle with exactly one interior lattice point is at most 9/2 -/
theorem lattice_triangle_area_bound (t : LatticeTriangle) 
  (h : interiorPointCount t = 1) : 
  triangleArea t ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_bound_l2988_298882


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2988_298856

theorem cube_root_simplification : (2^6 * 3^3 * 7^3 * 13^3 : ℝ)^(1/3) = 1092 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2988_298856


namespace NUMINAMATH_CALUDE_factor_theorem_application_l2988_298826

theorem factor_theorem_application (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 3 * c * x + 35)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l2988_298826


namespace NUMINAMATH_CALUDE_third_draw_white_probability_l2988_298821

/-- The probability of drawing a white ball on the third draw from an urn with 6 white and 5 black balls -/
theorem third_draw_white_probability (white : ℕ) (black : ℕ) :
  white = 6 → black = 5 → (white / (white + black) : ℚ) = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_third_draw_white_probability_l2988_298821


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2988_298889

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 5*m + 6 : ℂ) + (m^2 - 3*m : ℂ) * Complex.I = Complex.I * ((m^2 - 3*m : ℂ)) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2988_298889


namespace NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l2988_298806

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hyp_short : shorterLeg = hypotenuse / 2
  hyp_long : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
structure TriangleSequence where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  hyp_relation1 : t1.longerLeg = t2.hypotenuse
  hyp_relation2 : t2.longerLeg = t3.hypotenuse
  hyp_relation3 : t3.longerLeg = t4.hypotenuse
  largest_hyp : t1.hypotenuse = 16

theorem smallest_triangle_longer_leg (seq : TriangleSequence) : seq.t4.longerLeg = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l2988_298806


namespace NUMINAMATH_CALUDE_triplet_sum_not_two_l2988_298802

theorem triplet_sum_not_two : ∃! (a b c : ℚ), 
  ((a, b, c) = (3/4, 1/2, 3/4) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5) ∨ 
   (a, b, c) = (3/5, 7/10, 7/10) ∨ 
   (a, b, c) = (33/10, -8/5, 3/10) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5)) ∧ 
  a + b + c ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_triplet_sum_not_two_l2988_298802


namespace NUMINAMATH_CALUDE_walking_speed_problem_l2988_298836

/-- The speed of person P in miles per hour -/
def speed_P : ℝ := 7.5

/-- The speed of person Q in miles per hour -/
def speed_Q : ℝ := speed_P + 3

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The distance from the meeting point to Town Y in miles -/
def meeting_distance : ℝ := 15

theorem walking_speed_problem :
  (distance - meeting_distance) / speed_P = (distance + meeting_distance) / speed_Q :=
sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2988_298836


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2988_298816

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 49 / 64) :
  (4 * a) / (4 * b) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l2988_298816


namespace NUMINAMATH_CALUDE_paper_folding_height_l2988_298844

/-- Given a square piece of paper with side length 100 cm, 
    with cuts from each corner starting 8 cm from the corner and meeting at 45°,
    prove that the perpendicular height of the folded shape is 8 cm. -/
theorem paper_folding_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 100 →
  cut_distance = 8 →
  cut_angle = 45 →
  let diagonal_length := side_length * Real.sqrt 2
  let cut_length := cut_distance * Real.sqrt 2
  let height := Real.sqrt (cut_length^2 - (cut_length / 2)^2)
  height = 8 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_height_l2988_298844


namespace NUMINAMATH_CALUDE_square_roots_sum_zero_l2988_298854

theorem square_roots_sum_zero (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : (x - 4) + 3 = 0) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_zero_l2988_298854


namespace NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_three_i_l2988_298872

theorem complex_modulus_three_fourths_minus_three_i :
  Complex.abs (3/4 - 3*I) = (3 * Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_three_i_l2988_298872


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l2988_298875

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l2988_298875


namespace NUMINAMATH_CALUDE_survey_result_l2988_298895

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : Nat
  yes_responses : Nat
  id_range : Nat × Nat

/-- Calculates the expected number of students who have cheated -/
def expected_cheaters (data : SurveyData) : Nat :=
  let odd_ids := (data.id_range.2 - data.id_range.1 + 1) / 2
  let expected_yes_to_odd := odd_ids / 2
  let expected_cheaters_half := data.yes_responses - expected_yes_to_odd
  2 * expected_cheaters_half

/-- Theorem stating the expected number of cheaters based on the survey data -/
theorem survey_result (data : SurveyData) 
  (h1 : data.total_students = 2000)
  (h2 : data.yes_responses = 510)
  (h3 : data.id_range = (1, 2000)) :
  expected_cheaters data = 20 := by
  sorry


end NUMINAMATH_CALUDE_survey_result_l2988_298895


namespace NUMINAMATH_CALUDE_inequality_implies_max_a_l2988_298820

theorem inequality_implies_max_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_max_a_l2988_298820


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2988_298859

theorem complex_arithmetic_equality : -7 * 3 - (-5 * -4) + (-9 * -6) = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2988_298859


namespace NUMINAMATH_CALUDE_average_equation_holds_for_all_reals_solution_is_all_reals_l2988_298818

theorem average_equation_holds_for_all_reals (y : ℝ) : 
  ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

theorem solution_is_all_reals : 
  ∀ y : ℝ, ((2*y + 5) + (3*y + 4) + (7*y - 2)) / 3 = 4*y + 7/3 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_holds_for_all_reals_solution_is_all_reals_l2988_298818


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2988_298837

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2988_298837


namespace NUMINAMATH_CALUDE_clerical_percentage_after_reduction_l2988_298862

/-- Represents a department in the company -/
structure Department where
  total : Nat
  clerical_fraction : Rat
  reduction : Rat

/-- Calculates the number of clerical staff in a department after reduction -/
def clerical_after_reduction (d : Department) : Rat :=
  (d.total : Rat) * d.clerical_fraction * (1 - d.reduction)

/-- The company structure with its departments -/
structure Company where
  dept_a : Department
  dept_b : Department
  dept_c : Department

/-- The specific company instance from the problem -/
def company_x : Company :=
  { dept_a := { total := 4000, clerical_fraction := 1/4, reduction := 1/4 },
    dept_b := { total := 6000, clerical_fraction := 1/6, reduction := 1/10 },
    dept_c := { total := 2000, clerical_fraction := 1/8, reduction := 0 } }

/-- Total number of employees in the company -/
def total_employees : Nat := 12000

/-- Theorem stating the percentage of clerical staff after reductions -/
theorem clerical_percentage_after_reduction :
  (clerical_after_reduction company_x.dept_a +
   clerical_after_reduction company_x.dept_b +
   clerical_after_reduction company_x.dept_c) /
  (total_employees : Rat) * 100 = 1900 / 12000 * 100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_percentage_after_reduction_l2988_298862


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2988_298831

/-- The total surface area of a right cylinder with height 10 cm and radius 3 cm is 78π cm². -/
theorem cylinder_surface_area : 
  let h : ℝ := 10  -- height in cm
  let r : ℝ := 3   -- radius in cm
  let lateral_area := 2 * Real.pi * r * h
  let base_area := Real.pi * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 78 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2988_298831


namespace NUMINAMATH_CALUDE_statement_b_false_statement_c_false_l2988_298890

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := |x - y + 3|

-- Statement B is false
theorem statement_b_false :
  ¬ (∀ x y : ℝ, 3 * (star x y) = star (3 * x + 3) (3 * y + 3)) :=
sorry

-- Statement C is false
theorem statement_c_false :
  ¬ (∀ x : ℝ, star x (-3) = x) :=
sorry

end NUMINAMATH_CALUDE_statement_b_false_statement_c_false_l2988_298890


namespace NUMINAMATH_CALUDE_a_range_l2988_298885

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2988_298885


namespace NUMINAMATH_CALUDE_bella_stamps_l2988_298863

/-- The number of stamps Bella bought -/
def total_stamps (snowflake truck rose : ℕ) : ℕ := snowflake + truck + rose

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps : ∃ (snowflake truck rose : ℕ),
  snowflake = 11 ∧
  truck = snowflake + 9 ∧
  rose = truck - 13 ∧
  total_stamps snowflake truck rose = 38 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_l2988_298863


namespace NUMINAMATH_CALUDE_distance_between_closest_points_of_circles_l2988_298880

/-- Given two circles with centers at (1,1) and (20,5), both tangent to the x-axis,
    the distance between their closest points is √377 - 6. -/
theorem distance_between_closest_points_of_circles :
  let center1 : ℝ × ℝ := (1, 1)
  let center2 : ℝ × ℝ := (20, 5)
  let radius1 : ℝ := center1.2  -- y-coordinate of center1
  let radius2 : ℝ := center2.2  -- y-coordinate of center2
  let distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_between_centers - (radius1 + radius2) = Real.sqrt 377 - 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_closest_points_of_circles_l2988_298880


namespace NUMINAMATH_CALUDE_apple_expense_calculation_l2988_298870

/-- Proves that the amount spent on apples is the difference between the total amount and the sum of other expenses and remaining money. -/
theorem apple_expense_calculation (total amount_oranges amount_candy amount_left : ℕ) 
  (h1 : total = 95)
  (h2 : amount_oranges = 14)
  (h3 : amount_candy = 6)
  (h4 : amount_left = 50) :
  total - (amount_oranges + amount_candy + amount_left) = 25 :=
by sorry

end NUMINAMATH_CALUDE_apple_expense_calculation_l2988_298870


namespace NUMINAMATH_CALUDE_sugar_for_frosting_l2988_298866

theorem sugar_for_frosting (total_sugar : Real) (cake_sugar : Real) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) :
  total_sugar - cake_sugar = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_frosting_l2988_298866


namespace NUMINAMATH_CALUDE_total_seashells_l2988_298899

theorem total_seashells (sam mary lucy : ℕ) 
  (h_sam : sam = 18) 
  (h_mary : mary = 47) 
  (h_lucy : lucy = 32) : 
  sam + mary + lucy = 97 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2988_298899


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2988_298825

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 5 ∧ 7 * x - 3 * y = 20 → x = 5 ∧ y = 5 := by
  sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  3 * (x + y) - 4 * (x - y) = 16 ∧ (x + y) / 2 + (x - y) / 6 = 1 → 
  x = 1/3 ∧ y = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2988_298825


namespace NUMINAMATH_CALUDE_axe_sharpening_cost_l2988_298807

theorem axe_sharpening_cost
  (trees_per_sharpening : ℕ)
  (total_sharpening_cost : ℚ)
  (min_trees_chopped : ℕ)
  (h1 : trees_per_sharpening = 13)
  (h2 : total_sharpening_cost = 35)
  (h3 : min_trees_chopped ≥ 91) :
  let sharpenings := min_trees_chopped / trees_per_sharpening
  total_sharpening_cost / sharpenings = 5 := by
sorry

end NUMINAMATH_CALUDE_axe_sharpening_cost_l2988_298807


namespace NUMINAMATH_CALUDE_fifteen_percent_of_800_is_120_l2988_298852

theorem fifteen_percent_of_800_is_120 :
  ∀ x : ℝ, (15 / 100) * x = 120 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_800_is_120_l2988_298852


namespace NUMINAMATH_CALUDE_exists_larger_area_figure_l2988_298894

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  -- We don't need to define the internal structure for this problem
  area : ℝ
  perimeter : ℝ

/-- A chord of a convex figure -/
structure Chord (F : ConvexFigure) where
  -- We don't need to define the internal structure for this problem
  dividesPerimeterInHalf : Bool
  dividesAreaUnequally : Bool

/-- Theorem: If a convex figure has a chord that divides its perimeter in half
    and its area unequally, then there exists another figure with the same
    perimeter but larger area -/
theorem exists_larger_area_figure (F : ConvexFigure) 
  (h : ∃ c : Chord F, c.dividesPerimeterInHalf ∧ c.dividesAreaUnequally) :
  ∃ G : ConvexFigure, G.perimeter = F.perimeter ∧ G.area > F.area :=
by sorry

end NUMINAMATH_CALUDE_exists_larger_area_figure_l2988_298894


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2988_298850

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x, 729 * x^3 + 64 = (a*x^2 + b*x + c) * (d*x^2 + e*x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2988_298850


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_5_l2988_298877

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point inside the square --/
structure PointInSquare (s : Square) where
  point : ℝ × ℝ
  inside : point.1 ≥ s.bottomLeft.1 ∧ point.1 ≤ s.topRight.1 ∧
           point.2 ≥ s.bottomLeft.2 ∧ point.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in the square --/
def probability (s : Square) (event : PointInSquare s → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_5 :
  let s : Square := ⟨(0, 0), (4, 4)⟩
  probability s (fun p => p.point.1 + p.point.2 < 5) = 29 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_5_l2988_298877


namespace NUMINAMATH_CALUDE_peaches_after_seven_days_l2988_298808

def peaches_after_days (initial_total : ℕ) (initial_ripe : ℕ) (days : ℕ) : ℕ × ℕ :=
  sorry

theorem peaches_after_seven_days :
  let initial_total := 18
  let initial_ripe := 4
  let ripen_pattern (d : ℕ) := d + 1
  let eat_pattern (d : ℕ) := d
  let (ripe, unripe) := peaches_after_days initial_total initial_ripe 7
  ripe = 0 ∧ unripe = 0 :=
sorry

end NUMINAMATH_CALUDE_peaches_after_seven_days_l2988_298808


namespace NUMINAMATH_CALUDE_range_of_a_l2988_298879

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - a*2^(x+1) + a^2 - 1 ≥ 0) ↔ 
  a ∈ Set.Iic 1 ∪ Set.Ici 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2988_298879


namespace NUMINAMATH_CALUDE_permutation_combination_relation_l2988_298830

-- Define permutation function
def p (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

-- Define combination function
def c (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem permutation_combination_relation :
  ∃ k : ℕ, p 32 6 = k * c 32 6 ∧ k = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_relation_l2988_298830


namespace NUMINAMATH_CALUDE_min_floor_equation_l2988_298846

theorem min_floor_equation (n : ℕ) : 
  (∃ k : ℕ, k^2 + Int.floor (n / k^2 : ℚ) = 1991 ∧ 
   ∀ m : ℕ, m^2 + Int.floor (n / m^2 : ℚ) ≥ 1991) ↔ 
  990208 ≤ n ∧ n ≤ 991231 := by
sorry

end NUMINAMATH_CALUDE_min_floor_equation_l2988_298846


namespace NUMINAMATH_CALUDE_count_integers_with_8_and_9_between_700_and_1000_l2988_298847

def count_integers_with_8_and_9 (lower_bound upper_bound : ℕ) : ℕ :=
  (upper_bound - lower_bound + 1) / 100 * 2

theorem count_integers_with_8_and_9_between_700_and_1000 :
  count_integers_with_8_and_9 700 1000 = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_8_and_9_between_700_and_1000_l2988_298847


namespace NUMINAMATH_CALUDE_min_value_and_location_l2988_298813

theorem min_value_and_location (f : ℝ → ℝ) :
  (∀ x, f x = 2 * Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.cos (2 * x) ^ 2 - 3) →
  (∃ x_min ∈ Set.Icc (π / 16) (3 * π / 16), 
    (∀ x ∈ Set.Icc (π / 16) (3 * π / 16), f x_min ≤ f x) ∧
    x_min = 3 * π / 16 ∧
    f x_min = -(Real.sqrt 2 + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_location_l2988_298813


namespace NUMINAMATH_CALUDE_invitations_per_package_l2988_298819

theorem invitations_per_package 
  (total_packs : ℕ) 
  (total_invitations : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_invitations = 45) :
  total_invitations / total_packs = 9 :=
by sorry

end NUMINAMATH_CALUDE_invitations_per_package_l2988_298819


namespace NUMINAMATH_CALUDE_probability_of_common_books_l2988_298843

def total_books : ℕ := 12
def books_chosen : ℕ := 6
def books_in_common : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_chosen - books_in_common) * 
   Nat.choose (total_books - books_chosen) (books_chosen - books_in_common)) / 
  (Nat.choose total_books books_chosen * Nat.choose total_books books_chosen) = 50 / 116 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_common_books_l2988_298843


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2988_298873

theorem min_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 9) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b : ℝ),
    (a = (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))^2 ∧
     b = (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2) →
    a - b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2988_298873


namespace NUMINAMATH_CALUDE_ellipse_standard_form_l2988_298839

def ellipse_equation (x y t : ℝ) : Prop :=
  x = (3 * (Real.sin t - 2)) / (3 - Real.cos t) ∧
  y = (4 * (Real.cos t - 4)) / (3 - Real.cos t)

theorem ellipse_standard_form :
  ∃ (A B C D E F : ℤ),
    (∀ x y : ℝ, (∃ t : ℝ, ellipse_equation x y t) →
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧
    Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C)
      (Nat.gcd (Int.natAbs D) (Nat.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    A = 144 ∧ B = -144 ∧ C = 36 ∧ D = 0 ∧ E = 420 ∧ F = 1084 ∧
    Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D +
    Int.natAbs E + Int.natAbs F = 1828 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_form_l2988_298839


namespace NUMINAMATH_CALUDE_liner_and_water_speed_theorem_l2988_298832

/-- The distance between Chongqing and Shibaozhai in kilometers -/
def distance : ℝ := 270

/-- The time taken to travel downstream in hours -/
def downstream_time : ℝ := 9

/-- The time taken to travel upstream in hours -/
def upstream_time : ℝ := 13.5

/-- The speed of the liner in still water in km/h -/
def liner_speed : ℝ := 25

/-- The speed of the water flow in km/h -/
def water_speed : ℝ := 5

/-- The distance between Chongqing Port and the new dock in km -/
def new_dock_distance : ℝ := 162

theorem liner_and_water_speed_theorem :
  (downstream_time * (liner_speed + water_speed) = distance) ∧
  (upstream_time * (liner_speed - water_speed) = distance) ∧
  (new_dock_distance / (liner_speed + water_speed) = (distance - new_dock_distance) / (liner_speed - water_speed)) := by
  sorry

#check liner_and_water_speed_theorem

end NUMINAMATH_CALUDE_liner_and_water_speed_theorem_l2988_298832


namespace NUMINAMATH_CALUDE_translate_AB_to_origin_l2988_298869

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns the coordinates of B after translating 
    the line segment AB so that A coincides with the origin. -/
def translate_to_origin (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Theorem stating that translating the line segment AB 
    from A(-4, 0) to B(0, 2) so that A coincides with the origin 
    results in B having coordinates (4, 2). -/
theorem translate_AB_to_origin : 
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (0, 2)
  translate_to_origin A B = (4, 2) := by
  sorry


end NUMINAMATH_CALUDE_translate_AB_to_origin_l2988_298869


namespace NUMINAMATH_CALUDE_asha_win_probability_l2988_298817

theorem asha_win_probability (lose_prob : ℚ) (h1 : lose_prob = 4/9) :
  1 - lose_prob = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2988_298817


namespace NUMINAMATH_CALUDE_cos_105_cos_45_plus_sin_105_sin_45_l2988_298864

theorem cos_105_cos_45_plus_sin_105_sin_45 :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (105 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_cos_45_plus_sin_105_sin_45_l2988_298864


namespace NUMINAMATH_CALUDE_largest_common_term_under_1000_l2988_298881

/-- The first arithmetic progression {1, 4, 7, 10, ...} -/
def progression1 (n : ℕ) : ℕ := 1 + 3 * n

/-- The second arithmetic progression {5, 14, 23, 32, ...} -/
def progression2 (n : ℕ) : ℕ := 5 + 9 * n

/-- A term is common if it appears in both progressions -/
def is_common_term (a : ℕ) : Prop :=
  ∃ n m : ℕ, progression1 n = a ∧ progression2 m = a

theorem largest_common_term_under_1000 :
  (∀ a : ℕ, a < 1000 → is_common_term a → a ≤ 976) ∧
  is_common_term 976 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_under_1000_l2988_298881


namespace NUMINAMATH_CALUDE_no_common_solution_l2988_298857

theorem no_common_solution : ¬∃ y : ℝ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l2988_298857


namespace NUMINAMATH_CALUDE_min_value_of_function_l2988_298815

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + x ≥ 6 ∧ ∃ y > 2, 4 / (y - 2) + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2988_298815


namespace NUMINAMATH_CALUDE_opposite_of_four_l2988_298893

-- Define the concept of opposite number
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 4 is -4
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l2988_298893


namespace NUMINAMATH_CALUDE_simplify_expression_l2988_298848

theorem simplify_expression (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2988_298848


namespace NUMINAMATH_CALUDE_unruly_quadratic_max_sum_of_roots_l2988_298800

/-- A quadratic polynomial of the form q(x) = (x-r)^2 - s -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := (x - r)^2 - s

/-- The composition of a quadratic polynomial with itself -/
def ComposedQuadratic (r s : ℝ) (x : ℝ) : ℝ :=
  QuadraticPolynomial r s (QuadraticPolynomial r s x)

/-- Predicate for an unruly quadratic polynomial -/
def IsUnruly (r s : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), 
    (ComposedQuadratic r s x₁ = 0 ∧
     ComposedQuadratic r s x₂ = 0 ∧
     ComposedQuadratic r s x₃ = 0) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (∃ (x₄ : ℝ), ComposedQuadratic r s x₄ = 0 ∧
                 (∀ (x : ℝ), ComposedQuadratic r s x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (r s : ℝ) : ℝ := 2 * r

theorem unruly_quadratic_max_sum_of_roots :
  ∃ (r s : ℝ), IsUnruly r s ∧
    (∀ (r' s' : ℝ), IsUnruly r' s' → SumOfRoots r s ≥ SumOfRoots r' s') ∧
    QuadraticPolynomial r s 1 = 7/4 :=
sorry

end NUMINAMATH_CALUDE_unruly_quadratic_max_sum_of_roots_l2988_298800


namespace NUMINAMATH_CALUDE_triangle_sinB_sinC_l2988_298833

theorem triangle_sinB_sinC (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- 2c + b = 2a * cos B
  (2 * c + b = 2 * a * Real.cos B) →
  -- Area S = 3/2 * sqrt(3)
  (1/2 * b * c * Real.sin A = 3/2 * Real.sqrt 3) →
  -- c = 2
  (c = 2) →
  -- Then sin B * sin C = 9/38
  (Real.sin B * Real.sin C = 9/38) := by
sorry

end NUMINAMATH_CALUDE_triangle_sinB_sinC_l2988_298833


namespace NUMINAMATH_CALUDE_function_types_l2988_298876

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def p (x : ℝ) (a : ℝ) : ℝ := a * x^2

-- State the theorem
theorem function_types (a x : ℝ) (ha : a ≠ 0) (hx : x ≠ 0) :
  (∃ b c : ℝ, ∀ x, f a x = x^2 + b*x + c) ∧
  (∃ m b : ℝ, ∀ a, p x a = m*a + b) :=
sorry

end NUMINAMATH_CALUDE_function_types_l2988_298876


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l2988_298878

/-- Represents a box containing chocolate bars -/
structure ChocolateBox where
  bars : ℕ

/-- Represents a large box containing small boxes of chocolates -/
structure LargeBox where
  smallBoxes : ℕ
  smallBoxContents : ChocolateBox

/-- Calculates the total number of chocolate bars in a large box -/
def totalChocolateBars (box : LargeBox) : ℕ :=
  box.smallBoxes * box.smallBoxContents.bars

theorem chocolate_bar_count (largeBox : LargeBox) 
    (h1 : largeBox.smallBoxes = 15)
    (h2 : largeBox.smallBoxContents.bars = 20) : 
    totalChocolateBars largeBox = 300 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l2988_298878


namespace NUMINAMATH_CALUDE_line_points_k_value_l2988_298842

/-- Given a line represented by equations x = 2y + 5 and z = 3x - 4,
    and two points (m, n, p) and (m + 4, n + k, p + 3) lying on this line,
    prove that k = 2 -/
theorem line_points_k_value
  (m n p k : ℝ)
  (point1_on_line : m = 2 * n + 5 ∧ p = 3 * m - 4)
  (point2_on_line : (m + 4) = 2 * (n + k) + 5 ∧ (p + 3) = 3 * (m + 4) - 4) :
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l2988_298842


namespace NUMINAMATH_CALUDE_factor_expression_value_l2988_298812

theorem factor_expression_value (k m n : ℕ) : 
  k > 1 → m > 1 → n > 1 →
  (∃ (Z : ℕ), Z = 2^k * 3^m * 5^n ∧ (2^60 * 3^35 * 5^20 * 7^7) % Z = 0) →
  ∃ (k' m' n' : ℕ), k' > 1 ∧ m' > 1 ∧ n' > 1 ∧
    2^k' + 3^m' + k'^3 * m'^n' - n' = 43 :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_value_l2988_298812


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2988_298874

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 10) = 12 → y = 134 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2988_298874


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l2988_298804

/-- An arithmetic sequence {a_n} with a common ratio q -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_increasing : ∀ n, a (n + 1) > a n)
  (h_positive : a 1 > 0)
  (h_condition : ∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1))
  (h_arithmetic : ArithmeticSequence a q) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l2988_298804


namespace NUMINAMATH_CALUDE_georges_car_cylinders_l2988_298823

def oil_per_cylinder : ℕ := 8
def oil_already_added : ℕ := 16
def additional_oil_needed : ℕ := 32

theorem georges_car_cylinders :
  (oil_already_added + additional_oil_needed) / oil_per_cylinder = 6 :=
by sorry

end NUMINAMATH_CALUDE_georges_car_cylinders_l2988_298823


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l2988_298834

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l2988_298834


namespace NUMINAMATH_CALUDE_trig_problem_l2988_298829

theorem trig_problem (θ φ : Real) 
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧ 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ 
  Real.cos θ = -(Real.sqrt 5 / 5) ∧ 
  Real.cos φ = -(Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l2988_298829


namespace NUMINAMATH_CALUDE_halfway_fraction_l2988_298871

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) :
  (a + b) / 2 = 19/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2988_298871


namespace NUMINAMATH_CALUDE_lucys_cake_packs_l2988_298887

/-- Lucy's grocery shopping problem -/
theorem lucys_cake_packs (cookies chocolate total : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) :
  total - (cookies + chocolate) = 22 := by
  sorry

end NUMINAMATH_CALUDE_lucys_cake_packs_l2988_298887


namespace NUMINAMATH_CALUDE_max_product_of_sums_l2988_298840

theorem max_product_of_sums (a b c d e f : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a + b + c + d + e + f = 45 →
  (a + b + c) * (d + e + f) ≤ 550 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_sums_l2988_298840


namespace NUMINAMATH_CALUDE_expression_simplification_l2988_298849

theorem expression_simplification (x : ℝ) : 
  (3*x^2 + 4*x - 5)*(x - 2) + (x - 2)*(2*x^2 - 3*x + 9) - (4*x - 7)*(x - 2)*(x - 3) = 
  x^3 + x^2 + 12*x - 36 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2988_298849


namespace NUMINAMATH_CALUDE_cubic_three_roots_range_l2988_298858

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The number of distinct real roots of f -/
noncomputable def num_distinct_roots (a : ℝ) : ℕ := sorry

theorem cubic_three_roots_range (a : ℝ) :
  num_distinct_roots a = 3 → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_three_roots_range_l2988_298858


namespace NUMINAMATH_CALUDE_no_rational_roots_for_odd_m_n_l2988_298811

theorem no_rational_roots_for_odd_m_n (m n : ℤ) (hm : Odd m) (hn : Odd n) :
  ∀ x : ℚ, x^2 + 2*m*x + 2*n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_odd_m_n_l2988_298811


namespace NUMINAMATH_CALUDE_odd_digits_base4_437_l2988_298805

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 437 is 4 -/
theorem odd_digits_base4_437 : countOddDigits (toBase4 437) = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_437_l2988_298805


namespace NUMINAMATH_CALUDE_derivative_tangent_line_existence_no_derivative_no_tangent_slope_l2988_298860

-- Define a real-valued function f
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define the existence of a derivative at x₀
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - L * (x - x₀)| ≤ ε * |x - x₀|

-- Define the existence of a tangent line at x₀
def has_tangent_line_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m b, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - (m * x + b)| < ε * |x - x₀|

-- Define the existence of a slope of the tangent line at x₀
def has_tangent_slope_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - m * (x - x₀)| < ε * |x - x₀|

-- Theorem 1: Non-existence of derivative doesn't imply non-existence of tangent line
theorem derivative_tangent_line_existence (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → (has_tangent_line_at f x₀ ∨ ¬(has_tangent_line_at f x₀)) :=
sorry

-- Theorem 2: Non-existence of derivative implies non-existence of tangent slope
theorem no_derivative_no_tangent_slope (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → ¬(has_tangent_slope_at f x₀) :=
sorry

end NUMINAMATH_CALUDE_derivative_tangent_line_existence_no_derivative_no_tangent_slope_l2988_298860


namespace NUMINAMATH_CALUDE_total_stamps_l2988_298841

theorem total_stamps (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l2988_298841


namespace NUMINAMATH_CALUDE_isosceles_triangle_fold_crease_length_l2988_298827

theorem isosceles_triangle_fold_crease_length 
  (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 5 ∧ c = 6) :
  let m := c / 2
  let crease_length := Real.sqrt (a^2 + m^2)
  crease_length = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_fold_crease_length_l2988_298827


namespace NUMINAMATH_CALUDE_partnership_profit_l2988_298851

/-- Partnership profit calculation -/
theorem partnership_profit (a b c : ℚ) (b_share : ℚ) : 
  a = 3 * b ∧ b = (2/3) * c ∧ b_share = 600 →
  (11/2) * b_share = 3300 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l2988_298851


namespace NUMINAMATH_CALUDE_apple_grape_equivalence_l2988_298883

/-- If 3/4 of 12 apples are worth as much as 6 grapes, then 1/3 of 9 apples are worth as much as 2 grapes -/
theorem apple_grape_equivalence (apple_value grape_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 6 * grape_value → 
  (1 / 3 * 9 : ℚ) * apple_value = 2 * grape_value := by
  sorry

end NUMINAMATH_CALUDE_apple_grape_equivalence_l2988_298883
