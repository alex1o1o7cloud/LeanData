import Mathlib

namespace NUMINAMATH_CALUDE_greatest_product_of_digits_divisible_by_35_l559_55976

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_single_digit : tens < 10
  units_single_digit : units < 10

/-- Check if a number is divisible by another number -/
def isDivisibleBy (n m : Nat) : Prop := ∃ k, n = m * k

theorem greatest_product_of_digits_divisible_by_35 :
  ∀ n : TwoDigitNumber,
    isDivisibleBy (10 * n.tens + n.units) 35 →
    ∀ m : TwoDigitNumber,
      isDivisibleBy (10 * m.tens + m.units) 35 →
      n.units * n.tens ≤ 40 ∧
      (m.units * m.tens = 40 → n.units * n.tens = 40) :=
sorry

end NUMINAMATH_CALUDE_greatest_product_of_digits_divisible_by_35_l559_55976


namespace NUMINAMATH_CALUDE_probability_one_suit_each_probability_calculation_correct_l559_55964

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def NumberOfDraws : ℕ := 4

/-- Represents the probability of drawing one card from each suit in four draws with replacement -/
def ProbabilityOneSuitEach : ℚ := 3 / 32

/-- Theorem stating that the probability of drawing one card from each suit
    in four draws with replacement from a standard 52-card deck is 3/32 -/
theorem probability_one_suit_each :
  (3 / 4 : ℚ) * (1 / 2 : ℚ) * (1 / 4 : ℚ) = ProbabilityOneSuitEach :=
by sorry

/-- Theorem stating that the calculated probability is correct -/
theorem probability_calculation_correct :
  ProbabilityOneSuitEach = (3 : ℚ) / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_suit_each_probability_calculation_correct_l559_55964


namespace NUMINAMATH_CALUDE_compute_expression_l559_55979

theorem compute_expression : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l559_55979


namespace NUMINAMATH_CALUDE_officer_selection_count_l559_55948

/-- The number of members in the club -/
def clubSize : ℕ := 12

/-- The number of officers to be elected -/
def officerCount : ℕ := 5

/-- Calculates the number of ways to select distinct officers from club members -/
def officerSelections (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun acc i => acc * (n - i)) 1

/-- Theorem stating the number of ways to select 5 distinct officers from 12 members -/
theorem officer_selection_count :
  officerSelections clubSize officerCount = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l559_55948


namespace NUMINAMATH_CALUDE_horizontal_figure_area_l559_55970

/-- Represents a horizontally placed figure with specific properties -/
structure HorizontalFigure where
  /-- The oblique section diagram is an isosceles trapezoid -/
  is_isosceles_trapezoid : Bool
  /-- The base angle of the trapezoid is 45° -/
  base_angle : ℝ
  /-- The length of the legs of the trapezoid -/
  leg_length : ℝ
  /-- The length of the upper base of the trapezoid -/
  upper_base_length : ℝ

/-- Calculates the area of the original plane figure -/
def area (fig : HorizontalFigure) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure -/
theorem horizontal_figure_area (fig : HorizontalFigure) 
  (h1 : fig.is_isosceles_trapezoid = true)
  (h2 : fig.base_angle = π / 4)
  (h3 : fig.leg_length = 1)
  (h4 : fig.upper_base_length = 1) :
  area fig = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_horizontal_figure_area_l559_55970


namespace NUMINAMATH_CALUDE_joan_change_l559_55913

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost bill_amount : ℚ) : 
  cat_toy_cost = 8.77 →
  cage_cost = 10.97 →
  bill_amount = 20 →
  bill_amount - (cat_toy_cost + cage_cost) = 0.26 := by
sorry

end NUMINAMATH_CALUDE_joan_change_l559_55913


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l559_55934

/-- Hyperbola properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  eccentricity : ℝ
  vertex_to_asymptote : ℝ

/-- Theorem: Given a hyperbola with specific eccentricity and vertex-to-asymptote distance, prove its parameters -/
theorem hyperbola_parameters (h : Hyperbola) 
  (h_eccentricity : h.eccentricity = Real.sqrt 6 / 2)
  (h_vertex_to_asymptote : h.vertex_to_asymptote = 2 * Real.sqrt 6 / 3) :
  h.a = 2 * Real.sqrt 2 ∧ h.b = 2 := by
  sorry

#check hyperbola_parameters

end NUMINAMATH_CALUDE_hyperbola_parameters_l559_55934


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_square_l559_55917

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

theorem sum_to_k_perfect_square (k : ℕ) :
  k ≤ 49 →
  (∃ n : ℕ, n < 100 ∧ sum_to_k k = n^2) ↔
  k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_square_l559_55917


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_is_13_l559_55919

/-- Conference attendee information -/
structure ConferenceData where
  total : Nat
  writers : Nat
  editors : Nat
  both : Nat
  neither : Nat
  editors_gt_38 : editors > 38
  neither_eq_2both : neither = 2 * both
  total_sum : total = writers + editors - both + neither

/-- The maximum number of people who can be both writers and editors -/
def max_both_writers_and_editors (data : ConferenceData) : Nat :=
  13

/-- Theorem stating that 13 is the maximum number of people who can be both writers and editors -/
theorem max_both_writers_and_editors_is_13 (data : ConferenceData) 
  (h : data.total = 110 ∧ data.writers = 45) :
  max_both_writers_and_editors data = 13 := by
  sorry

#check max_both_writers_and_editors_is_13

end NUMINAMATH_CALUDE_max_both_writers_and_editors_is_13_l559_55919


namespace NUMINAMATH_CALUDE_inequality_solution_l559_55906

theorem inequality_solution (a b : ℝ) (h1 : ∀ x, (2*a - b)*x + a - 5*b > 0 ↔ x < 10/7) :
  ∀ x, a*x + b > 0 ↔ x < -3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l559_55906


namespace NUMINAMATH_CALUDE_min_omega_for_translated_sine_l559_55950

theorem min_omega_for_translated_sine (ω : ℝ) (h1 : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4 - π / 4) = k * π) →
  (∀ ω' : ℝ, ω' > 0 → (∃ k : ℤ, ω' * (3 * π / 4 - π / 4) = k * π) → ω' ≥ ω) →
  ω = 2 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_translated_sine_l559_55950


namespace NUMINAMATH_CALUDE_quarters_in_school_year_l559_55918

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The number of students in the art club -/
def students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of quarters in a school year is 4 -/
theorem quarters_in_school_year : 
  quarters_per_year * 2 * students * artworks_per_student_per_quarter = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_quarters_in_school_year_l559_55918


namespace NUMINAMATH_CALUDE_battle_station_staffing_l559_55931

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def permutations (n k : ℕ) : ℕ := 
  factorial n / factorial (n - k)

theorem battle_station_staffing :
  permutations 15 5 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l559_55931


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l559_55942

theorem right_triangle_leg_sum : ∃ (a b : ℕ), 
  (a + 1 = b) ∧                -- legs are consecutive whole numbers
  (a^2 + b^2 = 41^2) ∧         -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=              -- sum of legs is 57
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l559_55942


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l559_55903

/-- Given a man's rowing speed against the stream and in still water, 
    calculate his speed with the stream. -/
theorem mans_speed_with_stream 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 11) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 18 := by
  sorry

#check mans_speed_with_stream

end NUMINAMATH_CALUDE_mans_speed_with_stream_l559_55903


namespace NUMINAMATH_CALUDE_sequence_term_proof_l559_55975

def sequence_sum (n : ℕ) : ℕ := 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n-1)

theorem sequence_term_proof :
  ∀ n : ℕ, n ≥ 1 →
    sequence_term n = (if n = 1 then sequence_sum 1 else sequence_sum n - sequence_sum (n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l559_55975


namespace NUMINAMATH_CALUDE_inequality_proof_l559_55951

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l559_55951


namespace NUMINAMATH_CALUDE_parabola_point_order_l559_55990

/-- Given a parabola y = 2x² - 4x + m and three points on it, prove their y-coordinates are in ascending order -/
theorem parabola_point_order (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 2 * 3^2 - 4 * 3 + m)
  (h₂ : y₂ = 2 * 4^2 - 4 * 4 + m)
  (h₃ : y₃ = 2 * 5^2 - 4 * 5 + m) :
  y₁ < y₂ ∧ y₂ < y₃ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_order_l559_55990


namespace NUMINAMATH_CALUDE_diamond_seven_three_l559_55958

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ :=
  sorry

-- Axioms for the diamond operation
axiom diamond_zero (x : ℝ) : diamond x 0 = x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_rec (x y : ℝ) : diamond (x + 2) y = diamond x y + y + 2

-- Theorem to prove
theorem diamond_seven_three : diamond 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_diamond_seven_three_l559_55958


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l559_55972

theorem quadratic_equation_single_solution_sum (b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + b * x + 6 * x + 10
  (∃! x, f x = 0) → 
  ∃ b₁ b₂, b = b₁ ∨ b = b₂ ∧ b₁ + b₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l559_55972


namespace NUMINAMATH_CALUDE_existence_of_a_values_l559_55961

theorem existence_of_a_values (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_values_l559_55961


namespace NUMINAMATH_CALUDE_max_value_d_l559_55932

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_d_l559_55932


namespace NUMINAMATH_CALUDE_ship_power_at_6_knots_l559_55922

-- Define the quadratic function
def H (a b c : ℝ) (v : ℝ) : ℝ := a * v^2 + b * v + c

-- State the theorem
theorem ship_power_at_6_knots 
  (a b c : ℝ) 
  (h1 : H a b c 5 = 300)
  (h2 : H a b c 7 = 780)
  (h3 : H a b c 9 = 1420) :
  H a b c 6 = 520 := by
  sorry

end NUMINAMATH_CALUDE_ship_power_at_6_knots_l559_55922


namespace NUMINAMATH_CALUDE_owl_money_problem_l559_55936

theorem owl_money_problem (x : ℚ) : 
  (((3 * ((3 * ((3 * ((3 * x) - 50)) - 50)) - 50)) - 50) = 0) → 
  (x = 2000 / 81) := by
sorry

end NUMINAMATH_CALUDE_owl_money_problem_l559_55936


namespace NUMINAMATH_CALUDE_natural_number_representation_l559_55911

theorem natural_number_representation (n : ℕ) : 
  ∃ (x y : ℕ), n = x^3 / y^4 := by sorry

end NUMINAMATH_CALUDE_natural_number_representation_l559_55911


namespace NUMINAMATH_CALUDE_father_daughter_ages_l559_55908

theorem father_daughter_ages (father daughter : ℕ) : 
  father = 4 * daughter ∧ 
  father + 20 = 2 * (daughter + 20) → 
  father = 40 ∧ daughter = 10 := by
sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l559_55908


namespace NUMINAMATH_CALUDE_cards_in_basketball_box_dexter_basketball_cards_l559_55996

/-- The number of cards in each basketball card box -/
def cards_per_basketball_box (total_cards : ℕ) (basketball_boxes : ℕ) (football_boxes : ℕ) (cards_per_football_box : ℕ) : ℕ :=
  (total_cards - football_boxes * cards_per_football_box) / basketball_boxes

/-- Theorem stating the number of cards in each basketball card box -/
theorem cards_in_basketball_box :
  cards_per_basketball_box 255 9 6 20 = 15 := by
  sorry

/-- Main theorem proving the problem statement -/
theorem dexter_basketball_cards :
  ∃ (total_cards basketball_boxes football_boxes cards_per_football_box : ℕ),
    total_cards = 255 ∧
    basketball_boxes = 9 ∧
    football_boxes = basketball_boxes - 3 ∧
    cards_per_football_box = 20 ∧
    cards_per_basketball_box total_cards basketball_boxes football_boxes cards_per_football_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_basketball_box_dexter_basketball_cards_l559_55996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l559_55960

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l559_55960


namespace NUMINAMATH_CALUDE_surrounding_circle_area_l559_55933

/-- Given a circle of radius R surrounded by four equal circles, each touching 
    the given circle and each other, the area of one surrounding circle is πR²(3 + 2√2) -/
theorem surrounding_circle_area (R : ℝ) (R_pos : R > 0) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (R + r)^2 + (R + r)^2 = (2*r)^2 ∧ 
    r = R * (1 + Real.sqrt 2) ∧
    π * r^2 = π * R^2 * (3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_surrounding_circle_area_l559_55933


namespace NUMINAMATH_CALUDE_five_lapping_points_l559_55955

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- The circular track model -/
def CircularTrack := Unit

/-- Calculates the number of lapping points on a circular track -/
def numberOfLappingPoints (track : CircularTrack) (a b : Runner) : ℕ :=
  sorry

theorem five_lapping_points (track : CircularTrack) (a b : Runner) :
  a.speed > 0 ∧ b.speed > 0 ∧
  a.initialPosition = b.initialPosition + 10 ∧
  b.speed * 22 = a.speed * 32 →
  numberOfLappingPoints track a b = 5 :=
sorry

end NUMINAMATH_CALUDE_five_lapping_points_l559_55955


namespace NUMINAMATH_CALUDE_alice_average_speed_l559_55953

/-- Alice's cycling trip -/
theorem alice_average_speed :
  let distance1 : ℝ := 40  -- First segment distance in miles
  let speed1 : ℝ := 8      -- First segment speed in miles per hour
  let distance2 : ℝ := 20  -- Second segment distance in miles
  let speed2 : ℝ := 40     -- Second segment speed in miles per hour
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 120 / 11
  := by sorry

end NUMINAMATH_CALUDE_alice_average_speed_l559_55953


namespace NUMINAMATH_CALUDE_parkway_elementary_students_l559_55980

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := 500

/-- The number of boys in the fifth grade -/
def boys : ℕ := 350

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 86 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 115

/-- Theorem stating that the total number of students is 500 -/
theorem parkway_elementary_students :
  total_students = boys + girls_not_soccer + (soccer_players - (boys_soccer_percentage * soccer_players).num) :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_students_l559_55980


namespace NUMINAMATH_CALUDE_mango_tree_count_l559_55902

theorem mango_tree_count (mango_count coconut_count : ℕ) : 
  coconut_count = mango_count / 2 - 5 →
  mango_count + coconut_count = 85 →
  mango_count = 60 := by
sorry

end NUMINAMATH_CALUDE_mango_tree_count_l559_55902


namespace NUMINAMATH_CALUDE_painted_area_is_33_l559_55939

/-- Represents the arrangement of cubes -/
structure CubeArrangement where
  width : Nat
  length : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the total painted area for a given cube arrangement -/
def painted_area (arr : CubeArrangement) : Nat :=
  let top_area := arr.width * arr.length
  let side_area := 2 * (arr.width * arr.height + arr.length * arr.height)
  top_area + side_area

/-- The specific arrangement described in the problem -/
def problem_arrangement : CubeArrangement :=
  { width := 3
  , length := 3
  , height := 1
  , total_cubes := 14 }

/-- Theorem stating that the painted area for the given arrangement is 33 square meters -/
theorem painted_area_is_33 : painted_area problem_arrangement = 33 := by
  sorry

end NUMINAMATH_CALUDE_painted_area_is_33_l559_55939


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l559_55956

/-- The area of a square with adjacent vertices at (-1, 4) and (2, -3) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (-1, 4)
  let p2 : ℝ × ℝ := (2, -3)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l559_55956


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l559_55963

theorem trig_fraction_equality (α : ℝ) (h : (1 + Real.sin α) / Real.cos α = -1/2) :
  Real.cos α / (Real.sin α - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l559_55963


namespace NUMINAMATH_CALUDE_unique_kids_count_l559_55999

/-- The number of unique kids Julia played with across the week -/
def total_unique_kids (monday tuesday wednesday thursday friday : ℕ) 
  (wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday : ℕ) : ℕ :=
  monday + tuesday + (wednesday - wednesday_from_monday) + 
  (thursday - thursday_from_tuesday) + 
  (friday - friday_from_monday - (friday_from_wednesday - wednesday_from_monday))

theorem unique_kids_count :
  let monday := 12
  let tuesday := 7
  let wednesday := 15
  let thursday := 10
  let friday := 18
  let wednesday_from_monday := 5
  let thursday_from_tuesday := 7
  let friday_from_monday := 9
  let friday_from_wednesday := 5
  total_unique_kids monday tuesday wednesday thursday friday
    wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_kids_count_l559_55999


namespace NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l559_55995

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailing_zeros 345 = 84 := by
  sorry

end NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l559_55995


namespace NUMINAMATH_CALUDE_rectangle_side_length_l559_55974

/-- Given a rectangle with its bottom side on the x-axis from (-a, 0) to (a, 0),
    its top side on the parabola y = x^2, and its area equal to 81,
    prove that the length of its side parallel to the x-axis is 2∛(40.5). -/
theorem rectangle_side_length (a : ℝ) : 
  (2 * a * a^2 = 81) →  -- Area of the rectangle
  (2 * a = 2 * (40.5 : ℝ)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l559_55974


namespace NUMINAMATH_CALUDE_reflection_theorem_l559_55938

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)
  let p'' := (-p'.2, -p'.1)
  (p''.1, p''.2 + 1)

def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (7, 4)

theorem reflection_theorem :
  reflect_line (reflect_x C) = (-5, 8) := by sorry

end NUMINAMATH_CALUDE_reflection_theorem_l559_55938


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l559_55985

/-- Represents a traffic light cycle with durations for each color -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (c : TrafficLightCycle) : ℕ :=
  c.green + c.yellow + c.red

/-- Calculates the number of seconds where a color change can be observed in a 3-second interval -/
def changeObservationWindow (c : TrafficLightCycle) : ℕ :=
  3 * 3  -- 3 transitions, each with a 3-second window

/-- The probability of observing a color change in a random 3-second interval -/
def probabilityOfChange (c : TrafficLightCycle) : ℚ :=
  changeObservationWindow c / cycleDuration c

theorem traffic_light_change_probability :
  let c : TrafficLightCycle := ⟨50, 2, 40⟩
  probabilityOfChange c = 9 / 92 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l559_55985


namespace NUMINAMATH_CALUDE_base_height_example_l559_55966

/-- Given a sculpture height in feet and inches, and a total height of sculpture and base,
    calculate the height of the base in feet. -/
def base_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (total_height : ℚ) : ℚ :=
  total_height - (sculpture_feet : ℚ) - ((sculpture_inches : ℚ) / 12)

/-- Theorem stating that for a sculpture of 2 feet 10 inches and a total height of 3.5 feet,
    the base height is 2/3 feet. -/
theorem base_height_example : base_height 2 10 (7/2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_base_height_example_l559_55966


namespace NUMINAMATH_CALUDE_inequality_solution_set_l559_55943

theorem inequality_solution_set (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔ 
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ x > -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l559_55943


namespace NUMINAMATH_CALUDE_quadratic_equation_linear_term_l559_55952

theorem quadratic_equation_linear_term 
  (m : ℝ) 
  (h : 2 * m = 6) : 
  ∃ (a b c : ℝ), 
    a * x^2 + b * x + c = 0 ∧ 
    c = 6 ∧ 
    b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_linear_term_l559_55952


namespace NUMINAMATH_CALUDE_hyperbola_condition_l559_55927

/-- A curve is a hyperbola if it can be represented by an equation of the form
    (x²/a²) - (y²/b²) = 1 or (y²/a²) - (x²/b²) = 1, where a and b are non-zero real numbers. -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
    (∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∨
    (∀ x y, f x y ↔ (y^2 / a^2) - (x^2 / b^2) = 1)

/-- The curve represented by the equation x²/(k-3) - y²/(k+3) = 1 -/
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

theorem hyperbola_condition (k : ℝ) :
  (k > 3 → is_hyperbola (curve k)) ∧
  ¬(is_hyperbola (curve k) → k > 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l559_55927


namespace NUMINAMATH_CALUDE_tunnel_length_l559_55912

/-- Calculates the length of a tunnel given train parameters and transit time -/
theorem tunnel_length
  (train_length : Real)
  (train_speed_kmh : Real)
  (transit_time_min : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 72)
  (h3 : transit_time_min = 2.5) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let transit_time_s : Real := transit_time_min * 60
  let total_distance : Real := train_speed_ms * transit_time_s
  let tunnel_length_m : Real := total_distance - train_length
  let tunnel_length_km : Real := tunnel_length_m / 1000
  tunnel_length_km = 2.9 := by
sorry

end NUMINAMATH_CALUDE_tunnel_length_l559_55912


namespace NUMINAMATH_CALUDE_total_weight_problem_l559_55924

/-- The total weight problem -/
theorem total_weight_problem (a b c d : ℕ) 
  (h1 : a + b = 250)
  (h2 : b + c = 235)
  (h3 : c + d = 260)
  (h4 : a + d = 275) :
  a + b + c + d = 510 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_problem_l559_55924


namespace NUMINAMATH_CALUDE_masha_meeting_time_l559_55920

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents the scenario of Masha's journey home -/
structure MashaJourney where
  usual_end_time : Time
  usual_arrival_time : Time
  early_end_time : Time
  early_arrival_time : Time
  meeting_time : Time

/-- Calculate the time difference in minutes between two Time values -/
def time_diff_minutes (t1 t2 : Time) : ℤ :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The main theorem to prove -/
theorem masha_meeting_time (journey : MashaJourney) : 
  journey.usual_end_time = ⟨13, 0, by norm_num⟩ →
  journey.early_end_time = ⟨12, 0, by norm_num⟩ →
  time_diff_minutes journey.usual_arrival_time journey.early_arrival_time = 12 →
  journey.meeting_time = ⟨12, 54, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_masha_meeting_time_l559_55920


namespace NUMINAMATH_CALUDE_adam_figurines_count_l559_55910

/-- Adam's wood carving shop problem -/
theorem adam_figurines_count :
  -- Define the number of figurines per block for each wood type
  let basswood_figurines : ℕ := 3
  let butternut_figurines : ℕ := 4
  let aspen_figurines : ℕ := 2 * basswood_figurines
  let oak_figurines : ℕ := 5
  let cherry_figurines : ℕ := 7

  -- Define the number of blocks for each wood type
  let basswood_blocks : ℕ := 25
  let butternut_blocks : ℕ := 30
  let aspen_blocks : ℕ := 35
  let oak_blocks : ℕ := 40
  let cherry_blocks : ℕ := 45

  -- Calculate total figurines
  let total_figurines : ℕ := 
    basswood_blocks * basswood_figurines +
    butternut_blocks * butternut_figurines +
    aspen_blocks * aspen_figurines +
    oak_blocks * oak_figurines +
    cherry_blocks * cherry_figurines

  -- Prove that the total number of figurines is 920
  total_figurines = 920 := by
  sorry


end NUMINAMATH_CALUDE_adam_figurines_count_l559_55910


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_345_triangle_l559_55928

/-- The maximum area of a rectangle inscribed in a 3-4-5 right triangle -/
theorem max_area_rectangle_in_345_triangle : 
  ∃ (A : ℝ), A = 3 ∧ 
  ∀ (x y : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 
    (x ≤ 4 ∧ y ≤ 3 - (3/4) * x) ∨ (y ≤ 3 ∧ x ≤ 4 - (4/3) * y) →
    x * y ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_345_triangle_l559_55928


namespace NUMINAMATH_CALUDE_no_solution_equations_l559_55941

theorem no_solution_equations :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |2*x| + 3 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (x + 3) - 1 = 0) ∧
  (∃ x : ℝ, Real.sqrt (4 - x) - 3 = 0) ∧
  (∃ x : ℝ, |2*x| - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equations_l559_55941


namespace NUMINAMATH_CALUDE_hyperbola_parabola_shared_focus_l559_55921

/-- The focus of a parabola y² = 8x is at (2, 0) -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The equation of the hyperbola (x²/a² - y²/3 = 1) with a > 0 -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ x^2 / a^2 - y^2 / 3 = 1

/-- The focus of the hyperbola coincides with the focus of the parabola -/
def hyperbola_focus (a : ℝ) : ℝ × ℝ := parabola_focus

/-- Theorem: The value of 'a' for the hyperbola sharing a focus with the parabola is 1 -/
theorem hyperbola_parabola_shared_focus :
  ∃ (a : ℝ), is_hyperbola a (hyperbola_focus a).1 (hyperbola_focus a).2 ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_shared_focus_l559_55921


namespace NUMINAMATH_CALUDE_time_conversions_and_difference_l559_55989

/-- Converts 12-hour time (PM) to 24-hour format -/
def convert_pm_to_24h (hour : Nat) : Nat :=
  hour + 12

/-- Calculates the time difference in minutes between two times in 24-hour format -/
def time_diff_minutes (start_hour start_min end_hour end_min : Nat) : Nat :=
  (end_hour * 60 + end_min) - (start_hour * 60 + start_min)

theorem time_conversions_and_difference :
  (convert_pm_to_24h 5 = 17) ∧
  (convert_pm_to_24h 10 = 22) ∧
  (time_diff_minutes 16 40 17 20 = 40) :=
by sorry

end NUMINAMATH_CALUDE_time_conversions_and_difference_l559_55989


namespace NUMINAMATH_CALUDE_triangle_angle_less_than_right_angle_l559_55904

theorem triangle_angle_less_than_right_angle 
  (A B C : ℝ) (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 2/b = 1/a + 1/c) : B < π/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_less_than_right_angle_l559_55904


namespace NUMINAMATH_CALUDE_square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l559_55901

/-- The smallest positive integer x such that 720x is a perfect square -/
def smallest_square_factor : ℕ := 5

/-- The smallest positive integer y such that 720y is a perfect fourth power -/
def smallest_fourth_power_factor : ℕ := 1125

/-- 720 * smallest_square_factor is a perfect square -/
theorem square_property : ∃ (n : ℕ), 720 * smallest_square_factor = n^2 := by sorry

/-- 720 * smallest_fourth_power_factor is a perfect fourth power -/
theorem fourth_power_property : ∃ (n : ℕ), 720 * smallest_fourth_power_factor = n^4 := by sorry

/-- smallest_square_factor is the smallest positive integer with the square property -/
theorem smallest_square :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_square_factor → ¬∃ (n : ℕ), 720 * k = n^2 := by sorry

/-- smallest_fourth_power_factor is the smallest positive integer with the fourth power property -/
theorem smallest_fourth_power :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_fourth_power_factor → ¬∃ (n : ℕ), 720 * k = n^4 := by sorry

/-- The sum of smallest_square_factor and smallest_fourth_power_factor -/
def sum_of_factors : ℕ := smallest_square_factor + smallest_fourth_power_factor

/-- The sum of the factors is 1130 -/
theorem sum_is_1130 : sum_of_factors = 1130 := by sorry

end NUMINAMATH_CALUDE_square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l559_55901


namespace NUMINAMATH_CALUDE_coins_taken_out_l559_55965

/-- The number of coins Tina put in during the first hour -/
def first_hour_coins : ℕ := 20

/-- The number of coins Tina put in during each of the second and third hours -/
def second_third_hour_coins : ℕ := 30

/-- The number of coins Tina put in during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins left in the jar after the fifth hour -/
def coins_left : ℕ := 100

/-- The total number of coins Tina put in the jar -/
def total_coins_in : ℕ := first_hour_coins + 2 * second_third_hour_coins + fourth_hour_coins

/-- Theorem: The number of coins Tina's mother took out is equal to the total number of coins Tina put in minus the number of coins left in the jar after the fifth hour -/
theorem coins_taken_out : total_coins_in - coins_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_coins_taken_out_l559_55965


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_17_vertex_graph_l559_55978

/-- A coloring of edges in a complete graph -/
def EdgeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph has a monochromatic triangle if there exist three vertices
    such that all edges between them have the same color -/
def has_monochromatic_triangle (n : ℕ) (coloring : EdgeColoring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    coloring i j = coloring j k ∧ coloring j k = coloring i k

/-- In any complete graph with 17 vertices where each edge is colored in one of three colors,
    there exist three vertices such that all edges between them are the same color -/
theorem monochromatic_triangle_in_17_vertex_graph :
  ∀ (coloring : EdgeColoring 17), has_monochromatic_triangle 17 coloring :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_in_17_vertex_graph_l559_55978


namespace NUMINAMATH_CALUDE_parallel_unit_vector_l559_55984

/-- Given a vector a = (12, 5), prove that its parallel unit vector is (12/13, 5/13) or (-12/13, -5/13) -/
theorem parallel_unit_vector (a : ℝ × ℝ) (h : a = (12, 5)) :
  ∃ u : ℝ × ℝ, (u.1 * u.1 + u.2 * u.2 = 1) ∧ 
  (∃ k : ℝ, u.1 = k * a.1 ∧ u.2 = k * a.2) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_unit_vector_l559_55984


namespace NUMINAMATH_CALUDE_trip_length_calculation_l559_55959

theorem trip_length_calculation (total : ℚ) 
  (h1 : total / 4 + 16 + total / 6 = total) : total = 192 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_length_calculation_l559_55959


namespace NUMINAMATH_CALUDE_pastry_combinations_l559_55988

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def combinations_with_repetition (n k : ℕ) : ℕ := 
  Nat.choose (n + k - 1) k

/-- The number of pastry types available -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries to be bought -/
def total_pastries : ℕ := 9

/-- Theorem stating that the number of ways to buy 9 pastries from 3 types is 55 -/
theorem pastry_combinations : 
  combinations_with_repetition total_pastries num_pastry_types = 55 := by
  sorry

end NUMINAMATH_CALUDE_pastry_combinations_l559_55988


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l559_55914

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l559_55914


namespace NUMINAMATH_CALUDE_public_foundation_share_l559_55907

/-- Represents the distribution of charitable funds by a private company. -/
structure CharityFunds where
  X : ℝ  -- Total amount raised
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ  -- Number of organizations in public foundation
  W : ℕ  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group
  B : ℝ  -- Amount received by special project
  h1 : Y > 0 ∧ Y ≤ 100  -- Ensure Y is a valid percentage
  h2 : Z > 0  -- Ensure there's at least one organization in the public foundation
  h3 : W > 0  -- Ensure there's at least one local non-profit group
  h4 : X > 0  -- Ensure a positive amount is raised
  h5 : B = (1/3) * X * (1 - Y/100)  -- Amount received by special project
  h6 : A = (2/3) * X * (1 - Y/100) / W  -- Amount received by each local non-profit group

/-- Theorem stating the amount received by each organization in the public foundation. -/
theorem public_foundation_share (cf : CharityFunds) :
  (cf.Y / 100) * cf.X / cf.Z = (cf.Y / 100) * cf.X / cf.Z :=
by sorry

end NUMINAMATH_CALUDE_public_foundation_share_l559_55907


namespace NUMINAMATH_CALUDE_cost_increase_l559_55916

theorem cost_increase (t b : ℝ) : 
  let original_cost := t * b^5
  let new_cost := (3*t) * (2*b)^5
  (new_cost / original_cost) * 100 = 9600 := by
sorry

end NUMINAMATH_CALUDE_cost_increase_l559_55916


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l559_55962

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let group1_customers : ℕ := 4
  let group1_paintings_per_customer : ℕ := 2
  let group2_customers : ℕ := 12
  let group2_paintings_per_customer : ℕ := 1
  let group3_customers : ℕ := 4
  let group3_paintings_per_customer : ℕ := 4
  let total_paintings_sold := 
    group1_customers * group1_paintings_per_customer +
    group2_customers * group2_paintings_per_customer +
    group3_customers * group3_paintings_per_customer
  total_customers = group1_customers + group2_customers + group3_customers →
  total_paintings_sold = 36 := by
sorry


end NUMINAMATH_CALUDE_tracy_art_fair_sales_l559_55962


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l559_55945

theorem factorial_fraction_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * N * (N - 1)) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l559_55945


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_seven_l559_55940

theorem remainder_of_power_minus_seven (n : Nat) : (10^23 - 7) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_seven_l559_55940


namespace NUMINAMATH_CALUDE_absent_men_count_solve_work_scenario_l559_55954

/-- Represents the work scenario with absences -/
structure WorkScenario where
  total_men : ℕ
  original_days : ℕ
  actual_days : ℕ
  absent_men : ℕ

/-- Calculates the total work in man-days -/
def total_work (s : WorkScenario) : ℕ := s.total_men * s.original_days

/-- Calculates the work done by remaining men -/
def remaining_work (s : WorkScenario) : ℕ := (s.total_men - s.absent_men) * s.actual_days

/-- Theorem stating that 8 men became absent -/
theorem absent_men_count (s : WorkScenario) 
  (h1 : s.total_men = 48)
  (h2 : s.original_days = 15)
  (h3 : s.actual_days = 18)
  (h4 : total_work s = remaining_work s) :
  s.absent_men = 8 := by
  sorry

/-- Main theorem proving the solution -/
theorem solve_work_scenario : 
  ∃ (s : WorkScenario), s.total_men = 48 ∧ s.original_days = 15 ∧ s.actual_days = 18 ∧ 
  total_work s = remaining_work s ∧ s.absent_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_absent_men_count_solve_work_scenario_l559_55954


namespace NUMINAMATH_CALUDE_chicken_ratio_problem_l559_55968

/-- Given the following conditions:
    - Wendi initially has 4 chickens
    - She increases the number of chickens by a ratio r
    - One chicken is eaten by a neighbor's dog
    - Wendi finds and brings home 6 more chickens
    - The final number of chickens is 13
    Prove that the ratio r is equal to 2 -/
theorem chicken_ratio_problem (r : ℚ) : 
  (4 * r - 1 + 6 : ℚ) = 13 → r = 2 := by sorry

end NUMINAMATH_CALUDE_chicken_ratio_problem_l559_55968


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l559_55987

/-- The equation x^2 - xy - 6y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : ∃ (m₁ m₂ : ℝ),
  ∀ (x y : ℝ), x^2 - x*y - 6*y^2 = 0 ↔ (x = m₁*y ∨ x = m₂*y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l559_55987


namespace NUMINAMATH_CALUDE_book_page_increase_l559_55925

/-- Represents a book with chapters that increase in page count -/
structure Book where
  total_pages : ℕ
  num_chapters : ℕ
  first_chapter_pages : ℕ
  page_increase : ℕ

/-- Calculates the total pages in a book based on its structure -/
def calculate_total_pages (b : Book) : ℕ :=
  b.first_chapter_pages * b.num_chapters + 
  (b.num_chapters * (b.num_chapters - 1) * b.page_increase) / 2

/-- Theorem stating the page increase for the given book specifications -/
theorem book_page_increase (b : Book) 
  (h1 : b.total_pages = 95)
  (h2 : b.num_chapters = 5)
  (h3 : b.first_chapter_pages = 13)
  (h4 : calculate_total_pages b = b.total_pages) :
  b.page_increase = 3 := by
  sorry

#eval calculate_total_pages { total_pages := 95, num_chapters := 5, first_chapter_pages := 13, page_increase := 3 }

end NUMINAMATH_CALUDE_book_page_increase_l559_55925


namespace NUMINAMATH_CALUDE_monotonic_f_iff_a_range_l559_55997

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x + 1

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem monotonic_f_iff_a_range :
  ∀ a : ℝ, (monotonically_increasing (f a)) ↔ a ≤ -1/3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_iff_a_range_l559_55997


namespace NUMINAMATH_CALUDE_max_discount_rate_l559_55929

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate
  (cost_price : ℝ)
  (original_price : ℝ)
  (min_profit_margin : ℝ)
  (h1 : cost_price = 4)
  (h2 : original_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l559_55929


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l559_55923

/-- A rhombus with side length 65 and shorter diagonal 56 has a longer diagonal of length 118 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 65 → shorter_diagonal = 56 → longer_diagonal = 118 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l559_55923


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_three_l559_55900

def A : Set ℝ := {x | 3 + 2*x - x^2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem intersection_nonempty_implies_a_less_than_three (a : ℝ) :
  (A ∩ B a).Nonempty → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_three_l559_55900


namespace NUMINAMATH_CALUDE_highest_score_in_test_l559_55992

/-- Given a math test with scores, prove the highest score -/
theorem highest_score_in_test (mark_score least_score highest_score : ℕ) : 
  mark_score = 2 * least_score →
  mark_score = 46 →
  highest_score - least_score = 75 →
  highest_score = 98 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_in_test_l559_55992


namespace NUMINAMATH_CALUDE_intersection_set_equality_l559_55915

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5 ∧ 0 < α ∧ α < π}
  S = {3 * π / 10, 4 * π / 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_equality_l559_55915


namespace NUMINAMATH_CALUDE_congruence_solution_l559_55905

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 15 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l559_55905


namespace NUMINAMATH_CALUDE_function_zero_implies_a_bound_l559_55981

/-- If the function f(x) = e^x - 2x + a has a zero, then a ≤ 2ln2 - 2 -/
theorem function_zero_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, Real.exp x - 2 * x + a = 0) → a ≤ 2 * Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_implies_a_bound_l559_55981


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l559_55937

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l559_55937


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l559_55994

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  17 ∣ n ∧
  100 ≤ n ∧ 
  n ≤ 999 ∧
  ∀ m : ℕ, (17 ∣ m ∧ 100 ≤ m ∧ m ≤ 999) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l559_55994


namespace NUMINAMATH_CALUDE_ordering_of_a_ab_ab_squared_l559_55946

theorem ordering_of_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : b < -1) :
  a * b > a ∧ a > a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_a_ab_ab_squared_l559_55946


namespace NUMINAMATH_CALUDE_no_perfect_power_consecutive_product_l559_55973

theorem no_perfect_power_consecutive_product : 
  ∀ n : ℕ, ¬∃ (a k : ℕ), k > 1 ∧ n * (n + 1) = a ^ k :=
sorry

end NUMINAMATH_CALUDE_no_perfect_power_consecutive_product_l559_55973


namespace NUMINAMATH_CALUDE_min_distance_to_line_l559_55998

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ p ∈ line, Real.sqrt (p.1^2 + p.2^2) ≥ d ∧
    ∃ q ∈ line, Real.sqrt (q.1^2 + q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l559_55998


namespace NUMINAMATH_CALUDE_inequality_solution_l559_55982

theorem inequality_solution : 
  {x : ℕ | 2 * x - 1 < 5} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l559_55982


namespace NUMINAMATH_CALUDE_store_revenue_l559_55949

def shirt_price : ℚ := 10
def jean_price : ℚ := 2 * shirt_price
def jacket_price : ℚ := 3 * jean_price
def sock_price : ℚ := 2

def shirt_quantity : ℕ := 20
def jean_quantity : ℕ := 10
def jacket_quantity : ℕ := 15
def sock_quantity : ℕ := 30

def jacket_discount : ℚ := 0.1
def sock_bulk_discount : ℚ := 0.2

def shirt_revenue : ℚ := (shirt_quantity / 2 : ℚ) * shirt_price
def jean_revenue : ℚ := (jean_quantity : ℚ) * jean_price
def jacket_revenue : ℚ := (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount)
def sock_revenue : ℚ := (sock_quantity : ℚ) * sock_price * (1 - sock_bulk_discount)

def total_revenue : ℚ := shirt_revenue + jean_revenue + jacket_revenue + sock_revenue

theorem store_revenue : total_revenue = 1158 := by sorry

end NUMINAMATH_CALUDE_store_revenue_l559_55949


namespace NUMINAMATH_CALUDE_area_of_specific_rectangle_l559_55971

/-- A rectangle with a diagonal divided into four equal segments -/
structure DividedRectangle where
  /-- The length of each segment of the diagonal -/
  segment_length : ℝ
  /-- The diagonal is divided into four equal segments -/
  diagonal_length : ℝ := 4 * segment_length
  /-- The parallel lines are perpendicular to the diagonal -/
  perpendicular_lines : Bool

/-- The area of a rectangle with a divided diagonal -/
def area (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific rectangle is 16√3 -/
theorem area_of_specific_rectangle :
  let rect : DividedRectangle := {
    segment_length := 2,
    perpendicular_lines := true
  }
  area rect = 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_rectangle_l559_55971


namespace NUMINAMATH_CALUDE_phone_problem_solution_l559_55957

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchase_price : ℝ
  selling_price : ℝ

/-- The problem setup -/
def phone_problem : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x y : ℕ),
    (x * a.purchase_price + y * b.purchase_price = 32000) ∧
    (x * (a.selling_price - a.purchase_price) + y * (b.selling_price - b.purchase_price) = 4400) ∧
    x = 6 ∧ y = 4

/-- The profit maximization problem -/
def profit_maximization : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x : ℕ),
    x ≥ 10 ∧
    (30 - x) ≤ 2 * x ∧
    400 * x + 500 * (30 - x) = 14000 ∧
    ∀ (y : ℕ), y ≥ 10 → (30 - y) ≤ 2 * y → 400 * y + 500 * (30 - y) ≤ 14000

theorem phone_problem_solution : 
  phone_problem ∧ profit_maximization :=
sorry

end NUMINAMATH_CALUDE_phone_problem_solution_l559_55957


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_three_l559_55967

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_three_l559_55967


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l559_55909

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) :
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l559_55909


namespace NUMINAMATH_CALUDE_last_two_digits_product_l559_55983

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →
  (n % 8 = 0) → 
  ((n % 100) / 10 + n % 10 = 12) → 
  ((n % 100) / 10) * (n % 10) = 32 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l559_55983


namespace NUMINAMATH_CALUDE_percentage_problem_l559_55947

theorem percentage_problem (P : ℝ) : 
  (0.1 * 0.3 * (P / 100) * 6000 = 90) → P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l559_55947


namespace NUMINAMATH_CALUDE_cars_triangle_right_angle_l559_55935

/-- Represents a car traveling on a triangular path -/
structure Car where
  speedAB : ℝ
  speedBC : ℝ
  speedCA : ℝ

/-- Represents a triangle with three cars traveling on its sides -/
structure TriangleWithCars where
  -- Lengths of the sides of the triangle
  ab : ℝ
  bc : ℝ
  ca : ℝ
  -- The three cars
  car1 : Car
  car2 : Car
  car3 : Car

/-- The theorem stating that if three cars travel on a triangle and return at the same time, 
    the angle ABC is 90 degrees -/
theorem cars_triangle_right_angle (t : TriangleWithCars) : 
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car2.speedAB + t.bc / t.car2.speedBC + t.ca / t.car2.speedCA) ∧
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car3.speedAB + t.bc / t.car3.speedBC + t.ca / t.car3.speedCA) ∧
  (t.car1.speedAB = 12) ∧ (t.car1.speedBC = 10) ∧ (t.car1.speedCA = 15) ∧
  (t.car2.speedAB = 15) ∧ (t.car2.speedBC = 15) ∧ (t.car2.speedCA = 10) ∧
  (t.car3.speedAB = 10) ∧ (t.car3.speedBC = 20) ∧ (t.car3.speedCA = 12) →
  ∃ (A B C : ℝ × ℝ), 
    let angleABC := Real.arccos ((t.ab^2 + t.bc^2 - t.ca^2) / (2 * t.ab * t.bc))
    angleABC = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_triangle_right_angle_l559_55935


namespace NUMINAMATH_CALUDE_tetrahedron_regularity_l559_55926

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)

-- Define properties of the tetrahedron
def has_inscribed_sphere (t : Tetrahedron) : Prop := sorry

def sphere_touches_incenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_orthocenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_centroid (t : Tetrahedron) : Prop := sorry

def is_regular (t : Tetrahedron) : Prop := sorry

-- Theorem statement
theorem tetrahedron_regularity (t : Tetrahedron) :
  has_inscribed_sphere t ∧
  sphere_touches_incenter t ∧
  sphere_touches_orthocenter t ∧
  sphere_touches_centroid t →
  is_regular t :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_regularity_l559_55926


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l559_55991

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
def draw (state : UrnState) : UrnState :=
  if state.red / (state.red + state.blue) > state.blue / (state.red + state.blue)
  then UrnState.mk (state.red + 1) state.blue
  else UrnState.mk state.red (state.blue + 1)

/-- Performs n draw operations -/
def performDraws (n : ℕ) (initial : UrnState) : UrnState :=
  match n with
  | 0 => initial
  | n + 1 => draw (performDraws n initial)

/-- Calculates the probability of selecting a red ball n times in a row -/
def redProbability (n : ℕ) (initial : UrnState) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 
    let state := performDraws n initial
    (state.red : ℚ) / (state.red + state.blue) * redProbability n initial

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial := UrnState.mk 2 1
  let final := performDraws 5 initial
  final.red = 8 ∧ final.blue = 4 ∧ redProbability 5 initial = 2/7 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l559_55991


namespace NUMINAMATH_CALUDE_mindmaster_secret_codes_l559_55977

/-- The number of colors available for the pegs. -/
def num_colors : ℕ := 7

/-- The number of slots in each code. -/
def code_length : ℕ := 5

/-- The total number of possible codes without restrictions. -/
def total_codes : ℕ := num_colors ^ code_length

/-- The number of colors excluding red. -/
def non_red_colors : ℕ := num_colors - 1

/-- The number of codes without any red pegs. -/
def codes_without_red : ℕ := non_red_colors ^ code_length

/-- The number of valid secret codes in Mindmaster. -/
def valid_secret_codes : ℕ := total_codes - codes_without_red

theorem mindmaster_secret_codes : valid_secret_codes = 9031 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_secret_codes_l559_55977


namespace NUMINAMATH_CALUDE_weight_comparison_l559_55969

/-- Given the weights of Mildred, Carol, and Tom, prove statements about their combined weights -/
theorem weight_comparison (mildred_weight carol_weight tom_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = 9)
  (h3 : tom_weight = 20) :
  let combined_weight := carol_weight + tom_weight
  (combined_weight = 29) ∧ 
  (mildred_weight = combined_weight + 30) := by
  sorry

end NUMINAMATH_CALUDE_weight_comparison_l559_55969


namespace NUMINAMATH_CALUDE_store_purchase_divisibility_l559_55944

theorem store_purchase_divisibility (m n k : ℕ) :
  ∃ p : ℕ, 3 * m + 4 * n + 5 * k = 11 * p →
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_divisibility_l559_55944


namespace NUMINAMATH_CALUDE_max_value_interval_l559_55993

open Real

noncomputable def f (a x : ℝ) : ℝ := 3 * log x - x^2 + (a - 1/2) * x

theorem max_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 3, ∀ y ∈ Set.Ioo 1 3, f a x ≥ f a y) ↔ a ∈ Set.Ioo (-1/2) (11/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_interval_l559_55993


namespace NUMINAMATH_CALUDE_triangle_relation_l559_55930

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- State the theorem
theorem triangle_relation (abc : Triangle) (a'b'c' : Triangle) 
  (h1 : abc.angleB = a'b'c'.angleB) 
  (h2 : abc.angleA + a'b'c'.angleA = π) : 
  abc.a * a'b'c'.a = abc.b * a'b'c'.b + abc.c * a'b'c'.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_relation_l559_55930


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l559_55986

/-- A symmetric trapezoid EFGH with given properties -/
structure SymmetricTrapezoid :=
  (EF : ℝ)  -- Length of top base EF
  (GH : ℝ)  -- Length of bottom base GH
  (height : ℝ)  -- Height from EF to GH
  (isSymmetric : Bool)  -- Is the trapezoid symmetric?
  (EFGHEqual : Bool)  -- Are EF and GH equal in length?

/-- Properties of the specific trapezoid in the problem -/
def problemTrapezoid : SymmetricTrapezoid :=
  { EF := 10,
    GH := 22,
    height := 6,
    isSymmetric := true,
    EFGHEqual := true }

/-- Theorem stating the perimeter of the trapezoid -/
theorem trapezoid_perimeter (t : SymmetricTrapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = t.EF + 12)
  (h3 : t.height = 6)
  (h4 : t.isSymmetric = true)
  (h5 : t.EFGHEqual = true) :
  (t.EF + t.GH + 2 * Real.sqrt (t.height^2 + ((t.GH - t.EF) / 2)^2)) = 32 + 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l559_55986
