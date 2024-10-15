import Mathlib

namespace NUMINAMATH_CALUDE_student_B_visited_C_l935_93551

structure Student :=
  (name : String)
  (visited : Finset String)

def University : Type := String

theorem student_B_visited_C (studentA studentB studentC : Student) 
  (univA univB univC : University) :
  studentA.name = "A" →
  studentB.name = "B" →
  studentC.name = "C" →
  univA = "A" →
  univB = "B" →
  univC = "C" →
  studentA.visited.card > studentB.visited.card →
  univA ∉ studentA.visited →
  univB ∉ studentB.visited →
  ∃ (u : University), u ∈ studentA.visited ∧ u ∈ studentB.visited ∧ u ∈ studentC.visited →
  univC ∈ studentB.visited :=
by sorry

end NUMINAMATH_CALUDE_student_B_visited_C_l935_93551


namespace NUMINAMATH_CALUDE_product_increase_l935_93515

theorem product_increase (A B : ℝ) (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_l935_93515


namespace NUMINAMATH_CALUDE_largest_number_with_digit_constraints_l935_93557

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem largest_number_with_digit_constraints : 
  ∀ n : ℕ, sum_of_digits n = 13 ∧ product_of_digits n = 36 → n ≤ 3322111 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_constraints_l935_93557


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l935_93598

theorem fraction_irreducibility (n : ℕ) : 
  Irreducible ((2 * n^2 + 11 * n - 18) / (n + 7)) ↔ n % 3 = 0 ∨ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l935_93598


namespace NUMINAMATH_CALUDE_sum_and_equality_implies_b_value_l935_93514

theorem sum_and_equality_implies_b_value
  (a b c : ℝ)
  (sum_eq : a + b + c = 117)
  (equality : a + 8 = b - 10 ∧ b - 10 = 4 * c) :
  b = 550 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equality_implies_b_value_l935_93514


namespace NUMINAMATH_CALUDE_quidditch_tournament_equal_wins_l935_93592

/-- Represents a team in the Quidditch tournament -/
structure Team :=
  (id : Nat)

/-- Represents the tournament setup -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (wins : Team → Nat)
  (h_num_teams : teams.card = num_teams)
  (h_wins_bound : ∀ t ∈ teams, wins t < num_teams)
  (h_total_wins : (teams.sum wins) = num_teams * (num_teams - 1) / 2)

/-- Main theorem statement -/
theorem quidditch_tournament_equal_wins (tournament : Tournament) 
  (h_eight_teams : tournament.num_teams = 8) :
  ∃ (A B C D : Team), A ∈ tournament.teams ∧ B ∈ tournament.teams ∧ 
    C ∈ tournament.teams ∧ D ∈ tournament.teams ∧ A ≠ B ∧ C ≠ D ∧ 
    tournament.wins A + tournament.wins B = tournament.wins C + tournament.wins D :=
by sorry

end NUMINAMATH_CALUDE_quidditch_tournament_equal_wins_l935_93592


namespace NUMINAMATH_CALUDE_inequality_proof_l935_93581

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c ≤ 3) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l935_93581


namespace NUMINAMATH_CALUDE_max_victory_margin_l935_93550

/-- Represents the vote count for a candidate in two time periods -/
structure VoteCount where
  first_period : ℕ
  second_period : ℕ

/-- The election scenario with given conditions -/
def ElectionScenario : Prop :=
  ∃ (petya vasya : VoteCount),
    -- Total votes condition
    petya.first_period + petya.second_period + vasya.first_period + vasya.second_period = 27 ∧
    -- First two hours condition
    petya.first_period = vasya.first_period + 9 ∧
    -- Last hour condition
    vasya.second_period = petya.second_period + 9 ∧
    -- Petya wins condition
    petya.first_period + petya.second_period > vasya.first_period + vasya.second_period

/-- The theorem stating the maximum possible margin of Petya's victory -/
theorem max_victory_margin (h : ElectionScenario) :
  ∃ (petya vasya : VoteCount),
    petya.first_period + petya.second_period - (vasya.first_period + vasya.second_period) ≤ 9 :=
  sorry

end NUMINAMATH_CALUDE_max_victory_margin_l935_93550


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l935_93558

theorem unique_solution_cubic_equation :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 41 ↔ x = 5 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l935_93558


namespace NUMINAMATH_CALUDE_students_in_one_subject_is_32_l935_93521

/-- Represents the number of students in each class and their intersections -/
structure ClassEnrollment where
  calligraphy : ℕ
  art : ℕ
  instrumental : ℕ
  calligraphy_art : ℕ
  calligraphy_instrumental : ℕ
  art_instrumental : ℕ
  all_three : ℕ

/-- Calculates the number of students enrolled in only one subject -/
def studentsInOneSubject (e : ClassEnrollment) : ℕ :=
  e.calligraphy + e.art + e.instrumental - 2 * (e.calligraphy_art + e.calligraphy_instrumental + e.art_instrumental) + 3 * e.all_three

/-- The main theorem stating that given the enrollment conditions, 32 students are in only one subject -/
theorem students_in_one_subject_is_32 (e : ClassEnrollment)
  (h1 : e.calligraphy = 29)
  (h2 : e.art = 28)
  (h3 : e.instrumental = 27)
  (h4 : e.calligraphy_art = 13)
  (h5 : e.calligraphy_instrumental = 12)
  (h6 : e.art_instrumental = 11)
  (h7 : e.all_three = 5) :
  studentsInOneSubject e = 32 := by
  sorry


end NUMINAMATH_CALUDE_students_in_one_subject_is_32_l935_93521


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l935_93562

theorem min_value_of_reciprocal_sum (p q r : ℝ) (a b : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r →
  p < q ∧ q < r →
  p^3 - a*p^2 + b*p - 48 = 0 →
  q^3 - a*q^2 + b*q - 48 = 0 →
  r^3 - a*r^2 + b*r - 48 = 0 →
  1/p + 2/q + 3/r ≥ 3/2 ∧ ∃ p' q' r' a' b', 
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧
    p' < q' ∧ q' < r' ∧
    p'^3 - a'*p'^2 + b'*p' - 48 = 0 ∧
    q'^3 - a'*q'^2 + b'*q' - 48 = 0 ∧
    r'^3 - a'*r'^2 + b'*r' - 48 = 0 ∧
    1/p' + 2/q' + 3/r' = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l935_93562


namespace NUMINAMATH_CALUDE_profit_difference_l935_93582

/-- The profit difference between selling a certain house and a standard house -/
theorem profit_difference (C : ℝ) : 
  let certain_house_cost : ℝ := C + 100000
  let standard_house_price : ℝ := 320000
  let certain_house_price : ℝ := 1.5 * standard_house_price
  let certain_house_profit : ℝ := certain_house_price - certain_house_cost
  let standard_house_profit : ℝ := standard_house_price - C
  certain_house_profit - standard_house_profit = 60000 := by
  sorry

#check profit_difference

end NUMINAMATH_CALUDE_profit_difference_l935_93582


namespace NUMINAMATH_CALUDE_equation_solution_l935_93559

theorem equation_solution (x : Real) :
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔
  (∃ k : ℤ, x = 2 * π / 3 + 2 * k * π ∨ x = 7 * π / 6 + 2 * k * π ∨ x = -π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l935_93559


namespace NUMINAMATH_CALUDE_josiah_saved_24_days_l935_93544

/-- The number of days Josiah saved -/
def josiah_days : ℕ := sorry

/-- Josiah's daily savings in dollars -/
def josiah_daily_savings : ℚ := 1/4

/-- Leah's daily savings in dollars -/
def leah_daily_savings : ℚ := 1/2

/-- Number of days Leah saved -/
def leah_days : ℕ := 20

/-- Number of days Megan saved -/
def megan_days : ℕ := 12

/-- Total amount saved by all three children in dollars -/
def total_savings : ℚ := 28

theorem josiah_saved_24_days :
  josiah_days = 24 ∧
  josiah_daily_savings * josiah_days + 
  leah_daily_savings * leah_days + 
  (2 * leah_daily_savings) * megan_days = total_savings := by sorry

end NUMINAMATH_CALUDE_josiah_saved_24_days_l935_93544


namespace NUMINAMATH_CALUDE_set_operations_l935_93512

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Define the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 5}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 6}) ∧
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x ≤ 6}) ∧
  ((A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l935_93512


namespace NUMINAMATH_CALUDE_special_tetrahedron_equal_angle_l935_93564

/-- A tetrahedron with specific dihedral angle properties -/
structure SpecialTetrahedron where
  /-- The tetrahedron has three dihedral angles of 90° that do not belong to the same vertex -/
  three_right_angles : ℕ
  /-- All other dihedral angles are equal -/
  equal_other_angles : ℝ
  /-- The number of 90° angles is exactly 3 -/
  right_angle_count : three_right_angles = 3

/-- The theorem stating the value of the equal dihedral angles in the special tetrahedron -/
theorem special_tetrahedron_equal_angle (t : SpecialTetrahedron) :
  t.equal_other_angles = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_special_tetrahedron_equal_angle_l935_93564


namespace NUMINAMATH_CALUDE_quadratic_solution_l935_93509

theorem quadratic_solution (h : 81 * (4/9)^2 - 145 * (4/9) + 64 = 0) :
  81 * (-16/9)^2 - 145 * (-16/9) + 64 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l935_93509


namespace NUMINAMATH_CALUDE_snake_paint_calculation_l935_93586

theorem snake_paint_calculation (cube_paint : ℕ) (snake_length : ℕ) (segment_length : ℕ) 
  (segment_paint : ℕ) (end_paint : ℕ) : 
  cube_paint = 60 → 
  snake_length = 2016 → 
  segment_length = 6 → 
  segment_paint = 240 → 
  end_paint = 20 → 
  (snake_length / segment_length * segment_paint + end_paint : ℕ) = 80660 := by
  sorry

end NUMINAMATH_CALUDE_snake_paint_calculation_l935_93586


namespace NUMINAMATH_CALUDE_p_h_neg_three_equals_eight_l935_93525

-- Define the function h
def h (x : ℝ) : ℝ := 2 * x^2 - 10

-- Define the theorem
theorem p_h_neg_three_equals_eight 
  (p : ℝ → ℝ) -- p is a function from reals to reals
  (h_def : ∀ x, h x = 2 * x^2 - 10) -- definition of h
  (p_h_three : p (h 3) = 8) -- given condition
  : p (h (-3)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_p_h_neg_three_equals_eight_l935_93525


namespace NUMINAMATH_CALUDE_total_age_problem_l935_93517

theorem total_age_problem (a b c : ℕ) : 
  b = 4 → a = b + 2 → b = 2 * c → a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_age_problem_l935_93517


namespace NUMINAMATH_CALUDE_positive_sum_greater_than_abs_diff_l935_93530

theorem positive_sum_greater_than_abs_diff (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end NUMINAMATH_CALUDE_positive_sum_greater_than_abs_diff_l935_93530


namespace NUMINAMATH_CALUDE_cubic_function_properties_l935_93553

-- Define the function f(x) = ax³ + bx²
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  (a = -6 ∧ b = 9) ∧
  (∀ x : ℝ, f (-6) 9 x ≥ f (-6) 9 0) ∧
  f (-6) 9 0 = 0 := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_cubic_function_properties_l935_93553


namespace NUMINAMATH_CALUDE_triangle_area_sine_relation_l935_93567

theorem triangle_area_sine_relation (a b c : ℝ) (A : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a^2 - b^2 - c^2 + 2*b*c = (1/2) * b * c * Real.sin A) →
  Real.sin A = 8/17 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sine_relation_l935_93567


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l935_93572

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, m * x^2 + x + m > 0) → m > (1/4 : ℝ) ∧
  ¬(m > (1/4 : ℝ) → ∀ x : ℝ, m * x^2 + x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l935_93572


namespace NUMINAMATH_CALUDE_parallel_transitive_l935_93585

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_l935_93585


namespace NUMINAMATH_CALUDE_equal_numbers_sum_of_squares_l935_93590

theorem equal_numbers_sum_of_squares (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 26 →
  c = 22 →
  d = e →
  d^2 + e^2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_sum_of_squares_l935_93590


namespace NUMINAMATH_CALUDE_dispatch_methods_count_l935_93534

def num_male_servants : ℕ := 5
def num_female_servants : ℕ := 4
def num_total_servants : ℕ := num_male_servants + num_female_servants
def num_selected : ℕ := 3
def num_areas : ℕ := 3

theorem dispatch_methods_count :
  (Nat.choose num_total_servants num_selected - 
   Nat.choose num_male_servants num_selected - 
   Nat.choose num_female_servants num_selected) * 
  (Nat.factorial num_selected) = 420 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_methods_count_l935_93534


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l935_93543

/-- Proves that the mean profit for the first 15 days is 285 Rs, given the conditions of the problem. -/
theorem shop_profit_calculation (total_days : ℕ) (mean_profit : ℚ) (last_half_mean : ℚ) :
  total_days = 30 →
  mean_profit = 350 →
  last_half_mean = 415 →
  (total_days * mean_profit - (total_days / 2) * last_half_mean) / (total_days / 2) = 285 := by
  sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l935_93543


namespace NUMINAMATH_CALUDE_shower_has_three_walls_l935_93574

/-- Represents the properties of a shower with tiled walls -/
structure Shower :=
  (width_tiles : ℕ)
  (height_tiles : ℕ)
  (total_tiles : ℕ)

/-- Calculates the number of walls in a shower -/
def number_of_walls (s : Shower) : ℚ :=
  s.total_tiles / (s.width_tiles * s.height_tiles)

/-- Theorem: The shower has 3 walls -/
theorem shower_has_three_walls (s : Shower) 
  (h1 : s.width_tiles = 8)
  (h2 : s.height_tiles = 20)
  (h3 : s.total_tiles = 480) : 
  number_of_walls s = 3 := by
  sorry

#eval number_of_walls { width_tiles := 8, height_tiles := 20, total_tiles := 480 }

end NUMINAMATH_CALUDE_shower_has_three_walls_l935_93574


namespace NUMINAMATH_CALUDE_alarm_system_probability_l935_93533

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one_alerts := 1 - (1 - p) * (1 - p)
  prob_at_least_one_alerts = 0.64 := by
sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l935_93533


namespace NUMINAMATH_CALUDE_betty_orange_boxes_l935_93501

/-- The minimum number of boxes needed to store oranges given specific conditions -/
def min_boxes (total_oranges : ℕ) (first_box : ℕ) (second_box : ℕ) (max_per_box : ℕ) : ℕ :=
  2 + (total_oranges - first_box - second_box + max_per_box - 1) / max_per_box

/-- Proof that Betty needs 5 boxes to store her oranges -/
theorem betty_orange_boxes : 
  min_boxes 120 30 25 30 = 5 :=
by sorry

end NUMINAMATH_CALUDE_betty_orange_boxes_l935_93501


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_4_l935_93597

theorem nearest_integer_to_3_plus_sqrt2_to_4 :
  ∃ (n : ℤ), n = 386 ∧ ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 2)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^4 - m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_4_l935_93597


namespace NUMINAMATH_CALUDE_problem_statement_l935_93507

theorem problem_statement (a b c d : ℤ) (x : ℝ) : 
  x = (a + b * Real.sqrt c) / d →
  (7 * x / 8) + 2 = 4 / x →
  (a * c * d) / b = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l935_93507


namespace NUMINAMATH_CALUDE_square_circle_difference_l935_93568

-- Define the square and circle
def square_diagonal : ℝ := 8
def circle_diameter : ℝ := 8

-- Theorem statement
theorem square_circle_difference :
  let square_side := (square_diagonal ^ 2 / 2).sqrt
  let square_area := square_side ^ 2
  let square_perimeter := 4 * square_side
  let circle_radius := circle_diameter / 2
  let circle_area := π * circle_radius ^ 2
  let circle_perimeter := 2 * π * circle_radius
  (circle_area - square_area = 16 * π - 32) ∧
  (circle_perimeter - square_perimeter = 8 * π - 16 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_square_circle_difference_l935_93568


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_games_l935_93596

theorem ice_cream_sales_for_games (game_cost : ℕ) (ice_cream_price : ℕ) : 
  game_cost = 60 → ice_cream_price = 5 → (2 * game_cost) / ice_cream_price = 24 := by
  sorry

#check ice_cream_sales_for_games

end NUMINAMATH_CALUDE_ice_cream_sales_for_games_l935_93596


namespace NUMINAMATH_CALUDE_prob_both_white_l935_93540

/-- Represents an urn with white and black balls -/
structure Urn :=
  (white : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.white + u.black)

/-- The first urn -/
def urn1 : Urn := ⟨2, 10⟩

/-- The second urn -/
def urn2 : Urn := ⟨8, 4⟩

/-- Theorem: The probability of drawing white balls from both urns is 1/9 -/
theorem prob_both_white : prob_white urn1 * prob_white urn2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_l935_93540


namespace NUMINAMATH_CALUDE_f_of_g_composition_l935_93552

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 1

theorem f_of_g_composition : f (1 + g 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_composition_l935_93552


namespace NUMINAMATH_CALUDE_green_blue_difference_after_border_l935_93536

/-- Represents a hexagonal figure with tiles --/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a single border layer of a hexagon --/
def single_border_tiles : ℕ := 6 * 3

/-- Calculates the number of tiles needed for a double border layer of a hexagon --/
def double_border_tiles : ℕ := single_border_tiles + 6 * 4

/-- Adds a double border of green tiles to a hexagonal figure --/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + double_border_tiles }

/-- The main theorem to prove --/
theorem green_blue_difference_after_border (initial_figure : HexagonalFigure)
    (h1 : initial_figure.blue_tiles = 20)
    (h2 : initial_figure.green_tiles = 10) :
    let new_figure := add_double_border initial_figure
    new_figure.green_tiles - new_figure.blue_tiles = 32 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_border_l935_93536


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_pair_value_l935_93561

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up all people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any woman-woman pairs -/
def pairings_without_woman_pairs : ℕ := num_women.factorial

/-- The probability of at least one woman-woman pair -/
def prob_at_least_one_woman_pair : ℚ :=
  (total_pairings - pairings_without_woman_pairs : ℚ) / total_pairings

theorem prob_at_least_one_woman_pair_value :
  prob_at_least_one_woman_pair = (10395 - 720) / 10395 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_pair_value_l935_93561


namespace NUMINAMATH_CALUDE_sodium_hypochlorite_weight_approx_l935_93506

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of sodium hypochlorite (NaOCl) in g/mol -/
def sodium_hypochlorite_weight : ℝ := sodium_weight + oxygen_weight + chlorine_weight

/-- The given molecular weight of a certain substance -/
def given_weight : ℝ := 74

/-- Theorem stating that the molecular weight of sodium hypochlorite is approximately equal to the given weight -/
theorem sodium_hypochlorite_weight_approx : 
  ∃ ε > 0, |sodium_hypochlorite_weight - given_weight| < ε :=
sorry

end NUMINAMATH_CALUDE_sodium_hypochlorite_weight_approx_l935_93506


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l935_93502

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l935_93502


namespace NUMINAMATH_CALUDE_new_average_weight_l935_93576

/-- Given a bowling team with the following properties:
  * The original team has 7 players
  * The original average weight is 103 kg
  * Two new players join the team
  * One new player weighs 110 kg
  * The other new player weighs 60 kg
  
  Prove that the new average weight of the team is 99 kg -/
theorem new_average_weight 
  (original_players : Nat) 
  (original_avg_weight : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 103)
  (h3 : new_player1_weight = 110)
  (h4 : new_player2_weight = 60) :
  let total_weight := original_players * original_avg_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  total_weight / new_total_players = 99 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l935_93576


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l935_93589

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l935_93589


namespace NUMINAMATH_CALUDE_cloak_purchase_change_l935_93579

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakPurchase where
  silver_paid : ℕ
  gold_change : ℕ

/-- Proves that buying an invisibility cloak for 14 gold coins results in a change of 10 silver coins --/
theorem cloak_purchase_change 
  (purchase1 : CloakPurchase)
  (purchase2 : CloakPurchase)
  (h1 : purchase1.silver_paid = 20 ∧ purchase1.gold_change = 4)
  (h2 : purchase2.silver_paid = 15 ∧ purchase2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14)
  : ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_purchase_change_l935_93579


namespace NUMINAMATH_CALUDE_total_covid_cases_l935_93577

/-- Theorem: Total COVID-19 cases in New York, California, and Texas --/
theorem total_covid_cases (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  california = texas + 400 →
  new_york + california + texas = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_covid_cases_l935_93577


namespace NUMINAMATH_CALUDE_polynomial_degree_example_l935_93500

/-- The degree of a polynomial (3x^5 + 2x^4 - x^2 + 5)(4x^{11} - 8x^8 + 3x^5 - 10) - (x^3 + 7)^6 -/
theorem polynomial_degree_example : 
  let p₁ : Polynomial ℝ := X^5 * 3 + X^4 * 2 - X^2 + 5
  let p₂ : Polynomial ℝ := X^11 * 4 - X^8 * 8 + X^5 * 3 - 10
  let p₃ : Polynomial ℝ := (X^3 + 7)^6
  (p₁ * p₂ - p₃).degree = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_example_l935_93500


namespace NUMINAMATH_CALUDE_max_value_implies_m_equals_20_l935_93595

/-- The function f(x) = -x^3 + 6x^2 - m --/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 - m

/-- The maximum value of f(x) is 12 --/
def max_value : ℝ := 12

theorem max_value_implies_m_equals_20 :
  (∃ x₀ : ℝ, ∀ x : ℝ, f x m ≤ f x₀ m) ∧ (∃ x₁ : ℝ, f x₁ m = max_value) → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_equals_20_l935_93595


namespace NUMINAMATH_CALUDE_library_books_count_l935_93587

/-- The number of bookshelves in the library -/
def num_bookshelves : ℕ := 28

/-- The number of floors in each bookshelf -/
def floors_per_bookshelf : ℕ := 6

/-- The number of books left on a floor after taking two books -/
def books_left_after_taking_two : ℕ := 20

/-- The total number of books in the library -/
def total_books : ℕ := num_bookshelves * floors_per_bookshelf * (books_left_after_taking_two + 2)

theorem library_books_count : total_books = 3696 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l935_93587


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l935_93541

theorem quadratic_root_problem (m : ℝ) : 
  ((0 : ℝ) = 0 → (m - 2) * 0^2 + 4 * 0 + 2 - |m| = 0) ∧ 
  (m - 2 ≠ 0) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l935_93541


namespace NUMINAMATH_CALUDE_rectangular_field_length_l935_93527

theorem rectangular_field_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 81 = (1 / 8) * (l * w)) :
  l = 36 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l935_93527


namespace NUMINAMATH_CALUDE_business_investment_problem_l935_93571

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Proves that given the conditions of the business problem, partner a's investment is 16000 -/
theorem business_investment_problem 
  (a b c : Partner)
  (h1 : b.profitShare = 1800)
  (h2 : a.profitShare - c.profitShare = 720)
  (h3 : b.investment = 10000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare * b.investment = b.profitShare * a.investment)
  (h6 : b.profitShare * c.investment = c.profitShare * b.investment)
  (h7 : a.profitShare * c.investment = c.profitShare * a.investment) :
  a.investment = 16000 := by
  sorry


end NUMINAMATH_CALUDE_business_investment_problem_l935_93571


namespace NUMINAMATH_CALUDE_work_completion_time_l935_93519

/-- The time (in days) it takes for A to complete the work alone -/
def a_time : ℝ := 30

/-- The time (in days) it takes for A and B to complete the work together -/
def ab_time : ℝ := 19.411764705882355

/-- The time (in days) it takes for B to complete the work alone -/
def b_time : ℝ := 55

/-- Theorem stating that if A can do the work in 30 days, and A and B together can do the work in 19.411764705882355 days, then B can do the work alone in 55 days -/
theorem work_completion_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧ 
  (a_time > 0) ∧ (b_time > 0) ∧ (ab_time > 0) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l935_93519


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l935_93508

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l935_93508


namespace NUMINAMATH_CALUDE_cauchy_schwarz_2d_l935_93599

theorem cauchy_schwarz_2d (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_2d_l935_93599


namespace NUMINAMATH_CALUDE_angle_equality_l935_93503

theorem angle_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = π/3 ∧ β = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l935_93503


namespace NUMINAMATH_CALUDE_g_of_3_equals_64_l935_93578

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all nonzero x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2

/-- Given g satisfying the property, prove that g(3) = 64 -/
theorem g_of_3_equals_64 (g : ℝ → ℝ) (h : g_property g) : g 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_64_l935_93578


namespace NUMINAMATH_CALUDE_min_draws_for_twelve_balls_l935_93591

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  black : Nat

/-- Represents the minimum number of balls needed to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating the minimum number of draws required -/
theorem min_draws_for_twelve_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 22)
  (h_yellow : counts.yellow = 18)
  (h_blue : counts.blue = 15)
  (h_black : counts.black = 10) :
  minDrawsForColor counts 12 = 55 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_twelve_balls_l935_93591


namespace NUMINAMATH_CALUDE_total_flowers_and_sticks_l935_93563

/-- The number of pots -/
def num_pots : ℕ := 466

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 53

/-- The number of sticks in each pot -/
def sticks_per_pot : ℕ := 181

/-- The total number of flowers and sticks in all pots -/
def total_items : ℕ := num_pots * flowers_per_pot + num_pots * sticks_per_pot

theorem total_flowers_and_sticks : total_items = 109044 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_and_sticks_l935_93563


namespace NUMINAMATH_CALUDE_fraction_inequality_l935_93539

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / c > a / d := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l935_93539


namespace NUMINAMATH_CALUDE_scoops_left_is_16_l935_93556

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of people (Lucas, Danny, Connor) who want 2 scoops of chocolate each -/
def chocolate_lovers : ℕ := 3

/-- Represents the number of scoops each chocolate lover wants -/
def scoops_per_chocolate_lover : ℕ := 2

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents how many times more scoops Shannon wants compared to Olivia -/
def shannons_multiplier : ℕ := 2

/-- Theorem stating that the number of scoops left is 16 -/
theorem scoops_left_is_16 : 
  marys_cartons * scoops_per_carton - 
  (ethans_scoops + 
   chocolate_lovers * scoops_per_chocolate_lover + 
   olivias_scoops + 
   shannons_multiplier * olivias_scoops) = 16 := by
  sorry

end NUMINAMATH_CALUDE_scoops_left_is_16_l935_93556


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l935_93523

theorem cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20)
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) :
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l935_93523


namespace NUMINAMATH_CALUDE_square_difference_ratio_l935_93588

theorem square_difference_ratio : 
  (1632^2 - 1629^2) / (1635^2 - 1626^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l935_93588


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l935_93538

/-- Represents a right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- Represents a square inscribed in the triangle with one vertex at the right angle -/
def corner_square (t : RightTriangle) := 
  { x : ℝ // x > 0 ∧ x / t.a = x / t.b }

/-- Represents a square inscribed in the triangle with one side on the hypotenuse -/
def hypotenuse_square (t : RightTriangle) := 
  { y : ℝ // y > 0 ∧ y / t.a = y / t.b }

/-- The main theorem to be proved -/
theorem inscribed_squares_ratio 
  (t : RightTriangle) 
  (h : t.a = 5 ∧ t.b = 12 ∧ t.c = 13) :
  ∀ (x : corner_square t) (y : hypotenuse_square t), 
  x.val = y.val := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l935_93538


namespace NUMINAMATH_CALUDE_sum_of_coordinates_on_h_l935_93528

def g (x : ℝ) : ℝ := x + 3

def h (x : ℝ) : ℝ := (g x)^2

theorem sum_of_coordinates_on_h : ∃ (x y : ℝ), 
  (2, 5) = (2, g 2) ∧ 
  (x, y) = (2, h 2) ∧ 
  x + y = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_on_h_l935_93528


namespace NUMINAMATH_CALUDE_square_plus_abs_eq_zero_l935_93575

theorem square_plus_abs_eq_zero (x y : ℝ) :
  x^2 + |y + 8| = 0 → x = 0 ∧ y = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_eq_zero_l935_93575


namespace NUMINAMATH_CALUDE_square_side_length_l935_93526

-- Define the circumference of the largest inscribed circle
def circle_circumference : ℝ := 37.69911184307752

-- Define π as a constant (approximation)
def π : ℝ := 3.141592653589793

-- Theorem statement
theorem square_side_length (circle_circumference : ℝ) (π : ℝ) :
  let radius := circle_circumference / (2 * π)
  let diameter := 2 * radius
  diameter = 12 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l935_93526


namespace NUMINAMATH_CALUDE_trail_mix_weight_l935_93510

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08

/-- The weight of dried apricots in pounds -/
def dried_apricots : ℝ := 0.12

/-- The weight of sunflower seeds in pounds -/
def sunflower_seeds : ℝ := 0.09

/-- The weight of coconut flakes in pounds -/
def coconut_flakes : ℝ := 0.15

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins + dried_apricots + sunflower_seeds + coconut_flakes

theorem trail_mix_weight : total_weight = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l935_93510


namespace NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l935_93511

def calculate_gratuity (base_price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (gratuity_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * gratuity_rate

theorem restaurant_gratuity_calculation :
  let striploin_gratuity := calculate_gratuity 80 0.10 0.05 0.15
  let wine_gratuity := calculate_gratuity 10 0.15 0 0.20
  let dessert_gratuity := calculate_gratuity 12 0.05 0.10 0.10
  let water_gratuity := calculate_gratuity 3 0 0 0.05
  striploin_gratuity + wine_gratuity + dessert_gratuity + water_gratuity = 16.12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l935_93511


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l935_93593

def a : ℝ × ℝ := (4, 2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  let v₂ : ℝ × ℝ := (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l935_93593


namespace NUMINAMATH_CALUDE_set_operations_l935_93531

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l935_93531


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l935_93583

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 24 = 2

theorem greatest_valid_integer : 
  is_valid 194 ∧ ∀ m : ℕ, is_valid m → m ≤ 194 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l935_93583


namespace NUMINAMATH_CALUDE_book_arrangement_count_l935_93504

theorem book_arrangement_count : ℕ := by
  -- Define the total number of books
  let total_books : ℕ := 6
  -- Define the number of identical copies for each book type
  let identical_copies1 : ℕ := 3
  let identical_copies2 : ℕ := 2
  let unique_book : ℕ := 1

  -- Assert that the sum of all book types equals the total number of books
  have h_total : identical_copies1 + identical_copies2 + unique_book = total_books := by sorry

  -- Define the number of distinct arrangements
  let arrangements : ℕ := Nat.factorial total_books / (Nat.factorial identical_copies1 * Nat.factorial identical_copies2)

  -- Prove that the number of distinct arrangements is 60
  have h_result : arrangements = 60 := by sorry

  -- Return the result
  exact 60

end NUMINAMATH_CALUDE_book_arrangement_count_l935_93504


namespace NUMINAMATH_CALUDE_farm_animals_l935_93555

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 24 + 2 * (cows + chickens + ducks)) →
  (ducks = chickens / 2) →
  (cows = 12) := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l935_93555


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_min_sum_positive_reals_tight_l935_93522

theorem min_sum_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ (1 / 2 : ℝ) :=
by sorry

theorem min_sum_positive_reals_tight (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (3 * b) + b / (6 * c) + c / (9 * a) < (1 / 2 : ℝ) + ε :=
by sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_min_sum_positive_reals_tight_l935_93522


namespace NUMINAMATH_CALUDE_square_area_proof_l935_93547

theorem square_area_proof (x : ℚ) : 
  (5 * x - 22 : ℚ) = (34 - 4 * x) → 
  (5 * x - 22 : ℚ) > 0 →
  ((5 * x - 22) ^ 2 : ℚ) = 6724 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l935_93547


namespace NUMINAMATH_CALUDE_pens_probability_theorem_l935_93594

def total_pens : ℕ := 8
def red_pens : ℕ := 4
def blue_pens : ℕ := 4
def pens_to_pick : ℕ := 4

def probability_leftmost_blue_not_picked_rightmost_red_picked : ℚ :=
  4 / 49

theorem pens_probability_theorem :
  let total_arrangements := Nat.choose total_pens red_pens
  let total_pick_ways := Nat.choose total_pens pens_to_pick
  let favorable_red_arrangements := Nat.choose (total_pens - 2) (red_pens - 1)
  let favorable_pick_ways := Nat.choose (total_pens - 2) (pens_to_pick - 1)
  (favorable_red_arrangements * favorable_pick_ways : ℚ) / (total_arrangements * total_pick_ways) =
    probability_leftmost_blue_not_picked_rightmost_red_picked :=
by
  sorry

#check pens_probability_theorem

end NUMINAMATH_CALUDE_pens_probability_theorem_l935_93594


namespace NUMINAMATH_CALUDE_base_number_proof_l935_93532

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^22) (h2 : n = 21) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l935_93532


namespace NUMINAMATH_CALUDE_larger_number_problem_l935_93570

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 35) : L = 1631 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l935_93570


namespace NUMINAMATH_CALUDE_plates_in_second_purchase_is_20_l935_93569

/-- The cost of one paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of one paper cup -/
def cup_cost : ℝ := sorry

/-- The number of plates in the second purchase -/
def plates_in_second_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $7.50 -/
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

/-- The total cost of some plates and 40 cups is $1.50 -/
axiom second_purchase : plates_in_second_purchase * plate_cost + 40 * cup_cost = 1.50

theorem plates_in_second_purchase_is_20 : plates_in_second_purchase = 20 := by sorry

end NUMINAMATH_CALUDE_plates_in_second_purchase_is_20_l935_93569


namespace NUMINAMATH_CALUDE_stock_percentage_l935_93535

/-- Given the income, price per unit, and total investment of a stock,
    calculate the percentage of the stock. -/
theorem stock_percentage
  (income : ℝ)
  (price_per_unit : ℝ)
  (total_investment : ℝ)
  (h1 : income = 900)
  (h2 : price_per_unit = 102)
  (h3 : total_investment = 4590)
  : (income / total_investment) * 100 = (900 : ℝ) / 4590 * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_l935_93535


namespace NUMINAMATH_CALUDE_difference_in_base8_l935_93537

/-- Converts a base 8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base 8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 8) ((m % 8) :: acc)
    convert n []

theorem difference_in_base8 :
  let a := base8ToDecimal [1, 2, 3, 4]
  let b := base8ToDecimal [7, 6, 5]
  decimalToBase8 (a - b) = [2, 2, 5] :=
by sorry

end NUMINAMATH_CALUDE_difference_in_base8_l935_93537


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l935_93529

theorem unique_quadratic_solution (p : ℝ) : 
  (p ≠ 0 ∧ ∃! x, p * x^2 - 10 * x + 2 = 0) ↔ p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l935_93529


namespace NUMINAMATH_CALUDE_bags_sold_on_tuesday_l935_93518

theorem bags_sold_on_tuesday (total_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ) 
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : wednesday_sales = 100)
  (h4 : thursday_sales = 110)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℝ) * 0.25 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) :
  tuesday_sales = 70 := by
  sorry

end NUMINAMATH_CALUDE_bags_sold_on_tuesday_l935_93518


namespace NUMINAMATH_CALUDE_gcd_problem_l935_93524

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l935_93524


namespace NUMINAMATH_CALUDE_problem_statements_l935_93549

theorem problem_statements :
  (¬ ∀ a b c : ℝ, a > b → a * c^2 > b * c^2) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧
  (¬ ∀ a b : ℝ, |a| > b → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l935_93549


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l935_93505

/-- Calculates the total earnings for a given number of hours based on the described payment structure -/
def jasonEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 9
  let remainingHours := hours % 9
  let earningsPerCycle := (List.range 9).sum
  fullCycles * earningsPerCycle + (List.range remainingHours).sum

theorem jason_borrowed_amount :
  jasonEarnings 27 = 135 := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l935_93505


namespace NUMINAMATH_CALUDE_a_values_l935_93560

def A (a : ℝ) : Set ℝ := {0, 1, a^2 - 2*a}

theorem a_values (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l935_93560


namespace NUMINAMATH_CALUDE_alex_coin_distribution_distribution_satisfies_conditions_l935_93542

/-- The minimum number of additional coins needed -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's scenario -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

/-- Proof that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  ∀ i j, i ≠ j → i ≤ num_friends → j ≤ num_friends → 
  (i : ℕ) ≠ (j : ℕ) ∧ (i : ℕ) ≥ 1 ∧ (j : ℕ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_distribution_satisfies_conditions_l935_93542


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l935_93584

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  (num_Ba : ℝ) * atomic_weight_Ba + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

theorem compound_molecular_weight : 
  molecular_weight = 171.35 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l935_93584


namespace NUMINAMATH_CALUDE_adam_initial_money_l935_93580

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam received in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_money : ℚ := airplane_cost + change_received

theorem adam_initial_money : initial_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_initial_money_l935_93580


namespace NUMINAMATH_CALUDE_ab_value_for_given_equation_l935_93520

theorem ab_value_for_given_equation (a b : ℕ+) 
  (h : (2 * a + b) * (2 * b + a) = 4752) : 
  a * b = 520 := by
sorry

end NUMINAMATH_CALUDE_ab_value_for_given_equation_l935_93520


namespace NUMINAMATH_CALUDE_product_95_105_l935_93548

theorem product_95_105 : 95 * 105 = 9975 := by
  have h1 : 95 = 100 - 5 := by sorry
  have h2 : 105 = 100 + 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_product_95_105_l935_93548


namespace NUMINAMATH_CALUDE_power_product_equality_l935_93573

theorem power_product_equality : 3^3 * 2^2 * 7^2 * 11 = 58212 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l935_93573


namespace NUMINAMATH_CALUDE_fraction_comparison_l935_93565

theorem fraction_comparison : (200200201 : ℚ) / 200200203 > (300300301 : ℚ) / 300300304 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l935_93565


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l935_93546

-- Define the set A
def A : Set ℝ := {a | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define the set B
def B : Set ℝ := {x | ∀ a ∈ Set.Icc (-2 : ℝ) 2, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l935_93546


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l935_93566

theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I) / (1 + Complex.I) + 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l935_93566


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l935_93554

/-- A linear function passing through the first, second, and third quadrants implies positive slope and y-intercept -/
theorem linear_function_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + b) ∧ 
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + b) ∧ 
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + b)) →
  k > 0 ∧ b > 0 := by
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l935_93554


namespace NUMINAMATH_CALUDE_decreasing_direct_proportion_negative_k_l935_93513

/-- A direct proportion function y = kx where y decreases as x increases -/
structure DecreasingDirectProportion where
  k : ℝ
  decreasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → k * x₁ > k * x₂

/-- Theorem: If y = kx is a decreasing direct proportion function, then k < 0 -/
theorem decreasing_direct_proportion_negative_k (f : DecreasingDirectProportion) : f.k < 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_direct_proportion_negative_k_l935_93513


namespace NUMINAMATH_CALUDE_tv_show_main_characters_l935_93516

/-- Represents the TV show payment structure and calculates the number of main characters -/
def tv_show_characters : ℕ := by
  -- Define the number of minor characters
  let minor_characters : ℕ := 4
  -- Define the payment for each minor character
  let minor_payment : ℕ := 15000
  -- Define the total payment per episode
  let total_payment : ℕ := 285000
  -- Calculate the payment for each main character (3 times minor payment)
  let main_payment : ℕ := 3 * minor_payment
  -- Calculate the total payment for minor characters
  let minor_total : ℕ := minor_characters * minor_payment
  -- Calculate the remaining payment for main characters
  let main_total : ℕ := total_payment - minor_total
  -- Calculate the number of main characters
  exact main_total / main_payment

/-- Theorem stating that the number of main characters in the TV show is 5 -/
theorem tv_show_main_characters :
  tv_show_characters = 5 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_main_characters_l935_93516


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l935_93545

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    with one asymptote having a slope angle of π/3 and eccentricity e,
    the minimum value of (a² + e)/b is 2√6/3 -/
theorem hyperbola_min_value (a b : ℝ) (e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ e = c / a) →
  (∀ k : ℝ, k > 0 → (a^2 + e) / b ≥ 2 * Real.sqrt 6 / 3) ∧
  (∃ k : ℝ, k > 0 ∧ (a^2 + e) / b = 2 * Real.sqrt 6 / 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l935_93545
