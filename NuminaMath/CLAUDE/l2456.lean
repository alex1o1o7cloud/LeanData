import Mathlib

namespace NUMINAMATH_CALUDE_perp_line_plane_relation_l2456_245612

-- Define the concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the perpendicularity relations
def perp_to_countless_lines (L : Line) (α : Plane) : Prop := sorry
def perp_to_plane (L : Line) (α : Plane) : Prop := sorry

-- State the theorem
theorem perp_line_plane_relation (L : Line) (α : Plane) :
  (perp_to_plane L α → perp_to_countless_lines L α) ∧
  ∃ L α, perp_to_countless_lines L α ∧ ¬perp_to_plane L α :=
sorry

end NUMINAMATH_CALUDE_perp_line_plane_relation_l2456_245612


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2456_245647

/-- Represents a seating arrangement of adults and children -/
def SeatingArrangement := Fin 6 → Bool

/-- Checks if a seating arrangement is valid (no two children sit next to each other) -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 6, arrangement i → arrangement ((i + 1) % 6) → False

/-- The number of valid seating arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 72 valid seating arrangements -/
theorem seating_arrangements_count :
  num_valid_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2456_245647


namespace NUMINAMATH_CALUDE_distance_to_place_l2456_245699

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  (rowing_speed + current_velocity) * (rowing_speed - current_velocity) * round_trip_time / 
  (rowing_speed + current_velocity + rowing_speed - current_velocity) = 2.4 := by
sorry

end NUMINAMATH_CALUDE_distance_to_place_l2456_245699


namespace NUMINAMATH_CALUDE_part_one_part_two_l2456_245694

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem for part 1
theorem part_one : star 2 (-4) = -12 := by sorry

-- Theorem for part 2
theorem part_two : ∀ x : ℝ, star (x + 5) 3 = 0 ↔ x = -8 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2456_245694


namespace NUMINAMATH_CALUDE_cross_section_area_formula_l2456_245688

/-- Regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) :=
  (edge_length : a > 0)

/-- Plane passing through the midpoint of an edge and perpendicular to an adjacent edge -/
structure CrossSectionPlane (t : RegularTetrahedron a) :=
  (passes_through_midpoint : Bool)
  (perpendicular_to_adjacent : Bool)

/-- The area of the cross-section formed by the plane -/
def cross_section_area (t : RegularTetrahedron a) (p : CrossSectionPlane t) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_formula (a : ℝ) (t : RegularTetrahedron a) (p : CrossSectionPlane t) :
  p.passes_through_midpoint ∧ p.perpendicular_to_adjacent →
  cross_section_area t p = (a^2 * Real.sqrt 2) / 16 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_formula_l2456_245688


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2456_245685

/-- Ellipse C with given properties -/
structure Ellipse :=
  (center : ℝ × ℝ)
  (major_axis : ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : center = (0, 0))
  (h_major_axis : major_axis = 4)
  (h_point : point_on_ellipse = (1, Real.sqrt 3 / 2))

/-- Line with slope 1/2 passing through a point -/
structure Line (P : ℝ × ℝ) :=
  (slope : ℝ)
  (h_slope : slope = 1/2)

/-- Theorem about the ellipse C and intersecting lines -/
theorem ellipse_theorem (C : Ellipse) :
  (∃ (eq : ℝ × ℝ → Prop), ∀ (x y : ℝ), eq (x, y) ↔ x^2/4 + y^2 = 1) ∧
  (∀ (P : ℝ × ℝ), P.2 = 0 → P.1 ∈ Set.Icc (-2 : ℝ) 2 →
    ∀ (l : Line P) (A B : ℝ × ℝ),
      (∃ (t : ℝ), A = (t, (t - P.1)/2) ∧ A.1^2/4 + A.2^2 = 1) →
      (∃ (t : ℝ), B = (t, (t - P.1)/2) ∧ B.1^2/4 + B.2^2 = 1) →
      (A.1 - P.1)^2 + A.2^2 + (B.1 - P.1)^2 + B.2^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2456_245685


namespace NUMINAMATH_CALUDE_yellow_red_ball_arrangements_l2456_245652

theorem yellow_red_ball_arrangements :
  let total_balls : ℕ := 7
  let yellow_balls : ℕ := 4
  let red_balls : ℕ := 3
  Nat.choose total_balls yellow_balls = 35 := by sorry

end NUMINAMATH_CALUDE_yellow_red_ball_arrangements_l2456_245652


namespace NUMINAMATH_CALUDE_x_calculation_l2456_245661

theorem x_calculation (m n p q x : ℝ) :
  x^2 + (2*m*p + 2*n*q)^2 + (2*m*q - 2*n*p)^2 = (m^2 + n^2 + p^2 + q^2)^2 →
  x = m^2 + n^2 - p^2 - q^2 ∨ x = -(m^2 + n^2 - p^2 - q^2) := by
sorry

end NUMINAMATH_CALUDE_x_calculation_l2456_245661


namespace NUMINAMATH_CALUDE_residue_14_power_2046_mod_17_l2456_245676

theorem residue_14_power_2046_mod_17 : 14^2046 % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_14_power_2046_mod_17_l2456_245676


namespace NUMINAMATH_CALUDE_martha_cards_remaining_l2456_245624

theorem martha_cards_remaining (initial_cards : ℝ) (cards_to_emily : ℝ) (cards_to_olivia : ℝ) :
  initial_cards = 76.5 →
  cards_to_emily = 3.1 →
  cards_to_olivia = 5.2 →
  initial_cards - (cards_to_emily + cards_to_olivia) = 68.2 := by
sorry

end NUMINAMATH_CALUDE_martha_cards_remaining_l2456_245624


namespace NUMINAMATH_CALUDE_min_value_expression_l2456_245670

theorem min_value_expression (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 2) 
  (heq : 2 * a + b - 6 = 0) : 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 1 → y > 2 → 2 * x + y - 6 = 0 → 
    1 / (x - 1) + 2 / (y - 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2456_245670


namespace NUMINAMATH_CALUDE_media_team_selection_count_l2456_245646

/-- The number of domestic media teams -/
def domestic_teams : ℕ := 6

/-- The number of foreign media teams -/
def foreign_teams : ℕ := 3

/-- The total number of teams to be selected -/
def selected_teams : ℕ := 3

/-- Represents whether domestic teams can ask questions consecutively -/
def consecutive_domestic : Prop := False

theorem media_team_selection_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_media_team_selection_count_l2456_245646


namespace NUMINAMATH_CALUDE_elisa_books_problem_l2456_245673

theorem elisa_books_problem :
  ∀ (total science math lit : ℕ),
  science = 24 →
  total = science + math + lit →
  total < 100 →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 :=
by sorry

end NUMINAMATH_CALUDE_elisa_books_problem_l2456_245673


namespace NUMINAMATH_CALUDE_arcade_tickets_l2456_245649

theorem arcade_tickets (whack_a_mole skee_ball spent remaining : ℕ) :
  skee_ball = 25 ∧ spent = 7 ∧ remaining = 50 →
  whack_a_mole + skee_ball = remaining + spent →
  whack_a_mole = 7 := by
sorry

end NUMINAMATH_CALUDE_arcade_tickets_l2456_245649


namespace NUMINAMATH_CALUDE_xyz_maximum_l2456_245629

theorem xyz_maximum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_one : x + y + z = 1) (sum_inv_eq_ten : 1/x + 1/y + 1/z = 10) :
  xyz ≤ 4/125 ∧ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 1/x + 1/y + 1/z = 10 ∧ x*y*z = 4/125 :=
by sorry

end NUMINAMATH_CALUDE_xyz_maximum_l2456_245629


namespace NUMINAMATH_CALUDE_molecular_weight_one_mole_l2456_245608

/-- The molecular weight of Aluminium hydroxide for a given number of moles. -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles for which we know the molecular weight. -/
def known_moles : ℝ := 4

/-- The known molecular weight for the given number of moles. -/
def known_weight : ℝ := 312

/-- Theorem stating that the molecular weight of one mole of Aluminium hydroxide is 78 g/mol. -/
theorem molecular_weight_one_mole :
  molecular_weight 1 = 78 :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_one_mole_l2456_245608


namespace NUMINAMATH_CALUDE_uncle_wang_flower_pots_l2456_245686

theorem uncle_wang_flower_pots :
  ∃! x : ℕ,
    ∃ a : ℕ,
      x / 2 + x / 4 + x / 7 + a = x ∧
      1 ≤ a ∧ a < 6 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_wang_flower_pots_l2456_245686


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2456_245625

theorem sufficient_not_necessary_condition (x : ℝ) :
  {x | 1 / x > 1} ⊂ {x | Real.exp (x - 1) < 1} ∧ {x | 1 / x > 1} ≠ {x | Real.exp (x - 1) < 1} :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2456_245625


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2456_245654

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 120 → 
  5 * x = m → 
  y - 12 = m → 
  z + 12 = m → 
  x ≤ y ∧ x ≤ z → 
  y ≥ z → 
  x * y * z = 4095360 / 1331 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2456_245654


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2456_245689

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2456_245689


namespace NUMINAMATH_CALUDE_nested_root_simplification_l2456_245637

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^3 * Real.sqrt (x^4))) = (x^9)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l2456_245637


namespace NUMINAMATH_CALUDE_paris_time_correct_l2456_245610

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24
  mValid : minutes < 60

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a datetime with date and time -/
structure DateTime where
  date : Date
  time : Time

def time_difference : ℤ := -7

def beijing_time : DateTime := {
  date := { year := 2023, month := 10, day := 26 },
  time := { hours := 5, minutes := 0, hValid := by sorry, mValid := by sorry }
}

/-- Calculates the Paris time given the Beijing time and time difference -/
def calculate_paris_time (beijing : DateTime) (diff : ℤ) : DateTime :=
  sorry

theorem paris_time_correct :
  let paris_time := calculate_paris_time beijing_time time_difference
  paris_time.date.day = 25 ∧
  paris_time.date.month = 10 ∧
  paris_time.time.hours = 22 ∧
  paris_time.time.minutes = 0 :=
by sorry

end NUMINAMATH_CALUDE_paris_time_correct_l2456_245610


namespace NUMINAMATH_CALUDE_cube_arrangement_theorem_l2456_245663

/-- Represents a cube with colored faces -/
structure Cube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents an arrangement of cubes into a larger cube -/
structure CubeArrangement where
  cubes : List Cube
  visible_red_faces : Nat
  visible_blue_faces : Nat

/-- The theorem to be proved -/
theorem cube_arrangement_theorem 
  (cubes : List Cube) 
  (first_arrangement : CubeArrangement) :
  (cubes.length = 8) →
  (∀ c ∈ cubes, c.blue_faces = 2 ∧ c.red_faces = 4) →
  (first_arrangement.cubes = cubes) →
  (first_arrangement.visible_red_faces = 8) →
  (first_arrangement.visible_blue_faces = 16) →
  (∃ second_arrangement : CubeArrangement,
    second_arrangement.cubes = cubes ∧
    second_arrangement.visible_red_faces = 24 ∧
    second_arrangement.visible_blue_faces = 0) :=
by sorry

end NUMINAMATH_CALUDE_cube_arrangement_theorem_l2456_245663


namespace NUMINAMATH_CALUDE_exists_self_appended_perfect_square_l2456_245609

theorem exists_self_appended_perfect_square :
  ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ),
    A > 0 ∧ 
    10^n ≤ A ∧ A < 10^(n+1) ∧
    A * (10^n + 1) = B^2 :=
sorry

end NUMINAMATH_CALUDE_exists_self_appended_perfect_square_l2456_245609


namespace NUMINAMATH_CALUDE_value_of_x_l2456_245696

theorem value_of_x (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2456_245696


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2456_245600

/-- Given a line with slope -3 passing through (2, 5), prove m + b = 8 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 → 
  5 = m * 2 + b → 
  m + b = 8 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2456_245600


namespace NUMINAMATH_CALUDE_power_of_power_l2456_245691

theorem power_of_power : (3^2)^4 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2456_245691


namespace NUMINAMATH_CALUDE_percentage_problem_l2456_245655

theorem percentage_problem (x : ℝ) : 80 = 16.666666666666668 / 100 * x → x = 480 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2456_245655


namespace NUMINAMATH_CALUDE_sum_of_digits_A_squared_l2456_245614

/-- For a number with n digits, all being 9 -/
def A (n : ℕ) : ℕ := 10^n - 1

/-- Sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_A_squared :
  sum_of_digits ((A 221)^2) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_A_squared_l2456_245614


namespace NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2456_245677

/-- Represents a repeating decimal with a given numerator and denominator -/
def repeating_decimal (n : ℕ) (d : ℕ) : ℚ := n / d

/-- The sum of three specific repeating decimals -/
def decimal_sum : ℚ :=
  repeating_decimal 1 3 + repeating_decimal 2 99 + repeating_decimal 4 9999

theorem decimal_sum_equals_fraction : decimal_sum = 10581 / 29889 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2456_245677


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l2456_245659

theorem simplify_complex_expression (x : ℝ) (hx : x > 0) : 
  Real.sqrt (2 * (1 + Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2))) = (x^2 + 1) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l2456_245659


namespace NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_34_l2456_245611

def game_condition (M : ℕ) : Prop :=
  M ≤ 1999 ∧
  3 * M < 2000 ∧
  3 * M + 80 < 2000 ∧
  3 * (3 * M + 80) < 2000 ∧
  3 * (3 * M + 80) + 80 < 2000 ∧
  3 * (3 * (3 * M + 80) + 80) ≥ 2000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 34 → ¬(game_condition n) ∧ game_condition 34 :=
by sorry

theorem sum_of_digits_34 : (3 : ℕ) + 4 = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_34_l2456_245611


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l2456_245660

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def totalBricks (rows : ℕ) (bottomRowBricks : ℕ) : ℕ :=
  (rows * (2 * bottomRowBricks - rows + 1)) / 2

/-- Theorem: A brick wall with 5 rows, where the bottom row has 38 bricks
    and each subsequent row has one less brick than the row below it,
    contains a total of 180 bricks. -/
theorem brick_wall_theorem :
  totalBricks 5 38 = 180 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l2456_245660


namespace NUMINAMATH_CALUDE_prob_C_gets_10000_equal_expected_values_l2456_245630

/-- Represents the bonus distribution problem for a work group of three people. -/
structure BonusDistribution where
  total_bonus : ℝ
  p₁ : ℝ  -- Probability of taking 10,000 yuan
  p₂ : ℝ  -- Probability of taking 20,000 yuan

/-- The total bonus is 40,000 yuan -/
def bonus_amount : ℝ := 40000

/-- The probability of A or B taking 10,000 yuan plus the probability of taking 20,000 yuan equals 1 -/
axiom prob_sum (bd : BonusDistribution) : bd.p₁ + bd.p₂ = 1

/-- Expected bonus for A or B -/
def expected_bonus_AB (bd : BonusDistribution) : ℝ := 10000 * bd.p₁ + 20000 * bd.p₂

/-- Expected bonus for C -/
def expected_bonus_C (bd : BonusDistribution) : ℝ := 
  20000 * bd.p₁^2 + 10000 * 2 * bd.p₁ * bd.p₂

/-- Theorem: When p₁ = p₂ = 1/2, the probability that C gets 10,000 yuan is 1/2 -/
theorem prob_C_gets_10000 (bd : BonusDistribution) 
  (h₁ : bd.p₁ = 1/2) (h₂ : bd.p₂ = 1/2) : 
  bd.p₁ * bd.p₂ + bd.p₁ * bd.p₂ = 1/2 := by sorry

/-- Theorem: When expected values are equal, p₁ = 2/3 and p₂ = 1/3 -/
theorem equal_expected_values (bd : BonusDistribution) 
  (h : expected_bonus_AB bd = expected_bonus_C bd) : 
  bd.p₁ = 2/3 ∧ bd.p₂ = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_C_gets_10000_equal_expected_values_l2456_245630


namespace NUMINAMATH_CALUDE_new_shoes_duration_proof_l2456_245695

/-- The duration of new shoes in years -/
def new_shoes_duration : ℝ := 2

/-- The cost of repairing used shoes -/
def used_shoes_repair_cost : ℝ := 10.5

/-- The duration of used shoes after repair in years -/
def used_shoes_duration : ℝ := 1

/-- The cost of new shoes -/
def new_shoes_cost : ℝ := 30

/-- The percentage increase in average cost per year of new shoes compared to repaired used shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

theorem new_shoes_duration_proof :
  new_shoes_duration = new_shoes_cost / (used_shoes_repair_cost * (1 + cost_increase_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_new_shoes_duration_proof_l2456_245695


namespace NUMINAMATH_CALUDE_court_cases_guilty_l2456_245639

theorem court_cases_guilty (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 27 → dismissed = 3 → delayed = 2 → 
  ∃ (guilty : ℕ), guilty = total - dismissed - (3 * (total - dismissed) / 4) - delayed ∧ guilty = 4 := by
sorry

end NUMINAMATH_CALUDE_court_cases_guilty_l2456_245639


namespace NUMINAMATH_CALUDE_score_sum_theorem_l2456_245643

def total_score (keith larry danny emma fiona : ℝ) : ℝ :=
  keith + larry + danny + emma + fiona

theorem score_sum_theorem (keith larry danny emma fiona : ℝ) 
  (h1 : keith = 3.5)
  (h2 : larry = 3.2 * keith)
  (h3 : danny = larry + 5.7)
  (h4 : emma = 2 * danny - 1.2)
  (h5 : fiona = (keith + larry + danny + emma) / 4) :
  total_score keith larry danny emma fiona = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_score_sum_theorem_l2456_245643


namespace NUMINAMATH_CALUDE_calculation_proof_l2456_245617

theorem calculation_proof : (24 / (8 + 2 - 5)) * 7 = 33.6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2456_245617


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2456_245613

/-- The probability of selecting a non-red jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 7
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let non_red : ℕ := green + yellow + blue + purple
  (non_red : ℚ) / total = 36 / 43 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2456_245613


namespace NUMINAMATH_CALUDE_pine_saplings_in_sample_l2456_245645

theorem pine_saplings_in_sample 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 20000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 100) : 
  (sample_size * pine_saplings) / total_saplings = 20 := by
sorry

end NUMINAMATH_CALUDE_pine_saplings_in_sample_l2456_245645


namespace NUMINAMATH_CALUDE_unique_solution_implies_sqrt_three_l2456_245615

theorem unique_solution_implies_sqrt_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 3 = 0) → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_sqrt_three_l2456_245615


namespace NUMINAMATH_CALUDE_fish_catch_difference_l2456_245656

/-- Given the number of fish caught by various birds and a fisherman, prove the difference in catch between the fisherman and pelican. -/
theorem fish_catch_difference (pelican kingfisher osprey fisherman : ℕ) : 
  pelican = 13 →
  kingfisher = pelican + 7 →
  osprey = 2 * kingfisher →
  fisherman = 4 * (pelican + kingfisher + osprey) →
  fisherman - pelican = 279 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_difference_l2456_245656


namespace NUMINAMATH_CALUDE_min_area_inscribed_equilateral_l2456_245651

/-- The minimum area of an inscribed equilateral triangle in a right triangle -/
theorem min_area_inscribed_equilateral (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let min_area := (Real.sqrt 3 * a^2 * b^2) / (4 * (a^2 + b^2 + Real.sqrt 3 * a * b))
  ∀ (D E F : ℝ × ℝ),
    let A := (0, 0)
    let B := (a, 0)
    let C := (0, b)
    (D.1 ≥ 0 ∧ D.1 ≤ a ∧ D.2 = 0) →  -- D is on BC
    (E.1 = 0 ∧ E.2 ≥ 0 ∧ E.2 ≤ b) →  -- E is on CA
    (F.2 = (b / a) * F.1 ∧ F.1 ≥ 0 ∧ F.1 ≤ a) →  -- F is on AB
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 →  -- DEF is equilateral
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (F.1 - D.1)^2 + (F.2 - D.2)^2 →
    let area := Real.sqrt 3 / 4 * ((D.1 - E.1)^2 + (D.2 - E.2)^2)
    area ≥ min_area :=
by sorry

end NUMINAMATH_CALUDE_min_area_inscribed_equilateral_l2456_245651


namespace NUMINAMATH_CALUDE_xiaoyue_speed_l2456_245621

/-- Prove that Xiaoyue's average speed is 50 km/h given the conditions of the problem -/
theorem xiaoyue_speed (x : ℝ) 
  (h1 : x > 0)  -- Xiaoyue's speed is positive
  (h2 : 20 / x - 18 / (1.2 * x) = 1 / 10) : x = 50 := by
  sorry

#check xiaoyue_speed

end NUMINAMATH_CALUDE_xiaoyue_speed_l2456_245621


namespace NUMINAMATH_CALUDE_inequality_range_l2456_245665

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 - m*x - 1) < 0) ↔ 
  (-4 < m ∧ m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2456_245665


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2456_245681

/-- An ellipse with one focus at (0,1) and eccentricity 1/2 has the standard equation x²/3 + y²/4 = 1 -/
theorem ellipse_standard_equation (x y : ℝ) : 
  let e : ℝ := 1/2
  let f : ℝ × ℝ := (0, 1)
  x^2/3 + y^2/4 = 1 ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = 1 ∧
      e = c/a ∧
      a^2 = b^2 + c^2 ∧
      x^2/a^2 + y^2/b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_standard_equation_l2456_245681


namespace NUMINAMATH_CALUDE_talia_drive_distance_l2456_245601

/-- Represents the total distance Talia drives in a day -/
def total_distance (house_to_park park_to_store house_to_store : ℝ) : ℝ :=
  house_to_park + park_to_store + house_to_store

/-- Theorem stating the total distance Talia drives -/
theorem talia_drive_distance :
  ∀ (house_to_park park_to_store house_to_store : ℝ),
    house_to_park = 5 →
    park_to_store = 3 →
    house_to_store = 8 →
    total_distance house_to_park park_to_store house_to_store = 16 := by
  sorry

end NUMINAMATH_CALUDE_talia_drive_distance_l2456_245601


namespace NUMINAMATH_CALUDE_problem_solution_l2456_245690

theorem problem_solution (x y : ℝ) (h1 : y > 2*x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2456_245690


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l2456_245605

/-- Calculates the annual training hours for an athlete with a specific schedule -/
def annualTrainingHours (sessionsPerDay : ℕ) (hoursPerSession : ℕ) (daysPerWeek : ℕ) : ℕ :=
  sessionsPerDay * hoursPerSession * daysPerWeek * 52

/-- Proves that James' annual training hours equal 2080 -/
theorem james_annual_training_hours :
  annualTrainingHours 2 4 5 = 2080 := by
  sorry

#eval annualTrainingHours 2 4 5

end NUMINAMATH_CALUDE_james_annual_training_hours_l2456_245605


namespace NUMINAMATH_CALUDE_john_shopping_cost_l2456_245678

/-- The total cost of buying shirts and ties -/
def total_cost (num_shirts : ℕ) (shirt_price : ℚ) (num_ties : ℕ) (tie_price : ℚ) : ℚ :=
  num_shirts * shirt_price + num_ties * tie_price

/-- Theorem: The total cost of 3 shirts at $15.75 each and 2 ties at $9.40 each is $66.05 -/
theorem john_shopping_cost : 
  total_cost 3 (15.75 : ℚ) 2 (9.40 : ℚ) = (66.05 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_john_shopping_cost_l2456_245678


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2456_245657

theorem same_grade_percentage :
  let total_students : ℕ := 40
  let same_grade_A : ℕ := 3
  let same_grade_B : ℕ := 6
  let same_grade_C : ℕ := 4
  let same_grade_D : ℕ := 2
  let total_same_grade := same_grade_A + same_grade_B + same_grade_C + same_grade_D
  (total_same_grade : ℚ) / total_students * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2456_245657


namespace NUMINAMATH_CALUDE_original_average_proof_l2456_245668

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 10 → new_avg = 160 → new_avg = 2 * original_avg → original_avg = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_average_proof_l2456_245668


namespace NUMINAMATH_CALUDE_senior_trip_fraction_l2456_245680

theorem senior_trip_fraction (total_students : ℝ) (seniors : ℝ) (juniors : ℝ)
  (h1 : juniors = (2/3) * seniors)
  (h2 : (1/4) * juniors + seniors * x = (1/2) * total_students)
  (h3 : total_students = seniors + juniors)
  (h4 : x ≥ 0 ∧ x ≤ 1) :
  x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_senior_trip_fraction_l2456_245680


namespace NUMINAMATH_CALUDE_pens_per_student_l2456_245607

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001)
  (h2 : total_pencils = 910)
  (h3 : max_students = 91) :
  total_pens / max_students = 11 := by
sorry

end NUMINAMATH_CALUDE_pens_per_student_l2456_245607


namespace NUMINAMATH_CALUDE_people_in_line_l2456_245622

theorem people_in_line (people_in_front : ℕ) (people_behind : ℕ) : 
  people_in_front = 11 → people_behind = 12 → people_in_front + people_behind + 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l2456_245622


namespace NUMINAMATH_CALUDE_mixed_doubles_selection_methods_l2456_245662

def male_athletes : ℕ := 5
def female_athletes : ℕ := 6
def selected_male : ℕ := 2
def selected_female : ℕ := 2

theorem mixed_doubles_selection_methods :
  (Nat.choose male_athletes selected_male) *
  (Nat.choose female_athletes selected_female) *
  (Nat.factorial selected_male) = 300 := by
sorry

end NUMINAMATH_CALUDE_mixed_doubles_selection_methods_l2456_245662


namespace NUMINAMATH_CALUDE_puppies_per_cage_l2456_245675

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 78) 
  (h2 : sold_puppies = 30) 
  (h3 : num_cages = 6) 
  (h4 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l2456_245675


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2456_245653

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35)
  (h2 : b + c = 55)
  (h3 : c + a = 62) :
  a + b + c = 76 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2456_245653


namespace NUMINAMATH_CALUDE_distance_between_first_two_points_l2456_245683

theorem distance_between_first_two_points
  (n : ℕ)
  (sum_first : ℝ)
  (sum_second : ℝ)
  (h_n : n = 11)
  (h_sum_first : sum_first = 2018)
  (h_sum_second : sum_second = 2000) :
  ∃ (x : ℝ),
    x = 2 ∧
    x * (n - 2) = sum_first - sum_second :=
by sorry

end NUMINAMATH_CALUDE_distance_between_first_two_points_l2456_245683


namespace NUMINAMATH_CALUDE_grandmother_mother_age_ratio_l2456_245658

/-- Represents the ages of Grace, her mother, and her grandmother -/
structure FamilyAges where
  grace : ℕ
  mother : ℕ
  grandmother : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.grace = 60 ∧
  ages.mother = 80 ∧
  ages.grace * 8 = ages.grandmother * 3 ∧
  ∃ k : ℕ, ages.grandmother = k * ages.mother

/-- The theorem to be proved -/
theorem grandmother_mother_age_ratio 
  (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ages.grandmother / ages.mother = 2 := by
  sorry


end NUMINAMATH_CALUDE_grandmother_mother_age_ratio_l2456_245658


namespace NUMINAMATH_CALUDE_necessary_condition_range_l2456_245619

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_range_l2456_245619


namespace NUMINAMATH_CALUDE_multiple_choice_questions_l2456_245604

theorem multiple_choice_questions (total : ℕ) (problem_solving_percent : ℚ) 
  (h1 : total = 50)
  (h2 : problem_solving_percent = 80 / 100) :
  (total : ℚ) * (1 - problem_solving_percent) = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiple_choice_questions_l2456_245604


namespace NUMINAMATH_CALUDE_angle_S_measure_l2456_245648

/-- Represents a convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angle measures
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  -- Convexity and angle sum property
  sum_angles : P + Q + R + S + T = 540
  -- Angle congruence properties
  PQR_congruent : P = Q ∧ Q = R
  ST_congruent : S = T
  -- Relation between P and S
  P_less_than_S : P + 30 = S

/-- 
Theorem: In a convex pentagon with the given properties, 
the measure of angle S is 126 degrees.
-/
theorem angle_S_measure (pentagon : ConvexPentagon) : pentagon.S = 126 := by
  sorry

end NUMINAMATH_CALUDE_angle_S_measure_l2456_245648


namespace NUMINAMATH_CALUDE_max_points_for_28_lines_l2456_245631

/-- The maximum number of lines that can be determined by n distinct points on a plane -/
def maxLines (n : ℕ) : ℕ :=
  if n ≤ 1 then 0
  else (n * (n - 1)) / 2

/-- Theorem: 8 is the maximum number of distinct points on a plane that can determine at most 28 lines -/
theorem max_points_for_28_lines :
  (∀ n : ℕ, n ≤ 8 → maxLines n ≤ 28) ∧
  (maxLines 8 = 28) :=
sorry

end NUMINAMATH_CALUDE_max_points_for_28_lines_l2456_245631


namespace NUMINAMATH_CALUDE_charity_raffle_proof_l2456_245628

/-- Calculates the total money raised from a charity raffle and donations. -/
def total_money_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation1 : ℚ) (num_donation1 : ℕ) (donation2 : ℚ) : ℚ :=
  (num_tickets : ℚ) * ticket_price + (num_donation1 : ℚ) * donation1 + donation2

/-- Proves that the total money raised is $100.00 given the specific conditions. -/
theorem charity_raffle_proof :
  let num_tickets : ℕ := 25
  let ticket_price : ℚ := 2
  let donation1 : ℚ := 15
  let num_donation1 : ℕ := 2
  let donation2 : ℚ := 20
  total_money_raised num_tickets ticket_price donation1 num_donation1 donation2 = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_charity_raffle_proof_l2456_245628


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l2456_245644

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 9

/-- The number of different books in the series -/
def num_books : ℕ := 10

/-- The number of books read -/
def books_read : ℕ := 14

theorem crazy_silly_school_movies :
  (books_read = num_movies + 5) →
  (books_read ≤ num_books) →
  (num_movies = 9) := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l2456_245644


namespace NUMINAMATH_CALUDE_percentage_equality_l2456_245616

theorem percentage_equality (x : ℝ) : 
  (15 / 100) * 75 = (x / 100) * 450 → x = 2.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l2456_245616


namespace NUMINAMATH_CALUDE_circle_triangle_area_relation_l2456_245679

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  -- The triangle has sides 20, 21, and 29
  20^2 + 21^2 = 29^2 →
  -- A circle is circumscribed about the triangle
  -- A, B, and C are areas of non-triangular regions
  -- C is the largest area
  C ≥ A ∧ C ≥ B →
  -- The area of the triangle is 210
  (20 * 21) / 2 = 210 →
  -- The diameter of the circle is 29
  -- C is half the area of the circle
  C = (29^2 * π) / 8 →
  -- Prove that A + B + 210 = C
  A + B + 210 = C :=
by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_relation_l2456_245679


namespace NUMINAMATH_CALUDE_shop_owner_profit_l2456_245664

/-- Represents the profit calculation for a shop owner using false weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.12 →
  selling_cheat = 0.30 →
  let actual_buy_amount := 1 + buying_cheat
  let actual_sell_amount := 1 - selling_cheat
  let sell_portions := actual_buy_amount / actual_sell_amount
  let revenue := sell_portions * 100
  let profit := revenue - 100
  let percentage_profit := (profit / 100) * 100
  percentage_profit = 60 := by
sorry


end NUMINAMATH_CALUDE_shop_owner_profit_l2456_245664


namespace NUMINAMATH_CALUDE_smallest_odd_prime_divisor_of_difference_of_squares_l2456_245627

def is_odd_prime (p : Nat) : Prop := Nat.Prime p ∧ p % 2 = 1

theorem smallest_odd_prime_divisor_of_difference_of_squares :
  ∃ (k : Nat), k = 3 ∧
  (∀ (m n : Nat), is_odd_prime m → is_odd_prime n → m < 10 → n < 10 → n < m →
    k ∣ (m^2 - n^2)) ∧
  (∀ (p : Nat), p < k → is_odd_prime p →
    ∃ (m n : Nat), is_odd_prime m ∧ is_odd_prime n ∧ m < 10 ∧ n < 10 ∧ n < m ∧
      ¬(p ∣ (m^2 - n^2))) := by
sorry

end NUMINAMATH_CALUDE_smallest_odd_prime_divisor_of_difference_of_squares_l2456_245627


namespace NUMINAMATH_CALUDE_total_winter_clothing_l2456_245602

/-- The number of boxes containing winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of pieces of winter clothing is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l2456_245602


namespace NUMINAMATH_CALUDE_equal_squares_of_equal_products_l2456_245698

theorem equal_squares_of_equal_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * (b + c + d) = b * (a + c + d))
  (h2 : a * (b + c + d) = c * (a + b + d))
  (h3 : a * (b + c + d) = d * (a + b + c)) :
  a^2 = b^2 ∧ a^2 = c^2 ∧ a^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_equal_squares_of_equal_products_l2456_245698


namespace NUMINAMATH_CALUDE_diverse_dates_2013_l2456_245684

/-- A date in the format DD/MM/YY -/
structure Date where
  day : Nat
  month : Nat
  year : Nat

/-- Check if a date is valid (day between 1 and 31, month between 1 and 12) -/
def Date.isValid (d : Date) : Prop :=
  1 ≤ d.day ∧ d.day ≤ 31 ∧ 1 ≤ d.month ∧ d.month ≤ 12

/-- Convert a date to a list of digits -/
def Date.toDigits (d : Date) : List Nat :=
  (d.day / 10) :: (d.day % 10) :: (d.month / 10) :: (d.month % 10) :: (d.year / 10) :: [d.year % 10]

/-- Check if a date is diverse (contains all digits from 0 to 5 exactly once) -/
def Date.isDiverse (d : Date) : Prop :=
  let digits := d.toDigits
  ∀ n : Nat, n ≤ 5 → (digits.count n = 1)

/-- The main theorem: there are exactly 2 diverse dates in 2013 -/
theorem diverse_dates_2013 :
  ∃! (dates : List Date), 
    (∀ d ∈ dates, d.year = 13 ∧ d.isValid ∧ d.isDiverse) ∧ 
    dates.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_diverse_dates_2013_l2456_245684


namespace NUMINAMATH_CALUDE_original_average_calculation_l2456_245692

theorem original_average_calculation (S : Finset ℝ) (A : ℝ) :
  Finset.card S = 10 →
  (Finset.sum S id + 8) / 10 = 7 →
  Finset.sum S id = 10 * A →
  A = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_original_average_calculation_l2456_245692


namespace NUMINAMATH_CALUDE_march_and_may_greatest_drop_l2456_245666

/-- Represents the months of the year --/
inductive Month
| January | February | March | April | May | June | July | August

/-- Price change for each month --/
def price_change : Month → ℝ
| Month.January  => -1.00
| Month.February => 1.50
| Month.March    => -3.00
| Month.April    => 2.00
| Month.May      => -3.00
| Month.June     => 0.50
| Month.July     => -2.50
| Month.August   => -1.50

/-- Predicate to check if a month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ n : Month, price_change m ≤ price_change n

/-- Theorem stating that March and May have the greatest monthly drop in price --/
theorem march_and_may_greatest_drop :
  has_greatest_drop Month.March ∧ has_greatest_drop Month.May :=
sorry

end NUMINAMATH_CALUDE_march_and_may_greatest_drop_l2456_245666


namespace NUMINAMATH_CALUDE_same_heads_probability_l2456_245633

def fair_coin_prob : ℚ := 1/2
def coin_prob_1 : ℚ := 3/5
def coin_prob_2 : ℚ := 2/3

def same_heads_prob : ℚ := 29/90

theorem same_heads_probability :
  let outcomes := (1 + 1) * (2 + 3) * (1 + 2)
  let squared_sum := (2^2 + 9^2 + 13^2 + 6^2 : ℚ)
  same_heads_prob = squared_sum / (outcomes^2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_same_heads_probability_l2456_245633


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2456_245606

/-- Given a quadratic equation x^2 + 4x - 1 = 0 with roots m and n, prove that m + n + mn = -5 -/
theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (∀ x, x^2 + 4*x - 1 = 0 ↔ x = m ∨ x = n) → m + n + m*n = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2456_245606


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_220_l2456_245603

/-- A trapezoid with the given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  angle_BCD : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + 2 * t.BC + t.CD

/-- Theorem stating that the perimeter of the given trapezoid is 220 -/
theorem trapezoid_perimeter_is_220 (t : Trapezoid) 
  (h1 : t.AB = 60)
  (h2 : t.CD = 40)
  (h3 : t.angle_BCD = 120 * π / 180)
  (h4 : t.BC = Real.sqrt (t.CD^2 + (t.AB - t.CD)^2 - 2 * t.CD * (t.AB - t.CD) * Real.cos t.angle_BCD)) :
  perimeter t = 220 := by
  sorry

#check trapezoid_perimeter_is_220

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_220_l2456_245603


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l2456_245667

theorem unique_solution_cubic_system (x y z : ℝ) :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 3 →
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l2456_245667


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l2456_245671

/-- Given a principal amount, final amount, and time period, 
    calculate the simple interest rate as a percentage. -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 7.6% -/
theorem simple_interest_rate_example : 
  simple_interest_rate 25000 34500 5 = 76/10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l2456_245671


namespace NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_half_l2456_245697

/-- Given two vectors a and b in R², if a is perpendicular to b,
    then the second component of a is equal to 1/2. -/
theorem perpendicular_vectors_implies_m_half (a b : ℝ × ℝ) :
  a.1 = 1 →
  a.2 = m →
  b.1 = -1 →
  b.2 = 2 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_half_l2456_245697


namespace NUMINAMATH_CALUDE_seokjin_pencils_used_l2456_245672

/-- The number of pencils Seokjin used -/
def pencils_seokjin_used : ℕ := 9

theorem seokjin_pencils_used 
  (initial_pencils : ℕ) 
  (pencils_given : ℕ) 
  (final_pencils : ℕ) 
  (h1 : initial_pencils = 12)
  (h2 : pencils_given = 4)
  (h3 : final_pencils = 7) :
  pencils_seokjin_used = initial_pencils - final_pencils + pencils_given :=
by sorry

#check seokjin_pencils_used

end NUMINAMATH_CALUDE_seokjin_pencils_used_l2456_245672


namespace NUMINAMATH_CALUDE_average_problem_l2456_245623

theorem average_problem (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2456_245623


namespace NUMINAMATH_CALUDE_sin_cube_identity_l2456_245638

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l2456_245638


namespace NUMINAMATH_CALUDE_arctan_sum_equals_arctan_29_22_l2456_245682

theorem arctan_sum_equals_arctan_29_22 (a b : ℝ) : 
  a = 3/4 → (a + 1) * (b + 1) = 9/4 → Real.arctan a + Real.arctan b = Real.arctan (29/22) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_arctan_29_22_l2456_245682


namespace NUMINAMATH_CALUDE_largest_value_l2456_245650

theorem largest_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2456_245650


namespace NUMINAMATH_CALUDE_train_length_l2456_245636

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2456_245636


namespace NUMINAMATH_CALUDE_pie_distribution_probability_l2456_245632

/-- Represents the total number of pies -/
def total_pies : ℕ := 6

/-- Represents the number of growth pies -/
def growth_pies : ℕ := 2

/-- Represents the number of shrink pies -/
def shrink_pies : ℕ := 4

/-- Represents the number of pies given to Mary -/
def pies_given : ℕ := 3

/-- The probability that one of the girls does not have a single growth pie -/
def prob_no_growth_pie : ℚ := 7/10

theorem pie_distribution_probability :
  prob_no_growth_pie = 1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given) :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_probability_l2456_245632


namespace NUMINAMATH_CALUDE_book_price_percentage_l2456_245620

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.7 * suggested_retail_price
  let purchase_price := 0.5 * marked_price
  purchase_price / suggested_retail_price = 0.35 := by sorry

end NUMINAMATH_CALUDE_book_price_percentage_l2456_245620


namespace NUMINAMATH_CALUDE_binomial_10_3_l2456_245693

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2456_245693


namespace NUMINAMATH_CALUDE_remainder_theorem_l2456_245642

theorem remainder_theorem : ∃ q : ℕ, 2^404 + 404 = (2^203 + 2^101 + 1) * q + 403 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2456_245642


namespace NUMINAMATH_CALUDE_intersection_max_value_l2456_245618

/-- The polynomial function f(x) = x^6 - 10x^5 + 30x^4 - 20x^3 + 50x^2 - 24x + 48 -/
def f (x : ℝ) : ℝ := x^6 - 10*x^5 + 30*x^4 - 20*x^3 + 50*x^2 - 24*x + 48

/-- The line function g(x) = 8x -/
def g (x : ℝ) : ℝ := 8*x

theorem intersection_max_value :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (∀ x : ℝ, f x = g x → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → y ≤ x) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → x = 6) :=
sorry

end NUMINAMATH_CALUDE_intersection_max_value_l2456_245618


namespace NUMINAMATH_CALUDE_derivative_of_y_l2456_245641

noncomputable def y (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_y (x : ℝ) (h : x ≠ 0) :
  deriv y x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2456_245641


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l2456_245626

theorem final_sum_after_transformation (a b c S : ℝ) (h : a + b + c = S) :
  3 * (a - 4) + 3 * (b - 4) + 3 * (c - 4) = 3 * S - 36 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l2456_245626


namespace NUMINAMATH_CALUDE_initial_game_cost_l2456_245669

theorem initial_game_cost (triple_value : ℝ → ℝ) (sold_percentage : ℝ) (sold_amount : ℝ) :
  triple_value = (λ x => 3 * x) →
  sold_percentage = 0.4 →
  sold_amount = 240 →
  ∃ (initial_cost : ℝ), sold_percentage * triple_value initial_cost = sold_amount ∧ initial_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_initial_game_cost_l2456_245669


namespace NUMINAMATH_CALUDE_distance_calculation_l2456_245687

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 54

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 6

/-- The time Brad runs before meeting Maxwell, in hours -/
def brad_time : ℝ := maxwell_time - 1

theorem distance_calculation :
  distance_between_homes = maxwell_speed * maxwell_time + brad_speed * brad_time :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l2456_245687


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2456_245640

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2456_245640


namespace NUMINAMATH_CALUDE_bank_through_window_l2456_245635

/-- Represents a letter as seen through a clear glass window from the inside --/
inductive MirroredLetter
  | Normal (c : Char)
  | Inverted (c : Char)

/-- Represents a word as seen through a clear glass window from the inside --/
def MirroredWord := List MirroredLetter

/-- Converts a character to its mirrored version --/
def mirrorChar (c : Char) : MirroredLetter :=
  match c with
  | 'B' => MirroredLetter.Inverted 'В'
  | 'A' => MirroredLetter.Normal 'A'
  | 'N' => MirroredLetter.Inverted 'И'
  | 'K' => MirroredLetter.Inverted 'И'
  | _ => MirroredLetter.Normal c

/-- Converts a string to its mirrored version --/
def mirrorWord (s : String) : MirroredWord :=
  s.toList.reverse.map mirrorChar

/-- Converts a MirroredWord to a string --/
def mirroredWordToString (w : MirroredWord) : String :=
  w.map (fun l => match l with
    | MirroredLetter.Normal c => c
    | MirroredLetter.Inverted c => c
  ) |>.asString

theorem bank_through_window :
  mirroredWordToString (mirrorWord "BANK") = "ИAИВ" := by
  sorry

#eval mirroredWordToString (mirrorWord "BANK")

end NUMINAMATH_CALUDE_bank_through_window_l2456_245635


namespace NUMINAMATH_CALUDE_sum_of_angles_l2456_245634

theorem sum_of_angles (α β : Real) : 
  (∃ x y : Real, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (0 < α ∧ α < π/2) →
  (0 < β ∧ β < π/2) →
  α + β = 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_l2456_245634


namespace NUMINAMATH_CALUDE_four_tangent_lines_with_equal_intercepts_l2456_245674

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let d := |l.a * x₀ + l.b * y₀ + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d = c.radius

/-- Check if a line has equal intercepts on both axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The main theorem -/
theorem four_tangent_lines_with_equal_intercepts :
  let c : Circle := { center := (3, 3), radius := Real.sqrt 8 }
  ∃ (lines : Finset Line),
    lines.card = 4 ∧
    ∀ l ∈ lines, isTangent l c ∧ hasEqualIntercepts l ∧
    ∀ l', isTangent l' c → hasEqualIntercepts l' → l' ∈ lines :=
sorry

end NUMINAMATH_CALUDE_four_tangent_lines_with_equal_intercepts_l2456_245674
