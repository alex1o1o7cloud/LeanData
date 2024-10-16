import Mathlib

namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_equation_l1949_194981

/-- Represents the average monthly growth rate of profit from January to March -/
def monthly_growth_rate : ℝ → Prop :=
  fun x => 3 * (1 + x)^2 = 3.63

/-- The profit in January -/
def january_profit : ℝ := 30000

/-- The profit in March -/
def march_profit : ℝ := 36300

/-- Theorem stating the equation for the average monthly growth rate -/
theorem average_monthly_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate x ∧
    march_profit = january_profit * (1 + x)^2 :=
  sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_equation_l1949_194981


namespace NUMINAMATH_CALUDE_tutor_schedules_lcm_l1949_194960

/-- The work schedules of the tutors -/
def tutor_schedules : List Nat := [5, 6, 8, 9, 10]

/-- The theorem stating that the LCM of the tutor schedules is 360 -/
theorem tutor_schedules_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedules_lcm_l1949_194960


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1949_194902

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1949_194902


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l1949_194905

theorem ticket_price_reduction (x : ℝ) (y : ℝ) (h1 : x > 0) : 
  (4/3 * x * (50 - y) = 5/4 * x * 50) → y = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l1949_194905


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1949_194900

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebras_per_tiger : ℕ := 2

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The total number of zebra enclosures in the zoo -/
def total_zebra_enclosures : ℕ := tiger_enclosures * zebras_per_tiger

/-- The total number of giraffe enclosures in the zoo -/
def total_giraffe_enclosures : ℕ := total_zebra_enclosures * giraffe_to_zebra_ratio

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 
  tiger_enclosures * tigers_per_enclosure + 
  total_zebra_enclosures * zebras_per_enclosure + 
  total_giraffe_enclosures * giraffes_per_enclosure

theorem zoo_animal_count : total_animals = 144 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1949_194900


namespace NUMINAMATH_CALUDE_dentist_age_fraction_l1949_194977

theorem dentist_age_fraction (F : ℚ) : 
  let current_age : ℕ := 32
  let age_8_years_ago : ℕ := current_age - 8
  let age_8_years_hence : ℕ := current_age + 8
  F * age_8_years_ago = (1 : ℚ) / 10 * age_8_years_hence →
  F = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_fraction_l1949_194977


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_condition_l1949_194991

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

-- State the theorem
theorem f_inequality_iff_a_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ -1 / (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_condition_l1949_194991


namespace NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l1949_194911

theorem sum_of_divisors_900_prime_factors : 
  let n := 900
  let sum_of_divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id
  (Nat.factors sum_of_divisors).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l1949_194911


namespace NUMINAMATH_CALUDE_tangent_line_circle_l1949_194995

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle -/
def is_tangent (r : ℝ) : Prop :=
  r > 0 ∧ (r / Real.sqrt 2 = 2 * Real.sqrt r)

theorem tangent_line_circle (r : ℝ) : is_tangent r ↔ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l1949_194995


namespace NUMINAMATH_CALUDE_cube_sum_equation_l1949_194933

/-- Given real numbers p, q, and r satisfying certain conditions, 
    their cubes sum to 181. -/
theorem cube_sum_equation (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 := by
  sorry


end NUMINAMATH_CALUDE_cube_sum_equation_l1949_194933


namespace NUMINAMATH_CALUDE_parabola_tangent_lines_l1949_194904

/-- The parabola defined by x^2 = 4y with focus (0, 1) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- The line perpendicular to the y-axis passing through the focus -/
def PerpendicularLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

/-- The intersection points of the parabola and the perpendicular line -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  Parabola ∩ PerpendicularLine

/-- The tangent line at a point on the parabola -/
def TangentLine (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1 + p.2 * q.2 + (p.1^2 / 4 + 1) = 0}

theorem parabola_tangent_lines :
  ∀ p ∈ IntersectionPoints,
    TangentLine p = {q : ℝ × ℝ | q.1 + q.2 + 1 = 0} ∨
    TangentLine p = {q : ℝ × ℝ | q.1 - q.2 - 1 = 0} :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_lines_l1949_194904


namespace NUMINAMATH_CALUDE_total_points_scored_l1949_194954

theorem total_points_scored (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l1949_194954


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l1949_194926

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x + 1| ≤ 1 → (x - 1) * (x + 2) ≤ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≤ 0 ∧ |x + 1| > 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l1949_194926


namespace NUMINAMATH_CALUDE_curve_in_second_quadrant_implies_a_range_l1949_194909

theorem curve_in_second_quadrant_implies_a_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) →
  a > 2 :=
by sorry

end NUMINAMATH_CALUDE_curve_in_second_quadrant_implies_a_range_l1949_194909


namespace NUMINAMATH_CALUDE_flush_probability_l1949_194912

/-- Represents the number of players in the card game -/
def num_players : ℕ := 4

/-- Represents the total number of cards in the deck -/
def total_cards : ℕ := 20

/-- Represents the number of cards per suit -/
def cards_per_suit : ℕ := 5

/-- Represents the number of cards dealt to each player -/
def cards_per_player : ℕ := 5

/-- Calculates the probability of at least one player having a flush after card exchange -/
def probability_of_flush : ℚ := 8 / 969

/-- Theorem stating the probability of at least one player having a flush after card exchange -/
theorem flush_probability : 
  probability_of_flush = 8 / 969 :=
sorry

end NUMINAMATH_CALUDE_flush_probability_l1949_194912


namespace NUMINAMATH_CALUDE_nested_bracket_calculation_l1949_194947

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_calculation :
  bracket (bracket 100 20 60) (bracket 7 2 3) (bracket 20 10 10) = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_nested_bracket_calculation_l1949_194947


namespace NUMINAMATH_CALUDE_intersection_M_N_l1949_194997

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1949_194997


namespace NUMINAMATH_CALUDE_max_value_of_f_min_value_of_expression_equality_condition_l1949_194946

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (y : ℝ), f y = m :=
sorry

-- Theorem for the minimum value of the expression
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) ≥ 1/3 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) = 1/3 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_min_value_of_expression_equality_condition_l1949_194946


namespace NUMINAMATH_CALUDE_max_label_proof_l1949_194967

/-- Counts the number of '5' digits used to label boxes from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The maximum number that can be labeled using 50 '5' digits --/
def max_label : ℕ := 235

theorem max_label_proof :
  count_fives max_label ≤ 50 ∧
  ∀ m : ℕ, m > max_label → count_fives m > 50 :=
sorry

end NUMINAMATH_CALUDE_max_label_proof_l1949_194967


namespace NUMINAMATH_CALUDE_students_in_both_teams_l1949_194913

theorem students_in_both_teams (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) 
  (h2 : baseball = 25) 
  (h3 : hockey = 19) : 
  baseball + hockey - total = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_teams_l1949_194913


namespace NUMINAMATH_CALUDE_three_numbers_sum_to_perfect_square_l1949_194917

def numbers : List Nat := [4784887, 2494651, 8595087, 1385287, 9042451, 9406087]

theorem three_numbers_sum_to_perfect_square :
  ∃ (a b c : Nat), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (n : Nat), a + b + c = n * n :=
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_to_perfect_square_l1949_194917


namespace NUMINAMATH_CALUDE_brownies_count_l1949_194929

/-- Given a box that can hold 7 brownies and 49 full boxes of brownies,
    prove that the total number of brownies is 343. -/
theorem brownies_count (brownies_per_box : ℕ) (full_boxes : ℕ) 
  (h1 : brownies_per_box = 7)
  (h2 : full_boxes = 49) : 
  brownies_per_box * full_boxes = 343 := by
  sorry

end NUMINAMATH_CALUDE_brownies_count_l1949_194929


namespace NUMINAMATH_CALUDE_tire_circumference_l1949_194973

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 →
  car_speed_kmh = 96 →
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 4 := by
sorry

end NUMINAMATH_CALUDE_tire_circumference_l1949_194973


namespace NUMINAMATH_CALUDE_c_class_size_l1949_194976

/-- The number of students in each class -/
structure ClassSize where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : ClassSize) : Prop :=
  s.A = 44 ∧ s.B = s.A + 2 ∧ s.C = s.B - 1

/-- The theorem to prove -/
theorem c_class_size (s : ClassSize) (h : problem_conditions s) : s.C = 45 := by
  sorry


end NUMINAMATH_CALUDE_c_class_size_l1949_194976


namespace NUMINAMATH_CALUDE_percentage_difference_l1949_194966

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1949_194966


namespace NUMINAMATH_CALUDE_cost_splitting_difference_l1949_194937

def bob_paid : ℚ := 130
def alice_paid : ℚ := 110
def jessica_paid : ℚ := 160

def total_paid : ℚ := bob_paid + alice_paid + jessica_paid
def share_per_person : ℚ := total_paid / 3

def bob_owes : ℚ := share_per_person - bob_paid
def alice_owes : ℚ := share_per_person - alice_paid
def jessica_receives : ℚ := jessica_paid - share_per_person

theorem cost_splitting_difference :
  bob_owes - alice_owes = -20 := by sorry

end NUMINAMATH_CALUDE_cost_splitting_difference_l1949_194937


namespace NUMINAMATH_CALUDE_negation_both_even_l1949_194941

theorem negation_both_even (a b : ℤ) :
  ¬(Even a ∧ Even b) ↔ ¬(Even a ∧ Even b) :=
sorry

end NUMINAMATH_CALUDE_negation_both_even_l1949_194941


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1949_194906

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_prod : a 4 * a 10 = 16) : 
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1949_194906


namespace NUMINAMATH_CALUDE_sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l1949_194927

theorem sin_negative_ten_pi_thirds_equals_sqrt_three_halves :
  Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l1949_194927


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l1949_194983

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 30 →         -- Perimeter is 30
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 :=
by sorry

#check max_triangle_side_length

end NUMINAMATH_CALUDE_max_triangle_side_length_l1949_194983


namespace NUMINAMATH_CALUDE_decimal_expansion_three_eighths_no_repeat_l1949_194958

/-- The length of the smallest repeating block in the decimal expansion of 3/8 is 0. -/
theorem decimal_expansion_three_eighths_no_repeat : 
  (∃ n : ℕ, ∃ k : ℕ, (3 : ℚ) / 8 = (n : ℚ) / 10^k ∧ k > 0) := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_three_eighths_no_repeat_l1949_194958


namespace NUMINAMATH_CALUDE_pencil_black_fraction_l1949_194932

theorem pencil_black_fraction :
  ∀ (total_length blue_length white_length black_length : ℝ),
    total_length = 8 →
    blue_length = 3.5 →
    white_length = (total_length - blue_length) / 2 →
    black_length = total_length - blue_length - white_length →
    black_length / total_length = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_pencil_black_fraction_l1949_194932


namespace NUMINAMATH_CALUDE_point_movement_theorem_l1949_194938

/-- The initial position of a point on a number line that ends at the origin after moving right 7 units and then left 4 units -/
def initial_position : ℤ := -3

/-- A point's movement on a number line -/
def point_movement (start : ℤ) : ℤ := start + 7 - 4

theorem point_movement_theorem :
  point_movement initial_position = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l1949_194938


namespace NUMINAMATH_CALUDE_problem_1_l1949_194990

theorem problem_1 : (1) - 8 + 12 - 16 - 23 = -35 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1949_194990


namespace NUMINAMATH_CALUDE_confidence_level_error_probability_l1949_194944

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := { r : ℝ // 0 < r ∧ r < 1 }

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

theorem confidence_level_error_probability 
  (cl : ConfidenceLevel) 
  (hp : cl.val = 0.95) :
  (calculateErrorProbability cl).val = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_confidence_level_error_probability_l1949_194944


namespace NUMINAMATH_CALUDE_uncle_fyodor_cannot_always_win_l1949_194908

/-- Represents a sandwich with sausage and cheese -/
structure Sandwich :=
  (hasSausage : Bool)

/-- Represents the state of the game -/
structure GameState :=
  (sandwiches : List Sandwich)
  (turn : Nat)

/-- Uncle Fyodor's move: eat one sandwich from either end -/
def uncleFyodorMove (state : GameState) : GameState :=
  sorry

/-- Matroskin's move: remove sausage from one sandwich or do nothing -/
def matroskinMove (state : GameState) : GameState :=
  sorry

/-- Play the game until all sandwiches are eaten -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Check if Uncle Fyodor wins (last sandwich eaten contains sausage) -/
def uncleFyodorWins (finalState : GameState) : Bool :=
  sorry

/-- Theorem: There exists a natural number N for which Uncle Fyodor cannot guarantee a win -/
theorem uncle_fyodor_cannot_always_win :
  ∃ N : Nat, ∀ uncleFyodorStrategy : GameState → GameState,
    ∃ matroskinStrategy : GameState → GameState,
      let initialState := GameState.mk (List.replicate N (Sandwich.mk true)) 0
      ¬(uncleFyodorWins (playGame initialState)) :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_fyodor_cannot_always_win_l1949_194908


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_4_3_l1949_194920

theorem smallest_four_digit_mod_4_3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 3 [ZMOD 4] → 1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_4_3_l1949_194920


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1949_194996

theorem abs_inequality_equivalence (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 8 ↔ x ∈ Set.Icc (-3) 3 ∪ Set.Icc 7 13 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1949_194996


namespace NUMINAMATH_CALUDE_solution_range_l1949_194988

-- Define the operation @
def op (p q : ℝ) : ℝ := p + q - p * q

-- Define the main theorem
theorem solution_range (m : ℝ) :
  (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    op 2 (x₁ : ℝ) > 0 ∧ op (x₁ : ℝ) 3 ≤ m ∧
    op 2 (x₂ : ℝ) > 0 ∧ op (x₂ : ℝ) 3 ≤ m ∧
    (∀ (x : ℤ), x ≠ x₁ ∧ x ≠ x₂ → op 2 (x : ℝ) ≤ 0 ∨ op (x : ℝ) 3 > m)) →
  3 ≤ m ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_solution_range_l1949_194988


namespace NUMINAMATH_CALUDE_polygon_area_l1949_194970

/-- A polygon on a unit grid with vertices at (0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0) -/
def polygon : List (ℤ × ℤ) := [(0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0)]

/-- The area enclosed by the polygon -/
def enclosed_area (p : List (ℤ × ℤ)) : ℚ := sorry

/-- Theorem stating that the area enclosed by the polygon is 37.5 square units -/
theorem polygon_area : enclosed_area polygon = 37.5 := by sorry

end NUMINAMATH_CALUDE_polygon_area_l1949_194970


namespace NUMINAMATH_CALUDE_quadratic_circle_theorem_l1949_194992

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

-- Define the condition that f intersects the axes at three points
def intersects_axes_at_three_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b 0 ≠ 0

-- Define the equation of the circle
def circle_equation (b : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*x - (b + 1)*y + b

-- Main theorem
theorem quadratic_circle_theorem (b : ℝ) 
  (h : intersects_axes_at_three_points b) :
  (b < 1 ∧ b ≠ 0) ∧
  (∀ x y : ℝ, circle_equation b x y = 0 ↔ 
    (x = 0 ∧ f b y = 0) ∨ (y = 0 ∧ f b x = 0) ∨ (x = 0 ∧ y = b)) ∧
  (circle_equation b 0 1 = 0 ∧ circle_equation b (-2) 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_circle_theorem_l1949_194992


namespace NUMINAMATH_CALUDE_partnership_profit_b_profit_calculation_l1949_194943

/-- Profit calculation in a partnership --/
theorem partnership_profit (a_investment b_investment : ℕ) 
  (a_period b_period : ℕ) (total_profit : ℕ) : ℕ :=
  let a_share := a_investment * a_period
  let b_share := b_investment * b_period
  let total_share := a_share + b_share
  let b_profit := (b_share * total_profit) / total_share
  b_profit

/-- B's profit in the given partnership scenario --/
theorem b_profit_calculation : 
  partnership_profit 3 1 2 1 31500 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_b_profit_calculation_l1949_194943


namespace NUMINAMATH_CALUDE_min_markers_to_sell_is_1200_l1949_194962

/-- Represents the number of markers bought -/
def markers_bought : ℕ := 2000

/-- Represents the cost price of each marker in cents -/
def cost_price : ℕ := 20

/-- Represents the selling price of each marker in cents -/
def selling_price : ℕ := 50

/-- Represents the minimum profit desired in cents -/
def min_profit : ℕ := 20000

/-- Calculates the minimum number of markers that must be sold to achieve the desired profit -/
def min_markers_to_sell : ℕ :=
  (markers_bought * cost_price + min_profit) / (selling_price - cost_price)

/-- Theorem stating that the minimum number of markers to sell is 1200 -/
theorem min_markers_to_sell_is_1200 : min_markers_to_sell = 1200 := by
  sorry

#eval min_markers_to_sell

end NUMINAMATH_CALUDE_min_markers_to_sell_is_1200_l1949_194962


namespace NUMINAMATH_CALUDE_sector_central_angle_l1949_194935

/-- Given a sector with radius 2 cm and area 4 cm², 
    prove that the radian measure of its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r^2 * α → α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1949_194935


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1949_194955

theorem complex_number_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1949_194955


namespace NUMINAMATH_CALUDE_may_total_scarves_l1949_194961

/-- The number of scarves that can be knitted from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May can make -/
def total_scarves : ℕ := scarves_per_yarn * (red_yarns + blue_yarns + yellow_yarns)

theorem may_total_scarves : total_scarves = 36 := by
  sorry

end NUMINAMATH_CALUDE_may_total_scarves_l1949_194961


namespace NUMINAMATH_CALUDE_dot_product_sum_and_a_l1949_194986

/-- Given vectors a and b in ℝ², prove that the dot product of (a + b) and a equals 1. -/
theorem dot_product_sum_and_a (a b : ℝ × ℝ) (h1 : a = (1/2, Real.sqrt 3/2)) 
    (h2 : b = (-Real.sqrt 3/2, 1/2)) : 
    (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_and_a_l1949_194986


namespace NUMINAMATH_CALUDE_probability_two_boys_three_girls_l1949_194956

def probability_boy_or_girl : ℝ := 0.5

def number_of_children : ℕ := 5

def number_of_boys : ℕ := 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_two_boys_three_girls :
  (binomial_coefficient number_of_children number_of_boys : ℝ) *
  probability_boy_or_girl ^ number_of_boys *
  probability_boy_or_girl ^ (number_of_children - number_of_boys) =
  0.3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_boys_three_girls_l1949_194956


namespace NUMINAMATH_CALUDE_f_range_l1949_194916

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then (Real.log x) / x else Real.exp x + 1

theorem f_range :
  Set.range f = Set.union (Set.Ioc 0 (1 / Real.exp 1)) (Set.Ioo 1 (Real.exp 1 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l1949_194916


namespace NUMINAMATH_CALUDE_total_commute_time_l1949_194945

def first_bus_duration : ℕ := 40
def first_wait_duration : ℕ := 10
def second_bus_duration : ℕ := 50
def second_wait_duration : ℕ := 15
def third_bus_duration : ℕ := 95

theorem total_commute_time :
  first_bus_duration + first_wait_duration + second_bus_duration +
  second_wait_duration + third_bus_duration = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_l1949_194945


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1949_194978

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  a = (1, 2) ∧ b = (3, 1) → angle = π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1949_194978


namespace NUMINAMATH_CALUDE_carter_performs_30_nights_l1949_194919

/-- The number of nights Carter performs, given his drum stick usage pattern --/
def carter_performance_nights (sticks_per_show : ℕ) (sticks_tossed : ℕ) (total_sticks : ℕ) : ℕ :=
  total_sticks / (sticks_per_show + sticks_tossed)

/-- Theorem stating that Carter performs for 30 nights under the given conditions --/
theorem carter_performs_30_nights :
  carter_performance_nights 5 6 330 = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_performs_30_nights_l1949_194919


namespace NUMINAMATH_CALUDE_phone_number_probability_l1949_194975

theorem phone_number_probability (n : ℕ) (h : n = 10) :
  let p : ℚ := 1 / n
  1 - (1 - p) * (1 - p) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l1949_194975


namespace NUMINAMATH_CALUDE_garden_perimeter_l1949_194936

/-- The perimeter of a rectangular garden with width 12 meters and an area equal to that of a 16m × 12m playground is 56 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 56 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1949_194936


namespace NUMINAMATH_CALUDE_wedding_chairs_count_l1949_194998

/-- The number of rows of chairs -/
def num_rows : ℕ := 7

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 12

/-- The number of extra chairs added -/
def extra_chairs : ℕ := 11

/-- The total number of chairs -/
def total_chairs : ℕ := num_rows * chairs_per_row + extra_chairs

theorem wedding_chairs_count : total_chairs = 95 := by
  sorry

end NUMINAMATH_CALUDE_wedding_chairs_count_l1949_194998


namespace NUMINAMATH_CALUDE_m_upper_bound_l1949_194982

theorem m_upper_bound (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 0, f x = Real.exp x + Real.exp (-x)) →
  (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) →
  m ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_m_upper_bound_l1949_194982


namespace NUMINAMATH_CALUDE_carolyns_silverware_percentage_l1949_194993

/-- Represents the count of each type of silverware --/
structure SilverwareCount where
  knives : Int
  forks : Int
  spoons : Int
  teaspoons : Int

/-- Calculates the total count of silverware --/
def total_count (s : SilverwareCount) : Int :=
  s.knives + s.forks + s.spoons + s.teaspoons

/-- Represents a trade of silverware --/
structure Trade where
  give_knives : Int
  give_forks : Int
  give_spoons : Int
  give_teaspoons : Int
  receive_knives : Int
  receive_forks : Int
  receive_spoons : Int
  receive_teaspoons : Int

/-- Applies a trade to a silverware count --/
def apply_trade (s : SilverwareCount) (t : Trade) : SilverwareCount :=
  { knives := s.knives - t.give_knives + t.receive_knives,
    forks := s.forks - t.give_forks + t.receive_forks,
    spoons := s.spoons - t.give_spoons + t.receive_spoons,
    teaspoons := s.teaspoons - t.give_teaspoons + t.receive_teaspoons }

/-- Theorem representing Carolyn's silverware problem --/
theorem carolyns_silverware_percentage :
  let initial_count : SilverwareCount := { knives := 6, forks := 12, spoons := 18, teaspoons := 24 }
  let trade1 : Trade := { give_knives := 10, give_forks := 0, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 0, receive_teaspoons := 6 }
  let trade2 : Trade := { give_knives := 0, give_forks := 8, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 3, receive_teaspoons := 0 }
  let after_trades := apply_trade (apply_trade initial_count trade1) trade2
  let final_count := { after_trades with knives := after_trades.knives + 7 }
  (final_count.knives : Real) / (total_count final_count : Real) * 100 = 3 / 58 * 100 :=
by sorry

end NUMINAMATH_CALUDE_carolyns_silverware_percentage_l1949_194993


namespace NUMINAMATH_CALUDE_door_rod_equation_l1949_194950

theorem door_rod_equation (x : ℝ) : (x - 4)^2 + (x - 2)^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_door_rod_equation_l1949_194950


namespace NUMINAMATH_CALUDE_total_tv_time_l1949_194959

def missy_reality_shows : List ℕ := [28, 35, 42, 39, 29]
def missy_cartoons : ℕ := 2
def missy_cartoon_duration : ℕ := 10

def john_action_movies : List ℕ := [90, 110, 95]
def john_comedy_duration : ℕ := 25

def lily_documentaries : List ℕ := [45, 55, 60, 52]

def ad_breaks : List ℕ := [8, 6, 12, 9, 7, 11]

def num_viewers : ℕ := 3

theorem total_tv_time :
  (missy_reality_shows.sum + missy_cartoons * missy_cartoon_duration +
   john_action_movies.sum + john_comedy_duration +
   lily_documentaries.sum +
   num_viewers * ad_breaks.sum) = 884 := by
  sorry

end NUMINAMATH_CALUDE_total_tv_time_l1949_194959


namespace NUMINAMATH_CALUDE_probability_end_multiple_of_three_is_31_90_l1949_194923

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def probability_end_multiple_of_three : ℚ :=
  let total_cards := 10
  let prob_left := 1 / 3
  let prob_right := 2 / 3
  let prob_start_multiple_3 := 3 / 10
  let prob_start_one_more := 4 / 10
  let prob_start_one_less := 3 / 10
  let prob_end_multiple_3_from_multiple_3 := prob_left * prob_right + prob_right * prob_left
  let prob_end_multiple_3_from_one_more := prob_right * prob_right
  let prob_end_multiple_3_from_one_less := prob_left * prob_left
  prob_start_multiple_3 * prob_end_multiple_3_from_multiple_3 +
  prob_start_one_more * prob_end_multiple_3_from_one_more +
  prob_start_one_less * prob_end_multiple_3_from_one_less

theorem probability_end_multiple_of_three_is_31_90 :
  probability_end_multiple_of_three = 31 / 90 := by
  sorry

end NUMINAMATH_CALUDE_probability_end_multiple_of_three_is_31_90_l1949_194923


namespace NUMINAMATH_CALUDE_bread_baking_pattern_l1949_194901

/-- A sequence of bread loaves baked over 6 days -/
def BreadSequence : Type := Fin 6 → ℕ

/-- The condition that the daily increase grows by 1 each day -/
def IncreasingDifference (s : BreadSequence) : Prop :=
  ∀ i : Fin 4, s (i + 1) - s i < s (i + 2) - s (i + 1)

/-- The known values for days 1, 2, 3, 4, and 6 -/
def KnownValues (s : BreadSequence) : Prop :=
  s 0 = 5 ∧ s 1 = 7 ∧ s 2 = 10 ∧ s 3 = 14 ∧ s 5 = 25

theorem bread_baking_pattern (s : BreadSequence) 
  (h1 : IncreasingDifference s) (h2 : KnownValues s) : s 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_bread_baking_pattern_l1949_194901


namespace NUMINAMATH_CALUDE_six_people_reseating_l1949_194940

def reseating_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => reseating_ways (k + 2) + reseating_ways (k + 1)

theorem six_people_reseating :
  reseating_ways 6 = 13 := by sorry

end NUMINAMATH_CALUDE_six_people_reseating_l1949_194940


namespace NUMINAMATH_CALUDE_cars_clearing_time_l1949_194989

/-- Calculates the time for two cars to be clear of each other from the moment they meet -/
theorem cars_clearing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 20 := by
  sorry

#check cars_clearing_time

end NUMINAMATH_CALUDE_cars_clearing_time_l1949_194989


namespace NUMINAMATH_CALUDE_expression_evaluation_l1949_194907

theorem expression_evaluation : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1949_194907


namespace NUMINAMATH_CALUDE_minimum_employees_l1949_194939

theorem minimum_employees (work_days : ℕ) (rest_days : ℕ) (daily_requirement : ℕ) : 
  work_days = 5 →
  rest_days = 2 →
  daily_requirement = 32 →
  ∃ min_employees : ℕ,
    min_employees = (daily_requirement * 7 + work_days - 1) / work_days ∧
    min_employees * work_days ≥ daily_requirement * 7 ∧
    ∀ n : ℕ, n < min_employees → n * work_days < daily_requirement * 7 :=
by
  sorry

#eval (32 * 7 + 5 - 1) / 5  -- Should output 45

end NUMINAMATH_CALUDE_minimum_employees_l1949_194939


namespace NUMINAMATH_CALUDE_horse_race_probability_l1949_194910

theorem horse_race_probability (X Y Z : ℝ) 
  (no_draw : X + Y + Z = 1)
  (prob_X : X = 1/4)
  (prob_Y : Y = 3/5) : 
  Z = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_horse_race_probability_l1949_194910


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1949_194951

-- Problem 1
theorem problem_one : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt (1 + 1/3) / Real.sqrt (2 + 1/3) * Real.sqrt (1 + 2/5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1949_194951


namespace NUMINAMATH_CALUDE_xyz_product_l1949_194952

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1949_194952


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1949_194972

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y z w : ℝ, l₁ a x y ∧ l₂ a z w → (a = 2 * (a + 1))

-- Statement to prove
theorem sufficient_not_necessary (a : ℝ) :
  (a = 1 → parallel a) ∧ ¬(parallel a → a = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1949_194972


namespace NUMINAMATH_CALUDE_train_speed_l1949_194931

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 300 →
  crossing_time = 17.998560115190784 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.00468) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l1949_194931


namespace NUMINAMATH_CALUDE_misplaced_sheets_count_l1949_194949

/-- Represents a booklet of printed notes -/
structure Booklet where
  total_pages : ℕ
  total_sheets : ℕ
  misplaced_sheets : ℕ
  avg_remaining : ℝ

/-- The theorem stating the number of misplaced sheets -/
theorem misplaced_sheets_count (b : Booklet) 
  (h1 : b.total_pages = 60)
  (h2 : b.total_sheets = 30)
  (h3 : b.avg_remaining = 21) :
  b.misplaced_sheets = 15 := by
  sorry

#check misplaced_sheets_count

end NUMINAMATH_CALUDE_misplaced_sheets_count_l1949_194949


namespace NUMINAMATH_CALUDE_divisibility_by_24_l1949_194914

theorem divisibility_by_24 (n : Nat) : n ≤ 9 → (712 * 10 + n) % 24 = 0 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l1949_194914


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1949_194999

/-- A parallelogram with three known vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (1, 1)
  v2 : ℝ × ℝ := (2, 2)
  v3 : ℝ × ℝ := (3, -1)

/-- The possible fourth vertices of the parallelogram -/
def fourth_vertices : Set (ℝ × ℝ) :=
  {(2, -2), (4, 0)}

/-- Theorem stating that the fourth vertex of the parallelogram
    is one of the two possible points -/
theorem parallelogram_fourth_vertex (p : Parallelogram) :
  ∃ v4 : ℝ × ℝ, v4 ∈ fourth_vertices := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1949_194999


namespace NUMINAMATH_CALUDE_kelly_apple_count_l1949_194930

/-- 
Theorem: Given Kelly's initial apple count and the number of additional apples picked,
prove that the total number of apples is the sum of these two quantities.
-/
theorem kelly_apple_count (initial_apples additional_apples : ℕ) :
  initial_apples = 56 →
  additional_apples = 49 →
  initial_apples + additional_apples = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apple_count_l1949_194930


namespace NUMINAMATH_CALUDE_student_sister_weight_l1949_194918

/-- The combined weight of a student and his sister -/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 71 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 := by
sorry

end NUMINAMATH_CALUDE_student_sister_weight_l1949_194918


namespace NUMINAMATH_CALUDE_sequence_ratio_l1949_194985

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the difference of two terms in the arithmetic sequence 
    to a term in the geometric sequence is 1/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂) ∧ (a₂ - (-8 : ℝ) = a₁ - a₂) ∧  -- Arithmetic sequence condition
  (b₁ / (-2 : ℝ) = b₂ / b₁) ∧ (b₂ / b₁ = b₃ / b₂) ∧ (b₃ / b₂ = (-8 : ℝ) / b₃) →  -- Geometric sequence condition
  (a₂ - a₁) / b₂ = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1949_194985


namespace NUMINAMATH_CALUDE_uncle_welly_roses_l1949_194928

/-- Proves that Uncle Welly planted 20 more roses yesterday compared to two days ago -/
theorem uncle_welly_roses : 
  ∀ (roses_two_days_ago roses_yesterday roses_today : ℕ),
  roses_two_days_ago = 50 →
  roses_today = 2 * roses_two_days_ago →
  roses_yesterday > roses_two_days_ago →
  roses_two_days_ago + roses_yesterday + roses_today = 220 →
  roses_yesterday - roses_two_days_ago = 20 := by
sorry


end NUMINAMATH_CALUDE_uncle_welly_roses_l1949_194928


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1949_194934

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → (p.1^2 + p.2^2 = 26 * p.1)) ∧ S.card ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1949_194934


namespace NUMINAMATH_CALUDE_circle_ratio_l1949_194965

theorem circle_ratio (r R c d : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < c) (h4 : c < d) :
  (π * R^2) = (c / d) * (π * R^2 - π * r^2) →
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l1949_194965


namespace NUMINAMATH_CALUDE_percent_relation_l1949_194974

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l1949_194974


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1949_194964

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) *
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 9) =
  16 * Real.sqrt 105 / 315 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1949_194964


namespace NUMINAMATH_CALUDE_number_theory_problem_no_solution_2014_l1949_194987

theorem number_theory_problem (a x y : ℕ+) (h : x ≠ y) :
  a * x + Nat.gcd a x + Nat.lcm a x ≠ a * y + Nat.gcd a y + Nat.lcm a y :=
by sorry

theorem no_solution_2014 :
  ¬∃ (a b : ℕ+), a * b + Nat.gcd a b + Nat.lcm a b = 2014 :=
by sorry

end NUMINAMATH_CALUDE_number_theory_problem_no_solution_2014_l1949_194987


namespace NUMINAMATH_CALUDE_son_work_time_l1949_194922

def work_problem (man_time son_time combined_time : ℝ) : Prop :=
  man_time > 0 ∧ son_time > 0 ∧ combined_time > 0 ∧
  1 / man_time + 1 / son_time = 1 / combined_time

theorem son_work_time (man_time combined_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : combined_time = 4)
  : ∃ (son_time : ℝ), work_problem man_time son_time combined_time ∧ son_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l1949_194922


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1949_194980

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) →
  ∃ (k : ℤ), b = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1949_194980


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l1949_194971

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l1949_194971


namespace NUMINAMATH_CALUDE_perfect_cube_divisibility_l1949_194942

theorem perfect_cube_divisibility (a b : ℕ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : (a^3 + b^3 + a*b) % (a*b*(a-b)) = 0) : 
  ∃ (k : ℕ), a * b = k^3 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_divisibility_l1949_194942


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l1949_194994

theorem circular_garden_ratio : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference / area = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l1949_194994


namespace NUMINAMATH_CALUDE_D_72_l1949_194984

/-- D(n) is the number of ways to write n as a product of integers greater than 1, 
    considering the order of factors. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(72) = 97 -/
theorem D_72 : D 72 = 97 := by sorry

end NUMINAMATH_CALUDE_D_72_l1949_194984


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l1949_194968

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

theorem circle_center_coordinates :
  ∃ (r θ : ℝ), polar_equation r θ ∧ r = 1 ∧ θ = π / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l1949_194968


namespace NUMINAMATH_CALUDE_remainder_theorem_l1949_194963

theorem remainder_theorem (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1949_194963


namespace NUMINAMATH_CALUDE_direct_proportional_function_points_l1949_194979

/-- A direct proportional function passing through (2, -3) also passes through (4, -6) -/
theorem direct_proportional_function_points : ∃ (k : ℝ), k * 2 = -3 ∧ k * 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportional_function_points_l1949_194979


namespace NUMINAMATH_CALUDE_banana_groups_l1949_194924

theorem banana_groups (total_bananas : ℕ) (bananas_per_group : ℕ) 
  (h1 : total_bananas = 203) 
  (h2 : bananas_per_group = 29) : 
  (total_bananas / bananas_per_group : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l1949_194924


namespace NUMINAMATH_CALUDE_m_range_l1949_194921

def P (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1949_194921


namespace NUMINAMATH_CALUDE_snow_on_tuesday_l1949_194969

theorem snow_on_tuesday (monday_snow : ℝ) (total_snow : ℝ) (h1 : monday_snow = 0.32) (h2 : total_snow = 0.53) :
  total_snow - monday_snow = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_snow_on_tuesday_l1949_194969


namespace NUMINAMATH_CALUDE_probability_three_blue_six_trials_l1949_194948

/-- The probability of drawing exactly k blue marbles in n trials,
    given b blue marbles and r red marbles in a bag,
    where each draw is independent and the marble is replaced after each draw. -/
def probability_k_blue (n k b r : ℕ) : ℚ :=
  (n.choose k) * ((b : ℚ) / (b + r : ℚ))^k * ((r : ℚ) / (b + r : ℚ))^(n - k)

/-- The main theorem stating the probability of drawing exactly three blue marbles
    in six trials from a bag with 8 blue marbles and 6 red marbles. -/
theorem probability_three_blue_six_trials :
  probability_k_blue 6 3 8 6 = 34560 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_blue_six_trials_l1949_194948


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_35_l1949_194915

theorem closest_integer_to_sqrt_35 :
  ∀ x : ℝ, x = Real.sqrt 35 → (5 < x ∧ x < 6) → ∀ n : ℤ, |x - 6| ≤ |x - n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_35_l1949_194915


namespace NUMINAMATH_CALUDE_rose_garden_delivery_l1949_194903

theorem rose_garden_delivery (red yellow white : ℕ) : 
  red + yellow = 120 →
  red + white = 105 →
  yellow + white = 45 →
  red + yellow + white = 135 →
  (red = 90 ∧ white = 15 ∧ yellow = 30) := by
  sorry

end NUMINAMATH_CALUDE_rose_garden_delivery_l1949_194903


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l1949_194925

theorem parking_lot_wheels (num_cars num_bikes : ℕ) (wheels_per_car wheels_per_bike : ℕ) :
  num_cars = 14 →
  num_bikes = 5 →
  wheels_per_car = 4 →
  wheels_per_bike = 2 →
  num_cars * wheels_per_car + num_bikes * wheels_per_bike = 66 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l1949_194925


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1949_194957

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, Real.sqrt (2 * x + 6) - Real.sqrt (2 * x - 2) = 2 :=
by
  -- The unique solution is x = 1.5
  use (3/2)
  constructor
  · -- Prove that x = 1.5 satisfies the equation
    sorry
  · -- Prove that any solution must equal 1.5
    sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1949_194957


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1949_194953

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, ∀ x : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1949_194953
