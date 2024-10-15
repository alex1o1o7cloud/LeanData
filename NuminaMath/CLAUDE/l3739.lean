import Mathlib

namespace NUMINAMATH_CALUDE_lcm_of_given_numbers_l3739_373972

/-- The least common multiple of 360, 450, 560, 900, and 1176 is 176400. -/
theorem lcm_of_given_numbers : Nat.lcm 360 (Nat.lcm 450 (Nat.lcm 560 (Nat.lcm 900 1176))) = 176400 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_given_numbers_l3739_373972


namespace NUMINAMATH_CALUDE_min_value_expression_l3739_373911

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2)).sqrt) / (x * y * z) ≥ 4 ∧
  (∃ (a : ℝ), a > 0 ∧ (((a^2 + a^2 + a^2) * (4 * a^2 + a^2 + a^2)).sqrt) / (a * a * a) = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3739_373911


namespace NUMINAMATH_CALUDE_work_completion_time_l3739_373914

-- Define the work rates
def work_rate_a_and_b : ℚ := 1 / 6
def work_rate_a : ℚ := 1 / 11
def work_rate_c : ℚ := 1 / 13

-- Define the theorem
theorem work_completion_time :
  let work_rate_b : ℚ := work_rate_a_and_b - work_rate_a
  let work_rate_abc : ℚ := work_rate_a + work_rate_b + work_rate_c
  (1 : ℚ) / work_rate_abc = 858 / 209 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3739_373914


namespace NUMINAMATH_CALUDE_sector_properties_l3739_373999

/-- Properties of a circular sector --/
theorem sector_properties :
  -- Given a sector with central angle α and radius R
  ∀ (α R : ℝ),
  -- When α = π/3 and R = 10
  α = π / 3 ∧ R = 10 →
  -- The arc length is 10π/3
  (α * R = 10 * π / 3) ∧
  -- The area is 50π/3
  (1 / 2 * α * R^2 = 50 * π / 3) ∧
  -- For a sector with perimeter 12
  ∀ (r l : ℝ),
  (l + 2 * r = 12) →
  -- The maximum area is 9
  (1 / 2 * l * r ≤ 9) ∧
  -- The maximum area occurs when α = 2
  (1 / 2 * l * r = 9 → l / r = 2) := by
sorry

end NUMINAMATH_CALUDE_sector_properties_l3739_373999


namespace NUMINAMATH_CALUDE_sum_of_digits_of_even_numbers_up_to_12000_l3739_373990

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- Sum of digits of all even numbers in a sequence from 1 to n -/
def sumOfDigitsOfEvenNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all even numbers from 1 to 12000 is 129348 -/
theorem sum_of_digits_of_even_numbers_up_to_12000 :
  sumOfDigitsOfEvenNumbers 12000 = 129348 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_even_numbers_up_to_12000_l3739_373990


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l3739_373958

/-- Given a > 3, the function f(x) = x^2 - ax + 1 has exactly one zero point in the interval (0, 2) -/
theorem unique_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ x^2 - a*x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l3739_373958


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l3739_373988

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l3739_373988


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l3739_373991

/-- The number of boxes -/
def num_boxes : ℕ := 1005

/-- The probability of stopping at box n -/
def P (n : ℕ) : ℚ := 2 / (2 * n + 1)

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

theorem smallest_n_below_threshold :
  ∀ k : ℕ, k < num_boxes → P k ≥ threshold ∧
  P num_boxes < threshold ∧
  ∀ m : ℕ, m > num_boxes → P m < threshold :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l3739_373991


namespace NUMINAMATH_CALUDE_meetings_on_elliptical_track_l3739_373986

/-- Represents the elliptical track --/
structure EllipticalTrack where
  majorAxis : ℝ
  minorAxis : ℝ

/-- Represents a boy running on the track --/
structure Runner where
  speed : ℝ

/-- Calculates the number of meetings between two runners on an elliptical track --/
def numberOfMeetings (track : EllipticalTrack) (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem meetings_on_elliptical_track :
  let track := EllipticalTrack.mk 100 60
  let runner1 := Runner.mk 7
  let runner2 := Runner.mk 11
  numberOfMeetings track runner1 runner2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_meetings_on_elliptical_track_l3739_373986


namespace NUMINAMATH_CALUDE_two_part_number_problem_l3739_373965

theorem two_part_number_problem (x y k : ℕ) : 
  x + y = 24 →
  x = 13 →
  k * x + 5 * y = 146 →
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_part_number_problem_l3739_373965


namespace NUMINAMATH_CALUDE_tournament_equation_l3739_373908

/-- Represents a single round-robin tournament --/
structure Tournament where
  teams : ℕ
  games : ℕ

/-- The number of games in a single round-robin tournament --/
def Tournament.gameCount (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

/-- Theorem: In a single round-robin tournament with x teams and 28 games, 
    the equation (1/2)x(x-1) = 28 holds --/
theorem tournament_equation (t : Tournament) (h : t.games = 28) : 
  t.gameCount = 28 := by
  sorry

end NUMINAMATH_CALUDE_tournament_equation_l3739_373908


namespace NUMINAMATH_CALUDE_school_field_trip_buses_l3739_373952

/-- The number of buses needed for a school field trip --/
def buses_needed (fifth_graders sixth_graders seventh_graders : ℕ)
  (teachers_per_grade parents_per_grade : ℕ)
  (bus_capacity : ℕ) : ℕ :=
  let total_students := fifth_graders + sixth_graders + seventh_graders
  let total_chaperones := (teachers_per_grade + parents_per_grade) * 3
  let total_people := total_students + total_chaperones
  (total_people + bus_capacity - 1) / bus_capacity

theorem school_field_trip_buses :
  buses_needed 109 115 118 4 2 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_field_trip_buses_l3739_373952


namespace NUMINAMATH_CALUDE_R_when_S_is_9_l3739_373928

-- Define the equation R = gS - 6
def R (g S : ℚ) : ℚ := g * S - 6

-- State the theorem
theorem R_when_S_is_9 :
  ∀ g : ℚ, R g 7 = 18 → R g 9 = 174 / 7 := by
  sorry

end NUMINAMATH_CALUDE_R_when_S_is_9_l3739_373928


namespace NUMINAMATH_CALUDE_simplify_squared_terms_l3739_373936

theorem simplify_squared_terms (a : ℝ) : 2 * a^2 - 3 * a^2 = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_squared_terms_l3739_373936


namespace NUMINAMATH_CALUDE_amy_music_files_l3739_373903

theorem amy_music_files :
  ∀ (initial_music_files initial_video_files deleted_files remaining_files : ℕ),
    initial_video_files = 21 →
    deleted_files = 23 →
    remaining_files = 2 →
    initial_music_files + initial_video_files - deleted_files = remaining_files →
    initial_music_files = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l3739_373903


namespace NUMINAMATH_CALUDE_maple_to_pine_ratio_l3739_373940

theorem maple_to_pine_ratio (total_trees : ℕ) (oaks : ℕ) (firs : ℕ) (palms : ℕ) 
  (h1 : total_trees = 150)
  (h2 : oaks = 20)
  (h3 : firs = 35)
  (h4 : palms = 25)
  (h5 : ∃ (maple pine : ℕ), total_trees = oaks + firs + palms + maple + pine ∧ maple = 2 * pine) :
  ∃ (m p : ℕ), m / p = 2 ∧ m > 0 ∧ p > 0 := by
  sorry

end NUMINAMATH_CALUDE_maple_to_pine_ratio_l3739_373940


namespace NUMINAMATH_CALUDE_no_valid_operation_l3739_373909

-- Define the set of basic arithmetic operations
inductive BasicOperation
  | Add
  | Subtract
  | Multiply
  | Divide

-- Define a function to apply a basic operation
def applyOperation (op : BasicOperation) (a b : ℤ) : ℤ :=
  match op with
  | BasicOperation.Add => a + b
  | BasicOperation.Subtract => a - b
  | BasicOperation.Multiply => a * b
  | BasicOperation.Divide => a / b

-- Theorem statement
theorem no_valid_operation :
  ¬ ∃ (op : BasicOperation), (applyOperation op 8 2) + 5 - (3 - 2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_no_valid_operation_l3739_373909


namespace NUMINAMATH_CALUDE_birds_on_fence_l3739_373944

theorem birds_on_fence : ∃ (B : ℕ), ∃ (x : ℝ), 
  (Real.sqrt (B : ℝ) = x) ∧ 
  (2 * x^2 + 10 = 50) ∧ 
  (B = 20) := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3739_373944


namespace NUMINAMATH_CALUDE_picture_area_theorem_l3739_373943

theorem picture_area_theorem (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : 2*x*y + 9*x + 4*y + 12 = 60) : 
  x * y = 15 := by
sorry

end NUMINAMATH_CALUDE_picture_area_theorem_l3739_373943


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3739_373924

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem arithmetic_sequence_general_term 
  (seq : ArithmeticSequence)
  (h3 : seq.a 4 = 10)
  (h4 : IsGeometricSequence (seq.a 3) (seq.a 6) (seq.a 10)) :
  ∃ k : ℝ, ∀ n : ℕ, seq.a n = n + k ∧ k = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3739_373924


namespace NUMINAMATH_CALUDE_same_terminal_side_characterization_l3739_373938

def has_same_terminal_side (α : ℝ) : Prop :=
  ∃ k : ℤ, α = 30 + k * 360

theorem same_terminal_side_characterization (α : ℝ) :
  has_same_terminal_side α ↔ (α = -30 ∨ α = 390) :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_characterization_l3739_373938


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3739_373920

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 19)
  (eq2 : 3 * u + 5 * v = 1) : 
  u + v = 147 / 129 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3739_373920


namespace NUMINAMATH_CALUDE_cubic_root_implies_h_value_l3739_373954

theorem cubic_root_implies_h_value :
  ∀ h : ℝ, (3 : ℝ)^3 + h * 3 - 20 = 0 → h = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_h_value_l3739_373954


namespace NUMINAMATH_CALUDE_de_length_theorem_l3739_373923

-- Define the circle Ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points on the circle
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (Ω : Circle) (A B C D E : Point) : Prop :=
  -- A and B lie on circle Ω
  (A.x - Ω.center.1)^2 + (A.y - Ω.center.2)^2 = Ω.radius^2 ∧
  (B.x - Ω.center.1)^2 + (B.y - Ω.center.2)^2 = Ω.radius^2 ∧
  -- C and D are trisection points of major arc AB
  -- (This condition is simplified for the sake of the Lean statement)
  (C.x - Ω.center.1)^2 + (C.y - Ω.center.2)^2 = Ω.radius^2 ∧
  (D.x - Ω.center.1)^2 + (D.y - Ω.center.2)^2 = Ω.radius^2 ∧
  -- E is on line AB (simplified condition)
  (E.y - A.y) * (B.x - A.x) = (E.x - A.x) * (B.y - A.y) ∧
  -- Given distances
  ((D.x - C.x)^2 + (D.y - C.y)^2) = 64 ∧  -- DC = 8
  ((D.x - B.x)^2 + (D.y - B.y)^2) = 121   -- DB = 11

-- Main theorem
theorem de_length_theorem (Ω : Circle) (A B C D E : Point) 
  (h : problem_setup Ω A B C D E) :
  ∃ (a b : ℕ), (((E.x - D.x)^2 + (E.y - D.y)^2) = a^2 * b) ∧ 
  (∀ (p : ℕ), p^2 ∣ b → p = 1) → 
  a + b = 37 := by
  sorry

end NUMINAMATH_CALUDE_de_length_theorem_l3739_373923


namespace NUMINAMATH_CALUDE_goods_train_speed_l3739_373934

/-- The speed of a train given the speed of another train traveling in the opposite direction,
    the length of the first train, and the time it takes to pass an observer on the other train. -/
theorem goods_train_speed
  (speed_A : ℝ)
  (length_B : ℝ)
  (passing_time : ℝ)
  (h1 : speed_A = 100)  -- km/h
  (h2 : length_B = 0.28)  -- km (280 m converted to km)
  (h3 : passing_time = 9 / 3600)  -- hours (9 seconds converted to hours)
  : ∃ (speed_B : ℝ), speed_B = 12 := by
  sorry


end NUMINAMATH_CALUDE_goods_train_speed_l3739_373934


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3739_373949

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x ∈ Finset.range 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3739_373949


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3739_373939

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2) * (5*x^12 + 3*x^11 + 2*x^10 - x^9) = 
  15*x^13 - x^12 - 7*x^10 + 2*x^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3739_373939


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l3739_373926

def reverse_number (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n)
  List.foldl (fun acc d => 10 * acc + d) 0 digits

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem greatest_four_digit_number (m : ℕ) 
  (h1 : is_four_digit m)
  (h2 : m % 36 = 0)
  (h3 : m % 7 = 0)
  (h4 : (reverse_number m) % 36 = 0) :
  m ≤ 5796 ∧ ∃ (n : ℕ), n = 5796 ∧ 
    is_four_digit n ∧ 
    n % 36 = 0 ∧ 
    n % 7 = 0 ∧ 
    (reverse_number n) % 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l3739_373926


namespace NUMINAMATH_CALUDE_mayoral_race_vote_distribution_l3739_373922

theorem mayoral_race_vote_distribution (total_voters : ℝ) 
  (h1 : total_voters > 0) 
  (dem_percent : ℝ) 
  (h2 : dem_percent = 0.6) 
  (dem_vote_for_a : ℝ) 
  (h3 : dem_vote_for_a = 0.7) 
  (total_vote_for_a : ℝ) 
  (h4 : total_vote_for_a = 0.5) : 
  (total_vote_for_a * total_voters - dem_vote_for_a * dem_percent * total_voters) / 
  ((1 - dem_percent) * total_voters) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_mayoral_race_vote_distribution_l3739_373922


namespace NUMINAMATH_CALUDE_investors_in_securities_and_equities_l3739_373931

theorem investors_in_securities_and_equities 
  (total_investors : ℕ) 
  (investors_in_equities : ℕ) 
  (investors_in_both : ℕ) 
  (h1 : total_investors = 100)
  (h2 : investors_in_equities = 80)
  (h3 : investors_in_both = 25)
  (h4 : investors_in_both ≤ investors_in_equities)
  (h5 : investors_in_both ≤ total_investors) :
  investors_in_both = 25 := by
sorry

end NUMINAMATH_CALUDE_investors_in_securities_and_equities_l3739_373931


namespace NUMINAMATH_CALUDE_catch_up_theorem_l3739_373987

/-- The time it takes for me to walk from home to school in minutes -/
def my_walk_time : ℝ := 30

/-- The time it takes for my brother to walk from home to school in minutes -/
def brother_walk_time : ℝ := 40

/-- The time my brother left before me in minutes -/
def brother_head_start : ℝ := 5

/-- The time it takes for me to catch up with my brother in minutes -/
def catch_up_time : ℝ := 15

/-- Theorem stating that I will catch up with my brother after 15 minutes -/
theorem catch_up_theorem :
  let my_speed := 1 / my_walk_time
  let brother_speed := 1 / brother_walk_time
  let relative_speed := my_speed - brother_speed
  let head_start_distance := brother_speed * brother_head_start
  head_start_distance / relative_speed = catch_up_time := by
  sorry


end NUMINAMATH_CALUDE_catch_up_theorem_l3739_373987


namespace NUMINAMATH_CALUDE_residue_of_7_power_1234_mod_13_l3739_373980

theorem residue_of_7_power_1234_mod_13 : 7^1234 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_power_1234_mod_13_l3739_373980


namespace NUMINAMATH_CALUDE_pants_cost_l3739_373956

theorem pants_cost (initial_amount : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (money_left : ℕ) : 
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  money_left = 20 →
  initial_amount - (shirt_cost * num_shirts) - money_left = 26 :=
by sorry

end NUMINAMATH_CALUDE_pants_cost_l3739_373956


namespace NUMINAMATH_CALUDE_angle_range_l3739_373917

theorem angle_range (α : Real) (h1 : α ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin α < 0) (h3 : Real.cos α > 0) :
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_range_l3739_373917


namespace NUMINAMATH_CALUDE_quarters_in_eighth_l3739_373935

theorem quarters_in_eighth : (1 : ℚ) / 8 / ((1 : ℚ) / 4) = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarters_in_eighth_l3739_373935


namespace NUMINAMATH_CALUDE_unique_solution_system_l3739_373996

theorem unique_solution_system :
  ∃! (x y z : ℝ), 
    x * y + y * z + z * x = 1 ∧ 
    5 * x + 8 * y + 9 * z = 12 ∧
    x = 1 ∧ y = (1 : ℝ) / 2 ∧ z = (1 : ℝ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3739_373996


namespace NUMINAMATH_CALUDE_workshop_groups_l3739_373994

theorem workshop_groups (total_participants : ℕ) (max_group_size : ℕ) (h1 : total_participants = 36) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ), num_groups * max_group_size ≥ total_participants ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_participants → k ≥ num_groups :=
by sorry

end NUMINAMATH_CALUDE_workshop_groups_l3739_373994


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3739_373916

def U : Set ℝ := Set.univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≤ -1 ∨ x > 2}

theorem complement_intersection_theorem :
  (Set.compl B ∩ A) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3739_373916


namespace NUMINAMATH_CALUDE_rock_paper_scissors_lizard_spock_probability_l3739_373902

theorem rock_paper_scissors_lizard_spock_probability :
  let num_players : ℕ := 3
  let num_options : ℕ := 5
  let options_defeated_per_choice : ℕ := 2

  let prob_one_choice_defeats_another : ℚ := options_defeated_per_choice / num_options
  let prob_one_player_defeats_both_others : ℚ := prob_one_choice_defeats_another ^ 2
  let total_probability : ℚ := num_players * prob_one_player_defeats_both_others

  total_probability = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_lizard_spock_probability_l3739_373902


namespace NUMINAMATH_CALUDE_network_connections_l3739_373966

/-- Calculates the number of unique connections in a network of switches -/
def calculate_connections (num_switches : ℕ) (connections_per_switch : ℕ) : ℕ :=
  (num_switches * connections_per_switch) / 2

/-- Theorem: In a network of 30 switches, where each switch is connected to exactly 5 other switches,
    the total number of unique connections is 75. -/
theorem network_connections :
  calculate_connections 30 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l3739_373966


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3739_373963

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given two vectors a and b in R², where a = (-4, 3) and b = (6, m),
    if a is perpendicular to b, then m = 8 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-4, 3)
  let b : ℝ × ℝ := (6, m)
  perpendicular a b → m = 8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3739_373963


namespace NUMINAMATH_CALUDE_inverse_of_100_mod_101_l3739_373976

theorem inverse_of_100_mod_101 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_100_mod_101_l3739_373976


namespace NUMINAMATH_CALUDE_sqrt_product_l3739_373950

theorem sqrt_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l3739_373950


namespace NUMINAMATH_CALUDE_combined_stickers_l3739_373948

/-- The combined total of cat stickers for June and Bonnie after receiving gifts from their grandparents -/
theorem combined_stickers (june_initial : ℕ) (bonnie_initial : ℕ) (gift : ℕ) 
  (h1 : june_initial = 76)
  (h2 : bonnie_initial = 63)
  (h3 : gift = 25) :
  june_initial + bonnie_initial + 2 * gift = 189 := by
  sorry

#check combined_stickers

end NUMINAMATH_CALUDE_combined_stickers_l3739_373948


namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l3739_373915

-- Define the percentage of Democrat and Republican voters
def democrat_percentage : ℝ := 0.60
def republican_percentage : ℝ := 1 - democrat_percentage

-- Define the percentage of Democrats and Republicans voting for candidate A
def democrat_vote_for_a : ℝ := 0.85
def republican_vote_for_a : ℝ := 0.20

-- Define the theorem
theorem expected_votes_for_candidate_a :
  democrat_percentage * democrat_vote_for_a + republican_percentage * republican_vote_for_a = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l3739_373915


namespace NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3739_373964

/-- The size of the multiplication table -/
def table_size : ℕ := 15

/-- The count of odd numbers from 1 to table_size -/
def odd_count : ℕ := (table_size + 1) / 2

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of odd entries in the multiplication table -/
def odd_entries : ℕ := odd_count * odd_count

/-- The fraction of odd numbers in the multiplication table -/
def odd_fraction : ℚ := odd_entries / total_entries

theorem odd_fraction_in_multiplication_table :
  odd_fraction = 64 / 225 := by
  sorry

end NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3739_373964


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l3739_373997

theorem smaller_root_of_equation : 
  let f (x : ℚ) := (x - 3/5) * (x - 3/5) + 2 * (x - 3/5) * (x - 1/3)
  ∃ r : ℚ, f r = 0 ∧ r = 19/45 ∧ ∀ s : ℚ, f s = 0 → s ≠ r → r < s :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l3739_373997


namespace NUMINAMATH_CALUDE_days_without_calls_l3739_373993

/-- Represents the number of days in the year -/
def total_days : ℕ := 366

/-- Represents the frequency of calls for each niece -/
def call_frequencies : List ℕ := [2, 3, 4]

/-- Calculates the number of days with at least one call -/
def days_with_calls (frequencies : List ℕ) (total : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of days without calls -/
theorem days_without_calls :
  total_days - days_with_calls call_frequencies total_days = 122 :=
sorry

end NUMINAMATH_CALUDE_days_without_calls_l3739_373993


namespace NUMINAMATH_CALUDE_joe_monthly_income_correct_l3739_373981

/-- Joe's monthly income in dollars -/
def monthly_income : ℝ := 2120

/-- The fraction of Joe's income that goes to taxes -/
def tax_rate : ℝ := 0.4

/-- The amount Joe pays in taxes each month in dollars -/
def monthly_tax : ℝ := 848

/-- Theorem stating that Joe's monthly income is correct given the tax rate and monthly tax amount -/
theorem joe_monthly_income_correct : 
  tax_rate * monthly_income = monthly_tax := by sorry

end NUMINAMATH_CALUDE_joe_monthly_income_correct_l3739_373981


namespace NUMINAMATH_CALUDE_power_function_problem_l3739_373961

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x > 0, f x = x^α

-- State the theorem
theorem power_function_problem (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 9 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_problem_l3739_373961


namespace NUMINAMATH_CALUDE_data_transformation_theorem_l3739_373989

variable {α : Type*} [LinearOrderedField α]

def average (data : Finset α) (f : α → α) : α :=
  (data.sum f) / data.card

def variance (data : Finset α) (f : α → α) (μ : α) : α :=
  (data.sum (fun x => (f x - μ) ^ 2)) / data.card

theorem data_transformation_theorem (data : Finset α) (f : α → α) :
  (average data (fun x => f x - 80) = 1.2) →
  (variance data (fun x => f x - 80) 1.2 = 4.4) →
  (average data f = 81.2) ∧ (variance data f 81.2 = 4.4) := by
  sorry

end NUMINAMATH_CALUDE_data_transformation_theorem_l3739_373989


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l3739_373925

def length_increase (n : ℕ) : ℚ :=
  (n + 3 : ℚ) / 3

theorem barry_sotter_magic (n : ℕ) : length_increase n = 50 → n = 147 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l3739_373925


namespace NUMINAMATH_CALUDE_committee_meeting_pencils_committee_meeting_pencils_correct_l3739_373941

/-- The number of pencils brought to a committee meeting -/
theorem committee_meeting_pencils : ℕ :=
  let associate_prof : ℕ := 2  -- number of associate professors
  let assistant_prof : ℕ := 7  -- number of assistant professors
  let total_people : ℕ := 9    -- total number of people present
  let total_charts : ℕ := 16   -- total number of charts brought

  -- Each associate professor brings 2 pencils and 1 chart
  -- Each assistant professor brings 1 pencil and 2 charts
  -- The total number of pencils is what we want to prove
  have h1 : associate_prof + assistant_prof = total_people := by sorry
  have h2 : associate_prof + 2 * assistant_prof = total_charts := by sorry
  
  11

theorem committee_meeting_pencils_correct : committee_meeting_pencils = 11 := by sorry

end NUMINAMATH_CALUDE_committee_meeting_pencils_committee_meeting_pencils_correct_l3739_373941


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l3739_373921

/-- The least possible side length of a square when measured as 5 cm to the nearest centimeter -/
def least_side_length : ℝ := 4.5

/-- The reported measurement of the square's side length in centimeters -/
def reported_length : ℕ := 5

/-- Theorem: The least possible area of a square with sides measured as 5 cm to the nearest centimeter is 20.25 cm² -/
theorem least_possible_area_of_square (side : ℝ) 
    (h1 : side ≥ least_side_length) 
    (h2 : side < least_side_length + 1) 
    (h3 : ⌊side⌋ = reported_length ∨ ⌈side⌉ = reported_length) : 
  least_side_length ^ 2 ≤ side ^ 2 :=
sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l3739_373921


namespace NUMINAMATH_CALUDE_bowling_score_problem_l3739_373929

theorem bowling_score_problem (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 50 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score = 135 := by
sorry

end NUMINAMATH_CALUDE_bowling_score_problem_l3739_373929


namespace NUMINAMATH_CALUDE_negation_equivalence_l3739_373970

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3739_373970


namespace NUMINAMATH_CALUDE_largest_root_bound_l3739_373942

theorem largest_root_bound (b₀ b₁ b₂ b₃ : ℝ) (h₀ : |b₀| ≤ 1) (h₁ : |b₁| ≤ 1) (h₂ : |b₂| ≤ 1) (h₃ : |b₃| ≤ 1) :
  ∃ r : ℝ, (5/2 < r ∧ r < 3) ∧
    (∀ x : ℝ, x > 0 → x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_bound_l3739_373942


namespace NUMINAMATH_CALUDE_new_person_weight_is_68_l3739_373905

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 68 kg -/
theorem new_person_weight_is_68 :
  new_person_weight 6 3.5 47 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_68_l3739_373905


namespace NUMINAMATH_CALUDE_line_equation_l3739_373969

/-- A line passing through (2, 3) with slope -1 has equation x + y - 5 = 0 -/
theorem line_equation (x y : ℝ) :
  (∀ t : ℝ, x = 2 - t ∧ y = 3 + t) →  -- Parametric form of line through (2, 3) with slope -1
  x + y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3739_373969


namespace NUMINAMATH_CALUDE_max_siskins_achievable_24_siskins_l3739_373975

/-- Represents a row of poles with siskins -/
structure PoleRow :=
  (num_poles : Nat)
  (occupied : Finset Nat)

/-- The rules for siskin movement on poles -/
def valid_configuration (pr : PoleRow) : Prop :=
  pr.num_poles = 25 ∧
  pr.occupied.card ≤ pr.num_poles ∧
  ∀ i ∈ pr.occupied, i ≤ pr.num_poles ∧
  ∀ i ∈ pr.occupied, (i + 1 ∉ pr.occupied ∨ i = pr.num_poles) ∧
                     (i - 1 ∉ pr.occupied ∨ i = 1)

/-- The theorem stating the maximum number of siskins -/
theorem max_siskins (pr : PoleRow) :
  valid_configuration pr → pr.occupied.card ≤ 24 :=
sorry

/-- The theorem stating that 24 siskins is achievable -/
theorem achievable_24_siskins :
  ∃ pr : PoleRow, valid_configuration pr ∧ pr.occupied.card = 24 :=
sorry

end NUMINAMATH_CALUDE_max_siskins_achievable_24_siskins_l3739_373975


namespace NUMINAMATH_CALUDE_final_quantity_of_B_l3739_373951

/-- Represents the quantity of each product type -/
structure ProductQuantities where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the products -/
def totalCost (q : ProductQuantities) : ℕ :=
  2 * q.a + 3 * q.b + 5 * q.c

/-- Represents the problem constraints -/
structure ProblemConstraints where
  initial : ProductQuantities
  final : ProductQuantities
  initialCost : totalCost initial = 20
  finalCost : totalCost final = 20
  returnedTwoItems : initial.a + initial.b + initial.c = final.a + final.b + final.c + 2
  atLeastOne : final.a ≥ 1 ∧ final.b ≥ 1 ∧ final.c ≥ 1

theorem final_quantity_of_B (constraints : ProblemConstraints) : constraints.final.b = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_quantity_of_B_l3739_373951


namespace NUMINAMATH_CALUDE_grid_division_l3739_373945

theorem grid_division (n : ℕ) : 
  (∃ m : ℕ, n^2 = 7 * m ∧ m > 0) ↔ (n > 7 ∧ ∃ k : ℕ, n = 7 * k) := by
sorry

end NUMINAMATH_CALUDE_grid_division_l3739_373945


namespace NUMINAMATH_CALUDE_greatest_three_digit_special_number_l3739_373960

/-- A number is a three-digit number if it's between 100 and 999, inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is one more than a multiple of 9 if it leaves a remainder of 1 when divided by 9 -/
def is_one_more_than_multiple_of_9 (n : ℕ) : Prop := n % 9 = 1

/-- A number is three more than a multiple of 5 if it leaves a remainder of 3 when divided by 5 -/
def is_three_more_than_multiple_of_5 (n : ℕ) : Prop := n % 5 = 3

theorem greatest_three_digit_special_number : 
  is_three_digit 973 ∧ 
  is_one_more_than_multiple_of_9 973 ∧ 
  is_three_more_than_multiple_of_5 973 ∧ 
  ∀ m : ℕ, (is_three_digit m ∧ 
            is_one_more_than_multiple_of_9 m ∧ 
            is_three_more_than_multiple_of_5 m) → 
           m ≤ 973 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_special_number_l3739_373960


namespace NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_l3739_373983

theorem proposition_a_necessary_not_sufficient :
  (∀ a b : ℝ, a > b ∧ a⁻¹ > b⁻¹ → a > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ a > b ∧ a⁻¹ ≤ b⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_l3739_373983


namespace NUMINAMATH_CALUDE_circle_F_value_l3739_373918

-- Define the circle equation
def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_F_value :
  ∃ F : ℝ, (∀ x y : ℝ, circle_equation x y F ↔ (x - 1)^2 + (y + 1)^2 = circle_radius^2) ∧ F = -2 :=
sorry

end NUMINAMATH_CALUDE_circle_F_value_l3739_373918


namespace NUMINAMATH_CALUDE_smaller_tv_diagonal_l3739_373998

theorem smaller_tv_diagonal (d : ℝ) : 
  d > 0 → 
  (28 / Real.sqrt 2)^2 = (d / Real.sqrt 2)^2 + 79.5 → 
  d = 25 := by
sorry

end NUMINAMATH_CALUDE_smaller_tv_diagonal_l3739_373998


namespace NUMINAMATH_CALUDE_first_group_size_l3739_373967

/-- Represents the rate of work in meters of cloth colored per man per day -/
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℚ :=
  (length : ℚ) / ((men : ℚ) * (days : ℚ))

theorem first_group_size (m : ℕ) : 
  rate_of_work m 48 2 = rate_of_work 2 36 3 → m = 4 := by
  sorry

#check first_group_size

end NUMINAMATH_CALUDE_first_group_size_l3739_373967


namespace NUMINAMATH_CALUDE_riddle_ratio_l3739_373968

theorem riddle_ratio (josh_riddles ivory_riddles taso_riddles : ℕ) : 
  josh_riddles = 8 →
  ivory_riddles = josh_riddles + 4 →
  taso_riddles = 24 →
  (taso_riddles : ℚ) / ivory_riddles = 2 := by
  sorry

end NUMINAMATH_CALUDE_riddle_ratio_l3739_373968


namespace NUMINAMATH_CALUDE_exp_sum_equals_ten_l3739_373992

theorem exp_sum_equals_ten (a b : ℝ) (h1 : Real.log 3 = a) (h2 : Real.log 7 = b) :
  Real.exp a + Real.exp b = 10 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_equals_ten_l3739_373992


namespace NUMINAMATH_CALUDE_f_always_negative_f_less_than_2x_minus_3_l3739_373930

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x - 1

-- Part 1: Range of a for which f(x) < 0 for all x
theorem f_always_negative (a : ℝ) : 
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution to f(x) < 2x - 3 when a > 0
theorem f_less_than_2x_minus_3 (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < 2 * x - 3} = 
    if a < 2 then Set.Ioo 1 (2/a)
    else if a = 2 then ∅ 
    else Set.Ioo (2/a) 1 :=
sorry

end NUMINAMATH_CALUDE_f_always_negative_f_less_than_2x_minus_3_l3739_373930


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l3739_373946

-- Define the two lines
def line1 (x y : ℝ) : Prop := 6 * x - 8 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define what it means for a point to be between two lines
def between (x y : ℝ) : Prop :=
  ∃ (y1 y2 : ℝ), line1 x y1 ∧ line2 x y2 ∧ ((y1 < y ∧ y < y2) ∨ (y2 < y ∧ y < y1))

-- Theorem statement
theorem point_between_parallel_lines :
  ∀ b : ℤ, between 5 (b : ℝ) → b = 4 := by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l3739_373946


namespace NUMINAMATH_CALUDE_total_amount_raised_l3739_373962

/-- Represents the sizes of rubber ducks --/
inductive DuckSize
  | Small
  | Medium
  | Large

/-- Calculates the price of a duck given its size --/
def price (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 2
  | DuckSize.Medium => 3
  | DuckSize.Large => 5

/-- Calculates the bulk discount rate for a given size and quantity --/
def bulkDiscountRate (s : DuckSize) (quantity : ℕ) : ℚ :=
  match s with
  | DuckSize.Small => if quantity ≥ 10 then 0.1 else 0
  | DuckSize.Medium => if quantity ≥ 15 then 0.15 else 0
  | DuckSize.Large => if quantity ≥ 20 then 0.2 else 0

/-- Returns the sales tax rate for a given duck size --/
def salesTaxRate (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 0.05
  | DuckSize.Medium => 0.07
  | DuckSize.Large => 0.09

/-- Calculates the total amount raised for a given duck size and quantity --/
def amountRaised (s : DuckSize) (quantity : ℕ) : ℚ :=
  let basePrice := price s * quantity
  let discountedPrice := basePrice * (1 - bulkDiscountRate s quantity)
  discountedPrice * (1 + salesTaxRate s)

/-- Theorem stating the total amount raised for charity --/
theorem total_amount_raised :
  amountRaised DuckSize.Small 150 +
  amountRaised DuckSize.Medium 221 +
  amountRaised DuckSize.Large 185 = 1693.1 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_raised_l3739_373962


namespace NUMINAMATH_CALUDE_positive_root_range_l3739_373933

-- Define the function f(x) = mx² - 3x + 1
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 3 * x + 1

-- Theorem statement
theorem positive_root_range (m : ℝ) :
  (∃ x > 0, f m x = 0) ↔ m ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_positive_root_range_l3739_373933


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3739_373978

theorem complex_modulus_problem (z : ℂ) : 
  ((1 - Complex.I) / Complex.I) * z = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3739_373978


namespace NUMINAMATH_CALUDE_consecutive_integers_product_210_l3739_373995

theorem consecutive_integers_product_210 (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 210) → (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_210_l3739_373995


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l3739_373959

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l3739_373959


namespace NUMINAMATH_CALUDE_set_range_proof_l3739_373927

theorem set_range_proof (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ensuring the order of numbers
  a = 2 ∧  -- Smallest number is 2
  b = 5 ∧  -- Median is 5
  (a + b + c) / 3 = 5 →  -- Mean is 5
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_set_range_proof_l3739_373927


namespace NUMINAMATH_CALUDE_smallest_common_pet_count_l3739_373985

theorem smallest_common_pet_count : ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 15 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 15 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_pet_count_l3739_373985


namespace NUMINAMATH_CALUDE_tims_car_value_l3739_373977

/-- Represents the value of a car over time -/
def car_value (initial_value : ℕ) (depreciation_rate : ℕ) (years : ℕ) : ℕ :=
  initial_value - depreciation_rate * years

/-- Theorem stating the value of Tim's car after 6 years -/
theorem tims_car_value : car_value 20000 1000 6 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_tims_car_value_l3739_373977


namespace NUMINAMATH_CALUDE_alternate_seating_count_l3739_373953

/-- The number of ways to seat 4 boys and 1 girl alternately in a row -/
def alternateSeating : ℕ :=
  let numBoys : ℕ := 4
  let numGirls : ℕ := 1
  let numPositionsForGirl : ℕ := numBoys + 1
  let numArrangementsForBoys : ℕ := Nat.factorial numBoys
  numPositionsForGirl * numArrangementsForBoys

theorem alternate_seating_count : alternateSeating = 120 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_count_l3739_373953


namespace NUMINAMATH_CALUDE_juice_consumption_l3739_373974

theorem juice_consumption (refrigerator pantry bought left : ℕ) 
  (h1 : refrigerator = 4)
  (h2 : pantry = 4)
  (h3 : bought = 5)
  (h4 : left = 10) :
  refrigerator + pantry + bought - left = 3 := by
  sorry

end NUMINAMATH_CALUDE_juice_consumption_l3739_373974


namespace NUMINAMATH_CALUDE_mortgage_payment_months_l3739_373910

theorem mortgage_payment_months (first_payment : ℝ) (ratio : ℝ) (total_amount : ℝ) : 
  first_payment = 100 →
  ratio = 3 →
  total_amount = 2952400 →
  (∃ n : ℕ, first_payment * (1 - ratio^n) / (1 - ratio) = total_amount ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_mortgage_payment_months_l3739_373910


namespace NUMINAMATH_CALUDE_negation_equivalence_l3739_373982

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Doctor : U → Prop)
variable (ExcellentCook : U → Prop)
variable (PoorCook : U → Prop)

-- Define the statements
def AllDoctorsExcellentCooks : Prop := ∀ x, Doctor x → ExcellentCook x
def AtLeastOneDoctorPoorCook : Prop := ∃ x, Doctor x ∧ PoorCook x

-- Theorem to prove
theorem negation_equivalence :
  AtLeastOneDoctorPoorCook U Doctor PoorCook ↔ 
  ¬(AllDoctorsExcellentCooks U Doctor ExcellentCook) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3739_373982


namespace NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3739_373913

/-- Represents the mathematical concepts that could be used in optimization methods -/
inductive OptimizationConcept
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- Represents Hua Luogeng's optimal selection method -/
def HuaOptimalSelectionMethod : Type := OptimizationConcept

/-- The concept used in Hua Luogeng's optimal selection method -/
def concept_used : HuaOptimalSelectionMethod := OptimizationConcept.GoldenRatio

/-- Theorem stating that the concept used in Hua Luogeng's optimal selection method is the golden ratio -/
theorem hua_method_uses_golden_ratio :
  concept_used = OptimizationConcept.GoldenRatio :=
by sorry

end NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3739_373913


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3739_373932

theorem triangle_inequalities (a b c s : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_s : s = a + b + c) : 
  ((13 / 27 : ℝ) * s^2 ≤ a^2 + b^2 + c^2 + 4 / s * a * b * c ∧ 
   a^2 + b^2 + c^2 + 4 / s * a * b * c < s^2 / 2) ∧
  (s^2 / 4 < a * b + b * c + c * a - 2 / s * a * b * c ∧ 
   a * b + b * c + c * a - 2 / s * a * b * c < (7 / 27 : ℝ) * s^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3739_373932


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_l3739_373973

theorem opposite_numbers_solution (x y a : ℝ) : 
  x + 2*y = 2*a - 1 →
  x - y = 6 →
  x = -y →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_l3739_373973


namespace NUMINAMATH_CALUDE_triangle_area_l3739_373957

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → a = 2 * c → B = π / 3 → 
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3739_373957


namespace NUMINAMATH_CALUDE_log_weight_after_cutting_l3739_373971

/-- Given a log of length 20 feet that is cut in half, where each linear foot weighs 150 pounds,
    prove that each cut piece weighs 1500 pounds. -/
theorem log_weight_after_cutting (original_length : ℝ) (weight_per_foot : ℝ) :
  original_length = 20 →
  weight_per_foot = 150 →
  (original_length / 2) * weight_per_foot = 1500 := by
  sorry

#check log_weight_after_cutting

end NUMINAMATH_CALUDE_log_weight_after_cutting_l3739_373971


namespace NUMINAMATH_CALUDE_triangle_sides_from_perimeters_l3739_373912

/-- Given the perimeters of three figures formed by two identical squares and two identical triangles,
    prove that the lengths of the sides of the triangle are 5, 12, and 10. -/
theorem triangle_sides_from_perimeters (p1 p2 p3 : ℕ) 
  (h1 : p1 = 74) (h2 : p2 = 84) (h3 : p3 = 82) : 
  ∃ (a b c : ℕ), a = 5 ∧ b = 12 ∧ c = 10 ∧ 
  (∃ (s : ℕ), 2 * s + a + b + c = p1) ∧
  (∃ (s : ℕ), 2 * s + a + b + c + 2 * a = p2) ∧
  (∃ (s : ℕ), 2 * s + 2 * b + 2 * a = p3) :=
by sorry


end NUMINAMATH_CALUDE_triangle_sides_from_perimeters_l3739_373912


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element6_l3739_373947

theorem pascal_triangle_row20_element6 : Nat.choose 20 5 = 7752 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element6_l3739_373947


namespace NUMINAMATH_CALUDE_largest_value_proof_l3739_373900

theorem largest_value_proof (a b c : ℚ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : b < 1) (h5 : a > b) : 
  (max (-1/(a*b)) (max (1/b) (max |a*c| (max (1/b^2) (1/a^2))))) = 1/b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_proof_l3739_373900


namespace NUMINAMATH_CALUDE_number_difference_l3739_373919

/-- Given two positive integers where the larger number is 1596,
    and when divided by the smaller number results in a quotient of 6 and a remainder of 15,
    prove that the difference between these two numbers is equal to the calculated difference. -/
theorem number_difference (smaller larger : ℕ) (h1 : larger = 1596) 
    (h2 : larger = 6 * smaller + 15) : larger - smaller = larger - smaller := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3739_373919


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3739_373984

theorem trigonometric_identity (θ : Real) 
  (h : Real.sin (π / 3 - θ) = 1 / 2) : 
  Real.cos (π / 6 + θ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3739_373984


namespace NUMINAMATH_CALUDE_adam_total_score_l3739_373901

/-- Calculates the total points scored in a game given points per round and number of rounds -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that given 71 points per round and 4 rounds, the total points is 284 -/
theorem adam_total_score : totalPoints 71 4 = 284 := by
  sorry

end NUMINAMATH_CALUDE_adam_total_score_l3739_373901


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l3739_373907

def g (x : ℝ) : ℝ := -3 * x^3 + 12

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l3739_373907


namespace NUMINAMATH_CALUDE_simplify_expressions_l3739_373937

theorem simplify_expressions :
  (- (99 + 71 / 72) * 36 = - (3599 + 1 / 2)) ∧
  ((-3) * (1 / 4) - 2.5 * (-2.45) + (3 + 1 / 2) * (25 / 100) = 6 + 1 / 4) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3739_373937


namespace NUMINAMATH_CALUDE_right_triangle_has_three_altitudes_l3739_373906

/-- A triangle is a geometric figure with three vertices and three sides. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side or its extension. -/
def Altitude (t : Triangle) (v : Fin 3) : Set (ℝ × ℝ) :=
  sorry

/-- A right triangle is a triangle with one right angle. -/
def IsRightTriangle (t : Triangle) : Prop :=
  sorry

/-- The number of altitudes in a triangle. -/
def NumberOfAltitudes (t : Triangle) : ℕ :=
  sorry

/-- Theorem: A right triangle has three altitudes. -/
theorem right_triangle_has_three_altitudes (t : Triangle) :
  IsRightTriangle t → NumberOfAltitudes t = 3 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_has_three_altitudes_l3739_373906


namespace NUMINAMATH_CALUDE_triangle_property_l3739_373904

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB = (1/2)b + c, then A = 2π/3 and (b^2 + c^2 + bc) / (4R^2) = 3/4,
    where R is the radius of the circumcircle of triangle ABC -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B = (1/2) * b + c →
  R > 0 →
  A = 2 * π / 3 ∧ (b^2 + c^2 + b*c) / (4 * R^2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_triangle_property_l3739_373904


namespace NUMINAMATH_CALUDE_complete_square_min_value_m_greater_n_right_triangle_l3739_373979

-- 1. Complete the square
theorem complete_square (x : ℝ) : x^2 - 4*x + 5 = (x - 2)^2 + 1 := by sorry

-- 2. Minimum value
theorem min_value : ∃ (m : ℝ), ∀ (x : ℝ), x^2 - 2*x + 3 ≥ m ∧ ∃ (y : ℝ), y^2 - 2*y + 3 = m := by sorry

-- 3. Relationship between M and N
theorem m_greater_n (a : ℝ) : a^2 - a > a - 2 := by sorry

-- 4. Triangle shape
theorem right_triangle (a b c : ℝ) : 
  a^2 + b^2 + c^2 - 6*a - 10*b - 8*c + 50 = 0 → 
  a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_complete_square_min_value_m_greater_n_right_triangle_l3739_373979


namespace NUMINAMATH_CALUDE_evaluate_expression_l3739_373955

theorem evaluate_expression : 
  Real.sqrt ((16^10 + 4^15) / (16^7 + 4^16 - 4^8)) = 2 * Real.sqrt 1025 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3739_373955
