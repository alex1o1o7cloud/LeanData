import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identity_l185_18563

theorem trigonometric_identity : 
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l185_18563


namespace NUMINAMATH_CALUDE_expansion_equals_fifth_power_l185_18580

theorem expansion_equals_fifth_power (y : ℝ) : 
  (y - 1)^5 + 5*(y - 1)^4 + 10*(y - 1)^3 + 10*(y - 1)^2 + 5*(y - 1) + 1 = y^5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fifth_power_l185_18580


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l185_18582

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define parallelism between two planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    p1.normal.x = k * p2.normal.x ∧
    p1.normal.y = k * p2.normal.y ∧
    p1.normal.z = k * p2.normal.z

-- State the theorem
theorem perpendicular_planes_parallel (l : Line3D) (p1 p2 : Plane3D) :
  perpendicular l p1 → perpendicular l p2 → parallel p1 p2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l185_18582


namespace NUMINAMATH_CALUDE_tangent_line_equality_l185_18566

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Define the derivatives of f and g
def f' (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b
def g' (x : ℝ) : ℝ := 2*x - 3

-- State the theorem
theorem tangent_line_equality (a b : ℝ) :
  f a b 2 = g 2 ∧ f' a b 2 = g' 2 →
  a = -2 ∧ b = 5 ∧ ∀ x y, y = x - 2 ↔ x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l185_18566


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l185_18515

theorem vasya_driving_distance
  (total_distance : ℝ)
  (anton_distance : ℝ)
  (vasya_distance : ℝ)
  (sasha_distance : ℝ)
  (dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance) :
  vasya_distance = (2 : ℝ) / 5 * total_distance :=
by sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l185_18515


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l185_18521

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l185_18521


namespace NUMINAMATH_CALUDE_original_average_calc_l185_18592

/-- Given a set of 10 numbers, if increasing one number by 6 changes the average to 6.8,
    then the original average was 6.2 -/
theorem original_average_calc (S : Finset ℝ) (original_sum : ℝ) :
  Finset.card S = 10 →
  (original_sum + 6) / 10 = 6.8 →
  original_sum / 10 = 6.2 :=
by sorry

end NUMINAMATH_CALUDE_original_average_calc_l185_18592


namespace NUMINAMATH_CALUDE_min_members_in_association_l185_18573

/-- Represents an association with men and women members -/
structure Association where
  men : ℕ
  women : ℕ

/-- Calculates the total number of members in the association -/
def Association.totalMembers (a : Association) : ℕ := a.men + a.women

/-- Calculates the number of homeowners in the association -/
def Association.homeowners (a : Association) : ℚ := 0.1 * a.men + 0.2 * a.women

/-- Theorem stating the minimum number of members in the association -/
theorem min_members_in_association :
  ∃ (a : Association), a.homeowners ≥ 18 ∧
  (∀ (b : Association), b.homeowners ≥ 18 → a.totalMembers ≤ b.totalMembers) ∧
  a.totalMembers = 91 := by
  sorry

end NUMINAMATH_CALUDE_min_members_in_association_l185_18573


namespace NUMINAMATH_CALUDE_square_of_105_l185_18570

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l185_18570


namespace NUMINAMATH_CALUDE_inequality_solution_l185_18542

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / ((x - 3)^2) < 0 ↔ -3 < x ∧ x < 3 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l185_18542


namespace NUMINAMATH_CALUDE_james_touchdown_points_l185_18554

/-- The number of points per touchdown in James' football season -/
def points_per_touchdown : ℕ := by sorry

/-- The number of touchdowns James scores per game -/
def touchdowns_per_game : ℕ := 4

/-- The number of games in the season -/
def games_in_season : ℕ := 15

/-- The number of 2-point conversions James scores in the season -/
def two_point_conversions : ℕ := 6

/-- The total points James scores in the season -/
def total_points : ℕ := 372

theorem james_touchdown_points :
  points_per_touchdown * touchdowns_per_game * games_in_season +
  2 * two_point_conversions = total_points ∧
  points_per_touchdown = 6 := by sorry

end NUMINAMATH_CALUDE_james_touchdown_points_l185_18554


namespace NUMINAMATH_CALUDE_division_remainder_problem_l185_18517

theorem division_remainder_problem (N : ℕ) 
  (h1 : N / 8 = 8) 
  (h2 : N % 5 = 4) : 
  N % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l185_18517


namespace NUMINAMATH_CALUDE_earnings_per_dog_l185_18586

def dogs_monday_wednesday_friday : ℕ := 7
def dogs_tuesday : ℕ := 12
def dogs_thursday : ℕ := 9
def weekly_earnings : ℕ := 210

def total_dogs : ℕ := dogs_monday_wednesday_friday * 3 + dogs_tuesday + dogs_thursday

theorem earnings_per_dog :
  weekly_earnings / total_dogs = 5 := by sorry

end NUMINAMATH_CALUDE_earnings_per_dog_l185_18586


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l185_18533

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_draw = 0.4) : 
  p_win + p_draw = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l185_18533


namespace NUMINAMATH_CALUDE_father_ate_chocolates_father_ate_two_chocolates_l185_18511

theorem father_ate_chocolates (total_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (father_left : ℕ) : ℕ :=
  let num_people := num_sisters + 1
  let chocolates_per_person := total_chocolates / num_people
  let given_to_father := num_people * (chocolates_per_person / 2)
  let father_initial := given_to_father - given_to_mother
  father_initial - father_left

theorem father_ate_two_chocolates :
  father_ate_chocolates 20 4 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_father_ate_chocolates_father_ate_two_chocolates_l185_18511


namespace NUMINAMATH_CALUDE_transformation_is_left_shift_l185_18539

/-- A function representing a horizontal shift transformation -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x + shift)

/-- The original function composition -/
def originalFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x - 1)

/-- The transformed function composition -/
def transformedFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x + 1)

theorem transformation_is_left_shift (f : ℝ → ℝ) :
  transformedFunc f = horizontalShift (originalFunc f) 1 := by
  sorry

end NUMINAMATH_CALUDE_transformation_is_left_shift_l185_18539


namespace NUMINAMATH_CALUDE_amy_age_2005_l185_18545

/-- Amy's age at the end of 2000 -/
def amy_age_2000 : ℕ := sorry

/-- Amy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℕ := sorry

/-- The year 2000 -/
def year_2000 : ℕ := 2000

/-- The sum of Amy's and her grandfather's birth years -/
def birth_years_sum : ℕ := 3900

theorem amy_age_2005 : 
  grandfather_age_2000 = 3 * amy_age_2000 →
  year_2000 - amy_age_2000 + (year_2000 - grandfather_age_2000) = birth_years_sum →
  amy_age_2000 + 5 = 30 := by sorry

end NUMINAMATH_CALUDE_amy_age_2005_l185_18545


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l185_18524

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed of the car in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 80) 
  (h2 : average_speed = 70) : 
  (2 * average_speed - speed_first_hour) = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l185_18524


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l185_18594

theorem min_value_expression (x : ℝ) :
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  ∃ x : ℝ, x = 2/3 ∧ 
    Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l185_18594


namespace NUMINAMATH_CALUDE_max_girls_for_five_boys_valid_arrangement_l185_18595

/-- The maximum number of girls that can be arranged in a "Mathematical Ballet" -/
def max_girls (num_boys : ℕ) : ℕ :=
  (num_boys.choose 2) * 2

/-- Theorem stating the maximum number of girls for 5 boys -/
theorem max_girls_for_five_boys :
  max_girls 5 = 20 := by
  sorry

/-- Theorem proving the validity of the arrangement -/
theorem valid_arrangement (num_boys : ℕ) (num_girls : ℕ) :
  num_girls ≤ max_girls num_boys →
  ∃ (boy_positions : Fin num_boys → ℝ × ℝ)
    (girl_positions : Fin num_girls → ℝ × ℝ),
    ∀ (g : Fin num_girls),
      ∃ (b1 b2 : Fin num_boys),
        b1 ≠ b2 ∧
        dist (girl_positions g) (boy_positions b1) = 5 ∧
        dist (girl_positions g) (boy_positions b2) = 5 ∧
        ∀ (b : Fin num_boys),
          b ≠ b1 ∧ b ≠ b2 →
          dist (girl_positions g) (boy_positions b) ≠ 5 := by
  sorry


end NUMINAMATH_CALUDE_max_girls_for_five_boys_valid_arrangement_l185_18595


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l185_18500

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 312 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l185_18500


namespace NUMINAMATH_CALUDE_friend_team_assignments_l185_18591

/-- The number of ways to assign n distinguishable objects to k categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- The number of friends -/
def num_friends : ℕ := 6

/-- The number of teams -/
def num_teams : ℕ := 4

theorem friend_team_assignments :
  assignments num_friends num_teams = 4096 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignments_l185_18591


namespace NUMINAMATH_CALUDE_vacuum_time_proof_l185_18505

theorem vacuum_time_proof (upstairs downstairs total : ℕ) : 
  upstairs = 2 * downstairs + 5 →
  upstairs = 27 →
  total = upstairs + downstairs →
  total = 38 := by
sorry

end NUMINAMATH_CALUDE_vacuum_time_proof_l185_18505


namespace NUMINAMATH_CALUDE_circle_condition_l185_18565

/-- The equation of a potential circle with a parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 4*y + m = 0

/-- A predicate to check if an equation represents a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) : is_circle m ↔ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l185_18565


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l185_18510

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  horizontal_lines : Nat
  vertical_lines : Nat

/-- Calculates the number of squares on a checkerboard -/
def count_squares (board : Checkerboard) : Nat :=
  sorry

/-- Calculates the number of rectangles on a checkerboard -/
def count_rectangles (board : Checkerboard) : Nat :=
  sorry

/-- The main theorem stating the ratio of squares to rectangles on a 6x6 checkerboard -/
theorem squares_to_rectangles_ratio (board : Checkerboard) :
  board.rows = 6 ∧ board.cols = 6 ∧ board.horizontal_lines = 5 ∧ board.vertical_lines = 5 →
  (count_squares board : Rat) / (count_rectangles board : Rat) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l185_18510


namespace NUMINAMATH_CALUDE_books_sold_on_friday_l185_18568

theorem books_sold_on_friday (initial_stock : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (not_sold : ℕ)
  (h1 : initial_stock = 800)
  (h2 : monday = 60)
  (h3 : tuesday = 10)
  (h4 : wednesday = 20)
  (h5 : thursday = 44)
  (h6 : not_sold = 600) :
  initial_stock - not_sold - (monday + tuesday + wednesday + thursday) = 66 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_on_friday_l185_18568


namespace NUMINAMATH_CALUDE_article_choice_correct_l185_18547

-- Define the possible article choices
inductive Article
  | A
  | The
  | None

-- Define the structure for an article combination
structure ArticleCombination where
  first : Article
  second : Article

-- Define the conditions of the problem
def is_general_reference (a : Article) : Prop :=
  a = Article.A

def is_specific_reference (a : Article) : Prop :=
  a = Article.The

-- Define the correct combination
def correct_combination : ArticleCombination :=
  { first := Article.A, second := Article.The }

-- Theorem to prove
theorem article_choice_correct
  (german_engineer_general : is_general_reference correct_combination.first)
  (car_invention_specific : is_specific_reference correct_combination.second) :
  correct_combination = { first := Article.A, second := Article.The } := by
  sorry

end NUMINAMATH_CALUDE_article_choice_correct_l185_18547


namespace NUMINAMATH_CALUDE_repel_creatures_l185_18574

/-- Represents the number of cloves needed to repel creatures -/
def cloves_needed (vampires wights vampire_bats : ℕ) : ℕ :=
  let vampires_cloves := (3 * vampires + 1) / 2
  let wights_cloves := wights
  let bats_cloves := (3 * vampire_bats + 7) / 8
  vampires_cloves + wights_cloves + bats_cloves

/-- Theorem stating the number of cloves needed to repel specific numbers of creatures -/
theorem repel_creatures : cloves_needed 30 12 40 = 72 := by
  sorry

end NUMINAMATH_CALUDE_repel_creatures_l185_18574


namespace NUMINAMATH_CALUDE_stream_speed_l185_18536

/-- Given a boat's speed in still water and its downstream travel time and distance,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ) :
  boat_speed = 24 →
  downstream_time = 7 →
  downstream_distance = 196 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l185_18536


namespace NUMINAMATH_CALUDE_union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l185_18502

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem union_and_complement_when_a_is_one :
  (A ∪ B 1 = {x | -1 ≤ x ∧ x ≤ 3}) ∧
  (Set.univ \ B 1 = {x | x < 0 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ∈ Set.Ioi (-2) ∪ Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l185_18502


namespace NUMINAMATH_CALUDE_fraction_sum_l185_18558

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 14 = (5 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l185_18558


namespace NUMINAMATH_CALUDE_complex_number_equality_l185_18579

theorem complex_number_equality (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  (a + b * i : ℂ) = -1 + 2 * i := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l185_18579


namespace NUMINAMATH_CALUDE_engine_batches_l185_18529

theorem engine_batches :
  ∀ (total_engines : ℕ) (defective_engines : ℕ) (non_defective_engines : ℕ) (engines_per_batch : ℕ),
    defective_engines = total_engines / 4 →
    non_defective_engines = 300 →
    engines_per_batch = 80 →
    total_engines = defective_engines + non_defective_engines →
    total_engines / engines_per_batch = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_engine_batches_l185_18529


namespace NUMINAMATH_CALUDE_science_club_election_l185_18540

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

theorem science_club_election (total : ℕ) (former : ℕ) (board_size : ℕ)
  (h1 : total = 18)
  (h2 : former = 8)
  (h3 : board_size = 6)
  (h4 : former ≤ total)
  (h5 : board_size ≤ total) :
  binomial total board_size - binomial (total - former) board_size = 18354 := by
  sorry

end NUMINAMATH_CALUDE_science_club_election_l185_18540


namespace NUMINAMATH_CALUDE_library_reorganization_l185_18525

theorem library_reorganization (total_books : ℕ) (books_per_section : ℕ) (remainder : ℕ) : 
  total_books = 1521 * 41 →
  books_per_section = 45 →
  remainder = total_books % books_per_section →
  remainder = 36 := by
sorry

end NUMINAMATH_CALUDE_library_reorganization_l185_18525


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l185_18569

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l185_18569


namespace NUMINAMATH_CALUDE_two_digit_integers_with_remainder_three_l185_18581

theorem two_digit_integers_with_remainder_three : 
  (Finset.filter 
    (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) 
    (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_with_remainder_three_l185_18581


namespace NUMINAMATH_CALUDE_kenny_jumping_jacks_wednesday_l185_18551

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_jumping_jacks_wednesday (lastWeek : ℕ) (thisWeek : WeeklyJumpingJacks) 
    (h1 : lastWeek = 324)
    (h2 : thisWeek.sunday = 34)
    (h3 : thisWeek.monday = 20)
    (h4 : thisWeek.tuesday = 0)
    (h5 : thisWeek.thursday = 64 ∨ thisWeek.wednesday = 64)
    (h6 : thisWeek.friday = 23)
    (h7 : thisWeek.saturday = 61)
    (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.wednesday = 59 := by
  sorry

#check kenny_jumping_jacks_wednesday

end NUMINAMATH_CALUDE_kenny_jumping_jacks_wednesday_l185_18551


namespace NUMINAMATH_CALUDE_solution_set_transformation_l185_18531

theorem solution_set_transformation (k a b c : ℝ) :
  (∀ x, (x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 2 3) ↔ 
    (k * x / (a * x - 1) + (b * x - 1) / (c * x - 1) < 0)) →
  (∀ x, (x ∈ Set.Ioo (-1/2 : ℝ) (-1/3) ∪ Set.Ioo (1/2) 1) ↔ 
    (k / (x + a) + (x + b) / (x + c) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l185_18531


namespace NUMINAMATH_CALUDE_blackboard_numbers_l185_18535

theorem blackboard_numbers (n : ℕ) (h : n = 1987) :
  let S := n * (n + 1) / 2
  let remaining_sum := S % 7
  ∃ x, x ≤ 6 ∧ (x + 987) % 7 = remaining_sum :=
by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l185_18535


namespace NUMINAMATH_CALUDE_exists_equivalent_expression_l185_18553

/-- Define a type for the unknown operations -/
inductive UnknownOp
| add
| sub

/-- Define a function that applies the unknown operation -/
def applyOp (op : UnknownOp) (x y : ℝ) : ℝ :=
  match op with
  | UnknownOp.add => x + y
  | UnknownOp.sub => x - y

/-- Define a function that represents the reversed subtraction -/
def revSub (x y : ℝ) : ℝ := y - x

theorem exists_equivalent_expression :
  ∃ (op1 op2 : UnknownOp) (f1 f2 : ℝ → ℝ → ℝ),
    (f1 = applyOp op1 ∧ f2 = applyOp op2) ∨
    (f1 = applyOp op1 ∧ f2 = revSub) ∨
    (f1 = revSub ∧ f2 = applyOp op2) →
    ∀ (a b : ℝ), ∃ (expr : ℝ), expr = 20 * a - 18 * b :=
by sorry

end NUMINAMATH_CALUDE_exists_equivalent_expression_l185_18553


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l185_18552

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def B : ℕ := 3000

/-- The sum of four-digit odd numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 7500 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l185_18552


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_polynomial_l185_18575

theorem integer_roots_of_cubic_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, r^3 + a₂ * r^2 + a₁ * r + 24 = 0 → r ∣ 24 := by
sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_polynomial_l185_18575


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l185_18559

/-- Definition of "equation number pair" -/
def is_equation_number_pair (a b : ℝ) : Prop :=
  ∃ x : ℝ, (a / x) + 1 = b ∧ x = 1 / (a + b)

/-- Part 1: Prove [3,-5] is an "equation number pair" and [-2,4] is not -/
theorem part_one :
  is_equation_number_pair 3 (-5) ∧ ¬is_equation_number_pair (-2) 4 := by sorry

/-- Part 2: If [n,3-n] is an "equation number pair", then n = 1/2 -/
theorem part_two (n : ℝ) :
  is_equation_number_pair n (3 - n) → n = 1/2 := by sorry

/-- Part 3: If [m-k,k] is an "equation number pair" (m ≠ -1, m ≠ 0, k ≠ 1), then k = (m^2 + 1) / (m + 1) -/
theorem part_three (m k : ℝ) (hm1 : m ≠ -1) (hm2 : m ≠ 0) (hk : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l185_18559


namespace NUMINAMATH_CALUDE_track_circumference_l185_18527

/-- Represents the circular track and the runners' positions --/
structure TrackSystem where
  circumference : ℝ
  first_meeting_distance : ℝ
  second_meeting_distance : ℝ

/-- The conditions of the problem --/
def problem_conditions (t : TrackSystem) : Prop :=
  t.first_meeting_distance = 150 ∧
  t.second_meeting_distance = t.circumference - 90 ∧
  2 * t.circumference = t.first_meeting_distance * 2 + t.second_meeting_distance

/-- The theorem stating that the circumference is 300 yards --/
theorem track_circumference (t : TrackSystem) :
  problem_conditions t → t.circumference = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_track_circumference_l185_18527


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l185_18576

/-- Calculates the cost per liter of fuel given the conditions of the problem -/
theorem fuel_cost_per_liter 
  (service_cost : ℝ) 
  (minivan_count : ℕ) 
  (truck_count : ℕ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_multiplier : ℝ) :
  service_cost = 2.20 →
  minivan_count = 3 →
  truck_count = 2 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank_multiplier = 2.2 →
  (total_cost - (service_cost * (minivan_count + truck_count))) / 
  (minivan_count * minivan_tank + truck_count * (minivan_tank * truck_tank_multiplier)) = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l185_18576


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l185_18507

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l185_18507


namespace NUMINAMATH_CALUDE_sqrt3_minus1_power0_plus_2_power_neg1_l185_18564

theorem sqrt3_minus1_power0_plus_2_power_neg1 : (Real.sqrt 3 - 1) ^ 0 + 2 ^ (-1 : ℤ) = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_minus1_power0_plus_2_power_neg1_l185_18564


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l185_18572

theorem diophantine_equation_solutions :
  ∀ n k m : ℕ, 5^n - 3^k = m^2 →
    ((n = 0 ∧ k = 0 ∧ m = 0) ∨ (n = 2 ∧ k = 2 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l185_18572


namespace NUMINAMATH_CALUDE_probability_theorem_l185_18588

def is_valid (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧
  1 ≤ b ∧ b ≤ 60 ∧
  1 ≤ c ∧ c ≤ 60 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def satisfies_condition (a b c : ℕ) : Prop :=
  ∃ m : ℕ, (a * b * c + a + b + c) = 6 * m - 2

def total_combinations : ℕ := Nat.choose 60 3

def valid_combinations : ℕ := 14620

theorem probability_theorem :
  (valid_combinations : ℚ) / total_combinations = 2437 / 5707 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l185_18588


namespace NUMINAMATH_CALUDE_select_players_correct_l185_18584

/-- The number of ways to select k players from m teams, each with n players,
    such that no two selected players are from the same team -/
def select_players (m n k : ℕ) : ℕ :=
  Nat.choose m k * n^k

/-- Theorem stating that select_players gives the correct number of ways
    to form the committee under the given conditions -/
theorem select_players_correct (m n k : ℕ) (h : k ≤ m) :
  select_players m n k = Nat.choose m k * n^k :=
by sorry

end NUMINAMATH_CALUDE_select_players_correct_l185_18584


namespace NUMINAMATH_CALUDE_all_statements_valid_l185_18528

/-- Represents a simple programming language statement --/
inductive Statement
  | Assignment (var : String) (value : Int)
  | MultiAssignment (vars : List String) (values : List Int)
  | Input (prompt : Option String) (var : String)
  | Print (prompt : Option String) (expr : Option String)

/-- Checks if a statement is valid according to our rules --/
def isValid : Statement → Bool
  | Statement.Assignment _ _ => true
  | Statement.MultiAssignment vars values => vars.length == values.length
  | Statement.Input _ _ => true
  | Statement.Print _ _ => true

/-- The set of corrected statements --/
def correctedStatements : List Statement := [
  Statement.MultiAssignment ["A", "B"] [50, 50],
  Statement.MultiAssignment ["x", "y", "z"] [1, 2, 3],
  Statement.Input (some "How old are you?") "x",
  Statement.Input none "x",
  Statement.Print (some "A+B=") (some "C"),
  Statement.Print (some "Good-bye!") none
]

theorem all_statements_valid : ∀ s ∈ correctedStatements, isValid s := by sorry

end NUMINAMATH_CALUDE_all_statements_valid_l185_18528


namespace NUMINAMATH_CALUDE_intersection_A_B_l185_18503

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l185_18503


namespace NUMINAMATH_CALUDE_bargain_bin_books_l185_18596

theorem bargain_bin_books (initial_books : ℕ) : 
  initial_books - 3 + 10 = 11 → initial_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l185_18596


namespace NUMINAMATH_CALUDE_bus_passing_time_l185_18548

theorem bus_passing_time (distance : ℝ) (time : ℝ) (bus_length : ℝ) : 
  distance = 12 → time = 5 → bus_length = 200 →
  (bus_length / (distance * 1000 / (time * 60))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_passing_time_l185_18548


namespace NUMINAMATH_CALUDE_amanda_notebooks_l185_18526

/-- Calculates the final number of notebooks Amanda has -/
def final_notebooks (initial : ℕ) (ordered : ℕ) (lost : ℕ) : ℕ :=
  initial + ordered - lost

theorem amanda_notebooks :
  final_notebooks 10 6 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l185_18526


namespace NUMINAMATH_CALUDE_inequality_proof_l185_18597

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) : 
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < 39/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l185_18597


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l185_18549

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (3^11 + 5^13) → p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l185_18549


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l185_18557

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_count : ℕ) (h_count : ℕ) (br_count : ℕ) 
  (n_weight : ℝ) (h_weight : ℝ) (br_weight : ℝ) : ℝ :=
  n_count * n_weight + h_count * h_weight + br_count * br_weight

/-- The molecular weight of a compound with 1 N, 4 H, and 1 Br atom is 97.95 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 4 1 14.01 1.01 79.90 = 97.95 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l185_18557


namespace NUMINAMATH_CALUDE_expression_evaluation_l185_18543

theorem expression_evaluation : 1583 + 240 / 60 * 5 - 283 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l185_18543


namespace NUMINAMATH_CALUDE_cubic_polynomial_interpolation_l185_18578

-- Define the set of cubic polynomials over ℝ
def CubicPolynomial : Type := ℝ → ℝ

-- Define the property of being a cubic polynomial
def IsCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, p x = a * x^3 + b * x^2 + c * x + d

-- Theorem statement
theorem cubic_polynomial_interpolation
  (P Q R : CubicPolynomial)
  (hP : IsCubicPolynomial P)
  (hQ : IsCubicPolynomial Q)
  (hR : IsCubicPolynomial R)
  (h_order : ∀ x : ℝ, P x ≤ Q x ∧ Q x ≤ R x)
  (h_equal : ∃ x₀ : ℝ, P x₀ = R x₀) :
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ ∀ x : ℝ, Q x = k * P x + (1 - k) * R x :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_interpolation_l185_18578


namespace NUMINAMATH_CALUDE_function_has_one_zero_l185_18593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 - x

theorem function_has_one_zero (a : ℝ) (h1 : |a| ≥ 1 / (2 * Real.exp 1)) 
  (h2 : ∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_function_has_one_zero_l185_18593


namespace NUMINAMATH_CALUDE_intersection_of_lines_l185_18589

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (3, -2, 6) → 
  B = (13, -12, 11) → 
  C = (1, 5, -3) → 
  D = (3, -1, 9) → 
  ∃ t s : ℝ, 
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = 
    (1 + 2*s, 5 - 6*s, -3 + 12*s) ∧
    (3 + 10*t, -2 - 10*t, 6 + 5*t) = (7.5, -6.5, 8.25) := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l185_18589


namespace NUMINAMATH_CALUDE_tetrahedron_edge_lengths_l185_18506

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  -- Edge lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Ensure edges form a valid triangle
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Faces are congruent triangles with one 60° angle
  congruent_faces : a^2 + b^2 - a*b/2 = c^2
  -- Circumscribed sphere has diameter 23
  circumsphere : a^2 + b^2 + c^2 = 2 * 23^2

/-- The tetrahedron satisfies the given conditions and has the specified edge lengths -/
theorem tetrahedron_edge_lengths : 
  ∃ (t : Tetrahedron), t.a = 16 ∧ t.b = 21 ∧ t.c = 19 :=
by sorry


end NUMINAMATH_CALUDE_tetrahedron_edge_lengths_l185_18506


namespace NUMINAMATH_CALUDE_experiment_sequences_l185_18598

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the condition that procedure A can only be first or last -/
def a_first_or_last : ℕ := 2

/-- Represents the number of ways to arrange C and D adjacently -/
def cd_adjacent : ℕ := 2

/-- Represents the number of ways to arrange the remaining procedures -/
def remaining_arrangements : ℕ := 3

/-- The total number of possible sequences for the experiment -/
def total_sequences : ℕ := a_first_or_last * cd_adjacent * remaining_arrangements.factorial

theorem experiment_sequences :
  total_sequences = 24 := by sorry

end NUMINAMATH_CALUDE_experiment_sequences_l185_18598


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l185_18509

theorem set_equality_implies_sum (m n : ℝ) : 
  let P : Set ℝ := {m / n, 1}
  let Q : Set ℝ := {n, 0}
  P = Q → m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l185_18509


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_and_5_l185_18583

theorem three_digit_divisible_by_11_and_5 : 
  (Finset.filter (fun n => n % 55 = 0) (Finset.range 900 ⊔ Finset.range 100)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_and_5_l185_18583


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l185_18534

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem statement -/
theorem parallel_lines_m_equals_one (m : ℝ) :
  parallel_lines 1 (1 + m) (m - 2) (2 * m) 4 16 → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l185_18534


namespace NUMINAMATH_CALUDE_present_worth_from_discounts_l185_18516

/-- Present worth of a bill given true discount and banker's discount -/
theorem present_worth_from_discounts (TD BD : ℚ) : 
  TD = 36 → BD = 37.62 → 
  ∃ P : ℚ, P = 800 ∧ BD = (TD * (P + TD)) / P := by
  sorry

#check present_worth_from_discounts

end NUMINAMATH_CALUDE_present_worth_from_discounts_l185_18516


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l185_18520

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (additional_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_purchase = 100)
  (h4 : additional_decaf_percent = 60) :
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let additional_decaf := additional_purchase * (additional_decaf_percent / 100)
  let total_decaf := initial_decaf + additional_decaf
  let total_stock := initial_stock + additional_purchase
  (total_decaf / total_stock) * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l185_18520


namespace NUMINAMATH_CALUDE_integral_inequality_l185_18560

theorem integral_inequality (a : ℝ) (ha : a > 1) :
  (1 / (a - 1)) * (1 - (Real.log a / (a - 1))) < 
  (a - Real.log a - 1) / (a * (Real.log a)^2) ∧
  (a - Real.log a - 1) / (a * (Real.log a)^2) < 
  (1 / Real.log a) * (1 - (Real.log (Real.log a + 1) / Real.log a)) := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l185_18560


namespace NUMINAMATH_CALUDE_remainder_three_to_ninth_mod_five_l185_18577

theorem remainder_three_to_ninth_mod_five : 3^9 ≡ 3 [MOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_ninth_mod_five_l185_18577


namespace NUMINAMATH_CALUDE_binary_arithmetic_theorem_l185_18585

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_arithmetic_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, false, true]  -- 1011₂
  let result := [false, true, false, true, true, false, true]  -- 1011010₂
  (binary_to_nat a * binary_to_nat b + binary_to_nat c) = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_theorem_l185_18585


namespace NUMINAMATH_CALUDE_percentage_increase_johns_raise_l185_18556

theorem percentage_increase (original new : ℝ) (h1 : original > 0) (h2 : new > original) :
  (new - original) / original * 100 = 100 ↔ new = 2 * original :=
by sorry

theorem johns_raise :
  let original : ℝ := 40
  let new : ℝ := 80
  (new - original) / original * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_johns_raise_l185_18556


namespace NUMINAMATH_CALUDE_pet_farm_ratio_l185_18504

theorem pet_farm_ratio (rabbit_count : ℕ) (hamster_count : ℕ) : 
  (rabbit_count : ℚ) / hamster_count = 4 / 5 → 
  rabbit_count = 20 → 
  hamster_count = 25 := by
sorry

end NUMINAMATH_CALUDE_pet_farm_ratio_l185_18504


namespace NUMINAMATH_CALUDE_owls_on_fence_l185_18512

/-- The number of owls on a fence after more owls join is the sum of the initial number and the number that joined. -/
theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  let total_owls := initial_owls + joining_owls
  total_owls = initial_owls + joining_owls :=
by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l185_18512


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l185_18561

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.2 * G) : 
  R = 1.6 * G := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l185_18561


namespace NUMINAMATH_CALUDE_min_sum_first_two_terms_l185_18544

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 3024) / (1 + a (n + 1))

/-- The theorem stating the minimum possible value of a₁ + a₂ -/
theorem min_sum_first_two_terms (a : ℕ → ℕ) (h : ValidSequence a) :
    ∀ b : ℕ → ℕ, ValidSequence b → a 1 + a 2 ≤ b 1 + b 2 :=
  sorry

end NUMINAMATH_CALUDE_min_sum_first_two_terms_l185_18544


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sums_l185_18523

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sums (a : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence (λ n : ℕ => a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sums_l185_18523


namespace NUMINAMATH_CALUDE_area_of_integral_triangle_with_perimeter_12_l185_18501

/-- Represents a triangle with integral sides --/
structure IntegralTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of an integral triangle with perimeter 12 is 2√6 --/
theorem area_of_integral_triangle_with_perimeter_12 (t : IntegralTriangle) : 
  Real.sqrt (6 * (6 - t.a) * (6 - t.b) * (6 - t.c)) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_area_of_integral_triangle_with_perimeter_12_l185_18501


namespace NUMINAMATH_CALUDE_not_all_pairs_perfect_square_l185_18567

theorem not_all_pairs_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_all_pairs_perfect_square_l185_18567


namespace NUMINAMATH_CALUDE_geometry_relations_l185_18562

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l : Line) (m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : subset m β) :
  (perpendicular l β → plane_perpendicular α β) ∧
  (parallel α β → line_parallel l β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l185_18562


namespace NUMINAMATH_CALUDE_coefficient_of_a_l185_18599

theorem coefficient_of_a (a b : ℝ) (h1 : a = 2) (h2 : b = 15) : 
  42 * b = 630 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_a_l185_18599


namespace NUMINAMATH_CALUDE_line_canonical_to_general_equations_l185_18546

/-- Given a line in 3D space defined by canonical equations, prove that the general equations are equivalent. -/
theorem line_canonical_to_general_equations :
  ∀ (x y z : ℝ),
  ((x - 2) / 3 = (y + 1) / 5 ∧ (x - 2) / 3 = (z - 3) / (-1)) ↔
  (5 * x - 3 * y = 13 ∧ x + 3 * z = 11) :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_to_general_equations_l185_18546


namespace NUMINAMATH_CALUDE_min_value_quadratic_l185_18513

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 + 10*x - 8*y + 34 ≥ -7 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 + 10*a - 8*b + 34 = -7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l185_18513


namespace NUMINAMATH_CALUDE_equation_solution_l185_18519

theorem equation_solution : 
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 3) = 1 / (x + 6) + 1 / (x + 2)) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l185_18519


namespace NUMINAMATH_CALUDE_jersey_profit_is_152_l185_18530

/-- The amount of money made from selling jerseys during a game -/
def money_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem stating that the money made from selling jerseys is $152 -/
theorem jersey_profit_is_152 :
  let profit_per_jersey : ℕ := 76
  let profit_per_tshirt : ℕ := 204
  let tshirts_sold : ℕ := 158
  let jerseys_sold : ℕ := 2
  money_from_jerseys profit_per_jersey jerseys_sold = 152 := by
sorry

end NUMINAMATH_CALUDE_jersey_profit_is_152_l185_18530


namespace NUMINAMATH_CALUDE_yellow_balls_count_l185_18550

theorem yellow_balls_count (total_balls : ℕ) (yellow_probability : ℚ) 
  (h1 : total_balls = 40)
  (h2 : yellow_probability = 3/10) :
  (yellow_probability * total_balls : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l185_18550


namespace NUMINAMATH_CALUDE_lcm_24_36_42_l185_18508

theorem lcm_24_36_42 : Nat.lcm 24 (Nat.lcm 36 42) = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_42_l185_18508


namespace NUMINAMATH_CALUDE_polynomial_division_l185_18518

theorem polynomial_division (x : ℂ) : 
  ∃! (a : ℤ), ∃ (p : ℂ → ℂ), (x^2 - x + a) * p x = x^15 + x^2 + 100 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l185_18518


namespace NUMINAMATH_CALUDE_oil_barrels_problem_l185_18587

theorem oil_barrels_problem (a b : ℝ) : 
  a > 0 ∧ b > 0 →  -- Initial amounts are positive
  (2/3 * a + 1/5 * (b + 1/3 * a) = 24) ∧  -- Amount in A after transfers
  ((b + 1/3 * a) * 4/5 = 24) →  -- Amount in B after transfers
  a - b = 6 := by sorry

end NUMINAMATH_CALUDE_oil_barrels_problem_l185_18587


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l185_18537

theorem quadratic_complete_square (x : ℝ) :
  25 * x^2 + 20 * x - 1000 = 0 →
  ∃ (p t : ℝ), (x + p)^2 = t ∧ t = 104/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l185_18537


namespace NUMINAMATH_CALUDE_ellipse_condition_l185_18555

/-- Represents a curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

theorem ellipse_condition (c : Curve) :
  (is_ellipse c → c.a > 0 ∧ c.b > 0) ∧
  (∃ c : Curve, c.a > 0 ∧ c.b > 0 ∧ ¬is_ellipse c) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l185_18555


namespace NUMINAMATH_CALUDE_team_games_played_l185_18590

/-- Proves that a team with the given win percentages played 125 games in total -/
theorem team_games_played (first_100_win_rate : Real) (remaining_win_rate : Real) (total_win_rate : Real) :
  first_100_win_rate = 0.75 →
  remaining_win_rate = 0.50 →
  total_win_rate = 0.70 →
  ∃ (total_games : ℕ),
    total_games = 125 ∧
    (75 + remaining_win_rate * (total_games - 100 : ℝ)) / total_games = total_win_rate := by
  sorry

end NUMINAMATH_CALUDE_team_games_played_l185_18590


namespace NUMINAMATH_CALUDE_second_fraction_in_compound_ratio_l185_18522

theorem second_fraction_in_compound_ratio
  (compound_ratio : ℝ)
  (h_ratio : compound_ratio = 0.07142857142857142)
  (f1 : ℝ) (h_f1 : f1 = 2/3)
  (f3 : ℝ) (h_f3 : f3 = 1/3)
  (f4 : ℝ) (h_f4 : f4 = 3/8) :
  ∃ x : ℝ, x * f1 * f3 * f4 = compound_ratio ∧ x = 0.8571428571428571 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_in_compound_ratio_l185_18522


namespace NUMINAMATH_CALUDE_root_product_theorem_l185_18532

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a + 1)^2 - p*(b + 1/a + 1) + r = 0) →
  r = 19/3 := by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l185_18532


namespace NUMINAMATH_CALUDE_cone_base_diameter_l185_18571

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle -/
structure Cone where
  /-- The radius of the base of the cone -/
  radius : ℝ
  /-- The slant height of the cone -/
  slant_height : ℝ
  /-- The lateral surface unfolds into a semicircle -/
  lateral_surface_semicircle : slant_height = 2 * radius
  /-- The surface area of the cone is 3π -/
  surface_area : π * radius^2 + π * radius * slant_height = 3 * π

/-- The diameter of the base of the cone is 2 -/
theorem cone_base_diameter (c : Cone) : 2 * c.radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l185_18571


namespace NUMINAMATH_CALUDE_problem_statement_l185_18538

theorem problem_statement (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l185_18538


namespace NUMINAMATH_CALUDE_meet_once_l185_18541

/-- Represents the meeting of Michael and the garbage truck -/
structure MeetingProblem where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def numberOfMeetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def problemInstance : MeetingProblem :=
  { michaelSpeed := 4
  , truckSpeed := 12
  , pailDistance := 300
  , truckStopTime := 40
  , initialDistance := 300 }

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : numberOfMeetings problemInstance = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l185_18541


namespace NUMINAMATH_CALUDE_delores_purchase_shortfall_l185_18514

def initial_amount : ℚ := 450
def computer_cost : ℚ := 500
def computer_discount_rate : ℚ := 0.2
def printer_cost : ℚ := 50
def printer_tax_rate : ℚ := 0.2

def computer_discount : ℚ := computer_cost * computer_discount_rate
def discounted_computer_cost : ℚ := computer_cost - computer_discount
def printer_tax : ℚ := printer_cost * printer_tax_rate
def total_printer_cost : ℚ := printer_cost + printer_tax
def total_spent : ℚ := discounted_computer_cost + total_printer_cost

theorem delores_purchase_shortfall :
  initial_amount - total_spent = -10 := by sorry

end NUMINAMATH_CALUDE_delores_purchase_shortfall_l185_18514
