import Mathlib

namespace trains_meeting_time_l2295_229598

/-- Two trains meeting problem -/
theorem trains_meeting_time
  (distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (start_time_diff : ℝ)
  (h_distance : distance = 200)
  (h_speed_A : speed_A = 20)
  (h_speed_B : speed_B = 25)
  (h_start_time_diff : start_time_diff = 1) :
  let initial_distance_A := speed_A * start_time_diff
  let remaining_distance := distance - initial_distance_A
  let relative_speed := speed_A + speed_B
  let meeting_time := remaining_distance / relative_speed
  meeting_time + start_time_diff = 5 := by sorry

end trains_meeting_time_l2295_229598


namespace smallest_n_has_9_digits_l2295_229517

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def has_9_digits (n : ℕ) : Prop := n ≥ 100000000 ∧ n < 1000000000

theorem smallest_n_has_9_digits :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 30 ∧ is_perfect_cube (m^2) ∧ is_perfect_square (m^5))) ∧
    is_divisible_by n 30 ∧
    is_perfect_cube (n^2) ∧
    is_perfect_square (n^5) ∧
    has_9_digits n :=
sorry

end smallest_n_has_9_digits_l2295_229517


namespace percentage_difference_l2295_229595

theorem percentage_difference : 
  (38 / 100 : ℚ) * 80 - (12 / 100 : ℚ) * 160 = 11.2 := by sorry

end percentage_difference_l2295_229595


namespace problem_1_l2295_229515

theorem problem_1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

end problem_1_l2295_229515


namespace min_vertical_distance_l2295_229552

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 - 3*x - 5

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), vertical_distance x₀ ≤ vertical_distance x ∧ vertical_distance x₀ = 4 :=
sorry

end min_vertical_distance_l2295_229552


namespace cricket_match_average_l2295_229513

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The conditions of the cricket match -/
def cricket_match_conditions (scores : BatsmanScores) : Prop :=
  scores.d = scores.e + 5 ∧
  scores.e = scores.a - 8 ∧
  scores.b = scores.d + scores.e ∧
  scores.b + scores.c = 107 ∧
  scores.e = 20

/-- The theorem stating that the average score is 36 -/
theorem cricket_match_average (scores : BatsmanScores) 
  (h : cricket_match_conditions scores) : 
  (scores.a + scores.b + scores.c + scores.d + scores.e) / 5 = 36 := by
  sorry

#check cricket_match_average

end cricket_match_average_l2295_229513


namespace ln_increasing_on_positive_reals_l2295_229500

-- Define the open interval (0, +∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem ln_increasing_on_positive_reals :
  StrictMonoOn Real.log openPositiveReals :=
sorry

end ln_increasing_on_positive_reals_l2295_229500


namespace yulia_number_l2295_229541

theorem yulia_number (x : ℝ) : x + 13 = 4 * (x + 1) → x = 3 := by
  sorry

end yulia_number_l2295_229541


namespace doris_eggs_l2295_229536

/-- Represents the number of eggs in a package -/
inductive EggPackage
  | small : EggPackage
  | large : EggPackage

/-- Returns the number of eggs in a package -/
def eggs_in_package (p : EggPackage) : Nat :=
  match p with
  | EggPackage.small => 6
  | EggPackage.large => 11

/-- Calculates the total number of eggs bought given the number of large packs -/
def total_eggs (large_packs : Nat) : Nat :=
  large_packs * eggs_in_package EggPackage.large

/-- Proves that Doris bought 55 eggs in total -/
theorem doris_eggs :
  total_eggs 5 = 55 := by sorry

end doris_eggs_l2295_229536


namespace tournament_max_points_l2295_229597

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (points_for_loss : Nat)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : Nat :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_from_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_from_top

/-- The main theorem to be proved -/
theorem tournament_max_points :
  ∀ t : Tournament,
    t.num_teams = 8 ∧
    t.games_per_pair = 2 ∧
    t.points_for_win = 3 ∧
    t.points_for_draw = 1 ∧
    t.points_for_loss = 0 →
    max_points_for_top_teams t = 36 := by
  sorry

end tournament_max_points_l2295_229597


namespace min_participants_in_tournament_l2295_229561

theorem min_participants_in_tournament : ∃ (n : ℕ) (k : ℕ),
  n = 11 ∧
  k < n / 2 ∧
  k > (45 * n) / 100 ∧
  ∀ (m : ℕ) (j : ℕ), m < n →
    (j < m / 2 ∧ j > (45 * m) / 100) → False :=
by sorry

end min_participants_in_tournament_l2295_229561


namespace furniture_assembly_time_l2295_229511

theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 4) 
  (h3 : time_per_piece = 6) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end furniture_assembly_time_l2295_229511


namespace total_initials_eq_thousand_l2295_229543

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 3

/-- The total number of possible three-letter sets of initials using letters A through J -/
def total_initials : ℕ := num_letters ^ initials_length

/-- Theorem stating that the total number of possible three-letter sets of initials
    using letters A through J is 1000 -/
theorem total_initials_eq_thousand : total_initials = 1000 := by
  sorry

end total_initials_eq_thousand_l2295_229543


namespace school_trip_ratio_l2295_229520

theorem school_trip_ratio (total : ℕ) (remaining : ℕ) : 
  total = 1000 → 
  remaining = 250 → 
  (total / 2 - remaining) / remaining = 1 := by
  sorry

end school_trip_ratio_l2295_229520


namespace real_complex_intersection_l2295_229524

-- Define the set of real numbers
def RealNumbers : Set ℂ := {z : ℂ | z.im = 0}

-- Define the set of complex numbers
def ComplexNumbers : Set ℂ := Set.univ

-- Theorem statement
theorem real_complex_intersection :
  RealNumbers ∩ ComplexNumbers = RealNumbers := by sorry

end real_complex_intersection_l2295_229524


namespace black_highest_probability_l2295_229551

-- Define the bag contents
def total_balls : ℕ := 8
def white_balls : ℕ := 1
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def black_balls : ℕ := 3

-- Define probabilities
def prob_white : ℚ := white_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_yellow : ℚ := yellow_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

-- Theorem statement
theorem black_highest_probability :
  prob_black > prob_white ∧ 
  prob_black > prob_red ∧ 
  prob_black > prob_yellow :=
sorry

end black_highest_probability_l2295_229551


namespace february_discount_correct_l2295_229588

/-- Represents the discount percentage applied in February -/
def discount_percentage : ℝ := 7

/-- Represents the initial markup percentage -/
def initial_markup : ℝ := 20

/-- Represents the New Year markup percentage -/
def new_year_markup : ℝ := 25

/-- Represents the profit percentage in February -/
def february_profit : ℝ := 39.5

/-- Theorem stating that the discount percentage in February is correct given the markups and profit -/
theorem february_discount_correct :
  let cost := 100 -- Assuming a base cost of 100 for simplicity
  let initial_price := cost * (1 + initial_markup / 100)
  let new_year_price := initial_price * (1 + new_year_markup / 100)
  let final_price := new_year_price * (1 - discount_percentage / 100)
  final_price - cost = february_profit * cost / 100 :=
sorry


end february_discount_correct_l2295_229588


namespace expand_expression_l2295_229581

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l2295_229581


namespace pams_apples_l2295_229580

theorem pams_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : gerald_apples_per_bag = 40) :
  pam_bags * (3 * gerald_apples_per_bag) = 1200 := by
  sorry

end pams_apples_l2295_229580


namespace volleyball_scoring_l2295_229566

/-- Volleyball team scoring problem -/
theorem volleyball_scoring
  (lizzie_score : ℕ)
  (nathalie_score : ℕ)
  (aimee_score : ℕ)
  (teammates_score : ℕ)
  (total_score : ℕ)
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score > lizzie_score)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : total_score = 50)
  (h5 : teammates_score = 17)
  (h6 : lizzie_score + nathalie_score + aimee_score + teammates_score = total_score) :
  nathalie_score = lizzie_score + 3 := by
  sorry

end volleyball_scoring_l2295_229566


namespace cos_probability_l2295_229535

/-- The probability that cos(πx/2) is between 0 and 1/2 when x is randomly selected from [-1, 1] -/
theorem cos_probability : 
  ∃ (P : Set ℝ → ℝ), 
    (∀ x ∈ Set.Icc (-1) 1, P {y | 0 ≤ Real.cos (π * y / 2) ∧ Real.cos (π * y / 2) ≤ 1/2} = 1/3) :=
by sorry

end cos_probability_l2295_229535


namespace can_obtain_all_graphs_l2295_229549

/-- Represents a candidate in the election -/
structure Candidate where
  id : Nat

/-- Represents a voter's ranking of candidates -/
structure Ranking where
  preferences : List Candidate

/-- Represents the election system -/
structure ElectionSystem where
  candidates : Finset Candidate
  voters : Finset Nat
  rankings : Nat → Ranking

/-- Represents a directed graph -/
structure DirectedGraph where
  vertices : Finset Candidate
  edges : Candidate → Candidate → Bool

/-- Counts the number of votes where a is ranked higher than b -/
def countPreferences (system : ElectionSystem) (a b : Candidate) : Nat :=
  sorry

/-- Checks if there should be an edge from a to b based on majority preference -/
def hasEdge (system : ElectionSystem) (a b : Candidate) : Bool :=
  2 * countPreferences system a b > system.voters.card

/-- Constructs a directed graph based on the election system -/
def constructGraph (system : ElectionSystem) : DirectedGraph :=
  sorry

/-- Theorem stating that any connected complete directed graph can be obtained -/
theorem can_obtain_all_graphs (n : Nat) :
  ∃ (system : ElectionSystem),
    system.candidates.card = n ∧
    system.voters.card = n ∧
    ∀ (g : DirectedGraph),
      g.vertices = system.candidates →
      ∃ (newSystem : ElectionSystem),
        newSystem.candidates = system.candidates ∧
        constructGraph newSystem = g :=
  sorry

end can_obtain_all_graphs_l2295_229549


namespace volume_eq_cross_section_area_l2295_229508

/-- A right prism with an equilateral triangular base -/
structure EquilateralPrism where
  /-- Side length of the equilateral triangular base -/
  a : ℝ
  /-- Angle between the cross-section plane and the base -/
  φ : ℝ
  /-- Area of the cross-section -/
  Q : ℝ
  /-- Side length is positive -/
  h_a_pos : 0 < a
  /-- Angle is between 0 and π/2 -/
  h_φ_range : 0 < φ ∧ φ < Real.pi / 2
  /-- Area is positive -/
  h_Q_pos : 0 < Q

/-- The volume of the equilateral prism -/
def volume (p : EquilateralPrism) : ℝ := p.Q

theorem volume_eq_cross_section_area (p : EquilateralPrism) :
  volume p = p.Q := by sorry

end volume_eq_cross_section_area_l2295_229508


namespace circle_theorem_l2295_229593

/-- The circle passing through points A(1, 4) and B(3, 2) with its center on the line y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 4.5)^2 + y^2 = 28.25

/-- Point A -/
def point_A : ℝ × ℝ := (1, 4)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 2)

/-- Point P -/
def point_P : ℝ × ℝ := (2, 4)

/-- A point is inside the circle if the left side of the equation is less than the right side -/
def is_inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 4.5)^2 + p.2^2 < 28.25

theorem circle_theorem :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  is_inside_circle point_P :=
by sorry

end circle_theorem_l2295_229593


namespace intersection_point_sum_l2295_229505

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

end intersection_point_sum_l2295_229505


namespace quadratic_inequality_solution_range_l2295_229527

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 := by
  sorry

end quadratic_inequality_solution_range_l2295_229527


namespace image_of_two_zero_l2295_229534

/-- A mapping that transforms a point (x, y) into (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The image of the point (2, 0) under the mapping f is (2, 2) -/
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end image_of_two_zero_l2295_229534


namespace inequality_relation_l2295_229559

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end inequality_relation_l2295_229559


namespace xy_system_implies_x2_plus_y2_l2295_229582

theorem xy_system_implies_x2_plus_y2 (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 80 := by
sorry

end xy_system_implies_x2_plus_y2_l2295_229582


namespace lemon_heads_package_count_l2295_229586

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- The number of Lemon Heads left after eating -/
def lemon_heads_left : ℕ := 0

/-- The number of Lemon Heads per package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / boxes_finished

theorem lemon_heads_package_count : lemon_heads_per_package = 6 := by
  sorry

end lemon_heads_package_count_l2295_229586


namespace decimal_to_fraction_l2295_229557

theorem decimal_to_fraction : (0.38 : ℚ) = 19 / 50 := by sorry

end decimal_to_fraction_l2295_229557


namespace average_of_ABCD_l2295_229569

theorem average_of_ABCD (A B C D : ℚ) 
  (eq1 : 1001 * C - 2004 * A = 4008)
  (eq2 : 1001 * B + 3005 * A - 1001 * D = 6010) :
  (A + B + C + D) / 4 = (5 + D) / 2 := by
  sorry

end average_of_ABCD_l2295_229569


namespace fraction_simplification_l2295_229532

theorem fraction_simplification :
  (20 : ℚ) / 21 * 35 / 54 * 63 / 50 = 7 / 9 := by
  sorry

end fraction_simplification_l2295_229532


namespace pebble_difference_l2295_229545

theorem pebble_difference (candy_pebbles : ℕ) (lance_multiplier : ℕ) : 
  candy_pebbles = 4 →
  lance_multiplier = 3 →
  lance_multiplier * candy_pebbles - candy_pebbles = 8 := by
  sorry

end pebble_difference_l2295_229545


namespace first_runner_time_l2295_229503

/-- Represents a 600-meter relay race with three runners -/
structure RelayRace where
  runner1_time : ℝ
  runner2_time : ℝ
  runner3_time : ℝ

/-- The conditions of the specific relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.runner2_time = race.runner1_time + 2 ∧
  race.runner3_time = race.runner1_time - 3 ∧
  race.runner1_time + race.runner2_time + race.runner3_time = 71

/-- Theorem stating that given the race conditions, the first runner's time is 24 seconds -/
theorem first_runner_time (race : RelayRace) :
  race_conditions race → race.runner1_time = 24 :=
by
  sorry

end first_runner_time_l2295_229503


namespace marks_profit_l2295_229592

/-- The profit Mark makes from selling a Magic card -/
def profit (initial_cost : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_cost * value_multiplier - initial_cost

/-- Theorem stating that Mark's profit is $200 -/
theorem marks_profit : profit 100 3 = 200 := by
  sorry

end marks_profit_l2295_229592


namespace initial_number_of_people_l2295_229548

/-- Given a group of people where replacing one person increases the average weight,
    this theorem proves the initial number of people in the group. -/
theorem initial_number_of_people
  (n : ℕ) -- Initial number of people
  (weight_increase_per_person : ℝ) -- Average weight increase per person
  (weight_difference : ℝ) -- Weight difference between new and replaced person
  (h1 : weight_increase_per_person = 2.5)
  (h2 : weight_difference = 20)
  (h3 : n * weight_increase_per_person = weight_difference) :
  n = 8 :=
by sorry

end initial_number_of_people_l2295_229548


namespace tissues_cost_is_two_l2295_229576

def cost_of_tissues (toilet_paper_rolls : ℕ) (paper_towel_rolls : ℕ) (tissue_boxes : ℕ)
                    (toilet_paper_cost : ℚ) (paper_towel_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - (toilet_paper_rolls * toilet_paper_cost + paper_towel_rolls * paper_towel_cost)) / tissue_boxes

theorem tissues_cost_is_two :
  cost_of_tissues 10 7 3 (3/2) 2 35 = 2 :=
by sorry

end tissues_cost_is_two_l2295_229576


namespace oxford_high_school_population_is_1247_l2295_229555

/-- Represents the number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let full_time_teachers : ℕ := 80
  let part_time_teachers : ℕ := 5
  let principal : ℕ := 1
  let vice_principals : ℕ := 3
  let librarians : ℕ := 2
  let guidance_counselors : ℕ := 6
  let other_staff : ℕ := 25
  let classes : ℕ := 40
  let avg_students_per_class : ℕ := 25
  let part_time_students : ℕ := 250

  let full_time_students : ℕ := classes * avg_students_per_class
  let total_staff : ℕ := full_time_teachers + part_time_teachers + principal + 
                         vice_principals + librarians + guidance_counselors + other_staff
  let total_students : ℕ := full_time_students + (part_time_students / 2)

  total_staff + total_students

/-- Theorem stating that the total number of people in Oxford High School is 1247 -/
theorem oxford_high_school_population_is_1247 : 
  oxford_high_school_population = 1247 := by
  sorry

end oxford_high_school_population_is_1247_l2295_229555


namespace unique_triple_solution_l2295_229567

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 + z^2 = 3 ∧
  (x + y + z) * (x^2 + y^2 + z^2) = 9 :=
by sorry

end unique_triple_solution_l2295_229567


namespace unique_circle_construction_l2295_229526

/-- A line in a plane -/
structure Line : Type :=
  (l : Set (Real × Real))

/-- A point in a plane -/
structure Point : Type :=
  (x : Real) (y : Real)

/-- A circle in a plane -/
structure Circle : Type :=
  (center : Point) (radius : Real)

/-- Predicate to check if a point belongs to a line -/
def PointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Predicate to check if a circle passes through a point -/
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

/-- Predicate to check if a circle is tangent to a line at a point -/
def CircleTangentToLineAt (c : Circle) (l : Line) (p : Point) : Prop := sorry

/-- Main theorem: Existence and uniqueness of a circle passing through B and tangent to l at A -/
theorem unique_circle_construction (l : Line) (A B : Point) 
  (h1 : PointOnLine A l) 
  (h2 : ¬PointOnLine B l) : 
  ∃! k : Circle, CirclePassesThrough k B ∧ CircleTangentToLineAt k l A := by
  sorry

end unique_circle_construction_l2295_229526


namespace peanut_price_is_correct_l2295_229577

/-- The price of cashews per pound in dollars -/
def cashew_price : ℚ := 5

/-- The total weight of the mixture in pounds -/
def total_weight : ℚ := 25

/-- The total cost of the mixture in dollars -/
def total_cost : ℚ := 92

/-- The weight of cashews used in the mixture in pounds -/
def cashew_weight : ℚ := 11

/-- The price of peanuts per pound in dollars -/
def peanut_price : ℚ := (total_cost - cashew_price * cashew_weight) / (total_weight - cashew_weight)

theorem peanut_price_is_correct : peanut_price = 264/100 := by
  sorry

end peanut_price_is_correct_l2295_229577


namespace solve_for_n_l2295_229584

def first_seven_multiples_of_six : List ℕ := [6, 12, 18, 24, 30, 36, 42]

def a : ℚ := (List.sum first_seven_multiples_of_six) / 7

def b (n : ℕ) : ℕ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) : a ^ 2 - (b n) ^ 2 = 0 → n = 12 := by
  sorry

end solve_for_n_l2295_229584


namespace fly_distance_from_ceiling_l2295_229510

/-- Given a point (2, 7, z) in 3D space, where z is unknown, and its distance 
    from the origin (0, 0, 0) is 10 units, prove that z = √47. -/
theorem fly_distance_from_ceiling :
  ∀ z : ℝ, (2:ℝ)^2 + 7^2 + z^2 = 10^2 → z = Real.sqrt 47 := by
  sorry

end fly_distance_from_ceiling_l2295_229510


namespace total_journey_time_l2295_229504

/-- Represents the problem of Joe's journey to school -/
structure JourneyToSchool where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe takes to walk 1/3 of the distance

/-- Conditions of the problem -/
def journey_conditions (j : JourneyToSchool) : Prop :=
  j.run_speed = 4 * j.walk_speed ∧
  j.walk_time = 9 ∧
  j.walk_speed * j.walk_time = j.d / 3

/-- The theorem to be proved -/
theorem total_journey_time (j : JourneyToSchool) 
  (h : journey_conditions j) : 
  ∃ (total_time : ℝ), total_time = 13.5 ∧ 
    total_time = j.walk_time + (2 * j.d / 3) / j.run_speed :=
by sorry

end total_journey_time_l2295_229504


namespace bike_ride_speed_l2295_229574

theorem bike_ride_speed (x : ℝ) : 
  (210 / x = 210 / (x - 5) - 1) → x = 35 := by
sorry

end bike_ride_speed_l2295_229574


namespace B_work_time_l2295_229563

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℝ := 20

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_in_2_days : ℝ := 1 - 0.7666666666666666

theorem B_work_time (days_for_A : ℝ) (h1 : days_for_A = 15) :
  2 * (1 / days_for_A + 1 / days_for_B) = work_completed_in_2_days := by
  sorry

#check B_work_time

end B_work_time_l2295_229563


namespace f_properties_l2295_229547

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x + a - 3) / Real.log a

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (Function.Injective (f a) ↔ (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2)) :=
by sorry

end f_properties_l2295_229547


namespace dogs_not_eating_l2295_229525

theorem dogs_not_eating (total : ℕ) (like_apples : ℕ) (like_chicken : ℕ) (like_both : ℕ) :
  total = 75 →
  like_apples = 18 →
  like_chicken = 55 →
  like_both = 10 →
  total - (like_apples + like_chicken - like_both) = 12 := by
  sorry

end dogs_not_eating_l2295_229525


namespace student_congress_size_l2295_229512

/-- The number of classes in the school -/
def num_classes : ℕ := 40

/-- The number of representatives sent from each class -/
def representatives_per_class : ℕ := 3

/-- The sample size (number of students in the "Student Congress") -/
def sample_size : ℕ := num_classes * representatives_per_class

theorem student_congress_size :
  sample_size = 120 :=
by sorry

end student_congress_size_l2295_229512


namespace product_expansion_l2295_229568

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x) - 5 * x^3) = 3 / x - (15 / 7) * x^3 := by
  sorry

end product_expansion_l2295_229568


namespace train_length_l2295_229538

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h1 : speed_kmph = 90) (h2 : time_sec = 5) :
  speed_kmph * (1000 / 3600) * time_sec = 125 := by
  sorry

end train_length_l2295_229538


namespace smallest_box_volume_l2295_229514

/-- Represents a triangular pyramid (tetrahedron) -/
structure Pyramid where
  height : ℝ
  base_side : ℝ

/-- Represents a rectangular prism -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def box_volume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Checks if a box can safely contain a pyramid -/
def can_contain (b : Box) (p : Pyramid) : Prop :=
  b.height ≥ p.height ∧ b.length ≥ p.base_side ∧ b.width ≥ p.base_side

/-- The smallest box that can safely contain the pyramid -/
def smallest_box (p : Pyramid) : Box :=
  { length := 10, width := 10, height := p.height }

/-- Theorem: The volume of the smallest box that can safely contain the given pyramid is 1500 cubic inches -/
theorem smallest_box_volume (p : Pyramid) (h1 : p.height = 15) (h2 : p.base_side = 8) :
  box_volume (smallest_box p) = 1500 :=
by sorry

end smallest_box_volume_l2295_229514


namespace f_max_value_l2295_229523

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Theorem: The maximum value of f(x) = -x^2 + 2x + 3 is 4 -/
theorem f_max_value : ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l2295_229523


namespace parallel_vectors_x_value_l2295_229507

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (4, 2)
  let n : ℝ × ℝ := (x, -3)
  parallel m n → x = -6 := by
sorry

end parallel_vectors_x_value_l2295_229507


namespace bernie_postcard_transaction_l2295_229560

theorem bernie_postcard_transaction (initial_postcards : ℕ) 
  (sell_price : ℕ) (buy_price : ℕ) : 
  initial_postcards = 18 → 
  sell_price = 15 → 
  buy_price = 5 → 
  (initial_postcards / 2 * sell_price) / buy_price = 27 :=
by
  sorry

end bernie_postcard_transaction_l2295_229560


namespace quadratic_problem_l2295_229562

def quadratic_function (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

theorem quadratic_problem (b c : ℝ) :
  (∀ x, quadratic_function b c x < 0 ↔ 1 < x ∧ x < 3) →
  (quadratic_function b c = λ x => x^2 - 4*x + 3) ∧
  (∀ m, (∀ x, quadratic_function b c x > m*x - 1) ↔ -8 < m ∧ m < 0) :=
sorry

end quadratic_problem_l2295_229562


namespace polynomial_constant_term_l2295_229509

theorem polynomial_constant_term (a b c d e : ℝ) :
  (2^7 * a + 2^5 * b + 2^3 * c + 2 * d + e = 23) →
  ((-2)^7 * a + (-2)^5 * b + (-2)^3 * c + (-2) * d + e = -35) →
  e = -6 := by sorry

end polynomial_constant_term_l2295_229509


namespace selection_schemes_with_women_l2295_229501

/-- The number of ways to select 4 individuals from 4 men and 2 women, with at least 1 woman included -/
def selection_schemes (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose (total_men + total_women) to_select - Nat.choose total_men to_select

theorem selection_schemes_with_women (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) 
    (h1 : total_men = 4)
    (h2 : total_women = 2)
    (h3 : to_select = 4) :
  selection_schemes total_men total_women to_select = 14 := by
  sorry

end selection_schemes_with_women_l2295_229501


namespace salary_increase_proof_l2295_229537

theorem salary_increase_proof (original_salary : ℝ) 
  (h1 : original_salary * 1.8 = 25000) 
  (h2 : original_salary > 0) : 
  25000 - original_salary = 11111.11 := by
sorry

end salary_increase_proof_l2295_229537


namespace museum_trip_l2295_229587

theorem museum_trip (first_bus : ℕ) (second_bus : ℕ) (third_bus : ℕ) (fourth_bus : ℕ) :
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  first_bus + second_bus + third_bus + fourth_bus = 75 →
  fourth_bus - first_bus = 9 := by
sorry

end museum_trip_l2295_229587


namespace expression_value_l2295_229564

theorem expression_value (b : ℚ) (h : b = 1/3) :
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end expression_value_l2295_229564


namespace initial_fish_count_l2295_229529

def fish_eaten_per_day : ℕ := 2
def days_before_adding : ℕ := 14
def fish_added : ℕ := 8
def days_after_adding : ℕ := 7
def final_fish_count : ℕ := 26

theorem initial_fish_count (initial_count : ℕ) : 
  initial_count - (fish_eaten_per_day * days_before_adding) + fish_added - 
  (fish_eaten_per_day * days_after_adding) = final_fish_count → 
  initial_count = 60 := by
sorry

end initial_fish_count_l2295_229529


namespace moon_speed_conversion_l2295_229553

/-- The speed of the moon around the Earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.03

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon around the Earth in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion :
  moon_speed_km_per_hour = 3708 := by sorry

end moon_speed_conversion_l2295_229553


namespace base_7_addition_sum_l2295_229544

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

end base_7_addition_sum_l2295_229544


namespace probability_green_then_blue_l2295_229590

def total_marbles : ℕ := 10
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 6

theorem probability_green_then_blue :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
sorry

end probability_green_then_blue_l2295_229590


namespace laticia_socks_count_l2295_229579

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := first_week + 4

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted over four weeks -/
def total_socks : ℕ := first_week + second_week + third_week + fourth_week

theorem laticia_socks_count : total_socks = 53 := by
  sorry

end laticia_socks_count_l2295_229579


namespace dress_discount_percentage_l2295_229570

/-- Proves that the discount percentage is 15% given the conditions of the dress pricing problem -/
theorem dress_discount_percentage : ∀ (original_price : ℝ) (discount_percentage : ℝ),
  original_price > 0 →
  discount_percentage > 0 →
  discount_percentage < 100 →
  original_price * (1 - discount_percentage / 100) = 68 →
  68 * 1.25 = original_price - 5 →
  discount_percentage = 15 := by
  sorry

end dress_discount_percentage_l2295_229570


namespace at_least_one_trinomial_has_two_roots_l2295_229596

theorem at_least_one_trinomial_has_two_roots 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃) 
  (h2 : b₁ * b₂ * b₃ > 1) : 
  ∃ (i : Fin 3), 
    let f := fun x => x^2 + 2 * ([a₁, a₂, a₃].get i) * x + ([b₁, b₂, b₃].get i)
    (∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end at_least_one_trinomial_has_two_roots_l2295_229596


namespace sum_of_reciprocals_l2295_229575

theorem sum_of_reciprocals (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (hcf_eq : Nat.gcd x y = 3)
  (lcm_eq : Nat.lcm x y = 100) :
  (1 : ℚ) / x + (1 : ℚ) / y = 3 / 20 := by
  sorry

end sum_of_reciprocals_l2295_229575


namespace zeros_not_adjacent_probability_l2295_229528

def total_arrangements : ℕ := Nat.choose 6 2

def non_adjacent_arrangements : ℕ := Nat.choose 5 2

theorem zeros_not_adjacent_probability :
  (non_adjacent_arrangements : ℚ) / total_arrangements = 2 / 3 :=
sorry

end zeros_not_adjacent_probability_l2295_229528


namespace jeans_cost_l2295_229502

theorem jeans_cost (total_cost coat_cost shoe_cost : ℕ) (h1 : total_cost = 110) (h2 : coat_cost = 40) (h3 : shoe_cost = 30) : 
  ∃ (jeans_cost : ℕ), jeans_cost * 2 + coat_cost + shoe_cost = total_cost ∧ jeans_cost = 20 := by
  sorry

end jeans_cost_l2295_229502


namespace arrangements_of_distinct_letters_l2295_229594

-- Define the number of distinct letters
def num_distinct_letters : ℕ := 7

-- Define the function to calculate the number of arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem arrangements_of_distinct_letters : 
  num_arrangements num_distinct_letters = 5040 := by
  sorry

end arrangements_of_distinct_letters_l2295_229594


namespace neither_sufficient_nor_necessary_l2295_229518

theorem neither_sufficient_nor_necessary : ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
                                           ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) := by
  sorry

end neither_sufficient_nor_necessary_l2295_229518


namespace room_height_proof_l2295_229521

/-- Proves that the height of a room with given dimensions and openings is 6 feet -/
theorem room_height_proof (width length : ℝ) (doorway1_width doorway1_height : ℝ)
  (window_width window_height : ℝ) (doorway2_width doorway2_height : ℝ)
  (total_paint_area : ℝ) (h : ℝ) :
  width = 20 ∧ length = 20 ∧
  doorway1_width = 3 ∧ doorway1_height = 7 ∧
  window_width = 6 ∧ window_height = 4 ∧
  doorway2_width = 5 ∧ doorway2_height = 7 ∧
  total_paint_area = 560 ∧
  total_paint_area = 4 * width * h - (doorway1_width * doorway1_height + window_width * window_height + doorway2_width * doorway2_height) →
  h = 6 := by
  sorry

#check room_height_proof

end room_height_proof_l2295_229521


namespace lauras_remaining_pay_l2295_229558

/-- Calculates the remaining amount of Laura's pay after expenses --/
def remaining_pay (hourly_rate : ℚ) (hours_per_day : ℚ) (days_worked : ℚ) 
                  (food_clothing_percentage : ℚ) (rent : ℚ) : ℚ :=
  let total_earnings := hourly_rate * hours_per_day * days_worked
  let food_clothing_expense := total_earnings * food_clothing_percentage
  let remaining_after_food_clothing := total_earnings - food_clothing_expense
  remaining_after_food_clothing - rent

/-- Theorem stating that Laura's remaining pay is $250 --/
theorem lauras_remaining_pay :
  remaining_pay 10 8 10 (1/4) 350 = 250 := by
  sorry

end lauras_remaining_pay_l2295_229558


namespace track_completion_time_l2295_229539

/-- Represents a runner on the circular track -/
structure Runner :=
  (position : ℝ)
  (speed : ℝ)

/-- Represents the circular track -/
structure Track :=
  (circumference : ℝ)
  (runners : List Runner)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℝ)

/-- The main theorem to be proved -/
theorem track_completion_time 
  (track : Track) 
  (meeting1 : Meeting) 
  (meeting2 : Meeting) 
  (meeting3 : Meeting) :
  meeting1.runner1 = meeting2.runner1 ∧ 
  meeting1.runner2 = meeting2.runner2 ∧
  meeting2.runner2 = meeting3.runner1 ∧
  meeting2.runner1 = meeting3.runner2 ∧
  meeting2.time - meeting1.time = 15 ∧
  meeting3.time - meeting2.time = 25 →
  track.circumference = 80 := by
  sorry

end track_completion_time_l2295_229539


namespace max_value_of_f_l2295_229533

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-3/2) 2

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ interval, f a x ≤ 1) ∧
  (∃ x ∈ interval, f a x = 1) ↔
  (a = 3/4 ∨ a = (-3-2*Real.sqrt 2)/2) := by sorry

end max_value_of_f_l2295_229533


namespace bob_salary_last_year_l2295_229573

/-- Mario's salary this year -/
def mario_salary_this_year : ℝ := 4000

/-- Mario's salary increase percentage -/
def mario_increase_percentage : ℝ := 0.40

/-- Bob's salary last year as a multiple of Mario's salary this year -/
def bob_salary_multiple : ℝ := 3

theorem bob_salary_last_year :
  let mario_salary_last_year := mario_salary_this_year / (1 + mario_increase_percentage)
  let bob_salary_last_year := bob_salary_multiple * mario_salary_this_year
  bob_salary_last_year = 12000 := by sorry

end bob_salary_last_year_l2295_229573


namespace ratio_of_powers_l2295_229550

theorem ratio_of_powers : (2^17 * 3^19) / 6^18 = 3/2 := by
  sorry

end ratio_of_powers_l2295_229550


namespace ladder_length_l2295_229585

theorem ladder_length (a b : ℝ) (ha : a = 20) (hb : b = 15) :
  Real.sqrt (a^2 + b^2) = 25 := by
  sorry

end ladder_length_l2295_229585


namespace cos_shift_equivalence_l2295_229565

theorem cos_shift_equivalence (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.cos (2 * (x + π / 6)) := by
  sorry

end cos_shift_equivalence_l2295_229565


namespace quadratic_properties_l2295_229589

def f (x : ℝ) := x^2 - 6*x + 8

theorem quadratic_properties :
  (∀ x, f x = (x - 2) * (x - 4)) ∧
  (∀ x, f x ≥ f 3) ∧
  (f 3 = -1) := by
  sorry

end quadratic_properties_l2295_229589


namespace coefficient_x_squared_expansion_l2295_229546

/-- The coefficient of x^2 in the expansion of (1 + 1/x)(1-x)^7 is -14 -/
theorem coefficient_x_squared_expansion : ℤ := by
  sorry

end coefficient_x_squared_expansion_l2295_229546


namespace systematic_sampling_correct_l2295_229531

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  populationSize : ℕ
  sampleSize : ℕ
  firstItem : ℕ
  samplingInterval : ℕ

/-- Generates the sample based on the systematic sampling scheme -/
def generateSample (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.firstItem + i * s.samplingInterval)

/-- Theorem: The systematic sampling for the given problem yields the correct sample -/
theorem systematic_sampling_correct :
  let s : SystematicSampling := {
    populationSize := 50,
    sampleSize := 5,
    firstItem := 7,
    samplingInterval := 10
  }
  generateSample s = [7, 17, 27, 37, 47] := by
  sorry


end systematic_sampling_correct_l2295_229531


namespace origin_outside_circle_l2295_229583

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2 = 0}
  (0, 0) ∉ circle := by
sorry

end origin_outside_circle_l2295_229583


namespace train_length_calculation_l2295_229599

/-- Calculates the length of a train given its speed, time to cross a platform, and the platform length. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Proves that a train with speed 35 m/s crossing a 250.056 m platform in 20 seconds has a length of 449.944 m. -/
theorem train_length_calculation :
  train_length 35 20 250.056 = 449.944 := by
  sorry

#eval train_length 35 20 250.056

end train_length_calculation_l2295_229599


namespace smaller_hexagon_area_ratio_l2295_229519

/-- A regular hexagon with side length 4 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 4)

/-- Midpoint of a side of the hexagon -/
structure Midpoint :=
  (point : ℝ × ℝ)

/-- The smaller hexagon formed by connecting midpoints of alternating sides -/
structure SmallerHexagon :=
  (vertices : List (ℝ × ℝ))
  (is_regular : Bool)

/-- The ratio of the area of the smaller hexagon to the area of the original hexagon -/
def area_ratio (original : RegularHexagon) (smaller : SmallerHexagon) : ℚ :=
  49/36

theorem smaller_hexagon_area_ratio 
  (original : RegularHexagon) 
  (G H I J K L : Midpoint) 
  (smaller : SmallerHexagon) :
  area_ratio original smaller = 49/36 :=
sorry

end smaller_hexagon_area_ratio_l2295_229519


namespace acute_triangle_tangent_inequality_l2295_229540

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C < π) :
  (1 / 3) * ((Real.tan A)^2 / (Real.tan B * Real.tan C) +
             (Real.tan B)^2 / (Real.tan C * Real.tan A) +
             (Real.tan C)^2 / (Real.tan A * Real.tan B)) +
  3 * (1 / (Real.tan A + Real.tan B + Real.tan C))^(2/3) ≥ 2 := by
sorry

end acute_triangle_tangent_inequality_l2295_229540


namespace y_intercept_range_l2295_229578

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k * x + 1

-- Define the line l passing through (-2, 0) and the midpoint of AB
def line_l (m b x y : ℝ) : Prop := y = m * x + b

-- Define the condition that k is in the valid range
def valid_k (k : ℝ) : Prop := 1 < k ∧ k < Real.sqrt 2

-- Define the range of b
def b_range (b : ℝ) : Prop := b < -2 - Real.sqrt 2 ∨ b > 2

-- Main theorem
theorem y_intercept_range (k m b : ℝ) : 
  valid_k k →
  (∃ x1 y1 x2 y2 : ℝ, 
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    line1 k x1 y1 ∧ line1 k x2 y2 ∧
    x1 < 0 ∧ x2 < 0 ∧
    line_l m b (-2) 0 ∧
    line_l m b ((x1 + x2) / 2) ((y1 + y2) / 2)) →
  b_range b :=
sorry

end y_intercept_range_l2295_229578


namespace sector_central_angle_l2295_229542

/-- Given a circular sector with area 1 and perimeter 4, its central angle is 2 radians -/
theorem sector_central_angle (S : ℝ) (P : ℝ) (α : ℝ) :
  S = 1 →  -- area of the sector
  P = 4 →  -- perimeter of the sector
  S = (1/2) * α * (P - α)^2 / α^2 →  -- area formula for a sector
  α = 2 := by sorry

end sector_central_angle_l2295_229542


namespace cheryl_basil_harvest_l2295_229522

-- Define the variables and constants
def basil_per_pesto : ℝ := 4
def harvest_weeks : ℕ := 8
def total_pesto : ℝ := 32

-- Define the theorem
theorem cheryl_basil_harvest :
  (basil_per_pesto * total_pesto) / harvest_weeks = 16 := by
  sorry

end cheryl_basil_harvest_l2295_229522


namespace irrational_expression_l2295_229572

theorem irrational_expression (x : ℝ) : 
  Irrational ((x - 3 * Real.sqrt (x^2 + 4)) / 2) := by sorry

end irrational_expression_l2295_229572


namespace arrangement_count_is_twelve_l2295_229591

/-- The number of elements to be arranged -/
def n : ℕ := 4

/-- The condition that A is adjacent to B -/
def adjacent_condition : Prop := true  -- We don't need to define this explicitly in Lean

/-- The number of ways to arrange n elements with the adjacent condition -/
def arrangement_count (n : ℕ) (adjacent_condition : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangement_count_is_twelve :
  arrangement_count n adjacent_condition = 12 := by sorry

end arrangement_count_is_twelve_l2295_229591


namespace smallest_prime_perimeter_scalene_triangle_l2295_229556

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a < b ∧ b < c →  -- scalene condition
    is_prime a ∧ is_prime b ∧ is_prime c →  -- prime side lengths
    a = 5 →  -- smallest side is 5
    triangle_inequality a b c →  -- valid triangle
    is_prime (a + b + c) →  -- prime perimeter
    a + b + c ≥ 23 :=
sorry

end smallest_prime_perimeter_scalene_triangle_l2295_229556


namespace drink_volume_theorem_l2295_229516

/-- Represents the parts of each ingredient in the drink recipe. -/
structure DrinkRecipe where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ
  drPepper : ℕ
  fanta : ℕ

/-- Calculates the total parts in a drink recipe. -/
def totalParts (recipe : DrinkRecipe) : ℕ :=
  recipe.coke + recipe.sprite + recipe.mountainDew + recipe.drPepper + recipe.fanta

/-- Theorem stating that given the specific drink recipe and the amount of Coke,
    the total volume of the drink is 48 ounces. -/
theorem drink_volume_theorem (recipe : DrinkRecipe)
    (h1 : recipe.coke = 4)
    (h2 : recipe.sprite = 2)
    (h3 : recipe.mountainDew = 5)
    (h4 : recipe.drPepper = 3)
    (h5 : recipe.fanta = 2)
    (h6 : 12 = recipe.coke * 3) :
    (totalParts recipe) * 3 = 48 := by
  sorry

end drink_volume_theorem_l2295_229516


namespace mark_sprint_distance_l2295_229554

/-- The distance traveled by Mark given his sprint time and speed -/
theorem mark_sprint_distance (time : ℝ) (speed : ℝ) (h1 : time = 24.0) (h2 : speed = 6.0) :
  time * speed = 144.0 := by
  sorry

end mark_sprint_distance_l2295_229554


namespace phantoms_initial_money_l2295_229506

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

end phantoms_initial_money_l2295_229506


namespace xiao_ming_tasks_minimum_time_l2295_229571

def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

def minimum_time : ℕ := 85

theorem xiao_ming_tasks_minimum_time :
  minimum_time = max review_time (max rest_time homework_time) :=
by sorry

end xiao_ming_tasks_minimum_time_l2295_229571


namespace well_digging_payment_l2295_229530

/-- The total amount paid to two workers for digging a well --/
def total_amount_paid (hours_day1 hours_day2 hours_day3 : ℕ) (hourly_rate : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := 2 * total_hours
  total_man_hours * hourly_rate

/-- Theorem stating that the total amount paid is $660 --/
theorem well_digging_payment :
  total_amount_paid 10 8 15 10 = 660 := by
  sorry

end well_digging_payment_l2295_229530
