import Mathlib

namespace height_to_sphere_ratio_l1004_100457

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  H : ℝ  -- height of the truncated cone
  s : ℝ  -- radius of the inscribed sphere
  R_positive : R > 0
  r_positive : r > 0
  H_positive : H > 0
  s_positive : s > 0
  sphere_inscribed : s = Real.sqrt (R * r)
  volume_relation : π * H * (R^2 + R*r + r^2) / 3 = 4 * π * s^3

/-- The ratio of the height of the truncated cone to the radius of the sphere is 4 -/
theorem height_to_sphere_ratio (cone : TruncatedConeWithSphere) : 
  cone.H / cone.s = 4 := by
  sorry

end height_to_sphere_ratio_l1004_100457


namespace intersection_of_circles_l1004_100426

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define the theorem
theorem intersection_of_circles (r : ℝ) (hr : r > 0) :
  (∃! p, p ∈ A ∩ B r) → r = 3 ∨ r = 7 := by
  sorry


end intersection_of_circles_l1004_100426


namespace division_result_l1004_100450

theorem division_result : (4 : ℚ) / (8 / 13) = 13 / 2 := by sorry

end division_result_l1004_100450


namespace new_students_count_l1004_100435

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) :
  initial_students = 10 →
  left_students = 4 →
  final_students = 48 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end new_students_count_l1004_100435


namespace total_players_l1004_100485

/-- The total number of players in a game scenario -/
theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) :
  kabaddi = 10 →
  kho_kho_only = 15 →
  both = 5 →
  kabaddi + kho_kho_only - both = 25 := by
sorry

end total_players_l1004_100485


namespace product_evaluation_l1004_100493

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l1004_100493


namespace dogs_not_liking_any_food_l1004_100470

/-- Given a kennel of dogs with specified food preferences, prove the number of dogs
    that don't like any of watermelon, salmon, or chicken. -/
theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ)
  (watermelon_and_salmon watermelon_and_chicken_not_salmon salmon_and_chicken_not_watermelon : ℕ)
  (h1 : total = 80)
  (h2 : watermelon = 21)
  (h3 : salmon = 58)
  (h4 : watermelon_and_salmon = 12)
  (h5 : chicken = 15)
  (h6 : watermelon_and_chicken_not_salmon = 7)
  (h7 : salmon_and_chicken_not_watermelon = 10) :
  total - (watermelon_and_salmon + (salmon - watermelon_and_salmon - salmon_and_chicken_not_watermelon) +
           (watermelon - watermelon_and_salmon - watermelon_and_chicken_not_salmon) +
           salmon_and_chicken_not_watermelon + watermelon_and_chicken_not_salmon) = 13 := by
  sorry

end dogs_not_liking_any_food_l1004_100470


namespace one_pair_three_different_probability_l1004_100465

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def probability_one_pair_three_different : ℚ :=
  27 / 66

theorem one_pair_three_different_probability :
  (total_socks = socks_per_color * num_colors) →
  (probability_one_pair_three_different =
    (num_colors * (socks_per_color.choose 2) *
     (socks_per_color ^ (num_colors - 1))) /
    (total_socks.choose drawn_socks)) :=
by sorry

end one_pair_three_different_probability_l1004_100465


namespace rachel_homework_l1004_100462

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 10 →
  math_pages + reading_pages = 23 →
  reading_pages > math_pages →
  reading_pages - math_pages = 3 :=
by sorry

end rachel_homework_l1004_100462


namespace fraction_equality_l1004_100479

theorem fraction_equality (a b : ℝ) (h : a/b = 2) : a/(a-b) = 2 := by
  sorry

end fraction_equality_l1004_100479


namespace julia_number_l1004_100458

theorem julia_number (j m : ℂ) : 
  j * m = 48 - 24*I → 
  m = 7 + 4*I → 
  j = 432/65 - 360/65*I := by sorry

end julia_number_l1004_100458


namespace babysitting_cost_difference_l1004_100414

/-- Represents the babysitting scenario with given rates and conditions -/
structure BabysittingScenario where
  current_rate : ℕ -- Rate of current babysitter in dollars per hour
  new_base_rate : ℕ -- Base rate of new babysitter in dollars per hour
  new_scream_charge : ℕ -- Extra charge for each scream by new babysitter
  hours : ℕ -- Number of hours of babysitting
  screams : ℕ -- Number of times kids scream during babysitting

/-- Calculates the cost difference between current and new babysitter -/
def costDifference (scenario : BabysittingScenario) : ℕ :=
  scenario.current_rate * scenario.hours - 
  (scenario.new_base_rate * scenario.hours + scenario.new_scream_charge * scenario.screams)

/-- Theorem stating the cost difference for the given scenario -/
theorem babysitting_cost_difference :
  ∃ (scenario : BabysittingScenario),
    scenario.current_rate = 16 ∧
    scenario.new_base_rate = 12 ∧
    scenario.new_scream_charge = 3 ∧
    scenario.hours = 6 ∧
    scenario.screams = 2 ∧
    costDifference scenario = 18 := by
  sorry

end babysitting_cost_difference_l1004_100414


namespace xy_greater_than_xz_l1004_100441

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z := by sorry

end xy_greater_than_xz_l1004_100441


namespace max_teams_in_tournament_l1004_100425

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- Calculates the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n = 7 ∧ 
  total_games n ≤ max_games ∧ 
  ∀ (m : ℕ), m > n → total_games m > max_games :=
sorry

end max_teams_in_tournament_l1004_100425


namespace marcus_points_l1004_100495

/-- Proves that Marcus scored 28 points in the basketball game -/
theorem marcus_points (total_points : ℕ) (other_players : ℕ) (avg_points : ℕ) : 
  total_points = 63 → other_players = 5 → avg_points = 7 → 
  total_points - (other_players * avg_points) = 28 := by
  sorry

end marcus_points_l1004_100495


namespace pentagon_rectangle_angle_sum_l1004_100409

/-- The sum of an interior angle of a regular pentagon and an interior angle of a rectangle is 198°. -/
theorem pentagon_rectangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let rectangle_angle : ℝ := 90
  pentagon_angle + rectangle_angle = 198 := by sorry

end pentagon_rectangle_angle_sum_l1004_100409


namespace hyperbola_eccentricity_l1004_100433

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  let parabola := fun x y : ℝ => y^2 = 8*x
  let hyperbola := fun x y : ℝ => x^2/a^2 - y^2 = 1
  let focus : ℝ × ℝ := (2, 0)
  (∀ x y, parabola x y → (x, y) = focus) →
  hyperbola (focus.1) (focus.2) →
  let e := Real.sqrt ((a^2 + a^2) / a^2)
  e = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l1004_100433


namespace triangle_equilateral_if_angles_arithmetic_and_geometric_l1004_100430

theorem triangle_equilateral_if_angles_arithmetic_and_geometric :
  ∀ (a b c : ℝ),
  -- The angles form an arithmetic sequence
  (∃ d : ℝ, b = a + d ∧ c = b + d) →
  -- The angles form a geometric sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  -- The sum of angles is 180°
  a + b + c = 180 →
  -- The triangle is equilateral (all angles are equal)
  a = b ∧ b = c := by
sorry

end triangle_equilateral_if_angles_arithmetic_and_geometric_l1004_100430


namespace transformer_load_calculation_l1004_100487

/-- Calculates the minimum current load for a transformer given the number of units,
    running current per unit, and the starting current multiplier. -/
def minTransformerLoad (numUnits : ℕ) (runningCurrent : ℕ) (startingMultiplier : ℕ) : ℕ :=
  numUnits * (startingMultiplier * runningCurrent)

theorem transformer_load_calculation :
  let numUnits : ℕ := 3
  let runningCurrent : ℕ := 40
  let startingMultiplier : ℕ := 2
  minTransformerLoad numUnits runningCurrent startingMultiplier = 240 := by
  sorry

#eval minTransformerLoad 3 40 2

end transformer_load_calculation_l1004_100487


namespace chicken_surprise_serving_weight_l1004_100454

/-- Represents the recipe for Chicken Surprise -/
structure ChickenSurprise where
  servings : ℕ
  chickenPounds : ℚ
  stuffingOunces : ℕ

/-- Calculates the weight of one serving of Chicken Surprise in ounces -/
def servingWeight (recipe : ChickenSurprise) : ℚ :=
  let totalOunces := recipe.chickenPounds * 16 + recipe.stuffingOunces
  totalOunces / recipe.servings

/-- Theorem stating that one serving of Chicken Surprise is 8 ounces -/
theorem chicken_surprise_serving_weight :
  let recipe := ChickenSurprise.mk 12 (9/2) 24
  servingWeight recipe = 8 := by
  sorry

end chicken_surprise_serving_weight_l1004_100454


namespace coefficients_of_equation_l1004_100484

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation 3x^2 - x - 2 = 0 -/
def equation : QuadraticEquation :=
  { a := 3, b := -1, c := -2 }

theorem coefficients_of_equation :
  equation.a = 3 ∧ equation.b = -1 ∧ equation.c = -2 := by
  sorry

end coefficients_of_equation_l1004_100484


namespace imaginary_part_of_complex_division_l1004_100471

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁ = 2 - I → z₂ = 1 - 3*I → Complex.im (z₂ / z₁) = -1 := by
  sorry

end imaginary_part_of_complex_division_l1004_100471


namespace equation_solutions_l1004_100436

theorem equation_solutions :
  (∃ x : ℝ, (3 + x) * (30 / 100) = 4.8 ∧ x = 13) ∧
  (∃ x : ℝ, 5 / x = (9 / 2) / (8 / 5) ∧ x = 16 / 9) :=
by sorry

end equation_solutions_l1004_100436


namespace fifteenth_term_of_sequence_l1004_100446

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ : ℤ) (h : a₂ = a₁ + 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 53 :=
by sorry

end fifteenth_term_of_sequence_l1004_100446


namespace equilateral_triangle_construction_l1004_100489

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def rotatePoint (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

theorem equilateral_triangle_construction 
  (A : ℝ × ℝ) (S₁ S₂ : Circle) : 
  ∃ (B C : ℝ × ℝ), 
    (∃ (t : ℝ), B = rotatePoint C A (-π/3)) ∧
    (∃ (t : ℝ), C = rotatePoint B A (π/3)) ∧
    (∃ (t : ℝ), B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2}) ∧
    (∃ (t : ℝ), C ∈ {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}) :=
by sorry

end equilateral_triangle_construction_l1004_100489


namespace sam_winning_probability_l1004_100424

theorem sam_winning_probability :
  let hit_probability : ℚ := 2/5
  let miss_probability : ℚ := 3/5
  let p : ℚ := p -- p represents the probability of Sam winning
  (hit_probability = 2/5) →
  (miss_probability = 3/5) →
  (p = hit_probability + miss_probability * miss_probability * p) →
  p = 5/8 := by
sorry

end sam_winning_probability_l1004_100424


namespace winning_candidate_votes_l1004_100448

/-- Proves that the winning candidate received 11628 votes in the described election scenario -/
theorem winning_candidate_votes :
  let total_votes : ℝ := (4136 + 7636) / (1 - 0.4969230769230769)
  let winning_votes : ℝ := 0.4969230769230769 * total_votes
  ⌊winning_votes⌋ = 11628 := by
  sorry

end winning_candidate_votes_l1004_100448


namespace base8_digit_sum_l1004_100452

/-- Represents a digit in base 8 -/
def Digit8 : Type := { n : ℕ // n > 0 ∧ n < 8 }

/-- Converts a three-digit number in base 8 to its decimal equivalent -/
def toDecimal (p q r : Digit8) : ℕ := 64 * p.val + 8 * q.val + r.val

/-- The sum of three permutations of digits in base 8 -/
def sumPermutations (p q r : Digit8) : ℕ :=
  toDecimal p q r + toDecimal r q p + toDecimal q p r

/-- The value of PPP0 in base 8 -/
def ppp0 (p : Digit8) : ℕ := 512 * p.val + 64 * p.val + 8 * p.val

theorem base8_digit_sum (p q r : Digit8) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_sum : sumPermutations p q r = ppp0 p) : 
  q.val + r.val = 7 := by sorry

end base8_digit_sum_l1004_100452


namespace thirty_percent_less_than_ninety_l1004_100483

theorem thirty_percent_less_than_ninety (x : ℝ) : x + x / 2 = 63 → x = 42 := by
  sorry

end thirty_percent_less_than_ninety_l1004_100483


namespace overtime_pay_is_correct_l1004_100482

/-- Represents the time interval between minute and hour hand overlaps on a normal clock in minutes -/
def normal_overlap : ℚ := 720 / 11

/-- Represents the time interval between minute and hour hand overlaps on the slow clock in minutes -/
def slow_overlap : ℕ := 69

/-- Represents the normal workday duration in hours -/
def normal_workday : ℕ := 8

/-- Represents the regular hourly pay rate in dollars -/
def regular_rate : ℚ := 4

/-- Represents the overtime pay rate multiplier -/
def overtime_multiplier : ℚ := 3/2

/-- Theorem stating that the overtime pay is $2.60 given the specified conditions -/
theorem overtime_pay_is_correct :
  let actual_time_ratio : ℚ := slow_overlap / normal_overlap
  let actual_time_worked : ℚ := normal_workday * actual_time_ratio
  let overtime_hours : ℚ := actual_time_worked - normal_workday
  let overtime_pay : ℚ := overtime_hours * regular_rate * overtime_multiplier
  overtime_pay = 13/5 := by sorry

end overtime_pay_is_correct_l1004_100482


namespace area_bisectors_perpendicular_l1004_100438

/-- Triangle with two sides of length 6 and one side of length 8 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b ∧ a = 6 ∧ c = 8

/-- Area bisector of a triangle -/
def AreaBisector (t : IsoscelesTriangle) := ℝ → ℝ

/-- The angle between two lines -/
def AngleBetween (l1 l2 : ℝ → ℝ) : ℝ := sorry

theorem area_bisectors_perpendicular (t : IsoscelesTriangle) 
  (b1 b2 : AreaBisector t) (h : b1 ≠ b2) : 
  AngleBetween b1 b2 = π / 2 := by sorry

end area_bisectors_perpendicular_l1004_100438


namespace cube_surface_area_from_volume_l1004_100466

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side_length ^ 3 → 
  surface_area = 6 * side_length ^ 2 → 
  surface_area = 486 := by
sorry

end cube_surface_area_from_volume_l1004_100466


namespace age_order_l1004_100445

structure Person where
  name : String
  age : ℕ

def age_relationship (sergei sasha tolia : Person) : Prop :=
  sergei.age = 2 * (sergei.age + tolia.age - sergei.age)

theorem age_order (sergei sasha tolia : Person) 
  (h : age_relationship sergei sasha tolia) : 
  sergei.age > tolia.age ∧ tolia.age > sasha.age :=
by
  sorry

#check age_order

end age_order_l1004_100445


namespace inequality_solution_set_l1004_100496

theorem inequality_solution_set (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 0 ↔ x ∈ {y : ℝ | 1/3 ≤ y ∧ y < 2} :=
sorry

end inequality_solution_set_l1004_100496


namespace largest_negative_integer_l1004_100494

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
sorry

end largest_negative_integer_l1004_100494


namespace slope_range_for_inclination_angle_l1004_100405

theorem slope_range_for_inclination_angle (α : Real) :
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 →
  ∃ k : Real, (k < -1 ∨ k = -1 ∨ k = 1 ∨ k > 1) ∧ k = Real.tan α :=
sorry

end slope_range_for_inclination_angle_l1004_100405


namespace mandy_gets_fifteen_l1004_100460

def chocolate_bar : ℕ := 60

def michael_share (total : ℕ) : ℕ := total / 2

def paige_share (remaining : ℕ) : ℕ := remaining / 2

def mandy_share (total : ℕ) : ℕ :=
  let after_michael := total - michael_share total
  after_michael - paige_share after_michael

theorem mandy_gets_fifteen :
  mandy_share chocolate_bar = 15 := by
  sorry

end mandy_gets_fifteen_l1004_100460


namespace paving_stone_size_l1004_100440

theorem paving_stone_size (length width : ℝ) (num_stones : ℕ) (stone_side : ℝ) : 
  length = 30 → 
  width = 18 → 
  num_stones = 135 → 
  (length * width) = (num_stones : ℝ) * stone_side^2 → 
  stone_side = 2 := by
  sorry

end paving_stone_size_l1004_100440


namespace parallel_vectors_y_value_l1004_100447

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, -1 + y)
  are_parallel a b → y = 7 := by
  sorry

end parallel_vectors_y_value_l1004_100447


namespace lcm_gcd_problem_l1004_100468

theorem lcm_gcd_problem (x y : ℕ+) : 
  Nat.lcm x y = 5940 → 
  Nat.gcd x y = 22 → 
  x = 220 → 
  y = 594 := by
sorry

end lcm_gcd_problem_l1004_100468


namespace two_cyclists_problem_l1004_100463

/-- Two cyclists problem -/
theorem two_cyclists_problem (MP : ℝ) : 
  (∀ (t : ℝ), t > 0 → 
    (MP / t = 42 / ((420 / (MP + 30)) + 1/3)) ∧
    (MP + 30) / t = 42 / (420 / MP)) →
  MP = 180 := by
sorry

end two_cyclists_problem_l1004_100463


namespace max_intersections_theorem_l1004_100439

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure TwoPolygonConfig where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  same_plane : Bool
  m₁_le_m₂ : Q₁.sides ≤ Q₂.sides
  share_at_most_one_vertex : Bool
  share_no_sides : Bool

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : TwoPolygonConfig) : ℕ := 
  config.Q₁.sides * config.Q₂.sides

/-- Theorem: The maximum number of intersections between two convex polygons
    under the given conditions is the product of their number of sides -/
theorem max_intersections_theorem (config : TwoPolygonConfig) : 
  max_intersections config = config.Q₁.sides * config.Q₂.sides := by
  sorry

end max_intersections_theorem_l1004_100439


namespace geometric_sequence_differences_l1004_100474

/-- The type of sequences of real numbers of length n -/
def RealSequence (n : ℕ) := Fin n → ℝ

/-- The condition that a sequence is strictly increasing -/
def StrictlyIncreasing {n : ℕ} (a : RealSequence n) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- The set of differences between elements of a sequence -/
def Differences {n : ℕ} (a : RealSequence n) : Set ℝ :=
  {x : ℝ | ∃ i j : Fin n, i < j ∧ x = a j - a i}

/-- The set of powers of r from 1 to k -/
def PowerSet (r : ℝ) (k : ℕ) : Set ℝ :=
  {x : ℝ | ∃ m : ℕ, m ≤ k ∧ x = r ^ m}

/-- The main theorem -/
theorem geometric_sequence_differences (n : ℕ) (h : n ≥ 2) :
  (∃ (a : RealSequence n) (r : ℝ),
    StrictlyIncreasing a ∧
    r > 0 ∧
    Differences a = PowerSet r (n * (n - 1) / 2)) ↔
  n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end geometric_sequence_differences_l1004_100474


namespace not_divisible_by_two_or_five_l1004_100429

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 3)^2 + (n - 1)^2 + (n + 1)^2 + (n + 3)^2}

theorem not_divisible_by_two_or_five :
  ∀ x ∈ T, ¬(∃ k : ℤ, x = 2 * k ∨ x = 5 * k) :=
by sorry

end not_divisible_by_two_or_five_l1004_100429


namespace problem_solution_l1004_100443

def A (a : ℝ) := { x : ℝ | a - 1 ≤ x ∧ x ≤ a + 1 }
def B := { x : ℝ | -1 ≤ x ∧ x ≤ 4 }

theorem problem_solution :
  (∀ a : ℝ, a = 2 → A a ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 4 }) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∪ B = B → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∩ B = ∅ → a < -2 ∨ a > 5) :=
by sorry

end problem_solution_l1004_100443


namespace target_probabilities_l1004_100459

/-- Probability of hitting a target -/
structure TargetProbability where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Assumptions about the probabilities -/
axiom prob_bounds (p : TargetProbability) :
  0 ≤ p.A ∧ p.A ≤ 1 ∧
  0 ≤ p.B ∧ p.B ≤ 1 ∧
  0 ≤ p.C ∧ p.C ≤ 1

/-- Given probabilities -/
def given_probs : TargetProbability :=
  { A := 0.7, B := 0.6, C := 0.5 }

/-- Probability of at least one person hitting the target -/
def prob_at_least_one (p : TargetProbability) : ℝ :=
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C)

/-- Probability of exactly two people hitting the target -/
def prob_exactly_two (p : TargetProbability) : ℝ :=
  p.A * p.B * (1 - p.C) + p.A * (1 - p.B) * p.C + (1 - p.A) * p.B * p.C

/-- Probability of hitting exactly k times in n trials -/
def prob_k_of_n (p q : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * q^(n - k)

theorem target_probabilities (p : TargetProbability) 
  (h : p = given_probs) : 
  prob_at_least_one p = 0.94 ∧ 
  prob_exactly_two p = 0.44 ∧ 
  prob_k_of_n p.A (1 - p.A) 3 2 = 0.441 := by
  sorry

end target_probabilities_l1004_100459


namespace cost_of_seven_sandwiches_six_sodas_l1004_100490

/-- Calculates the total cost of purchasing sandwiches and sodas at Sally's Snack Shop -/
def snack_shop_cost (sandwich_count : ℕ) (soda_count : ℕ) : ℕ :=
  let sandwich_price := 4
  let soda_price := 3
  let bulk_discount := 10
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_price + soda_count * soda_price
  if total_items > 10 then total_cost - bulk_discount else total_cost

/-- Theorem stating that purchasing 7 sandwiches and 6 sodas costs $36 -/
theorem cost_of_seven_sandwiches_six_sodas :
  snack_shop_cost 7 6 = 36 := by
  sorry

end cost_of_seven_sandwiches_six_sodas_l1004_100490


namespace fifty_roses_cost_l1004_100472

/-- The cost of a bouquet of roses, given the number of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  let baseCost := 24 * roses / 12
  if roses ≥ 45 then baseCost * (1 - 1/10) else baseCost

theorem fifty_roses_cost :
  bouquetCost 50 = 90 := by sorry

end fifty_roses_cost_l1004_100472


namespace password_probability_l1004_100413

def even_two_digit_numbers : ℕ := 45
def vowels : ℕ := 5
def total_letters : ℕ := 26
def prime_two_digit_numbers : ℕ := 21
def total_two_digit_numbers : ℕ := 90

theorem password_probability :
  (even_two_digit_numbers / total_two_digit_numbers) *
  (vowels / total_letters) *
  (prime_two_digit_numbers / total_two_digit_numbers) =
  7 / 312 := by sorry

end password_probability_l1004_100413


namespace repeating_decimal_interval_l1004_100442

/-- A number is a repeating decimal with period p if it can be expressed as m / (10^p - 1) for some integer m. -/
def is_repeating_decimal (x : ℚ) (p : ℕ) : Prop :=
  ∃ (m : ℤ), x = m / (10^p - 1)

theorem repeating_decimal_interval :
  ∀ n : ℕ,
    n < 2000 →
    is_repeating_decimal (1 / n) 8 →
    is_repeating_decimal (1 / (n + 6)) 6 →
    801 ≤ n ∧ n ≤ 1200 := by
  sorry

end repeating_decimal_interval_l1004_100442


namespace remainder_problem_l1004_100410

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 2 := by
  sorry

end remainder_problem_l1004_100410


namespace apartment_count_l1004_100427

theorem apartment_count (total_keys : ℕ) (keys_per_apartment : ℕ) (num_complexes : ℕ) :
  total_keys = 72 →
  keys_per_apartment = 3 →
  num_complexes = 2 →
  ∃ (apartments_per_complex : ℕ), 
    apartments_per_complex * keys_per_apartment * num_complexes = total_keys ∧
    apartments_per_complex = 12 :=
by
  sorry

end apartment_count_l1004_100427


namespace water_jars_count_l1004_100481

/-- Proves that 7 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars results in 12 water-filled jars -/
theorem water_jars_count (total_water : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_water = 7 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  ∃ (x : ℚ), x > 0 ∧ x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2) = total_water ∧
  (3 * x : ℚ) = 12 :=
by sorry

end water_jars_count_l1004_100481


namespace road_project_completion_time_l1004_100451

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  initialWorkers : ℕ
  daysWorked : ℝ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total number of days required to complete the road project -/
def totalDaysRequired (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that given the project conditions, it will be completed in 15 days -/
theorem road_project_completion_time (project : RoadProject)
  (h1 : project.totalLength = 10)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysWorked = 5)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  totalDaysRequired project = 15 :=
sorry

end road_project_completion_time_l1004_100451


namespace log_simplification_l1004_100418

theorem log_simplification (p q r s z u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hz : z > 0) (hu : u > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * u)) = Real.log (u / z) := by
  sorry

end log_simplification_l1004_100418


namespace line_circle_intersection_count_l1004_100453

/-- The number of intersection points between a line and a circle -/
theorem line_circle_intersection_count (k : ℝ) : 
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
  (∀ (x y : ℝ), (k * x - y - k = 0 ∧ x^2 + y^2 = 2) ↔ (x, y) = p ∨ (x, y) = q) :=
sorry

end line_circle_intersection_count_l1004_100453


namespace math_majors_consecutive_probability_l1004_100431

-- Define the total number of people and the number of math majors
def total_people : ℕ := 12
def math_majors : ℕ := 5

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the probability of math majors sitting consecutively
def prob_consecutive_math_majors : ℚ := (total_people : ℚ) / (choose total_people math_majors : ℚ)

-- State the theorem
theorem math_majors_consecutive_probability :
  prob_consecutive_math_majors = 1 / 66 :=
sorry

end math_majors_consecutive_probability_l1004_100431


namespace compare_fractions_l1004_100497

theorem compare_fractions : -3/2 > -(5/3) := by sorry

end compare_fractions_l1004_100497


namespace katie_earnings_l1004_100432

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money earned by Katie -/
def total_money : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_money = 21 := by
  sorry

end katie_earnings_l1004_100432


namespace square_sum_given_sum_and_product_l1004_100401

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 32 := by
sorry

end square_sum_given_sum_and_product_l1004_100401


namespace pool_cost_per_person_l1004_100464

theorem pool_cost_per_person
  (total_earnings : ℝ)
  (num_people : ℕ)
  (amount_left : ℝ)
  (h1 : total_earnings = 30)
  (h2 : num_people = 10)
  (h3 : amount_left = 5) :
  (total_earnings - amount_left) / num_people = 2.5 := by
  sorry

end pool_cost_per_person_l1004_100464


namespace room_length_calculation_l1004_100476

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 12.0 ∧ width = 8.0 ∧ area = width * length → length = 1.5 := by
  sorry

end room_length_calculation_l1004_100476


namespace perpendicular_lines_a_values_l1004_100400

-- Define the coefficients of the two lines as functions of a
def line1_coeff (a : ℝ) : ℝ × ℝ := (1 - a, a)
def line2_coeff (a : ℝ) : ℝ × ℝ := (2*a + 3, a - 1)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (line1_coeff a).1 * (line2_coeff a).1 + (line1_coeff a).2 * (line2_coeff a).2 = 0

-- State the theorem
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 1 ∨ a = -3 :=
by
  sorry

end perpendicular_lines_a_values_l1004_100400


namespace barrel_filling_time_l1004_100421

theorem barrel_filling_time (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  y - x = 1/4 ∧ 
  66/y - 40/x = 3 →
  40/x = 5 ∨ 40/x = 96 :=
by sorry

end barrel_filling_time_l1004_100421


namespace toy_store_inventory_l1004_100416

structure Toy where
  name : String
  week1_sales : ℕ
  week2_sales : ℕ
  remaining : ℕ

def initial_stock (t : Toy) : ℕ :=
  t.remaining + t.week1_sales + t.week2_sales

theorem toy_store_inventory (action_figures board_games puzzles stuffed_animals : Toy) 
  (h1 : action_figures.name = "Action Figures" ∧ action_figures.week1_sales = 38 ∧ action_figures.week2_sales = 26 ∧ action_figures.remaining = 19)
  (h2 : board_games.name = "Board Games" ∧ board_games.week1_sales = 27 ∧ board_games.week2_sales = 15 ∧ board_games.remaining = 8)
  (h3 : puzzles.name = "Puzzles" ∧ puzzles.week1_sales = 43 ∧ puzzles.week2_sales = 39 ∧ puzzles.remaining = 12)
  (h4 : stuffed_animals.name = "Stuffed Animals" ∧ stuffed_animals.week1_sales = 20 ∧ stuffed_animals.week2_sales = 18 ∧ stuffed_animals.remaining = 30) :
  initial_stock action_figures = 83 ∧ 
  initial_stock board_games = 50 ∧ 
  initial_stock puzzles = 94 ∧ 
  initial_stock stuffed_animals = 68 := by
  sorry

end toy_store_inventory_l1004_100416


namespace train_crossing_time_l1004_100499

/-- The time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 72 → 
  slower_speed = 36 → 
  train_length = 100 → 
  (train_length / ((faster_speed - slower_speed) * (5/18))) = 10 := by
  sorry

end train_crossing_time_l1004_100499


namespace perpendicular_unit_vector_l1004_100407

/-- Given a vector a = (2, 1), prove that (√5/5, -2√5/5) is a unit vector perpendicular to a. -/
theorem perpendicular_unit_vector (a : ℝ × ℝ) (h : a = (2, 1)) :
  let b : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ (b.1^2 + b.2^2 = 1) :=
by sorry

end perpendicular_unit_vector_l1004_100407


namespace complex_equation_solution_l1004_100419

theorem complex_equation_solution (z : ℂ) 
  (h : 12 * Complex.abs z ^ 2 = 2 * Complex.abs (z + 2) ^ 2 + Complex.abs (z ^ 2 + 1) ^ 2 + 31) :
  z + 6 / z = -2 := by
  sorry

end complex_equation_solution_l1004_100419


namespace not_right_triangle_l1004_100408

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end not_right_triangle_l1004_100408


namespace intersection_M_N_l1004_100417

def M : Set ℝ := {x | x / (x - 1) ≥ 0}

def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end intersection_M_N_l1004_100417


namespace polynomial_absolute_value_l1004_100423

/-- A second-degree polynomial with real coefficients -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The absolute value of f at 1, 2, and 3 is equal to 9 -/
def AbsValueCondition (f : ℝ → ℝ) : Prop :=
  |f 1| = 9 ∧ |f 2| = 9 ∧ |f 3| = 9

theorem polynomial_absolute_value (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : AbsValueCondition f) : 
  |f 0| = 9 := by
  sorry

end polynomial_absolute_value_l1004_100423


namespace unique_A_for_3AA1_multiple_of_9_l1004_100498

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def four_digit_3AA1 (A : ℕ) : ℕ := 3000 + 100 * A + 10 * A + 1

theorem unique_A_for_3AA1_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_3AA1 A) ∧ A = 7 := by
sorry

end unique_A_for_3AA1_multiple_of_9_l1004_100498


namespace max_items_for_alex_washing_l1004_100478

/-- Represents a washing machine with its characteristics and items to wash -/
structure WashingMachine where
  total_items : ℕ
  cycle_duration : ℕ  -- in minutes
  total_wash_time : ℕ  -- in minutes

/-- Calculates the maximum number of items that can be washed per cycle -/
def max_items_per_cycle (wm : WashingMachine) : ℕ :=
  wm.total_items / (wm.total_wash_time / wm.cycle_duration)

/-- Theorem stating the maximum number of items per cycle for the given washing machine -/
theorem max_items_for_alex_washing (wm : WashingMachine) 
  (h1 : wm.total_items = 60)
  (h2 : wm.cycle_duration = 45)
  (h3 : wm.total_wash_time = 180) :
  max_items_per_cycle wm = 15 := by
  sorry

end max_items_for_alex_washing_l1004_100478


namespace angle_30_less_than_complement_l1004_100444

theorem angle_30_less_than_complement (x : ℝ) : x = 90 - x - 30 → x = 30 := by
  sorry

end angle_30_less_than_complement_l1004_100444


namespace no_real_solutions_cube_root_equation_l1004_100488

theorem no_real_solutions_cube_root_equation :
  ¬∃ x : ℝ, (x ^ (1/3 : ℝ)) = 15 / (6 - x ^ (1/3 : ℝ)) := by
  sorry

end no_real_solutions_cube_root_equation_l1004_100488


namespace point_between_parallel_lines_l1004_100480

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 6 * x - 8 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define what it means for a point to be between two lines
def between_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ ¬line2 x y) ∨ (¬line1 x y ∧ line2 x y) ∨ (¬line1 x y ∧ ¬line2 x y)

-- Theorem statement
theorem point_between_parallel_lines :
  between_lines 5 b → b = 4 := by sorry

end point_between_parallel_lines_l1004_100480


namespace cans_bought_with_euros_l1004_100437

/-- The number of cans of soda that can be bought for a given amount of euros. -/
def cans_per_euros (T R E : ℚ) : ℚ :=
  (5 * E * T) / R

/-- Given that T cans of soda can be purchased for R quarters,
    and 1 euro is equivalent to 5 quarters,
    the number of cans of soda that can be bought for E euros is (5ET)/R -/
theorem cans_bought_with_euros (T R E : ℚ) (hT : T > 0) (hR : R > 0) (hE : E ≥ 0) :
  cans_per_euros T R E = (5 * E * T) / R :=
by sorry

end cans_bought_with_euros_l1004_100437


namespace smallest_square_side_l1004_100402

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  total : ℕ
  unit_squares : ℕ
  other_squares : List ℕ

/-- Checks if a partition is valid for a given square -/
def is_valid_partition (s : Square) (p : Partition) : Prop :=
  p.total = 15 ∧
  p.unit_squares = 12 ∧
  p.other_squares.length = 3 ∧
  (p.unit_squares + p.other_squares.sum) = s.side * s.side ∧
  ∀ x ∈ p.other_squares, x > 0

/-- The theorem stating the smallest possible square side length -/
theorem smallest_square_side : 
  ∃ (s : Square) (p : Partition), 
    is_valid_partition s p ∧ 
    (∀ (s' : Square) (p' : Partition), is_valid_partition s' p' → s.side ≤ s'.side) ∧
    s.side = 5 := by
  sorry

end smallest_square_side_l1004_100402


namespace cards_given_to_jeff_l1004_100403

/-- Nell's initial number of cards -/
def nell_initial : ℕ := 528

/-- Nell's remaining number of cards -/
def nell_remaining : ℕ := 252

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := nell_initial - nell_remaining

theorem cards_given_to_jeff : cards_given = 276 := by
  sorry

end cards_given_to_jeff_l1004_100403


namespace geometric_sequence_sixth_term_geometric_sequence_general_term_l1004_100477

/-- Geometric sequence -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem geometric_sequence_sixth_term :
  let a₁ := 3
  let q := -2
  geometric_sequence a₁ q 6 = -96 := by sorry

theorem geometric_sequence_general_term :
  let a₃ := 20
  let a₆ := 160
  ∃ q : ℝ, ∀ n : ℕ, geometric_sequence (a₃ / q^2) q n = 5 * 2^(n - 1) := by sorry

end geometric_sequence_sixth_term_geometric_sequence_general_term_l1004_100477


namespace apps_deleted_l1004_100412

theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) 
  (h1 : initial_apps = 16) (h2 : remaining_apps = 8) : 
  initial_apps - remaining_apps = 8 := by
  sorry

end apps_deleted_l1004_100412


namespace range_of_a_l1004_100428

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 2 ≤ 0) → 
  -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l1004_100428


namespace simplification_proof_l1004_100486

theorem simplification_proof (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  (a - 1) / (a^2 - 1) + 1 / (a + 1) = 2 / (a + 1) := by
  sorry

end simplification_proof_l1004_100486


namespace square_properties_l1004_100406

/-- Properties of a square with side length 30 cm -/
theorem square_properties :
  let s : ℝ := 30
  let area : ℝ := s^2
  let diagonal : ℝ := s * Real.sqrt 2
  (area = 900 ∧ diagonal = 30 * Real.sqrt 2) := by
  sorry

end square_properties_l1004_100406


namespace log_product_equality_l1004_100492

theorem log_product_equality : (Real.log 2 / Real.log 3 + Real.log 5 / Real.log 3) * (Real.log 9 / Real.log 10) = 2 := by
  sorry

end log_product_equality_l1004_100492


namespace factorization_sum_l1004_100473

theorem factorization_sum (a b : ℤ) :
  (∀ x : ℚ, 25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -24 := by
sorry

end factorization_sum_l1004_100473


namespace problem_solution_l1004_100455

theorem problem_solution (a b : ℝ) (h : a^2 - 2*b^2 - 2 = 0) :
  -3*a^2 + 6*b^2 + 2023 = 2017 := by
  sorry

end problem_solution_l1004_100455


namespace license_plate_combinations_l1004_100491

def alphabet_size : ℕ := 26
def letter_positions : ℕ := 4
def odd_digits : ℕ := 5

theorem license_plate_combinations :
  (Nat.choose alphabet_size 2) * (Nat.choose letter_positions 2) * (odd_digits * (odd_digits - 1)) = 39000 := by
  sorry

end license_plate_combinations_l1004_100491


namespace factor_expression_l1004_100475

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) := by
  sorry

end factor_expression_l1004_100475


namespace backpack_cost_theorem_l1004_100411

/-- Calculates the total cost of personalized backpacks with a discount -/
def total_cost (num_backpacks : ℕ) (original_price : ℚ) (discount_rate : ℚ) (monogram_fee : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_discounted := num_backpacks.cast * discounted_price
  let total_monogram := num_backpacks.cast * monogram_fee
  total_discounted + total_monogram

/-- Theorem stating that the total cost of 5 backpacks with given prices and discount is $140.00 -/
theorem backpack_cost_theorem :
  total_cost 5 20 (1/5) 12 = 140 := by
  sorry

end backpack_cost_theorem_l1004_100411


namespace initial_alloy_weight_l1004_100469

/-- Represents the composition of an alloy --/
structure Alloy where
  zinc : ℝ
  copper : ℝ

/-- The initial ratio of zinc to copper in the alloy --/
def initial_ratio : ℚ := 5 / 3

/-- The final ratio of zinc to copper after adding zinc --/
def final_ratio : ℚ := 3 / 1

/-- The amount of zinc added to the alloy --/
def added_zinc : ℝ := 8

/-- Theorem stating the initial weight of the alloy --/
theorem initial_alloy_weight (a : Alloy) :
  (a.zinc / a.copper = initial_ratio) →
  ((a.zinc + added_zinc) / a.copper = final_ratio) →
  (a.zinc + a.copper = 16) :=
by sorry

end initial_alloy_weight_l1004_100469


namespace one_intersection_condition_tangent_lines_at_point_l1004_100404

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the range of m
theorem one_intersection_condition (m : ℝ) :
  (∃! x, f x = m) ↔ (m < -2 ∨ m > 2) :=
sorry

-- Theorem for the tangent lines
theorem tangent_lines_at_point :
  let P : ℝ × ℝ := (2, -6)
  ∃ (l₁ l₂ : ℝ → ℝ),
    (∀ x, l₁ x = -3*x) ∧
    (∀ x, l₂ x = 24*x - 54) ∧
    (∀ t, ∃ x, (x, f x) = (t, l₁ t) ∨ (x, f x) = (t, l₂ t)) ∧
    (l₁ 2 = -6) ∧ (l₂ 2 = -6) :=
sorry

end one_intersection_condition_tangent_lines_at_point_l1004_100404


namespace real_roots_of_quartic_equation_l1004_100415

theorem real_roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^4 + 4 * x^3 + 3 * x^2 + x - 1
  let x₁ : ℝ := (-1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 3) / 2
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ (f x₁ = 0 ∧ f x₂ = 0) :=
by sorry

end real_roots_of_quartic_equation_l1004_100415


namespace spells_base7_to_base10_l1004_100420

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of spells in base 7 --/
def spellsBase7 : Nat := 653

/-- Theorem: The number of spells in base 7 (653) is equal to 332 in base 10 --/
theorem spells_base7_to_base10 :
  base7ToBase10 (spellsBase7 / 100) ((spellsBase7 / 10) % 10) (spellsBase7 % 10) = 332 := by
  sorry

end spells_base7_to_base10_l1004_100420


namespace sister_age_l1004_100434

theorem sister_age (B S : ℕ) (h : B = B * S) : S = 1 := by
  sorry

end sister_age_l1004_100434


namespace divisibility_statements_l1004_100456

theorem divisibility_statements :
  (12 % 2 = 0) ∧
  (123 % 3 = 0) ∧
  (1234 % 4 ≠ 0) ∧
  (12345 % 5 = 0) ∧
  (123456 % 6 = 0) :=
by sorry

end divisibility_statements_l1004_100456


namespace sugar_flour_difference_l1004_100461

-- Define constants based on the problem conditions
def flour_recipe : Real := 2.25  -- kg
def sugar_recipe : Real := 5.5   -- lb
def flour_added : Real := 1      -- kg
def kg_to_lb : Real := 2.205     -- 1 kg = 2.205 lb
def kg_to_g : Real := 1000       -- 1 kg = 1000 g

-- Theorem statement
theorem sugar_flour_difference :
  let flour_remaining := (flour_recipe - flour_added) * kg_to_g
  let sugar_needed := (sugar_recipe / kg_to_lb) * kg_to_g
  ∃ ε > 0, abs (sugar_needed - flour_remaining - 1244.8) < ε :=
by sorry

end sugar_flour_difference_l1004_100461


namespace josh_bought_four_cookies_l1004_100449

/-- Calculates the number of cookies Josh bought given his initial money,
    the cost of other items, the cost per cookie, and the remaining money. -/
def cookies_bought (initial_money : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ)
                   (cookie_cost : ℚ) (remaining_money : ℚ) : ℚ :=
  ((initial_money - hat_cost - pencil_cost - remaining_money) / cookie_cost)

/-- Proves that Josh bought 4 cookies given the problem conditions. -/
theorem josh_bought_four_cookies :
  cookies_bought 20 10 2 1.25 3 = 4 := by
  sorry

end josh_bought_four_cookies_l1004_100449


namespace f_of_one_equals_negative_two_l1004_100422

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_of_one_equals_negative_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_def : ∀ x, x < 0 → f x = x - x^4) :
  f 1 = -2 := by
  sorry

end f_of_one_equals_negative_two_l1004_100422


namespace robotics_club_mentor_age_l1004_100467

theorem robotics_club_mentor_age (total_members : ℕ) (avg_age : ℕ) 
  (num_boys num_girls num_mentors : ℕ) (avg_age_boys avg_age_girls : ℕ) :
  total_members = 50 →
  avg_age = 20 →
  num_boys = 25 →
  num_girls = 20 →
  num_mentors = 5 →
  avg_age_boys = 18 →
  avg_age_girls = 19 →
  (total_members * avg_age - num_boys * avg_age_boys - num_girls * avg_age_girls) / num_mentors = 34 :=
by
  sorry

end robotics_club_mentor_age_l1004_100467
