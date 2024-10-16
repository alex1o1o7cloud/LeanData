import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_multiples_of_three_l3275_327541

theorem consecutive_multiples_of_three (n : ℕ) : 
  3 * (n - 1) + 3 * (n + 1) = 150 → 3 * n = 75 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_three_l3275_327541


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l3275_327559

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ n : ℕ, 
  n > 2 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l3275_327559


namespace NUMINAMATH_CALUDE_log_inequality_characterization_l3275_327508

theorem log_inequality_characterization (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha_neq_1 : a ≠ 1) :
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  (b = 1 ∧ a ≠ 1) ∨ (a > b ∧ b > 1) ∨ (b > 1 ∧ 1 > a) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_characterization_l3275_327508


namespace NUMINAMATH_CALUDE_bee_count_l3275_327567

theorem bee_count (flowers : ℕ) (bee_difference : ℕ) : 
  flowers = 5 → bee_difference = 2 → flowers - bee_difference = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_count_l3275_327567


namespace NUMINAMATH_CALUDE_exponent_division_l3275_327543

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^12 / a^4 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3275_327543


namespace NUMINAMATH_CALUDE_curve_C_properties_l3275_327580

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

theorem curve_C_properties (t : ℝ) :
  (is_hyperbola t ↔ ∃ (a b : ℝ), C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
  (is_ellipse_x_axis t ↔ ∃ (a b : ℝ), a > b ∧ C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l3275_327580


namespace NUMINAMATH_CALUDE_sum_of_features_l3275_327540

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- The number of edges in a rectangular prism -/
def num_edges (prism : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (prism : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (prism : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces is 26 -/
theorem sum_of_features (prism : RectangularPrism) :
  num_edges prism + num_corners prism + num_faces prism = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_features_l3275_327540


namespace NUMINAMATH_CALUDE_primitive_root_modulo_p_alpha_implies_modulo_p_l3275_327514

theorem primitive_root_modulo_p_alpha_implies_modulo_p
  (p : Nat) (α : Nat) (x : Nat)
  (h_prime : Nat.Prime p)
  (h_pos : α > 0)
  (h_primitive_p_alpha : IsPrimitiveRoot x (p ^ α)) :
  IsPrimitiveRoot x p :=
sorry

end NUMINAMATH_CALUDE_primitive_root_modulo_p_alpha_implies_modulo_p_l3275_327514


namespace NUMINAMATH_CALUDE_not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l3275_327530

-- Statement 1
theorem not_iff_eq_mul_eq (a b c : ℝ) : ¬(a = b ↔ a * c = b * c) :=
sorry

-- Statement 2
theorem irrational_iff_irrational_plus_five (a : ℝ) : Irrational (a + 5) ↔ Irrational a :=
sorry

-- Statement 3
theorem not_gt_implies_sq_gt (a b : ℝ) : ¬(a > b → a^2 > b^2) :=
sorry

-- Statement 4
theorem lt_three_implies_lt_five (a : ℝ) : a < 3 → a < 5 :=
sorry

end NUMINAMATH_CALUDE_not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l3275_327530


namespace NUMINAMATH_CALUDE_cut_cube_total_count_l3275_327562

/-- Represents a cube cut into smaller equal cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly 2 faces -/
  two_face_painted : ℕ

/-- Theorem stating that a cube cut into smaller equal cubes with 12 two-face painted cubes has 27 total smaller cubes -/
theorem cut_cube_total_count (c : CutCube) (h1 : c.two_face_painted = 12) :
  c.edge_count ^ 3 = 27 := by
  sorry

#check cut_cube_total_count

end NUMINAMATH_CALUDE_cut_cube_total_count_l3275_327562


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l3275_327507

/-- 
Given two trigonometric functions f(x) = 3sin(2x - π/6) and g(x) = 3sin(x + π/2),
prove that the graph of g(x) can be obtained from the graph of f(x) by 
extending the x-coordinates to twice their original values and 
then shifting the resulting graph to the left by 2π/3 units.
-/
theorem sin_graph_transformation (x : ℝ) : 
  3 * Real.sin (x + π/2) = 3 * Real.sin ((2*x - π/6) / 2 + 2*π/3) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l3275_327507


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_is_two_to_one_l3275_327593

/-- Represents the daily sales of ground beef in kilograms -/
structure DailySales where
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the ratio of Friday to Thursday sales is 2:1 -/
theorem friday_to_thursday_ratio_is_two_to_one (sales : DailySales) : 
  sales.thursday = 210 →
  sales.saturday = 130 →
  sales.sunday = sales.saturday / 2 →
  sales.thursday + sales.friday + sales.saturday + sales.sunday = 825 →
  sales.friday / sales.thursday = 2 := by
  sorry

#check friday_to_thursday_ratio_is_two_to_one

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_is_two_to_one_l3275_327593


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l3275_327547

theorem mashed_potatoes_count :
  let bacon_count : ℕ := 42
  let difference : ℕ := 366
  let mashed_potatoes_count : ℕ := bacon_count + difference
  mashed_potatoes_count = 408 := by
sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l3275_327547


namespace NUMINAMATH_CALUDE_new_person_weight_l3275_327534

/-- Calculates the weight of a new person given the following conditions:
  * There are 6 people initially
  * Replacing one person weighing 69 kg with a new person increases the average weight by 1.8 kg
-/
theorem new_person_weight (num_people : Nat) (weight_increase : Real) (replaced_weight : Real) :
  num_people = 6 →
  weight_increase = 1.8 →
  replaced_weight = 69 →
  ∃ (new_weight : Real), new_weight = 79.8 ∧
    new_weight = replaced_weight + num_people * weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3275_327534


namespace NUMINAMATH_CALUDE_egg_problem_l3275_327532

/-- The initial number of eggs in the basket -/
def initial_eggs : ℕ := 120

/-- The number of broken eggs -/
def broken_eggs : ℕ := 20

/-- The total price in fillérs -/
def total_price : ℕ := 600

/-- Proves that the initial number of eggs was 120 -/
theorem egg_problem :
  initial_eggs = 120 ∧
  broken_eggs = 20 ∧
  total_price = 600 ∧
  (total_price : ℚ) / initial_eggs + 1 = total_price / (initial_eggs - broken_eggs) :=
by sorry

end NUMINAMATH_CALUDE_egg_problem_l3275_327532


namespace NUMINAMATH_CALUDE_prob_three_green_in_seven_trials_l3275_327531

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 4

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of successful trials (picking green marbles) -/
def num_success : ℕ := 3

/-- The probability of picking a green marble in a single trial -/
def prob_green : ℚ := green_marbles / total_marbles

/-- The probability of picking a purple marble in a single trial -/
def prob_purple : ℚ := purple_marbles / total_marbles

/-- The probability of picking exactly three green marbles in seven trials -/
theorem prob_three_green_in_seven_trials :
  (Nat.choose num_trials num_success : ℚ) * prob_green ^ num_success * prob_purple ^ (num_trials - num_success) = 280 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_green_in_seven_trials_l3275_327531


namespace NUMINAMATH_CALUDE_initial_knives_count_l3275_327597

/-- Represents the initial number of knives --/
def initial_knives : ℕ := 24

/-- Represents the initial number of teaspoons --/
def initial_teaspoons : ℕ := 2 * initial_knives

/-- Represents the additional knives --/
def additional_knives : ℕ := initial_knives / 3

/-- Represents the additional teaspoons --/
def additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3

/-- The total number of cutlery pieces after additions --/
def total_cutlery : ℕ := 112

theorem initial_knives_count : 
  initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery :=
by sorry

end NUMINAMATH_CALUDE_initial_knives_count_l3275_327597


namespace NUMINAMATH_CALUDE_toms_flying_robots_l3275_327545

theorem toms_flying_robots (michael_robots : ℕ) (tom_robots : ℕ) : 
  michael_robots = 12 →
  michael_robots = 4 * tom_robots →
  tom_robots = 3 := by
sorry

end NUMINAMATH_CALUDE_toms_flying_robots_l3275_327545


namespace NUMINAMATH_CALUDE_age_difference_l3275_327527

/-- Given three people a, b, and c, where the total age of a and b is 20 years more than
    the total age of b and c, prove that c is 20 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 20) : a = c + 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3275_327527


namespace NUMINAMATH_CALUDE_min_xy_value_l3275_327521

theorem min_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) :
  ∀ z, x * y ≥ z → z ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3275_327521


namespace NUMINAMATH_CALUDE_soccer_league_games_l3275_327586

/-- The number of games played in a soccer league with given conditions -/
def total_games (n : ℕ) (promo_per_team : ℕ) : ℕ :=
  (n * (n - 1) + n * promo_per_team) / 2

/-- Theorem: In a soccer league with 15 teams, where each team plays every other team twice 
    and has 2 additional promotional games, the total number of games played is 120 -/
theorem soccer_league_games : total_games 15 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3275_327586


namespace NUMINAMATH_CALUDE_intersection_sum_l3275_327515

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + c) →
  (∀ x y : ℝ, y = 5 * x + d) →
  16 = 2 * 4 + c →
  16 = 5 * 4 + d →
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3275_327515


namespace NUMINAMATH_CALUDE_smallest_integer_side_of_triangle_l3275_327528

theorem smallest_integer_side_of_triangle (s : ℕ) : 
  (4 : ℝ) ≤ s ∧ 
  (7.8 : ℝ) + s > 11 ∧ 
  (7.8 : ℝ) + 11 > s ∧ 
  11 + s > (7.8 : ℝ) ∧
  ∀ (t : ℕ), t < s → 
    ((7.8 : ℝ) + (t : ℝ) ≤ 11 ∨ 
     (7.8 : ℝ) + 11 ≤ (t : ℝ) ∨ 
     11 + (t : ℝ) ≤ (7.8 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_side_of_triangle_l3275_327528


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l3275_327590

/-- Represents a cube with smaller cubes removed from alternate corners -/
structure ModifiedCube where
  side_length : ℕ
  removed_cube_side_length : ℕ
  removed_corners : ℕ

/-- Calculates the number of edges in a modified cube -/
def edge_count (c : ModifiedCube) : ℕ :=
  12 + 3 * c.removed_corners

/-- Theorem stating that a cube of side length 4 with unit cubes removed from 4 corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.side_length = 4 ∧ 
    c.removed_cube_side_length = 1 ∧ 
    c.removed_corners = 4 → 
    edge_count c = 24 := by
  sorry

#check modified_cube_edge_count

end NUMINAMATH_CALUDE_modified_cube_edge_count_l3275_327590


namespace NUMINAMATH_CALUDE_nuts_per_student_l3275_327500

theorem nuts_per_student (bags : ℕ) (students : ℕ) (nuts_per_bag : ℕ) 
  (h1 : bags = 65) 
  (h2 : students = 13) 
  (h3 : nuts_per_bag = 15) : 
  (bags * nuts_per_bag) / students = 75 := by
sorry

end NUMINAMATH_CALUDE_nuts_per_student_l3275_327500


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3275_327503

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_with_complement : M ∩ (I \ N) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3275_327503


namespace NUMINAMATH_CALUDE_middle_number_problem_l3275_327522

theorem middle_number_problem (x y z : ℕ) : 
  x < y → y < z → x + y = 20 → x + z = 25 → y + z = 29 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l3275_327522


namespace NUMINAMATH_CALUDE_range_of_m_l3275_327509

def has_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c ≥ 0

def p (m : ℝ) : Prop :=
  has_real_roots 1 m 1

def q (m : ℝ) : Prop :=
  ¬(has_real_roots 4 (4*(m-2)) 1)

def exactly_one_true (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_m : 
  {m : ℝ | exactly_one_true (p m) (q m)} = 
  {m : ℝ | m ≤ -2 ∨ (1 < m ∧ m < 2) ∨ m ≥ 3} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3275_327509


namespace NUMINAMATH_CALUDE_flash_interval_value_l3275_327546

/-- The number of flashes in ¾ of an hour -/
def flashes : ℕ := 240

/-- The duration in hours -/
def duration : ℚ := 3/4

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time interval between flashes in seconds -/
def flash_interval : ℚ := (duration * seconds_per_hour) / flashes

theorem flash_interval_value : flash_interval = 45/4 := by sorry

end NUMINAMATH_CALUDE_flash_interval_value_l3275_327546


namespace NUMINAMATH_CALUDE_triangle_inequality_l3275_327504

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3275_327504


namespace NUMINAMATH_CALUDE_gum_purchase_cost_l3275_327524

/-- Calculates the total cost in dollars for buying gum with a discount -/
def total_cost_with_discount (price_per_piece : ℚ) (num_pieces : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_cost_cents := price_per_piece * num_pieces
  let discount_amount := discount_rate * total_cost_cents
  let final_cost_cents := total_cost_cents - discount_amount
  final_cost_cents / 100

/-- Theorem: The total cost of buying 1500 pieces of gum at 2 cents each with a 10% discount is $27 -/
theorem gum_purchase_cost :
  total_cost_with_discount 2 1500 (10/100) = 27 := by
  sorry


end NUMINAMATH_CALUDE_gum_purchase_cost_l3275_327524


namespace NUMINAMATH_CALUDE_train_length_l3275_327510

/-- Given a train traveling at 270 kmph and crossing a pole in 5 seconds, its length is 375 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (h1 : speed_kmph = 270) (h2 : crossing_time = 5) :
  let speed_ms := speed_kmph * 1000 / 3600
  speed_ms * crossing_time = 375 := by sorry

end NUMINAMATH_CALUDE_train_length_l3275_327510


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3275_327538

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 = 8 * x
def equation2 (y : ℝ) : Prop := y^2 - 10 * y - 1 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 0 ∨ x = 4)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26)) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3275_327538


namespace NUMINAMATH_CALUDE_negation_of_existence_leq_negation_of_proposition_l3275_327579

theorem negation_of_existence_leq (p : ℝ → Prop) :
  (¬ ∃ x₀ : ℝ, p x₀) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, Real.exp x₀ - x₀ - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_leq_negation_of_proposition_l3275_327579


namespace NUMINAMATH_CALUDE_absent_laborers_l3275_327598

/-- Proves that 6 laborers were absent given the problem conditions -/
theorem absent_laborers (total_laborers : ℕ) (planned_days : ℕ) (actual_days : ℕ)
  (h1 : total_laborers = 15)
  (h2 : planned_days = 9)
  (h3 : actual_days = 15)
  (h4 : total_laborers * planned_days = (total_laborers - absent) * actual_days) :
  absent = 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_laborers_l3275_327598


namespace NUMINAMATH_CALUDE_max_value_a_squared_b_l3275_327553

theorem max_value_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ * (a₀ + b₀) = 27 ∧ a₀^2 * b₀ = 54 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_squared_b_l3275_327553


namespace NUMINAMATH_CALUDE_most_appropriate_survey_method_l3275_327542

/-- Represents different survey methods -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents different survey scenarios -/
inductive SurveyScenario
| CityFloatingPopulation
| AirplaneSecurityCheck
| ShellKillingRadius
| ClassMathScores

/-- Determines if a survey method is appropriate for a given scenario -/
def is_appropriate (method : SurveyMethod) (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.CityFloatingPopulation => method = SurveyMethod.Sampling
  | SurveyScenario.AirplaneSecurityCheck => method = SurveyMethod.Census
  | SurveyScenario.ShellKillingRadius => method = SurveyMethod.Sampling
  | SurveyScenario.ClassMathScores => method = SurveyMethod.Census

/-- Theorem stating that using a census method for class math scores is the most appropriate -/
theorem most_appropriate_survey_method :
  is_appropriate SurveyMethod.Census SurveyScenario.ClassMathScores ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.CityFloatingPopulation) ∧
  ¬(is_appropriate SurveyMethod.Sampling SurveyScenario.AirplaneSecurityCheck) ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.ShellKillingRadius) :=
by sorry

end NUMINAMATH_CALUDE_most_appropriate_survey_method_l3275_327542


namespace NUMINAMATH_CALUDE_book_spending_is_correct_l3275_327516

def allowance : ℚ := 50

def game_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/5
def toy_fraction : ℚ := 2/5

def book_spending : ℚ := allowance - (allowance * game_fraction + allowance * snack_fraction + allowance * toy_fraction)

theorem book_spending_is_correct : book_spending = 7.5 := by sorry

end NUMINAMATH_CALUDE_book_spending_is_correct_l3275_327516


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3275_327526

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,5}
def B : Finset Nat := {1,4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {3,5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3275_327526


namespace NUMINAMATH_CALUDE_roots_greater_than_two_range_l3275_327518

theorem roots_greater_than_two_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-4)*x + (6-m) = 0 → x > 2) →
  -2 < m ∧ m ≤ 2 - 2*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_greater_than_two_range_l3275_327518


namespace NUMINAMATH_CALUDE_sequence_problem_l3275_327520

theorem sequence_problem (a : Fin 100 → ℝ) 
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3275_327520


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l3275_327517

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 3*a*x + 2*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p x 2 ∨ q x) : 2 < x ∧ x < 4 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h' : ∃ x, ¬(p x a) ∧ q x) : 3/2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l3275_327517


namespace NUMINAMATH_CALUDE_solution_set_eq_two_and_neg_two_l3275_327576

def solution_set : Set ℝ :=
  {x : ℝ | Real.sqrt ((5 + 2 * Real.sqrt 6) ^ x) + Real.sqrt ((5 - 2 * Real.sqrt 6) ^ x) = 10}

theorem solution_set_eq_two_and_neg_two : solution_set = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_two_and_neg_two_l3275_327576


namespace NUMINAMATH_CALUDE_base6_divisibility_by_19_l3275_327525

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base6_divisibility_by_19 (y : ℕ) (h : y < 6) :
  isDivisibleBy19 (base6ToDecimal 2 5 y 3) ↔ y = 2 := by sorry

end NUMINAMATH_CALUDE_base6_divisibility_by_19_l3275_327525


namespace NUMINAMATH_CALUDE_positive_solution_to_equation_l3275_327592

theorem positive_solution_to_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1 / x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_to_equation_l3275_327592


namespace NUMINAMATH_CALUDE_recipe_calculation_l3275_327584

/-- The amount of flour Julia uses in mL -/
def flour_amount : ℕ := 800

/-- The base amount of flour in mL for the recipe ratio -/
def base_flour : ℕ := 200

/-- The amount of milk in mL needed for the base amount of flour -/
def milk_per_base : ℕ := 60

/-- The number of eggs needed for the base amount of flour -/
def eggs_per_base : ℕ := 1

/-- The amount of milk needed for Julia's recipe -/
def milk_needed : ℕ := (flour_amount / base_flour) * milk_per_base

/-- The number of eggs needed for Julia's recipe -/
def eggs_needed : ℕ := (flour_amount / base_flour) * eggs_per_base

theorem recipe_calculation : 
  milk_needed = 240 ∧ eggs_needed = 4 := by sorry

end NUMINAMATH_CALUDE_recipe_calculation_l3275_327584


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3275_327582

/-- If a quadratic equation with real coefficients has 5 + 3i as a root, then its constant term is 34 -/
theorem quadratic_constant_term (b c : ℝ) : 
  (∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 5 + 3*Complex.I) →
  c = 34 := by sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3275_327582


namespace NUMINAMATH_CALUDE_y_intercept_after_translation_l3275_327563

/-- A linear function f(x) = mx + b -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- Vertical translation of a function by k units -/
def VerticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- Y-intercept of a function -/
def YIntercept (f : ℝ → ℝ) : ℝ := f 0

theorem y_intercept_after_translation :
  let f := LinearFunction (-2) 3
  let g := VerticalTranslation f 2
  YIntercept g = 5 := by sorry

end NUMINAMATH_CALUDE_y_intercept_after_translation_l3275_327563


namespace NUMINAMATH_CALUDE_factorization_equality_l3275_327512

theorem factorization_equality (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3275_327512


namespace NUMINAMATH_CALUDE_water_volume_in_solution_l3275_327556

/-- Calculates the volume of a component in a solution given the total volume and the component's proportion -/
def component_volume (total_volume : ℝ) (proportion : ℝ) : ℝ :=
  total_volume * proportion

theorem water_volume_in_solution (total_volume : ℝ) (water_proportion : ℝ) 
  (h1 : total_volume = 1.20)
  (h2 : water_proportion = 0.50) :
  component_volume total_volume water_proportion = 0.60 := by
  sorry

#eval component_volume 1.20 0.50

end NUMINAMATH_CALUDE_water_volume_in_solution_l3275_327556


namespace NUMINAMATH_CALUDE_total_spent_on_souvenirs_l3275_327591

/-- The amount spent on t-shirts -/
def t_shirts : ℝ := 201

/-- The amount spent on key chains and bracelets -/
def key_chains_and_bracelets : ℝ := 347

/-- The difference between key_chains_and_bracelets and t_shirts -/
def difference : ℝ := 146

theorem total_spent_on_souvenirs :
  key_chains_and_bracelets = t_shirts + difference →
  t_shirts + key_chains_and_bracelets = 548 :=
by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_souvenirs_l3275_327591


namespace NUMINAMATH_CALUDE_fraction_of_states_1790_to_1799_l3275_327555

theorem fraction_of_states_1790_to_1799 (total_states : ℕ) (states_1790_to_1799 : ℕ) : 
  total_states = 30 → states_1790_to_1799 = 9 → (states_1790_to_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_states_1790_to_1799_l3275_327555


namespace NUMINAMATH_CALUDE_retailer_profit_calculation_l3275_327552

/-- Calculates the actual profit percentage for a retailer who marks up goods
    by a certain percentage and then offers a discount. -/
theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (markup_percentage_is_40 : markup_percentage = 40)
  (discount_percentage_is_25 : discount_percentage = 25)
  : let marked_price := cost_price * (1 + markup_percentage / 100)
    let selling_price := marked_price * (1 - discount_percentage / 100)
    let profit := selling_price - cost_price
    let profit_percentage := (profit / cost_price) * 100
    profit_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_calculation_l3275_327552


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l3275_327594

def polynomial (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 18*x - 24

theorem sum_of_absolute_roots : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 6 + 2 * Real.sqrt 6 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l3275_327594


namespace NUMINAMATH_CALUDE_selection_methods_count_l3275_327568

def total_volunteers : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selection_size : ℕ := 3

theorem selection_methods_count : 
  (Nat.choose boys 2 * Nat.choose girls 1) + (Nat.choose boys 1 * Nat.choose girls 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3275_327568


namespace NUMINAMATH_CALUDE_triangle_circle_area_l3275_327575

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let angle1 : ℝ := 45 * π / 180
  let angle2 : ℝ := 15 * π / 180
  let angle3 : ℝ := π - angle1 - angle2
  let height : ℝ := a * (Real.sqrt 3 - 1) / (2 * Real.sqrt 3)
  let circle_area : ℝ := π * height^2
  circle_area / 3 = π * a^2 * (2 - Real.sqrt 3) / 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_l3275_327575


namespace NUMINAMATH_CALUDE_area_of_triangle_BXC_l3275_327501

/-- Represents a trapezoid with bases and area -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Represents the triangle formed by the intersection of diagonals -/
structure DiagonalTriangle where
  area : ℝ

/-- Theorem stating the area of triangle BXC in the given trapezoid -/
theorem area_of_triangle_BXC (trapezoid : Trapezoid) (triangle : DiagonalTriangle) :
  trapezoid.base1 = 15 ∧ 
  trapezoid.base2 = 35 ∧ 
  trapezoid.area = 375 →
  triangle.area = 78.75 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_BXC_l3275_327501


namespace NUMINAMATH_CALUDE_frame_254_width_2_l3275_327571

/-- The number of cells in a square frame with given side length and width -/
def frame_cells (side_length : ℕ) (width : ℕ) : ℕ :=
  side_length ^ 2 - (side_length - 2 * width) ^ 2

/-- Theorem: A 254 × 254 frame with width 2 has 2016 cells -/
theorem frame_254_width_2 :
  frame_cells 254 2 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_frame_254_width_2_l3275_327571


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l3275_327581

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 60)
  (h2 : tea_drinkers = 90) :
  coffee_drinkers + tea_drinkers - 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l3275_327581


namespace NUMINAMATH_CALUDE_root_existence_quadratic_root_existence_l3275_327558

theorem root_existence (f : ℝ → ℝ) (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) 
  (h3 : Continuous f) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem quadratic_root_existence (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_existence_quadratic_root_existence_l3275_327558


namespace NUMINAMATH_CALUDE_principal_calculation_l3275_327589

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem principal_calculation (interest rate time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 0.13)
  (h3 : time = 5)
  : ∃ (principal : ℝ), simple_interest principal rate time = interest ∧ principal = 6180 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3275_327589


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l3275_327533

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) + (1 - a)) = Complex.I * (a + Complex.I) / (1 + Complex.I)) → 
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l3275_327533


namespace NUMINAMATH_CALUDE_student_height_correction_l3275_327511

theorem student_height_correction (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 20 →
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_avg = 174.25 →
  ∃ (actual_height : ℝ), 
    actual_height = 166 ∧
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end NUMINAMATH_CALUDE_student_height_correction_l3275_327511


namespace NUMINAMATH_CALUDE_range_of_sum_equal_product_l3275_327550

theorem range_of_sum_equal_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = x * y) :
  x + y ≥ 4 ∧ ∀ z ≥ 4, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + y = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_equal_product_l3275_327550


namespace NUMINAMATH_CALUDE_no_unique_solution_l3275_327513

/-- For a system of two linear equations to have no unique solution, 
    the ratios of coefficients and constants must be equal. -/
theorem no_unique_solution (k e : ℝ) : 
  (∃ k, ¬∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + e * y = 30) →
  e = 10 := by
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l3275_327513


namespace NUMINAMATH_CALUDE_subtraction_division_fractions_l3275_327595

theorem subtraction_division_fractions : ((3 / 4 : ℚ) - (5 / 8 : ℚ)) / 2 = (1 / 16 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_fractions_l3275_327595


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3275_327573

theorem quadratic_roots_transformation (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 ↔ x = x₁^2 + x₂^2 ∨ x = 2 * x₁ * x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3275_327573


namespace NUMINAMATH_CALUDE_rug_dimension_l3275_327561

theorem rug_dimension (x : ℝ) : 
  x > 0 ∧ 
  x ≤ 8 ∧
  7 ≤ 8 ∧
  x * 7 = 64 * (1 - 0.78125) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_rug_dimension_l3275_327561


namespace NUMINAMATH_CALUDE_exists_student_with_sqrt_k_classes_l3275_327502

/-- Represents a school with students and classes -/
structure School where
  n : ℕ  -- number of classes
  k : ℕ  -- number of students
  shared_class : Fin k → Fin k → Fin n
  class_size : Fin n → ℕ
  h1 : ∀ i j, i ≠ j → shared_class i j = shared_class j i
  h2 : ∀ i, class_size i < k
  h3 : ¬ ∃ m, k - 1 = m * m

/-- The number of classes a student attends -/
def num_classes_attended (s : School) (student : Fin s.k) : ℕ :=
  (Finset.univ.filter (λ c : Fin s.n => ∃ other, s.shared_class student other = c)).card

/-- Main theorem: There exists a student who has attended at least √k classes -/
theorem exists_student_with_sqrt_k_classes (s : School) :
  ∃ student : Fin s.k, s.k.sqrt ≤ num_classes_attended s student := by
  sorry

end NUMINAMATH_CALUDE_exists_student_with_sqrt_k_classes_l3275_327502


namespace NUMINAMATH_CALUDE_average_salary_problem_l3275_327519

theorem average_salary_problem (salary_raj roshan : ℕ) : 
  (salary_raj + roshan) / 2 = 4000 →
  ((salary_raj + roshan + 7000) : ℚ) / 3 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_problem_l3275_327519


namespace NUMINAMATH_CALUDE_min_triangle_area_l3275_327569

/-- Triangle DEF with vertices D(0,0) and E(24,10), and F having integer coordinates -/
structure Triangle where
  F : ℤ × ℤ

/-- Area of triangle DEF given coordinates of F -/
def triangleArea (t : Triangle) : ℚ :=
  let (x, y) := t.F
  (1 : ℚ) / 2 * |10 * x - 24 * y|

/-- The minimum non-zero area of triangle DEF is 5 -/
theorem min_triangle_area :
  ∃ (t : Triangle), triangleArea t > 0 ∧
  ∀ (t' : Triangle), triangleArea t' > 0 → triangleArea t ≤ triangleArea t' :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3275_327569


namespace NUMINAMATH_CALUDE_x_less_than_y_l3275_327578

theorem x_less_than_y (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (x y : ℝ) (hx : x = (0.1993 : ℝ)^b * (0.1997 : ℝ)^a) 
  (hy : y = (0.1993 : ℝ)^a * (0.1997 : ℝ)^b) : x < y := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l3275_327578


namespace NUMINAMATH_CALUDE_dot_product_range_l3275_327570

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

-- Define points M and N on the hypotenuse
def OnHypotenuse (M N : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
  M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
  N = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)

-- Define the distance between M and N
def MNDistance (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 2

-- Define the dot product of CM and CN
def DotProduct (C M N : ℝ × ℝ) : ℝ :=
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2)

theorem dot_product_range (A B C M N : ℝ × ℝ) :
  Triangle A B C →
  OnHypotenuse M N A B →
  MNDistance M N →
  (3/2 : ℝ) ≤ DotProduct C M N ∧ DotProduct C M N ≤ 2 := by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3275_327570


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3275_327537

def P (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, ∃ q, P a b c x = (x - 1)^3 * q) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3275_327537


namespace NUMINAMATH_CALUDE_heather_walk_distance_l3275_327583

theorem heather_walk_distance : 
  let car_to_entrance : Float := 0.645
  let to_carnival : Float := 1.235
  let to_animals : Float := 0.875
  let to_food : Float := 1.537
  let food_to_car : Float := 0.932
  car_to_entrance + to_carnival + to_animals + to_food + food_to_car = 5.224 := by
  sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l3275_327583


namespace NUMINAMATH_CALUDE_target_probabilities_l3275_327554

/-- Probability of hitting a target -/
structure TargetProbability where
  prob : ℚ
  prob_nonneg : 0 ≤ prob
  prob_le_one : prob ≤ 1

/-- Model for the target shooting scenario -/
structure TargetScenario where
  A : TargetProbability
  B : TargetProbability

/-- Given scenario with person A and B's probabilities -/
def given_scenario : TargetScenario :=
  { A := { prob := 3/4, prob_nonneg := by norm_num, prob_le_one := by norm_num },
    B := { prob := 4/5, prob_nonneg := by norm_num, prob_le_one := by norm_num } }

/-- Probability that A hits and B misses after one shot each -/
def prob_A_hits_B_misses (s : TargetScenario) : ℚ :=
  s.A.prob * (1 - s.B.prob)

/-- Probability of k successes in n independent trials -/
def binomial_prob (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- Probability that A and B have equal hits after two shots each -/
def prob_equal_hits (s : TargetScenario) : ℚ :=
  (binomial_prob s.A.prob 2 0) * (binomial_prob s.B.prob 2 0) +
  (binomial_prob s.A.prob 2 1) * (binomial_prob s.B.prob 2 1) +
  (binomial_prob s.A.prob 2 2) * (binomial_prob s.B.prob 2 2)

theorem target_probabilities (s : TargetScenario := given_scenario) :
  (prob_A_hits_B_misses s = 3/20) ∧
  (prob_equal_hits s = 193/400) := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l3275_327554


namespace NUMINAMATH_CALUDE_triangle_problem_l3275_327544

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the problem statement
theorem triangle_problem (t : Triangle) 
  (h1 : Real.tan t.A = Real.sin t.B)  -- tan A = sin B
  (h2 : ∃ (D : ℝ), 2 * D = t.a ∧ t.b = t.c)  -- BD = DC (implying 2D = a and b = c)
  (h3 : t.c = t.b) :  -- AD = AB (implying c = b)
  (2 * t.a * t.c = t.b^2 + t.c^2 - t.a^2) ∧ 
  (Real.sin t.A / Real.sin t.C = 2 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3275_327544


namespace NUMINAMATH_CALUDE_cos_B_value_triangle_area_l3275_327551

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.b = t.c ∧ 2 * Real.sin t.B = Real.sqrt 3 * Real.sin t.A

-- Theorem for part (i)
theorem cos_B_value (t : Triangle) (h : SpecialTriangle t) : 
  Real.cos t.B = Real.sqrt 3 / 3 := by
  sorry

-- Theorem for part (ii)
theorem triangle_area (t : Triangle) (h : SpecialTriangle t) (ha : t.a = 2) :
  (1/2) * t.a * t.b * Real.sin t.B = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_B_value_triangle_area_l3275_327551


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l3275_327572

def initial_stickers (total : ℝ) (store : ℝ) (birthday : ℝ) (sister : ℝ) (mother : ℝ) : ℝ :=
  total - (store + birthday + sister + mother)

theorem mika_initial_stickers :
  initial_stickers 130 26 20 6 58 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_initial_stickers_l3275_327572


namespace NUMINAMATH_CALUDE_miss_world_regional_competition_l3275_327577

/-- Miss World Regional Competition Theorem -/
theorem miss_world_regional_competition 
  (total : Nat) 
  (blue_white : Nat) 
  (light_black : Nat) 
  (brown : Nat) 
  (h1 : total = 48)
  (h2 : blue_white = 12)
  (h3 : light_black = 28)
  (h4 : brown = 15) : 
  (∃ (a b : Nat), 
    a = brown - (total - light_black - blue_white) ∧ 
    b = total - light_black - blue_white ∧ 
    a = 7 ∧ 
    b = 8) := by
  sorry


end NUMINAMATH_CALUDE_miss_world_regional_competition_l3275_327577


namespace NUMINAMATH_CALUDE_simplify_expression_l3275_327535

theorem simplify_expression : (5^7 + 2^8) * (1^5 - (-1)^5)^10 = 80263680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3275_327535


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3275_327566

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => a n ^ 2 + 3

def b : ℕ → ℕ
  | 0 => 0
  | n + 1 => b n ^ 2 + 2 ^ (n + 1)

theorem a_greater_than_b : b 2003 < a 2003 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3275_327566


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l3275_327574

/- Define the associations and their sizes -/
def associations : Fin 3 → ℕ
| 0 => 27  -- Association A
| 1 => 9   -- Association B
| 2 => 18  -- Association C

/- Total number of athletes -/
def total_athletes : ℕ := (associations 0) + (associations 1) + (associations 2)

/- Number of athletes to be selected -/
def selected_athletes : ℕ := 6

/- Theorem for stratified sampling -/
theorem stratified_sampling_proportion (i : Fin 3) :
  (associations i) * selected_athletes = (associations i) * total_athletes / total_athletes :=
sorry

/- Theorem for number of ways to choose 2 from 6 -/
theorem choose_two_from_six :
  Nat.choose selected_athletes 2 = 15 :=
sorry

/- Theorem for probability of selecting at least one from last two -/
theorem prob_at_least_one_from_last_two :
  (Nat.choose 4 1 * Nat.choose 2 1 + Nat.choose 2 2) / Nat.choose 6 2 = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l3275_327574


namespace NUMINAMATH_CALUDE_lightsaber_to_other_toys_ratio_l3275_327596

-- Define the cost of other Star Wars toys
def other_toys_cost : ℕ := 1000

-- Define the total spent
def total_spent : ℕ := 3000

-- Define the cost of the lightsaber
def lightsaber_cost : ℕ := total_spent - other_toys_cost

-- Theorem statement
theorem lightsaber_to_other_toys_ratio :
  (lightsaber_cost : ℚ) / other_toys_cost = 2 := by sorry

end NUMINAMATH_CALUDE_lightsaber_to_other_toys_ratio_l3275_327596


namespace NUMINAMATH_CALUDE_dans_initial_green_marbles_count_l3275_327523

def dans_initial_green_marbles : ℕ := sorry

def mikes_taken_marbles : ℕ := 23

def dans_remaining_green_marbles : ℕ := 9

theorem dans_initial_green_marbles_count : 
  dans_initial_green_marbles = dans_remaining_green_marbles + mikes_taken_marbles := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_green_marbles_count_l3275_327523


namespace NUMINAMATH_CALUDE_intersection_of_given_lines_l3275_327557

/-- The intersection point of two lines in 2D space -/
def intersection_point (line1_start : ℝ × ℝ) (line1_dir : ℝ × ℝ) (line2_start : ℝ × ℝ) (line2_dir : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem intersection_of_given_lines :
  let line1_start : ℝ × ℝ := (2, -3)
  let line1_dir : ℝ × ℝ := (3, 4)
  let line2_start : ℝ × ℝ := (-1, 4)
  let line2_dir : ℝ × ℝ := (5, -1)
  intersection_point line1_start line1_dir line2_start line2_dir = (124/5, 137/5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_given_lines_l3275_327557


namespace NUMINAMATH_CALUDE_smallest_number_of_ducks_l3275_327505

def duck_flock_size : ℕ := 18
def seagull_flock_size : ℕ := 10

theorem smallest_number_of_ducks (total_ducks total_seagulls : ℕ) : 
  total_ducks = total_seagulls → 
  total_ducks % duck_flock_size = 0 →
  total_seagulls % seagull_flock_size = 0 →
  total_ducks ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_ducks_l3275_327505


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3275_327587

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 5 * a 11 = 3
  property2 : a 3 + a 13 = 4

/-- The theorem stating the possible values of a_15 / a_5 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 15 / seq.a 5 = 1/3 ∨ seq.a 15 / seq.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3275_327587


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l3275_327506

theorem fundraiser_group_composition (p : ℕ) : 
  (∃ (initial_girls : ℕ),
    -- Initial condition: 30% of the group are girls
    initial_girls = (3 * p) / 10 ∧
    -- After changes: 25% of the group are girls
    (initial_girls - 3 : ℚ) / (p + 2) = 1 / 4 →
    -- Prove that the initial number of girls was 21
    initial_girls = 21) :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_group_composition_l3275_327506


namespace NUMINAMATH_CALUDE_distance_rode_bus_l3275_327585

/-- The distance Craig walked from the bus stop to home, in miles -/
def distance_walked : ℝ := 0.17

/-- The difference between the distance Craig rode the bus and the distance he walked, in miles -/
def distance_difference : ℝ := 3.67

/-- Theorem: The distance Craig rode the bus is 3.84 miles -/
theorem distance_rode_bus : distance_walked + distance_difference = 3.84 := by
  sorry

end NUMINAMATH_CALUDE_distance_rode_bus_l3275_327585


namespace NUMINAMATH_CALUDE_candy_ratio_is_five_thirds_l3275_327599

/-- Represents the number of M&M candies Penelope has -/
def mm_candies : ℕ := 25

/-- Represents the number of Starbursts candies Penelope has -/
def starbursts_candies : ℕ := 15

/-- Represents the ratio of M&M candies to Starbursts candies -/
def candy_ratio : Rat := mm_candies / starbursts_candies

theorem candy_ratio_is_five_thirds : candy_ratio = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_is_five_thirds_l3275_327599


namespace NUMINAMATH_CALUDE_total_amount_is_fifteen_l3275_327560

/-- Represents the share distribution among three people -/
structure ShareDistribution where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the total amount given a share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: Given the specified share distribution and the first person's share, 
    the total amount is 15 rupees -/
theorem total_amount_is_fifteen 
  (s : ShareDistribution) 
  (h1 : s.first = 10)
  (h2 : s.second = 0.3 * s.first)
  (h3 : s.third = 0.2 * s.first) : 
  totalAmount s = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_fifteen_l3275_327560


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3275_327529

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 3) : 
  ∀ z, x + 2*y ≥ z → z ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3275_327529


namespace NUMINAMATH_CALUDE_problem_statement_l3275_327548

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1)^2 = 0 → 3*b - a = -5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3275_327548


namespace NUMINAMATH_CALUDE_all_trinomials_no_roots_l3275_327536

/-- Represents a quadratic trinomial ax² + bx + c -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℤ :=
  q.b ^ 2 - 4 * q.a * q.c

/-- Checks if a quadratic trinomial has no real roots -/
def has_no_real_roots (q : QuadraticTrinomial) : Prop :=
  discriminant q < 0

/-- Creates all permutations of three coefficients -/
def all_permutations (a b c : ℤ) : List QuadraticTrinomial :=
  [
    ⟨a, b, c⟩, ⟨a, c, b⟩,
    ⟨b, a, c⟩, ⟨b, c, a⟩,
    ⟨c, a, b⟩, ⟨c, b, a⟩
  ]

theorem all_trinomials_no_roots
  (a b c : ℤ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∀ q ∈ all_permutations a b c, has_no_real_roots q :=
sorry

end NUMINAMATH_CALUDE_all_trinomials_no_roots_l3275_327536


namespace NUMINAMATH_CALUDE_distance_to_place_l3275_327565

/-- Proves that the distance to a place is 72 km given the specified conditions -/
theorem distance_to_place (still_water_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  still_water_speed = 10 →
  current_speed = 2 →
  total_time = 15 →
  ∃ (distance : ℝ), distance = 72 ∧
    distance / (still_water_speed - current_speed) +
    distance / (still_water_speed + current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l3275_327565


namespace NUMINAMATH_CALUDE_train_length_train_length_is_120_l3275_327539

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  relative_speed * passing_time - initial_distance

/-- Proof that the train length is 120 meters under given conditions -/
theorem train_length_is_120 :
  train_length 9 45 120 24 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_120_l3275_327539


namespace NUMINAMATH_CALUDE_total_orange_balloons_l3275_327549

-- Define the initial number of orange balloons
def initial_orange_balloons : ℝ := 9.0

-- Define the number of orange balloons found
def found_orange_balloons : ℝ := 2.0

-- Theorem to prove
theorem total_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_total_orange_balloons_l3275_327549


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3275_327564

/-- A point in the xy-plane is represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of being in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- The point (-1,4) -/
def point : Point := ⟨-1, 4⟩

/-- Theorem: The point (-1,4) is in the second quadrant -/
theorem point_in_second_quadrant : isInSecondQuadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3275_327564


namespace NUMINAMATH_CALUDE_warehouse_paint_area_l3275_327588

/-- Calculates the area to be painted in a rectangular warehouse with a door. -/
def areaToBePainted (length width height doorWidth doorHeight : ℝ) : ℝ :=
  2 * (length * height + width * height) - (doorWidth * doorHeight)

/-- Theorem stating the area to be painted for the given warehouse dimensions. -/
theorem warehouse_paint_area :
  areaToBePainted 8 6 3.5 1 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_paint_area_l3275_327588
