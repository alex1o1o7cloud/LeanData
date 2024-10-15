import Mathlib

namespace NUMINAMATH_CALUDE_adam_chocolate_boxes_l1637_163745

/-- The number of boxes of chocolate candy Adam bought -/
def chocolate_boxes : ℕ := sorry

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes : ℕ := 5

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 4

/-- The total number of candies Adam had -/
def total_candies : ℕ := 28

theorem adam_chocolate_boxes :
  chocolate_boxes = 2 :=
by sorry

end NUMINAMATH_CALUDE_adam_chocolate_boxes_l1637_163745


namespace NUMINAMATH_CALUDE_bisection_method_structures_l1637_163767

-- Define the function for which we're finding the root
def f (x : ℝ) := x^2 - 2

-- Define the bisection method structure
structure BisectionMethod where
  sequential : Bool
  conditional : Bool
  loop : Bool

-- Theorem statement
theorem bisection_method_structures :
  ∀ (ε : ℝ) (a b : ℝ), 
    ε > 0 → a < b → f a * f b < 0 →
    ∃ (m : BisectionMethod),
      m.sequential ∧ m.conditional ∧ m.loop ∧
      ∃ (x : ℝ), a ≤ x ∧ x ≤ b ∧ |f x| < ε :=
sorry

end NUMINAMATH_CALUDE_bisection_method_structures_l1637_163767


namespace NUMINAMATH_CALUDE_other_jelly_correct_l1637_163717

/-- Given a total amount of jelly and the amount of one type, 
    calculate the amount of the other type -/
def other_jelly_amount (total : ℕ) (one_type : ℕ) : ℕ :=
  total - one_type

/-- Theorem: The amount of the other type of jelly is the difference
    between the total amount and the amount of one type -/
theorem other_jelly_correct (total : ℕ) (one_type : ℕ) 
  (h : one_type ≤ total) : 
  other_jelly_amount total one_type = total - one_type :=
by
  sorry

#eval other_jelly_amount 6310 4518

end NUMINAMATH_CALUDE_other_jelly_correct_l1637_163717


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1637_163789

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1637_163789


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l1637_163710

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0. -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) →
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l1637_163710


namespace NUMINAMATH_CALUDE_club_members_theorem_l1637_163720

theorem club_members_theorem (total : ℕ) (left_handed : ℕ) (rock_fans : ℕ) (right_handed_non_fans : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_fans = 18)
  (h4 : right_handed_non_fans = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ (left_handed_rock_fans : ℕ),
    left_handed_rock_fans = 6 ∧
    left_handed_rock_fans ≤ left_handed ∧
    left_handed_rock_fans ≤ rock_fans ∧
    left_handed_rock_fans + (left_handed - left_handed_rock_fans) + 
    (rock_fans - left_handed_rock_fans) + right_handed_non_fans = total :=
by
  sorry

end NUMINAMATH_CALUDE_club_members_theorem_l1637_163720


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l1637_163787

theorem parallel_vectors_tan_theta (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (a : Fin 2 → ℝ) (b : Fin 2 → ℝ)
  (h_a : a = ![1 - Real.sin θ, 1])
  (h_b : b = ![1 / 2, 1 + Real.sin θ])
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  Real.tan θ = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l1637_163787


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l1637_163735

/-- Given parametric equations x = 1 + 2cosθ and y = 2sinθ, 
    prove they are equivalent to the Cartesian equation (x-1)² + y² = 4 -/
theorem parametric_to_cartesian :
  ∀ (x y θ : ℝ), 
  x = 1 + 2 * Real.cos θ ∧ 
  y = 2 * Real.sin θ → 
  (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l1637_163735


namespace NUMINAMATH_CALUDE_work_completion_time_l1637_163711

/-- The time it takes for worker c to complete the work alone -/
def time_c : ℝ := 12

/-- The time it takes for worker a to complete the work alone -/
def time_a : ℝ := 16

/-- The time it takes for worker b to complete the work alone -/
def time_b : ℝ := 6

/-- The time it takes for workers a, b, and c to complete the work together -/
def time_abc : ℝ := 3.2

theorem work_completion_time :
  1 / time_a + 1 / time_b + 1 / time_c = 1 / time_abc :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1637_163711


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1637_163737

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = 1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1637_163737


namespace NUMINAMATH_CALUDE_cos_inequality_solution_set_l1637_163719

theorem cos_inequality_solution_set (x : ℝ) : 
  (Real.cos x + 1/2 ≤ 0) ↔ 
  (∃ k : ℤ, 2*k*Real.pi + 2*Real.pi/3 ≤ x ∧ x ≤ 2*k*Real.pi + 4*Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_cos_inequality_solution_set_l1637_163719


namespace NUMINAMATH_CALUDE_lamp_height_difference_l1637_163780

/-- The height difference between two lamps -/
theorem lamp_height_difference (old_height new_height : ℝ) 
  (h1 : old_height = 1) 
  (h2 : new_height = 2.3333333333333335) : 
  new_height - old_height = 1.3333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_lamp_height_difference_l1637_163780


namespace NUMINAMATH_CALUDE_transportation_charges_proof_l1637_163759

def transportation_charges (purchase_price repair_cost profit_percentage actual_selling_price : ℕ) : ℕ :=
  let total_cost_before_transport := purchase_price + repair_cost
  let profit := (total_cost_before_transport * profit_percentage) / 100
  let calculated_selling_price := total_cost_before_transport + profit
  actual_selling_price - calculated_selling_price

theorem transportation_charges_proof :
  transportation_charges 9000 5000 50 22500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_transportation_charges_proof_l1637_163759


namespace NUMINAMATH_CALUDE_line_shift_theorem_l1637_163752

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_horizontal (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + l.slope * units }

/-- Shifts a line vertically -/
def shift_vertical (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - units }

/-- The theorem stating that shifting the line y = 2x - 1 left by 3 units
    and then down by 4 units results in the line y = 2x + 1 -/
theorem line_shift_theorem :
  let initial_line := Line.mk 2 (-1)
  let shifted_left := shift_horizontal initial_line 3
  let final_line := shift_vertical shifted_left 4
  final_line = Line.mk 2 1 := by
  sorry


end NUMINAMATH_CALUDE_line_shift_theorem_l1637_163752


namespace NUMINAMATH_CALUDE_game_winner_l1637_163797

/-- Represents the possible moves in the game -/
inductive Move
| one
| two
| three

/-- Represents a player in the game -/
inductive Player
| first
| second

/-- Defines the game state -/
structure GameState :=
  (position : Nat)
  (currentPlayer : Player)

/-- Determines if a move is valid given the current position -/
def isValidMove (pos : Nat) (move : Move) : Bool :=
  match move with
  | Move.one => pos ≥ 1
  | Move.two => pos ≥ 2
  | Move.three => pos ≥ 3

/-- Applies a move to the current position -/
def applyMove (pos : Nat) (move : Move) : Nat :=
  match move with
  | Move.one => pos - 1
  | Move.two => pos - 2
  | Move.three => pos - 3

/-- Switches the current player -/
def switchPlayer (player : Player) : Player :=
  match player with
  | Player.first => Player.second
  | Player.second => Player.first

/-- Determines the winner given an initial position -/
def winningPlayer (initialPos : Nat) : Player :=
  if initialPos = 4 ∨ initialPos = 8 ∨ initialPos = 12 then
    Player.second
  else
    Player.first

/-- The main theorem to prove -/
theorem game_winner (initialPos : Nat) :
  initialPos ≤ 14 →
  winningPlayer initialPos = Player.second ↔ (initialPos = 4 ∨ initialPos = 8 ∨ initialPos = 12) :=
by sorry

end NUMINAMATH_CALUDE_game_winner_l1637_163797


namespace NUMINAMATH_CALUDE_unique_solution_l1637_163721

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution :
  ∀ k : ℕ, (factorial (k / 2) * (k / 4) = 2016 + k^2) ↔ k = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1637_163721


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1637_163729

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) : 
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1637_163729


namespace NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l1637_163738

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through
    the points (2, 5), (8, 5), and (9, 11), prove that the x-coordinate
    of its vertex is 5. -/
theorem quadratic_vertex_x_coordinate
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_point1 : f 2 = 5)
  (h_point2 : f 8 = 5)
  (h_point3 : f 9 = 11) :
  ∃ (vertex_x : ℝ), vertex_x = 5 ∧ ∀ x, f x ≤ f vertex_x :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l1637_163738


namespace NUMINAMATH_CALUDE_crafts_club_members_crafts_club_members_proof_l1637_163771

theorem crafts_club_members : ℕ → Prop :=
  fun n =>
    let necklaces_per_member : ℕ := 2
    let beads_per_necklace : ℕ := 50
    let total_beads : ℕ := 900
    n * (necklaces_per_member * beads_per_necklace) = total_beads →
    n = 9

-- Proof
theorem crafts_club_members_proof : crafts_club_members 9 := by
  sorry

end NUMINAMATH_CALUDE_crafts_club_members_crafts_club_members_proof_l1637_163771


namespace NUMINAMATH_CALUDE_problem_statement_l1637_163773

theorem problem_statement (x : ℝ) (h : x = 2) : 4 * x^2 + (1/2) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1637_163773


namespace NUMINAMATH_CALUDE_trip_time_calculation_l1637_163760

theorem trip_time_calculation (normal_distance : ℝ) (normal_time : ℝ) (additional_distance : ℝ) :
  normal_distance = 150 →
  normal_time = 3 →
  additional_distance = 100 →
  let speed := normal_distance / normal_time
  let total_distance := normal_distance + additional_distance
  let total_time := total_distance / speed
  total_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l1637_163760


namespace NUMINAMATH_CALUDE_pi_approximation_proof_l1637_163748

theorem pi_approximation_proof :
  let π := 4 * Real.sin (52 * π / 180)
  (2 * π * Real.sqrt (16 - π^2) - 8 * Real.sin (44 * π / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * Real.sin (22 * π / 180)^2) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pi_approximation_proof_l1637_163748


namespace NUMINAMATH_CALUDE_cos_sum_square_75_15_l1637_163791

theorem cos_sum_square_75_15 :
  Real.cos (75 * π / 180) ^ 2 + Real.cos (15 * π / 180) ^ 2 + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_square_75_15_l1637_163791


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l1637_163705

/-- Proves that given an escalator moving upward at 10 ft/sec with a length of 112 feet,
    if a person takes 8 seconds to cover the entire length,
    then the person's walking rate on the escalator is 4 ft/sec. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 10)
  (h2 : escalator_length = 112)
  (h3 : time_taken = 8)
  : ∃ (walking_rate : ℝ),
    walking_rate = 4 ∧
    escalator_length = (walking_rate + escalator_speed) * time_taken :=
by sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l1637_163705


namespace NUMINAMATH_CALUDE_line_equation_l1637_163774

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically --/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

/-- Check if two lines are identical --/
def Line.identical (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept = l2.intercept

/-- Check if two lines are symmetric about a point --/
def symmetricAbout (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ x y, (y = l1.slope * x + l1.intercept) ↔ 
         (2 * p.2 - y = l2.slope * (2 * p.1 - x) + l2.intercept)

theorem line_equation (l : Line) : 
  (translate (translate l 3 5) 1 (-2)).identical l ∧ 
  symmetricAbout l (translate l 3 5) (2, 3) →
  l.slope = 3/4 ∧ l.intercept = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1637_163774


namespace NUMINAMATH_CALUDE_min_value_theorem_l1637_163700

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  2 * x + 18 / x ≥ 12 ∧ (2 * x + 18 / x = 12 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1637_163700


namespace NUMINAMATH_CALUDE_cosine_sine_equivalence_l1637_163788

theorem cosine_sine_equivalence (θ : ℝ) : 
  Real.cos (3 * Real.pi / 2 - θ) = Real.sin (Real.pi + θ) ∧ 
  Real.cos (3 * Real.pi / 2 - θ) = Real.cos (Real.pi / 2 + θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_equivalence_l1637_163788


namespace NUMINAMATH_CALUDE_geometric_sequence_cars_below_threshold_l1637_163766

/- Define the sequence of ordinary cars -/
def a : ℕ → ℝ
  | 0 => 300  -- Initial value for 2020
  | n + 1 => 0.9 * a n + 8

/- Define the transformed sequence -/
def b (n : ℕ) : ℝ := a n - 80

/- Theorem statement -/
theorem geometric_sequence : ∀ n : ℕ, b (n + 1) = 0.9 * b n := by
  sorry

/- Additional theorem to show the year when cars are less than 1.5 million -/
theorem cars_below_threshold (n : ℕ) : a n < 150 → n ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_cars_below_threshold_l1637_163766


namespace NUMINAMATH_CALUDE_first_group_number_is_9_l1637_163718

/-- Represents a systematic sampling method -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  group_number : ℕ → ℕ
  h_population : population > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_population : sample_size ≤ population

/-- The number drawn by the first group in a systematic sampling -/
def first_group_number (s : SystematicSampling) : ℕ :=
  s.group_number 1

/-- Theorem stating that the first group number is 9 given the problem conditions -/
theorem first_group_number_is_9 (s : SystematicSampling)
    (h_population : s.population = 960)
    (h_sample_size : s.sample_size = 32)
    (h_fifth_group : s.group_number 5 = 129) :
    first_group_number s = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_is_9_l1637_163718


namespace NUMINAMATH_CALUDE_log_stack_sum_l1637_163734

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in the sequence of logs -/
def num_terms : ℕ := 15 - 5 + 1

theorem log_stack_sum :
  arithmetic_sum 5 15 num_terms = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l1637_163734


namespace NUMINAMATH_CALUDE_inequality_proof_l1637_163746

theorem inequality_proof (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3*y + x*y^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1637_163746


namespace NUMINAMATH_CALUDE_can_display_sequence_l1637_163785

/-- 
Given a sequence where:
- The first term is 2
- Each subsequent term increases by 3
- The 9th term is 26
Prove that this sequence exists and satisfies these conditions.
-/
theorem can_display_sequence : 
  ∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n, a (n + 1) = a n + 3) ∧ 
    a 9 = 26 := by
  sorry

end NUMINAMATH_CALUDE_can_display_sequence_l1637_163785


namespace NUMINAMATH_CALUDE_sugar_loss_calculation_l1637_163706

/-- Given an initial amount of sugar, number of bags, and loss percentage,
    calculate the remaining amount of sugar. -/
def remaining_sugar (initial_sugar : ℝ) (num_bags : ℕ) (loss_percent : ℝ) : ℝ :=
  initial_sugar * (1 - loss_percent)

/-- Theorem: Given 24 kilos of sugar divided equally into 4 bags,
    with 15% loss in each bag, the total remaining sugar is 20.4 kilos. -/
theorem sugar_loss_calculation : remaining_sugar 24 4 0.15 = 20.4 := by
  sorry

#check sugar_loss_calculation

end NUMINAMATH_CALUDE_sugar_loss_calculation_l1637_163706


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l1637_163750

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 20

/-- The height the monkey climbs in one hour during the first 17 hours -/
def hourly_climb : ℕ := 3

/-- The height the monkey slips back in one hour during the first 17 hours -/
def hourly_slip : ℕ := 2

/-- The number of hours it takes the monkey to reach the top of the tree -/
def total_hours : ℕ := 18

/-- The height the monkey climbs in the last hour -/
def final_climb : ℕ := 3

theorem monkey_climb_theorem :
  tree_height = (total_hours - 1) * (hourly_climb - hourly_slip) + final_climb :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l1637_163750


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1637_163762

/-- Represents a right circular cylinder inscribed in a right circular cone. -/
structure InscribedCylinder where
  /-- Radius of the inscribed cylinder -/
  radius : ℝ
  /-- Height of the inscribed cylinder -/
  height : ℝ
  /-- Diameter of the cone -/
  cone_diameter : ℝ
  /-- Altitude of the cone -/
  cone_altitude : ℝ
  /-- The cylinder's diameter is equal to its height -/
  cylinder_property : height = 2 * radius
  /-- The cone has a diameter of 20 -/
  cone_diameter_value : cone_diameter = 20
  /-- The cone has an altitude of 24 -/
  cone_altitude_value : cone_altitude = 24

/-- Theorem stating that the radius of the inscribed cylinder is 60/11 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) : c.radius = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1637_163762


namespace NUMINAMATH_CALUDE_common_solution_range_l1637_163751

theorem common_solution_range (x y : ℝ) : 
  (∃ x, x^2 + y^2 - 11 = 0 ∧ x^2 - 4*y + 7 = 0) ↔ 7/4 ≤ y ∧ y ≤ Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_range_l1637_163751


namespace NUMINAMATH_CALUDE_inequality_proof_l1637_163731

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1637_163731


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1637_163736

theorem absolute_value_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : abs a > abs b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1637_163736


namespace NUMINAMATH_CALUDE_expression_evaluation_l1637_163724

theorem expression_evaluation :
  let x : ℤ := -1
  (x + 1) * (x - 2) + 2 * (x + 4) * (x - 4) = -30 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1637_163724


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_fraction_l1637_163761

theorem ivy_collectors_edition_fraction (dina_dolls : ℕ) (ivy_collectors : ℕ) : 
  dina_dolls = 60 →
  ivy_collectors = 20 →
  2 * (dina_dolls / 2) = dina_dolls →
  (ivy_collectors : ℚ) / (dina_dolls / 2 : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_fraction_l1637_163761


namespace NUMINAMATH_CALUDE_expression_equals_1997_with_ten_threes_l1637_163782

theorem expression_equals_1997_with_ten_threes : 
  ∃ (a b c d e f g h i j : ℕ), 
    a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 ∧ e = 3 ∧ f = 3 ∧ g = 3 ∧ h = 3 ∧ i = 3 ∧ j = 3 ∧
    a * (b * 111 + c) + d * (e * 111 + f) - g / h = 1997 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_1997_with_ten_threes_l1637_163782


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1637_163726

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1637_163726


namespace NUMINAMATH_CALUDE_abs_neg_x_eq_five_implies_x_plus_minus_five_l1637_163742

theorem abs_neg_x_eq_five_implies_x_plus_minus_five (x : ℝ) : 
  |(-x)| = 5 → x = -5 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_neg_x_eq_five_implies_x_plus_minus_five_l1637_163742


namespace NUMINAMATH_CALUDE_find_m_l1637_163790

theorem find_m : ∃ m : ℝ, 10^m = 10^2 * Real.sqrt (10^90 / 0.0001) ∧ m = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1637_163790


namespace NUMINAMATH_CALUDE_lobster_rolls_count_total_plates_sum_l1637_163795

/-- The number of plates of lobster rolls served at a banquet -/
def lobster_rolls : ℕ := 55 - (14 + 16)

/-- The total number of plates served at the banquet -/
def total_plates : ℕ := 55

/-- The number of plates of spicy hot noodles served at the banquet -/
def spicy_hot_noodles : ℕ := 14

/-- The number of plates of seafood noodles served at the banquet -/
def seafood_noodles : ℕ := 16

/-- Theorem stating that the number of lobster roll plates is 25 -/
theorem lobster_rolls_count : lobster_rolls = 25 := by
  sorry

/-- Theorem stating that the total number of plates is the sum of all dishes -/
theorem total_plates_sum : 
  total_plates = lobster_rolls + spicy_hot_noodles + seafood_noodles := by
  sorry

end NUMINAMATH_CALUDE_lobster_rolls_count_total_plates_sum_l1637_163795


namespace NUMINAMATH_CALUDE_circle_properties_l1637_163714

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem circle_properties :
  ∃ (a : ℝ),
    -- P is on circle C
    C a (a + 1) ∧
    -- |PQ| = 2√10
    (a - (-2))^2 + ((a + 1) - 3)^2 = 40 ∧
    -- Slope of PQ is 1/3
    (3 - (a + 1)) / (-2 - a) = 1/3 ∧
    -- For any point M on C
    ∀ (m n : ℝ), C m n →
      -- Maximum value of |MQ| is 6√2
      (m - (-2))^2 + (n - 3)^2 ≤ 72 ∧
      -- Minimum value of |MQ| is 2√2
      (m - (-2))^2 + (n - 3)^2 ≥ 8 ∧
      -- Maximum value of (n-3)/(m+2) is 2 + √3
      (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 ∧
      -- Minimum value of (n-3)/(m+2) is 2 - √3
      (n - 3) / (m + 2) ≥ 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1637_163714


namespace NUMINAMATH_CALUDE_percentage_calculation_l1637_163753

theorem percentage_calculation (number : ℝ) (result : ℝ) (P : ℝ) : 
  number = 4400 → 
  result = 99 → 
  P * number = result → 
  P = 0.0225 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1637_163753


namespace NUMINAMATH_CALUDE_distribute_five_students_three_dorms_l1637_163725

/-- The number of ways to distribute students into dormitories -/
def distribute_students (n : ℕ) (m : ℕ) (min : ℕ) (max : ℕ) (restricted : ℕ) : ℕ := sorry

/-- The theorem stating the number of ways to distribute 5 students into 3 dormitories -/
theorem distribute_five_students_three_dorms :
  distribute_students 5 3 1 2 1 = 60 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_dorms_l1637_163725


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1637_163728

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℚ), x ≠ (z : ℝ)) ∧
  (∀ (a b : ℕ), x ≠ Real.sqrt (a / b))

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.1) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1637_163728


namespace NUMINAMATH_CALUDE_sphere_cube_volume_constant_l1637_163744

/-- The value of K when a sphere has the same surface area as a cube with side length 3
    and its volume is expressed as (K * sqrt(6)) / sqrt(π) -/
theorem sphere_cube_volume_constant (cube_side : ℝ) (sphere_volume : ℝ → ℝ) : 
  cube_side = 3 →
  (4 * π * (sphere_volume K / ((4 / 3) * π))^(2/3) = 6 * cube_side^2) →
  sphere_volume K = K * Real.sqrt 6 / Real.sqrt π →
  K = 27 * Real.sqrt 6 / Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_volume_constant_l1637_163744


namespace NUMINAMATH_CALUDE_min_value_theorem_l1637_163798

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 1) :
  2/a + 3/b ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1637_163798


namespace NUMINAMATH_CALUDE_find_number_l1637_163777

theorem find_number : ∃ x : ℝ, (3 * x / 5 - 220) * 4 + 40 = 360 ∧ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1637_163777


namespace NUMINAMATH_CALUDE_hockey_league_teams_l1637_163702

/-- The number of games played in the hockey season -/
def total_games : ℕ := 1710

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in a season based on the number of teams -/
def calculate_games (n : ℕ) : ℕ :=
  (n * (n - 1) * games_per_pair) / 2

theorem hockey_league_teams :
  ∃ (n : ℕ), n > 0 ∧ calculate_games n = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l1637_163702


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1637_163793

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1637_163793


namespace NUMINAMATH_CALUDE_total_diagonals_total_internal_angles_l1637_163769

/-- Number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- Number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Calculate the number of diagonals in a polygon -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Calculate the sum of internal angles in a polygon -/
def internal_angles_sum (n : ℕ) : ℕ := (n - 2) * 180

/-- The total number of diagonals in a pentagon and an octagon is 25 -/
theorem total_diagonals : 
  diagonals pentagon_sides + diagonals octagon_sides = 25 := by sorry

/-- The sum of internal angles of a pentagon and an octagon is 1620° -/
theorem total_internal_angles : 
  internal_angles_sum pentagon_sides + internal_angles_sum octagon_sides = 1620 := by sorry

end NUMINAMATH_CALUDE_total_diagonals_total_internal_angles_l1637_163769


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1637_163768

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1637_163768


namespace NUMINAMATH_CALUDE_existence_of_powers_of_seven_with_difference_divisible_by_2021_l1637_163707

theorem existence_of_powers_of_seven_with_difference_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ (7^n - 7^m) % 2021 = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_powers_of_seven_with_difference_divisible_by_2021_l1637_163707


namespace NUMINAMATH_CALUDE_unit_circle_image_l1637_163703

def unit_circle_mapping (z : ℂ) : Prop := Complex.abs z = 1

theorem unit_circle_image :
  ∀ z : ℂ, unit_circle_mapping z → Complex.abs (z^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_image_l1637_163703


namespace NUMINAMATH_CALUDE_ratio_theorem_l1637_163783

theorem ratio_theorem (x y : ℝ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l1637_163783


namespace NUMINAMATH_CALUDE_square_construction_impossibility_l1637_163781

theorem square_construction_impossibility (k : ℕ) (h : k ≥ 2) :
  ¬ (∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 = 1 ∧ x.2 ≤ k) ∧ 
    (S.sum (λ x => x.1 * x.2) = k * k) ∧
    (S.card ≤ k)) := by
  sorry

end NUMINAMATH_CALUDE_square_construction_impossibility_l1637_163781


namespace NUMINAMATH_CALUDE_smallest_integer_for_prime_quadratic_l1637_163749

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_value (n : ℤ) : ℕ := Int.natAbs n

def quadratic_expression (x : ℤ) : ℤ := 8 * x^2 - 53 * x + 21

theorem smallest_integer_for_prime_quadratic :
  ∀ x : ℤ, x < 8 → ¬(is_prime (abs_value (quadratic_expression x))) ∧
  is_prime (abs_value (quadratic_expression 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_for_prime_quadratic_l1637_163749


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1637_163723

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11 ≥ 8) ∧ ∃ x, 2 * x + (a 11) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1637_163723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1637_163713

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2015 = 10) →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1637_163713


namespace NUMINAMATH_CALUDE_intersection_value_l1637_163708

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of the first line -/
def line1 (a c : ℝ) (x y : ℝ) : Prop := a * x - 3 * y = c

/-- The equation of the second line -/
def line2 (b c : ℝ) (x y : ℝ) : Prop := 3 * x + b * y = -c

/-- The theorem stating that c = 39 given the conditions -/
theorem intersection_value (a b c : ℝ) : 
  perpendicular (a / 3) (-3 / b) →
  line1 a c 2 (-3) →
  line2 b c 2 (-3) →
  c = 39 := by sorry

end NUMINAMATH_CALUDE_intersection_value_l1637_163708


namespace NUMINAMATH_CALUDE_carols_birthday_l1637_163794

/-- Represents a date with a month and a day -/
structure Date where
  month : String
  day : Nat

/-- The list of possible dates for Carol's birthday -/
def possible_dates : List Date := [
  ⟨"January", 4⟩, ⟨"March", 8⟩, ⟨"June", 7⟩, ⟨"October", 7⟩,
  ⟨"January", 5⟩, ⟨"April", 8⟩, ⟨"June", 5⟩, ⟨"October", 4⟩,
  ⟨"January", 11⟩, ⟨"April", 9⟩, ⟨"July", 13⟩, ⟨"October", 8⟩
]

/-- Alberto knows the month but not the exact date -/
def alberto_knows_month (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.month = d.month ∧ other ≠ d

/-- Bernardo knows the day but not the exact date -/
def bernardo_knows_day (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.day = d.day ∧ other ≠ d

/-- Alberto's first statement: He can't determine the date, and he's sure Bernardo can't either -/
def alberto_statement1 (d : Date) : Prop :=
  alberto_knows_month d ∧ bernardo_knows_day d

/-- After Alberto's statement, Bernardo can determine the date -/
def bernardo_statement (d : Date) : Prop :=
  alberto_statement1 d ∧
  ∀ (other : Date), other ∈ possible_dates → other.day = d.day → alberto_statement1 other → other = d

/-- After Bernardo's statement, Alberto can also determine the date -/
def alberto_statement2 (d : Date) : Prop :=
  bernardo_statement d ∧
  ∀ (other : Date), other ∈ possible_dates → other.month = d.month → bernardo_statement other → other = d

/-- The theorem stating that Carol's birthday must be June 7 -/
theorem carols_birthday :
  ∃! (d : Date), d ∈ possible_dates ∧ alberto_statement2 d ∧ d = ⟨"June", 7⟩ := by
  sorry

end NUMINAMATH_CALUDE_carols_birthday_l1637_163794


namespace NUMINAMATH_CALUDE_shooting_competition_l1637_163709

theorem shooting_competition (p_tie p_win : ℝ) 
  (h_tie : p_tie = 1/2)
  (h_win : p_win = 1/3) :
  p_tie + p_win = 5/6 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_l1637_163709


namespace NUMINAMATH_CALUDE_lab_workstations_l1637_163741

theorem lab_workstations (total_students : ℕ) (two_student_stations : ℕ) (three_student_stations : ℕ) :
  total_students = 38 →
  two_student_stations = 10 →
  two_student_stations * 2 + three_student_stations * 3 = total_students →
  two_student_stations + three_student_stations = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_lab_workstations_l1637_163741


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l1637_163715

def total_dogs : Nat := 12
def group1_size : Nat := 4
def group2_size : Nat := 5
def group3_size : Nat := 3

theorem dog_grouping_combinations :
  (total_dogs = group1_size + group2_size + group3_size) →
  (Nat.choose (total_dogs - 2) (group1_size - 1) * Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) = 5775) := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l1637_163715


namespace NUMINAMATH_CALUDE_plot_length_l1637_163739

/-- Proves that the length of a rectangular plot is 65 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 30 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  perimeter = total_cost / cost_per_meter →
  length = 65 := by
sorry


end NUMINAMATH_CALUDE_plot_length_l1637_163739


namespace NUMINAMATH_CALUDE_power_four_congruence_l1637_163779

theorem power_four_congruence (n : ℕ) (a : ℤ) (hn : n > 0) (ha : a^3 ≡ 1 [ZMOD n]) :
  a^4 ≡ a [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_power_four_congruence_l1637_163779


namespace NUMINAMATH_CALUDE_exponent_negative_product_squared_l1637_163701

theorem exponent_negative_product_squared (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_negative_product_squared_l1637_163701


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1637_163727

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1637_163727


namespace NUMINAMATH_CALUDE_simplify_expression_l1637_163747

theorem simplify_expression (y : ℝ) : 7*y - 3 + 2*y + 15 = 9*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1637_163747


namespace NUMINAMATH_CALUDE_four_person_apartments_count_l1637_163716

/-- Represents the number of 4-person apartments in each building -/
def four_person_apartments : ℕ := sorry

/-- The number of identical buildings in the complex -/
def num_buildings : ℕ := 4

/-- The number of studio apartments in each building -/
def studio_apartments : ℕ := 10

/-- The number of 2-person apartments in each building -/
def two_person_apartments : ℕ := 20

/-- The occupancy rate of the apartment complex -/
def occupancy_rate : ℚ := 3/4

/-- The total number of people living in the apartment complex -/
def total_occupants : ℕ := 210

/-- Theorem stating that the number of 4-person apartments in each building is 5 -/
theorem four_person_apartments_count : four_person_apartments = 5 :=
  by sorry

end NUMINAMATH_CALUDE_four_person_apartments_count_l1637_163716


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1637_163786

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {(x, y) | y = 2 * x - 1}
def B : Set (ℝ × ℝ) := {(x, y) | y = x + 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {(4, 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1637_163786


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l1637_163776

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1 / 3

-- State the theorem
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l1637_163776


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1637_163712

theorem arithmetic_computation : 2 + 5 * 3^2 - 4 + 6 * 2 / 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1637_163712


namespace NUMINAMATH_CALUDE_no_ab_term_l1637_163784

/-- The polynomial does not contain the term ab if and only if m = -2 -/
theorem no_ab_term (a b m : ℝ) : 
  2 * (a^2 + a*b - 5*b^2) - (a^2 - m*a*b + 2*b^2) = a^2 - 12*b^2 ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_no_ab_term_l1637_163784


namespace NUMINAMATH_CALUDE_f_properties_l1637_163704

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), ∀ (x : ℝ), f a x = f a (x + T)) ∧ 
  (∃ (min_val : ℝ), min_val = 0 → 
    (a = 1 ∧ 
     (∃ (max_val : ℝ), max_val = 4 ∧ ∀ (x : ℝ), f a x ≤ max_val) ∧
     (∃ (k : ℤ), ∀ (x : ℝ), f a x = f a (↑k * Real.pi / 2 + Real.pi / 6 - x)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1637_163704


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1637_163732

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1637_163732


namespace NUMINAMATH_CALUDE_factorization_equality_l1637_163778

theorem factorization_equality (x y : ℝ) : 
  x^2 + 4*y^2 - 4*x*y - 1 = (x - 2*y + 1)*(x - 2*y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1637_163778


namespace NUMINAMATH_CALUDE_problem_statement_l1637_163765

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1637_163765


namespace NUMINAMATH_CALUDE_f_negative_six_equals_negative_one_l1637_163772

def f (x : ℝ) : ℝ := sorry

theorem f_negative_six_equals_negative_one :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + 6) = f x) →  -- f has period 6
  (∀ x, -3 ≤ x ∧ x ≤ 3 → f x = (x + 1) * (x - 1)) →  -- f(x) = (x+1)(x-a) for -3 ≤ x ≤ 3, where a = 1
  f (-6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_six_equals_negative_one_l1637_163772


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l1637_163792

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 →                                 -- n is the number of sides, must be greater than 2
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →   -- sum of interior angles formula
  n = 12 :=                               -- conclusion: the polygon has 12 sides
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l1637_163792


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l1637_163757

theorem inverse_matrices_solution :
  ∀ (a b : ℚ),
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 3], ![2, 5]]
  let B : Matrix (Fin 2) (Fin 2) ℚ := ![![b, -1/5], ![1/2, 1/10]]
  let I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]
  A * B = I → a = 3/2 ∧ b = -5/4 := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_solution_l1637_163757


namespace NUMINAMATH_CALUDE_cubic_function_value_l1637_163733

/-- Given a cubic function f(x) = ax^3 + bx - 4 where f(-2) = 2, prove that f(2) = -10 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x - 4)
  (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l1637_163733


namespace NUMINAMATH_CALUDE_sixth_plate_cookies_l1637_163755

def cookie_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 5
  | 1 => 7
  | k + 2 => cookie_sequence n (k + 1) + (k + 2)

theorem sixth_plate_cookies :
  cookie_sequence 5 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sixth_plate_cookies_l1637_163755


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l1637_163758

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the interval [0, 4)
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l1637_163758


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l1637_163730

/-- Proves that in an arithmetic progression with first term 2, last term 62, and 31 terms, the common difference is 2. -/
theorem arithmetic_progression_common_difference 
  (first_term : ℕ) 
  (last_term : ℕ) 
  (num_terms : ℕ) 
  (h1 : first_term = 2) 
  (h2 : last_term = 62) 
  (h3 : num_terms = 31) : 
  (last_term - first_term) / (num_terms - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l1637_163730


namespace NUMINAMATH_CALUDE_perimeter_ratio_is_one_l1637_163722

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the original paper --/
def original_paper : Rectangle := { width := 8, height := 12 }

/-- Represents one of the rectangles after folding and cutting --/
def folded_cut_rectangle : Rectangle := { width := 4, height := 6 }

/-- Theorem stating that the ratio of perimeters is 1 --/
theorem perimeter_ratio_is_one : 
  perimeter folded_cut_rectangle / perimeter folded_cut_rectangle = 1 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_is_one_l1637_163722


namespace NUMINAMATH_CALUDE_percentage_of_absent_students_l1637_163754

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 43 → (((total - present : ℚ) / total) * 100 = 14) := by sorry

end NUMINAMATH_CALUDE_percentage_of_absent_students_l1637_163754


namespace NUMINAMATH_CALUDE_matts_working_ratio_l1637_163770

/-- Matt's working schedule problem -/
theorem matts_working_ratio :
  let monday_minutes : ℕ := 450
  let wednesday_minutes : ℕ := 300
  let tuesday_minutes : ℕ := wednesday_minutes - 75
  tuesday_minutes * 2 = monday_minutes := by sorry

end NUMINAMATH_CALUDE_matts_working_ratio_l1637_163770


namespace NUMINAMATH_CALUDE_field_trip_adults_l1637_163743

/-- Given a field trip scenario, prove the number of adults attending. -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 9 → num_students = 40 → num_vans = 6 → 
  (num_vans * van_capacity - num_students : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1637_163743


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1637_163775

/-- The remainder when x^3 + 3 is divided by x^2 + 2 is -2x + 3 -/
theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^3 + 3 = (x^2 + 2) * q + (-2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1637_163775


namespace NUMINAMATH_CALUDE_pedal_triangles_existence_and_angles_l1637_163764

/-- A triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The pedal triangle of a given triangle -/
structure PedalTriangle where
  original : Triangle
  pedal : Triangle

/-- The theorem statement -/
theorem pedal_triangles_existence_and_angles 
  (T : Triangle) 
  (h1 : T.angle1 = 24) 
  (h2 : T.angle2 = 60) 
  (h3 : T.angle3 = 96) : 
  ∃! (pedals : Finset PedalTriangle), 
    Finset.card pedals = 4 ∧ 
    ∀ P ∈ pedals, 
      (P.pedal.angle1 = 102 ∧ 
       P.pedal.angle2 = 30 ∧ 
       P.pedal.angle3 = 48) := by
  sorry


end NUMINAMATH_CALUDE_pedal_triangles_existence_and_angles_l1637_163764


namespace NUMINAMATH_CALUDE_value_subtracted_after_multiplication_l1637_163799

theorem value_subtracted_after_multiplication (N : ℝ) (V : ℝ) : 
  N = 12 → 4 * N - V = 9 * (N - 7) → V = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_subtracted_after_multiplication_l1637_163799


namespace NUMINAMATH_CALUDE_triangle_reconstruction_from_altitude_feet_l1637_163740

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle with vertices A, B, C -/
structure Triangle :=
  (A B C : Point)

/-- Represents the feet of altitudes of a triangle -/
structure AltitudeFeet :=
  (A1 B1 C1 : Point)

/-- Predicate to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Function to get the feet of altitudes of a triangle -/
def getAltitudeFeet (t : Triangle) : AltitudeFeet :=
  sorry

/-- Predicate to check if a triangle can be reconstructed from given altitude feet -/
def canReconstructTriangle (feet : AltitudeFeet) : Prop :=
  sorry

/-- Theorem stating that an acute-angled triangle can be reconstructed from its altitude feet -/
theorem triangle_reconstruction_from_altitude_feet
  (t : Triangle) (h : isAcuteAngled t) :
  canReconstructTriangle (getAltitudeFeet t) :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_from_altitude_feet_l1637_163740


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1637_163796

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1637_163796


namespace NUMINAMATH_CALUDE_soccer_team_goals_l1637_163756

theorem soccer_team_goals (total_players : ℕ) (total_goals : ℕ) (games_played : ℕ) 
  (h1 : total_players = 24)
  (h2 : total_goals = 150)
  (h3 : games_played = 15)
  (h4 : (total_players / 3) * games_played = total_goals - 30) : 
  30 = total_goals - (total_players / 3) * games_played := by
sorry

end NUMINAMATH_CALUDE_soccer_team_goals_l1637_163756


namespace NUMINAMATH_CALUDE_vector_sum_equals_expected_l1637_163763

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-3, 4]

-- Define the sum of the vectors
def sum_ab : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]

-- Theorem statement
theorem vector_sum_equals_expected : sum_ab = ![-1, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equals_expected_l1637_163763
