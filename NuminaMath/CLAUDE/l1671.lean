import Mathlib

namespace find_v5_l1671_167152

def sequence_relation (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem find_v5 (v : ℕ → ℝ) (h1 : sequence_relation v) (h2 : v 4 = 15) (h3 : v 7 = 255) :
  v 5 = 45 := by
  sorry

end find_v5_l1671_167152


namespace christmas_tree_decoration_l1671_167184

theorem christmas_tree_decoration (b t : ℕ) : 
  (t = b + 1) →  -- Chuck's condition
  (2 * b = t - 1) →  -- Huck's condition
  (b = 3 ∧ t = 4) :=  -- Conclusion
by sorry

end christmas_tree_decoration_l1671_167184


namespace imaginary_unit_sum_l1671_167186

theorem imaginary_unit_sum (i : ℂ) (hi : i * i = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end imaginary_unit_sum_l1671_167186


namespace georges_income_proof_l1671_167195

/-- George's monthly income in dollars -/
def monthly_income : ℝ := 240

/-- The amount George spent on groceries in dollars -/
def grocery_expense : ℝ := 20

/-- The amount George has left in dollars -/
def amount_left : ℝ := 100

/-- Theorem stating that George's monthly income is correct given the conditions -/
theorem georges_income_proof :
  monthly_income / 2 - grocery_expense = amount_left := by sorry

end georges_income_proof_l1671_167195


namespace f₁_solution_set_f₂_min_value_l1671_167122

-- Part 1
def f₁ (x : ℝ) : ℝ := |3*x - 1| + |x + 3|

theorem f₁_solution_set : 
  {x : ℝ | f₁ x ≥ 4} = {x : ℝ | x ≤ -3 ∨ x ≥ 1/2} := by sorry

-- Part 2
def f₂ (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem f₂_min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (hmin : ∃ x, ∀ y, f₂ b c x ≤ f₂ b c y) 
  (hval : ∃ x, f₂ b c x = 1) : 
  (1/b + 1/c) ≥ 4 ∧ ∃ b c, (1/b + 1/c = 4 ∧ b > 0 ∧ c > 0) := by sorry

end f₁_solution_set_f₂_min_value_l1671_167122


namespace container_max_volume_l1671_167111

/-- The volume function for the container --/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function --/
def volume_derivative (x : ℝ) : ℝ := 12 * (x^2 - 46*x + 360)

theorem container_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  ∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x ∧
  x = 10 := by
  sorry

end container_max_volume_l1671_167111


namespace circle_existence_l1671_167149

-- Define the lines and the given circle
def line1 (x y : ℝ) : Prop := x + y = 7
def line2 (x y : ℝ) : Prop := x - 7*y = -33
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 28*x + 6*y + 165 = 0

-- Define the distance ratio condition
def distance_ratio (x y u v : ℝ) : Prop :=
  |x + y - 7| / Real.sqrt 2 = 5 * |x - 7*y + 33| / Real.sqrt 50

-- Define the intersection point of the two lines
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the orthogonality condition
def orthogonal_intersection (x y u v r : ℝ) : Prop :=
  (u - 14)^2 + (v + 3)^2 = r^2 + 40

-- Define the two resulting circles
def circle1 (x y : ℝ) : Prop := (x - 11)^2 + (y - 8)^2 = 87
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 168

theorem circle_existence :
  ∃ (x y u₁ v₁ u₂ v₂ : ℝ),
    (∀ (a b : ℝ), intersection_point a b → (circle1 a b ∨ circle2 a b)) ∧
    distance_ratio u₁ v₁ u₁ v₁ ∧
    distance_ratio u₂ v₂ u₂ v₂ ∧
    orthogonal_intersection u₁ v₁ u₁ v₁ (Real.sqrt 87) ∧
    orthogonal_intersection u₂ v₂ u₂ v₂ (Real.sqrt 168) :=
  sorry


end circle_existence_l1671_167149


namespace building_height_ratio_l1671_167110

/-- Given three buildings with specific height relationships, prove the ratio of the second to the first building's height. -/
theorem building_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 600 →
  h₃ = 3 * (h₁ + h₂) →
  h₁ + h₂ + h₃ = 7200 →
  h₂ / h₁ = 2 := by
sorry

end building_height_ratio_l1671_167110


namespace boxes_sold_proof_l1671_167169

/-- The number of boxes sold on Friday -/
def friday_boxes : ℕ := 40

/-- The number of boxes sold on Saturday -/
def saturday_boxes : ℕ := 2 * friday_boxes - 10

/-- The number of boxes sold on Sunday -/
def sunday_boxes : ℕ := (saturday_boxes) / 2

theorem boxes_sold_proof :
  friday_boxes + saturday_boxes + sunday_boxes = 145 :=
by sorry

end boxes_sold_proof_l1671_167169


namespace distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l1671_167108

-- Define the speeds and time
def roja_speed : ℝ := 5
def pooja_speed : ℝ := 3
def sooraj_speed : ℝ := 4
def time : ℝ := 4

-- Define the distances traveled
def roja_distance : ℝ := roja_speed * time
def pooja_distance : ℝ := pooja_speed * time
def sooraj_distance : ℝ := sooraj_speed * time

-- Theorem for the distance between Roja and Pooja
theorem distance_roja_pooja : 
  Real.sqrt (roja_distance ^ 2 + pooja_distance ^ 2) = Real.sqrt 544 :=
sorry

-- Theorem for the distance between Sooraj and the pole
theorem distance_sooraj_pole : sooraj_distance = 16 :=
sorry

-- Theorem for the angle between Roja and Pooja's directions
theorem angle_roja_pooja : ∃ (angle : ℝ), angle = 90 :=
sorry

end distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l1671_167108


namespace halfway_fraction_l1671_167173

theorem halfway_fraction (a b : ℚ) (ha : a = 1/7) (hb : b = 1/4) :
  (a + b) / 2 = 11/56 := by sorry

end halfway_fraction_l1671_167173


namespace triangle_with_angle_ratio_1_2_3_is_right_triangle_l1671_167176

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
  sorry

end triangle_with_angle_ratio_1_2_3_is_right_triangle_l1671_167176


namespace square_difference_division_problem_solution_l1671_167105

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by
  sorry

theorem problem_solution : (315^2 - 285^2) / 30 = 600 :=
by
  sorry

end square_difference_division_problem_solution_l1671_167105


namespace convex_polygon_sides_l1671_167198

-- Define the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the given sum of angles
def given_sum : ℝ := 2340

-- Theorem statement
theorem convex_polygon_sides : 
  ∃ (n : ℕ), n > 2 ∧ 
  sum_interior_angles n - given_sum > 0 ∧ 
  sum_interior_angles n - given_sum ≤ 360 ∧
  n = 16 := by sorry

end convex_polygon_sides_l1671_167198


namespace equation_equivalence_l1671_167112

theorem equation_equivalence (x Q : ℝ) (h : 5 * (5 * x + 7 * Real.pi) = Q) :
  10 * (10 * x + 14 * Real.pi + 2) = 4 * Q + 20 := by
  sorry

end equation_equivalence_l1671_167112


namespace chinese_heritage_tv_event_is_random_l1671_167134

/-- Represents a TV event -/
structure TVEvent where
  program : String
  canOccur : Bool
  hasUncertainty : Bool

/-- Classifies an event as certain, impossible, or random -/
inductive EventClassification
  | Certain
  | Impossible
  | Random

/-- Determines if an event is random based on its properties -/
def isRandomEvent (e : TVEvent) : Bool :=
  e.canOccur ∧ e.hasUncertainty

/-- Classifies a TV event based on its properties -/
def classifyTVEvent (e : TVEvent) : EventClassification :=
  if isRandomEvent e then EventClassification.Random
  else if e.canOccur then EventClassification.Certain
  else EventClassification.Impossible

/-- The main theorem stating that turning on the TV and broadcasting
    "Chinese Intangible Cultural Heritage" is a random event -/
theorem chinese_heritage_tv_event_is_random :
  let e := TVEvent.mk "Chinese Intangible Cultural Heritage" true true
  classifyTVEvent e = EventClassification.Random := by
  sorry


end chinese_heritage_tv_event_is_random_l1671_167134


namespace x_plus_q_in_terms_of_q_l1671_167189

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) : x + q = 2*q - 3 := by
  sorry

end x_plus_q_in_terms_of_q_l1671_167189


namespace zach_ben_score_difference_l1671_167161

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end zach_ben_score_difference_l1671_167161


namespace crushing_load_square_pillars_l1671_167144

theorem crushing_load_square_pillars (T H : ℝ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / H^3 = 93.75 := by
  sorry

end crushing_load_square_pillars_l1671_167144


namespace males_in_choir_is_twelve_l1671_167129

/-- Represents the number of musicians in each group -/
structure MusicianCounts where
  orchestra_males : ℕ
  orchestra_females : ℕ
  choir_females : ℕ
  total_musicians : ℕ

/-- Calculates the number of males in the choir based on given conditions -/
def males_in_choir (counts : MusicianCounts) : ℕ :=
  let orchestra_total := counts.orchestra_males + counts.orchestra_females
  let band_total := 2 * orchestra_total
  let choir_total := counts.total_musicians - (orchestra_total + band_total)
  choir_total - counts.choir_females

/-- Theorem stating that the number of males in the choir is 12 -/
theorem males_in_choir_is_twelve (counts : MusicianCounts)
  (h1 : counts.orchestra_males = 11)
  (h2 : counts.orchestra_females = 12)
  (h3 : counts.choir_females = 17)
  (h4 : counts.total_musicians = 98) :
  males_in_choir counts = 12 := by
  sorry

#eval males_in_choir ⟨11, 12, 17, 98⟩

end males_in_choir_is_twelve_l1671_167129


namespace geometric_sequence_ratio_l1671_167187

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Theorem: For a geometric sequence with common ratio q, if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio 
  (a q : ℝ) 
  (h1 : geometric_sequence a q 1 + geometric_sequence a q 3 = 10)
  (h2 : geometric_sequence a q 4 + geometric_sequence a q 6 = 5/4) :
  q = 1/2 := by
  sorry

end geometric_sequence_ratio_l1671_167187


namespace no_finite_planes_cover_all_cubes_l1671_167192

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the specifics of a plane for this statement

/-- Represents a cube in the integer grid -/
structure GridCube where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Checks if a plane intersects a grid cube -/
def plane_intersects_cube (p : Plane) (c : GridCube) : Prop :=
  sorry -- Definition not needed for the statement

/-- The main theorem stating that it's impossible to have a finite number of planes
    intersecting all cubes in the integer grid -/
theorem no_finite_planes_cover_all_cubes :
  ∀ (planes : Finset Plane), ∃ (c : GridCube),
    ∀ (p : Plane), p ∈ planes → ¬(plane_intersects_cube p c) := by
  sorry


end no_finite_planes_cover_all_cubes_l1671_167192


namespace residue_of_7_2050_mod_19_l1671_167121

theorem residue_of_7_2050_mod_19 : 7^2050 % 19 = 11 := by
  sorry

end residue_of_7_2050_mod_19_l1671_167121


namespace hyperbola_minimum_value_l1671_167181

theorem hyperbola_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let c := Real.sqrt (a^2 + b^2)  -- focal distance
  (c / a = e) →
  (∀ a' b', a' > 0 → b' > 0 → c / a' = e → (b'^2 + 1) / (3 * a') ≥ (b^2 + 1) / (3 * a)) →
  (b^2 + 1) / (3 * a) = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_minimum_value_l1671_167181


namespace initial_cartons_processed_l1671_167171

/-- Proves that the initial number of cartons processed is 400 --/
theorem initial_cartons_processed (num_customers : ℕ) (returned_cartons : ℕ) (total_accepted : ℕ) :
  num_customers = 4 →
  returned_cartons = 60 →
  total_accepted = 160 →
  (num_customers * (total_accepted / num_customers + returned_cartons)) = 400 := by
sorry

end initial_cartons_processed_l1671_167171


namespace power_multiplication_l1671_167190

theorem power_multiplication (x : ℝ) : x^6 * x^2 = x^8 := by
  sorry

end power_multiplication_l1671_167190


namespace sum_of_fourth_and_fifth_terms_l1671_167154

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem sum_of_fourth_and_fifth_terms (a₀ : ℝ) (r : ℝ) :
  (geometric_sequence a₀ r 5 = 4) →
  (geometric_sequence a₀ r 6 = 1) →
  (geometric_sequence a₀ r 2 = 256) →
  (geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80) := by
  sorry

end sum_of_fourth_and_fifth_terms_l1671_167154


namespace saltwater_concentration_l1671_167106

/-- Represents a saltwater solution -/
structure SaltWaterSolution where
  salt : ℝ
  water : ℝ
  concentration : ℝ
  concentration_def : concentration = salt / (salt + water) * 100

/-- The condition that adding 200g of water halves the concentration -/
def half_concentration (s : SaltWaterSolution) : Prop :=
  s.salt / (s.salt + s.water + 200) = s.concentration / 2

/-- The condition that adding 25g of salt doubles the concentration -/
def double_concentration (s : SaltWaterSolution) : Prop :=
  (s.salt + 25) / (s.salt + s.water + 25) = 2 * s.concentration / 100

/-- The main theorem to prove -/
theorem saltwater_concentration 
  (s : SaltWaterSolution) 
  (h1 : half_concentration s) 
  (h2 : double_concentration s) : 
  s.concentration = 10 := by
  sorry


end saltwater_concentration_l1671_167106


namespace binomial_coefficient_equation_solution_l1671_167157

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) ↔ (x = 2 ∨ x = 4) := by
  sorry

end binomial_coefficient_equation_solution_l1671_167157


namespace journey_to_the_west_readers_l1671_167163

theorem journey_to_the_west_readers (total : ℕ) (either : ℕ) (dream : ℕ) (both : ℕ) 
  (h1 : total = 100)
  (h2 : either = 90)
  (h3 : dream = 80)
  (h4 : both = 60)
  (h5 : either ≤ total)
  (h6 : dream ≤ total)
  (h7 : both ≤ dream)
  (h8 : both ≤ either) : 
  ∃ (journey : ℕ), journey = 70 ∧ journey = either + both - dream := by
  sorry

end journey_to_the_west_readers_l1671_167163


namespace polynomial_perfect_square_l1671_167116

theorem polynomial_perfect_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end polynomial_perfect_square_l1671_167116


namespace sqrt_two_times_sqrt_eight_equals_four_l1671_167151

theorem sqrt_two_times_sqrt_eight_equals_four : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end sqrt_two_times_sqrt_eight_equals_four_l1671_167151


namespace regular_polygon_sides_l1671_167113

/-- A regular polygon with perimeter 150 cm and side length 10 cm has 15 sides -/
theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (num_sides : ℕ) :
  perimeter = 150 ∧ side_length = 10 ∧ perimeter = num_sides * side_length → num_sides = 15 := by
  sorry

end regular_polygon_sides_l1671_167113


namespace max_sum_of_squares_l1671_167117

theorem max_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : 
  x^2 + y^2 ≤ 24421 ∧ ∃ (a b : ℤ), a^2 - b^2 = 221 ∧ a^2 + b^2 = 24421 :=
sorry

end max_sum_of_squares_l1671_167117


namespace choose_three_cooks_from_ten_l1671_167130

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end choose_three_cooks_from_ten_l1671_167130


namespace x_in_P_sufficient_not_necessary_for_x_in_Q_l1671_167143

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1) + Real.sqrt (3 - x)}

-- State the theorem
theorem x_in_P_sufficient_not_necessary_for_x_in_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end x_in_P_sufficient_not_necessary_for_x_in_Q_l1671_167143


namespace distance_traveled_l1671_167140

theorem distance_traveled (initial_reading lunch_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : lunch_reading = 372.0) :
  lunch_reading - initial_reading = 159.7 := by
  sorry

end distance_traveled_l1671_167140


namespace initial_sweets_count_prove_initial_sweets_count_l1671_167172

theorem initial_sweets_count : ℕ → Prop :=
  fun S => 
    (S / 2 + 4 + 7 = S) → 
    (S = 22)

-- Proof
theorem prove_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end initial_sweets_count_prove_initial_sweets_count_l1671_167172


namespace linear_equation_solution_l1671_167159

/-- The linear equation 5x - y = 2 is satisfied by the point (1, 3) -/
theorem linear_equation_solution : 5 * 1 - 3 = 2 := by
  sorry

end linear_equation_solution_l1671_167159


namespace robot_returns_to_start_l1671_167104

/-- Represents a robot's movement pattern -/
structure RobotMovement where
  turn_interval : ℕ  -- Time in seconds between turns
  turn_angle : ℕ     -- Angle of turn in degrees

/-- Represents the state of the robot -/
structure RobotState where
  position : ℤ × ℤ   -- (x, y) coordinates
  direction : ℕ      -- 0: North, 1: East, 2: South, 3: West

/-- Calculates the new position after one movement -/
def move (state : RobotState) : RobotState :=
  match state.direction with
  | 0 => { state with position := (state.position.1, state.position.2 + 1) }
  | 1 => { state with position := (state.position.1 + 1, state.position.2) }
  | 2 => { state with position := (state.position.1, state.position.2 - 1) }
  | 3 => { state with position := (state.position.1 - 1, state.position.2) }
  | _ => state

/-- Calculates the new direction after turning -/
def turn (state : RobotState) : RobotState :=
  { state with direction := (state.direction + 1) % 4 }

/-- Simulates the robot's movement for a given number of seconds -/
def simulate (movement : RobotMovement) (initial_state : RobotState) (time : ℕ) : RobotState :=
  if time = 0 then initial_state
  else
    let new_state := if time % movement.turn_interval = 0 
                     then turn (move initial_state)
                     else move initial_state
    simulate movement new_state (time - 1)

/-- Theorem: The robot returns to its starting point after 6 minutes -/
theorem robot_returns_to_start (movement : RobotMovement) 
  (h1 : movement.turn_interval = 15)
  (h2 : movement.turn_angle = 90) :
  let initial_state : RobotState := ⟨(0, 0), 0⟩
  let final_state := simulate movement initial_state (6 * 60)
  final_state.position = initial_state.position :=
by sorry


end robot_returns_to_start_l1671_167104


namespace inequality_statements_l1671_167160

theorem inequality_statements (a b c : ℝ) :
  (a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end inequality_statements_l1671_167160


namespace zero_subset_A_l1671_167166

def A : Set ℝ := {x | x > -3}

theorem zero_subset_A : {0} ⊆ A := by sorry

end zero_subset_A_l1671_167166


namespace h_ratio_theorem_l1671_167132

/-- Sum of even integers from 2 to n, inclusive, for even n -/
def h (n : ℕ) : ℚ :=
  if n % 2 = 0 then (n / 2) * (n + 2) / 4 else 0

theorem h_ratio_theorem (m k n : ℕ) (h_even : Even n) :
  h (m * n) / h (k * n) = (m : ℚ) / k * (m / k + 1) := by
  sorry

end h_ratio_theorem_l1671_167132


namespace northwest_molded_break_even_price_l1671_167103

/-- Calculate the break-even price per handle for Northwest Molded -/
theorem northwest_molded_break_even_price 
  (variable_cost : ℝ) 
  (fixed_cost : ℝ) 
  (break_even_quantity : ℝ) :
  variable_cost = 0.60 →
  fixed_cost = 7640 →
  break_even_quantity = 1910 →
  (fixed_cost + variable_cost * break_even_quantity) / break_even_quantity = 4.60 :=
by sorry

end northwest_molded_break_even_price_l1671_167103


namespace prob_A_not_lose_l1671_167148

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the property of mutually exclusive events
def mutually_exclusive (p q : ℝ) : Prop := p + q ≤ 1

-- State the theorem
theorem prob_A_not_lose : 
  mutually_exclusive prob_A_win prob_draw →
  prob_A_win + prob_draw = 0.8 :=
by
  sorry

end prob_A_not_lose_l1671_167148


namespace ellipse_range_l1671_167155

-- Define the set of real numbers m for which the equation represents an ellipse
def ellipse_set (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1 ∧ 
  (m + 2 > 0) ∧ (-m - 1 > 0) ∧ (m + 2 ≠ -m - 1)

-- Define the target range for m
def target_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- Theorem statement
theorem ellipse_range :
  ∀ m : ℝ, ellipse_set m ↔ target_range m :=
sorry

end ellipse_range_l1671_167155


namespace april_earnings_l1671_167138

/-- Calculates the total money earned from selling flowers -/
def total_money_earned (rose_price tulip_price daisy_price : ℕ) 
                       (roses_sold tulips_sold daisies_sold : ℕ) : ℕ :=
  rose_price * roses_sold + tulip_price * tulips_sold + daisy_price * daisies_sold

/-- Proves that April earned $78 from selling flowers -/
theorem april_earnings : 
  total_money_earned 4 3 2 9 6 12 = 78 := by sorry

end april_earnings_l1671_167138


namespace marble_count_l1671_167107

theorem marble_count (r g b : ℕ) : 
  g + b = 6 →
  r + b = 8 →
  r + g = 4 →
  r + g + b = 9 := by sorry

end marble_count_l1671_167107


namespace tiles_needed_to_complete_pool_l1671_167141

/-- Given a pool with blue and red tiles, calculate the number of additional tiles needed to complete it. -/
theorem tiles_needed_to_complete_pool 
  (blue_tiles : ℕ) 
  (red_tiles : ℕ) 
  (total_required : ℕ) 
  (h1 : blue_tiles = 48)
  (h2 : red_tiles = 32)
  (h3 : total_required = 100) :
  total_required - (blue_tiles + red_tiles) = 20 := by
  sorry

end tiles_needed_to_complete_pool_l1671_167141


namespace quadratic_roots_l1671_167178

/-- Given a quadratic function f(x) = ax^2 + bx with specific values, 
    prove that the roots of f(x) = 6 are -2 and 3. -/
theorem quadratic_roots (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x)
    (h_m2 : f (-2) = 6)
    (h_m1 : f (-1) = 2)
    (h_0  : f 0 = 0)
    (h_1  : f 1 = 0)
    (h_2  : f 2 = 2)
    (h_3  : f 3 = 6) :
  (∃ x, f x = 6) ∧ (f (-2) = 6 ∧ f 3 = 6) ∧ 
  (∀ x, f x = 6 → x = -2 ∨ x = 3) :=
sorry

end quadratic_roots_l1671_167178


namespace max_phi_difference_l1671_167182

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem max_phi_difference (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  (phi (n^2 + 2*n) - phi (n^2) ≤ 72) ∧
  (∃ m : ℕ, 1 ≤ m ∧ m ≤ 100 ∧ phi (m^2 + 2*m) - phi (m^2) = 72) :=
sorry

end max_phi_difference_l1671_167182


namespace symmetric_difference_A_B_l1671_167167

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x : ℝ | x < 0}

-- Define set difference
def set_difference (M N : Set ℝ) : Set ℝ := {x : ℝ | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  (set_difference M N) ∪ (set_difference N M)

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {x : ℝ | x < -1 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end symmetric_difference_A_B_l1671_167167


namespace inscribed_cube_volume_l1671_167146

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l1671_167146


namespace virgo_boat_trip_duration_l1671_167188

/-- Represents the duration of a trip to Virgo island -/
structure VirgoTrip where
  boat_time : ℝ
  plane_time : ℝ
  total_time : ℝ

/-- Conditions for a valid Virgo trip -/
def is_valid_virgo_trip (trip : VirgoTrip) : Prop :=
  trip.plane_time = 4 * trip.boat_time ∧
  trip.total_time = trip.boat_time + trip.plane_time ∧
  trip.total_time = 10

theorem virgo_boat_trip_duration :
  ∀ (trip : VirgoTrip), is_valid_virgo_trip trip → trip.boat_time = 2 := by
  sorry

end virgo_boat_trip_duration_l1671_167188


namespace reciprocal_of_negative_five_l1671_167142

theorem reciprocal_of_negative_five :
  ∃ x : ℝ, x * (-5) = 1 ∧ x = -(1/5) := by sorry

end reciprocal_of_negative_five_l1671_167142


namespace circle_intersection_range_l1671_167128

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 4) ↔ 
  (-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ 
  (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2) := by
sorry

end circle_intersection_range_l1671_167128


namespace garden_constant_value_l1671_167100

/-- Represents a square garden with area and perimeter --/
structure SquareGarden where
  area : ℝ
  perimeter : ℝ

/-- The constant in the relationship between area and perimeter --/
def garden_constant (g : SquareGarden) : ℝ :=
  g.area - 2 * g.perimeter

theorem garden_constant_value :
  ∀ g : SquareGarden,
    g.area = g.perimeter^2 / 16 →
    g.area = 2 * g.perimeter + garden_constant g →
    g.perimeter = 38 →
    garden_constant g = 14.25 := by
  sorry

end garden_constant_value_l1671_167100


namespace find_k_l1671_167119

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) → k = -18 := by
  sorry

end find_k_l1671_167119


namespace complex_division_result_l1671_167199

theorem complex_division_result : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_division_result_l1671_167199


namespace arithmetic_sequence_fifth_term_l1671_167165

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the 5th term equals 10. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) :
  a 5 = 10 := by
  sorry

end arithmetic_sequence_fifth_term_l1671_167165


namespace average_weight_solution_l1671_167147

def average_weight_problem (a b c : ℝ) : Prop :=
  let avg_abc := (a + b + c) / 3
  let avg_ab := (a + b) / 2
  (avg_abc = 45) ∧ (avg_ab = 40) ∧ (b = 31) → ((b + c) / 2 = 43)

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c :=
by
  sorry

end average_weight_solution_l1671_167147


namespace find_x_l1671_167185

theorem find_x : ∃ x : ℕ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^12 : ℝ) ∧ x = 15 := by
  sorry

end find_x_l1671_167185


namespace cubic_expression_equality_l1671_167153

theorem cubic_expression_equality : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end cubic_expression_equality_l1671_167153


namespace cow_count_is_fifteen_l1671_167150

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The main theorem stating that the number of cows is 15 -/
theorem cow_count_is_fifteen :
  ∃ (ac : AnimalCount), totalLegs ac = 2 * totalHeads ac + 30 ∧ ac.cows = 15 := by
  sorry

#check cow_count_is_fifteen

end cow_count_is_fifteen_l1671_167150


namespace root_zero_implies_m_six_l1671_167109

theorem root_zero_implies_m_six (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x + m - 6 = 0 ∧ x = 0) → m = 6 := by
  sorry

end root_zero_implies_m_six_l1671_167109


namespace decimal_259_to_base5_l1671_167139

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

theorem decimal_259_to_base5 :
  decimal_to_base5 259 = [4, 1, 0, 2] := by sorry

end decimal_259_to_base5_l1671_167139


namespace parallelepiped_volume_exists_parallelepiped_with_volume_144_l1671_167170

/-- Represents the dimensions of a rectangular parallelepiped with a right triangle base -/
structure Parallelepiped where
  a : ℕ
  height : ℕ
  base_is_right_triangle : a^2 + (a+1)^2 = (a+2)^2

/-- The volume of the parallelepiped is 144 -/
theorem parallelepiped_volume (p : Parallelepiped) (h : p.height = 12) : a * (a + 1) * p.height = 144 := by
  sorry

/-- There exists a parallelepiped satisfying the conditions with volume 144 -/
theorem exists_parallelepiped_with_volume_144 : ∃ (p : Parallelepiped), p.height = 12 ∧ a * (a + 1) * p.height = 144 := by
  sorry

end parallelepiped_volume_exists_parallelepiped_with_volume_144_l1671_167170


namespace fraction_multiplication_simplification_l1671_167191

theorem fraction_multiplication_simplification :
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 := by
  sorry

end fraction_multiplication_simplification_l1671_167191


namespace downstream_speed_l1671_167164

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the downstream speed of a man given his upstream and still water speeds -/
theorem downstream_speed (s : RowingSpeed) (h1 : s.upstream = 25) (h2 : s.stillWater = 40) :
  s.downstream = 55 := by
  sorry

#check downstream_speed

end downstream_speed_l1671_167164


namespace complex_modulus_problem_l1671_167101

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l1671_167101


namespace truck_rental_miles_l1671_167196

theorem truck_rental_miles (rental_fee charge_per_mile total_paid : ℚ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 := by
  sorry

end truck_rental_miles_l1671_167196


namespace toys_sold_l1671_167133

/-- Given a selling price, cost price per toy, and a gain equal to the cost of 3 toys,
    prove that the number of toys sold is 18. -/
theorem toys_sold (selling_price : ℕ) (cost_per_toy : ℕ) (h1 : selling_price = 18900) 
    (h2 : cost_per_toy = 900) : 
  (selling_price - 3 * cost_per_toy) / cost_per_toy = 18 := by
  sorry

end toys_sold_l1671_167133


namespace dave_tickets_l1671_167115

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 18 tickets -/
theorem dave_tickets : final_tickets 25 22 15 = 18 := by
  sorry

end dave_tickets_l1671_167115


namespace tom_four_times_cindy_l1671_167126

/-- Tom's current age -/
def t : ℕ := sorry

/-- Cindy's current age -/
def c : ℕ := sorry

/-- In five years, Tom will be twice as old as Cindy -/
axiom future_condition : t + 5 = 2 * (c + 5)

/-- Thirteen years ago, Tom was three times as old as Cindy -/
axiom past_condition : t - 13 = 3 * (c - 13)

/-- The number of years ago when Tom was four times as old as Cindy -/
def years_ago : ℕ := sorry

theorem tom_four_times_cindy : years_ago = 19 := by sorry

end tom_four_times_cindy_l1671_167126


namespace log_stack_sum_l1671_167125

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 4 logs
This theorem proves that the total number of logs in the stack is 114.
-/
theorem log_stack_sum : ∀ (n : ℕ) (a l : ℤ),
  n = 15 - 4 + 1 →
  a = 15 →
  l = 4 →
  n * (a + l) / 2 = 114 := by
  sorry

end log_stack_sum_l1671_167125


namespace remainder_of_7_pow_2023_mod_17_l1671_167137

theorem remainder_of_7_pow_2023_mod_17 : 7^2023 % 17 = 12 := by
  sorry

end remainder_of_7_pow_2023_mod_17_l1671_167137


namespace even_increasing_neg_implies_l1671_167175

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if
    for all x, y ∈ (-∞, 0), x < y implies f(x) < f(y) -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f x < f y

theorem even_increasing_neg_implies (f : ℝ → ℝ)
    (h_even : IsEven f) (h_inc : IncreasingOnNegatives f) :
    f (-1) > f 2 := by
  sorry

end even_increasing_neg_implies_l1671_167175


namespace parabola_point_relation_l1671_167123

-- Define the parabola function
def parabola (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- Define the theorem
theorem parabola_point_relation (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : parabola c (-4) = y₁)
  (h2 : parabola c (-2) = y₂)
  (h3 : parabola c (1/2) = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ :=
sorry

end parabola_point_relation_l1671_167123


namespace complement_A_union_B_l1671_167135

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x : ℝ | Real.log (x - 2) ≤ 0}

-- State the theorem
theorem complement_A_union_B : 
  (Set.compl A) ∪ B = Set.Icc (-1 : ℝ) 3 := by sorry

end complement_A_union_B_l1671_167135


namespace bug_position_after_2015_jumps_l1671_167197

/-- Represents the five points on the circle -/
inductive Point
  | one
  | two
  | three
  | four
  | five

/-- Determines if a point is odd-numbered -/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Performs one jump according to the rules -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.two

/-- Performs n jumps starting from a given point -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (jumpNTimes start n)

theorem bug_position_after_2015_jumps :
  jumpNTimes Point.five 2015 = Point.five :=
sorry

end bug_position_after_2015_jumps_l1671_167197


namespace f_zero_values_l1671_167114

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x y : ℝ, f (x + y) = f x * f y)
variable (h3 : deriv f 0 = 2)

-- Theorem statement
theorem f_zero_values : f 0 = 0 ∨ f 0 = 1 := by
  sorry

end f_zero_values_l1671_167114


namespace five_people_handshakes_l1671_167127

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 5 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 10. -/
theorem five_people_handshakes : number_of_handshakes 5 = 10 := by
  sorry

end five_people_handshakes_l1671_167127


namespace camper_difference_l1671_167174

/-- Represents the number of campers for each week -/
structure CamperCounts where
  threeWeeksAgo : ℕ
  twoWeeksAgo : ℕ
  lastWeek : ℕ

/-- The camping site scenario -/
def campingSite : CamperCounts → Prop
  | ⟨threeWeeksAgo, twoWeeksAgo, lastWeek⟩ =>
    threeWeeksAgo + twoWeeksAgo + lastWeek = 150 ∧
    twoWeeksAgo = 40 ∧
    lastWeek = 80 ∧
    threeWeeksAgo < twoWeeksAgo

theorem camper_difference (c : CamperCounts) (h : campingSite c) :
  c.twoWeeksAgo - c.threeWeeksAgo = 10 := by
  sorry

end camper_difference_l1671_167174


namespace sequence_problem_l1671_167120

theorem sequence_problem (x : ℕ → ℤ) 
  (h1 : x 1 = 8)
  (h2 : x 4 = 2)
  (h3 : ∀ n : ℕ, n > 0 → x (n + 2) + x n = 2 * x (n + 1)) :
  x 10 = -10 := by sorry

end sequence_problem_l1671_167120


namespace fraction_equality_l1671_167194

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_equality_l1671_167194


namespace triangle_determinant_zero_l1671_167179

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end triangle_determinant_zero_l1671_167179


namespace sphere_surface_area_rectangular_prism_l1671_167193

theorem sphere_surface_area_rectangular_prism (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * π * radius^2 = 50 * π := by sorry

end sphere_surface_area_rectangular_prism_l1671_167193


namespace circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l1671_167156

/-- A circle passing through the points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The circle passes through the point (0,0) -/
theorem circle_passes_through_origin :
  circle_equation 0 0 := by sorry

/-- The circle passes through the point (4,0) -/
theorem circle_passes_through_four_zero :
  circle_equation 4 0 := by sorry

/-- The circle passes through the point (-1,1) -/
theorem circle_passes_through_neg_one_one :
  circle_equation (-1) 1 := by sorry

/-- The equation represents a circle -/
theorem is_circle_equation :
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 := by sorry

end circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l1671_167156


namespace homework_time_ratio_l1671_167177

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Represents the ratio of time spent on two subjects -/
structure TimeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of time spent on geography to history -/
def calculateRatio (time : HomeworkTime) : TimeRatio :=
  { numerator := time.geography, denominator := time.history }

theorem homework_time_ratio (time : HomeworkTime) :
  time.biology = 20 →
  time.history = 2 * time.biology →
  time.geography > time.history →
  time.geography > time.biology →
  time.biology + time.history + time.geography = 180 →
  calculateRatio time = { numerator := 3, denominator := 1 } := by
  sorry

#check homework_time_ratio

end homework_time_ratio_l1671_167177


namespace sin_negative_31pi_over_6_l1671_167136

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end sin_negative_31pi_over_6_l1671_167136


namespace student_sums_proof_l1671_167162

def total_sums (right_sums wrong_sums : ℕ) : ℕ :=
  right_sums + wrong_sums

theorem student_sums_proof (right_sums : ℕ) 
  (h1 : right_sums = 18) 
  (h2 : ∃ wrong_sums : ℕ, wrong_sums = 2 * right_sums) : 
  total_sums right_sums (2 * right_sums) = 54 := by
  sorry

end student_sums_proof_l1671_167162


namespace exists_counterexample_l1671_167158

-- Define the types for cards
inductive Letter : Type
| S | T | U

inductive Number : Type
| Two | Five | Seven | Eleven

-- Define a card as a pair of a Letter and a Number
def Card : Type := Letter × Number

-- Define the set of cards
def cards : List Card := [
  (Letter.S, Number.Two),
  (Letter.S, Number.Five),
  (Letter.S, Number.Seven),
  (Letter.S, Number.Eleven),
  (Letter.T, Number.Two),
  (Letter.T, Number.Five),
  (Letter.T, Number.Seven),
  (Letter.T, Number.Eleven),
  (Letter.U, Number.Two),
  (Letter.U, Number.Five),
  (Letter.U, Number.Seven),
  (Letter.U, Number.Eleven)
]

-- Define what a consonant is
def isConsonant (l : Letter) : Bool :=
  match l with
  | Letter.S => true
  | Letter.T => true
  | Letter.U => false

-- Define what a prime number is
def isPrime (n : Number) : Bool :=
  match n with
  | Number.Two => true
  | Number.Five => true
  | Number.Seven => true
  | Number.Eleven => true

-- Sam's statement
def samsStatement (c : Card) : Bool :=
  ¬(isConsonant c.1) ∨ isPrime c.2

-- Theorem to prove
theorem exists_counterexample :
  ∃ c ∈ cards, ¬(samsStatement c) :=
sorry

end exists_counterexample_l1671_167158


namespace minimum_framing_feet_l1671_167180

def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

def enlarged_width : ℕ := original_width * enlargement_factor
def enlarged_height : ℕ := original_height * enlargement_factor

def final_width : ℕ := enlarged_width + 2 * border_width
def final_height : ℕ := enlarged_height + 2 * border_width

def perimeter_inches : ℕ := 2 * (final_width + final_height)

def inches_per_foot : ℕ := 12

theorem minimum_framing_feet :
  (perimeter_inches + inches_per_foot - 1) / inches_per_foot = 10 := by
  sorry

end minimum_framing_feet_l1671_167180


namespace fruit_seller_apples_l1671_167118

/-- Proves that if a fruit seller sells 60% of their apples and has 300 apples left, 
    then they originally had 750 apples. -/
theorem fruit_seller_apples (original : ℕ) (sold_percent : ℚ) (remaining : ℕ) 
    (h1 : sold_percent = 60 / 100)
    (h2 : remaining = 300)
    (h3 : (1 - sold_percent) * original = remaining) : 
  original = 750 := by
  sorry

end fruit_seller_apples_l1671_167118


namespace smallest_multiple_with_remainder_l1671_167145

theorem smallest_multiple_with_remainder : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → 
    (m % 5 = 0 ∧ m % 7 = 0 ∧ m % 3 = 1) → False) ∧ 
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 3 = 1 :=
by
  use 70
  sorry

end smallest_multiple_with_remainder_l1671_167145


namespace coprime_implies_divisible_power_minus_one_l1671_167124

theorem coprime_implies_divisible_power_minus_one (a n : ℕ) (h : Nat.Coprime a n) :
  ∃ m : ℕ, n ∣ (a^m - 1) := by
sorry

end coprime_implies_divisible_power_minus_one_l1671_167124


namespace lcm_1806_1230_l1671_167131

theorem lcm_1806_1230 : Nat.lcm 1806 1230 = 247230 := by
  sorry

end lcm_1806_1230_l1671_167131


namespace boats_in_lake_l1671_167183

theorem boats_in_lake (people_per_boat : ℕ) (total_people : ℕ) (number_of_boats : ℕ) : 
  people_per_boat = 3 → total_people = 15 → number_of_boats * people_per_boat = total_people → 
  number_of_boats = 5 := by
  sorry

end boats_in_lake_l1671_167183


namespace board_cut_lengths_l1671_167168

/-- Given a board of 180 cm cut into three pieces, prove the lengths of the pieces. -/
theorem board_cut_lengths :
  ∀ (L M S : ℝ),
  L + M + S = 180 ∧
  L = M + S + 30 ∧
  M = L / 2 - 10 →
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end board_cut_lengths_l1671_167168


namespace product_inequality_l1671_167102

theorem product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

#check product_inequality

end product_inequality_l1671_167102
