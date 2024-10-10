import Mathlib

namespace ellipse_focus_m_value_l2865_286547

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    if the left focus is at (-4,0), then m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1 → (x + 4)^2 + y^2 = (5 + m)^2) →
  m = 3 :=
by sorry

end ellipse_focus_m_value_l2865_286547


namespace max_perimeter_right_triangle_l2865_286544

theorem max_perimeter_right_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 5) :
  a + b + c ≤ 5 + 5 * Real.sqrt 2 := by
sorry

end max_perimeter_right_triangle_l2865_286544


namespace interest_rate_calculation_l2865_286570

theorem interest_rate_calculation (initial_amount : ℝ) (final_amount : ℝ) 
  (second_year_rate : ℝ) (first_year_rate : ℝ) : 
  initial_amount = 6000 ∧ 
  final_amount = 6552 ∧ 
  second_year_rate = 0.05 ∧
  first_year_rate = 0.04 →
  final_amount = initial_amount + 
    (initial_amount * first_year_rate) + 
    ((initial_amount + initial_amount * first_year_rate) * second_year_rate) :=
by sorry

end interest_rate_calculation_l2865_286570


namespace sum_of_ABC_values_l2865_286564

/-- A function that represents the number A5B79C given digits A, B, and C -/
def number (A B C : ℕ) : ℕ := A * 100000 + 5 * 10000 + B * 1000 + 7 * 100 + 9 * 10 + C

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n ≤ 9

/-- The sum of all possible values of A+B+C given the conditions -/
def sum_of_possible_values : ℕ := 29

/-- The main theorem -/
theorem sum_of_ABC_values (A B C : ℕ) 
  (hA : is_single_digit A) (hB : is_single_digit B) (hC : is_single_digit C)
  (h_div : (number A B C) % 11 = 0) : 
  sum_of_possible_values = 29 := by
  sorry

end sum_of_ABC_values_l2865_286564


namespace distribute_5_3_l2865_286572

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 21 := by
  sorry

end distribute_5_3_l2865_286572


namespace sweet_potato_problem_l2865_286522

-- Define the problem parameters
def total_harvested : ℕ := 80
def sold_to_adams : ℕ := 20
def sold_to_lenon : ℕ := 15
def traded_for_pumpkins : ℕ := 10
def pumpkins_received : ℕ := 5
def pumpkin_weight : ℕ := 3
def donation_percentage : Rat := 5 / 100

-- Define the theorem
theorem sweet_potato_problem :
  let remaining_before_donation := total_harvested - (sold_to_adams + sold_to_lenon + traded_for_pumpkins)
  let donation := (remaining_before_donation : Rat) * donation_percentage
  let remaining_after_donation := remaining_before_donation - ⌈donation⌉
  remaining_after_donation = 33 ∧ pumpkins_received * pumpkin_weight = 15 := by
  sorry


end sweet_potato_problem_l2865_286522


namespace f_minimum_properties_l2865_286506

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem f_minimum_properties {x_0 : ℝ} (h_pos : x_0 > 0) 
  (h_min : ∀ x > 0, f x ≥ f x_0) : 
  f x_0 = x_0 + 1 ∧ f x_0 < 3 := by
  sorry

end f_minimum_properties_l2865_286506


namespace circle_center_coordinate_sum_l2865_286537

theorem circle_center_coordinate_sum :
  ∀ (x y h k : ℝ),
  (∀ (x' y' : ℝ), x'^2 + y'^2 = 4*x' - 6*y' + 9 ↔ (x' - h)^2 + (y' - k)^2 = (h^2 + k^2 - 9 + 4*h - 6*k)) →
  h + k = -1 :=
by sorry

end circle_center_coordinate_sum_l2865_286537


namespace ruby_pizza_tip_l2865_286535

/-- Represents the pizza order scenario --/
structure PizzaOrder where
  base_price : ℕ        -- Price of a pizza without toppings
  topping_price : ℕ     -- Price of each topping
  num_pizzas : ℕ        -- Number of pizzas ordered
  num_toppings : ℕ      -- Total number of toppings
  total_with_tip : ℕ    -- Total cost including tip

/-- Calculates the tip amount for a given pizza order --/
def calculate_tip (order : PizzaOrder) : ℕ :=
  order.total_with_tip - (order.base_price * order.num_pizzas + order.topping_price * order.num_toppings)

/-- Theorem stating that the tip for Ruby's pizza order is $5 --/
theorem ruby_pizza_tip :
  let order : PizzaOrder := {
    base_price := 10,
    topping_price := 1,
    num_pizzas := 3,
    num_toppings := 4,
    total_with_tip := 39
  }
  calculate_tip order = 5 := by
  sorry


end ruby_pizza_tip_l2865_286535


namespace stratified_sampling_senior_high_l2865_286596

theorem stratified_sampling_senior_high (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1800)
  (h2 : senior_students = 600)
  (h3 : sample_size = 180) :
  (senior_students * sample_size) / total_students = 60 := by
  sorry

end stratified_sampling_senior_high_l2865_286596


namespace repeating_decimal_interval_l2865_286586

def is_repeating_decimal_of_period (n : ℕ) (p : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ n ∣ (10^p - 1) ∧ ∀ (q : ℕ), q < p → ¬(n ∣ (10^q - 1))

theorem repeating_decimal_interval :
  ∀ (n : ℕ),
    n > 0 →
    n < 2000 →
    is_repeating_decimal_of_period n 4 →
    is_repeating_decimal_of_period (n + 4) 6 →
    801 ≤ n ∧ n ≤ 1200 :=
by sorry

end repeating_decimal_interval_l2865_286586


namespace problem_statement_l2865_286575

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 44 := by
  sorry

end problem_statement_l2865_286575


namespace curve_C_equation_l2865_286582

/-- The equation of curve C -/
def curve_C (a : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop :=
  y = -2 * x + 4

/-- M and N are distinct intersection points of curve C and line l -/
def intersection_points (a : ℝ) (M N : ℝ × ℝ) : Prop :=
  M ≠ N ∧ curve_C a M.1 M.2 ∧ curve_C a N.1 N.2 ∧ line_l M.1 M.2 ∧ line_l N.1 N.2

/-- The distance from origin O to M is equal to the distance from O to N -/
def equal_distances (M N : ℝ × ℝ) : Prop :=
  M.1^2 + M.2^2 = N.1^2 + N.2^2

theorem curve_C_equation (a : ℝ) (M N : ℝ × ℝ) :
  a ≠ 0 →
  intersection_points a M N →
  equal_distances M N →
  ∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y = 0 :=
by sorry

end curve_C_equation_l2865_286582


namespace lion_path_theorem_l2865_286541

/-- A broken line path within a circle -/
structure BrokenLinePath where
  points : List (Real × Real)
  inside_circle : ∀ p ∈ points, p.1^2 + p.2^2 ≤ 100

/-- The total length of a broken line path -/
def pathLength (path : BrokenLinePath) : Real :=
  sorry

/-- The sum of turning angles in a broken line path -/
def sumTurningAngles (path : BrokenLinePath) : Real :=
  sorry

/-- Main theorem: If a broken line path within a circle of radius 10 meters
    has a total length of 30,000 meters, then the sum of all turning angles
    along the path is at least 2998 radians -/
theorem lion_path_theorem (path : BrokenLinePath) 
    (h : pathLength path = 30000) :
  sumTurningAngles path ≥ 2998 := by
  sorry

end lion_path_theorem_l2865_286541


namespace polynomial_roots_l2865_286510

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem polynomial_roots : 
  ∃ (a b c d e : ℝ), 
    (a = -1 - Real.sqrt 3) ∧
    (b = -1 + Real.sqrt 3) ∧
    (c = -1) ∧
    (d = 1) ∧
    (e = 2) ∧
    (∀ x : ℝ, polynomial x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) := by
  sorry

end polynomial_roots_l2865_286510


namespace brain_info_scientific_notation_l2865_286557

/-- The number of pieces of information the human brain can record per day -/
def brain_info_capacity : ℕ := 86000000

/-- Scientific notation representation of brain_info_capacity -/
def brain_info_scientific : ℝ := 8.6 * (10 ^ 7)

theorem brain_info_scientific_notation :
  (brain_info_capacity : ℝ) = brain_info_scientific := by
  sorry

end brain_info_scientific_notation_l2865_286557


namespace product_inequality_l2865_286546

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end product_inequality_l2865_286546


namespace largest_x_sqrt_3x_eq_5x_l2865_286542

theorem largest_x_sqrt_3x_eq_5x : 
  ∃ (x_max : ℚ), x_max = 3/25 ∧ 
  (∀ x : ℚ, x ≥ 0 → (Real.sqrt (3 * x) = 5 * x) → x ≤ x_max) ∧
  (Real.sqrt (3 * x_max) = 5 * x_max) := by
  sorry

end largest_x_sqrt_3x_eq_5x_l2865_286542


namespace dice_probability_l2865_286579

def red_die := Finset.range 6
def blue_die := Finset.range 6

def event_M (x : ℕ) : Prop := x % 3 = 0 ∧ x ∈ red_die
def event_N (x y : ℕ) : Prop := x + y > 8 ∧ x ∈ red_die ∧ y ∈ blue_die

def P_MN : ℚ := 5 / 36
def P_M : ℚ := 1 / 3

theorem dice_probability : (P_MN / P_M) = 5 / 12 := by sorry

end dice_probability_l2865_286579


namespace unique_number_with_three_prime_divisors_l2865_286576

theorem unique_number_with_three_prime_divisors (x : ℕ) (n : ℕ) :
  Odd n →
  x = 6^n + 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    x = 11 * p * q ∧
    ∀ r : ℕ, Prime r → r ∣ x → (r = 11 ∨ r = p ∨ r = q)) →
  x = 7777 := by
sorry

end unique_number_with_three_prime_divisors_l2865_286576


namespace sector_radius_l2865_286569

theorem sector_radius (θ : ℝ) (L : ℝ) (R : ℝ) :
  θ = 60 → L = π → L = (θ * π * R) / 180 → R = 3 :=
by sorry

end sector_radius_l2865_286569


namespace slope_angle_of_line_l2865_286540

/-- The slope angle of a line passing through points (0,√3) and (2,3√3) is π/3 -/
theorem slope_angle_of_line (A B : ℝ × ℝ) : 
  A = (0, Real.sqrt 3) → 
  B = (2, 3 * Real.sqrt 3) → 
  let slope := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan slope = π / 3 := by sorry

end slope_angle_of_line_l2865_286540


namespace twenty_political_science_majors_l2865_286598

/-- The number of applicants who majored in political science -/
def political_science_majors (total : ℕ) (high_gpa : ℕ) (not_ps_low_gpa : ℕ) (ps_high_gpa : ℕ) : ℕ :=
  total - not_ps_low_gpa - (high_gpa - ps_high_gpa)

/-- Theorem stating that 20 applicants majored in political science -/
theorem twenty_political_science_majors :
  political_science_majors 40 20 10 5 = 20 := by
  sorry

#eval political_science_majors 40 20 10 5

end twenty_political_science_majors_l2865_286598


namespace nonagon_trapezium_existence_l2865_286527

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex : Type
  | A | B | C | D | E | F | G | H | I

/-- A function to determine if four vertices form a trapezium -/
def is_trapezium (v1 v2 v3 v4 : Vertex) : Prop :=
  sorry -- The actual implementation would depend on the geometry of the nonagon

/-- Main theorem: Given any five vertices of a regular nonagon, 
    there always exists a subset of four vertices among them that form a trapezium -/
theorem nonagon_trapezium_existence 
  (chosen : Finset Vertex) 
  (h : chosen.card = 5) : 
  ∃ (v1 v2 v3 v4 : Vertex), v1 ∈ chosen ∧ v2 ∈ chosen ∧ v3 ∈ chosen ∧ v4 ∈ chosen ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    is_trapezium v1 v2 v3 v4 :=
  sorry


end nonagon_trapezium_existence_l2865_286527


namespace distance_to_nearest_town_l2865_286562

theorem distance_to_nearest_town (d : ℝ) 
  (h1 : ¬(d ≥ 8))  -- Alice's statement is false
  (h2 : ¬(d ≤ 7))  -- Bob's statement is false
  (h3 : d ≠ 5)     -- Charlie's statement is false
  : 7 < d ∧ d < 8 := by
  sorry

end distance_to_nearest_town_l2865_286562


namespace arithmetic_sequence_problem_l2865_286520

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequenceWithSum where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequenceWithSum) 
  (h1 : seq.a 8 - seq.a 5 = 9)
  (h2 : seq.S 8 - seq.S 5 = 66) :
  seq.a 33 = 100 := by
  sorry


end arithmetic_sequence_problem_l2865_286520


namespace complex_fraction_equality_l2865_286545

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I * 2) = -2 + Complex.I := by sorry

end complex_fraction_equality_l2865_286545


namespace impossible_to_face_up_all_coins_l2865_286531

/-- Represents the state of all coins -/
def CoinState := List Bool

/-- Represents a flip operation on 6 coins -/
def Flip := List Nat

/-- The initial state of the coins -/
def initialState : CoinState := 
  (List.replicate 1000 true) ++ (List.replicate 997 false)

/-- Applies a flip to a coin state -/
def applyFlip (state : CoinState) (flip : Flip) : CoinState :=
  sorry

/-- Checks if all coins are facing up -/
def allFacingUp (state : CoinState) : Bool :=
  state.all id

/-- Theorem stating that it's impossible to make all coins face up -/
theorem impossible_to_face_up_all_coins :
  ∀ (flips : List Flip), 
    ¬(allFacingUp (flips.foldl applyFlip initialState)) :=
by
  sorry

end impossible_to_face_up_all_coins_l2865_286531


namespace parabola_axis_symmetry_l2865_286538

/-- 
Given a parabola defined by y = a * x^2 with axis of symmetry y = -2,
prove that a = 1/8.
-/
theorem parabola_axis_symmetry (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 
  (∀ x : ℝ, -2 = a * x^2) → 
  a = 1/8 := by sorry

end parabola_axis_symmetry_l2865_286538


namespace mario_moving_sidewalk_time_l2865_286501

/-- The time it takes Mario to walk from A to B on a moving sidewalk -/
theorem mario_moving_sidewalk_time (d : ℝ) (w : ℝ) (v : ℝ) : 
  d > 0 ∧ w > 0 ∧ v > 0 →  -- distances and speeds are positive
  d / w = 90 →             -- time to walk when sidewalk is off
  d / v = 45 →             -- time to be carried without walking
  d / (w + v) = 30 :=      -- time to walk on moving sidewalk
by sorry

end mario_moving_sidewalk_time_l2865_286501


namespace remainder_problem_l2865_286553

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end remainder_problem_l2865_286553


namespace jokes_count_l2865_286566

/-- The total number of jokes told by Jessy and Alan over two Saturdays -/
def total_jokes (jessy_first : ℕ) (alan_first : ℕ) : ℕ :=
  let first_saturday := jessy_first + alan_first
  let second_saturday := 2 * jessy_first + 2 * alan_first
  first_saturday + second_saturday

/-- Theorem stating the total number of jokes told by Jessy and Alan -/
theorem jokes_count : total_jokes 11 7 = 54 := by
  sorry

end jokes_count_l2865_286566


namespace club_members_count_l2865_286513

def sock_cost : ℝ := 6
def tshirt_cost : ℝ := sock_cost + 7
def cap_cost : ℝ := tshirt_cost - 3
def total_cost_per_member : ℝ := 2 * (sock_cost + tshirt_cost + cap_cost)
def total_club_cost : ℝ := 3630

theorem club_members_count : 
  ∃ n : ℕ, n = 63 ∧ (n : ℝ) * total_cost_per_member = total_club_cost :=
sorry

end club_members_count_l2865_286513


namespace identical_asymptotes_hyperbolas_l2865_286593

theorem identical_asymptotes_hyperbolas (M : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) → M = 225/16 := by
  sorry

end identical_asymptotes_hyperbolas_l2865_286593


namespace universiade_volunteer_count_l2865_286584

/-- Represents the result of a stratified sampling by gender -/
structure StratifiedSample where
  total_pool : ℕ
  selected_group : ℕ
  selected_male : ℕ
  selected_female : ℕ

/-- Calculates the number of female students in the pool based on stratified sampling -/
def femaleInPool (sample : StratifiedSample) : ℕ :=
  (sample.selected_female * sample.total_pool) / sample.selected_group

theorem universiade_volunteer_count :
  ∀ (sample : StratifiedSample),
    sample.total_pool = 200 →
    sample.selected_group = 30 →
    sample.selected_male = 12 →
    sample.selected_female = sample.selected_group - sample.selected_male →
    femaleInPool sample = 120 := by
  sorry

#eval femaleInPool { total_pool := 200, selected_group := 30, selected_male := 12, selected_female := 18 }

end universiade_volunteer_count_l2865_286584


namespace total_bathing_suits_l2865_286515

def one_piece : ℕ := 8500
def two_piece : ℕ := 12750
def trunks : ℕ := 5900
def shorts : ℕ := 7250
def children : ℕ := 1100

theorem total_bathing_suits :
  one_piece + two_piece + trunks + shorts + children = 35500 := by
  sorry

end total_bathing_suits_l2865_286515


namespace consecutive_integers_average_l2865_286571

theorem consecutive_integers_average (highest : ℕ) (h : highest = 36) :
  let set := List.range 7
  let numbers := set.map (λ i => highest - (6 - i))
  (numbers.sum : ℚ) / 7 = 33 := by
  sorry

end consecutive_integers_average_l2865_286571


namespace chicken_cost_is_40_cents_l2865_286525

/-- The cost of chicken per plate given the total number of plates, 
    cost of rice per plate, and total spent on food. -/
def chicken_cost_per_plate (total_plates : ℕ) (rice_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (total_plates : ℚ) * rice_cost_per_plate) / (total_plates : ℚ)

/-- Theorem stating that the cost of chicken per plate is $0.40 
    given the specific conditions of the problem. -/
theorem chicken_cost_is_40_cents :
  chicken_cost_per_plate 100 (1/10) 50 = 2/5 := by
  sorry

end chicken_cost_is_40_cents_l2865_286525


namespace triple_base_and_exponent_l2865_286585

theorem triple_base_and_exponent (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * y ^ b → y = 27 * a ^ 2 := by
  sorry

end triple_base_and_exponent_l2865_286585


namespace disjunction_false_implies_negation_true_l2865_286502

variable (p q : Prop)

theorem disjunction_false_implies_negation_true :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end disjunction_false_implies_negation_true_l2865_286502


namespace round_trip_time_l2865_286588

/-- Calculates the total time for a round trip on a river given the rower's speed, river speed, and distance. -/
theorem round_trip_time (rower_speed river_speed distance : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : distance = 2.67) : 
  (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1.00125 := by
  sorry

#eval (2.67 / (6 - 2)) + (2.67 / (6 + 2))

end round_trip_time_l2865_286588


namespace no_house_spirits_l2865_286590

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseSpirit : U → Prop)
variable (LovesMischief : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem no_house_spirits
  (h1 : ∀ x, HouseSpirit x → LovesMischief x)
  (h2 : ∀ x, HouseSpirit x → LovesCleanlinessAndOrder x)
  (h3 : ∀ x, LovesCleanlinessAndOrder x → ¬LovesMischief x) :
  ¬∃ x, HouseSpirit x :=
by sorry

end no_house_spirits_l2865_286590


namespace six_b_equals_twenty_l2865_286521

theorem six_b_equals_twenty (a b : ℚ) 
  (h1 : 10 * a = b) 
  (h2 : b = 20) 
  (h3 : 120 * a * b = 800) : 
  6 * b = 20 := by
sorry

end six_b_equals_twenty_l2865_286521


namespace binomial_expansion_coefficient_l2865_286561

/-- Given that the coefficients of the first three terms in the expansion of (x + 1/(2x))^n form an arithmetic sequence,
    prove that the coefficient of the x^4 term in the expansion is 7. -/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ d : ℚ, (1 : ℚ) = (n.choose 0 : ℚ) ∧ 
             (1/2 : ℚ) * (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ 
             (1/4 : ℚ) * (n.choose 2 : ℚ) = (n.choose 0 : ℚ) + 2*d) → 
  (1/4 : ℚ) * (n.choose 4 : ℚ) = 7 := by
sorry

end binomial_expansion_coefficient_l2865_286561


namespace exact_location_determination_l2865_286507

-- Define a type for location descriptors
inductive LocationDescriptor
  | CinemaLocation (row : ℕ) (hall : ℕ) (cinema : String)
  | Direction (angle : ℝ)
  | StreetSection (street : String)
  | Coordinates (longitude : ℝ) (latitude : ℝ)

-- Define a function to check if a location descriptor can determine an exact location
def canDetermineExactLocation (descriptor : LocationDescriptor) : Prop :=
  match descriptor with
  | LocationDescriptor.Coordinates _ _ => True
  | _ => False

-- Theorem statement
theorem exact_location_determination
  (cinema_loc : LocationDescriptor)
  (direction : LocationDescriptor)
  (street_section : LocationDescriptor)
  (coordinates : LocationDescriptor)
  (h1 : cinema_loc = LocationDescriptor.CinemaLocation 2 3 "Pacific Cinema")
  (h2 : direction = LocationDescriptor.Direction 40)
  (h3 : street_section = LocationDescriptor.StreetSection "Middle section of Tianfu Avenue")
  (h4 : coordinates = LocationDescriptor.Coordinates 116 42) :
  canDetermineExactLocation coordinates ∧
  ¬canDetermineExactLocation cinema_loc ∧
  ¬canDetermineExactLocation direction ∧
  ¬canDetermineExactLocation street_section :=
sorry

end exact_location_determination_l2865_286507


namespace unique_prime_for_equal_sets_l2865_286559

theorem unique_prime_for_equal_sets (p : Nat) (g : Nat) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  (∀ a : Nat, 1 ≤ a → a < p → g^a % p ≠ 1) → 
  g^(p-1) % p = 1 → 
  (∀ k : Nat, 1 ≤ k → k ≤ (p-1)/2 → ∃ m : Nat, 1 ≤ m ∧ m ≤ (p-1)/2 ∧ (k^2 + 1) % p = g^m % p) → 
  (∀ m : Nat, 1 ≤ m → m ≤ (p-1)/2 → ∃ k : Nat, 1 ≤ k ∧ k ≤ (p-1)/2 ∧ g^m % p = (k^2 + 1) % p) → 
  p = 3 := by
sorry

end unique_prime_for_equal_sets_l2865_286559


namespace range_of_m_l2865_286567

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_but_not_sufficient m ↔ (m ≥ 1 ∨ m ≤ -7) :=
sorry

end range_of_m_l2865_286567


namespace difference_of_squares_l2865_286558

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l2865_286558


namespace solution_equivalence_l2865_286595

/-- Given prime numbers p and q with p < q, the positive integer solutions (x, y) to 
    1/x + 1/y = 1/p - 1/q are equivalent to the positive integer solutions of 
    ((q - p)x - pq)((q - p)y - pq) = p^2q^2 -/
theorem solution_equivalence (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p < q) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 / x + 1 / y = 1 / p - 1 / q) ↔ 
  ((q - p) * x - p * q) * ((q - p) * y - p * q) = p^2 * q^2 :=
by sorry

end solution_equivalence_l2865_286595


namespace smaller_prime_l2865_286543

theorem smaller_prime (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y)
  (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x < y := by
  sorry

end smaller_prime_l2865_286543


namespace hyperbola_asymptote_a_value_l2865_286504

theorem hyperbola_asymptote_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ((3 * x + 2 * y = 0) ∨ (3 * x - 2 * y = 0))) →
  a = 2 := by sorry

end hyperbola_asymptote_a_value_l2865_286504


namespace dodecagon_area_times_hundred_l2865_286555

/-- The area of a regular dodecagon inscribed in a unit circle -/
def dodecagonArea : ℝ := 3

/-- 100 times the area of a regular dodecagon inscribed in a unit circle -/
def hundredTimesDodecagonArea : ℝ := 100 * dodecagonArea

theorem dodecagon_area_times_hundred : hundredTimesDodecagonArea = 300 := by
  sorry

end dodecagon_area_times_hundred_l2865_286555


namespace cos_pi_third_minus_alpha_l2865_286514

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end cos_pi_third_minus_alpha_l2865_286514


namespace signup_ways_eq_81_l2865_286591

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students signing up --/
def num_students : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def signup_ways : ℕ := num_groups ^ num_students

/-- Theorem: The number of ways four students can sign up for one of three interest groups is 81 --/
theorem signup_ways_eq_81 : signup_ways = 81 := by
  sorry

end signup_ways_eq_81_l2865_286591


namespace rectangular_box_surface_area_l2865_286526

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : a + b + c = 35) 
  (diagonal : a^2 + b^2 + c^2 = 21^2) : 
  2 * (a*b + b*c + c*a) = 784 := by
  sorry

end rectangular_box_surface_area_l2865_286526


namespace teacher_grading_problem_l2865_286528

/-- Calculates the number of problems left to grade given the total number of worksheets,
    the number of graded worksheets, and the number of problems per worksheet. -/
def problems_left_to_grade (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_per_worksheet : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

/-- Proves that given 9 total worksheets, 5 graded worksheets, and 4 problems per worksheet,
    there are 16 problems left to grade. -/
theorem teacher_grading_problem :
  problems_left_to_grade 9 5 4 = 16 := by
  sorry

end teacher_grading_problem_l2865_286528


namespace perpendicular_vectors_l2865_286563

/-- Given vectors a and b in ℝ², and c = a + k*b, prove that if a ⊥ c, then k = -10/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (3, 1))
  (h2 : b = (1, 0))
  (h3 : c = a + k • b)
  (h4 : a.1 * c.1 + a.2 * c.2 = 0) : 
  k = -10/3 := by
  sorry

end perpendicular_vectors_l2865_286563


namespace select_three_from_eight_l2865_286532

theorem select_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end select_three_from_eight_l2865_286532


namespace unique_valid_number_l2865_286517

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  n = (n / 10)^3 + (n % 10)^3 - 3

theorem unique_valid_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 32 := by sorry

end unique_valid_number_l2865_286517


namespace solution_set_inequality_l2865_286550

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end solution_set_inequality_l2865_286550


namespace ticket_price_reduction_l2865_286581

theorem ticket_price_reduction 
  (original_price : ℚ)
  (sold_increase_ratio : ℚ)
  (revenue_increase_ratio : ℚ)
  (price_reduction : ℚ) :
  original_price = 50 →
  sold_increase_ratio = 1/3 →
  revenue_increase_ratio = 1/4 →
  (original_price - price_reduction) * (1 + sold_increase_ratio) = original_price * (1 + revenue_increase_ratio) →
  price_reduction = 25/2 := by
sorry

end ticket_price_reduction_l2865_286581


namespace expression_evaluation_l2865_286530

/-- Proves that the given expression evaluates to the specified value -/
theorem expression_evaluation (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  3500 - (1000 / (20.50 + x * 10)) / (y^2 - 2*z) = 3496.6996699669967 := by
  sorry

#eval (3500 - (1000 / (20.50 + 3 * 10)) / (4^2 - 2*5) : Float)

end expression_evaluation_l2865_286530


namespace infinite_geometric_series_first_term_l2865_286580

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = (1/4 : ℝ)) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by
sorry

end infinite_geometric_series_first_term_l2865_286580


namespace product_difference_sum_l2865_286578

theorem product_difference_sum : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    a * b = 50 ∧ 
    (max a b - min a b) = 5 → 
    a + b = 15 := by
  sorry

end product_difference_sum_l2865_286578


namespace lunch_break_duration_l2865_286574

-- Define the painting rates and lunch break
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the workday durations
def monday_duration : ℝ := 9
def tuesday_duration : ℝ := 7
def wednesday_duration : ℝ := 12

-- Define the portions painted each day
def monday_portion : ℝ := 0.6
def tuesday_portion : ℝ := 0.3
def wednesday_portion : ℝ := 0.1

-- Theorem statement
theorem lunch_break_duration (d : PaintingData) : 
  (monday_duration - d.lunch_break) * (d.paula_rate + d.helpers_rate) = monday_portion ∧
  (tuesday_duration - d.lunch_break) * d.helpers_rate = tuesday_portion ∧
  (wednesday_duration - d.lunch_break) * d.paula_rate = wednesday_portion →
  d.lunch_break = 1 := by
  sorry

#check lunch_break_duration

end lunch_break_duration_l2865_286574


namespace prob_grad_degree_is_three_nineteenths_l2865_286516

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (antecedent : ℕ)
  (consequent : ℕ)

/-- Represents the company's employee composition -/
structure Company :=
  (grad_ratio : Ratio)     -- Ratio of graduates with graduate degree to non-graduates
  (nongrad_ratio : Ratio)  -- Ratio of graduates without graduate degree to non-graduates

/-- Calculates the probability of a randomly picked college graduate having a graduate degree -/
def probability_grad_degree (c : Company) : ℚ :=
  let lcm := Nat.lcm c.grad_ratio.consequent c.nongrad_ratio.consequent
  let grad_scaled := c.grad_ratio.antecedent * (lcm / c.grad_ratio.consequent)
  let nongrad_scaled := c.nongrad_ratio.antecedent * (lcm / c.nongrad_ratio.consequent)
  grad_scaled / (grad_scaled + nongrad_scaled)

/-- The main theorem to be proved -/
theorem prob_grad_degree_is_three_nineteenths :
  ∀ c : Company,
    c.grad_ratio = ⟨1, 8⟩ →
    c.nongrad_ratio = ⟨2, 3⟩ →
    probability_grad_degree c = 3 / 19 :=
by
  sorry


end prob_grad_degree_is_three_nineteenths_l2865_286516


namespace lucy_fish_count_l2865_286503

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end lucy_fish_count_l2865_286503


namespace percentage_problem_l2865_286548

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 5600) = 126) → P = 0.3 := by
  sorry

end percentage_problem_l2865_286548


namespace student_calculation_correct_result_problem_statement_l2865_286556

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem student_calculation (c d : ℕ) : ℚ :=
  74 * (1 + (c : ℚ) / 10 + (d : ℚ) / 100) + 3

theorem correct_result (c d : ℕ) : ℚ :=
  74 * repeating_decimal c d + 3

theorem problem_statement (c d : ℕ) : 
  correct_result c d - student_calculation c d = 1.2 → c = 1 ∧ d = 6 :=
sorry

end student_calculation_correct_result_problem_statement_l2865_286556


namespace fraction_to_decimal_l2865_286533

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l2865_286533


namespace yuna_survey_l2865_286524

theorem yuna_survey (math_lovers : ℕ) (korean_lovers : ℕ) (both_lovers : ℕ)
  (h1 : math_lovers = 27)
  (h2 : korean_lovers = 28)
  (h3 : both_lovers = 22) :
  math_lovers + korean_lovers - both_lovers = 33 := by
  sorry

end yuna_survey_l2865_286524


namespace valid_speaking_orders_l2865_286568

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

theorem valid_speaking_orders : 
  (choose special_students 1 * choose (total_students - special_students) (selected_students - 1) * arrange selected_students selected_students) +
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange selected_students selected_students) -
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange (selected_students - 1) (selected_students - 1) * arrange 2 2) = 600 := by
  sorry

end valid_speaking_orders_l2865_286568


namespace distance_between_points_l2865_286565

/-- The distance between two points (5, -3) and (9, 6) in a 2D plane is √97 units. -/
theorem distance_between_points : Real.sqrt 97 = Real.sqrt ((9 - 5)^2 + (6 - (-3))^2) := by
  sorry

end distance_between_points_l2865_286565


namespace basketball_chess_fans_l2865_286500

/-- The number of students who like basketball or chess given the following conditions:
  * 40% of students like basketball
  * 10% of students like chess
  * 250 students were interviewed
-/
theorem basketball_chess_fans (total_students : ℕ) (basketball_percent : ℚ) (chess_percent : ℚ) :
  total_students = 250 →
  basketball_percent = 40 / 100 →
  chess_percent = 10 / 100 →
  (basketball_percent + chess_percent) * total_students = 125 := by
sorry

end basketball_chess_fans_l2865_286500


namespace triplet_sum_not_seven_l2865_286599

theorem triplet_sum_not_seven : 
  let triplet_A := (3/2, 4/3, 13/6)
  let triplet_B := (4, -3, 6)
  let triplet_C := (2.5, 3.1, 1.4)
  let triplet_D := (7.4, -9.4, 9.0)
  let triplet_E := (-3/4, -9/4, 8)
  
  let sum_A := triplet_A.1 + triplet_A.2.1 + triplet_A.2.2
  let sum_B := triplet_B.1 + triplet_B.2.1 + triplet_B.2.2
  let sum_C := triplet_C.1 + triplet_C.2.1 + triplet_C.2.2
  let sum_D := triplet_D.1 + triplet_D.2.1 + triplet_D.2.2
  let sum_E := triplet_E.1 + triplet_E.2.1 + triplet_E.2.2
  
  (sum_A ≠ 7 ∧ sum_E ≠ 7) ∧ (sum_B = 7 ∧ sum_C = 7 ∧ sum_D = 7) := by
  sorry

end triplet_sum_not_seven_l2865_286599


namespace harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l2865_286509

/-- Definition of a T sequence -/
def is_T_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n

/-- The sequence A_n(n, 1/n) is a T sequence -/
theorem harmonic_is_T_sequence :
  is_T_sequence (fun n ↦ 1 / (n : ℝ)) := by sorry

/-- Property of T sequences for certain index relationships -/
theorem T_sequence_property (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a p ≥ (q - p : ℝ) * (a (p + 1) - a p) := by sorry

/-- Another property of T sequences for certain index relationships -/
theorem T_sequence_property_2 (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a n > a p - a m := by sorry

end harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l2865_286509


namespace max_red_balls_l2865_286573

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Yellow
  | Green

/-- The number marked on a ball of a given color -/
def ballNumber (c : BallColor) : Nat :=
  match c with
  | BallColor.Red => 4
  | BallColor.Yellow => 5
  | BallColor.Green => 6

/-- The total number of balls drawn -/
def totalBalls : Nat := 8

/-- The sum of numbers on all drawn balls -/
def totalSum : Nat := 39

/-- A configuration of drawn balls -/
structure BallConfiguration where
  red : Nat
  yellow : Nat
  green : Nat
  sum_eq : red + yellow + green = totalBalls
  number_sum_eq : red * ballNumber BallColor.Red + 
                  yellow * ballNumber BallColor.Yellow + 
                  green * ballNumber BallColor.Green = totalSum

/-- The maximum number of red balls in any valid configuration is 4 -/
theorem max_red_balls : 
  ∀ (config : BallConfiguration), config.red ≤ 4 :=
sorry

end max_red_balls_l2865_286573


namespace monomial_count_is_four_l2865_286508

/-- A monomial is an algebraic expression consisting of one term. It can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (expr : String) : Bool := sorry

/-- The list of algebraic expressions given in the problem -/
def expressions : List String := ["-2/3*a^3*b", "xy/2", "-4", "-2/a", "0", "x-y"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  exprs.filter is_monomial |>.length

/-- The theorem to be proved -/
theorem monomial_count_is_four : count_monomials expressions = 4 := by sorry

end monomial_count_is_four_l2865_286508


namespace theater_ticket_difference_l2865_286505

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end theater_ticket_difference_l2865_286505


namespace inequality_chain_l2865_286554

theorem inequality_chain (b a x : ℝ) (h1 : b > a) (h2 : a > x) (h3 : x > 0) :
  x^2 < x*a ∧ x*a < a^2 ∧ a^2 < x*b := by
  sorry

end inequality_chain_l2865_286554


namespace decimal_difference_l2865_286511

-- Define the repeating decimal 0.2̅4̅
def repeating_decimal : ℚ := 8 / 33

-- Define the terminating decimal 0.24
def terminating_decimal : ℚ := 24 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 825 := by
  sorry

end decimal_difference_l2865_286511


namespace repeating_decimal_sum_l2865_286518

theorem repeating_decimal_sum : 
  (234 : ℚ) / 999 - (567 : ℚ) / 999 + (891 : ℚ) / 999 = (186 : ℚ) / 333 := by
  sorry

end repeating_decimal_sum_l2865_286518


namespace rachel_homework_l2865_286589

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 →
  math_pages = reading_pages + 4 →
  reading_pages = 3 :=
by
  sorry

end rachel_homework_l2865_286589


namespace square_sum_of_reciprocal_and_sum_l2865_286597

theorem square_sum_of_reciprocal_and_sum (x₁ x₂ : ℝ) :
  x₁ = 2 / (Real.sqrt 5 + Real.sqrt 3) →
  x₂ = Real.sqrt 5 + Real.sqrt 3 →
  x₁^2 + x₂^2 = 16 := by
sorry

end square_sum_of_reciprocal_and_sum_l2865_286597


namespace urea_formation_proof_l2865_286512

-- Define the chemical species
inductive Species
  | NH3
  | CO2
  | H2O
  | NH4_2CO3
  | NH4OH
  | NH2CONH2

-- Define the reaction equations
inductive Reaction
  | ammonium_carbonate_formation
  | ammonium_carbonate_hydrolysis
  | urea_formation

-- Define the initial quantities
def initial_quantities : Species → ℚ
  | Species.NH3 => 2
  | Species.CO2 => 1
  | Species.H2O => 1
  | _ => 0

-- Define the stoichiometric coefficients for each reaction
def stoichiometry : Reaction → Species → ℚ
  | Reaction.ammonium_carbonate_formation, Species.NH3 => -2
  | Reaction.ammonium_carbonate_formation, Species.CO2 => -1
  | Reaction.ammonium_carbonate_formation, Species.NH4_2CO3 => 1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4_2CO3 => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.H2O => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4OH => 2
  | Reaction.ammonium_carbonate_hydrolysis, Species.CO2 => 1
  | Reaction.urea_formation, Species.NH4OH => -1
  | Reaction.urea_formation, Species.CO2 => -1
  | Reaction.urea_formation, Species.NH2CONH2 => 1
  | Reaction.urea_formation, Species.H2O => 1
  | _, _ => 0

-- Define the function to calculate the amount of Urea formed
def urea_formed (reactions : List Reaction) : ℚ :=
  sorry

-- Theorem statement
theorem urea_formation_proof :
  urea_formed [Reaction.ammonium_carbonate_formation,
               Reaction.ammonium_carbonate_hydrolysis,
               Reaction.urea_formation] = 1 :=
sorry

end urea_formation_proof_l2865_286512


namespace least_number_of_radios_l2865_286592

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
theorem least_number_of_radios (n d : ℕ) (h_d_pos : d > 0) : 
  (∃ (d : ℕ), d > 0 ∧ 
    10 * n - 30 - (3 * d) / (2 * n) = 80 ∧ 
    ∀ m : ℕ, m < n → ¬(∃ (d' : ℕ), d' > 0 ∧ 10 * m - 30 - (3 * d') / (2 * m) = 80)) →
  n = 11 :=
sorry

end least_number_of_radios_l2865_286592


namespace solutions_of_z_sixth_power_eq_neg_64_l2865_286539

-- Define the complex number z
variable (z : ℂ)

-- Define the equation
def equation (z : ℂ) : Prop := z^6 = -64

-- State the theorem
theorem solutions_of_z_sixth_power_eq_neg_64 :
  (∀ z : ℂ, equation z ↔ z = 2*I ∨ z = -2*I) :=
sorry

end solutions_of_z_sixth_power_eq_neg_64_l2865_286539


namespace streamers_for_confetti_l2865_286552

/-- The price relationship between streamers and confetti packages -/
def price_relationship (p q : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x

/-- The theorem stating the number of streamer packages that can be bought for 10 confetti packages -/
theorem streamers_for_confetti (p q : ℝ) :
  price_relationship p q →
  |p - q| = 90 →
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x ∧
  10 * x = 4 * y :=
by sorry

end streamers_for_confetti_l2865_286552


namespace point_A_not_in_square_l2865_286583

-- Define the points
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, -4)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (1, 1)
def E : ℝ × ℝ := (3, -2)

-- Define a function to calculate the squared distance between two points
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define what it means for four points to form a square
def is_square (p q r s : ℝ × ℝ) : Prop :=
  let sides := [squared_distance p q, squared_distance q r, squared_distance r s, squared_distance s p]
  let diagonals := [squared_distance p r, squared_distance q s]
  (sides.all (· = sides.head!)) ∧ (diagonals.all (· = 2 * sides.head!))

-- Theorem statement
theorem point_A_not_in_square :
  ¬(is_square A B C D ∨ is_square A B C E ∨ is_square A B D E ∨ is_square A C D E) ∧
  (is_square B C D E) := by sorry

end point_A_not_in_square_l2865_286583


namespace retailer_items_sold_l2865_286551

/-- The problem of determining the number of items sold by a retailer -/
theorem retailer_items_sold 
  (profit_per_item : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (min_items_with_discount : ℝ) : 
  profit_per_item = 30 →
  profit_percentage = 0.16 →
  discount_percentage = 0.05 →
  min_items_with_discount = 156.86274509803923 →
  ∃ (items_sold : ℕ), items_sold = 100 := by
  sorry

end retailer_items_sold_l2865_286551


namespace fraction_denominator_l2865_286536

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 := by
  sorry

end fraction_denominator_l2865_286536


namespace jiale_pricing_correct_l2865_286577

/-- Represents the pricing and discount options for teapots and teacups -/
structure TeaSetPricing where
  teapot_price : ℝ
  teacup_price : ℝ
  option1 : ℝ → ℝ  -- Cost function for Option 1
  option2 : ℝ → ℝ  -- Cost function for Option 2

/-- The specific pricing structure for Jiale Supermarket -/
def jiale_pricing : TeaSetPricing :=
  { teapot_price := 90
    teacup_price := 25
    option1 := λ x => 25 * x + 325
    option2 := λ x => 22.5 * x + 405 }

/-- Theorem stating the correctness of the cost calculations -/
theorem jiale_pricing_correct (x : ℝ) (h : x > 5) :
  let p := jiale_pricing
  p.option1 x = 25 * x + 325 ∧ p.option2 x = 22.5 * x + 405 := by
  sorry

#check jiale_pricing_correct

end jiale_pricing_correct_l2865_286577


namespace unique_number_satisfying_conditions_l2865_286529

def is_between (x a b : ℕ) : Prop := a < x ∧ x < b

def is_single_digit (x : ℕ) : Prop := x < 10

theorem unique_number_satisfying_conditions :
  ∃! x : ℕ, is_between x 5 9 ∧ is_single_digit x ∧ x > 7 :=
sorry

end unique_number_satisfying_conditions_l2865_286529


namespace max_quarters_and_dimes_l2865_286549

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total amount Eva has in cents -/
def total_amount : ℕ := 480

/-- 
Given $4.80 in U.S. coins with an equal number of quarters and dimes,
prove that the maximum number of quarters (and dimes) is 13.
-/
theorem max_quarters_and_dimes :
  ∃ (n : ℕ), n * (quarter_value + dime_value) ≤ total_amount ∧
             ∀ (m : ℕ), m * (quarter_value + dime_value) ≤ total_amount → m ≤ n ∧
             n = 13 :=
sorry

end max_quarters_and_dimes_l2865_286549


namespace total_fish_caught_l2865_286534

/-- The total number of fish caught by Jason, Ryan, and Jeffery is 100 -/
theorem total_fish_caught (jeffery_fish : ℕ) (h1 : jeffery_fish = 60) 
  (h2 : ∃ ryan_fish : ℕ, jeffery_fish = 2 * ryan_fish) 
  (h3 : ∃ jason_fish : ℕ, ∃ ryan_fish : ℕ, ryan_fish = 3 * jason_fish) : 
  ∃ total : ℕ, total = jeffery_fish + ryan_fish + jason_fish ∧ total = 100 :=
by sorry

end total_fish_caught_l2865_286534


namespace simplify_and_evaluate_l2865_286587

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) : 
  (2 - 2*x/(x-2)) / ((x^2 - 4) / (x^2 - 4*x + 4)) = -2/3 := by
  sorry

end simplify_and_evaluate_l2865_286587


namespace mika_stickers_l2865_286523

/-- The number of stickers Mika has left after various additions and subtractions -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Theorem stating that Mika is left with 2 stickers -/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end mika_stickers_l2865_286523


namespace green_blue_difference_l2865_286594

/-- Represents the number of tiles in a hexagonal figure -/
structure HexFigure where
  blue : Nat
  green : Nat

/-- Calculates the number of tiles needed for a double border -/
def doubleBorderTiles : Nat := 2 * 18

/-- The initial hexagonal figure -/
def initialFigure : HexFigure := { blue := 13, green := 6 }

/-- Creates a new figure with twice as many tiles -/
def doubleFigure (f : HexFigure) : HexFigure :=
  { blue := 2 * f.blue, green := 2 * f.green }

/-- Adds a double border of green tiles to a figure -/
def addGreenBorder (f : HexFigure) : HexFigure :=
  { blue := f.blue, green := f.green + doubleBorderTiles }

/-- Calculates the total tiles for two figures -/
def totalTiles (f1 f2 : HexFigure) : HexFigure :=
  { blue := f1.blue + f2.blue, green := f1.green + f2.green }

theorem green_blue_difference :
  let secondFigure := addGreenBorder (doubleFigure initialFigure)
  let totalFigure := totalTiles initialFigure secondFigure
  totalFigure.green - totalFigure.blue = 15 := by sorry

end green_blue_difference_l2865_286594


namespace krtecek_return_distance_l2865_286519

/-- Represents a direction in 2D space -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Calculates the net displacement in centimeters for a list of movements -/
def netDisplacement (movements : List Movement) : ℝ × ℝ :=
  sorry

/-- Calculates the distance to the starting point given a net displacement -/
def distanceToStart (displacement : ℝ × ℝ) : ℝ :=
  sorry

/-- The list of Krteček's movements -/
def krtecekMovements : List Movement := [
  ⟨500, Direction.North⟩,
  ⟨230, Direction.West⟩,
  ⟨150, Direction.South⟩,
  ⟨370, Direction.West⟩,
  ⟨620, Direction.South⟩,
  ⟨53, Direction.East⟩,
  ⟨270, Direction.North⟩
]

theorem krtecek_return_distance :
  distanceToStart (netDisplacement krtecekMovements) = 547 := by
  sorry

end krtecek_return_distance_l2865_286519


namespace minimum_words_for_90_percent_l2865_286560

/-- Represents the French exam vocabulary test -/
structure FrenchExam where
  total_words : ℕ
  learned_words : ℕ
  score_threshold : ℚ

/-- Calculate the score for a given exam -/
def calculate_score (exam : FrenchExam) : ℚ :=
  (exam.learned_words + (exam.total_words - exam.learned_words) / 10) / exam.total_words

/-- Theorem stating the minimum number of words to learn for a 90% score -/
theorem minimum_words_for_90_percent (exam : FrenchExam) 
    (h1 : exam.total_words = 800)
    (h2 : exam.score_threshold = 9/10) :
    (∀ n : ℕ, n < 712 → calculate_score ⟨exam.total_words, n, exam.score_threshold⟩ < exam.score_threshold) ∧
    calculate_score ⟨exam.total_words, 712, exam.score_threshold⟩ ≥ exam.score_threshold :=
  sorry


end minimum_words_for_90_percent_l2865_286560
