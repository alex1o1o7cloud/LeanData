import Mathlib

namespace max_sectional_area_of_cone_l1260_126060

/-- The maximum sectional area of a cone --/
theorem max_sectional_area_of_cone (θ : Real) (l : Real) : 
  θ = π / 3 → l = 3 → (∀ α, 0 ≤ α ∧ α ≤ 2*π/3 → (1/2) * l^2 * Real.sin α ≤ 9/2) ∧ 
  ∃ α, 0 ≤ α ∧ α ≤ 2*π/3 ∧ (1/2) * l^2 * Real.sin α = 9/2 :=
by sorry

#check max_sectional_area_of_cone

end max_sectional_area_of_cone_l1260_126060


namespace purely_imaginary_complex_number_l1260_126055

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by
sorry

end purely_imaginary_complex_number_l1260_126055


namespace greatest_integer_with_gcf_four_l1260_126030

theorem greatest_integer_with_gcf_four : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_integer_with_gcf_four_l1260_126030


namespace polygon_exterior_angles_l1260_126093

theorem polygon_exterior_angles (n : ℕ) (h : n > 2) :
  (n : ℝ) * 60 = 360 → n = 6 := by
  sorry

end polygon_exterior_angles_l1260_126093


namespace children_on_bus_l1260_126089

theorem children_on_bus (initial_children : ℕ) (children_who_got_on : ℕ) : 
  initial_children = 18 → children_who_got_on = 7 → 
  initial_children + children_who_got_on = 25 := by
  sorry

end children_on_bus_l1260_126089


namespace log_2_base_10_bound_l1260_126032

theorem log_2_base_10_bound (h1 : 2^11 = 2048) (h2 : 2^12 = 4096) (h3 : 10^4 = 10000) :
  Real.log 2 / Real.log 10 < 4/11 := by
sorry

end log_2_base_10_bound_l1260_126032


namespace greatest_sum_consecutive_integers_l1260_126009

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → 
  (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) →
  n + (n + 1) = 43 := by
  sorry

end greatest_sum_consecutive_integers_l1260_126009


namespace cyclic_win_sets_count_l1260_126008

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins for each team
  losses : ℕ  -- number of losses for each team

/-- Conditions for the tournament -/
def tournament_conditions (t : Tournament) : Prop :=
  t.n * (t.n - 1) / 2 = t.wins * t.n ∧ 
  t.wins = 12 ∧ 
  t.losses = 8 ∧ 
  t.wins + t.losses = t.n - 1

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- Main theorem -/
theorem cyclic_win_sets_count (t : Tournament) : 
  tournament_conditions t → cyclic_win_sets t = 868 := by sorry

end cyclic_win_sets_count_l1260_126008


namespace solution_set_quadratic_inequality_l1260_126028

theorem solution_set_quadratic_inequality :
  Set.Ioo 0 3 = {x : ℝ | x^2 - 3*x < 0} :=
by sorry

end solution_set_quadratic_inequality_l1260_126028


namespace front_axle_wheels_count_l1260_126061

/-- Represents a truck with a specific wheel configuration -/
structure Truck where
  total_wheels : ℕ
  wheels_per_axle : ℕ
  front_axle_wheels : ℕ
  toll : ℚ

/-- Calculates the number of axles for a given truck -/
def num_axles (t : Truck) : ℕ :=
  (t.total_wheels - t.front_axle_wheels) / t.wheels_per_axle + 1

/-- Calculates the toll for a given number of axles -/
def toll_formula (x : ℕ) : ℚ :=
  (3/2) + (3/2) * (x - 2)

/-- Theorem stating that a truck with the given specifications has 2 wheels on its front axle -/
theorem front_axle_wheels_count (t : Truck) 
    (h1 : t.total_wheels = 18)
    (h2 : t.wheels_per_axle = 4)
    (h3 : t.toll = 6)
    (h4 : t.toll = toll_formula (num_axles t)) :
    t.front_axle_wheels = 2 := by
  sorry

end front_axle_wheels_count_l1260_126061


namespace sum_reciprocals_l1260_126067

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(a + ω) + 1/(b + ω) + 1/(c + ω) + 1/(d + ω) = 4/ω) :
  1/(a + 2) + 1/(b + 2) + 1/(c + 2) + 1/(d + 2) = 2 := by
sorry

end sum_reciprocals_l1260_126067


namespace intersection_M_N_l1260_126084

def M : Set ℝ := {x | 2 * x - x^2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end intersection_M_N_l1260_126084


namespace miles_driven_l1260_126076

/-- Given a journey with a total distance and remaining distance, 
    calculate the distance already traveled. -/
theorem miles_driven (total_journey : ℕ) (remaining : ℕ) 
    (h1 : total_journey = 1200) 
    (h2 : remaining = 816) : 
  total_journey - remaining = 384 := by
  sorry

end miles_driven_l1260_126076


namespace rita_remaining_money_l1260_126033

def initial_amount : ℕ := 400
def dress_cost : ℕ := 20
def pants_cost : ℕ := 12
def jacket_cost : ℕ := 30
def transportation_cost : ℕ := 5
def dress_quantity : ℕ := 5
def pants_quantity : ℕ := 3
def jacket_quantity : ℕ := 4

theorem rita_remaining_money :
  initial_amount - 
  (dress_cost * dress_quantity + 
   pants_cost * pants_quantity + 
   jacket_cost * jacket_quantity + 
   transportation_cost) = 139 := by
sorry

end rita_remaining_money_l1260_126033


namespace right_triangle_count_l1260_126073

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Definition of the rectangle ABCD and points E, F, G -/
def rectangle_setup :=
  let A := Point.mk 0 0
  let B := Point.mk 6 0
  let C := Point.mk 6 4
  let D := Point.mk 0 4
  let E := Point.mk 3 0
  let F := Point.mk 3 4
  let G := Point.mk 2 4
  (A, B, C, D, E, F, G)

/-- Function to count right triangles -/
def count_right_triangles (points : Point × Point × Point × Point × Point × Point × Point) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of right triangles is 16 -/
theorem right_triangle_count :
  count_right_triangles rectangle_setup = 16 := by
  sorry

end right_triangle_count_l1260_126073


namespace exists_divisible_pair_l1260_126006

def u : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => u (n + 1) + u n + 1

theorem exists_divisible_pair :
  ∃ n : ℕ, n ≥ 1 ∧ (2011^2012 ∣ u n) ∧ (2011^2012 ∣ u (n + 1)) := by
  sorry

end exists_divisible_pair_l1260_126006


namespace five_by_five_to_fifty_l1260_126065

/-- Represents a square cut into pieces --/
structure CutSquare :=
  (side : ℕ)
  (pieces : ℕ)

/-- Represents the result of reassembling cut pieces --/
structure ReassembledSquares :=
  (count : ℕ)
  (side : ℚ)

/-- Function that cuts a square and reassembles the pieces --/
def cut_and_reassemble (s : CutSquare) : ReassembledSquares :=
  sorry

/-- Theorem stating that a 5x5 square can be cut and reassembled into 50 equal squares --/
theorem five_by_five_to_fifty :
  ∃ (cs : CutSquare) (rs : ReassembledSquares),
    cs.side = 5 ∧
    rs = cut_and_reassemble cs ∧
    rs.count = 50 ∧
    rs.side * rs.side * rs.count = cs.side * cs.side :=
  sorry

end five_by_five_to_fifty_l1260_126065


namespace quadratic_equations_solutions_l1260_126094

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x - 16 = 0) ∧
  (∃ x : ℝ, 2*x^2 - 3*x - 5 = 0) →
  (∃ x : ℝ, x = 8 ∨ x = -2) ∧
  (∃ x : ℝ, x = 5/2 ∨ x = -1) :=
by sorry

end quadratic_equations_solutions_l1260_126094


namespace roots_sum_to_four_l1260_126020

theorem roots_sum_to_four : ∃ (x y : ℝ), x^2 - 4*x - 1 = 0 ∧ y^2 - 4*y - 1 = 0 ∧ x + y = 4 := by
  sorry

end roots_sum_to_four_l1260_126020


namespace probability_bounds_l1260_126029

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

/-- The maximum probability of a number in any segment being divisible by 10 -/
def max_probability : ℚ := 1

/-- The minimum non-zero probability of a number in any segment being divisible by 10 -/
def min_nonzero_probability : ℚ := 1 / 19

theorem probability_bounds :
  ∀ s : Segment, 
    probability_divisible_by_10 s ≤ max_probability ∧
    (probability_divisible_by_10 s ≠ 0 → probability_divisible_by_10 s ≥ min_nonzero_probability) :=
by sorry

end probability_bounds_l1260_126029


namespace multiply_72_68_l1260_126050

theorem multiply_72_68 : 72 * 68 = 4896 := by
  -- Proof goes here
  sorry

end multiply_72_68_l1260_126050


namespace quadratic_expression_value_l1260_126005

theorem quadratic_expression_value
  (a b c x : ℝ)
  (h1 : (2 - a)^2 + Real.sqrt (a^2 + b + c) + |c + 8| = 0)
  (h2 : a * x^2 + b * x + c = 0) :
  3 * x^2 + 6 * x + 1 = 13 := by sorry

end quadratic_expression_value_l1260_126005


namespace job_completion_time_l1260_126045

theorem job_completion_time (y : ℝ) : y > 0 → (
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ y = 2
) := by sorry

end job_completion_time_l1260_126045


namespace odd_function_negative_domain_l1260_126017

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = 2^x + 1) :
  ∀ x < 0, f x = -2^(-x) - 1 := by
sorry

end odd_function_negative_domain_l1260_126017


namespace min_value_sum_product_l1260_126000

theorem min_value_sum_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) ≥ 9 / 2 := by
  sorry

end min_value_sum_product_l1260_126000


namespace debt_equality_time_l1260_126099

/-- The number of days for two debts to become equal -/
def daysUntilEqualDebt (initialDebt1 initialDebt2 interestRate1 interestRate2 : ℚ) : ℚ :=
  (initialDebt2 - initialDebt1) / (initialDebt1 * interestRate1 - initialDebt2 * interestRate2)

/-- Theorem: Darren and Fergie's debts will be equal after 25 days -/
theorem debt_equality_time : 
  daysUntilEqualDebt 200 300 (8/100) (4/100) = 25 := by sorry

end debt_equality_time_l1260_126099


namespace zachs_bike_savings_l1260_126075

/-- Represents the problem of calculating how much more money Zach needs to earn --/
theorem zachs_bike_savings (bike_cost : ℕ) (discount_rate : ℚ) 
  (weekly_allowance : ℕ) (lawn_mowing_min lawn_mowing_max : ℕ) 
  (garage_cleaning : ℕ) (babysitting_rate babysitting_hours : ℕ) 
  (loan_to_repay : ℕ) (current_savings : ℕ) : 
  bike_cost = 150 →
  discount_rate = 1/10 →
  weekly_allowance = 5 →
  lawn_mowing_min = 8 →
  lawn_mowing_max = 12 →
  garage_cleaning = 15 →
  babysitting_rate = 7 →
  babysitting_hours = 3 →
  loan_to_repay = 10 →
  current_savings = 65 →
  ∃ (additional_money : ℕ), additional_money = 27 ∧ 
    (bike_cost - (discount_rate * bike_cost).floor) - current_savings + loan_to_repay = 
    weekly_allowance + lawn_mowing_max + garage_cleaning + (babysitting_rate * babysitting_hours) + additional_money :=
by sorry

end zachs_bike_savings_l1260_126075


namespace min_value_theorem_min_value_achievable_l1260_126037

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 :=
by sorry

end min_value_theorem_min_value_achievable_l1260_126037


namespace smallest_perimeter_consecutive_integer_triangle_l1260_126058

/-- A triangle with consecutive integer side lengths greater than 1 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  greater_than_one : a > 1

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The triangle inequality -/
def satisfies_triangle_inequality (t : ConsecutiveIntegerTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem smallest_perimeter_consecutive_integer_triangle :
  ∀ t : ConsecutiveIntegerTriangle,
  satisfies_triangle_inequality t →
  perimeter t ≥ 12 :=
sorry

end smallest_perimeter_consecutive_integer_triangle_l1260_126058


namespace total_addresses_is_40_l1260_126047

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem total_addresses_is_40 : total_addresses = 40 := by
  sorry

end total_addresses_is_40_l1260_126047


namespace modulus_of_complex_fraction_l1260_126022

theorem modulus_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (3 - I)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end modulus_of_complex_fraction_l1260_126022


namespace dogwood_trees_in_park_l1260_126016

theorem dogwood_trees_in_park (current trees_today trees_tomorrow total : ℕ) : 
  trees_today = 41 → 
  trees_tomorrow = 20 → 
  total = 100 → 
  current + trees_today + trees_tomorrow = total → 
  current = 39 := by
  sorry

end dogwood_trees_in_park_l1260_126016


namespace salad_dressing_weight_l1260_126081

theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density lemon_juice_density : ℝ) :
  bowl_capacity = 200 →
  oil_fraction = 3/5 →
  vinegar_fraction = 1/4 →
  oil_density = 5 →
  vinegar_density = 4 →
  lemon_juice_density = 2.5 →
  let lemon_juice_fraction : ℝ := 1 - oil_fraction - vinegar_fraction
  let oil_volume : ℝ := bowl_capacity * oil_fraction
  let vinegar_volume : ℝ := bowl_capacity * vinegar_fraction
  let lemon_juice_volume : ℝ := bowl_capacity * lemon_juice_fraction
  let total_weight : ℝ := oil_volume * oil_density + vinegar_volume * vinegar_density + lemon_juice_volume * lemon_juice_density
  total_weight = 875 := by
sorry


end salad_dressing_weight_l1260_126081


namespace min_discriminant_l1260_126059

/-- A quadratic trinomial that satisfies the problem conditions -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  nonnegative : ∀ x, a * x^2 + b * x + c ≥ 0
  below_curve : ∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)

/-- The discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℝ := q.b^2 - 4 * q.a * q.c

/-- The theorem stating the minimum value of the discriminant -/
theorem min_discriminant :
  (∀ q : QuadraticTrinomial, discriminant q ≥ -4) ∧
  (∃ q : QuadraticTrinomial, discriminant q = -4) := by sorry

end min_discriminant_l1260_126059


namespace arithmetic_sequence_ratio_l1260_126098

/-- An arithmetic sequence with first four terms a, x, b, 2x has the property that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) : 
  (∃ d : ℝ, x - a = d ∧ b - x = d ∧ 2*x - b = d) → a/b = 1/3 := by
  sorry

end arithmetic_sequence_ratio_l1260_126098


namespace hotel_assignment_theorem_l1260_126053

/-- The number of ways to assign 6 friends to 6 rooms with given constraints -/
def assignmentWays : ℕ := sorry

/-- The total number of rooms available -/
def totalRooms : ℕ := 6

/-- The number of friends to be assigned -/
def totalFriends : ℕ := 6

/-- The maximum number of friends allowed per room -/
def maxFriendsPerRoom : ℕ := 2

/-- The maximum number of rooms that can be used -/
def maxRoomsUsed : ℕ := 5

theorem hotel_assignment_theorem :
  assignmentWays = 10440 ∧
  totalRooms = 6 ∧
  totalFriends = 6 ∧
  maxFriendsPerRoom = 2 ∧
  maxRoomsUsed = 5 := by sorry

end hotel_assignment_theorem_l1260_126053


namespace monthly_savings_prediction_l1260_126048

/-- Linear regression equation for monthly savings prediction -/
def linear_regression (x : ℝ) (b_hat : ℝ) (a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

theorem monthly_savings_prediction 
  (n : ℕ) (x_bar : ℝ) (b_hat : ℝ) (a_hat : ℝ) :
  n = 10 →
  x_bar = 8 →
  b_hat = 0.3 →
  a_hat = -0.4 →
  linear_regression 7 b_hat a_hat = 1.7 :=
by sorry

end monthly_savings_prediction_l1260_126048


namespace complex_absolute_value_l1260_126083

theorem complex_absolute_value (z : ℂ) : z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end complex_absolute_value_l1260_126083


namespace baseball_season_games_l1260_126026

/-- The number of baseball games in a season -/
def games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: There are 14 baseball games in a season -/
theorem baseball_season_games :
  games_in_season 7 2 = 14 := by
  sorry

end baseball_season_games_l1260_126026


namespace problem_statement_l1260_126074

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (h : b * Real.log a - a * Real.log b = a - b) : 
  (a + b - a * b > 1) ∧ (a + b > 2) ∧ (1 / a + 1 / b > 2) := by
  sorry

end problem_statement_l1260_126074


namespace excess_calories_is_770_l1260_126082

/-- Calculates the excess calories consumed by James after snacking and exercising -/
def excess_calories : ℕ :=
  let cheezit_bags : ℕ := 3
  let cheezit_oz_per_bag : ℕ := 2
  let cheezit_cal_per_oz : ℕ := 150
  let chocolate_bars : ℕ := 2
  let chocolate_cal_per_bar : ℕ := 250
  let popcorn_cal : ℕ := 500
  let run_minutes : ℕ := 40
  let run_cal_per_minute : ℕ := 12
  let swim_minutes : ℕ := 30
  let swim_cal_per_minute : ℕ := 15
  let cycle_minutes : ℕ := 20
  let cycle_cal_per_minute : ℕ := 10

  let total_calories_consumed : ℕ := 
    cheezit_bags * cheezit_oz_per_bag * cheezit_cal_per_oz +
    chocolate_bars * chocolate_cal_per_bar +
    popcorn_cal

  let total_calories_burned : ℕ := 
    run_minutes * run_cal_per_minute +
    swim_minutes * swim_cal_per_minute +
    cycle_minutes * cycle_cal_per_minute

  total_calories_consumed - total_calories_burned

theorem excess_calories_is_770 : excess_calories = 770 := by
  sorry

end excess_calories_is_770_l1260_126082


namespace flea_return_probability_l1260_126064

/-- A flea jumps on a number line with the following properties:
    - It starts at 0
    - Each jump has a length of 1
    - The probability of jumping in the same direction as the previous jump is p
    - The probability of jumping in the opposite direction is 1-p -/
def FleaJump (p : ℝ) := 
  {flea : ℕ → ℝ // flea 0 = 0 ∧ ∀ n, |flea (n+1) - flea n| = 1}

/-- The probability that the flea returns to 0 -/
noncomputable def ReturnProbability (p : ℝ) : ℝ := sorry

/-- The theorem stating the probability of the flea returning to 0 -/
theorem flea_return_probability (p : ℝ) : 
  ReturnProbability p = if p = 1 then 0 else 1 := by sorry

end flea_return_probability_l1260_126064


namespace average_of_a_and_b_l1260_126010

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 20 → (a + b) / 2 = 41 := by
sorry

end average_of_a_and_b_l1260_126010


namespace simplify_expression_l1260_126038

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / (3*x))^2) = (Real.sqrt (x^6 - 4*x^3 + 36*x^2 + 4)) / (3*x) := by
  sorry

end simplify_expression_l1260_126038


namespace toms_spending_ratio_l1260_126071

def monthly_allowance : ℚ := 12
def first_week_spending_ratio : ℚ := 1/3
def remaining_money : ℚ := 6

theorem toms_spending_ratio :
  let first_week_spending := monthly_allowance * first_week_spending_ratio
  let money_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := money_after_first_week - remaining_money
  second_week_spending / money_after_first_week = 1/4 := by
sorry

end toms_spending_ratio_l1260_126071


namespace reduced_oil_price_l1260_126011

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  price_reduction_percent : ℝ
  additional_amount : ℝ
  fixed_cost : ℝ

/-- Theorem stating the reduced price of oil given the conditions --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction)
  (h1 : scenario.price_reduction_percent = 20)
  (h2 : scenario.additional_amount = 4)
  (h3 : scenario.fixed_cost = 600)
  (h4 : scenario.reduced_price = scenario.original_price * (1 - scenario.price_reduction_percent / 100))
  (h5 : scenario.fixed_cost = (scenario.fixed_cost / scenario.original_price + scenario.additional_amount) * scenario.reduced_price) :
  scenario.reduced_price = 30 := by
  sorry

#check reduced_oil_price

end reduced_oil_price_l1260_126011


namespace units_digit_G_100_l1260_126046

/-- The sequence G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G_100 is 4 -/
theorem units_digit_G_100 : unitsDigit (G 100) = 4 := by sorry

end units_digit_G_100_l1260_126046


namespace arithmetic_sequence_sum_relation_l1260_126025

/-- An arithmetic sequence is represented by its sums of first n, 2n, and 3n terms. -/
structure ArithmeticSequenceSums where
  S : ℝ  -- Sum of first n terms
  T : ℝ  -- Sum of first 2n terms
  R : ℝ  -- Sum of first 3n terms

/-- 
For any arithmetic sequence, given the sums of its first n, 2n, and 3n terms,
the sum of the first 3n terms equals three times the difference between
the sum of the first 2n terms and the sum of the first n terms.
-/
theorem arithmetic_sequence_sum_relation (seq : ArithmeticSequenceSums) : 
  seq.R = 3 * (seq.T - seq.S) := by
  sorry


end arithmetic_sequence_sum_relation_l1260_126025


namespace sum_first_100_even_integers_l1260_126086

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The sum of the first 100 positive even integers is 10100 -/
theorem sum_first_100_even_integers : sumFirstNEvenIntegers 100 = 10100 := by
  sorry

end sum_first_100_even_integers_l1260_126086


namespace max_value_condition_l1260_126002

noncomputable def f (x a : ℝ) : ℝ := -(Real.sin x + a/2)^2 + 3 + a^2/4

theorem max_value_condition (a : ℝ) :
  (∀ x, f x a ≤ 5) ∧ (∃ x, f x a = 5) ↔ a = 3 ∨ a = -3 := by sorry

end max_value_condition_l1260_126002


namespace divisible_by_twelve_l1260_126056

theorem divisible_by_twelve (n : ℤ) : 12 ∣ n^2 * (n^2 - 1) := by
  sorry

end divisible_by_twelve_l1260_126056


namespace curve_M_properties_l1260_126080

-- Define the curve M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^(1/2) + p.2^(1/2) = 1}

-- Theorem statement
theorem curve_M_properties :
  (∃ (p : ℝ × ℝ), p ∈ M ∧ Real.sqrt (p.1^2 + p.2^2) < Real.sqrt 2 / 2) ∧
  (∀ (S : Set (ℝ × ℝ)), S ⊆ M → MeasureTheory.volume S ≤ 1/2) := by
  sorry

end curve_M_properties_l1260_126080


namespace channel_count_is_164_l1260_126034

/-- Calculates the final number of channels after a series of changes --/
def final_channels (initial : ℕ) : ℕ :=
  let after_first := initial - 20 + 12
  let after_second := after_first - 10 + 8
  let after_third := after_second + 15 - 5
  let overlap := (after_third * 10) / 100
  let after_fourth := after_third + (25 - overlap)
  after_fourth + 7 - 3

/-- Theorem stating that given the initial number of channels and the series of changes, 
    the final number of channels is 164 --/
theorem channel_count_is_164 : final_channels 150 = 164 := by
  sorry

end channel_count_is_164_l1260_126034


namespace cannot_be_even_after_odd_operations_l1260_126097

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to determine the parity of a number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- Function to toggle the parity -/
def toggleParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

theorem cannot_be_even_after_odd_operations
  (initial : ℕ)
  (operations : ℕ)
  (h_initial_even : getParity initial = Parity.Even)
  (h_operations_odd : getParity operations = Parity.Odd) :
  ∃ (final : ℕ), getParity final = Parity.Odd ∧
    ∃ (f : ℕ → ℕ), (∀ n, f n = n + 1 ∨ f n = n - 1) ∧
      (f^[operations] initial = final) :=
sorry

end cannot_be_even_after_odd_operations_l1260_126097


namespace choir_average_age_l1260_126085

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 8)
  (h2 : num_males = 17)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 768 / 25 := by
sorry

end choir_average_age_l1260_126085


namespace simplify_expression_l1260_126087

theorem simplify_expression : 1 - 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 := by
  sorry

end simplify_expression_l1260_126087


namespace floor_pi_plus_four_l1260_126014

theorem floor_pi_plus_four : ⌊Real.pi + 4⌋ = 7 := by
  sorry

end floor_pi_plus_four_l1260_126014


namespace smallest_n_divisible_l1260_126068

/-- The smallest positive integer n such that (x+1)^n - 1 is divisible by x^2 + 1 modulo 3 -/
def smallest_n : ℕ := 8

/-- The divisor polynomial -/
def divisor_poly (x : ℤ) : ℤ := x^2 + 1

/-- The dividend polynomial -/
def dividend_poly (x : ℤ) (n : ℕ) : ℤ := (x + 1)^n - 1

/-- Divisibility modulo 3 -/
def is_divisible_mod_3 (a b : ℤ → ℤ) : Prop :=
  ∃ (p q : ℤ → ℤ), ∀ x, a x = b x * p x + 3 * q x

theorem smallest_n_divisible :
  (∀ n < smallest_n, ¬ is_divisible_mod_3 (dividend_poly · n) divisor_poly) ∧
  is_divisible_mod_3 (dividend_poly · smallest_n) divisor_poly :=
sorry

end smallest_n_divisible_l1260_126068


namespace rebus_puzzle_solution_l1260_126062

theorem rebus_puzzle_solution :
  ∀ (A B C : ℕ),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A ≠ B → B ≠ C → A ≠ C →
    A < 10 → B < 10 → C < 10 →
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) →
    (100 * A + 10 * C + C) = 1416 →
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end rebus_puzzle_solution_l1260_126062


namespace students_behind_minyoung_l1260_126091

/-- Given a line of students with Minyoung in it, this theorem proves
    that the number of students behind Minyoung is equal to the total
    number of students minus the number of students in front of Minyoung
    minus 1 (Minyoung herself). -/
theorem students_behind_minyoung
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 35)
  (h2 : students_in_front = 27) :
  total_students - students_in_front - 1 = 7 :=
by sorry

end students_behind_minyoung_l1260_126091


namespace correct_notebooks_A_correct_min_full_price_sales_l1260_126024

/-- Represents the bookstore problem with notebooks of two types -/
structure BookstoreProblem where
  total_notebooks : ℕ
  cost_price_A : ℕ
  cost_price_B : ℕ
  total_cost : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  discount_A : ℚ
  profit_threshold : ℕ

/-- The specific instance of the bookstore problem -/
def problem : BookstoreProblem :=
  { total_notebooks := 350
  , cost_price_A := 12
  , cost_price_B := 15
  , total_cost := 4800
  , selling_price_A := 20
  , selling_price_B := 25
  , discount_A := 0.7
  , profit_threshold := 2348 }

/-- The number of type A notebooks purchased -/
def notebooks_A (p : BookstoreProblem) : ℕ := sorry

/-- The number of type B notebooks purchased -/
def notebooks_B (p : BookstoreProblem) : ℕ := sorry

/-- The minimum number of notebooks of each type that must be sold at full price -/
def min_full_price_sales (p : BookstoreProblem) : ℕ := sorry

/-- Theorem stating the correct number of type A notebooks -/
theorem correct_notebooks_A : notebooks_A problem = 150 := by sorry

/-- Theorem stating the correct minimum number of full-price sales -/
theorem correct_min_full_price_sales : min_full_price_sales problem = 128 := by sorry

end correct_notebooks_A_correct_min_full_price_sales_l1260_126024


namespace cylinder_min_surface_area_l1260_126072

/-- Given a cone with base radius 4 and slant height 5, and a cylinder with equal volume,
    the surface area of the cylinder is minimized when its base radius is 2. -/
theorem cylinder_min_surface_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r^2 * h = (1/3) * π * 4^2 * 3) →
  (∀ r' h' : ℝ, r' > 0 → h' > 0 → π * r'^2 * h' = (1/3) * π * 4^2 * 3 → 
    2 * π * r * (r + h) ≤ 2 * π * r' * (r' + h')) →
  r = 2 := by sorry

end cylinder_min_surface_area_l1260_126072


namespace custom_op_result_l1260_126095

/-- Define the custom operation ã — -/
def custom_op (a b : ℤ) : ℤ := 2*a - 3*b + a*b

/-- Theorem stating that (1 ã — 2) - 2 = -4 -/
theorem custom_op_result : custom_op 1 2 - 2 = -4 := by
  sorry

end custom_op_result_l1260_126095


namespace min_value_function_l1260_126035

theorem min_value_function (x : ℝ) (h : x > 3) : 
  (1 / (x - 3)) + x ≥ 5 ∧ ∃ y > 3, (1 / (y - 3)) + y = 5 := by
  sorry

end min_value_function_l1260_126035


namespace square_of_1037_l1260_126044

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := by
  -- The proof goes here
  sorry

end square_of_1037_l1260_126044


namespace derivative_product_at_one_and_neg_one_l1260_126079

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- State the theorem
theorem derivative_product_at_one_and_neg_one :
  (deriv f 1) * (deriv f (-1)) = 9 := by sorry

end derivative_product_at_one_and_neg_one_l1260_126079


namespace a_less_than_two_necessary_not_sufficient_l1260_126019

/-- A quadratic equation x^2 + ax + 1 = 0 with real coefficient a -/
def quadratic_equation (a : ℝ) (x : ℂ) : Prop :=
  x^2 + a*x + 1 = 0

/-- The equation has complex roots -/
def has_complex_roots (a : ℝ) : Prop :=
  ∃ x : ℂ, quadratic_equation a x ∧ x.im ≠ 0

theorem a_less_than_two_necessary_not_sufficient :
  (∀ a : ℝ, has_complex_roots a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ ¬has_complex_roots a) :=
sorry

end a_less_than_two_necessary_not_sufficient_l1260_126019


namespace system_solution_l1260_126077

theorem system_solution (x y u v : ℝ) : 
  (x^2 + y^2 + u^2 + v^2 = 4) ∧
  (x*u + y*v + x*v + y*u = 0) ∧
  (x*y*u + y*u*v + u*v*x + v*x*y = -2) ∧
  (x*y*u*v = -1) →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = -1 + Real.sqrt 2 ∧ y = -1 - Real.sqrt 2 ∧ u = 1 ∧ v = -1)) :=
by sorry

end system_solution_l1260_126077


namespace complex_sum_properties_l1260_126057

open Complex

/-- Given complex numbers z and u with the specified properties, prove the required statements -/
theorem complex_sum_properties (α β : ℝ) (z u : ℂ) 
  (hz : z = Complex.exp (I * α))  -- z = cos α + i sin α
  (hu : u = Complex.exp (I * β))  -- u = cos β + i sin β
  (hsum : z + u = (4/5 : ℂ) + (3/5 : ℂ) * I) : 
  (Complex.tan (α + β) = 24/7) ∧ (z^2 + u^2 + z*u = 0) := by
  sorry


end complex_sum_properties_l1260_126057


namespace enchiladas_ordered_l1260_126039

/-- The number of enchiladas you ordered -/
def your_enchiladas : ℕ := 3

/-- The cost of each taco in dollars -/
def taco_cost : ℚ := 9/10

/-- Your bill in dollars (without tax) -/
def your_bill : ℚ := 78/10

/-- Your friend's bill in dollars (without tax) -/
def friend_bill : ℚ := 127/10

/-- The cost of each enchilada in dollars -/
def enchilada_cost : ℚ := 2

theorem enchiladas_ordered :
  (2 * taco_cost + your_enchiladas * enchilada_cost = your_bill) ∧
  (3 * taco_cost + 5 * enchilada_cost = friend_bill) :=
by sorry

end enchiladas_ordered_l1260_126039


namespace imaginary_part_of_i_times_1_plus_2i_l1260_126007

theorem imaginary_part_of_i_times_1_plus_2i (i : ℂ) (h : i * i = -1) :
  Complex.im (i * (1 + 2*i)) = 1 := by
  sorry

end imaginary_part_of_i_times_1_plus_2i_l1260_126007


namespace sum_of_factors_of_30_l1260_126042

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_of_30 : sum_of_factors 30 = 72 := by sorry

end sum_of_factors_of_30_l1260_126042


namespace unique_pie_solution_l1260_126018

/-- Represents the number of pies of each type -/
structure PieCount where
  raspberry : ℕ
  blueberry : ℕ
  strawberry : ℕ

/-- Checks if the given pie counts satisfy the problem conditions -/
def satisfiesConditions (pies : PieCount) : Prop :=
  pies.raspberry = (pies.raspberry + pies.blueberry + pies.strawberry) / 2 ∧
  pies.blueberry = pies.raspberry - 14 ∧
  pies.strawberry = (pies.raspberry + pies.blueberry) / 2

/-- Theorem stating that the given pie counts are the unique solution -/
theorem unique_pie_solution :
  ∃! (pies : PieCount), satisfiesConditions pies ∧
    pies.raspberry = 21 ∧ pies.blueberry = 7 ∧ pies.strawberry = 14 := by
  sorry

end unique_pie_solution_l1260_126018


namespace complex_power_sum_l1260_126041

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^2010 + (1 / z^2010) = 0 := by
  sorry

end complex_power_sum_l1260_126041


namespace system_of_equations_l1260_126054

theorem system_of_equations (x y m : ℝ) : 
  x - y = 5 → 
  x + 2*y = 3*m - 1 → 
  2*x + y = 13 → 
  m = 3 := by
sorry

end system_of_equations_l1260_126054


namespace tommy_savings_tommy_current_savings_l1260_126070

/-- Calculates the amount of money Tommy already has -/
theorem tommy_savings (num_books : ℕ) (cost_per_book : ℕ) (amount_to_save : ℕ) : ℕ :=
  num_books * cost_per_book - amount_to_save

/-- Proves that Tommy already has $13 -/
theorem tommy_current_savings : tommy_savings 8 5 27 = 13 := by
  sorry

end tommy_savings_tommy_current_savings_l1260_126070


namespace sailboat_two_sail_speed_l1260_126013

/-- Represents the speed of a sailboat in knots -/
structure SailboatSpeed :=
  (speed : ℝ)

/-- Represents the travel conditions for a sailboat -/
structure TravelConditions :=
  (oneSpeedSail : SailboatSpeed)
  (twoSpeedSail : SailboatSpeed)
  (timeOneSail : ℝ)
  (timeTwoSail : ℝ)
  (totalDistance : ℝ)
  (nauticalMileToLandMile : ℝ)

/-- The main theorem stating the speed of the sailboat with two sails -/
theorem sailboat_two_sail_speed 
  (conditions : TravelConditions)
  (h1 : conditions.oneSpeedSail.speed = 25)
  (h2 : conditions.timeOneSail = 4)
  (h3 : conditions.timeTwoSail = 4)
  (h4 : conditions.totalDistance = 345)
  (h5 : conditions.nauticalMileToLandMile = 1.15)
  : conditions.twoSpeedSail.speed = 50 := by
  sorry

#check sailboat_two_sail_speed

end sailboat_two_sail_speed_l1260_126013


namespace vacation_book_pairs_l1260_126003

/-- The number of ways to choose two books of different genres -/
def different_genre_pairs (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem stating that choosing two books of different genres from the given collection results in 33 pairs -/
theorem vacation_book_pairs :
  different_genre_pairs 3 4 3 = 33 := by
  sorry

end vacation_book_pairs_l1260_126003


namespace danielles_rooms_l1260_126023

theorem danielles_rooms (d h g : ℕ) : 
  h = 3 * d →  -- Heidi's apartment has 3 times as many rooms as Danielle's
  g = h / 9 →  -- Grant's apartment has 1/9 as many rooms as Heidi's
  g = 2 →      -- Grant's apartment has 2 rooms
  d = 6        -- Prove that Danielle's apartment has 6 rooms
:= by sorry

end danielles_rooms_l1260_126023


namespace revenue_change_l1260_126036

theorem revenue_change (R : ℝ) (p : ℝ) (h1 : R > 0) :
  (R + p / 100 * R) * (1 - p / 100) = R * (1 - 4 / 100) →
  p = 20 := by
sorry

end revenue_change_l1260_126036


namespace range_of_sum_l1260_126012

theorem range_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  0 < x + y ∧ x + y < 1 := by
sorry

end range_of_sum_l1260_126012


namespace line_translation_invariance_l1260_126004

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_invariance (l : Line) (dx dy : ℝ) :
  l.slope = -2 ∧ l.intercept = -2 ∧ dx = -1 ∧ dy = 2 →
  translate l dx dy = l :=
sorry

end line_translation_invariance_l1260_126004


namespace art_museum_survey_l1260_126027

theorem art_museum_survey (total : ℕ) (not_enjoyed_not_understood : ℕ) (enjoyed : ℕ) (understood : ℕ) :
  total = 400 →
  not_enjoyed_not_understood = 100 →
  enjoyed = understood →
  (enjoyed : ℚ) / total = 3 / 8 := by
  sorry

end art_museum_survey_l1260_126027


namespace sum_of_integers_l1260_126063

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 245) (h2 : x.val * y.val = 120) :
  (x.val : ℝ) + y.val = Real.sqrt 485 := by
  sorry

end sum_of_integers_l1260_126063


namespace sin_n_squared_not_converge_to_zero_l1260_126049

theorem sin_n_squared_not_converge_to_zero :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ |Real.sin (n^2 : ℝ)| ≥ ε :=
sorry

end sin_n_squared_not_converge_to_zero_l1260_126049


namespace sum_in_base10_l1260_126096

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_in_base10 : 
  base14ToBase10 356 + base13ToBase10 409 = 1505 := by sorry

end sum_in_base10_l1260_126096


namespace quadratic_two_real_roots_l1260_126066

theorem quadratic_two_real_roots 
  (a b c : ℝ) 
  (h : a * (a + b + c) < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end quadratic_two_real_roots_l1260_126066


namespace sum_f_positive_l1260_126021

-- Define the function f
def f (x : ℝ) : ℝ := x + x^3

-- State the theorem
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) 
  (h₂ : x₂ + x₃ > 0) 
  (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end sum_f_positive_l1260_126021


namespace sector_max_area_l1260_126090

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area (r : ℝ) (α : ℝ) (h1 : r > 0) (h2 : α > 0) 
  (h3 : 2 * r + r * α = 36) : 
  (∀ β : ℝ, β > 0 → 2 * r + r * β = 36 → r * r * α / 2 ≤ r * r * β / 2) → α = 2 := 
sorry

end sector_max_area_l1260_126090


namespace factorial_ratio_100_98_l1260_126051

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_100_98 : factorial 100 / factorial 98 = 9900 := by
  sorry

end factorial_ratio_100_98_l1260_126051


namespace system_solution_l1260_126015

-- Define the solution set
def solution_set := {x : ℝ | 0 < x ∧ x < 1}

-- Define the system of inequalities
def inequality1 (x : ℝ) := |x| - 1 < 0
def inequality2 (x : ℝ) := x^2 - 3*x < 0

-- Theorem statement
theorem system_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ (inequality1 x ∧ inequality2 x) :=
by sorry

end system_solution_l1260_126015


namespace microphotonics_allocation_l1260_126031

/-- Represents the budget allocation for Megatech Corporation -/
structure BudgetAllocation where
  total : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ
  basic_astrophysics_degrees : ℝ
  microphotonics : ℝ

/-- The theorem stating that the microphotonics allocation is 14% given the conditions -/
theorem microphotonics_allocation
  (budget : BudgetAllocation)
  (h1 : budget.total = 100)
  (h2 : budget.home_electronics = 19)
  (h3 : budget.food_additives = 10)
  (h4 : budget.genetically_modified_microorganisms = 24)
  (h5 : budget.industrial_lubricants = 8)
  (h6 : budget.basic_astrophysics_degrees = 90)
  (h7 : budget.microphotonics = budget.total - (budget.home_electronics + budget.food_additives + budget.genetically_modified_microorganisms + budget.industrial_lubricants + (budget.basic_astrophysics_degrees / 360 * 100))) :
  budget.microphotonics = 14 := by
  sorry

end microphotonics_allocation_l1260_126031


namespace system_solution_unique_l1260_126043

theorem system_solution_unique (x y : ℚ) : 
  (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ↔ (x = 1/2) ∧ (y = -3) := by
  sorry

end system_solution_unique_l1260_126043


namespace reflected_beam_angle_l1260_126088

/-- Given a fixed beam of light falling on a mirror at an acute angle α with its projection
    on the mirror plane, and the mirror rotated by an acute angle β around this projection,
    the angle θ between the two reflected beams (before and after rotation) is given by
    θ = arccos(1 - 2 * sin²α * sin²β) -/
theorem reflected_beam_angle (α β : Real) (h_α : 0 < α ∧ α < π/2) (h_β : 0 < β ∧ β < π/2) :
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) :=
sorry

end reflected_beam_angle_l1260_126088


namespace fraction_sum_equality_l1260_126078

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  1 / (x - 2) + 2 / (x + 2) + 4 / (4 - x^2) = 3 / (x + 2) := by
  sorry

end fraction_sum_equality_l1260_126078


namespace log_base_10_of_7_l1260_126052

theorem log_base_10_of_7 (p q : ℝ) 
  (hp : Real.log 5 / Real.log 4 = p) 
  (hq : Real.log 7 / Real.log 5 = q) : 
  Real.log 7 / Real.log 10 = 2 * p * q / (2 * p + 1) := by
  sorry

end log_base_10_of_7_l1260_126052


namespace same_remainder_divisor_l1260_126001

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end same_remainder_divisor_l1260_126001


namespace opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l1260_126040

-- Statement 1
theorem opposite_equal_implies_zero (x : ℝ) : x = -x → x = 0 := by sorry

-- Statement 2
theorem abs_equal_implies_equal_or_opposite (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by sorry

-- Statement 3
theorem sum_product_condition_implies_abs_equality (a b : ℝ) : 
  a + b < 0 → ab > 0 → |7*a + 3*b| = -(7*a + 3*b) := by sorry

-- Statement 4
theorem abs_plus_self_nonnegative (m : ℚ) : |m| + m ≥ 0 := by sorry

end opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l1260_126040


namespace prob_at_least_three_speak_l1260_126069

/-- The probability of a single baby speaking -/
def p : ℚ := 1/5

/-- The number of babies in the cluster -/
def n : ℕ := 7

/-- The probability that exactly k out of n babies will speak -/
def prob_exactly (k : ℕ) : ℚ :=
  (n.choose k) * (1 - p)^(n - k) * p^k

/-- The probability that at least 3 out of 7 babies will speak -/
theorem prob_at_least_three_speak : 
  1 - (prob_exactly 0 + prob_exactly 1 + prob_exactly 2) = 45349/78125 := by
  sorry

end prob_at_least_three_speak_l1260_126069


namespace bruce_mangoes_l1260_126092

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    the quantity and price of grapes, and the price of mangoes. -/
def mangoes_purchased (total_paid : ℕ) (grape_qty : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_paid - grape_qty * grape_price) / mango_price : ℕ)

/-- Proves that Bruce purchased 9 kg of mangoes given the problem conditions. -/
theorem bruce_mangoes :
  mangoes_purchased 1055 8 70 55 = 9 := by
  sorry

end bruce_mangoes_l1260_126092
