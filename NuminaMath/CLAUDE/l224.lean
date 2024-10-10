import Mathlib

namespace original_alcohol_percentage_l224_22424

/-- Proves that a 20-litre mixture of alcohol and water, when mixed with 3 litres of water,
    resulting in a new mixture with 17.391304347826086% alcohol, must have originally
    contained 20% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_percentage : ℝ)
  (h1 : original_volume = 20)
  (h2 : added_water = 3)
  (h3 : new_percentage = 17.391304347826086) :
  (original_volume * (100 / (original_volume + added_water)) * new_percentage / 100) = 20 :=
sorry

end original_alcohol_percentage_l224_22424


namespace fraction_sum_between_extremes_l224_22452

theorem fraction_sum_between_extremes 
  (a b c d n p x y : ℚ) 
  (h_pos : b > 0 ∧ d > 0 ∧ p > 0 ∧ y > 0)
  (h_order : a/b > c/d ∧ c/d > n/p ∧ n/p > x/y) : 
  x/y < (a + c + n + x) / (b + d + p + y) ∧ 
  (a + c + n + x) / (b + d + p + y) < a/b := by
sorry

end fraction_sum_between_extremes_l224_22452


namespace smallest_whole_number_above_triangle_perimeter_l224_22436

theorem smallest_whole_number_above_triangle_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 8 > 25 →
  s + 25 > 8 →
  8 + 25 > s →
  (∃ n : ℕ, n = 67 ∧ ∀ m : ℕ, (m : ℝ) > 8 + 25 + s → m ≥ n) :=
by sorry

end smallest_whole_number_above_triangle_perimeter_l224_22436


namespace exponent_equation_l224_22454

theorem exponent_equation (a : ℝ) (m : ℝ) (h1 : a ≠ 0) (h2 : a^5 * (a^m)^3 = a^11) : m = 2 := by
  sorry

end exponent_equation_l224_22454


namespace solution_set_quadratic_inequality_l224_22410

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by
  sorry

end solution_set_quadratic_inequality_l224_22410


namespace simplify_expression_l224_22447

theorem simplify_expression : (6 + 6 + 12) / 3 - 2 * 2 = 4 := by
  sorry

end simplify_expression_l224_22447


namespace solve_otimes_equation_l224_22486

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 3 * b

-- State the theorem
theorem solve_otimes_equation :
  ∃! x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 := by
  sorry

end solve_otimes_equation_l224_22486


namespace fixed_point_on_line_l224_22412

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 1

-- State the theorem
theorem fixed_point_on_line 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a (-1) = 2) 
  (b : ℝ) 
  (h4 : b * (-1) + 2 + 1 = 0) :
  b = 3 := by sorry

end fixed_point_on_line_l224_22412


namespace base4_addition_subtraction_l224_22499

/-- Converts a base 4 number represented as a list of digits to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits. -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_addition_subtraction :
  let a := base4ToNat [3, 2, 1]
  let b := base4ToNat [2, 0, 3]
  let c := base4ToNat [1, 1, 2]
  let result := base4ToNat [1, 0, 2, 1]
  (a + b) - c = result := by sorry

end base4_addition_subtraction_l224_22499


namespace fayes_remaining_money_fayes_remaining_money_is_30_l224_22449

/-- Calculates the remaining money for Faye after receiving money from her mother and making purchases. -/
theorem fayes_remaining_money (initial_money : ℝ) (cupcake_price : ℝ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℝ) (cookie_box_quantity : ℕ) : ℝ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Proves that Faye's remaining money is $30 given the initial conditions. -/
theorem fayes_remaining_money_is_30 : 
  fayes_remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end fayes_remaining_money_fayes_remaining_money_is_30_l224_22449


namespace carnival_earnings_value_l224_22437

/-- The total earnings from two ring toss games at a carnival -/
def carnival_earnings : ℕ :=
  let game1_period1 := 88
  let game1_rate1 := 761
  let game1_period2 := 20
  let game1_rate2 := 487
  let game2_period1 := 66
  let game2_rate1 := 569
  let game2_period2 := 15
  let game2_rate2 := 932
  let game1_earnings := game1_period1 * game1_rate1 + game1_period2 * game1_rate2
  let game2_earnings := game2_period1 * game2_rate1 + game2_period2 * game2_rate2
  game1_earnings + game2_earnings

theorem carnival_earnings_value : carnival_earnings = 128242 := by
  sorry

end carnival_earnings_value_l224_22437


namespace acute_angle_range_l224_22405

theorem acute_angle_range (α : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α < Real.cos α) : 
  α < π / 4 := by
sorry

end acute_angle_range_l224_22405


namespace quadratic_inequality_solution_l224_22473

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 2*x + c < 0 ↔ -1/3 < x ∧ x < 1/2) → 
  a = 12 ∧ c = -2 := by
  sorry

end quadratic_inequality_solution_l224_22473


namespace polygon_area_bound_l224_22417

/-- A polygon with n vertices -/
structure Polygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

/-- The area of a polygon -/
def area (P : Polygon) : ℝ := sorry

/-- The length of a line segment between two points -/
def distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Area of a polygon with constrained sides and diagonals -/
theorem polygon_area_bound (P : Polygon) 
  (h1 : ∀ (i j : Fin P.n), distance (P.vertices i) (P.vertices j) ≤ 1) : 
  area P < Real.sqrt 3 / 2 := by sorry

end polygon_area_bound_l224_22417


namespace simple_interest_problem_l224_22490

/-- Given a principal sum and an interest rate, if increasing the rate by 5% over 10 years
    results in Rs. 600 more interest, then the principal sum must be Rs. 1200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 600 →
  P = 1200 := by
sorry

end simple_interest_problem_l224_22490


namespace product_lcm_hcf_relation_l224_22467

theorem product_lcm_hcf_relation (a b : ℕ+) 
  (h_product : a * b = 571536)
  (h_lcm : Nat.lcm a b = 31096) :
  Nat.gcd a b = 18 := by
  sorry

end product_lcm_hcf_relation_l224_22467


namespace q_of_q_of_q_2000_pow_2000_l224_22489

/-- Sum of digits of a natural number -/
def q (n : ℕ) : ℕ := sorry

/-- Theorem stating that q(q(q(2000^2000))) = 4 -/
theorem q_of_q_of_q_2000_pow_2000 : q (q (q (2000^2000))) = 4 := by sorry

end q_of_q_of_q_2000_pow_2000_l224_22489


namespace percent_relation_l224_22471

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
sorry

end percent_relation_l224_22471


namespace roses_in_vase_l224_22464

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of roses is 18 given the initial and added amounts -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end roses_in_vase_l224_22464


namespace kopeck_payment_l224_22422

theorem kopeck_payment (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b := by
  sorry

end kopeck_payment_l224_22422


namespace same_color_sock_pairs_l224_22453

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 3
def num_blue_socks : Nat := 2
def num_red_socks : Nat := 2

def total_socks : Nat := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs : 
  choose num_white_socks 2 + choose num_brown_socks 2 + choose num_blue_socks 2 + choose num_red_socks 2 = 15 := by
  sorry

end same_color_sock_pairs_l224_22453


namespace price_reduction_effect_l224_22428

theorem price_reduction_effect (P S : ℝ) (P_reduced : ℝ) (S_increased : ℝ) :
  P_reduced = 0.8 * P →
  S_increased = 1.8 * S →
  P_reduced * S_increased = 1.44 * P * S :=
by sorry

end price_reduction_effect_l224_22428


namespace prove_a_equals_two_l224_22483

/-- Given two differentiable functions f and g on ℝ, prove that a = 2 -/
theorem prove_a_equals_two
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_g_nonzero : ∀ x, g x ≠ 0)
  (h_f_def : ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x)
  (h_inequality : ∀ x, f x * (deriv g x) < (deriv f x) * g x)
  (h_sum : f 1 / g 1 + f (-1) / g (-1) = 5) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x :=
sorry

end prove_a_equals_two_l224_22483


namespace car_speed_problem_l224_22414

/-- Proves that given the conditions, car R's average speed is 50 miles per hour -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 800 →
  time_diff = 2 →
  speed_diff = 10 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_diff = distance / (speed_R + speed_diff) ∧
    speed_R = 50 := by
  sorry

end car_speed_problem_l224_22414


namespace largest_number_l224_22472

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.998) 
  (hb : b = 0.989) 
  (hc : c = 0.999) 
  (hd : d = 0.990) 
  (he : e = 0.980) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end largest_number_l224_22472


namespace wrapping_paper_ratio_l224_22404

theorem wrapping_paper_ratio : 
  ∀ (p1 p2 p3 : ℝ),
  p1 = 2 →
  p3 = p1 + p2 →
  p1 + p2 + p3 = 7 →
  p2 / p1 = 3 / 4 :=
by
  sorry

end wrapping_paper_ratio_l224_22404


namespace absolute_value_sum_zero_implies_value_l224_22443

theorem absolute_value_sum_zero_implies_value (x y : ℝ) :
  |x - 4| + |5 + y| = 0 → 2*x + 3*y = -7 := by
  sorry

end absolute_value_sum_zero_implies_value_l224_22443


namespace natural_number_equation_solutions_l224_22451

theorem natural_number_equation_solutions :
  ∀ (a b c d : ℕ), 
    a * b = c + d ∧ a + b = c * d →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) ∨
     (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∧
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5)) :=
by sorry

end natural_number_equation_solutions_l224_22451


namespace ellipse_focus_coincides_with_center_l224_22481

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Returns the focus with larger x-coordinate for an ellipse -/
def focus_with_larger_x (e : Ellipse) : Point :=
  e.center

theorem ellipse_focus_coincides_with_center (e : Ellipse) 
    (h1 : e.center = ⟨3, -2⟩)
    (h2 : e.semi_major_axis = 3)
    (h3 : e.semi_minor_axis = 3) :
  focus_with_larger_x e = ⟨3, -2⟩ := by
  sorry

#check ellipse_focus_coincides_with_center

end ellipse_focus_coincides_with_center_l224_22481


namespace max_projection_area_parallelepiped_l224_22487

/-- The maximum area of the orthogonal projection of a rectangular parallelepiped -/
theorem max_projection_area_parallelepiped (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : ℝ), S = a * Real.sqrt (a^2 + b^2) ∧
  ∀ (S' : ℝ), S' ≤ S :=
sorry

end max_projection_area_parallelepiped_l224_22487


namespace smallest_yellow_marbles_l224_22418

theorem smallest_yellow_marbles (n : ℕ) (h1 : n > 0) 
  (h2 : n % 2 = 0) (h3 : n % 3 = 0) (h4 : 4 ≤ n) : 
  ∃ (y : ℕ), y = n - (n / 2 + n / 3 + 4) ∧ 
  (∀ (m : ℕ), m > 0 → m % 2 = 0 → m % 3 = 0 → 4 ≤ m → 
    m - (m / 2 + m / 3 + 4) ≥ 0 → n ≤ m) :=
by sorry

end smallest_yellow_marbles_l224_22418


namespace arithmetic_mean_of_fractions_l224_22420

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) (hxa : x^2 ≠ a) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end arithmetic_mean_of_fractions_l224_22420


namespace boat_travel_time_difference_l224_22421

def distance : ℝ := 90
def downstream_time : ℝ := 2.5191640969412834

theorem boat_travel_time_difference (v : ℝ) : 
  v > 3 →
  distance / (v - 3) - distance / (v + 3) = downstream_time →
  ∃ (diff : ℝ), abs (diff - 0.5088359030587166) < 1e-10 ∧ 
                 distance / (v - 3) - downstream_time = diff :=
by sorry

end boat_travel_time_difference_l224_22421


namespace function_properties_l224_22462

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + a

-- Define the theorem
theorem function_properties (a : ℝ) :
  -- Condition: x ∈ [-π/6, π/3]
  (∀ x, -π/6 ≤ x ∧ x ≤ π/3 →
    -- Condition: sum of max and min values is 3/2
    (⨆ x, f x a) + (⨅ x, f x a) = 3/2) →
  -- 1. Smallest positive period is π
  (∀ x, f (x + π) a = f x a) ∧
  (∀ T, T > 0 ∧ (∀ x, f (x + T) a = f x a) → T ≥ π) ∧
  -- 2. Interval of monotonic decrease
  (∀ k : ℤ, ∀ x y, k * π + π/6 ≤ x ∧ x ≤ y ∧ y ≤ k * π + 2*π/3 →
    f y a ≤ f x a) ∧
  -- 3. Solution set of f(x) > 1
  (∀ x, 0 < x ∧ x < π/3 ↔ f x a > 1) :=
sorry

end function_properties_l224_22462


namespace f_has_three_zeros_l224_22488

/-- The function f(x) = x^3 - bx^2 - 4 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 - 4

/-- The theorem stating that f has three distinct real zeros iff b < -3 -/
theorem f_has_three_zeros (b : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f b x = 0 ∧ f b y = 0 ∧ f b z = 0) ↔ 
  b < -3 := by sorry

end f_has_three_zeros_l224_22488


namespace two_true_propositions_l224_22423

theorem two_true_propositions : 
  let original := ∀ a : ℝ, a > 2 → a > 1
  let converse := ∀ a : ℝ, a > 1 → a > 2
  let inverse := ∀ a : ℝ, a ≤ 2 → a ≤ 1
  let contrapositive := ∀ a : ℝ, a ≤ 1 → a ≤ 2
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end two_true_propositions_l224_22423


namespace fibonacci_gcd_l224_22433

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end fibonacci_gcd_l224_22433


namespace probability_two_females_selected_l224_22434

/-- The probability of selecting 2 females out of 6 finalists (4 females and 2 males) -/
theorem probability_two_females_selected (total : Nat) (females : Nat) (selected : Nat) 
  (h1 : total = 6) 
  (h2 : females = 4)
  (h3 : selected = 2) : 
  (Nat.choose females selected : ℚ) / (Nat.choose total selected) = 2 / 5 := by
  sorry

end probability_two_females_selected_l224_22434


namespace perpendicular_vectors_k_value_l224_22497

/-- Given plane vectors a and b, if ka + b is perpendicular to a, then k = -1/5 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) :
  k = -1/5 := by
  sorry

end perpendicular_vectors_k_value_l224_22497


namespace mean_temperature_l224_22492

def temperatures : List ℤ := [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -4/5 := by
  sorry

end mean_temperature_l224_22492


namespace max_teams_is_six_l224_22413

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := n.choose 2 * games_between_teams

/-- The theorem stating that 6 is the maximum number of teams that can participate -/
theorem max_teams_is_six :
  ∀ n : ℕ, n > 6 → total_games n > max_games ∧
  total_games 6 ≤ max_games :=
sorry

end max_teams_is_six_l224_22413


namespace mean_home_runs_l224_22415

def home_runs : List (Nat × Nat) := [(5, 5), (9, 3), (7, 4), (11, 2)]

theorem mean_home_runs :
  let total_home_runs := (home_runs.map (λ (hr, players) => hr * players)).sum
  let total_players := (home_runs.map (λ (_, players) => players)).sum
  (total_home_runs : ℚ) / total_players = 729/100 := by sorry

end mean_home_runs_l224_22415


namespace customers_who_left_l224_22444

theorem customers_who_left (initial_customers : ℕ) (new_customers : ℕ) (final_customers : ℕ) :
  initial_customers = 19 →
  new_customers = 36 →
  final_customers = 41 →
  initial_customers - (initial_customers - new_customers - final_customers) + new_customers = final_customers :=
by
  sorry

end customers_who_left_l224_22444


namespace football_team_analysis_l224_22476

/-- Represents a football team's performance in a season -/
structure FootballTeam where
  total_matches : ℕ
  played_matches : ℕ
  lost_matches : ℕ
  current_points : ℕ

/-- Calculates the number of wins given the team's performance -/
def wins (team : FootballTeam) : ℕ :=
  (team.current_points - (team.played_matches - team.lost_matches)) / 2

/-- Calculates the maximum possible points after all matches -/
def max_points (team : FootballTeam) : ℕ :=
  team.current_points + (team.total_matches - team.played_matches) * 3

/-- Calculates the minimum number of wins needed to reach a goal -/
def min_wins_needed (team : FootballTeam) (goal : ℕ) : ℕ :=
  ((goal - team.current_points) + 2) / 3

theorem football_team_analysis (team : FootballTeam)
  (h1 : team.total_matches = 16)
  (h2 : team.played_matches = 9)
  (h3 : team.lost_matches = 2)
  (h4 : team.current_points = 19) :
  wins team = 6 ∧
  max_points team = 40 ∧
  min_wins_needed team 34 = 4 := by
  sorry

#eval wins { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 }
#eval max_points { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 }
#eval min_wins_needed { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 } 34

end football_team_analysis_l224_22476


namespace complex_power_2015_l224_22416

/-- Given a complex number i such that i^2 = -1, i^3 = -i, and i^4 = 1,
    prove that i^2015 = -i -/
theorem complex_power_2015 (i : ℂ) (hi2 : i^2 = -1) (hi3 : i^3 = -i) (hi4 : i^4 = 1) :
  i^2015 = -i := by sorry

end complex_power_2015_l224_22416


namespace boat_speed_in_still_water_l224_22470

/-- The speed of a boat in still water, given that the time taken to row upstream
    is twice the time taken to row downstream, and the speed of the stream is 12 kmph. -/
theorem boat_speed_in_still_water : ∃ (V_b : ℝ),
  (∀ (t : ℝ), t > 0 → (V_b + 12) * t = (V_b - 12) * (2 * t)) ∧ V_b = 36 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l224_22470


namespace solve_equation_l224_22477

theorem solve_equation (x y : ℝ) (h1 : x = 12) (h2 : ((17.28 / x) / (y * 0.2)) = 2) : y = 3.6 := by
  sorry

end solve_equation_l224_22477


namespace virginia_average_rainfall_l224_22484

/-- The average rainfall in Virginia over five months --/
def average_rainfall (march april may june july : Float) : Float :=
  (march + april + may + june + july) / 5

/-- Theorem stating that the average rainfall in Virginia is 4 inches --/
theorem virginia_average_rainfall :
  average_rainfall 3.79 4.5 3.95 3.09 4.67 = 4 := by
  sorry

end virginia_average_rainfall_l224_22484


namespace multiply_decimals_l224_22440

theorem multiply_decimals : 0.9 * 0.007 = 0.0063 := by
  sorry

end multiply_decimals_l224_22440


namespace cube_volume_from_surface_area_l224_22435

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end cube_volume_from_surface_area_l224_22435


namespace vincent_stickers_l224_22458

theorem vincent_stickers (yesterday : ℕ) (extra_today : ℕ) : 
  yesterday = 15 → extra_today = 10 → yesterday + (yesterday + extra_today) = 40 := by
  sorry

end vincent_stickers_l224_22458


namespace f_monotone_and_inequality_l224_22480

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - Real.log (x + 1) + Real.log (x - 1)

theorem f_monotone_and_inequality (k : ℝ) (h₁ : -1 ≤ k) (h₂ : k ≤ 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  (∀ x > 1, x * (f x₁ + f x₂) ≥ (x + 1) * (f x + 2 - 2*x)) :=
by sorry

end f_monotone_and_inequality_l224_22480


namespace inverse_sum_product_l224_22456

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3*x + 4*y ≠ 0) :
  (3*x + 4*y)⁻¹ * ((3*x)⁻¹ + (4*y)⁻¹) = (12*x*y)⁻¹ :=
by sorry

end inverse_sum_product_l224_22456


namespace expression_evaluation_l224_22461

theorem expression_evaluation (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  let expr := (((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2) / (x^5 + 1)^2)^2 *
               (((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2) / (x^5 - 1)^2)^2
  expr = 1 := by
  sorry

end expression_evaluation_l224_22461


namespace chord_convex_quadrilateral_probability_l224_22446

/-- Given six points on a circle, the probability that four randomly chosen chords
    form a convex quadrilateral is 1/91. -/
theorem chord_convex_quadrilateral_probability (n : ℕ) (h : n = 6) :
  (Nat.choose n 4 : ℚ) / (Nat.choose (Nat.choose n 2) 4) = 1 / 91 :=
sorry

end chord_convex_quadrilateral_probability_l224_22446


namespace complement_intersection_theorem_l224_22403

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end complement_intersection_theorem_l224_22403


namespace arctan_sum_equals_pi_over_four_l224_22401

theorem arctan_sum_equals_pi_over_four (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/n) = π/4 →
  n = 47 := by
  sorry

end arctan_sum_equals_pi_over_four_l224_22401


namespace french_toast_slices_l224_22494

/- Define the problem parameters -/
def weeks_per_year : ℕ := 52
def days_per_week : ℕ := 2
def loaves_used : ℕ := 26
def slices_per_loaf : ℕ := 12
def slices_for_daughters : ℕ := 1

/- Define the function to calculate slices per person -/
def slices_per_person : ℚ :=
  let total_slices := loaves_used * slices_per_loaf
  let total_days := weeks_per_year * days_per_week
  let slices_per_day := total_slices / total_days
  let slices_for_parents := slices_per_day - slices_for_daughters
  slices_for_parents / 2

/- State the theorem -/
theorem french_toast_slices :
  slices_per_person = 1 := by sorry

end french_toast_slices_l224_22494


namespace quadratic_root_difference_l224_22482

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ (x - y)^2 = 9) →
  p = Real.sqrt (4*q + 9) :=
sorry

end quadratic_root_difference_l224_22482


namespace quadratic_sum_of_coefficients_l224_22448

/-- A quadratic function passing through (1,0) and (-3,0) with minimum value 25 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_one : a + b + c = 0
  passes_through_neg_three : 9*a - 3*b + c = 0
  has_minimum_25 : ∀ x, a*x^2 + b*x + c ≥ 25

/-- The sum of coefficients a + b + c equals -75/4 for the given quadratic function -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = -75/4 := by
  sorry

end quadratic_sum_of_coefficients_l224_22448


namespace cannot_row_against_fast_stream_l224_22460

/-- A man rowing a boat in a stream -/
structure Rower where
  speedWithStream : ℝ
  speedInStillWater : ℝ

/-- Determine if a rower can go against the stream -/
def canRowAgainstStream (r : Rower) : Prop :=
  r.speedInStillWater > r.speedWithStream - r.speedInStillWater

/-- Theorem: A man cannot row against the stream if his speed in still water
    is less than the stream's speed -/
theorem cannot_row_against_fast_stream (r : Rower)
  (h1 : r.speedWithStream = 10)
  (h2 : r.speedInStillWater = 2) :
  ¬(canRowAgainstStream r) := by
  sorry

#check cannot_row_against_fast_stream

end cannot_row_against_fast_stream_l224_22460


namespace quadratic_inequality_solutions_quadratic_inequality_always_negative_l224_22474

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Define the solution set for the first case
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

-- Define the solution set for the second case
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0 ↔ solution_set_1 x) → k = -2/5 :=
sorry

theorem quadratic_inequality_always_negative (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0) → k < -Real.sqrt 6 / 6 :=
sorry

end quadratic_inequality_solutions_quadratic_inequality_always_negative_l224_22474


namespace parallel_lines_a_equals_three_l224_22430

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation: ax + 3y + 4 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 4 = 0

/-- The second line equation: x + (a-2)y + a^2 - 5 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a^2 - 5 = 0

/-- Theorem: If the two lines are parallel, then a = 3 -/
theorem parallel_lines_a_equals_three :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
sorry

end parallel_lines_a_equals_three_l224_22430


namespace sum_of_remaining_segments_l224_22459

/-- Represents a rectangular figure with some interior segments -/
structure RectFigure where
  left : ℝ
  right : ℝ
  bottomLeft : ℝ
  topLeft : ℝ
  topRight : ℝ

/-- Calculates the sum of remaining segments after removing four sides -/
def remainingSum (f : RectFigure) : ℝ :=
  f.left + f.right + (f.bottomLeft + f.topLeft + f.topRight) + f.topRight

/-- Theorem stating that for the given measurements, the sum of remaining segments is 23 -/
theorem sum_of_remaining_segments :
  let f : RectFigure := {
    left := 10,
    right := 7,
    bottomLeft := 3,
    topLeft := 1,
    topRight := 1
  }
  remainingSum f = 23 := by sorry

end sum_of_remaining_segments_l224_22459


namespace car_speed_problem_l224_22468

theorem car_speed_problem (S : ℝ) : 
  (S * 1.3 + 10 = 205) → S = 150 := by
  sorry

end car_speed_problem_l224_22468


namespace minimum_days_to_exceed_500_l224_22411

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem minimum_days_to_exceed_500 :
  ∃ (n : ℕ), n = 6 ∧
  (∀ (k : ℕ), k < n → bacteria_count 4 3 k ≤ 500) ∧
  bacteria_count 4 3 n > 500 :=
sorry

end minimum_days_to_exceed_500_l224_22411


namespace sphere_tangent_plane_distance_l224_22432

/-- Given three spheres where two smaller spheres touch each other externally and
    each touches a larger sphere internally, with radii as specified,
    the distance from the center of the largest sphere to the tangent plane
    at the touching point of the smaller spheres is R/5. -/
theorem sphere_tangent_plane_distance (R : ℝ) : ℝ := by
  -- Define the radii of the smaller spheres
  let r₁ := R / 2
  let r₂ := R / 3
  
  -- Define the distance from the center of the largest sphere
  -- to the tangent plane at the touching point of the smaller spheres
  let d : ℝ := R / 5
  
  -- The proof would go here
  sorry

#check sphere_tangent_plane_distance

end sphere_tangent_plane_distance_l224_22432


namespace factor_implies_m_value_l224_22402

theorem factor_implies_m_value (m : ℤ) : 
  (∃ a : ℤ, ∀ x : ℤ, x^2 - m*x - 15 = (x + 3) * (x - a)) → m = 2 :=
by
  sorry

end factor_implies_m_value_l224_22402


namespace ball_placement_theorem_l224_22442

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process up to n steps --/
def ballPlacement (n : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem :
  ballPlacement 1729 = sumDigits (toBase7 1729) :=
sorry

end ball_placement_theorem_l224_22442


namespace trapezoid_reconstruction_l224_22400

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Checks if two line segments are parallel -/
def parallel (p q r s : Point) : Prop :=
  (q.y - p.y) * (s.x - r.x) = (s.y - r.y) * (q.x - p.x)

/-- Checks if a point divides two line segments proportionally -/
def divides_proportionally (o p q r s : Point) : Prop :=
  (o.x - p.x) * (s.y - q.y) = (o.y - p.y) * (s.x - q.x)

/-- Theorem: Given three points A, B, C, and a point O, 
    there exists a point D such that ABCD forms a trapezoid 
    with O as the intersection of its diagonals -/
theorem trapezoid_reconstruction 
  (A B C O : Point) 
  (h1 : collinear A O C) 
  (h2 : ¬ collinear A B C) : 
  ∃ D : Point, 
    parallel A B C D ∧ 
    collinear B O D ∧
    divides_proportionally O A C B D :=
sorry

end trapezoid_reconstruction_l224_22400


namespace polynomial_composition_l224_22445

theorem polynomial_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2) →
  (∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) := by
sorry

end polynomial_composition_l224_22445


namespace cone_lateral_surface_angle_l224_22406

/-- Given a cone with base radius 5 and slant height 15, prove that the central angle of the sector in the unfolded lateral surface is 120 degrees -/
theorem cone_lateral_surface_angle (base_radius : ℝ) (slant_height : ℝ) (central_angle : ℝ) : 
  base_radius = 5 → 
  slant_height = 15 → 
  central_angle * slant_height / 180 * π = 2 * π * base_radius → 
  central_angle = 120 := by
sorry

end cone_lateral_surface_angle_l224_22406


namespace sarah_cupcake_ratio_l224_22408

theorem sarah_cupcake_ratio :
  ∀ (michael_cookies sarah_initial_cupcakes sarah_final_desserts : ℕ)
    (sarah_saved_cupcakes : ℕ),
  michael_cookies = 5 →
  sarah_initial_cupcakes = 9 →
  sarah_final_desserts = 11 →
  sarah_final_desserts = sarah_initial_cupcakes - sarah_saved_cupcakes + michael_cookies →
  (sarah_saved_cupcakes : ℚ) / sarah_initial_cupcakes = 1 / 3 :=
by
  sorry

end sarah_cupcake_ratio_l224_22408


namespace root_condition_l224_22407

open Real

theorem root_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) →
  a ≤ -1 ∧
  (∃ a : ℝ, a ≤ -1 ∧ ∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) ∧
  (∃ a : ℝ, a > -1 ∧ ∀ x : ℝ, x ≥ (exp 1) → a + log x ≠ 0) :=
by sorry

end root_condition_l224_22407


namespace pine_tree_branches_l224_22409

/-- The number of branches in a pine tree -/
def num_branches : ℕ := 23

/-- The movements of the squirrel from the middle branch to the top -/
def movements : List ℤ := [5, -7, 4, 9]

/-- The number of branches from the middle to the top -/
def branches_to_top : ℕ := (movements.sum).toNat

theorem pine_tree_branches :
  num_branches = 2 * branches_to_top + 1 :=
by sorry

end pine_tree_branches_l224_22409


namespace sports_club_membership_l224_22495

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) : 
  total = 30 → badminton = 17 → tennis = 19 → both = 9 →
  total - (badminton + tennis - both) = 3 := by
  sorry

end sports_club_membership_l224_22495


namespace gas_volume_ranking_l224_22427

/-- Gas volume per capita for a region -/
structure GasVolume where
  region : String
  volume : Float

/-- Theorem: Russia has the highest gas volume per capita, followed by Non-West, then West -/
theorem gas_volume_ranking (west non_west russia : GasVolume) 
  (h_west : west.region = "West" ∧ west.volume = 21428)
  (h_non_west : non_west.region = "Non-West" ∧ non_west.volume = 26848.55)
  (h_russia : russia.region = "Russia" ∧ russia.volume = 302790.13) :
  russia.volume > non_west.volume ∧ non_west.volume > west.volume :=
by sorry

end gas_volume_ranking_l224_22427


namespace pizza_sharing_ratio_l224_22438

theorem pizza_sharing_ratio (total_slices : ℕ) (waiter_slices : ℕ) : 
  total_slices = 78 → 
  waiter_slices - 20 = 28 → 
  (total_slices - waiter_slices) / waiter_slices = 5 / 8 := by
  sorry

end pizza_sharing_ratio_l224_22438


namespace greenwood_school_quiz_l224_22466

theorem greenwood_school_quiz (f s : ℕ) (h1 : f > 0) (h2 : s > 0) :
  (3 * f : ℚ) / 4 = (s : ℚ) / 3 → s = 3 * f := by
  sorry

end greenwood_school_quiz_l224_22466


namespace gcd_of_powers_of_101_l224_22485

theorem gcd_of_powers_of_101 : 
  Nat.Prime 101 → Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_101_l224_22485


namespace solve_equation_1_solve_equation_2_l224_22496

-- Equation 1: x^2 - 6x + 1 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 + 2 * Real.sqrt 2 ∧ 
             x₂ = 3 - 2 * Real.sqrt 2 ∧ 
             x₁^2 - 6*x₁ + 1 = 0 ∧ 
             x₂^2 - 6*x₂ + 1 = 0 := by
  sorry

-- Equation 2: 2x^2 + 3x - 5 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ 
             x₂ = -5/2 ∧ 
             2*x₁^2 + 3*x₁ - 5 = 0 ∧ 
             2*x₂^2 + 3*x₂ - 5 = 0 := by
  sorry

end solve_equation_1_solve_equation_2_l224_22496


namespace half_angle_quadrant_l224_22469

-- Define the concept of an angle being in a specific quadrant
def in_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < α ∧ α < k * 360 + 180

def in_first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 < α ∧ α < n * 360 + 90

def in_third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 180 < α ∧ α < n * 360 + 270

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_second_quadrant α → (in_first_quadrant (α/2) ∨ in_third_quadrant (α/2)) :=
by sorry

end half_angle_quadrant_l224_22469


namespace triangle_side_less_than_semiperimeter_l224_22498

theorem triangle_side_less_than_semiperimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end triangle_side_less_than_semiperimeter_l224_22498


namespace total_files_deleted_l224_22450

def initial_files : ℕ := 24
def final_files : ℕ := 21

def deletions : List ℕ := [5, 10]
def additions : List ℕ := [7, 5]

theorem total_files_deleted :
  (initial_files + additions.sum - deletions.sum = final_files) →
  deletions.sum = 15 := by
  sorry

end total_files_deleted_l224_22450


namespace folded_square_perimeter_ratio_l224_22463

theorem folded_square_perimeter_ratio :
  let square_side : ℝ := 10
  let folded_width : ℝ := square_side / 2
  let folded_height : ℝ := square_side
  let triangle_perimeter : ℝ := folded_width + folded_height + Real.sqrt (folded_width ^ 2 + folded_height ^ 2)
  let pentagon_perimeter : ℝ := 2 * folded_height + folded_width + Real.sqrt (folded_width ^ 2 + folded_height ^ 2) + folded_width
  triangle_perimeter / pentagon_perimeter = (3 + Real.sqrt 5) / (6 + Real.sqrt 5) := by
  sorry

end folded_square_perimeter_ratio_l224_22463


namespace rectangle_coverage_l224_22457

/-- A shape composed of 6 unit squares -/
structure Shape :=
  (area : ℕ)
  (h_area : area = 6)

/-- A rectangle with dimensions m × n -/
structure Rectangle (m n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (h_width : width = m)
  (h_height : height = n)

/-- Predicate for a rectangle that can be covered by shapes -/
def is_coverable (m n : ℕ) : Prop :=
  (3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ m ∧ 12 ∣ n)

theorem rectangle_coverage (m n : ℕ) (hm : m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) (hn : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) :
  ∃ (s : Shape), ∃ (r : Rectangle m n), is_coverable m n ↔ 
    (∃ (arrangement : ℕ → ℕ → Shape), 
      (∀ i j, i < m ∧ j < n → (arrangement i j).area = 6) ∧
      (∀ i j, i < m ∧ j < n → ∃ k l, k < m ∧ l < n ∧ arrangement i j = arrangement k l) ∧
      (∀ i j k l, i < m ∧ j < n ∧ k < m ∧ l < n → 
        (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)) :=
by sorry

end rectangle_coverage_l224_22457


namespace melanie_dimes_l224_22419

/-- Calculates the total number of dimes Melanie has after receiving dimes from her parents. -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proves that Melanie has 19 dimes in total. -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end melanie_dimes_l224_22419


namespace problem_1_problem_2_l224_22429

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (-2 * a^2)^2 * (-b^2) / (4 * a^3 * b^2) = -a := by sorry

-- Problem 2
theorem problem_2 : 2023^2 - 2021 * 2025 = 4 := by sorry

end problem_1_problem_2_l224_22429


namespace males_in_band_not_orchestra_l224_22439

/-- Represents the number of students in a group -/
structure GroupCount where
  female : ℕ
  male : ℕ

/-- Represents the counts for band, orchestra, and choir -/
structure MusicGroups where
  band : GroupCount
  orchestra : GroupCount
  choir : GroupCount
  all_three : GroupCount
  total : ℕ

def music_groups : MusicGroups := {
  band := { female := 120, male := 90 },
  orchestra := { female := 90, male := 120 },
  choir := { female := 50, male := 40 },
  all_three := { female := 30, male := 20 },
  total := 250
}

theorem males_in_band_not_orchestra (g : MusicGroups) (h : g = music_groups) :
  g.band.male - (g.band.male + g.orchestra.male + g.choir.male - g.total) = 20 := by
  sorry

end males_in_band_not_orchestra_l224_22439


namespace intersection_A_B_l224_22455

def set_A : Set ℝ := {x | x^2 - 11*x - 12 < 0}

def set_B : Set ℝ := {x | ∃ n : ℤ, x = 3*n + 1}

theorem intersection_A_B :
  set_A ∩ set_B = {1, 4, 7, 10} := by sorry

end intersection_A_B_l224_22455


namespace sum_remainder_theorem_l224_22465

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (fun i => a₁ + i * d)

theorem sum_remainder_theorem (a₁ d aₙ : ℕ) (h₁ : a₁ = 3) (h₂ : d = 8) (h₃ : aₙ = 283) :
  (arithmetic_sequence a₁ d aₙ).sum % 8 = 4 := by
  sorry

end sum_remainder_theorem_l224_22465


namespace horner_method_f_neg_four_l224_22431

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 - 8x^2 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℤ) : ℤ := 12 - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_f_neg_four :
  horner_eval [3, 5, 6, 0, -8, 0, 12] (-4) = f (-4) ∧ f (-4) = -845 := by
  sorry

end horner_method_f_neg_four_l224_22431


namespace complex_modulus_sqrt_two_l224_22479

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
sorry

end complex_modulus_sqrt_two_l224_22479


namespace final_sum_is_correct_l224_22441

/-- Represents the state of the three calculators -/
structure CalculatorState where
  calc1 : ℤ
  calc2 : ℤ
  calc3 : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { calc1 := 2 * state.calc1,
    calc2 := state.calc2 ^ 2,
    calc3 := -state.calc3 }

/-- Iterates the operations n times -/
def iterateOperations (n : ℕ) (state : CalculatorState) : CalculatorState :=
  match n with
  | 0 => state
  | n + 1 => applyOperations (iterateOperations n state)

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2, calc2 := 0, calc3 := -2 }

/-- The main theorem to prove -/
theorem final_sum_is_correct :
  let finalState := iterateOperations 51 initialState
  finalState.calc1 + finalState.calc2 + finalState.calc3 = 2^52 + 2 := by
  sorry

end final_sum_is_correct_l224_22441


namespace triangle_area_circumradius_l224_22478

theorem triangle_area_circumradius (a b c R : ℝ) (α β γ : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  c / Real.sin γ = 2 * R →
  S = 1/2 * a * b * Real.sin γ →
  S = a * b * c / (4 * R) := by
sorry

end triangle_area_circumradius_l224_22478


namespace set_operations_l224_22425

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem set_operations :
  (A ∪ B = {x | x ≥ 3}) ∧
  (A ∩ B = {x | 4 ≤ x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) ∩ (A ∪ B) = {x | (3 ≤ x ∧ x < 4) ∨ x ≥ 10}) := by
  sorry

end set_operations_l224_22425


namespace sin_cos_equation_solvability_l224_22493

theorem sin_cos_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := by
  sorry

end sin_cos_equation_solvability_l224_22493


namespace min_value_theorem_l224_22426

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    a / (Real.sin θ)^(3/2) + b / (Real.cos θ)^(3/2) ≥ (a^(4/7) + b^(4/7))^(7/4) := by
  sorry

end min_value_theorem_l224_22426


namespace sum_of_powers_of_two_l224_22491

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 := by
  sorry

end sum_of_powers_of_two_l224_22491


namespace parallel_vectors_x_value_l224_22475

/-- Given two parallel vectors a and b in R², prove that if a = (x, 3) and b = (4, 6), then x = 2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 3)
  let b : ℝ × ℝ := (4, 6)
  (∃ (k : ℝ), a = k • b) → x = 2 := by
sorry

end parallel_vectors_x_value_l224_22475
