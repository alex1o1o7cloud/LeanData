import Mathlib

namespace dennis_floor_l2569_256943

/-- Given the floor arrangements of Frank, Charlie, and Dennis, prove that Dennis lives on the 6th floor. -/
theorem dennis_floor : 
  ∀ (frank_floor charlie_floor dennis_floor : ℕ),
  frank_floor = 16 →
  charlie_floor = frank_floor / 4 →
  dennis_floor = charlie_floor + 2 →
  dennis_floor = 6 := by
sorry

end dennis_floor_l2569_256943


namespace constant_dot_product_implies_ratio_l2569_256984

/-- Given that O is the origin, P is any point on the line 2x + y - 2 = 0,
    a = (m, n) is a non-zero vector, and the dot product of OP and a is always constant,
    then m/n = 2. -/
theorem constant_dot_product_implies_ratio (m n : ℝ) :
  (∀ x y : ℝ, 2 * x + y - 2 = 0 →
    ∃ k : ℝ, ∀ x' y' : ℝ, 2 * x' + y' - 2 = 0 →
      m * x' + n * y' = k) →
  m ≠ 0 ∨ n ≠ 0 →
  m / n = 2 :=
by sorry

end constant_dot_product_implies_ratio_l2569_256984


namespace intersection_complement_equal_l2569_256942

-- Define the universe set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 5, 7}

-- Theorem statement
theorem intersection_complement_equal : S ∩ (U \ T) = {1, 2, 4} := by
  sorry

end intersection_complement_equal_l2569_256942


namespace andrew_kept_130_stickers_l2569_256916

def total_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def andrew_kept_stickers : ℕ := total_stickers - (daniel_stickers + (daniel_stickers + fred_extra_stickers))

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end andrew_kept_130_stickers_l2569_256916


namespace sock_selection_combinations_l2569_256968

theorem sock_selection_combinations : Nat.choose 7 4 = 35 := by
  sorry

end sock_selection_combinations_l2569_256968


namespace aaron_used_three_boxes_l2569_256915

/-- Given the initial number of can lids, final number of can lids, and lids per box,
    calculate the number of boxes used. -/
def boxes_used (initial_lids : ℕ) (final_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  (final_lids - initial_lids) / lids_per_box

/-- Theorem stating that Aaron used 3 boxes of canned tomatoes. -/
theorem aaron_used_three_boxes :
  boxes_used 14 53 13 = 3 := by
  sorry

end aaron_used_three_boxes_l2569_256915


namespace clock_problem_l2569_256973

/-- Represents a clock with hourly chimes -/
structure Clock :=
  (current_hour : Nat)
  (total_chimes : Nat)

/-- Function to calculate the number of chimes for a given hour -/
def chimes_for_hour (h : Nat) : Nat :=
  if h = 0 then 12 else h

/-- Function to check if the hour and minute hands overlap -/
def hands_overlap (h : Nat) : Prop :=
  h ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]

/-- Theorem representing the clock problem -/
theorem clock_problem (c : Clock) (h : Nat) : 
  c.total_chimes = 12 ∧ 
  hands_overlap h ∧ 
  c.current_hour = 3 ∧
  chimes_for_hour 3 + chimes_for_hour 4 + chimes_for_hour 5 = 12 →
  h - c.current_hour = 3 := by
  sorry

#check clock_problem

end clock_problem_l2569_256973


namespace min_value_theorem_l2569_256991

noncomputable def f (x : ℝ) : ℝ := min (3^x - 1) (-x^2 + 2*x + 1)

theorem min_value_theorem (m a b : ℝ) :
  (∀ x, f x ≤ m) ∧  -- m is the maximum value of f
  (∃ x, f x = m) ∧  -- m is attained for some x
  (a > 0) ∧ (b > 0) ∧ (a + 2*b = m) →  -- conditions on a and b
  (∀ a' b', a' > 0 → b' > 0 → a' + 2*b' = m → 
    2 / (a' + 1) + 1 / b' ≥ 8/3) ∧  -- 8/3 is the minimum value
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ a' + 2*b' = m ∧ 
    2 / (a' + 1) + 1 / b' = 8/3)  -- minimum is attained for some a' and b'
  := by sorry

end min_value_theorem_l2569_256991


namespace whitney_money_left_l2569_256938

/-- The amount of money Whitney has left after her purchase at the school book fair --/
def money_left_over : ℕ :=
  let initial_money : ℕ := 2 * 20
  let poster_cost : ℕ := 5
  let notebook_cost : ℕ := 4
  let bookmark_cost : ℕ := 2
  let poster_quantity : ℕ := 2
  let notebook_quantity : ℕ := 3
  let bookmark_quantity : ℕ := 2
  let total_cost : ℕ := poster_cost * poster_quantity + 
                        notebook_cost * notebook_quantity + 
                        bookmark_cost * bookmark_quantity
  initial_money - total_cost

theorem whitney_money_left : money_left_over = 14 := by
  sorry

end whitney_money_left_l2569_256938


namespace max_tuesday_13ths_l2569_256959

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the months of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .February => 28
  | .April | .June | .September | .November => 30
  | _ => 31

/-- Returns the day of the week for the 13th of a given month, 
    given the day of the week for January 13th -/
def dayOf13th (m : Month) (jan13 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Counts the number of times the 13th falls on a Tuesday in a year -/
def countTuesday13ths (jan13 : DayOfWeek) : Nat :=
  sorry

theorem max_tuesday_13ths :
  ∃ (jan13 : DayOfWeek), countTuesday13ths jan13 = 3 ∧
  ∀ (d : DayOfWeek), countTuesday13ths d ≤ 3 :=
  sorry

end max_tuesday_13ths_l2569_256959


namespace shaded_area_theorem_l2569_256983

theorem shaded_area_theorem (square_side : ℝ) (h : square_side = 12) :
  let triangle_base : ℝ := square_side * 3 / 4
  let triangle_height : ℝ := square_side / 4
  triangle_base * triangle_height / 2 = 13.5 := by
  sorry

end shaded_area_theorem_l2569_256983


namespace total_profit_calculation_l2569_256939

/-- Given three partners a, b, and c with their capital investments and profit shares,
    prove that the total profit is 16500. -/
theorem total_profit_calculation (a b c : ℕ) (profit_b : ℕ) :
  (2 * a = 3 * b) →  -- Twice a's capital equals thrice b's capital
  (b = 4 * c) →      -- b's capital is 4 times c's capital
  (profit_b = 6000) →  -- b's share of the profit is 6000
  (∃ (total_profit : ℕ), 
    total_profit = 16500 ∧
    total_profit * 4 = profit_b * 11) := by
  sorry

#check total_profit_calculation

end total_profit_calculation_l2569_256939


namespace product_digit_sum_l2569_256954

theorem product_digit_sum (k : ℕ) : k = 222 ↔ 9 * k = 2000 ∧ ∃! (n : ℕ), 9 * n = 2000 := by
  sorry

end product_digit_sum_l2569_256954


namespace import_tax_problem_l2569_256949

/-- The import tax problem -/
theorem import_tax_problem (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) 
  (h1 : tax_rate = 0.07)
  (h2 : tax_paid = 112.70)
  (h3 : total_value = 2610) :
  ∃ (excess : ℝ), excess = 1000 ∧ tax_rate * (total_value - excess) = tax_paid :=
by sorry

end import_tax_problem_l2569_256949


namespace final_expression_l2569_256957

theorem final_expression (x : ℝ) : ((3 * x + 5) - 5 * x) / 3 = (-2 * x + 5) / 3 := by
  sorry

end final_expression_l2569_256957


namespace complement_intersection_theorem_l2569_256906

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end complement_intersection_theorem_l2569_256906


namespace cans_per_bag_l2569_256900

/-- Given that Paul filled 6 bags on Saturday, 3 bags on Sunday, and collected a total of 72 cans,
    prove that the number of cans in each bag is 8. -/
theorem cans_per_bag (saturday_bags : Nat) (sunday_bags : Nat) (total_cans : Nat) :
  saturday_bags = 6 →
  sunday_bags = 3 →
  total_cans = 72 →
  total_cans / (saturday_bags + sunday_bags) = 8 := by
  sorry

end cans_per_bag_l2569_256900


namespace distance_sum_difference_bound_l2569_256952

-- Define a convex dodecagon
def ConvexDodecagon : Type := Unit

-- Define a point inside the dodecagon
def Point : Type := Unit

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the vertices of the dodecagon
def vertices (d : ConvexDodecagon) : Finset Point := sorry

-- Define the sum of distances from a point to all vertices
def sum_distances (p : Point) (d : ConvexDodecagon) : ℝ :=
  (vertices d).sum (λ v => distance p v)

-- The main theorem
theorem distance_sum_difference_bound
  (d : ConvexDodecagon) (p q : Point)
  (h : distance p q = 10) :
  |sum_distances p d - sum_distances q d| < 100 := by
  sorry

end distance_sum_difference_bound_l2569_256952


namespace simplify_expression_l2569_256961

theorem simplify_expression (x y : ℝ) : 7*x + 9 - 2*x + 3*y = 5*x + 3*y + 9 := by
  sorry

end simplify_expression_l2569_256961


namespace chord_bisector_l2569_256930

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_bisector :
  ellipse P.1 P.2 →
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    P = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    (∀ (x y : ℝ), line_equation x y ↔ ∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2)) :=
sorry

end chord_bisector_l2569_256930


namespace perpendicular_slope_l2569_256969

theorem perpendicular_slope (x y : ℝ) :
  let given_line := {(x, y) | 5 * x - 2 * y = 10}
  let given_slope := 5 / 2
  let perpendicular_slope := -1 / given_slope
  perpendicular_slope = -2 / 5 := by
  sorry

end perpendicular_slope_l2569_256969


namespace total_tires_is_101_l2569_256925

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  car_tires * cars_seen + 
  bicycle_tires * bicycles_seen + 
  pickup_truck_tires * pickup_trucks_seen + 
  tricycle_tires * tricycles_seen

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end total_tires_is_101_l2569_256925


namespace rationalize_and_simplify_sqrt_5_12_l2569_256988

theorem rationalize_and_simplify_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_and_simplify_sqrt_5_12_l2569_256988


namespace unoccupied_volume_of_cube_l2569_256913

/-- The volume of a cube not occupied by five spheres --/
theorem unoccupied_volume_of_cube (π : Real) : 
  let cube_edge : Real := 2
  let sphere_radius : Real := 1
  let cube_volume : Real := cube_edge ^ 3
  let sphere_volume : Real := (4 / 3) * π * sphere_radius ^ 3
  let total_sphere_volume : Real := 5 * sphere_volume
  cube_volume - total_sphere_volume = 8 - (20 / 3) * π := by sorry

end unoccupied_volume_of_cube_l2569_256913


namespace distance_between_points_l2569_256956

/-- The distance between two points A and B given train travel conditions -/
theorem distance_between_points (v_pas v_freight : ℝ) (d : ℝ) : 
  (d / v_freight - d / v_pas = 3.2) →
  (v_pas * (d / v_freight) = d + 288) →
  (d / (v_freight + 10) - d / (v_pas + 10) = 2.4) →
  d = 360 := by
sorry

end distance_between_points_l2569_256956


namespace junior_score_l2569_256928

theorem junior_score (n : ℝ) (junior_proportion : ℝ) (senior_proportion : ℝ) 
  (overall_average : ℝ) (senior_average : ℝ) :
  junior_proportion = 0.3 →
  senior_proportion = 0.7 →
  overall_average = 79 →
  senior_average = 75 →
  junior_proportion + senior_proportion = 1 →
  let junior_score := (overall_average - senior_average * senior_proportion) / junior_proportion
  junior_score = 88 := by
  sorry

end junior_score_l2569_256928


namespace sin_plus_cos_shift_l2569_256981

theorem sin_plus_cos_shift (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end sin_plus_cos_shift_l2569_256981


namespace fourth_term_of_geometric_progression_l2569_256934

theorem fourth_term_of_geometric_progression (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = Real.rpow 2 (1/3) →
  c = Real.rpow 2 (1/6) →
  b / a = c / b →
  c * (b / a) = 1 :=
by sorry

end fourth_term_of_geometric_progression_l2569_256934


namespace keychain_cost_decrease_l2569_256985

theorem keychain_cost_decrease (P : ℝ) : 
  P - P * 0.35 - (P - P * 0.50) = 15 ∧ P - P * 0.50 = 50 → 
  P - P * 0.35 = 65 := by
sorry

end keychain_cost_decrease_l2569_256985


namespace polynomial_inequality_l2569_256920

theorem polynomial_inequality (m : ℚ) : 5 * m^2 - 8 * m + 1 > 4 * m^2 - 8 * m - 1 := by
  sorry

end polynomial_inequality_l2569_256920


namespace cubic_equation_solutions_l2569_256975

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3 + 3 * Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (3 - 3 * Complex.I * Real.sqrt 3) / 2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  ∀ z : ℂ, z^3 = -27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end cubic_equation_solutions_l2569_256975


namespace prob_at_least_one_woman_l2569_256962

/-- The probability of selecting at least one woman when choosing 3 people at random from a group of 5 men and 5 women -/
theorem prob_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total_people = men + women → 
  men = 5 → 
  women = 5 → 
  selected = 3 → 
  (1 : ℚ) - (men.choose selected : ℚ) / (total_people.choose selected : ℚ) = 11 / 12 := by
  sorry

end prob_at_least_one_woman_l2569_256962


namespace group_purchase_equation_l2569_256926

theorem group_purchase_equation (x : ℕ) : 
  (∀ (required : ℕ), 8 * x = required + 3 ∧ 7 * x = required - 4) → 
  8 * x - 3 = 7 * x + 4 := by
sorry

end group_purchase_equation_l2569_256926


namespace probability_of_black_ball_l2569_256998

theorem probability_of_black_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.42) 
  (h2 : p_white = 0.28) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  1 - p_red - p_white = 0.30 := by
sorry

end probability_of_black_ball_l2569_256998


namespace sum_of_first_five_terms_l2569_256967

/-- Coordinate of point P_n on y-axis -/
def a (n : ℕ+) : ℚ := 2 / n

/-- Area of triangle formed by line through P_n and P_{n+1} and coordinate axes -/
def b (n : ℕ+) : ℚ := 4 + 1 / n - 1 / (n + 1)

/-- Sum of first n terms of sequence {b_n} -/
def S (n : ℕ+) : ℚ := 4 * n + n / (n + 1)

/-- Theorem: The sum of the first 5 terms of sequence {b_n} is 125/6 -/
theorem sum_of_first_five_terms : S 5 = 125 / 6 := by sorry

end sum_of_first_five_terms_l2569_256967


namespace midpoint_triangle_area_for_specific_configuration_l2569_256910

/-- Configuration of three congruent circles -/
structure CircleConfiguration where
  radius : ℝ
  passes_through_centers : Prop

/-- Triangle formed by midpoints of arcs -/
structure MidpointTriangle where
  config : CircleConfiguration
  area : ℝ

/-- The main theorem -/
theorem midpoint_triangle_area_for_specific_configuration :
  ∀ (config : CircleConfiguration) (triangle : MidpointTriangle),
    config.radius = 2 ∧
    config.passes_through_centers ∧
    triangle.config = config →
    ∃ (a b : ℕ),
      triangle.area = Real.sqrt 3 ∧
      triangle.area = Real.sqrt a - b ∧
      100 * a + b = 300 := by
  sorry

end midpoint_triangle_area_for_specific_configuration_l2569_256910


namespace expression_simplification_and_evaluation_l2569_256932

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4*x + 4)) = 1/6 := by
  sorry

end expression_simplification_and_evaluation_l2569_256932


namespace sum_even_divisors_140_l2569_256995

/-- Sum of even positive divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all even positive divisors of 140 is 288 -/
theorem sum_even_divisors_140 : sumEvenDivisors 140 = 288 := by sorry

end sum_even_divisors_140_l2569_256995


namespace football_team_analysis_l2569_256974

structure FootballTeam where
  total_matches : Nat
  played_matches : Nat
  lost_matches : Nat
  points : Nat

def win_points : Nat := 3
def draw_points : Nat := 1
def loss_points : Nat := 0

def team : FootballTeam := {
  total_matches := 14,
  played_matches := 8,
  lost_matches := 1,
  points := 17
}

def wins_in_first_8 (t : FootballTeam) : Nat :=
  (t.points - (t.played_matches - t.lost_matches - 1)) / 2

def max_possible_points (t : FootballTeam) : Nat :=
  t.points + (t.total_matches - t.played_matches) * win_points

def min_wins_needed (t : FootballTeam) (target : Nat) : Nat :=
  ((target - t.points + 2) / win_points).min (t.total_matches - t.played_matches)

theorem football_team_analysis (t : FootballTeam) :
  wins_in_first_8 t = 5 ∧
  max_possible_points t = 35 ∧
  min_wins_needed t 29 = 3 := by
  sorry

end football_team_analysis_l2569_256974


namespace negation_of_existence_real_roots_l2569_256901

theorem negation_of_existence_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end negation_of_existence_real_roots_l2569_256901


namespace extrema_sum_implies_a_range_l2569_256966

/-- Given a function f(x) = ax - x^2 - ln x, if f(x) has extrema and the sum of these extrema
    is not less than 4 + ln 2, then a ∈ [2√3, +∞). -/
theorem extrema_sum_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x > 0, a * x - x^2 - Real.log x ≤ max (a * x₁ - x₁^2 - Real.log x₁) (a * x₂ - x₂^2 - Real.log x₂)) ∧
    (a * x₁ - x₁^2 - Real.log x₁) + (a * x₂ - x₂^2 - Real.log x₂) ≥ 4 + Real.log 2) →
  a ≥ 2 * Real.sqrt 3 :=
sorry

end extrema_sum_implies_a_range_l2569_256966


namespace final_sum_after_transformation_l2569_256987

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end final_sum_after_transformation_l2569_256987


namespace number_of_stools_l2569_256927

/-- Represents the number of legs on a stool -/
def stool_legs : ℕ := 3

/-- Represents the number of legs on a chair -/
def chair_legs : ℕ := 4

/-- Represents the total number of legs in the room when people sit on all furniture -/
def total_legs : ℕ := 39

/-- Theorem stating that the number of three-legged stools is 3 -/
theorem number_of_stools (x y : ℕ) 
  (h : stool_legs * x + chair_legs * y = total_legs) : x = 3 := by
  sorry

end number_of_stools_l2569_256927


namespace remainder_31_pow_31_plus_31_mod_32_l2569_256999

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end remainder_31_pow_31_plus_31_mod_32_l2569_256999


namespace consecutive_even_negative_integers_sum_l2569_256989

theorem consecutive_even_negative_integers_sum (n m : ℤ) : 
  n < 0 ∧ m < 0 ∧ 
  Even n ∧ Even m ∧ 
  m = n + 2 ∧ 
  n * m = 2496 → 
  n + m = -102 := by sorry

end consecutive_even_negative_integers_sum_l2569_256989


namespace mass_of_man_in_boat_l2569_256948

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
def mass_of_man (boat_length boat_breadth boat_sinkage water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinkage * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specified conditions. -/
theorem mass_of_man_in_boat : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end mass_of_man_in_boat_l2569_256948


namespace alternating_sum_of_coefficients_l2569_256935

theorem alternating_sum_of_coefficients : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
  (∀ x : ℝ, (1 + 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -32 := by
sorry

end alternating_sum_of_coefficients_l2569_256935


namespace odd_function_sum_l2569_256960

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 = -2) :
  f (-3) + f 0 = 2 := by
  sorry

end odd_function_sum_l2569_256960


namespace nilpotent_is_zero_fourth_power_eq_self_l2569_256905

class SpecialRing (A : Type*) extends Ring A where
  special_property : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6

variable {A : Type*} [SpecialRing A]

theorem nilpotent_is_zero (x : A) (n : ℕ) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 := by
  sorry

theorem fourth_power_eq_self (x : A) : x^4 = x := by
  sorry

end nilpotent_is_zero_fourth_power_eq_self_l2569_256905


namespace cube_roots_of_unity_l2569_256955

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
  sorry

end cube_roots_of_unity_l2569_256955


namespace inequality_solution_range_l2569_256971

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) → 
  (9 ≤ m ∧ m < 12) :=
by sorry

end inequality_solution_range_l2569_256971


namespace cookie_price_calculation_l2569_256908

/-- Represents a neighborhood with homes and boxes sold per home -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total boxes sold in a neighborhood -/
def totalBoxesSold (n : Neighborhood) : ℕ :=
  n.homes * n.boxesPerHome

/-- The price per box of cookies -/
def pricePerBox : ℚ := 2

theorem cookie_price_calculation 
  (neighborhoodA neighborhoodB : Neighborhood)
  (hA : neighborhoodA = ⟨10, 2⟩)
  (hB : neighborhoodB = ⟨5, 5⟩)
  (hRevenue : 50 = pricePerBox * max (totalBoxesSold neighborhoodA) (totalBoxesSold neighborhoodB)) :
  pricePerBox = 2 := by
sorry

end cookie_price_calculation_l2569_256908


namespace complex_calculation_result_l2569_256947

theorem complex_calculation_result : 
  ((0.60 * 50 * 0.45 * 30) - (0.40 * 35 / (0.25 * 20))) * ((3/5 * 100) + (2/7 * 49)) = 29762.8 := by
  sorry

end complex_calculation_result_l2569_256947


namespace sqrt_product_eq_180_l2569_256945

theorem sqrt_product_eq_180 : Real.sqrt 75 * Real.sqrt 48 * (27 ^ (1/3 : ℝ)) = 180 := by
  sorry

end sqrt_product_eq_180_l2569_256945


namespace dakota_medical_bill_l2569_256970

/-- Calculates the total medical bill for Dakota's hospital stay -/
def total_medical_bill (
  days : ℕ)
  (bed_charge_per_day : ℕ)
  (specialist_fee_per_hour : ℕ)
  (specialist_time_minutes : ℕ)
  (ambulance_ride_cost : ℕ)
  (iv_cost : ℕ)
  (surgery_duration_hours : ℕ)
  (surgeon_fee_per_hour : ℕ)
  (assistant_fee_per_hour : ℕ)
  (physical_therapy_fee_per_hour : ℕ)
  (physical_therapy_duration_hours : ℕ)
  (medication_a_times_per_day : ℕ)
  (medication_a_cost_per_pill : ℕ)
  (medication_b_duration_hours : ℕ)
  (medication_b_cost_per_hour : ℕ)
  (medication_c_times_per_day : ℕ)
  (medication_c_cost_per_injection : ℕ) : ℕ :=
  let bed_charges := days * bed_charge_per_day
  let specialist_fees := 2 * (specialist_fee_per_hour * specialist_time_minutes / 60) * days
  let iv_charges := days * iv_cost
  let surgery_costs := surgery_duration_hours * (surgeon_fee_per_hour + assistant_fee_per_hour)
  let physical_therapy_fees := physical_therapy_fee_per_hour * physical_therapy_duration_hours * days
  let medication_a_cost := medication_a_times_per_day * medication_a_cost_per_pill * days
  let medication_b_cost := medication_b_duration_hours * medication_b_cost_per_hour * days
  let medication_c_cost := medication_c_times_per_day * medication_c_cost_per_injection * days
  bed_charges + specialist_fees + ambulance_ride_cost + iv_charges + surgery_costs + 
  physical_therapy_fees + medication_a_cost + medication_b_cost + medication_c_cost

/-- Theorem stating that the total medical bill for Dakota's hospital stay is $11,635 -/
theorem dakota_medical_bill : 
  total_medical_bill 3 900 250 15 1800 200 2 1500 800 300 1 3 20 2 45 2 35 = 11635 := by
  sorry

end dakota_medical_bill_l2569_256970


namespace alcohol_quantity_in_mixture_l2569_256976

theorem alcohol_quantity_in_mixture (initial_alcohol : ℝ) (initial_water : ℝ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry

end alcohol_quantity_in_mixture_l2569_256976


namespace erica_money_l2569_256982

def total_money : ℕ := 91
def sam_money : ℕ := 38

theorem erica_money : total_money - sam_money = 53 := by
  sorry

end erica_money_l2569_256982


namespace triangle_area_problem_l2569_256912

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 72 → x = 6 * Real.sqrt 2 := by
  sorry

end triangle_area_problem_l2569_256912


namespace triangle_perimeter_not_85_l2569_256919

theorem triangle_perimeter_not_85 (a b c : ℝ) : 
  a = 24 → b = 18 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 85 := by
  sorry

end triangle_perimeter_not_85_l2569_256919


namespace parabola_translation_l2569_256931

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation amount
def translation_amount : ℝ := 3

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := original_parabola x + translation_amount

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = -2 * x^2 + 3 :=
by sorry

end parabola_translation_l2569_256931


namespace cages_needed_cages_needed_is_five_l2569_256940

def initial_puppies : ℕ := 45
def sold_puppies : ℕ := 11
def puppies_per_cage : ℕ := 7

theorem cages_needed : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage

theorem cages_needed_is_five : cages_needed = 5 := by
  sorry

end cages_needed_cages_needed_is_five_l2569_256940


namespace percentage_relationship_l2569_256958

theorem percentage_relationship (p t j w : ℝ) : 
  j = 0.75 * p → 
  j = 0.80 * t → 
  t = p * (1 - w / 100) → 
  w = 6.25 := by
sorry

end percentage_relationship_l2569_256958


namespace subset_M_N_l2569_256921

theorem subset_M_N : ∀ (x y : ℝ), (|x| + |y| ≤ 1) → (x^2 + y^2 ≤ |x| + |y|) := by
  sorry

end subset_M_N_l2569_256921


namespace geometric_sequence_sum_l2569_256979

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end geometric_sequence_sum_l2569_256979


namespace unique_solution_for_equation_l2569_256941

theorem unique_solution_for_equation :
  ∃! (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) ∧
  a = 2 ^ (1 / 672) ∧
  b = -2 * 2 ^ (1 / 672) ∧
  c = -4 ∧
  d = 4 :=
by sorry

end unique_solution_for_equation_l2569_256941


namespace robotics_club_non_participants_l2569_256994

theorem robotics_club_non_participants (total students_in_electronics students_in_programming students_in_both : ℕ) 
  (h1 : total = 80)
  (h2 : students_in_electronics = 45)
  (h3 : students_in_programming = 50)
  (h4 : students_in_both = 30) :
  total - (students_in_electronics + students_in_programming - students_in_both) = 15 := by
  sorry

end robotics_club_non_participants_l2569_256994


namespace x_value_proof_l2569_256946

theorem x_value_proof (x : ℝ) (h : (1/2 : ℝ) - (1/3 : ℝ) = 3/x) : x = 18 := by
  sorry

end x_value_proof_l2569_256946


namespace vector_sum_magnitude_l2569_256904

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l2569_256904


namespace distance_scientific_notation_l2569_256951

/-- The distance from the Chinese space station to the apogee of the Earth in meters -/
def distance : ℝ := 347000

/-- The coefficient in the scientific notation representation -/
def coefficient : ℝ := 3.47

/-- The exponent in the scientific notation representation -/
def exponent : ℕ := 5

/-- Theorem stating that the distance is equal to its scientific notation representation -/
theorem distance_scientific_notation : distance = coefficient * (10 ^ exponent) := by
  sorry

end distance_scientific_notation_l2569_256951


namespace instrument_purchase_plan_l2569_256950

-- Define the cost prices of instruments A and B
def cost_A : ℕ := 400
def cost_B : ℕ := 300

-- Define the selling prices of instruments A and B
def sell_A : ℕ := 760
def sell_B : ℕ := 540

-- Define the function for the number of B given A
def num_B (a : ℕ) : ℕ := 3 * a + 10

-- Define the total cost function
def total_cost (a : ℕ) : ℕ := cost_A * a + cost_B * (num_B a)

-- Define the profit function
def profit (a : ℕ) : ℕ := (sell_A - cost_A) * a + (sell_B - cost_B) * (num_B a)

-- Theorem statement
theorem instrument_purchase_plan :
  (2 * cost_A + 3 * cost_B = 1700) ∧
  (3 * cost_A + cost_B = 1500) ∧
  (∀ a : ℕ, total_cost a ≤ 30000 → profit a ≥ 21600 → 
    (a = 18 ∧ num_B a = 64) ∨ 
    (a = 19 ∧ num_B a = 67) ∨ 
    (a = 20 ∧ num_B a = 70)) :=
by sorry

end instrument_purchase_plan_l2569_256950


namespace max_profit_children_clothing_l2569_256965

/-- Profit function for children's clothing sales -/
def profit (x : ℝ) : ℝ :=
  (x - 30) * (-2 * x + 200) - 450

/-- Theorem: Maximum profit for children's clothing sales -/
theorem max_profit_children_clothing :
  let x_min : ℝ := 30
  let x_max : ℝ := 60
  ∀ x ∈ Set.Icc x_min x_max,
    profit x ≤ profit x_max ∧
    profit x_max = 1950 := by
  sorry

#check max_profit_children_clothing

end max_profit_children_clothing_l2569_256965


namespace find_d_l2569_256924

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = 2*d + Real.sqrt (a + b + c + d)) : 
  d = (-7 + Real.sqrt 33) / 8 := by
sorry

end find_d_l2569_256924


namespace fraction_meaningful_l2569_256993

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end fraction_meaningful_l2569_256993


namespace dvds_sold_per_day_is_497_l2569_256937

/-- Represents the DVD business model -/
structure DVDBusiness where
  initialCost : ℕ
  productionCost : ℕ
  sellingPriceFactor : ℚ
  daysPerWeek : ℕ
  totalWeeks : ℕ
  totalProfit : ℕ

/-- Calculates the number of DVDs sold per day -/
def calculateDVDsSoldPerDay (business : DVDBusiness) : ℕ :=
  let sellingPrice := business.productionCost * business.sellingPriceFactor
  let profitPerDVD := sellingPrice - business.productionCost
  let totalDays := business.daysPerWeek * business.totalWeeks
  let profitPerDay := business.totalProfit / totalDays
  (profitPerDay / profitPerDVD).floor.toNat

/-- Theorem stating that the number of DVDs sold per day is 497 -/
theorem dvds_sold_per_day_is_497 (business : DVDBusiness) 
  (h1 : business.initialCost = 2000)
  (h2 : business.productionCost = 6)
  (h3 : business.sellingPriceFactor = 2.5)
  (h4 : business.daysPerWeek = 5)
  (h5 : business.totalWeeks = 20)
  (h6 : business.totalProfit = 448000) :
  calculateDVDsSoldPerDay business = 497 := by
  sorry

#eval calculateDVDsSoldPerDay {
  initialCost := 2000,
  productionCost := 6,
  sellingPriceFactor := 2.5,
  daysPerWeek := 5,
  totalWeeks := 20,
  totalProfit := 448000
}

end dvds_sold_per_day_is_497_l2569_256937


namespace daily_increase_amount_l2569_256997

def fine_sequence (x : ℚ) : ℕ → ℚ
  | 0 => 0.05
  | n + 1 => min (fine_sequence x n + x) (2 * fine_sequence x n)

theorem daily_increase_amount :
  ∃ x : ℚ, x > 0 ∧ fine_sequence x 4 = 0.70 ∧ 
  ∀ n : ℕ, n > 0 → fine_sequence x n = fine_sequence x (n-1) + x :=
by sorry

end daily_increase_amount_l2569_256997


namespace jogger_distance_ahead_l2569_256964

def jogger_speed : ℝ := 9 -- km/hr
def train_speed : ℝ := 45 -- km/hr
def train_length : ℝ := 210 -- meters
def passing_time : ℝ := 41 -- seconds

theorem jogger_distance_ahead (jogger_speed train_speed train_length passing_time : ℝ) :
  jogger_speed = 9 ∧ 
  train_speed = 45 ∧ 
  train_length = 210 ∧ 
  passing_time = 41 →
  (train_speed - jogger_speed) * passing_time / 3600 * 1000 - train_length = 200 := by
  sorry

end jogger_distance_ahead_l2569_256964


namespace divisibility_by_441_l2569_256909

theorem divisibility_by_441 (a b : ℕ) (h : 21 ∣ (a^2 + b^2)) : 441 ∣ (a^2 + b^2) := by
  sorry

end divisibility_by_441_l2569_256909


namespace city_fuel_efficiency_l2569_256936

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure CarFuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure CarRange where
  highway : ℝ
  city : ℝ

/-- The difference between highway and city fuel efficiency in miles per gallon -/
def efficiency_difference : ℝ := 12

theorem city_fuel_efficiency 
  (car_range : CarRange)
  (car_efficiency : CarFuelEfficiency)
  (h1 : car_range.highway = 800)
  (h2 : car_range.city = 500)
  (h3 : car_efficiency.city = car_efficiency.highway - efficiency_difference)
  (h4 : car_range.highway / car_efficiency.highway = car_range.city / car_efficiency.city) :
  car_efficiency.city = 20 := by
sorry

end city_fuel_efficiency_l2569_256936


namespace project_wage_difference_l2569_256963

theorem project_wage_difference (total_pay : ℝ) (p_hours q_hours : ℝ) 
  (hp : total_pay = 420)
  (hpq : q_hours = p_hours + 10)
  (hw : p_hours * (1.5 * (total_pay / q_hours)) = total_pay) :
  1.5 * (total_pay / q_hours) - (total_pay / q_hours) = 7 := by
  sorry

end project_wage_difference_l2569_256963


namespace odd_function_property_l2569_256918

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f (-1) = 2) :
  f 1 = -2 := by
  sorry

end odd_function_property_l2569_256918


namespace star_calculation_l2569_256953

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem statement
theorem star_calculation : star (star 7 15) 10 = 52 / 9 := by
  sorry

end star_calculation_l2569_256953


namespace condo_units_l2569_256990

/-- Calculates the total number of units in a condo building -/
def total_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_units : ℕ) (penthouse_floors : ℕ) : ℕ :=
  (total_floors - penthouse_floors) * regular_units + penthouse_floors * penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units -/
theorem condo_units : total_units 23 12 2 2 = 256 := by
  sorry

end condo_units_l2569_256990


namespace min_value_expression_l2569_256917

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 3*b + 4*c = 1) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2*x + 3*y + 4*z = 1 → 
    1/a + 2/b + 3/c ≤ 1/x + 2/y + 3/z) ∧ 
  1/a + 2/b + 3/c = 20 + 4*Real.sqrt 3 + 20*Real.sqrt 2 :=
by sorry

end min_value_expression_l2569_256917


namespace circle_properties_l2569_256914

def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

theorem circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, circle_equation x y → x = 0 ∧ y = k) ∧
  (∀ x y : ℝ, circle_equation x y → (x - 0)^2 + (y - 2)^2 = 1) ∧
  circle_equation 1 2 :=
sorry

end circle_properties_l2569_256914


namespace min_abs_sum_with_constraints_l2569_256923

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    α' + β' + γ' = 2 → α' * β' * γ' = 4 → 
    |α'| + |β'| + |γ'| ≥ v :=
sorry

end min_abs_sum_with_constraints_l2569_256923


namespace least_b_value_l2569_256933

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem least_b_value (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_a_factors : num_factors a = 4) 
  (h_b_factors : num_factors b = a) 
  (h_b_div_a : a ∣ b) : 
  ∀ c, c > 0 ∧ num_factors c = a ∧ a ∣ c → b ≤ c ∧ b = 12 := by
  sorry

end least_b_value_l2569_256933


namespace harriett_found_three_dollars_l2569_256980

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The number of quarters Harriett found -/
def quarters_found : ℕ := 10

/-- The number of dimes Harriett found -/
def dimes_found : ℕ := 3

/-- The number of nickels Harriett found -/
def nickels_found : ℕ := 3

/-- The number of pennies Harriett found -/
def pennies_found : ℕ := 5

/-- The total value of the coins Harriett found -/
def total_value : ℚ := 
  quarters_found * quarter_value + 
  dimes_found * dime_value + 
  nickels_found * nickel_value + 
  pennies_found * penny_value

theorem harriett_found_three_dollars : total_value = 3 := by
  sorry

end harriett_found_three_dollars_l2569_256980


namespace triangle_acute_from_sine_ratio_l2569_256911

theorem triangle_acute_from_sine_ratio (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5 * k ∧ Real.sin B = 11 * k ∧ Real.sin C = 13 * k) :
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2 := by
  sorry

end triangle_acute_from_sine_ratio_l2569_256911


namespace cheese_distribution_l2569_256972

theorem cheese_distribution (M : ℝ) (x y : ℝ) : 
  -- Total cheese weight
  M > 0 →
  -- White's slice is exactly one-quarter of the total
  y = M / 4 →
  -- Thin's slice weighs x
  -- Fat's slice weighs x + 20
  -- White's slice weighs y
  -- Gray's slice weighs y + 8
  x + (x + 20) + y + (y + 8) = M →
  -- Gray cuts 8 grams, Fat cuts 20 grams
  -- To achieve equal distribution, Fat and Thin should each get 14 grams
  14 = (28 : ℝ) / 2 ∧
  x + 14 = y ∧
  (x + 20) - 20 + 14 = y ∧
  (y + 8) - 8 = y :=
by
  sorry

end cheese_distribution_l2569_256972


namespace acute_triangle_sine_sum_l2569_256977

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi)
  (acute_angles : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end acute_triangle_sine_sum_l2569_256977


namespace car_travel_time_l2569_256996

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) (t : ℝ) 
  (h1 : distance = 630)
  (h2 : new_speed = 70)
  (h3 : time_ratio = 3/2)
  (h4 : distance = (distance / t) * (time_ratio * t))
  (h5 : distance = new_speed * (time_ratio * t)) :
  t = 6 := by
sorry

end car_travel_time_l2569_256996


namespace pipe_A_rate_l2569_256907

/-- The rate at which Pipe A fills the tank -/
def rate_A : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 800

/-- The rate at which Pipe B fills the tank in liters per minute -/
def rate_B : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def rate_C : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 48

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

theorem pipe_A_rate : 
  rate_A = 40 ∧ 
  (fill_time / cycle_duration) * (rate_A + rate_B - rate_C) = tank_capacity := by
  sorry

end pipe_A_rate_l2569_256907


namespace arithmetic_sequence_property_l2569_256922

/-- An arithmetic sequence with its sum satisfying a specific quadratic equation -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  A : ℝ      -- Coefficient of n^2
  B : ℝ      -- Coefficient of n
  h1 : A ≠ 0
  h2 : ∀ n : ℕ, a n + S n = A * n^2 + B * n + 1

/-- The main theorem: if an arithmetic sequence satisfies the given condition, then (B-1)/A = 3 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : (seq.B - 1) / seq.A = 3 := by
  sorry

end arithmetic_sequence_property_l2569_256922


namespace probability_of_event_A_l2569_256902

theorem probability_of_event_A (P_B P_AB P_AUB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAUB : P_AUB = 0.6)
  : ∃ P_A : ℝ, P_A = 0.45 ∧ P_AUB = P_A + P_B - P_AB :=
by
  sorry

end probability_of_event_A_l2569_256902


namespace regression_change_l2569_256986

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 7 - 3 * x

-- Theorem statement
theorem regression_change (x₁ x₂ : ℝ) (h : x₂ = x₁ + 2) :
  regression_equation x₁ - regression_equation x₂ = 6 := by
  sorry

end regression_change_l2569_256986


namespace square_equals_cube_root_16_l2569_256929

theorem square_equals_cube_root_16 : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end square_equals_cube_root_16_l2569_256929


namespace donut_problem_l2569_256978

theorem donut_problem (initial_donuts : ℕ) (h1 : initial_donuts = 50) : 
  let after_bill_eats := initial_donuts - 2
  let after_secretary_takes := after_bill_eats - 4
  let stolen_by_coworkers := after_secretary_takes / 2
  initial_donuts - 2 - 4 - stolen_by_coworkers = 22 := by
  sorry

end donut_problem_l2569_256978


namespace function_inequality_l2569_256992

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, HasDerivAt f (f' x) x ∧ f' x < 1/3)

-- State the theorem
theorem function_inequality (x : ℝ) :
  f x < x/3 + 2/3 ↔ x > 1 := by sorry

end function_inequality_l2569_256992


namespace cubic_value_given_quadratic_l2569_256944

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 - 2*x - 1 = 0 → 3*x^3 - 10*x^2 + 5*x + 2027 = 2023 := by
  sorry

end cubic_value_given_quadratic_l2569_256944


namespace optimal_price_is_160_l2569_256903

/-- Represents the price and occupancy data for a hotel room --/
structure PriceOccupancy where
  price : ℝ
  occupancy : ℝ

/-- Calculates the daily income for a given price and occupancy --/
def dailyIncome (po : PriceOccupancy) (totalRooms : ℝ) : ℝ :=
  po.price * po.occupancy * totalRooms

/-- Theorem: The optimal price for maximizing daily income is 160 yuan --/
theorem optimal_price_is_160 (totalRooms : ℝ) 
  (priceOccupancyData : List PriceOccupancy) 
  (h1 : totalRooms = 100)
  (h2 : priceOccupancyData = [
    ⟨200, 0.65⟩, 
    ⟨180, 0.75⟩, 
    ⟨160, 0.85⟩, 
    ⟨140, 0.95⟩
  ]) : 
  ∃ (optimalPO : PriceOccupancy), 
    optimalPO ∈ priceOccupancyData ∧ 
    optimalPO.price = 160 ∧
    ∀ (po : PriceOccupancy), 
      po ∈ priceOccupancyData → 
      dailyIncome optimalPO totalRooms ≥ dailyIncome po totalRooms :=
by sorry

end optimal_price_is_160_l2569_256903
