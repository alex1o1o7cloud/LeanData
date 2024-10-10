import Mathlib

namespace cube_sum_identity_l1947_194705

theorem cube_sum_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3*x*y*z) / (x*y*z) = 6 := by
  sorry

end cube_sum_identity_l1947_194705


namespace min_value_implies_t_l1947_194729

/-- Given a real number t, f(x) is defined as the sum of the absolute values of (x-t) and (5-x) -/
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |5 - x|

/-- The theorem states that if the minimum value of f(x) is 3, then t must be either 2 or 8 -/
theorem min_value_implies_t (t : ℝ) (h : ∀ x, f t x ≥ 3) (h_min : ∃ x, f t x = 3) : t = 2 ∨ t = 8 := by
  sorry

end min_value_implies_t_l1947_194729


namespace three_and_negative_three_are_opposite_l1947_194726

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem three_and_negative_three_are_opposite :
  are_opposite 3 (-3) :=
sorry

end three_and_negative_three_are_opposite_l1947_194726


namespace min_computers_to_purchase_l1947_194788

/-- Represents the problem of finding the minimum number of computers to purchase --/
theorem min_computers_to_purchase (total_devices : ℕ) (computer_cost whiteboard_cost max_cost : ℚ) :
  total_devices = 30 →
  computer_cost = 1/2 →
  whiteboard_cost = 3/2 →
  max_cost = 30 →
  ∃ (min_computers : ℕ),
    min_computers = 15 ∧
    ∀ (x : ℕ),
      x < 15 →
      (x : ℚ) * computer_cost + (total_devices - x : ℚ) * whiteboard_cost > max_cost :=
by sorry

end min_computers_to_purchase_l1947_194788


namespace greatest_base9_digit_sum_l1947_194700

/-- Represents a positive integer in base 9 --/
structure Base9Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 9

/-- Converts a Base9Int to its decimal (base 10) representation --/
def toDecimal (n : Base9Int) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Calculates the sum of digits of a Base9Int --/
def digitSum (n : Base9Int) : Nat :=
  n.digits.sum

/-- The main theorem to be proved --/
theorem greatest_base9_digit_sum :
  ∃ (max : Nat), 
    (∀ (n : Base9Int), toDecimal n < 3000 → digitSum n ≤ max) ∧ 
    (∃ (n : Base9Int), toDecimal n < 3000 ∧ digitSum n = max) ∧
    max = 24 := by
  sorry

end greatest_base9_digit_sum_l1947_194700


namespace frigate_catches_smuggler_l1947_194767

/-- Represents the chase scenario between a frigate and a smuggler's ship -/
structure ChaseScenario where
  initial_distance : ℝ
  frigate_speed : ℝ
  smuggler_speed : ℝ
  chase_duration : ℝ

/-- Calculates the distance traveled by a ship given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that the frigate catches up to the smuggler's ship after 3 hours -/
theorem frigate_catches_smuggler (scenario : ChaseScenario) 
    (h1 : scenario.initial_distance = 12)
    (h2 : scenario.frigate_speed = 14)
    (h3 : scenario.smuggler_speed = 10)
    (h4 : scenario.chase_duration = 3) :
    distance_traveled scenario.frigate_speed scenario.chase_duration = 
    scenario.initial_distance + distance_traveled scenario.smuggler_speed scenario.chase_duration :=
  sorry

#check frigate_catches_smuggler

end frigate_catches_smuggler_l1947_194767


namespace domain_of_g_l1947_194791

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- State the theorem
theorem domain_of_g :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 3) →
  (∀ x, g x ≠ 0 → x ∈ Set.Icc (-3/2) 1) :=
sorry

end domain_of_g_l1947_194791


namespace min_value_expression_min_value_achieved_l1947_194780

theorem min_value_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) ≥ 2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) 
  (ha : a = 0) 
  (hb : b = 0) 
  (hc : c = 0) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) = 2 :=
by sorry

end min_value_expression_min_value_achieved_l1947_194780


namespace exponential_inequality_l1947_194707

theorem exponential_inequality (a b : ℝ) (h : a < b) :
  let f := fun (x : ℝ) => Real.exp x
  let A := f b - f a
  let B := (1/2) * (b - a) * (f a + f b)
  A < B := by sorry

end exponential_inequality_l1947_194707


namespace sunday_bicycles_bought_l1947_194763

/-- Represents the number of bicycles in Hank's store. -/
def BicycleCount := ℤ

/-- Represents the change in bicycle count for a day. -/
structure DailyChange where
  sold : ℕ
  bought : ℕ

/-- Calculates the net change in bicycle count for a day. -/
def netChange (dc : DailyChange) : ℤ :=
  dc.bought - dc.sold

/-- Represents the changes in bicycle count over three days. -/
structure ThreeDayChanges where
  friday : DailyChange
  saturday : DailyChange
  sunday_sold : ℕ

theorem sunday_bicycles_bought 
  (changes : ThreeDayChanges)
  (h_friday : changes.friday = ⟨10, 15⟩)
  (h_saturday : changes.saturday = ⟨12, 8⟩)
  (h_sunday_sold : changes.sunday_sold = 9)
  (h_net_increase : netChange changes.friday + netChange changes.saturday + 
    (sunday_bought - changes.sunday_sold) = 3)
  : ∃ (sunday_bought : ℕ), sunday_bought = 11 :=
by
  sorry

end sunday_bicycles_bought_l1947_194763


namespace concert_tickets_l1947_194709

theorem concert_tickets (section_a_price section_b_price : ℝ)
  (total_tickets : ℕ) (total_revenue : ℝ) :
  section_a_price = 8 →
  section_b_price = 4.25 →
  total_tickets = 4500 →
  total_revenue = 30000 →
  ∃ (section_a_sold section_b_sold : ℕ),
    section_a_sold + section_b_sold = total_tickets ∧
    section_a_price * (section_a_sold : ℝ) + section_b_price * (section_b_sold : ℝ) = total_revenue ∧
    section_b_sold = 1600 :=
by sorry

end concert_tickets_l1947_194709


namespace max_value_of_2x_plus_y_l1947_194769

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
by sorry

end max_value_of_2x_plus_y_l1947_194769


namespace animal_lifespan_l1947_194737

theorem animal_lifespan (bat_lifespan hamster_lifespan frog_lifespan : ℕ) : 
  bat_lifespan = 10 →
  hamster_lifespan = bat_lifespan - 6 →
  frog_lifespan = 4 * hamster_lifespan →
  bat_lifespan + hamster_lifespan + frog_lifespan = 30 := by
  sorry

end animal_lifespan_l1947_194737


namespace player_in_first_and_last_game_l1947_194721

/-- Represents a chess tournament. -/
structure ChessTournament (n : ℕ) where
  /-- The number of players in the tournament. -/
  num_players : ℕ
  /-- The total number of games played in the tournament. -/
  num_games : ℕ
  /-- Condition that the number of players is 2n+3. -/
  player_count : num_players = 2*n + 3
  /-- Condition that the number of games is (2n+3)*(2n+2)/2. -/
  game_count : num_games = (num_players * (num_players - 1)) / 2
  /-- Function that returns true if a player played in a specific game. -/
  played_in_game : ℕ → ℕ → Prop
  /-- Condition that each player rests for at least n games after each match. -/
  rest_condition : ∀ p g₁ g₂, played_in_game p g₁ → played_in_game p g₂ → g₁ < g₂ → g₂ - g₁ > n

/-- Theorem stating that a player who played in the first game also played in the last game. -/
theorem player_in_first_and_last_game (n : ℕ) (tournament : ChessTournament n) :
  ∃ p, tournament.played_in_game p 1 ∧ tournament.played_in_game p tournament.num_games :=
sorry

end player_in_first_and_last_game_l1947_194721


namespace intersection_slope_problem_l1947_194765

/-- Given two lines intersecting at (40, 30), where one line has a slope of 6
    and the distance between their x-intercepts is 10,
    prove that the slope of the other line is 2. -/
theorem intersection_slope_problem (m : ℝ) : 
  let line1 : ℝ → ℝ := λ x => m * x - 40 * m + 30
  let line2 : ℝ → ℝ := λ x => 6 * x - 210
  let x_intercept1 : ℝ := (40 * m - 30) / m
  let x_intercept2 : ℝ := 35
  (∃ x y, line1 x = line2 x ∧ x = 40 ∧ y = 30) →  -- Lines intersect at (40, 30)
  |x_intercept1 - x_intercept2| = 10 →           -- Distance between x-intercepts is 10
  m = 2                                          -- Slope of the first line is 2
  := by sorry

end intersection_slope_problem_l1947_194765


namespace problem_solution_l1947_194734

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end problem_solution_l1947_194734


namespace geometric_progression_problem_l1947_194781

theorem geometric_progression_problem (b₁ q : ℝ) 
  (h_decreasing : |q| < 1)
  (h_sum_diff : b₁ / (1 - q^2) - (b₁ * q) / (1 - q^2) = 10)
  (h_sum_squares_diff : b₁^2 / (1 - q^4) - (b₁^2 * q^2) / (1 - q^4) = 20) :
  b₁ = 5 ∧ q = -1/2 := by
sorry

end geometric_progression_problem_l1947_194781


namespace rectangular_field_area_l1947_194757

/-- The area of a rectangular field with one side 15 meters and diagonal 17 meters is 120 square meters. -/
theorem rectangular_field_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * Real.sqrt (diagonal^2 - side^2) → area = 120 :=
by sorry

end rectangular_field_area_l1947_194757


namespace gcd_lcm_sum_l1947_194776

theorem gcd_lcm_sum : Nat.gcd 25 64 + Nat.lcm 15 20 = 61 := by sorry

end gcd_lcm_sum_l1947_194776


namespace smallest_n_with_properties_l1947_194789

def has_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a * 10^c + d₁ * 10^b + d₂

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_properties : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    is_terminating_decimal n ∧ 
    has_digits n 9 5 → 
    n ≥ 9000000 :=
sorry

end smallest_n_with_properties_l1947_194789


namespace two_different_color_chips_probability_l1947_194751

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def total_chips : ℕ := blue_chips + yellow_chips + red_chips

def probability_different_colors : ℚ :=
  (blue_chips * yellow_chips + blue_chips * red_chips + yellow_chips * red_chips) * 2 /
  (total_chips * total_chips)

theorem two_different_color_chips_probability :
  probability_different_colors = 47 / 72 := by
  sorry

end two_different_color_chips_probability_l1947_194751


namespace fraction_value_l1947_194778

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 4 * d) 
  (h4 : d ≠ 0) : a * c / (b * d) = 12 := by
  sorry

end fraction_value_l1947_194778


namespace adults_attending_play_l1947_194760

/-- Proves the number of adults attending a play given the total attendance,
    admission prices, and total receipts. -/
theorem adults_attending_play
  (total_people : ℕ)
  (adult_price child_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : child_price = 1)
  (h4 : total_receipts = 960) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end adults_attending_play_l1947_194760


namespace course_selection_methods_l1947_194768

/-- The number of courses in Group A -/
def group_A_courses : ℕ := 3

/-- The number of courses in Group B -/
def group_B_courses : ℕ := 4

/-- The total number of courses that must be selected -/
def total_selected : ℕ := 3

/-- The function to calculate the number of ways to select courses -/
def select_courses (group_A : ℕ) (group_B : ℕ) (total : ℕ) : ℕ :=
  Nat.choose group_A 2 * Nat.choose group_B 1 +
  Nat.choose group_A 1 * Nat.choose group_B 2

/-- Theorem stating that the number of different selection methods is 30 -/
theorem course_selection_methods :
  select_courses group_A_courses group_B_courses total_selected = 30 := by
  sorry

end course_selection_methods_l1947_194768


namespace camel_division_theorem_l1947_194735

/-- A representation of the "camel" figure --/
structure CamelFigure where
  area : ℕ
  has_spaced_cells : Bool

/-- Represents a division of the figure --/
inductive Division
  | GridLines
  | Arbitrary

/-- Represents the result of attempting to form a square --/
inductive SquareFormation
  | Possible
  | Impossible

/-- Function to determine if a square can be formed from the division --/
def can_form_square (figure : CamelFigure) (division : Division) : SquareFormation :=
  match division with
  | Division.GridLines => 
      if figure.has_spaced_cells then SquareFormation.Impossible else SquareFormation.Possible
  | Division.Arbitrary => 
      if figure.area == 25 then SquareFormation.Possible else SquareFormation.Impossible

/-- The main theorem about the camel figure --/
theorem camel_division_theorem (camel : CamelFigure) 
    (h1 : camel.area = 25) 
    (h2 : camel.has_spaced_cells = true) : 
    (can_form_square camel Division.GridLines = SquareFormation.Impossible) ∧ 
    (can_form_square camel Division.Arbitrary = SquareFormation.Possible) := by
  sorry

end camel_division_theorem_l1947_194735


namespace alchemerion_age_proof_l1947_194722

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 360

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

theorem alchemerion_age_proof :
  (alchemerion_age = 3 * son_age) ∧
  (father_age = 2 * alchemerion_age + 40) ∧
  (alchemerion_age + son_age + father_age = 1240) :=
by sorry

end alchemerion_age_proof_l1947_194722


namespace right_triangle_third_vertex_l1947_194777

theorem right_triangle_third_vertex 
  (v1 : ℝ × ℝ) 
  (v2 : ℝ × ℝ) 
  (x : ℝ) :
  v1 = (4, 3) →
  v2 = (0, 0) →
  x > 0 →
  (1/2 : ℝ) * x * 3 = 24 →
  x = 16 := by
sorry

end right_triangle_third_vertex_l1947_194777


namespace remainder_17_pow_2090_mod_23_l1947_194732

theorem remainder_17_pow_2090_mod_23 : 17^2090 % 23 = 12 := by
  sorry

end remainder_17_pow_2090_mod_23_l1947_194732


namespace second_car_speed_l1947_194790

/-- Theorem: Given two cars starting from opposite ends of a 500-mile highway
    at the same time, with one car traveling at 40 mph and both cars meeting
    after 5 hours, the speed of the second car is 60 mph. -/
theorem second_car_speed
  (highway_length : ℝ)
  (first_car_speed : ℝ)
  (meeting_time : ℝ)
  (second_car_speed : ℝ) :
  highway_length = 500 →
  first_car_speed = 40 →
  meeting_time = 5 →
  highway_length = first_car_speed * meeting_time + second_car_speed * meeting_time →
  second_car_speed = 60 :=
by
  sorry

#check second_car_speed

end second_car_speed_l1947_194790


namespace abc_inequality_l1947_194753

theorem abc_inequality (a b c : ℝ) (ha : -1 ≤ a ∧ a ≤ 2) (hb : -1 ≤ b ∧ b ≤ 2) (hc : -1 ≤ c ∧ c ≤ 2) :
  a * b * c + 4 ≥ a * b + b * c + c * a := by
sorry

end abc_inequality_l1947_194753


namespace x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l1947_194761

theorem x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative :
  (∀ x : ℝ, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l1947_194761


namespace tan_theta_value_l1947_194738

theorem tan_theta_value (θ : Real) : 
  Real.tan (π / 4 + θ) = 1 / 2 → Real.tan θ = -1 / 3 := by
  sorry

end tan_theta_value_l1947_194738


namespace proportion_check_l1947_194712

/-- A set of four line segments forms a proportion if the product of the means equals the product of the extremes. -/
def is_proportion (a b c d : ℝ) : Prop := b * c = a * d

/-- The given sets of line segments -/
def set_A : Fin 4 → ℝ := ![2, 3, 5, 6]
def set_B : Fin 4 → ℝ := ![1, 2, 3, 5]
def set_C : Fin 4 → ℝ := ![1, 3, 3, 7]
def set_D : Fin 4 → ℝ := ![3, 2, 4, 6]

theorem proportion_check :
  ¬ is_proportion (set_A 0) (set_A 1) (set_A 2) (set_A 3) ∧
  ¬ is_proportion (set_B 0) (set_B 1) (set_B 2) (set_B 3) ∧
  ¬ is_proportion (set_C 0) (set_C 1) (set_C 2) (set_C 3) ∧
  is_proportion (set_D 0) (set_D 1) (set_D 2) (set_D 3) := by
  sorry

end proportion_check_l1947_194712


namespace exist_three_quadratic_polynomials_l1947_194779

theorem exist_three_quadratic_polynomials :
  ∃ (f g h : ℝ → ℝ),
    (∀ x, f x = (x - 3)^2 - 1) ∧
    (∀ x, g x = x^2 - 1) ∧
    (∀ x, h x = (x + 3)^2 - 1) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
    (∃ y₁ y₂, y₁ ≠ y₂ ∧ g y₁ = 0 ∧ g y₂ = 0) ∧
    (∃ z₁ z₂, z₁ ≠ z₂ ∧ h z₁ = 0 ∧ h z₂ = 0) ∧
    (∀ x, (f x + g x) ≠ 0) ∧
    (∀ x, (f x + h x) ≠ 0) ∧
    (∀ x, (g x + h x) ≠ 0) :=
by sorry

end exist_three_quadratic_polynomials_l1947_194779


namespace solve_equations_l1947_194773

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x - 4 = 2 * x + 5
def equation2 (x : ℝ) : Prop := (x - 3) / 4 - (2 * x + 1) / 2 = 1

-- State the theorem
theorem solve_equations :
  (∃! x : ℝ, equation1 x) ∧ (∃! x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation1 x → x = 9) ∧
  (∀ x : ℝ, equation2 x → x = -6) :=
by sorry

end solve_equations_l1947_194773


namespace simplify_fraction_l1947_194702

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end simplify_fraction_l1947_194702


namespace f_properties_l1947_194730

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.pi/2) * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3/4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = 1/4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1/2) := by
  sorry

end f_properties_l1947_194730


namespace parabola_through_point_l1947_194714

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A parabola opens upwards if a > 0 --/
def Parabola.opensUpwards (p : Parabola) : Prop := p.a > 0

/-- A point (x, y) lies on a parabola if it satisfies the parabola's equation --/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The theorem states that there exists a parabola that opens upwards and passes through (0, -2) --/
theorem parabola_through_point : ∃ p : Parabola, 
  p.opensUpwards ∧ p.containsPoint 0 (-2) ∧ p.a = 1 ∧ p.b = 0 ∧ p.c = -2 := by
  sorry

end parabola_through_point_l1947_194714


namespace train_speed_calculation_l1947_194701

theorem train_speed_calculation (v : ℝ) : 
  v > 0 → -- The speed is positive
  (v + 42) * (5 / 18) * 9 = 280 → -- Equation derived from the problem
  v = 70 := by
sorry

end train_speed_calculation_l1947_194701


namespace expression_value_l1947_194799

/-- Custom operation for real numbers -/
def custom_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the value of the expression when x^2 - 3x + 1 = 0 -/
theorem expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  custom_op (x + 1) (x - 2) (3*x) (x - 1) = 1 := by
  sorry

end expression_value_l1947_194799


namespace original_book_count_l1947_194752

/-- Represents a bookshelf with three layers of books -/
structure Bookshelf :=
  (layer1 : ℕ)
  (layer2 : ℕ)
  (layer3 : ℕ)

/-- The total number of books on the bookshelf -/
def total_books (b : Bookshelf) : ℕ := b.layer1 + b.layer2 + b.layer3

/-- The bookshelf after moving books between layers -/
def move_books (b : Bookshelf) : Bookshelf :=
  { layer1 := b.layer1 - 20,
    layer2 := b.layer2 + 20 + 17,
    layer3 := b.layer3 - 17 }

/-- Theorem stating the original number of books on each layer -/
theorem original_book_count :
  ∀ b : Bookshelf,
    total_books b = 270 →
    (let b' := move_books b
     b'.layer1 = b'.layer2 ∧ b'.layer2 = b'.layer3) →
    b.layer1 = 110 ∧ b.layer2 = 53 ∧ b.layer3 = 107 :=
by sorry


end original_book_count_l1947_194752


namespace tan_2alpha_l1947_194758

theorem tan_2alpha (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 5) :
  Real.tan (2 * α) = -4/7 := by sorry

end tan_2alpha_l1947_194758


namespace sock_shoe_permutations_l1947_194784

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_permutations : ℕ := Nat.factorial total_items / (2^num_legs)

theorem sock_shoe_permutations :
  valid_permutations = Nat.factorial total_items / (2^num_legs) :=
by sorry

end sock_shoe_permutations_l1947_194784


namespace ten_pound_bag_cost_l1947_194743

/-- Represents the cost and weight of a bag of grass seed -/
structure Bag where
  weight : ℕ
  cost : ℚ

/-- Represents the purchase constraints and known information -/
structure PurchaseInfo where
  minWeight : ℕ
  maxWeight : ℕ
  leastCost : ℚ
  bag5lb : Bag
  bag25lb : Bag

/-- Calculates the cost of a 10-pound bag given the purchase information -/
def calculate10lbBagCost (info : PurchaseInfo) : ℚ :=
  info.leastCost - 3 * info.bag25lb.cost

/-- Theorem stating that the cost of the 10-pound bag is $1.98 -/
theorem ten_pound_bag_cost (info : PurchaseInfo) 
  (h1 : info.minWeight = 65)
  (h2 : info.maxWeight = 80)
  (h3 : info.leastCost = 98.73)
  (h4 : info.bag5lb = ⟨5, 13.80⟩)
  (h5 : info.bag25lb = ⟨25, 32.25⟩) :
  calculate10lbBagCost info = 1.98 := by sorry

end ten_pound_bag_cost_l1947_194743


namespace factor_x_squared_minus_64_l1947_194708

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l1947_194708


namespace problem_solution_l1947_194739

-- Define the linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define the quadratic function
def quadratic_function (m n : ℝ) : ℝ → ℝ := λ x ↦ x^2 + m * x + n

theorem problem_solution 
  (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : linear_function k b (-3) = 0)
  (h3 : linear_function k b 0 = -3)
  (h4 : quadratic_function m n (-3) = 0)
  (h5 : quadratic_function m n 0 = 3)
  (h6 : n > 0)
  (h7 : m ≤ 5) :
  (∃ t : ℝ, 
    (k = -1 ∧ b = -3) ∧ 
    (∃ x y : ℝ, x^2 + m*x + n = -x - 3 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ x^2 + m*x + n) ∧
    (-9/4 < t ∧ t ≤ -1/4 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ t)) := by
  sorry

end problem_solution_l1947_194739


namespace prob_different_colors_is_two_thirds_l1947_194782

/-- Represents the possible colors for socks -/
inductive SockColor
| Red
| Blue

/-- Represents the possible colors for headbands -/
inductive HeadbandColor
| Red
| Blue
| Green

/-- The probability of choosing different colors for socks and headbands -/
def prob_different_colors : ℚ :=
  2 / 3

theorem prob_different_colors_is_two_thirds :
  prob_different_colors = 2 / 3 :=
by sorry

end prob_different_colors_is_two_thirds_l1947_194782


namespace max_groups_is_100_l1947_194764

/-- Represents the number of cards for each value -/
def CardCount : ℕ := 200

/-- Represents the target sum for each group -/
def TargetSum : ℕ := 9

/-- Represents the maximum number of groups that can be formed -/
def MaxGroups : ℕ := 100

/-- Proves that the maximum number of groups that can be formed is 100 -/
theorem max_groups_is_100 :
  ∀ (groups : ℕ) (cards_5 cards_2 cards_1 : ℕ),
    cards_5 = CardCount →
    cards_2 = CardCount →
    cards_1 = CardCount →
    (∀ g : ℕ, g ≤ groups → ∃ (a b c : ℕ),
      a + b + c = TargetSum ∧
      a * 5 + b * 2 + c * 1 = TargetSum ∧
      a ≤ cards_5 ∧ b ≤ cards_2 ∧ c ≤ cards_1) →
    groups ≤ MaxGroups :=
  sorry

end max_groups_is_100_l1947_194764


namespace arithmetic_sequence_count_l1947_194798

theorem arithmetic_sequence_count (start end_ diff : ℕ) (h1 : start = 24) (h2 : end_ = 162) (h3 : diff = 6) :
  (end_ - start) / diff + 1 = 24 := by
  sorry

end arithmetic_sequence_count_l1947_194798


namespace a_less_than_sqrt_a_iff_l1947_194793

theorem a_less_than_sqrt_a_iff (a : ℝ) : 0 < a ∧ a < 1 ↔ a < Real.sqrt a := by sorry

end a_less_than_sqrt_a_iff_l1947_194793


namespace point_in_second_quadrant_l1947_194744

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end point_in_second_quadrant_l1947_194744


namespace line_plane_relationship_l1947_194766

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_or_intersect_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (β : Plane) 
  (h1 : intersect a b) 
  (h2 : parallel a β) : 
  line_parallel_or_intersect_plane b β :=
sorry

end line_plane_relationship_l1947_194766


namespace sibling_age_equation_l1947_194742

/-- Represents the ages of two siblings -/
structure SiblingAges where
  sister : ℕ
  brother : ℕ

/-- The condition of the ages this year -/
def this_year (ages : SiblingAges) : Prop :=
  ages.brother = 2 * ages.sister

/-- The condition of the ages four years ago -/
def four_years_ago (ages : SiblingAges) : Prop :=
  (ages.brother - 4) = 3 * (ages.sister - 4)

/-- The theorem representing the problem -/
theorem sibling_age_equation (x : ℕ) :
  ∃ (ages : SiblingAges),
    ages.sister = x ∧
    this_year ages ∧
    four_years_ago ages →
    2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end sibling_age_equation_l1947_194742


namespace ink_remaining_proof_l1947_194755

/-- The total area a full marker can cover, in square inches -/
def full_marker_coverage : ℝ := 48

/-- The area covered by the rectangles, in square inches -/
def area_covered : ℝ := 24

/-- The percentage of ink remaining after covering the rectangles -/
def ink_remaining_percentage : ℝ := 50

theorem ink_remaining_proof :
  (full_marker_coverage - area_covered) / full_marker_coverage * 100 = ink_remaining_percentage :=
by sorry

end ink_remaining_proof_l1947_194755


namespace johnnys_walk_legs_l1947_194715

/-- The number of legs for a given organism type -/
def legs_count (organism : String) : ℕ :=
  match organism with
  | "human" => 2
  | "dog" => 4
  | _ => 0

/-- The total number of legs for a group of organisms -/
def total_legs (humans : ℕ) (dogs : ℕ) : ℕ :=
  humans * legs_count "human" + dogs * legs_count "dog"

/-- Theorem stating that the total number of legs in Johnny's walking group is 12 -/
theorem johnnys_walk_legs :
  let humans : ℕ := 2  -- Johnny and his son
  let dogs : ℕ := 2    -- Johnny's two dogs
  total_legs humans dogs = 12 := by
  sorry

end johnnys_walk_legs_l1947_194715


namespace required_plane_satisfies_conditions_l1947_194756

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The given plane equation -/
def givenPlane : PlaneEquation := { A := 2, B := -1, C := 4, D := -7 }

/-- The two points that the required plane passes through -/
def point1 : Point3D := { x := 2, y := -1, z := 0 }
def point2 : Point3D := { x := 0, y := 3, z := 1 }

/-- The equation of the required plane -/
def requiredPlane : PlaneEquation := { A := 17, B := 10, C := -6, D := -24 }

/-- Function to check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Function to check if two planes are perpendicular -/
def arePlanesPerp (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- Theorem stating that the required plane satisfies all conditions -/
theorem required_plane_satisfies_conditions :
  satisfiesPlaneEquation point1 requiredPlane ∧
  satisfiesPlaneEquation point2 requiredPlane ∧
  arePlanesPerp requiredPlane givenPlane ∧
  requiredPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs requiredPlane.A) (Int.natAbs requiredPlane.B)) (Int.natAbs requiredPlane.C)) (Int.natAbs requiredPlane.D) = 1 :=
by sorry


end required_plane_satisfies_conditions_l1947_194756


namespace tangent_line_of_cubic_with_even_derivative_l1947_194736

/-- The tangent line equation for a cubic function with specific properties -/
theorem tangent_line_of_cubic_with_even_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a - 3))
  (∀ x, f' x = f' (-x)) →
  (λ x ↦ -3*x) = (λ x ↦ f' 0 * x) :=
by sorry

end tangent_line_of_cubic_with_even_derivative_l1947_194736


namespace floor_ceiling_solution_l1947_194713

theorem floor_ceiling_solution (c : ℝ) : 
  (∃ (n : ℤ), n = ⌊c⌋ ∧ 3 * (n : ℝ)^2 + 8 * (n : ℝ) - 35 = 0) ∧
  (let frac := c - ⌊c⌋; 4 * frac^2 - 12 * frac + 5 = 0 ∧ 0 ≤ frac ∧ frac < 1) →
  c = -9/2 := by
sorry

end floor_ceiling_solution_l1947_194713


namespace otimes_h_h_otimes_h_h_l1947_194775

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem otimes_h_h_otimes_h_h (h : ℝ) : otimes (otimes h h) (otimes h h) = 8 * h^4 := by
  sorry

end otimes_h_h_otimes_h_h_l1947_194775


namespace parabola_c_value_l1947_194710

theorem parabola_c_value (b c : ℝ) : 
  (2^2 + 2*b + c = 10) → 
  (4^2 + 4*b + c = 31) → 
  c = -3 := by
sorry

end parabola_c_value_l1947_194710


namespace parallel_vectors_t_value_l1947_194746

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ → ℝ × ℝ := fun t ↦ (-4, t)
  ∀ t : ℝ, parallel m (n t) → t = -16 := by
sorry

end parallel_vectors_t_value_l1947_194746


namespace regular_polygon_exterior_angle_18_has_20_sides_l1947_194759

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 := by
sorry

end regular_polygon_exterior_angle_18_has_20_sides_l1947_194759


namespace derivative_F_at_one_l1947_194749

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^3 - 1) + f (1 - x^3)

theorem derivative_F_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  deriv (F f) 1 = 0 := by
  sorry

end derivative_F_at_one_l1947_194749


namespace segment_ratio_l1947_194731

/-- Given five consecutive points on a line, prove that the ratio of two specific segments is 2:1 -/
theorem segment_ratio (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (d - e = 4) ∧        -- de = 4
  (a - b = 5) ∧        -- ab = 5
  (a - c = 11) ∧       -- ac = 11
  (a - e = 18) →       -- ae = 18
  (c - b) / (d - c) = 2 / 1 := by
sorry

end segment_ratio_l1947_194731


namespace intersection_point_unique_l1947_194754

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 8 = (y - 8) / (-5) ∧ (x - 1) / 8 = (z + 5) / 12

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  x - 2*y - 3*z + 18 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (9, 3, 7)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point := by
  sorry

end intersection_point_unique_l1947_194754


namespace cos_2alpha_value_l1947_194727

theorem cos_2alpha_value (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, Real.sqrt 2 / 2) →
  ‖a‖ = Real.sqrt 3 / 2 →
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_2alpha_value_l1947_194727


namespace negation_existence_divisibility_l1947_194724

theorem negation_existence_divisibility :
  (¬ ∃ n : ℕ+, 10 ∣ (n^2 + 3*n)) ↔ (∀ n : ℕ+, ¬(10 ∣ (n^2 + 3*n))) := by
  sorry

end negation_existence_divisibility_l1947_194724


namespace axis_of_symmetry_l1947_194741

/-- For a quadratic function y = ax^2 - 4ax + 1 where a ≠ 0, the axis of symmetry is x = 2 -/
theorem axis_of_symmetry (a : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 4 * a * x + 1
  (∀ x : ℝ, f (2 + x) = f (2 - x)) := by
sorry


end axis_of_symmetry_l1947_194741


namespace april_roses_unsold_l1947_194748

/-- Calculates the number of roses left unsold after a sale -/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Proves that the number of roses left unsold is 4 -/
theorem april_roses_unsold :
  roses_left_unsold 13 4 36 = 4 := by
  sorry

end april_roses_unsold_l1947_194748


namespace dessert_probability_l1947_194762

theorem dessert_probability (p_dessert : ℝ) (p_dessert_no_coffee : ℝ) :
  p_dessert = 0.6 →
  p_dessert_no_coffee = 0.2 * p_dessert →
  1 - p_dessert = 0.4 := by
sorry

end dessert_probability_l1947_194762


namespace fraction_inequality_l1947_194704

theorem fraction_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 3 > 9 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end fraction_inequality_l1947_194704


namespace fraction_equals_zero_l1947_194725

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end fraction_equals_zero_l1947_194725


namespace biathlon_run_distance_l1947_194740

/-- A biathlon consisting of a bicycle race and a running race. -/
structure Biathlon where
  total_distance : ℝ
  bicycle_distance : ℝ
  run_velocity : ℝ
  bicycle_velocity : ℝ

/-- The theorem stating that for a specific biathlon, the running distance is 10 miles. -/
theorem biathlon_run_distance (b : Biathlon) 
  (h1 : b.total_distance = 155) 
  (h2 : b.bicycle_distance = 145) 
  (h3 : b.run_velocity = 10)
  (h4 : b.bicycle_velocity = 29) : 
  b.total_distance - b.bicycle_distance = 10 := by
  sorry

end biathlon_run_distance_l1947_194740


namespace inversion_similarity_l1947_194794

/-- Inversion of a point with respect to a circle -/
def inversion (O : ℝ × ℝ) (R : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Similarity of triangles -/
def triangles_similar (A B C D E F : ℝ × ℝ) : Prop := sorry

theorem inversion_similarity 
  (O A B : ℝ × ℝ) 
  (R : ℝ) 
  (A' B' : ℝ × ℝ) 
  (h1 : A' = inversion O R A) 
  (h2 : B' = inversion O R B) : 
  triangles_similar O A B B' O A' := 
sorry

end inversion_similarity_l1947_194794


namespace meal_combinations_count_l1947_194792

/-- The number of main dishes available on the menu -/
def num_main_dishes : ℕ := 12

/-- The number of appetizers available to choose from -/
def num_appetizers : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Calculates the number of different meal combinations -/
def meal_combinations : ℕ := num_main_dishes ^ num_people * num_appetizers

/-- Theorem stating that the number of meal combinations is 720 -/
theorem meal_combinations_count : meal_combinations = 720 := by sorry

end meal_combinations_count_l1947_194792


namespace range_of_a_l1947_194703

-- Define the sets A and B
def A : Set ℝ := {0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a > 0 := by
  sorry

end range_of_a_l1947_194703


namespace third_roll_probability_l1947_194772

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 3
def biased_die_prob : ℚ := 2 / 3

-- Define the probability of rolling sixes or fives twice for each die
def fair_die_two_rolls : ℚ := fair_die_prob ^ 2
def biased_die_two_rolls : ℚ := biased_die_prob ^ 2

-- Define the normalized probabilities of using each die given the first two rolls
def prob_fair_die : ℚ := fair_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)
def prob_biased_die : ℚ := biased_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)

-- Theorem: The probability of rolling a six or five on the third roll is 3/5
theorem third_roll_probability : 
  prob_fair_die * fair_die_prob + prob_biased_die * biased_die_prob = 3 / 5 :=
by sorry

end third_roll_probability_l1947_194772


namespace arithmetic_sequence_problem_l1947_194795

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 6 + a 11 = 100) : 
  2 * a 7 - a 8 = 20 := by
  sorry

end arithmetic_sequence_problem_l1947_194795


namespace other_x_intercept_is_negative_one_l1947_194774

/-- A quadratic function with vertex (h, k) and one x-intercept at (r, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  k : ℝ
  r : ℝ
  vertex_x : h = 2
  vertex_y : k = -3
  intercept : r = 5

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 2 * f.h - f.r

theorem other_x_intercept_is_negative_one (f : QuadraticFunction) :
  other_x_intercept f = -1 := by
  sorry

end other_x_intercept_is_negative_one_l1947_194774


namespace monotonic_increasing_condition_l1947_194785

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 - 2*x else a*x - 1

/-- Proposition: If f is monotonically increasing on ℝ, then 0 < a ≤ 1/2 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end monotonic_increasing_condition_l1947_194785


namespace box_width_is_twenty_l1947_194723

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling a box -/
structure CubeFill where
  box : BoxDimensions
  cubeCount : ℕ
  cubeSideLength : ℕ

/-- Theorem stating that a box with given dimensions filled with 56 cubes has a width of 20 inches -/
theorem box_width_is_twenty
  (box : BoxDimensions)
  (fill : CubeFill)
  (h1 : box.length = 35)
  (h2 : box.depth = 10)
  (h3 : fill.box = box)
  (h4 : fill.cubeCount = 56)
  (h5 : fill.cubeSideLength * fill.cubeSideLength * fill.cubeSideLength * fill.cubeCount = box.length * box.width * box.depth)
  (h6 : fill.cubeSideLength ∣ box.length ∧ fill.cubeSideLength ∣ box.width ∧ fill.cubeSideLength ∣ box.depth)
  : box.width = 20 := by
  sorry

#check box_width_is_twenty

end box_width_is_twenty_l1947_194723


namespace partnership_profit_calculation_l1947_194706

/-- Calculates the total profit given investments and one partner's profit share -/
def calculate_total_profit (investment_A investment_B investment_C profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_A + investment_B + investment_C
  (profit_share_A * total_investment) / investment_A

theorem partnership_profit_calculation 
  (investment_A investment_B investment_C profit_share_A : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3630) :
  calculate_total_profit investment_A investment_B investment_C profit_share_A = 12100 := by
  sorry

end partnership_profit_calculation_l1947_194706


namespace nested_fraction_value_l1947_194770

theorem nested_fraction_value : 1 + 2 / (3 + 4/5) = 29/19 := by
  sorry

end nested_fraction_value_l1947_194770


namespace forty_percent_of_jacquelines_candy_bars_l1947_194719

def fred_candy_bars : ℕ := 12
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6
def total_fred_and_bob : ℕ := fred_candy_bars + uncle_bob_candy_bars
def jacqueline_candy_bars : ℕ := 10 * total_fred_and_bob

theorem forty_percent_of_jacquelines_candy_bars :
  (40 : ℚ) / 100 * jacqueline_candy_bars = 120 := by
  sorry

end forty_percent_of_jacquelines_candy_bars_l1947_194719


namespace angle_relation_l1947_194745

theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1/2) (h6 : Real.cos β = -7 * Real.sqrt 2 / 10) :
  2 * α - β = -3 * π / 4 := by
  sorry

end angle_relation_l1947_194745


namespace net_population_increase_l1947_194771

/-- Calculates the net population increase given birth, immigration, emigration, death rate, and initial population. -/
theorem net_population_increase
  (births : ℕ)
  (immigrants : ℕ)
  (emigrants : ℕ)
  (death_rate : ℚ)
  (initial_population : ℕ)
  (h_births : births = 90171)
  (h_immigrants : immigrants = 16320)
  (h_emigrants : emigrants = 8212)
  (h_death_rate : death_rate = 8 / 10000)
  (h_initial_population : initial_population = 2876543) :
  (births + immigrants) - (emigrants + Int.floor (death_rate * initial_population)) = 96078 :=
by sorry

end net_population_increase_l1947_194771


namespace largest_expression_l1947_194796

theorem largest_expression : 
  let a := 15847
  let b := 3174
  let expr1 := a + 1 / b
  let expr2 := a - 1 / b
  let expr3 := a * (1 / b)
  let expr4 := a / (1 / b)
  let expr5 := a ^ 1.03174
  (expr4 > expr1) ∧ 
  (expr4 > expr2) ∧ 
  (expr4 > expr3) ∧ 
  (expr4 > expr5) := by
  sorry

end largest_expression_l1947_194796


namespace base_8_addition_problem_l1947_194728

/-- Converts a base 8 digit to base 10 -/
def to_base_10 (d : Nat) : Nat :=
  d

/-- Converts a base 10 number to base 8 -/
def to_base_8 (n : Nat) : Nat :=
  n

theorem base_8_addition_problem (X Y : Nat) 
  (h1 : X < 8 ∧ Y < 8)  -- X and Y are single digits in base 8
  (h2 : to_base_8 (4 * 8 * 8 + X * 8 + Y) + to_base_8 (5 * 8 + 3) = to_base_8 (6 * 8 * 8 + 1 * 8 + X)) :
  to_base_10 X + to_base_10 Y = 5 := by
  sorry

end base_8_addition_problem_l1947_194728


namespace trajectory_and_slope_product_l1947_194787

-- Define the points and the trajectory
def A : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (1, 2)
def P : ℝ × ℝ := (0, -2)

def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1 ∧ p.2 ≠ 0}

-- Define the conditions
structure Triangle (A B C : ℝ × ℝ) : Prop where
  b_on_x_axis : B.2 = 0
  equal_sides : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2
  midpoint_on_y : B.1 + C.1 = 0

-- Define the theorem
theorem trajectory_and_slope_product 
  (B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hC : C ∈ Γ) 
  (l : Set (ℝ × ℝ)) 
  (hl : P ∈ l) 
  (M N : ℝ × ℝ) 
  (hM : M ∈ l ∩ Γ) 
  (hN : N ∈ l ∩ Γ) 
  (hMN : M ≠ N) :
  -- Part I: C satisfies the equation of Γ
  C.2 ^ 2 = 4 * C.1 ∧ 
  -- Part II: Product of slopes is constant
  (M.2 - Q.2) / (M.1 - Q.1) * (N.2 - Q.2) / (N.1 - Q.1) = 4 := by
  sorry

end trajectory_and_slope_product_l1947_194787


namespace minimum_square_formation_l1947_194711

theorem minimum_square_formation :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (∃ (m : ℕ), 11*n + 1 = m^2) ∧
    (∀ (x : ℕ), x < n → ¬(∃ (j : ℕ), x = j^2) ∨ ¬(∃ (l : ℕ), 11*x + 1 = l^2)) :=
by
  -- The proof would go here
  sorry

end minimum_square_formation_l1947_194711


namespace mirror_area_l1947_194720

theorem mirror_area (frame_length frame_width frame_side_width : ℝ) 
  (h1 : frame_length = 80)
  (h2 : frame_width = 60)
  (h3 : frame_side_width = 10) :
  (frame_length - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 2400 :=
by sorry

end mirror_area_l1947_194720


namespace percent_decrease_proof_l1947_194718

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end percent_decrease_proof_l1947_194718


namespace solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l1947_194783

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x + 2|

-- Part 1
theorem solution_set_for_a_eq_neg_one (x : ℝ) :
  (f (-1) x ≥ x + 5) ↔ (x ≤ -2 ∨ x ≥ 4) := by sorry

-- Part 2
theorem range_of_a_for_inequality (a : ℝ) (h : a < 2) :
  (∀ x ∈ Set.Ioo (-5) (-3), f a x > x^2 + 2*x - 5) ↔ (a ≤ -2) := by sorry

end solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l1947_194783


namespace paco_salty_cookies_left_l1947_194747

/-- The number of salty cookies Paco had left -/
def salty_cookies_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

theorem paco_salty_cookies_left :
  salty_cookies_left 26 9 = 17 := by
  sorry

end paco_salty_cookies_left_l1947_194747


namespace student_team_repetition_l1947_194797

theorem student_team_repetition (n : ℕ) (h : n > 0) :
  ∀ (arrangement : ℕ → Fin (n^2) → Fin n),
  ∃ (week1 week2 : ℕ) (student1 student2 : Fin (n^2)),
    week1 < week2 ∧ week2 ≤ n + 2 ∧
    student1 ≠ student2 ∧
    arrangement week1 student1 = arrangement week1 student2 ∧
    arrangement week2 student1 = arrangement week2 student2 :=
by sorry

end student_team_repetition_l1947_194797


namespace sum_of_digits_square_l1947_194716

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: A positive integer equals the square of the sum of its digits if and only if it's 1 or 81 -/
theorem sum_of_digits_square (n : ℕ+) : n = (sum_of_digits n)^2 ↔ n = 1 ∨ n = 81 := by
  sorry

end sum_of_digits_square_l1947_194716


namespace shifted_parabola_passes_through_point_l1947_194750

/-- The original parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  g (-1) = 1 := by sorry

end shifted_parabola_passes_through_point_l1947_194750


namespace square_root_equation_l1947_194717

theorem square_root_equation (y : ℝ) : 
  Real.sqrt (9 + Real.sqrt (4 * y - 5)) = Real.sqrt 10 → y = (3/2 : ℝ) := by
  sorry

end square_root_equation_l1947_194717


namespace johanna_turtle_loss_l1947_194733

/-- The fraction of turtles Johanna loses -/
def johanna_loss_fraction (owen_initial : ℕ) (johanna_diff : ℕ) (owen_final : ℕ) : ℚ :=
  let owen_after_month := 2 * owen_initial
  let johanna_initial := owen_initial - johanna_diff
  1 - (owen_final - owen_after_month) / johanna_initial

theorem johanna_turtle_loss 
  (owen_initial : ℕ) 
  (johanna_diff : ℕ) 
  (owen_final : ℕ) 
  (h1 : owen_initial = 21)
  (h2 : johanna_diff = 5)
  (h3 : owen_final = 50) :
  johanna_loss_fraction owen_initial johanna_diff owen_final = 1/2 := by
  sorry

#eval johanna_loss_fraction 21 5 50

end johanna_turtle_loss_l1947_194733


namespace jeremy_dosage_l1947_194786

/-- Represents the duration of Jeremy's medication course in weeks -/
def duration : ℕ := 2

/-- Represents the number of pills Jeremy takes in total -/
def total_pills : ℕ := 112

/-- Represents the dosage of each pill in milligrams -/
def pill_dosage : ℕ := 500

/-- Represents the interval between doses in hours -/
def dose_interval : ℕ := 6

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the total milligrams of medication taken over the entire course -/
def total_mg : ℕ := total_pills * pill_dosage

/-- Calculates the number of doses taken per day -/
def doses_per_day : ℕ := hours_per_day / dose_interval

/-- Calculates the total number of doses taken over the entire course -/
def total_doses : ℕ := duration * 7 * doses_per_day

/-- Theorem stating that Jeremy takes 1000 mg every 6 hours -/
theorem jeremy_dosage : total_mg / total_doses = 1000 := by
  sorry

end jeremy_dosage_l1947_194786
