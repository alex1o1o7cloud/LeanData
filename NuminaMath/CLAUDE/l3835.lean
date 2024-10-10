import Mathlib

namespace area_ratio_GHI_JKL_l3835_383579

/-- Triangle GHI with sides 7, 24, and 25 -/
def triangle_GHI : Set (ℝ × ℝ) := sorry

/-- Triangle JKL with sides 9, 40, and 41 -/
def triangle_JKL : Set (ℝ × ℝ) := sorry

/-- Area of a triangle -/
def area (triangle : Set (ℝ × ℝ)) : ℝ := sorry

/-- The ratio of the areas of triangle GHI to triangle JKL is 7/15 -/
theorem area_ratio_GHI_JKL : 
  (area triangle_GHI) / (area triangle_JKL) = 7 / 15 := by sorry

end area_ratio_GHI_JKL_l3835_383579


namespace pencils_per_box_l3835_383505

/-- Given Arnel's pencil distribution scenario, prove that each box contains 5 pencils. -/
theorem pencils_per_box (num_boxes : ℕ) (num_friends : ℕ) (pencils_kept : ℕ) (pencils_per_friend : ℕ) :
  num_boxes = 10 →
  num_friends = 5 →
  pencils_kept = 10 →
  pencils_per_friend = 8 →
  (∃ (pencils_per_box : ℕ), 
    pencils_per_box * num_boxes = pencils_kept + num_friends * pencils_per_friend ∧
    pencils_per_box = 5) :=
by sorry

end pencils_per_box_l3835_383505


namespace max_value_7b_plus_5c_l3835_383593

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_value_7b_plus_5c :
  ∀ a b c : ℝ,
  (∃ a' : ℝ, a' ∈ Set.Icc 1 2 ∧
    (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a' b c x ≤ 1)) →
  (∀ k : ℝ, 7 * b + 5 * c ≤ k) →
  k = -6 :=
sorry

end max_value_7b_plus_5c_l3835_383593


namespace intersection_A_B_subset_A_C_iff_a_in_0_2_l3835_383519

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def B : Set ℝ := {x | (x-1)/(x-3) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a+4)*x + a^2 + 4*a ≤ 0}

-- Define the interval (3, 4]
def interval_3_4 : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- Theorem statements
theorem intersection_A_B : A ∩ B = interval_3_4 := by sorry

theorem subset_A_C_iff_a_in_0_2 :
  ∀ a : ℝ, A ⊆ C a ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end intersection_A_B_subset_A_C_iff_a_in_0_2_l3835_383519


namespace gemma_change_is_five_l3835_383533

-- Define the given conditions
def number_of_pizzas : ℕ := 4
def price_per_pizza : ℕ := 10
def tip_amount : ℕ := 5
def payment_amount : ℕ := 50

-- Define the function to calculate the change
def calculate_change (pizzas : ℕ) (price : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (pizzas * price + tip)

-- Theorem statement
theorem gemma_change_is_five :
  calculate_change number_of_pizzas price_per_pizza tip_amount payment_amount = 5 := by
  sorry

end gemma_change_is_five_l3835_383533


namespace sum_and_reciprocal_square_integer_l3835_383556

theorem sum_and_reciprocal_square_integer (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m :=
sorry

end sum_and_reciprocal_square_integer_l3835_383556


namespace blue_parrots_count_l3835_383582

theorem blue_parrots_count (total_parrots : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total_parrots = 92 →
  green_fraction = 3/4 →
  blue_parrots = total_parrots - (green_fraction * total_parrots).num →
  blue_parrots = 23 := by
sorry

end blue_parrots_count_l3835_383582


namespace gcf_of_180_252_315_l3835_383546

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by sorry

end gcf_of_180_252_315_l3835_383546


namespace water_flow_rates_verify_conditions_l3835_383506

/-- Represents the water flow model with introducing and removing pipes -/
structure WaterFlowModel where
  /-- Water flow rate of one introducing pipe in m³/h -/
  inlet_rate : ℝ
  /-- Water flow rate of one removing pipe in m³/h -/
  outlet_rate : ℝ

/-- Theorem stating the correct water flow rates given the problem conditions -/
theorem water_flow_rates (model : WaterFlowModel) : 
  (5 * (4 * model.inlet_rate - 3 * model.outlet_rate) = 1000) ∧ 
  (2 * (2 * model.inlet_rate - 2 * model.outlet_rate) = 180) →
  model.inlet_rate = 65 ∧ model.outlet_rate = 20 := by
  sorry

/-- Function to calculate the net water gain in a given time period -/
def net_water_gain (model : WaterFlowModel) (inlet_count outlet_count : ℕ) (hours : ℝ) : ℝ :=
  hours * (inlet_count * model.inlet_rate - outlet_count * model.outlet_rate)

/-- Verifies that the calculated rates satisfy the given conditions -/
theorem verify_conditions (model : WaterFlowModel) 
  (h1 : model.inlet_rate = 65) 
  (h2 : model.outlet_rate = 20) : 
  net_water_gain model 4 3 5 = 1000 ∧ 
  net_water_gain model 2 2 2 = 180 := by
  sorry

end water_flow_rates_verify_conditions_l3835_383506


namespace school_event_ticket_revenue_l3835_383511

theorem school_event_ticket_revenue :
  ∀ (f h : ℕ) (p : ℚ),
    f + h = 160 →
    f * p + h * (p / 2) = 2400 →
    f * p = 800 :=
by sorry

end school_event_ticket_revenue_l3835_383511


namespace remainder_problem_l3835_383597

theorem remainder_problem (x y : ℕ) (hx : x > 0) (hy : y ≥ 0)
  (h1 : ∃ r, x ≡ r [MOD 11] ∧ 0 ≤ r ∧ r < 11)
  (h2 : 2 * x ≡ 1 [MOD 6])
  (h3 : 3 * y = (2 * x) / 6)
  (h4 : 7 * y - x = 3) :
  ∃ r, x ≡ r [MOD 11] ∧ r = 4 := by
sorry

end remainder_problem_l3835_383597


namespace reciprocal_problem_l3835_383526

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 16) : 200 * (1 / x) = 100 := by
  sorry

end reciprocal_problem_l3835_383526


namespace total_books_is_54_l3835_383530

def darla_books : ℕ := 6

def katie_books : ℕ := darla_books / 2

def darla_katie_books : ℕ := darla_books + katie_books

def gary_books : ℕ := 5 * darla_katie_books

def total_books : ℕ := darla_books + katie_books + gary_books

theorem total_books_is_54 : total_books = 54 := by
  sorry

end total_books_is_54_l3835_383530


namespace both_questions_correct_l3835_383584

/-- Represents a class of students and their test results. -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly. -/
def both_correct (c : ClassTestResults) : ℕ :=
  c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that given the specific class conditions, 
    22 students answered both questions correctly. -/
theorem both_questions_correct 
  (c : ClassTestResults) 
  (h1 : c.total_students = 30)
  (h2 : c.correct_q1 = 25)
  (h3 : c.correct_q2 = 22)
  (h4 : c.absent = 5) :
  both_correct c = 22 := by
  sorry

#eval both_correct ⟨30, 25, 22, 5⟩

end both_questions_correct_l3835_383584


namespace surface_area_five_cube_removal_l3835_383539

/-- The surface area of a cube after removing central columns -/
def surface_area_after_removal (n : ℕ) : ℕ :=
  let original_surface_area := 6 * n^2
  let removed_surface_area := 6 * (n^2 - 1)
  let added_internal_surface := 2 * 3 * 4 * (n - 1)
  removed_surface_area + added_internal_surface

/-- Theorem stating that the surface area of a 5×5×5 cube after removing central columns is 192 -/
theorem surface_area_five_cube_removal :
  surface_area_after_removal 5 = 192 := by
  sorry

#eval surface_area_after_removal 5

end surface_area_five_cube_removal_l3835_383539


namespace distance_after_ten_reflections_l3835_383572

/-- Represents a circular billiard table with a ball's trajectory -/
structure BilliardTable where
  radius : ℝ
  p_distance : ℝ  -- Distance of point P from the center
  reflection_angle : ℝ  -- Angle of reflection

/-- Calculates the distance between P and the ball's position after n reflections -/
noncomputable def distance_after_reflections (table : BilliardTable) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the distance after 10 reflections for the given table -/
theorem distance_after_ten_reflections (table : BilliardTable) 
  (h1 : table.radius = 1)
  (h2 : table.p_distance = 0.4)
  (h3 : table.reflection_angle = Real.arcsin ((Real.sqrt 57 - 5) / 8)) :
  ∃ (ε : ℝ), abs (distance_after_reflections table 10 - 0.0425) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
sorry

end distance_after_ten_reflections_l3835_383572


namespace reciprocal_solutions_imply_m_value_l3835_383592

theorem reciprocal_solutions_imply_m_value (m : ℝ) : 
  (∃ x y : ℝ, 6 * x + 3 = 0 ∧ 3 * y + m = 15 ∧ x * y = 1) → m = 21 := by
  sorry

end reciprocal_solutions_imply_m_value_l3835_383592


namespace solve_investment_problem_l3835_383581

def investment_problem (total_investment : ℝ) (first_account_investment : ℝ) 
  (second_account_rate : ℝ) (total_interest : ℝ) : Prop :=
  let second_account_investment := total_investment - first_account_investment
  let first_account_rate := (total_interest - (second_account_investment * second_account_rate)) / first_account_investment
  first_account_rate = 0.08

theorem solve_investment_problem : 
  investment_problem 8000 3000 0.05 490 := by
  sorry

end solve_investment_problem_l3835_383581


namespace angle_quadrant_l3835_383560

theorem angle_quadrant (θ : Real) : 
  (Real.sin θ * Real.cos θ > 0) → 
  (0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2) ∨
  (Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2) := by
  sorry

end angle_quadrant_l3835_383560


namespace sum_of_symmetric_roots_l3835_383598

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x, f (3 + x) = f (3 - x)) 
  (h_roots : ∃! (roots : Finset ℝ), roots.card = 6 ∧ ∀ x ∈ roots, f x = 0) : 
  ∃ (roots : Finset ℝ), roots.card = 6 ∧ (∀ x ∈ roots, f x = 0) ∧ (roots.sum id = 18) := by
sorry

end sum_of_symmetric_roots_l3835_383598


namespace smores_graham_crackers_per_smore_l3835_383576

theorem smores_graham_crackers_per_smore (total_graham_crackers : ℕ) 
  (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) :
  total_graham_crackers = 48 →
  initial_marshmallows = 6 →
  additional_marshmallows = 18 →
  (total_graham_crackers / (initial_marshmallows + additional_marshmallows) : ℕ) = 2 := by
  sorry

end smores_graham_crackers_per_smore_l3835_383576


namespace right_triangle_area_l3835_383562

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 20 →  -- hypotenuse is 20 inches
  θ = π / 6 →  -- one angle is 30° (π/6 radians)
  area = 50 * Real.sqrt 3 →  -- area is 50√3 square inches
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧  -- Pythagorean theorem
    a * b / 2 = area ∧  -- area formula for a triangle
    Real.sin θ = a / h  -- trigonometric relation
  := by sorry

end right_triangle_area_l3835_383562


namespace rectangles_in_grid_l3835_383535

def grid_size : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in an n x n grid -/
def num_rectangles (n : ℕ) : ℕ := (choose_two n) ^ 2

theorem rectangles_in_grid :
  num_rectangles grid_size = 100 :=
by sorry

end rectangles_in_grid_l3835_383535


namespace azar_winning_configurations_l3835_383573

/-- Represents a tic-tac-toe board configuration -/
def TicTacToeBoard := List (Option Bool)

/-- Checks if a given board configuration is valid according to the game rules -/
def is_valid_board (board : TicTacToeBoard) : Bool :=
  board.length = 9 ∧ 
  (board.filter (· = some true)).length = 4 ∧
  (board.filter (· = some false)).length = 3

/-- Checks if Azar (X) has won in the given board configuration -/
def azar_wins (board : TicTacToeBoard) : Bool :=
  sorry

/-- Counts the number of valid winning configurations for Azar -/
def count_winning_configurations : Nat :=
  sorry

theorem azar_winning_configurations : 
  count_winning_configurations = 100 := by sorry

end azar_winning_configurations_l3835_383573


namespace quadratic_inequality_solution_derived_inequality_solutions_l3835_383529

def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

def solution_set (x : ℝ) : Prop := x < 1 ∨ x > 2

def derived_inequality (x m : ℝ) : Prop := x^2 - (m + 2)*x + 2*m < 0

theorem quadratic_inequality_solution :
  ∀ x, quadratic_inequality x ↔ solution_set x :=
sorry

theorem derived_inequality_solutions :
  (∀ x, ¬(derived_inequality x 2)) ∧
  (∀ m, m < 2 → ∀ x, derived_inequality x m ↔ m < x ∧ x < 2) ∧
  (∀ m, m > 2 → ∀ x, derived_inequality x m ↔ 2 < x ∧ x < m) :=
sorry

end quadratic_inequality_solution_derived_inequality_solutions_l3835_383529


namespace arithmetic_mean_of_special_set_l3835_383509

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := {1 - 1 / n, 1 + 1 / n^2} ∪ Finset.range (n - 2)
  (Finset.sum set id) / n = 1 - 1 / n^2 + 1 / n^3 := by
  sorry

end arithmetic_mean_of_special_set_l3835_383509


namespace polynomial_evaluation_l3835_383578

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 → x^3 - 3*x^2 - 9*x + 5 = 10 := by
  sorry

end polynomial_evaluation_l3835_383578


namespace problem_statement_l3835_383508

theorem problem_statement (w x y : ℝ) 
  (h1 : 6/w + 6/x = 6/y) 
  (h2 : w*x = y) 
  (h3 : (w + x)/2 = 0.5) : 
  y = 0.25 := by
sorry

end problem_statement_l3835_383508


namespace nested_fraction_equality_l3835_383548

theorem nested_fraction_equality : 
  (1 : ℚ) / (2 - 1 / (2 - 1 / (2 - 1 / 2))) = 3 / 4 := by sorry

end nested_fraction_equality_l3835_383548


namespace f_neg_a_value_l3835_383570

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

theorem f_neg_a_value (a : ℝ) (h : f a = 4) : f (-a) = -2 := by
  sorry

end f_neg_a_value_l3835_383570


namespace smallest_solution_of_equation_l3835_383516

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = -3 ∧ 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 12) ∧
    (∀ y : ℝ, (3 * y / (y - 3) + (3 * y^2 - 27) / y = 12) → y ≥ x) := by
  sorry

end smallest_solution_of_equation_l3835_383516


namespace prob_our_team_l3835_383568

/-- A sports team with boys, girls, and Alice -/
structure Team where
  total : ℕ
  boys : ℕ
  girls : ℕ
  has_alice : Bool

/-- Definition of our specific team -/
def our_team : Team :=
  { total := 12
  , boys := 7
  , girls := 5
  , has_alice := true
  }

/-- The probability of choosing two girls, one of whom is Alice -/
def prob_two_girls_with_alice (t : Team) : ℚ :=
  if t.has_alice then
    (t.girls - 1 : ℚ) / (t.total.choose 2 : ℚ)
  else
    0

/-- Theorem stating the probability for our specific team -/
theorem prob_our_team :
  prob_two_girls_with_alice our_team = 2 / 33 := by
  sorry


end prob_our_team_l3835_383568


namespace bus_seat_difference_l3835_383541

theorem bus_seat_difference :
  let left_seats : ℕ := 15
  let seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 9
  let total_capacity : ℕ := 90
  let right_seats : ℕ := (total_capacity - (left_seats * seat_capacity + back_seat_capacity)) / seat_capacity
  left_seats - right_seats = 3 := by
  sorry

end bus_seat_difference_l3835_383541


namespace min_xy_over_x2_plus_y2_l3835_383557

theorem min_xy_over_x2_plus_y2 (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 1) (hy : 2/5 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≥ 1/2 ∧ ∃ (x₀ y₀ : ℝ), 1/2 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2/5 ≤ y₀ ∧ y₀ ≤ 1/2 ∧ x₀ * y₀ / (x₀^2 + y₀^2) = 1/2 :=
by sorry

end min_xy_over_x2_plus_y2_l3835_383557


namespace geometry_theorem_l3835_383596

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lines_parallel : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (subset m β ∧ parallel α β → line_parallel m α) ∧
  (perpendicular m α ∧ perpendicular n β ∧ parallel α β → lines_parallel m n) :=
sorry

end geometry_theorem_l3835_383596


namespace soda_price_increase_l3835_383521

theorem soda_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 15 ∧ increase_percentage = 50 ∧ new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 10 := by
sorry

end soda_price_increase_l3835_383521


namespace ten_player_tournament_matches_l3835_383534

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry

end ten_player_tournament_matches_l3835_383534


namespace circles_internally_tangent_with_one_common_tangent_l3835_383565

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12*y + 4 = 0

-- Define the centers and radii of the circles
def center_M : ℝ × ℝ := (-1, 2)
def center_N : ℝ × ℝ := (2, 6)
def radius_M : ℝ := 1
def radius_N : ℝ := 6

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Define the common tangent line equation
def common_tangent (x y : ℝ) : Prop := 3*x + 4*y = 0

theorem circles_internally_tangent_with_one_common_tangent :
  (distance_between_centers = radius_N - radius_M) ∧
  (∃! (l : ℝ × ℝ → Prop), ∀ x y, l (x, y) ↔ common_tangent x y) ∧
  (∀ x y, circle_M x y ∧ circle_N x y → common_tangent x y) :=
sorry

end circles_internally_tangent_with_one_common_tangent_l3835_383565


namespace quadratic_solution_l3835_383590

theorem quadratic_solution (b : ℝ) : 
  ((-2 : ℝ)^2 + b * (-2) - 63 = 0) → b = 33.5 := by
  sorry

end quadratic_solution_l3835_383590


namespace equal_even_odd_probability_l3835_383555

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def prob_even : ℚ := 1/2
def prob_odd : ℚ := 1/2

theorem equal_even_odd_probability :
  (num_dice.choose (num_dice / 2)) * (prob_even ^ num_dice) = 35/128 := by
  sorry

end equal_even_odd_probability_l3835_383555


namespace h_value_l3835_383583

/-- The value of h for which the given conditions are satisfied -/
def h : ℝ := 32

/-- The y-coordinate of the first graph -/
def graph1 (x : ℝ) : ℝ := 4 * (x - h)^2 + 4032 - 4 * h^2

/-- The y-coordinate of the second graph -/
def graph2 (x : ℝ) : ℝ := 5 * (x - h)^2 + 5040 - 5 * h^2

theorem h_value :
  (graph1 0 = 4032) ∧
  (graph2 0 = 5040) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph1 x1 = 0 ∧ graph1 x2 = 0) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph2 x1 = 0 ∧ graph2 x2 = 0) :=
by sorry

end h_value_l3835_383583


namespace calculate_expression_l3835_383577

theorem calculate_expression : 5 + 12 / 3 - 2^3 = 1 := by
  sorry

end calculate_expression_l3835_383577


namespace no_periodic_sequence_for_factorial_digits_l3835_383504

/-- a_n is the first non-zero digit from the right in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that for all natural numbers N, the sequence of first non-zero digits
    from the right in the decimal representation of (N+k)! for k ≥ 1 is not periodic. -/
theorem no_periodic_sequence_for_factorial_digits :
  ∀ N : ℕ, ¬ ∃ T : ℕ+, ∀ k : ℕ, first_nonzero_digit (N + k + 1) = first_nonzero_digit (N + k + 1 + T) :=
sorry

end no_periodic_sequence_for_factorial_digits_l3835_383504


namespace smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l3835_383554

/-- A function to check if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function to check if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_five_digit_palindrome_product :
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 171 → isFiveDigitPalindrome (n * 111) :=
by sorry

/-- The result theorem -/
theorem result_171 :
  isThreeDigitPalindrome 171 ∧ ¬ isFiveDigitPalindrome (171 * 111) :=
by sorry

end smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l3835_383554


namespace intersection_M_N_l3835_383510

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_M_N_l3835_383510


namespace vector_subtraction_and_scalar_multiplication_l3835_383500

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 3 → ℝ := ![3, -2, 4]
  let v₂ : Fin 3 → ℝ := ![2, -1, 5]
  v₁ - 3 • v₂ = ![(-3 : ℝ), 1, -11] := by
  sorry

end vector_subtraction_and_scalar_multiplication_l3835_383500


namespace characterization_of_functions_l3835_383594

/-- A function is completely multiplicative if f(xy) = f(x)f(y) for all x, y -/
def CompletelyMultiplicative (f : ℤ → ℕ) : Prop :=
  ∀ x y, f (x * y) = f x * f y

/-- The p-adic valuation of an integer -/
noncomputable def vp (p : ℕ) (x : ℤ) : ℕ := sorry

/-- The main theorem characterizing the required functions -/
theorem characterization_of_functions (f : ℤ → ℕ) : 
  (CompletelyMultiplicative f ∧ 
   ∀ a b : ℤ, b ≠ 0 → ∃ q r : ℤ, a = b * q + r ∧ f r < f b) ↔ 
  (∃ n s : ℕ, ∃ p0 : ℕ, Nat.Prime p0 ∧ 
   ∀ x : ℤ, f x = (Int.natAbs x)^n * s^(vp p0 x)) :=
sorry

end characterization_of_functions_l3835_383594


namespace product_repeating_decimal_nine_l3835_383552

theorem product_repeating_decimal_nine (x : ℚ) : x = 1/3 → x * 9 = 3 := by
  sorry

end product_repeating_decimal_nine_l3835_383552


namespace simplify_square_roots_l3835_383518

theorem simplify_square_roots : 
  Real.sqrt 24 - Real.sqrt 12 + 6 * Real.sqrt (2/3) = 4 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end simplify_square_roots_l3835_383518


namespace league_games_l3835_383502

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 →
  total_games = 760 →
  games_per_matchup = 4 →
  games_per_matchup * (num_teams - 1) * num_teams / 2 = total_games :=
by
  sorry

end league_games_l3835_383502


namespace triangles_in_circle_impossible_l3835_383567

theorem triangles_in_circle_impossible :
  ∀ (A₁ A₂ : ℝ), A₁ > 1 → A₂ > 1 → A₁ + A₂ > π :=
sorry

end triangles_in_circle_impossible_l3835_383567


namespace aquarium_length_l3835_383532

/-- The length of an aquarium given its volume, breadth, and water height -/
theorem aquarium_length (volume : ℝ) (breadth height : ℝ) (h1 : volume = 10000)
  (h2 : breadth = 20) (h3 : height = 10) : volume / (breadth * height) = 50 := by
  sorry

end aquarium_length_l3835_383532


namespace max_non_empty_intersection_l3835_383520

-- Define the set A_n
def A (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- Define the intersection of sets A_1 to A_n
def intersection_up_to (n : ℕ) : Set ℝ := ⋂ i ∈ Finset.range n, A (i + 1)

-- State the theorem
theorem max_non_empty_intersection :
  (∃ (n : ℕ), intersection_up_to n ≠ ∅ ∧
    ∀ (m : ℕ), m > n → intersection_up_to m = ∅) ∧
  (∀ (n : ℕ), intersection_up_to n ≠ ∅ → n ≤ 4) :=
sorry

end max_non_empty_intersection_l3835_383520


namespace polynomial_expansion_problem_l3835_383550

theorem polynomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
sorry

end polynomial_expansion_problem_l3835_383550


namespace complementary_angles_ratio_l3835_383512

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end complementary_angles_ratio_l3835_383512


namespace midpoint_path_area_ratio_l3835_383585

/-- Represents a particle moving along the edges of an equilateral triangle -/
structure Particle where
  position : ℝ × ℝ
  speed : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents the path traced by the midpoint of two particles -/
def MidpointPath (p1 p2 : Particle) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a set of points in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem midpoint_path_area_ratio
  (triangle : EquilateralTriangle)
  (p1 p2 : Particle)
  (h1 : p1.position = triangle.A ∧ p2.position = triangle.B)
  (h2 : p1.speed = p2.speed)
  : (area (MidpointPath p1 p2)) / (area {triangle.A, triangle.B, triangle.C}) = 1/4 :=
sorry

end midpoint_path_area_ratio_l3835_383585


namespace women_in_first_class_l3835_383571

theorem women_in_first_class (total_passengers : ℕ) 
  (percent_women : ℚ) (percent_women_first_class : ℚ) :
  total_passengers = 180 →
  percent_women = 65 / 100 →
  percent_women_first_class = 15 / 100 →
  ⌈(total_passengers : ℚ) * percent_women * percent_women_first_class⌉ = 18 :=
by sorry

end women_in_first_class_l3835_383571


namespace range_of_f_l3835_383588

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_l3835_383588


namespace quadratic_distinct_roots_l3835_383544

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*m + 1)*x₁ + m^2 = 0 ∧ x₂^2 - (2*m + 1)*x₂ + m^2 = 0) ↔ 
  m > -1/4 :=
sorry

end quadratic_distinct_roots_l3835_383544


namespace periodic_even_function_theorem_l3835_383513

def periodic_even_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x, f (-x) = f x)

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic_even : periodic_even_function f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end periodic_even_function_theorem_l3835_383513


namespace range_of_a_l3835_383561

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 4 > 0) → -2 < a ∧ a < 2 := by
  sorry

end range_of_a_l3835_383561


namespace race_probability_l3835_383517

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 8 →
  prob_Y = 1/4 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ prob_X : ℚ, prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 1/2 :=
by sorry

end race_probability_l3835_383517


namespace douglas_county_x_votes_l3835_383503

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_x_percent : ℝ := 64

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_y_percent : ℝ := 46

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℝ := 2

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percent : ℝ := 58

theorem douglas_county_x_votes :
  douglas_x_percent * county_ratio + douglas_y_percent = total_percent * (county_ratio + 1) :=
by sorry

end douglas_county_x_votes_l3835_383503


namespace cube_edge_ratio_l3835_383527

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end cube_edge_ratio_l3835_383527


namespace union_A_B_when_m_4_B_subset_A_iff_m_range_l3835_383591

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Theorem for part (1)
theorem union_A_B_when_m_4 : A ∪ B 4 = {2, 4, 8} := by sorry

-- Theorem for part (2)
theorem B_subset_A_iff_m_range (m : ℝ) : 
  B m ⊆ A ↔ (m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1/2) := by sorry

end union_A_B_when_m_4_B_subset_A_iff_m_range_l3835_383591


namespace hair_color_theorem_l3835_383564

def hair_color_problem (start_age : ℕ) (current_age : ℕ) (future_colors : ℕ) (years_to_future : ℕ) : ℕ :=
  let current_colors := future_colors - years_to_future
  let years_adding_colors := current_age - start_age
  let initial_colors := current_colors - years_adding_colors
  initial_colors + 1

theorem hair_color_theorem :
  hair_color_problem 15 18 8 3 = 3 := by
  sorry

end hair_color_theorem_l3835_383564


namespace inequality_proof_l3835_383559

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 := by
  sorry

end inequality_proof_l3835_383559


namespace x_value_when_y_is_five_l3835_383549

/-- A line in the coordinate plane passing through the origin with slope 1/4 -/
structure Line :=
  (slope : ℚ)
  (passes_origin : Bool)

/-- A point in the coordinate plane -/
structure Point :=
  (x : ℚ)
  (y : ℚ)

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x

theorem x_value_when_y_is_five (k : Line) (p1 p2 : Point) :
  k.slope = 1/4 →
  k.passes_origin = true →
  point_on_line p1 k →
  point_on_line p2 k →
  p1.x * p2.y = 160 →
  p1.y = 8 →
  p2.x = 20 →
  p2.y = 5 →
  p1.x = 32 := by
  sorry

end x_value_when_y_is_five_l3835_383549


namespace clothing_cost_price_l3835_383575

theorem clothing_cost_price (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) : 
  original_price = 132 →
  discount_rate = 0.1 →
  profit_rate = 0.1 →
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) →
  cost_price = 108 := by
sorry

end clothing_cost_price_l3835_383575


namespace union_equality_iff_a_in_range_l3835_383531

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) : 
  A ∪ B a = A ↔ a ∈ Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

end union_equality_iff_a_in_range_l3835_383531


namespace xy_value_l3835_383569

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 126) : x * y = -5 := by
  sorry

end xy_value_l3835_383569


namespace solve_for_y_l3835_383587

theorem solve_for_y (x y : ℤ) (h1 : x^2 - 5*x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end solve_for_y_l3835_383587


namespace point_P_coordinates_l3835_383514

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the conditions
def lies_on_y_axis (P : Point) : Prop :=
  P.x = 0

def parallel_to_x_axis (P Q : Point) : Prop :=
  P.y = Q.y

def equal_distance_to_axes (P : Point) : Prop :=
  |P.x| = |P.y|

-- Main theorem
theorem point_P_coordinates (a : ℝ) :
  let P : Point := ⟨2*a - 2, a + 5⟩
  let Q : Point := ⟨2, 5⟩
  (lies_on_y_axis P ∨ parallel_to_x_axis P Q ∨ equal_distance_to_axes P) →
  (P = ⟨12, 12⟩ ∨ P = ⟨-12, -12⟩ ∨ P = ⟨-4, 4⟩ ∨ P = ⟨4, -4⟩) :=
by
  sorry

end point_P_coordinates_l3835_383514


namespace ceiling_times_self_216_l3835_383523

theorem ceiling_times_self_216 :
  ∃! x : ℝ, ⌈x⌉ * x = 216 ∧ x = 14.4 := by sorry

end ceiling_times_self_216_l3835_383523


namespace relative_prime_theorem_l3835_383589

theorem relative_prime_theorem (u v w : ℤ) :
  (Nat.gcd u.natAbs v.natAbs = 1 ∧ Nat.gcd v.natAbs w.natAbs = 1 ∧ Nat.gcd u.natAbs w.natAbs = 1) ↔
  Nat.gcd (u * v + v * w + w * u).natAbs (u * v * w).natAbs = 1 := by
  sorry

#check relative_prime_theorem

end relative_prime_theorem_l3835_383589


namespace red_toys_after_removal_l3835_383599

/-- Theorem: Number of red toys after removal --/
theorem red_toys_after_removal
  (total : ℕ)
  (h_total : total = 134)
  (red white : ℕ)
  (h_initial : red + white = total)
  (h_after_removal : red - 2 = 2 * white) :
  red - 2 = 88 := by
  sorry

end red_toys_after_removal_l3835_383599


namespace quadratic_function_value_l3835_383574

/-- Given a quadratic function f(x) = x^2 + px + q where f(3) = 0 and f(2) = 0, 
    prove that f(0) = 6. -/
theorem quadratic_function_value (p q : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + p*x + q) 
  (h2 : f 3 = 0) 
  (h3 : f 2 = 0) : 
  f 0 = 6 := by
  sorry

end quadratic_function_value_l3835_383574


namespace log_problem_l3835_383525

theorem log_problem (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = k * y) → 
  k = 9 := by
sorry

end log_problem_l3835_383525


namespace solve_for_b_l3835_383595

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (70/100) * y = (8*y) / b + (3*y) / 10) : b = 20 := by
  sorry

end solve_for_b_l3835_383595


namespace max_value_cubic_quartic_sum_l3835_383543

theorem max_value_cubic_quartic_sum (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_one : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a + b^3 + c^4 = 1 :=
by sorry

end max_value_cubic_quartic_sum_l3835_383543


namespace right_triangle_altitude_l3835_383536

theorem right_triangle_altitude (a b c m : ℝ) (h_positive : a > 0) : 
  b^2 + c^2 = a^2 →   -- Pythagorean theorem
  m^2 = (b - c)^2 →   -- Difference of legs equals altitude
  b * c = a * m →     -- Area relation
  m = (a * (Real.sqrt 5 - 1)) / 2 := by sorry

end right_triangle_altitude_l3835_383536


namespace events_mutually_exclusive_not_complementary_l3835_383586

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "D gets the red card"
def D_gets_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Theorem stating that the events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ D_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ D_gets_red d)) :=
sorry

end events_mutually_exclusive_not_complementary_l3835_383586


namespace painter_job_completion_six_to_four_painters_l3835_383566

/-- The number of work-days required for a group of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : ℚ := painters * days

theorem painter_job_completion 
  (initial_painters : ℕ) 
  (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters > 0 → 
  new_painters > 0 → 
  initial_days > 0 → 
  work_days initial_painters initial_days = work_days new_painters ((initial_painters * initial_days) / new_painters) :=
by
  sorry

theorem six_to_four_painters :
  work_days 6 (14/10) = work_days 4 (21/10) :=
by
  sorry

end painter_job_completion_six_to_four_painters_l3835_383566


namespace expansion_equals_fourth_power_l3835_383553

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end expansion_equals_fourth_power_l3835_383553


namespace water_cost_for_family_of_six_l3835_383522

/-- The cost of fresh water for a family for one day -/
def water_cost (family_size : ℕ) (purification_cost : ℚ) (water_per_person : ℚ) : ℚ :=
  family_size * water_per_person * purification_cost

/-- Proof that the water cost for a family of 6 is $3 -/
theorem water_cost_for_family_of_six :
  water_cost 6 1 (1/2) = 3 := by
  sorry

end water_cost_for_family_of_six_l3835_383522


namespace salt_to_flour_ratio_l3835_383524

/-- Represents the ingredients for making pizza --/
structure PizzaIngredients where
  water : ℕ
  flour : ℕ
  salt : ℕ

/-- Theorem stating the ratio of salt to flour in the pizza recipe --/
theorem salt_to_flour_ratio (ingredients : PizzaIngredients) : 
  ingredients.water = 10 →
  ingredients.flour = 16 →
  ingredients.water + ingredients.flour + ingredients.salt = 34 →
  ingredients.salt * 2 = ingredients.flour := by
  sorry

end salt_to_flour_ratio_l3835_383524


namespace distance_point_to_line_l3835_383580

/-- The distance from a point (2√2, 2√2) to the line x + y - √2 = 0 is 3 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (2 * Real.sqrt 2, 2 * Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y - Real.sqrt 2 = 0
  ∃ (d : ℝ), d = 3 ∧ 
    d = (|point.1 + point.2 - Real.sqrt 2|) / Real.sqrt 2 := by
  sorry

end distance_point_to_line_l3835_383580


namespace quadratic_roots_ratio_l3835_383501

/-- Given a quadratic equation x^2 + 12x + k = 0 where k is a real number,
    if the nonzero roots are in the ratio 3:1, then k = 27. -/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 12*x + k = 0 ∧ y^2 + 12*y + k = 0) → k = 27 := by
  sorry

end quadratic_roots_ratio_l3835_383501


namespace equation_solution_l3835_383545

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 9 / x + 20 / x^2 = 0 → 2 / x = 1 / 2 ∨ 2 / x = 2 / 5 := by
  sorry

end equation_solution_l3835_383545


namespace square_root_problem_l3835_383540

theorem square_root_problem (m : ℝ) : (Real.sqrt (m - 1) = 2) → m = 5 := by
  sorry

end square_root_problem_l3835_383540


namespace smallest_integer_l3835_383538

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) 
  (h2 : Nat.lcm a b / Nat.gcd a b = 75) : 
  ∀ c : ℕ+, (c < b → ¬(Nat.lcm a c / Nat.gcd a c = 75)) → b = 500 := by
  sorry

end smallest_integer_l3835_383538


namespace triangle_problem_l3835_383507

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.a = 7/2 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3) ∧
  (t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0) →
  -- Conclusions to prove
  (t.A = π/3) ∧ (t.b + t.c = 11/2) := by
  sorry


end triangle_problem_l3835_383507


namespace book_sale_revenue_l3835_383558

theorem book_sale_revenue (total_books : ℕ) (sold_price : ℕ) (remaining_books : ℕ) : 
  (2 * total_books = 3 * remaining_books) →
  (sold_price = 5) →
  (remaining_books = 50) →
  (2 * total_books / 3 * sold_price = 500) :=
by
  sorry

end book_sale_revenue_l3835_383558


namespace min_value_of_expression_l3835_383537

/-- Given vectors OA, OB, OC, where O is the origin, prove that the minimum value of 1/a + 2/b is 8 -/
theorem min_value_of_expression (a b : ℝ) (OA OB OC : ℝ × ℝ) : 
  a > 0 → b > 0 → 
  OA = (1, -2) → OB = (a, -1) → OC = (-b, 0) →
  (∃ (t : ℝ), (OB.1 - OA.1, OB.2 - OA.2) = t • (OC.1 - OA.1, OC.2 - OA.2)) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' ≥ 8) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1 / a' + 2 / b' = 8) :=
by sorry

end min_value_of_expression_l3835_383537


namespace fraction_equality_l3835_383547

theorem fraction_equality : (1622^2 - 1615^2) / (1629^2 - 1608^2) = 1 / 3 := by
  sorry

end fraction_equality_l3835_383547


namespace imaginary_power_sum_l3835_383515

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^22 + i^222 = -2 := by
  sorry

end imaginary_power_sum_l3835_383515


namespace bertsGoldenRetrieverWeight_l3835_383563

/-- Calculates the full adult weight of a golden retriever given its growth pattern -/
def goldenRetrieverAdultWeight (initialWeight : ℕ) : ℕ :=
  let weightAtWeek9 := initialWeight * 2
  let weightAt3Months := weightAtWeek9 * 2
  let weightAt5Months := weightAt3Months * 2
  let finalWeightIncrease := 30
  weightAt5Months + finalWeightIncrease

/-- Theorem stating that the full adult weight of Bert's golden retriever is 78 pounds -/
theorem bertsGoldenRetrieverWeight :
  goldenRetrieverAdultWeight 6 = 78 := by
  sorry

end bertsGoldenRetrieverWeight_l3835_383563


namespace radio_quiz_win_probability_l3835_383528

/-- Represents a quiz show with multiple-choice questions. -/
structure QuizShow where
  num_questions : ℕ
  num_options : ℕ
  min_correct : ℕ

/-- Calculates the probability of winning a quiz show by random guessing. -/
def win_probability (quiz : QuizShow) : ℚ :=
  sorry

/-- The specific quiz show described in the problem. -/
def radio_quiz : QuizShow :=
  { num_questions := 4
  , num_options := 4
  , min_correct := 2 }

/-- Theorem stating the probability of winning the radio quiz. -/
theorem radio_quiz_win_probability :
  win_probability radio_quiz = 121 / 256 :=
by sorry

end radio_quiz_win_probability_l3835_383528


namespace smallest_height_l3835_383551

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of the square base
  h : ℝ  -- height of the box
  area : ℝ -- surface area of the box

/-- The height of the box is twice the side length plus one -/
def height_constraint (b : Box) : Prop :=
  b.h = 2 * b.x + 1

/-- The surface area of the box is at least 150 square units -/
def area_constraint (b : Box) : Prop :=
  b.area ≥ 150

/-- The surface area is calculated as 2x^2 + 4x(2x + 1) -/
def surface_area_calc (b : Box) : Prop :=
  b.area = 2 * b.x^2 + 4 * b.x * (2 * b.x + 1)

/-- Main theorem: The smallest possible integer height is 9 units -/
theorem smallest_height (b : Box) 
  (h1 : height_constraint b) 
  (h2 : area_constraint b) 
  (h3 : surface_area_calc b) : 
  ∃ (min_height : ℕ), min_height = 9 ∧ 
    ∀ (h : ℕ), (∃ (b' : Box), height_constraint b' ∧ area_constraint b' ∧ surface_area_calc b' ∧ b'.h = h) → 
      h ≥ min_height :=
sorry

end smallest_height_l3835_383551


namespace arithmetic_sequence_properties_l3835_383542

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : seq.S 1023 - seq.S 1000 = 23) : 
  seq.a 1012 = 1 ∧ seq.S 2023 = 2023 := by
  sorry


end arithmetic_sequence_properties_l3835_383542
