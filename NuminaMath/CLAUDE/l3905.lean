import Mathlib

namespace largest_angle_in_special_triangle_l3905_390594

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = (4/3) * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 70°
  max a (max b c) = 70 := by
sorry


end largest_angle_in_special_triangle_l3905_390594


namespace equidistant_point_on_number_line_l3905_390519

/-- Given points A (-1) and B (5) on a number line, if point P is equidistant from A and B, then P represents the number 2. -/
theorem equidistant_point_on_number_line :
  let a : ℝ := -1
  let b : ℝ := 5
  ∀ p : ℝ, |p - a| = |p - b| → p = 2 := by
  sorry

end equidistant_point_on_number_line_l3905_390519


namespace waiter_tips_fraction_l3905_390528

theorem waiter_tips_fraction (salary : ℚ) (h : salary > 0) :
  let tips := (5 / 2) * salary
  let total_income := salary + tips
  tips / total_income = 5 / 7 := by
  sorry

end waiter_tips_fraction_l3905_390528


namespace solve_homework_problem_l3905_390571

def homework_problem (total_problems : ℕ) (completed_at_stop1 : ℕ) (completed_at_stop2 : ℕ) (completed_at_stop3 : ℕ) : Prop :=
  let completed_on_bus := completed_at_stop1 + completed_at_stop2 + completed_at_stop3
  let remaining_problems := total_problems - completed_on_bus
  remaining_problems = 3

theorem solve_homework_problem :
  homework_problem 9 2 3 1 := by
  sorry

end solve_homework_problem_l3905_390571


namespace trees_survival_difference_l3905_390533

theorem trees_survival_difference (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 13)
  (h2 : dead_trees = 6) :
  initial_trees - dead_trees - dead_trees = 1 :=
by sorry

end trees_survival_difference_l3905_390533


namespace angle_sum_is_right_angle_l3905_390568

theorem angle_sum_is_right_angle (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_trig : (Real.cos α / Real.sin β) + (Real.cos β / Real.sin α) = 2) : 
  α + β = π/2 := by
sorry

end angle_sum_is_right_angle_l3905_390568


namespace equation_solutions_l3905_390513

-- Define the equation
def equation (a x : ℝ) : Prop :=
  ((1 - x^2)^2 + 2*a^2 + 5*a)^7 - ((3*a + 2)*(1 - x^2) + 3)^7 = 
  5 - 2*a - (3*a + 2)*x^2 - 2*a^2 - (1 - x^2)^2

-- Define the interval
def in_interval (x : ℝ) : Prop :=
  -Real.sqrt 6 / 2 ≤ x ∧ x ≤ Real.sqrt 2

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ in_interval x₁ ∧ in_interval x₂ ∧ 
  equation a x₁ ∧ equation a x₂

-- State the theorem
theorem equation_solutions :
  ∀ a : ℝ, has_two_distinct_solutions a ↔ 
  (0.25 ≤ a ∧ a < 1) ∨ (-3.5 ≤ a ∧ a < -2) := by sorry

end equation_solutions_l3905_390513


namespace ceiling_sqrt_225_l3905_390575

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end ceiling_sqrt_225_l3905_390575


namespace circle_tangency_l3905_390577

/-- Definition of circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle O₂ -/
def circle_O₂ (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

/-- The distance between the centers of two internally tangent circles
    is equal to the difference of their radii -/
def internally_tangent (a : ℝ) : Prop := 
  (4^2 + a^2).sqrt = 5 - 1

theorem circle_tangency (a : ℝ) 
  (h : internally_tangent a) : a = 0 := by sorry

end circle_tangency_l3905_390577


namespace no_regular_polygon_inscription_l3905_390504

-- Define an ellipse with unequal axes
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ
  axesUnequal : majorAxis ≠ minorAxis

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  moreThanFourSides : sides > 4

-- Define the concept of inscribing a polygon in an ellipse
def isInscribed (p : RegularPolygon) (e : Ellipse) : Prop :=
  sorry -- Definition of inscription

-- Theorem statement
theorem no_regular_polygon_inscription 
  (e : Ellipse) (p : RegularPolygon) : ¬ isInscribed p e := by
  sorry

#check no_regular_polygon_inscription

end no_regular_polygon_inscription_l3905_390504


namespace complex_product_l3905_390538

theorem complex_product (A B C : ℂ) : 
  A = 7 + 3*I ∧ B = I ∧ C = 7 - 3*I → A * B * C = 58 * I :=
by sorry

end complex_product_l3905_390538


namespace mike_additional_money_needed_l3905_390562

def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_savings_percentage : ℝ := 0.40

theorem mike_additional_money_needed :
  let discounted_phone := phone_cost * (1 - phone_discount)
  let discounted_smartwatch := smartwatch_cost * (1 - smartwatch_discount)
  let total_before_tax := discounted_phone + discounted_smartwatch
  let total_with_tax := total_before_tax * (1 + sales_tax)
  let mike_savings := total_with_tax * mike_savings_percentage
  total_with_tax - mike_savings = 1023.99 := by sorry

end mike_additional_money_needed_l3905_390562


namespace triangle_area_l3905_390563

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b^2 + c^2 = a^2 - b*c →
  (a * b * Real.cos C) = -4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
sorry

end triangle_area_l3905_390563


namespace root_equation_solution_l3905_390510

theorem root_equation_solution (p q r : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/p) * (M^(1/q))^(1/p) * ((M^(1/r))^(1/q))^(1/p))^p = M^(15/24)) :
  q = 2 := by
  sorry

end root_equation_solution_l3905_390510


namespace tenth_valid_number_l3905_390547

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 13

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem tenth_valid_number : nth_valid_number 10 = 166 := sorry

end tenth_valid_number_l3905_390547


namespace fertilizer_prices_l3905_390515

theorem fertilizer_prices (price_A price_B : ℝ)
  (h1 : price_A = price_B + 100)
  (h2 : 2 * price_A + price_B = 1700) :
  price_A = 600 ∧ price_B = 500 := by
  sorry

end fertilizer_prices_l3905_390515


namespace equation_is_linear_l3905_390582

/-- A linear equation in one variable -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Check if an equation is a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The specific equation 3x = 2x -/
def f (x : ℝ) : ℝ := 3 * x - 2 * x

theorem equation_is_linear : is_linear_equation f := by sorry

end equation_is_linear_l3905_390582


namespace distance_between_projections_l3905_390574

/-- Given a point A(-1, 2, -3) in ℝ³, prove that the distance between its projection
    onto the yOz plane and its projection onto the x-axis is √14. -/
theorem distance_between_projections :
  let A : ℝ × ℝ × ℝ := (-1, 2, -3)
  let P₁ : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)  -- projection onto yOz plane
  let P₂ : ℝ × ℝ × ℝ := (A.1, 0, 0)        -- projection onto x-axis
  (P₁.1 - P₂.1)^2 + (P₁.2.1 - P₂.2.1)^2 + (P₁.2.2 - P₂.2.2)^2 = 14 :=
by sorry

end distance_between_projections_l3905_390574


namespace right_triangle_quadratic_roots_l3905_390554

theorem right_triangle_quadratic_roots (m : ℝ) : 
  let f := fun x : ℝ => x^2 - (2*m - 1)*x + 4*(m - 1)
  ∃ (a b : ℝ), 
    (f a = 0 ∧ f b = 0) ∧  -- BC and AC are roots of the quadratic equation
    (a ≠ b) ∧               -- Distinct roots
    (a > 0 ∧ b > 0) ∧       -- Positive lengths
    (a^2 + b^2 = 25) →      -- Pythagorean theorem (AB^2 = 5^2 = 25)
  m = 4 := by
sorry

end right_triangle_quadratic_roots_l3905_390554


namespace algae_growth_time_l3905_390501

/-- The growth factor of the algae population every 5 hours -/
def growth_factor : ℕ := 3

/-- The initial number of algae cells -/
def initial_cells : ℕ := 200

/-- The target number of algae cells -/
def target_cells : ℕ := 145800

/-- The time in hours for one growth cycle -/
def cycle_time : ℕ := 5

/-- The function to calculate the number of cells after a given number of cycles -/
def cells_after_cycles (n : ℕ) : ℕ :=
  initial_cells * growth_factor ^ n

/-- The theorem stating the time taken for the algae to grow to at least the target number of cells -/
theorem algae_growth_time : ∃ (t : ℕ), 
  cells_after_cycles (t / cycle_time) ≥ target_cells ∧ 
  ∀ (s : ℕ), s < t → cells_after_cycles (s / cycle_time) < target_cells :=
by sorry

end algae_growth_time_l3905_390501


namespace no_such_polynomials_l3905_390535

/-- A polynomial is a perfect square if it's the square of another non-constant polynomial -/
def IsPerfectSquare (p : Polynomial ℝ) : Prop :=
  ∃ q : Polynomial ℝ, q.degree > 0 ∧ p = q^2

theorem no_such_polynomials :
  ¬∃ (f g : Polynomial ℝ),
    f.degree > 0 ∧ g.degree > 0 ∧
    ¬IsPerfectSquare f ∧
    ¬IsPerfectSquare g ∧
    IsPerfectSquare (f.comp g) ∧
    IsPerfectSquare (g.comp f) :=
by sorry

end no_such_polynomials_l3905_390535


namespace cosine_value_from_ratio_l3905_390545

theorem cosine_value_from_ratio (α : Real) (h : (1 - Real.cos α) / Real.sin α = 3) :
  Real.cos α = -4/5 := by
  sorry

end cosine_value_from_ratio_l3905_390545


namespace next_coincidence_after_lcm_robinsons_next_busy_day_l3905_390599

/-- Represents a periodic event --/
structure PeriodicEvent where
  period : ℕ

/-- Calculates the least common multiple (LCM) of a list of natural numbers --/
def lcmList (list : List ℕ) : ℕ :=
  list.foldl Nat.lcm 1

/-- Theorem: The next coincidence of periodic events occurs after their LCM --/
theorem next_coincidence_after_lcm (events : List PeriodicEvent) : 
  let periods := events.map (·.period)
  let nextCoincidence := lcmList periods
  ∀ t : ℕ, t < nextCoincidence → ¬ (∀ e ∈ events, t % e.period = 0) :=
by sorry

/-- Robinson Crusoe's activities --/
def robinsons_activities : List PeriodicEvent := [
  { period := 2 },  -- Water replenishment
  { period := 3 },  -- Fruit collection
  { period := 5 }   -- Hunting
]

/-- Theorem: Robinson's next busy day is 30 days after the current busy day --/
theorem robinsons_next_busy_day :
  lcmList (robinsons_activities.map (·.period)) = 30 :=
by sorry

end next_coincidence_after_lcm_robinsons_next_busy_day_l3905_390599


namespace library_shelf_theorem_l3905_390539

/-- Represents the thickness of a biology book -/
def biology_thickness : ℝ := 1

/-- Represents the thickness of a history book -/
def history_thickness : ℝ := 2 * biology_thickness

/-- Represents the length of the shelf -/
def shelf_length : ℝ := 1

theorem library_shelf_theorem 
  (B G P Q F : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ F ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ F ∧ 
                P ≠ Q ∧ P ≠ F ∧ 
                Q ≠ F)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ F > 0)
  (h_fill1 : B * biology_thickness + G * history_thickness = shelf_length)
  (h_fill2 : P * biology_thickness + Q * history_thickness = shelf_length)
  (h_fill3 : F * biology_thickness = shelf_length) :
  F = B + 2*G ∧ F = P + 2*Q :=
sorry

end library_shelf_theorem_l3905_390539


namespace julio_fishing_l3905_390548

theorem julio_fishing (fish_per_hour : ℕ) (hours : ℕ) (lost_fish : ℕ) (total_fish : ℕ) : 
  hours = 9 → lost_fish = 15 → total_fish = 48 → fish_per_hour * hours - lost_fish = total_fish → fish_per_hour = 7 := by
sorry

end julio_fishing_l3905_390548


namespace vector_relations_l3905_390597

def a : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_relations :
  (parallel (a.1 + (b (-1)).1, a.2 + (b (-1)).2) c) ∧
  (perpendicular (a.1 + (b (3/2)).1, a.2 + (b (3/2)).2) c) := by
  sorry

end vector_relations_l3905_390597


namespace soccer_team_statistics_l3905_390550

theorem soccer_team_statistics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 10 →
  both_subjects = 6 →
  ∃ (statistics_players : ℕ),
    statistics_players = 23 ∧
    statistics_players + physics_players - both_subjects = total_players :=
by sorry

end soccer_team_statistics_l3905_390550


namespace quadratic_inequality_roots_l3905_390589

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
sorry

end quadratic_inequality_roots_l3905_390589


namespace yoongi_hoseok_age_sum_yoongi_hoseok_age_sum_proof_l3905_390556

/-- The sum of Yoongi's and Hoseok's ages is 26 years -/
theorem yoongi_hoseok_age_sum : ℕ → ℕ → ℕ → Prop :=
  fun yoongi_age hoseok_age aunt_age =>
    (aunt_age = yoongi_age + 23) →
    (yoongi_age = hoseok_age + 4) →
    (aunt_age = 38) →
    (yoongi_age + hoseok_age = 26)

/-- Proof of the theorem -/
theorem yoongi_hoseok_age_sum_proof :
  ∃ (yoongi_age hoseok_age aunt_age : ℕ),
    yoongi_hoseok_age_sum yoongi_age hoseok_age aunt_age :=
by
  sorry

end yoongi_hoseok_age_sum_yoongi_hoseok_age_sum_proof_l3905_390556


namespace inequality_proof_l3905_390578

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end inequality_proof_l3905_390578


namespace fish_lives_12_years_l3905_390506

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by
  sorry

end fish_lives_12_years_l3905_390506


namespace emerald_puzzle_l3905_390558

theorem emerald_puzzle :
  ∃ n : ℕ,
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 6 ∧
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 6 → n ≤ m) ∧
    n % 9 = 4 := by
  sorry

end emerald_puzzle_l3905_390558


namespace inscribed_quadrilateral_fourth_side_l3905_390557

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (s : ℝ) 
  (h1 : r = 150 * Real.sqrt 3) 
  (h2 : s = 150) : 
  ∃ (x : ℝ), x = 150 * (Real.sqrt 3 - 3) ∧ 
  (s + s + s + x)^2 = 3 * (2 * r)^2 :=
sorry

end inscribed_quadrilateral_fourth_side_l3905_390557


namespace adjacent_lateral_faces_angle_l3905_390587

/-- A regular quadrilateral pyramid is a pyramid with a square base and four congruent triangular faces. -/
structure RegularQuadrilateralPyramid where
  /-- The side length of the square base -/
  base_side : ℝ
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ

/-- The theorem states that if the lateral face of a regular quadrilateral pyramid
    forms a 45° angle with the base plane, then the angle between adjacent lateral faces is 120°. -/
theorem adjacent_lateral_faces_angle
  (pyramid : RegularQuadrilateralPyramid)
  (h : pyramid.lateral_base_angle = Real.pi / 4) :
  let adjacent_angle := Real.arccos (-1/3)
  adjacent_angle = 2 * Real.pi / 3 := by
  sorry

end adjacent_lateral_faces_angle_l3905_390587


namespace total_earnings_of_three_workers_l3905_390598

/-- The total earnings of three workers given their combined earnings -/
theorem total_earnings_of_three_workers
  (earnings_a : ℕ) (earnings_b : ℕ) (earnings_c : ℕ)
  (h1 : earnings_a + earnings_c = 400)
  (h2 : earnings_b + earnings_c = 300)
  (h3 : earnings_c = 100) :
  earnings_a + earnings_b + earnings_c = 600 :=
by sorry

end total_earnings_of_three_workers_l3905_390598


namespace cattle_train_departure_time_l3905_390523

/-- Proves that the cattle train left 6 hours before the diesel train --/
theorem cattle_train_departure_time (cattle_speed diesel_speed : ℝ) 
  (time_difference total_time : ℝ) (total_distance : ℝ) : 
  cattle_speed = 56 →
  diesel_speed = cattle_speed - 33 →
  total_time = 12 →
  total_distance = 1284 →
  total_distance = diesel_speed * total_time + cattle_speed * total_time + cattle_speed * time_difference →
  time_difference = 6 := by
  sorry

end cattle_train_departure_time_l3905_390523


namespace root_shift_polynomial_l3905_390553

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end root_shift_polynomial_l3905_390553


namespace mikes_total_work_hours_l3905_390564

/-- Calculates the total hours worked given a work schedule --/
def totalHoursWorked (hours_per_day1 hours_per_day2 hours_per_day3 : ℕ) 
                     (days1 days2 days3 : ℕ) : ℕ :=
  hours_per_day1 * days1 + hours_per_day2 * days2 + hours_per_day3 * days3

/-- Proves that Mike's total work hours is 93 --/
theorem mikes_total_work_hours :
  totalHoursWorked 3 4 5 5 7 10 = 93 := by
  sorry

#eval totalHoursWorked 3 4 5 5 7 10

end mikes_total_work_hours_l3905_390564


namespace g_of_2_l3905_390517

/-- Given a function g(x) = px^8 + qx^4 + rx + 7 where g(-2) = -5,
    prove that g(2) = 2p(256) + 2q(16) + 19 -/
theorem g_of_2 (p q r : ℝ) (g : ℝ → ℝ) 
    (h1 : ∀ x, g x = p * x^8 + q * x^4 + r * x + 7)
    (h2 : g (-2) = -5) :
  g 2 = 2 * p * 256 + 2 * q * 16 + 19 := by
  sorry

end g_of_2_l3905_390517


namespace two_positions_from_six_candidates_l3905_390588

/-- The number of ways to select two distinct positions from a group of candidates. -/
def selectTwoPositions (n : ℕ) : ℕ := n * (n - 1)

/-- The number of candidates. -/
def numCandidates : ℕ := 6

/-- The observed number of ways to select two positions. -/
def observedSelections : ℕ := 30

/-- Theorem stating that selecting 2 distinct positions from 6 candidates results in 30 possible selections. -/
theorem two_positions_from_six_candidates :
  selectTwoPositions numCandidates = observedSelections := by
  sorry


end two_positions_from_six_candidates_l3905_390588


namespace train_speed_l3905_390596

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 140)
  (h2 : platform_length = 260)
  (h3 : time_to_pass = 23.998080153587715) : 
  ∃ (speed : ℝ), abs (speed - 60.0048) < 0.0001 := by
  sorry

end train_speed_l3905_390596


namespace sticker_difference_l3905_390514

/-- Proves the difference in stickers received by Mandy and Justin -/
theorem sticker_difference (initial_stickers : ℕ) 
  (friends : ℕ) (stickers_per_friend : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 72 →
  friends = 3 →
  stickers_per_friend = 4 →
  remaining_stickers = 42 →
  ∃ (mandy_stickers justin_stickers : ℕ),
    mandy_stickers = friends * stickers_per_friend + 2 ∧
    justin_stickers < mandy_stickers ∧
    initial_stickers = remaining_stickers + friends * stickers_per_friend + mandy_stickers + justin_stickers ∧
    mandy_stickers - justin_stickers = 10 :=
by sorry

end sticker_difference_l3905_390514


namespace hash_difference_l3905_390546

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by sorry

end hash_difference_l3905_390546


namespace toothpicks_required_l3905_390503

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 3000

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The total number of toothpicks if no sides were shared -/
def total_potential_toothpicks : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- The theorem stating the total number of toothpicks required -/
theorem toothpicks_required : 
  (total_potential_toothpicks - boundary_toothpicks) / 2 + boundary_toothpicks = 6761700 := by
  sorry

end toothpicks_required_l3905_390503


namespace number_of_planes_l3905_390518

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- Theorem: The number of commercial planes is 25 -/
theorem number_of_planes : 
  (total_wings / wings_per_plane : ℕ) = 25 := by sorry

end number_of_planes_l3905_390518


namespace pencil_box_cost_l3905_390524

/-- The cost of Linda's purchases -/
def purchase_cost (notebook_price : ℝ) (notebook_quantity : ℕ) (pen_price : ℝ) (pencil_price : ℝ) : ℝ :=
  notebook_price * notebook_quantity + pen_price + pencil_price

/-- The theorem stating the cost of the box of pencils -/
theorem pencil_box_cost : 
  ∃ (pencil_price : ℝ),
    purchase_cost 1.20 3 1.70 pencil_price = 6.80 ∧ 
    pencil_price = 1.50 := by
  sorry

#check pencil_box_cost

end pencil_box_cost_l3905_390524


namespace unique_satisfying_number_l3905_390580

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number satisfies the given condition -/
def satisfiesCondition (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n + 8802 > reverseDigits n

theorem unique_satisfying_number : 
  ∀ n : Nat, satisfiesCondition n ↔ n = 1099 :=
by sorry

end unique_satisfying_number_l3905_390580


namespace toph_fish_count_l3905_390592

theorem toph_fish_count (total_people : ℕ) (average_fish : ℕ) (aang_fish : ℕ) (sokka_fish : ℕ) :
  total_people = 3 →
  average_fish = 8 →
  aang_fish = 7 →
  sokka_fish = 5 →
  average_fish * total_people - aang_fish - sokka_fish = 12 :=
by sorry

end toph_fish_count_l3905_390592


namespace train_crossing_time_l3905_390573

/-- Given a train and platform with specific dimensions and crossing time, 
    calculate the time it takes for the train to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 450)
  (h3 : platform_crossing_time = 45)
  : ∃ (signal_pole_time : ℝ), 
    (signal_pole_time ≥ 17.9 ∧ signal_pole_time ≤ 18.1) := by
  sorry

end train_crossing_time_l3905_390573


namespace f_roots_l3905_390591

-- Define the function f
def f (x : ℝ) : ℝ := 
  let matrix := !![1, 1, 1; x, -1, 1; x^2, 2, 1]
  Matrix.det matrix

-- State the theorem
theorem f_roots : 
  {x : ℝ | f x = 0} = {-3/2, 1} := by sorry

end f_roots_l3905_390591


namespace sum_of_combinations_l3905_390512

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_combinations : 
  (∀ m n : ℕ, binomial m n + binomial (m - 1) n = binomial m (n + 1)) →
  (binomial 3 3 + binomial 4 3 + binomial 5 3 + binomial 6 3 + 
   binomial 7 3 + binomial 8 3 + binomial 9 3 + binomial 10 3 = 330) :=
by sorry

end sum_of_combinations_l3905_390512


namespace ratio_problem_l3905_390525

theorem ratio_problem (x y : ℤ) : 
  (y = 3 * x) → -- The two integers are in the ratio of 1 to 3
  (x + 10 = y) → -- Adding 10 to the smaller number makes them equal
  y = 15 := by -- The larger integer is 15
sorry

end ratio_problem_l3905_390525


namespace number_plus_ten_l3905_390590

theorem number_plus_ten (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end number_plus_ten_l3905_390590


namespace min_value_of_sum_of_squares_l3905_390586

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 ≥ 6 := by
sorry

end min_value_of_sum_of_squares_l3905_390586


namespace complex_equation_product_l3905_390585

theorem complex_equation_product (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a * b = 6 := by
  sorry

end complex_equation_product_l3905_390585


namespace cubic_function_min_value_l3905_390541

/-- Given a cubic function f(x) with a known maximum value on [-2, 2],
    prove that its minimum value on the same interval is -37. -/
theorem cubic_function_min_value (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2 * x^3 - 6 * x^2 + m) →
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x) →
  ∃ x ∈ Set.Icc (-2) 2, f x = -37 ∧ ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x :=
by sorry

end cubic_function_min_value_l3905_390541


namespace reinforcement_size_l3905_390500

/-- Calculates the size of the reinforcement given the initial garrison size, 
    initial provision duration, days passed before reinforcement, and 
    remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  calculate_reinforcement 2000 54 18 20 = 1600 := by
  sorry

#eval calculate_reinforcement 2000 54 18 20

end reinforcement_size_l3905_390500


namespace circumscribed_quadrilateral_altitudes_collinear_l3905_390534

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

structure Quadrilateral : Type :=
  (A B C D : Point)

-- Define the properties
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

def is_altitude (P Q R S : Point) : Prop := sorry

-- Define the theorem
theorem circumscribed_quadrilateral_altitudes_collinear 
  (ABCD : Quadrilateral) (O : Point) (c : Circle) 
  (A₁ B₁ C₁ D₁ : Point) : 
  is_circumscribed ABCD c →
  c.center = O →
  is_altitude A O B A₁ →
  is_altitude B O A B₁ →
  is_altitude C O D C₁ →
  is_altitude D O C D₁ →
  ∃ (l : Set Point), A₁ ∈ l ∧ B₁ ∈ l ∧ C₁ ∈ l ∧ D₁ ∈ l ∧ 
    ∀ (P Q : Point), P ∈ l → Q ∈ l → ∃ (t : ℝ), Q.x = P.x + t * (Q.y - P.y) :=
by sorry

end circumscribed_quadrilateral_altitudes_collinear_l3905_390534


namespace line_increase_percentage_l3905_390505

/-- Given that increasing the number of lines by 60 results in 240 lines,
    prove that the percentage increase is 100/3%. -/
theorem line_increase_percentage : ℝ → Prop :=
  fun original_lines =>
    (original_lines + 60 = 240) →
    ((60 / original_lines) * 100 = 100 / 3)

/-- Proof of the theorem -/
lemma prove_line_increase_percentage : ∃ x : ℝ, line_increase_percentage x := by
  sorry

end line_increase_percentage_l3905_390505


namespace geometric_sequence_ratio_l3905_390520

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (∀ n, a (n + 1) > a n) →      -- increasing sequence
  a 2 = 2 →                     -- a_2 = 2
  a 4 - a 3 = 4 →               -- a_4 - a_3 = 4
  q = 2 := by
sorry

end geometric_sequence_ratio_l3905_390520


namespace supplement_of_complement_of_42_l3905_390537

-- Define the original angle
def original_angle : ℝ := 42

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_42 : 
  supplement (complement original_angle) = 132 := by sorry

end supplement_of_complement_of_42_l3905_390537


namespace number_difference_l3905_390530

theorem number_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) : 
  |x - y| = 7 := by
sorry

end number_difference_l3905_390530


namespace z_in_fourth_quadrant_l3905_390595

def z : ℂ := Complex.I^3 * (1 + Complex.I) * Complex.I

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l3905_390595


namespace average_of_eleven_numbers_l3905_390544

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 19 →
  last_six_avg = 27 →
  sixth_number = 34 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 22 :=
by sorry

end average_of_eleven_numbers_l3905_390544


namespace chessboard_rearrangement_impossibility_l3905_390531

theorem chessboard_rearrangement_impossibility :
  ∀ (initial_placement final_placement : Fin 8 → Fin 8 → Bool),
  (∀ i j : Fin 8, (∃! k : Fin 8, initial_placement i k = true) ∧ 
                  (∃! k : Fin 8, initial_placement k j = true)) →
  (∀ i j : Fin 8, (∃! k : Fin 8, final_placement i k = true) ∧ 
                  (∃! k : Fin 8, final_placement k j = true)) →
  (∀ i j : Fin 8, initial_placement i j = true → 
    ∃ i' j' : Fin 8, final_placement i' j' = true ∧ 
    (i'.val + j'.val : ℕ) > (i.val + j.val)) →
  False :=
sorry

end chessboard_rearrangement_impossibility_l3905_390531


namespace sum_of_four_numbers_l3905_390516

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end sum_of_four_numbers_l3905_390516


namespace sin_cos_fourth_power_sum_l3905_390572

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 := by
  sorry

end sin_cos_fourth_power_sum_l3905_390572


namespace chess_tournament_games_l3905_390567

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 19 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 171. --/
theorem chess_tournament_games :
  num_games 19 = 171 := by
sorry

end chess_tournament_games_l3905_390567


namespace remainder_theorem_l3905_390579

theorem remainder_theorem : 4 * 6^24 + 3^48 ≡ 5 [ZMOD 7] := by
  sorry

end remainder_theorem_l3905_390579


namespace complex_equation_solution_l3905_390561

theorem complex_equation_solution (Z : ℂ) : (2 + 4*I) / Z = 1 - I → Z = -1 + 3*I := by
  sorry

end complex_equation_solution_l3905_390561


namespace shopkeeper_change_l3905_390559

/-- Represents the change given by the shopkeeper -/
structure Change where
  total_bills : ℕ
  bill_value_1 : ℕ
  bill_value_2 : ℕ
  noodles_value : ℕ

/-- The problem statement -/
theorem shopkeeper_change (c : Change) (h1 : c.total_bills = 16)
    (h2 : c.bill_value_1 = 10) (h3 : c.bill_value_2 = 5) (h4 : c.noodles_value = 5)
    (h5 : 100 = c.noodles_value + c.bill_value_1 * x + c.bill_value_2 * (c.total_bills - x)) :
    x = 3 :=
  sorry

end shopkeeper_change_l3905_390559


namespace nested_bracket_equals_two_l3905_390529

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_equals_two :
  bracket (bracket 60 30 90) (bracket 2 1 3) (bracket 10 5 15) = 2 := by
  sorry

end nested_bracket_equals_two_l3905_390529


namespace min_value_x_l3905_390583

theorem min_value_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : ∀ (a b : ℝ), a > 0 → b > 0 → 1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2)
  (h2 : ∀ (a b : ℝ), a > 0 → b > 0 → 4 * a + b * (1 - a) = 0) :
  x ≥ 1 ∧ ∀ (y : ℝ), y > 0 → y < 1 → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / a^2 + 16 / b^2 < 1 + y / 2 - y^2 :=
by sorry

end min_value_x_l3905_390583


namespace isosceles_triangle_vertex_angle_l3905_390566

-- Define an isosceles triangle with one interior angle of 40°
structure IsoscelesTriangle where
  base_angle : ℝ
  is_isosceles : True
  has_40_degree_angle : base_angle = 40 ∨ 180 - 2 * base_angle = 40

-- Theorem stating that the vertex angle is either 40° or 100°
theorem isosceles_triangle_vertex_angle (t : IsoscelesTriangle) :
  (180 - 2 * t.base_angle = 40) ∨ (180 - 2 * t.base_angle = 100) :=
by sorry


end isosceles_triangle_vertex_angle_l3905_390566


namespace triangle_problem_l3905_390527

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) :
  (t.a = 3 ∧ t.b = 5 ∧ t.B = 2 * π / 3 → Real.sin t.A = 3 * Real.sqrt 3 / 10) ∧
  (t.a = 3 ∧ t.b = 5 ∧ t.C = 2 * π / 3 → t.c = 7) := by
  sorry

end triangle_problem_l3905_390527


namespace angle_C_value_l3905_390543

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- State the theorem
theorem angle_C_value (t : Triangle) 
  (h : t.a^2 + t.b^2 - t.c^2 + t.a * t.b = 0) : 
  t.angleC = 2 * Real.pi / 3 := by
  sorry

end angle_C_value_l3905_390543


namespace hare_wolf_distance_l3905_390560

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

def relative_speed : ℝ := hare_speed - wolf_speed

theorem hare_wolf_distance :
  ∀ d : ℝ, d > 0 ∧ d < track_length / 2 →
  (d = distance_traveled relative_speed ∨ d = track_length - distance_traveled relative_speed) →
  d = 40 ∨ d = 60 := by sorry

end hare_wolf_distance_l3905_390560


namespace greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3905_390532

theorem greatest_whole_number_inequality (x : ℤ) : 
  (6 * x - 4 < 5 - 3 * x) → x ≤ 0 :=
by sorry

theorem zero_satisfies_inequality : 
  6 * 0 - 4 < 5 - 3 * 0 :=
by sorry

theorem no_positive_integer_satisfies_inequality (x : ℤ) :
  x > 0 → ¬(6 * x - 4 < 5 - 3 * x) :=
by sorry

end greatest_whole_number_inequality_zero_satisfies_inequality_no_positive_integer_satisfies_inequality_l3905_390532


namespace race_time_proof_l3905_390540

/-- In a 1000-meter race, runner A beats runner B by either 25 meters or 10 seconds. -/
theorem race_time_proof (v : ℝ) (t : ℝ) (h1 : v > 0) (h2 : t > 0) : 
  (1000 = v * t ∧ 975 = v * (t + 10)) → t = 400 := by
  sorry

end race_time_proof_l3905_390540


namespace simplify_expression_l3905_390551

theorem simplify_expression (y : ℝ) : 4*y + 5*y + 6*y + 2 = 15*y + 2 := by
  sorry

end simplify_expression_l3905_390551


namespace green_to_blue_ratio_l3905_390521

/-- Represents the number of chairs of each color in a classroom --/
structure ClassroomChairs where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- The conditions of the classroom chair problem --/
def classroom_conditions (c : ClassroomChairs) : Prop :=
  c.blue = 10 ∧
  ∃ k : ℕ, c.green = k * c.blue ∧
  c.white = c.green + c.blue - 13 ∧
  c.blue + c.green + c.white = 67

/-- The theorem stating that the ratio of green to blue chairs is 3:1 --/
theorem green_to_blue_ratio (c : ClassroomChairs) 
  (h : classroom_conditions c) : c.green = 3 * c.blue :=
sorry

end green_to_blue_ratio_l3905_390521


namespace bottle_caps_remaining_l3905_390593

theorem bottle_caps_remaining (initial_caps : ℕ) (removed_caps : ℕ) :
  initial_caps = 16 → removed_caps = 6 → initial_caps - removed_caps = 10 := by
  sorry

end bottle_caps_remaining_l3905_390593


namespace container_volume_ratio_l3905_390565

theorem container_volume_ratio 
  (container1 container2 container3 : ℝ) 
  (h1 : 3/5 * container1 = 2/3 * container2) 
  (h2 : 2/3 * container2 - 1/2 * container3 = 1/2 * container3) 
  (h3 : container1 > 0) 
  (h4 : container2 > 0) 
  (h5 : container3 > 0) : 
  container2 / container3 = 2/3 := by
sorry


end container_volume_ratio_l3905_390565


namespace area_ratio_is_one_twentyfifth_l3905_390526

/-- A square inscribed in a circle with a smaller square as described -/
structure InscribedSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  t : ℝ
  /-- The larger square is inscribed in the circle -/
  larger_inscribed : s = r * Real.sqrt 2
  /-- The smaller square has one side coinciding with the larger square -/
  coinciding_side : t ≤ s
  /-- Two vertices of the smaller square are on the circle -/
  smaller_on_circle : t * Real.sqrt ((s/2)^2 + (t/2)^2) = r * s

/-- The ratio of the areas of the smaller square to the larger square is 1/25 -/
theorem area_ratio_is_one_twentyfifth (sq : InscribedSquares) :
  (sq.t^2) / (sq.s^2) = 1 / 25 := by
  sorry

end area_ratio_is_one_twentyfifth_l3905_390526


namespace third_shiny_on_fifth_probability_l3905_390508

def total_pennies : ℕ := 10
def shiny_pennies : ℕ := 5
def dull_pennies : ℕ := 5
def draws : ℕ := 5

def probability_third_shiny_on_fifth : ℚ :=
  (Nat.choose 4 2 * Nat.choose 6 2) / Nat.choose total_pennies draws

theorem third_shiny_on_fifth_probability :
  probability_third_shiny_on_fifth = 5 / 14 := by sorry

end third_shiny_on_fifth_probability_l3905_390508


namespace square_sum_reciprocal_l3905_390509

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l3905_390509


namespace complex_magnitude_problem_l3905_390549

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 24 - 16 * Complex.I ∧ Complex.abs w = Real.sqrt 52 →
  Complex.abs z = 4 := by
sorry

end complex_magnitude_problem_l3905_390549


namespace no_divisible_by_six_l3905_390570

theorem no_divisible_by_six : ∀ y : ℕ, y < 10 → ¬(36000 + 100 * y + 25) % 6 = 0 := by
  sorry

end no_divisible_by_six_l3905_390570


namespace series_solution_l3905_390584

-- Define the series
def S (x y : ℝ) : ℝ := 1 + 2*x*y + 3*(x*y)^2 + 4*(x*y)^3 + 5*(x*y)^4 + 6*(x*y)^5 + 7*(x*y)^6 + 8*(x*y)^7

-- State the theorem
theorem series_solution :
  ∃ (x y : ℝ), S x y = 16 ∧ x = 3/4 ∧ (y = 1 ∨ y = -1) := by
  sorry

end series_solution_l3905_390584


namespace max_snacks_with_15_dollars_l3905_390569

/-- Represents the number of snacks that can be bought with a given amount of money -/
def maxSnacks (money : ℕ) : ℕ :=
  let singlePrice := 2  -- Price of a single snack
  let packOf4Price := 5  -- Price of a pack of 4 snacks
  let packOf7Price := 8  -- Price of a pack of 7 snacks
  -- Function to calculate the maximum number of snacks
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the maximum number of snacks that can be bought with $15 is 12 -/
theorem max_snacks_with_15_dollars :
  maxSnacks 15 = 12 :=
by sorry

end max_snacks_with_15_dollars_l3905_390569


namespace cylinder_cut_surface_increase_l3905_390576

/-- Represents the possible shapes of the increased surface area when cutting a cylinder --/
inductive IncreasedSurfaceShape
  | Circle
  | Rectangle

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a way to cut a cylinder into two equal parts --/
structure CutMethod where
  (cylinder : Cylinder)
  (increasedShape : IncreasedSurfaceShape)

/-- States that there exist at least two different ways to cut a cylinder 
    resulting in different increased surface area shapes --/
theorem cylinder_cut_surface_increase 
  (c : Cylinder) : 
  ∃ (cut1 cut2 : CutMethod), 
    cut1.cylinder = c ∧ 
    cut2.cylinder = c ∧ 
    cut1.increasedShape ≠ cut2.increasedShape :=
sorry

end cylinder_cut_surface_increase_l3905_390576


namespace no_ten_goals_possible_l3905_390507

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the number of goals scored by each player -/
def GoalCount := Player → ℕ

/-- Represents the statements made by each player -/
def Statements := Player → Player → ℕ

/-- Checks if the statements are consistent with the goal count and the truth-lie condition -/
def ConsistentStatements (gc : GoalCount) (s : Statements) : Prop :=
  ∀ p : Player, (s p p = gc p ∧ s p (nextPlayer p) ≠ gc (nextPlayer p)) ∨
                (s p p ≠ gc p ∧ s p (nextPlayer p) = gc (nextPlayer p))
where
  nextPlayer : Player → Player
  | Player.Anton => Player.Ilya
  | Player.Ilya => Player.Sergey
  | Player.Sergey => Player.Anton

/-- The main theorem stating that it's impossible to have a total of 10 goals -/
theorem no_ten_goals_possible (gc : GoalCount) (s : Statements) :
  ConsistentStatements gc s → (gc Player.Anton + gc Player.Ilya + gc Player.Sergey ≠ 10) := by
  sorry

end no_ten_goals_possible_l3905_390507


namespace equation_solution_l3905_390511

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (-7 + Real.sqrt 105) / 4) ∧ 
    (x₂ = (-7 - Real.sqrt 105) / 4) ∧ 
    (∀ x : ℝ, (4 * x^2 + 8 * x - 5 ≠ 0) → (2 * x - 1 ≠ 0) → 
      ((3 * x - 7) / (4 * x^2 + 8 * x - 5) = x / (2 * x - 1)) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l3905_390511


namespace problem_1_l3905_390502

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}
def C : Set ℝ := {0, 1/3, 1/5}

theorem problem_1 : ∀ a : ℝ, B a ⊆ A ↔ a ∈ C := by sorry

end problem_1_l3905_390502


namespace octagon_side_length_l3905_390552

/-- The side length of a regular octagon formed from the same wire as a regular pentagon --/
theorem octagon_side_length (pentagon_side : ℝ) (h : pentagon_side = 16) : 
  let pentagon_perimeter := 5 * pentagon_side
  let octagon_side := pentagon_perimeter / 8
  octagon_side = 10 := by
sorry

end octagon_side_length_l3905_390552


namespace functional_equation_solution_l3905_390536

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end functional_equation_solution_l3905_390536


namespace max_central_rectangle_area_l3905_390542

/-- Given a square of side length 23 divided into 9 rectangles, with 4 known areas,
    prove that the maximum area of the central rectangle is 180 -/
theorem max_central_rectangle_area :
  ∀ (a b c d e f : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a + b + c = 23 →
    d + e + f = 23 →
    a * d = 13 →
    b * f = 111 →
    c * e = 37 →
    a * f = 123 →
    b * e ≤ 180 :=
by sorry

end max_central_rectangle_area_l3905_390542


namespace fifteenth_term_is_3_to_8_l3905_390522

def sequence_term (n : ℕ) : ℤ :=
  if n % 4 == 1 then (-3) ^ (n / 4 + 1)
  else if n % 4 == 3 then 3 ^ (n / 2)
  else 1

theorem fifteenth_term_is_3_to_8 :
  sequence_term 15 = 3^8 := by sorry

end fifteenth_term_is_3_to_8_l3905_390522


namespace tangent_line_at_origin_l3905_390581

/-- The equation of the tangent line to y = 2ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_at_origin (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.log (x + 1)
  let f' : ℝ → ℝ := λ x => 2 / (x + 1)
  let tangent_line : ℝ → ℝ := λ x => 2 * x
  (∀ x, HasDerivAt f (f' x) x) →
  HasDerivAt f 2 0 →
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - tangent_line x| ≤ ε * |x|
  := by sorry

end tangent_line_at_origin_l3905_390581


namespace jill_has_six_peaches_l3905_390555

-- Define the number of peaches each person has
def steven_peaches : ℕ := 19
def jake_peaches : ℕ := steven_peaches - 18
def jill_peaches : ℕ := steven_peaches - 13

-- Theorem to prove
theorem jill_has_six_peaches : jill_peaches = 6 := by
  sorry

end jill_has_six_peaches_l3905_390555
