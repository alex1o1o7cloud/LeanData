import Mathlib

namespace tv_sales_increase_l2788_278815

theorem tv_sales_increase (original_price : ℝ) (original_quantity : ℝ) 
  (h_positive_price : original_price > 0) (h_positive_quantity : original_quantity > 0) :
  let new_price := 0.9 * original_price
  let new_total_value := 1.665 * (original_price * original_quantity)
  ∃ (new_quantity : ℝ), 
    new_price * new_quantity = new_total_value ∧ 
    (new_quantity - original_quantity) / original_quantity = 0.85 :=
by sorry

end tv_sales_increase_l2788_278815


namespace roots_quadratic_equation_l2788_278885

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 8*m + 5 = 0) → 
  (n^2 - 8*n + 5 = 0) → 
  (1 / (m - 1) + 1 / (n - 1) = -3) := by
sorry

end roots_quadratic_equation_l2788_278885


namespace forest_to_verdant_green_conversion_l2788_278851

/-- Represents the ratio of blue to yellow paint in forest green -/
def forest_green_ratio : ℚ := 4 / 3

/-- Represents the ratio of yellow to blue paint in verdant green -/
def verdant_green_ratio : ℚ := 4 / 3

/-- The amount of yellow paint added to change forest green to verdant green -/
def yellow_paint_added : ℝ := 2.333333333333333

/-- The original amount of yellow paint in the forest green mixture -/
def original_yellow_paint : ℝ := 3

theorem forest_to_verdant_green_conversion :
  let b := forest_green_ratio * original_yellow_paint
  (original_yellow_paint + yellow_paint_added) / b = verdant_green_ratio :=
by sorry

end forest_to_verdant_green_conversion_l2788_278851


namespace keychain_thread_calculation_l2788_278804

theorem keychain_thread_calculation (class_friends : ℕ) (club_friends : ℕ) (total_thread : ℕ) : 
  class_friends = 6 →
  club_friends = class_friends / 2 →
  total_thread = 108 →
  total_thread / (class_friends + club_friends) = 12 :=
by sorry

end keychain_thread_calculation_l2788_278804


namespace james_twitch_income_l2788_278811

/-- Calculates the monthly income from Twitch subscriptions given the subscriber counts, costs, and revenue percentages for each tier. -/
def monthly_twitch_income (tier1_subs tier2_subs tier3_subs : ℕ) 
                          (tier1_cost tier2_cost tier3_cost : ℚ) 
                          (tier1_percent tier2_percent tier3_percent : ℚ) : ℚ :=
  tier1_subs * tier1_cost * tier1_percent +
  tier2_subs * tier2_cost * tier2_percent +
  tier3_subs * tier3_cost * tier3_percent

/-- Proves that James' monthly income from Twitch subscriptions is $2065.41 given the specified conditions. -/
theorem james_twitch_income : 
  monthly_twitch_income 130 75 45 (499/100) (999/100) (2499/100) (70/100) (80/100) (90/100) = 206541/100 := by
  sorry

end james_twitch_income_l2788_278811


namespace lemonade_sales_calculation_l2788_278875

/-- Calculates the total sales for lemonade glasses sold over two days -/
theorem lemonade_sales_calculation (price_per_glass : ℚ) (saturday_sales sunday_sales : ℕ) :
  price_per_glass = 25 / 100 →
  saturday_sales = 41 →
  sunday_sales = 53 →
  (saturday_sales + sunday_sales : ℚ) * price_per_glass = 2350 / 100 := by
  sorry

#eval (41 + 53 : ℚ) * (25 / 100) -- Optional: to verify the result

end lemonade_sales_calculation_l2788_278875


namespace range_of_a_l2788_278825

theorem range_of_a (a : ℝ) : 
  Real.sqrt ((2*a - 1)^2) = 1 - 2*a → a ≤ 1/2 := by sorry

end range_of_a_l2788_278825


namespace strawberry_harvest_l2788_278893

theorem strawberry_harvest (base height : ℝ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  base = 10 →
  height = 12 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 8 →
  (1/2 * base * height * plants_per_sqft * strawberries_per_plant : ℝ) = 2400 := by
  sorry

end strawberry_harvest_l2788_278893


namespace larger_number_problem_l2788_278873

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 10 → L = 1636 := by
  sorry

end larger_number_problem_l2788_278873


namespace isosceles_triangle_vertex_angle_l2788_278839

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : ∃ i, t.angles i = 80) :
  (t.angles 0 = 80 ∨ t.angles 0 = 20) ∨
  (t.angles 1 = 80 ∨ t.angles 1 = 20) ∨
  (t.angles 2 = 80 ∨ t.angles 2 = 20) :=
by sorry

end isosceles_triangle_vertex_angle_l2788_278839


namespace systematic_sampling_used_l2788_278860

/-- Represents the sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents the auditorium setup and sampling process --/
structure AuditoriumSampling where
  total_seats : Nat
  seats_per_row : Nat
  selected_seat_number : Nat
  num_selected : Nat

/-- Determines the sampling method based on the auditorium setup and selection process --/
def determine_sampling_method (setup : AuditoriumSampling) : SamplingMethod :=
  sorry

/-- Theorem stating that the sampling method used is systematic sampling --/
theorem systematic_sampling_used (setup : AuditoriumSampling) 
  (h1 : setup.total_seats = 25)
  (h2 : setup.seats_per_row = 20)
  (h3 : setup.selected_seat_number = 15)
  (h4 : setup.num_selected = 25) :
  determine_sampling_method setup = SamplingMethod.Systematic :=
  sorry

end systematic_sampling_used_l2788_278860


namespace no_integer_solutions_l2788_278853

theorem no_integer_solutions : ¬ ∃ x : ℤ, ∃ k : ℤ, x^2 + x + 13 = 121 * k := by
  sorry

end no_integer_solutions_l2788_278853


namespace pauls_lost_crayons_l2788_278886

/-- Paul's crayon problem -/
theorem pauls_lost_crayons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) :
  initial = 1453 →
  given_away = 563 →
  remaining = 332 →
  initial - given_away - remaining = 558 := by
  sorry

end pauls_lost_crayons_l2788_278886


namespace geometric_sequence_common_ratio_l2788_278802

/-- Given a geometric sequence {a_n} with common ratio q, if the sum of the first 3 terms is 7
    and the sum of the first 6 terms is 63, then q = 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 = 7) →       -- Sum of first 3 terms is 7
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 63) →  -- Sum of first 6 terms is 63
  q = 2 :=
by sorry

end geometric_sequence_common_ratio_l2788_278802


namespace distance_to_school_proof_l2788_278856

/-- The distance from Layla's house to the high school -/
def distance_to_school : ℝ := 3

theorem distance_to_school_proof :
  ∀ (total_distance : ℝ),
  (2 * distance_to_school + 4 = total_distance) →
  (total_distance = 10) →
  distance_to_school = 3 := by
sorry

end distance_to_school_proof_l2788_278856


namespace matilda_age_is_35_l2788_278881

-- Define the ages as natural numbers
def louis_age : ℕ := 14
def jerica_age : ℕ := 2 * louis_age
def matilda_age : ℕ := jerica_age + 7

-- Theorem statement
theorem matilda_age_is_35 : matilda_age = 35 := by
  sorry

end matilda_age_is_35_l2788_278881


namespace slope_intercept_form_parallel_lines_a_value_l2788_278877

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b) :=
sorry

theorem parallel_lines_a_value :
  (∀ x y : ℝ, a * x + 4 * y + 1 = 0 ↔ 2 * x + y - 2 = 0) → a = 8 :=
sorry

end slope_intercept_form_parallel_lines_a_value_l2788_278877


namespace hyperbola_focus_l2788_278812

/-- The hyperbola equation -2x^2 + 3y^2 + 8x - 18y - 8 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0

/-- A point (x, y) is a focus of the hyperbola if it satisfies the focus condition -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p q : ℝ), hyperbola_equation p q →
  (p - x)^2 + (q - y)^2 = ((p - 2)^2 / (2 * b^2) - (q - 3)^2 / (2 * a^2) + 1)^2 * (a^2 + b^2)

theorem hyperbola_focus :
  is_focus 2 7.5 :=
sorry

end hyperbola_focus_l2788_278812


namespace pole_wire_length_l2788_278872

def pole_problem (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) (short_pole_elevation : ℝ) : Prop :=
  let effective_short_pole_height : ℝ := short_pole_height + short_pole_elevation
  let vertical_distance : ℝ := tall_pole_height - effective_short_pole_height
  let wire_length : ℝ := Real.sqrt (base_distance^2 + vertical_distance^2)
  wire_length = Real.sqrt 445

theorem pole_wire_length :
  pole_problem 18 6 20 3 :=
by
  sorry

end pole_wire_length_l2788_278872


namespace line_equation_proof_l2788_278813

/-- Given two lines in the xy-plane -/
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0

/-- The intersection point of the two lines -/
def intersection_point : ℝ × ℝ := sorry

/-- The equation of the line passing through (2, 1) and the intersection point -/
def target_line (x y : ℝ) : Prop := 5*x - 7*y - 3 = 0

theorem line_equation_proof :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = intersection_point) →
  target_line (2 : ℝ) 1 ∧
  target_line (intersection_point.1) (intersection_point.2) :=
by sorry

end line_equation_proof_l2788_278813


namespace geometric_arithmetic_ratio_l2788_278854

/-- Given a geometric sequence with positive terms where a₂, ½a₃, and a₁ form an arithmetic sequence,
    the ratio (a₄ + a₅)/(a₃ + a₄) equals (1 + √5)/2. -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arith : a 2 - a 1 = (1/2 : ℝ) * a 3 - a 2) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end geometric_arithmetic_ratio_l2788_278854


namespace line_passes_through_fixed_point_l2788_278870

/-- The line mx-y+3+m=0 passes through the point (-1, 3) for any real number m -/
theorem line_passes_through_fixed_point (m : ℝ) : m * (-1) - 3 + 3 + m = 0 := by
  sorry

end line_passes_through_fixed_point_l2788_278870


namespace employee_salary_proof_l2788_278864

/-- Given two employees with a total weekly salary and a salary ratio, prove the salary of one employee. -/
theorem employee_salary_proof (total : ℚ) (ratio : ℚ) (n_salary : ℚ) : 
  total = 583 →
  ratio = 1.2 →
  n_salary + ratio * n_salary = total →
  n_salary = 265 := by
sorry

end employee_salary_proof_l2788_278864


namespace sumata_family_driving_l2788_278869

/-- The Sumata family's driving problem -/
theorem sumata_family_driving (days : ℝ) (miles_per_day : ℝ) 
  (h1 : days = 5.0)
  (h2 : miles_per_day = 50) :
  days * miles_per_day = 250 := by
  sorry

end sumata_family_driving_l2788_278869


namespace remainder_of_3_power_20_l2788_278863

theorem remainder_of_3_power_20 (a : ℕ) : 
  a = (1 + 2)^20 → a % 10 = 1 := by
  sorry

end remainder_of_3_power_20_l2788_278863


namespace solution_set_equality_l2788_278819

theorem solution_set_equality (a : ℝ) : 
  (∀ x, (a - 1) * x < a + 5 ↔ 2 * x < 4) → a = 7 := by
  sorry

end solution_set_equality_l2788_278819


namespace april_production_l2788_278855

/-- Calculates the production after n months given an initial production and monthly growth rate -/
def production_after_months (initial_production : ℕ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_production * (1 + growth_rate) ^ months

/-- Proves that the production in April is 926,100 pencils given the initial conditions -/
theorem april_production :
  let initial_production := 800000
  let growth_rate := 0.05
  let months := 3
  ⌊production_after_months initial_production growth_rate months⌋ = 926100 := by
  sorry

end april_production_l2788_278855


namespace cos_75_degrees_l2788_278882

theorem cos_75_degrees : 
  let cos_75 := Real.cos (75 * π / 180)
  let cos_60 := Real.cos (60 * π / 180)
  let sin_60 := Real.sin (60 * π / 180)
  let cos_15 := Real.cos (15 * π / 180)
  let sin_15 := Real.sin (15 * π / 180)
  cos_60 = 1/2 ∧ sin_60 = Real.sqrt 3 / 2 →
  cos_75 = cos_60 * cos_15 - sin_60 * sin_15 →
  cos_75 = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end cos_75_degrees_l2788_278882


namespace gas_cost_proof_l2788_278833

/-- The total cost of gas for a trip to New York City -/
def total_cost : ℝ := 82.50

/-- The number of friends initially splitting the cost -/
def initial_friends : ℕ := 3

/-- The number of friends who joined later -/
def additional_friends : ℕ := 2

/-- The total number of friends after more joined -/
def total_friends : ℕ := initial_friends + additional_friends

/-- The amount by which each original friend's cost decreased -/
def cost_decrease : ℝ := 11

theorem gas_cost_proof :
  (total_cost / initial_friends) - (total_cost / total_friends) = cost_decrease :=
sorry

end gas_cost_proof_l2788_278833


namespace S_in_quadrants_I_and_II_l2788_278879

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 2 * p.1 ∧ p.2 > 4 - p.1}

-- Define quadrants I and II
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem stating that S is contained in quadrants I and II
theorem S_in_quadrants_I_and_II : S ⊆ quadrantI ∪ quadrantII := by
  sorry


end S_in_quadrants_I_and_II_l2788_278879


namespace cubic_roots_condition_l2788_278822

theorem cubic_roots_condition (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = (x - α)*(x - β)*(x - γ)) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = (x - α^3)*(x - β^3)*(x - γ^3)) →
  c = a*b ∧ b ≤ 0 :=
by sorry

end cubic_roots_condition_l2788_278822


namespace sin_sum_identity_l2788_278899

theorem sin_sum_identity (x : ℝ) (h : Real.sin (2 * x + π / 5) = Real.sqrt 3 / 3) :
  Real.sin (4 * π / 5 - 2 * x) + Real.sin (3 * π / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 := by
  sorry

end sin_sum_identity_l2788_278899


namespace chess_tournament_games_l2788_278832

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 8 players, where each player plays every other player 
    exactly once, the total number of games played is 28. -/
theorem chess_tournament_games :
  num_games 8 = 28 := by
  sorry

end chess_tournament_games_l2788_278832


namespace arcade_spend_proof_l2788_278803

/-- Calculates the total amount spent at an arcade given the play time and cost per interval. -/
def arcade_spend (play_time_hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  let total_minutes := play_time_hours * 60
  let num_intervals := total_minutes / interval_minutes
  num_intervals * cost_per_interval

/-- Proves that playing at an arcade for 3 hours at $0.50 per 6 minutes costs $15. -/
theorem arcade_spend_proof :
  arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end arcade_spend_proof_l2788_278803


namespace problem_statement_l2788_278836

theorem problem_statement (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin θ - Real.cos θ = -Real.sqrt 14 / 4) : 
  (2 * (Real.cos θ)^2 - 1) / Real.cos (π/4 + θ) = 3/2 := by
  sorry

end problem_statement_l2788_278836


namespace equidistant_point_on_x_axis_l2788_278818

theorem equidistant_point_on_x_axis :
  ∃ x : ℝ,
    (x^2 + 4*x + 4 = x^2 + 16) ∧
    (∀ y : ℝ, y ≠ x → (y^2 + 4*y + 4 ≠ y^2 + 16)) →
    x = 3 := by
  sorry

end equidistant_point_on_x_axis_l2788_278818


namespace math_test_questions_math_test_questions_proof_l2788_278835

theorem math_test_questions : ℕ → Prop :=
  fun total_questions =>
    let word_problems : ℕ := 17
    let addition_subtraction_problems : ℕ := 28
    let steve_answered : ℕ := 38
    let difference : ℕ := 7
    
    (total_questions - steve_answered = difference) ∧
    (word_problems + addition_subtraction_problems ≤ total_questions) ∧
    (steve_answered < total_questions) →
    total_questions = 45

-- The proof is omitted
theorem math_test_questions_proof : math_test_questions 45 := by sorry

end math_test_questions_math_test_questions_proof_l2788_278835


namespace constant_product_percentage_change_l2788_278862

theorem constant_product_percentage_change (x y : ℝ) (C : ℝ) (h : x * y = C) :
  x * (1 + 0.2) * (y * (1 - 1/6)) = C := by sorry

end constant_product_percentage_change_l2788_278862


namespace expression_evaluation_l2788_278880

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem expression_evaluation (y : ℕ) (x : ℕ) (h1 : y = 2) (h2 : x = y + 1) :
  5 * (factorial y) * (x ^ y) + 3 * (factorial x) * (y ^ x) = 234 := by
  sorry

end expression_evaluation_l2788_278880


namespace quadratic_factor_l2788_278841

theorem quadratic_factor (k : ℝ) : 
  (∃ b : ℝ, (X + 5) * (X + b) = X^2 - k*X - 15) → 
  (X - 3) * (X + 5) = X^2 - k*X - 15 :=
by sorry

end quadratic_factor_l2788_278841


namespace forty_percent_of_number_equals_144_l2788_278898

theorem forty_percent_of_number_equals_144 (x : ℝ) : 0.4 * x = 144 → x = 360 := by
  sorry

end forty_percent_of_number_equals_144_l2788_278898


namespace game_cost_proof_l2788_278892

/-- The cost of a video game that Ronald and Max want to buy --/
def game_cost : ℕ := 60

/-- The price of each ice cream --/
def ice_cream_price : ℕ := 5

/-- The number of ice creams they need to sell to afford the game --/
def ice_creams_needed : ℕ := 24

/-- The number of people splitting the cost of the game --/
def people_splitting_cost : ℕ := 2

/-- Theorem stating that the game cost is correct given the conditions --/
theorem game_cost_proof : 
  game_cost = (ice_cream_price * ice_creams_needed) / people_splitting_cost :=
by sorry

end game_cost_proof_l2788_278892


namespace ratio_equality_l2788_278896

theorem ratio_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a / 2 = b / 3) (h5 : b / 3 = c / 5) : (a + b) / (c - a) = 5 / 3 := by
  sorry

end ratio_equality_l2788_278896


namespace remaining_children_meals_l2788_278837

theorem remaining_children_meals (total_children_meals : ℕ) 
  (adults_consumed : ℕ) (child_adult_ratio : ℚ) :
  total_children_meals = 90 →
  adults_consumed = 42 →
  child_adult_ratio = 90 / 70 →
  total_children_meals - (↑adults_consumed * child_adult_ratio).floor = 36 :=
by
  sorry

end remaining_children_meals_l2788_278837


namespace diagonal_less_than_half_perimeter_l2788_278809

-- Define a quadrilateral with sides a, b, c, d and diagonal x
structure Quadrilateral :=
  (a b c d x : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (positive_diagonal : 0 < x)

-- Theorem: The diagonal is less than half the perimeter
theorem diagonal_less_than_half_perimeter (q : Quadrilateral) :
  q.x < (q.a + q.b + q.c + q.d) / 2 := by
  sorry

end diagonal_less_than_half_perimeter_l2788_278809


namespace polyhedron_volume_l2788_278897

-- Define the polygons
def right_triangle (a b c : ℝ) := a^2 + b^2 = c^2
def rectangle (l w : ℝ) := l > 0 ∧ w > 0
def equilateral_triangle (s : ℝ) := s > 0

-- Define the polyhedron
def polyhedron (A E F : ℝ → ℝ → ℝ → Prop) 
               (B C D : ℝ → ℝ → Prop) 
               (G : ℝ → Prop) := 
  A 1 2 (Real.sqrt 5) ∧ 
  E 1 2 (Real.sqrt 5) ∧ 
  F 1 2 (Real.sqrt 5) ∧ 
  B 1 2 ∧ 
  C 2 3 ∧ 
  D 1 3 ∧ 
  G (Real.sqrt 5)

-- State the theorem
theorem polyhedron_volume 
  (A E F : ℝ → ℝ → ℝ → Prop) 
  (B C D : ℝ → ℝ → Prop) 
  (G : ℝ → Prop) : 
  polyhedron right_triangle right_triangle right_triangle 
              rectangle rectangle rectangle 
              equilateral_triangle → 
  ∃ v : ℝ, v = 6 := by
  sorry

end polyhedron_volume_l2788_278897


namespace money_problem_l2788_278806

theorem money_problem (a b : ℚ) (h1 : 7 * a + b = 89) (h2 : 4 * a - b = 38) :
  a = 127 / 11 ∧ b = 90 / 11 := by
  sorry

end money_problem_l2788_278806


namespace arithmetic_sequence_y_value_l2788_278858

/-- Given an arithmetic sequence with first three terms y + 1, 3y - 2, and 9 - 2y, prove that y = 2 -/
theorem arithmetic_sequence_y_value (y : ℝ) : 
  (∃ d : ℝ, (3*y - 2) - (y + 1) = d ∧ (9 - 2*y) - (3*y - 2) = d) → y = 2 := by
  sorry

end arithmetic_sequence_y_value_l2788_278858


namespace exists_expression_for_100_l2788_278845

/-- An arithmetic expression using only the number 3, parentheses, and basic arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def count_threes : Expr → ℕ
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an arithmetic expression using fewer than ten threes that evaluates to 100. -/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry


end exists_expression_for_100_l2788_278845


namespace game_solvable_l2788_278820

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The possible moves in the game -/
inductive Move
  | Red (k : ℤ)
  | Blue (k : ℤ)

/-- Apply a move to the game state -/
def applyMove (r : ℚ) (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Red k => 
      { red := state.blue + r^k * (state.red - state.blue),
        blue := state.blue }
  | Move.Blue k => 
      { red := state.red,
        blue := state.red + r^k * (state.blue - state.red) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Apply a sequence of moves to the initial game state -/
def applyMoveSequence (r : ℚ) (moves : MoveSequence) : GameState :=
  moves.foldl (applyMove r) { red := 0, blue := 1 }

/-- The main theorem -/
theorem game_solvable (r : ℚ) : 
  (∃ (moves : MoveSequence), moves.length ≤ 2021 ∧ (applyMoveSequence r moves).red = 1) ↔ 
  (∃ (m : ℕ), m ≥ 1 ∧ m ≤ 1010 ∧ r = (m + 1) / m) :=
sorry

end game_solvable_l2788_278820


namespace sum_of_five_consecutive_even_numbers_l2788_278884

theorem sum_of_five_consecutive_even_numbers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end sum_of_five_consecutive_even_numbers_l2788_278884


namespace billy_score_problem_l2788_278826

/-- Billy's video game score problem -/
theorem billy_score_problem (old_score : ℕ) (rounds : ℕ) : 
  old_score = 725 → rounds = 363 → (old_score + 1) / rounds = 2 := by
  sorry

end billy_score_problem_l2788_278826


namespace factor_expression_l2788_278834

theorem factor_expression (y : ℝ) : 4 * y * (y + 2) + 6 * (y + 2) = (y + 2) * (2 * (2 * y + 3)) := by
  sorry

end factor_expression_l2788_278834


namespace annulus_chord_circle_area_equality_l2788_278888

theorem annulus_chord_circle_area_equality (R r x : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : R^2 = r^2 + x^2) :
  π * x^2 = π * (R^2 - r^2) :=
by sorry

end annulus_chord_circle_area_equality_l2788_278888


namespace interval_condition_l2788_278831

theorem interval_condition (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 :=
by sorry

end interval_condition_l2788_278831


namespace sum_of_reciprocals_of_roots_l2788_278828

/-- Given a quadratic polynomial 7x^2 + 4x + 9, if α and β are the reciprocals of its roots,
    then their sum α + β equals -4/9 -/
theorem sum_of_reciprocals_of_roots (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 4 * a + 9 = 0) ∧ 
              (7 * b^2 + 4 * b + 9 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) → 
  α + β = -4/9 := by
sorry

end sum_of_reciprocals_of_roots_l2788_278828


namespace apple_picking_ratio_l2788_278816

/-- Represents the number of apples picked in each hour -/
structure ApplePicking where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (a : ApplePicking) : ℕ :=
  a.first_hour + a.second_hour + a.third_hour

/-- Theorem: The ratio of apples picked in the second hour to the first hour is 2:1 -/
theorem apple_picking_ratio (a : ApplePicking) :
  a.first_hour = 66 →
  a.third_hour = a.first_hour / 3 →
  total_apples a = 220 →
  a.second_hour = 2 * a.first_hour :=
by
  sorry


end apple_picking_ratio_l2788_278816


namespace solution_set_quadratic_equation_l2788_278868

theorem solution_set_quadratic_equation :
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end solution_set_quadratic_equation_l2788_278868


namespace fraction_of_total_l2788_278805

theorem fraction_of_total (total : ℚ) (r_amount : ℚ) : 
  total = 9000 → r_amount = 3600 → r_amount / total = 2 / 5 := by
  sorry

end fraction_of_total_l2788_278805


namespace ratio_equality_l2788_278827

theorem ratio_equality (a b c : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : a + c / b - c = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
sorry

end ratio_equality_l2788_278827


namespace murtha_pebbles_after_20_days_l2788_278859

def pebbles_collected (n : ℕ) : ℕ := n + 1

def pebbles_given_away (n : ℕ) : ℕ := if n % 5 = 0 then 3 else 0

def total_pebbles (days : ℕ) : ℕ :=
  2 + (Finset.range (days - 1)).sum pebbles_collected - (Finset.range days).sum pebbles_given_away

theorem murtha_pebbles_after_20_days :
  total_pebbles 20 = 218 := by sorry

end murtha_pebbles_after_20_days_l2788_278859


namespace w_in_terms_of_abc_l2788_278850

theorem w_in_terms_of_abc (w a b c x y z : ℝ) 
  (hdistinct : w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (heq1 : x + y + z = 1)
  (heq2 : x*a^2 + y*b^2 + z*c^2 = w^2)
  (heq3 : x*a^3 + y*b^3 + z*c^3 = w^3)
  (heq4 : x*a^4 + y*b^4 + z*c^4 = w^4) :
  w = -a*b*c / (a*b + b*c + c*a) := by
sorry

end w_in_terms_of_abc_l2788_278850


namespace cake_division_l2788_278810

theorem cake_division (total_cake : ℚ) (num_people : ℕ) :
  total_cake = 7/8 ∧ num_people = 4 →
  total_cake / num_people = 7/32 := by
sorry

end cake_division_l2788_278810


namespace product_prs_is_27_l2788_278895

theorem product_prs_is_27 (p r s : ℕ) 
  (eq1 : 4^p + 4^3 = 272)
  (eq2 : 3^r + 27 = 54)
  (eq3 : 2^(s+2) + 10 = 42) :
  p * r * s = 27 := by
  sorry

end product_prs_is_27_l2788_278895


namespace symmetric_line_l2788_278876

/-- Given a line L with equation x + 2y - 1 = 0 and a point P(1, -1),
    the line symmetric to L with respect to P has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x + 2*y - 1 = 0) → -- original line equation
  (∃ (x' y' : ℝ), (x' = 2 - x ∧ y' = -2 - y) ∧ (x' + 2*y' - 1 = 0)) → -- symmetry condition
  (x + 2*y - 3 = 0) -- symmetric line equation
:= by sorry

end symmetric_line_l2788_278876


namespace cos_2α_in_second_quadrant_l2788_278801

theorem cos_2α_in_second_quadrant (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end cos_2α_in_second_quadrant_l2788_278801


namespace mistaken_calculation_correction_l2788_278821

theorem mistaken_calculation_correction (x : ℤ) : 
  x - 15 + 27 = 41 → x - 27 + 15 = 17 := by
  sorry

end mistaken_calculation_correction_l2788_278821


namespace a_share_is_4080_l2788_278894

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 4080 given the investments and total profit. -/
theorem a_share_is_4080 
  (investment_a : ℚ) 
  (investment_b : ℚ) 
  (investment_c : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 13600) :
  calculate_share_of_profit investment_a investment_b investment_c total_profit = 4080 := by
  sorry

#eval calculate_share_of_profit 6300 4200 10500 13600

end a_share_is_4080_l2788_278894


namespace binomial_expansion_example_l2788_278840

theorem binomial_expansion_example : 8^3 + 3*(8^2)*2 + 3*8*(2^2) + 2^3 = 1000 := by
  sorry

end binomial_expansion_example_l2788_278840


namespace smallest_of_three_l2788_278838

theorem smallest_of_three : ∀ (a b c : ℕ), a = 10 ∧ b = 11 ∧ c = 12 → a < b ∧ a < c := by
  sorry

end smallest_of_three_l2788_278838


namespace largest_difference_l2788_278824

def U : ℕ := 3 * 2005^2006
def V : ℕ := 2005^2006
def W : ℕ := 2004 * 2005^2005
def X : ℕ := 3 * 2005^2005
def Y : ℕ := 2005^2005
def Z : ℕ := 2005^2004

theorem largest_difference : 
  (U - V > V - W) ∧ (U - V > W - X) ∧ (U - V > X - Y) ∧ (U - V > Y - Z) :=
by sorry

end largest_difference_l2788_278824


namespace book_purchase_change_l2788_278866

/-- The change received when buying two books with given prices and paying with a fixed amount. -/
theorem book_purchase_change (book1_price book2_price payment : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : payment = 20) : 
  payment - (book1_price + book2_price) = 8 := by
sorry

end book_purchase_change_l2788_278866


namespace fixed_points_range_l2788_278830

/-- The function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- A fixed point of f is a real number x such that f(x) = x -/
def is_fixed_point (a : ℝ) (x : ℝ) : Prop := f a x = x

/-- The proposition that f has exactly two different fixed points in [1,3] -/
def has_two_fixed_points (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧
  is_fixed_point a x ∧ is_fixed_point a y ∧
  ∀ (z : ℝ), z ∈ Set.Icc 1 3 → is_fixed_point a z → (z = x ∨ z = y)

/-- The main theorem stating the range of a -/
theorem fixed_points_range :
  ∀ a : ℝ, has_two_fixed_points a ↔ a ∈ Set.Icc (-10/3) (-3) :=
sorry

end fixed_points_range_l2788_278830


namespace john_jury_duty_days_l2788_278823

/-- Calculates the total number of days spent on jury duty given the specified conditions. -/
def juryDutyDays (jurySelectionDays : ℕ) (trialMultiplier : ℕ) (deliberationFullDays : ℕ) (deliberationHoursPerDay : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let deliberationHours := deliberationFullDays * deliberationHoursPerDay
  let deliberationDays := deliberationHours / 24
  jurySelectionDays + trialDays + deliberationDays

/-- Theorem stating that under the given conditions, John spends 14 days on jury duty. -/
theorem john_jury_duty_days :
  juryDutyDays 2 4 6 16 = 14 := by
  sorry

#eval juryDutyDays 2 4 6 16

end john_jury_duty_days_l2788_278823


namespace area_enclosed_by_g_l2788_278889

open Real MeasureTheory

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

theorem area_enclosed_by_g : 
  ∫ (x : ℝ) in (0)..(π / 3), g x = 3 / 4 := by sorry

end area_enclosed_by_g_l2788_278889


namespace cubic_equation_c_value_l2788_278842

/-- Given a cubic equation with coefficients a, b, c, d, returns whether it has three distinct positive roots -/
def has_three_distinct_positive_roots (a b c d : ℝ) : Prop := sorry

/-- Given three real numbers, returns their sum of base-3 logarithms -/
def sum_of_log3 (x y z : ℝ) : ℝ := sorry

theorem cubic_equation_c_value (c d : ℝ) :
  has_three_distinct_positive_roots 4 (5 * c) (3 * d) c →
  ∃ (x y z : ℝ), sum_of_log3 x y z = 3 ∧ 
    4 * x^3 + 5 * c * x^2 + 3 * d * x + c = 0 ∧
    4 * y^3 + 5 * c * y^2 + 3 * d * y + c = 0 ∧
    4 * z^3 + 5 * c * z^2 + 3 * d * z + c = 0 →
  c = -108 := by
  sorry

end cubic_equation_c_value_l2788_278842


namespace only_η_hypergeometric_l2788_278883

/-- Represents the total number of balls -/
def total_balls : ℕ := 10

/-- Represents the number of black balls -/
def black_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Represents the score for a black ball -/
def black_score : ℕ := 2

/-- Represents the score for a white ball -/
def white_score : ℕ := 1

/-- Represents the maximum number drawn -/
def X : ℕ → ℕ := sorry

/-- Represents the minimum number drawn -/
def Y : ℕ → ℕ := sorry

/-- Represents the total score of the drawn balls -/
def ξ : ℕ → ℕ := sorry

/-- Represents the number of black balls drawn -/
def η : ℕ → ℕ := sorry

/-- Defines a hypergeometric distribution -/
def is_hypergeometric (f : ℕ → ℕ) : Prop := sorry

theorem only_η_hypergeometric :
  is_hypergeometric η ∧
  ¬is_hypergeometric X ∧
  ¬is_hypergeometric Y ∧
  ¬is_hypergeometric ξ :=
sorry

end only_η_hypergeometric_l2788_278883


namespace triangle_count_specific_l2788_278843

/-- The number of triangles formed by points on two sides of a triangle -/
def triangles_from_points (n m : ℕ) : ℕ :=
  Nat.choose (n + m + 1) 3 - Nat.choose (n + 1) 3 - Nat.choose (m + 1) 3

/-- Theorem: The number of triangles formed by 5 points on one side,
    6 points on another side, and 1 shared vertex is 165 -/
theorem triangle_count_specific : triangles_from_points 5 6 = 165 := by
  sorry

end triangle_count_specific_l2788_278843


namespace tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l2788_278847

/-- The function f(x) = e^x - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

/-- Theorem for the tangent line equation when a = 2 --/
theorem tangent_line_at_zero (x y : ℝ) :
  (f 2) 0 = 1 →
  (∀ h, deriv (f 2) h = Real.exp h - 2) →
  x + y - 1 = 0 ↔ y - 1 = -(x - 0) :=
sorry

/-- Theorem for f(x) > 0 when a = 2 --/
theorem f_positive_when_a_eq_two :
  ∀ x, f 2 x > 0 :=
sorry

/-- Theorem for the maximum value of f(x) when a > 1 --/
theorem max_value_on_interval (a : ℝ) :
  a > 1 →
  ∃ x ∈ Set.Icc 0 a, ∀ y ∈ Set.Icc 0 a, f a x ≥ f a y ∧ f a x = Real.exp a - a^2 :=
sorry

end tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l2788_278847


namespace volume_ratio_equal_surface_area_l2788_278887

/-- Given an equilateral cone, an equilateral cylinder, and a sphere, all with equal surface area F,
    their volumes are in the ratio 2 : √6 : 3. -/
theorem volume_ratio_equal_surface_area (F : ℝ) (F_pos : F > 0) :
  ∃ (K₁ K₂ K₃ : ℝ),
    (K₁ > 0 ∧ K₂ > 0 ∧ K₃ > 0) ∧
    (K₁ = F * Real.sqrt F / (9 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cone
    (K₂ = F * Real.sqrt F * Real.sqrt 6 / (18 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cylinder
    (K₃ = F * Real.sqrt F / (6 * Real.sqrt Real.pi)) ∧  -- Volume of sphere
    (K₁ / 2 = K₂ / Real.sqrt 6 ∧ K₁ / 2 = K₃ / 3) :=
by sorry

end volume_ratio_equal_surface_area_l2788_278887


namespace log_problem_l2788_278867

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 9 = x) → 
  (Real.log 81 / Real.log 3 = k * x) → 
  k = 8 := by
sorry

end log_problem_l2788_278867


namespace water_bottles_left_l2788_278808

theorem water_bottles_left (initial_bottles : ℕ) (bottles_drunk : ℕ) : 
  initial_bottles = 301 → bottles_drunk = 144 → initial_bottles - bottles_drunk = 157 := by
  sorry

end water_bottles_left_l2788_278808


namespace snow_probability_l2788_278890

theorem snow_probability (p1 p2 : ℚ) : 
  p1 = 1/4 → p2 = 1/3 → 
  1 - (1 - p1)^4 * (1 - p2)^3 = 68359/100000 := by sorry

end snow_probability_l2788_278890


namespace toothpick_grid_theorem_l2788_278846

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then grid.height * grid.width else 0
  horizontal + vertical + diagonal

/-- The theorem to be proved -/
theorem toothpick_grid_theorem (grid : ToothpickGrid) :
  grid.height = 15 → grid.width = 12 → grid.has_diagonals = true →
  total_toothpicks grid = 567 := by
  sorry

end toothpick_grid_theorem_l2788_278846


namespace batsman_sixes_l2788_278857

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def boundary_value : ℕ := 4
def six_value : ℕ := 6

theorem batsman_sixes :
  ∃ (sixes : ℕ),
    sixes * six_value + boundaries * boundary_value + (total_runs / 2) = total_runs ∧
    sixes = 8 := by
  sorry

end batsman_sixes_l2788_278857


namespace factors_of_96_with_square_sum_208_l2788_278800

theorem factors_of_96_with_square_sum_208 :
  ∀ a b : ℕ+,
    a * b = 96 ∧ 
    a^2 + b^2 = 208 →
    (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) := by
  sorry

end factors_of_96_with_square_sum_208_l2788_278800


namespace complex_modulus_problem_l2788_278848

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I) * (1 - Complex.I) = -2) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_modulus_problem_l2788_278848


namespace pears_picked_total_l2788_278817

/-- The number of pears Alyssa picked -/
def alyssa_pears : ℕ := 42

/-- The number of pears Nancy picked -/
def nancy_pears : ℕ := 17

/-- The total number of pears picked -/
def total_pears : ℕ := alyssa_pears + nancy_pears

theorem pears_picked_total : total_pears = 59 := by
  sorry

end pears_picked_total_l2788_278817


namespace surface_area_of_rmon_l2788_278865

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (baseSideLength : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (position : ℝ)

/-- The solid RMON created by slicing the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (m : EdgePoint)
  (n : EdgePoint)
  (o : EdgePoint)

/-- Calculate the surface area of the sliced solid -/
noncomputable def surfaceArea (solid : SlicedSolid) : ℝ :=
  sorry

/-- Main theorem: The surface area of RMON is 30.62 square units -/
theorem surface_area_of_rmon (solid : SlicedSolid) 
  (h1 : solid.prism.height = 10)
  (h2 : solid.prism.baseSideLength = 10)
  (h3 : solid.m.position = 1/4)
  (h4 : solid.n.position = 1/4)
  (h5 : solid.o.position = 1/4) :
  surfaceArea solid = 30.62 := by
  sorry

end surface_area_of_rmon_l2788_278865


namespace chicken_difference_l2788_278891

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The number of chickens free ranging -/
def free_ranging_chickens : ℕ := 52

/-- The difference between double the number of chickens in the run and the number of chickens free ranging -/
theorem chicken_difference : 2 * run_chickens - free_ranging_chickens = 4 := by
  sorry

end chicken_difference_l2788_278891


namespace machine_present_value_l2788_278814

/-- The present value of a machine given its depreciation rate, selling price after two years, and profit made. -/
theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : depreciation_rate = 0.2)
  (h2 : selling_price = 118000.00000000001)
  (h3 : profit = 22000) :
  ∃ (present_value : ℝ),
    present_value = 150000.00000000002 ∧
    present_value * (1 - depreciation_rate)^2 = selling_price - profit :=
by sorry

end machine_present_value_l2788_278814


namespace sin_transformation_l2788_278852

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 - π / 6) = 2 * Real.sin ((x - π / 2) / 3) := by sorry

end sin_transformation_l2788_278852


namespace donut_selection_problem_l2788_278829

/-- The number of ways to select n items from k types with at least one of each type -/
def selectWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_selection_problem :
  selectWithMinimum 6 3 = 10 := by
  sorry

end donut_selection_problem_l2788_278829


namespace infinitely_many_composite_mersenne_numbers_l2788_278874

theorem infinitely_many_composite_mersenne_numbers :
  ∀ k : ℕ, ∃ n : ℕ, 
    Odd n ∧ 
    ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n - 1 = a * b :=
by sorry

end infinitely_many_composite_mersenne_numbers_l2788_278874


namespace heptagonal_prism_faces_and_vertices_l2788_278844

/-- A heptagonal prism is a three-dimensional shape with two heptagonal bases and rectangular lateral faces. -/
structure HeptagonalPrism where
  baseFaces : Nat
  lateralFaces : Nat
  baseVertices : Nat

/-- Properties of a heptagonal prism -/
def heptagonalPrismProperties : HeptagonalPrism where
  baseFaces := 2
  lateralFaces := 7
  baseVertices := 7

/-- Theorem: A heptagonal prism has 9 faces and 14 vertices -/
theorem heptagonal_prism_faces_and_vertices :
  let h := heptagonalPrismProperties
  (h.baseFaces + h.lateralFaces = 9) ∧ (h.baseVertices * h.baseFaces = 14) := by
  sorry

end heptagonal_prism_faces_and_vertices_l2788_278844


namespace carnival_spending_theorem_l2788_278861

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let game_cost := 2 * food_cost
  initial_amount - (food_cost + ride_cost + game_cost)

theorem carnival_spending_theorem :
  carnival_spending 80 15 = 5 := by
  sorry

end carnival_spending_theorem_l2788_278861


namespace max_value_of_n_l2788_278849

theorem max_value_of_n (a b c d n : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : 1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ n / (a - d)) :
  n ≤ 9 ∧ ∃ (a b c d : ℝ), a > b ∧ b > c ∧ c > d ∧ 
    1 / (a - b) + 1 / (b - c) + 1 / (c - d) = 9 / (a - d) :=
sorry

end max_value_of_n_l2788_278849


namespace intersection_complement_l2788_278871

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement : M ∩ (U \ N) = {1} := by
  sorry

end intersection_complement_l2788_278871


namespace second_smallest_prime_perimeter_l2788_278807

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are distinct -/
def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

/-- The main theorem stating that the second smallest perimeter of a scalene triangle
    with distinct prime sides and a prime perimeter is 29 -/
theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    areDistinct a b c ∧
    isTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      areDistinct x y z ∧
      isTriangle x y z ∧
      isPrime (x + y + z) ∧
      (x + y + z < 29) →
      (x + y + z = 23)) :=
by sorry

end second_smallest_prime_perimeter_l2788_278807


namespace inequality_proof_l2788_278878

theorem inequality_proof (x y : ℝ) : 
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧ 
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end inequality_proof_l2788_278878
