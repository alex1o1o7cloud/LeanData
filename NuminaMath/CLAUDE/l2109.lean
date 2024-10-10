import Mathlib

namespace parallel_lines_m_values_l2109_210942

theorem parallel_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), 2*x + (m+1)*y + 4 = 0) ∧ 
  (∃ (x y : ℝ), m*x + 3*y - 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2*x₁ + (m+1)*y₁ + 4 = 0 ∧ m*x₂ + 3*y₂ - 2 = 0 → 
    (2 / (m+1) = m / 3)) →
  m = -3 ∨ m = 2 := by
sorry

end parallel_lines_m_values_l2109_210942


namespace canal_length_scientific_notation_l2109_210910

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The length of the Beijing-Hangzhou Grand Canal in meters -/
def canal_length : ℕ := 1790000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem canal_length_scientific_notation :
  to_scientific_notation canal_length = ScientificNotation.mk 1.79 6 :=
sorry

end canal_length_scientific_notation_l2109_210910


namespace largest_power_l2109_210928

theorem largest_power (a b c d e : ℕ) :
  a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 8 ∧ e = 16 →
  c^8 ≥ a^20 ∧ c^8 ≥ b^14 ∧ c^8 ≥ d^5 ∧ c^8 ≥ e^3 :=
by sorry

#check largest_power

end largest_power_l2109_210928


namespace fish_pond_population_l2109_210918

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (second_catch : ℚ) →
  (initial_tagged * second_catch : ℚ) / (tagged_in_second : ℚ) = 3200 :=
by sorry

end fish_pond_population_l2109_210918


namespace tangent_lines_imply_a_greater_than_three_l2109_210990

/-- The function f(x) = -x^3 + ax^2 - 2x --/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 2*x

/-- The derivative of f(x) --/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x - 2

/-- The condition for a line to be tangent to f(x) at point t --/
def is_tangent (a : ℝ) (t : ℝ) : Prop :=
  -1 + t^3 - a*t^2 + 2*t = (-3*t^2 + 2*a*t - 2)*(-t)

/-- The theorem statement --/
theorem tangent_lines_imply_a_greater_than_three (a : ℝ) :
  (∃ t₁ t₂ t₃ : ℝ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧
    is_tangent a t₁ ∧ is_tangent a t₂ ∧ is_tangent a t₃) →
  a > 3 :=
sorry

end tangent_lines_imply_a_greater_than_three_l2109_210990


namespace two_integers_sum_squares_product_perfect_square_l2109_210926

/-- There exist two integers less than 10 whose sum of squares plus their product is a perfect square. -/
theorem two_integers_sum_squares_product_perfect_square :
  ∃ a b : ℤ, a < 10 ∧ b < 10 ∧ ∃ k : ℤ, a^2 + b^2 + a*b = k^2 := by
  sorry

end two_integers_sum_squares_product_perfect_square_l2109_210926


namespace probability_of_six_red_balls_l2109_210911

def total_balls : ℕ := 100
def red_balls : ℕ := 80
def white_balls : ℕ := 20
def drawn_balls : ℕ := 10
def red_drawn : ℕ := 6

theorem probability_of_six_red_balls :
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls = 
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls := by sorry

end probability_of_six_red_balls_l2109_210911


namespace f_equals_cos_2x_l2109_210937

theorem f_equals_cos_2x (x : ℝ) : 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.cos (2 * x) := by sorry

end f_equals_cos_2x_l2109_210937


namespace optimal_threshold_at_intersection_l2109_210950

/-- Represents the height distribution of vehicles for a given class --/
def HeightDistribution := ℝ → ℝ

/-- The cost for class 1 vehicles --/
def class1Cost : ℝ := 200

/-- The cost for class 2 vehicles --/
def class2Cost : ℝ := 300

/-- The height distribution for class 1 vehicles --/
noncomputable def class1Distribution : HeightDistribution := sorry

/-- The height distribution for class 2 vehicles --/
noncomputable def class2Distribution : HeightDistribution := sorry

/-- The intersection point of the two height distributions --/
noncomputable def intersectionPoint : ℝ := sorry

/-- The error function for a given threshold --/
def errorFunction (h : ℝ) : ℝ := sorry

/-- Theorem: The optimal threshold that minimizes classification errors
    is at the intersection point of the two height distributions --/
theorem optimal_threshold_at_intersection :
  ∀ h : ℝ, h ≠ intersectionPoint → errorFunction h > errorFunction intersectionPoint :=
by sorry

end optimal_threshold_at_intersection_l2109_210950


namespace jerome_has_zero_left_l2109_210976

/-- Represents Jerome's financial transactions --/
def jerome_transactions (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ) : ℝ := by
  -- Convert initial amount to dollars
  let initial_dollars := initial_euros * exchange_rate
  -- Subtract Meg's amount
  let after_meg := initial_dollars - meg_amount
  -- Subtract Bianca's amount (thrice Meg's)
  let after_bianca := after_meg - (3 * meg_amount)
  -- Give all remaining money to Nathan
  exact 0

/-- Theorem stating that Jerome has $0 left after transactions --/
theorem jerome_has_zero_left : 
  ∀ (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ),
  initial_euros > 0 ∧ exchange_rate > 0 ∧ meg_amount > 0 →
  jerome_transactions initial_euros exchange_rate meg_amount = 0 := by
  sorry

#check jerome_has_zero_left

end jerome_has_zero_left_l2109_210976


namespace vasya_distance_fraction_l2109_210952

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℚ
  vasya : ℚ
  sasha : ℚ
  dima : ℚ

/-- Theorem stating that given the conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_distance_fraction 
  (df : DistanceFractions)
  (h1 : df.anton = df.vasya / 2)
  (h2 : df.sasha = df.anton + df.dima)
  (h3 : df.dima = 1 / 10)
  (h4 : df.anton + df.vasya + df.sasha + df.dima = 1) :
  df.vasya = 2 / 5 := by
  sorry

end vasya_distance_fraction_l2109_210952


namespace student_number_problem_l2109_210994

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end student_number_problem_l2109_210994


namespace firm_partners_count_l2109_210909

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 35) = 1 / 34 →
  partners = 14 :=
by sorry

end firm_partners_count_l2109_210909


namespace quadratic_sum_bounds_l2109_210931

theorem quadratic_sum_bounds (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 11)
  (eq2 : b^2 + b*c + c^2 = 11) :
  0 ≤ c^2 + c*a + a^2 ∧ c^2 + c*a + a^2 ≤ 44 := by
  sorry

end quadratic_sum_bounds_l2109_210931


namespace not_all_naturals_equal_l2109_210904

-- Define the statement we want to disprove
def all_naturals_equal (n : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (∀ i j, i < n → j < n → a i = a j)

-- Theorem stating that the above statement is false
theorem not_all_naturals_equal : ¬ (∀ n : ℕ, all_naturals_equal n) := by
  sorry

-- Note: The proof is omitted (replaced with 'sorry') as per the instructions

end not_all_naturals_equal_l2109_210904


namespace students_not_enrolled_l2109_210929

theorem students_not_enrolled (total : ℕ) (biology_frac : ℚ) (chemistry_frac : ℚ) (physics_frac : ℚ) 
  (h_total : total = 1500)
  (h_biology : biology_frac = 2/5)
  (h_chemistry : chemistry_frac = 3/8)
  (h_physics : physics_frac = 1/10)
  (h_no_overlap : biology_frac + chemistry_frac + physics_frac ≤ 1) :
  total - (⌊biology_frac * total⌋ + ⌊chemistry_frac * total⌋ + ⌊physics_frac * total⌋) = 188 := by
  sorry

end students_not_enrolled_l2109_210929


namespace percentage_difference_l2109_210963

theorem percentage_difference : 
  (55 / 100 * 40) - (4 / 5 * 25) = 2 := by
  sorry

end percentage_difference_l2109_210963


namespace managers_wage_l2109_210982

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The wages at Joe's Steakhouse satisfy the given conditions -/
def valid_wages (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.25 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.1875

theorem managers_wage (w : Wages) (h : valid_wages w) : w.manager = 8.5 := by
  sorry

end managers_wage_l2109_210982


namespace original_circle_area_l2109_210930

/-- Given a circle whose area increases by 8 times and whose circumference
    increases by 50.24 centimeters, prove that its original area is 50.24 square centimeters. -/
theorem original_circle_area (r : ℝ) (h1 : r > 0) : 
  (π * (r + 50.24 / (2 * π))^2 = 9 * π * r^2) ∧ 
  (2 * π * (r + 50.24 / (2 * π)) = 2 * π * r + 50.24) → 
  π * r^2 = 50.24 := by
  sorry

end original_circle_area_l2109_210930


namespace percentage_of_women_in_parent_group_l2109_210914

theorem percentage_of_women_in_parent_group (women_fulltime : Real) 
  (men_fulltime : Real) (total_not_fulltime : Real) :
  women_fulltime = 0.9 →
  men_fulltime = 0.75 →
  total_not_fulltime = 0.19 →
  ∃ (w : Real), w ≥ 0 ∧ w ≤ 1 ∧
    w * (1 - women_fulltime) + (1 - w) * (1 - men_fulltime) = total_not_fulltime ∧
    w = 0.4 := by
  sorry

end percentage_of_women_in_parent_group_l2109_210914


namespace system_solution_unique_l2109_210945

theorem system_solution_unique : 
  ∃! (x y z : ℝ), 3*x + 2*y - z = 4 ∧ 2*x - y + 3*z = 9 ∧ x - 2*y + 2*z = 3 ∧ x = 1 ∧ y = 2 ∧ z = 3 :=
by sorry

end system_solution_unique_l2109_210945


namespace number_division_puzzle_l2109_210959

theorem number_division_puzzle (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = (a + b) / (2 * a) ∧ a / b ≠ 1 → a / b = -1/2 := by
sorry

end number_division_puzzle_l2109_210959


namespace ben_has_fifteen_shirts_l2109_210989

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of additional shirts Ben has compared to Joe -/
def ben_extra_shirts : ℕ := 8

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := joe_shirts + ben_extra_shirts

theorem ben_has_fifteen_shirts : ben_shirts = 15 := by
  sorry

end ben_has_fifteen_shirts_l2109_210989


namespace determinant_evaluation_l2109_210969

theorem determinant_evaluation (x y z : ℝ) : 
  Matrix.det !![x + 1, y, z; y, x + 1, z; z, y, x + 1] = 
    x^3 + 3*x^2 + 3*x + 1 - x*y*z - x*y^2 - y*z^2 - z*x^2 - z*x + y*z^2 + z*y^2 := by
  sorry

end determinant_evaluation_l2109_210969


namespace problem_statement_l2109_210940

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ m = 1/4) ∧
  (∀ x : ℝ, 4/a + 1/b ≥ |2*x - 1| - |x + 2| ↔ -6 ≤ x ∧ x ≤ 12) :=
by sorry

end problem_statement_l2109_210940


namespace colin_average_mile_time_l2109_210901

def average_mile_time (first_mile : ℕ) (second_mile : ℕ) (third_mile : ℕ) (fourth_mile : ℕ) : ℚ :=
  (first_mile + second_mile + third_mile + fourth_mile) / 4

theorem colin_average_mile_time :
  average_mile_time 6 5 5 4 = 5 := by
  sorry

end colin_average_mile_time_l2109_210901


namespace equation_solution_l2109_210984

theorem equation_solution : 
  ∃ x : ℚ, (x + 3*x = 500 - (4*x + 5*x)) ∧ (x = 500/13) := by
  sorry

end equation_solution_l2109_210984


namespace overlapping_circles_area_l2109_210988

/-- The area of the common part of two equal circles with radius R,
    where the circumference of each circle passes through the center of the other. -/
theorem overlapping_circles_area (R : ℝ) (R_pos : R > 0) :
  ∃ (A : ℝ), A = R^2 * (4 * Real.pi - 3 * Real.sqrt 3) / 6 ∧
  A = 2 * (1/3 * Real.pi * R^2 - R^2 * Real.sqrt 3 / 4) :=
by sorry

end overlapping_circles_area_l2109_210988


namespace forest_farms_theorem_l2109_210991

-- Define a farm as a pair of natural numbers (total years, high-quality years)
def Farm := ℕ × ℕ

-- Function to calculate probability of selecting two high-quality years
def prob_two_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  (high.choose 2 : ℚ) / (total.choose 2 : ℚ)

-- Function to calculate probability of selecting a high-quality year
def prob_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  high / total

-- Distribution type for discrete random variable
def Distribution := List (ℕ × ℚ)

-- Function to calculate the distribution of high-quality projects
def distribution_high_quality (f1 f2 f3 : Farm) : Distribution :=
  sorry  -- Placeholder for the actual calculation

-- Main theorem
theorem forest_farms_theorem (farm_b farm_c : Farm) :
  -- Part 1
  prob_two_high_quality (7, 4) = 2/7 ∧
  -- Part 2
  distribution_high_quality (6, 3) (7, 4) (10, 5) = 
    [(0, 3/28), (1, 5/14), (2, 11/28), (3, 1/7)] ∧
  -- Part 3
  ∃ (avg_b avg_c : ℚ), 
    prob_high_quality farm_b = 4/7 ∧ 
    prob_high_quality farm_c = 1/2 ∧ 
    avg_b ≠ avg_c :=
by sorry

end forest_farms_theorem_l2109_210991


namespace pyramid_surface_area_l2109_210939

/-- The total surface area of a pyramid formed from a cube -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (1/2 * base_side * slant_height)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

end pyramid_surface_area_l2109_210939


namespace profit_achieved_min_lemons_optimal_l2109_210960

/-- The number of lemons bought in one purchase -/
def lemons_bought : ℕ := 4

/-- The cost in cents for buying lemons_bought lemons -/
def buying_cost : ℕ := 25

/-- The number of lemons sold in one sale -/
def lemons_sold : ℕ := 7

/-- The revenue in cents from selling lemons_sold lemons -/
def selling_revenue : ℕ := 50

/-- The desired profit in cents -/
def desired_profit : ℕ := 150

/-- The minimum number of lemons needed to be sold to achieve the desired profit -/
def min_lemons_to_sell : ℕ := 169

theorem profit_achieved (n : ℕ) : n ≥ min_lemons_to_sell →
  (n * selling_revenue / lemons_sold - n * buying_cost / lemons_bought) ≥ desired_profit :=
by sorry

theorem min_lemons_optimal : 
  ∀ m : ℕ, m < min_lemons_to_sell →
  (m * selling_revenue / lemons_sold - m * buying_cost / lemons_bought) < desired_profit :=
by sorry

end profit_achieved_min_lemons_optimal_l2109_210960


namespace inequality_solution_set_l2109_210938

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 3) = {x | (x - 3) * (x + 2) < 0} :=
by sorry

end inequality_solution_set_l2109_210938


namespace reflected_ray_tangent_to_circle_l2109_210913

-- Define the initial ray of light
def initial_ray (x y : ℝ) : Prop := x + 2*y + 2 + Real.sqrt 5 = 0 ∧ y ≥ 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Define a function to check if a point is on the circle
def on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem reflected_ray_tangent_to_circle :
  ∃ (x y : ℝ), initial_ray x y ∧ 
               x_axis y ∧
               on_circle x y ∧
               ∀ (x' y' : ℝ), on_circle x' y' → 
                 ((x' - x)^2 + (y' - y)^2 ≥ 1 ∨ (x' = x ∧ y' = y)) :=
sorry

end reflected_ray_tangent_to_circle_l2109_210913


namespace unique_values_l2109_210995

/-- The polynomial we're working with -/
def f (p q : ℤ) (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 8

/-- The condition that the polynomial is divisible by (x + 2)(x - 1) -/
def is_divisible (p q : ℤ) : Prop :=
  ∀ x : ℝ, (x + 2 = 0 ∨ x - 1 = 0) → f p q x = 0

/-- The theorem stating that p = -54 and q = -48 are the unique values satisfying the condition -/
theorem unique_values :
  ∃! (p q : ℤ), is_divisible p q ∧ p = -54 ∧ q = -48 := by sorry

end unique_values_l2109_210995


namespace square_area_increase_l2109_210954

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.2 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.44
  := by sorry

end square_area_increase_l2109_210954


namespace loss_equals_twenty_pencils_l2109_210934

/-- The number of pencils purchased -/
def num_pencils : ℕ := 70

/-- The ratio of cost to selling price for the total purchase -/
def cost_to_sell_ratio : ℚ := 1.2857142857142856

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem loss_equals_twenty_pencils :
  ∀ (cost_per_pencil sell_per_pencil : ℚ),
  cost_per_pencil = cost_to_sell_ratio * sell_per_pencil →
  (num_pencils : ℚ) * (cost_per_pencil - sell_per_pencil) = (loss_in_pencils : ℚ) * sell_per_pencil :=
by sorry

end loss_equals_twenty_pencils_l2109_210934


namespace school_buses_l2109_210941

def bus_seats (columns : ℕ) (rows : ℕ) : ℕ := columns * rows

def total_capacity (num_buses : ℕ) (seats_per_bus : ℕ) : ℕ := num_buses * seats_per_bus

theorem school_buses (columns : ℕ) (rows : ℕ) (total_students : ℕ) (num_buses : ℕ) :
  columns = 4 →
  rows = 10 →
  total_students = 240 →
  total_capacity num_buses (bus_seats columns rows) = total_students →
  num_buses = 6 := by
  sorry

#check school_buses

end school_buses_l2109_210941


namespace union_of_A_and_B_l2109_210992

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end union_of_A_and_B_l2109_210992


namespace olivias_initial_money_l2109_210983

theorem olivias_initial_money (initial_money : ℕ) : 
  (initial_money + 91 - (91 + 39) = 14) → initial_money = 53 := by
  sorry

end olivias_initial_money_l2109_210983


namespace prime_n_l2109_210921

theorem prime_n (p h n : ℕ) : 
  Nat.Prime p → 
  h < p → 
  n = p * h + 1 → 
  (2^(n-1) - 1) % n = 0 → 
  (2^h - 1) % n ≠ 0 → 
  Nat.Prime n := by
sorry

end prime_n_l2109_210921


namespace total_equipment_cost_l2109_210986

def num_players : ℕ := 16
def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80

theorem total_equipment_cost :
  (num_players : ℚ) * (jersey_cost + shorts_cost + socks_cost) = 752 := by
  sorry

end total_equipment_cost_l2109_210986


namespace quadratic_sets_solution_l2109_210977

/-- Given sets A and B defined by quadratic equations, prove the values of a, b, and c -/
theorem quadratic_sets_solution :
  ∀ (a b c : ℝ),
  let A := {x : ℝ | x^2 + a*x + b = 0}
  let B := {x : ℝ | x^2 + c*x + 15 = 0}
  (A ∪ B = {3, 5} ∧ A ∩ B = {3}) →
  (a = -6 ∧ b = 9 ∧ c = -8) :=
by
  sorry


end quadratic_sets_solution_l2109_210977


namespace book_arrangement_proof_l2109_210985

def num_arrangements (num_books : ℕ) (num_pushkin : ℕ) (num_tarle : ℕ) (height_pushkin : ℕ) (height_tarle : ℕ) (height_center : ℕ) : ℕ :=
  3 * (Nat.factorial 2) * (Nat.factorial 4)

theorem book_arrangement_proof :
  num_arrangements 7 2 4 30 25 40 = 144 := by
  sorry

end book_arrangement_proof_l2109_210985


namespace rectangular_prism_surface_area_l2109_210967

/-- The surface area of a rectangular prism formed by three cubes -/
def surface_area_rectangular_prism (a : ℝ) : ℝ := 14 * a^2

/-- The surface area of a single cube -/
def surface_area_cube (a : ℝ) : ℝ := 6 * a^2

theorem rectangular_prism_surface_area (a : ℝ) (h : a > 0) :
  surface_area_rectangular_prism a = 3 * surface_area_cube a - 4 * a^2 :=
sorry

end rectangular_prism_surface_area_l2109_210967


namespace unique_digit_equation_l2109_210987

/-- Represents a mapping from symbols to digits -/
def SymbolMap := Char → Fin 10

/-- Checks if a SymbolMap assigns unique digits to different symbols -/
def isValidMap (m : SymbolMap) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Represents the equation "华 ÷ (3 * 好) = 杯赛" -/
def equationHolds (m : SymbolMap) : Prop :=
  (m '华').val = (m '杯').val * 100 + (m '赛').val * 10 + (m '赛').val

theorem unique_digit_equation :
  ∀ m : SymbolMap,
    isValidMap m →
    equationHolds m →
    (m '好').val = 2 := by sorry

end unique_digit_equation_l2109_210987


namespace seating_arrangement_exists_l2109_210957

-- Define a type for people
def Person : Type := Fin 5

-- Define a relation for acquaintance
def Acquainted : Person → Person → Prop := sorry

-- Define the condition that among any 3 people, 2 know each other and 2 don't
axiom acquaintance_condition : 
  ∀ (a b c : Person), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((Acquainted a b ∧ Acquainted a c) ∨ 
     (Acquainted a b ∧ Acquainted b c) ∨ 
     (Acquainted a c ∧ Acquainted b c)) ∧
    ((¬Acquainted a b ∧ ¬Acquainted a c) ∨ 
     (¬Acquainted a b ∧ ¬Acquainted b c) ∨ 
     (¬Acquainted a c ∧ ¬Acquainted b c))

-- Define a circular arrangement
def CircularArrangement : Type := Fin 5 → Person

-- Define the property that each person is adjacent to two acquaintances
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ (i : Fin 5), 
    Acquainted (arr i) (arr ((i + 1) % 5)) ∧ 
    Acquainted (arr i) (arr ((i + 4) % 5))

-- The theorem to be proved
theorem seating_arrangement_exists : 
  ∃ (arr : CircularArrangement), ValidArrangement arr :=
sorry

end seating_arrangement_exists_l2109_210957


namespace chips_yield_more_ounces_l2109_210996

def total_ounces (budget : ℚ) (price_per_bag : ℚ) (ounces_per_bag : ℚ) : ℚ :=
  (budget / price_per_bag).floor * ounces_per_bag

theorem chips_yield_more_ounces : 
  let budget : ℚ := 7
  let candy_price : ℚ := 1
  let candy_ounces : ℚ := 12
  let chips_price : ℚ := 1.4
  let chips_ounces : ℚ := 17
  total_ounces budget chips_price chips_ounces > total_ounces budget candy_price candy_ounces := by
  sorry

end chips_yield_more_ounces_l2109_210996


namespace S_is_infinite_l2109_210974

/-- Number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (m : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by sorry

end S_is_infinite_l2109_210974


namespace pauls_caramel_candy_boxes_l2109_210943

/-- Given that Paul bought 6 boxes of chocolate candy, each box has 9 pieces,
    and he had 90 candies in total, prove that he bought 4 boxes of caramel candy. -/
theorem pauls_caramel_candy_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) :
  chocolate_boxes = 6 →
  pieces_per_box = 9 →
  total_candies = 90 →
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box = 4 := by
  sorry

end pauls_caramel_candy_boxes_l2109_210943


namespace gcd_plus_lcm_eq_sum_iff_divides_l2109_210932

theorem gcd_plus_lcm_eq_sum_iff_divides (x y : ℕ) :
  (Nat.gcd x y + x * y / Nat.gcd x y = x + y) ↔ (y ∣ x ∨ x ∣ y) := by
  sorry

end gcd_plus_lcm_eq_sum_iff_divides_l2109_210932


namespace integer_product_condition_l2109_210946

theorem integer_product_condition (a : ℚ) : 
  (∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = k) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end integer_product_condition_l2109_210946


namespace pauls_birthday_crayons_l2109_210978

/-- The number of crayons Paul received for his birthday -/
def crayons_received (crayons_left : ℕ) (crayons_lost_or_given : ℕ) 
  (crayons_lost : ℕ) (crayons_given : ℕ) : ℕ :=
  crayons_left + crayons_lost_or_given

/-- Theorem stating the number of crayons Paul received for his birthday -/
theorem pauls_birthday_crayons :
  ∃ (crayons_lost crayons_given : ℕ),
    crayons_lost = 2 * crayons_given ∧
    crayons_lost + crayons_given = 9750 ∧
    crayons_received 2560 9750 crayons_lost crayons_given = 12310 := by
  sorry


end pauls_birthday_crayons_l2109_210978


namespace roots_sum_and_product_l2109_210903

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 7 = 0 → 
  q^2 - 5*q + 7 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 559 := by
sorry

end roots_sum_and_product_l2109_210903


namespace stratified_sampling_first_grade_l2109_210933

theorem stratified_sampling_first_grade (total_students : ℕ) (sampled_students : ℕ) (first_grade_students : ℕ) :
  total_students = 2400 →
  sampled_students = 100 →
  first_grade_students = 840 →
  (first_grade_students * sampled_students) / total_students = 35 :=
by
  sorry

end stratified_sampling_first_grade_l2109_210933


namespace least_integer_with_12_factors_l2109_210970

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by sorry

end least_integer_with_12_factors_l2109_210970


namespace smallest_number_proof_l2109_210973

theorem smallest_number_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 73)
  (h4 : c - b = 5)
  (h5 : b - a = 6) :
  a = 56 / 3 := by
sorry

end smallest_number_proof_l2109_210973


namespace email_difference_l2109_210956

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end email_difference_l2109_210956


namespace square_dissection_theorem_l2109_210993

/-- A dissection of a square is a list of polygons that can be rearranged to form the original square. -/
def Dissection (n : ℕ) := List (List (ℕ × ℕ))

/-- A function that checks if a list of polygons can be arranged to form a square of side length n. -/
def CanFormSquare (pieces : List (List (ℕ × ℕ))) (n : ℕ) : Prop := sorry

/-- A function that checks if two lists of polygons are equivalent up to translation and rotation. -/
def AreEquivalent (pieces1 pieces2 : List (List (ℕ × ℕ))) : Prop := sorry

theorem square_dissection_theorem :
  ∃ (d : Dissection 7),
    d.length ≤ 5 ∧
    ∃ (s1 s2 s3 : List (List (ℕ × ℕ))),
      CanFormSquare s1 6 ∧
      CanFormSquare s2 3 ∧
      CanFormSquare s3 2 ∧
      AreEquivalent (s1 ++ s2 ++ s3) d :=
sorry

end square_dissection_theorem_l2109_210993


namespace compound_oxygen_atoms_l2109_210962

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (compound : Compound) : ℕ :=
  compound.h * atomic_weight "H" +
  compound.c * atomic_weight "C" +
  compound.o * atomic_weight "O"

/-- Theorem: A compound with 2 H atoms, 1 C atom, and molecular weight 62 amu has 3 O atoms -/
theorem compound_oxygen_atoms (compound : Compound) :
  compound.h = 2 ∧ compound.c = 1 ∧ molecular_weight compound = 62 →
  compound.o = 3 := by
  sorry

end compound_oxygen_atoms_l2109_210962


namespace negation_of_universal_proposition_l2109_210951

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end negation_of_universal_proposition_l2109_210951


namespace school_population_l2109_210935

/-- Given the initial number of girls and boys in a school, and the number of additional girls who joined,
    calculate the total number of pupils after the new girls joined. -/
theorem school_population (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
sorry

end school_population_l2109_210935


namespace sum_of_three_integers_l2109_210900

theorem sum_of_three_integers (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + b + c = 131 := by
sorry

end sum_of_three_integers_l2109_210900


namespace min_sticks_to_break_12_can_form_square_15_l2109_210924

-- Define a function to calculate the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define a function to check if it's possible to form a square without breaking sticks
def can_form_square (n : ℕ) : Bool :=
  sum_to_n n % 4 = 0

-- Define a function to find the minimum number of sticks to break
def min_sticks_to_break (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else if n = 12 then 2
  else sorry  -- We don't have a general formula for other cases

-- Theorem for n = 12
theorem min_sticks_to_break_12 :
  min_sticks_to_break 12 = 2 :=
by sorry

-- Theorem for n = 15
theorem can_form_square_15 :
  can_form_square 15 = true :=
by sorry

end min_sticks_to_break_12_can_form_square_15_l2109_210924


namespace smallest_four_digit_palindrome_div_by_3_odd_first_l2109_210944

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ Odd (n / 1000)

/-- The theorem stating that 1221 is the smallest four-digit palindrome 
    divisible by 3 with an odd first digit -/
theorem smallest_four_digit_palindrome_div_by_3_odd_first : 
  (∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 ∧ has_odd_first_digit n → n ≥ 1221) ∧
  is_four_digit_palindrome 1221 ∧ 1221 % 3 = 0 ∧ has_odd_first_digit 1221 :=
by sorry

end smallest_four_digit_palindrome_div_by_3_odd_first_l2109_210944


namespace quadratic_inequality_relationship_l2109_210916

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- Define proposition A
def proposition_A (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement
theorem quadratic_inequality_relationship :
  (∀ a : ℝ, proposition_A a → proposition_B a) ∧
  (∃ a : ℝ, proposition_B a ∧ ¬proposition_A a) :=
sorry

end quadratic_inequality_relationship_l2109_210916


namespace tax_ratio_is_300_2001_l2109_210915

/-- Represents the lottery winnings and expenses scenario --/
structure LotteryScenario where
  winnings : ℚ
  taxRate : ℚ
  loanRate : ℚ
  savings : ℚ
  investmentRate : ℚ
  funMoney : ℚ

/-- Calculates the tax amount given a lottery scenario --/
def calculateTax (scenario : LotteryScenario) : ℚ :=
  scenario.winnings * scenario.taxRate

/-- Theorem stating that the tax ratio is 300:2001 given the specific scenario --/
theorem tax_ratio_is_300_2001 (scenario : LotteryScenario)
  (h1 : scenario.winnings = 12006)
  (h2 : scenario.loanRate = 1/3)
  (h3 : scenario.savings = 1000)
  (h4 : scenario.investmentRate = 1/5)
  (h5 : scenario.funMoney = 2802)
  (h6 : scenario.winnings * (1 - scenario.taxRate) * (1 - scenario.loanRate) - scenario.savings * (1 + scenario.investmentRate) = 2 * scenario.funMoney) :
  (calculateTax scenario) / scenario.winnings = 300 / 2001 := by
sorry

#eval 300 / 2001

end tax_ratio_is_300_2001_l2109_210915


namespace processing_time_theorem_l2109_210979

/-- Calculates the total processing time in hours for a set of pictures --/
def total_processing_time (tree_count : ℕ) (flower_count : ℕ) (grass_count : ℕ) 
  (tree_time : ℚ) (flower_time : ℚ) (grass_time : ℚ) : ℚ :=
  ((tree_count : ℚ) * tree_time + (flower_count : ℚ) * flower_time + (grass_count : ℚ) * grass_time) / 60

/-- Theorem stating the total processing time for the given set of pictures --/
theorem processing_time_theorem : 
  total_processing_time 320 400 240 (3/2) (5/2) 1 = 860/30 := by
  sorry

end processing_time_theorem_l2109_210979


namespace sqrt_six_range_l2109_210971

theorem sqrt_six_range : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_six_range_l2109_210971


namespace complex_equation_solutions_l2109_210902

open Complex

theorem complex_equation_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1)) ∧
    S.card = 8 ∧
    (∀ z : ℂ, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1) → z ∈ S) :=
by sorry

end complex_equation_solutions_l2109_210902


namespace num_divisors_360_eq_24_l2109_210927

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end num_divisors_360_eq_24_l2109_210927


namespace greatest_even_perfect_square_under_200_l2109_210955

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem greatest_even_perfect_square_under_200 :
  ∀ n : ℕ, is_perfect_square n → is_even n → n < 200 → n ≤ 196 :=
sorry

end greatest_even_perfect_square_under_200_l2109_210955


namespace parabola_c_value_l2109_210999

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 5 →  -- vertex condition
  p.x_coord 6 = 0 →  -- point condition
  p.c = 0 := by
sorry

end parabola_c_value_l2109_210999


namespace homework_ratio_l2109_210964

theorem homework_ratio (total : ℕ) (finished : ℕ) 
  (h1 : total = 65) (h2 : finished = 45) : 
  (finished : ℚ) / (total - finished) = 9 / 4 := by
  sorry

end homework_ratio_l2109_210964


namespace square_minus_a_nonpositive_iff_a_geq_four_l2109_210919

theorem square_minus_a_nonpositive_iff_a_geq_four :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end square_minus_a_nonpositive_iff_a_geq_four_l2109_210919


namespace sum_of_digits_of_power_l2109_210953

/-- Sum of tens and ones digits of (3+4)^11 -/
theorem sum_of_digits_of_power : ∃ (n : ℕ), 
  (3 + 4)^11 = n ∧ 
  (n / 10 % 10 + n % 10 = 7) := by
  sorry

end sum_of_digits_of_power_l2109_210953


namespace kanul_spending_l2109_210947

/-- The total amount Kanul had initially -/
def T : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The fraction of the total amount spent as cash -/
def cash_fraction : ℝ := 0.30

theorem kanul_spending :
  raw_materials + machinery + cash_fraction * T = T := by sorry

end kanul_spending_l2109_210947


namespace constant_ratio_solution_l2109_210997

/-- The constant ratio of (3x - 4) to (y + 15) -/
def k (x y : ℚ) : ℚ := (3 * x - 4) / (y + 15)

theorem constant_ratio_solution (x₀ y₀ x₁ y₁ : ℚ) 
  (h₀ : y₀ = 4)
  (h₁ : x₀ = 5)
  (h₂ : y₁ = 15)
  (h₃ : k x₀ y₀ = k x₁ y₁) :
  x₁ = 406 / 57 := by
  sorry

end constant_ratio_solution_l2109_210997


namespace triangle_construction_from_nagel_point_vertex_and_altitude_foot_l2109_210906

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- The Nagel point of a triangle -/
def nagelPoint (t : Triangle) : Point2D := sorry

/-- The foot of the altitude from a vertex -/
def altitudeFoot (t : Triangle) (v : Point2D) : Point2D := sorry

/-- Theorem: Given a Nagel point, a vertex, and the foot of the altitude from that vertex,
    a triangle can be constructed -/
theorem triangle_construction_from_nagel_point_vertex_and_altitude_foot
  (N : Point2D) (B : Point2D) (F : Point2D) :
  ∃ (t : Triangle), nagelPoint t = N ∧ t.B = B ∧ altitudeFoot t B = F :=
sorry

end triangle_construction_from_nagel_point_vertex_and_altitude_foot_l2109_210906


namespace probability_two_red_two_green_l2109_210925

theorem probability_two_red_two_green (total_red : ℕ) (total_green : ℕ) (drawn : ℕ) : 
  total_red = 10 → total_green = 8 → drawn = 4 →
  (Nat.choose total_red 2 * Nat.choose total_green 2) / Nat.choose (total_red + total_green) drawn = 7 / 17 := by
  sorry

end probability_two_red_two_green_l2109_210925


namespace function_symmetry_l2109_210948

theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x + 2 * f (1 / x) = 3 * x) :
  ∀ x ≠ 0, f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
sorry

end function_symmetry_l2109_210948


namespace rectangle_area_relation_l2109_210908

/-- 
For a rectangle with area 12 and sides of length x and y,
the function relationship between y and x is y = 12/x.
-/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 12) : 
  y = 12 / x := by
  sorry

end rectangle_area_relation_l2109_210908


namespace circular_arrangement_students_l2109_210975

theorem circular_arrangement_students (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 10 ≤ n ∧ 40 ≤ n) 
  (h3 : (40 - 10) * 2 = n) : n = 60 := by
  sorry

end circular_arrangement_students_l2109_210975


namespace ice_cream_consumption_l2109_210922

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream (friday_amount saturday_amount : Real) : Real :=
  friday_amount + saturday_amount

/-- Theorem stating the total amount of ice cream eaten -/
theorem ice_cream_consumption : 
  total_ice_cream 3.25 0.25 = 3.50 := by
  sorry

end ice_cream_consumption_l2109_210922


namespace negative_six_divided_by_three_l2109_210917

theorem negative_six_divided_by_three : (-6) / 3 = -2 := by
  sorry

end negative_six_divided_by_three_l2109_210917


namespace sum_of_first_20_lucky_numbers_mod_1000_l2109_210936

def isLucky (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

def luckyNumbers : List ℕ :=
  (List.range 20).map (λ i => 7 * (10^i - 1) / 9)

theorem sum_of_first_20_lucky_numbers_mod_1000 :
  (luckyNumbers.sum) % 1000 = 70 := by
  sorry

end sum_of_first_20_lucky_numbers_mod_1000_l2109_210936


namespace fold_triangle_crease_length_l2109_210905

theorem fold_triangle_crease_length 
  (A B C : ℝ × ℝ) 
  (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_side_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_side_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4)
  (h_side_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5) :
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let F : ℝ × ℝ := 
    let m := (B.2 - A.2) / (B.1 - A.1)
    let b := D.2 - m * D.1
    ((E.2 - b) / m, E.2)
  Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = 15/8 := by
sorry

end fold_triangle_crease_length_l2109_210905


namespace tenth_square_area_l2109_210923

/-- The area of the nth square in a sequence where each square is formed by connecting
    the midpoints of the previous square's sides, and the first square has a side length of 2. -/
def square_area (n : ℕ) : ℚ :=
  2 * (1 / 2) ^ (n - 1)

/-- Theorem stating that the area of the 10th square in the sequence is 1/256. -/
theorem tenth_square_area :
  square_area 10 = 1 / 256 := by
  sorry

end tenth_square_area_l2109_210923


namespace union_of_A_and_B_l2109_210961

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by
  sorry

end union_of_A_and_B_l2109_210961


namespace geometric_sequence_property_l2109_210949

def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_pos_geo : is_positive_geometric_sequence a)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end geometric_sequence_property_l2109_210949


namespace multiples_of_12_between_30_and_200_l2109_210958

theorem multiples_of_12_between_30_and_200 : 
  (Finset.filter (fun n => n % 12 = 0 ∧ n ≥ 30 ∧ n ≤ 200) (Finset.range 201)).card = 14 := by
  sorry

end multiples_of_12_between_30_and_200_l2109_210958


namespace intersection_A_complement_B_l2109_210981

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -2 ∨ x > 5}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end intersection_A_complement_B_l2109_210981


namespace quadratic_inequality_solution_l2109_210968

theorem quadratic_inequality_solution (x : ℤ) :
  1 ≤ x ∧ x ≤ 10 → (x^2 < 3*x ↔ x = 1 ∨ x = 2) := by
  sorry

end quadratic_inequality_solution_l2109_210968


namespace inequality_system_solution_set_l2109_210912

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 > 0 ∧ x - 3 < 2}
  S = Set.Ioo (-1) 5 := by
sorry

end inequality_system_solution_set_l2109_210912


namespace selling_price_articles_l2109_210907

/-- Proves that if the cost price of 50 articles equals the selling price of N articles,
    and the gain percent is 25%, then N = 40. -/
theorem selling_price_articles (C : ℝ) (N : ℕ) (h1 : N * (C + 0.25 * C) = 50 * C) : N = 40 := by
  sorry

#check selling_price_articles

end selling_price_articles_l2109_210907


namespace regular_polygon_with_160_degree_angles_l2109_210920

theorem regular_polygon_with_160_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- A polygon must have at least 3 sides
  (∀ i : ℕ, i < n → 160 = (n - 2) * 180 / n) →  -- Each interior angle is 160°
  n = 18 := by
  sorry

end regular_polygon_with_160_degree_angles_l2109_210920


namespace no_real_solution_complex_roots_l2109_210966

theorem no_real_solution_complex_roots :
  ∀ x : ℂ, (2 * x - 36) / 3 = (3 * x^2 + 6 * x + 1) / 4 →
  (∃ b : ℝ, x = -5/9 + b * I ∨ x = -5/9 - b * I) ∧
  (∀ y : ℝ, (2 * y - 36) / 3 ≠ (3 * y^2 + 6 * y + 1) / 4) :=
by sorry

end no_real_solution_complex_roots_l2109_210966


namespace smallest_angle_25_sided_polygon_l2109_210998

/-- Represents a convex polygon with n sides and angles in an arithmetic sequence --/
structure ConvexPolygon (n : ℕ) where
  -- The common difference of the arithmetic sequence of angles
  d : ℕ
  -- The smallest angle in the polygon
  smallest_angle : ℕ
  -- Ensure the polygon is convex (all angles less than 180°)
  convex : smallest_angle + (n - 1) * d < 180
  -- Ensure the sum of angles is correct for an n-sided polygon
  angle_sum : smallest_angle * n + (n * (n - 1) * d) / 2 = (n - 2) * 180

theorem smallest_angle_25_sided_polygon :
  ∃ (p : ConvexPolygon 25), p.smallest_angle = 154 := by
  sorry

end smallest_angle_25_sided_polygon_l2109_210998


namespace complex_cube_root_l2109_210980

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I :=
by sorry

end complex_cube_root_l2109_210980


namespace apple_tree_width_proof_l2109_210972

/-- The width of an apple tree in Quinton's backyard -/
def apple_tree_width : ℝ := 10

/-- The space between apple trees -/
def apple_tree_space : ℝ := 12

/-- The width of a peach tree -/
def peach_tree_width : ℝ := 12

/-- The space between peach trees -/
def peach_tree_space : ℝ := 15

/-- The total space taken by all trees -/
def total_space : ℝ := 71

theorem apple_tree_width_proof :
  2 * apple_tree_width + apple_tree_space + 2 * peach_tree_width + peach_tree_space = total_space :=
by sorry

end apple_tree_width_proof_l2109_210972


namespace count_standing_orders_l2109_210965

/-- The number of different standing orders for 9 students -/
def standing_orders : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 9

/-- The position of the tallest student (middle position) -/
def tallest_position : ℕ := 5

/-- The rank of the student who must stand next to the tallest -/
def adjacent_rank : ℕ := 4

/-- Theorem stating the number of different standing orders -/
theorem count_standing_orders :
  standing_orders = 20 ∧
  num_students = 9 ∧
  tallest_position = 5 ∧
  adjacent_rank = 4 := by
  sorry


end count_standing_orders_l2109_210965
