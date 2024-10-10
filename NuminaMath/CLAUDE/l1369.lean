import Mathlib

namespace ellipse_I_equation_ellipse_II_equation_l1369_136921

-- Part I
def ellipse_I (x y : ℝ) := x^2 / 2 + y^2 = 1

theorem ellipse_I_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_I x y ↔ 
    (x + 1)^2 + y^2 + ((x - 1)^2 + y^2).sqrt = 2 * a ∧
    a^2 - 1 = b^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1) ∧
  ellipse_I (1/2) (Real.sqrt 14 / 4) :=
sorry

-- Part II
def ellipse_II (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

theorem ellipse_II_equation :
  ellipse_II (Real.sqrt 2) (-1) ∧
  ellipse_II (-1) (Real.sqrt 6 / 2) :=
sorry

end ellipse_I_equation_ellipse_II_equation_l1369_136921


namespace triangle_inequalities_l1369_136904

-- Define a triangle with heights and an internal point
structure Triangle :=
  (h₁ h₂ h₃ u v w : ℝ)
  (h₁_pos : h₁ > 0)
  (h₂_pos : h₂ > 0)
  (h₃_pos : h₃ > 0)
  (u_pos : u > 0)
  (v_pos : v > 0)
  (w_pos : w > 0)

-- Theorem statement
theorem triangle_inequalities (t : Triangle) :
  (t.h₁ / t.u + t.h₂ / t.v + t.h₃ / t.w ≥ 9) ∧
  (t.h₁ * t.h₂ * t.h₃ ≥ 27 * t.u * t.v * t.w) ∧
  ((t.h₁ - t.u) * (t.h₂ - t.v) * (t.h₃ - t.w) ≥ 8 * t.u * t.v * t.w) :=
by sorry

end triangle_inequalities_l1369_136904


namespace unique_function_determination_l1369_136954

theorem unique_function_determination (f : ℝ → ℝ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y - 1)) :
  ∀ x : ℝ, f x = x + 1 := by
sorry

end unique_function_determination_l1369_136954


namespace isosceles_base_length_l1369_136984

/-- Represents a triangle with a perimeter -/
structure Triangle where
  perimeter : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle extends Triangle

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle extends Triangle where
  base : ℝ
  leg : ℝ

/-- Theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_base_length 
  (et : EquilateralTriangle) 
  (it : IsoscelesTriangle) 
  (h1 : et.perimeter = 60) 
  (h2 : it.perimeter = 45) 
  (h3 : it.leg = et.perimeter / 3) : 
  it.base = 5 := by sorry

end isosceles_base_length_l1369_136984


namespace cubic_sum_over_product_l1369_136928

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_sq_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end cubic_sum_over_product_l1369_136928


namespace apple_count_bottle_apple_relation_l1369_136978

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := 72

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 32

/-- The number of apples -/
def apples : ℕ := 78

/-- The difference between the number of bottles and apples -/
def bottle_apple_difference : ℕ := 26

/-- Theorem stating that the number of apples is 78 -/
theorem apple_count : apples = 78 := by
  sorry

/-- Theorem proving the relationship between bottles and apples -/
theorem bottle_apple_relation : 
  regular_soda + diet_soda = apples + bottle_apple_difference := by
  sorry

end apple_count_bottle_apple_relation_l1369_136978


namespace reciprocal_of_negative_five_l1369_136937

theorem reciprocal_of_negative_five :
  ∀ x : ℚ, x * (-5) = 1 → x = -1/5 := by
  sorry

end reciprocal_of_negative_five_l1369_136937


namespace parallel_lines_a_value_l1369_136938

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m₁ n₁ m₂ n₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁

/-- Line l₁ with equation x + ay + 6 = 0 -/
def l₁ (a : ℝ) (x y : ℝ) : Prop :=
  x + a * y + 6 = 0

/-- Line l₂ with equation (a-2)x + 3ay + 18 = 0 -/
def l₂ (a : ℝ) (x y : ℝ) : Prop :=
  (a - 2) * x + 3 * a * y + 18 = 0

/-- The main theorem stating that when l₁ and l₂ are parallel, a = 0 -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel_lines 1 a (a - 2) (3 * a) → a = 0 :=
by
  sorry

end parallel_lines_a_value_l1369_136938


namespace fixed_point_on_quadratic_graph_l1369_136901

/-- The fixed point on the graph of y = 9x^2 + mx - 5m for any real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (5 : ℝ)^2 + m * 5 - 5 * m = 225 := by
  sorry

end fixed_point_on_quadratic_graph_l1369_136901


namespace models_after_price_increase_l1369_136900

-- Define the original price, price increase percentage, and initial number of models
def original_price : ℚ := 45/100
def price_increase_percent : ℚ := 15/100
def initial_models : ℕ := 30

-- Calculate the new price after the increase
def new_price : ℚ := original_price * (1 + price_increase_percent)

-- Calculate the total savings
def total_savings : ℚ := original_price * initial_models

-- Define the theorem
theorem models_after_price_increase :
  ⌊total_savings / new_price⌋ = 26 := by
  sorry

#eval ⌊total_savings / new_price⌋

end models_after_price_increase_l1369_136900


namespace athletes_total_yards_l1369_136985

/-- Calculates the total yards run by three athletes over a given number of games -/
def total_yards (yards_per_game_1 yards_per_game_2 yards_per_game_3 : ℕ) (num_games : ℕ) : ℕ :=
  (yards_per_game_1 + yards_per_game_2 + yards_per_game_3) * num_games

/-- Proves that the total yards run by three athletes over 4 games is 204 yards -/
theorem athletes_total_yards :
  total_yards 18 22 11 4 = 204 := by
  sorry

#eval total_yards 18 22 11 4

end athletes_total_yards_l1369_136985


namespace functional_equation_characterization_l1369_136931

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f x + a * ⌊y⌋

/-- The set of valid 'a' values -/
def ValidASet : Set ℝ :=
  {a | ∃ n : ℤ, a = -(n^2 : ℝ)}

/-- The main theorem stating the equivalence -/
theorem functional_equation_characterization (a : ℝ) :
  (∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f a) ↔ a ∈ ValidASet :=
sorry

end functional_equation_characterization_l1369_136931


namespace profit_ratio_proportional_to_investment_l1369_136942

/-- The ratio of profits for two investors is proportional to their investments -/
theorem profit_ratio_proportional_to_investment 
  (p_investment q_investment : ℕ) 
  (hp : p_investment = 40000) 
  (hq : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
sorry

end profit_ratio_proportional_to_investment_l1369_136942


namespace B_power_101_l1369_136962

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 0, 0; 0, 1, 0]

theorem B_power_101 : B^101 = !![0, 0, 0; 0, 0, 0; 0, 0, 1] := by sorry

end B_power_101_l1369_136962


namespace sum_of_coefficients_excluding_constant_l1369_136950

/-- The sum of the coefficients of the terms, excluding the constant term, 
    in the expansion of (x^2 - 2/x)^6 is -239 -/
theorem sum_of_coefficients_excluding_constant (x : ℝ) : 
  let f := (x^2 - 2/x)^6
  let all_coeff_sum := (1 - 2)^6
  let constant_term := 240
  all_coeff_sum - constant_term = -239 := by
sorry

end sum_of_coefficients_excluding_constant_l1369_136950


namespace trigonometric_problem_l1369_136930

theorem trigonometric_problem (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) :
  (Real.cos β = 63 / 65) ∧
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
  sorry

end trigonometric_problem_l1369_136930


namespace system_solution_l1369_136976

theorem system_solution : ∃ (X Y : ℝ), 
  (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
  (X^2 * Y + X * Y + 1 = 0) ∧ 
  (X = -2) ∧ (Y = -1/2) := by
  sorry

end system_solution_l1369_136976


namespace max_expected_expenditure_l1369_136968

-- Define the parameters of the linear regression equation
def b : ℝ := 0.8
def a : ℝ := 2

-- Define the revenue
def revenue : ℝ := 10

-- Define the error bound
def error_bound : ℝ := 0.5

-- Theorem statement
theorem max_expected_expenditure :
  ∀ e : ℝ, |e| < error_bound →
  ∃ y : ℝ, y = b * revenue + a + e ∧ y ≤ 10.5 ∧
  ∀ y' : ℝ, (∃ e' : ℝ, |e'| < error_bound ∧ y' = b * revenue + a + e') → y' ≤ y :=
by sorry

end max_expected_expenditure_l1369_136968


namespace person_age_in_1930_l1369_136915

theorem person_age_in_1930 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (birth_year ≤ 1930) →
  (death_year > 1930) →
  (age_at_death = death_year - birth_year) →
  (age_at_death = birth_year / 31) →
  (1930 - birth_year = 39) :=
by sorry

end person_age_in_1930_l1369_136915


namespace not_right_triangle_3_5_7_l1369_136935

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (3, 5, 7) cannot form the sides of a right triangle -/
theorem not_right_triangle_3_5_7 : ¬ is_right_triangle 3 5 7 := by
  sorry

end not_right_triangle_3_5_7_l1369_136935


namespace angle_CRT_is_72_degrees_l1369_136918

-- Define the triangle CAT
structure Triangle (C A T : Type) where
  angle_ACT : ℝ
  angle_ATC : ℝ
  angle_CAT : ℝ

-- Define the theorem
theorem angle_CRT_is_72_degrees 
  (CAT : Triangle C A T) 
  (h1 : CAT.angle_ACT = CAT.angle_ATC) 
  (h2 : CAT.angle_CAT = 36) 
  (h3 : ∃ (R : Type), (angle_CTR : ℝ) = CAT.angle_ATC / 2) : 
  (angle_CRT : ℝ) = 72 := by
  sorry

end angle_CRT_is_72_degrees_l1369_136918


namespace quadratic_roots_sign_l1369_136912

theorem quadratic_roots_sign (a : ℝ) (h1 : a > 0) (h2 : a ≠ 0) :
  ¬∃ (c : ℝ → Prop), ∀ (x y : ℝ),
    (c a → (a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) ∧
    (¬c a → ¬(a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ x ≠ y ∧ x > 0 ∧ y < 0)) :=
by
  sorry

end quadratic_roots_sign_l1369_136912


namespace marges_garden_plants_l1369_136960

/-- Calculates the final number of plants in Marge's garden --/
def final_plant_count (total_seeds sunflower_seeds marigold_seeds seeds_not_grown : ℕ)
  (marigold_growth_rate sunflower_growth_rate : ℚ)
  (sunflower_wilt_rate marigold_eaten_rate pest_control_rate : ℚ)
  (weed_strangle_rate : ℚ) (weeds_pulled weeds_kept : ℕ) : ℕ :=
  sorry

/-- The theorem stating the final number of plants in Marge's garden --/
theorem marges_garden_plants :
  final_plant_count 23 13 10 5
    (4/10) (6/10) (1/4) (1/2) (3/4)
    (1/3) 2 1 = 6 :=
  sorry

end marges_garden_plants_l1369_136960


namespace equation_holds_l1369_136933

/-- Prove that for positive integers a, b, c with given conditions, 
    the equation (12a + b)(12a + c) = 144a(a + 1) + b + c holds. -/
theorem equation_holds (a b c : ℕ) 
  (ha : a < 12) (hb : b < 12) (hc : c < 12) (hbc : b + c = 12) : 
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c := by
  sorry

end equation_holds_l1369_136933


namespace largest_c_for_five_in_range_l1369_136979

/-- The quadratic function f(x) = 2x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem: The largest value of c such that 5 is in the range of f(x) = 2x^2 - 4x + c is 7 -/
theorem largest_c_for_five_in_range : 
  (∃ (x : ℝ), f 7 x = 5) ∧ 
  (∀ (c : ℝ), c > 7 → ¬∃ (x : ℝ), f c x = 5) := by
  sorry

end largest_c_for_five_in_range_l1369_136979


namespace krystiana_earnings_l1369_136952

/-- Represents the monthly earnings from an apartment building --/
def apartment_earnings (
  first_floor_price : ℕ)
  (second_floor_price : ℕ)
  (first_floor_rooms : ℕ)
  (second_floor_rooms : ℕ)
  (third_floor_rooms : ℕ)
  (third_floor_occupied : ℕ) : ℕ :=
  (first_floor_price * first_floor_rooms) +
  (second_floor_price * second_floor_rooms) +
  (2 * first_floor_price * third_floor_occupied)

/-- Krystiana's apartment building earnings theorem --/
theorem krystiana_earnings :
  apartment_earnings 15 20 3 3 3 2 = 165 := by
  sorry

#eval apartment_earnings 15 20 3 3 3 2

end krystiana_earnings_l1369_136952


namespace books_left_to_read_l1369_136923

theorem books_left_to_read 
  (total_books : ℕ) 
  (mcgregor_books : ℕ) 
  (floyd_books : ℕ) 
  (h1 : total_books = 89) 
  (h2 : mcgregor_books = 34) 
  (h3 : floyd_books = 32) : 
  total_books - (mcgregor_books + floyd_books) = 23 := by
sorry

end books_left_to_read_l1369_136923


namespace unique_solution_quadratic_inequality_l1369_136906

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) ↔ a = 2 := by
  sorry

end unique_solution_quadratic_inequality_l1369_136906


namespace complex_addition_l1369_136991

theorem complex_addition : ∃ z : ℂ, (5 - 3*I + z = -2 + 9*I) ∧ (z = -7 + 12*I) := by
  sorry

end complex_addition_l1369_136991


namespace shower_tiles_count_l1369_136958

/-- Represents the layout of a shower wall -/
structure WallLayout where
  rectangularTiles : ℕ
  triangularTiles : ℕ
  hexagonalTiles : ℕ
  squareTiles : ℕ

/-- Calculates the total number of tiles in the shower -/
def totalTiles (wall1 wall2 wall3 : WallLayout) : ℕ :=
  wall1.rectangularTiles + wall1.triangularTiles +
  wall2.rectangularTiles + wall2.triangularTiles + wall2.hexagonalTiles +
  wall3.squareTiles + wall3.triangularTiles

/-- Theorem stating the total number of tiles in the shower -/
theorem shower_tiles_count :
  let wall1 : WallLayout := ⟨12 * 30, 150, 0, 0⟩
  let wall2 : WallLayout := ⟨14, 0, 5 * 6, 0⟩
  let wall3 : WallLayout := ⟨0, 150, 0, 40⟩
  totalTiles wall1 wall2 wall3 = 744 := by
  sorry

end shower_tiles_count_l1369_136958


namespace simplify_and_rationalize_l1369_136927

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l1369_136927


namespace inequality_solution_set_l1369_136934

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, 12 * x^2 - a * x > a^2) ↔
    (a > 0 ∧ (x < -a/4 ∨ x > a/3)) ∨
    (a = 0 ∧ x ≠ 0) ∨
    (a < 0 ∧ (x < a/3 ∨ x > -a/4)) :=
by sorry

end inequality_solution_set_l1369_136934


namespace symmetry_sum_l1369_136964

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 1) (2, b) → a + b = 1 := by
  sorry

end symmetry_sum_l1369_136964


namespace quadratic_form_sum_l1369_136983

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by
sorry

end quadratic_form_sum_l1369_136983


namespace fraction_decomposition_l1369_136980

theorem fraction_decomposition (x A B C : ℚ) : 
  (6*x^2 - 13*x + 6) / (2*x^3 + 3*x^2 - 11*x - 6) = 
  A / (x + 1) + B / (2*x - 3) + C / (x - 2) →
  A = 1 ∧ B = 4 ∧ C = 1 := by
sorry

end fraction_decomposition_l1369_136980


namespace partial_fraction_decomposition_l1369_136966

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 17/7 ∧ D = 11/7 ∧
  ∀ (x : ℚ), x ≠ 5 ∧ x ≠ -2 →
    (4*x - 3) / (x^2 - 3*x - 10) = C / (x - 5) + D / (x + 2) :=
by sorry

end partial_fraction_decomposition_l1369_136966


namespace min_value_of_z_l1369_136911

theorem min_value_of_z (x y : ℝ) :
  2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 35 ≥ 24 := by
  sorry

end min_value_of_z_l1369_136911


namespace train_wheel_rows_l1369_136925

/-- Proves that the number of rows of wheels per carriage is 3, given the conditions of the train station. -/
theorem train_wheel_rows (num_trains : ℕ) (carriages_per_train : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 →
  carriages_per_train = 4 →
  wheels_per_row = 5 →
  total_wheels = 240 →
  (num_trains * carriages_per_train * wheels_per_row * (total_wheels / (num_trains * carriages_per_train * wheels_per_row))) = total_wheels →
  (total_wheels / (num_trains * carriages_per_train * wheels_per_row)) = 3 :=
by
  sorry

end train_wheel_rows_l1369_136925


namespace sunday_game_revenue_proof_l1369_136908

def sunday_game_revenue (total_revenue : ℚ) (revenue_difference : ℚ) : ℚ :=
  (total_revenue + revenue_difference) / 2

theorem sunday_game_revenue_proof (total_revenue revenue_difference : ℚ) 
  (h1 : total_revenue = 4994.50)
  (h2 : revenue_difference = 1330.50) :
  sunday_game_revenue total_revenue revenue_difference = 3162.50 := by
  sorry

end sunday_game_revenue_proof_l1369_136908


namespace ratio_expression_l1369_136955

theorem ratio_expression (a b : ℚ) (h : a / b = 4 / 1) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
  sorry

end ratio_expression_l1369_136955


namespace evaporation_problem_l1369_136967

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem evaporation_problem (y : Solution) 
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.total = 8)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_y : ℝ)
  (h5 : added_y = 4)
  (new_liquid_x_percent : ℝ)
  (h6 : new_liquid_x_percent = 0.45) :
  y.total * y.liquid_x_percent + (y.total * y.water_percent - evaporated_water) = 4 := by
  sorry

#check evaporation_problem

end evaporation_problem_l1369_136967


namespace bodyguard_hourly_rate_l1369_136994

/-- Proves that the hourly rate for each bodyguard is $20 -/
theorem bodyguard_hourly_rate :
  let num_bodyguards : ℕ := 2
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 7
  let total_weekly_payment : ℕ := 2240
  (num_bodyguards * hours_per_day * days_per_week * hourly_rate = total_weekly_payment) →
  hourly_rate = 20 := by
  sorry

end bodyguard_hourly_rate_l1369_136994


namespace stair_climbing_problem_l1369_136905

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (num_flights : ℕ) (height_per_flight : ℚ) (step_height_inches : ℚ) : ℚ :=
  (num_flights * height_per_flight) / (step_height_inches / 12)

/-- Proves that climbing 9 flights of 10 feet each, with steps of 18 inches, results in 60 steps. -/
theorem stair_climbing_problem :
  steps_climbed 9 10 18 = 60 := by
  sorry

end stair_climbing_problem_l1369_136905


namespace central_angles_sum_l1369_136973

theorem central_angles_sum (x : ℝ) : 
  (6 * x + (7 * x + 10) + (2 * x + 10) + x = 360) → x = 21.25 := by
  sorry

end central_angles_sum_l1369_136973


namespace tourist_journey_days_l1369_136909

def tourist_journey (first_section second_section : ℝ) (speed_difference : ℝ) : Prop :=
  ∃ (x : ℝ),
    -- x is the number of days for the second section
    -- First section takes (x/2 + 1) days
    (x/2 + 1) * (second_section/x - speed_difference) = first_section ∧
    x * (second_section/x) = second_section ∧
    -- Total journey takes 4 days
    x + (x/2 + 1) = 4

theorem tourist_journey_days :
  tourist_journey 246 276 15 :=
sorry

end tourist_journey_days_l1369_136909


namespace contradiction_assumption_for_at_most_two_even_l1369_136940

theorem contradiction_assumption_for_at_most_two_even 
  (a b c : ℕ) : 
  (¬ (∃ (x y : ℕ), {a, b, c} \ {x, y} ⊆ {n : ℕ | Even n})) ↔ 
  (Even a ∧ Even b ∧ Even c) :=
sorry

end contradiction_assumption_for_at_most_two_even_l1369_136940


namespace numeria_base_l1369_136974

theorem numeria_base (s : ℕ) : s > 1 →
  (s^3 - 8*s^2 - 9*s + 1 = 0) →
  (2*s*(s - 4) = 0) →
  s = 4 := by
sorry

end numeria_base_l1369_136974


namespace birthday_stickers_calculation_l1369_136963

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers : ℝ := sorry

/-- Mika's initial number of stickers -/
def initial_stickers : ℝ := 20.0

/-- Number of stickers Mika bought -/
def bought_stickers : ℝ := 26.0

/-- Number of stickers Mika received from her sister -/
def sister_stickers : ℝ := 6.0

/-- Number of stickers Mika received from her mother -/
def mother_stickers : ℝ := 58.0

/-- Mika's final total number of stickers -/
def final_stickers : ℝ := 130.0

theorem birthday_stickers_calculation :
  birthday_stickers = final_stickers - (initial_stickers + bought_stickers + sister_stickers + mother_stickers) :=
by sorry

end birthday_stickers_calculation_l1369_136963


namespace rectangle_perimeter_l1369_136932

/-- Given a rectangle divided into three congruent squares, where each square has a perimeter of 24 inches, prove that the perimeter of the original rectangle is 48 inches. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 24) : 
  let square_side := square_perimeter / 4
  let rectangle_length := 3 * square_side
  let rectangle_width := square_side
  2 * (rectangle_length + rectangle_width) = 48 :=
by sorry

end rectangle_perimeter_l1369_136932


namespace complement_intersection_theorem_l1369_136953

def U : Set Nat := {0, 1, 2, 4, 8}
def A : Set Nat := {1, 2, 8}
def B : Set Nat := {2, 4, 8}

theorem complement_intersection_theorem : 
  (U \ (A ∩ B)) = {0, 1, 4} := by sorry

end complement_intersection_theorem_l1369_136953


namespace white_balls_count_l1369_136941

/-- The number of white balls in a bag with specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 100 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  red = 17 ∧ 
  purple = 3 ∧ 
  prob_not_red_purple = 4/5 →
  ∃ white : ℕ, white = 50 ∧ total = white + green + yellow + red + purple :=
by
  sorry

end white_balls_count_l1369_136941


namespace shaded_area_square_with_quarter_circles_l1369_136988

/-- The area of the shaded region inside a square with quarter circles at its corners -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = square_side / 3) :
  square_side ^ 2 - π * circle_radius ^ 2 = 225 - 25 * π := by
sorry

end shaded_area_square_with_quarter_circles_l1369_136988


namespace k_range_proof_l1369_136981

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

-- Define the range of k
def k_range (k : ℝ) : Prop := k > 2

-- State the theorem
theorem k_range_proof :
  (∀ k, (∀ x, p x k ↔ q x) → k_range k) ∧
  (∀ k, k_range k → (∀ x, p x k ↔ q x)) :=
sorry

end k_range_proof_l1369_136981


namespace smallest_b_value_l1369_136995

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 8)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, x.val < b.val → 
    (∃ y : ℕ+, y.val - x.val = 8 ∧ 
      Nat.gcd ((y.val^3 + x.val^3) / (y.val + x.val)) (y.val * x.val) ≠ 16) ∧
    b.val = 3 :=
sorry

end smallest_b_value_l1369_136995


namespace segment_inequalities_l1369_136977

/-- Given a line segment AD with points B and C, prove inequalities about their lengths -/
theorem segment_inequalities 
  (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  a < c/2 ∧ b < a + c/2 := by
  sorry

end segment_inequalities_l1369_136977


namespace g_of_5_l1369_136972

def g (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 28*x^2 - 20*x - 80

theorem g_of_5 : g 5 = -5 := by
  sorry

end g_of_5_l1369_136972


namespace lcm_gcf_product_24_36_l1369_136946

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end lcm_gcf_product_24_36_l1369_136946


namespace winning_candidate_percentage_l1369_136907

/-- Theorem: In an election with 3 candidates receiving 2500, 5000, and 15000 votes respectively,
    the winning candidate received 75% of the total votes. -/
theorem winning_candidate_percentage (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 2500)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 75 := by
  sorry


end winning_candidate_percentage_l1369_136907


namespace initial_children_on_bus_proof_initial_children_l1369_136990

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    initial_children + 7 = 25

theorem proof_initial_children : 
  ∃ initial_children : ℕ, initial_children_on_bus initial_children ∧ initial_children = 18 := by
  sorry

end initial_children_on_bus_proof_initial_children_l1369_136990


namespace fourth_root_of_207360000_l1369_136914

theorem fourth_root_of_207360000 : Real.sqrt (Real.sqrt 207360000) = 120 := by sorry

end fourth_root_of_207360000_l1369_136914


namespace equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l1369_136945

/-- A sequence is an equal variance sequence if the difference of squares of consecutive terms is constant. -/
def EqualVarianceSequence (a : ℕ+ → ℝ) :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

/-- The square of an equal variance sequence is an arithmetic sequence. -/
theorem equal_variance_square_arithmetic (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) :
  ∃ d : ℝ, ∀ n : ℕ+, (a (n + 1))^2 - (a n)^2 = d := by sorry

/-- The sequence (-1)^n is an equal variance sequence. -/
theorem neg_one_power_equal_variance :
  EqualVarianceSequence (fun n => (-1 : ℝ) ^ (n : ℕ)) := by sorry

/-- If a_n is an equal variance sequence, then a_{kn} is also an equal variance sequence for any positive integer k. -/
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (h : EqualVarianceSequence a) (k : ℕ+) :
  EqualVarianceSequence (fun n => a (k * n)) := by sorry

end equal_variance_square_arithmetic_neg_one_power_equal_variance_equal_variance_subsequence_l1369_136945


namespace polynomial_non_real_root_l1369_136965

theorem polynomial_non_real_root (q : ℝ) : 
  ∃ (z : ℂ), z.im ≠ 0 ∧ z^4 - 2*q*z^3 - z^2 - 2*q*z + 1 = 0 := by sorry

end polynomial_non_real_root_l1369_136965


namespace jeans_price_increase_l1369_136975

theorem jeans_price_increase (manufacturing_cost : ℝ) : 
  let retailer_price := manufacturing_cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.82 := by
sorry

end jeans_price_increase_l1369_136975


namespace convexity_condition_l1369_136902

/-- A plane curve C defined by r = a - b cos θ, where a and b are positive reals and a > b -/
structure PlaneCurve where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The curve C is convex -/
def is_convex (C : PlaneCurve) : Prop := sorry

/-- Main theorem: C is convex if and only if b/a ≤ 1/2 -/
theorem convexity_condition (C : PlaneCurve) : 
  is_convex C ↔ C.b / C.a ≤ 1/2 := by sorry

end convexity_condition_l1369_136902


namespace right_triangle_area_l1369_136970

/-- 
Given a right-angled triangle with perpendicular sides a and b,
prove that its area is 1/2 when a + b = 4 and a² + b² = 14
-/
theorem right_triangle_area (a b : ℝ) 
  (sum_sides : a + b = 4) 
  (sum_squares : a^2 + b^2 = 14) : 
  (1/2) * a * b = 1/2 := by
  sorry

end right_triangle_area_l1369_136970


namespace circle_parabola_intersection_l1369_136986

/-- Circle with center (0,1) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}

/-- Parabola defined by y = ax² -/
def P (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1^2}

/-- Theorem stating the condition for C and P to intersect at points other than (0,0) -/
theorem circle_parabola_intersection (a : ℝ) :
  (∃ p : ℝ × ℝ, p ∈ C ∩ P a ∧ p ≠ (0, 0)) ↔ a > 1/2 := by sorry

end circle_parabola_intersection_l1369_136986


namespace hyperbola_derivative_l1369_136924

variable (a b x y : ℝ)
variable (h : x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_derivative :
  ∃ (dy_dx : ℝ), dy_dx = (b^2 * x) / (a^2 * y) := by sorry

end hyperbola_derivative_l1369_136924


namespace negative_128_squared_div_64_l1369_136996

theorem negative_128_squared_div_64 : ((-128)^2) / 64 = 256 := by sorry

end negative_128_squared_div_64_l1369_136996


namespace discount_percentage_l1369_136956

theorem discount_percentage (d : ℝ) (h : d > 0) : 
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 100 ∧ 
  (1 - x / 100) * 0.9 * d = 0.765 * d ∧ 
  x = 15 := by
sorry

end discount_percentage_l1369_136956


namespace profit_maximum_at_five_l1369_136992

/-- Profit function parameters -/
def a : ℝ := -10
def b : ℝ := 100
def c : ℝ := 2000

/-- Profit function -/
def profit_function (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The point where the maximum profit occurs -/
def max_profit_point : ℝ := 5

theorem profit_maximum_at_five :
  ∀ x : ℝ, profit_function x ≤ profit_function max_profit_point :=
by sorry


end profit_maximum_at_five_l1369_136992


namespace two_a_minus_b_equals_two_l1369_136969

theorem two_a_minus_b_equals_two (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : -b^3 + 12*b^2 - 47*b + 180 = 0) : 
  2*a - b = 2 := by
sorry

end two_a_minus_b_equals_two_l1369_136969


namespace jim_skips_proof_l1369_136993

/-- The number of times Bob can skip a rock. -/
def bob_skips : ℕ := 12

/-- The number of rocks Bob and Jim each skipped. -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim. -/
def total_skips : ℕ := 270

/-- The number of times Jim can skip a rock. -/
def jim_skips : ℕ := 15

theorem jim_skips_proof : 
  bob_skips * rocks_skipped + jim_skips * rocks_skipped = total_skips :=
by sorry

end jim_skips_proof_l1369_136993


namespace mixture_solution_l1369_136948

/-- Represents the mixture composition and constraints -/
structure Mixture where
  d : ℝ  -- diesel amount
  p : ℝ  -- petrol amount
  w : ℝ  -- water amount
  e : ℝ  -- ethanol amount
  total_volume : ℝ  -- total volume of the mixture

/-- The mixture satisfies the given constraints -/
def satisfies_constraints (m : Mixture) : Prop :=
  m.d = 4 ∧ 
  m.p = 4 ∧ 
  m.d / m.total_volume = 0.2 ∧
  m.p / m.total_volume = 0.15 ∧
  m.e / m.total_volume = 0.25 ∧
  m.w / m.total_volume = 0.4 ∧
  m.total_volume ≤ 30 ∧
  m.total_volume = m.d + m.p + m.w + m.e

/-- The theorem to be proved -/
theorem mixture_solution :
  ∃ (m : Mixture), satisfies_constraints m ∧ m.w = 8 ∧ m.e = 5 ∧ m.total_volume = 20 := by
  sorry


end mixture_solution_l1369_136948


namespace point_positions_l1369_136920

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation x - y + m = 0 -/
def line_equation (p : Point) (m : ℝ) : ℝ := p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def opposite_sides (a b : Point) (m : ℝ) : Prop :=
  line_equation a m * line_equation b m < 0

theorem point_positions (m : ℝ) : 
  let a : Point := ⟨2, 1⟩
  let b : Point := ⟨1, 3⟩
  opposite_sides a b m ↔ -1 < m ∧ m < 2 := by
  sorry

end point_positions_l1369_136920


namespace red_balls_in_stratified_sample_l1369_136999

/-- Calculates the number of red balls to be sampled in a stratified sampling by color -/
def stratifiedSampleRedBalls (totalPopulation : ℕ) (totalRedBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalRedBalls * sampleSize) / totalPopulation

/-- Theorem: The number of red balls in a stratified sample of 100 from 1000 balls with 50 red balls is 5 -/
theorem red_balls_in_stratified_sample :
  stratifiedSampleRedBalls 1000 50 100 = 5 := by
  sorry

end red_balls_in_stratified_sample_l1369_136999


namespace positive_difference_of_solutions_l1369_136922

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x + 20 = x + 51

-- Define the solutions of the quadratic equation
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

-- State the theorem
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 4 * Real.sqrt 10 :=
sorry

end positive_difference_of_solutions_l1369_136922


namespace high_school_enrollment_l1369_136951

/-- The number of students in a high school with given enrollment in music and art classes -/
def total_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (music - both) + (art - both) + both + neither

/-- Theorem stating that the total number of students is 500 given the specific enrollment numbers -/
theorem high_school_enrollment : total_students 30 20 10 460 = 500 := by
  sorry

end high_school_enrollment_l1369_136951


namespace power_sum_equality_l1369_136961

theorem power_sum_equality (x y : ℕ+) :
  x^(y:ℕ) + y^(x:ℕ) = x^(x:ℕ) + y^(y:ℕ) ↔ x = y :=
sorry

end power_sum_equality_l1369_136961


namespace max_value_of_f_l1369_136947

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/16 ∧ ∀ (t : ℝ), f t ≤ M ∧ ∃ (t₀ : ℝ), f t₀ = M :=
sorry

end max_value_of_f_l1369_136947


namespace birthday_stickers_l1369_136949

theorem birthday_stickers (initial : ℕ) (given_away : ℕ) (final : ℕ) : 
  initial = 269 → given_away = 48 → final = 423 → 
  final - given_away - initial = 202 :=
by sorry

end birthday_stickers_l1369_136949


namespace problem_solution_l1369_136971

open Real

theorem problem_solution :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ 9/2) ∧
    (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1/a₀ + 4/b₀ = 9/2) ∧
    (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 ∧ b' > 0 → 1/a' + 4/b' ≥ abs (2*x - 1) - abs (x + 1)) →
      -5/2 ≤ x ∧ x ≤ 13/2) :=
by
  sorry


end problem_solution_l1369_136971


namespace nail_hammering_l1369_136936

theorem nail_hammering (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4 : ℝ) / 7 + 4 / 7 * k + 4 / 7 * k^2 = 1 := by
  sorry

end nail_hammering_l1369_136936


namespace square_sum_over_28_squared_equals_8_l1369_136959

theorem square_sum_over_28_squared_equals_8 :
  ∃ x : ℝ, (x^2 + x^2) / 28^2 = 8 ∧ x = 56 := by
  sorry

end square_sum_over_28_squared_equals_8_l1369_136959


namespace no_integer_solution_l1369_136939

theorem no_integer_solution : ¬∃ (x y : ℤ), Real.sqrt ((x^2 : ℝ) + x + 1) + Real.sqrt ((y^2 : ℝ) - y + 1) = 11 := by
  sorry

end no_integer_solution_l1369_136939


namespace students_in_section_B_l1369_136982

/-- Proves the number of students in section B given the class information -/
theorem students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_A = 24)
  (h2 : avg_weight_A = 40)
  (h3 : avg_weight_B = 35)
  (h4 : avg_weight_total = 38) : 
  ∃ (students_B : ℕ), 
    (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total ∧ 
    students_B = 16 := by
  sorry

end students_in_section_B_l1369_136982


namespace opposite_numbers_l1369_136998

theorem opposite_numbers : ∀ x : ℚ, |x| = -x → x = -2/3 := by
  sorry

end opposite_numbers_l1369_136998


namespace M_dense_in_itself_l1369_136957

/-- The set M of real numbers of the form (m+n)/√(m²+n²), where m and n are positive integers. -/
def M : Set ℝ :=
  {x : ℝ | ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ x = (m + n : ℝ) / Real.sqrt ((m^2 + n^2 : ℕ))}

/-- M is dense in itself -/
theorem M_dense_in_itself : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y := by
  sorry

end M_dense_in_itself_l1369_136957


namespace functional_equation_solution_l1369_136910

-- Define the property that the function must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end functional_equation_solution_l1369_136910


namespace roses_equation_initial_roses_count_l1369_136943

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 22

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by sorry

end roses_equation_initial_roses_count_l1369_136943


namespace necessary_not_sufficient_l1369_136916

theorem necessary_not_sufficient (a b : ℝ) : 
  (∃ b, a ≠ 0 ∧ a * b = 0) ∧ (a * b ≠ 0 → a ≠ 0) := by
  sorry

end necessary_not_sufficient_l1369_136916


namespace football_throw_percentage_l1369_136944

theorem football_throw_percentage (parker_throw grant_throw kyle_throw : ℝ) :
  parker_throw = 16 →
  kyle_throw = 2 * grant_throw →
  kyle_throw = parker_throw + 24 →
  (grant_throw - parker_throw) / parker_throw = 0.25 := by
  sorry

end football_throw_percentage_l1369_136944


namespace liar_count_l1369_136919

/-- Represents a district in the town -/
inductive District
| A
| B
| Γ
| Δ

/-- Structure representing the town -/
structure Town where
  knights : Nat
  liars : Nat
  affirmativeAnswers : District → Nat

/-- The conditions of the problem -/
def townConditions (t : Town) : Prop :=
  t.affirmativeAnswers District.A +
  t.affirmativeAnswers District.B +
  t.affirmativeAnswers District.Γ +
  t.affirmativeAnswers District.Δ = 500 ∧
  t.knights * 4 = 200 ∧
  t.affirmativeAnswers District.A = t.knights + 95 ∧
  t.affirmativeAnswers District.B = t.knights + 115 ∧
  t.affirmativeAnswers District.Γ = t.knights + 157 ∧
  t.affirmativeAnswers District.Δ = t.knights + 133 ∧
  t.liars * 3 + t.knights = 500

theorem liar_count (t : Town) (h : townConditions t) : t.liars = 100 := by
  sorry

end liar_count_l1369_136919


namespace correct_rounded_sum_l1369_136929

def round_to_nearest_ten (n : Int) : Int :=
  10 * ((n + 5) / 10)

theorem correct_rounded_sum : round_to_nearest_ten (68 + 57) = 130 := by
  sorry

end correct_rounded_sum_l1369_136929


namespace penny_whale_species_l1369_136913

theorem penny_whale_species (shark_species eel_species total_species : ℕ) 
  (h1 : shark_species = 35)
  (h2 : eel_species = 15)
  (h3 : total_species = 55) :
  total_species - (shark_species + eel_species) = 5 := by
sorry

end penny_whale_species_l1369_136913


namespace transformation_C_not_equivalent_l1369_136903

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x + y = 5
def equation2 (x y : ℝ) : Prop := 3 * x + 4 * y = 7

-- Define the incorrect transformation
def transformation_C (x y : ℝ) : Prop := x = (7 + 4 * y) / 3

-- Theorem stating that the transformation is not equivalent to equation2
theorem transformation_C_not_equivalent :
  ∃ x y : ℝ, equation2 x y ∧ ¬(transformation_C x y) :=
sorry

end transformation_C_not_equivalent_l1369_136903


namespace function_inequality_l1369_136926

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f x = f (x + period)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 6)
  (h2 : monotone_decreasing_on f 0 3)
  (h3 : symmetric_about f 3) :
  f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 := by
  sorry

end function_inequality_l1369_136926


namespace tetrahedron_side_sum_squares_l1369_136997

/-- A tetrahedron with side lengths a, b, c and circumradius 1 -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the side lengths of the tetrahedron is 8 -/
theorem tetrahedron_side_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry


end tetrahedron_side_sum_squares_l1369_136997


namespace ellipse_min_sum_l1369_136989

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 4/n = 1) :
  m + n ≥ 9 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 4/n₀ = 1 ∧ m₀ + n₀ = 9 := by
  sorry

end ellipse_min_sum_l1369_136989


namespace tan_eq_two_solution_set_l1369_136987

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} := by
sorry

end tan_eq_two_solution_set_l1369_136987


namespace range_of_a_l1369_136917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 2 - 3 * a else 2^x - 1

theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a < 2/3 :=
sorry

end range_of_a_l1369_136917
