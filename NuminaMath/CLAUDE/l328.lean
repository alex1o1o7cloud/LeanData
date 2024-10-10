import Mathlib

namespace unique_intersection_implies_line_equation_l328_32861

/-- Given a line y = mx + b passing through (2, 3), prove that if there exists exactly one k
    where x = k intersects y = x^2 - 4x + 4 and y = mx + b at points 6 units apart,
    then m = -6 and b = 15 -/
theorem unique_intersection_implies_line_equation 
  (m b : ℝ) 
  (passes_through : 3 = 2 * m + b) 
  (h : ∃! k : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = k^2 - 4*k + 4 ∧ 
    y₂ = m*k + b ∧ 
    (y₁ - y₂)^2 = 36) : 
  m = -6 ∧ b = 15 := by
sorry

end unique_intersection_implies_line_equation_l328_32861


namespace geometric_sequence_condition_l328_32827

/-- For a geometric sequence with common ratio q, the condition a_5 * a_6 < a_4^2 is necessary but not sufficient for 0 < q < 1 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence definition
  (a 5 * a 6 < a 4 ^ 2) →       -- given condition
  (∃ q', 0 < q' ∧ q' < 1 ∧ ¬(a 5 * a 6 < a 4 ^ 2 → 0 < q' ∧ q' < 1)) ∧
  (0 < q ∧ q < 1 → a 5 * a 6 < a 4 ^ 2) :=
by sorry

end geometric_sequence_condition_l328_32827


namespace teal_color_survey_l328_32887

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 120)
  (h_green : green = 70)
  (h_both : both = 35)
  (h_neither : neither = 20) :
  ∃ blue : ℕ, blue = 65 ∧ 
    blue + green - both + neither = total :=
by sorry

end teal_color_survey_l328_32887


namespace b_200_equals_179101_l328_32883

/-- Sequence a_n defined as n(n+1)/2 -/
def a (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is not divisible by 3 -/
def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

/-- Sequence b_n derived from a_n by removing terms divisible by 3 and rearranging -/
def b (n : ℕ) : ℕ := a (3 * n - 2)

/-- Theorem stating that the 200th term of sequence b_n is 179101 -/
theorem b_200_equals_179101 : b 200 = 179101 := by sorry

end b_200_equals_179101_l328_32883


namespace x_plus_y_equals_ten_l328_32859

theorem x_plus_y_equals_ten (x y : ℝ) 
  (hx : x + Real.log x / Real.log 10 = 10) 
  (hy : y + 10^y = 10) : 
  x + y = 10 := by
sorry

end x_plus_y_equals_ten_l328_32859


namespace theresa_work_hours_l328_32855

/-- The average number of hours Theresa needs to work per week -/
def required_average : ℝ := 9

/-- The number of weeks Theresa needs to maintain the average -/
def total_weeks : ℕ := 7

/-- The hours Theresa worked in the first 6 weeks -/
def first_six_weeks : List ℝ := [10, 8, 9, 11, 6, 8]

/-- The sum of hours Theresa worked in the first 6 weeks -/
def sum_first_six : ℝ := first_six_weeks.sum

/-- The number of hours Theresa needs to work in the seventh week -/
def hours_seventh_week : ℝ := 11

theorem theresa_work_hours :
  (sum_first_six + hours_seventh_week) / total_weeks = required_average := by
  sorry

end theresa_work_hours_l328_32855


namespace star_problem_l328_32837

-- Define the ⭐ operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_problem : star (star 3 9) 4 = 7 / 4 := by sorry

end star_problem_l328_32837


namespace digit_product_equation_l328_32893

def digit_product (k : ℕ) : ℕ :=
  if k = 0 then 0
  else if k < 10 then k
  else (k % 10) * digit_product (k / 10)

theorem digit_product_equation : 
  ∀ k : ℕ, k > 0 → (digit_product k = (25 * k) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
sorry

end digit_product_equation_l328_32893


namespace product_of_fifth_and_eighth_l328_32869

/-- A geometric progression with terms a_n -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n : ℕ, a n = a₁ * r^(n - 1)

/-- The 3rd and 10th terms are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  (a 3)^2 - 3*(a 3) - 5 = 0 ∧ (a 10)^2 - 3*(a 10) - 5 = 0

theorem product_of_fifth_and_eighth (a : ℕ → ℝ) 
  (h1 : geometric_progression a) (h2 : roots_condition a) : 
  a 5 * a 8 = -5 := by
  sorry

end product_of_fifth_and_eighth_l328_32869


namespace exists_thirteen_cubes_l328_32882

/-- Represents a 4x4 board with cube stacks -/
def Board := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if the board configuration is valid -/
def valid_board (b : Board) : Prop :=
  ∀ n : Fin 8, ∃! (i j k l : Fin 4), 
    b i j = n + 1 ∧ b k l = n + 1 ∧ (i ≠ k ∨ j ≠ l)

/-- Theorem stating that there exists a pair of cells with 13 cubes total -/
theorem exists_thirteen_cubes (b : Board) (h : valid_board b) : 
  ∃ (i j k l : Fin 4), b i j + b k l = 13 :=
sorry

end exists_thirteen_cubes_l328_32882


namespace larger_number_proof_l328_32841

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 830) (h3 : x = 22 * y + 2) : x = 794 := by
  sorry

end larger_number_proof_l328_32841


namespace cos_two_alpha_l328_32848

theorem cos_two_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 4) = 1 / 3) : 
  Real.cos (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end cos_two_alpha_l328_32848


namespace bottle_caps_difference_l328_32843

-- Define the number of bottle caps found and thrown away each day
def monday_found : ℕ := 36
def monday_thrown : ℕ := 45
def tuesday_found : ℕ := 58
def tuesday_thrown : ℕ := 30
def wednesday_found : ℕ := 80
def wednesday_thrown : ℕ := 70

-- Define the final number of bottle caps left
def final_caps : ℕ := 65

-- Theorem to prove
theorem bottle_caps_difference :
  (monday_found + tuesday_found + wednesday_found) -
  (monday_thrown + tuesday_thrown + wednesday_thrown) = 29 :=
by sorry

end bottle_caps_difference_l328_32843


namespace cos_squared_minus_sin_squared_l328_32885

theorem cos_squared_minus_sin_squared (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ y / x = Real.tan α) →
  Real.cos α ^ 2 - Real.sin α ^ 2 = -3/5 := by
  sorry

end cos_squared_minus_sin_squared_l328_32885


namespace percentage_problem_l328_32895

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end percentage_problem_l328_32895


namespace sandbox_width_l328_32889

/-- A rectangular sandbox with perimeter 30 feet and length twice the width has a width of 5 feet. -/
theorem sandbox_width :
  ∀ (width length : ℝ),
  width > 0 →
  length > 0 →
  length = 2 * width →
  2 * length + 2 * width = 30 →
  width = 5 := by
sorry

end sandbox_width_l328_32889


namespace total_games_is_506_l328_32834

/-- Represents the structure of a soccer league --/
structure SoccerLeague where
  total_teams : Nat
  divisions : Nat
  teams_per_division : Nat
  regular_season_intra_division_matches : Nat
  regular_season_cross_division_matches : Nat
  mid_season_tournament_matches_per_team : Nat
  mid_season_tournament_intra_division_matches : Nat
  playoff_teams_per_division : Nat
  playoff_stages : Nat

/-- Calculates the total number of games in the soccer league season --/
def total_games (league : SoccerLeague) : Nat :=
  -- Regular season games
  (league.teams_per_division * (league.teams_per_division - 1) * league.regular_season_intra_division_matches * league.divisions) / 2 +
  (league.total_teams * league.regular_season_cross_division_matches) / 2 +
  -- Mid-season tournament games
  (league.total_teams * league.mid_season_tournament_matches_per_team) / 2 +
  -- Playoff games
  (league.playoff_teams_per_division * 2 - 1) * 2 * league.divisions

/-- Theorem stating that the total number of games in the given league structure is 506 --/
theorem total_games_is_506 (league : SoccerLeague) 
  (h1 : league.total_teams = 24)
  (h2 : league.divisions = 2)
  (h3 : league.teams_per_division = 12)
  (h4 : league.regular_season_intra_division_matches = 2)
  (h5 : league.regular_season_cross_division_matches = 4)
  (h6 : league.mid_season_tournament_matches_per_team = 5)
  (h7 : league.mid_season_tournament_intra_division_matches = 3)
  (h8 : league.playoff_teams_per_division = 4)
  (h9 : league.playoff_stages = 3) :
  total_games league = 506 := by
  sorry

end total_games_is_506_l328_32834


namespace sqrt_x_minus_two_real_l328_32810

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_two_real_l328_32810


namespace cat_weight_difference_l328_32802

/-- Given the weights of two cats belonging to Meg and Anne, prove the weight difference --/
theorem cat_weight_difference 
  (weight_meg : ℝ) 
  (weight_anne : ℝ) 
  (h1 : weight_meg / weight_anne = 13 / 21)
  (h2 : weight_meg = 20 + 0.5 * weight_anne) :
  weight_anne - weight_meg = 64 := by
  sorry

end cat_weight_difference_l328_32802


namespace weight_of_b_l328_32816

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43)
  (h2 : (a + b) / 2 = 48)
  (h3 : (b + c) / 2 = 42) :
  b = 51 := by
sorry

end weight_of_b_l328_32816


namespace least_subtrahend_for_divisibility_specific_case_l328_32858

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem specific_case : 
  ∃ (k : Nat), k < 47 ∧ (929 - k) % 47 = 0 ∧ ∀ (m : Nat), m < k → (929 - m) % 47 ≠ 0 ∧ k = 44 :=
by
  sorry

end least_subtrahend_for_divisibility_specific_case_l328_32858


namespace polynomial_expansion_equality_l328_32818

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end polynomial_expansion_equality_l328_32818


namespace diet_soda_bottles_l328_32815

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 17)
  (h2 : regular_soda_bottles = 9) :
  total_bottles - regular_soda_bottles = 8 := by
  sorry

#check diet_soda_bottles

end diet_soda_bottles_l328_32815


namespace circle_radius_l328_32892

theorem circle_radius (x y : ℝ) :
  (16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 76 = 0) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = 3/2 :=
by sorry

end circle_radius_l328_32892


namespace arithmetic_expression_equality_l328_32884

theorem arithmetic_expression_equality : 12 - 7 * (-32) + 16 / (-4) = 232 := by
  sorry

end arithmetic_expression_equality_l328_32884


namespace intersection_curve_length_theorem_l328_32808

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertex : Point3D
  edge_length : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The length of the curve formed by the intersection of a unit cube's surface
    and a sphere centered at one of the cube's vertices with radius 2√3/3 -/
def intersection_curve_length (c : Cube) (s : Sphere) : ℝ := sorry

/-- Main theorem statement -/
theorem intersection_curve_length_theorem (c : Cube) (s : Sphere) :
  c.edge_length = 1 ∧
  s.center = c.vertex ∧
  s.radius = 2 * Real.sqrt 3 / 3 →
  intersection_curve_length c s = 5 * Real.sqrt 3 * Real.pi / 6 := by
  sorry

end intersection_curve_length_theorem_l328_32808


namespace correct_system_of_equations_l328_32866

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of the purchase in yuan -/
def total_cost : ℝ := 445

/-- The number of basketballs purchased -/
def num_basketballs : ℕ := 3

/-- The number of soccer balls purchased -/
def num_soccer_balls : ℕ := 7

/-- The price difference between a basketball and a soccer ball in yuan -/
def price_difference : ℝ := 5

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem correct_system_of_equations : 
  (num_basketballs * basketball_price + num_soccer_balls * soccer_ball_price = total_cost) ∧ 
  (basketball_price = soccer_ball_price + price_difference) := by
  sorry

end correct_system_of_equations_l328_32866


namespace negative_slope_implies_negative_correlation_l328_32865

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Given a linear regression with negative slope, 
    the correlation coefficient is between -1 and 0 -/
theorem negative_slope_implies_negative_correlation 
  (reg : LinearRegression) 
  (x y : ℝ → ℝ) 
  (h_reg : ∀ t, y t = reg.a + reg.b * x t) 
  (h_neg : reg.b < 0) : 
  -1 < correlation_coefficient x y ∧ correlation_coefficient x y < 0 := by
  sorry

end negative_slope_implies_negative_correlation_l328_32865


namespace store_sale_revenue_l328_32828

/-- Calculates the amount left after a store's inventory sale --/
theorem store_sale_revenue (total_items : ℕ) (category_a_items : ℕ) (category_b_items : ℕ) (category_c_items : ℕ)
  (price_a : ℝ) (price_b : ℝ) (price_c : ℝ)
  (discount_a : ℝ) (discount_b : ℝ) (discount_c : ℝ)
  (sales_percent_a : ℝ) (sales_percent_b : ℝ) (sales_percent_c : ℝ)
  (return_rate : ℝ) (advertising_cost : ℝ) (creditors_amount : ℝ) :
  total_items = category_a_items + category_b_items + category_c_items →
  category_a_items = 1000 →
  category_b_items = 700 →
  category_c_items = 300 →
  price_a = 50 →
  price_b = 75 →
  price_c = 100 →
  discount_a = 0.8 →
  discount_b = 0.7 →
  discount_c = 0.6 →
  sales_percent_a = 0.85 →
  sales_percent_b = 0.75 →
  sales_percent_c = 0.9 →
  return_rate = 0.03 →
  advertising_cost = 2000 →
  creditors_amount = 15000 →
  ∃ (revenue : ℝ), revenue = 13172.50 ∧ 
    revenue = (category_a_items * sales_percent_a * price_a * (1 - discount_a) * (1 - return_rate) +
               category_b_items * sales_percent_b * price_b * (1 - discount_b) * (1 - return_rate) +
               category_c_items * sales_percent_c * price_c * (1 - discount_c) * (1 - return_rate)) -
              advertising_cost - creditors_amount :=
by
  sorry


end store_sale_revenue_l328_32828


namespace q_polynomial_form_l328_32833

/-- Given a function q(x) satisfying the equation 
    q(x) + (2x^6 + 4x^4 + 12x^2) = (10x^4 + 36x^3 + 37x^2 + 5),
    prove that q(x) = -2x^6 + 6x^4 + 36x^3 + 25x^2 + 5 -/
theorem q_polynomial_form (x : ℝ) (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2*x^6 + 4*x^4 + 12*x^2) = 10*x^4 + 36*x^3 + 37*x^2 + 5) :
  q x = -2*x^6 + 6*x^4 + 36*x^3 + 25*x^2 + 5 := by
  sorry

end q_polynomial_form_l328_32833


namespace dust_retention_proof_l328_32806

/-- The average annual dust retention of a locust leaf in milligrams. -/
def locust_dust_retention : ℝ := 22

/-- The average annual dust retention of a ginkgo leaf in milligrams. -/
def ginkgo_dust_retention : ℝ := 2 * locust_dust_retention - 4

theorem dust_retention_proof :
  11 * ginkgo_dust_retention = 20 * locust_dust_retention :=
by sorry

end dust_retention_proof_l328_32806


namespace parabola_axis_distance_l328_32825

/-- Given a parabola x^2 = ay, if the distance from the point (0,1) to its axis of symmetry is 2, then a = -12 or a = 4. -/
theorem parabola_axis_distance (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y → 
    (|y - 1 - (-a/4)| = 2 ↔ (a = -12 ∨ a = 4))) :=
by sorry

end parabola_axis_distance_l328_32825


namespace min_value_expression_l328_32874

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x - y^2 = 1 → x^2 + 2*y^2 + 4*x - 1 ≥ min :=
sorry

end min_value_expression_l328_32874


namespace other_number_is_25_l328_32803

theorem other_number_is_25 (x y : ℤ) : 
  (3 * x + 4 * y + 2 * x = 160) → 
  ((x = 12 ∧ y ≠ 12) ∨ (y = 12 ∧ x ≠ 12)) → 
  (x = 25 ∨ y = 25) := by
sorry

end other_number_is_25_l328_32803


namespace friend_jogging_time_l328_32894

/-- Proves that if a person completes a route in 3 hours, and another person travels at twice the speed of the first person, then the second person will complete the same route in 90 minutes. -/
theorem friend_jogging_time (my_time : ℝ) (friend_speed : ℝ) (my_speed : ℝ) :
  my_time = 3 →
  friend_speed = 2 * my_speed →
  friend_speed * (90 / 60) = my_speed * my_time :=
by
  sorry

#check friend_jogging_time

end friend_jogging_time_l328_32894


namespace quadratic_completion_l328_32886

theorem quadratic_completion (y : ℝ) : ∃ b : ℝ, y^2 + 14*y + 60 = (y + b)^2 + 11 := by
  sorry

end quadratic_completion_l328_32886


namespace dog_arrangement_count_l328_32849

theorem dog_arrangement_count : ∀ (n m k : ℕ),
  n = 15 →
  m = 3 →
  k = 12 →
  (Nat.choose k 3) * (Nat.choose (k - 3) 6) = 18480 :=
by
  sorry

end dog_arrangement_count_l328_32849


namespace sharona_bought_four_more_pencils_l328_32835

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 11

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 13

/-- The number of pencils Sharona bought -/
def sharona_pencils : ℕ := 17

/-- The amount Jamar paid in cents -/
def jamar_paid : ℕ := 143

/-- The amount Sharona paid in cents -/
def sharona_paid : ℕ := 187

theorem sharona_bought_four_more_pencils :
  pencil_price > 1 ∧
  pencil_price * jamar_pencils = jamar_paid ∧
  pencil_price * sharona_pencils = sharona_paid →
  sharona_pencils - jamar_pencils = 4 :=
by sorry

end sharona_bought_four_more_pencils_l328_32835


namespace initial_points_count_l328_32876

/-- Represents the number of points after performing the point-adding operation n times -/
def pointsAfterOperations (k : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => 2 * (pointsAfterOperations k n) - 1

/-- The theorem stating that if 101 points result after two operations, then 26 points were initially marked -/
theorem initial_points_count : 
  ∀ k : ℕ, pointsAfterOperations k 2 = 101 → k = 26 := by
  sorry

end initial_points_count_l328_32876


namespace max_sum_cubes_l328_32872

theorem max_sum_cubes (e f g h i : ℝ) (h1 : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  ∃ (M : ℝ), M = 5^(3/4) ∧ e^3 + f^3 + g^3 + h^3 + i^3 ≤ M ∧
  ∃ (e' f' g' h' i' : ℝ), e'^4 + f'^4 + g'^4 + h'^4 + i'^4 = 5 ∧
                          e'^3 + f'^3 + g'^3 + h'^3 + i'^3 = M :=
by sorry

end max_sum_cubes_l328_32872


namespace intersection_line_of_circles_l328_32809

-- Define the circles O₁ and O₂ in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = Real.sin θ
def circle_O₂ (ρ θ : ℝ) : Prop := ρ = Real.cos θ

-- Define the line in Cartesian coordinates
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), circle_O₁ ρ θ ∧ circle_O₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  intersection_line x y :=
sorry

end intersection_line_of_circles_l328_32809


namespace rectangular_solid_surface_area_l328_32871

/-- The surface area of a rectangular solid with edge lengths 2, 3, and 4 is 52 -/
theorem rectangular_solid_surface_area : 
  let a : ℝ := 4
  let b : ℝ := 3
  let h : ℝ := 2
  2 * (a * b + b * h + a * h) = 52 := by sorry

end rectangular_solid_surface_area_l328_32871


namespace find_b_l328_32868

-- Define the inverse relationship between a² and √b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the conditions
def condition1 : ℝ := 3
def condition2 : ℝ := 36

-- Define the target equation
def target_equation (a b : ℝ) : Prop := a * b = 54

-- Theorem statement
theorem find_b :
  ∀ a b : ℝ,
  inverse_relation a b →
  inverse_relation condition1 condition2 →
  target_equation a b →
  b = 18 * (4^(1/3)) :=
by
  sorry

end find_b_l328_32868


namespace bill_bouquet_profit_l328_32812

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) in dollars -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit in dollars -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bill_bouquet_profit :
  bouquets_to_buy = 125 := by sorry

end bill_bouquet_profit_l328_32812


namespace residual_plot_vertical_axis_l328_32898

/-- Represents a residual plot in regression analysis -/
structure ResidualPlot where
  verticalAxis : Set ℝ
  horizontalAxis : Set ℝ

/-- Definition of a residual in regression analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents residuals -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) : 
  plot.verticalAxis = Set.range (λ r : Residual => r) := by
  sorry

end residual_plot_vertical_axis_l328_32898


namespace function_range_l328_32805

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end function_range_l328_32805


namespace polynomial_factorization_l328_32862

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^7 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end polynomial_factorization_l328_32862


namespace cakes_served_today_l328_32804

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
  sorry

end cakes_served_today_l328_32804


namespace table_tennis_tournament_l328_32814

theorem table_tennis_tournament (n : ℕ) (x : ℕ) : 
  n > 3 → 
  Nat.choose (n - 3) 2 + 6 - x = 50 → 
  Nat.choose n 2 = 50 → 
  x = 1 := by
sorry

end table_tennis_tournament_l328_32814


namespace remainder_problem_l328_32856

theorem remainder_problem (R : ℕ) : 
  (29 = Nat.gcd (1255 - 8) (1490 - R)) →
  (1255 % 29 = 8) →
  (1490 % 29 = R) →
  R = 11 := by
sorry

end remainder_problem_l328_32856


namespace pasture_fence_posts_l328_32844

/-- Calculates the number of posts needed for a given length of fence -/
def posts_for_length (length : ℕ) (post_spacing : ℕ) : ℕ :=
  (length / post_spacing) + 1

/-- The pasture dimensions -/
def pasture_width : ℕ := 36
def pasture_length : ℕ := 75

/-- The spacing between posts -/
def post_spacing : ℕ := 15

/-- The total number of posts required for the pasture -/
def total_posts : ℕ :=
  posts_for_length pasture_width post_spacing +
  2 * (posts_for_length pasture_length post_spacing - 1)

theorem pasture_fence_posts :
  total_posts = 14 := by sorry

end pasture_fence_posts_l328_32844


namespace no_double_application_successor_function_l328_32860

theorem no_double_application_successor_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 := by
  sorry

end no_double_application_successor_function_l328_32860


namespace allie_billie_meeting_l328_32846

/-- The distance Allie skates before meeting Billie -/
def allie_distance (ab_distance : ℝ) (allie_speed billie_speed : ℝ) (allie_angle : ℝ) : ℝ :=
  let x := 160
  x

theorem allie_billie_meeting 
  (ab_distance : ℝ) 
  (allie_speed billie_speed : ℝ) 
  (allie_angle : ℝ) 
  (h1 : ab_distance = 100)
  (h2 : allie_speed = 8)
  (h3 : billie_speed = 7)
  (h4 : allie_angle = 60 * π / 180)
  (h5 : ∀ x, x > 0 → x ≠ 160 → 
    (x / allie_speed ≠ 
    Real.sqrt (x^2 + ab_distance^2 - 2 * x * ab_distance * Real.cos allie_angle) / billie_speed)) :
  allie_distance ab_distance allie_speed billie_speed allie_angle = 160 := by
  sorry

end allie_billie_meeting_l328_32846


namespace polynomial_never_equals_33_l328_32863

theorem polynomial_never_equals_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end polynomial_never_equals_33_l328_32863


namespace integral_cos_squared_sin_l328_32881

theorem integral_cos_squared_sin (x : Real) :
  deriv (fun x => -Real.cos x ^ 3 / 3) x = Real.cos x ^ 2 * Real.sin x := by
  sorry

end integral_cos_squared_sin_l328_32881


namespace prob_three_is_half_l328_32854

/-- The decimal representation of 7/11 -/
def decimal_rep : ℚ := 7 / 11

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [6, 3]

/-- The probability of selecting a specific digit from the repeating sequence -/
def prob_digit (d : ℕ) : ℚ :=
  (repeating_sequence.count d : ℚ) / repeating_sequence.length

theorem prob_three_is_half :
  prob_digit 3 = 1 / 2 := by sorry

end prob_three_is_half_l328_32854


namespace circle_equation_l328_32817

/-- Theorem: The equation of a circle with center (1, -1) and radius 2 is (x-1)^2 + (y+1)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -1) ∧ 
    radius = 2 ∧ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2)) ↔ 
  ((x - 1)^2 + (y + 1)^2 = 4) :=
by sorry

end circle_equation_l328_32817


namespace frog_climb_time_l328_32888

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ
  time_to_near_top : ℝ

/-- Calculates the total time for the frog to climb to the top of the well -/
def total_climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the total climb time is 20 minutes -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time_ratio = 1/3)
  (h5 : f.time_to_near_top = 17) :
  total_climb_time f = 20 := by
  sorry

end frog_climb_time_l328_32888


namespace actual_weight_calculation_l328_32823

/-- The dealer's percent -/
def dealer_percent : ℝ := 53.84615384615387

/-- The actual weight used per kg -/
def actual_weight : ℝ := 0.4615384615384613

/-- Theorem stating that the actual weight used per kg is correct given the dealer's percent -/
theorem actual_weight_calculation (ε : ℝ) (h : ε > 0) : 
  |actual_weight - (1 - dealer_percent / 100)| < ε :=
by sorry

end actual_weight_calculation_l328_32823


namespace nineteen_power_calculation_l328_32832

theorem nineteen_power_calculation : (19^11 / 19^8) * 19^3 = 47015881 := by
  sorry

end nineteen_power_calculation_l328_32832


namespace hyperbola_equation_l328_32826

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Given ellipse equation
  (x^2 / 144 + y^2 / 169 = 1) →
  -- Hyperbola passes through (0, 2)
  (∃ (a b : ℝ), y^2 / a^2 - x^2 / b^2 = 1 ∧ 2^2 / a^2 - 0^2 / b^2 = 1) →
  -- Hyperbola shares a common focus with the ellipse
  (∃ (c : ℝ), c^2 = 169 - 144 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  (y^2 / 4 - x^2 / 21 = 1) :=
by sorry

end hyperbola_equation_l328_32826


namespace both_companies_participate_both_will_participate_l328_32899

/-- Represents a company in country A --/
structure Company where
  expectedIncome : ℝ
  investmentCost : ℝ

/-- The market conditions for the new technology development --/
structure MarketConditions where
  V : ℝ  -- Income if developed alone
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  h1 : 0 < α
  h2 : α < 1

/-- Calculate the expected income for a company when both participate --/
def expectedIncomeBothParticipate (m : MarketConditions) : ℝ :=
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V

/-- Theorem: Condition for both companies to participate --/
theorem both_companies_participate (m : MarketConditions) :
  expectedIncomeBothParticipate m - m.IC ≥ 0 ↔
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V - m.IC ≥ 0 := by
  sorry

/-- Function to determine if a company will participate --/
def willParticipate (c : Company) (m : MarketConditions) : Prop :=
  c.expectedIncome - c.investmentCost ≥ 0

/-- Theorem: Both companies will participate if the condition is met --/
theorem both_will_participate (c1 c2 : Company) (m : MarketConditions)
  (h : expectedIncomeBothParticipate m - m.IC ≥ 0) :
  willParticipate c1 m ∧ willParticipate c2 m := by
  sorry

end both_companies_participate_both_will_participate_l328_32899


namespace derivative_of_f_l328_32853

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log 2

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.log 2 := by sorry

end derivative_of_f_l328_32853


namespace sample_grade12_is_40_l328_32821

/-- Represents the stratified sampling problem for a school with three grades. -/
structure School where
  total_students : ℕ
  grade10_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from grade 12 in a stratified sampling. -/
def sampleGrade12 (s : School) : ℕ :=
  s.sample_size - (s.grade10_students * s.sample_size / s.total_students + s.grade11_students * s.sample_size / s.total_students)

/-- Theorem stating that for the given school parameters, the number of students
    to be sampled from grade 12 is 40. -/
theorem sample_grade12_is_40 (s : School)
    (h1 : s.total_students = 2400)
    (h2 : s.grade10_students = 820)
    (h3 : s.grade11_students = 780)
    (h4 : s.sample_size = 120) :
    sampleGrade12 s = 40 := by
  sorry

end sample_grade12_is_40_l328_32821


namespace problem_statement_l328_32875

theorem problem_statement (t : ℝ) :
  let x := 3 - 1.5 * t
  let y := 3 * t + 4
  x = 6 → y = -2 := by
sorry

end problem_statement_l328_32875


namespace cricket_team_average_age_l328_32890

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (average_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 11 →
  average_age = 29 →
  wicket_keeper_age_diff = 3 →
  remaining_average_diff = 1 →
  ∃ (captain_age : ℝ),
    team_size * average_age = 
      (team_size - 2) * (average_age - remaining_average_diff) + 
      captain_age + 
      (average_age + wicket_keeper_age_diff) :=
by sorry

end cricket_team_average_age_l328_32890


namespace unique_six_digit_reverse_multiple_l328_32840

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d => acc * 10 + d) 0

theorem unique_six_digit_reverse_multiple : 
  ∃! n : ℕ, is_six_digit n ∧ n * 9 = reverse_digits n :=
by sorry

end unique_six_digit_reverse_multiple_l328_32840


namespace max_distinct_values_exists_four_valued_function_l328_32830

/-- A function that assigns a number to each vector in space -/
def VectorFunction (n : ℕ) := (Fin n → ℝ) → ℝ

/-- The property of the vector function as described in the problem -/
def HasMaxProperty (n : ℕ) (f : VectorFunction n) : Prop :=
  ∀ (u v : Fin n → ℝ) (α β : ℝ), 
    f (fun i => α * u i + β * v i) ≤ max (f u) (f v)

/-- The theorem stating that a function with the given property can take at most 4 distinct values -/
theorem max_distinct_values (n : ℕ) (f : VectorFunction n) 
    (h : HasMaxProperty n f) : 
    ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S ≤ 4 := by
  sorry

/-- The theorem stating that there exists a function taking exactly 4 distinct values -/
theorem exists_four_valued_function : 
    ∃ (f : VectorFunction 3), HasMaxProperty 3 f ∧ 
      ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S = 4 := by
  sorry

end max_distinct_values_exists_four_valued_function_l328_32830


namespace inequality_solution_set_l328_32820

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 5*x - 6 > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end inequality_solution_set_l328_32820


namespace pathway_width_l328_32811

theorem pathway_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ + 4 = 14 := by
  sorry

end pathway_width_l328_32811


namespace min_reciprocal_sum_l328_32839

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 := by sorry

end min_reciprocal_sum_l328_32839


namespace opposites_sum_l328_32851

theorem opposites_sum (a b : ℝ) (h : a + b = 0) : 2006*a + 2 + 2006*b = 2 := by
  sorry

end opposites_sum_l328_32851


namespace square_area_ratio_l328_32831

theorem square_area_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (perimeter_ratio : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l328_32831


namespace angle_between_vectors_l328_32879

/-- Given two vectors a and b in ℝ², where a = (3, 0) and a + 2b = (1, 2√3),
    prove that the angle between a and b is 120°. -/
theorem angle_between_vectors (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (3, 0)
  (a.1 + 2 * b.1, a.2 + 2 * b.2) = (1, 2 * Real.sqrt 3) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
  sorry

end angle_between_vectors_l328_32879


namespace probability_sum_multiple_of_three_l328_32877

def die_faces : ℕ := 6

def total_outcomes : ℕ := die_faces * die_faces

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def favorable_outcomes : ℕ := 12

theorem probability_sum_multiple_of_three :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end probability_sum_multiple_of_three_l328_32877


namespace opposite_of_five_l328_32829

theorem opposite_of_five : -(5 : ℤ) = -5 := by sorry

end opposite_of_five_l328_32829


namespace blocks_needed_per_color_l328_32847

/-- Represents the dimensions of a clay block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical pot -/
structure PotDimensions where
  height : ℝ
  diameter : ℝ

/-- Calculates the number of blocks needed for each color -/
def blocksPerColor (block : BlockDimensions) (pot : PotDimensions) (layerHeight : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 7 blocks of each color are needed -/
theorem blocks_needed_per_color 
  (block : BlockDimensions)
  (pot : PotDimensions)
  (layerHeight : ℝ)
  (h1 : block.length = 4)
  (h2 : block.width = 3)
  (h3 : block.height = 2)
  (h4 : pot.height = 10)
  (h5 : pot.diameter = 5)
  (h6 : layerHeight = 2.5) :
  blocksPerColor block pot layerHeight = 7 := by
  sorry

end blocks_needed_per_color_l328_32847


namespace factorial_8_divisors_l328_32836

-- Define 8!
def factorial_8 : ℕ := 8*7*6*5*4*3*2*1

-- Define a function to count positive divisors
def count_positive_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_8_divisors : count_positive_divisors factorial_8 = 96 := by
  sorry

end factorial_8_divisors_l328_32836


namespace layla_score_comparison_l328_32819

/-- Represents a player in the game -/
inductive Player : Type
| Layla : Player
| Nahima : Player
| Ramon : Player
| Aria : Player

/-- Represents a round in the game -/
inductive Round : Type
| First : Round
| Second : Round
| Third : Round

/-- The scoring function for the game -/
def score (p : Player) (r : Round) : ℕ → ℕ :=
  match r with
  | Round.First => (· * 2)
  | Round.Second => (· * 3)
  | Round.Third => id

/-- The total score of a player across all rounds -/
def totalScore (p : Player) (s1 s2 s3 : ℕ) : ℕ :=
  score p Round.First s1 + score p Round.Second s2 + score p Round.Third s3

theorem layla_score_comparison :
  ∀ (nahima_total ramon_total aria_total : ℕ),
  totalScore Player.Layla 120 90 (760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90) = 760 →
  nahima_total + ramon_total + aria_total = 1330 - 760 →
  760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90 =
    nahima_total + ramon_total + aria_total - 320 :=
by sorry

end layla_score_comparison_l328_32819


namespace pattern_cannot_form_cube_l328_32842

/-- Represents a square in the pattern -/
structure Square :=
  (id : ℕ)

/-- Represents the pattern of squares -/
structure Pattern :=
  (center : Square)
  (top : Square)
  (left : Square)
  (right : Square)
  (front : Square)

/-- Represents a cube -/
structure Cube :=
  (faces : Fin 6 → Square)

/-- Defines the given pattern -/
def given_pattern : Pattern :=
  { center := ⟨0⟩
  , top := ⟨1⟩
  , left := ⟨2⟩
  , right := ⟨3⟩
  , front := ⟨4⟩ }

/-- Theorem stating that the given pattern cannot form a cube -/
theorem pattern_cannot_form_cube :
  ¬ ∃ (c : Cube), c.faces 0 = given_pattern.center ∧
                  c.faces 1 = given_pattern.top ∧
                  c.faces 2 = given_pattern.left ∧
                  c.faces 3 = given_pattern.right ∧
                  c.faces 4 = given_pattern.front :=
by
  sorry


end pattern_cannot_form_cube_l328_32842


namespace halfway_point_l328_32822

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 3/10) :
  (a + b) / 2 = 17/80 := by
  sorry

end halfway_point_l328_32822


namespace sum_of_digits_power_two_l328_32891

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_power_two : 
  (∀ n : ℕ, (n - s n) % 9 = 0) → 
  (2^2009 < 10^672) → 
  s (s (s (2^2009))) = 5 := by sorry

end sum_of_digits_power_two_l328_32891


namespace mans_speed_against_current_l328_32864

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 12 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 22 5 = 12 := by
  sorry

#eval speed_against_current 22 5

end mans_speed_against_current_l328_32864


namespace sin_2alpha_value_l328_32857

theorem sin_2alpha_value (α : Real) :
  2 * Real.cos (2 * α) = Real.sin (π / 4 - α) →
  Real.sin (2 * α) = -7 / 8 := by
  sorry

end sin_2alpha_value_l328_32857


namespace second_frog_hops_l328_32838

/-- Represents the number of hops taken by each frog -/
structure FrogHops :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- The conditions of the frog hopping problem -/
def frog_problem (hops : FrogHops) : Prop :=
  hops.first = 4 * hops.second ∧
  hops.second = 2 * hops.third ∧
  hops.fourth = 3 * hops.second ∧
  hops.first + hops.second + hops.third + hops.fourth = 156 ∧
  60 ≤ 120  -- represents the time constraint (60 meters in 2 minutes or less)

/-- The theorem stating that the second frog takes 18 hops -/
theorem second_frog_hops :
  ∃ (hops : FrogHops), frog_problem hops ∧ hops.second = 18 :=
by sorry

end second_frog_hops_l328_32838


namespace sufficient_not_necessary_l328_32807

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1/2 : ℝ)^a < (1/2 : ℝ)^b → Real.log (a + 1) > Real.log b) ∧
  ¬(Real.log (a + 1) > Real.log b → (1/2 : ℝ)^a < (1/2 : ℝ)^b) :=
sorry

end sufficient_not_necessary_l328_32807


namespace necklace_cost_l328_32870

/-- The cost of Scarlet's necklace given her savings and expenses -/
theorem necklace_cost (savings : ℕ) (earrings_cost : ℕ) (remaining : ℕ) : 
  savings = 80 → earrings_cost = 23 → remaining = 9 → 
  savings - earrings_cost - remaining = 48 := by
  sorry

end necklace_cost_l328_32870


namespace vector_simplification_l328_32852

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) 
  (h1 : A - C = A - B - (C - B)) 
  (h2 : B - D = B - C - (D - C)) : 
  A - C - (B - D) + (C - D) - (A - B) = 0 := by
  sorry

end vector_simplification_l328_32852


namespace m_range_l328_32897

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x * (x - 1) > 6}
def C (m : ℝ) : Set ℝ := {x | -1 + m < x ∧ x < 2 * m}

theorem m_range (m : ℝ) : 
  (C m).Nonempty ∧ C m ⊆ (A ∩ (Set.univ \ B)) → -1 < m ∧ m ≤ 1/2 := by
  sorry

end m_range_l328_32897


namespace sequence_can_be_arithmetic_and_geometric_l328_32845

theorem sequence_can_be_arithmetic_and_geometric :
  ∃ (a d : ℝ) (n : ℕ), a + d = 9 ∧ a + n * d = 729 ∧ a = 3 ∧
  ∃ (b r : ℝ) (m : ℕ), b * r = 9 ∧ b * r^m = 729 ∧ b = 3 := by
  sorry

end sequence_can_be_arithmetic_and_geometric_l328_32845


namespace blue_parrots_count_l328_32880

theorem blue_parrots_count (total : ℕ) (red_fraction green_fraction : ℚ) : 
  total = 120 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 20 := by
  sorry

end blue_parrots_count_l328_32880


namespace merry_go_round_revolutions_l328_32896

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 = 30) (hr2 : r2 = 10) (hn1 : n1 = 40) :
  let d1 := 2 * Real.pi * r1 * n1
  let n2 := d1 / (2 * Real.pi * r2)
  n2 = 120 := by sorry

end merry_go_round_revolutions_l328_32896


namespace stating_medication_duration_l328_32801

/-- Represents the number of pills in one supply of medication -/
def pills_per_supply : ℕ := 60

/-- Represents the fraction of a pill taken each time -/
def pill_fraction : ℚ := 1/3

/-- Represents the number of days between each dose -/
def days_between_doses : ℕ := 3

/-- Represents the number of types of medication -/
def medication_types : ℕ := 2

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- 
Theorem stating that the combined supply of medication will last 540 days,
which is approximately 18 months.
-/
theorem medication_duration :
  (pills_per_supply : ℚ) * days_between_doses / pill_fraction * medication_types = 540 ∧
  540 / days_per_month = 18 := by
  sorry


end stating_medication_duration_l328_32801


namespace gift_cost_proof_l328_32867

theorem gift_cost_proof (initial_friends : Nat) (dropped_out : Nat) (share_increase : Int) :
  initial_friends = 10 →
  dropped_out = 4 →
  share_increase = 8 →
  ∃ (cost : Int),
    cost > 0 ∧
    (cost / (initial_friends - dropped_out : Int) = cost / initial_friends + share_increase) ∧
    cost = 120 := by
  sorry

end gift_cost_proof_l328_32867


namespace bottom_right_height_l328_32850

/-- Represents a rectangle with area and height -/
structure Rectangle where
  area : ℝ
  height : Option ℝ

/-- Represents the layout of six rectangles -/
structure RectangleLayout where
  topLeft : Rectangle
  topMiddle : Rectangle
  topRight : Rectangle
  bottomLeft : Rectangle
  bottomMiddle : Rectangle
  bottomRight : Rectangle

/-- Given the layout of rectangles, prove the height of the bottom right rectangle is 5 -/
theorem bottom_right_height (layout : RectangleLayout) :
  layout.topLeft.area = 18 ∧
  layout.bottomLeft.area = 12 ∧
  layout.bottomMiddle.area = 16 ∧
  layout.topMiddle.area = 32 ∧
  layout.topRight.area = 48 ∧
  layout.bottomRight.area = 30 ∧
  layout.topLeft.height = some 6 →
  layout.bottomRight.height = some 5 := by
  sorry

#check bottom_right_height

end bottom_right_height_l328_32850


namespace integer_product_sum_l328_32824

theorem integer_product_sum (x y : ℤ) : y = x + 2 ∧ x * y = 644 → x + y = 50 := by
  sorry

end integer_product_sum_l328_32824


namespace two_thousand_twentieth_digit_l328_32878

/-- Represents the decimal number formed by concatenating integers from 1 to 1000 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in the number x -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The 2020th digit after the decimal point in x is 7 -/
theorem two_thousand_twentieth_digit : nth_digit 2020 = 7 := by sorry

end two_thousand_twentieth_digit_l328_32878


namespace solution_set_of_inequality_l328_32800

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  {a : ℝ | f 1 < f a} = {a : ℝ | a < -1 ∨ 1 < a} :=
sorry

end solution_set_of_inequality_l328_32800


namespace geometric_sequence_first_term_l328_32813

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  ratio : ℝ

/-- Returns the nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * seq.ratio ^ (n - 1)

theorem geometric_sequence_first_term
  (seq : GeometricSequence)
  (h3 : seq.nthTerm 3 = 720)
  (h7 : seq.nthTerm 7 = 362880) :
  seq.firstTerm = 20 := by
sorry

end geometric_sequence_first_term_l328_32813


namespace last_digit_of_seven_power_seven_power_l328_32873

theorem last_digit_of_seven_power_seven_power (n : ℕ) : 7^(7^7) ≡ 3 [ZMOD 10] := by
  sorry

end last_digit_of_seven_power_seven_power_l328_32873
