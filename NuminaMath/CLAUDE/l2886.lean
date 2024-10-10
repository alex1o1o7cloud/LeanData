import Mathlib

namespace opponent_score_proof_l2886_288603

def championship_game (total_points : ℕ) (num_games : ℕ) (point_difference : ℕ) : Prop :=
  let avg_points : ℚ := (total_points : ℚ) / num_games
  let uf_championship_score : ℚ := avg_points / 2 - 2
  let opponent_score : ℚ := uf_championship_score + point_difference
  opponent_score = 15

theorem opponent_score_proof :
  championship_game 720 24 2 := by
  sorry

end opponent_score_proof_l2886_288603


namespace ednas_neighbors_l2886_288664

/-- The number of cookies Edna made -/
def total_cookies : ℕ := 150

/-- The number of cookies each neighbor (except Sarah) took -/
def cookies_per_neighbor : ℕ := 10

/-- The number of cookies Sarah took -/
def sarah_cookies : ℕ := 12

/-- The number of cookies left for the last neighbor -/
def cookies_left : ℕ := 8

/-- The number of Edna's neighbors -/
def num_neighbors : ℕ := 14

theorem ednas_neighbors :
  total_cookies = num_neighbors * cookies_per_neighbor + (sarah_cookies - cookies_per_neighbor) + cookies_left :=
by sorry

end ednas_neighbors_l2886_288664


namespace key_arrangement_count_l2886_288607

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of effective units to arrange (treating the adjacent pair as one unit) -/
def effective_units : ℕ := total_keys - 1

/-- The number of ways to arrange the adjacent pair -/
def adjacent_pair_arrangements : ℕ := 2

/-- The number of distinct circular arrangements of n objects -/
def circular_arrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := circular_arrangements effective_units * adjacent_pair_arrangements

theorem key_arrangement_count : total_arrangements = 48 := by sorry

end key_arrangement_count_l2886_288607


namespace seagull_problem_l2886_288610

theorem seagull_problem (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 18 → initial = 36 := by
  sorry

end seagull_problem_l2886_288610


namespace two_fifths_in_four_fifths_minus_one_tenth_l2886_288625

theorem two_fifths_in_four_fifths_minus_one_tenth : 
  (4/5 - 1/10) / (2/5) = 7/4 := by
  sorry

end two_fifths_in_four_fifths_minus_one_tenth_l2886_288625


namespace time_to_drain_pool_l2886_288612

/-- The time it takes to drain a rectangular pool given its dimensions, capacity, and drainage rate. -/
theorem time_to_drain_pool 
  (length width depth : ℝ) 
  (capacity : ℝ) 
  (drainage_rate : ℝ) 
  (h1 : length = 150)
  (h2 : width = 50)
  (h3 : depth = 10)
  (h4 : capacity = 0.8)
  (h5 : drainage_rate = 60) :
  (length * width * depth * capacity) / drainage_rate = 1000 := by
  sorry


end time_to_drain_pool_l2886_288612


namespace min_value_fraction_lower_bound_achievable_l2886_288615

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (x + y) / (x * y * z) ≥ 16 := by
  sorry

theorem lower_bound_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ (x + y) / (x * y * z) = 16 := by
  sorry

end min_value_fraction_lower_bound_achievable_l2886_288615


namespace problem_solution_l2886_288618

theorem problem_solution : 
  (Real.sqrt 48 - Real.sqrt 27 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3) ∧
  ((Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 - 1) := by
  sorry

end problem_solution_l2886_288618


namespace set_a_values_l2886_288636

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem set_a_values (a : ℝ) : A ∪ B a = A ↔ a = -2 ∨ a ≥ 4 ∨ a < -4 := by
  sorry

end set_a_values_l2886_288636


namespace exterior_angle_sum_is_360_l2886_288668

/-- A convex polygon with n sides and equilateral triangles attached to each side -/
structure ConvexPolygonWithTriangles where
  n : ℕ  -- number of sides of the original polygon
  [n_pos : Fact (n > 0)]

/-- The sum of exterior angles of a convex polygon with attached equilateral triangles -/
def exterior_angle_sum (p : ConvexPolygonWithTriangles) : ℝ :=
  360

/-- Theorem: The sum of all exterior angles in a convex polygon with attached equilateral triangles is 360° -/
theorem exterior_angle_sum_is_360 (p : ConvexPolygonWithTriangles) :
  exterior_angle_sum p = 360 := by
  sorry

end exterior_angle_sum_is_360_l2886_288668


namespace technician_round_trip_completion_l2886_288683

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let one_way := distance
  let return_portion := 0.4 * distance
  let total_distance := 2 * distance
  let completed_distance := one_way + return_portion
  (completed_distance / total_distance) * 100 = 70 := by
sorry

end technician_round_trip_completion_l2886_288683


namespace remainder_of_large_power_l2886_288635

theorem remainder_of_large_power (n : ℕ) : 
  4^(4^(4^4)) ≡ 656 [ZMOD 1000] :=
sorry

end remainder_of_large_power_l2886_288635


namespace integral_3x_plus_sinx_l2886_288666

theorem integral_3x_plus_sinx (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = (3 / 8) * Real.pi^2 + 1 := by
  sorry

end integral_3x_plus_sinx_l2886_288666


namespace candy_heating_rate_l2886_288614

/-- Candy heating problem -/
theorem candy_heating_rate
  (initial_temp : ℝ)
  (max_temp : ℝ)
  (final_temp : ℝ)
  (cooling_rate : ℝ)
  (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : final_temp = 170)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46)
  : ∃ (heating_rate : ℝ), heating_rate = 5 := by
  sorry

end candy_heating_rate_l2886_288614


namespace theater_capacity_is_50_l2886_288639

/-- The maximum capacity of a movie theater -/
def theater_capacity (ticket_price : ℕ) (tickets_sold : ℕ) (loss_amount : ℕ) : ℕ :=
  tickets_sold + loss_amount / ticket_price

/-- Theorem: The maximum capacity of the movie theater is 50 people -/
theorem theater_capacity_is_50 :
  theater_capacity 8 24 208 = 50 := by
  sorry

end theater_capacity_is_50_l2886_288639


namespace road_repair_hours_l2886_288600

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 39)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 26)
  (h5 : hours2 = 3)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
sorry

end road_repair_hours_l2886_288600


namespace johnny_tables_l2886_288617

/-- The number of tables that can be built given a total number of planks and planks required per table -/
def tables_built (total_planks : ℕ) (planks_per_table : ℕ) : ℕ :=
  total_planks / planks_per_table

/-- Theorem: Given 45 planks of wood and 9 planks required per table, 5 tables can be built -/
theorem johnny_tables : tables_built 45 9 = 5 := by
  sorry

end johnny_tables_l2886_288617


namespace factorization_equality_l2886_288688

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end factorization_equality_l2886_288688


namespace sum_remainder_mod_nine_l2886_288647

theorem sum_remainder_mod_nine : (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := by
  sorry

end sum_remainder_mod_nine_l2886_288647


namespace power_relation_l2886_288667

theorem power_relation (x m n : ℝ) (hm : x^m = 3) (hn : x^n = 5) :
  x^(2*m - 3*n) = 9/125 := by
  sorry

end power_relation_l2886_288667


namespace f_properties_l2886_288674

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi/6) + 1

theorem f_properties :
  let T := Real.pi
  let interval := Set.Icc (Real.pi/4) ((2*Real.pi)/3)
  (∀ x, f (x + T) = f x) ∧  -- Smallest positive period
  (∀ x ∈ interval, f x ≤ 2) ∧  -- Maximum value
  (∃ x ∈ interval, f x = 2) ∧  -- Maximum value is attained
  (∀ x ∈ interval, f x ≥ -1) ∧  -- Minimum value
  (∃ x ∈ interval, f x = -1) :=  -- Minimum value is attained
by
  sorry

end f_properties_l2886_288674


namespace markup_percentage_is_20_l2886_288637

/-- Calculate the markup percentage given cost price, discount, and profit percentage --/
def calculate_markup_percentage (cost_price discount : ℕ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price + (cost_price * profit_percentage / 100)
  let marked_price := selling_price + discount
  (marked_price - cost_price) / cost_price * 100

/-- Theorem stating that the markup percentage is 20% given the specified conditions --/
theorem markup_percentage_is_20 :
  calculate_markup_percentage 180 50 20 = 20 := by
  sorry

end markup_percentage_is_20_l2886_288637


namespace toy_store_revenue_l2886_288611

theorem toy_store_revenue (december : ℝ) (november : ℝ) (january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/6) * november) : 
  december = (20/7) * ((november + january) / 2) := by
sorry

end toy_store_revenue_l2886_288611


namespace monomial_equality_l2886_288695

-- Define variables
variable (a b : ℝ)
variable (x : ℝ)

-- Define the theorem
theorem monomial_equality (h : x * (2 * a^2 * b) = 2 * a^3 * b) : x = a := by
  sorry

end monomial_equality_l2886_288695


namespace stadium_length_conversion_l2886_288622

/-- Converts yards to feet given the number of yards and the conversion factor. -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Proves that 62 yards is equal to 186 feet when converted. -/
theorem stadium_length_conversion :
  let stadium_length_yards : ℕ := 62
  let yards_to_feet_conversion : ℕ := 3
  yards_to_feet stadium_length_yards yards_to_feet_conversion = 186 := by
  sorry

#check stadium_length_conversion

end stadium_length_conversion_l2886_288622


namespace weighted_average_is_70_55_l2886_288638

def mathematics_score : ℝ := 76
def science_score : ℝ := 65
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 55
def computer_science_score : ℝ := 89
def history_score : ℝ := 74
def geography_score : ℝ := 63
def physics_score : ℝ := 78
def chemistry_score : ℝ := 71

def mathematics_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def social_studies_weight : ℝ := 0.10
def english_weight : ℝ := 0.15
def biology_weight : ℝ := 0.10
def computer_science_weight : ℝ := 0.05
def history_weight : ℝ := 0.05
def geography_weight : ℝ := 0.10
def physics_weight : ℝ := 0.05
def chemistry_weight : ℝ := 0.05

def weighted_average : ℝ :=
  mathematics_score * mathematics_weight +
  science_score * science_weight +
  social_studies_score * social_studies_weight +
  english_score * english_weight +
  biology_score * biology_weight +
  computer_science_score * computer_science_weight +
  history_score * history_weight +
  geography_score * geography_weight +
  physics_score * physics_weight +
  chemistry_score * chemistry_weight

theorem weighted_average_is_70_55 : weighted_average = 70.55 := by
  sorry

end weighted_average_is_70_55_l2886_288638


namespace lara_age_proof_l2886_288689

/-- Lara's age 10 years from now, given her age 7 years ago -/
def lara_future_age (age_7_years_ago : ℕ) : ℕ :=
  age_7_years_ago + 7 + 10

/-- Theorem stating Lara's age 10 years from now -/
theorem lara_age_proof :
  lara_future_age 9 = 26 := by
  sorry

end lara_age_proof_l2886_288689


namespace matrix_commute_equality_l2886_288654

theorem matrix_commute_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A + B = A * B →
  A * B = ![![1, 2], ![3, 4]] →
  (A * B = B * A) →
  B * A = ![![1, 2], ![3, 4]] := by
  sorry

end matrix_commute_equality_l2886_288654


namespace lemniscate_orthogonal_trajectories_l2886_288604

-- Define the lemniscate family
def lemniscate (a : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = a * Real.cos (2 * φ)

-- Define the orthogonal trajectory
def orthogonal_trajectory (C : ℝ) (ρ φ : ℝ) : Prop :=
  ρ^2 = C * Real.sin (2 * φ)

-- Theorem statement
theorem lemniscate_orthogonal_trajectories (a C : ℝ) (ρ φ : ℝ) :
  lemniscate a ρ φ → orthogonal_trajectory C ρ φ :=
by
  sorry

end lemniscate_orthogonal_trajectories_l2886_288604


namespace smallest_number_with_remainder_two_l2886_288602

theorem smallest_number_with_remainder_two (n : ℕ) : 
  (n % 3 = 2 ∧ n % 4 = 2 ∧ n % 6 = 2 ∧ n % 8 = 2) → n ≥ 26 :=
by sorry

end smallest_number_with_remainder_two_l2886_288602


namespace max_difference_on_board_l2886_288658

/-- A type representing a 10x10 board with numbers from 1 to 100 -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- A predicate that checks if a board is valid (each number appears exactly once) -/
def is_valid_board (b : Board) : Prop :=
  ∀ n : Fin 100, ∃! (i j : Fin 10), b i j = n

/-- The main theorem statement -/
theorem max_difference_on_board :
  ∀ b : Board, is_valid_board b →
    ∃ (i j k : Fin 10), 
      (i = k ∨ j = k) ∧ 
      ((b i j : ℕ) ≥ (b k j : ℕ) + 54 ∨ (b k j : ℕ) ≥ (b i j : ℕ) + 54) :=
by sorry

end max_difference_on_board_l2886_288658


namespace cube_edge_ratio_l2886_288685

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 := by
  sorry

end cube_edge_ratio_l2886_288685


namespace special_polygon_area_l2886_288679

/-- A polygon with special properties -/
structure SpecialPolygon where
  sides : ℕ
  perimeter : ℝ
  is_perpendicular : Bool
  is_equal_length : Bool

/-- The area of a special polygon -/
def area (p : SpecialPolygon) : ℝ := sorry

/-- Theorem: The area of a special polygon with 36 sides and perimeter 72 is 144 -/
theorem special_polygon_area :
  ∀ (p : SpecialPolygon),
    p.sides = 36 ∧
    p.perimeter = 72 ∧
    p.is_perpendicular ∧
    p.is_equal_length →
    area p = 144 := by
  sorry

end special_polygon_area_l2886_288679


namespace abby_and_damon_weight_l2886_288687

/-- Given the weights of pairs of people, prove that Abby and Damon's combined weight is 285 pounds. -/
theorem abby_and_damon_weight
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 260)  -- Abby and Bart's combined weight
  (h2 : b + c = 245)  -- Bart and Cindy's combined weight
  (h3 : c + d = 270)  -- Cindy and Damon's combined weight
  : a + d = 285 := by
  sorry

#check abby_and_damon_weight

end abby_and_damon_weight_l2886_288687


namespace side_to_base_ratio_l2886_288620

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  -- The length of one side of the isosceles triangle
  side : ℝ
  -- The length of the base of the isosceles triangle
  base : ℝ
  -- The distance from the vertex to the point of tangency on the side
  vertex_to_tangency : ℝ
  -- Ensure the triangle is isosceles
  isosceles : side > 0
  -- Ensure the point of tangency divides the side in 7:5 ratio
  tangency_ratio : vertex_to_tangency / (side - vertex_to_tangency) = 7 / 5

/-- 
Theorem: In an isosceles triangle with an inscribed circle, 
if the point of tangency on one side divides it in the ratio 7:5 (starting from the vertex), 
then the ratio of the side to the base is 6:5.
-/
theorem side_to_base_ratio 
  (triangle : IsoscelesTriangleWithInscribedCircle) : 
  triangle.side / triangle.base = 6 / 5 := by
  sorry

end side_to_base_ratio_l2886_288620


namespace circle_coloring_theorem_l2886_288692

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a coloring of the plane -/
def Coloring := ℝ × ℝ → Bool

/-- Checks if two points are on opposite sides of a circle -/
def oppositeSides (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2)
  let d2 := Real.sqrt ((p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2)
  (d1 < c.radius ∧ d2 > c.radius) ∨ (d1 > c.radius ∧ d2 < c.radius)

/-- Checks if a coloring is valid for a given set of circles -/
def validColoring (circles : List Circle) (coloring : Coloring) : Prop :=
  ∀ c ∈ circles, ∀ p1 p2 : ℝ × ℝ, oppositeSides c p1 p2 → coloring p1 ≠ coloring p2

theorem circle_coloring_theorem (n : ℕ) (hn : n > 0) (circles : List Circle) 
    (hc : circles.length = n) : 
    ∃ coloring : Coloring, validColoring circles coloring := by
  sorry

end circle_coloring_theorem_l2886_288692


namespace equation_solution_l2886_288653

theorem equation_solution : ∃ x : ℚ, 3 * x + 6 = |(-19 + 5)| ∧ x = 8 / 3 := by
  sorry

end equation_solution_l2886_288653


namespace min_value_ab_l2886_288680

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - 2 * a - b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y - 2 * x - y = 0 → a * b ≤ x * y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x * y = 8 :=
by sorry

end min_value_ab_l2886_288680


namespace mary_nancy_balloon_ratio_l2886_288681

def nancy_balloons : ℕ := 7
def mary_balloons : ℕ := 28

theorem mary_nancy_balloon_ratio :
  mary_balloons / nancy_balloons = 4 := by sorry

end mary_nancy_balloon_ratio_l2886_288681


namespace player_a_not_losing_probability_l2886_288650

theorem player_a_not_losing_probability
  (p_win : ℝ)
  (p_draw : ℝ)
  (h_win : p_win = 0.3)
  (h_draw : p_draw = 0.5) :
  p_win + p_draw = 0.8 := by
  sorry

end player_a_not_losing_probability_l2886_288650


namespace season_games_l2886_288652

/-- Represents the basketball season statistics --/
structure SeasonStats where
  total_points : ℕ
  avg_free_throws : ℕ
  avg_two_pointers : ℕ
  avg_three_pointers : ℕ

/-- Calculates the number of games in the season --/
def calculate_games (stats : SeasonStats) : ℕ :=
  stats.total_points / (stats.avg_free_throws + 2 * stats.avg_two_pointers + 3 * stats.avg_three_pointers)

/-- Theorem stating that the number of games in the season is 15 --/
theorem season_games (stats : SeasonStats) 
  (h1 : stats.total_points = 345)
  (h2 : stats.avg_free_throws = 4)
  (h3 : stats.avg_two_pointers = 5)
  (h4 : stats.avg_three_pointers = 3) :
  calculate_games stats = 15 := by
  sorry

end season_games_l2886_288652


namespace rectangular_box_diagonals_l2886_288672

theorem rectangular_box_diagonals 
  (x y z : ℝ) 
  (surface_area : 2 * (x*y + y*z + z*x) = 106) 
  (edge_sum : 4 * (x + y + z) = 52) :
  4 * Real.sqrt (x^2 + y^2 + z^2) = 12 * Real.sqrt 7 := by
  sorry

end rectangular_box_diagonals_l2886_288672


namespace p_necessary_not_sufficient_for_p_and_q_l2886_288645

theorem p_necessary_not_sufficient_for_p_and_q :
  (∃ p q : Prop, (p ∧ q → p) ∧ ¬(p → p ∧ q)) := by sorry

end p_necessary_not_sufficient_for_p_and_q_l2886_288645


namespace range_interval_length_l2886_288673

-- Define the geometric sequence and its sum
def a (n : ℕ) : ℚ := 3/2 * (-1/2)^(n-1)
def S (n : ℕ) : ℚ := 1 - (-1/2)^n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := S n + 1 / S n

-- State the theorem
theorem range_interval_length :
  (∀ n : ℕ, n > 0 → -2 * S 2 + 4 * S 4 = 2 * S 3) →
  (∃ L : ℚ, L > 0 ∧ ∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L ∧ b n ∈ Set.Icc x y) ∧
  (∀ L' : ℚ, L' > 0 → (∀ n : ℕ, n > 0 → ∃ x y : ℚ, x < y ∧ y - x = L' ∧ b n ∈ Set.Icc x y) → L' ≥ 1/6) :=
sorry

end range_interval_length_l2886_288673


namespace square_has_four_axes_of_symmetry_l2886_288606

-- Define the shapes
inductive Shape
  | Square
  | Rhombus
  | Rectangle
  | IsoscelesTrapezoid

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : Nat :=
  match s with
  | Shape.Square => 4
  | Shape.Rhombus => 2
  | Shape.Rectangle => 2
  | Shape.IsoscelesTrapezoid => 1

-- Theorem statement
theorem square_has_four_axes_of_symmetry :
  ∀ s : Shape, axesOfSymmetry s = 4 → s = Shape.Square := by
  sorry

#check square_has_four_axes_of_symmetry

end square_has_four_axes_of_symmetry_l2886_288606


namespace probability_range_for_event_A_l2886_288662

theorem probability_range_for_event_A (p : ℝ) : 
  (0 ≤ p ∧ p < 1) →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2) →
  0.4 ≤ p ∧ p < 1 :=
sorry

end probability_range_for_event_A_l2886_288662


namespace least_positive_integer_with_given_remainders_l2886_288616

theorem least_positive_integer_with_given_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  x % 9 = 8 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 9 = 8 → x ≤ y :=
by
  -- The proof goes here
  sorry

end least_positive_integer_with_given_remainders_l2886_288616


namespace min_disks_is_ten_l2886_288634

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Rat
  files_06MB : Nat
  files_10MB : Nat
  files_03MB : Nat

/-- Calculates the minimum number of disks needed for the given storage problem. -/
def min_disks_needed (problem : StorageProblem) : Nat :=
  sorry

/-- Theorem stating that the minimum number of disks needed is 10 for the given problem. -/
theorem min_disks_is_ten (problem : StorageProblem) 
  (h1 : problem.total_files = 25)
  (h2 : problem.disk_capacity = 2)
  (h3 : problem.files_06MB = 5)
  (h4 : problem.files_10MB = 10)
  (h5 : problem.files_03MB = 10) :
  min_disks_needed problem = 10 := by
  sorry

end min_disks_is_ten_l2886_288634


namespace class_size_l2886_288655

theorem class_size (initial_avg : ℝ) (misread_weight : ℝ) (correct_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  final_avg = 58.9 →
  ∃ n : ℕ, n > 0 ∧ n * initial_avg + (correct_weight - misread_weight) = n * final_avg ∧ n = 20 :=
by sorry

end class_size_l2886_288655


namespace rachel_painting_time_l2886_288609

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time : ℝ → ℝ → ℝ → Prop :=
  fun matt_time patty_time rachel_time =>
    matt_time = 12 ∧
    patty_time = matt_time / 3 ∧
    rachel_time = 2 * patty_time + 5 →
    rachel_time = 13

/-- Proof of the theorem -/
lemma rachel_painting_time_proof : rachel_painting_time 12 4 13 := by
  sorry


end rachel_painting_time_l2886_288609


namespace tangent_length_to_circle_l2886_288691

/-- The length of the tangent from the origin to a circle passing through specific points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (4, 5) → B = (8, 10) → C = (7, 17) → 
  ∃ (circle : Set (ℝ × ℝ)) (tangent : ℝ × ℝ → ℝ),
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (tangent (0, 0) = 2 * Real.sqrt 41) := by
  sorry

end tangent_length_to_circle_l2886_288691


namespace sarah_marriage_age_l2886_288684

/-- The age at which a person will get married according to the game -/
def marriage_age (current_age : ℕ) (name_length : ℕ) : ℕ :=
  name_length + 2 * current_age

/-- Sarah's current age -/
def sarah_age : ℕ := 9

/-- The number of letters in Sarah's name -/
def sarah_name_length : ℕ := 5

/-- Theorem stating that Sarah will get married at age 23 according to the game -/
theorem sarah_marriage_age : marriage_age sarah_age sarah_name_length = 23 := by
  sorry

end sarah_marriage_age_l2886_288684


namespace value_of_m_l2886_288694

theorem value_of_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end value_of_m_l2886_288694


namespace line_through_point_l2886_288613

/-- The value of k for which the line -3/4 - 3kx = 7y passes through (1/3, -8) -/
theorem line_through_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3) = 7 * (-8) → k = 55.25 := by
  sorry

end line_through_point_l2886_288613


namespace min_distance_to_circle_l2886_288631

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos θ + ρ * Real.sin θ + 4 = 0

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y = 0

/-- Distance between a point (ρ, θ) and its tangent to circle C -/
noncomputable def distance_to_tangent (ρ θ : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum distance and its occurrence -/
theorem min_distance_to_circle (ρ θ : ℝ) :
  line_l ρ θ →
  distance_to_tangent ρ θ ≥ 2 ∧
  (distance_to_tangent ρ θ = 2 ↔ ρ = 2 ∧ θ = Real.pi) :=
  sorry

end min_distance_to_circle_l2886_288631


namespace triangle_formation_l2886_288675

-- Define the triangle formation condition
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_formation (a : ℝ) : 
  can_form_triangle 5 a 9 ↔ a = 6 :=
sorry

end triangle_formation_l2886_288675


namespace remaining_fabric_is_294_l2886_288699

/-- Represents the flag-making scenario with given initial conditions -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSize : ℕ
  wideFlagWidth : ℕ
  wideFlagHeight : ℕ
  tallFlagWidth : ℕ
  tallFlagHeight : ℕ
  squareFlagsMade : ℕ
  wideFlagsMade : ℕ
  tallFlagsMade : ℕ

/-- Calculates the remaining fabric after making flags -/
def remainingFabric (scenario : FlagScenario) : ℕ :=
  scenario.totalFabric -
  (scenario.squareFlagSize * scenario.squareFlagSize * scenario.squareFlagsMade +
   scenario.wideFlagWidth * scenario.wideFlagHeight * scenario.wideFlagsMade +
   scenario.tallFlagWidth * scenario.tallFlagHeight * scenario.tallFlagsMade)

/-- Theorem stating that the remaining fabric is 294 square feet -/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSize = 4)
  (h3 : scenario.wideFlagWidth = 5)
  (h4 : scenario.wideFlagHeight = 3)
  (h5 : scenario.tallFlagWidth = 3)
  (h6 : scenario.tallFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.wideFlagsMade = 20)
  (h9 : scenario.tallFlagsMade = 10) :
  remainingFabric scenario = 294 := by
  sorry

end remaining_fabric_is_294_l2886_288699


namespace thirteenth_result_l2886_288669

theorem thirteenth_result (results : List ℝ) 
  (h1 : results.length = 25)
  (h2 : results.sum / 25 = 19)
  (h3 : (results.take 12).sum / 12 = 14)
  (h4 : (results.drop 13).sum / 12 = 17) :
  results[12] = 103 := by
  sorry

end thirteenth_result_l2886_288669


namespace actual_average_height_l2886_288629

/-- The actual average height of boys in a class with measurement errors -/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (error1 : ℝ) (error2 : ℝ) : 
  n = 40 → 
  initial_avg = 184 → 
  error1 = 166 - 106 → 
  error2 = 190 - 180 → 
  (n * initial_avg - (error1 + error2)) / n = 182.25 := by
  sorry

end actual_average_height_l2886_288629


namespace barry_cycling_time_difference_barry_cycling_proof_l2886_288686

theorem barry_cycling_time_difference : ℝ → Prop :=
  λ time_diff : ℝ =>
    let total_distance : ℝ := 4 * 3
    let time_at_varying_speeds : ℝ := 2 * (3 / 6) + 1 * (3 / 3) + 1 * (3 / 5)
    let time_at_constant_speed : ℝ := total_distance / 5
    let time_diff_hours : ℝ := time_at_varying_speeds - time_at_constant_speed
    time_diff = time_diff_hours * 60 ∧ time_diff = 42

theorem barry_cycling_proof : barry_cycling_time_difference 42 := by
  sorry

end barry_cycling_time_difference_barry_cycling_proof_l2886_288686


namespace shaded_ratio_is_one_ninth_l2886_288690

-- Define the structure of our square grid
def SquareGrid :=
  { n : ℕ // n > 0 }

-- Define the large square
def LargeSquare : SquareGrid :=
  ⟨6, by norm_num⟩

-- Define the number of squares in the shaded region
def ShadedSquares : ℕ := 4

-- Define the ratio of shaded area to total area
def ShadedRatio (grid : SquareGrid) (shaded : ℕ) : ℚ :=
  shaded / (grid.val ^ 2 : ℚ)

-- Theorem statement
theorem shaded_ratio_is_one_ninth :
  ShadedRatio LargeSquare ShadedSquares = 1 / 9 := by
  sorry

end shaded_ratio_is_one_ninth_l2886_288690


namespace sin_225_degrees_l2886_288670

theorem sin_225_degrees : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l2886_288670


namespace multiplier_problem_l2886_288643

theorem multiplier_problem (x m : ℝ) (h1 : x = -10) (h2 : m * x - 8 = -12) : m = 0.4 := by
  sorry

end multiplier_problem_l2886_288643


namespace parabola_properties_l2886_288630

/-- Parabola with symmetric axis at x = -2 passing through (1, -2) and c > 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetric_axis : a * (-2) + b = 0
  passes_through : a * 1^2 + b * 1 + c = -2
  c_positive : c > 0

theorem parabola_properties (p : Parabola) :
  p.a < 0 ∧ 16 * p.a + p.c > 4 * p.b := by
  sorry

end parabola_properties_l2886_288630


namespace miss_adamson_class_size_l2886_288659

theorem miss_adamson_class_size :
  let num_classes : ℕ := 4
  let sheets_per_student : ℕ := 5
  let total_sheets : ℕ := 400
  let total_students : ℕ := total_sheets / sheets_per_student
  let students_per_class : ℕ := total_students / num_classes
  students_per_class = 20 := by
  sorry

end miss_adamson_class_size_l2886_288659


namespace ultramindmaster_codes_l2886_288693

/-- The number of available colors in UltraMindmaster -/
def num_colors : ℕ := 8

/-- The number of slots in each secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in UltraMindmaster -/
def num_codes : ℕ := num_colors ^ num_slots

theorem ultramindmaster_codes :
  num_codes = 32768 := by
  sorry

end ultramindmaster_codes_l2886_288693


namespace sum_equals_300_l2886_288623

theorem sum_equals_300 : 192 + 58 + 42 + 8 = 300 := by sorry

end sum_equals_300_l2886_288623


namespace class_size_l2886_288651

theorem class_size (top_scorers : Nat) (zero_scorers : Nat) (top_score : Nat) (rest_avg : Nat) (class_avg : Nat) :
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_avg = 45 →
  class_avg = 42 →
  ∃ (N : Nat), N = 25 ∧ 
    (N * class_avg = top_scorers * top_score + zero_scorers * 0 + (N - top_scorers - zero_scorers) * rest_avg) :=
by sorry

end class_size_l2886_288651


namespace smallest_four_digit_divisible_by_35_l2886_288698

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end smallest_four_digit_divisible_by_35_l2886_288698


namespace steps_per_floor_l2886_288649

/-- Proves that the number of steps across each floor is 30 --/
theorem steps_per_floor (
  num_floors : ℕ) 
  (steps_per_second : ℕ)
  (total_time : ℕ)
  (h1 : num_floors = 9)
  (h2 : steps_per_second = 3)
  (h3 : total_time = 90)
  : (steps_per_second * total_time) / num_floors = 30 := by
  sorry

end steps_per_floor_l2886_288649


namespace ellipse_cos_angle_l2886_288656

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (M : ℝ × ℝ) : Prop :=
  ellipse M.1 M.2

-- Define perpendicularity condition
def perpendicular_condition (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₁.1) * (F₂.1 - F₁.1) + (M.2 - F₁.2) * (F₂.2 - F₁.2) = 0

-- Theorem statement
theorem ellipse_cos_angle (M F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse M →
  perpendicular_condition M F₁ F₂ →
  let MF₁ := Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2)
  let MF₂ := Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2)
  MF₁ / MF₂ = 2/7 :=
sorry

end ellipse_cos_angle_l2886_288656


namespace pizza_delivery_gas_remaining_l2886_288642

theorem pizza_delivery_gas_remaining (start_amount used_amount : ℚ) 
  (h1 : start_amount = 0.5)
  (h2 : used_amount = 0.33) : 
  start_amount - used_amount = 0.17 := by
sorry

end pizza_delivery_gas_remaining_l2886_288642


namespace complex_cube_root_sum_l2886_288697

theorem complex_cube_root_sum (a b : ℤ) (z : ℂ) : 
  z = a + b * Complex.I ∧ z^3 = 2 + 11 * Complex.I → a + b = 3 := by
  sorry

end complex_cube_root_sum_l2886_288697


namespace number_division_problem_l2886_288628

theorem number_division_problem (x : ℝ) : x / 0.3 = 7.3500000000000005 → x = 2.205 := by
  sorry

end number_division_problem_l2886_288628


namespace earth_surface_usage_l2886_288663

/-- The fraction of the Earth's surface that is land -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable -/
def inhabitable_fraction : ℚ := 2/3

/-- The fraction of inhabitable land used for agriculture and urban development -/
def used_fraction : ℚ := 3/4

/-- The fraction of the Earth's surface used for agriculture or urban purposes -/
def agriculture_urban_fraction : ℚ := land_fraction * inhabitable_fraction * used_fraction

theorem earth_surface_usage :
  agriculture_urban_fraction = 1/6 := by sorry

end earth_surface_usage_l2886_288663


namespace remainder_proof_l2886_288657

theorem remainder_proof (R1 : ℕ) : 
  (129 = Nat.gcd (1428 - R1) (2206 - 13)) → 
  (2206 % 129 = 13) → 
  (1428 % 129 = 19) :=
by
  sorry

end remainder_proof_l2886_288657


namespace perpendicular_line_through_point_l2886_288608

/-- Given a line L1 with equation x + 3y + 4 = 0, prove that the line L2 with equation 3x - y - 5 = 0
    passes through the point (2, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 3 * y + 4 = 0) →  -- Equation of line L1
  (3 * 2 - 1 - 5 = 0) →  -- L2 passes through (2, 1)
  (3 * (1 / 3) = -1) →   -- Slopes are negative reciprocals
  (3 * x - y - 5 = 0) -- Equation of line L2
  := by sorry

end perpendicular_line_through_point_l2886_288608


namespace ryan_study_difference_l2886_288632

/-- Ryan's daily study hours for different languages -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- The difference in study hours between Chinese and Spanish -/
def chineseSpanishDifference (h : StudyHours) : ℤ :=
  h.chinese - h.spanish

/-- Theorem stating the difference in study hours between Chinese and Spanish -/
theorem ryan_study_difference :
  ∀ (h : StudyHours),
    h.english = 2 → h.chinese = 5 → h.spanish = 4 →
    chineseSpanishDifference h = 1 := by
  sorry

end ryan_study_difference_l2886_288632


namespace power_equation_solution_l2886_288601

theorem power_equation_solution (x : ℝ) : (1 / 8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end power_equation_solution_l2886_288601


namespace complex_magnitude_l2886_288696

theorem complex_magnitude (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : a ≠ 0) 
  (h3 : Complex.abs z ^ 2 - 2 * z = Complex.mk 1 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l2886_288696


namespace train_length_calculation_l2886_288644

/-- The length of a train given crossing time and speeds --/
theorem train_length_calculation (crossing_time : ℝ) (man_speed : ℝ) (train_speed : ℝ) :
  crossing_time = 39.99680025597952 →
  man_speed = 2 →
  train_speed = 56 →
  ∃ (train_length : ℝ), abs (train_length - 599.95) < 0.01 :=
by
  sorry

end train_length_calculation_l2886_288644


namespace y_takes_70_days_l2886_288682

-- Define the work completion rates
def mahesh_rate : ℚ := 1 / 35
def rajesh_rate : ℚ := 1 / 30

-- Define the amount of work Mahesh completes
def mahesh_work : ℚ := mahesh_rate * 20

-- Define the amount of work Rajesh completes
def rajesh_work : ℚ := 1 - mahesh_work

-- Define Y's completion time
def y_completion_time : ℚ := 70

-- Theorem statement
theorem y_takes_70_days :
  y_completion_time = 70 := by sorry

end y_takes_70_days_l2886_288682


namespace stock_price_decrease_l2886_288648

/-- The percentage decrease required for a stock to return to its original price after a 40% increase -/
theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let increased_price := 1.4 * initial_price
  let decrease_percent := (increased_price - initial_price) / increased_price
  decrease_percent = 0.2857142857142857 := by
sorry

end stock_price_decrease_l2886_288648


namespace gcd_a4_3a2_1_a3_2a_eq_one_l2886_288624

theorem gcd_a4_3a2_1_a3_2a_eq_one (a : ℕ) : 
  Nat.gcd (a^4 + 3*a^2 + 1) (a^3 + 2*a) = 1 := by
  sorry

end gcd_a4_3a2_1_a3_2a_eq_one_l2886_288624


namespace gcd_of_B_is_two_l2886_288641

/-- The set of all numbers that can be represented as the sum of four consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

/-- The greatest common divisor of all numbers in set B is 2 -/
theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l2886_288641


namespace consecutive_triangular_not_square_infinitely_many_square_products_l2886_288676

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: The product of two consecutive triangular numbers is not a perfect square -/
theorem consecutive_triangular_not_square (n : ℕ) (h : n > 1) :
  ¬ ∃ m : ℕ, triangular_number (n - 1) * triangular_number n = m^2 := by sorry

/-- Statement: For each triangular number, there exist infinitely many larger triangular numbers
    such that their product is a perfect square -/
theorem infinitely_many_square_products (n : ℕ) :
  ∃ f : ℕ → ℕ, Monotone f ∧ (∀ k : ℕ, f k > n) ∧
  (∀ k : ℕ, ∃ m : ℕ, triangular_number n * triangular_number (f k) = m^2) := by sorry

end consecutive_triangular_not_square_infinitely_many_square_products_l2886_288676


namespace blackboard_numbers_l2886_288661

theorem blackboard_numbers (n : ℕ) (S : ℕ) (x : ℕ) : 
  S / n = 30 →
  (S + 100) / (n + 1) = 40 →
  (S + 100 + x) / (n + 2) = 50 →
  x = 120 := by
sorry

end blackboard_numbers_l2886_288661


namespace orthocenter_symmetry_and_equal_circles_l2886_288671

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point := sorry

/-- Checks if four points are on the same circle -/
def on_same_circle (A B C D : Point) (S : Circle) : Prop := sorry

/-- Checks if two quadrilaterals are symmetric with respect to a point -/
def symmetric_quadrilaterals (A B C D A' B' C' D' H : Point) : Prop := sorry

/-- Checks if four points are on a circle with the same radius as another circle -/
def on_equal_circle (A B C D : Point) (S : Circle) : Prop := sorry

theorem orthocenter_symmetry_and_equal_circles 
  (A₁ A₂ A₃ A₄ : Point) (S : Circle)
  (h_same_circle : on_same_circle A₁ A₂ A₃ A₄ S)
  (H₁ := orthocenter A₂ A₃ A₄)
  (H₂ := orthocenter A₁ A₃ A₄)
  (H₃ := orthocenter A₁ A₂ A₄)
  (H₄ := orthocenter A₁ A₂ A₃) :
  ∃ (H : Point),
    (symmetric_quadrilaterals A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ H) ∧
    (on_equal_circle A₁ A₂ H₃ H₄ S) ∧
    (on_equal_circle A₁ A₃ H₂ H₄ S) ∧
    (on_equal_circle A₁ A₄ H₂ H₃ S) ∧
    (on_equal_circle A₂ A₃ H₁ H₄ S) ∧
    (on_equal_circle A₂ A₄ H₁ H₃ S) ∧
    (on_equal_circle A₃ A₄ H₁ H₂ S) ∧
    (on_equal_circle H₁ H₂ H₃ H₄ S) :=
  sorry

end orthocenter_symmetry_and_equal_circles_l2886_288671


namespace solution_set_xfx_less_than_zero_l2886_288665

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem solution_set_xfx_less_than_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_positive f)
  (h_f_neg_three : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = {x | x < -3 ∨ x > 3} :=
sorry

end solution_set_xfx_less_than_zero_l2886_288665


namespace star_two_three_l2886_288640

-- Define the star operation
def star (c d : ℝ) : ℝ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

-- State the theorem
theorem star_two_three : star 2 3 = 125 := by
  sorry

end star_two_three_l2886_288640


namespace digit_sum_is_two_l2886_288626

/-- Given a four-digit number abcd and a three-digit number bcd, where a, b, c, d are distinct digits 
    and abcd - bcd is a two-digit number, the sum of a, b, c, and d is 2. -/
theorem digit_sum_is_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  1000 * a + 100 * b + 10 * c + d > 999 →
  1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100 →
  a + b + c + d = 2 := by
  sorry

end digit_sum_is_two_l2886_288626


namespace min_value_abc_min_value_abc_attainable_l2886_288627

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 3*c = a*b*c) : 
  a*b*c ≥ 9*Real.sqrt 2 := by
sorry

theorem min_value_abc_attainable : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + 2*b + 3*c = a*b*c ∧ a*b*c = 9*Real.sqrt 2 := by
sorry

end min_value_abc_min_value_abc_attainable_l2886_288627


namespace negation_equivalence_l2886_288619

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l2886_288619


namespace arrangement_count_is_540_l2886_288605

/-- The number of ways to arrange teachers and students into groups and locations -/
def arrangement_count : ℕ :=
  (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) * (Nat.factorial 3)

/-- Theorem stating that the number of arrangements is 540 -/
theorem arrangement_count_is_540 : arrangement_count = 540 := by
  sorry

end arrangement_count_is_540_l2886_288605


namespace teaching_years_sum_l2886_288677

/-- The combined years of teaching for Virginia, Adrienne, and Dennis -/
def combined_years (virginia adrienne dennis : ℕ) : ℕ := virginia + adrienne + dennis

/-- Theorem stating the combined years of teaching given the conditions -/
theorem teaching_years_sum :
  ∀ (virginia adrienne dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 40 →
  combined_years virginia adrienne dennis = 93 := by
sorry

end teaching_years_sum_l2886_288677


namespace line_through_point_l2886_288660

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → (2 = 2 * 1 + b) → b = 0 := by
  sorry

end line_through_point_l2886_288660


namespace leftover_value_is_230_l2886_288633

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a collection of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat

def roll_size : RollSize := { quarters := 25, dimes := 40 }

def john_coins : Coins := { quarters := 47, dimes := 71 }
def mark_coins : Coins := { quarters := 78, dimes := 132 }

def combine_coins (c1 c2 : Coins) : Coins :=
  { quarters := c1.quarters + c2.quarters,
    dimes := c1.dimes + c2.dimes }

def leftover_coins (c : Coins) (r : RollSize) : Coins :=
  { quarters := c.quarters % r.quarters,
    dimes := c.dimes % r.dimes }

def coin_value (c : Coins) : Rat :=
  (c.quarters : Rat) * (1/4) + (c.dimes : Rat) * (1/10)

theorem leftover_value_is_230 :
  let combined := combine_coins john_coins mark_coins
  let leftover := leftover_coins combined roll_size
  coin_value leftover = 23/10 := by sorry

end leftover_value_is_230_l2886_288633


namespace equation_proof_l2886_288646

theorem equation_proof : (36 / 18) * (36 / 72) = 1 := by
  sorry

end equation_proof_l2886_288646


namespace fibonacci_factorial_last_two_digits_sum_l2886_288621

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_last_two_digits (series : List ℕ) : ℕ :=
  (series.map (λ n => last_two_digits (factorial n))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end fibonacci_factorial_last_two_digits_sum_l2886_288621


namespace ellipse_range_l2886_288678

theorem ellipse_range (m n : ℝ) : 
  (m^2 / 3 + n^2 / 8 = 1) → 
  ∃ x : ℝ, x = Real.sqrt 3 * m ∧ -3 ≤ x ∧ x ≤ 3 :=
by sorry

end ellipse_range_l2886_288678
