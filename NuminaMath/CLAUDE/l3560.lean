import Mathlib

namespace weightlifting_winner_l3560_356088

theorem weightlifting_winner (A B C : ℕ) 
  (sum_AB : A + B = 220)
  (sum_AC : A + C = 240)
  (sum_BC : B + C = 250) :
  max A (max B C) = 135 := by
sorry

end weightlifting_winner_l3560_356088


namespace geometric_series_first_term_l3560_356006

theorem geometric_series_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 10)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10))) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end geometric_series_first_term_l3560_356006


namespace bike_trip_distance_l3560_356016

/-- Calculates the total distance traveled given outbound and return times and average speed -/
def total_distance (outbound_time return_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := (outbound_time + return_time) / 60
  total_time * average_speed

/-- Proves that the total distance traveled is 4 miles given the specified conditions -/
theorem bike_trip_distance :
  let outbound_time : ℚ := 15
  let return_time : ℚ := 25
  let average_speed : ℚ := 6
  total_distance outbound_time return_time average_speed = 4 := by
  sorry

#eval total_distance 15 25 6

end bike_trip_distance_l3560_356016


namespace parallel_lines_m_value_l3560_356094

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ / b₁ = a₂ / b₂

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end parallel_lines_m_value_l3560_356094


namespace hyperbola_foci_l3560_356065

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The foci of the given hyperbola are located at (0, ±√29) -/
theorem hyperbola_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔ 
    (∃ (x y : ℝ), hyperbola_equation x y ∧ 
      f = (x, y) ∧ 
      (∀ (x' y' : ℝ), hyperbola_equation x' y' → 
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) + (Real.sqrt 29))^2 ∨
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) - (Real.sqrt 29))^2)) :=
sorry

end hyperbola_foci_l3560_356065


namespace min_value_of_f_l3560_356087

/-- The quadratic function f(x) = 2(x-3)^2 + 2 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

/-- Theorem: The minimum value of f(x) = 2(x-3)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry

end min_value_of_f_l3560_356087


namespace complex_number_location_l3560_356010

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = 1 + Complex.I) :
  is_in_fourth_quadrant z :=
sorry

end complex_number_location_l3560_356010


namespace collinear_points_b_value_l3560_356071

/-- Given three points in 2D space -/
def point1 : ℝ × ℝ := (4, -3)
def point2 (b : ℝ) : ℝ × ℝ := (2*b + 1, 5)
def point3 (b : ℝ) : ℝ × ℝ := (-b + 3, 1)

/-- Function to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

/-- Theorem stating that b = 1/4 when the given points are collinear -/
theorem collinear_points_b_value :
  collinear point1 (point2 b) (point3 b) → b = 1/4 := by sorry

end collinear_points_b_value_l3560_356071


namespace circle_equation_and_intersection_points_l3560_356030

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 1)^2 = 2}

-- Define the line y = x
def line_tangent : Set (ℝ × ℝ) :=
  {p | p.1 = p.2}

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + a = 0}

theorem circle_equation_and_intersection_points (a : ℝ) 
  (h1 : a ≠ 0)
  (h2 : ∃ p, p ∈ circle_C ∩ line_tangent)
  (h3 : ∃ A B, A ∈ circle_C ∩ line_l a ∧ B ∈ circle_C ∩ line_l a ∧ A ≠ B)
  (h4 : ∀ A B, A ∈ circle_C ∩ line_l a → B ∈ circle_C ∩ line_l a → A ≠ B → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :
  (∀ p, p ∈ circle_C ↔ (p.1 - 3)^2 + (p.2 - 1)^2 = 2) ∧
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) := by sorry

end circle_equation_and_intersection_points_l3560_356030


namespace bead_calculation_l3560_356048

theorem bead_calculation (blue_beads yellow_beads : ℕ) 
  (h1 : blue_beads = 23)
  (h2 : yellow_beads = 16) : 
  let total_beads := blue_beads + yellow_beads
  let parts := 3
  let beads_per_part := total_beads / parts
  let removed_beads := 10
  let remaining_beads := beads_per_part - removed_beads
  remaining_beads * 2 = 6 := by
sorry

end bead_calculation_l3560_356048


namespace hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l3560_356037

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : ℝ :=
  let hyperbola_equation (x y : ℝ) := x^2/16 - y^2/9 = 1
  let vertices_distance := 8
  vertices_distance

/-- Proof of the theorem -/
theorem hyperbola_vertices_distance_proof : hyperbola_vertices_distance = 8 := by
  sorry

end hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l3560_356037


namespace square_of_negative_square_l3560_356029

theorem square_of_negative_square (x : ℝ) : (-x^2)^2 = x^4 := by
  sorry

end square_of_negative_square_l3560_356029


namespace sqrt_3_powers_l3560_356072

theorem sqrt_3_powers (n : ℕ) (h : n ≥ 1) :
  ∃ (k w : ℕ), 
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) + ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (2 * k : ℝ)) ∧
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) - ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (w : ℝ) * Real.sqrt 3) ∧
    w > 0 :=
by sorry

end sqrt_3_powers_l3560_356072


namespace overlapping_strips_area_l3560_356018

theorem overlapping_strips_area (left_length right_length total_length : ℝ)
  (left_only_area right_only_area : ℝ) :
  left_length = 9 →
  right_length = 7 →
  total_length = 16 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end overlapping_strips_area_l3560_356018


namespace rhombus_diagonal_length_l3560_356032

/-- 
Proves that in a rhombus with an area of 120 cm² and one diagonal of 20 cm, 
the length of the other diagonal is 12 cm.
-/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal1 : ℝ) 
  (diagonal2 : ℝ) 
  (h1 : area = 120) 
  (h2 : diagonal1 = 20) 
  (h3 : area = (diagonal1 * diagonal2) / 2) : 
  diagonal2 = 12 := by
sorry

end rhombus_diagonal_length_l3560_356032


namespace imaginary_part_of_one_plus_i_to_fifth_l3560_356026

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i) ^ 5) = -4 := by
  sorry

end imaginary_part_of_one_plus_i_to_fifth_l3560_356026


namespace division_remainder_l3560_356063

theorem division_remainder : Int.mod 1234567 256 = 503 := by
  sorry

end division_remainder_l3560_356063


namespace square_and_cube_sum_l3560_356019

theorem square_and_cube_sum (p q : ℝ) 
  (h1 : p * q = 15) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 34 ∧ p^3 + q^3 = 152 := by
  sorry

end square_and_cube_sum_l3560_356019


namespace quadratic_inequality_properties_l3560_356040

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0 ↔ x ≤ 3 ∨ x ≥ 4) →
  a > 0 ∧ a + b + c > 0 := by
sorry

end quadratic_inequality_properties_l3560_356040


namespace vegetable_bins_l3560_356069

theorem vegetable_bins (total_bins soup_bins pasta_bins : ℚ)
  (h1 : total_bins = 0.75)
  (h2 : soup_bins = 0.12)
  (h3 : pasta_bins = 0.5)
  (h4 : total_bins = soup_bins + pasta_bins + (total_bins - soup_bins - pasta_bins)) :
  total_bins - soup_bins - pasta_bins = 0.13 := by
sorry

end vegetable_bins_l3560_356069


namespace chess_club_mixed_groups_l3560_356025

/-- Represents the chess club and its game statistics -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ := 
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- The main theorem stating the number of mixed groups -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_vs_boy_games := 30,
    girl_vs_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end chess_club_mixed_groups_l3560_356025


namespace max_square_plots_l3560_356021

/-- Represents the dimensions of the park -/
structure ParkDimensions where
  width : ℕ
  length : ℕ

/-- Represents the constraints for the park division -/
structure ParkConstraints where
  dimensions : ParkDimensions
  pathwayMaterial : ℕ

/-- Calculates the number of square plots given the number of plots along the width -/
def calculatePlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the total length of pathways given the number of plots along the width -/
def calculatePathwayLength (n : ℕ) : ℕ := 120 * n - 90

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (constraints : ParkConstraints) 
  (h1 : constraints.dimensions.width = 30)
  (h2 : constraints.dimensions.length = 60)
  (h3 : constraints.pathwayMaterial = 2010) :
  ∃ (n : ℕ), calculatePlots n = 578 ∧ 
             calculatePathwayLength n ≤ constraints.pathwayMaterial ∧
             ∀ (m : ℕ), m > n → calculatePathwayLength m > constraints.pathwayMaterial :=
  by sorry


end max_square_plots_l3560_356021


namespace max_product_of_three_integers_l3560_356004

/-- 
Given three integers where two are equal and their sum is 2000,
prove that their maximum product is 8000000000/27.
-/
theorem max_product_of_three_integers (x y z : ℤ) : 
  x = y ∧ x + y + z = 2000 → 
  x * y * z ≤ 8000000000 / 27 := by
sorry

end max_product_of_three_integers_l3560_356004


namespace optimal_price_reduction_l3560_356080

/-- Represents the watermelon vendor's business model -/
structure WatermelonVendor where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialDailySales : ℝ
  salesIncreaseRate : ℝ
  fixedCosts : ℝ

/-- Calculates the daily sales volume based on price reduction -/
def dailySalesVolume (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  w.initialDailySales + w.salesIncreaseRate * priceReduction * 10

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  (w.initialSellingPrice - priceReduction - w.initialPurchasePrice) * 
  (dailySalesVolume w priceReduction) - w.fixedCosts

/-- Theorem stating the optimal price reduction for maximum sales and 200 yuan profit -/
theorem optimal_price_reduction (w : WatermelonVendor) 
  (h1 : w.initialPurchasePrice = 2)
  (h2 : w.initialSellingPrice = 3)
  (h3 : w.initialDailySales = 200)
  (h4 : w.salesIncreaseRate = 40)
  (h5 : w.fixedCosts = 24) :
  ∃ (x : ℝ), x = 0.3 ∧ 
  dailyProfit w x = 200 ∧ 
  ∀ (y : ℝ), dailyProfit w y = 200 → dailySalesVolume w x ≥ dailySalesVolume w y := by
  sorry


end optimal_price_reduction_l3560_356080


namespace function_properties_l3560_356068

variable (a b : ℝ × ℝ)

def f (x : ℝ) : ℝ := (x * a.1 + b.1) * (x * b.2 - a.2)

theorem function_properties
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0)  -- perpendicular vectors
  (h4 : a.1^2 + a.2^2 ≠ b.1^2 + b.2^2)  -- different magnitudes
  : (∃ k : ℝ, ∀ x : ℝ, f a b x = k * x) ∧  -- first-order function
    (∀ x : ℝ, f a b x = -f a b (-x))  -- odd function
  := by sorry

end function_properties_l3560_356068


namespace union_A_B_l3560_356099

def A : Set ℕ := {1, 2}

def B : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem union_A_B : A ∪ B = {1, 2, 3, 4} := by sorry

end union_A_B_l3560_356099


namespace factor_calculation_l3560_356083

theorem factor_calculation (x : ℝ) (factor : ℝ) : 
  x = 4 → (2 * x + 9) * factor = 51 → factor = 3 := by
  sorry

end factor_calculation_l3560_356083


namespace quadratic_inequality_solution_l3560_356055

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_solution_l3560_356055


namespace jerry_won_47_tickets_l3560_356097

/-- The number of tickets Jerry won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem: Jerry won 47 tickets later at the arcade -/
theorem jerry_won_47_tickets :
  tickets_won_later 4 2 49 = 47 := by
  sorry

end jerry_won_47_tickets_l3560_356097


namespace total_interest_calculation_total_interest_is_1530_l3560_356014

/-- Calculates the total interest earned on two certificates of deposit --/
theorem total_interest_calculation (total_investment : ℝ) (rate1 rate2 : ℝ) 
  (fraction_higher_rate : ℝ) : ℝ :=
  let amount_higher_rate := total_investment * fraction_higher_rate
  let amount_lower_rate := total_investment - amount_higher_rate
  let interest_higher_rate := amount_higher_rate * rate2
  let interest_lower_rate := amount_lower_rate * rate1
  interest_higher_rate + interest_lower_rate

/-- Proves that the total interest earned is $1,530 given the problem conditions --/
theorem total_interest_is_1530 : 
  total_interest_calculation 20000 0.06 0.09 0.55 = 1530 := by
  sorry

end total_interest_calculation_total_interest_is_1530_l3560_356014


namespace step_height_calculation_step_height_proof_l3560_356007

theorem step_height_calculation (num_flights : ℕ) (flight_height : ℕ) (total_steps : ℕ) (inches_per_foot : ℕ) : ℕ :=
  let total_height_feet := num_flights * flight_height
  let total_height_inches := total_height_feet * inches_per_foot
  total_height_inches / total_steps

theorem step_height_proof :
  step_height_calculation 9 10 60 12 = 18 := by
  sorry

end step_height_calculation_step_height_proof_l3560_356007


namespace distribute_four_balls_three_boxes_l3560_356044

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition a set of n objects into k non-empty subsets -/
def stirling_second (n k : ℕ) : ℕ := sorry

theorem distribute_four_balls_three_boxes : 
  distribute_balls 4 3 = 14 := by sorry

end distribute_four_balls_three_boxes_l3560_356044


namespace complex_number_in_first_quadrant_l3560_356095

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l3560_356095


namespace vector_operation_result_l3560_356093

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (3, -2, 1)
  let b : ℝ × ℝ × ℝ := (-2, 4, 0)
  let c : ℝ × ℝ × ℝ := (3, 0, 2)
  a - 2 • b + 4 • c = (19, -10, 9) :=
by sorry

end vector_operation_result_l3560_356093


namespace dislike_both_tv_and_sports_l3560_356079

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 40 / 100
def sports_dislike_percentage : ℚ := 15 / 100

theorem dislike_both_tv_and_sports :
  ∃ (n : ℕ), n = (total_surveyed : ℚ) * tv_dislike_percentage * sports_dislike_percentage ∧ n = 90 :=
by sorry

end dislike_both_tv_and_sports_l3560_356079


namespace inscribed_circle_radius_l3560_356009

/-- Configuration of semicircles and inscribed circle -/
structure CircleConfiguration where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- Conditions for the circle configuration -/
def valid_configuration (c : CircleConfiguration) : Prop :=
  c.R = 18 ∧ c.r = 9 ∧ c.x > 0 ∧ c.x < c.R ∧
  (c.R - c.x)^2 - c.x^2 = (c.r + c.x)^2 - c.x^2

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius (c : CircleConfiguration) 
  (h : valid_configuration c) : c.x = 8 := by
  sorry


end inscribed_circle_radius_l3560_356009


namespace modulus_one_plus_i_to_sixth_l3560_356035

theorem modulus_one_plus_i_to_sixth (i : ℂ) : i * i = -1 → Complex.abs ((1 + i)^6) = 8 := by
  sorry

end modulus_one_plus_i_to_sixth_l3560_356035


namespace rectangular_floor_tiles_l3560_356024

theorem rectangular_floor_tiles (width : ℕ) (length : ℕ) (diagonal_tiles : ℕ) :
  (2 * width = 3 * length) →  -- length-to-width ratio is 3:2
  (diagonal_tiles * diagonal_tiles = 13 * width * width) →  -- diagonal covers whole number of tiles
  (2 * diagonal_tiles - 1 = 45) →  -- total tiles on both diagonals is 45
  (width * length = 245) :=  -- total tiles covering the floor
by sorry

end rectangular_floor_tiles_l3560_356024


namespace addition_subtraction_integers_l3560_356047

theorem addition_subtraction_integers : (1 + (-2) - 8 - (-9) : ℤ) = 0 := by
  sorry

end addition_subtraction_integers_l3560_356047


namespace expression_evaluation_l3560_356053

theorem expression_evaluation : (4 + 6 + 7) * 2 - 2 + 3 / 3 = 33 := by
  sorry

end expression_evaluation_l3560_356053


namespace cos_five_pi_sixth_minus_alpha_l3560_356038

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (5 * π / 6 - α) = -1 / 3 := by
  sorry

end cos_five_pi_sixth_minus_alpha_l3560_356038


namespace order_of_expressions_l3560_356036

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(1/2)
  let b : ℝ := Real.log 2015 / Real.log 2014
  let c : ℝ := Real.log 2 / Real.log 4
  b > a ∧ a > c := by sorry

end order_of_expressions_l3560_356036


namespace largest_angle_in_18_sided_polygon_l3560_356086

theorem largest_angle_in_18_sided_polygon (n : ℕ) (sum_other_angles : ℝ) :
  n = 18 ∧ sum_other_angles = 2754 →
  (n - 2) * 180 - sum_other_angles = 126 :=
by sorry

end largest_angle_in_18_sided_polygon_l3560_356086


namespace triangle_side_sum_l3560_356064

/-- Represents a triangle with side lengths a, b, and c, and angles A, B, and C. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if the angles of a triangle are in arithmetic progression -/
def anglesInArithmeticProgression (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.A + 2*d

/-- Represents a value that can be expressed as p + √q + √r where p, q, r are integers -/
structure SpecialValue where
  p : ℤ
  q : ℤ
  r : ℤ

/-- The theorem to be proved -/
theorem triangle_side_sum (t : Triangle) (x₁ x₂ : SpecialValue) :
  t.a = 6 ∧ t.b = 8 ∧
  anglesInArithmeticProgression t ∧
  t.A = 30 * π / 180 ∧
  (t.c = Real.sqrt (x₁.q : ℝ) ∨ t.c = (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ)) →
  (x₁.p : ℝ) + Real.sqrt (x₁.q : ℝ) + Real.sqrt (x₁.r : ℝ) +
  (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ) + Real.sqrt (x₂.r : ℝ) =
  7 + Real.sqrt 36 + Real.sqrt 83 :=
by sorry

end triangle_side_sum_l3560_356064


namespace tank_depth_l3560_356050

/-- Proves that a tank with given dimensions and plastering cost has a depth of 6 meters -/
theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) : 
  length = 25 → 
  width = 12 → 
  cost_per_sqm = 0.75 → 
  total_cost = 558 → 
  ∃ d : ℝ, d = 6 ∧ cost_per_sqm * (2 * (length * d) + 2 * (width * d) + (length * width)) = total_cost :=
by
  sorry

#check tank_depth

end tank_depth_l3560_356050


namespace A_empty_iff_a_in_range_l3560_356005

/-- The set of solutions to the quadratic equation ax^2 - ax + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 = 0}

/-- The theorem stating that A is empty if and only if a is in [0, 4) -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end A_empty_iff_a_in_range_l3560_356005


namespace rope_cutting_probability_rope_cutting_probability_proof_l3560_356078

/-- The probability of cutting a 1-meter rope at a random point such that the longer piece is at least three times as large as the shorter piece is 1/2. -/
theorem rope_cutting_probability : Real → Prop :=
  fun p => p = 1/2 ∧ 
    ∀ c : Real, 0 ≤ c ∧ c ≤ 1 →
      (((1 - c ≥ 3 * c) ∨ (c ≥ 3 * (1 - c))) ↔ (c ≤ 1/4 ∨ c ≥ 3/4)) ∧
      p = (1/4 - 0) + (1 - 3/4)

/-- Proof of the rope cutting probability theorem -/
theorem rope_cutting_probability_proof : ∃ p, rope_cutting_probability p :=
  sorry

end rope_cutting_probability_rope_cutting_probability_proof_l3560_356078


namespace average_of_three_numbers_l3560_356034

theorem average_of_three_numbers (M : ℝ) (h1 : 12 < M) (h2 : M < 25) : 
  ∃ k : ℝ, k = 5 ∧ (8 + 15 + (M + k)) / 3 = 18 :=
by
  sorry

end average_of_three_numbers_l3560_356034


namespace sum_of_cubes_and_cube_of_sum_l3560_356042

theorem sum_of_cubes_and_cube_of_sum : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end sum_of_cubes_and_cube_of_sum_l3560_356042


namespace geometric_sequence_third_term_l3560_356070

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSeq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_incr_geom : IsIncreasingGeometricSeq a)
  (h_sum : a 4 + a 6 = 6)
  (h_prod : a 2 * a 8 = 8) :
  a 3 = Real.sqrt 2 := by
sorry

end geometric_sequence_third_term_l3560_356070


namespace artistic_parents_l3560_356015

theorem artistic_parents (total : ℕ) (dad : ℕ) (mom : ℕ) (both : ℕ) : 
  total = 40 → dad = 18 → mom = 20 → both = 11 →
  total - (dad + mom - both) = 13 := by
sorry

end artistic_parents_l3560_356015


namespace sum_of_five_consecutive_even_integers_l3560_356017

theorem sum_of_five_consecutive_even_integers (a : ℤ) : 
  (a + (a + 4) = 150) → (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385) :=
by
  sorry

end sum_of_five_consecutive_even_integers_l3560_356017


namespace square_area_ratio_sqrt_l3560_356008

theorem square_area_ratio_sqrt (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  Real.sqrt ((side_c ^ 2) / (side_d ^ 2)) = 3 / 4 := by
  sorry

end square_area_ratio_sqrt_l3560_356008


namespace pen_cost_calculation_l3560_356041

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of 12 pens -/
theorem pen_cost_calculation (total_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) : 
  total_cost = 150 →
  3 * pen_cost + 5 * pencil_cost = total_cost →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 450 := by
sorry

end pen_cost_calculation_l3560_356041


namespace ram_money_l3560_356089

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Krishan's amount,
    calculate the amount of money Ram has. -/
theorem ram_money (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  krishan = 3468 →
  ram = 588 := by
sorry

end ram_money_l3560_356089


namespace solution_in_second_quadrant_l3560_356051

theorem solution_in_second_quadrant :
  ∃ (x y : ℝ), 
    (y = 2*x + 2) ∧ 
    (y = -x + 1) ∧ 
    (x < 0) ∧ 
    (y > 0) := by
  sorry

end solution_in_second_quadrant_l3560_356051


namespace number_of_possible_lists_l3560_356049

def number_of_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (number_of_balls ^ list_length : ℕ) = 50625 := by
  sorry

end number_of_possible_lists_l3560_356049


namespace hancho_drank_03L_l3560_356081

/-- The amount of milk Hancho drank -/
def hancho_consumption (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) : ℝ :=
  initial_amount - (yeseul_consumption + (yeseul_consumption + gayoung_extra) + remaining)

/-- Theorem stating that Hancho drank 0.3 L of milk given the initial conditions -/
theorem hancho_drank_03L (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) 
  (h1 : initial_amount = 1)
  (h2 : yeseul_consumption = 0.1)
  (h3 : gayoung_extra = 0.2)
  (h4 : remaining = 0.3) :
  hancho_consumption initial_amount yeseul_consumption gayoung_extra remaining = 0.3 := by
  sorry

end hancho_drank_03L_l3560_356081


namespace serving_size_is_six_ounces_l3560_356011

-- Define the given constants
def concentrate_cans : ℕ := 12
def water_cans_per_concentrate : ℕ := 4
def ounces_per_can : ℕ := 12
def total_servings : ℕ := 120

-- Define the theorem
theorem serving_size_is_six_ounces :
  let total_cans := concentrate_cans * (water_cans_per_concentrate + 1)
  let total_ounces := total_cans * ounces_per_can
  let serving_size := total_ounces / total_servings
  serving_size = 6 := by sorry

end serving_size_is_six_ounces_l3560_356011


namespace gangster_undetected_conditions_l3560_356098

/-- Configuration of streets and houses -/
structure StreetConfig where
  a : ℝ  -- Side length of houses
  street_distance : ℝ  -- Distance between parallel streets
  house_gap : ℝ  -- Distance between neighboring houses
  police_interval : ℝ  -- Interval between police officers

/-- Movement parameters -/
structure MovementParams where
  police_speed : ℝ  -- Speed of police officers
  gangster_speed : ℝ  -- Speed of the gangster
  gangster_direction : Bool  -- True if moving towards police, False otherwise

/-- Predicate to check if the gangster remains undetected -/
def remains_undetected (config : StreetConfig) (params : MovementParams) : Prop :=
  (params.gangster_direction = true) ∧ 
  ((params.gangster_speed = 2 * params.police_speed) ∨ 
   (params.gangster_speed = params.police_speed / 2))

/-- Main theorem: Conditions for the gangster to remain undetected -/
theorem gangster_undetected_conditions 
  (config : StreetConfig) 
  (params : MovementParams) :
  config.street_distance = 3 * config.a ∧ 
  config.house_gap = 2 * config.a ∧
  config.police_interval = 9 * config.a ∧
  params.police_speed > 0 →
  remains_undetected config params ↔ 
  (params.gangster_direction = true ∧ 
   (params.gangster_speed = 2 * params.police_speed ∨ 
    params.gangster_speed = params.police_speed / 2)) :=
by sorry

end gangster_undetected_conditions_l3560_356098


namespace cost_price_of_cloth_l3560_356039

/-- Represents the cost price of one metre of cloth -/
def costPricePerMetre (totalMetres : ℕ) (sellingPrice : ℕ) (profitPerMetre : ℕ) : ℕ :=
  (sellingPrice - profitPerMetre * totalMetres) / totalMetres

/-- Theorem stating that the cost price of one metre of cloth is 85 rupees -/
theorem cost_price_of_cloth :
  costPricePerMetre 85 8925 20 = 85 := by
  sorry

end cost_price_of_cloth_l3560_356039


namespace weekly_earnings_proof_l3560_356033

/-- Calculates the total earnings for a repair shop given the number of repairs and their costs. -/
def total_earnings (phone_repairs laptop_repairs computer_repairs : ℕ) 
  (phone_cost laptop_cost computer_cost : ℕ) : ℕ :=
  phone_repairs * phone_cost + laptop_repairs * laptop_cost + computer_repairs * computer_cost

/-- Theorem: The total earnings for the week is $121 given the specified repairs and costs. -/
theorem weekly_earnings_proof :
  total_earnings 5 2 2 11 15 18 = 121 := by
  sorry

end weekly_earnings_proof_l3560_356033


namespace parallelepiped_construction_impossible_l3560_356003

/-- Represents the five shapes of blocks -/
inductive BlockShape
  | I
  | L
  | T
  | Plus
  | J

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ

/-- Represents the construction requirements -/
structure ConstructionRequirements where
  total_blocks : ℕ
  shapes : List BlockShape
  volume : ℕ

/-- Checks if a parallelepiped satisfies the edge conditions -/
def valid_edges (p : Parallelepiped) : Prop :=
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1

/-- Checks if a parallelepiped can be constructed with given requirements -/
def can_construct (p : Parallelepiped) (req : ConstructionRequirements) : Prop :=
  p.volume = req.volume ∧ valid_edges p

/-- Main theorem: Impossibility of constructing the required parallelepiped -/
theorem parallelepiped_construction_impossible (req : ConstructionRequirements) :
  req.total_blocks = 48 ∧ 
  req.shapes = [BlockShape.I, BlockShape.L, BlockShape.T, BlockShape.Plus, BlockShape.J] ∧
  req.volume = 1990 →
  ¬∃ (p : Parallelepiped), can_construct p req :=
sorry

end parallelepiped_construction_impossible_l3560_356003


namespace non_intercept_line_conditions_l3560_356020

/-- A line that cannot be converted to intercept form -/
def NonInterceptLine (m : ℝ) : Prop :=
  ∃ (x y : ℝ), m * (x + y - 1) + (3 * y - 4 * x + 5) = 0 ∧
  ((m - 4 = 0) ∨ (m + 3 = 0) ∨ (-m + 5 = 0))

/-- The theorem stating the conditions for a line that cannot be converted to intercept form -/
theorem non_intercept_line_conditions :
  ∀ m : ℝ, NonInterceptLine m ↔ (m = 4 ∨ m = -3 ∨ m = 5) :=
by sorry

end non_intercept_line_conditions_l3560_356020


namespace segment_multiplication_l3560_356031

-- Define a segment as a pair of points
def Segment (α : Type*) := α × α

-- Define the length of a segment
def length {α : Type*} (s : Segment α) : ℝ := sorry

-- Define the multiplication of a segment by a scalar
def scaleSegment {α : Type*} (s : Segment α) (n : ℕ) : Segment α := sorry

-- Theorem statement
theorem segment_multiplication {α : Type*} (AB : Segment α) (n : ℕ) :
  ∃ (AC : Segment α), length AC = n * length AB :=
sorry

end segment_multiplication_l3560_356031


namespace unique_symmetric_shape_l3560_356082

-- Define a type for the shapes
inductive Shape : Type
  | A | B | C | D | E

-- Define a function to represent symmetry with respect to the vertical line
def isSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem unique_symmetric_shape :
  ∃! s : Shape, isSymmetric s :=
by
  sorry

end unique_symmetric_shape_l3560_356082


namespace sin_cos_pi_12_simplification_l3560_356012

theorem sin_cos_pi_12_simplification :
  1/2 * Real.sin (π/12) * Real.cos (π/12) = 1/8 := by
  sorry

end sin_cos_pi_12_simplification_l3560_356012


namespace max_b_value_l3560_356085

theorem max_b_value (a b c : ℕ) : 
  1 < c → c < b → b < a → a * b * c = 360 → b ≤ 12 :=
by sorry

end max_b_value_l3560_356085


namespace polynomial_expansion_l3560_356027

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 + 4 * z - 5
def q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the expanded result
def expanded_result (z : ℝ) : ℝ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- Theorem statement
theorem polynomial_expansion (z : ℝ) : p z * q z = expanded_result z := by
  sorry

end polynomial_expansion_l3560_356027


namespace smallest_stairs_l3560_356062

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 4 ∧ n % 7 = 3) → n ≥ 52 :=
by sorry

end smallest_stairs_l3560_356062


namespace trapezoid_semicircle_area_l3560_356054

-- Define the trapezoid
def trapezoid : Set (ℝ × ℝ) :=
  {p | p = (5, 11) ∨ p = (16, 11) ∨ p = (16, -2) ∨ p = (5, -2)}

-- Define the semicircle
def semicircle : Set (ℝ × ℝ) :=
  {p | (p.1 - 10.5)^2 + (p.2 + 2)^2 ≤ 5.5^2 ∧ p.2 ≤ -2}

-- Define the area to be calculated
def bounded_area : ℝ := sorry

-- Theorem statement
theorem trapezoid_semicircle_area :
  bounded_area = 15.125 * Real.pi := by sorry

end trapezoid_semicircle_area_l3560_356054


namespace trigonometric_equation_proof_l3560_356043

theorem trigonometric_equation_proof (α : ℝ) : 
  (Real.sin (2 * α) + Real.sin (5 * α) - Real.sin (3 * α)) / 
  (Real.cos α + 1 - 2 * (Real.sin (2 * α))^2) = 2 * Real.sin α := by
  sorry

end trigonometric_equation_proof_l3560_356043


namespace project_selection_count_l3560_356090

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of projects to be selected from each category -/
def projects_per_category : ℕ := 2

/-- Calculates the number of ways to select projects with the given conditions -/
def select_projects : ℕ :=
  Nat.choose num_key_projects projects_per_category *
  Nat.choose num_general_projects projects_per_category -
  Nat.choose (num_key_projects - 1) projects_per_category *
  Nat.choose (num_general_projects - 1) projects_per_category

theorem project_selection_count :
  select_projects = 60 := by sorry

end project_selection_count_l3560_356090


namespace second_third_smallest_average_l3560_356096

theorem second_third_smallest_average (a b c d e : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- five different positive integers
  (a + b + c + d + e : ℚ) / 5 = 5 ∧  -- average is 5
  ∀ x y z w v : ℕ+, x < y ∧ y < z ∧ z < w ∧ w < v → 
    (x + y + z + w + v : ℚ) / 5 = 5 → (v - x : ℚ) ≤ (e - a) →  -- difference is maximized
  (b + c : ℚ) / 2 = 5/2 :=  -- average of second and third smallest is 2.5
sorry

end second_third_smallest_average_l3560_356096


namespace line_circle_distance_sum_l3560_356057

-- Define the lines and circle
def line_l1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line_l2 (x y a : ℝ) : Prop := 4 * x - 2 * y + a = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the distance sum condition
def distance_sum_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, circle_C x y →
    (|2*x - y + 1| / Real.sqrt 5 + |4*x - 2*y + a| / Real.sqrt 20 = 2 * Real.sqrt 5)

-- Theorem statement
theorem line_circle_distance_sum (a : ℝ) :
  distance_sum_condition a → (a = 10 ∨ a = -18) :=
sorry

end line_circle_distance_sum_l3560_356057


namespace positive_integer_sum_with_square_is_thirty_l3560_356060

theorem positive_integer_sum_with_square_is_thirty (P : ℕ+) : P^2 + P = 30 → P = 5 := by
  sorry

end positive_integer_sum_with_square_is_thirty_l3560_356060


namespace point_coordinates_l3560_356091

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating that a point in the second quadrant with given distances from axes has specific coordinates -/
theorem point_coordinates (A : Point) 
    (h1 : secondQuadrant A) 
    (h2 : distanceFromXAxis A = 5) 
    (h3 : distanceFromYAxis A = 6) : 
  A.x = -6 ∧ A.y = 5 := by
  sorry


end point_coordinates_l3560_356091


namespace function_properties_l3560_356067

-- Define the function f(x) = ax³ + bx
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties (a b : ℝ) :
  -- Condition 1: Tangent line at x=3 is parallel to 24x - y + 1 = 0
  f' a b 3 = 24 →
  -- Condition 2: Function has an extremum at x=1
  f' a b 1 = 0 →
  -- Condition 3: a = 1
  a = 1 →
  -- Conclusion 1: f(x) = x³ - 3x
  (∀ x, f a b x = x^3 - 3*x) ∧
  -- Conclusion 2: Interval of monotonic decrease is [-1, 1]
  (∀ x, x ∈ Set.Icc (-1) 1 → f' a b x ≤ 0) ∧
  -- Conclusion 3: For f(x) to be decreasing on [-1, 1], b ≤ -3
  (∀ x, x ∈ Set.Icc (-1) 1 → f' 1 b x ≤ 0) → b ≤ -3 :=
by sorry


end function_properties_l3560_356067


namespace triangle_area_relation_l3560_356066

/-- Given a triangle T with area Δ, and two triangles T' and T'' formed by successive altitudes
    with areas Δ' and Δ'' respectively, prove that if Δ' = 30 and Δ'' = 20, then Δ = 45. -/
theorem triangle_area_relation (Δ Δ' Δ'' : ℝ) : Δ' = 30 → Δ'' = 20 → Δ = 45 :=
by sorry

end triangle_area_relation_l3560_356066


namespace expected_balls_in_original_position_l3560_356073

/-- Represents the number of balls arranged in a circle -/
def numBalls : ℕ := 6

/-- Represents the number of people performing swaps -/
def numSwaps : ℕ := 3

/-- Probability that a specific ball is not involved in a single swap -/
def probNotSwapped : ℚ := 4 / 6

/-- Probability that a ball remains in its original position after all swaps -/
def probInOriginalPosition : ℚ := probNotSwapped ^ numSwaps

/-- Expected number of balls in their original positions after all swaps -/
def expectedBallsInOriginalPosition : ℚ := numBalls * probInOriginalPosition

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_in_original_position :
  expectedBallsInOriginalPosition = 48 / 27 := by
  sorry

end expected_balls_in_original_position_l3560_356073


namespace square_sum_difference_equals_338_l3560_356058

theorem square_sum_difference_equals_338 :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 := by
  sorry

end square_sum_difference_equals_338_l3560_356058


namespace abs_value_sum_diff_l3560_356022

theorem abs_value_sum_diff (a b c : ℝ) : 
  (|a| = 1) → (|b| = 2) → (|c| = 3) → (a > b) → (b > c) → 
  (a + b - c = 2 ∨ a + b - c = 0) := by
sorry

end abs_value_sum_diff_l3560_356022


namespace rectangular_to_polar_conversion_l3560_356075

theorem rectangular_to_polar_conversion :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = -r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end rectangular_to_polar_conversion_l3560_356075


namespace parallel_vectors_magnitude_l3560_356000

def vector_a : Fin 2 → ℝ := ![1, -2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_magnitude (x : ℝ) :
  (∃ k : ℝ, vector_a = k • vector_b x) →
  Real.sqrt ((vector_a 0 - vector_b x 0)^2 + (vector_a 1 - vector_b x 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l3560_356000


namespace cubic_root_sum_product_l3560_356059

theorem cubic_root_sum_product (p q r : ℂ) : 
  (2 * p^3 - 4 * p^2 + 7 * p - 3 = 0) →
  (2 * q^3 - 4 * q^2 + 7 * q - 3 = 0) →
  (2 * r^3 - 4 * r^2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7/2 := by
  sorry

end cubic_root_sum_product_l3560_356059


namespace second_concert_proof_l3560_356074

/-- The attendance of the first concert -/
def first_concert_attendance : ℕ := 65899

/-- The additional attendance at the second concert -/
def additional_attendance : ℕ := 119

/-- The attendance of the second concert -/
def second_concert_attendance : ℕ := first_concert_attendance + additional_attendance

theorem second_concert_proof : second_concert_attendance = 66018 := by
  sorry

end second_concert_proof_l3560_356074


namespace quadratic_point_value_l3560_356092

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x ≤ 4)
  (h2 : quadratic_function a b c 2 = 4)
  (h3 : quadratic_function a b c 0 = -7) :
  quadratic_function a b c 5 = -83/4 := by
  sorry

end quadratic_point_value_l3560_356092


namespace remainder_problem_l3560_356061

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end remainder_problem_l3560_356061


namespace percentage_problem_l3560_356002

theorem percentage_problem (N : ℝ) : 
  (0.4 * N = 4/5 * 25 + 4) → N = 60 := by
  sorry

end percentage_problem_l3560_356002


namespace arctan_sum_of_roots_l3560_356028

theorem arctan_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  x₂^2 - x₂ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  Real.arctan x₁ + Real.arctan x₂ = π / 5 := by
sorry

end arctan_sum_of_roots_l3560_356028


namespace rectangle_perimeter_area_sum_l3560_356052

-- Define the coordinates of the rectangle
def vertex1 : ℤ × ℤ := (1, 2)
def vertex2 : ℤ × ℤ := (1, 6)
def vertex3 : ℤ × ℤ := (7, 6)
def vertex4 : ℤ × ℤ := (7, 2)

-- Define the function to calculate the perimeter and area sum
def perimeterAreaSum (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let width := (v3.1 - v1.1).natAbs
  let height := (v2.2 - v1.2).natAbs
  2 * (width + height) + width * height

-- Theorem statement
theorem rectangle_perimeter_area_sum :
  perimeterAreaSum vertex1 vertex2 vertex3 vertex4 = 44 := by
  sorry


end rectangle_perimeter_area_sum_l3560_356052


namespace segment_length_product_l3560_356077

theorem segment_length_product (a : ℝ) : 
  (∃ b : ℝ, b ≠ a ∧ 
   ((3 * a - 5)^2 + (2 * a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2) ∧
   ((3 * b - 5)^2 + (2 * b - 5 - (-2))^2 = (3 * Real.sqrt 13)^2)) →
  (a * b = -1080 / 169) :=
by sorry

end segment_length_product_l3560_356077


namespace fraction_to_decimal_l3560_356084

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l3560_356084


namespace sum_of_tenth_powers_l3560_356023

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end sum_of_tenth_powers_l3560_356023


namespace max_min_on_interval_l3560_356013

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0) ∧
  (f 2 = -1)

/-- Theorem stating the existence and values of maximum and minimum on [-6,6] -/
theorem max_min_on_interval (f : ℝ → ℝ) (h : f_properties f) :
  (∃ max_val : ℝ, IsGreatest {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} max_val ∧ max_val = 3) ∧
  (∃ min_val : ℝ, IsLeast {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} min_val ∧ min_val = -3) :=
sorry

end max_min_on_interval_l3560_356013


namespace arithmetic_geometric_sequence_solution_l3560_356045

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence_solution :
  ∀ x y z : ℝ,
  is_arithmetic_sequence x y z →
  x + y + z = -3 →
  is_geometric_sequence (x + y) (y + z) (z + x) →
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = -7 ∧ y = -1 ∧ z = 5)) :=
sorry

end arithmetic_geometric_sequence_solution_l3560_356045


namespace largest_four_digit_divisible_by_35_l3560_356056

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 35 = 0 → n ≤ 9985 :=
by sorry

end largest_four_digit_divisible_by_35_l3560_356056


namespace min_volume_to_prevent_explosion_l3560_356076

/-- Represents the relationship between pressure and volume for a balloon -/
structure Balloon where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  h1 : ∀ v, pressure v = k / v
  h2 : pressure 3 = 8000
  h3 : ∀ v, pressure v > 40000 → volume v < v

/-- The minimum volume to prevent the balloon from exploding is 0.6 m³ -/
theorem min_volume_to_prevent_explosion (b : Balloon) : 
  ∀ v, v ≥ 0.6 → b.pressure v ≤ 40000 :=
sorry

#check min_volume_to_prevent_explosion

end min_volume_to_prevent_explosion_l3560_356076


namespace repeating_decimal_4_8_equals_44_9_l3560_356046

/-- Represents a repeating decimal where the digit 8 repeats infinitely after the decimal point -/
def repeating_decimal_4_8 : ℚ := 4 + 8/9

theorem repeating_decimal_4_8_equals_44_9 : 
  repeating_decimal_4_8 = 44/9 := by sorry

end repeating_decimal_4_8_equals_44_9_l3560_356046


namespace exponent_multiplication_l3560_356001

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by sorry

end exponent_multiplication_l3560_356001
