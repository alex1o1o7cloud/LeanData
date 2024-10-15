import Mathlib

namespace NUMINAMATH_CALUDE_library_shelf_count_l3778_377820

theorem library_shelf_count (notebooks : ℕ) (pen_difference : ℕ) : 
  notebooks = 30 → pen_difference = 50 → notebooks + (notebooks + pen_difference) = 110 :=
by sorry

end NUMINAMATH_CALUDE_library_shelf_count_l3778_377820


namespace NUMINAMATH_CALUDE_tournament_teams_count_l3778_377870

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  n : ℕ  -- number of teams
  winner_points : ℕ
  last_place_points : ℕ
  winner_points_eq : winner_points = 26
  last_place_points_eq : last_place_points = 20

/-- Theorem stating that under the given conditions, the number of teams must be 12 -/
theorem tournament_teams_count (t : FootballTournament) : t.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_tournament_teams_count_l3778_377870


namespace NUMINAMATH_CALUDE_article_cost_price_l3778_377882

theorem article_cost_price (marked_price : ℝ) (cost_price : ℝ) : 
  marked_price = 112.5 →
  0.95 * marked_price = 1.25 * cost_price →
  cost_price = 85.5 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3778_377882


namespace NUMINAMATH_CALUDE_trumpet_cost_l3778_377821

/-- The cost of Mike's trumpet, given his total spending and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 584 / 100) : 
  total_spent - song_book_cost = 14516 / 100 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l3778_377821


namespace NUMINAMATH_CALUDE_twelve_times_minus_square_l3778_377856

theorem twelve_times_minus_square (x : ℕ) (h : x = 6) : 12 * x - x^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_twelve_times_minus_square_l3778_377856


namespace NUMINAMATH_CALUDE_investment_proof_l3778_377823

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let initial_investment : ℝ := 400
  let interest_rate : ℝ := 0.12
  let time_period : ℕ := 5
  let final_balance : ℝ := 705.03
  compound_interest initial_investment interest_rate time_period = final_balance :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l3778_377823


namespace NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l3778_377845

/-- Calculates the profit percentage for a mixture of butter sold at a certain price -/
theorem butter_mixture_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 34)
  (h2 : price1 = 150)
  (h3 : weight2 = 36)
  (h4 : price2 = 125)
  (h5 : selling_price = 192) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_price_per_kg := total_cost / total_weight
  let profit_percentage := (selling_price - cost_price_per_kg) / cost_price_per_kg * 100
  ∃ ε > 0, abs (profit_percentage - 40) < ε :=
by sorry


end NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l3778_377845


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3778_377877

theorem complex_power_modulus : Complex.abs ((2 : ℂ) + Complex.I * Real.sqrt 11) ^ 4 = 225 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3778_377877


namespace NUMINAMATH_CALUDE_four_special_numbers_exist_l3778_377838

theorem four_special_numbers_exist : ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (¬(2 ∣ a) ∧ ¬(3 ∣ a) ∧ ¬(4 ∣ a)) ∧
  (¬(2 ∣ b) ∧ ¬(3 ∣ b) ∧ ¬(4 ∣ b)) ∧
  (¬(2 ∣ c) ∧ ¬(3 ∣ c) ∧ ¬(4 ∣ c)) ∧
  (¬(2 ∣ d) ∧ ¬(3 ∣ d) ∧ ¬(4 ∣ d)) ∧
  (2 ∣ (a + b)) ∧ (2 ∣ (a + c)) ∧ (2 ∣ (a + d)) ∧
  (2 ∣ (b + c)) ∧ (2 ∣ (b + d)) ∧ (2 ∣ (c + d)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (b + c + d)) ∧
  (4 ∣ (a + b + c + d)) := by
  sorry

#check four_special_numbers_exist

end NUMINAMATH_CALUDE_four_special_numbers_exist_l3778_377838


namespace NUMINAMATH_CALUDE_circle_contains_three_points_l3778_377800

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : Real

/-- Theorem: Given 51 points randomly placed on a unit square, 
    there always exists a circle of radius 1/7 that contains at least 3 of these points -/
theorem circle_contains_three_points 
  (points : Finset Point) 
  (h_count : points.card = 51) 
  (h_in_square : ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1) :
  ∃ c : Circle, c.radius = 1/7 ∧ (points.filter (λ p => (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2)).card ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_circle_contains_three_points_l3778_377800


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l3778_377818

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)) =
  56 * (X : ℕ → ℕ) 3 * (Y : ℕ → ℕ) 5 + (Finset.range 9).sum (fun k => 
    if k ≠ 3 
    then Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)
    else 0) :=
by sorry

#check coefficient_x3y5_in_expansion_of_x_plus_y_8

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l3778_377818


namespace NUMINAMATH_CALUDE_train_journey_time_l3778_377825

theorem train_journey_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) :
  speed1 = 48 →
  speed2 = 60 →
  time2 = 2/3 →
  distance = speed1 * time1 →
  distance = speed2 * time2 →
  time1 = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_time_l3778_377825


namespace NUMINAMATH_CALUDE_multiple_of_1897_l3778_377859

theorem multiple_of_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_1897_l3778_377859


namespace NUMINAMATH_CALUDE_litter_collection_weight_l3778_377850

theorem litter_collection_weight 
  (gina_bags : ℕ) 
  (neighborhood_multiplier : ℕ) 
  (bag_weight : ℕ) 
  (h1 : gina_bags = 8)
  (h2 : neighborhood_multiplier = 120)
  (h3 : bag_weight = 6) : 
  (gina_bags + gina_bags * neighborhood_multiplier) * bag_weight = 5808 := by
  sorry

end NUMINAMATH_CALUDE_litter_collection_weight_l3778_377850


namespace NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l3778_377834

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_balls_4_boxes : distribute_balls 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l3778_377834


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3778_377854

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3778_377854


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3778_377874

theorem reciprocal_sum_pairs : 
  (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card = 9 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3778_377874


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_l3778_377855

/-- A rectangular prism is a three-dimensional shape with specific properties. -/
structure RectangularPrism where
  -- We don't need to define the specific properties here

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26 -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_l3778_377855


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3778_377852

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3778_377852


namespace NUMINAMATH_CALUDE_school_furniture_prices_l3778_377826

/-- The price of a table in yuan -/
def table_price : ℕ := 36

/-- The price of a chair in yuan -/
def chair_price : ℕ := 9

/-- The total cost of 2 tables and 3 chairs in yuan -/
def total_cost : ℕ := 99

theorem school_furniture_prices :
  (2 * table_price + 3 * chair_price = total_cost) ∧
  (table_price = 4 * chair_price) ∧
  (table_price = 36) ∧
  (chair_price = 9) := by
  sorry

end NUMINAMATH_CALUDE_school_furniture_prices_l3778_377826


namespace NUMINAMATH_CALUDE_difference_max_min_change_l3778_377814

def initial_yes : ℝ := 40
def initial_no : ℝ := 30
def initial_maybe : ℝ := 30
def final_yes : ℝ := 60
def final_no : ℝ := 20
def final_maybe : ℝ := 20

def min_change : ℝ := 20
def max_change : ℝ := 40

theorem difference_max_min_change :
  max_change - min_change = 20 :=
sorry

end NUMINAMATH_CALUDE_difference_max_min_change_l3778_377814


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3778_377842

theorem sum_remainder_mod_seven : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3778_377842


namespace NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l3778_377899

theorem alcohol_water_mixture_ratio 
  (p q r : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hr : r > 0) :
  let jar1_ratio := p / (p + 1)
  let jar2_ratio := q / (q + 1)
  let jar3_ratio := r / (r + 1)
  let total_alcohol := jar1_ratio + jar2_ratio + jar3_ratio
  let total_water := 1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1)
  total_alcohol / total_water = (p*q*r + p*q + p*r + q*r + p + q + r) / (p*q + p*r + q*r + p + q + r + 1) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l3778_377899


namespace NUMINAMATH_CALUDE_integral_circle_area_l3778_377824

theorem integral_circle_area (f : ℝ → ℝ) (a b r : ℝ) (h : ∀ x ∈ Set.Icc a b, f x = Real.sqrt (r^2 - x^2)) :
  (∫ x in a..b, f x) = (π * r^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_integral_circle_area_l3778_377824


namespace NUMINAMATH_CALUDE_rectangle_configuration_l3778_377844

/-- The side length of square S2 in the given rectangle configuration. -/
def side_length_S2 : ℕ := 1300

/-- The side length of squares S1 and S3 in the given rectangle configuration. -/
def side_length_S1_S3 : ℕ := side_length_S2 + 50

/-- The width of the entire rectangle. -/
def total_width : ℕ := 4000

/-- The height of the entire rectangle. -/
def total_height : ℕ := 2500

/-- The theorem stating that the given configuration satisfies all conditions. -/
theorem rectangle_configuration :
  side_length_S1_S3 + side_length_S2 + side_length_S1_S3 = total_width ∧
  ∃ (r : ℕ), 2 * r + side_length_S2 = total_height :=
by sorry

end NUMINAMATH_CALUDE_rectangle_configuration_l3778_377844


namespace NUMINAMATH_CALUDE_fourth_person_height_l3778_377830

def height_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  (h₁ + h₂ + h₃ + h₄) / 4 = 76

theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (h : height_problem h₁ h₂ h₃ h₄) : 
  h₄ = 82 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3778_377830


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l3778_377860

/-- A partnership business between A and B -/
structure Partnership where
  /-- A's investment as a multiple of B's investment -/
  a_investment_multiple : ℝ
  /-- B's profit -/
  b_profit : ℝ
  /-- Total profit -/
  total_profit : ℝ

/-- The ratio of A's investment to B's investment in the partnership -/
def investment_ratio (p : Partnership) : ℝ := p.a_investment_multiple

/-- Theorem stating the investment ratio in the given partnership scenario -/
theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.b_profit = 4000)
  (h2 : p.total_profit = 28000) : 
  investment_ratio p = 3 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l3778_377860


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l3778_377840

theorem smallest_x_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 3 = 2 ∧ 
  (x : ℤ) % 4 = 3 ∧ 
  (x : ℤ) % 5 = 4 ∧
  ∀ y : ℕ+, 
    (y : ℤ) % 3 = 2 → 
    (y : ℤ) % 4 = 3 → 
    (y : ℤ) % 5 = 4 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l3778_377840


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l3778_377810

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h1 : y₁ = quadratic_function 3)
  (h2 : y₂ = quadratic_function 1) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l3778_377810


namespace NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l3778_377894

/-- Parabola defined by the equation 8y = (x-3)^2 -/
def parabola (x y : ℝ) : Prop := 8 * y = (x - 3)^2

/-- Point M -/
def M : ℝ × ℝ := (0, -2)

/-- Tangent line equation -/
def is_tangent_line (m b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ 
    (∀ x y, y = m * x + b ↔ (x = x₀ ∧ y = y₀ ∨ (y - y₀) = (x - x₀) * ((x₀ - 3) / 4)))

/-- Theorem stating the properties of the tangent lines and the triangle -/
theorem parabola_tangents_and_triangle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent lines equations
    is_tangent_line (-2) (-2) ∧
    is_tangent_line (1/2) (-2) ∧
    -- Points A and B are on the parabola
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    -- A and B are on the tangent lines
    y₁ = -2 * x₁ - 2 ∧
    y₂ = 1/2 * x₂ - 2 ∧
    -- Tangent lines are perpendicular
    (-2) * (1/2) = -1 ∧
    -- Area of triangle ABM
    abs ((x₁ - 0) * (y₂ - (-2)) - (x₂ - 0) * (y₁ - (-2))) / 2 = 125/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l3778_377894


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3778_377880

theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, x^2 + a^2 + 2*x - 2*a - 2 = 0 ∧ 
   ∀ y : ℝ, y ≠ a → x^2 + y^2 + 2*x - 2*y - 2 > 0) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3778_377880


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3778_377864

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3778_377864


namespace NUMINAMATH_CALUDE_relay_race_distance_l3778_377890

/-- Proves that in a 5-member relay team where one member runs twice the distance of others,
    and the total race distance is 18 km, each of the other members runs 3 km. -/
theorem relay_race_distance (team_size : ℕ) (ralph_multiplier : ℕ) (total_distance : ℝ) :
  team_size = 5 →
  ralph_multiplier = 2 →
  total_distance = 18 →
  ∃ (other_distance : ℝ),
    other_distance = 3 ∧
    (team_size - 1) * other_distance + ralph_multiplier * other_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l3778_377890


namespace NUMINAMATH_CALUDE_equation_solution_l3778_377878

theorem equation_solution :
  ∀ y : ℝ, (5 + 3.2 * y = 2.1 * y - 25) ↔ (y = -300 / 11) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3778_377878


namespace NUMINAMATH_CALUDE_carpenter_woodblocks_needed_l3778_377836

/-- Calculates the total number of woodblocks needed by a carpenter to build a house. -/
theorem carpenter_woodblocks_needed 
  (initial_logs : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ) : 
  (initial_logs + additional_logs_needed) * woodblocks_per_log = 80 :=
by
  sorry

#check carpenter_woodblocks_needed 8 5 8

end NUMINAMATH_CALUDE_carpenter_woodblocks_needed_l3778_377836


namespace NUMINAMATH_CALUDE_square_product_inequality_l3778_377887

theorem square_product_inequality (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_inequality_l3778_377887


namespace NUMINAMATH_CALUDE_line_symmetry_l3778_377863

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := y = x + 2
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x_mid y_mid : ℝ), g x_mid y_mid ∧
    (x₂ - x_mid = x_mid - x₁) ∧ (y₂ - y_mid = y_mid - y₁)

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line2 symmetric_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l3778_377863


namespace NUMINAMATH_CALUDE_rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l3778_377862

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rhombus
  | Trapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Rhombus => True
  | _ => false

-- Theorem stating that only Rhombus is always symmetrical
theorem rhombus_always_symmetrical :
  ∀ (s : Shape), isSymmetrical s ↔ s = Shape.Rhombus :=
by sorry

-- Additional theorems to show that other shapes are not always symmetrical
theorem triangle_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Triangle ∧ ¬(isSymmetrical t) :=
by sorry

theorem parallelogram_not_always_symmetrical :
  ∃ (p : Shape), p = Shape.Parallelogram ∧ ¬(isSymmetrical p) :=
by sorry

theorem trapezoid_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Trapezoid ∧ ¬(isSymmetrical t) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l3778_377862


namespace NUMINAMATH_CALUDE_rhombus_area_l3778_377827

/-- Represents a rhombus with diagonals 2a and 2b, and an acute angle θ. -/
structure Rhombus where
  a : ℕ+
  b : ℕ+
  θ : ℝ
  acute_angle : 0 < θ ∧ θ < π / 2

/-- The area of a rhombus is 2ab, where a and b are half the lengths of its diagonals. -/
theorem rhombus_area (r : Rhombus) : Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) / 2 * Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) * Real.sin r.θ / 2 = 2 * r.a * r.b := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3778_377827


namespace NUMINAMATH_CALUDE_red_star_selection_probability_l3778_377833

/-- The probability of selecting a specific book from a set of books -/
def probability_of_selection (total_books : ℕ) (target_books : ℕ) : ℚ :=
  target_books / total_books

/-- Theorem: The probability of selecting "The Red Star Shines Over China" from 4 books is 1/4 -/
theorem red_star_selection_probability :
  probability_of_selection 4 1 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_red_star_selection_probability_l3778_377833


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3778_377876

/-- Given a hyperbola with equation x^2/4 - y^2/12 = -1, 
    prove that the equation of the ellipse with its vertices at the foci of the hyperbola 
    and its foci at the vertices of the hyperbola is x^2/4 + y^2/16 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (x' y' : ℝ), (x'^2 / 4 + y'^2 / 16 = 1 ∧ 
    (∀ (a b c : ℝ), (a > b ∧ b > 0 ∧ c > 0) → 
      (y'^2 / a^2 + x'^2 / b^2 = 1 ↔ 
        (a = 4 ∧ b^2 = 4 ∧ c = 2 * Real.sqrt 3)))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3778_377876


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l3778_377835

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l3778_377835


namespace NUMINAMATH_CALUDE_train_speed_l3778_377857

/-- Calculate the speed of a train given its length, platform length, and time to cross -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) : 
  train_length = 250 →
  platform_length = 520 →
  time = 50.395968322534195 →
  ∃ (speed : ℝ), abs (speed - 54.99) < 0.01 ∧ 
    speed = (train_length + platform_length) / time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3778_377857


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l3778_377807

theorem polynomial_factor_theorem (c : ℚ) : 
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 19 * x^2 - c * x - 49)) → c = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l3778_377807


namespace NUMINAMATH_CALUDE_syllogism_form_is_correct_l3778_377872

-- Define deductive reasoning
structure DeductiveReasoning where
  general_to_specific : Bool
  syllogism_form : Bool
  conclusion_correctness : Bool
  conclusion_depends_on_premises : Bool

-- Define the correct properties of deductive reasoning
def correct_deductive_reasoning : DeductiveReasoning :=
  { general_to_specific := true,
    syllogism_form := true,
    conclusion_correctness := false,
    conclusion_depends_on_premises := true }

-- Theorem to prove
theorem syllogism_form_is_correct (dr : DeductiveReasoning) :
  dr = correct_deductive_reasoning → dr.syllogism_form = true :=
by sorry

end NUMINAMATH_CALUDE_syllogism_form_is_correct_l3778_377872


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3778_377884

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 →
    1/(x' - 1) + 4/(y' - 1) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3778_377884


namespace NUMINAMATH_CALUDE_women_average_age_is_23_l3778_377851

/-- The average age of two women given the conditions of the problem -/
def average_age_of_women (initial_men_count : ℕ) 
                         (age_increase : ℕ) 
                         (replaced_man1_age : ℕ) 
                         (replaced_man2_age : ℕ) : ℚ :=
  let total_age_increase := initial_men_count * age_increase
  let total_women_age := total_age_increase + replaced_man1_age + replaced_man2_age
  total_women_age / 2

/-- Theorem stating that the average age of the women is 23 years -/
theorem women_average_age_is_23 : 
  average_age_of_women 8 2 20 10 = 23 := by
  sorry

end NUMINAMATH_CALUDE_women_average_age_is_23_l3778_377851


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l3778_377847

/-- Given a linear function y = (m² + 1)x + 2n where m and n are constants,
    and two points A(2a - 1, y₁) and B(a² + 1, y₂) on this function,
    prove that y₁ < y₂ -/
theorem y1_less_than_y2 (m n a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (m^2 + 1) * (2*a - 1) + 2*n) 
  (h2 : y₂ = (m^2 + 1) * (a^2 + 1) + 2*n) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l3778_377847


namespace NUMINAMATH_CALUDE_playground_length_l3778_377895

theorem playground_length (garden_width garden_perimeter playground_width : ℝ) 
  (hw : garden_width = 24)
  (hp : garden_perimeter = 64)
  (pw : playground_width = 12)
  (area_eq : garden_width * ((garden_perimeter / 2) - garden_width) = playground_width * (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width)) :
  (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width) = 16 := by
sorry

end NUMINAMATH_CALUDE_playground_length_l3778_377895


namespace NUMINAMATH_CALUDE_product_of_roots_l3778_377885

theorem product_of_roots (x : ℝ) : 
  let a : ℝ := 24
  let b : ℝ := 36
  let c : ℝ := -648
  let equation := a * x^2 + b * x + c
  let root_product := c / a
  equation = 0 → root_product = -27 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3778_377885


namespace NUMINAMATH_CALUDE_a_range_theorem_l3778_377805

-- Define the sequence a_n
def a_n (a n : ℝ) : ℝ := a * n^2 + n + 5

-- State the theorem
theorem a_range_theorem (a : ℝ) :
  (∀ n : ℕ, a_n a n < a_n a (n + 1) ∧ n ≤ 3) ∧
  (∀ n : ℕ, a_n a n > a_n a (n + 1) ∧ n ≥ 8) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l3778_377805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3778_377869

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (m : ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_roots : (a 2)^2 + m * (a 2) - 8 = 0 ∧ (a 8)^2 + m * (a 8) - 8 = 0)
  (h_sum : a 4 + a 6 = (a 5)^2 + 1) :
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3778_377869


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3778_377803

theorem trigonometric_inequality (a b α β : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ α ∧ α ≤ Real.pi / 2) 
  (h4 : 0 ≤ β ∧ β ≤ Real.pi / 2) 
  (h5 : a * b * Real.cos (α - β) ≤ Real.sqrt ((1 - a^2) * (1 - b^2))) :
  a * Real.cos α + b * Real.sin β ≤ 1 + a * b * Real.sin (β - α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3778_377803


namespace NUMINAMATH_CALUDE_smallest_common_multiple_lcm_14_10_smallest_number_of_students_l3778_377875

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ 14 ∣ n ∧ 10 ∣ n → n ≥ 70 := by
  sorry

theorem lcm_14_10 : Nat.lcm 14 10 = 70 := by
  sorry

theorem smallest_number_of_students : ∃ (n : ℕ), n > 0 ∧ 14 ∣ n ∧ 10 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 14 ∣ m ∧ 10 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_lcm_14_10_smallest_number_of_students_l3778_377875


namespace NUMINAMATH_CALUDE_sphere_volume_equals_area_l3778_377861

theorem sphere_volume_equals_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_area_l3778_377861


namespace NUMINAMATH_CALUDE_max_xy_value_l3778_377879

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 16 ∧ x₀ * y₀ = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3778_377879


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3778_377837

theorem root_sum_theorem (x : ℝ) (a b c d : ℝ) : 
  (1/x + 1/(x+4) - 1/(x+6) - 1/(x+10) + 1/(x+12) + 1/(x+16) - 1/(x+18) - 1/(x+20) = 0) →
  (∃ (sign1 sign2 : Bool), x = -a + (-1)^(sign1.toNat : ℕ) * Real.sqrt (b + (-1)^(sign2.toNat : ℕ) * c * Real.sqrt d)) →
  a + b + c + d = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3778_377837


namespace NUMINAMATH_CALUDE_theater_ticket_profit_l3778_377843

/-- Calculates the total profit from ticket sales given the ticket prices and quantities sold. -/
theorem theater_ticket_profit
  (adult_price : ℕ)
  (kid_price : ℕ)
  (total_tickets : ℕ)
  (kid_tickets : ℕ)
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : kid_tickets = 75) :
  (total_tickets - kid_tickets) * adult_price + kid_tickets * kid_price = 750 :=
by sorry


end NUMINAMATH_CALUDE_theater_ticket_profit_l3778_377843


namespace NUMINAMATH_CALUDE_sphere_radius_from_cylinder_l3778_377848

/-- The radius of a sphere formed by recasting a cylindrical iron block -/
theorem sphere_radius_from_cylinder (cylinder_radius : ℝ) (cylinder_height : ℝ) (sphere_radius : ℝ) : 
  cylinder_radius = 2 →
  cylinder_height = 9 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  sphere_radius = 3 := by
  sorry

#check sphere_radius_from_cylinder

end NUMINAMATH_CALUDE_sphere_radius_from_cylinder_l3778_377848


namespace NUMINAMATH_CALUDE_factorization_proof_l3778_377893

theorem factorization_proof (x y : ℝ) : 2 * x^3 - 18 * x * y^2 = 2 * x * (x + 3 * y) * (x - 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3778_377893


namespace NUMINAMATH_CALUDE_circle_elimination_count_l3778_377898

/-- Calculates the total number of counts in a circle elimination game. -/
def totalCounts (initialPeople : ℕ) : ℕ :=
  let rec countRounds (remaining : ℕ) (acc : ℕ) : ℕ :=
    if remaining ≤ 2 then acc
    else
      let eliminated := remaining / 3
      let newRemaining := remaining - eliminated
      countRounds newRemaining (acc + remaining)
  countRounds initialPeople 0

/-- Theorem stating that for 21 initial people, the total count is 64. -/
theorem circle_elimination_count :
  totalCounts 21 = 64 := by
  sorry

end NUMINAMATH_CALUDE_circle_elimination_count_l3778_377898


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3778_377804

/-- The x-intercept of the line 5y - 7x = 35 is (-5, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  5 * y - 7 * x = 35 → y = 0 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3778_377804


namespace NUMINAMATH_CALUDE_like_terms_power_l3778_377816

theorem like_terms_power (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-1) * y^2 = -2 * x^2 * y^n) → 
  (-m : ℤ)^n = 9 := by
sorry

end NUMINAMATH_CALUDE_like_terms_power_l3778_377816


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3778_377829

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (coffee : ℕ)
  (tea : ℕ)
  (both : ℕ)
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3778_377829


namespace NUMINAMATH_CALUDE_f_sum_negative_l3778_377873

def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

theorem f_sum_negative (a b c : ℝ) 
  (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) : 
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_negative_l3778_377873


namespace NUMINAMATH_CALUDE_candy_distribution_l3778_377867

theorem candy_distribution (total_candy : ℕ) (pieces_per_student : ℕ) (num_students : ℕ) :
  total_candy = 344 →
  pieces_per_student = 8 →
  total_candy = num_students * pieces_per_student →
  num_students = 43 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3778_377867


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3778_377813

theorem sum_of_coefficients : ∃ (a b d c : ℕ+), 
  (∀ (a' b' d' c' : ℕ+), 
    (a' * Real.sqrt 3 + b' * Real.sqrt 11 + d' * Real.sqrt 2) / c' = 
    Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 →
    c ≤ c') ∧
  (a * Real.sqrt 3 + b * Real.sqrt 11 + d * Real.sqrt 2) / c = 
    Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 ∧
  a + b + d + c = 325 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3778_377813


namespace NUMINAMATH_CALUDE_square_division_perimeter_l3778_377888

theorem square_division_perimeter 
  (original_perimeter : ℝ) 
  (h_original_perimeter : original_perimeter = 200) : 
  ∃ (smaller_square_perimeter : ℝ), 
    smaller_square_perimeter = 100 ∧
    ∃ (original_side : ℝ), 
      4 * original_side = original_perimeter ∧
      ∃ (rectangle_width rectangle_height : ℝ),
        rectangle_width = original_side ∧
        rectangle_height = original_side / 2 ∧
        smaller_square_perimeter = 4 * rectangle_height :=
by sorry

end NUMINAMATH_CALUDE_square_division_perimeter_l3778_377888


namespace NUMINAMATH_CALUDE_burger_orders_l3778_377889

theorem burger_orders (total : ℕ) (burger_ratio : ℕ) : 
  total = 45 → burger_ratio = 2 → 
  ∃ (hotdog : ℕ), 
    hotdog + burger_ratio * hotdog = total ∧
    burger_ratio * hotdog = 30 := by
  sorry

end NUMINAMATH_CALUDE_burger_orders_l3778_377889


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3778_377828

theorem perfect_square_condition (x : ℝ) :
  (∃ a : ℤ, 4 * x^5 - 7 = a^2) ∧ 
  (∃ b : ℤ, 4 * x^13 - 7 = b^2) → 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3778_377828


namespace NUMINAMATH_CALUDE_apples_left_over_l3778_377817

theorem apples_left_over (greg_sarah_apples susan_apples mark_apples : ℕ) : 
  greg_sarah_apples = 18 →
  susan_apples = 2 * (greg_sarah_apples / 2) →
  mark_apples = susan_apples - 5 →
  (greg_sarah_apples + susan_apples + mark_apples) - 40 = 9 :=
by sorry

end NUMINAMATH_CALUDE_apples_left_over_l3778_377817


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3778_377871

/-- Given positive integers A, B, C, and integer D, where A, B, C form an arithmetic sequence,
    B, C, D form a geometric sequence, and C/B = 7/3, the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℤ, C - B = B - A) →  -- arithmetic sequence condition
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- geometric sequence condition
  C = (7 * B) / 3 →
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ r : ℤ, C' - B' = B' - A') →
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) →
    C' = (7 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3778_377871


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3778_377809

/-- The perimeter of a semicircle with radius 3.1 cm is equal to π * 3.1 + 6.2 cm. -/
theorem semicircle_perimeter :
  let r : Real := 3.1
  let perimeter := π * r + 2 * r
  perimeter = π * 3.1 + 6.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3778_377809


namespace NUMINAMATH_CALUDE_tank_problem_solution_l3778_377812

def tank_problem (initial_capacity : ℝ) (initial_loss_rate : ℝ) (initial_loss_time : ℝ)
  (second_loss_time : ℝ) (fill_rate : ℝ) (fill_time : ℝ) (final_missing : ℝ)
  (second_loss_rate : ℝ) : Prop :=
  let remaining_after_first_loss := initial_capacity - initial_loss_rate * initial_loss_time
  let remaining_after_second_loss := remaining_after_first_loss - second_loss_rate * second_loss_time
  let final_amount := remaining_after_second_loss + fill_rate * fill_time
  final_amount = initial_capacity - final_missing

theorem tank_problem_solution :
  tank_problem 350000 32000 5 10 40000 3 140000 10000 := by
  sorry

end NUMINAMATH_CALUDE_tank_problem_solution_l3778_377812


namespace NUMINAMATH_CALUDE_ratio_proof_l3778_377811

theorem ratio_proof (x y z : ℚ) :
  (5 * x + 4 * y - 6 * z) / (4 * x - 5 * y + 7 * z) = 1 / 27 ∧
  (5 * x + 4 * y - 6 * z) / (6 * x + 5 * y - 4 * z) = 1 / 18 →
  ∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_ratio_proof_l3778_377811


namespace NUMINAMATH_CALUDE_fraction_simplification_and_result_l3778_377849

theorem fraction_simplification_and_result (a : ℤ) (h : a = 2018) : 
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧ 
  2 * a + 1 = 4037 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_result_l3778_377849


namespace NUMINAMATH_CALUDE_algebraic_identities_l3778_377841

theorem algebraic_identities :
  -- Part 1
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 ∧
  -- Part 2
  (Real.sqrt 5 - Real.sqrt 6)^2 - (Real.sqrt 5 + Real.sqrt 6)^2 = -4 * Real.sqrt 30 ∧
  -- Part 3
  (2 * Real.sqrt (3/2) - Real.sqrt (1/2)) * (1/2 * Real.sqrt 8 + Real.sqrt (2/3)) = 5/3 * Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3778_377841


namespace NUMINAMATH_CALUDE_terms_before_one_l3778_377839

/-- An arithmetic sequence with first term 100 and common difference -3 -/
def arithmeticSequence : ℕ → ℤ := λ n => 100 - 3 * (n - 1)

/-- The position of 1 in the sequence -/
def positionOfOne : ℕ := 34

theorem terms_before_one :
  (∀ k < positionOfOne, arithmeticSequence k > 1) ∧
  arithmeticSequence positionOfOne = 1 ∧
  positionOfOne - 1 = 33 := by sorry

end NUMINAMATH_CALUDE_terms_before_one_l3778_377839


namespace NUMINAMATH_CALUDE_penny_halfDollar_same_probability_l3778_377822

/-- Represents the outcome of a single coin flip -/
inductive CoinSide
| Heads
| Tails

/-- Represents the outcome of flipping six different coins -/
structure SixCoinFlip :=
  (penny : CoinSide)
  (nickel : CoinSide)
  (dime : CoinSide)
  (quarter : CoinSide)
  (halfDollar : CoinSide)
  (dollar : CoinSide)

/-- The set of all possible outcomes when flipping six coins -/
def allOutcomes : Finset SixCoinFlip := sorry

/-- The set of outcomes where the penny and half-dollar show the same side -/
def sameOutcomes : Finset SixCoinFlip := sorry

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
def probability (event : Finset SixCoinFlip) : Rat :=
  (event.card : Rat) / (allOutcomes.card : Rat)

theorem penny_halfDollar_same_probability :
  probability sameOutcomes = 1/2 := by sorry

end NUMINAMATH_CALUDE_penny_halfDollar_same_probability_l3778_377822


namespace NUMINAMATH_CALUDE_technicians_in_exchange_group_and_expectation_l3778_377808

/-- Represents the distribution of job certificates --/
structure JobCertificates where
  junior : Nat
  intermediate : Nat
  senior : Nat
  technician : Nat
  seniorTechnician : Nat

/-- The total number of apprentices --/
def totalApprentices : Nat := 200

/-- The distribution of job certificates --/
def certificateDistribution : JobCertificates :=
  { junior := 20
  , intermediate := 60
  , senior := 60
  , technician := 40
  , seniorTechnician := 20 }

/-- The number of people selected for the exchange group --/
def exchangeGroupSize : Nat := 10

/-- The number of people chosen as representatives to speak --/
def speakersSize : Nat := 3

/-- Theorem stating the number of technicians in the exchange group and the expected number of technicians among speakers --/
theorem technicians_in_exchange_group_and_expectation :
  let totalTechnicians := certificateDistribution.technician + certificateDistribution.seniorTechnician
  let techniciansInExchangeGroup := (totalTechnicians * exchangeGroupSize) / totalApprentices
  let expectationOfTechnicians : Rat := 9 / 10
  techniciansInExchangeGroup = 3 ∧ 
  expectationOfTechnicians = (0 * (7 / 24 : Rat) + 1 * (21 / 40 : Rat) + 2 * (7 / 40 : Rat) + 3 * (1 / 120 : Rat)) := by
  sorry

end NUMINAMATH_CALUDE_technicians_in_exchange_group_and_expectation_l3778_377808


namespace NUMINAMATH_CALUDE_beth_class_size_l3778_377846

/-- The number of students in Beth's class over three years -/
def final_class_size (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final class size given the initial conditions -/
theorem beth_class_size :
  final_class_size 150 30 15 = 165 := by
  sorry

end NUMINAMATH_CALUDE_beth_class_size_l3778_377846


namespace NUMINAMATH_CALUDE_parabola_translation_l3778_377883

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola --/
def original : Parabola := { a := -2, h := -2, k := 3 }

/-- The translated parabola --/
def translated : Parabola := { a := -2, h := 1, k := -1 }

/-- The translation that moves the original parabola to the translated parabola --/
def translation : Translation := { dx := 3, dy := -4 }

theorem parabola_translation : 
  ∀ (x y : ℝ), 
  (y = -2 * (x - translated.h)^2 + translated.k) ↔ 
  (y + translation.dy = -2 * ((x - translation.dx) - original.h)^2 + original.k) :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l3778_377883


namespace NUMINAMATH_CALUDE_system_solution_l3778_377896

-- Define the system of equations
def system_equations (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

-- Theorem statement
theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), system_equations a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧ 
    x₁ = 1 / |a₁ - a₄| ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / |a₁ - a₄| :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3778_377896


namespace NUMINAMATH_CALUDE_total_value_of_coins_l3778_377868

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a half dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of quarters found -/
def num_quarters : ℕ := 14

/-- The number of dimes found -/
def num_dimes : ℕ := 7

/-- The number of nickels found -/
def num_nickels : ℕ := 9

/-- The number of pennies found -/
def num_pennies : ℕ := 13

/-- The number of half dollars found -/
def num_half_dollars : ℕ := 4

/-- The total value of the coins found -/
theorem total_value_of_coins : 
  (num_quarters : ℚ) * quarter_value + 
  (num_dimes : ℚ) * dime_value + 
  (num_nickels : ℚ) * nickel_value + 
  (num_pennies : ℚ) * penny_value + 
  (num_half_dollars : ℚ) * half_dollar_value = 6.78 := by sorry

end NUMINAMATH_CALUDE_total_value_of_coins_l3778_377868


namespace NUMINAMATH_CALUDE_max_odd_digits_on_board_l3778_377806

/-- A function that counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 digits -/
def hasTenDigits (n : ℕ) : Prop := sorry

theorem max_odd_digits_on_board (a b : ℕ) (h1 : hasTenDigits a) (h2 : hasTenDigits b) :
  countOddDigits a + countOddDigits b + countOddDigits (a + b) ≤ 30 ∧
  ∃ (a' b' : ℕ), hasTenDigits a' ∧ hasTenDigits b' ∧
    countOddDigits a' + countOddDigits b' + countOddDigits (a' + b') = 30 :=
sorry

end NUMINAMATH_CALUDE_max_odd_digits_on_board_l3778_377806


namespace NUMINAMATH_CALUDE_calculate_original_nes_price_l3778_377886

/-- Calculates the original price of an NES given trade-in values, discounts, and final payment -/
theorem calculate_original_nes_price
  (snes_value : ℝ)
  (snes_credit_rate : ℝ)
  (gameboy_value : ℝ)
  (gameboy_credit_rate : ℝ)
  (ps2_value : ℝ)
  (ps2_credit_rate : ℝ)
  (nes_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (payment : ℝ)
  (change : ℝ)
  (h1 : snes_value = 150)
  (h2 : snes_credit_rate = 0.8)
  (h3 : gameboy_value = 50)
  (h4 : gameboy_credit_rate = 0.75)
  (h5 : ps2_value = 100)
  (h6 : ps2_credit_rate = 0.6)
  (h7 : nes_discount_rate = 0.2)
  (h8 : sales_tax_rate = 0.08)
  (h9 : payment = 100)
  (h10 : change = 12) :
  ∃ (original_price : ℝ), abs (original_price - 101.85) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_calculate_original_nes_price_l3778_377886


namespace NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l3778_377853

def z1 (m : ℝ) : ℂ := Complex.mk (m^2 + m + 1) (m^2 + m - 4)
def z2 : ℂ := Complex.mk 3 (-2)

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, z1 m = z2 ∧ m ≠ 1) ∧ (z1 1 = z2) := by sorry

end NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l3778_377853


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3778_377831

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    and eccentricity √3, its asymptotes have the equation y = ±√2x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt 3
  let c := e * a
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3778_377831


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l3778_377819

theorem sum_greater_than_product (a b : ℕ+) : a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l3778_377819


namespace NUMINAMATH_CALUDE_combined_monthly_profit_is_90_l3778_377801

/-- Represents a book with its purchase price, sale price, and months held before sale -/
structure Book where
  purchase_price : ℕ
  sale_price : ℕ
  months_held : ℕ

/-- Calculates the monthly profit for a single book -/
def monthly_profit (book : Book) : ℚ :=
  (book.sale_price - book.purchase_price : ℚ) / book.months_held

/-- Calculates the combined monthly rate of profit for a list of books -/
def combined_monthly_profit (books : List Book) : ℚ :=
  books.map monthly_profit |>.sum

theorem combined_monthly_profit_is_90 (books : List Book) : combined_monthly_profit books = 90 :=
  by
  have h1 : books = [
    { purchase_price := 50, sale_price := 90, months_held := 1 },
    { purchase_price := 120, sale_price := 150, months_held := 2 },
    { purchase_price := 75, sale_price := 110, months_held := 0 }
  ] := by sorry
  rw [h1]
  simp [combined_monthly_profit, monthly_profit]
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_combined_monthly_profit_is_90_l3778_377801


namespace NUMINAMATH_CALUDE_travel_time_A_l3778_377815

/-- The time it takes for A to travel 60 miles given the conditions -/
theorem travel_time_A (y : ℝ) 
  (h1 : y > 0) -- B's speed is positive
  (h2 : (60 / y) - (60 / (y + 2)) = 3/4) -- Time difference equation
  : 60 / (y + 2) = 30/7 := by
sorry

end NUMINAMATH_CALUDE_travel_time_A_l3778_377815


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3778_377897

theorem sufficient_not_necessary_condition (A B : Set α) 
  (h1 : A ∩ B = A) (h2 : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3778_377897


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l3778_377892

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_W : W > 0) :
  (1.16 * L) * (W * (1 - x / 100)) = 1.102 * (L * W) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l3778_377892


namespace NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l3778_377865

/-- Proves that the selling price yielding the same profit as the loss is 54,
    given the cost price and a known selling price that results in a loss. -/
theorem selling_price_equal_profit_loss
  (cost_price : ℝ)
  (loss_price : ℝ)
  (h1 : cost_price = 47)
  (h2 : loss_price = 40)
  : ∃ (selling_price : ℝ),
    selling_price - cost_price = cost_price - loss_price ∧
    selling_price = 54 :=
by
  sorry

#check selling_price_equal_profit_loss

end NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l3778_377865


namespace NUMINAMATH_CALUDE_arrangement_schemes_l3778_377802

theorem arrangement_schemes (teachers students : ℕ) (h1 : teachers = 2) (h2 : students = 4) : 
  (teachers.choose 1) * (students.choose 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_schemes_l3778_377802


namespace NUMINAMATH_CALUDE_square_of_nines_l3778_377891

theorem square_of_nines (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nines_l3778_377891


namespace NUMINAMATH_CALUDE_percentage_votes_against_l3778_377858

/-- Given a total number of votes and the difference between votes in favor and against,
    calculate the percentage of votes against the proposal. -/
theorem percentage_votes_against (total_votes : ℕ) (favor_minus_against : ℕ) 
    (h1 : total_votes = 340)
    (h2 : favor_minus_against = 68) : 
    (total_votes - favor_minus_against) / 2 / total_votes * 100 = 40 := by
  sorry

#check percentage_votes_against

end NUMINAMATH_CALUDE_percentage_votes_against_l3778_377858


namespace NUMINAMATH_CALUDE_distance_between_points_l3778_377866

/-- The distance between two points in 3D space is the square root of the sum of the squares of the differences of their coordinates. -/
theorem distance_between_points (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2) = Real.sqrt 185 ↔
  x₁ = -2 ∧ y₁ = 4 ∧ z₁ = 1 ∧ x₂ = 3 ∧ y₂ = -8 ∧ z₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3778_377866


namespace NUMINAMATH_CALUDE_least_bench_sections_thirteen_is_least_l3778_377832

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 5 * M = 13 * M → M ≥ 13 := by
  sorry

theorem thirteen_is_least : ∃ M : ℕ, M > 0 ∧ 5 * M = 13 * M ∧ M = 13 := by
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_thirteen_is_least_l3778_377832


namespace NUMINAMATH_CALUDE_coconut_jelly_beans_count_l3778_377881

def total_jelly_beans : ℕ := 4000
def red_fraction : ℚ := 3/4
def coconut_fraction : ℚ := 1/4

theorem coconut_jelly_beans_count : 
  (red_fraction * total_jelly_beans : ℚ) * coconut_fraction = 750 := by
  sorry

end NUMINAMATH_CALUDE_coconut_jelly_beans_count_l3778_377881
