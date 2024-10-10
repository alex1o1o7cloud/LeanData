import Mathlib

namespace five_lines_sixteen_sections_l162_16239

/-- The number of sections created by drawing n line segments through a rectangle,
    assuming each new line intersects all previous lines. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else max_sections (n - 1) + n

/-- The theorem stating that 5 line segments create 16 sections in a rectangle. -/
theorem five_lines_sixteen_sections :
  max_sections 5 = 16 := by
  sorry

end five_lines_sixteen_sections_l162_16239


namespace fraction_of_seats_sold_l162_16293

/-- Proves that the fraction of seats sold is 0.75 given the auditorium layout and earnings --/
theorem fraction_of_seats_sold (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℚ) (total_earnings : ℚ) :
  rows = 20 →
  seats_per_row = 10 →
  ticket_price = 10 →
  total_earnings = 1500 →
  (total_earnings / ticket_price) / (rows * seats_per_row : ℚ) = 0.75 := by
  sorry

end fraction_of_seats_sold_l162_16293


namespace truck_speed_truck_speed_is_52_l162_16235

/-- The speed of the truck given two cars with different speeds meeting it at different times --/
theorem truck_speed (speed_A speed_B : ℝ) (time_A time_B : ℝ) : ℝ :=
  let distance_A := speed_A * time_A
  let distance_B := speed_B * time_B
  (distance_A - distance_B) / (time_B - time_A)

/-- Proof that the truck's speed is 52 km/h given the problem conditions --/
theorem truck_speed_is_52 :
  truck_speed 102 80 6 7 = 52 := by
  sorry

end truck_speed_truck_speed_is_52_l162_16235


namespace decimal_to_binary_13_l162_16271

theorem decimal_to_binary_13 : (13 : ℕ) = 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end decimal_to_binary_13_l162_16271


namespace line_plane_relationships_l162_16278

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (lies_on : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (on_different_planes : Line → Line → Prop)

-- Define the theorem
theorem line_plane_relationships 
  (l a : Line) (α : Plane)
  (h1 : parallel_line_plane l α)
  (h2 : lies_on a α) :
  perpendicular l a ∨ parallel_lines l a ∨ on_different_planes l a :=
sorry

end line_plane_relationships_l162_16278


namespace min_total_distance_l162_16246

/-- The number of trees planted -/
def num_trees : ℕ := 20

/-- The distance between adjacent trees in meters -/
def tree_distance : ℕ := 10

/-- The function that calculates the total distance traveled for a given tree position -/
def total_distance (n : ℕ) : ℕ :=
  10 * n^2 - 210 * n + 2100

/-- The theorem stating that the minimum total distance is 2000 meters -/
theorem min_total_distance :
  ∃ (n : ℕ), n > 0 ∧ n ≤ num_trees ∧ total_distance n = 2000 ∧
  ∀ (m : ℕ), m > 0 → m ≤ num_trees → total_distance m ≥ 2000 :=
sorry

end min_total_distance_l162_16246


namespace tina_total_time_l162_16275

def assignment_time : ℕ := 15
def total_sticky_keys : ℕ := 25
def time_per_key : ℕ := 5
def cleaned_keys : ℕ := 1

def remaining_keys : ℕ := total_sticky_keys - cleaned_keys
def cleaning_time : ℕ := remaining_keys * time_per_key
def total_time : ℕ := cleaning_time + assignment_time

theorem tina_total_time : total_time = 135 := by
  sorry

end tina_total_time_l162_16275


namespace exponential_function_property_l162_16222

theorem exponential_function_property (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  f (a + b) = 2 → f (2 * a) * f (2 * b) = 4 := by
  sorry

end exponential_function_property_l162_16222


namespace initial_bacteria_count_l162_16266

-- Define the doubling interval in seconds
def doubling_interval : ℕ := 30

-- Define the total time of the experiment in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 262144

-- Define the function to calculate the number of bacteria after a given time
def bacteria_count (initial : ℕ) (time : ℕ) : ℕ :=
  initial * (2 ^ (time / doubling_interval))

-- Theorem statement
theorem initial_bacteria_count :
  ∃ initial : ℕ, bacteria_count initial total_time = final_bacteria ∧ initial = 1024 :=
sorry

end initial_bacteria_count_l162_16266


namespace imaginary_part_of_z_l162_16226

theorem imaginary_part_of_z : Complex.im ((1 - Complex.I) / Complex.I) = -1 := by
  sorry

end imaginary_part_of_z_l162_16226


namespace lcm_problem_l162_16216

theorem lcm_problem (A B : ℕ+) (h1 : A * B = 45276) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2058 := by
  sorry

end lcm_problem_l162_16216


namespace S_intersect_T_l162_16202

def S : Set ℝ := {x | (x + 5) / (5 - x) > 0}
def T : Set ℝ := {x | x^2 + 4*x - 21 < 0}

theorem S_intersect_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end S_intersect_T_l162_16202


namespace solution_implies_m_value_l162_16242

theorem solution_implies_m_value (x m : ℝ) :
  x = 1 → 2 * x + m - 6 = 0 → m = 4 := by
  sorry

end solution_implies_m_value_l162_16242


namespace bird_watching_ratio_l162_16295

/-- Given the conditions of Camille's bird watching, prove the ratio of robins to cardinals -/
theorem bird_watching_ratio :
  ∀ (cardinals blue_jays sparrows robins : ℕ),
    cardinals = 3 →
    blue_jays = 2 * cardinals →
    sparrows = 3 * cardinals + 1 →
    cardinals + blue_jays + sparrows + robins = 31 →
    robins / cardinals = 4 := by
  sorry

#check bird_watching_ratio

end bird_watching_ratio_l162_16295


namespace log_sum_equality_l162_16298

theorem log_sum_equality : 
  2 * Real.log 9 / Real.log 10 + 3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 3 / Real.log 10 + 5 * Real.log 2 / Real.log 10 + 
  Real.log 16 / Real.log 10 = Real.log 215233856 / Real.log 10 := by
  sorry

end log_sum_equality_l162_16298


namespace train_speed_l162_16213

/-- The average speed of a train without stoppages, given its speed with stoppages and stop time -/
theorem train_speed (speed_with_stops : ℝ) (stop_time : ℝ) : 
  speed_with_stops = 200 → stop_time = 20 → 
  (speed_with_stops * 60) / (60 - stop_time) = 300 := by
  sorry

end train_speed_l162_16213


namespace min_irrational_root_distance_l162_16223

theorem min_irrational_root_distance (a b c : ℕ+) (h_a : a ≤ 10) :
  let f := fun x : ℝ => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  let roots := {x : ℝ | f x = 0}
  let distance := fun (x y : ℝ) => |x - y|
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ Irrational (distance x y)) →
  (∀ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y → distance x y ≥ Real.sqrt 13 / 9) :=
by sorry

end min_irrational_root_distance_l162_16223


namespace interest_rate_for_doubling_l162_16290

/-- Represents the number of years required for the principal to double. -/
def years_to_double : ℝ := 10

/-- Theorem stating that if a principal doubles in 10 years due to simple interest,
    then the rate of interest is 10% per annum. -/
theorem interest_rate_for_doubling (P : ℝ) (P_pos : P > 0) :
  ∃ R : ℝ, R > 0 ∧ P + (P * R * years_to_double / 100) = 2 * P ∧ R = 10 := by
sorry

end interest_rate_for_doubling_l162_16290


namespace factorization_cubic_minus_linear_l162_16287

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by sorry

end factorization_cubic_minus_linear_l162_16287


namespace absolute_value_implication_l162_16259

theorem absolute_value_implication (x : ℝ) : |x - 1| < 2 → x < 3 := by
  sorry

end absolute_value_implication_l162_16259


namespace cannot_construct_configuration_l162_16204

/-- Represents a rhombus figure with two colors -/
structure ColoredRhombus where
  white_part : Set (ℝ × ℝ)
  gray_part : Set (ℝ × ℝ)
  is_rhombus : white_part ∪ gray_part = unit_rhombus
  no_overlap : white_part ∩ gray_part = ∅

/-- Represents a configuration of multiple rhombuses -/
def Configuration := Set (ColoredRhombus × (ℝ × ℝ))

/-- Rotates a point around the origin -/
def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translates a point -/
def translate (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Applies rotation and translation to a ColoredRhombus -/
def transform (r : ColoredRhombus) (θ : ℝ) (v : ℝ × ℝ) : ColoredRhombus := sorry

/-- Checks if a configuration can be constructed from a given rhombus -/
def is_constructible (r : ColoredRhombus) (c : Configuration) : Prop := sorry

/-- The specific configuration that we claim is impossible to construct -/
def impossible_configuration : Configuration := sorry

/-- The main theorem stating that the impossible configuration cannot be constructed -/
theorem cannot_construct_configuration (r : ColoredRhombus) : 
  ¬(is_constructible r impossible_configuration) := by sorry

end cannot_construct_configuration_l162_16204


namespace geometric_sequence_ratio_l162_16264

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arith : a 3 = 2 * (2 * a 1) - a 2) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
sorry

end geometric_sequence_ratio_l162_16264


namespace evaluate_expression_l162_16248

theorem evaluate_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end evaluate_expression_l162_16248


namespace cage_cost_calculation_l162_16256

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_calculation : total_cost - cat_toy_cost = 11.73 := by
  sorry

end cage_cost_calculation_l162_16256


namespace circle_diameter_from_area_l162_16237

/-- Given a circle with area 25π m², prove its diameter is 10 m. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 25 * π →
    2 * r = 10 :=
by
  sorry

end circle_diameter_from_area_l162_16237


namespace extreme_values_and_three_roots_l162_16291

/-- The function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_three_roots 
  (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∃ y₁ y₂ y₃, f a b c y₁ = 2*c ∧ f a b c y₂ = 2*c ∧ f a b c y₃ = 2*c ∧ y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃) →
  (f' a b 1 = 0 ∧ f' a b (-2/3) = 0) →
  (a = -1/2 ∧ b = -2 ∧ 1/2 ≤ c ∧ c < 22/27) :=
by sorry

end extreme_values_and_three_roots_l162_16291


namespace spherical_ball_radius_l162_16262

/-- Given a cylindrical tub and a spherical iron ball, this theorem proves the radius of the ball
    based on the water level rise in the tub. -/
theorem spherical_ball_radius
  (tub_radius : ℝ)
  (water_rise : ℝ)
  (ball_radius : ℝ)
  (h1 : tub_radius = 12)
  (h2 : water_rise = 6.75)
  (h3 : (4 / 3) * Real.pi * ball_radius ^ 3 = Real.pi * tub_radius ^ 2 * water_rise) :
  ball_radius = 9 := by
  sorry

#check spherical_ball_radius

end spherical_ball_radius_l162_16262


namespace rectangle_area_theorem_l162_16289

/-- A rectangle in a 2D coordinate system --/
structure Rectangle where
  x1 : ℝ
  x2 : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: If a rectangle with vertices (-8, y), (1, y), (1, -7), and (-8, -7) has an area of 72, then y = 1 --/
theorem rectangle_area_theorem (y : ℝ) :
  let r := Rectangle.mk (-8) 1 y (-7)
  r.area = 72 → y = 1 := by
  sorry


end rectangle_area_theorem_l162_16289


namespace palace_visitors_l162_16283

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- Theorem stating the total number of visitors over two days -/
theorem palace_visitors : total_visitors 583 246 = 829 := by
  sorry

end palace_visitors_l162_16283


namespace a_necessary_for_c_l162_16206

theorem a_necessary_for_c (A B C : Prop) 
  (h1 : ¬A ↔ ¬B) (h2 : ¬B → ¬C) : C → A := by
  sorry

end a_necessary_for_c_l162_16206


namespace smallest_n_for_inequality_l162_16208

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 5 → 2^n > n^2 ∧ ∀ k : ℕ, k < 5 → 2^k ≤ k^2 := by
  sorry

end smallest_n_for_inequality_l162_16208


namespace tile_perimeter_increase_l162_16229

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration --/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { tiles := initial.tiles + added, perimeter := initial.perimeter + 3 }

/-- The theorem to be proved --/
theorem tile_perimeter_increase :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 10 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 3 ∧
    final.perimeter = 19 := by
  sorry

end tile_perimeter_increase_l162_16229


namespace total_cloud_count_l162_16263

def cloud_count (carson_funny : ℕ) (brother_multiplier : ℕ) (sister_divisor : ℕ) (cousin_multiplier : ℕ) : ℕ :=
  let brother_dragons := carson_funny * brother_multiplier
  let sister_sailboats := carson_funny / sister_divisor
  let cousin_birds := cousin_multiplier * (carson_funny + sister_sailboats)
  carson_funny + brother_dragons + sister_sailboats + cousin_birds

theorem total_cloud_count :
  cloud_count 12 5 2 2 = 114 := by
  sorry

end total_cloud_count_l162_16263


namespace bird_count_l162_16225

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 26) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 13 := by
  sorry

end bird_count_l162_16225


namespace division_problem_l162_16211

theorem division_problem (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end division_problem_l162_16211


namespace equality_from_quadratic_equation_l162_16274

theorem equality_from_quadratic_equation 
  (m n p : ℝ) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) 
  (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 
  2 * p = m + n := by
  sorry

end equality_from_quadratic_equation_l162_16274


namespace ratio_product_l162_16240

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 8 := by
  sorry

end ratio_product_l162_16240


namespace fourth_year_area_l162_16230

def initial_area : ℝ := 10000
def increase_rate : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + increase_rate) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 := by
  sorry

end fourth_year_area_l162_16230


namespace total_savings_l162_16252

/-- The total savings over two months given the savings in September and the difference in October -/
theorem total_savings (september : ℕ) (difference : ℕ) : 
  september = 260 → difference = 30 → september + (september + difference) = 550 := by
  sorry

end total_savings_l162_16252


namespace triangle_problem_l162_16296

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  Real.cos B = 7/9 →
  a * c * Real.cos B = 7 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 ∧ Real.sin (A - B) = 10 * Real.sqrt 2 / 27 := by
  sorry

end triangle_problem_l162_16296


namespace evaluate_expression_l162_16234

theorem evaluate_expression : (10010 - 12 * 3) * 2 ^ 3 = 79792 := by
  sorry

end evaluate_expression_l162_16234


namespace count_distinguishable_triangles_l162_16272

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents a large equilateral triangle constructed from four smaller triangles -/
structure LargeTriangle where
  corner1 : Fin num_colors
  corner2 : Fin num_colors
  corner3 : Fin num_colors
  center : Fin num_colors

/-- Two large triangles are considered equivalent if they can be matched by rotations or reflections -/
def equivalent (t1 t2 : LargeTriangle) : Prop :=
  ∃ (perm : Fin 3 → Fin 3), 
    (t1.corner1 = t2.corner1 ∧ t1.corner2 = t2.corner2 ∧ t1.corner3 = t2.corner3) ∨
    (t1.corner1 = t2.corner2 ∧ t1.corner2 = t2.corner3 ∧ t1.corner3 = t2.corner1) ∨
    (t1.corner1 = t2.corner3 ∧ t1.corner2 = t2.corner1 ∧ t1.corner3 = t2.corner2)

/-- The set of all distinguishable large triangles -/
def distinguishable_triangles : Finset LargeTriangle :=
  sorry

theorem count_distinguishable_triangles : 
  Finset.card distinguishable_triangles = 960 := by
  sorry

end count_distinguishable_triangles_l162_16272


namespace jills_gifts_and_charity_l162_16203

/-- Calculates the amount Jill uses for gifts and charitable causes --/
def gifts_and_charity (net_salary : ℚ) : ℚ :=
  let discretionary_income := (1 / 5) * net_salary
  let vacation_fund := (30 / 100) * discretionary_income
  let savings := (20 / 100) * discretionary_income
  let eating_out := (35 / 100) * discretionary_income
  discretionary_income - (vacation_fund + savings + eating_out)

/-- Theorem stating that Jill uses $99 for gifts and charitable causes --/
theorem jills_gifts_and_charity :
  gifts_and_charity 3300 = 99 := by
  sorry

end jills_gifts_and_charity_l162_16203


namespace parametric_curve_extrema_l162_16281

open Real

theorem parametric_curve_extrema :
  let x : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * cos t
  let y : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * sin t
  let t_domain := {t : ℝ | 0 ≤ t ∧ t ≤ 2 * π}
  (∀ t ∈ t_domain, x t ≤ 4) ∧
  (∃ t ∈ t_domain, x t = 4) ∧
  (∀ t ∈ t_domain, x t ≥ -1/2) ∧
  (∃ t ∈ t_domain, x t = -1/2) :=
by sorry

end parametric_curve_extrema_l162_16281


namespace g_of_5_l162_16284

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem g_of_5 : g 5 = 17 / 3 := by
  sorry

end g_of_5_l162_16284


namespace dog_walking_distance_l162_16245

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 ∧ dog1_daily_miles = 2 →
  dog2_daily_miles = 8 := by
sorry

end dog_walking_distance_l162_16245


namespace certain_value_is_one_l162_16251

theorem certain_value_is_one (w x : ℝ) (h1 : 13 = 13 * w / x) (h2 : w^2 = 1) : x = 1 := by
  sorry

end certain_value_is_one_l162_16251


namespace x_values_in_A_l162_16282

def A (x : ℝ) : Set ℝ := {-3, x + 2, x^2 - 4*x}

theorem x_values_in_A (x : ℝ) : 5 ∈ A x ↔ x = -1 ∨ x = 5 := by
  sorry

end x_values_in_A_l162_16282


namespace power_equation_solution_l162_16233

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end power_equation_solution_l162_16233


namespace complex_equation_sum_l162_16299

theorem complex_equation_sum (a b : ℝ) :
  (3 + b * I) / (1 - I) = a + b * I → a + b = 3 :=
by sorry

end complex_equation_sum_l162_16299


namespace radian_measure_of_negative_120_degrees_l162_16212

theorem radian_measure_of_negative_120_degrees :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian (-120) = -(2 * π / 3) := by sorry

end radian_measure_of_negative_120_degrees_l162_16212


namespace max_value_inequality_max_value_achieved_l162_16270

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_value_achieved : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end max_value_inequality_max_value_achieved_l162_16270


namespace dish_sets_budget_l162_16220

theorem dish_sets_budget (total_budget : ℕ) (sets_at_20 : ℕ) (price_per_set : ℕ) :
  total_budget = 6800 →
  sets_at_20 = 178 →
  price_per_set = 20 →
  total_budget - (sets_at_20 * price_per_set) = 3240 :=
by sorry

end dish_sets_budget_l162_16220


namespace reggie_has_70_marbles_l162_16231

/-- Calculates the number of marbles Reggie has after playing a series of games -/
def reggies_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) : ℕ :=
  (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game

/-- Proves that Reggie has 70 marbles after playing 9 games, losing 1, with 10 marbles bet per game -/
theorem reggie_has_70_marbles :
  reggies_marbles 9 1 10 = 70 := by
  sorry

#eval reggies_marbles 9 1 10

end reggie_has_70_marbles_l162_16231


namespace mixture_problem_l162_16200

theorem mixture_problem (initial_ratio_A B : ℚ) (drawn_off filled_B : ℚ) (final_ratio_A B : ℚ) :
  initial_ratio_A = 7 →
  initial_ratio_B = 5 →
  drawn_off = 9 →
  filled_B = 9 →
  final_ratio_A = 7 →
  final_ratio_B = 9 →
  ∃ x : ℚ,
    let initial_A := initial_ratio_A * x
    let initial_B := initial_ratio_B * x
    let removed_A := (initial_ratio_A / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let removed_B := (initial_ratio_B / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let remaining_A := initial_A - removed_A
    let remaining_B := initial_B - removed_B + filled_B
    remaining_A / remaining_B = final_ratio_A / final_ratio_B ∧
    initial_A = 23.625 :=
by sorry

end mixture_problem_l162_16200


namespace parabola_single_intersection_l162_16292

/-- A parabola with equation y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 + 2*x + k = 0) ↔ k = 1 := by sorry

end parabola_single_intersection_l162_16292


namespace function_symmetry_l162_16210

/-- For any function f(x) = x^5 - ax^3 + bx + 2, f(x) + f(-x) = 4 for all real x -/
theorem function_symmetry (a b : ℝ) :
  let f := fun (x : ℝ) => x^5 - a*x^3 + b*x + 2
  ∀ x, f x + f (-x) = 4 := by sorry

end function_symmetry_l162_16210


namespace quadratic_inequality_solution_set_l162_16286

theorem quadratic_inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m > 0} = {x : ℝ | x < m ∨ x > m + 1} := by
  sorry

end quadratic_inequality_solution_set_l162_16286


namespace decimal_representation_5_11_l162_16253

/-- The decimal representation of 5/11 has a repeating sequence of length 2 -/
def repeating_length : ℕ := 2

/-- The 150th decimal place in the representation of 5/11 -/
def decimal_place : ℕ := 150

/-- The result we want to prove -/
def result : ℕ := 5

theorem decimal_representation_5_11 :
  (decimal_place % repeating_length = 0) ∧
  (result = 5) := by
  sorry

end decimal_representation_5_11_l162_16253


namespace inverse_proportion_problem_l162_16207

/-- Two numbers are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  (∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ inversely_proportional x₀ y₀) →
  (x = -10 → y = -67.5) :=
by sorry

end inverse_proportion_problem_l162_16207


namespace tan_alpha_2_implies_expression_3_l162_16276

theorem tan_alpha_2_implies_expression_3 (α : Real) (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 - (Real.cos α) * (Real.sin α) + 1 = 3 := by
  sorry

end tan_alpha_2_implies_expression_3_l162_16276


namespace fraction_change_l162_16255

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 
  (2*a) * (2*b) / (2*(2*a) + 2*b) = 2 * (a * b / (2*a + b)) :=
sorry

end fraction_change_l162_16255


namespace special_triangle_f_measure_l162_16267

/-- A triangle with two equal angles and the third angle 20 degrees less than the others. -/
structure SpecialTriangle where
  /-- Angle D in degrees -/
  angleD : ℝ
  /-- Angle E in degrees -/
  angleE : ℝ
  /-- Angle F in degrees -/
  angleF : ℝ
  /-- Sum of angles in the triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180
  /-- Angles D and E are equal -/
  d_eq_e : angleD = angleE
  /-- Angle F is 20 degrees less than angle D -/
  f_less_20 : angleF = angleD - 20

theorem special_triangle_f_measure (t : SpecialTriangle) : t.angleF = 40 := by
  sorry

end special_triangle_f_measure_l162_16267


namespace equation_solution_l162_16294

theorem equation_solution (x : ℝ) (h : x * (x - 1) ≠ 0) :
  (x / (x - 1) - 2 / x = 1) ↔ (x = 2) :=
by sorry

end equation_solution_l162_16294


namespace max_product_sum_300_l162_16257

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end max_product_sum_300_l162_16257


namespace max_value_theorem_l162_16214

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * c * Real.sqrt 2 + 2 * a * b ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
  a'^2 + b'^2 + c'^2 = 1 ∧ 
  2 * a' * c' * Real.sqrt 2 + 2 * a' * b' = Real.sqrt 3 :=
sorry

end max_value_theorem_l162_16214


namespace kevin_savings_exceeds_ten_l162_16224

def kevin_savings (n : ℕ) : ℚ :=
  2 * (3^n - 1) / (3 - 1)

theorem kevin_savings_exceeds_ten :
  ∃ n : ℕ, kevin_savings n > 1000 ∧ ∀ m : ℕ, m < n → kevin_savings m ≤ 1000 :=
by sorry

end kevin_savings_exceeds_ten_l162_16224


namespace game_outcome_probability_l162_16247

/-- Represents the probability of a specific outcome in a game with 8 rounds and 3 players. -/
def game_probability (p_alex p_mel p_chelsea : ℝ) : Prop :=
  p_alex = 1/2 ∧
  p_mel = 2 * p_chelsea ∧
  p_alex + p_mel + p_chelsea = 1 ∧
  0 ≤ p_alex ∧ p_alex ≤ 1 ∧
  0 ≤ p_mel ∧ p_mel ≤ 1 ∧
  0 ≤ p_chelsea ∧ p_chelsea ≤ 1

/-- The probability of a specific outcome in the game. -/
def outcome_probability (p_alex p_mel p_chelsea : ℝ) : ℝ :=
  (p_alex ^ 4) * (p_mel ^ 3) * p_chelsea

/-- The number of ways to arrange 4 wins for Alex, 3 for Mel, and 1 for Chelsea in 8 rounds. -/
def arrangements : ℕ := 
  Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating the probability of the specific game outcome. -/
theorem game_outcome_probability :
  ∀ p_alex p_mel p_chelsea : ℝ,
  game_probability p_alex p_mel p_chelsea →
  (arrangements : ℝ) * outcome_probability p_alex p_mel p_chelsea = 35/324 :=
by
  sorry


end game_outcome_probability_l162_16247


namespace projectile_trajectory_area_l162_16209

theorem projectile_trajectory_area (v₀ g : ℝ) (h₁ : v₀ > 0) (h₂ : g > 0) :
  let v := fun t => v₀ + t * v₀  -- v varies from v₀ to 2v₀
  let x := fun t => (v t)^2 / (2 * g)
  let y := fun t => (v t)^2 / (4 * g)
  let area := ∫ t in (0)..(1), y (v t) * (x (v 1) - x (v 0))
  area = 3 * v₀^4 / (8 * g^2) :=
by sorry

end projectile_trajectory_area_l162_16209


namespace coefficient_of_x_squared_l162_16243

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^4) + 3 * (2*x - 3*x^2 + 4*x^3) - 2 * (2*x^4 - 3*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), ∀ x, expression x = a*x^4 + b*x^3 + 2*x^2 + d*x + e :=
by sorry

end coefficient_of_x_squared_l162_16243


namespace not_divisible_seven_digit_numbers_l162_16205

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def uses_digits_1_to_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 7 → ∃ k : ℕ, n / (10^k) % 10 = d

theorem not_divisible_seven_digit_numbers (A B : ℕ) :
  is_seven_digit A ∧ is_seven_digit B ∧
  uses_digits_1_to_7 A ∧ uses_digits_1_to_7 B ∧
  A ≠ B →
  ¬(∃ k : ℕ, A = k * B) :=
sorry

end not_divisible_seven_digit_numbers_l162_16205


namespace isosceles_triangle_perimeter_l162_16227

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end isosceles_triangle_perimeter_l162_16227


namespace perpendicular_vectors_x_value_l162_16215

/-- Given two perpendicular vectors a and b in ℝ², where a = (3, x) and b = (y, 1),
    prove that x = -7/4 -/
theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3, x]
  let b : Fin 2 → ℝ := ![y, 1]
  (∀ i j, a i * b j = 0) → x = -7/4 := by
  sorry

end perpendicular_vectors_x_value_l162_16215


namespace sin_15_times_sin_75_l162_16219

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end sin_15_times_sin_75_l162_16219


namespace negation_of_existence_negation_of_proposition_l162_16241

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 4^n) ↔ (∀ n : ℕ, n^2 ≤ 4^n) :=
by sorry

end negation_of_existence_negation_of_proposition_l162_16241


namespace max_candies_in_25_days_l162_16201

/-- Represents the dentist's instructions for candy consumption --/
structure CandyRules :=
  (max_daily : ℕ)
  (threshold : ℕ)
  (reduced_max : ℕ)
  (reduced_days : ℕ)

/-- Calculates the maximum number of candies that can be eaten in a given number of days --/
def max_candies (rules : CandyRules) (days : ℕ) : ℕ :=
  sorry

/-- The dentist's specific instructions --/
def dentist_rules : CandyRules :=
  { max_daily := 10
  , threshold := 7
  , reduced_max := 5
  , reduced_days := 2 }

/-- Theorem stating the maximum number of candies Sonia can eat in 25 days --/
theorem max_candies_in_25_days :
  max_candies dentist_rules 25 = 178 :=
sorry

end max_candies_in_25_days_l162_16201


namespace book_count_l162_16288

theorem book_count : ∃ (B : ℕ), 
  B > 0 ∧
  (2 * B) % 5 = 0 ∧  -- Two-fifths of books are reading books
  (3 * B) % 10 = 0 ∧ -- Three-tenths of books are math books
  (B * 3) / 10 - 1 = (B * 3) / 10 - (B * 3) % 10 / 10 - 1 ∧ -- Science books are one fewer than math books
  ((2 * B) / 5 + (3 * B) / 10 + ((3 * B) / 10 - 1) + 1 = B) ∧ -- Sum of all book types equals total
  B = 10 := by
  sorry

end book_count_l162_16288


namespace square_coverage_l162_16258

theorem square_coverage (k n : ℕ) : k > 1 → (k ^ 2 = 2 ^ (n + 1) * n + 1) → (k = 7 ∧ n = 3) := by
  sorry

end square_coverage_l162_16258


namespace muffin_apples_count_l162_16279

def initial_apples : ℕ := 62
def refrigerated_apples : ℕ := 25

def apples_for_muffins : ℕ :=
  initial_apples - (initial_apples / 2 + refrigerated_apples)

theorem muffin_apples_count :
  apples_for_muffins = 6 :=
by sorry

end muffin_apples_count_l162_16279


namespace no_solution_trigonometric_equation_l162_16218

open Real

theorem no_solution_trigonometric_equation (m : ℝ) :
  ¬ ∃ x : ℝ, (sin (3 * x) * cos (π / 3 - x) + 1) / (sin (π / 3 - 7 * x) - cos (π / 6 + x) + m) = 0 :=
by sorry

end no_solution_trigonometric_equation_l162_16218


namespace special_geometric_sequence_ratio_l162_16273

/-- A geometric sequence with positive terms where a₁, (1/2)a₃, 2a₂ form an arithmetic sequence -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  arithmetic : a 1 + 2 * a 2 = a 3

/-- The ratio of (a₁₁ + a₁₂) to (a₉ + a₁₀) equals 3 + 2√2 -/
theorem special_geometric_sequence_ratio 
  (seq : SpecialGeometricSequence) :
  (seq.a 11 + seq.a 12) / (seq.a 9 + seq.a 10) = 3 + 2 * Real.sqrt 2 := by
  sorry

end special_geometric_sequence_ratio_l162_16273


namespace negation_of_universal_proposition_l162_16254

theorem negation_of_universal_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ ∀ x : ℝ, x < 0 → a^x > 1) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) := by
  sorry

end negation_of_universal_proposition_l162_16254


namespace negation_of_existence_negation_of_squared_positive_l162_16269

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) :=
by sorry

end negation_of_existence_negation_of_squared_positive_l162_16269


namespace prob_blue_or_green_with_replacement_l162_16236

def total_balls : ℕ := 15
def blue_balls : ℕ := 8
def green_balls : ℕ := 2

def prob_blue : ℚ := blue_balls / total_balls
def prob_green : ℚ := green_balls / total_balls

def prob_two_blue : ℚ := prob_blue * prob_blue
def prob_two_green : ℚ := prob_green * prob_green

theorem prob_blue_or_green_with_replacement :
  prob_two_blue + prob_two_green = 68 / 225 := by
  sorry

end prob_blue_or_green_with_replacement_l162_16236


namespace inequality_preservation_l162_16277

theorem inequality_preservation (m n : ℝ) (h : m > n) : 2 + m > 2 + n := by
  sorry

end inequality_preservation_l162_16277


namespace hyperbola_eccentricity_l162_16217

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Left focus of a hyperbola -/
def left_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_min : ∀ p : HyperbolaPoint h, 
    (distance (p.x, p.y) (right_focus h))^2 / distance (p.x, p.y) (left_focus h) ≥ 9 * h.a) :
  eccentricity h = 5 := by sorry

end hyperbola_eccentricity_l162_16217


namespace gcd_3375_9180_l162_16249

theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry

end gcd_3375_9180_l162_16249


namespace train_passing_jogger_time_l162_16221

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6) -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6) -- Convert 45 km/hr to m/s
  (h3 : train_length = 100)
  (h4 : initial_distance = 240) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 34 := by
sorry

end train_passing_jogger_time_l162_16221


namespace lydias_current_age_l162_16238

/-- Represents the time it takes for an apple tree to bear fruit -/
def apple_tree_fruit_time : ℕ := 7

/-- Represents Lydia's age when she planted the tree -/
def planting_age : ℕ := 4

/-- Represents Lydia's age when she can eat an apple from her tree for the first time -/
def first_apple_age : ℕ := 11

/-- Represents Lydia's current age -/
def current_age : ℕ := 11

theorem lydias_current_age :
  current_age = first_apple_age ∧
  current_age = planting_age + apple_tree_fruit_time :=
by sorry

end lydias_current_age_l162_16238


namespace shopping_expense_calculation_l162_16285

theorem shopping_expense_calculation (T : ℝ) (x : ℝ) 
  (h1 : 0 < T) 
  (h2 : 0.5 * T + 0.2 * T + x * T = T) 
  (h3 : 0.04 * 0.5 * T + 0 * 0.2 * T + 0.08 * x * T = 0.044 * T) : 
  x = 0.3 := by
sorry

end shopping_expense_calculation_l162_16285


namespace map_distance_theorem_l162_16261

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 700

/-- Represents the length of a line on the map in inches -/
def map_line_length : ℝ := 5.5

/-- Calculates the actual distance represented by a line on the map -/
def actual_distance (scale : ℝ) (map_length : ℝ) : ℝ :=
  scale * map_length

/-- Proves that a 5.5-inch line on a map with a scale of 1 inch = 700 feet 
    represents 3850 feet in reality -/
theorem map_distance_theorem : 
  actual_distance map_scale map_line_length = 3850 := by
  sorry

end map_distance_theorem_l162_16261


namespace fraction_zero_implies_x_equals_three_l162_16260

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (3 - |x|) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end fraction_zero_implies_x_equals_three_l162_16260


namespace last_triangle_perimeter_l162_16250

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triangle in the sequence based on the incircle tangency points --/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) --/
def isValidTriangle (t : Triangle) : Bool := sorry

/-- The sequence of triangles starting from T₁ --/
def triangleSequence : List Triangle := sorry

/-- The last valid triangle in the sequence --/
def lastValidTriangle : Triangle := sorry

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℚ := sorry

theorem last_triangle_perimeter :
  let t₁ : Triangle := { a := 2011, b := 2012, c := 2013 }
  perimeter (lastValidTriangle) = 1509 / 128 := by sorry

end last_triangle_perimeter_l162_16250


namespace deepak_age_l162_16280

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 50 →
  deepak_age = 33 := by
sorry

end deepak_age_l162_16280


namespace number_problem_l162_16265

theorem number_problem (x : ℚ) : (3 * x / 2 + 6 = 11) → x = 10/3 := by
  sorry

end number_problem_l162_16265


namespace sin_plus_cos_for_point_l162_16297

/-- Given that the terminal side of angle θ passes through point P(-3,4),
    prove that sin θ + cos θ = 1/5 -/
theorem sin_plus_cos_for_point (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) →
  Real.sin θ + Real.cos θ = 1/5 := by
  sorry

end sin_plus_cos_for_point_l162_16297


namespace apple_pies_theorem_l162_16228

def total_apples : ℕ := 128
def unripe_apples : ℕ := 23
def apples_per_pie : ℕ := 7

theorem apple_pies_theorem : 
  (total_apples - unripe_apples) / apples_per_pie = 15 :=
by sorry

end apple_pies_theorem_l162_16228


namespace circle_plus_inequality_equiv_l162_16244

/-- The custom operation ⊕ defined on ℝ -/
def circle_plus (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the equivalence between the inequality and the range of x -/
theorem circle_plus_inequality_equiv (x : ℝ) :
  circle_plus (x - 1) (x + 2) < 0 ↔ x < -1 ∨ x > 1 := by
  sorry

end circle_plus_inequality_equiv_l162_16244


namespace total_laundry_time_l162_16268

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTime (lt : LaundryTime) : ℕ := lt.washing + lt.drying

/-- Given laundry times for whites, darks, and colors, proves that the total time is 344 minutes -/
theorem total_laundry_time (whites darks colors : LaundryTime)
    (h1 : whites = ⟨72, 50⟩)
    (h2 : darks = ⟨58, 65⟩)
    (h3 : colors = ⟨45, 54⟩) :
    totalTime whites + totalTime darks + totalTime colors = 344 := by
  sorry


end total_laundry_time_l162_16268


namespace parabola_vertex_l162_16232

/-- Define a parabola with equation y = (x+2)^2 + 3 -/
def parabola (x : ℝ) : ℝ := (x + 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, 3)

/-- Theorem: The vertex of the parabola y = (x+2)^2 + 3 is (-2, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l162_16232
