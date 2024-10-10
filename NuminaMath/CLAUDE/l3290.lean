import Mathlib

namespace base_conversion_difference_l3290_329057

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_difference : 
  let base_6_num := [0, 1, 2, 3, 4]  -- 43210 in base 6 (least significant digit first)
  let base_7_num := [0, 1, 2, 3]     -- 3210 in base 7 (least significant digit first)
  to_base_10 base_6_num 6 - to_base_10 base_7_num 7 = 4776 := by
  sorry


end base_conversion_difference_l3290_329057


namespace cos_squared_plus_sin_double_l3290_329056

/-- If the terminal side of angle α passes through point P(2, 1) in the Cartesian coordinate system, then cos²α + sin(2α) = 8/5 -/
theorem cos_squared_plus_sin_double (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end cos_squared_plus_sin_double_l3290_329056


namespace solution_of_system_l3290_329063

/-- The system of equations:
    1. xy + 5yz - 6xz = -2z
    2. 2xy + 9yz - 9xz = -12z
    3. yz - 2xz = 6z
-/
def system_of_equations (x y z : ℝ) : Prop :=
  x*y + 5*y*z - 6*x*z = -2*z ∧
  2*x*y + 9*y*z - 9*x*z = -12*z ∧
  y*z - 2*x*z = 6*z

theorem solution_of_system :
  (∃ (x y z : ℝ), system_of_equations x y z ∧ (x = -2 ∧ y = 2 ∧ z = 1/6)) ∧
  (∀ (x : ℝ), system_of_equations x 0 0) ∧
  (∀ (y : ℝ), system_of_equations 0 y 0) :=
by sorry

end solution_of_system_l3290_329063


namespace triangle_minimum_perimeter_l3290_329060

/-- Given a triangle with sides a, b, c, semiperimeter p, and area S,
    the perimeter is minimized when the triangle is equilateral. -/
theorem triangle_minimum_perimeter 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (S : ℝ) (hS : S > 0)
  (harea : S^2 = p * (p - a) * (p - b) * (p - c)) :
  ∃ (min_a min_b min_c : ℝ),
    min_a = min_b ∧ min_b = min_c ∧
    min_a + min_b + min_c ≤ a + b + c ∧
    (min_a + min_b + min_c) / 2 * ((min_a + min_b + min_c) / 2 - min_a) * 
    ((min_a + min_b + min_c) / 2 - min_b) * ((min_a + min_b + min_c) / 2 - min_c) = S^2 := by
  sorry

end triangle_minimum_perimeter_l3290_329060


namespace walking_distance_l3290_329019

theorem walking_distance (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) (actual_distance : ℝ) : 
  original_speed = 12 →
  increased_speed = 20 →
  increased_speed * (actual_distance / original_speed) = original_speed * (actual_distance / original_speed) + additional_distance →
  additional_distance = 24 →
  actual_distance = 36 := by
sorry

end walking_distance_l3290_329019


namespace island_with_2008_roads_sum_of_roads_formula_l3290_329095

def number_of_roads (n : ℕ) : ℕ := 55 + n.choose 2

def sum_of_roads (n : ℕ) : ℕ := 55 * n + (n + 1).choose 3

theorem island_with_2008_roads : ∃ n : ℕ, n > 0 ∧ number_of_roads n = 2008 := by sorry

theorem sum_of_roads_formula (n : ℕ) (h : n > 0) : 
  (Finset.range n).sum (λ k => number_of_roads (k + 1)) = sum_of_roads n := by sorry

end island_with_2008_roads_sum_of_roads_formula_l3290_329095


namespace percent_calculation_l3290_329037

theorem percent_calculation (x : ℝ) (h : 0.6 * x = 42) : 0.5 * x = 35 := by
  sorry

end percent_calculation_l3290_329037


namespace inequality_range_l3290_329051

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end inequality_range_l3290_329051


namespace circumcenter_equidistant_l3290_329067

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

end circumcenter_equidistant_l3290_329067


namespace range_start_number_l3290_329020

theorem range_start_number (n : ℕ) (h1 : n ≤ 79) (h2 : n % 11 = 0) 
  (h3 : ∀ k, k ∈ Finset.range 5 → (n - k * 11) % 11 = 0) : n - 4 * 11 = 33 :=
sorry

end range_start_number_l3290_329020


namespace diamond_composition_l3290_329093

/-- Define the diamond operation -/
def diamond (k : ℝ) (x y : ℝ) : ℝ := x^2 - k*y

/-- Theorem stating the result of h ◇ (h ◇ h) -/
theorem diamond_composition (h : ℝ) : diamond 3 h (diamond 3 h h) = -2*h^2 + 9*h := by
  sorry

end diamond_composition_l3290_329093


namespace subway_distance_difference_l3290_329087

def distance (s : ℝ) : ℝ := 0.5 * s^3 + s^2

theorem subway_distance_difference : 
  distance 7 - distance 4 = 172.5 := by sorry

end subway_distance_difference_l3290_329087


namespace hyperbola_equation_l3290_329046

/-- Given an ellipse and a hyperbola with common foci and specified eccentricity,
    prove the equation of the hyperbola. -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 3 ∧ x^2 / a^2 + y^2 / b^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c^2 = 12 - 3) →                                    -- Foci distance
  (∃ e : ℝ, e = 3/2) →                                         -- Hyperbola eccentricity
  x^2 / 4 - y^2 / 5 = 1                                        -- Hyperbola equation
:= by sorry

end hyperbola_equation_l3290_329046


namespace alpha_beta_sum_l3290_329088

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1233) / (x^2 + 81*x - 3969)) →
  α + β = 143 := by
sorry

end alpha_beta_sum_l3290_329088


namespace levi_goal_difference_l3290_329011

/-- The number of baskets Levi wants to beat his brother by -/
def basketDifference (leviInitial : ℕ) (brotherInitial : ℕ) (brotherIncrease : ℕ) (leviIncrease : ℕ) : ℕ :=
  (leviInitial + leviIncrease) - (brotherInitial + brotherIncrease)

/-- Theorem stating that Levi wants to beat his brother by 5 baskets -/
theorem levi_goal_difference : basketDifference 8 12 3 12 = 5 := by
  sorry

end levi_goal_difference_l3290_329011


namespace solution_set_part1_solution_set_part2_l3290_329045

-- Part 1
def positive_integer_solutions (x : ℕ) : Prop :=
  4 * (x + 2) < 18 + 2 * x

theorem solution_set_part1 :
  {x : ℕ | positive_integer_solutions x} = {1, 2, 3, 4} :=
sorry

-- Part 2
def inequality_system (x : ℝ) : Prop :=
  5 * x + 2 ≥ 4 * x + 1 ∧ (x + 1) / 4 > (x - 3) / 2 + 1

theorem solution_set_part2 :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 3} :=
sorry

end solution_set_part1_solution_set_part2_l3290_329045


namespace lee_annual_salary_l3290_329078

/-- Lee's annual salary calculation --/
theorem lee_annual_salary (monthly_savings : ℕ) (saving_months : ℕ) : 
  monthly_savings = 1000 →
  saving_months = 10 →
  (monthly_savings * saving_months : ℕ) = (2 * (60000 / 12) : ℕ) →
  60000 = (monthly_savings * saving_months * 6 : ℕ) := by
  sorry

#check lee_annual_salary

end lee_annual_salary_l3290_329078


namespace gnuff_tutoring_time_l3290_329064

/-- Calculates the number of minutes tutored given the total amount paid, flat rate, and per-minute rate. -/
def minutes_tutored (total_amount : ℕ) (flat_rate : ℕ) (per_minute_rate : ℕ) : ℕ :=
  (total_amount - flat_rate) / per_minute_rate

/-- Theorem stating that given the specific rates and total amount, the number of minutes tutored is 18. -/
theorem gnuff_tutoring_time :
  minutes_tutored 146 20 7 = 18 := by
  sorry

#eval minutes_tutored 146 20 7

end gnuff_tutoring_time_l3290_329064


namespace molecular_weight_correct_l3290_329035

/-- The molecular weight of C6H8O7 in g/mol -/
def molecular_weight : ℝ := 192

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles in g -/
def given_total_weight : ℝ := 1344

/-- Theorem: The molecular weight of C6H8O7 is correct given the condition -/
theorem molecular_weight_correct : 
  molecular_weight * given_moles = given_total_weight := by
  sorry

end molecular_weight_correct_l3290_329035


namespace city_fuel_efficiency_l3290_329083

/-- Represents the fuel efficiency of a car -/
structure FuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions of the problem -/
def problem_conditions (fe : FuelEfficiency) : Prop :=
  fe.highway * fe.tank_size = 420 ∧
  fe.city * fe.tank_size = 336 ∧
  fe.city = fe.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (h : problem_conditions fe) : 
  fe.city = 24 := by
  sorry

end city_fuel_efficiency_l3290_329083


namespace candy_eaten_l3290_329055

theorem candy_eaten (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 8 →
  sister_candy = 23 →
  remaining_candy = 23 →
  katie_candy + sister_candy - remaining_candy = 8 :=
by sorry

end candy_eaten_l3290_329055


namespace roots_sum_of_squares_l3290_329009

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end roots_sum_of_squares_l3290_329009


namespace sufficient_not_necessary_l3290_329082

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a^2 > 2*a ∧ ¬(a > 2)) :=
by sorry

end sufficient_not_necessary_l3290_329082


namespace min_stamps_for_50_cents_l3290_329053

/-- Represents the number of stamps needed to make a certain value -/
structure StampCombination :=
  (fives : Nat) -- number of 5-cent stamps
  (fours : Nat) -- number of 4-cent stamps

/-- Calculates the total value of stamps in cents -/
def totalValue (sc : StampCombination) : Nat :=
  5 * sc.fives + 4 * sc.fours

/-- Calculates the total number of stamps -/
def totalStamps (sc : StampCombination) : Nat :=
  sc.fives + sc.fours

/-- Checks if a StampCombination is valid (i.e., totals 50 cents) -/
def isValid (sc : StampCombination) : Prop :=
  totalValue sc = 50

/-- Theorem: The minimum number of stamps needed to make 50 cents 
    using only 5-cent and 4-cent stamps is 11 -/
theorem min_stamps_for_50_cents :
  ∃ (sc : StampCombination), 
    isValid sc ∧ 
    totalStamps sc = 11 ∧ 
    (∀ (sc' : StampCombination), isValid sc' → totalStamps sc' ≥ totalStamps sc) :=
  sorry

end min_stamps_for_50_cents_l3290_329053


namespace inverse_variation_problem_l3290_329005

/-- Given that y varies inversely as x², prove that x = 2 when y = 8, 
    given that y = 2 when x = 4. -/
theorem inverse_variation_problem (y x : ℝ) (h : x > 0) : 
  (∃ (k : ℝ), ∀ (x : ℝ), x > 0 → y * x^2 = k) → 
  (2 * 4^2 = 8 * x^2) →
  (y = 8) →
  (x = 2) := by
sorry

end inverse_variation_problem_l3290_329005


namespace first_die_sides_l3290_329047

theorem first_die_sides (p : ℝ) (n : ℕ) : 
  p = 0.023809523809523808 →  -- Given probability
  p = 1 / (n * 7) →           -- Probability formula
  n = 6                       -- Number of sides on first die
:= by sorry

end first_die_sides_l3290_329047


namespace speeding_ticket_theorem_l3290_329061

/-- Represents the percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percentage : ℝ := 16.666666666666664

/-- Theorem stating that 40% of motorists who exceed the speed limit do not receive speeding tickets -/
theorem speeding_ticket_theorem :
  (exceed_limit_percentage - ticket_percentage) / exceed_limit_percentage * 100 = 40 := by
  sorry

end speeding_ticket_theorem_l3290_329061


namespace minsu_age_is_15_l3290_329033

/-- Minsu's age this year -/
def minsu_age : ℕ := 15

/-- Minsu's mother's age this year -/
def mother_age : ℕ := minsu_age + 28

/-- The age difference between Minsu and his mother is 28 years this year -/
axiom age_difference : mother_age = minsu_age + 28

/-- After 13 years, the mother's age will be twice Minsu's age -/
axiom future_age_relation : mother_age + 13 = 2 * (minsu_age + 13)

/-- Theorem: Minsu's age this year is 15 -/
theorem minsu_age_is_15 : minsu_age = 15 := by
  sorry

end minsu_age_is_15_l3290_329033


namespace a_to_b_value_l3290_329012

theorem a_to_b_value (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end a_to_b_value_l3290_329012


namespace square_of_difference_of_roots_l3290_329054

theorem square_of_difference_of_roots (p q : ℝ) : 
  (2 * p^2 + 7 * p - 30 = 0) → 
  (2 * q^2 + 7 * q - 30 = 0) → 
  (p - q)^2 = 289 / 4 := by
  sorry

end square_of_difference_of_roots_l3290_329054


namespace cube_plane_intersection_theorem_l3290_329098

/-- A regular polygon that can be formed by the intersection of a cube and a plane -/
inductive CubeIntersectionPolygon
  | Triangle
  | Quadrilateral
  | Hexagon

/-- The set of all possible regular polygons that can be formed by the intersection of a cube and a plane -/
def possibleIntersectionPolygons : Set CubeIntersectionPolygon :=
  {CubeIntersectionPolygon.Triangle, CubeIntersectionPolygon.Quadrilateral, CubeIntersectionPolygon.Hexagon}

/-- A function that determines if a given regular polygon can be formed by the intersection of a cube and a plane -/
def isValidIntersectionPolygon (p : CubeIntersectionPolygon) : Prop :=
  p ∈ possibleIntersectionPolygons

theorem cube_plane_intersection_theorem :
  ∀ (p : CubeIntersectionPolygon), isValidIntersectionPolygon p ↔
    (p = CubeIntersectionPolygon.Triangle ∨
     p = CubeIntersectionPolygon.Quadrilateral ∨
     p = CubeIntersectionPolygon.Hexagon) :=
by sorry


end cube_plane_intersection_theorem_l3290_329098


namespace a_2023_coordinates_l3290_329004

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns the conjugate point of a given point -/
def conjugate (p : Point) : Point :=
  { x := -p.y + 1, y := p.x + 1 }

/-- Returns the nth point in the sequence starting from A₁ -/
def nthPoint (n : ℕ) : Point :=
  match n % 4 with
  | 1 => { x := 3, y := 1 }
  | 2 => { x := 0, y := 4 }
  | 3 => { x := -3, y := 1 }
  | _ => { x := 0, y := -2 }

theorem a_2023_coordinates : nthPoint 2023 = { x := -3, y := 1 } := by
  sorry

end a_2023_coordinates_l3290_329004


namespace point_inside_circle_l3290_329066

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A point P at distance d from the center of the circle -/
structure Point where
  P : ℝ × ℝ
  d : ℝ

/-- Definition of a point being inside a circle -/
def is_inside (c : Circle) (p : Point) : Prop :=
  p.d < c.r

/-- Theorem: If the distance from a point to the center of a circle
    is less than the radius, then the point is inside the circle -/
theorem point_inside_circle (c : Circle) (p : Point) 
    (h : p.d < c.r) : is_inside c p := by
  sorry

end point_inside_circle_l3290_329066


namespace system_of_equations_solutions_l3290_329096

theorem system_of_equations_solutions :
  -- System (1)
  let x₁ := -1
  let y₁ := 1
  -- System (2)
  let x₂ := 5 / 2
  let y₂ := -2
  -- Proof statements
  (x₁ = y₁ - 2 ∧ 3 * x₁ + 2 * y₁ = -1) ∧
  (2 * x₂ - 3 * y₂ = 11 ∧ 4 * x₂ + 5 * y₂ = 0) := by
  sorry

end system_of_equations_solutions_l3290_329096


namespace inequality_solution_set_l3290_329072

theorem inequality_solution_set (x : ℝ) :
  (Set.Icc (-2 : ℝ) 3 : Set ℝ) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} :=
sorry

end inequality_solution_set_l3290_329072


namespace express_delivery_growth_rate_l3290_329052

theorem express_delivery_growth_rate (initial_packages : ℕ) (final_packages : ℕ) (x : ℝ) :
  initial_packages = 200 →
  final_packages = 242 →
  initial_packages * (1 + x)^2 = final_packages :=
by sorry

end express_delivery_growth_rate_l3290_329052


namespace total_new_games_l3290_329034

/-- Given Katie's and her friends' game collections, prove the total number of new games they have together. -/
theorem total_new_games 
  (katie_new : ℕ) 
  (katie_percent : ℚ) 
  (friends_new : ℕ) 
  (friends_percent : ℚ) 
  (h1 : katie_new = 84) 
  (h2 : katie_percent = 75 / 100) 
  (h3 : friends_new = 8) 
  (h4 : friends_percent = 10 / 100) : 
  katie_new + friends_new = 92 := by
  sorry

end total_new_games_l3290_329034


namespace man_work_time_l3290_329039

/-- Represents the time taken to complete a piece of work -/
structure WorkTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the rate at which work is completed -/
def WorkRate := ℝ

theorem man_work_time (total_work : ℝ) 
  (h_total_work_pos : total_work > 0)
  (combined_time : WorkTime) 
  (son_time : WorkTime) 
  (h_combined : combined_time.days = 3)
  (h_son : son_time.days = 7.5) :
  ∃ (man_time : WorkTime), man_time.days = 5 :=
sorry

end man_work_time_l3290_329039


namespace article_selling_price_l3290_329026

theorem article_selling_price (cost_price : ℝ) (selling_price : ℝ) : 
  (selling_price - cost_price = cost_price - 448) → 
  (768 = 1.2 * cost_price) → 
  selling_price = 832 := by
sorry

end article_selling_price_l3290_329026


namespace pool_filling_cost_l3290_329070

/-- The cost to fill Toby's swimming pool -/
theorem pool_filling_cost 
  (fill_time : ℕ) 
  (flow_rate : ℕ) 
  (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate * water_cost : ℚ) = 5 := by sorry

end pool_filling_cost_l3290_329070


namespace cobys_road_trip_l3290_329065

/-- Coby's road trip problem -/
theorem cobys_road_trip 
  (distance_to_idaho : ℝ) 
  (distance_from_idaho : ℝ) 
  (speed_from_idaho : ℝ) 
  (total_time : ℝ) 
  (h1 : distance_to_idaho = 640)
  (h2 : distance_from_idaho = 550)
  (h3 : speed_from_idaho = 50)
  (h4 : total_time = 19) :
  let time_from_idaho := distance_from_idaho / speed_from_idaho
  let time_to_idaho := total_time - time_from_idaho
  distance_to_idaho / time_to_idaho = 80 := by
sorry


end cobys_road_trip_l3290_329065


namespace metallic_sheet_length_l3290_329030

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  width : ℝ
  length : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given the conditions of the problem, the length of the sheet is 48 m. -/
theorem metallic_sheet_length
  (sheet : MetallicSheet)
  (h1 : sheet.width = 36)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.length = 48 := by
  sorry

#check metallic_sheet_length

end metallic_sheet_length_l3290_329030


namespace pattern_c_cannot_fold_l3290_329014

/-- Represents a pattern of squares with folding lines -/
structure SquarePattern where
  squares : Finset (ℝ × ℝ)  -- Set of coordinates for squares
  foldLines : Finset ((ℝ × ℝ) × (ℝ × ℝ))  -- Set of folding lines

/-- Represents the set of all possible patterns -/
def AllPatterns : Finset SquarePattern := sorry

/-- Predicate to check if a pattern can be folded into a cube without overlap -/
def canFoldIntoCube (p : SquarePattern) : Prop := sorry

/-- The specific Pattern C -/
def PatternC : SquarePattern := sorry

/-- Theorem stating that Pattern C is the only pattern that cannot be folded into a cube -/
theorem pattern_c_cannot_fold :
  PatternC ∈ AllPatterns ∧
  ¬(canFoldIntoCube PatternC) ∧
  ∀ p ∈ AllPatterns, p ≠ PatternC → canFoldIntoCube p :=
sorry

end pattern_c_cannot_fold_l3290_329014


namespace base_sum_theorem_l3290_329008

theorem base_sum_theorem : ∃ (R₁ R₂ : ℕ), 
  (R₁ > 1 ∧ R₂ > 1) ∧
  (4 * R₁ + 5) * (R₂^2 - 1) = (3 * R₂ + 4) * (R₁^2 - 1) ∧
  (5 * R₁ + 4) * (R₂^2 - 1) = (4 * R₂ + 3) * (R₁^2 - 1) ∧
  R₁ + R₂ = 23 := by
  sorry

end base_sum_theorem_l3290_329008


namespace billy_can_play_24_songs_l3290_329085

/-- The number of songs in Billy's music book -/
def total_songs : ℕ := 52

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The number of songs Billy can play -/
def playable_songs : ℕ := total_songs - songs_to_learn

theorem billy_can_play_24_songs : playable_songs = 24 := by
  sorry

end billy_can_play_24_songs_l3290_329085


namespace complement_not_always_smaller_than_supplement_l3290_329002

theorem complement_not_always_smaller_than_supplement :
  ¬ (∀ θ : Real, 0 < θ ∧ θ < π → (π / 2 - θ) < (π - θ)) := by
  sorry

end complement_not_always_smaller_than_supplement_l3290_329002


namespace speaking_orders_count_l3290_329024

def total_students : ℕ := 6
def speakers_to_select : ℕ := 4
def specific_students : ℕ := 2

theorem speaking_orders_count : 
  (total_students.choose speakers_to_select * speakers_to_select.factorial) -
  ((total_students - specific_students).choose speakers_to_select * speakers_to_select.factorial) = 336 :=
by sorry

end speaking_orders_count_l3290_329024


namespace pear_juice_blend_percentage_l3290_329022

/-- Represents the amount of juice extracted from a fruit -/
structure JuiceYield where
  fruit : String
  amount : ℚ
  count : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (pear_yield orange_yield : JuiceYield) : ℚ :=
  let pear_juice := pear_yield.amount / pear_yield.count
  let orange_juice := orange_yield.amount / orange_yield.count
  let total_juice := pear_juice + orange_juice
  (pear_juice / total_juice) * 100

/-- Theorem: The percentage of pear juice in the blend is 40% -/
theorem pear_juice_blend_percentage :
  let pear_yield := JuiceYield.mk "pear" 8 3
  let orange_yield := JuiceYield.mk "orange" 8 2
  pear_juice_percentage pear_yield orange_yield = 40 := by
  sorry

end pear_juice_blend_percentage_l3290_329022


namespace cashew_mixture_problem_l3290_329090

/-- Represents the price of peanuts per pound -/
def peanut_price : ℝ := 2.40

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.00

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := 60

/-- Represents the selling price of the mixture per pound -/
def mixture_price : ℝ := 3.00

/-- Represents the amount of cashews in pounds -/
def cashew_amount : ℝ := 10

theorem cashew_mixture_problem :
  ∃ (peanut_amount : ℝ),
    peanut_amount + cashew_amount = total_weight ∧
    peanut_price * peanut_amount + cashew_price * cashew_amount = mixture_price * total_weight :=
by
  sorry

end cashew_mixture_problem_l3290_329090


namespace triangle_side_length_l3290_329017

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = Real.sqrt 3 →
  Real.tan B = Real.sqrt 2 / 4 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 2 / 3 := by sorry

end triangle_side_length_l3290_329017


namespace complex_fraction_real_l3290_329043

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / (2 - Complex.I)).im = 0 → a = -1/2 := by
  sorry

end complex_fraction_real_l3290_329043


namespace prob_sum_less_than_10_l3290_329016

/-- A fair die with faces labeled 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The sample space of rolling a fair die twice -/
def SampleSpace : Finset (ℕ × ℕ) :=
  FairDie.product FairDie

/-- The event where the sum of face values is less than 10 -/
def EventSumLessThan10 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun (x, y) => x + y < 10)

/-- Theorem: The probability of the sum of face values being less than 10
    when rolling a fair six-sided die twice is 5/6 -/
theorem prob_sum_less_than_10 :
  (EventSumLessThan10.card : ℚ) / SampleSpace.card = 5 / 6 := by
  sorry

end prob_sum_less_than_10_l3290_329016


namespace unique_divisor_remainder_l3290_329007

theorem unique_divisor_remainder : ∃! (d r : ℤ),
  (1210 % d = r) ∧
  (1690 % d = r) ∧
  (2670 % d = r) ∧
  (d > 0) ∧
  (0 ≤ r) ∧
  (r < d) ∧
  (d - 4*r = -20) := by
  sorry

end unique_divisor_remainder_l3290_329007


namespace no_prime_pair_divisibility_l3290_329023

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end no_prime_pair_divisibility_l3290_329023


namespace tangent_circles_a_values_l3290_329084

/-- Two circles that intersect at exactly one point -/
structure TangentCircles where
  /-- The parameter 'a' in the equation of the second circle -/
  a : ℝ
  /-- The first circle: x^2 + y^2 = 4 -/
  circle1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 = 4
  /-- The second circle: (x-a)^2 + y^2 = 1 -/
  circle2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  /-- The circles intersect at exactly one point -/
  intersect_at_one_point : ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2

/-- The theorem stating that 'a' must be in the set {1, -1, 3, -3} -/
theorem tangent_circles_a_values (tc : TangentCircles) : tc.a = 1 ∨ tc.a = -1 ∨ tc.a = 3 ∨ tc.a = -3 := by
  sorry

end tangent_circles_a_values_l3290_329084


namespace sin_450_degrees_l3290_329068

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end sin_450_degrees_l3290_329068


namespace john_cookies_problem_l3290_329071

theorem john_cookies_problem (cookies_left : ℕ) (cookies_eaten : ℕ) (dozen : ℕ) :
  cookies_left = 21 →
  cookies_eaten = 3 →
  dozen = 12 →
  (cookies_left + cookies_eaten) / dozen = 2 := by
  sorry

end john_cookies_problem_l3290_329071


namespace power_value_theorem_l3290_329069

theorem power_value_theorem (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(2*m - n) = 4/3 := by
  sorry

end power_value_theorem_l3290_329069


namespace james_night_out_cost_l3290_329001

/-- Calculate the total amount James spent for a night out -/
theorem james_night_out_cost : 
  let entry_fee : ℚ := 25
  let friends_count : ℕ := 8
  let rounds_count : ℕ := 3
  let james_drinks_count : ℕ := 7
  let cocktail_price : ℚ := 8
  let non_alcoholic_price : ℚ := 4
  let james_cocktails_count : ℕ := 6
  let burger_price : ℚ := 18
  let tip_percentage : ℚ := 0.25

  let friends_drinks_cost := friends_count * rounds_count * cocktail_price
  let james_drinks_cost := james_cocktails_count * cocktail_price + 
                           (james_drinks_count - james_cocktails_count) * non_alcoholic_price
  let food_cost := burger_price
  let subtotal := entry_fee + friends_drinks_cost + james_drinks_cost + food_cost
  let tip := subtotal * tip_percentage
  let total_cost := subtotal + tip

  total_cost = 358.75 := by sorry

end james_night_out_cost_l3290_329001


namespace johns_age_to_tonyas_age_ratio_l3290_329006

/-- Proves that the ratio of John's age to Tonya's age is 1:2 given the specified conditions --/
theorem johns_age_to_tonyas_age_ratio :
  ∀ (john mary tonya : ℕ),
    john = 2 * mary →
    tonya = 60 →
    (john + mary + tonya) / 3 = 35 →
    john / tonya = 1 / 2 := by
  sorry

end johns_age_to_tonyas_age_ratio_l3290_329006


namespace no_loops_in_process_flowchart_l3290_329000

-- Define the basic concepts
def ProcessFlowchart : Type := Unit
def AlgorithmFlowchart : Type := Unit
def Process : Type := Unit
def FlowLine : Type := Unit

-- Define the properties of process flowcharts
def is_similar_to (pf : ProcessFlowchart) (af : AlgorithmFlowchart) : Prop := sorry
def refine_step_by_step (p : Process) : Prop := sorry
def connect_adjacent_processes (fl : FlowLine) : Prop := sorry
def is_directional (fl : FlowLine) : Prop := sorry

-- Define the concept of a loop
def Loop : Type := Unit
def contains_loop (pf : ProcessFlowchart) (l : Loop) : Prop := sorry

-- State the theorem
theorem no_loops_in_process_flowchart (pf : ProcessFlowchart) (af : AlgorithmFlowchart) 
  (p : Process) (fl : FlowLine) :
  is_similar_to pf af →
  refine_step_by_step p →
  connect_adjacent_processes fl →
  is_directional fl →
  ∀ l : Loop, ¬ contains_loop pf l := by
  sorry

end no_loops_in_process_flowchart_l3290_329000


namespace parallel_lines_m_value_l3290_329076

-- Define the lines l₁ and l₂
def l₁ (x y m : ℝ) : Prop := x + (1 + m) * y + (m - 2) = 0
def l₂ (x y m : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the parallel condition
def parallel (m : ℝ) : Prop := ∀ x y, l₁ x y m → l₂ x y m

-- Theorem statement
theorem parallel_lines_m_value (m : ℝ) : parallel m → m = 1 := by
  sorry

end parallel_lines_m_value_l3290_329076


namespace sum_first_six_primes_mod_seventh_prime_l3290_329038

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end sum_first_six_primes_mod_seventh_prime_l3290_329038


namespace seven_balls_three_boxes_l3290_329015

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 64 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute 7 3 = 64 := by sorry

end seven_balls_three_boxes_l3290_329015


namespace barn_paint_area_l3290_329092

/-- Calculates the total area to be painted for a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  let wall_area1 := 2 * width * height
  let wall_area2 := 2 * length * height
  let ceiling_area := width * length
  2 * (wall_area1 + wall_area2) + 2 * ceiling_area

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_paint_area :
  total_paint_area 12 15 6 = 1008 := by sorry

end barn_paint_area_l3290_329092


namespace cosine_identities_l3290_329099

theorem cosine_identities :
  (Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1/2) ∧
  (Real.cos (π/7) - Real.cos (2*π/7) + Real.cos (3*π/7) = 1/2) := by
  sorry

end cosine_identities_l3290_329099


namespace decagon_circle_intersection_undecagon_no_circle_intersection_l3290_329041

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Constructs circles on each side of a polygon -/
def constructCircles {n : ℕ} (p : Polygon n) : Fin n → Circle :=
  sorry

/-- Checks if a point is a common intersection of all circles -/
def isCommonIntersection {n : ℕ} (p : Polygon n) (circles : Fin n → Circle) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Checks if a point is a vertex of the polygon -/
def isVertex {n : ℕ} (p : Polygon n) (point : ℝ × ℝ) : Prop :=
  sorry

theorem decagon_circle_intersection :
  ∃ (p : Polygon 10) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point ∧
    ¬isVertex p point :=
  sorry

theorem undecagon_no_circle_intersection :
  ∀ (p : Polygon 11) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point →
    isVertex p point :=
  sorry

end decagon_circle_intersection_undecagon_no_circle_intersection_l3290_329041


namespace square_sum_lower_bound_l3290_329040

theorem square_sum_lower_bound (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
sorry

end square_sum_lower_bound_l3290_329040


namespace quadratic_inequality_condition_l3290_329025

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 < a ∧ a < 4) := by sorry

end quadratic_inequality_condition_l3290_329025


namespace symmetry_implies_axis_l3290_329048

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function g if 
    for every point (x, g(x)) on the graph, (3-x, g(x)) is also on the graph -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  IsSymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g := by
  sorry

end symmetry_implies_axis_l3290_329048


namespace pencils_left_l3290_329062

/-- Given two boxes of pencils with fourteen pencils each, prove that after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given_away : ℕ) : 
  boxes = 2 → pencils_per_box = 14 → pencils_given_away = 6 →
  boxes * pencils_per_box - pencils_given_away = 22 := by
sorry

end pencils_left_l3290_329062


namespace parallel_planes_iff_parallel_lines_l3290_329021

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (π : Plane) : Prop := sorry

/-- Two objects are different -/
def different {α : Type*} (a b : α) : Prop := a ≠ b

theorem parallel_planes_iff_parallel_lines 
  (m n : Line) (α β : Plane) 
  (h1 : different m n)
  (h2 : different α β)
  (h3 : perpendicular_line_plane m β)
  (h4 : perpendicular_line_plane n β) :
  parallel_planes α β ↔ parallel_lines m n := by sorry

end parallel_planes_iff_parallel_lines_l3290_329021


namespace cone_surface_area_l3290_329044

theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  π * (base_circumference / (2 * π)) * (base_circumference / (2 * π) + slant_height) = 3 * π := by
  sorry

end cone_surface_area_l3290_329044


namespace rope_folding_theorem_l3290_329091

def rope_segments (n : ℕ) : ℕ := 2^n + 1

theorem rope_folding_theorem :
  rope_segments 5 = 33 := by sorry

end rope_folding_theorem_l3290_329091


namespace scissors_freedom_theorem_l3290_329050

/-- Represents the state of the rope and scissors system -/
structure RopeScissorsState where
  loopThroughScissors : Bool
  ropeEndsFixed : Bool
  noKnotsUntied : Bool

/-- Represents a single manipulation of the rope -/
inductive RopeManipulation
  | PullLoop
  | PassLoopAroundEnds
  | ReverseDirection

/-- Defines a sequence of rope manipulations -/
def ManipulationSequence := List RopeManipulation

/-- Predicate to check if a manipulation sequence frees the scissors -/
def freesScissors (seq : ManipulationSequence) : Prop := sorry

/-- The main theorem stating that there exists a sequence of manipulations that frees the scissors -/
theorem scissors_freedom_theorem (initialState : RopeScissorsState) 
  (h1 : initialState.loopThroughScissors = true)
  (h2 : initialState.ropeEndsFixed = true)
  (h3 : initialState.noKnotsUntied = true) :
  ∃ (seq : ManipulationSequence), freesScissors seq := by
  sorry


end scissors_freedom_theorem_l3290_329050


namespace no_prime_sum_53_l3290_329010

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- State the theorem
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end no_prime_sum_53_l3290_329010


namespace chrysanthemum_arrangement_count_l3290_329059

/-- Represents the number of pots for each color of chrysanthemum -/
structure ChrysanthemumCounts where
  white : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemums with given conditions -/
def chrysanthemumArrangements (counts : ChrysanthemumCounts) : Nat :=
  sorry

/-- The main theorem stating the number of arrangements for the given problem -/
theorem chrysanthemum_arrangement_count :
  let counts : ChrysanthemumCounts := { white := 2, yellow := 2, red := 1 }
  chrysanthemumArrangements counts = 16 := by
  sorry

end chrysanthemum_arrangement_count_l3290_329059


namespace complex_number_in_fourth_quadrant_l3290_329013

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a^2 - 4*a + 5) - 6*Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_number_in_fourth_quadrant_l3290_329013


namespace sixteen_four_eight_calculation_l3290_329073

theorem sixteen_four_eight_calculation : (16^2 / 4^3) * 8^3 = 2048 := by
  sorry

end sixteen_four_eight_calculation_l3290_329073


namespace flower_pot_cost_l3290_329075

/-- The cost of the largest pot in a set of 6 pots -/
def largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_diff * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_diff * (num_pots - 1)

/-- Theorem stating the cost of the largest pot given the problem conditions -/
theorem flower_pot_cost :
  largest_pot_cost 8.25 6 0.1 = 1.625 := by
  sorry

end flower_pot_cost_l3290_329075


namespace min_sum_abs_roots_irrational_quadratic_l3290_329018

theorem min_sum_abs_roots_irrational_quadratic (p q : ℤ) 
  (h_irrational : ∀ (α : ℝ), α^2 + p*α + q = 0 → ¬ IsAlgebraic ℚ α) :
  ∃ (α₁ α₂ : ℝ), 
    α₁^2 + p*α₁ + q = 0 ∧ 
    α₂^2 + p*α₂ + q = 0 ∧ 
    |α₁| + |α₂| ≥ Real.sqrt 5 ∧
    (∃ (p' q' : ℤ) (β₁ β₂ : ℝ), 
      β₁^2 + p'*β₁ + q' = 0 ∧ 
      β₂^2 + p'*β₂ + q' = 0 ∧ 
      |β₁| + |β₂| = Real.sqrt 5) := by
  sorry

end min_sum_abs_roots_irrational_quadratic_l3290_329018


namespace tunnel_length_tunnel_length_specific_l3290_329032

/-- Calculates the length of a tunnel given train specifications and travel time -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (travel_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let travel_time_s := travel_time_min * 60
  let total_distance := train_speed_ms * travel_time_s
  let tunnel_length_m := total_distance - train_length
  let tunnel_length_km := tunnel_length_m / 1000
  tunnel_length_km

/-- The length of the tunnel is 1.7 km given the specified conditions -/
theorem tunnel_length_specific : tunnel_length 100 72 1.5 = 1.7 := by
  sorry

end tunnel_length_tunnel_length_specific_l3290_329032


namespace vertex_not_at_minus_two_one_l3290_329049

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-2)^2 + 1 --/
def givenParabola : Parabola := { a := -2, h := 2, k := 1 }

/-- The vertex of a parabola --/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Theorem stating that the vertex of the given parabola is not at (-2,1) --/
theorem vertex_not_at_minus_two_one :
  vertex givenParabola ≠ (-2, 1) := by
  sorry

end vertex_not_at_minus_two_one_l3290_329049


namespace six_distinct_areas_l3290_329081

/-- Represents a point in a one-dimensional space -/
structure Point1D where
  x : ℝ

/-- Represents a line in a two-dimensional space -/
structure Line2D where
  points : List Point1D
  y : ℝ

/-- The configuration of points as described in the problem -/
structure PointConfiguration where
  line1 : Line2D
  line2 : Line2D
  w : Point1D
  x : Point1D
  y : Point1D
  z : Point1D
  p : Point1D
  q : Point1D

/-- Checks if the configuration satisfies the given conditions -/
def validConfiguration (config : PointConfiguration) : Prop :=
  config.w.x < config.x.x ∧ config.x.x < config.y.x ∧ config.y.x < config.z.x ∧
  config.x.x - config.w.x = 1 ∧
  config.y.x - config.x.x = 2 ∧
  config.z.x - config.y.x = 3 ∧
  config.q.x - config.p.x = 4 ∧
  config.line1.y ≠ config.line2.y ∧
  config.line1.points = [config.w, config.x, config.y, config.z] ∧
  config.line2.points = [config.p, config.q]

/-- Calculates the number of possible distinct triangle areas -/
def distinctTriangleAreas (config : PointConfiguration) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 6 possible distinct triangle areas -/
theorem six_distinct_areas (config : PointConfiguration) 
  (h : validConfiguration config) : distinctTriangleAreas config = 6 := by
  sorry

end six_distinct_areas_l3290_329081


namespace city_population_problem_l3290_329080

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p + 50) → p = 1500 := by
  sorry

end city_population_problem_l3290_329080


namespace intersection_line_circle_l3290_329058

/-- Given a line y = 2x + 1 intersecting a circle x^2 + y^2 + ax + 2y + 1 = 0 at points A and B,
    and a line mx + y + 2 = 0 that bisects chord AB perpendicularly, prove that a = 4 -/
theorem intersection_line_circle (a m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = 2 * A.1 + 1 ∧ B.2 = 2 * B.1 + 1) ∧ 
    (A.1^2 + A.2^2 + a * A.1 + 2 * A.2 + 1 = 0 ∧ 
     B.1^2 + B.2^2 + a * B.1 + 2 * B.2 + 1 = 0) ∧
    (∃ C : ℝ × ℝ, C ∈ Set.Icc A B ∧ 
      m * C.1 + C.2 + 2 = 0 ∧
      (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)) →
  a = 4 := by
  sorry

end intersection_line_circle_l3290_329058


namespace simplified_expression_terms_l3290_329079

/-- The number of terms in the simplified expression of (x+y+z)^2006 + (x-y-z)^2006 -/
def num_terms : ℕ := 1008016

/-- The exponent used in the expression -/
def exponent : ℕ := 2006

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 :=
sorry

end simplified_expression_terms_l3290_329079


namespace jacqueline_boxes_l3290_329074

/-- The number of erasers per box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := 40

/-- The number of boxes Jacqueline has -/
def num_boxes : ℕ := total_erasers / erasers_per_box

theorem jacqueline_boxes : num_boxes = 4 := by
  sorry

end jacqueline_boxes_l3290_329074


namespace composite_power_sum_l3290_329089

theorem composite_power_sum (n : ℕ) (h : n % 6 = 4) : 3 ∣ (n^n + (n+1)^(n+1)) := by
  sorry

end composite_power_sum_l3290_329089


namespace cost_price_of_ball_l3290_329028

/-- Given that selling 11 balls at Rs. 720 results in a loss equal to the cost price of 5 balls,
    prove that the cost price of one ball is Rs. 120. -/
theorem cost_price_of_ball (cost : ℕ) : 
  (11 * cost - 720 = 5 * cost) → cost = 120 := by
  sorry

end cost_price_of_ball_l3290_329028


namespace area_of_right_triangle_with_inscribed_circle_l3290_329031

-- Define the right triangle with inscribed circle
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧
  a^2 + b^2 = c^2 ∧
  c = 13 ∧
  (r + 6) = a ∧
  (r + 7) = b

-- Theorem statement
theorem area_of_right_triangle_with_inscribed_circle 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) :
  (1/2 : ℝ) * a * b = 42 :=
by
  sorry

#check area_of_right_triangle_with_inscribed_circle

end area_of_right_triangle_with_inscribed_circle_l3290_329031


namespace seashell_difference_l3290_329003

theorem seashell_difference (fred_shells tom_shells : ℕ) 
  (h1 : fred_shells = 43)
  (h2 : tom_shells = 15) :
  fred_shells - tom_shells = 28 := by
  sorry

end seashell_difference_l3290_329003


namespace solution_count_l3290_329097

/-- The number of solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_solutions : ℕ := 4

/-- The number of real solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_real_solutions : ℕ := 2

/-- The number of complex solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_complex_solutions : ℕ := 2

/-- Definition of the first equation: y = (x+1)^3 -/
def equation1 (x y : ℂ) : Prop := y = (x + 1)^3

/-- Definition of the second equation: xy + y = 1 -/
def equation2 (x y : ℂ) : Prop := x * y + y = 1

/-- A solution is a pair (x, y) that satisfies both equations -/
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

/-- The main theorem stating the number and nature of solutions -/
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = num_solutions ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (sr : Finset (ℝ × ℝ)), sr.card = num_real_solutions ∧
      (∀ (p : ℝ × ℝ), p ∈ sr ↔ is_solution p.1 p.2)) ∧
    (∃ (sc : Finset (ℂ × ℂ)), sc.card = num_complex_solutions ∧
      (∀ (p : ℂ × ℂ), p ∈ sc ↔ (is_solution p.1 p.2 ∧ ¬(p.1.im = 0 ∧ p.2.im = 0))))) :=
sorry

end solution_count_l3290_329097


namespace degree_to_radian_conversion_l3290_329094

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  855 * (π / 180) = 59 * π / 12 :=
by sorry

end degree_to_radian_conversion_l3290_329094


namespace min_throws_for_repeated_sum_l3290_329029

/-- The minimum number of distinct sums possible when rolling three six-sided dice -/
def distinct_sums : ℕ := 16

/-- The minimum number of throws needed to guarantee a repeated sum -/
def min_throws : ℕ := distinct_sums + 1

/-- Theorem stating that the minimum number of throws to ensure a repeated sum is 17 -/
theorem min_throws_for_repeated_sum :
  min_throws = 17 := by sorry

end min_throws_for_repeated_sum_l3290_329029


namespace blocks_added_l3290_329042

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 86) 
  (h2 : final_blocks = 95) : 
  final_blocks - initial_blocks = 9 := by
  sorry

end blocks_added_l3290_329042


namespace basketball_surface_area_l3290_329077

/-- The surface area of a sphere with diameter 24 centimeters is 576π square centimeters. -/
theorem basketball_surface_area : 
  let diameter : ℝ := 24
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 576 * Real.pi := by sorry

end basketball_surface_area_l3290_329077


namespace geometric_progression_first_term_l3290_329086

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10) 
  (h2 : sum_first_two = 7) : 
  ∃ a r : ℝ, 
    S = a / (1 - r) ∧ 
    sum_first_two = a + a * r ∧ 
    a = 10 * (1 + Real.sqrt (3 / 10)) := by
  sorry

end geometric_progression_first_term_l3290_329086


namespace workers_savings_l3290_329027

theorem workers_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) 
  (h3 : 12 * f * P = 6 * (1 - f) * P) : f = 1 / 3 := by
  sorry

end workers_savings_l3290_329027


namespace inequality_proof_l3290_329036

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c :=
by sorry

end inequality_proof_l3290_329036
