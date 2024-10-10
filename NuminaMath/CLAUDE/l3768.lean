import Mathlib

namespace typing_speed_ratio_l3768_376801

theorem typing_speed_ratio (T M : ℝ) (h1 : T > 0) (h2 : M > 0) 
  (h3 : T + M = 12) (h4 : T + 1.25 * M = 14) : M / T = 2 := by
  sorry

end typing_speed_ratio_l3768_376801


namespace wall_height_calculation_l3768_376891

/-- Calculates the height of a wall given its dimensions and the number and size of bricks used. -/
theorem wall_height_calculation (wall_length : Real) (wall_thickness : Real) 
  (brick_count : Nat) (brick_length : Real) (brick_width : Real) (brick_height : Real) : 
  wall_length = 900 ∧ wall_thickness = 22.5 ∧ brick_count = 7200 ∧ 
  brick_length = 25 ∧ brick_width = 11.25 ∧ brick_height = 6 → 
  (wall_length * wall_thickness * (brick_count * brick_length * brick_width * brick_height) / 
  (wall_length * wall_thickness)) = 600 := by
  sorry

end wall_height_calculation_l3768_376891


namespace minBrokenLine_l3768_376872

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def sameSide (A B : Point) (l : Line) : Prop := sorry

def reflectPoint (A : Point) (l : Line) : Point := sorry

def onLine (X : Point) (l : Line) : Prop := sorry

def intersectionPoint (l : Line) (A B : Point) : Point := sorry

def brokenLineLength (A X B : Point) : ℝ := sorry

-- State the theorem
theorem minBrokenLine (l : Line) (A B : Point) :
  sameSide A B l →
  ∃ X : Point, onLine X l ∧
    ∀ Y : Point, onLine Y l →
      brokenLineLength A X B ≤ brokenLineLength A Y B :=
  by
    sorry

end minBrokenLine_l3768_376872


namespace sum_of_decimals_l3768_376847

/-- The sum of 0.2, 0.03, 0.004, 0.0005, and 0.00006 is equal to 5864/25000 -/
theorem sum_of_decimals : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 5864 / 25000 := by
  sorry

end sum_of_decimals_l3768_376847


namespace bus_ticket_savings_l3768_376890

/-- The cost of a single bus ticket in dollars -/
def single_ticket_cost : ℚ := 1.50

/-- The cost of a package of 5 bus tickets in dollars -/
def package_cost : ℚ := 5.75

/-- The number of tickets required -/
def required_tickets : ℕ := 40

/-- The number of tickets in a package -/
def tickets_per_package : ℕ := 5

/-- Theorem stating the savings when buying packages instead of single tickets -/
theorem bus_ticket_savings :
  single_ticket_cost * required_tickets -
  package_cost * (required_tickets / tickets_per_package) = 14 := by
  sorry

end bus_ticket_savings_l3768_376890


namespace min_fencing_length_l3768_376874

/-- Represents the dimensions of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden, excluding one side (against the wall) -/
def Garden.fencingLength (g : Garden) : ℝ := g.length + 2 * g.width

/-- The minimum fencing length for a garden with area 50 m² is 20 meters -/
theorem min_fencing_length :
  ∀ g : Garden, g.area = 50 → g.fencingLength ≥ 20 ∧ 
  ∃ g' : Garden, g'.area = 50 ∧ g'.fencingLength = 20 := by
  sorry


end min_fencing_length_l3768_376874


namespace system_two_solutions_l3768_376844

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 289 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ (|x| - 8)^2 + (|y| - 15)^2 = a) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end system_two_solutions_l3768_376844


namespace globe_division_count_l3768_376896

/-- The number of parts a globe's surface is divided into, given the number of parallels and meridians -/
def globe_divisions (parallels : ℕ) (meridians : ℕ) : ℕ :=
  meridians * (parallels + 1)

/-- Theorem: A globe with 17 parallels and 24 meridians is divided into 432 parts -/
theorem globe_division_count : globe_divisions 17 24 = 432 := by
  sorry

end globe_division_count_l3768_376896


namespace count_satisfying_integers_l3768_376825

def satisfies_conditions (n : ℤ) : Prop :=
  (n + 5) * (n - 5) * (n - 15) < 0 ∧ n > 7

theorem count_satisfying_integers :
  ∃ (S : Finset ℤ), (∀ n ∈ S, satisfies_conditions n) ∧ 
                    (∀ n, satisfies_conditions n → n ∈ S) ∧
                    S.card = 7 := by
  sorry

end count_satisfying_integers_l3768_376825


namespace problem_solving_probability_l3768_376816

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end problem_solving_probability_l3768_376816


namespace proposition_is_false_l3768_376837

theorem proposition_is_false : ¬(∀ x : ℤ, x ∈ ({1, -1, 0} : Set ℤ) → 2*x + 1 > 0) := by
  sorry

end proposition_is_false_l3768_376837


namespace interest_rate_calculation_l3768_376860

/-- Given simple interest, principal, and time, calculate the interest rate in paise per rupee per month -/
theorem interest_rate_calculation (simple_interest principal time : ℚ) 
  (h1 : simple_interest = 4.8)
  (h2 : principal = 8)
  (h3 : time = 12) :
  (simple_interest / (principal * time)) * 100 = 5 := by
  sorry

end interest_rate_calculation_l3768_376860


namespace triangle_height_l3768_376823

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ)
  (h_area : area = 24)
  (h_base : base = 8)
  (h_triangle_area : area = (base * height) / 2) :
  height = 6 := by
sorry

end triangle_height_l3768_376823


namespace star_two_three_star_two_neg_six_neg_two_thirds_l3768_376888

-- Define the operation *
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem for 2 * 3 = 5/3
theorem star_two_three : star 2 3 = 5/3 := by sorry

-- Theorem for 2 * (-6) * (-2/3) = -2/3
theorem star_two_neg_six_neg_two_thirds : star (star 2 (-6)) (-2/3) = -2/3 := by sorry

end star_two_three_star_two_neg_six_neg_two_thirds_l3768_376888


namespace trajectory_equation_l3768_376877

/-- The trajectory of point P satisfies x² + y² = 1, given a line l: x cos θ + y sin θ = 1,
    where OP is perpendicular to l at P, and O is the origin. -/
theorem trajectory_equation (θ : ℝ) (x y : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y ∧
    (x * Real.cos θ + y * Real.sin θ = 1) ∧
    (∃ (t : ℝ), P = (t * Real.cos θ, t * Real.sin θ))) →
  x^2 + y^2 = 1 :=
by sorry

end trajectory_equation_l3768_376877


namespace perfect_square_condition_l3768_376845

theorem perfect_square_condition (n : ℕ) : 
  (∃ (a : ℕ), 2^n + 3 = a^2) ↔ n = 0 := by
sorry

end perfect_square_condition_l3768_376845


namespace triangle_properties_l3768_376808

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- Given conditions for the specific triangle -/
def special_triangle (t : AcuteTriangle) : Prop :=
  t.a = 2 * t.b * Real.sin t.A ∧ t.a = 3 * Real.sqrt 3 ∧ t.c = 5

theorem triangle_properties (t : AcuteTriangle) (h : special_triangle t) : 
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end triangle_properties_l3768_376808


namespace exchange_calculation_l3768_376812

/-- Exchange rate between lire and dollars -/
def exchange_rate : ℚ := 2500 / 2

/-- Amount of dollars to be exchanged -/
def dollars_to_exchange : ℚ := 5

/-- Function to calculate lire received for a given amount of dollars -/
def lire_received (dollars : ℚ) : ℚ := dollars * exchange_rate

theorem exchange_calculation :
  lire_received dollars_to_exchange = 6250 := by
  sorry

end exchange_calculation_l3768_376812


namespace wall_bricks_count_l3768_376895

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Efficiency loss when working together (in bricks per hour) -/
def efficiency_loss : ℕ := 12

/-- Time taken by both bricklayers working together -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (time_together : ℚ) * ((total_bricks / time1 : ℚ) + (total_bricks / time2 : ℚ) - efficiency_loss) = total_bricks := by
  sorry

#check wall_bricks_count

end wall_bricks_count_l3768_376895


namespace frank_candy_total_l3768_376828

/-- Given that Frank puts 11 pieces of candy in each bag and makes 2 bags,
    prove that the total number of candy pieces is 22. -/
theorem frank_candy_total (pieces_per_bag : ℕ) (num_bags : ℕ) 
    (h1 : pieces_per_bag = 11) (h2 : num_bags = 2) : 
    pieces_per_bag * num_bags = 22 := by
  sorry

end frank_candy_total_l3768_376828


namespace min_cost_for_equal_distribution_l3768_376800

def tangerines_needed (initial : ℕ) (people : ℕ) : ℕ :=
  (people - initial % people) % people

def cost_of_additional_tangerines (initial : ℕ) (people : ℕ) (price : ℕ) : ℕ :=
  tangerines_needed initial people * price

theorem min_cost_for_equal_distribution (initial : ℕ) (people : ℕ) (price : ℕ) 
  (h1 : initial = 98) (h2 : people = 12) (h3 : price = 450) :
  cost_of_additional_tangerines initial people price = 4500 := by
  sorry

end min_cost_for_equal_distribution_l3768_376800


namespace gary_egg_collection_l3768_376829

/-- Calculates the number of eggs collected per week given the initial number of chickens,
    the multiplication factor after two years, eggs laid per chicken per day, and days in a week. -/
def eggs_per_week (initial_chickens : ℕ) (multiplication_factor : ℕ) (eggs_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  initial_chickens * multiplication_factor * eggs_per_day * days_in_week

/-- Proves that Gary collects 1344 eggs per week given the initial conditions. -/
theorem gary_egg_collection :
  eggs_per_week 4 8 6 7 = 1344 :=
by sorry

end gary_egg_collection_l3768_376829


namespace prime_sum_theorem_l3768_376805

theorem prime_sum_theorem (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  Nat.Prime (7 * p + q) → 
  Nat.Prime (2 * q + 11) → 
  p^q + q^p = 17 := by
sorry

end prime_sum_theorem_l3768_376805


namespace max_third_term_is_16_l3768_376881

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithmeticSequence where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference
  sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithmeticSequence, third_term seq = 16 :=
sorry

end max_third_term_is_16_l3768_376881


namespace smallest_y_for_perfect_cube_l3768_376817

def x : ℕ := 7 * 24 * 48

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 ∧ z < y → ¬is_perfect_cube (x * z) :=
by sorry

end smallest_y_for_perfect_cube_l3768_376817


namespace logarithm_sum_simplification_l3768_376835

theorem logarithm_sum_simplification :
  1 / (Real.log 2 / Real.log 7 + 1) +
  1 / (Real.log 3 / Real.log 11 + 1) +
  1 / (Real.log 5 / Real.log 13 + 1) = 3 := by
  sorry

end logarithm_sum_simplification_l3768_376835


namespace bridget_weight_l3768_376815

/-- Given that Martha weighs 2 pounds and Bridget is 37 pounds heavier than Martha,
    prove that Bridget weighs 39 pounds. -/
theorem bridget_weight (martha_weight : ℕ) (weight_difference : ℕ) :
  martha_weight = 2 →
  weight_difference = 37 →
  martha_weight + weight_difference = 39 :=
by sorry

end bridget_weight_l3768_376815


namespace simple_interest_months_l3768_376820

/-- Simple interest calculation -/
theorem simple_interest_months (principal : ℝ) (rate : ℝ) (interest : ℝ) : 
  principal = 10000 →
  rate = 0.08 →
  interest = 800 →
  (interest / (principal * rate)) * 12 = 12 := by
sorry

end simple_interest_months_l3768_376820


namespace road_trip_time_calculation_l3768_376855

/-- Calculates the total time for a road trip given the specified conditions -/
theorem road_trip_time_calculation (distance : ℝ) (speed : ℝ) (break_interval : ℝ) (break_duration : ℝ) (hotel_search_time : ℝ) : 
  distance = 2790 →
  speed = 62 →
  break_interval = 5 →
  break_duration = 0.5 →
  hotel_search_time = 0.5 →
  (distance / speed + 
   (⌊distance / speed / break_interval⌋ - 1) * break_duration + 
   hotel_search_time) = 49.5 := by
  sorry

#check road_trip_time_calculation

end road_trip_time_calculation_l3768_376855


namespace cubic_roots_sum_l3768_376889

theorem cubic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = -2 := by
sorry

end cubic_roots_sum_l3768_376889


namespace ellipse_chord_length_l3768_376882

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ  -- Slope
  c : ℝ  -- y-intercept

theorem ellipse_chord_length (C : Ellipse) (L : Line) :
  C.b = 1 ∧ C.e = Real.sqrt 3 / 2 ∧ L.m = 1 ∧ L.c = 1 →
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  Real.sqrt ((8/5)^2 + (8/5)^2) = 8 * Real.sqrt 2 / 5 :=
by sorry

end ellipse_chord_length_l3768_376882


namespace weight_of_b_l3768_376836

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 45) :
  b = 35 := by
  sorry

end weight_of_b_l3768_376836


namespace dima_speed_ratio_l3768_376819

/-- Represents the time it takes Dima to walk from home to school -/
def walk_time : ℝ := 24

/-- Represents the time it takes Dima to run from home to school -/
def run_time : ℝ := 12

/-- Represents the time remaining before the school bell rings when Dima realizes he forgot his phone -/
def time_remaining : ℝ := 15

/-- States that Dima walks halfway to school before realizing he forgot his phone -/
axiom halfway_condition : walk_time / 2 = time_remaining - 3

/-- States that if Dima runs back home and then to school, he'll be 3 minutes late -/
axiom run_condition : run_time / 2 + run_time = time_remaining + 3

/-- States that if Dima runs back home and then walks to school, he'll be 15 minutes late -/
axiom run_walk_condition : run_time / 2 + walk_time = time_remaining + 15

/-- Theorem stating that Dima's running speed is twice his walking speed -/
theorem dima_speed_ratio : walk_time / run_time = 2 := by sorry

end dima_speed_ratio_l3768_376819


namespace ticket_distribution_l3768_376852

/-- The number of ways to distribute identical objects among people --/
def distribution_methods (n : ℕ) (m : ℕ) : ℕ :=
  if n + 1 = m then m else 0

/-- Theorem: There are 5 ways to distribute 4 identical tickets among 5 people --/
theorem ticket_distribution : distribution_methods 4 5 = 5 := by
  sorry

end ticket_distribution_l3768_376852


namespace ferris_wheel_capacity_l3768_376884

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The weight limit for each large seat (in pounds) -/
def weight_limit_per_seat : ℕ := 1500

/-- The average weight of each person (in pounds) -/
def avg_weight_per_person : ℕ := 180

/-- The maximum number of people that can ride on large seats without violating the weight limit -/
def max_people_on_large_seats : ℕ := 
  (num_large_seats * (weight_limit_per_seat / avg_weight_per_person))

theorem ferris_wheel_capacity : max_people_on_large_seats = 56 := by
  sorry

end ferris_wheel_capacity_l3768_376884


namespace income_calculation_l3768_376894

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given the specified conditions, the person's income is 18000. -/
theorem income_calculation :
  let income_ratio : ℕ := 9
  let expenditure_ratio : ℕ := 8
  let savings : ℕ := 2000
  calculate_income income_ratio expenditure_ratio savings = 18000 := by
  sorry

#eval calculate_income 9 8 2000

end income_calculation_l3768_376894


namespace area_fraction_above_line_l3768_376875

/-- A square with side length 3 -/
def square_side : ℝ := 3

/-- The first point of the line -/
def point1 : ℝ × ℝ := (3, 2)

/-- The second point of the line -/
def point2 : ℝ × ℝ := (6, 0)

/-- The theorem stating that the fraction of the square's area above the line is 2/3 -/
theorem area_fraction_above_line : 
  let square_area := square_side ^ 2
  let triangle_base := point2.1 - point1.1
  let triangle_height := point1.2
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  (area_above_line / square_area) = (2 : ℝ) / 3 := by sorry

end area_fraction_above_line_l3768_376875


namespace feet_per_mile_l3768_376848

/-- Proves that if an object travels 200 feet in 2 seconds with a speed of 68.18181818181819 miles per hour, then there are 5280 feet in one mile. -/
theorem feet_per_mile (distance : ℝ) (time : ℝ) (speed : ℝ) (feet_per_mile : ℝ) :
  distance = 200 →
  time = 2 →
  speed = 68.18181818181819 →
  distance / time = speed * feet_per_mile / 3600 →
  feet_per_mile = 5280 := by
  sorry

end feet_per_mile_l3768_376848


namespace ice_cream_sales_theorem_l3768_376840

def ice_cream_sales (monday tuesday : ℕ) : Prop :=
  ∃ (wednesday thursday total : ℕ),
    wednesday = 2 * tuesday ∧
    thursday = (3 * wednesday) / 2 ∧
    total = monday + tuesday + wednesday + thursday ∧
    total = 82000

theorem ice_cream_sales_theorem :
  ice_cream_sales 10000 12000 := by
  sorry

end ice_cream_sales_theorem_l3768_376840


namespace election_votes_calculation_l3768_376833

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 :=
by
  sorry

end election_votes_calculation_l3768_376833


namespace log_difference_divided_l3768_376899

theorem log_difference_divided : (Real.log 1 - Real.log 25) / 100 = -20 := by sorry

end log_difference_divided_l3768_376899


namespace triangle_side_length_l3768_376839

/-- In a triangle ABC, given that tan B = √3, AB = 3, and the area is (3√3)/2, prove that AC = √7 -/
theorem triangle_side_length (B : Real) (C : Real) (tanB : Real.tan B = Real.sqrt 3) 
  (AB : Real) (hAB : AB = 3) (area : Real) (harea : area = (3 * Real.sqrt 3) / 2) : 
  Real.sqrt ((AB^2) + (2^2) - 2 * AB * 2 * Real.cos B) = Real.sqrt 7 := by
  sorry

end triangle_side_length_l3768_376839


namespace base_85_modulo_17_l3768_376853

theorem base_85_modulo_17 (b : ℕ) : 
  0 ≤ b ∧ b ≤ 16 → (352936524 : ℕ) ≡ b [MOD 17] ↔ b = 4 := by
  sorry

end base_85_modulo_17_l3768_376853


namespace computer_contracts_probability_l3768_376858

theorem computer_contracts_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5) 
  (h2 : p_not_software = 3/5) 
  (h3 : p_at_least_one = 5/6) : 
  p_hardware + (1 - p_not_software) - p_at_least_one = 11/30 :=
by sorry

end computer_contracts_probability_l3768_376858


namespace smallest_possible_abs_z_l3768_376821

theorem smallest_possible_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 17 ∧ Complex.abs w = 7 / Real.sqrt 113 :=
sorry

end smallest_possible_abs_z_l3768_376821


namespace sin_48_greater_cos_48_l3768_376854

theorem sin_48_greater_cos_48 : Real.sin (48 * π / 180) > Real.cos (48 * π / 180) := by
  sorry

end sin_48_greater_cos_48_l3768_376854


namespace two_sector_area_l3768_376822

theorem two_sector_area (r : ℝ) (h : r = 15) : 
  2 * (45 / 360) * (π * r^2) = 56.25 * π := by
  sorry

end two_sector_area_l3768_376822


namespace boat_distance_is_105_l3768_376897

/-- Given a boat traveling downstream, calculate the distance covered. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance covered by the boat downstream is 105 km. -/
theorem boat_distance_is_105 :
  let boat_speed : ℝ := 16
  let stream_speed : ℝ := 5
  let time : ℝ := 5
  distance_downstream boat_speed stream_speed time = 105 := by
  sorry

end boat_distance_is_105_l3768_376897


namespace line_y_intercept_l3768_376824

/-- Given a line with equation 3x - y + 6 = 0, prove that its y-intercept is 6 -/
theorem line_y_intercept (x y : ℝ) (h : 3 * x - y + 6 = 0) : y = 6 ↔ x = 0 :=
sorry

end line_y_intercept_l3768_376824


namespace proportion_equality_l3768_376804

-- Define variables a and b
variable (a b : ℝ)

-- Define the given condition
def condition : Prop := 2 * a = 5 * b

-- State the theorem to be proved
theorem proportion_equality (h : condition a b) : a / 5 = b / 2 := by
  sorry

end proportion_equality_l3768_376804


namespace parabola_focus_l3768_376856

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -4*y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, -1)

/-- Theorem: The focus of the parabola x^2 = -4y is (0, -1) -/
theorem parabola_focus :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = focus :=
sorry

end parabola_focus_l3768_376856


namespace f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l3768_376838

-- Definition of a "cone-bottomed" function
def is_cone_bottomed (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

-- Specific functions
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := x^2 + 1

-- Theorems to prove
theorem f_is_cone_bottomed : is_cone_bottomed f := sorry

theorem g_is_not_cone_bottomed : ¬ is_cone_bottomed g := sorry

theorem h_max_cone_bottomed_constant :
  ∀ M : ℝ, (is_cone_bottomed h ∧ ∀ N : ℝ, is_cone_bottomed h → N ≤ M) → M = 2 := sorry

end f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l3768_376838


namespace quadrilateral_area_is_18_l3768_376803

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) -
             (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

/-- Theorem: The area of the quadrilateral with vertices at (0,0), (4,0), (6,3), and (4,6) is 18 -/
theorem quadrilateral_area_is_18 :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨6, 3⟩
  let p4 : Point := ⟨4, 6⟩
  quadrilateralArea p1 p2 p3 p4 = 18 := by
  sorry

end quadrilateral_area_is_18_l3768_376803


namespace f_properties_l3768_376861

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 5 else -2 * x + 8

theorem f_properties :
  (f 2 = 4) ∧
  (f (f (-1)) = 0) ∧
  (∀ x, f x ≥ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
by sorry

end f_properties_l3768_376861


namespace profit_share_ratio_l3768_376879

def total_profit : ℝ := 500
def share_difference : ℝ := 100

theorem profit_share_ratio :
  ∀ (x y : ℝ),
  x + y = total_profit →
  x - y = share_difference →
  x / total_profit = 3 / 5 :=
by sorry

end profit_share_ratio_l3768_376879


namespace circle_line_intersection_l3768_376883

/-- The circle C₁ -/
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- M is a point on both C₁ and l -/
def M (m : ℝ) : ℝ × ℝ := sorry

/-- N is a point on both C₁ and l, distinct from M -/
def N (m : ℝ) : ℝ × ℝ := sorry

/-- OM is perpendicular to ON -/
def perpendicular (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

theorem circle_line_intersection (m : ℝ) :
  C₁ (M m).1 (M m).2 m ∧
  C₁ (N m).1 (N m).2 m ∧
  l (M m).1 (M m).2 ∧
  l (N m).1 (N m).2 ∧
  perpendicular (M m) (N m) →
  m = 8/5 :=
sorry

end circle_line_intersection_l3768_376883


namespace number_of_children_l3768_376814

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) (h1 : crayons_per_child = 3) (h2 : total_crayons = 18) :
  total_crayons / crayons_per_child = 6 := by
  sorry

end number_of_children_l3768_376814


namespace abs_inequality_solution_set_l3768_376859

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end abs_inequality_solution_set_l3768_376859


namespace min_triangles_to_cover_l3768_376811

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 :=
by sorry

end min_triangles_to_cover_l3768_376811


namespace square_sheet_area_l3768_376827

theorem square_sheet_area (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end square_sheet_area_l3768_376827


namespace smallest_with_twenty_divisors_l3768_376893

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 20 positive divisors -/
def has_twenty_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_twenty_divisors : 
  (∀ m : ℕ+, m < 432 → ¬(has_twenty_divisors m)) ∧ has_twenty_divisors 432 := by sorry

end smallest_with_twenty_divisors_l3768_376893


namespace intersection_M_complement_N_l3768_376867

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 + x = 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1} := by sorry

end intersection_M_complement_N_l3768_376867


namespace running_reduction_is_five_l3768_376818

/-- Carly's running distances over four weeks -/
def running_distances : Fin 4 → ℚ
  | 0 => 2                        -- Week 1: 2 miles
  | 1 => 2 * 2 + 3                -- Week 2: twice as long as week 1 plus 3 extra miles
  | 2 => (2 * 2 + 3) * (9/7)      -- Week 3: 9/7 as much as week 2
  | 3 => 4                        -- Week 4: 4 miles due to injury

/-- The reduction in Carly's running distance when she was injured -/
def running_reduction : ℚ :=
  running_distances 2 - running_distances 3

theorem running_reduction_is_five :
  running_reduction = 5 := by sorry

end running_reduction_is_five_l3768_376818


namespace area_change_not_triple_l3768_376886

theorem area_change_not_triple :
  ∀ (s r : ℝ), s > 0 → r > 0 →
  (3 * s)^2 ≠ 3 * s^2 ∧ π * (3 * r)^2 ≠ 3 * (π * r^2) :=
by sorry

end area_change_not_triple_l3768_376886


namespace box_width_calculation_l3768_376865

/-- Given a rectangular box with specified dimensions and features, calculate its width -/
theorem box_width_calculation (length : ℝ) (road_width : ℝ) (lawn_area : ℝ) : 
  length = 60 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (width : ℝ), width = 37.15 ∧ length * width - 2 * (length / 3) * road_width = lawn_area :=
by sorry

end box_width_calculation_l3768_376865


namespace inequality_solution_set_l3768_376873

theorem inequality_solution_set (x : ℝ) : 
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3/2) := by
  sorry

end inequality_solution_set_l3768_376873


namespace inscribed_octagon_area_l3768_376870

/-- An inscribed convex octagon with alternating side lengths of 2 and 6√2 -/
structure InscribedOctagon where
  -- The octagon is inscribed in a circle (implied by the problem)
  isInscribed : Bool
  -- The octagon is convex
  isConvex : Bool
  -- The octagon has 8 sides
  numSides : Nat
  -- Four sides have length 2
  shortSideLength : ℝ
  -- Four sides have length 6√2
  longSideLength : ℝ
  -- Conditions
  inscribed_condition : isInscribed = true
  convex_condition : isConvex = true
  sides_condition : numSides = 8
  short_side_condition : shortSideLength = 2
  long_side_condition : longSideLength = 6 * Real.sqrt 2

/-- The area of the inscribed convex octagon -/
def area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed convex octagon is 124 -/
theorem inscribed_octagon_area (o : InscribedOctagon) : area o = 124 := by sorry

end inscribed_octagon_area_l3768_376870


namespace partial_fraction_decomposition_l3768_376846

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2 * x^2 + 5 * x - 3) / (x^2 - x - 42)
  let g (x : ℝ) := (11/13) / (x - 7) + (15/13) / (x + 6)
  ∀ x : ℝ, x ≠ 7 → x ≠ -6 → f x = g x :=
by sorry

end partial_fraction_decomposition_l3768_376846


namespace john_total_distance_l3768_376864

-- Define the driving speed
def speed : ℝ := 45

-- Define the first driving duration
def duration1 : ℝ := 2

-- Define the second driving duration
def duration2 : ℝ := 3

-- Theorem to prove
theorem john_total_distance :
  speed * (duration1 + duration2) = 225 := by
  sorry

end john_total_distance_l3768_376864


namespace base_3_8_digit_difference_l3768_376862

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- The theorem stating the difference in the number of digits between base-3 and base-8 representations of 2035 -/
theorem base_3_8_digit_difference :
  numDigits 2035 3 - numDigits 2035 8 = 3 := by
  sorry

end base_3_8_digit_difference_l3768_376862


namespace cosine_two_local_minima_l3768_376826

/-- A function f(x) = cos(ωx) has exactly two local minimum points in [0, π/2] iff 6 ≤ ω < 10 -/
theorem cosine_two_local_minima (ω : ℝ) (h : ω > 0) :
  (∃! (n : ℕ), n = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) →
    (∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), y ∈ Set.Ioo (x - ε) (x + ε) →
      Real.cos (ω * y) ≥ Real.cos (ω * x))) ↔
  6 ≤ ω ∧ ω < 10 :=
sorry

end cosine_two_local_minima_l3768_376826


namespace no_quadratic_trinomial_sequence_with_all_integral_roots_l3768_376876

/-- A sequence of quadratic trinomials -/
def QuadraticTrinomialSequence := ℕ → (ℝ → ℝ)

/-- Condition: P_n is the sum of the two preceding trinomials for n ≥ 3 -/
def IsSumOfPrecedingTrinomials (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, n ≥ 3 → P n = P (n - 1) + P (n - 2)

/-- Condition: P_1 and P_2 do not have common roots -/
def NoCommonRoots (P : QuadraticTrinomialSequence) : Prop :=
  ∀ x : ℝ, P 1 x = 0 → P 2 x ≠ 0

/-- Condition: P_n has at least one integral root for all n -/
def HasIntegralRoot (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, ∃ k : ℤ, P n k = 0

/-- Theorem: There does not exist a sequence of quadratic trinomials satisfying all conditions -/
theorem no_quadratic_trinomial_sequence_with_all_integral_roots :
  ¬ ∃ P : QuadraticTrinomialSequence,
    IsSumOfPrecedingTrinomials P ∧ NoCommonRoots P ∧ HasIntegralRoot P :=
by
  sorry

end no_quadratic_trinomial_sequence_with_all_integral_roots_l3768_376876


namespace video_streaming_cost_theorem_l3768_376834

/-- Calculates the total cost for one person's share of a video streaming subscription over a year -/
theorem video_streaming_cost_theorem 
  (monthly_cost : ℝ) 
  (num_people_sharing : ℕ) 
  (months_in_year : ℕ) 
  (h1 : monthly_cost = 14) 
  (h2 : num_people_sharing = 2) 
  (h3 : months_in_year = 12) :
  (monthly_cost / num_people_sharing) * months_in_year = 84 := by
  sorry

end video_streaming_cost_theorem_l3768_376834


namespace fraction_simplification_l3768_376885

theorem fraction_simplification :
  (18 : ℚ) / 22 * 52 / 24 * 33 / 39 * 22 / 52 = 33 / 52 := by
  sorry

end fraction_simplification_l3768_376885


namespace m_gt_neg_one_sufficient_not_necessary_l3768_376810

/-- Represents the equation of a conic section in the form (x^2 / a) - (y^2 / b) = 1 --/
structure ConicSection where
  a : ℝ
  b : ℝ

/-- Defines when a ConicSection represents a hyperbola --/
def is_hyperbola (c : ConicSection) : Prop :=
  c.a > 0 ∧ c.b > 0

/-- The conic section defined by the given equation --/
def conic_equation (m : ℝ) : ConicSection :=
  { a := 2 + m, b := 1 + m }

/-- The theorem to be proved --/
theorem m_gt_neg_one_sufficient_not_necessary :
  (∀ m : ℝ, m > -1 → is_hyperbola (conic_equation m)) ∧
  ¬(∀ m : ℝ, is_hyperbola (conic_equation m) → m > -1) :=
sorry

end m_gt_neg_one_sufficient_not_necessary_l3768_376810


namespace integer_solution_l3768_376849

theorem integer_solution (x : ℤ) : x + 8 > 9 ∧ -3*x > -15 → x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end integer_solution_l3768_376849


namespace charlene_necklaces_l3768_376809

theorem charlene_necklaces (sold : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : sold = 16) (h2 : given_away = 18) (h3 : left = 26) :
  sold + given_away + left = 60 := by
  sorry

end charlene_necklaces_l3768_376809


namespace binomial_sixteen_nine_l3768_376871

theorem binomial_sixteen_nine (h1 : Nat.choose 15 7 = 6435)
                              (h2 : Nat.choose 15 8 = 6435)
                              (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 16 9 = 11440 := by
  sorry

end binomial_sixteen_nine_l3768_376871


namespace xy_squared_l3768_376892

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y - x - y = 2) : 
  x^2 * y^2 = 1/4 := by
sorry

end xy_squared_l3768_376892


namespace total_precious_stones_l3768_376802

/-- The number of precious stones in agate -/
def agate_stones : ℕ := 30

/-- The number of precious stones in olivine -/
def olivine_stones : ℕ := agate_stones + 5

/-- The number of precious stones in diamond -/
def diamond_stones : ℕ := olivine_stones + 11

/-- The total number of precious stones in agate, olivine, and diamond -/
def total_stones : ℕ := agate_stones + olivine_stones + diamond_stones

theorem total_precious_stones : total_stones = 111 := by
  sorry

end total_precious_stones_l3768_376802


namespace second_rate_is_five_percent_l3768_376832

def total_sum : ℚ := 2678
def second_part : ℚ := 1648
def first_part : ℚ := total_sum - second_part
def first_rate : ℚ := 3 / 100
def first_duration : ℚ := 8
def second_duration : ℚ := 3

def first_interest : ℚ := first_part * first_rate * first_duration

theorem second_rate_is_five_percent : 
  ∃ (second_rate : ℚ), 
    second_rate * 100 = 5 ∧ 
    first_interest = second_part * second_rate * second_duration :=
sorry

end second_rate_is_five_percent_l3768_376832


namespace a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3768_376850

theorem a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (x > y → x^2 > y^2)) ∧
  (a^2 > b^2 ∧ a ≤ b) := by
  sorry

end a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3768_376850


namespace isosceles_triangle_perimeter_l3768_376841

/-- An isosceles triangle with side lengths a and b satisfying a certain equation has perimeter 10 -/
theorem isosceles_triangle_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  (∃ c : ℝ, c > 0 ∧ a + a + c = b + b) → -- Isosceles triangle condition
  2 * Real.sqrt (3 * a - 6) + 3 * Real.sqrt (2 - a) = b - 4 → -- Given equation
  a + a + b = 10 := by -- Perimeter is 10
sorry

end isosceles_triangle_perimeter_l3768_376841


namespace ellipse_major_axis_length_l3768_376806

/-- Given an ellipse defined by the equation 4x² + y² = 16, 
    its major axis has length 8. -/
theorem ellipse_major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, 4 * x^2 + y^2 = 16 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * max a b = 8 :=
sorry

end ellipse_major_axis_length_l3768_376806


namespace fraction_value_l3768_376887

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end fraction_value_l3768_376887


namespace dividend_calculation_l3768_376898

theorem dividend_calculation (x : ℕ) (h : x > 1) :
  let divisor := 3 * x^2
  let quotient := 5 * x
  let remainder := 7 * x + 9
  let dividend := divisor * quotient + remainder
  dividend = 15 * x^3 + 7 * x + 9 := by
sorry

end dividend_calculation_l3768_376898


namespace area_PQR_approx_5_96_l3768_376869

-- Define the square pyramid
def square_pyramid (side_length : ℝ) (height : ℝ) :=
  {base_side : ℝ // base_side = side_length ∧ height > 0}

-- Define points P, Q, and R
def point_P (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_Q (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_R (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry

-- Define the area of triangle PQR
def area_PQR (pyramid : square_pyramid 4 8) : ℝ := sorry

-- Theorem statement
theorem area_PQR_approx_5_96 (pyramid : square_pyramid 4 8) :
  ∃ ε > 0, |area_PQR pyramid - 5.96| < ε :=
sorry

end area_PQR_approx_5_96_l3768_376869


namespace total_missed_pitches_l3768_376830

-- Define the constants from the problem
def pitches_per_token : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def macy_hits : ℕ := 50
def piper_hits : ℕ := 55

-- Theorem statement
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token + piper_tokens * pitches_per_token) - (macy_hits + piper_hits) = 315 := by
  sorry


end total_missed_pitches_l3768_376830


namespace power_of_two_equality_l3768_376878

theorem power_of_two_equality (x : ℕ) : 32^10 + 32^10 + 32^10 + 32^10 + 32^10 = 2^x ↔ x = 52 := by
  sorry

end power_of_two_equality_l3768_376878


namespace solution_set_implies_sum_l3768_376843

/-- If the solution set of (x-a)(x-b) < 0 is (-1,2), then a+b = 1 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, (x-a)*(x-b) < 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end solution_set_implies_sum_l3768_376843


namespace fourth_score_calculation_l3768_376842

theorem fourth_score_calculation (s1 s2 s3 s4 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76)
  (h_average : (s1 + s2 + s3 + s4) / 4 = 75) : s4 = 92 := by
  sorry

end fourth_score_calculation_l3768_376842


namespace system_solution_l3768_376863

theorem system_solution (x y z : ℝ) : 
  x + y + z = 3 ∧ 
  x^2 + y^2 + z^2 = 7 ∧ 
  x^3 + y^3 + z^3 = 15 ↔ 
  (x = 1 ∧ y = 1 + Real.sqrt 2 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 ∧ y = 1 - Real.sqrt 2 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 - Real.sqrt 2 ∧ z = 1) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 ∧ z = 1) :=
by sorry

end system_solution_l3768_376863


namespace ball_bounce_distance_l3768_376866

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let downDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundFactor^i)
  let upDistances := List.range bounces |>.map (fun i => initialHeight * reboundFactor^(i+1))
  (downDistances.sum + upDistances.sum)

/-- The total distance traveled by a ball dropped from 150 feet, rebounding 1/3 of its fall distance each time, after 5 bounces is equal to 298.14 feet -/
theorem ball_bounce_distance :
  totalDistance 150 (1/3) 5 = 298.14 := by
  sorry

end ball_bounce_distance_l3768_376866


namespace stick_division_theorem_l3768_376868

/-- Represents a stick with markings -/
structure MarkedStick where
  divisions : List Nat

/-- Calculates the number of pieces a stick is divided into when cut at all markings -/
def numberOfPieces (stick : MarkedStick) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem stick_division_theorem :
  let stick : MarkedStick := { divisions := [10, 12, 15] }
  numberOfPieces stick = 28 := by
  sorry

end stick_division_theorem_l3768_376868


namespace altitude_equals_harmonic_mean_of_excircle_radii_l3768_376880

/-- For a triangle ABC with altitude h_a from vertex A, area t, semiperimeter s,
    and excircle radii r_b and r_c, the altitude h_a is equal to 2t/a. -/
theorem altitude_equals_harmonic_mean_of_excircle_radii 
  (a b c : ℝ) 
  (h_a : ℝ) 
  (t : ℝ) 
  (s : ℝ) 
  (r_b r_c : ℝ) 
  (h_s : s = (a + b + c) / 2) 
  (h_r_b : r_b = t / (s - b)) 
  (h_r_c : r_c = t / (s - c)) 
  (h_positive : a > 0 ∧ t > 0) : 
  h_a = 2 * t / a := by
  sorry

end altitude_equals_harmonic_mean_of_excircle_radii_l3768_376880


namespace first_day_price_is_four_l3768_376857

/-- Represents the pen sales scenario over three days -/
structure PenSales where
  day1_price : ℝ
  day1_quantity : ℝ

/-- The revenue is the same for all three days -/
def same_revenue (s : PenSales) : Prop :=
  s.day1_price * s.day1_quantity = 
  (s.day1_price - 1) * (s.day1_quantity + 100) ∧
  s.day1_price * s.day1_quantity = 
  (s.day1_price + 2) * (s.day1_quantity - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four :
  ∃ (s : PenSales), same_revenue s ∧ s.day1_price = 4 := by
  sorry

end first_day_price_is_four_l3768_376857


namespace alice_spending_percentage_l3768_376851

theorem alice_spending_percentage (alice_initial bob_initial alice_final : ℝ) :
  bob_initial = 0.9 * alice_initial →
  alice_final = 0.9 * bob_initial →
  (alice_initial - alice_final) / alice_initial = 0.19 := by
  sorry

end alice_spending_percentage_l3768_376851


namespace simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l3768_376813

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := -x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := -3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y := by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -2) :
  2 * A x y - 3 * B x y = 28 := by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 := by sorry

end simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l3768_376813


namespace number_of_keepers_l3768_376807

/-- Represents the number of feet for each animal type --/
def animalFeet : Nat → Nat
| 0 => 2  -- hen
| 1 => 4  -- goat
| 2 => 4  -- camel
| 3 => 8  -- spider
| 4 => 8  -- octopus
| _ => 0

/-- Represents the count of each animal type --/
def animalCount : Nat → Nat
| 0 => 50  -- hens
| 1 => 45  -- goats
| 2 => 8   -- camels
| 3 => 12  -- spiders
| 4 => 6   -- octopuses
| _ => 0

/-- Calculates the total number of animal feet --/
def totalAnimalFeet : Nat :=
  List.range 5
    |> List.map (fun i => animalFeet i * animalCount i)
    |> List.sum

/-- Calculates the total number of animal heads --/
def totalAnimalHeads : Nat :=
  List.range 5
    |> List.map animalCount
    |> List.sum

/-- Theorem stating the number of keepers in the caravan --/
theorem number_of_keepers :
  ∃ k : Nat,
    k = 39 ∧
    totalAnimalFeet + (2 * k - 2) = totalAnimalHeads + k + 372 :=
by
  sorry


end number_of_keepers_l3768_376807


namespace revenue_growth_equation_l3768_376831

theorem revenue_growth_equation (x : ℝ) : 
  let january_revenue : ℝ := 900000
  let total_revenue : ℝ := 1440000
  90000 * (1 + x) + 90000 * (1 + x)^2 = total_revenue - january_revenue :=
by sorry

end revenue_growth_equation_l3768_376831
