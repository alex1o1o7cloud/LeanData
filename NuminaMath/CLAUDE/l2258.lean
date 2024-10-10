import Mathlib

namespace b_investment_is_13650_l2258_225842

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.c_investment

/-- Theorem stating that B's investment is 13650 given the specific partnership details. -/
theorem b_investment_is_13650 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.c_investment = 10500)
  (h3 : p.total_profit = 12100)
  (h4 : p.a_profit_share = 3630) :
  calculate_b_investment p = 13650 := by
  sorry

end b_investment_is_13650_l2258_225842


namespace order_of_a_b_c_l2258_225872

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_a_b_c : a > b ∧ a > c ∧ c > b := by sorry

end order_of_a_b_c_l2258_225872


namespace share_purchase_price_l2258_225816

/-- Calculates the purchase price of shares given dividend rate, par value, and ROI -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (par_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : par_value = 40)
  (h3 : roi = 0.25) : 
  (dividend_rate * par_value) / roi = 20 := by
  sorry

#check share_purchase_price

end share_purchase_price_l2258_225816


namespace binomial_expansion_example_l2258_225835

theorem binomial_expansion_example : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end binomial_expansion_example_l2258_225835


namespace painted_cube_problem_l2258_225824

theorem painted_cube_problem (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → 
  n = 2 ∧ n^3 = 8 := by
sorry

end painted_cube_problem_l2258_225824


namespace lisa_minimum_score_l2258_225819

def minimum_score_for_geometry (term1 term2 term3 term4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (term1 + term2 + term3 + term4)

theorem lisa_minimum_score :
  let term1 := 84
  let term2 := 80
  let term3 := 82
  let term4 := 87
  let required_average := 85
  minimum_score_for_geometry term1 term2 term3 term4 required_average = 92 := by
sorry

end lisa_minimum_score_l2258_225819


namespace odd_function_property_l2258_225810

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end odd_function_property_l2258_225810


namespace car_distance_18_hours_l2258_225884

/-- Calculates the total distance traveled by a car with increasing speed -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let finalSpeed := initialSpeed + speedIncrease * (hours - 1)
  hours * (initialSpeed + finalSpeed) / 2

/-- Theorem stating the total distance traveled by the car in 18 hours -/
theorem car_distance_18_hours :
  totalDistance 30 5 18 = 1305 := by
  sorry

end car_distance_18_hours_l2258_225884


namespace fundraiser_total_l2258_225846

/-- Calculates the total amount raised from cake sales and donations --/
def total_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
                 (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales := total_slices * price_per_slice
  let donation1 := total_slices * donation1_per_slice
  let donation2 := total_slices * donation2_per_slice
  sales + donation1 + donation2

/-- Theorem stating that under given conditions, the total amount raised is $140 --/
theorem fundraiser_total : 
  total_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end fundraiser_total_l2258_225846


namespace function_strictly_increasing_iff_a_in_range_l2258_225803

/-- The function f(x) = (a-2)a^x is strictly increasing if and only if a is in the set (0,1) ∪ (2,+∞) -/
theorem function_strictly_increasing_iff_a_in_range (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (((a - 2) * a^x₁ - (a - 2) * a^x₂) / (x₁ - x₂)) > 0) ↔
  (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 2) :=
sorry

end function_strictly_increasing_iff_a_in_range_l2258_225803


namespace museum_visit_arrangements_l2258_225869

theorem museum_visit_arrangements (n m : ℕ) (hn : n = 6) (hm : m = 6) : 
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 15 * 625 := by
  sorry

end museum_visit_arrangements_l2258_225869


namespace time_difference_to_halfway_l2258_225823

/-- Time difference for Steve and Danny to reach halfway point -/
theorem time_difference_to_halfway (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 31 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 15.5 := by
sorry

end time_difference_to_halfway_l2258_225823


namespace cos_2alpha_from_tan_alpha_plus_pi_4_l2258_225851

theorem cos_2alpha_from_tan_alpha_plus_pi_4 (α : Real) 
  (h : Real.tan (α + π/4) = 2) : 
  Real.cos (2 * α) = 4/5 := by
  sorry

end cos_2alpha_from_tan_alpha_plus_pi_4_l2258_225851


namespace soccer_stars_points_l2258_225807

/-- Calculates the total points for a soccer team given their game results -/
def calculate_total_points (total_games wins losses : ℕ) : ℕ :=
  let draws := total_games - wins - losses
  let points_per_win := 3
  let points_per_draw := 1
  let points_per_loss := 0
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

/-- Theorem stating that the Soccer Stars team's total points is 46 -/
theorem soccer_stars_points :
  calculate_total_points 20 14 2 = 46 := by
  sorry

end soccer_stars_points_l2258_225807


namespace sufficient_not_necessary_condition_l2258_225879

theorem sufficient_not_necessary_condition (x : ℝ) (h : x > 0) :
  (x + 1 / x ≥ 2) ∧ (∃ a : ℝ, a > 1 ∧ ∀ y : ℝ, y > 0 → y + a / y ≥ 2) :=
by sorry

end sufficient_not_necessary_condition_l2258_225879


namespace arm_wrestling_streaks_l2258_225885

/-- Represents the outcome of a single round of arm wrestling -/
inductive Winner : Type
| Richard : Winner
| Shreyas : Winner

/-- Counts the number of streaks in a list of outcomes -/
def count_streaks (outcomes : List Winner) : Nat :=
  sorry

/-- Generates all possible outcomes for n rounds of arm wrestling -/
def generate_outcomes (n : Nat) : List (List Winner) :=
  sorry

/-- Counts the number of outcomes with more than k streaks in n rounds -/
def count_outcomes_with_more_than_k_streaks (n k : Nat) : Nat :=
  sorry

theorem arm_wrestling_streaks :
  count_outcomes_with_more_than_k_streaks 10 3 = 932 :=
sorry

end arm_wrestling_streaks_l2258_225885


namespace cube_root_equation_solution_l2258_225867

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l2258_225867


namespace union_of_A_and_B_l2258_225881

def A : Set Int := {-1, 0, 2}
def B : Set Int := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end union_of_A_and_B_l2258_225881


namespace jeff_vehicle_collection_l2258_225843

theorem jeff_vehicle_collection (trucks : ℕ) : 
  let cars := 2 * trucks
  trucks + cars = 3 * trucks := by
sorry

end jeff_vehicle_collection_l2258_225843


namespace triangle_proof_l2258_225876

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.BC = 7 ∧ t.AB = 3 ∧ (Real.sin t.C) / (Real.sin t.B) = 3/5

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : triangle_properties t) :
  t.AC = 5 ∧ t.A = Real.pi * 2/3 := by
  sorry

end triangle_proof_l2258_225876


namespace perpendicular_lines_from_perpendicular_planes_l2258_225870

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at_line (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perpendicular_to_line (n l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (l m n : Line)
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at_line α β l)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β) :
  line_perpendicular_to_line n l :=
sorry

end perpendicular_lines_from_perpendicular_planes_l2258_225870


namespace angle_measure_in_triangle_l2258_225868

/-- Given a triangle DEF where the measure of angle D is three times the measure of angle F,
    and angle F measures 18°, prove that the measure of angle E is 108°. -/
theorem angle_measure_in_triangle (D E F : ℝ) (h1 : D = 3 * F) (h2 : F = 18) :
  E = 108 := by
  sorry

end angle_measure_in_triangle_l2258_225868


namespace perpendicular_lines_sum_l2258_225860

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def lies_on (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular (-a/4) (2/5) →
  lies_on 1 c a 4 (-2) →
  lies_on 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end perpendicular_lines_sum_l2258_225860


namespace additional_bottles_needed_l2258_225804

/-- Represents the number of bottles in a case of water -/
def bottles_per_case : ℕ := 24

/-- Represents the number of cases purchased -/
def cases_purchased : ℕ := 13

/-- Represents the duration of the camp in days -/
def camp_duration : ℕ := 3

/-- Represents the number of children in the first group -/
def group1_children : ℕ := 14

/-- Represents the number of children in the second group -/
def group2_children : ℕ := 16

/-- Represents the number of children in the third group -/
def group3_children : ℕ := 12

/-- Represents the number of bottles consumed by each child per day -/
def bottles_per_child_per_day : ℕ := 3

/-- Calculates the total number of children in the camp -/
def total_children : ℕ :=
  let first_three := group1_children + group2_children + group3_children
  first_three + first_three / 2

/-- Calculates the total number of bottles needed for the entire camp -/
def total_bottles_needed : ℕ :=
  total_children * bottles_per_child_per_day * camp_duration

/-- Calculates the number of bottles already purchased -/
def bottles_purchased : ℕ :=
  cases_purchased * bottles_per_case

/-- Theorem stating that 255 additional bottles are needed -/
theorem additional_bottles_needed : 
  total_bottles_needed - bottles_purchased = 255 := by
  sorry

end additional_bottles_needed_l2258_225804


namespace arithmetic_sequence_common_difference_l2258_225833

-- Define an arithmetic sequence with first term a₁ and common difference d
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, ∀ n : ℕ, arithmeticSequence 2 d n = 2 + (n - 1) * d ∧ arithmeticSequence 2 d 2 = 1 :=
by sorry

end arithmetic_sequence_common_difference_l2258_225833


namespace implication_equivalence_l2258_225821

theorem implication_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) := by sorry

end implication_equivalence_l2258_225821


namespace trajectory_and_intersection_l2258_225834

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The condition for point P -/
def point_condition (x y : ℝ) : Prop :=
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2*(x + 1)

/-- The perpendicularity condition for OM and ON -/
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Point P satisfies the condition
  ∀ x y : ℝ, point_condition x y →
  -- The trajectory is y² = 4x
  (trajectory x y) ∧
  -- For any non-zero m where y = x + m intersects the trajectory at M and N
  ∀ m : ℝ, m ≠ 0 →
    ∃ x₁ y₁ x₂ y₂ : ℝ,
      -- M and N are on the trajectory
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      -- M and N are on the line y = x + m
      y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
      -- OM is perpendicular to ON
      perpendicular_condition x₁ y₁ x₂ y₂ →
      -- Then m = -4
      m = -4 :=
sorry

end trajectory_and_intersection_l2258_225834


namespace peter_marbles_l2258_225861

/-- The number of marbles Peter lost -/
def lost_marbles : ℕ := 15

/-- The number of marbles Peter currently has -/
def current_marbles : ℕ := 18

/-- The initial number of marbles Peter had -/
def initial_marbles : ℕ := lost_marbles + current_marbles

theorem peter_marbles : initial_marbles = 33 := by
  sorry

end peter_marbles_l2258_225861


namespace total_onions_grown_l2258_225894

theorem total_onions_grown (sara_onions sally_onions fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 := by
sorry

end total_onions_grown_l2258_225894


namespace absent_students_sum_l2258_225814

/-- Proves that the sum of absent students over three days equals 200 --/
theorem absent_students_sum (T : ℕ) (A1 A2 A3 : ℕ) : 
  T = 280 →
  A3 = T / 7 →
  A2 = 2 * A3 →
  T - A2 + 40 = T - A1 →
  A1 + A2 + A3 = 200 := by
  sorry

end absent_students_sum_l2258_225814


namespace cow_field_difference_l2258_225858

theorem cow_field_difference (total : ℕ) (males : ℕ) (females : ℕ) : 
  total = 300 →
  females = 2 * males →
  total = males + females →
  (females / 2 : ℕ) - (males / 2 : ℕ) = 50 := by
  sorry

end cow_field_difference_l2258_225858


namespace circle_and_line_properties_l2258_225852

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k*(x + 1) + 1

-- Theorem statement
theorem circle_and_line_properties :
  -- 1. The center of circle C is (1, 0)
  (∃ r : ℝ, ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + y^2 = r^2) ∧
  -- 2. The point (-1, 1) lies on line l for any real k
  (∀ k : ℝ, line_l k (-1) 1) ∧
  -- 3. Line l intersects circle C for any real k
  (∀ k : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l k x y) :=
by sorry

end circle_and_line_properties_l2258_225852


namespace sum_of_x_solutions_l2258_225839

theorem sum_of_x_solutions (x y : ℝ) : 
  y = 8 → x^2 + y^2 = 144 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ (x = x₁ ∨ x = x₂) := by
  sorry

end sum_of_x_solutions_l2258_225839


namespace quadratic_equation_solution_l2258_225836

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6) ∧ 
  (x₁^2 + 2*x₁ - 5 = 0 ∧ x₂^2 + 2*x₂ - 5 = 0) := by
  sorry

end quadratic_equation_solution_l2258_225836


namespace saras_high_school_basketball_games_l2258_225892

theorem saras_high_school_basketball_games 
  (defeated_games won_games total_games : ℕ) : 
  defeated_games = 4 → 
  won_games = 8 → 
  total_games = defeated_games + won_games → 
  total_games = 12 :=
by sorry

end saras_high_school_basketball_games_l2258_225892


namespace cylinder_height_l2258_225874

/-- The height of a right cylinder with radius 2 feet and surface area 12π square feet is 1 foot. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  (2 * π * 2^2 + 2 * π * 2 * h = 12 * π) → h = 1 :=
by sorry

end cylinder_height_l2258_225874


namespace distribute_10_4_l2258_225813

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem distribute_10_4 : distribute 10 4 = 286 := by
  sorry

end distribute_10_4_l2258_225813


namespace sin_cos_sum_equals_sqrt_three_half_l2258_225888

theorem sin_cos_sum_equals_sqrt_three_half : 
  Real.sin (10 * Real.pi / 180) * Real.cos (50 * Real.pi / 180) + 
  Real.cos (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt_three_half_l2258_225888


namespace john_slurpees_l2258_225880

def slurpee_problem (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : ℕ :=
  (money_given - change) / slurpee_cost

theorem john_slurpees :
  slurpee_problem 20 2 8 = 6 :=
by sorry

end john_slurpees_l2258_225880


namespace min_value_sum_equality_condition_l2258_225844

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) = 3 / Real.rpow 90 (1/3) ↔
  (a / (3 * b)) = (b / (5 * c)) ∧ (b / (5 * c)) = (c / (6 * a)) ∧ 
  (c / (6 * a)) = Real.rpow (1/90) (1/3) :=
sorry

end min_value_sum_equality_condition_l2258_225844


namespace two_by_two_paper_covers_nine_vertices_l2258_225865

/-- Represents a square paper on a grid -/
structure SquarePaper where
  side_length : ℕ
  min_vertices_covered : ℕ

/-- Counts the number of vertices covered by a square paper on a grid -/
def count_vertices_covered (paper : SquarePaper) : ℕ :=
  (paper.side_length + 1) ^ 2

/-- Theorem: A 2x2 square paper covering at least 7 vertices covers exactly 9 vertices -/
theorem two_by_two_paper_covers_nine_vertices (paper : SquarePaper)
  (h1 : paper.side_length = 2)
  (h2 : paper.min_vertices_covered ≥ 7) :
  count_vertices_covered paper = 9 := by
  sorry

end two_by_two_paper_covers_nine_vertices_l2258_225865


namespace consecutive_integers_reciprocal_sum_l2258_225827

/-- The sum of reciprocals of all pairs of three consecutive integers is an integer -/
def is_sum_reciprocals_integer (x : ℤ) : Prop :=
  ∃ (n : ℤ), (x / (x + 1) : ℚ) + (x / (x + 2) : ℚ) + ((x + 1) / x : ℚ) + 
             ((x + 1) / (x + 2) : ℚ) + ((x + 2) / x : ℚ) + ((x + 2) / (x + 1) : ℚ) = n

/-- The only sets of three consecutive integers satisfying the condition are {1, 2, 3} and {-3, -2, -1} -/
theorem consecutive_integers_reciprocal_sum :
  ∀ x : ℤ, is_sum_reciprocals_integer x ↔ (x = 1 ∨ x = -3) :=
sorry

end consecutive_integers_reciprocal_sum_l2258_225827


namespace binomial_ratio_equals_one_l2258_225859

-- Define the binomial coefficient for real numbers
noncomputable def binomial (r : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else (r * binomial (r - 1) (k - 1)) / k

-- State the theorem
theorem binomial_ratio_equals_one :
  (binomial (1/2 : ℝ) 1000 * 4^1000) / binomial 2000 1000 = 1 := by
  sorry

end binomial_ratio_equals_one_l2258_225859


namespace simplify_sqrt_expression_l2258_225801

theorem simplify_sqrt_expression :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 - 2 * Real.sqrt 80 = -6 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_expression_l2258_225801


namespace min_value_of_sum_of_squares_l2258_225826

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x - y - 2 * z = 6) :
  x^2 + y^2 + z^2 ≥ 4 := by
sorry

end min_value_of_sum_of_squares_l2258_225826


namespace jerky_order_fulfillment_l2258_225829

/-- The number of days needed to fulfill a jerky order -/
def days_to_fulfill_order (bags_per_batch : ℕ) (order_size : ℕ) (bags_in_stock : ℕ) : ℕ :=
  let bags_to_make := order_size - bags_in_stock
  (bags_to_make + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 10 60 20 = 4 := by
  sorry

end jerky_order_fulfillment_l2258_225829


namespace illuminated_part_depends_on_position_l2258_225857

/-- Represents a right circular cone on a plane -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone

/-- Represents the position of a light source -/
structure LightSource where
  H : ℝ  -- distance from the plane
  l : ℝ  -- distance from the height of the cone

/-- Represents the illuminated part of a circle -/
structure IlluminatedPart where
  angle : ℝ  -- angle of the illuminated arc

/-- Calculates the illuminated part of a circle with radius R on the plane -/
noncomputable def calculateIlluminatedPart (cone : Cone) (light : LightSource) (R : ℝ) : IlluminatedPart :=
  sorry

/-- Theorem stating that the illuminated part can be determined by the relative position of the light source -/
theorem illuminated_part_depends_on_position (cone : Cone) (light : LightSource) (R : ℝ) :
  ∃ (ip : IlluminatedPart), ip = calculateIlluminatedPart cone light R ∧
  (light.H > cone.h ∨ light.H = cone.h ∨ light.H < cone.h) :=
  sorry

end illuminated_part_depends_on_position_l2258_225857


namespace average_of_geometric_sequence_l2258_225862

/-- The average of the numbers 5y, 10y, 20y, 40y, and 80y is equal to 31y -/
theorem average_of_geometric_sequence (y : ℝ) : 
  (5*y + 10*y + 20*y + 40*y + 80*y) / 5 = 31*y := by
  sorry

end average_of_geometric_sequence_l2258_225862


namespace ship_meetings_count_l2258_225849

/-- Represents the number of ships sailing in each direction -/
def num_ships_per_direction : ℕ := 5

/-- Represents the total number of ships -/
def total_ships : ℕ := 2 * num_ships_per_direction

/-- Calculates the total number of meetings between ships -/
def total_meetings : ℕ := num_ships_per_direction * num_ships_per_direction

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count :
  total_meetings = 25 :=
by sorry

end ship_meetings_count_l2258_225849


namespace sum_coefficients_without_x_cubed_l2258_225812

theorem sum_coefficients_without_x_cubed : 
  let n : ℕ := 5
  let all_coeff_sum : ℕ := 2^n
  let x_cubed_coeff : ℕ := n.choose 3
  all_coeff_sum - x_cubed_coeff = 22 := by
  sorry

end sum_coefficients_without_x_cubed_l2258_225812


namespace tetrahedron_count_is_twelve_l2258_225895

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 6

/-- The number of ways to choose 4 vertices from 6 -/
def choose_four (prism : RegularTriangularPrism) : Nat :=
  Nat.choose 6 4

/-- The number of cases where 4 chosen points are coplanar -/
def coplanar_cases : Nat := 3

/-- The number of tetrahedrons that can be formed -/
def tetrahedron_count (prism : RegularTriangularPrism) : Nat :=
  choose_four prism - coplanar_cases

/-- Theorem: The number of tetrahedrons is 12 -/
theorem tetrahedron_count_is_twelve (prism : RegularTriangularPrism) :
  tetrahedron_count prism = 12 := by
  sorry

end tetrahedron_count_is_twelve_l2258_225895


namespace square_remainder_mod_nine_l2258_225854

theorem square_remainder_mod_nine (N : ℤ) : 
  (N % 9 = 2 ∨ N % 9 = 7) → (N^2 % 9 = 4) := by
  sorry

end square_remainder_mod_nine_l2258_225854


namespace smallest_c_value_l2258_225815

theorem smallest_c_value (c d : ℤ) : 
  (∃ (r₁ r₂ r₃ : ℤ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ (x : ℤ), x^3 - c*x^2 + d*x - 3990 = (x - r₁) * (x - r₂) * (x - r₃)) →
  c ≥ 56 :=
by sorry

end smallest_c_value_l2258_225815


namespace limit_expression_equals_six_l2258_225871

theorem limit_expression_equals_six :
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ →
    |((3 + h)^2 - 3^2) / h - 6| < ε :=
by sorry

end limit_expression_equals_six_l2258_225871


namespace expression_evaluation_l2258_225856

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -2) :
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end expression_evaluation_l2258_225856


namespace triangle_perimeter_bound_l2258_225896

theorem triangle_perimeter_bound :
  ∀ (a b c : ℝ),
  a = 7 →
  b ≥ 14 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c < 42 :=
by sorry

end triangle_perimeter_bound_l2258_225896


namespace parallelepiped_sphere_properties_l2258_225890

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Check if a sphere touches an edge of the parallelepiped -/
def touchesEdge (s : Sphere) (p1 p2 : Point3D) : Prop := sorry

/-- Check if a point is on an edge of the parallelepiped -/
def onEdge (p : Point3D) (p1 p2 : Point3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculate the volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

theorem parallelepiped_sphere_properties 
  (p : Parallelepiped) (s : Sphere) (K : Point3D) :
  (distance p.A p.A1 = distance p.B p.C) →  -- A₁A perpendicular to ABCD face
  (touchesEdge s p.B p.B1) →
  (touchesEdge s p.B1 p.C1) →
  (touchesEdge s p.C1 p.C) →
  (touchesEdge s p.C p.B) →
  (touchesEdge s p.C p.D) →
  (touchesEdge s p.A1 p.D1) →
  (onEdge K p.C p.D) →
  (distance p.C K = 4) →
  (distance K p.D = 1) →
  (distance p.A p.A1 = 8) ∧
  (volume p = 256) ∧
  (s.radius = 2 * Real.sqrt 5) := by sorry

end parallelepiped_sphere_properties_l2258_225890


namespace parallelogram_construction_l2258_225899

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the predicates
variable (lies_on : Point → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (is_center : Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallelogram_construction
  (l₁ l₂ l₃ l₄ : Line)
  (O : Point)
  (not_parallel : ¬ parallel l₁ l₂ ∧ ¬ parallel l₁ l₃ ∧ ¬ parallel l₁ l₄ ∧
                  ¬ parallel l₂ l₃ ∧ ¬ parallel l₂ l₄ ∧ ¬ parallel l₃ l₄)
  (O_not_on_lines : ¬ lies_on O l₁ ∧ ¬ lies_on O l₂ ∧ ¬ lies_on O l₃ ∧ ¬ lies_on O l₄) :
  ∃ (A B C D : Point),
    lies_on A l₁ ∧ lies_on B l₂ ∧ lies_on C l₃ ∧ lies_on D l₄ ∧
    is_center O A B C D :=
by sorry

end parallelogram_construction_l2258_225899


namespace certain_number_proof_l2258_225855

theorem certain_number_proof : ∃ n : ℝ, n = 36 ∧ n + 3 * 4.0 = 48 := by
  sorry

end certain_number_proof_l2258_225855


namespace unique_five_digit_number_l2258_225883

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_five_digit_number : ∃! n : ℕ, 
  is_five_digit n ∧ 
  (∃ pos : Fin 5, n + remove_digit n pos = 54321) :=
by
  use 49383
  sorry

end unique_five_digit_number_l2258_225883


namespace isabel_candy_theorem_l2258_225845

/-- The number of candy pieces Isabel has left after distribution -/
def remaining_candy (initial : ℕ) (friend : ℕ) (cousin : ℕ) (sister : ℕ) (distributed : ℕ) : ℤ :=
  (initial + friend + cousin + sister : ℤ) - distributed

/-- Theorem stating the number of candy pieces Isabel has left -/
theorem isabel_candy_theorem (x y z : ℕ) :
  remaining_candy 325 145 x y z = 470 + x + y - z := by
  sorry

end isabel_candy_theorem_l2258_225845


namespace problem_2017_l2258_225822

theorem problem_2017 : (2017^2 - 2017 + 1) / 2017 = 2016 + 1 / 2017 := by
  sorry

end problem_2017_l2258_225822


namespace susy_initial_followers_l2258_225864

/-- Represents the number of followers gained by a student over three weeks -/
structure FollowerGain where
  week1 : ℕ
  week2 : ℕ
  week3 : ℕ

/-- Represents a student with their school size and follower information -/
structure Student where
  schoolSize : ℕ
  initialFollowers : ℕ
  followerGain : FollowerGain

def totalFollowersAfterThreeWeeks (student : Student) : ℕ :=
  student.initialFollowers + student.followerGain.week1 + student.followerGain.week2 + student.followerGain.week3

theorem susy_initial_followers
  (susy : Student)
  (sarah : Student)
  (h1 : susy.schoolSize = 800)
  (h2 : sarah.schoolSize = 300)
  (h3 : susy.followerGain.week1 = 40)
  (h4 : susy.followerGain.week2 = susy.followerGain.week1 / 2)
  (h5 : susy.followerGain.week3 = susy.followerGain.week2 / 2)
  (h6 : sarah.initialFollowers = 50)
  (h7 : max (totalFollowersAfterThreeWeeks susy) (totalFollowersAfterThreeWeeks sarah) = 180) :
  susy.initialFollowers = 110 := by
  sorry

end susy_initial_followers_l2258_225864


namespace min_value_sum_reciprocals_l2258_225838

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  (1 / (a + 3 * b)) + (1 / (b + 3 * c)) + (1 / (c + 3 * a)) ≥ 3 := by
  sorry

end min_value_sum_reciprocals_l2258_225838


namespace gcd_m5_plus_125_m_plus_3_l2258_225887

theorem gcd_m5_plus_125_m_plus_3 (m : ℕ) (h : m > 16) :
  Nat.gcd (m^5 + 5^3) (m + 3) = if (m + 3) % 27 ≠ 0 then 1 else Nat.gcd 27 (m + 3) := by
  sorry

end gcd_m5_plus_125_m_plus_3_l2258_225887


namespace triangle_properties_l2258_225850

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  c = Real.sqrt 3 →
  c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A) →
  C = π/3 ∧ 0 < a - b/2 ∧ a - b/2 < 3/2 := by
  sorry


end triangle_properties_l2258_225850


namespace circle_in_rectangle_ratio_l2258_225863

theorem circle_in_rectangle_ratio (r s : ℝ) (h1 : r > 0) (h2 : s > 0) : 
  (π * r^2 = 2 * r * s - π * r^2) → (s / (2 * r) = π / 2) := by
  sorry

end circle_in_rectangle_ratio_l2258_225863


namespace problem_statement_l2258_225831

theorem problem_statement (a b q r : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_division : a^2 + b^2 = q * (a + b) + r) (h_constraint : q^2 + r = 2010) :
  a * b = 1643 := by
  sorry

end problem_statement_l2258_225831


namespace painting_price_decrease_l2258_225877

theorem painting_price_decrease (original_price : ℝ) (h_positive : original_price > 0) :
  let first_year_price := original_price * 1.25
  let final_price := original_price * 1.0625
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end painting_price_decrease_l2258_225877


namespace gas_cost_calculation_l2258_225882

theorem gas_cost_calculation (total_cost : ℚ) : 
  (total_cost / 5 - 15 = total_cost / 7) → 
  total_cost = 262.5 := by
sorry

end gas_cost_calculation_l2258_225882


namespace ellipse_major_axis_length_l2258_225847

-- Define the ellipse
structure Ellipse where
  isTangentToXAxis : Bool
  isTangentToYAxis : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

-- Define the theorem
theorem ellipse_major_axis_length 
  (e : Ellipse) 
  (h1 : e.isTangentToXAxis = true) 
  (h2 : e.isTangentToYAxis = true)
  (h3 : e.focus1 = (2, -3 + Real.sqrt 13))
  (h4 : e.focus2 = (2, -3 - Real.sqrt 13)) :
  ∃ (majorAxisLength : ℝ), majorAxisLength = 6 :=
sorry

end ellipse_major_axis_length_l2258_225847


namespace apple_cost_price_l2258_225848

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 →
  loss_fraction = 1/6 →
  selling_price = cost_price - (loss_fraction * cost_price) →
  cost_price = 21.6 := by
sorry

end apple_cost_price_l2258_225848


namespace delta_y_over_delta_x_l2258_225866

/-- Given a function f(x) = 2x² + 1, prove that Δy/Δx = 4 + 2Δx for points P(1, 3) and Q(1 + Δx, 3 + Δy) -/
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 1
  Δy = f (1 + Δx) - f 1 →
  Δx ≠ 0 →
  Δy / Δx = 4 + 2 * Δx := by
sorry

end delta_y_over_delta_x_l2258_225866


namespace unknown_number_problem_l2258_225837

theorem unknown_number_problem (x : ℚ) : (2 / 3) * x + 6 = 10 → x = 6 := by
  sorry

end unknown_number_problem_l2258_225837


namespace complement_of_M_l2258_225853

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 5}

theorem complement_of_M :
  (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_l2258_225853


namespace quadratic_function_property_l2258_225840

/-- A quadratic function with vertex (m, k) and point (k, m) on its graph -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  k : ℝ
  a_nonzero : a ≠ 0
  vertex_condition : k = a * m^2 + b * m + c
  point_condition : m = a * k^2 + b * k + c

/-- Theorem stating that a(m - k) > 0 for a quadratic function with the given conditions -/
theorem quadratic_function_property (f : QuadraticFunction) : f.a * (f.m - f.k) > 0 := by
  sorry

end quadratic_function_property_l2258_225840


namespace sum_of_squares_l2258_225897

theorem sum_of_squares (x y z : ℝ) 
  (h1 : x^2 - 6*y = 10)
  (h2 : y^2 - 8*z = -18)
  (h3 : z^2 - 10*x = -40) :
  x^2 + y^2 + z^2 = 50 := by
sorry

end sum_of_squares_l2258_225897


namespace inequality_solution_l2258_225808

theorem inequality_solution (x : ℝ) : (x^2 - 1) / ((x + 2)^2) ≥ 0 ↔ 
  x < -2 ∨ (-2 < x ∧ x ≤ -1) ∨ x ≥ 1 := by sorry

end inequality_solution_l2258_225808


namespace problem_1_problem_2_l2258_225802

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end problem_1_problem_2_l2258_225802


namespace apple_sorting_probability_l2258_225873

def ratio_large_to_small : ℚ := 9 / 1
def prob_large_to_small : ℚ := 5 / 100
def prob_small_to_large : ℚ := 2 / 100

theorem apple_sorting_probability : 
  let total_apples := ratio_large_to_small + 1
  let prob_large := ratio_large_to_small / total_apples
  let prob_small := 1 / total_apples
  let prob_large_sorted_large := 1 - prob_large_to_small
  let prob_small_sorted_large := prob_small_to_large
  let prob_sorted_large := prob_large * prob_large_sorted_large + prob_small * prob_small_sorted_large
  let prob_large_and_sorted_large := prob_large * prob_large_sorted_large
  (prob_large_and_sorted_large / prob_sorted_large) = 855 / 857 :=
by sorry

end apple_sorting_probability_l2258_225873


namespace area_FYG_is_86_4_l2258_225898

/-- A trapezoid with the given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of triangle FYG is 86.4 square units -/
theorem area_FYG_is_86_4 (t : Trapezoid) 
  (h1 : t.EF = 24)
  (h2 : t.GH = 36)
  (h3 : t.area = 360) :
  area_FYG t = 86.4 := by sorry

end area_FYG_is_86_4_l2258_225898


namespace linear_function_not_in_quadrant_III_l2258_225889

/-- A linear function with slope m and y-intercept b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => ∃ x y, x > 0 ∧ y > 0 ∧ y = f.m * x + f.b
  | Quadrant.II => ∃ x y, x < 0 ∧ y > 0 ∧ y = f.m * x + f.b
  | Quadrant.III => ∃ x y, x < 0 ∧ y < 0 ∧ y = f.m * x + f.b
  | Quadrant.IV => ∃ x y, x > 0 ∧ y < 0 ∧ y = f.m * x + f.b

theorem linear_function_not_in_quadrant_III (f : LinearFunction)
  (h1 : f.m < 0)
  (h2 : f.b > 0) :
  ¬(passesThrough f Quadrant.III) :=
sorry

end linear_function_not_in_quadrant_III_l2258_225889


namespace yoongis_subtraction_mistake_l2258_225818

theorem yoongis_subtraction_mistake (A B : ℕ) : 
  A ≥ 1 ∧ A ≤ 9 ∧ B = 9 ∧ 
  (10 * A + 6) - 57 = 39 →
  10 * A + B = 99 := by
sorry

end yoongis_subtraction_mistake_l2258_225818


namespace juice_price_proof_l2258_225875

def total_paid : ℚ := 370 / 100
def muffin_price : ℚ := 75 / 100
def muffin_count : ℕ := 3

theorem juice_price_proof :
  total_paid - (muffin_price * muffin_count) = 145 / 100 := by
  sorry

end juice_price_proof_l2258_225875


namespace area_of_large_rectangle_l2258_225886

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of large rectangle formed by three identical smaller rectangles -/
theorem area_of_large_rectangle (small_rect : Rectangle) 
    (h1 : small_rect.width = 7)
    (h2 : small_rect.height ≥ small_rect.width) : 
  (Rectangle.area { width := 3 * small_rect.height, height := small_rect.width }) = 294 := by
  sorry

#check area_of_large_rectangle

end area_of_large_rectangle_l2258_225886


namespace same_terminal_side_l2258_225828

theorem same_terminal_side (θ : ℝ) : ∃ k : ℤ, θ = (23 * π / 3 : ℝ) + 2 * π * k ↔ θ = (5 * π / 3 : ℝ) + 2 * π * k := by
  sorry

end same_terminal_side_l2258_225828


namespace complement_of_M_l2258_225811

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | (x - 1) * (x - 4) = 0}

theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 2 ∨ x = 3 := by
  sorry

end complement_of_M_l2258_225811


namespace soda_survey_result_l2258_225830

/-- The number of people who chose "Soda" in a survey of 520 people,
    where the central angle of the "Soda" sector is 270° (to the nearest whole degree). -/
def soda_count : ℕ := 390

/-- The total number of people surveyed. -/
def total_surveyed : ℕ := 520

/-- The central angle of the "Soda" sector in degrees. -/
def soda_angle : ℕ := 270

theorem soda_survey_result :
  (soda_count : ℚ) / total_surveyed * 360 ≥ soda_angle - (1/2 : ℚ) ∧
  (soda_count : ℚ) / total_surveyed * 360 < soda_angle + (1/2 : ℚ) :=
sorry

end soda_survey_result_l2258_225830


namespace geometric_series_ratio_l2258_225800

/-- Given a geometric series with positive terms {a_n}, if a_1, 1/2 * a_3, and 2 * a_2 form an arithmetic sequence, then a_5 / a_3 = 3 + 2√2 -/
theorem geometric_series_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : (a 1 + 2 * a 2) / 2 = a 3 / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_series_ratio_l2258_225800


namespace oblique_square_area_theorem_main_theorem_l2258_225891

/-- Represents a square in an oblique projection --/
structure ObliqueSquare where
  side : ℝ
  projectedSide : ℝ

/-- The area of the original square given its oblique projection --/
def originalArea (s : ObliqueSquare) : ℝ := s.side * s.side

/-- Theorem stating that for a square with a projected side of 4 units,
    the area of the original square can be either 16 or 64 --/
theorem oblique_square_area_theorem (s : ObliqueSquare) 
  (h : s.projectedSide = 4) :
  originalArea s = 16 ∨ originalArea s = 64 := by
  sorry

/-- Main theorem combining the above results --/
theorem main_theorem : 
  ∃ (s1 s2 : ObliqueSquare), 
    s1.projectedSide = 4 ∧ 
    s2.projectedSide = 4 ∧ 
    originalArea s1 = 16 ∧ 
    originalArea s2 = 64 := by
  sorry

end oblique_square_area_theorem_main_theorem_l2258_225891


namespace new_clock_conversion_l2258_225805

/-- Represents a time on the new clock -/
structure NewClockTime where
  hours : ℕ
  minutes : ℕ

/-- Represents a time in Beijing -/
structure BeijingTime where
  hours : ℕ
  minutes : ℕ

/-- Converts NewClockTime to total minutes -/
def newClockToMinutes (t : NewClockTime) : ℕ :=
  t.hours * 100 + t.minutes

/-- Converts BeijingTime to total minutes -/
def beijingToMinutes (t : BeijingTime) : ℕ :=
  t.hours * 60 + t.minutes

/-- The theorem to be proved -/
theorem new_clock_conversion (newClock : NewClockTime) (beijing : BeijingTime) :
  (newClockToMinutes ⟨5, 0⟩ = beijingToMinutes ⟨12, 0⟩) →
  (newClockToMinutes ⟨6, 75⟩ = beijingToMinutes ⟨16, 12⟩) := by
  sorry


end new_clock_conversion_l2258_225805


namespace molecular_weight_difference_l2258_225817

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

-- Define molecular weights of compounds
def molecular_weight_A : ℝ := atomic_weight_N + 4 * atomic_weight_H + atomic_weight_Br
def molecular_weight_B : ℝ := 2 * atomic_weight_O + atomic_weight_C + 3 * atomic_weight_H

-- Theorem statement
theorem molecular_weight_difference :
  molecular_weight_A - molecular_weight_B = 50.91 := by
  sorry

end molecular_weight_difference_l2258_225817


namespace age_difference_ratio_l2258_225825

/-- Represents the current ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy + 2 = 2 * (ages.julia + 2) ∧
  (ages.roy + 2) * (ages.kelly + 2) = 192

/-- The theorem to be proved -/
theorem age_difference_ratio (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy - ages.julia) / (ages.roy - ages.kelly) = 2 := by
  sorry

end age_difference_ratio_l2258_225825


namespace meaningful_expression_range_l2258_225806

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
sorry

end meaningful_expression_range_l2258_225806


namespace smallest_ellipse_area_l2258_225820

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ k : ℝ, k = 1/2 ∧ ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → π * a' * b' ≥ k * π := by
  sorry


end smallest_ellipse_area_l2258_225820


namespace blue_then_red_probability_l2258_225809

/-- The probability of drawing a blue ball first and a red ball second from a box 
    containing 15 balls (5 blue and 10 red) without replacement is 5/21. -/
theorem blue_then_red_probability (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 15 → blue = 5 → red = 10 →
  (blue : ℚ) / total * red / (total - 1) = 5 / 21 := by
  sorry

end blue_then_red_probability_l2258_225809


namespace quadratic_roots_equivalence_l2258_225832

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0 ↔
  a * c ≤ 0 → ¬(∀ x, a * x^2 - b * x + c = 0 → x > 0) :=
sorry

end quadratic_roots_equivalence_l2258_225832


namespace fish_added_l2258_225878

theorem fish_added (initial_fish final_fish : ℕ) (h1 : initial_fish = 10) (h2 : final_fish = 13) :
  final_fish - initial_fish = 3 := by
  sorry

end fish_added_l2258_225878


namespace factor_w4_minus_81_l2258_225893

theorem factor_w4_minus_81 (w : ℝ) : w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end factor_w4_minus_81_l2258_225893


namespace lexie_paintings_count_l2258_225841

/-- The number of rooms where paintings are placed -/
def num_rooms : ℕ := 4

/-- The number of paintings placed in each room -/
def paintings_per_room : ℕ := 8

/-- The total number of Lexie's watercolor paintings -/
def total_paintings : ℕ := num_rooms * paintings_per_room

theorem lexie_paintings_count : total_paintings = 32 := by
  sorry

end lexie_paintings_count_l2258_225841
