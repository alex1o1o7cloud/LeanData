import Mathlib

namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1809_180960

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1809_180960


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1809_180901

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 10 * x + 2 = 0) ↔ p = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1809_180901


namespace NUMINAMATH_CALUDE_problem_solution_l1809_180972

theorem problem_solution (a b : ℚ) 
  (h1 : a + b = 8/15) 
  (h2 : a - b = 2/15) : 
  a^2 - b^2 = 16/225 ∧ a * b = 1/25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1809_180972


namespace NUMINAMATH_CALUDE_odd_even_function_problem_l1809_180912

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_problem (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g)
  (h1 : f (-3) + g 3 = 2) (h2 : f 3 + g (-3) = 4) : 
  g 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_function_problem_l1809_180912


namespace NUMINAMATH_CALUDE_number_equation_proof_l1809_180994

theorem number_equation_proof : ∃ x : ℝ, x - (1004 / 20.08) = 4970 ∧ x = 5020 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l1809_180994


namespace NUMINAMATH_CALUDE_no_functions_satisfying_conditions_l1809_180971

theorem no_functions_satisfying_conditions :
  ¬ (∃ (f g : ℝ → ℝ), 
    (∀ (x y : ℝ), f (x^2 + g y) - f (x^2) + g y - g x ≤ 2 * y) ∧ 
    (∀ (x : ℝ), f x ≥ x^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_conditions_l1809_180971


namespace NUMINAMATH_CALUDE_square_with_triangles_removed_l1809_180982

theorem square_with_triangles_removed (s x y : ℝ) 
  (h1 : s - 2*x = 15)
  (h2 : s - 2*y = 9)
  (h3 : x > 0)
  (h4 : y > 0) :
  4 * (1/2 * x * y) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_square_with_triangles_removed_l1809_180982


namespace NUMINAMATH_CALUDE_intersection_and_union_union_equality_condition_l1809_180906

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 6}) ∧
  ((Aᶜ) ∪ B = {x | -1 < x ∧ x ≤ 6}) :=
sorry

-- Theorem for part (II)
theorem union_equality_condition (m : ℝ) :
  (B ∪ C m = B) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_union_union_equality_condition_l1809_180906


namespace NUMINAMATH_CALUDE_merry_go_round_time_l1809_180950

theorem merry_go_round_time (dave chuck erica : ℝ) : 
  chuck = 5 * dave →
  erica = chuck + 0.3 * chuck →
  erica = 65 →
  dave = 10 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_time_l1809_180950


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1809_180992

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1809_180992


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l1809_180928

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, x^(3/5) - 4 = 32 - x^(2/5) := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l1809_180928


namespace NUMINAMATH_CALUDE_smallest_shadow_area_l1809_180983

/-- The smallest area of the shadow cast by a cube onto a plane -/
theorem smallest_shadow_area (a b : ℝ) (h : b > a) (h_pos : a > 0) :
  ∃ (shadow_area : ℝ), shadow_area = (a^2 * b^2) / (b - a)^2 ∧
  ∀ (other_area : ℝ), other_area ≥ shadow_area := by
  sorry

end NUMINAMATH_CALUDE_smallest_shadow_area_l1809_180983


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_count_l1809_180930

theorem diophantine_equation_solutions_count : 
  ∃ (S : Finset ℤ), 
    (∀ p ∈ S, 1 ≤ p ∧ p ≤ 15) ∧ 
    (∀ p ∈ S, ∃ q : ℤ, p * q - 8 * p - 3 * q = 15) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_count_l1809_180930


namespace NUMINAMATH_CALUDE_product_and_square_calculation_l1809_180904

theorem product_and_square_calculation :
  (99 * 101 = 9999) ∧ (98^2 = 9604) := by
  sorry

end NUMINAMATH_CALUDE_product_and_square_calculation_l1809_180904


namespace NUMINAMATH_CALUDE_new_person_weight_l1809_180961

/-- Given a group of 8 people, where one person weighing 70 kg is replaced by a new person,
    causing the average weight to increase by 2.5 kg, 
    prove that the weight of the new person is 90 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1809_180961


namespace NUMINAMATH_CALUDE_doughnut_boxes_l1809_180932

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_boxes_l1809_180932


namespace NUMINAMATH_CALUDE_ajay_ride_distance_l1809_180924

/-- Ajay's riding speed in km/hour -/
def riding_speed : ℝ := 50

/-- Time taken for the ride in hours -/
def ride_time : ℝ := 18

/-- The distance Ajay can ride in the given time -/
def ride_distance : ℝ := riding_speed * ride_time

theorem ajay_ride_distance : ride_distance = 900 := by
  sorry

end NUMINAMATH_CALUDE_ajay_ride_distance_l1809_180924


namespace NUMINAMATH_CALUDE_pen_count_l1809_180995

/-- The number of pens in Maria's desk drawer -/
theorem pen_count (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) : 
  red = 8 →
  black = 2 * red →
  blue = black + 5 →
  green = blue / 2 →
  red + black + blue + green = 55 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l1809_180995


namespace NUMINAMATH_CALUDE_investment_growth_l1809_180991

/-- Computes the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem investment_growth : 
  let principal : ℝ := 3000
  let rate : ℝ := 0.07
  let years : ℕ := 25
  ⌊compound_interest principal rate years⌋ = 16281 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1809_180991


namespace NUMINAMATH_CALUDE_burger_distance_is_two_l1809_180910

/-- Represents the distance driven to various locations --/
structure Distances where
  school : ℕ
  softball : ℕ
  friend : ℕ
  home : ℕ

/-- Calculates the distance to the burger restaurant given the car's efficiency,
    initial gas, and distances driven to other locations --/
def distance_to_burger (efficiency : ℕ) (initial_gas : ℕ) (distances : Distances) : ℕ :=
  efficiency * initial_gas - (distances.school + distances.softball + distances.friend + distances.home)

/-- Theorem stating that the distance to the burger restaurant is 2 miles --/
theorem burger_distance_is_two :
  let efficiency := 19
  let initial_gas := 2
  let distances := Distances.mk 15 6 4 11
  distance_to_burger efficiency initial_gas distances = 2 := by
  sorry

#check burger_distance_is_two

end NUMINAMATH_CALUDE_burger_distance_is_two_l1809_180910


namespace NUMINAMATH_CALUDE_cougar_ratio_l1809_180958

theorem cougar_ratio (lions tigers total : ℕ) 
  (h1 : lions = 12)
  (h2 : tigers = 14)
  (h3 : total = 39) :
  (total - (lions + tigers)) * 2 = lions + tigers :=
by sorry

end NUMINAMATH_CALUDE_cougar_ratio_l1809_180958


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l1809_180935

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l1809_180935


namespace NUMINAMATH_CALUDE_museum_ticket_price_l1809_180900

theorem museum_ticket_price (group_size : ℕ) (total_with_tax : ℚ) (tax_rate : ℚ) :
  group_size = 25 →
  total_with_tax = 945 →
  tax_rate = 5 / 100 →
  ∃ (ticket_price : ℚ),
    ticket_price * group_size * (1 + tax_rate) = total_with_tax ∧
    ticket_price = 36 :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_price_l1809_180900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1809_180913

theorem arithmetic_sequence_solution (y : ℝ) (h : y > 0) :
  (2^2 + 5^2) / 2 = y^2 → y = Real.sqrt (29 / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1809_180913


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_1000_l1809_180915

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_plus_1000 :
  units_digit (sum_factorials 10 + 1000) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_1000_l1809_180915


namespace NUMINAMATH_CALUDE_camp_children_count_l1809_180967

/-- The initial number of children in the camp -/
def initial_children : ℕ := 50

/-- The fraction of boys in the initial group -/
def boys_fraction : ℚ := 4/5

/-- The number of boys added -/
def boys_added : ℕ := 50

/-- The fraction of girls in the final group -/
def final_girls_fraction : ℚ := 1/10

theorem camp_children_count :
  (initial_children : ℚ) * (1 - boys_fraction) = 
    final_girls_fraction * (initial_children + boys_added) := by
  sorry

end NUMINAMATH_CALUDE_camp_children_count_l1809_180967


namespace NUMINAMATH_CALUDE_jade_transactions_l1809_180969

theorem jade_transactions 
  (mabel_transactions : ℕ)
  (anthony_transactions : ℕ)
  (cal_transactions : ℕ)
  (jade_transactions : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h3 : cal_transactions = anthony_transactions * 2 / 3)
  (h4 : jade_transactions = cal_transactions + 19) :
  jade_transactions = 85 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l1809_180969


namespace NUMINAMATH_CALUDE_pants_cost_l1809_180984

theorem pants_cost (initial_money : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 109)
  (h2 : shirt_cost = 11)
  (h3 : num_shirts = 2)
  (h4 : money_left = 74) :
  initial_money - (num_shirts * shirt_cost) - money_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l1809_180984


namespace NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l1809_180942

theorem fraction_of_third_is_eighth (x : ℚ) : x * (1/3 : ℚ) = 1/8 → x = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l1809_180942


namespace NUMINAMATH_CALUDE_tuesday_poodles_l1809_180911

/-- Represents the number of hours it takes to walk a dog of a specific breed --/
def walkTime (breed : String) : ℕ :=
  match breed with
  | "Poodle" => 2
  | "Chihuahua" => 1
  | "Labrador" => 3
  | _ => 0

/-- Represents the schedule for a specific day --/
structure DaySchedule where
  poodles : ℕ
  chihuahuas : ℕ
  labradors : ℕ

def monday : DaySchedule := { poodles := 4, chihuahuas := 2, labradors := 0 }
def wednesday : DaySchedule := { poodles := 0, chihuahuas := 0, labradors := 4 }

def totalHours : ℕ := 32

theorem tuesday_poodles :
  ∃ (tuesday : DaySchedule),
    tuesday.chihuahuas = monday.chihuahuas ∧
    totalHours =
      (monday.poodles * walkTime "Poodle" +
       monday.chihuahuas * walkTime "Chihuahua" +
       wednesday.labradors * walkTime "Labrador" +
       tuesday.poodles * walkTime "Poodle" +
       tuesday.chihuahuas * walkTime "Chihuahua") ∧
    tuesday.poodles = 4 :=
  sorry

end NUMINAMATH_CALUDE_tuesday_poodles_l1809_180911


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1809_180919

theorem equation_solution_exists (a : ℝ) : 
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1809_180919


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1809_180956

theorem geometric_series_problem (q : ℝ) (b₁ : ℝ) (h₁ : |q| < 1) 
  (h₂ : b₁ / (1 - q) = 16) (h₃ : b₁^2 / (1 - q^2) = 153.6) :
  b₁ * q^3 = 3/16 ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l1809_180956


namespace NUMINAMATH_CALUDE_journey_possible_l1809_180940

/-- Represents the journey parameters and conditions -/
structure JourneyParams where
  total_distance : ℝ
  motorcycle_speed : ℝ
  baldwin_speed : ℝ
  clark_speed : ℝ
  (total_distance_positive : total_distance > 0)
  (speeds_positive : motorcycle_speed > 0 ∧ baldwin_speed > 0 ∧ clark_speed > 0)
  (motorcycle_fastest : motorcycle_speed > baldwin_speed ∧ motorcycle_speed > clark_speed)

/-- Represents a valid journey plan -/
structure JourneyPlan where
  params : JourneyParams
  baldwin_pickup : ℝ
  clark_pickup : ℝ
  (valid_pickups : 0 ≤ baldwin_pickup ∧ baldwin_pickup ≤ params.total_distance ∧
                   0 ≤ clark_pickup ∧ clark_pickup ≤ params.total_distance)

/-- Calculates the total time for a given journey plan -/
def totalTime (plan : JourneyPlan) : ℝ :=
  sorry

/-- Theorem stating that there exists a journey plan where everyone arrives in 5 hours -/
theorem journey_possible (params : JourneyParams) 
  (h1 : params.total_distance = 52)
  (h2 : params.motorcycle_speed = 20)
  (h3 : params.baldwin_speed = 5)
  (h4 : params.clark_speed = 4) :
  ∃ (plan : JourneyPlan), totalTime plan = 5 :=
sorry

end NUMINAMATH_CALUDE_journey_possible_l1809_180940


namespace NUMINAMATH_CALUDE_magnitude_of_scaled_complex_l1809_180918

theorem magnitude_of_scaled_complex (z : ℂ) :
  z = 3 - 2 * Complex.I →
  Complex.abs (-1/3 * z) = Real.sqrt 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_scaled_complex_l1809_180918


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l1809_180931

/-- A regular polygon with n sides. -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides. -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal from all diagonals in a regular polygon. -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l1809_180931


namespace NUMINAMATH_CALUDE_area_of_curve_l1809_180951

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_curve : area enclosed_region = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l1809_180951


namespace NUMINAMATH_CALUDE_harry_earnings_theorem_l1809_180944

/-- Harry's weekly dog-walking earnings -/
def harry_weekly_earnings : ℕ :=
  let monday_wednesday_friday_dogs := 7
  let tuesday_dogs := 12
  let thursday_dogs := 9
  let pay_per_dog := 5
  let days_with_7_dogs := 3
  
  (monday_wednesday_friday_dogs * days_with_7_dogs + tuesday_dogs + thursday_dogs) * pay_per_dog

/-- Theorem stating Harry's weekly earnings -/
theorem harry_earnings_theorem : harry_weekly_earnings = 210 := by
  sorry

end NUMINAMATH_CALUDE_harry_earnings_theorem_l1809_180944


namespace NUMINAMATH_CALUDE_equation_solution_l1809_180941

theorem equation_solution : ∃ x : ℚ, (1/6 : ℚ) + 2/x = 3/x + (1/15 : ℚ) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1809_180941


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1809_180923

theorem probability_three_heads_in_eight_tosses (n : Nat) (k : Nat) :
  n = 8 → k = 3 →
  (Nat.choose n k : Rat) / (2 ^ n : Rat) = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1809_180923


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1809_180920

/-- The asymptotes of the hyperbola x^2 - y^2/3 = 1 are y = ±√3 x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (k : ℝ), k^2 = 3 ∧
    (∀ x y, h x y → (y = k*x ∨ y = -k*x))
    ∧ (∀ ε > 0, ∃ δ > 0, ∀ x y, h x y → (|x| > δ → min (|y - k*x|) (|y + k*x|) < ε * |x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1809_180920


namespace NUMINAMATH_CALUDE_plot_length_is_60_l1809_180949

/-- Represents a rectangular plot with given properties --/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  cost_per_meter : ℝ
  total_cost : ℝ
  length_breadth_relation : length = breadth + 20
  perimeter_cost_relation : 2 * (length + breadth) = total_cost / cost_per_meter

/-- The length of the plot is 60 meters given the specified conditions --/
theorem plot_length_is_60 (plot : RectangularPlot) 
    (h1 : plot.cost_per_meter = 26.5)
    (h2 : plot.total_cost = 5300) : 
  plot.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_60_l1809_180949


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l1809_180986

/-- The area of the shaded region in a square with side length 20 cm and four quarter circles
    with radius 10 cm drawn at the corners is 400 - 100π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 20
  let circle_radius : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 400 - 100 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l1809_180986


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1809_180945

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalScore / b.innings

/-- The increase in average for a batsman after their last innings -/
def averageIncrease (b : Batsman) : ℚ :=
  average b - average { b with
    innings := b.innings - 1
    totalScore := b.totalScore - b.lastInningsScore
  }

theorem batsman_average_increase :
  ∀ b : Batsman,
    b.innings = 12 ∧
    b.lastInningsScore = 70 ∧
    average b = 37 →
    averageIncrease b = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1809_180945


namespace NUMINAMATH_CALUDE_max_blanks_proof_l1809_180989

/-- The width of the plywood sheet -/
def plywood_width : ℕ := 22

/-- The height of the plywood sheet -/
def plywood_height : ℕ := 15

/-- The width of the rectangular blank -/
def blank_width : ℕ := 3

/-- The height of the rectangular blank -/
def blank_height : ℕ := 5

/-- The maximum number of rectangular blanks that can be cut from the plywood sheet -/
def max_blanks : ℕ := 22

theorem max_blanks_proof :
  (plywood_width * plywood_height) ≥ (max_blanks * blank_width * blank_height) ∧
  (plywood_width * plywood_height) < ((max_blanks + 1) * blank_width * blank_height) :=
by sorry

end NUMINAMATH_CALUDE_max_blanks_proof_l1809_180989


namespace NUMINAMATH_CALUDE_sum_of_roots_l1809_180988

theorem sum_of_roots (a b : ℝ) 
  (ha : a^4 - 16*a^3 + 40*a^2 - 50*a + 25 = 0)
  (hb : b^4 - 24*b^3 + 216*b^2 - 720*b + 625 = 0) :
  a + b = 7 ∨ a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1809_180988


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1809_180959

theorem circle_radius_proof (r : ℝ) (x y : ℝ) : 
  x = π * r^2 →
  y = 2 * π * r - 6 →
  x + y = 94 * π →
  r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1809_180959


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l1809_180976

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 3 * (10 * x^2 + 10 * x + 15) - x * (10 * x - 55)
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -29/8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l1809_180976


namespace NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l1809_180978

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of cutting and rearranging a rectangle -/
structure CutAndRearrange where
  original : Rectangle
  new : Rectangle

/-- Defines the properties of a valid cut and rearrange operation -/
def isValidCutAndRearrange (cr : CutAndRearrange) : Prop :=
  cr.original.width * cr.original.height = cr.new.width * cr.new.height ∧
  cr.new.width ≠ cr.original.width ∧
  cr.new.height ≠ cr.original.height ∧
  (cr.new.width > cr.new.height → cr.new.width > cr.original.width ∧ cr.new.width > cr.original.height) ∧
  (cr.new.height > cr.new.width → cr.new.height > cr.original.width ∧ cr.new.height > cr.original.height)

/-- The main theorem to be proved -/
theorem rectangle_cut_and_rearrange :
  ∀ (cr : CutAndRearrange),
    cr.original.width = 9 ∧
    cr.original.height = 16 ∧
    isValidCutAndRearrange cr →
    max cr.new.width cr.new.height = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l1809_180978


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_l1809_180964

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_sum_equality (a b : ℝ) 
  (h1 : a > 1) (h2 : b > 1) (h3 : lg (a + b) = lg a + lg b) : 
  lg (a - 1) + lg (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_l1809_180964


namespace NUMINAMATH_CALUDE_job_filling_combinations_l1809_180934

def num_resumes : ℕ := 30
def num_unsuitable : ℕ := 20
def num_job_openings : ℕ := 5

theorem job_filling_combinations :
  (num_resumes - num_unsuitable).factorial / (num_resumes - num_unsuitable - num_job_openings).factorial = 30240 :=
by sorry

end NUMINAMATH_CALUDE_job_filling_combinations_l1809_180934


namespace NUMINAMATH_CALUDE_existence_of_positive_rationals_l1809_180974

theorem existence_of_positive_rationals (n : ℕ) (h : n ≥ 4) :
  ∃ (k : ℕ) (a : ℕ → ℚ),
    k ≥ 2 ∧
    (∀ i, i ∈ Finset.range k → a i > 0) ∧
    (Finset.sum (Finset.range k) a = n) ∧
    (Finset.prod (Finset.range k) a = n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_positive_rationals_l1809_180974


namespace NUMINAMATH_CALUDE_diamond_720_1001_cubed_l1809_180908

/-- The diamond operation on positive integers -/
def diamond (a b : ℕ+) : ℕ := sorry

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ := sorry

theorem diamond_720_1001_cubed : 
  (diamond 720 1001)^3 = 216 := by sorry

end NUMINAMATH_CALUDE_diamond_720_1001_cubed_l1809_180908


namespace NUMINAMATH_CALUDE_S_divisible_by_4003_l1809_180975

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := factorial 2001 + (List.range 2001).foldl (λ acc i => acc * (2002 + i)) 1

theorem S_divisible_by_4003 : S % 4003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_S_divisible_by_4003_l1809_180975


namespace NUMINAMATH_CALUDE_greatest_divisor_XYXY_l1809_180997

/-- A four-digit palindrome of the pattern XYXY -/
def XYXY (X Y : Nat) : Nat := 1000 * X + 100 * Y + 10 * X + Y

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The theorem stating that 11 is the greatest divisor of all XYXY palindromes -/
theorem greatest_divisor_XYXY :
  ∀ X Y : Nat, is_single_digit X → is_single_digit Y →
  (∀ d : Nat, d > 11 → ¬(d ∣ XYXY X Y)) ∧
  (11 ∣ XYXY X Y) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_XYXY_l1809_180997


namespace NUMINAMATH_CALUDE_log_inequality_l1809_180921

/-- Given 0 < a < b < 1 < c, prove that log_b(c) < log_a(c) < a^c -/
theorem log_inequality (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb1 : b < 1) (hc : 1 < c) :
  Real.log c / Real.log b < Real.log c / Real.log a ∧ Real.log c / Real.log a < a^c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1809_180921


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_7999999999_l1809_180981

theorem largest_prime_factor_of_7999999999 :
  let n : ℕ := 7999999999
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q) →
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q → q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ p = 4002001) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_7999999999_l1809_180981


namespace NUMINAMATH_CALUDE_total_boys_in_assembly_l1809_180939

/-- Represents the assembly of boys in two rows --/
structure Assembly where
  first_row : ℕ
  second_row : ℕ

/-- The position of a boy in a row --/
structure Position where
  from_left : ℕ
  from_right : ℕ

/-- Represents the assembly with given conditions --/
def school_assembly : Assembly where
  first_row := 24
  second_row := 24

/-- Rajan's position in the first row --/
def rajan_position : Position where
  from_left := 6
  from_right := school_assembly.first_row - 5

/-- Vinay's position in the first row --/
def vinay_position : Position where
  from_left := school_assembly.first_row - 9
  from_right := 10

/-- Number of boys between Rajan and Vinay --/
def boys_between : ℕ := 8

/-- Suresh's position in the second row --/
def suresh_position : Position where
  from_left := 5
  from_right := school_assembly.second_row - 4

theorem total_boys_in_assembly :
  school_assembly.first_row + school_assembly.second_row = 48 ∧
  school_assembly.first_row = school_assembly.second_row ∧
  rajan_position.from_left = 6 ∧
  vinay_position.from_right = 10 ∧
  vinay_position.from_left - rajan_position.from_left - 1 = boys_between ∧
  suresh_position.from_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_boys_in_assembly_l1809_180939


namespace NUMINAMATH_CALUDE_largest_number_of_piles_l1809_180903

theorem largest_number_of_piles (apples : Nat) (apricots : Nat) (cherries : Nat)
  (h1 : apples = 42)
  (h2 : apricots = 60)
  (h3 : cherries = 90) :
  Nat.gcd apples (Nat.gcd apricots cherries) = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_piles_l1809_180903


namespace NUMINAMATH_CALUDE_minimum_value_of_reciprocal_sum_l1809_180933

theorem minimum_value_of_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1 - n)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y ≥ 1/m + 1/n) →
  1/m + 1/n = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_reciprocal_sum_l1809_180933


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1809_180962

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  (p.A.x + p.C.x = p.B.x + p.D.x) ∧ (p.A.y + p.C.y = p.B.y + p.D.y)

theorem parallelogram_fourth_vertex :
  ∀ (p : Parallelogram),
  p.A = Point.mk (-2) 1 →
  p.B = Point.mk (-1) 3 →
  p.C = Point.mk 3 4 →
  is_parallelogram p →
  (p.D = Point.mk 2 2 ∨ p.D = Point.mk (-6) 0 ∨ p.D = Point.mk 4 6) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1809_180962


namespace NUMINAMATH_CALUDE_class_size_roses_class_size_l1809_180905

theorem class_size (girls_present : ℕ) (boys_absent : ℕ) : ℕ :=
  let boys_present := girls_present / 2
  let total_boys := boys_present + boys_absent
  let total_students := girls_present + total_boys
  total_students

theorem roses_class_size : class_size 140 40 = 250 := by
  sorry

end NUMINAMATH_CALUDE_class_size_roses_class_size_l1809_180905


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1809_180922

theorem root_exists_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-2) (-1) ∧ 2^x - x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1809_180922


namespace NUMINAMATH_CALUDE_max_area_PQR_max_area_incenters_l1809_180948

-- Define the equilateral triangle ABC with unit area
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the external equilateral triangles
def triangle_APB : Set (ℝ × ℝ) := sorry
def triangle_BQC : Set (ℝ × ℝ) := sorry
def triangle_CRA : Set (ℝ × ℝ) := sorry

-- Define the angles
def angle_APB : ℝ := 60
def angle_BQC : ℝ := 60
def angle_CRA : ℝ := 60

-- Define the points P, Q, R
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the incenters
def incenter_APB : ℝ × ℝ := sorry
def incenter_BQC : ℝ × ℝ := sorry
def incenter_CRA : ℝ × ℝ := sorry

-- Define the area function
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem for the maximum area of triangle PQR
theorem max_area_PQR :
  ∀ P Q R,
    P ∈ triangle_APB ∧ Q ∈ triangle_BQC ∧ R ∈ triangle_CRA →
    area {P, Q, R} ≤ 4 * Real.sqrt 3 :=
sorry

-- Theorem for the maximum area of triangle formed by incenters
theorem max_area_incenters :
  area {incenter_APB, incenter_BQC, incenter_CRA} ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_area_PQR_max_area_incenters_l1809_180948


namespace NUMINAMATH_CALUDE_train_length_l1809_180993

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 50 → time_s = 9 → 
  (speed_kmh * 1000 / 3600) * time_s = 125 := by sorry

end NUMINAMATH_CALUDE_train_length_l1809_180993


namespace NUMINAMATH_CALUDE_parallelogram_zk_product_l1809_180965

structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ

def is_valid_parallelogram (p : Parallelogram) (z k : ℝ) : Prop :=
  p.EF = 5 * z + 5 ∧
  p.FG = 4 * k^2 ∧
  p.GH = 40 ∧
  p.HE = k + 20

theorem parallelogram_zk_product (p : Parallelogram) (z k : ℝ) :
  is_valid_parallelogram p z k → z * k = (7 + 7 * Real.sqrt 321) / 8 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_zk_product_l1809_180965


namespace NUMINAMATH_CALUDE_scenario_equivalence_l1809_180943

/-- Represents the cost of trees in yuan -/
structure TreeCost where
  pine : ℝ
  cypress : ℝ

/-- Represents the given scenario for tree costs -/
def scenario (cost : TreeCost) : Prop :=
  2 * cost.pine + 3 * cost.cypress = 120 ∧
  2 * cost.pine - cost.cypress = 20

/-- The correct system of equations for the scenario -/
def correct_system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 120 ∧
  2 * x - y = 20

/-- Theorem stating that the correct system accurately represents the scenario -/
theorem scenario_equivalence :
  ∀ (cost : TreeCost), scenario cost ↔ correct_system cost.pine cost.cypress :=
by sorry

end NUMINAMATH_CALUDE_scenario_equivalence_l1809_180943


namespace NUMINAMATH_CALUDE_cookie_radius_l1809_180907

/-- Given a circle described by the equation x^2 + y^2 + 2x - 4y - 7 = 0, its radius is 2√3 -/
theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 7 = 0) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cookie_radius_l1809_180907


namespace NUMINAMATH_CALUDE_remaining_candy_l1809_180953

/-- Given a group of people who collected candy and ate some, calculate the remaining candy. -/
theorem remaining_candy (total_candy : ℕ) (num_people : ℕ) (candy_eaten_per_person : ℕ) :
  total_candy = 120 →
  num_people = 3 →
  candy_eaten_per_person = 6 →
  total_candy - (num_people * candy_eaten_per_person) = 102 := by
  sorry

end NUMINAMATH_CALUDE_remaining_candy_l1809_180953


namespace NUMINAMATH_CALUDE_wednesdays_temperature_l1809_180955

theorem wednesdays_temperature (monday tuesday wednesday : ℤ) : 
  tuesday = monday + 4 →
  wednesday = monday - 6 →
  tuesday = 22 →
  wednesday = 12 := by
sorry

end NUMINAMATH_CALUDE_wednesdays_temperature_l1809_180955


namespace NUMINAMATH_CALUDE_sales_solution_l1809_180917

def sales_problem (month1 month3 month4 month5 month6 average_sale : ℕ) : Prop :=
  ∃ (month2 : ℕ),
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average_sale ∧
    month2 = 6927

theorem sales_solution :
  sales_problem 6435 6855 7230 6562 7991 7000 :=
sorry

end NUMINAMATH_CALUDE_sales_solution_l1809_180917


namespace NUMINAMATH_CALUDE_expression_simplification_l1809_180970

theorem expression_simplification (a b c x y z : ℝ) :
  (c * x * (b * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + b * z * (b * x^2 + 3 * c^2 * x^2 + a^2 * y^2)) / (c * x + b * z) = 
  b * x^2 + a^2 * y^2 + c^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1809_180970


namespace NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l1809_180985

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by
  sorry

theorem problem_solution :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l1809_180985


namespace NUMINAMATH_CALUDE_total_sugar_needed_l1809_180963

def sugar_for_frosting : ℝ := 0.6
def sugar_for_cake : ℝ := 0.2

theorem total_sugar_needed : sugar_for_frosting + sugar_for_cake = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_needed_l1809_180963


namespace NUMINAMATH_CALUDE_basketball_team_starters_count_l1809_180946

theorem basketball_team_starters_count :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 2
  let remaining_players : ℕ := total_players - quadruplets
  let remaining_starters : ℕ := starters - quadruplets_in_lineup

  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose remaining_players remaining_starters) = 4752 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_count_l1809_180946


namespace NUMINAMATH_CALUDE_wage_problem_l1809_180968

/-- Given a sum of money S that can pay q's wages for 40 days and both p and q's wages for 15 days,
    prove that S can pay p's wages for 24 days. -/
theorem wage_problem (S P Q : ℝ) (hS_positive : S > 0) (hP_positive : P > 0) (hQ_positive : Q > 0)
  (hS_q : S = 40 * Q) (hS_pq : S = 15 * (P + Q)) :
  S = 24 * P := by
  sorry

end NUMINAMATH_CALUDE_wage_problem_l1809_180968


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1809_180998

def fill_time (rate_a rate_b rate_c : ℚ) : ℚ :=
  1 / (rate_a + rate_b + rate_c)

theorem cistern_fill_time :
  let rate_a : ℚ := 1 / 10
  let rate_b : ℚ := 1 / 12
  let rate_c : ℚ := -1 / 15
  fill_time rate_a rate_b rate_c = 60 / 7 :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1809_180998


namespace NUMINAMATH_CALUDE_boris_early_theorem_l1809_180926

/-- Represents the distance between two points -/
structure Distance where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents a speed (distance per unit time) -/
structure Speed where
  value : ℝ
  pos : 0 < value

/-- Represents a time duration -/
structure Time where
  value : ℝ
  nonneg : 0 ≤ value

/-- The scenario of Anna and Boris walking towards each other -/
structure WalkingScenario where
  d : Distance  -- distance between villages A and B
  v_A : Speed   -- Anna's speed
  v_B : Speed   -- Boris's speed
  t : Time      -- time they meet when starting simultaneously

variable (scenario : WalkingScenario)

/-- The distance Anna walks in the original scenario -/
def anna_distance : ℝ := scenario.v_A.value * scenario.t.value

/-- The distance Boris walks in the original scenario -/
def boris_distance : ℝ := scenario.v_B.value * scenario.t.value

/-- Condition: Anna and Boris meet when they start simultaneously -/
axiom meet_condition : anna_distance scenario + boris_distance scenario = scenario.d.value

/-- Condition: If Anna starts 30 minutes earlier, they meet 2 km closer to village B -/
axiom anna_early_condition : 
  scenario.v_A.value * (scenario.t.value + 0.5) + scenario.v_B.value * scenario.t.value 
  = scenario.d.value - 2

/-- Theorem: If Boris starts 30 minutes earlier, they meet 2 km closer to village A -/
theorem boris_early_theorem : 
  scenario.v_A.value * scenario.t.value + scenario.v_B.value * (scenario.t.value + 0.5) 
  = scenario.d.value + 2 := by
  sorry

end NUMINAMATH_CALUDE_boris_early_theorem_l1809_180926


namespace NUMINAMATH_CALUDE_athlete_calorie_burn_l1809_180977

/-- Calculates the total calories burned by an athlete during exercise -/
theorem athlete_calorie_burn 
  (running_rate : ℕ) 
  (walking_rate : ℕ) 
  (total_time : ℕ) 
  (running_time : ℕ) 
  (h1 : running_rate = 10)
  (h2 : walking_rate = 4)
  (h3 : total_time = 60)
  (h4 : running_time = 35)
  (h5 : running_time ≤ total_time) :
  running_rate * running_time + walking_rate * (total_time - running_time) = 450 := by
  sorry

#check athlete_calorie_burn

end NUMINAMATH_CALUDE_athlete_calorie_burn_l1809_180977


namespace NUMINAMATH_CALUDE_intersection_implies_y_zero_l1809_180929

theorem intersection_implies_y_zero (x y : ℝ) : 
  let A : Set ℝ := {2, Real.log x}
  let B : Set ℝ := {x, y}
  A ∩ B = {0} → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_y_zero_l1809_180929


namespace NUMINAMATH_CALUDE_crew_size_proof_l1809_180947

/-- The number of laborers present on a certain day -/
def present_laborers : ℕ := 10

/-- The percentage of laborers that showed up for work (as a rational number) -/
def attendance_percentage : ℚ := 385 / 1000

/-- The total number of laborers in the crew -/
def total_laborers : ℕ := 26

theorem crew_size_proof :
  (present_laborers : ℚ) / attendance_percentage = total_laborers := by
  sorry

end NUMINAMATH_CALUDE_crew_size_proof_l1809_180947


namespace NUMINAMATH_CALUDE_total_production_8_minutes_l1809_180952

/-- Represents a machine type in the factory -/
inductive MachineType
| A
| B
| C

/-- Represents the state of the factory at a given time -/
structure FactoryState where
  machineCount : MachineType → ℕ
  productionRate : MachineType → ℕ

/-- Calculates the total production for a given time interval -/
def totalProduction (state : FactoryState) (minutes : ℕ) : ℕ :=
  (state.machineCount MachineType.A * state.productionRate MachineType.A +
   state.machineCount MachineType.B * state.productionRate MachineType.B +
   state.machineCount MachineType.C * state.productionRate MachineType.C) * minutes

/-- The initial state of the factory -/
def initialState : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 4
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 2 minutes -/
def stateAfter2Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 4
    | MachineType.B => 7
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 270
    | MachineType.B => 200
    | MachineType.C => 150
}

/-- The state of the factory after 4 minutes -/
def stateAfter4Min : FactoryState := {
  machineCount := λ t => match t with
    | MachineType.A => 6
    | MachineType.B => 9
    | MachineType.C => 3
  productionRate := λ t => match t with
    | MachineType.A => 300
    | MachineType.B => 180
    | MachineType.C => 170
}

/-- Theorem stating the total production over 8 minutes -/
theorem total_production_8_minutes :
  totalProduction initialState 2 +
  totalProduction stateAfter2Min 2 +
  totalProduction stateAfter4Min 4 = 27080 := by
  sorry


end NUMINAMATH_CALUDE_total_production_8_minutes_l1809_180952


namespace NUMINAMATH_CALUDE_correct_calculation_l1809_180990

theorem correct_calculation (n m : ℝ) : n * m^2 - 2 * m^2 * n = -m^2 * n := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1809_180990


namespace NUMINAMATH_CALUDE_angle4_measure_l1809_180927

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- State the theorem
theorem angle4_measure
  (h1 : angle1 = 82)
  (h2 : angle2 = 34)
  (h3 : angle3 = 19)
  (h4 : angle5 = angle6 + 10)
  (h5 : angle1 + angle2 + angle3 + angle5 + angle6 = 180)
  (h6 : angle4 + angle5 + angle6 = 180) :
  angle4 = 135 := by
sorry

end NUMINAMATH_CALUDE_angle4_measure_l1809_180927


namespace NUMINAMATH_CALUDE_sigma_phi_inequality_l1809_180937

open Nat

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A natural number is prime if it has exactly two divisors -/
def isPrime (n : ℕ) : Prop := sorry

theorem sigma_phi_inequality (n : ℕ) (h : n > 1) :
  sigma n * phi n ≤ n^2 - 1 ∧ (sigma n * phi n = n^2 - 1 ↔ isPrime n) := by sorry

end NUMINAMATH_CALUDE_sigma_phi_inequality_l1809_180937


namespace NUMINAMATH_CALUDE_total_tax_percentage_l1809_180966

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.05 := by
sorry


end NUMINAMATH_CALUDE_total_tax_percentage_l1809_180966


namespace NUMINAMATH_CALUDE_exponent_division_l1809_180916

theorem exponent_division (a : ℝ) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1809_180916


namespace NUMINAMATH_CALUDE_rectangle_area_l1809_180980

/-- Given a rectangle with length three times its width and diagonal y, prove its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (y_pos : y > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
  w^2 + (3*w)^2 = y^2 ∧ 
  3 * w^2 = (3 * y^2) / 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1809_180980


namespace NUMINAMATH_CALUDE_no_subdivision_for_1986_plots_l1809_180914

theorem no_subdivision_for_1986_plots : ¬ ∃ (n : ℕ), 8 * n + 9 = 1986 := by
  sorry

end NUMINAMATH_CALUDE_no_subdivision_for_1986_plots_l1809_180914


namespace NUMINAMATH_CALUDE_balanced_numbers_count_l1809_180954

/-- A four-digit number abcd is balanced if a + b = c + d -/
def is_balanced (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + b = c + d

/-- Count of balanced four-digit numbers with sum 8 -/
def balanced_sum_8_count : ℕ := 72

/-- Count of balanced four-digit numbers with sum 16 -/
def balanced_sum_16_count : ℕ := 9

/-- Total count of balanced four-digit numbers -/
def total_balanced_count : ℕ := 615

/-- Theorem stating the counts of balanced numbers -/
theorem balanced_numbers_count :
  (balanced_sum_8_count = 72) ∧
  (balanced_sum_16_count = 9) ∧
  (total_balanced_count = 615) :=
sorry

end NUMINAMATH_CALUDE_balanced_numbers_count_l1809_180954


namespace NUMINAMATH_CALUDE_crayons_difference_l1809_180996

def birthday_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l1809_180996


namespace NUMINAMATH_CALUDE_valid_pairs_l1809_180999

def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / Nat.lcm x y + (1 : ℚ) / Nat.gcd x y = (1 : ℚ) / 2

theorem valid_pairs : 
  ∀ x y : ℕ+, is_valid_pair x y ↔ 
    ((x = 5 ∧ y = 20) ∨ 
     (x = 6 ∧ y = 12) ∨ 
     (x = 8 ∧ y = 8) ∨ 
     (x = 8 ∧ y = 12) ∨ 
     (x = 9 ∧ y = 24) ∨ 
     (x = 12 ∧ y = 15) ∨
     (y = 5 ∧ x = 20) ∨ 
     (y = 6 ∧ x = 12) ∨ 
     (y = 8 ∧ x = 12) ∨ 
     (y = 9 ∧ x = 24) ∨ 
     (y = 12 ∧ x = 15)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l1809_180999


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1809_180936

/-- The number of combinations of k items chosen from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 10 available toppings -/
def num_toppings : ℕ := 10

/-- We want to choose 3 toppings -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  combination num_toppings toppings_to_choose = 120 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1809_180936


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l1809_180987

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the circle in rectangular coordinates
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  circle_equation x y →
  1 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l1809_180987


namespace NUMINAMATH_CALUDE_all_terms_perfect_squares_l1809_180938

/-- A sequence of integers satisfying specific conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n ≥ 2, a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧
  (∀ m : ℕ, ∃ k : ℕ, ∀ i < m, ∃ j : ℤ, a (k + i) = j ^ 2)

/-- All terms in the special sequence are perfect squares -/
theorem all_terms_perfect_squares (a : ℕ → ℤ) (h : SpecialSequence a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_all_terms_perfect_squares_l1809_180938


namespace NUMINAMATH_CALUDE_faye_age_l1809_180957

/-- Given the ages of Chad, Diana, Eduardo, and Faye, prove that Faye is 18 years old. -/
theorem faye_age (C D E F : ℕ) 
  (h1 : D = E - 2)
  (h2 : E = C + 3)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 18 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l1809_180957


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l1809_180925

/-- Calculates the average hourly parking cost for a given duration -/
def averageHourlyCost (baseCost : ℚ) (baseHours : ℚ) (additionalHourlyRate : ℚ) (totalHours : ℚ) : ℚ :=
  let totalCost := baseCost + (totalHours - baseHours) * additionalHourlyRate
  totalCost / totalHours

/-- Proves that the average hourly cost for 9 hours of parking is $3.03 -/
theorem parking_cost_theorem :
  let baseCost : ℚ := 15
  let baseHours : ℚ := 2
  let additionalHourlyRate : ℚ := 1.75
  let totalHours : ℚ := 9
  averageHourlyCost baseCost baseHours additionalHourlyRate totalHours = 3.03 := by
  sorry

#eval averageHourlyCost 15 2 1.75 9

end NUMINAMATH_CALUDE_parking_cost_theorem_l1809_180925


namespace NUMINAMATH_CALUDE_sum_of_constants_l1809_180902

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 64 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 1)
  (h2 : a < b) : 
  a + 2*b + 3*c = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l1809_180902


namespace NUMINAMATH_CALUDE_count_numbers_eq_243_l1809_180909

/-- The count of three-digit numbers less than 500 that do not contain the digit 1 -/
def count_numbers : Nat :=
  let hundreds := {2, 3, 4}
  let other_digits := {0, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card hundreds) * (Finset.card other_digits) * (Finset.card other_digits)

/-- Theorem stating that the count of three-digit numbers less than 500 
    that do not contain the digit 1 is equal to 243 -/
theorem count_numbers_eq_243 : count_numbers = 243 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_eq_243_l1809_180909


namespace NUMINAMATH_CALUDE_fraction_comparison_l1809_180979

theorem fraction_comparison 
  (a b c d : ℤ) 
  (hc : c ≠ 0) 
  (hd : d ≠ 0) : 
  (c = d ∧ a > b → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a = b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a > b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1809_180979


namespace NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l1809_180973

theorem rational_power_difference_integer_implies_integer 
  (a b : ℚ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_distinct : a ≠ b) 
  (h_inf_int : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ∃ (k : ℤ), k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), (m : ℚ) = a ∧ (n : ℚ) = b :=
sorry

end NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l1809_180973
