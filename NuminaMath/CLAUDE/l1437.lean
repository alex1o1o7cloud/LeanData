import Mathlib

namespace polygon_sides_and_diagonals_l1437_143712

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (n > 2) →
  (180 * (n - 2) = 3 * 360 - 180) →
  (n = 7 ∧ (n * (n - 3)) / 2 = 14) :=
by
  sorry

end polygon_sides_and_diagonals_l1437_143712


namespace analysis_time_proof_l1437_143776

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time (in hours) required to analyze one bone -/
def time_per_bone : ℕ := 1

/-- The total time required to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end analysis_time_proof_l1437_143776


namespace log_calculation_l1437_143756

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_calculation : log10 5 * log10 20 + (log10 2)^2 = 1 := by
  sorry

end log_calculation_l1437_143756


namespace angle_positions_l1437_143760

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define the position of an angle
inductive AnglePosition
  | InQuadrant (q : Quadrant)
  | OnPositiveYAxis

-- Function to determine the position of 2θ
def doubleThetaPosition (θ : Real) : AnglePosition := sorry

-- Function to determine the position of θ/2
def halfThetaPosition (θ : Real) : Quadrant := sorry

-- Theorem statement
theorem angle_positions (θ : Real) 
  (h : ∃ (k : ℤ), 180 + k * 360 < θ ∧ θ < 270 + k * 360) : 
  (doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.First ∨
   doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.Second ∨
   doubleThetaPosition θ = AnglePosition.OnPositiveYAxis) ∧
  (halfThetaPosition θ = Quadrant.Second ∨
   halfThetaPosition θ = Quadrant.Fourth) := by
  sorry

end angle_positions_l1437_143760


namespace trigonometric_equation_solution_l1437_143791

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sqrt (3 + 4 * Real.sqrt 6 - (16 * Real.sqrt 3 - 8 * Real.sqrt 2) * Real.sin x) = 4 * Real.sin x - Real.sqrt 3) ↔ 
  ∃ k : ℤ, x = (-1)^k * (π / 4) + 2 * k * π :=
by sorry

end trigonometric_equation_solution_l1437_143791


namespace seven_row_triangle_pieces_l1437_143729

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of the triangle -/
structure TriangleStructure where
  rows : ℕ
  rodIncrease : ℕ
  extraConnectorRow : ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : TriangleStructure) : ℕ :=
  let totalRods := arithmeticSum 3 t.rodIncrease t.rows
  let totalConnectors := triangularNumber (t.rows + t.extraConnectorRow)
  totalRods + totalConnectors

/-- The main theorem to prove -/
theorem seven_row_triangle_pieces :
  let t : TriangleStructure := {
    rows := 7,
    rodIncrease := 3,
    extraConnectorRow := 1
  }
  totalPieces t = 120 := by sorry

end seven_row_triangle_pieces_l1437_143729


namespace sum_of_baby_ages_theorem_l1437_143725

/- Define the ages of the animals -/
def lioness_age : ℕ := 12
def hyena_age : ℕ := lioness_age / 2
def leopard_age : ℕ := 3 * hyena_age

/- Define the ages of the babies -/
def lioness_baby_age : ℕ := lioness_age / 2
def hyena_baby_age : ℕ := hyena_age / 2
def leopard_baby_age : ℕ := leopard_age / 2

/- Define the sum of the babies' ages after 5 years -/
def sum_of_baby_ages_after_5_years : ℕ := 
  (lioness_baby_age + 5) + (hyena_baby_age + 5) + (leopard_baby_age + 5)

theorem sum_of_baby_ages_theorem : sum_of_baby_ages_after_5_years = 33 := by
  sorry

end sum_of_baby_ages_theorem_l1437_143725


namespace theater_seating_l1437_143710

/-- Represents the number of seats in a given row of the theater. -/
def seats (n : ℕ) : ℕ := 3 * n + 57

theorem theater_seating :
  (seats 6 = 75) ∧
  (seats 8 = 81) ∧
  (∀ n : ℕ, seats n = 3 * n + 57) ∧
  (seats 21 = 120) := by
  sorry

#check theater_seating

end theater_seating_l1437_143710


namespace mistaken_multiplication_l1437_143723

theorem mistaken_multiplication (x : ℚ) : 
  6 * x = 12 → 7 * x = 14 := by
sorry

end mistaken_multiplication_l1437_143723


namespace triangle_properties_l1437_143732

theorem triangle_properties (A B C : ℝ) (a b c R : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  R > 0 →
  a = Real.sqrt 3 →
  A = π/3 →
  2 * R = a / Real.sin A →
  2 * R = b / Real.sin B →
  2 * R = c / Real.sin C →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  R = 1 ∧ ∀ (b' c' : ℝ), b' * c' ≤ 3 := by sorry

end triangle_properties_l1437_143732


namespace make_up_average_is_95_percent_l1437_143775

/-- Represents the average score of students who took the exam on the make-up date -/
def make_up_average (total_students : ℕ) (assigned_day_percent : ℚ) (assigned_day_average : ℚ) (overall_average : ℚ) : ℚ :=
  (overall_average * total_students - assigned_day_average * (assigned_day_percent * total_students)) / ((1 - assigned_day_percent) * total_students)

/-- Theorem stating the average score of students who took the exam on the make-up date -/
theorem make_up_average_is_95_percent :
  make_up_average 100 (70/100) (65/100) (74/100) = 95/100 := by
  sorry

end make_up_average_is_95_percent_l1437_143775


namespace smallest_x_absolute_value_l1437_143740

theorem smallest_x_absolute_value : ∃ x : ℝ, 
  (∀ y : ℝ, |5*y - 3| = 15 → x ≤ y) ∧ |5*x - 3| = 15 := by
  sorry

end smallest_x_absolute_value_l1437_143740


namespace sock_order_ratio_l1437_143793

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.black * (3 * price.blue) + order.blue * price.blue

theorem sock_order_ratio : ∀ (original : SockOrder) (price : SockPrice),
  original.black = 6 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 3 / 8 := by
  sorry

end sock_order_ratio_l1437_143793


namespace problem_statements_l1437_143719

theorem problem_statements :
  (∀ x : ℝ, (Real.sqrt (x + 1) * (2 * x - 1) ≥ 0) ↔ (x ≥ 1/2)) ∧
  (∀ x y : ℝ, (x > 1 ∧ y > 2) → (x + y > 3)) ∧
  (∃ x y : ℝ, (x + y > 3) ∧ ¬(x > 1 ∧ y > 2)) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) ∧
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end problem_statements_l1437_143719


namespace calculate_expression_l1437_143766

theorem calculate_expression : 15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 := by
  sorry

end calculate_expression_l1437_143766


namespace mrs_hilt_pies_l1437_143701

/-- The total number of pies Mrs. Hilt needs to bake for the bigger event -/
def total_pies (pecan_initial : Float) (apple_initial : Float) (cherry_initial : Float)
                (pecan_multiplier : Float) (apple_multiplier : Float) (cherry_multiplier : Float) : Float :=
  pecan_initial * pecan_multiplier + apple_initial * apple_multiplier + cherry_initial * cherry_multiplier

/-- Theorem stating that Mrs. Hilt needs to bake 193.5 pies for the bigger event -/
theorem mrs_hilt_pies : 
  total_pies 16.5 14.25 12.75 4.3 3.5 5.7 = 193.5 := by
  sorry

end mrs_hilt_pies_l1437_143701


namespace german_enrollment_l1437_143782

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_subjects = 12)
  (h3 : only_english = 18)
  (h4 : ∃ (german : ℕ), german > 0)
  (h5 : total_students = both_subjects + only_english + (total_students - both_subjects - only_english)) :
  total_students - only_english = 22 := by
  sorry

end german_enrollment_l1437_143782


namespace franks_initial_money_l1437_143788

/-- Frank's initial amount of money -/
def initial_money : ℕ := sorry

/-- The amount Frank spent on toys -/
def money_spent : ℕ := 8

/-- The amount Frank had left after spending -/
def money_left : ℕ := 8

/-- Theorem stating that Frank's initial money was $16 -/
theorem franks_initial_money : initial_money = 16 := by sorry

end franks_initial_money_l1437_143788


namespace rectangle_area_l1437_143754

/-- Given a rectangle with length 16 and diagonal 20, prove its area is 192. -/
theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 → 
  diagonal = 20 → 
  length^2 + width^2 = diagonal^2 → 
  length * width = 192 := by
sorry

end rectangle_area_l1437_143754


namespace football_game_cost_l1437_143742

/-- The cost of a football game, given the total spent and the costs of two other games. -/
theorem football_game_cost (total_spent strategy_cost batman_cost : ℚ) :
  total_spent = 35.52 ∧ strategy_cost = 9.46 ∧ batman_cost = 12.04 →
  total_spent - (strategy_cost + batman_cost) = 14.02 := by
  sorry

end football_game_cost_l1437_143742


namespace event_A_subset_event_B_l1437_143745

-- Define the sample space for tossing two coins
inductive CoinOutcome
  | HH -- Both heads
  | HT -- First head, second tail
  | TH -- First tail, second head
  | TT -- Both tails

-- Define the probability space
def coin_toss_space : Type := CoinOutcome

-- Define the events A and B
def event_A : Set coin_toss_space := {CoinOutcome.HH}
def event_B : Set coin_toss_space := {CoinOutcome.HH, CoinOutcome.TT}

-- State the theorem
theorem event_A_subset_event_B : event_A ⊆ event_B := by sorry

end event_A_subset_event_B_l1437_143745


namespace distance_walked_calculation_l1437_143795

/-- Calculates the distance walked given the walking time and speed. -/
def distance_walked (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: The distance walked is 499.98 meters given the specified conditions. -/
theorem distance_walked_calculation :
  let time : ℝ := 6
  let speed : ℝ := 83.33
  distance_walked time speed = 499.98 := by sorry

end distance_walked_calculation_l1437_143795


namespace exact_three_blue_marbles_probability_l1437_143748

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_trials : ℕ := 6
def num_blue_selections : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_three_blue_marbles_probability :
  Nat.choose num_trials num_blue_selections *
  (prob_blue ^ num_blue_selections) *
  (prob_red ^ (num_trials - num_blue_selections)) =
  3512320 / 11390625 := by
  sorry

end exact_three_blue_marbles_probability_l1437_143748


namespace average_weight_increase_l1437_143764

/-- Proves that replacing a person weighing 76 kg with a person weighing 119.4 kg
    in a group of 7 people increases the average weight by 6.2 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 7 * initial_average
  let new_total := initial_total - 76 + 119.4
  let new_average := new_total / 7
  new_average - initial_average = 6.2 := by sorry

end average_weight_increase_l1437_143764


namespace apple_street_length_in_km_l1437_143781

/-- The length of Apple Street in meters -/
def apple_street_length : ℝ := 3200

/-- The distance between intersections in meters -/
def intersection_distance : ℝ := 200

/-- The number of numbered intersections -/
def numbered_intersections : ℕ := 15

/-- The total number of intersections -/
def total_intersections : ℕ := numbered_intersections + 1

theorem apple_street_length_in_km :
  apple_street_length / 1000 = 3.2 := by sorry

end apple_street_length_in_km_l1437_143781


namespace employee_b_pay_is_220_l1437_143728

/-- Given two employees A and B with a total weekly pay and A's pay as a percentage of B's, 
    calculate B's weekly pay. -/
def calculate_employee_b_pay (total_pay : ℚ) (a_percentage : ℚ) : ℚ :=
  total_pay / (1 + a_percentage)

/-- Theorem stating that given the problem conditions, employee B's pay is 220. -/
theorem employee_b_pay_is_220 :
  calculate_employee_b_pay 550 (3/2) = 220 := by sorry

end employee_b_pay_is_220_l1437_143728


namespace investment_split_l1437_143730

/-- Proves the amount invested at 6% given total investment, interest rates, and total interest earned --/
theorem investment_split (total_investment : ℝ) (rate1 rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : total_interest = 1023)
  (h5 : ∃ (x y : ℝ), x + y = total_investment ∧ 
                     rate1 * x + rate2 * y = total_interest) :
  ∃ (x : ℝ), x = 6800 ∧ 
              ∃ (y : ℝ), y = total_investment - x ∧
                          rate1 * x + rate2 * y = total_interest :=
sorry

end investment_split_l1437_143730


namespace peaches_per_box_is_15_l1437_143717

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten by farmers, and the number of smaller boxes,
    calculate the number of peaches in each smaller box. -/
def peaches_per_box (initial_peaches_per_basket : ℕ) (num_baskets : ℕ) 
                    (peaches_eaten : ℕ) (num_smaller_boxes : ℕ) : ℕ :=
  ((initial_peaches_per_basket * num_baskets) - peaches_eaten) / num_smaller_boxes

/-- Theorem stating that given the specific conditions in the problem,
    the number of peaches in each smaller box is 15. -/
theorem peaches_per_box_is_15 :
  peaches_per_box 25 5 5 8 = 15 := by
  sorry

end peaches_per_box_is_15_l1437_143717


namespace max_angle_between_tangents_l1437_143784

/-- The parabola C₁ defined by y² = 4x -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The circle C₂ defined by (x-3)² + y² = 2 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

/-- The angle between two tangents drawn from a point to a circle -/
def angleBetweenTangents (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The maximum angle between tangents theorem -/
theorem max_angle_between_tangents :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  ∀ (p : ℝ × ℝ), p ∈ C₁ →
    angleBetweenTangents p C₂ ≤ θ ∧
    ∃ (q : ℝ × ℝ), q ∈ C₁ ∧ angleBetweenTangents q C₂ = θ :=
  sorry

end max_angle_between_tangents_l1437_143784


namespace min_value_sum_reciprocals_l1437_143768

theorem min_value_sum_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → a / b + b / a ≤ x / y + y / x) ∧ 
  (a / b + b / a = 2) := by
sorry

end min_value_sum_reciprocals_l1437_143768


namespace family_ages_l1437_143747

structure Family where
  father_age : ℝ
  eldest_son_age : ℝ
  daughter_age : ℝ
  youngest_son_age : ℝ

def is_valid_family (f : Family) : Prop :=
  f.father_age = f.eldest_son_age + 20 ∧
  f.father_age + 2 = 2 * (f.eldest_son_age + 2) ∧
  f.daughter_age = f.eldest_son_age - 5 ∧
  f.youngest_son_age = f.daughter_age / 2

theorem family_ages : 
  ∃ (f : Family), is_valid_family f ∧ 
    f.father_age = 38 ∧ 
    f.eldest_son_age = 18 ∧ 
    f.daughter_age = 13 ∧ 
    f.youngest_son_age = 6.5 := by
  sorry

end family_ages_l1437_143747


namespace officers_count_l1437_143774

/-- The number of ways to choose 4 distinct officers from a group of n people -/
def choose_officers (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

/-- The number of club members -/
def club_members : ℕ := 12

/-- Theorem stating that choosing 4 officers from 12 members results in 11880 possibilities -/
theorem officers_count : choose_officers club_members = 11880 := by
  sorry

end officers_count_l1437_143774


namespace valid_digits_l1437_143736

/-- Given a digit x, construct the number 20x06 -/
def construct_number (x : Nat) : Nat := 20000 + x * 100 + 6

/-- Predicate to check if a given digit satisfies the divisibility condition -/
def is_valid_digit (x : Nat) : Prop :=
  x < 10 ∧ (construct_number x) % 7 = 0

theorem valid_digits :
  ∀ x, is_valid_digit x ↔ (x = 0 ∨ x = 7) :=
sorry

end valid_digits_l1437_143736


namespace cookie_recipe_average_l1437_143750

/-- Represents the cookie recipe and calculates the average pieces per cookie. -/
def average_pieces_per_cookie (total_cookies : ℕ) (chocolate_chips : ℕ) : ℚ :=
  let mms : ℕ := chocolate_chips / 3
  let white_chips : ℕ := mms / 2
  let raisins : ℕ := white_chips * 2
  let total_pieces : ℕ := chocolate_chips + mms + white_chips + raisins
  (total_pieces : ℚ) / total_cookies

/-- Theorem stating that the average pieces per cookie is 4.125 given the specified recipe. -/
theorem cookie_recipe_average :
  average_pieces_per_cookie 48 108 = 4.125 := by
  sorry


end cookie_recipe_average_l1437_143750


namespace dogwood_trees_after_planting_l1437_143765

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (today : ℕ) (tomorrow : ℕ) : ℕ :=
  current + today + tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem dogwood_trees_after_planting :
  total_trees 39 41 20 = 100 := by
  sorry

end dogwood_trees_after_planting_l1437_143765


namespace total_peaches_sum_l1437_143757

/-- The total number of peaches after picking more -/
def total_peaches (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem total_peaches_sum (initial picked : Float) :
  total_peaches initial picked = initial + picked := by
  sorry

end total_peaches_sum_l1437_143757


namespace least_value_x_l1437_143737

theorem least_value_x (x y z : ℕ+) (hy : y = 7) (h_least : ∀ (a b c : ℕ+), a - b - c ≥ x - y - z → a - b - c ≥ 17) : x = 25 := by
  sorry

end least_value_x_l1437_143737


namespace imaginary_part_of_z_l1437_143743

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) :
  z.im = -3 := by sorry

end imaginary_part_of_z_l1437_143743


namespace base_conversion_theorem_l1437_143744

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldl (λ sum (i, d) => sum + d * b^i) 0

theorem base_conversion_theorem :
  let base_5_123 := to_base_10 [3, 2, 1] 5
  let base_8_107 := to_base_10 [7, 0, 1] 8
  let base_9_4321 := to_base_10 [1, 2, 3, 4] 9
  (2468 / base_5_123) * base_8_107 + base_9_4321 = 7789 := by sorry

end base_conversion_theorem_l1437_143744


namespace audience_with_envelopes_l1437_143704

theorem audience_with_envelopes (total_audience : ℕ) (winners : ℕ) (winning_percentage : ℚ) :
  total_audience = 100 →
  winners = 8 →
  winning_percentage = 1/5 →
  (winners : ℚ) / (winning_percentage * total_audience) = 2/5 := by
  sorry

end audience_with_envelopes_l1437_143704


namespace hyperbola_range_theorem_l1437_143708

/-- The range of m for which the equation represents a hyperbola -/
def hyperbola_range : Set ℝ := Set.union (Set.Ioo (-1) 1) (Set.Ioi 2)

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1 ∧
  ((|m| - 1 > 0 ∧ 2 - m < 0) ∨ (|m| - 1 < 0 ∧ 2 - m > 0))

/-- Theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range_theorem :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ hyperbola_range :=
sorry

end hyperbola_range_theorem_l1437_143708


namespace area_ratio_bound_l1437_143726

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  area_pos : area > 0

/-- The result of reflecting each vertex of a quadrilateral 
    with respect to the diagonal that does not contain it -/
def reflect_vertices (q : ConvexQuadrilateral) : ℝ := 
  sorry

theorem area_ratio_bound (q : ConvexQuadrilateral) : 
  reflect_vertices q / q.area < 3 := by
  sorry

end area_ratio_bound_l1437_143726


namespace quadratic_inequality_solution_l1437_143796

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 < 0 ↔ 1/2 < x ∧ x < 2) → a = 5/2 := by
  sorry

end quadratic_inequality_solution_l1437_143796


namespace chocolate_count_l1437_143755

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces of chocolate in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of pieces of chocolate -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end chocolate_count_l1437_143755


namespace proposition_1_proposition_2_false_proposition_3_l1437_143772

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Line → Line → Prop → Prop)
variable (point_on : Prop → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- Theorem 1
theorem proposition_1 (m l : Line) (α : Plane) (A : Prop) :
  contains α m →
  perpendicular l α →
  point_on A l →
  ¬point_on A m →
  ¬coplanar l m :=
sorry

-- Theorem 2
theorem proposition_2_false (l m : Line) (α β : Plane) :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel l α →
    parallel m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

-- Theorem 3
theorem proposition_3 (l m : Line) (α β : Plane) (A : Prop) :
  contains α l →
  contains α m →
  intersect l m A →
  parallel l β →
  parallel m β →
  parallel_planes α β :=
sorry

end proposition_1_proposition_2_false_proposition_3_l1437_143772


namespace lucy_speed_calculation_l1437_143749

-- Define the cycling speeds
def eugene_speed : ℚ := 5
def carlos_relative_speed : ℚ := 4/5
def lucy_relative_speed : ℚ := 6/7

-- Theorem to prove
theorem lucy_speed_calculation :
  let carlos_speed := eugene_speed * carlos_relative_speed
  let lucy_speed := carlos_speed * lucy_relative_speed
  lucy_speed = 24/7 := by
  sorry

end lucy_speed_calculation_l1437_143749


namespace trapezoid_area_l1437_143716

/-- The area of a trapezoid with height x, one base 3x, and the other base 5x, is 4x² -/
theorem trapezoid_area (x : ℝ) (h : x > 0) : 
  let height := x
  let base1 := 3 * x
  let base2 := 5 * x
  let area := height * (base1 + base2) / 2
  area = 4 * x^2 := by
sorry

end trapezoid_area_l1437_143716


namespace circle_a_l1437_143767

theorem circle_a (x y : ℝ) :
  (x - 3)^2 + (y + 2)^2 = 16 → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4 :=
by sorry


end circle_a_l1437_143767


namespace david_average_marks_l1437_143735

def david_marks : List ℝ := [70, 63, 80, 63, 65]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by
  sorry

end david_average_marks_l1437_143735


namespace regular_polygon_sides_l1437_143721

theorem regular_polygon_sides (n : ℕ) (angle : ℝ) : 
  n > 0 → 
  angle > 0 → 
  angle < 180 → 
  (360 : ℝ) / n = angle → 
  angle = 20 → 
  n = 18 := by
sorry

end regular_polygon_sides_l1437_143721


namespace polynomial_remainder_theorem_l1437_143720

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -15) ∧ (g (-3) = -140) → c = 36/7 ∧ d = -109/7 := by
  sorry

end polynomial_remainder_theorem_l1437_143720


namespace expansion_equals_fourth_power_l1437_143705

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end expansion_equals_fourth_power_l1437_143705


namespace parallel_planes_sufficient_not_necessary_l1437_143761

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "subset of" relation for a line and a plane
variable (subset_of : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (h1 : α ≠ β)
  (h2 : subset_of m α) :
  (parallel_planes α β → parallel_line_plane m β) ∧
  ¬(parallel_line_plane m β → parallel_planes α β) :=
sorry

end parallel_planes_sufficient_not_necessary_l1437_143761


namespace twenty_dollar_bills_l1437_143741

theorem twenty_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 280) (h2 : bill_denomination = 20) :
  total_amount / bill_denomination = 14 := by
sorry

end twenty_dollar_bills_l1437_143741


namespace stationery_store_pencils_l1437_143797

theorem stationery_store_pencils (pens pencils markers : ℕ) : 
  pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
  pens * 7 = markers * 5 →  -- ratio of pens to markers is 5:7
  pencils = pens + 4 →      -- 4 more pencils than pens
  markers = pens + 20 →     -- 20 more markers than pens
  pencils = 24 :=           -- prove that the number of pencils is 24
by sorry

end stationery_store_pencils_l1437_143797


namespace large_number_proof_l1437_143789

/-- A number composed of 80 hundred millions, 5 ten millions, and 6 ten thousands -/
def large_number : ℕ := 80 * 100000000 + 5 * 10000000 + 6 * 10000

/-- The same number expressed in units of ten thousand -/
def large_number_in_ten_thousands : ℕ := large_number / 10000

theorem large_number_proof :
  large_number = 8050060000 ∧ large_number_in_ten_thousands = 805006 := by
  sorry

end large_number_proof_l1437_143789


namespace floor_equation_solution_l1437_143707

theorem floor_equation_solution (x : ℝ) : 
  ⌊(3:ℝ) * x + 4⌋ = ⌊(5:ℝ) * x - 1⌋ ↔ 
  ((11:ℝ)/5 ≤ x ∧ x < 7/3) ∨ 
  ((12:ℝ)/5 ≤ x ∧ x < 13/5) ∨ 
  ((8:ℝ)/3 ≤ x ∧ x < 14/5) :=
sorry

end floor_equation_solution_l1437_143707


namespace parabola_points_product_l1437_143792

/-- Two distinct points on a parabola with opposite slopes to a fixed point -/
structure ParabolaPoints where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  distinct : (x₁, y₁) ≠ (x₂, y₂)
  on_parabola₁ : y₁^2 = x₁
  on_parabola₂ : y₂^2 = x₂
  same_side : y₁ * y₂ > 0
  opposite_slopes : (y₁ / (x₁ - 1)) = -(y₂ / (x₂ - 1))

/-- The product of y-coordinates equals 1 -/
theorem parabola_points_product (p : ParabolaPoints) : p.y₁ * p.y₂ = 1 := by
  sorry

end parabola_points_product_l1437_143792


namespace abs_sum_equals_eight_l1437_143773

theorem abs_sum_equals_eight (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end abs_sum_equals_eight_l1437_143773


namespace solve_for_m_l1437_143731

theorem solve_for_m (x : ℝ) (m : ℝ) : 
  (-3 * x = -5 * x + 4) → 
  (m^x - 9 = 0) → 
  (m = 3 ∨ m = -3) := by
sorry

end solve_for_m_l1437_143731


namespace max_reflections_l1437_143751

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 12

/-- Represents the number of reflections -/
def n : ℕ := 7

/-- Theorem stating that n is the maximum number of reflections possible -/
theorem max_reflections (angle : ℝ) (num_reflections : ℕ) :
  angle = angle_CDA →
  num_reflections = n →
  (∀ m : ℕ, m > num_reflections → angle * m > 90) ∧
  angle * num_reflections ≤ 90 :=
sorry

#check max_reflections

end max_reflections_l1437_143751


namespace consecutive_integers_product_sum_l1437_143738

theorem consecutive_integers_product_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
  sorry

end consecutive_integers_product_sum_l1437_143738


namespace book_price_change_l1437_143714

/-- Given a book with an initial price of $400, prove that after a 15% decrease
    followed by a 40% increase, the final price is $476. -/
theorem book_price_change (initial_price : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) :
  initial_price = 400 →
  decrease_percent = 15 →
  increase_percent = 40 →
  let price_after_decrease := initial_price * (1 - decrease_percent / 100)
  let final_price := price_after_decrease * (1 + increase_percent / 100)
  final_price = 476 := by
  sorry

#check book_price_change

end book_price_change_l1437_143714


namespace manuscript_completion_time_l1437_143709

/-- The time needed to complete the manuscript when two people work together after one has worked alone for some time. -/
theorem manuscript_completion_time
  (time_A : ℝ) -- Time for person A to complete the manuscript alone
  (time_B : ℝ) -- Time for person B to complete the manuscript alone
  (solo_work : ℝ) -- Time person A works alone before B joins
  (h_A_positive : time_A > 0)
  (h_B_positive : time_B > 0)
  (h_solo_work : 0 ≤ solo_work ∧ solo_work < time_A) :
  let remaining_time := (time_A * time_B - solo_work * time_B) / (time_A + time_B)
  remaining_time = 24 / 13 :=
by sorry

end manuscript_completion_time_l1437_143709


namespace x_values_proof_l1437_143706

theorem x_values_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 7 / 8) :
  x = 1 ∨ x = 8 := by
sorry

end x_values_proof_l1437_143706


namespace product_decomposition_l1437_143722

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def options : List ℕ := [2986, 2858, 2672, 2754]

theorem product_decomposition :
  ∃! (product : ℕ) (a b : ℕ), 
    product ∈ options ∧
    is_two_digit a ∧
    is_three_digit b ∧
    product = a * b :=
sorry

end product_decomposition_l1437_143722


namespace no_common_points_condition_l1437_143770

theorem no_common_points_condition (d : ℝ) : 
  (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 → 
    ((x.1^2 + x.2^2 ≤ 4 ∧ y.1^2 + y.2^2 ≤ 9) ∨ 
     (x.1^2 + x.2^2 ≤ 9 ∧ y.1^2 + y.2^2 ≤ 4)) → 
    (x.1^2 + x.2^2 - 4) * (y.1^2 + y.2^2 - 9) > 0) ↔ 
  (0 ≤ d ∧ d < 1) ∨ d > 5 :=
sorry

end no_common_points_condition_l1437_143770


namespace triangle_problem_l1437_143787

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b / c = 1,
    then bc = 1 and the area of triangle ABC is √3 / 4 -/
theorem triangle_problem (a b c A B C : ℝ) (h1 : (b^2 + c^2 - a^2) / Real.cos A = 2)
    (h2 : (a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) :
    b * c = 1 ∧ (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end triangle_problem_l1437_143787


namespace hyperbola_circle_intersection_l1437_143794

/-- Given a hyperbola and a circle with specific properties, prove that m = 2 -/
theorem hyperbola_circle_intersection (a b m : ℝ) : 
  a > 0 → b > 0 → m > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ c, c^2 = a^2 + b^2 ∧ c/a = Real.sqrt 2) →
  (∀ x y, (x - m)^2 + y^2 = 4) →
  (∃ x y, x = y ∧ (x - m)^2 + y^2 = 4 ∧ 2 * Real.sqrt (4 - (x - m)^2) = 2 * Real.sqrt 2) →
  m = 2 := by
sorry

end hyperbola_circle_intersection_l1437_143794


namespace pure_imaginary_complex_fraction_l1437_143700

theorem pure_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
  sorry

end pure_imaginary_complex_fraction_l1437_143700


namespace ramsey_bound_exists_l1437_143785

/-- The maximum degree of a graph -/
def maxDegree (G : SimpleGraph α) : ℕ := sorry

/-- The Ramsey number of a graph -/
def ramseyNumber (G : SimpleGraph α) : ℕ := sorry

/-- The order (number of vertices) of a graph -/
def graphOrder (G : SimpleGraph α) : ℕ := sorry

/-- For every positive integer Δ, there exists a constant c such that
    all graphs H with maximum degree at most Δ have R(H) ≤ c|H| -/
theorem ramsey_bound_exists {α : Type*} :
  ∀ Δ : ℕ, Δ > 0 →
  ∃ c : ℝ, c > 0 ∧
  ∀ (H : SimpleGraph α), maxDegree H ≤ Δ →
  (ramseyNumber H : ℝ) ≤ c * (graphOrder H) :=
sorry

end ramsey_bound_exists_l1437_143785


namespace factorization_sum_l1437_143724

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) → 
  a + 2 * b = -23 := by
sorry

end factorization_sum_l1437_143724


namespace students_per_table_is_three_l1437_143779

/-- The number of students sitting at each table in Miss Smith's English class --/
def students_per_table : ℕ :=
  let total_students : ℕ := 47
  let num_tables : ℕ := 6
  let students_in_bathroom : ℕ := 3
  let students_in_canteen : ℕ := 3 * students_in_bathroom
  let new_students : ℕ := 2 * 4
  let foreign_exchange_students : ℕ := 3 * 3
  let absent_students : ℕ := students_in_bathroom + students_in_canteen + new_students + foreign_exchange_students
  let present_students : ℕ := total_students - absent_students
  present_students / num_tables

theorem students_per_table_is_three : students_per_table = 3 := by
  sorry

end students_per_table_is_three_l1437_143779


namespace triangle_side_length_l1437_143739

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.cos A = 3/5 ∧
  Real.sin B = Real.sqrt 5 / 5 ∧
  a = 2 →
  c = 11 * Real.sqrt 5 / 10 := by
sorry

end triangle_side_length_l1437_143739


namespace quadratic_inequality_coefficient_sum_l1437_143753

theorem quadratic_inequality_coefficient_sum (a b : ℝ) :
  (∀ x, ax^2 + b*x - 4 > 0 ↔ 1 < x ∧ x < 2) →
  a + b = 4 := by
sorry

end quadratic_inequality_coefficient_sum_l1437_143753


namespace small_semicircle_radius_l1437_143798

/-- Configuration of tangent shapes --/
structure TangentShapes where
  R : ℝ  -- Radius of large semicircle
  r : ℝ  -- Radius of circle
  x : ℝ  -- Radius of small semicircle

/-- Predicate for valid configuration --/
def is_valid_config (shapes : TangentShapes) : Prop :=
  shapes.R = 12 ∧ shapes.r = 6 ∧ shapes.x > 0

/-- Theorem stating the radius of the small semicircle --/
theorem small_semicircle_radius (shapes : TangentShapes) 
  (h : is_valid_config shapes) : shapes.x = 4 := by
  sorry

end small_semicircle_radius_l1437_143798


namespace dans_age_proof_l1437_143759

/-- Dan's present age -/
def dans_age : ℕ := 8

/-- Theorem stating that Dan's age after 20 years will be 7 times his age 4 years ago -/
theorem dans_age_proof : dans_age + 20 = 7 * (dans_age - 4) := by
  sorry

end dans_age_proof_l1437_143759


namespace infinite_intersecting_lines_l1437_143733

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry  -- Definition of skew lines

/-- A set of three pairwise skew lines -/
structure SkewLineTriple where
  a : Line3D
  b : Line3D
  c : Line3D
  skew_ab : are_skew a b
  skew_bc : are_skew b c
  skew_ca : are_skew c a

/-- The set of lines intersecting all three lines in a SkewLineTriple -/
def intersecting_lines (triple : SkewLineTriple) : Set Line3D :=
  sorry  -- Definition of the set of intersecting lines

/-- Theorem stating that there are infinitely many intersecting lines -/
theorem infinite_intersecting_lines (triple : SkewLineTriple) :
  Set.Infinite (intersecting_lines triple) :=
sorry

end infinite_intersecting_lines_l1437_143733


namespace outfit_combinations_l1437_143702

def num_shirts : ℕ := 8
def num_ties : ℕ := 5
def num_pants : ℕ := 4

theorem outfit_combinations : num_shirts * num_ties * num_pants = 160 := by
  sorry

end outfit_combinations_l1437_143702


namespace three_in_A_l1437_143713

def A : Set ℝ := {x | x ≤ Real.sqrt 13}

theorem three_in_A : (3 : ℝ) ∈ A := by sorry

end three_in_A_l1437_143713


namespace projection_onto_common_vector_l1437_143786

/-- Given two vectors v1 and v2 in ℝ², prove that their projection onto a common vector u results in the vector q. -/
theorem projection_onto_common_vector (v1 v2 u q : ℝ × ℝ) : 
  v1 = (3, 2) → 
  v2 = (2, 5) → 
  q = (27/8, 7/8) → 
  ∃ (t : ℝ), q = v1 + t • (v2 - v1) ∧ 
  (q - v1) • (v2 - v1) = 0 ∧ 
  (q - v2) • (v2 - v1) = 0 :=
by sorry

end projection_onto_common_vector_l1437_143786


namespace laptops_in_shop_l1437_143703

theorem laptops_in_shop (rows : ℕ) (laptops_per_row : ℕ) 
  (h1 : rows = 5) (h2 : laptops_per_row = 8) : 
  rows * laptops_per_row = 40 := by
  sorry

end laptops_in_shop_l1437_143703


namespace power_product_equality_l1437_143752

theorem power_product_equality : 3^5 * 6^5 = 1889568 := by
  sorry

end power_product_equality_l1437_143752


namespace quartic_polynomial_theorem_l1437_143711

-- Define a quartic polynomial
def is_quartic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the condition that p(n) = 1/n^2 for n = 1, 2, 3, 4, 5
def satisfies_condition (p : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ)

theorem quartic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_quartic_polynomial p) 
  (h2 : satisfies_condition p) : 
  p 6 = -67/180 := by
  sorry

end quartic_polynomial_theorem_l1437_143711


namespace horner_v3_value_l1437_143727

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: The third intermediate value (v_3) in Horner's method for f(x) at x=1 is 7.9 -/
theorem horner_v3_value : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end horner_v3_value_l1437_143727


namespace monotonicity_of_f_l1437_143790

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

theorem monotonicity_of_f (a : ℝ) :
  (a > 0 → (∀ x y, x < y → x < -2*a/3 → f a x < f a y) ∧
            (∀ x y, x < y → 0 < x → f a x < f a y) ∧
            (∀ x y, -2*a/3 < x → x < y → y < 0 → f a x > f a y)) ∧
  (a = 0 → (∀ x y, x < y → f a x < f a y)) ∧
  (a < 0 → (∀ x y, x < y → y < 0 → f a x < f a y) ∧
            (∀ x y, x < y → -2*a/3 < x → f a x < f a y) ∧
            (∀ x y, 0 < x → x < y → y < -2*a/3 → f a x > f a y)) :=
by sorry

end monotonicity_of_f_l1437_143790


namespace trigonometric_identities_l1437_143769

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.cos α + Real.cos β + Real.cos γ = 4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) + 1) ∧
  (Real.cos α + Real.cos β - Real.cos γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.sin (γ/2) - 1) :=
by sorry

end trigonometric_identities_l1437_143769


namespace problem_statements_l1437_143762

theorem problem_statements :
  -- Statement 1
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)) ∧
  -- Statement 2
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x ≤ 2 ∧ x^2 - 3*x + 2 > 0) ∧
  -- Statement 3
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- Statement 4
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end problem_statements_l1437_143762


namespace complement_of_M_in_U_l1437_143758

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_of_M_in_U : Set.compl M = Set.Icc 0 2 := by sorry

end complement_of_M_in_U_l1437_143758


namespace compute_expression_l1437_143778

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end compute_expression_l1437_143778


namespace remainder_problem_l1437_143734

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1225 % d = r ∧ 
  1681 % d = r ∧ 
  2756 % d = r → 
  d - r = 10 := by
sorry

end remainder_problem_l1437_143734


namespace intersection_sum_l1437_143777

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f and g intersect at points (10, 15) and (18, 7),
    then a + c = 28 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 10 ∨ x = 18) →
  -2 * |10 - a| + b = 15 →
  -2 * |18 - a| + b = 7 →
  2 * |10 - c| + d = 15 →
  2 * |18 - c| + d = 7 →
  a + c = 28 := by
sorry

end intersection_sum_l1437_143777


namespace gcd_102_238_l1437_143783

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l1437_143783


namespace average_popped_percentage_is_82_l1437_143718

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  popped : ℕ
  total : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def percentPopped (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem average_popped_percentage_is_82 (bag1 bag2 bag3 : PopcornBag)
    (h1 : bag1 = ⟨60, 75⟩)
    (h2 : bag2 = ⟨42, 50⟩)
    (h3 : bag3 = ⟨82, 100⟩) :
    (percentPopped bag1 + percentPopped bag2 + percentPopped bag3) / 3 = 82 := by
  sorry

end average_popped_percentage_is_82_l1437_143718


namespace union_equality_condition_min_value_of_expression_min_value_achieved_l1437_143715

-- Option B
theorem union_equality_condition (A B : Set α) :
  (A ∪ B = B) ↔ (A ∩ B = A) := by sorry

-- Option D
theorem min_value_of_expression {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  (2 * x / (x - 1) + 4 * y / (y - 1)) ≥ 6 + 4 * Real.sqrt 2 := by sorry

-- Theorem stating that the minimum value is achieved
theorem min_value_achieved {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  ∃ (x₀ y₀ : ℝ), x₀ > 1 ∧ y₀ > 1 ∧ x₀ + y₀ = x₀ * y₀ ∧
    (2 * x₀ / (x₀ - 1) + 4 * y₀ / (y₀ - 1)) = 6 + 4 * Real.sqrt 2 := by sorry

end union_equality_condition_min_value_of_expression_min_value_achieved_l1437_143715


namespace intersection_point_on_both_lines_unique_intersection_point_l1437_143746

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-1/8, 1/2)

/-- First line equation: y = -4x -/
def line1 (x y : ℚ) : Prop := y = -4 * x

/-- Second line equation: y - 2 = 12x -/
def line2 (x y : ℚ) : Prop := y - 2 = 12 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
sorry

end intersection_point_on_both_lines_unique_intersection_point_l1437_143746


namespace john_memory_card_cost_l1437_143780

/-- Calculates the amount spent on memory cards given the following conditions:
  * Pictures taken per day
  * Number of years
  * Images per memory card
  * Cost per memory card
-/
def memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (cost_per_card : ℕ) : ℕ :=
  let total_pictures := pictures_per_day * years * 365
  let cards_needed := (total_pictures + images_per_card - 1) / images_per_card
  cards_needed * cost_per_card

/-- Theorem stating that under the given conditions, John spends $13140 on memory cards -/
theorem john_memory_card_cost :
  memory_card_cost 10 3 50 60 = 13140 := by
  sorry


end john_memory_card_cost_l1437_143780


namespace day_crew_load_fraction_l1437_143763

/-- 
Proves that the day crew loads 8/11 of all boxes given the conditions about night and day crews.
-/
theorem day_crew_load_fraction 
  (D : ℝ) -- Number of boxes loaded by each day crew worker
  (W : ℝ) -- Number of workers in the day crew
  (h1 : D > 0) -- Assumption that D is positive
  (h2 : W > 0) -- Assumption that W is positive
  : (D * W) / ((D * W) + ((3/4 * D) * (1/2 * W))) = 8/11 := by
  sorry

end day_crew_load_fraction_l1437_143763


namespace complex_equation_solution_l1437_143799

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : z = Complex.I := by
  sorry

end complex_equation_solution_l1437_143799


namespace max_distance_for_given_tires_l1437_143771

/-- Represents the maximum distance a car can travel with tire switching -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 42000 56000 = 42000 := by
  sorry

end max_distance_for_given_tires_l1437_143771
