import Mathlib

namespace polygon_perimeter_is_52_l666_66643

/-- The perimeter of a polygon formed by removing six 2x2 squares from the sides of an 8x12 rectangle -/
def polygon_perimeter (rectangle_length : ℕ) (rectangle_width : ℕ) (square_side : ℕ) (num_squares : ℕ) : ℕ :=
  2 * (rectangle_length + rectangle_width) + 2 * num_squares * square_side

theorem polygon_perimeter_is_52 :
  polygon_perimeter 12 8 2 6 = 52 := by
  sorry

end polygon_perimeter_is_52_l666_66643


namespace distance_between_5th_and_29th_red_light_l666_66677

/-- Represents the color of a light in the sequence -/
inductive Color
  | Red
  | Blue
  | Green

/-- Defines the repeating pattern of lights -/
def pattern : List Color := [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue, Color.Green, Color.Green]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- Calculates the distance between two positions in the sequence -/
def distance_between (pos1 pos2 : ℕ) : ℕ :=
  (pos2 - pos1) * light_distance

/-- Converts a distance in inches to feet -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_5th_and_29th_red_light :
  inches_to_feet (distance_between (red_light_position 5) (red_light_position 29)) = 37 := by
  sorry

end distance_between_5th_and_29th_red_light_l666_66677


namespace locus_is_hyperbola_l666_66625

/-- The locus of points equidistant from two fixed points is a hyperbola -/
theorem locus_is_hyperbola (P : ℝ × ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (4, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F - dist P O = 1 →
  ∃ (a b : ℝ), (P.1 / a)^2 - (P.2 / b)^2 = 1 :=
by sorry

end locus_is_hyperbola_l666_66625


namespace fraction_difference_zero_l666_66680

theorem fraction_difference_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + b = a * b) : 
  1 / a - 1 / b = 0 := by
sorry

end fraction_difference_zero_l666_66680


namespace probability_at_least_two_succeed_l666_66646

theorem probability_at_least_two_succeed (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 1/2) (h₂ : p₂ = 1/4) (h₃ : p₃ = 1/5) : 
  p₁ * p₂ * (1 - p₃) + p₁ * (1 - p₂) * p₃ + (1 - p₁) * p₂ * p₃ + p₁ * p₂ * p₃ = 9/40 := by
  sorry

end probability_at_least_two_succeed_l666_66646


namespace square_perimeter_when_area_equals_diagonal_l666_66675

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → s^2 = s * Real.sqrt 2 → 4 * s = 4 * Real.sqrt 2 := by
  sorry

end square_perimeter_when_area_equals_diagonal_l666_66675


namespace simplify_radicals_l666_66679

theorem simplify_radicals : 
  (Real.sqrt 440 / Real.sqrt 55) - (Real.sqrt 210 / Real.sqrt 70) = 2 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end simplify_radicals_l666_66679


namespace ned_remaining_pieces_l666_66613

/-- The number of boxes Ned originally bought -/
def total_boxes : ℝ := 14.0

/-- The number of boxes Ned gave to his little brother -/
def given_boxes : ℝ := 7.0

/-- The number of pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of pieces Ned still had -/
def remaining_pieces : ℝ := (total_boxes - given_boxes) * pieces_per_box

theorem ned_remaining_pieces :
  remaining_pieces = 42.0 := by sorry

end ned_remaining_pieces_l666_66613


namespace equilateral_triangle_perimeter_l666_66653

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l666_66653


namespace total_flour_used_l666_66657

-- Define the ratios
def cake_ratio : Fin 3 → ℕ
  | 0 => 3  -- flour
  | 1 => 2  -- butter
  | 2 => 1  -- sugar
  | _ => 0

def cream_ratio : Fin 2 → ℕ
  | 0 => 2  -- butter
  | 1 => 3  -- sugar
  | _ => 0

def cookie_ratio : Fin 3 → ℕ
  | 0 => 5  -- flour
  | 1 => 3  -- butter
  | 2 => 2  -- sugar
  | _ => 0

-- Define the additional flour
def additional_flour : ℕ := 300

-- Theorem statement
theorem total_flour_used (x y : ℕ) :
  (3 * x + additional_flour) / (2 * x + 2 * y) = 5 / 3 →
  (2 * x + 2 * y) / (x + 3 * y) = 3 / 2 →
  3 * x + additional_flour = 1200 :=
by sorry

end total_flour_used_l666_66657


namespace hcf_problem_l666_66600

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 128) :
  Nat.gcd a b = 20 := by
  sorry

end hcf_problem_l666_66600


namespace ship_journey_distance_l666_66670

/-- The total distance traveled by a ship in three days -/
def ship_total_distance (first_day_distance : ℝ) : ℝ :=
  let second_day_distance := 3 * first_day_distance
  let third_day_distance := second_day_distance + 110
  first_day_distance + second_day_distance + third_day_distance

/-- Theorem stating the total distance traveled by the ship -/
theorem ship_journey_distance : ship_total_distance 100 = 810 := by
  sorry

end ship_journey_distance_l666_66670


namespace max_non_intersecting_diagonals_correct_l666_66626

/-- The maximum number of non-intersecting diagonals in a convex n-gon -/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem: The maximum number of non-intersecting diagonals in a convex n-gon is n - 3 -/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end max_non_intersecting_diagonals_correct_l666_66626


namespace second_year_increase_is_twenty_percent_l666_66619

/-- Calculates the percentage increase in the second year given initial population,
    first year increase, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let after_first_year := initial_population * (1 + first_year_increase)
  let second_year_factor := final_population / after_first_year
  (second_year_factor - 1) * 100

/-- Theorem stating that given the problem conditions, the second year increase is 20%. -/
theorem second_year_increase_is_twenty_percent :
  second_year_increase 1000 (10 / 100) 1320 = 20 := by
  sorry

#eval second_year_increase 1000 (10 / 100) 1320

end second_year_increase_is_twenty_percent_l666_66619


namespace triangle_theorem_l666_66647

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = t.a * t.b

def angleBisectorIntersection (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the angle bisector condition
  True

def cdLength (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This represents CD = 2
  True

-- Theorem statement
theorem triangle_theorem (t : Triangle) (D : ℝ × ℝ) :
  satisfiesCondition t →
  angleBisectorIntersection t D →
  cdLength t D →
  (t.C = 2 * Real.pi / 3) ∧
  (∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧
    ∀ (a b : ℝ), a > 0 ∧ b > 0 → 2 * a + b ≥ min) :=
by sorry

end triangle_theorem_l666_66647


namespace set_equality_condition_l666_66620

-- Define set A
def A : Set ℝ := {x | (x + 1)^2 * (2 - x) / (4 + x) ≥ 0 ∧ x ≠ -4}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2*a + 1) ≤ 0}

-- Theorem statement
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ -3/2 < a ∧ a ≤ 3/2 := by sorry

end set_equality_condition_l666_66620


namespace function_solution_set_l666_66697

-- Define the function f
def f (x a : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by
  sorry

end function_solution_set_l666_66697


namespace ice_cream_flavors_l666_66656

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors Ice-cream-o-rama can create -/
theorem ice_cream_flavors : distribute 6 4 = 84 := by
  sorry

end ice_cream_flavors_l666_66656


namespace arithmetic_sequence_sum_l666_66623

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 8 = 3 →
  a 1 + a 10 = 3 :=
by
  sorry


end arithmetic_sequence_sum_l666_66623


namespace arithmetic_sequence_minimum_l666_66688

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (a 2 * a 12).sqrt = 4 →  -- Geometric mean of a_2 and a_12 is 4
  (∃ r : ℝ, ∀ n, a (n + 1) = a n + r) →  -- Arithmetic sequence
  (∃ m : ℝ, ∀ r : ℝ, 2 * a 5 + 8 * a 9 ≥ m) →  -- Minimum exists
  a 3 = 4 :=
by sorry

end arithmetic_sequence_minimum_l666_66688


namespace slower_train_time_l666_66635

/-- Represents a train traveling between two stations -/
structure Train where
  speed : ℝ
  remainingDistance : ℝ

/-- The problem setup -/
def trainProblem (fasterTrain slowerTrain : Train) : Prop :=
  fasterTrain.speed = 3 * slowerTrain.speed ∧
  fasterTrain.remainingDistance = slowerTrain.remainingDistance ∧
  fasterTrain.remainingDistance = 4 * fasterTrain.speed

/-- The theorem to prove -/
theorem slower_train_time
    (fasterTrain slowerTrain : Train)
    (h : trainProblem fasterTrain slowerTrain) :
    slowerTrain.remainingDistance / slowerTrain.speed = 12 := by
  sorry

#check slower_train_time

end slower_train_time_l666_66635


namespace equation_equivalence_l666_66608

theorem equation_equivalence (x : ℝ) : 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end equation_equivalence_l666_66608


namespace sara_picked_37_peaches_l666_66629

/-- The number of peaches Sara picked -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sara picked 37 peaches -/
theorem sara_picked_37_peaches (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 24)
  (h2 : final_peaches = 61) :
  peaches_picked initial_peaches final_peaches = 37 := by
  sorry

#check sara_picked_37_peaches

end sara_picked_37_peaches_l666_66629


namespace smallest_valid_n_l666_66611

def is_valid_n (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (10 * (n % 10) + n / 10 - 5 = 2 * n)

theorem smallest_valid_n :
  is_valid_n 13 ∧ ∀ m, is_valid_n m → m ≥ 13 := by
  sorry

end smallest_valid_n_l666_66611


namespace not_necessarily_equal_numbers_l666_66684

theorem not_necessarily_equal_numbers : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a + b^2 + c^2 = b + a^2 + c^2) ∧
  (a + b^2 + c^2 = c + a^2 + b^2) ∧
  ¬(a = b ∧ b = c) := by
  sorry

end not_necessarily_equal_numbers_l666_66684


namespace race_distance_l666_66655

/-- Calculates the total distance of a race given the conditions --/
theorem race_distance (speed_a speed_b : ℝ) (head_start winning_margin : ℝ) : 
  speed_a / speed_b = 5 / 4 →
  head_start = 100 →
  winning_margin = 200 →
  (speed_a * ((head_start + winning_margin) / speed_b)) - head_start = 600 :=
by
  sorry


end race_distance_l666_66655


namespace polynomial_division_theorem_l666_66621

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 2*x^5 + 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 12 = 
  (x - 1) * (x^5 - x^4 + 2*x^3 - 2*x^2 + 3*x - 3) + 9 := by
  sorry

end polynomial_division_theorem_l666_66621


namespace interest_rate_calculation_l666_66603

theorem interest_rate_calculation (principal : ℝ) (difference : ℝ) (time : ℕ) (rate : ℝ) : 
  principal = 15000 →
  difference = 150 →
  time = 2 →
  principal * ((1 + rate)^time - 1) - principal * rate * time = difference →
  rate = 0.1 := by
sorry

end interest_rate_calculation_l666_66603


namespace fraction_pair_sum_equality_l666_66637

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end fraction_pair_sum_equality_l666_66637


namespace circle_center_coordinates_l666_66666

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 48
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = -12
def centerLine (x y : ℝ) : Prop := x - y = 0

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  isTangentToLine1 : line1 center.1 center.2
  isTangentToLine2 : line2 center.1 center.2
  centerOnLine : centerLine center.1 center.2

-- Theorem statement
theorem circle_center_coordinates :
  ∀ (c : Circle), c.center = (18/7, 18/7) :=
sorry

end circle_center_coordinates_l666_66666


namespace line_segment_coordinates_l666_66695

theorem line_segment_coordinates (y : ℝ) : 
  y > 0 → 
  ((2 - 6)^2 + (y - 10)^2 = 10^2) →
  (y = 10 - 2 * Real.sqrt 21 ∨ y = 10 + 2 * Real.sqrt 21) :=
by sorry

end line_segment_coordinates_l666_66695


namespace unique_solution_exponential_equation_l666_66676

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) * (2 : ℝ)^2 ∧ x = -6 := by
  sorry

end unique_solution_exponential_equation_l666_66676


namespace largest_n_power_inequality_l666_66698

theorem largest_n_power_inequality : ∃ (n : ℕ), n = 11 ∧ 
  (∀ m : ℕ, m^200 < 5^300 → m ≤ n) ∧ n^200 < 5^300 :=
by sorry

end largest_n_power_inequality_l666_66698


namespace teacher_selection_plans_l666_66662

theorem teacher_selection_plans (male_teachers female_teachers selected_teachers : ℕ) 
  (h1 : male_teachers = 5)
  (h2 : female_teachers = 4)
  (h3 : selected_teachers = 3) :
  (Nat.choose male_teachers 2 * Nat.choose female_teachers 1 * Nat.factorial selected_teachers) +
  (Nat.choose male_teachers 1 * Nat.choose female_teachers 2 * Nat.factorial selected_teachers) = 420 := by
  sorry

end teacher_selection_plans_l666_66662


namespace problem_solution_l666_66687

theorem problem_solution (a b : ℚ) 
  (h1 : 7 * a + 3 * b = 0) 
  (h2 : b - 4 = a) : 
  9 * b = 126 / 5 := by
  sorry

end problem_solution_l666_66687


namespace part_one_part_two_l666_66609

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Theorem for part 1
theorem part_one : (Set.univ \ A) ∩ B 1 = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for part 2
theorem part_two : ∀ a : ℝ, (Set.univ \ A) ∩ B a = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end part_one_part_two_l666_66609


namespace girls_trying_out_l666_66678

theorem girls_trying_out (girls : ℕ) (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 32 →
  called_back = 10 →
  didnt_make_cut = 39 →
  girls + boys = called_back + didnt_make_cut →
  girls = 17 := by
  sorry

end girls_trying_out_l666_66678


namespace slower_train_speed_l666_66632

/-- Prove that given two trains moving in the same direction, with the faster train
    traveling at 50 km/hr, taking 15 seconds to pass a man in the slower train,
    and having a length of 75 meters, the speed of the slower train is 32 km/hr. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (faster_train_length : ℝ)
  (h1 : faster_train_speed = 50)
  (h2 : passing_time = 15)
  (h3 : faster_train_length = 75) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 32 ∧
    (faster_train_speed - slower_train_speed) * 1000 / 3600 = faster_train_length / passing_time :=
by sorry

end slower_train_speed_l666_66632


namespace triangle_kite_property_l666_66664

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define points D, H, M, N
variable (D H M N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_angle_bisector (A D B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_altitude (A H B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_circle (M B D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_kite (A M H N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem triangle_kite_property 
  (h_acute : is_acute_triangle A B C)
  (h_bisector : is_angle_bisector A D B C)
  (h_altitude : is_altitude A H B C)
  (h_circle_M : on_circle M B D)
  (h_circle_N : on_circle N C D) :
  is_kite A M H N :=
sorry

end triangle_kite_property_l666_66664


namespace triangle_area_theorem_l666_66602

/-- Given a triangle with vertices (0, 0), (x, 3x), and (2x, 0), 
    if its area is 150 square units and x > 0, then x = 5√2 -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * (3*x) = 150 → x = 5 * Real.sqrt 2 := by
  sorry

end triangle_area_theorem_l666_66602


namespace q_profit_share_l666_66624

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateProfitShare (investmentP investmentQ totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentP + investmentQ
  let shareQ := (investmentQ * totalProfit) / totalInvestment
  shareQ

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit --/
theorem q_profit_share :
  calculateProfitShare 54000 36000 18000 = 7200 := by
  sorry

#eval calculateProfitShare 54000 36000 18000

end q_profit_share_l666_66624


namespace angle_B_is_70_l666_66667

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def rightTriangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A = 20 ∧ t.C = 90

-- Theorem statement
theorem angle_B_is_70 (t : Triangle) (h : rightTriangle t) : t.B = 70 := by
  sorry

end angle_B_is_70_l666_66667


namespace subset_condition_l666_66674

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem subset_condition (a : ℝ) : B ⊆ A a ↔ a ≥ 3 := by
  sorry

end subset_condition_l666_66674


namespace equation_transformation_l666_66682

theorem equation_transformation (x : ℝ) : 
  ((x - 1) / 2 - 1 = (3 * x + 1) / 3) ↔ (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by sorry

end equation_transformation_l666_66682


namespace starters_count_l666_66683

/-- The number of ways to select 7 starters from a team of 16 players,
    including a set of twins, with the condition that at least one but
    no more than two twins must be included. -/
def select_starters (total_players : ℕ) (num_twins : ℕ) (num_starters : ℕ) : ℕ :=
  let non_twin_players := total_players - num_twins
  let one_twin := num_twins * Nat.choose non_twin_players (num_starters - 1)
  let both_twins := Nat.choose non_twin_players (num_starters - num_twins)
  one_twin + both_twins

theorem starters_count :
  select_starters 16 2 7 = 8008 := by
  sorry

end starters_count_l666_66683


namespace largest_angle_and_sinC_l666_66699

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle with a = 7, b = 3, c = 5 -/
def givenTriangle : Triangle where
  a := 7
  b := 3
  c := 5
  A := sorry
  B := sorry
  C := sorry

theorem largest_angle_and_sinC (t : Triangle) (h : t = givenTriangle) :
  (t.A > t.B ∧ t.A > t.C) ∧ t.A = Real.pi * (2/3) ∧ Real.sin t.C = 5 * Real.sqrt 3 / 14 := by
  sorry

#check largest_angle_and_sinC

end largest_angle_and_sinC_l666_66699


namespace product_xy_equals_four_l666_66693

-- Define variables
variable (a b x y : ℕ)

-- State the theorem
theorem product_xy_equals_four
  (h1 : x = a)
  (h2 : y = b)
  (h3 : a + a = b * a)
  (h4 : y = a)
  (h5 : a * a = a + a)
  (h6 : b = 3) :
  x * y = 4 := by
sorry

end product_xy_equals_four_l666_66693


namespace sum_of_digits_power_of_two_l666_66606

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Main theorem -/
theorem sum_of_digits_power_of_two : sum_of_digits (sum_of_digits (sum_of_digits (2^2006))) = 4 := by
  sorry


end sum_of_digits_power_of_two_l666_66606


namespace number_line_properties_l666_66601

-- Definition of distance between points on a number line
def distance (a b : ℚ) : ℚ := |a - b|

-- Statements to prove
theorem number_line_properties :
  -- 1. The distance between 2 and 5 is 3
  distance 2 5 = 3 ∧
  -- 2. The distance between x and -6 is |x + 6|
  ∀ x : ℚ, distance x (-6) = |x + 6| ∧
  -- 3. For -2 < x < 2, |x-2|+|x+2| = 4
  ∀ x : ℚ, -2 < x → x < 2 → |x-2|+|x+2| = 4 ∧
  -- 4. For |x-1|+|x+3| > 4, x > 1 or x < -3
  ∀ x : ℚ, |x-1|+|x+3| > 4 → x > 1 ∨ x < -3 ∧
  -- 5. The minimum value of |x-3|+|x+2|+|x+1| is 5, occurring at x = -1
  (∀ x : ℚ, |x-3|+|x+2|+|x+1| ≥ 5) ∧ (|-1-3|+|-1+2|+|-1+1| = 5) ∧
  -- 6. The maximum value of y when |x-1|+|x+2|=10-|y-3|-|y+4| is 3
  ∀ x y : ℚ, |x-1|+|x+2| = 10-|y-3|-|y+4| → y ≤ 3 :=
by sorry

end number_line_properties_l666_66601


namespace lending_time_problem_l666_66652

/-- The problem of finding the lending time for the second part of a sum --/
theorem lending_time_problem (total_sum : ℝ) (second_part : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) :
  total_sum = 2743 →
  second_part = 1688 →
  rate1 = 0.03 →
  time1 = 8 →
  rate2 = 0.05 →
  (total_sum - second_part) * rate1 * time1 = second_part * rate2 * 3 :=
by
  sorry

#check lending_time_problem

end lending_time_problem_l666_66652


namespace sandys_carrots_l666_66685

/-- Sandy's carrot problem -/
theorem sandys_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) 
  (h1 : initial_carrots = 6)
  (h2 : taken_carrots = 3) :
  initial_carrots - taken_carrots = 3 := by
  sorry

end sandys_carrots_l666_66685


namespace problem_solution_l666_66628

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end problem_solution_l666_66628


namespace employee_share_l666_66640

def total_profit : ℝ := 50
def num_employees : ℕ := 9
def self_percentage : ℝ := 0.1

theorem employee_share : 
  (total_profit - self_percentage * total_profit) / num_employees = 5 := by
sorry

end employee_share_l666_66640


namespace points_collinear_l666_66651

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for three points -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem points_collinear : 
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨3, 8⟩
  let p3 : Point2D := ⟨4, 11⟩
  collinear p1 p2 p3 := by
  sorry

end points_collinear_l666_66651


namespace product_minus_constant_l666_66604

theorem product_minus_constant (P Q R S : ℕ+) : 
  (P + Q + R + S : ℝ) = 104 →
  (P : ℝ) + 5 = (Q : ℝ) - 5 →
  (P : ℝ) + 5 = (R : ℝ) * 2 →
  (P : ℝ) + 5 = (S : ℝ) / 2 →
  (P : ℝ) * (Q : ℝ) * (R : ℝ) * (S : ℝ) - 200 = 267442.5 := by
sorry

end product_minus_constant_l666_66604


namespace chord_length_circle_line_l666_66650

/-- The length of the chord intersected by a line on a circle -/
theorem chord_length_circle_line (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + y^2 = 4
  let line := fun (x y : ℝ) => 4*x - 3*y - 3 = 0
  let center := (2, 0)
  let radius := 2
  let d := |4*2 - 3*0 - 3| / Real.sqrt (4^2 + (-3)^2)
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle x₁ y₁ ∧ circle x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * (radius^2 - d^2) :=
by
  sorry

#check chord_length_circle_line

end chord_length_circle_line_l666_66650


namespace race_time_differences_l666_66641

def race_distance : ℝ := 10
def john_speed : ℝ := 15
def alice_time : ℝ := 48
def bob_time : ℝ := 52
def charlie_time : ℝ := 55

theorem race_time_differences :
  let john_time := race_distance / john_speed * 60
  (alice_time - john_time = 8) ∧
  (bob_time - john_time = 12) ∧
  (charlie_time - john_time = 15) := by
  sorry

end race_time_differences_l666_66641


namespace officer_hopps_ticket_goal_l666_66630

theorem officer_hopps_ticket_goal :
  let days_in_may : ℕ := 31
  let first_period_days : ℕ := 15
  let first_period_average : ℕ := 8
  let second_period_average : ℕ := 5
  let second_period_days : ℕ := days_in_may - first_period_days
  let first_period_tickets : ℕ := first_period_days * first_period_average
  let second_period_tickets : ℕ := second_period_days * second_period_average
  let total_tickets : ℕ := first_period_tickets + second_period_tickets
  total_tickets = 200 := by
sorry

end officer_hopps_ticket_goal_l666_66630


namespace mary_picked_nine_lemons_l666_66618

/-- The number of lemons picked by Sally -/
def sally_lemons : ℕ := 7

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons picked by Mary -/
def mary_lemons : ℕ := total_lemons - sally_lemons

theorem mary_picked_nine_lemons : mary_lemons = 9 := by
  sorry

end mary_picked_nine_lemons_l666_66618


namespace seven_distinct_reverse_numbers_l666_66607

def is_reverse_after_adding_18 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ n + 18 = (n % 10) * 10 + (n / 10)

theorem seven_distinct_reverse_numbers :
  ∃ (S : Finset ℕ), S.card = 7 ∧ ∀ n ∈ S, is_reverse_after_adding_18 n ∧
    ∀ m ∈ S, m ≠ n → m ≠ n := by
  sorry

end seven_distinct_reverse_numbers_l666_66607


namespace flagpole_height_l666_66654

/-- Given a right triangle with hypotenuse 5 meters, where a person of height 1.6 meters
    touches the hypotenuse at a point 4 meters from one end of the base,
    prove that the height of the triangle (perpendicular to the base) is 8 meters. -/
theorem flagpole_height (base : ℝ) (hypotenuse : ℝ) (person_height : ℝ) (person_distance : ℝ) :
  base = 5 →
  person_height = 1.6 →
  person_distance = 4 →
  ∃ (height : ℝ), height = 8 ∧ height * base = person_height * person_distance :=
by sorry

end flagpole_height_l666_66654


namespace gcd_bound_from_lcm_l666_66616

theorem gcd_bound_from_lcm (a b : ℕ) : 
  a ≥ 1000000 ∧ a < 10000000 ∧ 
  b ≥ 1000000 ∧ b < 10000000 ∧ 
  Nat.lcm a b ≥ 100000000000 ∧ Nat.lcm a b < 1000000000000 →
  Nat.gcd a b < 1000 := by
sorry

end gcd_bound_from_lcm_l666_66616


namespace triangles_cover_two_thirds_l666_66642

/-- Represents a tiling unit in the pattern -/
structure TilingUnit where
  /-- Side length of smaller shapes (triangles and squares) -/
  small_side : ℝ
  /-- Number of triangles in the unit -/
  num_triangles : ℕ
  /-- Number of squares in the unit -/
  num_squares : ℕ
  /-- Assertion that there are 2 triangles and 3 squares -/
  shape_count : num_triangles = 2 ∧ num_squares = 3
  /-- Assertion that all shapes have equal area -/
  equal_area : small_side^2 = 2 * (small_side^2 / 2)
  /-- Side length of the larger square formed by the unit -/
  large_side : ℝ
  /-- Assertion that large side is 3 times the small side -/
  side_relation : large_side = 3 * small_side

/-- Theorem stating that triangles cover 2/3 of the total area -/
theorem triangles_cover_two_thirds (u : TilingUnit) :
  (u.num_triangles * (u.small_side^2 / 2)) / u.large_side^2 = 2 / 3 := by
  sorry

end triangles_cover_two_thirds_l666_66642


namespace investment_change_l666_66692

theorem investment_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * 1.4
  let day2_value := day1_value * 0.75
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end investment_change_l666_66692


namespace savings_calculation_l666_66668

/-- Calculates the total savings over a 4-month period with varying weekly savings rates --/
def total_savings (smartphone_cost initial_savings gym_membership first_period_weekly_savings second_period_weekly_savings : ℕ) : ℕ :=
  let first_period_savings := first_period_weekly_savings * 4 * 2
  let second_period_savings := second_period_weekly_savings * 4 * 2
  first_period_savings + second_period_savings

/-- Proves that given the specified conditions, the total savings after 4 months is $1040 --/
theorem savings_calculation (smartphone_cost : ℕ) (initial_savings : ℕ) (gym_membership : ℕ) 
  (h1 : smartphone_cost = 800)
  (h2 : initial_savings = 200)
  (h3 : gym_membership = 50)
  (h4 : total_savings 800 200 50 50 80 = 1040) :
  total_savings smartphone_cost initial_savings gym_membership 50 80 = 1040 :=
by sorry

#eval total_savings 800 200 50 50 80

end savings_calculation_l666_66668


namespace solar_panel_installation_l666_66631

theorem solar_panel_installation
  (total_homes : ℕ)
  (panels_per_home : ℕ)
  (shortage : ℕ)
  (h1 : total_homes = 20)
  (h2 : panels_per_home = 10)
  (h3 : shortage = 50)
  : (total_homes * panels_per_home - shortage) / panels_per_home = 15 := by
  sorry

end solar_panel_installation_l666_66631


namespace sin_sum_product_zero_l666_66673

theorem sin_sum_product_zero : 
  Real.sin (523 * π / 180) * Real.sin (943 * π / 180) + 
  Real.sin (1333 * π / 180) * Real.sin (313 * π / 180) = 0 := by
  sorry

end sin_sum_product_zero_l666_66673


namespace kamals_math_marks_l666_66649

def english_marks : ℕ := 96
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end kamals_math_marks_l666_66649


namespace max_value_inequality_l666_66612

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x^2 + 1/y^2) * (x^2 + 1/y^2 - 100) + (y^2 + 1/x^2) * (y^2 + 1/x^2 - 100) ≤ -5000 := by
sorry

end max_value_inequality_l666_66612


namespace absolute_value_integral_l666_66610

theorem absolute_value_integral : ∫ x in (0:ℝ)..(4:ℝ), |x - 2| = 4 := by
  sorry

end absolute_value_integral_l666_66610


namespace z_squared_minus_norm_squared_l666_66665

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Theorem statement
theorem z_squared_minus_norm_squared :
  z^2 - Complex.abs z^2 = 2 * Complex.I - 2 := by
  sorry

end z_squared_minus_norm_squared_l666_66665


namespace emily_subtraction_l666_66694

theorem emily_subtraction : 50^2 - 49^2 = 99 := by
  sorry

end emily_subtraction_l666_66694


namespace probability_not_red_l666_66660

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10
def non_red_jelly_beans : ℕ := 8 + 9 + 10

theorem probability_not_red :
  (non_red_jelly_beans : ℚ) / total_jelly_beans = 27 / 34 := by
  sorry

end probability_not_red_l666_66660


namespace monotonic_sine_range_l666_66627

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/3, π/4] iff 0 < ω ≤ 12/7 -/
theorem monotonic_sine_range (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), Monotone (fun x => 2 * Real.sin (ω * x))) ↔
  ω ≤ 12/7 := by
  sorry

end monotonic_sine_range_l666_66627


namespace largest_2023_digit_prime_squared_minus_one_div_30_l666_66639

/-- p is the largest prime with 2023 digits -/
def p : Nat := sorry

/-- p^2 - 1 is divisible by 30 -/
theorem largest_2023_digit_prime_squared_minus_one_div_30 : 
  30 ∣ (p^2 - 1) := by sorry

end largest_2023_digit_prime_squared_minus_one_div_30_l666_66639


namespace arccos_zero_l666_66614

theorem arccos_zero (h : Set.Icc 0 π = Set.range acos) : acos 0 = π / 2 := by sorry

end arccos_zero_l666_66614


namespace system_solution_l666_66638

theorem system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

end system_solution_l666_66638


namespace matrix_equation_solution_l666_66605

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_equation_solution 
  (B : Matrix n n ℝ) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 • (1 : Matrix n n ℝ)) * (B - 5 • (1 : Matrix n n ℝ)) = 0) :
  B + 9 • B⁻¹ = 8 • (1 : Matrix n n ℝ) := by
sorry

end matrix_equation_solution_l666_66605


namespace some_number_value_l666_66622

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * x * 49) : x = 315 := by
  sorry

end some_number_value_l666_66622


namespace P_in_fourth_quadrant_l666_66690

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 2, y := -5 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end P_in_fourth_quadrant_l666_66690


namespace sum_of_incircle_areas_l666_66663

/-- Given a triangle ABC with side lengths a, b, c and inradius r, 
    the sum of the areas of its incircle and the incircles of the three smaller triangles 
    formed by tangent lines to the incircle parallel to the sides of ABC 
    is equal to (7πr²)/4. -/
theorem sum_of_incircle_areas (a b c r : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  r = K / s →
  (π * r^2) + 3 * (π * (r/2)^2) = (7 * π * r^2) / 4 := by
  sorry

end sum_of_incircle_areas_l666_66663


namespace daniel_driving_speed_l666_66644

/-- The speed at which Daniel drove on Monday for the first 32 miles -/
def monday_first_speed (x : ℝ) : ℝ := 2 * x

theorem daniel_driving_speed (x : ℝ) (h_x_pos : x > 0) :
  let total_distance : ℝ := 100
  let monday_first_distance : ℝ := 32
  let monday_second_distance : ℝ := total_distance - monday_first_distance
  let sunday_time : ℝ := total_distance / x
  let monday_time : ℝ := monday_first_distance / (monday_first_speed x) + monday_second_distance / (x / 2)
  let time_increase_ratio : ℝ := 1.52
  monday_time = time_increase_ratio * sunday_time :=
by sorry

#check daniel_driving_speed

end daniel_driving_speed_l666_66644


namespace oscar_voting_theorem_l666_66615

/-- Represents a vote for an actor and an actress -/
structure Vote where
  actor : ℕ
  actress : ℕ

/-- The problem statement -/
theorem oscar_voting_theorem 
  (votes : Finset Vote) 
  (vote_count : votes.card = 3366)
  (unique_counts : ∀ n : ℕ, 1 ≤ n → n ≤ 100 → 
    (∃ a : ℕ, (votes.filter (λ v => v.actor = a)).card = n) ∨ 
    (∃ b : ℕ, (votes.filter (λ v => v.actress = b)).card = n)) :
  ∃ v₁ v₂ : Vote, v₁ ∈ votes ∧ v₂ ∈ votes ∧ v₁ ≠ v₂ ∧ v₁.actor = v₂.actor ∧ v₁.actress = v₂.actress :=
by
  sorry

end oscar_voting_theorem_l666_66615


namespace paige_homework_problem_l666_66672

/-- The number of problems Paige has left to do for homework -/
def problems_left (math science history language_arts finished_at_school unfinished_math : ℕ) : ℕ :=
  math + science + history + language_arts - finished_at_school + unfinished_math

theorem paige_homework_problem :
  problems_left 43 12 10 5 44 3 = 29 := by
  sorry

end paige_homework_problem_l666_66672


namespace power_of_five_reciprocal_l666_66681

theorem power_of_five_reciprocal (x y : ℕ) : 
  (2^x : ℕ) ∣ 144 ∧ 
  (∀ k > x, ¬((2^k : ℕ) ∣ 144)) ∧ 
  (3^y : ℕ) ∣ 144 ∧ 
  (∀ k > y, ¬((3^k : ℕ) ∣ 144)) →
  (1/5 : ℚ)^(y - x) = 25 := by
sorry

end power_of_five_reciprocal_l666_66681


namespace units_digit_of_first_four_composites_product_l666_66634

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem units_digit_of_first_four_composites_product :
  (product_of_list first_four_composites) % 10 = 8 := by
  sorry

end units_digit_of_first_four_composites_product_l666_66634


namespace sum_of_circle_areas_l666_66686

/-- Represents a right triangle with sides 6, 8, and 10, where the vertices
    are centers of mutually externally tangent circles -/
structure TriangleWithCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 8 -/
  r : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 6 -/
  s : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 10 -/
  t : ℝ
  /-- The sum of radii of circles centered at vertices adjacent to side 6 equals 6 -/
  adj_6 : r + s = 6
  /-- The sum of radii of circles centered at vertices adjacent to side 8 equals 8 -/
  adj_8 : s + t = 8
  /-- The sum of radii of circles centered at vertices adjacent to side 10 equals 10 -/
  adj_10 : r + t = 10

/-- The sum of the areas of the three circles in a TriangleWithCircles is 56π -/
theorem sum_of_circle_areas (twc : TriangleWithCircles) :
  π * (twc.r^2 + twc.s^2 + twc.t^2) = 56 * π := by
  sorry

end sum_of_circle_areas_l666_66686


namespace coefficient_of_x_l666_66645

/-- The coefficient of x in the simplified expression 5(2x - 3) + 7(10 - 3x^2 + 2x) - 9(4x - 2) is -12 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 5*(2*x - 3) + 7*(10 - 3*x^2 + 2*x) - 9*(4*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-12)*x + b + c := by
sorry

end coefficient_of_x_l666_66645


namespace remaining_length_is_twelve_l666_66671

/-- Represents a rectangle with specific dimensions -/
structure Rectangle :=
  (left : ℝ)
  (top1 : ℝ)
  (top2 : ℝ)
  (top3 : ℝ)

/-- Calculates the total length of remaining segments after removing sides -/
def remaining_length (r : Rectangle) : ℝ :=
  r.left + r.top1 + r.top2 + r.top3

/-- Theorem stating that for a rectangle with given dimensions, 
    the remaining length after removing sides is 12 units -/
theorem remaining_length_is_twelve (r : Rectangle) 
  (h1 : r.left = 8)
  (h2 : r.top1 = 2)
  (h3 : r.top2 = 1)
  (h4 : r.top3 = 1) :
  remaining_length r = 12 := by
  sorry

#check remaining_length_is_twelve

end remaining_length_is_twelve_l666_66671


namespace largest_multiple_of_15_under_500_l666_66659

theorem largest_multiple_of_15_under_500 :
  ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 :=
by sorry

end largest_multiple_of_15_under_500_l666_66659


namespace division_remainder_problem_l666_66689

theorem division_remainder_problem (x y : ℤ) (r : ℕ) 
  (h1 : x > 0)
  (h2 : x = 10 * y + r)
  (h3 : 0 ≤ r ∧ r < 10)
  (h4 : 2 * x = 7 * (3 * y) + 1)
  (h5 : 11 * y - x = 2) :
  r = 3 := by sorry

end division_remainder_problem_l666_66689


namespace negation_of_proposition_l666_66636

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l666_66636


namespace farmer_field_area_l666_66661

/-- Represents the farmer's field ploughing problem -/
def FarmerField (initial_productivity : ℝ) (productivity_increase : ℝ) (days_saved : ℕ) : Prop :=
  ∃ (total_days : ℕ) (field_area : ℝ),
    field_area = initial_productivity * total_days ∧
    field_area = (2 * initial_productivity) + 
      ((total_days - days_saved - 2) * (initial_productivity * (1 + productivity_increase))) ∧
    field_area = 1440

/-- Theorem stating that the field area is 1440 hectares given the problem conditions -/
theorem farmer_field_area :
  FarmerField 120 0.25 2 :=
sorry

end farmer_field_area_l666_66661


namespace kevin_sells_50_crates_l666_66633

/-- Kevin's weekly fruit sales --/
def weekly_fruit_sales (grapes mangoes passion_fruits : ℕ) : ℕ :=
  grapes + mangoes + passion_fruits

/-- Theorem: Kevin sells 50 crates of fruit per week --/
theorem kevin_sells_50_crates :
  weekly_fruit_sales 13 20 17 = 50 := by
  sorry

end kevin_sells_50_crates_l666_66633


namespace max_coefficients_bound_l666_66691

variable (p q x y A B C α β γ : ℝ)

theorem max_coefficients_bound 
  (h_p : 0 ≤ p ∧ p ≤ 1) 
  (h_q : 0 ≤ q ∧ q ≤ 1) 
  (h_eq1 : ∀ x y, (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2)
  (h_eq2 : ∀ x y, (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2) :
  max A (max B C) ≥ 4/9 ∧ max α (max β γ) ≥ 4/9 :=
by sorry

end max_coefficients_bound_l666_66691


namespace complex_fraction_simplification_l666_66648

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - i) / (3 + 4 * i) = 2 / 5 - 11 / 25 * i :=
by
  sorry

end complex_fraction_simplification_l666_66648


namespace product_of_solutions_l666_66658

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|5 * x₁ - 1| + 4 = 54) → 
  (|5 * x₂ - 1| + 4 = 54) → 
  x₁ ≠ x₂ →
  x₁ * x₂ = -99.96 := by
sorry

end product_of_solutions_l666_66658


namespace percentage_and_reduction_l666_66617

-- Define the relationship between two numbers
def is_five_percent_more (a b : ℝ) : Prop := a = b * 1.05

-- Define the reduction of 10 kilograms by 10%
def reduced_by_ten_percent (x : ℝ) : ℝ := x * 0.9

theorem percentage_and_reduction :
  (∀ a b : ℝ, is_five_percent_more a b → a = b * 1.05) ∧
  (reduced_by_ten_percent 10 = 9) := by
  sorry

end percentage_and_reduction_l666_66617


namespace library_visitors_on_sunday_l666_66696

/-- The average number of visitors on non-Sunday days -/
def avg_visitors_non_sunday : ℕ := 240

/-- The total number of days in the month -/
def total_days : ℕ := 30

/-- The number of Sundays in the month -/
def num_sundays : ℕ := 5

/-- The average number of visitors per day in the month -/
def avg_visitors_per_day : ℕ := 300

/-- The average number of visitors on Sundays -/
def avg_visitors_sunday : ℕ := 600

theorem library_visitors_on_sunday :
  num_sundays * avg_visitors_sunday + (total_days - num_sundays) * avg_visitors_non_sunday =
  total_days * avg_visitors_per_day := by
  sorry

end library_visitors_on_sunday_l666_66696


namespace smallest_yellow_marbles_l666_66669

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 6 = 0) (h2 : n ≥ 72) : ∃ (blue red green yellow : ℕ),
  blue = n / 2 ∧
  red = n / 3 ∧
  green = 12 ∧
  yellow = n - (blue + red + green) ∧
  yellow = 0 ∧
  blue + red + green + yellow = n :=
sorry

end smallest_yellow_marbles_l666_66669
