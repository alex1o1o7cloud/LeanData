import Mathlib

namespace quadratic_inequality_l2385_238584

theorem quadratic_inequality (x : ℝ) : x^2 + 9*x + 8 < 0 ↔ -8 < x ∧ x < -1 := by
  sorry

end quadratic_inequality_l2385_238584


namespace work_completion_proof_l2385_238557

/-- The number of men initially planned to complete the work -/
def initial_men : ℕ := 38

/-- The number of days it takes the initial group to complete the work -/
def initial_days : ℕ := 10

/-- The number of men sent to another project -/
def men_sent_away : ℕ := 25

/-- The number of days it takes to complete the work after sending men away -/
def new_days : ℕ := 30

/-- The total amount of work in man-days -/
def total_work : ℕ := initial_men * initial_days

theorem work_completion_proof :
  initial_men * initial_days = (initial_men - men_sent_away) * new_days :=
by sorry

end work_completion_proof_l2385_238557


namespace M_value_l2385_238556

def M : ℕ → ℕ 
  | 0 => 0
  | n + 1 => (2*n + 2)^2 + (2*n)^2 - (2*n - 2)^2 + M n

theorem M_value : M 25 = 2600 := by
  sorry

end M_value_l2385_238556


namespace specific_rectangle_triangles_l2385_238563

/-- Represents a rectangle with a grid and diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_spacing : ℕ
  horizontal_spacing : ℕ

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The main theorem stating the number of triangles in the specific configuration -/
theorem specific_rectangle_triangles :
  let rect : GridRectangle := {
    width := 40,
    height := 10,
    vertical_spacing := 10,
    horizontal_spacing := 5
  }
  count_triangles rect = 74 := by
  sorry

end specific_rectangle_triangles_l2385_238563


namespace omega_range_l2385_238574

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := cos (ω * x + φ)

theorem omega_range (ω φ α : ℝ) :
  ω > 0 →
  f ω φ α = 0 →
  deriv (f ω φ) α > 0 →
  (∀ x ∈ Set.Icc α (π + α), ¬ IsLocalMin (f ω φ) x) →
  ω ∈ Set.Ioo 1 (3/2) :=
sorry

end omega_range_l2385_238574


namespace carpet_fit_l2385_238589

theorem carpet_fit (carpet_area : ℝ) (cut_length : ℝ) (room_area : ℝ) : 
  carpet_area = 169 →
  cut_length = 2 →
  room_area = (Real.sqrt carpet_area) * (Real.sqrt carpet_area - cut_length) →
  room_area = 143 := by
sorry

end carpet_fit_l2385_238589


namespace committee_formation_ways_l2385_238531

theorem committee_formation_ways (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end committee_formation_ways_l2385_238531


namespace height_difference_l2385_238525

/-- Heights of people in centimeters -/
structure Heights where
  janet : ℝ
  charlene : ℝ
  pablo : ℝ
  ruby : ℝ

/-- Problem conditions -/
def problem_conditions (h : Heights) : Prop :=
  h.janet = 62 ∧
  h.charlene = 2 * h.janet ∧
  h.pablo = h.charlene + 70 ∧
  h.ruby = 192 ∧
  h.pablo > h.ruby

/-- Theorem stating the height difference between Pablo and Ruby -/
theorem height_difference (h : Heights) 
  (hc : problem_conditions h) : h.pablo - h.ruby = 2 := by
  sorry

end height_difference_l2385_238525


namespace set_intersection_theorem_l2385_238536

theorem set_intersection_theorem (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) → m = 3 := by
  sorry

end set_intersection_theorem_l2385_238536


namespace floor_negative_seven_fourths_l2385_238549

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l2385_238549


namespace six_people_arrangement_l2385_238542

def arrangement_count (n : ℕ) : ℕ := 
  (n.choose 2) * ((n-2).choose 2) * ((n-4).choose 2)

theorem six_people_arrangement : arrangement_count 6 = 90 := by
  sorry

end six_people_arrangement_l2385_238542


namespace sqrt_122_between_integers_product_l2385_238592

theorem sqrt_122_between_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 122 ∧ 
  Real.sqrt 122 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 132 := by
  sorry

end sqrt_122_between_integers_product_l2385_238592


namespace tangled_legs_scenario_l2385_238558

/-- The number of legs tangled in leashes when two dog walkers meet --/
def tangled_legs (dogs_group1 : ℕ) (dogs_group2 : ℕ) (legs_per_dog : ℕ) (walkers : ℕ) (legs_per_walker : ℕ) : ℕ :=
  (dogs_group1 + dogs_group2) * legs_per_dog + walkers * legs_per_walker

/-- Theorem stating the number of legs tangled in leashes in the given scenario --/
theorem tangled_legs_scenario : tangled_legs 5 3 4 2 2 = 36 := by
  sorry

end tangled_legs_scenario_l2385_238558


namespace rectangle_properties_l2385_238573

theorem rectangle_properties (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) :
  (2 * (x + y) = 50) ∧ 
  (∃ k : ℤ, (x + 5) * (y + 5) - (x - 2) * (y - 2) = 7 * k) ∧
  (x = y + 5 → ∃ a b : ℕ, a * b = (x + 5) * (y + 5) ∧ (a = x ∨ b = x) ∧ (a = y + 5 ∨ b = y + 5)) :=
by sorry

end rectangle_properties_l2385_238573


namespace line_equation_l2385_238516

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, 4)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the property of having equal intercepts on both axes
def equal_intercepts (a b c : ℝ) : Prop := ∃ k : ℝ, a * k = c ∧ b * k = c ∧ k ≠ 0

-- Define the main theorem
theorem line_equation :
  ∃ (a b c : ℝ),
    -- The line passes through point A
    (a * A.1 + b * A.2 + c = 0) ∧
    -- The line is perpendicular to 2x + y - 5 = 0
    (a * 2 + b * 1 = 0) ∧
    -- The line passes through point B
    (a * B.1 + b * B.2 + c = 0) ∧
    -- The line has equal intercepts on both axes
    (equal_intercepts a b c) ∧
    -- The equation of the line is either x + y - 5 = 0 or 4x - y = 0
    ((a = 1 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 0)) :=
sorry

end line_equation_l2385_238516


namespace tallest_tree_height_l2385_238510

theorem tallest_tree_height (h_shortest h_middle h_tallest : ℝ) : 
  h_middle = (2/3) * h_tallest →
  h_shortest = (1/2) * h_middle →
  h_shortest = 50 →
  h_tallest = 150 := by
sorry

end tallest_tree_height_l2385_238510


namespace no_valid_grid_l2385_238501

/-- Represents a grid of stars -/
def StarGrid := Fin 10 → Fin 10 → Bool

/-- Counts the number of stars in a 2x2 square starting at (i, j) -/
def countStars2x2 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat

/-- Counts the number of stars in a 3x1 rectangle starting at (i, j) -/
def countStars3x1 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid i (j+2)).toNat

/-- Checks if the grid satisfies the conditions -/
def isValidGrid (grid : StarGrid) : Prop :=
  (∀ i j, i < 9 ∧ j < 9 → countStars2x2 grid i j = 2) ∧
  (∀ i j, j < 8 → countStars3x1 grid i j = 1)

theorem no_valid_grid : ¬∃ (grid : StarGrid), isValidGrid grid := by
  sorry

end no_valid_grid_l2385_238501


namespace hyperbola_tangent_intersection_product_l2385_238545

/-- The hyperbola with equation x²/4 - y² = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - p.2^2 = 1}

/-- The asymptotes of the hyperbola -/
def asymptotes : Set (Set (ℝ × ℝ)) :=
  {{p : ℝ × ℝ | p.2 = p.1 / 2}, {p : ℝ × ℝ | p.2 = -p.1 / 2}}

/-- A line tangent to the hyperbola at point P -/
def tangent_line (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | ∃ t : ℝ, Q = (P.1 + t, P.2 + t * (P.2 / P.1))}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem hyperbola_tangent_intersection_product (P : ℝ × ℝ) 
  (h_P : P ∈ hyperbola) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_N : N ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_M_ne_N : M ≠ N) :
  dot_product M N = 3 := by
  sorry


end hyperbola_tangent_intersection_product_l2385_238545


namespace no_linear_term_in_product_l2385_238585

theorem no_linear_term_in_product (m : ℚ) : 
  (∀ x : ℚ, (x - 2) * (x^2 + m*x + 1) = x^3 + (m-2)*x^2 + 0*x + (-2)) → m = 1/2 := by
  sorry

end no_linear_term_in_product_l2385_238585


namespace function_difference_inequality_l2385_238521

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, HasDerivAt f (f' x) x ∧ HasDerivAt g (g' x) x ∧ f' x > g' x)

-- State the theorem
theorem function_difference_inequality (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) :
  f x₁ - f x₂ < g x₁ - g x₂ := by
  sorry

end function_difference_inequality_l2385_238521


namespace square_minus_product_plus_triple_l2385_238587

theorem square_minus_product_plus_triple (x y : ℝ) :
  x - y + 3 = 0 → x^2 - x*y + 3*y = 9 := by
  sorry

end square_minus_product_plus_triple_l2385_238587


namespace a_max_value_l2385_238546

def a (n : ℕ+) : ℚ := n / (n^2 + 90)

theorem a_max_value : ∀ n : ℕ+, a n ≤ 1/19 := by
  sorry

end a_max_value_l2385_238546


namespace gcd_of_mersenne_numbers_l2385_238517

theorem gcd_of_mersenne_numbers : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 2^13 - 1 := by
  sorry

end gcd_of_mersenne_numbers_l2385_238517


namespace prime_odd_sum_2009_l2385_238540

theorem prime_odd_sum_2009 :
  ∃! (a b : ℕ), Prime a ∧ Odd b ∧ a^2 + b = 2009 ∧ (a + b : ℕ) = 45 := by
  sorry

end prime_odd_sum_2009_l2385_238540


namespace stating_batsman_average_increase_l2385_238586

/-- 
Represents a batsman's scoring record.
-/
structure BatsmanRecord where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- 
Calculates the increase in average given the batsman's record before and after an inning.
-/
def averageIncrease (before after : BatsmanRecord) : ℚ :=
  after.average - before.average

/-- 
Theorem stating that given a batsman's score of 85 runs in the 17th inning
and an average of 37 runs after the 17th inning, the increase in the batsman's average is 3 runs.
-/
theorem batsman_average_increase :
  ∀ (before : BatsmanRecord),
    before.inningsPlayed = 16 →
    (BatsmanRecord.mk 17 (before.totalRuns + 85) 37).average - before.average = 3 := by
  sorry

end stating_batsman_average_increase_l2385_238586


namespace pauls_lawn_mowing_earnings_l2385_238591

/-- 
Given that:
1. Paul's total money is the sum of money from mowing lawns and $28 from weed eating
2. Paul spends $9 per week
3. Paul's money lasts for 8 weeks

Prove that Paul made $44 mowing lawns.
-/
theorem pauls_lawn_mowing_earnings :
  ∀ (M : ℕ), -- M represents the amount Paul made mowing lawns
  (M + 28 = 9 * 8) → -- Total money equals weekly spending times number of weeks
  M = 44 := by
sorry

end pauls_lawn_mowing_earnings_l2385_238591


namespace fifth_road_length_l2385_238505

/-- Represents a road network with four cities and five roads -/
structure RoadNetwork where
  road1 : ℕ
  road2 : ℕ
  road3 : ℕ
  road4 : ℕ
  road5 : ℕ

/-- The given road network satisfies the triangle inequality -/
def satisfiesTriangleInequality (rn : RoadNetwork) : Prop :=
  rn.road5 < rn.road1 + rn.road2 ∧
  rn.road5 + rn.road3 > rn.road4

/-- Theorem: Given the specific road lengths, the fifth road must be 17 km long -/
theorem fifth_road_length (rn : RoadNetwork) 
  (h1 : rn.road1 = 10)
  (h2 : rn.road2 = 8)
  (h3 : rn.road3 = 5)
  (h4 : rn.road4 = 21)
  (h5 : satisfiesTriangleInequality rn) :
  rn.road5 = 17 := by
  sorry


end fifth_road_length_l2385_238505


namespace find_number_l2385_238568

theorem find_number (G N : ℕ) (h1 : G = 129) (h2 : N % G = 9) (h3 : 2206 % G = 13) : N = 2202 := by
  sorry

end find_number_l2385_238568


namespace negation_of_proposition_l2385_238522

theorem negation_of_proposition :
  (¬(∀ x y : ℝ, x > 0 ∧ y > 0 → x + y > 0)) ↔
  (∀ x y : ℝ, ¬(x > 0 ∧ y > 0) → x + y ≤ 0) :=
by sorry

end negation_of_proposition_l2385_238522


namespace factorial_simplification_l2385_238597

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (10 : ℕ).factorial) = 2669 := by
  sorry

end factorial_simplification_l2385_238597


namespace equal_positive_reals_from_inequalities_l2385_238518

theorem equal_positive_reals_from_inequalities 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0)
  (ineq₁ : (x₁^2 - x₃*x₃)*(x₂^2 - x₃*x₃) ≤ 0)
  (ineq₂ : (x₃^2 - x₁*x₁)*(x₃^2 - x₁*x₁) ≤ 0)
  (ineq₃ : (x₃^2 - x₃*x₂)*(x₁^2 - x₃*x₂) ≤ 0)
  (ineq₄ : (x₁^2 - x₁*x₃)*(x₃^2 - x₁*x₃) ≤ 0)
  (ineq₅ : (x₃^2 - x₂*x₁)*(x₁^2 - x₂*x₁) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
  sorry


end equal_positive_reals_from_inequalities_l2385_238518


namespace pizza_area_increase_l2385_238509

theorem pizza_area_increase : 
  let small_diameter : ℝ := 12
  let large_diameter : ℝ := 18
  let small_area := Real.pi * (small_diameter / 2)^2
  let large_area := Real.pi * (large_diameter / 2)^2
  let area_increase := large_area - small_area
  let percent_increase := (area_increase / small_area) * 100
  percent_increase = 125 :=
by sorry

end pizza_area_increase_l2385_238509


namespace intersection_of_A_and_complement_of_B_l2385_238537

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {1} := by sorry

end intersection_of_A_and_complement_of_B_l2385_238537


namespace austin_work_hours_on_monday_l2385_238526

/-- Proves that Austin works 2 hours on Mondays to earn enough for a $180 bicycle in 6 weeks -/
theorem austin_work_hours_on_monday : 
  let hourly_rate : ℕ := 5
  let bicycle_cost : ℕ := 180
  let weeks : ℕ := 6
  let wednesday_hours : ℕ := 1
  let friday_hours : ℕ := 3
  ∃ (monday_hours : ℕ), 
    weeks * (hourly_rate * (monday_hours + wednesday_hours + friday_hours)) = bicycle_cost ∧ 
    monday_hours = 2 := by
  sorry

end austin_work_hours_on_monday_l2385_238526


namespace yellow_balls_count_l2385_238500

/-- Given a bag with 50 balls of two colors, if the frequency of picking one color (yellow)
    stabilizes around 0.3, then the number of yellow balls is 15. -/
theorem yellow_balls_count (total_balls : ℕ) (yellow_frequency : ℚ) 
  (h1 : total_balls = 50)
  (h2 : yellow_frequency = 3/10) : 
  ∃ (yellow_balls : ℕ), yellow_balls = 15 ∧ yellow_balls / total_balls = yellow_frequency := by
  sorry

end yellow_balls_count_l2385_238500


namespace ellipse_equation_1_l2385_238561

/-- Given an ellipse with semi-major axis a = 6 and eccentricity e = 1/3,
    prove that its standard equation is x²/36 + y²/32 = 1 -/
theorem ellipse_equation_1 (x y : ℝ) (a b c : ℝ) (h1 : a = 6) (h2 : c/a = 1/3) :
  x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1 := by sorry

end ellipse_equation_1_l2385_238561


namespace final_answer_calculation_l2385_238566

theorem final_answer_calculation (chosen_number : ℕ) (h : chosen_number = 800) : 
  (chosen_number / 5 : ℚ) - 154 = 6 := by
  sorry

end final_answer_calculation_l2385_238566


namespace lowest_possible_score_l2385_238576

/-- Represents a set of test scores -/
structure TestScores where
  scores : List ℕ
  deriving Repr

/-- Calculates the average of a list of numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Checks if a number is within a given range -/
def inRange (n : ℕ) (lower upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem lowest_possible_score 
  (first_three : TestScores)
  (h1 : first_three.scores = [82, 90, 88])
  (h2 : first_three.scores.length = 3)
  (total_tests : ℕ)
  (h3 : total_tests = 6)
  (desired_average : ℚ)
  (h4 : desired_average = 85)
  (range_lower range_upper : ℕ)
  (h5 : range_lower = 70 ∧ range_upper = 85)
  (max_score : ℕ)
  (h6 : max_score = 100) :
  ∃ (remaining : TestScores),
    remaining.scores.length = 3 ∧
    (∃ (score : ℕ), score ∈ remaining.scores ∧ inRange score range_lower range_upper) ∧
    (∃ (lowest : ℕ), lowest ∈ remaining.scores ∧ lowest = 65) ∧
    average (first_three.scores ++ remaining.scores) = desired_average ∧
    (∀ (s : ℕ), s ∈ (first_three.scores ++ remaining.scores) → s ≤ max_score) :=
by sorry

end lowest_possible_score_l2385_238576


namespace shifted_roots_l2385_238507

variable (x : ℝ)

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 5*x + 7

-- Define the roots a, b, c of the original polynomial
axiom roots_exist : ∃ a b c : ℝ, original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the shifted polynomial
def shifted_poly (x : ℝ) : ℝ := x^3 + 6*x^2 + 7*x + 5

theorem shifted_roots (a b c : ℝ) : 
  original_poly a = 0 → original_poly b = 0 → original_poly c = 0 →
  shifted_poly (a - 2) = 0 ∧ shifted_poly (b - 2) = 0 ∧ shifted_poly (c - 2) = 0 :=
sorry

end shifted_roots_l2385_238507


namespace equivalent_proposition_and_truth_l2385_238593

theorem equivalent_proposition_and_truth :
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ↔
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ∧
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
by sorry

end equivalent_proposition_and_truth_l2385_238593


namespace rectangle_width_l2385_238595

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end rectangle_width_l2385_238595


namespace log_sqrt10_1000sqrt10_l2385_238513

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end log_sqrt10_1000sqrt10_l2385_238513


namespace union_of_M_and_N_l2385_238567

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} :=
by sorry

end union_of_M_and_N_l2385_238567


namespace log_equation_solution_l2385_238578

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x - 4 * Real.log 5 = -3) ∧ (x = 0.625) :=
by sorry

end log_equation_solution_l2385_238578


namespace absolute_value_expression_l2385_238512

theorem absolute_value_expression : 
  let x : ℤ := -2023
  ‖‖|x| - (x + 3)‖ - (|x| - 3)‖ - (x - 3) = 4049 := by
  sorry

end absolute_value_expression_l2385_238512


namespace advertisement_arrangements_l2385_238571

theorem advertisement_arrangements : ℕ := by
  -- Define the total number of advertisements
  let total_ads : ℕ := 6
  -- Define the number of commercial advertisements
  let commercial_ads : ℕ := 4
  -- Define the number of public service advertisements
  let public_service_ads : ℕ := 2
  -- Define the condition that public service ads must be at the beginning and end
  let public_service_at_ends : Prop := true

  -- The theorem to prove
  have : (public_service_at_ends ∧ 
          total_ads = commercial_ads + public_service_ads) → 
         (Nat.factorial public_service_ads * Nat.factorial commercial_ads = 48) := by
    sorry

  -- The final statement
  exact 48

end advertisement_arrangements_l2385_238571


namespace garden_area_ratio_l2385_238508

/-- Given a rectangular garden with initial length and width, and increase percentages for both dimensions, prove that the ratio of the original area to the redesigned area is 1/3. -/
theorem garden_area_ratio (initial_length initial_width : ℝ) 
  (length_increase width_increase : ℝ) : 
  initial_length = 10 →
  initial_width = 5 →
  length_increase = 0.5 →
  width_increase = 1 →
  (initial_length * initial_width) / 
  ((initial_length * (1 + length_increase)) * (initial_width * (1 + width_increase))) = 1/3 :=
by sorry

end garden_area_ratio_l2385_238508


namespace adjusted_ratio_equals_three_halves_l2385_238598

theorem adjusted_ratio_equals_three_halves :
  (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end adjusted_ratio_equals_three_halves_l2385_238598


namespace domain_of_f_l2385_238599

noncomputable def f (x : ℝ) := Real.log (x - 1) + Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end domain_of_f_l2385_238599


namespace average_sales_is_84_l2385_238575

/-- Sales data for each month -/
def sales : List Int := [120, 80, -20, 100, 140]

/-- Number of months -/
def num_months : Nat := 5

/-- Theorem: The average sales per month is 84 dollars -/
theorem average_sales_is_84 : (sales.sum / num_months : Int) = 84 := by
  sorry

end average_sales_is_84_l2385_238575


namespace min_triangular_faces_l2385_238515

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  non_triangular_faces : ℕ
  euler : faces + vertices = edges + 2
  more_faces : faces > vertices
  face_sum : faces = triangular_faces + non_triangular_faces
  edge_inequality : edges ≥ (3 * triangular_faces + 4 * non_triangular_faces) / 2

/-- The minimum number of triangular faces in a convex polyhedron with more faces than vertices is 6 --/
theorem min_triangular_faces (p : ConvexPolyhedron) : p.triangular_faces ≥ 6 := by
  sorry

end min_triangular_faces_l2385_238515


namespace sum_of_ac_l2385_238550

theorem sum_of_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end sum_of_ac_l2385_238550


namespace greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l2385_238539

theorem greatest_two_digit_multiple_of_13 : ℕ → Prop :=
  fun n =>
    (n ≤ 99) ∧ 
    (n ≥ 10) ∧ 
    (∃ k : ℕ, n = 13 * k) ∧ 
    (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ (∃ j : ℕ, m = 13 * j) → m ≤ n) →
    n = 91

-- The proof would go here, but we'll skip it as requested
theorem greatest_two_digit_multiple_of_13_is_91 : greatest_two_digit_multiple_of_13 91 := by
  sorry

end greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l2385_238539


namespace socks_theorem_l2385_238534

def socks_problem (initial_pairs : ℕ) : Prop :=
  let week1 := 12
  let week2 := week1 + 4
  let week3 := (week1 + week2) / 2
  let week4 := week3 - 3
  let total := 57
  initial_pairs = total - (week1 + week2 + week3 + week4)

theorem socks_theorem : ∃ (x : ℕ), socks_problem x :=
sorry

end socks_theorem_l2385_238534


namespace lychee_theorem_l2385_238553

def lychee_yield (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 =>
    if i < 9 then 2 * lychee_yield n i + 1
    else if i < 15 then lychee_yield n 9
    else (lychee_yield n i) / 2

def total_yield (n : ℕ) : ℕ :=
  (List.range n).map (lychee_yield n) |>.sum

theorem lychee_theorem : total_yield 25 = 8173 := by
  sorry

end lychee_theorem_l2385_238553


namespace divisible_by_sixteen_l2385_238572

theorem divisible_by_sixteen (n : ℕ) : ∃ k : ℤ, (2*n - 1)^3 - (2*n)^2 + 2*n + 1 = 16 * k := by
  sorry

end divisible_by_sixteen_l2385_238572


namespace inequality_implication_l2385_238562

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end inequality_implication_l2385_238562


namespace geometric_sequence_terms_l2385_238564

theorem geometric_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 + a n = 82 →
  a 3 * a (n - 2) = 81 →
  S_n = 121 →
  (∀ k, S_k = (a 1 * (1 - (a 2 / a 1)^k)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  n = 5 := by
sorry


end geometric_sequence_terms_l2385_238564


namespace sum_base4_numbers_l2385_238532

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem sum_base4_numbers : 
  let a := [2, 0, 2]  -- 202₄
  let b := [0, 3, 3]  -- 330₄
  let c := [0, 0, 0, 1]  -- 1000₄
  let sum_base10 := base4ToBase10 a + base4ToBase10 b + base4ToBase10 c
  base10ToBase4 sum_base10 = [2, 3, 1, 2] ∧ sum_base10 = 158 := by
  sorry

end sum_base4_numbers_l2385_238532


namespace special_triangle_sides_l2385_238514

/-- Represents a triangle with known height, base, and sum of two sides --/
structure SpecialTriangle where
  height : ℝ
  base : ℝ
  sum_of_sides : ℝ

/-- The two unknown sides of the triangle --/
structure TriangleSides where
  side1 : ℝ
  side2 : ℝ

/-- Theorem stating that for a triangle with height 24, base 28, and sum of two sides 56,
    the lengths of these two sides are 26 and 30 --/
theorem special_triangle_sides (t : SpecialTriangle) 
    (h1 : t.height = 24)
    (h2 : t.base = 28)
    (h3 : t.sum_of_sides = 56) :
  ∃ (s : TriangleSides), s.side1 = 26 ∧ s.side2 = 30 ∧ s.side1 + s.side2 = t.sum_of_sides :=
by
  sorry


end special_triangle_sides_l2385_238514


namespace min_value_sum_product_equality_condition_l2385_238528

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 4 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) = 4 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end min_value_sum_product_equality_condition_l2385_238528


namespace min_b_minus_a_l2385_238554

noncomputable def f (a b x : ℝ) : ℝ := (2 * x^2 + x) * Real.log x - (2 * a + 1) * x^2 - (a + 1) * x + b

theorem min_b_minus_a (a b : ℝ) :
  (∀ x > 0, f a b x ≥ 0) → 
  ∃ m, m = 3/4 + Real.log 2 ∧ b - a ≥ m ∧ ∀ ε > 0, ∃ a' b', b' - a' < m + ε :=
sorry

end min_b_minus_a_l2385_238554


namespace circuit_reliability_l2385_238559

-- Define the probabilities of element failures
def p1 : ℝ := 0.2
def p2 : ℝ := 0.3
def p3 : ℝ := 0.4

-- Define the probability of the circuit not breaking
def circuit_not_break : ℝ := (1 - p1) * (1 - p2) * (1 - p3)

-- Theorem statement
theorem circuit_reliability : circuit_not_break = 0.336 := by
  sorry

end circuit_reliability_l2385_238559


namespace original_prices_l2385_238583

-- Define the sale prices and discount rates
def book_sale_price : ℚ := 8
def book_discount_rate : ℚ := 1 / 8
def pen_sale_price : ℚ := 4
def pen_discount_rate : ℚ := 1 / 5

-- Theorem statement
theorem original_prices :
  (book_sale_price / book_discount_rate = 64) ∧
  (pen_sale_price / pen_discount_rate = 20) :=
by sorry

end original_prices_l2385_238583


namespace marble_bag_problem_l2385_238590

theorem marble_bag_problem (total_marbles : ℕ) (red_marbles : ℕ) 
  (probability_non_red : ℚ) : 
  red_marbles = 12 → 
  probability_non_red = 36 / 49 → 
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = probability_non_red) → 
  total_marbles = 84 := by
  sorry

end marble_bag_problem_l2385_238590


namespace largest_prime_divisor_of_13_squared_plus_84_squared_l2385_238541

theorem largest_prime_divisor_of_13_squared_plus_84_squared : 
  (Nat.factors (13^2 + 84^2)).maximum = some 17 := by
  sorry

end largest_prime_divisor_of_13_squared_plus_84_squared_l2385_238541


namespace complex_equation_solution_l2385_238570

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = -2 * Complex.I →
  z = -1 - Complex.I := by
sorry

end complex_equation_solution_l2385_238570


namespace race_time_difference_l2385_238535

/-- Represents the time difference in minutes between two runners finishing a race -/
def timeDifference (malcolmSpeed Joshua : ℝ) (raceDistance : ℝ) : ℝ :=
  raceDistance * Joshua - raceDistance * malcolmSpeed

theorem race_time_difference :
  let malcolmSpeed := 6
  let Joshua := 8
  let raceDistance := 10
  timeDifference malcolmSpeed Joshua raceDistance = 20 := by
  sorry

end race_time_difference_l2385_238535


namespace first_number_is_thirty_l2385_238579

theorem first_number_is_thirty (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : 2 * (x - y) = 20) : 
  x = 30 := by
sorry

end first_number_is_thirty_l2385_238579


namespace sum_of_three_square_roots_inequality_l2385_238577

theorem sum_of_three_square_roots_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 21 := by
  sorry

end sum_of_three_square_roots_inequality_l2385_238577


namespace alternating_draw_probability_l2385_238529

/-- The number of white balls in the box -/
def num_white : ℕ := 6

/-- The number of black balls in the box -/
def num_black : ℕ := 6

/-- The total number of balls in the box -/
def total_balls : ℕ := num_white + num_black

/-- The number of ways to arrange all balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_white

/-- The number of alternating color sequences -/
def alternating_sequences : ℕ := 2

/-- The probability of drawing balls with alternating colors -/
def alternating_probability : ℚ := alternating_sequences / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 462 := by
  sorry

end alternating_draw_probability_l2385_238529


namespace gcd_product_is_square_l2385_238547

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
  sorry

end gcd_product_is_square_l2385_238547


namespace brown_eyed_brunettes_count_l2385_238581

/-- Represents the characteristics of girls in a school -/
structure SchoolGirls where
  total : ℕ
  blueEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes -/
def brownEyedBrunettes (s : SchoolGirls) : ℕ :=
  s.brownEyed - (s.total - s.brunettes - s.blueEyedBlondes)

/-- Theorem stating the number of brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (s : SchoolGirls) 
  (h1 : s.total = 60)
  (h2 : s.blueEyedBlondes = 20)
  (h3 : s.brunettes = 35)
  (h4 : s.brownEyed = 25) :
  brownEyedBrunettes s = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, blueEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end brown_eyed_brunettes_count_l2385_238581


namespace set_A_membership_l2385_238596

def A : Set ℝ := {x | 2 * x - 3 < 0}

theorem set_A_membership : 1 ∈ A ∧ 2 ∉ A := by
  sorry

end set_A_membership_l2385_238596


namespace hexagon_area_sum_l2385_238502

-- Define the hexagon structure
structure Hexagon :=
  (sideLength : ℝ)
  (numSegments : ℕ)

-- Define the theorem
theorem hexagon_area_sum (h : Hexagon) (a b : ℕ) : 
  h.sideLength = 3 ∧ h.numSegments = 12 →
  ∃ (area : ℝ), area = a * Real.sqrt b ∧ a + b = 30 :=
by sorry

end hexagon_area_sum_l2385_238502


namespace eraser_cost_l2385_238569

theorem eraser_cost (total_cartons : ℕ) (total_cost : ℕ) (pencil_cost : ℕ) (pencil_cartons : ℕ) :
  total_cartons = 100 →
  total_cost = 360 →
  pencil_cost = 6 →
  pencil_cartons = 20 →
  (total_cost - pencil_cost * pencil_cartons) / (total_cartons - pencil_cartons) = 3 := by
sorry

end eraser_cost_l2385_238569


namespace no_solution_for_inequalities_l2385_238548

theorem no_solution_for_inequalities :
  ¬∃ x : ℝ, (4 * x^2 + 7 * x - 2 < 0) ∧ (3 * x - 1 > 0) := by
  sorry

end no_solution_for_inequalities_l2385_238548


namespace five_hundredth_term_is_negative_one_l2385_238520

def sequence_term (n : ℕ) : ℚ :=
  match n % 3 with
  | 1 => 2
  | 2 => -1
  | 0 => 1/2
  | _ => 0 -- This case should never occur

theorem five_hundredth_term_is_negative_one :
  sequence_term 500 = -1 := by
  sorry

end five_hundredth_term_is_negative_one_l2385_238520


namespace stock_value_change_l2385_238582

theorem stock_value_change (initial_value : ℝ) (day1_decrease : ℝ) (day2_increase : ℝ) :
  day1_decrease = 0.2 →
  day2_increase = 0.3 →
  (1 - day1_decrease) * (1 + day2_increase) = 1.04 := by
  sorry

end stock_value_change_l2385_238582


namespace geometric_arithmetic_mean_ratio_sum_l2385_238503

/-- Given a geometric sequence a, b, c and their arithmetic means m and n,
    prove that a/m + c/n = 2 -/
theorem geometric_arithmetic_mean_ratio_sum (a b c m n : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end geometric_arithmetic_mean_ratio_sum_l2385_238503


namespace marble_problem_l2385_238524

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 4 * brian) 
  (h4 : daryl = 6 * caden) 
  (h5 : angela + brian + caden + daryl = 186) : 
  a = 93 / 44 := by
sorry

end marble_problem_l2385_238524


namespace instantaneous_velocity_at_3_l2385_238519

-- Define the position function
def s (t : ℝ) : ℝ := 5 * t^2

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 10 * t

-- Theorem stating that the instantaneous velocity at t=3 is 30
theorem instantaneous_velocity_at_3 : v 3 = 30 := by
  sorry

end instantaneous_velocity_at_3_l2385_238519


namespace consecutive_terms_iff_equation_l2385_238538

/-- Sequence a_k defined by a_0 = 0, a_1 = n, and a_{k+1} = n^2 * a_k - a_{k-1} -/
def sequence_a (n : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => n
  | (k + 2) => n^2 * sequence_a n (k + 1) - sequence_a n k

/-- Predicate to check if two integers are consecutive terms in the sequence -/
def are_consecutive_terms (n a b : ℕ) : Prop :=
  ∃ k : ℕ, sequence_a n k = a ∧ sequence_a n (k + 1) = b

theorem consecutive_terms_iff_equation (n : ℕ) (hn : n > 0) (a b : ℕ) (hab : a ≤ b) :
  are_consecutive_terms n a b ↔ a^2 + b^2 = n^2 * (a * b + 1) :=
sorry

end consecutive_terms_iff_equation_l2385_238538


namespace m_eq_neg_one_iff_pure_imaginary_l2385_238523

/-- A complex number z is defined as m² - 1 + (m² - 3m + 2)i, where m is a real number and i is the imaginary unit. -/
def z (m : ℝ) : ℂ := (m^2 - 1) + (m^2 - 3*m + 2)*Complex.I

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Theorem: m = -1 is both sufficient and necessary for z to be a pure imaginary number. -/
theorem m_eq_neg_one_iff_pure_imaginary (m : ℝ) :
  m = -1 ↔ is_pure_imaginary (z m) := by sorry

end m_eq_neg_one_iff_pure_imaginary_l2385_238523


namespace triangle_theorem_l2385_238533

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A * Real.sin t.B + t.b * (Real.cos t.A)^2 = 4/3 * t.a

/-- The additional condition for part 2 -/
def additional_condition (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + 1/4 * t.b^2

theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : additional_condition t) : 
  t.b / t.a = 4/3 ∧ t.C = π/3 := by
  sorry

end

end triangle_theorem_l2385_238533


namespace exam_failure_percentage_l2385_238552

theorem exam_failure_percentage 
  (pass_english : ℝ) 
  (pass_math : ℝ) 
  (pass_either : ℝ) 
  (h1 : pass_english = 0.63) 
  (h2 : pass_math = 0.65) 
  (h3 : pass_either = 0.55) : 
  1 - pass_either = 0.45 := by
  sorry

end exam_failure_percentage_l2385_238552


namespace hostel_problem_l2385_238530

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 32

/-- The number of days the provisions would last if 50 men left -/
def reduced_days : ℕ := 40

/-- The number of men that left the hostel -/
def men_left : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_problem :
  initial_men = 250 ∧
  (initial_days : ℚ) * initial_men = reduced_days * (initial_men - men_left) :=
sorry

end hostel_problem_l2385_238530


namespace trigonometric_identity_l2385_238580

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.sin α ^ 2 + Real.sin α * Real.sin β + Real.sin β ^ 2 = 3 / 4 := by
  sorry

end trigonometric_identity_l2385_238580


namespace integral_even_function_l2385_238544

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_integral : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end integral_even_function_l2385_238544


namespace arithmetic_geometric_ratio_l2385_238504

/-- An arithmetic sequence with common difference d and first term a₁ -/
def arithmetic_sequence (d a₁ : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

/-- Theorem: For an arithmetic sequence {aₙ} where the common difference d ≠ 0 and
    the first term a₁ ≠ 0, if a₂, a₄, a₈ form a geometric sequence,
    then (a₁ + a₅ + a₉) / (a₂ + a₃) = 3 -/
theorem arithmetic_geometric_ratio
  (d a₁ : ℝ)
  (hd : d ≠ 0)
  (ha₁ : a₁ ≠ 0)
  (h_geom : (arithmetic_sequence d a₁ 4) ^ 2 = 
            (arithmetic_sequence d a₁ 2) * (arithmetic_sequence d a₁ 8)) :
  (arithmetic_sequence d a₁ 1 + arithmetic_sequence d a₁ 5 + arithmetic_sequence d a₁ 9) /
  (arithmetic_sequence d a₁ 2 + arithmetic_sequence d a₁ 3) = 3 :=
by sorry

end arithmetic_geometric_ratio_l2385_238504


namespace distance_traveled_l2385_238527

/-- Proves that given a speed of 20 km/hr and a time of 2.5 hours, the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 2.5) :
  speed * time = 50 := by
  sorry

end distance_traveled_l2385_238527


namespace basil_plant_selling_price_l2385_238565

/-- Proves that the selling price per basil plant is $5.00 given the costs and net profit --/
theorem basil_plant_selling_price 
  (seed_cost : ℝ) 
  (soil_cost : ℝ) 
  (num_plants : ℕ) 
  (net_profit : ℝ) 
  (h1 : seed_cost = 2)
  (h2 : soil_cost = 8)
  (h3 : num_plants = 20)
  (h4 : net_profit = 90) :
  (net_profit + seed_cost + soil_cost) / num_plants = 5 := by
  sorry

#check basil_plant_selling_price

end basil_plant_selling_price_l2385_238565


namespace eighteenth_replacement_in_december_l2385_238560

def months_in_year : ℕ := 12
def replacement_interval : ℕ := 7
def target_replacement : ℕ := 18

def month_of_replacement (n : ℕ) : ℕ :=
  ((n - 1) * replacement_interval) % months_in_year + 1

theorem eighteenth_replacement_in_december :
  month_of_replacement target_replacement = 12 := by
  sorry

end eighteenth_replacement_in_december_l2385_238560


namespace macaroons_remaining_l2385_238594

/-- The number of red macaroons initially baked -/
def initial_red : ℕ := 50

/-- The number of green macaroons initially baked -/
def initial_green : ℕ := 40

/-- The number of green macaroons eaten -/
def green_eaten : ℕ := 15

/-- The number of red macaroons eaten is twice the number of green macaroons eaten -/
def red_eaten : ℕ := 2 * green_eaten

/-- The total number of remaining macaroons -/
def remaining_macaroons : ℕ := (initial_red - red_eaten) + (initial_green - green_eaten)

theorem macaroons_remaining :
  remaining_macaroons = 45 := by
  sorry

end macaroons_remaining_l2385_238594


namespace john_plays_three_times_a_month_l2385_238543

/-- The number of times John plays paintball in a month -/
def plays_per_month : ℕ := sorry

/-- The number of boxes John buys each time he plays -/
def boxes_per_play : ℕ := 3

/-- The cost of each box of paintballs in dollars -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars -/
def total_spent_per_month : ℕ := 225

/-- Theorem stating that John plays paintball 3 times a month -/
theorem john_plays_three_times_a_month : 
  plays_per_month = 3 ∧ 
  plays_per_month * boxes_per_play * cost_per_box = total_spent_per_month := by
  sorry

end john_plays_three_times_a_month_l2385_238543


namespace solution_set_implies_k_value_l2385_238555

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end solution_set_implies_k_value_l2385_238555


namespace smallest_x_value_solution_exists_l2385_238588

theorem smallest_x_value (x : ℝ) : 
  (x^2 - x - 72) / (x - 9) = 3 / (x + 6) → x ≥ -9 :=
by
  sorry

theorem solution_exists : 
  ∃ x : ℝ, (x^2 - x - 72) / (x - 9) = 3 / (x + 6) ∧ x = -9 :=
by
  sorry

end smallest_x_value_solution_exists_l2385_238588


namespace basketball_games_l2385_238511

theorem basketball_games (x : ℕ) : 
  (3 * x / 4 : ℚ) = x * 3 / 4 ∧ 
  (2 * (x + 10) / 3 : ℚ) = x * 3 / 4 + 5 → 
  x = 20 := by
  sorry

end basketball_games_l2385_238511


namespace polynomial_division_remainder_l2385_238551

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℤ),
  X^4 + X^2 = (X^2 + 3*X + 2) * q + r ∧ 
  r.degree < (X^2 + 3*X + 2).degree ∧ 
  r = -18*X - 16 := by sorry

end polynomial_division_remainder_l2385_238551


namespace total_marbles_is_27_l2385_238506

/-- The total number of green and red marbles owned by Sara, Tom, and Lisa -/
def total_green_red_marbles (sara_green sara_red : ℕ) (tom_green tom_red : ℕ) (lisa_green lisa_red : ℕ) : ℕ :=
  sara_green + sara_red + tom_green + tom_red + lisa_green + lisa_red

/-- Theorem stating that the total number of green and red marbles is 27 -/
theorem total_marbles_is_27 :
  total_green_red_marbles 3 5 4 7 5 3 = 27 := by
  sorry

end total_marbles_is_27_l2385_238506
