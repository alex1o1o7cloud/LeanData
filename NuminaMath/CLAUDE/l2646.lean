import Mathlib

namespace exam_mean_is_115_l2646_264650

/-- The mean score of the exam -/
def mean : ℝ := 115

/-- The standard deviation of the exam scores -/
def std_dev : ℝ := 40

/-- Theorem stating that the given conditions imply the mean score is 115 -/
theorem exam_mean_is_115 :
  (55 = mean - 1.5 * std_dev) ∧
  (75 = mean - 2 * std_dev) ∧
  (85 = mean + 1.5 * std_dev) ∧
  (100 = mean + 3.5 * std_dev) →
  mean = 115 := by sorry

end exam_mean_is_115_l2646_264650


namespace nearest_whole_number_solution_l2646_264640

theorem nearest_whole_number_solution (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 :=
sorry

end nearest_whole_number_solution_l2646_264640


namespace soda_price_calculation_l2646_264656

def pizza_price : ℚ := 12
def fries_price : ℚ := (3 / 10)
def goal_amount : ℚ := 500
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def remaining_amount : ℚ := 258

theorem soda_price_calculation :
  ∃ (soda_price : ℚ),
    soda_price * sodas_sold = 
      goal_amount - remaining_amount - 
      (pizza_price * pizzas_sold + fries_price * fries_sold) ∧
    soda_price = 2 :=
by sorry

end soda_price_calculation_l2646_264656


namespace concert_revenue_l2646_264660

theorem concert_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_tickets : total_tickets = 200)
  (h_revenue : total_revenue = 3000) : ℕ :=
by
  -- Let f be the number of full-price tickets
  -- Let d be the number of discount tickets
  -- Let p be the price of a full-price ticket
  have h1 : ∃ (f d p : ℕ), 
    f + d = total_tickets ∧ 
    f * p + d * (p / 3) = total_revenue ∧ 
    f * p = 1500
  sorry

  exact 1500


end concert_revenue_l2646_264660


namespace mike_action_figures_l2646_264637

/-- The number of action figures each shelf can hold -/
def figures_per_shelf : ℕ := 8

/-- The number of shelves Mike needs -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Mike has -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem mike_action_figures :
  total_figures = 64 :=
by sorry

end mike_action_figures_l2646_264637


namespace sandy_shopping_total_l2646_264670

/-- The total amount Sandy spent on clothes after discounts, coupon, and tax -/
def total_spent (shorts shirt jacket shoes accessories discount coupon tax : ℚ) : ℚ :=
  let initial_total := shorts + shirt + jacket + shoes + accessories
  let discounted_total := initial_total * (1 - discount)
  let after_coupon := discounted_total - coupon
  let final_total := after_coupon * (1 + tax)
  final_total

/-- Theorem stating the total amount Sandy spent on clothes -/
theorem sandy_shopping_total :
  total_spent 13.99 12.14 7.43 8.50 10.75 0.10 5.00 0.075 = 45.72 := by
  sorry

end sandy_shopping_total_l2646_264670


namespace complex_equality_implies_values_l2646_264639

theorem complex_equality_implies_values (x y : ℝ) : 
  (Complex.mk (x - 1) y = Complex.mk 0 1 - Complex.mk (3*x) 0) → 
  (x = 1/4 ∧ y = 1) := by
  sorry

end complex_equality_implies_values_l2646_264639


namespace line_with_y_intercept_two_l2646_264633

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line in slope-intercept form. -/
def line_equation (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Theorem: The equation of a line with y-intercept 2 is y = kx + 2 -/
theorem line_with_y_intercept_two (k : ℝ) :
  ∃ (l : Line), l.y_intercept = 2 ∧ ∀ (x y : ℝ), y = line_equation l x ↔ y = k * x + 2 := by
  sorry

end line_with_y_intercept_two_l2646_264633


namespace baby_sea_turtles_on_sand_l2646_264672

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) : total = 42 → swept_fraction = 1/3 → total - (swept_fraction * total).floor = 28 := by
  sorry

end baby_sea_turtles_on_sand_l2646_264672


namespace geometric_sequence_problem_l2646_264644

theorem geometric_sequence_problem (q : ℝ) (S₆ : ℝ) (b₁ : ℝ) (b₅ : ℝ) : 
  q = 3 → 
  S₆ = 1820 → 
  S₆ = b₁ * (1 - q^6) / (1 - q) → 
  b₅ = b₁ * q^4 →
  b₁ = 5 ∧ b₅ = 405 := by
  sorry

#check geometric_sequence_problem

end geometric_sequence_problem_l2646_264644


namespace drums_per_day_l2646_264677

/-- Given that 90 drums are filled in 6 days, prove that 15 drums are filled per day -/
theorem drums_per_day :
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 90 →
    total_days = 6 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 15 := by
  sorry

end drums_per_day_l2646_264677


namespace dans_remaining_money_l2646_264675

/-- Dan's initial money in dollars -/
def initial_money : ℝ := 5

/-- Cost of the candy bar in dollars -/
def candy_bar_cost : ℝ := 2

/-- Theorem: Dan's remaining money after buying the candy bar is $3 -/
theorem dans_remaining_money : 
  initial_money - candy_bar_cost = 3 := by
  sorry

end dans_remaining_money_l2646_264675


namespace current_rabbits_in_cage_l2646_264663

/-- The number of rabbits Jasper saw in the park -/
def rabbits_in_park : ℕ := 60

/-- The number of rabbits currently in the cage -/
def rabbits_in_cage : ℕ := 13

/-- The number of rabbits to be added to the cage -/
def rabbits_to_add : ℕ := 7

theorem current_rabbits_in_cage :
  rabbits_in_cage + rabbits_to_add = rabbits_in_park / 3 ∧
  rabbits_in_cage = 13 :=
by sorry

end current_rabbits_in_cage_l2646_264663


namespace region_D_properties_l2646_264614

def region_D (x y : ℝ) : Prop :=
  2 ≤ x ∧ x ≤ 6 ∧
  1 ≤ y ∧ y ≤ 3 ∧
  x^2 / 9 + y^2 / 4 < 1 ∧
  4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧
  0 < y ∧ y < x

theorem region_D_properties (x y : ℝ) :
  region_D x y →
  (2 ≤ x ∧ x ≤ 6) ∧
  (1 ≤ y ∧ y ≤ 3) ∧
  (x^2 / 9 + y^2 / 4 < 1) ∧
  (4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9) ∧
  (0 < y ∧ y < x) :=
by sorry

end region_D_properties_l2646_264614


namespace child_height_calculation_l2646_264641

/-- Given a child's current height and growth since last visit, 
    calculate the child's height at the last visit. -/
def height_at_last_visit (current_height growth : ℝ) : ℝ :=
  current_height - growth

/-- Theorem stating that given the specific measurements, 
    the child's height at the last visit was 38.5 inches. -/
theorem child_height_calculation : 
  height_at_last_visit 41.5 3 = 38.5 := by
  sorry

end child_height_calculation_l2646_264641


namespace not_sufficient_nor_necessary_l2646_264629

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ a b : ℝ, a + b > 0 ∧ a * b ≤ 0) ∧ 
  (∃ a b : ℝ, a * b > 0 ∧ a + b ≤ 0) := by
  sorry

end not_sufficient_nor_necessary_l2646_264629


namespace remainder_mod_six_l2646_264615

theorem remainder_mod_six (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end remainder_mod_six_l2646_264615


namespace zero_unique_for_multiplication_and_division_l2646_264619

theorem zero_unique_for_multiplication_and_division :
  ∀ x : ℝ, (∀ a : ℝ, x * a = x ∧ (a ≠ 0 → x / a = x)) → x = 0 :=
by sorry

end zero_unique_for_multiplication_and_division_l2646_264619


namespace flour_sugar_difference_l2646_264659

theorem flour_sugar_difference (total_flour sugar_needed flour_added : ℕ) : 
  total_flour = 14 →
  sugar_needed = 9 →
  flour_added = 4 →
  (total_flour - flour_added) - sugar_needed = 1 := by
  sorry

end flour_sugar_difference_l2646_264659


namespace p_implies_m_range_p_and_q_implies_m_range_l2646_264671

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a = m - 1 ∧ b = 3 - m ∧ 
  ∀ (x y : ℝ), x^2 / a + y^2 / b = 1 → ∃ (c : ℝ), x^2 + (y^2 - c^2) = a * b

def q (m : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = x^2 - m*x + 9/16 ∧ y > 0

-- Theorem 1
theorem p_implies_m_range (m : ℝ) : p m → 1 < m ∧ m < 2 := by sorry

-- Theorem 2
theorem p_and_q_implies_m_range (m : ℝ) : p m ∧ q m → 1 < m ∧ m < 3/2 := by sorry

end p_implies_m_range_p_and_q_implies_m_range_l2646_264671


namespace symmetric_points_fourth_quadrant_l2646_264686

/-- Given two points A(a, 3) and B(2, b) symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Conditions derived from symmetry
  (a > 0 ∧ b < 0)     -- Definition of fourth quadrant
  := by sorry

end symmetric_points_fourth_quadrant_l2646_264686


namespace product_of_Q_at_roots_of_P_l2646_264634

/-- The polynomial P(x) = x^5 - x^2 + 1 -/
def P (x : ℂ) : ℂ := x^5 - x^2 + 1

/-- The polynomial Q(x) = x^2 + 1 -/
def Q (x : ℂ) : ℂ := x^2 + 1

theorem product_of_Q_at_roots_of_P :
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℂ),
    (P r₁ = 0) ∧ (P r₂ = 0) ∧ (P r₃ = 0) ∧ (P r₄ = 0) ∧ (P r₅ = 0) ∧
    (Q r₁ * Q r₂ * Q r₃ * Q r₄ * Q r₅ = 5) := by
  sorry

end product_of_Q_at_roots_of_P_l2646_264634


namespace ryan_learning_days_l2646_264625

/-- Given that Ryan spends 4 hours on Chinese daily and a total of 24 hours on Chinese,
    prove that the number of days he learns is 6. -/
theorem ryan_learning_days :
  ∀ (hours_chinese_per_day : ℕ) (total_hours_chinese : ℕ),
    hours_chinese_per_day = 4 →
    total_hours_chinese = 24 →
    total_hours_chinese / hours_chinese_per_day = 6 :=
by sorry

end ryan_learning_days_l2646_264625


namespace hyperbola_midpoint_locus_l2646_264661

/-- Given a hyperbola x^2 - y^2/4 = 1, if two perpendicular lines through its center O
    intersect the hyperbola at points A and B, then the locus of the midpoint P of chord AB
    satisfies the equation 3(4x^2 - y^2)^2 = 4(16x^2 + y^2). -/
theorem hyperbola_midpoint_locus (x y : ℝ) :
  (∃ (m n : ℝ),
    -- A and B are on the hyperbola
    4 * (x - m)^2 - (y - n)^2 = 4 ∧
    4 * (x + m)^2 - (y + n)^2 = 4 ∧
    -- OA ⊥ OB
    x^2 + y^2 = m^2 + n^2 ∧
    -- (x, y) is the midpoint of AB
    (x - m, y - n) = (-x - m, -y - n)) →
  3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) := by
sorry


end hyperbola_midpoint_locus_l2646_264661


namespace arctan_sum_theorem_l2646_264648

theorem arctan_sum_theorem (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end arctan_sum_theorem_l2646_264648


namespace sum_of_zeros_greater_than_one_l2646_264606

noncomputable def g (x m : ℝ) : ℝ := Real.log x - x + x + 1 / (2 * x) - m

theorem sum_of_zeros_greater_than_one (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) 
  (hz₁ : g x₁ m = 0) (hz₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := by
sorry

end sum_of_zeros_greater_than_one_l2646_264606


namespace lineup_count_l2646_264695

/-- The number of team members -/
def team_size : ℕ := 16

/-- The number of positions in the lineup -/
def lineup_size : ℕ := 5

/-- The number of pre-assigned positions -/
def pre_assigned : ℕ := 2

/-- Calculate the number of ways to choose a lineup -/
def lineup_ways : ℕ :=
  (team_size - pre_assigned) * (team_size - pre_assigned - 1) * (team_size - pre_assigned - 2)

theorem lineup_count : lineup_ways = 2184 := by sorry

end lineup_count_l2646_264695


namespace elevator_problem_solution_l2646_264653

/-- Represents the elevator problem with given conditions -/
structure ElevatorProblem where
  total_floors : ℕ
  first_half_time : ℕ
  mid_section_rate : ℕ
  final_section_rate : ℕ

/-- Calculates the total time in hours for the elevator to reach the bottom -/
def total_time (p : ElevatorProblem) : ℚ :=
  let first_half := p.first_half_time
  let mid_section := (p.total_floors / 4) * p.mid_section_rate
  let final_section := (p.total_floors / 4) * p.final_section_rate
  (first_half + mid_section + final_section) / 60

/-- Theorem stating that for the given problem, the total time is 2 hours -/
theorem elevator_problem_solution :
  let problem := ElevatorProblem.mk 20 15 5 16
  total_time problem = 2 := by sorry

end elevator_problem_solution_l2646_264653


namespace hyperbola_equation_l2646_264662

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = -2*x ∧ y^2/a^2 - x^2/b^2 = 1) →
  (∃ (x y : ℝ), x^2 = 4*Real.sqrt 10*y ∧ x^2 + y^2 = a^2 - b^2) →
  a^2 = 8 ∧ b^2 = 2 := by
sorry

end hyperbola_equation_l2646_264662


namespace line_circle_intersection_a_values_l2646_264607

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter of the line equation 4x + 3y + a = 0 -/
  a : ℝ
  /-- The line 4x + 3y + a = 0 intersects the circle (x-1)^2 + (y-2)^2 = 9 -/
  intersects : ∃ (x y : ℝ), 4*x + 3*y + a = 0 ∧ (x-1)^2 + (y-2)^2 = 9
  /-- The distance between intersection points is 4√2 -/
  chord_length : ∃ (A B : ℝ × ℝ), 
    (4*(A.1) + 3*(A.2) + a = 0) ∧ ((A.1-1)^2 + (A.2-2)^2 = 9) ∧
    (4*(B.1) + 3*(B.2) + a = 0) ∧ ((B.1-1)^2 + (B.2-2)^2 = 9) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32

/-- The theorem stating the possible values of a -/
theorem line_circle_intersection_a_values (lci : LineCircleIntersection) :
  lci.a = -5 ∨ lci.a = -15 := by sorry

end line_circle_intersection_a_values_l2646_264607


namespace probability_of_pair_l2646_264680

def deck_size : ℕ := 38
def num_sets : ℕ := 10
def cards_in_reduced_set : ℕ := 3
def cards_in_full_set : ℕ := 4

def total_combinations : ℕ := (deck_size * (deck_size - 1)) / 2

def favorable_outcomes : ℕ := 
  (cards_in_reduced_set * (cards_in_reduced_set - 1)) / 2 + 
  (num_sets - 1) * (cards_in_full_set * (cards_in_full_set - 1)) / 2

theorem probability_of_pair (m n : ℕ) (h : m.gcd n = 1) :
  (m : ℚ) / n = favorable_outcomes / total_combinations → m = 57 ∧ n = 703 := by
  sorry

end probability_of_pair_l2646_264680


namespace inscribed_square_perimeter_l2646_264667

/-- The perimeter of a square inscribed in a right triangle -/
theorem inscribed_square_perimeter (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let x := a * b / (a + b)
  4 * x = 4 * a * b / (a + b) := by
  sorry

end inscribed_square_perimeter_l2646_264667


namespace sum_first_15_even_positive_l2646_264602

/-- The sum of the first n even positive integers -/
def sum_first_n_even_positive (n : ℕ) : ℕ :=
  n * (n + 1)

/-- Theorem: The sum of the first 15 even positive integers is 240 -/
theorem sum_first_15_even_positive :
  sum_first_n_even_positive 15 = 240 := by
  sorry

#eval sum_first_n_even_positive 15  -- This should output 240

end sum_first_15_even_positive_l2646_264602


namespace two_true_propositions_l2646_264683

theorem two_true_propositions :
  let original := ∀ x : ℝ, x > 0 → x^2 > 0
  let converse := ∀ x : ℝ, x^2 > 0 → x > 0
  let negation := ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0
  let contrapositive := ∀ x : ℝ, x^2 ≤ 0 → x ≤ 0
  (original ∧ ¬converse ∧ ¬negation ∧ contrapositive) :=
by
  sorry

end two_true_propositions_l2646_264683


namespace tan_45_degrees_equals_one_l2646_264628

theorem tan_45_degrees_equals_one :
  let θ : Real := 45 * π / 180  -- Convert 45 degrees to radians
  let tan_θ := Real.tan θ
  let sin_θ := Real.sin θ
  let cos_θ := Real.cos θ
  (∀ α, Real.tan α = Real.sin α / Real.cos α) →  -- General tangent identity
  sin_θ = Real.sqrt 2 / 2 →  -- Given value of sin 45°
  cos_θ = Real.sqrt 2 / 2 →  -- Given value of cos 45°
  tan_θ = 1 := by
sorry

end tan_45_degrees_equals_one_l2646_264628


namespace no_2000_digit_square_with_1999_fives_l2646_264612

theorem no_2000_digit_square_with_1999_fives : 
  ¬ ∃ n : ℕ, 
    (10^1999 ≤ n) ∧ (n < 10^2000) ∧  -- 2000-digit integer
    (∃ k : ℕ, n = k^2) ∧              -- perfect square
    (∃ d : ℕ, d < 10 ∧                -- at least 1999 digits of "5"
      (n / 10 = 5 * (10^1998 - 1) / 9 + d * 10^1998 ∨
       n % 10 ≠ 5 ∧ n / 10 = 5 * (10^1999 - 1) / 9)) :=
by sorry

end no_2000_digit_square_with_1999_fives_l2646_264612


namespace congruence_problem_l2646_264608

theorem congruence_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 
  (3 * (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1515151514)) % 9 = n :=
by
  -- The proof goes here
  sorry

end congruence_problem_l2646_264608


namespace water_pouring_theorem_l2646_264679

-- Define the pouring process
def remaining_water (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

-- Theorem statement
theorem water_pouring_theorem :
  remaining_water 8 = 1/5 := by
  sorry

end water_pouring_theorem_l2646_264679


namespace inequality_implies_range_l2646_264620

theorem inequality_implies_range (a : ℝ) : (1^2 * a + 2 * 1 + 1 < 0) → a < -3 := by
  sorry

end inequality_implies_range_l2646_264620


namespace bugs_eat_seventeen_flowers_l2646_264613

/-- Represents the number of flowers eaten by each type of bug -/
structure BugEating where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Represents the number of bugs of each type -/
structure BugCount where
  typeA : Nat
  typeB : Nat
  typeC : Nat

/-- Calculates the total number of flowers eaten by all bugs -/
def totalFlowersEaten (eating : BugEating) (count : BugCount) : Nat :=
  eating.typeA * count.typeA + eating.typeB * count.typeB + eating.typeC * count.typeC

theorem bugs_eat_seventeen_flowers : 
  let eating : BugEating := { typeA := 2, typeB := 3, typeC := 5 }
  let count : BugCount := { typeA := 3, typeB := 2, typeC := 1 }
  totalFlowersEaten eating count = 17 := by
  sorry

end bugs_eat_seventeen_flowers_l2646_264613


namespace paul_sunday_bags_l2646_264605

/-- The number of bags Paul filled on Saturday -/
def saturday_bags : ℕ := 6

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- The number of bags Paul filled on Sunday -/
def sunday_bags : ℕ := (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

theorem paul_sunday_bags :
  sunday_bags = 3 := by sorry

end paul_sunday_bags_l2646_264605


namespace soccer_game_scoring_l2646_264610

theorem soccer_game_scoring (team_a_first_half : ℕ) : 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) / 2 + 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) - 2 = 26 →
  team_a_first_half = 8 := by
sorry

end soccer_game_scoring_l2646_264610


namespace final_x_value_l2646_264673

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ  -- Current value of X
  s : ℕ  -- Current sum S
  n : ℕ  -- Number of iterations

/-- Updates the state for the next iteration -/
def nextState (state : State) : State :=
  { x := state.x + 2,
    s := state.s + state.x + 2,
    n := state.n + 1 }

/-- Computes the final state when S ≥ 10000 -/
def finalState : State :=
  sorry

/-- The main theorem to prove -/
theorem final_x_value :
  finalState.x = 201 ∧ finalState.s ≥ 10000 ∧
  ∀ (prev : State), prev.n < finalState.n → prev.s < 10000 :=
sorry

end final_x_value_l2646_264673


namespace cos_4alpha_minus_9pi_over_2_l2646_264623

theorem cos_4alpha_minus_9pi_over_2 (α : ℝ) : 
  4.53 * (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) / 
  (Real.cos (2 * Real.pi - 2 * α) + 2 * (Real.cos (2 * α + Real.pi))^2 - 1) = 2 * Real.cos (2 * α) →
  Real.cos (4 * α - 9 * Real.pi / 2) = Real.cos (4 * α - Real.pi / 2) := by
sorry

end cos_4alpha_minus_9pi_over_2_l2646_264623


namespace f_f_zero_l2646_264603

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end f_f_zero_l2646_264603


namespace total_movies_equals_sum_watched_and_to_watch_l2646_264642

/-- The 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : ℕ
  total_movies : ℕ
  books_read : ℕ
  movies_watched : ℕ
  movies_to_watch : ℕ

/-- Theorem: The total number of movies in the series is equal to the sum of movies watched and movies left to watch -/
theorem total_movies_equals_sum_watched_and_to_watch (css : CrazySillySchool) 
  (h1 : css.total_books = 4)
  (h2 : css.books_read = 19)
  (h3 : css.movies_watched = 7)
  (h4 : css.movies_to_watch = 10) :
  css.total_movies = css.movies_watched + css.movies_to_watch :=
by
  sorry

#eval 7 + 10  -- Expected output: 17

end total_movies_equals_sum_watched_and_to_watch_l2646_264642


namespace trapezoid_area_l2646_264631

/-- A trapezoid ABCD with the following properties:
  * BC = 5
  * Distance from A to line BC is 3
  * Distance from D to line BC is 7
-/
structure Trapezoid where
  BC : ℝ
  dist_A_to_BC : ℝ
  dist_D_to_BC : ℝ
  h_BC : BC = 5
  h_dist_A : dist_A_to_BC = 3
  h_dist_D : dist_D_to_BC = 7

/-- The area of the trapezoid ABCD -/
def area (t : Trapezoid) : ℝ := 25

/-- Theorem stating that the area of the trapezoid ABCD is 25 -/
theorem trapezoid_area (t : Trapezoid) : area t = 25 := by
  sorry

end trapezoid_area_l2646_264631


namespace correct_calculation_l2646_264678

theorem correct_calculation (a b : ℝ) : 3 * a * b^2 - 5 * b^2 * a = -2 * a * b^2 := by
  sorry

end correct_calculation_l2646_264678


namespace unique_triangle_side_length_l2646_264674

open Real

theorem unique_triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 4 →
  b = 2 * Real.sqrt 2 →
  0 < B →
  B < 3 * π / 4 →
  0 < C →
  C < 3 * π / 4 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  (∀ B' C' b' c', 
    0 < B' → B' < 3 * π / 4 → 
    0 < C' → C' < 3 * π / 4 → 
    A + B' + C' = π → 
    2 / sin A = b' / sin B' → 
    2 / sin A = c' / sin C' → 
    (B' = B ∧ C' = C ∧ b' = b ∧ c' = c)) →
  b = 2 * Real.sqrt 2 :=
by sorry

end unique_triangle_side_length_l2646_264674


namespace scientific_notation_correct_l2646_264601

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def target_number : ℕ := 101000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.01
    exponent := 5
    coeff_range := by sorry }

theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = target_number := by
  sorry

end scientific_notation_correct_l2646_264601


namespace spherical_coordinate_equivalence_l2646_264657

-- Define the type for spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the standard representation constraints
def isStandardRepresentation (coord : SphericalCoord) : Prop :=
  coord.ρ > 0 ∧ 0 ≤ coord.θ ∧ coord.θ < 2 * Real.pi ∧ 0 ≤ coord.φ ∧ coord.φ ≤ Real.pi

-- Define the equivalence relation between spherical coordinates
def sphericalEquivalent (coord1 coord2 : SphericalCoord) : Prop :=
  coord1.ρ = coord2.ρ ∧
  (coord1.θ % (2 * Real.pi) = coord2.θ % (2 * Real.pi)) ∧
  ((coord1.φ % (2 * Real.pi) = coord2.φ % (2 * Real.pi)) ∨
   (coord1.φ % (2 * Real.pi) = 2 * Real.pi - (coord2.φ % (2 * Real.pi))))

-- Theorem statement
theorem spherical_coordinate_equivalence :
  let original := SphericalCoord.mk 5 (5 * Real.pi / 6) (9 * Real.pi / 5)
  let standard := SphericalCoord.mk 5 (11 * Real.pi / 6) (Real.pi / 5)
  sphericalEquivalent original standard ∧ isStandardRepresentation standard :=
by sorry

end spherical_coordinate_equivalence_l2646_264657


namespace max_profit_at_max_price_verify_conditions_l2646_264669

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 450

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 15) * (sales_volume x)

-- Define the maximum allowed price
def max_price : ℝ := 28

-- Theorem statement
theorem max_profit_at_max_price :
  (∀ x, x ≤ max_price → profit x ≤ profit max_price) ∧
  profit max_price = 2210 := by
  sorry

-- Verify the conditions given in the problem
theorem verify_conditions :
  sales_volume 20 = 250 ∧
  profit 25 = 2000 ∧
  (∃ k b, ∀ x, sales_volume x = k * x + b) := by
  sorry

end max_profit_at_max_price_verify_conditions_l2646_264669


namespace inequality_proof_l2646_264643

theorem inequality_proof (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) : 
  1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9 := by
  sorry

end inequality_proof_l2646_264643


namespace hyperbola_properties_l2646_264651

/-- Given a hyperbola C with equation x²/(m²+3) - y²/m² = 1 where m > 0,
    and asymptote equation y = ±(1/2)x, prove the following properties --/
theorem hyperbola_properties (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (m^2 + 3) - y^2 / m^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (1/2) * x ∨ y = -(1/2) * x}
  (∀ (x y : ℝ), (x, y) ∈ C → (x, y) ∈ asymptote → x ≠ 0 → y / x = 1/2 ∨ y / x = -1/2) →
  (m = 1) ∧
  (∃ (x y : ℝ), (x, y) ∈ C ∧ y = Real.log (x - 1) ∧ 
    (x^2 / (m^2 + 3) = 1 ∨ y = 0)) ∧
  (∀ (x y : ℝ), y^2 - x^2 / 4 = 1 ↔ (x, y) ∈ asymptote) :=
by sorry

end hyperbola_properties_l2646_264651


namespace ellipse_circle_theorem_l2646_264645

/-- Definition of the ellipse C -/
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / b^2 = 1 ∧ b > 0

/-- Definition of the circle O -/
def circle_O (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2 ∧ r > 0

/-- Definition of the right focus F of ellipse C -/
def right_focus (F : ℝ × ℝ) (b : ℝ) : Prop :=
  F.1 > 0 ∧ ellipse_C b F.1 F.2

/-- Definition of the tangent lines from F to circle O -/
def tangent_lines (F A B : ℝ × ℝ) (r : ℝ) : Prop :=
  circle_O r A.1 A.2 ∧ circle_O r B.1 B.2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

/-- Definition of right triangle ABF -/
def right_triangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0

/-- Definition of maximum distance between points on C and O -/
def max_distance (b r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C b x₁ y₁ ∧ circle_O r x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (Real.sqrt 3 + 1)^2

/-- Main theorem -/
theorem ellipse_circle_theorem
  (b r : ℝ) (F A B : ℝ × ℝ) :
  ellipse_C b F.1 F.2 →
  right_focus F b →
  circle_O r A.1 A.2 →
  tangent_lines F A B r →
  right_triangle A B F →
  max_distance b r →
  (r = 1 ∧ b = 1) ∧
  (∀ (k m : ℝ), k < 0 → m > 0 →
    (∃ (P Q : ℝ × ℝ),
      ellipse_C b P.1 P.2 ∧ ellipse_C b Q.1 Q.2 ∧
      P.2 = k * P.1 + m ∧ Q.2 = k * Q.1 + m ∧
      (P.1 - F.1)^2 + (P.2 - F.2)^2 +
      (Q.1 - F.1)^2 + (Q.2 - F.2)^2 +
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12)) :=
sorry

end ellipse_circle_theorem_l2646_264645


namespace min_weeks_for_puppies_l2646_264681

/-- Represents the different types of puppies Bob can buy -/
inductive PuppyType
  | GoldenRetriever
  | Poodle
  | Beagle

/-- Calculates the minimum number of weeks Bob needs to compete to afford a puppy -/
def min_weeks_to_afford (puppy : PuppyType) (entrance_fee : ℕ) (first_place_prize : ℕ) (current_savings : ℕ) : ℕ :=
  match puppy with
  | PuppyType.GoldenRetriever => 10
  | PuppyType.Poodle => 7
  | PuppyType.Beagle => 5

/-- Theorem stating the minimum number of weeks Bob needs to compete for each puppy type -/
theorem min_weeks_for_puppies 
  (entrance_fee : ℕ) 
  (first_place_prize : ℕ) 
  (second_place_prize : ℕ) 
  (third_place_prize : ℕ) 
  (current_savings : ℕ) 
  (golden_price : ℕ) 
  (poodle_price : ℕ) 
  (beagle_price : ℕ) 
  (h1 : entrance_fee = 10)
  (h2 : first_place_prize = 100)
  (h3 : second_place_prize = 70)
  (h4 : third_place_prize = 40)
  (h5 : current_savings = 180)
  (h6 : golden_price = 1000)
  (h7 : poodle_price = 800)
  (h8 : beagle_price = 600) :
  (min_weeks_to_afford PuppyType.GoldenRetriever entrance_fee first_place_prize current_savings = 10) ∧
  (min_weeks_to_afford PuppyType.Poodle entrance_fee first_place_prize current_savings = 7) ∧
  (min_weeks_to_afford PuppyType.Beagle entrance_fee first_place_prize current_savings = 5) :=
by sorry

end min_weeks_for_puppies_l2646_264681


namespace sufficient_condition_not_necessary_condition_l2646_264666

/-- A quadratic function represented by its coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Predicate for a quadratic function passing through the origin -/
def passes_through_origin (f : QuadraticFunction) : Prop :=
  f.a * 0^2 + f.b * 0 + f.c = 0

/-- Theorem stating that b = c = 0 is a sufficient condition -/
theorem sufficient_condition (f : QuadraticFunction) (h1 : f.b = 0) (h2 : f.c = 0) :
  passes_through_origin f := by sorry

/-- Theorem stating that b = c = 0 is not a necessary condition -/
theorem not_necessary_condition :
  ∃ f : QuadraticFunction, passes_through_origin f ∧ f.b ≠ 0 := by sorry

end sufficient_condition_not_necessary_condition_l2646_264666


namespace max_removable_marbles_l2646_264668

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCount where
  yellow : Nat
  red : Nat
  black : Nat

/-- The initial number of marbles in the bag -/
def initialMarbles : MarbleCount := ⟨8, 7, 5⟩

/-- The condition that must be satisfied after removing marbles -/
def satisfiesCondition (mc : MarbleCount) : Prop :=
  (mc.yellow ≥ 4 ∧ (mc.red ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.red ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.black ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.red ≥ 3))

/-- The maximum number of marbles that can be removed -/
def maxRemovable : Nat := 7

theorem max_removable_marbles :
  (∀ (removed : Nat), removed ≤ maxRemovable →
    ∀ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed →
      satisfiesCondition remaining) ∧
  (∀ (removed : Nat), removed > maxRemovable →
    ∃ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed ∧
      ¬satisfiesCondition remaining) := by
  sorry

end max_removable_marbles_l2646_264668


namespace arithmetic_simplification_l2646_264684

theorem arithmetic_simplification :
  (30 - (2030 - 30 * 2)) + (2030 - (30 * 2 - 30)) = 60 := by
  sorry

end arithmetic_simplification_l2646_264684


namespace pie_arrangement_rows_l2646_264626

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Proves that 16 pecan pies and 14 apple pies, when arranged in rows of 5 pies each, result in 6 complete rows. -/
theorem pie_arrangement_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end pie_arrangement_rows_l2646_264626


namespace small_box_width_l2646_264687

/-- Proves that the width of smaller boxes is 50 cm given the conditions of the problem -/
theorem small_box_width (large_length large_width large_height : ℝ)
                        (small_length small_height : ℝ)
                        (max_small_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_height = 0.4 →
  max_small_boxes = 1000 →
  ∃ (small_width : ℝ),
    small_width = 0.5 ∧
    (max_small_boxes : ℝ) * small_length * small_width * small_height =
    large_length * large_width * large_height :=
by sorry

#check small_box_width

end small_box_width_l2646_264687


namespace inequality_proof_l2646_264699

theorem inequality_proof (t : ℝ) (h : 0 ≤ t ∧ t ≤ 6) : 
  Real.sqrt 6 ≤ Real.sqrt (-t + 6) + Real.sqrt t ∧ 
  Real.sqrt (-t + 6) + Real.sqrt t ≤ 2 * Real.sqrt 3 :=
by sorry

end inequality_proof_l2646_264699


namespace fermat_prime_power_of_two_l2646_264600

theorem fermat_prime_power_of_two (n : ℕ+) : 
  (Nat.Prime (2^(n : ℕ) + 1)) → (∃ k : ℕ, n = 2^k) := by
  sorry

end fermat_prime_power_of_two_l2646_264600


namespace f_value_at_2_l2646_264609

/-- Given a function f(x) = a*sin(x) + b*x*cos(x) - 2c*tan(x) + x^2 where f(-2) = 3,
    prove that f(2) = 5 -/
theorem f_value_at_2 (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * x * Real.cos x - 2 * c * Real.tan x + x^2)
  (h2 : f (-2) = 3) :
  f 2 = 5 := by
  sorry

end f_value_at_2_l2646_264609


namespace negative_integers_abs_not_greater_than_4_l2646_264697

def negativeIntegersWithAbsNotGreaterThan4 : Set ℤ :=
  {x : ℤ | x < 0 ∧ |x| ≤ 4}

theorem negative_integers_abs_not_greater_than_4 :
  negativeIntegersWithAbsNotGreaterThan4 = {-1, -2, -3, -4} := by
  sorry

end negative_integers_abs_not_greater_than_4_l2646_264697


namespace seashells_left_l2646_264636

def total_seashells : ℕ := 679
def clam_shells : ℕ := 325
def conch_shells : ℕ := 210
def oyster_shells : ℕ := 144
def starfish : ℕ := 110

def clam_percentage : ℚ := 40 / 100
def conch_percentage : ℚ := 25 / 100
def oyster_fraction : ℚ := 1 / 3

theorem seashells_left : 
  (clam_shells - Int.floor (clam_percentage * clam_shells)) +
  (conch_shells - Int.ceil (conch_percentage * conch_shells)) +
  (oyster_shells - Int.floor (oyster_fraction * oyster_shells)) +
  starfish = 558 := by
sorry

end seashells_left_l2646_264636


namespace jericho_money_left_l2646_264654

/-- The amount of money Jericho has initially -/
def jerichos_money : ℚ := 30

/-- The amount Jericho owes Annika -/
def annika_debt : ℚ := 14

/-- The amount Jericho owes Manny -/
def manny_debt : ℚ := annika_debt / 2

/-- Theorem stating that Jericho will be left with $9 after paying his debts -/
theorem jericho_money_left : jerichos_money - (annika_debt + manny_debt) = 9 := by
  sorry

end jericho_money_left_l2646_264654


namespace steves_salary_l2646_264647

theorem steves_salary (take_home_pay : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) 
  (h1 : take_home_pay = 27200)
  (h2 : tax_rate = 0.20)
  (h3 : healthcare_rate = 0.10)
  (h4 : union_dues = 800) :
  ∃ (original_salary : ℝ), 
    original_salary * (1 - tax_rate - healthcare_rate) - union_dues = take_home_pay ∧ 
    original_salary = 40000 := by
sorry

end steves_salary_l2646_264647


namespace north_southland_population_increase_l2646_264621

/-- The number of hours between each birth in North Southland -/
def hours_between_births : ℕ := 6

/-- The number of deaths per day in North Southland -/
def deaths_per_day : ℕ := 2

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The population increase in North Southland per year -/
def population_increase : ℕ := 
  (24 / hours_between_births - deaths_per_day) * days_per_year

/-- The population increase in North Southland rounded to the nearest hundred -/
def rounded_population_increase : ℕ := 
  (population_increase + 50) / 100 * 100

theorem north_southland_population_increase :
  rounded_population_increase = 700 := by sorry

end north_southland_population_increase_l2646_264621


namespace third_year_sample_size_l2646_264694

/-- Calculates the number of students to be sampled from a specific grade in a stratified sampling. -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample : ℕ) : ℕ :=
  (grade_population * total_sample) / total_population

theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1200
  let total_sample : ℕ := 50
  stratified_sample_size total_population third_year_population total_sample = 20 := by
  sorry

end third_year_sample_size_l2646_264694


namespace sufficient_but_not_necessary_condition_l2646_264688

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (∀ x, x < 1 → x < 2) ∧ (∃ x, x < 2 ∧ ¬(x < 1)) := by sorry

end sufficient_but_not_necessary_condition_l2646_264688


namespace max_ice_creams_l2646_264696

/-- Given a budget and costs of items, calculate the maximum number of ice creams that can be bought -/
theorem max_ice_creams (budget : ℕ) (pancake_cost ice_cream_cost pancakes_bought : ℕ) : 
  budget = 60 →
  pancake_cost = 5 →
  ice_cream_cost = 8 →
  pancakes_bought = 5 →
  (budget - pancake_cost * pancakes_bought) / ice_cream_cost = 4 := by
  sorry

end max_ice_creams_l2646_264696


namespace right_triangle_area_l2646_264617

theorem right_triangle_area (longer_leg : ℝ) (angle : ℝ) :
  longer_leg = 10 →
  angle = 30 * (π / 180) →
  ∃ (area : ℝ), area = (50 * Real.sqrt 3) / 3 ∧
  area = (1 / 2) * longer_leg * (longer_leg / Real.sqrt 3) :=
by sorry

end right_triangle_area_l2646_264617


namespace worker_savings_l2646_264646

theorem worker_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) : 
  12 * f * P = 2 * (1 - f) * P → f = 1 / 7 := by
  sorry

end worker_savings_l2646_264646


namespace fishing_problem_l2646_264611

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jason + ryan + jeffery = 100 →
  jeffery = 60 →
  ryan = 30 := by sorry

end fishing_problem_l2646_264611


namespace average_score_range_l2646_264685

/-- Represents the score distribution in the math competition --/
structure ScoreDistribution where
  score_100 : ℕ
  score_90_99 : ℕ
  score_80_89 : ℕ
  score_70_79 : ℕ
  score_60_69 : ℕ
  score_50_59 : ℕ
  score_48 : ℕ

/-- Calculates the minimum possible average score --/
def min_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 90 * sd.score_90_99 + 80 * sd.score_80_89 + 70 * sd.score_70_79 +
   60 * sd.score_60_69 + 50 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- Calculates the maximum possible average score --/
def max_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 99 * sd.score_90_99 + 89 * sd.score_80_89 + 79 * sd.score_70_79 +
   69 * sd.score_60_69 + 59 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- The score distribution for the given problem --/
def zhi_cheng_distribution : ScoreDistribution :=
  { score_100 := 2
  , score_90_99 := 9
  , score_80_89 := 17
  , score_70_79 := 28
  , score_60_69 := 36
  , score_50_59 := 7
  , score_48 := 1
  }

/-- Theorem stating the range of the overall average score --/
theorem average_score_range :
  min_average_score zhi_cheng_distribution ≥ 68.88 ∧
  max_average_score zhi_cheng_distribution ≤ 77.61 := by
  sorry

end average_score_range_l2646_264685


namespace root_in_interval_implies_a_range_l2646_264604

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
by sorry

end root_in_interval_implies_a_range_l2646_264604


namespace solve_salary_problem_l2646_264630

def salary_problem (a b : ℝ) : Prop :=
  a + b = 4000 ∧
  0.05 * a = 0.15 * b ∧
  a = 3000

theorem solve_salary_problem :
  ∃ (a b : ℝ), salary_problem a b :=
sorry

end solve_salary_problem_l2646_264630


namespace natalia_clip_sales_l2646_264691

/-- The total number of clips Natalia sold in April and May -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ := april_sales + may_sales

/-- Theorem stating the total number of clips sold given the conditions -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ), 
  april_sales = 48 → 
  total_clips april_sales (april_sales / 2) = 72 := by
sorry

end natalia_clip_sales_l2646_264691


namespace quadratic_roots_relation_l2646_264690

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 4 * r - 8 = 0) ∧ 
               (2 * s^2 - 4 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 11 := by
sorry

end quadratic_roots_relation_l2646_264690


namespace max_angles_theorem_l2646_264616

theorem max_angles_theorem (k : ℕ) :
  let n := 2 * k
  ∃ (max_angles : ℕ), max_angles = 2 * k - 1 ∧
    ∀ (num_angles : ℕ), num_angles ≤ max_angles :=
by sorry

end max_angles_theorem_l2646_264616


namespace triangle_side_length_l2646_264689

theorem triangle_side_length (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_thirty_deg : a / c = 1 / 2) (h_hypotenuse : c = 6 * Real.sqrt 2) :
  b = 2 * Real.sqrt 6 := by
  sorry

end triangle_side_length_l2646_264689


namespace gallery_to_work_blocks_l2646_264692

/-- The number of blocks from start to work -/
def total_blocks : ℕ := 37

/-- The number of blocks to the store -/
def store_blocks : ℕ := 11

/-- The number of blocks to the gallery -/
def gallery_blocks : ℕ := 6

/-- The number of blocks already walked -/
def walked_blocks : ℕ := 5

/-- The number of remaining blocks to work after walking 5 blocks -/
def remaining_blocks : ℕ := 20

/-- The number of blocks from the gallery to work -/
def gallery_to_work : ℕ := total_blocks - walked_blocks - store_blocks - gallery_blocks

theorem gallery_to_work_blocks :
  gallery_to_work = 15 :=
by sorry

end gallery_to_work_blocks_l2646_264692


namespace arithmetic_sequence_sum_l2646_264693

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 6 = 12 →
  a 2 + a 9 = 12 := by
  sorry

end arithmetic_sequence_sum_l2646_264693


namespace circle_intersection_range_l2646_264676

-- Define the circles
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 4 = 0

def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the intersection condition
def intersect_at_all_times (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ circle_O x y

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect_at_all_times a ↔ 
    ((-2 * Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * Real.sqrt 2)) :=
by sorry

end circle_intersection_range_l2646_264676


namespace all_stars_arrangement_l2646_264649

/-- The number of ways to arrange All-Stars from different teams in a row -/
def arrange_all_stars (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial cubs) * (Nat.factorial red_sox) * (Nat.factorial yankees)

/-- Theorem stating that there are 6912 ways to arrange 10 All-Stars with the given conditions -/
theorem all_stars_arrangement :
  arrange_all_stars 4 4 2 = 6912 := by
  sorry

end all_stars_arrangement_l2646_264649


namespace expression_evaluation_l2646_264682

theorem expression_evaluation : (((2200 - 2081)^2 + 100) : ℚ) / 196 = 73 := by sorry

end expression_evaluation_l2646_264682


namespace valid_triples_l2646_264627

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, y = x * r ∧ z = y * r

def validTriple (a b c : Nat) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  isGeometricSequence (a + 1) (b + 1) (c + 1)

theorem valid_triples :
  {t : Nat × Nat × Nat | validTriple t.1 t.2.1 t.2.2} =
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 23, 71), (11, 23, 47)} :=
by sorry

end valid_triples_l2646_264627


namespace games_planned_this_month_l2646_264618

theorem games_planned_this_month
  (total_attended : ℕ)
  (planned_last_month : ℕ)
  (missed : ℕ)
  (h1 : total_attended = 12)
  (h2 : planned_last_month = 17)
  (h3 : missed = 16)
  : ℕ := by
  sorry

end games_planned_this_month_l2646_264618


namespace x_range_proof_l2646_264622

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem x_range_proof (f : ℝ → ℝ) (h_odd : is_odd f) (h_decreasing : is_decreasing f)
  (h_domain : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ∈ Set.Icc (-3 : ℝ) 3)
  (h_inequality : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f (x^2 - 2*x) + f (x - 2) < 0) :
  ∀ x, x ∈ Set.Ioc (2 : ℝ) 3 := by
sorry

end x_range_proof_l2646_264622


namespace star_emilio_sum_difference_l2646_264658

def star_list : List Nat := List.range 40

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 104 := by sorry

end star_emilio_sum_difference_l2646_264658


namespace sum_of_squares_of_cubic_roots_l2646_264624

/-- Given a cubic equation dx³ - ex² + fx - g = 0 with real coefficients,
    the sum of squares of its roots is (e² - 2df) / d². -/
theorem sum_of_squares_of_cubic_roots
  (d e f g : ℝ) (a b c : ℝ)
  (hroots : d * (X - a) * (X - b) * (X - c) = d * X^3 - e * X^2 + f * X - g) :
  a^2 + b^2 + c^2 = (e^2 - 2*d*f) / d^2 := by
  sorry

end sum_of_squares_of_cubic_roots_l2646_264624


namespace centroid_vector_sum_l2646_264664

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC with centroid G and an arbitrary point P,
    prove that PG = 1/3(PA + PB + PC) -/
theorem centroid_vector_sum (A B C P G : V) 
  (h : G = (1/3 : ℝ) • (A + B + C)) : 
  G - P = (1/3 : ℝ) • ((A - P) + (B - P) + (C - P)) := by
  sorry

end centroid_vector_sum_l2646_264664


namespace james_lollipops_distribution_l2646_264638

/-- The number of lollipops James has left after distributing to his friends -/
def lollipops_left (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- Theorem stating that James has 0 lollipops left after distribution -/
theorem james_lollipops_distribution :
  let total_lollipops : ℕ := 56 + 130 + 10 + 238
  let num_friends : ℕ := 14
  lollipops_left total_lollipops num_friends = 0 := by sorry

end james_lollipops_distribution_l2646_264638


namespace power_of_product_l2646_264698

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l2646_264698


namespace inverse_proportion_ratio_l2646_264635

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end inverse_proportion_ratio_l2646_264635


namespace f_neg_two_l2646_264652

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg_two : f (-2) = -1 := by sorry

end f_neg_two_l2646_264652


namespace min_value_function_l2646_264655

theorem min_value_function (x : ℝ) (h : x > 1) : 
  ∀ y : ℝ, y = 4 / (x - 1) + x → y ≥ 5 :=
by sorry

end min_value_function_l2646_264655


namespace unique_integer_solution_l2646_264632

theorem unique_integer_solution : ∃! x : ℤ, 3 * (x + 200000) = 10 * x + 2 :=
  by sorry

end unique_integer_solution_l2646_264632


namespace cube_congruence_for_prime_l2646_264665

theorem cube_congruence_for_prime (p : ℕ) (k : ℕ) 
  (hp : Nat.Prime p) (hform : p = 3 * k + 1) : 
  ∃ a b : ℕ, 0 < a ∧ a < b ∧ b < Real.sqrt p ∧ a^3 ≡ b^3 [MOD p] := by
  sorry

end cube_congruence_for_prime_l2646_264665
