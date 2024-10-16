import Mathlib

namespace NUMINAMATH_CALUDE_max_roads_in_graphia_l2926_292686

/-- A graph representing the towns and roads in Graphia. -/
structure Graphia where
  towns : Finset Nat
  roads : Finset (Nat × Nat)

/-- The number of towns in Graphia. -/
def num_towns : Nat := 100

/-- A function representing Peter's travel pattern. -/
def peter_travel (g : Graphia) (start : Nat) : List Nat := sorry

/-- The condition that each town is visited exactly twice. -/
def all_towns_visited_twice (g : Graphia) : Prop :=
  ∀ t ∈ g.towns, (List.count t (List.join (List.map (peter_travel g) (List.range num_towns)))) = 2

/-- The theorem stating the maximum number of roads in Graphia. -/
theorem max_roads_in_graphia :
  ∀ g : Graphia,
    g.towns.card = num_towns →
    all_towns_visited_twice g →
    g.roads.card ≤ 4851 :=
sorry

end NUMINAMATH_CALUDE_max_roads_in_graphia_l2926_292686


namespace NUMINAMATH_CALUDE_annie_cookies_l2926_292665

theorem annie_cookies (x : ℝ) 
  (h1 : x + 2*x + 2.8*x = 29) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_l2926_292665


namespace NUMINAMATH_CALUDE_black_ball_probability_l2926_292684

theorem black_ball_probability 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h_red : p_red = 0.42) 
  (h_white : p_white = 0.28) :
  1 - p_red - p_white = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_black_ball_probability_l2926_292684


namespace NUMINAMATH_CALUDE_min_value_rational_function_l2926_292645

theorem min_value_rational_function :
  (∀ x : ℝ, x > -1 → ((x^2 + 7*x + 10) / (x + 1)) ≥ 9) ∧
  (∃ x : ℝ, x > -1 ∧ ((x^2 + 7*x + 10) / (x + 1)) = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l2926_292645


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2926_292618

/-- Represents the numbers of various individuals in a mathematical problem. -/
structure Numbers where
  k : ℕ
  joyce : ℕ
  xavier : ℕ
  coraline : ℕ
  jayden : ℕ
  mickey : ℕ
  yvonne : ℕ
  natalie : ℕ

/-- The conditions of the problem are satisfied. -/
def satisfies_conditions (n : Numbers) : Prop :=
  n.k > 1 ∧
  n.joyce = 5 * n.k ∧
  n.xavier = 4 * n.joyce ∧
  n.coraline = n.xavier + 50 ∧
  n.jayden = n.coraline - 40 ∧
  n.mickey = n.jayden + 20 ∧
  n.yvonne = (n.xavier + n.joyce) * n.k ∧
  n.natalie = (n.yvonne - n.coraline) / 2

/-- The theorem to be proved. -/
theorem sum_of_numbers (n : Numbers) 
  (h : satisfies_conditions n) : 
  n.joyce + n.xavier + n.coraline + n.jayden + n.mickey + n.yvonne + n.natalie = 365 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2926_292618


namespace NUMINAMATH_CALUDE_revenue_maximized_at_064_l2926_292625

/-- Revenue function for electricity pricing -/
def revenue (x : ℝ) : ℝ := (1 + 50 * (x - 0.8)^2) * (x - 0.5)

/-- The domain of the revenue function -/
def price_range (x : ℝ) : Prop := 0.5 < x ∧ x < 0.8

theorem revenue_maximized_at_064 :
  ∃ (x : ℝ), price_range x ∧
    (∀ (y : ℝ), price_range y → revenue y ≤ revenue x) ∧
    x = 0.64 :=
sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_064_l2926_292625


namespace NUMINAMATH_CALUDE_alpha_value_l2926_292653

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = -16 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2926_292653


namespace NUMINAMATH_CALUDE_multiply_by_213_equals_3408_l2926_292633

theorem multiply_by_213_equals_3408 (x : ℝ) : 213 * x = 3408 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_213_equals_3408_l2926_292633


namespace NUMINAMATH_CALUDE_inequality_solution_l2926_292690

theorem inequality_solution (x y : ℝ) :
  x + y^2 + Real.sqrt (x - y^2 - 1) ≤ 1 ∧
  x - y^2 - 1 ≥ 0 →
  x = 1 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2926_292690


namespace NUMINAMATH_CALUDE_derivative_of_sine_at_pi_sixth_l2926_292623

/-- Given f(x) = sin(2x + π/6), prove that f'(π/6) = 0 -/
theorem derivative_of_sine_at_pi_sixth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x + π / 6)) :
  deriv f (π / 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sine_at_pi_sixth_l2926_292623


namespace NUMINAMATH_CALUDE_intersection_subsets_count_l2926_292651

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}

theorem intersection_subsets_count :
  Finset.card (Finset.powerset (M ∩ N)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_subsets_count_l2926_292651


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2926_292685

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_coordinates :
  let A : Point2D := { x := 2, y := -8 }
  let B : Point2D := symmetricYAxis A
  B.x = -2 ∧ B.y = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2926_292685


namespace NUMINAMATH_CALUDE_parallelogram_area_l2926_292628

def u : Fin 3 → ℝ := ![4, 2, -3]
def v : Fin 3 → ℝ := ![2, -4, 5]

theorem parallelogram_area : 
  Real.sqrt ((u 0 * v 1 - u 1 * v 0)^2 + (u 0 * v 2 - u 2 * v 0)^2 + (u 1 * v 2 - u 2 * v 1)^2) = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2926_292628


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2926_292682

def a : Fin 2 → ℝ := ![0, 1]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_magnitude_proof : 
  ‖(2 • (a : Fin 2 → ℝ)) + (b : Fin 2 → ℝ)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2926_292682


namespace NUMINAMATH_CALUDE_problem_statement_l2926_292606

theorem problem_statement :
  (∀ a b c d : ℝ, a^6 + b^6 + c^6 + d^6 - 6*a*b*c*d ≥ -2) ∧
  (∀ k : ℕ, k % 2 = 1 → k ≥ 5 →
    ∃ M_k : ℝ, ∀ a b c d : ℝ, a^k + b^k + c^k + d^k - k*a*b*c*d ≥ M_k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2926_292606


namespace NUMINAMATH_CALUDE_complement_union_equality_l2926_292631

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 3, 4}

-- Define set B
def B : Set ℕ := {1, 3}

-- Theorem statement
theorem complement_union_equality : (U \ A) ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_equality_l2926_292631


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2926_292634

/-- Given 4 siblings where the ages of the older siblings are 3, 6, and 7 years more than 
    the youngest, and the average age of all siblings is 30, 
    the age of the youngest sibling is 26. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2926_292634


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2926_292676

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2926_292676


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2926_292696

/-- The value of a pig in dollars -/
def pig_value : ℕ := 500

/-- The value of a goat in dollars -/
def goat_value : ℕ := 350

/-- The smallest positive debt that can be resolved -/
def smallest_debt : ℕ := 50

theorem smallest_resolvable_debt :
  smallest_debt = Nat.gcd pig_value goat_value ∧
  ∃ (p g : ℤ), smallest_debt = p * pig_value + g * goat_value :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2926_292696


namespace NUMINAMATH_CALUDE_max_prob_highest_second_l2926_292699

/-- Represents a player in the chess game -/
structure Player where
  winProb : ℝ
  winProb_pos : winProb > 0

/-- Represents the chess game with three players -/
structure ChessGame where
  p₁ : Player
  p₂ : Player
  p₃ : Player
  prob_order : p₃.winProb > p₂.winProb ∧ p₂.winProb > p₁.winProb

/-- Calculates the probability of winning two consecutive games given the order of players -/
def probTwoConsecutiveWins (game : ChessGame) (second : Player) : ℝ :=
  2 * (second.winProb * (game.p₁.winProb + game.p₂.winProb + game.p₃.winProb - second.winProb) - 
       2 * game.p₁.winProb * game.p₂.winProb * game.p₃.winProb)

/-- Theorem stating that the probability of winning two consecutive games is maximized 
    when the player with the highest winning probability is played second -/
theorem max_prob_highest_second (game : ChessGame) :
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₂ ∧
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₁ := by
  sorry


end NUMINAMATH_CALUDE_max_prob_highest_second_l2926_292699


namespace NUMINAMATH_CALUDE_rowing_distance_l2926_292683

/-- Calculates the total distance traveled by a man rowing in a river --/
theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) : 
  v_man = 7 →
  v_river = 1.2 →
  total_time = 1 →
  let d := (v_man^2 - v_river^2) * total_time / (2 * v_man)
  2 * d = 7 := by sorry

end NUMINAMATH_CALUDE_rowing_distance_l2926_292683


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2926_292609

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 65,
    breadth := 35,
    fencingCostPerMeter := 26.5
  }
  totalFencingCost plot = 5300 := by sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2926_292609


namespace NUMINAMATH_CALUDE_common_solution_y_value_l2926_292636

theorem common_solution_y_value : ∃ (x y : ℝ), 
  (x^2 + y^2 - 16 = 0) ∧ 
  (x^2 - 3*y + 12 = 0) → 
  y = 4 := by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l2926_292636


namespace NUMINAMATH_CALUDE_card_drawing_probability_ratio_l2926_292655

theorem card_drawing_probability_ratio :
  let total_cards : ℕ := 60
  let num_range : ℕ := 15
  let cards_per_num : ℕ := 4
  let draw_count : ℕ := 4

  let p : ℚ := (num_range : ℚ) / (Nat.choose total_cards draw_count)
  let q : ℚ := (num_range * (num_range - 1) * Nat.choose cards_per_num 3 * Nat.choose cards_per_num 1 : ℚ) / 
                (Nat.choose total_cards draw_count)

  q / p = 224 := by sorry

end NUMINAMATH_CALUDE_card_drawing_probability_ratio_l2926_292655


namespace NUMINAMATH_CALUDE_angie_leftover_money_l2926_292626

def angie_finances (salary : ℕ) (necessities : ℕ) (taxes : ℕ) : ℕ :=
  salary - (necessities + taxes)

theorem angie_leftover_money :
  angie_finances 80 42 20 = 18 := by sorry

end NUMINAMATH_CALUDE_angie_leftover_money_l2926_292626


namespace NUMINAMATH_CALUDE_knight_journey_distance_l2926_292608

/-- The distance Knight Milivoj traveled on Monday -/
def monday_distance : ℕ := 25

/-- The additional distance Knight Milivoj traveled on Thursday compared to Monday -/
def thursday_additional_distance : ℕ := 6

/-- The distance Knight Milivoj traveled on Friday -/
def friday_distance : ℕ := 11

/-- The distance between Zubín and Veselín -/
def zubin_veselin_distance : ℕ := 17

theorem knight_journey_distance : 
  zubin_veselin_distance = 
    (monday_distance + thursday_additional_distance + friday_distance) - monday_distance :=
by sorry

end NUMINAMATH_CALUDE_knight_journey_distance_l2926_292608


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2926_292642

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_inequality : x₁^2 - 4*a*x₁ + 3*a^2 < 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 < 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), (y₁^2 - 4*a*y₁ + 3*a^2 < 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 < 0) → 
  (y₁ + y₂ + a / (y₁ * y₂)) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2926_292642


namespace NUMINAMATH_CALUDE_school_cinema_visit_payment_l2926_292615

/-- Represents the ticket pricing structure and student count for a cinema visit -/
structure CinemaVisit where
  individual_price : ℝ  -- Price of an individual ticket
  group_price : ℝ       -- Price of a group ticket for 10 people
  student_discount : ℝ  -- Discount rate for students (as a decimal)
  student_count : ℕ     -- Number of students

/-- Calculates the minimum amount to be paid for a school cinema visit -/
def minimum_payment (cv : CinemaVisit) : ℝ :=
  let group_size := 10
  let full_groups := cv.student_count / group_size
  let total_group_price := (full_groups * cv.group_price) * (1 - cv.student_discount)
  total_group_price

/-- The theorem stating the minimum payment for the given scenario -/
theorem school_cinema_visit_payment :
  let cv : CinemaVisit := {
    individual_price := 6,
    group_price := 40,
    student_discount := 0.1,
    student_count := 1258
  }
  minimum_payment cv = 4536 := by sorry

end NUMINAMATH_CALUDE_school_cinema_visit_payment_l2926_292615


namespace NUMINAMATH_CALUDE_pencils_per_row_l2926_292658

/-- Theorem: Number of pencils in each row when 6 pencils are equally distributed into 2 rows -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 6) (h2 : num_rows = 2) :
  total_pencils / num_rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2926_292658


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l2926_292635

theorem set_inclusion_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2*x + 3}
  let C := {z : ℝ | ∃ x ∈ A, z = x^2}
  C ⊆ B → (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l2926_292635


namespace NUMINAMATH_CALUDE_unique_divisiblity_solution_l2926_292614

theorem unique_divisiblity_solution : 
  ∀ a m n : ℕ+, 
    (a^n.val + 1 ∣ (a+1)^m.val) → 
    (a = 2 ∧ m = 2 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisiblity_solution_l2926_292614


namespace NUMINAMATH_CALUDE_abc_inequality_l2926_292695

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (sum_sq : a^2 + b^2 + c^2 = 9) :
  a * b * c + 1 > 3 * a :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2926_292695


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2926_292673

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := ℝ → ℝ

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : PositiveRealFunction, SatisfiesEquation f α) ↔
    (α = 1 ∧ ∃ f : PositiveRealFunction, SatisfiesEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2926_292673


namespace NUMINAMATH_CALUDE_customer_difference_l2926_292637

theorem customer_difference (initial : Nat) (remaining : Nat) : 
  initial = 11 → remaining = 3 → (initial - remaining) - remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l2926_292637


namespace NUMINAMATH_CALUDE_plain_cookie_price_l2926_292610

theorem plain_cookie_price 
  (chocolate_chip_price : ℝ) 
  (total_boxes : ℝ) 
  (total_value : ℝ) 
  (plain_boxes : ℝ) 
  (h1 : chocolate_chip_price = 1.25)
  (h2 : total_boxes = 1585)
  (h3 : total_value = 1586.25)
  (h4 : plain_boxes = 793.125) : 
  (total_value - (total_boxes - plain_boxes) * chocolate_chip_price) / plain_boxes = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_plain_cookie_price_l2926_292610


namespace NUMINAMATH_CALUDE_other_communities_count_l2926_292654

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 238 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l2926_292654


namespace NUMINAMATH_CALUDE_wax_remaining_after_detailing_l2926_292663

-- Define the initial amounts of wax
def waxA_initial : ℕ := 10
def waxB_initial : ℕ := 15

-- Define the amounts required for each vehicle
def waxA_car : ℕ := 4
def waxA_suv : ℕ := 6
def waxB_car : ℕ := 3
def waxB_suv : ℕ := 5

-- Define the amounts spilled
def waxA_spilled : ℕ := 3
def waxB_spilled : ℕ := 4

-- Theorem to prove
theorem wax_remaining_after_detailing :
  (waxA_initial - waxA_spilled - waxA_car) + (waxB_initial - waxB_spilled - waxB_suv) = 9 := by
  sorry

end NUMINAMATH_CALUDE_wax_remaining_after_detailing_l2926_292663


namespace NUMINAMATH_CALUDE_multiply_negative_four_with_three_halves_l2926_292675

theorem multiply_negative_four_with_three_halves : (-4 : ℚ) * (3/2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_four_with_three_halves_l2926_292675


namespace NUMINAMATH_CALUDE_points_per_player_l2926_292611

theorem points_per_player (total_points : ℕ) (num_players : ℕ) 
  (h1 : total_points = 18) (h2 : num_players = 9) :
  total_points / num_players = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_per_player_l2926_292611


namespace NUMINAMATH_CALUDE_shelf_capacity_l2926_292656

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- The total number of CDs that can fit on a shelf -/
def total_cds : ℕ := cds_per_rack * racks_per_shelf

theorem shelf_capacity : total_cds = 32 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l2926_292656


namespace NUMINAMATH_CALUDE_fraction_value_l2926_292644

theorem fraction_value : (1 : ℚ) / (4 * 5) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2926_292644


namespace NUMINAMATH_CALUDE_bisection_termination_condition_l2926_292659

/-- Bisection method termination condition -/
theorem bisection_termination_condition 
  (f : ℝ → ℝ) (x₁ x₂ e : ℝ) (h_e : e > 0) :
  |x₁ - x₂| < e → ∃ x, x ∈ [x₁, x₂] ∧ |f x| < e :=
sorry

end NUMINAMATH_CALUDE_bisection_termination_condition_l2926_292659


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2926_292616

def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2926_292616


namespace NUMINAMATH_CALUDE_nut_mixture_weight_l2926_292601

/-- Given a mixture of nuts with 5 parts almonds to 2 parts walnuts by weight,
    and 250 pounds of almonds, the total weight of the mixture is 350 pounds. -/
theorem nut_mixture_weight (almond_parts : ℕ) (walnut_parts : ℕ) (almond_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 2 →
  almond_weight = 250 →
  (almond_weight / almond_parts) * (almond_parts + walnut_parts) = 350 := by
  sorry

#check nut_mixture_weight

end NUMINAMATH_CALUDE_nut_mixture_weight_l2926_292601


namespace NUMINAMATH_CALUDE_washington_dc_july_4th_avg_temp_l2926_292639

def washington_dc_july_4th_temps : List ℝ := [90, 90, 90, 79, 71]

theorem washington_dc_july_4th_avg_temp :
  (washington_dc_july_4th_temps.sum / washington_dc_july_4th_temps.length : ℝ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_washington_dc_july_4th_avg_temp_l2926_292639


namespace NUMINAMATH_CALUDE_largest_c_value_l2926_292689

theorem largest_c_value (c : ℝ) (h : (3*c + 4)*(c - 2) = 9*c) : 
  c ≤ 4 ∧ ∃ (c : ℝ), (3*c + 4)*(c - 2) = 9*c ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l2926_292689


namespace NUMINAMATH_CALUDE_exp_ln_five_l2926_292677

theorem exp_ln_five : Real.exp (Real.log 5) = 5 := by sorry

end NUMINAMATH_CALUDE_exp_ln_five_l2926_292677


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2926_292627

theorem consecutive_integers_square_sum (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = a + 2 → 
  d = a + 3 → 
  e = a + 4 → 
  a^2 + b^2 + c^2 = d^2 + e^2 → 
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2926_292627


namespace NUMINAMATH_CALUDE_surface_area_cube_with_corners_removed_l2926_292604

/-- Represents the dimensions of a cube in centimeters -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube with given dimensions -/
def cubeSurfaceArea (d : CubeDimensions) : ℝ :=
  6 * d.length * d.width

/-- Calculates the surface area of a cube with corners removed -/
def cubeWithCornersRemovedSurfaceArea (originalCube : CubeDimensions) (removedCorner : CubeDimensions) : ℝ :=
  cubeSurfaceArea originalCube

/-- Theorem stating that the surface area of a 4x4x4 cube with 1x1x1 corners removed is 96 cm² -/
theorem surface_area_cube_with_corners_removed :
  let originalCube : CubeDimensions := ⟨4, 4, 4⟩
  let removedCorner : CubeDimensions := ⟨1, 1, 1⟩
  cubeWithCornersRemovedSurfaceArea originalCube removedCorner = 96 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_cube_with_corners_removed_l2926_292604


namespace NUMINAMATH_CALUDE_sum_distinct_digits_mod_1000_l2926_292667

/-- The sum of all four-digit positive integers with distinct digits -/
def T : ℕ := sorry

/-- Predicate to check if a number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

theorem sum_distinct_digits_mod_1000 : 
  T % 1000 = 400 :=
sorry

end NUMINAMATH_CALUDE_sum_distinct_digits_mod_1000_l2926_292667


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_l2926_292643

theorem magnitude_of_complex_square : Complex.abs ((3 - 4*Complex.I)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_l2926_292643


namespace NUMINAMATH_CALUDE_rent_is_1000_l2926_292612

/-- Calculates the rent given salary, remaining amount, and the relationship between rent and other expenses. -/
def calculate_rent (salary : ℕ) (remaining : ℕ) : ℕ :=
  let total_expenses := salary - remaining
  total_expenses / 3

/-- Proves that the rent is $1000 given the conditions -/
theorem rent_is_1000 (salary : ℕ) (remaining : ℕ) 
  (h1 : salary = 5000)
  (h2 : remaining = 2000)
  (h3 : calculate_rent salary remaining = 1000) : 
  calculate_rent salary remaining = 1000 := by
  sorry

#eval calculate_rent 5000 2000

end NUMINAMATH_CALUDE_rent_is_1000_l2926_292612


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2926_292657

/-- Given a pentagon formed by placing an equilateral triangle atop a square,
    where the side length of the square equals the height of the triangle,
    prove that the ratio of the triangle's area to the pentagon's area is (3(√3 - 1))/6 -/
theorem triangle_to_pentagon_area_ratio :
  ∀ s : ℝ, s > 0 →
  let h := s * (Real.sqrt 3 / 2)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  let square_area := h^2
  let pentagon_area := triangle_area + square_area
  triangle_area / pentagon_area = (3 * (Real.sqrt 3 - 1)) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2926_292657


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_2100210021_base_3_l2926_292607

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem largest_prime_divisor_of_2100210021_base_3 :
  let n := base_3_to_decimal [1, 2, 0, 0, 1, 2, 0, 0, 1, 2]
  ∃ (p : Nat), is_prime p ∧ p ∣ n ∧ ∀ (q : Nat), is_prime q → q ∣ n → q ≤ p ∧ p = 46501 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_2100210021_base_3_l2926_292607


namespace NUMINAMATH_CALUDE_round_trip_speed_l2926_292692

/-- Proves that given a person's average speed for a round trip is 75 km/hr,
    and the return speed is 50% faster than the initial speed,
    the initial speed is 62.5 km/hr. -/
theorem round_trip_speed (v : ℝ) : 
  (v > 0) →                           -- Initial speed is positive
  (2 / (1 / v + 1 / (1.5 * v)) = 75)  -- Average speed is 75 km/hr
  → v = 62.5 := by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l2926_292692


namespace NUMINAMATH_CALUDE_area_ratio_incenter_centroids_l2926_292669

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_incenter_centroids 
  (ABC : Triangle) 
  (P : ℝ × ℝ) 
  (G₁ G₂ G₃ : ℝ × ℝ) :
  P = incenter ABC →
  G₁ = centroid (Triangle.mk P ABC.B ABC.C) →
  G₂ = centroid (Triangle.mk ABC.A P ABC.C) →
  G₃ = centroid (Triangle.mk ABC.A ABC.B P) →
  area (Triangle.mk G₁ G₂ G₃) = (1 / 9) * area ABC :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_incenter_centroids_l2926_292669


namespace NUMINAMATH_CALUDE_percentage_problem_l2926_292650

theorem percentage_problem (X : ℝ) (h : 0.2 * X = 300) : 
  ∃ P : ℝ, (P / 100) * X = 1800 ∧ P = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2926_292650


namespace NUMINAMATH_CALUDE_equation_solution_l2926_292679

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (6*x)^5 = (18*x)^4 ∧ x = 27/2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2926_292679


namespace NUMINAMATH_CALUDE_existence_of_divisible_difference_l2926_292638

theorem existence_of_divisible_difference (x : Fin 2022 → ℤ) :
  ∃ i j : Fin 2022, i ≠ j ∧ (2021 : ℤ) ∣ (x j - x i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_difference_l2926_292638


namespace NUMINAMATH_CALUDE_equation_solution_l2926_292640

theorem equation_solution (k : ℝ) : 
  (7 * (-1)^3 - 3 * (-1)^2 + k * (-1) + 5 = 0) → 
  (k^3 + 2 * k^2 - 11 * k - 85 = -105) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2926_292640


namespace NUMINAMATH_CALUDE_choir_size_proof_l2926_292670

theorem choir_size_proof : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 ∧ 
  n % 10 = 0 ∧ 
  n % 11 = 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) ∧
  n = 1080 :=
by sorry

end NUMINAMATH_CALUDE_choir_size_proof_l2926_292670


namespace NUMINAMATH_CALUDE_f_sum_negative_l2926_292619

/-- The function f satisfying the given conditions -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

/-- The theorem statement -/
theorem f_sum_negative (m : ℝ) (a b : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) < 0) →
  a < 0 →
  0 < b →
  abs a < abs b →
  f m a + f m b < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2926_292619


namespace NUMINAMATH_CALUDE_original_number_proof_l2926_292678

theorem original_number_proof (x : ℝ) : 3 * (2 * x^2 + 15) - 7 = 91 → x = Real.sqrt (53 / 6) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2926_292678


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2926_292652

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (h_right_angle : X^2 + Y^2 = Z^2)  -- Y is the right angle
  (h_cos : Real.cos X = 3/5)
  (h_hypotenuse : Z = 10) :
  Y = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2926_292652


namespace NUMINAMATH_CALUDE_prob_three_common_books_l2926_292688

/-- The number of books in Mr. Johnson's list -/
def total_books : ℕ := 12

/-- The number of books each student must choose -/
def books_to_choose : ℕ := 5

/-- The number of common books we're interested in -/
def common_books : ℕ := 3

/-- The probability of Alice and Bob selecting exactly 3 common books -/
def prob_common_books : ℚ := 55 / 209

theorem prob_three_common_books :
  (Nat.choose total_books common_books *
   Nat.choose (total_books - common_books) (books_to_choose - common_books) *
   Nat.choose (total_books - books_to_choose) (books_to_choose - common_books)) /
  (Nat.choose total_books books_to_choose)^2 = prob_common_books :=
sorry

end NUMINAMATH_CALUDE_prob_three_common_books_l2926_292688


namespace NUMINAMATH_CALUDE_complex_modulus_l2926_292646

theorem complex_modulus (x y : ℝ) (h : (1 + Complex.I) * x = 1 - Complex.I * y) :
  Complex.abs (x - Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2926_292646


namespace NUMINAMATH_CALUDE_coefficient_x4_is_1120_l2926_292662

open BigOperators

/-- The coefficient of x^4 in the expansion of (x^2 + 2/x)^8 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 8 4) * 2^4

/-- Theorem stating that the coefficient of x^4 in (x^2 + 2/x)^8 is 1120 -/
theorem coefficient_x4_is_1120 : coefficient_x4 = 1120 := by
  sorry

#eval coefficient_x4  -- This will evaluate the expression and show the result

end NUMINAMATH_CALUDE_coefficient_x4_is_1120_l2926_292662


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l2926_292630

def is_nonzero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def number_to_digits (n : ℕ) : ℕ := 
  if n = 0 then 1 else (Nat.log 10 n).succ

theorem sum_has_five_digits (A B : ℕ) 
  (hA : is_nonzero_digit A) (hB : is_nonzero_digit B) : 
  number_to_digits (19876 + (10000 * A + 1000 * B + 320) + (200 * B + 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l2926_292630


namespace NUMINAMATH_CALUDE_initial_kittens_count_l2926_292647

theorem initial_kittens_count (kittens_to_jessica kittens_to_sara kittens_left : ℕ) :
  kittens_to_jessica = 3 →
  kittens_to_sara = 6 →
  kittens_left = 9 →
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l2926_292647


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2926_292681

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the quadratic equation
def quadraticEquation (x k : ℤ) : Prop := x^2 - 99*x + k = 0

-- Define a function to check if both roots are prime
def bothRootsPrime (k : ℤ) : Prop :=
  ∃ p q : ℤ, 
    quadraticEquation p k ∧ 
    quadraticEquation q k ∧ 
    p ≠ q ∧ 
    isPrime p.natAbs ∧ 
    isPrime q.natAbs

-- Theorem statement
theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, bothRootsPrime k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2926_292681


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2926_292694

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 - 2*x^3 + 3*x + 1) % (x - 2) = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2926_292694


namespace NUMINAMATH_CALUDE_relationship_abc_l2926_292620

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sin (145 * π / 180) →
  b = Real.cos (52 * π / 180) →
  c = Real.tan (47 * π / 180) →
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2926_292620


namespace NUMINAMATH_CALUDE_solution_exists_l2926_292660

theorem solution_exists : ∃ M : ℤ, (14 : ℤ)^2 * (35 : ℤ)^2 = (10 : ℤ)^2 * (M - 10)^2 := by
  use 59
  sorry

#check solution_exists

end NUMINAMATH_CALUDE_solution_exists_l2926_292660


namespace NUMINAMATH_CALUDE_expression_evaluation_l2926_292622

theorem expression_evaluation : (3 / 2) * 12 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2926_292622


namespace NUMINAMATH_CALUDE_black_triangles_2008_l2926_292641

/-- Given a sequence of triangles in the pattern ▲▲△△▲△, 
    this function returns the number of black triangles in n triangles -/
def black_triangles (n : ℕ) : ℕ :=
  (n - n % 6) / 2 + min 2 (n % 6)

/-- Theorem: In a sequence of 2008 triangles following the pattern ▲▲△△▲△,
    there are 1004 black triangles -/
theorem black_triangles_2008 : black_triangles 2008 = 1004 := by
  sorry

end NUMINAMATH_CALUDE_black_triangles_2008_l2926_292641


namespace NUMINAMATH_CALUDE_floor_sqrt_116_l2926_292671

theorem floor_sqrt_116 : ⌊Real.sqrt 116⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_116_l2926_292671


namespace NUMINAMATH_CALUDE_train_length_l2926_292693

/-- Given a train that crosses a 300-meter platform in 36 seconds and a signal pole in 18 seconds, 
    prove that the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : platform_length = 300)
    (h2 : platform_time = 36)
    (h3 : pole_time = 18) : 
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2926_292693


namespace NUMINAMATH_CALUDE_fraction_subtraction_proof_l2926_292649

theorem fraction_subtraction_proof : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_proof_l2926_292649


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_l2926_292613

/-- Calculates the transport cost given the purchase details of a refrigerator --/
theorem refrigerator_transport_cost 
  (purchase_price_after_discount : ℕ)
  (discount_rate : ℚ)
  (installation_cost : ℕ)
  (selling_price_for_profit : ℕ) :
  purchase_price_after_discount = 12500 →
  discount_rate = 1/5 →
  installation_cost = 250 →
  selling_price_for_profit = 18560 →
  (purchase_price_after_discount / (1 - discount_rate) * (1 + 4/25) : ℚ) = selling_price_for_profit →
  (selling_price_for_profit : ℚ) - purchase_price_after_discount - installation_cost = 5810 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_transport_cost_l2926_292613


namespace NUMINAMATH_CALUDE_det_A_l2926_292602

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 5, 6, -4; 3, 3, 7]

theorem det_A : Matrix.det A = 168 := by sorry

end NUMINAMATH_CALUDE_det_A_l2926_292602


namespace NUMINAMATH_CALUDE_simplify_expression_l2926_292605

variable (x y : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2926_292605


namespace NUMINAMATH_CALUDE_kelly_found_games_l2926_292672

def initial_games : ℕ := 80
def games_to_give_away : ℕ := 105
def games_left : ℕ := 6

theorem kelly_found_games : 
  ∃ (found_games : ℕ), 
    initial_games + found_games = games_to_give_away + games_left ∧ 
    found_games = 31 :=
by sorry

end NUMINAMATH_CALUDE_kelly_found_games_l2926_292672


namespace NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_achieved_l2926_292680

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 47 →
  a * d + b * c = 88 →
  c * d = 54 →
  a^2 + b^2 + c^2 + d^2 ≤ 254 := by
  sorry

theorem max_sum_of_squares_achieved (a b c d : ℝ) : 
  ∃ (a b c d : ℝ),
    a + b = 12 ∧
    a * b + c + d = 47 ∧
    a * d + b * c = 88 ∧
    c * d = 54 ∧
    a^2 + b^2 + c^2 + d^2 = 254 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_achieved_l2926_292680


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2926_292648

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (x/2) * Real.cos (3*x/2) - Real.sin x * Real.sin (3*x) - Real.sin (2*x) * Real.sin (3*x) = 0) →
  ∃ k : ℤ, x = π/9 * (2*k + 1) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2926_292648


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2926_292698

/-- A circle that passes through a fixed point and is tangent to a line -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 1)^2 = (center.2 + 1)^2

/-- The trajectory of the center of the moving circle -/
def trajectory (c : MovingCircle) : Prop :=
  c.center.1^2 = 4 * c.center.2

/-- Theorem stating that the trajectory of the center is x^2 = 4y -/
theorem trajectory_is_parabola (c : MovingCircle) : trajectory c := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2926_292698


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2926_292629

theorem product_of_three_numbers (x y z : ℚ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 23625 / 686 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2926_292629


namespace NUMINAMATH_CALUDE_derivative_problems_l2926_292632

open Real

theorem derivative_problems :
  (∀ x > 0, deriv (λ x => (log x) / x) x = (1 - log x) / x^2) ∧
  (∀ x, deriv (λ x => x * exp x) x = (x + 1) * exp x) ∧
  (∀ x, deriv (λ x => cos (2 * x)) x = -2 * sin (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_problems_l2926_292632


namespace NUMINAMATH_CALUDE_largest_value_l2926_292624

def expr_a : ℕ := 3 + 1 + 2 + 8
def expr_b : ℕ := 3 * 1 + 2 + 8
def expr_c : ℕ := 3 + 1 * 2 + 8
def expr_d : ℕ := 3 + 1 + 2 * 8
def expr_e : ℕ := 3 * 1 * 2 * 8

theorem largest_value :
  expr_e ≥ expr_a ∧ 
  expr_e ≥ expr_b ∧ 
  expr_e ≥ expr_c ∧ 
  expr_e ≥ expr_d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2926_292624


namespace NUMINAMATH_CALUDE_brookes_cows_solution_l2926_292617

/-- Represents the problem of determining the number of cows Brooke has --/
def brookes_cows (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) (total_earnings : ℚ) : Prop :=
  milk_price = 3 ∧
  butter_conversion = 2 ∧
  butter_price = 3/2 ∧
  milk_per_cow = 4 ∧
  num_customers = 6 ∧
  milk_per_customer = 6 ∧
  total_earnings = 144 ∧
  ∃ (num_cows : ℕ), 
    (↑num_cows : ℚ) * milk_per_cow = 
      (↑num_customers * milk_per_customer) + 
      ((total_earnings - (↑num_customers * milk_per_customer * milk_price)) / butter_price * (1 / butter_conversion))

theorem brookes_cows_solution :
  ∀ (milk_price butter_conversion butter_price milk_per_cow : ℚ)
    (num_customers : ℕ) (milk_per_customer total_earnings : ℚ),
  brookes_cows milk_price butter_conversion butter_price milk_per_cow num_customers milk_per_customer total_earnings →
  ∃ (num_cows : ℕ), num_cows = 12 :=
by sorry

end NUMINAMATH_CALUDE_brookes_cows_solution_l2926_292617


namespace NUMINAMATH_CALUDE_cinematic_academy_members_l2926_292691

/-- The minimum fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_fraction : ℚ := 1 / 4

/-- The smallest number of top-10 lists a film can appear on and still be considered -/
def min_lists : ℚ := 198.75

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 795

/-- Theorem stating that the number of members in the Cinematic Academy is 795 -/
theorem cinematic_academy_members :
  academy_members = ⌈(min_lists / min_fraction : ℚ)⌉ := by
  sorry

end NUMINAMATH_CALUDE_cinematic_academy_members_l2926_292691


namespace NUMINAMATH_CALUDE_triangle_properties_l2926_292687

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = (2 * Real.sqrt 3 / 3) * t.b * Real.sin (t.A + π/3))
  (h2 : t.a + t.c = 4) :
  t.B = π/3 ∧ 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2926_292687


namespace NUMINAMATH_CALUDE_total_combinations_is_twelve_l2926_292621

/-- Represents the number of compulsory subjects -/
def compulsory_subjects : ℕ := 3

/-- Represents the number of subjects to choose from for the first choice -/
def first_choice_subjects : ℕ := 2

/-- Represents the number of subjects to choose from for the second choice -/
def second_choice_subjects : ℕ := 4

/-- Represents the number of subjects to select in the second choice -/
def second_choice_select : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Theorem stating that the total number of combinations is 12 -/
theorem total_combinations_is_twelve :
  1 * first_choice_subjects * (choose second_choice_subjects second_choice_select) = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_is_twelve_l2926_292621


namespace NUMINAMATH_CALUDE_amount_owed_after_one_year_l2926_292666

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the total amount owed after one year --/
theorem amount_owed_after_one_year :
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  total_amount_owed principal rate time = 56.70 := by
sorry

end NUMINAMATH_CALUDE_amount_owed_after_one_year_l2926_292666


namespace NUMINAMATH_CALUDE_y_range_l2926_292674

theorem y_range (y : ℝ) (h1 : y > 0) (h2 : Real.log y / Real.log 3 ≤ 3 - Real.log (9 * y) / Real.log 3) : 
  0 < y ∧ y ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_range_l2926_292674


namespace NUMINAMATH_CALUDE_right_triangle_proof_l2926_292668

-- Define a structure for a triangle with three angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define the theorem
theorem right_triangle_proof (t : Triangle) (h : t.γ = t.α + t.β) : 
  t.α = 90 ∨ t.β = 90 ∨ t.γ = 90 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_proof_l2926_292668


namespace NUMINAMATH_CALUDE_gcd_299_667_l2926_292600

theorem gcd_299_667 : Nat.gcd 299 667 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_299_667_l2926_292600


namespace NUMINAMATH_CALUDE_min_button_presses_correct_l2926_292664

/-- Represents the time difference in minutes between the correct time and the displayed time -/
def time_difference : ℤ := 13

/-- Represents the increase in minutes when the first button is pressed -/
def button1_adjustment : ℤ := 9

/-- Represents the decrease in minutes when the second button is pressed -/
def button2_adjustment : ℤ := 20

/-- Represents the equation for adjusting the clock -/
def clock_adjustment (a b : ℤ) : Prop :=
  button1_adjustment * a - button2_adjustment * b = time_difference

/-- The minimum number of button presses required -/
def min_button_presses : ℕ := 24

/-- Theorem stating that the minimum number of button presses to correctly set the clock is 24 -/
theorem min_button_presses_correct :
  ∃ (a b : ℤ), clock_adjustment a b ∧ a ≥ 0 ∧ b ≥ 0 ∧ a + b = min_button_presses ∧
  (∀ (c d : ℤ), clock_adjustment c d → c ≥ 0 → d ≥ 0 → c + d ≥ min_button_presses) :=
by sorry

end NUMINAMATH_CALUDE_min_button_presses_correct_l2926_292664


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2926_292603

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2926_292603


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2926_292661

/-- Represents a workshop in the factory -/
inductive Workshop
| A
| B
| C

/-- The production quantity for each workshop -/
def production : Workshop → ℕ
| Workshop.A => 120
| Workshop.B => 80
| Workshop.C => 60

/-- The number of pieces sampled from workshop C -/
def sampledFromC : ℕ := 3

/-- The total production across all workshops -/
def totalProduction : ℕ := production Workshop.A + production Workshop.B + production Workshop.C

/-- The sampling fraction based on workshop C -/
def samplingFraction : ℚ := sampledFromC / production Workshop.C

theorem stratified_sample_size :
  ∃ n : ℕ, n = (samplingFraction * totalProduction).num ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2926_292661


namespace NUMINAMATH_CALUDE_subset_with_fourth_power_product_l2926_292697

theorem subset_with_fourth_power_product 
  (M : Finset ℕ+) 
  (distinct : M.card = 1985) 
  (prime_bound : ∀ n ∈ M, ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 23) :
  ∃ (a b c d : ℕ+), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ+), a * b * c * d = m ^ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_with_fourth_power_product_l2926_292697
