import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_l1847_184743

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 500 / 3 → 
  area = side^2 → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1847_184743


namespace NUMINAMATH_CALUDE_tenth_stage_toothpicks_l1847_184712

/-- The number of toothpicks in the nth stage of the sequence -/
def toothpicks (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 10th stage of the sequence has 31 toothpicks -/
theorem tenth_stage_toothpicks : toothpicks 10 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tenth_stage_toothpicks_l1847_184712


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1847_184716

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24) ∧ (a ≠ b) ∧ (a.val + b.val = 96) ∧ (∀ (c d : ℕ+), ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24) → (c ≠ d) → (c.val + d.val ≥ 96)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1847_184716


namespace NUMINAMATH_CALUDE_initial_money_proof_l1847_184725

/-- The amount of money Mrs. Hilt had initially -/
def initial_money : ℕ := 15

/-- The cost of the pencil in cents -/
def pencil_cost : ℕ := 11

/-- The amount of money left after buying the pencil -/
def money_left : ℕ := 4

/-- Theorem stating that the initial money equals the sum of the pencil cost and money left -/
theorem initial_money_proof : initial_money = pencil_cost + money_left := by
  sorry

end NUMINAMATH_CALUDE_initial_money_proof_l1847_184725


namespace NUMINAMATH_CALUDE_geometric_mean_scaling_l1847_184726

theorem geometric_mean_scaling (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) 
  (h₅ : a₅ > 0) (h₆ : a₆ > 0) (h₇ : a₇ > 0) (h₈ : a₈ > 0) :
  (((5 * a₁) * (5 * a₂) * (5 * a₃) * (5 * a₄) * (5 * a₅) * (5 * a₆) * (5 * a₇) * (5 * a₈)) ^ (1/8 : ℝ)) = 
  5 * ((a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ * a₈) ^ (1/8 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_scaling_l1847_184726


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1847_184707

theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let circle_radius : ℝ := r
  let triangle_side_length : ℝ := r * Real.sqrt 3
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length ^ 2
  triangle_area = 75 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1847_184707


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l1847_184721

/-- Given a bag with m red balls and n white balls, where m ≥ n ≥ 2, prove that if the probability
    of drawing two red balls is an integer multiple of the probability of drawing one red and one
    white ball, then m must be odd. Also, find all pairs (m, n) such that m + n ≤ 40 and the
    probability of drawing two balls of the same color equals the probability of drawing two balls
    of different colors. -/
theorem ball_probability_theorem (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 2) :
  (∃ k : ℕ, Nat.choose m 2 * (Nat.choose (m + n) 2) = k * m * n * (Nat.choose (m + n) 2)) →
  Odd m ∧
  (m + n ≤ 40 →
    Nat.choose m 2 + Nat.choose n 2 = m * n →
    ∃ (p q : ℕ), p = m ∧ q = n) :=
by sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l1847_184721


namespace NUMINAMATH_CALUDE_work_completion_time_l1847_184792

theorem work_completion_time 
  (total_men : ℕ) 
  (initial_days : ℕ) 
  (absent_men : ℕ) 
  (h1 : total_men = 15)
  (h2 : initial_days = 8)
  (h3 : absent_men = 3) : 
  (total_men * initial_days) / (total_men - absent_men) = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1847_184792


namespace NUMINAMATH_CALUDE_sequence_sum_l1847_184788

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = n^3) :
  a 6 + a 7 + a 8 + a 9 = 604 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1847_184788


namespace NUMINAMATH_CALUDE_r_lower_bound_l1847_184783

theorem r_lower_bound (a b c d : ℕ+) (r : ℚ) :
  r = 1 - (a : ℚ) / b - (c : ℚ) / d →
  a + c ≤ 1982 →
  r ≥ 0 →
  r > 1 / (1983 : ℚ)^3 := by
  sorry

end NUMINAMATH_CALUDE_r_lower_bound_l1847_184783


namespace NUMINAMATH_CALUDE_omino_tilings_2_by_10_l1847_184719

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of omino tilings for a 1-by-n rectangle -/
def ominoTilings1ByN (n : ℕ) : ℕ := fib (n + 1)

/-- Number of omino tilings for a 2-by-n rectangle -/
def ominoTilings2ByN (n : ℕ) : ℕ := (ominoTilings1ByN n) ^ 2

theorem omino_tilings_2_by_10 : ominoTilings2ByN 10 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_omino_tilings_2_by_10_l1847_184719


namespace NUMINAMATH_CALUDE_circle_polygons_l1847_184766

theorem circle_polygons (n : ℕ) (h : n = 15) :
  let quadrilaterals := Nat.choose n 4
  let triangles := Nat.choose n 3
  quadrilaterals + triangles = 1820 := by
  sorry

end NUMINAMATH_CALUDE_circle_polygons_l1847_184766


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1847_184720

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1847_184720


namespace NUMINAMATH_CALUDE_tic_tac_toe_4x4_carl_wins_l1847_184724

/-- Represents a 4x4 tic-tac-toe board --/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a winning line on the board --/
structure WinningLine :=
  (positions : List (Fin 4 × Fin 4))
  (is_valid : positions.length = 4)

/-- All possible winning lines on a 4x4 board --/
def winningLines : List WinningLine := sorry

/-- Checks if a given board configuration is valid --/
def isValidBoard (b : Board) : Prop := sorry

/-- Checks if Carl wins with exactly 4 O's --/
def carlWinsWithFourO (b : Board) : Prop := sorry

/-- The number of ways Carl can win with exactly 4 O's --/
def numWaysToWin : ℕ := sorry

theorem tic_tac_toe_4x4_carl_wins :
  numWaysToWin = 4950 := by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_4x4_carl_wins_l1847_184724


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1847_184713

/-- Proves that for a parabola y = x^2 + bx + c passing through (1, -1) and (3, 9), c = -3 -/
theorem parabola_coefficient (b c : ℝ) : 
  (1^2 + b*1 + c = -1) → 
  (3^2 + b*3 + c = 9) → 
  c = -3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1847_184713


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1847_184734

/-- Given an ellipse with equation x^2 + 9y^2 = 8100, 
    the distance between its foci is 120√2 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, x^2 + 9*y^2 = 8100 → x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2*c = 120*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1847_184734


namespace NUMINAMATH_CALUDE_total_charts_brought_l1847_184751

/-- Represents the number of associate professors -/
def associate_profs : ℕ := 2

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := 7

/-- Represents the total number of people present -/
def total_people : ℕ := 9

/-- Represents the total number of pencils brought -/
def total_pencils : ℕ := 11

/-- Represents the number of pencils each associate professor brings -/
def pencils_per_associate : ℕ := 2

/-- Represents the number of pencils each assistant professor brings -/
def pencils_per_assistant : ℕ := 1

/-- Represents the number of charts each associate professor brings -/
def charts_per_associate : ℕ := 1

/-- Represents the number of charts each assistant professor brings -/
def charts_per_assistant : ℕ := 2

theorem total_charts_brought : 
  associate_profs * charts_per_associate + assistant_profs * charts_per_assistant = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_charts_brought_l1847_184751


namespace NUMINAMATH_CALUDE_pen_probabilities_l1847_184701

/-- The number of pens in the box -/
def total_pens : ℕ := 6

/-- The number of first-class pens -/
def first_class_pens : ℕ := 4

/-- The number of second-class pens -/
def second_class_pens : ℕ := 2

/-- The number of pens drawn -/
def drawn_pens : ℕ := 2

/-- The probability of drawing exactly one first-class pen -/
def prob_one_first_class : ℚ := 8 / 15

/-- The probability of drawing at least one second-class pen -/
def prob_second_class : ℚ := 3 / 5

theorem pen_probabilities :
  (total_pens = first_class_pens + second_class_pens) →
  (prob_one_first_class = (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1 : ℚ) / Nat.choose total_pens drawn_pens) ∧
  (prob_second_class = 1 - (Nat.choose first_class_pens drawn_pens : ℚ) / Nat.choose total_pens drawn_pens) :=
by sorry

end NUMINAMATH_CALUDE_pen_probabilities_l1847_184701


namespace NUMINAMATH_CALUDE_mama_bird_worms_l1847_184765

/-- The number of additional worms Mama bird needs to catch -/
def additional_worms_needed (num_babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (papa_worms : ℕ) (mama_worms : ℕ) (stolen_worms : ℕ) : ℕ :=
  num_babies * worms_per_baby_per_day * days - (papa_worms + mama_worms - stolen_worms)

/-- Theorem stating that Mama bird needs to catch 34 more worms -/
theorem mama_bird_worms : 
  additional_worms_needed 6 3 3 9 13 2 = 34 := by sorry

end NUMINAMATH_CALUDE_mama_bird_worms_l1847_184765


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_2_pow_18_minus_1_l1847_184796

theorem no_two_digit_factors_of_2_pow_18_minus_1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ¬(2^18 - 1) % n = 0 := by sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_2_pow_18_minus_1_l1847_184796


namespace NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l1847_184753

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 5 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the Fibonacci series divided by powers of 5 equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l1847_184753


namespace NUMINAMATH_CALUDE_average_equation_l1847_184778

theorem average_equation (a : ℝ) : ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l1847_184778


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l1847_184784

theorem quadratic_expression_equality (a b : ℝ) : 
  ((-11 * -8)^(3/2) + 5 * Real.sqrt 16) * ((a - 2) + (b + 3)) = 
  ((176 * Real.sqrt 22) + 20) * (a + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l1847_184784


namespace NUMINAMATH_CALUDE_skating_time_calculation_l1847_184756

theorem skating_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 150 →
  speed = 12 →
  time = distance / speed →
  time = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_skating_time_calculation_l1847_184756


namespace NUMINAMATH_CALUDE_bales_stacked_l1847_184762

theorem bales_stacked (initial_bales current_bales : ℕ) 
  (h1 : initial_bales = 54)
  (h2 : current_bales = 82) :
  current_bales - initial_bales = 28 := by
  sorry

end NUMINAMATH_CALUDE_bales_stacked_l1847_184762


namespace NUMINAMATH_CALUDE_total_eggs_collected_l1847_184718

def benjamin_eggs : ℕ := 6

def carla_eggs : ℕ := 3 * benjamin_eggs

def trisha_eggs : ℕ := benjamin_eggs - 4

def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l1847_184718


namespace NUMINAMATH_CALUDE_march_text_messages_l1847_184708

/-- Represents the number of text messages sent in the nth month -/
def T (n : ℕ) : ℕ := n^3 - n^2 + n

/-- Theorem stating that the number of text messages in the 5th month (March) is 105 -/
theorem march_text_messages : T 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_march_text_messages_l1847_184708


namespace NUMINAMATH_CALUDE_work_completion_time_l1847_184777

/-- Given:
  - A can finish the work in 6 days
  - B worked for 10 days and left the job
  - A alone can finish the remaining work in 2 days
  Prove that B can finish the work in 15 days -/
theorem work_completion_time
  (total_work : ℝ)
  (a_completion_time : ℝ)
  (b_work_days : ℝ)
  (a_remaining_time : ℝ)
  (h1 : a_completion_time = 6)
  (h2 : b_work_days = 10)
  (h3 : a_remaining_time = 2)
  (h4 : total_work > 0) :
  ∃ (b_completion_time : ℝ),
    b_completion_time = 15 ∧
    (total_work / a_completion_time) * a_remaining_time =
      total_work - (total_work / b_completion_time) * b_work_days :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l1847_184777


namespace NUMINAMATH_CALUDE_book_sale_profit_l1847_184763

theorem book_sale_profit (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage1 : ℝ) (gain_percentage2 : ℝ) 
  (h1 : total_cost = 1080)
  (h2 : loss_percentage = 0.1)
  (h3 : gain_percentage1 = 0.15)
  (h4 : gain_percentage2 = 0.25)
  (h5 : (1 - loss_percentage) * (total_cost / 2) = 
        (1 + gain_percentage1) * (total_cost * 2 / 6) + 
        (1 + gain_percentage2) * (total_cost / 6)) :
  total_cost / 2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l1847_184763


namespace NUMINAMATH_CALUDE_dave_trips_l1847_184776

/-- The number of trays Dave can carry at a time -/
def trays_per_trip : ℕ := 12

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 26

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 49

/-- The number of trays on the third table -/
def trays_table3 : ℕ := 65

/-- The number of trays on the fourth table -/
def trays_table4 : ℕ := 38

/-- The total number of trays Dave needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2 + trays_table3 + trays_table4

/-- The minimum number of trips Dave needs to make -/
def min_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem dave_trips : min_trips = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_trips_l1847_184776


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1847_184790

-- Problem 1
def problem1 (a : ℚ) : ℚ := a * (a - 4) - (a + 6) * (a - 2)

theorem problem1_solution :
  problem1 (-1/2) = 16 := by sorry

-- Problem 2
def problem2 (x y : ℤ) : ℤ := (x + 2*y) * (x - 2*y) - (2*x - y) * (-2*x - y)

theorem problem2_solution :
  problem2 8 (-8) = 0 := by sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1847_184790


namespace NUMINAMATH_CALUDE_intersection_point_l1847_184744

-- Define the two linear functions
def f (x : ℝ) (m : ℝ) : ℝ := x + m
def g (x : ℝ) : ℝ := 2 * x - 2

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ y : ℝ, f 0 m = y ∧ g 0 = y) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1847_184744


namespace NUMINAMATH_CALUDE_problem_solving_probability_l1847_184794

theorem problem_solving_probability (p_A p_B : ℝ) (h_A : p_A = 1/5) (h_B : p_B = 1/3) :
  1 - (1 - p_A) * (1 - p_B) = 7/15 :=
by sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l1847_184794


namespace NUMINAMATH_CALUDE_at_least_one_first_class_l1847_184780

theorem at_least_one_first_class (n m k : ℕ) (h1 : n = 20) (h2 : m = 16) (h3 : k = 3) :
  (Nat.choose m 1 * Nat.choose (n - m) 2) +
  (Nat.choose m 2 * Nat.choose (n - m) 1) +
  (Nat.choose m 3) = 1136 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_first_class_l1847_184780


namespace NUMINAMATH_CALUDE_map_distance_between_mountains_l1847_184799

/-- Given a map with a known scale factor, this theorem proves that the distance
    between two mountains on the map is 312 inches, given their actual distance
    and a reference point. -/
theorem map_distance_between_mountains
  (actual_distance : ℝ)
  (map_reference : ℝ)
  (actual_reference : ℝ)
  (h1 : actual_distance = 136)
  (h2 : map_reference = 28)
  (h3 : actual_reference = 12.205128205128204)
  : (actual_distance / (actual_reference / map_reference)) = 312 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_between_mountains_l1847_184799


namespace NUMINAMATH_CALUDE_sqrt_one_minus_x_domain_l1847_184741

theorem sqrt_one_minus_x_domain : ∀ x : ℝ, 
  (x ≤ 1 ↔ ∃ y : ℝ, y ^ 2 = 1 - x) ∧
  (x = 2 → ¬∃ y : ℝ, y ^ 2 = 1 - x) :=
sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_x_domain_l1847_184741


namespace NUMINAMATH_CALUDE_triangle_inequality_with_altitudes_l1847_184700

/-- Given a triangle with sides a > b and corresponding altitudes h_a and h_b,
    prove that a + h_a ≥ b + h_b with equality iff the angle between a and b is 90° -/
theorem triangle_inequality_with_altitudes (a b h_a h_b : ℝ) (S : ℝ) (γ : ℝ) :
  a > b →
  S = (1/2) * a * h_a →
  S = (1/2) * b * h_b →
  S = (1/2) * a * b * Real.sin γ →
  (a + h_a ≥ b + h_b) ∧ (a + h_a = b + h_b ↔ γ = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_altitudes_l1847_184700


namespace NUMINAMATH_CALUDE_smallest_number_l1847_184759

-- Define the numbers
def A : ℝ := 5.67823
def B : ℝ := 5.678333333 -- Approximation of 5.678̅3
def C : ℝ := 5.678383838 -- Approximation of 5.67̅83
def D : ℝ := 5.678378378 -- Approximation of 5.6̅783
def E : ℝ := 5.678367836 -- Approximation of 5.̅6783

-- Theorem statement
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1847_184759


namespace NUMINAMATH_CALUDE_rainfall_2011_l1847_184793

/-- The total rainfall in Rainville for 2011, given the average monthly rainfall in 2010 and the increase in 2011. -/
def total_rainfall_2011 (avg_2010 : ℝ) (increase : ℝ) : ℝ :=
  (avg_2010 + increase) * 12

/-- Theorem stating that the total rainfall in Rainville for 2011 was 483.6 mm. -/
theorem rainfall_2011 : total_rainfall_2011 36.8 3.5 = 483.6 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_2011_l1847_184793


namespace NUMINAMATH_CALUDE_animal_rescue_donation_l1847_184781

/-- Represents the daily earnings and costs for a 5-day bake sale --/
structure BakeSaleData :=
  (earnings : Fin 5 → ℕ)
  (costs : Fin 5 → ℕ)

/-- Represents the distribution percentages for charities --/
structure CharityDistribution :=
  (homeless_shelter : ℚ)
  (food_bank : ℚ)
  (park_restoration : ℚ)
  (animal_rescue : ℚ)

/-- Calculates the total donation to the Animal Rescue Center --/
def calculateAnimalRescueDonation (data : BakeSaleData) (dist : CharityDistribution) (personal_contribution : ℕ) : ℚ :=
  sorry

/-- Theorem stating the total donation to the Animal Rescue Center --/
theorem animal_rescue_donation
  (data : BakeSaleData)
  (dist : CharityDistribution)
  (h1 : data.earnings = ![450, 550, 400, 600, 500])
  (h2 : data.costs = ![80, 100, 70, 120, 90])
  (h3 : dist.homeless_shelter = 30 / 100)
  (h4 : dist.food_bank = 25 / 100)
  (h5 : dist.park_restoration = 20 / 100)
  (h6 : dist.animal_rescue = 25 / 100)
  (h7 : personal_contribution = 20) :
  calculateAnimalRescueDonation data dist personal_contribution = 535 := by
  sorry

end NUMINAMATH_CALUDE_animal_rescue_donation_l1847_184781


namespace NUMINAMATH_CALUDE_ralphs_cards_l1847_184752

/-- Given Ralph's initial and additional cards, prove the total number of cards. -/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) :
  initial_cards + additional_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_cards_l1847_184752


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1847_184729

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 3*x}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -2^x}

-- Define the symmetric difference operation
def symmetricDifference (X Y : Set ℝ) : Set ℝ := (X \ Y) ∪ (Y \ X)

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y : ℝ | y < -9/4 ∨ y ≥ 0} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1847_184729


namespace NUMINAMATH_CALUDE_diameter_scientific_notation_l1847_184731

-- Define the original diameter value
def original_diameter : ℝ := 0.000000103

-- Define the scientific notation components
def coefficient : ℝ := 1.03
def exponent : ℤ := -7

-- Theorem to prove the equality
theorem diameter_scientific_notation :
  original_diameter = coefficient * (10 : ℝ) ^ exponent :=
by
  sorry

end NUMINAMATH_CALUDE_diameter_scientific_notation_l1847_184731


namespace NUMINAMATH_CALUDE_smallest_block_volume_l1847_184702

theorem smallest_block_volume (N : ℕ) : 
  (∃ x y z : ℕ, 
    N = x * y * z ∧ 
    (x - 1) * (y - 1) * (z - 1) = 231 ∧
    ∀ a b c : ℕ, a * b * c = N → (a - 1) * (b - 1) * (c - 1) = 231 → 
      x * y * z ≤ a * b * c) → 
  N = 384 := by
sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l1847_184702


namespace NUMINAMATH_CALUDE_simplify_fraction_l1847_184714

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1847_184714


namespace NUMINAMATH_CALUDE_winning_scenarios_is_60_l1847_184757

/-- The number of different winning scenarios for a lottery ticket distribution -/
def winning_scenarios : ℕ :=
  let total_tickets : ℕ := 8
  let num_people : ℕ := 4
  let tickets_per_person : ℕ := 2
  let first_prize : ℕ := 1
  let second_prize : ℕ := 1
  let third_prize : ℕ := 1
  let non_winning_tickets : ℕ := 5

  -- The actual computation of winning scenarios
  60

/-- Theorem stating that the number of winning scenarios is 60 -/
theorem winning_scenarios_is_60 : winning_scenarios = 60 := by
  sorry

end NUMINAMATH_CALUDE_winning_scenarios_is_60_l1847_184757


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l1847_184732

open Set

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  (Bᶜ = {x | x ≤ 2 ∨ x ≥ 10}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l1847_184732


namespace NUMINAMATH_CALUDE_A_minus_2B_value_of_2B_minus_A_l1847_184771

/-- Given two expressions A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + a*b + 3*b

def B (a b : ℝ) : ℝ := a^2 - a*b + a

/-- Theorem stating the equality of A - 2B and its simplified form -/
theorem A_minus_2B (a b : ℝ) : A a b - 2 * B a b = 3*a*b + 3*b - 2*a := by sorry

/-- Theorem stating the value of 2B - A under the given condition -/
theorem value_of_2B_minus_A (a b : ℝ) (h : (a + 1)^2 + |b - 3| = 0) : 
  2 * B a b - A a b = -2 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_value_of_2B_minus_A_l1847_184771


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1847_184767

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 7 = 2 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1847_184767


namespace NUMINAMATH_CALUDE_raffle_tickets_sold_l1847_184748

/-- Given that a school sold $620 worth of raffle tickets at $4 per ticket,
    prove that the number of tickets sold is 155. -/
theorem raffle_tickets_sold (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ)
  (h1 : total_money = 620)
  (h2 : ticket_cost = 4)
  (h3 : total_money = ticket_cost * num_tickets) :
  num_tickets = 155 := by
  sorry

end NUMINAMATH_CALUDE_raffle_tickets_sold_l1847_184748


namespace NUMINAMATH_CALUDE_most_reasonable_sampling_method_l1847_184727

/-- Represents the different sampling methods --/
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

/-- Represents the educational stages --/
inductive EducationalStage
| Primary
| JuniorHigh
| SeniorHigh

/-- Represents whether there are significant differences in vision conditions --/
def HasSignificantDifferences : Prop := True

/-- The most reasonable sampling method given the conditions --/
def MostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

theorem most_reasonable_sampling_method
  (h1 : HasSignificantDifferences → ∀ (s1 s2 : EducationalStage), s1 ≠ s2 → ∃ (diff : ℝ), diff > 0)
  (h2 : ¬HasSignificantDifferences → ∀ (gender1 gender2 : Bool), ∀ (ε : ℝ), ε > 0 → ∃ (diff : ℝ), diff < ε)
  : MostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by sorry

end NUMINAMATH_CALUDE_most_reasonable_sampling_method_l1847_184727


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1847_184789

theorem sum_of_a_and_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1847_184789


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l1847_184710

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the lengths of semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define an endpoint of the major axis
def C : ℝ × ℝ := (center.1 + a, center.2)

-- Define an endpoint of the minor axis
def D : ℝ × ℝ := (center.1, center.2 + b)

-- Theorem statement
theorem ellipse_axis_endpoints_distance : 
  let distance := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  distance = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l1847_184710


namespace NUMINAMATH_CALUDE_bigger_part_problem_l1847_184774

theorem bigger_part_problem (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 34 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_problem_l1847_184774


namespace NUMINAMATH_CALUDE_relatively_prime_power_sums_l1847_184749

theorem relatively_prime_power_sums (a n m : ℕ) (h_odd : Odd a) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_neq : n ≠ m) :
  Nat.gcd (a^(2^m) + 2^(2^m)) (a^(2^n) + 2^(2^n)) = 1 := by
sorry

end NUMINAMATH_CALUDE_relatively_prime_power_sums_l1847_184749


namespace NUMINAMATH_CALUDE_total_students_is_150_l1847_184758

/-- In a school, when there are 60 boys, girls become 60% of the total number of students. -/
def school_condition (total_students : ℕ) : Prop :=
  (60 : ℝ) / total_students + 0.6 = 1

/-- The theorem states that under the given condition, the total number of students is 150. -/
theorem total_students_is_150 : ∃ (total_students : ℕ), 
  school_condition total_students ∧ total_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_150_l1847_184758


namespace NUMINAMATH_CALUDE_average_candies_per_packet_l1847_184740

def candy_counts : List Nat := [5, 7, 9, 11, 13, 15]
def num_packets : Nat := 6

theorem average_candies_per_packet :
  (candy_counts.sum / num_packets : ℚ) = 10 := by sorry

end NUMINAMATH_CALUDE_average_candies_per_packet_l1847_184740


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1847_184735

-- Define the line
def line (x y : ℝ) : Prop := x + y = 3

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define the chord length
def chord_length (l : (ℝ → ℝ → Prop)) (c : (ℝ → ℝ → Prop)) : ℝ := 
  sorry  -- The actual computation of chord length would go here

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  chord_length line curve = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1847_184735


namespace NUMINAMATH_CALUDE_total_peppers_weight_l1847_184779

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.33

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.33

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.66 pounds -/
theorem total_peppers_weight : total_peppers = 0.66 := by sorry

end NUMINAMATH_CALUDE_total_peppers_weight_l1847_184779


namespace NUMINAMATH_CALUDE_olivias_albums_l1847_184775

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album. -/
def pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) : ℕ :=
  total_pictures / num_albums

/-- Prove that given 40 total pictures and 8 albums, 
    there are 5 pictures in each album. -/
theorem olivias_albums : 
  let total_pictures : ℕ := 40
  let num_albums : ℕ := 8
  pictures_per_album total_pictures num_albums = 5 := by
  sorry

end NUMINAMATH_CALUDE_olivias_albums_l1847_184775


namespace NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l1847_184761

/-- A triangle with sides a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The condition R(b+c) = a√(bc) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.R * (t.b + t.c) = t.a * Real.sqrt (t.b * t.c)

/-- Definition of an isosceles right triangle -/
def isIsoscelesRight (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a * t.a + t.b * t.b = t.c * t.c

/-- The main theorem -/
theorem triangle_condition_implies_isosceles_right (t : Triangle) :
  satisfiesCondition t → isIsoscelesRight t :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_isosceles_right_l1847_184761


namespace NUMINAMATH_CALUDE_interesting_triple_reduction_l1847_184770

/-- Definition of an interesting triple -/
def is_interesting_triple (a b c : ℕ) : Prop :=
  (c^2 + 1) ∣ ((a^2 + 1) * (b^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (a^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (b^2 + 1))

/-- Main theorem -/
theorem interesting_triple_reduction (a b c : ℕ) 
  (h : is_interesting_triple a b c) :
  ∃ u v : ℕ, is_interesting_triple u v c ∧ u * v < c^3 := by
  sorry

end NUMINAMATH_CALUDE_interesting_triple_reduction_l1847_184770


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1847_184739

theorem right_triangle_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) :
  let variance := (a^2 + b^2 + 9) / 3 - ((a + b + 3) / 3)^2
  let std_dev := Real.sqrt variance
  let min_std_dev := Real.sqrt 2 - 1
  let optimal_leg := 3 * Real.sqrt 2 / 2
  (variance < 5) ∧
  (std_dev ≥ min_std_dev) ∧
  (std_dev = min_std_dev ↔ a = optimal_leg ∧ b = optimal_leg) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1847_184739


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l1847_184786

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 28 m × 10 m × 5 m is 1400 cubic meters -/
theorem pond_volume_calculation : pond_volume 28 10 5 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l1847_184786


namespace NUMINAMATH_CALUDE_intersection_irrationality_l1847_184733

theorem intersection_irrationality (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ x : ℚ, x^2 - 2*p*x + 2*q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_irrationality_l1847_184733


namespace NUMINAMATH_CALUDE_committee_size_is_four_l1847_184782

/-- The number of boys in the total group -/
def num_boys : ℕ := 5

/-- The number of girls in the total group -/
def num_girls : ℕ := 6

/-- The number of boys required in each committee -/
def boys_in_committee : ℕ := 2

/-- The number of girls required in each committee -/
def girls_in_committee : ℕ := 2

/-- The total number of possible committees -/
def total_committees : ℕ := 150

/-- The number of people in each committee -/
def committee_size : ℕ := boys_in_committee + girls_in_committee

theorem committee_size_is_four :
  committee_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_committee_size_is_four_l1847_184782


namespace NUMINAMATH_CALUDE_combined_shoe_size_l1847_184797

theorem combined_shoe_size (jasmine_size : ℝ) (alexa_size : ℝ) (clara_size : ℝ) (molly_shoe_size : ℝ) (molly_sandal_size : ℝ) :
  jasmine_size = 7 →
  alexa_size = 2 * jasmine_size →
  clara_size = 3 * jasmine_size →
  molly_shoe_size = 1.5 * jasmine_size →
  molly_sandal_size = molly_shoe_size - 0.5 →
  jasmine_size + alexa_size + clara_size + molly_shoe_size + molly_sandal_size = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l1847_184797


namespace NUMINAMATH_CALUDE_train_rate_problem_l1847_184787

/-- The constant rate of Train A when two trains meet under specific conditions -/
theorem train_rate_problem (total_distance : ℝ) (train_b_rate : ℝ) (train_a_distance : ℝ) :
  total_distance = 350 →
  train_b_rate = 30 →
  train_a_distance = 200 →
  ∃ (train_a_rate : ℝ),
    train_a_rate * (total_distance - train_a_distance) / train_b_rate = train_a_distance ∧
    train_a_rate = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_rate_problem_l1847_184787


namespace NUMINAMATH_CALUDE_camera_imaging_formula_l1847_184745

/-- Given the camera imaging formula, prove the relationship between focal length,
    object distance, and image distance. -/
theorem camera_imaging_formula (f u v : ℝ) (hf : f ≠ 0) (hu : u ≠ 0) (hv : v ≠ 0) (hv_neq_f : v ≠ f) :
  1 / f = 1 / u + 1 / v → v = f * u / (u - f) := by
  sorry

end NUMINAMATH_CALUDE_camera_imaging_formula_l1847_184745


namespace NUMINAMATH_CALUDE_binomial_20_5_l1847_184754

theorem binomial_20_5 : Nat.choose 20 5 = 11628 := by sorry

end NUMINAMATH_CALUDE_binomial_20_5_l1847_184754


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1847_184723

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →  -- First term
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- Arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1847_184723


namespace NUMINAMATH_CALUDE_scientific_notation_equals_original_number_l1847_184706

def scientific_notation_value : ℝ := 6.7 * (10 ^ 6)

theorem scientific_notation_equals_original_number : scientific_notation_value = 6700000 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equals_original_number_l1847_184706


namespace NUMINAMATH_CALUDE_square_area_proof_l1847_184728

theorem square_area_proof (x : ℝ) :
  (6 * x - 27 = 30 - 2 * x) →
  (6 * x - 27) ^ 2 = 248.0625 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1847_184728


namespace NUMINAMATH_CALUDE_election_winner_votes_l1847_184798

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (runner_up_percentage : ℚ) 
  (other_candidates_percentage : ℚ) 
  (invalid_votes_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 48/100 →
  runner_up_percentage = 34/100 →
  other_candidates_percentage = 15/100 →
  invalid_votes_percentage = 3/100 →
  (winner_percentage - runner_up_percentage) * total_votes = vote_difference →
  vote_difference = 2112 →
  winner_percentage * total_votes = 7241 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1847_184798


namespace NUMINAMATH_CALUDE_first_donor_amount_l1847_184737

theorem first_donor_amount (d1 d2 d3 d4 : ℝ) 
  (h1 : d2 = 2 * d1)
  (h2 : d3 = 3 * d2)
  (h3 : d4 = 4 * d3)
  (h4 : d1 + d2 + d3 + d4 = 132) :
  d1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_donor_amount_l1847_184737


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1847_184747

/- Define the number of celebrities and photos -/
def total_celebrities : ℕ := 5
def labeled_photos : ℕ := 3
def unlabeled_photos : ℕ := total_celebrities - labeled_photos

/- Define the function to calculate the number of permutations -/
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n k) * (Nat.factorial k)

/- Theorem statement -/
theorem correct_matching_probability :
  (1 : ℚ) / (permutations total_celebrities unlabeled_photos) = 1 / 20 := by
  sorry


end NUMINAMATH_CALUDE_correct_matching_probability_l1847_184747


namespace NUMINAMATH_CALUDE_mistake_permutations_four_letter_word_l1847_184715

/-- The number of permutations of a word with repeated letters -/
def permutations_with_repetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial r

/-- The number of mistake permutations for a 4-letter word with one letter repeated twice -/
theorem mistake_permutations_four_letter_word : 
  permutations_with_repetition 4 2 - 1 = 11 := by
  sorry

#eval permutations_with_repetition 4 2 - 1

end NUMINAMATH_CALUDE_mistake_permutations_four_letter_word_l1847_184715


namespace NUMINAMATH_CALUDE_average_score_two_classes_l1847_184760

theorem average_score_two_classes (students1 students2 : ℕ) (avg1 avg2 : ℚ) :
  students1 = 20 →
  students2 = 30 →
  avg1 = 80 →
  avg2 = 70 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = 74 :=
by sorry

end NUMINAMATH_CALUDE_average_score_two_classes_l1847_184760


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l1847_184730

theorem ratio_sum_theorem (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 2 / 3) :
  (x + y) / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l1847_184730


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1847_184736

theorem rectangle_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 20
  let length_decrease : ℝ := 5
  let width_increase : ℝ := 5
  let new_length : ℝ := original_length - length_decrease
  let new_width : ℝ := original_width + width_increase
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 75
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1847_184736


namespace NUMINAMATH_CALUDE_sqrt_3x_minus_6_meaningful_l1847_184795

theorem sqrt_3x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 * x - 6) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3x_minus_6_meaningful_l1847_184795


namespace NUMINAMATH_CALUDE_leg_lengths_are_3_or_4_l1847_184738

/-- Represents an isosceles triangle with integer side lengths --/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ
  sum_eq_10 : base + 2 * leg = 10

/-- The set of possible leg lengths for an isosceles triangle formed from a 10cm wire --/
def possible_leg_lengths : Set ℕ :=
  {l | ∃ t : IsoscelesTriangle, t.leg = l}

/-- Theorem stating that the only possible leg lengths are 3 and 4 --/
theorem leg_lengths_are_3_or_4 : possible_leg_lengths = {3, 4} := by
  sorry

#check leg_lengths_are_3_or_4

end NUMINAMATH_CALUDE_leg_lengths_are_3_or_4_l1847_184738


namespace NUMINAMATH_CALUDE_ab_value_l1847_184709

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 405 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1847_184709


namespace NUMINAMATH_CALUDE_cherry_soda_count_l1847_184742

theorem cherry_soda_count (total_cans : ℕ) (orange_ratio : ℕ) (cherry_count : ℕ) : 
  total_cans = 24 →
  orange_ratio = 2 →
  total_cans = cherry_count + orange_ratio * cherry_count →
  cherry_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_soda_count_l1847_184742


namespace NUMINAMATH_CALUDE_direct_proportion_second_fourth_quadrants_l1847_184750

/-- A function f(x) = ax^b is a direct proportion function if and only if b = 1 -/
def is_direct_proportion (a : ℝ) (b : ℝ) : Prop :=
  b = 1

/-- A function f(x) = ax^b has its graph in the second and fourth quadrants if and only if a < 0 -/
def in_second_and_fourth_quadrants (a : ℝ) : Prop :=
  a < 0

/-- The main theorem stating that for y=(m+1)x^(m^2-3) to be a direct proportion function
    with its graph in the second and fourth quadrants, m must be -2 -/
theorem direct_proportion_second_fourth_quadrants :
  ∀ m : ℝ, is_direct_proportion (m + 1) (m^2 - 3) ∧ 
            in_second_and_fourth_quadrants (m + 1) →
            m = -2 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_second_fourth_quadrants_l1847_184750


namespace NUMINAMATH_CALUDE_repeat2016_product_of_palindromes_l1847_184705

/-- A natural number is a palindrome if it reads the same forwards and backwards. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- Repeats the digits 2016 n times to form a natural number. -/
def repeat2016 (n : ℕ) : ℕ := sorry

/-- Theorem: Any number formed by repeating 2016 n times is the product of two palindromes. -/
theorem repeat2016_product_of_palindromes (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℕ), isPalindrome a ∧ isPalindrome b ∧ repeat2016 n = a * b := by sorry

end NUMINAMATH_CALUDE_repeat2016_product_of_palindromes_l1847_184705


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1847_184746

/-- A dodecahedron is a polyhedron with 20 vertices -/
structure Dodecahedron where
  vertices : Finset ℕ
  vertex_count : vertices.card = 20

/-- Each vertex in a dodecahedron is connected by an edge to 3 other vertices -/
def connected_vertices (d : Dodecahedron) (v : ℕ) : Finset ℕ :=
  sorry

/-- The number of vertices connected by an edge to a given vertex is 3 -/
axiom connected_vertices_count (d : Dodecahedron) (v : ℕ) : 
  v ∈ d.vertices → (connected_vertices d v).card = 3

/-- An interior diagonal is a segment connecting two vertices which do not lie on a common edge -/
def interior_diagonals (d : Dodecahedron) : Finset (ℕ × ℕ) :=
  sorry

/-- The number of interior diagonals in a dodecahedron is 160 -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  (interior_diagonals d).card = 160 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1847_184746


namespace NUMINAMATH_CALUDE_digit_rearrangement_divisibility_l1847_184703

def is_digit_rearrangement (n m : ℕ) : Prop :=
  ∃ (digits_n digits_m : List ℕ), 
    digits_n.length > 0 ∧
    digits_m.length > 0 ∧
    digits_n.sum = digits_m.sum ∧
    n = digits_n.foldl (λ acc d => acc * 10 + d) 0 ∧
    m = digits_m.foldl (λ acc d => acc * 10 + d) 0

def satisfies_property (d : ℕ) : Prop :=
  d > 0 ∧ ∀ n m : ℕ, n > 0 → is_digit_rearrangement n m → (d ∣ n → d ∣ m)

theorem digit_rearrangement_divisibility :
  {d : ℕ | satisfies_property d} = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_digit_rearrangement_divisibility_l1847_184703


namespace NUMINAMATH_CALUDE_expression_evaluation_l1847_184711

theorem expression_evaluation :
  let a : ℤ := -2
  let expr := 3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a))
  expr = 10 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1847_184711


namespace NUMINAMATH_CALUDE_jason_total_games_l1847_184791

/-- The total number of games Jason will attend over three months -/
def total_games (this_month last_month next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating the total number of games Jason will attend -/
theorem jason_total_games : 
  total_games 11 17 16 = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l1847_184791


namespace NUMINAMATH_CALUDE_nancy_finished_problems_l1847_184772

/-- Given that Nancy had 101 homework problems initially, still has 6 pages of problems to do,
    and each page has 9 problems, prove that she finished 47 problems. -/
theorem nancy_finished_problems (total_problems : ℕ) (pages_left : ℕ) (problems_per_page : ℕ)
    (h1 : total_problems = 101)
    (h2 : pages_left = 6)
    (h3 : problems_per_page = 9) :
    total_problems - (pages_left * problems_per_page) = 47 := by
  sorry


end NUMINAMATH_CALUDE_nancy_finished_problems_l1847_184772


namespace NUMINAMATH_CALUDE_subset_iff_m_range_l1847_184769

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_iff_m_range (m : ℝ) : B m ⊆ A ↔ m ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_m_range_l1847_184769


namespace NUMINAMATH_CALUDE_cube_sum_equality_l1847_184768

theorem cube_sum_equality (x y z a b c : ℝ) 
  (hx : x^2 = a) (hy : y^2 = b) (hz : z^2 = c) :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧ x^3 + y^3 + z^3 = s * (a^(3/2) + b^(3/2) + c^(3/2)) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l1847_184768


namespace NUMINAMATH_CALUDE_buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1847_184785

/-- Represents a bet on a horse race --/
structure HorseBet where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the odds for each horse --/
def odds : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 1

/-- Calculates the total bet amount --/
def totalBet (bet : HorseBet) : ℕ :=
  bet.first + bet.second + bet.third

/-- Calculates the return for a given horse winning --/
def returnForHorse (bet : HorseBet) (horse : Fin 3) : ℚ :=
  match horse with
  | 0 => (bet.first : ℚ) * (odds 0 + 1)
  | 1 => (bet.second : ℚ) * (odds 1 + 1)
  | 2 => (bet.third : ℚ) * (odds 2 + 1)

/-- Checks if a bet guarantees a minimum return --/
def guaranteesReturn (bet : HorseBet) (minReturn : ℚ) : Prop :=
  ∀ horse : Fin 3, returnForHorse bet horse ≥ minReturn

theorem buratino_bet_exists :
  ∃ bet : HorseBet, totalBet bet = 50 ∧ guaranteesReturn bet 52 :=
sorry

theorem pierrot_bet_impossible :
  ¬∃ bet : HorseBet, totalBet bet = 25 ∧ guaranteesReturn bet 26 :=
sorry

theorem papa_carlo_minimum_bet :
  (∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) + 5)) ∧
  (∀ s : ℕ, s < 95 → ¬∃ bet : HorseBet, totalBet bet = s ∧ guaranteesReturn bet ((s : ℚ) + 5)) :=
sorry

theorem karabas_barabas_impossible :
  ¬∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) * (106 / 100)) :=
sorry

end NUMINAMATH_CALUDE_buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1847_184785


namespace NUMINAMATH_CALUDE_class_fund_problem_l1847_184764

theorem class_fund_problem (total_amount : ℕ) (twenty_bill_count : ℕ) (other_bill_count : ℕ) 
  (h1 : total_amount = 120)
  (h2 : other_bill_count = 2 * twenty_bill_count)
  (h3 : twenty_bill_count = 3) :
  total_amount - (twenty_bill_count * 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_problem_l1847_184764


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l1847_184704

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = 4*y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) →
  16 * b^2 / (a * c) = 100 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l1847_184704


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1847_184717

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 3 = 2 * a 1 →             -- given condition
  (a 4 + 2 * a 7) / 2 = 5 / 4 →     -- given condition
  q = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1847_184717


namespace NUMINAMATH_CALUDE_buy_one_get_one_free_cost_l1847_184722

/-- Calculates the total cost of cans under a "buy 1 get one free" offer -/
def totalCost (totalCans : ℕ) (pricePerCan : ℚ) : ℚ :=
  (totalCans / 2 : ℚ) * pricePerCan

/-- Proves that the total cost for 30 cans at $0.60 each under a "buy 1 get one free" offer is $9 -/
theorem buy_one_get_one_free_cost :
  totalCost 30 (60 / 100) = 9 := by
  sorry

end NUMINAMATH_CALUDE_buy_one_get_one_free_cost_l1847_184722


namespace NUMINAMATH_CALUDE_function_properties_l1847_184773

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x, x ∈ Set.Icc 1 a → f a x ∈ Set.Icc 1 a) ∧
  (∀ x, x ∈ Set.Icc 1 a → f a x = x) →
  a = 2 ∧
  (∀ x ≤ 2, ∀ y ≤ x, f a x ≤ f a y) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → |f a x₁ - f a x₂| ≤ 4) →
  2 ≤ a ∧ a ≤ 3 ∧
  (∃ x ∈ Set.Icc 1 3, f a x = 0) →
  Real.sqrt 5 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1847_184773


namespace NUMINAMATH_CALUDE_solution_set_part1_integer_a_part2_l1847_184755

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |2*x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 5 x ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
sorry

-- Part 2
theorem integer_a_part2 (a : ℤ) :
  (f a 5 ≥ 3 ∧ f a 6 < 3) → a = 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_integer_a_part2_l1847_184755
