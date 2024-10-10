import Mathlib

namespace three_over_x_is_fraction_l1507_150730

/-- A fraction is defined as an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (b : ℚ → ℚ), ∀ x, f x = a / (b x) ∧ b x ≠ 0

/-- The function f(x) = 3/x is a fraction. -/
theorem three_over_x_is_fraction :
  is_fraction (λ x : ℚ => 3 / x) :=
sorry

end three_over_x_is_fraction_l1507_150730


namespace cubic_inequality_l1507_150772

theorem cubic_inequality (a b c : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) : 
  2*a^3 + 9*c ≤ 7*a*b ∧ 
  (2*a^3 + 9*c = 7*a*b ↔ ∃ r : ℝ, r > 0 ∧ ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = r) :=
sorry

end cubic_inequality_l1507_150772


namespace smallest_next_divisor_l1507_150700

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisor (d m : ℕ) : Prop := ∃ k, m = d * k

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : is_divisor 437 m) :
  ∃ d : ℕ, 
    is_divisor d m ∧ 
    d > 437 ∧ 
    (∀ d' : ℕ, is_divisor d' m → d' > 437 → d ≤ d') ∧ 
    d = 475 :=
sorry

end smallest_next_divisor_l1507_150700


namespace soup_weight_after_four_days_l1507_150764

/-- The weight of soup remaining after four days of reduction -/
def remaining_soup_weight (initial_weight : ℝ) (day1_reduction day2_reduction day3_reduction day4_reduction : ℝ) : ℝ :=
  initial_weight * (1 - day1_reduction) * (1 - day2_reduction) * (1 - day3_reduction) * (1 - day4_reduction)

/-- Theorem stating the remaining weight of soup after four days -/
theorem soup_weight_after_four_days :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |remaining_soup_weight 80 0.40 0.35 0.55 0.50 - 7.02| < ε :=
sorry

end soup_weight_after_four_days_l1507_150764


namespace sunglasses_cap_probability_l1507_150794

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 120) 
  (h2 : total_caps = 84) 
  (h3 : total_hats = 60) 
  (h4 : prob_cap_and_sunglasses = 3 / 7) : 
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 3 / 10 := by
sorry

end sunglasses_cap_probability_l1507_150794


namespace division_problem_l1507_150711

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 689 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end division_problem_l1507_150711


namespace no_integer_geometric_progression_angles_l1507_150728

/-- Represents the angles of a triangle in geometric progression -/
structure TriangleAngles where
  a : ℕ+  -- first angle
  r : ℕ+  -- common ratio
  h1 : a < a * r  -- angles are distinct
  h2 : a * r < a * r * r  -- angles are distinct
  h3 : a + a * r + a * r * r = 180  -- sum of angles is 180 degrees

/-- There are no triangles with angles that are distinct positive integers in a geometric progression -/
theorem no_integer_geometric_progression_angles : ¬∃ (t : TriangleAngles), True :=
sorry

end no_integer_geometric_progression_angles_l1507_150728


namespace conditional_probability_of_longevity_l1507_150768

theorem conditional_probability_of_longevity 
  (p_20 : ℝ) 
  (p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) : 
  p_25 / p_20 = 0.5 := by
  sorry

end conditional_probability_of_longevity_l1507_150768


namespace scientific_notation_of_240000_l1507_150793

theorem scientific_notation_of_240000 : 
  240000 = 2.4 * (10 ^ 5) := by sorry

end scientific_notation_of_240000_l1507_150793


namespace toys_produced_daily_l1507_150719

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 3400

/-- The number of working days per week -/
def working_days : ℕ := 5

/-- The number of toys produced each day -/
def toys_per_day : ℕ := toys_per_week / working_days

/-- Theorem stating that the number of toys produced each day is 680 -/
theorem toys_produced_daily : toys_per_day = 680 := by
  sorry

end toys_produced_daily_l1507_150719


namespace rectangle_area_perimeter_relation_l1507_150797

/-- Given a rectangle with length x^2 and width x + 5, prove that if its area
    equals three times its perimeter, then x = 3. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (x^2 * (x + 5) = 3 * (2 * x^2 + 2 * (x + 5))) → x = 3 := by
  sorry

end rectangle_area_perimeter_relation_l1507_150797


namespace minimum_cost_theorem_l1507_150771

/-- The unit price of a volleyball in yuan -/
def volleyball_price : ℝ := 50

/-- The unit price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 80

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 11

/-- The minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 2

/-- The cost function for purchasing x volleyballs -/
def cost_function (x : ℝ) : ℝ := -30 * x + 880

/-- The theorem stating the minimum cost of purchasing the balls -/
theorem minimum_cost_theorem :
  ∃ (x : ℝ), 
    0 ≤ x ∧ 
    x ≤ total_balls - min_soccer_balls ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_balls - min_soccer_balls → 
      cost_function x ≤ cost_function y ∧
      cost_function x = 610 :=
sorry

end minimum_cost_theorem_l1507_150771


namespace sum_and_ratio_implies_difference_l1507_150709

theorem sum_and_ratio_implies_difference (x y : ℝ) 
  (h1 : x + y = 500) 
  (h2 : x / y = 0.75) : 
  y - x = 71.42 := by
sorry

end sum_and_ratio_implies_difference_l1507_150709


namespace intersection_of_A_and_B_l1507_150756

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l1507_150756


namespace binary_calculation_l1507_150744

theorem binary_calculation : 
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end binary_calculation_l1507_150744


namespace polynomial_equality_l1507_150747

theorem polynomial_equality (x : ℝ) : 
  (x - 2/3) * (x + 1/2) = x^2 - (1/6)*x - 1/3 := by
  sorry

end polynomial_equality_l1507_150747


namespace at_least_one_blue_multiple_of_three_l1507_150741

/-- Represents a marked point on the circle --/
structure MarkedPoint where
  value : Int

/-- Represents a chord on the circle --/
structure Chord where
  points : List MarkedPoint

/-- The configuration of chords and points on the circle --/
structure CircleConfiguration where
  chords : List Chord
  endpointZeros : Nat
  endpointOnes : Nat

/-- Calculates yellow numbers (sum of endpoint values) for a chord --/
def yellowNumbers (chord : Chord) : List Int :=
  sorry

/-- Calculates blue numbers (absolute difference of endpoint values) for a chord --/
def blueNumbers (chord : Chord) : List Int :=
  sorry

/-- Checks if the yellow numbers are consecutive from 0 to N --/
def isConsecutiveYellow (yellowNums : List Int) : Bool :=
  sorry

theorem at_least_one_blue_multiple_of_three 
  (config : CircleConfiguration) 
  (h1 : config.chords.length = 2019)
  (h2 : config.endpointZeros = 2019)
  (h3 : config.endpointOnes = 2019)
  (h4 : ∀ c ∈ config.chords, c.points.length ≥ 2)
  (h5 : let allYellow := config.chords.map yellowNumbers |>.join
        isConsecutiveYellow allYellow) :
  ∃ (b : Int), b ∈ (config.chords.map blueNumbers |>.join) ∧ b % 3 = 0 := by
  sorry


end at_least_one_blue_multiple_of_three_l1507_150741


namespace unpainted_cubes_in_6x6x6_l1507_150752

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  paintedSquaresPerFace : Nat
  paintedFaces : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedUnitCubes cube
where
  /-- Calculates the number of painted unit cubes, accounting for overlaps -/
  paintedUnitCubes (cube : PaintedCube) : Nat :=
    let totalPaintedSquares := cube.paintedSquaresPerFace * cube.paintedFaces
    let edgeOverlap := 12 * 2  -- 12 edges, each counted twice
    let cornerOverlap := 8 * 2  -- 8 corners, each counted thrice (so subtract 2)
    totalPaintedSquares - edgeOverlap - cornerOverlap

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    paintedSquaresPerFace := 13,
    paintedFaces := 6
  }
  unpaintedUnitCubes cube = 210 := by
  sorry

end unpainted_cubes_in_6x6x6_l1507_150752


namespace range_of_a_l1507_150734

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l1507_150734


namespace max_value_x_plus_sqrt_one_minus_x_squared_l1507_150705

theorem max_value_x_plus_sqrt_one_minus_x_squared :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x + Real.sqrt (1 - x^2) ≤ Real.sqrt 2 := by
  sorry

end max_value_x_plus_sqrt_one_minus_x_squared_l1507_150705


namespace malcolm_flat_path_time_l1507_150708

/-- Represents the time in minutes for different parts of Malcolm's routes to school -/
structure RouteTime where
  uphill : ℕ
  path : ℕ
  final : ℕ

/-- Calculates the total time for the first route -/
def first_route_time (r : RouteTime) : ℕ :=
  r.uphill + r.path + r.final

/-- Calculates the total time for the second route -/
def second_route_time (flat_path : ℕ) : ℕ :=
  flat_path + 2 * flat_path

/-- Theorem stating the correct time Malcolm spent on the flat path in the second route -/
theorem malcolm_flat_path_time : ∃ (r : RouteTime) (flat_path : ℕ),
  r.uphill = 6 ∧
  r.path = 2 * r.uphill ∧
  r.final = (r.uphill + r.path) / 3 ∧
  second_route_time flat_path = first_route_time r + 18 ∧
  flat_path = 14 := by
  sorry

end malcolm_flat_path_time_l1507_150708


namespace sum_of_a_values_l1507_150720

theorem sum_of_a_values (a b : ℝ) (h1 : a + 1/b = 8) (h2 : b + 1/a = 3) : 
  ∃ (a₁ a₂ : ℝ), (a₁ + 1/b = 8 ∧ b + 1/a₁ = 3) ∧ 
                 (a₂ + 1/b = 8 ∧ b + 1/a₂ = 3) ∧ 
                 a₁ ≠ a₂ ∧ 
                 a₁ + a₂ = 8 := by
  sorry

end sum_of_a_values_l1507_150720


namespace mean_score_of_all_students_l1507_150783

theorem mean_score_of_all_students
  (avg_score_group1 : ℝ)
  (avg_score_group2 : ℝ)
  (ratio_students : ℚ)
  (h1 : avg_score_group1 = 90)
  (h2 : avg_score_group2 = 75)
  (h3 : ratio_students = 2/5) :
  let total_score := avg_score_group1 * (ratio_students * s) + avg_score_group2 * s
  let total_students := ratio_students * s + s
  total_score / total_students = 79 :=
by
  sorry

#check mean_score_of_all_students

end mean_score_of_all_students_l1507_150783


namespace longest_segment_in_quarter_pie_l1507_150751

theorem longest_segment_in_quarter_pie (d : ℝ) (h : d = 20) : 
  let r := d / 2
  let l := 2 * r * Real.sin (π / 4)
  l^2 = 200 := by
  sorry

end longest_segment_in_quarter_pie_l1507_150751


namespace roots_product_l1507_150759

theorem roots_product (a b : ℝ) : 
  a^2 + a - 2020 = 0 → b^2 + b - 2020 = 0 → (a - 1) * (b - 1) = -2018 := by
  sorry

end roots_product_l1507_150759


namespace box_decoration_combinations_l1507_150745

/-- The number of paint color options available -/
def num_colors : ℕ := 4

/-- The number of decoration options available -/
def num_decorations : ℕ := 3

/-- The total number of combinations for painting and decorating a box -/
def total_combinations : ℕ := num_colors * num_decorations

theorem box_decoration_combinations :
  total_combinations = 12 :=
by sorry

end box_decoration_combinations_l1507_150745


namespace cubic_equation_solutions_l1507_150701

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3 + x + Real.sqrt 3
  ∃ (z₁ z₂ z₃ : ℂ), 
    z₁ = -Real.sqrt 3 ∧ 
    z₂ = -Real.sqrt 3 + Complex.I ∧ 
    z₃ = -Real.sqrt 3 - Complex.I ∧
    (∀ z : ℂ, f z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end cubic_equation_solutions_l1507_150701


namespace kiyana_grapes_l1507_150723

/-- Proves that if Kiyana has 24 grapes and gives away half of them, the number of grapes she gives away is 12. -/
theorem kiyana_grapes : 
  let total_grapes : ℕ := 24
  let grapes_given_away : ℕ := total_grapes / 2
  grapes_given_away = 12 := by
  sorry

end kiyana_grapes_l1507_150723


namespace non_pine_trees_count_l1507_150769

/-- Given a park with 350 trees, where 70% are pine trees, prove that 105 trees are not pine trees. -/
theorem non_pine_trees_count (total_trees : ℕ) (pine_percentage : ℚ) : 
  total_trees = 350 → pine_percentage = 70 / 100 →
  (total_trees : ℚ) - (pine_percentage * total_trees) = 105 := by
  sorry

end non_pine_trees_count_l1507_150769


namespace complement_of_P_in_U_l1507_150706

-- Define the universal set U
def U : Set ℝ := {x | x ≥ 0}

-- Define the set P
def P : Set ℝ := {1}

-- Theorem statement
theorem complement_of_P_in_U :
  (U \ P) = {x : ℝ | x ≥ 0 ∧ x ≠ 1} := by sorry

end complement_of_P_in_U_l1507_150706


namespace rectangle_problem_l1507_150726

-- Define the rectangle EFGH
structure Rectangle (EFGH : Type) where
  is_rectangle : EFGH → Prop

-- Define point M on FG
def M_on_FG (EFGH : Type) (M : EFGH) : Prop := sorry

-- Define angle EMH as 90°
def angle_EMH_90 (EFGH : Type) (E M H : EFGH) : Prop := sorry

-- Define UV perpendicular to FG
def UV_perp_FG (EFGH : Type) (U V F G : EFGH) : Prop := sorry

-- Define FU = UM
def FU_eq_UM (EFGH : Type) (F U M : EFGH) : Prop := sorry

-- Define MH intersects UV at N
def MH_intersect_UV_at_N (EFGH : Type) (M H U V N : EFGH) : Prop := sorry

-- Define S on GH such that SE passes through N
def S_on_GH_SE_through_N (EFGH : Type) (S G H E N : EFGH) : Prop := sorry

-- Define triangle MNE with given measurements
def triangle_MNE (EFGH : Type) (M N E : EFGH) : Prop :=
  let ME := 25
  let EN := 20
  let MN := 20
  sorry

-- Theorem statement
theorem rectangle_problem (EFGH : Type) 
  (E F G H M U V N S : EFGH) 
  (rect : Rectangle EFGH) 
  (h1 : M_on_FG EFGH M)
  (h2 : angle_EMH_90 EFGH E M H)
  (h3 : UV_perp_FG EFGH U V F G)
  (h4 : FU_eq_UM EFGH F U M)
  (h5 : MH_intersect_UV_at_N EFGH M H U V N)
  (h6 : S_on_GH_SE_through_N EFGH S G H E N)
  (h7 : triangle_MNE EFGH M N E) :
  ∃ (FM NV : ℝ), FM = 15 ∧ NV = 5 * Real.sqrt 7 := by
  sorry

end rectangle_problem_l1507_150726


namespace a_gt_one_iff_a_gt_zero_l1507_150777

theorem a_gt_one_iff_a_gt_zero {a : ℝ} : a > 1 ↔ a > 0 := by sorry

end a_gt_one_iff_a_gt_zero_l1507_150777


namespace point_b_coordinates_l1507_150776

/-- Given point A (-1, 5) and vector a (2, 3), if vector AB = 3 * vector a, 
    then the coordinates of point B are (5, 14). -/
theorem point_b_coordinates 
  (A : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (-1, 5)) 
  (h2 : a = (2, 3)) 
  (h3 : B.1 - A.1 = 3 * a.1 ∧ B.2 - A.2 = 3 * a.2) : 
  B = (5, 14) := by
sorry


end point_b_coordinates_l1507_150776


namespace fourth_root_over_sixth_root_of_seven_l1507_150748

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ)^(1/4) / (7 : ℝ)^(1/6) = (7 : ℝ)^(1/12) := by
  sorry

end fourth_root_over_sixth_root_of_seven_l1507_150748


namespace cost_per_share_is_50_l1507_150729

/-- Represents the savings and investment scenario of a married couple --/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  husband_monthly_savings : ℕ
  savings_period_months : ℕ
  investment_fraction : ℚ
  num_shares_bought : ℕ

/-- Calculates the cost per share of stock based on the given savings scenario --/
def cost_per_share (scenario : SavingsScenario) : ℚ :=
  let total_savings := (scenario.wife_weekly_savings * 4 * scenario.savings_period_months +
                        scenario.husband_monthly_savings * scenario.savings_period_months)
  let investment_amount := (total_savings : ℚ) * scenario.investment_fraction
  investment_amount / scenario.num_shares_bought

/-- Theorem stating that the cost per share is $50 for the given scenario --/
theorem cost_per_share_is_50 (scenario : SavingsScenario)
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.husband_monthly_savings = 225)
  (h3 : scenario.savings_period_months = 4)
  (h4 : scenario.investment_fraction = 1/2)
  (h5 : scenario.num_shares_bought = 25) :
  cost_per_share scenario = 50 := by
  sorry

end cost_per_share_is_50_l1507_150729


namespace cora_cookie_expenditure_l1507_150722

def cookies_per_day : ℕ := 3
def cookie_cost : ℕ := 18
def days_in_april : ℕ := 30

theorem cora_cookie_expenditure :
  cookies_per_day * cookie_cost * days_in_april = 1620 := by
  sorry

end cora_cookie_expenditure_l1507_150722


namespace geometric_sum_15_l1507_150742

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
sorry

end geometric_sum_15_l1507_150742


namespace total_pies_sold_l1507_150779

/-- Represents the daily pie sales for a week -/
structure WeekSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total sales for a week -/
def totalSales (sales : WeekSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

/-- The actual sales data for the week -/
def actualSales : WeekSales := {
  monday := 8,
  tuesday := 12,
  wednesday := 14,
  thursday := 20,
  friday := 20,
  saturday := 20,
  sunday := 20
}

/-- Theorem: The total number of pies sold in the week is 114 -/
theorem total_pies_sold : totalSales actualSales = 114 := by
  sorry

end total_pies_sold_l1507_150779


namespace distinct_prime_factors_of_90_l1507_150774

theorem distinct_prime_factors_of_90 : Finset.card (Nat.factors 90).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_90_l1507_150774


namespace train_distance_problem_l1507_150716

/-- The distance between two points A and B, given two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (hv1 : v1 = 50) (hv2 : v2 = 60) (hd : d = 100) :
  let x := (v1 * d) / (v2 - v1)
  x + (x + d) = 1100 :=
by sorry

end train_distance_problem_l1507_150716


namespace system_solution_l1507_150703

theorem system_solution : ∃ (x y : ℝ), 
  (x - 2*y = 3) ∧ (3*x - y = 4) ∧ (x = 1) ∧ (y = -1) := by
  sorry

end system_solution_l1507_150703


namespace science_fair_participants_l1507_150773

/-- The number of unique students participating in the Science Fair --/
def unique_students (robotics astronomy chemistry all_three : ℕ) : ℕ :=
  robotics + astronomy + chemistry - 2 * all_three

/-- Theorem stating the number of unique students in the Science Fair --/
theorem science_fair_participants : unique_students 15 10 12 2 = 33 := by
  sorry

end science_fair_participants_l1507_150773


namespace sum_of_multiples_of_4_between_34_and_135_l1507_150786

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let firstMultiple := (lower + 3) / 4 * 4
  let lastMultiple := upper / 4 * 4
  let n := (lastMultiple - firstMultiple) / 4 + 1
  n * (firstMultiple + lastMultiple) / 2

theorem sum_of_multiples_of_4_between_34_and_135 :
  sumOfMultiplesOf4 34 135 = 2100 := by
  sorry

end sum_of_multiples_of_4_between_34_and_135_l1507_150786


namespace raghu_investment_l1507_150758

theorem raghu_investment (total_investment : ℝ) (vishal_investment : ℝ → ℝ) (trishul_investment : ℝ → ℝ) :
  total_investment = 7225 ∧
  (∀ r, vishal_investment r = 1.1 * (trishul_investment r)) ∧
  (∀ r, trishul_investment r = 0.9 * r) →
  ∃ r, r = 2500 ∧ r + trishul_investment r + vishal_investment r = total_investment :=
by sorry

end raghu_investment_l1507_150758


namespace soccer_team_lineup_count_l1507_150717

theorem soccer_team_lineup_count :
  let team_size : ℕ := 16
  let positions_to_fill : ℕ := 5
  (team_size.factorial) / ((team_size - positions_to_fill).factorial) = 524160 := by
  sorry

end soccer_team_lineup_count_l1507_150717


namespace inscribed_squares_equal_area_l1507_150724

/-- 
Given an isosceles right triangle with an inscribed square parallel to the legs,
prove that if this square has an area of 625, then a square inscribed with sides
parallel and perpendicular to the hypotenuse also has an area of 625.
-/
theorem inscribed_squares_equal_area (side : ℝ) (h_area : side^2 = 625) :
  let hypotenuse := side * Real.sqrt 2
  let side_hyp_square := hypotenuse / 2
  side_hyp_square^2 = 625 := by sorry

end inscribed_squares_equal_area_l1507_150724


namespace investment_difference_l1507_150753

def initial_investment : ℕ := 10000

def alice_multiplier : ℕ := 3
def bob_multiplier : ℕ := 7

def alice_final : ℕ := initial_investment * alice_multiplier
def bob_final : ℕ := initial_investment * bob_multiplier

theorem investment_difference : bob_final - alice_final = 40000 := by
  sorry

end investment_difference_l1507_150753


namespace valid_arrangements_count_l1507_150781

/-- Represents a seating arrangement for cousins in a van --/
structure SeatingArrangement where
  row1 : Fin 4 → Fin 7
  row2 : Fin 4 → Option (Fin 7)

/-- Represents a pair of cousins --/
inductive CousinPair
  | Pair1
  | Pair2
  | Pair3

/-- Returns the pair that a cousin belongs to, if any --/
def cousinPair (cousin : Fin 7) : Option CousinPair := sorry

/-- Checks if a seating arrangement is valid according to the rules --/
def isValidArrangement (arr : SeatingArrangement) : Prop := sorry

/-- Counts the number of valid seating arrangements --/
def countValidArrangements : Nat := sorry

/-- Theorem stating that the number of valid seating arrangements is 240 --/
theorem valid_arrangements_count :
  countValidArrangements = 240 := by sorry

end valid_arrangements_count_l1507_150781


namespace fraction_equality_l1507_150736

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1001) : (w + z)/(w - z) = -501 := by
  sorry

end fraction_equality_l1507_150736


namespace section_area_of_specific_pyramid_l1507_150798

/-- Regular quadrilateral pyramid with square base -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  angle_with_base : ℝ

/-- The area of the section created by the intersecting plane -/
noncomputable def section_area (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem section_area_of_specific_pyramid :
  let p : RegularQuadPyramid := ⟨8, 9⟩
  let plane : IntersectingPlane := ⟨Real.arctan (3/4)⟩
  section_area p plane = 45 := by sorry

end section_area_of_specific_pyramid_l1507_150798


namespace range_of_expression_l1507_150762

def line_equation (x : ℝ) : ℝ := -2 * x + 8

theorem range_of_expression (x₁ y₁ : ℝ) :
  y₁ = line_equation x₁ →
  x₁ ∈ Set.Icc 2 5 →
  (y₁ + 1) / (x₁ + 1) ∈ Set.Icc (-1/6) (5/3) := by
  sorry

end range_of_expression_l1507_150762


namespace exam_papers_count_l1507_150721

theorem exam_papers_count (average : ℝ) (new_average : ℝ) (geography_increase : ℝ) (history_increase : ℝ) :
  average = 63 →
  new_average = 65 →
  geography_increase = 20 →
  history_increase = 2 →
  ∃ n : ℕ, n * average + geography_increase + history_increase = n * new_average ∧ n = 11 :=
by sorry

end exam_papers_count_l1507_150721


namespace myrtle_dropped_eggs_l1507_150704

/-- Represents the problem of calculating how many eggs Myrtle dropped --/
theorem myrtle_dropped_eggs (hens : ℕ) (eggs_per_hen : ℕ) (days : ℕ) (neighbor_took : ℕ) (myrtle_has : ℕ)
  (h1 : hens = 3)
  (h2 : eggs_per_hen = 3)
  (h3 : days = 7)
  (h4 : neighbor_took = 12)
  (h5 : myrtle_has = 46) :
  hens * eggs_per_hen * days - neighbor_took - myrtle_has = 5 := by
  sorry

#check myrtle_dropped_eggs

end myrtle_dropped_eggs_l1507_150704


namespace evaluate_power_of_power_l1507_150733

theorem evaluate_power_of_power : (3^3)^2 = 729 := by sorry

end evaluate_power_of_power_l1507_150733


namespace ellipse_from_hyperbola_l1507_150731

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the equation of an ellipse
    with foci at the vertices of the hyperbola and vertices at the foci of the hyperbola
    is x²/9 + y²/5 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (a b c : ℝ),
    (a^2 = 9 ∧ b^2 = 5 ∧ c^2 = 4) ∧
    (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end ellipse_from_hyperbola_l1507_150731


namespace ellipse_chord_slope_l1507_150702

/-- The slope of a chord of an ellipse bisected by a given point -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Q(x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 1) →         -- Midpoint x-coordinate is 1
  ((y₁ + y₂) / 2 = 1) →         -- Midpoint y-coordinate is 1
  (y₂ - y₁) / (x₂ - x₁) = -1/4  -- Slope of PQ is -1/4
:= by sorry

end ellipse_chord_slope_l1507_150702


namespace largest_n_squared_sum_largest_n_exists_largest_n_is_three_l1507_150755

theorem largest_n_squared_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) →
  n ≤ 3 :=
by sorry

theorem largest_n_exists : 
  ∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18 :=
by sorry

theorem largest_n_is_three : 
  (∃ (n : ℕ+), (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) ∧
  (∀ (m : ℕ+), (∃ (a b c : ℕ+), m^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 18) → m ≤ n)) →
  (∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) :=
by sorry

end largest_n_squared_sum_largest_n_exists_largest_n_is_three_l1507_150755


namespace dawn_monthly_savings_l1507_150749

theorem dawn_monthly_savings (annual_salary : ℝ) (months_per_year : ℕ) (savings_rate : ℝ) : 
  annual_salary = 48000 ∧ 
  months_per_year = 12 ∧ 
  savings_rate = 0.1 → 
  (annual_salary / months_per_year) * savings_rate = 400 := by
sorry

end dawn_monthly_savings_l1507_150749


namespace M_intersect_N_eq_l1507_150765

def M : Set ℝ := {x | -x^2 - 5*x + 6 > 0}

def N : Set ℝ := {x | |x + 1| < 1}

theorem M_intersect_N_eq : M ∩ N = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end M_intersect_N_eq_l1507_150765


namespace haley_small_gardens_l1507_150754

/-- The number of small gardens Haley had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Theorem stating that Haley had 7 small gardens -/
theorem haley_small_gardens : 
  num_small_gardens 56 35 3 = 7 := by
  sorry

end haley_small_gardens_l1507_150754


namespace sin_780_degrees_l1507_150799

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_780_degrees_l1507_150799


namespace inequality_proof_l1507_150763

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end inequality_proof_l1507_150763


namespace triple_debt_days_l1507_150710

def loan_amount : ℝ := 20
def daily_interest_rate : ℝ := 0.10

def days_to_triple_debt : ℕ := 20

theorem triple_debt_days :
  ∀ n : ℕ, (n : ℝ) ≥ (days_to_triple_debt : ℝ) ↔
  loan_amount * (1 + n * daily_interest_rate) ≥ 3 * loan_amount :=
by sorry

end triple_debt_days_l1507_150710


namespace otimes_neg_two_neg_one_l1507_150715

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end otimes_neg_two_neg_one_l1507_150715


namespace vertex_of_f_l1507_150712

/-- The quadratic function f(x) = -3(x+1)^2 - 2 -/
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -2)

/-- Theorem: The vertex of the quadratic function f is (-1, -2) -/
theorem vertex_of_f : 
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end vertex_of_f_l1507_150712


namespace brick_wall_problem_l1507_150778

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  rows : Nat
  total_bricks : Nat
  bottom_row_bricks : Nat
  row_difference : Nat

/-- Calculates the sum of bricks in all rows of the wall -/
def sum_of_bricks (wall : BrickWall) : Nat :=
  wall.rows * wall.bottom_row_bricks - (wall.rows * (wall.rows - 1) * wall.row_difference) / 2

/-- Theorem stating the properties of the specific brick wall -/
theorem brick_wall_problem : ∃ (wall : BrickWall), 
  wall.rows = 5 ∧ 
  wall.total_bricks = 200 ∧ 
  wall.row_difference = 1 ∧
  sum_of_bricks wall = wall.total_bricks ∧
  wall.bottom_row_bricks = 42 := by
  sorry

end brick_wall_problem_l1507_150778


namespace x_squared_plus_reciprocal_l1507_150714

theorem x_squared_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  x^4 + 1/x^4 = 23 → x^2 + 1/x^2 = 5 := by
  sorry

end x_squared_plus_reciprocal_l1507_150714


namespace equation_factorization_l1507_150738

theorem equation_factorization :
  ∀ x : ℝ, (5*x - 1)^2 = 3*(5*x - 1) ↔ (5*x - 1)*(5*x - 4) = 0 :=
by sorry

end equation_factorization_l1507_150738


namespace ribbon_fraction_l1507_150732

theorem ribbon_fraction (total_fraction : ℚ) (num_packages : ℕ) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_packages = 5) :
  total_fraction / num_packages = 1 / 12 := by
  sorry

end ribbon_fraction_l1507_150732


namespace green_ball_probability_l1507_150727

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The game setup -/
def game : List Container := [
  { red := 8, green := 4 },
  { red := 7, green := 4 },
  { red := 7, green := 4 }
]

/-- The probability of selecting each container -/
def containerProb : Rat := 1 / 3

/-- Calculates the probability of drawing a green ball from a given container -/
def greenProbFromContainer (c : Container) : Rat :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of drawing a green ball -/
def totalGreenProb : Rat :=
  (game.map greenProbFromContainer).sum / game.length

/-- The main theorem: the probability of drawing a green ball is 35/99 -/
theorem green_ball_probability : totalGreenProb = 35 / 99 := by
  sorry

end green_ball_probability_l1507_150727


namespace unique_number_property_l1507_150770

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end unique_number_property_l1507_150770


namespace four_numbers_problem_l1507_150767

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end four_numbers_problem_l1507_150767


namespace matrix_power_1000_l1507_150790

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_1000 :
  A^1000 = !![1, 0; 2000, 1] := by sorry

end matrix_power_1000_l1507_150790


namespace fruit_basket_cost_l1507_150746

/-- Represents the composition and prices of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_dozen_price : ℚ
  avocado_price : ℚ
  grape_half_bunch_price : ℚ

/-- Calculates the total cost of the fruit basket -/
def total_cost (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  (fb.strawberry_count / 12) * fb.strawberry_dozen_price +
  fb.avocado_count * fb.avocado_price +
  2 * fb.grape_half_bunch_price

/-- The given fruit basket -/
def given_basket : FruitBasket := {
  banana_count := 4
  apple_count := 3
  strawberry_count := 24
  avocado_count := 2
  banana_price := 1
  apple_price := 2
  strawberry_dozen_price := 4
  avocado_price := 3
  grape_half_bunch_price := 2
}

/-- Theorem stating that the total cost of the given fruit basket is $28 -/
theorem fruit_basket_cost : total_cost given_basket = 28 := by
  sorry

end fruit_basket_cost_l1507_150746


namespace roots_not_real_l1507_150760

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z m : ℂ) : Prop :=
  5 * z^2 - 7 * i * z - m = 0

-- State the theorem
theorem roots_not_real (m : ℂ) :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ m ∧ quadratic_equation z₂ m ∧
  z₁ ≠ z₂ ∧ ¬(z₁.im = 0) ∧ ¬(z₂.im = 0) := by
  sorry

end roots_not_real_l1507_150760


namespace scientific_notation_digits_l1507_150788

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

/-- Conversion from scientific notation to standard form -/
def scientific_to_standard (mantissa : ℚ) (exponent : ℤ) : ℚ :=
  mantissa * (10 : ℚ) ^ exponent

theorem scientific_notation_digits :
  let mantissa : ℚ := 721 / 100
  let exponent : ℤ := 11
  let standard_form := scientific_to_standard mantissa exponent
  num_digits (Nat.floor standard_form) = 12 := by
sorry

end scientific_notation_digits_l1507_150788


namespace product_53_57_l1507_150785

theorem product_53_57 (h : 2021 = 43 * 47) : 53 * 57 = 3021 := by
  sorry

end product_53_57_l1507_150785


namespace simplification_fraction_l1507_150743

theorem simplification_fraction (k : ℤ) : 
  let simplified := (6 * k + 18) / 6
  ∃ (a b : ℤ), simplified = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end simplification_fraction_l1507_150743


namespace max_value_sqrt_sum_l1507_150740

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y, -36 ≤ y ∧ y ≤ 36 ∧ Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end max_value_sqrt_sum_l1507_150740


namespace game_points_percentage_l1507_150750

theorem game_points_percentage (samanta mark eric : ℕ) : 
  samanta = mark + 8 →
  eric = 6 →
  samanta + mark + eric = 32 →
  (mark - eric : ℚ) / eric * 100 = 50 := by
sorry

end game_points_percentage_l1507_150750


namespace intersection_of_M_and_N_l1507_150796

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end intersection_of_M_and_N_l1507_150796


namespace quadratic_coefficient_b_l1507_150780

theorem quadratic_coefficient_b (a b c y₁ y₂ y₃ : ℝ) : 
  y₁ = a + b + c →
  y₂ = a - b + c →
  y₃ = 4*a + 2*b + c →
  y₁ - y₂ = 8 →
  y₃ = y₁ + 2 →
  b = 4 := by
sorry


end quadratic_coefficient_b_l1507_150780


namespace pet_store_combinations_l1507_150766

def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 5
def num_customers : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * Nat.factorial num_customers = 288000 := by
  sorry

end pet_store_combinations_l1507_150766


namespace pure_imaginary_condition_l1507_150737

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - 1) (a - 2)
  (z.re = 0 ∧ z.im ≠ 0) → (a = -1 ∨ a = 1) :=
by sorry

end pure_imaginary_condition_l1507_150737


namespace fencing_requirement_l1507_150735

/-- Represents a rectangular field with given dimensions and fencing requirements. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a rectangular field. -/
def required_fencing (field : RectangularField) : ℝ :=
  field.length + 2 * field.width

/-- Theorem stating the required fencing for a specific rectangular field. -/
theorem fencing_requirement (field : RectangularField)
  (h1 : field.area = 650)
  (h2 : field.uncovered_side = 20)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 85 :=
sorry

end fencing_requirement_l1507_150735


namespace triangle_properties_l1507_150787

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  A = π / 3 →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  c = 3 ∧
  Real.sin B = Real.sqrt 21 / 7 ∧
  π * (a / (2 * Real.sin A))^2 = 7 * π / 3 :=
by sorry

end triangle_properties_l1507_150787


namespace more_girls_than_boys_l1507_150782

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 345 →
  boys = 138 →
  girls > boys →
  total = girls + boys →
  girls - boys = 69 :=
by sorry

end more_girls_than_boys_l1507_150782


namespace max_time_at_8_l1507_150795

noncomputable def y (t : ℝ) : ℝ := -1/8 * t^3 - 3/4 * t^2 + 36*t - 629/4

theorem max_time_at_8 :
  ∃ (t_max : ℝ), t_max = 8 ∧
  ∀ (t : ℝ), 6 ≤ t ∧ t ≤ 9 → y t ≤ y t_max :=
by sorry

end max_time_at_8_l1507_150795


namespace inequality_equivalence_system_of_inequalities_equivalence_l1507_150784

theorem inequality_equivalence (x : ℝ) :
  (1 - (x - 3) / 6 > x / 3) ↔ (x < 3) :=
sorry

theorem system_of_inequalities_equivalence (x : ℝ) :
  (x + 1 ≥ 3 * (x - 3) ∧ (x + 2) / 3 - (x - 1) / 4 > 1) ↔ (1 < x ∧ x ≤ 5) :=
sorry

end inequality_equivalence_system_of_inequalities_equivalence_l1507_150784


namespace count_special_integers_l1507_150713

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0 ∧ d ≤ 9

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_five_digit (n : Nat) : Bool :=
  10000 ≤ n ∧ n ≤ 99999

theorem count_special_integers :
  (Finset.filter (λ n : Nat => is_five_digit n ∧ has_only_even_digits n ∧ n % 5 = 0)
    (Finset.range 100000)).card = 500 := by
  sorry

end count_special_integers_l1507_150713


namespace inverse_variation_problem_l1507_150757

/-- Given that 3y varies inversely as the square of x, prove that y = 5/9 when x = 6, 
    given the initial condition y = 5 when x = 2 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 3 * y = k / (x^2)) →  -- Inverse variation relationship
  (3 * 5 = k / (2^2)) →                     -- Initial condition
  ∃ y : ℝ, 3 * y = k / (6^2) ∧ y = 5/9      -- Conclusion for x = 6
  := by sorry

end inverse_variation_problem_l1507_150757


namespace sales_after_reduction_profit_after_optimal_reduction_l1507_150775

/-- Represents a store's sales and pricing strategy -/
structure Store where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  min_profit : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase * price_reduction

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_profit - price_reduction

/-- Calculates the total daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  new_sales s price_reduction * new_profit_per_item s price_reduction

/-- The store's initial conditions -/
def my_store : Store :=
  { initial_sales := 20
  , initial_profit := 40
  , sales_increase := 2
  , min_profit := 25 }

theorem sales_after_reduction (s : Store) :
  new_sales s 3 = 26 :=
sorry

theorem profit_after_optimal_reduction (s : Store) :
  ∃ (x : ℝ), x = 10 ∧ 
    daily_profit s x = 1200 ∧ 
    new_profit_per_item s x ≥ s.min_profit :=
sorry

end sales_after_reduction_profit_after_optimal_reduction_l1507_150775


namespace no_linear_term_condition_l1507_150789

theorem no_linear_term_condition (x m : ℝ) : 
  (∀ a b c : ℝ, (x - m) * (x - 3) = a * x^2 + c → m = -3) ∧
  (m = -3 → ∃ a c : ℝ, (x - m) * (x - 3) = a * x^2 + c) :=
sorry

end no_linear_term_condition_l1507_150789


namespace count_divisible_by_33_l1507_150718

/-- Represents a 10-digit number of the form a2016b2017 -/
def NumberForm (a b : Nat) : Nat :=
  a * 10^9 + 2 * 10^8 + 0 * 10^7 + 1 * 10^6 + 6 * 10^5 + b * 10^4 + 2 * 10^3 + 0 * 10^2 + 1 * 10 + 7

/-- Predicate to check if a number is a single digit -/
def IsSingleDigit (n : Nat) : Prop := n < 10

/-- Main theorem -/
theorem count_divisible_by_33 :
  ∃! (count : Nat), ∃ (S : Finset (Nat × Nat)),
    (∀ (pair : Nat × Nat), pair ∈ S ↔ 
      IsSingleDigit pair.1 ∧ 
      IsSingleDigit pair.2 ∧ 
      (NumberForm pair.1 pair.2) % 33 = 0) ∧
    S.card = count ∧
    count = 3 := by sorry

end count_divisible_by_33_l1507_150718


namespace smallest_factor_l1507_150739

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 36 → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 12^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * 36) ∧ 3^3 ∣ (936 * 36) ∧ 12^2 ∣ (936 * 36)) := by
  sorry

end smallest_factor_l1507_150739


namespace smallest_k_divisible_by_24_l1507_150725

-- Define q as the largest prime with 2021 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2021 digits
axiom q_digits : 10^2020 ≤ q ∧ q < 10^2021

-- Theorem to prove
theorem smallest_k_divisible_by_24 :
  ∃ k : ℕ, k > 0 ∧ 24 ∣ (q^2 - k) ∧ ∀ m : ℕ, m > 0 → 24 ∣ (q^2 - m) → k ≤ m :=
sorry

end smallest_k_divisible_by_24_l1507_150725


namespace f_of_2_eq_0_l1507_150761

/-- The function f(x) = x^3 - 3x^2 + 2x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: f(2) = 0 -/
theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end f_of_2_eq_0_l1507_150761


namespace imaginary_part_of_one_minus_i_l1507_150792

theorem imaginary_part_of_one_minus_i :
  Complex.im (1 - Complex.I) = -1 := by sorry

end imaginary_part_of_one_minus_i_l1507_150792


namespace hyperbola_tangent_slope_range_l1507_150707

/-- The range of slopes for a line passing through the right focus of a hyperbola
    and intersecting its right branch at exactly one point. -/
theorem hyperbola_tangent_slope_range (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →  -- Equation of the hyperbola
  ∃ (m : ℝ), -- Slope of the line
    (m ≥ -Real.sqrt 3 ∧ m ≤ Real.sqrt 3) ∧ -- Range of slopes
    (∃ (x₀ y₀ : ℝ), -- Point of intersection
      x₀^2 / 4 - y₀^2 / 12 = 1 ∧ -- Point lies on the hyperbola
      y₀ = m * (x₀ - (Real.sqrt 5))) ∧ -- Line passes through right focus (√5, 0)
    (∀ (x₁ y₁ : ℝ), -- Uniqueness of intersection
      x₁ ≠ x₀ →
      x₁^2 / 4 - y₁^2 / 12 = 1 →
      y₁ ≠ m * (x₁ - (Real.sqrt 5))) :=
by sorry

end hyperbola_tangent_slope_range_l1507_150707


namespace betty_age_l1507_150791

theorem betty_age (carol alice betty : ℝ) 
  (h1 : carol = 5 * alice)
  (h2 : carol = 2 * betty)
  (h3 : alice = carol - 12) :
  betty = 7.5 := by
sorry

end betty_age_l1507_150791
