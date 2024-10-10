import Mathlib

namespace initial_sheep_count_l539_53967

theorem initial_sheep_count (horses : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) :
  horses = 100 →
  chickens = 9 →
  goats = 37 →
  male_animals = 53 →
  ∃ (sheep : ℕ), 
    (((horses + sheep + chickens) / 2 : ℚ) + goats : ℚ) = (2 * male_animals : ℚ) ∧
    sheep = 29 :=
by sorry

end initial_sheep_count_l539_53967


namespace product_of_numbers_l539_53988

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := by
  sorry

end product_of_numbers_l539_53988


namespace sum_remainder_three_l539_53999

theorem sum_remainder_three (m : ℤ) : (9 - m + (m + 4)) % 5 = 3 := by
  sorry

end sum_remainder_three_l539_53999


namespace two_colonies_reach_limit_same_time_l539_53958

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  growthRate : ℕ → ℕ
  limitDay : ℕ

/-- The number of days it takes for two colonies to reach the habitat's limit -/
def daysToLimitTwoColonies (colony : BacteriaColony) : ℕ := sorry

theorem two_colonies_reach_limit_same_time (colony : BacteriaColony) 
  (h1 : ∀ n : ℕ, colony.growthRate n = 2 * colony.growthRate (n - 1))
  (h2 : colony.limitDay = 16) :
  daysToLimitTwoColonies colony = colony.limitDay := by sorry

end two_colonies_reach_limit_same_time_l539_53958


namespace permutation_fraction_equality_l539_53937

def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem permutation_fraction_equality : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by sorry

end permutation_fraction_equality_l539_53937


namespace point_A_in_second_quadrant_l539_53993

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: The point A(-2, 3) is in the second quadrant -/
theorem point_A_in_second_quadrant :
  let A : Point2D := ⟨-2, 3⟩
  isInSecondQuadrant A := by
  sorry


end point_A_in_second_quadrant_l539_53993


namespace martin_bell_rings_l539_53935

theorem martin_bell_rings (s b : ℕ) : s = 4 + b / 3 → s + b = 52 → b = 36 := by sorry

end martin_bell_rings_l539_53935


namespace sum_of_extreme_prime_factors_of_1365_l539_53985

theorem sum_of_extreme_prime_factors_of_1365 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1365 ∧ largest ∣ 1365 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≥ smallest) ∧
    smallest + largest = 16 :=
by sorry

end sum_of_extreme_prime_factors_of_1365_l539_53985


namespace fraction_problem_l539_53933

theorem fraction_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 24.000000000000004) 
  (h2 : (1/4) * N = x * (N + 1) + 1) : 
  x = 0.20000000000000004 := by
  sorry

end fraction_problem_l539_53933


namespace fruit_purchase_cost_is_correct_l539_53976

/-- Calculates the total cost in dollars for a fruit purchase with given conditions -/
def fruitPurchaseCost (grapeKg : ℝ) (grapeRate : ℝ) (mangoKg : ℝ) (mangoRate : ℝ)
                      (appleKg : ℝ) (appleRate : ℝ) (orangeKg : ℝ) (orangeRate : ℝ)
                      (grapeMangoeDiscountRate : ℝ) (appleOrangeFixedDiscount : ℝ)
                      (salesTaxRate : ℝ) (fixedTax : ℝ) (exchangeRate : ℝ) : ℝ :=
  let grapeCost := grapeKg * grapeRate
  let mangoCost := mangoKg * mangoRate
  let appleCost := appleKg * appleRate
  let orangeCost := orangeKg * orangeRate
  let grapeMangoeTotal := grapeCost + mangoCost
  let appleOrangeTotal := appleCost + orangeCost
  let grapeMangoeDiscount := grapeMangoeTotal * grapeMangoeDiscountRate
  let discountedGrapeMangoe := grapeMangoeTotal - grapeMangoeDiscount
  let discountedAppleOrange := appleOrangeTotal - appleOrangeFixedDiscount
  let totalDiscountedCost := discountedGrapeMangoe + discountedAppleOrange
  let salesTax := totalDiscountedCost * salesTaxRate
  let totalTax := salesTax + fixedTax
  let totalAmount := totalDiscountedCost + totalTax
  totalAmount * exchangeRate

/-- Theorem stating that the fruit purchase cost under given conditions is $323.79 -/
theorem fruit_purchase_cost_is_correct :
  fruitPurchaseCost 7 68 9 48 5 55 4 38 0.1 25 0.05 15 0.25 = 323.79 := by
  sorry


end fruit_purchase_cost_is_correct_l539_53976


namespace quadratic_equation_real_roots_l539_53938

theorem quadratic_equation_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 + 2*x - k = 0 := by
  sorry

end quadratic_equation_real_roots_l539_53938


namespace problem_solution_l539_53901

theorem problem_solution (x y z t : ℝ) 
  (eq1 : x = y^2 - 16*x^2)
  (eq2 : y = z^2 - 4*x^2)
  (eq3 : z = t^2 - x^2)
  (eq4 : t = x - 1) :
  x = 1/9 := by
sorry

end problem_solution_l539_53901


namespace red_light_probability_is_two_fifths_l539_53961

/-- Represents the duration of each light color in seconds -/
structure LightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (d : LightDuration) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : LightDuration) : ℚ :=
  d.red / totalCycleTime d

/-- Theorem: The probability of seeing a red light is 2/5 given the specified light durations -/
theorem red_light_probability_is_two_fifths (d : LightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end red_light_probability_is_two_fifths_l539_53961


namespace office_printer_paper_duration_l539_53969

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that under the given conditions, the printer paper will last 6 days. -/
theorem office_printer_paper_duration :
  let packs : ℕ := 2
  let sheets_per_pack : ℕ := 240
  let daily_usage : ℕ := 80
  printer_paper_duration packs sheets_per_pack daily_usage = 6 := by
  sorry


end office_printer_paper_duration_l539_53969


namespace pages_per_chapter_l539_53950

theorem pages_per_chapter 
  (total_chapters : ℕ) 
  (total_pages : ℕ) 
  (h1 : total_chapters = 31) 
  (h2 : total_pages = 1891) :
  total_pages / total_chapters = 61 := by
sorry

end pages_per_chapter_l539_53950


namespace max_ab_given_extremum_l539_53907

/-- Given positive real numbers a and b, and a function f with an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_given_extremum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (a * b ≤ 9) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 9 ∧
    let f₀ := fun x => 4 * x^3 - a₀ * x^2 - 2 * b₀ * x + 2
    ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f₀ x ≤ f₀ 1 ∨ f₀ x ≥ f₀ 1) :=
by sorry


end max_ab_given_extremum_l539_53907


namespace hannah_cookies_sold_l539_53903

/-- Proves that Hannah sold 40 cookies given the conditions of the problem -/
theorem hannah_cookies_sold : ℕ :=
  let cookie_price : ℚ := 8 / 10
  let cupcake_price : ℚ := 2
  let cupcakes_sold : ℕ := 30
  let spoon_set_price : ℚ := 13 / 2
  let spoon_sets_bought : ℕ := 2
  let money_left : ℚ := 79

  let cookies_sold : ℕ := 40

  have h1 : cookie_price * cookies_sold + cupcake_price * cupcakes_sold = 
            spoon_set_price * spoon_sets_bought + money_left := by sorry

  cookies_sold


end hannah_cookies_sold_l539_53903


namespace fruit_purchase_cost_l539_53963

-- Define the cost of mangoes and oranges
def mango_cost : ℝ := 0.60
def orange_cost : ℝ := 0.40

-- Define the weight of mangoes and oranges Kelly buys
def mango_weight : ℝ := 5
def orange_weight : ℝ := 5

-- Define the discount percentage
def discount_rate : ℝ := 0.10

-- Define the function to calculate the total cost after discount
def total_cost_after_discount (m_cost o_cost m_weight o_weight disc_rate : ℝ) : ℝ :=
  let total_cost := (m_cost * 2 * m_weight) + (o_cost * 4 * o_weight)
  total_cost * (1 - disc_rate)

-- Theorem statement
theorem fruit_purchase_cost :
  total_cost_after_discount mango_cost orange_cost mango_weight orange_weight discount_rate = 12.60 := by
  sorry

end fruit_purchase_cost_l539_53963


namespace sequence_convergence_l539_53913

theorem sequence_convergence (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : 
  a 1 = 0 ∨ a 1 = 2 := by
  sorry

end sequence_convergence_l539_53913


namespace turnip_potato_ratio_l539_53941

theorem turnip_potato_ratio (total_potatoes : ℝ) (total_turnips : ℝ) (base_potatoes : ℝ) 
  (h1 : total_potatoes = 20)
  (h2 : total_turnips = 8)
  (h3 : base_potatoes = 5) :
  (base_potatoes / total_potatoes) * total_turnips = 2 := by
  sorry

end turnip_potato_ratio_l539_53941


namespace combine_numbers_to_24_l539_53954

theorem combine_numbers_to_24 : (10 * 10 - 4) / 4 = 24 := by
  sorry

end combine_numbers_to_24_l539_53954


namespace quadrilateral_area_theorem_l539_53979

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the signed area of a triangle
def signedArea (A B C : Point) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_area_theorem 
  (A B C D O K L : Point) 
  (h1 : K.x = (A.x + C.x) / 2 ∧ K.y = (A.y + C.y) / 2)  -- K is midpoint of AC
  (h2 : L.x = (B.x + D.x) / 2 ∧ L.y = (B.y + D.y) / 2)  -- L is midpoint of BD
  : (signedArea A O B) + (signedArea C O D) - 
    ((signedArea B O C) - (signedArea D O A)) = 
    4 * (signedArea K O L) := by sorry

end quadrilateral_area_theorem_l539_53979


namespace algorithm_output_is_36_l539_53981

def algorithm_result : ℕ := 
  let s := (List.range 3).foldl (fun acc i => acc + (i + 1)) 0
  let t := (List.range 3).foldl (fun acc i => acc * (i + 1)) 1
  s * t

theorem algorithm_output_is_36 : algorithm_result = 36 := by
  sorry

end algorithm_output_is_36_l539_53981


namespace smallest_student_count_l539_53928

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfiesRatios (gc : GradeCount) : Prop :=
  gc.eighth * 4 = gc.seventh * 7 ∧ gc.seventh * 9 = gc.sixth * 10

/-- Theorem stating the smallest possible total number of students --/
theorem smallest_student_count :
  ∃ (gc : GradeCount), satisfiesRatios gc ∧
    gc.eighth + gc.seventh + gc.sixth = 73 ∧
    (∀ (gc' : GradeCount), satisfiesRatios gc' →
      gc'.eighth + gc'.seventh + gc'.sixth ≥ 73) :=
by sorry

end smallest_student_count_l539_53928


namespace rectangles_on_4x3_grid_l539_53986

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of rectangles that can be formed on a grid. -/
def rectangles_on_grid (columns rows : ℕ) : ℕ :=
  binomial columns 2 * binomial rows 2

/-- Theorem: The number of rectangles on a 4x3 grid is 18. -/
theorem rectangles_on_4x3_grid : rectangles_on_grid 4 3 = 18 := by
  sorry

end rectangles_on_4x3_grid_l539_53986


namespace finite_quadruples_factorial_sum_l539_53955

theorem finite_quadruples_factorial_sum : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    ∀ (a b c n : ℕ), 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < n → 
      (n.factorial = a^(n-1) + b^(n-1) + c^(n-1)) → 
      (a, b, c, n) ∈ S := by
sorry

end finite_quadruples_factorial_sum_l539_53955


namespace candy_bar_cost_proof_l539_53923

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John used -/
def quarters_used : ℕ := 4

/-- The number of dimes John used -/
def dimes_used : ℕ := 3

/-- The number of nickels John used -/
def nickels_used : ℕ := 1

/-- The amount of change John received in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 131

theorem candy_bar_cost_proof :
  (quarters_used * quarter_value + dimes_used * dime_value + nickels_used * nickel_value) - change_received = candy_bar_cost :=
by sorry

end candy_bar_cost_proof_l539_53923


namespace solution_to_equation_l539_53971

theorem solution_to_equation (x : ℝ) (h : 1/4 - 1/5 = 1/x) : x = 20 := by
  sorry

end solution_to_equation_l539_53971


namespace stacy_paper_pages_per_day_l539_53972

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a 63-page paper due in 3 days,
    21 pages need to be written per day to finish on time. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 63 3 = 21 := by
  sorry

end stacy_paper_pages_per_day_l539_53972


namespace solution_set_f_nonnegative_range_of_a_l539_53957

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = Set.Iic (-3/5) ∪ Set.Ici 1 := by sorry

-- Theorem 2: Range of a given the condition
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) →
  a ∈ Set.Iic (-3) ∪ Set.Ici 1 := by sorry

end solution_set_f_nonnegative_range_of_a_l539_53957


namespace percentage_equality_l539_53980

theorem percentage_equality : ∃ P : ℝ, (P / 100) * 400 = (20 / 100) * 700 ∧ P = 35 := by
  sorry

end percentage_equality_l539_53980


namespace tournament_dominance_chain_l539_53902

/-- Represents a round-robin tournament with 8 players -/
structure Tournament :=
  (players : Finset (Fin 8))
  (defeated : Fin 8 → Fin 8 → Prop)
  (round_robin : ∀ i j, i ≠ j → (defeated i j ∨ defeated j i))
  (asymmetric : ∀ i j, defeated i j → ¬ defeated j i)

/-- The main theorem to be proved -/
theorem tournament_dominance_chain (t : Tournament) :
  ∃ (a b c d : Fin 8),
    a ∈ t.players ∧ b ∈ t.players ∧ c ∈ t.players ∧ d ∈ t.players ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.defeated a b ∧ t.defeated a c ∧ t.defeated a d ∧
    t.defeated b c ∧ t.defeated b d ∧
    t.defeated c d :=
sorry

end tournament_dominance_chain_l539_53902


namespace goods_trade_scientific_notation_l539_53964

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The value of one trillion -/
def trillion : ℝ := 1000000000000

theorem goods_trade_scientific_notation :
  to_scientific_notation (42.1 * trillion) =
    ScientificNotation.mk 4.21 13 (by norm_num) :=
  sorry

end goods_trade_scientific_notation_l539_53964


namespace largest_stamp_collection_l539_53956

theorem largest_stamp_collection (n : ℕ) (friends : ℕ) (extra : ℕ) : 
  friends = 15 →
  extra = 5 →
  n < 150 →
  n % friends = extra →
  ∀ m, m < 150 → m % friends = extra → m ≤ n →
  n = 140 :=
sorry

end largest_stamp_collection_l539_53956


namespace intersection_point_l539_53905

def line1 (x y : ℚ) : Prop := y = 3 * x + 1
def line2 (x y : ℚ) : Prop := y + 1 = -7 * x

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1/5, 2/5) := by
  sorry

end intersection_point_l539_53905


namespace small_cube_volume_ratio_l539_53991

/-- Given a larger cube composed of smaller cubes, this theorem proves
    the relationship between the volumes of the larger cube and each smaller cube. -/
theorem small_cube_volume_ratio (V_L V_S : ℝ) (h : V_L > 0) (h_cube : V_L = 125 * V_S) :
  V_S = V_L / 125 := by
  sorry

end small_cube_volume_ratio_l539_53991


namespace smallest_m_for_inequality_l539_53975

theorem smallest_m_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∀ m : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) → 
  m ≥ 27 :=
sorry

end smallest_m_for_inequality_l539_53975


namespace cube_intersection_figures_l539_53946

-- Define the set of possible plane figures
inductive PlaneFigure
| EquilateralTriangle
| Trapezoid
| RightAngledTriangle
| Rectangle

-- Define the set of plane figures that can be obtained from cube intersection
def CubeIntersectionFigures : Set PlaneFigure :=
  {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle}

-- Theorem statement
theorem cube_intersection_figures :
  CubeIntersectionFigures = {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle} :=
by sorry

end cube_intersection_figures_l539_53946


namespace cos_seven_pi_fourth_l539_53900

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end cos_seven_pi_fourth_l539_53900


namespace max_value_of_a_l539_53925

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) → 
  a ≤ -1 :=
sorry

end max_value_of_a_l539_53925


namespace factor_expression_l539_53973

theorem factor_expression (x : ℝ) : x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end factor_expression_l539_53973


namespace mike_weekly_pullups_l539_53916

/-- Calculates the number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_entry : ℕ) (entries_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_entry * entries_per_day * days_per_week

/-- Proves that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end mike_weekly_pullups_l539_53916


namespace balanced_sequence_equality_l539_53904

/-- A sequence of five real numbers is balanced if, when any one number is removed, 
    the remaining four can be divided into two groups of two numbers each 
    such that the sum of one group equals the sum of the other group. -/
def IsBalanced (a b c d e : ℝ) : Prop :=
  (b + c = d + e) ∧ (a + c = d + e) ∧ (a + b = d + e) ∧
  (a + c = b + e) ∧ (a + d = b + e)

theorem balanced_sequence_equality (a b c d e : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_balanced : IsBalanced a b c d e)
  (h_sum1 : e + c = b + d)
  (h_sum2 : e + a = c + d) :
  a = b ∧ b = c ∧ c = d ∧ d = e := by
  sorry

end balanced_sequence_equality_l539_53904


namespace determinant_of_specific_matrix_l539_53936

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
sorry

end determinant_of_specific_matrix_l539_53936


namespace original_sandbox_capacity_l539_53997

/-- Given a rectangular sandbox, this theorem proves that if a new sandbox with twice the dimensions
    has a capacity of 80 cubic feet, then the original sandbox has a capacity of 10 cubic feet. -/
theorem original_sandbox_capacity
  (length width height : ℝ)
  (new_sandbox_capacity : ℝ → ℝ → ℝ → ℝ)
  (h_new_sandbox : new_sandbox_capacity (2 * length) (2 * width) (2 * height) = 80) :
  length * width * height = 10 := by
  sorry

end original_sandbox_capacity_l539_53997


namespace go_match_probability_l539_53939

/-- The probability that two more games will conclude a Go match given the specified conditions -/
theorem go_match_probability (p_a : ℝ) (p_b : ℝ) : 
  p_a = 0.6 →
  p_b = 0.4 →
  p_a + p_b = 1 →
  (p_a ^ 2 + p_b ^ 2 : ℝ) = 0.52 := by
  sorry

end go_match_probability_l539_53939


namespace circle_area_from_circumference_l539_53960

/-- The area of a circle with circumference 31.4 meters is 246.49/π square meters -/
theorem circle_area_from_circumference :
  let circumference : ℝ := 31.4
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius^2
  area = 246.49 / Real.pi := by sorry

end circle_area_from_circumference_l539_53960


namespace church_attendance_l539_53914

theorem church_attendance (total people : ℕ) (children : ℕ) (female_adults : ℕ) :
  total = 200 →
  children = 80 →
  female_adults = 60 →
  total = children + female_adults + (total - children - female_adults) →
  total - children - female_adults = 60 :=
by
  sorry

end church_attendance_l539_53914


namespace concave_hexagon_guard_theorem_l539_53992

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave hexagon represented by its vertices -/
structure ConcaveHexagon where
  vertices : Fin 6 → Point
  is_concave : Bool

/-- Represents visibility between two points -/
def visible (p1 p2 : Point) (h : ConcaveHexagon) : Prop :=
  sorry

/-- A guard's position -/
structure Guard where
  position : Point

/-- Checks if a point is visible to at least one guard -/
def visible_to_guards (p : Point) (guards : List Guard) (h : ConcaveHexagon) : Prop :=
  ∃ g ∈ guards, visible g.position p h

theorem concave_hexagon_guard_theorem (h : ConcaveHexagon) :
  ∃ (guards : List Guard), guards.length ≤ 2 ∧
    ∀ (p : Point), (∃ i : Fin 6, p = h.vertices i) → visible_to_guards p guards h :=
  sorry

end concave_hexagon_guard_theorem_l539_53992


namespace smallest_area_ellipse_l539_53929

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- Checks if an ellipse contains a circle with center (h, 0) and radius 2 -/
def Ellipse.contains_circle (e : Ellipse) (h : ℝ) : Prop :=
  ∀ x y : ℝ, (x - h)^2 + y^2 = 4 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- The theorem stating the smallest possible area of the ellipse -/
theorem smallest_area_ellipse (e : Ellipse) 
  (h_contains_circle1 : e.contains_circle 2)
  (h_contains_circle2 : e.contains_circle (-2)) :
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    ∀ e' : Ellipse, e'.contains_circle 2 → e'.contains_circle (-2) → 
      π * e'.a * e'.b ≥ k * π :=
sorry

end smallest_area_ellipse_l539_53929


namespace sector_angle_unchanged_l539_53982

theorem sector_angle_unchanged 
  (r₁ r₂ : ℝ) 
  (s₁ s₂ : ℝ) 
  (θ₁ θ₂ : ℝ) 
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_radius : r₂ = 2 * r₁)
  (h_arc : s₂ = 2 * s₁)
  (h_angle₁ : s₁ = r₁ * θ₁)
  (h_angle₂ : s₂ = r₂ * θ₂) :
  θ₂ = θ₁ := by
sorry

end sector_angle_unchanged_l539_53982


namespace cost_of_72_tulips_l539_53989

/-- Represents the cost of tulips given the number of tulips -/
def tulip_cost (n : ℕ) : ℚ :=
  let base_cost := (36 : ℚ) * n / 18
  if n > 50 then base_cost * (1 - 1/5) else base_cost

/-- Theorem stating the cost of 72 tulips -/
theorem cost_of_72_tulips : tulip_cost 72 = 115.2 := by
  sorry

end cost_of_72_tulips_l539_53989


namespace fish_caught_difference_l539_53934

/-- Represents the number of fish caught by each fisherman -/
def fish_caught (season_length first_rate second_rate_1 second_rate_2 second_rate_3 : ℕ) 
  (second_period_1 second_period_2 : ℕ) : ℕ := 
  let first_total := first_rate * season_length
  let second_total := second_rate_1 * second_period_1 + 
                      second_rate_2 * second_period_2 + 
                      second_rate_3 * (season_length - second_period_1 - second_period_2)
  (max first_total second_total) - (min first_total second_total)

/-- The difference in fish caught between the two fishermen is 3 -/
theorem fish_caught_difference : 
  fish_caught 213 3 1 2 4 30 60 = 3 := by sorry

end fish_caught_difference_l539_53934


namespace perfect_square_difference_l539_53921

theorem perfect_square_difference (x y : ℝ) : (x - y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end perfect_square_difference_l539_53921


namespace f_properties_l539_53948

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), 0 < x → x ≤ π / 3 → 2 ≤ f x ∧ f x ≤ 3) ∧
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ ≤ π / 3 ∧ 0 < x₂ ∧ x₂ ≤ π / 3 ∧ f x₁ = 2 ∧ f x₂ = 3) :=
by sorry

end f_properties_l539_53948


namespace complex_equation_solution_l539_53926

theorem complex_equation_solution (i : ℂ) (m : ℝ) 
  (h1 : i ^ 2 = -1)
  (h2 : (2 : ℂ) / (1 + i) = 1 + m * i) : 
  m = -1 := by sorry

end complex_equation_solution_l539_53926


namespace golden_ratio_and_relations_l539_53994

theorem golden_ratio_and_relations :
  -- Part 1: Golden Ratio
  (∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2) ∧
  -- Part 2: Relation between a and b
  (∀ m a b : ℝ, a^2 + m*a = 1 → b^2 - 2*m*b = 4 → b ≠ -2*a → a*b = 2) ∧
  -- Part 3: Relation between p, q, and n
  (∀ n p q : ℝ, p ≠ q → p^2 + n*p - 1 = q → q^2 + n*q - 1 = p → p*q - n = 0) :=
by sorry

end golden_ratio_and_relations_l539_53994


namespace green_block_weight_l539_53943

/-- The weight of the yellow block in pounds -/
def yellow_weight : ℝ := 0.6

/-- The difference in weight between the yellow and green blocks in pounds -/
def weight_difference : ℝ := 0.2

/-- The weight of the green block in pounds -/
def green_weight : ℝ := yellow_weight - weight_difference

theorem green_block_weight : green_weight = 0.4 := by
  sorry

end green_block_weight_l539_53943


namespace afternoon_sales_l539_53906

/-- Represents the amount of pears sold by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning) 
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 510) : 
  sales.afternoon = 340 := by
  sorry

#check afternoon_sales

end afternoon_sales_l539_53906


namespace complex_number_location_l539_53908

/-- The complex number z = (2+i)/(1+i) is located in Quadrant IV -/
theorem complex_number_location :
  let z : ℂ := (2 + I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end complex_number_location_l539_53908


namespace exam_scores_l539_53995

theorem exam_scores (total_students : Nat) (high_scorers : Nat) (high_score : Nat) 
  (rest_average : Nat) (class_average : Nat) 
  (h1 : total_students = 25)
  (h2 : high_scorers = 3)
  (h3 : high_score = 95)
  (h4 : rest_average = 45)
  (h5 : class_average = 42) : 
  ∃ zero_scorers : Nat, 
    (zero_scorers + high_scorers + (total_students - zero_scorers - high_scorers)) = total_students ∧
    (high_scorers * high_score + (total_students - zero_scorers - high_scorers) * rest_average) 
      = (total_students * class_average) ∧
    zero_scorers = 5 := by
  sorry

end exam_scores_l539_53995


namespace z_in_first_quadrant_iff_m_gt_two_l539_53984

-- Define the complex number z
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  in_first_quadrant (z m) ↔ m > 2 := by sorry

end z_in_first_quadrant_iff_m_gt_two_l539_53984


namespace partition_naturals_with_property_l539_53919

theorem partition_naturals_with_property : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition b ∨ 
      partition a = partition c ∨ 
      partition b = partition c) := by sorry

end partition_naturals_with_property_l539_53919


namespace integral_roots_system_l539_53940

theorem integral_roots_system : ∃! (x y z : ℕ),
  (z^x = y^(3*x)) ∧
  (2^z = 8 * 4^x) ∧
  (x + y + z = 18) ∧
  x = 6 ∧ y = 2 ∧ z = 15 := by sorry

end integral_roots_system_l539_53940


namespace cereal_spending_ratio_is_two_to_one_l539_53922

/-- The ratio of Snap's spending to Crackle's spending on cereal -/
def cereal_spending_ratio : ℚ :=
  let total_spent : ℚ := 150
  let pop_spent : ℚ := 15
  let crackle_spent : ℚ := 3 * pop_spent
  let snap_spent : ℚ := total_spent - crackle_spent - pop_spent
  snap_spent / crackle_spent

/-- Theorem stating that the ratio of Snap's spending to Crackle's spending is 2:1 -/
theorem cereal_spending_ratio_is_two_to_one :
  cereal_spending_ratio = 2 := by
  sorry

end cereal_spending_ratio_is_two_to_one_l539_53922


namespace regular_polygon_interior_angle_l539_53917

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  let interior_angle := (n - 2) * 180 / n
  interior_angle = 135 → n = 8 := by
sorry

end regular_polygon_interior_angle_l539_53917


namespace z_ratio_equals_neg_i_l539_53962

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the condition that z₁ and z₂ are symmetric with respect to the imaginary axis
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

-- Theorem statement
theorem z_ratio_equals_neg_i
  (h_sym : symmetric_wrt_imaginary_axis z₁ z₂)
  (h_z₁ : z₁ = 1 + I) :
  z₁ / z₂ = -I :=
sorry

end z_ratio_equals_neg_i_l539_53962


namespace articles_sold_l539_53953

theorem articles_sold (cost_price : ℝ) (h : cost_price > 0) : 
  ∃ (N : ℕ), (20 : ℝ) * cost_price = N * (2 * cost_price) ∧ N = 10 :=
by sorry

end articles_sold_l539_53953


namespace missing_fraction_sum_l539_53927

theorem missing_fraction_sum (x : ℚ) : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (3/5 : ℚ) = (8/60 : ℚ) := by
  sorry

end missing_fraction_sum_l539_53927


namespace a_cubed_congruent_implies_a_sixth_congruent_l539_53945

theorem a_cubed_congruent_implies_a_sixth_congruent (n : ℕ+) (a : ℤ) 
  (h : a^3 ≡ 1 [ZMOD n]) : a^6 ≡ 1 [ZMOD n] := by
  sorry

end a_cubed_congruent_implies_a_sixth_congruent_l539_53945


namespace inequality_proof_l539_53998

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end inequality_proof_l539_53998


namespace negative_integer_squared_plus_self_equals_twelve_l539_53987

theorem negative_integer_squared_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end negative_integer_squared_plus_self_equals_twelve_l539_53987


namespace store_coloring_books_l539_53944

theorem store_coloring_books 
  (sold : ℕ) 
  (shelves : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : sold = 33) 
  (h2 : shelves = 9) 
  (h3 : books_per_shelf = 6) : 
  sold + shelves * books_per_shelf = 87 := by
  sorry

end store_coloring_books_l539_53944


namespace divisibility_by_three_and_nine_l539_53965

theorem divisibility_by_three_and_nine (n : ℕ) :
  (∃ (d₁ d₂ : ℕ), d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧ 
   (n * 10 + d₁) % 9 = 0 ∧ (n * 10 + d₂) % 9 = 0) →
  (∃! (digits : Finset ℕ), digits.card = 4 ∧ 
   ∀ d ∈ digits, d < 10 ∧ (n * 10 + d) % 3 = 0) :=
by sorry

end divisibility_by_three_and_nine_l539_53965


namespace triangle_arctan_sum_l539_53930

/-- Given a triangle ABC with sides a, b, c and angles α, β, γ in arithmetic progression
    with the smallest angle α = π/6, prove that arctan(a/(c+b)) + arctan(b/(c+a)) = π/4 -/
theorem triangle_arctan_sum (a b c : ℝ) (α β γ : ℝ) :
  α = π/6 →
  β = α + (γ - α)/2 →
  γ = α + 2*(γ - α)/2 →
  α + β + γ = π →
  a^2 + b^2 = c^2 →
  Real.arctan (a/(c+b)) + Real.arctan (b/(c+a)) = π/4 := by
  sorry

end triangle_arctan_sum_l539_53930


namespace vector_addition_l539_53924

theorem vector_addition (a b : ℝ × ℝ) :
  a = (5, -3) → b = (-6, 4) → a + b = (-1, 1) := by
  sorry

end vector_addition_l539_53924


namespace cat_mouse_position_after_299_moves_l539_53920

/-- Represents the four rooms for the cat --/
inductive CatRoom
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the eight segments for the mouse --/
inductive MouseSegment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- Function to determine cat's position after n moves --/
def catPosition (n : ℕ) : CatRoom :=
  match (n - n / 100) % 4 with
  | 0 => CatRoom.TopLeft
  | 1 => CatRoom.TopRight
  | 2 => CatRoom.BottomRight
  | _ => CatRoom.BottomLeft

/-- Function to determine mouse's position after n moves --/
def mousePosition (n : ℕ) : MouseSegment :=
  match n % 8 with
  | 0 => MouseSegment.TopLeft
  | 1 => MouseSegment.TopMiddle
  | 2 => MouseSegment.TopRight
  | 3 => MouseSegment.RightMiddle
  | 4 => MouseSegment.BottomRight
  | 5 => MouseSegment.BottomMiddle
  | 6 => MouseSegment.BottomLeft
  | _ => MouseSegment.LeftMiddle

theorem cat_mouse_position_after_299_moves :
  catPosition 299 = CatRoom.TopLeft ∧
  mousePosition 299 = MouseSegment.RightMiddle :=
by sorry

end cat_mouse_position_after_299_moves_l539_53920


namespace carries_strawberry_harvest_l539_53912

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected strawberry harvest based on garden dimensions and planting information -/
def expectedStrawberryHarvest (dimensions : GardenDimensions) (plantsPerSquareFoot : ℝ) (strawberriesPerPlant : ℝ) : ℝ :=
  dimensions.length * dimensions.width * plantsPerSquareFoot * strawberriesPerPlant

/-- Theorem stating that Carrie's garden will yield 1920 strawberries -/
theorem carries_strawberry_harvest :
  let dimensions : GardenDimensions := { length := 6, width := 8 }
  let plantsPerSquareFoot : ℝ := 4
  let strawberriesPerPlant : ℝ := 10
  expectedStrawberryHarvest dimensions plantsPerSquareFoot strawberriesPerPlant = 1920 := by
  sorry

end carries_strawberry_harvest_l539_53912


namespace problem_statement_l539_53959

def prop_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def prop_q (m : ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (prop_p m → m ∈ Set.Icc 1 2) ∧
  (¬(prop_p m ∧ prop_q m 1) ∧ (prop_p m ∨ prop_q m 1) →
    m < 1 ∨ (1 < m ∧ m ≤ 2)) := by sorry

end problem_statement_l539_53959


namespace registration_methods_count_l539_53977

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of extracurricular activity groups -/
def num_groups : ℕ := 3

/-- Each student must sign up for exactly one group -/
axiom one_group_per_student : True

/-- The total number of different registration methods -/
def total_registration_methods : ℕ := num_groups ^ num_students

theorem registration_methods_count :
  total_registration_methods = 3^4 :=
by sorry

end registration_methods_count_l539_53977


namespace certain_number_proof_l539_53996

theorem certain_number_proof (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end certain_number_proof_l539_53996


namespace sun_division_l539_53990

theorem sun_division (x y z total : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.3 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 45 →  -- y's share is Rs. 45
  total = x + y + z →  -- Total is the sum of all shares
  total = 175 :=  -- The total amount is Rs. 175
by sorry

end sun_division_l539_53990


namespace jesse_blocks_count_l539_53974

theorem jesse_blocks_count (building_blocks farm_blocks fence_blocks remaining_blocks : ℕ) 
  (h1 : building_blocks = 80)
  (h2 : farm_blocks = 123)
  (h3 : fence_blocks = 57)
  (h4 : remaining_blocks = 84) :
  building_blocks + farm_blocks + fence_blocks + remaining_blocks = 344 :=
by sorry

end jesse_blocks_count_l539_53974


namespace leak_drain_time_l539_53968

-- Define the pump filling rate
def pump_rate : ℚ := 1 / 2

-- Define the time to fill with leak
def fill_time_with_leak : ℚ := 7 / 3

-- Define the combined rate (pump - leak)
def combined_rate : ℚ := 1 / fill_time_with_leak

-- Define the leak rate
def leak_rate : ℚ := pump_rate - combined_rate

-- Theorem statement
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) 
  (combined_rate : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 7 / 3 →
  combined_rate = 1 / fill_time_with_leak →
  leak_rate = pump_rate - combined_rate →
  1 / leak_rate = 14 := by
  sorry

end leak_drain_time_l539_53968


namespace shaded_area_circle_configuration_l539_53949

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_circle_configuration (R : ℝ) (h : R = 8) : 
  R^2 * Real.pi - 3 * (R/2)^2 * Real.pi = 16 * Real.pi :=
by sorry

end shaded_area_circle_configuration_l539_53949


namespace system_implies_quadratic_l539_53951

theorem system_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y - 2 = 0) ∧ (3 * x + 2 * y - 6 = 0) →
  y^2 - 13 * y + 26 = 0 := by
  sorry

end system_implies_quadratic_l539_53951


namespace continued_fraction_equality_l539_53909

theorem continued_fraction_equality : 
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end continued_fraction_equality_l539_53909


namespace baker_cakes_sold_l539_53966

/-- The number of cakes a baker intends to sell given certain pricing conditions -/
theorem baker_cakes_sold (n : ℝ) (h1 : n > 0) : ∃ x : ℕ,
  (n * x = 320) ∧
  (0.8 * n * (x + 2) = 320) ∧
  (x = 8) := by
sorry

end baker_cakes_sold_l539_53966


namespace probability_for_specific_cube_l539_53983

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  total_cubes : ℕ
  full_face_painted : ℕ
  half_face_painted : ℕ

/-- Calculates the probability of selecting one cube with exactly one painted face
    and one cube with no painted faces when two cubes are randomly selected -/
def probability_one_painted_one_unpainted (cube : PaintedCube) : ℚ :=
  let one_face_painted := cube.full_face_painted - cube.half_face_painted
  let no_face_painted := cube.total_cubes - cube.full_face_painted - cube.half_face_painted
  let total_combinations := (cube.total_cubes * (cube.total_cubes - 1)) / 2
  let favorable_outcomes := one_face_painted * no_face_painted
  favorable_outcomes / total_combinations

/-- The main theorem stating the probability for the specific cube configuration -/
theorem probability_for_specific_cube : 
  let cube := PaintedCube.mk 5 125 25 12
  probability_one_painted_one_unpainted cube = 44 / 155 := by
  sorry

end probability_for_specific_cube_l539_53983


namespace candy_per_package_l539_53910

/-- Given that Robin has 45 packages of candy and 405 pieces of candies in total,
    prove that there are 9 pieces of candy in each package. -/
theorem candy_per_package (packages : ℕ) (total_pieces : ℕ) 
    (h1 : packages = 45) (h2 : total_pieces = 405) : 
    total_pieces / packages = 9 := by
  sorry

end candy_per_package_l539_53910


namespace range_of_x_l539_53947

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, x ≤ y → y ≤ 0 → f y ≤ f x
axiom f_at_neg_one : f (-1) = 1/2

-- State the theorem
theorem range_of_x (x : ℝ) : 2 * f (2*x - 1) - 1 < 0 ↔ 0 < x ∧ x < 1 := by sorry

end range_of_x_l539_53947


namespace birds_on_fence_l539_53978

theorem birds_on_fence (initial_birds landing_birds : ℕ) :
  initial_birds = 12 →
  landing_birds = 8 →
  initial_birds + landing_birds = 20 := by
sorry

end birds_on_fence_l539_53978


namespace extracurricular_materials_choice_l539_53915

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The total number of extracurricular reading materials -/
def totalMaterials : ℕ := 6

/-- The number of materials each student chooses -/
def materialsPerStudent : ℕ := 2

/-- The number of common materials between students -/
def commonMaterials : ℕ := 1

theorem extracurricular_materials_choice :
  (choose totalMaterials commonMaterials) *
  (arrange (totalMaterials - commonMaterials) (materialsPerStudent - commonMaterials)) = 120 := by
  sorry


end extracurricular_materials_choice_l539_53915


namespace smallest_power_congruence_l539_53970

/-- For any integer r ≥ 3, the smallest positive integer d₀ such that 7^d₀ ≡ 1 (mod 2^r) is 2^(r-2) -/
theorem smallest_power_congruence (r : ℕ) (hr : r ≥ 3) :
  (∃ (d₀ : ℕ), d₀ > 0 ∧ 7^d₀ ≡ 1 [MOD 2^r] ∧
    ∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) ∧
  (∀ (d₀ : ℕ), d₀ > 0 → 7^d₀ ≡ 1 [MOD 2^r] ∧
    (∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) →
    d₀ = 2^(r-2)) :=
by sorry

end smallest_power_congruence_l539_53970


namespace compound_interest_calculation_l539_53942

-- Define the initial amount
def initial_amount : ℚ := 6160

-- Define the interest rates
def interest_rate_year1 : ℚ := 10 / 100
def interest_rate_year2 : ℚ := 12 / 100

-- Define the function to calculate the amount after one year
def amount_after_one_year (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * (1 + rate)

-- Define the function to calculate the final amount after two years
def final_amount : ℚ :=
  amount_after_one_year (amount_after_one_year initial_amount interest_rate_year1) interest_rate_year2

-- State the theorem
theorem compound_interest_calculation :
  final_amount = 7589.12 := by sorry

end compound_interest_calculation_l539_53942


namespace chosen_number_proof_l539_53918

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end chosen_number_proof_l539_53918


namespace james_travel_time_l539_53952

-- Define the parameters
def driving_speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

-- Define the theorem
theorem james_travel_time :
  (distance / driving_speed) + stop_time = 7 :=
by sorry

end james_travel_time_l539_53952


namespace money_distribution_correctness_l539_53931

def bag_distribution : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 489]

def sum_subset (l : List Nat) (subset : List Bool) : Nat :=
  (l.zip subset).foldl (λ acc (x, b) => acc + if b then x else 0) 0

theorem money_distribution_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 1000 →
    ∃ subset : List Bool, subset.length = 10 ∧ sum_subset bag_distribution subset = n :=
by sorry

end money_distribution_correctness_l539_53931


namespace paths_count_is_36_l539_53932

/-- Represents the circular arrangement of numbers -/
structure CircularArrangement where
  center : Nat
  surrounding : Nat
  zeroAdjacent : Nat
  fiveAdjacent : Nat

/-- Calculates the number of distinct paths to form 2005 -/
def countPaths (arrangement : CircularArrangement) : Nat :=
  arrangement.surrounding * arrangement.zeroAdjacent * arrangement.fiveAdjacent

/-- The specific arrangement for the problem -/
def problemArrangement : CircularArrangement :=
  { center := 2
  , surrounding := 6
  , zeroAdjacent := 2
  , fiveAdjacent := 3 }

theorem paths_count_is_36 :
  countPaths problemArrangement = 36 := by
  sorry

end paths_count_is_36_l539_53932


namespace smallest_number_divisible_by_all_l539_53911

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 7) % 12 = 0 ∧
  (n - 7) % 16 = 0 ∧
  (n - 7) % 18 = 0 ∧
  (n - 7) % 21 = 0 ∧
  (n - 7) % 28 = 0 ∧
  (n - 7) % 35 = 0 ∧
  (n - 7) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65527 ∧
  ∀ m : ℕ, m < 65527 → ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l539_53911
