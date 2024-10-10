import Mathlib

namespace range_of_a_l198_19805

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 32) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = 32) →
  a ∈ Set.Icc 2 8 :=
by sorry

end range_of_a_l198_19805


namespace work_completion_time_l198_19854

/-- Given that 72 men can complete a piece of work in 18 days, and the number of men and days are inversely proportional, prove that 144 men will complete the same work in 9 days. -/
theorem work_completion_time 
  (men : ℕ → ℝ)
  (days : ℕ → ℝ)
  (h1 : men 1 = 72)
  (h2 : days 1 = 18)
  (h3 : ∀ k : ℕ, k > 0 → men k * days k = men 1 * days 1) :
  men 2 = 144 ∧ days 2 = 9 := by
sorry

end work_completion_time_l198_19854


namespace billie_bakes_three_pies_l198_19817

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := sorry

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_bakes_three_pies : 
  pies_per_day * baking_days = pies_eaten + cans_needed / cans_per_pie ∧ 
  pies_per_day = 3 := by sorry

end billie_bakes_three_pies_l198_19817


namespace smallest_angle_in_345_ratio_triangle_l198_19819

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end smallest_angle_in_345_ratio_triangle_l198_19819


namespace prime_power_plus_one_mod_240_l198_19829

theorem prime_power_plus_one_mod_240 (n : ℕ+) (h : Nat.Prime (2^n.val + 1)) :
  (2^n.val + 1) % 240 = 17 ∨ (2^n.val + 1) % 240 = 3 ∨ (2^n.val + 1) % 240 = 5 :=
by sorry

end prime_power_plus_one_mod_240_l198_19829


namespace partnership_profit_l198_19814

/-- A partnership problem with four partners A, B, C, and D -/
theorem partnership_profit (total_capital : ℝ) (total_profit : ℝ) : 
  (1 / 3 : ℝ) * total_capital / total_capital = 810 / total_profit →
  (1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ) + 
    (1 - ((1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ))) = 1 →
  total_profit = 2430 := by
  sorry

#check partnership_profit

end partnership_profit_l198_19814


namespace smallest_k_sum_squares_multiple_360_l198_19892

theorem smallest_k_sum_squares_multiple_360 : 
  ∃ k : ℕ+, (∀ m : ℕ+, m < k → ¬(∃ n : ℕ, m * (m + 1) * (2 * m + 1) = 6 * 360 * n)) ∧ 
  (∃ n : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 360 * n) ∧ 
  k = 432 := by
  sorry

end smallest_k_sum_squares_multiple_360_l198_19892


namespace system_solution_l198_19861

theorem system_solution :
  let eq1 (x y : ℝ) := x * (y - 1) + y * (x + 1) = 6
  let eq2 (x y : ℝ) := (x - 1) * (y + 1) = 1
  (eq1 (4/3) 2 ∧ eq2 (4/3) 2) ∧ (eq1 (-2) (-4/3) ∧ eq2 (-2) (-4/3)) :=
by sorry

end system_solution_l198_19861


namespace twin_primes_with_prime_expression_l198_19860

/-- Twin primes are prime numbers that differ by 2 -/
def TwinPrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p = q + 2 ∨ q = p + 2)

/-- The expression p^2 - pq + q^2 -/
def Expression (p q : ℕ) : ℕ :=
  p^2 - p*q + q^2

theorem twin_primes_with_prime_expression :
  ∀ p q : ℕ, TwinPrimes p q ∧ Nat.Prime (Expression p q) ↔ (p = 5 ∧ q = 3) ∨ (p = 3 ∧ q = 5) :=
sorry

end twin_primes_with_prime_expression_l198_19860


namespace meaningful_fraction_range_l198_19816

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 7)) ↔ x ≠ 7 := by
  sorry

end meaningful_fraction_range_l198_19816


namespace diamond_value_l198_19804

theorem diamond_value (diamond : ℕ) : 
  diamond < 10 →  -- Ensuring diamond is a digit
  (9 * diamond + 3 = 10 * diamond + 2) →  -- Equivalent to ◇3_9 = ◇2_10
  diamond = 1 := by
sorry

end diamond_value_l198_19804


namespace optimal_price_reduction_l198_19878

/-- Represents the daily profit function for a mall's product sales -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

/-- Theorem stating that a price reduction of $12 results in a daily profit of $3572 -/
theorem optimal_price_reduction (initial_sales : ℕ) (initial_profit : ℝ)
    (h1 : initial_sales = 70)
    (h2 : initial_profit = 50) :
    daily_profit initial_sales initial_profit 12 = 3572 := by
  sorry

end optimal_price_reduction_l198_19878


namespace stating_scale_theorem_l198_19885

/-- Represents a curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Applies a scaling transformation to a curve in the y-axis direction -/
def scale_y (c : Curve) (k : ℝ) : Curve :=
  { equation := λ x y => c.equation x (y / k) }

/-- The original curve x^2 - 4y^2 = 16 -/
def original_curve : Curve :=
  { equation := λ x y => x^2 - 4*y^2 = 16 }

/-- The transformed curve x^2 - y^2 = 16 -/
def transformed_curve : Curve :=
  { equation := λ x y => x^2 - y^2 = 16 }

/-- 
Theorem stating that scaling the original curve by factor 2 in the y-direction 
results in the transformed curve
-/
theorem scale_theorem : scale_y original_curve 2 = transformed_curve := by
  sorry

end stating_scale_theorem_l198_19885


namespace curve_is_two_semicircles_l198_19832

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  |x| - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def is_semicircle (center_x center_y radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ x ≥ center_x

-- Theorem statement
theorem curve_is_two_semicircles :
  ∀ x y : ℝ, curve_equation x y ↔
    (is_semicircle 1 1 1 x y ∨ is_semicircle (-1) 1 1 x y) :=
sorry

end curve_is_two_semicircles_l198_19832


namespace circular_garden_max_area_l198_19872

theorem circular_garden_max_area (fence_length : ℝ) (h : fence_length = 200) :
  let radius := fence_length / (2 * Real.pi)
  let area := Real.pi * radius ^ 2
  area = 10000 / Real.pi := by
  sorry

end circular_garden_max_area_l198_19872


namespace train_speed_calculation_l198_19826

theorem train_speed_calculation (rail_length : Real) (time_period : Real) : 
  rail_length = 40 ∧ time_period = 30 / 60 →
  ∃ (ε : Real), ε > 0 ∧ ∀ (train_speed : Real),
    train_speed > 0 →
    |train_speed - (train_speed * 5280 / 60 / rail_length * time_period)| < ε :=
by sorry

end train_speed_calculation_l198_19826


namespace arithmetic_square_root_of_one_fourth_l198_19830

theorem arithmetic_square_root_of_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end arithmetic_square_root_of_one_fourth_l198_19830


namespace train_distance_trains_ab_distance_l198_19896

/-- The distance between two trains' starting points given their speed and meeting point -/
theorem train_distance (speed : ℝ) (distance_a : ℝ) : speed > 0 → distance_a > 0 →
  2 * distance_a = (distance_a * speed + distance_a * speed) / speed := by
  sorry

/-- The specific problem of trains A and B -/
theorem trains_ab_distance : 
  let speed : ℝ := 50
  let distance_a : ℝ := 225
  2 * distance_a = 450 := by
  sorry

end train_distance_trains_ab_distance_l198_19896


namespace nested_sqrt_simplification_l198_19884

theorem nested_sqrt_simplification : 
  Real.sqrt (9 * Real.sqrt (27 * Real.sqrt 81)) = 9 * Real.sqrt 3 := by sorry

end nested_sqrt_simplification_l198_19884


namespace no_division_for_all_n_l198_19841

theorem no_division_for_all_n : ∀ n : ℕ, Nat.gcd (n + 2) (n^3 - 2*n^2 - 5*n + 7) = 1 := by
  sorry

end no_division_for_all_n_l198_19841


namespace product_expansion_terms_count_l198_19812

theorem product_expansion_terms_count :
  let a_terms := 3  -- number of terms in (a₁ + a₂ + a₃)
  let b_terms := 4  -- number of terms in (b₁ + b₂ + b₃ + b₄)
  let c_terms := 5  -- number of terms in (c₁ + c₂ + c₃ + c₄ + c₅)
  a_terms * b_terms * c_terms = 60 := by
sorry

end product_expansion_terms_count_l198_19812


namespace new_athlete_rate_is_15_l198_19869

/-- The rate at which new athletes arrived at the Ultimate Fitness Camp --/
def new_athlete_rate (initial_athletes : ℕ) (leaving_rate : ℕ) (leaving_hours : ℕ) 
  (arrival_hours : ℕ) (total_difference : ℕ) : ℕ :=
  let athletes_left := leaving_rate * leaving_hours
  let remaining_athletes := initial_athletes - athletes_left
  let final_athletes := initial_athletes - total_difference
  let new_athletes := final_athletes - remaining_athletes
  new_athletes / arrival_hours

/-- Theorem stating the rate at which new athletes arrived --/
theorem new_athlete_rate_is_15 : 
  new_athlete_rate 300 28 4 7 7 = 15 := by sorry

end new_athlete_rate_is_15_l198_19869


namespace factory_weekly_production_l198_19889

/-- Represents the production of toys in a factory --/
structure ToyProduction where
  days_per_week : ℕ
  toys_per_day : ℕ
  constant_daily_production : Bool

/-- Calculates the weekly toy production --/
def weekly_production (tp : ToyProduction) : ℕ :=
  tp.days_per_week * tp.toys_per_day

/-- Theorem stating the weekly toy production for the given factory --/
theorem factory_weekly_production :
  ∀ (tp : ToyProduction),
    tp.days_per_week = 4 →
    tp.toys_per_day = 1500 →
    tp.constant_daily_production →
    weekly_production tp = 6000 := by
  sorry

end factory_weekly_production_l198_19889


namespace equation_solution_l198_19873

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end equation_solution_l198_19873


namespace polynomial_divisibility_and_divisor_l198_19815

theorem polynomial_divisibility_and_divisor : ∃ m : ℤ,
  (∀ x : ℝ, (4 * x^2 - 6 * x + m) % (x - 3) = 0) ∧
  m = -18 ∧
  36 % m = 0 := by sorry

end polynomial_divisibility_and_divisor_l198_19815


namespace ball_radius_is_10_ball_surface_area_is_400pi_l198_19836

/-- Represents a spherical ball floating on water that leaves a circular hole in ice --/
structure FloatingBall where
  /-- The radius of the circular hole left in the ice --/
  holeRadius : ℝ
  /-- The depth of the hole left in the ice --/
  holeDepth : ℝ
  /-- The radius of the ball --/
  ballRadius : ℝ

/-- The properties of the floating ball problem --/
def floatingBallProblem : FloatingBall where
  holeRadius := 6
  holeDepth := 2
  ballRadius := 10

/-- Theorem stating that the radius of the ball is 10 cm --/
theorem ball_radius_is_10 (ball : FloatingBall) :
  ball.holeRadius = 6 ∧ ball.holeDepth = 2 → ball.ballRadius = 10 := by sorry

/-- Theorem stating that the surface area of the ball is 400π cm² --/
theorem ball_surface_area_is_400pi (ball : FloatingBall) :
  ball.ballRadius = 10 → 4 * Real.pi * ball.ballRadius ^ 2 = 400 * Real.pi := by sorry

end ball_radius_is_10_ball_surface_area_is_400pi_l198_19836


namespace quadratic_roots_nm_l198_19820

theorem quadratic_roots_nm (m n : ℝ) : 
  (∀ x, 2 * x^2 + m * x + n = 0 ↔ x = -2 ∨ x = 1) → 
  n^m = 16 := by
  sorry

end quadratic_roots_nm_l198_19820


namespace necessary_not_sufficient_for_ellipse_l198_19831

/-- Predicate to determine if an equation represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that a > 0 and b > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem necessary_not_sufficient_for_ellipse :
  (∀ a b : ℝ, is_ellipse a b → a > 0 ∧ b > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ¬is_ellipse a b) :=
sorry

end necessary_not_sufficient_for_ellipse_l198_19831


namespace quadratic_form_equivalence_l198_19824

theorem quadratic_form_equivalence (x : ℝ) : 
  (2*x - 1) * (x + 2) + 1 = 2*(x + 3/4)^2 - 17/8 := by
  sorry

end quadratic_form_equivalence_l198_19824


namespace cheryl_walking_distance_l198_19867

/-- Calculates the total distance Cheryl walked based on her journey segments -/
def total_distance_walked (
  speed1 : ℝ) (time1 : ℝ)
  (speed2 : ℝ) (time2 : ℝ)
  (speed3 : ℝ) (time3 : ℝ)
  (speed4 : ℝ) (time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that Cheryl's total walking distance is 32 miles -/
theorem cheryl_walking_distance :
  total_distance_walked 2 3 4 2 1 3 3 5 = 32 := by
  sorry

#eval total_distance_walked 2 3 4 2 1 3 3 5

end cheryl_walking_distance_l198_19867


namespace equidistant_point_sum_of_distances_equidistant_times_l198_19833

-- Define the points A and B
def A : ℝ := -2
def B : ℝ := 4

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the distances from P to A and B
def distPA (x : ℝ) : ℝ := |x - A|
def distPB (x : ℝ) : ℝ := |x - B|

-- Define the positions of M and N after t seconds
def M (t : ℝ) : ℝ := A - t
def N (t : ℝ) : ℝ := B - 3*t

-- Define the origin O
def O : ℝ := 0

-- Theorem 1: The point equidistant from A and B
theorem equidistant_point : ∃ x : ℝ, distPA x = distPB x ∧ x = 1 := by sorry

-- Theorem 2: Points where sum of distances from A and B is 8
theorem sum_of_distances : ∃ x₁ x₂ : ℝ, 
  distPA x₁ + distPB x₁ = 8 ∧ 
  distPA x₂ + distPB x₂ = 8 ∧ 
  x₁ = -3 ∧ x₂ = 5 := by sorry

-- Theorem 3: Times when one point is equidistant from the other two
theorem equidistant_times : ∃ t₁ t₂ t₃ t₄ t₅ : ℝ,
  (|M t₁| = |N t₁| ∧ t₁ = 1/2) ∧
  (N t₂ = O ∧ t₂ = 4/3) ∧
  (|N t₃ - O| = |N t₃ - M t₃| ∧ t₃ = 2) ∧
  (M t₄ = N t₄ ∧ t₄ = 3) ∧
  (|M t₅ - O| = |M t₅ - N t₅| ∧ t₅ = 8) := by sorry

end equidistant_point_sum_of_distances_equidistant_times_l198_19833


namespace sin_2alpha_value_l198_19862

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) : 
  Real.sin (2 * α) = -5/9 := by
  sorry

end sin_2alpha_value_l198_19862


namespace product_expression_value_l198_19864

def product_expression : ℚ :=
  (3^3 - 2^3) / (3^3 + 2^3) *
  (4^3 - 3^3) / (4^3 + 3^3) *
  (5^3 - 4^3) / (5^3 + 4^3) *
  (6^3 - 5^3) / (6^3 + 5^3) *
  (7^3 - 6^3) / (7^3 + 6^3)

theorem product_expression_value : product_expression = 17 / 901 := by
  sorry

end product_expression_value_l198_19864


namespace triangle_abc_properties_l198_19898

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  B = π / 3 ∧ a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 :=
by sorry

end triangle_abc_properties_l198_19898


namespace lines_parallel_or_skew_l198_19850

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subsetLinePlane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the skew relation for lines
variable (skewLines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_or_skew
  (a b : Line) (α β : Plane)
  (h_diff_lines : a ≠ b)
  (h_diff_planes : α ≠ β)
  (h_parallel_planes : parallelPlanes α β)
  (h_a_in_α : subsetLinePlane a α)
  (h_b_in_β : subsetLinePlane b β) :
  parallelLines a b ∨ skewLines a b :=
sorry

end lines_parallel_or_skew_l198_19850


namespace distance_to_point_l198_19800

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 6

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The distance between two points in 2D space -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance between the center of the circle and (10, 3) is √68 -/
theorem distance_to_point : distance circle_center (10, 3) = Real.sqrt 68 := by sorry

end distance_to_point_l198_19800


namespace average_price_theorem_l198_19846

theorem average_price_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  let p := (2 * a * b) / (a + b)
  a < p ∧ p < Real.sqrt (a * b) := by
  sorry

end average_price_theorem_l198_19846


namespace fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l198_19875

/-- Represents a survey option -/
inductive SurveyOption
  | A
  | B
  | C
  | D

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresPrecision : Bool
  easyToConduct : Bool
  nonDestructive : Bool
  manageableSubjects : Bool

/-- Defines the characteristics of a comprehensive survey method -/
def isComprehensiveSurvey (c : SurveyCharacteristics) : Prop :=
  c.requiresPrecision ∧ c.easyToConduct ∧ c.nonDestructive ∧ c.manageableSubjects

/-- Characteristics of the 50-meter dash survey -/
def fiftyMeterDashSurvey : SurveyCharacteristics :=
  { requiresPrecision := true
    easyToConduct := true
    nonDestructive := true
    manageableSubjects := true }

/-- Theorem stating that the 50-meter dash survey is suitable for a comprehensive survey method -/
theorem fiftyMeterDashIsSuitable : isComprehensiveSurvey fiftyMeterDashSurvey :=
  sorry

/-- Function to determine the suitable survey option -/
def suitableSurveyOption : SurveyOption :=
  SurveyOption.A

/-- Theorem stating that the suitable survey option is correct -/
theorem suitableSurveyIsCorrect : suitableSurveyOption = SurveyOption.A :=
  sorry

end fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l198_19875


namespace last_remaining_number_l198_19808

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else
  let m := markingProcess (n / 2)
  if m * 2 > n then 2 * m - 1 else 2 * m + 1

/-- The theorem stating that for 120 numbers, the last remaining number is 64 -/
theorem last_remaining_number :
  markingProcess 120 = 64 := by
  sorry

end last_remaining_number_l198_19808


namespace james_socks_l198_19807

theorem james_socks (red_pairs : ℕ) (black : ℕ) (white : ℕ) : 
  black = red_pairs -- number of black socks is equal to the number of pairs of red socks
  → white = 2 * (2 * red_pairs + black) -- number of white socks is twice the number of red and black socks combined
  → 2 * red_pairs + black + white = 90 -- total number of socks is 90
  → red_pairs = 10 := by
  sorry

end james_socks_l198_19807


namespace odd_monotone_function_range_theorem_l198_19827

/-- A function that is odd and monotonically increasing on non-negative reals -/
def OddMonotoneFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

/-- The theorem statement -/
theorem odd_monotone_function_range_theorem (f : ℝ → ℝ) (h : OddMonotoneFunction f) :
  {x : ℝ | f (x^2 - x - 1) < f 5} = Set.Ioo (-2) 3 := by
  sorry

end odd_monotone_function_range_theorem_l198_19827


namespace abc_sum_sixteen_l198_19857

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by
sorry

end abc_sum_sixteen_l198_19857


namespace no_solution_for_system_l198_19893

theorem no_solution_for_system :
  ¬∃ (x y : ℝ), (2 * x - 3 * y = 7) ∧ (4 * x - 6 * y = 20) := by
  sorry

end no_solution_for_system_l198_19893


namespace sequence_sum_l198_19865

theorem sequence_sum (A B C D E F G H : ℝ) 
  (h1 : C = 7)
  (h2 : ∀ (X Y Z : ℝ), (X = A ∧ Y = B ∧ Z = C) ∨ 
                        (X = B ∧ Y = C ∧ Z = D) ∨ 
                        (X = C ∧ Y = D ∧ Z = E) ∨ 
                        (X = D ∧ Y = E ∧ Z = F) ∨ 
                        (X = E ∧ Y = F ∧ Z = G) ∨ 
                        (X = F ∧ Y = G ∧ Z = H) → X + Y + Z = 36) : 
  A + H = 29 := by sorry

end sequence_sum_l198_19865


namespace rectangle_perimeter_l198_19891

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23)
  (h2 : L * B = 2520) :
  2 * (L + B) = 206 := by
  sorry

end rectangle_perimeter_l198_19891


namespace g_100_zeros_l198_19813

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ g 100 a = 0 ∧ g 100 b = 0 ∧
  ∀ (x : ℝ), g 100 x = 0 → x = a ∨ x = b :=
sorry

end g_100_zeros_l198_19813


namespace percentage_of_360_l198_19858

theorem percentage_of_360 : (32 / 100) * 360 = 115.2 := by sorry

end percentage_of_360_l198_19858


namespace constant_term_expansion_l198_19839

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + x + 5) * (x^5 + x^3 + 15) = x^9 + x^7 + 15*x^4 + x^6 + x^4 + 15*x + 5*x^5 + 5*x^3 + 75 := by
  sorry

#check constant_term_expansion

end constant_term_expansion_l198_19839


namespace optimal_road_network_l198_19838

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a configuration of 4 observation stations -/
structure Configuration where
  stations : Fin 4 → Point
  valid : ∀ i, 0 ≤ (stations i).x ∧ (stations i).x ≤ 10 ∧ 0 ≤ (stations i).y ∧ (stations i).y ≤ 10

/-- Represents a network of roads -/
structure RoadNetwork where
  horizontal : List ℝ  -- y-coordinates of horizontal roads
  vertical : List ℝ    -- x-coordinates of vertical roads

/-- Checks if a road network connects all stations to both top and bottom edges -/
def connects (c : Configuration) (n : RoadNetwork) : Prop :=
  ∀ i, ∃ h v,
    h ∈ n.horizontal ∧ v ∈ n.vertical ∧
    ((c.stations i).x = v ∨ (c.stations i).y = h)

/-- Calculates the total length of a road network -/
def networkLength (n : RoadNetwork) : ℝ :=
  (n.horizontal.length * 10 : ℝ) + (n.vertical.sum : ℝ)

/-- The main theorem to be proved -/
theorem optimal_road_network :
  (∀ c : Configuration, ∃ n : RoadNetwork, connects c n ∧ networkLength n ≤ 25) ∧
  (∀ ε > 0, ∃ c : Configuration, ∀ n : RoadNetwork, connects c n → networkLength n > 25 - ε) :=
sorry

end optimal_road_network_l198_19838


namespace no_natural_solution_l198_19809

theorem no_natural_solution : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_solution_l198_19809


namespace election_winner_votes_l198_19859

/-- In an election with two candidates, where the winner received 62% of votes
    and won by 408 votes, the number of votes cast for the winning candidate is 1054. -/
theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 408 →
  (total_votes : ℝ) * 0.62 = 1054 := by
  sorry

end election_winner_votes_l198_19859


namespace three_greater_than_sqrt_seven_l198_19874

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by sorry

end three_greater_than_sqrt_seven_l198_19874


namespace quadratic_equivalence_l198_19877

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 + 2*x + 4 ↔ y = (x + 1)^2 + 3 := by sorry

end quadratic_equivalence_l198_19877


namespace max_rotation_surface_area_l198_19855

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  r : ℝ  -- radius of the circumscribed circle
  A : ℝ × ℝ  -- coordinates of point A
  B : ℝ × ℝ  -- coordinates of point B
  C : ℝ × ℝ  -- coordinates of point C

/-- Calculates the surface area generated by rotating side BC around the tangent at A -/
def rotationSurfaceArea (triangle : InscribedTriangle) : ℝ :=
  sorry

/-- Theorem: The maximum surface area generated by rotating side BC of an inscribed triangle
    around the tangent at A is achieved when the triangle is equilateral and equals 3r²π√3 -/
theorem max_rotation_surface_area (triangle : InscribedTriangle) :
  rotationSurfaceArea triangle ≤ 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ∧
  (rotationSurfaceArea triangle = 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ↔
   triangle.A.1^2 + triangle.A.2^2 = triangle.r^2 ∧
   triangle.B.1^2 + triangle.B.2^2 = triangle.r^2 ∧
   triangle.C.1^2 + triangle.C.2^2 = triangle.r^2 ∧
   (triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2 ∧
   (triangle.A.1 - triangle.C.1)^2 + (triangle.A.2 - triangle.C.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) :=
by sorry


end max_rotation_surface_area_l198_19855


namespace max_ab_is_nine_l198_19823

/-- The function f(x) defined in the problem -/
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

/-- The second derivative of f(x) -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 24 * x - 2 * a

theorem max_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f' a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end max_ab_is_nine_l198_19823


namespace groups_formed_equals_seven_l198_19835

/-- Given a class with boys and girls, and a group size, calculate the number of groups formed. -/
def calculateGroups (boys : ℕ) (girls : ℕ) (groupSize : ℕ) : ℕ :=
  (boys + girls) / groupSize

/-- Theorem: Given 9 boys, 12 girls, and groups of 3 members, 7 groups are formed. -/
theorem groups_formed_equals_seven :
  calculateGroups 9 12 3 = 7 := by
  sorry

end groups_formed_equals_seven_l198_19835


namespace prize_stickers_l198_19899

/-- The number of stickers Christine already has -/
def current_stickers : ℕ := 11

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := 19

/-- The total number of stickers needed for a prize -/
def total_stickers : ℕ := current_stickers + additional_stickers

theorem prize_stickers : total_stickers = 30 := by sorry

end prize_stickers_l198_19899


namespace miss_spelling_paper_sheets_l198_19851

theorem miss_spelling_paper_sheets : ∃ (total_sheets : ℕ) (num_pupils : ℕ),
  total_sheets = 3 * num_pupils + 31 ∧
  total_sheets = 4 * num_pupils + 8 ∧
  total_sheets = 100 :=
by
  sorry

end miss_spelling_paper_sheets_l198_19851


namespace determinant_scaling_l198_19863

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![2 * a, 2 * b], ![2 * c, 2 * d]] = 20 := by
  sorry

end determinant_scaling_l198_19863


namespace impossibility_of_transformation_l198_19853

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_bound : a < 10
  b_bound : b < 10
  c_bound : c < 10
  d_bound : d < 10

/-- The invariant quantity M for a four-digit number --/
def invariant_M (n : FourDigitNumber) : Int :=
  (n.d + n.b) - (n.a + n.c)

/-- The allowed operations on four-digit numbers --/
inductive Operation
  | AddAdjacent (i : Fin 3)
  | SubtractAdjacent (i : Fin 3)

/-- Applying an operation to a four-digit number --/
def apply_operation (n : FourDigitNumber) (op : Operation) : Option FourDigitNumber :=
  sorry

/-- The main theorem: it's impossible to transform 1234 into 2002 --/
theorem impossibility_of_transformation :
  ∀ (ops : List Operation),
    let start := FourDigitNumber.mk 1 2 3 4 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    let target := FourDigitNumber.mk 2 0 0 2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ∀ (result : FourDigitNumber),
      (ops.foldl (fun n op => (apply_operation n op).getD n) start = result) →
      result ≠ target :=
by
  sorry

end impossibility_of_transformation_l198_19853


namespace domino_pile_sum_theorem_l198_19866

/-- Definition of a domino set -/
def DominoSet := { n : ℕ | n ≤ 28 }

/-- The total sum of points on all domino pieces -/
def totalSum : ℕ := 168

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if four numbers are consecutive -/
def areConsecutive (a b c d : ℕ) : Prop := b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem to be proved -/
theorem domino_pile_sum_theorem :
  ∃ (a b c d : ℕ), 
    isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
    areConsecutive a b c d ∧
    a + b + c + d = totalSum :=
sorry

end domino_pile_sum_theorem_l198_19866


namespace parabola_equation_l198_19897

/-- A parabola with vertex at the origin and directrix x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = kx -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The directrix of the parabola has equation x = 2 -/
  directrix_at_two : ∀ y, ¬ equation 2 y

/-- The equation of the parabola is y² = -16x -/
theorem parabola_equation (C : Parabola) : 
  C.equation = fun x y ↦ y^2 = -16*x := by sorry

end parabola_equation_l198_19897


namespace fill_time_AB_is_2_4_hours_l198_19848

-- Define the constants for the fill times
def fill_time_ABC : ℝ := 2
def fill_time_AC : ℝ := 3
def fill_time_BC : ℝ := 4

-- Define the rates of water flow for each valve
def rate_A : ℝ := sorry
def rate_B : ℝ := sorry
def rate_C : ℝ := sorry

-- Define the volume of the tank
def tank_volume : ℝ := sorry

-- Theorem to prove
theorem fill_time_AB_is_2_4_hours : 
  tank_volume / (rate_A + rate_B) = 2.4 := by sorry

end fill_time_AB_is_2_4_hours_l198_19848


namespace abs_reciprocal_neg_six_l198_19887

theorem abs_reciprocal_neg_six : |1 / (-6)| = 1 / 6 := by
  sorry

end abs_reciprocal_neg_six_l198_19887


namespace pentagon_area_sum_l198_19825

/-- Given two integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u,v)
    B is the reflection of A across y = x
    C is the reflection of B across y = -x
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 615, then u + v = 45. -/
theorem pentagon_area_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v) : 
  let A := (u, v)
  let B := (v, u)
  let C := (-u, v)
  let D := (-u, -v)
  let E := (u, -v)
  let area := u^2 + 3*u*v
  area = 615 → u + v = 45 := by sorry

end pentagon_area_sum_l198_19825


namespace find_carols_number_l198_19821

/-- A prime number between 10 and 99, inclusive. -/
def TwoDigitPrime := {p : Nat // p.Prime ∧ 10 ≤ p ∧ p ≤ 99}

/-- The problem statement -/
theorem find_carols_number 
  (a b c : TwoDigitPrime) 
  (h1 : b.val + c.val = 14)
  (h2 : a.val + c.val = 20)
  (h3 : a.val + b.val = 18)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  c.val = 11 := by
  sorry

#check find_carols_number

end find_carols_number_l198_19821


namespace at_least_one_leq_neg_two_l198_19886

theorem at_least_one_leq_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by sorry

end at_least_one_leq_neg_two_l198_19886


namespace complex_purely_imaginary_implies_a_eq_one_l198_19849

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is nonzero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from the real number a. -/
def complexNumber (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 2⟩

/-- If the complex number ((a^2 - 3a + 2) + (a - 2)i) is purely imaginary, then a = 1. -/
theorem complex_purely_imaginary_implies_a_eq_one (a : ℝ) :
  isPurelyImaginary (complexNumber a) → a = 1 := by
  sorry


end complex_purely_imaginary_implies_a_eq_one_l198_19849


namespace triangle_properties_l198_19856

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle)
  (h : t.a / (Real.cos t.C * Real.sin t.B) = t.b / Real.sin t.B + t.c / Real.cos t.C) :
  t.B = π / 4 ∧
  (t.b = Real.sqrt 2 → 
    ∀ (area : ℝ), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ (Real.sqrt 2 + 1) / 2) :=
by sorry

end triangle_properties_l198_19856


namespace pants_price_calculation_l198_19871

/-- Given 10 pairs of pants with a 20% discount, followed by a 10% tax,
    resulting in a final price of $396, prove that the original retail
    price of each pair of pants is $45. -/
theorem pants_price_calculation (quantity : Nat) (discount_rate : Real)
    (tax_rate : Real) (final_price : Real) :
  quantity = 10 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  final_price = 396 →
  ∃ (original_price : Real),
    original_price = 45 ∧
    final_price = quantity * original_price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

end pants_price_calculation_l198_19871


namespace yellow_red_paper_area_comparison_l198_19828

theorem yellow_red_paper_area_comparison (x : ℝ) (h : x > 0) :
  let yellow_area := 2 * x
  let larger_part := x / (1 - 0.25)
  let smaller_part := yellow_area - larger_part
  (x - smaller_part) / smaller_part = 0.5
  := by sorry

end yellow_red_paper_area_comparison_l198_19828


namespace square_2004_content_l198_19882

/-- Represents the content of a square in the sequence -/
inductive SquareContent
  | A
  | AB
  | ABCD
  | Number (n : ℕ)

/-- Returns the letter content of the nth square -/
def letterContent (n : ℕ) : SquareContent :=
  match n % 3 with
  | 0 => SquareContent.ABCD
  | 1 => SquareContent.A
  | 2 => SquareContent.AB
  | _ => SquareContent.A  -- This case is mathematically impossible, but needed for completeness

/-- Returns the number content of the nth square -/
def numberContent (n : ℕ) : SquareContent :=
  SquareContent.Number n

/-- Combines letter and number content for the nth square -/
def squareContent (n : ℕ) : (SquareContent × SquareContent) :=
  (letterContent n, numberContent n)

/-- The main theorem to prove -/
theorem square_2004_content :
  squareContent 2004 = (SquareContent.ABCD, SquareContent.Number 2004) := by
  sorry


end square_2004_content_l198_19882


namespace parabola_intersection_condition_l198_19852

theorem parabola_intersection_condition (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end parabola_intersection_condition_l198_19852


namespace girls_in_class_l198_19842

/-- In a class with the following properties:
    - There are 18 boys over 160 cm tall
    - These 18 boys constitute 3/4 of all boys
    - The total number of boys is 2/3 of the total number of students
    Then the number of girls in the class is 12 -/
theorem girls_in_class (tall_boys : ℕ) (total_boys : ℕ) (total_students : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = (3 / 4 : ℚ) * total_boys)
  (h3 : total_boys = (2 / 3 : ℚ) * total_students) :
  total_students - total_boys = 12 := by
  sorry

#check girls_in_class

end girls_in_class_l198_19842


namespace profit_and_maximum_l198_19801

noncomputable section

-- Define the sales volume function
def p (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def y (x : ℝ) : ℝ := 16 - 4 / (x + 1) - x

-- Theorem for the profit function and its maximum
theorem profit_and_maximum (a : ℝ) (h_a : a > 0) :
  -- The profit function
  (∀ x, 0 ≤ x ∧ x ≤ a → y x = 16 - 4 / (x + 1) - x) ∧
  -- Maximum profit when a ≥ 1
  (a ≥ 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 13 ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) ∧
  -- Maximum profit when a < 1
  (a < 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 16 - 4 / (a + 1) - a ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) :=
sorry

end

end profit_and_maximum_l198_19801


namespace product_of_numbers_l198_19822

theorem product_of_numbers (x y : ℝ) : x + y = 25 ∧ x - y = 7 → x * y = 144 := by sorry

end product_of_numbers_l198_19822


namespace johnnys_hourly_wage_l198_19844

/-- Johnny's hourly wage calculation --/
theorem johnnys_hourly_wage :
  let total_earned : ℚ := 11.75
  let hours_worked : ℕ := 5
  let hourly_wage : ℚ := total_earned / hours_worked
  hourly_wage = 2.35 := by
  sorry

end johnnys_hourly_wage_l198_19844


namespace speed_ratio_inverse_of_time_ratio_l198_19811

/-- Proves that the ratio of speeds for two runners completing the same race
    is the inverse of the ratio of their completion times. -/
theorem speed_ratio_inverse_of_time_ratio
  (total_time : ℝ)
  (rickey_time : ℝ)
  (prejean_time : ℝ)
  (h1 : total_time = rickey_time + prejean_time)
  (h2 : rickey_time = 40)
  (h3 : total_time = 70)
  : (prejean_time / rickey_time) = (3 : ℝ) / 4 := by
  sorry

end speed_ratio_inverse_of_time_ratio_l198_19811


namespace kaleb_book_count_l198_19868

theorem kaleb_book_count (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by
  sorry

end kaleb_book_count_l198_19868


namespace a_range_l198_19881

def sequence_a (a : ℝ) : ℕ+ → ℝ
  | ⟨1, _⟩ => a
  | ⟨n+1, _⟩ => 4*(n+1) + (-1)^(n+1) * (8 - 2*a)

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, sequence_a a n < sequence_a a (n + 1)) →
  (3 < a ∧ a < 5) :=
by sorry

end a_range_l198_19881


namespace count_divisors_of_360_l198_19834

theorem count_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end count_divisors_of_360_l198_19834


namespace quadratic_roots_sum_of_cubes_reciprocals_l198_19847

theorem quadratic_roots_sum_of_cubes_reciprocals 
  (a b c r s : ℝ) 
  (h1 : 3 * a * r^2 + 5 * b * r + 7 * c = 0) 
  (h2 : 3 * a * s^2 + 5 * b * s + 7 * c = 0) 
  (h3 : r ≠ 0) 
  (h4 : s ≠ 0) 
  (h5 : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-5 * b * (25 * b^2 - 63 * c)) / (343 * c^3) := by
  sorry

end quadratic_roots_sum_of_cubes_reciprocals_l198_19847


namespace y_intercept_of_parallel_line_through_point_l198_19837

/-- A line in the plane can be represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if its coordinates satisfy the line equation -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem y_intercept_of_parallel_line_through_point 
  (l1 : Line) (p : Point) :
  l1.slope = 3 →
  parallel l1 { slope := 3, yIntercept := -2 } →
  pointOnLine p l1 →
  p.x = 5 →
  p.y = 7 →
  l1.yIntercept = -8 := by
  sorry

end y_intercept_of_parallel_line_through_point_l198_19837


namespace picture_ratio_proof_l198_19870

theorem picture_ratio_proof (total pictures : ℕ) (vertical_count : ℕ) (haphazard_count : ℕ) :
  total = 30 →
  vertical_count = 10 →
  haphazard_count = 5 →
  (total - vertical_count - haphazard_count) * 2 = total := by
  sorry

end picture_ratio_proof_l198_19870


namespace mosaic_length_l198_19810

theorem mosaic_length 
  (height_feet : ℝ) 
  (tile_size_inch : ℝ) 
  (total_tiles : ℕ) : ℝ :=
  let height_inch : ℝ := height_feet * 12
  let area_inch_sq : ℝ := total_tiles * tile_size_inch ^ 2
  let length_inch : ℝ := area_inch_sq / height_inch
  let length_feet : ℝ := length_inch / 12
  by
    have h1 : height_feet = 10 := by sorry
    have h2 : tile_size_inch = 1 := by sorry
    have h3 : total_tiles = 21600 := by sorry
    sorry

#check mosaic_length

end mosaic_length_l198_19810


namespace glass_bowl_purchase_price_l198_19876

theorem glass_bowl_purchase_price 
  (total_bowls : ℕ) 
  (sold_bowls : ℕ) 
  (selling_price : ℚ) 
  (percentage_gain : ℚ) :
  total_bowls = 118 →
  sold_bowls = 102 →
  selling_price = 15 →
  percentage_gain = 8050847457627118 / 100000000000000000 →
  ∃ (purchase_price : ℚ),
    purchase_price = 12 ∧
    sold_bowls * selling_price - total_bowls * purchase_price = 
      (percentage_gain / 100) * (total_bowls * purchase_price) := by
  sorry

end glass_bowl_purchase_price_l198_19876


namespace ellipse_param_sum_l198_19806

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: sum of ellipse parameters equals 14 -/
theorem ellipse_param_sum (e : Ellipse) : 
  let ep := compute_ellipse_params e
  e.F₁ = (0, 2) → e.F₂ = (6, 2) → e.distance_sum = 10 →
  ep.h + ep.k + ep.a + ep.b = 14 :=
by sorry

end ellipse_param_sum_l198_19806


namespace quadratic_factorization_l198_19888

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) :
  (∀ x, x^2 - 16*x + 63 = (x - a)*(x - b)) →
  3*b - a = 12 := by
  sorry

end quadratic_factorization_l198_19888


namespace valid_arrangements_count_l198_19840

/-- Represents the number of people wearing each color of clothing -/
structure ClothingCounts where
  blue : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of valid arrangements for a given set of clothing counts -/
def validArrangements (counts : ClothingCounts) : Nat :=
  sorry

/-- The specific problem instance -/
def problemInstance : ClothingCounts :=
  { blue := 2, yellow := 2, red := 1 }

/-- The main theorem stating that the number of valid arrangements for the problem instance is 48 -/
theorem valid_arrangements_count :
  validArrangements problemInstance = 48 := by
  sorry

end valid_arrangements_count_l198_19840


namespace product_difference_bound_l198_19890

theorem product_difference_bound (n : ℕ+) (a b : ℕ+) 
  (h : (a : ℝ) * b = (n : ℝ)^2 + n + 1) : 
  |((a : ℝ) - b)| ≥ 2 * Real.sqrt 2 := by
  sorry

end product_difference_bound_l198_19890


namespace difference_of_fractions_l198_19843

/-- Proves that the difference between 1/10 of 8000 and 1/20% of 8000 is equal to 796 -/
theorem difference_of_fractions : 
  (8000 / 10) - (8000 * (1 / 20) / 100) = 796 := by
  sorry

end difference_of_fractions_l198_19843


namespace lcm_ratio_implies_gcd_l198_19818

theorem lcm_ratio_implies_gcd (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  Nat.gcd A B = 30 := by
  sorry

end lcm_ratio_implies_gcd_l198_19818


namespace candies_distribution_proof_l198_19894

def least_candies_to_remove (total_candies : ℕ) (num_friends : ℕ) : ℕ :=
  total_candies % num_friends

theorem candies_distribution_proof (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 25) (h2 : num_friends = 4) :
  least_candies_to_remove total_candies num_friends = 1 := by
  sorry

end candies_distribution_proof_l198_19894


namespace total_volume_of_four_cubes_l198_19803

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) :
  edge_length = 5 → num_cubes = 4 → (edge_length ^ 3) * num_cubes = 500 := by
  sorry

end total_volume_of_four_cubes_l198_19803


namespace business_income_calculation_l198_19802

theorem business_income_calculation (spending income : ℚ) (profit : ℚ) : 
  spending / income = 5 / 9 →
  profit = income - spending →
  profit = 48000 →
  income = 108000 := by
sorry

end business_income_calculation_l198_19802


namespace max_sum_with_negative_l198_19883

def S : Finset Int := {-7, -5, -3, 0, 2, 4, 6}

def is_valid_selection (a b c : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a < 0 ∨ b < 0 ∨ c < 0)

theorem max_sum_with_negative :
  ∃ (a b c : Int), is_valid_selection a b c ∧
    a + b + c = 7 ∧
    ∀ (x y z : Int), is_valid_selection x y z → x + y + z ≤ 7 :=
by sorry

end max_sum_with_negative_l198_19883


namespace expression_evaluation_l198_19895

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2) = -6 := by
  sorry

end expression_evaluation_l198_19895


namespace train_length_l198_19845

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 63 → time_s = 20 → length_m = speed_kmh * (1000 / 3600) * time_s → length_m = 350 := by
  sorry

#check train_length

end train_length_l198_19845


namespace lisa_spoon_count_l198_19879

/-- The total number of spoons Lisa has after combining old and new sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Theorem stating that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end lisa_spoon_count_l198_19879


namespace minimum_value_of_f_max_a_for_decreasing_f_properties_l198_19880

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Theorem 1
theorem minimum_value_of_f (a : ℝ) :
  f a 0 = -2 → a = -2 ∧ ∃ x₀, ∀ x, f (-2) x ≥ f (-2) x₀ ∧ f (-2) x₀ = -10 :=
sorry

-- Theorem 2
theorem max_a_for_decreasing (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y) →
  a ≤ 10 :=
sorry

-- Theorem combining both results
theorem f_properties :
  (∃ a, f a 0 = -2 ∧ a = -2 ∧ ∃ x₀, ∀ x, f a x ≥ f a x₀ ∧ f a x₀ = -10) ∧
  (∃ a_max, a_max = 10 ∧ ∀ a > a_max, ¬(∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y)) :=
sorry

end minimum_value_of_f_max_a_for_decreasing_f_properties_l198_19880
