import Mathlib

namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l458_45861

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 60

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Gamma T-Shirt Company's setup fee -/
def gamma_setup : ℕ := 10

/-- Gamma T-Shirt Company's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_acme_cheaper <
  gamma_setup + gamma_per_shirt * min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper →
    acme_setup + acme_per_shirt * n ≥ gamma_setup + gamma_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l458_45861


namespace NUMINAMATH_CALUDE_complex_number_problem_l458_45880

theorem complex_number_problem (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 2 + 3 * Complex.I →
  α = 6 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l458_45880


namespace NUMINAMATH_CALUDE_tan_4290_degrees_l458_45849

theorem tan_4290_degrees : Real.tan (4290 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4290_degrees_l458_45849


namespace NUMINAMATH_CALUDE_exponential_function_first_quadrant_l458_45898

theorem exponential_function_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x > 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end NUMINAMATH_CALUDE_exponential_function_first_quadrant_l458_45898


namespace NUMINAMATH_CALUDE_proposition_truth_l458_45864

theorem proposition_truth : (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l458_45864


namespace NUMINAMATH_CALUDE_katarina_miles_l458_45893

/-- The total miles run by all four runners -/
def total_miles : ℕ := 195

/-- The number of miles run by Harriet -/
def harriet_miles : ℕ := 48

/-- The number of runners who ran the same distance as Harriet -/
def same_distance_runners : ℕ := 3

theorem katarina_miles : 
  total_miles - harriet_miles * same_distance_runners = 51 := by sorry

end NUMINAMATH_CALUDE_katarina_miles_l458_45893


namespace NUMINAMATH_CALUDE_max_radius_in_wine_glass_l458_45828

theorem max_radius_in_wine_glass :
  let f : ℝ → ℝ := λ x ↦ x^4
  let max_r : ℝ := (3/4) * Real.rpow 2 (1/3)
  ∀ r > 0,
    (∀ x y : ℝ, (y - r)^2 + x^2 = r^2 → y ≥ f x) ∧
    (0 - r)^2 + 0^2 = r^2 →
    r ≤ max_r :=
by sorry

end NUMINAMATH_CALUDE_max_radius_in_wine_glass_l458_45828


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l458_45830

/-- An arithmetic sequence {a_n} with a_2 = 2 and a_5 = 8 -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} with all terms positive and b_1 = 1 -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of the geometric sequence {b_n} -/
def T (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 2) ∧
  (a 2 = 2) ∧
  (a 5 = 8) ∧
  (∀ n : ℕ, n ≥ 1 → b n > 0) ∧
  (b 1 = 1) ∧
  (b 2 + b 3 = a 4) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l458_45830


namespace NUMINAMATH_CALUDE_poll_percentage_equal_l458_45853

theorem poll_percentage_equal (total : ℕ) (women_favor_percent : ℚ) (women_opposed : ℕ) : 
  total = 120 → women_favor_percent = 35/100 → women_opposed = 39 →
  ∃ (women men : ℕ), 
    women + men = total ∧ 
    women_opposed = (1 - women_favor_percent) * women ∧
    women = men ∧ 
    women / total = 1/2 ∧ 
    men / total = 1/2 := by
  sorry

#check poll_percentage_equal

end NUMINAMATH_CALUDE_poll_percentage_equal_l458_45853


namespace NUMINAMATH_CALUDE_intersection_singleton_k_value_l458_45891

theorem intersection_singleton_k_value (k : ℝ) : 
  let A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2 * (p.1 + p.2)}
  let B : Set (ℝ × ℝ) := {p | k * p.1 - p.2 + k + 3 ≥ 0}
  (Set.Subsingleton (A ∩ B)) → k = -2 - Real.sqrt 3 :=
by sorry

#check intersection_singleton_k_value

end NUMINAMATH_CALUDE_intersection_singleton_k_value_l458_45891


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_condition_l458_45821

theorem sine_cosine_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c > Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_condition_l458_45821


namespace NUMINAMATH_CALUDE_solve_rock_problem_l458_45852

def rock_problem (joshua_rocks : ℕ) : Prop :=
  let jose_rocks := joshua_rocks - 14
  let albert_rocks := jose_rocks + 28
  let clara_rocks := jose_rocks / 2
  let maria_rocks := clara_rocks + 18
  albert_rocks - joshua_rocks = 14

theorem solve_rock_problem :
  rock_problem 80 := by sorry

end NUMINAMATH_CALUDE_solve_rock_problem_l458_45852


namespace NUMINAMATH_CALUDE_strawberry_yield_per_row_l458_45818

theorem strawberry_yield_per_row :
  let total_rows : ℕ := 7
  let total_yield : ℕ := 1876
  let yield_per_row : ℕ := total_yield / total_rows
  yield_per_row = 268 := by sorry

end NUMINAMATH_CALUDE_strawberry_yield_per_row_l458_45818


namespace NUMINAMATH_CALUDE_diagonal_intersection_coincidence_l458_45811

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by its four vertices -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Predicate to check if a quadrilateral is circumscribed around a circle -/
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Function to get the tangency points of a circumscribed quadrilateral -/
def tangency_points (q : Quadrilateral) (c : Circle) : 
  (Point × Point × Point × Point) := sorry

/-- Function to get the intersection point of two diagonals -/
def diagonal_intersection (q : Quadrilateral) : Point := sorry

/-- The main theorem -/
theorem diagonal_intersection_coincidence 
  (q : Quadrilateral) (c : Circle) 
  (h : is_circumscribed q c) : 
  let (E, F, G, K) := tangency_points q c
  let q' := Quadrilateral.mk E F G K
  diagonal_intersection q = diagonal_intersection q' := by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_coincidence_l458_45811


namespace NUMINAMATH_CALUDE_changgi_weight_l458_45895

/-- Given the weights of three people with certain relationships, prove Changgi's weight -/
theorem changgi_weight (total_weight chaeyoung_hyeonjeong_diff changgi_chaeyoung_diff : ℝ) 
  (h1 : total_weight = 106.6)
  (h2 : chaeyoung_hyeonjeong_diff = 7.7)
  (h3 : changgi_chaeyoung_diff = 4.8) : 
  ∃ (changgi chaeyoung hyeonjeong : ℝ),
    changgi + chaeyoung + hyeonjeong = total_weight ∧
    chaeyoung = hyeonjeong + chaeyoung_hyeonjeong_diff ∧
    changgi = chaeyoung + changgi_chaeyoung_diff ∧
    changgi = 41.3 := by
  sorry

end NUMINAMATH_CALUDE_changgi_weight_l458_45895


namespace NUMINAMATH_CALUDE_function_properties_l458_45845

def f (a x : ℝ) : ℝ := |2*x + a| + |x - 1|

theorem function_properties :
  (∀ x : ℝ, f 3 x < 6 ↔ -8/3 < x ∧ x < 4/3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ a ≤ -3/2 ∨ a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l458_45845


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_is_20_point_5_l458_45885

/-- Represents the possible daily rainfall amounts in inches -/
inductive DailyRainfall
  | NoRain
  | LightRain
  | HeavyRain

/-- The probability of each rainfall outcome -/
def rainProbability : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0.3
  | DailyRainfall.LightRain => 0.3
  | DailyRainfall.HeavyRain => 0.4

/-- The amount of rainfall for each outcome in inches -/
def rainAmount : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0
  | DailyRainfall.LightRain => 3
  | DailyRainfall.HeavyRain => 8

/-- The number of days in the week -/
def daysInWeek : ℕ := 5

/-- The expected total rainfall for the week -/
def expectedWeeklyRainfall : ℝ :=
  daysInWeek * (rainProbability DailyRainfall.NoRain * rainAmount DailyRainfall.NoRain +
                rainProbability DailyRainfall.LightRain * rainAmount DailyRainfall.LightRain +
                rainProbability DailyRainfall.HeavyRain * rainAmount DailyRainfall.HeavyRain)

/-- Theorem: The expected total rainfall for the week is 20.5 inches -/
theorem expected_weekly_rainfall_is_20_point_5 :
  expectedWeeklyRainfall = 20.5 := by sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_is_20_point_5_l458_45885


namespace NUMINAMATH_CALUDE_person_B_processes_8_components_per_hour_l458_45825

/-- The number of components processed per hour by person B -/
def components_per_hour_B : ℕ := sorry

/-- The number of components processed per hour by person A -/
def components_per_hour_A : ℕ := components_per_hour_B + 2

/-- The time it takes for person A to process 25 components -/
def time_A : ℚ := 25 / components_per_hour_A

/-- The time it takes for person B to process 20 components -/
def time_B : ℚ := 20 / components_per_hour_B

/-- Theorem stating that person B processes 8 components per hour -/
theorem person_B_processes_8_components_per_hour :
  components_per_hour_B = 8 ∧ time_A = time_B := by sorry

end NUMINAMATH_CALUDE_person_B_processes_8_components_per_hour_l458_45825


namespace NUMINAMATH_CALUDE_unique_integer_solution_l458_45805

theorem unique_integer_solution : 
  ∀ n : ℤ, (⌊(n^2 / 4 : ℚ) + n⌋ - ⌊n / 2⌋^2 = 5) ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l458_45805


namespace NUMINAMATH_CALUDE_blue_face_ratio_one_third_l458_45872

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- The theorem stating that the ratio of blue faces to total faces is 1/3 iff n = 3 -/
theorem blue_face_ratio_one_third (n : ℕ) (c : Cube n) :
  (blue_faces c : ℚ) / (total_faces c : ℚ) = 1/3 ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_face_ratio_one_third_l458_45872


namespace NUMINAMATH_CALUDE_probability_ellipse_x_foci_value_l458_45809

/-- The probability that x²/m² + y²/n² = 1 represents an ellipse with foci on the x-axis,
    given m ∈ [1,5] and n ∈ [2,4] -/
def probability_ellipse_x_foci (m n : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is equal to some value P -/
theorem probability_ellipse_x_foci_value :
  ∃ P, ∀ m n, m ∈ Set.Icc 1 5 → n ∈ Set.Icc 2 4 →
    probability_ellipse_x_foci m n = P :=
  sorry

end NUMINAMATH_CALUDE_probability_ellipse_x_foci_value_l458_45809


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l458_45841

-- Define the ⬥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x - 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = -36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l458_45841


namespace NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l458_45854

/-- Represents the number of ways to make a certain amount with given coins -/
def WaysToMakeAmount (oneCent twoCent fiveCent target : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  WaysToMakeAmount 8 4 1 8 = 7 := by sorry

end NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l458_45854


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l458_45851

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * (x^2 - 1) - 1 - 8*x

-- State the theorem
theorem x_range_for_inequality :
  (∀ x : ℝ, (∀ m : ℝ, -1 ≤ m ∧ m ≤ 4 → f m x < 0) ↔ 0 < x ∧ x < 5/2) :=
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l458_45851


namespace NUMINAMATH_CALUDE_yellow_white_flowers_l458_45843

theorem yellow_white_flowers (total : ℕ) (red_yellow : ℕ) (red_white : ℕ) (red_minus_white : ℕ) :
  total = 44 →
  red_yellow = 17 →
  red_white = 14 →
  red_minus_white = 4 →
  red_yellow + red_white - (red_white + (total - red_yellow - red_white)) = red_minus_white →
  total - red_yellow - red_white = 13 := by
sorry

end NUMINAMATH_CALUDE_yellow_white_flowers_l458_45843


namespace NUMINAMATH_CALUDE_min_box_height_l458_45882

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- States that the surface area is at least 150 square units -/
def surface_area_constraint : ℝ → Prop := λ x => surface_area x ≥ 150

theorem min_box_height :
  ∃ x : ℝ, x > 0 ∧ surface_area_constraint x ∧
    box_height x = 10 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area_constraint y → surface_area x ≤ surface_area y :=
by sorry

end NUMINAMATH_CALUDE_min_box_height_l458_45882


namespace NUMINAMATH_CALUDE_N_subset_M_l458_45804

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l458_45804


namespace NUMINAMATH_CALUDE_solution_set_f_gt_g_min_m_for_inequality_l458_45844

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| - |x + 1|
def g (x : ℝ) : ℝ := -x

-- Theorem for the solution of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | -3 < x ∧ x < 1 ∨ x > 3} := by sorry

-- Theorem for the minimum value of m
theorem min_m_for_inequality (x : ℝ) :
  ∃ m : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m) ∧
  (∀ m' : ℝ, (∀ x : ℝ, f x - 2*x ≤ 2*(g x) + m') → m ≤ m') ∧
  m = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_g_min_m_for_inequality_l458_45844


namespace NUMINAMATH_CALUDE_bus_fare_problem_l458_45829

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | Ten : Coin
  | Fifteen : Coin
  | Twenty : Coin

/-- The value of a coin in kopecks -/
def coinValue : Coin → ℕ
  | Coin.Ten => 10
  | Coin.Fifteen => 15
  | Coin.Twenty => 20

/-- A list of coins -/
def CoinList : Type := List Coin

/-- The total value of a list of coins in kopecks -/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- A function that checks if it's possible to distribute coins to passengers -/
def canDistribute (coins : CoinList) (passengers : ℕ) (farePerPassenger : ℕ) : Prop :=
  ∃ (distribution : List CoinList),
    distribution.length = passengers ∧
    (∀ c ∈ distribution, totalValue c = farePerPassenger) ∧
    distribution.join = coins

theorem bus_fare_problem :
  (¬ ∃ (coins : CoinList), coins.length = 24 ∧ canDistribute coins 20 5) ∧
  (∃ (coins : CoinList), coins.length = 25 ∧ canDistribute coins 20 5) := by
  sorry

end NUMINAMATH_CALUDE_bus_fare_problem_l458_45829


namespace NUMINAMATH_CALUDE_treewidth_bramble_equivalence_l458_45847

-- Define a graph type
structure Graph where
  V : Type
  E : V → V → Prop

-- Define treewidth
def treewidth (G : Graph) : ℕ := sorry

-- Define bramble
def bramble (G : Graph) : Type := sorry

-- Define order of a bramble
def bramble_order (B : bramble G) : ℕ := sorry

-- The main theorem
theorem treewidth_bramble_equivalence (G : Graph) (k : ℕ) :
  (treewidth G ≥ k) ↔ (∃ B : bramble G, bramble_order B > k) := by sorry

end NUMINAMATH_CALUDE_treewidth_bramble_equivalence_l458_45847


namespace NUMINAMATH_CALUDE_percentage_returned_is_65_percent_l458_45876

/-- Represents the library's special collection --/
structure SpecialCollection where
  initial_count : ℕ
  final_count : ℕ
  loaned_out : ℕ

/-- Calculates the percentage of loaned books returned --/
def percentage_returned (sc : SpecialCollection) : ℚ :=
  (sc.loaned_out - (sc.initial_count - sc.final_count)) / sc.loaned_out * 100

/-- Theorem stating that the percentage of loaned books returned is 65% --/
theorem percentage_returned_is_65_percent (sc : SpecialCollection) 
  (h1 : sc.initial_count = 150)
  (h2 : sc.final_count = 122)
  (h3 : sc.loaned_out = 80) : 
  percentage_returned sc = 65 := by
  sorry

#eval percentage_returned { initial_count := 150, final_count := 122, loaned_out := 80 }

end NUMINAMATH_CALUDE_percentage_returned_is_65_percent_l458_45876


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l458_45835

/-- The focal length of an ellipse with equation x^2 + 2y^2 = 2 is 2 -/
theorem ellipse_focal_length : 
  let ellipse_eq : ℝ → ℝ → Prop := λ x y => x^2 + 2*y^2 = 2
  ∃ a b c : ℝ, 
    (∀ x y, ellipse_eq x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
    c^2 = a^2 - b^2 ∧
    2 * c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l458_45835


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l458_45834

theorem neon_signs_blink_together : Nat.lcm (Nat.lcm 7 11) 13 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l458_45834


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l458_45886

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℤ) (q : ℤ) :
  is_geometric_sequence a →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 4 * a 7 = -512 →
  a 3 + a 8 = 124 →
  a 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l458_45886


namespace NUMINAMATH_CALUDE_billy_final_lap_is_150_seconds_l458_45879

/-- Represents the swimming competition between Billy and Margaret -/
structure SwimmingCompetition where
  billy_first_5_laps : ℕ  -- time in seconds
  billy_next_3_laps : ℕ  -- time in seconds
  billy_9th_lap : ℕ      -- time in seconds
  margaret_total_time : ℕ -- time in seconds
  billy_win_margin : ℕ   -- time in seconds

/-- Calculates Billy's final lap time given the competition details -/
def billy_final_lap_time (comp : SwimmingCompetition) : ℕ :=
  comp.margaret_total_time - comp.billy_win_margin - 
  (comp.billy_first_5_laps + comp.billy_next_3_laps + comp.billy_9th_lap)

/-- Theorem stating that Billy's final lap time is 150 seconds -/
theorem billy_final_lap_is_150_seconds (comp : SwimmingCompetition) 
  (h1 : comp.billy_first_5_laps = 120)
  (h2 : comp.billy_next_3_laps = 240)
  (h3 : comp.billy_9th_lap = 60)
  (h4 : comp.margaret_total_time = 600)
  (h5 : comp.billy_win_margin = 30) :
  billy_final_lap_time comp = 150 := by
  sorry

end NUMINAMATH_CALUDE_billy_final_lap_is_150_seconds_l458_45879


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l458_45842

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l458_45842


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l458_45832

/-- The length of the generatrix of a cone formed by a semi-circular iron sheet -/
def generatrix_length (base_radius : ℝ) : ℝ :=
  2 * base_radius

/-- Theorem: The length of the generatrix of the cone is 8 cm -/
theorem cone_generatrix_length :
  let base_radius : ℝ := 4
  generatrix_length base_radius = 8 := by
  sorry

#check cone_generatrix_length

end NUMINAMATH_CALUDE_cone_generatrix_length_l458_45832


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_37_l458_45839

theorem smallest_five_digit_congruent_to_2_mod_37 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
  (n ≡ 2 [ZMOD 37]) ∧         -- Congruent to 2 modulo 37
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m ≡ 2 [ZMOD 37]) → n ≤ m) ∧  -- Smallest such number
  n = 10027 :=                -- The number is 10027
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_37_l458_45839


namespace NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l458_45838

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 180 × 360 × 450 rectangular solid made of unit cubes, 
    an internal diagonal passes through 720 cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_by_diagonal 180 360 450 = 720 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l458_45838


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l458_45867

/-- Given that p is inversely proportional to q+2 and p = 1 when q = 4,
    prove that p = 2 when q = 1. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ q, p = k / (q + 2)) 
  (h1 : p = 1 → q = 4) : p = 2 → q = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l458_45867


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l458_45806

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ w : ℝ, w^3 - 9*w^2 + a*w - b = 0 ↔ (w = x ∨ w = y ∨ w = z))) →
  a + b = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l458_45806


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l458_45827

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 - 2*x₁ + m + 1 = 0 ∧ x₂^2 - 2*x₂ + m + 1 = 0) ↔ 
  (-1 < m ∧ m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l458_45827


namespace NUMINAMATH_CALUDE_simplify_expression_l458_45813

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (12 * x^2 * y^3) / (8 * x * y^2) = 18 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l458_45813


namespace NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l458_45863

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents the operation of replacing a vertex with its symmetric point -/
def symmetricOperation (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Checks if a quadrilateral is permissible (sides are pairwise different and it remains convex) -/
def isPermissible (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if two quadrilaterals are equal -/
def areEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Main theorem statement -/
theorem quadrilateral_symmetry_theorem (q : Quadrilateral) 
  (h_permissible : isPermissible q) :
  (∃ (q_inscribed : Quadrilateral), isInscribed q_inscribed ∧ 
    isPermissible q_inscribed ∧ 
    areEqual (symmetricOperation (symmetricOperation (symmetricOperation q_inscribed))) q_inscribed) ∧
  (areEqual (symmetricOperation (symmetricOperation (symmetricOperation 
    (symmetricOperation (symmetricOperation (symmetricOperation q)))))) q) :=
  sorry


end NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l458_45863


namespace NUMINAMATH_CALUDE_sum_of_fractions_l458_45822

theorem sum_of_fractions : (1 : ℚ) / 2 + (1 : ℚ) / 4 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l458_45822


namespace NUMINAMATH_CALUDE_zeros_in_nine_nines_squared_l458_45817

/-- The number of zeros in the square of a number consisting of n repeated 9s -/
def zeros_in_square (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- The number 999,999,999 -/
def nine_nines : ℕ := 999999999

theorem zeros_in_nine_nines_squared :
  zeros_in_square 9 = 8 :=
sorry

#check zeros_in_nine_nines_squared

end NUMINAMATH_CALUDE_zeros_in_nine_nines_squared_l458_45817


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l458_45824

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a line with slope 1
def line_with_slope_1 (b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Define the intersection points A and B
def intersection_points (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    x₁ ≠ x₂

-- Define the condition for circle AB passing through P
def circle_condition (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    (x₁ - point_P.1) * (x₂ - point_P.1) + (y₁ - point_P.2) * (y₂ - point_P.2) = 0

-- Theorem statement
theorem parabola_line_intersection :
  ∃ (b : ℝ), intersection_points b ∧ circle_condition b ∧ b = -7 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l458_45824


namespace NUMINAMATH_CALUDE_base7_to_base9_conversion_l458_45884

/-- Converts a number from base 7 to base 10 --/
def base7To10 (n : Nat) : Nat :=
  (n % 10) + 7 * ((n / 10) % 10) + 49 * (n / 100)

/-- Converts a number from base 10 to base 9 --/
def base10To9 (n : Nat) : Nat :=
  if n < 9 then n
  else (n % 9) + 10 * (base10To9 (n / 9))

theorem base7_to_base9_conversion :
  base10To9 (base7To10 536) = 332 :=
sorry

end NUMINAMATH_CALUDE_base7_to_base9_conversion_l458_45884


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l458_45870

theorem smallest_five_digit_congruent_to_2_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 → n ≥ 10013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l458_45870


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l458_45889

theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 72 * π → cone_volume = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l458_45889


namespace NUMINAMATH_CALUDE_divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l458_45869

/-- Represents a positional numeral system with a given base. -/
structure NumeralSystem (base : ℕ) where
  (digits : List ℕ)
  (valid_digits : ∀ d ∈ digits, d < base)

/-- The value of a number in a given numeral system. -/
def value (base : ℕ) (num : NumeralSystem base) : ℕ :=
  (num.digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

/-- The sum of digits of a number in a given numeral system. -/
def digit_sum (base : ℕ) (num : NumeralSystem base) : ℕ :=
  num.digits.sum

/-- Divisibility rule for 3 in a given numeral system. -/
def divisibility_rule_3 (base : ℕ) : Prop :=
  ∀ (num : NumeralSystem base), 
    (value base num) % 3 = 0 ↔ (digit_sum base num) % 3 = 0

theorem divisibility_rule_3_decimal_true : 
  divisibility_rule_3 10 := by sorry

theorem divisibility_rule_3_duodecimal_false : 
  ¬(divisibility_rule_3 12) := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l458_45869


namespace NUMINAMATH_CALUDE_eight_lines_theorem_l458_45866

/-- Represents a collection of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- Calculates the number of regions formed by a given line configuration -/
def num_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: Eight non-parallel, non-concurrent lines divide a plane into 37 regions -/
theorem eight_lines_theorem (config : LineConfiguration) :
  config.num_lines = 8 ∧ config.no_parallel ∧ config.no_concurrent →
  num_regions config = 37 :=
by sorry

end NUMINAMATH_CALUDE_eight_lines_theorem_l458_45866


namespace NUMINAMATH_CALUDE_cube_volumes_theorem_l458_45856

/-- The edge length of the first cube in centimeters -/
def x : ℝ := 18

/-- The volume of a cube with edge length l -/
def cube_volume (l : ℝ) : ℝ := l^3

/-- The edge length of the second cube in centimeters -/
def second_edge : ℝ := x - 4

/-- The edge length of the third cube in centimeters -/
def third_edge : ℝ := second_edge - 2

/-- The volume of water remaining in the first cube after filling the second -/
def remaining_first : ℝ := cube_volume x - cube_volume second_edge

/-- The volume of water remaining in the second cube after filling the third -/
def remaining_second : ℝ := cube_volume second_edge - cube_volume third_edge

theorem cube_volumes_theorem : 
  remaining_first = 3 * remaining_second + 40 ∧ 
  cube_volume x = 5832 ∧ 
  cube_volume second_edge = 2744 ∧ 
  cube_volume third_edge = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volumes_theorem_l458_45856


namespace NUMINAMATH_CALUDE_power_comparison_l458_45873

theorem power_comparison : 2^1000 < 5^500 ∧ 5^500 < 3^750 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l458_45873


namespace NUMINAMATH_CALUDE_equality_of_areas_l458_45812

theorem equality_of_areas (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  (∃ r : Real, r > 0 ∧ 
    (r^2 * θ / 2 = r^2 * Real.tan θ / 2 - r^2 * θ / 2)) ↔ 
  Real.tan θ = 2 * θ := by
sorry

end NUMINAMATH_CALUDE_equality_of_areas_l458_45812


namespace NUMINAMATH_CALUDE_divisibility_cycle_l458_45800

theorem divisibility_cycle (a b c : ℕ+) : 
  (∃ k₁ : ℕ, (2^(a:ℕ) - 1) = k₁ * (b:ℕ)) ∧
  (∃ k₂ : ℕ, (2^(b:ℕ) - 1) = k₂ * (c:ℕ)) ∧
  (∃ k₃ : ℕ, (2^(c:ℕ) - 1) = k₃ * (a:ℕ)) →
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_cycle_l458_45800


namespace NUMINAMATH_CALUDE_weight_of_new_person_l458_45837

theorem weight_of_new_person
  (n : ℕ) -- number of persons
  (old_weight : ℝ) -- weight of the person being replaced
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 15) -- there are 15 persons
  (h2 : old_weight = 80) -- the replaced person weighs 80 kg
  (h3 : avg_increase = 3.2) -- average weight increases by 3.2 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l458_45837


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l458_45833

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := dog_teeth * num_dogs + cat_teeth * num_cats + pig_teeth * num_pigs

theorem vann_teeth_cleaning :
  total_teeth = 706 := by sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l458_45833


namespace NUMINAMATH_CALUDE_base_7_divisibility_l458_45815

def base_7_to_decimal (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k, n = 9 * k

theorem base_7_divisibility (x : ℕ) :
  (x < 7) →
  (is_divisible_by_9 (base_7_to_decimal 4 5 x 2)) ↔ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_7_divisibility_l458_45815


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l458_45848

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l458_45848


namespace NUMINAMATH_CALUDE_bakers_total_cost_l458_45865

/-- Calculates the total cost of baker's ingredients --/
theorem bakers_total_cost : 
  let flour_boxes := 3
  let flour_price := 3
  let egg_trays := 3
  let egg_price := 10
  let milk_liters := 7
  let milk_price := 5
  let soda_boxes := 2
  let soda_price := 3
  
  flour_boxes * flour_price + 
  egg_trays * egg_price + 
  milk_liters * milk_price + 
  soda_boxes * soda_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_bakers_total_cost_l458_45865


namespace NUMINAMATH_CALUDE_line_intersects_circle_l458_45801

/-- The line l: y = k(x - 1) intersects the circle C: x² + y² - 3x = 1 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * (x - 1) ∧ x^2 + y^2 - 3*x = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l458_45801


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l458_45803

theorem complex_modulus_problem (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) - i) / (a + i)).im ≠ 0 ∧ (((1 : ℂ) - i) / (a + i)).re = 0 →
  Complex.abs ((2 * a + 1 : ℂ) + Complex.I * Real.sqrt 2) = Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l458_45803


namespace NUMINAMATH_CALUDE_units_digit_not_eight_l458_45857

theorem units_digit_not_eight (a b : Nat) :
  a ∈ Finset.range 100 → b ∈ Finset.range 100 →
  (2^a + 5^b) % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_not_eight_l458_45857


namespace NUMINAMATH_CALUDE_secret_spreading_day_l458_45859

/-- The number of students who know the secret on the nth day -/
def students_knowing_secret (n : ℕ) : ℕ := 3^(n+1) - 1

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : ∃ (n : ℕ), students_knowing_secret n = 3280 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_secret_spreading_day_l458_45859


namespace NUMINAMATH_CALUDE_distance_between_P_and_Q_l458_45881

theorem distance_between_P_and_Q : ∀ (pq : ℝ),
  (∃ (x : ℝ),
    -- A walks 30 km each day
    30 * x = pq ∧
    -- B starts after A has walked 72 km
    72 + 30 * (pq / 80) = x ∧
    -- B walks 1/10 of the total distance each day
    (pq / 10) * (pq / 80) = pq - x ∧
    -- B meets A after walking for 1/8 of the daily km covered
    (pq / 10) * (1 / 8) = pq / 80) →
  pq = 320 ∨ pq = 180 := by
sorry

end NUMINAMATH_CALUDE_distance_between_P_and_Q_l458_45881


namespace NUMINAMATH_CALUDE_max_area_wire_rectangle_or_square_l458_45892

/-- The maximum area enclosed by a rectangle or square formed from a wire of length 2 meters -/
theorem max_area_wire_rectangle_or_square : 
  let wire_length : ℝ := 2
  let max_area : ℝ := (1 : ℝ) / 4
  ∀ l w : ℝ, 
    0 < l ∧ 0 < w →  -- positive length and width
    2 * (l + w) ≤ wire_length →  -- perimeter constraint
    l * w ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_wire_rectangle_or_square_l458_45892


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l458_45883

theorem sqrt_product_equals_sqrt_of_product : 
  Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l458_45883


namespace NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l458_45894

theorem max_right_triangle_area_in_rectangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  ∃ (area : ℝ), area = 90 ∧ 
  ∀ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ a ∧ 
    0 ≤ y ∧ y ≤ b ∧ 
    x^2 + y^2 = z^2 ∧ 
    z ≤ (a^2 + b^2)^(1/2) →
    (1/2) * x * y ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l458_45894


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l458_45874

def vector_a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def vector_b : Fin 2 → ℝ := ![-2, 6]

theorem parallel_vectors_k_value :
  (∃ (c : ℝ), c ≠ 0 ∧ (∀ i, vector_a k i = c * vector_b i)) →
  k = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l458_45874


namespace NUMINAMATH_CALUDE_problem_solution_l458_45860

theorem problem_solution (a b : ℝ) 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : 1 / a + 1 / b = 1) 
  (h4 : a * b = 6) : 
  b = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l458_45860


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_l458_45816

/-- Represents the number of ways to arrange plants under lamps -/
def arrangement_count (cactus_count : ℕ) (orchid_count : ℕ) (yellow_lamp_count : ℕ) (blue_lamp_count : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem rachel_plant_arrangement :
  arrangement_count 3 1 3 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_l458_45816


namespace NUMINAMATH_CALUDE_simon_age_proof_l458_45840

/-- Alvin's age in years -/
def alvin_age : ℕ := 30

/-- Simon's age in years -/
def simon_age : ℕ := 10

/-- The difference between half of Alvin's age and Simon's age -/
def age_difference : ℕ := 5

theorem simon_age_proof :
  simon_age = alvin_age / 2 - age_difference :=
by sorry

end NUMINAMATH_CALUDE_simon_age_proof_l458_45840


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l458_45890

theorem original_price_after_discounts (final_price : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (original_price : ℝ) : 
  final_price = 144 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.2 ∧ 
  final_price = original_price * (1 - discount1) * (1 - discount2) → 
  original_price = 200 := by sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l458_45890


namespace NUMINAMATH_CALUDE_return_trip_duration_l458_45823

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  outbound_time : ℝ  -- time for outbound trip (against wind)
  still_air_time : ℝ  -- time for return trip in still air

/-- The theorem stating the return trip duration --/
theorem return_trip_duration (fs : FlightScenario) : 
  fs.outbound_time = 120 →  -- Condition 1
  fs.d = fs.outbound_time * (fs.p - fs.w) →  -- Derived from Condition 1
  fs.still_air_time = fs.d / fs.p →  -- Definition of still air time
  fs.d / (fs.p + fs.w) = fs.still_air_time - 15 →  -- Condition 3
  (fs.d / (fs.p + fs.w) = 15 ∨ fs.d / (fs.p + fs.w) = 85) :=
by sorry

#check return_trip_duration

end NUMINAMATH_CALUDE_return_trip_duration_l458_45823


namespace NUMINAMATH_CALUDE_store_purchase_cost_l458_45814

/-- Given the prices of pens, notebooks, and pencils satisfying certain conditions,
    prove that 4 pens, 5 notebooks, and 5 pencils cost 73 rubles. -/
theorem store_purchase_cost (pen_price notebook_price pencil_price : ℚ) :
  (2 * pen_price + 3 * notebook_price + pencil_price = 33) →
  (pen_price + notebook_price + 2 * pencil_price = 20) →
  (4 * pen_price + 5 * notebook_price + 5 * pencil_price = 73) :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_cost_l458_45814


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l458_45820

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∀ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l458_45820


namespace NUMINAMATH_CALUDE_one_meeting_before_completion_l458_45888

/-- Represents the number of meetings between two runners on a circular track. -/
def number_of_meetings (circumference : ℝ) (speed1 speed2 : ℝ) : ℕ :=
  sorry

/-- Theorem stating that under given conditions, the runners meet once before completing a lap. -/
theorem one_meeting_before_completion :
  let circumference : ℝ := 300
  let speed1 : ℝ := 7
  let speed2 : ℝ := 3
  number_of_meetings circumference speed1 speed2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_meeting_before_completion_l458_45888


namespace NUMINAMATH_CALUDE_olivia_trip_length_l458_45836

theorem olivia_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
    total_length = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_olivia_trip_length_l458_45836


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l458_45897

theorem fractional_equation_positive_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = (1 - x) / (2 - x) - 3) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l458_45897


namespace NUMINAMATH_CALUDE_washing_machines_removed_l458_45831

theorem washing_machines_removed (
  num_containers : ℕ) (crates_per_container : ℕ) (boxes_per_crate : ℕ)
  (machines_per_box : ℕ) (num_workers : ℕ) (machines_removed_per_box : ℕ)
  (h1 : num_containers = 100)
  (h2 : crates_per_container = 30)
  (h3 : boxes_per_crate = 15)
  (h4 : machines_per_box = 10)
  (h5 : num_workers = 6)
  (h6 : machines_removed_per_box = 4)
  : (num_containers * crates_per_container * boxes_per_crate * machines_removed_per_box * num_workers) = 180000 := by
  sorry

#check washing_machines_removed

end NUMINAMATH_CALUDE_washing_machines_removed_l458_45831


namespace NUMINAMATH_CALUDE_min_distance_between_lines_l458_45862

/-- The minimum distance between a point on the line 3x+4y-12=0 and a point on the line 6x+8y+5=0 is 29/10 -/
theorem min_distance_between_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 12 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ (d : ℝ), d = 29/10 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ line1 → q ∈ line2 → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_lines_l458_45862


namespace NUMINAMATH_CALUDE_union_A_B_when_m_half_B_subset_A_iff_m_range_l458_45802

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + m) * (x - 2*m - 1) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Statement 1
theorem union_A_B_when_m_half : 
  A (1/2) ∪ B = {x | -2 < x ∧ x < 2} := by sorry

-- Statement 2
theorem B_subset_A_iff_m_range :
  ∀ m : ℝ, B ⊆ A m ↔ m ≤ -3/2 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_half_B_subset_A_iff_m_range_l458_45802


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l458_45878

theorem binomial_coefficient_relation (n : ℕ) : 
  (n ≥ 2) →
  (Nat.choose n 2 * 3^(n-2) = 5 * 3^n) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l458_45878


namespace NUMINAMATH_CALUDE_tablecloth_diameter_is_ten_l458_45887

/-- The diameter of a circular tablecloth with a given radius --/
def tablecloth_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a circular tablecloth with a radius of 5 feet is 10 feet --/
theorem tablecloth_diameter_is_ten :
  tablecloth_diameter 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_diameter_is_ten_l458_45887


namespace NUMINAMATH_CALUDE_min_value_of_expression_l458_45819

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2027 ≥ 2026 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2027 = 2026 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l458_45819


namespace NUMINAMATH_CALUDE_city_water_consumption_most_suitable_l458_45846

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  population_size : Nat
  practicality_of_sampling : Bool

/-- Determines if a survey scenario is suitable for sampling -/
def is_suitable_for_sampling (scenario : SurveyScenario) : Bool :=
  scenario.population_size > 1000 && scenario.practicality_of_sampling

/-- The list of survey scenarios -/
def survey_scenarios : List SurveyScenario := [
  { description := "Security check for passengers before boarding a plane",
    population_size := 300,
    practicality_of_sampling := false },
  { description := "Survey of the vision of students in Grade 8, Class 1 of a certain school",
    population_size := 40,
    practicality_of_sampling := false },
  { description := "Survey of the average daily water consumption in a certain city",
    population_size := 100000,
    practicality_of_sampling := true },
  { description := "Survey of the sleep time of 20 centenarians in a certain county",
    population_size := 20,
    practicality_of_sampling := false }
]

theorem city_water_consumption_most_suitable :
  ∃ (scenario : SurveyScenario),
    scenario ∈ survey_scenarios ∧
    scenario.description = "Survey of the average daily water consumption in a certain city" ∧
    is_suitable_for_sampling scenario ∧
    ∀ (other : SurveyScenario),
      other ∈ survey_scenarios →
      other ≠ scenario →
      ¬(is_suitable_for_sampling other) :=
by sorry

end NUMINAMATH_CALUDE_city_water_consumption_most_suitable_l458_45846


namespace NUMINAMATH_CALUDE_equation_solutions_l458_45808

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => x * (5 * x + 2) - 6 * (5 * x + 2)
  (f 6 = 0 ∧ f (-2/5) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 6 ∨ x = -2/5 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l458_45808


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l458_45875

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 14 ∧ (10154 - x) % 30 = 0 ∧ ∀ (y : ℕ), y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l458_45875


namespace NUMINAMATH_CALUDE_smallest_median_l458_45896

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 4, 3, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_median :
  ∀ x : ℤ, ∃ m : ℤ, is_median m (number_set x) ∧ 
  (∀ m' : ℤ, is_median m' (number_set x) → m ≤ m') ∧
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_median_l458_45896


namespace NUMINAMATH_CALUDE_sushi_father_lollipops_l458_45855

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops eaten -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

theorem sushi_father_lollipops : 
  initial_lollipops = eaten_lollipops + remaining_lollipops :=
by sorry

end NUMINAMATH_CALUDE_sushi_father_lollipops_l458_45855


namespace NUMINAMATH_CALUDE_exponent_simplification_l458_45871

theorem exponent_simplification (x : ℝ) : 4 * x^3 - 3 * x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l458_45871


namespace NUMINAMATH_CALUDE_trendy_haircut_cost_is_8_l458_45807

/-- The cost of a trendy haircut -/
def trendy_haircut_cost : ℕ → Prop
| cost => 
  let normal_cost : ℕ := 5
  let special_cost : ℕ := 6
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let total_weekly_earnings : ℕ := 413
  (normal_cost * normal_per_day + special_cost * special_per_day + cost * trendy_per_day) * days_per_week = total_weekly_earnings

theorem trendy_haircut_cost_is_8 : trendy_haircut_cost 8 := by
  sorry

end NUMINAMATH_CALUDE_trendy_haircut_cost_is_8_l458_45807


namespace NUMINAMATH_CALUDE_spring_length_dependent_on_mass_l458_45877

/-- Represents the relationship between spring length and object mass -/
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

theorem spring_length_dependent_on_mass :
  ∃ (f : ℝ → ℝ), ∀ (mass : ℝ), spring_length mass = f mass ∧
  ¬ (∃ (g : ℝ → ℝ), ∀ (length : ℝ), mass = g length) :=
sorry

end NUMINAMATH_CALUDE_spring_length_dependent_on_mass_l458_45877


namespace NUMINAMATH_CALUDE_cut_cube_edges_l458_45850

/-- A cube with one corner cut off, creating a new triangular face -/
structure CutCube where
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges created by the cut -/
  new_edges : ℕ
  /-- The cut creates a triangular face -/
  triangular_face : Bool

/-- The total number of edges in the cut cube -/
def CutCube.total_edges (c : CutCube) : ℕ := c.original_edges + c.new_edges

/-- Theorem stating that a cube with one corner cut off has 15 edges -/
theorem cut_cube_edges :
  ∀ (c : CutCube),
  c.original_edges = 12 ∧ c.new_edges = 3 ∧ c.triangular_face = true →
  c.total_edges = 15 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_edges_l458_45850


namespace NUMINAMATH_CALUDE_m_range_proof_l458_45826

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a < b ∧ (m + 1 = a) ∧ (3 - m = b)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 3

-- Theorem statement
theorem m_range_proof (m : ℝ) : (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l458_45826


namespace NUMINAMATH_CALUDE_floor_ceil_product_l458_45868

theorem floor_ceil_product : ⌊(0.998 : ℝ)⌋ * ⌈(1.999 : ℝ)⌉ = 0 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_product_l458_45868


namespace NUMINAMATH_CALUDE_inequality_proof_l458_45899

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 1) : 
  (a + 1/a) * (b + 1/b) * (c + 1/c) ≥ 1000/27 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l458_45899


namespace NUMINAMATH_CALUDE_divide_l_shaped_ice_sheet_l458_45858

/-- Represents an L-shaped ice sheet composed of three unit squares -/
structure LShapedIceSheet :=
  (area : ℝ := 3)

/-- Represents a part of the divided ice sheet -/
structure IceSheetPart :=
  (area : ℝ)

/-- Theorem stating that the L-shaped ice sheet can be divided into four equal parts -/
theorem divide_l_shaped_ice_sheet (sheet : LShapedIceSheet) :
  ∃ (part1 part2 part3 part4 : IceSheetPart),
    part1.area = 3/4 ∧
    part2.area = 3/4 ∧
    part3.area = 3/4 ∧
    part4.area = 3/4 ∧
    part1.area + part2.area + part3.area + part4.area = sheet.area :=
sorry

end NUMINAMATH_CALUDE_divide_l_shaped_ice_sheet_l458_45858


namespace NUMINAMATH_CALUDE_l₃_is_symmetric_to_l₁_l458_45810

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := x + y = 0

/-- The equation of line l₃ -/
def l₃ (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- A point is symmetric to another point with respect to l₂ -/
def symmetric_wrt_l₂ (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -y₁ ∧ y₂ = -x₁

theorem l₃_is_symmetric_to_l₁ :
  ∀ x y : ℝ, l₃ x y ↔ ∃ x₁ y₁ : ℝ, l₁ x₁ y₁ ∧ symmetric_wrt_l₂ x y x₁ y₁ :=
sorry

end NUMINAMATH_CALUDE_l₃_is_symmetric_to_l₁_l458_45810
