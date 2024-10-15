import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l2124_212442

def integer_sequence : List Int := List.range 10 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_sequence (seq : List Int := integer_sequence) :
  seq.sum / seq.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l2124_212442


namespace NUMINAMATH_CALUDE_two_sets_of_points_l2124_212443

/-- Given two sets of points in a plane, if the total number of connecting lines
    is 136 and the sum of connecting lines between the groups is 66,
    then one set contains 10 points and the other contains 7 points. -/
theorem two_sets_of_points (x y : ℕ) : 
  x + y = 17 ∧ 
  (x * (x - 1) + y * (y - 1)) / 2 = 136 ∧ 
  x * y = 66 →
  (x = 10 ∧ y = 7) ∨ (x = 7 ∧ y = 10) :=
by sorry

end NUMINAMATH_CALUDE_two_sets_of_points_l2124_212443


namespace NUMINAMATH_CALUDE_value_of_expression_l2124_212484

theorem value_of_expression (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : |-b| = 5) 
  (h3 : a < b) : 
  2*a - 3*b = -11 ∨ 2*a - 3*b = -19 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2124_212484


namespace NUMINAMATH_CALUDE_toms_weekly_distance_l2124_212435

/-- Represents Tom's weekly exercise schedule --/
structure ExerciseSchedule where
  monday_run_morning : Real
  monday_run_evening : Real
  wednesday_run_morning : Real
  wednesday_run_evening : Real
  friday_run_first : Real
  friday_run_second : Real
  friday_run_third : Real
  tuesday_cycle_morning : Real
  tuesday_cycle_evening : Real
  thursday_cycle_morning : Real
  thursday_cycle_evening : Real

/-- Calculates the total distance Tom runs and cycles in a week --/
def total_distance (schedule : ExerciseSchedule) : Real :=
  schedule.monday_run_morning + schedule.monday_run_evening +
  schedule.wednesday_run_morning + schedule.wednesday_run_evening +
  schedule.friday_run_first + schedule.friday_run_second + schedule.friday_run_third +
  schedule.tuesday_cycle_morning + schedule.tuesday_cycle_evening +
  schedule.thursday_cycle_morning + schedule.thursday_cycle_evening

/-- Tom's actual exercise schedule --/
def toms_schedule : ExerciseSchedule :=
  { monday_run_morning := 6
  , monday_run_evening := 4
  , wednesday_run_morning := 5.25
  , wednesday_run_evening := 5
  , friday_run_first := 3
  , friday_run_second := 4.5
  , friday_run_third := 2
  , tuesday_cycle_morning := 10
  , tuesday_cycle_evening := 8
  , thursday_cycle_morning := 7
  , thursday_cycle_evening := 12
  }

/-- Theorem stating that Tom's total weekly distance is 66.75 miles --/
theorem toms_weekly_distance : total_distance toms_schedule = 66.75 := by
  sorry


end NUMINAMATH_CALUDE_toms_weekly_distance_l2124_212435


namespace NUMINAMATH_CALUDE_min_teachers_for_given_problem_l2124_212456

/-- Represents the number of teachers for each subject -/
structure SubjectTeachers where
  english : Nat
  history : Nat
  geography : Nat

/-- The minimum number of teachers required given the subject teachers -/
def minTeachersRequired (s : SubjectTeachers) : Nat :=
  sorry

/-- Theorem stating the minimum number of teachers required for the given problem -/
theorem min_teachers_for_given_problem :
  let s : SubjectTeachers := ⟨9, 7, 6⟩
  minTeachersRequired s = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_for_given_problem_l2124_212456


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2124_212416

theorem polynomial_evaluation : let x : ℝ := 3
  x^6 - 4*x^2 + 3*x = 702 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2124_212416


namespace NUMINAMATH_CALUDE_smallest_in_S_l2124_212436

def S : Set Int := {0, -17, 4, 3, -2}

theorem smallest_in_S : ∀ x ∈ S, -17 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_in_S_l2124_212436


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2124_212493

def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2124_212493


namespace NUMINAMATH_CALUDE_jamie_hourly_rate_l2124_212487

/-- Represents Jamie's flyer delivery job -/
structure FlyerJob where
  days_per_week : ℕ
  hours_per_day : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the hourly rate given a flyer delivery job -/
def hourly_rate (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.days_per_week * job.hours_per_day * job.total_weeks)

/-- Theorem stating that Jamie's hourly rate is $10 -/
theorem jamie_hourly_rate :
  let job : FlyerJob := {
    days_per_week := 2,
    hours_per_day := 3,
    total_weeks := 6,
    total_earnings := 360
  }
  hourly_rate job = 10 := by sorry

end NUMINAMATH_CALUDE_jamie_hourly_rate_l2124_212487


namespace NUMINAMATH_CALUDE_marias_additional_cupcakes_l2124_212438

/-- Given that Maria initially made 19 cupcakes, sold 5, and ended up with 24 cupcakes,
    prove that she made 10 additional cupcakes. -/
theorem marias_additional_cupcakes :
  let initial_cupcakes : ℕ := 19
  let sold_cupcakes : ℕ := 5
  let final_cupcakes : ℕ := 24
  let additional_cupcakes := final_cupcakes - (initial_cupcakes - sold_cupcakes)
  additional_cupcakes = 10 := by sorry

end NUMINAMATH_CALUDE_marias_additional_cupcakes_l2124_212438


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2124_212459

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a point is on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem circle_radius_problem (A B C D : Circle) : 
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent A C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 2 ∧
  point_on_circle D.center A →
  B.radius = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2124_212459


namespace NUMINAMATH_CALUDE_max_segments_on_unit_disc_l2124_212452

/-- The maximum number of segments with lengths greater than 1 determined by n points on a unit disc -/
def maxSegments (n : ℕ) : ℚ :=
  2 * n^2 / 5

/-- Theorem stating the maximum number of segments with lengths greater than 1 -/
theorem max_segments_on_unit_disc (n : ℕ) (h : n ≥ 2) :
  maxSegments n = (2 * n^2 : ℚ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_segments_on_unit_disc_l2124_212452


namespace NUMINAMATH_CALUDE_min_x_plus_y_l2124_212491

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l2124_212491


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2124_212454

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, a^2 + 1, a^2 - 3}
  let B : Set ℝ := {-4, a - 1, a + 1}
  A ∩ B = {-2} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2124_212454


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2124_212475

/-- The area of the shaded region in a grid composed of three rectangles minus a triangle --/
theorem shaded_area_calculation (bottom_height bottom_width middle_height middle_width top_height top_width triangle_base triangle_height : ℕ) 
  (h_bottom : bottom_height = 3 ∧ bottom_width = 5)
  (h_middle : middle_height = 4 ∧ middle_width = 7)
  (h_top : top_height = 5 ∧ top_width = 12)
  (h_triangle : triangle_base = 12 ∧ triangle_height = 5) :
  (bottom_height * bottom_width + middle_height * middle_width + top_height * top_width) - (triangle_base * triangle_height / 2) = 73 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2124_212475


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2124_212440

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 94) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2124_212440


namespace NUMINAMATH_CALUDE_area_under_sine_curve_l2124_212413

theorem area_under_sine_curve : 
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := 2 * π / 3
  let curve (x : ℝ) := 2 * Real.sin x
  ∫ x in lower_bound..upper_bound, curve x = 3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_sine_curve_l2124_212413


namespace NUMINAMATH_CALUDE_cubic_inequality_l2124_212458

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2124_212458


namespace NUMINAMATH_CALUDE_last_three_average_l2124_212483

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 6 →
  numbers.sum / 6 = 60 →
  (numbers.take 3).sum / 3 = 55 →
  (numbers.drop 3).sum = 195 →
  (numbers.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l2124_212483


namespace NUMINAMATH_CALUDE_equality_of_squares_l2124_212444

theorem equality_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_equality_of_squares_l2124_212444


namespace NUMINAMATH_CALUDE_steven_peach_count_l2124_212466

-- Define the number of peaches Jake and Steven have
def jake_peaches : ℕ := 7
def steven_peaches : ℕ := jake_peaches + 12

-- Theorem to prove
theorem steven_peach_count : steven_peaches = 19 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l2124_212466


namespace NUMINAMATH_CALUDE_a_share_profit_l2124_212481

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem: A's share in the profit is 3660 given the investments and total profit -/
theorem a_share_profit (investment_a investment_b investment_c total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12200) :
  calculate_share_profit investment_a investment_b investment_c total_profit = 3660 := by
  sorry

#eval calculate_share_profit 6300 4200 10500 12200

end NUMINAMATH_CALUDE_a_share_profit_l2124_212481


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l2124_212429

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) :
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a * 1.25 ∧
  m = b * (1 - p / 100) ∧
  m / x = 0.2
  → p = 80 := by sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l2124_212429


namespace NUMINAMATH_CALUDE_additional_amount_needed_l2124_212430

def fundraiser_goal : ℕ := 750
def bronze_donation : ℕ := 25
def silver_donation : ℕ := 50
def gold_donation : ℕ := 100
def bronze_count : ℕ := 10
def silver_count : ℕ := 7
def gold_count : ℕ := 1

theorem additional_amount_needed : 
  fundraiser_goal - (bronze_donation * bronze_count + silver_donation * silver_count + gold_donation * gold_count) = 50 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l2124_212430


namespace NUMINAMATH_CALUDE_driver_speed_ratio_l2124_212460

/-- Two drivers meet halfway between cities A and B. The first driver left earlier
    than the second driver by an amount of time equal to half the time it would have
    taken them to meet if they had left simultaneously. This theorem proves the ratio
    of their speeds. -/
theorem driver_speed_ratio
  (x : ℝ)  -- Distance between cities A and B
  (v₁ v₂ : ℝ)  -- Speeds of the first and second driver respectively
  (h₁ : v₁ > 0)  -- First driver's speed is positive
  (h₂ : v₂ > 0)  -- Second driver's speed is positive
  (h₃ : x > 0)  -- Distance between cities is positive
  (h₄ : x / (2 * v₁) = x / (2 * v₂) + x / (2 * (v₁ + v₂)))  -- Meeting condition
  : v₂ / v₁ = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_driver_speed_ratio_l2124_212460


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2124_212477

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 6) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2124_212477


namespace NUMINAMATH_CALUDE_william_bottle_caps_l2124_212467

/-- Given that William initially had 2 bottle caps and now has 43 bottle caps in total,
    prove that he bought 41 bottle caps. -/
theorem william_bottle_caps :
  let initial_caps : ℕ := 2
  let total_caps : ℕ := 43
  let bought_caps : ℕ := total_caps - initial_caps
  bought_caps = 41 := by sorry

end NUMINAMATH_CALUDE_william_bottle_caps_l2124_212467


namespace NUMINAMATH_CALUDE_fraction_simplification_l2124_212437

theorem fraction_simplification :
  let x := (1/2 - 1/3) / (3/7 + 1/9)
  x * (1/4) = 21/272 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2124_212437


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l2124_212488

theorem product_greater_than_sum {a b : ℝ} (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l2124_212488


namespace NUMINAMATH_CALUDE_total_badges_sum_l2124_212473

/-- The total number of spelling badges for Hermione, Luna, and Celestia -/
def total_badges (hermione_badges luna_badges celestia_badges : ℕ) : ℕ :=
  hermione_badges + luna_badges + celestia_badges

/-- Theorem stating that the total number of spelling badges is 83 -/
theorem total_badges_sum : total_badges 14 17 52 = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_badges_sum_l2124_212473


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2124_212468

theorem perfect_square_divisibility (a p q : ℕ+) (h1 : ∃ k : ℕ+, a = k ^ 2) 
  (h2 : a = p * q) (h3 : (2021 : ℕ) ∣ p ^ 3 + q ^ 3 + p ^ 2 * q + p * q ^ 2) :
  (2021 : ℕ) ∣ Nat.sqrt a.val := by sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2124_212468


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2124_212406

theorem quadratic_equation_solution : 
  ∃ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = 1 ∨ y = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2124_212406


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l2124_212425

theorem complex_roots_quadratic (a b : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁ = a + 3*I ∧ z₂ = b + 7*I ∧ 
   z₁^2 - (10 + 10*I)*z₁ + (70 + 16*I) = 0 ∧
   z₂^2 - (10 + 10*I)*z₂ + (70 + 16*I) = 0) →
  a = -3.5 ∧ b = 13.5 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l2124_212425


namespace NUMINAMATH_CALUDE_library_books_l2124_212449

theorem library_books (initial_books : ℕ) : 
  initial_books - 124 + 22 = 234 → initial_books = 336 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l2124_212449


namespace NUMINAMATH_CALUDE_purchase_ways_count_l2124_212478

/-- Represents the number of oreo flavors --/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors --/
def milk_flavors : ℕ := 4

/-- Represents the total number of products they purchase collectively --/
def total_products : ℕ := 4

/-- Represents the maximum number of same flavor items Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha and Beta can purchase products --/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products --/
theorem purchase_ways_count : purchase_ways = 2143 := by sorry

end NUMINAMATH_CALUDE_purchase_ways_count_l2124_212478


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l2124_212402

theorem initial_birds_on_fence (initial_birds storks additional_birds final_birds : ℕ) :
  initial_birds + storks > 0 ∧ 
  storks = 46 ∧ 
  additional_birds = 6 ∧ 
  final_birds = 10 ∧ 
  initial_birds + additional_birds = final_birds →
  initial_birds = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l2124_212402


namespace NUMINAMATH_CALUDE_infinitely_many_special_integers_l2124_212450

theorem infinitely_many_special_integers (k : ℕ) (hk : k > 1) :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ x ∈ S, 
    (∃ (a b : ℕ), x = a^k - b^k) ∧ 
    (¬∃ (c d : ℕ), x = c^k + d^k)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_integers_l2124_212450


namespace NUMINAMATH_CALUDE_percussion_probability_l2124_212448

def total_sounds : ℕ := 6
def percussion_sounds : ℕ := 3

theorem percussion_probability :
  (percussion_sounds.choose 2 : ℚ) / (total_sounds.choose 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_percussion_probability_l2124_212448


namespace NUMINAMATH_CALUDE_cement_mixture_percentage_l2124_212439

/-- Calculates the percentage of cement in the second mixture for concrete production --/
theorem cement_mixture_percentage 
  (total_concrete : Real) 
  (final_cement_percentage : Real)
  (first_mixture_percentage : Real)
  (second_mixture_amount : Real) :
  let total_cement := total_concrete * final_cement_percentage / 100
  let first_mixture_amount := total_concrete - second_mixture_amount
  let first_mixture_cement := first_mixture_amount * first_mixture_percentage / 100
  let second_mixture_cement := total_cement - first_mixture_cement
  second_mixture_cement / second_mixture_amount * 100 = 80 :=
by
  sorry

#check cement_mixture_percentage 10 62 20 7

end NUMINAMATH_CALUDE_cement_mixture_percentage_l2124_212439


namespace NUMINAMATH_CALUDE_parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l2124_212482

-- Define a square
def Square : Type := Unit

-- Define a straight cut from edge to edge
def StraightCut (s : Square) : Type := Unit

-- Define parallel cuts
def ParallelCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define intersecting cuts
def IntersectingCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define the number of pieces resulting from cuts
def NumberOfPieces (s : Square) (c1 c2 : StraightCut s) : ℕ := sorry

-- Theorem for parallel cuts
theorem parallel_cuts_three_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : ParallelCuts s c1 c2) : NumberOfPieces s c1 c2 = 3 := by sorry

-- Theorem for intersecting cuts
theorem intersecting_cuts_four_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : IntersectingCuts s c1 c2) : NumberOfPieces s c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l2124_212482


namespace NUMINAMATH_CALUDE_permutation_game_winning_strategy_l2124_212470

/-- The game on permutation group S_n -/
def PermutationGame (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ (strategy : ℕ → Bool),
    (n ≥ 4 ∧ Odd n → strategy n = false) ∧
    (n = 2 ∨ n = 3 → strategy n = true)

/-- Theorem stating the winning strategies for different values of n -/
theorem permutation_game_winning_strategy :
  ∀ n : ℕ, PermutationGame n :=
sorry

end NUMINAMATH_CALUDE_permutation_game_winning_strategy_l2124_212470


namespace NUMINAMATH_CALUDE_converse_correctness_l2124_212480

-- Define the original proposition
def original_prop (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- Define the converse proposition
def converse_prop (a b : ℝ) : Prop := (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0)

-- Theorem stating that the converse_prop is indeed the converse of original_prop
theorem converse_correctness : 
  ∀ (a b : ℝ), converse_prop a b ↔ (¬(a^2 + b^2 = 0) → ¬(a = 0 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_converse_correctness_l2124_212480


namespace NUMINAMATH_CALUDE_a_equals_three_iff_parallel_l2124_212446

def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 6 * a = 0

def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

theorem a_equals_three_iff_parallel :
  ∀ (a : ℝ), a = 3 ↔ parallel a :=
sorry

end NUMINAMATH_CALUDE_a_equals_three_iff_parallel_l2124_212446


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2124_212490

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = -11) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2124_212490


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2124_212427

theorem salary_change_percentage (x : ℝ) : 
  (1 - x/100) * (1 + x/100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2124_212427


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l2124_212457

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l2124_212457


namespace NUMINAMATH_CALUDE_steak_cooking_time_l2124_212428

def waffle_time : ℕ := 10
def total_time : ℕ := 28
def num_steaks : ℕ := 3

theorem steak_cooking_time :
  ∃ (steak_time : ℕ), steak_time * num_steaks + waffle_time = total_time ∧ steak_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_steak_cooking_time_l2124_212428


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2124_212497

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2124_212497


namespace NUMINAMATH_CALUDE_stocking_discount_percentage_l2124_212418

-- Define the given conditions
def num_grandchildren : ℕ := 5
def num_children : ℕ := 4
def stockings_per_person : ℕ := 5
def stocking_price : ℚ := 20
def monogram_price : ℚ := 5
def total_cost_after_discount : ℚ := 1035

-- Define the theorem
theorem stocking_discount_percentage :
  let total_people := num_grandchildren + num_children
  let total_stockings := total_people * stockings_per_person
  let stocking_cost := total_stockings * stocking_price
  let monogram_cost := total_stockings * monogram_price
  let total_cost_before_discount := stocking_cost + monogram_cost
  let discount_amount := total_cost_before_discount - total_cost_after_discount
  let discount_percentage := (discount_amount / total_cost_before_discount) * 100
  discount_percentage = 8 := by
sorry

end NUMINAMATH_CALUDE_stocking_discount_percentage_l2124_212418


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l2124_212453

/-- Proves that the weight of the second batch of coffee is 100 pounds given the initial conditions --/
theorem coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (total_decaf_percent : ℝ) 
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.70)
  (h4 : total_decaf_percent = 0.30) : 
  ∃ (second_batch : ℝ), 
    initial_decaf_percent * initial_stock + second_batch_decaf_percent * second_batch = 
    total_decaf_percent * (initial_stock + second_batch) ∧ 
    second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l2124_212453


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2124_212434

theorem absolute_value_equation_solutions (z : ℝ) :
  ∃ (x y : ℝ), (|x - y^2| = z*x + y^2 ∧ z*x + y^2 ≥ 0) ↔
  ((x = 0 ∧ y = 0) ∨
   (∃ (y : ℝ), x = 2*y^2/(1-z) ∧ z ≠ 1 ∧ z > -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2124_212434


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2124_212433

theorem angle_sum_theorem (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_equation : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2124_212433


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2124_212465

theorem trigonometric_equation_solution (z : ℝ) : 
  (Real.sin (3 * z) + Real.sin z ^ 3 = (3 * Real.sqrt 3 / 4) * Real.sin (2 * z)) ↔ 
  (∃ k : ℤ, z = k * Real.pi) ∨ 
  (∃ n : ℤ, z = Real.pi / 2 * (2 * n + 1)) ∨ 
  (∃ l : ℤ, z = Real.pi / 6 + 2 * Real.pi * l ∨ z = -Real.pi / 6 + 2 * Real.pi * l) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2124_212465


namespace NUMINAMATH_CALUDE_a_age_is_eleven_l2124_212494

/-- Represents a person in the problem -/
inductive Person
  | A
  | B
  | C

/-- Represents a statement made by a person -/
structure Statement where
  person : Person
  content : Nat → Nat → Nat → Prop

/-- The set of all statements made by the three people -/
def statements : List Statement := sorry

/-- Predicate to check if a set of ages is consistent with the true statements -/
def consistent (a b c : Nat) : Prop := sorry

/-- Theorem stating that A's age is 11 -/
theorem a_age_is_eleven :
  ∃ (a b c : Nat),
    consistent a b c ∧
    (∀ (x y z : Nat), consistent x y z → (x = a ∧ y = b ∧ z = c)) ∧
    a = 11 := by sorry

end NUMINAMATH_CALUDE_a_age_is_eleven_l2124_212494


namespace NUMINAMATH_CALUDE_baby_grab_theorem_l2124_212417

/-- Represents the number of possible outcomes when a baby grabs one item from a set of items -/
def possible_outcomes (educational living entertainment : ℕ) : ℕ :=
  educational + living + entertainment

/-- Theorem: The number of possible outcomes when a baby grabs one item
    is equal to the sum of educational, living, and entertainment items -/
theorem baby_grab_theorem (educational living entertainment : ℕ) :
  possible_outcomes educational living entertainment =
  educational + living + entertainment := by
  sorry

end NUMINAMATH_CALUDE_baby_grab_theorem_l2124_212417


namespace NUMINAMATH_CALUDE_sum_of_scores_l2124_212400

/-- The sum of scores in a guessing game -/
theorem sum_of_scores (hajar_score : ℕ) (score_difference : ℕ) : 
  hajar_score = 24 →
  score_difference = 21 →
  hajar_score + (hajar_score + score_difference) = 69 := by
  sorry

#check sum_of_scores

end NUMINAMATH_CALUDE_sum_of_scores_l2124_212400


namespace NUMINAMATH_CALUDE_power_multiplication_l2124_212403

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2124_212403


namespace NUMINAMATH_CALUDE_cos_2x_quadratic_equation_l2124_212451

theorem cos_2x_quadratic_equation (a b c : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, a * (Real.cos x)^2 + b * Real.cos x + c = 0) →
    (∀ x, f (Real.cos (2 * x)) = 0) ∧
    (∃ p q r : ℝ, ∀ y, f y = p * y^2 + q * y + r ∧
      p = a^2 ∧
      q = 2 * (a^2 + 2 * a * c - b^2) ∧
      r = (a^2 + 2 * c)^2 - 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_cos_2x_quadratic_equation_l2124_212451


namespace NUMINAMATH_CALUDE_distribute_volunteers_eq_twelve_l2124_212445

/-- The number of ways to distribute 8 volunteer positions to 3 schools -/
def distribute_volunteers : ℕ :=
  let total_positions := 8
  let num_schools := 3
  let total_partitions := Nat.choose (total_positions - 1) (num_schools - 1)
  let equal_allocations := 3 * 3  -- (1,1,6), (2,2,4), (3,3,2)
  total_partitions - equal_allocations

/-- Theorem: The number of ways to distribute 8 volunteer positions to 3 schools,
    with each school receiving at least one position and the allocations being unequal, is 12 -/
theorem distribute_volunteers_eq_twelve : distribute_volunteers = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribute_volunteers_eq_twelve_l2124_212445


namespace NUMINAMATH_CALUDE_g_15_equals_281_l2124_212462

/-- The function g defined for all natural numbers -/
def g (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem stating that g(15) equals 281 -/
theorem g_15_equals_281 : g 15 = 281 := by
  sorry

end NUMINAMATH_CALUDE_g_15_equals_281_l2124_212462


namespace NUMINAMATH_CALUDE_find_first_number_l2124_212464

/-- A sequence where the sum of two numbers is always 1 less than their actual arithmetic sum -/
def SpecialSequence (a b c : ℕ) : Prop := a + b = c + 1

/-- The theorem to prove -/
theorem find_first_number (x : ℕ) :
  SpecialSequence x 9 16 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_first_number_l2124_212464


namespace NUMINAMATH_CALUDE_min_bags_for_candy_distribution_l2124_212415

theorem min_bags_for_candy_distribution : ∃ (n : ℕ), n > 0 ∧ 
  77 % n = 0 ∧ (7 * n) % 77 = 0 ∧ (11 * n) % 77 = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → 
    77 % m ≠ 0 ∨ (7 * m) % 77 ≠ 0 ∨ (11 * m) % 77 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_bags_for_candy_distribution_l2124_212415


namespace NUMINAMATH_CALUDE_alex_growth_rate_l2124_212489

/-- Alex's growth rate problem -/
theorem alex_growth_rate :
  let required_height : ℚ := 54
  let current_height : ℚ := 48
  let growth_rate_upside_down : ℚ := 1 / 12
  let hours_upside_down_per_month : ℚ := 2
  let months_in_year : ℕ := 12
  let height_difference := required_height - current_height
  let growth_from_hanging := growth_rate_upside_down * hours_upside_down_per_month * months_in_year
  let natural_growth := height_difference - growth_from_hanging
  natural_growth / months_in_year = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_alex_growth_rate_l2124_212489


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2124_212486

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ (x > -2 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2124_212486


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2124_212479

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2124_212479


namespace NUMINAMATH_CALUDE_infinite_non_triangular_arithmetic_sequence_l2124_212485

-- Define triangular numbers
def isTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n - 1) / 2

-- Define an arithmetic sequence
def isArithmeticSequence (s : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, s (n + 1) = s n + d

-- Theorem statement
theorem infinite_non_triangular_arithmetic_sequence :
  ∃ s : ℕ → ℕ, isArithmeticSequence s ∧ (∀ n : ℕ, ¬ isTriangular (s n)) :=
sorry

end NUMINAMATH_CALUDE_infinite_non_triangular_arithmetic_sequence_l2124_212485


namespace NUMINAMATH_CALUDE_geography_textbook_cost_l2124_212423

/-- The cost of a geography textbook given the following conditions:
  1. 35 English textbooks and 35 geography textbooks are ordered
  2. An English book costs $7.50
  3. The total amount of the order is $630
-/
theorem geography_textbook_cost :
  let num_books : ℕ := 35
  let english_book_cost : ℚ := 7.5
  let total_cost : ℚ := 630
  let geography_book_cost : ℚ := (total_cost - num_books * english_book_cost) / num_books
  geography_book_cost = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_geography_textbook_cost_l2124_212423


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2124_212414

theorem gcd_polynomial_and_multiple (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a^3 + 2 * a^2 + 6 * a + 76) a = 76 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l2124_212414


namespace NUMINAMATH_CALUDE_limit_special_function_l2124_212471

/-- The limit of (x^2 + 2x - 3) / (x^2 + 4x - 5) raised to the power of 1 / (2-x) as x approaches 1 is equal to 2/3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(((x^2 + 2*x - 3) / (x^2 + 4*x - 5))^(1/(2-x))) - (2/3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_special_function_l2124_212471


namespace NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l2124_212496

theorem smallest_n_sqrt_difference (n : ℕ) : 
  (n ≥ 2501) ↔ (Real.sqrt n - Real.sqrt (n - 1) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l2124_212496


namespace NUMINAMATH_CALUDE_janet_owes_22000_l2124_212419

/-- Calculates the total amount Janet owes for wages and taxes for one month -/
def total_owed (warehouse_workers : ℕ) (managers : ℕ) (warehouse_wage : ℚ) (manager_wage : ℚ)
  (days_per_month : ℕ) (hours_per_day : ℕ) (fica_tax_rate : ℚ) : ℚ :=
  let total_hours := days_per_month * hours_per_day
  let warehouse_total := warehouse_workers * warehouse_wage * total_hours
  let manager_total := managers * manager_wage * total_hours
  let total_wages := warehouse_total + manager_total
  let fica_taxes := total_wages * fica_tax_rate
  total_wages + fica_taxes

theorem janet_owes_22000 :
  total_owed 4 2 15 20 25 8 (1/10) = 22000 := by
  sorry

end NUMINAMATH_CALUDE_janet_owes_22000_l2124_212419


namespace NUMINAMATH_CALUDE_inequality_solutions_imply_range_l2124_212463

theorem inequality_solutions_imply_range (a : ℝ) : 
  (∃ x₁ x₂ : ℕ+, x₁ ≠ x₂ ∧ 
    (∀ x : ℕ+, 2 * (x : ℝ) + a ≤ 1 ↔ (x = x₁ ∨ x = x₂))) →
  -5 < a ∧ a ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_imply_range_l2124_212463


namespace NUMINAMATH_CALUDE_triangle_area_formula_right_angle_l2124_212455

theorem triangle_area_formula_right_angle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (1/2) * (a * b) / Real.sin (π/2) = (1/2) * a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_formula_right_angle_l2124_212455


namespace NUMINAMATH_CALUDE_M_intersect_N_l2124_212499

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2124_212499


namespace NUMINAMATH_CALUDE_range_of_a_l2124_212410

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 1) →
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a y < f a x) →
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2124_212410


namespace NUMINAMATH_CALUDE_complex_quadrant_l2124_212424

theorem complex_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2124_212424


namespace NUMINAMATH_CALUDE_league_games_count_l2124_212421

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem league_games_count :
  let total_teams : ℕ := 8
  let teams_per_game : ℕ := 2
  number_of_games total_teams = 28 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l2124_212421


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l2124_212441

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:2 ratio,
    and the coordinates of B and C are known, prove that the sum of A's coordinates is 9. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (B.1 - C.1) / (B.1 - A.1) = 2/3 →
  B = (2, 8) →
  C = (5, 2) →
  A.1 + A.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l2124_212441


namespace NUMINAMATH_CALUDE_triangle_similarity_and_area_l2124_212404

/-- Triangle similarity and area theorem -/
theorem triangle_similarity_and_area (PQ QR YZ : ℝ) (area_XYZ : ℝ) :
  PQ = 8 →
  QR = 16 →
  YZ = 24 →
  area_XYZ = 144 →
  ∃ (XY : ℝ),
    (XY / PQ = YZ / QR) ∧
    (area_XYZ = (1/2) * YZ * (2 * area_XYZ / YZ)) ∧
    XY = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_and_area_l2124_212404


namespace NUMINAMATH_CALUDE_percentage_of_boats_eaten_by_fish_l2124_212492

theorem percentage_of_boats_eaten_by_fish 
  (initial_boats : ℕ) 
  (shot_boats : ℕ) 
  (remaining_boats : ℕ) 
  (h1 : initial_boats = 30) 
  (h2 : shot_boats = 2) 
  (h3 : remaining_boats = 22) : 
  (initial_boats - shot_boats - remaining_boats) / initial_boats * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boats_eaten_by_fish_l2124_212492


namespace NUMINAMATH_CALUDE_missing_fibonacci_term_l2124_212498

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem missing_fibonacci_term : ∃ x : ℕ, 
  fibonacci 0 = 1 ∧ 
  fibonacci 1 = 1 ∧ 
  fibonacci 2 = 2 ∧ 
  fibonacci 3 = 3 ∧ 
  fibonacci 4 = 5 ∧ 
  fibonacci 5 = x ∧ 
  fibonacci 6 = 13 ∧ 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_fibonacci_term_l2124_212498


namespace NUMINAMATH_CALUDE_tan_alpha_2_implications_l2124_212495

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * (Real.sin α)^2 + (1/3) * Real.sin α * Real.cos α + (1/2) * (Real.cos α)^2 + 1 = 43/30 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implications_l2124_212495


namespace NUMINAMATH_CALUDE_travis_apple_sale_price_l2124_212447

/-- Calculates the price per box of apples given the total number of apples,
    apples per box, and desired total revenue. -/
def price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : ℚ :=
  (total_revenue : ℚ) / ((total_apples / apples_per_box) : ℚ)

/-- Proves that given Travis's conditions, he must sell each box for $35. -/
theorem travis_apple_sale_price :
  price_per_box 10000 50 7000 = 35 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sale_price_l2124_212447


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2124_212405

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Alyssa bought from Jason -/
def cards_bought : ℕ := 224

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := 452

/-- Theorem stating that Jason's initial number of Pokemon cards was 676 -/
theorem jason_initial_cards : initial_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2124_212405


namespace NUMINAMATH_CALUDE_kerosene_cost_calculation_l2124_212401

/-- The cost of a certain number of eggs in cents -/
def egg_cost : ℝ := sorry

/-- The cost of a pound of rice in cents -/
def rice_cost : ℝ := 24

/-- The cost of a half-liter of kerosene in cents -/
def half_liter_kerosene_cost : ℝ := sorry

/-- The cost of a liter of kerosene in cents -/
def liter_kerosene_cost : ℝ := sorry

theorem kerosene_cost_calculation :
  (egg_cost = rice_cost) →
  (half_liter_kerosene_cost = 6 * egg_cost) →
  (liter_kerosene_cost = 2 * half_liter_kerosene_cost) →
  liter_kerosene_cost = 288 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_cost_calculation_l2124_212401


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l2124_212409

/-- Represents a bowl of fruit with apples and oranges -/
structure FruitBowl where
  apples : ℕ
  oranges : ℕ

/-- Calculates the percentage of apples in a fruit bowl -/
def applePercentage (bowl : FruitBowl) : ℚ :=
  (bowl.apples : ℚ) / ((bowl.apples + bowl.oranges) : ℚ) * 100

theorem apple_percentage_after_removal :
  let initialBowl : FruitBowl := { apples := 12, oranges := 23 }
  let removedOranges : ℕ := 15
  let finalBowl : FruitBowl := { apples := initialBowl.apples, oranges := initialBowl.oranges - removedOranges }
  applePercentage finalBowl = 60 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l2124_212409


namespace NUMINAMATH_CALUDE_workshop_percentage_approx_29_l2124_212420

/-- Calculates the percentage of a work day spent in workshops -/
def workshop_percentage (work_day_hours : ℕ) (workshop1_minutes : ℕ) (workshop2_multiplier : ℕ) : ℚ :=
  let work_day_minutes : ℕ := work_day_hours * 60
  let workshop2_minutes : ℕ := workshop1_minutes * workshop2_multiplier
  let total_workshop_minutes : ℕ := workshop1_minutes + workshop2_minutes
  (total_workshop_minutes : ℚ) / (work_day_minutes : ℚ) * 100

/-- The percentage of the work day spent in workshops is approximately 29% -/
theorem workshop_percentage_approx_29 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |workshop_percentage 8 35 3 - 29| < ε :=
sorry

end NUMINAMATH_CALUDE_workshop_percentage_approx_29_l2124_212420


namespace NUMINAMATH_CALUDE_perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l2124_212422

-- Define two perpendicular planes
axiom Plane1 : Type
axiom Plane2 : Type
axiom perpendicular_planes : Plane1 → Plane2 → Prop

-- Define a line
axiom Line : Type

-- Define a line being in a plane
axiom line_in_plane : Line → Plane1 → Prop
axiom line_in_plane2 : Line → Plane2 → Prop

-- Define perpendicularity between lines
axiom perpendicular_lines : Line → Line → Prop

-- Define perpendicularity between a line and a plane
axiom perpendicular_line_plane : Line → Plane1 → Prop
axiom perpendicular_line_plane2 : Line → Plane2 → Prop

-- Define the intersection line of two planes
axiom intersection_line : Plane1 → Plane2 → Line

-- Define a point
axiom Point : Type

-- Define a point being in a plane
axiom point_in_plane : Point → Plane1 → Prop

-- Define drawing a perpendicular line from a point to a line
axiom perpendicular_from_point : Point → Line → Line

-- Theorem 1: A line in one plane must be perpendicular to countless lines in the other plane
theorem perpendicular_to_countless_lines 
  (p1 : Plane1) (p2 : Plane2) (l : Line) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  ∃ (S : Set Line), (∀ l' ∈ S, line_in_plane2 l' p2 ∧ perpendicular_lines l l') ∧ Set.Infinite S :=
sorry

-- Theorem 2: If a perpendicular to the intersection line is drawn from any point in one plane, 
-- then this perpendicular must be perpendicular to the other plane
theorem perpendicular_to_intersection_perpendicular_to_plane 
  (p1 : Plane1) (p2 : Plane2) (pt : Point) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : point_in_plane pt p1) :
  let i := intersection_line p1 p2
  let perp := perpendicular_from_point pt i
  perpendicular_line_plane2 perp p2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l2124_212422


namespace NUMINAMATH_CALUDE_range_of_m_l2124_212461

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |3 - x| + |5 + x| > m) ↔ m < 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2124_212461


namespace NUMINAMATH_CALUDE_susan_bought_sixty_peaches_l2124_212411

/-- Represents the number of peaches in Susan's knapsack -/
def knapsack_peaches : ℕ := 12

/-- Represents the number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Calculates the number of peaches in each cloth bag -/
def peaches_per_cloth_bag : ℕ := 2 * knapsack_peaches

/-- Calculates the total number of peaches Susan bought -/
def total_peaches : ℕ := num_cloth_bags * peaches_per_cloth_bag + knapsack_peaches

/-- Theorem stating that Susan bought 60 peaches in total -/
theorem susan_bought_sixty_peaches : total_peaches = 60 := by
  sorry

end NUMINAMATH_CALUDE_susan_bought_sixty_peaches_l2124_212411


namespace NUMINAMATH_CALUDE_ellipse_condition_l2124_212472

/-- A curve represented by the equation x²/(7-m) + y²/(m-3) = 1 is an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  7 - m > 0 ∧ m - 3 > 0 ∧ 7 - m ≠ m - 3

/-- The condition 3 < m < 7 is necessary but not sufficient for the curve to be an ellipse -/
theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → 3 < m ∧ m < 7) ∧
  ¬(3 < m ∧ m < 7 → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2124_212472


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l2124_212407

/-- Given a curve y = x^4 + ax + 1 with a tangent at (-1, a+2) having slope 8, prove a = -6 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + a*x + 1
  let point : ℝ × ℝ := (-1, a + 2)
  let slope : ℝ := 8
  (f (-1) = a + 2) ∧ 
  (deriv f (-1) = slope) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l2124_212407


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2124_212431

theorem average_weight_decrease (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) :
  n = 30 →
  initial_avg = 102 →
  new_weight = 40 →
  let total_weight := n * initial_avg
  let new_total_weight := total_weight + new_weight
  let new_avg := new_total_weight / (n + 1)
  initial_avg - new_avg = 2 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2124_212431


namespace NUMINAMATH_CALUDE_greg_ate_four_halves_l2124_212426

/-- Represents the number of whole cookies made -/
def total_cookies : ℕ := 14

/-- Represents the number of halves each cookie is cut into -/
def halves_per_cookie : ℕ := 2

/-- Represents the number of halves Brad ate -/
def brad_halves : ℕ := 6

/-- Represents the number of halves left -/
def left_halves : ℕ := 18

/-- Theorem stating that Greg ate 4 halves -/
theorem greg_ate_four_halves : 
  total_cookies * halves_per_cookie - brad_halves - left_halves = 4 := by
  sorry

end NUMINAMATH_CALUDE_greg_ate_four_halves_l2124_212426


namespace NUMINAMATH_CALUDE_min_sum_at_six_l2124_212476

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  h1 : a 1 + a 5 = -14  -- Given condition
  h2 : S 9 = -27  -- Given condition
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property

/-- The theorem stating that S_n is minimized when n = 6 -/
theorem min_sum_at_six (seq : ArithmeticSequence) : 
  ∃ (n : ℕ), ∀ (m : ℕ), seq.S n ≤ seq.S m ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_six_l2124_212476


namespace NUMINAMATH_CALUDE_percentage_relationship_l2124_212412

theorem percentage_relationship (p j t : ℝ) (r : ℝ) 
  (h1 : j = p * (1 - 0.25))
  (h2 : j = t * (1 - 0.20))
  (h3 : t = p * (1 - r / 100)) :
  r = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2124_212412


namespace NUMINAMATH_CALUDE_no_divisible_sum_difference_l2124_212469

theorem no_divisible_sum_difference : 
  ¬∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ 
  ((∃ k : ℤ, A = k * (A + B) ∧ ∃ m : ℤ, B = m * (A - B)) ∨
   (∃ k : ℤ, B = k * (A + B) ∧ ∃ m : ℤ, A = m * (A - B))) :=
by sorry

end NUMINAMATH_CALUDE_no_divisible_sum_difference_l2124_212469


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2124_212408

theorem trigonometric_identity (α : ℝ) : 
  4 * Real.sin (2 * α - 3/2 * Real.pi) * Real.sin (Real.pi/6 + 2 * α) * Real.sin (Real.pi/6 - 2 * α) = Real.cos (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2124_212408


namespace NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l2124_212432

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: If f(1+x) = f(1-x) for all x, then a = 1
theorem symmetry_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a (1+x) = f a (1-x)) → a = 1 := by sorry

-- Theorem 2: If f is monotonically increasing on [1, +∞), then a ≤ 1
theorem monotonic_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x < f a y) → a ≤ 1 := by sorry

-- Theorem 3: The maximum value of f on [-1, 1] is 2
theorem max_value_on_interval (a : ℝ) :
  ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ m := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l2124_212432


namespace NUMINAMATH_CALUDE_board_game_ratio_l2124_212474

theorem board_game_ratio (total_students : ℕ) (reading_students : ℕ) (homework_students : ℕ) :
  total_students = 24 →
  reading_students = total_students / 2 →
  homework_students = 4 →
  (total_students - (reading_students + homework_students)) * 3 = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_board_game_ratio_l2124_212474
