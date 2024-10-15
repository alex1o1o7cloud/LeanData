import Mathlib

namespace NUMINAMATH_CALUDE_rabbit_catches_cat_l1665_166552

/-- Proves that a rabbit catches up to a cat in 1 hour given their speeds and the cat's head start -/
theorem rabbit_catches_cat (rabbit_speed cat_speed : ℝ) (head_start : ℝ) : 
  rabbit_speed = 25 →
  cat_speed = 20 →
  head_start = 0.25 →
  (rabbit_speed - cat_speed) * 1 = cat_speed * head_start := by
  sorry

#check rabbit_catches_cat

end NUMINAMATH_CALUDE_rabbit_catches_cat_l1665_166552


namespace NUMINAMATH_CALUDE_full_merit_scholarship_percentage_l1665_166515

theorem full_merit_scholarship_percentage
  (total_students : ℕ)
  (half_merit_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : half_merit_percentage = 1 / 10)
  (h3 : no_scholarship_count = 255) :
  (total_students - (half_merit_percentage * total_students).floor - no_scholarship_count) / total_students = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_full_merit_scholarship_percentage_l1665_166515


namespace NUMINAMATH_CALUDE_conditional_probability_balls_l1665_166584

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of event A (drawing two balls of different colors) -/
def probA : ℚ := (choose 5 1 * choose 3 1 + choose 5 1 * choose 4 1 + choose 3 1 * choose 4 1) / choose 12 2

/-- The probability of event B (drawing one yellow and one blue ball) -/
def probB : ℚ := (choose 5 1 * choose 4 1) / choose 12 2

/-- The probability of both events A and B occurring -/
def probAB : ℚ := probB

theorem conditional_probability_balls :
  probAB / probA = 20 / 47 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_balls_l1665_166584


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1665_166544

/-- 
Given:
- A point P is rotated 750 degrees clockwise about point Q, resulting in point R.
- The same point P is rotated y degrees counterclockwise about point Q, also resulting in point R.
- y < 360

Prove that y = 330.
-/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (750 % 360 : ℝ) + y = 360 → y = 330 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1665_166544


namespace NUMINAMATH_CALUDE_not_even_if_unequal_l1665_166510

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem not_even_if_unequal :
  f (-2) ≠ f 2 → ¬(IsEven f) := by
  sorry

end NUMINAMATH_CALUDE_not_even_if_unequal_l1665_166510


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1665_166554

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (6 - x)) :
  is_symmetric_about f 3 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1665_166554


namespace NUMINAMATH_CALUDE_simplify_expression_l1665_166586

theorem simplify_expression (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1665_166586


namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_l1665_166597

theorem symmetric_sine_cosine (φ : ℝ) (h1 : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + φ) - Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (2*π - x) = f x) →
  Real.cos (2*φ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_l1665_166597


namespace NUMINAMATH_CALUDE_lcm_of_ratio_3_4_l1665_166591

/-- Given two natural numbers with a ratio of 3:4, where one number is 45 and the other is 60, 
    their least common multiple (LCM) is 180. -/
theorem lcm_of_ratio_3_4 (a b : ℕ) (h_ratio : 3 * b = 4 * a) (h_a : a = 45) (h_b : b = 60) : 
  Nat.lcm a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_3_4_l1665_166591


namespace NUMINAMATH_CALUDE_square_sum_implies_abs_sum_l1665_166585

theorem square_sum_implies_abs_sum (a b : ℝ) :
  a^2 + b^2 > 1 → |a| + |b| > 1 := by sorry

end NUMINAMATH_CALUDE_square_sum_implies_abs_sum_l1665_166585


namespace NUMINAMATH_CALUDE_system_equivalent_to_line_l1665_166587

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system is equivalent to the solution line -/
theorem system_equivalent_to_line :
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end NUMINAMATH_CALUDE_system_equivalent_to_line_l1665_166587


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l1665_166562

/-- The number of strawberries in each basket, given the conditions of the hedgehog problem -/
theorem hedgehog_strawberries (num_hedgehogs : ℕ) (num_baskets : ℕ) 
  (strawberries_per_hedgehog : ℕ) (remaining_fraction : ℚ) :
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_hedgehog = 1050 →
  remaining_fraction = 2/9 →
  ∃ (total_strawberries : ℕ),
    total_strawberries = num_hedgehogs * strawberries_per_hedgehog / (1 - remaining_fraction) ∧
    total_strawberries / num_baskets = 900 :=
by sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l1665_166562


namespace NUMINAMATH_CALUDE_line_contains_point_l1665_166588

/-- Given a line with equation -2/3 - 3kx = 7y that contains the point (1/3, -5), 
    prove that the value of k is 103/3. -/
theorem line_contains_point (k : ℚ) : 
  (-2/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-5 : ℚ) → k = 103/3 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l1665_166588


namespace NUMINAMATH_CALUDE_number_increased_by_45_percent_l1665_166577

theorem number_increased_by_45_percent (x : ℝ) : x * 1.45 = 870 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_increased_by_45_percent_l1665_166577


namespace NUMINAMATH_CALUDE_same_prime_factors_imply_power_of_two_l1665_166545

theorem same_prime_factors_imply_power_of_two (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_same_prime_factors_imply_power_of_two_l1665_166545


namespace NUMINAMATH_CALUDE_ellipse_properties_l1665_166517

/-- Represents an ellipse defined by the equation (x^2 / 36) + (y^2 / 9) = 4 -/
def Ellipse := {(x, y) : ℝ × ℝ | (x^2 / 36) + (y^2 / 9) = 4}

/-- The distance between the foci of the ellipse -/
def focalDistance (e : Set (ℝ × ℝ)) : ℝ := 
  5.196

/-- The eccentricity of the ellipse -/
def eccentricity (e : Set (ℝ × ℝ)) : ℝ := 
  0.866

theorem ellipse_properties : 
  focalDistance Ellipse = 5.196 ∧ eccentricity Ellipse = 0.866 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1665_166517


namespace NUMINAMATH_CALUDE_freshman_groups_l1665_166582

theorem freshman_groups (total_freshmen : Nat) (group_decrease : Nat) :
  total_freshmen = 2376 →
  group_decrease = 9 →
  ∃ (initial_groups final_groups : Nat),
    initial_groups = final_groups + group_decrease ∧
    total_freshmen % initial_groups = 0 ∧
    total_freshmen % final_groups = 0 ∧
    total_freshmen / final_groups < 30 ∧
    final_groups = 99 := by
  sorry

end NUMINAMATH_CALUDE_freshman_groups_l1665_166582


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_l1665_166593

/-- Given a parabola x² = 2py (p > 0) and a line y = 2x + p/2 that intersects
    the parabola at points A and B, prove that the length of AB is 10p. -/
theorem parabola_line_intersection_length (p : ℝ) (h : p > 0) : 
  let parabola := fun x y => x^2 = 2*p*y
  let line := fun x y => y = 2*x + p/2
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    ‖A - B‖ = 10*p :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_l1665_166593


namespace NUMINAMATH_CALUDE_stream_speed_l1665_166523

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat's travel. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 13 →
  downstream_distance = 68 →
  downstream_time = 4 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 17 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l1665_166523


namespace NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_prob_l1665_166507

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing up in a circular arrangement of n people, each flipping a fair coin. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_prob :
  noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_prob_l1665_166507


namespace NUMINAMATH_CALUDE_max_balls_in_specific_cylinder_l1665_166578

/-- The maximum number of unit balls that can be placed in a cylinder -/
def max_balls_in_cylinder (cylinder_diameter : ℝ) (cylinder_height : ℝ) (ball_diameter : ℝ) : ℕ :=
  sorry

/-- Theorem: In a cylinder with diameter √2 + 1 and height 8, the maximum number of balls with diameter 1 that can be placed is 36 -/
theorem max_balls_in_specific_cylinder :
  max_balls_in_cylinder (Real.sqrt 2 + 1) 8 1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_specific_cylinder_l1665_166578


namespace NUMINAMATH_CALUDE_basketball_score_calculation_l1665_166509

/-- Given a basketball player who made 7 two-point shots and 3 three-point shots,
    the total points scored is equal to 23. -/
theorem basketball_score_calculation (two_point_shots three_point_shots : ℕ) : 
  two_point_shots = 7 →
  three_point_shots = 3 →
  2 * two_point_shots + 3 * three_point_shots = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_calculation_l1665_166509


namespace NUMINAMATH_CALUDE_meter_to_skips_l1665_166508

theorem meter_to_skips 
  (b c d e f g : ℝ) 
  (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : b * 1 = c * 1)  -- b hops = c skips
  (jump_hop : d * 1 = e * 1)  -- d jumps = e hops
  (jump_meter : f * 1 = g * 1)  -- f jumps = g meters
  : 1 = (c * e * f) / (b * d * g) := by
  sorry

end NUMINAMATH_CALUDE_meter_to_skips_l1665_166508


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1665_166553

theorem potato_bag_weight (bag_weight : ℝ) (h : bag_weight = 36) :
  bag_weight / (bag_weight / 2) = 2 ∧ bag_weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1665_166553


namespace NUMINAMATH_CALUDE_two_lines_in_cube_l1665_166558

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  point : Point
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point : Point
  normal : ℝ × ℝ × ℝ

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

/-- Checks if a point is on an edge of the cube -/
def point_on_edge (c : Cube) (p : Point) : Prop := sorry

/-- Counts the number of lines passing through a point and making a specific angle with two planes -/
def count_lines (c : Cube) (p : Point) (angle : ℝ) (plane1 plane2 : Plane) : ℕ := sorry

/-- The main theorem statement -/
theorem two_lines_in_cube (c : Cube) (p : Point) :
  point_on_edge c p →
  let plane_abcd := Plane.mk sorry sorry
  let plane_abc1d1 := Plane.mk sorry sorry
  count_lines c p (30 * π / 180) plane_abcd plane_abc1d1 = 2 := by sorry

end NUMINAMATH_CALUDE_two_lines_in_cube_l1665_166558


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1665_166559

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) → 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1665_166559


namespace NUMINAMATH_CALUDE_weight_replacement_l1665_166524

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 80 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - (initial_count * average_increase) ∧
    old_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1665_166524


namespace NUMINAMATH_CALUDE_damien_jogging_distance_l1665_166567

/-- The number of miles Damien jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- The total distance Damien runs over three weeks -/
def total_distance : ℕ := miles_per_day * weekdays_per_week * num_weeks

theorem damien_jogging_distance :
  total_distance = 75 := by
  sorry

end NUMINAMATH_CALUDE_damien_jogging_distance_l1665_166567


namespace NUMINAMATH_CALUDE_combined_variance_is_100_l1665_166531

/-- Calculates the combined variance of two classes given their individual statistics -/
def combinedVariance (nA nB : ℕ) (meanA meanB : ℝ) (varA varB : ℝ) : ℝ :=
  let n := nA + nB
  let pA := nA / n
  let pB := nB / n
  let combinedMean := pA * meanA + pB * meanB
  pA * (varA + (meanA - combinedMean)^2) + pB * (varB + (meanB - combinedMean)^2)

/-- The variance of the combined scores of Class A and Class B is 100 -/
theorem combined_variance_is_100 :
  combinedVariance 50 40 76 85 96 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_variance_is_100_l1665_166531


namespace NUMINAMATH_CALUDE_min_coins_for_eternal_collection_l1665_166538

/-- Represents the JMO kingdom with its citizens and coin distribution. -/
structure Kingdom (n : ℕ) where
  /-- The number of citizens in the kingdom is 2^n. -/
  citizens : ℕ := 2^n
  /-- The value of paper bills used in the kingdom. -/
  bill_value : ℕ := 2^n
  /-- The possible values of coins in the kingdom. -/
  coin_values : List ℕ := List.range n |>.map (fun a => 2^a)

/-- The sum of digits function in base 2. -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of coins required for the king to collect money every night eternally. -/
theorem min_coins_for_eternal_collection (n : ℕ) (h : n > 0) : 
  ∃ (S : ℕ), S = n * 2^(n-1) ∧ 
  ∀ (S' : ℕ), S' < S → ¬(∃ (distribution : ℕ → ℕ), 
    (∀ i, i < 2^n → distribution i ≤ sum_of_digits i) ∧
    (∀ t : ℕ, ∃ (new_distribution : ℕ → ℕ), 
      (∀ i, i < 2^n → new_distribution i = distribution ((i + 1) % 2^n) + 1) ∧
      (∀ i, i < 2^n → new_distribution i ≤ sum_of_digits i))) :=
sorry

end NUMINAMATH_CALUDE_min_coins_for_eternal_collection_l1665_166538


namespace NUMINAMATH_CALUDE_man_crossing_street_speed_l1665_166547

/-- Proves that a man crossing a 600 m street in 5 minutes has a speed of 7.2 km/h -/
theorem man_crossing_street_speed :
  let distance_m : ℝ := 600
  let time_min : ℝ := 5
  let distance_km : ℝ := distance_m / 1000
  let time_h : ℝ := time_min / 60
  let speed_km_h : ℝ := distance_km / time_h
  speed_km_h = 7.2 := by sorry

end NUMINAMATH_CALUDE_man_crossing_street_speed_l1665_166547


namespace NUMINAMATH_CALUDE_apple_ratio_l1665_166504

def total_apples : ℕ := 496
def green_apples : ℕ := 124

theorem apple_ratio : 
  let red_apples := total_apples - green_apples
  (red_apples : ℚ) / green_apples = 93 / 31 := by
sorry

end NUMINAMATH_CALUDE_apple_ratio_l1665_166504


namespace NUMINAMATH_CALUDE_abs_neg_one_ninth_l1665_166550

theorem abs_neg_one_ninth : |(-1 : ℚ) / 9| = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_ninth_l1665_166550


namespace NUMINAMATH_CALUDE_sushi_cost_l1665_166590

theorem sushi_cost (j e : ℝ) (h1 : j + e = 200) (h2 : e = 9 * j) : e = 180 := by
  sorry

end NUMINAMATH_CALUDE_sushi_cost_l1665_166590


namespace NUMINAMATH_CALUDE_expansion_unique_solution_l1665_166555

/-- The number of terms in the expansion of (a+b+c+d+e+1)^n that include all five variables
    a, b, c, d, e, each to some positive power. -/
def numTerms (n : ℕ) : ℕ := Nat.choose n 5

/-- The proposition that 16 is the unique positive integer n such that the expansion of
    (a+b+c+d+e+1)^n contains exactly 2002 terms with all five variables a, b, c, d, e
    each to some positive power. -/
theorem expansion_unique_solution : 
  ∃! (n : ℕ), n > 0 ∧ numTerms n = 2002 ∧ n = 16 := by sorry

end NUMINAMATH_CALUDE_expansion_unique_solution_l1665_166555


namespace NUMINAMATH_CALUDE_no_common_complex_root_l1665_166542

theorem no_common_complex_root :
  ¬ ∃ (α : ℂ) (a b : ℚ), α^5 - α - 1 = 0 ∧ α^2 + a*α + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_complex_root_l1665_166542


namespace NUMINAMATH_CALUDE_set_no_duplicate_elements_l1665_166565

theorem set_no_duplicate_elements {α : Type*} (S : Set α) :
  ∀ x ∈ S, ∀ y ∈ S, x = y → x = y :=
by sorry

end NUMINAMATH_CALUDE_set_no_duplicate_elements_l1665_166565


namespace NUMINAMATH_CALUDE_focus_of_parabola_l1665_166519

/-- The focus of the parabola x = -1/4 * y^2 -/
def parabola_focus : ℝ × ℝ := (-1, 0)

/-- The equation of the parabola -/
def is_on_parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- Theorem stating that the focus of the parabola x = -1/4 * y^2 is at (-1, 0) -/
theorem focus_of_parabola :
  let (f, g) := parabola_focus
  ∀ (x y : ℝ), is_on_parabola x y →
    (x - f)^2 + y^2 = (x - (-f))^2 :=
by sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l1665_166519


namespace NUMINAMATH_CALUDE_simplify_abs_sum_l1665_166522

def second_quadrant (a b : ℝ) : Prop := a < 0 ∧ b > 0

theorem simplify_abs_sum (a b : ℝ) (h : second_quadrant a b) : 
  |a - b| + |b - a| = -2*a + 2*b := by
sorry

end NUMINAMATH_CALUDE_simplify_abs_sum_l1665_166522


namespace NUMINAMATH_CALUDE_optimal_investment_plan_l1665_166541

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment plan --/
structure InvestmentPlan where
  projectA : ℝ
  projectB : ℝ

def totalInvestment (plan : InvestmentPlan) : ℝ :=
  plan.projectA + plan.projectB

def potentialProfit (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxProfitRate + plan.projectB * projectB.maxProfitRate

def potentialLoss (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxLossRate + plan.projectB * projectB.maxLossRate

theorem optimal_investment_plan 
  (projectA : Project)
  (projectB : Project)
  (h_profitA : projectA.maxProfitRate = 1)
  (h_profitB : projectB.maxProfitRate = 0.5)
  (h_lossA : projectA.maxLossRate = 0.3)
  (h_lossB : projectB.maxLossRate = 0.1)
  (optimalPlan : InvestmentPlan)
  (h_optimalA : optimalPlan.projectA = 40000)
  (h_optimalB : optimalPlan.projectB = 60000) :
  (∀ plan : InvestmentPlan, 
    totalInvestment plan ≤ 100000 ∧ 
    potentialLoss plan projectA projectB ≤ 18000 →
    potentialProfit plan projectA projectB ≤ potentialProfit optimalPlan projectA projectB) ∧
  totalInvestment optimalPlan ≤ 100000 ∧
  potentialLoss optimalPlan projectA projectB ≤ 18000 :=
sorry

end NUMINAMATH_CALUDE_optimal_investment_plan_l1665_166541


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l1665_166506

theorem roots_opposite_signs (n : ℝ) : 
  n^2 + n - 1 = 0 → 
  ∃ (x : ℝ), x ≠ 0 ∧ 
    (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) ∧
    (-x^2 + (n-2)*(-x)) / (2*n*(-x) - 4) = (n+1) / (n-1) := by
  sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l1665_166506


namespace NUMINAMATH_CALUDE_hyperbola_parabola_coincidence_l1665_166537

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right vertex of the hyperbola
def hyperbola_right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

theorem hyperbola_parabola_coincidence (a : ℝ) (h : a > 0) :
  hyperbola_right_vertex a = parabola_focus → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_coincidence_l1665_166537


namespace NUMINAMATH_CALUDE_no_solution_exists_l1665_166533

-- Function to reverse a number
def reverseNumber (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists :
  ¬ ∃ (x : ℕ), x + 276 = 435 ∧ reverseNumber x = 731 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1665_166533


namespace NUMINAMATH_CALUDE_project_completion_theorem_l1665_166535

/-- The number of days to complete a project given two workers with different rates -/
def project_completion_days (a_rate b_rate : ℚ) (a_quit_before : ℕ) : ℕ :=
  let total_days := 20
  total_days

theorem project_completion_theorem (a_rate b_rate : ℚ) (a_quit_before : ℕ) :
  a_rate = 1/20 ∧ b_rate = 1/40 ∧ a_quit_before = 10 →
  (project_completion_days a_rate b_rate a_quit_before - a_quit_before) * a_rate +
  project_completion_days a_rate b_rate a_quit_before * b_rate = 1 :=
by
  sorry

#eval project_completion_days (1/20) (1/40) 10

end NUMINAMATH_CALUDE_project_completion_theorem_l1665_166535


namespace NUMINAMATH_CALUDE_stating_min_toothpicks_theorem_l1665_166572

/-- Represents a figure made of toothpicks and triangles -/
structure TriangleFigure where
  total_toothpicks : ℕ
  upward_1triangles : ℕ
  downward_1triangles : ℕ
  upward_2triangles : ℕ

/-- 
  Given a TriangleFigure, calculates the minimum number of toothpicks 
  that must be removed to eliminate all triangles
-/
def min_toothpicks_to_remove (figure : TriangleFigure) : ℕ :=
  sorry

/-- 
  Theorem stating that for the given figure, 
  the minimum number of toothpicks to remove is 15
-/
theorem min_toothpicks_theorem (figure : TriangleFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.upward_1triangles = 22)
  (h3 : figure.downward_1triangles = 14)
  (h4 : figure.upward_2triangles = 4) :
  min_toothpicks_to_remove figure = 15 :=
by sorry

end NUMINAMATH_CALUDE_stating_min_toothpicks_theorem_l1665_166572


namespace NUMINAMATH_CALUDE_peters_age_l1665_166536

theorem peters_age (peter_age jacob_age : ℕ) : 
  (peter_age - 10 = (jacob_age - 10) / 3) →
  (jacob_age = peter_age + 12) →
  peter_age = 16 := by
sorry

end NUMINAMATH_CALUDE_peters_age_l1665_166536


namespace NUMINAMATH_CALUDE_arkos_population_2070_l1665_166503

def population_growth (initial_population : ℕ) (start_year end_year doubling_period : ℕ) : ℕ :=
  initial_population * (2 ^ ((end_year - start_year) / doubling_period))

theorem arkos_population_2070 :
  population_growth 150 1960 2070 20 = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_arkos_population_2070_l1665_166503


namespace NUMINAMATH_CALUDE_problem_solution_l1665_166526

def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 1/2) ∧
  ((∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 1 → f x a ≤ |2*x + 1|) → 0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1665_166526


namespace NUMINAMATH_CALUDE_prob_hit_twice_in_three_shots_l1665_166500

/-- The probability of hitting a target exactly twice in three independent shots -/
theorem prob_hit_twice_in_three_shots 
  (p1 : Real) (p2 : Real) (p3 : Real)
  (h1 : p1 = 0.4) (h2 : p2 = 0.5) (h3 : p3 = 0.7) :
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 + p1 * (1 - p2) * p3 = 0.41 := by
sorry

end NUMINAMATH_CALUDE_prob_hit_twice_in_three_shots_l1665_166500


namespace NUMINAMATH_CALUDE_jack_plates_left_l1665_166574

def plates_left (flower_plates checked_plates striped_plates : ℕ) : ℕ :=
  let polka_plates := checked_plates ^ 2
  let wave_plates := (4 * checked_plates) / 9
  let smashed_flower := (flower_plates * 10) / 100
  let smashed_checked := (checked_plates * 15) / 100
  let smashed_striped := (striped_plates * 20) / 100
  flower_plates - smashed_flower + checked_plates - smashed_checked + 
  striped_plates - smashed_striped + polka_plates + wave_plates

theorem jack_plates_left : plates_left 6 9 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_jack_plates_left_l1665_166574


namespace NUMINAMATH_CALUDE_apples_in_basket_l1665_166561

/-- Calculates the number of apples remaining in a basket after removals. -/
def remaining_apples (initial : ℕ) (ricki_removal : ℕ) : ℕ :=
  initial - (ricki_removal + 2 * ricki_removal)

/-- Theorem stating that given the initial conditions, 32 apples remain. -/
theorem apples_in_basket : remaining_apples 74 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l1665_166561


namespace NUMINAMATH_CALUDE_probability_five_diamond_ace_l1665_166514

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define card properties
def isFive (card : StandardDeck) : Prop := sorry
def isDiamond (card : StandardDeck) : Prop := sorry
def isAce (card : StandardDeck) : Prop := sorry

-- Define the probability of drawing three specific cards
def probabilityOfDraw (deck : Type) (pred1 pred2 pred3 : deck → Prop) : ℚ := sorry

-- Theorem statement
theorem probability_five_diamond_ace :
  probabilityOfDraw StandardDeck isFive isDiamond isAce = 85 / 44200 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_diamond_ace_l1665_166514


namespace NUMINAMATH_CALUDE_gamma_value_l1665_166548

/-- Given that γ is directly proportional to the square of δ, 
    and γ = 25 when δ = 5, prove that γ = 64 when δ = 8 -/
theorem gamma_value (γ δ : ℝ) (h1 : ∃ (k : ℝ), ∀ x, γ = k * x^2) 
  (h2 : γ = 25 ∧ δ = 5) : 
  (δ = 8 → γ = 64) := by
  sorry


end NUMINAMATH_CALUDE_gamma_value_l1665_166548


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1665_166576

theorem simplify_sqrt_difference : 
  Real.sqrt 300 / Real.sqrt 75 - Real.sqrt 220 / Real.sqrt 55 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1665_166576


namespace NUMINAMATH_CALUDE_sophie_donuts_left_l1665_166528

/-- Calculates the number of donuts left for Sophie after giving some away --/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given_to_mom : ℕ) (donuts_given_to_sister : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_given_to_mom * donuts_per_box - donuts_given_to_sister

/-- Proves that Sophie has 30 donuts left --/
theorem sophie_donuts_left : donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_left_l1665_166528


namespace NUMINAMATH_CALUDE_contractor_problem_l1665_166549

/-- Represents the initial number of people hired by the contractor -/
def initial_people : ℕ := 10

/-- Represents the total number of days allocated for the job -/
def total_days : ℕ := 100

/-- Represents the number of days worked before firing people -/
def days_before_firing : ℕ := 20

/-- Represents the fraction of work completed before firing people -/
def work_fraction_before_firing : ℚ := 1/4

/-- Represents the number of people fired -/
def people_fired : ℕ := 2

/-- Represents the number of days needed to complete the job after firing people -/
def days_after_firing : ℕ := 75

theorem contractor_problem :
  ∃ (p : ℕ), 
    p = initial_people ∧
    p * days_before_firing = work_fraction_before_firing * (p * total_days) ∧
    (p - people_fired) * days_after_firing = (1 - work_fraction_before_firing) * (p * total_days) :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l1665_166549


namespace NUMINAMATH_CALUDE_square_division_perimeter_paradox_l1665_166518

theorem square_division_perimeter_paradox :
  ∃ (a : ℚ) (x : ℚ), 0 < x ∧ x < a ∧ 
    (2 * (a + x)).isInt ∧ 
    (2 * (2 * a - x)).isInt ∧ 
    ¬(4 * a).isInt := by
  sorry

end NUMINAMATH_CALUDE_square_division_perimeter_paradox_l1665_166518


namespace NUMINAMATH_CALUDE_smallest_power_l1665_166532

theorem smallest_power (a b c d : ℕ) : 
  2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_power_l1665_166532


namespace NUMINAMATH_CALUDE_waiter_new_customers_l1665_166502

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 47) 
  (h2 : customers_left = 41) 
  (h3 : final_customers = 26) : 
  final_customers - (initial_customers - customers_left) = 20 :=
by sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l1665_166502


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1665_166521

/-- The number of letters in the English alphabet -/
def alphabet_count : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_count - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The total number of possible license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count

theorem license_plate_combinations : license_plate_count = 22050 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1665_166521


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1665_166530

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) : 
  v + a + d = 93 →
  v = a + 9 →
  d = v + 9 →
  d = 40 := by
sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1665_166530


namespace NUMINAMATH_CALUDE_row_sum_equals_square_l1665_166513

theorem row_sum_equals_square (k : ℕ) (h : k > 0) : 
  let n := 2 * k - 1
  let a := k
  let l := 3 * k - 2
  (n * (a + l)) / 2 = (2 * k - 1)^2 := by
sorry

end NUMINAMATH_CALUDE_row_sum_equals_square_l1665_166513


namespace NUMINAMATH_CALUDE_quadratic_expression_l1665_166543

theorem quadratic_expression (m : ℤ) : 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = (m - 8) * (m + 3)) → 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = m^2 - 5*m - 24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_l1665_166543


namespace NUMINAMATH_CALUDE_apartment_number_change_l1665_166566

/-- Represents a building with apartments and entrances. -/
structure Building where
  num_entrances : ℕ
  apartments_per_entrance : ℕ

/-- Calculates the apartment number given the entrance number and apartment number within the entrance. -/
def apartment_number (b : Building) (entrance : ℕ) (apartment_in_entrance : ℕ) : ℕ :=
  (entrance - 1) * b.apartments_per_entrance + apartment_in_entrance

/-- Theorem stating that if an apartment's number changes from 636 to 242 when entrance numbering is reversed in a 5-entrance building, then the total number of apartments is 985. -/
theorem apartment_number_change (b : Building) 
  (h1 : b.num_entrances = 5)
  (h2 : ∃ (e1 e2 a : ℕ), 
    apartment_number b e1 a = 636 ∧ 
    apartment_number b (b.num_entrances - e1 + 1) a = 242) :
  b.num_entrances * b.apartments_per_entrance = 985 := by
  sorry

#check apartment_number_change

end NUMINAMATH_CALUDE_apartment_number_change_l1665_166566


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l1665_166546

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l1665_166546


namespace NUMINAMATH_CALUDE_solution_of_exponential_equation_l1665_166564

theorem solution_of_exponential_equation :
  ∃ x : ℝ, (2 : ℝ)^(x - 3) = 8^(x + 1) ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_solution_of_exponential_equation_l1665_166564


namespace NUMINAMATH_CALUDE_walk_in_closet_doorway_width_l1665_166575

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening (door or window) -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width + room.length) * room.height

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ := rectangleArea opening.width opening.height

theorem walk_in_closet_doorway_width 
  (room : RoomDimensions)
  (doorway1 : Opening)
  (window : Opening)
  (closetDoorwayHeight : ℝ)
  (areaToPaint : ℝ)
  (h1 : room.width = 20)
  (h2 : room.length = 20)
  (h3 : room.height = 8)
  (h4 : doorway1.width = 3)
  (h5 : doorway1.height = 7)
  (h6 : window.width = 6)
  (h7 : window.height = 4)
  (h8 : closetDoorwayHeight = 7)
  (h9 : areaToPaint = 560) :
  ∃ (closetDoorwayWidth : ℝ), 
    closetDoorwayWidth = 5 ∧
    areaToPaint = totalWallArea room - openingArea doorway1 - openingArea window - rectangleArea closetDoorwayWidth closetDoorwayHeight :=
by sorry

end NUMINAMATH_CALUDE_walk_in_closet_doorway_width_l1665_166575


namespace NUMINAMATH_CALUDE_inequality_statements_l1665_166598

theorem inequality_statements :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a > b ↔ a^3 > b^3) ∧
  (∃ a b : ℝ, a > b ∧ |a| ≤ |b|) ∧
  (∃ a b c : ℝ, a * c^2 ≤ b * c^2 ∧ a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_l1665_166598


namespace NUMINAMATH_CALUDE_min_box_value_l1665_166512

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 32 * x^2 + box * x + 32) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  (∃ a' b' box', (∀ x, (a' * x + b') * (b' * x + a') = 32 * x^2 + box' * x + 32) ∧
                 a' ≠ b' ∧ a' ≠ box' ∧ b' ≠ box' ∧
                 box' ≥ 80) →
  box ≥ 80 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l1665_166512


namespace NUMINAMATH_CALUDE_unique_a_value_l1665_166501

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1665_166501


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1665_166595

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -8)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1665_166595


namespace NUMINAMATH_CALUDE_apple_tree_yield_l1665_166539

theorem apple_tree_yield (apple_trees peach_trees : ℕ) 
  (peach_yield total_yield : ℝ) (h1 : apple_trees = 30) 
  (h2 : peach_trees = 45) (h3 : peach_yield = 65) 
  (h4 : total_yield = 7425) : 
  (total_yield - peach_trees * peach_yield) / apple_trees = 150 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_yield_l1665_166539


namespace NUMINAMATH_CALUDE_nancy_boots_count_l1665_166527

theorem nancy_boots_count :
  ∀ (B : ℕ),
  (∃ (S H : ℕ),
    S = B + 9 ∧
    H = 3 * (S + B) ∧
    2 * B + 2 * S + 2 * H = 168) →
  B = 6 := by
sorry

end NUMINAMATH_CALUDE_nancy_boots_count_l1665_166527


namespace NUMINAMATH_CALUDE_translated_points_exponent_l1665_166569

/-- Given two points A and B, and their translations A₁ and B₁, prove that a^b = 32 -/
theorem translated_points_exponent (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) : 
  A = (-1, 3) → 
  B = (2, -3) → 
  A₁ = (a, 1) → 
  B₁ = (5, -b) → 
  A₁.1 - A.1 = 3 → 
  A.2 - A₁.2 = 2 → 
  B₁.1 - B.1 = 3 → 
  B.2 - B₁.2 = 2 → 
  a^b = 32 := by
sorry

end NUMINAMATH_CALUDE_translated_points_exponent_l1665_166569


namespace NUMINAMATH_CALUDE_banana_count_l1665_166570

theorem banana_count (apples oranges total : ℕ) (h1 : apples = 9) (h2 : oranges = 15) (h3 : total = 146) :
  ∃ bananas : ℕ, 
    3 * (apples + oranges + bananas) + (apples - 2 + oranges - 2 + bananas - 2) = total ∧ 
    bananas = 52 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l1665_166570


namespace NUMINAMATH_CALUDE_merchant_discount_l1665_166571

/-- Prove that given a 75% markup and a 57.5% profit after discount, the discount offered is 10%. -/
theorem merchant_discount (C : ℝ) (C_pos : C > 0) : 
  let M := 1.75 * C  -- Marked up price (75% markup)
  let S := 1.575 * C -- Selling price (57.5% profit)
  let D := (M - S) / M * 100 -- Discount percentage
  D = 10 := by sorry

end NUMINAMATH_CALUDE_merchant_discount_l1665_166571


namespace NUMINAMATH_CALUDE_building_floors_upper_bound_l1665_166560

theorem building_floors_upper_bound 
  (num_elevators : ℕ) 
  (floors_per_elevator : ℕ) 
  (h1 : num_elevators = 7)
  (h2 : floors_per_elevator = 6)
  (h3 : ∀ (f1 f2 : ℕ), f1 ≠ f2 → ∃ (e : ℕ), e ≤ num_elevators ∧ 
    (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) :
  ∃ (max_floors : ℕ), max_floors ≤ 14 ∧ 
    ∀ (n : ℕ), (∀ (f1 f2 : ℕ), f1 ≤ n ∧ f2 ≤ n ∧ f1 ≠ f2 → 
      ∃ (e : ℕ), e ≤ num_elevators ∧ 
        (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) → 
    n ≤ max_floors := by
  sorry

end NUMINAMATH_CALUDE_building_floors_upper_bound_l1665_166560


namespace NUMINAMATH_CALUDE_triangle_side_length_l1665_166594

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 120) (h3 : b = 45) (h4 : c = 15) 
  (side_b : ℝ) (h5 : side_b = 4 * Real.sqrt 6) : 
  side_b * Real.sin a / Real.sin b = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1665_166594


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1665_166525

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1665_166525


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1665_166592

theorem quadratic_equation_solution :
  let equation := fun x : ℂ => 3 * x^2 + 7 - (6 * x - 4)
  let solution1 := 1 + (2 * Real.sqrt 6 / 3) * I
  let solution2 := 1 - (2 * Real.sqrt 6 / 3) * I
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 6 / 3
  (equation solution1 = 0) ∧
  (equation solution2 = 0) ∧
  (a + b^2 = 11/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1665_166592


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1665_166599

theorem arithmetic_calculations : 
  ((-23) + 13 - 12 = -22) ∧ 
  ((-2)^3 / 4 + 3 * (-5) = -17) ∧ 
  ((-24) * (1/2 - 3/4 - 1/8) = 9) ∧ 
  ((2-7) / 5^2 + (-1)^2023 * (1/10) = -3/10) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1665_166599


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1665_166596

/-- Given two positive integers with LCM 420 and in the ratio 4:7, prove their sum is 165 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 420 → a * 7 = b * 4 → a + b = 165 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1665_166596


namespace NUMINAMATH_CALUDE_catch_up_distance_l1665_166563

/-- Prove that B catches up with A 200 km from the start -/
theorem catch_up_distance (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 10 → 
  speed_B = 20 → 
  time_diff = 10 → 
  speed_B * (time_diff + (speed_B * time_diff - speed_A * time_diff) / (speed_B - speed_A)) = 200 := by
  sorry

#check catch_up_distance

end NUMINAMATH_CALUDE_catch_up_distance_l1665_166563


namespace NUMINAMATH_CALUDE_polynomial_equality_l1665_166589

theorem polynomial_equality (x : ℝ) (h : ℝ → ℝ) : 
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 → 
  h x = -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1665_166589


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1665_166520

theorem arithmetic_simplification : 2000 - 80 + 200 - 120 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1665_166520


namespace NUMINAMATH_CALUDE_hanks_reading_time_l1665_166529

/-- Represents Hank's weekly reading habits -/
structure ReadingHabits where
  weekdayMorningMinutes : ℕ
  weekdayEveningMinutes : ℕ
  weekdayDays : ℕ
  weekendMultiplier : ℕ

/-- Calculates the total reading time in minutes for a week -/
def totalReadingTime (habits : ReadingHabits) : ℕ :=
  let weekdayTotal := habits.weekdayDays * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  let weekendDays := 7 - habits.weekdayDays
  let weekendTotal := weekendDays * habits.weekendMultiplier * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  weekdayTotal + weekendTotal

/-- Theorem stating that Hank's total reading time in a week is 810 minutes -/
theorem hanks_reading_time :
  let hanksHabits : ReadingHabits := {
    weekdayMorningMinutes := 30,
    weekdayEveningMinutes := 60,
    weekdayDays := 5,
    weekendMultiplier := 2
  }
  totalReadingTime hanksHabits = 810 := by
  sorry


end NUMINAMATH_CALUDE_hanks_reading_time_l1665_166529


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1665_166534

/-- Given two vectors a and b in ℝ², prove that when a = (1,3) and b = (x,1) are perpendicular, x = -3 -/
theorem perpendicular_vectors (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i, i < 2 → a i * b i = 0) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1665_166534


namespace NUMINAMATH_CALUDE_topology_check_l1665_166583

-- Define the set X
def X : Set Char := {'{', 'a', 'b', 'c', '}'}

-- Define the four sets v
def v1 : Set (Set Char) := {∅, {'a'}, {'c'}, {'a', 'b', 'c'}}
def v2 : Set (Set Char) := {∅, {'b'}, {'c'}, {'b', 'c'}, {'a', 'b', 'c'}}
def v3 : Set (Set Char) := {∅, {'a'}, {'a', 'b'}, {'a', 'c'}}
def v4 : Set (Set Char) := {∅, {'a', 'c'}, {'b', 'c'}, {'c'}, {'a', 'b', 'c'}}

-- Define the topology property
def is_topology (v : Set (Set Char)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋃₀ S ∈ v) ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋂₀ S ∈ v)

-- Theorem statement
theorem topology_check :
  is_topology v2 ∧ is_topology v4 ∧ ¬is_topology v1 ∧ ¬is_topology v3 :=
sorry

end NUMINAMATH_CALUDE_topology_check_l1665_166583


namespace NUMINAMATH_CALUDE_steves_cookies_l1665_166568

theorem steves_cookies (total_spent milk_cost cereal_cost banana_cost apple_cost : ℚ)
  (cereal_boxes banana_count apple_count : ℕ)
  (h_total : total_spent = 25)
  (h_milk : milk_cost = 3)
  (h_cereal : cereal_cost = 7/2)
  (h_cereal_boxes : cereal_boxes = 2)
  (h_banana : banana_cost = 1/4)
  (h_banana_count : banana_count = 4)
  (h_apple : apple_cost = 1/2)
  (h_apple_count : apple_count = 4)
  (h_cookie_cost : ∀ x, x = 2 * milk_cost) :
  ∃ (cookie_boxes : ℕ), cookie_boxes = 2 ∧
    total_spent = milk_cost + cereal_cost * cereal_boxes + 
      banana_cost * banana_count + apple_cost * apple_count + 
      (2 * milk_cost) * cookie_boxes :=
by sorry

end NUMINAMATH_CALUDE_steves_cookies_l1665_166568


namespace NUMINAMATH_CALUDE_CD_distance_l1665_166516

-- Define the points on a line
variable (A B C D : ℝ)

-- Define the order of points on the line
axiom order : A ≤ B ∧ B ≤ C ∧ C ≤ D

-- Define the given distances
axiom AB_dist : B - A = 2
axiom AC_dist : C - A = 5
axiom BD_dist : D - B = 6

-- Theorem to prove
theorem CD_distance : D - C = 3 := by
  sorry

end NUMINAMATH_CALUDE_CD_distance_l1665_166516


namespace NUMINAMATH_CALUDE_max_value_z_plus_x_l1665_166511

theorem max_value_z_plus_x :
  ∀ x y z t : ℝ,
  x^2 + y^2 = 4 →
  z^2 + t^2 = 9 →
  x*t + y*z ≥ 6 →
  z + x ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_z_plus_x_l1665_166511


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1665_166556

theorem diophantine_equation_solution :
  ∀ (p q r k : ℕ),
    p.Prime ∧ q.Prime ∧ r.Prime ∧ k > 0 →
    p^2 + q^2 + 49*r^2 = 9*k^2 - 101 →
    ((p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1665_166556


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l1665_166540

/-- Given a sequence of sums of powers of a and b, prove that a^7 + b^7 = 29 -/
theorem sum_of_seventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l1665_166540


namespace NUMINAMATH_CALUDE_mutually_exclusive_pairs_l1665_166579

-- Define a type for events
inductive Event
| SevenRing
| EightRing
| AtLeastOneHit
| AHitsBMisses
| AtLeastOneBlack
| BothRed
| NoBlack
| ExactlyOneRed

-- Define a function to check if two events are mutually exclusive
def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ¬(∃ (outcome : Set Event), outcome.Nonempty ∧ e1 ∈ outcome ∧ e2 ∈ outcome)

-- Define the pairs of events
def pair1 : (Event × Event) := (Event.SevenRing, Event.EightRing)
def pair2 : (Event × Event) := (Event.AtLeastOneHit, Event.AHitsBMisses)
def pair3 : (Event × Event) := (Event.AtLeastOneBlack, Event.BothRed)
def pair4 : (Event × Event) := (Event.NoBlack, Event.ExactlyOneRed)

-- State the theorem
theorem mutually_exclusive_pairs :
  mutuallyExclusive pair1.1 pair1.2 ∧
  ¬(mutuallyExclusive pair2.1 pair2.2) ∧
  mutuallyExclusive pair3.1 pair3.2 ∧
  mutuallyExclusive pair4.1 pair4.2 := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_pairs_l1665_166579


namespace NUMINAMATH_CALUDE_outfit_combinations_l1665_166557

/-- The number of available shirts, pants, and hats -/
def num_items : ℕ := 7

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_combinations : valid_outfits = 336 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1665_166557


namespace NUMINAMATH_CALUDE_greatest_integer_b_no_real_roots_l1665_166551

theorem greatest_integer_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 10 ≠ 0) → b ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_no_real_roots_l1665_166551


namespace NUMINAMATH_CALUDE_sector_angle_l1665_166580

/-- Given a circular sector with arc length and area both equal to 6,
    prove that the central angle in radians is 3. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  r * α = 6 →  -- arc length formula
  (1 / 2) * r * α = 6 →  -- area formula
  α = 3 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l1665_166580


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1665_166573

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 10 → 
  ∀ x, x^2 - 16*x + 100 = 0 ↔ (x = a ∨ x = b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1665_166573


namespace NUMINAMATH_CALUDE_least_multiple_of_29_above_500_l1665_166505

theorem least_multiple_of_29_above_500 : 
  ∀ n : ℕ, n > 0 ∧ 29 ∣ n ∧ n > 500 → n ≥ 522 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_29_above_500_l1665_166505


namespace NUMINAMATH_CALUDE_jade_cal_difference_l1665_166581

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions / 10)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 83

-- Theorem to prove
theorem jade_cal_difference : jade_transactions - cal_transactions = 17 := by
  sorry

end NUMINAMATH_CALUDE_jade_cal_difference_l1665_166581
