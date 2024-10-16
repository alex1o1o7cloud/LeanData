import Mathlib

namespace NUMINAMATH_CALUDE_dolphins_score_l1984_198441

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 72 →
  sharks_points - dolphins_points = 20 →
  sharks_points ≥ 2 * dolphins_points →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 26 := by
sorry

end NUMINAMATH_CALUDE_dolphins_score_l1984_198441


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l1984_198408

-- Define a complex number z
def z (a : ℝ) : ℂ := Complex.I * (1 + a * Complex.I)

-- State the theorem
theorem pure_imaginary_implies_a_zero (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_zero_l1984_198408


namespace NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l1984_198400

def family_weights (total_weight daughter_weight daughter_child_weight : ℝ) : Prop :=
  total_weight = 120 ∧ daughter_child_weight = 60 ∧ daughter_weight = 48

theorem child_grandmother_weight_ratio 
  (total_weight daughter_weight daughter_child_weight : ℝ) 
  (h : family_weights total_weight daughter_weight daughter_child_weight) : 
  (daughter_child_weight - daughter_weight) / (total_weight - daughter_child_weight) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l1984_198400


namespace NUMINAMATH_CALUDE_digit_squaring_l1984_198442

theorem digit_squaring (A B C : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ B ≠ C →
  (A + 1 > 1) →
  (A * (A + 1)^3 + A * (A + 1)^2 + A * (A + 1) + A)^2 = 
    A * (A + 1)^7 + A * (A + 1)^6 + A * (A + 1)^5 + B * (A + 1)^4 + C * (A + 1)^3 + C * (A + 1)^2 + C * (A + 1) + B →
  A = 2 ∧ B = 1 ∧ C = 0 := by
sorry

end NUMINAMATH_CALUDE_digit_squaring_l1984_198442


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l1984_198495

theorem absolute_value_sum_zero (a b : ℝ) (h : |a - 3| + |b + 5| = 0) : 
  (a + b = -2) ∧ (|a| + |b| = 8) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l1984_198495


namespace NUMINAMATH_CALUDE_four_point_lines_l1984_198407

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in a plane --/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Function to determine if a point is on a line --/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to count distinct lines determined by a set of points --/
def countDistinctLines (points : List Point) : ℕ := sorry

/-- The main theorem --/
theorem four_point_lines (A B C D : Point) :
  isOnLine C (Line.mk 1 0 (-A.x)) = false →
  isOnLine D (Line.mk 1 0 (-A.x)) = false →
  (countDistinctLines [A, B, C, D] = 6 ∨ countDistinctLines [A, B, C, D] = 4) :=
sorry

end NUMINAMATH_CALUDE_four_point_lines_l1984_198407


namespace NUMINAMATH_CALUDE_number_is_360_l1984_198422

theorem number_is_360 : ∃ x : ℝ, (0.5 * x = 180) ∧ (x = 360) := by
  sorry

end NUMINAMATH_CALUDE_number_is_360_l1984_198422


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1984_198479

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * 1 + 2 * a = 0) ↔ a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1984_198479


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1984_198415

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 →
  initial_mean = 150 →
  incorrect_value = 135 →
  correct_value = 165 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 151 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1984_198415


namespace NUMINAMATH_CALUDE_bird_count_l1984_198451

theorem bird_count (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  stones = 40 →
  trees = 3 * stones + stones →
  birds = 2 * (trees + stones) →
  birds = 400 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l1984_198451


namespace NUMINAMATH_CALUDE_building_heights_sum_l1984_198421

/-- The heights of four buildings, where each subsequent building is a fraction of the height of the previous one. -/
def building_heights (h₁ : ℝ) (f₂ f₃ f₄ : ℝ) : Fin 4 → ℝ
  | 0 => h₁
  | 1 => h₁ * f₂
  | 2 => h₁ * f₂ * f₃
  | 3 => h₁ * f₂ * f₃ * f₄

/-- The sum of the heights of four buildings. -/
def total_height (h₁ : ℝ) (f₂ f₃ f₄ : ℝ) : ℝ :=
  (building_heights h₁ f₂ f₃ f₄ 0) +
  (building_heights h₁ f₂ f₃ f₄ 1) +
  (building_heights h₁ f₂ f₃ f₄ 2) +
  (building_heights h₁ f₂ f₃ f₄ 3)

theorem building_heights_sum :
  total_height 100 (1/2) (1/2) (1/5) = 180 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l1984_198421


namespace NUMINAMATH_CALUDE_least_m_for_x_bound_l1984_198456

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_for_x_bound :
  ∃ m : ℕ, m = 33 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k < m, x k > 3 + 1 / 2^10 :=
sorry

end NUMINAMATH_CALUDE_least_m_for_x_bound_l1984_198456


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l1984_198499

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l1984_198499


namespace NUMINAMATH_CALUDE_gas_cost_problem_l1984_198468

theorem gas_cost_problem (x : ℝ) : 
  (x / 4 - x / 7 = 15) → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_problem_l1984_198468


namespace NUMINAMATH_CALUDE_share_difference_example_l1984_198404

/-- Given a total profit and proportions for distribution among three parties,
    calculate the difference between the shares of the second and third parties. -/
def shareDifference (totalProfit : ℕ) (propA propB propC : ℕ) : ℕ :=
  let totalParts := propA + propB + propC
  let partValue := totalProfit / totalParts
  let shareB := propB * partValue
  let shareC := propC * partValue
  shareC - shareB

/-- Prove that for a total profit of 20000 distributed in the proportion 2:3:5,
    the difference between C's and B's shares is 4000. -/
theorem share_difference_example : shareDifference 20000 2 3 5 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_example_l1984_198404


namespace NUMINAMATH_CALUDE_constant_sum_perpendicular_distances_l1984_198478

/-- A regular pentagon with circumradius R -/
structure RegularPentagon where
  R : ℝ
  R_pos : R > 0

/-- A point inside a regular pentagon -/
structure InnerPoint (p : RegularPentagon) where
  x : ℝ
  y : ℝ
  inside : x^2 + y^2 < p.R^2

/-- The sum of perpendicular distances from a point to the sides of a regular pentagon -/
noncomputable def sum_perpendicular_distances (p : RegularPentagon) (k : InnerPoint p) : ℝ :=
  sorry

/-- Theorem stating that the sum of perpendicular distances is constant -/
theorem constant_sum_perpendicular_distances (p : RegularPentagon) :
  ∃ (c : ℝ), ∀ (k : InnerPoint p), sum_perpendicular_distances p k = c :=
sorry

end NUMINAMATH_CALUDE_constant_sum_perpendicular_distances_l1984_198478


namespace NUMINAMATH_CALUDE_matt_total_skips_l1984_198401

/-- The number of skips per second -/
def skips_per_second : ℕ := 3

/-- The duration of jumping in minutes -/
def jump_duration : ℕ := 10

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem: Given the conditions, Matt's total number of skips is 1800 -/
theorem matt_total_skips :
  skips_per_second * jump_duration * seconds_per_minute = 1800 := by
  sorry

end NUMINAMATH_CALUDE_matt_total_skips_l1984_198401


namespace NUMINAMATH_CALUDE_total_marbles_after_exchanges_l1984_198470

def initial_green : ℕ := 32
def initial_violet : ℕ := 38
def initial_blue : ℕ := 46

def mike_takes_green : ℕ := 23
def mike_gives_red : ℕ := 15
def alice_takes_violet : ℕ := 15
def alice_gives_yellow : ℕ := 20
def bob_takes_blue : ℕ := 31
def bob_gives_white : ℕ := 12

def mike_returns_green : ℕ := 10
def mike_takes_red : ℕ := 7
def alice_returns_violet : ℕ := 8
def alice_takes_yellow : ℕ := 9
def bob_returns_blue : ℕ := 17
def bob_takes_white : ℕ := 5

def final_green : ℕ := initial_green - mike_takes_green + mike_returns_green
def final_violet : ℕ := initial_violet - alice_takes_violet + alice_returns_violet
def final_blue : ℕ := initial_blue - bob_takes_blue + bob_returns_blue
def final_red : ℕ := mike_gives_red - mike_takes_red
def final_yellow : ℕ := alice_gives_yellow - alice_takes_yellow
def final_white : ℕ := bob_gives_white - bob_takes_white

theorem total_marbles_after_exchanges :
  final_green + final_violet + final_blue + final_red + final_yellow + final_white = 108 :=
by sorry

end NUMINAMATH_CALUDE_total_marbles_after_exchanges_l1984_198470


namespace NUMINAMATH_CALUDE_total_coronavirus_cases_l1984_198477

-- Define the number of cases for each state
def new_york_cases : ℕ := 2000
def california_cases : ℕ := new_york_cases / 2
def texas_cases : ℕ := california_cases - 400

-- Theorem to prove
theorem total_coronavirus_cases : 
  new_york_cases + california_cases + texas_cases = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_coronavirus_cases_l1984_198477


namespace NUMINAMATH_CALUDE_simplify_expression_solve_quadratic_equation_l1984_198489

-- Part 1
theorem simplify_expression :
  Real.sqrt 18 / Real.sqrt 9 - Real.sqrt (1/4) * 2 * Real.sqrt 2 + Real.sqrt 32 = 4 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 2*x = 3 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_quadratic_equation_l1984_198489


namespace NUMINAMATH_CALUDE_heaviest_weight_in_geometric_progression_l1984_198461

/-- Given four weights in geometric progression, the heaviest can be found using a balance twice -/
theorem heaviest_weight_in_geometric_progression 
  (b : ℝ) (d : ℝ) (h_b_pos : b > 0) (h_d_gt_one : d > 1) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
    (∀ i : Fin 4, b * d ^ i.val ≤ b * d ^ 3) ∧
    (∀ i : Fin 4, i.val ≠ 3 → b * d ^ i.val < b * d ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_heaviest_weight_in_geometric_progression_l1984_198461


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1984_198426

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole1_diameter : ℝ := 4
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 3
  let hole_depth : ℝ := 6
  
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole1_volume := π * (hole1_diameter / 2) ^ 2 * hole_depth
  let hole2_volume := π * (hole2_diameter / 2) ^ 2 * hole_depth
  let hole3_volume := π * (hole3_diameter / 2) ^ 2 * hole_depth
  
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = 2242.5 * π :=
by sorry


end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1984_198426


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l1984_198465

theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  let original_volume := π * r^2 * h
  let new_volume := π * (3*r)^2 * (2*h)
  original_volume = 30 → new_volume = 540 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l1984_198465


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1984_198490

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1984_198490


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l1984_198402

theorem trigonometric_system_solution (θ : ℝ) (a b : ℝ) 
  (eq1 : Real.sin θ + Real.cos θ = a)
  (eq2 : Real.sin θ - Real.cos θ = b)
  (eq3 : Real.sin θ * Real.sin θ - Real.cos θ * Real.cos θ - Real.sin θ = -b * b) :
  ((a = Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = -Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = 1 ∧ b = -1) ∨
   (a = -1 ∧ b = 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l1984_198402


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1984_198460

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1984_198460


namespace NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l1984_198474

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 ≥ 2039 :=
sorry

theorem existence_of_minimum : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_existence_of_minimum_l1984_198474


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l1984_198493

/-- Calculates the total number of heartbeats for a cyclist during a race. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that a cyclist's heart beats 16800 times during a 35-mile race. -/
theorem cyclist_heartbeats :
  let heart_rate := 120  -- heartbeats per minute
  let race_distance := 35  -- miles
  let pace := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 16800 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_heartbeats_l1984_198493


namespace NUMINAMATH_CALUDE_arrange_85550_eq_16_l1984_198457

/-- The number of ways to arrange the digits of 85550 to form a 5-digit number -/
def arrange_85550 : ℕ :=
  let digits : Multiset ℕ := {8, 5, 5, 5, 0}
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  16

/-- Theorem stating that the number of ways to arrange the digits of 85550
    to form a 5-digit number is 16 -/
theorem arrange_85550_eq_16 : arrange_85550 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_85550_eq_16_l1984_198457


namespace NUMINAMATH_CALUDE_equation_solutions_l1984_198453

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = -2 ∧
    (x1 + 3)^2 = x1 + 3 ∧ (x2 + 3)^2 = x2 + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1984_198453


namespace NUMINAMATH_CALUDE_train_length_l1984_198413

/-- Given a train with constant speed that crosses a tree in 120 seconds
    and passes a 700m long platform in 190 seconds,
    the length of the train is 1200 meters. -/
theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length / 120 = speed) →
  ((train_length + 700) / 190 = speed) →
  train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1984_198413


namespace NUMINAMATH_CALUDE_fraction_addition_l1984_198403

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1984_198403


namespace NUMINAMATH_CALUDE_guitar_price_theorem_l1984_198455

-- Define the suggested retail price
variable (P : ℝ)

-- Define the prices at Guitar Center and Sweetwater
def guitar_center_price (P : ℝ) : ℝ := 0.85 * P + 100
def sweetwater_price (P : ℝ) : ℝ := 0.90 * P

-- State the theorem
theorem guitar_price_theorem (h : abs (guitar_center_price P - sweetwater_price P) = 50) : 
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_theorem_l1984_198455


namespace NUMINAMATH_CALUDE_new_customers_calculation_l1984_198463

theorem new_customers_calculation (initial_customers final_customers : ℕ) 
  (h1 : initial_customers = 3)
  (h2 : final_customers = 8) :
  final_customers - initial_customers = 5 := by
  sorry

end NUMINAMATH_CALUDE_new_customers_calculation_l1984_198463


namespace NUMINAMATH_CALUDE_probability_of_meeting_theorem_l1984_198484

/-- Represents the practice schedule of a person --/
structure PracticeSchedule where
  start_time : ℝ
  duration : ℝ

/-- Represents the practice schedules of two people over multiple days --/
structure PracticeScenario where
  your_schedule : PracticeSchedule
  friend_schedule : PracticeSchedule
  num_days : ℕ

/-- Calculates the probability of meeting given two practice schedules --/
def probability_of_meeting (s : PracticeScenario) : ℝ :=
  sorry

/-- Calculates the probability of meeting on at least k days out of n days --/
def probability_of_meeting_at_least (s : PracticeScenario) (k : ℕ) : ℝ :=
  sorry

theorem probability_of_meeting_theorem :
  let s : PracticeScenario := {
    your_schedule := { start_time := 0, duration := 3 },
    friend_schedule := { start_time := 5, duration := 1 },
    num_days := 5
  }
  probability_of_meeting_at_least s 2 = 232 / 243 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_meeting_theorem_l1984_198484


namespace NUMINAMATH_CALUDE_expression_equals_one_l1984_198450

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  let x := b^2 + c^2 - b - c + 1 + b*c
  (a^2*b^2) / (x^2) + (a^2*c^2) / (x^2) + (b^2*c^2) / (x^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1984_198450


namespace NUMINAMATH_CALUDE_park_animals_ratio_l1984_198428

theorem park_animals_ratio (lions leopards elephants : ℕ) : 
  lions = 200 →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 →
  lions = 2 * leopards :=
by
  sorry

end NUMINAMATH_CALUDE_park_animals_ratio_l1984_198428


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l1984_198472

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_six_sqrt_two_l1984_198472


namespace NUMINAMATH_CALUDE_salt_solution_weight_l1984_198405

/-- 
Given a salt solution with initial concentration of 10% and final concentration of 30%,
this theorem proves that if 28.571428571428573 kg of pure salt is added,
the initial weight of the solution was 100 kg.
-/
theorem salt_solution_weight 
  (initial_concentration : Real) 
  (final_concentration : Real)
  (added_salt : Real) 
  (initial_weight : Real) :
  initial_concentration = 0.10 →
  final_concentration = 0.30 →
  added_salt = 28.571428571428573 →
  initial_concentration * initial_weight + added_salt = 
    final_concentration * (initial_weight + added_salt) →
  initial_weight = 100 := by
  sorry

#check salt_solution_weight

end NUMINAMATH_CALUDE_salt_solution_weight_l1984_198405


namespace NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1984_198482

/-- A projective transformation on a straight line -/
structure ProjectiveTransformation :=
  (f : ℝ → ℝ)

/-- The property that a projective transformation preserves cross-ratio -/
def PreservesCrossRatio (t : ProjectiveTransformation) : Prop :=
  ∀ a b c d : ℝ, (a - c) * (b - d) / ((b - c) * (a - d)) = 
    (t.f a - t.f c) * (t.f b - t.f d) / ((t.f b - t.f c) * (t.f a - t.f d))

/-- Two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  (t₁ t₂ : ProjectiveTransformation)
  (h₁ : PreservesCrossRatio t₁)
  (h₂ : PreservesCrossRatio t₂)
  (a b c : ℝ)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (eq_a : t₁.f a = t₂.f a)
  (eq_b : t₁.f b = t₂.f b)
  (eq_c : t₁.f c = t₂.f c) :
  ∀ x : ℝ, t₁.f x = t₂.f x :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1984_198482


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1984_198473

theorem arithmetic_mean_of_fractions : 
  (3 / 8 + 5 / 12) / 2 = 19 / 48 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1984_198473


namespace NUMINAMATH_CALUDE_frequency_converges_to_half_l1984_198464

/-- A coin toss experiment -/
structure CoinToss where
  /-- The probability of getting heads in a single toss -/
  probHeads : ℝ
  /-- The coin is fair -/
  isFair : probHeads = 0.5

/-- The frequency of heads after n tosses -/
def frequency (c : CoinToss) (n : ℕ) : ℝ :=
  sorry

/-- The theorem stating that the frequency of heads converges to 0.5 as n approaches infinity -/
theorem frequency_converges_to_half (c : CoinToss) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency c n - 0.5| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_half_l1984_198464


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1984_198409

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The product of two functions -/
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ :=
  fun x ↦ f x * g x

theorem necessary_not_sufficient :
  (∀ f g : ℝ → ℝ, (IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g) →
    IsEven (FunctionProduct f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionProduct f g) ∧
    ¬(IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1984_198409


namespace NUMINAMATH_CALUDE_replacement_solution_concentration_l1984_198444

/-- Given an 80% chemical solution, if 50% of it is replaced with a solution
    of unknown concentration P%, resulting in a 50% chemical solution,
    then P% must be 20%. -/
theorem replacement_solution_concentration
  (original_concentration : ℝ)
  (replaced_fraction : ℝ)
  (final_concentration : ℝ)
  (replacement_concentration : ℝ)
  (h1 : original_concentration = 0.8)
  (h2 : replaced_fraction = 0.5)
  (h3 : final_concentration = 0.5)
  (h4 : final_concentration = (1 - replaced_fraction) * original_concentration
                            + replaced_fraction * replacement_concentration) :
  replacement_concentration = 0.2 := by
sorry

end NUMINAMATH_CALUDE_replacement_solution_concentration_l1984_198444


namespace NUMINAMATH_CALUDE_expansion_equality_constant_term_proof_l1984_198445

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 -/
def constantTerm : ℕ := 160

/-- The original expression (1/x^2 + 4x^2 + 4)^3 can be rewritten as (2x + 1/x)^6 -/
theorem expansion_equality (x : ℝ) (hx : x ≠ 0) :
  (1 / x^2 + 4 * x^2 + 4)^3 = (2 * x + 1 / x)^6 := by sorry

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 is equal to constantTerm -/
theorem constant_term_proof :
  constantTerm = 160 := by sorry

end NUMINAMATH_CALUDE_expansion_equality_constant_term_proof_l1984_198445


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1984_198469

theorem circle_center_coordinate_sum (x y : ℝ) : 
  (x^2 + y^2 = 8*x - 6*y - 20) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 20) ∧ h + k = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1984_198469


namespace NUMINAMATH_CALUDE_binary_sum_equals_852_l1984_198432

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, true, false, true, false, true, false, true]

theorem binary_sum_equals_852 : 
  binary_to_decimal num1 + binary_to_decimal num2 = 852 := by
sorry

end NUMINAMATH_CALUDE_binary_sum_equals_852_l1984_198432


namespace NUMINAMATH_CALUDE_omega_sequence_monotone_increasing_l1984_198475

/-- Definition of an Ω sequence -/
def is_omega_sequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 ≤ a (n + 1)) ∧
  (∃ M : ℝ, ∀ n : ℕ+, a n ≤ M)

/-- Theorem: For any Ω sequence of positive integers, each term is less than or equal to the next term -/
theorem omega_sequence_monotone_increasing
  (d : ℕ+ → ℕ+)
  (h_omega : is_omega_sequence (λ n => (d n : ℝ))) :
  ∀ n : ℕ+, d n ≤ d (n + 1) := by
sorry

end NUMINAMATH_CALUDE_omega_sequence_monotone_increasing_l1984_198475


namespace NUMINAMATH_CALUDE_points_per_recycled_bag_l1984_198466

theorem points_per_recycled_bag 
  (total_bags : ℕ) 
  (unrecycled_bags : ℕ) 
  (total_points : ℕ) 
  (h1 : total_bags = 11) 
  (h2 : unrecycled_bags = 2) 
  (h3 : total_points = 45) :
  total_points / (total_bags - unrecycled_bags) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_recycled_bag_l1984_198466


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l1984_198452

theorem sqrt_sum_equation (x : ℝ) (h : x ≥ 1/2) :
  (∃ y ∈ Set.Icc (1/2 : ℝ) 1, x = y ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = Real.sqrt 2) ∧
  (¬ ∃ y ≥ 1/2, Real.sqrt (y + Real.sqrt (2*y - 1)) + Real.sqrt (y - Real.sqrt (2*y - 1)) = 1) ∧
  (x = 3/2 ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l1984_198452


namespace NUMINAMATH_CALUDE_parabola_line_tangency_l1984_198429

/-- 
Given a parabola y = ax^2 + 6 and a line y = 2x + k, where k is a constant,
this theorem states the condition for tangency between the parabola and the line.
-/
theorem parabola_line_tangency (a k : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + 6 ∧ y = 2 * x + k) →
  (k ≠ 6) →
  (a = -1 / (k - 6)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_tangency_l1984_198429


namespace NUMINAMATH_CALUDE_felix_lift_problem_l1984_198440

/-- Felix's weight lifting problem -/
theorem felix_lift_problem (felix_weight : ℝ) (felix_brother_weight : ℝ) (felix_brother_lift : ℝ) :
  (felix_brother_weight = 2 * felix_weight) →
  (felix_brother_lift = 3 * felix_brother_weight) →
  (felix_brother_lift = 600) →
  (1.5 * felix_weight = 150) :=
by
  sorry

#check felix_lift_problem

end NUMINAMATH_CALUDE_felix_lift_problem_l1984_198440


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1984_198476

/-- The area of a square with a diagonal of 3.8 meters is 7.22 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  let s := d / Real.sqrt 2
  s ^ 2 = 7.22 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1984_198476


namespace NUMINAMATH_CALUDE_keith_digimon_packs_l1984_198420

/-- The cost of one pack of Digimon cards in dollars -/
def digimon_pack_cost : ℚ := 445/100

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℚ := 606/100

/-- The total amount Keith spent on cards in dollars -/
def total_spent : ℚ := 2386/100

/-- The number of Digimon card packs Keith bought -/
def digimon_packs : ℕ := 4

theorem keith_digimon_packs :
  digimon_packs * digimon_pack_cost + baseball_deck_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_keith_digimon_packs_l1984_198420


namespace NUMINAMATH_CALUDE_series_sum_l1984_198427

/-- The general term of the series -/
def a (n : ℕ) : ℚ := (3 * n^2 + 2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

/-- The series sum -/
noncomputable def S : ℚ := ∑' n, a n

/-- Theorem: The sum of the series is 7/6 -/
theorem series_sum : S = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1984_198427


namespace NUMINAMATH_CALUDE_solve_x_and_y_l1984_198414

-- Define the universal set I
def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}

-- Define set A
def A : Set ℝ := {5}

-- Define the complement of A with respect to I
def complement_A (x : ℝ) (y : ℝ) : Set ℝ := {2, y}

-- Theorem statement
theorem solve_x_and_y (x : ℝ) (y : ℝ) :
  (5 ∈ I x) ∧ (complement_A x y = (I x) \ A) →
  ((x = -4 ∨ x = 2) ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_x_and_y_l1984_198414


namespace NUMINAMATH_CALUDE_base8_to_base7_conversion_l1984_198454

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The given number in base 8 -/
def givenNumber : ℕ := 653

/-- The expected result in base 7 -/
def expectedResult : ℕ := 1150

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 givenNumber) = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base7_conversion_l1984_198454


namespace NUMINAMATH_CALUDE_sin_arithmetic_is_geometric_ratio_l1984_198497

def is_arithmetic_sequence (α : ℕ → ℝ) (β : ℝ) :=
  ∀ n, α (n + 1) = α n + β

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem sin_arithmetic_is_geometric_ratio (α : ℕ → ℝ) (β : ℝ) :
  is_arithmetic_sequence α β →
  (∃ q, is_geometric_sequence (fun n ↦ Real.sin (α n)) q) →
  ∃ q, (q = 1 ∨ q = -1) ∧ is_geometric_sequence (fun n ↦ Real.sin (α n)) q :=
by sorry

end NUMINAMATH_CALUDE_sin_arithmetic_is_geometric_ratio_l1984_198497


namespace NUMINAMATH_CALUDE_special_numbers_l1984_198447

def is_special (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∀ d : ℕ+, d ∣ n → (d - k : ℤ) ∣ n

theorem special_numbers (n : ℕ+) :
  is_special n ↔ n = 3 ∨ n = 4 ∨ n = 6 ∨ Nat.Prime n.val :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l1984_198447


namespace NUMINAMATH_CALUDE_angle_sum_quarter_range_l1984_198434

-- Define acute and obtuse angles
def acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def obtuse_angle (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Theorem statement
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : acute_angle α) (h_obtuse : obtuse_angle β) :
  Real.pi / 8 < (α + β) / 4 ∧ (α + β) / 4 < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end NUMINAMATH_CALUDE_angle_sum_quarter_range_l1984_198434


namespace NUMINAMATH_CALUDE_unique_digit_product_l1984_198481

theorem unique_digit_product : ∃! (x y z : ℕ), 
  (x < 10 ∧ y < 10 ∧ z < 10) ∧ 
  (10 ≤ 10 * x + y) ∧ (10 * x + y ≤ 99) ∧
  (1 ≤ z) ∧
  (x * (10 * x + y) = 111 * z) := by
sorry

end NUMINAMATH_CALUDE_unique_digit_product_l1984_198481


namespace NUMINAMATH_CALUDE_zero_to_zero_undefined_l1984_198419

theorem zero_to_zero_undefined : ¬ ∃ (x : ℝ), 0^(0 : ℝ) = x := by
  sorry

end NUMINAMATH_CALUDE_zero_to_zero_undefined_l1984_198419


namespace NUMINAMATH_CALUDE_max_value_fraction_l1984_198411

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ a b : ℝ, -6 ≤ a ∧ a ≤ -3 → 1 ≤ b ∧ b ≤ 5 → (a + b + 1) / (a + 1) ≤ (x + y + 1) / (x + 1)) →
  (x + y + 1) / (x + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1984_198411


namespace NUMINAMATH_CALUDE_parabola_directrix_l1984_198449

/-- Definition of a parabola with equation y^2 = 6x -/
def parabola (x y : ℝ) : Prop := y^2 = 6*x

/-- Definition of the directrix of a parabola -/
def directrix (x : ℝ) : Prop := x = -3/2

/-- Theorem: The directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → directrix x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1984_198449


namespace NUMINAMATH_CALUDE_michael_has_270_eggs_l1984_198487

/-- Calculates the number of eggs Michael has after buying and giving away crates. -/
def michaels_eggs (initial_crates : ℕ) (given_crates : ℕ) (bought_crates : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  (initial_crates - given_crates + bought_crates) * eggs_per_crate

/-- Proves that Michael has 270 eggs after his transactions. -/
theorem michael_has_270_eggs :
  michaels_eggs 6 2 5 30 = 270 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_270_eggs_l1984_198487


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l1984_198496

theorem tree_planting_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) 
  (total_fallen : ℕ) (final_trees : ℕ) :
  initial_mahogany = 50 →
  initial_narra = 30 →
  total_fallen = 5 →
  final_trees = 88 →
  ∃ (fallen_narra fallen_mahogany planted : ℕ),
    fallen_narra + fallen_mahogany = total_fallen ∧
    fallen_mahogany = fallen_narra + 1 ∧
    planted = final_trees - (initial_mahogany + initial_narra - total_fallen) ∧
    planted * 2 = fallen_narra * 13 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l1984_198496


namespace NUMINAMATH_CALUDE_infected_and_positive_probability_l1984_198480

/-- The infection rate of the novel coronavirus -/
def infection_rate : ℝ := 0.005

/-- The probability of testing positive given infection -/
def positive_given_infection : ℝ := 0.99

/-- The probability that a citizen is infected and tests positive -/
def infected_and_positive : ℝ := infection_rate * positive_given_infection

theorem infected_and_positive_probability :
  infected_and_positive = 0.00495 := by sorry

end NUMINAMATH_CALUDE_infected_and_positive_probability_l1984_198480


namespace NUMINAMATH_CALUDE_indeterminate_product_sum_l1984_198448

theorem indeterminate_product_sum (A B : ℝ) 
  (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) : 
  ∃ (x y z : ℝ), x < 1 ∧ y = 1 ∧ z > 1 ∧ 
  (A * B + 0.1 = x ∨ A * B + 0.1 = y ∨ A * B + 0.1 = z) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_product_sum_l1984_198448


namespace NUMINAMATH_CALUDE_sum_three_numbers_l1984_198438

theorem sum_three_numbers (a b c N : ℚ) : 
  a + b + c = 84 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  c / 7 = N → 
  N = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l1984_198438


namespace NUMINAMATH_CALUDE_basketball_three_pointers_l1984_198416

/-- Represents the number of 3-point shots in a basketball game -/
def three_point_shots (total_points total_shots : ℕ) : ℕ :=
  sorry

/-- The number of 3-point shots is 4 when the total points is 26 and total shots is 11 -/
theorem basketball_three_pointers :
  three_point_shots 26 11 = 4 :=
sorry

end NUMINAMATH_CALUDE_basketball_three_pointers_l1984_198416


namespace NUMINAMATH_CALUDE_f_max_at_two_l1984_198467

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- Theorem stating that f has a maximum value of 24 at x = 2 -/
theorem f_max_at_two :
  ∃ (x_max : ℝ), x_max = 2 ∧ f x_max = 24 ∧ ∀ (x : ℝ), f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_max_at_two_l1984_198467


namespace NUMINAMATH_CALUDE_gcd_repeated_digits_l1984_198446

def is_repeated_digit (n : ℕ) : Prop :=
  ∃ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_digits :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), is_repeated_digit n → d ∣ n) ∧
  (∀ (k : ℕ), k > 0 → (∀ (n : ℕ), is_repeated_digit n → k ∣ n) → k ∣ d) :=
sorry

end NUMINAMATH_CALUDE_gcd_repeated_digits_l1984_198446


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1984_198430

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
  k > 10 →
  (isPalindrome k 2 ∧ isPalindrome k 4) →
  k ≥ 17 ∧
  isPalindrome 17 2 ∧
  isPalindrome 17 4 := by
    sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1984_198430


namespace NUMINAMATH_CALUDE_x_value_l1984_198443

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.1 * 500 - 5) ∧ (x = 180) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1984_198443


namespace NUMINAMATH_CALUDE_multiply_and_subtract_problem_solution_l1984_198431

theorem multiply_and_subtract (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem problem_solution : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end NUMINAMATH_CALUDE_multiply_and_subtract_problem_solution_l1984_198431


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1984_198424

theorem max_value_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) ≤ 1/18 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) = 1/18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1984_198424


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1984_198462

theorem right_triangle_legs (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  ((a = 16 ∧ b = 63) ∨ (a = 63 ∧ b = 16)) →  -- Possible leg lengths
  ∃ (x y : ℕ), x^2 + y^2 = 65^2 ∧ (x = 16 ∧ y = 63) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1984_198462


namespace NUMINAMATH_CALUDE_total_cost_of_shirts_proof_total_cost_of_shirts_l1984_198439

/-- The total cost of two shirts, where the first shirt costs $6 more than the second,
    and the first shirt costs $15, is $24. -/
theorem total_cost_of_shirts : ℕ → Prop :=
  fun n => n = 24 ∧ ∃ (cost1 cost2 : ℕ),
    cost1 = 15 ∧
    cost1 = cost2 + 6 ∧
    n = cost1 + cost2

/-- Proof of the theorem -/
theorem proof_total_cost_of_shirts : total_cost_of_shirts 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_shirts_proof_total_cost_of_shirts_l1984_198439


namespace NUMINAMATH_CALUDE_lesser_number_l1984_198471

theorem lesser_number (x y : ℝ) (sum : x + y = 60) (diff : x - y = 8) : 
  min x y = 26 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_l1984_198471


namespace NUMINAMATH_CALUDE_total_marbles_l1984_198418

/-- Given 5 bags with 5 marbles each and 1 bag with 8 marbles,
    the total number of marbles in all 6 bags is 33. -/
theorem total_marbles (bags_of_five : Nat) (marbles_per_bag : Nat) (extra_bag : Nat) :
  bags_of_five = 5 →
  marbles_per_bag = 5 →
  extra_bag = 8 →
  bags_of_five * marbles_per_bag + extra_bag = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l1984_198418


namespace NUMINAMATH_CALUDE_solve_a_l1984_198494

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a : ∃ (a : ℝ), star a 5 = 9 ∧ a = 17 := by sorry

end NUMINAMATH_CALUDE_solve_a_l1984_198494


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_is_117_143_l1984_198491

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zincCopperRatio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def givenMixture : Mixture :=
  { totalWeight := 78
  , zincWeight := 35.1 }

/-- Theorem stating that the ratio of zinc to copper in the given mixture is 117:143 -/
theorem zinc_copper_ratio_is_117_143 :
  zincCopperRatio givenMixture = { numerator := 117, denominator := 143 } :=
  sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_is_117_143_l1984_198491


namespace NUMINAMATH_CALUDE_sin_cos_alpha_l1984_198483

theorem sin_cos_alpha (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_alpha_l1984_198483


namespace NUMINAMATH_CALUDE_compute_expression_l1984_198406

theorem compute_expression : 45 * 28 + 45 * 72 - 10 * 45 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1984_198406


namespace NUMINAMATH_CALUDE_market_spending_l1984_198425

theorem market_spending (mildred_spent candice_spent joseph_spent_percentage joseph_spent mom_total : ℝ) :
  mildred_spent = 25 →
  candice_spent = 35 →
  joseph_spent_percentage = 0.8 →
  joseph_spent = 45 →
  mom_total = 150 →
  mom_total - (mildred_spent + candice_spent + joseph_spent) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_market_spending_l1984_198425


namespace NUMINAMATH_CALUDE_inverse_f_at_neg_1_l1984_198435

-- Define f as a function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition that f(2) = -1
axiom f_at_2 : f 2 = -1

-- State the theorem to be proved
theorem inverse_f_at_neg_1 : Function.invFun f (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_neg_1_l1984_198435


namespace NUMINAMATH_CALUDE_lilith_water_bottles_l1984_198485

/-- The number of water bottles Lilith originally had -/
def num_bottles : ℕ := 60

/-- The original selling price per bottle in dollars -/
def original_price : ℚ := 2

/-- The reduced selling price per bottle in dollars -/
def reduced_price : ℚ := 185/100

theorem lilith_water_bottles :
  (original_price * num_bottles : ℚ) - (reduced_price * num_bottles) = 9 :=
sorry

end NUMINAMATH_CALUDE_lilith_water_bottles_l1984_198485


namespace NUMINAMATH_CALUDE_optimal_room_allocation_l1984_198410

theorem optimal_room_allocation (total_people : Nat) (large_room_capacity : Nat) 
  (h1 : total_people = 26) (h2 : large_room_capacity = 3) : 
  ∃ (small_room_capacity : Nat), 
    small_room_capacity = total_people - large_room_capacity ∧ 
    small_room_capacity > 0 ∧
    (∀ (x : Nat), x > 0 ∧ x < small_room_capacity → 
      (total_people - large_room_capacity) % x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_optimal_room_allocation_l1984_198410


namespace NUMINAMATH_CALUDE_not_prime_m_plus_n_minus_one_l1984_198417

theorem not_prime_m_plus_n_minus_one (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2)
  (h3 : (m + n - 1) ∣ (m^2 + n^2 - 1)) : ¬ Nat.Prime (m + n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_m_plus_n_minus_one_l1984_198417


namespace NUMINAMATH_CALUDE_slower_plane_speed_l1984_198486

/-- Given two planes flying in opposite directions for 3 hours, where one plane's speed is twice 
    the other's, and they end up 2700 miles apart, prove that the slower plane's speed is 300 
    miles per hour. -/
theorem slower_plane_speed (slower_speed faster_speed : ℝ) 
    (h1 : faster_speed = 2 * slower_speed)
    (h2 : 3 * slower_speed + 3 * faster_speed = 2700) : 
  slower_speed = 300 := by
  sorry

end NUMINAMATH_CALUDE_slower_plane_speed_l1984_198486


namespace NUMINAMATH_CALUDE_base7_addition_subtraction_l1984_198423

-- Define a function to convert base 7 to decimal
def base7ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Define a function to convert decimal to base 7
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

-- Define the numbers in base 7
def n1 : List Nat := [0, 0, 0, 1]  -- 1000₇
def n2 : List Nat := [6, 6, 6]     -- 666₇
def n3 : List Nat := [4, 3, 2, 1]  -- 1234₇

-- State the theorem
theorem base7_addition_subtraction :
  decimalToBase7 (base7ToDecimal n1 + base7ToDecimal n2 - base7ToDecimal n3) = [4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_subtraction_l1984_198423


namespace NUMINAMATH_CALUDE_complex_number_real_l1984_198488

theorem complex_number_real (a : ℝ) : 
  (∃ (r : ℝ), Complex.mk r 0 = Complex.mk 0 2 - (Complex.I * a) / (1 - Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_l1984_198488


namespace NUMINAMATH_CALUDE_two_number_difference_l1984_198498

theorem two_number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 23320) : b - a = 19080 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l1984_198498


namespace NUMINAMATH_CALUDE_q_investment_time_l1984_198433

-- Define the investment ratio
def investment_ratio : ℚ := 7 / 5

-- Define the profit ratio
def profit_ratio : ℚ := 7 / 10

-- Define P's investment time in months
def p_time : ℚ := 2

-- Define Q's investment time as a variable
variable (q_time : ℚ)

-- Theorem statement
theorem q_investment_time : 
  (investment_ratio * p_time) / (q_time / investment_ratio) = profit_ratio → q_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l1984_198433


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1984_198492

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1984_198492


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l1984_198437

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

/-- Theorem: The equation of the circle with diameter endpoints A(1,4) and B(3,-2) -/
theorem circle_equation_from_diameter_endpoints :
  let A : Point := ⟨1, 4⟩
  let B : Point := ⟨3, -2⟩
  let center : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  let radius : ℝ := Real.sqrt ((B.x - center.x)^2 + (B.y - center.y)^2)
  CircleEquation center radius = fun x y ↦ (x - 2)^2 + (y - 1)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l1984_198437


namespace NUMINAMATH_CALUDE_f_is_power_and_increasing_l1984_198436

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

-- Define an increasing function on (0, +∞)
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- Define the function f(x) = x^(1/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- Theorem statement
theorem f_is_power_and_increasing :
  isPowerFunction f ∧ isIncreasing f :=
sorry

end NUMINAMATH_CALUDE_f_is_power_and_increasing_l1984_198436


namespace NUMINAMATH_CALUDE_circuit_board_count_l1984_198458

theorem circuit_board_count (T P : ℕ) : 
  (64 + P = T) →  -- Total boards = Failed + Passed
  (64 + P / 8 = 456) →  -- Total faulty boards
  T = 3200 := by
  sorry

end NUMINAMATH_CALUDE_circuit_board_count_l1984_198458


namespace NUMINAMATH_CALUDE_fraction_inequality_l1984_198412

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a / b > (a + c) / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1984_198412


namespace NUMINAMATH_CALUDE_hobbit_burrow_assignment_l1984_198459

-- Define the burrows
inductive Burrow
| A | B | C | D | E | F

-- Define the hobbits
inductive Hobbit
| Frodo | Sam | Merry | Pippin

-- Define the concept of distance between burrows
def closer_to (b1 b2 b3 : Burrow) : Prop := sorry

-- Define the concept of closeness to river and forest
def closer_to_river (b1 b2 : Burrow) : Prop := sorry
def farther_from_forest (b1 b2 : Burrow) : Prop := sorry

-- Define the assignment of hobbits to burrows
def assignment : Hobbit → Burrow
| Hobbit.Frodo => Burrow.E
| Hobbit.Sam => Burrow.A
| Hobbit.Merry => Burrow.C
| Hobbit.Pippin => Burrow.F

-- Theorem statement
theorem hobbit_burrow_assignment :
  (∀ h1 h2 : Hobbit, h1 ≠ h2 → assignment h1 ≠ assignment h2) ∧
  (∀ b : Burrow, b ≠ Burrow.B ∧ b ≠ Burrow.D → ∃ h : Hobbit, assignment h = b) ∧
  (closer_to Burrow.B Burrow.A Burrow.E) ∧
  (closer_to Burrow.D Burrow.A Burrow.E) ∧
  (closer_to_river Burrow.E Burrow.C) ∧
  (farther_from_forest Burrow.E Burrow.F) :=
by sorry

end NUMINAMATH_CALUDE_hobbit_burrow_assignment_l1984_198459
