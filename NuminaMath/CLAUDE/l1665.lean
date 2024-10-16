import Mathlib

namespace NUMINAMATH_CALUDE_max_xy_min_inverse_sum_l1665_166565

-- Define the conditions
variable (x y : ℝ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x + 4*y = 4)

-- Theorem for the maximum value of xy
theorem max_xy : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → xy ≤ 1 := by
  sorry

-- Theorem for the minimum value of 1/x + 2/y
theorem min_inverse_sum : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → 1/x + 2/y ≥ (9 + 4*Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_min_inverse_sum_l1665_166565


namespace NUMINAMATH_CALUDE_final_women_count_l1665_166566

/-- Represents the number of people in each category --/
structure Population :=
  (men : ℕ)
  (women : ℕ)
  (children : ℕ)
  (elderly : ℕ)

/-- Theorem stating the final number of women in the room --/
theorem final_women_count (initial : Population) 
  (h1 : initial.men + initial.women + initial.children + initial.elderly > 0)
  (h2 : initial.men = 4 * initial.elderly / 2)
  (h3 : initial.women = 5 * initial.elderly / 2)
  (h4 : initial.children = 3 * initial.elderly / 2)
  (h5 : initial.men + 2 = 14)
  (h6 : initial.children - 5 = 7)
  (h7 : initial.elderly - 3 = 6) :
  2 * (initial.women - 3) = 24 := by
  sorry

#check final_women_count

end NUMINAMATH_CALUDE_final_women_count_l1665_166566


namespace NUMINAMATH_CALUDE_norris_savings_l1665_166526

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris saved in December -/
def december_savings : ℕ := 35

/-- The amount of money Norris saved in January -/
def january_savings : ℕ := 40

/-- The total amount of money Norris saved from September to January -/
def total_savings : ℕ := september_savings + october_savings + november_savings + december_savings + january_savings

theorem norris_savings : total_savings = 160 := by
  sorry

end NUMINAMATH_CALUDE_norris_savings_l1665_166526


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1665_166568

theorem product_of_square_roots (p : ℝ) : 
  Real.sqrt (8 * p^2) * Real.sqrt (12 * p^3) * Real.sqrt (18 * p^5) = 24 * p^5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1665_166568


namespace NUMINAMATH_CALUDE_cubic_equation_with_geometric_progression_roots_l1665_166582

theorem cubic_equation_with_geometric_progression_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 11*x^2 + a*x - 8 = 0 ∧
    y^3 - 11*y^2 + a*y - 8 = 0 ∧
    z^3 - 11*z^2 + a*z - 8 = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 22 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_geometric_progression_roots_l1665_166582


namespace NUMINAMATH_CALUDE_signup_ways_eq_64_l1665_166594

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of interest groups --/
def num_groups : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def num_ways : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of ways to sign up is 64 --/
theorem signup_ways_eq_64 : num_ways = 64 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_eq_64_l1665_166594


namespace NUMINAMATH_CALUDE_expand_staircase_4_to_7_l1665_166599

/-- Calculates the number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 2 * n * n + 2 * n

/-- The number of additional toothpicks needed to expand from m steps to n steps -/
def additional_toothpicks (m n : ℕ) : ℕ :=
  toothpicks n - toothpicks m

theorem expand_staircase_4_to_7 :
  additional_toothpicks 4 7 = 48 := by
  sorry

#eval additional_toothpicks 4 7

end NUMINAMATH_CALUDE_expand_staircase_4_to_7_l1665_166599


namespace NUMINAMATH_CALUDE_distance_between_sets_l1665_166556

/-- The distance between two sets A and B, where
    A = {y | y = 2x - 1, x ∈ ℝ} and
    B = {y | y = x² + 1, x ∈ ℝ},
    is defined as the minimum value of |a - b|, where a ∈ A and b ∈ B. -/
theorem distance_between_sets :
  ∃ (x y : ℝ), |((2 * x) - 1) - (y^2 + 1)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_sets_l1665_166556


namespace NUMINAMATH_CALUDE_min_value_expression_l1665_166520

theorem min_value_expression (a b c : ℤ) (h1 : c > 0) (h2 : a = b + c) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 ∧
  ∃ (a b : ℤ), ∃ (c : ℤ), c > 0 ∧ a = b + c ∧
    (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1665_166520


namespace NUMINAMATH_CALUDE_adjacent_empty_seats_l1665_166588

theorem adjacent_empty_seats (n : ℕ) (k : ℕ) : n = 6 → k = 3 →
  (number_of_arrangements : ℕ) →
  (number_of_arrangements = 
    -- Case 1: Two adjacent empty seats at the ends
    (2 * (Nat.choose 3 1) * (Nat.choose 3 2)) +
    -- Case 2: Two adjacent empty seats not at the ends
    (3 * (Nat.choose 3 2) * (Nat.choose 2 1))) →
  number_of_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_adjacent_empty_seats_l1665_166588


namespace NUMINAMATH_CALUDE_chandler_can_buy_bike_l1665_166540

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 800

/-- The total amount of gift money Chandler received in dollars -/
def gift_money : ℕ := 100 + 50 + 20 + 30

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks Chandler needs to save to buy the mountain bike -/
def weeks_to_save : ℕ := 30

/-- Theorem stating that Chandler can buy the bike after saving for the calculated number of weeks -/
theorem chandler_can_buy_bike :
  gift_money + weekly_earnings * weeks_to_save = bike_cost :=
by sorry

end NUMINAMATH_CALUDE_chandler_can_buy_bike_l1665_166540


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l1665_166512

theorem least_prime_factor_of_5_5_minus_5_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l1665_166512


namespace NUMINAMATH_CALUDE_change_received_l1665_166533

/-- The change received when buying a football and baseball with given costs and payment amount -/
theorem change_received (football_cost baseball_cost payment : ℚ) : 
  football_cost = 9.14 →
  baseball_cost = 6.81 →
  payment = 20 →
  payment - (football_cost + baseball_cost) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l1665_166533


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_l1665_166501

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let midpoint_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)
  (midpoint_distance (a/2) 0) + (midpoint_distance a (b/2)) +
  (midpoint_distance (a/2) b) + (midpoint_distance 0 (b/2)) =
  3.5 + Real.sqrt 13 + Real.sqrt 18.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_l1665_166501


namespace NUMINAMATH_CALUDE_initial_velocity_is_three_l1665_166554

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem initial_velocity_is_three :
  velocity 0 = 3 :=
sorry

end NUMINAMATH_CALUDE_initial_velocity_is_three_l1665_166554


namespace NUMINAMATH_CALUDE_circumcircle_diameter_l1665_166575

/-- Given a triangle ABC with side length a = 2 and angle A = 60°,
    prove that the diameter of its circumcircle is 4√3/3 -/
theorem circumcircle_diameter (a : ℝ) (A : ℝ) (h1 : a = 2) (h2 : A = π/3) :
  (2 * a) / Real.sin A = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_l1665_166575


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1665_166567

theorem min_value_reciprocal_sum (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h_sum : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ m₀ n₀ : ℝ, 0 < m₀ ∧ 0 < n₀ ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1665_166567


namespace NUMINAMATH_CALUDE_females_who_chose_malt_l1665_166597

/-- Represents the number of cheerleaders who chose each drink -/
structure CheerleaderChoices where
  coke : ℕ
  malt : ℕ

/-- Represents the gender distribution of cheerleaders -/
structure CheerleaderGenders where
  males : ℕ
  females : ℕ

theorem females_who_chose_malt 
  (choices : CheerleaderChoices)
  (genders : CheerleaderGenders)
  (h1 : genders.males = 10)
  (h2 : genders.females = 16)
  (h3 : choices.malt = 2 * choices.coke)
  (h4 : choices.coke + choices.malt = genders.males + genders.females)
  (h5 : choices.malt ≥ genders.males)
  (h6 : genders.males = 6) :
  choices.malt - genders.males = 10 := by
sorry

end NUMINAMATH_CALUDE_females_who_chose_malt_l1665_166597


namespace NUMINAMATH_CALUDE_correct_transformation_l1665_166592

def original_expression : List Int := [-17, 3, -5, -8]

def transformed_expression : List Int := [-17, 3, 5, -8]

theorem correct_transformation :
  (original_expression.map (fun x => if x < 0 then -x else x)).foldl (· - ·) 0 =
  transformed_expression.foldl (· + ·) 0 :=
sorry

end NUMINAMATH_CALUDE_correct_transformation_l1665_166592


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1665_166503

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1665_166503


namespace NUMINAMATH_CALUDE_sin_increases_with_angle_l1665_166549

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (law_of_sines : a / Real.sin A = b / Real.sin B)

-- Theorem statement
theorem sin_increases_with_angle (abc : Triangle) (h : abc.A > abc.B) :
  Real.sin abc.A > Real.sin abc.B :=
sorry

end NUMINAMATH_CALUDE_sin_increases_with_angle_l1665_166549


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1665_166506

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (1 + i) * z = 1 + 3 * i →
  z = 2 + i := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1665_166506


namespace NUMINAMATH_CALUDE_union_complement_equality_complement_intersection_equality_l1665_166580

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {1, 3, 5, 7}

-- Theorem for part (1)
theorem union_complement_equality :
  A ∪ (U \ B) = {2, 4, 5, 6} := by sorry

-- Theorem for part (2)
theorem complement_intersection_equality :
  U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_complement_intersection_equality_l1665_166580


namespace NUMINAMATH_CALUDE_relationship_a_b_l1665_166590

theorem relationship_a_b (a b : ℝ) (ha : a^(1/5) > 1) (hb : 1 > b^(1/5)) : a > 1 ∧ 1 > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_a_b_l1665_166590


namespace NUMINAMATH_CALUDE_brick_width_l1665_166555

/-- The surface area of a rectangular prism. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a brick with given dimensions and surface area. -/
theorem brick_width (l h : ℝ) (sa : ℝ) (hl : l = 10) (hh : h = 3) (hsa : sa = 164) :
  ∃ w : ℝ, w = 4 ∧ surface_area l w h = sa :=
sorry

end NUMINAMATH_CALUDE_brick_width_l1665_166555


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l1665_166578

theorem wrapping_paper_usage
  (total_paper : ℚ)
  (small_presents : ℕ)
  (large_presents : ℕ)
  (h1 : total_paper = 5 / 12)
  (h2 : small_presents = 4)
  (h3 : large_presents = 2)
  (h4 : small_presents + large_presents = 6) :
  ∃ (small_paper large_paper : ℚ),
    small_paper * small_presents + large_paper * large_presents = total_paper ∧
    large_paper = 2 * small_paper ∧
    small_paper = 5 / 96 ∧
    large_paper = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l1665_166578


namespace NUMINAMATH_CALUDE_chipped_marbles_are_36_l1665_166508

def marble_bags : List Nat := [16, 18, 22, 24, 26, 30, 36]

structure MarbleDistribution where
  jane_bags : List Nat
  george_bags : List Nat
  chipped_bag : Nat

def is_valid_distribution (d : MarbleDistribution) : Prop :=
  d.jane_bags.length = 4 ∧
  d.george_bags.length = 2 ∧
  d.chipped_bag ∈ marble_bags ∧
  d.jane_bags.sum = 3 * d.george_bags.sum ∧
  (∀ b ∈ d.jane_bags ++ d.george_bags, b ≠ d.chipped_bag) ∧
  (∀ b ∈ marble_bags, b ∉ d.jane_bags → b ∉ d.george_bags → b = d.chipped_bag)

theorem chipped_marbles_are_36 :
  ∀ d : MarbleDistribution, is_valid_distribution d → d.chipped_bag = 36 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_are_36_l1665_166508


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1665_166557

theorem algebraic_expression_value (m n : ℝ) (h : 2*m - 3*n = -2) :
  4*m - 6*n + 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1665_166557


namespace NUMINAMATH_CALUDE_project_update_lcm_l1665_166536

theorem project_update_lcm : Nat.lcm 5 (Nat.lcm 9 (Nat.lcm 10 13)) = 1170 := by
  sorry

end NUMINAMATH_CALUDE_project_update_lcm_l1665_166536


namespace NUMINAMATH_CALUDE_marketValueTheorem_l1665_166546

/-- Calculates the market value of a machine after two years, given initial value,
    depreciation rates, and inflation rate. -/
def marketValueAfterTwoYears (initialValue : ℝ) (depreciation1 : ℝ) (depreciation2 : ℝ) (inflation : ℝ) : ℝ :=
  let value1 := initialValue * (1 - depreciation1) * (1 + inflation)
  let value2 := value1 * (1 - depreciation2) * (1 + inflation)
  value2

/-- Theorem stating that the market value of a machine with given parameters
    after two years is approximately 4939.20. -/
theorem marketValueTheorem :
  ∃ ε > 0, ε < 0.01 ∧ 
  |marketValueAfterTwoYears 8000 0.3 0.2 0.05 - 4939.20| < ε :=
sorry

end NUMINAMATH_CALUDE_marketValueTheorem_l1665_166546


namespace NUMINAMATH_CALUDE_circular_road_width_l1665_166522

theorem circular_road_width (r R : ℝ) (h1 : r = R / 3) (h2 : 2 * π * r + 2 * π * R = 88) :
  R - r = 22 / π := by
  sorry

end NUMINAMATH_CALUDE_circular_road_width_l1665_166522


namespace NUMINAMATH_CALUDE_paris_total_study_hours_l1665_166527

/-- The number of hours Paris studies during the semester -/
def paris_study_hours : ℕ :=
  let weeks_in_semester : ℕ := 15
  let weekday_study_hours : ℕ := 3
  let saturday_study_hours : ℕ := 4
  let sunday_study_hours : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weekly_study_hours : ℕ := weekday_study_hours * weekdays_per_week + saturday_study_hours + sunday_study_hours
  weekly_study_hours * weeks_in_semester

theorem paris_total_study_hours : paris_study_hours = 360 := by
  sorry

end NUMINAMATH_CALUDE_paris_total_study_hours_l1665_166527


namespace NUMINAMATH_CALUDE_computer_sticker_price_l1665_166581

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
    (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
    sticker_price = 750 := by
  sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l1665_166581


namespace NUMINAMATH_CALUDE_intersection_equality_l1665_166531

theorem intersection_equality (a : ℝ) : 
  (∀ x, (1 < x ∧ x < 7) ∧ (a + 1 < x ∧ x < 2*a + 5) ↔ (3 < x ∧ x < 7)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_l1665_166531


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l1665_166504

/-- The number of apples Sarah gives to teachers -/
def apples_to_teachers (initial : ℕ) (locker : ℕ) (classmates : ℕ) (friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - locker - classmates - friends - eaten - left

theorem sarah_apples_to_teachers :
  apples_to_teachers 50 10 8 5 1 4 = 22 := by
  sorry

#eval apples_to_teachers 50 10 8 5 1 4

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l1665_166504


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1665_166577

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 2 * q^3 + 3 * q^2 - 7 * q + 9) + (5 * q^3 - 8 * q^2 + 6 * q - 1) =
  4 * q^4 + 3 * q^3 - 5 * q^2 - q + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1665_166577


namespace NUMINAMATH_CALUDE_workday_end_time_l1665_166569

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a workday -/
structure Workday where
  startTime : Time
  lunchStartTime : Time
  workDuration : Nat  -- in minutes
  lunchDuration : Nat  -- in minutes

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

theorem workday_end_time (wd : Workday) : 
  wd.startTime = { hours := 7, minutes := 0 } →
  wd.lunchStartTime = { hours := 11, minutes := 30 } →
  wd.workDuration = 480 → -- 8 hours in minutes
  wd.lunchDuration = 30 →
  addMinutes wd.startTime (wd.workDuration + wd.lunchDuration) = { hours := 15, minutes := 30 } :=
by sorry

end NUMINAMATH_CALUDE_workday_end_time_l1665_166569


namespace NUMINAMATH_CALUDE_picture_distance_from_right_end_l1665_166542

/-- Given a wall and a picture with specific dimensions and placement,
    calculate the distance from the right end of the wall to the nearest edge of the picture. -/
theorem picture_distance_from_right_end 
  (wall_width : ℝ) 
  (picture_width : ℝ) 
  (left_gap : ℝ) 
  (h1 : wall_width = 24)
  (h2 : picture_width = 4)
  (h3 : left_gap = 5) :
  wall_width - (left_gap + picture_width) = 15 := by
  sorry

#check picture_distance_from_right_end

end NUMINAMATH_CALUDE_picture_distance_from_right_end_l1665_166542


namespace NUMINAMATH_CALUDE_not_parallel_to_skew_line_l1665_166564

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem not_parallel_to_skew_line 
  (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b :=
sorry

end NUMINAMATH_CALUDE_not_parallel_to_skew_line_l1665_166564


namespace NUMINAMATH_CALUDE_sum_of_factors_l1665_166598

theorem sum_of_factors (x : ℝ) : 
  x = 2 → (x^3 - x^2 + 1) + (x^2 + x + 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1665_166598


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l1665_166541

theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : days = 50)
  (h3 : evaporation_percentage = 40) : 
  (initial_water * evaporation_percentage / 100) / days = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l1665_166541


namespace NUMINAMATH_CALUDE_max_current_speed_is_26_l1665_166584

/-- The maximum possible integer value for the river current speed --/
def max_current_speed : ℕ := 26

/-- The speed at which Mumbo runs --/
def mumbo_speed : ℕ := 11

/-- The speed at which Yumbo walks --/
def yumbo_speed : ℕ := 6

/-- Represents the travel scenario described in the problem --/
structure TravelScenario where
  x : ℝ  -- distance from origin to Mumbo's raft storage
  y : ℝ  -- distance from origin to Yumbo's raft storage
  v : ℕ  -- speed of the river current

/-- Condition that Yumbo arrives earlier than Mumbo --/
def yumbo_arrives_earlier (s : TravelScenario) : Prop :=
  s.y / yumbo_speed < s.x / mumbo_speed + (s.x + s.y) / s.v

/-- Main theorem stating that 26 is the maximum possible current speed --/
theorem max_current_speed_is_26 :
  ∀ s : TravelScenario,
    s.x > 0 ∧ s.y > 0 ∧ s.x < s.y ∧ s.v ≥ 6 ∧ yumbo_arrives_earlier s
    → s.v ≤ max_current_speed :=
by sorry

#check max_current_speed_is_26

end NUMINAMATH_CALUDE_max_current_speed_is_26_l1665_166584


namespace NUMINAMATH_CALUDE_square_area_15cm_l1665_166525

/-- The area of a square with side length 15 cm is 225 square centimeters. -/
theorem square_area_15cm (side_length : ℝ) (area : ℝ) : 
  side_length = 15 → area = side_length ^ 2 → area = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_15cm_l1665_166525


namespace NUMINAMATH_CALUDE_repeating_digits_equation_l1665_166573

theorem repeating_digits_equation (a b c : ℕ) : 
  (∀ n : ℕ, a * (10^n - 1) / 9 * 10^n + b * (10^n - 1) / 9 + 1 = (c * (10^n - 1) / 9 + 1)^2) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 5 ∧ c = 3) ∨ 
   (a = 4 ∧ b = 8 ∧ c = 6) ∨ 
   (a = 9 ∧ b = 9 ∧ c = 9)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_digits_equation_l1665_166573


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1665_166517

theorem intersection_point_sum (a b : ℝ) :
  (2 = (1/3) * 1 + a) ∧ (1 = (1/3) * 2 + b) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1665_166517


namespace NUMINAMATH_CALUDE_problem_solution_l1665_166562

theorem problem_solution (y : ℝ) (d e f : ℕ+) :
  y = Real.sqrt ((Real.sqrt 75) / 2 + 5 / 2) →
  y^100 = 3*y^98 + 18*y^96 + 15*y^94 - y^50 + (d : ℝ)*y^46 + (e : ℝ)*y^44 + (f : ℝ)*y^40 →
  (d : ℝ) + (e : ℝ) + (f : ℝ) = 556.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1665_166562


namespace NUMINAMATH_CALUDE_triangle_theorem_l1665_166516

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a^2 + t.b^2 - t.c^2) * Real.tan t.C = Real.sqrt 2 * t.a * t.b)
  (h2 : t.c = 2)
  (h3 : t.b = 2 * Real.sqrt 2) :
  t.C = π/4 ∧ t.a = 2 ∧ (1/2 * t.a * t.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1665_166516


namespace NUMINAMATH_CALUDE_sequence_relations_l1665_166579

theorem sequence_relations (x y : ℕ → ℝ) 
  (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2)
  (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) :
  (∀ k, x k = 6 * x (k - 1) - x (k - 2)) ∧
  (∀ k, x k = 34 * x (k - 2) - x (k - 4)) ∧
  (∀ k, x k = 198 * x (k - 3) - x (k - 6)) ∧
  (∀ k, y k = 6 * y (k - 1) - y (k - 2)) ∧
  (∀ k, y k = 34 * y (k - 2) - y (k - 4)) ∧
  (∀ k, y k = 198 * y (k - 3) - y (k - 6)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_relations_l1665_166579


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1665_166560

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y : ℝ, x + y = 2 → 2^x + 2^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1665_166560


namespace NUMINAMATH_CALUDE_museum_ring_display_height_l1665_166528

/-- Calculates the total vertical distance of a sequence of rings -/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let n := (top_diameter - bottom_diameter) / 2 + 1
  let sum_inside_diameters := n * (top_diameter - thickness + bottom_diameter - thickness) / 2
  sum_inside_diameters + 2 * thickness

/-- Theorem stating that the total vertical distance for the given ring sequence is 325 cm -/
theorem museum_ring_display_height : total_vertical_distance 36 4 1 = 325 := by
  sorry

#eval total_vertical_distance 36 4 1

end NUMINAMATH_CALUDE_museum_ring_display_height_l1665_166528


namespace NUMINAMATH_CALUDE_yardley_snowfall_l1665_166561

/-- The total snowfall in Yardley is the sum of morning and afternoon snowfall -/
theorem yardley_snowfall (morning_snowfall afternoon_snowfall : ℚ) 
  (h1 : morning_snowfall = 0.125)
  (h2 : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_yardley_snowfall_l1665_166561


namespace NUMINAMATH_CALUDE_expense_increase_percentage_is_ten_percent_l1665_166563

/-- Calculates the percentage increase in monthly expenses given the initial salary,
    savings rate, and new savings amount. -/
def calculate_expense_increase_percentage (salary : ℚ) (savings_rate : ℚ) (new_savings : ℚ) : ℚ :=
  let original_savings := salary * savings_rate
  let original_expenses := salary - original_savings
  let additional_expense := original_savings - new_savings
  (additional_expense / original_expenses) * 100

/-- Theorem stating that for the given conditions, the expense increase percentage is 10% -/
theorem expense_increase_percentage_is_ten_percent :
  calculate_expense_increase_percentage 20000 (1/10) 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_is_ten_percent_l1665_166563


namespace NUMINAMATH_CALUDE_algebraic_operation_equality_l1665_166593

theorem algebraic_operation_equality (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_operation_equality_l1665_166593


namespace NUMINAMATH_CALUDE_sequence_general_formula_l1665_166585

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 3 + 2^n,
    this theorem proves the general formula for a_n. -/
theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 3 + 2^n) :
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l1665_166585


namespace NUMINAMATH_CALUDE_lillys_fish_l1665_166550

theorem lillys_fish (total : ℕ) (rosys_fish : ℕ) (lillys_fish : ℕ) : 
  total = 21 → rosys_fish = 11 → total = rosys_fish + lillys_fish → lillys_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lillys_fish_l1665_166550


namespace NUMINAMATH_CALUDE_max_colors_without_monochromatic_trapezium_l1665_166514

/-- Regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Sequence of regular n-gons where each subsequent polygon's vertices are midpoints of the previous polygon's edges -/
def NGonSequence (n : ℕ) (m : ℕ) : ℕ → RegularNGon n
  | 0 => sorry
  | i + 1 => sorry

/-- A coloring of vertices of m n-gons using k colors -/
def Coloring (n m k : ℕ) := Fin m → Fin n → Fin k

/-- Predicate to check if four points form an isosceles trapezium -/
def IsIsoscelesTrapezium (a b c d : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a coloring contains a monochromatic isosceles trapezium -/
def HasMonochromaticIsoscelesTrapezium (n m k : ℕ) (coloring : Coloring n m k) : Prop := sorry

/-- The maximum number of colors that can be used without forming a monochromatic isosceles trapezium -/
theorem max_colors_without_monochromatic_trapezium 
  (n : ℕ) (m : ℕ) (h : m ≥ n^2 - n + 1) :
  (∃ (k : ℕ), k = n - 1 ∧ 
    (∀ (coloring : Coloring n m k), HasMonochromaticIsoscelesTrapezium n m k coloring) ∧
    (∃ (coloring : Coloring n m (k + 1)), ¬HasMonochromaticIsoscelesTrapezium n m (k + 1) coloring)) :=
sorry

end NUMINAMATH_CALUDE_max_colors_without_monochromatic_trapezium_l1665_166514


namespace NUMINAMATH_CALUDE_career_preference_theorem_l1665_166574

/-- Represents the degrees in a circle graph for a career preference -/
def career_preference_degrees (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) : ℚ :=
  ((male_ratio * male_preference + female_ratio * female_preference) / 
   (male_ratio + female_ratio)) * 360

/-- Theorem: The degrees for the given career preference -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/4) (3/4) = 198 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_theorem_l1665_166574


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1665_166505

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem repeating_decimal_sum :
  SingleDigitRepeatingDecimal 0 3 + TwoDigitRepeatingDecimal 0 6 = 13 / 33 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1665_166505


namespace NUMINAMATH_CALUDE_total_rejection_is_0_75_percent_l1665_166535

-- Define the rejection rates and inspection proportion
def john_rejection_rate : ℝ := 0.005
def jane_rejection_rate : ℝ := 0.009
def jane_inspection_proportion : ℝ := 0.625

-- Define the total rejection percentage
def total_rejection_percentage : ℝ :=
  jane_rejection_rate * jane_inspection_proportion +
  john_rejection_rate * (1 - jane_inspection_proportion)

-- Theorem statement
theorem total_rejection_is_0_75_percent :
  total_rejection_percentage = 0.0075 := by
  sorry

#eval total_rejection_percentage

end NUMINAMATH_CALUDE_total_rejection_is_0_75_percent_l1665_166535


namespace NUMINAMATH_CALUDE_minimum_guests_l1665_166553

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ min_guests : ℕ, min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l1665_166553


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1665_166547

/-- The lateral area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1665_166547


namespace NUMINAMATH_CALUDE_value_of_q_l1665_166558

theorem value_of_q (a q : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * q) : q = 49 := by
  sorry

end NUMINAMATH_CALUDE_value_of_q_l1665_166558


namespace NUMINAMATH_CALUDE_division_remainder_549547_by_7_l1665_166507

theorem division_remainder_549547_by_7 : 
  549547 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_division_remainder_549547_by_7_l1665_166507


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l1665_166523

/-- The number of ways to arrange n students in a row -/
def arrange (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with 2 specific students not at the ends -/
def arrangeNotAtEnds (n : ℕ) : ℕ :=
  arrange (n - 2) * (arrange (n - 3))

/-- The number of ways to arrange n students in a row with 2 specific students adjacent -/
def arrangeAdjacent (n : ℕ) : ℕ :=
  2 * arrange (n - 1)

/-- The number of ways to arrange n students in a row with 2 specific students not adjacent -/
def arrangeNotAdjacent (n : ℕ) : ℕ :=
  arrange n - arrangeAdjacent n

/-- The main theorem -/
theorem student_arrangement_theorem :
  (arrangeNotAtEnds 5 = 36) ∧
  (arrangeAdjacent 5 * arrangeNotAdjacent 3 = 24) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l1665_166523


namespace NUMINAMATH_CALUDE_comparison_inequality_l1665_166521

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1665_166521


namespace NUMINAMATH_CALUDE_cassy_jars_left_l1665_166502

/-- The number of jars left unpacked when Cassy fills all boxes -/
def jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
                       (jars_per_box2 : ℕ) (num_boxes2 : ℕ) 
                       (total_jars : ℕ) : ℕ :=
  total_jars - (jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2)

theorem cassy_jars_left :
  jars_left_unpacked 12 10 10 30 500 = 80 := by
  sorry

end NUMINAMATH_CALUDE_cassy_jars_left_l1665_166502


namespace NUMINAMATH_CALUDE_min_value_of_u_l1665_166586

/-- Given that x and y are real numbers satisfying 2x + y ≥ 1, 
    the function u = x² + 4x + y² - 2y has a minimum value of -9/5 -/
theorem min_value_of_u (x y : ℝ) (h : 2 * x + y ≥ 1) :
  ∃ (min_u : ℝ), min_u = -9/5 ∧ ∀ (x' y' : ℝ), 2 * x' + y' ≥ 1 → 
    x'^2 + 4*x' + y'^2 - 2*y' ≥ min_u :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_u_l1665_166586


namespace NUMINAMATH_CALUDE_f_value_at_5pi_3_l1665_166576

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

theorem f_value_at_5pi_3 (f : ℝ → ℝ) :
  is_even f →
  smallest_positive_period f π →
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sin (x/2)) →
  f (5*π/3) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_3_l1665_166576


namespace NUMINAMATH_CALUDE_fraction_proof_l1665_166529

theorem fraction_proof (t k : ℚ) (f : ℚ) 
  (h1 : t = f * (k - 32))
  (h2 : t = 105)
  (h3 : k = 221) :
  f = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_proof_l1665_166529


namespace NUMINAMATH_CALUDE_cone_volume_l1665_166551

/-- The volume of a cone with given conditions -/
theorem cone_volume (r h : ℝ) (hr : r = 3) 
  (hθ : 2 * π * r = 2 * π * r / 3 * 9) : 
  (1 / 3) * π * r^2 * h = 18 * Real.sqrt 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1665_166551


namespace NUMINAMATH_CALUDE_cheap_module_count_l1665_166543

/-- Represents the stock of modules -/
structure ModuleStock where
  expensive_count : ℕ
  cheap_count : ℕ

/-- The cost of an expensive module -/
def expensive_cost : ℚ := 10

/-- The cost of a cheap module -/
def cheap_cost : ℚ := 3.5

/-- The total value of the stock -/
def total_value (stock : ModuleStock) : ℚ :=
  (stock.expensive_count : ℚ) * expensive_cost + (stock.cheap_count : ℚ) * cheap_cost

/-- The total count of modules in the stock -/
def total_count (stock : ModuleStock) : ℕ :=
  stock.expensive_count + stock.cheap_count

theorem cheap_module_count (stock : ModuleStock) :
  total_value stock = 45 ∧ total_count stock = 11 → stock.cheap_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_cheap_module_count_l1665_166543


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1665_166511

/-- The value of d for which the line y = 4x + d is tangent to the parabola y^2 = 16x -/
theorem tangent_line_to_parabola : 
  ∃ (d : ℝ), (∀ x y : ℝ, y = 4*x + d ∧ y^2 = 16*x → (∃! x', y' = 4*x' + d ∧ y'^2 = 16*x')) → d = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1665_166511


namespace NUMINAMATH_CALUDE_solutions_for_20_l1665_166544

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ :=
  4 * n

/-- The property that |x| + |y| = 1 has 4 solutions -/
axiom base_case : num_solutions 1 = 4

/-- The property that the number of solutions increases by 4 for each unit increase -/
axiom induction_step : ∀ n : ℕ, num_solutions (n + 1) = num_solutions n + 4

/-- The theorem to be proved -/
theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_20_l1665_166544


namespace NUMINAMATH_CALUDE_kanul_total_amount_l1665_166596

theorem kanul_total_amount (T : ℝ) : 
  T = 500 + 400 + 0.1 * T → T = 1000 := by
  sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l1665_166596


namespace NUMINAMATH_CALUDE_milk_ratio_l1665_166552

def total_cartons : ℕ := 24
def regular_cartons : ℕ := 3

theorem milk_ratio :
  let chocolate_cartons := total_cartons - regular_cartons
  (chocolate_cartons : ℚ) / regular_cartons = 7 / 1 :=
by sorry

end NUMINAMATH_CALUDE_milk_ratio_l1665_166552


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1665_166538

/-- The interest rate problem --/
theorem interest_rate_problem
  (principal : ℝ)
  (rate_a : ℝ)
  (time : ℝ)
  (gain_b : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_a = 10)
  (h3 : time = 3)
  (h4 : gain_b = 157.5)
  : ∃ (rate_c : ℝ), rate_c = 11.5 ∧
    gain_b = (principal * rate_c / 100 * time) - (principal * rate_a / 100 * time) :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1665_166538


namespace NUMINAMATH_CALUDE_A_intersect_B_l1665_166509

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {y | ∃ x ∈ A, y = Real.exp x}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1665_166509


namespace NUMINAMATH_CALUDE_fifi_closet_total_hangers_l1665_166589

/-- The number of colored hangers in Fifi's closet -/
def total_hangers (pink green blue yellow : ℕ) : ℕ := pink + green + blue + yellow

/-- The conditions of Fifi's closet hangers -/
def fifi_closet_conditions (pink green blue yellow : ℕ) : Prop :=
  pink = 7 ∧ green = 4 ∧ blue = green - 1 ∧ yellow = blue - 1

/-- Theorem: The total number of colored hangers in Fifi's closet is 16 -/
theorem fifi_closet_total_hangers :
  ∃ (pink green blue yellow : ℕ),
    fifi_closet_conditions pink green blue yellow ∧
    total_hangers pink green blue yellow = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifi_closet_total_hangers_l1665_166589


namespace NUMINAMATH_CALUDE_det_special_matrix_l1665_166559

theorem det_special_matrix (x : ℝ) : 
  Matrix.det (![![x + 3, x, x], ![x, x + 3, x], ![x, x, x + 3]]) = 27 * x + 27 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1665_166559


namespace NUMINAMATH_CALUDE_factor_ab_squared_minus_25a_l1665_166570

theorem factor_ab_squared_minus_25a (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_ab_squared_minus_25a_l1665_166570


namespace NUMINAMATH_CALUDE_no_common_values_under_180_l1665_166571

theorem no_common_values_under_180 : 
  ¬ ∃ x : ℕ, x < 180 ∧ x % 13 = 2 ∧ x % 8 = 5 := by
sorry

end NUMINAMATH_CALUDE_no_common_values_under_180_l1665_166571


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l1665_166518

theorem smallest_number_with_remainder (n : ℕ) : 
  n = 1996 ↔ 
  (n > 1992 ∧ 
   n % 9 = 7 ∧ 
   ∀ m, m > 1992 ∧ m % 9 = 7 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l1665_166518


namespace NUMINAMATH_CALUDE_min_value_fraction_l1665_166515

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (x + 2 * y) / (x * y) ≥ 3 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ (x₀ + 2 * y₀) / (x₀ * y₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1665_166515


namespace NUMINAMATH_CALUDE_committee_count_12_5_l1665_166534

/-- The number of ways to choose a committee of size k from a group of n people -/
def committeeCount (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the entire group -/
def groupSize : ℕ := 12

/-- The size of the committee to be chosen -/
def committeeSize : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from 12 people is 792 -/
theorem committee_count_12_5 : 
  committeeCount groupSize committeeSize = 792 := by sorry

end NUMINAMATH_CALUDE_committee_count_12_5_l1665_166534


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l1665_166545

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l1665_166545


namespace NUMINAMATH_CALUDE_at_op_difference_l1665_166539

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem at_op_difference : at_op 9 6 - at_op 6 9 = -9 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l1665_166539


namespace NUMINAMATH_CALUDE_train_passing_jogger_l1665_166572

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (v_jogger v_train : ℝ) (train_length initial_distance : ℝ) :
  v_jogger = 10 * 1000 / 3600 →
  v_train = 46 * 1000 / 3600 →
  train_length = 120 →
  initial_distance = 340 →
  (initial_distance + train_length) / (v_train - v_jogger) = 46 := by
  sorry

#check train_passing_jogger

end NUMINAMATH_CALUDE_train_passing_jogger_l1665_166572


namespace NUMINAMATH_CALUDE_problem_solution_l1665_166537

theorem problem_solution (a b c : ℕ) 
  (ha : a > 0 ∧ a < 10) 
  (hb : b > 0 ∧ b < 10) 
  (hc : c > 0 ∧ c < 10) 
  (h_prob : (1/a + 1/b + 1/c) - (1/a * 1/b + 1/a * 1/c + 1/b * 1/c) + (1/a * 1/b * 1/c) = 7/15) : 
  (1 - 1/a) * (1 - 1/b) * (1 - 1/c) = 8/15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1665_166537


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l1665_166587

/-- Represents the lottery with MegaBall, WinnerBalls, and BonusBall -/
structure Lottery where
  megaBallCount : ℕ
  winnerBallCount : ℕ
  bonusBallCount : ℕ
  winnerBallsPicked : ℕ

/-- Calculates the probability of winning the lottery -/
def winningProbability (l : Lottery) : ℚ :=
  1 / (l.megaBallCount * (l.winnerBallCount.choose l.winnerBallsPicked) * l.bonusBallCount)

/-- The specific lottery configuration -/
def ourLottery : Lottery :=
  { megaBallCount := 30
    winnerBallCount := 50
    bonusBallCount := 15
    winnerBallsPicked := 5 }

/-- Theorem stating the probability of winning our specific lottery -/
theorem lottery_winning_probability :
    winningProbability ourLottery = 1 / 953658000 := by
  sorry

#eval winningProbability ourLottery

end NUMINAMATH_CALUDE_lottery_winning_probability_l1665_166587


namespace NUMINAMATH_CALUDE_tangent_sum_equality_l1665_166500

theorem tangent_sum_equality (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_equality : Real.tan (α - β) = Real.sin (2 * β)) :
  Real.tan α + Real.tan β = 2 * Real.tan (2 * β) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_equality_l1665_166500


namespace NUMINAMATH_CALUDE_third_wednesday_not_22nd_l1665_166530

def is_third_wednesday (day : ℕ) : Prop :=
  ∃ (first_wednesday : ℕ), 
    1 ≤ first_wednesday ∧ 
    first_wednesday ≤ 7 ∧ 
    day = first_wednesday + 14

theorem third_wednesday_not_22nd : 
  ¬ is_third_wednesday 22 :=
sorry

end NUMINAMATH_CALUDE_third_wednesday_not_22nd_l1665_166530


namespace NUMINAMATH_CALUDE_logans_father_cartons_l1665_166548

/-- The number of cartons Logan's father usually receives -/
def usual_cartons : ℕ := 50

/-- The number of jars in each carton -/
def jars_per_carton : ℕ := 20

/-- The number of cartons received in the particular week -/
def received_cartons : ℕ := usual_cartons - 20

/-- The number of damaged jars from partially damaged cartons -/
def partially_damaged_jars : ℕ := 5 * 3

/-- The number of damaged jars from the totally damaged carton -/
def totally_damaged_jars : ℕ := jars_per_carton

/-- The total number of damaged jars -/
def total_damaged_jars : ℕ := partially_damaged_jars + totally_damaged_jars

/-- The number of jars good for sale in the particular week -/
def good_jars : ℕ := 565

theorem logans_father_cartons :
  jars_per_carton * received_cartons - total_damaged_jars = good_jars :=
by sorry

end NUMINAMATH_CALUDE_logans_father_cartons_l1665_166548


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l1665_166595

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l1665_166595


namespace NUMINAMATH_CALUDE_range_of_t_l1665_166591

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y
axiom f_inequality : ∀ t, f (3*t) + f ((1/3) - t) > 0

-- Define the set of t that satisfies the conditions
def T : Set ℝ := {t | -1/6 < t ∧ t ≤ 1/3}

-- Theorem to prove
theorem range_of_t : ∀ t, (f (3*t) + f ((1/3) - t) > 0) ↔ t ∈ T := by sorry

end NUMINAMATH_CALUDE_range_of_t_l1665_166591


namespace NUMINAMATH_CALUDE_circles_intersection_tangent_equality_points_l1665_166524

-- Define the circles and ellipse
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 6*y + 32 = 0
def C2 (a x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*(8-a)*y + 4*a + 12 = 0
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Theorem for part (I)
theorem circles_intersection :
  ∀ a : ℝ, C1 4 2 ∧ C1 6 4 ∧ C2 a 4 2 ∧ C2 a 6 4 := by sorry

-- Theorem for part (II)
theorem tangent_equality_points :
  ∀ x y : ℝ, Ellipse x y →
    (∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₁*x - 2*(8-a₁)*y + 4*a₁ + 12) ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₂*x - 2*(8-a₂)*y + 4*a₂ + 12)) ↔
    ((x = 2 ∧ y = 0) ∨ (x = 6/5 ∧ y = -4/5)) := by sorry

end NUMINAMATH_CALUDE_circles_intersection_tangent_equality_points_l1665_166524


namespace NUMINAMATH_CALUDE_vasya_problem_impossible_l1665_166532

theorem vasya_problem_impossible : ¬ ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ + 3 * x₂ + 15 * x₃ = 100 ∧ 11 * x₁ + 8 * x₂ = 144 := by
  sorry

end NUMINAMATH_CALUDE_vasya_problem_impossible_l1665_166532


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_max_area_triangle_AOB_l1665_166513

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y b : ℝ) : Prop := x + 2*y - 2*b = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (b : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ line A.1 A.2 b ∧ line B.1 B.2 b

-- Define the circle with diameter AB
def circle_equation (x y x0 y0 r : ℝ) : Prop := (x - x0)^2 + (y - y0)^2 = r^2

-- Part I: Circle tangent to x-axis
theorem circle_tangent_to_x_axis 
  (A B : ℝ × ℝ) (b : ℝ) 
  (h_intersect : intersection_points A B b) 
  (h_tangent : ∃ (x0 y0 r : ℝ), circle_equation A.1 A.2 x0 y0 r ∧ 
                                circle_equation B.1 B.2 x0 y0 r ∧ 
                                y0 = r) : 
  ∃ (x0 y0 : ℝ), circle_equation x y x0 y0 4 ∧ x0 = 24/5 ∧ y0 = -4 :=
sorry

-- Part II: Maximum area of triangle AOB
theorem max_area_triangle_AOB 
  (A B : ℝ × ℝ) (b : ℝ) 
  (h_intersect : intersection_points A B b) 
  (h_negative_y : b < 0) : 
  ∃ (max_area : ℝ), max_area = 32 * Real.sqrt 3 / 9 ∧ 
    ∀ (area : ℝ), area ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_max_area_triangle_AOB_l1665_166513


namespace NUMINAMATH_CALUDE_opposite_values_imply_result_l1665_166583

theorem opposite_values_imply_result (a b : ℝ) : 
  |a + 2| = -(b - 3)^2 → a^b + 3*(a - b) = -23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_imply_result_l1665_166583


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1665_166510

theorem cubic_equation_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = a ∨ x = -b ∨ x = c) : 
  a = 1 ∧ b = -1 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1665_166510


namespace NUMINAMATH_CALUDE_hendricks_guitar_price_l1665_166519

theorem hendricks_guitar_price 
  (gerald_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : gerald_price = 250) 
  (h2 : discount_percentage = 20) :
  gerald_price * (1 - discount_percentage / 100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_hendricks_guitar_price_l1665_166519
