import Mathlib

namespace NUMINAMATH_CALUDE_min_k_value_l820_82049

theorem min_k_value (k : ℕ) : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) →
  k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l820_82049


namespace NUMINAMATH_CALUDE_laundry_calculation_correct_l820_82046

/-- Represents the laundry problem setup -/
structure LaundrySetup where
  tub_capacity : Real
  clothes_weight : Real
  required_concentration : Real
  initial_detergent : Real

/-- Calculates the additional detergent and water needed for the laundry -/
def calculate_additions (setup : LaundrySetup) : Real × Real :=
  let additional_detergent := setup.tub_capacity * setup.required_concentration - setup.initial_detergent - setup.clothes_weight
  let additional_water := setup.tub_capacity - setup.clothes_weight - setup.initial_detergent - additional_detergent
  (additional_detergent, additional_water)

/-- The main theorem stating the correct additional amounts -/
theorem laundry_calculation_correct (setup : LaundrySetup) 
  (h1 : setup.tub_capacity = 15)
  (h2 : setup.clothes_weight = 4)
  (h3 : setup.required_concentration = 0.004)
  (h4 : setup.initial_detergent = 0.04) :
  calculate_additions setup = (0.004, 10.956) := by
  sorry

#eval calculate_additions { 
  tub_capacity := 15, 
  clothes_weight := 4, 
  required_concentration := 0.004, 
  initial_detergent := 0.04 
}

end NUMINAMATH_CALUDE_laundry_calculation_correct_l820_82046


namespace NUMINAMATH_CALUDE_fraction_equality_l820_82066

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a + c = 2*b) 
  (h2 : 2*b*d = c*(b + d)) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) : 
  a / b = c / d := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l820_82066


namespace NUMINAMATH_CALUDE_pencil_distribution_l820_82095

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (max_students : ℕ) :
  num_pens = 2010 →
  max_students = 30 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 30 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l820_82095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l820_82009

/-- Given an arithmetic sequence {a_n} with common difference d, 
    if a_1 + a_8 + a_15 = 72, then a_5 + 3d = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 8 + a 15 = 72 →           -- given sum condition
  a 5 + 3 * d = 24 := by            -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l820_82009


namespace NUMINAMATH_CALUDE_recipe_cups_needed_l820_82061

theorem recipe_cups_needed (servings : ℝ) (cups_per_serving : ℝ) 
  (h1 : servings = 18.0) 
  (h2 : cups_per_serving = 2.0) : 
  servings * cups_per_serving = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_recipe_cups_needed_l820_82061


namespace NUMINAMATH_CALUDE_class_survey_l820_82079

theorem class_survey (total_students : ℕ) (green_students : ℕ) (yellow_students : ℕ) (girls : ℕ) : 
  total_students = 30 →
  green_students = total_students / 2 →
  yellow_students = 9 →
  girls * 3 = (total_students - green_students - yellow_students) * 3 + girls →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_class_survey_l820_82079


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l820_82099

theorem subset_implies_a_equals_one :
  ∀ (a : ℝ),
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a - 2, 2 * a - 2}
  A ⊆ B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l820_82099


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l820_82035

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l820_82035


namespace NUMINAMATH_CALUDE_geometric_series_sum_l820_82014

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometricSum a r n = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l820_82014


namespace NUMINAMATH_CALUDE_smallest_tree_height_l820_82019

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l820_82019


namespace NUMINAMATH_CALUDE_product_second_fourth_is_seven_l820_82070

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The tenth term of the sequence is 25 -/
  tenth_term : a₁ + 9 * d = 25
  /-- The common difference is 3 -/
  diff_is_3 : d = 3

/-- The product of the second and fourth terms is 7 -/
theorem product_second_fourth_is_seven (seq : ArithmeticSequence) :
  (seq.a₁ + seq.d) * (seq.a₁ + 3 * seq.d) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_second_fourth_is_seven_l820_82070


namespace NUMINAMATH_CALUDE_multiply_nine_negative_three_l820_82093

theorem multiply_nine_negative_three : 9 * (-3) = -27 := by
  sorry

end NUMINAMATH_CALUDE_multiply_nine_negative_three_l820_82093


namespace NUMINAMATH_CALUDE_circle_proof_l820_82068

-- Define the points
def A : ℝ × ℝ := (5, 2)
def B : ℝ × ℝ := (3, 2)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define the circle equation for the first part
def circle_eq1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Define the circle equation for the second part
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

theorem circle_proof :
  -- Part 1
  (∀ x y : ℝ, circle_eq1 x y ↔ 
    ((x, y) = A ∨ (x, y) = B) ∧ 
    (∃ cx cy : ℝ, line_eq cx cy ∧ (x - cx)^2 + (y - cy)^2 = (5 - cx)^2 + (2 - cy)^2)) ∧
  -- Part 2
  (∀ x y : ℝ, circle_eq2 x y ↔ 
    ((x, y) = O ∨ (x, y) = (2, 0) ∨ (x, y) = (0, 4)) ∧ 
    (∃ cx cy r : ℝ, (x - cx)^2 + (y - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (2 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (4 - cy)^2 = r^2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_proof_l820_82068


namespace NUMINAMATH_CALUDE_cycle_original_price_l820_82051

/-- Given a cycle sold at a 20% loss for Rs. 1120, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1120)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price = 1400 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l820_82051


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l820_82053

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l820_82053


namespace NUMINAMATH_CALUDE_floor_sum_equals_140_l820_82097

theorem floor_sum_equals_140 (p q r s : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500) (h2 : r^2 + s^2 = 2500) (h3 : p * r = 1200) (h4 : q * s = 1200) :
  ⌊p + q + r + s⌋ = 140 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_140_l820_82097


namespace NUMINAMATH_CALUDE_sunset_colors_proof_l820_82074

/-- The number of colors the sky turns during a sunset --/
def sunset_colors (sunset_duration : ℕ) (color_change_interval : ℕ) : ℕ :=
  sunset_duration / color_change_interval

theorem sunset_colors_proof (hours : ℕ) (minutes_per_hour : ℕ) (color_change_interval : ℕ) :
  hours = 2 →
  minutes_per_hour = 60 →
  color_change_interval = 10 →
  sunset_colors (hours * minutes_per_hour) color_change_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_proof_l820_82074


namespace NUMINAMATH_CALUDE_student_scores_l820_82022

theorem student_scores (M P C : ℕ) : 
  M + P = 60 →
  C = P + 10 →
  (M + C) / 2 = 35 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l820_82022


namespace NUMINAMATH_CALUDE_morning_run_distance_l820_82036

/-- Represents the distances of various activities in miles -/
structure DailyActivities where
  morningRun : ℝ
  afternoonWalk : ℝ
  eveningBikeRide : ℝ

/-- Calculates the total distance covered in a day -/
def totalDistance (activities : DailyActivities) : ℝ :=
  activities.morningRun + activities.afternoonWalk + activities.eveningBikeRide

/-- Theorem stating that given the conditions, the morning run distance is 2 miles -/
theorem morning_run_distance 
  (activities : DailyActivities)
  (h1 : totalDistance activities = 18)
  (h2 : activities.afternoonWalk = 2 * activities.morningRun)
  (h3 : activities.eveningBikeRide = 12) :
  activities.morningRun = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_run_distance_l820_82036


namespace NUMINAMATH_CALUDE_x_plus_y_values_l820_82048

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) :
  (x + y = -3) ∨ (x + y = -9) :=
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l820_82048


namespace NUMINAMATH_CALUDE_painter_work_days_l820_82092

/-- Represents the number of work-days required for a given number of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : Prop :=
  painters * days = 6 * 2

theorem painter_work_days :
  work_days 6 2 → work_days 4 3 := by
  sorry

end NUMINAMATH_CALUDE_painter_work_days_l820_82092


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l820_82063

/-- Given a quadratic polynomial P(x) = x^2 + ax + b, 
    if P(10) + P(30) = 40, then P(20) = -80 -/
theorem quadratic_polynomial_property (a b : ℝ) : 
  let P : ℝ → ℝ := λ x => x^2 + a*x + b
  (P 10 + P 30 = 40) → P 20 = -80 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l820_82063


namespace NUMINAMATH_CALUDE_intersection_condition_l820_82082

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem intersection_condition (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a < -4 ∨ a ≥ 4 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l820_82082


namespace NUMINAMATH_CALUDE_unique_lattice_point_l820_82030

theorem unique_lattice_point : 
  ∃! (x y : ℤ), x^2 - y^2 = 75 ∧ x - y = 5 := by sorry

end NUMINAMATH_CALUDE_unique_lattice_point_l820_82030


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l820_82090

theorem arithmetic_progression_of_primes (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ →  -- The numbers are prime
  p₁ > 3 → p₂ > 3 → p₃ > 3 →        -- The numbers are greater than 3
  p₁ < p₂ ∧ p₂ < p₃ →               -- The numbers are in ascending order
  p₂ = p₁ + d →                     -- Definition of arithmetic progression
  p₃ = p₁ + 2*d →                   -- Definition of arithmetic progression
  ∃ k : ℕ, d = 6 * k                -- The common difference is divisible by 6
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l820_82090


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l820_82012

/-- A triangle with interior angles in the ratio 1:2:3 has its largest angle equal to 90 degrees -/
theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l820_82012


namespace NUMINAMATH_CALUDE_coordinate_proof_l820_82041

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) in the first quadrant of a Cartesian coordinate system,
prove that under certain conditions, their coordinates are (1, 5) and (8, 9) respectively.
-/
theorem coordinate_proof (x₁ y₁ x₂ y₂ : ℕ) : 
  -- Both coordinates are positive integers
  0 < x₁ ∧ 0 < y₁ ∧ 0 < x₂ ∧ 0 < y₂ →
  -- Angle OA > 45°
  y₁ > x₁ →
  -- Angle OB < 45°
  x₂ > y₂ →
  -- Area difference condition
  x₂ * y₂ = x₁ * y₁ + 67 →
  -- Conclusion: coordinates are (1, 5) and (8, 9)
  x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 8 ∧ y₂ = 9 := by
sorry

end NUMINAMATH_CALUDE_coordinate_proof_l820_82041


namespace NUMINAMATH_CALUDE_purple_walls_count_l820_82088

theorem purple_walls_count (total_rooms : ℕ) (walls_per_room : ℕ) (green_ratio : ℚ) : 
  total_rooms = 10 → 
  walls_per_room = 8 → 
  green_ratio = 3/5 → 
  (total_rooms - total_rooms * green_ratio) * walls_per_room = 32 := by
sorry

end NUMINAMATH_CALUDE_purple_walls_count_l820_82088


namespace NUMINAMATH_CALUDE_miley_purchase_cost_l820_82057

/-- Calculates the total cost of Miley's purchase including discounts and sales tax -/
def total_cost (cellphone_price earbuds_price case_price : ℝ)
               (cellphone_discount earbuds_discount case_discount sales_tax : ℝ) : ℝ :=
  let cellphone_total := 2 * cellphone_price * (1 - cellphone_discount)
  let earbuds_total := 2 * earbuds_price * (1 - earbuds_discount)
  let case_total := 2 * case_price * (1 - case_discount)
  let subtotal := cellphone_total + earbuds_total + case_total
  subtotal * (1 + sales_tax)

/-- Theorem stating that the total cost of Miley's purchase is $2006.64 -/
theorem miley_purchase_cost :
  total_cost 800 150 40 0.05 0.10 0.15 0.08 = 2006.64 := by
  sorry

end NUMINAMATH_CALUDE_miley_purchase_cost_l820_82057


namespace NUMINAMATH_CALUDE_mass_of_man_is_80kg_l820_82039

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating that the mass of the man is 80 kg -/
theorem mass_of_man_is_80kg :
  mass_of_man 4 2 0.01 1000 = 80 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_80kg_l820_82039


namespace NUMINAMATH_CALUDE_no_nonzero_solution_l820_82094

theorem no_nonzero_solution :
  ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 2 / y = 1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_l820_82094


namespace NUMINAMATH_CALUDE_sum_of_three_integers_with_product_625_l820_82098

theorem sum_of_three_integers_with_product_625 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 51 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_with_product_625_l820_82098


namespace NUMINAMATH_CALUDE_ordered_pairs_1806_l820_82004

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_ordered_pairs (n : ℕ) : ℕ := sorry

theorem ordered_pairs_1806 :
  prime_factorization 1806 = [(2, 1), (3, 2), (101, 1)] →
  count_ordered_pairs 1806 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_1806_l820_82004


namespace NUMINAMATH_CALUDE_eddy_rate_is_correct_l820_82037

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_rate : ℝ     -- Hillary's climbing rate in ft/hr
  hillary_stop : ℝ     -- Distance from summit where Hillary stops
  hillary_descent : ℝ  -- Hillary's descent rate in ft/hr
  start_time : ℝ       -- Start time in hours (0 represents 06:00)
  meet_time : ℝ        -- Time when Hillary and Eddy meet in hours

/-- Calculates Eddy's climbing rate given a climbing scenario -/
def eddy_rate (scenario : ClimbingScenario) : ℝ :=
  -- The actual calculation of Eddy's rate
  sorry

/-- Theorem stating that Eddy's climbing rate is 5000/6 ft/hr given the specific scenario -/
theorem eddy_rate_is_correct (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_rate = 800)
  (h3 : scenario.hillary_stop = 1000)
  (h4 : scenario.hillary_descent = 1000)
  (h5 : scenario.start_time = 0)
  (h6 : scenario.meet_time = 6) : 
  eddy_rate scenario = 5000 / 6 := by
  sorry

end NUMINAMATH_CALUDE_eddy_rate_is_correct_l820_82037


namespace NUMINAMATH_CALUDE_problem_solution_l820_82078

theorem problem_solution :
  (∃ a b c : ℝ, a * c = b * c ∧ a ≠ b) ∧
  (∀ a : ℝ, (¬ ∃ q : ℚ, a + 5 = q) ↔ (¬ ∃ q : ℚ, a = q)) ∧
  ((∀ a b : ℝ, a = b → a^2 = b^2) ∧ (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b)) ∧
  (∃ x : ℝ, x^2 < 1) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l820_82078


namespace NUMINAMATH_CALUDE_samuel_money_left_l820_82032

/-- Calculates the amount Samuel has left after receiving a share of the total amount and spending on drinks -/
def samuel_remaining_money (total : ℝ) (share_fraction : ℝ) (spend_fraction : ℝ) : ℝ :=
  total * share_fraction - total * spend_fraction

/-- Theorem stating that given the conditions in the problem, Samuel has $132 left -/
theorem samuel_money_left :
  let total : ℝ := 240
  let share_fraction : ℝ := 3/4
  let spend_fraction : ℝ := 1/5
  samuel_remaining_money total share_fraction spend_fraction = 132 := by
  sorry


end NUMINAMATH_CALUDE_samuel_money_left_l820_82032


namespace NUMINAMATH_CALUDE_inequality_proof_l820_82045

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l820_82045


namespace NUMINAMATH_CALUDE_right_triangle_k_values_l820_82023

/-- A right-angled triangle in a 2D Cartesian coordinate system. -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
                    (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 ∨
                    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem stating the possible values of k in the given right-angled triangle. -/
theorem right_triangle_k_values (triangle : RightTriangle)
  (h1 : triangle.B.1 - triangle.A.1 = 2 ∧ triangle.B.2 - triangle.A.2 = 1)
  (h2 : triangle.C.1 - triangle.A.1 = 3)
  (h3 : ∃ k, triangle.C.2 - triangle.A.2 = k) :
  ∃ k, (k = -6 ∨ k = -1) ∧ triangle.C.2 - triangle.A.2 = k :=
sorry


end NUMINAMATH_CALUDE_right_triangle_k_values_l820_82023


namespace NUMINAMATH_CALUDE_sqrt_two_between_integers_l820_82091

theorem sqrt_two_between_integers (n : ℕ+) : 
  (n : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < (n : ℝ) + 1 → n = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_between_integers_l820_82091


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l820_82073

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.sin α + Real.cos α < 0) : 
  α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l820_82073


namespace NUMINAMATH_CALUDE_jacks_healthcare_contribution_l820_82069

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate as a percentage. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (contribution_rate / 100)

/-- Proves that Jack's healthcare contribution is 57.5 cents per hour. -/
theorem jacks_healthcare_contribution :
  healthcare_contribution 25 2.3 = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_healthcare_contribution_l820_82069


namespace NUMINAMATH_CALUDE_girl_multiplication_mistake_l820_82001

theorem girl_multiplication_mistake (x : ℤ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_mistake_l820_82001


namespace NUMINAMATH_CALUDE_sum_of_roots_l820_82058

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 6) = 7) (hb : b * (b - 6) = 7) (hab : a ≠ b) :
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l820_82058


namespace NUMINAMATH_CALUDE_area_bisector_l820_82055

/-- A polygon in the xy-plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- The polygon described in the problem -/
def problemPolygon : Polygon :=
  { vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (6, 2), (6, 0)] }

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculate the area of a polygon on one side of a line y = mx passing through the origin -/
def areaOneSide (p : Polygon) (m : ℝ) : ℝ := sorry

/-- The main theorem -/
theorem area_bisector (p : Polygon) :
  p = problemPolygon →
  areaOneSide p (5/3) = (area p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bisector_l820_82055


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l820_82028

/-- Three circles are pairwise tangent if the distance between their centers
    is equal to the sum of their radii -/
def pairwise_tangent (r₁ r₂ r₃ : ℝ) (d₁₂ d₁₃ d₂₃ : ℝ) : Prop :=
  d₁₂ = r₁ + r₂ ∧ d₁₃ = r₁ + r₃ ∧ d₂₃ = r₂ + r₃

/-- The segments connecting the centers of three circles form a right triangle
    if the square of the longest side equals the sum of squares of the other two sides -/
def right_triangle (d₁₂ d₁₃ d₂₃ : ℝ) : Prop :=
  d₂₃^2 = d₁₂^2 + d₁₃^2

theorem smallest_circle_radius
  (r : ℝ)
  (h₁ : r > 0)
  (h₂ : r < 4)
  (h₃ : pairwise_tangent r 4 6 (r + 4) (r + 6) 10)
  (h₄ : right_triangle (r + 4) (r + 6) 10) :
  r = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l820_82028


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l820_82052

theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-2, k) →
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0) →
  k = 14 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l820_82052


namespace NUMINAMATH_CALUDE_rancher_cattle_movement_l820_82010

/-- A problem about a rancher moving cattle to higher ground. -/
theorem rancher_cattle_movement
  (total_cattle : ℕ)
  (truck_capacity : ℕ)
  (truck_speed : ℝ)
  (total_time : ℝ)
  (h1 : total_cattle = 400)
  (h2 : truck_capacity = 20)
  (h3 : truck_speed = 60)
  (h4 : total_time = 40)
  : (total_time * truck_speed) / (2 * (total_cattle / truck_capacity)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rancher_cattle_movement_l820_82010


namespace NUMINAMATH_CALUDE_cereal_eating_time_l820_82076

theorem cereal_eating_time (fat_rate mr_thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 25 →
  mr_thin_rate = 1 / 40 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + mr_thin_rate) : ℚ) = 1000 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l820_82076


namespace NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_l820_82040

theorem x_lt_2_necessary_not_sufficient :
  ∃ (x : ℝ), x^2 - x - 2 < 0 → x < 2 ∧
  ∃ (y : ℝ), y < 2 ∧ ¬(y^2 - y - 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_l820_82040


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l820_82081

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of containers with one missing egg -/
def partial_containers : ℕ := 3

/-- The minimum number of eggs specified in the problem -/
def min_eggs : ℕ := 150

/-- The number of eggs in the solution -/
def solution_eggs : ℕ := 162

/-- Theorem stating that the smallest number of eggs satisfying the conditions is 162 -/
theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  (∃ c : ℕ, n = full_container * c - partial_containers) →
  n > min_eggs →
  n ≥ solution_eggs ∧
  (∀ m : ℕ, m < solution_eggs → 
    (∀ d : ℕ, m ≠ full_container * d - partial_containers) ∨ m ≤ min_eggs) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l820_82081


namespace NUMINAMATH_CALUDE_positive_A_value_l820_82025

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 72) (h2 : A > 0) : A = 11 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l820_82025


namespace NUMINAMATH_CALUDE_sharon_wants_254_supplies_l820_82005

/-- The number of kitchen supplies Sharon wants to buy -/
def sharons_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 3 * angela_pots + 6
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_supplies 20 = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_wants_254_supplies_l820_82005


namespace NUMINAMATH_CALUDE_torn_pages_sum_not_1990_l820_82075

/-- Represents a sheet in the notebook -/
structure Sheet :=
  (number : ℕ)
  (h_range : number ≥ 1 ∧ number ≤ 96)

/-- The sum of page numbers on a sheet -/
def sheet_sum (s : Sheet) : ℕ := 4 * s.number - 1

/-- A selection of 25 sheets -/
def SheetSelection := { sel : Finset Sheet // sel.card = 25 }

theorem torn_pages_sum_not_1990 (sel : SheetSelection) :
  (sel.val.sum sheet_sum) ≠ 1990 := by
  sorry


end NUMINAMATH_CALUDE_torn_pages_sum_not_1990_l820_82075


namespace NUMINAMATH_CALUDE_isosceles_triangle_construction_impossibility_l820_82021

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  -- Side length of the two equal sides
  side : ℝ
  -- Base angle (half of the apex angle)
  base_angle : ℝ
  -- Height from the apex to the base
  height : ℝ
  -- Length of the angle bisector from the apex
  bisector : ℝ
  -- Constraint that the base angle is positive and less than π/2
  angle_constraint : 0 < base_angle ∧ base_angle < π/2

/-- Represents the ability to construct a geometric figure -/
def Constructible (α : Type) : Prop := sorry

/-- Represents the ability to trisect an angle -/
def AngleTrisectable (angle : ℝ) : Prop := sorry

/-- The main theorem stating the impossibility of general isosceles triangle construction -/
theorem isosceles_triangle_construction_impossibility 
  (h : ℝ) (l : ℝ) (h_pos : h > 0) (l_pos : l > 0) :
  ¬∀ (t : IsoscelesTriangle), 
    t.height = h ∧ t.bisector = l → 
    Constructible IsoscelesTriangle ∧ 
    ¬∀ (angle : ℝ), AngleTrisectable angle :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_construction_impossibility_l820_82021


namespace NUMINAMATH_CALUDE_high_school_students_l820_82008

theorem high_school_students (total_students : ℕ) : 
  (total_students * 40 / 100 : ℕ) * 70 / 100 = 140 → 
  total_students = 500 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l820_82008


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l820_82064

theorem smallest_x_absolute_value_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l820_82064


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l820_82002

theorem complex_square_plus_self (z : ℂ) (h : z = 1 + I) : z^2 + z = 1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l820_82002


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l820_82026

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l820_82026


namespace NUMINAMATH_CALUDE_union_and_complement_of_sets_l820_82047

-- Define the sets A and B
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem union_and_complement_of_sets :
  (A ∪ B = {x | x ≤ 3 ∨ x > 4}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x | x < -4 ∨ x ≥ -1}) := by
  sorry

end NUMINAMATH_CALUDE_union_and_complement_of_sets_l820_82047


namespace NUMINAMATH_CALUDE_cos_225_degrees_l820_82017

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l820_82017


namespace NUMINAMATH_CALUDE_birth_year_problem_l820_82015

theorem birth_year_problem : ∃! x : ℕ, x ∈ Finset.range 50 ∧ x^2 - x = 1892 := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l820_82015


namespace NUMINAMATH_CALUDE_prime_arithmetic_mean_median_l820_82006

theorem prime_arithmetic_mean_median (a b c : ℕ) : 
  a = 2 → 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a < b → 
  b < c → 
  b ≠ a + 1 → 
  (a + b + c) / 3 = 6 * b → 
  c / b = 83 / 5 := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_mean_median_l820_82006


namespace NUMINAMATH_CALUDE_base_eight_digit_product_l820_82065

/-- Represents a number in base 8 as a list of digits --/
def BaseEight := List Nat

/-- Converts a natural number to its base 8 representation --/
def toBaseEight (n : Nat) : BaseEight :=
  sorry

/-- Decrements each digit in a BaseEight number by 1, removing 0s --/
def decrementDigits (b : BaseEight) : BaseEight :=
  sorry

/-- Computes the product of a list of natural numbers --/
def product (l : List Nat) : Nat :=
  sorry

theorem base_eight_digit_product (n : Nat) :
  n = 7654 →
  product (decrementDigits (toBaseEight n)) = 10 :=
sorry

end NUMINAMATH_CALUDE_base_eight_digit_product_l820_82065


namespace NUMINAMATH_CALUDE_reading_ratio_two_to_three_nights_l820_82000

/-- Represents the number of pages read on each night -/
structure ReadingPattern where
  threeNightsAgo : ℕ
  twoNightsAgo : ℕ
  lastNight : ℕ
  tonight : ℕ

/-- Theorem stating the ratio of pages read two nights ago to three nights ago -/
theorem reading_ratio_two_to_three_nights (r : ReadingPattern) : 
  r.threeNightsAgo = 15 →
  r.lastNight = r.twoNightsAgo + 5 →
  r.tonight = 20 →
  r.threeNightsAgo + r.twoNightsAgo + r.lastNight + r.tonight = 100 →
  r.twoNightsAgo / r.threeNightsAgo = 2 := by
  sorry

#check reading_ratio_two_to_three_nights

end NUMINAMATH_CALUDE_reading_ratio_two_to_three_nights_l820_82000


namespace NUMINAMATH_CALUDE_f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l820_82054

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Statement 1
theorem f_one_root (c : ℝ) (h : c > 0) : 
  ∃! x, f x 0 c = 0 := by sorry

-- Statement 2
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f (-x) b 0 = -(f x b 0) := by sorry

-- Statement 3
theorem f_symmetric (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c := by sorry

-- Statement 4
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end NUMINAMATH_CALUDE_f_one_root_f_odd_when_c_zero_f_symmetric_f_more_than_two_roots_l820_82054


namespace NUMINAMATH_CALUDE_bauble_painting_friends_l820_82038

/-- The number of friends needed to complete the bauble painting task -/
def friends_needed (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) : ℕ :=
  let first_group_baubles_per_color := total_baubles / (first_group_colors + 2 * second_group_colors)
  let second_group_baubles_per_color := 2 * first_group_baubles_per_color
  let baubles_per_hour_needed := total_baubles / available_hours
  baubles_per_hour_needed / baubles_per_hour

theorem bauble_painting_friends (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) 
  (h1 : total_baubles = 1000)
  (h2 : total_colors = 20)
  (h3 : first_group_colors = 15)
  (h4 : second_group_colors = 5)
  (h5 : baubles_per_hour = 10)
  (h6 : available_hours = 50)
  (h7 : first_group_colors + second_group_colors = total_colors) :
  friends_needed total_baubles total_colors first_group_colors second_group_colors baubles_per_hour available_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_bauble_painting_friends_l820_82038


namespace NUMINAMATH_CALUDE_circle_area_difference_l820_82034

/-- The difference in area between two circles -/
theorem circle_area_difference : 
  let r1 : ℝ := 30  -- radius of the first circle
  let d2 : ℝ := 15  -- diameter of the second circle
  π * r1^2 - π * (d2/2)^2 = 843.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l820_82034


namespace NUMINAMATH_CALUDE_tea_mixture_price_l820_82013

/-- Given two teas mixed in equal proportions, proves that if one tea costs 74 rupees per kg
    and the mixture costs 69 rupees per kg, then the other tea costs 64 rupees per kg. -/
theorem tea_mixture_price (price_tea2 mixture_price : ℝ) 
  (h1 : price_tea2 = 74)
  (h2 : mixture_price = 69) :
  ∃ (price_tea1 : ℝ), 
    price_tea1 = 64 ∧ 
    (price_tea1 + price_tea2) / 2 = mixture_price :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l820_82013


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l820_82077

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^5 = y^2 + 4 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l820_82077


namespace NUMINAMATH_CALUDE_statement_holds_for_given_numbers_l820_82072

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def given_numbers : List ℕ := [45, 54, 63, 81]

theorem statement_holds_for_given_numbers :
  ∀ n ∈ given_numbers, (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_statement_holds_for_given_numbers_l820_82072


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l820_82059

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l820_82059


namespace NUMINAMATH_CALUDE_power_of_two_equation_l820_82033

theorem power_of_two_equation (r : ℤ) : 
  2^2001 - 2^2000 - 2^1999 + 2^1998 = r * 2^1998 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l820_82033


namespace NUMINAMATH_CALUDE_minimum_value_range_l820_82083

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The theorem stating the range of m for which f(x) has a minimum on (m, 6-m^2) --/
theorem minimum_value_range (m : ℝ) : 
  (∃ (c : ℝ), c ∈ Set.Ioo m (6 - m^2) ∧ 
    (∀ x ∈ Set.Ioo m (6 - m^2), f c ≤ f x)) ↔ 
  m ∈ Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_minimum_value_range_l820_82083


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l820_82007

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l820_82007


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l820_82096

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 3 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (1/x + 1/y = 1 + 2*Real.sqrt 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l820_82096


namespace NUMINAMATH_CALUDE_satellite_upgraded_fraction_l820_82050

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on a satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

theorem satellite_upgraded_fraction :
  ∀ s : Satellite,
    s.units = 24 →
    s.non_upgraded_per_unit * 6 = s.total_upgraded →
    upgraded_fraction s = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_satellite_upgraded_fraction_l820_82050


namespace NUMINAMATH_CALUDE_building_area_scientific_notation_l820_82016

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem building_area_scientific_notation :
  toScientificNotation 258000 = ScientificNotation.mk 2.58 5 sorry := by sorry

end NUMINAMATH_CALUDE_building_area_scientific_notation_l820_82016


namespace NUMINAMATH_CALUDE_percentage_difference_l820_82018

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l820_82018


namespace NUMINAMATH_CALUDE_inverse_variation_result_l820_82071

/-- Given that c² varies inversely with d⁴, this function represents their relationship -/
def inverse_relation (k : ℝ) (c d : ℝ) : Prop :=
  c^2 * d^4 = k

theorem inverse_variation_result (k : ℝ) :
  inverse_relation k 8 2 →
  inverse_relation k c 4 →
  c^2 = 4 := by
  sorry

#check inverse_variation_result

end NUMINAMATH_CALUDE_inverse_variation_result_l820_82071


namespace NUMINAMATH_CALUDE_smallest_modulus_of_z_l820_82087

theorem smallest_modulus_of_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 3*I) = 15) :
  Complex.abs z ≥ 8/5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_modulus_of_z_l820_82087


namespace NUMINAMATH_CALUDE_shampoo_duration_l820_82031

-- Define the amount of rose shampoo Janet has
def rose_shampoo : ℚ := 1/3

-- Define the amount of jasmine shampoo Janet has
def jasmine_shampoo : ℚ := 1/4

-- Define the amount of shampoo Janet uses per day
def daily_usage : ℚ := 1/12

-- Theorem statement
theorem shampoo_duration :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_duration_l820_82031


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l820_82011

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l820_82011


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l820_82003

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (1 + Complex.I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l820_82003


namespace NUMINAMATH_CALUDE_cube_difference_positive_l820_82084

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l820_82084


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l820_82024

/-- Proves the percentage drop in lawnmower value after 6 months -/
theorem lawnmower_value_drop (initial_value : ℝ) (final_value : ℝ) (yearly_drop_percent : ℝ) :
  initial_value = 100 →
  final_value = 60 →
  yearly_drop_percent = 20 →
  final_value = initial_value * (1 - yearly_drop_percent / 100) →
  (initial_value - (final_value / (1 - yearly_drop_percent / 100))) / initial_value * 100 = 25 := by
  sorry

#check lawnmower_value_drop

end NUMINAMATH_CALUDE_lawnmower_value_drop_l820_82024


namespace NUMINAMATH_CALUDE_work_completion_time_l820_82056

theorem work_completion_time (man_time son_time : ℝ) (h1 : man_time = 6) (h2 : son_time = 6) :
  1 / (1 / man_time + 1 / son_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l820_82056


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l820_82089

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l820_82089


namespace NUMINAMATH_CALUDE_linear_function_slope_condition_l820_82027

/-- Given a linear function y = (m-2)x + 2 + m with two points on its graph,
    prove that if x₁ < x₂ and y₁ > y₂, then m < 2 -/
theorem linear_function_slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = (m - 2) * x₁ + 2 + m)
  (h2 : y₂ = (m - 2) * x₂ + 2 + m)
  (h3 : x₁ < x₂)
  (h4 : y₁ > y₂) :
  m < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_slope_condition_l820_82027


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l820_82062

/-- The function f(x) = 2a^(x+1) - 3 has a fixed point at (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l820_82062


namespace NUMINAMATH_CALUDE_opposite_signs_absolute_difference_l820_82029

theorem opposite_signs_absolute_difference (a b : ℝ) :
  (abs a = 4) → (abs b = 2) → (a * b < 0) → abs (a - b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_absolute_difference_l820_82029


namespace NUMINAMATH_CALUDE_equation_solution_l820_82020

theorem equation_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l820_82020


namespace NUMINAMATH_CALUDE_max_weekly_profit_l820_82086

-- Define the price reduction x
def x : ℝ := 5

-- Define the original cost per unit
def original_cost : ℝ := 5

-- Define the original selling price per unit
def original_price : ℝ := 14

-- Define the initial weekly sales volume
def initial_volume : ℝ := 75

-- Define the proportionality constant k
def k : ℝ := 5

-- Define the increase in sales volume as a function of price reduction
def m (x : ℝ) : ℝ := k * x^2

-- Define the weekly sales profit as a function of price reduction
def y (x : ℝ) : ℝ := (original_price - x - original_cost) * (initial_volume + m x)

-- State the theorem
theorem max_weekly_profit :
  y x = 800 ∧ ∀ z, 0 ≤ z ∧ z < 9 → y z ≤ y x :=
sorry

end NUMINAMATH_CALUDE_max_weekly_profit_l820_82086


namespace NUMINAMATH_CALUDE_card_drawing_probability_l820_82044

theorem card_drawing_probability : 
  let cards : Finset ℕ := {1, 2, 3, 4, 5}
  let odd_cards : Finset ℕ := {1, 3, 5}
  let even_cards : Finset ℕ := {2, 4}
  let total_cards := cards.card
  let odd_count := odd_cards.card
  let even_count := even_cards.card

  let prob_first_odd := odd_count / total_cards
  let prob_second_even_given_first_odd := even_count / (total_cards - 1)

  prob_second_even_given_first_odd = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_card_drawing_probability_l820_82044


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_72_l820_82060

theorem sqrt_sum_equals_sqrt_72 (k : ℕ+) :
  Real.sqrt 2 + Real.sqrt 8 + Real.sqrt 18 = Real.sqrt k → k = 72 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_72_l820_82060


namespace NUMINAMATH_CALUDE_modulo_equivalence_exists_unique_l820_82042

theorem modulo_equivalence_exists_unique : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_exists_unique_l820_82042


namespace NUMINAMATH_CALUDE_cos_225_degrees_l820_82085

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l820_82085


namespace NUMINAMATH_CALUDE_square_condition_l820_82067

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

theorem square_condition (n : ℕ) :
  n > 0 → (is_perfect_square ((n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_square_condition_l820_82067


namespace NUMINAMATH_CALUDE_melanie_trout_count_l820_82080

/-- Prove that Melanie caught 8 trouts given the conditions -/
theorem melanie_trout_count (tom_count : ℕ) (melanie_count : ℕ) 
  (h1 : tom_count = 16) 
  (h2 : tom_count = 2 * melanie_count) : 
  melanie_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_melanie_trout_count_l820_82080


namespace NUMINAMATH_CALUDE_average_increase_l820_82043

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  avgRuns : ℚ

/-- Calculate the new average after scoring additional runs -/
def newAverage (player : CricketPlayer) (additionalRuns : ℕ) : ℚ :=
  (player.totalRuns + additionalRuns) / (player.innings + 1)

/-- The main theorem about the increase in average -/
theorem average_increase (player : CricketPlayer) (additionalRuns : ℕ) :
  player.innings = 10 ∧ 
  player.avgRuns = 35 ∧ 
  additionalRuns = 79 →
  newAverage player additionalRuns - player.avgRuns = 4 := by
sorry


end NUMINAMATH_CALUDE_average_increase_l820_82043
