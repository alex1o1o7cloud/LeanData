import Mathlib

namespace NUMINAMATH_CALUDE_lg_calculation_l3999_399909

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 5 * lg 20 + (lg 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l3999_399909


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_relation_l3999_399991

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- Side length a -/
  a : ℝ
  /-- Side length b -/
  b : ℝ
  /-- Side length c -/
  c : ℝ
  /-- Side length d, which is also the diameter of the semicircle -/
  d : ℝ
  /-- All side lengths are positive -/
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  /-- The quadrilateral is inscribed in a semicircle with diameter d -/
  inscribed : True

/-- The main theorem about the relationship between side lengths of an inscribed quadrilateral -/
theorem inscribed_quadrilateral_relation (q : InscribedQuadrilateral) :
  q.d^3 - (q.a^2 + q.b^2 + q.c^2) * q.d - 2 * q.a * q.b * q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_relation_l3999_399991


namespace NUMINAMATH_CALUDE_largest_domain_is_plus_minus_one_l3999_399936

def is_valid_domain (S : Set ℝ) : Prop :=
  (∀ x ∈ S, x ≠ 0) ∧ 
  (∀ x ∈ S, (1 / x) ∈ S) ∧
  (∃ g : ℝ → ℝ, ∀ x ∈ S, g x + g (1 / x) = 2 * x)

theorem largest_domain_is_plus_minus_one :
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} := by sorry

end NUMINAMATH_CALUDE_largest_domain_is_plus_minus_one_l3999_399936


namespace NUMINAMATH_CALUDE_new_salary_after_raise_l3999_399994

def original_salary : ℝ := 500
def raise_percentage : ℝ := 6

theorem new_salary_after_raise :
  original_salary * (1 + raise_percentage / 100) = 530 := by
  sorry

end NUMINAMATH_CALUDE_new_salary_after_raise_l3999_399994


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3999_399987

def f (x : ℝ) := x^2 - 6*x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3999_399987


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l3999_399910

theorem polynomial_coefficient_b (a c d : ℝ) : 
  ∃ (p q r s : ℂ),
    (∀ x : ℂ, x^4 + a*x^3 + 49*x^2 + c*x + d = 0 ↔ x = p ∨ x = q ∨ x = r ∨ x = s) ∧
    p + q = 5 + 2*I ∧
    r * s = 10 - I ∧
    p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧
    p.im ≠ 0 ∧ q.im ≠ 0 ∧ r.im ≠ 0 ∧ s.im ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l3999_399910


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3999_399954

theorem min_value_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3999_399954


namespace NUMINAMATH_CALUDE_johnny_fish_count_l3999_399952

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 120 →
  sony_multiplier = 7 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 15 := by
sorry

end NUMINAMATH_CALUDE_johnny_fish_count_l3999_399952


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3999_399966

theorem triangle_angle_problem (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π → 
  C = π / 5 → 
  a * Real.cos B - b * Real.cos A = c → 
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3999_399966


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3999_399923

theorem necessary_but_not_sufficient :
  (∃ a : ℝ, (a < 1 → a ≤ 1) ∧ ¬(a ≤ 1 → a < 1)) ∧
  (∃ x y : ℝ, (x = 1 ∧ y = 0 → x^2 + y^2 = 1) ∧ ¬(x^2 + y^2 = 1 → x = 1 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3999_399923


namespace NUMINAMATH_CALUDE_ellipse_equation_l3999_399922

theorem ellipse_equation (A B : ℝ × ℝ) (h1 : A = (0, 5/3)) (h2 : B = (1, 1)) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ 16 * x^2 + 9 * y^2 = 225) :=
sorry


end NUMINAMATH_CALUDE_ellipse_equation_l3999_399922


namespace NUMINAMATH_CALUDE_harry_apples_l3999_399946

theorem harry_apples (x : ℕ) : x + 5 = 84 → x = 79 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l3999_399946


namespace NUMINAMATH_CALUDE_obrien_hats_count_l3999_399919

/-- The number of hats Fire chief Simpson has -/
def simpson_hats : ℕ := 15

/-- The initial number of hats Policeman O'Brien had -/
def obrien_initial_hats : ℕ := 2 * simpson_hats + 5

/-- The number of hats Policeman O'Brien lost -/
def obrien_lost_hats : ℕ := 1

/-- The current number of hats Policeman O'Brien has -/
def obrien_current_hats : ℕ := obrien_initial_hats - obrien_lost_hats

theorem obrien_hats_count : obrien_current_hats = 34 := by
  sorry

end NUMINAMATH_CALUDE_obrien_hats_count_l3999_399919


namespace NUMINAMATH_CALUDE_jonathan_calorie_deficit_l3999_399969

/-- Jonathan's calorie consumption and burning schedule --/
structure CalorieSchedule where
  regularDailyIntake : ℕ
  saturdayIntake : ℕ
  dailyBurn : ℕ

/-- Calculate the weekly caloric deficit --/
def weeklyCalorieDeficit (schedule : CalorieSchedule) : ℕ :=
  7 * schedule.dailyBurn - (6 * schedule.regularDailyIntake + schedule.saturdayIntake)

/-- Theorem stating Jonathan's weekly caloric deficit --/
theorem jonathan_calorie_deficit :
  let schedule : CalorieSchedule := {
    regularDailyIntake := 2500,
    saturdayIntake := 3500,
    dailyBurn := 3000
  }
  weeklyCalorieDeficit schedule = 2500 := by
  sorry


end NUMINAMATH_CALUDE_jonathan_calorie_deficit_l3999_399969


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l3999_399942

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 25 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l3999_399942


namespace NUMINAMATH_CALUDE_correct_seasons_before_announcement_l3999_399962

/-- The number of seasons before the announcement of a TV show. -/
def seasons_before_announcement : ℕ := 9

/-- The number of episodes in a regular season. -/
def regular_season_episodes : ℕ := 22

/-- The number of episodes in the last season. -/
def last_season_episodes : ℕ := 26

/-- The duration of each episode in hours. -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all episodes in hours. -/
def total_watch_time : ℕ := 112

theorem correct_seasons_before_announcement :
  seasons_before_announcement * regular_season_episodes + last_season_episodes =
  total_watch_time / (episode_duration : ℚ) := by sorry

end NUMINAMATH_CALUDE_correct_seasons_before_announcement_l3999_399962


namespace NUMINAMATH_CALUDE_john_is_25_l3999_399930

-- Define John's age and his mother's age
def john_age : ℕ := sorry
def mother_age : ℕ := sorry

-- State the conditions
axiom age_difference : mother_age = john_age + 30
axiom sum_of_ages : john_age + mother_age = 80

-- Theorem to prove
theorem john_is_25 : john_age = 25 := by sorry

end NUMINAMATH_CALUDE_john_is_25_l3999_399930


namespace NUMINAMATH_CALUDE_mrs_wilsborough_tickets_l3999_399950

def prove_regular_tickets_bought : Prop :=
  let initial_savings : ℕ := 500
  let vip_ticket_cost : ℕ := 100
  let vip_tickets_bought : ℕ := 2
  let regular_ticket_cost : ℕ := 50
  let money_left : ℕ := 150
  let total_spent : ℕ := initial_savings - money_left
  let vip_tickets_total_cost : ℕ := vip_ticket_cost * vip_tickets_bought
  let regular_tickets_total_cost : ℕ := total_spent - vip_tickets_total_cost
  let regular_tickets_bought : ℕ := regular_tickets_total_cost / regular_ticket_cost
  regular_tickets_bought = 3

theorem mrs_wilsborough_tickets : prove_regular_tickets_bought := by
  sorry

end NUMINAMATH_CALUDE_mrs_wilsborough_tickets_l3999_399950


namespace NUMINAMATH_CALUDE_picture_placement_l3999_399958

/-- Given a wall and a picture with specified widths and offset, calculate the distance from the nearest end of the wall to the nearest edge of the picture. -/
theorem picture_placement (wall_width picture_width offset : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 5)
  (ho : offset = 2) :
  let center := (wall_width - picture_width) / 2
  let distance_to_nearest_edge := center - offset
  distance_to_nearest_edge = 8 := by sorry

end NUMINAMATH_CALUDE_picture_placement_l3999_399958


namespace NUMINAMATH_CALUDE_class_size_problem_l3999_399929

theorem class_size_problem (x : ℕ) : 
  x ≥ 46 → 
  (7 : ℚ) / 24 * x < 15 → 
  x = 48 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l3999_399929


namespace NUMINAMATH_CALUDE_farmer_cows_l3999_399964

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1/4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_l3999_399964


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3999_399975

/-- Represents the proportion of a population group -/
structure PopulationProportion where
  value : ℚ
  nonneg : 0 ≤ value

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_size : ℕ
  middle_aged_size : ℕ
  middle_aged_size_le_total : middle_aged_size ≤ total_size

/-- Given population proportions and a stratified sample, proves the total sample size -/
theorem stratified_sample_size 
  (elderly : PopulationProportion)
  (middle_aged : PopulationProportion)
  (young : PopulationProportion)
  (sample : StratifiedSample)
  (h1 : elderly.value + middle_aged.value + young.value = 1)
  (h2 : elderly.value = 2 / 10)
  (h3 : middle_aged.value = 3 / 10)
  (h4 : young.value = 5 / 10)
  (h5 : sample.middle_aged_size = 12) :
  sample.total_size = 40 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3999_399975


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l3999_399967

/-- A box containing colored balls -/
structure ColoredBallBox where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of drawing a yellow ball from a box -/
def probabilityYellowBall (box : ColoredBallBox) : ℚ :=
  box.yellowBalls / (box.redBalls + box.yellowBalls)

/-- Theorem: The probability of drawing a yellow ball from a box with 3 red and 2 yellow balls is 2/5 -/
theorem yellow_ball_probability :
  let box : ColoredBallBox := ⟨3, 2⟩
  probabilityYellowBall box = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_yellow_ball_probability_l3999_399967


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3999_399931

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (h1 : (2*x + 3) % 7 = 0) 
  (h2 : (3*y - 4) % 7 = 0) : 
  (∃ n : ℕ+, (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) → 
  (∃ n : ℕ+, n = 4 ∧ (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3999_399931


namespace NUMINAMATH_CALUDE_lucy_money_ratio_l3999_399935

theorem lucy_money_ratio : 
  ∀ (initial_amount spent remaining : ℚ),
    initial_amount = 30 →
    remaining = 15 →
    spent + remaining = initial_amount * (2/3) →
    spent / (initial_amount * (2/3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_ratio_l3999_399935


namespace NUMINAMATH_CALUDE_percentage_relation_l3999_399976

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.05 * x) (h2 : b = 0.25 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3999_399976


namespace NUMINAMATH_CALUDE_factor_expression_l3999_399900

theorem factor_expression (b : ℝ) : 63 * b^2 + 189 * b = 63 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3999_399900


namespace NUMINAMATH_CALUDE_remainder_theorem_l3999_399955

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x^2 + 3*u*y + v^2) % y = (2*v^2) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3999_399955


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l3999_399982

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 4) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l3999_399982


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3999_399940

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3999_399940


namespace NUMINAMATH_CALUDE_sector_perimeter_l3999_399968

/-- Given a sector with central angle 54° and radius 20 cm, its perimeter is (6π + 40) cm -/
theorem sector_perimeter (θ : Real) (r : Real) : 
  θ = 54 * Real.pi / 180 → r = 20 → 
  (θ * r) + 2 * r = 6 * Real.pi + 40 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3999_399968


namespace NUMINAMATH_CALUDE_trig_identity_l3999_399913

theorem trig_identity : 
  Real.sin (155 * π / 180) * Real.sin (55 * π / 180) + 
  Real.cos (25 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3999_399913


namespace NUMINAMATH_CALUDE_f_of_5_equals_22_l3999_399948

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem f_of_5_equals_22 : f 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_22_l3999_399948


namespace NUMINAMATH_CALUDE_expansion_equality_l3999_399901

theorem expansion_equality (x : ℝ) : 24 * (x + 3) * (2 * x - 4) = 48 * x^2 + 48 * x - 288 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l3999_399901


namespace NUMINAMATH_CALUDE_intersection_point_inside_circle_l3999_399927

/-- The intersection point of two lines is inside a circle iff a is within a specific range -/
theorem intersection_point_inside_circle (a : ℝ) :
  let P : ℝ × ℝ := (a, 3 * a)  -- Intersection point of y = x + 2a and y = 2x + a
  (P.1 - 1)^2 + (P.2 - 1)^2 < 4 ↔ -1/5 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_inside_circle_l3999_399927


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l3999_399998

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digit_sum_base7 :
  let a := 35
  let b := 12
  let product := toBase7 (toBase10 a * toBase10 b)
  sumDigitsBase7 product = 15 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l3999_399998


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3999_399978

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a - right,
    b := p.b - down }

theorem parabola_shift_theorem (p : Parabola) :
  shift_parabola { a := 2, b := 3 } 3 2 = { a := -1, b := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3999_399978


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l3999_399928

theorem number_multiplied_by_9999 :
  ∃ x : ℕ, x * 9999 = 724817410 ∧ x = 72492 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l3999_399928


namespace NUMINAMATH_CALUDE_car_travel_time_l3999_399957

/-- Proves that given the conditions of two cars A and B, the time taken by Car B to reach its destination is 1 hour. -/
theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 100 →
  time_A = 6 →
  ratio = 3 →
  (speed_A * time_A) / (speed_B * (speed_A * time_A / (ratio * speed_B))) = 1 := by
  sorry


end NUMINAMATH_CALUDE_car_travel_time_l3999_399957


namespace NUMINAMATH_CALUDE_work_efficiency_l3999_399945

/-- Given two workers A and B, where A can finish a work in 18 days and B can do the same work in half the time taken by A, this theorem proves that working together, they can finish 1/6 of the work in one day. -/
theorem work_efficiency (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 → 
  days_B = days_A / 2 → 
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_l3999_399945


namespace NUMINAMATH_CALUDE_distinct_tower_heights_94_bricks_l3999_399943

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights possible -/
def distinctTowerHeights (brickCount : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let maxY := 4
  List.range (maxY + 1)
    |> List.map (fun y => brickCount - y + 1)
    |> List.sum

/-- Theorem stating the number of distinct tower heights -/
theorem distinct_tower_heights_94_bricks :
  let brickDimensions : BrickDimensions := ⟨4, 10, 19⟩
  distinctTowerHeights 94 brickDimensions = 465 := by
  sorry

#eval distinctTowerHeights 94 ⟨4, 10, 19⟩

end NUMINAMATH_CALUDE_distinct_tower_heights_94_bricks_l3999_399943


namespace NUMINAMATH_CALUDE_max_plots_for_given_garden_l3999_399921

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Represents the constraints for partitioning the garden -/
structure PartitionConstraints where
  fencing_available : ℝ
  min_plots_per_row : ℕ

/-- Calculates the maximum number of square plots given garden dimensions and constraints -/
def max_square_plots (garden : GardenDimensions) (constraints : PartitionConstraints) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square plots for the given problem -/
theorem max_plots_for_given_garden :
  let garden := GardenDimensions.mk 30 60
  let constraints := PartitionConstraints.mk 3000 4
  max_square_plots garden constraints = 1250 := by
  sorry

end NUMINAMATH_CALUDE_max_plots_for_given_garden_l3999_399921


namespace NUMINAMATH_CALUDE_product_of_fractions_is_zero_l3999_399902

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions_is_zero :
  (fraction 1) * (fraction 2) * (fraction 3) * (fraction 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_is_zero_l3999_399902


namespace NUMINAMATH_CALUDE_infinite_k_no_prime_sequence_l3999_399970

theorem infinite_k_no_prime_sequence :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ k ∈ S, ∃ (x : ℕ → ℕ),
      x 1 = 1 ∧
      x 2 = k + 2 ∧
      (∀ n, x (n + 2) = (k + 1) * x (n + 1) - x n) ∧
      ∀ n, ¬ Nat.Prime (x n) :=
sorry

end NUMINAMATH_CALUDE_infinite_k_no_prime_sequence_l3999_399970


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3999_399949

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^23 + i^28 + i^33 + i^38 + i^43 = -i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l3999_399949


namespace NUMINAMATH_CALUDE_position_relationships_complete_l3999_399944

-- Define the type for position relationships
inductive PositionRelationship
  | Intersection
  | Parallel
  | Skew

-- Define a type for straight lines in 3D space
structure Line3D where
  -- We don't need to specify the internal structure of Line3D for this statement

-- Define the function that determines the position relationship between two lines
noncomputable def positionRelationship (l1 l2 : Line3D) : PositionRelationship :=
  sorry

-- Theorem statement
theorem position_relationships_complete (l1 l2 : Line3D) :
  ∃ (r : PositionRelationship), positionRelationship l1 l2 = r :=
sorry

end NUMINAMATH_CALUDE_position_relationships_complete_l3999_399944


namespace NUMINAMATH_CALUDE_expression_evaluation_l3999_399972

theorem expression_evaluation : 
  4 * Real.sin (60 * π / 180) - abs (-2) - Real.sqrt 12 + (-1) ^ 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3999_399972


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l3999_399932

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron 
  (V : ℝ) 
  (S₁ S₂ S₃ S₄ : ℝ) 
  (R : ℝ) 
  (h_volume : V > 0)
  (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
  (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l3999_399932


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3999_399937

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.mk 2 a) / (Complex.mk 2 (-1)) = Complex.I * b) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3999_399937


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l3999_399941

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c → a + b + c = 24 → c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l3999_399941


namespace NUMINAMATH_CALUDE_tiles_needed_for_room_main_tiling_theorem_l3999_399986

/-- Represents the tiling pattern where n is the number of days and f(n) is the number of tiles placed on day n. -/
def tilingPattern (n : ℕ) : ℕ := n

/-- Represents the total number of tiles placed after n days. -/
def totalTiles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The surface area of the room in square units. -/
def roomArea : ℕ := 18144

/-- The theorem stating that 2016 tiles are needed to cover the room. -/
theorem tiles_needed_for_room :
  ∃ (sideLength : ℕ), sideLength > 0 ∧ totalTiles 63 = 2016 ∧ 2016 * sideLength^2 = roomArea :=
by
  sorry

/-- The main theorem proving that 2016 tiles are needed and follow the tiling pattern. -/
theorem main_tiling_theorem :
  ∃ (n : ℕ), totalTiles n = 2016 ∧
    (∀ (k : ℕ), k ≤ n → tilingPattern k = k) ∧
    (∃ (sideLength : ℕ), sideLength > 0 ∧ 2016 * sideLength^2 = roomArea) :=
by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_for_room_main_tiling_theorem_l3999_399986


namespace NUMINAMATH_CALUDE_blood_donation_theorem_l3999_399926

/-- Represents the number of people for each blood type -/
structure BloodDonors where
  typeO : Nat
  typeA : Nat
  typeB : Nat
  typeAB : Nat

/-- Calculates the number of ways to select one person to donate blood -/
def selectOneDonor (donors : BloodDonors) : Nat :=
  donors.typeO + donors.typeA + donors.typeB + donors.typeAB

/-- Calculates the number of ways to select one person from each blood type -/
def selectFourDonors (donors : BloodDonors) : Nat :=
  donors.typeO * donors.typeA * donors.typeB * donors.typeAB

/-- Theorem statement for the blood donation problem -/
theorem blood_donation_theorem (donors : BloodDonors) :
  selectOneDonor donors = donors.typeO + donors.typeA + donors.typeB + donors.typeAB ∧
  selectFourDonors donors = donors.typeO * donors.typeA * donors.typeB * donors.typeAB := by
  sorry

/-- Example with the given numbers -/
def example_donors : BloodDonors :=
  { typeO := 28, typeA := 7, typeB := 9, typeAB := 3 }

#eval selectOneDonor example_donors  -- Expected: 47
#eval selectFourDonors example_donors  -- Expected: 5292

end NUMINAMATH_CALUDE_blood_donation_theorem_l3999_399926


namespace NUMINAMATH_CALUDE_rectangle_to_square_length_l3999_399988

theorem rectangle_to_square_length (width : ℝ) (height : ℝ) (y : ℝ) :
  width = 10 →
  height = 20 →
  (width * height = y * y * 16) →
  y = 5 * Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_length_l3999_399988


namespace NUMINAMATH_CALUDE_cubic_inequality_reciprocal_l3999_399925

theorem cubic_inequality_reciprocal (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b > 0) :
  1 / a < 1 / b := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_reciprocal_l3999_399925


namespace NUMINAMATH_CALUDE_sine_shift_left_l3999_399959

/-- Shifting a sine function to the left --/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := Real.sin (t + π / 6)
  ∀ y : ℝ, f (x + π / 6) = g x :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_l3999_399959


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3999_399992

/-- The surface area of a cylinder with a square cross-section of side length 2 is 6π. -/
theorem cylinder_surface_area (π : ℝ) (h : π = Real.pi) : 
  let side_length : ℝ := 2
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let lateral_area : ℝ := 2 * π * radius * height
  let base_area : ℝ := 2 * π * radius^2
  lateral_area + base_area = 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3999_399992


namespace NUMINAMATH_CALUDE_yearly_savings_ratio_l3999_399956

-- Define the fraction of salary spent each month
def fraction_spent : ℚ := 0.6666666666666667

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem yearly_savings_ratio :
  (1 - fraction_spent) * months_in_year = 4 := by
  sorry

end NUMINAMATH_CALUDE_yearly_savings_ratio_l3999_399956


namespace NUMINAMATH_CALUDE_students_dislike_both_l3999_399947

/-- Given a class of students and their food preferences, calculate the number of students who don't like either food. -/
theorem students_dislike_both (total : ℕ) (like_fries : ℕ) (like_burgers : ℕ) (like_both : ℕ) 
  (h1 : total = 25)
  (h2 : like_fries = 15)
  (h3 : like_burgers = 10)
  (h4 : like_both = 6)
  (h5 : like_both ≤ like_fries ∧ like_both ≤ like_burgers) :
  total - (like_fries + like_burgers - like_both) = 6 := by
  sorry

#check students_dislike_both

end NUMINAMATH_CALUDE_students_dislike_both_l3999_399947


namespace NUMINAMATH_CALUDE_line_slope_l3999_399965

/-- The slope of the line 4x - 7y = 28 is 4/7 -/
theorem line_slope (x y : ℝ) : 4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3999_399965


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3999_399906

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 1 where f(2012) = 3,
    prove that f(-2012) = -1 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := fun x => a * x^5 + b * x^3 + c * x + 1
  (f 2012 = 3) → (f (-2012) = -1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3999_399906


namespace NUMINAMATH_CALUDE_factor_expression_l3999_399985

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3999_399985


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3999_399961

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_eq : chord1 = 40
  /-- The second chord has length 40 -/
  chord2_eq : chord2 = 40
  /-- The third chord has length 30 -/
  chord3_eq : chord3 = 30

/-- The theorem stating that the distance between adjacent parallel lines is 20√6 -/
theorem parallel_lines_distance (c : ParallelLinesCircle) : c.d = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3999_399961


namespace NUMINAMATH_CALUDE_picnic_attendance_l3999_399905

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance (p : PicnicAttendance) 
  (h1 : p.men = p.women + 40)
  (h2 : p.adults = p.children + 40)
  (h3 : p.men = 90)
  : p.men + p.women + p.children = 240 := by
  sorry

#check picnic_attendance

end NUMINAMATH_CALUDE_picnic_attendance_l3999_399905


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3999_399933

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3999_399933


namespace NUMINAMATH_CALUDE_power_function_value_l3999_399953

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = Real.sqrt 2 / 2) :
  f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l3999_399953


namespace NUMINAMATH_CALUDE_triangle_inequality_l3999_399983

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤ 1 + (1 / 2) * (Real.cos ((A - B) / 4))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3999_399983


namespace NUMINAMATH_CALUDE_course_size_l3999_399908

theorem course_size (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (3 : ℚ) / 10 * total + (2 : ℚ) / 10 * total + 
        (1 : ℚ) / 10 * total + 12 + 5 = total) : 
  total = 170 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l3999_399908


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3999_399917

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) → 
  x ≥ (5 - Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3999_399917


namespace NUMINAMATH_CALUDE_inequality_and_range_l3999_399914

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 2|

-- Define the theorem
theorem inequality_and_range :
  -- Part I: Solution set of f(x) < 4 - |x-1|
  (∀ x : ℝ, f x < 4 - |x - 1| ↔ x > -5/4 ∧ x < 1/2) ∧
  -- Part II: Range of a
  (∀ m n a : ℝ, m > 0 → n > 0 → m + n = 1 → a > 0 →
    (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
    a ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_range_l3999_399914


namespace NUMINAMATH_CALUDE_expand_expression_l3999_399995

theorem expand_expression (x y : ℝ) : 
  (6*x + 8 - 3*y) * (4*x - 5*y) = 24*x^2 - 42*x*y + 32*x - 40*y + 15*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3999_399995


namespace NUMINAMATH_CALUDE_semicircle_radius_in_specific_triangle_l3999_399979

/-- An isosceles triangle with a semicircle inscribed on its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base
  /-- The radius plus the height of the triangle equals the length of the equal sides -/
  radius_plus_height_eq_side : radius + height = Real.sqrt ((base / 2) ^ 2 + height ^ 2)

/-- The radius of the inscribed semicircle in the specific isosceles triangle -/
theorem semicircle_radius_in_specific_triangle :
  ∃ (t : IsoscelesTriangleWithSemicircle), t.base = 20 ∧ t.height = 12 ∧ t.radius = 12 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_specific_triangle_l3999_399979


namespace NUMINAMATH_CALUDE_not_always_increasing_sum_of_increasing_and_decreasing_l3999_399912

-- Define the concept of an increasing function
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a decreasing function
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem not_always_increasing_sum_of_increasing_and_decreasing :
  ¬(∀ f g : ℝ → ℝ, Increasing f → Decreasing g → Increasing (λ x ↦ f x + g x)) :=
sorry

end NUMINAMATH_CALUDE_not_always_increasing_sum_of_increasing_and_decreasing_l3999_399912


namespace NUMINAMATH_CALUDE_factorization_theorem_l3999_399977

theorem factorization_theorem (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3999_399977


namespace NUMINAMATH_CALUDE_shop_length_calculation_l3999_399916

/-- Given a shop with specified rent and dimensions, calculate its length -/
theorem shop_length_calculation (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) :
  monthly_rent = 2400 →
  width = 8 →
  annual_rent_per_sqft = 360 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shop_length_calculation_l3999_399916


namespace NUMINAMATH_CALUDE_sheep_problem_l3999_399920

theorem sheep_problem (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - difference - mary_initial)) = 566 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_problem_l3999_399920


namespace NUMINAMATH_CALUDE_current_calculation_l3999_399984

/-- Given complex numbers V₁, V₂, Z, V, and I, prove that I = -1 + i -/
theorem current_calculation (V₁ V₂ Z V I : ℂ) 
  (h1 : V₁ = 2 + I)
  (h2 : V₂ = -1 + 4*I)
  (h3 : Z = 2 + 2*I)
  (h4 : V = V₁ + V₂)
  (h5 : I = V / Z) :
  I = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_current_calculation_l3999_399984


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l3999_399924

theorem radical_product_equals_27 :
  let a := 81
  let b := 27
  let c := 9
  (a = 3^4) → (b = 3^3) → (c = 3^2) →
  (a^(1/4) * b^(1/3) * c^(1/2) : ℝ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l3999_399924


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3999_399960

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3 - 2^3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3999_399960


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3999_399989

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3999_399989


namespace NUMINAMATH_CALUDE_colored_cells_count_l3999_399974

theorem colored_cells_count (k l : ℕ) : 
  k * l = 74 → 
  (∃ (rows cols : ℕ), 
    rows = 2 * k + 1 ∧ 
    cols = 2 * l + 1 ∧ 
    (rows * cols - 74 = 301 ∨ rows * cols - 74 = 373)) := by
  sorry

end NUMINAMATH_CALUDE_colored_cells_count_l3999_399974


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3999_399915

theorem quadratic_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3999_399915


namespace NUMINAMATH_CALUDE_finite_decimal_consecutive_denominators_l3999_399963

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℤ) (k : ℕ), q = a / (b * 10^k) ∧ b ≠ 0

theorem finite_decimal_consecutive_denominators :
  ∀ n : ℕ, (is_finite_decimal (1 / n) ∧ is_finite_decimal (1 / (n + 1))) ↔ (n = 1 ∨ n = 4) :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_consecutive_denominators_l3999_399963


namespace NUMINAMATH_CALUDE_coconut_trips_l3999_399996

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l3999_399996


namespace NUMINAMATH_CALUDE_circle_equation_l3999_399904

/-- Given a circle C with radius 3 and center symmetric to (1,0) about y=x,
    prove that its standard equation is x^2 + (y-1)^2 = 9 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) →
  (center.1, center.2) = (0, 1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 1)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3999_399904


namespace NUMINAMATH_CALUDE_janice_stair_climb_l3999_399939

/-- The number of times Janice goes up the stairs in a day. -/
def times_up : ℕ := 5

/-- The number of flights of stairs for each trip up. -/
def flights_per_trip : ℕ := 3

/-- The number of times Janice goes down the stairs in a day. -/
def times_down : ℕ := 3

/-- The total number of flights walked (up and down) in a day. -/
def total_flights : ℕ := 24

theorem janice_stair_climb :
  times_up * flights_per_trip + times_down * flights_per_trip = total_flights :=
by sorry

end NUMINAMATH_CALUDE_janice_stair_climb_l3999_399939


namespace NUMINAMATH_CALUDE_custom_mult_example_l3999_399907

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

/-- Theorem stating that (6/5) * (3/4) = 144/5 under the custom multiplication -/
theorem custom_mult_example : custom_mult 6 5 3 4 = 144 / 5 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l3999_399907


namespace NUMINAMATH_CALUDE_volume_of_S_l3999_399973

-- Define the solid S' in the first octant
def S' : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
                   x + 2*y ≤ 1 ∧ 2*x + z ≤ 1 ∧ y + 2*z ≤ 1}

-- State the theorem about the volume of S'
theorem volume_of_S' : MeasureTheory.volume S' = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_S_l3999_399973


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3999_399938

theorem sine_cosine_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 6 + Real.cos x ^ 6 + 2 * a * Real.sin x * Real.cos x ≥ 0) ↔ 
  |a| ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3999_399938


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l3999_399997

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 :
  binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l3999_399997


namespace NUMINAMATH_CALUDE_school_population_after_additions_l3999_399911

theorem school_population_after_additions 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (initial_teachers : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (additional_teachers : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : initial_boys = 761) 
  (h3 : initial_teachers = 54) 
  (h4 : additional_girls = 682) 
  (h5 : additional_boys = 8) 
  (h6 : additional_teachers = 3) : 
  initial_girls + initial_boys + initial_teachers + 
  additional_girls + additional_boys + additional_teachers = 2240 :=
by
  sorry


end NUMINAMATH_CALUDE_school_population_after_additions_l3999_399911


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l3999_399971

theorem polar_to_rectangular_transformation (x y : ℝ) (r θ : ℝ) 
  (h1 : x = 12 ∧ y = 5)
  (h2 : r = (x^2 + y^2).sqrt)
  (h3 : θ = Real.arctan (y / x)) :
  let new_r := 2 * r^2
  let new_θ := 3 * θ
  (new_r * Real.cos new_θ = 338 * 828 / 2197) ∧
  (new_r * Real.sin new_θ = 338 * 2035 / 2197) := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l3999_399971


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3999_399990

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3999_399990


namespace NUMINAMATH_CALUDE_vectors_in_plane_implies_x_eq_neg_one_l3999_399951

-- Define the vectors
def a (x : ℝ) : Fin 3 → ℝ := ![1, x, -2]
def b : Fin 3 → ℝ := ![0, 1, 2]
def c : Fin 3 → ℝ := ![1, 0, 0]

-- Define the condition that vectors lie in the same plane
def vectors_in_same_plane (x : ℝ) : Prop :=
  ∃ (m n : ℝ), a x = m • b + n • c

-- Theorem statement
theorem vectors_in_plane_implies_x_eq_neg_one :
  ∀ x : ℝ, vectors_in_same_plane x → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_vectors_in_plane_implies_x_eq_neg_one_l3999_399951


namespace NUMINAMATH_CALUDE_equation_solution_l3999_399918

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 ∧ x₂ = (-1 - Real.sqrt 17) / 2) ∧
  (∀ x : ℝ, x^2 - |x - 1| - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3999_399918


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l3999_399981

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- Represents the two parts of a triangle after division along a median -/
structure TriangleParts where
  part1 : Triangle
  part2 : Triangle

theorem triangle_division_theorem (t1 t2 t3 : Triangle) 
  (h_identical : t1 = t2 ∧ t2 = t3) :
  ∃ (p1 p2 p3 : TriangleParts) (result : Triangle),
    (p1.part1.a = t1.a ∧ p1.part1.b = t1.b) ∧
    (p2.part1.a = t2.a ∧ p2.part1.b = t2.b) ∧
    (p3.part1.a = t3.a ∧ p3.part1.b = t3.b) ∧
    (p1.part1.α + p2.part1.α + p3.part1.α = 2 * π) ∧
    (result.a = t1.a ∧ result.b = t1.b ∧ result.c = t1.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l3999_399981


namespace NUMINAMATH_CALUDE_history_or_geography_not_both_count_l3999_399903

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The number of students taking history -/
def history : ℕ := 30

/-- The number of students taking geography only -/
def geography_only : ℕ := 12

/-- The number of students taking history or geography but not both -/
def history_or_geography_not_both : ℕ := (history - both) + geography_only

theorem history_or_geography_not_both_count : history_or_geography_not_both = 27 := by
  sorry

end NUMINAMATH_CALUDE_history_or_geography_not_both_count_l3999_399903


namespace NUMINAMATH_CALUDE_square_tiles_count_l3999_399980

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
  | 0 => 3  -- triangular
  | 1 => 4  -- square
  | 2 => 5  -- pentagonal
  | _ => 0  -- should never happen

/-- The proposition that given the conditions, there are 10 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 120) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3*t + 4*s + 5*p = total_edges ∧
    s = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3999_399980


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3999_399934

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Checks if the probabilities of the five specified events are equal -/
def equalProbabilities (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  let y := mc.yellow
  Nat.choose r 5 = w * Nat.choose r 4 ∧
  Nat.choose r 5 = w * b * Nat.choose r 3 ∧
  Nat.choose r 5 = w * b * g * Nat.choose r 2 ∧
  Nat.choose r 5 = w * b * g * y * r

/-- Theorem stating that the smallest number of marbles satisfying the conditions is 13 -/
theorem smallest_marble_count :
  ∃ (mc : MarbleCount), totalMarbles mc = 13 ∧ equalProbabilities mc ∧
  (∀ (mc' : MarbleCount), equalProbabilities mc' → totalMarbles mc' ≥ 13) := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3999_399934


namespace NUMINAMATH_CALUDE_crayon_count_initial_crayon_count_l3999_399999

theorem crayon_count (crayons_taken : ℕ) (crayons_left : ℕ) : ℕ :=
  crayons_taken + crayons_left

theorem initial_crayon_count : crayon_count 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_initial_crayon_count_l3999_399999


namespace NUMINAMATH_CALUDE_seventeen_above_zero_l3999_399993

/-- Represents temperature in degrees Celsius -/
structure Temperature where
  value : ℝ
  unit : String
  is_celsius : unit = "°C"

/-- The zero point of the Celsius scale -/
def celsius_zero : Temperature := ⟨10, "°C", rfl⟩

/-- The temperature to be compared -/
def temp_to_compare : Temperature := ⟨17, "°C", rfl⟩

/-- Theorem stating that 17°C represents a temperature above zero degrees Celsius -/
theorem seventeen_above_zero :
  temp_to_compare.value > celsius_zero.value → 
  ∃ (t : ℝ), t > 0 ∧ temp_to_compare.value = t :=
by sorry

end NUMINAMATH_CALUDE_seventeen_above_zero_l3999_399993
