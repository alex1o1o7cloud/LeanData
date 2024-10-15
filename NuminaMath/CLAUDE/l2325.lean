import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2325_232528

/-- For a quadratic equation x^2 - 2x + m = 0 to have real roots, m must be less than or equal to 1 -/
theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2325_232528


namespace NUMINAMATH_CALUDE_solve_oliver_money_problem_l2325_232526

def oliver_money_problem (initial_amount savings puzzle_cost gift final_amount : ℕ) 
  (frisbee_cost : ℕ) : Prop :=
  initial_amount + savings + gift - puzzle_cost - frisbee_cost = final_amount

theorem solve_oliver_money_problem :
  ∃ (frisbee_cost : ℕ), oliver_money_problem 9 5 3 8 15 frisbee_cost ∧ frisbee_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_oliver_money_problem_l2325_232526


namespace NUMINAMATH_CALUDE_triangle_area_part1_triangle_side_part2_l2325_232555

-- Part 1
theorem triangle_area_part1 (A B C : ℝ) (a b c : ℝ) :
  A = π/6 → C = π/4 → a = 2 →
  (1/2) * a * b * Real.sin C = 1 + Real.sqrt 3 :=
sorry

-- Part 2
theorem triangle_side_part2 (A B C : ℝ) (a b c : ℝ) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 → b = 2 → C = π/3 →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_part1_triangle_side_part2_l2325_232555


namespace NUMINAMATH_CALUDE_ray_AB_bisects_angle_PAQ_l2325_232533

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points T, A, and B
def point_T : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) : Prop :=
  ∃ k, y = k * x + 1

-- Define points P and Q as intersections of line l and the ellipse
def point_P : ℝ × ℝ := sorry
def point_Q : ℝ × ℝ := sorry

-- State the theorem
theorem ray_AB_bisects_angle_PAQ :
  circle_C point_T.1 point_T.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  point_A.2 > point_B.2 ∧
  point_A.2 - point_B.2 = 3 ∧
  line_l point_P.1 point_P.2 ∧
  line_l point_Q.1 point_Q.2 ∧
  ellipse point_P.1 point_P.2 ∧
  ellipse point_Q.1 point_Q.2 →
  -- The conclusion that ray AB bisects angle PAQ
  -- This would typically involve showing that the angles are equal
  -- or that the dot product of vectors is zero, but we'll leave it as 'sorry'
  sorry :=
sorry

end NUMINAMATH_CALUDE_ray_AB_bisects_angle_PAQ_l2325_232533


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2325_232557

/-- Represents a cone with given properties -/
structure Cone where
  surfaceArea : ℝ
  lateralSurfaceIsSemicircle : Prop

/-- Theorem stating that a cone with surface area 3π and lateral surface unfolding 
    into a semicircle has a base diameter of √6 -/
theorem cone_base_diameter (c : Cone) 
    (h1 : c.surfaceArea = 3 * Real.pi)
    (h2 : c.lateralSurfaceIsSemicircle) : 
    ∃ (d : ℝ), d = Real.sqrt 6 ∧ d = 2 * (Real.sqrt ((3 : ℝ) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2325_232557


namespace NUMINAMATH_CALUDE_total_air_removed_l2325_232547

def air_removal_fractions : List Rat := [1/3, 1/4, 1/5, 1/6, 1/7]

def remaining_air (fractions : List Rat) : Rat :=
  fractions.foldl (fun acc f => acc * (1 - f)) 1

theorem total_air_removed (fractions : List Rat) :
  fractions = air_removal_fractions →
  1 - remaining_air fractions = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_total_air_removed_l2325_232547


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2325_232567

def f (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2325_232567


namespace NUMINAMATH_CALUDE_power_multiplication_l2325_232534

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2325_232534


namespace NUMINAMATH_CALUDE_car_journey_theorem_l2325_232538

theorem car_journey_theorem (local_distance : ℝ) (local_speed : ℝ) (highway_speed : ℝ) (average_speed : ℝ) (highway_distance : ℝ) :
  local_distance = 60 ∧
  local_speed = 20 ∧
  highway_speed = 60 ∧
  average_speed = 36 ∧
  average_speed = (local_distance + highway_distance) / (local_distance / local_speed + highway_distance / highway_speed) →
  highway_distance = 120 := by
sorry

end NUMINAMATH_CALUDE_car_journey_theorem_l2325_232538


namespace NUMINAMATH_CALUDE_money_distribution_l2325_232569

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 320 → 
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2325_232569


namespace NUMINAMATH_CALUDE_right_triangle_area_l2325_232504

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 12 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 18 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2325_232504


namespace NUMINAMATH_CALUDE_triangle_side_length_l2325_232549

theorem triangle_side_length (PQ PR PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 3.5) :
  ∃ QR : ℝ, QR = 9 ∧ PM^2 = (1/2) * (PQ^2 + PR^2 + QR^2) - (1/4) * QR^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2325_232549


namespace NUMINAMATH_CALUDE_binomial_11_choose_9_l2325_232587

theorem binomial_11_choose_9 : Nat.choose 11 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_choose_9_l2325_232587


namespace NUMINAMATH_CALUDE_amy_total_crumbs_l2325_232565

/-- Theorem: Amy's total crumbs given Arthur's total crumbs -/
theorem amy_total_crumbs (c : ℝ) : ℝ := by
  -- Define Arthur's trips and crumbs per trip
  let arthur_trips : ℝ := c / (c / c)
  let arthur_crumbs_per_trip : ℝ := c / arthur_trips

  -- Define Amy's trips and crumbs per trip
  let amy_trips : ℝ := 2 * arthur_trips
  let amy_crumbs_per_trip : ℝ := 1.5 * arthur_crumbs_per_trip

  -- Calculate Amy's total crumbs
  let amy_total : ℝ := amy_trips * amy_crumbs_per_trip

  -- Prove that Amy's total crumbs equals 3c
  sorry

end NUMINAMATH_CALUDE_amy_total_crumbs_l2325_232565


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2325_232511

/-- The number of fish tagged on April 1 -/
def tagged_april : ℕ := 120

/-- The number of fish captured on August 1 -/
def captured_august : ℕ := 150

/-- The number of tagged fish found in the August 1 sample -/
def tagged_in_august : ℕ := 5

/-- The proportion of fish that left the pond between April 1 and August 1 -/
def left_pond : ℚ := 3/10

/-- The proportion of fish in the August sample that were not in the pond in April -/
def new_fish_proportion : ℚ := 1/2

/-- The estimated number of fish in the pond on April 1 -/
def fish_population : ℕ := 1800

/-- Theorem stating that given the conditions, the fish population on April 1 was 1800 -/
theorem fish_population_estimate :
  tagged_april = 120 →
  captured_august = 150 →
  tagged_in_august = 5 →
  left_pond = 3/10 →
  new_fish_proportion = 1/2 →
  fish_population = 1800 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l2325_232511


namespace NUMINAMATH_CALUDE_clothing_price_comparison_l2325_232507

theorem clothing_price_comparison (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 120 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  original_price * (1 + increase_rate) * (1 - discount_rate) < original_price :=
by sorry

end NUMINAMATH_CALUDE_clothing_price_comparison_l2325_232507


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_243_l2325_232571

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 143*x - 210

-- Define the roots of the polynomial
variables (p q r : ℝ)

-- State that p, q, r are the roots of f
axiom roots_of_f : f p = 0 ∧ f q = 0 ∧ f r = 0

-- Define A, B, C as real numbers
variables (A B C : ℝ)

-- State the partial fraction decomposition
axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 24*s^2 + 143*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)

-- The theorem to prove
theorem sum_of_reciprocals_equals_243 :
  1 / A + 1 / B + 1 / C = 243 :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_243_l2325_232571


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l2325_232514

/-- The number of different four-digit numbers that can be formed using two 1s, one 2, and one 0 -/
def four_digit_numbers : ℕ :=
  let zero_placements := 3  -- 0 can be placed in hundreds, tens, or ones place
  let two_placements := 3   -- 2 can be placed in any of the remaining 3 positions
  zero_placements * two_placements

/-- Proof that the number of different four-digit numbers formed is 9 -/
theorem four_digit_numbers_count : four_digit_numbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l2325_232514


namespace NUMINAMATH_CALUDE_later_purchase_cost_l2325_232594

/-- The cost of a single bat in dollars -/
def bat_cost : ℕ := 500

/-- The cost of a single ball in dollars -/
def ball_cost : ℕ := 100

/-- The number of bats in the later purchase -/
def num_bats : ℕ := 3

/-- The number of balls in the later purchase -/
def num_balls : ℕ := 5

/-- The total cost of the later purchase -/
def total_cost : ℕ := num_bats * bat_cost + num_balls * ball_cost

theorem later_purchase_cost : total_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_later_purchase_cost_l2325_232594


namespace NUMINAMATH_CALUDE_gold_calculation_l2325_232572

-- Define the amount of gold Greg has
def gregs_gold : ℕ := 20

-- Define Katie's gold in terms of Greg's
def katies_gold : ℕ := 4 * gregs_gold

-- Define the total amount of gold
def total_gold : ℕ := gregs_gold + katies_gold

-- Theorem to prove
theorem gold_calculation : total_gold = 100 := by
  sorry

end NUMINAMATH_CALUDE_gold_calculation_l2325_232572


namespace NUMINAMATH_CALUDE_travel_time_difference_l2325_232543

/-- Proves the equation for the travel time difference between two groups -/
theorem travel_time_difference 
  (x : ℝ) -- walking speed in km/h
  (h1 : x > 0) -- walking speed is positive
  (distance : ℝ) -- distance traveled
  (h2 : distance = 4) -- distance is 4 km
  (time_diff : ℝ) -- time difference in hours
  (h3 : time_diff = 1/3) -- time difference is 1/3 hours
  : 
  distance / x - distance / (2 * x) = time_diff :=
by sorry

end NUMINAMATH_CALUDE_travel_time_difference_l2325_232543


namespace NUMINAMATH_CALUDE_flower_combinations_l2325_232523

/-- The number of valid combinations of roses and carnations -/
def valid_combinations : ℕ := sorry

/-- Predicate for valid combination of roses and carnations -/
def is_valid_combination (r c : ℕ) : Prop :=
  3 * r + 2 * c = 70 ∧ r + c ≥ 20

theorem flower_combinations :
  valid_combinations = 12 ∨
  valid_combinations = 13 ∨
  valid_combinations = 15 ∨
  valid_combinations = 17 ∨
  valid_combinations = 18 :=
sorry

end NUMINAMATH_CALUDE_flower_combinations_l2325_232523


namespace NUMINAMATH_CALUDE_page_number_added_twice_l2325_232560

theorem page_number_added_twice (n : ℕ) (x : ℕ) 
  (h1 : x ≤ n) 
  (h2 : n * (n + 1) / 2 + x = 3050) : 
  x = 47 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l2325_232560


namespace NUMINAMATH_CALUDE_total_pictures_uploaded_l2325_232529

/-- Proves that the total number of pictures uploaded is 25 -/
theorem total_pictures_uploaded (first_album : ℕ) (num_other_albums : ℕ) (pics_per_other_album : ℕ) 
  (h1 : first_album = 10)
  (h2 : num_other_albums = 5)
  (h3 : pics_per_other_album = 3) :
  first_album + num_other_albums * pics_per_other_album = 25 := by
  sorry

#check total_pictures_uploaded

end NUMINAMATH_CALUDE_total_pictures_uploaded_l2325_232529


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2325_232563

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 is √3/2 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2325_232563


namespace NUMINAMATH_CALUDE_corrected_mean_l2325_232537

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 20 ∧ original_mean = 36 ∧ incorrect_value = 40 ∧ correct_value = 25 →
  (n * original_mean - (incorrect_value - correct_value)) / n = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l2325_232537


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2325_232579

theorem fraction_to_decimal : (11 : ℚ) / 16 = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2325_232579


namespace NUMINAMATH_CALUDE_cubic_tangent_ratio_l2325_232593

-- Define the cubic function
def cubic (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the points A, T, B on the x-axis
structure RootPoints where
  α : ℝ
  γ : ℝ
  β : ℝ

-- Define the theorem
theorem cubic_tangent_ratio 
  (a b c : ℝ) 
  (roots : RootPoints) 
  (h1 : cubic a b c roots.α = 0)
  (h2 : cubic a b c roots.γ = 0)
  (h3 : cubic a b c roots.β = 0)
  (h4 : roots.α < roots.γ)
  (h5 : roots.γ < roots.β) :
  (roots.β - roots.α) / ((roots.α + roots.γ)/2 - (roots.β + roots.γ)/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_ratio_l2325_232593


namespace NUMINAMATH_CALUDE_no_max_cos_squared_sum_l2325_232515

theorem no_max_cos_squared_sum (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  ∃ d > 0, B - A = d ∧ C - B = d →  -- Arithmetic sequence with positive difference
  ¬ ∃ M : ℝ, ∀ A' B' C' : ℝ,
    (0 < A' ∧ 0 < B' ∧ 0 < C' ∧
     A' + B' + C' = π ∧
     ∃ d > 0, B' - A' = d ∧ C' - B' = d) →
    Real.cos A' ^ 2 + Real.cos C' ^ 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_max_cos_squared_sum_l2325_232515


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2325_232508

open Real

theorem trigonometric_identity :
  sin (12 * π / 180) * cos (36 * π / 180) * sin (48 * π / 180) * cos (72 * π / 180) * tan (18 * π / 180) =
  1/2 * (sin (12 * π / 180)^2 + sin (12 * π / 180) * cos (6 * π / 180)) * sin (18 * π / 180)^2 / cos (18 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2325_232508


namespace NUMINAMATH_CALUDE_max_a_value_l2325_232568

theorem max_a_value (a : ℤ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2))) →
  (∃ y : ℝ, y ≥ 0 ∧ (y + a)/(y - 1) + 2*a/(1 - y) = 2) →
  (∀ a' : ℤ, 
    (∃ (x₁' x₂' x₃' : ℤ), 
      (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2) ↔ (x = x₁' ∨ x = x₂' ∨ x = x₃')) ∧
      (∀ x : ℤ, x ≠ x₁' ∧ x ≠ x₂' ∧ x ≠ x₃' → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2))) →
    (∃ y : ℝ, y ≥ 0 ∧ (y + a')/(y - 1) + 2*a'/(1 - y) = 2) →
    a' ≤ a) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_max_a_value_l2325_232568


namespace NUMINAMATH_CALUDE_hania_age_in_5_years_l2325_232599

-- Define the current year as a reference point
def current_year : ℕ := 2023

-- Define Samir's age in 5 years
def samir_age_in_5_years : ℕ := 20

-- Define the relationship between Samir's current age and Hania's age 10 years ago
axiom samir_hania_age_relation : 
  ∃ (samir_current_age hania_age_10_years_ago : ℕ),
    samir_current_age = samir_age_in_5_years - 5 ∧
    samir_current_age = hania_age_10_years_ago / 2

-- Theorem to prove
theorem hania_age_in_5_years : 
  ∃ (hania_current_age : ℕ),
    hania_current_age + 5 = 45 :=
sorry

end NUMINAMATH_CALUDE_hania_age_in_5_years_l2325_232599


namespace NUMINAMATH_CALUDE_system_solution_condition_l2325_232531

/-- The system of equations has a solution for any a if and only if 0 ≤ b ≤ 2. -/
theorem system_solution_condition (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2325_232531


namespace NUMINAMATH_CALUDE_systematic_sample_count_in_range_l2325_232518

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (total / sampleSize)) % total + 1)

/-- Count numbers in a given range -/
def countInRange (list : List ℕ) (low : ℕ) (high : ℕ) : ℕ :=
  list.filter (fun n => low ≤ n && n ≤ high) |>.length

theorem systematic_sample_count_in_range :
  let total := 840
  let sampleSize := 42
  let start := 13
  let sample := systematicSample total sampleSize start
  countInRange sample 490 700 = 11 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_count_in_range_l2325_232518


namespace NUMINAMATH_CALUDE_rectangle_perimeter_13km_l2325_232524

/-- The perimeter of a rectangle with both sides equal to 13 km is 52 km. -/
theorem rectangle_perimeter_13km (l w : ℝ) : 
  l = 13 → w = 13 → 2 * (l + w) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_13km_l2325_232524


namespace NUMINAMATH_CALUDE_parabola_area_theorem_l2325_232513

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Theorem: For a parabola y^2 = px with p > 0, focus F on the x-axis, and a slanted line through F
    intersecting the parabola at A and B, if the area of triangle OAB is 2√2 (where O is the origin),
    then p = 4√2. -/
theorem parabola_area_theorem (par : Parabola) (F A B : Point) :
  F.x = par.p / 2 →  -- Focus F is on x-axis
  F.y = 0 →
  (∃ m b : ℝ, A.y = m * A.x + b ∧ B.y = m * B.x + b ∧ F.y = m * F.x + b) →  -- A, B, F are on a slanted line
  A.y^2 = par.p * A.x →  -- A is on the parabola
  B.y^2 = par.p * B.x →  -- B is on the parabola
  abs ((A.x * B.y - B.x * A.y) / 2) = 2 * Real.sqrt 2 →  -- Area of triangle OAB is 2√2
  par.p = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_area_theorem_l2325_232513


namespace NUMINAMATH_CALUDE_kyle_stars_theorem_l2325_232544

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) : ℕ :=
  (initial_bottles + additional_bottles) * stars_per_bottle

/-- Theorem stating the total number of stars Kyle needs to make -/
theorem kyle_stars_theorem :
  total_stars 2 3 15 = 75 := by
  sorry

end NUMINAMATH_CALUDE_kyle_stars_theorem_l2325_232544


namespace NUMINAMATH_CALUDE_adjustment_ways_l2325_232502

def front_row : ℕ := 4
def back_row : ℕ := 8
def students_to_move : ℕ := 2

def ways_to_select : ℕ := Nat.choose back_row students_to_move
def ways_to_insert : ℕ := Nat.factorial (front_row + students_to_move) / Nat.factorial front_row

theorem adjustment_ways : 
  ways_to_select * ways_to_insert = 840 := by sorry

end NUMINAMATH_CALUDE_adjustment_ways_l2325_232502


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2325_232550

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (x : ℕ) (hx : x > 0) : 
  (∀ n : ℕ, n < 5064 → ¬(is_divisible_by (n - 24) x ∧ 
                         is_divisible_by (n - 24) 10 ∧ 
                         is_divisible_by (n - 24) 15 ∧ 
                         is_divisible_by (n - 24) 20 ∧ 
                         (n - 24) / x = 84 ∧ 
                         (n - 24) / 10 = 84 ∧ 
                         (n - 24) / 15 = 84 ∧ 
                         (n - 24) / 20 = 84)) ∧
  (is_divisible_by (5064 - 24) x ∧ 
   is_divisible_by (5064 - 24) 10 ∧ 
   is_divisible_by (5064 - 24) 15 ∧ 
   is_divisible_by (5064 - 24) 20 ∧ 
   (5064 - 24) / x = 84 ∧ 
   (5064 - 24) / 10 = 84 ∧ 
   (5064 - 24) / 15 = 84 ∧ 
   (5064 - 24) / 20 = 84) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2325_232550


namespace NUMINAMATH_CALUDE_expected_heads_is_56_l2325_232598

/-- The number of fair coins --/
def n : ℕ := 90

/-- The probability of getting heads on a single fair coin toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting tails followed by two consecutive heads --/
def p_tails_then_heads : ℚ := 1/2 * 1/4

/-- The total probability of a coin showing heads under the given rules --/
def p_total : ℚ := p_heads + p_tails_then_heads

/-- The expected number of coins showing heads --/
def expected_heads : ℚ := n * p_total

theorem expected_heads_is_56 : expected_heads = 56 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_56_l2325_232598


namespace NUMINAMATH_CALUDE_remainder_problem_l2325_232521

theorem remainder_problem (divisor remainder_1657 : ℕ) 
  (h1 : divisor = 127)
  (h2 : remainder_1657 = 6)
  (h3 : ∃ k : ℕ, 1657 = k * divisor + remainder_1657)
  (h4 : ∃ m r : ℕ, 2037 = m * divisor + r ∧ r < divisor)
  (h5 : ∀ d : ℕ, d > divisor → ¬(∃ k1 k2 r1 r2 : ℕ, 1657 = k1 * d + r1 ∧ 2037 = k2 * d + r2 ∧ r1 < d ∧ r2 < d)) :
  ∃ m : ℕ, 2037 = m * divisor + 5 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2325_232521


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2325_232578

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  24 * a^3 - 36 * a^2 + 16 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 16 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 16 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2325_232578


namespace NUMINAMATH_CALUDE_eggs_in_fridge_l2325_232510

/-- Given a chef with eggs and cake-making information, calculate the number of eggs left in the fridge. -/
theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (cakes_made : ℕ) : 
  total_eggs = 60 → eggs_per_cake = 5 → cakes_made = 10 → 
  total_eggs - (eggs_per_cake * cakes_made) = 10 := by
  sorry

#check eggs_in_fridge

end NUMINAMATH_CALUDE_eggs_in_fridge_l2325_232510


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_A_subset_complement_B_iff_l2325_232585

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Part 1
theorem union_A_B_when_a_is_one : 
  A ∪ B 1 = {x : ℝ | 0 < x ∧ x < 5} := by sorry

-- Part 2
theorem A_subset_complement_B_iff : 
  ∀ a : ℝ, A ⊆ (Set.univ \ B a) ↔ a ≤ 0 ∨ a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_A_subset_complement_B_iff_l2325_232585


namespace NUMINAMATH_CALUDE_library_books_count_l2325_232574

theorem library_books_count :
  ∀ (n : ℕ),
    500 < n ∧ n < 650 ∧
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 →
    n = 595 :=
by sorry

end NUMINAMATH_CALUDE_library_books_count_l2325_232574


namespace NUMINAMATH_CALUDE_inequality_cubic_quadratic_l2325_232519

theorem inequality_cubic_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end NUMINAMATH_CALUDE_inequality_cubic_quadratic_l2325_232519


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2325_232570

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2325_232570


namespace NUMINAMATH_CALUDE_correct_operation_l2325_232542

theorem correct_operation (m : ℝ) : (2 * m^3)^2 / (2 * m)^2 = m^4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2325_232542


namespace NUMINAMATH_CALUDE_ford_younger_than_christopher_l2325_232509

/-- Proves that Ford is 2 years younger than Christopher given the conditions of the problem -/
theorem ford_younger_than_christopher :
  ∀ (george christopher ford : ℕ),
    george = christopher + 8 →
    george + christopher + ford = 60 →
    christopher = 18 →
    ∃ (y : ℕ), ford = christopher - y ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_ford_younger_than_christopher_l2325_232509


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2325_232562

def U : Set ℝ := Set.univ
def A : Set ℝ := {-3, -2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {-3, -2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2325_232562


namespace NUMINAMATH_CALUDE_positive_integer_sum_with_square_twelve_l2325_232540

theorem positive_integer_sum_with_square_twelve (M : ℕ+) :
  (M : ℝ)^2 + M = 12 → M = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_sum_with_square_twelve_l2325_232540


namespace NUMINAMATH_CALUDE_boots_sold_on_monday_l2325_232582

/-- Represents the sales data for a shoe store on a given day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing structure of the shoe store -/
structure Pricing where
  shoe_price : ℚ
  boot_price : ℚ

def monday_sales : DailySales :=
  { shoes := 22, boots := 24, total := 460 }

def tuesday_sales : DailySales :=
  { shoes := 8, boots := 32, total := 560 }

def store_pricing : Pricing :=
  { shoe_price := 2, boot_price := 17 }

theorem boots_sold_on_monday :
  ∃ (x : ℕ), 
    x = monday_sales.boots ∧
    store_pricing.boot_price = store_pricing.shoe_price + 15 ∧
    monday_sales.shoes * store_pricing.shoe_price + x * store_pricing.boot_price = monday_sales.total ∧
    tuesday_sales.shoes * store_pricing.shoe_price + tuesday_sales.boots * store_pricing.boot_price = tuesday_sales.total :=
by sorry

end NUMINAMATH_CALUDE_boots_sold_on_monday_l2325_232582


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2325_232584

theorem perfect_square_sum (n : ℕ+) : 
  (∃ m : ℕ, 4^7 + 4^n.val + 4^1998 = m^2) → (n.val = 1003 ∨ n.val = 3988) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2325_232584


namespace NUMINAMATH_CALUDE_median_sum_squares_l2325_232576

theorem median_sum_squares (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let m₁ := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m₂ := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m₃ := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m₁^2 + m₂^2 + m₃^2 = 442.5 := by
sorry

end NUMINAMATH_CALUDE_median_sum_squares_l2325_232576


namespace NUMINAMATH_CALUDE_soap_box_dimension_proof_l2325_232552

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem soap_box_dimension_proof 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton : carton = ⟨25, 48, 60⟩)
  (h_soap : soap = ⟨8, soap.width, 5⟩)
  (h_max_boxes : (300 : ℝ) * boxVolume soap ≤ boxVolume carton) :
  soap.width ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_soap_box_dimension_proof_l2325_232552


namespace NUMINAMATH_CALUDE_third_roll_wraps_four_gifts_l2325_232553

/-- Represents the number of gifts wrapped with the third roll of paper. -/
def gifts_wrapped_third_roll (total_rolls : ℕ) (total_gifts : ℕ) (gifts_first_roll : ℕ) (gifts_second_roll : ℕ) : ℕ :=
  total_gifts - (gifts_first_roll + gifts_second_roll)

/-- Proves that given 3 rolls of wrapping paper and 12 gifts, if 1 roll wraps 3 gifts
    and 1 roll wraps 5 gifts, then the number of gifts wrapped with the third roll is 4. -/
theorem third_roll_wraps_four_gifts :
  gifts_wrapped_third_roll 3 12 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_wraps_four_gifts_l2325_232553


namespace NUMINAMATH_CALUDE_ad_ratio_proof_l2325_232566

theorem ad_ratio_proof (page1_ads page2_ads page3_ads page4_ads total_ads : ℕ) : 
  page1_ads = 12 →
  page3_ads = page2_ads + 24 →
  page4_ads = (3 * page2_ads) / 4 →
  total_ads = page1_ads + page2_ads + page3_ads + page4_ads →
  (2 * total_ads) / 3 = 68 →
  page2_ads / page1_ads = 2 := by
sorry

end NUMINAMATH_CALUDE_ad_ratio_proof_l2325_232566


namespace NUMINAMATH_CALUDE_train_distance_problem_l2325_232522

theorem train_distance_problem (speed1 speed2 extra_distance : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 60)
  (h3 : extra_distance = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (distance1 distance2 : ℝ),
    distance1 > 0 ∧
    distance2 > 0 ∧
    distance2 = distance1 + extra_distance ∧
    distance1 / speed1 = distance2 / speed2 ∧
    distance1 + distance2 = 1100 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2325_232522


namespace NUMINAMATH_CALUDE_fifth_term_geometric_progression_l2325_232591

theorem fifth_term_geometric_progression :
  ∀ (b : ℕ → ℝ),
  (∀ n, b (n + 1) = b (n + 2) - b n) →  -- Each term from the second is the difference of adjacent terms
  b 1 = 7 - 3 * Real.sqrt 5 →           -- First term
  (∀ n, b (n + 1) > b n) →              -- Increasing progression
  b 5 = 2 :=                            -- Fifth term is 2
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_progression_l2325_232591


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2325_232577

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2325_232577


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l2325_232580

theorem prime_arithmetic_sequence_ones_digit (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  p % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l2325_232580


namespace NUMINAMATH_CALUDE_binary_101_is_5_l2325_232586

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₍₂₎ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_is_5_l2325_232586


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2325_232590

-- Define the polynomial
def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 48*x + 54

-- Define divisibility by (x - p)^2
def is_divisible_by_square (p : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, f x = (x - p)^2 * q x

-- Theorem statement
theorem polynomial_divisibility :
  ∀ p : ℝ, is_divisible_by_square p → p = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2325_232590


namespace NUMINAMATH_CALUDE_angle_sum_equals_pi_over_four_l2325_232506

theorem angle_sum_equals_pi_over_four (α β : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : Real.tan α = 1 / 7)
  (h6 : Real.tan β = 3 / 4) : 
  α + β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_equals_pi_over_four_l2325_232506


namespace NUMINAMATH_CALUDE_attendance_rate_proof_l2325_232556

theorem attendance_rate_proof (total_students : ℕ) (absent_students : ℕ) :
  total_students = 50 →
  absent_students = 2 →
  (((total_students - absent_students) : ℚ) / total_students) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_attendance_rate_proof_l2325_232556


namespace NUMINAMATH_CALUDE_total_marks_calculation_l2325_232558

theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 120 →
  average_mark = 35 →
  (num_candidates : ℚ) * average_mark = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l2325_232558


namespace NUMINAMATH_CALUDE_function_range_l2325_232512

/-- Given a real number m and a function f, prove that if there exists x₀ satisfying certain conditions, then m belongs to the specified range. -/
theorem function_range (m : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt 3 * Real.sin (π * x / m)) :
  (∃ x₀, (f x₀ = Real.sqrt 3 ∨ f x₀ = -Real.sqrt 3) ∧ x₀^2 + (f x₀)^2 < m^2) →
  m < -2 ∨ m > 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l2325_232512


namespace NUMINAMATH_CALUDE_unique_root_of_unity_polynomial_l2325_232525

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

def is_cube_root_of_unity (z : ℂ) : Prop :=
  ∃ k : ℕ, z^(3*k) = 1

theorem unique_root_of_unity_polynomial (c d : ℤ) :
  ∃! z : ℂ, is_root_of_unity z ∧ is_cube_root_of_unity z ∧ z^3 + c*z + d = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_of_unity_polynomial_l2325_232525


namespace NUMINAMATH_CALUDE_distance_at_speed1_l2325_232548

def total_distance : ℝ := 250
def speed1 : ℝ := 40
def speed2 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_at_speed1 (x : ℝ) 
  (h1 : x / speed1 + (total_distance - x) / speed2 = total_time) :
  x = 124 := by
  sorry

end NUMINAMATH_CALUDE_distance_at_speed1_l2325_232548


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_46_l2325_232546

theorem sin_cos_sum_14_46 :
  Real.sin (14 * π / 180) * Real.cos (46 * π / 180) +
  Real.sin (46 * π / 180) * Real.cos (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_46_l2325_232546


namespace NUMINAMATH_CALUDE_trapezoid_area_property_l2325_232596

/-- Represents the area of a trapezoid with bases and altitude in arithmetic progression -/
def trapezoid_area (a : ℝ) : ℝ := a ^ 2

/-- The area of a trapezoid with bases and altitude in arithmetic progression
    can be any non-negative real number -/
theorem trapezoid_area_property :
  ∀ (J : ℝ), J ≥ 0 → ∃ (a : ℝ), trapezoid_area a = J :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_property_l2325_232596


namespace NUMINAMATH_CALUDE_inequality_solutions_l2325_232581

-- Define the inequalities
def ineq1a (x : ℝ) := 2*x + 8 > 5*x + 2
def ineq1b (x : ℝ) := 2*x + 8 + 4/(x-1) > 5*x + 2 + 4/(x-1)

def ineq2a (x : ℝ) := 2*x + 8 < 5*x + 2
def ineq2b (x : ℝ) := 2*x + 8 + 4/(x-1) < 5*x + 2 + 4/(x-1)

def ineq3a (x : ℝ) := 3/(x-1) > (x+2)/(x-2)
def ineq3b (x : ℝ) := 3/(x-1) + (3*x-4)/(x-1) > (x+2)/(x-2) + (3*x-4)/(x-1)

-- Define the theorem
theorem inequality_solutions :
  (∃ x : ℝ, ineq1a x ≠ ineq1b x) ∧
  (∀ x : ℝ, ineq2a x ↔ ineq2b x) ∧
  (∀ x : ℝ, x ≠ 1 → x ≠ 2 → (ineq3a x ↔ ineq3b x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2325_232581


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2325_232530

/-- Given a line L1 with equation mx - m^2y = 1 passing through point P(2, 1),
    prove that the perpendicular line L2 at P has equation x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∀ x y, x + y - 3 = 0 ↔ 
    (m * x - m^2 * y = 1 → 
      (x - 2) * (x - 2) + (y - 1) * (y - 1) = 
      (2 - 2) * (2 - 2) + (1 - 1) * (1 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2325_232530


namespace NUMINAMATH_CALUDE_unique_prime_seventh_power_l2325_232597

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q, Prime q ∧ p + 25 = q^7 ↔ p = 103 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_seventh_power_l2325_232597


namespace NUMINAMATH_CALUDE_midpoint_one_sixth_one_ninth_l2325_232592

theorem midpoint_one_sixth_one_ninth :
  (1 / 6 + 1 / 9) / 2 = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_midpoint_one_sixth_one_ninth_l2325_232592


namespace NUMINAMATH_CALUDE_min_value_S_l2325_232595

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) 
  (h2 : (2*a + b*c)*(2*b + c*a)*(2*c + a*b) > 200) : 
  ∃ (m : ℤ), m = 256 ∧ 
  ∀ (x y z : ℤ), x + y + z = 2 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) > 200 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l2325_232595


namespace NUMINAMATH_CALUDE_division_by_fraction_l2325_232573

theorem division_by_fraction : (10 + 6) / (1 / 4) = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_l2325_232573


namespace NUMINAMATH_CALUDE_xy_squared_sum_l2325_232559

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l2325_232559


namespace NUMINAMATH_CALUDE_argument_not_pi_over_four_l2325_232520

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z-|z+1|| = |z+|z-1||
def condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.abs (z + 1)) = Complex.abs (z + Complex.abs (z - 1))

-- Theorem statement
theorem argument_not_pi_over_four (h : condition z) :
  Complex.arg z ≠ Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_argument_not_pi_over_four_l2325_232520


namespace NUMINAMATH_CALUDE_grinder_loss_percentage_l2325_232539

/-- Represents the financial transaction of buying and selling items --/
structure Transaction where
  grinder_cp : ℝ  -- Cost price of grinder
  mobile_cp : ℝ   -- Cost price of mobile
  mobile_profit_percent : ℝ  -- Profit percentage on mobile
  total_profit : ℝ  -- Overall profit
  grinder_loss_percent : ℝ  -- Loss percentage on grinder (to be proved)

/-- Theorem stating the conditions and the result to be proved --/
theorem grinder_loss_percentage
  (t : Transaction)
  (h1 : t.grinder_cp = 15000)
  (h2 : t.mobile_cp = 8000)
  (h3 : t.mobile_profit_percent = 10)
  (h4 : t.total_profit = 500)
  : t.grinder_loss_percent = 2 := by
  sorry


end NUMINAMATH_CALUDE_grinder_loss_percentage_l2325_232539


namespace NUMINAMATH_CALUDE_shift_selection_count_l2325_232501

def workers : Nat := 3
def positions : Nat := 2

theorem shift_selection_count : (workers * (workers - 1) = 6) := by
  sorry

end NUMINAMATH_CALUDE_shift_selection_count_l2325_232501


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l2325_232527

/-- Represents the weight used by the dealer in grams -/
def dealer_weight : ℝ := 500

/-- Represents the standard weight of 1 kg in grams -/
def standard_weight : ℝ := 1000

/-- The dealer's profit percentage -/
def profit_percentage : ℝ := 50

theorem dishonest_dealer_profit :
  dealer_weight / standard_weight = 1 - (100 / (100 + profit_percentage)) :=
sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l2325_232527


namespace NUMINAMATH_CALUDE_min_value_theorem_l2325_232516

theorem min_value_theorem (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2*m + n = 1) :
  1/m + 2/n ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), 0 < m₀ ∧ 0 < n₀ ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2325_232516


namespace NUMINAMATH_CALUDE_dans_balloons_l2325_232583

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → tims_balloons = 7 * dans_balloons → dans_balloons = 29 := by
  sorry

end NUMINAMATH_CALUDE_dans_balloons_l2325_232583


namespace NUMINAMATH_CALUDE_product_of_digits_8056_base_8_l2325_232536

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

theorem product_of_digits_8056_base_8 :
  (base_8_representation 8056).foldl (·*·) 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_8056_base_8_l2325_232536


namespace NUMINAMATH_CALUDE_number_of_unique_lines_l2325_232561

/-- The set of possible coefficients for A and B -/
def S : Finset ℕ := {0, 1, 2, 3, 5}

/-- A line is represented by a pair of distinct coefficients (A, B) -/
def Line : Type := { p : ℕ × ℕ // p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 }

/-- The set of all possible lines -/
def AllLines : Finset Line := sorry

theorem number_of_unique_lines : Finset.card AllLines = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_of_unique_lines_l2325_232561


namespace NUMINAMATH_CALUDE_new_person_weight_l2325_232505

/-- Given a group of 8 persons, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the new person weighs 93 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2325_232505


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l2325_232532

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) :
  (S - R / 100 * S) * (1 + 25 / 100) = S → R = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l2325_232532


namespace NUMINAMATH_CALUDE_ellipse_properties_l2325_232500

/-- Ellipse C in the Cartesian coordinate system -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line intersecting ellipse C -/
def L (k x : ℝ) : ℝ := k * (x - 4)

/-- Point A: left vertex of ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- Point M: first intersection of line L and ellipse C -/
noncomputable def M (k : ℝ) : ℝ × ℝ := 
  let x₁ := (16 * k^2 + 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₁, L k x₁)

/-- Point N: second intersection of line L and ellipse C -/
noncomputable def N (k : ℝ) : ℝ × ℝ := 
  let x₂ := (16 * k^2 - 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₂, L k x₂)

/-- Point P: intersection of x = 1 and line BM -/
noncomputable def P (k : ℝ) : ℝ × ℝ := 
  let x₁ := (M k).1
  (1, k * (x₁ - 4) / (x₁ - 2))

/-- Area of triangle OMN -/
noncomputable def area_OMN (k : ℝ) : ℝ := 
  8 * Real.sqrt (1/k^2 - 12) / (1/k^2 + 4)

theorem ellipse_properties (k : ℝ) (hk : k ≠ 0) :
  (∃ (t : ℝ), t • (A.1 - (P k).1, A.2 - (P k).2) = ((N k).1 - A.1, (N k).2 - A.2)) ∧
  (∀ (k : ℝ), k ≠ 0 → area_OMN k ≤ 1) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ area_OMN k = 1) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2325_232500


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2325_232517

/-- Given a line with equation y - 3 = -3(x - 6), 
    prove that the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 3 = -3 * (x_int - 6)) ∧ 
    (0 - 3 = -3 * (x_int - 6)) ∧
    (y_int - 3 = -3 * (0 - 6)) ∧
    (x_int + y_int = 28)) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2325_232517


namespace NUMINAMATH_CALUDE_kenny_time_ratio_l2325_232551

/-- Proves that the ratio of Kenny's running time to basketball playing time is 2:1 -/
theorem kenny_time_ratio : 
  ∀ (basketball_time trumpet_time running_time : ℕ),
    basketball_time = 10 →
    trumpet_time = 40 →
    trumpet_time = 2 * running_time →
    running_time / basketball_time = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_kenny_time_ratio_l2325_232551


namespace NUMINAMATH_CALUDE_robins_total_distance_l2325_232554

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_total_distance (distance_to_center : ℕ) (initial_distance : ℕ) : 
  distance_to_center = 500 → initial_distance = 200 → 
  initial_distance + initial_distance + distance_to_center = 900 := by
sorry

end NUMINAMATH_CALUDE_robins_total_distance_l2325_232554


namespace NUMINAMATH_CALUDE_base_k_conversion_l2325_232575

theorem base_k_conversion (k : ℕ) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l2325_232575


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2325_232589

theorem line_segment_endpoint (y : ℝ) : 
  y < 0 → 
  ((3 - 1)^2 + (-2 - y)^2)^(1/2) = 15 → 
  y = -2 - (221 : ℝ)^(1/2) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2325_232589


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2325_232535

theorem quadratic_roots_sum_of_squares : 
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 3*x₁ - 5 = 0) → (x₂^2 - 3*x₂ - 5 = 0) → (x₁ ≠ x₂) → 
  x₁^2 + x₂^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2325_232535


namespace NUMINAMATH_CALUDE_bacon_count_l2325_232588

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 330

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 61

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_count : bacon = 269 := by
  sorry

end NUMINAMATH_CALUDE_bacon_count_l2325_232588


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l2325_232503

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 7*Complex.I) = Real.sqrt 34 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l2325_232503


namespace NUMINAMATH_CALUDE_eventual_stability_l2325_232541

/-- Represents the state of the circular arrangement at a given time step -/
def CircularState := Vector Bool 101

/-- Defines the update rule for a single element based on its neighbors -/
def updateElement (left right current : Bool) : Bool :=
  if left ≠ current ∧ right ≠ current then !current else current

/-- Applies the update rule to the entire circular arrangement -/
def updateState (state : CircularState) : CircularState :=
  Vector.ofFn (fun i =>
    updateElement
      (state.get ((i - 1 + 101) % 101))
      (state.get ((i + 1) % 101))
      (state.get i))

/-- Predicate to check if a state is stable (doesn't change under update) -/
def isStable (state : CircularState) : Prop :=
  updateState state = state

/-- The main theorem: there exists a stable state reachable from any initial state -/
theorem eventual_stability :
  ∀ (initialState : CircularState),
  ∃ (n : ℕ) (stableState : CircularState),
  (∀ k, k ≥ n → (updateState^[k] initialState) = stableState) ∧
  isStable stableState :=
sorry


end NUMINAMATH_CALUDE_eventual_stability_l2325_232541


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l2325_232564

theorem sum_of_max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = x + y + z) :
  ∃ (min_val max_val : ℝ),
    (∀ a b c : ℝ, a^2 + b^2 + c^2 = a + b + c → min_val ≤ a + b + c ∧ a + b + c ≤ max_val) ∧
    min_val + max_val = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l2325_232564


namespace NUMINAMATH_CALUDE_permutations_not_adjacent_l2325_232545

/-- The number of permutations of three 'a's, four 'b's, and two 'c's -/
def total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'a's are adjacent -/
def perm_a_adjacent : ℕ := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'b's are adjacent -/
def perm_b_adjacent : ℕ := Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Permutations where all 'c's are adjacent -/
def perm_c_adjacent : ℕ := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)

/-- Permutations where both 'a's and 'b's are adjacent -/
def perm_ab_adjacent : ℕ := Nat.factorial 4 / Nat.factorial 2

/-- Permutations where both 'a's and 'c's are adjacent -/
def perm_ac_adjacent : ℕ := Nat.factorial 6 / Nat.factorial 4

/-- Permutations where both 'b's and 'c's are adjacent -/
def perm_bc_adjacent : ℕ := Nat.factorial 5 / Nat.factorial 3

/-- Permutations where 'a's, 'b's, and 'c's are all adjacent -/
def perm_abc_adjacent : ℕ := Nat.factorial 3

theorem permutations_not_adjacent : 
  total_permutations - (perm_a_adjacent + perm_b_adjacent + perm_c_adjacent - 
  perm_ab_adjacent - perm_ac_adjacent - perm_bc_adjacent + perm_abc_adjacent) = 871 := by
  sorry

end NUMINAMATH_CALUDE_permutations_not_adjacent_l2325_232545
