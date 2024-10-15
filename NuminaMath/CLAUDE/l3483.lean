import Mathlib

namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_even_numbers_l3483_348365

theorem smallest_of_three_consecutive_even_numbers (a b c : ℕ) : 
  (∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4) →  -- consecutive even numbers
  a + b + c = 162 →                                      -- sum is 162
  a = 52 :=                                              -- smallest number is 52
by sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_even_numbers_l3483_348365


namespace NUMINAMATH_CALUDE_locus_of_q_l3483_348317

/-- The ellipse in the problem -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The hyperbola that is the locus of Q -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | (q.1^2 / a^2) - (q.2^2 / b^2) = 1}

/-- P and P' form a vertical chord of the ellipse -/
def VerticalChord (a b : ℝ) (p p' : ℝ × ℝ) : Prop :=
  p ∈ Ellipse a b ∧ p' ∈ Ellipse a b ∧ p.1 = p'.1

/-- Q is the intersection of A'P and AP' -/
def IntersectionPoint (a : ℝ) (p p' q : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ,
    q.1 = t * (p.1 + a) + (1 - t) * (-a) ∧
    q.2 = t * p.2 ∧
    q.1 = s * (p'.1 - a) + (1 - s) * a ∧
    q.2 = s * p'.2

/-- The main theorem -/
theorem locus_of_q (a b : ℝ) (p p' q : ℝ × ℝ) 
    (h_ab : a > 0 ∧ b > 0)
    (h_ellipse : p ∈ Ellipse a b ∧ p' ∈ Ellipse a b)
    (h_vertical : VerticalChord a b p p')
    (h_intersect : IntersectionPoint a p p' q) :
  q ∈ Hyperbola a b := by
  sorry

end NUMINAMATH_CALUDE_locus_of_q_l3483_348317


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3483_348302

theorem sqrt_product_plus_one : 
  Real.sqrt ((21:ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3483_348302


namespace NUMINAMATH_CALUDE_lucy_grocery_cost_l3483_348339

/-- Represents the total cost of Lucy's grocery purchases in USD -/
def total_cost_usd (cookies_packs : ℕ) (cookies_price : ℚ)
                   (noodles_packs : ℕ) (noodles_price : ℚ)
                   (soup_cans : ℕ) (soup_price : ℚ)
                   (cereals_boxes : ℕ) (cereals_price : ℚ)
                   (crackers_packs : ℕ) (crackers_price : ℚ)
                   (usd_to_eur : ℚ) (usd_to_gbp : ℚ) : ℚ :=
  cookies_packs * cookies_price +
  (noodles_packs * noodles_price) / usd_to_eur +
  (soup_cans * soup_price) / usd_to_gbp +
  cereals_boxes * cereals_price +
  (crackers_packs * crackers_price) / usd_to_eur

/-- The theorem stating that Lucy's total grocery cost is $183.92 -/
theorem lucy_grocery_cost :
  total_cost_usd 12 (5/2) 16 (9/5) 28 (6/5) 5 (17/5) 45 (11/10) (17/20) (3/4) = 18392/100 := by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_cost_l3483_348339


namespace NUMINAMATH_CALUDE_shell_collection_division_l3483_348383

theorem shell_collection_division (lino_morning : ℝ) (maria_morning : ℝ) 
  (lino_afternoon : ℝ) (maria_afternoon : ℝ) 
  (h1 : lino_morning = 292.5) 
  (h2 : maria_morning = 375.25)
  (h3 : lino_afternoon = 324.75)
  (h4 : maria_afternoon = 419.3) : 
  (lino_morning + lino_afternoon + maria_morning + maria_afternoon) / 2 = 705.9 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_division_l3483_348383


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3483_348385

/-- The maximum area of an equilateral triangle inscribed in a 12x17 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 325 * Real.sqrt 3 - 612 ∧ 
  ∀ (triangle_area : ℝ), 
    (∃ (x y : ℝ), 
      0 ≤ x ∧ x ≤ 12 ∧ 
      0 ≤ y ∧ y ≤ 17 ∧ 
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3483_348385


namespace NUMINAMATH_CALUDE_polynomial_division_l3483_348356

theorem polynomial_division (a : ℝ) (h : a ≠ 0) :
  (9 * a^6 - 12 * a^3) / (3 * a^3) = 3 * a^3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l3483_348356


namespace NUMINAMATH_CALUDE_sarah_toads_count_l3483_348328

/-- Proves that Sarah has 100 toads given the conditions of the problem -/
theorem sarah_toads_count : ∀ (tim_toads jim_toads sarah_toads : ℕ),
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 2 * jim_toads →
  sarah_toads = 100 := by
sorry

end NUMINAMATH_CALUDE_sarah_toads_count_l3483_348328


namespace NUMINAMATH_CALUDE_probability_of_specific_distribution_l3483_348359

-- Define the number of balls and boxes
def num_balls : ℕ := 6
def num_boxes : ℕ := 4

-- Define the probability of a specific distribution
def prob_specific_distribution : ℚ := 45 / 128

-- Define the function to calculate the total number of ways to distribute balls
def total_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

-- Define the function to calculate the number of ways to distribute balls in the specific pattern
def specific_distribution_count (balls : ℕ) (boxes : ℕ) : ℕ :=
  (balls.choose 3) * ((balls - 3).choose 2) * (boxes.factorial)

-- Theorem statement
theorem probability_of_specific_distribution :
  (specific_distribution_count num_balls num_boxes : ℚ) / (total_distributions num_balls num_boxes : ℚ) = prob_specific_distribution :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_distribution_l3483_348359


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3483_348315

theorem expression_equals_zero :
  (1 - Real.sqrt 2) ^ 0 + |2 - Real.sqrt 5| + (-1) ^ 2022 - (1/3) * Real.sqrt 45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3483_348315


namespace NUMINAMATH_CALUDE_find_a_l3483_348337

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 < a^2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B (a : ℝ) : Set ℝ := A a ∩ B

-- State the theorem
theorem find_a : ∀ a : ℝ, A_intersect_B a = {x : ℝ | 1 < x ∧ x < 2} → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3483_348337


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l3483_348375

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) (seeds_per_bed : ℕ) :
  total_seeds = 32 →
  num_flowerbeds = 8 →
  total_seeds = num_flowerbeds * seeds_per_bed →
  seeds_per_bed = 4 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l3483_348375


namespace NUMINAMATH_CALUDE_prob_one_letter_each_name_l3483_348360

/-- Probability of selecting one letter from each person's name -/
theorem prob_one_letter_each_name :
  let total_cards : ℕ := 14
  let elena_cards : ℕ := 5
  let mark_cards : ℕ := 4
  let julia_cards : ℕ := 5
  let num_permutations : ℕ := 6  -- 3! permutations of 3 items
  
  elena_cards + mark_cards + julia_cards = total_cards →
  
  (elena_cards : ℚ) / total_cards *
  (mark_cards : ℚ) / (total_cards - 1) *
  (julia_cards : ℚ) / (total_cards - 2) *
  num_permutations = 25 / 91 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_letter_each_name_l3483_348360


namespace NUMINAMATH_CALUDE_weekly_running_distance_l3483_348306

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def total_distance_run (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_running_distance :
  total_distance_run 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_weekly_running_distance_l3483_348306


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3483_348352

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x + 11) - (x^6 + 2 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 5 * x^4 - x^3 + x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3483_348352


namespace NUMINAMATH_CALUDE_jerry_medical_bills_l3483_348304

/-- The amount Jerry is claiming for medical bills -/
def medical_bills : ℝ := sorry

/-- Jerry's annual salary -/
def annual_salary : ℝ := 50000

/-- Number of years of lost salary -/
def years_of_lost_salary : ℕ := 30

/-- Total lost salary -/
def total_lost_salary : ℝ := annual_salary * years_of_lost_salary

/-- Punitive damages multiplier -/
def punitive_multiplier : ℕ := 3

/-- Percentage of claim Jerry receives -/
def claim_percentage : ℝ := 0.8

/-- Total amount Jerry receives -/
def total_received : ℝ := 5440000

/-- Theorem stating the amount of medical bills Jerry is claiming -/
theorem jerry_medical_bills :
  claim_percentage * (total_lost_salary + medical_bills + 
    punitive_multiplier * (total_lost_salary + medical_bills)) = total_received ∧
  medical_bills = 200000 := by sorry

end NUMINAMATH_CALUDE_jerry_medical_bills_l3483_348304


namespace NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l3483_348387

/-- Given a parabola y² = 4x and a line passing through (1,0) intersecting the parabola at A and B,
    prove that OB · OC = -5, where C is symmetric to A with respect to the y-axis -/
theorem parabola_line_intersection_dot_product :
  ∀ (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ),
  -- Line passes through (1,0)
  y₁ = k * (x₁ - 1) →
  y₂ = k * (x₂ - 1) →
  -- A and B are on the parabola
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  -- A and B are distinct points
  x₁ ≠ x₂ →
  -- C is symmetric to A with respect to y-axis
  let xc := -x₁
  let yc := y₁
  -- OB · OC = -5
  x₂ * xc + y₂ * yc = -5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l3483_348387


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l3483_348386

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  b / c = 2 / 5 ∧ 
  a^2 + b^2 + c^2 = 1862 → 
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l3483_348386


namespace NUMINAMATH_CALUDE_unique_solution_l3483_348350

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem unique_solution : 
  ∃! x : ℕ, 
    digit_product x = 44 * x - 86868 ∧ 
    is_perfect_cube (digit_sum x) ∧
    x = 1989 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3483_348350


namespace NUMINAMATH_CALUDE_jerrys_average_score_l3483_348398

theorem jerrys_average_score (current_total : ℝ) (desired_average : ℝ) (fourth_test_score : ℝ) :
  (current_total / 3 + 2 = desired_average) →
  (current_total + fourth_test_score) / 4 = desired_average →
  fourth_test_score = 98 →
  current_total / 3 = 90 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l3483_348398


namespace NUMINAMATH_CALUDE_smallest_number_l3483_348340

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 2) (hc : c = -4) (hd : d = -1) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3483_348340


namespace NUMINAMATH_CALUDE_gcd_repeating_six_digit_l3483_348344

def is_repeating_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeating_six_digit :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, is_repeating_six_digit n → d ∣ n) ∧
  (∀ d' : ℕ, d' > 0 → (∀ n : ℕ, is_repeating_six_digit n → d' ∣ n) → d' ≤ d) ∧
  d = 1001 :=
sorry

end NUMINAMATH_CALUDE_gcd_repeating_six_digit_l3483_348344


namespace NUMINAMATH_CALUDE_total_shaded_area_is_71_l3483_348362

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  r.width * r.height

theorem total_shaded_area_is_71 (rect1 rect2 overlap : Rectangle)
    (h1 : rect1.width = 4 ∧ rect1.height = 12)
    (h2 : rect2.width = 5 ∧ rect2.height = 7)
    (h3 : overlap.width = 3 ∧ overlap.height = 4) :
    area rect1 + area rect2 - area overlap = 71 := by
  sorry

#check total_shaded_area_is_71

end NUMINAMATH_CALUDE_total_shaded_area_is_71_l3483_348362


namespace NUMINAMATH_CALUDE_suraj_average_after_ninth_innings_l3483_348334

/-- Represents a cricket player's performance -/
structure CricketPerformance where
  innings : ℕ
  lowestScore : ℕ
  highestScore : ℕ
  fiftyPlusInnings : ℕ
  totalRuns : ℕ

/-- Calculates the average runs per innings -/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

theorem suraj_average_after_ninth_innings 
  (suraj : CricketPerformance)
  (h1 : suraj.innings = 8)
  (h2 : suraj.lowestScore = 25)
  (h3 : suraj.highestScore = 80)
  (h4 : suraj.fiftyPlusInnings = 3)
  (h5 : average suraj + 6 = average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 }) :
  average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 } = 42 := by
  sorry


end NUMINAMATH_CALUDE_suraj_average_after_ninth_innings_l3483_348334


namespace NUMINAMATH_CALUDE_custom_mul_equality_l3483_348303

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := (a - b^3)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_mul_equality (x y : ℝ) :
  custom_mul ((x - y)^2) ((y^2 - x^2)^2) = ((x - y)^2 - (y^4 - 2*x^2*y^2 + x^4)^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_equality_l3483_348303


namespace NUMINAMATH_CALUDE_only_parallel_assertion_correct_l3483_348322

/-- Represents a line in 3D space -/
structure Line3D where
  -- This is just a placeholder definition
  dummy : Unit

/-- Perpendicular relation between two lines -/
def perpendicular (a b : Line3D) : Prop :=
  sorry

/-- Skew relation between two lines -/
def skew (a b : Line3D) : Prop :=
  sorry

/-- Intersection relation between two lines -/
def intersects (a b : Line3D) : Prop :=
  sorry

/-- Coplanar relation between two lines -/
def coplanar (a b : Line3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel (a b : Line3D) : Prop :=
  sorry

/-- Theorem stating that only the parallel assertion is correct -/
theorem only_parallel_assertion_correct (a b c : Line3D) :
  (¬ (∀ a b c, perpendicular a b → perpendicular b c → perpendicular a c)) ∧
  (¬ (∀ a b c, skew a b → skew b c → skew a c)) ∧
  (¬ (∀ a b c, intersects a b → intersects b c → intersects a c)) ∧
  (¬ (∀ a b c, coplanar a b → coplanar b c → coplanar a c)) ∧
  (∀ a b c, parallel a b → parallel b c → parallel a c) :=
by sorry

end NUMINAMATH_CALUDE_only_parallel_assertion_correct_l3483_348322


namespace NUMINAMATH_CALUDE_mary_nickels_count_l3483_348390

/-- The number of nickels Mary has after receiving some from her dad and sister -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_dad + from_sister

/-- Theorem stating that Mary's total nickels is the sum of her initial amount and what she received -/
theorem mary_nickels_count : total_nickels 7 12 9 = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_count_l3483_348390


namespace NUMINAMATH_CALUDE_skew_lines_projection_not_two_points_l3483_348377

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D

-- Define a type for points in 2D space (the projection plane)
structure Point2D where
  -- Add necessary fields to represent a point in 2D

-- Define a projection function from 3D to 2D
def project (l : Line3D) : Point2D :=
  sorry

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem skew_lines_projection_not_two_points 
  (l1 l2 : Line3D) (h : are_skew l1 l2) : 
  ¬(∃ (p1 p2 : Point2D), project l1 = p1 ∧ project l2 = p2 ∧ p1 ≠ p2) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_projection_not_two_points_l3483_348377


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l3483_348333

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 60)
  (h_gcd : Nat.gcd x y = 10) : 
  x * y = 600 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l3483_348333


namespace NUMINAMATH_CALUDE_angle_A_is_60_l3483_348323

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the conditions
axiom acute_triangle : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90
axiom side_a : a = 2 * Real.sqrt 3
axiom side_b : b = 2 * Real.sqrt 2
axiom angle_B : B = 45

-- Theorem to prove
theorem angle_A_is_60 : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_l3483_348323


namespace NUMINAMATH_CALUDE_circle_symmetry_l3483_348366

/-- Given a circle and a line of symmetry, prove that another circle is symmetric to the given circle about the line. -/
theorem circle_symmetry (x y : ℝ) :
  let original_circle := (x - 1)^2 + (y - 2)^2 = 1
  let symmetry_line := x - y - 2 = 0
  let symmetric_circle := (x - 4)^2 + (y + 1)^2 = 1
  (∀ (x₀ y₀ : ℝ), original_circle → 
    ∃ (x₁ y₁ : ℝ), symmetric_circle ∧ 
    ((x₀ + x₁) / 2 - (y₀ + y₁) / 2 - 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3483_348366


namespace NUMINAMATH_CALUDE_expression_evaluation_l3483_348319

theorem expression_evaluation : 
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3483_348319


namespace NUMINAMATH_CALUDE_vertex_determines_parameters_l3483_348396

def quadratic_function (h k : ℝ) (x : ℝ) : ℝ := -3 * (x - h)^2 + k

theorem vertex_determines_parameters (h k : ℝ) :
  (∀ x, quadratic_function h k x = quadratic_function 1 (-2) x) →
  h = 1 ∧ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_determines_parameters_l3483_348396


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3483_348336

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 50

-- Define the theorem
theorem train_distance_theorem :
  ∀ (t : ℝ), -- t represents the time taken for trains to meet
  t > 0 → -- time is positive
  speed_train1 * t + speed_train2 * t = -- total distance is sum of distances traveled by both trains
  speed_train1 * t + (speed_train1 * t + distance_difference) → -- one train travels 50 km more
  speed_train1 * t + (speed_train1 * t + distance_difference) = 450 -- total distance is 450 km
  := by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l3483_348336


namespace NUMINAMATH_CALUDE_distance_travelled_l3483_348318

-- Define the velocity function
def v (t : ℝ) : ℝ := 2 * t - 3

-- Define the theorem
theorem distance_travelled (t₀ t₁ : ℝ) (h : 0 ≤ t₀ ∧ t₁ = 5) :
  ∫ t in t₀..t₁, |v t| = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_l3483_348318


namespace NUMINAMATH_CALUDE_pizza_slices_with_both_toppings_l3483_348395

-- Define the total number of slices
def total_slices : ℕ := 24

-- Define the number of slices with pepperoni
def pepperoni_slices : ℕ := 12

-- Define the number of slices with mushrooms
def mushroom_slices : ℕ := 14

-- Define the number of vegetarian slices
def vegetarian_slices : ℕ := 4

-- Theorem to prove
theorem pizza_slices_with_both_toppings :
  ∃ n : ℕ, 
    -- Every slice has at least one condition met
    (n + (pepperoni_slices - n) + (mushroom_slices - n) + vegetarian_slices = total_slices) ∧
    -- n is the number of slices with both pepperoni and mushrooms
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_with_both_toppings_l3483_348395


namespace NUMINAMATH_CALUDE_words_with_vowels_l3483_348371

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels :
  total_words - words_without_vowels = 6752 := by sorry

end NUMINAMATH_CALUDE_words_with_vowels_l3483_348371


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3483_348354

theorem divisibility_by_six (n : ℕ) : 6 ∣ (n^3 - 7*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3483_348354


namespace NUMINAMATH_CALUDE_encoded_CDE_is_174_l3483_348312

/-- Represents the encoding of a base-6 digit --/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding base-6 value --/
def digit_to_base6 : Digit → Nat
| Digit.A => 5
| Digit.B => 0
| Digit.C => 4
| Digit.D => 5
| Digit.E => 0
| Digit.F => 1

/-- Converts a base-6 number represented as a list of Digits to base-10 --/
def base6_to_base10 (digits : List Digit) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + (digit_to_base6 d) * (6^i)) 0

/-- The main theorem to prove --/
theorem encoded_CDE_is_174 :
  base6_to_base10 [Digit.C, Digit.D, Digit.E] = 174 :=
by sorry

end NUMINAMATH_CALUDE_encoded_CDE_is_174_l3483_348312


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l3483_348355

theorem min_value_absolute_sum (x : ℝ) : 
  |x - 4| + |x + 6| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 6| + |y - 5| = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l3483_348355


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l3483_348376

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 18/5

/-- The amount John had left after spending at the arcade and toy store, in dollars -/
def remaining_amount : ℚ := 24/25

theorem johns_arcade_spending :
  let remaining_after_arcade : ℚ := weekly_allowance * (1 - arcade_fraction)
  let spent_at_toy_store : ℚ := remaining_after_arcade * (1/3)
  remaining_after_arcade - spent_at_toy_store = remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l3483_348376


namespace NUMINAMATH_CALUDE_uphill_distance_l3483_348329

/-- Proves that the uphill distance is 45 km given the conditions of the problem -/
theorem uphill_distance (flat_speed : ℝ) (uphill_speed : ℝ) (extra_flat_distance : ℝ) :
  flat_speed = 20 →
  uphill_speed = 12 →
  extra_flat_distance = 30 →
  ∃ (uphill_distance : ℝ),
    uphill_distance / uphill_speed = (uphill_distance + extra_flat_distance) / flat_speed ∧
    uphill_distance = 45 :=
by sorry

end NUMINAMATH_CALUDE_uphill_distance_l3483_348329


namespace NUMINAMATH_CALUDE_power_equality_l3483_348342

theorem power_equality (n : ℕ) : 4^n = 64^2 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3483_348342


namespace NUMINAMATH_CALUDE_total_sticks_is_129_l3483_348320

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

/-- Theorem stating that the total number of sticks needed is 129 -/
theorem total_sticks_is_129 : total_sticks = 129 := by
  sorry

#eval total_sticks

end NUMINAMATH_CALUDE_total_sticks_is_129_l3483_348320


namespace NUMINAMATH_CALUDE_kindergarten_card_problem_l3483_348311

/-- Represents the distribution of cards among children in a kindergarten. -/
structure CardDistribution where
  ma_three : ℕ  -- Number of children with three "MA" cards
  ma_two : ℕ    -- Number of children with two "MA" cards and one "NY" card
  ny_two : ℕ    -- Number of children with two "NY" cards and one "MA" card
  ny_three : ℕ  -- Number of children with three "NY" cards

/-- The conditions given in the problem. -/
def problem_conditions (d : CardDistribution) : Prop :=
  d.ma_three + d.ma_two = 20 ∧
  d.ny_two + d.ny_three = 30 ∧
  d.ma_two + d.ny_two = 40

/-- The theorem stating that given the problem conditions, 
    the number of children with all three cards the same is 10. -/
theorem kindergarten_card_problem (d : CardDistribution) :
  problem_conditions d → d.ma_three + d.ny_three = 10 := by
  sorry


end NUMINAMATH_CALUDE_kindergarten_card_problem_l3483_348311


namespace NUMINAMATH_CALUDE_equation_solution_l3483_348349

theorem equation_solution : ∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3483_348349


namespace NUMINAMATH_CALUDE_camping_trip_purchases_l3483_348301

/-- Given Rebecca's camping trip purchases, prove the difference between water bottles and tent stakes --/
theorem camping_trip_purchases (total_items tent_stakes drink_mix water_bottles : ℕ) : 
  total_items = 22 →
  tent_stakes = 4 →
  drink_mix = 3 * tent_stakes →
  total_items = tent_stakes + drink_mix + water_bottles →
  water_bottles - tent_stakes = 2 := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_purchases_l3483_348301


namespace NUMINAMATH_CALUDE_fifth_page_stickers_l3483_348307

def sticker_sequence (n : ℕ) : ℕ := 8 * n

theorem fifth_page_stickers : sticker_sequence 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_page_stickers_l3483_348307


namespace NUMINAMATH_CALUDE_f_composed_with_g_l3483_348373

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x - 2

theorem f_composed_with_g : f (2 + g 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l3483_348373


namespace NUMINAMATH_CALUDE_tank_filling_solution_l3483_348369

/-- Represents the tank filling problem -/
def TankFillingProblem (tankCapacity : Real) (initialFillRatio : Real) 
  (fillingRate : Real) (drain1Rate : Real) (drain2Rate : Real) : Prop :=
  let remainingVolume := tankCapacity * (1 - initialFillRatio)
  let netFlowRate := fillingRate - drain1Rate - drain2Rate
  let timeToFill := remainingVolume / netFlowRate
  timeToFill = 6

/-- The theorem stating the solution to the tank filling problem -/
theorem tank_filling_solution :
  TankFillingProblem 1000 0.5 (1/2) (1/4) (1/6) := by
  sorry

#check tank_filling_solution

end NUMINAMATH_CALUDE_tank_filling_solution_l3483_348369


namespace NUMINAMATH_CALUDE_factorial_divisibility_power_of_two_l3483_348310

theorem factorial_divisibility_power_of_two (n : ℕ) : 
  (∃ k : ℕ, n = 2^k) ↔ (n.factorial % 2^(n-1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_power_of_two_l3483_348310


namespace NUMINAMATH_CALUDE_intersection_polyhedron_volume_l3483_348345

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The polyhedron formed by the intersection of a regular tetrahedron with its image under symmetry relative to the midpoint of its height -/
def IntersectionPolyhedron (t : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The volume of the intersection polyhedron is (a^3 * √2) / 54 -/
theorem intersection_polyhedron_volume (t : RegularTetrahedron) :
    volume (IntersectionPolyhedron t) = (t.edge_length^3 * Real.sqrt 2) / 54 :=
  sorry

end NUMINAMATH_CALUDE_intersection_polyhedron_volume_l3483_348345


namespace NUMINAMATH_CALUDE_four_numbers_with_consecutive_sums_l3483_348367

theorem four_numbers_with_consecutive_sums : ∃ (a b c d : ℕ),
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) ∧
  (a + b = 2023) ∧
  (a + c = 2024) ∧
  (a + d = 2026) ∧
  (b + c = 2025) ∧
  (b + d = 2027) ∧
  (c + d = 2028) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_consecutive_sums_l3483_348367


namespace NUMINAMATH_CALUDE_range_of_a_l3483_348378

/-- Proposition p -/
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

/-- The set of x satisfying proposition p -/
def A : Set ℝ := {x | p x}

/-- The set of x satisfying proposition q -/
def B (a : ℝ) : Set ℝ := {x | q x a}

/-- The condition that ¬p is a necessary but not sufficient condition for ¬q -/
def condition (a : ℝ) : Prop := A ⊂ B a ∧ A ≠ B a

/-- The theorem stating the range of a -/
theorem range_of_a : ∀ a : ℝ, condition a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3483_348378


namespace NUMINAMATH_CALUDE_side_significant_digits_l3483_348346

-- Define the area of the square
def area : ℝ := 0.6400

-- Define the precision of the area measurement
def area_precision : ℝ := 0.0001

-- Define the function to calculate the number of significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt area
  count_significant_digits side = 4 := by sorry

end NUMINAMATH_CALUDE_side_significant_digits_l3483_348346


namespace NUMINAMATH_CALUDE_total_lockers_l3483_348325

/-- Represents the layout of lockers in a school -/
structure LockerLayout where
  left : ℕ  -- Number of lockers to the left of Yunjeong's locker
  right : ℕ  -- Number of lockers to the right of Yunjeong's locker
  front : ℕ  -- Number of lockers in front of Yunjeong's locker
  back : ℕ  -- Number of lockers behind Yunjeong's locker

/-- Theorem stating the total number of lockers given Yunjeong's locker position -/
theorem total_lockers (layout : LockerLayout) : 
  layout.left = 6 → 
  layout.right = 12 → 
  layout.front = 7 → 
  layout.back = 13 → 
  (layout.left + 1 + layout.right) * (layout.front + 1 + layout.back) = 399 := by
  sorry

#check total_lockers

end NUMINAMATH_CALUDE_total_lockers_l3483_348325


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3483_348381

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3483_348381


namespace NUMINAMATH_CALUDE_mira_total_distance_l3483_348361

/-- Mira's jogging schedule for five days -/
structure JoggingSchedule where
  monday_speed : ℝ
  monday_time : ℝ
  tuesday_speed : ℝ
  tuesday_time : ℝ
  wednesday_speed : ℝ
  wednesday_time : ℝ
  thursday_speed : ℝ
  thursday_time : ℝ
  friday_speed : ℝ
  friday_time : ℝ

/-- Calculate the total distance jogged given a schedule -/
def total_distance (schedule : JoggingSchedule) : ℝ :=
  schedule.monday_speed * schedule.monday_time +
  schedule.tuesday_speed * schedule.tuesday_time +
  schedule.wednesday_speed * schedule.wednesday_time +
  schedule.thursday_speed * schedule.thursday_time +
  schedule.friday_speed * schedule.friday_time

/-- Mira's actual jogging schedule -/
def mira_schedule : JoggingSchedule := {
  monday_speed := 4
  monday_time := 2
  tuesday_speed := 5
  tuesday_time := 1.5
  wednesday_speed := 6
  wednesday_time := 2
  thursday_speed := 5
  thursday_time := 2.5
  friday_speed := 3
  friday_time := 1
}

/-- Theorem stating that Mira jogs a total of 43 miles in five days -/
theorem mira_total_distance : total_distance mira_schedule = 43 := by
  sorry

end NUMINAMATH_CALUDE_mira_total_distance_l3483_348361


namespace NUMINAMATH_CALUDE_range_of_a_l3483_348316

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Define the property that the quadratic is always positive
def always_positive (a : ℝ) : Prop := ∀ x : ℝ, f a x > 0

-- Theorem statement
theorem range_of_a : Set.Icc 0 3 = {a : ℝ | always_positive a} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3483_348316


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l3483_348397

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l3483_348397


namespace NUMINAMATH_CALUDE_sine_inequality_l3483_348330

theorem sine_inequality : 
  let sin60 := Real.sqrt 3 / 2
  let sin62 := (Real.sqrt 2 / 2) * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let sin64 := 2 * (Real.cos (13 * π / 180))^2 - 1
  sin60 < sin62 ∧ sin62 < sin64 := by sorry

end NUMINAMATH_CALUDE_sine_inequality_l3483_348330


namespace NUMINAMATH_CALUDE_solve_for_a_l3483_348341

theorem solve_for_a : ∀ x a : ℝ, 2 * x + a - 9 = 0 → x = 2 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3483_348341


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3483_348327

/-- Represents the cost of a window in dollars -/
def window_cost : ℕ := 100

/-- Represents the number of windows purchased to get free windows -/
def windows_for_offer : ℕ := 9

/-- Represents the number of free windows given in the offer -/
def free_windows : ℕ := 2

/-- Represents the number of windows Dave needs -/
def dave_windows : ℕ := 10

/-- Represents the number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculates the cost of purchasing windows with the special offer -/
def calculate_cost (num_windows : ℕ) : ℕ :=
  let paid_windows := num_windows - (num_windows / windows_for_offer) * free_windows
  paid_windows * window_cost

/-- Theorem stating that there are no savings when Dave and Doug purchase windows together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3483_348327


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3483_348384

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (5 * Real.sin (α/2)^2 + 8 * Real.sin (α/2) * Real.cos (α/2) + 11 * Real.cos (α/2)^2 - 8) / 
  (Real.sqrt 2 * Real.sin (α - Real.pi/4)) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3483_348384


namespace NUMINAMATH_CALUDE_container_capacity_l3483_348338

theorem container_capacity (x : ℝ) 
  (h1 : (1/4) * x + 300 = (3/4) * x) : x = 600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3483_348338


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l3483_348372

theorem ivy_collectors_edition_dolls 
  (dina_dolls : ℕ)
  (ivy_dolls : ℕ)
  (h1 : dina_dolls = 60)
  (h2 : dina_dolls = 2 * ivy_dolls)
  (h3 : ivy_dolls > 0)
  : (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l3483_348372


namespace NUMINAMATH_CALUDE_stratified_sample_grade10_l3483_348324

theorem stratified_sample_grade10 (total_sample : ℕ) (grade12 : ℕ) (grade11 : ℕ) (grade10 : ℕ) :
  total_sample = 50 →
  grade12 = 750 →
  grade11 = 850 →
  grade10 = 900 →
  (grade10 * total_sample) / (grade12 + grade11 + grade10) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_grade10_l3483_348324


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l3483_348382

/-- The post position -/
def post : ℝ × ℝ := (2, 5)

/-- The rope length -/
def rope_length : ℝ := 8

/-- The rectangle's vertices -/
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (0, 10), (10, 0), (10, 10)]

/-- Check if a point is within the rectangle -/
def in_rectangle (p : ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10

/-- Check if a point is within the rope's reach -/
def in_rope_reach (p : ℝ × ℝ) : Prop :=
  (p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2

/-- The maximum distance from origin theorem -/
theorem max_distance_from_origin :
  ∃ (p : ℝ × ℝ), in_rectangle p ∧ in_rope_reach p ∧
  ∀ (q : ℝ × ℝ), in_rectangle q → in_rope_reach q →
  p.1^2 + p.2^2 ≥ q.1^2 + q.2^2 ∧
  p.1^2 + p.2^2 = 125 :=
sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l3483_348382


namespace NUMINAMATH_CALUDE_minimum_seating_arrangement_l3483_348347

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  s.total_chairs % s.seated_people = 0

/-- Checks if any additional person must sit next to someone -/
def forces_adjacent_seating (s : CircularSeating) : Prop :=
  s.total_chairs / s.seated_people ≤ 4

/-- The main theorem to prove -/
theorem minimum_seating_arrangement :
  ∃ (s : CircularSeating), 
    s.total_chairs = 75 ∧
    is_valid_seating s ∧
    forces_adjacent_seating s ∧
    (∀ (t : CircularSeating), 
      t.total_chairs = 75 → 
      is_valid_seating t → 
      forces_adjacent_seating t → 
      s.seated_people ≤ t.seated_people) ∧
    s.seated_people = 19 :=
  sorry

end NUMINAMATH_CALUDE_minimum_seating_arrangement_l3483_348347


namespace NUMINAMATH_CALUDE_line_perpendicular_and_tangent_l3483_348343

/-- The given line -/
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

/-- The given curve -/
def given_curve (x y : ℝ) : Prop := y = x^3 + 3*x^2 - 5

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A line is tangent to a curve if it touches the curve at exactly one point -/
def tangent_to_curve (line : (ℝ → ℝ → Prop)) (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃! p : ℝ × ℝ, line p.1 p.2 ∧ curve p.1 p.2

theorem line_perpendicular_and_tangent :
  (∃ m₁ m₂ : ℝ, perpendicular m₁ m₂ ∧ 
    (∀ x y : ℝ, given_line x y → y = m₁*x + 1/6) ∧
    (∀ x y : ℝ, target_line x y → y = m₂*x - 2)) ∧
  tangent_to_curve target_line given_curve :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_and_tangent_l3483_348343


namespace NUMINAMATH_CALUDE_AB_product_l3483_348370

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 0, -2]
def B_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -1/2; 0, 2]

theorem AB_product :
  let B := B_inv⁻¹
  A * B = !![1, 5/4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_AB_product_l3483_348370


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l3483_348313

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), ∃ (k : ℕ), sum_of_digits (N + k * M) ≠ sum_of_digits N :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l3483_348313


namespace NUMINAMATH_CALUDE_tv_price_proof_l3483_348364

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem tv_price_proof (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  let total_price := a * 10000 + 6000 + 700 + 90 + b
  is_divisible_by total_price 72 →
  (total_price / 72 : ℚ) = 511 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_proof_l3483_348364


namespace NUMINAMATH_CALUDE_functional_equation_unique_solution_l3483_348331

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem functional_equation_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_unique_solution_l3483_348331


namespace NUMINAMATH_CALUDE_total_digits_100000_l3483_348391

def total_digits (n : ℕ) : ℕ :=
  let d1 := 9
  let d2 := 90 * 2
  let d3 := 900 * 3
  let d4 := 9000 * 4
  let d5 := (n - 10000 + 1) * 5
  let d6 := if n = 100000 then 6 else 0
  d1 + d2 + d3 + d4 + d5 + d6

theorem total_digits_100000 :
  total_digits 100000 = 488895 := by
  sorry

end NUMINAMATH_CALUDE_total_digits_100000_l3483_348391


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3483_348335

theorem fraction_inequality_solution_set (x : ℝ) :
  (x - 1) / (x - 3) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3483_348335


namespace NUMINAMATH_CALUDE_emma_milk_containers_l3483_348326

/-- The number of weeks Emma buys milk -/
def weeks : ℕ := 3

/-- The number of school days in a week -/
def school_days_per_week : ℕ := 5

/-- The total number of milk containers Emma buys in 3 weeks -/
def total_containers : ℕ := 30

/-- The number of containers Emma buys each school day -/
def containers_per_day : ℚ := total_containers / (weeks * school_days_per_week)

theorem emma_milk_containers : containers_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_emma_milk_containers_l3483_348326


namespace NUMINAMATH_CALUDE_tenth_thousand_digit_is_seven_l3483_348314

def digit_sequence (n : ℕ) : ℕ :=
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  let digits_1_to_999 := digits_1_to_9 + digits_10_to_99 + digits_100_to_999
  let remaining_digits := n - digits_1_to_999
  let full_numbers_1000_onward := remaining_digits / 4
  let digits_from_full_numbers := full_numbers_1000_onward * 4
  let last_number := 1000 + full_numbers_1000_onward
  let remaining_digits_in_last_number := remaining_digits - digits_from_full_numbers
  if remaining_digits_in_last_number = 0 then
    (last_number - 1) % 10
  else
    (last_number / (10 ^ (4 - remaining_digits_in_last_number))) % 10

theorem tenth_thousand_digit_is_seven :
  digit_sequence 10000 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tenth_thousand_digit_is_seven_l3483_348314


namespace NUMINAMATH_CALUDE_tims_balloons_l3483_348392

def dans_balloons : ℝ := 29.0
def dans_multiple : ℝ := 7.0

theorem tims_balloons : ⌊dans_balloons / dans_multiple⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_tims_balloons_l3483_348392


namespace NUMINAMATH_CALUDE_school_store_pricing_l3483_348393

/-- Given the cost of pencils and notebooks in a school store, 
    calculate the cost of a specific combination. -/
theorem school_store_pricing 
  (pencil_cost notebook_cost : ℚ) 
  (h1 : 6 * pencil_cost + 6 * notebook_cost = 390/100)
  (h2 : 8 * pencil_cost + 4 * notebook_cost = 328/100) : 
  20 * pencil_cost + 14 * notebook_cost = 1012/100 := by
  sorry

end NUMINAMATH_CALUDE_school_store_pricing_l3483_348393


namespace NUMINAMATH_CALUDE_dress_final_price_l3483_348374

/-- The final price of a dress after multiple discounts and tax -/
def finalPrice (d : ℝ) : ℝ :=
  let price1 := d * (1 - 0.45)  -- After first discount
  let price2 := price1 * (1 - 0.30)  -- After second discount
  let price3 := price2 * (1 - 0.25)  -- After third discount
  let price4 := price3 * (1 - 0.50)  -- After staff discount
  price4 * (1 + 0.10)  -- After sales tax

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) : finalPrice d = 0.1588125 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_final_price_l3483_348374


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_and_sum_constant_l3483_348389

/-- Represents the quadratic equation -3(x-1)^2 + m = 0 --/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  -3 * (x - 1)^2 + m = 0

/-- The discriminant of the quadratic equation --/
def discriminant (m : ℝ) : ℝ :=
  12 * m

theorem quadratic_roots_distinct_and_sum_constant (m : ℝ) (h : m > 0) :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    quadratic_equation m x₁ ∧
    quadratic_equation m x₂ ∧
    x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_and_sum_constant_l3483_348389


namespace NUMINAMATH_CALUDE_shortest_distance_theorem_l3483_348380

theorem shortest_distance_theorem (a b c : ℝ) :
  a = 8 ∧ b = 6 ∧ c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_theorem_l3483_348380


namespace NUMINAMATH_CALUDE_carpet_ratio_l3483_348363

theorem carpet_ratio (house1 house2 house3 total : ℕ) 
  (h1 : house1 = 12)
  (h2 : house2 = 20)
  (h3 : house3 = 10)
  (h_total : total = 62)
  (h_sum : house1 + house2 + house3 + (total - (house1 + house2 + house3)) = total) :
  (total - (house1 + house2 + house3)) / house3 = 2 := by
sorry

end NUMINAMATH_CALUDE_carpet_ratio_l3483_348363


namespace NUMINAMATH_CALUDE_factorial_ratio_l3483_348394

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3483_348394


namespace NUMINAMATH_CALUDE_preimage_of_2_neg4_l3483_348308

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem preimage_of_2_neg4 : f (-1, -3) = (2, -4) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_2_neg4_l3483_348308


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l3483_348332

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 3/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_7_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  a + 2*b = 7 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l3483_348332


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3483_348358

theorem smallest_angle_in_triangle (x y z : ℝ) (hx : x = 60) (hy : y = 70) 
  (hsum : x + y + z = 180) : min x (min y z) = 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3483_348358


namespace NUMINAMATH_CALUDE_exactly_two_out_of_four_probability_l3483_348309

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability mass function -/
def binomialPMF (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_out_of_four_probability :
  binomialPMF n k p = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_four_probability_l3483_348309


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3483_348348

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 6297 ∧ ∀ m : ℕ, m < 6297 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3483_348348


namespace NUMINAMATH_CALUDE_another_hamiltonian_cycle_l3483_348351

/-- A graph with n vertices where each vertex has exactly 3 neighbors -/
structure ThreeRegularGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  degree_three : ∀ v : Fin n, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- A Hamiltonian cycle in a graph -/
def HamiltonianCycle {n : ℕ} (G : ThreeRegularGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.toFinset = G.vertices }

/-- Two Hamiltonian cycles are equivalent if one can be obtained from the other by rotation or reflection -/
def EquivalentCycles {n : ℕ} (G : ThreeRegularGraph n) (c1 c2 : HamiltonianCycle G) : Prop :=
  ∃ (k : ℕ) (reflect : Bool),
    c2.val = if reflect then c1.val.reverse.rotateRight k else c1.val.rotateRight k

theorem another_hamiltonian_cycle {n : ℕ} (G : ThreeRegularGraph n) (c : HamiltonianCycle G) :
  ∃ (c' : HamiltonianCycle G), ¬EquivalentCycles G c c' :=
sorry

end NUMINAMATH_CALUDE_another_hamiltonian_cycle_l3483_348351


namespace NUMINAMATH_CALUDE_sixth_year_fee_l3483_348353

def membership_fee (initial_fee : ℕ) (yearly_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * yearly_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end NUMINAMATH_CALUDE_sixth_year_fee_l3483_348353


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l3483_348399

theorem mean_proportional_problem (n : ℝ) : (156 : ℝ) ^ 2 = n * 104 → n = 234 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l3483_348399


namespace NUMINAMATH_CALUDE_exist_three_integers_sum_zero_thirteenth_powers_square_l3483_348300

theorem exist_three_integers_sum_zero_thirteenth_powers_square :
  ∃ (a b c : ℤ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧  -- nonzero
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- pairwise distinct
    a + b + c = 0 ∧          -- sum is zero
    ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_exist_three_integers_sum_zero_thirteenth_powers_square_l3483_348300


namespace NUMINAMATH_CALUDE_sequence_max_value_l3483_348388

theorem sequence_max_value (n : ℤ) : -2 * n^2 + 29 * n + 3 ≤ 108 := by
  sorry

end NUMINAMATH_CALUDE_sequence_max_value_l3483_348388


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3483_348379

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3483_348379


namespace NUMINAMATH_CALUDE_sidney_cat_food_l3483_348321

/-- Represents the amount of food each adult cat eats per day -/
def adult_cat_food : ℝ := 1

theorem sidney_cat_food :
  let num_kittens : ℕ := 4
  let num_adult_cats : ℕ := 3
  let initial_food : ℕ := 7
  let kitten_food_per_day : ℚ := 3/4
  let additional_food : ℕ := 35
  let days : ℕ := 7
  
  (num_kittens : ℝ) * kitten_food_per_day * days +
  (num_adult_cats : ℝ) * adult_cat_food * days =
  (initial_food : ℝ) + additional_food :=
by sorry

#check sidney_cat_food

end NUMINAMATH_CALUDE_sidney_cat_food_l3483_348321


namespace NUMINAMATH_CALUDE_equation_solutions_l3483_348368

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x * y + y * z + z * x = 2 * (x + y + z)} =
  {(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (2, 2, 2), (4, 1, 2), (4, 2, 1)} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3483_348368


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3483_348357

theorem absolute_value_equation_solution (x : ℝ) : 
  |3*x - 2| + |3*x + 1| = 3 ↔ x = -2/3 ∨ (-1/3 < x ∧ x ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3483_348357


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3483_348305

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that the symmetric point of (2, -8) with respect to the y-axis is (-2, -8). -/
theorem symmetric_point_theorem :
  let A : Point := ⟨2, -8⟩
  let B : Point := ⟨-2, -8⟩
  symmetricAboutYAxis A B := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3483_348305
