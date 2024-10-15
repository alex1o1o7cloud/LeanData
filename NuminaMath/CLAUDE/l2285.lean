import Mathlib

namespace NUMINAMATH_CALUDE_triangle_altitude_l2285_228503

/-- Given a triangle with area 800 square feet and base 40 feet, its altitude is 40 feet. -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 800 → base = 40 → area = (1/2) * base * altitude → altitude = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2285_228503


namespace NUMINAMATH_CALUDE_min_perimeter_two_isosceles_triangles_l2285_228552

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.side : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 4

theorem min_perimeter_two_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 524 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_two_isosceles_triangles_l2285_228552


namespace NUMINAMATH_CALUDE_triangle_rds_area_l2285_228590

/-- The area of a triangle RDS with given coordinates and perpendicular sides -/
theorem triangle_rds_area (k : ℝ) : 
  let R : ℝ × ℝ := (0, 15)
  let D : ℝ × ℝ := (3, 15)
  let S : ℝ × ℝ := (0, k)
  -- RD is perpendicular to RS (implied by coordinates)
  (45 - 3 * k) / 2 = (1 / 2) * 3 * (15 - k) := by sorry

end NUMINAMATH_CALUDE_triangle_rds_area_l2285_228590


namespace NUMINAMATH_CALUDE_damage_cost_calculation_l2285_228557

def tire_cost (prices : List ℕ) (quantities : List ℕ) : ℕ :=
  List.sum (List.zipWith (· * ·) prices quantities)

def window_cost (prices : List ℕ) : ℕ :=
  List.sum prices

def fence_cost (plank_price : ℕ) (plank_quantity : ℕ) (labor_cost : ℕ) : ℕ :=
  plank_price * plank_quantity + labor_cost

theorem damage_cost_calculation (tire_prices : List ℕ) (tire_quantities : List ℕ)
    (window_prices : List ℕ) (paint_job_cost : ℕ)
    (fence_plank_price : ℕ) (fence_plank_quantity : ℕ) (fence_labor_cost : ℕ) :
    tire_prices = [230, 250, 280] →
    tire_quantities = [2, 2, 2] →
    window_prices = [700, 800, 900] →
    paint_job_cost = 1200 →
    fence_plank_price = 35 →
    fence_plank_quantity = 5 →
    fence_labor_cost = 150 →
    tire_cost tire_prices tire_quantities +
    window_cost window_prices +
    paint_job_cost +
    fence_cost fence_plank_price fence_plank_quantity fence_labor_cost = 5445 := by
  sorry

end NUMINAMATH_CALUDE_damage_cost_calculation_l2285_228557


namespace NUMINAMATH_CALUDE_albert_run_distance_l2285_228518

/-- Calculates the total distance run on a circular track -/
def totalDistance (trackLength : ℕ) (lapsRun : ℕ) (additionalLaps : ℕ) : ℕ :=
  trackLength * (lapsRun + additionalLaps)

/-- Proves that running 11 laps on a 9-meter track results in 99 meters total distance -/
theorem albert_run_distance :
  totalDistance 9 6 5 = 99 := by
  sorry

end NUMINAMATH_CALUDE_albert_run_distance_l2285_228518


namespace NUMINAMATH_CALUDE_more_likely_same_l2285_228523

/-- Represents the number of crows on each tree -/
structure CrowCounts where
  white_birch : ℕ
  black_birch : ℕ
  white_oak : ℕ
  black_oak : ℕ

/-- Conditions from the problem -/
def valid_crow_counts (c : CrowCounts) : Prop :=
  c.white_birch > 0 ∧
  c.white_birch + c.black_birch = 50 ∧
  c.white_oak + c.black_oak = 50 ∧
  c.black_birch ≥ c.white_birch ∧
  c.black_oak ≥ c.white_oak - 1

/-- Probability of number of white crows on birch remaining the same -/
def prob_same (c : CrowCounts) : ℚ :=
  (c.black_birch * (c.black_oak + 1) + c.white_birch * (c.white_oak + 1)) / 2550

/-- Probability of number of white crows on birch changing -/
def prob_change (c : CrowCounts) : ℚ :=
  (c.black_birch * c.white_oak + c.white_birch * c.black_oak) / 2550

/-- Theorem stating that it's more likely for the number of white crows to remain the same -/
theorem more_likely_same (c : CrowCounts) (h : valid_crow_counts c) :
  prob_same c > prob_change c :=
sorry

end NUMINAMATH_CALUDE_more_likely_same_l2285_228523


namespace NUMINAMATH_CALUDE_points_earned_proof_l2285_228578

def video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem points_earned_proof :
  video_game_points 8 5 6 = 10 := by
sorry

end NUMINAMATH_CALUDE_points_earned_proof_l2285_228578


namespace NUMINAMATH_CALUDE_problem_solution_l2285_228535

theorem problem_solution : (2010^2 - 2010) / 2010 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2285_228535


namespace NUMINAMATH_CALUDE_three_digit_number_relation_l2285_228500

theorem three_digit_number_relation (h t u : ℕ) : 
  h ≥ 1 ∧ h ≤ 9 ∧  -- h is a single digit
  t ≥ 0 ∧ t ≤ 9 ∧  -- t is a single digit
  u ≥ 0 ∧ u ≤ 9 ∧  -- u is a single digit
  h = t + 2 ∧      -- hundreds digit is 2 more than tens digit
  h + t + u = 27   -- sum of digits is 27
  → ∃ (r : ℕ → ℕ → Prop), r t u  -- there exists some relation r between t and u
:= by sorry

end NUMINAMATH_CALUDE_three_digit_number_relation_l2285_228500


namespace NUMINAMATH_CALUDE_curve_is_circle_l2285_228510

/-- The curve represented by the equation |x-1| = √(1-(y+1)²) -/
def curve_equation (x y : ℝ) : Prop := |x - 1| = Real.sqrt (1 - (y + 1)^2)

/-- The equation of a circle with center (1, -1) and radius 1 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

/-- Theorem stating that the curve equation represents a circle -/
theorem curve_is_circle :
  ∀ x y : ℝ, curve_equation x y ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_curve_is_circle_l2285_228510


namespace NUMINAMATH_CALUDE_lines_dont_form_triangle_iff_l2285_228587

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines given in the problem -/
def line1 : Line := ⟨4, 1, 4⟩
def line2 (m : ℝ) : Line := ⟨m, 1, 0⟩
def line3 (m : ℝ) : Line := ⟨2, -3*m, 4⟩

/-- The condition for the lines not forming a triangle -/
def lines_dont_form_triangle (m : ℝ) : Prop :=
  are_parallel line1 (line2 m) ∨ 
  are_parallel line1 (line3 m) ∨ 
  are_parallel (line2 m) (line3 m)

theorem lines_dont_form_triangle_iff (m : ℝ) : 
  lines_dont_form_triangle m ↔ m = 4 ∨ m = -1/6 := by sorry

end NUMINAMATH_CALUDE_lines_dont_form_triangle_iff_l2285_228587


namespace NUMINAMATH_CALUDE_baseball_hits_percentage_l2285_228546

theorem baseball_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_baseball_hits_percentage_l2285_228546


namespace NUMINAMATH_CALUDE_hyperbola_center_l2285_228543

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (5, 0) → f2 = (9, 4) → center = (7, 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2285_228543


namespace NUMINAMATH_CALUDE_money_distribution_l2285_228545

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2285_228545


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2285_228529

/-- An equilateral triangle with perimeter 15 meters has sides of length 5 meters. -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) : 
  perimeter = 15 → 
  (∃ side : ℝ, side > 0 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side) ∧ 
    3 * side = perimeter) → 
  (∃ side : ℝ, side = 5 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side)) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2285_228529


namespace NUMINAMATH_CALUDE_log_343_property_l2285_228551

theorem log_343_property (x : ℝ) (h : Real.log (343 : ℝ) / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ)) ∧ 
  (∀ (n : ℕ), n ≥ 2 → ¬∃ (m : ℤ), x = (m : ℝ) ^ (1 / n : ℝ)) ∧
  (¬∃ (n : ℤ), x = (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_log_343_property_l2285_228551


namespace NUMINAMATH_CALUDE_find_b_plus_c_l2285_228548

theorem find_b_plus_c (a b c d : ℚ)
  (eq1 : a * b + a * c + b * d + c * d = 40)
  (eq2 : a + d = 6)
  (eq3 : a * b + b * c + c * d + d * a = 28) :
  b + c = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_b_plus_c_l2285_228548


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_equals_four_l2285_228581

theorem sqrt_neg_four_squared_equals_four : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_equals_four_l2285_228581


namespace NUMINAMATH_CALUDE_cookie_theorem_l2285_228544

def cookie_problem (initial_cookies : ℕ) (given_to_friend : ℕ) (eaten : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - given_to_friend
  let given_to_family := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - given_to_family
  remaining_after_family - eaten

theorem cookie_theorem : cookie_problem 19 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l2285_228544


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_100_l2285_228567

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_100 : sum_factorials 100 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_100_l2285_228567


namespace NUMINAMATH_CALUDE_union_M_N_when_a_9_M_superset_N_iff_a_range_l2285_228507

-- Define the sets M and N
def M : Set ℝ := {x | (x + 5) / (x - 8) ≥ 0}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for part 1
theorem union_M_N_when_a_9 :
  M ∪ N 9 = {x : ℝ | x ≤ -5 ∨ x ≥ 8} := by sorry

-- Theorem for part 2
theorem M_superset_N_iff_a_range (a : ℝ) :
  M ⊇ N a ↔ a ≤ -6 ∨ a > 9 := by sorry

end NUMINAMATH_CALUDE_union_M_N_when_a_9_M_superset_N_iff_a_range_l2285_228507


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2285_228531

-- Define the set difference operation
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetricDifference (M N : Set ℝ) : Set ℝ := (setDifference M N) ∪ (setDifference N M)

-- Define sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x < -9/4 ∨ x ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2285_228531


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l2285_228570

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundays * sundayVisitors + otherDays * otherDayVisitors
  (totalVisitors : ℚ) / totalDays

/-- Theorem: The average number of visitors per day is 276 -/
theorem average_visitors_is_276 :
  averageVisitorsPerDay 510 240 = 276 := by
  sorry


end NUMINAMATH_CALUDE_average_visitors_is_276_l2285_228570


namespace NUMINAMATH_CALUDE_positive_sum_one_inequality_l2285_228509

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_one_inequality_l2285_228509


namespace NUMINAMATH_CALUDE_eighth_term_is_22_n_equals_8_when_an_is_22_l2285_228525

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 8th term of the sequence is 22 -/
theorem eighth_term_is_22 : arithmeticSequence 8 = 22 := by
  sorry

/-- Theorem proving that if the nth term is 22, then n must be 8 -/
theorem n_equals_8_when_an_is_22 (n : ℕ) (h : arithmeticSequence n = 22) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_22_n_equals_8_when_an_is_22_l2285_228525


namespace NUMINAMATH_CALUDE_f_increasing_range_of_a_l2285_228532

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - a/2)*x + 2

/-- The theorem stating the range of values for a -/
theorem f_increasing_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (8/3) 4 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_of_a_l2285_228532


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l2285_228558

theorem boys_ratio_in_class (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (n : ℚ) / (n + m : ℚ) = 2 / 5 ↔ 
  (n : ℚ) / (n + m : ℚ) = 2 / 3 * (m : ℚ) / (n + m : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l2285_228558


namespace NUMINAMATH_CALUDE_no_natural_solution_l2285_228527

theorem no_natural_solution : ∀ x : ℕ, 19 * x^2 + 97 * x ≠ 1997 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2285_228527


namespace NUMINAMATH_CALUDE_total_pigeons_l2285_228589

def initial_pigeons : ℕ := 1
def joined_pigeons : ℕ := 1

theorem total_pigeons : initial_pigeons + joined_pigeons = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_pigeons_l2285_228589


namespace NUMINAMATH_CALUDE_problem_solution_l2285_228556

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 * y = 2)
  (h2 : y^2 * z = 4)
  (h3 : z^2 / x = 5) :
  x = 5^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2285_228556


namespace NUMINAMATH_CALUDE_binomial_distributions_l2285_228584

/-- A random variable follows a binomial distribution if it represents the number of successes
    in a fixed number of independent Bernoulli trials with the same probability of success. -/
def IsBinomialDistribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability mass function for the number of shots needed to hit the target for the first time -/
def GeometricDistribution (p : ℝ) (X : ℕ → ℝ) : Prop :=
  0 < p ∧ p ≤ 1 ∧ ∀ k, k > 0 → X k = (1-p)^(k-1) * p

/-- The distribution of computer virus infections -/
def VirusInfection (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of hitting a target in n shots -/
def TargetHits (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of cars refueling at a gas station -/
def CarRefueling (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

theorem binomial_distributions (n : ℕ) (p : ℝ) (X₁ X₂ X₃ X₄ : ℕ → ℝ) :
  VirusInfection n X₁ ∧
  GeometricDistribution p X₂ ∧
  TargetHits n X₃ ∧
  CarRefueling X₄ →
  IsBinomialDistribution X₁ ∧
  ¬IsBinomialDistribution X₂ ∧
  IsBinomialDistribution X₃ ∧
  IsBinomialDistribution X₄ :=
sorry

end NUMINAMATH_CALUDE_binomial_distributions_l2285_228584


namespace NUMINAMATH_CALUDE_original_price_calculation_l2285_228533

-- Define the original cost price as a real number
variable (P : ℝ)

-- Define the selling price
def selling_price : ℝ := 1800

-- Define the sequence of operations on the price
def price_after_operations (original_price : ℝ) : ℝ :=
  original_price * 0.90 * 1.05 * 1.12 * 0.85

-- Define the final selling price with profit
def final_price (original_price : ℝ) : ℝ :=
  price_after_operations original_price * 1.20

-- Theorem stating the relationship between original price and selling price
theorem original_price_calculation :
  final_price P = selling_price :=
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2285_228533


namespace NUMINAMATH_CALUDE_pq_length_is_25_over_3_l2285_228569

/-- Triangle DEF with given side lengths and a parallel segment PQ on DE -/
structure TriangleWithParallelSegment where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FD -/
  fd : ℝ
  /-- Length of segment PQ -/
  pq : ℝ
  /-- PQ is parallel to EF -/
  pq_parallel_ef : Bool
  /-- PQ is on DE -/
  pq_on_de : Bool
  /-- PQ is one-third of DE -/
  pq_is_third_of_de : pq = de / 3

/-- The length of PQ in the given triangle configuration is 25/3 -/
theorem pq_length_is_25_over_3 (t : TriangleWithParallelSegment)
  (h_de : t.de = 25)
  (h_ef : t.ef = 29)
  (h_fd : t.fd = 32)
  (h_pq_parallel : t.pq_parallel_ef = true)
  (h_pq_on_de : t.pq_on_de = true) :
  t.pq = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_is_25_over_3_l2285_228569


namespace NUMINAMATH_CALUDE_place_value_comparison_l2285_228562

def number : ℚ := 52648.2097

def tens_place_value : ℚ := 10
def tenths_place_value : ℚ := 0.1

theorem place_value_comparison : 
  tens_place_value / tenths_place_value = 100 := by sorry

end NUMINAMATH_CALUDE_place_value_comparison_l2285_228562


namespace NUMINAMATH_CALUDE_trains_meeting_point_l2285_228572

/-- The speed of the Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The speed of the Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The meeting point of the two trains -/
def meeting_point : ℝ := 480

theorem trains_meeting_point :
  ∃ t : ℝ, t > 0 ∧ bombay_speed * (t + time_difference) = rajdhani_speed * t ∧
  rajdhani_speed * t = meeting_point := by sorry

end NUMINAMATH_CALUDE_trains_meeting_point_l2285_228572


namespace NUMINAMATH_CALUDE_forgotten_poems_sally_forgotten_poems_l2285_228530

/-- Given the number of initially memorized poems and the number of poems that can be recited,
    prove that the number of forgotten poems is their difference. -/
theorem forgotten_poems (initially_memorized recitable : ℕ) :
  initially_memorized ≥ recitable →
  initially_memorized - recitable = initially_memorized - recitable :=
by
  sorry

/-- Application to Sally's specific case -/
theorem sally_forgotten_poems :
  let initially_memorized := 8
  let recitable := 3
  initially_memorized - recitable = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_forgotten_poems_sally_forgotten_poems_l2285_228530


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2285_228547

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. --/
def isArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℤ) (d : ℤ) (h : isArithmeticSequence a d)
  (h2 : a 2 = 12) (hn : a n = -20) (hd : d = -2) :
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2285_228547


namespace NUMINAMATH_CALUDE_robot_types_count_l2285_228516

theorem robot_types_count (shapes : ℕ) (colors : ℕ) (h1 : shapes = 3) (h2 : colors = 4) :
  shapes * colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_robot_types_count_l2285_228516


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l2285_228508

/-- The number of schools -/
def num_schools : ℕ := 4

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def other_representatives : ℕ := 1

/-- The total number of members in the club -/
def total_members : ℕ := num_schools * members_per_school

/-- The number of ways to arrange the presidency meeting -/
def meeting_arrangements : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose other_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements : 
  meeting_arrangements = 17280 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l2285_228508


namespace NUMINAMATH_CALUDE_min_abs_z_plus_2i_l2285_228540

-- Define the complex number z
variable (z : ℂ)

-- Define the condition from the problem
def condition (z : ℂ) : Prop := Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*Complex.I))

-- State the theorem
theorem min_abs_z_plus_2i :
  (∀ z, condition z → Complex.abs (z + 2*Complex.I) ≥ 5/2) ∧
  (∃ z, condition z ∧ Complex.abs (z + 2*Complex.I) = 5/2) :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_2i_l2285_228540


namespace NUMINAMATH_CALUDE_mixed_fruit_cost_calculation_l2285_228536

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 34

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22.666666666666668

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost : ℝ := 264.1764705882353

theorem mixed_fruit_cost_calculation :
  mixed_fruit_cost * mixed_fruit_volume + acai_cost * acai_volume = 
  cocktail_cost * (mixed_fruit_volume + acai_volume) := by sorry

end NUMINAMATH_CALUDE_mixed_fruit_cost_calculation_l2285_228536


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2285_228564

def numbers : List ℝ := [17, 25, 38]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2285_228564


namespace NUMINAMATH_CALUDE_max_nondegenerate_triangles_l2285_228542

/-- Represents a triangle with colored sides -/
structure ColoredTriangle where
  blue : ℝ
  red : ℝ
  white : ℝ
  is_nondegenerate : blue + red > white ∧ blue + white > red ∧ red + white > blue

/-- The number of triangles -/
def num_triangles : ℕ := 2009

/-- A collection of 2009 non-degenerated triangles with colored sides -/
def triangle_collection : Fin num_triangles → ColoredTriangle := sorry

/-- Sorted blue sides -/
def sorted_blue : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).blue

/-- Sorted red sides -/
def sorted_red : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).red

/-- Sorted white sides -/
def sorted_white : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).white

/-- Sides are sorted in non-decreasing order -/
axiom sides_sorted : 
  (∀ i j, i ≤ j → sorted_blue i ≤ sorted_blue j) ∧
  (∀ i j, i ≤ j → sorted_red i ≤ sorted_red j) ∧
  (∀ i j, i ≤ j → sorted_white i ≤ sorted_white j)

/-- The main theorem: The maximum number of indices for which we can form non-degenerated triangles is 2009 -/
theorem max_nondegenerate_triangles : 
  (∃ f : Fin num_triangles → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) ∧
  (∀ k > num_triangles, ¬∃ f : Fin k → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) :=
by sorry

end NUMINAMATH_CALUDE_max_nondegenerate_triangles_l2285_228542


namespace NUMINAMATH_CALUDE_flag_making_problem_l2285_228566

/-- The number of students in each group making flags -/
def students_per_group : ℕ := 10

/-- The total number of flags to be made -/
def total_flags : ℕ := 240

/-- The number of groups initially assigned to make flags -/
def initial_groups : ℕ := 3

/-- The number of groups after reassignment -/
def final_groups : ℕ := 2

/-- The additional number of flags each student has to make after reassignment -/
def additional_flags_per_student : ℕ := 4

theorem flag_making_problem :
  (total_flags / final_groups - total_flags / initial_groups) / students_per_group = additional_flags_per_student :=
by sorry

end NUMINAMATH_CALUDE_flag_making_problem_l2285_228566


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2285_228524

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + |x - 7|

-- Define the interval [3, 9]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

-- State the theorem
theorem sum_of_max_min_g : 
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ I, g x ≤ max_val) ∧
    (∃ x ∈ I, g x = max_val) ∧
    (∀ x ∈ I, min_val ≤ g x) ∧
    (∃ x ∈ I, g x = min_val) ∧
    max_val + min_val = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2285_228524


namespace NUMINAMATH_CALUDE_f_has_two_roots_l2285_228534

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_roots_l2285_228534


namespace NUMINAMATH_CALUDE_solve_equation_l2285_228593

theorem solve_equation : ∃! y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) ∧ y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2285_228593


namespace NUMINAMATH_CALUDE_sine_equality_l2285_228573

/-- Given three nonzero real numbers a, b, c and three real angles α, β, γ,
    if a sin α + b sin β + c sin γ = 0 and a cos α + b cos β + c cos γ = 0,
    then sin(β - γ)/a = sin(γ - α)/b = sin(α - β)/c -/
theorem sine_equality (a b c α β γ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * Real.sin α + b * Real.sin β + c * Real.sin γ = 0)
  (h2 : a * Real.cos α + b * Real.cos β + c * Real.cos γ = 0) :
  Real.sin (β - γ) / a = Real.sin (γ - α) / b ∧ 
  Real.sin (γ - α) / b = Real.sin (α - β) / c :=
by sorry

end NUMINAMATH_CALUDE_sine_equality_l2285_228573


namespace NUMINAMATH_CALUDE_effective_area_percentage_difference_l2285_228576

/-- Calculates the effective area percentage difference between two circular fields -/
theorem effective_area_percentage_difference
  (r1 r2 : ℝ)  -- radii of the two fields
  (sqi1 sqi2 : ℝ)  -- soil quality indices
  (wa1 wa2 : ℝ)  -- water allocations
  (cyf1 cyf2 : ℝ)  -- crop yield factors
  (h_ratio : r2 = (10 / 4) * r1)  -- radius ratio condition
  (h_sqi1 : sqi1 = 0.8)
  (h_sqi2 : sqi2 = 1.2)
  (h_wa1 : wa1 = 15000)
  (h_wa2 : wa2 = 30000)
  (h_cyf1 : cyf1 = 1.5)
  (h_cyf2 : cyf2 = 2) :
  let ea1 := π * r1^2 * sqi1 * wa1 * cyf1
  let ea2 := π * r2^2 * sqi2 * wa2 * cyf2
  (ea2 - ea1) / ea1 * 100 = 1566.67 := by
  sorry

end NUMINAMATH_CALUDE_effective_area_percentage_difference_l2285_228576


namespace NUMINAMATH_CALUDE_chicken_wing_distribution_l2285_228521

theorem chicken_wing_distribution (total_wings : ℕ) (num_people : ℕ) 
  (h1 : total_wings = 35) (h2 : num_people = 12) :
  let wings_per_person := total_wings / num_people
  let leftover_wings := total_wings % num_people
  wings_per_person = 2 ∧ leftover_wings = 11 := by
  sorry

end NUMINAMATH_CALUDE_chicken_wing_distribution_l2285_228521


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2285_228595

/-- Calculates the total number of heartbeats during an athlete's activity --/
def totalHeartbeats (joggingHeartRate walkingHeartRate : ℕ) 
                    (walkingDuration : ℕ) 
                    (joggingDistance joggingPace : ℕ) : ℕ :=
  let joggingDuration := joggingDistance * joggingPace
  let joggingBeats := joggingDuration * joggingHeartRate
  let walkingBeats := walkingDuration * walkingHeartRate
  joggingBeats + walkingBeats

/-- Proves that the total number of heartbeats is 9900 given the specified conditions --/
theorem athlete_heartbeats :
  totalHeartbeats 120 90 30 10 6 = 9900 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l2285_228595


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l2285_228555

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ n : ℕ, n > 0 ∧ digit_sum n = 1000 ∧ digit_sum (n^2) = 1000^2 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l2285_228555


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2285_228577

/-- Given two planar vectors a and b, prove that the magnitude of (a - 2b) is 5. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (1, 2) →
  ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2285_228577


namespace NUMINAMATH_CALUDE_retail_overhead_expenses_l2285_228586

/-- A problem about calculating overhead expenses in retail --/
theorem retail_overhead_expenses 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percent : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : selling_price = 300)
  (h3 : profit_percent = 25) :
  ∃ (overhead_expenses : ℝ),
    selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100) ∧
    overhead_expenses = 15 := by
  sorry

end NUMINAMATH_CALUDE_retail_overhead_expenses_l2285_228586


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2285_228598

-- Define the discriminant function for a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- State the theorem
theorem quadratic_discriminant :
  discriminant 1 3 1 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2285_228598


namespace NUMINAMATH_CALUDE_least_whole_number_for_ratio_l2285_228585

theorem least_whole_number_for_ratio : 
  ∃ x : ℕ, x > 0 ∧ 
    (∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
    (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_least_whole_number_for_ratio_l2285_228585


namespace NUMINAMATH_CALUDE_train_length_l2285_228501

/-- The length of a train given its relative speed and passing time -/
theorem train_length (relative_speed : ℝ) (passing_time : ℝ) : 
  relative_speed = 72 - 36 →
  passing_time = 12 →
  relative_speed * (1000 / 3600) * passing_time = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2285_228501


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2285_228582

theorem triangle_is_equilateral (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Angle A is acute
  3 * b = 2 * Real.sqrt 3 * a * Real.sin B →  -- Given equation
  Real.cos B = Real.cos C →  -- Given condition
  0 < B ∧ B < π →  -- B is a valid angle
  0 < C ∧ C < π →  -- C is a valid angle
  A + B + C = π →  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3  -- Equilateral triangle
  := by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2285_228582


namespace NUMINAMATH_CALUDE_total_frogs_in_pond_l2285_228579

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_in_pond_l2285_228579


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2285_228511

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 199 := by
  sorry

#check fifteenth_term_of_sequence

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2285_228511


namespace NUMINAMATH_CALUDE_simplify_expression_l2285_228559

theorem simplify_expression :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2)) / (Real.sqrt (x^2 - 4*x*Real.sqrt 2 + 8)) -
               (Real.sqrt (x + 2 * Real.sqrt 2)) / (Real.sqrt (x^2 + 4*x*Real.sqrt 2 + 8))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2285_228559


namespace NUMINAMATH_CALUDE_cube_root_problem_l2285_228502

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2285_228502


namespace NUMINAMATH_CALUDE_six_steps_position_l2285_228520

/-- Given a number line with equally spaced markings where 8 steps cover 48 units,
    prove that 6 steps from 0 reach position 36. -/
theorem six_steps_position (total_distance : ℕ) (total_steps : ℕ) (steps : ℕ) :
  total_distance = 48 →
  total_steps = 8 →
  steps = 6 →
  (total_distance / total_steps) * steps = 36 := by
  sorry

end NUMINAMATH_CALUDE_six_steps_position_l2285_228520


namespace NUMINAMATH_CALUDE_f_max_min_values_l2285_228565

-- Define the function f(x) = |x-2| + |x-3| - |x-1|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Define the condition that |x-2| + |x-3| is minimized
def is_minimized (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem f_max_min_values :
  (∃ (x : ℝ), is_minimized x) →
  (∃ (max min : ℝ), 
    (∀ (y : ℝ), is_minimized y → f y ≤ max) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = max) ∧
    (∀ (y : ℝ), is_minimized y → min ≤ f y) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = min) ∧
    max = 0 ∧ min = -1) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_values_l2285_228565


namespace NUMINAMATH_CALUDE_air_inhaled_24_hours_l2285_228596

/-- The volume of air inhaled in 24 hours given the breathing rate and volume per breath -/
theorem air_inhaled_24_hours 
  (breaths_per_minute : ℕ) 
  (air_per_breath : ℚ) 
  (h1 : breaths_per_minute = 17) 
  (h2 : air_per_breath = 5/9) : 
  (breaths_per_minute : ℚ) * air_per_breath * (24 * 60) = 13600 := by
  sorry

end NUMINAMATH_CALUDE_air_inhaled_24_hours_l2285_228596


namespace NUMINAMATH_CALUDE_prob_red_then_blue_is_one_thirteenth_l2285_228575

def total_marbles : ℕ := 4 + 3 + 6

def red_marbles : ℕ := 4
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def prob_red_then_blue : ℚ := (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1)

theorem prob_red_then_blue_is_one_thirteenth :
  prob_red_then_blue = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_blue_is_one_thirteenth_l2285_228575


namespace NUMINAMATH_CALUDE_average_math_score_l2285_228553

/-- Represents the total number of students -/
def total_students : ℕ := 500

/-- Represents the number of male students -/
def male_students : ℕ := 300

/-- Represents the number of female students -/
def female_students : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 60

/-- Represents the average score of male students in the sample -/
def male_avg_score : ℝ := 110

/-- Represents the average score of female students in the sample -/
def female_avg_score : ℝ := 100

/-- Theorem stating that the average math score of first-year students is 106 points -/
theorem average_math_score : 
  (male_students : ℝ) / total_students * male_avg_score + 
  (female_students : ℝ) / total_students * female_avg_score = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_math_score_l2285_228553


namespace NUMINAMATH_CALUDE_circle_center_l2285_228599

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = 16

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = 16 is (4, -2) -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 4 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2285_228599


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2285_228591

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ Bᶜ = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2285_228591


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2285_228568

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ k : ℤ, x n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2285_228568


namespace NUMINAMATH_CALUDE_figure_placement_count_l2285_228528

/-- Represents a configuration of figure placements -/
structure FigurePlacement where
  pages : Fin 6 → Fin 3
  order_preserved : ∀ i j : Fin 4, i < j → pages i ≤ pages j

/-- The number of valid figure placements -/
def count_placements : ℕ := sorry

/-- Theorem stating the correct number of placements -/
theorem figure_placement_count : count_placements = 225 := by sorry

end NUMINAMATH_CALUDE_figure_placement_count_l2285_228528


namespace NUMINAMATH_CALUDE_golf_carts_needed_l2285_228537

theorem golf_carts_needed (patrons_per_cart : ℕ) (car_patrons : ℕ) (bus_patrons : ℕ) : 
  patrons_per_cart = 3 →
  car_patrons = 12 →
  bus_patrons = 27 →
  ((car_patrons + bus_patrons) + patrons_per_cart - 1) / patrons_per_cart = 13 := by
sorry

end NUMINAMATH_CALUDE_golf_carts_needed_l2285_228537


namespace NUMINAMATH_CALUDE_line_outside_circle_l2285_228506

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- A line with a given distance from a point -/
structure Line where
  distanceFromPoint : ℝ

/-- Relationship between a line and a circle -/
inductive Relationship
  | inside
  | tangent
  | outside

/-- Function to determine the relationship between a line and a circle -/
def relationshipBetweenLineAndCircle (c : Circle) (l : Line) : Relationship :=
  sorry

/-- Theorem stating that a line is outside a circle under given conditions -/
theorem line_outside_circle (c : Circle) (l : Line) 
  (h1 : c.diameter = 4)
  (h2 : l.distanceFromPoint = 3) :
  relationshipBetweenLineAndCircle c l = Relationship.outside :=
sorry

end NUMINAMATH_CALUDE_line_outside_circle_l2285_228506


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l2285_228514

/-- The line y = x + b is tangent to the curve x = √(1 - y²) if and only if b = -√2 -/
theorem line_tangent_to_curve (b : ℝ) : 
  (∀ x y : ℝ, y = x + b ∧ x = Real.sqrt (1 - y^2) → 
    (∃! p : ℝ × ℝ, p.1 = Real.sqrt (1 - p.2^2) ∧ p.2 = p.1 + b)) ↔ 
  b = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l2285_228514


namespace NUMINAMATH_CALUDE_milestone_solution_l2285_228594

def milestone_problem (initial_number : ℕ) (second_number : ℕ) (third_number : ℕ) : Prop :=
  let a := initial_number / 10
  let b := initial_number % 10
  (initial_number = 10 * a + b) ∧
  (second_number = 10 * b + a) ∧
  (third_number = 100 * a + b) ∧
  (0 < a) ∧ (a < 10) ∧ (0 < b) ∧ (b < 10)

theorem milestone_solution :
  ∃ (initial_number second_number : ℕ),
    milestone_problem initial_number second_number 106 :=
  sorry

end NUMINAMATH_CALUDE_milestone_solution_l2285_228594


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2285_228550

theorem smallest_number_divisible (n : ℕ) : n = 1009 ↔ 
  (∀ m : ℕ, m < n → ¬(12 ∣ (m - 2) ∧ 16 ∣ (m - 2) ∧ 18 ∣ (m - 2) ∧ 21 ∣ (m - 2) ∧ 28 ∣ (m - 2))) ∧
  (12 ∣ (n - 2) ∧ 16 ∣ (n - 2) ∧ 18 ∣ (n - 2) ∧ 21 ∣ (n - 2) ∧ 28 ∣ (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2285_228550


namespace NUMINAMATH_CALUDE_ramanujan_hardy_complex_game_l2285_228504

theorem ramanujan_hardy_complex_game (product h r : ℂ) : 
  product = 24 - 10*I ∧ h = 3 + 4*I ∧ product = h * r →
  r = 112/25 - 126/25*I := by sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_complex_game_l2285_228504


namespace NUMINAMATH_CALUDE_person_A_silver_sheets_l2285_228517

-- Define the exchange rates
def red_to_gold_rate : ℚ := 5 / 2
def gold_to_red_and_silver_rate : ℚ := 1

-- Define the initial number of sheets
def initial_red_sheets : ℕ := 3
def initial_gold_sheets : ℕ := 3

-- Define the function to calculate the total silver sheets
def total_silver_sheets : ℕ :=
  let gold_to_silver := initial_gold_sheets
  let red_to_silver := (initial_red_sheets + initial_gold_sheets) / 3 * 2
  gold_to_silver + red_to_silver

-- Theorem statement
theorem person_A_silver_sheets :
  total_silver_sheets = 7 :=
sorry

end NUMINAMATH_CALUDE_person_A_silver_sheets_l2285_228517


namespace NUMINAMATH_CALUDE_chess_piece_loss_l2285_228541

theorem chess_piece_loss (total_pieces : ℕ) (arianna_lost : ℕ) : 
  total_pieces = 20 →
  arianna_lost = 3 →
  32 - total_pieces = arianna_lost + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_piece_loss_l2285_228541


namespace NUMINAMATH_CALUDE_calculate_expression_l2285_228519

theorem calculate_expression : (1/2)⁻¹ + |3 - Real.sqrt 12| + (-1)^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2285_228519


namespace NUMINAMATH_CALUDE_parking_garage_has_four_stories_l2285_228597

/-- Represents a parking garage with the given specifications -/
structure ParkingGarage where
  spots_per_level : ℕ
  open_spots_level1 : ℕ
  open_spots_level2 : ℕ
  open_spots_level3 : ℕ
  open_spots_level4 : ℕ
  full_spots_total : ℕ

/-- Calculates the number of stories in the parking garage -/
def number_of_stories (garage : ParkingGarage) : ℕ :=
  (garage.full_spots_total + garage.open_spots_level1 + garage.open_spots_level2 +
   garage.open_spots_level3 + garage.open_spots_level4) / garage.spots_per_level

/-- Theorem stating that the parking garage has exactly 4 stories -/
theorem parking_garage_has_four_stories (garage : ParkingGarage) :
  garage.spots_per_level = 100 ∧
  garage.open_spots_level1 = 58 ∧
  garage.open_spots_level2 = garage.open_spots_level1 + 2 ∧
  garage.open_spots_level3 = garage.open_spots_level2 + 5 ∧
  garage.open_spots_level4 = 31 ∧
  garage.full_spots_total = 186 →
  number_of_stories garage = 4 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_has_four_stories_l2285_228597


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2285_228563

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 2)
  (h2 : seq.S 3 = 12) :
  seq.a 5 = 10 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2285_228563


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l2285_228513

theorem whole_number_between_bounds (M : ℤ) : 9 < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9.5 → M = 37 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l2285_228513


namespace NUMINAMATH_CALUDE_days_from_thursday_l2285_228580

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (advance_days start m)

theorem days_from_thursday :
  advance_days DayOfWeek.Thursday 53 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_days_from_thursday_l2285_228580


namespace NUMINAMATH_CALUDE_toy_value_proof_l2285_228561

theorem toy_value_proof (total_toys : ℕ) (total_worth : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_worth = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    other_toy_value * (total_toys - 1) + special_toy_value = total_worth ∧
    other_toy_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_value_proof_l2285_228561


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l2285_228583

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line ax + by = c if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop := a * x + b * y = c

theorem parallel_lines_intersection (c d : ℝ) : 
  parallel (3 / 4) (-6 / d) ∧ 
  point_on_line 3 (-4) c 2 (-3) ∧
  point_on_line 6 d (2 * c) 2 (-3) →
  c = 18 ∧ d = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l2285_228583


namespace NUMINAMATH_CALUDE_power_of_power_l2285_228574

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2285_228574


namespace NUMINAMATH_CALUDE_dodgeball_team_theorem_l2285_228549

/-- The number of players in the dodgeball league -/
def total_players : ℕ := 12

/-- The number of players on each team -/
def team_size : ℕ := 6

/-- The number of times two specific players are on the same team -/
def same_team_count : ℕ := 210

/-- The total number of possible team combinations -/
def total_combinations : ℕ := Nat.choose total_players team_size

theorem dodgeball_team_theorem :
  ∀ (player1 player2 : Fin total_players),
    player1 ≠ player2 →
    (Nat.choose (total_players - 2) (team_size - 2) : ℕ) = same_team_count :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_team_theorem_l2285_228549


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2285_228538

/-- Given a parabola y = (1/m)x^2 where m ≠ 0, its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2285_228538


namespace NUMINAMATH_CALUDE_hyeji_total_water_intake_l2285_228515

-- Define the conversion rate from liters to milliliters
def liters_to_ml (liters : ℝ) : ℝ := liters * 1000

-- Define Hyeji's daily water intake in liters
def daily_intake : ℝ := 2

-- Define the additional amount Hyeji drank in milliliters
def additional_intake : ℝ := 460

-- Theorem to prove
theorem hyeji_total_water_intake :
  liters_to_ml daily_intake + additional_intake = 2460 := by
  sorry

end NUMINAMATH_CALUDE_hyeji_total_water_intake_l2285_228515


namespace NUMINAMATH_CALUDE_smallest_n_with_75_divisors_l2285_228505

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_n_with_75_divisors :
  ∃ n : ℕ, 
    is_multiple_of_75 n ∧ 
    count_divisors n = 75 ∧ 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_75 m ∧ count_divisors m = 75)) ∧
    n / 75 = 432 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_75_divisors_l2285_228505


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_ten_zeros_l2285_228571

theorem smallest_multiplier_for_ten_zeros (n : ℕ) : 
  (∀ m : ℕ, m < 78125000 → ¬(∃ k : ℕ, 128 * m = k * 10^10)) ∧ 
  (∃ k : ℕ, 128 * 78125000 = k * 10^10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_ten_zeros_l2285_228571


namespace NUMINAMATH_CALUDE_people_in_hall_l2285_228539

theorem people_in_hall (total_chairs : ℕ) (seated_people : ℕ) (empty_chairs : ℕ) :
  seated_people = (5 : ℕ) * total_chairs / 8 →
  empty_chairs = 8 →
  seated_people = total_chairs - empty_chairs →
  seated_people * 2 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_hall_l2285_228539


namespace NUMINAMATH_CALUDE_sandys_correct_sums_l2285_228592

theorem sandys_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 55) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧ 
    correct_sums = 23 := by
  sorry

end NUMINAMATH_CALUDE_sandys_correct_sums_l2285_228592


namespace NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2285_228522

/-- A three-digit palindrome -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b

/-- The greatest common factor of all three-digit palindromes is 1 -/
theorem gcf_three_digit_palindromes :
  ∃ (g : ℕ), g > 0 ∧ 
    (∀ n : ℕ, ThreeDigitPalindrome n → g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n : ℕ, ThreeDigitPalindrome n → d ∣ n) → d ≤ g) ∧
    g = 1 :=
sorry

end NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2285_228522


namespace NUMINAMATH_CALUDE_pet_sitting_earnings_l2285_228554

def hourly_rate : ℕ := 5
def hours_week1 : ℕ := 20
def hours_week2 : ℕ := 30

theorem pet_sitting_earnings : 
  hourly_rate * (hours_week1 + hours_week2) = 250 := by
  sorry

end NUMINAMATH_CALUDE_pet_sitting_earnings_l2285_228554


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2285_228560

theorem shoe_box_problem (pairs : ℕ) (prob : ℝ) (total : ℕ) : 
  pairs = 100 →
  prob = 0.005025125628140704 →
  (pairs : ℝ) / ((total * (total - 1)) / 2) = prob →
  total = 200 :=
sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2285_228560


namespace NUMINAMATH_CALUDE_fraction_equality_l2285_228588

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
    (h3 : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2285_228588


namespace NUMINAMATH_CALUDE_sqrt2_plus_1_power_l2285_228512

theorem sqrt2_plus_1_power (n : ℕ+) :
  ∃ m : ℕ+, (Real.sqrt 2 + 1) ^ n.val = Real.sqrt m.val + Real.sqrt (m.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_plus_1_power_l2285_228512


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2285_228526

/-- Given six positive consecutive integers starting with c, their average d,
    prove that the average of 7 consecutive integers starting with d is c + 5.5 -/
theorem consecutive_integers_average (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 5.5) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2285_228526
