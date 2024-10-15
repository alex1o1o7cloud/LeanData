import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l733_73358

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l733_73358


namespace NUMINAMATH_CALUDE_finite_solutions_except_two_l733_73344

/-- The set of positive integer solutions x for the equation xn+1 | n^2+kn+1 -/
def S (n k : ℕ+) : Set ℕ+ :=
  {x | ∃ m : ℕ+, (x * n + 1) * m = n^2 + k * n + 1}

/-- The set of positive integers n for which S n k has at least two elements -/
def P (k : ℕ+) : Set ℕ+ :=
  {n | ∃ x y : ℕ+, x ≠ y ∧ x ∈ S n k ∧ y ∈ S n k}

theorem finite_solutions_except_two :
  ∀ k : ℕ+, k ≠ 2 → Set.Finite (P k) :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_except_two_l733_73344


namespace NUMINAMATH_CALUDE_runner_stops_in_third_quarter_l733_73345

theorem runner_stops_in_third_quarter 
  (track_circumference : ℝ) 
  (total_distance : ℝ) 
  (quarter_length : ℝ) :
  track_circumference = 50 →
  total_distance = 5280 →
  quarter_length = track_circumference / 4 →
  ∃ (n : ℕ) (remaining_distance : ℝ),
    total_distance = n * track_circumference + remaining_distance ∧
    remaining_distance > 2 * quarter_length ∧
    remaining_distance ≤ 3 * quarter_length :=
by sorry

end NUMINAMATH_CALUDE_runner_stops_in_third_quarter_l733_73345


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l733_73357

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_deriv x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l733_73357


namespace NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l733_73327

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m) (m^2 - m - 6)

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 3 ∨ m = -2 := by sorry

-- Theorem for when z is a pure imaginary number
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 0 := by sorry

-- Theorem for when z is in the third quadrant
theorem z_in_third_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im < 0 ↔ 0 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l733_73327


namespace NUMINAMATH_CALUDE_perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l733_73306

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_line_perpendicular_and_parallel
  (l : Line) (α β : Plane) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_from_perpendicular_and_parallel
  (α β γ : Plane) :
  perpendicular α β → parallel α γ → perpendicular γ β := by sorry

end NUMINAMATH_CALUDE_perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l733_73306


namespace NUMINAMATH_CALUDE_point_in_region_l733_73326

def in_region (x y : ℝ) : Prop := 2 * x + y - 6 ≤ 0

theorem point_in_region :
  in_region 0 6 ∧
  ¬in_region 0 7 ∧
  ¬in_region 5 0 ∧
  ¬in_region 2 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l733_73326


namespace NUMINAMATH_CALUDE_total_time_cutting_grass_l733_73380

-- Define the time to cut one lawn in minutes
def time_per_lawn : ℕ := 30

-- Define the number of lawns cut on Saturday
def lawns_saturday : ℕ := 8

-- Define the number of lawns cut on Sunday
def lawns_sunday : ℕ := 8

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_time_cutting_grass :
  (time_per_lawn * (lawns_saturday + lawns_sunday)) / minutes_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_time_cutting_grass_l733_73380


namespace NUMINAMATH_CALUDE_heart_then_club_probability_l733_73346

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (hearts : ℕ)
  (clubs : ℕ)

/-- Calculates the probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total_cards * d.clubs / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a heart first and a club second from a standard 52-card deck -/
theorem heart_then_club_probability :
  let standard_deck : Deck := ⟨52, 13, 13⟩
  probability_heart_then_club standard_deck = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_heart_then_club_probability_l733_73346


namespace NUMINAMATH_CALUDE_function_positivity_condition_equiv_a_range_l733_73370

/-- The function f(x) = ax² - (2-a)x + 1 --/
def f (a x : ℝ) : ℝ := a * x^2 - (2 - a) * x + 1

/-- The function g(x) = x --/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the equivalence of the condition and the range of a --/
theorem function_positivity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x : ℝ, max (f a x) (g x) > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_condition_equiv_a_range_l733_73370


namespace NUMINAMATH_CALUDE_recreation_area_tents_l733_73343

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : Campsite) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents : ∃ (c : Campsite), 
  c.north = 100 ∧ 
  c.east = 2 * c.north ∧ 
  c.center = 4 * c.north ∧ 
  c.south = 200 ∧ 
  total_tents c = 900 := by
  sorry


end NUMINAMATH_CALUDE_recreation_area_tents_l733_73343


namespace NUMINAMATH_CALUDE_objects_meet_distance_l733_73336

/-- The distance traveled by object A when it meets object B -/
def distance_A_traveled (t : ℝ) : ℝ := t^2 - t

/-- The distance traveled by object B when it meets object A -/
def distance_B_traveled (t : ℝ) : ℝ := t + 4 * t^2

/-- The initial distance between objects A and B -/
def initial_distance : ℝ := 405

theorem objects_meet_distance (t : ℝ) (h : t > 0) 
  (h1 : distance_A_traveled t + distance_B_traveled t = initial_distance) : 
  distance_A_traveled t = 72 := by
  sorry

end NUMINAMATH_CALUDE_objects_meet_distance_l733_73336


namespace NUMINAMATH_CALUDE_log_drift_theorem_l733_73324

/-- The time it takes for a log to drift downstream -/
def log_drift_time (downstream_time upstream_time : ℝ) : ℝ :=
  6 * (upstream_time - downstream_time)

/-- Theorem: Given the downstream and upstream travel times of a boat, 
    the time for a log to drift downstream is 12 hours -/
theorem log_drift_theorem (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 2)
  (h2 : upstream_time = 3) : 
  log_drift_time downstream_time upstream_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_drift_theorem_l733_73324


namespace NUMINAMATH_CALUDE_value_exceeds_initial_price_min_avg_value_l733_73366

-- Define the value of M at the beginning of the nth year
def value (n : ℕ) : ℚ :=
  if n ≤ 3 then
    20 * (1/2)^(n-1)
  else
    4 * n - 7

-- Define the sum of values for the first n years
def sum_values (n : ℕ) : ℚ :=
  if n ≤ 3 then
    40 - 5 * 2^(3-n)
  else
    2 * n^2 - 5 * n + 32

-- Define the average value over n years
def avg_value (n : ℕ) : ℚ :=
  sum_values n / n

-- Theorem 1: Value exceeds initial price at the beginning of the 7th year
theorem value_exceeds_initial_price :
  ∀ k < 7, value k ≤ 20 ∧ value 7 > 20 :=
sorry

-- Theorem 2: Minimum average value is 11, occurring at n = 4
theorem min_avg_value :
  ∀ n : ℕ, n ≥ 1 → avg_value n ≥ 11 ∧ avg_value 4 = 11 :=
sorry

end NUMINAMATH_CALUDE_value_exceeds_initial_price_min_avg_value_l733_73366


namespace NUMINAMATH_CALUDE_cookies_per_bag_l733_73398

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (bags : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  bags = 6 →
  (chocolate_chip + oatmeal) / bags = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l733_73398


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l733_73302

/-- The length of the path traveled by point F when rolling a quarter-circle region -/
theorem quarter_circle_roll_path_length 
  (EF : ℝ) -- Length of EF (radius of the quarter-circle)
  (h_EF : EF = 3 / Real.pi) -- Given condition that EF = 3/π cm
  : (2 * Real.pi * EF) = 6 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l733_73302


namespace NUMINAMATH_CALUDE_B_max_at_45_l733_73350

/-- The binomial coefficient function -/
def choose (n k : ℕ) : ℕ := sorry

/-- The B_k function as defined in the problem -/
def B (k : ℕ) : ℝ := (choose 500 k) * (0.1 ^ k)

/-- Theorem stating that B(k) is maximum when k = 45 -/
theorem B_max_at_45 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 45 := by sorry

end NUMINAMATH_CALUDE_B_max_at_45_l733_73350


namespace NUMINAMATH_CALUDE_parabola_vertex_l733_73381

/-- Given a parabola with equation y = -x^2 + ax + b where the solution to y ≤ 0
    is (-∞, -1] ∪ [7, ∞), prove that the vertex of this parabola is (3, 16) -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ≤ -1 ∨ x ≥ 7) →
  ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = 16 ∧
    ∀ x, -x^2 + a*x + b = -(x - vertex_x)^2 + vertex_y :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l733_73381


namespace NUMINAMATH_CALUDE_iphone_price_reduction_l733_73339

/-- 
Calculates the final price of an item after two consecutive price reductions.
-/
theorem iphone_price_reduction (initial_price : ℝ) 
  (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 1000 →
  first_reduction = 0.1 →
  second_reduction = 0.2 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 720 := by
sorry

end NUMINAMATH_CALUDE_iphone_price_reduction_l733_73339


namespace NUMINAMATH_CALUDE_conference_married_men_fraction_l733_73340

theorem conference_married_men_fraction
  (total_women : ℕ)
  (single_women : ℕ)
  (married_women : ℕ)
  (married_men : ℕ)
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_conference_married_men_fraction_l733_73340


namespace NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l733_73389

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 2}

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_singleton_one_l733_73389


namespace NUMINAMATH_CALUDE_problem_solution_l733_73393

theorem problem_solution (x : ℝ) (h : |x| = x + 2) :
  19 * x^99 + 3 * x + 27 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l733_73393


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l733_73356

-- Definition of a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part 1
theorem fixed_points_of_specific_quadratic :
  let f := quadratic 1 (-1) (-3)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ ∧ x₁ = -1 ∧ x₂ = 3 := by sorry

-- Part 2
theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  let f := quadratic 2 (-(3+a)) (a-1)
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ →
  (∀ y₁ y₂ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧ is_fixed_point f y₁ ∧ is_fixed_point f y₂ →
    y₁ / y₂ + y₂ / y₁ ≥ 8) := by sorry

-- Part 3
theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point (quadratic a (b+1) (b-1)) x) ↔
  0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l733_73356


namespace NUMINAMATH_CALUDE_clothing_distribution_l733_73331

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (remaining_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : remaining_loads = 5)
  : (total - first_load) / remaining_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l733_73331


namespace NUMINAMATH_CALUDE_units_digit_of_5_to_12_l733_73390

theorem units_digit_of_5_to_12 : ∃ n : ℕ, 5^12 ≡ 5 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_units_digit_of_5_to_12_l733_73390


namespace NUMINAMATH_CALUDE_triangle_reconstruction_unique_l733_73334

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle on a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an acute triangle -/
structure AcuteTriangle where
  A : Point
  B : Point
  C : Point

/-- Represents the given information for triangle reconstruction -/
structure ReconstructionData where
  circumcircle : Circle
  C₀ : Point  -- Intersection of angle bisector from C with circumcircle
  A₁ : Point  -- Intersection of altitude from A with circumcircle
  B₁ : Point  -- Intersection of altitude from B with circumcircle

/-- Function to reconstruct the triangle from the given data -/
def reconstructTriangle (data : ReconstructionData) : AcuteTriangle :=
  sorry

/-- Theorem stating that the triangle can be uniquely reconstructed -/
theorem triangle_reconstruction_unique (data : ReconstructionData) :
  ∃! (triangle : AcuteTriangle),
    (Circle.center data.circumcircle).x ^ 2 + (Circle.center data.circumcircle).y ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.C₀.x - triangle.C.x) ^ 2 + (data.C₀.y - triangle.C.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.A₁.x - triangle.A.x) ^ 2 + (data.A₁.y - triangle.A.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.B₁.x - triangle.B.x) ^ 2 + (data.B₁.y - triangle.B.y) ^ 2 = 
      data.circumcircle.radius ^ 2 :=
  sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_unique_l733_73334


namespace NUMINAMATH_CALUDE_data_groups_is_six_l733_73368

/-- Given a dataset, calculate the number of groups it should be divided into -/
def calculateGroups (maxValue minValue interval : ℕ) : ℕ :=
  let range := maxValue - minValue
  let preliminaryGroups := (range + interval - 1) / interval
  preliminaryGroups

/-- Theorem stating that for the given conditions, the number of groups is 6 -/
theorem data_groups_is_six :
  calculateGroups 36 15 4 = 6 := by
  sorry

#eval calculateGroups 36 15 4

end NUMINAMATH_CALUDE_data_groups_is_six_l733_73368


namespace NUMINAMATH_CALUDE_power_identity_l733_73332

theorem power_identity (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(3*m + 2*n) = 72 := by
sorry

end NUMINAMATH_CALUDE_power_identity_l733_73332


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l733_73341

theorem trigonometric_expression_value (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.sin (3 * Real.pi / 2 + α)) /
  (1 + Real.sin α ^ 2 - Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi + α) ^ 2) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l733_73341


namespace NUMINAMATH_CALUDE_john_writing_years_l733_73311

/-- Represents the number of months in a year -/
def months_per_year : ℕ := 12

/-- Represents the number of months it takes John to write a book -/
def months_per_book : ℕ := 2

/-- Represents the average earnings per book in dollars -/
def earnings_per_book : ℕ := 30000

/-- Represents the total earnings from writing in dollars -/
def total_earnings : ℕ := 3600000

/-- Calculates the number of years John has been writing -/
def years_writing : ℚ :=
  (total_earnings / earnings_per_book) / (months_per_year / months_per_book)

theorem john_writing_years :
  years_writing = 20 := by sorry

end NUMINAMATH_CALUDE_john_writing_years_l733_73311


namespace NUMINAMATH_CALUDE_count_triples_eq_30787_l733_73314

/-- 
Counts the number of ordered triples (x,y,z) of non-negative integers 
satisfying x ≤ y ≤ z and x + y + z ≤ 100
-/
def count_triples : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (x, y, z) := t
    x ≤ y ∧ y ≤ z ∧ x + y + z ≤ 100
  ) (Finset.product (Finset.range 101) (Finset.product (Finset.range 101) (Finset.range 101)))).card

theorem count_triples_eq_30787 : count_triples = 30787 := by
  sorry


end NUMINAMATH_CALUDE_count_triples_eq_30787_l733_73314


namespace NUMINAMATH_CALUDE_consecutive_sum_largest_l733_73386

theorem consecutive_sum_largest (n : ℕ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) = 180) → (n+4 = 38) :=
by
  sorry

#check consecutive_sum_largest

end NUMINAMATH_CALUDE_consecutive_sum_largest_l733_73386


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l733_73342

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l733_73342


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l733_73320

theorem arithmetic_expression_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l733_73320


namespace NUMINAMATH_CALUDE_cistern_width_is_four_l733_73354

/-- Represents the dimensions and properties of a cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a width of 4 meters --/
theorem cistern_width_is_four :
  ∃ (c : Cistern),
    c.length = 6 ∧
    c.depth = 1.25 ∧
    c.wetSurfaceArea = 49 ∧
    totalWetSurfaceArea c = c.wetSurfaceArea ∧
    c.width = 4 := by
  sorry


end NUMINAMATH_CALUDE_cistern_width_is_four_l733_73354


namespace NUMINAMATH_CALUDE_cubic_increasing_and_odd_l733_73308

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_increasing_and_odd_l733_73308


namespace NUMINAMATH_CALUDE_stickers_per_page_l733_73395

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) 
  (h1 : total_stickers = 220) 
  (h2 : total_pages = 22) 
  (h3 : total_stickers > 0) 
  (h4 : total_pages > 0) : 
  total_stickers / total_pages = 10 :=
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l733_73395


namespace NUMINAMATH_CALUDE_number_value_l733_73300

theorem number_value (x : ℝ) (number : ℝ) 
  (h1 : 5 - 5/x = number + 4/x) 
  (h2 : x = 9) : 
  number = 4 := by
sorry

end NUMINAMATH_CALUDE_number_value_l733_73300


namespace NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l733_73359

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) := by
sorry

end NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l733_73359


namespace NUMINAMATH_CALUDE_oranges_per_sack_l733_73349

/-- Proves that the number of oranges per sack is 50, given the harvest conditions --/
theorem oranges_per_sack (total_sacks : ℕ) (discarded_sacks : ℕ) (total_oranges : ℕ)
  (h1 : total_sacks = 76)
  (h2 : discarded_sacks = 64)
  (h3 : total_oranges = 600) :
  total_oranges / (total_sacks - discarded_sacks) = 50 := by
  sorry

#check oranges_per_sack

end NUMINAMATH_CALUDE_oranges_per_sack_l733_73349


namespace NUMINAMATH_CALUDE_qq_level_difference_l733_73347

/-- Represents the QQ level system -/
structure QQLevel where
  activedays : ℕ
  stars : ℕ
  moons : ℕ
  suns : ℕ

/-- Calculates the total number of stars for a given level -/
def totalStars (level : ℕ) : ℕ := level

/-- Calculates the number of active days required for a given level -/
def activeDaysForLevel (level : ℕ) : ℕ := level * (level + 4)

/-- Converts stars to an equivalent QQ level -/
def starsToLevel (stars : ℕ) : ℕ := stars

/-- Theorem: The difference in active days between 1 sun and 2 moons 1 star is 203 -/
theorem qq_level_difference : 
  let sunLevel := starsToLevel (4 * 4)
  let currentLevel := starsToLevel (2 * 4 + 1)
  activeDaysForLevel sunLevel - activeDaysForLevel currentLevel = 203 := by
  sorry


end NUMINAMATH_CALUDE_qq_level_difference_l733_73347


namespace NUMINAMATH_CALUDE_quadratic_inequality_l733_73352

theorem quadratic_inequality : ∀ x : ℝ, 2*x^2 + 5*x + 3 > x^2 + 4*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l733_73352


namespace NUMINAMATH_CALUDE_divisibility_proof_l733_73335

theorem divisibility_proof : (2 ∣ 32) ∧ (20 ∣ 320) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l733_73335


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l733_73313

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l733_73313


namespace NUMINAMATH_CALUDE_f_difference_l733_73397

/-- The function f(x) = x^4 + 3x^3 + 2x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + 7*x

/-- Theorem: f(6) - f(-6) = 1380 -/
theorem f_difference : f 6 - f (-6) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l733_73397


namespace NUMINAMATH_CALUDE_lisa_walking_distance_l733_73382

/-- Lisa's walking problem -/
theorem lisa_walking_distance
  (walking_speed : ℕ)  -- Lisa's walking speed in meters per minute
  (daily_duration : ℕ)  -- Lisa's daily walking duration in minutes
  (days : ℕ)  -- Number of days
  (h1 : walking_speed = 10)  -- Lisa walks 10 meters each minute
  (h2 : daily_duration = 60)  -- Lisa walks for an hour (60 minutes) every day
  (h3 : days = 2)  -- We're considering two days
  : walking_speed * daily_duration * days = 1200 :=
by
  sorry

#check lisa_walking_distance

end NUMINAMATH_CALUDE_lisa_walking_distance_l733_73382


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l733_73383

/-- A coloring of positive integers -/
def Coloring := ℕ+ → Fin 2009

/-- Predicate for a valid coloring satisfying the problem conditions -/
def ValidColoring (f : Coloring) : Prop :=
  (∀ c : Fin 2009, Set.Infinite {n : ℕ+ | f n = c}) ∧
  (∀ a b c : ℕ+, ∀ i j k : Fin 2009,
    i ≠ j ∧ j ≠ k ∧ i ≠ k → f a = i ∧ f b = j ∧ f c = k → a * b ≠ c)

/-- Theorem stating the existence of a valid coloring -/
theorem exists_valid_coloring : ∃ f : Coloring, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l733_73383


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l733_73369

/-- The quadratic function f(x) = x^2 + tx - t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- Theorem stating that t ≥ 0 is a sufficient but not necessary condition for f to have a root -/
theorem sufficient_not_necessary_condition (t : ℝ) :
  (∀ t ≥ 0, ∃ x, f t x = 0) ∧
  (∃ t < 0, ∃ x, f t x = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l733_73369


namespace NUMINAMATH_CALUDE_men_earnings_l733_73325

/-- Represents the total earnings of workers over a week. -/
structure Earnings where
  men : ℝ
  women : ℝ
  boys : ℝ

/-- Represents the work rates and hours of different groups of workers. -/
structure WorkData where
  X : ℝ  -- Number of women equivalent to 5 men
  M : ℝ  -- Hours worked by men
  W : ℝ  -- Hours worked by women
  B : ℝ  -- Hours worked by boys
  rm : ℝ  -- Wage rate for men per hour
  rw : ℝ  -- Wage rate for women per hour
  rb : ℝ  -- Wage rate for boys per hour

/-- Theorem stating the total earnings for men given the problem conditions. -/
theorem men_earnings (data : WorkData) (total : Earnings) :
  (5 : ℝ) * data.X * data.W * data.rw = (8 : ℝ) * data.B * data.rb →
  total.men + total.women + total.boys = 180 →
  total.men = (5 : ℝ) * data.M * data.rm :=
by sorry

end NUMINAMATH_CALUDE_men_earnings_l733_73325


namespace NUMINAMATH_CALUDE_yogurt_combinations_l733_73379

/-- The number of yogurt flavors -/
def num_flavors : ℕ := 5

/-- The number of toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 2

/-- The number of doubling options (double first, double second, or no doubling) -/
def doubling_options : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem yogurt_combinations :
  num_flavors * choose num_toppings toppings_to_choose * doubling_options = 315 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l733_73379


namespace NUMINAMATH_CALUDE_sally_lost_cards_l733_73361

def pokemon_cards_lost (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem sally_lost_cards (initial : ℕ) (received : ℕ) (current : ℕ)
  (h1 : initial = 27)
  (h2 : received = 41)
  (h3 : current = 48) :
  pokemon_cards_lost initial received current = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_cards_l733_73361


namespace NUMINAMATH_CALUDE_elvis_squares_l733_73328

theorem elvis_squares (total_matchsticks : ℕ) (elvis_square_size : ℕ) (ralph_square_size : ℕ) 
  (ralph_squares : ℕ) (leftover_matchsticks : ℕ) :
  total_matchsticks = 50 →
  elvis_square_size = 4 →
  ralph_square_size = 8 →
  ralph_squares = 3 →
  leftover_matchsticks = 6 →
  ∃ (elvis_squares : ℕ), 
    elvis_squares * elvis_square_size + ralph_squares * ralph_square_size + leftover_matchsticks = total_matchsticks ∧
    elvis_squares = 5 := by
  sorry

end NUMINAMATH_CALUDE_elvis_squares_l733_73328


namespace NUMINAMATH_CALUDE_line_segment_length_l733_73367

/-- Represents a line segment in 3D space -/
structure LineSegment3D where
  length : ℝ

/-- Represents a space region around a line segment -/
structure SpaceRegion where
  segment : LineSegment3D
  radius : ℝ
  volume : ℝ

/-- Theorem: If a space region containing all points within 5 units of a line segment 
    in three-dimensional space has a volume of 500π, then the length of the line segment is 40/3 units. -/
theorem line_segment_length (region : SpaceRegion) 
  (h1 : region.radius = 5)
  (h2 : region.volume = 500 * Real.pi) : 
  region.segment.length = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l733_73367


namespace NUMINAMATH_CALUDE_symmetric_function_theorem_l733_73387

/-- A function is symmetric to another function with respect to the origin -/
def SymmetricToOrigin (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-x) = -y

/-- The main theorem -/
theorem symmetric_function_theorem (f : ℝ → ℝ) :
  SymmetricToOrigin f (λ x ↦ 3 - 2*x) → ∀ x, f x = -2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_theorem_l733_73387


namespace NUMINAMATH_CALUDE_find_y_l733_73322

theorem find_y : ∃ y : ℝ, y > 0 ∧ 0.02 * y * y = 18 ∧ y = 30 := by sorry

end NUMINAMATH_CALUDE_find_y_l733_73322


namespace NUMINAMATH_CALUDE_find_other_number_l733_73374

theorem find_other_number (a b : ℕ+) (hcf lcm : ℕ+) : 
  Nat.gcd a.val b.val = hcf.val →
  Nat.lcm a.val b.val = lcm.val →
  hcf * lcm = a * b →
  a = 154 →
  hcf = 14 →
  lcm = 396 →
  b = 36 := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l733_73374


namespace NUMINAMATH_CALUDE_absolute_value_equality_l733_73315

theorem absolute_value_equality (a : ℝ) : 
  |a| = |5 + 1/3| → a = 5 + 1/3 ∨ a = -(5 + 1/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l733_73315


namespace NUMINAMATH_CALUDE_total_balls_in_box_l733_73363

theorem total_balls_in_box (black_balls : ℕ) (white_balls : ℕ) : 
  black_balls = 8 →
  white_balls = 6 * black_balls →
  black_balls + white_balls = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l733_73363


namespace NUMINAMATH_CALUDE_chameleon_distance_theorem_l733_73351

/-- A chameleon is a sequence of 3n letters, with exactly n occurrences of each of the letters a, b, and c -/
def Chameleon (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'a' = n ∧ s.count 'b' = n ∧ s.count 'c' = n }

/-- The number of swaps required to transform one chameleon into another -/
def swaps_required (n : ℕ) (X Y : Chameleon n) : ℕ := sorry

theorem chameleon_distance_theorem (n : ℕ) (hn : n > 0) (X : Chameleon n) :
  ∃ Y : Chameleon n, swaps_required n X Y ≥ (3 * n^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_chameleon_distance_theorem_l733_73351


namespace NUMINAMATH_CALUDE_problem_solution_l733_73392

theorem problem_solution : 3 * 3^4 + 9^30 / 9^28 = 324 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l733_73392


namespace NUMINAMATH_CALUDE_max_total_pieces_l733_73319

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white_pieces : ℕ
  black_pieces : ℕ

/-- The size of the chessboard -/
def board_size : ℕ := 8

/-- Condition: In each row and column, the number of white pieces is twice the number of black pieces -/
def valid_distribution (config : ChessboardConfig) : Prop :=
  config.white_pieces = 2 * config.black_pieces

/-- The total number of pieces on the board -/
def total_pieces (config : ChessboardConfig) : ℕ :=
  config.white_pieces + config.black_pieces

/-- The maximum number of pieces that can be placed on the board -/
def max_pieces : ℕ := board_size * board_size

theorem max_total_pieces :
  ∃ (config : ChessboardConfig),
    valid_distribution config ∧
    (∀ (other : ChessboardConfig),
      valid_distribution other →
      total_pieces other ≤ total_pieces config) ∧
    total_pieces config = 48 :=
  sorry

end NUMINAMATH_CALUDE_max_total_pieces_l733_73319


namespace NUMINAMATH_CALUDE_bushes_needed_bushes_needed_proof_l733_73337

/-- The number of containers of blueberries yielded by each bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem: The number of bushes needed to obtain the target number of zucchinis -/
theorem bushes_needed : ℕ := 12

/-- Proof that bushes_needed is correct -/
theorem bushes_needed_proof : 
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade :=
by sorry

end NUMINAMATH_CALUDE_bushes_needed_bushes_needed_proof_l733_73337


namespace NUMINAMATH_CALUDE_candy_store_spending_correct_l733_73384

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 345 / 100

/-- Fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3 / 5

/-- Fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 1 / 3

/-- Amount spent at the candy store -/
def candy_store_spending : ℚ := 
  weekly_allowance * (1 - arcade_fraction) * (1 - toy_store_fraction)

theorem candy_store_spending_correct : 
  candy_store_spending = 92 / 100 := by sorry

end NUMINAMATH_CALUDE_candy_store_spending_correct_l733_73384


namespace NUMINAMATH_CALUDE_power_division_equality_l733_73310

theorem power_division_equality : (10^8 : ℝ) / (2 * 10^6) = 50 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l733_73310


namespace NUMINAMATH_CALUDE_wig_cost_calculation_l733_73321

theorem wig_cost_calculation (plays : ℕ) (acts_per_play : ℕ) (wigs_per_act : ℕ) (cost_per_wig : ℕ) :
  plays = 2 →
  acts_per_play = 5 →
  wigs_per_act = 2 →
  cost_per_wig = 5 →
  plays * acts_per_play * wigs_per_act * cost_per_wig = 100 :=
by sorry

end NUMINAMATH_CALUDE_wig_cost_calculation_l733_73321


namespace NUMINAMATH_CALUDE_dogs_in_center_l733_73329

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  jump : ℕ
  fetch : ℕ
  shake : ℕ
  jumpFetch : ℕ
  fetchShake : ℕ
  jumpShake : ℕ
  allThree : ℕ
  none : ℕ

/-- The total number of dogs in the center -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.jumpFetch - d.allThree) +
  (d.fetchShake - d.allThree) +
  (d.jumpShake - d.allThree) +
  (d.jump - d.jumpFetch - d.jumpShake + d.allThree) +
  (d.fetch - d.jumpFetch - d.fetchShake + d.allThree) +
  (d.shake - d.jumpShake - d.fetchShake + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the center is 115 -/
theorem dogs_in_center (d : DogTricks)
  (h_jump : d.jump = 70)
  (h_fetch : d.fetch = 40)
  (h_shake : d.shake = 50)
  (h_jumpFetch : d.jumpFetch = 30)
  (h_fetchShake : d.fetchShake = 20)
  (h_jumpShake : d.jumpShake = 25)
  (h_allThree : d.allThree = 15)
  (h_none : d.none = 15) :
  totalDogs d = 115 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_center_l733_73329


namespace NUMINAMATH_CALUDE_angle_300_in_fourth_quadrant_l733_73377

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_in_fourth_quadrant (angle : ℝ) : Prop :=
  270 < angle ∧ angle < 360

/-- Prove that 300° is in the fourth quadrant -/
theorem angle_300_in_fourth_quadrant :
  is_in_fourth_quadrant 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_300_in_fourth_quadrant_l733_73377


namespace NUMINAMATH_CALUDE_selling_prices_correct_l733_73353

def calculate_selling_price (cost : ℚ) (profit_percent : ℚ) (tax_percent : ℚ) : ℚ :=
  let pre_tax_price := cost * (1 + profit_percent)
  pre_tax_price * (1 + tax_percent)

theorem selling_prices_correct : 
  let cost_A : ℚ := 650
  let cost_B : ℚ := 1200
  let cost_C : ℚ := 800
  let profit_A : ℚ := 1/10
  let profit_B : ℚ := 3/20
  let profit_C : ℚ := 1/5
  let tax : ℚ := 1/20
  
  (calculate_selling_price cost_A profit_A tax = 75075/100) ∧
  (calculate_selling_price cost_B profit_B tax = 1449) ∧
  (calculate_selling_price cost_C profit_C tax = 1008) :=
by sorry

end NUMINAMATH_CALUDE_selling_prices_correct_l733_73353


namespace NUMINAMATH_CALUDE_complex_square_on_negative_imaginary_axis_l733_73301

/-- A complex number z lies on the negative half of the imaginary axis if its real part is 0 and its imaginary part is negative -/
def lies_on_negative_imaginary_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

theorem complex_square_on_negative_imaginary_axis (a : ℝ) :
  lies_on_negative_imaginary_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_imaginary_axis_l733_73301


namespace NUMINAMATH_CALUDE_f_difference_l733_73372

-- Define a linear function f
variable (f : ℝ → ℝ)

-- Define the linearity of f
variable (hf : ∀ x y : ℝ, ∀ c : ℝ, f (x + c * y) = f x + c * f y)

-- Define the condition f(d+1) - f(d) = 3 for all real numbers d
variable (h : ∀ d : ℝ, f (d + 1) - f d = 3)

-- State the theorem
theorem f_difference (f : ℝ → ℝ) (hf : ∀ x y : ℝ, ∀ c : ℝ, f (x + c * y) = f x + c * f y) 
  (h : ∀ d : ℝ, f (d + 1) - f d = 3) : 
  f 3 - f 5 = -6 := by sorry

end NUMINAMATH_CALUDE_f_difference_l733_73372


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l733_73399

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n : ℕ := 102932847
  let d : ℕ := 25
  ∃ (x : ℕ), x = 22 ∧ x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l733_73399


namespace NUMINAMATH_CALUDE_michelle_candy_sugar_l733_73396

/-- The total grams of sugar in Michelle's candy purchase -/
def total_sugar (num_bars : ℕ) (sugar_per_bar : ℕ) (lollipop_sugar : ℕ) : ℕ :=
  num_bars * sugar_per_bar + lollipop_sugar

/-- Theorem: The total sugar in Michelle's candy purchase is 177 grams -/
theorem michelle_candy_sugar :
  total_sugar 14 10 37 = 177 := by sorry

end NUMINAMATH_CALUDE_michelle_candy_sugar_l733_73396


namespace NUMINAMATH_CALUDE_sqrt_undefined_range_l733_73323

theorem sqrt_undefined_range (a : ℝ) : ¬ (∃ x : ℝ, x ^ 2 = 2 * a - 1) → a < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_undefined_range_l733_73323


namespace NUMINAMATH_CALUDE_triangle_side_range_l733_73318

theorem triangle_side_range :
  ∀ m : ℝ,
  (3 > 0 ∧ 1 - 2*m > 0 ∧ 8 > 0) →
  (3 + (1 - 2*m) > 8 ∧ 3 + 8 > 1 - 2*m ∧ (1 - 2*m) + 8 > 3) →
  (-5 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l733_73318


namespace NUMINAMATH_CALUDE_soda_price_ratio_l733_73330

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (volume_A : ℝ) (volume_B : ℝ) (price_A : ℝ) (price_B : ℝ)
  (h_volume : volume_A = 1.25 * volume_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive : volume_B > 0 ∧ price_B > 0) :
  (price_A / volume_A) / (price_B / volume_B) = 17 / 25 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l733_73330


namespace NUMINAMATH_CALUDE_combined_selling_price_theorem_l733_73385

/-- Calculates the selling price of an article including profit and tax -/
def sellingPrice (cost : ℚ) (profitPercent : ℚ) (taxRate : ℚ) : ℚ :=
  let priceBeforeTax := cost * (1 + profitPercent)
  priceBeforeTax * (1 + taxRate)

/-- Calculates the combined selling price of three articles -/
def combinedSellingPrice (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) : ℚ :=
  sellingPrice cost1 profit1 taxRate +
  sellingPrice cost2 profit2 taxRate +
  sellingPrice cost3 profit3 taxRate

theorem combined_selling_price_theorem (cost1 cost2 cost3 : ℚ) (profit1 profit2 profit3 : ℚ) (taxRate : ℚ) :
  combinedSellingPrice cost1 cost2 cost3 profit1 profit2 profit3 taxRate =
  sellingPrice 500 (45/100) (12/100) +
  sellingPrice 300 (30/100) (12/100) +
  sellingPrice 1000 (20/100) (12/100) := by
  sorry

end NUMINAMATH_CALUDE_combined_selling_price_theorem_l733_73385


namespace NUMINAMATH_CALUDE_smallest_number_l733_73333

theorem smallest_number (a b c d e : ℚ) : 
  a = 0.803 → b = 0.8003 → c = 0.8 → d = 0.8039 → e = 0.809 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l733_73333


namespace NUMINAMATH_CALUDE_new_species_growth_pattern_l733_73304

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J -- J-shaped growth
  | S -- S-shaped growth

/-- Represents the population growth pattern over time --/
structure PopulationGrowth where
  initialShape : GrowthShape
  finalShape : GrowthShape

/-- Represents a new species entering an area --/
structure NewSpecies where
  enteredArea : Bool

/-- Theorem stating the population growth pattern for a new species --/
theorem new_species_growth_pattern (species : NewSpecies) 
  (h : species.enteredArea = true) : 
  ∃ (growth : PopulationGrowth), 
    growth.initialShape = GrowthShape.J ∧ 
    growth.finalShape = GrowthShape.S :=
  sorry

end NUMINAMATH_CALUDE_new_species_growth_pattern_l733_73304


namespace NUMINAMATH_CALUDE_problem_solution_l733_73305

/-- The number of people initially working on the problem -/
def initial_people : ℕ := 1

/-- The initial working time in hours -/
def initial_time : ℕ := 10

/-- The working time after adding one person, in hours -/
def reduced_time : ℕ := 5

theorem problem_solution :
  initial_people * initial_time = (initial_people + 1) * reduced_time :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l733_73305


namespace NUMINAMATH_CALUDE_nikita_produces_two_per_hour_l733_73388

-- Define the productivity of Ivan and Nikita
def ivan_productivity : ℝ := sorry
def nikita_productivity : ℝ := sorry

-- Define the conditions from the problem
axiom monday_condition : 3 * ivan_productivity + 2 * nikita_productivity = 7
axiom tuesday_condition : 5 * ivan_productivity + 3 * nikita_productivity = 11

-- Theorem to prove
theorem nikita_produces_two_per_hour : nikita_productivity = 2 := by
  sorry

end NUMINAMATH_CALUDE_nikita_produces_two_per_hour_l733_73388


namespace NUMINAMATH_CALUDE_range_of_g_l733_73364

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := f (x^2 + 1)

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l733_73364


namespace NUMINAMATH_CALUDE_sum_lower_bound_l733_73378

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b + 3 = a * b) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l733_73378


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l733_73355

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  (Real.sqrt (x^4 + x^2 + 2*x + 1) + Real.sqrt (x^4 - 2*x^3 + 5*x^2 - 4*x + 1)) / x ≥ Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l733_73355


namespace NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l733_73317

theorem negation_of_divisible_by_two_is_even :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, 2 ∣ n ∧ ¬Even n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l733_73317


namespace NUMINAMATH_CALUDE_largest_of_consecutive_odd_divisible_by_3_l733_73375

/-- Three consecutive odd natural numbers divisible by 3 whose sum is 72 -/
def ConsecutiveOddDivisibleBy3 (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) ∧
  (b = a + 6 ∧ c = a + 12) ∧
  (a + b + c = 72)

theorem largest_of_consecutive_odd_divisible_by_3 {a b c : ℕ} 
  (h : ConsecutiveOddDivisibleBy3 a b c) : 
  max a (max b c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_consecutive_odd_divisible_by_3_l733_73375


namespace NUMINAMATH_CALUDE_parking_lot_bikes_l733_73360

/-- The number of bikes in a parking lot with cars and bikes. -/
def numBikes (numCars : ℕ) (totalWheels : ℕ) (wheelsPerCar : ℕ) (wheelsPerBike : ℕ) : ℕ :=
  (totalWheels - numCars * wheelsPerCar) / wheelsPerBike

theorem parking_lot_bikes :
  numBikes 14 76 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_bikes_l733_73360


namespace NUMINAMATH_CALUDE_kyles_rose_expense_l733_73394

def roses_last_year : ℕ := 12
def roses_this_year : ℕ := roses_last_year / 2
def roses_needed : ℕ := 2 * roses_last_year
def price_per_rose : ℕ := 3

theorem kyles_rose_expense : 
  (roses_needed - roses_this_year) * price_per_rose = 54 := by sorry

end NUMINAMATH_CALUDE_kyles_rose_expense_l733_73394


namespace NUMINAMATH_CALUDE_trigonometric_identities_l733_73376

theorem trigonometric_identities (x : ℝ) : 
  ((Real.sqrt 3) / 2 * Real.cos x - (1 / 2) * Real.sin x = Real.cos (x + π / 6)) ∧ 
  (Real.sin x + Real.cos x = Real.sqrt 2 * Real.sin (x + π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l733_73376


namespace NUMINAMATH_CALUDE_sum_of_f_values_l733_73312

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 2/3 → f x + f ((x - 1) / (3 * x - 2)) = x

/-- The main theorem stating the sum of f(0), f(1), and f(2) -/
theorem sum_of_f_values (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 0 + f 1 + f 2 = 87/40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l733_73312


namespace NUMINAMATH_CALUDE_fiona_reach_food_prob_l733_73365

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the predator pads
def predator_pads : Set ℕ := {2, 5, 8}

-- Define the food pad
def food_pad : ℕ := 14

-- Define Fiona's starting pad
def start_pad : ℕ := 0

-- Define the probability of hopping to the next pad
def hop_prob : ℚ := 1/2

-- Define the probability of jumping 3 pads
def jump_prob : ℚ := 1/2

-- Define a function to calculate the probability of reaching a pad safely
def safe_prob (pad : ℕ) : ℚ := sorry

-- Theorem statement
theorem fiona_reach_food_prob : 
  safe_prob food_pad = 5/1024 := sorry

end NUMINAMATH_CALUDE_fiona_reach_food_prob_l733_73365


namespace NUMINAMATH_CALUDE_even_sum_condition_l733_73316

theorem even_sum_condition (m n : ℤ) :
  (∀ m n : ℤ, Even m ∧ Even n → Even (m + n)) ∧
  (∃ m n : ℤ, Even (m + n) ∧ (¬Even m ∨ ¬Even n)) :=
sorry

end NUMINAMATH_CALUDE_even_sum_condition_l733_73316


namespace NUMINAMATH_CALUDE_valid_arrangements_l733_73362

/-- The number of ways to arrange plates on a circular table. -/
def circularArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + green + orange + yellow))

/-- The number of arrangements with green plates adjacent. -/
def greenAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + 1 + orange + yellow))

/-- The number of arrangements with orange plates adjacent. -/
def orangeAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial 1 * Nat.factorial yellow * 
   (blue + red + green + 1 + yellow))

/-- The number of arrangements with both green and orange plates adjacent. -/
def bothAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   (blue + red + 1 + 1 + yellow))

/-- The main theorem stating the number of valid arrangements. -/
theorem valid_arrangements (blue red green orange yellow : ℕ) 
  (h_blue : blue = 6) (h_red : red = 3) (h_green : green = 3) 
  (h_orange : orange = 2) (h_yellow : yellow = 1) :
  circularArrangements blue red green orange yellow - 
  (greenAdjacentArrangements blue red green orange yellow + 
   orangeAdjacentArrangements blue red green orange yellow - 
   bothAdjacentArrangements blue red green orange yellow) =
  circularArrangements 6 3 3 2 1 - 
  (greenAdjacentArrangements 6 3 3 2 1 + 
   orangeAdjacentArrangements 6 3 3 2 1 - 
   bothAdjacentArrangements 6 3 3 2 1) := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l733_73362


namespace NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l733_73338

theorem odd_binomial_coefficients_count (n : ℕ) : 
  (∃ m : ℕ, (Finset.filter (fun k => Nat.choose n k % 2 = 1) (Finset.range (n + 1))).card = 2^m) := by
  sorry

end NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l733_73338


namespace NUMINAMATH_CALUDE_root_square_relation_l733_73303

/-- The polynomial h(x) = x^3 + x^2 + 2x + 8 -/
def h (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 8

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem root_square_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, j b c d x = 0 ↔ ∃ r : ℝ, h r = 0 ∧ x = r^2) →
  b = 1 ∧ c = -8 ∧ d = 32 := by
sorry

end NUMINAMATH_CALUDE_root_square_relation_l733_73303


namespace NUMINAMATH_CALUDE_parallel_lines_and_not_always_parallel_planes_l733_73373

-- Define the line equations
def line1 (a x y : ℝ) : Prop := a * x + 3 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, line1 a x y ↔ line2 a x y

-- Define a plane
def Plane : Type := ℝ × ℝ × ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define distance between a point and a plane
def distance (p : Point) (plane : Plane) : ℝ := sorry

-- Define non-collinear points
def nonCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define parallel planes
def parallelPlanes (α β : Plane) : Prop := sorry

-- Statement of the theorem
theorem parallel_lines_and_not_always_parallel_planes :
  (∀ a, parallel a ↔ a = -3) ∧
  ¬(∀ α β : Plane, ∀ p1 p2 p3 : Point,
    nonCollinear p1 p2 p3 →
    distance p1 β = distance p2 β ∧ distance p2 β = distance p3 β →
    parallelPlanes α β) := by sorry

end NUMINAMATH_CALUDE_parallel_lines_and_not_always_parallel_planes_l733_73373


namespace NUMINAMATH_CALUDE_product_divisible_by_17_l733_73348

theorem product_divisible_by_17 : 
  17 ∣ (2002 + 3) * (2003 + 3) * (2004 + 3) * (2005 + 3) * (2006 + 3) * (2007 + 3) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_17_l733_73348


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l733_73309

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- The difference in jump distance between the frog and the grasshopper -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 22 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l733_73309


namespace NUMINAMATH_CALUDE_kelly_initial_games_kelly_initial_games_proof_l733_73371

theorem kelly_initial_games : ℕ → Prop :=
  fun initial : ℕ =>
    let found : ℕ := 31
    let give_away : ℕ := 105
    let remaining : ℕ := 6
    initial + found - give_away = remaining →
    initial = 80

-- The proof is omitted
theorem kelly_initial_games_proof : kelly_initial_games 80 := by sorry

end NUMINAMATH_CALUDE_kelly_initial_games_kelly_initial_games_proof_l733_73371


namespace NUMINAMATH_CALUDE_essay_section_length_l733_73391

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_multiplier : ℕ) 
  (num_body_sections : ℕ) 
  (total_length : ℕ) 
  (h1 : intro_length = 450)
  (h2 : conclusion_multiplier = 3)
  (h3 : num_body_sections = 4)
  (h4 : total_length = 5000)
  : (total_length - (intro_length + intro_length * conclusion_multiplier)) / num_body_sections = 800 := by
  sorry

end NUMINAMATH_CALUDE_essay_section_length_l733_73391


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l733_73307

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 12)) ∧
  Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l733_73307
