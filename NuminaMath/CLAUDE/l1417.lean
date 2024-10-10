import Mathlib

namespace exists_interior_points_l1417_141734

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is in the interior of a triangle -/
def interior_point (p : Point) (a b c : Point) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u + v + w = 1 ∧
  p.x = u * a.x + v * b.x + w * c.x ∧
  p.y = u * a.y + v * b.y + w * c.y

/-- The main theorem -/
theorem exists_interior_points (n : ℕ) (S : Finset Point) :
  S.card = n →
  (∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S → ¬collinear a b c) →
  ∃ (P : Finset Point), P.card = 2 * n - 5 ∧
    ∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S →
      ∃ (p : Point), p ∈ P ∧ interior_point p a b c :=
sorry

end exists_interior_points_l1417_141734


namespace tangent_circle_rectangle_area_l1417_141792

/-- A rectangle with a tangent circle passing through one vertex -/
structure TangentCircleRectangle where
  /-- Length of the rectangle -/
  l : ℝ
  /-- Width of the rectangle -/
  w : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to two adjacent sides of the rectangle -/
  tangent : l = 2 * r
  /-- The circle passes through the opposite corner -/
  passes_through : w = r

/-- The area of a rectangle with a tangent circle passing through one vertex is 2r² -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.l * rect.w = 2 * rect.r^2 := by
  sorry

end tangent_circle_rectangle_area_l1417_141792


namespace inequality_solution_l1417_141725

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≤ 1 / 2 ↔ -5 < x ∧ x ≤ 9 := by
  sorry

end inequality_solution_l1417_141725


namespace smallest_integer_in_set_l1417_141778

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 ≤ 2 * ((7 * n + 21) / 7)) → n ≥ 0 := by
  sorry

end smallest_integer_in_set_l1417_141778


namespace soap_brand_usage_ratio_l1417_141747

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_A : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_A = 60)
  (h4 : both = 5)
  (h5 : neither + only_A + both < total) :
  (total - (neither + only_A + both)) / both = 11 :=
by sorry

end soap_brand_usage_ratio_l1417_141747


namespace tire_circumference_l1417_141742

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 →
  car_speed_kmh = 96 →
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 4 := by
sorry

end tire_circumference_l1417_141742


namespace smallest_d_value_l1417_141783

theorem smallest_d_value (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h : ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3*d*x^2 + 3*x + e^3) :
  d ≥ 1 ∧ ∃ (a₀ b₀ c₀ e₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ e₀ > 0 ∧
    (∀ x : ℝ, (x + a₀) * (x + b₀) * (x + c₀) = x^3 + 3*x^2 + 3*x + e₀^3) := by
  sorry

#check smallest_d_value

end smallest_d_value_l1417_141783


namespace sallys_payment_l1417_141727

/-- Proves that the amount Sally paid with is $20, given that she bought 3 frames at $3 each and received $11 in change. -/
theorem sallys_payment (num_frames : ℕ) (frame_cost : ℕ) (change : ℕ) : 
  num_frames = 3 → frame_cost = 3 → change = 11 → 
  num_frames * frame_cost + change = 20 :=
by
  sorry

end sallys_payment_l1417_141727


namespace sequence_properties_l1417_141770

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

def c (a b : ℕ → ℝ) (n : ℕ) : ℝ := a n + b n

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q d : ℝ)
  (h_geom : geometric_sequence a q)
  (h_arith : arithmetic_sequence b d)
  (h_q : q ≠ 1)
  (h_d : d ≠ 0) :
  (¬ arithmetic_sequence (c a b) ((c a b 2) - (c a b 1))) ∧
  (a 1 = 1 ∧ q = 2 → 
    ∃ f : ℝ → ℝ, f d = (c a b 2) ∧ 
    (∀ x : ℝ, x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 0 → f x = x^2 + 3*x)) ∧
  (¬ geometric_sequence (c a b) ((c a b 2) / (c a b 1))) :=
sorry

end sequence_properties_l1417_141770


namespace era_burger_left_l1417_141789

/-- Represents the problem of Era's burger distribution --/
def era_burger_problem (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) : Prop :=
  total_burgers = 5 ∧
  num_friends = 4 ∧
  slices_per_burger = 2 ∧
  friend1_slices = 1 ∧
  friend2_slices = 2 ∧
  friend3_slices = 3 ∧
  friend4_slices = 3

/-- Theorem stating that Era has 1 slice of burger left --/
theorem era_burger_left (total_burgers num_friends slices_per_burger 
  friend1_slices friend2_slices friend3_slices friend4_slices : ℕ) :
  era_burger_problem total_burgers num_friends slices_per_burger 
    friend1_slices friend2_slices friend3_slices friend4_slices →
  total_burgers * slices_per_burger - (friend1_slices + friend2_slices + friend3_slices + friend4_slices) = 1 :=
by
  sorry

end era_burger_left_l1417_141789


namespace unique_solution_sqrt_equation_l1417_141701

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, Real.sqrt (2 * x + 6) - Real.sqrt (2 * x - 2) = 2 :=
by
  -- The unique solution is x = 1.5
  use (3/2)
  constructor
  · -- Prove that x = 1.5 satisfies the equation
    sorry
  · -- Prove that any solution must equal 1.5
    sorry

end unique_solution_sqrt_equation_l1417_141701


namespace tutor_schedules_lcm_l1417_141785

/-- The work schedules of the tutors -/
def tutor_schedules : List Nat := [5, 6, 8, 9, 10]

/-- The theorem stating that the LCM of the tutor schedules is 360 -/
theorem tutor_schedules_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10 = 360 := by
  sorry

end tutor_schedules_lcm_l1417_141785


namespace unique_solution_l1417_141790

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : ((x + 2)^2) / (y + z - 2) + ((y + 4)^2) / (z + x - 4) + ((z + 6)^2) / (x + y - 6) = 36) :
  x = 10 ∧ y = 8 ∧ z = 6 := by
sorry

end unique_solution_l1417_141790


namespace unique_solution_l1417_141756

/-- Jessica's work hours as a function of t -/
def jessica_hours (t : ℤ) : ℤ := 3 * t - 10

/-- Jessica's hourly rate as a function of t -/
def jessica_rate (t : ℤ) : ℤ := 4 * t - 9

/-- Bob's work hours as a function of t -/
def bob_hours (t : ℤ) : ℤ := t + 12

/-- Bob's hourly rate as a function of t -/
def bob_rate (t : ℤ) : ℤ := 2 * t + 1

/-- Predicate to check if t satisfies the equation -/
def satisfies_equation (t : ℤ) : Prop :=
  jessica_hours t * jessica_rate t = bob_hours t * bob_rate t

theorem unique_solution :
  ∃! t : ℤ, t > 3 ∧ satisfies_equation t := by sorry

end unique_solution_l1417_141756


namespace angle_between_vectors_l1417_141713

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  a = (1, 2) ∧ b = (3, 1) → angle = π / 4 := by sorry

end angle_between_vectors_l1417_141713


namespace sum_first_12_even_numbers_l1417_141748

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_12_even_numbers :
  (first_n_even_numbers 12).sum = 156 := by
  sorry

end sum_first_12_even_numbers_l1417_141748


namespace expression_equals_six_l1417_141722

theorem expression_equals_six :
  (Real.sqrt 27 + Real.sqrt 48) / Real.sqrt 3 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 6 := by
  sorry

end expression_equals_six_l1417_141722


namespace division_problem_l1417_141765

theorem division_problem (n : ℕ) : 
  n / 16 = 10 ∧ n % 16 = 1 → n = 161 := by
  sorry

end division_problem_l1417_141765


namespace quadratic_radicals_combination_l1417_141716

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ x + 1 = k * (2 * x)) → x = 1 := by
  sorry

end quadratic_radicals_combination_l1417_141716


namespace dog_reachable_area_l1417_141761

/-- The area outside a regular pentagon that a tethered dog can reach -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 3 → 
  ∃ (area : ℝ), area = 7.6 * Real.pi ∧ 
  area = (rope_length^2 * Real.pi * (288 / 360)) + 
         (2 * (side_length^2 * Real.pi * (72 / 360))) :=
sorry

end dog_reachable_area_l1417_141761


namespace two_rotational_homotheties_l1417_141758

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rotational homothety -/
structure RotationalHomothety where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- Applies a rotational homothety to a circle -/
def applyRotationalHomothety (h : RotationalHomothety) (c : Circle) : Circle :=
  sorry

/-- Checks if two circles are equal -/
def circlesEqual (c1 c2 : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem two_rotational_homotheties 
  (S₁ S₂ : Circle) 
  (h : S₁.center ≠ S₂.center) : 
  ∃! (pair : (RotationalHomothety × RotationalHomothety)),
    (circlesEqual (applyRotationalHomothety pair.1 S₁) S₂) ∧
    (circlesEqual (applyRotationalHomothety pair.2 S₁) S₂) ∧
    (pair.1.angle = π/2) ∧ (pair.2.angle = π/2) ∧
    (pair.1.center ≠ pair.2.center) :=
  sorry

end two_rotational_homotheties_l1417_141758


namespace quadratic_equation_solutions_linear_equation_solutions_l1417_141737

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 5*x - 1 = 0) ↔ (x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2) :=
sorry

theorem linear_equation_solutions (x : ℝ) :
  (7*x*(5*x + 2) = 6*(5*x + 2)) ↔ (x = -2/5 ∨ x = 6/7) :=
sorry

end quadratic_equation_solutions_linear_equation_solutions_l1417_141737


namespace quadratic_inequality_solution_set_l1417_141755

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0}
  (0 < a ∧ a < 1 → S = {x : ℝ | 1 < x ∧ x < 1/a}) ∧
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1/a < x ∧ x < 1}) :=
by sorry

end quadratic_inequality_solution_set_l1417_141755


namespace rain_probability_both_days_l1417_141795

theorem rain_probability_both_days 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (prob_sunday_given_saturday : ℝ) 
  (h1 : prob_saturday = 0.4)
  (h2 : prob_sunday = 0.3)
  (h3 : prob_sunday_given_saturday = 0.5) :
  prob_saturday * prob_sunday_given_saturday = 0.2 := by
  sorry

end rain_probability_both_days_l1417_141795


namespace min_value_expression_l1417_141760

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
  sorry

end min_value_expression_l1417_141760


namespace unique_common_root_existence_l1417_141762

theorem unique_common_root_existence :
  ∃! m : ℝ, ∃! x : ℝ, 
    (x^2 + m*x + 2 = 0 ∧ x^2 + 2*x + m = 0) ∧
    (∀ y : ℝ, y^2 + m*y + 2 = 0 ∧ y^2 + 2*y + m = 0 → y = x) ∧
    m = -3 ∧ x = 1 :=
by sorry

end unique_common_root_existence_l1417_141762


namespace quadratic_function_inequality_l1417_141750

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + 4 - a

-- State the theorem
theorem quadratic_function_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ > x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  f a x₁ > f a x₂ := by
  sorry

end quadratic_function_inequality_l1417_141750


namespace stratified_sampling_group_b_l1417_141754

-- Define the total number of cities and the number in Group B
def total_cities : ℕ := 48
def group_b_cities : ℕ := 18

-- Define the total sample size
def sample_size : ℕ := 16

-- Define the function to calculate the number of cities to sample from Group B
def cities_to_sample_from_b : ℕ := 
  (group_b_cities * sample_size) / total_cities

-- Theorem statement
theorem stratified_sampling_group_b :
  cities_to_sample_from_b = 6 := by sorry

end stratified_sampling_group_b_l1417_141754


namespace max_distance_on_C_l1417_141731

noncomputable section

open Real

-- Define the curve in polar coordinates
def C (θ : ℝ) : ℝ := 4 * sin θ

-- Define a point on the curve
def point_on_C (θ : ℝ) : ℝ × ℝ := (C θ * cos θ, C θ * sin θ)

-- Define the distance between two points on the curve
def distance_on_C (θ₁ θ₂ : ℝ) : ℝ :=
  let (x₁, y₁) := point_on_C θ₁
  let (x₂, y₂) := point_on_C θ₂
  sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem max_distance_on_C :
  ∃ (M : ℝ), M = 4 ∧ ∀ (θ₁ θ₂ : ℝ), distance_on_C θ₁ θ₂ ≤ M :=
sorry

end

end max_distance_on_C_l1417_141731


namespace number_2018_in_group_27_l1417_141741

/-- The sum of even numbers up to the k-th group -/
def S (k : ℕ) : ℕ := (3 * k^2 - k) / 2

/-- The proposition that 2018 belongs to the 27th group -/
theorem number_2018_in_group_27 : 
  S 26 < 1009 ∧ 1009 ≤ S 27 := by sorry

end number_2018_in_group_27_l1417_141741


namespace decimal_expansion_three_eighths_no_repeat_l1417_141702

/-- The length of the smallest repeating block in the decimal expansion of 3/8 is 0. -/
theorem decimal_expansion_three_eighths_no_repeat : 
  (∃ n : ℕ, ∃ k : ℕ, (3 : ℚ) / 8 = (n : ℚ) / 10^k ∧ k > 0) := by sorry

end decimal_expansion_three_eighths_no_repeat_l1417_141702


namespace probability_is_two_fifths_l1417_141788

/-- A diagram with five triangles, two of which are shaded. -/
structure Diagram where
  triangles : Finset (Fin 5)
  shaded : Finset (Fin 5)
  total_triangles : triangles.card = 5
  shaded_triangles : shaded.card = 2
  shaded_subset : shaded ⊆ triangles

/-- The probability of selecting a shaded triangle from the diagram. -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded.card / d.triangles.card

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5. -/
theorem probability_is_two_fifths (d : Diagram) :
  probability_shaded d = 2/5 := by
  sorry

end probability_is_two_fifths_l1417_141788


namespace factorization_equality_l1417_141711

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y^3 = 3 * y * (x + y) * (x - y) := by
  sorry

end factorization_equality_l1417_141711


namespace cookies_eaten_l1417_141712

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 28) (h2 : remaining = 7) :
  initial - remaining = 21 := by
  sorry

end cookies_eaten_l1417_141712


namespace simplify_sqrt_expression_l1417_141796

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 242 / Real.sqrt 121) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end simplify_sqrt_expression_l1417_141796


namespace product_of_four_consecutive_integers_l1417_141773

theorem product_of_four_consecutive_integers (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 := by
  sorry

end product_of_four_consecutive_integers_l1417_141773


namespace max_distance_origin_to_line_l1417_141787

/-- Given a line l with equation ax + by + c = 0, where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin O(0,0) to the line l is √5. -/
theorem max_distance_origin_to_line (a b c : ℝ) :
  (∃ d : ℝ, a - b = b - c) →  -- a, b, c form an arithmetic sequence
  (∃ x y : ℝ, a * x + b * y + c = 0) →  -- line equation exists
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≥ Real.sqrt (x^2 + y^2)) →  -- distance definition
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≤ Real.sqrt 5) →  -- upper bound
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ Real.sqrt (x^2 + y^2) = Real.sqrt 5)  -- maximum distance achieved
  := by sorry

end max_distance_origin_to_line_l1417_141787


namespace area_ratio_of_squares_l1417_141745

theorem area_ratio_of_squares (side_A side_B : ℝ) (h1 : side_A = 36) (h2 : side_B = 42) :
  (side_A ^ 2) / (side_B ^ 2) = 36 / 49 := by
  sorry

end area_ratio_of_squares_l1417_141745


namespace divisible_by_six_l1417_141735

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11*(m : ℤ) = 6*k := by sorry

end divisible_by_six_l1417_141735


namespace valerie_skipping_rate_l1417_141719

/-- Roberto's skipping rate in skips per hour -/
def roberto_rate : ℕ := 4200

/-- Total skips for Roberto and Valerie in 15 minutes -/
def total_skips : ℕ := 2250

/-- Duration of skipping in minutes -/
def duration : ℕ := 15

/-- Valerie's skipping rate in skips per minute -/
def valerie_rate : ℕ := 80

theorem valerie_skipping_rate :
  (roberto_rate * duration / 60 + valerie_rate * duration = total_skips) ∧
  (valerie_rate = (total_skips - roberto_rate * duration / 60) / duration) :=
sorry

end valerie_skipping_rate_l1417_141719


namespace linear_equation_condition_l1417_141726

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a + 3) * x^(|a| - 2) + 5 = k * x + m) → a = 3 := by
  sorry

end linear_equation_condition_l1417_141726


namespace park_fencing_cost_l1417_141705

/-- Proves that the fencing cost per meter is 50 paise for a rectangular park with given conditions -/
theorem park_fencing_cost
  (ratio : ℚ) -- Ratio of length to width
  (area : ℝ) -- Area of the park in square meters
  (total_cost : ℝ) -- Total cost of fencing
  (h_ratio : ratio = 3 / 2) -- The sides are in the ratio 3:2
  (h_area : area = 7350) -- The area is 7350 sq m
  (h_total_cost : total_cost = 175) -- The total cost of fencing is 175
  : ℝ :=
by
  -- Proof goes here
  sorry

#check park_fencing_cost

end park_fencing_cost_l1417_141705


namespace min_value_theorem_l1417_141732

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + y = 1) : 
  (∀ x' y' : ℝ, x' > -1 → y' > 0 → x' + y' = 1 → 
    1 / (x' + 1) + 4 / y' ≥ 1 / (x + 1) + 4 / y) ∧ 
  1 / (x + 1) + 4 / y = 9/2 := by
sorry

end min_value_theorem_l1417_141732


namespace smallest_square_multiplier_l1417_141780

def y : ℕ := 2^10 * 3^15 * 4^20 * 5^25 * 6^30 * 7^35 * 8^40 * 9^45

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ 
  (∀ k : ℕ, k < n → ¬∃ m : ℕ, k * y = m^2) ↔ 
  n = 105 :=
sorry

end smallest_square_multiplier_l1417_141780


namespace gumball_theorem_l1417_141736

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : Nat)
  (white : Nat)
  (blue : Nat)
  (green : Nat)

/-- The least number of gumballs needed to guarantee four of the same color -/
def leastGumballsNeeded (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine, 
    the least number of gumballs needed is 13 -/
theorem gumball_theorem (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 7) :
  leastGumballsNeeded machine = 13 := by
  sorry

#check gumball_theorem

end gumball_theorem_l1417_141736


namespace farm_sheep_ratio_l1417_141791

/-- Proves that the ratio of sheep sold to total sheep is 2:3 given the farm conditions --/
theorem farm_sheep_ratio :
  ∀ (goats sheep sold_sheep : ℕ) (sale_amount : ℚ),
    goats + sheep = 360 →
    goats * 7 = sheep * 5 →
    sale_amount = 7200 →
    sale_amount = (goats / 2) * 40 + sold_sheep * 30 →
    sold_sheep / sheep = 2 / 3 :=
by sorry


end farm_sheep_ratio_l1417_141791


namespace green_percentage_approx_l1417_141763

/-- Represents the count of people preferring each color --/
structure ColorPreferences where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the percentage of people who preferred green --/
def greenPercentage (prefs : ColorPreferences) : ℚ :=
  (prefs.green : ℚ) / (prefs.red + prefs.blue + prefs.green + prefs.yellow + prefs.purple + prefs.orange) * 100

/-- Theorem stating that the percentage of people who preferred green is approximately 16.67% --/
theorem green_percentage_approx (prefs : ColorPreferences)
  (h1 : prefs.red = 70)
  (h2 : prefs.blue = 80)
  (h3 : prefs.green = 50)
  (h4 : prefs.yellow = 40)
  (h5 : prefs.purple = 30)
  (h6 : prefs.orange = 30) :
  ∃ ε > 0, |greenPercentage prefs - 50/3| < ε ∧ ε < 1/100 := by
  sorry

end green_percentage_approx_l1417_141763


namespace total_amount_spent_l1417_141786

def meal_prices : List Float := [12, 15, 10, 18, 20]
def ice_cream_prices : List Float := [2, 3, 3, 4, 4]
def tip_percentage : Float := 0.15
def tax_percentage : Float := 0.08

theorem total_amount_spent :
  let total_meal_cost := meal_prices.sum
  let total_ice_cream_cost := ice_cream_prices.sum
  let tip := tip_percentage * total_meal_cost
  let tax := tax_percentage * total_meal_cost
  total_meal_cost + total_ice_cream_cost + tip + tax = 108.25 := by
sorry

end total_amount_spent_l1417_141786


namespace logarithm_exponent_equality_special_case_2019_l1417_141774

theorem logarithm_exponent_equality (x : ℝ) (hx : x > 1) : 
  x^(Real.log (Real.log x)) = (Real.log x)^(Real.log x) :=
by
  sorry

-- The main theorem
theorem special_case_2019 : 
  2019^(Real.log (Real.log 2019)) - (Real.log 2019)^(Real.log 2019) = 0 :=
by
  sorry

end logarithm_exponent_equality_special_case_2019_l1417_141774


namespace cube_sum_equation_l1417_141715

/-- Given real numbers p, q, and r satisfying certain conditions, 
    their cubes sum to 181. -/
theorem cube_sum_equation (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 := by
  sorry


end cube_sum_equation_l1417_141715


namespace quadratic_inequality_solution_set_l1417_141749

theorem quadratic_inequality_solution_set 
  (a b c m n : ℝ) 
  (h1 : a ≠ 0)
  (h2 : m > 0)
  (h3 : Set.Ioo m n = {x | a * x^2 + b * x + c > 0}) :
  {x | c * x^2 + b * x + a < 0} = Set.Iic (1/n) ∪ Set.Ioi (1/m) := by
  sorry

end quadratic_inequality_solution_set_l1417_141749


namespace tan_alpha_value_l1417_141772

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 3) : Real.tan α = 1/2 := by
  sorry

end tan_alpha_value_l1417_141772


namespace jerry_lawsuit_amount_correct_l1417_141709

/-- Calculates the amount Jerry gets from his lawsuit --/
def jerryLawsuitAmount (annualSalary : ℕ) (years : ℕ) (medicalBills : ℕ) (punitiveMultiplier : ℕ) (awardedPercentage : ℚ) : ℚ :=
  let totalSalary := annualSalary * years
  let directDamages := totalSalary + medicalBills
  let punitiveDamages := directDamages * punitiveMultiplier
  let totalAsked := directDamages + punitiveDamages
  totalAsked * awardedPercentage

theorem jerry_lawsuit_amount_correct :
  jerryLawsuitAmount 50000 30 200000 3 (4/5) = 5440000 := by
  sorry

#eval jerryLawsuitAmount 50000 30 200000 3 (4/5)

end jerry_lawsuit_amount_correct_l1417_141709


namespace geometric_sequence_ratio_l1417_141706

/-- Given a geometric sequence with common ratio 2, prove that S_4 / a_2 = 15/2 --/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio is 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / a 2 = 15 / 2 := by
sorry

end geometric_sequence_ratio_l1417_141706


namespace quadratic_roots_sum_l1417_141794

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 5*m - 2023 = 0) → (n^2 + 5*n - 2023 = 0) → m^2 + 7*m + 2*n = 2013 := by
  sorry

end quadratic_roots_sum_l1417_141794


namespace z_in_second_quadrant_l1417_141757

-- Define the complex number z
def z : ℂ := -1 + 3 * Complex.I

-- Theorem stating that z is in the second quadrant
theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end z_in_second_quadrant_l1417_141757


namespace solution_of_system_l1417_141717

theorem solution_of_system (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂) ∧ 
  (2 * x₂^2 / (1 + x₂^2) = x₃) ∧ 
  (2 * x₃^2 / (1 + x₃^2) = x₁) → 
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) := by
  sorry

end solution_of_system_l1417_141717


namespace miquel_point_midpoint_l1417_141704

-- Define the points
variable (A B C D O M T S : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define O as the intersection of diagonals
def is_diagonal_intersection (O A B C D : EuclideanPlane) : Prop := sorry

-- Define the circumcircles
def on_circumcircle (P Q R S : EuclideanPlane) : Prop := sorry

-- Define M as the intersection of circumcircles OAD and OBC
def is_circumcircle_intersection (M O A B C D : EuclideanPlane) : Prop := sorry

-- Define T and S on the line OM and on their respective circumcircles
def on_line_and_circumcircle (P Q R S T : EuclideanPlane) : Prop := sorry

-- Define the midpoint
def is_midpoint (M S T : EuclideanPlane) : Prop := sorry

-- The theorem
theorem miquel_point_midpoint 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_diagonal_intersection O A B C D)
  (h3 : is_circumcircle_intersection M O A B C D)
  (h4 : on_line_and_circumcircle O M A B T)
  (h5 : on_line_and_circumcircle O M C D S) :
  is_midpoint M S T := by sorry

end miquel_point_midpoint_l1417_141704


namespace parabola_point_distances_l1417_141768

theorem parabola_point_distances (a c : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  a > 0 →
  y₂ = -9 * a + c →
  y₁ = a * x₁^2 - 6 * a * x₁ + c →
  y₂ = a * x₂^2 - 6 * a * x₂ + c →
  y₃ = a * x₃^2 - 6 * a * x₃ + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₂ - x₃| :=
by sorry

end parabola_point_distances_l1417_141768


namespace tyler_saltwater_animals_l1417_141798

/-- Represents the number of aquariums of each type -/
structure AquariumCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of animals in each type of aquarium -/
structure AquariumAnimals where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total number of saltwater animals -/
def totalSaltwaterAnimals (counts : AquariumCounts) (animals : AquariumAnimals) : ℕ :=
  counts.typeA * animals.typeA + counts.typeB * animals.typeB + counts.typeC * animals.typeC

/-- Tyler's aquarium setup -/
def tylerAquariums : AquariumCounts :=
  { typeA := 10
    typeB := 14
    typeC := 6 }

/-- Number of animals in each type of Tyler's aquariums -/
def tylerAnimals : AquariumAnimals :=
  { typeA := 12 * 4  -- 12 corals with 4 animals each
    typeB := 18 + 10 -- 18 large fish and 10 small fish
    typeC := 25 + 20 -- 25 invertebrates and 20 small fish
  }

theorem tyler_saltwater_animals :
  totalSaltwaterAnimals tylerAquariums tylerAnimals = 1142 := by
  sorry

end tyler_saltwater_animals_l1417_141798


namespace integers_abs_le_two_l1417_141720

theorem integers_abs_le_two : 
  {x : ℤ | |x| ≤ 2} = {-2, -1, 0, 1, 2} := by sorry

end integers_abs_le_two_l1417_141720


namespace tangent_line_equation_l1417_141767

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- The tangent line -/
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- Theorem: The tangent line to circle C passing through point P has the equation x + 2y - 10 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, C x y → (x = P.1 ∧ y = P.2) → tangent_line x y :=
sorry

end tangent_line_equation_l1417_141767


namespace missing_number_equation_l1417_141707

theorem missing_number_equation : ∃ x : ℤ, 1234562 - 12 * 3 * x = 1234490 ∧ x = 2 := by
  sorry

end missing_number_equation_l1417_141707


namespace fox_initial_money_l1417_141797

/-- The amount of money the fox has after n bridge crossings -/
def fox_money (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => 2 * fox_money a₀ n - 2^2019

theorem fox_initial_money :
  ∀ a₀ : ℕ, fox_money a₀ 2019 = 0 → a₀ = 2^2019 - 1 := by
  sorry

#check fox_initial_money

end fox_initial_money_l1417_141797


namespace units_digit_of_sum_cubes_l1417_141703

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_cubes : units_digit (24^3 + 42^3) = 2 := by
  sorry

end units_digit_of_sum_cubes_l1417_141703


namespace range_of_a_l1417_141740

/-- The set of x satisfying p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of x satisfying q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : set_p a ⊆ set_q)
  (h3 : (Set.univ \ set_p a) ⊂ (Set.univ \ set_q)) :
  -4 ≤ a ∧ a < 0 ∨ a ≤ -4 := by
  sorry

end range_of_a_l1417_141740


namespace p_iff_q_l1417_141752

theorem p_iff_q (a b : ℝ) :
  (a > 2 ∧ b > 3) ↔ (a + b > 5 ∧ (a - 2) * (b - 3) > 0) := by
  sorry

end p_iff_q_l1417_141752


namespace final_price_after_discounts_l1417_141769

/-- Calculates the final price of an item after two successive discounts --/
theorem final_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 200 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

#check final_price_after_discounts

end final_price_after_discounts_l1417_141769


namespace hyperbola_focus_a_value_l1417_141739

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = Real.sqrt 13 ∧ y = 0

-- Theorem statement
theorem hyperbola_focus_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, hyperbola x y a → right_focus x y) → a = 4 :=
by sorry

end hyperbola_focus_a_value_l1417_141739


namespace smallest_number_proof_l1417_141777

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end smallest_number_proof_l1417_141777


namespace cos_is_even_l1417_141728

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end cos_is_even_l1417_141728


namespace total_tv_time_l1417_141784

def missy_reality_shows : List ℕ := [28, 35, 42, 39, 29]
def missy_cartoons : ℕ := 2
def missy_cartoon_duration : ℕ := 10

def john_action_movies : List ℕ := [90, 110, 95]
def john_comedy_duration : ℕ := 25

def lily_documentaries : List ℕ := [45, 55, 60, 52]

def ad_breaks : List ℕ := [8, 6, 12, 9, 7, 11]

def num_viewers : ℕ := 3

theorem total_tv_time :
  (missy_reality_shows.sum + missy_cartoons * missy_cartoon_duration +
   john_action_movies.sum + john_comedy_duration +
   lily_documentaries.sum +
   num_viewers * ad_breaks.sum) = 884 := by
  sorry

end total_tv_time_l1417_141784


namespace abcg_over_defh_value_l1417_141721

theorem abcg_over_defh_value (a b c d e f g h : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4)
  (h8 : b ≠ 0)
  (h9 : c ≠ 0)
  (h10 : d ≠ 0)
  (h11 : e ≠ 0)
  (h12 : f ≠ 0)
  (h13 : g ≠ 0)
  (h14 : h ≠ 0) :
  a * b * c * g / (d * e * f * h) = 5 / 48 := by
  sorry

end abcg_over_defh_value_l1417_141721


namespace prime_quadratic_integer_roots_l1417_141781

theorem prime_quadratic_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ^ 2 - p * x - 156 * p = 0 ∧ y ^ 2 - p * y - 156 * p = 0) → 
  p = 13 := by
sorry

end prime_quadratic_integer_roots_l1417_141781


namespace unique_p_for_equal_roots_l1417_141718

/-- The quadratic equation x^2 - px + p^2 = 0 has equal roots for exactly one real value of p -/
theorem unique_p_for_equal_roots :
  ∃! p : ℝ, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! y : ℝ, y^2 - p*y + p^2 = 0) := by sorry

end unique_p_for_equal_roots_l1417_141718


namespace cubic_polynomial_satisfies_conditions_l1417_141738

def q (x : ℚ) : ℚ := (17 * x^3 - 30 * x^2 + x + 12) / 6

theorem cubic_polynomial_satisfies_conditions :
  q (-1) = -6 ∧ q 2 = 5 ∧ q 0 = 2 ∧ q 1 = 0 := by
  sorry

end cubic_polynomial_satisfies_conditions_l1417_141738


namespace may_total_scarves_l1417_141746

/-- The number of scarves that can be knitted from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May can make -/
def total_scarves : ℕ := scarves_per_yarn * (red_yarns + blue_yarns + yellow_yarns)

theorem may_total_scarves : total_scarves = 36 := by
  sorry

end may_total_scarves_l1417_141746


namespace alf3_weight_l1417_141751

/-- The molecular weight of a compound -/
def molecularWeight (alWeight fWeight : ℝ) : ℝ := alWeight + 3 * fWeight

/-- The total weight of a given number of moles of a compound -/
def totalWeight (molWeight : ℝ) (moles : ℝ) : ℝ := molWeight * moles

/-- Theorem stating the total weight of 10 moles of aluminum fluoride -/
theorem alf3_weight : 
  let alWeight : ℝ := 26.98
  let fWeight : ℝ := 19.00
  let moles : ℝ := 10
  totalWeight (molecularWeight alWeight fWeight) moles = 839.8 := by
sorry

end alf3_weight_l1417_141751


namespace cube_dot_path_length_l1417_141710

theorem cube_dot_path_length (cube_edge : ℝ) (dot_path : ℝ) : 
  cube_edge = 2 →
  dot_path = Real.sqrt 5 * Real.pi →
  ∃ (rotation_radius : ℝ),
    rotation_radius = Real.sqrt (1^2 + 2^2) ∧
    dot_path = 4 * (1/4 * 2 * Real.pi * rotation_radius) :=
by sorry

end cube_dot_path_length_l1417_141710


namespace pizza_slices_per_person_l1417_141776

theorem pizza_slices_per_person 
  (total_slices : Nat) 
  (people : Nat) 
  (slices_left : Nat) 
  (h1 : total_slices = 16) 
  (h2 : people = 6) 
  (h3 : slices_left = 4) 
  (h4 : people > 0) : 
  (total_slices - slices_left) / people = 2 := by
sorry

end pizza_slices_per_person_l1417_141776


namespace same_solution_k_value_l1417_141753

theorem same_solution_k_value : ∃ (k : ℝ), 
  (∀ (x : ℝ), (2 * x + 4 = 4 * (x - 2)) ↔ (-x + k = 2 * x - 1)) → k = 17 := by
  sorry

end same_solution_k_value_l1417_141753


namespace cars_clearing_time_l1417_141700

/-- Calculates the time for two cars to be clear of each other from the moment they meet -/
theorem cars_clearing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 20 := by
  sorry

#check cars_clearing_time

end cars_clearing_time_l1417_141700


namespace completing_square_transformation_l1417_141779

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2*x - 7 = 0 ↔ (x - 1)^2 = 8 :=
by sorry

end completing_square_transformation_l1417_141779


namespace sum_of_circle_areas_l1417_141729

/-- Given a right triangle with sides 5, 12, and 13, where the vertices are centers of
    three mutually externally tangent circles, the sum of the areas of these circles is 113π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r + s = b →
  s + t = a →
  r + t = c →
  π * (r^2 + s^2 + t^2) = 113 * π := by
sorry

end sum_of_circle_areas_l1417_141729


namespace star_equation_solution_l1417_141744

def star (a b : ℝ) : ℝ := a^2 * b + 2 * b - a

theorem star_equation_solution :
  ∀ x : ℝ, star 7 x = 85 → x = 92 / 51 := by
  sorry

end star_equation_solution_l1417_141744


namespace distance_scientific_notation_equivalence_l1417_141714

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The distance between two mountain peaks in meters -/
def distance : ℝ := 14000000

/-- The scientific notation representation of the distance -/
def distanceScientific : ScientificNotation := {
  coefficient := 1.4
  exponent := 7
  h1 := by sorry
}

theorem distance_scientific_notation_equivalence :
  distance = distanceScientific.coefficient * (10 : ℝ) ^ distanceScientific.exponent :=
by sorry

end distance_scientific_notation_equivalence_l1417_141714


namespace triangle_formation_l1417_141793

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths of any two sides
    is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given three line segments with lengths 4cm, 5cm, and 6cm, they can form a triangle. -/
theorem triangle_formation : can_form_triangle 4 5 6 := by
  sorry

end triangle_formation_l1417_141793


namespace cafeteria_shirts_l1417_141771

theorem cafeteria_shirts (total : Nat) (vertical : Nat) 
  (h1 : total = 40)
  (h2 : vertical = 5)
  (h3 : ∃ (checkered : Nat), total = checkered + 4 * checkered + vertical) :
  ∃ (checkered : Nat), checkered = 7 ∧ 
    total = checkered + 4 * checkered + vertical := by
  sorry

end cafeteria_shirts_l1417_141771


namespace y_intersection_is_six_l1417_141743

/-- The quadratic function f(x) = -2(x-1)(x+3) -/
def f (x : ℝ) : ℝ := -2 * (x - 1) * (x + 3)

/-- The y-coordinate of the intersection point with the y-axis is 6 -/
theorem y_intersection_is_six : f 0 = 6 := by
  sorry

end y_intersection_is_six_l1417_141743


namespace exists_multiple_factorization_l1417_141730

/-- The set Vn for a given n > 2 -/
def Vn (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is indecomposable in Vn if it cannot be expressed as a product of two elements in Vn -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ Vn n ∧ ¬∃ p q : ℕ, p ∈ Vn n ∧ q ∈ Vn n ∧ p * q = m

/-- The main theorem statement -/
theorem exists_multiple_factorization (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ Vn n ∧
    ∃ (f₁ f₂ : List ℕ),
      f₁ ≠ f₂ ∧
      (∀ x ∈ f₁, Indecomposable n x) ∧
      (∀ x ∈ f₂, Indecomposable n x) ∧
      r = f₁.prod ∧
      r = f₂.prod :=
sorry

end exists_multiple_factorization_l1417_141730


namespace pascals_triangle_divisibility_l1417_141733

theorem pascals_triangle_divisibility (p : ℕ) (hp : Prime p) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ¬(p ∣ Nat.choose n k)) ↔
  ∃ (s q : ℕ), s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by sorry

end pascals_triangle_divisibility_l1417_141733


namespace abs_two_minus_sqrt_five_l1417_141724

theorem abs_two_minus_sqrt_five : |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l1417_141724


namespace consecutive_squares_sum_l1417_141782

theorem consecutive_squares_sum (x : ℕ) :
  x^2 + (x+1)^2 + (x+2)^2 = 2030 → x + 1 = 26 := by
  sorry

end consecutive_squares_sum_l1417_141782


namespace least_four_digit_13_heavy_l1417_141723

theorem least_four_digit_13_heavy : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 13 > 8 → n ≥ 1004 :=
by sorry

end least_four_digit_13_heavy_l1417_141723


namespace sum_of_angles_l1417_141759

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^5 = -32 * i

-- Define the form of solutions
def solution_form (z : ℂ) (s : ℝ) (α : ℝ) : Prop :=
  z = s * (Complex.cos α + i * Complex.sin α)

-- Define the conditions on s and α
def valid_solution (s : ℝ) (α : ℝ) : Prop :=
  s > 0 ∧ 0 ≤ α ∧ α < 2 * Real.pi

-- Theorem statement
theorem sum_of_angles (z₁ z₂ z₃ z₄ z₅ : ℂ) (s₁ s₂ s₃ s₄ s₅ α₁ α₂ α₃ α₄ α₅ : ℝ) :
  equation z₁ ∧ equation z₂ ∧ equation z₃ ∧ equation z₄ ∧ equation z₅ ∧
  solution_form z₁ s₁ α₁ ∧ solution_form z₂ s₂ α₂ ∧ solution_form z₃ s₃ α₃ ∧
  solution_form z₄ s₄ α₄ ∧ solution_form z₅ s₅ α₅ ∧
  valid_solution s₁ α₁ ∧ valid_solution s₂ α₂ ∧ valid_solution s₃ α₃ ∧
  valid_solution s₄ α₄ ∧ valid_solution s₅ α₅ →
  α₁ + α₂ + α₃ + α₄ + α₅ = 5.5 * Real.pi := by
  sorry

end sum_of_angles_l1417_141759


namespace farm_animals_l1417_141775

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8)
  (h2 : total_legs = 24) :
  ∃ (ducks dogs : ℕ),
    ducks + dogs = total_animals ∧
    2 * ducks + 4 * dogs = total_legs ∧
    ducks = 4 := by
  sorry

end farm_animals_l1417_141775


namespace probability_two_teams_play_l1417_141708

/-- The probability that two specific teams play each other in a single-elimination tournament -/
theorem probability_two_teams_play (n : ℕ) (h : n = 16) : 
  (2 : ℚ) / ((n : ℚ) * (n - 1)) = 1 / 8 :=
sorry

end probability_two_teams_play_l1417_141708


namespace polynomial_without_xy_term_l1417_141799

theorem polynomial_without_xy_term (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 - 3*y^2 - 8) ↔ k = 2 := by
  sorry

end polynomial_without_xy_term_l1417_141799


namespace quadratic_one_solution_l1417_141764

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end quadratic_one_solution_l1417_141764


namespace sqrt_32_div_sqrt_2_eq_4_l1417_141766

theorem sqrt_32_div_sqrt_2_eq_4 : Real.sqrt 32 / Real.sqrt 2 = 4 := by
  sorry

end sqrt_32_div_sqrt_2_eq_4_l1417_141766
