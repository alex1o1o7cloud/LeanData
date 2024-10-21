import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1021_102128

theorem inequality_proof (x y : ℝ) :
  x^2 * Real.sqrt (1 + 2*y^2) + y^2 * Real.sqrt (1 + 2*x^2) ≥ x*y*(x + y + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1021_102128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_is_cos_l1021_102108

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2013_is_cos : f 2013 = Real.cos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_is_cos_l1021_102108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_5_factorial_l1021_102115

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_factor (a b : ℕ) : Bool := b % a = 0

def count_factors (n : ℕ) (set : List ℕ) : ℕ :=
  set.filter (λ x => is_factor x n) |>.length

theorem probability_factor_5_factorial : 
  let set := List.range 24
  let five_factorial := factorial 5
  (count_factors five_factorial set : ℚ) / (set.length : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_5_factorial_l1021_102115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grove_cleaning_time_theorem_l1021_102132

/-- Represents the time taken to clean a row of trees -/
structure RowCleaningTime where
  baseTime : ℚ
  helpers : ℕ

/-- Calculates the actual cleaning time for a row given the base time and number of helpers -/
def actualCleaningTime (row : RowCleaningTime) : ℚ :=
  row.baseTime / (row.helpers + 1 : ℚ)

/-- Represents the grove and cleaning conditions -/
structure GroveCleaningConditions where
  rows : ℕ
  firstRow : RowCleaningTime
  secondRow : RowCleaningTime
  remainingRows : RowCleaningTime

/-- Calculates the total cleaning time for the grove -/
def totalCleaningTime (grove : GroveCleaningConditions) : ℚ :=
  actualCleaningTime grove.firstRow +
  actualCleaningTime grove.secondRow +
  (grove.rows - 2 : ℚ) * actualCleaningTime grove.remainingRows

/-- The main theorem stating that the total cleaning time is approximately 0.689 hours -/
theorem grove_cleaning_time_theorem (grove : GroveCleaningConditions)
  (h1 : grove.rows = 8)
  (h2 : grove.firstRow = { baseTime := 8, helpers := 3 })
  (h3 : grove.secondRow = { baseTime := 10, helpers := 2 })
  (h4 : grove.remainingRows = { baseTime := 12, helpers := 1 }) :
  ∃ ε > 0, |totalCleaningTime grove / 60 - 689/1000| < ε := by
  sorry

#eval totalCleaningTime {
  rows := 8,
  firstRow := { baseTime := 8, helpers := 3 },
  secondRow := { baseTime := 10, helpers := 2 },
  remainingRows := { baseTime := 12, helpers := 1 }
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grove_cleaning_time_theorem_l1021_102132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1021_102116

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- The problem statement -/
theorem distance_point_to_line_example : 
  distance_point_to_line 2 0 1 (-1) (-1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l1021_102116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1021_102191

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (2 * x - 1)

-- State the theorem
theorem max_value_of_f :
  ∃ y : ℝ, y = -1 ∧ ∀ x < (1/2 : ℝ), f x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1021_102191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_dry_short_haired_l1021_102120

/-- Represents the time in minutes to dry a short-haired dog -/
def time_short_haired : ℕ := sorry

/-- Represents the time in minutes to dry a full-haired dog -/
def time_full_haired : ℕ := sorry

/-- The number of short-haired dogs -/
def num_short_haired : ℕ := 6

/-- The number of full-haired dogs -/
def num_full_haired : ℕ := 9

/-- The total time in minutes to dry all dogs -/
def total_time : ℕ := 4 * 60

axiom full_haired_double : time_full_haired = 2 * time_short_haired

theorem time_to_dry_short_haired : 
  time_short_haired * num_short_haired + time_full_haired * num_full_haired = total_time →
  time_short_haired = 10 := by
  sorry

#check time_to_dry_short_haired

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_dry_short_haired_l1021_102120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_seating_l1021_102179

structure Society where
  n : ℕ
  members : Finset (Fin n)
  knows : Fin n → Fin n → Bool
  knows_sym : ∀ i j, knows i j = knows j i
  knows_lower_bound : ∀ i, ∃ j, i ≠ j ∧ knows i j = true
  knows_upper_bound : ∀ i, (members.filter (fun j => knows i j)).card ≤ n - 2

def valid_seating (s : Society) (a b c d : Fin s.n) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  s.knows a d = true ∧ s.knows a b = false ∧ s.knows a c = false ∧
  s.knows b c = true ∧ s.knows b d = false ∧
  s.knows c d = false

theorem exists_valid_seating (s : Society) : ∃ a b c d, valid_seating s a b c d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_seating_l1021_102179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_at_start_l1021_102187

def horse_lap_time (k : ℕ) : ℕ :=
  if k % 2 = 0 then k else 2 * k

def is_at_start (t k : ℕ) : Bool :=
  t % (horse_lap_time k) = 0

def count_horses_at_start (t : ℕ) : ℕ :=
  (List.range 12).filter (λ k => is_at_start t (k + 1)) |>.length

theorem least_time_six_horses_at_start :
  ∀ t, t > 0 → count_horses_at_start t ≥ 6 → t ≥ 120 ∧
  count_horses_at_start 120 ≥ 6 :=
by
  sorry

#eval count_horses_at_start 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_at_start_l1021_102187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_2a_nonnegative_implies_a_range_l1021_102174

theorem sin_minus_2a_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, Real.sin x - 2 * a ≥ 0) ↔ a ∈ Set.Iic (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_2a_nonnegative_implies_a_range_l1021_102174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1021_102162

-- Define the sets A and B
def A : Set ℝ := {-1, 2}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

-- Define the theorem
theorem problem_solution : 
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({-1/2, 0, 1} : Set ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1021_102162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_9a_count_l1021_102161

/-- The number of students from grade 9A -/
def a : ℤ := sorry

/-- The number of students from grade 9B -/
def b : ℤ := sorry

/-- The number of students from grade 9C -/
def c : ℤ := sorry

/-- Mary's count for 9A is within 2 of the actual number -/
axiom mary_count_9a : abs (a - 27) ≤ 2

/-- Mary's count for 9B is within 2 of the actual number -/
axiom mary_count_9b : abs (b - 29) ≤ 2

/-- Mary's count for 9C is within 2 of the actual number -/
axiom mary_count_9c : abs (c - 30) ≤ 2

/-- Ilia's total count is within 4 of the actual total -/
axiom ilia_total_count : abs ((a + b + c) - 96) ≤ 4

/-- The theorem to be proved -/
theorem students_9a_count : a = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_9a_count_l1021_102161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_blue_pill_cost_l1021_102192

/-- The cost of a blue pill given the conditions of Bob's treatment --/
def blue_pill_cost (treatment_duration : ℕ) (daily_blue_pills : ℕ) (daily_red_pills : ℕ) 
  (blue_red_diff : ℚ) (total_cost : ℚ) : ℚ :=
  let days : ℕ := treatment_duration * 7
  let daily_cost : ℚ := total_cost / days
  (daily_cost + blue_red_diff) / 2

/-- The specific instance of Bob's treatment --/
theorem bobs_blue_pill_cost : 
  blue_pill_cost 3 1 1 2 819 = 20.5 := by
  -- Unfold the definition of blue_pill_cost
  unfold blue_pill_cost
  -- Simplify the arithmetic
  simp [Nat.cast_mul, Nat.cast_ofNat]
  -- The proof is completed by reflexivity
  rfl

#eval blue_pill_cost 3 1 1 2 819

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_blue_pill_cost_l1021_102192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1021_102105

/-- Represents the time (in seconds) it takes for two trains to cross each other -/
noncomputable def crossingTime (trainLength : ℝ) (crossTime1 : ℝ) (crossTime2 : ℝ) : ℝ :=
  (2 * trainLength) / (trainLength / crossTime1 + trainLength / crossTime2)

/-- Theorem stating that under the given conditions, the crossing time is 12 seconds -/
theorem trains_crossing_time :
  let trainLength : ℝ := 120
  let crossTime1 : ℝ := 10
  let crossTime2 : ℝ := 15
  crossingTime trainLength crossTime1 crossTime2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1021_102105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hikers_total_distance_l1021_102119

/-- A hiker's journey over three days -/
noncomputable def hikers_journey (day1_distance day1_speed day2_speed_increase : ℝ) : ℝ :=
  let day1_hours := day1_distance / day1_speed
  let day2_hours := day1_hours - 1
  let day2_speed := day1_speed + day2_speed_increase
  let day3_hours := day1_hours
  let day3_speed := day2_speed
  let day2_distance := day2_hours * day2_speed
  let day3_distance := day3_hours * day3_speed
  day1_distance + day2_distance + day3_distance

/-- Theorem stating the total distance walked by the hiker -/
theorem hikers_total_distance :
  hikers_journey 18 3 1 = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hikers_total_distance_l1021_102119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_length_range_l1021_102118

/-- The circle C in the Cartesian coordinate system -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2

/-- A point on the x-axis -/
def point_on_x_axis (x : ℝ) : Prop := true

/-- The length of a line segment between two points -/
noncomputable def segment_length (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Tangent line from a point to the circle -/
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  circle_C x y ∧ ((x - x0) * 2 * x + (y - y0) * 2 * (y - 3) = 0)

theorem tangent_segment_length_range :
  ∀ (xa : ℝ), point_on_x_axis xa →
  ∀ (xp yp xq yq : ℝ),
    is_tangent xa 0 xp yp →
    is_tangent xa 0 xq yq →
    let pq_length := segment_length xp yp xq yq
    2 * Real.sqrt 14 / 3 ≤ pq_length ∧ pq_length < 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_length_range_l1021_102118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1021_102186

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ :=
  (Finset.filter (· ∣ n.val) (Finset.range n.val.succ)).card

/-- A positive integer has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ+) : Prop :=
  num_factors n = 8

theorem smallest_with_eight_factors :
  ∃ (n : ℕ+), has_eight_factors n ∧ ∀ (m : ℕ+), has_eight_factors m → n ≤ m :=
by
  use 24
  constructor
  · sorry -- Proof that 24 has exactly eight factors
  · sorry -- Proof that 24 is the smallest such number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1021_102186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1021_102153

/-- Proves that given a group of 15 men, where 3 become absent, and the remaining group
    completes the work in 10 days, the original plan was to complete the work in 8 days. -/
theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
    (h1 : total_men = 15)
    (h2 : absent_men = 3)
    (h3 : actual_days = 10) :
    (total_men : ℚ) * ((total_men - absent_men : ℕ) : ℚ)⁻¹ * (actual_days : ℚ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1021_102153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1021_102154

/-- The angle between two vectors in radians -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 1)
  ∀ b : ℝ × ℝ,
    (b.1^2 + b.2^2 = 6) →
    (a.1 * b.1 + a.2 * b.2 = -3) →
    angle_between a b = 5 * Real.pi / 6 :=
by
  intro b h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1021_102154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_angle_through_point_l1021_102184

theorem sin_plus_cos_for_angle_through_point :
  ∀ (α : ℝ),
  (∃ (r : ℝ), r > 0 ∧ 2 = r * (Real.cos α) ∧ -1 = r * (Real.sin α)) →
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_angle_through_point_l1021_102184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arithmetic_seq_with_two_same_digit_sum_l1021_102131

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- An arithmetic sequence -/
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + n * d

theorem exists_arithmetic_seq_with_two_same_digit_sum :
  ∃ (a d : ℕ), 
    (∀ n, arithmetic_seq a d n ∈ Set.univ) ∧
    (∃ (i j : ℕ), i ≠ j ∧ 
      digit_sum (arithmetic_seq a d i) = digit_sum (arithmetic_seq a d j)) ∧
    (∀ (k l m : ℕ), k ≠ l ∧ l ≠ m ∧ m ≠ k →
      digit_sum (arithmetic_seq a d k) ≠ digit_sum (arithmetic_seq a d l) ∨
      digit_sum (arithmetic_seq a d l) ≠ digit_sum (arithmetic_seq a d m) ∨
      digit_sum (arithmetic_seq a d m) ≠ digit_sum (arithmetic_seq a d k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arithmetic_seq_with_two_same_digit_sum_l1021_102131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1021_102194

-- Define the constants
noncomputable def a : ℝ := (1/9) ^ (1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log 9
noncomputable def c : ℝ := 3 ^ (1/9)

-- State the theorem
theorem relationship_abc : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1021_102194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1021_102159

/-- The ellipse C: x^2/4 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The upper vertex of the ellipse -/
def M : ℝ × ℝ := (0, 1)

/-- The lower vertex of the ellipse -/
def N : ℝ × ℝ := (0, -1)

/-- A point on the line y = 2 -/
def T (t : ℝ) : ℝ × ℝ := (t, 2)

/-- The line TM -/
def line_TM (t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 / t + 2}

/-- The line TN -/
def line_TN (t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 / t - 2}

/-- Point E: intersection of TM and the ellipse -/
noncomputable def E (t : ℝ) : ℝ × ℝ := (-8*t / (t^2 + 4), (t^2 - 4) / (t^2 + 4))

/-- Point F: intersection of TN and the ellipse -/
noncomputable def F (t : ℝ) : ℝ × ℝ := (24*t / (t^2 + 36), (-t^2 + 36) / (t^2 + 36))

/-- The ratio k of the areas of triangles TMN and TEF -/
noncomputable def k (t : ℝ) : ℝ := (t^4 + 40*t^2 + 144) / (t^4 + 24*t^2 + 144)

theorem max_k_value (t : ℝ) (h : t ≠ 0) :
  ∃ (k_max : ℝ), k_max = 4/3 ∧ ∀ t', k t' ≤ k_max ∧ (k t' = k_max ↔ t' = 2*Real.sqrt 3 ∨ t' = -2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1021_102159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_properties_l1021_102151

-- Define the eccentricity equation
def eccentricity_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1

-- Define the parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

theorem conic_sections_properties :
  (∃ e₁ e₂ : ℝ, eccentricity_equation e₁ ∧ eccentricity_equation e₂ ∧ 
    0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1) ∧ 
  (∃ f : ℝ, (∀ x y : ℝ, hyperbola x y → x^2 - f^2 = 25) ∧ 
            (∀ x y : ℝ, ellipse x y → x^2 + f^2 = 35)) ∧
  (∀ p : ℝ, p > 0 → 
    ∃ A B : ℝ × ℝ, 
      let F := (p/2, 0);
      let D := (-p/2, 0);
      let M := ((A.1 + B.1)/2, (A.2 + B.2)/2);
      parabola A.1 A.2 p ∧ parabola B.1 B.2 p ∧
      (A.1 - F.1)^2 + (A.2 - F.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
      (B.1 - F.1)^2 + (B.2 - F.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2 ∧
      (M.1 - F.1)^2 + (M.2 - F.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_properties_l1021_102151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l1021_102145

-- Define the container
noncomputable def container_side : ℝ := 12

-- Define the ice cube
noncomputable def ice_cube_side : ℝ := 3

-- Define the number of ice cubes
def num_ice_cubes : ℕ := 20

-- Define the fraction of water in the container
noncomputable def water_fraction : ℝ := 1/3

-- Theorem statement
theorem unoccupied_volume :
  let container_volume := container_side ^ 3
  let water_volume := water_fraction * container_volume
  let ice_cube_volume := ice_cube_side ^ 3
  let total_ice_volume := (num_ice_cubes : ℝ) * ice_cube_volume
  let occupied_volume := water_volume + total_ice_volume
  container_volume - occupied_volume = 612 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l1021_102145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSeries_eq_four_thirds_l1021_102111

/-- The double infinite series defined by the given formula -/
noncomputable def doubleSeries : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), (2 : ℝ) ^ (-(4 * k + j + (k + j)^2 : ℤ))

/-- Theorem stating that the double infinite series equals 4/3 -/
theorem doubleSeries_eq_four_thirds : doubleSeries = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSeries_eq_four_thirds_l1021_102111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1021_102156

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the left focus
def left_focus (F1 : ℝ × ℝ) : Prop := F1 = (0, -2)

-- Main theorem
theorem ellipse_line_intersection 
  (A B : ℝ × ℝ) (F1 : ℝ × ℝ) 
  (h_intersect : intersection_points A B)
  (h_focus : left_focus F1) :
  let AB_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let triangle_area := (1/2) * AB_length * (2 * Real.sqrt 2)
  AB_length = (8/3) * Real.sqrt 2 ∧ triangle_area = 16/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1021_102156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MB_value_l1021_102102

/-- Represents a right-angled triangle ABC with C as the right angle -/
structure RightTriangle where
  B : ℝ  -- angle B
  h : ℝ  -- half of CB
  d : ℝ  -- half of CA

/-- The length of MB in the triangle -/
noncomputable def MB (t : RightTriangle) : ℝ := 2 * t.h * t.d / (2 * t.h + t.d)

/-- The theorem statement -/
theorem MB_value (t : RightTriangle) 
  (hB : t.B = 45)  -- angle B is 45 degrees
  (hSum : MB t + Real.sqrt ((MB t + 2 * t.h)^2 + (2 * t.d)^2) = 2 * t.h + 2 * t.d) 
  : MB t = 2 * t.h * t.d / (2 * t.h + t.d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_MB_value_l1021_102102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_range_theorem_l1021_102114

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Define the first expression
noncomputable def expr1 (x y : ℝ) : ℝ := (2*x + y - 1) / x

-- Define the second expression
noncomputable def expr2 (x y : ℝ) : ℝ := |x + y + 1|

-- State the theorem
theorem circle_range_theorem (x y : ℝ) (h : circle_equation x y) :
  (expr1 x y ∈ Set.Icc 2 (10/3)) ∧ (expr2 x y ∈ Set.Icc (5 - Real.sqrt 2) (5 + Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_range_theorem_l1021_102114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_positive_l1021_102138

-- Define the expression
def f (x : ℝ) : ℝ := (x + 1) * (x + 3) * (x - 2)

-- Define the set where the expression is positive
def S : Set ℝ := {x | x ∈ Set.Ioo (-3) (-1) ∪ Set.Ioi 2}

-- State the theorem
theorem expression_positive : ∀ x : ℝ, f x > 0 ↔ x ∈ S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_positive_l1021_102138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1021_102137

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x + 1)}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1 : ℝ) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1021_102137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_5_l1021_102126

open Real

-- Define the curve C
noncomputable def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*(cos θ) - 2 = 0

-- Define line l₁
noncomputable def line_l₁ (ρ θ : ℝ) : Prop :=
  2*ρ*(sin (θ + π/3)) + 3*(sqrt 3) = 0

-- Define line l₂
def line_l₂ (θ : ℝ) : Prop :=
  θ = π/3

-- Define point P
noncomputable def point_P : ℝ × ℝ :=
  (2, π/3)

-- Define point Q
noncomputable def point_Q : ℝ × ℝ :=
  (-3, π/3)

-- State the theorem
theorem distance_PQ_is_5 :
  let (ρ₁, θ₁) := point_P
  let (ρ₂, θ₂) := point_Q
  curve_C ρ₁ θ₁ ∧ line_l₂ θ₁ ∧ line_l₁ ρ₂ θ₂ ∧ line_l₂ θ₂ →
  abs (ρ₁ - ρ₂) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_5_l1021_102126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_l1021_102100

def customSequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 2) = a (n + 1) - a n) ∧ (a 1 = 2) ∧ (a 2 = 3)

theorem sequence_2018 (a : ℕ → ℤ) (h : customSequence a) : a 2018 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_l1021_102100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_even_sine_function_l1021_102196

theorem alpha_value_for_even_sine_function (α : Real) : 
  (0 < α ∧ α < π / 2) → 
  (∀ x, Real.sin (2 * x + π / 4 + α) = Real.sin (-2 * x + π / 4 + α)) → 
  α = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_even_sine_function_l1021_102196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_markup_l1021_102165

/-- Calculate the percentage markup given the selling price and cost price of a book -/
noncomputable def percentage_markup (selling_price cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

theorem bookseller_markup :
  let selling_price : ℝ := 11.00
  let cost_price : ℝ := 9.90
  abs (percentage_markup selling_price cost_price - 11.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookseller_markup_l1021_102165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1021_102101

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of the function f -/
noncomputable def f (x B : ℝ) : ℝ := Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B)

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : ∀ x, f x t.B = f (-x) t.B)  -- f is an even function
  (h2 : t.b = f (π / 12) t.B)       -- b = f(π/12)
  (h3 : t.a = 3)                    -- a = 3
  : 
  t.b = Real.sqrt 3 ∧               -- b = √3
  (t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 ∨ 
   t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 4) -- Area S is either (3√3)/2 or (3√3)/4
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1021_102101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1021_102185

theorem trig_identity : 
  4 * (Real.sin (49 * π / 48)^3 * Real.cos (49 * π / 16) + 
       Real.cos (49 * π / 48)^3 * Real.sin (49 * π / 16)) * 
    Real.cos (49 * π / 12) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1021_102185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_two_iff_c_eq_three_fourths_l1021_102141

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 8*x + 5*x^2 - 3*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 1 - x - 3*x^2 + 4*x^4

/-- The combined polynomial h(x, c) = f(x) + c * g(x) -/
def h (x c : ℝ) : ℝ := f x + c * g x

/-- Helper function to get the coefficient of x^4 in h(x, c) -/
def coeff_x4 (c : ℝ) : ℝ := -3 + 4*c

/-- Helper function to get the coefficient of x^2 in h(x, c) -/
def coeff_x2 (c : ℝ) : ℝ := 5 - 3*c

/-- The theorem stating that h(x, c) has degree 2 if and only if c = 3/4 -/
theorem degree_two_iff_c_eq_three_fourths :
  (coeff_x4 (3/4) = 0 ∧ coeff_x2 (3/4) ≠ 0) ↔ (3/4 : ℝ) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_two_iff_c_eq_three_fourths_l1021_102141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_l1021_102121

def lcm_value : ℕ := 232848

/-- The number of ordered pairs of positive integers with a given LCM -/
def count_pairs (n : ℕ) : ℕ :=
  (Finset.filter (λ p : ℕ × ℕ ↦ Nat.lcm p.fst p.snd = n) (Finset.range n ×ˢ Finset.range n)).card

theorem lcm_pairs_count :
  count_pairs lcm_value = 945 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_pairs_count_l1021_102121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1021_102180

theorem sufficient_not_necessary (θ : ℝ) : 
  (|θ - π/12| < π/12 → Real.sin θ < 1/2) ∧ 
  ¬(Real.sin θ < 1/2 → |θ - π/12| < π/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1021_102180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1021_102125

noncomputable section

-- Define the square ABCD
def A : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (39/4, 37/4)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * (x - 5)^2

-- State the theorem
theorem parabola_equation :
  ∃ B : ℝ × ℝ,
    -- ABCD is a square
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
    -- B is on the parabola
    parabola B.1 B.2 ∧
    -- A is on the parabola
    parabola A.1 A.2 ∧
    -- Parabola is tangent to x-axis
    ∃ t : ℝ, parabola t 0 ∧ ∀ x : ℝ, x ≠ t → parabola x 0 → False :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1021_102125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_non_coplanar_triangles_l1021_102183

/-- Represents a cube -/
structure Cube where
  vertices : Fin 8
  faces : Fin 6

/-- Represents a triangle formed by three vertices of a cube -/
structure CubeTriangle where
  cube : Cube
  v1 : Fin 8
  v2 : Fin 8
  v3 : Fin 8

/-- Predicate to determine if a triangle is coplanar (all vertices on the same face) -/
def isCoplanar (t : CubeTriangle) : Prop :=
  ∃ (f : Fin 6), (t.v1.val : Nat) ∈ FaceVertices f ∧ 
                 (t.v2.val : Nat) ∈ FaceVertices f ∧ 
                 (t.v3.val : Nat) ∈ FaceVertices f
where
  FaceVertices : Fin 6 → Set Nat := sorry  -- Define this function appropriately

/-- The total number of triangles that can be formed from any three vertices of a cube -/
def totalTriangles : ℕ :=
  Nat.choose 8 3

/-- The number of coplanar triangles in a cube -/
def coplanarTriangles : ℕ := 6

/-- The number of non-coplanar triangles in a cube -/
def nonCoplanarTriangles : ℕ :=
  totalTriangles - coplanarTriangles

theorem cube_non_coplanar_triangles :
  nonCoplanarTriangles = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_non_coplanar_triangles_l1021_102183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_positive_sum_l1021_102170

/-- An arithmetic sequence with specific properties -/
structure ArithSeq where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h1 : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property
  h2 : -1 < a 7 / a 6            -- Given condition
  h3 : a 7 / a 6 < 0             -- Given condition
  h4 : d < 0                     -- Implied by S_n having a maximum value
  h5 : 0 < a 1                   -- Implied by S_n having a maximum value

/-- Sum of first n terms of an arithmetic sequence -/
def S_n (seq : ArithSeq) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proved -/
theorem max_n_positive_sum (seq : ArithSeq) :
  (∀ n > 12, S_n seq n ≤ 0) ∧ S_n seq 12 > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_positive_sum_l1021_102170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l1021_102160

/-- Parametric curve definition -/
noncomputable def x (t : ℝ) : ℝ := 3 * Real.sin t + Real.cos t

/-- Parametric curve definition -/
noncomputable def y (t : ℝ) : ℝ := 3 * Real.cos t

/-- Constants for the Cartesian equation -/
noncomputable def a : ℝ := 1/3
noncomputable def b : ℝ := -2/9
noncomputable def c : ℝ := 0

/-- Theorem stating that the parametric curve satisfies the Cartesian equation -/
theorem curve_equation (t : ℝ) : a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l1021_102160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l1021_102152

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := Set.union (Set.Iio (-1)) (Set.Ioi 3)

-- Define the parameters a and b based on the given solution set
def a : ℝ := 5
def b : ℝ := -3

-- Define the second inequality
def inequality_2 (x : ℝ) : Prop := x^2 + a*x - 2*b < 0

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := Set.Ioo (-3) (-2)

-- State the theorem
theorem solution_set_equivalence :
  (∀ x, x ∈ solution_set_1 ↔ (a*x + 1)/(x + b) > 1) →
  (∀ x, x ∈ solution_set_2 ↔ inequality_2 x) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l1021_102152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_plus_cos_l1021_102169

theorem tan_value_from_sin_plus_cos (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = 1/5)
  (h2 : θ ∈ Set.Ioo 0 Real.pi) : 
  Real.tan θ = 12/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_plus_cos_l1021_102169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1021_102181

/-- Given a triangle PQR with inradius r and circumradius R, 
    prove that its area is (20√110)/3 under certain conditions. -/
theorem triangle_area (P Q R : ℝ) (r R : ℝ) : 
  r = 4 → 
  R = 13 → 
  3 * Real.cos Q = Real.cos P + Real.cos R → 
  ∃ (x y z : ℕ), 
    (x : ℝ) * Real.sqrt y / z = 20 * Real.sqrt 110 / 3 ∧ 
    Nat.Coprime x z ∧ 
    ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1021_102181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1021_102199

open Real

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x ≥ 0) →
  (∀ x > 0, DifferentiableAt ℝ f x) →
  (∀ x > 0, f x + x * deriv f x ≤ 0) →
  0 < a →
  a < b →
  a * f b ≤ b * f a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1021_102199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_size_change_l1021_102149

/-- Proves that a 40% increase in length and 1.1199999999999999 times increase in area
    results in approximately 20% decrease in width for a rectangle -/
theorem garden_size_change (original_length original_width : ℝ) 
  (h_positive_length : original_length > 0)
  (h_positive_width : original_width > 0) :
  let new_length := 1.4 * original_length
  let new_area := 1.1199999999999999 * (original_length * original_width)
  let new_width := new_area / new_length
  ∃ ε > 0, |((original_width - new_width) / original_width) - 0.2| < ε := by
  sorry

#check garden_size_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_size_change_l1021_102149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_celsius_l1021_102112

-- Define the temperature conversion function
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * 5 / 9

-- Define the given conditions
def boiling_point_f : ℝ := 212
def ice_melting_point_f : ℝ := 32
def ice_melting_point_c : ℝ := 0
def example_temp_c : ℝ := 45
def example_temp_f : ℝ := 113

-- State the theorem
theorem water_boiling_point_celsius :
  fahrenheit_to_celsius boiling_point_f = 100 :=
by
  -- Unfold the definition of fahrenheit_to_celsius
  unfold fahrenheit_to_celsius
  -- Simplify the expression
  simp [boiling_point_f]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_celsius_l1021_102112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_head_start_l1021_102171

/-- Given two runners a and b, where a's speed is 21/19 times b's speed,
    the head start fraction that a should give b for a dead heat is 2/21 of the race length. -/
theorem race_head_start (v_a v_b L : ℝ) (h : v_a = (21 / 19) * v_b) :
  L / v_a = (L - L * (2 / 21)) / v_b := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_head_start_l1021_102171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_share_of_profit_l1021_102195

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_share_of_profit (total_profit investment_amount investment_duration total_investment_value : ℚ) : ℚ :=
  total_profit * (investment_amount * investment_duration) / total_investment_value

theorem jose_share_of_profit : 
  (let tom_investment := 3000
   let jose_investment := 4500
   let tom_duration := 12
   let jose_duration := 10
   let total_profit := 5400
   let total_investment_value := tom_investment * tom_duration + jose_investment * jose_duration
   calculate_share_of_profit total_profit jose_investment jose_duration total_investment_value) = 3000 := by
  sorry

#eval calculate_share_of_profit 5400 4500 10 (3000 * 12 + 4500 * 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_share_of_profit_l1021_102195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_percentage_approx_l1021_102130

noncomputable def deposit : ℝ := 3800
noncomputable def monthly_income : ℝ := 17272.73

noncomputable def percentage_deposited : ℝ := (deposit / monthly_income) * 100

theorem deposit_percentage_approx :
  |percentage_deposited - 21.99| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_percentage_approx_l1021_102130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l1021_102139

-- Define the parabola
def on_parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the distance from a point to the focus
noncomputable def distance_to_focus (x y : ℝ) : ℝ := Real.sqrt ((x - 0)^2 + (y - 1)^2)

theorem parabola_point_coordinates :
  ∀ x y : ℝ,
  on_parabola x y →
  distance_to_focus x y = 10 →
  (x = 6 ∨ x = -6) ∧ y = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l1021_102139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1021_102157

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x + 3)

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x ≥ 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 0} = solution_set :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1021_102157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_l1021_102164

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x - 2 * Real.sqrt 3

-- Define the focal distance
def focal_distance : ℝ := 4

-- Define the foci
def right_focus : ℝ × ℝ := (2, 0)
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the center of the ellipse
noncomputable def ellipse_center : ℝ × ℝ := (3, -Real.sqrt 3)

-- Define the line m passing through the left focus
def line_m (t : ℝ) (x y : ℝ) : Prop := x = t * y - 2

-- Define lambda as a function of t
noncomputable def lambda (t : ℝ) : ℝ := 2 * Real.sqrt 6 / (Real.sqrt (1 + t^2) + 2 / Real.sqrt (1 + t^2))

theorem max_lambda :
  ∀ t : ℝ, lambda t ≤ Real.sqrt 3 ∧
  ∃ t₀ : ℝ, lambda t₀ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_l1021_102164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l1021_102190

-- Definitions
def ExternallyTangent (c₁ c₂ : Set (ℝ × ℝ)) : Prop := sorry
def CircleWithRadius (r : ℝ) (c : Set (ℝ × ℝ)) : Prop := sorry
def InternallyTangent (c₁ c₂ : Set (ℝ × ℝ)) : Prop := sorry
def TangencyRadiiPerpendicular (c₁ c₂ c₃ : Set (ℝ × ℝ)) : Prop := sorry

theorem tangent_circles_radius (r₁ r₂ : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) :
  let R := (r₁ + r₂ + Real.sqrt ((r₁ + r₂)^2 + 4*r₁*r₂)) / 2
  ∃ (k₁ k₂ k : Set (ℝ × ℝ)),
    ExternallyTangent k₁ k₂ ∧
    CircleWithRadius r₁ k₁ ∧
    CircleWithRadius r₂ k₂ ∧
    InternallyTangent k k₁ ∧
    InternallyTangent k k₂ ∧
    TangencyRadiiPerpendicular k k₁ k₂ ∧
    CircleWithRadius R k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l1021_102190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_approx_l1021_102135

noncomputable section

-- Define the cube
def Cube (x : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {A | A = (0, 0, 0)} ∪
  {B | B = (x, 0, 0)} ∪
  {D | D = (0, x, 0)} ∪
  {F | F = (x, x, x)}

-- Define midpoints L and K
def L (x : ℝ) : ℝ × ℝ × ℝ := (0, x/2, 0)
def K (x : ℝ) : ℝ × ℝ × ℝ := (x/2, 0, 0)

-- Define the perpendicular distance function
noncomputable def perpDistance (F K L : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem cube_volume_approx (x : ℝ) :
  ∃ (cube : Set (ℝ × ℝ × ℝ)),
    cube = Cube x ∧
    perpDistance (x, x, x) (K x) (L x) = 10 →
    abs (x^3 - 323) < 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_approx_l1021_102135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_halves_l1021_102172

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x + 4 * y - 12 = 0
  line2 : ℝ × ℝ → Prop := λ (x, y) ↦ a * x + 8 * y + 11 = 0
  parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = 3 * k ∧ 8 = 4 * k

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  |11 + 12| / Real.sqrt (lines.a^2 + 8^2)

/-- Theorem stating that the distance between the given parallel lines is 7/2 -/
theorem distance_is_seven_halves (lines : ParallelLines) : 
  distance_between_parallel_lines lines = 7/2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_halves_l1021_102172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1021_102177

-- Define the function f(x) = |x| - 1
def f (x : ℝ) : ℝ := |x| - 1

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1021_102177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1021_102103

theorem polynomial_division_remainder (c : ℝ) : 
  (∃ q : Polynomial ℝ, (3 : ℝ) • X^3 + c • X^2 - (8 : ℝ) • X + 52 = (3 : ℝ) • X + 4 • q + 5) ↔ c = 32.625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1021_102103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1021_102193

/-- The general solution of the differential equation y'' - 6y' + 9y = 25e^x * sin(x) -/
noncomputable def general_solution (x : ℝ) (C₁ C₂ : ℝ) : ℝ :=
  (C₁ + C₂ * x) * (Real.exp (3 * x)) + (Real.exp x) * (4 * Real.cos x + 3 * Real.sin x)

/-- The second derivative of the general solution -/
noncomputable def general_solution_second_derivative (x : ℝ) (C₁ C₂ : ℝ) : ℝ :=
  (9 * C₁ + 9 * C₂ * x + 6 * C₂) * (Real.exp (3 * x)) + 
  (Real.exp x) * (2 * Real.cos x - 7 * Real.sin x)

theorem general_solution_satisfies_equation (x : ℝ) (C₁ C₂ : ℝ) :
  general_solution_second_derivative x C₁ C₂ - 
  6 * (deriv (general_solution · C₁ C₂)) x +
  9 * general_solution x C₁ C₂ = 
  25 * (Real.exp x) * (Real.sin x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1021_102193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circle_coverage_l1021_102107

-- Define the side length of the equilateral triangle
variable (a : ℝ) (ha : a > 0)

-- Define β
noncomputable def β : ℝ := Real.arccos (Real.sqrt 3 / 4)

-- Define the ratio of covered area to total area
noncomputable def coverageRatio : ℝ := (3 * β) / (8 * Real.pi)

-- Theorem statement
theorem equilateral_triangle_circle_coverage (a : ℝ) (ha : a > 0) :
  let R := a * Real.sqrt 3 / 3  -- Radius of circumscribed circle
  let A_circumcircle := Real.pi * R^2  -- Area of circumscribed circle
  let A_smallCircle := Real.pi * (a/2)^2  -- Area of one small circle
  (3 * A_smallCircle) / A_circumcircle = coverageRatio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circle_coverage_l1021_102107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l1021_102123

/-- A trinomial is a perfect square if it can be expressed as (ax + b)^2 for some real a and b. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value (m : ℝ) :
  IsPerfectSquareTrinomial 1 m 36 → m = 12 ∨ m = -12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l1021_102123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_expr_coeff_x2_l1021_102182

/-- The nested expression function -/
noncomputable def nested_expr (k : ℕ) (x : ℝ) : ℝ :=
  match k with
  | 0 => x
  | n + 1 => (nested_expr n x - 2)^2

/-- The coefficient of x^2 in the expanded nested expression -/
noncomputable def coeff_x2 (k : ℕ) : ℝ :=
  (4^(2*k - 1) - 4^(k - 1)) / 3

/-- Theorem stating that the coefficient of x^2 in the expanded nested expression
    is equal to the derived formula -/
theorem nested_expr_coeff_x2 (k : ℕ) (x : ℝ) :
  ∃ (a b c : ℝ), nested_expr k x = a * x^3 + (coeff_x2 k) * x^2 + b * x + c :=
by
  sorry

#check nested_expr_coeff_x2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_expr_coeff_x2_l1021_102182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sum_squares_divisible_by_sum_l1021_102176

theorem largest_n_sum_squares_divisible_by_sum : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > n → ¬((m * (m + 1) * (2 * m + 1)) / 6 = ((m * (m + 1) * (2 * m + 1)) / 6) / ((m * (m + 1)) / 2) * ((m * (m + 1)) / 2))) ∧
  ((n * (n + 1) * (2 * n + 1)) / 6 = ((n * (n + 1) * (2 * n + 1)) / 6) / ((n * (n + 1)) / 2) * ((n * (n + 1)) / 2)) ∧
  n = 1 :=
by sorry

#check largest_n_sum_squares_divisible_by_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sum_squares_divisible_by_sum_l1021_102176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_300_l1021_102144

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 390 then -x^3 / 900 + 400 * x else 90090

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ := 20000 + 100 * x

-- Define the profit function
noncomputable def Q (x : ℝ) : ℝ := R x - C x

-- Theorem statement
theorem profit_maximized_at_300 :
  ∀ x : ℝ, Q x ≤ Q 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_300_l1021_102144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1021_102113

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^2 + (Real.arcsin x)^2

theorem g_range :
  ∀ x ∈ Set.Icc (-1) 1, g x ∈ Set.Icc (Real.pi^2 / 4) (Real.pi^2 / 2) ∧
  ∃ y ∈ Set.Icc (-1) 1, g y = Real.pi^2 / 4 ∧
  ∃ z ∈ Set.Icc (-1) 1, g z = Real.pi^2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1021_102113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wilson_prime_variant_l1021_102104

theorem wilson_prime_variant (n : ℕ) (h1 : n > 4) (h2 : ¬ (Nat.factorial (n - 1)) % n = 0) : Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wilson_prime_variant_l1021_102104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_3m_l1021_102155

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (n : ℕ) : ℝ := sorry

/-- The theorem statement -/
theorem arithmetic_sequence_sum_3m
  (m : ℕ)
  (sum_m : arithmeticSum m = 30)
  (sum_2m : arithmeticSum (2 * m) = 100) :
  arithmeticSum (3 * m) = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_3m_l1021_102155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_ge_square_l1021_102134

theorem power_two_ge_square (n : ℕ) : 2^n ≥ n^2 ↔ n = 1 ∨ n = 2 ∨ n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_two_ge_square_l1021_102134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1021_102124

theorem trigonometric_values (α : ℝ) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin (α - π/3) = (3 + 4*Real.sqrt 3)/10 ∧ 
  Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1021_102124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pair_existence_l1021_102106

theorem unique_pair_existence : ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < Real.pi / 2 ∧
  Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pair_existence_l1021_102106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_fourth_quadrant_l1021_102142

theorem sin_alpha_fourth_quadrant (α : ℝ) :
  α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- α is in the fourth quadrant
  Real.cos α = 1 / 3 →                           -- cos α = 1/3
  Real.sin α = -2 * Real.sqrt 2 / 3 :=           -- sin α = -2√2/3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_fourth_quadrant_l1021_102142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_heads_probability_l1021_102168

/-- The number of coin tosses -/
def n : ℕ := 5

/-- The probability of getting heads on a single toss of a fair coin -/
noncomputable def p : ℝ := 1/2

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
noncomputable def binomial_probability (k : ℕ) : ℝ := 
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability of getting at least 2 heads in 5 tosses of a fair coin -/
theorem at_least_two_heads_probability : 
  1 - binomial_probability 0 - binomial_probability 1 = 0.8125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_heads_probability_l1021_102168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1021_102140

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x > 0}
def B : Set ℝ := {x | |x + 1| < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-3) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1021_102140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1021_102158

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 + (Real.log a / Real.log 10 + 2) * x + Real.log b / Real.log 10

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a b (-1) = -2) →
  (∀ x : ℝ, f a b x ≥ 2 * x) →
  (a = 100 ∧ b = 10) ∧
  (∀ x : ℝ, f a b x < x + 5 ↔ -4 < x ∧ x < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1021_102158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_proof_l1021_102136

/-- Calculates the average marks of a class given specific score distributions -/
def class_average (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (zero_scorers : ℕ) (rest_average : ℚ) : ℚ :=
  let rest_students := total_students - high_scorers - zero_scorers
  let total_marks := (high_scorers * high_score : ℚ) + (rest_students : ℚ) * rest_average
  total_marks / total_students

theorem exam_average_proof :
  class_average 25 5 95 3 45 = 49.6 := by
  -- Unfold the definition of class_average
  unfold class_average
  -- Simplify the arithmetic expressions
  simp [Nat.cast_sub, Nat.cast_mul]
  -- The proof is completed by normalization of rational numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_proof_l1021_102136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_proof_l1021_102109

/-- Converts degrees, minutes, and seconds to a real number representing degrees -/
noncomputable def to_degrees (deg : ℕ) (min : ℕ) (sec : ℕ) : ℝ :=
  (deg : ℝ) + (min : ℝ) / 60 + (sec : ℝ) / 3600

/-- Represents an angle that is 18°24'36" less than half of its complementary angle -/
def angle_condition (x : ℝ) : Prop :=
  x = (180 - x) / 2 - to_degrees 18 24 36

theorem angle_proof : ∃ x : ℝ, angle_condition x ∧ x = to_degrees 47 43 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_proof_l1021_102109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_linear_equation_l1021_102189

/-- A linear equation in one variable has the form ax + b = 0, where a ≠ 0 and b is a constant. -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation $\frac{|2x|}{3} + \frac{5}{x} = 3$ -/
noncomputable def f (x : ℝ) : ℝ := |2 * x| / 3 + 5 / x - 3

/-- Theorem stating that f is not a linear equation -/
theorem not_linear_equation : ¬ is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_linear_equation_l1021_102189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1021_102143

-- Define the points and locus
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2/3 = 1 ∧ p.1 ≥ 1}

-- Define the condition for point P
def P_condition (p : ℝ × ℝ) : Prop :=
  |((p.1 - F₁.1)^2 + (p.2 - F₁.2)^2).sqrt - ((p.1 - F₂.1)^2 + (p.2 - F₂.2)^2).sqrt| = 2

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x - F₂.1)

-- Define the theorem
theorem hyperbola_theorem :
  (∀ p : ℝ × ℝ, P_condition p ↔ p ∈ Γ) ∧
  (∀ k : ℝ, (k < -(3: ℝ).sqrt ∨ k > (3: ℝ).sqrt) ↔ 
    ∃ A B : ℝ × ℝ, A ∈ Γ ∧ B ∈ Γ ∧ A ≠ B ∧ 
    A.2 = line_l k A.1 ∧ B.2 = line_l k B.1) ∧
  (∃ M : ℝ × ℝ, M = (-1, 0) ∧
    ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ Γ ∧ B ∈ Γ ∧ A ≠ B ∧
    A.2 = line_l k A.1 ∧ B.2 = line_l k B.1 →
    ((M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1021_102143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1021_102127

theorem trigonometric_identities (a : Real) 
  (h1 : a ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin a = Real.sqrt 5 / 5) : 
  Real.tan (π / 4 + 2 * a) = -1 / 7 ∧ 
  Real.cos (5 * π / 6 - 2 * a) = -(3 * Real.sqrt 3 + 4) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1021_102127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_lines_theorem_l1021_102188

noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

structure Point where
  x : ℝ
  y : ℝ

noncomputable def line_through_points (p1 p2 : Point) (x : ℝ) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x) * (x - p1.x) + p1.y

def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem ellipse_intersecting_lines_theorem 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse 1 (Real.sqrt 6 / 2) a b) 
  (h4 : eccentricity a b = Real.sqrt 2 / 2) 
  (P : Point) 
  (hP : P = ⟨2, 0⟩) 
  (A B : Point) 
  (hA : ellipse A.x A.y a b) 
  (hB : ellipse B.x B.y a b) 
  (hPerp : perpendicular P A B) :
  ∃ (x : ℝ), x = 2/3 ∧ line_through_points A B x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_lines_theorem_l1021_102188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irina_deposit_return_l1021_102163

/-- Represents the bank deposit problem --/
structure BankDeposit where
  initial_deposit : ℚ
  annual_interest_rate : ℚ
  deposit_duration_months : ℕ
  exchange_rate : ℚ
  insurance_limit : ℚ

/-- Calculates the amount to be returned given a bank deposit --/
noncomputable def amount_to_return (deposit : BankDeposit) : ℚ :=
  let amount_with_interest := deposit.initial_deposit * (1 + deposit.annual_interest_rate * (deposit.deposit_duration_months / 12))
  let amount_in_rubles := amount_with_interest * deposit.exchange_rate
  min amount_in_rubles deposit.insurance_limit

/-- Theorem stating the amount to be returned for Irina Mikhaylovna's deposit --/
theorem irina_deposit_return :
  let deposit := BankDeposit.mk 23904 (1/20) 3 (5815/100) 1400000
  amount_to_return deposit = 1400000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irina_deposit_return_l1021_102163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l1021_102110

/-- The function f for which we want to find the minimum positive period -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

/-- The minimum positive period of the function f -/
noncomputable def min_positive_period : ℝ := Real.pi

/-- Theorem stating that min_positive_period is the minimum positive period of f -/
theorem min_period_of_f : 
  (∀ x : ℝ, f (x + min_positive_period) = f x) ∧ 
  (∀ p : ℝ, 0 < p → p < min_positive_period → ∃ y : ℝ, f (y + p) ≠ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l1021_102110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1021_102178

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -(x^2 + 1) / x else 2^(x + 1)

-- Define function g
def g (x : ℝ) : ℝ := x^2 - x - 2

-- Theorem statement
theorem range_of_b (b : ℝ) :
  (∃ a : ℝ, g b + f a = 2) → b ∈ Set.Icc (-1) 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1021_102178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1021_102117

/-- Sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Conditions for the geometric sequence -/
theorem geometric_sequence_sum :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1021_102117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_company_break_even_l1021_102147

/-- Represents the cost function for a printing company -/
structure PrintingCompany where
  cost_per_copy : ℚ
  plate_making_fee : ℚ

/-- Calculates the total cost for a given number of copies -/
def total_cost (company : PrintingCompany) (copies : ℚ) : ℚ :=
  company.cost_per_copy * copies + company.plate_making_fee

/-- The break-even point between two printing companies -/
def break_even_point (company_a company_b : PrintingCompany) : ℚ :=
  (company_b.plate_making_fee - company_a.plate_making_fee) /
  (company_a.cost_per_copy - company_b.cost_per_copy)

theorem printing_company_break_even :
  let company_a : PrintingCompany := ⟨2/10, 500⟩
  let company_b : PrintingCompany := ⟨4/10, 0⟩
  break_even_point company_a company_b = 2500 :=
by
  -- Unfold definitions
  unfold break_even_point
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_company_break_even_l1021_102147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l1021_102198

/-- The number of apples Greg and Sarah have to split -/
def total_greg_sarah : ℕ := 18

/-- The number of apples Susan has relative to Greg -/
def susan_multiplier : ℕ := 2

/-- The number of apples Mark has fewer than Susan -/
def mark_difference : ℕ := 5

/-- The number of apples mom needs for the pie -/
def mom_pie_apples : ℕ := 40

/-- The number of apples left over for mom -/
def mom_leftover_apples : ℕ := 9

theorem apple_distribution (greg_sarah_apples susan_apples mark_apples : ℕ) :
  greg_sarah_apples = total_greg_sarah / 2 →
  susan_apples = susan_multiplier * greg_sarah_apples →
  mark_apples = susan_apples - mark_difference →
  2 * greg_sarah_apples + susan_apples + mark_apples - mom_pie_apples = mom_leftover_apples :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l1021_102198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1021_102150

/-- The set M defined as {y | y = x² - 1, x ∈ ℝ} -/
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

/-- The set N defined as {x | y = √(3 - x²), x ∈ ℝ} -/
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

/-- The intersection of sets M and N is equal to the closed interval [-1, √3] -/
theorem intersection_M_N : M ∩ N = Set.Icc (-1) (Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1021_102150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_balance_l1021_102166

/-- Represents a weight placement on a scale -/
inductive Side
| Left
| Right
| Any

/-- Represents the state of a scale -/
structure ScaleState where
  left : ℕ
  right : ℕ

/-- Places a weight on the scale -/
def placeWeight (state : ScaleState) (weight : ℕ) : ScaleState :=
  if state.left < state.right then
    { left := state.left + weight, right := state.right }
  else if state.left > state.right then
    { left := state.left, right := state.right + weight }
  else
    { left := state.left + weight / 2, right := state.right + (weight + 1) / 2 }

/-- Theorem: Placing weights on a scale results in balance -/
theorem scale_balance (n : ℕ) (weights : List ℕ) :
  weights.length = n + 1 →
  weights.sum = 2 * n →
  (∀ w ∈ weights, w > 0) →
  List.Sorted (· ≥ ·) weights →
  let finalState := weights.foldl placeWeight { left := 0, right := 0 }
  finalState.left = finalState.right := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_balance_l1021_102166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_triangle_sides_l1021_102175

/-- A rectangular prism with an equilateral triangle base -/
structure TriangularPrism where
  /-- The side length of the equilateral triangle base -/
  base_side : ℝ
  /-- The height of the prism -/
  height : ℝ

/-- An intersecting plane that forms an isosceles right triangle -/
structure IntersectingPlane where
  /-- The prism that the plane intersects -/
  prism : TriangularPrism
  /-- The shorter side length of the intersecting triangle -/
  short_side : ℝ
  /-- The longer side length of the intersecting triangle -/
  long_side : ℝ

/-- The theorem stating the side lengths of the intersecting triangle -/
theorem intersecting_triangle_sides
  (p : TriangularPrism)
  (i : IntersectingPlane)
  (h1 : p.base_side = 1)
  (h2 : i.prism = p)
  (h3 : i.short_side = i.long_side / Real.sqrt 2)
  : i.short_side = Real.sqrt (3/2) ∧ i.long_side = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_triangle_sides_l1021_102175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1021_102146

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

theorem decreasing_interval_of_f (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + π) = f ω x) :
  ∀ x y, π/3 < x ∧ x < y ∧ y < 5*π/6 → f ω x > f ω y := by
  sorry

#check decreasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1021_102146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_edges_l1021_102148

/-- Represents a cube with colored edges -/
structure ColoredCube where
  edges : Fin 12 → Bool  -- True for black, False for red

/-- Maps a face and its edge to the corresponding global edge index -/
def face_edge : Fin 6 → Fin 4 → Fin 12 := sorry

/-- Checks if two edges are adjacent on a face -/
def are_adjacent : Fin 4 → Fin 4 → Bool := sorry

/-- Checks if a coloring is valid according to the given conditions -/
def is_valid_coloring (cube : ColoredCube) : Prop :=
  (∀ face : Fin 6, ∃ edge : Fin 4, cube.edges (face_edge face edge)) ∧
  (∀ face : Fin 6, ∀ edge1 edge2 : Fin 4, 
    are_adjacent edge1 edge2 → 
    ¬(cube.edges (face_edge face edge1) ∧ cube.edges (face_edge face edge2)))

/-- Counts the number of black edges in a cube -/
def count_black_edges (cube : ColoredCube) : Nat :=
  (List.filter id (List.ofFn cube.edges)).length

/-- The main theorem stating the minimum number of black edges required -/
theorem min_black_edges : 
  (∃ cube : ColoredCube, is_valid_coloring cube ∧ count_black_edges cube = 4) ∧
  (∀ cube : ColoredCube, is_valid_coloring cube → count_black_edges cube ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_edges_l1021_102148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_hyperbola_l1021_102167

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola

/-- Determine the type of conic section from its general equation -/
noncomputable def determineConicType (a b c d e f : ℝ) : ConicType :=
  if a = b ∧ c = 0 then ConicType.Circle
  else if a = 0 ∨ b = 0 then ConicType.Parabola
  else if a * b > 0 then ConicType.Ellipse
  else ConicType.Hyperbola

/-- The equation (x-3)^2 = 3(y+4)^2 + 27 represents a hyperbola -/
theorem equation_represents_hyperbola :
  determineConicType 1 (-3) 0 (-6) (-24) (-66) = ConicType.Hyperbola := by
  unfold determineConicType
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_hyperbola_l1021_102167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_zero_equals_two_l1021_102133

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2 - x else x^2 - x

theorem f_of_f_zero_equals_two : f (f 0) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_zero_equals_two_l1021_102133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1021_102197

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 3*x + y + 1 = 0
def line3 (x y : ℝ) : Prop := 2*x + 3*y + 5 = 0
def result_line1 (x y : ℝ) : Prop := 3*x - 2*y - 11 = 0
def result_line2 (x y : ℝ) : Prop := 2*x - y = 0
def result_line3 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1*a2 + b1*b2 = 0

-- Define equal intercepts
def equal_intercepts (a b : ℝ) : Prop := a = b

theorem problem_solution :
  -- Part 1
  (∃ x y : ℝ, intersection_point x y ∧ result_line1 x y) ∧
  perpendicular 3 (-2) 2 3 ∧
  -- Part 2
  result_line2 1 2 ∧ result_line3 1 2 ∧
  equal_intercepts 1 1 ∧ equal_intercepts 3 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1021_102197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_with_chord_l1021_102129

/-- Two points in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A chord of a circle -/
structure Chord where
  start : Point
  finish : Point

/-- The length of a chord -/
noncomputable def chordLength (c : Chord) : ℝ :=
  Real.sqrt ((c.finish.x - c.start.x)^2 + (c.finish.y - c.start.y)^2)

/-- Predicate to check if a point lies on a circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Predicate to check if a chord belongs to a circle -/
def chordBelongsToCircle (chord : Chord) (circle : Circle) : Prop :=
  pointOnCircle chord.start circle ∧ pointOnCircle chord.finish circle

/-- Main theorem -/
theorem circle_through_points_with_chord 
  (A B : Point) (O : Circle) (l : ℝ) : 
  ∃ (C : Circle), 
    pointOnCircle A C ∧ 
    pointOnCircle B C ∧ 
    ∃ (chord : Chord), 
      chordBelongsToCircle chord O ∧ 
      chordLength chord = l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_with_chord_l1021_102129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_distances_l1021_102122

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the distance from a point to a line segment
noncomputable def distancePointToSegment (P : Point) (A B : Point) : ℝ := sorry

-- Define the sum of distances from a point to the sides of a triangle
noncomputable def sumDistancesToSides (P : Point) (T : Triangle) : ℝ :=
  distancePointToSegment P T.A T.B +
  distancePointToSegment P T.B T.C +
  distancePointToSegment P T.C T.A

-- Define predicates for circumcircle, incircle, and point inside triangle
def isCircumcircle (c : Circle) (t : Triangle) : Prop := sorry
def isIncircle (c : Circle) (t : Triangle) : Prop := sorry
def isInside (p : Point) (t : Triangle) : Prop := sorry

-- Theorem statement
theorem equal_sum_distances
  (T1 T2 : Triangle)
  (circumcircle incircle : Circle)
  (P : Point)
  (h1 : isCircumcircle circumcircle T1)
  (h2 : isCircumcircle circumcircle T2)
  (h3 : isIncircle incircle T1)
  (h4 : isIncircle incircle T2)
  (h5 : isInside P T1)
  (h6 : isInside P T2) :
  sumDistancesToSides P T1 = sumDistancesToSides P T2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_distances_l1021_102122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OBC_l1021_102173

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (m n x y : ℝ) : Prop := x = m * y + n

-- Define the area of ΔOBC
noncomputable def area_OBC (x₁ y₁ x₂ y₂ : ℝ) : ℝ := abs (x₁ * y₂ - x₂ * y₁) / 2

-- Theorem statement
theorem max_area_triangle_OBC :
  ∀ (x₁ y₁ x₂ y₂ m n : ℝ),
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
  line_l m n x₁ y₁ ∧ line_l m n x₂ y₂ ∧
  x₁ > 0 ∧ y₁ > 0 ∧  -- B in first quadrant
  3 * y₁ + y₂ = 0 →
  (∀ x y, ellipse x y ∧ line_l m n x y → area_OBC x₁ y₁ x₂ y₂ ≤ Real.sqrt 3) ∧
  (area_OBC x₁ y₁ x₂ y₂ = Real.sqrt 3 ↔ 2 * Real.sqrt 3 * x₁ - 2 * y₁ - Real.sqrt 30 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_OBC_l1021_102173
