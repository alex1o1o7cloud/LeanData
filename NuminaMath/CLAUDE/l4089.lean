import Mathlib

namespace NUMINAMATH_CALUDE_lcm_plus_hundred_l4089_408904

theorem lcm_plus_hundred (a b : ℕ) (h1 : a = 1056) (h2 : b = 792) :
  Nat.lcm a b + 100 = 3268 := by sorry

end NUMINAMATH_CALUDE_lcm_plus_hundred_l4089_408904


namespace NUMINAMATH_CALUDE_max_difference_PA_PB_l4089_408909

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point A on the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Given point B -/
def B : ℝ × ℝ := (1, 1)

/-- Distance squared between two points -/
def dist_squared (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2

theorem max_difference_PA_PB :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
  dist_squared P A - dist_squared P B ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_difference_PA_PB_l4089_408909


namespace NUMINAMATH_CALUDE_tonys_rope_length_l4089_408964

/-- Represents a rope with its length and knot loss -/
structure Rope where
  length : Float
  knotLoss : Float

/-- Calculates the total length of ropes after tying them together -/
def totalRopeLength (ropes : List Rope) : Float :=
  let totalOriginalLength := ropes.map (·.length) |>.sum
  let totalLossFromKnots := ropes.map (·.knotLoss) |>.sum
  totalOriginalLength - totalLossFromKnots

/-- Theorem stating the total length of Tony's ropes after tying -/
theorem tonys_rope_length :
  let ropes : List Rope := [
    { length := 8, knotLoss := 1.2 },
    { length := 20, knotLoss := 1.5 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 2, knotLoss := 1 },
    { length := 7, knotLoss := 0.8 },
    { length := 5, knotLoss := 1.2 },
    { length := 5, knotLoss := 1.2 }
  ]
  totalRopeLength ropes = 42.1 := by
  sorry

end NUMINAMATH_CALUDE_tonys_rope_length_l4089_408964


namespace NUMINAMATH_CALUDE_sum_of_hundredth_powers_divisibility_l4089_408951

/-- QuadraticTrinomial represents a quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  p : ℤ
  q : ℤ

/-- Condition for the discriminant to be positive -/
def has_positive_discriminant (t : QuadraticTrinomial) : Prop :=
  t.p^2 - 4*t.q > 0

/-- Condition for coefficients to be divisible by 5 -/
def coeffs_divisible_by_5 (t : QuadraticTrinomial) : Prop :=
  5 ∣ t.p ∧ 5 ∣ t.q

/-- The sum of the hundredth powers of the roots -/
noncomputable def sum_of_hundredth_powers (t : QuadraticTrinomial) : ℝ :=
  let α := (-t.p + Real.sqrt (t.p^2 - 4*t.q)) / 2
  let β := (-t.p - Real.sqrt (t.p^2 - 4*t.q)) / 2
  α^100 + β^100

/-- The main theorem -/
theorem sum_of_hundredth_powers_divisibility
  (t : QuadraticTrinomial)
  (h_pos : has_positive_discriminant t)
  (h_div : coeffs_divisible_by_5 t) :
  ∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^50 : ℝ) ∧
  ∀ (n : ℕ), n > 50 → ¬∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_sum_of_hundredth_powers_divisibility_l4089_408951


namespace NUMINAMATH_CALUDE_not_always_true_converse_l4089_408928

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem not_always_true_converse
  (h1 : contained_in b α)
  (h2 : ¬ contained_in c α) :
  ¬ (∀ (α β : Plane), plane_perpendicular α β → perpendicular b β) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_converse_l4089_408928


namespace NUMINAMATH_CALUDE_value_of_3x_plus_y_l4089_408958

theorem value_of_3x_plus_y (x y : ℝ) (h : (2*x + y)^3 + x^3 + 3*x + y = 0) : 3*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3x_plus_y_l4089_408958


namespace NUMINAMATH_CALUDE_multiply_72519_by_9999_l4089_408966

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_by_9999_l4089_408966


namespace NUMINAMATH_CALUDE_cattle_land_is_40_l4089_408968

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  crop_production : ℕ

/-- Calculates the land dedicated to rearing cattle -/
def cattle_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.crop_production)

/-- Theorem stating that the land dedicated to rearing cattle is 40 hectares -/
theorem cattle_land_is_40 (farm : FarmLand) 
    (h1 : farm.total = 150)
    (h2 : farm.house_and_machinery = 25)
    (h3 : farm.future_expansion = 15)
    (h4 : farm.crop_production = 70) : 
  cattle_land farm = 40 := by
  sorry

#eval cattle_land { total := 150, house_and_machinery := 25, future_expansion := 15, crop_production := 70 }

end NUMINAMATH_CALUDE_cattle_land_is_40_l4089_408968


namespace NUMINAMATH_CALUDE_range_of_a_full_range_of_a_l4089_408986

/-- Given sets A and B, prove the range of a when A ∩ B = A -/
theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x^2 - a < 0}
  let B := {x : ℝ | x < 2}
  A ∩ B = A → a ≤ 4 :=
by
  sorry

/-- The full range of a includes all numbers less than or equal to 4 -/
theorem full_range_of_a : 
  ∃ a : ℝ, 
    let A := {x : ℝ | x^2 - a < 0}
    let B := {x : ℝ | x < 2}
    (A ∩ B = A) ∧ (a ≤ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_full_range_of_a_l4089_408986


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l4089_408943

theorem normal_distribution_two_std_dev_below_mean 
  (μ σ : ℝ) 
  (h_mean : μ = 14.5) 
  (h_std_dev : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l4089_408943


namespace NUMINAMATH_CALUDE_min_moves_for_n_triangles_l4089_408923

/-- Represents a robot on a vertex of a polygon -/
structure Robot where
  vertex : ℕ
  target : ℕ

/-- Represents the state of the polygon -/
structure PolygonState where
  n : ℕ
  robots : List Robot

/-- A move rotates a robot to point at a new target -/
def move (state : PolygonState) (robot_index : ℕ) : PolygonState :=
  sorry

/-- Checks if three robots form a triangle -/
def is_triangle (r1 r2 r3 : Robot) : Bool :=
  sorry

/-- Counts the number of triangles in the current state -/
def count_triangles (state : PolygonState) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_for_n_triangles (n : ℕ) :
  ∃ (initial_state : PolygonState),
    initial_state.n = n ∧
    initial_state.robots.length = 3 * n ∧
    ∀ (final_state : PolygonState),
      (count_triangles final_state = n) →
      (∃ (move_sequence : List ℕ),
        final_state = (move_sequence.foldl move initial_state) ∧
        move_sequence.length ≥ (9 * n^2 - 7 * n) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_n_triangles_l4089_408923


namespace NUMINAMATH_CALUDE_boys_girls_ratio_l4089_408990

/-- Given a classroom with boys and girls, prove that the initial ratio of boys to girls is 1:1 --/
theorem boys_girls_ratio (total : ℕ) (left : ℕ) (B G : ℕ) : 
  total = 32 →  -- Total number of boys and girls initially
  left = 8 →  -- Number of girls who left
  B = 2 * (G - left) →  -- After girls left, there are twice as many boys as girls
  B + G = total →  -- Total is the sum of boys and girls
  B = G  -- The number of boys equals the number of girls, implying a 1:1 ratio
  := by sorry

end NUMINAMATH_CALUDE_boys_girls_ratio_l4089_408990


namespace NUMINAMATH_CALUDE_triangle_property_l4089_408917

open Real

theorem triangle_property (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  cos (2 * A) - 3 * cos (B + C) - 1 = 0 ∧
  R = 1 →
  A = π / 3 ∧ 
  (∃ (S : ℝ), S ≤ 3 * sqrt 3 / 4 ∧ 
    ∀ (S' : ℝ), (∃ (a b c : ℝ), 
      a = 2 * R * sin A ∧
      b = 2 * R * sin B ∧
      c = 2 * R * sin C ∧
      S' = 1 / 2 * a * b * sin C) → 
    S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l4089_408917


namespace NUMINAMATH_CALUDE_negate_negative_twenty_l4089_408993

theorem negate_negative_twenty : -(-20) = 20 := by
  sorry

end NUMINAMATH_CALUDE_negate_negative_twenty_l4089_408993


namespace NUMINAMATH_CALUDE_student_average_grade_l4089_408973

theorem student_average_grade
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 70)
  (h4 : avg_grade_two_years = 86)
  : ∃ x : ℚ, x = 596 / 6 ∧ 
    (courses_year_before * avg_grade_year_before + courses_last_year * x) / 
    (courses_year_before + courses_last_year) = avg_grade_two_years :=
by sorry

end NUMINAMATH_CALUDE_student_average_grade_l4089_408973


namespace NUMINAMATH_CALUDE_complex_reciprocal_l4089_408939

theorem complex_reciprocal (i : ℂ) : i * i = -1 → (1 : ℂ) / (1 - i) = (1 : ℂ) / 2 + i / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_l4089_408939


namespace NUMINAMATH_CALUDE_percentage_equivalence_l4089_408975

theorem percentage_equivalence : 
  ∃ P : ℝ, (35 / 100 * 400 : ℝ) = P / 100 * 700 ∧ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l4089_408975


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l4089_408914

/-- The ratio of upstream to downstream swimming time -/
theorem upstream_downstream_time_ratio 
  (swim_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : swim_speed = 9) 
  (h2 : stream_speed = 3) : 
  (swim_speed - stream_speed)⁻¹ / (swim_speed + stream_speed)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l4089_408914


namespace NUMINAMATH_CALUDE_smallest_number_meeting_criteria_l4089_408935

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def count_even_digits (n : ℕ) : ℕ :=
  (if n % 2 = 0 then 1 else 0) +
  (if (n / 10) % 2 = 0 then 1 else 0) +
  (if (n / 100) % 2 = 0 then 1 else 0) +
  (if (n / 1000) % 2 = 0 then 1 else 0)

def count_odd_digits (n : ℕ) : ℕ := 4 - count_even_digits n

def meets_criteria (n : ℕ) : Prop :=
  is_four_digit n ∧
  divisible_by_9 n ∧
  count_even_digits n = 3 ∧
  count_odd_digits n = 1

theorem smallest_number_meeting_criteria :
  ∀ n : ℕ, meets_criteria n → n ≥ 2043 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_meeting_criteria_l4089_408935


namespace NUMINAMATH_CALUDE_intersection_line_circle_l4089_408979

/-- Given a line y = 2x + 1 intersecting a circle x^2 + y^2 + ax + 2y + 1 = 0 at points A and B,
    and a line mx + y + 2 = 0 that bisects chord AB perpendicularly, prove that a = 4 -/
theorem intersection_line_circle (a m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = 2 * A.1 + 1 ∧ B.2 = 2 * B.1 + 1) ∧ 
    (A.1^2 + A.2^2 + a * A.1 + 2 * A.2 + 1 = 0 ∧ 
     B.1^2 + B.2^2 + a * B.1 + 2 * B.2 + 1 = 0) ∧
    (∃ C : ℝ × ℝ, C ∈ Set.Icc A B ∧ 
      m * C.1 + C.2 + 2 = 0 ∧
      (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l4089_408979


namespace NUMINAMATH_CALUDE_prob_even_product_three_dice_l4089_408948

-- Define a six-sided die
def SixSidedDie : Type := Fin 6

-- Define the probability of rolling an even number on a single die
def probEvenOnDie : ℚ := 1/2

-- Define the probability of rolling at least one even number on two dice
def probAtLeastOneEvenOnTwoDice : ℚ := 1 - (1 - probEvenOnDie) ^ 2

-- The main theorem
theorem prob_even_product_three_dice :
  probAtLeastOneEvenOnTwoDice = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_even_product_three_dice_l4089_408948


namespace NUMINAMATH_CALUDE_distance_between_points_l4089_408981

def point1 : ℝ × ℝ := (-2, 5)
def point2 : ℝ × ℝ := (4, -1)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4089_408981


namespace NUMINAMATH_CALUDE_speed_conversion_l4089_408988

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.556799999999999

/-- The speed in km/h we want to prove -/
def speed_kmph : ℝ := 56.00448

theorem speed_conversion : given_speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l4089_408988


namespace NUMINAMATH_CALUDE_soccer_games_played_l4089_408900

theorem soccer_games_played (win_percentage : ℝ) (games_won : ℝ) (total_games : ℝ) : 
  win_percentage = 0.40 → games_won = 63.2 → win_percentage * total_games = games_won → total_games = 158 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_played_l4089_408900


namespace NUMINAMATH_CALUDE_unique_ecuadorian_number_l4089_408925

def is_ecuadorian (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- Three-digit number
  n % 10 ≠ 0 ∧  -- Does not end in 0
  n % 36 = 0 ∧  -- Multiple of 36
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) > 0 ∧  -- abc - cba > 0
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) % 36 = 0  -- (abc - cba) is multiple of 36

theorem unique_ecuadorian_number : ∃! n : ℕ, is_ecuadorian n ∧ n = 864 := by sorry

end NUMINAMATH_CALUDE_unique_ecuadorian_number_l4089_408925


namespace NUMINAMATH_CALUDE_max_turns_max_duration_l4089_408997

/-- Represents the state of a soldier in the formation -/
inductive Direction
  | Left
  | Right

/-- Represents the formation of soldiers -/
def Formation := List Direction

/-- Counts the number of 180° turns for a given soldier -/
def countTurns (n : Nat) (f : Formation) : Nat :=
  sorry

/-- Calculates the total duration of movement in the formation -/
def movementDuration (f : Formation) : Nat :=
  sorry

/-- Theorem stating the maximum number of turns for any soldier -/
theorem max_turns (n : Nat) (f : Formation) (h : f.length = n) :
  ∀ i, i < n → countTurns n f ≤ n - 1 :=
  sorry

/-- Theorem stating the maximum duration of movement in the formation -/
theorem max_duration (n : Nat) (f : Formation) (h : f.length = n) :
  movementDuration f ≤ n - 1 :=
  sorry

end NUMINAMATH_CALUDE_max_turns_max_duration_l4089_408997


namespace NUMINAMATH_CALUDE_worker_efficiency_l4089_408921

/-- Given two workers A and B, where A is twice as efficient as B, and they complete a work together in 12 days, prove that A can complete the work alone in 18 days. -/
theorem worker_efficiency (work_rate_A work_rate_B : ℝ) (total_time : ℝ) :
  work_rate_A = 2 * work_rate_B →
  work_rate_A + work_rate_B = 1 / total_time →
  total_time = 12 →
  1 / work_rate_A = 18 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l4089_408921


namespace NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l4089_408942

variables {f g : ℝ → ℝ} {a b x : ℝ}

theorem product_decreasing_implies_inequality
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l4089_408942


namespace NUMINAMATH_CALUDE_mark_soup_cans_l4089_408991

/-- The number of cans of soup Mark bought -/
def soup_cans : ℕ := sorry

/-- The cost of one can of soup -/
def soup_cost : ℕ := 2

/-- The number of loaves of bread Mark bought -/
def bread_loaves : ℕ := 2

/-- The cost of one loaf of bread -/
def bread_cost : ℕ := 5

/-- The number of boxes of cereal Mark bought -/
def cereal_boxes : ℕ := 2

/-- The cost of one box of cereal -/
def cereal_cost : ℕ := 3

/-- The number of gallons of milk Mark bought -/
def milk_gallons : ℕ := 2

/-- The cost of one gallon of milk -/
def milk_cost : ℕ := 4

/-- The number of $10 bills Mark used to pay -/
def ten_dollar_bills : ℕ := 4

theorem mark_soup_cans : soup_cans = 8 := by sorry

end NUMINAMATH_CALUDE_mark_soup_cans_l4089_408991


namespace NUMINAMATH_CALUDE_no_such_function_exists_l4089_408971

open Set
open Function
open Real

theorem no_such_function_exists :
  ¬∃ f : {x : ℝ | x > 0} → {x : ℝ | x > 0},
    ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
      f ⟨x + y, add_pos hx hy⟩ ≥ f ⟨x, hx⟩ + y * f (f ⟨x, hx⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_no_such_function_exists_l4089_408971


namespace NUMINAMATH_CALUDE_sum_of_divisors_154_l4089_408961

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_154 : sum_of_divisors 154 = 288 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_154_l4089_408961


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l4089_408919

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum (a : ℕ → ℤ) :
  arithmetic_seq a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l4089_408919


namespace NUMINAMATH_CALUDE_total_pizza_cost_l4089_408931

def pizza_cost : ℕ := 8
def number_of_pizzas : ℕ := 3

theorem total_pizza_cost : pizza_cost * number_of_pizzas = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_cost_l4089_408931


namespace NUMINAMATH_CALUDE_total_animals_is_130_l4089_408945

/-- The total number of animals seen throughout the day -/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  morning_total + afternoon_beavers + afternoon_chipmunks

/-- Theorem stating the total number of animals seen is 130 -/
theorem total_animals_is_130 :
  total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_130_l4089_408945


namespace NUMINAMATH_CALUDE_midline_characterization_l4089_408915

/-- Triangle type -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Function to check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to check if a point is on the midline of a triangle -/
def on_midline (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem midline_characterization (t : Triangle) (M : ℝ × ℝ) :
  is_inside M t →
  (on_midline M t ↔ area ⟨M, t.A, t.B⟩ = area ⟨M, t.B, t.C⟩ + area ⟨M, t.C, t.A⟩) :=
by sorry

end NUMINAMATH_CALUDE_midline_characterization_l4089_408915


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4089_408903

theorem necessary_but_not_sufficient :
  let p := fun x : ℝ => x^2 - 2*x ≥ 3
  let q := fun x : ℝ => -1 < x ∧ x < 2
  (∀ x, q x → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ ¬(q x)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4089_408903


namespace NUMINAMATH_CALUDE_limit_expected_sides_l4089_408989

/-- The expected number of sides of a polygon after k cuts -/
def expected_sides (n : ℕ) (k : ℕ) : ℚ :=
  (n + 4 * k : ℚ) / (k + 1 : ℚ)

/-- Theorem: The limit of expected sides approaches 4 as k approaches infinity -/
theorem limit_expected_sides (n : ℕ) :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides n k - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_expected_sides_l4089_408989


namespace NUMINAMATH_CALUDE_min_value_function_l4089_408998

theorem min_value_function (p : ℝ) (h_p : p > 0) :
  (∃ (m : ℝ), m = 4 ∧ 
    ∀ (x : ℝ), x > 1 → x + p / (x - 1) ≥ m ∧ 
    ∃ (x₀ : ℝ), x₀ > 1 ∧ x₀ + p / (x₀ - 1) = m) →
  p = 9/4 := by sorry

end NUMINAMATH_CALUDE_min_value_function_l4089_408998


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_squared_l4089_408922

/-- A square inscribed in an ellipse with specific properties -/
structure InscribedSquare where
  /-- The ellipse equation: x^2 + 3y^2 = 3 -/
  ellipse : ∀ (x y : ℝ), x^2 + 3 * y^2 = 3 → True
  /-- One vertex of the square is at (0, 1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the square lies along the y-axis -/
  diagonal_on_y_axis : ∃ (v1 v2 : ℝ × ℝ), v1.1 = 0 ∧ v2.1 = 0 ∧ v1 ≠ v2

/-- The theorem stating the square of the side length of the inscribed square -/
theorem inscribed_square_side_length_squared (s : InscribedSquare) :
  ∃ (side_length : ℝ), side_length^2 = 5/3 - 2 * Real.sqrt (2/3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_squared_l4089_408922


namespace NUMINAMATH_CALUDE_relay_race_distance_l4089_408937

/-- Represents a runner in the relay race -/
structure Runner where
  name : String
  speed : ℝ
  time : ℝ

/-- Calculates the distance covered by a runner -/
def distance (runner : Runner) : ℝ := runner.speed * runner.time

theorem relay_race_distance (sadie ariana sarah : Runner)
  (h1 : sadie.speed = 3 ∧ sadie.time = 2)
  (h2 : ariana.speed = 6 ∧ ariana.time = 0.5)
  (h3 : sarah.speed = 4)
  (h4 : sadie.time + ariana.time + sarah.time = 4.5) :
  distance sadie + distance ariana + distance sarah = 17 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_l4089_408937


namespace NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l4089_408983

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l4089_408983


namespace NUMINAMATH_CALUDE_snowdrift_depth_ratio_l4089_408934

theorem snowdrift_depth_ratio (initial_depth second_day_depth third_day_snow fourth_day_snow final_depth : ℝ) :
  initial_depth = 20 →
  third_day_snow = 6 →
  fourth_day_snow = 18 →
  final_depth = 34 →
  second_day_depth + third_day_snow + fourth_day_snow = final_depth →
  second_day_depth / initial_depth = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_snowdrift_depth_ratio_l4089_408934


namespace NUMINAMATH_CALUDE_movie_shelf_distribution_l4089_408911

theorem movie_shelf_distribution (n : ℕ) : 
  (∃ k : ℕ, n + 1 = 2 * k) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_movie_shelf_distribution_l4089_408911


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_equals_twelve_l4089_408932

theorem sqrt_product_quotient_equals_twelve :
  Real.sqrt 27 * Real.sqrt (8/3) / Real.sqrt (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_equals_twelve_l4089_408932


namespace NUMINAMATH_CALUDE_base_conversion_difference_l4089_408978

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_difference : 
  let base_6_num := [0, 1, 2, 3, 4]  -- 43210 in base 6 (least significant digit first)
  let base_7_num := [0, 1, 2, 3]     -- 3210 in base 7 (least significant digit first)
  to_base_10 base_6_num 6 - to_base_10 base_7_num 7 = 4776 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_difference_l4089_408978


namespace NUMINAMATH_CALUDE_quadratic_roots_quadratic_function_l4089_408930

-- Part 1
theorem quadratic_roots (a b c : ℝ) 
  (h : Real.sqrt (a - 2) + abs (b + 1) + (c + 2)^2 = 0) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 17) / 4 ∧ 
              x2 = (1 - Real.sqrt 17) / 4 ∧
              f x1 = 0 ∧ f x2 = 0 :=
sorry

-- Part 2
theorem quadratic_function (a b c : ℝ) 
  (h1 : a * (-1)^2 + b * (-1) + c = 0)
  (h2 : a * 0^2 + b * 0 + c = -3)
  (h3 : a * 3^2 + b * 3 + c = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = x^2 - 2*x - 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_quadratic_function_l4089_408930


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4089_408959

theorem sum_of_coefficients (a b c d e : ℚ) :
  (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4089_408959


namespace NUMINAMATH_CALUDE_min_values_ab_l4089_408985

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (ab ≥ 8) ∧ (a + b ≥ 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_ab_l4089_408985


namespace NUMINAMATH_CALUDE_correct_calculation_result_l4089_408962

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l4089_408962


namespace NUMINAMATH_CALUDE_gnuff_tutoring_time_l4089_408982

/-- Calculates the number of minutes tutored given the total amount paid, flat rate, and per-minute rate. -/
def minutes_tutored (total_amount : ℕ) (flat_rate : ℕ) (per_minute_rate : ℕ) : ℕ :=
  (total_amount - flat_rate) / per_minute_rate

/-- Theorem stating that given the specific rates and total amount, the number of minutes tutored is 18. -/
theorem gnuff_tutoring_time :
  minutes_tutored 146 20 7 = 18 := by
  sorry

#eval minutes_tutored 146 20 7

end NUMINAMATH_CALUDE_gnuff_tutoring_time_l4089_408982


namespace NUMINAMATH_CALUDE_intersection_A_B_l4089_408947

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4089_408947


namespace NUMINAMATH_CALUDE_mean_median_difference_l4089_408913

theorem mean_median_difference (x : ℕ) : 
  let s := [x, x + 2, x + 4, x + 7, x + 27]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  let median := x + 4
  mean = median + 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l4089_408913


namespace NUMINAMATH_CALUDE_log_equation_solution_l4089_408941

theorem log_equation_solution (x : ℝ) :
  (x + 5 > 0) → (x - 3 > 0) → (x^2 - x - 15 > 0) →
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - x - 15) + 1) →
  (x = 2/3 + Real.sqrt 556 / 6 ∨ x = 2/3 - Real.sqrt 556 / 6) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4089_408941


namespace NUMINAMATH_CALUDE_election_winner_percentage_l4089_408908

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (winning_margin : ℕ) 
  (h1 : total_votes = 6900)
  (h2 : winning_margin = 1380) :
  (winning_margin : ℚ) / total_votes + 1/2 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l4089_408908


namespace NUMINAMATH_CALUDE_two_girls_probability_l4089_408999

def total_students : ℕ := 10
def num_boys : ℕ := 4
def num_selected : ℕ := 3

def probability_two_girls : ℚ :=
  (Nat.choose (total_students - num_boys) 2 * Nat.choose num_boys 1) /
  Nat.choose total_students num_selected

theorem two_girls_probability :
  probability_two_girls = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_two_girls_probability_l4089_408999


namespace NUMINAMATH_CALUDE_negation_exists_not_eq_forall_eq_l4089_408952

theorem negation_exists_not_eq_forall_eq :
  (¬ ∃ x : ℝ, x^2 ≠ 1) ↔ (∀ x : ℝ, x^2 = 1) := by sorry

end NUMINAMATH_CALUDE_negation_exists_not_eq_forall_eq_l4089_408952


namespace NUMINAMATH_CALUDE_larger_integer_value_l4089_408957

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 189) : 
  a = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l4089_408957


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4089_408910

/-- An isosceles triangle with two sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  7 + 7 + base = 23 →
  base = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4089_408910


namespace NUMINAMATH_CALUDE_combined_tax_rate_l4089_408938

/-- Represents the combined tax rate problem for Mork, Mindy, and Julie -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (julie_rate : ℚ) 
  (mork_income : ℚ) 
  (mindy_income : ℚ) 
  (julie_income : ℚ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  julie_rate = 35/100 →
  mindy_income = 4 * mork_income →
  julie_income = 2 * mork_income →
  julie_income = (1/2) * mindy_income →
  (mork_rate * mork_income + mindy_rate * mindy_income + julie_rate * julie_income) / 
  (mork_income + mindy_income + julie_income) = 215/700 :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l4089_408938


namespace NUMINAMATH_CALUDE_system_solution_l4089_408984

theorem system_solution (x y : ℝ) : 
  (x^3 * y + x * y^3 = 10 ∧ x^4 + y^4 = 17) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l4089_408984


namespace NUMINAMATH_CALUDE_car_price_theorem_l4089_408920

def asking_price_proof (P : ℝ) : Prop :=
  let first_offer := (9/10) * P
  let second_offer := P - 320
  (first_offer - second_offer = 200) → (P = 1200)

theorem car_price_theorem :
  ∀ P : ℝ, asking_price_proof P :=
sorry

end NUMINAMATH_CALUDE_car_price_theorem_l4089_408920


namespace NUMINAMATH_CALUDE_no_valid_coloring_l4089_408956

-- Define a color type
inductive Color
| Blue
| Red
| Green

-- Define a coloring function type
def Coloring := Nat → Color

-- Define the property that all three colors are used
def AllColorsUsed (f : Coloring) : Prop :=
  ∃ (a b c : Nat), a > 1 ∧ b > 1 ∧ c > 1 ∧ 
    f a = Color.Blue ∧ f b = Color.Red ∧ f c = Color.Green

-- Define the property that the product of two differently colored numbers
-- has a different color from both multipliers
def ValidColoring (f : Coloring) : Prop :=
  ∀ (a b : Nat), a > 1 → b > 1 → f a ≠ f b →
    f (a * b) ≠ f a ∧ f (a * b) ≠ f b

-- State the theorem
theorem no_valid_coloring :
  ¬∃ (f : Coloring), AllColorsUsed f ∧ ValidColoring f :=
sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l4089_408956


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l4089_408916

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1980) → (180 * ((n + 4) - 2) = 2700) := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l4089_408916


namespace NUMINAMATH_CALUDE_rectangle_area_l4089_408902

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle
  sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- Define the area of a rectangle
def area (rect : Rectangle) : ℝ :=
  sorry

-- Theorem statement
theorem rectangle_area (rect : Rectangle) :
  isRectangle rect →
  sideLength rect.A rect.B = 15 →
  sideLength rect.A rect.C = 17 →
  area rect = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4089_408902


namespace NUMINAMATH_CALUDE_cannot_cut_squares_l4089_408901

theorem cannot_cut_squares (paper_length paper_width : ℝ) 
  (square1_side square2_side : ℝ) (total_area : ℝ) : 
  paper_length = 10 →
  paper_width = 8 →
  square1_side / square2_side = 4 / 3 →
  square1_side^2 + square2_side^2 = total_area →
  total_area = 75 →
  square1_side + square2_side > paper_length :=
by sorry

end NUMINAMATH_CALUDE_cannot_cut_squares_l4089_408901


namespace NUMINAMATH_CALUDE_miriam_homework_time_l4089_408927

theorem miriam_homework_time (laundry_time bathroom_time room_time total_time : ℕ) 
  (h1 : laundry_time = 30)
  (h2 : bathroom_time = 15)
  (h3 : room_time = 35)
  (h4 : total_time = 120) :
  total_time - (laundry_time + bathroom_time + room_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_miriam_homework_time_l4089_408927


namespace NUMINAMATH_CALUDE_four_number_sequence_l4089_408929

def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def is_geometric_sequence (b c d : ℝ) : Prop := c * c = b * d

theorem four_number_sequence :
  ∀ (a b c d : ℝ),
    is_arithmetic_sequence a b c →
    is_geometric_sequence b c d →
    a + d = 16 →
    b + c = 12 →
    ((a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16)) :=
by sorry

end NUMINAMATH_CALUDE_four_number_sequence_l4089_408929


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l4089_408926

/-- The perimeter of an irregular pentagon with given side lengths is 52.9 cm -/
theorem pentagon_perimeter (s1 s2 s3 s4 s5 : ℝ) 
  (h1 : s1 = 5.2) (h2 : s2 = 10.3) (h3 : s3 = 15.8) (h4 : s4 = 8.7) (h5 : s5 = 12.9) :
  s1 + s2 + s3 + s4 + s5 = 52.9 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l4089_408926


namespace NUMINAMATH_CALUDE_moon_arrangements_l4089_408933

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "MOON" has 4 letters with one letter repeated twice -/
def moonWord : (ℕ × List ℕ) := (4, [2])

theorem moon_arrangements :
  distinctArrangements moonWord.fst moonWord.snd = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_l4089_408933


namespace NUMINAMATH_CALUDE_temperature_problem_l4089_408949

def temperature_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 1) = x n + x (n + 2)

theorem temperature_problem (x : ℕ → ℤ) 
  (h_seq : temperature_sequence x)
  (h_3 : x 3 = 5)
  (h_31 : x 31 = 2) :
  x 25 = -3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_problem_l4089_408949


namespace NUMINAMATH_CALUDE_inequality_proof_l4089_408954

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4089_408954


namespace NUMINAMATH_CALUDE_f_second_derivative_at_zero_l4089_408912

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x - Real.cos x

theorem f_second_derivative_at_zero :
  (deriv (deriv f)) 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_second_derivative_at_zero_l4089_408912


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l4089_408974

/-- The number of different positive, six-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, and 9 -/
def sixDigitIntegers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem count_six_digit_integers : sixDigitIntegers = 60 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l4089_408974


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l4089_408944

/-- The number of seats in a row -/
def total_seats : ℕ := 22

/-- The number of candidates to be seated -/
def num_candidates : ℕ := 4

/-- The minimum number of empty seats required between any two candidates -/
def min_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange the candidates -/
def seating_arrangements : ℕ := sorry

/-- Theorem stating that the number of seating arrangements is 840 -/
theorem seating_arrangements_count : seating_arrangements = 840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l4089_408944


namespace NUMINAMATH_CALUDE_box_length_l4089_408936

/-- Given a box with specified dimensions and cube properties, prove its length --/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  width = 16 →
  height = 6 →
  cube_volume = 3 →
  min_cubes = 384 →
  (min_cubes : ℝ) * cube_volume / (width * height) = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_box_length_l4089_408936


namespace NUMINAMATH_CALUDE_revolver_game_probability_l4089_408972

/-- Represents a six-shot revolver with one bullet -/
structure Revolver :=
  (chambers : Fin 6)
  (bullet : Fin 6)

/-- Represents the state of the game -/
inductive GameState
  | A
  | B

/-- The probability of firing the bullet on a single shot -/
def fire_probability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def not_fire_probability : ℚ := 1 - fire_probability

/-- The probability that A fires the bullet -/
noncomputable def prob_A_fires : ℚ :=
  fire_probability / (1 - not_fire_probability * not_fire_probability)

theorem revolver_game_probability :
  prob_A_fires = 6 / 11 :=
sorry

end NUMINAMATH_CALUDE_revolver_game_probability_l4089_408972


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l4089_408994

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^14 + i^19 + i^24 + i^29 + i^34 = -1 := by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_sum_of_i_powers_l4089_408994


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l4089_408965

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales 
  (total_goal : ℕ) 
  (friends : ℕ) 
  (tickets_per_friend : ℕ) 
  (second_day_sales : ℕ) 
  (h1 : total_goal = 80)
  (h2 : friends = 5)
  (h3 : tickets_per_friend = 4)
  (h4 : second_day_sales = 32) :
  total_goal - (friends * tickets_per_friend + second_day_sales) = 28 := by
sorry


end NUMINAMATH_CALUDE_amanda_ticket_sales_l4089_408965


namespace NUMINAMATH_CALUDE_tax_discount_commute_price_difference_is_zero_l4089_408907

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates the difference between applying tax then discount and applying discount then tax -/
def price_difference (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate) - price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that the price difference is always zero -/
theorem price_difference_is_zero (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price_difference price tax_rate discount_rate = 0 :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_price_difference_is_zero_l4089_408907


namespace NUMINAMATH_CALUDE_guy_has_sixty_cents_l4089_408953

/-- The amount of money each person has in cents -/
structure Money where
  lance : ℕ
  margaret : ℕ
  bill : ℕ
  guy : ℕ

/-- The total amount of money in cents -/
def total (m : Money) : ℕ := m.lance + m.margaret + m.bill + m.guy

/-- Theorem: Given the conditions, Guy has 60 cents -/
theorem guy_has_sixty_cents (m : Money) 
  (h1 : m.lance = 70)
  (h2 : m.margaret = 75)  -- Three-fourths of a dollar is 75 cents
  (h3 : m.bill = 60)      -- Six dimes is 60 cents
  (h4 : total m = 265) : 
  m.guy = 60 := by
  sorry


end NUMINAMATH_CALUDE_guy_has_sixty_cents_l4089_408953


namespace NUMINAMATH_CALUDE_brandy_excess_caffeine_l4089_408946

/-- Represents the caffeine consumption and tolerance of a person named Brandy --/
structure BrandyCaffeine where
  weight : ℝ
  baseLimit : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℝ
  energyDrinkConsumption : ℝ
  medicationEffect : ℝ

/-- Calculates the excess caffeine consumed by Brandy --/
def excessCaffeineConsumed (b : BrandyCaffeine) : ℝ :=
  let maxSafe := b.weight * b.baseLimit + b.additionalTolerance - b.medicationEffect
  let consumed := b.coffeeConsumption + b.energyDrinkConsumption
  consumed - maxSafe

/-- Theorem stating that Brandy has consumed 495 mg more caffeine than her adjusted maximum safe amount --/
theorem brandy_excess_caffeine :
  let b : BrandyCaffeine := {
    weight := 60,
    baseLimit := 2.5,
    additionalTolerance := 50,
    coffeeConsumption := 2 * 95,
    energyDrinkConsumption := 4 * 120,
    medicationEffect := 25
  }
  excessCaffeineConsumed b = 495 := by sorry

end NUMINAMATH_CALUDE_brandy_excess_caffeine_l4089_408946


namespace NUMINAMATH_CALUDE_josh_pencils_l4089_408976

theorem josh_pencils (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 31 → left = 111 → initial = given_away + left →
  initial = 142 := by sorry

end NUMINAMATH_CALUDE_josh_pencils_l4089_408976


namespace NUMINAMATH_CALUDE_expression_evaluation_l4089_408960

theorem expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 2) : 
  3 * x^2 - 4 * y + 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4089_408960


namespace NUMINAMATH_CALUDE_divisor_count_problem_l4089_408992

theorem divisor_count_problem (n : ℕ+) :
  (∃ (d : ℕ → ℕ), d (110 * n ^ 3) = 110) →
  (∃ (d : ℕ → ℕ), d (81 * n ^ 4) = 325) :=
by sorry

end NUMINAMATH_CALUDE_divisor_count_problem_l4089_408992


namespace NUMINAMATH_CALUDE_barn_paint_area_l4089_408963

/-- Calculates the total area to be painted for a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  let wall_area1 := 2 * width * height
  let wall_area2 := 2 * length * height
  let ceiling_area := width * length
  2 * (wall_area1 + wall_area2) + 2 * ceiling_area

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_paint_area :
  total_paint_area 12 15 6 = 1008 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l4089_408963


namespace NUMINAMATH_CALUDE_purchase_with_discounts_l4089_408970

/-- Calculates the final cost of a purchase with specific discounts -/
theorem purchase_with_discounts 
  (initial_total : ℝ) 
  (discounted_item_price : ℝ) 
  (item_discount_rate : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1) :
  initial_total - 
  (discounted_item_price * item_discount_rate) - 
  ((initial_total - (discounted_item_price * item_discount_rate)) * total_discount_rate) = 45 := by
  sorry

end NUMINAMATH_CALUDE_purchase_with_discounts_l4089_408970


namespace NUMINAMATH_CALUDE_cards_satisfy_conditions_l4089_408977

def card1 : Finset Nat := {1, 4, 7}
def card2 : Finset Nat := {2, 3, 4}
def card3 : Finset Nat := {2, 5, 7}

theorem cards_satisfy_conditions : 
  (card1 ∩ card2).card = 1 ∧ 
  (card1 ∩ card3).card = 1 ∧ 
  (card2 ∩ card3).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_cards_satisfy_conditions_l4089_408977


namespace NUMINAMATH_CALUDE_greatest_common_divisor_problem_l4089_408980

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, p.Prime ∧ k > 0 ∧ n = p ^ k

theorem greatest_common_divisor_problem (m : ℕ) 
  (h1 : (Nat.divisors (Nat.gcd 72 m)).card = 3)
  (h2 : is_prime_power m) :
  Nat.gcd 72 m = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_problem_l4089_408980


namespace NUMINAMATH_CALUDE_root_shrinking_method_l4089_408905

theorem root_shrinking_method (a b c p α β : ℝ) (ha : a ≠ 0) (hp : p ≠ 0) 
  (hα : a * α^2 + b * α + c = 0) (hβ : a * β^2 + b * β + c = 0) :
  (p^2 * a) * (α/p)^2 + (p * b) * (α/p) + c = 0 ∧
  (p^2 * a) * (β/p)^2 + (p * b) * (β/p) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_shrinking_method_l4089_408905


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l4089_408995

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l4089_408995


namespace NUMINAMATH_CALUDE_lcm_36_100_l4089_408996

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l4089_408996


namespace NUMINAMATH_CALUDE_real_root_implies_m_value_l4089_408918

theorem real_root_implies_m_value (m : ℂ) :
  (∃ x : ℝ, x^2 + (1 + 2*I)*x - 2*(m + 1) = 0) →
  (∃ b : ℝ, m = b*I) →
  (m = I ∨ m = -2*I) := by
sorry

end NUMINAMATH_CALUDE_real_root_implies_m_value_l4089_408918


namespace NUMINAMATH_CALUDE_meghans_money_is_550_l4089_408969

/-- Represents the number of bills of a specific denomination --/
structure BillCount where
  count : Nat
  denomination : Nat

/-- Calculates the total value of bills given their count and denomination --/
def billValue (b : BillCount) : Nat := b.count * b.denomination

/-- Represents Meghan's money --/
structure MeghansMoney where
  hundreds : BillCount
  fifties : BillCount
  tens : BillCount

/-- Calculates the total value of Meghan's money --/
def totalValue (m : MeghansMoney) : Nat :=
  billValue m.hundreds + billValue m.fifties + billValue m.tens

/-- Theorem stating that Meghan's total money is $550 --/
theorem meghans_money_is_550 (m : MeghansMoney) 
  (h1 : m.hundreds = { count := 2, denomination := 100 })
  (h2 : m.fifties = { count := 5, denomination := 50 })
  (h3 : m.tens = { count := 10, denomination := 10 }) :
  totalValue m = 550 := by sorry

end NUMINAMATH_CALUDE_meghans_money_is_550_l4089_408969


namespace NUMINAMATH_CALUDE_remainder_662_power_662_mod_13_l4089_408955

theorem remainder_662_power_662_mod_13 :
  662^662 % 13 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_662_power_662_mod_13_l4089_408955


namespace NUMINAMATH_CALUDE_candy_canes_count_l4089_408940

-- Define the problem parameters
def num_kids : ℕ := 3
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def total_stuffers : ℕ := 21

-- Define the function to calculate candy canes per stocking
def candy_canes_per_stocking : ℕ :=
  let non_candy_items_per_stocking := beanie_babies_per_stocking + books_per_stocking
  let total_non_candy_items := non_candy_items_per_stocking * num_kids
  let total_candy_canes := total_stuffers - total_non_candy_items
  total_candy_canes / num_kids

-- Theorem statement
theorem candy_canes_count : candy_canes_per_stocking = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_canes_count_l4089_408940


namespace NUMINAMATH_CALUDE_range_of_a_l4089_408924

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 + 5*x + 4 < 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4089_408924


namespace NUMINAMATH_CALUDE_equal_roots_implies_m_value_l4089_408950

/-- Prove that if the equation (x(x-1)-(m+1))/((x-1)(m-1)) = x/m has all equal roots, then m = -1/2 -/
theorem equal_roots_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m + 1) = 0) →
  m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_implies_m_value_l4089_408950


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4089_408967

theorem simplify_and_evaluate (a b : ℝ) (h : |2 - a + b| + (a * b + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4089_408967


namespace NUMINAMATH_CALUDE_min_value_fraction_l4089_408906

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l4089_408906


namespace NUMINAMATH_CALUDE_platform_length_l4089_408987

/-- The length of a platform passed by an accelerating train -/
theorem platform_length (l a t : ℝ) (h1 : l > 0) (h2 : a > 0) (h3 : t > 0) : ∃ P : ℝ,
  (l = (1/2) * a * t^2) →
  (l + P = (1/2) * a * (6*t)^2) →
  P = 17 * l := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l4089_408987
