import Mathlib

namespace parallel_lines_l1321_132159

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are coincident -/
def coincident (l1 l2 : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- The main theorem -/
theorem parallel_lines (a : ℝ) : 
  let l1 : Line := ⟨a, 3, a^2 - 5⟩
  let l2 : Line := ⟨1, a - 2, 4⟩
  (parallel l1 l2 ∧ ¬coincident l1 l2) ↔ a = 3 := by
  sorry

end parallel_lines_l1321_132159


namespace dorothy_age_ratio_l1321_132108

/-- Given Dorothy's sister's age and the condition about their future ages,
    prove that Dorothy is currently 3 times as old as her sister. -/
theorem dorothy_age_ratio (sister_age : ℕ) (dorothy_age : ℕ) : 
  sister_age = 5 →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age / sister_age = 3 := by
  sorry

end dorothy_age_ratio_l1321_132108


namespace fraction_comparison_geometric_sum_comparison_l1321_132160

theorem fraction_comparison (α β : ℝ) (hα : α = 1.00000000004) (hβ : β = 1.00000000002) :
  (1 + β) / (1 + β + β^2) > (1 + α) / (1 + α + α^2) := by sorry

theorem geometric_sum_comparison {a b : ℝ} {n : ℕ} (hab : a > b) (hb : b > 0) (hn : n > 0) :
  (b^n - 1) / (b^(n+1) - 1) > (a^n - 1) / (a^(n+1) - 1) := by sorry

end fraction_comparison_geometric_sum_comparison_l1321_132160


namespace linear_function_value_l1321_132141

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 5) 
  (h2 : f 2 = 8) 
  (h3 : f 3 = 11) 
  (h_linear : ∀ x, f x = a * x + b) : 
  f 4 = 14 := by
sorry

end linear_function_value_l1321_132141


namespace jason_attended_twelve_games_l1321_132167

def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (missed : ℕ) : ℕ :=
  planned_this_month + planned_last_month - missed

theorem jason_attended_twelve_games :
  games_attended 11 17 16 = 12 := by
  sorry

end jason_attended_twelve_games_l1321_132167


namespace sprint_jog_difference_value_l1321_132180

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ :=
  let sprint1 := 0.8932
  let sprint2 := 0.9821
  let sprint3 := 1.2534
  let jog1 := 0.7683
  let jog2 := 0.4356
  let jog3 := 0.6549
  (sprint1 + sprint2 + sprint3) - (jog1 + jog2 + jog3)

/-- Theorem stating that the difference between Darnel's total sprinting distance and total jogging distance is 1.2699 laps -/
theorem sprint_jog_difference_value : sprint_jog_difference = 1.2699 := by
  sorry

end sprint_jog_difference_value_l1321_132180


namespace marbles_selection_theorem_l1321_132133

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def ordinary_marbles : ℕ := total_marbles - special_marbles

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marbles_selection_theorem :
  (choose_marbles special_marbles 2 * choose_marbles ordinary_marbles 3) +
  (choose_marbles special_marbles 3 * choose_marbles ordinary_marbles 2) +
  (choose_marbles special_marbles 4 * choose_marbles ordinary_marbles 1) +
  (choose_marbles special_marbles 5 * choose_marbles ordinary_marbles 0) = 2121 :=
by sorry

end marbles_selection_theorem_l1321_132133


namespace quadratic_inequality_minimum_l1321_132155

theorem quadratic_inequality_minimum (b c : ℝ) : 
  (∀ x, (x^2 - (b+2)*x + c < 0) ↔ (2 < x ∧ x < 3)) →
  (∃ min : ℝ, min = 3 ∧ 
    ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ min ∧ 
    ∃ x₀ > 1, (x₀^2 - b*x₀ + c) / (x₀ - 1) = min) :=
by sorry

end quadratic_inequality_minimum_l1321_132155


namespace farmer_ploughing_problem_l1321_132198

/-- Represents the farmer's ploughing problem -/
def FarmerProblem (initial_productivity : ℝ) (productivity_increase : ℝ) 
  (total_area : ℝ) (days_ahead : ℕ) (initial_days : ℕ) : Prop :=
  let improved_productivity := initial_productivity * (1 + productivity_increase)
  let area_first_two_days := 2 * initial_productivity
  let remaining_area := total_area - area_first_two_days
  let remaining_days := remaining_area / improved_productivity
  initial_days = ⌈remaining_days⌉ + 2 + days_ahead

/-- The theorem statement for the farmer's ploughing problem -/
theorem farmer_ploughing_problem :
  FarmerProblem 120 0.25 1440 2 12 := by
  sorry

end farmer_ploughing_problem_l1321_132198


namespace wildflower_color_difference_l1321_132169

/-- Given the following conditions about wildflowers:
  - Total wildflowers picked: 44
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
Prove that there are 4 more flowers containing red than containing white. -/
theorem wildflower_color_difference 
  (total : ℕ) 
  (yellow_white : ℕ) 
  (red_yellow : ℕ) 
  (red_white : ℕ) 
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end wildflower_color_difference_l1321_132169


namespace completing_square_equivalence_l1321_132134

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end completing_square_equivalence_l1321_132134


namespace number_equation_l1321_132129

theorem number_equation : ∃ x : ℚ, (5 + 4/9) / 7 = 5 * x ∧ x = 49/315 := by
  sorry

end number_equation_l1321_132129


namespace square_diagonals_perpendicular_l1321_132181

structure Rhombus where
  diagonals_perpendicular : Bool

structure Square extends Rhombus

theorem square_diagonals_perpendicular (rhombus_property : Rhombus → Bool)
    (square_is_rhombus : Square → Rhombus)
    (h1 : ∀ r : Rhombus, rhombus_property r = r.diagonals_perpendicular)
    (h2 : ∀ s : Square, rhombus_property (square_is_rhombus s) = true) :
  ∀ s : Square, s.diagonals_perpendicular = true := by
  sorry

end square_diagonals_perpendicular_l1321_132181


namespace bella_steps_l1321_132115

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℚ := 1 / 3

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps :
  distance * speed_ratio / (1 + speed_ratio) / feet_per_step = steps_taken := by
  sorry

end bella_steps_l1321_132115


namespace similarity_coefficients_are_valid_l1321_132193

/-- A triangle with sides 2, 3, and 3 -/
structure OriginalTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : side3 = 3

/-- Similarity coefficients for the four triangles -/
structure SimilarityCoefficients where
  k1 : ℝ
  k2 : ℝ
  k3 : ℝ
  k4 : ℝ

/-- Predicate to check if the similarity coefficients are valid -/
def valid_coefficients (sc : SimilarityCoefficients) : Prop :=
  (sc.k1 = 1/2 ∧ sc.k2 = 1/2 ∧ sc.k3 = 1/2 ∧ sc.k4 = 1/2) ∨
  (sc.k1 = 6/13 ∧ sc.k2 = 4/13 ∧ sc.k3 = 9/13 ∧ sc.k4 = 6/13)

/-- Theorem stating that the similarity coefficients for the divided triangles are valid -/
theorem similarity_coefficients_are_valid (t : OriginalTriangle) (sc : SimilarityCoefficients) :
  valid_coefficients sc := by sorry

end similarity_coefficients_are_valid_l1321_132193


namespace difference_largest_smallest_l1321_132176

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  sorry

def smallest_number (digits : List Nat) : Nat :=
  sorry

theorem difference_largest_smallest :
  largest_number digits - smallest_number digits = 396 := by
  sorry

end difference_largest_smallest_l1321_132176


namespace power_two_33_mod_9_l1321_132175

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by sorry

end power_two_33_mod_9_l1321_132175


namespace intersection_range_l1321_132163

/-- The curve y = 1 + √(4 - x²) intersects with the line y = k(x + 2) + 5 at two points
    if and only if k is in the range [-1, -3/4) --/
theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (1 + Real.sqrt (4 - x₁^2) = k * (x₁ + 2) + 5) ∧
    (1 + Real.sqrt (4 - x₂^2) = k * (x₂ + 2) + 5)) ↔ 
  (k ≥ -1 ∧ k < -3/4) :=
sorry

end intersection_range_l1321_132163


namespace cube_sum_in_interval_l1321_132130

theorem cube_sum_in_interval (n : ℕ) : ∃ k x y : ℕ,
  (n : ℝ) - 4 * Real.sqrt (n : ℝ) ≤ k ∧
  k ≤ (n : ℝ) + 4 * Real.sqrt (n : ℝ) ∧
  k = x^3 + y^3 :=
by sorry

end cube_sum_in_interval_l1321_132130


namespace x_value_proof_l1321_132110

theorem x_value_proof (x : ℕ) : 
  (Nat.lcm x 18 - Nat.gcd x 18 = 120) → x = 42 := by
  sorry

end x_value_proof_l1321_132110


namespace expression_simplification_l1321_132194

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ 2) :
  ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2*x)) = x - 2 := by
  sorry

end expression_simplification_l1321_132194


namespace cube_diagonal_pairs_60_degrees_l1321_132142

/-- A regular hexahedron (cube) -/
structure Cube where
  /-- Number of faces in a cube -/
  faces : ℕ
  /-- Number of diagonals per face -/
  diagonals_per_face : ℕ
  /-- Total number of face diagonals -/
  total_diagonals : ℕ
  /-- Total number of possible diagonal pairs -/
  total_pairs : ℕ
  /-- Number of diagonal pairs that don't form a 60° angle -/
  non_60_pairs : ℕ

/-- The number of pairs of face diagonals in a cube that form a 60° angle -/
def pairs_forming_60_degrees (c : Cube) : ℕ :=
  c.total_pairs - c.non_60_pairs

/-- Theorem stating that in a regular hexahedron (cube), 
    the number of pairs of face diagonals that form a 60° angle is 48 -/
theorem cube_diagonal_pairs_60_degrees (c : Cube) 
  (h1 : c.faces = 6)
  (h2 : c.diagonals_per_face = 2)
  (h3 : c.total_diagonals = 12)
  (h4 : c.total_pairs = 66)
  (h5 : c.non_60_pairs = 18) :
  pairs_forming_60_degrees c = 48 := by
  sorry

end cube_diagonal_pairs_60_degrees_l1321_132142


namespace lowest_price_per_component_l1321_132112

/-- The lowest price per component that covers all costs for a computer manufacturer --/
theorem lowest_price_per_component 
  (cost_per_component : ℝ) 
  (shipping_cost_per_unit : ℝ) 
  (fixed_monthly_costs : ℝ) 
  (components_per_month : ℕ) 
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 7)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : components_per_month = 150) : 
  ∃ (price : ℝ), price = 197 ∧ 
    price * (components_per_month : ℝ) = 
      (cost_per_component + shipping_cost_per_unit) * (components_per_month : ℝ) + fixed_monthly_costs :=
by sorry

end lowest_price_per_component_l1321_132112


namespace division_problem_l1321_132151

theorem division_problem (dividend divisor : ℕ) : 
  (dividend / divisor = 3) → 
  (dividend % divisor = 20) → 
  (dividend + divisor + 3 + 20 = 303) → 
  (divisor = 65 ∧ dividend = 215) := by
  sorry

end division_problem_l1321_132151


namespace cos_36_degrees_l1321_132183

theorem cos_36_degrees (h : Real.sin (108 * π / 180) = 3 * Real.sin (36 * π / 180) - 4 * (Real.sin (36 * π / 180))^3) :
  Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
sorry

end cos_36_degrees_l1321_132183


namespace no_additional_omelets_l1321_132164

-- Define the number of eggs per omelet type
def eggs_plain : ℕ := 3
def eggs_cheese : ℕ := 4
def eggs_vegetable : ℕ := 5

-- Define the total number of eggs
def total_eggs : ℕ := 36

-- Define the number of omelets already requested
def plain_omelets : ℕ := 4
def cheese_omelets : ℕ := 2
def vegetable_omelets : ℕ := 3

-- Calculate the number of eggs used for requested omelets
def used_eggs : ℕ := plain_omelets * eggs_plain + cheese_omelets * eggs_cheese + vegetable_omelets * eggs_vegetable

-- Define the remaining eggs
def remaining_eggs : ℕ := total_eggs - used_eggs

-- Theorem: No additional omelets can be made
theorem no_additional_omelets :
  remaining_eggs < eggs_plain ∧ remaining_eggs < eggs_cheese ∧ remaining_eggs < eggs_vegetable :=
by sorry

end no_additional_omelets_l1321_132164


namespace cone_height_is_sqrt_3_l1321_132136

-- Define the cone structure
structure Cone where
  base_radius : ℝ
  height : ℝ
  slant_height : ℝ

-- Define the property of the cone's lateral surface
def lateral_surface_is_semicircle (c : Cone) : Prop :=
  c.slant_height = 2

-- Theorem statement
theorem cone_height_is_sqrt_3 (c : Cone) 
  (h_semicircle : lateral_surface_is_semicircle c) : 
  c.height = Real.sqrt 3 :=
sorry

end cone_height_is_sqrt_3_l1321_132136


namespace mixture_ratio_proof_l1321_132168

theorem mixture_ratio_proof (p q : ℝ) : 
  p + q = 35 →
  p / (q + 13) = 5 / 7 →
  p / q = 4 / 3 :=
by sorry

end mixture_ratio_proof_l1321_132168


namespace arithmetic_equation_l1321_132190

theorem arithmetic_equation : 4 * (8 - 2 + 3) - 7 = 29 := by
  sorry

end arithmetic_equation_l1321_132190


namespace jill_trips_to_fill_tank_l1321_132170

/-- Represents the water fetching problem with Jack and Jill -/
def WaterFetchingProblem (tank_capacity : ℕ) (bucket_capacity : ℕ) (jack_buckets : ℕ) 
  (jill_buckets : ℕ) (jack_trips : ℕ) (jill_trips : ℕ) (leak_rate : ℕ) : Prop :=
  ∃ (jill_total_trips : ℕ),
    -- The tank capacity is 600 gallons
    tank_capacity = 600 ∧
    -- Each bucket holds 5 gallons
    bucket_capacity = 5 ∧
    -- Jack carries 2 buckets per trip
    jack_buckets = 2 ∧
    -- Jill carries 1 bucket per trip
    jill_buckets = 1 ∧
    -- Jack makes 3 trips for every 2 trips Jill makes
    jack_trips = 3 ∧
    jill_trips = 2 ∧
    -- The tank leaks 2 gallons every time both return
    leak_rate = 2 ∧
    -- The number of trips Jill makes is 20
    jill_total_trips = 20 ∧
    -- The tank is filled after Jill's trips
    jill_total_trips * jill_trips * (jack_buckets * bucket_capacity * jack_trips + 
      jill_buckets * bucket_capacity * jill_trips - leak_rate) / (jack_trips + jill_trips) ≥ tank_capacity

/-- Theorem stating that given the conditions, Jill will make 20 trips before the tank is filled -/
theorem jill_trips_to_fill_tank : 
  WaterFetchingProblem 600 5 2 1 3 2 2 := by sorry

end jill_trips_to_fill_tank_l1321_132170


namespace smallest_two_digit_factor_of_5280_l1321_132137

theorem smallest_two_digit_factor_of_5280 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 5280 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 5280 → min x y ≥ 66) :=
by sorry

end smallest_two_digit_factor_of_5280_l1321_132137


namespace range_of_c_l1321_132145

def p (c : ℝ) := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > c

theorem range_of_c :
  ∀ c : ℝ, c > 0 →
  ((p c ∨ q c) ∧ ¬(p c ∧ q c)) →
  (c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1) :=
sorry

end range_of_c_l1321_132145


namespace apples_per_hour_l1321_132118

/-- Proves that eating the same number of apples every hour for 3 hours,
    totaling 15 apples, results in 5 apples per hour. -/
theorem apples_per_hour 
  (total_hours : ℕ) 
  (total_apples : ℕ) 
  (h1 : total_hours = 3) 
  (h2 : total_apples = 15) : 
  total_apples / total_hours = 5 := by
  sorry

end apples_per_hour_l1321_132118


namespace quadratic_roots_expression_l1321_132179

theorem quadratic_roots_expression (a b : ℝ) : 
  (3 * a^2 + 2 * a - 2 = 0) →
  (3 * b^2 + 2 * b - 2 = 0) →
  (2 * a / (a^2 - b^2) - 1 / (a - b) = -3/2) :=
by sorry

end quadratic_roots_expression_l1321_132179


namespace second_arrangement_column_size_l1321_132189

/-- Represents a group of people that can be arranged in columns. -/
structure PeopleGroup where
  /-- The total number of people in the group -/
  total : ℕ
  /-- The number of columns formed when 30 people stand in each column -/
  columns_with_30 : ℕ
  /-- The number of columns formed in the second arrangement -/
  columns_in_second : ℕ
  /-- Ensures that 30 people per column forms the specified number of columns -/
  h_first_arrangement : total = 30 * columns_with_30

/-- 
Given a group of people where 30 people per column forms 16 columns,
if the same group is rearranged into 12 columns,
then there will be 40 people in each column of the second arrangement.
-/
theorem second_arrangement_column_size (g : PeopleGroup) 
    (h_16_columns : g.columns_with_30 = 16)
    (h_12_columns : g.columns_in_second = 12) :
    g.total / g.columns_in_second = 40 := by
  sorry

end second_arrangement_column_size_l1321_132189


namespace constant_value_l1321_132197

theorem constant_value (t : ℝ) (constant : ℝ) :
  let x := constant - 2 * t
  let y := 2 * t - 2
  (t = 0.75 → x = y) →
  constant = 1 := by sorry

end constant_value_l1321_132197


namespace parallel_lines_chord_distance_l1321_132199

theorem parallel_lines_chord_distance (r : ℝ) (d : ℝ) : 
  r > 0 → d > 0 →
  36 * r^2 = 36 * 324 + (1/4) * d^2 * 36 →
  40 * r^2 = 40 * 400 + 40 * d^2 →
  d = Real.sqrt (304/3) :=
sorry

end parallel_lines_chord_distance_l1321_132199


namespace rotation_exists_l1321_132152

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  O : Point3D
  B : Point3D

-- Define congruence for triangles
def congruent (t1 t2 : Triangle3D) : Prop :=
  (t1.A.x - t1.O.x)^2 + (t1.A.y - t1.O.y)^2 + (t1.A.z - t1.O.z)^2 =
    (t2.A.x - t2.O.x)^2 + (t2.A.y - t2.O.y)^2 + (t2.A.z - t2.O.z)^2 ∧
  (t1.B.x - t1.O.x)^2 + (t1.B.y - t1.O.y)^2 + (t1.B.z - t1.O.z)^2 =
    (t2.B.x - t2.O.x)^2 + (t2.B.y - t2.O.y)^2 + (t2.B.z - t2.O.z)^2

-- Define when two triangles are not in the same plane
def not_coplanar (t1 t2 : Triangle3D) : Prop :=
  ¬ ∃ (a b c d : ℝ),
    a * (t1.A.x - t1.O.x) + b * (t1.A.y - t1.O.y) + c * (t1.A.z - t1.O.z) + d = 0 ∧
    a * (t1.B.x - t1.O.x) + b * (t1.B.y - t1.O.y) + c * (t1.B.z - t1.O.z) + d = 0 ∧
    a * (t2.A.x - t2.O.x) + b * (t2.A.y - t2.O.y) + c * (t2.A.z - t2.O.z) + d = 0 ∧
    a * (t2.B.x - t2.O.x) + b * (t2.B.y - t2.O.y) + c * (t2.B.z - t2.O.z) + d = 0

-- Define rotation in 3D space
structure Rotation3D where
  axis : Point3D
  angle : ℝ

-- Theorem statement
theorem rotation_exists (t1 t2 : Triangle3D)
  (h1 : congruent t1 t2)
  (h2 : t1.O = t2.O)
  (h3 : not_coplanar t1 t2) :
  ∃ (r : Rotation3D), r.axis.x * (t1.A.x - t1.O.x) + r.axis.y * (t1.A.y - t1.O.y) + r.axis.z * (t1.A.z - t1.O.z) = 0 ∧
                      r.axis.x * (t1.B.x - t1.O.x) + r.axis.y * (t1.B.y - t1.O.y) + r.axis.z * (t1.B.z - t1.O.z) = 0 :=
by sorry

end rotation_exists_l1321_132152


namespace reflection_of_point_2_5_l1321_132109

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of point (2, 5) across the x-axis is (2, -5) -/
theorem reflection_of_point_2_5 :
  let p := Point.mk 2 5
  reflectAcrossXAxis p = Point.mk 2 (-5) := by
  sorry

end reflection_of_point_2_5_l1321_132109


namespace fraction_simplification_l1321_132195

theorem fraction_simplification (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x + 1)) / ((2 / (x^2 - 1)) + (1 / (x + 1))) = (2*x - 2) / (x + 1) := by
  sorry

end fraction_simplification_l1321_132195


namespace trig_expression_max_value_trig_expression_max_achievable_l1321_132154

theorem trig_expression_max_value (A B C : Real) :
  (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 ≤ 1 :=
sorry

theorem trig_expression_max_achievable :
  ∃ (A B C : Real), (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 = 1 :=
sorry

end trig_expression_max_value_trig_expression_max_achievable_l1321_132154


namespace concentric_circles_circumference_difference_l1321_132166

/-- The difference in circumferences of two concentric circles -/
theorem concentric_circles_circumference_difference 
  (inner_diameter : ℝ) 
  (distance_between_circles : ℝ) : 
  inner_diameter = 100 → 
  distance_between_circles = 15 → 
  (inner_diameter + 2 * distance_between_circles) * π - inner_diameter * π = 30 * π := by
  sorry

end concentric_circles_circumference_difference_l1321_132166


namespace max_value_theorem_l1321_132132

theorem max_value_theorem (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -4) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -4 →
    Real.sqrt (4*a + 4) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 16) ≤
    Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16)) ∧
  Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16) = 2 * Real.sqrt 30 :=
by sorry

end max_value_theorem_l1321_132132


namespace find_subtracted_number_l1321_132185

theorem find_subtracted_number (x N : ℝ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 := by
  sorry

end find_subtracted_number_l1321_132185


namespace sixth_term_of_geometric_sequence_l1321_132113

/-- Given a geometric sequence with first term 3 and second term -1/2, 
    prove that its sixth term is -1/2592 -/
theorem sixth_term_of_geometric_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 3) (h₂ : a₂ = -1/2) :
  let r := a₂ / a₁
  let a_n (n : ℕ) := a₁ * r^(n - 1)
  a_n 6 = -1/2592 := by
sorry

end sixth_term_of_geometric_sequence_l1321_132113


namespace arccos_cos_three_l1321_132138

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end arccos_cos_three_l1321_132138


namespace sequence_properties_l1321_132148

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define a property for isolated points in a graph
def HasIsolatedPoints (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Theorem statement
theorem sequence_properties :
  (∃ (s : Sequence), True) ∧
  (∀ (s : Sequence), HasIsolatedPoints s) :=
sorry

end sequence_properties_l1321_132148


namespace arithmetic_sequence_with_geometric_mean_l1321_132188

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The 5th term is the geometric mean of the 3rd and 11th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 3 * a 11

theorem arithmetic_sequence_with_geometric_mean 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_mean_condition a) : 
  a 1 = -1 := by
  sorry


end arithmetic_sequence_with_geometric_mean_l1321_132188


namespace fraction_equals_decimal_l1321_132128

theorem fraction_equals_decimal : (8 : ℚ) / (4 * 25) = 0.08 := by sorry

end fraction_equals_decimal_l1321_132128


namespace square_difference_nonnegative_l1321_132135

theorem square_difference_nonnegative (a b : ℝ) : (a - b)^2 ≥ 0 := by
  sorry

end square_difference_nonnegative_l1321_132135


namespace shaded_percentage_is_75_percent_l1321_132121

/-- Represents a square grid composed of smaller squares -/
structure Grid where
  side_length : ℕ
  small_squares : ℕ
  shaded_squares : ℕ

/-- Calculates the percentage of shaded squares in the grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.shaded_squares : ℚ) / (g.small_squares : ℚ) * 100

/-- Theorem stating that the percentage of shaded squares is 75% -/
theorem shaded_percentage_is_75_percent (g : Grid) 
  (h1 : g.side_length = 8)
  (h2 : g.small_squares = g.side_length * g.side_length)
  (h3 : g.shaded_squares = 48) : 
  shaded_percentage g = 75 := by
  sorry

end shaded_percentage_is_75_percent_l1321_132121


namespace similar_triangle_lines_count_l1321_132140

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop := sorry

/-- A line in a 2D plane -/
structure Line :=
  (point : Point)
  (direction : ℝ × ℝ)

/-- Predicate to check if a line intersects a triangle -/
def intersects (L : Line) (T : Triangle) : Prop := sorry

/-- Predicate to check if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Function to count the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles -/
def countSimilarTriangleLines (T : Triangle) (P : Point) : ℕ := sorry

/-- Theorem stating that the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles is 6 -/
theorem similar_triangle_lines_count (T : Triangle) (P : Point) 
  (h : isInside P T) : countSimilarTriangleLines T P = 6 := by sorry

end similar_triangle_lines_count_l1321_132140


namespace circle_radius_l1321_132171

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end circle_radius_l1321_132171


namespace quadratic_has_two_real_roots_roots_difference_implies_m_values_l1321_132107

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - (m-1)*x + m - 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 m = 0 ∧ quadratic x2 m = 0 :=
sorry

-- Theorem 2: When the difference between the roots is 3, m = 0 or m = 6
theorem roots_difference_implies_m_values :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 0 = 0 ∧ quadratic x2 0 = 0 ∧ |x1 - x2| = 3 ∨
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 6 = 0 ∧ quadratic x2 6 = 0 ∧ |x1 - x2| = 3 :=
sorry

end quadratic_has_two_real_roots_roots_difference_implies_m_values_l1321_132107


namespace age_ratio_is_two_to_one_l1321_132106

/-- The ages of two people A and B satisfy certain conditions. -/
structure AgeRatio where
  a : ℕ  -- Current age of A
  b : ℕ  -- Current age of B
  past_future_ratio : a - 4 = b + 4  -- Ratio 1:1 for A's past and B's future
  future_past_ratio : a + 4 = 5 * (b - 4)  -- Ratio 5:1 for A's future and B's past

/-- The ratio of current ages of A and B is 2:1 -/
theorem age_ratio_is_two_to_one (ages : AgeRatio) : 
  2 * ages.b = ages.a := by sorry

end age_ratio_is_two_to_one_l1321_132106


namespace min_value_sum_reciprocals_l1321_132174

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_10 : a + b + c + d = 10) :
  (1/a + 9/b + 25/c + 49/d) ≥ 25.6 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 0 < d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 10 ∧
    1/a₀ + 9/b₀ + 25/c₀ + 49/d₀ = 25.6 :=
by sorry

end min_value_sum_reciprocals_l1321_132174


namespace max_value_of_f_l1321_132156

def f (x : ℝ) (a : ℝ) := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x a ≥ f y a) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) :=
by sorry

end max_value_of_f_l1321_132156


namespace min_value_product_l1321_132122

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by sorry

end min_value_product_l1321_132122


namespace quadratic_residue_prime_power_l1321_132105

theorem quadratic_residue_prime_power (p : Nat) (a : Nat) (k : Nat) :
  Nat.Prime p →
  Odd p →
  (∃ y : Nat, y^2 ≡ a [MOD p]) →
  ∃ z : Nat, z^2 ≡ a [MOD p^k] :=
sorry

end quadratic_residue_prime_power_l1321_132105


namespace lara_flowers_in_vase_l1321_132192

/-- Calculates the number of flowers Lara put in the vase --/
def flowersInVase (totalFlowers : ℕ) (toMom : ℕ) (extraToGrandma : ℕ) : ℕ :=
  let toGrandma := toMom + extraToGrandma
  let remainingAfterMomAndGrandma := totalFlowers - toMom - toGrandma
  let toSister := remainingAfterMomAndGrandma / 3
  let toBestFriend := toSister + toSister / 4
  remainingAfterMomAndGrandma - toSister - toBestFriend

theorem lara_flowers_in_vase :
  flowersInVase 52 15 6 = 5 := by
  sorry

end lara_flowers_in_vase_l1321_132192


namespace fourth_machine_works_twelve_hours_l1321_132177

/-- Represents a factory with machines producing material. -/
structure Factory where
  num_original_machines : ℕ
  hours_per_day_original : ℕ
  production_rate : ℕ
  price_per_kg : ℕ
  total_revenue : ℕ

/-- Calculates the hours worked by the fourth machine. -/
def fourth_machine_hours (f : Factory) : ℕ :=
  let original_production := f.num_original_machines * f.hours_per_day_original * f.production_rate
  let original_revenue := original_production * f.price_per_kg
  let fourth_machine_revenue := f.total_revenue - original_revenue
  let fourth_machine_production := fourth_machine_revenue / f.price_per_kg
  fourth_machine_production / f.production_rate

/-- Theorem stating the fourth machine works 12 hours a day. -/
theorem fourth_machine_works_twelve_hours (f : Factory) 
  (h1 : f.num_original_machines = 3)
  (h2 : f.hours_per_day_original = 23)
  (h3 : f.production_rate = 2)
  (h4 : f.price_per_kg = 50)
  (h5 : f.total_revenue = 8100) :
  fourth_machine_hours f = 12 := by
  sorry

end fourth_machine_works_twelve_hours_l1321_132177


namespace not_prime_n_l1321_132126

theorem not_prime_n (p a b c n : ℕ) : 
  Nat.Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ (a + (n-1) * b) →
  p^2 ∣ (b + (n-1) * c) →
  p^2 ∣ (c + (n-1) * a) →
  ¬(Nat.Prime n) :=
by sorry


end not_prime_n_l1321_132126


namespace snail_movement_bound_l1321_132158

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in minutes -/
  total_time : ℝ
  /-- The movement function: time → distance -/
  movement : ℝ → ℝ
  /-- Ensures the movement is non-negative -/
  non_negative : ∀ t, 0 ≤ movement t
  /-- Ensures the movement is monotonically increasing -/
  monotone : ∀ t₁ t₂, t₁ ≤ t₂ → movement t₁ ≤ movement t₂

/-- The observation condition: for any 1-minute interval, the snail moves exactly 1 meter -/
def observation_condition (sm : SnailMovement) : Prop :=
  ∀ t, 0 ≤ t ∧ t + 1 ≤ sm.total_time → sm.movement (t + 1) - sm.movement t = 1

/-- The theorem statement -/
theorem snail_movement_bound (sm : SnailMovement) 
    (h_time : sm.total_time = 6)
    (h_obs : observation_condition sm) :
    sm.movement sm.total_time ≤ 10 := by
  sorry

end snail_movement_bound_l1321_132158


namespace quadratic_function_determination_l1321_132173

/-- Given real numbers a, b, c, if f(x) = ax^2 + bx + c, g(x) = ax + b, 
    and the maximum value of g(x) is 2 when -1 ≤ x ≤ 1, then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination (a b c : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x, -1 ≤ x → x ≤ 1 → g x ≤ 2)
  (h_reaches_max : ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
sorry

end quadratic_function_determination_l1321_132173


namespace least_number_of_cans_l1321_132101

def maaza_liters : ℕ := 40
def pepsi_liters : ℕ := 144
def sprite_liters : ℕ := 368

theorem least_number_of_cans : 
  ∃ (can_size : ℕ), 
    can_size > 0 ∧
    maaza_liters % can_size = 0 ∧
    pepsi_liters % can_size = 0 ∧
    sprite_liters % can_size = 0 ∧
    (maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size = 69) ∧
    ∀ (other_size : ℕ), 
      other_size > 0 →
      maaza_liters % other_size = 0 →
      pepsi_liters % other_size = 0 →
      sprite_liters % other_size = 0 →
      (maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size ≥ 69) :=
by
  sorry

end least_number_of_cans_l1321_132101


namespace f_zero_at_three_l1321_132191

def f (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem f_zero_at_three (s : ℝ) : f 3 s = 0 ↔ s = -885 := by sorry

end f_zero_at_three_l1321_132191


namespace factorization_equality_l1321_132165

theorem factorization_equality (y : ℝ) : 49 - 16 * y^2 + 8 * y = (7 - 4 * y) * (7 + 4 * y) := by
  sorry

end factorization_equality_l1321_132165


namespace smallest_solution_congruence_l1321_132143

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (45 * x + 13) % 17 = 5 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 17 = 5 % 17 → x ≤ y :=
by
  use 11
  sorry

end smallest_solution_congruence_l1321_132143


namespace big_crash_frequency_is_20_l1321_132161

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency (total_accidents : ℕ) (total_time : ℕ) (collision_frequency : ℕ) : ℕ :=
  let regular_collisions := total_time / collision_frequency
  let big_crashes := total_accidents - regular_collisions
  total_time / big_crashes

/-- Theorem stating the frequency of big crashes given the problem conditions -/
theorem big_crash_frequency_is_20 :
  big_crash_frequency 36 (4 * 60) 10 = 20 := by
  sorry

#eval big_crash_frequency 36 (4 * 60) 10

end big_crash_frequency_is_20_l1321_132161


namespace basketball_free_throw_percentage_l1321_132104

theorem basketball_free_throw_percentage 
  (p : ℝ) 
  (h : 0 ≤ p ∧ p ≤ 1) 
  (h_prob : (1 - p)^2 + 2*p*(1 - p) = 16/25) : 
  p = 3/5 := by sorry

end basketball_free_throw_percentage_l1321_132104


namespace initial_papers_count_l1321_132125

/-- The number of papers Charles initially bought -/
def initial_papers : ℕ := sorry

/-- The number of pictures Charles drew today -/
def pictures_today : ℕ := 6

/-- The number of pictures Charles drew before work yesterday -/
def pictures_before_work : ℕ := 6

/-- The number of pictures Charles drew after work yesterday -/
def pictures_after_work : ℕ := 6

/-- The number of papers Charles has left -/
def papers_left : ℕ := 2

/-- Theorem stating that the initial number of papers is equal to the sum of papers used for pictures and papers left -/
theorem initial_papers_count : 
  initial_papers = pictures_today + pictures_before_work + pictures_after_work + papers_left :=
by sorry

end initial_papers_count_l1321_132125


namespace model_height_calculation_l1321_132111

/-- The height of the Eiffel Tower in meters -/
def eiffel_height : ℝ := 320

/-- The capacity of the Eiffel Tower's observation deck in number of people -/
def eiffel_capacity : ℝ := 800

/-- The space required per person in square meters -/
def space_per_person : ℝ := 1

/-- The equivalent capacity of Mira's model in number of people -/
def model_capacity : ℝ := 0.8

/-- The height of Mira's model in meters -/
def model_height : ℝ := 10.12

theorem model_height_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (model_height - eiffel_height * (model_capacity / eiffel_capacity).sqrt) < ε :=
sorry

end model_height_calculation_l1321_132111


namespace only_shanxi_spirit_census_l1321_132182

-- Define the survey types
inductive SurveyType
  | Census
  | Sample

-- Define the survey options
inductive SurveyOption
  | ArtilleryShells
  | TVRatings
  | FishSpecies
  | ShanxiSpiritAwareness

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.ArtilleryShells => SurveyType.Sample
  | SurveyOption.TVRatings => SurveyType.Sample
  | SurveyOption.FishSpecies => SurveyType.Sample
  | SurveyOption.ShanxiSpiritAwareness => SurveyType.Census

-- Theorem stating that only ShanxiSpiritAwareness is suitable for a census survey
theorem only_shanxi_spirit_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.ShanxiSpiritAwareness :=
by
  sorry


end only_shanxi_spirit_census_l1321_132182


namespace two_pipes_fill_time_l1321_132149

/-- Given two pipes filling a tank, where one pipe is 3 times as fast as the other,
    and the slower pipe can fill the tank in 160 minutes,
    prove that both pipes together can fill the tank in 40 minutes. -/
theorem two_pipes_fill_time (slow_pipe_time : ℝ) (fast_pipe_time : ℝ) : 
  slow_pipe_time = 160 →
  fast_pipe_time = slow_pipe_time / 3 →
  (1 / fast_pipe_time + 1 / slow_pipe_time)⁻¹ = 40 :=
by sorry

end two_pipes_fill_time_l1321_132149


namespace rational_solution_product_l1321_132184

theorem rational_solution_product : ∃ (k₁ k₂ : ℕ+), 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₁.val = 0) ∧ 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₂.val = 0) ∧ 
  (∀ (k : ℕ+), (∃ (x : ℚ), 3 * x^2 + 17 * x + k.val = 0) → k = k₁ ∨ k = k₂) ∧
  k₁.val * k₂.val = 336 := by
sorry

end rational_solution_product_l1321_132184


namespace sodium_atom_diameter_scientific_notation_l1321_132147

theorem sodium_atom_diameter_scientific_notation :
  ∃ (n : ℤ), (0.0000000599 : ℝ) = 5.99 * (10 : ℝ) ^ n → n = -8 := by
  sorry

end sodium_atom_diameter_scientific_notation_l1321_132147


namespace multiply_monomials_l1321_132124

theorem multiply_monomials (x : ℝ) : 2 * x * (5 * x^2) = 10 * x^3 := by
  sorry

end multiply_monomials_l1321_132124


namespace property_tax_increase_l1321_132100

/-- Represents the property tax increase in Township K --/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) : 
  new_value * tax_rate - initial_value * tax_rate = 800 := by
  sorry

#check property_tax_increase

end property_tax_increase_l1321_132100


namespace min_value_fraction_l1321_132146

theorem min_value_fraction (a b : ℝ) (h1 : a > 2*b) (h2 : b > 0) :
  (a^4 + 1) / (b * (a - 2*b)) ≥ 16 := by
  sorry

end min_value_fraction_l1321_132146


namespace museum_paintings_l1321_132114

theorem museum_paintings (initial : ℕ) (left : ℕ) (removed : ℕ) :
  initial = 1795 →
  left = 1322 →
  removed = initial - left →
  removed = 473 :=
by sorry

end museum_paintings_l1321_132114


namespace base_prime_repr_225_l1321_132144

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Theorem: The base prime representation of 225 is [2, 2, 0] -/
theorem base_prime_repr_225 : 
  base_prime_repr 225 = [2, 2, 0] :=
sorry

end base_prime_repr_225_l1321_132144


namespace circle_x_axis_intersection_l1321_132119

/-- A circle with diameter endpoints (3,2) and (11,8) intersects the x-axis at x = 7 -/
theorem circle_x_axis_intersection :
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (11, 8)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius : ℝ := ((p1.1 - center.1)^2 + (p1.2 - center.2)^2).sqrt
  ∃ x : ℝ, x ≠ p1.1 ∧ (x - center.1)^2 + center.2^2 = radius^2 ∧ x = 7 :=
by sorry

end circle_x_axis_intersection_l1321_132119


namespace probability_selecting_A_and_B_l1321_132178

theorem probability_selecting_A_and_B : 
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let total_combinations := Nat.choose total_students selected_students
  let favorable_combinations := Nat.choose (total_students - 2) (selected_students - 2)
  (favorable_combinations : ℚ) / total_combinations = 3 / 10 :=
sorry

end probability_selecting_A_and_B_l1321_132178


namespace find_x_value_l1321_132120

theorem find_x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((3 * x) / 7) = x) : x = 3 / 7 := by
  sorry

end find_x_value_l1321_132120


namespace rectangle_divided_by_line_l1321_132186

/-- 
Given a rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2),
if a line passing through the origin (0, 0) divides the rectangle into two identical quadrilaterals
and has a slope of 1/3, then x = 5.
-/
theorem rectangle_divided_by_line (x : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    -- l is a line passing through the origin
    (0, 0) ∈ l ∧
    -- l divides the rectangle into two identical quadrilaterals
    (∃ m : ℝ × ℝ, m ∈ l ∧ m.1 = (1 + x) / 2 ∧ m.2 = 1) ∧
    -- The slope of l is 1/3
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (q.2 - p.2) / (q.1 - p.1) = 1/3)) →
  x = 5 := by
sorry

end rectangle_divided_by_line_l1321_132186


namespace least_positive_integer_with_given_remainders_l1321_132117

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ∧
  (N % 11 = 10) ∧
  (∀ M : ℕ, M > 0 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 → M ≥ N) ∧
  N = 27719 :=
by sorry

end least_positive_integer_with_given_remainders_l1321_132117


namespace smallest_number_divisible_by_multiples_l1321_132103

theorem smallest_number_divisible_by_multiples (n : ℕ) : n = 200 ↔ 
  (∀ m : ℕ, m < n → ¬(15 ∣ (m - 20) ∧ 30 ∣ (m - 20) ∧ 45 ∣ (m - 20) ∧ 60 ∣ (m - 20))) ∧
  (15 ∣ (n - 20) ∧ 30 ∣ (n - 20) ∧ 45 ∣ (n - 20) ∧ 60 ∣ (n - 20)) := by
  sorry

end smallest_number_divisible_by_multiples_l1321_132103


namespace rectangle_area_l1321_132131

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l = 3 * w ∧ x^2 = l^2 + w^2 ∧ l * w = (3 * x^2) / 10 :=
by
  sorry

end rectangle_area_l1321_132131


namespace eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l1321_132116

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  h_a_pos : 1 < a
  h_a_bound : a < 2 + Real.sqrt 5

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.a^2 - 1) / (4 * e.a))

/-- Theorem: As 'a' increases, the eccentricity decreases -/
theorem eccentricity_decreases_as_a_increases (e1 e2 : Ellipse) 
    (h : e1.a < e2.a) : eccentricity e2 < eccentricity e1 := by
  sorry

/-- Corollary: As 'a' increases, the ellipse becomes closer to a circle -/
theorem ellipse_approaches_circle (e1 e2 : Ellipse) (h : e1.a < e2.a) :
    ∃ (c : ℝ), 0 < c ∧ c < 1 ∧ eccentricity e2 < c * eccentricity e1 := by
  sorry

end eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l1321_132116


namespace area_ratio_bound_for_special_triangles_l1321_132150

/-- Given two right-angled triangles where the incircle radius of the first equals
    the circumcircle radius of the second, prove that the ratio of their areas
    is at least 3 + 2√2 -/
theorem area_ratio_bound_for_special_triangles (S S' r : ℝ) :
  (∃ (a b c a' b' c' : ℝ),
    -- First triangle is right-angled
    a^2 + b^2 = c^2 ∧
    -- Second triangle is right-angled
    a'^2 + b'^2 = c'^2 ∧
    -- Incircle radius of first triangle equals circumcircle radius of second
    r = c' / 2 ∧
    -- Area formulas
    S = r^2 * (a/r + b/r + c/r - π) / 2 ∧
    S' = a' * b' / 2) →
  S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end area_ratio_bound_for_special_triangles_l1321_132150


namespace quadratic_polynomial_problem_l1321_132172

theorem quadratic_polynomial_problem (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  (∀ x, (x - 2) * (x + 2) * (x - 9) ∣ (p x)^3 - x) →
  p 14 = -36 / 79 := by
sorry

end quadratic_polynomial_problem_l1321_132172


namespace water_evaporation_per_day_l1321_132162

/-- Proves that given the initial conditions, the amount of water evaporated per day is correct -/
theorem water_evaporation_per_day 
  (initial_water : ℝ) 
  (evaporation_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_water = 10) 
  (h2 : evaporation_percentage = 7.000000000000001) 
  (h3 : days = 50) : 
  (initial_water * evaporation_percentage / 100) / days = 0.014000000000000002 := by
  sorry

#check water_evaporation_per_day

end water_evaporation_per_day_l1321_132162


namespace printer_time_calculation_l1321_132139

/-- Given a printer that prints 23 pages per minute, prove that it takes 15 minutes to print 345 pages. -/
theorem printer_time_calculation (print_rate : ℕ) (total_pages : ℕ) (time : ℕ) : 
  print_rate = 23 → total_pages = 345 → time = total_pages / print_rate → time = 15 := by
  sorry

end printer_time_calculation_l1321_132139


namespace river_speed_proof_l1321_132196

-- Define the problem parameters
def distance : ℝ := 200
def timeInterval : ℝ := 4
def speedA : ℝ := 36
def speedB : ℝ := 64

-- Define the river current speed as a variable
def riverSpeed : ℝ := 14

-- Theorem statement
theorem river_speed_proof :
  -- First meeting time
  let firstMeetTime : ℝ := distance / (speedA + speedB)
  -- Total time
  let totalTime : ℝ := firstMeetTime + timeInterval
  -- Total distance covered
  let totalDistance : ℝ := 3 * distance
  -- Equation for boat A's journey
  totalDistance = (speedA + riverSpeed + speedA - riverSpeed) * totalTime →
  -- Conclusion
  riverSpeed = 14 := by
  sorry

end river_speed_proof_l1321_132196


namespace problem_solution_l1321_132157

-- Statement ①
def statement1 (a b c : ℝ) : Prop :=
  (a > b → c^2 * a > c^2 * b)

-- Statement ②
def statement2 (m : ℝ) : Prop :=
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0)

-- Statement ③
def statement3 (x y : ℝ) : Prop :=
  (x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10)

theorem problem_solution :
  (¬ ∀ a b c : ℝ, ¬statement1 a b c) ∧
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ∧
  ((∀ x y : ℝ, x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10) ∧
   ¬(∀ x y : ℝ, x^2 - y^2 - 3*x + 7*y = 10 → x + y = 5)) :=
by sorry

end problem_solution_l1321_132157


namespace sqrt_inequality_l1321_132127

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end sqrt_inequality_l1321_132127


namespace smaller_circles_radius_l1321_132123

/-- Configuration of circles -/
structure CircleConfiguration where
  centralRadius : ℝ
  smallerRadius : ℝ
  numSmallerCircles : ℕ

/-- Defines a valid configuration of circles -/
def isValidConfiguration (config : CircleConfiguration) : Prop :=
  config.centralRadius = 1 ∧
  config.numSmallerCircles = 6 ∧
  -- Each smaller circle touches two others and the central circle
  -- (This condition is implicit in the geometry of the problem)
  True

/-- Theorem stating the radius of smaller circles in the given configuration -/
theorem smaller_circles_radius (config : CircleConfiguration)
  (h : isValidConfiguration config) :
  config.smallerRadius = 1 + Real.sqrt 2 := by
  sorry

end smaller_circles_radius_l1321_132123


namespace equation_solution_l1321_132153

theorem equation_solution (x : ℝ) : (25 : ℝ) / 75 = (x / 75) ^ 3 → x = 75 / (3 : ℝ) ^ (1/3) :=
by
  sorry

end equation_solution_l1321_132153


namespace track_circumference_l1321_132187

/-- The circumference of a circular track given specific meeting conditions of two travelers -/
theorem track_circumference : 
  ∀ (circumference : ℝ) 
    (speed_A speed_B : ℝ) 
    (first_meeting second_meeting : ℝ),
  speed_A > 0 →
  speed_B > 0 →
  first_meeting = 150 →
  second_meeting = circumference - 90 →
  first_meeting / (circumference / 2 - first_meeting) = 
    (circumference / 2 + 90) / (circumference - 90) →
  circumference = 720 := by
sorry

end track_circumference_l1321_132187


namespace isabel_piggy_bank_l1321_132102

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  (initial_amount / 2) / 2 = 51 → initial_amount = 204 := by
  sorry

end isabel_piggy_bank_l1321_132102
