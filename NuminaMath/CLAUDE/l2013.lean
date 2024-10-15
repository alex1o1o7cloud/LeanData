import Mathlib

namespace NUMINAMATH_CALUDE_triangle_equal_area_l2013_201398

/-- Given two triangles FGH and IJK with the specified properties, prove that JK = 10 -/
theorem triangle_equal_area (FG FH IJ IK : ℝ) (angle_GFH angle_IJK : ℝ) :
  FG = 5 →
  FH = 4 →
  angle_GFH = 30 * π / 180 →
  IJ = 2 →
  IK = 6 →
  angle_IJK = 30 * π / 180 →
  angle_GFH = angle_IJK →
  (1/2 * FG * FH * Real.sin angle_GFH) = (1/2 * IJ * 10 * Real.sin angle_IJK) →
  ∃ (JK : ℝ), JK = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equal_area_l2013_201398


namespace NUMINAMATH_CALUDE_plant_arrangements_eq_1271040_l2013_201386

/-- The number of ways to arrange 5 basil plants and 5 tomato plants with given conditions -/
def plant_arrangements : ℕ :=
  let basil_count : ℕ := 5
  let tomato_count : ℕ := 5
  let tomato_group1_size : ℕ := 2
  let tomato_group2_size : ℕ := 3
  let total_groups : ℕ := basil_count + 2  -- 5 basil plants + 2 tomato groups

  Nat.factorial total_groups *
  (Nat.choose total_groups basil_count * Nat.choose 2 1) *
  Nat.factorial tomato_group1_size *
  Nat.factorial tomato_group2_size

theorem plant_arrangements_eq_1271040 : plant_arrangements = 1271040 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangements_eq_1271040_l2013_201386


namespace NUMINAMATH_CALUDE_pattern_proof_l2013_201332

theorem pattern_proof (a : ℕ) : 4 * a * (a + 1) + 1 = (2 * a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l2013_201332


namespace NUMINAMATH_CALUDE_francine_work_schedule_l2013_201395

/-- The number of days Francine does not go to work every week -/
def days_not_working : ℕ :=
  7 - (2240 / (4 * 140))

theorem francine_work_schedule :
  days_not_working = 3 :=
sorry

end NUMINAMATH_CALUDE_francine_work_schedule_l2013_201395


namespace NUMINAMATH_CALUDE_parking_garage_floor_distance_l2013_201313

/-- Calculates the distance between floors in a parking garage --/
theorem parking_garage_floor_distance (
  total_floors : ℕ) 
  (gate_interval : ℕ) 
  (gate_time : ℝ) 
  (driving_speed : ℝ) 
  (total_time : ℝ)
  (h1 : total_floors = 12)
  (h2 : gate_interval = 3)
  (h3 : gate_time = 120) -- 2 minutes in seconds
  (h4 : driving_speed = 10)
  (h5 : total_time = 1440) :
  ∃ (distance : ℝ), abs (distance - 872.7) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_floor_distance_l2013_201313


namespace NUMINAMATH_CALUDE_at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l2013_201300

-- Define the possible makers of the caskets
inductive Maker
| Cellini
| CelliniSon
| Other

-- Define the caskets
structure Casket where
  material : String
  inscription : String
  maker : Maker

-- Define the problem setup
def goldenCasket : Casket := {
  material := "golden"
  inscription := "The silver casket was made by Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

def silverCasket : Casket := {
  material := "silver"
  inscription := "The golden casket was made by someone other than Cellini."
  maker := Maker.Other -- Initial assumption, will be proved
}

-- The main theorem to prove
theorem at_least_one_by_cellini_son (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  g.maker = Maker.CelliniSon ∨ s.maker = Maker.CelliniSon := by
  sorry

-- Additional helper theorems if needed
theorem not_both_by_cellini (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Cellini ∧ s.maker = Maker.Cellini) := by
  sorry

theorem not_both_by_other (g : Casket) (s : Casket)
  (hg : g = goldenCasket) (hs : s = silverCasket) :
  ¬(g.maker = Maker.Other ∧ s.maker = Maker.Other) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_by_cellini_son_not_both_by_cellini_not_both_by_other_l2013_201300


namespace NUMINAMATH_CALUDE_square_rotation_lateral_area_l2013_201352

/-- The lateral surface area of a cylinder formed by rotating a square around one of its sides -/
theorem square_rotation_lateral_area (side_length : ℝ) (h : side_length = 2) :
  2 * side_length * Real.pi = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_rotation_lateral_area_l2013_201352


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2013_201303

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 * (-1)^2 - 3 * m * (-1) + 1 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2013_201303


namespace NUMINAMATH_CALUDE_reduce_to_single_letter_l2013_201316

/-- Represents a circular arrangement of letters A and B -/
def CircularArrangement := List Bool

/-- Represents the operations that can be performed on the arrangement -/
inductive Operation
  | replaceABA
  | replaceBAB

/-- Applies an operation to a circular arrangement -/
def applyOperation (arr : CircularArrangement) (op : Operation) : CircularArrangement :=
  sorry

/-- Checks if the arrangement consists of only one type of letter -/
def isSingleLetter (arr : CircularArrangement) : Bool :=
  sorry

/-- Theorem stating that any initial arrangement of 41 letters can be reduced to a single letter -/
theorem reduce_to_single_letter (initial : CircularArrangement) :
  initial.length = 41 → ∃ (final : CircularArrangement), isSingleLetter final ∧ 
  ∃ (ops : List Operation), final = ops.foldl applyOperation initial :=
  sorry

end NUMINAMATH_CALUDE_reduce_to_single_letter_l2013_201316


namespace NUMINAMATH_CALUDE_distance_between_points_l2013_201338

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (-5, 3)
  let pointB : ℝ × ℝ := (6, 3)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2013_201338


namespace NUMINAMATH_CALUDE_battery_problem_l2013_201367

theorem battery_problem :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (4 * x + 18 * y + 16 * z = 2 * x + 15 * y + 24 * z) ∧
  (4 * x + 18 * y + 16 * z = 6 * x + 12 * y + 20 * z) →
  (4 * x + 18 * y + 16 * z) / z = 48 := by
sorry

end NUMINAMATH_CALUDE_battery_problem_l2013_201367


namespace NUMINAMATH_CALUDE_determine_x_value_l2013_201334

theorem determine_x_value (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end NUMINAMATH_CALUDE_determine_x_value_l2013_201334


namespace NUMINAMATH_CALUDE_triple_composition_even_l2013_201323

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2013_201323


namespace NUMINAMATH_CALUDE_power_twelve_minus_one_divisible_by_five_l2013_201336

theorem power_twelve_minus_one_divisible_by_five (a : ℤ) (h : ¬ 5 ∣ a) : 
  5 ∣ (a^12 - 1) := by
  sorry

end NUMINAMATH_CALUDE_power_twelve_minus_one_divisible_by_five_l2013_201336


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2013_201321

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Stratified
  | Systematic

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (total_size : ℕ)
  (subgroup_size : ℕ)
  (h_equal_subgroups : subgroup_size * 2 = total_size)

/-- Represents a sampling scenario -/
structure SamplingScenario :=
  (population : Population)
  (sample_size : ℕ)
  (h_sample_size_valid : sample_size ≤ population.total_size)

/-- Determines if a sampling method is appropriate for investigating subgroup differences -/
def is_appropriate_for_subgroup_investigation (method : SamplingMethod) (scenario : SamplingScenario) : Prop :=
  method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for investigating differences between equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (scenario : SamplingScenario)
  (h_equal_subgroups : scenario.population.subgroup_size * 2 = scenario.population.total_size) :
  is_appropriate_for_subgroup_investigation SamplingMethod.Stratified scenario :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2013_201321


namespace NUMINAMATH_CALUDE_slope_product_l2013_201378

/-- Given two lines with slopes m and n, where one line makes three times
    the angle with the horizontal as the other and has 3 times the slope,
    prove that mn = 9/4 -/
theorem slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m = 3 * n)
  (h4 : Real.arctan m = 3 * Real.arctan n) : m * n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_l2013_201378


namespace NUMINAMATH_CALUDE_middle_number_proof_l2013_201305

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b = 12) (h4 : a + c = 17) (h5 : b + c = 19) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2013_201305


namespace NUMINAMATH_CALUDE_inverse_function_zero_solution_l2013_201362

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = 0 is x = 2/b -/
theorem inverse_function_zero_solution
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 2 / (a * x + b)
  (f⁻¹) 0 = 2 / b :=
sorry

end NUMINAMATH_CALUDE_inverse_function_zero_solution_l2013_201362


namespace NUMINAMATH_CALUDE_bug_triangle_probability_sum_of_numerator_denominator_l2013_201347

/-- Probability of being at the starting vertex after n moves -/
def prob_at_start (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else
    let prev := prob_at_start (n - 1)
    let prev_prev := prob_at_start (n - 2)
    (prev_prev + 2 * prev) / 4

theorem bug_triangle_probability :
  prob_at_start 10 = 171 / 1024 :=
sorry

#eval Nat.gcd 171 1024  -- To verify that 171 and 1024 are coprime

theorem sum_of_numerator_denominator :
  171 + 1024 = 1195 :=
sorry

end NUMINAMATH_CALUDE_bug_triangle_probability_sum_of_numerator_denominator_l2013_201347


namespace NUMINAMATH_CALUDE_game_cost_before_tax_l2013_201388

theorem game_cost_before_tax 
  (weekly_savings : ℝ) 
  (weeks : ℕ) 
  (tax_rate : ℝ) 
  (total_saved : ℝ) 
  (h1 : weekly_savings = 5)
  (h2 : weeks = 11)
  (h3 : tax_rate = 0.1)
  (h4 : total_saved = weekly_savings * weeks)
  : ∃ (pre_tax_cost : ℝ), pre_tax_cost = 50 ∧ total_saved = pre_tax_cost * (1 + tax_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_game_cost_before_tax_l2013_201388


namespace NUMINAMATH_CALUDE_spherical_sector_volume_equals_cone_volume_l2013_201371

/-- The volume of a spherical sector is equal to the volume of specific cones -/
theorem spherical_sector_volume_equals_cone_volume (R h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) :
  let V := (2 * Real.pi * R^2 * h) / 3
  (V = (1/3) * Real.pi * R^2 * (2*h)) ∧ 
  (V = (1/3) * Real.pi * (R*Real.sqrt 2)^2 * h) :=
by
  sorry


end NUMINAMATH_CALUDE_spherical_sector_volume_equals_cone_volume_l2013_201371


namespace NUMINAMATH_CALUDE_maryann_rescue_l2013_201343

/-- The number of friends Maryann needs to rescue -/
def rescue_problem (cheap_time expensive_time total_time : ℕ) : Prop :=
  let time_per_friend := cheap_time + expensive_time
  ∃ (num_friends : ℕ), num_friends * time_per_friend = total_time

theorem maryann_rescue :
  rescue_problem 6 8 42 → ∃ (num_friends : ℕ), num_friends = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_maryann_rescue_l2013_201343


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_twenty_five_percent_less_proof_l2013_201391

theorem twenty_five_percent_less : ℝ → Prop :=
  fun x => (x + x / 4 = 80 * 3 / 4) → x = 48

-- The proof goes here
theorem twenty_five_percent_less_proof : twenty_five_percent_less 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_twenty_five_percent_less_proof_l2013_201391


namespace NUMINAMATH_CALUDE_stratified_sample_green_and_carp_l2013_201382

/-- Represents the total number of fish -/
def total_fish : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 20

/-- Represents the number of green fish -/
def green_fish : ℕ := 20

/-- Represents the number of carp -/
def carp : ℕ := 40

/-- Represents the sum of green fish and carp -/
def green_and_carp : ℕ := green_fish + carp

/-- Theorem stating the number of green fish and carp in the stratified sample -/
theorem stratified_sample_green_and_carp :
  (green_and_carp : ℚ) * sample_size / total_fish = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_green_and_carp_l2013_201382


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l2013_201308

/-- An ellipse with semi-major axis 3 and semi-minor axis 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 9) = 1}

/-- An equilateral triangle inscribed in the ellipse -/
structure InscribedTriangle where
  vertices : Fin 3 → ℝ × ℝ
  on_ellipse : ∀ i, vertices i ∈ Ellipse
  is_equilateral : ∀ i j, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  centroid_origin : (vertices 0 + vertices 1 + vertices 2) / 3 = (0, 0)

/-- The square of the area of the inscribed equilateral triangle -/
def square_area (t : InscribedTriangle) : ℝ := sorry

/-- The main theorem: The square of the area of the inscribed equilateral triangle is 507/16 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : square_area t = 507/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l2013_201308


namespace NUMINAMATH_CALUDE_mary_hourly_wage_l2013_201340

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Mary's work schedule --/
def mary_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 9
  , tue_thu_hours := 5
  , weekly_earnings := 407 }

/-- Theorem stating Mary's hourly wage is $11 --/
theorem mary_hourly_wage :
  hourly_wage mary_schedule = 11 := by sorry

end NUMINAMATH_CALUDE_mary_hourly_wage_l2013_201340


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2013_201376

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  R^2 - r^2 = 20 →
  ∃ c : ℝ, c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2013_201376


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2013_201315

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2013_201315


namespace NUMINAMATH_CALUDE_alex_born_in_1989_l2013_201394

/-- The year when the first Math Kangaroo test was held -/
def first_math_kangaroo_year : ℕ := 1991

/-- The number of the Math Kangaroo test Alex participated in -/
def alex_participation_number : ℕ := 9

/-- Alex's age when he participated in the Math Kangaroo test -/
def alex_age_at_participation : ℕ := 10

/-- Calculate the year of Alex's birth -/
def alex_birth_year : ℕ := first_math_kangaroo_year + alex_participation_number - 1 - alex_age_at_participation

theorem alex_born_in_1989 : alex_birth_year = 1989 := by
  sorry

end NUMINAMATH_CALUDE_alex_born_in_1989_l2013_201394


namespace NUMINAMATH_CALUDE_card_shop_problem_l2013_201324

/-- The total cost of cards bought from two boxes -/
def total_cost (cost1 cost2 : ℚ) (cards1 cards2 : ℕ) : ℚ :=
  cost1 * cards1 + cost2 * cards2

/-- Theorem: The total cost of 6 cards from each box is $18.00 -/
theorem card_shop_problem :
  total_cost (25/20) (35/20) 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_shop_problem_l2013_201324


namespace NUMINAMATH_CALUDE_probability_three_by_three_square_l2013_201374

/-- A square with 16 equally spaced points around its perimeter -/
structure SquareWithPoints :=
  (side_length : ℕ)
  (num_points : ℕ)

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  sorry

/-- Theorem stating the probability for a 3x3 square with 16 points -/
theorem probability_three_by_three_square :
  ∃ s : SquareWithPoints, s.side_length = 3 ∧ s.num_points = 16 ∧ 
  probability_one_unit_apart s = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_three_by_three_square_l2013_201374


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l2013_201355

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_N_complement_M :
  N ∩ (univ \ M) = Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l2013_201355


namespace NUMINAMATH_CALUDE_identity_proof_l2013_201377

theorem identity_proof (a b m n x y : ℝ) :
  (a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2) = 
  (a*n*y - a*m*x - b*m*y + b*n*x)^2 + (a*m*y + a*n*x + b*m*x - b*n*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2013_201377


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_l2013_201312

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  2 * (lg (Real.sqrt 2))^2 + lg (Real.sqrt 2) * lg 5 + 
  Real.sqrt ((lg (Real.sqrt 2))^2 - lg 2 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_one_l2013_201312


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2013_201322

def M : Set ℤ := {m : ℤ | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem complement_M_intersect_N :
  (Mᶜ : Set ℤ) ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2013_201322


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2013_201331

theorem polynomial_expansion_theorem (a k n : ℤ) : 
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) → 
  a - n + k = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2013_201331


namespace NUMINAMATH_CALUDE_smallest_number_l2013_201380

theorem smallest_number : ∀ (a b c d : ℚ), a = -2 ∧ b = 2 ∧ c = -1/2 ∧ d = 1/2 → a < b ∧ a < c ∧ a < d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2013_201380


namespace NUMINAMATH_CALUDE_solve_g_inequality_range_of_a_l2013_201384

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- Theorem for part (1)
theorem solve_g_inequality :
  ∀ x : ℝ, |g x| < 5 ↔ -2 < x ∧ x < 4 :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_solve_g_inequality_range_of_a_l2013_201384


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l2013_201319

theorem rectangular_plot_length 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) 
  (h1 : width = 40) 
  (h2 : num_poles = 52) 
  (h3 : pole_distance = 5) : 
  let perimeter := (num_poles - 1 : ℝ) * pole_distance
  let length := perimeter / 2 - width
  length = 87.5 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l2013_201319


namespace NUMINAMATH_CALUDE_part_one_part_two_l2013_201318

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part (1)
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Part (2)
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2013_201318


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_7_l2013_201360

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year < 10000

theorem first_year_after_2020_with_digit_sum_7 :
  ∃ (year : ℕ), is_valid_year year ∧ 
    year > 2020 ∧ 
    sum_of_digits year = 7 ∧
    (∀ y, is_valid_year y → y > 2020 → y < year → sum_of_digits y ≠ 7) ∧
    year = 2021 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_7_l2013_201360


namespace NUMINAMATH_CALUDE_total_groups_created_l2013_201328

def group_size : ℕ := 6
def eggs : ℕ := 18
def bananas : ℕ := 72
def marbles : ℕ := 66

theorem total_groups_created : 
  (eggs / group_size + bananas / group_size + marbles / group_size) = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_groups_created_l2013_201328


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_in_x_seq_l2013_201314

/-- A sequence is defined as x_n = 2013 + 317n, where n ≥ 0. -/
def x_seq (n : ℕ) : ℕ := 2013 + 317 * n

/-- A number is palindromic if its decimal representation reads the same forwards and backwards. -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- There are infinitely many palindromic numbers in the sequence x_n. -/
theorem infinitely_many_palindromes_in_x_seq :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ is_palindrome (x_seq n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_in_x_seq_l2013_201314


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2013_201392

theorem unique_solution_for_equation : ∃! (x y : ℤ), (x + 2)^4 - x^4 = y^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2013_201392


namespace NUMINAMATH_CALUDE_flood_damage_in_pounds_l2013_201370

def flood_damage_rupees : ℝ := 45000000
def exchange_rate : ℝ := 75

theorem flood_damage_in_pounds : 
  flood_damage_rupees / exchange_rate = 600000 := by sorry

end NUMINAMATH_CALUDE_flood_damage_in_pounds_l2013_201370


namespace NUMINAMATH_CALUDE_amount_spent_on_toys_l2013_201330

def initial_amount : ℕ := 16
def amount_left : ℕ := 8

theorem amount_spent_on_toys :
  initial_amount - amount_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_amount_spent_on_toys_l2013_201330


namespace NUMINAMATH_CALUDE_tangent_ratio_problem_l2013_201307

theorem tangent_ratio_problem (α : ℝ) (h : Real.tan (π - α) = 1/3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_problem_l2013_201307


namespace NUMINAMATH_CALUDE_marble_difference_l2013_201329

theorem marble_difference (drew_original : ℕ) (marcus_original : ℕ) : 
  (drew_original / 4 = 35) →  -- Drew gave 1/4 of his marbles, which is 35
  (drew_original * 3 / 4 = 35) →  -- Drew has 35 marbles after giving 1/4 away
  (marcus_original + 35 = 35) →  -- Marcus has 35 marbles after receiving Drew's 1/4
  (drew_original - marcus_original = 140) := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2013_201329


namespace NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l2013_201366

/-- Represents the ages of Viggo, his younger brother, and his sister -/
structure FamilyAges where
  viggo : ℕ
  brother : ℕ
  sister : ℕ

/-- Calculate the ages of the family members after a given number of years -/
def ageAfterYears (ages : FamilyAges) (years : ℕ) : FamilyAges :=
  { viggo := ages.viggo + years
  , brother := ages.brother + years
  , sister := ages.sister + years }

/-- The sum of ages of Viggo, his brother, and his sister -/
def sumOfAges (ages : FamilyAges) : ℕ :=
  ages.viggo + ages.brother + ages.sister

/-- Theorem stating the sum of ages five years from now -/
theorem sum_of_ages_in_five_years :
  ∃ (initialAges : FamilyAges),
    (initialAges.viggo = initialAges.brother + 12) ∧
    (initialAges.sister = initialAges.viggo + 5) ∧
    (initialAges.brother = 10) ∧
    (sumOfAges (ageAfterYears initialAges 5) = 74) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l2013_201366


namespace NUMINAMATH_CALUDE_tank_filling_l2013_201379

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ new_buckets : ℕ, 
    new_buckets = ⌈(original_buckets : ℚ) / capacity_ratio⌉ ∧
    new_buckets = 63 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_l2013_201379


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l2013_201357

/-- The amount of a bill given its true discount and banker's discount -/
def bill_amount (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount + true_discount

/-- Theorem stating that given a true discount of 360 and a banker's discount of 424.8, 
    the amount of the bill is 720 -/
theorem bill_amount_calculation :
  bill_amount 360 424.8 = 720 := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_calculation_l2013_201357


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2013_201353

theorem three_digit_number_problem : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 6 * n = 41 * 18 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2013_201353


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2013_201361

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →                        -- Right triangle (Pythagorean theorem)
  a / b = 3 / 4 →                          -- Ratio of legs
  r * c = a^2 →                            -- Altitude theorem for r
  s * c = b^2 →                            -- Altitude theorem for s
  r + s = c →                              -- Segments sum to hypotenuse
  r / s = 9 / 16 :=                        -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2013_201361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2013_201368

/-- For an arithmetic sequence with common difference d ≠ 0, 
    if a_3 is the geometric mean of a_2 and a_6, then a_6 / a_3 = 2 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 ^ 2 = a 2 * a 6 →
  a 6 / a 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2013_201368


namespace NUMINAMATH_CALUDE_intersection_false_necessary_not_sufficient_for_union_false_l2013_201341

theorem intersection_false_necessary_not_sufficient_for_union_false (P Q : Prop) :
  (¬(P ∨ Q) → ¬(P ∧ Q)) ∧ (∃ (P Q : Prop), ¬(P ∧ Q) ∧ (P ∨ Q)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_false_necessary_not_sufficient_for_union_false_l2013_201341


namespace NUMINAMATH_CALUDE_speed_difference_proof_l2013_201301

/-- Prove that given a distance of 8 miles, if person A travels for 40 minutes
    and person B travels for 1 hour, the difference in their average speeds is 4 mph. -/
theorem speed_difference_proof (distance : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance = 8 →
  time_A = 40 / 60 →
  time_B = 1 →
  (distance / time_A) - (distance / time_B) = 4 := by
sorry

end NUMINAMATH_CALUDE_speed_difference_proof_l2013_201301


namespace NUMINAMATH_CALUDE_museum_tickets_l2013_201342

/-- Calculates the maximum number of tickets that can be purchased given a regular price, 
    discount price, discount threshold, and budget. -/
def maxTickets (regularPrice discountPrice discountThreshold budget : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / regularPrice)
  let remainingBudget := budget - fullPriceTickets * regularPrice
  let discountTickets := remainingBudget / discountPrice
  fullPriceTickets + discountTickets

/-- Theorem stating that given the specific conditions of the problem, 
    the maximum number of tickets that can be purchased is 15. -/
theorem museum_tickets : maxTickets 11 8 10 150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_museum_tickets_l2013_201342


namespace NUMINAMATH_CALUDE_george_final_stickers_l2013_201351

/-- The number of stickers each person has --/
structure Stickers where
  bob : ℕ
  tom : ℕ
  dan : ℕ
  george : ℕ

/-- The conditions of the problem --/
def sticker_conditions (s : Stickers) : Prop :=
  s.dan = 2 * s.tom ∧
  s.tom = 3 * s.bob ∧
  s.george = 5 * s.dan ∧
  s.bob = 12

/-- The total number of stickers to be distributed --/
def extra_stickers : ℕ := 100

/-- The number of people --/
def num_people : ℕ := 4

/-- Theorem stating that George will have 505 stickers in total --/
theorem george_final_stickers (s : Stickers) 
  (h : sticker_conditions s) : 
  s.george + (s.bob + s.tom + s.dan + s.george + extra_stickers) / num_people = 505 := by
  sorry


end NUMINAMATH_CALUDE_george_final_stickers_l2013_201351


namespace NUMINAMATH_CALUDE_no_integer_root_for_any_a_l2013_201358

theorem no_integer_root_for_any_a : ∀ (a : ℤ), ¬∃ (x : ℤ), x^2 - 2023*x + 2022*a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_root_for_any_a_l2013_201358


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l2013_201333

theorem quadratic_completion_of_square :
  ∀ x : ℝ, (x^2 - 8*x + 10 = 0) ↔ ((x - 4)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l2013_201333


namespace NUMINAMATH_CALUDE_equation_solution_l2013_201373

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 6 * x + 2) / (x + 2) = 4 * x + 7 ∧ x = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2013_201373


namespace NUMINAMATH_CALUDE_relationship_theorem_l2013_201396

theorem relationship_theorem (x y z w : ℝ) :
  (x + y) / (y + z) = (z + w) / (w + x) →
  x = z ∨ x + y + w + z = 0 :=
by sorry

end NUMINAMATH_CALUDE_relationship_theorem_l2013_201396


namespace NUMINAMATH_CALUDE_problem_solution_l2013_201387

theorem problem_solution :
  (∀ n : ℕ, 2 * 8^n * 32^n = 2^17 → n = 2) ∧
  (∀ n : ℕ, ∀ x : ℝ, n > 0 → x^(2*n) = 2 → (2*x^(3*n))^2 - 3*(x^2)^(2*n) = 20) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2013_201387


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2013_201345

theorem interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 6100) 
  (h2 : time = 2) 
  (h3 : interest_difference = 61) : 
  ∃ (rate : ℝ), 
    rate = 1 ∧ 
    principal * ((1 + rate / 100) ^ time - 1) - principal * rate * time / 100 = interest_difference :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2013_201345


namespace NUMINAMATH_CALUDE_classroom_ratio_l2013_201326

theorem classroom_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 
    3 * ((num_girls : ℚ) / (num_boys + num_girls : ℚ)) / 5 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l2013_201326


namespace NUMINAMATH_CALUDE_fraction_of_fraction_l2013_201385

theorem fraction_of_fraction : 
  (2 / 5 : ℚ) * (1 / 3 : ℚ) / (3 / 4 : ℚ) = 8 / 45 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_l2013_201385


namespace NUMINAMATH_CALUDE_i_power_difference_l2013_201304

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the cyclic property of i with period 4
axiom i_cyclic (n : ℤ) : i^n = i^(n % 4)

-- Theorem to prove
theorem i_power_difference : i^37 - i^29 = 0 := by sorry

end NUMINAMATH_CALUDE_i_power_difference_l2013_201304


namespace NUMINAMATH_CALUDE_total_cars_is_180_l2013_201339

/-- The total number of cars produced over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (day1_production : ℕ) : ℕ :=
  day1_production + 2 * day1_production

/-- Theorem stating that the total number of cars produced is 180 when 60 cars were produced on the first day. -/
theorem total_cars_is_180 : total_cars 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_180_l2013_201339


namespace NUMINAMATH_CALUDE_shaded_area_is_ten_l2013_201359

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents the configuration of two adjacent squares -/
structure TwoSquares where
  small : Square
  large : Square
  adjacent : True  -- This is a placeholder for the adjacency condition

/-- Calculates the area of the shaded region formed by the diagonal of the larger square
    overlapping with the smaller square in a TwoSquares configuration -/
def shaded_area (squares : TwoSquares) : ℝ :=
  sorry

/-- Theorem stating that for a TwoSquares configuration with sides 4 and 12,
    the shaded area is 10 square units -/
theorem shaded_area_is_ten (squares : TwoSquares)
  (h1 : squares.small.side = 4)
  (h2 : squares.large.side = 12) :
  shaded_area squares = 10 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_ten_l2013_201359


namespace NUMINAMATH_CALUDE_hotdogs_served_today_l2013_201311

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_today_l2013_201311


namespace NUMINAMATH_CALUDE_point_D_coordinates_l2013_201365

def P : ℝ × ℝ := (2, -2)
def Q : ℝ × ℝ := (6, 4)

def is_on_segment (D P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q

def twice_distance (D P Q : ℝ × ℝ) : Prop :=
  ‖D - P‖ = 2 * ‖D - Q‖

theorem point_D_coordinates :
  ∃ D : ℝ × ℝ, is_on_segment D P Q ∧ twice_distance D P Q ∧ D = (3, -0.5) := by
sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l2013_201365


namespace NUMINAMATH_CALUDE_bird_families_to_asia_count_l2013_201337

/-- The number of bird families that flew away to Asia -/
def bird_families_to_asia (total_migrated : ℕ) (to_africa : ℕ) : ℕ :=
  total_migrated - to_africa

/-- Theorem stating that 80 bird families flew away to Asia -/
theorem bird_families_to_asia_count : 
  bird_families_to_asia 118 38 = 80 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_to_asia_count_l2013_201337


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l2013_201310

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 1, 1]

theorem matrix_equation_proof :
  N^3 - 3 • (N^2) + 4 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l2013_201310


namespace NUMINAMATH_CALUDE_people_per_car_l2013_201317

/-- Given 63 people and 3 cars, prove that each car will contain 21 people when evenly distributed. -/
theorem people_per_car (total_people : Nat) (num_cars : Nat) (people_per_car : Nat) : 
  total_people = 63 → num_cars = 3 → people_per_car * num_cars = total_people → people_per_car = 21 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l2013_201317


namespace NUMINAMATH_CALUDE_percent_above_sixty_percent_l2013_201327

theorem percent_above_sixty_percent (P Q : ℝ) (h : P > Q) :
  (P - 0.6 * Q) / Q * 100 = (100 * P - 60 * Q) / Q := by
  sorry

end NUMINAMATH_CALUDE_percent_above_sixty_percent_l2013_201327


namespace NUMINAMATH_CALUDE_problem_solution_l2013_201363

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {-1, 0, 1}

theorem problem_solution : (Set.univ \ M) ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2013_201363


namespace NUMINAMATH_CALUDE_cylinder_radius_comparison_l2013_201348

theorem cylinder_radius_comparison (h : ℝ) (r₁ : ℝ) (r₂ : ℝ) : 
  h > 0 → r₁ > 0 → r₂ > 0 → h = 4 → r₁ = 6 → 
  (π * r₂^2 * h = 3 * (π * r₁^2 * h)) → r₂ = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_comparison_l2013_201348


namespace NUMINAMATH_CALUDE_x_plus_y_equals_32_l2013_201393

theorem x_plus_y_equals_32 (x y : ℝ) 
  (h1 : (4 : ℝ)^x = 16^(y+1)) 
  (h2 : (27 : ℝ)^y = 9^(x-6)) : 
  x + y = 32 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_32_l2013_201393


namespace NUMINAMATH_CALUDE_triangle_BC_equation_l2013_201383

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.medianAB (t : Triangle) : Line :=
  { a := 5, b := -3, c := -3 }

def Triangle.medianAC (t : Triangle) : Line :=
  { a := 7, b := -3, c := -5 }

def Triangle.sideBC (t : Triangle) : Line :=
  { a := 2, b := -1, c := -2 }

theorem triangle_BC_equation (t : Triangle) 
  (h1 : t.A = (1, 2))
  (h2 : t.medianAB = { a := 5, b := -3, c := -3 })
  (h3 : t.medianAC = { a := 7, b := -3, c := -5 }) :
  t.sideBC = { a := 2, b := -1, c := -2 } := by
  sorry


end NUMINAMATH_CALUDE_triangle_BC_equation_l2013_201383


namespace NUMINAMATH_CALUDE_drama_club_problem_l2013_201349

theorem drama_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) (drama_only : ℕ)
  (h_total : total = 75)
  (h_math : math = 42)
  (h_physics : physics = 35)
  (h_both : both = 25)
  (h_drama_only : drama_only = 10) :
  total - ((math + physics - both) + drama_only) = 13 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_problem_l2013_201349


namespace NUMINAMATH_CALUDE_legoland_animals_l2013_201354

theorem legoland_animals (num_kangaroos : ℕ) (num_koalas : ℕ) : 
  num_kangaroos = 384 → 
  num_kangaroos = 8 * num_koalas → 
  num_kangaroos + num_koalas = 432 := by
sorry

end NUMINAMATH_CALUDE_legoland_animals_l2013_201354


namespace NUMINAMATH_CALUDE_golden_section_division_l2013_201350

/-- Given a line segment AB of length a, prove that the point H that divides AB
    such that AH = a(√5 - 1)/2 makes AH the mean proportional between AB and HB. -/
theorem golden_section_division (a : ℝ) (h : a > 0) :
  let x := a * (Real.sqrt 5 - 1) / 2
  x * x = a * (a - x) :=
by sorry

end NUMINAMATH_CALUDE_golden_section_division_l2013_201350


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2013_201364

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 5*x + 6 > 0 ∧ x ≠ 3) ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2013_201364


namespace NUMINAMATH_CALUDE_cone_base_radius_l2013_201369

/-- Given a right cone with slant height 27 cm and lateral surface forming
    a circular sector of 220° when unrolled, the radius of the base is 16.5 cm. -/
theorem cone_base_radius (s : ℝ) (θ : ℝ) (h1 : s = 27) (h2 : θ = 220 * π / 180) :
  let r := s * θ / (2 * π)
  r = 16.5 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2013_201369


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2013_201389

theorem smallest_number_divisible (n : ℕ) : n ≥ 1015 ∧ 
  (∀ m : ℕ, m < 1015 → ¬(12 ∣ (m - 7) ∧ 16 ∣ (m - 7) ∧ 18 ∣ (m - 7) ∧ 21 ∣ (m - 7) ∧ 28 ∣ (m - 7))) →
  (12 ∣ (n - 7) ∧ 16 ∣ (n - 7) ∧ 18 ∣ (n - 7) ∧ 21 ∣ (n - 7) ∧ 28 ∣ (n - 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2013_201389


namespace NUMINAMATH_CALUDE_initial_coloring_books_l2013_201372

theorem initial_coloring_books (books_removed : ℝ) (coupons_per_book : ℝ) (total_coupons : ℕ) :
  books_removed = 20 →
  coupons_per_book = 4 →
  total_coupons = 80 →
  ∃ (initial_books : ℕ), initial_books = 40 ∧ 
    (initial_books : ℝ) - books_removed = (total_coupons : ℝ) / coupons_per_book :=
by
  sorry

end NUMINAMATH_CALUDE_initial_coloring_books_l2013_201372


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2013_201356

theorem ratio_x_to_y (x y : ℚ) (h : (14*x - 5*y) / (17*x - 3*y) = 2/7) : x/y = 29/64 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2013_201356


namespace NUMINAMATH_CALUDE_age_problem_l2013_201306

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2013_201306


namespace NUMINAMATH_CALUDE_family_weights_calculation_l2013_201390

/-- Represents the weights of three generations in a family -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- Given the total weight of all three, the weight of daughter and grandchild, 
    and the relation between grandmother and grandchild weights, 
    prove the individual weights -/
theorem family_weights_calculation (w : FamilyWeights) : 
  w.grandmother + w.daughter + w.grandchild = 110 →
  w.daughter + w.grandchild = 60 →
  w.grandchild = w.grandmother / 5 →
  w.grandmother = 50 ∧ w.daughter = 50 ∧ w.grandchild = 10 := by
  sorry


end NUMINAMATH_CALUDE_family_weights_calculation_l2013_201390


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2013_201344

theorem triangle_side_difference (x : ℕ) : 
  x > 0 → x + 8 > 10 → x + 10 > 8 → 8 + 10 > x → 
  (∃ (max min : ℕ), 
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≤ max) ∧
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≥ min) ∧
    max - min = 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2013_201344


namespace NUMINAMATH_CALUDE_stock_investment_l2013_201375

theorem stock_investment (dividend_rate : ℚ) (dividend_earned : ℚ) (stock_price : ℚ) :
  dividend_rate = 9 / 100 →
  dividend_earned = 120 →
  stock_price = 135 →
  ∃ (investment : ℚ), investment = 1800 ∧ 
    dividend_earned = dividend_rate * (investment * 100 / stock_price) :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_l2013_201375


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2013_201309

theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → p + q ≠ 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p ≤ 7.275) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2013_201309


namespace NUMINAMATH_CALUDE_solution_set_inequality_for_a_b_l2013_201320

-- Define the inequality
def satisfies_inequality (x : ℝ) : Prop := abs (x + 1) + abs (x + 3) < 4

-- Theorem for the solution set
theorem solution_set :
  ∀ x : ℝ, satisfies_inequality x ↔ -4 < x ∧ x < 0 := by sorry

-- Theorem for the inequality between a and b
theorem inequality_for_a_b (a b : ℝ) 
  (ha : satisfies_inequality a) (hb : satisfies_inequality b) :
  2 * abs (a - b) < abs (a * b + 2 * a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_for_a_b_l2013_201320


namespace NUMINAMATH_CALUDE_even_count_pascal_15_rows_l2013_201346

/-- Counts the number of even entries in a single row of Pascal's Triangle -/
def countEvenInRow (n : ℕ) : ℕ := sorry

/-- Counts the total number of even entries in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the first 15 rows of Pascal's Triangle is 97 -/
theorem even_count_pascal_15_rows : countEvenInTriangle 15 = 97 := by sorry

end NUMINAMATH_CALUDE_even_count_pascal_15_rows_l2013_201346


namespace NUMINAMATH_CALUDE_collinear_points_condition_l2013_201399

/-- Given non-collinear plane vectors a and b, and points A, B, C such that
    AB = a - 2b and BC = 3a + kb, prove that A, B, and C are collinear iff k = -6 -/
theorem collinear_points_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), a = r • b) 
  (A B C : ℝ × ℝ) 
  (h_AB : B - A = a - 2 • b) 
  (h_BC : C - B = 3 • a + k • b) :
  (∃ (t : ℝ), C - A = t • (B - A)) ↔ k = -6 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_condition_l2013_201399


namespace NUMINAMATH_CALUDE_father_age_l2013_201397

/-- Represents the ages and relationships of family members -/
structure FamilyAges where
  peter : ℕ
  jane : ℕ
  harriet : ℕ
  emily : ℕ
  mother : ℕ
  aunt_lucy : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def family_conditions (f : FamilyAges) : Prop :=
  f.peter + 12 = 2 * (f.harriet + 12) ∧
  f.jane = f.emily + 10 ∧
  3 * f.peter = f.mother ∧
  f.mother = 60 ∧
  f.peter = f.jane + 5 ∧
  f.aunt_lucy = 52 ∧
  f.aunt_lucy = f.mother + 4 ∧
  f.father = f.aunt_lucy + 20

/-- The theorem to be proved -/
theorem father_age (f : FamilyAges) : 
  family_conditions f → f.father = 72 := by
  sorry

end NUMINAMATH_CALUDE_father_age_l2013_201397


namespace NUMINAMATH_CALUDE_skiing_scavenger_ratio_is_two_to_one_l2013_201302

/-- Given a total number of students and the number of students for a scavenger hunting trip,
    calculates the ratio of skiing trip students to scavenger hunting trip students. -/
def skiing_to_scavenger_ratio (total : ℕ) (scavenger : ℕ) : ℚ :=
  let skiing := total - scavenger
  (skiing : ℚ) / (scavenger : ℚ)

/-- Theorem stating that given 12000 total students and 4000 for scavenger hunting,
    the ratio of skiing to scavenger hunting students is 2:1. -/
theorem skiing_scavenger_ratio_is_two_to_one :
  skiing_to_scavenger_ratio 12000 4000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_skiing_scavenger_ratio_is_two_to_one_l2013_201302


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2013_201325

/-- The sum of the digits of (10^40 - 46) is 369. -/
theorem sum_of_digits_of_large_number : 
  (let k := 10^40 - 46
   Finset.sum (Finset.range 41) (λ i => (k / 10^i) % 10)) = 369 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2013_201325


namespace NUMINAMATH_CALUDE_perimeter_VWX_equals_5_plus_10_root_5_l2013_201335

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the right prism -/
structure Midpoints where
  v : ℝ × ℝ × ℝ
  w : ℝ × ℝ × ℝ
  x : ℝ × ℝ × ℝ

/-- Calculate the perimeter of triangle VWX in the right prism -/
def perimeterVWX (prism : RightPrism) (midpoints : Midpoints) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX -/
theorem perimeter_VWX_equals_5_plus_10_root_5 (prism : RightPrism) (midpoints : Midpoints) 
  (h1 : prism.height = 20)
  (h2 : prism.baseSideLength = 10)
  (h3 : midpoints.v = (5, 0, 0))
  (h4 : midpoints.w = (10, 5, 0))
  (h5 : midpoints.x = (5, 5, 10)) :
  perimeterVWX prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_VWX_equals_5_plus_10_root_5_l2013_201335


namespace NUMINAMATH_CALUDE_disjoint_subsets_count_l2013_201381

theorem disjoint_subsets_count (S : Finset ℕ) : 
  S = Finset.range 12 →
  (Finset.powerset S).card = 2^12 →
  let n := (3^12 - 2 * 2^12 + 1) / 2
  (n : ℕ) = 261625 ∧ n % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_count_l2013_201381
