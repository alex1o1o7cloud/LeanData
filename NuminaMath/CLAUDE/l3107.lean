import Mathlib

namespace NUMINAMATH_CALUDE_bob_questions_proof_l3107_310796

def question_rate (hour : Nat) : Nat :=
  match hour with
  | 0 => 13
  | n + 1 => 2 * question_rate n

def total_questions (hours : Nat) : Nat :=
  match hours with
  | 0 => 0
  | n + 1 => question_rate n + total_questions n

theorem bob_questions_proof :
  total_questions 3 = 91 :=
by sorry

end NUMINAMATH_CALUDE_bob_questions_proof_l3107_310796


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l3107_310731

/-- Given a paint mixture with a ratio of red:blue:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red blue white : ℚ) (h1 : red / white = 5 / 7) (h2 : white = 21) :
  red = 15 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l3107_310731


namespace NUMINAMATH_CALUDE_income_savings_theorem_l3107_310740

def income_savings_problem (income : ℝ) (savings : ℝ) : Prop :=
  let income_year2 : ℝ := income * 1.25
  let savings_year2 : ℝ := savings * 2
  let expenditure_year1 : ℝ := income - savings
  let expenditure_year2 : ℝ := income_year2 - savings_year2
  (expenditure_year1 + expenditure_year2 = 2 * expenditure_year1) ∧
  (savings / income = 0.25)

theorem income_savings_theorem (income : ℝ) (savings : ℝ) 
  (h : income > 0) : income_savings_problem income savings :=
by
  sorry

#check income_savings_theorem

end NUMINAMATH_CALUDE_income_savings_theorem_l3107_310740


namespace NUMINAMATH_CALUDE_barb_dress_savings_l3107_310722

theorem barb_dress_savings (original_price savings : ℝ) 
  (h1 : original_price = 180)
  (h2 : savings = 80)
  (h3 : original_price - savings < original_price / 2) :
  |(original_price / 2) - (original_price - savings)| = 10 :=
by sorry

end NUMINAMATH_CALUDE_barb_dress_savings_l3107_310722


namespace NUMINAMATH_CALUDE_tan_equality_solutions_l3107_310777

theorem tan_equality_solutions (n : ℤ) :
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) →
  n = -135 ∨ n = -45 ∨ n = 45 ∨ n = 135 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_solutions_l3107_310777


namespace NUMINAMATH_CALUDE_roberts_markers_count_l3107_310771

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := total_markers - initial_markers

theorem roberts_markers_count : roberts_markers = 109 := by
  sorry

end NUMINAMATH_CALUDE_roberts_markers_count_l3107_310771


namespace NUMINAMATH_CALUDE_mirror_wall_area_ratio_l3107_310749

theorem mirror_wall_area_ratio :
  let mirror_side : ℝ := 21
  let wall_width : ℝ := 28
  let wall_length : ℝ := 31.5
  let mirror_area := mirror_side ^ 2
  let wall_area := wall_width * wall_length
  mirror_area / wall_area = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_area_ratio_l3107_310749


namespace NUMINAMATH_CALUDE_balloon_radius_increase_l3107_310798

theorem balloon_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 24) (h₂ : c₂ = 30) :
  (c₂ / (2 * π)) - (c₁ / (2 * π)) = 3 / π := by sorry

end NUMINAMATH_CALUDE_balloon_radius_increase_l3107_310798


namespace NUMINAMATH_CALUDE_map_distance_to_actual_l3107_310705

/-- Given a map scale and a distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_to_actual (scale : ℚ) (map_distance : ℚ) :
  scale = 200000 →
  map_distance = 3.5 →
  (map_distance * scale) / 100000 = 7 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_l3107_310705


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3107_310729

def systematic_sampling (total_members : ℕ) (num_groups : ℕ) (group_number : ℕ) (number_in_group : ℕ) : ℕ :=
  number_in_group - (group_number - 1) * (total_members / num_groups)

theorem systematic_sampling_theorem (total_members num_groups group_5 group_3 : ℕ) 
  (h1 : total_members = 200)
  (h2 : num_groups = 40)
  (h3 : group_5 = 5)
  (h4 : group_3 = 3)
  (h5 : systematic_sampling total_members num_groups group_5 22 = 22) :
  systematic_sampling total_members num_groups group_3 22 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3107_310729


namespace NUMINAMATH_CALUDE_smallest_number_above_threshold_l3107_310743

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let filtered := numbers.filter (λ x => x ≥ threshold)
  filtered.minimum? = some 1.2 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_above_threshold_l3107_310743


namespace NUMINAMATH_CALUDE_edwards_summer_earnings_l3107_310737

/-- Given Edward's lawn mowing business earnings and expenses, prove the amount he made in the summer. -/
theorem edwards_summer_earnings (spring_earnings : ℕ) (supplies_cost : ℕ) (final_amount : ℕ) :
  spring_earnings = 2 →
  supplies_cost = 5 →
  final_amount = 24 →
  ∃ summer_earnings : ℕ, spring_earnings + summer_earnings - supplies_cost = final_amount ∧ summer_earnings = 27 :=
by sorry

end NUMINAMATH_CALUDE_edwards_summer_earnings_l3107_310737


namespace NUMINAMATH_CALUDE_sum_seven_to_ten_l3107_310748

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 2
  sum_third_fourth : a 3 + a 4 = 4

/-- The sum of the 7th to 10th terms of the geometric sequence is 48 -/
theorem sum_seven_to_ten (seq : GeometricSequence) :
  seq.a 7 + seq.a 8 + seq.a 9 + seq.a 10 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_to_ten_l3107_310748


namespace NUMINAMATH_CALUDE_boys_left_to_girl_l3107_310723

/-- Represents a group of children standing in a circle. -/
structure CircleGroup where
  boys : ℕ
  girls : ℕ
  boys_right_to_girl : ℕ

/-- The main theorem to be proved. -/
theorem boys_left_to_girl (group : CircleGroup) 
  (h1 : group.boys = 40)
  (h2 : group.girls = 28)
  (h3 : group.boys_right_to_girl = 18) :
  group.boys - (group.boys - group.boys_right_to_girl) = 18 := by
  sorry

#check boys_left_to_girl

end NUMINAMATH_CALUDE_boys_left_to_girl_l3107_310723


namespace NUMINAMATH_CALUDE_fixed_point_and_min_product_l3107_310707

/-- The line l passing through a fixed point P -/
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (2 + 2*m)*y - 8 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-4, 6)

/-- Line l₁ -/
def line_l1 (x : ℝ) : Prop := x = -1

/-- Line l₂ -/
def line_l2 (y : ℝ) : Prop := y = -1

/-- Theorem stating that P is the fixed point and the minimum value of |PM| · |PN| -/
theorem fixed_point_and_min_product :
  (∀ m : ℝ, line_l m (point_P.1) (point_P.2)) ∧
  (∃ min : ℝ, min = 42 ∧
    ∀ m : ℝ, ∀ M N : ℝ × ℝ,
      line_l m M.1 M.2 → line_l1 M.1 →
      line_l m N.1 N.2 → line_l2 N.2 →
      (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
      (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ≥ min^2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_min_product_l3107_310707


namespace NUMINAMATH_CALUDE_smallest_possible_value_l3107_310779

theorem smallest_possible_value (a b x : ℕ) : 
  a > 0 → b > 0 → x > 0 → a = 72 → 
  Nat.gcd a b = x + 6 → 
  Nat.lcm a b = 2 * x * (x + 6) → 
  b ≥ 24 ∧ (∃ (y : ℕ), y > 0 ∧ y + 6 ∣ 72 ∧ 
    Nat.gcd 72 24 = y + 6 ∧ 
    Nat.lcm 72 24 = 2 * y * (y + 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l3107_310779


namespace NUMINAMATH_CALUDE_product_of_fractions_l3107_310759

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions :
  (fraction 7) * (fraction 8) * (fraction 9) * (fraction 10) * (fraction 11) = 133 / 946 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3107_310759


namespace NUMINAMATH_CALUDE_diophantine_equation_properties_l3107_310732

theorem diophantine_equation_properties :
  (∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 365) ∧
  (∀ (n : ℕ+), n > 370 → ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = n) ∧
  (¬ ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 370) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_properties_l3107_310732


namespace NUMINAMATH_CALUDE_not_perfect_power_l3107_310770

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ¬ ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 10^k - 1 = m^n := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_power_l3107_310770


namespace NUMINAMATH_CALUDE_log_arithmetic_progression_implies_power_relation_l3107_310734

theorem log_arithmetic_progression_implies_power_relation
  (k m n x : ℝ)
  (hk : k > 0)
  (hm : m > 0)
  (hn : n > 0)
  (hx_pos : x > 0)
  (hx_neq_one : x ≠ 1)
  (h_arith_prog : 2 * (Real.log x / Real.log m) = 
                  (Real.log x / Real.log k) + (Real.log x / Real.log n)) :
  n^2 = (n*k)^(Real.log m / Real.log k) :=
by sorry

end NUMINAMATH_CALUDE_log_arithmetic_progression_implies_power_relation_l3107_310734


namespace NUMINAMATH_CALUDE_fifth_road_length_l3107_310724

/-- Represents a road network with four cities and five roads -/
structure RoadNetwork where
  road1 : ℕ
  road2 : ℕ
  road3 : ℕ
  road4 : ℕ
  road5 : ℕ

/-- The given road network satisfies the triangle inequality -/
def satisfiesTriangleInequality (rn : RoadNetwork) : Prop :=
  rn.road5 < rn.road1 + rn.road2 ∧
  rn.road5 + rn.road3 > rn.road4

/-- Theorem: Given the specific road lengths, the fifth road must be 17 km long -/
theorem fifth_road_length (rn : RoadNetwork) 
  (h1 : rn.road1 = 10)
  (h2 : rn.road2 = 8)
  (h3 : rn.road3 = 5)
  (h4 : rn.road4 = 21)
  (h5 : satisfiesTriangleInequality rn) :
  rn.road5 = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifth_road_length_l3107_310724


namespace NUMINAMATH_CALUDE_chord_length_at_specific_angle_shortest_chord_equation_l3107_310785

-- Define the circle
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 8}

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define a chord AB passing through P0
def chord (α : ℝ) : Set (ℝ × ℝ) := {p | p.2 - P0.2 = Real.tan α * (p.1 - P0.1)}

-- Define the length of a chord
def chordLength (α : ℝ) : ℝ := sorry

-- Theorem 1
theorem chord_length_at_specific_angle :
  chordLength (3 * Real.pi / 4) = Real.sqrt 30 := by sorry

-- Define the shortest chord
def shortestChord : Set (ℝ × ℝ) := sorry

-- Theorem 2
theorem shortest_chord_equation :
  shortestChord = {p | p.1 - 2 * p.2 + 5 = 0} := by sorry

end NUMINAMATH_CALUDE_chord_length_at_specific_angle_shortest_chord_equation_l3107_310785


namespace NUMINAMATH_CALUDE_eighth_row_middle_number_l3107_310792

/-- The number on the far right of the nth row -/
def rightNumber (n : ℕ) : ℕ := n^2

/-- The number of elements in the nth row -/
def rowElements (n : ℕ) : ℕ := 2*n - 1

/-- The middle number in a row -/
def middleNumber (n : ℕ) : ℕ := rightNumber (n-1) + (rowElements n + 1) / 2

theorem eighth_row_middle_number : middleNumber 8 = 57 := by sorry

end NUMINAMATH_CALUDE_eighth_row_middle_number_l3107_310792


namespace NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l3107_310745

theorem no_solutions_lcm_gcd_equation :
  ¬∃ (n : ℕ), n > 0 ∧ Nat.lcm n 200 = Nat.gcd n 200 + 1000 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l3107_310745


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3107_310772

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x ≥ 0}

-- Theorem statement
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Ioc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3107_310772


namespace NUMINAMATH_CALUDE_ratio_and_linear_equation_l3107_310762

theorem ratio_and_linear_equation (x y : ℚ) : 
  x / y = 4 → x = 18 - 3 * y → y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_and_linear_equation_l3107_310762


namespace NUMINAMATH_CALUDE_twenty_percent_equals_fiftyfour_l3107_310712

theorem twenty_percent_equals_fiftyfour (x : ℝ) : (20 / 100) * x = 54 → x = 270 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_equals_fiftyfour_l3107_310712


namespace NUMINAMATH_CALUDE_xy_yz_zx_geq_3_l3107_310747

theorem xy_yz_zx_geq_3 (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x + y + z = 1/x + 1/y + 1/z) : x*y + y*z + z*x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_yz_zx_geq_3_l3107_310747


namespace NUMINAMATH_CALUDE_unique_zero_point_g_l3107_310778

open Real

theorem unique_zero_point_g (f g : ℝ → ℝ) :
  (∀ x, f x = exp x * sin x - x) →
  (∀ x, 0 < x → x < π/2 → g x = (2 - 2*x - f x) / exp x) →
  ∃! x, 0 < x ∧ x < π/2 ∧ g x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_point_g_l3107_310778


namespace NUMINAMATH_CALUDE_garden_flowers_l3107_310730

theorem garden_flowers (red_flowers : ℕ) (additional_red : ℕ) (white_flowers : ℕ) :
  red_flowers = 347 →
  additional_red = 208 →
  white_flowers = red_flowers + additional_red →
  white_flowers = 555 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_l3107_310730


namespace NUMINAMATH_CALUDE_angle_E_measure_l3107_310780

/-- Given a quadrilateral EFGH with angle relationships, prove the measure of angle E. -/
theorem angle_E_measure (E F G H : ℝ) : 
  E + F + G + H = 360 →  -- Sum of angles in a quadrilateral
  E = 5 * H →            -- Relationship between E and H
  E = 4 * G →            -- Relationship between E and G
  E = (5/3) * F →        -- Relationship between E and F
  E = 1440/11 := by
sorry

end NUMINAMATH_CALUDE_angle_E_measure_l3107_310780


namespace NUMINAMATH_CALUDE_cone_volume_l3107_310751

theorem cone_volume (cylinder_volume : ℝ) (h : cylinder_volume = 30) :
  let cone_volume := cylinder_volume / 3
  cone_volume = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3107_310751


namespace NUMINAMATH_CALUDE_sequence_bound_l3107_310764

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a n)^2 ≤ a n - a (n + 1)

theorem sequence_bound (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ n : ℕ, a n < 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l3107_310764


namespace NUMINAMATH_CALUDE_inequality_proof_l3107_310715

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b + c) / (a + c) > b / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3107_310715


namespace NUMINAMATH_CALUDE_min_score_theorem_l3107_310769

/-- Represents the normal distribution parameters -/
structure NormalParams where
  μ : ℝ
  σ : ℝ

/-- Represents the problem parameters -/
structure ProblemParams where
  total_students : ℕ
  top_rank : ℕ
  normal_params : NormalParams

/-- The probability of being within one standard deviation of the mean -/
def prob_within_one_std : ℝ := 0.6827

/-- The probability of being within two standard deviations of the mean -/
def prob_within_two_std : ℝ := 0.9545

/-- Calculates the minimum score to be in the top rank -/
def min_score_for_top_rank (params : ProblemParams) : ℝ :=
  params.normal_params.μ + 2 * params.normal_params.σ

/-- Theorem stating the minimum score to be in the top 9100 out of 400,000 students -/
theorem min_score_theorem (params : ProblemParams)
  (h1 : params.total_students = 400000)
  (h2 : params.top_rank = 9100)
  (h3 : params.normal_params.μ = 98)
  (h4 : params.normal_params.σ = 10) :
  min_score_for_top_rank params = 118 := by
  sorry

#eval min_score_for_top_rank { total_students := 400000, top_rank := 9100, normal_params := { μ := 98, σ := 10 } }

end NUMINAMATH_CALUDE_min_score_theorem_l3107_310769


namespace NUMINAMATH_CALUDE_boys_camp_total_l3107_310782

theorem boys_camp_total (total : ℝ) 
  (school_a_percentage : total * 0.2 = total * (20 / 100))
  (science_percentage : (total * 0.2) * 0.3 = (total * 0.2) * (30 / 100))
  (non_science_count : (total * 0.2) * 0.7 = 49) : 
  total = 350 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3107_310782


namespace NUMINAMATH_CALUDE_total_distance_calculation_l3107_310761

/-- Calculates the total distance traveled given the distances and number of trips for each mode of transportation -/
def total_distance (plane_distance : Float) (train_distance : Float) (bus_distance : Float)
                   (plane_trips : Nat) (train_trips : Nat) (bus_trips : Nat) : Float :=
  plane_distance * plane_trips.toFloat +
  train_distance * train_trips.toFloat +
  bus_distance * bus_trips.toFloat

/-- Theorem stating that the total distance traveled is 11598.4 miles -/
theorem total_distance_calculation :
  total_distance 256.0 120.5 35.2 32 16 42 = 11598.4 := by
  sorry

#eval total_distance 256.0 120.5 35.2 32 16 42

end NUMINAMATH_CALUDE_total_distance_calculation_l3107_310761


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l3107_310794

/-- Given points P, Q, and R in a coordinate plane, prove that the value of m 
    that minimizes the sum of distances PR + QR is 7/2, under specific conditions. -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) (x m : ℝ) : 
  P = (7, 7) →
  Q = (3, 2) →
  R = (x, m) →
  ((-7 : ℝ), 7) ∈ {(x, y) | y = 3*x - 4} →
  (∀ m' : ℝ, 
    Real.sqrt ((7 - x)^2 + (7 - m')^2) + Real.sqrt ((3 - x)^2 + (2 - m')^2) ≥ 
    Real.sqrt ((7 - x)^2 + (7 - m)^2) + Real.sqrt ((3 - x)^2 + (2 - m)^2)) →
  m = 7/2 := by
sorry

end NUMINAMATH_CALUDE_minimize_distance_sum_l3107_310794


namespace NUMINAMATH_CALUDE_select_staff_eq_36_l3107_310703

/-- The number of ways to select staff for an event -/
def select_staff : ℕ :=
  let n_volunteers : ℕ := 5
  let n_translators : ℕ := 2
  let n_guides : ℕ := 2
  let n_flexible : ℕ := 1
  let n_abc : ℕ := 3  -- number of volunteers named A, B, or C

  -- Definition: Ways to choose at least one from A, B, C for translators and guides
  let ways_abc : ℕ := n_abc.choose n_translators

  -- Definition: Ways to arrange remaining volunteers
  let ways_arrange : ℕ := (n_volunteers - n_translators - n_guides).factorial

  ways_abc * ways_arrange

/-- Theorem: The number of ways to select staff is 36 -/
theorem select_staff_eq_36 : select_staff = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_staff_eq_36_l3107_310703


namespace NUMINAMATH_CALUDE_janes_stick_length_l3107_310711

/-- Given information about Pat's stick and its relationship to Sarah's and Jane's sticks,
    prove that Jane's stick is 22 inches long. -/
theorem janes_stick_length :
  -- Pat's stick length
  ∀ (pat_stick : ℕ),
  -- Covered portion of Pat's stick
  ∀ (covered_portion : ℕ),
  -- Conversion factor from feet to inches
  ∀ (feet_to_inches : ℕ),
  -- Conditions
  pat_stick = 30 →
  covered_portion = 7 →
  feet_to_inches = 12 →
  -- Sarah's stick is twice as long as the uncovered portion of Pat's stick
  ∃ (sarah_stick : ℕ), sarah_stick = 2 * (pat_stick - covered_portion) →
  -- Jane's stick is two feet shorter than Sarah's stick
  ∃ (jane_stick : ℕ), jane_stick = sarah_stick - 2 * feet_to_inches →
  -- Conclusion: Jane's stick is 22 inches long
  jane_stick = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_stick_length_l3107_310711


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_lines_l3107_310784

/-- Given two lines l₁ and l₂ in the form of linear equations,
    prove that the x-intercept of l₁ is 2 when l₁ is perpendicular to l₂ -/
theorem x_intercept_of_perpendicular_lines
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ (a + 3) * x + y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + (a - 1) * y + 4 = 0)
  (h_perp : (a + 3) * 1 + (a - 1) * 1 = 0) :
  ∃ x, l₁ x 0 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_lines_l3107_310784


namespace NUMINAMATH_CALUDE_expression_equality_implies_square_l3107_310726

theorem expression_equality_implies_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) :
  ∃ n : ℕ, x * y + 4 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_implies_square_l3107_310726


namespace NUMINAMATH_CALUDE_race_time_a_l3107_310750

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- B runs 960 meters in the time A finishes
  960 = b.speed * a.time ∧
  -- A finishes 8 seconds before B
  b.time = a.time + 8 ∧
  -- A and B have different speeds
  a.speed ≠ b.speed

/-- The theorem stating A's finishing time -/
theorem race_time_a (a b : Runner) (h : Race a b) : a.time = 200 :=
  sorry

#check race_time_a

end NUMINAMATH_CALUDE_race_time_a_l3107_310750


namespace NUMINAMATH_CALUDE_second_player_wins_l3107_310709

/-- A game where players take turns removing coins from a pile. -/
structure CoinGame where
  coins : ℕ              -- Number of coins in the pile
  max_take : ℕ           -- Maximum number of coins a player can take in one turn
  min_take : ℕ           -- Minimum number of coins a player can take in one turn

/-- Represents a player in the game. -/
inductive Player
| First
| Second

/-- Defines a winning strategy for a player. -/
def has_winning_strategy (game : CoinGame) (player : Player) : Prop :=
  ∃ (strategy : ℕ → ℕ), 
    (∀ n, game.min_take ≤ strategy n ∧ strategy n ≤ game.max_take) ∧
    (player = Player.First → strategy game.coins = game.coins) ∧
    (player = Player.Second → 
      ∀ first_move, game.min_take ≤ first_move ∧ first_move ≤ game.max_take →
        strategy (game.coins - first_move) = game.coins - first_move)

/-- The main theorem stating that the second player has a winning strategy in the specific game. -/
theorem second_player_wins :
  let game : CoinGame := { coins := 2016, max_take := 3, min_take := 1 }
  has_winning_strategy game Player.Second :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l3107_310709


namespace NUMINAMATH_CALUDE_jerome_money_problem_l3107_310701

theorem jerome_money_problem (certain_amount : ℕ) : 
  (2 * certain_amount - (8 + 3 * 8) = 54) → 
  certain_amount = 43 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_problem_l3107_310701


namespace NUMINAMATH_CALUDE_marbles_distribution_l3107_310700

theorem marbles_distribution (total_marbles : ℕ) (marble_loving_boys : ℕ) 
  (h1 : total_marbles = 26) (h2 : marble_loving_boys = 13) :
  total_marbles / marble_loving_boys = 2 := by
sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3107_310700


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3107_310787

theorem fraction_sum_equality : (3 : ℚ) / 30 + 9 / 300 + 27 / 3000 = 0.139 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3107_310787


namespace NUMINAMATH_CALUDE_lending_interest_rate_lending_rate_is_six_percent_l3107_310741

/-- Calculates the lending interest rate given the borrowing details and yearly gain -/
theorem lending_interest_rate 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (duration : ℝ) 
  (yearly_gain : ℝ) : ℝ :=
let borrowed_interest := borrowed_amount * borrowing_rate * duration / 100
let total_gain := yearly_gain * duration
let lending_rate := (total_gain + borrowed_interest) * 100 / (borrowed_amount * duration)
lending_rate

/-- The lending interest rate is 6% given the specified conditions -/
theorem lending_rate_is_six_percent : 
  lending_interest_rate 5000 4 2 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_lending_rate_is_six_percent_l3107_310741


namespace NUMINAMATH_CALUDE_license_plate_increase_l3107_310781

-- Define the number of possible letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters and digits in old and new license plates
def old_num_letters : ℕ := 2
def old_num_digits : ℕ := 3
def new_num_letters : ℕ := 2
def new_num_digits : ℕ := 4

-- Calculate the number of possible old and new license plates
def num_old_plates : ℕ := num_letters^old_num_letters * num_digits^old_num_digits
def num_new_plates : ℕ := num_letters^new_num_letters * num_digits^new_num_digits

-- Theorem: The ratio of new to old license plates is 10
theorem license_plate_increase : num_new_plates / num_old_plates = 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3107_310781


namespace NUMINAMATH_CALUDE_f_increasing_implies_f_1_ge_25_l3107_310773

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_increasing_implies_f_1_ge_25 (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_f_1_ge_25_l3107_310773


namespace NUMINAMATH_CALUDE_largest_k_for_right_triangle_inequality_l3107_310756

theorem largest_k_for_right_triangle_inequality :
  ∃ (k : ℝ), k = (3 * Real.sqrt 2 - 4) / 2 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → a^2 + b^2 = c^2 →
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3) ∧
  (∀ (k' : ℝ), k' > k →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
      a^3 + b^3 + c^3 < k' * (a + b + c)^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_right_triangle_inequality_l3107_310756


namespace NUMINAMATH_CALUDE_jellybean_difference_l3107_310765

theorem jellybean_difference (gigi_jellybeans rory_jellybeans lorelai_jellybeans : ℕ) : 
  gigi_jellybeans = 15 →
  rory_jellybeans > gigi_jellybeans →
  lorelai_jellybeans = 3 * (rory_jellybeans + gigi_jellybeans) →
  lorelai_jellybeans = 180 →
  rory_jellybeans - gigi_jellybeans = 30 := by
sorry

end NUMINAMATH_CALUDE_jellybean_difference_l3107_310765


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l3107_310738

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x² -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit left and 2 units down -/
def translation : Translation :=
  { dx := -1, dy := -2 }

/-- The resulting parabola after translation -/
def translated_parabola (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  let p := original_parabola
  let t := translation
  let result := translated_parabola p t
  result.a = 1 ∧ result.b = 2 ∧ result.c = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l3107_310738


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l3107_310719

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l3107_310719


namespace NUMINAMATH_CALUDE_no_savings_from_radio_offer_l3107_310733

-- Define the in-store price
def in_store_price : ℚ := 139.99

-- Define the radio offer components
def radio_payment : ℚ := 33.00
def num_payments : ℕ := 4
def shipping_charge : ℚ := 11.99

-- Calculate the total radio offer price
def radio_offer_price : ℚ := radio_payment * num_payments + shipping_charge

-- Define the conversion factor from dollars to cents
def dollars_to_cents : ℕ := 100

-- Theorem statement
theorem no_savings_from_radio_offer : 
  (radio_offer_price - in_store_price) * dollars_to_cents = 0 := by sorry

end NUMINAMATH_CALUDE_no_savings_from_radio_offer_l3107_310733


namespace NUMINAMATH_CALUDE_burger_cost_is_five_l3107_310776

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℝ := 4

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

/-- The total cost of the order in dollars -/
def total_cost : ℝ := 17

/-- The cost of the burger in dollars -/
def burger_cost : ℝ := total_cost - (sandwich_cost + num_smoothies * smoothie_cost)

theorem burger_cost_is_five :
  burger_cost = 5 := by sorry

end NUMINAMATH_CALUDE_burger_cost_is_five_l3107_310776


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3107_310744

theorem cubic_equation_solutions :
  ∃ (x₁ x₂ x₃ : ℂ),
    (x₁^3 + 4*x₁^2*(Real.sqrt 3) + 12*x₁ + 8*(Real.sqrt 3)) + (x₁ + 2*(Real.sqrt 3)) = 0 ∧
    (x₂^3 + 4*x₂^2*(Real.sqrt 3) + 12*x₂ + 8*(Real.sqrt 3)) + (x₂ + 2*(Real.sqrt 3)) = 0 ∧
    (x₃^3 + 4*x₃^2*(Real.sqrt 3) + 12*x₃ + 8*(Real.sqrt 3)) + (x₃ + 2*(Real.sqrt 3)) = 0 ∧
    x₁ = -2*(Real.sqrt 3) ∧
    x₂ = -2*(Real.sqrt 3) + Complex.I ∧
    x₃ = -2*(Real.sqrt 3) - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3107_310744


namespace NUMINAMATH_CALUDE_negation_of_implication_l3107_310708

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∀ x : ℝ, x ≤ 1 → x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3107_310708


namespace NUMINAMATH_CALUDE_function_inequality_l3107_310713

/-- Given a function f(x) = 2^((a-x)^2) where a is a real number,
    if f(1) > f(3) and f(2) > f(3), then |a-1| > |a-2|. -/
theorem function_inequality (a : ℝ) :
  let f : ℝ → ℝ := λ x => 2^((a-x)^2)
  (f 1 > f 3) → (f 2 > f 3) → |a-1| > |a-2| := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3107_310713


namespace NUMINAMATH_CALUDE_vasya_clock_problem_l3107_310739

theorem vasya_clock_problem :
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ b ≤ 59 ∧ (100 * a + b) % (60 * a + b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_vasya_clock_problem_l3107_310739


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l3107_310725

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / 8000) + (b_investment / 8000) + (c_investment / 8000)
  let profit_per_part := c_profit / (c_investment / 8000)
  ratio_sum * profit_per_part

/-- Theorem stating that given the specific investments and C's profit, the total profit is 92000 -/
theorem partnership_profit_calculation :
  calculate_total_profit 24000 32000 36000 36000 = 92000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l3107_310725


namespace NUMINAMATH_CALUDE_students_taking_history_or_statistics_l3107_310799

theorem students_taking_history_or_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_not_statistics : ℕ) : 
  total = 90 → history = 36 → statistics = 32 → history_not_statistics = 25 →
  ∃ (both : ℕ), history - both = history_not_statistics ∧ history + statistics - both = 57 := by
sorry

end NUMINAMATH_CALUDE_students_taking_history_or_statistics_l3107_310799


namespace NUMINAMATH_CALUDE_race_head_start_l3107_310775

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  L / Va = (L - H) / Vb := by sorry

end NUMINAMATH_CALUDE_race_head_start_l3107_310775


namespace NUMINAMATH_CALUDE_composition_result_l3107_310757

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^2

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x / 2
noncomputable def g_inv (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem composition_result :
  f (g_inv (f_inv (f_inv (g (f 8))))) = 16 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l3107_310757


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3107_310727

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3107_310727


namespace NUMINAMATH_CALUDE_roots_location_l3107_310788

theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x₁ x₂ : ℝ), 
    (a < x₁ ∧ x₁ < b) ∧ 
    (b < x₂ ∧ x₂ < c) ∧ 
    (∀ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_roots_location_l3107_310788


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3107_310789

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3107_310789


namespace NUMINAMATH_CALUDE_no_solution_exists_l3107_310767

theorem no_solution_exists : ¬ ∃ (x : ℕ), 
  (18 + x = 2 * (5 + x)) ∧ 
  (18 + x = 3 * (2 + x)) ∧ 
  ((18 + x) + (5 + x) + (2 + x) = 50) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3107_310767


namespace NUMINAMATH_CALUDE_area_inside_circle_outside_rectangle_l3107_310763

/-- The area inside a circle but outside a rectangle with shared center --/
theorem area_inside_circle_outside_rectangle (π : Real) :
  let circle_radius : Real := 1 / 3
  let rectangle_length : Real := 3
  let rectangle_width : Real := 1.5
  let circle_area : Real := π * circle_radius ^ 2
  let rectangle_area : Real := rectangle_length * rectangle_width
  let rectangle_diagonal : Real := (rectangle_length ^ 2 + rectangle_width ^ 2).sqrt
  circle_radius < rectangle_diagonal / 2 →
  circle_area = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_circle_outside_rectangle_l3107_310763


namespace NUMINAMATH_CALUDE_sum_of_absolute_ratios_l3107_310710

theorem sum_of_absolute_ratios (x y z : ℚ) 
  (sum_zero : x + y + z = 0) 
  (product_nonzero : x * y * z ≠ 0) : 
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = 1) ∨
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = -1) :=
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_ratios_l3107_310710


namespace NUMINAMATH_CALUDE_walters_age_l3107_310721

theorem walters_age (walter_age_1994 : ℕ) (mother_age_1994 : ℕ) : 
  walter_age_1994 = mother_age_1994 / 3 →
  (1994 - walter_age_1994) + (1994 - mother_age_1994) = 3900 →
  walter_age_1994 + 10 = 32 :=
by sorry

end NUMINAMATH_CALUDE_walters_age_l3107_310721


namespace NUMINAMATH_CALUDE_annual_forest_gathering_handshakes_count_l3107_310742

/-- The number of handshakes at the Annual Forest Gathering -/
def annual_forest_gathering_handshakes (num_goblins num_elves : ℕ) : ℕ :=
  (num_goblins.choose 2) + (num_goblins * num_elves)

/-- Theorem stating the number of handshakes at the Annual Forest Gathering -/
theorem annual_forest_gathering_handshakes_count :
  annual_forest_gathering_handshakes 25 18 = 750 := by
  sorry

end NUMINAMATH_CALUDE_annual_forest_gathering_handshakes_count_l3107_310742


namespace NUMINAMATH_CALUDE_min_sum_squares_l3107_310774

/-- B-neighborhood of A is defined as the solution set of |x - A| < B where A ∈ ℝ and B > 0 -/
def neighborhood (A : ℝ) (B : ℝ) : Set ℝ :=
  {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) :
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 →
  ∃ (m : ℝ), m = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3107_310774


namespace NUMINAMATH_CALUDE_correct_num_friends_l3107_310768

/-- The number of friends Jeremie is going with to the amusement park. -/
def num_friends : ℕ := 3

/-- The cost of a ticket in dollars. -/
def ticket_cost : ℕ := 18

/-- The cost of a snack set in dollars. -/
def snack_cost : ℕ := 5

/-- The total cost for Jeremie and her friends in dollars. -/
def total_cost : ℕ := 92

/-- Theorem stating that the number of friends Jeremie is going with is correct. -/
theorem correct_num_friends :
  (num_friends + 1) * (ticket_cost + snack_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_num_friends_l3107_310768


namespace NUMINAMATH_CALUDE_equation_solution_l3107_310783

theorem equation_solution : ∃ x : ℝ, (3 / (x + 2) = 2 / x) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3107_310783


namespace NUMINAMATH_CALUDE_problem_statement_l3107_310753

theorem problem_statement (x : ℝ) (h : Real.exp (x * Real.log 9) + Real.exp (x * Real.log 3) = 6) :
  Real.exp ((1 / x) * Real.log 16) + Real.exp ((1 / x) * Real.log 4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3107_310753


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l3107_310714

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l3107_310714


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l3107_310754

theorem pizza_payment_difference :
  -- Define the total number of slices
  let total_slices : ℕ := 12
  -- Define the cost of the plain pizza
  let plain_cost : ℚ := 12
  -- Define the additional cost for mushrooms
  let mushroom_cost : ℚ := 3
  -- Define the number of slices with mushrooms (one-third of the pizza)
  let mushroom_slices : ℕ := total_slices / 3
  -- Define the number of slices Laura ate
  let laura_slices : ℕ := mushroom_slices + 2
  -- Define the number of slices Jessica ate
  let jessica_slices : ℕ := total_slices - laura_slices
  -- Calculate the total cost of the pizza
  let total_cost : ℚ := plain_cost + mushroom_cost
  -- Calculate the cost per slice
  let cost_per_slice : ℚ := total_cost / total_slices
  -- Calculate Laura's payment
  let laura_payment : ℚ := laura_slices * cost_per_slice
  -- Calculate Jessica's payment (only plain slices)
  let jessica_payment : ℚ := jessica_slices * (plain_cost / total_slices)
  -- The difference in payment
  laura_payment - jessica_payment = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l3107_310754


namespace NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l3107_310717

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, 7 * |x - 4*a| + |x - a^2| + 6*x - 3*a ≠ 0) ↔ a < -17 ∨ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l3107_310717


namespace NUMINAMATH_CALUDE_monkey_doll_price_difference_is_two_l3107_310706

def monkey_doll_price_difference (total_spending : ℕ) (large_doll_price : ℕ) (extra_small_dolls : ℕ) : ℕ :=
  let large_dolls := total_spending / large_doll_price
  let small_dolls := large_dolls + extra_small_dolls
  let small_doll_price := total_spending / small_dolls
  large_doll_price - small_doll_price

theorem monkey_doll_price_difference_is_two :
  monkey_doll_price_difference 300 6 25 = 2 := by sorry

end NUMINAMATH_CALUDE_monkey_doll_price_difference_is_two_l3107_310706


namespace NUMINAMATH_CALUDE_total_cards_l3107_310716

def sallys_cards (initial : ℕ) (dans_gift : ℕ) (purchased : ℕ) : ℕ :=
  initial + dans_gift + purchased

theorem total_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3107_310716


namespace NUMINAMATH_CALUDE_power_multiplication_l3107_310736

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3107_310736


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l3107_310735

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (distance_per_interval : ℝ) (interval_duration : ℝ) 
  (initial_ride_duration : ℝ) (break_duration : ℝ) (final_ride_duration : ℝ) 
  (h1 : distance_per_interval = 1.5)
  (h2 : interval_duration = 7)
  (h3 : initial_ride_duration = 21)
  (h4 : break_duration = 5)
  (h5 : final_ride_duration = 14) :
  (initial_ride_duration / interval_duration) * distance_per_interval + 
  (final_ride_duration / interval_duration) * distance_per_interval = 7.5 := by
  sorry

#check suzanna_bike_ride

end NUMINAMATH_CALUDE_suzanna_bike_ride_l3107_310735


namespace NUMINAMATH_CALUDE_number_wall_solution_l3107_310766

/-- Represents a number wall with four levels -/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle_left : ℕ)
  (bottom_middle_right : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value of the top block in the number wall -/
def top_block (wall : NumberWall) : ℕ :=
  wall.bottom_left + wall.bottom_middle_left + wall.bottom_middle_right + wall.bottom_right + 30

/-- Theorem: In a number wall where the top block is 42, and the bottom row contains m, 5, 3, and 6 from left to right, the value of m is 12 -/
theorem number_wall_solution (wall : NumberWall) 
  (h1 : wall.bottom_middle_left = 5)
  (h2 : wall.bottom_middle_right = 3)
  (h3 : wall.bottom_right = 6)
  (h4 : top_block wall = 42) : 
  wall.bottom_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l3107_310766


namespace NUMINAMATH_CALUDE_matthews_water_bottle_size_l3107_310702

/-- Calculates the size of Matthew's water bottle based on his drinking habits -/
theorem matthews_water_bottle_size 
  (glasses_per_day : ℕ) 
  (ounces_per_glass : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : ounces_per_glass = 5)
  (h3 : fills_per_week = 4) :
  (glasses_per_day * ounces_per_glass * 7) / fills_per_week = 35 := by
  sorry

end NUMINAMATH_CALUDE_matthews_water_bottle_size_l3107_310702


namespace NUMINAMATH_CALUDE_inequality_proof_l3107_310755

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  ((1 + a * b) / (1 + a)) ^ 2008 + 
  ((1 + b * c) / (1 + b)) ^ 2008 + 
  ((1 + c * d) / (1 + c)) ^ 2008 + 
  ((1 + d * a) / (1 + d)) ^ 2008 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3107_310755


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_221_l3107_310791

theorem largest_gcd_of_sum_221 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) :
  Nat.gcd a b ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_221_l3107_310791


namespace NUMINAMATH_CALUDE_large_bulb_cost_l3107_310718

def prove_large_bulb_cost (small_bulbs : ℕ) (large_bulbs : ℕ) (initial_amount : ℕ) (small_bulb_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  small_bulbs = 3 →
  large_bulbs = 1 →
  initial_amount = 60 →
  small_bulb_cost = 8 →
  remaining_amount = 24 →
  (initial_amount - remaining_amount - small_bulbs * small_bulb_cost) / large_bulbs = 12

theorem large_bulb_cost : prove_large_bulb_cost 3 1 60 8 24 := by
  sorry

end NUMINAMATH_CALUDE_large_bulb_cost_l3107_310718


namespace NUMINAMATH_CALUDE_g_is_max_g_symmetric_points_l3107_310797

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := 
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1/(2*a)
  else Real.sqrt 2

theorem g_is_max (a : ℝ) : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ g a := by sorry

theorem g_symmetric_points (a : ℝ) : 
  ((-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1) ↔ g a = g (1/a) := by sorry

end NUMINAMATH_CALUDE_g_is_max_g_symmetric_points_l3107_310797


namespace NUMINAMATH_CALUDE_circle_equations_l3107_310720

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ α, x α = 2 + 2 * Real.cos α
  h_y : ∀ α, y α = 2 * Real.sin α

/-- The Cartesian equation of the circle -/
def cartesian_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ

theorem circle_equations (c : Circle) :
  (∀ x y, cartesian_equation c x y ↔ ∃ α, c.x α = x ∧ c.y α = y) ∧
  (∀ ρ θ, polar_equation ρ θ ↔ cartesian_equation c (ρ * Real.cos θ) (ρ * Real.sin θ)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equations_l3107_310720


namespace NUMINAMATH_CALUDE_problem_statement_l3107_310758

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (a - c > 2 * b) ∧ (a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3107_310758


namespace NUMINAMATH_CALUDE_only_expr3_not_factorable_l3107_310752

-- Define the expressions
def expr1 (a b : ℝ) := a^2 - b^2
def expr2 (x y z : ℝ) := 49*x^2 - y^2*z^2
def expr3 (x y : ℝ) := -x^2 - y^2
def expr4 (m n p : ℝ) := 16*m^2*n^2 - 25*p^2

-- Define the difference of squares formula
def diff_of_squares (a b : ℝ) := (a + b) * (a - b)

-- Theorem statement
theorem only_expr3_not_factorable :
  (∃ (a b : ℝ), expr1 a b = diff_of_squares a b) ∧
  (∃ (x y z : ℝ), expr2 x y z = diff_of_squares (7*x) (y*z)) ∧
  (∀ (x y : ℝ), ¬∃ (a b : ℝ), expr3 x y = diff_of_squares a b) ∧
  (∃ (m n p : ℝ), expr4 m n p = diff_of_squares (4*m*n) (5*p)) :=
sorry

end NUMINAMATH_CALUDE_only_expr3_not_factorable_l3107_310752


namespace NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_12_l3107_310704

theorem sum_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_12_l3107_310704


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3107_310793

/-- Given a right circular cylinder with radius 2 intersected by a plane forming an ellipse,
    if the major axis of the ellipse is 25% longer than the minor axis,
    then the length of the major axis is 5. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3107_310793


namespace NUMINAMATH_CALUDE_marble_leftover_l3107_310746

theorem marble_leftover (n m k : ℤ) : (7*n + 2 + 7*m + 5 + 7*k + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_leftover_l3107_310746


namespace NUMINAMATH_CALUDE_rational_abs_eq_neg_l3107_310760

theorem rational_abs_eq_neg (a : ℚ) (h : |a| = -a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_abs_eq_neg_l3107_310760


namespace NUMINAMATH_CALUDE_colorings_theorem_l3107_310795

/-- The number of ways to color five cells in a 5x5 grid with one colored cell in each row and column. -/
def total_colorings : ℕ := 120

/-- The number of ways to color five cells in a 5x5 grid without one corner cell, 
    with one colored cell in each row and column. -/
def colorings_without_one_corner : ℕ := 96

/-- The number of ways to color five cells in a 5x5 grid without two corner cells, 
    with one colored cell in each row and column. -/
def colorings_without_two_corners : ℕ := 78

theorem colorings_theorem : 
  colorings_without_two_corners = total_colorings - 2 * (total_colorings - colorings_without_one_corner) + 6 :=
by sorry

end NUMINAMATH_CALUDE_colorings_theorem_l3107_310795


namespace NUMINAMATH_CALUDE_airplane_passengers_l3107_310786

theorem airplane_passengers (total : ℕ) (children : ℕ) (h1 : total = 80) (h2 : children = 20) :
  let adults := total - children
  let men := adults / 2
  men = 30 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l3107_310786


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_1000_l3107_310790

def first_n_odd (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * i + 1)
def first_n_even (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * (i + 1))

theorem sum_difference_even_odd_1000 : 
  (first_n_even 1000).sum - (first_n_odd 1000).sum = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_1000_l3107_310790


namespace NUMINAMATH_CALUDE_candy_cost_per_pack_l3107_310728

theorem candy_cost_per_pack (number_of_packs : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_packs = 3 →
  amount_paid = 20 →
  change_received = 11 →
  (amount_paid - change_received) / number_of_packs = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_cost_per_pack_l3107_310728
