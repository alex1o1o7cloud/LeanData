import Mathlib

namespace NUMINAMATH_CALUDE_probability_theorem_l906_90677

/-- The number of roots of unity for z^1997 - 1 = 0 --/
def n : ℕ := 1997

/-- The set of complex roots of z^1997 - 1 = 0 --/
def roots : Set ℂ := {z : ℂ | z^n = 1}

/-- The condition that needs to be satisfied --/
def condition (v w : ℂ) : Prop := Real.sqrt (2 + Real.sqrt 3) ≤ Complex.abs (v + w)

/-- The number of pairs (v, w) satisfying the condition --/
def satisfying_pairs : ℕ := 332 * (n - 1)

/-- The total number of possible pairs (v, w) --/
def total_pairs : ℕ := n * (n - 1)

/-- The theorem to be proved --/
theorem probability_theorem :
  (satisfying_pairs : ℚ) / total_pairs = 83 / 499 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l906_90677


namespace NUMINAMATH_CALUDE_propositions_truth_l906_90678

theorem propositions_truth :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (1 / x₀) > -x₀ + 1) ∧
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > x₀ - 1) ∧
  (¬ ∀ x : ℝ, x > 0 → (1/2)^x > Real.log x / Real.log (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l906_90678


namespace NUMINAMATH_CALUDE_divisibility_by_72_l906_90665

theorem divisibility_by_72 (n : ℕ) : 
  ∃ d : ℕ, d < 10 ∧ 32235717 * 10 + d = n * 72 :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_72_l906_90665


namespace NUMINAMATH_CALUDE_banana_arrangements_l906_90618

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let repeated_letter1 : ℕ := 3  -- 'a' appears 3 times
  let repeated_letter2 : ℕ := 2  -- 'n' appears 2 times
  factorial total_letters / (factorial repeated_letter1 * factorial repeated_letter2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l906_90618


namespace NUMINAMATH_CALUDE_second_grade_selection_theorem_l906_90640

/-- Represents the school population --/
structure School :=
  (total_students : ℕ)
  (first_grade_male_prob : ℝ)

/-- Represents the sampling method --/
structure Sampling :=
  (total_volunteers : ℕ)
  (method : String)

/-- Calculates the number of students selected from the second grade --/
def second_grade_selection (s : School) (samp : Sampling) : ℕ :=
  sorry

theorem second_grade_selection_theorem (s : School) (samp : Sampling) :
  s.total_students = 4000 →
  s.first_grade_male_prob = 0.2 →
  samp.total_volunteers = 100 →
  samp.method = "stratified" →
  second_grade_selection s samp = 30 :=
sorry

end NUMINAMATH_CALUDE_second_grade_selection_theorem_l906_90640


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l906_90697

/-- Given a line passing through points (2, -3) and (5, 6), prove that m + b = -6 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b ↔ (x = 2 ∧ y = -3) ∨ (x = 5 ∧ y = 6)) →
  m + b = -6 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l906_90697


namespace NUMINAMATH_CALUDE_f_one_lower_bound_l906_90637

/-- Given a quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞),
    prove that f(1) ≥ 25. -/
theorem f_one_lower_bound
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = 4 * x^2 - m * x + 5)
  (h2 : ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f x < f y) :
  f 1 ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_f_one_lower_bound_l906_90637


namespace NUMINAMATH_CALUDE_cricketer_specific_average_l906_90600

/-- Represents the average score calculation for a cricketer's matches -/
def cricketer_average_score (total_matches : ℕ) (first_set_matches : ℕ) (second_set_matches : ℕ) 
  (first_set_average : ℚ) (second_set_average : ℚ) : ℚ :=
  ((first_set_matches : ℚ) * first_set_average + (second_set_matches : ℚ) * second_set_average) / (total_matches : ℚ)

/-- Theorem stating the average score calculation for a specific cricketer's performance -/
theorem cricketer_specific_average : 
  cricketer_average_score 25 10 15 60 70 = 66 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_specific_average_l906_90600


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l906_90666

/-- Given a rhombus with one diagonal of length 30 meters and an area of 600 square meters,
    prove that the length of the other diagonal is 40 meters. -/
theorem rhombus_other_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 30)
    (h₂ : area = 600)
    (h₃ : area = d₁ * d₂ / 2) : 
  d₂ = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l906_90666


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l906_90605

-- Define the edge lengths
def edge_length_1 : ℚ := 5
def edge_length_2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes
def volume_1 : ℚ := edge_length_1^3
def volume_2 : ℚ := edge_length_2^3

-- Theorem statement
theorem cube_volume_ratio :
  volume_1 / volume_2 = 125 / 13824 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l906_90605


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l906_90606

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determines if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Determines if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- The main theorem stating that three collinear points out of four
    is sufficient but not necessary for four points to be coplanar -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  ∃ (p q r s : Point3D),
    (collinear p q r → coplanar p q r s) ∧
    (coplanar p q r s ∧ ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s)) := by
  sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l906_90606


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_radius_l906_90690

/-- The radius of a circle that is tangent to the parabola y = 1/4 * x^2 at a point where the tangent line to the parabola is also tangent to the circle. -/
theorem parabola_circle_tangent_radius : ∃ (r : ℝ) (P : ℝ × ℝ),
  r > 0 ∧
  (P.2 = (1/4) * P.1^2) ∧
  ((P.1 - 1)^2 + (P.2 - 2)^2 = r^2) ∧
  (∃ (m : ℝ), (∀ (x y : ℝ), y - P.2 = m * (x - P.1) → 
    y = (1/4) * x^2 ∨ (x - 1)^2 + (y - 2)^2 = r^2)) →
  r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_radius_l906_90690


namespace NUMINAMATH_CALUDE_exists_irrational_greater_than_neg_three_l906_90642

theorem exists_irrational_greater_than_neg_three :
  ∃ x : ℝ, Irrational x ∧ x > -3 := by sorry

end NUMINAMATH_CALUDE_exists_irrational_greater_than_neg_three_l906_90642


namespace NUMINAMATH_CALUDE_years_since_same_average_l906_90615

/-- Represents a club with members and their ages -/
structure Club where
  members : Nat
  avgAge : ℝ

/-- Represents the replacement of a member in the club -/
structure Replacement where
  oldMemberAge : ℝ
  newMemberAge : ℝ

/-- Theorem: The number of years since the average age was the same
    is equal to the age difference between the replaced and new member -/
theorem years_since_same_average (c : Club) (r : Replacement) :
  c.members = 5 →
  r.oldMemberAge - r.newMemberAge = 15 →
  c.avgAge * c.members = (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge) →
  (r.oldMemberAge - r.newMemberAge : ℝ) = (c.avgAge * c.members - (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge)) / c.members :=
by
  sorry


end NUMINAMATH_CALUDE_years_since_same_average_l906_90615


namespace NUMINAMATH_CALUDE_student_scores_l906_90680

theorem student_scores (M P C : ℕ) : 
  M + P = 60 →
  C = P + 10 →
  (M + C) / 2 = 35 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l906_90680


namespace NUMINAMATH_CALUDE_local_min_condition_l906_90627

/-- The function f(x) = x^3 - 3bx + b has a local minimum in the interval (0, 1) -/
def has_local_min_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f x ≤ f y

/-- The main theorem -/
theorem local_min_condition (b : ℝ) :
  has_local_min_in_interval (fun x => x^3 - 3*b*x + b) 0 1 → b ∈ Set.Ioo 0 1 := by
  sorry


end NUMINAMATH_CALUDE_local_min_condition_l906_90627


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l906_90616

/-- The number of operations for Horner's method on a polynomial of degree n -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f (x : ℝ) : ℝ := 3*x^6 + 4*x^5 + 5*x^4 + 6*x^3 + 7*x^2 + 8*x + 1

/-- The degree of the polynomial f -/
def degree_f : ℕ := 6

theorem horner_operations_for_f :
  horner_operations degree_f = 12 :=
sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l906_90616


namespace NUMINAMATH_CALUDE_number_of_math_classes_school_play_volunteers_l906_90647

/-- Given information about volunteers for a school Christmas play, prove the number of participating math classes. -/
theorem number_of_math_classes (total_needed : ℕ) (students_per_class : ℕ) (teachers : ℕ) (more_needed : ℕ) : ℕ :=
  let current_volunteers := total_needed - more_needed
  let x := (current_volunteers - teachers) / students_per_class
  x

/-- Prove that the number of math classes participating is 6. -/
theorem school_play_volunteers : number_of_math_classes 50 5 13 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_math_classes_school_play_volunteers_l906_90647


namespace NUMINAMATH_CALUDE_roots_satisfy_conditions_l906_90655

theorem roots_satisfy_conditions : ∃ (x y : ℝ),
  x + y = 10 ∧
  |x - y| = 12 ∧
  x^2 - 10*x - 22 = 0 ∧
  y^2 - 10*y - 22 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_satisfy_conditions_l906_90655


namespace NUMINAMATH_CALUDE_standard_deviation_constant_addition_original_sd_equals_new_sd_l906_90609

/-- The standard deviation of a list of real numbers -/
noncomputable def standardDeviation (l : List ℝ) : ℝ := sorry

/-- Adding a constant to each element in a list -/
def addConstant (l : List ℝ) (c : ℝ) : List ℝ := sorry

theorem standard_deviation_constant_addition 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = standardDeviation (addConstant original c) :=
sorry

theorem original_sd_equals_new_sd 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = 2 → 
  standardDeviation (addConstant original c) = 2 :=
sorry

end NUMINAMATH_CALUDE_standard_deviation_constant_addition_original_sd_equals_new_sd_l906_90609


namespace NUMINAMATH_CALUDE_negative_one_less_than_negative_two_thirds_l906_90696

theorem negative_one_less_than_negative_two_thirds : -1 < -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_less_than_negative_two_thirds_l906_90696


namespace NUMINAMATH_CALUDE_simplify_expression_l906_90661

theorem simplify_expression (x : ℝ) : (4*x)^4 + (5*x)*(x^3) = 261*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l906_90661


namespace NUMINAMATH_CALUDE_correct_attitude_towards_superstitions_l906_90604

/-- Represents different types of online superstitions -/
inductive OnlineSuperstition
  | AstrologicalFate
  | HoroscopeInterpretation
  | NorthStarBook
  | DreamInterpretation

/-- Represents possible attitudes towards online superstitions -/
inductive Attitude
  | Accept
  | StayAway
  | RespectDiversity
  | ImproveDiscernment

/-- Defines the correct attitude for teenage students -/
def correct_attitude : Attitude := Attitude.ImproveDiscernment

/-- Theorem stating the correct attitude towards online superstitions -/
theorem correct_attitude_towards_superstitions :
  ∀ (s : OnlineSuperstition), correct_attitude = Attitude.ImproveDiscernment :=
by sorry

end NUMINAMATH_CALUDE_correct_attitude_towards_superstitions_l906_90604


namespace NUMINAMATH_CALUDE_square_condition_l906_90644

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

theorem square_condition (n : ℕ) :
  n > 0 → (is_perfect_square ((n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_square_condition_l906_90644


namespace NUMINAMATH_CALUDE_min_value_theorem_l906_90634

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 4) (h2 : x > y) (h3 : y > 0) :
  (∀ a b : ℝ, a + b = 4 → a > b → b > 0 → (2 / (a - b) + 1 / b) ≥ 2) ∧ 
  (∃ a b : ℝ, a + b = 4 ∧ a > b ∧ b > 0 ∧ 2 / (a - b) + 1 / b = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l906_90634


namespace NUMINAMATH_CALUDE_largest_dividend_l906_90672

theorem largest_dividend (dividend quotient divisor remainder : ℕ) : 
  dividend = quotient * divisor + remainder →
  remainder < divisor →
  quotient = 32 →
  divisor = 18 →
  dividend ≤ 593 := by
sorry

end NUMINAMATH_CALUDE_largest_dividend_l906_90672


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l906_90670

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents an investigation with its sample size and population size -/
structure Investigation where
  sampleSize : Nat
  populationSize : Nat

/-- The company's sales outlet data -/
def companyData : List Region :=
  [⟨"A", 150⟩, ⟨"B", 120⟩, ⟨"C", 180⟩, ⟨"D", 150⟩]

/-- Total number of outlets -/
def totalOutlets : Nat := (companyData.map Region.outlets).sum

/-- Investigation ① -/
def investigation1 : Investigation :=
  ⟨100, totalOutlets⟩

/-- Investigation ② -/
def investigation2 : Investigation :=
  ⟨7, 10⟩

/-- Determines the appropriate sampling method for an investigation -/
def appropriateSamplingMethod (i : Investigation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling ∧
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l906_90670


namespace NUMINAMATH_CALUDE_log_12_5_value_l906_90676

-- Define the given conditions
axiom a : ℝ
axiom b : ℝ
axiom lg_2_eq_a : Real.log 2 = a
axiom ten_pow_b_eq_3 : (10 : ℝ)^b = 3

-- State the theorem to be proved
theorem log_12_5_value : Real.log 5 / Real.log 12 = (1 - a) / (2 * a + b) := by sorry

end NUMINAMATH_CALUDE_log_12_5_value_l906_90676


namespace NUMINAMATH_CALUDE_max_product_l906_90659

theorem max_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_l906_90659


namespace NUMINAMATH_CALUDE_fraction_inequality_l906_90635

theorem fraction_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hab : a < b) (hcd : c < d) : 
  (a + c) / (b + c) < (a + d) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l906_90635


namespace NUMINAMATH_CALUDE_smaller_root_comparison_l906_90685

theorem smaller_root_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b / a > b' / a') :=
sorry

end NUMINAMATH_CALUDE_smaller_root_comparison_l906_90685


namespace NUMINAMATH_CALUDE_hilt_water_fountain_trips_l906_90689

/-- The number of times Mrs. Hilt will go to the water fountain -/
def water_fountain_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / (2 * distance_to_fountain)

/-- Theorem: Mrs. Hilt will go to the water fountain 2 times -/
theorem hilt_water_fountain_trips :
  water_fountain_trips 30 120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hilt_water_fountain_trips_l906_90689


namespace NUMINAMATH_CALUDE_vector_problem_l906_90682

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Parallel vectors in R^2 -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v 0 * w 1 = t * v 1 * w 0

theorem vector_problem :
  (∃ k : ℝ, parallel (fun i => a i + k * c i) (fun i => 2 * b i + c i) → k = -11/18) ∧
  (∃ m n : ℝ, (∀ i, a i = m * b i - n * c i) → m = 5/9 ∧ n = -8/9) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l906_90682


namespace NUMINAMATH_CALUDE_multiple_of_nine_three_odd_l906_90620

theorem multiple_of_nine_three_odd (n : ℕ) :
  (∀ m : ℕ, 9 ∣ m → 3 ∣ m) →
  (Odd n ∧ 9 ∣ n) →
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_three_odd_l906_90620


namespace NUMINAMATH_CALUDE_equation_solution_l906_90664

theorem equation_solution (x : ℚ) (h1 : x ≠ 0) (h2 : x ≠ -5) :
  (2 * x / (x + 5) - 1 = (x + 5) / x) ↔ (x = -5/3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l906_90664


namespace NUMINAMATH_CALUDE_rectangular_field_area_l906_90624

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- length of the uncovered side
  (fence_length : ℝ) -- total length of fencing for three sides
  (h1 : L = 25) -- the uncovered side is 25 feet
  (h2 : fence_length = 95.4) -- the total fencing required is 95.4 feet
  : L * ((fence_length - L) / 2) = 880 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l906_90624


namespace NUMINAMATH_CALUDE_old_bridge_traffic_l906_90625

/-- Represents the number of vehicles passing through the old bridge every month -/
def old_bridge_monthly_traffic : ℕ := sorry

/-- Represents the number of vehicles passing through the new bridge every month -/
def new_bridge_monthly_traffic : ℕ := sorry

/-- The new bridge has twice the capacity of the old one -/
axiom new_bridge_capacity : new_bridge_monthly_traffic = 2 * old_bridge_monthly_traffic

/-- The number of vehicles passing through the new bridge increased by 60% compared to the old bridge -/
axiom traffic_increase : new_bridge_monthly_traffic = old_bridge_monthly_traffic + (60 * old_bridge_monthly_traffic) / 100

/-- The total number of vehicles passing through both bridges in a year is 62,400 -/
axiom total_yearly_traffic : 12 * (old_bridge_monthly_traffic + new_bridge_monthly_traffic) = 62400

theorem old_bridge_traffic : old_bridge_monthly_traffic = 2000 :=
sorry

end NUMINAMATH_CALUDE_old_bridge_traffic_l906_90625


namespace NUMINAMATH_CALUDE_sphere_volume_larger_than_cube_l906_90633

/-- Given a sphere and a cube with equal surface areas, the volume of the sphere is larger than the volume of the cube. -/
theorem sphere_volume_larger_than_cube (r : ℝ) (s : ℝ) (h : 4 * Real.pi * r^2 = 6 * s^2) :
  (4/3) * Real.pi * r^3 > s^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_larger_than_cube_l906_90633


namespace NUMINAMATH_CALUDE_min_box_height_l906_90658

/-- The minimum height of a box with a square base, where the height is 5 units more
    than the side length of the base, and the surface area is at least 120 square units. -/
theorem min_box_height (x : ℝ) (h1 : x > 0) : 
  let height := x + 5
  let surface_area := 2 * x^2 + 4 * x * height
  surface_area ≥ 120 → height ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l906_90658


namespace NUMINAMATH_CALUDE_limit_alternating_log_infinity_l906_90630

/-- The limit of (-1)^n * log(n) as n approaches infinity is infinity. -/
theorem limit_alternating_log_infinity :
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |(-1:ℝ)^n * Real.log n| > M :=
sorry

end NUMINAMATH_CALUDE_limit_alternating_log_infinity_l906_90630


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l906_90641

-- Define the circle Ω
def Ω : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 - 2*x - 4*y = 0}

-- Define the points that the circle passes through
def origin : ℝ × ℝ := (0, 0)
def point1 : ℝ × ℝ := (2, 4)
def point2 : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem max_distance_on_circle :
  origin ∈ Ω ∧ point1 ∈ Ω ∧ point2 ∈ Ω →
  ∃ (max_dist : ℝ),
    (∀ p ∈ Ω, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ max_dist) ∧
    (∃ q ∈ Ω, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = max_dist) ∧
    max_dist = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l906_90641


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l906_90651

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ + x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l906_90651


namespace NUMINAMATH_CALUDE_alexander_shopping_cost_l906_90607

/-- Calculates the total cost of Alexander's shopping trip -/
def shopping_cost (apple_count : ℕ) (apple_price : ℕ) (orange_count : ℕ) (orange_price : ℕ) : ℕ :=
  apple_count * apple_price + orange_count * orange_price

/-- Theorem: Alexander spends $9 on his shopping trip -/
theorem alexander_shopping_cost :
  shopping_cost 5 1 2 2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_alexander_shopping_cost_l906_90607


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l906_90626

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-19, -7; 10, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![85/14, -109/14; -3, 4]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l906_90626


namespace NUMINAMATH_CALUDE_triangle_inequality_l906_90614

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_inequality (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c))  -- Given condition
  (h2 : t.C > Real.pi / 2)          -- Angle C is obtuse
  : t.a < 2 * t.b ∧ 2 * t.b < t.c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l906_90614


namespace NUMINAMATH_CALUDE_problem_solution_l906_90629

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem problem_solution :
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  (∀ a : ℝ, A ∩ C a = ∅ ↔ a ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l906_90629


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l906_90653

/-- The number of people in the group --/
def group_size : ℕ := 5

/-- The number of arrangements where two specific people are adjacent --/
def adjacent_arrangements : ℕ := 2 * (group_size - 1).factorial

/-- The number of arrangements where three specific people are adjacent --/
def triple_adjacent_arrangements : ℕ := 2 * (group_size - 2).factorial

/-- The number of valid arrangements --/
def valid_arrangements : ℕ := adjacent_arrangements - triple_adjacent_arrangements

theorem photo_arrangements_count : valid_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l906_90653


namespace NUMINAMATH_CALUDE_ahmed_has_thirteen_goats_l906_90648

/-- The number of goats Adam has -/
def adam_goats : ℕ := 7

/-- The number of goats Andrew has -/
def andrew_goats : ℕ := 5 + 2 * adam_goats

/-- The number of goats Ahmed has -/
def ahmed_goats : ℕ := andrew_goats - 6

/-- Theorem stating that Ahmed has 13 goats -/
theorem ahmed_has_thirteen_goats : ahmed_goats = 13 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_has_thirteen_goats_l906_90648


namespace NUMINAMATH_CALUDE_quadrilateral_area_l906_90671

/-- The area of a quadrilateral with vertices A(1, 3), B(1, 1), C(5, 6), and D(4, 3) is 8.5 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (4, 3)
  let area := abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2 +
              abs (A.1 * (C.2 - D.2) + C.1 * (D.2 - A.2) + D.1 * (A.2 - C.2)) / 2
  area = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l906_90671


namespace NUMINAMATH_CALUDE_at_most_one_right_or_obtuse_angle_l906_90656

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  -- Sum of angles in a triangle is 180 degrees
  sum_180 : angle1 + angle2 + angle3 = 180

-- Theorem: At most one angle in a triangle is greater than or equal to 90 degrees
theorem at_most_one_right_or_obtuse_angle (t : Triangle) :
  (t.angle1 ≥ 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 ≥ 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 ≥ 90) :=
by
  sorry


end NUMINAMATH_CALUDE_at_most_one_right_or_obtuse_angle_l906_90656


namespace NUMINAMATH_CALUDE_function_values_l906_90617

/-- A function from ℝ² to ℝ² defined by f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem stating that if f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem function_values (k b : ℝ) : f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l906_90617


namespace NUMINAMATH_CALUDE_min_value_theorem_l906_90660

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 25 / x ≥ 10 ∧ (x + 25 / x = 10 ↔ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l906_90660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l906_90603

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence property
  a 1 = 0 →                        -- first term is 0
  d ≠ 0 →                          -- common difference is non-zero
  a k = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) →  -- sum condition
  k = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l906_90603


namespace NUMINAMATH_CALUDE_guest_to_master_bedroom_ratio_l906_90699

/-- Proves that the ratio of the guest bedroom's size to the master bedroom suite's size is 1:4 -/
theorem guest_to_master_bedroom_ratio :
  -- Total house size
  ∀ (total_size : ℝ),
  -- Size of living room, dining room, and kitchen combined
  ∀ (common_area_size : ℝ),
  -- Size of master bedroom suite
  ∀ (master_suite_size : ℝ),
  -- Conditions from the problem
  total_size = 2300 →
  common_area_size = 1000 →
  master_suite_size = 1040 →
  -- The ratio of guest bedroom size to master bedroom suite size is 1:4
  (total_size - common_area_size - master_suite_size) / master_suite_size = 1 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_guest_to_master_bedroom_ratio_l906_90699


namespace NUMINAMATH_CALUDE_shampoo_duration_l906_90663

-- Define the amount of rose shampoo Janet has
def rose_shampoo : ℚ := 1/3

-- Define the amount of jasmine shampoo Janet has
def jasmine_shampoo : ℚ := 1/4

-- Define the amount of shampoo Janet uses per day
def daily_usage : ℚ := 1/12

-- Theorem statement
theorem shampoo_duration :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_duration_l906_90663


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l906_90694

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l906_90694


namespace NUMINAMATH_CALUDE_butterfly_ratio_l906_90675

/-- Prove that the ratio of blue butterflies to yellow butterflies is 2:1 -/
theorem butterfly_ratio (total : ℕ) (black : ℕ) (blue : ℕ) 
  (h1 : total = 11)
  (h2 : black = 5)
  (h3 : blue = 4)
  : (blue : ℚ) / (total - black - blue) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_ratio_l906_90675


namespace NUMINAMATH_CALUDE_min_cost_rose_garden_l906_90652

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : Float
  tulip : Float
  orchid : Float
  rose : Float
  peony : Float

/-- Represents the dimensions of each region in the flower bed -/
structure FlowerBedRegions where
  bottom_left : Nat × Nat
  top_left : Nat × Nat
  bottom_right : Nat × Nat
  middle_right : Nat × Nat
  top_right : Nat × Nat

/-- Calculates the minimum cost for Rose's garden -/
def calculateMinCost (costs : FlowerCost) (regions : FlowerBedRegions) : Float :=
  sorry

/-- Theorem stating that the minimum cost for Rose's garden is $173.75 -/
theorem min_cost_rose_garden (costs : FlowerCost) (regions : FlowerBedRegions) :
  costs.sunflower = 0.75 ∧
  costs.tulip = 1.25 ∧
  costs.orchid = 1.75 ∧
  costs.rose = 2 ∧
  costs.peony = 2.5 ∧
  regions.bottom_left = (7, 2) ∧
  regions.top_left = (5, 5) ∧
  regions.bottom_right = (6, 4) ∧
  regions.middle_right = (8, 3) ∧
  regions.top_right = (8, 3) →
  calculateMinCost costs regions = 173.75 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_rose_garden_l906_90652


namespace NUMINAMATH_CALUDE_fraction_equality_l906_90643

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a + c = 2*b) 
  (h2 : 2*b*d = c*(b + d)) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) : 
  a / b = c / d := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l906_90643


namespace NUMINAMATH_CALUDE_squared_ratios_sum_ge_sum_l906_90646

theorem squared_ratios_sum_ge_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c := by sorry

end NUMINAMATH_CALUDE_squared_ratios_sum_ge_sum_l906_90646


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l906_90654

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l906_90654


namespace NUMINAMATH_CALUDE_stick_cutting_l906_90623

theorem stick_cutting (short_length long_length : ℝ) : 
  long_length = short_length + 18 →
  short_length + long_length = 30 →
  long_length / short_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_stick_cutting_l906_90623


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l906_90638

theorem bottle_capacity_proof (num_boxes : ℕ) (bottles_per_box : ℕ) (fill_ratio : ℚ) (total_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_ratio = 3/4 →
  total_volume = 4500 →
  (num_boxes * bottles_per_box * fill_ratio * (12 : ℚ) = total_volume) := by
  sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l906_90638


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l906_90693

theorem workshop_salary_problem (total_workers : ℕ) (all_avg_salary : ℚ) 
  (num_technicians : ℕ) (tech_avg_salary : ℚ) :
  total_workers = 21 →
  all_avg_salary = 8000 →
  num_technicians = 7 →
  tech_avg_salary = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := all_avg_salary * total_workers
  let tech_total_salary := tech_avg_salary * num_technicians
  let remaining_total_salary := total_salary - tech_total_salary
  let remaining_avg_salary := remaining_total_salary / remaining_workers
  remaining_avg_salary = 6000 := by
sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l906_90693


namespace NUMINAMATH_CALUDE_quadratic_roots_fourth_power_sum_l906_90639

/-- For a quadratic equation x² - 2ax - 1/a² = 0 with roots x₁ and x₂,
    prove that x₁⁴ + x₂⁴ = 16 + 8√2 if and only if a = ± ∛∛(1/8) -/
theorem quadratic_roots_fourth_power_sum (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*a*x₁ - 1/a^2 = 0 → 
  x₂^2 - 2*a*x₂ - 1/a^2 = 0 → 
  (x₁^4 + x₂^4 = 16 + 8*Real.sqrt 2) ↔ 
  (a = Real.rpow (1/8) (1/8) ∨ a = -Real.rpow (1/8) (1/8)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_fourth_power_sum_l906_90639


namespace NUMINAMATH_CALUDE_cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l906_90601

-- Part 1
theorem cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six :
  (8 : ℝ) ^ (1/3) - (Real.sqrt 12 * Real.sqrt 6) / Real.sqrt 3 = 2 - 2 * Real.sqrt 6 := by sorry

-- Part 2
theorem square_minus_product_equals_five_plus_two_sqrt_three :
  (Real.sqrt 3 + 1)^2 - (2 * Real.sqrt 2 + 3) * (2 * Real.sqrt 2 - 3) = 5 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l906_90601


namespace NUMINAMATH_CALUDE_merchandise_profit_rate_l906_90692

/-- Given a merchandise with cost price x, prove that the profit rate is 5% -/
theorem merchandise_profit_rate (x : ℝ) (h : 1.1 * x - 10 = 210) : 
  (210 - x) / x * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_merchandise_profit_rate_l906_90692


namespace NUMINAMATH_CALUDE_ladder_slide_l906_90687

theorem ladder_slide (initial_length initial_base_distance slip_distance : ℝ) 
  (h1 : initial_length = 30)
  (h2 : initial_base_distance = 6)
  (h3 : slip_distance = 5) :
  let initial_height := Real.sqrt (initial_length ^ 2 - initial_base_distance ^ 2)
  let new_height := initial_height - slip_distance
  let new_base_distance := Real.sqrt (initial_length ^ 2 - new_height ^ 2)
  new_base_distance - initial_base_distance = Real.sqrt (11 + 120 * Real.sqrt 6) - 6 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l906_90687


namespace NUMINAMATH_CALUDE_unique_lattice_point_l906_90662

theorem unique_lattice_point : 
  ∃! (x y : ℤ), x^2 - y^2 = 75 ∧ x - y = 5 := by sorry

end NUMINAMATH_CALUDE_unique_lattice_point_l906_90662


namespace NUMINAMATH_CALUDE_pi_over_two_not_fraction_l906_90612

-- Define what a fraction is
def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- State the theorem
theorem pi_over_two_not_fraction : ¬ is_fraction (π / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_over_two_not_fraction_l906_90612


namespace NUMINAMATH_CALUDE_circle_area_difference_l906_90668

/-- The difference in area between two circles -/
theorem circle_area_difference : 
  let r1 : ℝ := 30  -- radius of the first circle
  let d2 : ℝ := 15  -- diameter of the second circle
  π * r1^2 - π * (d2/2)^2 = 843.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l906_90668


namespace NUMINAMATH_CALUDE_girl_multiplication_mistake_l906_90602

theorem girl_multiplication_mistake (x : ℤ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_mistake_l906_90602


namespace NUMINAMATH_CALUDE_existence_of_twin_primes_l906_90628

theorem existence_of_twin_primes : ∃ n : ℕ, Prime n ∧ Prime (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twin_primes_l906_90628


namespace NUMINAMATH_CALUDE_negation_of_forall_proposition_l906_90613

open Set

theorem negation_of_forall_proposition :
  (¬ ∀ x ∈ (Set.Ioo 0 1), x^2 - x < 0) ↔ (∃ x ∈ (Set.Ioo 0 1), x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_proposition_l906_90613


namespace NUMINAMATH_CALUDE_maddie_bought_two_white_packs_l906_90631

/-- Represents the problem of determining the number of packs of white T-shirts Maddie bought. -/
def maddies_tshirt_problem (white_packs : ℕ) : Prop :=
  let blue_packs : ℕ := 4
  let white_per_pack : ℕ := 5
  let blue_per_pack : ℕ := 3
  let cost_per_shirt : ℕ := 3
  let total_spent : ℕ := 66
  
  (white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt = total_spent

/-- Theorem stating that Maddie bought 2 packs of white T-shirts. -/
theorem maddie_bought_two_white_packs : ∃ (white_packs : ℕ), white_packs = 2 ∧ maddies_tshirt_problem white_packs :=
sorry

end NUMINAMATH_CALUDE_maddie_bought_two_white_packs_l906_90631


namespace NUMINAMATH_CALUDE_max_value_AMC_l906_90636

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 12) :
  A * M * C + A * M + M * C + C * A ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_value_AMC_l906_90636


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l906_90611

theorem triangle_angle_ratio (a b c : ℝ) (h_sum : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 4*x ∧ b = 5*x ∧ c = 9*x) (h_smallest : min a (min b c) > 40) :
  max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l906_90611


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l906_90669

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l906_90669


namespace NUMINAMATH_CALUDE_min_marked_cells_for_unique_determination_l906_90691

/-- Represents a 9x9 board -/
def Board := Fin 9 → Fin 9 → Bool

/-- An L-shaped piece covering 3 cells -/
structure LPiece where
  x : Fin 9
  y : Fin 9
  orientation : Fin 4

/-- Checks if a given L-piece is uniquely determined by the marked cells -/
def isUniqueDetermination (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Checks if all possible L-piece placements are uniquely determined -/
def allPiecesUnique (board : Board) : Bool :=
  sorry

/-- Counts the number of marked cells on the board -/
def countMarkedCells (board : Board) : Nat :=
  sorry

/-- The main theorem: The minimum number of marked cells for unique determination is 63 -/
theorem min_marked_cells_for_unique_determination :
  ∃ (board : Board), allPiecesUnique board ∧ countMarkedCells board = 63 ∧
  ∀ (other_board : Board), allPiecesUnique other_board → countMarkedCells other_board ≥ 63 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_for_unique_determination_l906_90691


namespace NUMINAMATH_CALUDE_optimal_oil_storage_l906_90608

/-- Represents the optimal solution for storing oil in barrels -/
structure OilStorage where
  small_barrels : ℕ
  large_barrels : ℕ

/-- Checks if a given oil storage solution is valid -/
def is_valid_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  solution.small_barrels * small_capacity + solution.large_barrels * large_capacity = total_oil

/-- Checks if a given oil storage solution is optimal -/
def is_optimal_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  is_valid_solution total_oil small_capacity large_capacity solution ∧
  ∀ (other : OilStorage), 
    is_valid_solution total_oil small_capacity large_capacity other → 
    solution.small_barrels + solution.large_barrels ≤ other.small_barrels + other.large_barrels

/-- Theorem stating that the given solution is optimal for the oil storage problem -/
theorem optimal_oil_storage :
  is_optimal_solution 95 5 6 ⟨1, 15⟩ := by sorry

end NUMINAMATH_CALUDE_optimal_oil_storage_l906_90608


namespace NUMINAMATH_CALUDE_power_of_two_equation_l906_90667

theorem power_of_two_equation (r : ℤ) : 
  2^2001 - 2^2000 - 2^1999 + 2^1998 = r * 2^1998 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l906_90667


namespace NUMINAMATH_CALUDE_radio_cost_price_l906_90621

/-- 
Given a radio sold for Rs. 1330 with a 30% loss, 
prove that the original cost price was Rs. 1900.
-/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 30) : 
  (selling_price / (1 - loss_percentage / 100)) = 1900 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l906_90621


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l906_90619

theorem absolute_value_sum_zero (m n : ℝ) :
  |1 + m| + |n - 2| = 0 → m = -1 ∧ n = 2 ∧ m^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l906_90619


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l906_90686

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of no two heads appearing consecutively in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  if n ≤ 1 then 1 else sorry

theorem prob_no_consecutive_heads_10 : 
  prob_no_consecutive_heads n = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l906_90686


namespace NUMINAMATH_CALUDE_three_divides_difference_l906_90673

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Reverses a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.ones
  , tens := n.tens
  , ones := n.hundreds
  , is_valid := by sorry }

/-- Converts a ThreeDigitNumber to a natural number -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The difference between a number and its reverse -/
def difference (n : ThreeDigitNumber) : Int :=
  Int.natAbs (to_nat n - to_nat (reverse n))

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.hundreds ≠ n.ones) :
  3 ∣ difference n := by
  sorry

end NUMINAMATH_CALUDE_three_divides_difference_l906_90673


namespace NUMINAMATH_CALUDE_impossibleTransformation_l906_90657

-- Define the button colors
inductive Color
| A
| B
| C

-- Define the configuration as a list of colors
def Configuration := List Color

-- Define the card values
inductive CardValue
| One
| NegOne
| Zero

-- Function to calculate the card value between two adjacent colors
def getCardValue (c1 c2 : Color) : CardValue :=
  match c1, c2 with
  | Color.B, Color.A => CardValue.One
  | Color.A, Color.C => CardValue.One
  | Color.A, Color.B => CardValue.NegOne
  | Color.C, Color.A => CardValue.NegOne
  | _, _ => CardValue.Zero

-- Function to calculate the sum of card values for a configuration
def sumCardValues (config : Configuration) : Int :=
  let pairs := List.zip config (config.rotateLeft 1)
  let cardValues := pairs.map (fun (c1, c2) => getCardValue c1 c2)
  cardValues.foldl (fun sum cv => 
    sum + match cv with
    | CardValue.One => 1
    | CardValue.NegOne => -1
    | CardValue.Zero => 0
  ) 0

-- Define the initial and final configurations
def initialConfig : Configuration := [Color.A, Color.C, Color.B, Color.C, Color.B]
def finalConfig : Configuration := [Color.A, Color.B, Color.C, Color.B, Color.C]

-- Theorem: It's impossible to transform the initial configuration to the final configuration
theorem impossibleTransformation : 
  ∀ (swapSequence : List (Configuration → Configuration)),
  (∀ (config : Configuration), sumCardValues config = sumCardValues (swapSequence.foldl (fun c f => f c) config)) →
  swapSequence.foldl (fun c f => f c) initialConfig ≠ finalConfig :=
sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l906_90657


namespace NUMINAMATH_CALUDE_kho_kho_players_l906_90650

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho : ℕ) : 
  total = 25 → kabadi = 10 → both = 5 → kho_kho = total - kabadi + both := by
  sorry

end NUMINAMATH_CALUDE_kho_kho_players_l906_90650


namespace NUMINAMATH_CALUDE_morning_run_distance_l906_90649

/-- Represents the distances of various activities in miles -/
structure DailyActivities where
  morningRun : ℝ
  afternoonWalk : ℝ
  eveningBikeRide : ℝ

/-- Calculates the total distance covered in a day -/
def totalDistance (activities : DailyActivities) : ℝ :=
  activities.morningRun + activities.afternoonWalk + activities.eveningBikeRide

/-- Theorem stating that given the conditions, the morning run distance is 2 miles -/
theorem morning_run_distance 
  (activities : DailyActivities)
  (h1 : totalDistance activities = 18)
  (h2 : activities.afternoonWalk = 2 * activities.morningRun)
  (h3 : activities.eveningBikeRide = 12) :
  activities.morningRun = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_run_distance_l906_90649


namespace NUMINAMATH_CALUDE_am_gm_inequality_application_l906_90610

theorem am_gm_inequality_application (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 ∧
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2 ↔ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_application_l906_90610


namespace NUMINAMATH_CALUDE_root_ordering_l906_90683

/-- Given a quadratic function f(x) = (x-m)(x-n) + 2 where m < n,
    and α, β are the roots of f(x) = 0 with α < β,
    prove that m < α < β < n -/
theorem root_ordering (m n α β : ℝ) (hm : m < n) (hα : α < β)
  (hf : ∀ x, (x - m) * (x - n) + 2 = 0 ↔ x = α ∨ x = β) :
  m < α ∧ α < β ∧ β < n :=
sorry

end NUMINAMATH_CALUDE_root_ordering_l906_90683


namespace NUMINAMATH_CALUDE_circle_proof_l906_90645

-- Define the points
def A : ℝ × ℝ := (5, 2)
def B : ℝ × ℝ := (3, 2)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define the circle equation for the first part
def circle_eq1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Define the circle equation for the second part
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

theorem circle_proof :
  -- Part 1
  (∀ x y : ℝ, circle_eq1 x y ↔ 
    ((x, y) = A ∨ (x, y) = B) ∧ 
    (∃ cx cy : ℝ, line_eq cx cy ∧ (x - cx)^2 + (y - cy)^2 = (5 - cx)^2 + (2 - cy)^2)) ∧
  -- Part 2
  (∀ x y : ℝ, circle_eq2 x y ↔ 
    ((x, y) = O ∨ (x, y) = (2, 0) ∨ (x, y) = (0, 4)) ∧ 
    (∃ cx cy r : ℝ, (x - cx)^2 + (y - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (2 - cx)^2 + (0 - cy)^2 = r^2 ∧ 
                    (0 - cx)^2 + (4 - cy)^2 = r^2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_proof_l906_90645


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l906_90622

/-- A geometric sequence with a_3 = 1 and a_7 = 9 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 1 ∧ 
  a 7 = 9

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l906_90622


namespace NUMINAMATH_CALUDE_x_plus_ten_equals_forty_l906_90695

theorem x_plus_ten_equals_forty (x y : ℝ) (h1 : x / y = 6 / 3) (h2 : y = 15) : x + 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_ten_equals_forty_l906_90695


namespace NUMINAMATH_CALUDE_joe_team_wins_l906_90684

/-- Represents the number of points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- Represents the number of draws Joe's team had -/
def joe_team_draws : ℕ := 3

/-- Represents the number of wins the first-place team had -/
def first_place_wins : ℕ := 2

/-- Represents the number of ties the first-place team had -/
def first_place_ties : ℕ := 2

/-- Represents the point difference between the first-place team and Joe's team -/
def point_difference : ℕ := 2

/-- Theorem stating that Joe's team won exactly one game -/
theorem joe_team_wins : ℕ := by
  sorry

end NUMINAMATH_CALUDE_joe_team_wins_l906_90684


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l906_90674

/-- The number of pens left in a jar after removing some pens -/
theorem pens_left_in_jar
  (initial_blue : ℕ)
  (initial_black : ℕ)
  (initial_red : ℕ)
  (blue_removed : ℕ)
  (black_removed : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : black_removed = 7)
  : initial_blue + initial_black + initial_red - blue_removed - black_removed = 25 := by
  sorry


end NUMINAMATH_CALUDE_pens_left_in_jar_l906_90674


namespace NUMINAMATH_CALUDE_x_y_squared_sum_l906_90698

theorem x_y_squared_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : 1 / (x + y) = 1 / x - 1 / y) :
  (x / y + y / x)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_x_y_squared_sum_l906_90698


namespace NUMINAMATH_CALUDE_same_problem_probability_l906_90688

/-- The probability of two students choosing the same problem out of three options --/
theorem same_problem_probability : 
  let num_problems : ℕ := 3
  let num_students : ℕ := 2
  let total_outcomes : ℕ := num_problems ^ num_students
  let favorable_outcomes : ℕ := num_problems
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_same_problem_probability_l906_90688


namespace NUMINAMATH_CALUDE_product_sum_theorem_l906_90632

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℤ) + (b : ℤ) + (c : ℤ) = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l906_90632


namespace NUMINAMATH_CALUDE_isosceles_triangle_construction_impossibility_l906_90679

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  -- Side length of the two equal sides
  side : ℝ
  -- Base angle (half of the apex angle)
  base_angle : ℝ
  -- Height from the apex to the base
  height : ℝ
  -- Length of the angle bisector from the apex
  bisector : ℝ
  -- Constraint that the base angle is positive and less than π/2
  angle_constraint : 0 < base_angle ∧ base_angle < π/2

/-- Represents the ability to construct a geometric figure -/
def Constructible (α : Type) : Prop := sorry

/-- Represents the ability to trisect an angle -/
def AngleTrisectable (angle : ℝ) : Prop := sorry

/-- The main theorem stating the impossibility of general isosceles triangle construction -/
theorem isosceles_triangle_construction_impossibility 
  (h : ℝ) (l : ℝ) (h_pos : h > 0) (l_pos : l > 0) :
  ¬∀ (t : IsoscelesTriangle), 
    t.height = h ∧ t.bisector = l → 
    Constructible IsoscelesTriangle ∧ 
    ¬∀ (angle : ℝ), AngleTrisectable angle :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_construction_impossibility_l906_90679


namespace NUMINAMATH_CALUDE_room_length_l906_90681

/-- Proves that a rectangular room with given volume, height, and width has a specific length -/
theorem room_length (volume : ℝ) (height : ℝ) (width : ℝ) (length : ℝ) 
  (h_volume : volume = 10000)
  (h_height : height = 10)
  (h_width : width = 10)
  (h_room_volume : volume = length * width * height) :
  length = 100 :=
by sorry

end NUMINAMATH_CALUDE_room_length_l906_90681
