import Mathlib

namespace NUMINAMATH_CALUDE_vector_equation_l1301_130148

-- Define the vector type
variable {V : Type*} [AddCommGroup V]

-- Define points in space
variable (A B C D : V)

-- Define vectors
def vec (X Y : V) : V := Y - X

-- Theorem statement
theorem vector_equation (A B C D : V) :
  vec D A + vec C D - vec C B = vec B A := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l1301_130148


namespace NUMINAMATH_CALUDE_common_ratio_of_specific_geometric_sequence_l1301_130127

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_specific_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2/3 →
  a 4 = ∫ x in (1:ℝ)..(4:ℝ), (1 + 2*x) →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_common_ratio_of_specific_geometric_sequence_l1301_130127


namespace NUMINAMATH_CALUDE_total_spent_by_pete_and_raymond_l1301_130171

def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def initial_amount : ℕ := 250

def pete_nickels : ℕ := 4
def pete_dimes : ℕ := 3
def pete_quarters : ℕ := 2

def raymond_dimes_left : ℕ := 7
def raymond_quarters_left : ℕ := 4
def raymond_nickels_left : ℕ := 5

theorem total_spent_by_pete_and_raymond : 
  (initial_amount - (raymond_dimes_left * dime_value + raymond_quarters_left * quarter_value + raymond_nickels_left * nickel_value)) +
  (pete_nickels * nickel_value + pete_dimes * dime_value + pete_quarters * quarter_value) = 155 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_by_pete_and_raymond_l1301_130171


namespace NUMINAMATH_CALUDE_square_is_three_l1301_130144

/-- Represents a digit in base 8 -/
def Digit8 := Fin 8

/-- The addition problem in base 8 -/
def addition_problem (x : Digit8) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (5 * 8^3 + 3 * 8^2 + 2 * 8 + x.val) +
    (x.val * 8^2 + 6 * 8 + 1) +
    (x.val * 8 + 4) =
    6 * 8^3 + 3 * 8^2 + x.val * 8 + 2 +
    carry1 * 8 + carry2 * 8^2 + carry3 * 8^3

/-- The theorem stating that 3 is the unique solution to the addition problem -/
theorem square_is_three :
  ∃! (x : Digit8), addition_problem x ∧ x.val = 3 := by sorry

end NUMINAMATH_CALUDE_square_is_three_l1301_130144


namespace NUMINAMATH_CALUDE_two_digit_multiple_problem_l1301_130189

theorem two_digit_multiple_problem : ∃ (n : ℕ), 
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- multiple of 2
  (n + 1) % 3 = 0 ∧  -- adding 1 results in multiple of 3
  (n + 2) % 4 = 0 ∧  -- adding 2 results in multiple of 4
  (n + 3) % 5 = 0 ∧  -- adding 3 results in multiple of 5
  (∀ m : ℕ, 10 ≤ m ∧ m < n → 
    (m % 2 ≠ 0 ∨ (m + 1) % 3 ≠ 0 ∨ (m + 2) % 4 ≠ 0 ∨ (m + 3) % 5 ≠ 0)) ∧
  n = 62 := by
sorry

end NUMINAMATH_CALUDE_two_digit_multiple_problem_l1301_130189


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1301_130164

theorem square_perimeter_sum (a b : ℝ) (h1 : a^2 + b^2 = 85) (h2 : a^2 - b^2 = 45) :
  4*a + 4*b = 4*(Real.sqrt 65 + 2*Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1301_130164


namespace NUMINAMATH_CALUDE_sum_remainder_mod_20_l1301_130184

theorem sum_remainder_mod_20 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_20_l1301_130184


namespace NUMINAMATH_CALUDE_opposite_equal_roots_iff_l1301_130191

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_equal_roots (d e f n : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ 0 ∧ y₂ = -y₁ ∧
  (y₁^2 + 2*d*y₁) / (e*y₁ + f) = n / (n - 2) ∧
  (y₂^2 + 2*d*y₂) / (e*y₂ + f) = n / (n - 2)

/-- The main theorem -/
theorem opposite_equal_roots_iff (d e f : ℝ) :
  ∀ n : ℝ, has_opposite_equal_roots d e f n ↔ n = 4*d / (2*d - e) :=
sorry

end NUMINAMATH_CALUDE_opposite_equal_roots_iff_l1301_130191


namespace NUMINAMATH_CALUDE_john_squat_increase_l1301_130165

/-- The additional weight John added to his squat after training -/
def additional_weight : ℝ := 265

/-- John's initial squat weight in pounds -/
def initial_weight : ℝ := 135

/-- The factor by which the magical bracer increases strength -/
def strength_increase_factor : ℝ := 7

/-- John's final squat weight in pounds -/
def final_weight : ℝ := 2800

theorem john_squat_increase :
  (initial_weight + additional_weight) * strength_increase_factor = final_weight :=
sorry

end NUMINAMATH_CALUDE_john_squat_increase_l1301_130165


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l1301_130166

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l1301_130166


namespace NUMINAMATH_CALUDE_cookies_per_bag_l1301_130179

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (num_bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    num_bags = 26 →
    total_cookies = 52 →
    num_bags * cookies_per_bag = total_cookies →
    cookies_per_bag = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l1301_130179


namespace NUMINAMATH_CALUDE_family_spending_proof_l1301_130134

def planned_spending (family_size : ℕ) (orange_cost : ℚ) (savings_percentage : ℚ) : ℚ :=
  (family_size : ℚ) * orange_cost / (savings_percentage / 100)

theorem family_spending_proof :
  let family_size : ℕ := 4
  let orange_cost : ℚ := 3/2
  let savings_percentage : ℚ := 40
  planned_spending family_size orange_cost savings_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_family_spending_proof_l1301_130134


namespace NUMINAMATH_CALUDE_hiker_walking_problem_l1301_130102

/-- A hiker's walking problem over three days -/
theorem hiker_walking_problem 
  (day1_distance : ℝ) 
  (day1_speed : ℝ) 
  (day2_speed_increase : ℝ) 
  (day3_speed : ℝ) 
  (day3_time : ℝ) 
  (total_distance : ℝ) 
  (h1 : day1_distance = 18) 
  (h2 : day1_speed = 3) 
  (h3 : day2_speed_increase = 1) 
  (h4 : day3_speed = 5) 
  (h5 : day3_time = 6) 
  (h6 : total_distance = 68) :
  day1_distance / day1_speed - 
  (total_distance - day1_distance - day3_speed * day3_time) / (day1_speed + day2_speed_increase) = 1 := by
  sorry

end NUMINAMATH_CALUDE_hiker_walking_problem_l1301_130102


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1301_130111

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

/-- If 2, m, 6 form an arithmetic sequence, then m = 4 -/
theorem arithmetic_sequence_middle_term : 
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1301_130111


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l1301_130139

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 n) → n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l1301_130139


namespace NUMINAMATH_CALUDE_shortest_distance_to_start_l1301_130103

/-- Proof of the shortest distance between the third meeting point and the starting point on a circular track -/
theorem shortest_distance_to_start (track_length : ℝ) (time : ℝ) (speed_diff : ℝ) : 
  track_length = 400 →
  time = 8 * 60 →
  speed_diff = 0.1 →
  ∃ (speed_b : ℝ), 
    time * (speed_b + speed_b + speed_diff) = track_length * 3 ∧
    (time * speed_b) % track_length = 176 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_to_start_l1301_130103


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1301_130183

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x ∧ x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1301_130183


namespace NUMINAMATH_CALUDE_roberto_outfits_l1301_130138

/-- Calculates the number of different outfits given the number of options for each clothing item. -/
def calculate_outfits (trousers shirts jackets ties : ℕ) : ℕ :=
  trousers * shirts * jackets * ties

/-- Theorem stating that Roberto can create 240 different outfits. -/
theorem roberto_outfits :
  calculate_outfits 5 6 4 2 = 240 := by
  sorry

#eval calculate_outfits 5 6 4 2

end NUMINAMATH_CALUDE_roberto_outfits_l1301_130138


namespace NUMINAMATH_CALUDE_smallest_difference_PR_QR_l1301_130160

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  PQ : ℕ
  QR : ℕ
  PR : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.PQ + t.QR > t.PR ∧ t.PQ + t.PR > t.QR ∧ t.QR + t.PR > t.PQ

/-- Represents the conditions of the problem -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.PQ + t.QR + t.PR = 2023 ∧
  t.PQ ≤ t.QR ∧ t.QR < t.PR ∧
  is_valid_triangle t

/-- The main theorem stating the smallest possible difference between PR and QR -/
theorem smallest_difference_PR_QR :
  ∃ (t : Triangle), satisfies_conditions t ∧
  ∀ (t' : Triangle), satisfies_conditions t' → t.PR - t.QR ≤ t'.PR - t'.QR ∧
  t.PR - t.QR = 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_PR_QR_l1301_130160


namespace NUMINAMATH_CALUDE_A_eq_real_iff_m_in_range_l1301_130125

/-- The set A defined by the quadratic inequality -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * m * x + 1 > 0}

/-- Theorem stating the equivalence between A being equal to ℝ and m being in [0, 1) -/
theorem A_eq_real_iff_m_in_range (m : ℝ) : A m = Set.univ ↔ m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_A_eq_real_iff_m_in_range_l1301_130125


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l1301_130110

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10/3 ∨ x ≥ 2} ∧
    ∀ x, x ∈ S ↔ f (2*x) + f (x + 4) ≥ 8) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a*b) / |a| > f (b/a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l1301_130110


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1301_130157

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1301_130157


namespace NUMINAMATH_CALUDE_parallel_segment_ratio_sum_l1301_130196

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Represents the parallel line segments drawn from a point P inside the triangle -/
structure ParallelSegments where
  a' : ℝ
  b' : ℝ
  c' : ℝ
  ha' : a' > 0
  hb' : b' > 0
  hc' : c' > 0

/-- Theorem: For any triangle and any point P inside it, the sum of ratios of 
    parallel segments to corresponding sides is always 1 -/
theorem parallel_segment_ratio_sum (t : Triangle) (p : ParallelSegments) :
  p.a' / t.a + p.b' / t.b + p.c' / t.c = 1 := by sorry

end NUMINAMATH_CALUDE_parallel_segment_ratio_sum_l1301_130196


namespace NUMINAMATH_CALUDE_determinant_scaling_l1301_130114

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 10 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 90 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l1301_130114


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l1301_130143

def material_a_initial : ℚ := 2/9
def material_b_initial : ℚ := 1/8
def material_c_initial : ℚ := 3/10

def material_a_leftover : ℚ := 4/18
def material_b_leftover : ℚ := 1/12
def material_c_leftover : ℚ := 3/15

def total_used : ℚ := 17/120

theorem cheryl_material_usage :
  (material_a_initial - material_a_leftover) +
  (material_b_initial - material_b_leftover) +
  (material_c_initial - material_c_leftover) = total_used := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l1301_130143


namespace NUMINAMATH_CALUDE_samuel_travel_distance_l1301_130100

/-- The total distance Samuel needs to travel to reach the hotel -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) (remaining_distance : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + remaining_distance

/-- Theorem stating that Samuel needs to travel 600 miles to reach the hotel -/
theorem samuel_travel_distance :
  total_distance 50 80 3 4 130 = 600 := by
  sorry

end NUMINAMATH_CALUDE_samuel_travel_distance_l1301_130100


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1301_130172

/-- Given a hyperbola with eccentricity 5/4 and semi-major axis length 4,
    prove that its equation is x²/16 - y²/9 = 1 --/
theorem hyperbola_equation (x y : ℝ) :
  let a : ℝ := 4
  let e : ℝ := 5/4
  let c : ℝ := e * a
  let b : ℝ := Real.sqrt (c^2 - a^2)
  (x^2 / a^2) - (y^2 / b^2) = 1 → x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1301_130172


namespace NUMINAMATH_CALUDE_dog_distance_total_dog_distance_l1301_130107

/-- Proves that a dog running back and forth between two points covers 4000 meters
    by the time a person walks the distance between the points. -/
theorem dog_distance (distance : ℝ) (walking_speed : ℝ) (dog_speed : ℝ) : ℝ :=
  let time := distance * 1000 / walking_speed
  dog_speed * time

/-- The main theorem that calculates the total distance run by the dog. -/
theorem total_dog_distance : dog_distance 1 50 200 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_dog_distance_total_dog_distance_l1301_130107


namespace NUMINAMATH_CALUDE_smallest_positive_shift_is_90_l1301_130135

/-- A function with a 30-unit shift property -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 30) = f x

/-- The smallest positive shift for the scaled function -/
def SmallestPositiveShift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  b > 0 ∧
  (∀ x : ℝ, f ((x - b) / 3) = f (x / 3)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, f ((x - c) / 3) = f (x / 3)) → b ≤ c)

theorem smallest_positive_shift_is_90 (f : ℝ → ℝ) (h : ShiftFunction f) :
  SmallestPositiveShift f 90 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_shift_is_90_l1301_130135


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l1301_130187

def quiz_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def required_score : ℕ := 106

theorem quiz_score_theorem :
  (List.sum quiz_scores + required_score) / num_quizzes = target_mean := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l1301_130187


namespace NUMINAMATH_CALUDE_undefined_fraction_min_x_l1301_130154

theorem undefined_fraction_min_x : 
  let f (x : ℝ) := (x - 3) / (6 * x^2 - 37 * x + 6)
  ∀ y < 1/6, ∃ ε > 0, ∀ x ∈ Set.Ioo (y - ε) (y + ε), f x ≠ 0⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_undefined_fraction_min_x_l1301_130154


namespace NUMINAMATH_CALUDE_h_max_at_72_l1301_130169

/-- The divisor function d(n) -/
def d (n : ℕ+) : ℕ := sorry

/-- The function h(n) = d(n)^2 / n^(1/4) -/
noncomputable def h (n : ℕ+) : ℝ := (d n)^2 / n.val^(1/4 : ℝ)

/-- The theorem stating that h(n) is maximized when n = 72 -/
theorem h_max_at_72 : ∀ n : ℕ+, n ≠ 72 → h n < h 72 := by sorry

end NUMINAMATH_CALUDE_h_max_at_72_l1301_130169


namespace NUMINAMATH_CALUDE_jordans_money_exceeds_alexs_by_12_5_percent_l1301_130174

/-- Proves that Jordan's money value exceeds Alex's by 12.5% given the specified conditions -/
theorem jordans_money_exceeds_alexs_by_12_5_percent 
  (exchange_rate : ℝ) 
  (alex_dollars : ℝ) 
  (jordan_pounds : ℝ) 
  (h1 : exchange_rate = 1.5)
  (h2 : alex_dollars = 600)
  (h3 : jordan_pounds = 450) :
  (jordan_pounds * exchange_rate - alex_dollars) / alex_dollars * 100 = 12.5 := by
  sorry

#check jordans_money_exceeds_alexs_by_12_5_percent

end NUMINAMATH_CALUDE_jordans_money_exceeds_alexs_by_12_5_percent_l1301_130174


namespace NUMINAMATH_CALUDE_square_area_expansion_l1301_130156

theorem square_area_expansion (a : ℝ) (h : a > 0) :
  (3 * a)^2 = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_expansion_l1301_130156


namespace NUMINAMATH_CALUDE_inscribed_circle_tangency_angles_l1301_130186

/-- A rhombus with an inscribed circle -/
structure RhombusWithInscribedCircle where
  /-- The measure of the acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The assumption that the acute angle is 37 degrees -/
  acute_angle_is_37 : acute_angle = 37

/-- The angles formed by the points of tangency on the inscribed circle -/
def tangency_angles (r : RhombusWithInscribedCircle) : List ℝ :=
  [180 - r.acute_angle, r.acute_angle, 180 - r.acute_angle, r.acute_angle]

/-- Theorem stating the angles formed by the points of tangency -/
theorem inscribed_circle_tangency_angles (r : RhombusWithInscribedCircle) :
  tangency_angles r = [143, 37, 143, 37] := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangency_angles_l1301_130186


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1301_130133

/-- The number of points in the grid -/
def total_points : ℕ := 9

/-- The number of rows in the grid -/
def rows : ℕ := 3

/-- The number of columns in the grid -/
def columns : ℕ := 3

/-- The number of collinear points in each column -/
def collinear_points_per_column : ℕ := 3

/-- Theorem: The number of distinct triangles formed by choosing three points from a 3x3 grid of 9 points is equal to 82 -/
theorem distinct_triangles_in_grid : 
  (Nat.choose total_points 3) - (Nat.choose collinear_points_per_column 3 * columns) = 82 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1301_130133


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l1301_130124

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l1301_130124


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1301_130182

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1301_130182


namespace NUMINAMATH_CALUDE_prob_not_red_is_six_sevenths_l1301_130116

/-- The number of red jelly beans in the bag -/
def red_beans : ℕ := 4

/-- The number of green jelly beans in the bag -/
def green_beans : ℕ := 7

/-- The number of yellow jelly beans in the bag -/
def yellow_beans : ℕ := 5

/-- The number of blue jelly beans in the bag -/
def blue_beans : ℕ := 9

/-- The number of purple jelly beans in the bag -/
def purple_beans : ℕ := 3

/-- The total number of jelly beans in the bag -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The probability of selecting a jelly bean that is not red -/
def prob_not_red : ℚ := (green_beans + yellow_beans + blue_beans + purple_beans : ℚ) / total_beans

theorem prob_not_red_is_six_sevenths : prob_not_red = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_red_is_six_sevenths_l1301_130116


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1301_130181

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_4 : a 4 = -8)
  (h_8 : a 8 = 2) :
  a 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1301_130181


namespace NUMINAMATH_CALUDE_adjacent_sides_equal_not_implies_parallelogram_l1301_130130

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Definition of a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1 = q.D.1 - q.C.1 ∧ q.A.2 - q.B.2 = q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1 = q.B.1 - q.C.1 ∧ q.A.2 - q.D.2 = q.B.2 - q.C.2)

/-- Definition of equality of two sides -/
def sides_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2

/-- Theorem: Adjacent sides being equal does not imply parallelogram -/
theorem adjacent_sides_equal_not_implies_parallelogram :
  ¬∀ (q : Quadrilateral), 
    (sides_equal q.A q.B q.A q.D ∧ sides_equal q.B q.C q.C q.D) → 
    is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_adjacent_sides_equal_not_implies_parallelogram_l1301_130130


namespace NUMINAMATH_CALUDE_min_rice_purchase_l1301_130170

theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + 2 * r) 
  (h2 : o ≤ 3 * r) : 
  r ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_rice_purchase_l1301_130170


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1301_130136

/-- A hyperbola with a focus on the y-axis and asymptotic lines y = ± (√5/2)x has eccentricity 3√5/5 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = Real.sqrt 5 * b) →
  (Real.sqrt ((a^2 + b^2) / a^2) = 3 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1301_130136


namespace NUMINAMATH_CALUDE_triangle_properties_l1301_130163

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * Real.sin (t.A - t.C) = Real.sin t.B ∧
  t.AB = 5

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 3 * (10 ^ (1/2 : ℝ)) / 10 ∧
  ∃ (height : ℝ), height = 6 ∧ 
    height * t.AB / 2 = Real.sin t.C * (Real.sin t.A * t.AB / Real.sin t.C) * (Real.sin t.B * t.AB / Real.sin t.C) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1301_130163


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_l1301_130141

/-- Given a square with side length 2a and an intersecting line y = x + a/2,
    the perimeter of one part divided by a equals (√17 + 8) / 2 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := x + a / 2
  let intersection_points := [(-a, -a/2), (a, -a), (a/2, a), (-a, a)]
  let perimeter := Real.sqrt (17 * a^2) / 2 + 4 * a
  perimeter / a = (Real.sqrt 17 + 8) / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_l1301_130141


namespace NUMINAMATH_CALUDE_machine_parts_processed_l1301_130158

/-- Given two machines processing parts for 'a' hours, where the second machine
    processed 'n' fewer parts and takes 'b' minutes longer per part than the first,
    prove the number of parts processed by each machine. -/
theorem machine_parts_processed
  (a b n : ℝ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  let x := (b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  let y := (-b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  (∀ t, 0 < t ∧ t < a → (t / x = t / (x - n) - b / 60)) ∧
  x > 0 ∧ y > 0 ∧ x - y = n :=
sorry


end NUMINAMATH_CALUDE_machine_parts_processed_l1301_130158


namespace NUMINAMATH_CALUDE_unique_solution_for_P_equals_2C_l1301_130106

def P (r n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

def C (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem unique_solution_for_P_equals_2C (n : ℕ+) : 
  P 8 n = 2 * C 8 2 ↔ n = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_P_equals_2C_l1301_130106


namespace NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l1301_130140

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem a_eq_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (∀ (a' b' : ℝ), isPureImaginary (Complex.mk a' b') → a' = 0) ∧
               (∃ (a'' b'' : ℝ), a'' = 0 ∧ ¬isPureImaginary (Complex.mk a'' b'')) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l1301_130140


namespace NUMINAMATH_CALUDE_shape_C_has_two_lines_of_symmetry_l1301_130185

-- Define a type for shapes
inductive Shape : Type
  | A
  | B
  | C
  | D

-- Define a function to count lines of symmetry
def linesOfSymmetry : Shape → ℕ
  | Shape.A => 4
  | Shape.B => 0
  | Shape.C => 2
  | Shape.D => 1

-- Theorem statement
theorem shape_C_has_two_lines_of_symmetry :
  linesOfSymmetry Shape.C = 2 ∧
  ∀ s : Shape, s ≠ Shape.C → linesOfSymmetry s ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_shape_C_has_two_lines_of_symmetry_l1301_130185


namespace NUMINAMATH_CALUDE_raccoon_nuts_problem_l1301_130195

theorem raccoon_nuts_problem (raccoon_holes possum_holes : ℕ) : 
  raccoon_holes + possum_holes = 25 →
  possum_holes = raccoon_holes - 3 →
  5 * raccoon_holes = 6 * possum_holes →
  5 * raccoon_holes = 70 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_nuts_problem_l1301_130195


namespace NUMINAMATH_CALUDE_same_terminal_side_negative_420_and_660_l1301_130104

-- Define a function to represent angles with the same terminal side
def same_terminal_side (θ : ℝ) (φ : ℝ) : Prop :=
  ∃ n : ℤ, φ = θ + n * 360

-- Theorem statement
theorem same_terminal_side_negative_420_and_660 :
  same_terminal_side (-420) 660 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_negative_420_and_660_l1301_130104


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1301_130142

theorem geometric_series_problem (n : ℝ) : 
  let a₁ := 15
  let r₁ := 5 / 15
  let S₁ := a₁ / (1 - r₁)
  let a₂ := 15
  let r₂ := (5 + n) / 15
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → n = 20/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l1301_130142


namespace NUMINAMATH_CALUDE_tan_half_angle_l1301_130129

theorem tan_half_angle (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) 
  (h2 : Real.sin (3*π/2 + α) = 4/5) : 
  Real.tan (α/2) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_angle_l1301_130129


namespace NUMINAMATH_CALUDE_expansion_terms_abc_10_expansion_terms_abc_10_equals_solutions_l1301_130131

/-- The number of terms in the expansion of (a+b)^n -/
def binomial_terms (n : ℕ) : ℕ := n + 1

/-- The number of non-negative integer solutions to i + j + k = n -/
def trinomial_terms (n : ℕ) : ℕ := (n + 2).choose 2

theorem expansion_terms_abc_10 : trinomial_terms 10 = 66 := by
  sorry

theorem expansion_terms_abc_10_equals_solutions :
  trinomial_terms 10 = (Nat.choose 12 2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_abc_10_expansion_terms_abc_10_equals_solutions_l1301_130131


namespace NUMINAMATH_CALUDE_aaron_position_2023_l1301_130168

/-- Represents a point on the 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Aaron's movement rules -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

/-- Move one step in the given direction -/
def move (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.East => { x := p.x + 1, y := p.y }
  | Direction.North => { x := p.x, y := p.y + 1 }
  | Direction.West => { x := p.x - 1, y := p.y }
  | Direction.South => { x := p.x, y := p.y - 1 }

/-- Aaron's position after n steps -/
def aaronPosition (n : Nat) : Point :=
  sorry  -- The actual implementation would go here

theorem aaron_position_2023 :
  aaronPosition 2023 = { x := 21, y := -22 } := by
  sorry


end NUMINAMATH_CALUDE_aaron_position_2023_l1301_130168


namespace NUMINAMATH_CALUDE_total_third_grade_students_l1301_130150

theorem total_third_grade_students : 
  let class_a : ℕ := 48
  let class_b : ℕ := 65
  let class_c : ℕ := 57
  let class_d : ℕ := 72
  class_a + class_b + class_c + class_d = 242 := by
sorry

end NUMINAMATH_CALUDE_total_third_grade_students_l1301_130150


namespace NUMINAMATH_CALUDE_complex_multiplication_l1301_130180

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1301_130180


namespace NUMINAMATH_CALUDE_yellow_face_probability_l1301_130119

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l1301_130119


namespace NUMINAMATH_CALUDE_correct_match_probability_l1301_130193

theorem correct_match_probability (n : ℕ) (h : n = 6) :
  (1 : ℚ) / (Nat.factorial n) = (1 : ℚ) / 720 :=
sorry

end NUMINAMATH_CALUDE_correct_match_probability_l1301_130193


namespace NUMINAMATH_CALUDE_initial_average_age_proof_l1301_130145

/-- Proves that the initial average age of a group is 16 years, given the specified conditions. -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 20 →
  new_count = 20 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * initial_avg_age + new_count * new_avg_age) / (initial_count + new_count) = final_avg_age →
  initial_avg_age = 16 := by
  sorry

#check initial_average_age_proof

end NUMINAMATH_CALUDE_initial_average_age_proof_l1301_130145


namespace NUMINAMATH_CALUDE_quadrilateral_has_four_sides_and_angles_l1301_130113

/-- Definition of a quadrilateral -/
structure Quadrilateral where
  sides : Fin 4 → Seg
  angles : Fin 4 → Angle

/-- Theorem: A quadrilateral has four sides and four angles -/
theorem quadrilateral_has_four_sides_and_angles (q : Quadrilateral) :
  (∃ (s : Fin 4 → Seg), q.sides = s) ∧ (∃ (a : Fin 4 → Angle), q.angles = a) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_has_four_sides_and_angles_l1301_130113


namespace NUMINAMATH_CALUDE_chips_cost_calculation_l1301_130147

/-- Given the original cost and discount of chips, calculate the actual amount spent -/
theorem chips_cost_calculation (original_cost discount : ℚ) 
  (h1 : original_cost = 35)
  (h2 : discount = 17) :
  original_cost - discount = 18 := by
  sorry

end NUMINAMATH_CALUDE_chips_cost_calculation_l1301_130147


namespace NUMINAMATH_CALUDE_power_equality_l1301_130155

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1301_130155


namespace NUMINAMATH_CALUDE_parrot_guinea_pig_ownership_l1301_130162

theorem parrot_guinea_pig_ownership (total : ℕ) (parrot : ℕ) (guinea_pig : ℕ) :
  total = 48 →
  parrot = 30 →
  guinea_pig = 35 →
  ∃ (both : ℕ), both = 17 ∧ total = parrot + guinea_pig - both :=
by
  sorry

end NUMINAMATH_CALUDE_parrot_guinea_pig_ownership_l1301_130162


namespace NUMINAMATH_CALUDE_andy_initial_minks_l1301_130118

/-- The number of mink skins required to make one coat -/
def skins_per_coat : ℕ := 15

/-- The number of babies each mink has -/
def babies_per_mink : ℕ := 6

/-- The fraction of minks set free by activists -/
def fraction_set_free : ℚ := 1/2

/-- The number of coats Andy can make -/
def coats_made : ℕ := 7

/-- Theorem stating that given the conditions, Andy must have bought 30 minks initially -/
theorem andy_initial_minks :
  ∀ x : ℕ,
  (x + x * babies_per_mink) * (1 - fraction_set_free) = coats_made * skins_per_coat →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_andy_initial_minks_l1301_130118


namespace NUMINAMATH_CALUDE_ounces_per_can_l1301_130101

/-- Represents the number of ounces in a cup of chickpeas -/
def ounces_per_cup : ℕ := 6

/-- Represents the number of cups needed for one serving of hummus -/
def cups_per_serving : ℕ := 1

/-- Represents the number of servings Thomas wants to make -/
def total_servings : ℕ := 20

/-- Represents the number of cans Thomas needs to buy -/
def cans_needed : ℕ := 8

/-- Theorem stating the number of ounces in each can of chickpeas -/
theorem ounces_per_can : 
  (total_servings * cups_per_serving * ounces_per_cup) / cans_needed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_can_l1301_130101


namespace NUMINAMATH_CALUDE_gcd_of_80_180_450_l1301_130177

theorem gcd_of_80_180_450 : Nat.gcd 80 (Nat.gcd 180 450) = 10 := by sorry

end NUMINAMATH_CALUDE_gcd_of_80_180_450_l1301_130177


namespace NUMINAMATH_CALUDE_unbounded_ratio_on_circle_l1301_130152

theorem unbounded_ratio_on_circle (x y : ℝ) (h : x^2 + y^2 - 2*x + 2*y - 1 = 0) :
  ∀ M : ℝ, ∃ x' y' : ℝ, x'^2 + y'^2 - 2*x' + 2*y' - 1 = 0 ∧ (y' - 3) / x' > M :=
sorry

end NUMINAMATH_CALUDE_unbounded_ratio_on_circle_l1301_130152


namespace NUMINAMATH_CALUDE_worker_daily_hours_l1301_130108

/-- A worker's vacation and pay structure -/
structure Worker where
  workDaysPerWeek : ℕ
  paidVacationDays : ℕ
  hourlyRate : ℚ
  missedPay : ℚ

/-- Calculate the daily work hours of a worker -/
def calculateDailyHours (w : Worker) : ℚ :=
  let totalWorkDays : ℕ := 2 * w.workDaysPerWeek
  let unpaidDays : ℕ := totalWorkDays - w.paidVacationDays
  let totalUnpaidHours : ℚ := w.missedPay / w.hourlyRate
  totalUnpaidHours / unpaidDays

theorem worker_daily_hours (w : Worker) 
  (h1 : w.workDaysPerWeek = 5)
  (h2 : w.paidVacationDays = 6)
  (h3 : w.hourlyRate = 15)
  (h4 : w.missedPay = 480) :
  calculateDailyHours w = 8 := by
  sorry

#eval calculateDailyHours ⟨5, 6, 15, 480⟩

end NUMINAMATH_CALUDE_worker_daily_hours_l1301_130108


namespace NUMINAMATH_CALUDE_sum_of_prime_divisors_of_N_l1301_130121

/-- The number of ways to choose a committee from 11 men and 12 women,
    where the number of women is always one more than the number of men. -/
def N : ℕ := (Finset.range 12).sum (λ k => Nat.choose 11 k * Nat.choose 12 (k + 1))

/-- The sum of prime numbers that divide N -/
def sum_of_prime_divisors (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).sum (λ p => if p ∣ n then p else 0)

theorem sum_of_prime_divisors_of_N : sum_of_prime_divisors N = 79 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_divisors_of_N_l1301_130121


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1301_130175

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 - 4 * Complex.I) : 
  z = -4 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1301_130175


namespace NUMINAMATH_CALUDE_johns_equation_l1301_130112

theorem johns_equation (a b c d e : ℤ) : 
  a = 2 → b = 3 → c = 4 → d = 5 →
  (a - b - c * d + e = a - (b - (c * (d - e)))) →
  e = 8 := by
sorry

end NUMINAMATH_CALUDE_johns_equation_l1301_130112


namespace NUMINAMATH_CALUDE_brand_a_millet_percentage_l1301_130176

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : ℝ
  sunflower : ℝ
  composition_sum : millet + sunflower = 100

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brand_a : BirdseedBrand
  brand_b : BirdseedBrand
  proportion_a : ℝ
  proportion_b : ℝ
  proportions_sum : proportion_a + proportion_b = 100
  sunflower_percent : ℝ
  sunflower_balance : proportion_a / 100 * brand_a.sunflower + proportion_b / 100 * brand_b.sunflower = sunflower_percent

/-- Theorem stating that Brand A has 40% millet given the problem conditions -/
theorem brand_a_millet_percentage 
  (brand_a : BirdseedBrand)
  (brand_b : BirdseedBrand)
  (mix : BirdseedMix)
  (ha : brand_a.sunflower = 60)
  (hb1 : brand_b.millet = 65)
  (hb2 : brand_b.sunflower = 35)
  (hm1 : mix.sunflower_percent = 50)
  (hm2 : mix.proportion_a = 60)
  (hm3 : mix.brand_a = brand_a)
  (hm4 : mix.brand_b = brand_b) :
  brand_a.millet = 40 :=
sorry

end NUMINAMATH_CALUDE_brand_a_millet_percentage_l1301_130176


namespace NUMINAMATH_CALUDE_asian_games_competition_l1301_130105

/-- Represents a player in the competition -/
structure Player where
  prelim_prob : ℚ  -- Probability of passing preliminary round
  final_prob : ℚ   -- Probability of passing final round

/-- The three players in the competition -/
def players : List Player := [
  ⟨1/2, 1/3⟩,  -- Player A
  ⟨1/3, 1/3⟩,  -- Player B
  ⟨1/2, 1/3⟩   -- Player C
]

/-- Probability of a player participating in the city competition -/
def city_comp_prob (p : Player) : ℚ := p.prelim_prob * p.final_prob

/-- Probability of at least one player participating in the city competition -/
def at_least_one_prob : ℚ :=
  1 - (players.map (λ p => 1 - city_comp_prob p)).prod

/-- Expected value of Option 1 (Lottery) -/
def option1_expected : ℚ := 3 * (1/3) * 600

/-- Expected value of Option 2 (Fixed Rewards) -/
def option2_expected : ℚ := 700

/-- Main theorem to prove -/
theorem asian_games_competition :
  at_least_one_prob = 31/81 ∧ option2_expected > option1_expected := by
  sorry


end NUMINAMATH_CALUDE_asian_games_competition_l1301_130105


namespace NUMINAMATH_CALUDE_positive_X_value_l1301_130194

-- Define the ⊠ operation
def boxtimes (X Y : ℤ) : ℤ := X^2 - 2*X + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℤ, (boxtimes X 7 = 164) ∧ (X > 0) ∧ (∀ Y : ℤ, (boxtimes Y 7 = 164) ∧ (Y > 0) → Y = X) :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l1301_130194


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1301_130178

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1301_130178


namespace NUMINAMATH_CALUDE_age_difference_is_ten_l1301_130120

/-- The age difference between Declan's elder son and younger son -/
def age_difference : ℕ → ℕ → ℕ
  | elder_age, younger_age => elder_age - younger_age

/-- The current age of Declan's elder son -/
def elder_son_age : ℕ := 40

/-- The age of Declan's younger son 30 years from now -/
def younger_son_future_age : ℕ := 60

/-- The number of years in the future when the younger son's age is known -/
def years_in_future : ℕ := 30

theorem age_difference_is_ten :
  age_difference elder_son_age (younger_son_future_age - years_in_future) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_ten_l1301_130120


namespace NUMINAMATH_CALUDE_base_sum_theorem_l1301_130161

/-- Represents a repeating decimal in a given base -/
def repeating_decimal (numerator denominator base : ℕ) : ℚ :=
  (numerator : ℚ) / ((base ^ 2 - 1) : ℚ)

theorem base_sum_theorem :
  ∃! (B₁ B₂ : ℕ), 
    B₁ > 1 ∧ B₂ > 1 ∧
    repeating_decimal 45 99 B₁ = repeating_decimal 3 9 B₂ ∧
    repeating_decimal 54 99 B₁ = repeating_decimal 6 9 B₂ ∧
    B₁ + B₂ = 20 := by sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l1301_130161


namespace NUMINAMATH_CALUDE_negation_of_every_scientist_is_curious_l1301_130199

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a scientist and being curious
variable (scientist : U → Prop)
variable (curious : U → Prop)

-- State the theorem
theorem negation_of_every_scientist_is_curious :
  (¬ ∀ x, scientist x → curious x) ↔ (∃ x, scientist x ∧ ¬ curious x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_every_scientist_is_curious_l1301_130199


namespace NUMINAMATH_CALUDE_train_length_l1301_130122

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove that its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 72 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = (speed * 1000 / 3600) * time →
  length = 180 := by sorry

end NUMINAMATH_CALUDE_train_length_l1301_130122


namespace NUMINAMATH_CALUDE_intersection_P_Q_nonempty_intersection_P_R_nonempty_l1301_130153

-- Define the sets P, Q, and R
def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 > 0}
def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2*x + 2 = 0}

-- Theorem for part 1
theorem intersection_P_Q_nonempty (a : ℝ) : 
  (P ∩ Q a).Nonempty → a > -1/2 :=
sorry

-- Theorem for part 2
theorem intersection_P_R_nonempty (a : ℝ) : 
  (P ∩ R a).Nonempty → a ≥ -1/2 ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_P_Q_nonempty_intersection_P_R_nonempty_l1301_130153


namespace NUMINAMATH_CALUDE_factorization_problems_l1301_130128

theorem factorization_problems (x y : ℝ) (m : ℝ) : 
  (x^2 - 4 = (x + 2) * (x - 2)) ∧ 
  (2*m*x^2 - 4*m*x + 2*m = 2*m*(x - 1)^2) ∧ 
  ((y^2 - 1)^2 - 6*(y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1301_130128


namespace NUMINAMATH_CALUDE_max_figures_per_shelf_l1301_130146

def initial_shelves : Nat := 3
def shelf1_figures : Nat := 9
def shelf2_figures : Nat := 14
def shelf3_figures : Nat := 7
def additional_shelves : Nat := 2
def new_shelf_max : Nat := 11

def total_figures : Nat := shelf1_figures + shelf2_figures + shelf3_figures
def total_shelves : Nat := initial_shelves + additional_shelves

theorem max_figures_per_shelf :
  ∃ (x : Nat), 
    x ≤ new_shelf_max ∧ 
    x * total_shelves = total_figures ∧
    ∀ (y : Nat), y ≤ new_shelf_max ∧ y * total_shelves = total_figures → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_max_figures_per_shelf_l1301_130146


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l1301_130197

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

-- State the theorem
theorem sales_volume_estimate :
  ∃ ε > 0, |regression_equation 10 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l1301_130197


namespace NUMINAMATH_CALUDE_complex_product_real_solutions_l1301_130198

theorem complex_product_real_solutions (x : ℝ) : 
  (Complex.I : ℂ) * ((x + Complex.I) * ((x + 3 : ℝ) + 2 * Complex.I) * ((x + 5 : ℝ) - Complex.I)).im = 0 ↔ 
  x = -1.5 ∨ x = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_solutions_l1301_130198


namespace NUMINAMATH_CALUDE_mabels_daisies_l1301_130115

theorem mabels_daisies (petals_per_daisy : ℕ) (remaining_petals : ℕ) (daisies_given_away : ℕ) : 
  petals_per_daisy = 8 →
  daisies_given_away = 2 →
  remaining_petals = 24 →
  ∃ (initial_daisies : ℕ), 
    initial_daisies * petals_per_daisy = 
      remaining_petals + daisies_given_away * petals_per_daisy ∧
    initial_daisies = 5 :=
by sorry

end NUMINAMATH_CALUDE_mabels_daisies_l1301_130115


namespace NUMINAMATH_CALUDE_total_votes_correct_l1301_130126

/-- The total number of votes in an election --/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that Candidate A received --/
def candidate_A_percentage : ℚ := 55 / 100

/-- The percentage of invalid votes --/
def invalid_percentage : ℚ := 15 / 100

/-- The number of valid votes Candidate A received --/
def candidate_A_votes : ℕ := 261800

/-- Theorem stating that the total number of votes is correct given the conditions --/
theorem total_votes_correct :
  (↑candidate_A_votes : ℚ) = 
    (1 - invalid_percentage) * candidate_A_percentage * (↑total_votes : ℚ) :=
sorry

end NUMINAMATH_CALUDE_total_votes_correct_l1301_130126


namespace NUMINAMATH_CALUDE_jessica_chocolate_bar_cost_l1301_130109

/-- Represents Jessica's purchase --/
structure Purchase where
  total_cost : ℕ
  gummy_bear_packs : ℕ
  chocolate_chip_bags : ℕ
  gummy_bear_cost : ℕ
  chocolate_chip_cost : ℕ

/-- Calculates the cost of chocolate bars in Jessica's purchase --/
def chocolate_bar_cost (p : Purchase) : ℕ :=
  p.total_cost - (p.gummy_bear_packs * p.gummy_bear_cost + p.chocolate_chip_bags * p.chocolate_chip_cost)

/-- Theorem stating that the cost of chocolate bars in Jessica's purchase is $30 --/
theorem jessica_chocolate_bar_cost :
  let p : Purchase := {
    total_cost := 150,
    gummy_bear_packs := 10,
    chocolate_chip_bags := 20,
    gummy_bear_cost := 2,
    chocolate_chip_cost := 5
  }
  chocolate_bar_cost p = 30 := by
  sorry


end NUMINAMATH_CALUDE_jessica_chocolate_bar_cost_l1301_130109


namespace NUMINAMATH_CALUDE_students_just_passed_l1301_130173

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 63 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l1301_130173


namespace NUMINAMATH_CALUDE_vector_expression_equality_l1301_130132

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) : 
  (1/2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l1301_130132


namespace NUMINAMATH_CALUDE_cistern_emptying_l1301_130190

/-- Represents the fraction of a cistern emptied in a given time -/
def fractionEmptied (time : ℚ) : ℚ :=
  if time = 8 then 1/3
  else if time = 16 then 2/3
  else 0

theorem cistern_emptying (t : ℚ) :
  fractionEmptied 8 = 1/3 →
  fractionEmptied 16 = 2 * fractionEmptied 8 :=
by sorry

end NUMINAMATH_CALUDE_cistern_emptying_l1301_130190


namespace NUMINAMATH_CALUDE_cosine_sum_equals_radius_ratio_l1301_130137

-- Define a triangle with its angles, circumradius, and inradius
structure Triangle where
  α : Real
  β : Real
  γ : Real
  R : Real
  r : Real
  angle_sum : α + β + γ = Real.pi
  positive_R : R > 0
  positive_r : r > 0

-- State the theorem
theorem cosine_sum_equals_radius_ratio (t : Triangle) :
  Real.cos t.α + Real.cos t.β + Real.cos t.γ = (t.R + t.r) / t.R :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_equals_radius_ratio_l1301_130137


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l1301_130117

/-- Given two points P and Q that are symmetric about a line l, 
    prove that the equation of line l is x - y + 1 = 0 -/
theorem symmetric_points_line_equation 
  (a b : ℝ) 
  (h : a ≠ b - 1) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 1 = 0) →
    (∀ (R : ℝ × ℝ), R ∈ l → (dist P R = dist Q R)) →
    true :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l1301_130117


namespace NUMINAMATH_CALUDE_system_solution_l1301_130167

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1), (0, 0, 0)}

theorem system_solution (x y z : ℝ) :
  (x * y = z ∧ x * z = y ∧ y * z = x) ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1301_130167


namespace NUMINAMATH_CALUDE_team_reading_balance_l1301_130123

/-- The number of pages in the novel --/
def total_pages : ℕ := 820

/-- Alice's reading speed in seconds per page --/
def alice_speed : ℕ := 25

/-- Bob's reading speed in seconds per page --/
def bob_speed : ℕ := 50

/-- Chandra's reading speed in seconds per page --/
def chandra_speed : ℕ := 35

/-- The number of pages Chandra should read --/
def chandra_pages : ℕ := 482

theorem team_reading_balance :
  bob_speed * (total_pages - chandra_pages) = chandra_speed * chandra_pages := by
  sorry

#check team_reading_balance

end NUMINAMATH_CALUDE_team_reading_balance_l1301_130123


namespace NUMINAMATH_CALUDE_yanna_kept_36_apples_l1301_130149

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := 6

/-- The number of apples Yanna kept -/
def apples_kept : ℕ := total_apples - (apples_to_zenny + apples_to_andrea)

theorem yanna_kept_36_apples : apples_kept = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_kept_36_apples_l1301_130149


namespace NUMINAMATH_CALUDE_circle_regions_l1301_130192

/-- The number of regions into which n circles divide a plane, 
    where each pair of circles intersects and no three circles 
    intersect at the same point. -/
def f (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating the properties of the function f -/
theorem circle_regions (n : ℕ) : 
  n > 0 → 
  (f 3 = 8) ∧ 
  (f n = n^2 - n + 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_regions_l1301_130192


namespace NUMINAMATH_CALUDE_vector_magnitude_l1301_130159

theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  Real.sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1301_130159


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1301_130188

/-- The number of years until the ratio of Mike's age to Sam's age is 3:2 -/
def years_until_ratio (m s : ℕ) : ℕ :=
  9

theorem age_ratio_theorem (m s : ℕ) 
  (h1 : m - 5 = 2 * (s - 5))
  (h2 : m - 12 = 3 * (s - 12)) :
  (m + years_until_ratio m s) / (s + years_until_ratio m s) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l1301_130188


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1301_130151

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1301_130151
