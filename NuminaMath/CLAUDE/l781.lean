import Mathlib

namespace NUMINAMATH_CALUDE_flower_beds_count_l781_78158

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 270) (h2 : seeds_per_bed = 9) :
  total_seeds / seeds_per_bed = 30 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l781_78158


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l781_78132

theorem arithmetic_expression_equality : 
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l781_78132


namespace NUMINAMATH_CALUDE_pool_capacity_after_addition_l781_78141

/-- Proves that adding 300 gallons to a pool with given conditions results in 40.38% capacity filled -/
theorem pool_capacity_after_addition
  (total_capacity : ℝ)
  (additional_water : ℝ)
  (increase_percentage : ℝ)
  (h1 : total_capacity = 1529.4117647058824)
  (h2 : additional_water = 300)
  (h3 : increase_percentage = 30)
  (h4 : (additional_water / total_capacity) * 100 = increase_percentage) :
  let final_percentage := (((increase_percentage / 100) * total_capacity) / total_capacity) * 100
  ∃ ε > 0, |final_percentage - 40.38| < ε :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_after_addition_l781_78141


namespace NUMINAMATH_CALUDE_rice_containers_l781_78177

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 29/4 →
  container_weight = 29 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_weight : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l781_78177


namespace NUMINAMATH_CALUDE_f_range_l781_78121

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x - 1

theorem f_range : Set.range f = Set.Icc (-5/4 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l781_78121


namespace NUMINAMATH_CALUDE_xiaohong_school_distance_l781_78113

/-- The distance between Xiaohong's home and school -/
def distance : ℝ := 2880

/-- The scheduled arrival time in minutes -/
def scheduled_time : ℝ := 29

theorem xiaohong_school_distance :
  (∃ t : ℝ, 
    distance = 120 * (t - 5) ∧
    distance = 90 * (t + 3)) →
  distance = 2880 :=
by sorry

end NUMINAMATH_CALUDE_xiaohong_school_distance_l781_78113


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_is_seven_tenths_l781_78124

/-- The probability of drawing at least one white ball when randomly selecting two balls from a bag containing 3 black balls and 2 white balls. -/
def prob_at_least_one_white : ℚ := 7/10

/-- The total number of balls in the bag. -/
def total_balls : ℕ := 5

/-- The number of black balls in the bag. -/
def black_balls : ℕ := 3

/-- The number of white balls in the bag. -/
def white_balls : ℕ := 2

/-- The theorem stating that the probability of drawing at least one white ball
    when randomly selecting two balls from a bag containing 3 black balls and
    2 white balls is equal to 7/10. -/
theorem prob_at_least_one_white_is_seven_tenths :
  prob_at_least_one_white = 7/10 ∧
  total_balls = black_balls + white_balls ∧
  black_balls = 3 ∧
  white_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_is_seven_tenths_l781_78124


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l781_78102

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l781_78102


namespace NUMINAMATH_CALUDE_book_arrangement_count_l781_78148

def num_books : ℕ := 10
def num_calculus : ℕ := 3
def num_algebra : ℕ := 4
def num_statistics : ℕ := 3

theorem book_arrangement_count :
  (num_calculus.factorial * num_statistics.factorial * (num_books - num_algebra).factorial) = 25920 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l781_78148


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_final_l781_78149

/-- The number of points after applying the procedure once -/
def points_after_first_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The number of points after applying the procedure twice -/
def points_after_second_procedure (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that it's possible to have 82 points after the two procedures -/
theorem exists_initial_points_for_82_final : ∃ n : ℕ, points_after_second_procedure n = 82 := by
  sorry

#eval points_after_second_procedure 10

end NUMINAMATH_CALUDE_exists_initial_points_for_82_final_l781_78149


namespace NUMINAMATH_CALUDE_max_sum_of_product_107_l781_78154

theorem max_sum_of_product_107 (a b : ℤ) (h : a * b = 107) :
  ∃ (c d : ℤ), c * d = 107 ∧ c + d ≥ a + b ∧ c + d = 108 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_107_l781_78154


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l781_78134

theorem geometric_sequence_b_value (b : ℝ) (h₁ : b > 0) :
  (∃ r : ℝ, r ≠ 0 ∧
    b = 10 * r ∧
    10 / 9 = b * r ∧
    10 / 81 = (10 / 9) * r) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l781_78134


namespace NUMINAMATH_CALUDE_max_area_is_one_l781_78145

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ
  h_nonzero : m ≠ 0

/-- The maximum area of triangle PMN for the given ellipse and line configuration -/
def max_area (e : Ellipse) (l : Line) : ℝ := 1

/-- Theorem stating the maximum area of triangle PMN is 1 -/
theorem max_area_is_one (e : Ellipse) (l : Line) 
  (h_focus : e.a^2 - e.b^2 = 9)
  (h_vertex : e.a^2 = 12)
  (h_line : l.c = 3) :
  max_area e l = 1 := by sorry

end NUMINAMATH_CALUDE_max_area_is_one_l781_78145


namespace NUMINAMATH_CALUDE_square_area_ratio_l781_78150

theorem square_area_ratio (r : ℝ) (h : r > 0) : 
  (4 * r^2) / (2 * r^2) = 2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l781_78150


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l781_78120

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let fish_from_sister : ℕ := 47
  initial_fish + fish_from_sister = 69 :=
by sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l781_78120


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l781_78166

/-- The system of equations has no solution if and only if n = -1/2 -/
theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + 2*z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l781_78166


namespace NUMINAMATH_CALUDE_lindsey_october_savings_l781_78186

/-- Represents the amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- Represents Lindsey's savings in September -/
def september_savings : ℕ := 50

/-- Represents Lindsey's savings in November -/
def november_savings : ℕ := 11

/-- Represents the amount Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- Represents the cost of the video game -/
def video_game_cost : ℕ := 87

/-- Represents the amount Lindsey had left after buying the video game -/
def remaining_money : ℕ := 36

/-- Represents the condition that Lindsey saved more than $75 -/
def saved_more_than_75 : Prop :=
  september_savings + october_savings + november_savings > 75

theorem lindsey_october_savings : 
  september_savings + october_savings + november_savings + mom_gift - video_game_cost = remaining_money ∧
  saved_more_than_75 :=
sorry

end NUMINAMATH_CALUDE_lindsey_october_savings_l781_78186


namespace NUMINAMATH_CALUDE_ball_count_equality_l781_78116

-- Define the initial state of the urns
def Urn := ℕ → ℕ

-- m: initial number of black balls in the first urn
-- n: initial number of white balls in the second urn
-- k: number of balls transferred between urns
def initial_state (m n k : ℕ) : Urn × Urn :=
  (λ _ => m, λ _ => n)

-- Function to represent the ball transfer process
def transfer_balls (state : Urn × Urn) (k : ℕ) : Urn × Urn :=
  let (urn1, urn2) := state
  let urn1_after := λ color =>
    if color = 0 then urn1 0 - k + (k - (urn2 1 - (urn2 1 - k)))
    else k - (urn2 1 - (urn2 1 - k))
  let urn2_after := λ color =>
    if color = 0 then k - (k - (urn2 1 - (urn2 1 - k)))
    else urn2 1 - k + (urn2 1 - (urn2 1 - k))
  (urn1_after, urn2_after)

theorem ball_count_equality (m n k : ℕ) :
  let (final_urn1, final_urn2) := transfer_balls (initial_state m n k) k
  final_urn1 1 = final_urn2 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_equality_l781_78116


namespace NUMINAMATH_CALUDE_no_collision_probability_correct_l781_78111

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Fin 12 → Type)
  (adjacent : Fin 12 → Fin 5 → Fin 12)

/-- An ant on the icosahedron -/
structure Ant :=
  (position : Fin 12)

/-- The probability of an ant moving to a specific adjacent vertex -/
def move_probability : ℚ := 1 / 5

/-- The number of ants -/
def num_ants : ℕ := 12

/-- The probability that no two ants arrive at the same vertex -/
def no_collision_probability (i : Icosahedron) : ℚ :=
  (Nat.factorial num_ants : ℚ) / (5 ^ num_ants)

theorem no_collision_probability_correct (i : Icosahedron) :
  no_collision_probability i = (Nat.factorial num_ants : ℚ) / (5 ^ num_ants) :=
sorry

end NUMINAMATH_CALUDE_no_collision_probability_correct_l781_78111


namespace NUMINAMATH_CALUDE_inequality_range_l781_78192

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l781_78192


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l781_78128

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l781_78128


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l781_78159

theorem rationalize_and_simplify :
  ∃ (A B C D : ℕ), 
    (A * Real.sqrt B + C) / D = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) ∧
    A * Real.sqrt B + C = 5 * Real.sqrt 2 + Real.sqrt 10 ∧
    D = 4 ∧
    A + B + C + D = 12 ∧
    ∀ (A' B' C' D' : ℕ), 
      (A' * Real.sqrt B' + C') / D' = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) →
      A' + B' + C' + D' ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l781_78159


namespace NUMINAMATH_CALUDE_a_power_of_two_l781_78199

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * a n + 2^n

theorem a_power_of_two (k : ℕ) : ∃ m : ℕ, a (2^k) = 2^m := by
  sorry

end NUMINAMATH_CALUDE_a_power_of_two_l781_78199


namespace NUMINAMATH_CALUDE_intersection_value_l781_78115

theorem intersection_value (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = k * x₁ ∧ y₁ = 1 / x₁ ∧
  y₂ = k * x₂ ∧ y₂ = 1 / x₂ ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ →
  x₁ * y₂ + x₂ * y₁ = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_value_l781_78115


namespace NUMINAMATH_CALUDE_B_power_100_is_identity_l781_78122

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_100_is_identity :
  B ^ 100 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_100_is_identity_l781_78122


namespace NUMINAMATH_CALUDE_arrangementsWithRestrictionFor6_l781_78170

/-- The number of ways to arrange n people in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person
    cannot be placed on either end -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * linearArrangements (n - 1)

/-- Theorem stating that the number of ways to arrange 6 people in a line,
    where one specific person cannot be placed on either end, is 480 -/
theorem arrangementsWithRestrictionFor6 :
    arrangementsWithRestriction 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangementsWithRestrictionFor6_l781_78170


namespace NUMINAMATH_CALUDE_fourth_square_area_l781_78172

-- Define the triangles and their properties
structure Triangle (X Y Z : ℝ × ℝ) where
  is_right : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem fourth_square_area 
  (XYZ : Triangle X Y Z) 
  (XZW : Triangle X Z W) 
  (square1_area : ℝ) 
  (square2_area : ℝ) 
  (square3_area : ℝ) 
  (h1 : square1_area = 25) 
  (h2 : square2_area = 4) 
  (h3 : square3_area = 49) : 
  ∃ (fourth_square_area : ℝ), fourth_square_area = 78 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_area_l781_78172


namespace NUMINAMATH_CALUDE_inequality_system_solution_l781_78103

theorem inequality_system_solution (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x ≤ 2 ∧ x > m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l781_78103


namespace NUMINAMATH_CALUDE_path_length_is_pi_l781_78193

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the path length of a dot on the center of the top face when the prism is rolled -/
def pathLength (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that the path length for a 2x2x4 cm prism is π cm -/
theorem path_length_is_pi :
  let prism := RectangularPrism.mk 4 2 2
  pathLength prism = π :=
sorry

end NUMINAMATH_CALUDE_path_length_is_pi_l781_78193


namespace NUMINAMATH_CALUDE_john_mary_chess_consecutive_l781_78139

theorem john_mary_chess_consecutive (n : ℕ) : 
  ¬(n % 16 = 0 ∧ (n + 1) % 25 = 0) ∧ ¬((n + 1) % 16 = 0 ∧ n % 25 = 0) := by
  sorry

end NUMINAMATH_CALUDE_john_mary_chess_consecutive_l781_78139


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statements_a_and_b_correct_l781_78198

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a + c > b + c := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

-- Combined theorem for A and B
theorem statements_a_and_b_correct :
  (∀ (a b c : ℝ), a > b → c < 0 → a + c > b + c) ∧
  (∀ (a b : ℝ), a > b → b > 0 → (a + b) / 2 > Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statements_a_and_b_correct_l781_78198


namespace NUMINAMATH_CALUDE_expression_equality_l781_78163

theorem expression_equality : (50 - (5020 - 520)) + (5020 - (520 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l781_78163


namespace NUMINAMATH_CALUDE_power_multiplication_l781_78153

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l781_78153


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l781_78167

theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l781_78167


namespace NUMINAMATH_CALUDE_ten_hash_four_l781_78130

/-- Operation # defined on real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Properties of the hash operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- Theorem stating that 10 # 4 = 58 -/
theorem ten_hash_four : hash 10 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ten_hash_four_l781_78130


namespace NUMINAMATH_CALUDE_characterization_of_divisibility_implication_l781_78194

theorem characterization_of_divisibility_implication (n : ℕ) (hn : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_divisibility_implication_l781_78194


namespace NUMINAMATH_CALUDE_milly_science_homework_time_l781_78155

/-- The time Milly spends studying various subjects -/
structure StudyTime where
  math : ℕ
  geography : ℕ
  science : ℕ
  total : ℕ

/-- Milly's study time satisfies the given conditions -/
def millysStudyTime : StudyTime where
  math := 60
  geography := 30
  science := 45
  total := 135

theorem milly_science_homework_time :
  ∀ (st : StudyTime),
    st.math = 60 →
    st.geography = st.math / 2 →
    st.total = 135 →
    st.science = st.total - st.math - st.geography →
    st.science = 45 := by
  sorry

end NUMINAMATH_CALUDE_milly_science_homework_time_l781_78155


namespace NUMINAMATH_CALUDE_square_of_fraction_l781_78161

theorem square_of_fraction (a b c : ℝ) (hc : c ≠ 0) :
  ((-2 * a^2 * b) / (3 * c))^2 = (4 * a^4 * b^2) / (9 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_fraction_l781_78161


namespace NUMINAMATH_CALUDE_complement_of_A_l781_78126

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A : (Aᶜ : Set ℕ) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l781_78126


namespace NUMINAMATH_CALUDE_abc_sum_problem_l781_78105

theorem abc_sum_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 12) :
  c + 1 / b = 21 / 83 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l781_78105


namespace NUMINAMATH_CALUDE_age_relation_l781_78156

/-- Proves that A was twice as old as B 10 years ago given the conditions -/
theorem age_relation (b_age : ℕ) (a_age : ℕ) (x : ℕ) : 
  b_age = 42 →
  a_age = b_age + 12 →
  a_age + 10 = 2 * (b_age - x) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_relation_l781_78156


namespace NUMINAMATH_CALUDE_caramel_apple_ice_cream_cost_difference_l781_78137

/-- The cost difference between a caramel apple and an ice cream cone -/
theorem caramel_apple_ice_cream_cost_difference 
  (caramel_apple_cost : ℕ) 
  (ice_cream_cost : ℕ) 
  (h1 : caramel_apple_cost = 25)
  (h2 : ice_cream_cost = 15) : 
  caramel_apple_cost - ice_cream_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_caramel_apple_ice_cream_cost_difference_l781_78137


namespace NUMINAMATH_CALUDE_rally_accident_probability_l781_78108

/-- The probability of a car successfully completing the rally --/
def rally_success_probability : ℚ :=
  let bridge_success : ℚ := 4/5
  let turn_success : ℚ := 7/10
  let tunnel_success : ℚ := 9/10
  let sand_success : ℚ := 3/5
  bridge_success * turn_success * tunnel_success * sand_success

/-- The probability of a car being involved in an accident during the rally --/
def accident_probability : ℚ := 1 - rally_success_probability

theorem rally_accident_probability :
  accident_probability = 1756 / 2500 :=
by sorry

end NUMINAMATH_CALUDE_rally_accident_probability_l781_78108


namespace NUMINAMATH_CALUDE_sector_area_l781_78180

theorem sector_area (diameter : ℝ) (central_angle : ℝ) :
  diameter = 6 →
  central_angle = 120 →
  (π * (diameter / 2)^2 * central_angle / 360) = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l781_78180


namespace NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l781_78191

theorem polygon_sides_when_angles_equal : ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_angles_equal_l781_78191


namespace NUMINAMATH_CALUDE_expression_value_l781_78112

theorem expression_value : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l781_78112


namespace NUMINAMATH_CALUDE_P_divisibility_l781_78162

/-- The polynomial P(x) defined in terms of a and b -/
def P (a b x : ℚ) : ℚ := (a + b) * x^5 + a * b * x^2 + 1

/-- The theorem stating the conditions for P(x) to be divisible by x^2 - 3x + 2 -/
theorem P_divisibility (a b : ℚ) : 
  (∀ x, (x^2 - 3*x + 2) ∣ P a b x) ↔ 
  ((a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1)) :=
sorry

end NUMINAMATH_CALUDE_P_divisibility_l781_78162


namespace NUMINAMATH_CALUDE_jason_debt_l781_78165

def mowing_value (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 3
  | 2 => 5
  | 0 => 7
  | _ => 0

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map mowing_value |>.sum

theorem jason_debt : total_earnings 25 = 123 := by
  sorry

end NUMINAMATH_CALUDE_jason_debt_l781_78165


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l781_78188

/-- An isosceles triangle with two sides of lengths 1 and 2 has a perimeter of 5 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 1 ∧ b = 2 ∧ c = 2 →  -- Two sides are 1 and 2, the third side must be 2 to form an isosceles triangle
  a + b + c = 5 :=         -- The perimeter is the sum of all sides
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l781_78188


namespace NUMINAMATH_CALUDE_lines_properties_l781_78135

/-- Two lines in 2D space -/
structure Lines where
  l1 : ℝ → ℝ → ℝ := fun x y => 2 * x + y + 4
  l2 : ℝ → ℝ → ℝ → ℝ := fun a x y => a * x + 4 * y + 1

/-- The intersection point of two lines when they are perpendicular -/
def intersection (lines : Lines) : ℝ × ℝ := sorry

/-- The distance between two lines when they are parallel -/
def distance (lines : Lines) : ℝ := sorry

/-- Main theorem about the properties of the two lines -/
theorem lines_properties (lines : Lines) :
  (intersection lines = (-3/2, -1) ∧ 
   distance lines = 3 * Real.sqrt 5 / 4) := by sorry

end NUMINAMATH_CALUDE_lines_properties_l781_78135


namespace NUMINAMATH_CALUDE_f_zero_is_zero_l781_78125

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem f_zero_is_zero (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 4) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_is_zero_l781_78125


namespace NUMINAMATH_CALUDE_team_a_construction_team_b_construction_l781_78190

-- Define the parameters
def total_length_1 : ℝ := 600
def initial_days : ℝ := 5
def additional_days : ℝ := 2
def daily_increase : ℝ := 20
def total_length_2 : ℝ := 1800
def team_b_initial : ℝ := 360
def team_b_increase : ℝ := 0.2

-- Define Team A's daily construction after increase
def team_a_daily (x : ℝ) : ℝ := x

-- Define Team B's daily construction after increase
def team_b_daily (m : ℝ) : ℝ := m * (1 + team_b_increase)

-- Theorem for Team A's daily construction
theorem team_a_construction :
  ∃ x : ℝ, initial_days * (team_a_daily x - daily_increase) + additional_days * team_a_daily x = total_length_1 ∧
  team_a_daily x = 100 := by sorry

-- Theorem for Team B's original daily construction
theorem team_b_construction :
  ∃ m : ℝ, team_b_initial / m + (total_length_2 / 2 - team_b_initial) / (team_b_daily m) = total_length_2 / 2 / 100 ∧
  m = 90 := by sorry

end NUMINAMATH_CALUDE_team_a_construction_team_b_construction_l781_78190


namespace NUMINAMATH_CALUDE_chocolate_heart_bags_l781_78171

theorem chocolate_heart_bags (total_candy : ℕ) (total_bags : ℕ) (kisses_bags : ℕ) (non_chocolate_pieces : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : kisses_bags = 3)
  (h4 : non_chocolate_pieces = 28)
  (h5 : total_candy % total_bags = 0) -- Ensure equal division
  : (total_bags - kisses_bags - (non_chocolate_pieces / (total_candy / total_bags))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_heart_bags_l781_78171


namespace NUMINAMATH_CALUDE_set_equality_gt_one_set_equality_odd_integers_l781_78146

-- Statement 1
theorem set_equality_gt_one : {x : ℝ | x > 1} = {y : ℝ | y > 1} := by sorry

-- Statement 2
theorem set_equality_odd_integers : {x : ℤ | ∃ k : ℤ, x = 2*k + 1} = {x : ℤ | ∃ k : ℤ, x = 2*k - 1} := by sorry

end NUMINAMATH_CALUDE_set_equality_gt_one_set_equality_odd_integers_l781_78146


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l781_78129

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l781_78129


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l781_78196

theorem max_value_of_sum_of_roots (x : ℝ) (h : 3 < x ∧ x < 6) :
  ∃ (k : ℝ), k = Real.sqrt 6 ∧ ∀ y : ℝ, (Real.sqrt (x - 3) + Real.sqrt (6 - x) ≤ y) → y ≥ k :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l781_78196


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l781_78104

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 1) * (x - 3) < 0} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l781_78104


namespace NUMINAMATH_CALUDE_negation_of_proposition_l781_78178

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l781_78178


namespace NUMINAMATH_CALUDE_courtyard_length_courtyard_length_is_25_l781_78174

/-- Proves that the length of a rectangular courtyard is 25 meters -/
theorem courtyard_length : ℝ → ℝ → ℝ → ℝ → Prop :=
  λ (width : ℝ) (num_bricks : ℝ) (brick_length : ℝ) (brick_width : ℝ) =>
    width = 16 ∧
    num_bricks = 20000 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 →
    (num_bricks * brick_length * brick_width) / width = 25

/-- The length of the courtyard is 25 meters -/
theorem courtyard_length_is_25 :
  courtyard_length 16 20000 0.2 0.1 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_courtyard_length_is_25_l781_78174


namespace NUMINAMATH_CALUDE_cassies_dogs_l781_78136

/-- The number of parrots Cassie has -/
def num_parrots : ℕ := 8

/-- The number of nails per dog foot -/
def nails_per_dog_foot : ℕ := 4

/-- The number of feet a dog has -/
def dog_feet : ℕ := 4

/-- The number of claws per parrot leg -/
def claws_per_parrot_leg : ℕ := 3

/-- The number of legs a parrot has -/
def parrot_legs : ℕ := 2

/-- The total number of nails Cassie needs to cut -/
def total_nails : ℕ := 113

/-- The number of dogs Cassie has -/
def num_dogs : ℕ := 4

theorem cassies_dogs :
  num_dogs = 4 :=
by sorry

end NUMINAMATH_CALUDE_cassies_dogs_l781_78136


namespace NUMINAMATH_CALUDE_domain_all_reals_l781_78189

theorem domain_all_reals (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (3 * k * x^2 - 4 * x + 7) / (-7 * x^2 - 4 * x + k)) ↔ 
  k < -4/7 :=
sorry

end NUMINAMATH_CALUDE_domain_all_reals_l781_78189


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l781_78118

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
  n % 10 = 2 ∧                 -- ends in 2
  2000000 + (n / 10) = 3 * n ∧ -- moving 2 to first position triples the number
  n = 857142 := by
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l781_78118


namespace NUMINAMATH_CALUDE_original_number_proof_l781_78127

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 32 = 87 * k) ∧ 
  (∀ m : ℕ, m < 32 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 119 :=
sorry

end NUMINAMATH_CALUDE_original_number_proof_l781_78127


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l781_78168

theorem quadratic_complete_square (x : ℝ) : ∃ (a k : ℝ), 
  3 * x^2 + 8 * x + 15 = a * (x - (-4/3))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l781_78168


namespace NUMINAMATH_CALUDE_A_must_be_four_l781_78197

/-- Represents a six-digit number in the form 32BA33 -/
def SixDigitNumber (A : Nat) : Nat :=
  320000 + A * 100 + 33

/-- Rounds a number to the nearest hundred -/
def roundToNearestHundred (n : Nat) : Nat :=
  ((n + 50) / 100) * 100

/-- Theorem stating that if 32BA33 rounds to 323400, then A must be 4 -/
theorem A_must_be_four :
  ∀ A : Nat, A < 10 →
  roundToNearestHundred (SixDigitNumber A) = 323400 →
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_A_must_be_four_l781_78197


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l781_78119

theorem complex_subtraction_simplification :
  (7 - 3*I) - (9 - 5*I) = -2 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l781_78119


namespace NUMINAMATH_CALUDE_empty_set_problem_l781_78181

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A ≠ ∅) ∧ (set_B ≠ ∅) ∧ (set_C ≠ ∅) ∧ (set_D = ∅) :=
sorry

end NUMINAMATH_CALUDE_empty_set_problem_l781_78181


namespace NUMINAMATH_CALUDE_total_books_read_l781_78131

def books_may : ℕ := 2
def books_june : ℕ := 6
def books_july : ℕ := 10
def books_august : ℕ := 14
def books_september : ℕ := 18

theorem total_books_read : books_may + books_june + books_july + books_august + books_september = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_l781_78131


namespace NUMINAMATH_CALUDE_stratified_sampling_girls_count_l781_78133

theorem stratified_sampling_girls_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 6)
  (h4 : sample_size = (sample_size / 2 - girls_boys_diff / 2) * 2 + girls_boys_diff) :
  (sample_size / 2 - girls_boys_diff / 2) * (total_students / sample_size) = 970 := by
  sorry

#check stratified_sampling_girls_count

end NUMINAMATH_CALUDE_stratified_sampling_girls_count_l781_78133


namespace NUMINAMATH_CALUDE_ball_attendees_l781_78187

theorem ball_attendees :
  ∀ (ladies gentlemen : ℕ),
  ladies + gentlemen < 50 →
  (3 * ladies) / 4 = (5 * gentlemen) / 7 →
  ladies + gentlemen = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l781_78187


namespace NUMINAMATH_CALUDE_total_contribution_proof_l781_78114

/-- Proves that the total contribution is $1040 given the specified conditions --/
theorem total_contribution_proof (niraj brittany angela : ℕ) : 
  niraj = 80 ∧ 
  brittany = 3 * niraj ∧ 
  angela = 3 * brittany → 
  niraj + brittany + angela = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_contribution_proof_l781_78114


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l781_78143

variable (a b x y : ℝ)

/-- Factorization of 3ax^2 - 6ax + 3a --/
theorem factorization_1 : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

/-- Factorization of 9x^2(a-b) + 4y^3(b-a) --/
theorem factorization_2 : 9*x^2*(a-b) + 4*y^3*(b-a) = (a-b)*(9*x^2 - 4*y^3) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l781_78143


namespace NUMINAMATH_CALUDE_hyperbola_properties_l781_78107

noncomputable section

/-- Definition of the hyperbola C -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of the focal length -/
def focal_length (a b : ℝ) : ℝ := 4 * Real.sqrt 2

/-- Definition of the point P on the hyperbola -/
def point_on_hyperbola (a b x₀ y₀ : ℝ) : Prop :=
  hyperbola a b x₀ y₀

/-- Definition of points P₁ and P₂ on the hyperbola -/
def points_on_hyperbola (a b x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂

/-- Definition of the vector relation -/
def vector_relation (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  3 * x₀ = x₁ + 2 * x₂ ∧ 3 * y₀ = y₁ + 2 * y₂

/-- Definition of perpendicular lines through P -/
def perpendicular_lines (a x₀ : ℝ) : Prop :=
  ∃ (y : ℝ), x₀ * y = -a^2

/-- Main theorem -/
theorem hyperbola_properties
  (a b x₀ y₀ x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : focal_length a b = 4 * Real.sqrt 2)
  (h₄ : point_on_hyperbola a b x₀ y₀)
  (h₅ : points_on_hyperbola a b x₀ y₀ x₁ y₁ x₂ y₂)
  (h₆ : vector_relation x₀ y₀ x₁ y₁ x₂ y₂)
  (h₇ : perpendicular_lines a x₀) :
  (x₁ * x₂ - y₁ * y₂ = 9) ∧
  (∃ (S : ℝ), S ≤ 9/2 ∧ (S = 9/2 ↔ a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (∃ (A B : ℝ × ℝ), ∀ (x y : ℝ),
    (x - 2 * Real.sqrt 2)^2 + y^2 = (x + 2 * Real.sqrt 2)^2 + y^2 →
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l781_78107


namespace NUMINAMATH_CALUDE_matrix_product_l781_78144

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, 1], ![2, 1, 2], ![1, 2, 3]]
def B : Matrix (Fin 3) (Fin 3) ℤ := ![![1, 1, -1], ![2, -1, 1], ![1, 0, 1]]
def C : Matrix (Fin 3) (Fin 3) ℤ := ![![6, 2, -1], ![6, 1, 1], ![8, -1, 4]]

theorem matrix_product : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_l781_78144


namespace NUMINAMATH_CALUDE_x_value_l781_78152

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value : ∀ x : ℝ, N ⊆ M x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l781_78152


namespace NUMINAMATH_CALUDE_fraction_of_seniors_studying_japanese_l781_78184

theorem fraction_of_seniors_studying_japanese 
  (num_juniors : ℝ) 
  (num_seniors : ℝ) 
  (fraction_juniors_studying : ℝ) 
  (fraction_total_studying : ℝ) :
  num_seniors = 3 * num_juniors →
  fraction_juniors_studying = 3 / 4 →
  fraction_total_studying = 0.4375 →
  (fraction_total_studying * (num_juniors + num_seniors) - fraction_juniors_studying * num_juniors) / num_seniors = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_seniors_studying_japanese_l781_78184


namespace NUMINAMATH_CALUDE_jacksons_running_distance_l781_78157

/-- Calculates the final daily running distance after a given number of weeks,
    starting from an initial distance and increasing by a fixed amount each week. -/
def finalRunningDistance (initialDistance : ℕ) (weeklyIncrease : ℕ) (totalWeeks : ℕ) : ℕ :=
  initialDistance + weeklyIncrease * (totalWeeks - 1)

/-- Proves that Jackson's final daily running distance is 7 miles
    after 5 weeks of training. -/
theorem jacksons_running_distance :
  finalRunningDistance 3 1 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_running_distance_l781_78157


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l781_78183

/-- The probability of getting heads in a single flip of the unfair coin -/
def p_heads : ℚ := 1/3

/-- The probability of getting tails in a single flip of the unfair coin -/
def p_tails : ℚ := 2/3

/-- The number of coin flips -/
def n : ℕ := 10

/-- The number of heads we want to get -/
def k : ℕ := 3

/-- The probability of getting exactly k heads in n flips of the unfair coin -/
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem unfair_coin_flip_probability :
  prob_k_heads n k p_heads = 512/1969 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l781_78183


namespace NUMINAMATH_CALUDE_complex_equation_solution_l781_78123

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 + Complex.I → z = (1 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l781_78123


namespace NUMINAMATH_CALUDE_reflected_arcs_area_l781_78138

/-- The area of the region bounded by 8 reflected arcs in a circle with an inscribed regular octagon -/
theorem reflected_arcs_area (s : ℝ) (h : s = 1) : 
  let r : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2)
  let arc_area : ℝ := π * (2 + Real.sqrt 2) / 2 - 2 * Real.sqrt 3
  octagon_area - arc_area = 2 * (1 + Real.sqrt 2) - π * (2 + Real.sqrt 2) / 2 + 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_reflected_arcs_area_l781_78138


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l781_78164

theorem no_simultaneous_squares (n : ℕ+) : ¬∃ (x y : ℕ+), (n + 1 : ℕ) = x^2 ∧ (4*n + 1 : ℕ) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l781_78164


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l781_78151

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, 4 * r^2 - 7 * r - 10 = 0 ∧ 4 * s^2 - 7 * s - 10 = 0 ∧
   ∀ x : ℚ, x^2 + b * x + c = 0 ↔ (x = r + 3 ∨ x = s + 3)) →
  c = 47 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l781_78151


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l781_78110

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l781_78110


namespace NUMINAMATH_CALUDE_zero_last_in_hundreds_l781_78109

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Get the units digit of a number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Check if a digit has appeared in the units position up to the nth Fibonacci number -/
def digit_appeared_units (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ units_digit (fib k) = d

/-- Check if a digit has appeared in the hundreds position up to the nth Fibonacci number -/
def digit_appeared_hundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundreds_digit (fib k) = d

/-- The main theorem: 0 is the last digit to appear in the hundreds position -/
theorem zero_last_in_hundreds :
  ∃ N : ℕ, ∀ d : ℕ, d < 10 →
    (∀ n ≥ N, digit_appeared_units d n → digit_appeared_hundreds d n) ∧
    (∃ n ≥ N, digit_appeared_units 0 n ∧ ¬digit_appeared_hundreds 0 n) :=
sorry

end NUMINAMATH_CALUDE_zero_last_in_hundreds_l781_78109


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l781_78160

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 0.2 →
  added_alcohol = 3.6 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l781_78160


namespace NUMINAMATH_CALUDE_intersection_theorem_l781_78117

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 > 0}

-- State the theorem
theorem intersection_theorem :
  M ∩ (Set.univ \ N) = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l781_78117


namespace NUMINAMATH_CALUDE_fib_8_and_sum_2016_l781_78185

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of first n terms of Fibonacci sequence -/
def fib_sum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

theorem fib_8_and_sum_2016 :
  fib 7 = 21 ∧
  ∀ m : ℕ, fib 2017 = m^2 + 1 → fib_sum 2016 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_fib_8_and_sum_2016_l781_78185


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_angles_l781_78175

/-- An isosceles triangle with a special angle bisector property -/
structure SpecialIsoscelesTriangle where
  -- The base angles of the isosceles triangle
  base_angle : ℝ
  -- The angle between the angle bisector from the vertex and the angle bisector to the lateral side
  bisector_angle : ℝ
  -- The condition that the bisector angle equals the vertex angle
  h_bisector_eq_vertex : bisector_angle = 180 - 2 * base_angle

/-- The possible angles of a special isosceles triangle -/
def special_triangle_angles (t : SpecialIsoscelesTriangle) : Prop :=
  (t.base_angle = 36 ∧ 180 - 2 * t.base_angle = 108) ∨
  (t.base_angle = 60 ∧ 180 - 2 * t.base_angle = 60)

/-- Theorem: The angles of a special isosceles triangle are either (36°, 36°, 108°) or (60°, 60°, 60°) -/
theorem special_isosceles_triangle_angles (t : SpecialIsoscelesTriangle) :
  special_triangle_angles t := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_angles_l781_78175


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l781_78169

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l781_78169


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l781_78106

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base_length :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 13 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 3 ∨ b = 3 ∨ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l781_78106


namespace NUMINAMATH_CALUDE_first_month_sale_l781_78101

theorem first_month_sale
  (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale2 = 6927)
  (h2 : sale3 = 6855)
  (h3 : sale4 = 7230)
  (h4 : sale5 = 6562)
  (h5 : sale6 = 6191)
  (h6 : avg_sale = 6700) :
  6 * avg_sale - (sale2 + sale3 + sale4 + sale5 + sale6) = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l781_78101


namespace NUMINAMATH_CALUDE_largest_five_digit_number_with_product_180_l781_78147

/-- Represents a five-digit number as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Checks if a given list represents a valid five-digit number -/
def is_valid_five_digit_number (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.all (· < 10) ∧ n.head! ≠ 0

/-- Computes the product of the digits of a number -/
def digit_product (n : FiveDigitNumber) : Nat :=
  n.prod

/-- Computes the sum of the digits of a number -/
def digit_sum (n : FiveDigitNumber) : Nat :=
  n.sum

/-- Compares two five-digit numbers -/
def is_greater (a b : FiveDigitNumber) : Prop :=
  a.foldl (fun acc d => acc * 10 + d) 0 > b.foldl (fun acc d => acc * 10 + d) 0

theorem largest_five_digit_number_with_product_180 :
  ∃ (M : FiveDigitNumber),
    is_valid_five_digit_number M ∧
    digit_product M = 180 ∧
    (∀ (N : FiveDigitNumber), is_valid_five_digit_number N → digit_product N = 180 → is_greater M N) ∧
    digit_sum M = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_number_with_product_180_l781_78147


namespace NUMINAMATH_CALUDE_a_collinear_b_l781_78173

/-- Two 2D vectors are collinear if and only if their cross product is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The vector a -/
def a : ℝ × ℝ := (1, 2)

/-- The vector b -/
def b : ℝ × ℝ := (-1, -2)

/-- Proof that vectors a and b are collinear -/
theorem a_collinear_b : collinear a b := by
  sorry

end NUMINAMATH_CALUDE_a_collinear_b_l781_78173


namespace NUMINAMATH_CALUDE_soccer_club_girls_l781_78142

theorem soccer_club_girls (total_members : ℕ) (meeting_attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : meeting_attendance = 18)
  (h3 : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧ 
    boys + girls / 3 = meeting_attendance) :
  ∃ (girls : ℕ), girls = 18 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧ 
                   boys + girls / 3 = meeting_attendance :=
by sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l781_78142


namespace NUMINAMATH_CALUDE_kylie_jewelry_beads_l781_78195

/-- The number of beads Kylie uses in total to make her jewelry over the week -/
def total_beads : ℕ :=
  let necklace_beads := 20
  let bracelet_beads := 10
  let earring_beads := 5
  let anklet_beads := 8
  let ring_beads := 7
  let monday_necklaces := 10
  let tuesday_necklaces := 2
  let wednesday_bracelets := 5
  let thursday_earrings := 3
  let friday_anklets := 4
  let friday_rings := 6
  (necklace_beads * (monday_necklaces + tuesday_necklaces)) +
  (bracelet_beads * wednesday_bracelets) +
  (earring_beads * thursday_earrings) +
  (anklet_beads * friday_anklets) +
  (ring_beads * friday_rings)

theorem kylie_jewelry_beads : total_beads = 379 := by
  sorry

end NUMINAMATH_CALUDE_kylie_jewelry_beads_l781_78195


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l781_78100

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop := sorry

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  Midpoint E B C →
  PointOnSegment D A C →
  Length A C = 1 →
  Angle B A C = π / 3 →  -- 60°
  Angle A B C = 5 * π / 9 →  -- 100°
  Angle A C B = π / 9 →  -- 20°
  Angle D E C = 4 * π / 9 →  -- 80°
  TriangleArea A B C + 2 * TriangleArea C D E = Real.sqrt 3 / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l781_78100


namespace NUMINAMATH_CALUDE_rhombus_symmetry_proposition_l781_78182

-- Define the set of all rhombuses
variable (Rhombus : Type)

-- Define the property of having central symmetry
variable (has_central_symmetry : Rhombus → Prop)

-- Define the universal quantifier proposition
def universal_proposition : Prop := ∀ r : Rhombus, has_central_symmetry r

-- Define the negation of the proposition
def negation_proposition : Prop := ∃ r : Rhombus, ¬has_central_symmetry r

-- Theorem stating that the original proposition is a universal quantifier
-- and its negation is an existential quantifier with negated property
theorem rhombus_symmetry_proposition :
  (universal_proposition Rhombus has_central_symmetry) ∧
  (negation_proposition Rhombus has_central_symmetry) :=
sorry

end NUMINAMATH_CALUDE_rhombus_symmetry_proposition_l781_78182


namespace NUMINAMATH_CALUDE_two_digit_seven_times_sum_of_digits_l781_78176

theorem two_digit_seven_times_sum_of_digits : 
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 10 ≤ n ∧ n < 100 ∧ n = 7 * (n / 10 + n % 10)) ∧ 
    Finset.card s = 4) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_seven_times_sum_of_digits_l781_78176


namespace NUMINAMATH_CALUDE_inequality_proof_l781_78179

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h_sum : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l781_78179


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l781_78140

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l781_78140
