import Mathlib

namespace NUMINAMATH_CALUDE_simplify_square_roots_l2542_254239

theorem simplify_square_roots : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 200 / Real.sqrt 50) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2542_254239


namespace NUMINAMATH_CALUDE_log_expression_evaluation_l2542_254221

theorem log_expression_evaluation :
  2 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (2 * Real.log 3 / Real.log 5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_evaluation_l2542_254221


namespace NUMINAMATH_CALUDE_negative_quarter_and_negative_four_power_l2542_254248

theorem negative_quarter_and_negative_four_power :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_and_negative_four_power_l2542_254248


namespace NUMINAMATH_CALUDE_tree_height_difference_l2542_254298

/-- The height difference between two trees -/
theorem tree_height_difference (pine_height maple_height : ℚ) 
  (h_pine : pine_height = 49/4)
  (h_maple : maple_height = 37/2) :
  maple_height - pine_height = 25/4 := by
  sorry

#eval (37/2 : ℚ) - (49/4 : ℚ)  -- Should output 25/4

end NUMINAMATH_CALUDE_tree_height_difference_l2542_254298


namespace NUMINAMATH_CALUDE_min_value_theorem_l2542_254219

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ ∀ (z : ℝ), z = 1/(x-y) + 9/(x+5*y) → z ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2542_254219


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l2542_254228

theorem sqrt_sum_equation (x : ℝ) 
  (h : Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) : 
  Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l2542_254228


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l2542_254240

-- Define the vector a
def a : ℝ × ℝ := (3, -4)

-- Theorem for the unit vector b parallel to a
theorem unit_vector_parallel_to_a :
  ∃ b : ℝ × ℝ, (b.1 = 3/5 ∧ b.2 = -4/5) ∨ (b.1 = -3/5 ∧ b.2 = 4/5) ∧
  (b.1 * a.1 + b.2 * a.2)^2 = (b.1^2 + b.2^2) * (a.1^2 + a.2^2) ∧
  b.1^2 + b.2^2 = 1 :=
sorry

-- Theorem for the unit vector c perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ c : ℝ × ℝ, (c.1 = 4/5 ∧ c.2 = 3/5) ∨ (c.1 = -4/5 ∧ c.2 = -3/5) ∧
  c.1 * a.1 + c.2 * a.2 = 0 ∧
  c.1^2 + c.2^2 = 1 :=
sorry

-- Theorem for the vector e obtained by rotating a 45° counterclockwise
theorem rotated_vector_e :
  ∃ e : ℝ × ℝ, e.1 = 7 * Real.sqrt 2 / 2 ∧ e.2 = - Real.sqrt 2 / 2 ∧
  e.1^2 + e.2^2 = a.1^2 + a.2^2 ∧
  e.1 * a.1 + e.2 * a.2 = Real.sqrt ((a.1^2 + a.2^2)^2 / 2) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l2542_254240


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2542_254276

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2542_254276


namespace NUMINAMATH_CALUDE_c1_c2_not_collinear_l2542_254293

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem c1_c2_not_collinear (a b : ℝ × ℝ × ℝ) 
  (h1 : a = ⟨-9, 5, 3⟩) 
  (h2 : b = ⟨7, 1, -2⟩) : 
  ¬ ∃ (k : ℝ), 2 • a - b = k • (3 • a + 5 • b) :=
sorry

end NUMINAMATH_CALUDE_c1_c2_not_collinear_l2542_254293


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2542_254241

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol mixture -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 600
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2542_254241


namespace NUMINAMATH_CALUDE_fourth_task_completion_time_l2542_254252

-- Define the start and end times
def start_time : ℕ := 12 * 60  -- 12:00 PM in minutes
def end_time : ℕ := 15 * 60    -- 3:00 PM in minutes

-- Define the number of tasks completed
def num_tasks : ℕ := 3

-- Theorem to prove
theorem fourth_task_completion_time 
  (h1 : end_time - start_time = num_tasks * (end_time - start_time) / num_tasks) -- Tasks are equally time-consuming
  (h2 : (end_time - start_time) % num_tasks = 0) -- Ensures division is exact
  : end_time + (end_time - start_time) / num_tasks = 16 * 60 := -- 4:00 PM in minutes
by
  sorry

end NUMINAMATH_CALUDE_fourth_task_completion_time_l2542_254252


namespace NUMINAMATH_CALUDE_tuition_fee_agreement_percentage_l2542_254285

theorem tuition_fee_agreement_percentage (total_parents : ℕ) (disagree_parents : ℕ) 
  (h1 : total_parents = 800) (h2 : disagree_parents = 640) : 
  (total_parents - disagree_parents : ℝ) / total_parents * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tuition_fee_agreement_percentage_l2542_254285


namespace NUMINAMATH_CALUDE_jaco_total_payment_l2542_254292

/-- Calculates the total amount a customer pays given item prices and a discount policy. -/
def calculateTotalWithDiscount (shoePrice sockPrice bagPrice : ℚ) : ℚ :=
  let totalBeforeDiscount := shoePrice + 2 * sockPrice + bagPrice
  let discountableAmount := max (totalBeforeDiscount - 100) 0
  let discount := discountableAmount * (1 / 10)
  totalBeforeDiscount - discount

/-- Theorem stating that Jaco will pay $118 for his purchases. -/
theorem jaco_total_payment :
  calculateTotalWithDiscount 74 2 42 = 118 := by
  sorry

#eval calculateTotalWithDiscount 74 2 42

end NUMINAMATH_CALUDE_jaco_total_payment_l2542_254292


namespace NUMINAMATH_CALUDE_yunas_math_score_l2542_254212

theorem yunas_math_score (score1 score2 : ℝ) (h1 : (score1 + score2) / 2 = 92) 
  (h2 : ∃ (score3 : ℝ), (score1 + score2 + score3) / 3 = 94) : 
  ∃ (score3 : ℝ), score3 = 98 ∧ (score1 + score2 + score3) / 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_yunas_math_score_l2542_254212


namespace NUMINAMATH_CALUDE_three_boys_ages_exist_l2542_254202

theorem three_boys_ages_exist : ∃ (A B C : ℝ), 
  A + B + C = 29.5 ∧ 
  C = 11.3 ∧ 
  (A = 2 * B ∨ B = 2 * C ∨ A = 2 * C) ∧
  A > 0 ∧ B > 0 ∧ C > 0 := by
  sorry

end NUMINAMATH_CALUDE_three_boys_ages_exist_l2542_254202


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_relation_l2542_254275

theorem sqrt_equality_implies_relation (a b c : ℕ+) :
  (a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ) ≥ 0 →
  Real.sqrt ((a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ)) = a.val - Real.sqrt ((b.val : ℝ) / (c.val : ℝ)) →
  b = a ^ 2 * c := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_relation_l2542_254275


namespace NUMINAMATH_CALUDE_pythagorean_cube_equation_solutions_l2542_254257

theorem pythagorean_cube_equation_solutions :
  ∀ a b c : ℕ+,
    a^2 + b^2 = c^2 ∧ a^3 + b^3 + 1 = (c - 1)^3 →
    ((a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_cube_equation_solutions_l2542_254257


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2542_254254

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- State the theorem
theorem hyperbola_vertex_distance :
  ∃ (x y : ℝ), hyperbola x y → 
    (let vertex_distance := 2 * (Real.sqrt 144);
     vertex_distance = 24) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2542_254254


namespace NUMINAMATH_CALUDE_equation_solution_l2542_254290

theorem equation_solution :
  ∀ x : ℝ, x ≠ 0 ∧ 8*x + 3 ≠ 0 ∧ 7*x - 3 ≠ 0 →
    (2 + 5/(4*x) - 15/(4*x*(8*x+3)) = 2*(7*x+1)/(7*x-3)) ↔ x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2542_254290


namespace NUMINAMATH_CALUDE_inequality_proof_l2542_254201

theorem inequality_proof (x : ℝ) : (x^2 - 16) / (x^2 + 10*x + 25) < 0 ↔ -4 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2542_254201


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_is_zero_l2542_254245

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 - 3*z = 8 - 6*i

-- Theorem statement
theorem sum_of_imaginary_parts_is_zero :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ 
  z₁ ≠ z₂ ∧ (z₁.im + z₂.im = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_is_zero_l2542_254245


namespace NUMINAMATH_CALUDE_correct_growth_rate_l2542_254265

/-- The average annual growth rate of book borrowing from 2020 to 2022 -/
def average_growth_rate : ℝ := 0.2

/-- The number of books borrowed in 2020 -/
def books_2020 : ℕ := 7500

/-- The number of books borrowed in 2022 -/
def books_2022 : ℕ := 10800

/-- The theorem stating that the average annual growth rate is correct -/
theorem correct_growth_rate : 
  books_2020 * (1 + average_growth_rate)^2 = books_2022 := by
  sorry

end NUMINAMATH_CALUDE_correct_growth_rate_l2542_254265


namespace NUMINAMATH_CALUDE_sandy_parentheses_problem_l2542_254235

theorem sandy_parentheses_problem (p q r s : ℤ) (h1 : p = 2) (h2 : q = 4) (h3 : r = 6) (h4 : s = 8) :
  ∃ t : ℤ, p + (q - (r + (s - t))) = p + q - r + s - 10 ∧ t = 8 := by
sorry

end NUMINAMATH_CALUDE_sandy_parentheses_problem_l2542_254235


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2542_254225

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with households of different income levels -/
structure Community where
  totalHouseholds : Nat
  highIncomeHouseholds : Nat
  middleIncomeHouseholds : Nat
  lowIncomeHouseholds : Nat

/-- Represents a group of senior soccer players -/
structure SoccerTeam where
  totalPlayers : Nat

/-- Determines the best sampling method for a given community and sample size -/
def bestSamplingMethodForCommunity (c : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a given soccer team and sample size -/
def bestSamplingMethodForSoccerTeam (t : SoccerTeam) (sampleSize : Nat) : SamplingMethod :=
  sorry

theorem correct_sampling_methods 
  (community : Community)
  (soccerTeam : SoccerTeam)
  (communitySampleSize : Nat)
  (soccerSampleSize : Nat)
  (h1 : community.totalHouseholds = 500)
  (h2 : community.highIncomeHouseholds = 125)
  (h3 : community.middleIncomeHouseholds = 280)
  (h4 : community.lowIncomeHouseholds = 95)
  (h5 : communitySampleSize = 100)
  (h6 : soccerTeam.totalPlayers = 12)
  (h7 : soccerSampleSize = 3) :
  bestSamplingMethodForCommunity community communitySampleSize = SamplingMethod.Stratified ∧
  bestSamplingMethodForSoccerTeam soccerTeam soccerSampleSize = SamplingMethod.Random :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2542_254225


namespace NUMINAMATH_CALUDE_train_length_l2542_254227

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : Real) (pass_time : Real) (platform_length : Real) :
  train_speed = 45 * (5/18) ∧ 
  pass_time = 44 ∧ 
  platform_length = 190 →
  (train_speed * pass_time) - platform_length = 360 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l2542_254227


namespace NUMINAMATH_CALUDE_max_n_for_300_triangles_max_n_is_102_l2542_254282

/-- Represents a convex polygon with interior points -/
structure ConvexPolygon where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ -- number of interior points
  no_collinear : Prop -- property that no three points are collinear

/-- The number of triangles formed in a convex polygon with interior points -/
def num_triangles (p : ConvexPolygon) : ℕ :=
  p.n + p.interior_points + 198

/-- Theorem stating the maximum value of n for which no more than 300 triangles can be formed -/
theorem max_n_for_300_triangles (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  p.n ≤ 102 := by
  sorry

/-- The maximum value of n is indeed 102 -/
theorem max_n_is_102 (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  ∃ (q : ConvexPolygon), q.n = 102 ∧ q.interior_points = 100 ∧ num_triangles q = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_n_for_300_triangles_max_n_is_102_l2542_254282


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l2542_254213

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 2/3) * (x - 5/6) + (x - 2/3) * (x - 2/3) - 1 = 0 →
  x = -1/12 ∨ x = 4/3 ∧ 
  -1/12 < 4/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l2542_254213


namespace NUMINAMATH_CALUDE_carpet_dimensions_l2542_254271

/-- Represents a rectangular carpet with integral side lengths -/
structure Carpet where
  width : ℕ
  length : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a carpet fits perfectly in a room (diagonally) -/
def fitsInRoom (c : Carpet) (r : Room) : Prop :=
  c.width ^ 2 + c.length ^ 2 = r.width ^ 2 + r.length ^ 2

theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 38 →
    r2.width = 50 →
    r1.length = r2.length →
    fitsInRoom c r1 →
    fitsInRoom c r2 →
    c.width = 25 ∧ c.length = 50 := by
  sorry


end NUMINAMATH_CALUDE_carpet_dimensions_l2542_254271


namespace NUMINAMATH_CALUDE_line_point_k_value_l2542_254286

/-- Given a line containing points (7, 10), (1, k), and (-5, 3), prove that k = 6.5 -/
theorem line_point_k_value : ∀ k : ℝ,
  (∃ (line : Set (ℝ × ℝ)),
    (7, 10) ∈ line ∧ (1, k) ∈ line ∧ (-5, 3) ∈ line ∧
    (∀ p q r : ℝ × ℝ, p ∈ line → q ∈ line → r ∈ line →
      (p.2 - q.2) * (q.1 - r.1) = (q.2 - r.2) * (p.1 - q.1))) →
  k = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_line_point_k_value_l2542_254286


namespace NUMINAMATH_CALUDE_xiaogong_speed_l2542_254246

/-- The speed of Xiaogong in meters per minute -/
def v_x : ℝ := 28

/-- The speed of Dachen in meters per minute -/
def v_d : ℝ := v_x + 20

/-- The total distance between points A and B in meters -/
def total_distance : ℝ := 1200

/-- The time Dachen walks before meeting Xiaogong, in minutes -/
def t_d : ℝ := 18

/-- The time Xiaogong walks before meeting Dachen, in minutes -/
def t_x : ℝ := 12

theorem xiaogong_speed :
  v_x * t_x + v_d * t_d = total_distance ∧
  v_d = v_x + 20 →
  v_x = 28 := by sorry

end NUMINAMATH_CALUDE_xiaogong_speed_l2542_254246


namespace NUMINAMATH_CALUDE_rooster_weight_unit_l2542_254211

/-- Represents units of mass measurement -/
inductive MassUnit
  | Kilogram
  | Ton
  | Gram

/-- The weight of a rooster in some unit -/
def roosterWeight : ℝ := 3

/-- Predicate to determine if a unit is appropriate for measuring rooster weight -/
def isAppropriateUnit (unit : MassUnit) : Prop :=
  match unit with
  | MassUnit.Kilogram => True
  | _ => False

theorem rooster_weight_unit :
  isAppropriateUnit MassUnit.Kilogram :=
sorry

end NUMINAMATH_CALUDE_rooster_weight_unit_l2542_254211


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l2542_254236

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem gcd_138_18_is_6 : Nat.gcd 138 18 = 6 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 138 < m ∧ m < 150 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_138_18_is_6_exists_no_greater_main_result_l2542_254236


namespace NUMINAMATH_CALUDE_inverse_101_mod_102_l2542_254206

theorem inverse_101_mod_102 : (101⁻¹ : ZMod 102) = 101 := by sorry

end NUMINAMATH_CALUDE_inverse_101_mod_102_l2542_254206


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2542_254229

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 3 = 16)
  (h_sum : a 3 + a 4 = 24) :
  a 5 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2542_254229


namespace NUMINAMATH_CALUDE_group_size_proof_l2542_254253

theorem group_size_proof (total : ℕ) (older : ℕ) (prob : ℚ) 
  (h1 : older = 90)
  (h2 : prob = 40/130)
  (h3 : prob = (total - older) / total) :
  total = 130 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2542_254253


namespace NUMINAMATH_CALUDE_product_plus_one_equals_square_l2542_254222

theorem product_plus_one_equals_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_equals_square_l2542_254222


namespace NUMINAMATH_CALUDE_smallest_prime_eight_less_than_square_l2542_254223

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_eight_less_than_square : 
  (∀ n : ℕ, n > 0 ∧ is_prime n ∧ (∃ m : ℕ, n = m * m - 8) → n ≥ 17) ∧ 
  (17 > 0 ∧ is_prime 17 ∧ ∃ m : ℕ, 17 = m * m - 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_eight_less_than_square_l2542_254223


namespace NUMINAMATH_CALUDE_ball_problem_l2542_254288

/-- Represents the contents of a box with balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Represents the random variable X (number of red balls drawn from box A) -/
inductive X
  | zero
  | one
  | two

def box_A : Box := { white := 2, red := 2 }
def box_B : Box := { white := 1, red := 3 }

def prob_X (x : X) : ℚ :=
  match x with
  | X.zero => 1/6
  | X.one => 2/3
  | X.two => 1/6

def expected_X : ℚ := 1

def prob_red_from_B : ℚ := 2/3

theorem ball_problem :
  (∀ x : X, prob_X x > 0) ∧ 
  (prob_X X.zero + prob_X X.one + prob_X X.two = 1) ∧
  (0 * prob_X X.zero + 1 * prob_X X.one + 2 * prob_X X.two = expected_X) ∧
  prob_red_from_B = 2/3 := by sorry

end NUMINAMATH_CALUDE_ball_problem_l2542_254288


namespace NUMINAMATH_CALUDE_cos_squared_sum_range_l2542_254243

theorem cos_squared_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  ∃ (x : ℝ), x ∈ Set.Icc (14/9 : ℝ) 2 ∧
  x = (Real.cos α)^2 + (Real.cos β)^2 ∧
  ∀ (y : ℝ), y = (Real.cos α)^2 + (Real.cos β)^2 → y ∈ Set.Icc (14/9 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_cos_squared_sum_range_l2542_254243


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2542_254215

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + 3*m + 2) (m^2 - m - 6)

theorem pure_imaginary_condition (m : ℝ) :
  IsPureImaginary (z m) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2542_254215


namespace NUMINAMATH_CALUDE_sin_three_zeros_l2542_254220

/-- Given a function f(x) = sin(ωx + π/3) with ω > 0, if f has exactly 3 zeros
    in the interval [0, 2π/3], then 4 ≤ ω < 11/2 -/
theorem sin_three_zeros (ω : ℝ) (h₁ : ω > 0) :
  (∃! (zeros : Finset ℝ), zeros.card = 3 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧
      Real.sin (ω * x + Real.pi / 3) = 0)) →
  4 ≤ ω ∧ ω < 11 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_three_zeros_l2542_254220


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2542_254250

/-- Represents a cylindrical water tank. -/
structure WaterTank where
  capacity : ℝ
  initialFill : ℝ

/-- Condition that the tank is 1/6 full initially. -/
def isInitiallySixthFull (tank : WaterTank) : Prop :=
  tank.initialFill / tank.capacity = 1 / 6

/-- Condition that the tank becomes 1/3 full after adding 5 liters. -/
def isThirdFullAfterAddingFive (tank : WaterTank) : Prop :=
  (tank.initialFill + 5) / tank.capacity = 1 / 3

/-- Theorem stating that if a water tank satisfies the given conditions, its capacity is 30 liters. -/
theorem water_tank_capacity
    (tank : WaterTank)
    (h1 : isInitiallySixthFull tank)
    (h2 : isThirdFullAfterAddingFive tank) :
    tank.capacity = 30 := by
  sorry


end NUMINAMATH_CALUDE_water_tank_capacity_l2542_254250


namespace NUMINAMATH_CALUDE_min_workers_proof_l2542_254231

/-- The minimum number of workers in team A that satisfies the given conditions -/
def min_workers_A : ℕ := 153

/-- The number of workers team B transfers to team A -/
def workers_transferred : ℕ := (11 * min_workers_A - 1620) / 7

theorem min_workers_proof :
  (∀ a b : ℕ,
    (a ≥ min_workers_A) →
    (b + 90 = 2 * (a - 90)) →
    (a + workers_transferred = 6 * (b - workers_transferred)) →
    (workers_transferred > 0) →
    (∃ k : ℕ, a + 1 = 7 * k)) →
  (∀ a : ℕ,
    (a < min_workers_A) →
    (¬∃ b : ℕ,
      (b + 90 = 2 * (a - 90)) ∧
      (a + workers_transferred = 6 * (b - workers_transferred)) ∧
      (workers_transferred > 0))) :=
by sorry

end NUMINAMATH_CALUDE_min_workers_proof_l2542_254231


namespace NUMINAMATH_CALUDE_bike_speed_calculation_l2542_254277

/-- Proves that given the conditions, the bike speed is 15 km/h -/
theorem bike_speed_calculation (distance : ℝ) (car_speed_multiplier : ℝ) (time_difference : ℝ) :
  distance = 15 →
  car_speed_multiplier = 4 →
  time_difference = 45 / 60 →
  ∃ (bike_speed : ℝ), 
    bike_speed > 0 ∧
    distance / bike_speed - distance / (car_speed_multiplier * bike_speed) = time_difference ∧
    bike_speed = 15 :=
by sorry

end NUMINAMATH_CALUDE_bike_speed_calculation_l2542_254277


namespace NUMINAMATH_CALUDE_max_additional_license_plates_l2542_254256

def initial_first_set : Finset Char := {'C', 'H', 'L', 'P', 'R'}
def initial_second_set : Finset Char := {'A', 'I', 'O'}
def initial_third_set : Finset Char := {'D', 'M', 'N', 'T'}

def initial_combinations : ℕ := initial_first_set.card * initial_second_set.card * initial_third_set.card

def max_additional_combinations : ℕ := 
  (initial_first_set.card * (initial_second_set.card + 2) * initial_third_set.card) - initial_combinations

theorem max_additional_license_plates : max_additional_combinations = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_license_plates_l2542_254256


namespace NUMINAMATH_CALUDE_current_speed_l2542_254295

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 16) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2542_254295


namespace NUMINAMATH_CALUDE_specific_building_occupancy_l2542_254237

/-- Represents the building structure and occupancy --/
structure Building where
  floors : Nat
  first_floor_apartments : Nat
  common_difference : Nat
  one_bedroom_occupancy : Nat
  two_bedroom_occupancy : Nat
  three_bedroom_occupancy : Nat

/-- Calculates the total number of people in the building --/
def total_occupancy (b : Building) : Nat :=
  let last_floor_apartments := b.first_floor_apartments + (b.floors - 1) * b.common_difference
  let total_apartments := (b.floors * (b.first_floor_apartments + last_floor_apartments)) / 2
  let apartments_per_type := total_apartments / 3
  apartments_per_type * (b.one_bedroom_occupancy + b.two_bedroom_occupancy + b.three_bedroom_occupancy)

/-- Theorem stating the total occupancy of the specific building --/
theorem specific_building_occupancy :
  let b : Building := {
    floors := 25,
    first_floor_apartments := 3,
    common_difference := 2,
    one_bedroom_occupancy := 2,
    two_bedroom_occupancy := 4,
    three_bedroom_occupancy := 5
  }
  total_occupancy b = 2475 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_occupancy_l2542_254237


namespace NUMINAMATH_CALUDE_probability_two_red_marbles_l2542_254284

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles. -/
theorem probability_two_red_marbles (red : ℕ) (green : ℕ) (total : ℕ) :
  red = 2 →
  green = 3 →
  total = red + green →
  (red / total) * ((red - 1) / (total - 1)) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_marbles_l2542_254284


namespace NUMINAMATH_CALUDE_salary_changes_l2542_254281

theorem salary_changes (initial_salary : ℝ) : 
  initial_salary = 2500 → 
  (initial_salary * (1 + 0.15) * (1 - 0.10)) = 2587.50 := by
sorry

end NUMINAMATH_CALUDE_salary_changes_l2542_254281


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2542_254216

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 8) : 
  a^2 + b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2542_254216


namespace NUMINAMATH_CALUDE_certain_number_proof_l2542_254289

theorem certain_number_proof (x : ℝ) : 0.60 * x = (4 / 5) * 25 + 4 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2542_254289


namespace NUMINAMATH_CALUDE_larry_channels_l2542_254270

/-- The number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed1 removed2 added1 added2 added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l2542_254270


namespace NUMINAMATH_CALUDE_convincing_statement_l2542_254210

-- Define the types of people
inductive Person
| Knight
| Knave

-- Define the wealth status of knights
inductive KnightWealth
| Poor
| Rich

-- Define a function to determine if a person tells the truth
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Knave => False

-- Define the statement "I am not a poor knight"
def statement (p : Person) (w : KnightWealth) : Prop :=
  p = Person.Knight ∧ w ≠ KnightWealth.Poor

-- Theorem to prove
theorem convincing_statement 
  (p : Person) (w : KnightWealth) : 
  tellsTruth p → statement p w → (p = Person.Knight ∧ w = KnightWealth.Rich) :=
by
  sorry


end NUMINAMATH_CALUDE_convincing_statement_l2542_254210


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2542_254208

theorem polynomial_factorization (a b : ℝ) :
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3 = -3 * a * b * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2542_254208


namespace NUMINAMATH_CALUDE_noelle_homework_assignments_l2542_254279

/-- The number of homework points Noelle needs to earn -/
def total_points : ℕ := 30

/-- The number of points in the first tier (1 assignment per point) -/
def first_tier : ℕ := 5

/-- The number of points in the second tier (2 assignments per point) -/
def second_tier : ℕ := 10

/-- The number of assignments required for each point in the first tier -/
def first_tier_assignments : ℕ := 1

/-- The number of assignments required for each point in the second tier -/
def second_tier_assignments : ℕ := 2

/-- The number of assignments required for each point in the third tier -/
def third_tier_assignments : ℕ := 3

/-- The total number of homework assignments Noelle needs to complete -/
def total_assignments : ℕ := 
  first_tier * first_tier_assignments + 
  second_tier * second_tier_assignments + 
  (total_points - first_tier - second_tier) * third_tier_assignments

theorem noelle_homework_assignments : total_assignments = 70 := by
  sorry

end NUMINAMATH_CALUDE_noelle_homework_assignments_l2542_254279


namespace NUMINAMATH_CALUDE_teachers_combined_age_l2542_254297

theorem teachers_combined_age
  (num_students : ℕ)
  (student_avg_age : ℚ)
  (num_teachers : ℕ)
  (total_avg_age : ℚ)
  (h1 : num_students = 30)
  (h2 : student_avg_age = 18)
  (h3 : num_teachers = 2)
  (h4 : total_avg_age = 19) :
  (num_students + num_teachers) * total_avg_age -
  (num_students * student_avg_age) = 68 := by
sorry

end NUMINAMATH_CALUDE_teachers_combined_age_l2542_254297


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2542_254269

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d * (Real.sqrt 2 - 1) / 2
  (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi * (24 * (Real.sqrt 2 - 1))^3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2542_254269


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2542_254263

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * 0.95 = 1.102 → x = 16 := by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2542_254263


namespace NUMINAMATH_CALUDE_no_valid_combination_l2542_254262

def nickel : ℕ := 5
def dime : ℕ := 10
def half_dollar : ℕ := 50

def is_valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c = nickel ∨ c = dime ∨ c = half_dollar) ∧
  coins.length = 6 ∧
  coins.sum = 90

theorem no_valid_combination : ¬ ∃ (coins : List ℕ), is_valid_combination coins := by
  sorry

end NUMINAMATH_CALUDE_no_valid_combination_l2542_254262


namespace NUMINAMATH_CALUDE_fib_50_mod_5_l2542_254273

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the remainder function
def remainder (a b : ℕ) : ℕ := a % b

-- Theorem statement
theorem fib_50_mod_5 : remainder (fib 50) 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_50_mod_5_l2542_254273


namespace NUMINAMATH_CALUDE_equation_solution_l2542_254272

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3)}

theorem equation_solution :
  {(a, b, c) : ℕ × ℕ × ℕ | (c - 1) * (a * b - b - a) = a + b - 2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2542_254272


namespace NUMINAMATH_CALUDE_union_and_intersection_conditions_l2542_254209

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m + 3}

theorem union_and_intersection_conditions (m : ℝ) :
  (A ∪ B m = A ↔ m ∈ Set.Ioi (-2) ∪ Set.Iio (-1/2)) ∧
  (A ∩ B m ≠ ∅ ↔ m ∈ Set.Ioo (-2) 1) := by
  sorry

end NUMINAMATH_CALUDE_union_and_intersection_conditions_l2542_254209


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_l2542_254204

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : n = 500 ∧ x = 0.1) :
  ∃ k : ℕ, k = 45 ∧
  ∀ j : ℕ, j ≤ n → (n.choose k) * x^k ≥ (n.choose j) * x^j :=
sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_l2542_254204


namespace NUMINAMATH_CALUDE_cos_double_angle_l2542_254278

theorem cos_double_angle (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((Real.cos α)^2 + (1 / 4)^2) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_l2542_254278


namespace NUMINAMATH_CALUDE_brett_marbles_difference_l2542_254283

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := 6

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := 5 * red_marbles

/-- The difference between blue and red marbles -/
def marble_difference : ℕ := blue_marbles - red_marbles

theorem brett_marbles_difference : marble_difference = 24 := by
  sorry

end NUMINAMATH_CALUDE_brett_marbles_difference_l2542_254283


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l2542_254249

theorem football_lineup_combinations (total_players : ℕ) 
  (offensive_linemen : ℕ) (running_backs : ℕ) : 
  total_players = 12 → offensive_linemen = 3 → running_backs = 4 →
  (offensive_linemen * running_backs * (total_players - 2) * (total_players - 3) = 1080) := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l2542_254249


namespace NUMINAMATH_CALUDE_rotate_5_plus_2i_l2542_254230

/-- Rotates a complex number by 90 degrees counter-clockwise around the origin -/
def rotate90 (z : ℂ) : ℂ := z * Complex.I

/-- The result of rotating 5 + 2i by 90 degrees counter-clockwise around the origin -/
theorem rotate_5_plus_2i : rotate90 (5 + 2*Complex.I) = -2 + 5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotate_5_plus_2i_l2542_254230


namespace NUMINAMATH_CALUDE_parabola_tangent_property_fixed_point_property_l2542_254218

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a point on the axis of the parabola
def point_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Define tangent points
def tangent_points (A B : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ point_on_axis G

-- Define the perpendicular condition
def perpendicular (A M N : ℝ × ℝ) : Prop :=
  (M.1 - A.1) * (N.1 - A.1) + (M.2 - A.2) * (N.2 - A.2) = 0

-- Main theorem
theorem parabola_tangent_property (G : ℝ × ℝ) (A B : ℝ × ℝ) :
  tangent_points A B G → A.1 * B.1 + A.2 * B.2 = -3 :=
sorry

-- Fixed point theorem
theorem fixed_point_property (G A M N : ℝ × ℝ) :
  G.1 = 0 ∧ tangent_points A (2, 1) G ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧ perpendicular A M N →
  ∃ t : ℝ, t * (M.1 - 2) + (1 - t) * (N.1 - 2) = 0 ∧
         t * (M.2 - 5) + (1 - t) * (N.2 - 5) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_property_fixed_point_property_l2542_254218


namespace NUMINAMATH_CALUDE_f_maximum_properties_l2542_254200

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h1 : ∀ x > 0, f x ≤ f x₀) 
  (h2 : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ > 1/9 := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_properties_l2542_254200


namespace NUMINAMATH_CALUDE_unique_A_with_positive_integer_solutions_l2542_254233

/-- A positive single-digit integer -/
def SingleDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The equation x^2 - (2A)x + 10A = 0 has positive integer solutions -/
def HasPositiveIntegerSolutions (A : SingleDigit) : Prop :=
  ∃ x : ℕ+, x^2 - (2 * A.val) * x + 10 * A.val = 0

/-- There exists exactly one positive single-digit integer A such that
    the equation x^2 - (2A)x + 10A = 0 has positive integer solutions -/
theorem unique_A_with_positive_integer_solutions :
  ∃! (A : SingleDigit), HasPositiveIntegerSolutions A :=
sorry

end NUMINAMATH_CALUDE_unique_A_with_positive_integer_solutions_l2542_254233


namespace NUMINAMATH_CALUDE_fishermen_distribution_l2542_254287

theorem fishermen_distribution (x y z : ℕ) : 
  x + y + z = 16 →
  13 * x + 5 * y + 4 * z = 113 →
  x = 5 ∧ y = 4 ∧ z = 7 := by
sorry

end NUMINAMATH_CALUDE_fishermen_distribution_l2542_254287


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2542_254242

theorem complex_equation_solution (z : ℂ) : (2 + z) / (2 - z) = I → z = 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2542_254242


namespace NUMINAMATH_CALUDE_sum_of_special_sequence_l2542_254264

theorem sum_of_special_sequence (a b c d : ℤ) : 
  (Even a) → (Even b) → (Even c) → (Even d) →
  (0 < a) → (a < b) → (b < c) → (c < d) →
  (d - a = 90) →
  (∃ m : ℤ, b - a = m ∧ c - b = m) →  -- arithmetic sequence condition
  (c * c = b * d) →  -- geometric sequence condition
  (a + b + c + d = 194) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_sequence_l2542_254264


namespace NUMINAMATH_CALUDE_thousandth_coprime_to_105_l2542_254224

/-- The sequence of positive integers coprime to 105, arranged in ascending order -/
def coprimeSeq : ℕ → ℕ := sorry

/-- The 1000th term of the sequence is 2186 -/
theorem thousandth_coprime_to_105 : coprimeSeq 1000 = 2186 := by sorry

end NUMINAMATH_CALUDE_thousandth_coprime_to_105_l2542_254224


namespace NUMINAMATH_CALUDE_carls_marbles_l2542_254203

theorem carls_marbles (initial_marbles : ℕ) : 
  (initial_marbles / 2 + 10 + 25 = 41) → initial_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_carls_marbles_l2542_254203


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l2542_254214

def f (a b x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

theorem max_value_of_symmetric_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-4 - x)) →
  (∃ x : ℝ, f a b x = 0 ∧ f a b (-4 - x) = 0) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x ≤ m ∧ ∃ x₀ : ℝ, f a b x₀ = m) →
  (∃ m : ℝ, (∀ x : ℝ, f a b x ≤ m) ∧ (∃ x₀ : ℝ, f a b x₀ = m) ∧ m = 16) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l2542_254214


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l2542_254244

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 17 :=
sorry

theorem max_value_achievable : 
  ∃ (x y : ℝ), (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) = Real.sqrt 17 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l2542_254244


namespace NUMINAMATH_CALUDE_work_completion_time_l2542_254266

/-- The time taken for three workers to complete a task together,
    given their individual completion times. -/
theorem work_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 10)
  (hy : y_time = 15)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 60 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2542_254266


namespace NUMINAMATH_CALUDE_book_sales_total_l2542_254261

/-- Calculates the total amount received from book sales given the number of books and their prices -/
def totalAmountReceived (fictionBooks nonFictionBooks childrensBooks : ℕ)
                        (fictionPrice nonFictionPrice childrensPrice : ℚ)
                        (fictionSoldRatio nonFictionSoldRatio childrensSoldRatio : ℚ) : ℚ :=
  (fictionBooks : ℚ) * fictionSoldRatio * fictionPrice +
  (nonFictionBooks : ℚ) * nonFictionSoldRatio * nonFictionPrice +
  (childrensBooks : ℚ) * childrensSoldRatio * childrensPrice

/-- The total amount received from book sales is $799 -/
theorem book_sales_total : 
  totalAmountReceived 60 84 42 5 7 3 (3/4) (5/6) (2/3) = 799 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_total_l2542_254261


namespace NUMINAMATH_CALUDE_bisection_method_step_l2542_254280

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

#check bisection_method_step

end NUMINAMATH_CALUDE_bisection_method_step_l2542_254280


namespace NUMINAMATH_CALUDE_calculation_difference_l2542_254291

def harry_calculation : ℤ := 12 - (3 + 4 * 2)

def terry_calculation : ℤ :=
  let step1 := 12 - 3
  let step2 := step1 + 4
  step2 * 2

theorem calculation_difference :
  harry_calculation - terry_calculation = -25 := by sorry

end NUMINAMATH_CALUDE_calculation_difference_l2542_254291


namespace NUMINAMATH_CALUDE_girls_fraction_is_half_l2542_254294

/-- Given a class of students, prove that the fraction of the number of girls
    that equals 1/3 of the total number of students is 1/2, when the ratio of
    boys to girls is 1/2. -/
theorem girls_fraction_is_half (T G B : ℚ) : 
  T > 0 → G > 0 → B > 0 →
  T = G + B →
  B / G = 1 / 2 →
  ∃ (f : ℚ), f * G = (1 / 3) * T ∧ f = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_half_l2542_254294


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2542_254205

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 24 18 = 87 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2542_254205


namespace NUMINAMATH_CALUDE_gloves_with_pair_count_l2542_254226

-- Define the number of glove pairs
def num_pairs : ℕ := 4

-- Define the total number of gloves
def total_gloves : ℕ := 2 * num_pairs

-- Define the number of gloves to pick
def gloves_to_pick : ℕ := 4

-- Theorem statement
theorem gloves_with_pair_count :
  (Nat.choose total_gloves gloves_to_pick) - (2^num_pairs) = 54 := by
  sorry

end NUMINAMATH_CALUDE_gloves_with_pair_count_l2542_254226


namespace NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_range_of_m_l2542_254258

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement 1
theorem solution_set_inequality (a : ℝ) :
  (∀ x, f a x ≤ 0 ↔ x ∈ Set.Icc 1 2) →
  (∀ x, f a x ≥ 1 - x^2 ↔ x ∈ Set.Iic (1/2) ∪ Set.Ici 1) :=
sorry

-- Statement 2
theorem range_of_a :
  (∀ a, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
    a ∈ Set.Iic (1/3)) :=
sorry

-- Statement 3
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  (∀ m, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f (-3) x₁ = g m x₂) →
    m ∈ Set.Ioo 7 (31/4)) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_range_of_m_l2542_254258


namespace NUMINAMATH_CALUDE_range_of_f_l2542_254274

def f (x : ℝ) : ℝ := -x^2 + 2*x - 3

theorem range_of_f :
  ∃ (a b : ℝ), a = -3 ∧ b = -2 ∧
  (∀ x ∈ Set.Icc 0 2, a ≤ f x ∧ f x ≤ b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc 0 2, f x = y) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2542_254274


namespace NUMINAMATH_CALUDE_max_value_sum_l2542_254232

theorem max_value_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 2024) : 
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 2024 → 
      x * z + 3 * y * z + 4 * z * w + 8 * z * v ≤ N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 2024) ∧
    (N = a_N * c_N + 3 * b_N * c_N + 4 * c_N * d_N + 8 * c_N * e_N) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 48 + 3028 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l2542_254232


namespace NUMINAMATH_CALUDE_dot_product_constant_l2542_254267

/-- Definition of ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of curve E -/
def curve_E (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of point D -/
def point_D : ℝ × ℝ := (-2, 0)

/-- Definition of line passing through origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- Theorem: Dot product of DA and DB is constant -/
theorem dot_product_constant (A B : ℝ × ℝ) :
  curve_E A.1 A.2 →
  curve_E B.1 B.2 →
  (∃ k : ℝ, line_through_origin k A.1 A.2 ∧ line_through_origin k B.1 B.2) →
  ((A.1 + 2) * (B.1 + 2) + (A.2 * B.2) = 3) :=
sorry

end NUMINAMATH_CALUDE_dot_product_constant_l2542_254267


namespace NUMINAMATH_CALUDE_special_function_sum_l2542_254255

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Main theorem -/
theorem special_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_prop : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_l2542_254255


namespace NUMINAMATH_CALUDE_unique_solution_cubic_l2542_254238

theorem unique_solution_cubic (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 2 = 0) ↔ b = 7/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_l2542_254238


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l2542_254217

theorem equidistant_point_on_y_axis : 
  ∃ y : ℚ, y = 13/6 ∧ 
  (∀ (x : ℚ), x = 0 → 
    (x^2 + y^2) = ((x - 2)^2 + (y - 3)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l2542_254217


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l2542_254260

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 3*x^2 - 10

theorem point_P_satisfies_conditions : 
  let x₀ : ℝ := -2
  let y₀ : ℝ := 15
  (x₀ < 0) ∧ 
  (C x₀ = y₀) ∧ 
  (C' x₀ = 2) := by sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l2542_254260


namespace NUMINAMATH_CALUDE_problem_statement_l2542_254268

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2542_254268


namespace NUMINAMATH_CALUDE_sqrt_seven_expressions_l2542_254251

theorem sqrt_seven_expressions (a b : ℝ) 
  (ha : a = Real.sqrt 7 + 2) 
  (hb : b = Real.sqrt 7 - 2) : 
  a^2 * b + b^2 * a = 6 * Real.sqrt 7 ∧ 
  a^2 + a * b + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_expressions_l2542_254251


namespace NUMINAMATH_CALUDE_sally_age_proof_l2542_254207

theorem sally_age_proof (sally_age_five_years_ago : ℕ) : 
  sally_age_five_years_ago = 7 → 
  (sally_age_five_years_ago + 5 + 2 : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sally_age_proof_l2542_254207


namespace NUMINAMATH_CALUDE_mirror_area_l2542_254296

/-- The area of a rectangular mirror fitting exactly inside a frame -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) 
  (h1 : frame_length = 100)
  (h2 : frame_width = 130)
  (h3 : frame_thickness = 15) : 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 7000 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l2542_254296


namespace NUMINAMATH_CALUDE_time_to_plant_trees_l2542_254299

-- Define the rate of planting trees
def trees_per_minute : ℚ := 10 / 3

-- Define the total number of trees to be planted
def total_trees : ℕ := 2500

-- Define the time it takes to plant all trees in hours
def planting_time : ℚ := 12.5

-- Theorem to prove
theorem time_to_plant_trees :
  trees_per_minute * 60 * planting_time = total_trees :=
sorry

end NUMINAMATH_CALUDE_time_to_plant_trees_l2542_254299


namespace NUMINAMATH_CALUDE_marias_apple_sales_l2542_254259

/-- Given Maria's apple sales, prove the amount sold in the second hour -/
theorem marias_apple_sales (first_hour_sales second_hour_sales : ℝ) 
  (h1 : first_hour_sales = 10)
  (h2 : (first_hour_sales + second_hour_sales) / 2 = 6) : 
  second_hour_sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_marias_apple_sales_l2542_254259


namespace NUMINAMATH_CALUDE_root_difference_indeterminate_l2542_254247

/-- A function with the property f(1 + x) = f(1 - x) for all real x -/
def symmetric_around_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

/-- A function has exactly two distinct real roots -/
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y

theorem root_difference_indeterminate (f : ℝ → ℝ) 
  (h1 : symmetric_around_one f) 
  (h2 : has_two_distinct_roots f) : 
  ¬ ∃ d : ℝ, ∀ x y : ℝ, f x = 0 → f y = 0 → x ≠ y → |x - y| = d :=
sorry

end NUMINAMATH_CALUDE_root_difference_indeterminate_l2542_254247


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2542_254234

/-- The probability of winning exactly k games out of n games, given a probability p of winning each game. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of games played -/
def num_games : ℕ := 6

/-- The number of games won -/
def num_wins : ℕ := 4

/-- The probability of winning a single game -/
def win_probability : ℚ := 3/5

theorem grunters_win_probability :
  binomial_probability num_games num_wins win_probability = 4860/15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2542_254234
