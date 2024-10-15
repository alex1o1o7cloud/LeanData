import Mathlib

namespace NUMINAMATH_CALUDE_volume_per_part_l390_39075

/-- Given two rectangular prisms and a number of equal parts filling these prisms,
    calculate the volume of each part. -/
theorem volume_per_part
  (length width height : ℝ)
  (num_prisms num_parts : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_num_prisms : num_prisms = 2)
  (h_num_parts : num_parts = 16) :
  (num_prisms * length * width * height) / num_parts = 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_per_part_l390_39075


namespace NUMINAMATH_CALUDE_intersection_distance_l390_39011

theorem intersection_distance : ∃ (p q : ℕ+), 
  (∀ (d : ℕ+), d ∣ p ∧ d ∣ q → d = 1) ∧ 
  (∃ (x₁ x₂ : ℝ), 
    2 = x₁^2 + 2*x₁ - 2 ∧ 
    2 = x₂^2 + 2*x₂ - 2 ∧ 
    (x₂ - x₁)^2 = 20 ∧
    (x₂ - x₁)^2 * q^2 = p) ∧
  p - q = 19 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l390_39011


namespace NUMINAMATH_CALUDE_mike_tv_hours_l390_39065

-- Define the number of hours Mike watches TV daily
def tv_hours : ℝ := 4

-- Define the number of days in a week Mike plays video games
def gaming_days : ℕ := 3

-- Define the total hours spent on both activities in a week
def total_hours : ℝ := 34

-- Theorem statement
theorem mike_tv_hours :
  -- Condition: On gaming days, Mike plays for half as long as he watches TV
  (gaming_days * (tv_hours / 2) +
  -- Condition: Mike watches TV every day of the week
   7 * tv_hours = total_hours) →
  -- Conclusion: Mike watches TV for 4 hours every day
  tv_hours = 4 := by
sorry

end NUMINAMATH_CALUDE_mike_tv_hours_l390_39065


namespace NUMINAMATH_CALUDE_probability_two_present_one_absent_l390_39026

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 50

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students we are considering -/
def n_students : ℕ := 3

/-- The number of students that should be present -/
def n_present : ℕ := 2

theorem probability_two_present_one_absent :
  (n_students.choose n_present : ℚ) * p_present ^ n_present * p_absent ^ (n_students - n_present) = 1728 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_present_one_absent_l390_39026


namespace NUMINAMATH_CALUDE_xy_eq_x_plus_y_plus_3_l390_39063

theorem xy_eq_x_plus_y_plus_3 (x y : ℕ) : 
  x * y = x + y + 3 ↔ (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_xy_eq_x_plus_y_plus_3_l390_39063


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_value_l390_39056

def M (a : ℝ) : Set ℝ := {x | x - a = 0}
def N (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_value (a : ℝ) :
  M a ∩ N a = N a → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_value_l390_39056


namespace NUMINAMATH_CALUDE_max_children_theorem_l390_39024

/-- Represents the movie theater pricing and budget scenario -/
structure MovieTheater where
  budget : ℕ
  adultTicketCost : ℕ
  childTicketCost : ℕ
  childTicketGroupDiscount : ℕ
  groupDiscountThreshold : ℕ
  snackCost : ℕ

/-- Calculates the maximum number of children that can be taken to the movies -/
def maxChildren (mt : MovieTheater) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of children is 12 with group discount -/
theorem max_children_theorem (mt : MovieTheater) 
  (h1 : mt.budget = 100)
  (h2 : mt.adultTicketCost = 12)
  (h3 : mt.childTicketCost = 6)
  (h4 : mt.childTicketGroupDiscount = 4)
  (h5 : mt.groupDiscountThreshold = 5)
  (h6 : mt.snackCost = 3) :
  maxChildren mt = 12 ∧ 
  12 * mt.childTicketGroupDiscount + 12 * mt.snackCost + mt.adultTicketCost ≤ mt.budget :=
sorry

end NUMINAMATH_CALUDE_max_children_theorem_l390_39024


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l390_39078

theorem linear_systems_solutions :
  -- First system
  (∃ x y : ℝ, x + y = 5 ∧ 4*x - 2*y = 2 ∧ x = 2 ∧ y = 3) ∧
  -- Second system
  (∃ x y : ℝ, 3*x - 2*y = 13 ∧ 4*x + 3*y = 6 ∧ x = 3 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l390_39078


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l390_39059

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l390_39059


namespace NUMINAMATH_CALUDE_monthly_average_production_l390_39085

/-- The daily average production for a month given production rates for different periods -/
theorem monthly_average_production 
  (days_first_period : ℕ) 
  (days_second_period : ℕ) 
  (avg_first_period : ℕ) 
  (avg_second_period : ℕ) 
  (h1 : days_first_period = 25)
  (h2 : days_second_period = 5)
  (h3 : avg_first_period = 50)
  (h4 : avg_second_period = 38) :
  (days_first_period * avg_first_period + days_second_period * avg_second_period) / 
  (days_first_period + days_second_period) = 48 := by
  sorry

#check monthly_average_production

end NUMINAMATH_CALUDE_monthly_average_production_l390_39085


namespace NUMINAMATH_CALUDE_remainder_problem_l390_39012

theorem remainder_problem (x y : ℤ) 
  (hx : x % 126 = 11) 
  (hy : y % 126 = 25) : 
  (x + y + 23) % 63 = 59 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l390_39012


namespace NUMINAMATH_CALUDE_hyperbola_foci_l390_39045

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 7 - y^2 / 9 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem: The foci of the given hyperbola are located at (±4, 0) -/
theorem hyperbola_foci :
  let (a, b) := focus_coordinate
  (hyperbola_equation a b ∨ hyperbola_equation (-a) b) ∧
  ∀ (x y : ℝ), (x, y) ≠ (a, b) ∧ (x, y) ≠ (-a, b) →
    ¬(hyperbola_equation x y ∧ x^2 - y^2 = a^2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_l390_39045


namespace NUMINAMATH_CALUDE_frisbee_sales_receipts_l390_39054

theorem frisbee_sales_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 4 →
  3 * x + 4 * y = 196 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_sales_receipts_l390_39054


namespace NUMINAMATH_CALUDE_opposite_pairs_l390_39099

theorem opposite_pairs : 
  (3^2 = -(-3^2)) ∧ 
  (3^2 ≠ -2^3) ∧ 
  (3^2 ≠ -(-3)^2) ∧ 
  (-3^2 ≠ -(-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l390_39099


namespace NUMINAMATH_CALUDE_square_sum_problem_l390_39094

theorem square_sum_problem (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 6*y^2 = 108 := by sorry

end NUMINAMATH_CALUDE_square_sum_problem_l390_39094


namespace NUMINAMATH_CALUDE_remaining_balloons_l390_39007

def initial_balloons : ℕ := 30
def balloons_given : ℕ := 16

theorem remaining_balloons : initial_balloons - balloons_given = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l390_39007


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l390_39004

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l390_39004


namespace NUMINAMATH_CALUDE_simplify_fraction_l390_39092

theorem simplify_fraction : (120 : ℚ) / 1320 = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l390_39092


namespace NUMINAMATH_CALUDE_complex_number_relation_l390_39083

theorem complex_number_relation (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_relation_l390_39083


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l390_39090

-- Define the parameters of the binomial distribution
def n : ℕ := 6
def p : ℚ := 1/3

-- Define the probability mass function for the binomial distribution
def binomial_pmf (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- State the theorem
theorem binomial_probability_two_successes :
  binomial_pmf 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l390_39090


namespace NUMINAMATH_CALUDE_cost_price_calculation_l390_39025

theorem cost_price_calculation (C : ℝ) : C = 400 :=
  let SP := 0.8 * C
  have selling_price : SP = 0.8 * C := by sorry
  have increased_price : SP + 100 = 1.05 * C := by sorry
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l390_39025


namespace NUMINAMATH_CALUDE_min_value_3x_4y_l390_39021

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  3 * x + 4 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = x * y ∧ 3 * x + 4 * y = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_4y_l390_39021


namespace NUMINAMATH_CALUDE_parallelogram_height_l390_39098

theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 360 ∧ base = 30 ∧ area = base * height → height = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l390_39098


namespace NUMINAMATH_CALUDE_team_formation_count_l390_39079

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male doctors -/
def male_doctors : ℕ := 5

/-- The number of female doctors -/
def female_doctors : ℕ := 4

/-- The size of the team -/
def team_size : ℕ := 3

/-- The number of ways to form a team with both male and female doctors -/
def team_formations : ℕ := 
  choose male_doctors 2 * choose female_doctors 1 + 
  choose male_doctors 1 * choose female_doctors 2

theorem team_formation_count : team_formations = 70 := by sorry

end NUMINAMATH_CALUDE_team_formation_count_l390_39079


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_line_l390_39082

/-- Given a line l: x - y - 1 = 0 and two points A(-1, 1) and B(2, -2),
    prove that B is symmetric to A with respect to l. -/
theorem symmetric_point_wrt_line :
  let l : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (∀ x y, l x y ↔ x - y - 1 = 0) →
  l midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = -((B.1 - A.1) / (B.2 - A.2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_line_l390_39082


namespace NUMINAMATH_CALUDE_virtual_set_divisors_l390_39018

def isVirtual (A : Finset ℕ) : Prop :=
  A.card = 5 ∧
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) > 1) ∧
  (∀ (a b c d : ℕ), a ∈ A → b ∈ A → c ∈ A → d ∈ A → a ≠ b → b ≠ c → c ≠ d → a ≠ c → a ≠ d → b ≠ d → Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 1)

theorem virtual_set_divisors (A : Finset ℕ) (h : isVirtual A) :
  (Finset.prod A id).divisors.card ≥ 2020 := by
  sorry

end NUMINAMATH_CALUDE_virtual_set_divisors_l390_39018


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l390_39022

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (1 / (a + 3*b) + 1 / (b + 3*c) + 1 / (c + 3*a)) ≥ 3/4 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 
    1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l390_39022


namespace NUMINAMATH_CALUDE_streak_plate_method_claim_incorrect_l390_39069

/-- Represents the capability of the streak plate method -/
structure StreakPlateMethod where
  can_separate : Bool
  can_count : Bool

/-- The actual capabilities of the streak plate method -/
def actual_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := false }

/-- The claimed capabilities of the streak plate method in the statement -/
def claimed_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := true }

/-- Theorem stating that the claim about the streak plate method is incorrect -/
theorem streak_plate_method_claim_incorrect :
  actual_streak_plate_method ≠ claimed_streak_plate_method :=
by sorry

end NUMINAMATH_CALUDE_streak_plate_method_claim_incorrect_l390_39069


namespace NUMINAMATH_CALUDE_exam_scores_difference_l390_39027

theorem exam_scores_difference (score1 score2 : ℕ) : 
  score1 = 42 →
  score2 = 33 →
  score1 = (56 * (score1 + score2)) / 100 →
  score1 - score2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_scores_difference_l390_39027


namespace NUMINAMATH_CALUDE_ellipse_b_squared_value_l390_39008

/-- Given an ellipse and a hyperbola with coinciding foci, prove the value of b^2 for the ellipse -/
theorem ellipse_b_squared_value (b : ℝ) : 
  (∀ x y, x^2/25 + y^2/b^2 = 1 → x^2/169 - y^2/64 = 1/36) → 
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 233/36) →
  b^2 = 667/36 := by
sorry

end NUMINAMATH_CALUDE_ellipse_b_squared_value_l390_39008


namespace NUMINAMATH_CALUDE_petes_number_l390_39033

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l390_39033


namespace NUMINAMATH_CALUDE_floor_of_pi_l390_39015

theorem floor_of_pi : ⌊Real.pi⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_pi_l390_39015


namespace NUMINAMATH_CALUDE_twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l390_39009

-- Definition of perfect number
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem twenty_nine_is_perfect : is_perfect_number 29 := by sorry

-- Statement 2
theorem factorization_condition (m n : ℝ) :
  (∀ x : ℝ, x^2 - 6*x + 5 = (x - m)^2 + n) → m*n = -12 := by sorry

-- Statement 3
theorem equation_solution :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 5 = 0 → x + y = -1 := by sorry

-- Statement 4
theorem perfect_number_condition :
  ∃ k : ℤ, ∀ x y : ℤ, ∃ p q : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = p^2 + q^2 := by sorry

end NUMINAMATH_CALUDE_twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l390_39009


namespace NUMINAMATH_CALUDE_fraction_difference_times_two_l390_39091

theorem fraction_difference_times_two :
  let a := 4 + 6 + 8 + 10
  let b := 3 + 5 + 7 + 9
  (a / b - b / a) * 2 = 13 / 21 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_times_two_l390_39091


namespace NUMINAMATH_CALUDE_expression_factorization_l390_39032

theorem expression_factorization (x : ℝ) :
  (16 * x^7 + 32 * x^5 - 9) - (4 * x^7 - 8 * x^5 + 9) = 2 * (6 * x^7 + 20 * x^5 - 9) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l390_39032


namespace NUMINAMATH_CALUDE_unique_positive_solution_l390_39096

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l390_39096


namespace NUMINAMATH_CALUDE_right_triangle_area_l390_39052

theorem right_triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l390_39052


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_sum_power_l390_39003

theorem polynomial_factorization_and_sum_power (a b : ℤ) : 
  (∀ x : ℝ, x^2 + x - 6 = (x + a) * (x + b)) → (a + b)^2023 = 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_sum_power_l390_39003


namespace NUMINAMATH_CALUDE_solution_set_f_leq_3_min_m_for_inequality_l390_39053

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Theorem for part II
theorem min_m_for_inequality (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ m - x - 4/x) ↔ m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_3_min_m_for_inequality_l390_39053


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l390_39088

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x + 3) - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 1

theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l390_39088


namespace NUMINAMATH_CALUDE_max_product_sum_l390_39064

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 24) :
  (A * M * C + A * M + M * C + C * A) ≤ 704 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l390_39064


namespace NUMINAMATH_CALUDE_min_score_theorem_l390_39043

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

end NUMINAMATH_CALUDE_min_score_theorem_l390_39043


namespace NUMINAMATH_CALUDE_two_amoebas_fill_time_l390_39034

/-- The time (in minutes) it takes for amoebas to fill a bottle -/
def fill_time (initial_count : ℕ) : ℕ → ℕ
| 60 => 1  -- One amoeba fills the bottle in 60 minutes
| t => initial_count * 2^(t / 3)  -- Amoeba count at time t

/-- Theorem stating that two amoebas fill the bottle in 57 minutes -/
theorem two_amoebas_fill_time : fill_time 2 57 = fill_time 1 60 := by
  sorry

end NUMINAMATH_CALUDE_two_amoebas_fill_time_l390_39034


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l390_39044

/-- An isosceles trapezoid circumscribed about a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One base angle of the trapezoid -/
  baseAngle : ℝ

/-- The area of the isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 16)
  (h2 : t.baseAngle = Real.arcsin 0.8) :
  area t = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l390_39044


namespace NUMINAMATH_CALUDE_power_digits_sum_l390_39041

theorem power_digits_sum : ∃ (m n : ℕ), 
  (100 ≤ 2^m ∧ 2^m < 10000) ∧ 
  (100 ≤ 5^n ∧ 5^n < 10000) ∧ 
  (2^m / 100 % 10 = 5^n / 100 % 10) ∧
  (2^m / 100 % 10 + 5^n / 100 % 10 = 4) :=
sorry

end NUMINAMATH_CALUDE_power_digits_sum_l390_39041


namespace NUMINAMATH_CALUDE_first_player_wins_l390_39057

/-- Represents a position on the 8x8 grid --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Defines the possible moves --/
inductive Move
  | Right
  | Up
  | UpRight

/-- Applies a move to a position --/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.Right => ⟨p.x + 1, p.y⟩
  | Move.Up => ⟨p.x, p.y + 1⟩
  | Move.UpRight => ⟨p.x + 1, p.y + 1⟩

/-- Checks if a position is within the 8x8 grid --/
def isValidPosition (p : Position) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 8 ∧ 1 ≤ p.y ∧ p.y ≤ 8

/-- Defines a winning position --/
def isWinningPosition (p : Position) : Prop :=
  p.x = 8 ∧ p.y = 8

/-- Theorem: The first player has a winning strategy --/
theorem first_player_wins :
  ∃ (m : Move), isValidPosition (applyMove ⟨1, 1⟩ m) ∧
  ∀ (p : Position),
    isValidPosition p →
    ¬isWinningPosition p →
    (p.x % 2 = 0 ∧ p.y % 2 = 0) →
    ∃ (m : Move),
      isValidPosition (applyMove p m) ∧
      ¬(applyMove p m).x % 2 = 0 ∧
      ¬(applyMove p m).y % 2 = 0 :=
by sorry

#check first_player_wins

end NUMINAMATH_CALUDE_first_player_wins_l390_39057


namespace NUMINAMATH_CALUDE_product_of_special_set_l390_39047

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) (h_odd : Odd n) (h_n_gt_1 : n > 1) 
  (h_card : M.card = n) (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + x) : 
  M.prod id = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_set_l390_39047


namespace NUMINAMATH_CALUDE_abs_x_squared_plus_abs_x_minus_six_roots_sum_l390_39062

theorem abs_x_squared_plus_abs_x_minus_six_roots_sum (x : ℝ) :
  (|x|^2 + |x| - 6 = 0) → (∃ a b : ℝ, a + b = 0 ∧ |a|^2 + |a| - 6 = 0 ∧ |b|^2 + |b| - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_squared_plus_abs_x_minus_six_roots_sum_l390_39062


namespace NUMINAMATH_CALUDE_angle_supplement_l390_39037

theorem angle_supplement (α : ℝ) : 
  (90 - α = 125) → (180 - α = 125) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l390_39037


namespace NUMINAMATH_CALUDE_sum_of_max_min_F_l390_39051

-- Define the function f as an odd function on [-a, a]
def f (a : ℝ) (x : ℝ) : ℝ := sorry

-- Define F(x) = f(x) + 1
def F (a : ℝ) (x : ℝ) : ℝ := f a x + 1

-- Theorem statement
theorem sum_of_max_min_F (a : ℝ) (h : a > 0) :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-a) a ∧ x_min ∈ Set.Icc (-a) a ∧
  (∀ x ∈ Set.Icc (-a) a, F a x ≤ F a x_max) ∧
  (∀ x ∈ Set.Icc (-a) a, F a x_min ≤ F a x) ∧
  F a x_max + F a x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_F_l390_39051


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l390_39030

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l390_39030


namespace NUMINAMATH_CALUDE_sum_vector_magnitude_l390_39073

/-- Given two vectors a and b in ℝ³, prove that their sum has magnitude √26 -/
theorem sum_vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (1, -1, 0) → b = (3, -2, 1) → ‖a + b‖ = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_vector_magnitude_l390_39073


namespace NUMINAMATH_CALUDE_perfect_square_increased_by_prime_l390_39038

theorem perfect_square_increased_by_prime (n : ℕ) : ∃ n : ℕ, 
  (∃ a : ℕ, n^2 = a^2) ∧ 
  (∃ b : ℕ, n^2 + 461 = b^2) ∧ 
  (∃ c : ℕ, n^2 = 5 * c) ∧ 
  (∃ d : ℕ, n^2 + 461 = 5 * d) ∧ 
  n^2 = 52900 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_increased_by_prime_l390_39038


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l390_39048

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l390_39048


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l390_39074

theorem flu_transmission_rate : ∃ x : ℝ, 
  (x > 0) ∧ ((1 + x)^2 = 100) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l390_39074


namespace NUMINAMATH_CALUDE_geometric_sequence_product_threshold_l390_39042

theorem geometric_sequence_product_threshold (n : ℕ) : 
  (n > 0 ∧ 3^((n * (n + 1)) / 12) > 1000) ↔ n ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_threshold_l390_39042


namespace NUMINAMATH_CALUDE_set_intersection_complement_equality_l390_39095

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 3}

-- Define set N
def N : Set ℝ := {x | x ≤ 2}

-- Theorem statement
theorem set_intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_equality_l390_39095


namespace NUMINAMATH_CALUDE_construction_team_problem_l390_39055

/-- Represents the possible solutions for the original number of people in the second group -/
inductive Solution : Type
  | fiftySeven : Solution
  | twentyOne : Solution

/-- Checks if a given number satisfies the conditions of the problem -/
def satisfiesConditions (x : ℕ) : Prop :=
  ∃ (k : ℕ+), 96 - 16 = k * (x + 16) + 6

/-- The theorem stating that the only solutions are 58 and 21 -/
theorem construction_team_problem :
  ∀ x : ℕ, satisfiesConditions x ↔ (x = 58 ∨ x = 21) :=
sorry

end NUMINAMATH_CALUDE_construction_team_problem_l390_39055


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_150_l390_39049

theorem greatest_common_multiple_9_15_under_150 :
  ∃ n : ℕ, n = 135 ∧ 
  (∀ m : ℕ, m < 150 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  135 % 9 = 0 ∧ 135 % 15 = 0 ∧ 135 < 150 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_150_l390_39049


namespace NUMINAMATH_CALUDE_N_value_l390_39060

theorem N_value : 
  let N := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = (1 + Real.sqrt 6 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_N_value_l390_39060


namespace NUMINAMATH_CALUDE_min_rotation_angle_is_72_l390_39081

/-- A regular five-pointed star -/
structure RegularFivePointedStar where
  -- Add necessary properties here

/-- The minimum rotation angle for a regular five-pointed star to coincide with its original position -/
def min_rotation_angle (star : RegularFivePointedStar) : ℝ :=
  72

/-- Theorem stating that the minimum rotation angle for a regular five-pointed star 
    to coincide with its original position is 72 degrees -/
theorem min_rotation_angle_is_72 (star : RegularFivePointedStar) :
  min_rotation_angle star = 72 := by
  sorry

end NUMINAMATH_CALUDE_min_rotation_angle_is_72_l390_39081


namespace NUMINAMATH_CALUDE_sqrt_3_squared_times_5_to_6_l390_39005

theorem sqrt_3_squared_times_5_to_6 : Real.sqrt (3^2 * 5^6) = 375 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_squared_times_5_to_6_l390_39005


namespace NUMINAMATH_CALUDE_p_squared_minus_q_squared_l390_39031

theorem p_squared_minus_q_squared (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p - q = 4) : 
  p^2 - q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_p_squared_minus_q_squared_l390_39031


namespace NUMINAMATH_CALUDE_straight_flush_probability_l390_39050

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a poker hand -/
def PokerHand : ℕ := 5

/-- Represents the number of possible starting ranks for a straight flush -/
def StartingRanks : ℕ := 10

/-- Represents the number of suits in a standard deck -/
def Suits : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of possible 5-card hands -/
def TotalHands : ℕ := choose StandardDeck PokerHand

/-- Represents the total number of straight flushes -/
def StraightFlushes : ℕ := StartingRanks * Suits

/-- Theorem: The probability of drawing a straight flush is 1/64,974 -/
theorem straight_flush_probability :
  StraightFlushes / TotalHands = 1 / 64974 := by sorry

end NUMINAMATH_CALUDE_straight_flush_probability_l390_39050


namespace NUMINAMATH_CALUDE_store_profit_l390_39000

/-- Prove that a store makes a profit when selling pens purchased from two markets -/
theorem store_profit (m n : ℝ) (h : m > n) : 
  let selling_price := (m + n) / 2
  let profit_A := 40 * (selling_price - m)
  let profit_B := 60 * (selling_price - n)
  profit_A + profit_B > 0 := by
  sorry


end NUMINAMATH_CALUDE_store_profit_l390_39000


namespace NUMINAMATH_CALUDE_investment_schemes_count_l390_39019

/-- The number of projects to invest in -/
def num_projects : ℕ := 3

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- Calculates the number of investment schemes -/
def num_investment_schemes : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 60 -/
theorem investment_schemes_count :
  num_investment_schemes = 60 := by sorry

end NUMINAMATH_CALUDE_investment_schemes_count_l390_39019


namespace NUMINAMATH_CALUDE_sum_of_divisors_3k_plus_2_multiple_of_3_l390_39089

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- A number is of the form 3k + 2 -/
def is_3k_plus_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k + 2

theorem sum_of_divisors_3k_plus_2_multiple_of_3 (n : ℕ) (h : is_3k_plus_2 n) :
  3 ∣ sum_of_divisors n :=
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_3k_plus_2_multiple_of_3_l390_39089


namespace NUMINAMATH_CALUDE_odd_sum_floor_power_l390_39067

theorem odd_sum_floor_power (n : ℕ+) : 
  Odd (n + ⌊(Real.sqrt 2 + 1)^(n : ℝ)⌋) := by sorry

end NUMINAMATH_CALUDE_odd_sum_floor_power_l390_39067


namespace NUMINAMATH_CALUDE_train_length_l390_39016

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 100 → speed_kmh = 90 → 
  crossing_time * (speed_kmh * (1000 / 3600)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l390_39016


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l390_39014

theorem stamp_collection_problem : ∃! x : ℕ, 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 5 = 3 ∧ 
  x % 9 = 7 ∧ 
  150 < x ∧ 
  x ≤ 300 ∧ 
  x = 223 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l390_39014


namespace NUMINAMATH_CALUDE_sum_upper_bound_l390_39023

/-- Given positive real numbers x and y satisfying 2x + 8y - xy = 0,
    the sum x + y is always less than or equal to 18. -/
theorem sum_upper_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 2 * x + 8 * y - x * y = 0) : 
  x + y ≤ 18 := by
sorry

end NUMINAMATH_CALUDE_sum_upper_bound_l390_39023


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l390_39013

theorem sum_of_coefficients_is_two 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_two_l390_39013


namespace NUMINAMATH_CALUDE_opposite_of_eight_l390_39035

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of 8 is -8. -/
theorem opposite_of_eight : opposite 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l390_39035


namespace NUMINAMATH_CALUDE_ant_travel_distance_l390_39071

theorem ant_travel_distance (planet_radius : ℝ) (observer_height : ℝ) : 
  planet_radius = 156 → observer_height = 13 → 
  let horizon_distance := Real.sqrt ((planet_radius + observer_height)^2 - planet_radius^2)
  (2 * Real.pi * horizon_distance) = 130 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ant_travel_distance_l390_39071


namespace NUMINAMATH_CALUDE_first_day_over_threshold_l390_39001

/-- The number of paperclips Max starts with on Monday -/
def initial_paperclips : ℕ := 3

/-- The factor by which the number of paperclips increases each day -/
def daily_increase_factor : ℕ := 4

/-- The threshold number of paperclips -/
def threshold : ℕ := 200

/-- The function that calculates the number of paperclips on day n -/
def paperclips (n : ℕ) : ℕ := initial_paperclips * daily_increase_factor^(n - 1)

/-- The theorem stating that the 5th day is the first day with more than 200 paperclips -/
theorem first_day_over_threshold :
  ∀ n : ℕ, n > 0 → (paperclips n > threshold ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_first_day_over_threshold_l390_39001


namespace NUMINAMATH_CALUDE_lcm_of_18_and_50_l390_39046

theorem lcm_of_18_and_50 : Nat.lcm 18 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_and_50_l390_39046


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l390_39097

/-- The standard equation of an ellipse with specific parameters. -/
theorem ellipse_standard_equation 
  (foci_on_y_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) : 
  foci_on_y_axis ∧ major_axis_length = 20 ∧ eccentricity = 2/5 → 
  ∃ (x y : ℝ), y^2/100 + x^2/84 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l390_39097


namespace NUMINAMATH_CALUDE_second_pile_magazines_l390_39010

/-- A sequence of 5 terms representing the number of magazines in each pile. -/
def MagazineSequence : Type := Fin 5 → ℕ

/-- The properties of the magazine sequence based on the given information. -/
def IsValidMagazineSequence (s : MagazineSequence) : Prop :=
  s 0 = 3 ∧ s 2 = 6 ∧ s 3 = 9 ∧ s 4 = 13 ∧
  ∀ i : Fin 4, s (i + 1) - s i = s 1 - s 0

/-- Theorem stating that for any valid magazine sequence, the second term (index 1) must be 3. -/
theorem second_pile_magazines (s : MagazineSequence) 
  (h : IsValidMagazineSequence s) : s 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_pile_magazines_l390_39010


namespace NUMINAMATH_CALUDE_hall_dimension_difference_l390_39002

/-- For a rectangular hall with width equal to half its length and area 450 sq. m,
    the difference between length and width is 15 meters. -/
theorem hall_dimension_difference (length width : ℝ) : 
  width = length / 2 →
  length * width = 450 →
  length - width = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_dimension_difference_l390_39002


namespace NUMINAMATH_CALUDE_village_population_is_72_l390_39086

/-- The number of people a vampire drains per week -/
def vampire_drain_rate : ℕ := 3

/-- The number of people a werewolf eats per week -/
def werewolf_eat_rate : ℕ := 5

/-- The number of weeks the village lasts -/
def weeks_lasted : ℕ := 9

/-- The total number of people in the village -/
def village_population : ℕ := vampire_drain_rate * weeks_lasted + werewolf_eat_rate * weeks_lasted

theorem village_population_is_72 : village_population = 72 := by
  sorry

end NUMINAMATH_CALUDE_village_population_is_72_l390_39086


namespace NUMINAMATH_CALUDE_hexadecimal_to_decimal_l390_39070

theorem hexadecimal_to_decimal (m : ℕ) : 
  1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexadecimal_to_decimal_l390_39070


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l390_39006

theorem sum_of_coefficients_zero (x y z : ℝ) :
  (λ x y z => (2*x - 3*y + z)^20) 1 1 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l390_39006


namespace NUMINAMATH_CALUDE_expression_simplification_l390_39068

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (2 * x - 6) / (x - 2) / (5 / (x - 2) - x - 2) = Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l390_39068


namespace NUMINAMATH_CALUDE_quadratic_factorization_l390_39093

theorem quadratic_factorization (y A B : ℤ) : 
  (15 * y^2 - 94 * y + 56 = (A * y - 7) * (B * y - 8)) → 
  (A * B + A = 20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l390_39093


namespace NUMINAMATH_CALUDE_jacob_age_l390_39036

theorem jacob_age (maya drew peter john jacob : ℕ) 
  (h1 : drew = maya + 5)
  (h2 : peter = drew + 4)
  (h3 : john = 30)
  (h4 : john = 2 * maya)
  (h5 : jacob + 2 = (peter + 2) / 2) :
  jacob = 11 := by
  sorry

end NUMINAMATH_CALUDE_jacob_age_l390_39036


namespace NUMINAMATH_CALUDE_inequality_solution_set_l390_39029

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l390_39029


namespace NUMINAMATH_CALUDE_max_profit_at_15_verify_conditions_l390_39066

-- Define the relationship between price and sales quantity
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def profit (x : ℤ) : ℤ := sales_quantity x * (x - 8)

-- Theorem statement
theorem max_profit_at_15 :
  ∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525 ∧ profit 15 = 525 :=
by
  sorry

-- Verify the given conditions
theorem verify_conditions :
  sales_quantity 9 = 105 ∧ sales_quantity 11 = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_15_verify_conditions_l390_39066


namespace NUMINAMATH_CALUDE_distinct_sums_lower_bound_l390_39058

theorem distinct_sums_lower_bound (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, a i > 0) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  (Finset.powerset (Finset.range n)).card ≥ n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_lower_bound_l390_39058


namespace NUMINAMATH_CALUDE_ellipse_intersection_constant_sum_distance_l390_39039

/-- The slope of a line that intersects an ellipse such that the sum of squared distances
    from any point on the major axis to the intersection points is constant. -/
theorem ellipse_intersection_constant_sum_distance (k : ℝ) : 
  (∀ a : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (A.2 - 0 = k * (A.1 - a)) ∧
    (B.2 - 0 = k * (B.1 - a)) ∧
    ((A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = (512 - 800 * k^2) / (16 + 25 * k^2))) →
  k = 4/5 ∨ k = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_constant_sum_distance_l390_39039


namespace NUMINAMATH_CALUDE_reciprocal_of_product_l390_39076

theorem reciprocal_of_product : (((1 : ℚ) / 3) * (3 / 4))⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_product_l390_39076


namespace NUMINAMATH_CALUDE_candy_distribution_l390_39084

theorem candy_distribution (x : ℚ) 
  (laura_candies : x > 0)
  (mark_candies : ℚ → ℚ)
  (nina_candies : ℚ → ℚ)
  (oliver_candies : ℚ → ℚ)
  (mark_def : mark_candies x = 4 * x)
  (nina_def : nina_candies x = 2 * mark_candies x)
  (oliver_def : oliver_candies x = 6 * nina_candies x)
  (total_candies : x + mark_candies x + nina_candies x + oliver_candies x = 360) :
  x = 360 / 61 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l390_39084


namespace NUMINAMATH_CALUDE_two_numbers_difference_l390_39080

theorem two_numbers_difference (a b : ℝ) : 
  a + b = 10 → a^2 - b^2 = 40 → |a - b| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l390_39080


namespace NUMINAMATH_CALUDE_underdog_wins_in_nine_games_l390_39087

/- Define the probability of the favored team winning a single game -/
def p : ℚ := 3/4

/- Define the number of games needed to win the series -/
def games_to_win : ℕ := 5

/- Define the maximum number of games in the series -/
def max_games : ℕ := 9

/- Define the probability of the underdog team winning a single game -/
def q : ℚ := 1 - p

/- Define the number of ways to choose 4 games out of 8 -/
def ways_to_choose : ℕ := Nat.choose 8 4

theorem underdog_wins_in_nine_games :
  (ways_to_choose : ℚ) * q^4 * p^4 * q = 5670/262144 := by
  sorry

end NUMINAMATH_CALUDE_underdog_wins_in_nine_games_l390_39087


namespace NUMINAMATH_CALUDE_pie_shop_pricing_l390_39040

/-- The number of slices per whole pie -/
def slices_per_pie : ℕ := 4

/-- The number of pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue from selling all pies -/
def total_revenue : ℕ := 180

/-- The price per slice of pie -/
def price_per_slice : ℚ := 5

theorem pie_shop_pricing :
  price_per_slice = total_revenue / (pies_sold * slices_per_pie) := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_pricing_l390_39040


namespace NUMINAMATH_CALUDE_characterize_satisfying_polynomials_l390_39072

/-- A polynomial satisfying the given inequality. -/
structure SatisfyingPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  h_c : |c| ≤ 1
  h_ab : (|a| = 1 ∧ b = 0) ∨ (|a| < 1 ∧ |b| ≤ 2 * Real.sqrt (1 + a * c - |a + c|))

/-- The main theorem statement. -/
theorem characterize_satisfying_polynomials :
  ∀ (P : ℝ → ℝ), (∀ x : ℝ, |P x - x| ≤ x^2 + 1) ↔
    ∃ (p : SatisfyingPolynomial), ∀ x : ℝ, P x = p.a * x^2 + (p.b + 1) * x + p.c :=
sorry

end NUMINAMATH_CALUDE_characterize_satisfying_polynomials_l390_39072


namespace NUMINAMATH_CALUDE_max_value_product_l390_39028

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 6 * y < 90) :
  x * y * (90 - 5 * x - 6 * y) ≤ 900 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 6 * y₀ < 90 ∧ x₀ * y₀ * (90 - 5 * x₀ - 6 * y₀) = 900 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l390_39028


namespace NUMINAMATH_CALUDE_percentage_difference_l390_39061

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l390_39061


namespace NUMINAMATH_CALUDE_age_height_not_function_l390_39017

-- Define the types for our variables
def Age := ℕ
def Height := ℝ
def Radius := ℝ
def Circumference := ℝ
def Angle := ℝ
def SineValue := ℝ
def NumSides := ℕ
def SumInteriorAngles := ℝ

-- Define the relationships as functions
def radiusToCircumference : Radius → Circumference := sorry
def angleToSine : Angle → SineValue := sorry
def sidesToInteriorAnglesSum : NumSides → SumInteriorAngles := sorry

-- Define the relationship between age and height
def ageHeightRelation : Age → Set Height := sorry

-- Theorem to prove
theorem age_height_not_function :
  ¬(∃ (f : Age → Height), ∀ a : Age, ∃! h : Height, h ∈ ageHeightRelation a) :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l390_39017


namespace NUMINAMATH_CALUDE_souvenir_theorem_l390_39020

/-- Represents the souvenirs sold at the Beijing Winter Olympics store -/
structure Souvenir where
  costA : ℕ  -- Cost price of souvenir A
  costB : ℕ  -- Cost price of souvenir B
  totalA : ℕ -- Total cost for souvenir A
  totalB : ℕ -- Total cost for souvenir B

/-- Represents the sales data for the souvenirs -/
structure SalesData where
  initPriceA : ℕ  -- Initial selling price of A
  initPriceB : ℕ  -- Initial selling price of B
  initSoldA : ℕ   -- Initial units of A sold per day
  initSoldB : ℕ   -- Initial units of B sold per day
  priceChangeA : ℤ -- Price change for A
  priceChangeB : ℤ -- Price change for B
  soldChangeA : ℕ  -- Change in units sold for A per 1 yuan price change
  soldChangeB : ℕ  -- Change in units sold for B per 1 yuan price change
  totalSold : ℕ   -- Total souvenirs sold on a certain day

/-- Theorem stating the cost prices and maximum profit -/
theorem souvenir_theorem (s : Souvenir) (d : SalesData) 
  (h1 : s.costB = s.costA + 9)
  (h2 : s.totalA = 10400)
  (h3 : s.totalB = 14000)
  (h4 : d.initPriceA = 46)
  (h5 : d.initPriceB = 45)
  (h6 : d.initSoldA = 40)
  (h7 : d.initSoldB = 80)
  (h8 : d.soldChangeA = 4)
  (h9 : d.soldChangeB = 2)
  (h10 : d.totalSold = 140) :
  s.costA = 26 ∧ s.costB = 35 ∧ 
  ∃ (profit : ℕ), profit = 2000 ∧ 
  ∀ (p : ℕ), p ≤ profit := by
    sorry

end NUMINAMATH_CALUDE_souvenir_theorem_l390_39020


namespace NUMINAMATH_CALUDE_two_valid_numbers_l390_39077

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧  -- six-digit number
  (n % 72 = 0) ∧  -- divisible by 72
  (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = a * 100000 + 2016 * 10 + b)  -- formed by adding digits to 2016

theorem two_valid_numbers :
  {n : ℕ | is_valid_number n} = {920160, 120168} := by sorry

end NUMINAMATH_CALUDE_two_valid_numbers_l390_39077
