import Mathlib

namespace NUMINAMATH_CALUDE_double_mean_value_range_l361_36174

/-- A function is a double mean value function on an interval [a,b] if there exist
    x₁ and x₂ in (a,b) such that f'(x₁) = f'(x₂) = (f(b) - f(a)) / (b - a) -/
def IsDoubleMeanValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    deriv f x₁ = (f b - f a) / (b - a) ∧
    deriv f x₂ = (f b - f a) / (b - a)

/-- The main theorem: if f(x) = x³ - 6/5x² is a double mean value function on [0,t],
    then 3/5 < t < 6/5 -/
theorem double_mean_value_range (t : ℝ) :
  IsDoubleMeanValueFunction (fun x => x^3 - 6/5*x^2) 0 t →
  3/5 < t ∧ t < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_double_mean_value_range_l361_36174


namespace NUMINAMATH_CALUDE_part_one_part_two_l361_36164

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  (f a x ≥ 4 - |x - 1|) ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ := 1
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  (1/m + 1/(2*n) = a) →
  (∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = a → m*n ≤ k*l) →
  m*n = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l361_36164


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l361_36147

/-- A quadratic function with vertex form (x + h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∃ b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (f (-4) = 0 ∧ f 1 = -75) →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l361_36147


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l361_36159

/-- The time it takes for a train to pass a pole given its speed and the time it takes to cross a stationary train of known length -/
theorem train_passing_pole_time (v : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  v = 64.8 →
  t_cross = 25 →
  l_stationary = 360 →
  ∃ t_pole : ℝ, abs (t_pole - 19.44) < 0.01 ∧ 
  t_pole = (v * t_cross - l_stationary) / v :=
by sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l361_36159


namespace NUMINAMATH_CALUDE_solve_factorial_equation_l361_36157

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ (n : ℕ), n * factorial n + factorial n = 5040 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_factorial_equation_l361_36157


namespace NUMINAMATH_CALUDE_cool_parents_problem_l361_36179

theorem cool_parents_problem (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cool_parents_problem_l361_36179


namespace NUMINAMATH_CALUDE_tan_half_positive_in_second_quadrant_l361_36134

theorem tan_half_positive_in_second_quadrant (θ : Real) : 
  (π/2 < θ ∧ θ < π) → 0 < Real.tan (θ/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_positive_in_second_quadrant_l361_36134


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l361_36107

theorem circles_externally_tangent : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*y + 5 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 ∧
  (∀ (q : ℝ × ℝ), q ≠ p → (q ∈ circle1 → q ∉ circle2) ∧ (q ∈ circle2 → q ∉ circle1)) :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l361_36107


namespace NUMINAMATH_CALUDE_different_color_probability_l361_36146

/-- The probability of drawing two chips of different colors from a bag containing 
    7 blue chips, 5 yellow chips, and 4 red chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips := 7 + 5 + 4
  let p_blue := 7 / total_chips
  let p_yellow := 5 / total_chips
  let p_red := 4 / total_chips
  let p_different := p_blue * (p_yellow + p_red) + 
                     p_yellow * (p_blue + p_red) + 
                     p_red * (p_blue + p_yellow)
  p_different = 83 / 128 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l361_36146


namespace NUMINAMATH_CALUDE_valid_student_counts_exists_valid_distributions_l361_36114

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : ℕ
  groups_with_13 : ℕ
  total_students : ℕ

/-- Checks if a given distribution satisfies the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the only valid total numbers of students -/
theorem valid_student_counts :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

/-- Theorem proving the existence of valid distributions -/
theorem exists_valid_distributions :
  ∃ d₁ d₂ : StudentDistribution,
    is_valid_distribution d₁ ∧
    is_valid_distribution d₂ ∧
    d₁.total_students = 76 ∧
    d₂.total_students = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_valid_student_counts_exists_valid_distributions_l361_36114


namespace NUMINAMATH_CALUDE_Y_two_five_l361_36199

def Y (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2 + 3

theorem Y_two_five : Y 2 5 = 2 := by sorry

end NUMINAMATH_CALUDE_Y_two_five_l361_36199


namespace NUMINAMATH_CALUDE_integer_pair_solution_l361_36106

theorem integer_pair_solution (a b : ℤ) : 
  (a + b) / (a - b) = 3 ∧ (a + b) * (a - b) = 300 →
  (a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l361_36106


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l361_36117

theorem polynomial_multiplication (y : ℝ) :
  (3*y - 2 + 4) * (2*y^12 + 3*y^11 - y^9 - y^8) =
  6*y^13 + 13*y^12 + 6*y^11 - 3*y^10 - 5*y^9 - 2*y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l361_36117


namespace NUMINAMATH_CALUDE_supplement_of_complement_65_l361_36126

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65 :
  supplement (complement 65) = 155 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_65_l361_36126


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l361_36191

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 0.30) * (1 + 0.5873) = P → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l361_36191


namespace NUMINAMATH_CALUDE_white_balls_estimate_l361_36139

theorem white_balls_estimate (total_balls : ℕ) (total_draws : ℕ) (white_draws : ℕ) 
  (h_total_balls : total_balls = 20)
  (h_total_draws : total_draws = 100)
  (h_white_draws : white_draws = 40) :
  (white_draws : ℚ) / total_draws * total_balls = 8 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_estimate_l361_36139


namespace NUMINAMATH_CALUDE_unique_eight_times_digit_sum_l361_36138

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_eight_times_digit_sum :
  ∃! n : ℕ, n < 500 ∧ n > 0 ∧ n = 8 * sum_of_digits n := by sorry

end NUMINAMATH_CALUDE_unique_eight_times_digit_sum_l361_36138


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l361_36173

/-- The amount spent on a ferris wheel ride given initial tickets, remaining tickets, and cost per ticket -/
def ferris_wheel_cost (initial_tickets remaining_tickets cost_per_ticket : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * cost_per_ticket

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l361_36173


namespace NUMINAMATH_CALUDE_quadratic_factorization_l361_36125

theorem quadratic_factorization (y : ℝ) (A B : ℤ) 
  (h : ∀ y, 12 * y^2 - 65 * y + 42 = (A * y - 14) * (B * y - 3)) : 
  A * B + A = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l361_36125


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l361_36132

theorem bobby_candy_consumption (initial_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 30) (h2 : remaining_candy = 7) :
  initial_candy - remaining_candy = 23 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l361_36132


namespace NUMINAMATH_CALUDE_meadow_diaper_earnings_l361_36145

/-- Calculates the total money earned from selling diapers -/
def total_money (boxes : ℕ) (packs_per_box : ℕ) (diapers_per_pack : ℕ) (price_per_diaper : ℕ) : ℕ :=
  boxes * packs_per_box * diapers_per_pack * price_per_diaper

/-- Proves that Meadow's total earnings from selling diapers is $960,000 -/
theorem meadow_diaper_earnings :
  total_money 30 40 160 5 = 960000 := by
  sorry

#eval total_money 30 40 160 5

end NUMINAMATH_CALUDE_meadow_diaper_earnings_l361_36145


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l361_36197

theorem equal_roots_quadratic (h : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - 4 * y + h / 3 = 0 → y = x) ↔ h = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l361_36197


namespace NUMINAMATH_CALUDE_a_finishes_in_eight_days_l361_36118

/-- Given two workers A and B who can finish a job together in a certain number of days,
    this function calculates how long it takes for A to finish the job alone. -/
def time_for_a_alone (total_time_together : ℚ) (days_worked_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / total_time_together
  let work_done_together := work_rate_together * days_worked_together
  let remaining_work := 1 - work_done_together
  let work_rate_a := remaining_work / days_a_alone
  1 / work_rate_a

/-- Theorem stating that under the given conditions, A can finish the job alone in 8 days. -/
theorem a_finishes_in_eight_days :
  time_for_a_alone 40 10 6 = 8 := by sorry

end NUMINAMATH_CALUDE_a_finishes_in_eight_days_l361_36118


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l361_36148

theorem sqrt_expression_equality : 
  |Real.sqrt 2 - Real.sqrt 3| - Real.sqrt 4 + Real.sqrt 2 * (Real.sqrt 2 + 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l361_36148


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l361_36195

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ k : ℕ, (2023^2024 - 2025) % 100 = 10 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l361_36195


namespace NUMINAMATH_CALUDE_cookout_buns_l361_36120

/-- The number of packs of burger buns Alex needs to buy for his cookout. -/
def bun_packs_needed (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let total_guests := guests - no_meat_guests
  let total_burgers := total_guests * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ)
    (h1 : guests = 10)
    (h2 : burgers_per_guest = 3)
    (h3 : no_meat_guests = 1)
    (h4 : no_bread_guests = 1)
    (h5 : buns_per_pack = 8) :
  bun_packs_needed guests burgers_per_guest no_meat_guests no_bread_guests buns_per_pack = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_l361_36120


namespace NUMINAMATH_CALUDE_unique_solution_l361_36113

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1958 is the unique solution -/
theorem unique_solution : ∃! n : ℕ, n + S n = 1981 ∧ n = 1958 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l361_36113


namespace NUMINAMATH_CALUDE_sam_current_age_l361_36184

/-- Sam's current age -/
def sam_age : ℕ := 46

/-- Drew's current age -/
def drew_age : ℕ := 12

/-- Theorem stating Sam's current age is 46, given the conditions -/
theorem sam_current_age :
  (sam_age + 5 = 3 * (drew_age + 5)) → sam_age = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_current_age_l361_36184


namespace NUMINAMATH_CALUDE_fq_length_l361_36152

/-- Represents a right triangle with a tangent circle -/
structure RightTriangleWithCircle where
  /-- Length of the hypotenuse -/
  df : ℝ
  /-- Length of one leg -/
  de : ℝ
  /-- Point where the circle meets the hypotenuse -/
  q : ℝ
  /-- The hypotenuse is √85 -/
  hyp_length : df = Real.sqrt 85
  /-- One leg is 7 -/
  leg_length : de = 7
  /-- The circle is tangent to both legs -/
  circle_tangent : True

/-- The length of FQ in the given configuration is 6 -/
theorem fq_length (t : RightTriangleWithCircle) : t.df - t.q = 6 := by
  sorry

end NUMINAMATH_CALUDE_fq_length_l361_36152


namespace NUMINAMATH_CALUDE_trio_selection_l361_36115

theorem trio_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 3) :
  Nat.choose n k = 220 := by
  sorry

end NUMINAMATH_CALUDE_trio_selection_l361_36115


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l361_36167

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 4 / b) ≥ 9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 4 / b) < 9 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l361_36167


namespace NUMINAMATH_CALUDE_abs_a_minus_sqrt_a_squared_l361_36155

theorem abs_a_minus_sqrt_a_squared (a : ℝ) (h : a < 0) : |a - Real.sqrt (a^2)| = -2*a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_sqrt_a_squared_l361_36155


namespace NUMINAMATH_CALUDE_residue_system_product_condition_l361_36137

/-- A function that generates a complete residue system modulo n -/
def completeResidueSystem (n : ℕ) : Fin n → ℕ :=
  fun i => i.val

/-- Predicate to check if a list of natural numbers forms a complete residue system modulo n -/
def isCompleteResidueSystem (n : ℕ) (l : List ℕ) : Prop :=
  l.length = n ∧ ∀ k, 0 ≤ k ∧ k < n → ∃ x ∈ l, x % n = k

theorem residue_system_product_condition (n : ℕ) : 
  (∃ (a b : Fin n → ℕ), 
    isCompleteResidueSystem n (List.ofFn a) ∧
    isCompleteResidueSystem n (List.ofFn b) ∧
    isCompleteResidueSystem n (List.ofFn (fun i => (a i * b i) % n))) ↔ 
  n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_residue_system_product_condition_l361_36137


namespace NUMINAMATH_CALUDE_product_equals_sqrt_ratio_l361_36171

theorem product_equals_sqrt_ratio (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_product_equals_sqrt_ratio_l361_36171


namespace NUMINAMATH_CALUDE_irrational_cubic_roots_not_quadratic_roots_l361_36156

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a number is a root of a cubic polynomial -/
def is_root_cubic (x : ℝ) (p : CubicPolynomial) : Prop :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d = 0

/-- Predicate to check if a number is a root of a quadratic polynomial -/
def is_root_quadratic (x : ℝ) (q : QuadraticPolynomial) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

/-- Main theorem -/
theorem irrational_cubic_roots_not_quadratic_roots
  (p : CubicPolynomial)
  (h1 : ∃ x y z : ℝ, is_root_cubic x p ∧ is_root_cubic y p ∧ is_root_cubic z p)
  (h2 : ∀ x : ℝ, is_root_cubic x p → Irrational x)
  : ∀ q : QuadraticPolynomial, ∀ x : ℝ, is_root_cubic x p → ¬ is_root_quadratic x q :=
sorry

end NUMINAMATH_CALUDE_irrational_cubic_roots_not_quadratic_roots_l361_36156


namespace NUMINAMATH_CALUDE_evaluate_power_l361_36196

theorem evaluate_power : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_evaluate_power_l361_36196


namespace NUMINAMATH_CALUDE_triangle_proof_l361_36149

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = t.a * Real.cos t.B ∧ Real.sin t.C = 1/3

-- State the theorem
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.A = Real.pi/2 ∧ Real.cos (Real.pi + t.B) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l361_36149


namespace NUMINAMATH_CALUDE_third_ball_yarn_amount_l361_36175

/-- The amount of yarn (in feet) used for each ball -/
structure YarnBalls where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Properties of the yarn balls based on the given conditions -/
def validYarnBalls (y : YarnBalls) : Prop :=
  y.first = y.second / 2 ∧ 
  y.third = 3 * y.first ∧ 
  y.second = 18

/-- Theorem stating that the third ball uses 27 feet of yarn -/
theorem third_ball_yarn_amount (y : YarnBalls) (h : validYarnBalls y) : 
  y.third = 27 := by
  sorry

end NUMINAMATH_CALUDE_third_ball_yarn_amount_l361_36175


namespace NUMINAMATH_CALUDE_number_of_routes_l361_36129

/-- Recursive function representing the number of possible routes after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => M (n + 1) + M n

/-- The racing duration in minutes -/
def racingDuration : ℕ := 10

/-- Theorem stating that the number of possible routes after 10 minutes is 34 -/
theorem number_of_routes : M racingDuration = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_of_routes_l361_36129


namespace NUMINAMATH_CALUDE_geometric_series_sum_l361_36163

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1/4
  let S : ℚ := ∑' n, a * r^n
  S = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l361_36163


namespace NUMINAMATH_CALUDE_similarity_transformation_result_l361_36100

/-- A similarity transformation in 2D space -/
structure Similarity2D where
  center : ℝ × ℝ
  ratio : ℝ

/-- Apply a similarity transformation to a point -/
def apply_similarity (s : Similarity2D) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(s.ratio * (p.1 - s.center.1) + s.center.1, s.ratio * (p.2 - s.center.2) + s.center.2),
   (-s.ratio * (p.1 - s.center.1) + s.center.1, -s.ratio * (p.2 - s.center.2) + s.center.2)}

theorem similarity_transformation_result :
  let s : Similarity2D := ⟨(0, 0), 2⟩
  let A : ℝ × ℝ := (2, 2)
  apply_similarity s A = {(4, 4), (-4, -4)} := by
  sorry

end NUMINAMATH_CALUDE_similarity_transformation_result_l361_36100


namespace NUMINAMATH_CALUDE_smallest_reunion_time_l361_36198

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_time (t : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.length ≥ 4 ∧ 
    subset.all (λ x => x ∈ horse_lap_times) ∧
    subset.all (λ x => t % x = 0)

theorem smallest_reunion_time :
  ∃ (T : ℕ), T > 0 ∧ is_valid_time T ∧
    ∀ (t : ℕ), 0 < t ∧ t < T → ¬is_valid_time t :=
  sorry

end NUMINAMATH_CALUDE_smallest_reunion_time_l361_36198


namespace NUMINAMATH_CALUDE_collectible_toy_price_changes_l361_36128

/-- Represents the months of the year --/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june

/-- The price change for each month --/
def price_change : Month → ℝ
  | Month.january => -1.00
  | Month.february => 3.50
  | Month.march => -3.00
  | Month.april => 4.50
  | Month.may => -1.50
  | Month.june => -3.50

/-- The month with the greatest price drop --/
def greatest_drop : Month := Month.june

/-- The month with the greatest price increase --/
def greatest_increase : Month := Month.april

theorem collectible_toy_price_changes :
  (∀ m : Month, price_change greatest_drop ≤ price_change m) ∧
  (∀ m : Month, price_change m ≤ price_change greatest_increase) :=
by sorry

end NUMINAMATH_CALUDE_collectible_toy_price_changes_l361_36128


namespace NUMINAMATH_CALUDE_equal_area_triangle_octagon_ratio_l361_36176

/-- The ratio of side lengths of an equilateral triangle and a regular octagon with equal areas -/
theorem equal_area_triangle_octagon_ratio :
  ∀ (s_t s_o : ℝ),
  s_t > 0 → s_o > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2) →
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by sorry


end NUMINAMATH_CALUDE_equal_area_triangle_octagon_ratio_l361_36176


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l361_36169

theorem greatest_x_given_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l361_36169


namespace NUMINAMATH_CALUDE_remainder_of_sum_divided_by_256_l361_36127

theorem remainder_of_sum_divided_by_256 :
  (1234567 + 890123) % 256 = 74 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_divided_by_256_l361_36127


namespace NUMINAMATH_CALUDE_rooms_per_floor_l361_36140

theorem rooms_per_floor (total_earnings : ℕ) (hourly_rate : ℕ) (hours_per_room : ℕ) (num_floors : ℕ)
  (h1 : total_earnings = 3600)
  (h2 : hourly_rate = 15)
  (h3 : hours_per_room = 6)
  (h4 : num_floors = 4) :
  total_earnings / (hourly_rate * num_floors) = 10 := by
  sorry

#check rooms_per_floor

end NUMINAMATH_CALUDE_rooms_per_floor_l361_36140


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l361_36150

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point 8 (-15) = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l361_36150


namespace NUMINAMATH_CALUDE_max_x_minus_y_l361_36188

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l361_36188


namespace NUMINAMATH_CALUDE_journey_equations_correct_l361_36119

/-- Represents a journey between two locations with uphill and flat sections. -/
structure Journey where
  uphill_speed : ℝ
  flat_speed : ℝ
  downhill_speed : ℝ
  time_ab : ℝ
  time_ba : ℝ

/-- The correct system of equations for the journey. -/
def correct_equations (j : Journey) (x y : ℝ) : Prop :=
  x / j.uphill_speed + y / j.flat_speed = j.time_ab / 60 ∧
  y / j.flat_speed + x / j.downhill_speed = j.time_ba / 60

/-- Theorem stating that the given system of equations is correct for the journey. -/
theorem journey_equations_correct (j : Journey) (x y : ℝ) 
    (h1 : j.uphill_speed = 3)
    (h2 : j.flat_speed = 4)
    (h3 : j.downhill_speed = 5)
    (h4 : j.time_ab = 70)
    (h5 : j.time_ba = 54) :
  correct_equations j x y :=
sorry

end NUMINAMATH_CALUDE_journey_equations_correct_l361_36119


namespace NUMINAMATH_CALUDE_art_club_collection_l361_36178

/-- The number of artworks collected by the art club in two school years -/
def artworks_collected (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Theorem stating that the art club collects 240 artworks in two school years -/
theorem art_club_collection :
  artworks_collected 15 2 4 2 = 240 := by
  sorry

#eval artworks_collected 15 2 4 2

end NUMINAMATH_CALUDE_art_club_collection_l361_36178


namespace NUMINAMATH_CALUDE_correct_calculation_l361_36122

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l361_36122


namespace NUMINAMATH_CALUDE_parallelogram_means_input_output_l361_36177

/-- Represents the different symbols used in a program flowchart --/
inductive FlowchartSymbol
  | Parallelogram
  | Rectangle
  | Diamond
  | Oval

/-- Represents the different operations in a program flowchart --/
inductive FlowchartOperation
  | InputOutput
  | Process
  | Decision
  | Start_End

/-- Associates a FlowchartSymbol with its corresponding FlowchartOperation --/
def symbolMeaning : FlowchartSymbol → FlowchartOperation
  | FlowchartSymbol.Parallelogram => FlowchartOperation.InputOutput
  | FlowchartSymbol.Rectangle => FlowchartOperation.Process
  | FlowchartSymbol.Diamond => FlowchartOperation.Decision
  | FlowchartSymbol.Oval => FlowchartOperation.Start_End

theorem parallelogram_means_input_output :
  symbolMeaning FlowchartSymbol.Parallelogram = FlowchartOperation.InputOutput :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_means_input_output_l361_36177


namespace NUMINAMATH_CALUDE_jerrys_age_l361_36182

/-- Given that Mickey's age is 6 years less than 200% of Jerry's age and Mickey is 20 years old, prove that Jerry is 13 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 6) 
  (h2 : mickey_age = 20) : 
  jerry_age = 13 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l361_36182


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l361_36162

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List Int) : List Int :=
  l.filter (λ x => x > 0)

def range (l : List Int) : Int :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 10 →
  range (positive_integers k) = 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l361_36162


namespace NUMINAMATH_CALUDE_inequality_solution_l361_36130

theorem inequality_solution (x : ℝ) :
  (2 - 1 / (3 * x + 4) < 5) ↔ (x < -4/3 ∨ x > -13/9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l361_36130


namespace NUMINAMATH_CALUDE_game_points_difference_l361_36185

theorem game_points_difference (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points > mark_points →
  samanta_points + mark_points + eric_points = 32 →
  samanta_points - mark_points = 8 :=
by sorry

end NUMINAMATH_CALUDE_game_points_difference_l361_36185


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l361_36158

theorem shooting_competition_probability (p_10 p_9 p_8 p_7 : ℝ) 
  (h1 : p_10 = 0.15)
  (h2 : p_9 = 0.35)
  (h3 : p_8 = 0.2)
  (h4 : p_7 = 0.1) :
  p_7 = 0.3 :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l361_36158


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l361_36161

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 + Real.sqrt 6 ∧ x₁^2 - 4*x₁ = 2) ∧ 
              (x₂ = 2 - Real.sqrt 6 ∧ x₂^2 - 4*x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l361_36161


namespace NUMINAMATH_CALUDE_sequence_sum_equals_eight_l361_36105

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum_equals_eight 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a 1) ^ m) 
  (h_arithmetic : ∀ n m : ℕ, b (n + m) = b n + m * (b 1 - b 0)) 
  (h_relation : a 3 * a 11 = 4 * a 7) 
  (h_equal : b 7 = a 7) : 
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_eight_l361_36105


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l361_36133

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l361_36133


namespace NUMINAMATH_CALUDE_limit_nonexistent_l361_36170

/-- The limit of (x^2 - y^2) / (x^2 + y^2) as x and y approach 0 does not exist. -/
theorem limit_nonexistent :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 < δ^2 →
    |((x^2 - y^2) / (x^2 + y^2)) - L| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_nonexistent_l361_36170


namespace NUMINAMATH_CALUDE_final_crayon_count_l361_36181

def crayon_count (initial : ℕ) (added1 : ℕ) (removed : ℕ) (added2 : ℕ) : ℕ :=
  initial + added1 - removed + added2

theorem final_crayon_count :
  crayon_count 25 15 8 12 = 44 := by
  sorry

end NUMINAMATH_CALUDE_final_crayon_count_l361_36181


namespace NUMINAMATH_CALUDE_cubic_product_theorem_l361_36180

theorem cubic_product_theorem : 
  (2^3 - 1) / (2^3 + 1) * 
  (3^3 - 1) / (3^3 + 1) * 
  (4^3 - 1) / (4^3 + 1) * 
  (5^3 - 1) / (5^3 + 1) * 
  (6^3 - 1) / (6^3 + 1) * 
  (7^3 - 1) / (7^3 + 1) = 19 / 56 := by
  sorry

end NUMINAMATH_CALUDE_cubic_product_theorem_l361_36180


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l361_36189

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  is_geometric_sequence a →
  (∀ k, a k > 0) →
  a 3 = a 2 + 2 * a 1 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l361_36189


namespace NUMINAMATH_CALUDE_girl_squirrel_walnuts_l361_36141

theorem girl_squirrel_walnuts (initial : ℕ) (boy_adds : ℕ) (girl_eats : ℕ) (final : ℕ) :
  initial = 12 →
  boy_adds = 5 →
  girl_eats = 2 →
  final = 20 →
  ∃ girl_brings : ℕ, initial + boy_adds + girl_brings - girl_eats = final ∧ girl_brings = 5 :=
by sorry

end NUMINAMATH_CALUDE_girl_squirrel_walnuts_l361_36141


namespace NUMINAMATH_CALUDE_square_of_square_plus_eight_l361_36168

theorem square_of_square_plus_eight : (4^2 + 8)^2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_square_of_square_plus_eight_l361_36168


namespace NUMINAMATH_CALUDE_vector_subtraction_l361_36144

/-- Given complex numbers z1 and z2 representing vectors OA and OB respectively,
    prove that the complex number representing BA is equal to 5-5i. -/
theorem vector_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 - 3*I) (h2 : z2 = -3 + 2*I) :
  z1 - z2 = 5 - 5*I := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l361_36144


namespace NUMINAMATH_CALUDE_can_detect_drum_l361_36110

-- Define the stone type
def Stone : Type := ℕ

-- Define the set of 100 stones
def S : Finset Stone := sorry

-- Define the weight function
def weight : Stone → ℕ := sorry

-- Define the property that all stones have different weights
axiom different_weights : ∀ s₁ s₂ : Stone, s₁ ≠ s₂ → weight s₁ ≠ weight s₂

-- Define a subset of 10 stones
def Subset : Type := Finset Stone

-- Define the property that a subset has exactly 10 stones
def has_ten_stones (subset : Subset) : Prop := subset.card = 10

-- Define the ordering function (by the brownie)
def order_stones (subset : Subset) : List Stone := sorry

-- Define the potential swapping function (by the drum)
def swap_stones (ordered_stones : List Stone) : List Stone := sorry

-- Define the observation function (what Andryusha sees)
def observe (subset : Subset) : List Stone := sorry

-- The main theorem
theorem can_detect_drum :
  ∃ (f : Subset → Bool),
    (∀ subset : Subset, has_ten_stones subset →
      f subset = true ↔ observe subset ≠ order_stones subset) :=
sorry

end NUMINAMATH_CALUDE_can_detect_drum_l361_36110


namespace NUMINAMATH_CALUDE_pond_to_field_ratio_l361_36121

/-- Represents a rectangular field with a square pond inside -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_double_width : field_length = 2 * field_width
  field_length_16 : field_length = 16
  pond_side_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:2 -/
theorem pond_to_field_ratio (f : FieldWithPond) : 
  (f.pond_side ^ 2) / (f.field_length * f.field_width) = 1 / 2 := by
  sorry

#check pond_to_field_ratio

end NUMINAMATH_CALUDE_pond_to_field_ratio_l361_36121


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l361_36116

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths -/
theorem volume_of_specific_tetrahedron :
  tetrahedron_volume 3 4 6 5 (Real.sqrt 37) (2 * Real.sqrt 10) = (4 * Real.sqrt 77) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l361_36116


namespace NUMINAMATH_CALUDE_min_blocks_for_garden_wall_l361_36108

/-- Represents the configuration of a garden wall --/
structure WallConfig where
  length : ℕ
  height : ℕ
  blockHeight : ℕ
  shortBlockLength : ℕ
  longBlockLength : ℕ

/-- Calculates the minimum number of blocks required for the wall --/
def minBlocksRequired (config : WallConfig) : ℕ :=
  sorry

/-- The specific wall configuration from the problem --/
def gardenWall : WallConfig :=
  { length := 90
  , height := 8
  , blockHeight := 1
  , shortBlockLength := 2
  , longBlockLength := 3 }

/-- Theorem stating that the minimum number of blocks required is 244 --/
theorem min_blocks_for_garden_wall :
  minBlocksRequired gardenWall = 244 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_garden_wall_l361_36108


namespace NUMINAMATH_CALUDE_trees_planted_total_l361_36183

/-- Calculates the total number of trees planted given the number of apricot trees and the ratio of peach to apricot trees. -/
def total_trees (apricot_trees : ℕ) (peach_to_apricot_ratio : ℕ) : ℕ :=
  apricot_trees + peach_to_apricot_ratio * apricot_trees

/-- Theorem stating that given the specific conditions, the total number of trees planted is 232. -/
theorem trees_planted_total : total_trees 58 3 = 232 := by
  sorry

#eval total_trees 58 3

end NUMINAMATH_CALUDE_trees_planted_total_l361_36183


namespace NUMINAMATH_CALUDE_brick_height_calculation_l361_36143

/-- Prove that the height of each brick is 67.5 cm, given the wall dimensions,
    brick dimensions (except height), and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℕ) :
  wall_length = 900 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 7200 →
  ∃ (brick_height : ℝ),
    brick_height = 67.5 ∧
    wall_length * wall_width * wall_height =
      num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l361_36143


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_120_504_l361_36112

theorem lcm_gcf_ratio_120_504 : 
  (Nat.lcm 120 504) / (Nat.gcd 120 504) = 105 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_120_504_l361_36112


namespace NUMINAMATH_CALUDE_windshield_wiper_transformation_l361_36103

/-- Represents a geometric object --/
inductive GeometricObject
  | Point
  | Line
  | Surface
  | Volume

/-- Represents a geometric transformation --/
inductive GeometricTransformation
  | PointToLine
  | LineToSurface
  | SurfaceToVolume

/-- Represents a windshield wiper --/
structure Wiper where
  shape : GeometricObject

/-- Represents a car windshield --/
structure Windshield where
  shape : GeometricObject

/-- Represents the action of a wiper on a windshield --/
def wiperAction (w : Wiper) (s : Windshield) : GeometricTransformation :=
  match w.shape, s.shape with
  | GeometricObject.Line, GeometricObject.Surface => GeometricTransformation.LineToSurface
  | _, _ => GeometricTransformation.PointToLine  -- Default case, not relevant for our problem

theorem windshield_wiper_transformation (w : Wiper) (s : Windshield) 
  (h1 : w.shape = GeometricObject.Line) 
  (h2 : s.shape = GeometricObject.Surface) : 
  wiperAction w s = GeometricTransformation.LineToSurface := by
  sorry

end NUMINAMATH_CALUDE_windshield_wiper_transformation_l361_36103


namespace NUMINAMATH_CALUDE_wall_length_calculation_l361_36166

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove the length of the wall. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side ^ 2) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l361_36166


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l361_36151

def monthly_salary : ℚ := 4166.67
def initial_savings_rate : ℚ := 0.20
def new_savings : ℚ := 500

def initial_savings : ℚ := monthly_salary * initial_savings_rate
def original_expenses : ℚ := monthly_salary - initial_savings
def increase_in_expenses : ℚ := initial_savings - new_savings
def percentage_increase : ℚ := (increase_in_expenses / original_expenses) * 100

theorem expense_increase_percentage :
  percentage_increase = 10 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l361_36151


namespace NUMINAMATH_CALUDE_decagon_enclosure_l361_36102

theorem decagon_enclosure (m n : ℕ) (h1 : m = 10) : 
  (∀ k : ℕ, k ∈ Finset.range m → 
    (360 : ℝ) / n = 2 * ((m - 2 : ℝ) * 180 / m)) → n = 5 :=
by sorry

end NUMINAMATH_CALUDE_decagon_enclosure_l361_36102


namespace NUMINAMATH_CALUDE_inequality_theorem_l361_36111

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l361_36111


namespace NUMINAMATH_CALUDE_custom_op_result_l361_36123

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 12 8) 2 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l361_36123


namespace NUMINAMATH_CALUDE_coconut_trips_l361_36135

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l361_36135


namespace NUMINAMATH_CALUDE_project_completion_time_l361_36160

/-- Calculates the number of days needed to complete a project given extra hours,
    normal work hours, and total project hours. -/
def days_to_complete_project (extra_hours : ℕ) (normal_hours : ℕ) (project_hours : ℕ) : ℕ :=
  project_hours / (normal_hours + extra_hours)

/-- Theorem stating that under the given conditions, it takes 100 days to complete the project. -/
theorem project_completion_time :
  days_to_complete_project 5 10 1500 = 100 := by
  sorry

#eval days_to_complete_project 5 10 1500

end NUMINAMATH_CALUDE_project_completion_time_l361_36160


namespace NUMINAMATH_CALUDE_first_day_over_500_day_is_saturday_l361_36131

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 :
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 :=
by sorry

theorem day_is_saturday : 
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 → k = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_500_day_is_saturday_l361_36131


namespace NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l361_36154

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- The original point P -/
def P : Point2D := { x := 4, y := 1 }

/-- Theorem: The reflection of P(4,1) across the y-axis is (-4,1) -/
theorem reflection_of_P_across_y_axis :
  reflectAcrossYAxis P = { x := -4, y := 1 } := by
  sorry


end NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l361_36154


namespace NUMINAMATH_CALUDE_special_parallelogram_sides_l361_36142

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in radians
  acute_angle : ℝ
  -- The ratio of the parts of the obtuse angle divided by the diagonal
  obtuse_angle_ratio : ℝ
  -- The length of the shorter side
  short_side : ℝ
  -- The length of the longer side
  long_side : ℝ
  -- The perimeter is 90 cm
  perimeter_eq : perimeter = 90
  -- The acute angle is 60 degrees (π/3 radians)
  acute_angle_eq : acute_angle = π / 3
  -- The obtuse angle is divided in a 1:3 ratio
  obtuse_angle_ratio_eq : obtuse_angle_ratio = 1 / 3
  -- The perimeter is the sum of all sides
  perimeter_sum : perimeter = 2 * (short_side + long_side)
  -- The shorter side is half the longer side (derived from the 60° angle)
  side_ratio : short_side = long_side / 2

/-- Theorem: The sides of the special parallelogram are 15 cm and 30 cm -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.short_side = 15 ∧ p.long_side = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_sides_l361_36142


namespace NUMINAMATH_CALUDE_rational_equation_solution_l361_36165

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0) ↔ 
  (x = 3 ∨ x = 1 ∨ x = -3 ∨ x = -9) := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l361_36165


namespace NUMINAMATH_CALUDE_midpoint_condition_l361_36187

-- Define the triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  hyp_eq : hypotenuse = Real.sqrt (a^2 + b^2)

-- Define point P on the hypotenuse
def PointOnHypotenuse (triangle : RightTriangle) := 
  { x : ℝ // 0 ≤ x ∧ x ≤ triangle.hypotenuse }

-- Define s as AP² + PB²
def s (triangle : RightTriangle) (p : PointOnHypotenuse triangle) : ℝ :=
  p.val^2 + (triangle.hypotenuse - p.val)^2

-- Define CP²
def CP_squared (triangle : RightTriangle) : ℝ := triangle.a^2

-- Theorem statement
theorem midpoint_condition (triangle : RightTriangle) :
  ∀ p : PointOnHypotenuse triangle, 
    s triangle p = 2 * CP_squared triangle ↔ 
    p.val = triangle.hypotenuse / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_condition_l361_36187


namespace NUMINAMATH_CALUDE_pizza_meat_calculation_l361_36101

/-- Represents the number of pieces of each type of meat on a pizza --/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Calculates the total number of pieces of meat on each slice of pizza --/
def meat_per_slice (toppings : PizzaToppings) (slices : ℕ) : ℚ :=
  (toppings.pepperoni + toppings.ham + toppings.sausage : ℚ) / slices

theorem pizza_meat_calculation :
  let toppings : PizzaToppings := {
    pepperoni := 30,
    ham := 30 * 2,
    sausage := 30 + 12
  }
  let slices : ℕ := 6
  meat_per_slice toppings slices = 22 := by
  sorry

#eval meat_per_slice { pepperoni := 30, ham := 30 * 2, sausage := 30 + 12 } 6

end NUMINAMATH_CALUDE_pizza_meat_calculation_l361_36101


namespace NUMINAMATH_CALUDE_number117_is_1983_l361_36153

/-- The set of digits used to form the four-digit numbers -/
def digits : Finset Nat := {1, 3, 4, 5, 7, 8, 9}

/-- A four-digit number formed from the given digits without repetition -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d1 + 100 * n.d2 + 10 * n.d3 + n.d4

/-- The set of all valid four-digit numbers -/
def validNumbers : Finset FourDigitNumber := sorry

/-- The 117th number in the ascending sequence of valid four-digit numbers -/
def number117 : FourDigitNumber := sorry

theorem number117_is_1983 : number117.value = 1983 := by sorry

end NUMINAMATH_CALUDE_number117_is_1983_l361_36153


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l361_36194

theorem count_divisible_numbers : 
  let upper_bound := 242400
  let divisor := 303
  (Finset.filter 
    (fun k => (k^2 + 2*k) % divisor = 0) 
    (Finset.range (upper_bound + 1))).card = 3200 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l361_36194


namespace NUMINAMATH_CALUDE_grid_triangle_square_l361_36190

/-- A point on a 2D grid represented by integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle formed by three grid points -/
def triangleArea (A B C : GridPoint) : ℚ := sorry

/-- The squared distance between two grid points -/
def squaredDistance (A B : GridPoint) : ℤ := sorry

/-- Predicate to check if three grid points form three vertices of a square -/
def formSquareVertices (A B C : GridPoint) : Prop := sorry

theorem grid_triangle_square (A B C : GridPoint) :
  let T := triangleArea A B C
  (squaredDistance A B + squaredDistance B C)^2 < 8 * T + 1 →
  formSquareVertices A B C := by
  sorry

end NUMINAMATH_CALUDE_grid_triangle_square_l361_36190


namespace NUMINAMATH_CALUDE_travelers_checks_denomination_l361_36172

theorem travelers_checks_denomination
  (total_checks : ℕ)
  (total_worth : ℚ)
  (spent_checks : ℕ)
  (remaining_avg : ℚ)
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : spent_checks = 18)
  (h4 : remaining_avg = 75)
  (h5 : (total_checks - spent_checks) * remaining_avg + spent_checks * denomination = total_worth) :
  denomination = 50 := by
  sorry

end NUMINAMATH_CALUDE_travelers_checks_denomination_l361_36172


namespace NUMINAMATH_CALUDE_orange_stack_count_l361_36186

/-- Calculates the number of oranges in a triangular layer -/
def orangesInLayer (a b : ℕ) : ℕ := (a * b) / 2

/-- Calculates the total number of oranges in the stack -/
def totalOranges (baseWidth baseLength : ℕ) : ℕ :=
  let rec sumLayers (width length : ℕ) : ℕ :=
    if width = 0 ∨ length = 0 then 0
    else orangesInLayer width length + sumLayers (width - 1) (length - 1)
  sumLayers baseWidth baseLength

theorem orange_stack_count :
  totalOranges 6 9 = 78 := by
  sorry

#eval totalOranges 6 9  -- Should output 78

end NUMINAMATH_CALUDE_orange_stack_count_l361_36186


namespace NUMINAMATH_CALUDE_square_sum_implies_product_l361_36136

theorem square_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (15 - x) = 8) →
  ((10 + x) * (15 - x) = 1521 / 4) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l361_36136


namespace NUMINAMATH_CALUDE_certificate_recipients_l361_36109

theorem certificate_recipients (total : ℕ) (difference : ℕ) (recipients : ℕ) : 
  total = 120 → 
  difference = 36 → 
  recipients = total / 2 + difference / 2 → 
  recipients = 78 := by
sorry

end NUMINAMATH_CALUDE_certificate_recipients_l361_36109


namespace NUMINAMATH_CALUDE_complex_conversion_l361_36192

theorem complex_conversion :
  3 * Real.sqrt 2 * Complex.exp ((-5 * π * Complex.I) / 4) = -3 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_conversion_l361_36192


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l361_36124

theorem trigonometric_expressions (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6) ∧ 
  (Real.sin α ^ 2 + Real.sin (2 * α) = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l361_36124


namespace NUMINAMATH_CALUDE_ferris_break_length_is_correct_l361_36104

/-- Represents the job completion scenario with Audrey and Ferris --/
structure JobCompletion where
  audrey_solo_time : ℝ
  ferris_solo_time : ℝ
  collaboration_time : ℝ
  ferris_break_count : ℕ

/-- Calculates the length of each of Ferris' breaks in minutes --/
def ferris_break_length (job : JobCompletion) : ℝ :=
  2.5

/-- Theorem stating that Ferris' break length is 2.5 minutes under the given conditions --/
theorem ferris_break_length_is_correct (job : JobCompletion) 
  (h1 : job.audrey_solo_time = 4)
  (h2 : job.ferris_solo_time = 3)
  (h3 : job.collaboration_time = 2)
  (h4 : job.ferris_break_count = 6) :
  ferris_break_length job = 2.5 := by
  sorry

#eval ferris_break_length { audrey_solo_time := 4, ferris_solo_time := 3, collaboration_time := 2, ferris_break_count := 6 }

end NUMINAMATH_CALUDE_ferris_break_length_is_correct_l361_36104


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l361_36193

theorem complex_magnitude_example : Complex.abs (Complex.mk (7/8) 3) = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l361_36193
