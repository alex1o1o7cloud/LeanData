import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l3461_346109

def solution_set : Set ℝ :=
  {x | x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)}

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3461_346109


namespace NUMINAMATH_CALUDE_triangle_side_value_l3461_346167

/-- Triangle inequality theorem for a triangle with sides 2, 3, and m -/
def triangle_inequality (m : ℝ) : Prop :=
  2 + 3 > m ∧ 2 + m > 3 ∧ 3 + m > 2

/-- The only valid integer value for m is 3 -/
theorem triangle_side_value :
  ∀ m : ℕ, triangle_inequality (m : ℝ) ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3461_346167


namespace NUMINAMATH_CALUDE_sum_base4_equals_l3461_346166

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a decimal number to its base 4 representation as a list of digits -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals : 
  let a := base4ToDecimal [2, 0, 1]
  let b := base4ToDecimal [1, 3, 2]
  let c := base4ToDecimal [3, 0, 3]
  let d := base4ToDecimal [2, 2, 1]
  decimalToBase4 (a + b + c + d) = [0, 1, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_l3461_346166


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l3461_346156

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 36 →
  side = 12 →
  median = 10 →
  area = 1/2 * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l3461_346156


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3461_346134

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3461_346134


namespace NUMINAMATH_CALUDE_line_intercept_sum_l3461_346188

/-- Given a line with equation 2x - 5y + 10 = 0, prove that the absolute value of the sum of its x and y intercepts is 3. -/
theorem line_intercept_sum (a b : ℝ) : 
  (2 * a - 5 * 0 + 10 = 0) →  -- x-intercept condition
  (2 * 0 - 5 * b + 10 = 0) →  -- y-intercept condition
  |a + b| = 3 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l3461_346188


namespace NUMINAMATH_CALUDE_max_weight_proof_l3461_346162

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kg -/
def min_crate_weight : ℕ := 120

/-- The maximum weight of crates on a single trip in kg -/
def max_trip_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_trip_weight = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l3461_346162


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3461_346137

theorem quadratic_polynomial_unique (q : ℝ → ℝ) :
  (q = λ x => (67/30) * x^2 - (39/10) * x - 2/15) ↔
  (q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 20) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3461_346137


namespace NUMINAMATH_CALUDE_h_domain_l3461_346160

noncomputable def h (x : ℝ) : ℝ := (x^2 - 9) / (|x - 4| + x^2 - 1)

def domain_of_h : Set ℝ := {x | x < (1 + Real.sqrt 13) / 2 ∨ x > (1 + Real.sqrt 13) / 2}

theorem h_domain : 
  {x : ℝ | ∃ y, h x = y} = domain_of_h :=
by sorry

end NUMINAMATH_CALUDE_h_domain_l3461_346160


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3461_346121

theorem power_fraction_equality : (10 ^ 20 : ℚ) / (50 ^ 10) = 2 ^ 10 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3461_346121


namespace NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_proof_l3461_346146

/-- The perimeter of a rectangle with width 16 and length 19 is 70 -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 16, 19 => 70
  | _, _ => 0  -- Default case for other inputs

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def perimeter (width length : ℕ) : ℕ := 2 * (width + length)

theorem rectangle_perimeter_proof (width length : ℕ) (h1 : width = 16) (h2 : length = 19) :
  perimeter width length = rectangle_perimeter width length := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_proof_l3461_346146


namespace NUMINAMATH_CALUDE_john_scores_42_points_l3461_346153

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (points_per_interval : ℕ) (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_duration := num_periods * period_duration
  let num_intervals := total_duration / interval_duration
  points_per_interval * num_intervals

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  let points_per_interval := 2 * 2 + 1 * 3  -- 2 two-point shots and 1 three-point shot
  let interval_duration := 4                -- every 4 minutes
  let num_periods := 2                      -- 2 periods
  let period_duration := 12                 -- each period is 12 minutes
  total_points_scored points_per_interval interval_duration num_periods period_duration = 42 := by
  sorry


end NUMINAMATH_CALUDE_john_scores_42_points_l3461_346153


namespace NUMINAMATH_CALUDE_parabola_c_value_l3461_346180

/-- A parabola in the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ := p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 2 = -4 →   -- vertex (-4, 2)
  p.x_coord 4 = -2 →   -- point (-2, 4)
  p.x_coord 0 = -2 →   -- point (-2, 0)
  p.c = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3461_346180


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3461_346117

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ∧ x ≠ 3 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioo (-2) 2 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3461_346117


namespace NUMINAMATH_CALUDE_expression_evaluation_l3461_346196

theorem expression_evaluation : 
  let a : ℚ := 2
  let b : ℚ := 1/3
  3*(a^2 - a*b + 7) - 2*(3*a*b - a^2 + 1) + 3 = 36 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3461_346196


namespace NUMINAMATH_CALUDE_father_sons_average_age_l3461_346138

/-- The average age of a father and his two sons -/
def average_age (father_age son1_age son2_age : ℕ) : ℚ :=
  (father_age + son1_age + son2_age) / 3

/-- Theorem stating the average age of the father and his two sons -/
theorem father_sons_average_age :
  ∀ (father_age son1_age son2_age : ℕ),
  father_age = 32 →
  son1_age - son2_age = 4 →
  (son1_age - 5 + son2_age - 5) / 2 = 15 →
  average_age father_age son1_age son2_age = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_father_sons_average_age_l3461_346138


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3461_346131

/-- Given an arithmetic sequence where the first term is 3 and the 25th term is 51,
    prove that the 75th term is 151. -/
theorem arithmetic_sequence_75th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                -- first term is 3
    a 24 = 51 →                              -- 25th term is 51
    a 74 = 151 :=                            -- 75th term is 151
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3461_346131


namespace NUMINAMATH_CALUDE_jake_peaches_l3461_346132

/-- Given the number of peaches each person has, prove Jake has 17 peaches -/
theorem jake_peaches (jill steven jake : ℕ) 
  (h1 : jake + 6 = steven)
  (h2 : steven = jill + 18)
  (h3 : jill = 5) : 
  jake = 17 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l3461_346132


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l3461_346199

theorem parallelogram_sides_sum (x y : ℝ) : 
  (4*x + 4 = 18) → (15*y - 3 = 12) → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l3461_346199


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l3461_346170

theorem consecutive_product_not_power (n m : ℕ) (h : m ≥ 2) :
  ¬ ∃ a : ℕ, n * (n + 1) = a ^ m :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l3461_346170


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3461_346122

-- Define set A
def A : Set ℝ := {x | Real.sqrt (x^2 - 1) / Real.sqrt x = 0}

-- Define set B
def B : Set ℝ := {y | -2 ≤ y ∧ y ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3461_346122


namespace NUMINAMATH_CALUDE_hcf_problem_l3461_346194

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 84942) (h2 : Nat.lcm a b = 2574) :
  Nat.gcd a b = 33 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3461_346194


namespace NUMINAMATH_CALUDE_sprint_jog_difference_l3461_346135

-- Define the distances
def sprint_distance : ℚ := 875 / 1000
def jog_distance : ℚ := 75 / 100

-- Theorem statement
theorem sprint_jog_difference :
  sprint_distance - jog_distance = 125 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sprint_jog_difference_l3461_346135


namespace NUMINAMATH_CALUDE_max_b_line_circle_intersection_l3461_346178

/-- The maximum value of b for a line intersecting a circle under specific conditions -/
theorem max_b_line_circle_intersection (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∃ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ ∧ 
    (P₁.1^2 + P₁.2^2 = 4) ∧ 
    (P₂.1^2 + P₂.2^2 = 4) ∧ 
    (P₁.2 = P₁.1 + b) ∧ 
    (P₂.2 = P₂.1 + b))
  (h3 : ∀ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ → 
    (P₁.1^2 + P₁.2^2 = 4) → 
    (P₂.1^2 + P₂.2^2 = 4) → 
    (P₁.2 = P₁.1 + b) → 
    (P₂.2 = P₂.1 + b) → 
    ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 ≥ (P₁.1 + P₂.1)^2 + (P₁.2 + P₂.2)^2)) : 
  b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_b_line_circle_intersection_l3461_346178


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l3461_346106

def scores : List ℕ := [91, 82, 68, 99, 79, 86, 88, 76, 71, 58, 80, 89, 65, 85, 93]

def is_b_grade (score : ℕ) : Bool :=
  87 ≤ score && score ≤ 94

def count_b_grades (scores : List ℕ) : ℕ :=
  scores.filter is_b_grade |>.length

theorem percentage_of_b_grades :
  (count_b_grades scores : ℚ) / (scores.length : ℚ) * 100 = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l3461_346106


namespace NUMINAMATH_CALUDE_absolute_value_of_T_l3461_346141

def i : ℂ := Complex.I

def T : ℂ := (1 + i)^18 + (1 - i)^18

theorem absolute_value_of_T : Complex.abs T = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_T_l3461_346141


namespace NUMINAMATH_CALUDE_birds_on_fence_l3461_346174

theorem birds_on_fence : 
  let initial_birds : ℕ := 12
  let additional_birds : ℕ := 8
  let num_groups : ℕ := 3
  let birds_per_group : ℕ := 6
  initial_birds + additional_birds + num_groups * birds_per_group = 38 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3461_346174


namespace NUMINAMATH_CALUDE_circle_sum_inequality_l3461_346101

theorem circle_sum_inequality (a : Fin 100 → ℝ) (h : Function.Injective a) :
  ∃ i : Fin 100, a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100) := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_inequality_l3461_346101


namespace NUMINAMATH_CALUDE_maries_trip_l3461_346105

theorem maries_trip (total_distance : ℚ) 
  (h1 : total_distance / 4 + 15 + total_distance / 6 = total_distance) : 
  total_distance = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_maries_trip_l3461_346105


namespace NUMINAMATH_CALUDE_expression_evaluation_l3461_346185

theorem expression_evaluation : -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3461_346185


namespace NUMINAMATH_CALUDE_equation_solution_l3461_346173

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3461_346173


namespace NUMINAMATH_CALUDE_valid_B_l3461_346123

-- Define set A
def A : Set ℝ := {x | x ≥ 0}

-- Define the property that A ∩ B = B
def intersectionProperty (B : Set ℝ) : Prop := A ∩ B = B

-- Define the set {1,2}
def candidateB : Set ℝ := {1, 2}

-- Theorem statement
theorem valid_B : intersectionProperty candidateB := by sorry

end NUMINAMATH_CALUDE_valid_B_l3461_346123


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l3461_346161

def total_money : ℕ := 1050  -- £10.50 in pence
def notebook_cost : ℕ := 75  -- £0.75 in pence

theorem max_notebooks_purchasable :
  ∀ n : ℕ, n * notebook_cost ≤ total_money →
  n ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l3461_346161


namespace NUMINAMATH_CALUDE_intersection_of_ellipses_l3461_346193

theorem intersection_of_ellipses :
  ∃! (points : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ points ↔ (x^2 + 9*y^2 = 9 ∧ 9*x^2 + y^2 = 1)) ∧
    points.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_ellipses_l3461_346193


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3461_346149

structure Department where
  total : ℕ
  males : ℕ
  females : ℕ

def sample_size : ℕ := 3

def dept_A : Department := ⟨10, 6, 4⟩
def dept_B : Department := ⟨5, 3, 2⟩

def total_staff : ℕ := dept_A.total + dept_B.total

def stratified_sample (d : Department) : ℕ :=
  (sample_size * d.total) / total_staff

def prob_at_least_one_female (d : Department) (n : ℕ) : ℚ :=
  1 - (Nat.choose d.males n : ℚ) / (Nat.choose d.total n : ℚ)

def prob_male_count (k : ℕ) : ℚ := 
  if k = 0 then 4 / 75
  else if k = 1 then 22 / 75
  else if k = 2 then 34 / 75
  else if k = 3 then 1 / 3
  else 0

def expected_male_count : ℚ := 2

theorem stratified_sampling_theorem :
  (stratified_sample dept_A = 2) ∧
  (stratified_sample dept_B = 1) ∧
  (prob_at_least_one_female dept_A 2 = 2 / 3) ∧
  (∀ k, 0 ≤ k ∧ k ≤ 3 → prob_male_count k = prob_male_count k) ∧
  (Finset.sum (Finset.range 4) (λ k => k * prob_male_count k) = expected_male_count) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3461_346149


namespace NUMINAMATH_CALUDE_max_x_value_l3461_346127

theorem max_x_value (x y z : ℝ) 
  (eq1 : 3 * x + 2 * y + z = 10) 
  (eq2 : x * y + x * z + y * z = 6) : 
  x ≤ 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l3461_346127


namespace NUMINAMATH_CALUDE_work_completion_time_l3461_346104

theorem work_completion_time (work : ℝ) (time_renu : ℝ) (time_suma : ℝ) 
  (h1 : time_renu = 8) 
  (h2 : time_suma = 8) 
  (h3 : work > 0) :
  let rate_renu := work / time_renu
  let rate_suma := work / time_suma
  let combined_rate := rate_renu + rate_suma
  work / combined_rate = 4 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l3461_346104


namespace NUMINAMATH_CALUDE_grape_juice_amount_l3461_346163

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem: The amount of grape juice in the drink is 105 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h1 : drink.total = 300)
  (h2 : drink.orange_percent = 0.25)
  (h3 : drink.watermelon_percent = 0.40)
  (h4 : drink.grape_ounces = drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)) :
  drink.grape_ounces = 105 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_amount_l3461_346163


namespace NUMINAMATH_CALUDE_time_to_see_again_is_75_l3461_346186

/-- The time before Jenny and Kenny can see each other again -/
def time_to_see_again : ℝ → Prop := λ t =>
  let jenny_speed := 2 -- feet per second
  let kenny_speed := 4 -- feet per second
  let path_distance := 300 -- feet
  let building_diameter := 200 -- feet
  let initial_distance := 300 -- feet
  let jenny_position := λ t : ℝ => (-100 + jenny_speed * t, path_distance / 2)
  let kenny_position := λ t : ℝ => (-100 + kenny_speed * t, -path_distance / 2)
  let building_center := (0, 0)
  let building_radius := building_diameter / 2

  -- Line equation connecting Jenny and Kenny
  let line_equation := λ x y : ℝ =>
    y = -(path_distance / t) * x + path_distance - (initial_distance * path_distance / (2 * t))

  -- Circle equation representing the building
  let circle_equation := λ x y : ℝ =>
    x^2 + y^2 = building_radius^2

  -- Tangent condition
  let tangent_condition := λ x y : ℝ =>
    x * t = path_distance / 2 * y

  -- Point of tangency satisfies both line and circle equations
  ∃ x y : ℝ, line_equation x y ∧ circle_equation x y ∧ tangent_condition x y

theorem time_to_see_again_is_75 : time_to_see_again 75 :=
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_75_l3461_346186


namespace NUMINAMATH_CALUDE_car_speed_l3461_346139

/-- Theorem: Given a car travels 300 miles in 5 hours, its speed is 60 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 300) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3461_346139


namespace NUMINAMATH_CALUDE_balloon_count_correct_l3461_346195

/-- The number of red balloons Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons Sam has -/
def sam_balloons : ℕ := 46

/-- The number of red balloons Dan has -/
def dan_balloons : ℕ := 16

/-- The total number of red balloons -/
def total_balloons : ℕ := 72

theorem balloon_count_correct : fred_balloons + sam_balloons + dan_balloons = total_balloons := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_correct_l3461_346195


namespace NUMINAMATH_CALUDE_complex_number_extrema_l3461_346120

theorem complex_number_extrema (x y : ℝ) (z : ℂ) (h : z = x + y * I) 
  (h_bound : Complex.abs (z - I) ≤ 1) :
  let A := x * (Complex.abs (z - I)^2 - 1)
  ∃ (z_max z_min : ℂ),
    (∀ w : ℂ, Complex.abs (w - I) ≤ 1 → 
      x * (Complex.abs (w - I)^2 - 1) ≤ 2 * Real.sqrt 3 / 9) ∧
    (∀ w : ℂ, Complex.abs (w - I) ≤ 1 → 
      x * (Complex.abs (w - I)^2 - 1) ≥ -2 * Real.sqrt 3 / 9) ∧
    z_max = Real.sqrt 3 / 3 + I ∧
    z_min = -Real.sqrt 3 / 3 + I ∧
    x * (Complex.abs (z_max - I)^2 - 1) = 2 * Real.sqrt 3 / 9 ∧
    x * (Complex.abs (z_min - I)^2 - 1) = -2 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_extrema_l3461_346120


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l3461_346158

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) :=
  (h : R > r)

/-- Points on the circles -/
structure Points (R r : ℝ) extends ConcentricCircles R r :=
  (P : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = r^2)
  (hA : A.1^2 + A.2^2 = r^2)
  (hB : B.1^2 + B.2^2 = R^2)
  (hC : C.1^2 + C.2^2 = R^2)
  (hPerp : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)

/-- The theorem to be proved -/
theorem concentric_circles_theorem (R r : ℝ) (pts : Points R r) :
  let BC := (pts.B.1 - pts.C.1)^2 + (pts.B.2 - pts.C.2)^2
  let CA := (pts.C.1 - pts.A.1)^2 + (pts.C.2 - pts.A.2)^2
  let AB := (pts.A.1 - pts.B.1)^2 + (pts.A.2 - pts.B.2)^2
  let midpoint := ((pts.A.1 + pts.B.1) / 2, (pts.A.2 + pts.B.2) / 2)
  (BC + CA + AB = 6 * R^2 + 2 * r^2) ∧
  ((midpoint.1 + r/2)^2 + midpoint.2^2 = (R/2)^2) :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l3461_346158


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3461_346119

theorem system_of_equations_solution :
  let x : ℚ := -53/3
  let y : ℚ := -38/9
  (7 * x - 30 * y = 3) ∧ (3 * y - x = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3461_346119


namespace NUMINAMATH_CALUDE_problem_solution_l3461_346107

theorem problem_solution : ∃ n : ℕ, 2^13 - 2^11 = 3 * n ∧ n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3461_346107


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3461_346126

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3 / 4 = s / 2) → (3 * s = 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3461_346126


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l3461_346111

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The function f(x) = x³ + x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

theorem monotonic_cubic_function (m : ℝ) :
  IsMonotonic (f m) ↔ m ∈ Set.Ici (1/3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l3461_346111


namespace NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l3461_346114

/-- Given three consecutive natural numbers whose sum is 30, the middle number is 10. -/
theorem middle_of_three_consecutive_sum_30 :
  ∀ n : ℕ, n + (n + 1) + (n + 2) = 30 → n + 1 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l3461_346114


namespace NUMINAMATH_CALUDE_median_length_triangle_l3461_346191

/-- Given a triangle ABC with sides CB = 7, AC = 8, and AB = 9, 
    the length of the median to side AC is 7. -/
theorem median_length_triangle (A B C : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d B C = 7 ∧ d A C = 8 ∧ d A B = 9 →
  let D := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)  -- midpoint of AC
  d B D = 7 := by
sorry

end NUMINAMATH_CALUDE_median_length_triangle_l3461_346191


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3461_346184

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3461_346184


namespace NUMINAMATH_CALUDE_empty_seats_in_theater_l3461_346110

theorem empty_seats_in_theater (total_seats people_watching : ℕ) 
  (h1 : total_seats = 750)
  (h2 : people_watching = 532) :
  total_seats - people_watching = 218 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_in_theater_l3461_346110


namespace NUMINAMATH_CALUDE_fraction_closest_to_longest_side_specific_trapezoid_l3461_346152

/-- Represents a trapezoid field -/
structure TrapezoidField where
  base1 : ℝ
  base2 : ℝ
  angle1 : ℝ
  angle2 : ℝ

/-- The fraction of area closer to the longest side of the trapezoid field -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

/-- Theorem stating the fraction of area closest to the longest side for the given trapezoid -/
theorem fraction_closest_to_longest_side_specific_trapezoid :
  let field : TrapezoidField := {
    base1 := 200,
    base2 := 100,
    angle1 := 45,
    angle2 := 135
  }
  fraction_closest_to_longest_side field = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_closest_to_longest_side_specific_trapezoid_l3461_346152


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3461_346103

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 5 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 ≥ C*(x + y + 2)) ∧
  (∀ (D : ℝ), D > C → ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 < D*(x + y + 2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3461_346103


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3461_346155

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 - 4*m - 2 = 0) → (n^2 - 4*n - 2 = 0) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3461_346155


namespace NUMINAMATH_CALUDE_not_all_vertices_lattice_points_l3461_346190

/-- A polygon with 1994 sides where the length of the k-th side is √(4 + k^2) -/
structure Polygon1994 where
  vertices : Fin 1994 → ℤ × ℤ
  side_length : ∀ k : Fin 1994, Real.sqrt (4 + k.val ^ 2) = 
    Real.sqrt ((vertices (k + 1)).1 - (vertices k).1) ^ 2 + ((vertices (k + 1)).2 - (vertices k).2) ^ 2

/-- Theorem stating that it's impossible for all vertices of the polygon to be lattice points -/
theorem not_all_vertices_lattice_points (p : Polygon1994) : False := by
  sorry

end NUMINAMATH_CALUDE_not_all_vertices_lattice_points_l3461_346190


namespace NUMINAMATH_CALUDE_missing_number_proof_l3461_346113

theorem missing_number_proof : 
  ∃ x : ℝ, 248 + x - Real.sqrt (- Real.sqrt 0) = 16 ∧ x = -232 := by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3461_346113


namespace NUMINAMATH_CALUDE_star_commutative_iff_on_lines_l3461_346181

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem statement
theorem star_commutative_iff_on_lines (x y : ℝ) :
  star x y = star y x ↔ x = 0 ∨ y = 0 ∨ x + y = 0 ∨ x = y :=
sorry

end NUMINAMATH_CALUDE_star_commutative_iff_on_lines_l3461_346181


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3461_346108

/-- 
Given an isosceles triangle with side lengths m, n, and 4, where m and n are 
roots of x^2 - 6x + k + 2 = 0, prove that k = 7 or k = 6.
-/
theorem isosceles_triangle_quadratic_roots (m n k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- m and n are positive (side lengths)
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  m^2 - 6*m + k + 2 = 0 →  -- m is a root
  n^2 - 6*n + k + 2 = 0 →  -- n is a root
  k = 7 ∨ k = 6 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3461_346108


namespace NUMINAMATH_CALUDE_root_of_cubic_equation_l3461_346150

theorem root_of_cubic_equation :
  ∃ x : ℝ, (1/2 : ℝ) * x^3 + 4 = 0 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_of_cubic_equation_l3461_346150


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l3461_346133

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

/-- Calculates the length of a line segment -/
def length (P Q : Point) : ℝ := sorry

/-- Checks if a point is on a line segment -/
def on_segment (P Q R : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem trapezoid_triangle_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (h1 : perpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : length ABCD.A ABCD.D = 4)
  (h3 : length ABCD.A ABCD.B = 4)
  (h4 : length ABCD.D ABCD.C = 10)
  (h5 : on_segment E ABCD.D ABCD.C)
  (h6 : length ABCD.D E = 7)
  (h7 : parallel ABCD.B E ABCD.A ABCD.D) :
  triangle_area ABCD.B E ABCD.C = 6 := by sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l3461_346133


namespace NUMINAMATH_CALUDE_converse_statement_l3461_346154

theorem converse_statement (a b : ℝ) :
  (∀ a b, a > 1 ∧ b > 1 → a + b > 2) →
  (∀ a b, a + b ≤ 2 → a ≤ 1 ∨ b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_statement_l3461_346154


namespace NUMINAMATH_CALUDE_rectangle_to_square_dimension_l3461_346143

/-- Given a rectangle with dimensions 10 and 15, when cut into two congruent hexagons
    and repositioned to form a square, half the length of the square's side is (5√6)/2. -/
theorem rectangle_to_square_dimension (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (square_side : ℝ) (y : ℝ) :
  rectangle_width = 10 →
  rectangle_height = 15 →
  square_side^2 = rectangle_width * rectangle_height →
  y = square_side / 2 →
  y = (5 * Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_dimension_l3461_346143


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l3461_346182

/-- The probability of picking 3 red balls from a bag containing 7 red, 9 blue, and 5 green balls -/
theorem probability_three_red_balls (red blue green : ℕ) (total : ℕ) : 
  red = 7 → blue = 9 → green = 5 → total = red + blue + green →
  (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2)) = 1 / 38 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l3461_346182


namespace NUMINAMATH_CALUDE_f_min_max_l3461_346179

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_f_min_max_l3461_346179


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_value_l3461_346175

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element (k = 4) in Row 20 of Pascal's triangle -/
def pascal_triangle_row20_element5 : ℕ := binomial 20 4

theorem pascal_triangle_row20_element5_value :
  pascal_triangle_row20_element5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_value_l3461_346175


namespace NUMINAMATH_CALUDE_inequality_proof_l3461_346128

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  a * b > a * c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3461_346128


namespace NUMINAMATH_CALUDE_three_circles_inscribed_l3461_346165

theorem three_circles_inscribed (R : ℝ) (r : ℝ) : R = 9 → R = r * (1 + Real.sqrt 3) → r = (9 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_inscribed_l3461_346165


namespace NUMINAMATH_CALUDE_namjoon_books_l3461_346183

/-- The number of books Namjoon has in total -/
def total_books (a b c : ℕ) : ℕ := a + b + c

/-- Theorem stating the total number of books Namjoon has -/
theorem namjoon_books :
  ∀ (a b c : ℕ),
  a = 35 →
  b = a - 16 →
  c = b + 35 →
  total_books a b c = 108 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_books_l3461_346183


namespace NUMINAMATH_CALUDE_fencemaker_problem_l3461_346118

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side equals 56 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  width = 40 ∧ 
  length * width = 320 → 
  2 * length + width = 56 :=
by sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l3461_346118


namespace NUMINAMATH_CALUDE_circle_parabola_tangency_l3461_346157

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola with equation y = x^2 + 1 -/
def Parabola : Point → Prop :=
  fun p => p.y = p.x^2 + 1

/-- Check if a circle is tangent to the parabola at two points -/
def IsTangent (c : Circle) (p1 p2 : Point) : Prop :=
  Parabola p1 ∧ Parabola p2 ∧
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

/-- The main theorem -/
theorem circle_parabola_tangency 
  (c : Circle) (p1 p2 : Point) (h : IsTangent c p1 p2) :
  c.center.y - p1.y = p1.x^2 - 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_parabola_tangency_l3461_346157


namespace NUMINAMATH_CALUDE_unique_cube_difference_l3461_346100

theorem unique_cube_difference (m n : ℕ+) : 
  (∃ k : ℕ+, 2^n.val - 13^m.val = k^3) ↔ m = 2 ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_cube_difference_l3461_346100


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3461_346189

/-- A quadratic function f(x) = x^2 + 4x + c, where c is a constant. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- Theorem stating that for the quadratic function f(x) = x^2 + 4x + c,
    the inequality f(1) > f(0) > f(-2) holds for any constant c. -/
theorem quadratic_inequality (c : ℝ) : f c 1 > f c 0 ∧ f c 0 > f c (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3461_346189


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3461_346172

theorem fraction_sum_equality : 
  1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 9 / 20 = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3461_346172


namespace NUMINAMATH_CALUDE_line_in_quadrants_implies_ac_bc_negative_l3461_346192

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line lies in the first, second, and fourth quadrants -/
def liesInQuadrants (l : Line) : Prop :=
  ∃ (x y : ℝ), 
    (l.a * x + l.b * y + l.c = 0) ∧
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))

/-- Theorem stating the relationship between ac and bc for a line in the specified quadrants -/
theorem line_in_quadrants_implies_ac_bc_negative (l : Line) :
  liesInQuadrants l → (l.a * l.c < 0 ∧ l.b * l.c < 0) := by
  sorry

end NUMINAMATH_CALUDE_line_in_quadrants_implies_ac_bc_negative_l3461_346192


namespace NUMINAMATH_CALUDE_double_pieces_count_l3461_346112

/-- Represents the number of circles on top of a Lego piece -/
inductive PieceType
| Single
| Double
| Triple
| Quadruple

/-- The cost of a Lego piece in cents -/
def cost (p : PieceType) : ℕ :=
  match p with
  | .Single => 1
  | .Double => 2
  | .Triple => 3
  | .Quadruple => 4

/-- The total revenue in cents -/
def total_revenue : ℕ := 1000

/-- The number of single pieces sold -/
def single_count : ℕ := 100

/-- The number of triple pieces sold -/
def triple_count : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_count : ℕ := 165

theorem double_pieces_count :
  ∃ (double_count : ℕ),
    double_count * cost PieceType.Double =
      total_revenue
        - (single_count * cost PieceType.Single
          + triple_count * cost PieceType.Triple
          + quadruple_count * cost PieceType.Quadruple)
    ∧ double_count = 45 := by
  sorry


end NUMINAMATH_CALUDE_double_pieces_count_l3461_346112


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l3461_346151

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define a function that has exactly four roots
def HasFourRoots (f : ℝ → ℝ) : Prop := ∃ a b c d : ℝ, 
  (a < b ∧ b < c ∧ c < d) ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) 
  (h_even : EvenFunction f) (h_four_roots : HasFourRoots f) : 
  ∃ a b c d : ℝ, (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧ (a + b + c + d = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l3461_346151


namespace NUMINAMATH_CALUDE_password_selection_rule_probability_of_A_in_seventh_week_l3461_346115

/-- Represents the probability of password A being used in week k -/
def P (k : ℕ) : ℚ :=
  if k = 1 then 1
  else (3/4) * (-1/3)^(k-2) + 1/4

/-- The condition that the password for each week is chosen randomly from
    the three not used in the previous week -/
theorem password_selection_rule (k : ℕ) :
  k > 1 → P k = (1/3) * (1 - P (k-1)) :=
sorry

theorem probability_of_A_in_seventh_week :
  P 7 = 61/243 :=
sorry

end NUMINAMATH_CALUDE_password_selection_rule_probability_of_A_in_seventh_week_l3461_346115


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3461_346187

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- c is 7/2
  c = 7/2 ∧
  -- Area of triangle ABC is 3√3/2
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 ∧
  -- Relationship between tan A and tan B
  Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

-- Theorem statement
theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) : 
  C = Real.pi / 3 ∧ a + b = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3461_346187


namespace NUMINAMATH_CALUDE_additions_per_hour_l3461_346176

/-- Represents the number of operations a computer can perform per second -/
def operations_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the number of additions performed in an hour is 27 million -/
theorem additions_per_hour :
  (operations_per_second / 2) * seconds_per_hour = 27000000 := by
  sorry

end NUMINAMATH_CALUDE_additions_per_hour_l3461_346176


namespace NUMINAMATH_CALUDE_blue_marbles_count_l3461_346147

/-- Given a bag of marbles with a 3:5 ratio of red to blue marbles and 18 red marbles,
    prove that there are 30 blue marbles. -/
theorem blue_marbles_count (red_count : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ)
    (h_red_count : red_count = 18)
    (h_ratio : ratio_red = 3 ∧ ratio_blue = 5) :
    red_count * ratio_blue / ratio_red = 30 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l3461_346147


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l3461_346130

/-- The sum of the 7th to 10th terms of a sequence defined by S_n = 2n^2 - 3n + 1 is 116 -/
theorem sum_of_specific_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n + 1) →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  a 7 + a 8 + a 9 + a 10 = 116 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l3461_346130


namespace NUMINAMATH_CALUDE_consecutive_integers_not_sum_of_squares_l3461_346197

theorem consecutive_integers_not_sum_of_squares :
  ∃ m : ℕ+, ∀ k : ℕ, k < 2017 → ¬∃ a b : ℤ, (m + k : ℤ) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_not_sum_of_squares_l3461_346197


namespace NUMINAMATH_CALUDE_function_property_l3461_346168

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem function_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ > f x₂) →
  a < 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3461_346168


namespace NUMINAMATH_CALUDE_expression_evaluation_l3461_346142

theorem expression_evaluation : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3461_346142


namespace NUMINAMATH_CALUDE_polynomial_division_l3461_346116

-- Define the polynomials
def P (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 8 * x - 1
def D (x : ℝ) : ℝ := x - 3
def Q (x : ℝ) : ℝ := 3 * x^2 + 8
def R : ℝ := 23

-- State the theorem
theorem polynomial_division :
  ∀ x : ℝ, P x = D x * Q x + R := by sorry

end NUMINAMATH_CALUDE_polynomial_division_l3461_346116


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3461_346159

theorem solve_linear_equation (x y : ℝ) :
  2 * x - 3 * y = 4 → y = (2 * x - 4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3461_346159


namespace NUMINAMATH_CALUDE_exponent_division_l3461_346140

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3461_346140


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3461_346125

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 3 ∨ x = 1)) ∧
  (∀ y : ℝ, 4*y^2 - 3*y ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3461_346125


namespace NUMINAMATH_CALUDE_circle_equation_l3461_346144

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 5}

-- Define the line
def Line := {(x, y) : ℝ × ℝ | x - 2*y = 0}

-- Theorem statement
theorem circle_equation :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ Circle a → (x - a)^2 + y^2 = 5) ∧ 
    (∃ (x y : ℝ), (x, y) ∈ Circle a ∩ Line) ∧
    (a = 5 ∨ a = -5) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3461_346144


namespace NUMINAMATH_CALUDE_mask_production_optimization_l3461_346145

/-- Represents the production and profit parameters for a mask factory --/
structure MaskFactory where
  total_days : ℕ
  total_masks : ℕ
  min_type_a : ℕ
  daily_type_a : ℕ
  daily_type_b : ℕ
  profit_type_a : ℚ
  profit_type_b : ℚ

/-- The main theorem about mask production and profit optimization --/
theorem mask_production_optimization (f : MaskFactory) 
  (h_total_days : f.total_days = 8)
  (h_total_masks : f.total_masks = 50000)
  (h_min_type_a : f.min_type_a = 18000)
  (h_daily_type_a : f.daily_type_a = 6000)
  (h_daily_type_b : f.daily_type_b = 8000)
  (h_profit_type_a : f.profit_type_a = 1/2)
  (h_profit_type_b : f.profit_type_b = 3/10) :
  ∃ (profit_function : ℚ → ℚ) (x_range : Set ℚ) (max_profit : ℚ) (min_time : ℕ),
    (∀ x, profit_function x = 0.2 * x + 1.5) ∧
    x_range = {x | 1.8 ≤ x ∧ x ≤ 4.2} ∧
    max_profit = 2.34 ∧
    min_time = 7 :=
by sorry

#check mask_production_optimization

end NUMINAMATH_CALUDE_mask_production_optimization_l3461_346145


namespace NUMINAMATH_CALUDE_xy_product_l3461_346124

theorem xy_product (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (25:ℝ)^(x+y) / (5:ℝ)^(7*y) = 3125) : 
  x * y = 75 := by sorry

end NUMINAMATH_CALUDE_xy_product_l3461_346124


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3461_346169

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let perimeter : ℝ := 3 * side_length
  area / perimeter^2 = Real.sqrt 3 / 36 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3461_346169


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3461_346177

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 14*x_max + 45 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3461_346177


namespace NUMINAMATH_CALUDE_giorgio_cookies_l3461_346198

theorem giorgio_cookies (total_students : ℕ) (oatmeal_ratio : ℚ) (oatmeal_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : oatmeal_ratio = 1/10)
  (h3 : oatmeal_cookies = 8) :
  (oatmeal_cookies : ℚ) / (oatmeal_ratio * total_students) = 2 := by
  sorry

end NUMINAMATH_CALUDE_giorgio_cookies_l3461_346198


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l3461_346164

theorem austin_robot_purchase (num_robots : ℕ) (robot_cost tax change : ℚ) : 
  num_robots = 7 → 
  robot_cost = 8.75 → 
  tax = 7.22 → 
  change = 11.53 → 
  (num_robots : ℚ) * robot_cost + tax + change = 80 :=
by sorry

end NUMINAMATH_CALUDE_austin_robot_purchase_l3461_346164


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3461_346148

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes (-A/B) are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are identical if and only if their coefficients are proportional -/
def identical (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

theorem parallel_lines_a_value : 
  ∃! a : ℝ, parallel (a + 1) 3 3 1 (a - 1) 1 ∧ ¬identical (a + 1) 3 3 1 (a - 1) 1 ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3461_346148


namespace NUMINAMATH_CALUDE_charles_cleaning_time_l3461_346136

theorem charles_cleaning_time 
  (alice_time : ℝ) 
  (bob_time : ℝ) 
  (charles_time : ℝ) 
  (h1 : alice_time = 20) 
  (h2 : bob_time = 3/4 * alice_time) 
  (h3 : charles_time = 2/3 * bob_time) : 
  charles_time = 10 := by
sorry

end NUMINAMATH_CALUDE_charles_cleaning_time_l3461_346136


namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l3461_346102

theorem three_digit_number_puzzle :
  ∃ (x y z : ℕ),
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (100 * x + 10 * y + z) + (100 * z + 10 * y + x) = 1252 ∧
    x + y + z = 14 ∧
    x^2 + y^2 + z^2 = 84 ∧
    100 * x + 10 * y + z = 824 ∧
    100 * z + 10 * y + x = 428 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l3461_346102


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3461_346129

theorem solve_complex_equation (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a - i)^2 = 2*i) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3461_346129


namespace NUMINAMATH_CALUDE_comparison_of_fractions_l3461_346171

theorem comparison_of_fractions :
  (1/2 : ℚ) < (2/2 : ℚ) →
  (1 - 5/6 : ℚ) > (1 - 7/6 : ℚ) →
  (-π : ℝ) < -3.14 →
  (-2/3 : ℚ) > (-4/5 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_comparison_of_fractions_l3461_346171
