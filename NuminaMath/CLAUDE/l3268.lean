import Mathlib

namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l3268_326825

def is_valid_arrangement (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∈ Finset.range 10 → (n.digits 10).count d = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 18 → n % k = 0)

theorem valid_arrangement_exists : ∃ n : ℕ, is_valid_arrangement n :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l3268_326825


namespace NUMINAMATH_CALUDE_school_population_problem_l3268_326844

theorem school_population_problem :
  ∀ (initial_girls initial_boys : ℕ),
    initial_boys = initial_girls + 51 →
    (100 * initial_girls) / (initial_girls + initial_boys) = 
      (100 * (initial_girls - 41)) / ((initial_girls - 41) + (initial_boys - 19)) + 4 →
    initial_girls = 187 ∧ initial_boys = 238 :=
by
  sorry

#check school_population_problem

end NUMINAMATH_CALUDE_school_population_problem_l3268_326844


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l3268_326863

/-- Represents the annual growth rate of housing prices -/
def average_annual_growth_rate : ℝ := sorry

/-- The initial housing price in 2018 (yuan per square meter) -/
def initial_price : ℝ := 5000

/-- The final housing price in 2020 (yuan per square meter) -/
def final_price : ℝ := 6500

/-- The number of years of growth -/
def years_of_growth : ℕ := 2

/-- Theorem stating that the given equation correctly represents the housing price growth -/
theorem housing_price_growth_equation :
  initial_price * (1 + average_annual_growth_rate) ^ years_of_growth = final_price :=
sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l3268_326863


namespace NUMINAMATH_CALUDE_circle_equation_from_line_intersection_l3268_326846

/-- Given a line in polar coordinates that intersects the polar axis, 
    this theorem proves the equation of a circle centered at the intersection point. -/
theorem circle_equation_from_line_intersection (ρ θ : ℝ) :
  (ρ * Real.cos (θ + π/4) = Real.sqrt 2) →
  ∃ C : ℝ × ℝ,
    (C.1 = 2 ∧ C.2 = 0) ∧
    (∀ (ρ' θ' : ℝ), (ρ' * Real.cos θ' - C.1)^2 + (ρ' * Real.sin θ' - C.2)^2 = 1 ↔
                     ρ'^2 - 4*ρ'*Real.cos θ' + 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_from_line_intersection_l3268_326846


namespace NUMINAMATH_CALUDE_partition_spread_bound_l3268_326814

/-- The number of partitions of a natural number -/
def P (n : ℕ) : ℕ := sorry

/-- The spread of a partition -/
def spread (partition : List ℕ) : ℕ := sorry

/-- The sum of spreads of all partitions of a natural number -/
def Q (n : ℕ) : ℕ := sorry

/-- Theorem: Q(n) ≤ √(2n) · P(n) for all natural numbers n -/
theorem partition_spread_bound (n : ℕ) : Q n ≤ Real.sqrt (2 * n) * P n := by sorry

end NUMINAMATH_CALUDE_partition_spread_bound_l3268_326814


namespace NUMINAMATH_CALUDE_original_proposition_true_converse_false_l3268_326895

theorem original_proposition_true_converse_false :
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ a + b < 2) :=
by sorry

end NUMINAMATH_CALUDE_original_proposition_true_converse_false_l3268_326895


namespace NUMINAMATH_CALUDE_pirate_theorem_l3268_326894

/-- Represents a pirate in the group -/
structure Pirate where
  id : Nat
  targets : Finset Nat

/-- Represents the group of pirates -/
def PirateGroup := Finset Pirate

/-- Counts the number of pirates killed in a given order -/
def countKilled (group : PirateGroup) (order : List Pirate) : Nat :=
  sorry

/-- Main theorem: If there exists an order where 28 pirates are killed,
    then in any other order, at least 10 pirates must be killed -/
theorem pirate_theorem (group : PirateGroup) :
  (∃ order : List Pirate, countKilled group order = 28) →
  (∀ order : List Pirate, countKilled group order ≥ 10) :=
by
  sorry


end NUMINAMATH_CALUDE_pirate_theorem_l3268_326894


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l3268_326826

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of chords of a parabola -/
def midpointLocus (p : Parabola) (angle : ℝ) : Parabola :=
  sorry

/-- The ratio of distances between foci and vertices of two related parabolas -/
def focusVertexRatio (p1 p2 : Parabola) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (p : Parabola) :
  let q := midpointLocus p (π / 2)
  focusVertexRatio p q = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l3268_326826


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3268_326866

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3268_326866


namespace NUMINAMATH_CALUDE_smallest_12_digit_with_all_digits_div_36_proof_l3268_326871

def is_12_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

def smallest_12_digit_with_all_digits_div_36 : ℕ := 100023457896

theorem smallest_12_digit_with_all_digits_div_36_proof :
  (is_12_digit smallest_12_digit_with_all_digits_div_36) ∧
  (contains_all_digits smallest_12_digit_with_all_digits_div_36) ∧
  (smallest_12_digit_with_all_digits_div_36 % 36 = 0) ∧
  (∀ m : ℕ, m < smallest_12_digit_with_all_digits_div_36 →
    ¬(is_12_digit m ∧ contains_all_digits m ∧ m % 36 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_12_digit_with_all_digits_div_36_proof_l3268_326871


namespace NUMINAMATH_CALUDE_x_sum_greater_than_two_over_a_l3268_326861

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  (deriv (f a) x) / Real.exp (a * x)

theorem x_sum_greater_than_two_over_a 
  (a : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hx_dist : x₁ ≠ x₂) (hg_eq_f : g a x₁ = f a x₂) : 
  x₁ + x₂ > 2 / a := by
  sorry

end NUMINAMATH_CALUDE_x_sum_greater_than_two_over_a_l3268_326861


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l3268_326859

-- Define the reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the reflection across x = 3
def reflect_x3 (p : ℝ × ℝ) : ℝ × ℝ := (6 - p.1, p.2)

-- Define the composition of both reflections
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ := reflect_x3 (reflect_x p)

theorem parallelogram_reflection :
  double_reflect (4, 1) = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l3268_326859


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3268_326884

theorem trig_expression_equals_one :
  let expr := (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) /
              (Real.sin (26 * π / 180) * Real.cos (14 * π / 180) + 
               Real.cos (154 * π / 180) * Real.cos (94 * π / 180))
  expr = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3268_326884


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3268_326837

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3268_326837


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l3268_326806

/-- A geometric sequence with common ratio not equal to 1 -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  r ≠ 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_m_value
  (a : ℕ → ℝ) (r : ℝ) (m : ℕ)
  (h_geom : geometric_sequence a r)
  (h_eq1 : a 5 * a 6 + a 4 * a 7 = 18)
  (h_eq2 : a 1 * a m = 9) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l3268_326806


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l3268_326885

/-- M is a positive integer such that M^2 = 36^50 * 50^36 -/
def M : ℕ+ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 21 -/
theorem sum_of_digits_M : sum_of_digits M.val = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l3268_326885


namespace NUMINAMATH_CALUDE_converse_even_sum_l3268_326874

theorem converse_even_sum (a b : ℤ) : 
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) →
  (∀ a b : ℤ, Even (a + b) → (Even a ∧ Even b)) :=
sorry

end NUMINAMATH_CALUDE_converse_even_sum_l3268_326874


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3268_326831

-- Define the sets M, N, and K
def M : Set ℝ := {x : ℝ | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x : ℝ | 3 < x ∧ x < n}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = K n → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3268_326831


namespace NUMINAMATH_CALUDE_caviar_cost_calculation_l3268_326883

/-- The cost of caviar per person for Alex's New Year's Eve appetizer -/
def caviar_cost (chips_cost creme_fraiche_cost total_cost : ℚ) : ℚ :=
  total_cost - (chips_cost + creme_fraiche_cost)

/-- Theorem stating the cost of caviar per person -/
theorem caviar_cost_calculation :
  caviar_cost 3 5 27 = 19 := by
  sorry

end NUMINAMATH_CALUDE_caviar_cost_calculation_l3268_326883


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3268_326892

/-- Given two parallel lines y = (a - a^2)x - 2 and y = (3a + 1)x + 1, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, y = (a - a^2) * x - 2 ↔ y = (3*a + 1) * x + 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3268_326892


namespace NUMINAMATH_CALUDE_Q_sufficient_not_necessary_for_P_l3268_326817

-- Define the property P(x) as x^2 - 1 > 0
def P (x : ℝ) : Prop := x^2 - 1 > 0

-- Define the condition Q(x) as x < -1
def Q (x : ℝ) : Prop := x < -1

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary_for_P :
  (∀ x : ℝ, Q x → P x) ∧ ¬(∀ x : ℝ, P x → Q x) :=
sorry

end NUMINAMATH_CALUDE_Q_sufficient_not_necessary_for_P_l3268_326817


namespace NUMINAMATH_CALUDE_green_blue_tile_difference_l3268_326842

/-- Proves that the difference between green and blue tiles after adding two borders is 29 -/
theorem green_blue_tile_difference : 
  let initial_blue : ℕ := 13
  let initial_green : ℕ := 6
  let tiles_per_border : ℕ := 18
  let borders_added : ℕ := 2
  let final_green : ℕ := initial_green + borders_added * tiles_per_border
  let final_blue : ℕ := initial_blue
  final_green - final_blue = 29 := by
sorry


end NUMINAMATH_CALUDE_green_blue_tile_difference_l3268_326842


namespace NUMINAMATH_CALUDE_trig_identity_l3268_326805

theorem trig_identity : 
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3268_326805


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3268_326802

/-- Given a hyperbola with asymptotes y = ± 1/3 x and one focus at (0, 2√5),
    prove that its standard equation is y²/2 - x²/18 = 1 -/
theorem hyperbola_standard_equation 
  (asymptote : ℝ → ℝ)
  (focus : ℝ × ℝ)
  (h1 : ∀ x, asymptote x = 1/3 * x ∨ asymptote x = -1/3 * x)
  (h2 : focus = (0, 2 * Real.sqrt 5)) :
  ∃ f : ℝ × ℝ → ℝ, ∀ x y, f (x, y) = 0 ↔ y^2/2 - x^2/18 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3268_326802


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3268_326879

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := sorry
def focus2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_passes_through (p q r : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  line_passes_through focus1 A B →
  (A.1 - focus1.1)^2 + (A.2 - focus1.2)^2 + 
  (A.1 - focus2.1)^2 + (A.2 - focus2.2)^2 = 16 →
  (B.1 - focus1.1)^2 + (B.2 - focus1.2)^2 + 
  (B.1 - focus2.1)^2 + (B.2 - focus2.2)^2 = 16 →
  let perimeter := 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
    Real.sqrt ((A.1 - focus2.1)^2 + (A.2 - focus2.2)^2) +
    Real.sqrt ((B.1 - focus2.1)^2 + (B.2 - focus2.2)^2)
  perimeter = 8 := by sorry


end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3268_326879


namespace NUMINAMATH_CALUDE_distance_between_points_l3268_326873

theorem distance_between_points (b : ℝ) :
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) →
  (Real.sqrt ((3 * b - 1)^2 + (b + 1 - 4)^2) = 2 * Real.sqrt 13) →
  b * (5.47 - b) = 4.2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3268_326873


namespace NUMINAMATH_CALUDE_tenth_student_age_l3268_326878

theorem tenth_student_age (total_students : ℕ) (students_without_tenth : ℕ) 
  (avg_age_without_tenth : ℕ) (avg_age_increase : ℕ) :
  total_students = 10 →
  students_without_tenth = 9 →
  avg_age_without_tenth = 8 →
  avg_age_increase = 2 →
  (students_without_tenth * avg_age_without_tenth + 
    (total_students * (avg_age_without_tenth + avg_age_increase) - 
     students_without_tenth * avg_age_without_tenth)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_tenth_student_age_l3268_326878


namespace NUMINAMATH_CALUDE_sixth_card_is_twelve_l3268_326834

/-- A function that checks if a list of 6 integers can be divided into 3 pairs with equal sums -/
def can_be_paired (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧
  ∃ (a b c d e f : ℕ),
    numbers = [a, b, c, d, e, f] ∧
    a + b = c + d ∧ c + d = e + f

theorem sixth_card_is_twelve :
  ∀ (x : ℕ),
    x ≥ 1 ∧ x ≤ 20 →
    can_be_paired [2, 4, 9, 17, 19, x] →
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_card_is_twelve_l3268_326834


namespace NUMINAMATH_CALUDE_playground_teachers_l3268_326872

theorem playground_teachers (boys girls : ℕ) (h1 : boys = 57) (h2 : girls = 82)
  (h3 : girls = boys + teachers + 13) : teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_playground_teachers_l3268_326872


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3268_326882

/-- Given a curve C with polar equation ρ = 2cos(θ), 
    its Cartesian coordinate equation is x² + y² - 2x = 0 -/
theorem polar_to_cartesian_circle (x y : ℝ) :
  (∃ θ : ℝ, x = 2 * Real.cos θ * Real.cos θ ∧ y = 2 * Real.cos θ * Real.sin θ) ↔ 
  x^2 + y^2 - 2*x = 0 := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3268_326882


namespace NUMINAMATH_CALUDE_fibonacci_mod_127_l3268_326811

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_mod_127 :
  (∀ m : ℕ, m < 256 → (fib m % 127 ≠ 0 ∨ fib (m + 1) % 127 ≠ 1)) ∧
  fib 256 % 127 = 0 ∧ fib 257 % 127 = 1 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_mod_127_l3268_326811


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_hours_l3268_326835

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_4_hours (h : 4 > 0) :
  v 4 = 56 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_hours_l3268_326835


namespace NUMINAMATH_CALUDE_kelly_initial_games_l3268_326836

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l3268_326836


namespace NUMINAMATH_CALUDE_tan_sum_over_cos_simplification_l3268_326839

theorem tan_sum_over_cos_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / 
  Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_over_cos_simplification_l3268_326839


namespace NUMINAMATH_CALUDE_complement_of_A_l3268_326899

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3268_326899


namespace NUMINAMATH_CALUDE_fraction_equality_l3268_326886

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 3 * y) / (x + 4 * y) = 3) : 
  (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3268_326886


namespace NUMINAMATH_CALUDE_ellipse_equation_l3268_326818

/-- Given an ellipse C with semi-major axis a, semi-minor axis b, and eccentricity e,
    where a line passing through the right focus intersects C at points A and B,
    forming a triangle AF₁B with perimeter p, prove that the equation of C is x²/3 + y²/2 = 1 --/
theorem ellipse_equation (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (e : ℝ) (h_e : e = Real.sqrt 3 / 3)
  (p : ℝ) (h_p : p = 4 * Real.sqrt 3) :
  a^2 = 3 ∧ b^2 = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3268_326818


namespace NUMINAMATH_CALUDE_task_completion_time_l3268_326865

/-- Proves that the total time to complete a task is 8 days given the specified conditions -/
theorem task_completion_time 
  (john_rate : ℚ) 
  (jane_rate : ℚ) 
  (jane_leave_before_end : ℕ) :
  john_rate = 1/16 →
  jane_rate = 1/12 →
  jane_leave_before_end = 5 →
  ∃ (total_days : ℕ), total_days = 8 ∧ 
    (john_rate + jane_rate) * (total_days - jane_leave_before_end : ℚ) + 
    john_rate * (jane_leave_before_end : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l3268_326865


namespace NUMINAMATH_CALUDE_apples_ratio_l3268_326827

def apples_problem (tuesday wednesday thursday : ℕ) : Prop :=
  tuesday = 4 ∧
  thursday = tuesday / 2 ∧
  tuesday + wednesday + thursday = 14

theorem apples_ratio : 
  ∀ tuesday wednesday thursday : ℕ,
  apples_problem tuesday wednesday thursday →
  wednesday = 2 * tuesday :=
by sorry

end NUMINAMATH_CALUDE_apples_ratio_l3268_326827


namespace NUMINAMATH_CALUDE_total_players_count_l3268_326877

/-- Represents the number of people playing kabaddi -/
def kabaddi_players : ℕ := 10

/-- Represents the number of people playing kho kho only -/
def kho_kho_only_players : ℕ := 35

/-- Represents the number of people playing both games -/
def both_games_players : ℕ := 5

/-- Calculates the total number of players -/
def total_players : ℕ := kabaddi_players - both_games_players + kho_kho_only_players + both_games_players

theorem total_players_count : total_players = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l3268_326877


namespace NUMINAMATH_CALUDE_special_triangle_all_angles_60_l3268_326815

/-- A triangle with angles in arithmetic progression and sides in geometric progression -/
structure SpecialTriangle where
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Sides of the triangle opposite to angles A, B, C respectively
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles form an arithmetic progression
  angle_progression : ∃ (d : ℝ), B - A = C - B
  -- One angle is 60°
  one_angle_60 : A = 60 ∨ B = 60 ∨ C = 60
  -- Sum of angles is 180°
  angle_sum : A + B + C = 180
  -- Sides form a geometric progression
  side_progression : b^2 = a * c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- All sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Theorem: In a SpecialTriangle, all angles are 60° -/
theorem special_triangle_all_angles_60 (t : SpecialTriangle) : t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_all_angles_60_l3268_326815


namespace NUMINAMATH_CALUDE_gcf_of_75_and_90_l3268_326812

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_90_l3268_326812


namespace NUMINAMATH_CALUDE_three_solutions_condition_l3268_326808

theorem three_solutions_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    a * x₁ = |Real.log x₁| ∧ a * x₂ = |Real.log x₂| ∧ a * x₃ = |Real.log x₃|) ∧
  (∀ x : ℝ, a * x ≥ 0) ↔ 
  (-1 / Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_condition_l3268_326808


namespace NUMINAMATH_CALUDE_books_read_total_l3268_326858

/-- The number of books read by Megan, Kelcie, and Greg -/
def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

/-- Theorem stating the total number of books read by Megan, Kelcie, and Greg -/
theorem books_read_total :
  ∃ (megan_books kelcie_books greg_books : ℕ),
    megan_books = 32 ∧
    kelcie_books = megan_books / 4 ∧
    greg_books = 2 * kelcie_books + 9 ∧
    total_books megan_books kelcie_books greg_books = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_books_read_total_l3268_326858


namespace NUMINAMATH_CALUDE_gopal_krishan_ratio_l3268_326832

/-- The ratio of money between Gopal and Krishan given the conditions -/
theorem gopal_krishan_ratio :
  ∀ (ram gopal krishan : ℕ),
  ram = 735 →
  krishan = 4335 →
  7 * gopal = 17 * ram →
  (gopal : ℚ) / krishan = 1785 / 4335 :=
by sorry

end NUMINAMATH_CALUDE_gopal_krishan_ratio_l3268_326832


namespace NUMINAMATH_CALUDE_equation_equivalence_l3268_326893

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x - 4 * Real.pi) = Q) :
  10 * (6 * x - 8 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3268_326893


namespace NUMINAMATH_CALUDE_eight_additional_people_needed_l3268_326821

/-- The number of additional people needed to mow a lawn and trim its edges -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  let total_person_hours := initial_people * initial_time
  let people_mowing := total_person_hours / new_time
  let people_trimming := people_mowing / 3
  let total_people_needed := people_mowing + people_trimming
  total_people_needed - initial_people

/-- Theorem stating that 8 additional people are needed under the given conditions -/
theorem eight_additional_people_needed :
  additional_people_needed 8 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_additional_people_needed_l3268_326821


namespace NUMINAMATH_CALUDE_shirt_double_discount_l3268_326816

theorem shirt_double_discount (original_price : ℝ) (discount_rate : ℝ) : 
  original_price = 32 → 
  discount_rate = 0.25 → 
  (1 - discount_rate) * (1 - discount_rate) * original_price = 18 := by
sorry

end NUMINAMATH_CALUDE_shirt_double_discount_l3268_326816


namespace NUMINAMATH_CALUDE_square_area_after_cuts_l3268_326819

theorem square_area_after_cuts (x : ℝ) : 
  x > 0 → x - 3 > 0 → x - 5 > 0 → 
  x^2 - (x - 3) * (x - 5) = 81 → 
  x^2 = 144 := by
sorry

end NUMINAMATH_CALUDE_square_area_after_cuts_l3268_326819


namespace NUMINAMATH_CALUDE_find_A_l3268_326876

theorem find_A : ∃ A : ℕ, ∃ B : ℕ, 
  (100 ≤ 600 + 10 * A + B) ∧ 
  (600 + 10 * A + B < 1000) ∧
  (600 + 10 * A + B - 41 = 591) ∧
  A = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3268_326876


namespace NUMINAMATH_CALUDE_f_even_eq_ten_times_f_odd_l3268_326813

/-- The function f(k) counts the number of k-digit integers (including those with leading zeros)
    whose digits can be permuted to form a number divisible by 11. -/
def f (k : ℕ) : ℕ := sorry

/-- For any positive integer m, f(2m) = 10 * f(2m-1) -/
theorem f_even_eq_ten_times_f_odd (m : ℕ+) : f (2 * m) = 10 * f (2 * m - 1) := by sorry

end NUMINAMATH_CALUDE_f_even_eq_ten_times_f_odd_l3268_326813


namespace NUMINAMATH_CALUDE_no_gcd_inverting_function_l3268_326829

theorem no_gcd_inverting_function :
  ¬ (∃ f : ℕ+ → ℕ+, ∀ a b : ℕ+, Nat.gcd a.val b.val = 1 ↔ Nat.gcd (f a).val (f b).val > 1) :=
sorry

end NUMINAMATH_CALUDE_no_gcd_inverting_function_l3268_326829


namespace NUMINAMATH_CALUDE_angle_measure_l3268_326810

/-- The measure of an angle in degrees, given that its supplement is four times its complement. -/
theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 180) ∧ 
  (180 - x = 4 * (90 - x)) ∧ 
  (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3268_326810


namespace NUMINAMATH_CALUDE_solve_linear_systems_l3268_326896

theorem solve_linear_systems :
  (∃ (x1 y1 : ℝ), x1 + y1 = 3 ∧ 2*x1 + 3*y1 = 8 ∧ x1 = 1 ∧ y1 = 2) ∧
  (∃ (x2 y2 : ℝ), 5*x2 - 2*y2 = 4 ∧ 2*x2 - 3*y2 = -5 ∧ x2 = 2 ∧ y2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_linear_systems_l3268_326896


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3268_326824

theorem fraction_sum_equality (x y z : ℝ) 
  (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3268_326824


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3268_326851

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3268_326851


namespace NUMINAMATH_CALUDE_parenthesized_subtraction_equality_l3268_326888

theorem parenthesized_subtraction_equality :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_parenthesized_subtraction_equality_l3268_326888


namespace NUMINAMATH_CALUDE_milk_container_problem_l3268_326881

-- Define the capacity of container A
def A : ℝ := 1184

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 148

-- Theorem statement
theorem milk_container_problem :
  -- After transfer, B and C have equal quantities
  B + transfer = C - transfer ∧
  -- The sum of B and C equals A
  B + C = A ∧
  -- A is 1184 liters
  A = 1184 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l3268_326881


namespace NUMINAMATH_CALUDE_electric_guitar_price_l3268_326855

theorem electric_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (acoustic_price : ℕ) (electric_count : ℕ) : 
  total_guitars = 9 → 
  total_revenue = 3611 → 
  acoustic_price = 339 → 
  electric_count = 4 → 
  (total_revenue - (total_guitars - electric_count) * acoustic_price) / electric_count = 479 :=
by sorry

end NUMINAMATH_CALUDE_electric_guitar_price_l3268_326855


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l3268_326857

/-- In a right-angled triangle ABC, given the measures of its angles, prove the relationship between x and y. -/
theorem triangle_angle_relation (x y : ℝ) : 
  x > 0 → y > 0 → x + 3 * y = 90 → x + y = 90 - 2 * y := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l3268_326857


namespace NUMINAMATH_CALUDE_initial_speed_is_80_l3268_326838

/-- Represents the speed and duration of a segment of the trip -/
structure TripSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents Jeff's road trip -/
def JeffsTrip (initial_speed : ℝ) : List TripSegment :=
  [{ speed := initial_speed, duration := 6 },
   { speed := 60, duration := 4 },
   { speed := 40, duration := 2 }]

/-- Calculates the total distance of the trip -/
def totalDistance (trip : List TripSegment) : ℝ :=
  trip.map distance |>.sum

theorem initial_speed_is_80 :
  ∃ (v : ℝ), totalDistance (JeffsTrip v) = 800 ∧ v = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_80_l3268_326838


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3268_326869

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 6 * c) : 
  a * b * c = 12000 / 49 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3268_326869


namespace NUMINAMATH_CALUDE_quadratic_function_condition_l3268_326870

theorem quadratic_function_condition (m : ℝ) : (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_condition_l3268_326870


namespace NUMINAMATH_CALUDE_graph_shift_down_2_l3268_326843

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a vertical shift transformation
def vertical_shift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x => f x - c

-- Theorem stating that y = f(x) - 2 is equivalent to shifting y = f(x) down by 2 units
theorem graph_shift_down_2 :
  ∀ x : ℝ, vertical_shift f 2 x = f x - 2 :=
by
  sorry

#check graph_shift_down_2

end NUMINAMATH_CALUDE_graph_shift_down_2_l3268_326843


namespace NUMINAMATH_CALUDE_license_plate_update_l3268_326804

/-- The number of choices for the first section in the original format -/
def original_first : Nat := 5

/-- The number of choices for the second section in the original format -/
def original_second : Nat := 3

/-- The number of choices for the third section in both formats -/
def third : Nat := 5

/-- The number of additional choices for the first section in the updated format -/
def additional_first : Nat := 1

/-- The number of additional choices for the second section in the updated format -/
def additional_second : Nat := 1

/-- The largest possible number of additional license plates after updating the format -/
def additional_plates : Nat := 45

theorem license_plate_update :
  (original_first + additional_first) * (original_second + additional_second) * third -
  original_first * original_second * third = additional_plates := by
  sorry

end NUMINAMATH_CALUDE_license_plate_update_l3268_326804


namespace NUMINAMATH_CALUDE_bread_in_pond_l3268_326807

theorem bread_in_pond (total_bread : ℕ) : 
  (total_bread / 2 : ℕ) + 13 + 7 + 30 = total_bread → total_bread = 100 := by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l3268_326807


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3268_326828

/-- Represents a jump in 3D space -/
structure Jump where
  direction : Real × Real × Real
  length : Real

/-- Calculates the final position after a series of jumps -/
def finalPosition (jumps : List Jump) : Real × Real × Real :=
  sorry

/-- Calculates the distance between two points in 3D space -/
def distance (p1 p2 : Real × Real × Real) : Real :=
  sorry

/-- Calculates the probability of an event given a sample space -/
def probability (event : α → Prop) (sampleSpace : Set α) : Real :=
  sorry

theorem frog_jump_probability :
  let jumps := [
    { direction := sorry, length := 1 },
    { direction := sorry, length := 2 },
    { direction := sorry, length := 3 }
  ]
  let start := (0, 0, 0)
  let final := finalPosition jumps
  probability (λ jumps => distance start final ≤ 2) (sorry : Set (List Jump)) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3268_326828


namespace NUMINAMATH_CALUDE_area_to_paint_l3268_326823

-- Define the wall dimensions
def wall_height : ℝ := 10
def wall_width : ℝ := 15

-- Define the unpainted area dimensions
def unpainted_height : ℝ := 3
def unpainted_width : ℝ := 5

-- Theorem to prove
theorem area_to_paint : 
  wall_height * wall_width - unpainted_height * unpainted_width = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_l3268_326823


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l3268_326897

theorem circle_circumference_irrational (d : ℚ) :
  Irrational (Real.pi * (d : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l3268_326897


namespace NUMINAMATH_CALUDE_least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l3268_326898

theorem least_integer_square_36_more_than_thrice (x : ℤ) : x^2 = 3*x + 36 → x ≥ -6 := by
  sorry

theorem neg_six_satisfies_equation : (-6 : ℤ)^2 = 3*(-6) + 36 := by
  sorry

theorem least_integer_square_36_more_than_thrice_is_neg_six :
  ∃ (x : ℤ), x^2 = 3*x + 36 ∧ ∀ (y : ℤ), y^2 = 3*y + 36 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l3268_326898


namespace NUMINAMATH_CALUDE_amount_distributed_l3268_326848

/-- Proves that the amount distributed is 12000 given the conditions of the problem -/
theorem amount_distributed (A : ℕ) : 
  (A / 20 = A / 25 + 120) → A = 12000 := by
  sorry

end NUMINAMATH_CALUDE_amount_distributed_l3268_326848


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l3268_326864

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem alien_energy_conversion :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l3268_326864


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l3268_326862

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℚ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l3268_326862


namespace NUMINAMATH_CALUDE_cards_per_pack_l3268_326853

theorem cards_per_pack (num_packs : ℕ) (num_pages : ℕ) (cards_per_page : ℕ) : 
  num_packs = 60 → num_pages = 42 → cards_per_page = 10 →
  (num_pages * cards_per_page) / num_packs = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_pack_l3268_326853


namespace NUMINAMATH_CALUDE_vector_scalar_add_l3268_326854

theorem vector_scalar_add : 
  3 • !![5, -3] + !![(-4), 9] = !![11, 0] := by sorry

end NUMINAMATH_CALUDE_vector_scalar_add_l3268_326854


namespace NUMINAMATH_CALUDE_roger_apps_deletion_l3268_326841

/-- The number of apps Roger must delete for optimal phone function -/
def apps_to_delete (max_apps : ℕ) (recommended_apps : ℕ) : ℕ :=
  2 * recommended_apps - max_apps

/-- Theorem stating the number of apps Roger must delete -/
theorem roger_apps_deletion :
  apps_to_delete 50 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_roger_apps_deletion_l3268_326841


namespace NUMINAMATH_CALUDE_tangent_line_at_2_min_value_in_interval_l3268_326867

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 2 = f' 2 * (x - 2) :=
sorry

-- Theorem for the minimum value in the interval [-3, 3]
theorem min_value_in_interval : 
  ∃ x₀ ∈ Set.Icc (-3 : ℝ) 3, ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x₀ ≤ f x ∧ f x₀ = -17 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_min_value_in_interval_l3268_326867


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3268_326887

/-- A triangle with consecutive integer side lengths, where the smallest side is greater than 2 -/
structure ConsecutiveIntegerTriangle where
  n : ℕ
  gt_two : n > 2

/-- The perimeter of a ConsecutiveIntegerTriangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.n + (t.n + 1) + (t.n + 2)

/-- Predicate to check if a ConsecutiveIntegerTriangle is valid (satisfies triangle inequality) -/
def is_valid_triangle (t : ConsecutiveIntegerTriangle) : Prop :=
  t.n + (t.n + 1) > t.n + 2 ∧
  t.n + (t.n + 2) > t.n + 1 ∧
  (t.n + 1) + (t.n + 2) > t.n

theorem smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), 
    is_valid_triangle t ∧ 
    perimeter t = 12 ∧ 
    (∀ (t' : ConsecutiveIntegerTriangle), is_valid_triangle t' → perimeter t' ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3268_326887


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3268_326801

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3268_326801


namespace NUMINAMATH_CALUDE_adams_game_rounds_l3268_326856

/-- Given Adam's total score and points per round, prove the number of rounds played --/
theorem adams_game_rounds (total_points : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 283) 
  (h2 : points_per_round = 71) : 
  total_points / points_per_round = 4 := by
  sorry

end NUMINAMATH_CALUDE_adams_game_rounds_l3268_326856


namespace NUMINAMATH_CALUDE_set_A_characterization_union_A_B_characterization_l3268_326803

def A : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 2*x) / Real.log 10}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

theorem set_A_characterization : A = {x | x < 0 ∨ x > 2} := by sorry

theorem union_A_B_characterization : A ∪ B = {x | x < 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_set_A_characterization_union_A_B_characterization_l3268_326803


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3268_326860

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3268_326860


namespace NUMINAMATH_CALUDE_area_of_rectangle_in_18_gon_l3268_326880

/-- Given a regular 18-sided polygon with area 2016 square centimeters,
    the area of a rectangle formed by connecting the midpoints of four adjacent sides
    is 448 square centimeters. -/
theorem area_of_rectangle_in_18_gon (A : ℝ) (h : A = 2016) :
  let rectangle_area := A / 18 * 4
  rectangle_area = 448 :=
by sorry

end NUMINAMATH_CALUDE_area_of_rectangle_in_18_gon_l3268_326880


namespace NUMINAMATH_CALUDE_room_tile_coverage_l3268_326809

-- Define the room dimensions
def room_length : ℕ := 12
def room_width : ℕ := 20

-- Define the number of tiles
def num_tiles : ℕ := 40

-- Define the size of each tile
def tile_size : ℕ := 1

-- Theorem to prove
theorem room_tile_coverage : 
  (num_tiles : ℚ) / (room_length * room_width) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_room_tile_coverage_l3268_326809


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3268_326833

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 5*x]
  ∀ x : ℝ, (A.det = 16) ↔ (x = Real.sqrt (22/15) ∨ x = -Real.sqrt (22/15)) :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3268_326833


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l3268_326840

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 50 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l3268_326840


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3268_326875

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry is at x = 2 -/
def axis_of_symmetry (b : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) : 
  f b c (axis_of_symmetry b) < f b c 1 ∧ f b c 1 < f b c 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3268_326875


namespace NUMINAMATH_CALUDE_town_population_l3268_326890

def present_population : ℝ → Prop :=
  λ p => (1 + 0.04) * p = 1289.6

theorem town_population : ∃ p : ℝ, present_population p ∧ p = 1240 :=
  sorry

end NUMINAMATH_CALUDE_town_population_l3268_326890


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3268_326847

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_first : a 1 = 1)
  (h_sum : a 1 + a 2 + a 3 = 7) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3268_326847


namespace NUMINAMATH_CALUDE_trapezoid_height_l3268_326850

/-- A trapezoid with given side lengths has a height of 12 cm -/
theorem trapezoid_height (a b c d : ℝ) (ha : a = 25) (hb : b = 4) (hc : c = 20) (hd : d = 13) :
  ∃ h : ℝ, h = 12 ∧ h^2 = c^2 - ((a - b) / 2)^2 ∧ h^2 = d^2 - ((a - b) / 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_height_l3268_326850


namespace NUMINAMATH_CALUDE_tina_postcard_price_l3268_326868

/-- Proves that the price per postcard is $5, given the conditions of Tina's postcard sales. -/
theorem tina_postcard_price :
  let postcards_per_day : ℕ := 30
  let days_sold : ℕ := 6
  let total_earned : ℕ := 900
  let total_postcards : ℕ := postcards_per_day * days_sold
  let price_per_postcard : ℚ := total_earned / total_postcards
  price_per_postcard = 5 := by sorry

end NUMINAMATH_CALUDE_tina_postcard_price_l3268_326868


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l3268_326822

def polynomial_condition (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
    ∀ x, p x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem polynomial_value_at_zero (p : ℝ → ℝ) :
  polynomial_condition p →
  (∀ n : Nat, n ≤ 7 → p (3^n) = 1 / 3^n) →
  p 0 = 3280 / 2187 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l3268_326822


namespace NUMINAMATH_CALUDE_remainders_of_p_squared_mod_120_l3268_326852

theorem remainders_of_p_squared_mod_120 (p : Nat) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), r < 120 → (p^2 % 120 = r ↔ r = r₁ ∨ r = r₂)) := by
  sorry

end NUMINAMATH_CALUDE_remainders_of_p_squared_mod_120_l3268_326852


namespace NUMINAMATH_CALUDE_cube_of_negative_product_l3268_326845

theorem cube_of_negative_product (a b : ℝ) : (-2 * a * b) ^ 3 = -8 * a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_product_l3268_326845


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3268_326891

theorem geometric_sequence_problem (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 9/8 ∧ q = 2/3 ∧ aₙ = 1/3 ∧ aₙ = a₁ * q^(n-1) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3268_326891


namespace NUMINAMATH_CALUDE_expression_evaluation_l3268_326849

theorem expression_evaluation (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  (((a^2 - 1) / (a^2 - 2*a + 1) - 1 / (1 - a)) / (1 / (a^2 - a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3268_326849


namespace NUMINAMATH_CALUDE_different_city_probability_l3268_326889

theorem different_city_probability (pA_cityA pB_cityA : ℝ) 
  (h1 : 0 ≤ pA_cityA ∧ pA_cityA ≤ 1)
  (h2 : 0 ≤ pB_cityA ∧ pB_cityA ≤ 1)
  (h3 : pA_cityA = 0.6)
  (h4 : pB_cityA = 0.2) :
  (pA_cityA * (1 - pB_cityA)) + ((1 - pA_cityA) * pB_cityA) = 0.56 := by
sorry

end NUMINAMATH_CALUDE_different_city_probability_l3268_326889


namespace NUMINAMATH_CALUDE_function_lower_bound_l3268_326820

theorem function_lower_bound (c : ℝ) : ∀ x : ℝ, x^2 - 2*x + c ≥ c - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3268_326820


namespace NUMINAMATH_CALUDE_half_dollar_difference_l3268_326830

/-- Represents the number of coins of each type -/
structure CoinCount where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The problem constraints -/
def valid_coin_count (c : CoinCount) : Prop :=
  c.nickels + c.quarters + c.half_dollars = 60 ∧
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars = 1000

/-- The set of all valid coin counts -/
def valid_coin_counts : Set CoinCount :=
  {c | valid_coin_count c}

/-- The maximum number of half-dollars in any valid coin count -/
noncomputable def max_half_dollars : ℕ :=
  ⨆ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The minimum number of half-dollars in any valid coin count -/
noncomputable def min_half_dollars : ℕ :=
  ⨅ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The main theorem -/
theorem half_dollar_difference :
  max_half_dollars - min_half_dollars = 15 := by
  sorry

end NUMINAMATH_CALUDE_half_dollar_difference_l3268_326830


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l3268_326800

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic a b c x₁ = 0 ∧ 
  quadratic a b c x₂ = 0 ∧
  ∀ x : ℝ, quadratic a b c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l3268_326800
