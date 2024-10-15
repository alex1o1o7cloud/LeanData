import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l1479_147936

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f (x - 3) < 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f (x - 2*a) < a^2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1479_147936


namespace NUMINAMATH_CALUDE_alien_energy_cells_l1479_147912

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 321 in base 7 is equal to 162 in base 10 --/
theorem alien_energy_cells : base7ToBase10 3 2 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_cells_l1479_147912


namespace NUMINAMATH_CALUDE_diana_tue_thu_hours_l1479_147913

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Calculates the number of hours Diana works on Tuesday and Thursday --/
def hours_tue_thu (schedule : WorkSchedule) : ℕ :=
  schedule.weekly_earnings / schedule.hourly_rate - 3 * schedule.hours_mon_wed_fri

/-- Theorem stating that Diana works 30 hours on Tuesday and Thursday --/
theorem diana_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mon_wed_fri = 10)
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30) :
  hours_tue_thu schedule = 30 := by
  sorry

#eval hours_tue_thu { hours_mon_wed_fri := 10, weekly_earnings := 1800, hourly_rate := 30 }

end NUMINAMATH_CALUDE_diana_tue_thu_hours_l1479_147913


namespace NUMINAMATH_CALUDE_image_of_A_under_f_l1479_147940

def A : Set Int := {-1, 3, 5}

def f (x : Int) : Int := 2 * x - 1

theorem image_of_A_under_f :
  (Set.image f A) = {-3, 5, 9} := by
  sorry

end NUMINAMATH_CALUDE_image_of_A_under_f_l1479_147940


namespace NUMINAMATH_CALUDE_hall_length_l1479_147927

/-- The length of a rectangular hall given its width, height, and total area to be covered. -/
theorem hall_length (width height total_area : ℝ) (hw : width = 15) (hh : height = 5) 
  (ha : total_area = 950) : 
  ∃ length : ℝ, length = 32 ∧ total_area = length * width + 2 * (height * length + height * width) :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l1479_147927


namespace NUMINAMATH_CALUDE_max_a_correct_l1479_147990

/-- The inequality x^2 - 4x - a - 1 ≥ 0 has solutions for x ∈ [1, 4] -/
def has_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 4 ∧ x^2 - 4*x - a - 1 ≥ 0

/-- The maximum value of a for which the inequality has solutions -/
def max_a : ℝ := -1

theorem max_a_correct :
  ∀ a : ℝ, has_solutions a ↔ a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_a_correct_l1479_147990


namespace NUMINAMATH_CALUDE_emily_sixth_score_l1479_147993

def emily_scores : List ℕ := [94, 90, 85, 90, 105]

def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem emily_sixth_score :
  ∃ (sixth_score : ℕ),
    sixth_score > emily_scores.minimum ∧
    arithmetic_mean (emily_scores ++ [sixth_score]) = 95 ∧
    sixth_score = 106 := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l1479_147993


namespace NUMINAMATH_CALUDE_parabola_equation_from_axis_and_focus_l1479_147947

/-- A parabola with given axis of symmetry and focus -/
structure Parabola where
  axis_of_symmetry : ℝ
  focus : ℝ × ℝ

/-- The equation of a parabola given its parameters -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * x

/-- Theorem: For a parabola with axis of symmetry x = 1 and focus at (-1, 0), its equation is y² = -4x -/
theorem parabola_equation_from_axis_and_focus :
  ∀ (p : Parabola), p.axis_of_symmetry = 1 ∧ p.focus = (-1, 0) →
  parabola_equation p = fun x y => y^2 = -4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_axis_and_focus_l1479_147947


namespace NUMINAMATH_CALUDE_fencing_length_l1479_147906

/-- Calculates the required fencing length for a rectangular field -/
theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 40 →
  2 * (area / uncovered_side) + uncovered_side = 74 := by
  sorry


end NUMINAMATH_CALUDE_fencing_length_l1479_147906


namespace NUMINAMATH_CALUDE_equation_solution_l1479_147970

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1479_147970


namespace NUMINAMATH_CALUDE_b_eq_one_sufficient_not_necessary_l1479_147904

/-- The condition for the line and curve to have common points -/
def has_common_points (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 + (y - 1)^2 = 1

/-- The statement that b = 1 is sufficient but not necessary for common points -/
theorem b_eq_one_sufficient_not_necessary :
  (∀ k : ℝ, has_common_points k 1) ∧
  (∃ k b : ℝ, b ≠ 1 ∧ has_common_points k b) :=
sorry

end NUMINAMATH_CALUDE_b_eq_one_sufficient_not_necessary_l1479_147904


namespace NUMINAMATH_CALUDE_monomial_properties_l1479_147953

/-- Represents a monomial -3a²bc/5 -/
structure Monomial where
  coefficient : ℚ
  a_exponent : ℕ
  b_exponent : ℕ
  c_exponent : ℕ

/-- The specific monomial -3a²bc/5 -/
def our_monomial : Monomial :=
  { coefficient := -3/5
    a_exponent := 2
    b_exponent := 1
    c_exponent := 1 }

/-- The coefficient of a monomial is its numerical factor -/
def get_coefficient (m : Monomial) : ℚ := m.coefficient

/-- The degree of a monomial is the sum of its variable exponents -/
def get_degree (m : Monomial) : ℕ := m.a_exponent + m.b_exponent + m.c_exponent

theorem monomial_properties :
  (get_coefficient our_monomial = -3/5) ∧ (get_degree our_monomial = 4) := by
  sorry


end NUMINAMATH_CALUDE_monomial_properties_l1479_147953


namespace NUMINAMATH_CALUDE_properties_of_one_minus_sqrt_two_l1479_147923

theorem properties_of_one_minus_sqrt_two :
  let x : ℝ := 1 - Real.sqrt 2
  (- x = Real.sqrt 2 - 1) ∧
  (|x| = Real.sqrt 2 - 1) ∧
  (x⁻¹ = -1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_properties_of_one_minus_sqrt_two_l1479_147923


namespace NUMINAMATH_CALUDE_order_of_abc_l1479_147907

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a, b, and c as real numbers
variable (a b c : ℝ)

-- State the theorem
theorem order_of_abc (hf : Monotone f) (ha : a = f 2 ∧ a < 0) 
  (hb : f b = 2) (hc : f c = 0) : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1479_147907


namespace NUMINAMATH_CALUDE_mascot_sales_equation_l1479_147905

/-- Represents the sales growth of a mascot over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating the correct equation for the given sales scenario -/
theorem mascot_sales_equation :
  ∀ (x : ℝ), x > 0 →
  sales_growth 10 11.5 x :=
by
  sorry

end NUMINAMATH_CALUDE_mascot_sales_equation_l1479_147905


namespace NUMINAMATH_CALUDE_number_problem_l1479_147995

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N - (1/2 : ℝ) * (1/6 : ℝ) * N = 35 →
  (40/100 : ℝ) * N = -280 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l1479_147995


namespace NUMINAMATH_CALUDE_banana_cost_l1479_147967

/-- Given that 4 bananas cost $20, prove that one banana costs $5. -/
theorem banana_cost : 
  ∀ (cost : ℝ), (4 * cost = 20) → (cost = 5) := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_l1479_147967


namespace NUMINAMATH_CALUDE_revolver_game_probability_l1479_147949

/-- Represents a six-shot revolver with one bullet -/
def Revolver : Type := Unit

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- The probability of firing the bullet on a single shot -/
def singleShotProbability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def singleShotMissProbability : ℚ := 1 - singleShotProbability

/-- The starting player of the game -/
def startingPlayer : Player := Player.A

/-- The probability that the gun will fire while player A is holding it -/
noncomputable def probabilityAFires : ℚ := 6 / 11

/-- Theorem stating that the probability of A firing the gun is 6/11 -/
theorem revolver_game_probability :
  probabilityAFires = 6 / 11 :=
sorry

end NUMINAMATH_CALUDE_revolver_game_probability_l1479_147949


namespace NUMINAMATH_CALUDE_f_positive_solution_a_range_l1479_147996

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3|

-- Theorem for the solution of f(x) > 0
theorem f_positive_solution :
  ∀ x : ℝ, f x > 0 ↔ x < -4 ∨ x > 2/3 := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x : ℝ, a - 3*|x - 3| < f x) ↔ a < 7 := by sorry

end NUMINAMATH_CALUDE_f_positive_solution_a_range_l1479_147996


namespace NUMINAMATH_CALUDE_equation_solution_l1479_147901

theorem equation_solution (x : ℝ) : 
  (3 * x + 25 ≠ 0) → 
  ((8 * x^2 + 75 * x - 3) / (3 * x + 25) = 2 * x + 5 ↔ x = -16 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1479_147901


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l1479_147920

theorem square_difference_formula_inapplicable (a b : ℝ) :
  ¬∃ (x y : ℝ), (a - b) * (b - a) = x^2 - y^2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l1479_147920


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1479_147908

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + 2*n) :=
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1479_147908


namespace NUMINAMATH_CALUDE_twenty_triangles_l1479_147910

/-- Represents a rectangle divided into smaller rectangles with diagonal and vertical lines -/
structure DividedRectangle where
  smallRectangles : Nat
  diagonalsPerSmallRectangle : Nat
  verticalLinesPerSmallRectangle : Nat

/-- Counts the total number of triangles in the divided rectangle -/
def countTriangles (r : DividedRectangle) : Nat :=
  sorry

/-- Theorem stating that the specific configuration results in 20 triangles -/
theorem twenty_triangles :
  let r : DividedRectangle := {
    smallRectangles := 4,
    diagonalsPerSmallRectangle := 1,
    verticalLinesPerSmallRectangle := 1
  }
  countTriangles r = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_triangles_l1479_147910


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l1479_147932

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ m * x^2 + 2 * x + 1 = 0) ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l1479_147932


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l1479_147945

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem base4_multiplication_division :
  base10ToBase4 ((base4ToBase10 131 * base4ToBase10 21) / base4ToBase10 3) = 1113 := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l1479_147945


namespace NUMINAMATH_CALUDE_files_deleted_amy_deleted_files_l1479_147946

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  (initial_music + initial_video) - remaining

theorem amy_deleted_files : files_deleted 4 21 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_amy_deleted_files_l1479_147946


namespace NUMINAMATH_CALUDE_ice_cream_jog_speed_l1479_147971

/-- Calculates the required speed in miles per hour to cover a given distance within a time limit -/
def required_speed (time_limit : ℚ) (distance_blocks : ℕ) (block_length : ℚ) : ℚ :=
  (distance_blocks : ℚ) * block_length * (60 / time_limit)

theorem ice_cream_jog_speed :
  let time_limit : ℚ := 10  -- Time limit in minutes
  let distance_blocks : ℕ := 16  -- Distance in blocks
  let block_length : ℚ := 1/8  -- Length of each block in miles
  required_speed time_limit distance_blocks block_length = 12 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_jog_speed_l1479_147971


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l1479_147959

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

theorem tangent_slope_at_pi_over_four :
  (deriv f) (π/4) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l1479_147959


namespace NUMINAMATH_CALUDE_factor_x4_minus_16_l1479_147911

theorem factor_x4_minus_16 (x : ℂ) : x^4 - 16 = (x - 2) * (x + 2) * (x - 2*I) * (x + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_16_l1479_147911


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1479_147902

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 4 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1479_147902


namespace NUMINAMATH_CALUDE_inequality_proof_l1479_147950

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1479_147950


namespace NUMINAMATH_CALUDE_dodecahedron_diagonals_l1479_147955

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  vertex_face_incidence : vertices → Finset faces
  diag : Fin 20 → Fin 20 → Prop

/-- Properties of a dodecahedron -/
axiom dodecahedron_properties (D : Dodecahedron) :
  (D.vertices.card = 20) ∧
  (D.faces.card = 12) ∧
  (∀ v : D.vertices, (D.vertex_face_incidence v).card = 3) ∧
  (∀ v w : D.vertices, D.diag v w ↔ v ≠ w ∧ (D.vertex_face_incidence v ∩ D.vertex_face_incidence w).card = 0)

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (D : Dodecahedron) : ℕ :=
  (D.vertices.card * (D.vertices.card - 4)) / 2

/-- Theorem: A dodecahedron has 160 interior diagonals -/
theorem dodecahedron_diagonals (D : Dodecahedron) : interior_diagonals D = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_diagonals_l1479_147955


namespace NUMINAMATH_CALUDE_characterization_of_p_l1479_147980

/-- The polynomial equation in x with parameter p -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + x^2 + 3*p*x + 1

/-- A function has at least two distinct positive real roots -/
def has_two_distinct_positive_roots (g : ℝ → ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ g x = 0 ∧ g y = 0

/-- The main theorem: characterization of p for which f has at least two distinct positive real roots -/
theorem characterization_of_p (p : ℝ) : 
  has_two_distinct_positive_roots (f p) ↔ p < 1/4 := by sorry

end NUMINAMATH_CALUDE_characterization_of_p_l1479_147980


namespace NUMINAMATH_CALUDE_permutation_problem_arrangement_problem_photo_arrangement_problem_l1479_147964

-- Problem 1
theorem permutation_problem (m : ℕ) : 
  (Nat.factorial 10) / (Nat.factorial (10 - m)) = (Nat.factorial 10) / (Nat.factorial 4) → m = 6 := by
sorry

-- Problem 2
theorem arrangement_problem : 
  (Nat.factorial 3) = 6 := by
sorry

-- Problem 3
theorem photo_arrangement_problem : 
  2 * 4 * (Nat.factorial 4) = 192 := by
sorry

end NUMINAMATH_CALUDE_permutation_problem_arrangement_problem_photo_arrangement_problem_l1479_147964


namespace NUMINAMATH_CALUDE_cos_sum_17th_roots_l1479_147943

theorem cos_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_17th_roots_l1479_147943


namespace NUMINAMATH_CALUDE_custom_mul_theorem_l1479_147974

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^3

/-- Theorem stating that if a * 3 = 15 under the custom multiplication, then a = 21 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 3 = 15) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_theorem_l1479_147974


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_11_l1479_147956

theorem smallest_four_digit_mod_11 :
  ∀ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ n % 11 = 2 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_11_l1479_147956


namespace NUMINAMATH_CALUDE_amoeba_population_day_10_l1479_147937

/-- The number of amoebas on day n, given an initial population of 3 and daily doubling. -/
def amoeba_population (n : ℕ) : ℕ := 3 * 2^n

/-- Theorem stating that after 10 days, the amoeba population is 3072. -/
theorem amoeba_population_day_10 : amoeba_population 10 = 3072 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_day_10_l1479_147937


namespace NUMINAMATH_CALUDE_no_consecutive_power_l1479_147948

theorem no_consecutive_power (n : ℕ) : ¬ ∃ (m k : ℕ), k ≥ 2 ∧ n * (n + 1) = m ^ k := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_power_l1479_147948


namespace NUMINAMATH_CALUDE_solve_seed_problem_l1479_147938

def seed_problem (total_seeds : ℕ) (left_seeds : ℕ) (right_multiplier : ℕ) (seeds_left : ℕ) : Prop :=
  let right_seeds := right_multiplier * left_seeds
  let initially_thrown := left_seeds + right_seeds
  let joined_later := total_seeds - initially_thrown - seeds_left
  joined_later = total_seeds - (left_seeds + right_multiplier * left_seeds) - seeds_left

theorem solve_seed_problem :
  seed_problem 120 20 2 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_seed_problem_l1479_147938


namespace NUMINAMATH_CALUDE_quadratic_root_square_l1479_147984

theorem quadratic_root_square (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_l1479_147984


namespace NUMINAMATH_CALUDE_german_team_goals_l1479_147968

def journalist1_correct (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_correct (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_correct (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_correct x ∧ journalist2_correct x ∧ ¬journalist3_correct x) ∨
  (journalist1_correct x ∧ ¬journalist2_correct x ∧ journalist3_correct x) ∨
  (¬journalist1_correct x ∧ journalist2_correct x ∧ journalist3_correct x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l1479_147968


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1479_147999

theorem greatest_whole_number_satisfying_inequality :
  ∀ (n : ℤ), n ≤ 0 ↔ (3 : ℝ) * n + 2 < 5 - 2 * n :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1479_147999


namespace NUMINAMATH_CALUDE_expression_value_l1479_147973

theorem expression_value (x : ℝ) (h : Real.tan (Real.pi - x) = -2) :
  4 * Real.sin x ^ 2 - 3 * Real.sin x * Real.cos x - 5 * Real.cos x ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1479_147973


namespace NUMINAMATH_CALUDE_remainder_equality_l1479_147922

theorem remainder_equality (a b d s t : ℕ) 
  (h1 : a > b) 
  (h2 : a % d = s % d) 
  (h3 : b % d = t % d) : 
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l1479_147922


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_linear_l1479_147934

theorem integral_sqrt_plus_linear (f g : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..1, (Real.sqrt (1 - x^2) + 3*x)) = π/4 + 3/2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_linear_l1479_147934


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sarahs_scores_l1479_147991

def sarahs_scores : List ℝ := [87, 90, 86, 93, 89, 92]

theorem arithmetic_mean_of_sarahs_scores :
  (sarahs_scores.sum / sarahs_scores.length : ℝ) = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sarahs_scores_l1479_147991


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1479_147957

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (1 : ℝ) * x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1479_147957


namespace NUMINAMATH_CALUDE_minimal_distance_point_l1479_147919

/-- The point that minimizes the sum of distances to two fixed points on a given line -/
theorem minimal_distance_point 
  (A B P : ℝ × ℝ) 
  (h_A : A = (-3, 1)) 
  (h_B : B = (5, -1)) 
  (h_P : P.2 = -2) : 
  (P = (3, -2)) ↔ 
  (∀ Q : ℝ × ℝ, Q.2 = -2 → 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 
    Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) + Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_minimal_distance_point_l1479_147919


namespace NUMINAMATH_CALUDE_total_books_count_l1479_147960

-- Define the number of books per shelf
def booksPerShelf : ℕ := 6

-- Define the number of shelves for each category
def mysteryShelvesCount : ℕ := 8
def pictureShelvesCount : ℕ := 5
def sciFiShelvesCount : ℕ := 4
def nonFictionShelvesCount : ℕ := 3

-- Define the total number of books
def totalBooks : ℕ := 
  booksPerShelf * (mysteryShelvesCount + pictureShelvesCount + sciFiShelvesCount + nonFictionShelvesCount)

-- Theorem statement
theorem total_books_count : totalBooks = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1479_147960


namespace NUMINAMATH_CALUDE_adams_stairs_l1479_147987

theorem adams_stairs (total_steps : ℕ) (steps_left : ℕ) (steps_climbed : ℕ) : 
  total_steps = 96 → steps_left = 22 → steps_climbed = total_steps - steps_left → steps_climbed = 74 := by
  sorry

end NUMINAMATH_CALUDE_adams_stairs_l1479_147987


namespace NUMINAMATH_CALUDE_min_value_theorem_l1479_147976

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c

/-- The theorem statement -/
theorem min_value_theorem (funcs : ParallelLinearFunctions) 
  (h : ∃ (x : ℝ), ∀ (y : ℝ), (funcs.f y)^2 + 8 * funcs.g y ≥ (funcs.f x)^2 + 8 * funcs.g x)
  (min_value : (funcs.f x)^2 + 8 * funcs.g x = -29) :
  ∃ (z : ℝ), ∀ (w : ℝ), (funcs.g w)^2 + 8 * funcs.f w ≥ (funcs.g z)^2 + 8 * funcs.f z ∧ 
  (funcs.g z)^2 + 8 * funcs.f z = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1479_147976


namespace NUMINAMATH_CALUDE_basketball_game_score_l1479_147900

theorem basketball_game_score (a b k d : ℕ) : 
  a = b →  -- Tied at the end of first quarter
  (4*a + 14*k = 4*b + 6*d + 2) →  -- Eagles won by two points
  (4*a + 14*k ≤ 100) →  -- Eagles scored no more than 100
  (4*b + 6*d ≤ 100) →  -- Panthers scored no more than 100
  (2*a + k) + (2*b + d) = 59 := by
sorry

end NUMINAMATH_CALUDE_basketball_game_score_l1479_147900


namespace NUMINAMATH_CALUDE_centroid_altitude_length_l1479_147931

/-- Triangle XYZ with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Foot of the altitude from a point to a line segment -/
def altitude_foot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem centroid_altitude_length (t : Triangle) (h1 : t.a = 13) (h2 : t.b = 15) (h3 : t.c = 24) :
  let g := centroid t
  let yz := ((0, 0), (t.c, 0))  -- Assuming YZ is on the x-axis
  let q := altitude_foot g yz
  distance g q = 2.4 := by sorry

end NUMINAMATH_CALUDE_centroid_altitude_length_l1479_147931


namespace NUMINAMATH_CALUDE_elephant_exodus_rate_calculation_l1479_147994

/-- The rate of elephants leaving Utopia National Park during an exodus --/
def elephant_exodus_rate (initial_elephants : ℕ) (exodus_duration : ℕ) 
  (new_elephants_duration : ℕ) (new_elephants_rate : ℕ) (final_elephants : ℕ) : ℕ :=
  (initial_elephants - final_elephants + new_elephants_duration * new_elephants_rate) / exodus_duration

/-- Theorem stating the rate of elephants leaving during the exodus --/
theorem elephant_exodus_rate_calculation :
  elephant_exodus_rate 30000 4 7 1500 28980 = 2880 :=
by sorry

end NUMINAMATH_CALUDE_elephant_exodus_rate_calculation_l1479_147994


namespace NUMINAMATH_CALUDE_randy_third_quiz_score_l1479_147909

theorem randy_third_quiz_score 
  (first_quiz : ℕ) 
  (second_quiz : ℕ) 
  (fifth_quiz : ℕ) 
  (desired_average : ℕ) 
  (total_quizzes : ℕ) 
  (third_fourth_sum : ℕ) :
  first_quiz = 90 →
  second_quiz = 98 →
  fifth_quiz = 96 →
  desired_average = 94 →
  total_quizzes = 5 →
  third_fourth_sum = 186 →
  ∃ (fourth_quiz : ℕ), 
    (first_quiz + second_quiz + 94 + fourth_quiz + fifth_quiz) / total_quizzes = desired_average :=
by
  sorry


end NUMINAMATH_CALUDE_randy_third_quiz_score_l1479_147909


namespace NUMINAMATH_CALUDE_total_balls_in_box_l1479_147926

/-- Given a box with blue and red balls, calculate the total number of balls -/
theorem total_balls_in_box (blue_balls : ℕ) (red_balls : ℕ) : 
  blue_balls = 3 → red_balls = 2 → blue_balls + red_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l1479_147926


namespace NUMINAMATH_CALUDE_age_problem_l1479_147975

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 42 → 
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1479_147975


namespace NUMINAMATH_CALUDE_subSubfaces_12_9_l1479_147988

/-- The number of k-dimensional sub-subfaces in an n-dimensional cube -/
def subSubfaces (n k : ℕ) : ℕ := 2^(n - k) * (Nat.choose n k)

/-- Theorem: The number of 9-dimensional sub-subfaces in a 12-dimensional cube is 1760 -/
theorem subSubfaces_12_9 : subSubfaces 12 9 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_subSubfaces_12_9_l1479_147988


namespace NUMINAMATH_CALUDE_height_difference_l1479_147928

/-- Prove that the difference between 3 times Kim's height and Tamara's height is 4 inches -/
theorem height_difference (kim_height tamara_height : ℕ) : 
  tamara_height + kim_height = 92 →
  tamara_height = 68 →
  ∃ x, tamara_height = 3 * kim_height - x →
  3 * kim_height - tamara_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1479_147928


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1479_147958

theorem polynomial_evaluation : 
  let p : ℝ → ℝ := λ x => 2*x^4 + 3*x^3 - x^2 + 5*x - 2
  p 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1479_147958


namespace NUMINAMATH_CALUDE_shape_is_regular_tetrahedron_l1479_147917

/-- A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle -/
structure Shape3D where
  diagonalAngle : ℝ
  diagonalAngleIs60 : diagonalAngle = 60

/-- Definition of a regular tetrahedron -/
def RegularTetrahedron : Type := Unit

/-- Theorem: A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle is a regular tetrahedron -/
theorem shape_is_regular_tetrahedron (s : Shape3D) : RegularTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_shape_is_regular_tetrahedron_l1479_147917


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1479_147942

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 450) :
  (new_price - old_price) / old_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1479_147942


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1479_147962

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 7 / 13 → Nat.gcd a b = 15 → Nat.lcm a b = 91 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1479_147962


namespace NUMINAMATH_CALUDE_is_factorization_l1479_147952

/-- Proves that x^2 - 4x + 4 = (x - 2)^2 is a factorization --/
theorem is_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_is_factorization_l1479_147952


namespace NUMINAMATH_CALUDE_smallest_multiple_l1479_147998

theorem smallest_multiple (x : ℕ) : x = 16 ↔ (
  x > 0 ∧
  450 * x % 800 = 0 ∧
  ∀ y : ℕ, y > 0 → y < x → 450 * y % 800 ≠ 0
) := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1479_147998


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1479_147925

/-- A cubic polynomial with integer coefficients -/
def cubic_polynomial (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 9*a

/-- Predicate for a cubic polynomial having two coincident roots -/
def has_coincident_roots (a b : ℤ) : Prop :=
  ∃ r s : ℤ, r ≠ s ∧ 
    ∀ x : ℝ, cubic_polynomial a b x = (x - r)^2 * (x - s)

/-- Theorem stating that under given conditions, |ab| = 1344 -/
theorem cubic_polynomial_property (a b : ℤ) :
  a ≠ 0 → b ≠ 0 → has_coincident_roots a b → |a*b| = 1344 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1479_147925


namespace NUMINAMATH_CALUDE_unique_base_for_256_four_digits_l1479_147954

/-- A number n has exactly d digits in base b if and only if b^(d-1) ≤ n < b^d -/
def has_exactly_d_digits (n : ℕ) (b : ℕ) (d : ℕ) : Prop :=
  b^(d-1) ≤ n ∧ n < b^d

/-- The theorem statement -/
theorem unique_base_for_256_four_digits :
  ∃! b : ℕ, b ≥ 2 ∧ has_exactly_d_digits 256 b 4 :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_256_four_digits_l1479_147954


namespace NUMINAMATH_CALUDE_two_negative_roots_l1479_147981

/-- The polynomial function we're analyzing -/
def f (q : ℝ) (x : ℝ) : ℝ := x^4 + 2*q*x^3 - 3*x^2 + 2*q*x + 1

/-- Theorem stating that for any q < 1/4, the equation f q x = 0 has at least two distinct negative real roots -/
theorem two_negative_roots (q : ℝ) (h : q < 1/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ f q x₁ = 0 ∧ f q x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_negative_roots_l1479_147981


namespace NUMINAMATH_CALUDE_profit_share_ratio_l1479_147939

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) :
  p_investment = 30000 →
  q_investment = 45000 →
  (p_investment : ℚ) / (p_investment + q_investment) = 2 / 5 ∧
  (q_investment : ℚ) / (p_investment + q_investment) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l1479_147939


namespace NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l1479_147979

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (m + k) % 10

/-- The population size -/
def populationSize : Nat := 100

/-- The number of groups -/
def numGroups : Nat := 10

/-- The size of each group -/
def groupSize : Nat := populationSize / numGroups

/-- The starting number of the k-th group -/
def groupStart (k : Nat) : Nat :=
  (k - 1) * groupSize

theorem systematic_sampling_seventh_group :
  ∀ m : Nat,
    m = 6 →
    ∃ n : Nat,
      n = 63 ∧
      n ≥ groupStart 7 ∧
      n < groupStart 7 + groupSize ∧
      n % 10 = systematicSample m 7 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l1479_147979


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1479_147983

universe u

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1479_147983


namespace NUMINAMATH_CALUDE_february_average_rainfall_l1479_147986

-- Define the given conditions
def total_rainfall : ℝ := 280
def days_in_february : ℕ := 28
def hours_per_day : ℕ := 24

-- Define the theorem
theorem february_average_rainfall :
  total_rainfall / (days_in_february * hours_per_day : ℝ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_february_average_rainfall_l1479_147986


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1479_147982

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 1| - |x - 3| > a) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1479_147982


namespace NUMINAMATH_CALUDE_carpet_cost_specific_carpet_cost_l1479_147972

/-- The total cost of carpet squares needed to cover a rectangular floor and an irregular section -/
theorem carpet_cost (rectangular_length : ℝ) (rectangular_width : ℝ) (irregular_area : ℝ)
  (carpet_side : ℝ) (carpet_cost : ℝ) : ℝ :=
  let rectangular_area := rectangular_length * rectangular_width
  let carpet_area := carpet_side * carpet_side
  let rectangular_squares := rectangular_area / carpet_area
  let irregular_squares := irregular_area / carpet_area
  let total_squares := rectangular_squares + irregular_squares + 1 -- Adding 1 for potential waste
  total_squares * carpet_cost

/-- The specific problem statement -/
theorem specific_carpet_cost : carpet_cost 24 64 128 8 24 = 648 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_specific_carpet_cost_l1479_147972


namespace NUMINAMATH_CALUDE_patio_length_l1479_147935

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 →
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 100 →
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_patio_length_l1479_147935


namespace NUMINAMATH_CALUDE_a_positive_iff_sum_geq_two_l1479_147963

theorem a_positive_iff_sum_geq_two (a : ℝ) : a > 0 ↔ a + 1/a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_positive_iff_sum_geq_two_l1479_147963


namespace NUMINAMATH_CALUDE_unique_matrix_transformation_l1479_147965

theorem unique_matrix_transformation (A : Matrix (Fin 2) (Fin 2) ℝ) :
  ∃! M : Matrix (Fin 2) (Fin 2) ℝ,
    (∀ i j, (M * A) i j = if j = 1 then (if i = 0 then 2 * A i j else 3 * A i j) else A i j) ∧
    M = ![![1, 0], ![0, 3]] := by
  sorry

end NUMINAMATH_CALUDE_unique_matrix_transformation_l1479_147965


namespace NUMINAMATH_CALUDE_digit_interchange_theorem_l1479_147914

theorem digit_interchange_theorem (a b m : ℕ) (h1 : a > 0 ∧ a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = m * (a * b)) :
  10 * b + a = (11 - m) * (a * b) :=
sorry

end NUMINAMATH_CALUDE_digit_interchange_theorem_l1479_147914


namespace NUMINAMATH_CALUDE_x_value_l1479_147985

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem x_value : ∃ x : ℝ, oplus x (oplus 2 3) = 1 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1479_147985


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_sales_equation_l1479_147989

/-- The sales equation for Bing Dwen Dwen mascot -/
theorem bing_dwen_dwen_sales_equation (x : ℝ) : 
  (5000 : ℝ) * (1 + x) + (5000 : ℝ) * (1 + x)^2 = 22500 ↔ 
  (∃ (sales_feb4 sales_feb5 sales_feb6 : ℝ),
    sales_feb4 = 5000 ∧
    sales_feb5 = sales_feb4 * (1 + x) ∧
    sales_feb6 = sales_feb5 * (1 + x) ∧
    sales_feb5 + sales_feb6 = 22500) :=
by
  sorry

end NUMINAMATH_CALUDE_bing_dwen_dwen_sales_equation_l1479_147989


namespace NUMINAMATH_CALUDE_good_carrots_count_l1479_147944

theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : bad_carrots = 7) :
  carol_carrots + mom_carrots - bad_carrots = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l1479_147944


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_four_to_sixth_l1479_147924

theorem sqrt_five_squared_times_four_to_sixth (x : ℝ) : x = Real.sqrt (5^2 * 4^6) → x = 320 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_four_to_sixth_l1479_147924


namespace NUMINAMATH_CALUDE_problem_solution_l1479_147978

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 16) (h2 : x = 16) : y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1479_147978


namespace NUMINAMATH_CALUDE_trig_values_for_special_angle_l1479_147941

/-- The intersection point of two lines -/
def intersection_point (l₁ l₂ : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The angle whose terminal side passes through a given point -/
def angle_from_point (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The sine of an angle -/
def sine (α : ℝ) : ℝ :=
  sorry

/-- The cosine of an angle -/
def cosine (α : ℝ) : ℝ :=
  sorry

/-- The tangent of an angle -/
def tangent (α : ℝ) : ℝ :=
  sorry

theorem trig_values_for_special_angle :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ x - y = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ 2*x + y - 3 = 0
  let p := intersection_point l₁ l₂
  let α := angle_from_point p
  sine α = Real.sqrt 2 / 2 ∧ cosine α = Real.sqrt 2 / 2 ∧ tangent α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_values_for_special_angle_l1479_147941


namespace NUMINAMATH_CALUDE_bread_slice_cost_l1479_147966

-- Define the problem parameters
def num_loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def payment_amount : ℕ := 40  -- in dollars
def change_received : ℕ := 16  -- in dollars

-- Define the theorem
theorem bread_slice_cost :
  let total_cost : ℕ := payment_amount - change_received
  let total_slices : ℕ := num_loaves * slices_per_loaf
  let cost_per_slice_cents : ℕ := (total_cost * 100) / total_slices
  cost_per_slice_cents = 40 := by
  sorry

end NUMINAMATH_CALUDE_bread_slice_cost_l1479_147966


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1479_147918

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs y = abs x) ∧  -- Distance from y-axis equals distance from x-axis
    (abs y = abs ((x + y - 4) / Real.sqrt 2)) ∧  -- Distance from x-axis equals distance from line x + y = 4
    (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1479_147918


namespace NUMINAMATH_CALUDE_turtleneck_profit_percentage_l1479_147977

/-- Calculates the profit percentage on turtleneck sweaters sold in February 
    given specific markup and discount conditions. -/
theorem turtleneck_profit_percentage :
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let first_price := 1 + initial_markup
  let second_price := first_price + new_year_markup * first_price
  let final_price := second_price * (1 - february_discount)
  let profit_percentage := final_price - 1
  profit_percentage = 0.365 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_percentage_l1479_147977


namespace NUMINAMATH_CALUDE_total_bulbs_needed_l1479_147969

def ceiling_lights (medium_count : ℕ) : ℕ × ℕ × ℕ := 
  let large_count := 2 * medium_count
  let small_count := medium_count + 10
  (small_count, medium_count, large_count)

def bulb_count (lights : ℕ × ℕ × ℕ) : ℕ :=
  let (small, medium, large) := lights
  small * 1 + medium * 2 + large * 3

theorem total_bulbs_needed : 
  bulb_count (ceiling_lights 12) = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_bulbs_needed_l1479_147969


namespace NUMINAMATH_CALUDE_amount_with_r_l1479_147933

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l1479_147933


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l1479_147997

/-- Given a quadratic equation x^2 + 2(a-1)x + 2a + 6 = 0 with one positive and one negative real root,
    prove that a < -3 --/
theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
    x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
    y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l1479_147997


namespace NUMINAMATH_CALUDE_middle_of_five_consecutive_integers_l1479_147915

/-- Given 5 consecutive integers with a sum of 60, prove that the middle number is 12 -/
theorem middle_of_five_consecutive_integers (a b c d e : ℤ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_of_five_consecutive_integers_l1479_147915


namespace NUMINAMATH_CALUDE_snack_machine_purchase_l1479_147961

/-- The number of pieces of chocolate bought -/
def chocolate_pieces : ℕ := 2

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total number of quarters used -/
def total_quarters : ℕ := 11

theorem snack_machine_purchase :
  chocolate_pieces * chocolate_cost + 3 * candy_bar_cost + juice_cost = total_quarters * 25 :=
by sorry

end NUMINAMATH_CALUDE_snack_machine_purchase_l1479_147961


namespace NUMINAMATH_CALUDE_product_of_roots_l1479_147929

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃)) →
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = -28) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1479_147929


namespace NUMINAMATH_CALUDE_hamburger_cost_is_four_l1479_147992

/-- The cost of Morgan's lunch items and transaction details -/
structure LunchOrder where
  hamburger_cost : ℝ
  onion_rings_cost : ℝ
  smoothie_cost : ℝ
  total_paid : ℝ
  change_received : ℝ

/-- Theorem stating the cost of the hamburger in Morgan's lunch order -/
theorem hamburger_cost_is_four (order : LunchOrder)
  (h1 : order.onion_rings_cost = 2)
  (h2 : order.smoothie_cost = 3)
  (h3 : order.total_paid = 20)
  (h4 : order.change_received = 11) :
  order.hamburger_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_is_four_l1479_147992


namespace NUMINAMATH_CALUDE_waynes_blocks_l1479_147951

/-- Wayne's block collection problem -/
theorem waynes_blocks (initial_blocks final_blocks father_blocks : ℕ) 
  (h1 : father_blocks = 6)
  (h2 : final_blocks = 15)
  (h3 : final_blocks = initial_blocks + father_blocks) : 
  initial_blocks = 9 := by
  sorry

end NUMINAMATH_CALUDE_waynes_blocks_l1479_147951


namespace NUMINAMATH_CALUDE_money_distribution_l1479_147903

theorem money_distribution (p q r s : ℕ) : 
  p + q + r + s = 10000 →
  r = 2 * p →
  r = 3 * q →
  s = p + q →
  p = 1875 ∧ q = 1250 ∧ r = 3750 ∧ s = 3125 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l1479_147903


namespace NUMINAMATH_CALUDE_not_A_union_B_equiv_l1479_147921

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) ≥ 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem not_A_union_B_equiv : (Aᶜ ∪ B) = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_not_A_union_B_equiv_l1479_147921


namespace NUMINAMATH_CALUDE_solution_set_x_squared_gt_x_l1479_147930

theorem solution_set_x_squared_gt_x : 
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_gt_x_l1479_147930


namespace NUMINAMATH_CALUDE_fraction_equality_l1479_147916

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 27 : ℚ) = 865 / 1000 → a = 173 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1479_147916
