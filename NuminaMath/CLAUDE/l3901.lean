import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3901_390179

theorem square_sum_reciprocal (x : ℝ) (h : 18 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3901_390179


namespace NUMINAMATH_CALUDE_income_calculation_l3901_390170

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 5 →
  income - expenditure = savings →
  savings = 4000 →
  income = 10000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l3901_390170


namespace NUMINAMATH_CALUDE_no_solution_for_floor_equation_l3901_390136

theorem no_solution_for_floor_equation :
  ¬ ∃ s : ℝ, (⌊s⌋ : ℝ) + s = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_equation_l3901_390136


namespace NUMINAMATH_CALUDE_michaels_blocks_l3901_390159

/-- Given that Michael has some blocks stored in boxes, prove that the total number of blocks is 16 -/
theorem michaels_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (h1 : num_boxes = 8) (h2 : blocks_per_box = 2) :
  num_boxes * blocks_per_box = 16 := by
  sorry

end NUMINAMATH_CALUDE_michaels_blocks_l3901_390159


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3901_390107

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3901_390107


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3901_390155

theorem smallest_sum_of_sequence (E F G H : ℕ+) : 
  (∃ d : ℤ, (F : ℤ) - (E : ℤ) = d ∧ (G : ℤ) - (F : ℤ) = d) →  -- arithmetic sequence condition
  (∃ r : ℚ, (G : ℚ) / (F : ℚ) = r ∧ (H : ℚ) / (G : ℚ) = r) →  -- geometric sequence condition
  (G : ℚ) / (F : ℚ) = 4 / 3 →                                -- given ratio
  E + F + G + H ≥ 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3901_390155


namespace NUMINAMATH_CALUDE_congruence_solution_l3901_390145

theorem congruence_solution : ∃ x : ℤ, x ≡ 1 [ZMOD 7] ∧ x ≡ 2 [ZMOD 11] :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3901_390145


namespace NUMINAMATH_CALUDE_dogsled_race_time_difference_l3901_390186

/-- Proves that the difference in time taken to complete a 300-mile course between two teams,
    where one team's average speed is 5 miles per hour greater than the other team's speed
    of 20 miles per hour, is 3 hours. -/
theorem dogsled_race_time_difference :
  let course_length : ℝ := 300
  let team_b_speed : ℝ := 20
  let team_a_speed : ℝ := team_b_speed + 5
  let team_b_time : ℝ := course_length / team_b_speed
  let team_a_time : ℝ := course_length / team_a_speed
  team_b_time - team_a_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_time_difference_l3901_390186


namespace NUMINAMATH_CALUDE_inequality_proof_l3901_390115

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3901_390115


namespace NUMINAMATH_CALUDE_abc_product_absolute_value_l3901_390104

theorem abc_product_absolute_value (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_eq : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) : 
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_absolute_value_l3901_390104


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l3901_390178

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l3901_390178


namespace NUMINAMATH_CALUDE_paise_to_rupees_l3901_390153

/-- 
If 0.5% of a quantity is equal to 65 paise, then the quantity is equal to 130 rupees.
-/
theorem paise_to_rupees (a : ℝ) : (0.005 * a = 65) → (a = 130 * 100) := by
  sorry

end NUMINAMATH_CALUDE_paise_to_rupees_l3901_390153


namespace NUMINAMATH_CALUDE_D_180_l3901_390134

/-- 
D(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 180 -/
def prime_factorization_180 : List ℕ+ := [2, 2, 3, 3, 5]

/-- Theorem stating that D(180) = 43 -/
theorem D_180 : D 180 = 43 := by sorry

end NUMINAMATH_CALUDE_D_180_l3901_390134


namespace NUMINAMATH_CALUDE_stable_journey_population_l3901_390197

/-- Represents the interstellar vehicle Gibraltar --/
structure Gibraltar where
  full_capacity : ℕ
  family_units : ℕ
  members_per_family : ℕ

/-- Calculates the starting population for a stable journey --/
def starting_population (ship : Gibraltar) : ℕ :=
  ship.full_capacity / 3 - 100

/-- Theorem: The starting population for a stable journey is 300 people --/
theorem stable_journey_population (ship : Gibraltar) 
  (h1 : ship.family_units = 300)
  (h2 : ship.members_per_family = 4)
  (h3 : ship.full_capacity = ship.family_units * ship.members_per_family) :
  starting_population ship = 300 := by
  sorry

#eval starting_population { full_capacity := 1200, family_units := 300, members_per_family := 4 }

end NUMINAMATH_CALUDE_stable_journey_population_l3901_390197


namespace NUMINAMATH_CALUDE_trapezoid_larger_base_l3901_390123

/-- Given a trapezoid with base ratio 1:3 and midline length 24, 
    prove the larger base is 36 -/
theorem trapezoid_larger_base 
  (shorter_base longer_base midline : ℝ) 
  (h_ratio : longer_base = 3 * shorter_base) 
  (h_midline : midline = (shorter_base + longer_base) / 2) 
  (h_midline_length : midline = 24) : 
  longer_base = 36 := by
sorry


end NUMINAMATH_CALUDE_trapezoid_larger_base_l3901_390123


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3901_390168

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition
axiom z_condition : Complex.I^3 * z = 2 + Complex.I

-- Theorem to prove
theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3901_390168


namespace NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l3901_390139

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l3901_390139


namespace NUMINAMATH_CALUDE_chocolate_difference_is_fifteen_l3901_390138

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The number of chocolates Alix initially had -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates taken from Alix -/
def chocolates_taken : ℕ := 5

/-- The number of chocolates Alix has after some were taken -/
def alix_remaining_chocolates : ℕ := alix_initial_chocolates - chocolates_taken

/-- The difference in chocolates between Alix and Nick -/
def chocolate_difference : ℕ := alix_remaining_chocolates - nick_chocolates

theorem chocolate_difference_is_fifteen : chocolate_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_is_fifteen_l3901_390138


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3901_390149

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2/y)^(1/3 : ℝ) = 3 ↔ y = 1/11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3901_390149


namespace NUMINAMATH_CALUDE_factorization_proof_l3901_390158

theorem factorization_proof (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3901_390158


namespace NUMINAMATH_CALUDE_total_oranges_l3901_390199

/-- Given 3.0 children and 1.333333333 oranges per child, prove that the total number of oranges is 4. -/
theorem total_oranges (num_children : ℝ) (oranges_per_child : ℝ) 
  (h1 : num_children = 3.0) 
  (h2 : oranges_per_child = 1.333333333) : 
  num_children * oranges_per_child = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l3901_390199


namespace NUMINAMATH_CALUDE_point_one_and_ten_are_reciprocals_l3901_390125

/-- Two numbers are reciprocals if their product is 1 -/
def are_reciprocals (a b : ℝ) : Prop := a * b = 1

/-- 0.1 and 10 are reciprocals of each other -/
theorem point_one_and_ten_are_reciprocals : are_reciprocals 0.1 10 := by
  sorry

end NUMINAMATH_CALUDE_point_one_and_ten_are_reciprocals_l3901_390125


namespace NUMINAMATH_CALUDE_bishop_white_invariant_l3901_390183

/-- Represents a position on a chessboard -/
structure Position where
  i : Nat
  j : Nat
  h_valid : i < 8 ∧ j < 8

/-- Checks if a position is on a white square -/
def isWhite (p : Position) : Prop :=
  (p.i + p.j) % 2 = 1

/-- Represents a valid bishop move -/
inductive BishopMove : Position → Position → Prop where
  | diag (p q : Position) (k : Int) :
      q.i = p.i + k ∧ q.j = p.j + k → BishopMove p q

theorem bishop_white_invariant (p q : Position) (h : BishopMove p q) :
  isWhite p → isWhite q := by
  sorry

end NUMINAMATH_CALUDE_bishop_white_invariant_l3901_390183


namespace NUMINAMATH_CALUDE_prove_h_of_x_l3901_390128

/-- Given that 16x^4 + 5x^3 - 4x + 2 + h(x) = -8x^3 + 7x^2 - 6x + 5,
    prove that h(x) = -16x^4 - 13x^3 + 7x^2 - 2x + 3 -/
theorem prove_h_of_x (x : ℝ) (h : ℝ → ℝ) 
    (eq : 16 * x^4 + 5 * x^3 - 4 * x + 2 + h x = -8 * x^3 + 7 * x^2 - 6 * x + 5) : 
  h x = -16 * x^4 - 13 * x^3 + 7 * x^2 - 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_prove_h_of_x_l3901_390128


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3901_390114

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (new_birds : ℕ) : 
  initial_birds = 3 → initial_storks = 2 → new_birds = 5 →
  initial_birds + initial_storks + new_birds = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3901_390114


namespace NUMINAMATH_CALUDE_quadratic_product_equals_quadratic_l3901_390171

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b : ℤ) : ℤ → ℤ := fun x ↦ x^2 + a * x + b

theorem quadratic_product_equals_quadratic (a b n : ℤ) :
  ∃ M : ℤ, (QuadraticPolynomial a b n) * (QuadraticPolynomial a b (n + 1)) =
    QuadraticPolynomial a b M := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_equals_quadratic_l3901_390171


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3901_390150

/-- The displacement function for the object's motion -/
def displacement (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function, which is the derivative of the displacement function -/
def velocity (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 : velocity 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3901_390150


namespace NUMINAMATH_CALUDE_system_solutions_l3901_390180

def system (x y z : ℚ) : Prop :=
  x^2 + 2*y*z = x ∧ y^2 + 2*z*x = y ∧ z^2 + 2*x*y = z

def solutions : List (ℚ × ℚ × ℚ) :=
  [(0, 0, 0), (1/3, 1/3, 1/3), (1, 0, 0), (0, 1, 0), (0, 0, 1),
   (2/3, -1/3, -1/3), (-1/3, 2/3, -1/3), (-1/3, -1/3, 2/3)]

theorem system_solutions :
  ∀ x y z : ℚ, system x y z ↔ (x, y, z) ∈ solutions := by sorry

end NUMINAMATH_CALUDE_system_solutions_l3901_390180


namespace NUMINAMATH_CALUDE_floor_sum_count_l3901_390156

def count_integers (max : ℕ) : ℕ :=
  let count_for_form (k : ℕ) := (max - k) / 7 + 1
  (count_for_form 0) + (count_for_form 1) + (count_for_form 3) + (count_for_form 4)

theorem floor_sum_count :
  count_integers 1000 = 568 := by sorry

end NUMINAMATH_CALUDE_floor_sum_count_l3901_390156


namespace NUMINAMATH_CALUDE_abs_sum_diff_less_than_two_l3901_390131

theorem abs_sum_diff_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |a + b| + |a - b| < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_diff_less_than_two_l3901_390131


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3901_390165

theorem consecutive_page_numbers_sum (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧
  x * y = 20412 →
  x + y + z = 429 := by
sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3901_390165


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l3901_390194

/-- The ratio of the real semi-axis length to the imaginary axis length of the hyperbola 2x^2 - y^2 = 8 -/
theorem hyperbola_axis_ratio : ∃ (a b : ℝ), 
  (∀ x y : ℝ, 2 * x^2 - y^2 = 8 ↔ x^2 / (2 * a^2) - y^2 / (2 * b^2) = 1) ∧ 
  (a / (2 * b) = Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l3901_390194


namespace NUMINAMATH_CALUDE_infinitely_many_m_for_binomial_equality_l3901_390102

theorem infinitely_many_m_for_binomial_equality :
  ∀ n : ℕ, n ≥ 4 →
  ∃ m : ℕ, m ≥ 2 ∧
    m = (n^2 - 3*n + 2) / 2 ∧
    Nat.choose m 2 = 3 * Nat.choose n 4 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_for_binomial_equality_l3901_390102


namespace NUMINAMATH_CALUDE_bales_in_shed_l3901_390157

theorem bales_in_shed (initial_barn : ℕ) (added : ℕ) (final_barn : ℕ) : 
  initial_barn = 47 → added = 35 → final_barn = 82 → 
  final_barn = initial_barn + added → initial_barn + added = 82 → 0 = final_barn - (initial_barn + added) :=
by
  sorry

end NUMINAMATH_CALUDE_bales_in_shed_l3901_390157


namespace NUMINAMATH_CALUDE_correct_average_l3901_390142

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ incorrect_num = 26 ∧ correct_num = 66 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3901_390142


namespace NUMINAMATH_CALUDE_number_of_goats_l3901_390126

theorem number_of_goats (total_cost : ℕ) (num_cows : ℕ) (cow_price : ℕ) (goat_price : ℕ) : 
  total_cost = 1500 → 
  num_cows = 2 → 
  cow_price = 400 → 
  goat_price = 70 → 
  ∃ (num_goats : ℕ), num_goats = 10 ∧ total_cost = num_cows * cow_price + num_goats * goat_price :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_goats_l3901_390126


namespace NUMINAMATH_CALUDE_rope_length_proof_l3901_390147

theorem rope_length_proof (L : ℝ) 
  (h1 : L - 42 > 0)  -- Ensures the first rope has positive remaining length
  (h2 : L - 12 > 0)  -- Ensures the second rope has positive remaining length
  (h3 : L - 12 = 4 * (L - 42)) : 2 * L = 104 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_proof_l3901_390147


namespace NUMINAMATH_CALUDE_inequalities_check_l3901_390154

theorem inequalities_check :
  (∀ x : ℝ, x^2 + 3 > 2*x) ∧
  (∃ a b : ℝ, a^5 + b^5 < a^3*b^2 + a^2*b^3) ∧
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_check_l3901_390154


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3901_390146

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (p 0)

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 6 = 163 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3901_390146


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3901_390185

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3901_390185


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l3901_390122

/-- Represents a cone with its base radius -/
structure Cone :=
  (baseRadius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerBaseRadius : ℝ)

/-- Checks if three cones are touching each other -/
def areTouching (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- Checks if a truncated cone has a common generatrix with other cones -/
def hasCommonGeneratrix (tc : TruncatedCone) (c1 c2 c3 : Cone) : Prop :=
  -- This is a simplification. In reality, we'd need to check the geometric conditions.
  true

/-- The main theorem -/
theorem truncated_cone_radius 
  (c1 c2 c3 : Cone) 
  (tc : TruncatedCone) 
  (h1 : c1.baseRadius = 6) 
  (h2 : c2.baseRadius = 24) 
  (h3 : c3.baseRadius = 24) 
  (h4 : areTouching c1 c2 c3) 
  (h5 : hasCommonGeneratrix tc c1 c2 c3) : 
  tc.smallerBaseRadius = 2 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l3901_390122


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3901_390133

theorem junk_mail_distribution (blocks : ℕ) (houses_per_block : ℕ) (total_mail : ℕ) 
  (h1 : blocks = 16) 
  (h2 : houses_per_block = 17) 
  (h3 : total_mail = 1088) : 
  total_mail / (blocks * houses_per_block) = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3901_390133


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3901_390132

/-- Given a quadratic function y = -x^2 + 8x - 7 -/
def f (x : ℝ) : ℝ := -x^2 + 8*x - 7

theorem quadratic_function_properties :
  /- (1) y increases as x increases for x < 4 -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 4 → f x₁ < f x₂) ∧
  /- (2) y < 0 for x < 1 or x > 7 -/
  (∀ x : ℝ, (x < 1 ∨ x > 7) → f x < 0) :=
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3901_390132


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3901_390124

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3901_390124


namespace NUMINAMATH_CALUDE_felix_distance_covered_l3901_390111

/-- The initial speed in miles per hour -/
def initial_speed : ℝ := 66

/-- The number of hours Felix wants to drive -/
def drive_hours : ℝ := 4

/-- The factor by which Felix wants to increase his speed -/
def speed_increase_factor : ℝ := 2

/-- Calculates the distance covered given a speed and time -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance Felix will cover -/
theorem felix_distance_covered : 
  distance_covered (initial_speed * speed_increase_factor) drive_hours = 528 := by
  sorry

end NUMINAMATH_CALUDE_felix_distance_covered_l3901_390111


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3901_390140

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l3901_390140


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l3901_390117

/-- Given two externally tangent circles with common external tangents,
    prove that the area of the smaller circle is π(625 + 200√2) / 49 -/
theorem area_of_smaller_circle (P A B A' B' S L : ℝ × ℝ) : 
  let r := Real.sqrt ((5 + 10 * Real.sqrt 2) ^ 2 / 49)
  -- Two circles are externally tangent
  (∃ T : ℝ × ℝ, ‖S - T‖ = r ∧ ‖L - T‖ = 2*r) →
  -- PAB and PA'B' are common external tangents
  (‖P - A‖ = 5 ∧ ‖A - B‖ = 5 ∧ ‖P - A'‖ = 5 ∧ ‖A' - B'‖ = 5) →
  -- A and A' are on the smaller circle
  (‖S - A‖ = r ∧ ‖S - A'‖ = r) →
  -- B and B' are on the larger circle
  (‖L - B‖ = 2*r ∧ ‖L - B'‖ = 2*r) →
  -- Area of the smaller circle
  π * r^2 = π * (625 + 200 * Real.sqrt 2) / 49 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l3901_390117


namespace NUMINAMATH_CALUDE_solution_set_for_negative_two_minimum_value_for_one_range_of_m_l3901_390166

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2*x - 1

-- Part 1
theorem solution_set_for_negative_two (x : ℝ) :
  f (-2) x ≤ 0 ↔ x ≥ 1 :=
sorry

-- Part 2
theorem minimum_value_for_one (x : ℝ) :
  f 1 x + |x + 2| ≥ 0 :=
sorry

-- Range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f 1 x + |x + 2| ≤ m) ↔ m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_two_minimum_value_for_one_range_of_m_l3901_390166


namespace NUMINAMATH_CALUDE_digit_product_theorem_l3901_390161

theorem digit_product_theorem (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2244 →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_digit_product_theorem_l3901_390161


namespace NUMINAMATH_CALUDE_inequality_solution_l3901_390113

theorem inequality_solution (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ x ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3901_390113


namespace NUMINAMATH_CALUDE_difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l3901_390129

/-- Given two real numbers with a difference of a, prove that if we increase the subtrahend by 0.5, the new difference is a - 0.5 -/
theorem difference_after_subtrahend_increase (x y a : ℝ) (h : x - y = a) : 
  x - (y + 0.5) = a - 0.5 := by
sorry

/-- Alternative formulation using let bindings for clarity -/
theorem difference_after_subtrahend_increase_alt (a : ℝ) : 
  ∀ x y : ℝ, x - y = a → x - (y + 0.5) = a - 0.5 := by
sorry

end NUMINAMATH_CALUDE_difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l3901_390129


namespace NUMINAMATH_CALUDE_daily_savings_amount_l3901_390152

/-- Represents the number of days Ian saves money -/
def savingDays : ℕ := 40

/-- Represents the total amount saved in dimes -/
def totalSavedDimes : ℕ := 4

/-- Represents the value of a dime in cents -/
def dimeValueInCents : ℕ := 10

/-- Theorem: If Ian saves for 40 days and accumulates 4 dimes, his daily savings is 1 cent -/
theorem daily_savings_amount : 
  (totalSavedDimes * dimeValueInCents) / savingDays = 1 := by
  sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l3901_390152


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3901_390101

theorem sum_reciprocals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3901_390101


namespace NUMINAMATH_CALUDE_june_science_book_price_l3901_390143

/-- Calculates the price of each science book given June's school supply purchases. -/
theorem june_science_book_price (total_budget : ℕ) (math_book_price : ℕ) (math_book_count : ℕ)
  (art_book_price : ℕ) (music_book_cost : ℕ) :
  total_budget = 500 →
  math_book_price = 20 →
  math_book_count = 4 →
  art_book_price = 20 →
  music_book_cost = 160 →
  let science_book_count := math_book_count + 6
  let art_book_count := 2 * math_book_count
  let total_spent := math_book_price * math_book_count +
                     art_book_price * art_book_count +
                     music_book_cost
  let remaining_budget := total_budget - total_spent
  remaining_budget / science_book_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_june_science_book_price_l3901_390143


namespace NUMINAMATH_CALUDE_number_problem_l3901_390196

theorem number_problem : 
  ∃ x : ℚ, x = (3/7)*x + 200 ∧ x = 350 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3901_390196


namespace NUMINAMATH_CALUDE_tan_inequality_l3901_390144

theorem tan_inequality (x : ℝ) (h : 0 ≤ x ∧ x < 1) :
  (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan (Real.pi * x / 2) ∧
  Real.tan (Real.pi * x / 2) ≤ (Real.pi / 2) * (x / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l3901_390144


namespace NUMINAMATH_CALUDE_students_just_passed_l3901_390160

theorem students_just_passed (total_students : ℕ) 
  (first_division_percent : ℚ) (second_division_percent : ℚ) :
  total_students = 300 →
  first_division_percent = 26 / 100 →
  second_division_percent = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percent - second_division_percent) = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l3901_390160


namespace NUMINAMATH_CALUDE_toy_car_factory_ratio_l3901_390173

/-- The ratio of cars made today to cars made yesterday -/
def car_ratio (cars_yesterday cars_today : ℕ) : ℚ :=
  cars_today / cars_yesterday

theorem toy_car_factory_ratio : 
  let cars_yesterday : ℕ := 60
  let total_cars : ℕ := 180
  let cars_today : ℕ := total_cars - cars_yesterday
  car_ratio cars_yesterday cars_today = 2 := by
sorry

end NUMINAMATH_CALUDE_toy_car_factory_ratio_l3901_390173


namespace NUMINAMATH_CALUDE_fraction_positivity_implies_x_range_l3901_390182

theorem fraction_positivity_implies_x_range (x : ℝ) : (-6 : ℝ) / (7 - x) > 0 → x > 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_positivity_implies_x_range_l3901_390182


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3901_390110

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3901_390110


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_l3901_390184

/-- The commission rate of the online store -/
def commission_rate : ℚ := 1/5

/-- The price at which the distributor obtains the product from the producer -/
def producer_price : ℚ := 18

/-- The price observed by the buyer on the online store -/
def buyer_price : ℚ := 27

/-- The selling price of the distributor to the online store -/
def selling_price : ℚ := buyer_price / (1 + commission_rate)

/-- The profit made by the distributor per item -/
def profit : ℚ := selling_price - producer_price

/-- The profit percentage of the distributor -/
def profit_percentage : ℚ := profit / producer_price * 100

theorem distributor_profit_percentage :
  profit_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_l3901_390184


namespace NUMINAMATH_CALUDE_five_digit_sum_l3901_390198

def sum_of_digits (x : ℕ) : ℕ := 1 + 3 + 4 + 6 + x

def number_of_permutations : ℕ := 120  -- This is A₅⁵

theorem five_digit_sum (x : ℕ) :
  sum_of_digits x * number_of_permutations = 2640 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l3901_390198


namespace NUMINAMATH_CALUDE_square_difference_division_l3901_390105

theorem square_difference_division : (175^2 - 155^2) / 20 = 330 := by sorry

end NUMINAMATH_CALUDE_square_difference_division_l3901_390105


namespace NUMINAMATH_CALUDE_trig_simplification_l3901_390195

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3901_390195


namespace NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l3901_390176

theorem residue_of_negative_1235_mod_29 : Int.mod (-1235) 29 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l3901_390176


namespace NUMINAMATH_CALUDE_man_birth_year_l3901_390118

-- Define the birth year function
def birthYear (x : ℕ) : ℕ := x^2 - x - 2

-- State the theorem
theorem man_birth_year :
  ∃ x : ℕ, 
    (birthYear x > 1900) ∧ 
    (birthYear x < 1950) ∧ 
    (birthYear x = 1890) := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_l3901_390118


namespace NUMINAMATH_CALUDE_chinese_character_sum_l3901_390169

theorem chinese_character_sum (a b c d : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  100 * a + 10 * b + c + 100 * c + 10 * b + d = 1000 * a + 100 * b + 10 * c + d →
  1000 * a + 100 * b + 10 * c + d = 18 := by
sorry

end NUMINAMATH_CALUDE_chinese_character_sum_l3901_390169


namespace NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l3901_390103

/-- If the terminal side of angle α passes through point P(-5,-12), then sin(3π/2 + α) = 5/13 -/
theorem sin_three_pi_half_plus_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = -5 ∧ r * (Real.sin α) = -12) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l3901_390103


namespace NUMINAMATH_CALUDE_perfect_square_expression_l3901_390116

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l3901_390116


namespace NUMINAMATH_CALUDE_solve_equation_l3901_390181

theorem solve_equation :
  ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3901_390181


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relation_l3901_390119

theorem inverse_proportion_y_relation (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k < 0) 
  (h2 : y₁ = k / (-4)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relation_l3901_390119


namespace NUMINAMATH_CALUDE_species_x_count_day_6_l3901_390164

/-- Represents the number of days passed -/
def days : ℕ := 6

/-- The population growth factor for Species X per day -/
def species_x_growth : ℕ := 2

/-- The population growth factor for Species Y per day -/
def species_y_growth : ℕ := 4

/-- The total number of ants on Day 0 -/
def initial_total : ℕ := 40

/-- The total number of ants on Day 6 -/
def final_total : ℕ := 21050

/-- Theorem stating that the number of Species X ants on Day 6 is 2304 -/
theorem species_x_count_day_6 : ℕ := by
  sorry

end NUMINAMATH_CALUDE_species_x_count_day_6_l3901_390164


namespace NUMINAMATH_CALUDE_lemons_for_ten_gallons_l3901_390108

/-- The number of lemons required to make a certain amount of lemonade -/
structure LemonadeRecipe where
  lemons : ℕ
  gallons : ℕ

/-- Calculates the number of lemons needed for a given number of gallons,
    based on a known recipe. The result is rounded up to the nearest integer. -/
def calculate_lemons (recipe : LemonadeRecipe) (target_gallons : ℕ) : ℕ :=
  ((recipe.lemons : ℚ) * target_gallons / recipe.gallons).ceil.toNat

/-- The known recipe for lemonade -/
def known_recipe : LemonadeRecipe := ⟨48, 64⟩

/-- The target amount of lemonade to make -/
def target_gallons : ℕ := 10

/-- Theorem stating that 8 lemons are needed to make 10 gallons of lemonade -/
theorem lemons_for_ten_gallons :
  calculate_lemons known_recipe target_gallons = 8 := by sorry

end NUMINAMATH_CALUDE_lemons_for_ten_gallons_l3901_390108


namespace NUMINAMATH_CALUDE_sarah_interview_count_l3901_390175

theorem sarah_interview_count (oranges pears apples strawberries : ℕ) 
  (h_oranges : oranges = 70)
  (h_pears : pears = 120)
  (h_apples : apples = 147)
  (h_strawberries : strawberries = 113) :
  oranges + pears + apples + strawberries = 450 := by
  sorry

end NUMINAMATH_CALUDE_sarah_interview_count_l3901_390175


namespace NUMINAMATH_CALUDE_sequence_properties_l3901_390135

def a (n : ℕ) : ℚ := (2 * n) / (3 * n + 2)

theorem sequence_properties : 
  (a 3 = 6 / 11) ∧ 
  (∀ n : ℕ, a (n - 1) = (2 * n - 2) / (3 * n - 1)) ∧ 
  (a 8 = 8 / 13) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3901_390135


namespace NUMINAMATH_CALUDE_solve_equation_l3901_390190

theorem solve_equation (x : ℝ) : 0.3 * x = 45 → (10 / 3) * (0.3 * x) = 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3901_390190


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3901_390137

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = x}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {y : ℝ | y ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3901_390137


namespace NUMINAMATH_CALUDE_buddy_program_fraction_l3901_390188

theorem buddy_program_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (n / 4 : ℚ) = (s / 3 : ℚ) → 
  ((n / 4 + s / 3) / (n + s) : ℚ) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_buddy_program_fraction_l3901_390188


namespace NUMINAMATH_CALUDE_mutual_greetings_l3901_390189

theorem mutual_greetings (n : ℕ) (min_sent : ℕ) (h1 : n = 30) (h2 : min_sent = 16) :
  let total_sent := n * min_sent
  let total_pairs := n * (n - 1) / 2
  let mutual_greetings := {x : ℕ // x ≤ total_pairs ∧ 2 * x + (total_sent - 2 * x) ≤ total_sent}
  ∃ (x : mutual_greetings), x.val ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_mutual_greetings_l3901_390189


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3901_390106

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3901_390106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3901_390100

-- Define the sequence terms
def term (k : ℕ) (A B : ℝ) : ℝ := (4 + 3 * (k - 1)) * A + (5 + 3 * (k - 1)) * B

-- State the theorem
theorem arithmetic_sequence_15th_term (a b : ℝ) (A B : ℝ) (h1 : A = Real.log a) (h2 : B = Real.log b) :
  (∀ k : ℕ, k ≥ 1 → k ≤ 3 → term k A B = Real.log (a^(4 + 3*(k-1)) * b^(5 + 3*(k-1)))) →
  term 15 A B = Real.log (b^93) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3901_390100


namespace NUMINAMATH_CALUDE_number_of_binders_l3901_390191

theorem number_of_binders (total_sheets : ℕ) (sheets_per_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : sheets_per_binder = 490)
  (h3 : total_sheets % sheets_per_binder = 0) :
  total_sheets / sheets_per_binder = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_binders_l3901_390191


namespace NUMINAMATH_CALUDE_range_of_m_l3901_390141

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) → m ∈ Set.Ioo (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3901_390141


namespace NUMINAMATH_CALUDE_green_balloons_l3901_390163

theorem green_balloons (total : Nat) (red : Nat) (green : Nat) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_l3901_390163


namespace NUMINAMATH_CALUDE_five_coins_all_heads_or_tails_prob_l3901_390174

/-- The probability of getting all heads or all tails when flipping n fair coins -/
def all_heads_or_tails_prob (n : ℕ) : ℚ :=
  2 / 2^n

/-- Theorem: The probability of getting all heads or all tails when flipping 5 fair coins is 1/16 -/
theorem five_coins_all_heads_or_tails_prob :
  all_heads_or_tails_prob 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_five_coins_all_heads_or_tails_prob_l3901_390174


namespace NUMINAMATH_CALUDE_smallest_x_value_l3901_390127

theorem smallest_x_value (y : ℕ+) (x : ℕ) 
  (h : (4 : ℚ) / 5 = (y : ℚ) / ((200 : ℚ) + x)) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3901_390127


namespace NUMINAMATH_CALUDE_complex_power_sum_l3901_390192

theorem complex_power_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  Complex.abs (a^2020 + b^2020 + c^2020) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3901_390192


namespace NUMINAMATH_CALUDE_distance_calculation_l3901_390172

theorem distance_calculation (A B C D : ℝ) 
  (h1 : A = 350)
  (h2 : A + B = 600)
  (h3 : A + B + C + D = 1500)
  (h4 : D = 275) :
  C = 625 ∧ B + C = 875 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l3901_390172


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3901_390187

/-- The distance between the foci of a hyperbola defined by xy = 4 is 2√10. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / f₁.1^2 - (y - f₁.2)^2 / f₁.2^2 = 1) ∧
    (∀ (x y : ℝ), x * y = 4 → (x - f₂.1)^2 / f₂.1^2 - (y - f₂.2)^2 / f₂.2^2 = 1) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3901_390187


namespace NUMINAMATH_CALUDE_learning_machine_price_reduction_l3901_390162

/-- Represents the price reduction scenario of a learning machine -/
def price_reduction_equation (initial_price final_price : ℝ) (num_reductions : ℕ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^num_reductions = final_price

/-- The equation 2000(1-x)^2 = 1280 correctly represents the given price reduction scenario -/
theorem learning_machine_price_reduction :
  price_reduction_equation 2000 1280 2 x ↔ 2000 * (1 - x)^2 = 1280 :=
sorry

end NUMINAMATH_CALUDE_learning_machine_price_reduction_l3901_390162


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3901_390112

/-- The number of available toppings -/
def num_toppings : ℕ := 10

/-- The number of slice options -/
def num_slice_options : ℕ := 4

/-- The total number of sandwich combinations -/
def total_combinations : ℕ := num_slice_options * 2^num_toppings

/-- Theorem: The total number of sandwich combinations is 4096 -/
theorem sandwich_combinations :
  total_combinations = 4096 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3901_390112


namespace NUMINAMATH_CALUDE_poly_arrangement_l3901_390120

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 3*x*y^3 - x^2*y^3 - 9*y + x^3

/-- The polynomial arranged in ascending order of x -/
def arranged_poly (x y : ℝ) : ℝ := -9*y + 3*x*y^3 - x^2*y^3 + x^3

/-- Theorem stating that the arranged polynomial is equivalent to the original polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_arrangement_l3901_390120


namespace NUMINAMATH_CALUDE_percentage_difference_l3901_390109

theorem percentage_difference (A B x : ℝ) : 
  A > B ∧ B > 0 → A = B * (1 + x / 100) → x = 100 * (A - B) / B := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3901_390109


namespace NUMINAMATH_CALUDE_competition_score_l3901_390151

theorem competition_score (correct_points incorrect_points total_questions final_score : ℕ) 
  (h1 : correct_points = 6)
  (h2 : incorrect_points = 3)
  (h3 : total_questions = 15)
  (h4 : final_score = 36) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = final_score ∧
    correct_answers = 9 := by
  sorry

#check competition_score

end NUMINAMATH_CALUDE_competition_score_l3901_390151


namespace NUMINAMATH_CALUDE_square_root_div_five_l3901_390193

theorem square_root_div_five : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_div_five_l3901_390193


namespace NUMINAMATH_CALUDE_range_of_p_characterization_l3901_390177

def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

def range_of_p : Set ℝ := {p | B p ⊆ A}

theorem range_of_p_characterization :
  range_of_p = 
    {p | B p = ∅} ∪ 
    {p | B p ≠ ∅ ∧ ∀ x ∈ B p, x ∈ A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_characterization_l3901_390177


namespace NUMINAMATH_CALUDE_farm_dogs_l3901_390148

/-- Given a farm with dog-houses and dogs, calculate the total number of dogs. -/
def total_dogs (num_houses : ℕ) (dogs_per_house : ℕ) : ℕ :=
  num_houses * dogs_per_house

/-- Theorem: There are 20 dogs in total on the farm. -/
theorem farm_dogs : total_dogs 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_farm_dogs_l3901_390148


namespace NUMINAMATH_CALUDE_rowans_rate_l3901_390130

/-- Rowan's rowing problem -/
theorem rowans_rate (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ)
  (h1 : downstream_distance = 26)
  (h2 : downstream_time = 2)
  (h3 : upstream_time = 4)
  (h4 : downstream_time > 0)
  (h5 : upstream_time > 0) :
  ∃ (still_water_rate : ℝ) (current_rate : ℝ),
    still_water_rate = 9.75 ∧
    (still_water_rate + current_rate) * downstream_time = downstream_distance ∧
    (still_water_rate - current_rate) * upstream_time = downstream_distance :=
by
  sorry


end NUMINAMATH_CALUDE_rowans_rate_l3901_390130


namespace NUMINAMATH_CALUDE_circle_diameter_when_area_circumference_ratio_is_5_l3901_390121

-- Define the circle properties
def circle_area (M : ℝ) := M
def circle_circumference (N : ℝ) := N

-- Theorem statement
theorem circle_diameter_when_area_circumference_ratio_is_5 
  (M N : ℝ) 
  (h1 : M > 0) 
  (h2 : N > 0) 
  (h3 : circle_area M / circle_circumference N = 5) : 
  2 * (circle_circumference N / (2 * Real.pi)) = 20 := by
  sorry

#check circle_diameter_when_area_circumference_ratio_is_5

end NUMINAMATH_CALUDE_circle_diameter_when_area_circumference_ratio_is_5_l3901_390121


namespace NUMINAMATH_CALUDE_cedarwood_earnings_theorem_l3901_390167

/-- Represents the data for each school's participation in the community project -/
structure SchoolData where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total earnings for Cedarwood school given the project data -/
def cedarwoodEarnings (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) : ℚ :=
  let totalStudentDays := ashwood.students * ashwood.days + briarwood.students * briarwood.days + cedarwood.students * cedarwood.days
  let dailyWage := totalPaid / totalStudentDays
  dailyWage * (cedarwood.students * cedarwood.days)

/-- Theorem stating that Cedarwood school's earnings are 454.74 given the project conditions -/
theorem cedarwood_earnings_theorem (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) :
  ashwood.name = "Ashwood" ∧ ashwood.students = 9 ∧ ashwood.days = 4 ∧
  briarwood.name = "Briarwood" ∧ briarwood.students = 5 ∧ briarwood.days = 6 ∧
  cedarwood.name = "Cedarwood" ∧ cedarwood.students = 6 ∧ cedarwood.days = 8 ∧
  totalPaid = 1080 →
  cedarwoodEarnings ashwood briarwood cedarwood totalPaid = 454.74 := by
  sorry

#eval cedarwoodEarnings
  { name := "Ashwood", students := 9, days := 4 }
  { name := "Briarwood", students := 5, days := 6 }
  { name := "Cedarwood", students := 6, days := 8 }
  1080

end NUMINAMATH_CALUDE_cedarwood_earnings_theorem_l3901_390167
