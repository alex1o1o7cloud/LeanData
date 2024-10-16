import Mathlib

namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l2232_223235

-- (1) (3x-1)^2 = (x+1)^2
theorem equation_one_solutions (x : ℝ) :
  (3*x - 1)^2 = (x + 1)^2 ↔ x = 0 ∨ x = 1 := by sorry

-- (2) (x-1)^2+2x(x-1)=0
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = 1 ∨ x = 1/3 := by sorry

-- (3) x^2 - 4x + 1 = 0
theorem equation_three_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

-- (4) 2x^2 + 7x - 4 = 0
theorem equation_four_solutions (x : ℝ) :
  2*x^2 + 7*x - 4 = 0 ↔ x = 1/2 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l2232_223235


namespace NUMINAMATH_CALUDE_no_solution_for_2023_l2232_223236

theorem no_solution_for_2023 : ¬ ∃ (a b : ℤ), a^2 + b^2 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_2023_l2232_223236


namespace NUMINAMATH_CALUDE_line_parameterization_vector_l2232_223231

/-- The line equation y = (4x - 7) / 3 -/
def line_equation (x y : ℝ) : Prop := y = (4 * x - 7) / 3

/-- The parameterization of the line -/
def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

/-- The distance constraint -/
def distance_constraint (p : ℝ × ℝ) (t : ℝ) : Prop :=
  p.1 ≥ 5 → ‖(p.1 - 5, p.2 - 2)‖ = 2 * t

/-- The theorem statement -/
theorem line_parameterization_vector :
  ∃ (v : ℝ × ℝ), ∀ (x y t : ℝ),
    let p := parameterization v (6/5, 8/5) t
    line_equation p.1 p.2 ∧
    distance_constraint p t :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_vector_l2232_223231


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2232_223265

theorem sqrt_product_simplification (y : ℝ) (hy : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2232_223265


namespace NUMINAMATH_CALUDE_yanni_toy_cost_l2232_223208

/-- The cost of the toy Yanni bought -/
def toy_cost (initial_money mother_gift found_money money_left : ℚ) : ℚ :=
  initial_money + mother_gift + found_money - money_left

/-- Theorem stating the cost of the toy Yanni bought -/
theorem yanni_toy_cost :
  toy_cost 0.85 0.40 0.50 0.15 = 1.60 := by
  sorry

end NUMINAMATH_CALUDE_yanni_toy_cost_l2232_223208


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2232_223274

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(1) - f(-1) = -6, then b = -3 -/
theorem quadratic_function_property (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f 1 - f (-1) = -6) → b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2232_223274


namespace NUMINAMATH_CALUDE_existence_of_parameters_l2232_223261

theorem existence_of_parameters : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_parameters_l2232_223261


namespace NUMINAMATH_CALUDE_y_divisibility_l2232_223297

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 24 * k) ∧
  ¬(∃ k : ℕ, y = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l2232_223297


namespace NUMINAMATH_CALUDE_quadratic_coincidence_l2232_223291

/-- A quadratic function with vertex at the origin -/
def QuadraticAtOrigin (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- The translated quadratic function -/
def TranslatedQuadratic : ℝ → ℝ := λ x ↦ 2 * x^2 + x - 1

/-- Theorem stating that if a quadratic function with vertex at the origin
    can be translated to coincide with y = 2x² + x - 1,
    then its analytical expression is y = 2x² -/
theorem quadratic_coincidence (a : ℝ) :
  (∃ h k : ℝ, ∀ x, QuadraticAtOrigin a (x - h) + k = TranslatedQuadratic x) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coincidence_l2232_223291


namespace NUMINAMATH_CALUDE_stating_different_choices_eq_30_l2232_223299

/-- The number of courses available. -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses. -/
def courses_per_person : ℕ := 2

/-- 
The number of ways in which A and B can choose courses with at least one different course.
This is defined as a function that takes no arguments and returns a natural number.
-/
def different_choices : ℕ :=
  sorry

/-- 
Theorem stating that the number of ways A and B can choose courses
with at least one different course is equal to 30.
-/
theorem different_choices_eq_30 : different_choices = 30 :=
  sorry

end NUMINAMATH_CALUDE_stating_different_choices_eq_30_l2232_223299


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2232_223203

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) (a : ℝ) : Prop :=
  z / (a + 2 * i) = i

-- Define the condition that real part equals imaginary part
def real_equals_imag (z : ℂ) : Prop :=
  z.re = z.im

-- The theorem to prove
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : given_equation z a) 
  (h2 : real_equals_imag (z / (a + 2 * i))) : 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2232_223203


namespace NUMINAMATH_CALUDE_all_left_probability_l2232_223212

/-- Represents the particle movement experiment -/
structure ParticleExperiment where
  total_particles : ℕ
  initial_left : ℕ
  initial_right : ℕ

/-- The probability of all particles ending on the left side -/
def probability_all_left (exp : ParticleExperiment) : ℚ :=
  1 / 2

/-- The main theorem stating the probability of all particles ending on the left side -/
theorem all_left_probability (exp : ParticleExperiment) 
  (h1 : exp.total_particles = 100)
  (h2 : exp.initial_left = 32)
  (h3 : exp.initial_right = 68)
  (h4 : exp.initial_left + exp.initial_right = exp.total_particles) :
  probability_all_left exp = 1 / 2 := by
  sorry

#eval (100 * 1 + 2 : ℕ)

end NUMINAMATH_CALUDE_all_left_probability_l2232_223212


namespace NUMINAMATH_CALUDE_final_single_stone_piles_l2232_223230

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Initial game state -/
def initialState : GameState :=
  { piles := List.range 10 |>.map (· + 1) }

/-- Combines two piles and adds 2 stones -/
def combinePiles (state : GameState) (i j : Nat) : GameState :=
  sorry

/-- Splits a pile into two after removing 2 stones -/
def splitPile (state : GameState) (i : Nat) (split : Nat) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of piles with one stone -/
def countSingleStonePiles (state : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem final_single_stone_piles (finalState : GameState) :
  isGameOver finalState → countSingleStonePiles finalState = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_single_stone_piles_l2232_223230


namespace NUMINAMATH_CALUDE_no_rational_roots_l2232_223290

theorem no_rational_roots : ∀ (p q : ℤ), q ≠ 0 → 3 * (p / q)^4 - 2 * (p / q)^3 - 8 * (p / q)^2 + (p / q) + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2232_223290


namespace NUMINAMATH_CALUDE_female_student_count_l2232_223275

theorem female_student_count (total : ℕ) (combinations : ℕ) : 
  total = 8 → 
  combinations = 30 → 
  (∃ (male female : ℕ), 
    male + female = total ∧ 
    Nat.choose male 2 * female = combinations ∧ 
    (female = 2 ∨ female = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_female_student_count_l2232_223275


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2232_223281

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (hp : selling_price = 1170)
  (hq : profit_percentage = 20) : 
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_percentage / 100) = selling_price ∧ 
    cost_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2232_223281


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2232_223210

/-- Given a line in the form (3, -4) · ((x, y) - (2, 8)) = 0,
    prove that it's equivalent to y = (3/4)x + 6.5 -/
theorem line_equation_equivalence (x y : ℝ) :
  (3 * (x - 2) + (-4) * (y - 8) = 0) ↔ (y = (3/4) * x + 6.5) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2232_223210


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2232_223233

/-- Given a right-angled triangle ABC with ∠A = π/2, D is the foot of the altitude from A to BC,
    BD = m, and DC = n. This theorem proves that arctan(b/(m+c)) + arctan(c/(n+b)) = π/4. -/
theorem right_triangle_arctan_sum (a b c m n : ℝ) (h_right : a^2 = b^2 + c^2)
  (h_altitude : m * n = a^2) (h_sum : m * b + c * n = b * n + c * m) :
  Real.arctan (b / (m + c)) + Real.arctan (c / (n + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2232_223233


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2232_223227

-- Problem 1
theorem problem_1 : (-2)^3 / (-2)^2 * (1/2)^0 = -2 := by sorry

-- Problem 2
theorem problem_2 : 199 * 201 + 1 = 40000 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2232_223227


namespace NUMINAMATH_CALUDE_triangle_formation_proof_l2232_223205

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Given sticks of lengths 4 and 10, proves which of the given lengths can form a triangle -/
theorem triangle_formation_proof :
  let a : ℝ := 4
  let b : ℝ := 10
  (¬ can_form_triangle a b 3) ∧
  (¬ can_form_triangle a b 5) ∧
  (can_form_triangle a b 8) ∧
  (¬ can_form_triangle a b 15) := by
  sorry

#check triangle_formation_proof

end NUMINAMATH_CALUDE_triangle_formation_proof_l2232_223205


namespace NUMINAMATH_CALUDE_poster_ratio_l2232_223267

theorem poster_ratio (total medium large small : ℕ) : 
  total = 50 ∧ 
  medium = total / 2 ∧ 
  large = 5 ∧ 
  small = total - medium - large → 
  small * 5 = total * 2 := by
sorry

end NUMINAMATH_CALUDE_poster_ratio_l2232_223267


namespace NUMINAMATH_CALUDE_sum_six_consecutive_even_integers_l2232_223241

/-- The sum of six consecutive even integers, starting from m, is equal to 6m + 30 -/
theorem sum_six_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_even_integers_l2232_223241


namespace NUMINAMATH_CALUDE_volleyball_team_selection_count_l2232_223249

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 6 starters from a team of 15 players,
    including 4 quadruplets, with at least two quadruplets in the starting lineup -/
def volleyball_team_selection : ℕ :=
  choose 4 2 * choose 11 4 +
  choose 4 3 * choose 11 3 +
  choose 11 2

theorem volleyball_team_selection_count :
  volleyball_team_selection = 2695 := by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_count_l2232_223249


namespace NUMINAMATH_CALUDE_log_13_3x_bounds_l2232_223214

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_13_3x_bounds (x : ℝ) (h : log 7 (x + 6) = 2) : 
  1 < log 13 (3 * x) ∧ log 13 (3 * x) < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_13_3x_bounds_l2232_223214


namespace NUMINAMATH_CALUDE_sqrt_2023_minus_x_meaningful_l2232_223242

theorem sqrt_2023_minus_x_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2023 - x) ↔ x ≤ 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_minus_x_meaningful_l2232_223242


namespace NUMINAMATH_CALUDE_solution_set_supremum_a_l2232_223211

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: The solution set of f(x) > 3
theorem solution_set (x : ℝ) : f x > 3 ↔ x < 0 ∨ x > 3 := by sorry

-- Theorem 2: The supremum of a for which f(x) > a holds for all x
theorem supremum_a : ∀ a : ℝ, (∀ x : ℝ, f x > a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_supremum_a_l2232_223211


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2232_223294

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 ∧ 
  ∀ m : ℕ, (m : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 → m ≥ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2232_223294


namespace NUMINAMATH_CALUDE_era_burger_division_l2232_223253

/-- The number of slices each of the third and fourth friends receive when Era divides her burgers. -/
def slices_per_friend (total_burgers : ℕ) (first_friend_slices second_friend_slices era_slices : ℕ) : ℕ :=
  let total_slices := total_burgers * 2
  let remaining_slices := total_slices - (first_friend_slices + second_friend_slices + era_slices)
  remaining_slices / 2

/-- Theorem stating that under the given conditions, each of the third and fourth friends receives 3 slices. -/
theorem era_burger_division :
  slices_per_friend 5 1 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_era_burger_division_l2232_223253


namespace NUMINAMATH_CALUDE_mason_ring_cost_l2232_223259

/-- The cost of buying gold rings for all index fingers of a person -/
def total_cost (ring_price : ℕ) (num_index_fingers : ℕ) : ℕ :=
  ring_price * num_index_fingers

/-- Theorem: Mason will pay $48 for gold rings for his spouse's index fingers -/
theorem mason_ring_cost : total_cost 24 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_mason_ring_cost_l2232_223259


namespace NUMINAMATH_CALUDE_problem_statement_l2232_223260

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2014 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2232_223260


namespace NUMINAMATH_CALUDE_segment_sum_midpoint_inequality_l2232_223201

theorem segment_sum_midpoint_inequality
  (f : ℚ → ℤ) :
  ∃ (x y : ℚ), f x + f y ≤ 2 * f ((x + y) / 2) :=
sorry

end NUMINAMATH_CALUDE_segment_sum_midpoint_inequality_l2232_223201


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l2232_223244

/-- A parallelogram with side lengths 10, 12x-2, 5y+5, and 4 has x+y equal to 4/5 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (12 * x - 2 = 10) → (5 * y + 5 = 4) → x + y = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l2232_223244


namespace NUMINAMATH_CALUDE_impossible_sum_of_two_smaller_angles_l2232_223289

theorem impossible_sum_of_two_smaller_angles (α β γ : ℝ) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = 180 → 
  α ≤ γ → β ≤ γ → 
  α + β ≠ 130 :=
sorry

end NUMINAMATH_CALUDE_impossible_sum_of_two_smaller_angles_l2232_223289


namespace NUMINAMATH_CALUDE_expected_pairs_eq_63_l2232_223298

/-- The number of students in the gathering -/
def n : ℕ := 15

/-- The probability of any pair of students liking each other -/
def p : ℚ := 3/5

/-- The expected number of pairs that like each other -/
def expected_pairs : ℚ := p * (n.choose 2)

theorem expected_pairs_eq_63 : expected_pairs = 63 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_eq_63_l2232_223298


namespace NUMINAMATH_CALUDE_division_problem_l2232_223238

theorem division_problem : (72 : ℝ) / (6 / (3 / 2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2232_223238


namespace NUMINAMATH_CALUDE_inverse_of_A_l2232_223295

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2232_223295


namespace NUMINAMATH_CALUDE_rectangle_count_l2232_223276

/-- The region defined by y = 2x, y = -2, and x = 10 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ 2 * p.1 ∧ p.2 ≥ -2 ∧ p.1 ≤ 10}

/-- A 2x1 rectangle aligned parallel to the axes -/
def Rectangle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- A function that counts the number of integer-coordinate 2x1 rectangles that fit in the region -/
def countRectangles : ℕ :=
  sorry

theorem rectangle_count : countRectangles = 34 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l2232_223276


namespace NUMINAMATH_CALUDE_fraction_floor_value_l2232_223207

theorem fraction_floor_value : ⌊(1500^2 : ℝ) / ((500^2 : ℝ) - (496^2 : ℝ))⌋ = 564 := by
  sorry

end NUMINAMATH_CALUDE_fraction_floor_value_l2232_223207


namespace NUMINAMATH_CALUDE_initial_pencils_on_desk_l2232_223287

def pencils_in_drawer : ℕ := 43
def pencils_added : ℕ := 16
def total_pencils : ℕ := 78

theorem initial_pencils_on_desk :
  total_pencils = pencils_in_drawer + pencils_added + (total_pencils - pencils_in_drawer - pencils_added) ∧
  (total_pencils - pencils_in_drawer - pencils_added) = 19 :=
by sorry

end NUMINAMATH_CALUDE_initial_pencils_on_desk_l2232_223287


namespace NUMINAMATH_CALUDE_power_function_value_l2232_223220

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f (1/2) = 8 → f 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l2232_223220


namespace NUMINAMATH_CALUDE_prime_sum_product_l2232_223218

theorem prime_sum_product : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ 2 * p + 5 * q = 36 ∧ p * q = 26 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2232_223218


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2232_223264

theorem polynomial_division_quotient :
  ∀ (x : ℝ), x ≠ 1 →
  (x^6 + 8) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2232_223264


namespace NUMINAMATH_CALUDE_rotated_line_equation_l2232_223250

/-- Given a line l₁ with equation x - y - 3 = 0 rotated counterclockwise by 15° around
    the point (3,0) to obtain line l₂, the equation of l₂ is √3x - y - 3√3 = 0 --/
theorem rotated_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - y - 3 = 0
  let rotation_angle : ℝ := 15 * π / 180
  let rotation_center : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := fun x y ↦
    ∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧
    x - 3 = (x₀ - 3) * Real.cos rotation_angle - (y₀ - 0) * Real.sin rotation_angle ∧
    y - 0 = (x₀ - 3) * Real.sin rotation_angle + (y₀ - 0) * Real.cos rotation_angle
  l₂ x y ↔ Real.sqrt 3 * x - y - 3 * Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l2232_223250


namespace NUMINAMATH_CALUDE_lcm_852_1491_l2232_223245

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end NUMINAMATH_CALUDE_lcm_852_1491_l2232_223245


namespace NUMINAMATH_CALUDE_B_complete_work_in_40_days_l2232_223296

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 45

/-- The number of days A and B work together -/
def together_days : ℝ := 9

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 23

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 40

/-- Theorem stating that given the conditions, B can complete the work alone in 40 days -/
theorem B_complete_work_in_40_days :
  (together_days * (1 / A_days + 1 / B_days)) + (B_alone_days * (1 / B_days)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_B_complete_work_in_40_days_l2232_223296


namespace NUMINAMATH_CALUDE_function_properties_l2232_223229

def f (abc : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - abc

theorem function_properties (a b c abc : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : f abc a = 0) (h4 : f abc b = 0) (h5 : f abc c = 0) :
  (f abc 0) * (f abc 1) < 0 ∧ (f abc 0) * (f abc 3) > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2232_223229


namespace NUMINAMATH_CALUDE_die_roll_probability_l2232_223292

def roll_die : ℕ := 6
def num_trials : ℕ := 6
def min_success : ℕ := 5
def success_probability : ℚ := 1/3

theorem die_roll_probability : 
  (success_probability ^ num_trials) + 
  (Nat.choose num_trials min_success * success_probability ^ min_success * (1 - success_probability) ^ (num_trials - min_success)) = 13/729 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2232_223292


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_l2232_223263

/-- The number of times Will rode the Ferris wheel during the day -/
def daytime_rides : ℕ := 7

/-- The number of times Will rode the Ferris wheel at night -/
def nighttime_rides : ℕ := 6

/-- The total number of times Will rode the Ferris wheel -/
def total_rides : ℕ := daytime_rides + nighttime_rides

theorem ferris_wheel_rides : total_rides = 13 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_l2232_223263


namespace NUMINAMATH_CALUDE_prime_sequence_l2232_223240

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A + 14) ∧ 
  Nat.Prime (A + 18) ∧ 
  Nat.Prime (A + 32) ∧ 
  Nat.Prime (A + 36) → 
  A = 5 :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l2232_223240


namespace NUMINAMATH_CALUDE_acute_angle_condition_l2232_223223

/-- Given vectors a and b in ℝ², prove that x > -3 is a necessary but not sufficient condition
    for the angle between a and b to be acute -/
theorem acute_angle_condition (a b : ℝ × ℝ) (x : ℝ) 
    (ha : a = (2, 3)) (hb : b = (x, 2)) : 
    (∃ (y : ℝ), y > -3 ∧ y ≠ x ∧ 
      ((a.1 * b.1 + a.2 * b.2 > 0) ∧ 
       ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2))) ∧
    (x > -3 → 
      (a.1 * b.1 + a.2 * b.2 > 0) ∧ 
      ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2)) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_condition_l2232_223223


namespace NUMINAMATH_CALUDE_zach_score_l2232_223252

/-- Given that Ben scored 21 points in a football game and Zach scored 21 more points than Ben,
    prove that Zach scored 42 points. -/
theorem zach_score (ben_score : ℕ) (zach_ben_diff : ℕ) 
  (h1 : ben_score = 21)
  (h2 : zach_ben_diff = 21) :
  ben_score + zach_ben_diff = 42 := by
  sorry

end NUMINAMATH_CALUDE_zach_score_l2232_223252


namespace NUMINAMATH_CALUDE_circle_parabola_tangency_height_difference_l2232_223202

/-- Given a parabola y = 4x^2 and a circle tangent to it at two points,
    the height difference between the circle's center and the points of tangency is 1/8 -/
theorem circle_parabola_tangency_height_difference :
  ∀ (a b r : ℝ),
  (∀ x y : ℝ, y = 4 * x^2 → x^2 + (y - b)^2 = r^2) →  -- Circle equation
  (a^2 + (4 * a^2 - b)^2 = r^2) →                     -- Tangency condition at (a, 4a^2)
  ((-a)^2 + (4 * (-a)^2 - b)^2 = r^2) →               -- Tangency condition at (-a, 4a^2)
  b - 4 * a^2 = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_tangency_height_difference_l2232_223202


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_13792_l2232_223258

/-- Represents the sales data for a grocery shop over 6 months -/
structure SalesData where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the fourth month given the sales data and average -/
def fourthMonthSale (data : SalesData) (average : ℕ) : ℕ :=
  6 * average - (data.month1 + data.month2 + data.month3 + data.month5 + data.month6)

/-- Theorem stating that the fourth month's sale is 13792 given the conditions -/
theorem fourth_month_sale_is_13792 :
  let data : SalesData := {
    month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,  -- Unknown, to be calculated
    month5 := 6562,
    month6 := 4791
  }
  let average := 6500
  fourthMonthSale data average = 13792 := by
  sorry

#eval fourthMonthSale
  { month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,
    month5 := 6562,
    month6 := 4791 }
  6500

end NUMINAMATH_CALUDE_fourth_month_sale_is_13792_l2232_223258


namespace NUMINAMATH_CALUDE_janet_time_saved_l2232_223225

/-- The number of minutes Janet spends looking for keys daily -/
def keys_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding keys daily -/
def complain_time : ℕ := 3

/-- The number of minutes Janet spends searching for phone daily -/
def phone_time : ℕ := 5

/-- The number of minutes Janet spends looking for wallet daily -/
def wallet_time : ℕ := 4

/-- The number of minutes Janet spends trying to remember sunglasses location daily -/
def sunglasses_time : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Janet will save 154 minutes per week by stopping all these activities -/
theorem janet_time_saved :
  (keys_time + complain_time + phone_time + wallet_time + sunglasses_time) * days_in_week = 154 := by
  sorry

end NUMINAMATH_CALUDE_janet_time_saved_l2232_223225


namespace NUMINAMATH_CALUDE_shortcut_rectangle_ratio_l2232_223209

/-- A rectangle where the diagonal shortcut saves 1/3 of the longer side -/
structure ShortcutRectangle where
  x : ℝ  -- shorter side
  y : ℝ  -- longer side
  x_pos : 0 < x
  y_pos : 0 < y
  x_lt_y : x < y
  shortcut_saves : x + y - Real.sqrt (x^2 + y^2) = (1/3) * y

theorem shortcut_rectangle_ratio (r : ShortcutRectangle) : r.x / r.y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_shortcut_rectangle_ratio_l2232_223209


namespace NUMINAMATH_CALUDE_johns_spending_l2232_223237

/-- Given John's allowance B, prove that he spends 4/13 of B on movie ticket and soda combined -/
theorem johns_spending (B : ℝ) (t d : ℝ) 
  (ht : t = 0.25 * (B - d)) 
  (hd : d = 0.1 * (B - t)) : 
  t + d = (4 / 13) * B := by
sorry

end NUMINAMATH_CALUDE_johns_spending_l2232_223237


namespace NUMINAMATH_CALUDE_tax_rate_65_percent_l2232_223271

/-- Given a tax rate as a percentage, calculate the equivalent dollar amount per $100.00 -/
def tax_rate_to_dollars (percent : ℝ) : ℝ :=
  percent

theorem tax_rate_65_percent : tax_rate_to_dollars 65 = 65 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_65_percent_l2232_223271


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l2232_223266

/-- The expected value of flipping a set of coins -/
def expected_value (coin_values : List ℚ) : ℚ :=
  (coin_values.map (· / 2)).sum

/-- Theorem: The expected value of flipping a penny, nickel, dime, quarter, and half-dollar is 45.5 cents -/
theorem coin_flip_expected_value :
  expected_value [1, 5, 10, 25, 50] = 91/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l2232_223266


namespace NUMINAMATH_CALUDE_rational_inequality_solution_set_l2232_223277

theorem rational_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_set_l2232_223277


namespace NUMINAMATH_CALUDE_tuesday_steps_l2232_223213

/-- The number of steps Toby walked on each day of the week --/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Theorem stating that given the conditions, Toby walked 8300 steps on Tuesday --/
theorem tuesday_steps (w : WeekSteps) : 
  w.sunday = 9400 ∧ 
  w.monday = 9100 ∧ 
  (w.wednesday = 9200 ∨ w.thursday = 9200) ∧
  (w.wednesday = 8900 ∨ w.thursday = 8900) ∧
  w.friday + w.saturday = 18100 ∧
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday = 63000 
  → w.tuesday = 8300 := by
  sorry

#check tuesday_steps

end NUMINAMATH_CALUDE_tuesday_steps_l2232_223213


namespace NUMINAMATH_CALUDE_puppies_sold_is_24_l2232_223251

/-- Represents the pet store scenario --/
structure PetStore where
  initial_puppies : ℕ
  puppies_per_cage : ℕ
  cages_used : ℕ

/-- Calculates the number of puppies sold --/
def puppies_sold (store : PetStore) : ℕ :=
  store.initial_puppies - (store.puppies_per_cage * store.cages_used)

/-- Theorem stating that 24 puppies were sold --/
theorem puppies_sold_is_24 :
  ∃ (store : PetStore),
    store.initial_puppies = 56 ∧
    store.puppies_per_cage = 4 ∧
    store.cages_used = 8 ∧
    puppies_sold store = 24 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_is_24_l2232_223251


namespace NUMINAMATH_CALUDE_unique_prime_cube_sum_squares_l2232_223200

theorem unique_prime_cube_sum_squares :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p^3 = p^2 + q^2 + r^2 →
    p = 3 ∧ q = 3 ∧ r = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_cube_sum_squares_l2232_223200


namespace NUMINAMATH_CALUDE_range_of_a_l2232_223222

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ 4 * a * x) → |a| ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2232_223222


namespace NUMINAMATH_CALUDE_tracing_time_5x5_l2232_223262

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Time taken to trace a grid given a reference grid and its tracing time -/
def tracingTime (refGrid : Grid) (refTime : ℕ) (targetGrid : Grid) : ℕ :=
  (totalLength targetGrid * refTime) / (totalLength refGrid)

theorem tracing_time_5x5 :
  let refGrid : Grid := { width := 7, height := 3 }
  let targetGrid : Grid := { width := 5, height := 5 }
  tracingTime refGrid 26 targetGrid = 30 := by
  sorry

end NUMINAMATH_CALUDE_tracing_time_5x5_l2232_223262


namespace NUMINAMATH_CALUDE_function_positive_interval_implies_m_range_l2232_223283

theorem function_positive_interval_implies_m_range 
  (F : ℝ → ℝ) (m : ℝ) 
  (h_def : ∀ x, F x = -x^2 - m*x + 1) 
  (h_pos : ∀ x ∈ Set.Icc m (m+1), F x > 0) : 
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_function_positive_interval_implies_m_range_l2232_223283


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_negative_four_l2232_223247

theorem no_linear_term_implies_a_equals_negative_four (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + 4) * (x + a) = x^2 + b*x + c) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_negative_four_l2232_223247


namespace NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l2232_223256

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 6 * x) 
  (h2 : 3 * x - 4 * y < 2 * y - x) : 
  x < y ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l2232_223256


namespace NUMINAMATH_CALUDE_flour_for_dozen_cookies_l2232_223273

/-- Given information about cookie production and consumption, calculate the amount of flour needed for a dozen cookies -/
theorem flour_for_dozen_cookies 
  (bags : ℕ) 
  (weight_per_bag : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : bags = 4) 
  (h2 : weight_per_bag = 5) 
  (h3 : cookies_eaten = 15) 
  (h4 : cookies_left = 105) : 
  (12 : ℝ) * (bags * weight_per_bag : ℝ) / ((cookies_left + cookies_eaten) : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_dozen_cookies_l2232_223273


namespace NUMINAMATH_CALUDE_grade12_population_l2232_223293

/-- Represents the number of students in each grade -/
structure GradePopulation where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of students sampled from each grade -/
structure SampleSize where
  grade10 : Nat
  total : Nat

/-- Check if the sampling is proportional to the population -/
def isProportionalSampling (pop : GradePopulation) (sample : SampleSize) : Prop :=
  sample.grade10 * (pop.grade10 + pop.grade11 + pop.grade12) = 
  sample.total * pop.grade10

theorem grade12_population 
  (pop : GradePopulation)
  (sample : SampleSize)
  (h1 : pop.grade10 = 1000)
  (h2 : pop.grade11 = 1200)
  (h3 : sample.total = 66)
  (h4 : sample.grade10 = 20)
  (h5 : isProportionalSampling pop sample) :
  pop.grade12 = 1100 := by
  sorry

#check grade12_population

end NUMINAMATH_CALUDE_grade12_population_l2232_223293


namespace NUMINAMATH_CALUDE_peach_tree_average_production_l2232_223243

-- Define the number of apple trees
def num_apple_trees : ℕ := 30

-- Define the production of each apple tree in kg
def apple_production : ℕ := 150

-- Define the number of peach trees
def num_peach_trees : ℕ := 45

-- Define the total mass of fruit harvested in kg
def total_harvest : ℕ := 7425

-- Theorem to prove
theorem peach_tree_average_production :
  (total_harvest - num_apple_trees * apple_production) / num_peach_trees = 65 := by
  sorry

end NUMINAMATH_CALUDE_peach_tree_average_production_l2232_223243


namespace NUMINAMATH_CALUDE_g_has_two_zeros_l2232_223226

noncomputable def f (x : ℝ) : ℝ := (x - Real.sin x) / Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - 1 / (2 * Real.exp 2)

theorem g_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧
  ∀ (x : ℝ), g x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_g_has_two_zeros_l2232_223226


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2232_223280

/-- The total number of students at the end of the year in fourth grade -/
def total_students (initial : ℝ) (added : ℝ) (new : ℝ) : ℝ :=
  initial + added + new

/-- Theorem: The total number of students at the end of the year is 56.0 -/
theorem fourth_grade_students :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2232_223280


namespace NUMINAMATH_CALUDE_expression_simplification_l2232_223224

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2232_223224


namespace NUMINAMATH_CALUDE_printer_depreciation_l2232_223248

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else (resale_value (n - 1)) * (1 - subsequent_depreciation)

theorem printer_depreciation :
  resale_value 4 < target_value ∧
  ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end NUMINAMATH_CALUDE_printer_depreciation_l2232_223248


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l2232_223257

theorem min_perimeter_rectangle (w l : ℝ) (h1 : w > 0) (h2 : l > 0) (h3 : l = 2 * w) (h4 : w * l ≥ 500) :
  2 * w + 2 * l ≥ 30 * Real.sqrt 10 ∧ 
  (2 * w + 2 * l = 30 * Real.sqrt 10 → w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l2232_223257


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2232_223215

def n : ℕ := 240360

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id) = 62 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2232_223215


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2232_223278

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 90 ∧
           (∀ n : ℕ, is_five_digit n ∧ digit_product n = 90 → n ≤ M) ∧
           digit_sum M = 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2232_223278


namespace NUMINAMATH_CALUDE_quadratic_completion_l2232_223206

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 10*y + 47 = (y + a)^2 + k ∧ k = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2232_223206


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2232_223254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 2) - x^2

theorem function_inequality_implies_a_bound :
  ∀ a : ℝ,
  (∀ p q : ℝ, 0 < q ∧ q < p ∧ p < 1 →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
  a ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2232_223254


namespace NUMINAMATH_CALUDE_length_eb_is_two_l2232_223234

/-- An equilateral triangle with points on its sides -/
structure TriangleWithPoints where
  -- The side length of the equilateral triangle
  side : ℝ
  -- Lengths of segments
  ad : ℝ
  de : ℝ
  ef : ℝ
  fa : ℝ
  -- Conditions
  equilateral : side > 0
  d_on_ab : ad ≤ side
  e_on_bc : de ≤ side
  f_on_ca : fa ≤ side
  ad_value : ad = 4
  de_value : de = 8
  ef_value : ef = 10
  fa_value : fa = 6

/-- The length of segment EB in the triangle -/
def length_eb (t : TriangleWithPoints) : ℝ := 2

/-- Theorem: The length of EB is 2 -/
theorem length_eb_is_two (t : TriangleWithPoints) : length_eb t = 2 := by
  sorry

end NUMINAMATH_CALUDE_length_eb_is_two_l2232_223234


namespace NUMINAMATH_CALUDE_solve_for_y_l2232_223268

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2232_223268


namespace NUMINAMATH_CALUDE_fraction_not_on_time_l2232_223232

/-- Represents the attendees at the monthly meeting -/
structure Attendees where
  total : ℕ
  males : ℕ
  females : ℕ
  malesOnTime : ℕ
  femalesOnTime : ℕ

/-- The conditions of the problem -/
def meetingConditions (a : Attendees) : Prop :=
  a.males = (2 * a.total) / 3 ∧
  a.females = a.total - a.males ∧
  a.malesOnTime = (3 * a.males) / 4 ∧
  a.femalesOnTime = (5 * a.females) / 6

/-- The theorem to be proved -/
theorem fraction_not_on_time (a : Attendees) 
  (h : meetingConditions a) : 
  (a.total - (a.malesOnTime + a.femalesOnTime)) / a.total = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_not_on_time_l2232_223232


namespace NUMINAMATH_CALUDE_fraction_division_l2232_223204

theorem fraction_division : (4/9) / (5/8) = 32/45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l2232_223204


namespace NUMINAMATH_CALUDE_cyclic_trio_exists_l2232_223272

/-- Represents the result of a match between two players -/
inductive MatchResult
| Win
| Loss

/-- A tournament with a fixed number of players -/
structure Tournament where
  numPlayers : Nat
  results : Fin numPlayers → Fin numPlayers → MatchResult

/-- Predicate to check if player i defeated player j -/
def defeated (t : Tournament) (i j : Fin t.numPlayers) : Prop :=
  t.results i j = MatchResult.Win

theorem cyclic_trio_exists (t : Tournament) 
  (h1 : t.numPlayers = 12)
  (h2 : ∀ i j : Fin t.numPlayers, i ≠ j → (defeated t i j ∨ defeated t j i))
  (h3 : ∀ i : Fin t.numPlayers, ∃ j : Fin t.numPlayers, defeated t i j) :
  ∃ a b c : Fin t.numPlayers, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    defeated t a b ∧ defeated t b c ∧ defeated t c a :=
sorry

end NUMINAMATH_CALUDE_cyclic_trio_exists_l2232_223272


namespace NUMINAMATH_CALUDE_fish_count_approximation_l2232_223286

/-- Approximates the total number of fish in a pond based on a tagging and recapture experiment. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- Theorem stating that under the given conditions, the approximate number of fish in the pond is 313. -/
theorem fish_count_approximation :
  let initial_tagged := 50
  let second_catch := 50
  let tagged_in_second := 8
  approximate_fish_count initial_tagged second_catch tagged_in_second = 313 :=
by
  sorry

#eval approximate_fish_count 50 50 8

end NUMINAMATH_CALUDE_fish_count_approximation_l2232_223286


namespace NUMINAMATH_CALUDE_faye_coloring_books_l2232_223288

theorem faye_coloring_books (initial : ℝ) (given_away : ℝ) (additional_percentage : ℝ) :
  initial = 52.5 →
  given_away = 38.2 →
  additional_percentage = 25 →
  let remainder : ℝ := initial - given_away
  let additional_given : ℝ := (additional_percentage / 100) * remainder
  initial - given_away - additional_given = 10.725 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l2232_223288


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2232_223270

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose 3 2 * (Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) +
   Nat.choose 3 1 * Nat.choose red_marbles 2) *
  Nat.choose (total_marbles - (red_marbles + green_marbles + blue_marbles)) (marbles_to_choose - special_marbles_to_choose) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2232_223270


namespace NUMINAMATH_CALUDE_workshop_attendance_prove_workshop_attendance_l2232_223217

theorem workshop_attendance : ℕ → Prop :=
  fun total_scientists =>
    ∃ (wolf_laureates nobel_laureates wolf_and_nobel non_wolf_non_nobel : ℕ),
      wolf_laureates = 31 ∧
      wolf_and_nobel = 14 ∧
      nobel_laureates = 25 ∧
      nobel_laureates - wolf_and_nobel = non_wolf_non_nobel + 3 ∧
      total_scientists = wolf_laureates + (nobel_laureates - wolf_and_nobel) + non_wolf_non_nobel ∧
      total_scientists = 50

theorem prove_workshop_attendance : workshop_attendance 50 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_prove_workshop_attendance_l2232_223217


namespace NUMINAMATH_CALUDE_ttakji_square_arrangement_l2232_223255

/-- The number of ttakjis on one side of the large square -/
def n : ℕ := 61

/-- The number of ttakjis on the perimeter of the large square -/
def perimeter_ttakjis : ℕ := 240

theorem ttakji_square_arrangement :
  (4 * n - 4 = perimeter_ttakjis) ∧ (n^2 = 3721) := by sorry

end NUMINAMATH_CALUDE_ttakji_square_arrangement_l2232_223255


namespace NUMINAMATH_CALUDE_basketball_not_table_tennis_count_l2232_223279

/-- Represents the class of students and their sports preferences -/
structure ClassSports where
  total : ℕ
  basketball : ℕ
  tableTennis : ℕ
  neither : ℕ

/-- The number of students who like basketball but not table tennis -/
def basketballNotTableTennis (c : ClassSports) : ℕ :=
  c.basketball - (c.total - c.tableTennis - c.neither)

/-- Theorem stating the number of students who like basketball but not table tennis -/
theorem basketball_not_table_tennis_count (c : ClassSports) 
  (h1 : c.total = 30)
  (h2 : c.basketball = 15)
  (h3 : c.tableTennis = 10)
  (h4 : c.neither = 8) :
  basketballNotTableTennis c = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_not_table_tennis_count_l2232_223279


namespace NUMINAMATH_CALUDE_tan_squared_sum_lower_bound_l2232_223269

theorem tan_squared_sum_lower_bound 
  (α β γ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < π / 2)
  (h5 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / (9 ^ (1/3) - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_lower_bound_l2232_223269


namespace NUMINAMATH_CALUDE_solve_system_of_equations_solve_system_of_inequalities_l2232_223282

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x - y = 3) ∧ (x + y = 6)

theorem solve_system_of_equations :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 3 ∧ y = 3 :=
sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2)

theorem solve_system_of_inequalities :
  ∀ x : ℝ, system_of_inequalities x ↔ -2 < x ∧ x < -1 :=
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_solve_system_of_inequalities_l2232_223282


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2232_223219

theorem trigonometric_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  ((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = 7 / 17) ∧
  (Real.sin α ^ 2 + Real.sin (2 * α) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2232_223219


namespace NUMINAMATH_CALUDE_least_possible_xy_l2232_223285

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  ∃ (min_xy : ℕ), (x * y : ℕ) ≥ min_xy ∧ 
  (∃ (x' y' : ℕ+), (1 : ℚ) / x' + (1 : ℚ) / (3 * y') = (1 : ℚ) / 8 ∧ (x' * y' : ℕ) = min_xy) :=
sorry

end NUMINAMATH_CALUDE_least_possible_xy_l2232_223285


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2232_223284

theorem not_sufficient_not_necessary (p q : Prop) : 
  (¬(p ∧ q → p ∨ q)) ∧ (¬(p ∨ q → ¬(p ∧ q))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2232_223284


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2232_223246

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2232_223246


namespace NUMINAMATH_CALUDE_mike_spent_500_on_plants_l2232_223228

/-- The amount Mike spent on plants for himself -/
def mike_spent_on_plants : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_rose_bushes, rose_bush_price, friend_rose_bushes, num_aloes, aloe_price =>
    let self_rose_bushes := total_rose_bushes - friend_rose_bushes
    let rose_bush_cost := self_rose_bushes * rose_bush_price
    let aloe_cost := num_aloes * aloe_price
    rose_bush_cost + aloe_cost

theorem mike_spent_500_on_plants :
  mike_spent_on_plants 6 75 2 2 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_mike_spent_500_on_plants_l2232_223228


namespace NUMINAMATH_CALUDE_janet_stuffies_l2232_223239

theorem janet_stuffies (total : ℕ) (kept_fraction : ℚ) (given_fraction : ℚ) : 
  total = 60 →
  kept_fraction = 1/3 →
  given_fraction = 1/4 →
  (total - kept_fraction * total) * given_fraction = 10 := by
sorry

end NUMINAMATH_CALUDE_janet_stuffies_l2232_223239


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_4_l2232_223221

theorem smallest_lcm_with_gcd_4 :
  ∃ (m n : ℕ),
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 4 ∧
    Nat.lcm m n = 252912 ∧
    ∀ (a b : ℕ),
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 4 →
      Nat.lcm a b ≥ 252912 :=
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_4_l2232_223221


namespace NUMINAMATH_CALUDE_order_combination_savings_l2232_223216

/-- Calculates the discount percentage based on the number of photocopies -/
def discount_percentage (n : ℕ) : ℚ :=
  if n ≤ 50 then 0
  else if n ≤ 100 then 1/10
  else if n ≤ 200 then 1/4
  else 7/20

/-- Calculates the discounted cost for a given number of photocopies -/
def discounted_cost (n : ℕ) : ℚ :=
  let base_cost : ℚ := (n : ℚ) * 2/100
  base_cost * (1 - discount_percentage n)

/-- Theorem: The savings from combining orders is $0.225 -/
theorem order_combination_savings :
  discounted_cost 75 + discounted_cost 105 - discounted_cost 180 = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_order_combination_savings_l2232_223216
