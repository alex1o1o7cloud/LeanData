import Mathlib

namespace NUMINAMATH_CALUDE_kangaroo_problem_l3552_355200

/-- Represents the number of exchanges required to sort kangaroos -/
def kangaroo_exchanges (total : ℕ) (right_facing : ℕ) (left_facing : ℕ) : ℕ :=
  (right_facing.min 3) * left_facing + (right_facing - 3).max 0 * (left_facing - 2).max 0

/-- Theorem stating that for 10 kangaroos with 6 facing right and 4 facing left, 
    the number of exchanges is 18 -/
theorem kangaroo_problem : 
  kangaroo_exchanges 10 6 4 = 18 := by sorry

end NUMINAMATH_CALUDE_kangaroo_problem_l3552_355200


namespace NUMINAMATH_CALUDE_q_of_4_equals_6_l3552_355228

-- Define the function q
def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3*|x - 3|^(1/5) + 2

-- Theorem stating that q(4) = 6
theorem q_of_4_equals_6 : q 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_q_of_4_equals_6_l3552_355228


namespace NUMINAMATH_CALUDE_cos_double_angle_given_tan_l3552_355236

theorem cos_double_angle_given_tan (α : Real) (h : Real.tan α = 3) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_given_tan_l3552_355236


namespace NUMINAMATH_CALUDE_age_difference_l3552_355223

/-- The difference in total ages of (A,B) and (B,C) given C is 10 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 10) : A + B - (B + C) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3552_355223


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l3552_355257

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    where the intersection of lines AB and CF is at (3a, 16),
    prove that a = 5 and b = 4 -/
theorem ellipse_dimensions (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, a^2 = b^2 + c^2) →
  (∃ x y : ℝ, x = 3*a ∧ y = 16 ∧ x/(-a) + y/b = 1 ∧ x/c + y/(-b) = 1) →
  a = 5 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l3552_355257


namespace NUMINAMATH_CALUDE_remainder_of_123456789012_mod_252_l3552_355248

theorem remainder_of_123456789012_mod_252 : 123456789012 % 252 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_123456789012_mod_252_l3552_355248


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l3552_355233

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 136080 → n + (n + 1) + (n + 2) = 144 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l3552_355233


namespace NUMINAMATH_CALUDE_equation_solutions_l3552_355255

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = -9 ∧ x₁^2 + 12*x₁ + 27 = 0 ∧ x₂^2 + 12*x₂ + 27 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-5 + Real.sqrt 10) / 3 ∧ x₂ = (-5 - Real.sqrt 10) / 3 ∧
    3*x₁^2 + 10*x₁ + 5 = 0 ∧ 3*x₂^2 + 10*x₂ + 5 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4/3 ∧ x₂ = 2/3 ∧ (3*x₁ + 1)^2 - 9 = 0 ∧ (3*x₂ + 1)^2 - 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3552_355255


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3552_355237

/-- Given a geometric sequence {a_n} with a₁ = 1 and common ratio q = 3,
    if the sum of the first t terms S_t = 364, then the t-th term a_t = 243 -/
theorem geometric_sequence_problem (t : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence with q = 3
  a 1 = 1 →                    -- a₁ = 1
  (∀ n, S n = (a 1) * (1 - 3^n) / (1 - 3)) →  -- sum formula for geometric sequence
  S t = 364 →                  -- S_t = 364
  a t = 243 := by              -- a_t = 243
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3552_355237


namespace NUMINAMATH_CALUDE_rocky_path_trail_length_l3552_355271

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
def rocky_path_trail (a b c d e : ℝ) : Prop :=
  a + b = 24 ∧
  b + c = 28 ∧
  c + d + e = 36 ∧
  a + c = 28

theorem rocky_path_trail_length :
  ∀ a b c d e : ℝ, rocky_path_trail a b c d e → a + b + c + d + e = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rocky_path_trail_length_l3552_355271


namespace NUMINAMATH_CALUDE_max_dimes_grace_l3552_355232

/-- The value of a dime in cents -/
def dime_value : ℚ := 10

/-- The value of a penny in cents -/
def penny_value : ℚ := 1

/-- The total amount Grace has in cents -/
def total_amount : ℚ := 480

theorem max_dimes_grace : 
  ∀ d : ℕ, d * (dime_value + penny_value) ≤ total_amount → d ≤ 43 :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_grace_l3552_355232


namespace NUMINAMATH_CALUDE_community_center_chairs_l3552_355241

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Calculates the number of chairs needed given the total people and people per chair -/
def calculateChairs (totalPeople : ℕ) (peoplePerChair : ℕ) : ℚ :=
  (totalPeople : ℚ) / peoplePerChair

theorem community_center_chairs :
  let seatingCapacity := base6ToBase10 2 3 1
  let peoplePerChair := 3
  calculateChairs seatingCapacity peoplePerChair = 30.33 := by sorry

end NUMINAMATH_CALUDE_community_center_chairs_l3552_355241


namespace NUMINAMATH_CALUDE_select_students_l3552_355244

theorem select_students (n_boys : ℕ) (n_girls : ℕ) (n_select : ℕ) : 
  n_boys = 4 → n_girls = 2 → n_select = 4 →
  (Nat.choose n_boys (n_select - 1) * Nat.choose n_girls 1 + 
   Nat.choose n_boys (n_select - 2) * Nat.choose n_girls 2) = 14 := by
sorry

end NUMINAMATH_CALUDE_select_students_l3552_355244


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3552_355221

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 3}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3552_355221


namespace NUMINAMATH_CALUDE_quadratic_equation_identification_l3552_355279

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the equations from the problem
def eq_A (x y : ℝ) : Prop := 3 * x + y = 2
def eq_B (x : ℝ) : Prop := x = 2 * x^3 - 3
def eq_C (x : ℝ) : Prop := x^2 - 5 = 0
def eq_D (x : ℝ) : Prop := 2 * x + 1/x = 3

-- Theorem stating that eq_C is quadratic while others are not
theorem quadratic_equation_identification :
  (∃ f : ℝ → ℝ, (∀ x, f x = 0 ↔ eq_C x) ∧ is_quadratic_equation f) ∧
  (¬∃ f : ℝ → ℝ, (∀ x y, f x = 0 ↔ eq_A x y) ∧ is_quadratic_equation f) ∧
  (¬∃ f : ℝ → ℝ, (∀ x, f x = 0 ↔ eq_B x) ∧ is_quadratic_equation f) ∧
  (¬∃ f : ℝ → ℝ, (∀ x, f x = 0 ↔ eq_D x) ∧ is_quadratic_equation f) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_identification_l3552_355279


namespace NUMINAMATH_CALUDE_difference_exists_l3552_355205

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end NUMINAMATH_CALUDE_difference_exists_l3552_355205


namespace NUMINAMATH_CALUDE_solve_equation_l3552_355220

theorem solve_equation (x : ℝ) : 3 * x = (26 - x) + 14 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3552_355220


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_150_l3552_355226

/-- Given a train passing a platform and a man, calculate the platform length -/
theorem platform_length (train_speed : Real) (platform_time : Real) (man_time : Real) : Real :=
  let train_speed_ms := train_speed * 1000 / 3600
  let train_length := train_speed_ms * man_time
  let platform_length := train_speed_ms * platform_time - train_length
  platform_length

/-- Prove that the platform length is 150 meters given the specified conditions -/
theorem platform_length_is_150 :
  platform_length 54 30 20 = 150 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_150_l3552_355226


namespace NUMINAMATH_CALUDE_golden_section_length_l3552_355285

/-- Given a segment AB of length 2 with C as its golden section point (AC > BC),
    the length of AC is √5 - 1 -/
theorem golden_section_length (A B C : ℝ) : 
  (B - A = 2) →
  (C - A) / (B - C) = (1 + Real.sqrt 5) / 2 →
  C - A > B - C →
  C - A = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_golden_section_length_l3552_355285


namespace NUMINAMATH_CALUDE_secondary_spermatocyte_may_have_two_y_l3552_355210

-- Define the different stages of cell division
inductive CellDivisionStage
  | PrimarySpermatocyte
  | SecondarySpermatocyte
  | SpermatogoniumMitosis
  | SpermatogoniumMeiosis

-- Define the possible Y chromosome counts
inductive YChromosomeCount
  | Zero
  | One
  | Two

-- Define a function that returns the possible Y chromosome counts for each stage
def possibleYChromosomeCounts (stage : CellDivisionStage) : Set YChromosomeCount :=
  match stage with
  | CellDivisionStage.PrimarySpermatocyte => {YChromosomeCount.One}
  | CellDivisionStage.SecondarySpermatocyte => {YChromosomeCount.Zero, YChromosomeCount.One, YChromosomeCount.Two}
  | CellDivisionStage.SpermatogoniumMitosis => {YChromosomeCount.One}
  | CellDivisionStage.SpermatogoniumMeiosis => {YChromosomeCount.One}

-- Theorem stating that secondary spermatocytes may contain two Y chromosomes
theorem secondary_spermatocyte_may_have_two_y :
  YChromosomeCount.Two ∈ possibleYChromosomeCounts CellDivisionStage.SecondarySpermatocyte :=
by sorry

end NUMINAMATH_CALUDE_secondary_spermatocyte_may_have_two_y_l3552_355210


namespace NUMINAMATH_CALUDE_shelly_thread_needed_l3552_355249

def thread_per_keychain : ℕ := 12
def friends_in_classes : ℕ := 6
def friends_in_clubs : ℕ := friends_in_classes / 2

theorem shelly_thread_needed : 
  (friends_in_classes + friends_in_clubs) * thread_per_keychain = 108 := by
  sorry

end NUMINAMATH_CALUDE_shelly_thread_needed_l3552_355249


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3552_355295

theorem age_ratio_problem (p q : ℕ) 
  (h1 : p - 12 = (q - 12) / 2)
  (h2 : p + q = 42) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * q = b * p ∧ a = 3 ∧ b = 4 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3552_355295


namespace NUMINAMATH_CALUDE_trajectory_is_circle_l3552_355206

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define the points
variable (F₁ F₂ P Q : E)

-- Define the ellipse
def is_on_ellipse (P : E) (F₁ F₂ : E) (a : ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * a

-- Define the condition for Q
def extends_to_Q (P Q : E) (F₁ F₂ : E) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

-- Theorem statement
theorem trajectory_is_circle 
  (a : ℝ) 
  (h_ellipse : is_on_ellipse P F₁ F₂ a) 
  (h_extends : extends_to_Q P Q F₁ F₂) :
  ∃ (center : E) (radius : ℝ), dist Q center = radius :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_circle_l3552_355206


namespace NUMINAMATH_CALUDE_min_value_ab_l3552_355238

theorem min_value_ab (a b : ℝ) (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l3552_355238


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3552_355230

theorem remainder_divisibility (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3552_355230


namespace NUMINAMATH_CALUDE_inequality_proof_l3552_355263

theorem inequality_proof (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3552_355263


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3552_355203

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3552_355203


namespace NUMINAMATH_CALUDE_problem_statement_l3552_355246

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  2 * x^2 + 3 * y^2 + 4 * z^2 = 12 * b / a + 8 * c / b + 6 * a / c := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3552_355246


namespace NUMINAMATH_CALUDE_solution_exists_for_all_primes_l3552_355286

theorem solution_exists_for_all_primes (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_for_all_primes_l3552_355286


namespace NUMINAMATH_CALUDE_boat_distance_main_theorem_l3552_355219

/-- The distance between two boats given specific angles and fort height -/
theorem boat_distance (fort_height : ℝ) (angle1 angle2 base_angle : ℝ) : ℝ :=
  let boat_distance := 30
  by
    -- Assuming fort_height = 30, angle1 = 45°, angle2 = 30°, base_angle = 30°
    sorry

/-- Main theorem stating the distance between the boats is 30 meters -/
theorem main_theorem : boat_distance 30 (45 * π / 180) (30 * π / 180) (30 * π / 180) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_distance_main_theorem_l3552_355219


namespace NUMINAMATH_CALUDE_university_box_cost_l3552_355216

theorem university_box_cost (box_length box_width box_height : ℝ)
  (box_cost : ℝ) (total_volume : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧
  box_cost = 0.5 ∧ total_volume = 2160000 →
  (⌈total_volume / (box_length * box_width * box_height)⌉ : ℝ) * box_cost = 225 := by
  sorry

end NUMINAMATH_CALUDE_university_box_cost_l3552_355216


namespace NUMINAMATH_CALUDE_joan_book_sale_l3552_355288

/-- Given that Joan initially gathered 33 books and found 26 more,
    prove that the total number of books she has for sale is 59. -/
theorem joan_book_sale (initial_books : ℕ) (additional_books : ℕ) 
  (h1 : initial_books = 33) (h2 : additional_books = 26) : 
  initial_books + additional_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_joan_book_sale_l3552_355288


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3552_355229

theorem polynomial_simplification (x : ℝ) :
  (3 * x^4 + 2 * x^3 - 9 * x^2 + 4 * x - 5) + (-5 * x^4 - 3 * x^3 + x^2 - 4 * x + 7) =
  -2 * x^4 - x^3 - 8 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3552_355229


namespace NUMINAMATH_CALUDE_cheryls_mms_l3552_355207

/-- Cheryl's M&M's Problem -/
theorem cheryls_mms (initial : ℕ) (after_lunch : ℕ) (after_dinner : ℕ) (given_to_sister : ℕ) :
  initial = 25 →
  after_lunch = 7 →
  after_dinner = 5 →
  given_to_sister = initial - (after_lunch + after_dinner) →
  given_to_sister = 13 := by
sorry

end NUMINAMATH_CALUDE_cheryls_mms_l3552_355207


namespace NUMINAMATH_CALUDE_expression_simplification_l3552_355209

theorem expression_simplification (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a + 3) + 1 / (a^2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3552_355209


namespace NUMINAMATH_CALUDE_remaining_tanning_time_l3552_355297

/-- Calculates the remaining tanning time for the last two weeks of the month. -/
theorem remaining_tanning_time 
  (max_monthly_time : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (first_half_weeks : ℕ) 
  (h1 : max_monthly_time = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : first_half_weeks = 2) :
  max_monthly_time - (daily_time * days_per_week * first_half_weeks) = 80 :=
by
  sorry

#check remaining_tanning_time

end NUMINAMATH_CALUDE_remaining_tanning_time_l3552_355297


namespace NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l3552_355258

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5  -- First term
  let r : ℚ := 4/3  -- Common ratio
  let n : ℕ := 10  -- Term number we're looking for
  let a_n : ℚ := a * r^(n - 1)  -- Formula for nth term of geometric sequence
  a_n = 1310720/19683 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l3552_355258


namespace NUMINAMATH_CALUDE_negation_of_exists_exponential_nonpositive_l3552_355269

theorem negation_of_exists_exponential_nonpositive :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_exponential_nonpositive_l3552_355269


namespace NUMINAMATH_CALUDE_partnership_profit_l3552_355204

/-- Calculates the total profit of a partnership given the investments and one partner's profit share -/
theorem partnership_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 30000)
  (h2 : b_investment = 45000)
  (h3 : c_investment = 50000)
  (h4 : c_profit_share = 36000) :
  ∃ (total_profit : ℕ), total_profit = 90000 ∧
    total_profit * c_investment = (a_investment + b_investment + c_investment) * c_profit_share :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l3552_355204


namespace NUMINAMATH_CALUDE_min_triangle_area_l3552_355252

/-- Given a triangle ABC with side AB = 2 and 2/sin(A) + 1/tan(B) = 2√3, 
    its area is greater than or equal to 2√3/3 -/
theorem min_triangle_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.tan B ≠ 0)
  (h7 : 2 / Real.sin A + 1 / Real.tan B = 2 * Real.sqrt 3) :
  1 / 2 * 2 * Real.sin C ≥ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_min_triangle_area_l3552_355252


namespace NUMINAMATH_CALUDE_count_numbers_mod_three_eq_one_l3552_355259

theorem count_numbers_mod_three_eq_one (n : ℕ) : 
  (Finset.filter (fun x => x % 3 = 1) (Finset.range 50)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_mod_three_eq_one_l3552_355259


namespace NUMINAMATH_CALUDE_loss_recording_l3552_355245

/-- Represents the recording of a financial transaction -/
def record (amount : Int) : Int := amount

/-- A profit of $300 is recorded as $+300 -/
axiom profit_recording : record 300 = 300

/-- Theorem: If a profit of $300 is recorded as $+300, then a loss of $300 should be recorded as $-300 -/
theorem loss_recording : record (-300) = -300 := by
  sorry

end NUMINAMATH_CALUDE_loss_recording_l3552_355245


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3552_355272

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let special_number := 1 + 1 / n
  let regular_number := 1
  let sum := special_number + (n - 1) * regular_number
  sum / n = 1 + 1 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3552_355272


namespace NUMINAMATH_CALUDE_product_of_sum_one_equals_eight_l3552_355267

theorem product_of_sum_one_equals_eight 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hab : a * b + a + b = 3) 
  (hbc : b * c + b + c = 3) 
  (hac : a * c + a + c = 3) : 
  (a + 1) * (b + 1) * (c + 1) = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_one_equals_eight_l3552_355267


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l3552_355242

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 81) 
  (h2 : total_cost = 2088) : 
  (total_cost / (4 * Real.sqrt area)) = 58 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l3552_355242


namespace NUMINAMATH_CALUDE_expensive_candy_price_l3552_355217

/-- Proves that the price of the more expensive candy is $3 per pound -/
theorem expensive_candy_price
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_amount : ℝ)
  (cheap_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_amount = 16)
  (h4 : cheap_price = 2)
  : ∃ (expensive_price : ℝ),
    expensive_price * expensive_amount + cheap_price * (total_mixture - expensive_amount) =
    selling_price * total_mixture ∧ expensive_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_expensive_candy_price_l3552_355217


namespace NUMINAMATH_CALUDE_fraction_equality_l3552_355211

theorem fraction_equality (x : ℚ) : (4 + x) / (6 + x) = (2 + x) / (3 + x) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3552_355211


namespace NUMINAMATH_CALUDE_largest_power_of_five_l3552_355291

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials : ℕ := factorial 120 + factorial 121 + factorial 122

theorem largest_power_of_five (n : ℕ) : n ≤ 28 ↔ (5 ^ n ∣ sum_factorials) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_l3552_355291


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3552_355296

theorem ceiling_floor_difference : ⌈((15 / 8 : ℚ) ^ 2 * (-34 / 4 : ℚ))⌉ - ⌊(15 / 8 : ℚ) * ⌊-34 / 4⌋⌋ = -12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3552_355296


namespace NUMINAMATH_CALUDE_students_left_l3552_355292

/-- Given the initial numbers of boys and girls in a school, and the numbers of boys and girls who drop out, prove that the total number of students left is 17. -/
theorem students_left
  (initial_boys : ℕ)
  (initial_girls : ℕ)
  (boys_dropout : ℕ)
  (girls_dropout : ℕ)
  (h1 : initial_boys = 14)
  (h2 : initial_girls = 10)
  (h3 : boys_dropout = 4)
  (h4 : girls_dropout = 3) :
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_left_l3552_355292


namespace NUMINAMATH_CALUDE_intersection_implies_a_less_than_two_l3552_355294

def A : Set ℝ := {1}
def B (a : ℝ) : Set ℝ := {x | a - 2*x < 0}

theorem intersection_implies_a_less_than_two (a : ℝ) : 
  (A ∩ B a).Nonempty → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_less_than_two_l3552_355294


namespace NUMINAMATH_CALUDE_work_completion_l3552_355243

theorem work_completion (days_group1 : ℕ) (men_group2 : ℕ) (days_group2 : ℕ) :
  days_group1 = 18 →
  men_group2 = 27 →
  days_group2 = 24 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = men_group2 * days_group2 ∧ men_group1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3552_355243


namespace NUMINAMATH_CALUDE_smallest_a_for_parabola_l3552_355250

/-- The smallest possible value of a for a parabola with given conditions -/
theorem smallest_a_for_parabola (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + b + c = n) →
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y + 2 = a * (x - 3/4)^2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + b' + c' = n) ∧
    (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ y + 2 = a' * (x - 3/4)^2)) →
    a ≤ a') →
  a = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_parabola_l3552_355250


namespace NUMINAMATH_CALUDE_difference_of_31st_terms_l3552_355225

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem difference_of_31st_terms : 
  let C := arithmeticSequence 50 12
  let D := arithmeticSequence 50 (-8)
  |C 31 - D 31| = 600 := by sorry

end NUMINAMATH_CALUDE_difference_of_31st_terms_l3552_355225


namespace NUMINAMATH_CALUDE_tape_length_problem_l3552_355262

theorem tape_length_problem (original_length : ℝ) : 
  (original_length > 0) →
  (original_length * (1 - 1/5) * (1 - 3/4) = 1.5) →
  (original_length = 7.5) := by
sorry

end NUMINAMATH_CALUDE_tape_length_problem_l3552_355262


namespace NUMINAMATH_CALUDE_remaining_numbers_count_l3552_355254

theorem remaining_numbers_count (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 8 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 14 →
  (total - subset = 2 ∧ 
   (total * total_avg - subset * subset_avg) / (total - subset) = remaining_avg) :=
by sorry

end NUMINAMATH_CALUDE_remaining_numbers_count_l3552_355254


namespace NUMINAMATH_CALUDE_number_division_problem_l3552_355235

theorem number_division_problem : ∃ x : ℚ, x / 11 + 156 = 178 ∧ x = 242 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3552_355235


namespace NUMINAMATH_CALUDE_point_in_same_region_l3552_355213

/-- The line equation -/
def line_equation (x y : ℝ) : ℝ := 3*x + 2*y + 5

/-- Definition of being in the same region -/
def same_region (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line_equation x₁ y₁ > 0 ∧ line_equation x₂ y₂ > 0) ∨
  (line_equation x₁ y₁ < 0 ∧ line_equation x₂ y₂ < 0)

/-- Theorem stating that (-3,4) is in the same region as (0,0) -/
theorem point_in_same_region : same_region (-3) 4 0 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_same_region_l3552_355213


namespace NUMINAMATH_CALUDE_amoeba_population_l3552_355261

/-- The number of amoebas in the puddle after n days -/
def amoebas (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba population grows -/
def days : ℕ := 10

theorem amoeba_population : amoebas days = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_l3552_355261


namespace NUMINAMATH_CALUDE_subsets_without_consecutive_eq_fib_l3552_355212

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of subsets without consecutive elements -/
def subsets_without_consecutive (n : ℕ) : ℕ :=
  fib (n + 2)

/-- Theorem: The number of subsets of {1, 2, 3, ..., n} that do not contain
    two consecutive numbers is equal to the (n+2)th Fibonacci number -/
theorem subsets_without_consecutive_eq_fib (n : ℕ) :
  subsets_without_consecutive n = fib (n + 2) := by
  sorry


end NUMINAMATH_CALUDE_subsets_without_consecutive_eq_fib_l3552_355212


namespace NUMINAMATH_CALUDE_subtraction_rule_rational_l3552_355283

theorem subtraction_rule_rational (x : ℚ) : ∀ y : ℚ, y - x = y + (-x) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_rule_rational_l3552_355283


namespace NUMINAMATH_CALUDE_tom_deck_cost_l3552_355268

/-- The cost of Tom's deck of trading cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- Theorem: The cost of Tom's deck is $32 -/
theorem tom_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_tom_deck_cost_l3552_355268


namespace NUMINAMATH_CALUDE_edith_books_count_l3552_355282

theorem edith_books_count : ∀ (novels : ℕ) (writing_books : ℕ),
  novels = 80 →
  writing_books = 2 * novels →
  novels + writing_books = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_edith_books_count_l3552_355282


namespace NUMINAMATH_CALUDE_vector_distance_inequality_l3552_355264

noncomputable def max_T : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem vector_distance_inequality (a b : ℝ × ℝ) :
  (∀ m n : ℝ, let c := (m, 1 - m)
               let d := (n, 1 - n)
               (a.1 - c.1)^2 + (a.2 - c.2)^2 + (b.1 - d.1)^2 + (b.2 - d.2)^2 ≥ max_T^2) →
  (norm a = 1 ∧ norm b = 1 ∧ a.1 * b.1 + a.2 * b.2 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_vector_distance_inequality_l3552_355264


namespace NUMINAMATH_CALUDE_union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l3552_355215

-- Define the universe set
variable {U : Type}

-- Define sets A, B, C as subsets of U
variable (A B C : Set U)

-- Theorem 1
theorem union_empty_iff_both_empty :
  A ∪ B = ∅ ↔ A = ∅ ∧ B = ∅ := by sorry

-- Theorem 2
theorem union_eq_diff_iff_empty :
  A ∪ B = A \ B ↔ B = ∅ := by sorry

-- Theorem 3
theorem diff_eq_inter_iff_empty :
  A \ B = A ∩ B ↔ A = ∅ := by sorry

-- Additional theorems can be added similarly for the remaining equivalences

end NUMINAMATH_CALUDE_union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l3552_355215


namespace NUMINAMATH_CALUDE_peanuts_in_box_l3552_355234

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 4 peanuts and 2 more are added, the total is 6 -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l3552_355234


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_frac_l3552_355214

theorem ceiling_neg_sqrt_frac : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_frac_l3552_355214


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l3552_355240

/-- Given 5 consecutive points on a straight line, if certain conditions are met, 
    then the distance between the first two points is 5. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b) = 2 * (d - c) →  -- bc = 2cd
  (e - d) = 4 →            -- de = 4
  (c - a) = 11 →           -- ac = 11
  (e - a) = 18 →           -- ae = 18
  (b - a) = 5 :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l3552_355240


namespace NUMINAMATH_CALUDE_fathers_age_ratio_l3552_355260

theorem fathers_age_ratio (father_age ronit_age : ℕ) : 
  (father_age + 8 = (ronit_age + 8) * 5 / 2) →
  (father_age + 16 = (ronit_age + 16) * 2) →
  father_age = ronit_age * 4 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_ratio_l3552_355260


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3552_355231

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let M := let x_v := 3 * T / 2
            let y_v := -3 * a * T^2 / 4
            x_v + y_v
  (passes_through 0 0) ∧ 
  (passes_through (3 * T) 0) ∧
  (passes_through (3 * T + 1) 35) →
  ∀ m : ℝ, M ≤ m → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3552_355231


namespace NUMINAMATH_CALUDE_hash_property_l3552_355247

/-- Definition of operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 100 + 4 * b^2 + 8 * a * b

/-- Theorem stating the properties of the hash operation -/
theorem hash_property (a b : ℕ) : 
  hash a b = 100 ∧ a + b = 5 → hash a b = 100 + 4 * b^2 + 8 * a * b := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l3552_355247


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l3552_355280

theorem sqrt_expression_equals_one :
  1 + (Real.sqrt 2 - Real.sqrt 3) + |Real.sqrt 2 - Real.sqrt 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l3552_355280


namespace NUMINAMATH_CALUDE_history_paper_pages_l3552_355276

/-- Calculates the total number of pages in a paper given the number of days and pages per day -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days with 21 pages written per day has 63 pages in total -/
theorem history_paper_pages : total_pages 3 21 = 63 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3552_355276


namespace NUMINAMATH_CALUDE_negative_exponent_two_l3552_355201

theorem negative_exponent_two : 2⁻¹ = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_negative_exponent_two_l3552_355201


namespace NUMINAMATH_CALUDE_black_area_after_changes_l3552_355256

/-- Represents the fraction of black area remaining after a single change --/
def remaining_black_fraction : ℚ := 2 / 3

/-- Represents the number of changes --/
def num_changes : ℕ := 3

/-- Theorem stating that after three changes, 8/27 of the original area remains black --/
theorem black_area_after_changes :
  remaining_black_fraction ^ num_changes = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l3552_355256


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3552_355287

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| = 3^8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3552_355287


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3552_355218

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2 ∨ a = 5) (h2 : b = 2 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (c : ℝ), c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3552_355218


namespace NUMINAMATH_CALUDE_schur_inequality_special_case_l3552_355275

theorem schur_inequality_special_case (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_schur_inequality_special_case_l3552_355275


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3552_355289

theorem gcd_of_specific_numbers : Nat.gcd 33333333 666666666 = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3552_355289


namespace NUMINAMATH_CALUDE_range_of_a_l3552_355273

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define the relationship between p and q
def sufficient_not_necessary (p q : Prop) : Prop :=
  (¬p → ¬q) ∧ ¬(q → p)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, sufficient_not_necessary (p x) (q x a)) → a < -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3552_355273


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3552_355277

theorem abs_inequality_solution_set (x : ℝ) : 
  |x - 3| < 1 ↔ 2 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3552_355277


namespace NUMINAMATH_CALUDE_simplify_expression_l3552_355266

theorem simplify_expression (x y z : ℝ) : 
  (3 * x - (2 * y - 4 * z)) - ((3 * x - 2 * y) - 5 * z) = 9 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3552_355266


namespace NUMINAMATH_CALUDE_triangle_line_equation_l3552_355278

/-- A line with slope 3/4 that forms a triangle with the coordinate axes -/
structure TriangleLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The perimeter of the triangle formed by the line and the coordinate axes is 12 -/
  perimeter_eq : |b| + |-(4/3)*b| + Real.sqrt (b^2 + (-(4/3)*b)^2) = 12

/-- The equation of a TriangleLine is either 3x-4y+12=0 or 3x-4y-12=0 -/
theorem triangle_line_equation (l : TriangleLine) :
  (3 : ℝ) * l.b = 12 ∨ (3 : ℝ) * l.b = -12 := by sorry

end NUMINAMATH_CALUDE_triangle_line_equation_l3552_355278


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l3552_355270

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l3552_355270


namespace NUMINAMATH_CALUDE_max_cities_is_four_l3552_355265

/-- Represents the modes of transportation --/
inductive TransportMode
| Bus
| Train
| Airplane

/-- Represents a city in the country --/
structure City where
  id : Nat

/-- Represents the transportation network of the country --/
structure TransportNetwork where
  cities : List City
  connections : List City → List City → TransportMode → Prop

/-- Checks if the network satisfies the condition that no city is serviced by all three types of transportation --/
def noTripleService (network : TransportNetwork) : Prop :=
  ∀ c : City, c ∈ network.cities →
    ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      network.connections [c, c1] [c, c1] TransportMode.Bus ∧
      network.connections [c, c2] [c, c2] TransportMode.Train ∧
      network.connections [c, c3] [c, c3] TransportMode.Airplane)

/-- Checks if the network satisfies the condition that no three cities are connected by the same mode of transportation --/
def noTripleConnection (network : TransportNetwork) : Prop :=
  ∀ mode : TransportMode, ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    network.connections [c1, c2] [c1, c2] mode ∧
    network.connections [c2, c3] [c2, c3] mode ∧
    network.connections [c1, c3] [c1, c3] mode)

/-- The main theorem stating that the maximum number of cities is 4 --/
theorem max_cities_is_four :
  ∀ (network : TransportNetwork),
    (∀ (c1 c2 : City), c1 ≠ c2 → ∃ (mode : TransportMode), network.connections [c1, c2] [c1, c2] mode) →
    noTripleService network →
    noTripleConnection network →
    List.length network.cities ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_cities_is_four_l3552_355265


namespace NUMINAMATH_CALUDE_f_value_at_pi_over_4_f_monotone_increasing_l3552_355284

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) + 2 * (Real.cos x) ^ 2) / Real.cos x

def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

theorem f_value_at_pi_over_4 :
  f (Real.pi / 4) = 2 * Real.sqrt 2 :=
sorry

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo 0 (Real.pi / 4)) :=
sorry

end NUMINAMATH_CALUDE_f_value_at_pi_over_4_f_monotone_increasing_l3552_355284


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3552_355227

theorem decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3552_355227


namespace NUMINAMATH_CALUDE_largest_unreachable_sum_l3552_355293

theorem largest_unreachable_sum : 
  (∀ n > 88, ∃ a b : ℕ+, 8 * a + 11 * b = n) ∧ 
  (¬ ∃ a b : ℕ+, 8 * a + 11 * b = 88) :=
sorry

end NUMINAMATH_CALUDE_largest_unreachable_sum_l3552_355293


namespace NUMINAMATH_CALUDE_fraction_reducibility_l3552_355281

theorem fraction_reducibility (l : ℤ) :
  ∃ (d : ℤ), d > 1 ∧ d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7) ↔ ∃ (k : ℤ), l = 13 * k + 4 :=
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l3552_355281


namespace NUMINAMATH_CALUDE_max_min_m_l3552_355274

theorem max_min_m (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : 3*a + 2*b + c = 5) (h5 : 2*a + b - 3*c = 1) :
  let m := 3*a + b - 7*c
  ∃ (m_max m_min : ℝ), 
    (∀ x, x = m → x ≤ m_max) ∧ 
    (∀ x, x = m → x ≥ m_min) ∧ 
    m_max = -1/11 ∧ 
    m_min = -5/7 :=
sorry

end NUMINAMATH_CALUDE_max_min_m_l3552_355274


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3552_355222

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3552_355222


namespace NUMINAMATH_CALUDE_crane_sling_diameter_l3552_355253

/-- Represents the problem of finding the smallest suitable rope diameter for a crane sling --/
theorem crane_sling_diameter
  (M : ℝ)  -- Mass of the load in tons
  (n : ℕ)  -- Number of slings
  (α : ℝ)  -- Angle of each sling with vertical in radians
  (k : ℝ)  -- Safety factor
  (q : ℝ)  -- Maximum load per thread in N/mm²
  (g : ℝ)  -- Free fall acceleration in m/s²
  (h₁ : M = 20)
  (h₂ : n = 3)
  (h₃ : α = Real.pi / 6)  -- 30° in radians
  (h₄ : k = 6)
  (h₅ : q = 1000)
  (h₆ : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    (∀ (D' : ℕ), D' < D → 
      (Real.pi * D'^2 / 4) * q * 10^6 < 
      k * M * g * 1000 / (n * Real.cos α)) ∧
    (Real.pi * D^2 / 4) * q * 10^6 ≥ 
    k * M * g * 1000 / (n * Real.cos α) :=
sorry

end NUMINAMATH_CALUDE_crane_sling_diameter_l3552_355253


namespace NUMINAMATH_CALUDE_discount_percent_calculation_l3552_355224

theorem discount_percent_calculation (MP : ℝ) (CP : ℝ) (h1 : CP = 0.64 * MP) (h2 : 34.375 = (CP * 1.34375 - CP) / CP * 100) :
  (MP - CP * 1.34375) / MP * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_discount_percent_calculation_l3552_355224


namespace NUMINAMATH_CALUDE_exists_bijection_sum_inverse_neg_l3552_355208

theorem exists_bijection_sum_inverse_neg : 
  ∃ (f : ℝ → ℝ), Function.Bijective f ∧ ∀ x : ℝ, f x + (Function.invFun f) x = -x := by
  sorry

end NUMINAMATH_CALUDE_exists_bijection_sum_inverse_neg_l3552_355208


namespace NUMINAMATH_CALUDE_parabola_and_line_l3552_355251

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  y₀ : ℝ
  h_p_pos : p > 0
  h_on_parabola : y₀^2 = 2 * p * 2
  h_focus_dist : (2 - p/2)^2 + y₀^2 = 4^2

/-- A line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  m : ℝ
  h_not_origin : m ≠ 0
  h_two_points : ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (x₁^2 + (2*m - 8)*x₁ + m^2 = 0) ∧ 
    (x₂^2 + (2*m - 8)*x₂ + m^2 = 0)
  h_perpendicular : ∃ x₁ x₂ y₁ y₂, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem parabola_and_line (par : Parabola) (l : IntersectingLine par) :
  par.p = 4 ∧ l.m = -8 := by sorry

end NUMINAMATH_CALUDE_parabola_and_line_l3552_355251


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3552_355299

/-- The focal length of the ellipse 2x^2 + 3y^2 = 6 is 2 -/
theorem ellipse_focal_length : 
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + 3 * y^2 = 6}
  ∃ (f : ℝ), f = 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      ∃ (c₁ c₂ : ℝ × ℝ), 
        (c₁.1 - x)^2 + (c₁.2 - y)^2 + (c₂.1 - x)^2 + (c₂.2 - y)^2 = f^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l3552_355299


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3552_355290

-- Define the quadratic function
def f (x : ℝ) : ℝ := -6 * x^2 + 36 * x - 48

-- State the theorem
theorem quadratic_function_properties :
  f 2 = 0 ∧ f 4 = 0 ∧ f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3552_355290


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3552_355239

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factorization of 45
axiom factorization_of_45 : 45 = 3 * 3 * 5

-- Theorem statement
theorem no_primes_divisible_by_45 :
  ∀ p : ℕ, is_prime p → ¬(45 ∣ p) :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3552_355239


namespace NUMINAMATH_CALUDE_equal_piece_length_equal_piece_length_proof_l3552_355298

/-- Given a rope of 1165 cm cut into 154 pieces, where 4 pieces are 100mm each and the rest are equal,
    the length of each equal piece is 75 mm. -/
theorem equal_piece_length : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (special_pieces : ℕ) (special_length_mm : ℕ) =>
    total_length_cm = 1165 ∧
    total_pieces = 154 ∧
    equal_pieces = 150 ∧
    special_pieces = 4 ∧
    special_length_mm = 100 →
    (total_length_cm * 10 - special_pieces * special_length_mm) / equal_pieces = 75

/-- Proof of the theorem -/
theorem equal_piece_length_proof : equal_piece_length 1165 154 150 4 100 := by
  sorry

end NUMINAMATH_CALUDE_equal_piece_length_equal_piece_length_proof_l3552_355298


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3552_355202

/-- 
Given a parallelogram where one angle exceeds the other by 40 degrees,
prove that the measure of the smaller angle is 70 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (smaller_angle larger_angle : ℝ),
  -- Conditions
  (smaller_angle > 0) →  -- Angle measure is positive
  (larger_angle > 0) →  -- Angle measure is positive
  (larger_angle = smaller_angle + 40) →  -- One angle exceeds the other by 40
  (smaller_angle + larger_angle = 180) →  -- Adjacent angles are supplementary
  -- Conclusion
  smaller_angle = 70 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l3552_355202
