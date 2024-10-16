import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3599_359970

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the equations of its asymptotes are y = ± x/2, then b = 1 -/
theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∀ x : ℝ, (∃ y : ℝ, y = x / 2 ∨ y = -x / 2) → 
    x^2 / 4 - y^2 / b^2 = 1) →
  b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3599_359970


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3599_359941

/-- Given an arithmetic sequence with common difference d ≠ 0,
    if a₁, a₃, a₇ form a geometric sequence, then a₁/d = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  (∃ r, a 3 = a 1 * r ∧ a 7 = a 3 * r) →
  a 1 / d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3599_359941


namespace NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l3599_359944

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by sorry

end NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l3599_359944


namespace NUMINAMATH_CALUDE_system_solution_l3599_359995

theorem system_solution : 
  ∃ (x y : ℚ), 4 * x - 35 * y = -1 ∧ 3 * y - x = 5 ∧ x = -172/23 ∧ y = -19/23 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3599_359995


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3599_359973

theorem incorrect_inequality (m n : ℝ) (h : m > n) : ¬(-2 * m > -2 * n) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3599_359973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3599_359915

/-- 
Given an arithmetic sequence with three terms where the first term is 3² and the third term is 3⁴,
prove that the second term (z) is equal to 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term is 3²
    a 2 = 3^4 →                                       -- third term is 3⁴
    a 1 = 45 :=                                       -- second term (z) is 45
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3599_359915


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3599_359967

theorem infinitely_many_solutions : Set.Infinite {n : ℤ | (n - 3) * (n + 5) > 0} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3599_359967


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3599_359946

theorem lcm_hcf_relation (a b : ℕ) (h : a = 24 ∧ b = 198) :
  Nat.lcm a b = 792 :=
by
  sorry

#check lcm_hcf_relation

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3599_359946


namespace NUMINAMATH_CALUDE_line_equation_of_l_l3599_359926

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line l passing through (3,2) with slope -4 -/
def l : Line := { point := (3, 2), slope := -4 }

/-- Theorem: The equation of line l is 4x + y - 14 = 0 -/
theorem line_equation_of_l : 
  ∃ (eq : LineEquation), eq.a = 4 ∧ eq.b = 1 ∧ eq.c = -14 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y - l.point.2 = l.slope * (x - l.point.1) :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_l_l3599_359926


namespace NUMINAMATH_CALUDE_chipmunk_families_count_l3599_359943

theorem chipmunk_families_count (families_left families_went_away : ℕ) 
  (h1 : families_left = 21)
  (h2 : families_went_away = 65) :
  families_left + families_went_away = 86 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_count_l3599_359943


namespace NUMINAMATH_CALUDE_chessboard_rectangle_same_color_l3599_359927

-- Define the chessboard as a 4x7 matrix of booleans (true for black, false for white)
def Chessboard := Matrix (Fin 4) (Fin 7) Bool

-- Define a rectangle on the chessboard
def Rectangle (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  r1 < r2 ∧ c1 < c2

-- Define the property of a rectangle having all corners of the same color
def SameColorCorners (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  Rectangle board r1 r2 c1 c2 ∧
  board r1 c1 = board r1 c2 ∧
  board r1 c1 = board r2 c1 ∧
  board r1 c1 = board r2 c2

-- The main theorem
theorem chessboard_rectangle_same_color (board : Chessboard) :
  ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 7), SameColorCorners board r1 r2 c1 c2 := by
  sorry


end NUMINAMATH_CALUDE_chessboard_rectangle_same_color_l3599_359927


namespace NUMINAMATH_CALUDE_inequality_solutions_l3599_359947

theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ≤ -5 ∨ x ≥ 2) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3599_359947


namespace NUMINAMATH_CALUDE_harriets_age_l3599_359932

theorem harriets_age (mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  mother_age = 60 →
  peter_age = mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by
sorry

end NUMINAMATH_CALUDE_harriets_age_l3599_359932


namespace NUMINAMATH_CALUDE_pizza_price_correct_l3599_359907

/-- The price of one box of pizza -/
def pizza_price : ℝ := 12

/-- The price of one pack of potato fries -/
def fries_price : ℝ := 0.3

/-- The price of one can of soda -/
def soda_price : ℝ := 2

/-- The number of pizza boxes sold -/
def pizza_sold : ℕ := 15

/-- The number of potato fries packs sold -/
def fries_sold : ℕ := 40

/-- The number of soda cans sold -/
def soda_sold : ℕ := 25

/-- The fundraising goal -/
def goal : ℝ := 500

/-- The amount still needed to reach the goal -/
def amount_needed : ℝ := 258

theorem pizza_price_correct : 
  pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold = goal - amount_needed :=
by sorry

end NUMINAMATH_CALUDE_pizza_price_correct_l3599_359907


namespace NUMINAMATH_CALUDE_garden_area_calculation_l3599_359949

/-- The area of a rectangular garden plot -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden plot with length 1.2 meters and width 0.5 meters is 0.6 square meters -/
theorem garden_area_calculation :
  garden_area 1.2 0.5 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l3599_359949


namespace NUMINAMATH_CALUDE_two_valid_solutions_exist_l3599_359923

def is_valid_solution (a b c d e f g h i : ℕ) : Prop :=
  a ∈ Finset.range 10 ∧ b ∈ Finset.range 10 ∧ c ∈ Finset.range 10 ∧
  d ∈ Finset.range 10 ∧ e ∈ Finset.range 10 ∧ f ∈ Finset.range 10 ∧
  g ∈ Finset.range 10 ∧ h ∈ Finset.range 10 ∧ i ∈ Finset.range 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  (100 * a + 10 * b + c) + d = 10 * e + f ∧
  g * h = 10 * i + f

theorem two_valid_solutions_exist : ∃ (a b c d e f g h i : ℕ),
  is_valid_solution a b c d e f g h i ∧
  ∃ (j k l m n o p q r : ℕ),
  is_valid_solution j k l m n o p q r ∧
  (a ≠ j ∨ b ≠ k ∨ c ≠ l ∨ d ≠ m ∨ e ≠ n ∨ f ≠ o ∨ g ≠ p ∨ h ≠ q ∨ i ≠ r) :=
sorry

end NUMINAMATH_CALUDE_two_valid_solutions_exist_l3599_359923


namespace NUMINAMATH_CALUDE_cost_difference_l3599_359908

/-- The price difference between two types of candy in kopecks per kilogram -/
def price_difference : ℕ := 80

/-- The total amount of candy bought by each person in grams -/
def total_amount : ℕ := 150

/-- The cost of Andrey's purchase in kopecks -/
def andrey_cost (x : ℕ) : ℚ :=
  (150 * x + 8000 : ℚ) / 1000

/-- The cost of Yura's purchase in kopecks -/
def yura_cost (x : ℕ) : ℚ :=
  (150 * x + 6000 : ℚ) / 1000

/-- The theorem stating the difference in cost between Andrey's and Yura's purchases -/
theorem cost_difference (x : ℕ) :
  andrey_cost x - yura_cost x = 2 / 1000 := by sorry

end NUMINAMATH_CALUDE_cost_difference_l3599_359908


namespace NUMINAMATH_CALUDE_werewolf_identity_l3599_359956

structure Person where
  name : String
  is_knight : Bool
  is_werewolf : Bool
  is_liar : Bool

def A : Person := { name := "A", is_knight := true, is_werewolf := true, is_liar := false }
def B : Person := { name := "B", is_knight := false, is_werewolf := false, is_liar := true }
def C : Person := { name := "C", is_knight := false, is_werewolf := false, is_liar := true }

theorem werewolf_identity (A B C : Person) :
  (A.is_knight ↔ (A.is_liar ∨ B.is_liar ∨ C.is_liar)) →
  (B.is_knight ↔ C.is_knight) →
  ((A.is_werewolf ∧ A.is_knight) ∨ (B.is_werewolf ∧ B.is_knight)) →
  (A.is_werewolf ∨ B.is_werewolf) →
  ¬(A.is_werewolf ∧ B.is_werewolf) →
  A.is_werewolf := by
  sorry

#check werewolf_identity A B C

end NUMINAMATH_CALUDE_werewolf_identity_l3599_359956


namespace NUMINAMATH_CALUDE_room_length_is_correct_l3599_359959

/-- The length of a rectangular room -/
def room_length : ℝ := 5.5

/-- The width of the room -/
def room_width : ℝ := 3.75

/-- The cost of paving the floor -/
def paving_cost : ℝ := 12375

/-- The rate of paving per square meter -/
def paving_rate : ℝ := 600

/-- Theorem stating that the room length is correct given the conditions -/
theorem room_length_is_correct : 
  room_length * room_width * paving_rate = paving_cost := by sorry

end NUMINAMATH_CALUDE_room_length_is_correct_l3599_359959


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l3599_359992

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℚ) 
  (num_subjects : ℕ) 
  (h1 : english = 45) 
  (h2 : mathematics = 35) 
  (h3 : physics = 52) 
  (h4 : biology = 55) 
  (h5 : average = 46.8) 
  (h6 : num_subjects = 5) :
  ∃ (chemistry : ℕ), 
    (english + mathematics + physics + biology + chemistry : ℚ) / num_subjects = average ∧ 
    chemistry = 47 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l3599_359992


namespace NUMINAMATH_CALUDE_finite_perfect_squares_l3599_359939

/-- For positive integers a and b, the set of integers n for which both an^2 + b and a(n+1)^2 + b are perfect squares is finite -/
theorem finite_perfect_squares (a b : ℕ+) :
  {n : ℤ | ∃ x y : ℤ, (a : ℤ) * n^2 + (b : ℤ) = x^2 ∧ (a : ℤ) * (n + 1)^2 + (b : ℤ) = y^2}.Finite :=
by sorry

end NUMINAMATH_CALUDE_finite_perfect_squares_l3599_359939


namespace NUMINAMATH_CALUDE_angle_H_measure_l3599_359998

/-- Pentagon MATHS with specific angle conditions -/
structure Pentagon where
  M : ℝ  -- Measure of angle M
  A : ℝ  -- Measure of angle A
  T : ℝ  -- Measure of angle T
  H : ℝ  -- Measure of angle H
  S : ℝ  -- Measure of angle S
  angles_sum : M + A + T + H + S = 540
  equal_angles : M = T ∧ T = H
  supplementary : A + S = 180

/-- The measure of angle H in the specified pentagon is 120° -/
theorem angle_H_measure (p : Pentagon) : p.H = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_H_measure_l3599_359998


namespace NUMINAMATH_CALUDE_angle_equation_solution_l3599_359976

theorem angle_equation_solution (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α + Real.cos (2*β) - Real.cos (α + β) = 3/2) :
  α = π/3 ∧ β = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_equation_solution_l3599_359976


namespace NUMINAMATH_CALUDE_C_satisfies_equation_C_specific_value_l3599_359972

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Define C as a function of x and y
def C (x y : ℝ) : ℝ := -x^2 + 10*x*y - y^2

-- Theorem 1: C satisfies the given equation
theorem C_satisfies_equation (x y : ℝ) : 3 * A x y - 2 * B x y + C x y = 0 := by
  sorry

-- Theorem 2: C equals -57/4 when x = 1/2 and y = -2
theorem C_specific_value : C (1/2) (-2) = -57/4 := by
  sorry

end NUMINAMATH_CALUDE_C_satisfies_equation_C_specific_value_l3599_359972


namespace NUMINAMATH_CALUDE_unique_solution_system_l3599_359981

theorem unique_solution_system (x y : ℝ) :
  x^2 + y^2 = 2 ∧ 
  (x^2 / (2 - y)) + (y^2 / (2 - x)) = 2 →
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3599_359981


namespace NUMINAMATH_CALUDE_incorrect_survey_method_statement_l3599_359928

-- Define survey methods
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define scenarios
inductive Scenario
| StudentInterests
| ParentWorkConditions
| PopulationCensus
| LakeWaterQuality

-- Define function to determine appropriate survey method
def appropriateSurveyMethod (scenario : Scenario) : SurveyMethod :=
  match scenario with
  | Scenario.StudentInterests => SurveyMethod.Sampling
  | Scenario.ParentWorkConditions => SurveyMethod.Comprehensive
  | Scenario.PopulationCensus => SurveyMethod.Comprehensive
  | Scenario.LakeWaterQuality => SurveyMethod.Sampling

-- Theorem to prove
theorem incorrect_survey_method_statement :
  appropriateSurveyMethod Scenario.ParentWorkConditions ≠ SurveyMethod.Sampling :=
by sorry

end NUMINAMATH_CALUDE_incorrect_survey_method_statement_l3599_359928


namespace NUMINAMATH_CALUDE_triple_root_at_zero_l3599_359911

/-- The polynomial representing the difference between the two functions -/
def P (a b c d m n : ℝ) (x : ℝ) : ℝ :=
  x^7 - 9*x^6 + 27*x^5 + a*x^4 + b*x^3 + c*x^2 + d*x - m*x - n

/-- Theorem stating that the polynomial has a triple root at x = 0 -/
theorem triple_root_at_zero (a b c d m n : ℝ) : 
  ∃ (p q : ℝ), p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  ∀ (x : ℝ), P a b c d m n x = (x - p)^2 * (x - q)^2 * x^3 :=
sorry

end NUMINAMATH_CALUDE_triple_root_at_zero_l3599_359911


namespace NUMINAMATH_CALUDE_cookies_problem_l3599_359902

/-- Calculates the number of cookies taken out in four days given the initial count,
    remaining count after a week, and assuming equal daily removal. -/
def cookies_taken_in_four_days (initial : ℕ) (remaining : ℕ) : ℕ :=
  let total_taken := initial - remaining
  let daily_taken := total_taken / 7
  4 * daily_taken

/-- Proves that given 70 initial cookies and 28 remaining after a week,
    Paul took out 24 cookies in four days. -/
theorem cookies_problem :
  cookies_taken_in_four_days 70 28 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_problem_l3599_359902


namespace NUMINAMATH_CALUDE_silverware_cost_l3599_359999

/-- The cost of silverware given the conditions in the problem -/
theorem silverware_cost : 
  ∀ (silverware_cost dinner_plates_cost : ℝ),
  dinner_plates_cost = 0.5 * silverware_cost →
  silverware_cost + dinner_plates_cost = 30 →
  silverware_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_silverware_cost_l3599_359999


namespace NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l3599_359982

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ cube_set → b ∈ cube_set → (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_subtraction :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ a > b ∧ (a - b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b > 0 ∧ (a / b) ∉ cube_set :=
sorry

end NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_subtraction_cube_set_not_closed_under_division_l3599_359982


namespace NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l3599_359962

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) : 
  dave_money = 46 →
  kyle_initial_money = 3 * dave_money - 12 →
  kyle_initial_money / 3 = kyle_initial_money - (kyle_initial_money / 3) →
  kyle_initial_money - (kyle_initial_money / 3) = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l3599_359962


namespace NUMINAMATH_CALUDE_uncle_fyodor_sandwiches_l3599_359991

theorem uncle_fyodor_sandwiches (sharik matroskin fyodor : ℕ) : 
  matroskin = 3 * sharik →
  fyodor = sharik + 21 →
  fyodor = 2 * (sharik + matroskin) →
  fyodor = 24 := by
sorry

end NUMINAMATH_CALUDE_uncle_fyodor_sandwiches_l3599_359991


namespace NUMINAMATH_CALUDE_intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l3599_359968

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic operations and relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (plane_intersect : Plane → Plane → Line → Prop)

-- Theorem 1
theorem intersection_parallel_or_intersect
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : belongs_to n α) :
  parallel m n ∨ intersect m n :=
sorry

-- Theorem 2
theorem intersection_parallel_implies_parallel_to_plane
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : parallel m n) :
  parallel_plane n α ∨ parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l3599_359968


namespace NUMINAMATH_CALUDE_social_dance_attendance_l3599_359942

theorem social_dance_attendance (men : ℕ) (women : ℕ) 
  (men_partners : ℕ) (women_partners : ℕ) :
  men = 15 →
  men_partners = 4 →
  women_partners = 3 →
  men * men_partners = women * women_partners →
  women = 20 := by
sorry

end NUMINAMATH_CALUDE_social_dance_attendance_l3599_359942


namespace NUMINAMATH_CALUDE_modular_inverse_of_9_mod_23_l3599_359966

theorem modular_inverse_of_9_mod_23 : ∃ x : ℕ, x ∈ Finset.range 23 ∧ (9 * x) % 23 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_9_mod_23_l3599_359966


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3599_359901

/-- Line equation: ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a * x + y - 2 = 0

/-- Circle equation: (x - 1)^2 + (y - a)^2 = 16/3 -/
def circle_equation (a x y : ℝ) : Prop :=
  (x - 1)^2 + (y - a)^2 = 16/3

/-- Circle center: C(1, a) -/
def circle_center (a : ℝ) : ℝ × ℝ :=
  (1, a)

/-- Triangle ABC is equilateral -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

theorem line_circle_intersection (a : ℝ) :
  ∃ A B : ℝ × ℝ,
    line_equation a A.1 A.2 ∧
    line_equation a B.1 B.2 ∧
    circle_equation a A.1 A.2 ∧
    circle_equation a B.1 B.2 ∧
    is_equilateral_triangle A B (circle_center a) →
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3599_359901


namespace NUMINAMATH_CALUDE_one_twenty_million_properties_l3599_359974

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (sn : ScientificNotation) : ℕ := sorry

/-- Determines the place value of accuracy for a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions
  | TenMillions
  | HundredMillions

def getPlaceValueAccuracy (x : ℝ) : PlaceValue := sorry

theorem one_twenty_million_properties :
  let x : ℝ := 120000000
  let sn := toScientificNotation x
  countSignificantFigures sn = 2 ∧ getPlaceValueAccuracy x = PlaceValue.Millions := by sorry

end NUMINAMATH_CALUDE_one_twenty_million_properties_l3599_359974


namespace NUMINAMATH_CALUDE_four_digit_square_decrease_theorem_l3599_359919

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def all_digits_decreasable (n k : ℕ) : Prop :=
  ∀ d, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≥ k

def decrease_all_digits (n k : ℕ) : ℕ :=
  (n / 1000 - k) * 1000 + ((n / 100) % 10 - k) * 100 + ((n / 10) % 10 - k) * 10 + (n % 10 - k)

theorem four_digit_square_decrease_theorem :
  ∀ n : ℕ, is_four_digit n → is_perfect_square n →
  (∃ k : ℕ, k > 0 ∧ all_digits_decreasable n k ∧
   is_four_digit (decrease_all_digits n k) ∧ is_perfect_square (decrease_all_digits n k)) →
  n = 3136 ∨ n = 4489 := by sorry

end NUMINAMATH_CALUDE_four_digit_square_decrease_theorem_l3599_359919


namespace NUMINAMATH_CALUDE_inequality_proof_l3599_359922

theorem inequality_proof (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b + b * c + c * a = 1) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3599_359922


namespace NUMINAMATH_CALUDE_study_abroad_work_hours_l3599_359903

/-- Proves that working 28 hours per week for the remaining 10 weeks
    will meet the financial goal, given the initial plan and actual work done. -/
theorem study_abroad_work_hours
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (goal_amount : ℕ)
  (actual_full_weeks : ℕ)
  (actual_reduced_weeks : ℕ)
  (reduced_hours_per_week : ℕ)
  (h_initial_hours : initial_hours_per_week = 25)
  (h_initial_weeks : initial_weeks = 15)
  (h_goal_amount : goal_amount = 4500)
  (h_actual_full_weeks : actual_full_weeks = 3)
  (h_actual_reduced_weeks : actual_reduced_weeks = 2)
  (h_reduced_hours : reduced_hours_per_week = 10)
  : ∃ (remaining_hours_per_week : ℕ),
    remaining_hours_per_week = 28 ∧
    (initial_hours_per_week * actual_full_weeks +
     reduced_hours_per_week * actual_reduced_weeks +
     remaining_hours_per_week * (initial_weeks - actual_full_weeks - actual_reduced_weeks)) *
    (goal_amount / (initial_hours_per_week * initial_weeks)) = goal_amount :=
by sorry

end NUMINAMATH_CALUDE_study_abroad_work_hours_l3599_359903


namespace NUMINAMATH_CALUDE_congruence_properties_l3599_359997

theorem congruence_properties : ∀ n : ℤ,
  (n ≡ 0 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) ∧
  (n ≡ 1 [ZMOD 2] → ∃ k : ℤ, n = 2 * k + 1) ∧
  (n ≡ 2018 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_congruence_properties_l3599_359997


namespace NUMINAMATH_CALUDE_correlation_coefficient_equals_height_variation_total_variation_is_one_l3599_359920

/-- The correlation coefficient between height and weight -/
def correlation_coefficient : ℝ := 0.76

/-- The proportion of weight variation explained by height -/
def height_explained_variation : ℝ := 0.76

/-- The proportion of weight variation explained by random errors -/
def random_error_variation : ℝ := 0.24

/-- Theorem stating that the correlation coefficient is equal to the proportion of variation explained by height -/
theorem correlation_coefficient_equals_height_variation :
  correlation_coefficient = height_explained_variation :=
by sorry

/-- Theorem stating that the sum of variations explained by height and random errors is 1 -/
theorem total_variation_is_one :
  height_explained_variation + random_error_variation = 1 :=
by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_equals_height_variation_total_variation_is_one_l3599_359920


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3599_359953

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + α - 2023 = 0) → 
  (β^2 + β - 2023 = 0) → 
  (α^2 + 2*α + β = 2022) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3599_359953


namespace NUMINAMATH_CALUDE_homework_challenge_l3599_359924

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_for_points (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- The homework challenge theorem -/
theorem homework_challenge :
  assignments_for_points 30 = 60 := by
sorry

#eval assignments_for_points 30

end NUMINAMATH_CALUDE_homework_challenge_l3599_359924


namespace NUMINAMATH_CALUDE_centerville_library_budget_percentage_l3599_359980

/-- Proves that the percentage of Centerville's annual budget spent on the public library is 15% -/
theorem centerville_library_budget_percentage
  (library_expense : ℕ)
  (park_percentage : ℚ)
  (remaining_budget : ℕ)
  (h1 : library_expense = 3000)
  (h2 : park_percentage = 24 / 100)
  (h3 : remaining_budget = 12200)
  : ∃ (total_budget : ℕ), 
    (library_expense : ℚ) / total_budget = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_centerville_library_budget_percentage_l3599_359980


namespace NUMINAMATH_CALUDE_bus_children_difference_solve_bus_problem_l3599_359963

theorem bus_children_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_children children_off children_on final_children =>
    initial_children - children_off + children_on = final_children →
    children_off - children_on = 24

theorem solve_bus_problem :
  bus_children_difference 36 68 (68 - 24) 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_difference_solve_bus_problem_l3599_359963


namespace NUMINAMATH_CALUDE_grassy_width_is_55_l3599_359912

/-- Represents the dimensions and cost of a rectangular plot with a gravel path -/
structure Plot where
  length : ℝ
  path_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the width of the grassy area given the plot dimensions and gravel cost -/
def calculate_grassy_width (p : Plot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions and cost, the grassy width is 55 meters -/
theorem grassy_width_is_55 (p : Plot) 
  (h1 : p.length = 110)
  (h2 : p.path_width = 2.5)
  (h3 : p.gravel_cost_per_sqm = 0.6)
  (h4 : p.total_gravel_cost = 510) :
  calculate_grassy_width p = 55 :=
sorry

end NUMINAMATH_CALUDE_grassy_width_is_55_l3599_359912


namespace NUMINAMATH_CALUDE_pen_refill_purchase_comparison_l3599_359913

theorem pen_refill_purchase_comparison (p₁ p₂ : ℝ) (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 ∧
  (2 * p₁ * p₂) / (p₁ + p₂) = (p₁ + p₂) / 2 ↔ p₁ = p₂ := by
  sorry

end NUMINAMATH_CALUDE_pen_refill_purchase_comparison_l3599_359913


namespace NUMINAMATH_CALUDE_initial_boys_on_slide_l3599_359965

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_on_slide_l3599_359965


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3599_359975

/-- Proves that in a right triangle with non-hypotenuse side lengths of 5 and 12, the hypotenuse length is 13 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3599_359975


namespace NUMINAMATH_CALUDE_second_patient_hours_l3599_359938

/-- Represents the psychologist's pricing model and patient charges -/
structure TherapyPricing where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstPatientHours : ℕ
  firstPatientCharge : ℕ
  secondPatientCharge : ℕ

/-- 
Given a psychologist's pricing model where:
- The first hour costs $30 more than each additional hour
- A 5-hour therapy session costs $400
- Another therapy session costs $252

This theorem proves that the second therapy session lasted 3 hours.
-/
theorem second_patient_hours (tp : TherapyPricing) 
  (h1 : tp.firstHourCost = tp.additionalHourCost + 30)
  (h2 : tp.firstPatientHours = 5)
  (h3 : tp.firstPatientCharge = 400)
  (h4 : tp.firstPatientCharge = tp.firstHourCost + (tp.firstPatientHours - 1) * tp.additionalHourCost)
  (h5 : tp.secondPatientCharge = 252) : 
  ∃ (h : ℕ), tp.secondPatientCharge = tp.firstHourCost + (h - 1) * tp.additionalHourCost ∧ h = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_patient_hours_l3599_359938


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3599_359987

theorem complex_equation_solution (z : ℂ) : (z + Complex.I) * (2 + Complex.I) = 5 → z = 2 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3599_359987


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l3599_359914

theorem polynomial_coefficient_equality (a b c : ℚ) : 
  (∀ x, (7*x^2 - 5*x + 9/4)*(a*x^2 + b*x + c) = 21*x^4 - 24*x^3 + 28*x^2 - 37/4*x + 21/4) →
  (a = 3 ∧ b = -9/7) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l3599_359914


namespace NUMINAMATH_CALUDE_henrys_earnings_per_lawn_l3599_359964

theorem henrys_earnings_per_lawn 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : unmowed_lawns = 7) 
  (h3 : total_earnings = 25) : 
  total_earnings / (total_lawns - unmowed_lawns) = 5 := by
  sorry

end NUMINAMATH_CALUDE_henrys_earnings_per_lawn_l3599_359964


namespace NUMINAMATH_CALUDE_marcias_final_hair_length_l3599_359961

/-- Calculates the final hair length after a series of cuts and growth periods --/
def finalHairLength (initialLength : ℝ) 
                    (firstCutPercentage : ℝ) 
                    (firstGrowthMonths : ℕ) 
                    (firstGrowthRate : ℝ) 
                    (secondCutPercentage : ℝ) 
                    (secondGrowthMonths : ℕ) 
                    (secondGrowthRate : ℝ) 
                    (finalCutLength : ℝ) : ℝ :=
  let afterFirstCut := initialLength * (1 - firstCutPercentage)
  let afterFirstGrowth := afterFirstCut + (firstGrowthMonths : ℝ) * firstGrowthRate
  let afterSecondCut := afterFirstGrowth * (1 - secondCutPercentage)
  let afterSecondGrowth := afterSecondCut + (secondGrowthMonths : ℝ) * secondGrowthRate
  afterSecondGrowth - finalCutLength

/-- Theorem stating that Marcia's final hair length is 22.04 inches --/
theorem marcias_final_hair_length : 
  finalHairLength 24 0.3 3 1.5 0.2 5 1.8 4 = 22.04 := by
  sorry

end NUMINAMATH_CALUDE_marcias_final_hair_length_l3599_359961


namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l3599_359994

/-- Given a function g where g(2) = 5, and a function h where h(x) = (g(x))^2 for all x,
    the sum of the coordinates of the point (2, h(2)) is 27. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (h_def : ∀ x, h x = (g x)^2) 
    (g_val : g 2 = 5) : 
  2 + h 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l3599_359994


namespace NUMINAMATH_CALUDE_plane_line_relations_l3599_359918

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m)
  (h_l_perp_α : perpendicular l α)
  (h_m_subset_β : subset m β) :
  (parallel α β → line_perpendicular l m) ∧
  (perpendicular l β → line_parallel m α) :=
sorry

end NUMINAMATH_CALUDE_plane_line_relations_l3599_359918


namespace NUMINAMATH_CALUDE_book_price_range_l3599_359960

-- Define the price of the book
variable (x : ℝ)

-- Define the conditions based on the wrong guesses
def student_A_wrong : Prop := ¬(x ≥ 15)
def student_B_wrong : Prop := ¬(x ≤ 12)
def student_C_wrong : Prop := ¬(x ≤ 10)

-- Theorem statement
theorem book_price_range 
  (hA : student_A_wrong x)
  (hB : student_B_wrong x)
  (hC : student_C_wrong x) :
  12 < x ∧ x < 15 := by
  sorry

end NUMINAMATH_CALUDE_book_price_range_l3599_359960


namespace NUMINAMATH_CALUDE_davids_physics_marks_l3599_359989

theorem davids_physics_marks :
  let english_marks : ℕ := 51
  let math_marks : ℕ := 65
  let chemistry_marks : ℕ := 67
  let biology_marks : ℕ := 85
  let average_marks : ℕ := 70
  let total_subjects : ℕ := 5

  let total_marks : ℕ := average_marks * total_subjects
  let known_marks : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks : ℕ := total_marks - known_marks

  physics_marks = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l3599_359989


namespace NUMINAMATH_CALUDE_solid_color_non_yellow_purple_percentage_l3599_359929

/-- Represents the distribution of marble types and colors -/
structure MarbleDistribution where
  solid_colored : ℝ
  striped : ℝ
  dotted : ℝ
  swirl_patterned : ℝ
  red_solid : ℝ
  blue_solid : ℝ
  green_solid : ℝ
  yellow_solid : ℝ
  purple_solid : ℝ

/-- The given marble distribution -/
def given_distribution : MarbleDistribution :=
  { solid_colored := 0.70
    striped := 0.10
    dotted := 0.10
    swirl_patterned := 0.10
    red_solid := 0.25
    blue_solid := 0.25
    green_solid := 0.20
    yellow_solid := 0.15
    purple_solid := 0.15 }

/-- Theorem stating that 49% of all marbles are solid-colored and neither yellow nor purple -/
theorem solid_color_non_yellow_purple_percentage
  (d : MarbleDistribution)
  (h1 : d.solid_colored + d.striped + d.dotted + d.swirl_patterned = 1)
  (h2 : d.red_solid + d.blue_solid + d.green_solid + d.yellow_solid + d.purple_solid = 1)
  (h3 : d = given_distribution) :
  d.solid_colored * (d.red_solid + d.blue_solid + d.green_solid) = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_non_yellow_purple_percentage_l3599_359929


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3599_359979

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem perpendicular_line_equation (given_line : Line) (p : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  p.x = -1 ∧ p.y = 2 →
  ∃ (l : Line), l.contains p ∧ l.perpendicular given_line ∧
  l.a = 3 ∧ l.b = 2 ∧ l.c = -1 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l3599_359979


namespace NUMINAMATH_CALUDE_first_year_after_2021_with_sum_15_l3599_359951

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2021_with_sum_15 (year : ℕ) : Prop :=
  year > 2021 ∧
  sum_of_digits year = 15 ∧
  ∀ y : ℕ, 2021 < y ∧ y < year → sum_of_digits y ≠ 15

theorem first_year_after_2021_with_sum_15 :
  is_first_year_after_2021_with_sum_15 2049 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2021_with_sum_15_l3599_359951


namespace NUMINAMATH_CALUDE_total_height_is_148_inches_l3599_359948

-- Define the heights of sculptures in feet and inches
def sculpture1_feet : ℕ := 2
def sculpture1_inches : ℕ := 10
def sculpture2_feet : ℕ := 3
def sculpture2_inches : ℕ := 5
def sculpture3_feet : ℕ := 4
def sculpture3_inches : ℕ := 7

-- Define the heights of bases in inches
def base1_inches : ℕ := 4
def base2_inches : ℕ := 6
def base3_inches : ℕ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Function to convert feet and inches to total inches
def to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * inches_per_foot + inches

-- Theorem statement
theorem total_height_is_148_inches :
  to_inches sculpture1_feet sculpture1_inches + base1_inches +
  to_inches sculpture2_feet sculpture2_inches + base2_inches +
  to_inches sculpture3_feet sculpture3_inches + base3_inches = 148 := by
  sorry


end NUMINAMATH_CALUDE_total_height_is_148_inches_l3599_359948


namespace NUMINAMATH_CALUDE_paths_4x3_grid_l3599_359940

/-- The number of unique paths in a grid -/
def grid_paths (m n : ℕ) : ℕ := (m + n).choose m

/-- Theorem: The number of unique paths in a 4x3 grid is 35 -/
theorem paths_4x3_grid : grid_paths 4 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_paths_4x3_grid_l3599_359940


namespace NUMINAMATH_CALUDE_sum_product_ratio_l3599_359950

theorem sum_product_ratio (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l3599_359950


namespace NUMINAMATH_CALUDE_collinear_vectors_l3599_359921

/-- Given two vectors a and b in R², prove that if 2a + b is collinear with b,
    then the y-coordinate of a is 1/2. -/
theorem collinear_vectors (l x : ℝ) : 
  let a : ℝ × ℝ := (l, x)
  let b : ℝ × ℝ := (4, 2)
  (∃ (k : ℝ), (2 * a.1 + b.1, 2 * a.2 + b.2) = k • b) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l3599_359921


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l3599_359933

theorem gcd_linear_combination (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd (11 * a + 2 * b).natAbs (18 * a + 5 * b).natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l3599_359933


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3599_359925

/-- Given a geometric sequence with first term -2 and sum of first 3 terms -7/2,
    prove that the common ratio is either 1/2 or -3/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a1 : a 1 = -2)
  (h_S3 : (a 0) + (a 1) + (a 2) = -7/2) :
  (a 1) / (a 0) = 1/2 ∨ (a 1) / (a 0) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3599_359925


namespace NUMINAMATH_CALUDE_net_profit_calculation_l3599_359931

/-- Calculates the net profit given the purchase price, markup, and overhead percentage. -/
def calculate_net_profit (purchase_price markup overhead_percent : ℚ) : ℚ :=
  let overhead := purchase_price * overhead_percent
  markup - overhead

/-- Theorem stating that given the specific values in the problem, the net profit is $40.60. -/
theorem net_profit_calculation :
  let purchase_price : ℚ := 48
  let markup : ℚ := 55
  let overhead_percent : ℚ := 0.30
  calculate_net_profit purchase_price markup overhead_percent = 40.60 := by
  sorry

#eval calculate_net_profit 48 55 0.30

end NUMINAMATH_CALUDE_net_profit_calculation_l3599_359931


namespace NUMINAMATH_CALUDE_min_sum_of_fractions_l3599_359988

def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D E : Nat) 
  (h1 : A ∈ Digits) (h2 : B ∈ Digits) (h3 : C ∈ Digits) (h4 : D ∈ Digits) (h5 : E ∈ Digits)
  (h6 : A ≠ B) (h7 : A ≠ C) (h8 : A ≠ D) (h9 : A ≠ E)
  (h10 : B ≠ C) (h11 : B ≠ D) (h12 : B ≠ E)
  (h13 : C ≠ D) (h14 : C ≠ E)
  (h15 : D ≠ E) :
  (A : ℚ) / B + (C : ℚ) / D + (E : ℚ) / 9 ≥ 125 / 168 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_fractions_l3599_359988


namespace NUMINAMATH_CALUDE_circle_diameter_l3599_359986

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 9 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l3599_359986


namespace NUMINAMATH_CALUDE_coat_cost_proof_l3599_359957

/-- The cost of the more expensive coat -/
def expensive_coat_cost : ℝ := 300

/-- The lifespan of the more expensive coat in years -/
def expensive_coat_lifespan : ℕ := 15

/-- The cost of the cheaper coat -/
def cheaper_coat_cost : ℝ := 120

/-- The lifespan of the cheaper coat in years -/
def cheaper_coat_lifespan : ℕ := 5

/-- The number of years over which we compare the costs -/
def comparison_period : ℕ := 30

/-- The amount saved by buying the more expensive coat over the comparison period -/
def savings : ℝ := 120

theorem coat_cost_proof :
  expensive_coat_cost * (comparison_period / expensive_coat_lifespan) =
  cheaper_coat_cost * (comparison_period / cheaper_coat_lifespan) - savings :=
by sorry

end NUMINAMATH_CALUDE_coat_cost_proof_l3599_359957


namespace NUMINAMATH_CALUDE_mL_to_L_conversion_l3599_359936

-- Define the conversion rate
def mL_per_L : ℝ := 1000

-- Define the volume in milliliters
def volume_mL : ℝ := 27

-- Theorem to prove the conversion
theorem mL_to_L_conversion :
  volume_mL / mL_per_L = 0.027 := by
  sorry

end NUMINAMATH_CALUDE_mL_to_L_conversion_l3599_359936


namespace NUMINAMATH_CALUDE_weight_of_b_l3599_359904

/-- Given three weights a, b, and c, prove that b = 37 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 46 →
  b = 37 := by
sorry


end NUMINAMATH_CALUDE_weight_of_b_l3599_359904


namespace NUMINAMATH_CALUDE_kasun_family_children_l3599_359984

/-- Represents the Kasun family structure and ages -/
structure KasunFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  dog_age : ℕ
  children_total_age : ℕ

/-- The average age of the entire family is 22 years -/
def family_average (f : KasunFamily) : Prop :=
  (f.father_age + f.mother_age + f.children_total_age + f.dog_age) / (2 + f.num_children + 1) = 22

/-- The average age of the mother, children, and the pet dog is 18 years -/
def partial_average (f : KasunFamily) : Prop :=
  (f.mother_age + f.children_total_age + f.dog_age) / (1 + f.num_children + 1) = 18

/-- The theorem stating that the number of children in the Kasun family is 5 -/
theorem kasun_family_children (f : KasunFamily) 
  (h1 : family_average f)
  (h2 : partial_average f)
  (h3 : f.father_age = 50)
  (h4 : f.dog_age = 10) : 
  f.num_children = 5 := by
  sorry

end NUMINAMATH_CALUDE_kasun_family_children_l3599_359984


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3599_359906

/-- A rhombus with side length 40 units and shorter diagonal 56 units has a longer diagonal of length 24√17 units. -/
theorem rhombus_longer_diagonal (s : ℝ) (d₁ : ℝ) (d₂ : ℝ) 
    (h₁ : s = 40) 
    (h₂ : d₁ = 56) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : 
  d₂ = 24 * Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3599_359906


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l3599_359910

/-- The number of rectangles on a 6x6 checkerboard -/
def num_rectangles : ℕ := 441

/-- The number of squares on a 6x6 checkerboard -/
def num_squares : ℕ := 91

/-- Theorem stating that the ratio of squares to rectangles on a 6x6 checkerboard is 13/63 -/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 13 / 63 := by sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l3599_359910


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3599_359937

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3599_359937


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3599_359983

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3599_359983


namespace NUMINAMATH_CALUDE_valid_monomial_l3599_359996

def is_valid_monomial (m : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, m x y = -2 * x^a * y^b ∧ a + b = 3

theorem valid_monomial : 
  is_valid_monomial (fun x y ↦ -2 * x^2 * y) := by sorry

end NUMINAMATH_CALUDE_valid_monomial_l3599_359996


namespace NUMINAMATH_CALUDE_tan_inequality_l3599_359900

theorem tan_inequality (n : ℕ) (x : ℝ) (h1 : 0 < x) (h2 : x < π / (2 * n)) :
  (1/2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1/n) * Real.tan (n * x) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l3599_359900


namespace NUMINAMATH_CALUDE_stratified_sampling_pine_count_l3599_359935

/-- Represents the number of pine saplings in a stratified sample -/
def pineInSample (totalSaplings : ℕ) (totalPine : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalPine * sampleSize) / totalSaplings

/-- Theorem: In a stratified sampling of 150 saplings from a forest with 30,000 saplings,
    of which 4,000 are pine saplings, the number of pine saplings in the sample is 20. -/
theorem stratified_sampling_pine_count :
  pineInSample 30000 4000 150 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_pine_count_l3599_359935


namespace NUMINAMATH_CALUDE_medicine_price_proof_l3599_359955

/-- Proves that the original price of a medicine is $150 given the specified conditions --/
theorem medicine_price_proof (cashback_rate : Real) (rebate : Real) (final_cost : Real) :
  cashback_rate = 0.1 →
  rebate = 25 →
  final_cost = 110 →
  ∃ (original_price : Real),
    original_price - (cashback_rate * original_price + rebate) = final_cost ∧
    original_price = 150 := by
  sorry

#check medicine_price_proof

end NUMINAMATH_CALUDE_medicine_price_proof_l3599_359955


namespace NUMINAMATH_CALUDE_max_volume_container_l3599_359916

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : ℝ :=
  d.shortSide * d.longSide * d.height

/-- Represents the constraints of the problem --/
def isValidContainer (d : ContainerDimensions) : Prop :=
  d.longSide = d.shortSide + 0.5 ∧
  2 * (d.shortSide + d.longSide) + 4 * d.height = 14.8 ∧
  d.shortSide > 0 ∧ d.longSide > 0 ∧ d.height > 0

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    isValidContainer d ∧
    volume d = 1.8 ∧
    d.height = 1.2 ∧
    ∀ (d' : ContainerDimensions), isValidContainer d' → volume d' ≤ volume d :=
by sorry

end NUMINAMATH_CALUDE_max_volume_container_l3599_359916


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l3599_359930

theorem fixed_points_of_f (f : ℝ → ℝ) (hf : ∀ x, f x = 4 * x - x^2) :
  ∃ a b : ℝ, a ≠ b ∧ f a = b ∧ f b = a ∧
    ((a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
     (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l3599_359930


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3599_359945

theorem rectangle_length_proof (area_single : ℝ) (area_overlap : ℝ) (diagonal : ℝ) :
  area_single = 48 →
  area_overlap = 72 →
  diagonal = 6 →
  ∃ (length width : ℝ),
    length * width = area_single ∧
    length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3599_359945


namespace NUMINAMATH_CALUDE_curve_symmetry_about_origin_l3599_359917

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop := 3 * x^2 - 8 * x * y + 2 * y^2 = 0

-- Theorem stating the symmetry about the origin
theorem curve_symmetry_about_origin :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-x) (-y) :=
by sorry

end NUMINAMATH_CALUDE_curve_symmetry_about_origin_l3599_359917


namespace NUMINAMATH_CALUDE_incenter_circles_theorem_l3599_359969

-- Define the basic geometric objects
variable (A B C I : Point)
variable (O₁ O₂ O₃ : Point)
variable (A' B' C' : Point)

-- Define the incenter
def is_incenter (I : Point) (A B C : Point) : Prop := sorry

-- Define circles passing through points
def circle_through (O : Point) (P Q : Point) : Prop := sorry

-- Define perpendicular intersection of circles
def perpendicular_intersection (O : Point) (I : Point) : Prop := sorry

-- Define the other intersection point of two circles
def other_intersection (O₁ O₂ : Point) (P : Point) : Point := sorry

-- Define the circumradius of a triangle
def circumradius (A B C : Point) : ℝ := sorry

-- Define the radius of a circle
def circle_radius (O : Point) : ℝ := sorry

-- State the theorem
theorem incenter_circles_theorem 
  (h_incenter : is_incenter I A B C)
  (h_O₁ : circle_through O₁ B C)
  (h_O₂ : circle_through O₂ A C)
  (h_O₃ : circle_through O₃ A B)
  (h_perp₁ : perpendicular_intersection O₁ I)
  (h_perp₂ : perpendicular_intersection O₂ I)
  (h_perp₃ : perpendicular_intersection O₃ I)
  (h_A' : A' = other_intersection O₂ O₃ A)
  (h_B' : B' = other_intersection O₁ O₃ B)
  (h_C' : C' = other_intersection O₁ O₂ C) :
  circumradius A' B' C' = (1/2) * circle_radius I := by sorry

end NUMINAMATH_CALUDE_incenter_circles_theorem_l3599_359969


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3599_359977

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 5 = 0 ∧ 
                x₂^2 + m*x₂ + 5 = 0 ∧ 
                x₁ = 2*abs x₂ - 3) → 
  m = -9/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3599_359977


namespace NUMINAMATH_CALUDE_optimal_parking_allocation_l3599_359958

/-- Represents the parking space allocation problem -/
structure ParkingAllocation where
  total_spaces : ℕ
  above_ground_cost : ℚ
  underground_cost : ℚ
  total_budget : ℚ
  above_ground_spaces : ℕ
  underground_spaces : ℕ

/-- The optimal allocation satisfies the given conditions -/
def is_optimal_allocation (p : ParkingAllocation) : Prop :=
  p.total_spaces = 5000 ∧
  3 * p.above_ground_cost + 2 * p.underground_cost = 0.8 ∧
  2 * p.above_ground_cost + 4 * p.underground_cost = 1.2 ∧
  p.above_ground_cost = 0.1 ∧
  p.underground_cost = 0.25 ∧
  p.total_budget = 950 ∧
  p.above_ground_spaces + p.underground_spaces = p.total_spaces ∧
  p.above_ground_spaces * p.above_ground_cost + p.underground_spaces * p.underground_cost ≤ p.total_budget

/-- The theorem stating the optimal allocation -/
theorem optimal_parking_allocation :
  ∃ (p : ParkingAllocation), is_optimal_allocation p ∧ 
    p.above_ground_spaces = 2000 ∧ p.underground_spaces = 3000 := by
  sorry


end NUMINAMATH_CALUDE_optimal_parking_allocation_l3599_359958


namespace NUMINAMATH_CALUDE_optimal_price_for_equipment_l3599_359985

/-- Represents the selling price and annual sales volume relationship for a high-tech equipment -/
structure EquipmentSales where
  cost_price : ℝ
  price_volume_1 : ℝ × ℝ
  price_volume_2 : ℝ × ℝ
  max_price : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the equipment -/
def optimal_selling_price (sales : EquipmentSales) : ℝ :=
  sorry

/-- Theorem stating the optimal selling price for the given conditions -/
theorem optimal_price_for_equipment (sales : EquipmentSales)
  (h1 : sales.cost_price = 300000)
  (h2 : sales.price_volume_1 = (350000, 550))
  (h3 : sales.price_volume_2 = (400000, 500))
  (h4 : sales.max_price = 600000)
  (h5 : sales.target_profit = 80000000) :
  optimal_selling_price sales = 500000 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_for_equipment_l3599_359985


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l3599_359954

def second_smallest_prime : ℕ := 3

def largest_integer_less_than_150_with_three_divisors : ℕ := 121

theorem sum_of_special_numbers :
  second_smallest_prime + largest_integer_less_than_150_with_three_divisors = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l3599_359954


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3599_359934

theorem sqrt_equation_solution (t : ℝ) :
  (Real.sqrt (5 * Real.sqrt (t - 5)) = (10 - t + t^2)^(1/4)) →
  (t = 13 + Real.sqrt 34 ∨ t = 13 - Real.sqrt 34) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3599_359934


namespace NUMINAMATH_CALUDE_tan_alpha_equals_three_l3599_359905

theorem tan_alpha_equals_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10) : 
  Real.tan α = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_three_l3599_359905


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3599_359978

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x - 2) > 3 ↔ x < 0 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3599_359978


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l3599_359952

/-- The curve function f(x) = x³ - 2x --/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point of tangency --/
def point : ℝ × ℝ := (1, -1)

/-- The proposed tangent line equation --/
def tangent_line (x y : ℝ) : Prop := x - y - 2 = 0

theorem tangent_line_at_point :
  tangent_line point.1 point.2 ∧
  f point.1 = point.2 ∧
  (tangent_line point.1 point.2 → ∀ x y : ℝ, tangent_line x y ↔ y - point.2 = f' point.1 * (x - point.1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l3599_359952


namespace NUMINAMATH_CALUDE_valid_numbers_l3599_359990

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b k : ℕ),
    10 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b < 10^k ∧
    N = 10^k * a + b ∧
    Odd N ∧
    10^k * a + b = 149 * b

theorem valid_numbers :
  ∀ N : ℕ, is_valid_number N → (N = 745 ∨ N = 3725) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3599_359990


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l3599_359971

/-- The cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ
  cookie : ℝ

/-- Given conditions of the luncheon costs -/
def luncheon_conditions (cost : LuncheonCost) : Prop :=
  5 * cost.sandwich + 9 * cost.coffee + 2 * cost.pie + 3 * cost.cookie = 5.85 ∧
  6 * cost.sandwich + 12 * cost.coffee + 2 * cost.pie + 4 * cost.cookie = 7.20

/-- Theorem stating the cost of one of each item -/
theorem luncheon_cost_theorem (cost : LuncheonCost) :
  luncheon_conditions cost →
  cost.sandwich + cost.coffee + cost.pie + cost.cookie = 1.35 :=
by sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l3599_359971


namespace NUMINAMATH_CALUDE_comic_book_stacking_permutations_l3599_359909

theorem comic_book_stacking_permutations :
  let spiderman_books : ℕ := 7
  let archie_books : ℕ := 4
  let garfield_books : ℕ := 5
  let batman_books : ℕ := 3
  let total_books : ℕ := spiderman_books + archie_books + garfield_books + batman_books
  let non_batman_types : ℕ := 3  -- Spiderman, Archie, and Garfield

  (spiderman_books.factorial * archie_books.factorial * garfield_books.factorial * batman_books.factorial) *
  non_batman_types.factorial = 55085760 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_permutations_l3599_359909


namespace NUMINAMATH_CALUDE_four_Y_one_equals_27_l3599_359993

/-- Definition of the Y operation -/
def Y (a b : ℝ) : ℝ := 3 * (a^2 - 2*a*b + b^2)

/-- Theorem stating that 4 Y 1 = 27 -/
theorem four_Y_one_equals_27 : Y 4 1 = 27 := by sorry

end NUMINAMATH_CALUDE_four_Y_one_equals_27_l3599_359993
