import Mathlib

namespace NUMINAMATH_CALUDE_sphere_surface_area_for_given_prism_l3423_342311

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  height : ℝ
  volume : ℝ
  prism_on_sphere : Bool

/-- The surface area of a sphere given a PrismOnSphere -/
def sphere_surface_area (p : PrismOnSphere) : ℝ := sorry

theorem sphere_surface_area_for_given_prism :
  ∀ p : PrismOnSphere,
    p.height = 4 ∧ 
    p.volume = 16 ∧ 
    p.prism_on_sphere = true →
    sphere_surface_area p = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_for_given_prism_l3423_342311


namespace NUMINAMATH_CALUDE_circle_graph_parts_sum_to_one_l3423_342319

theorem circle_graph_parts_sum_to_one :
  let white : ℚ := 1/2
  let black : ℚ := 1/4
  let gray : ℚ := 1/8
  let blue : ℚ := 1/8
  white + black + gray + blue = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_parts_sum_to_one_l3423_342319


namespace NUMINAMATH_CALUDE_solve_for_y_l3423_342367

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3423_342367


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3423_342317

/-- Given two points on a line, prove that the sum of the slope and y-intercept is 3 -/
theorem line_slope_intercept_sum (x₁ y₁ x₂ y₂ m b : ℝ) : 
  x₁ = 1 → y₁ = 3 → x₂ = -3 → y₂ = -1 →
  (y₂ - y₁) = m * (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3423_342317


namespace NUMINAMATH_CALUDE_gerald_furniture_problem_l3423_342372

/-- Represents the problem of determining the maximum number of chairs Gerald can make --/
theorem gerald_furniture_problem 
  (x t c b : ℕ) 
  (r1 r2 r3 : ℕ) 
  (h_x : x = 2250)
  (h_t : t = 18)
  (h_c : c = 12)
  (h_b : b = 30)
  (h_ratio : r1 = 2 ∧ r2 = 3 ∧ r3 = 1) :
  ∃ (chairs : ℕ), 
    chairs ≤ (x / (t * r1 / r2 + c + b * r3 / r2)) ∧ 
    chairs = 66 := by
  sorry


end NUMINAMATH_CALUDE_gerald_furniture_problem_l3423_342372


namespace NUMINAMATH_CALUDE_parabola_translation_l3423_342301

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  vertical_shift (horizontal_shift original_parabola 2) (-3)

theorem parabola_translation :
  resulting_parabola.f = fun x => (x + 2)^2 - 3 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3423_342301


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l3423_342300

/-- The carpet area required for a rectangular room -/
def carpet_area (length width : ℝ) (wastage_factor : ℝ) : ℝ :=
  length * width * (1 + wastage_factor)

/-- Theorem: The carpet area for a 15 ft by 9 ft room with 10% wastage is 148.5 sq ft -/
theorem carpet_area_calculation :
  carpet_area 15 9 0.1 = 148.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l3423_342300


namespace NUMINAMATH_CALUDE_monomial_coefficient_degree_product_l3423_342374

/-- 
Given a monomial of the form $-\frac{3}{4}{x^2}{y^2}$, 
this theorem proves that the product of its coefficient and degree is -3.
-/
theorem monomial_coefficient_degree_product : 
  ∃ (m n : ℚ), (m = -3/4) ∧ (n = 4) ∧ (m * n = -3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_coefficient_degree_product_l3423_342374


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3423_342368

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

/-- The discriminant of the conic section -/
def discriminant : ℝ :=
  0^2 - 4 * 4 * (-9)

theorem conic_is_hyperbola :
  discriminant > 0 ∧ 
  (∃ a b c d : ℝ, ∀ x y : ℝ, 
    conic_equation x y ↔ ((x - a)^2 / b^2) - ((y - c)^2 / d^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3423_342368


namespace NUMINAMATH_CALUDE_second_hand_movement_l3423_342355

/-- Represents the movement of clock hands -/
def ClockMovement : Type :=
  { minutes : ℕ // minutes > 0 }

/-- Converts minutes to seconds -/
def minutesToSeconds (m : ClockMovement) : ℕ :=
  m.val * 60

/-- Calculates the number of circles the second hand moves -/
def secondHandCircles (m : ClockMovement) : ℕ :=
  minutesToSeconds m / 60

/-- The theorem to be proved -/
theorem second_hand_movement (m : ClockMovement) (h : m.val = 2) :
  secondHandCircles m = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_hand_movement_l3423_342355


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3423_342328

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3423_342328


namespace NUMINAMATH_CALUDE_xiao_ming_exam_probabilities_l3423_342377

/-- Represents the probabilities of scoring in different ranges in a math exam -/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring above 80 -/
def probAbove80 (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of passing the exam (scoring above 60) -/
def probPassing (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89 + p.between70and79 + p.between60and69

/-- Theorem stating the probabilities for Xiao Ming's math exam -/
theorem xiao_ming_exam_probabilities (p : ExamProbabilities)
    (h1 : p.above90 = 0.18)
    (h2 : p.between80and89 = 0.51)
    (h3 : p.between70and79 = 0.15)
    (h4 : p.between60and69 = 0.09) :
    probAbove80 p = 0.69 ∧ probPassing p = 0.93 := by
  sorry


end NUMINAMATH_CALUDE_xiao_ming_exam_probabilities_l3423_342377


namespace NUMINAMATH_CALUDE_friend_bike_speed_l3423_342310

/-- Proves that given Joann's speed and time, Fran's speed can be calculated for the same distance --/
theorem friend_bike_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) :
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check friend_bike_speed

end NUMINAMATH_CALUDE_friend_bike_speed_l3423_342310


namespace NUMINAMATH_CALUDE_square_of_simplified_fraction_l3423_342318

theorem square_of_simplified_fraction : 
  (126 / 882 : ℚ)^2 = 1 / 49 := by sorry

end NUMINAMATH_CALUDE_square_of_simplified_fraction_l3423_342318


namespace NUMINAMATH_CALUDE_trajectory_intersection_properties_l3423_342330

-- Define the trajectory of point M
def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) = |x| + 1

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  y = x + 1

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x - 1)

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (trajectory A.1 A.2 ∧ line_l2 A.1 A.2) ∧
    (trajectory B.1 B.2 ∧ line_l2 B.1 B.2) ∧
    (A ≠ B) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) ∧
    (Real.sqrt ((A.1 - point_F.1)^2 + (A.2 - point_F.2)^2) *
     Real.sqrt ((B.1 - point_F.1)^2 + (B.2 - point_F.2)^2) = 16) :=
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_properties_l3423_342330


namespace NUMINAMATH_CALUDE_hundred_with_five_threes_l3423_342394

-- Define a custom type for our arithmetic expressions
inductive Expr
  | const : ℕ → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.const n => if n = 3 then 1 else 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.const n => n
  | Expr.add e1 e2 => evaluate e1 + evaluate e2
  | Expr.sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_five_threes : 
  ∃ e : Expr, countThrees e = 5 ∧ evaluate e = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_five_threes_l3423_342394


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3423_342340

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 73 % 103 ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3423_342340


namespace NUMINAMATH_CALUDE_dog_bones_proof_l3423_342304

/-- The number of bones the dog dug up -/
def bones_dug_up : ℕ := 367

/-- The total number of bones the dog has now -/
def total_bones_now : ℕ := 860

/-- The initial number of bones the dog had -/
def initial_bones : ℕ := total_bones_now - bones_dug_up

theorem dog_bones_proof : initial_bones = 493 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_proof_l3423_342304


namespace NUMINAMATH_CALUDE_empty_set_is_proposition_l3423_342385

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s = "true") ∨ (s = "false")

-- The statement we want to prove is a proposition
def empty_set_statement : String := "The empty set is a subset of any set"

-- Theorem statement
theorem empty_set_is_proposition : is_proposition empty_set_statement := by
  sorry


end NUMINAMATH_CALUDE_empty_set_is_proposition_l3423_342385


namespace NUMINAMATH_CALUDE_increasing_sufficient_not_necessary_l3423_342359

/-- A function f: ℝ → ℝ is increasing on [1, +∞) -/
def IncreasingOnIntervalOneInf (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f x < f y

/-- A sequence a_n = f(n) is increasing -/
def IncreasingSequence (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → f n < f (n + 1)

/-- The main theorem stating that IncreasingOnIntervalOneInf is sufficient but not necessary for IncreasingSequence -/
theorem increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (IncreasingOnIntervalOneInf f → IncreasingSequence f) ∧
  ∃ g : ℝ → ℝ, IncreasingSequence g ∧ ¬IncreasingOnIntervalOneInf g :=
sorry

end NUMINAMATH_CALUDE_increasing_sufficient_not_necessary_l3423_342359


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l3423_342373

/-- Given that Alice takes 25 minutes to clean her room and Bob takes 2/5 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) 
  (h1 : alice_time = 25)
  (h2 : bob_fraction = 2 / 5) :
  bob_fraction * alice_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l3423_342373


namespace NUMINAMATH_CALUDE_smallest_common_divisor_l3423_342393

theorem smallest_common_divisor : ∃ (x : ℕ), 
  x - 16 = 136 ∧ 
  (∀ d : ℕ, d > 0 ∧ d ∣ 136 ∧ d ∣ 6 ∧ d ∣ 8 ∧ d ∣ 10 → d ≥ 2) ∧
  2 ∣ 136 ∧ 2 ∣ 6 ∧ 2 ∣ 8 ∧ 2 ∣ 10 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_common_divisor_l3423_342393


namespace NUMINAMATH_CALUDE_angela_finished_nine_problems_l3423_342339

/-- The number of math problems Angela and her friends are working on -/
def total_problems : ℕ := 20

/-- The number of problems Martha has finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna has finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Mark has finished -/
def mark_problems : ℕ := jenna_problems / 2

/-- The number of problems Angela has finished on her own -/
def angela_problems : ℕ := total_problems - (martha_problems + jenna_problems + mark_problems)

theorem angela_finished_nine_problems : angela_problems = 9 := by
  sorry

end NUMINAMATH_CALUDE_angela_finished_nine_problems_l3423_342339


namespace NUMINAMATH_CALUDE_parabola_smallest_a_l3423_342343

theorem parabola_smallest_a (a b c : ℝ) : 
  a > 0 ∧ 
  b^2 - 4*a*c = 7 ∧
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = a*(x - 1/3)^2) →
  a ≥ 63/20 ∧ ∃ b c : ℝ, b^2 - 4*a*c = 7 ∧ (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = 63/20*(x - 1/3)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_smallest_a_l3423_342343


namespace NUMINAMATH_CALUDE_complex_multiply_i_l3423_342324

theorem complex_multiply_i (i : ℂ) : i * i = -1 → (1 + i) * i = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiply_i_l3423_342324


namespace NUMINAMATH_CALUDE_mask_usage_duration_l3423_342390

theorem mask_usage_duration (total_masks : ℕ) (family_members : ℕ) (total_days : ℕ) 
  (h1 : total_masks = 100)
  (h2 : family_members = 5)
  (h3 : total_days = 80) :
  (total_masks : ℚ) / total_days / family_members = 1 / 4 := by
  sorry

#check mask_usage_duration

end NUMINAMATH_CALUDE_mask_usage_duration_l3423_342390


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l3423_342397

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∃ n : ℕ, 4 * b + 5 = n * n) → b ≥ 11 :=
by
  sorry

theorem base_11_perfect_square : 
  ∃ n : ℕ, 4 * 11 + 5 = n * n :=
by
  sorry

theorem eleven_is_smallest : 
  ∀ b : ℕ, b > 5 ∧ b < 11 → ¬(∃ n : ℕ, 4 * b + 5 = n * n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l3423_342397


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3423_342332

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3423_342332


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3423_342369

theorem circle_area_from_circumference (circumference : ℝ) (area : ℝ) :
  circumference = 18 →
  area = 81 / Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3423_342369


namespace NUMINAMATH_CALUDE_cupboard_pricing_l3423_342313

/-- The cost price of a cupboard --/
def C : ℝ := sorry

/-- The selling price of the first cupboard --/
def SP₁ : ℝ := 0.84 * C

/-- The selling price of the second cupboard before tax --/
def SP₂ : ℝ := 0.756 * C

/-- The final selling price of the second cupboard after tax --/
def SP₂' : ℝ := 0.82404 * C

/-- The theorem stating the relationship between the cost price and the selling prices --/
theorem cupboard_pricing :
  2.32 * C - (SP₁ + SP₂') = 1800 :=
sorry

end NUMINAMATH_CALUDE_cupboard_pricing_l3423_342313


namespace NUMINAMATH_CALUDE_absolute_sum_nonzero_iff_either_nonzero_l3423_342360

theorem absolute_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  |x| + |y| ≠ 0 ↔ x ≠ 0 ∨ y ≠ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_sum_nonzero_iff_either_nonzero_l3423_342360


namespace NUMINAMATH_CALUDE_profit_range_max_avg_profit_l3423_342384

/-- Cumulative profit function -/
def profit (x : ℕ) : ℚ :=
  -1/2 * x^2 + 60*x - 800

/-- Average daily profit function -/
def avgProfit (x : ℕ) : ℚ :=
  profit x / x

theorem profit_range (x : ℕ) (hx : x > 0) :
  profit x > 800 ↔ x > 40 ∧ x < 80 :=
sorry

theorem max_avg_profit :
  ∃ (x : ℕ), x > 0 ∧ ∀ (y : ℕ), y > 0 → avgProfit x ≥ avgProfit y ∧ x = 400 :=
sorry

end NUMINAMATH_CALUDE_profit_range_max_avg_profit_l3423_342384


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3423_342361

-- Define the condition for an ellipse
def is_ellipse (a b : ℝ) : Prop := ∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ a > 0 ∧ b > 0

-- Theorem stating that ab > 0 is necessary but not sufficient for an ellipse
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l3423_342361


namespace NUMINAMATH_CALUDE_triangle_properties_l3423_342341

theorem triangle_properties (a b c A B C : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_sides : a = 2)
  (h_equation : (b + 2) * (Real.sin A - Real.sin B) = c * (Real.sin B + Real.sin C)) :
  A = 2 * π / 3 ∧
  ∃ S : ℝ, S > 0 ∧ S ≤ Real.sqrt 3 / 3 ∧
    S = 1 / 2 * a * b * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3423_342341


namespace NUMINAMATH_CALUDE_concert_ticket_cost_daria_concert_money_l3423_342302

theorem concert_ticket_cost (ticket_price : ℕ) (current_money : ℕ) : ℕ :=
  let total_tickets : ℕ := 4
  let total_cost : ℕ := total_tickets * ticket_price
  let additional_money_needed : ℕ := total_cost - current_money
  additional_money_needed

theorem daria_concert_money : concert_ticket_cost 90 189 = 171 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_daria_concert_money_l3423_342302


namespace NUMINAMATH_CALUDE_land_division_theorem_l3423_342303

/-- Represents a rectangular piece of land --/
structure Land where
  length : ℝ
  width : ℝ

/-- Represents a division of land into three sections --/
structure LandDivision where
  section1 : Land
  section2 : Land
  section3 : Land

def Land.area (l : Land) : ℝ := l.length * l.width

def LandDivision.isValid (ld : LandDivision) (totalLand : Land) : Prop :=
  ld.section1.area + ld.section2.area + ld.section3.area = totalLand.area ∧
  ld.section1.area = ld.section2.area ∧
  ld.section2.area = ld.section3.area

def LandDivision.fenceLength (ld : LandDivision) : ℝ :=
  ld.section1.length + ld.section2.length + ld.section3.length

def countValidDivisions (totalLand : Land) : ℕ :=
  sorry

def minFenceLength (totalLand : Land) : ℝ :=
  sorry

theorem land_division_theorem (totalLand : Land) 
  (h1 : totalLand.length = 25)
  (h2 : totalLand.width = 36) :
  countValidDivisions totalLand = 4 ∧ 
  minFenceLength totalLand = 49 := by
  sorry

end NUMINAMATH_CALUDE_land_division_theorem_l3423_342303


namespace NUMINAMATH_CALUDE_abs_diff_one_if_sum_one_l3423_342354

theorem abs_diff_one_if_sum_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_one_if_sum_one_l3423_342354


namespace NUMINAMATH_CALUDE_queens_bounding_rectangle_l3423_342334

theorem queens_bounding_rectangle (a : Fin 2004 → Fin 2004) 
  (h_perm : Function.Bijective a) 
  (h_diag : ∀ i j : Fin 2004, i ≠ j → |a i - a j| ≠ |i - j|) :
  ∃ i j : Fin 2004, |i - j| + |a i - a j| = 2004 := by
  sorry

end NUMINAMATH_CALUDE_queens_bounding_rectangle_l3423_342334


namespace NUMINAMATH_CALUDE_total_population_proof_l3423_342387

def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

def greenville_population : ℕ := springfield_population - population_difference
def oakville_population : ℕ := 2 * population_difference

def total_population : ℕ := springfield_population + greenville_population + oakville_population

theorem total_population_proof : total_population = 1084972 := by
  sorry

end NUMINAMATH_CALUDE_total_population_proof_l3423_342387


namespace NUMINAMATH_CALUDE_absolute_value_sum_simplification_l3423_342383

theorem absolute_value_sum_simplification (x : ℝ) : 
  |x - 1| + |x - 2| + |x + 3| = 
    if x < -3 then -3*x
    else if x < 1 then 6 - x
    else if x < 2 then 4 + x
    else 3*x := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_simplification_l3423_342383


namespace NUMINAMATH_CALUDE_planes_perpendicular_if_line_perpendicular_and_parallel_l3423_342370

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_if_line_perpendicular_and_parallel
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_if_line_perpendicular_and_parallel_l3423_342370


namespace NUMINAMATH_CALUDE_milk_delivery_solution_l3423_342375

/-- Represents the milk delivery problem --/
def MilkDeliveryProblem (jarsPerCarton : ℕ) : Prop :=
  let usualCartons : ℕ := 50
  let actualCartons : ℕ := usualCartons - 20
  let damagedJarsInFiveCartons : ℕ := 5 * 3
  let totalDamagedJars : ℕ := damagedJarsInFiveCartons + jarsPerCarton
  let goodJars : ℕ := 565
  actualCartons * jarsPerCarton - totalDamagedJars = goodJars

/-- Theorem stating that the solution to the milk delivery problem is 20 jars per carton --/
theorem milk_delivery_solution : MilkDeliveryProblem 20 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_solution_l3423_342375


namespace NUMINAMATH_CALUDE_well_depth_is_30_l3423_342338

/-- The depth of a well that a man climbs out of in 27 days -/
def well_depth (daily_climb : ℕ) (daily_slip : ℕ) (total_days : ℕ) (final_climb : ℕ) : ℕ :=
  (total_days - 1) * (daily_climb - daily_slip) + final_climb

/-- Theorem stating the depth of the well is 30 meters -/
theorem well_depth_is_30 :
  well_depth 4 3 27 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_is_30_l3423_342338


namespace NUMINAMATH_CALUDE_harmonious_expressions_l3423_342380

-- Define the concept of a harmonious algebraic expression
def is_harmonious (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b ∧
  ∃ y ∈ Set.Icc a b, ∀ z ∈ Set.Icc a b, f y ≥ f z

-- Theorem statement
theorem harmonious_expressions :
  let a := -2
  let b := 2
  -- Part 1
  ¬ is_harmonious (fun x => |x - 1|) a b ∧
  -- Part 2
  ¬ is_harmonious (fun x => -x + 1) a b ∧
  is_harmonious (fun x => -x^2 + 2) a b ∧
  ¬ is_harmonious (fun x => x^2 + |x| - 4) a b ∧
  -- Part 3
  ∀ c : ℝ, is_harmonious (fun x => c / (|x| + 1) - 2) a b ↔ (0 ≤ c ∧ c ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_harmonious_expressions_l3423_342380


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3423_342358

/-- A regular polygon with perimeter 150 and side length 15 has 10 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_regular : p ≥ 3)
  (h_perimeter : perimeter = 150)
  (h_side : side_length = 15)
  (h_relation : perimeter = p * side_length) : p = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3423_342358


namespace NUMINAMATH_CALUDE_subset_sum_property_l3423_342356

theorem subset_sum_property (n : ℕ) (A B C : Finset ℕ) :
  (∀ i ∈ A ∪ B ∪ C, i ≤ 3*n) →
  A.card = n →
  B.card = n →
  C.card = n →
  (A ∩ B ∩ C).card = 0 →
  (A ∪ B ∪ C).card = 3*n →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ a + b = c :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_property_l3423_342356


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3423_342315

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 3 ↔ x - a < 1 ∧ x - 2*b > 3) → 
  a = 2 ∧ b = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3423_342315


namespace NUMINAMATH_CALUDE_rectangle_short_side_l3423_342346

/-- Proves that for a rectangle with perimeter 38 cm and long side 12 cm, the short side is 7 cm. -/
theorem rectangle_short_side (perimeter long_side short_side : ℝ) : 
  perimeter = 38 ∧ long_side = 12 ∧ perimeter = 2 * long_side + 2 * short_side → short_side = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_short_side_l3423_342346


namespace NUMINAMATH_CALUDE_lassis_from_twelve_mangoes_l3423_342329

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (11 * mangoes) / 2

/-- Theorem stating that Caroline can make 66 lassis from 12 mangoes -/
theorem lassis_from_twelve_mangoes :
  lassis_from_mangoes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_twelve_mangoes_l3423_342329


namespace NUMINAMATH_CALUDE_nine_point_circle_triangles_l3423_342327

/-- Given 9 points on a circle, this function calculates the number of triangles
    formed by the intersections of chords inside the circle. -/
def triangles_in_circle (n : ℕ) : ℕ :=
  if n = 9 then
    (Nat.choose n 6) * (Nat.choose 6 2) * (Nat.choose 4 2) / 6
  else
    0

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair
    of points and no three chords intersecting at a single point inside the circle,
    the number of triangles formed with all vertices in the interior is 210. -/
theorem nine_point_circle_triangles :
  triangles_in_circle 9 = 210 := by
  sorry

#eval triangles_in_circle 9

end NUMINAMATH_CALUDE_nine_point_circle_triangles_l3423_342327


namespace NUMINAMATH_CALUDE_distinct_triangles_in_regular_ngon_l3423_342353

theorem distinct_triangles_in_regular_ngon (n : ℕ) :
  Nat.choose n 3 = (n * (n - 1) * (n - 2)) / 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_regular_ngon_l3423_342353


namespace NUMINAMATH_CALUDE_tournament_outcomes_l3423_342306

/-- Represents a bowler in the tournament -/
inductive Bowler
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents a match between two bowlers -/
structure Match where
  player1 : Bowler
  player2 : Bowler

/-- Represents the tournament structure -/
structure Tournament where
  initialRound : List Match
  subsequentRounds : List Match

/-- Represents the outcome of the tournament -/
structure Outcome where
  prizeOrder : List Bowler

/-- The number of possible outcomes for the tournament -/
def numberOfOutcomes (t : Tournament) : Nat :=
  2^5

/-- Theorem stating that the number of possible outcomes is 32 -/
theorem tournament_outcomes (t : Tournament) :
  numberOfOutcomes t = 32 := by
  sorry

end NUMINAMATH_CALUDE_tournament_outcomes_l3423_342306


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3423_342316

theorem solve_linear_equation (x : ℤ) : 9823 + x = 13200 → x = 3377 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3423_342316


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l3423_342378

/-- Calculates the time required to remove wallpaper from remaining walls -/
def time_to_remove_wallpaper (time_per_wall : ℕ) (dining_room_walls : ℕ) (living_room_walls : ℕ) (walls_completed : ℕ) : ℕ :=
  time_per_wall * (dining_room_walls + living_room_walls - walls_completed)

/-- Proves that given the conditions, the time required to remove the remaining wallpaper is 14 hours -/
theorem wallpaper_removal_time :
  let time_per_wall : ℕ := 2
  let dining_room_walls : ℕ := 4
  let living_room_walls : ℕ := 4
  let walls_completed : ℕ := 1
  time_to_remove_wallpaper time_per_wall dining_room_walls living_room_walls walls_completed = 14 := by
  sorry

#eval time_to_remove_wallpaper 2 4 4 1

end NUMINAMATH_CALUDE_wallpaper_removal_time_l3423_342378


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3423_342337

def biased_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem three_heads_in_eight_tosses :
  biased_coin_probability 8 3 (1/3) = 1792/6561 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3423_342337


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_sum_27_l3423_342349

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ sum_of_digits n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ sum_of_digits m = 27 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_sum_27_l3423_342349


namespace NUMINAMATH_CALUDE_mike_books_before_sale_l3423_342345

def books_before_sale (books_bought books_after : ℕ) : ℕ :=
  books_after - books_bought

theorem mike_books_before_sale :
  books_before_sale 21 56 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mike_books_before_sale_l3423_342345


namespace NUMINAMATH_CALUDE_equal_projections_implies_a_equals_one_l3423_342376

-- Define the points and vectors
def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)

def OA (a : ℝ) : ℝ × ℝ := A a
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem equal_projections_implies_a_equals_one (a : ℝ) :
  dot_product (OA a) OC = dot_product OB OC → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_projections_implies_a_equals_one_l3423_342376


namespace NUMINAMATH_CALUDE_matrix_power_sum_l3423_342366

/-- Given a matrix B and its mth power, prove that b + m = 381 -/
theorem matrix_power_sum (b m : ℕ) : 
  let B : Matrix (Fin 3) (Fin 3) ℕ := !![1, 3, b; 0, 1, 5; 0, 0, 1]
  let B_pow_m : Matrix (Fin 3) (Fin 3) ℕ := !![1, 33, 4054; 0, 1, 55; 0, 0, 1]
  B^m = B_pow_m → b + m = 381 := by
sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l3423_342366


namespace NUMINAMATH_CALUDE_sequence_property_l3423_342348

/-- Two infinite sequences of rational numbers -/
def Sequence := ℕ → ℚ

/-- Property that a sequence is nonconstant -/
def Nonconstant (s : Sequence) : Prop :=
  ∃ i j, s i ≠ s j

/-- Property that (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer for all i and j -/
def IntegerProduct (s t : Sequence) : Prop :=
  ∀ i j, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

theorem sequence_property (s t : Sequence) 
  (hs : Nonconstant s) (ht : Nonconstant t) (h : IntegerProduct s t) :
  ∃ r : ℚ, (∀ i j : ℕ, ∃ m n : ℤ, (s i - s j) * r = m ∧ (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l3423_342348


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3423_342344

theorem necessary_but_not_sufficient_condition :
  ∃ (x : ℝ), ((-2 < x ∧ x < 3) ∧ ¬(x^2 - 2*x - 3 < 0)) ∧
  ∀ (y : ℝ), (y^2 - 2*y - 3 < 0) → (-2 < y ∧ y < 3) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3423_342344


namespace NUMINAMATH_CALUDE_distance_acaster_beetown_is_315_l3423_342322

/-- The distance from Acaster to Beetown in kilometers. -/
def distance_acaster_beetown : ℝ := 315

/-- Lewis's speed in km/h. -/
def lewis_speed : ℝ := 70

/-- Geraint's speed in km/h. -/
def geraint_speed : ℝ := 30

/-- The distance from the meeting point to Beetown in kilometers. -/
def distance_meeting_beetown : ℝ := 105

/-- The time Lewis spends in Beetown in hours. -/
def lewis_stop_time : ℝ := 1

theorem distance_acaster_beetown_is_315 :
  let total_time := distance_acaster_beetown / geraint_speed
  let lewis_travel_time := total_time - lewis_stop_time
  lewis_travel_time * lewis_speed = distance_acaster_beetown + distance_meeting_beetown ∧
  total_time * geraint_speed = distance_acaster_beetown - distance_meeting_beetown ∧
  distance_acaster_beetown = 315 := by
  sorry

#check distance_acaster_beetown_is_315

end NUMINAMATH_CALUDE_distance_acaster_beetown_is_315_l3423_342322


namespace NUMINAMATH_CALUDE_smallest_m_theorem_l3423_342320

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (m : ℕ) : Prop :=
  is_multiple_of_100 m ∧ count_divisors m = 100

theorem smallest_m_theorem :
  ∃! m : ℕ, satisfies_conditions m ∧
    ∀ n : ℕ, satisfies_conditions n → m ≤ n ∧
    m / 100 = 2700 := by sorry

end NUMINAMATH_CALUDE_smallest_m_theorem_l3423_342320


namespace NUMINAMATH_CALUDE_floor_length_is_twelve_l3423_342305

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 12 meters -/
theorem floor_length_is_twelve (floor : FloorWithRug) 
  (h1 : floor.width = 10)
  (h2 : floor.strip_width = 3)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_is_twelve_l3423_342305


namespace NUMINAMATH_CALUDE_green_hats_count_l3423_342357

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_price : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_price = 548 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_price ∧
    green_hats = 38 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l3423_342357


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l3423_342371

/-- The ellipse with equation x²/9 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- The left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- The dot product of vectors EF₁ and EF₂ -/
def dotProduct (E : ℝ × ℝ) : ℝ :=
  let (x, y) := E
  (-1-x)*(1-x) + (-y)*(-y)

theorem ellipse_dot_product_bounds :
  ∀ E : ℝ × ℝ, Ellipse E.1 E.2 → 7 ≤ dotProduct E ∧ dotProduct E ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l3423_342371


namespace NUMINAMATH_CALUDE_total_amount_divided_l3423_342333

/-- Proves that the total amount divided is 3500, given the specified conditions --/
theorem total_amount_divided (first_part : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (total_interest : ℝ) :
  first_part = 1550 →
  interest_rate1 = 0.03 →
  interest_rate2 = 0.05 →
  total_interest = 144 →
  ∃ (total : ℝ), 
    total = 3500 ∧
    first_part * interest_rate1 + (total - first_part) * interest_rate2 = total_interest :=
by
  sorry


end NUMINAMATH_CALUDE_total_amount_divided_l3423_342333


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3423_342363

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The condition that x = 2 is sufficient but not necessary for parallelism -/
theorem parallel_sufficient_not_necessary (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  (x = 2 → are_parallel a b) ∧
  ¬(are_parallel a b → x = 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l3423_342363


namespace NUMINAMATH_CALUDE_barycentric_centroid_vector_relation_l3423_342314

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC, a point X with absolute barycentric coordinates (α:β:γ),
    and M as the centroid of triangle ABC, prove that:
    3 XM⃗ = (α - β)AB⃗ + (β - γ)BC⃗ + (γ - α)CA⃗ -/
theorem barycentric_centroid_vector_relation
  (A B C X M : V) (α β γ : ℝ) :
  X = α • A + β • B + γ • C →
  M = (1/3 : ℝ) • (A + B + C) →
  3 • (X - M) = (α - β) • (B - A) + (β - γ) • (C - B) + (γ - α) • (A - C) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_centroid_vector_relation_l3423_342314


namespace NUMINAMATH_CALUDE_tangent_parallel_range_l3423_342388

open Real

/-- The function f(x) = x(m - e^(-2x)) --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m - Real.exp (-2 * x))

/-- The derivative of f with respect to x --/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m - (1 - 2*x) * Real.exp (-2 * x)

/-- Theorem stating the range of m for which there exist two distinct points
    on the curve y = f(x) where the tangent lines are parallel to y = x --/
theorem tangent_parallel_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv m x₁ = 1 ∧ f_deriv m x₂ = 1) ↔ 
  (1 - Real.exp (-2) < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_range_l3423_342388


namespace NUMINAMATH_CALUDE_unique_solution_l3423_342398

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a number satisfies the first division scheme -/
def satisfies_first_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ 
  (n.value / d = 10 + (n.value / 100 % 10)) ∧
  (n.value % d = (n.value / 10 % 10) * 10 + (n.value % 10))

/-- Checks if a number satisfies the second division scheme -/
def satisfies_second_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧
  (n.value / d = 168) ∧
  (n.value % d = 0)

/-- The main theorem stating that 1512 is the only number satisfying both schemes -/
theorem unique_solution : 
  ∃! (n : FourDigitNumber), 
    satisfies_first_scheme n ∧ 
    satisfies_second_scheme n ∧ 
    n.value = 1512 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3423_342398


namespace NUMINAMATH_CALUDE_power_multiplication_l3423_342364

theorem power_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3423_342364


namespace NUMINAMATH_CALUDE_perpendicular_vector_proof_l3423_342352

def line_direction : ℝ × ℝ := (3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vector_proof (v : ℝ × ℝ) :
  is_perpendicular v line_direction ∧ v.1 + v.2 = 1 → v = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_proof_l3423_342352


namespace NUMINAMATH_CALUDE_max_cosine_difference_value_l3423_342309

def max_cosine_difference (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  a₃ = a₂ + a₁ ∧ 
  a₄ = a₃ + a₂ ∧ 
  ∃ (a b c : ℝ), ∀ n ∈ ({1, 2, 3, 4} : Set ℕ), 
    a * n^2 + b * n + c = Real.cos (if n = 1 then a₁ 
                                    else if n = 2 then a₂ 
                                    else if n = 3 then a₃ 
                                    else a₄)

theorem max_cosine_difference_value :
  ∀ a₁ a₂ a₃ a₄ : ℝ, max_cosine_difference a₁ a₂ a₃ a₄ →
    Real.cos a₁ - Real.cos a₄ ≤ -9 + 3 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_cosine_difference_value_l3423_342309


namespace NUMINAMATH_CALUDE_angle_sequence_convergence_l3423_342335

noncomputable def angle_sequence (α : ℝ) : ℕ → ℝ
  | 0 => 0  -- Initial value doesn't affect the limit
  | n + 1 => (Real.pi - α - angle_sequence α n) / 2

theorem angle_sequence_convergence (α : ℝ) (h : 0 < α ∧ α < Real.pi) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |angle_sequence α n - L| < ε) ∧
             L = (Real.pi - α) / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_sequence_convergence_l3423_342335


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3423_342325

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  0 < y → y ≤ 10 →
  ∃ (a' b' c' : ℕ), is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (a' * 100 + b' * 10 + c' : ℚ) / 1000 = 1 / y ∧
    a' + b' + c' ≤ 8 ∧
    a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3423_342325


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l3423_342323

theorem x_positive_sufficient_not_necessary_for_abs_x_positive :
  (∀ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l3423_342323


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l3423_342326

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) :
  square_perimeter = 80 →
  triangle_height = 40 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * (square_perimeter / 4) →
  (square_perimeter / 4) = 20 :=
by sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l3423_342326


namespace NUMINAMATH_CALUDE_natural_numbers_difference_l3423_342382

theorem natural_numbers_difference (a b : ℕ) : 
  a + b = 20250 → 
  b % 15 = 0 → 
  a = b / 3 → 
  b - a = 10130 := by
sorry

end NUMINAMATH_CALUDE_natural_numbers_difference_l3423_342382


namespace NUMINAMATH_CALUDE_leonardo_sleep_fraction_l3423_342392

-- Define the number of minutes in an hour
def minutes_in_hour : ℕ := 60

-- Define Leonardo's sleep duration in minutes
def leonardo_sleep_minutes : ℕ := 12

-- Theorem to prove
theorem leonardo_sleep_fraction :
  (leonardo_sleep_minutes : ℚ) / minutes_in_hour = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_leonardo_sleep_fraction_l3423_342392


namespace NUMINAMATH_CALUDE_no_equal_perimeter_area_volume_cuboid_l3423_342342

theorem no_equal_perimeter_area_volume_cuboid :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a + b + c) = 2 * (a * b + b * c + c * a)) ∧
    (4 * (a + b + c) = a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_perimeter_area_volume_cuboid_l3423_342342


namespace NUMINAMATH_CALUDE_chord_length_is_three_l3423_342312

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passing through the focus and perpendicular to x-axis -/
def line (x : ℝ) : Prop := x = (focus.1)

/-- The chord length -/
def chord_length : ℝ := 3

/-- Theorem stating that the chord length cut by the line passing through
    the focus of the ellipse and perpendicular to the x-axis is equal to 3 -/
theorem chord_length_is_three :
  ∀ y₁ y₂ : ℝ,
  ellipse (focus.1) y₁ ∧ ellipse (focus.1) y₂ ∧ y₁ ≠ y₂ →
  |y₁ - y₂| = chord_length :=
sorry

end NUMINAMATH_CALUDE_chord_length_is_three_l3423_342312


namespace NUMINAMATH_CALUDE_f_properties_l3423_342379

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_period : ∀ x, f (x + 6) = f x + f 3
axiom f_increasing_on_0_3 : ∀ x₁ x₂, x₁ ∈ Set.Icc 0 3 → x₂ ∈ Set.Icc 0 3 → x₁ ≠ x₂ → 
  (f x₁ - f x₂) / (x₁ - x₂) > 0

-- Theorem to prove
theorem f_properties :
  (∀ x, f (x - 6) = f (-x)) ∧ 
  (¬ ∀ x₁ x₂, x₁ ∈ Set.Icc (-9) (-6) → x₂ ∈ Set.Icc (-9) (-6) → x₁ < x₂ → f x₁ < f x₂) ∧
  (¬ ∃ x₁ x₂ x₃ x₄ x₅, x₁ ∈ Set.Icc (-9) 9 ∧ x₂ ∈ Set.Icc (-9) 9 ∧ x₃ ∈ Set.Icc (-9) 9 ∧ 
    x₄ ∈ Set.Icc (-9) 9 ∧ x₅ ∈ Set.Icc (-9) 9 ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ 
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ 
    x₄ ≠ x₅) :=
by
  sorry


end NUMINAMATH_CALUDE_f_properties_l3423_342379


namespace NUMINAMATH_CALUDE_books_in_box_l3423_342351

def box_weight : ℕ := 42
def book_weight : ℕ := 3

theorem books_in_box : 
  box_weight / book_weight = 14 := by sorry

end NUMINAMATH_CALUDE_books_in_box_l3423_342351


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l3423_342386

theorem simplify_radical_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l3423_342386


namespace NUMINAMATH_CALUDE_evaluate_expression_l3423_342321

theorem evaluate_expression : (30 - (3030 - 303)) * (3030 - (303 - 30)) = -7435969 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3423_342321


namespace NUMINAMATH_CALUDE_fibonacci_closed_form_l3423_342391

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_closed_form_l3423_342391


namespace NUMINAMATH_CALUDE_order_of_abc_l3423_342399

theorem order_of_abc : ∀ (a b c : ℝ), 
  a = Real.exp 0.25 → 
  b = 1 → 
  c = -4 * Real.log 0.75 → 
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3423_342399


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3423_342365

theorem rectangle_dimension_change (x : ℝ) : 
  (1 + x / 100) * (1 - 5 / 100) = 1 + 14.000000000000002 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3423_342365


namespace NUMINAMATH_CALUDE_warehouse_length_calculation_l3423_342396

/-- Represents the dimensions and walking pattern around a rectangular warehouse. -/
structure Warehouse :=
  (width : ℝ)
  (length : ℝ)
  (circles : ℕ)
  (total_distance : ℝ)

/-- Theorem stating the length of the warehouse given specific conditions. -/
theorem warehouse_length_calculation (w : Warehouse) 
  (h1 : w.width = 400)
  (h2 : w.circles = 8)
  (h3 : w.total_distance = 16000)
  : w.length = 600 := by
  sorry

#check warehouse_length_calculation

end NUMINAMATH_CALUDE_warehouse_length_calculation_l3423_342396


namespace NUMINAMATH_CALUDE_courtyard_breadth_l3423_342350

/-- Proves that the breadth of a rectangular courtyard is 6 meters -/
theorem courtyard_breadth : 
  ∀ (length width stone_length stone_width stone_count : ℝ),
  length = 15 →
  stone_count = 15 →
  stone_length = 3 →
  stone_width = 2 →
  length * width = stone_count * stone_length * stone_width →
  width = 6 := by
sorry

end NUMINAMATH_CALUDE_courtyard_breadth_l3423_342350


namespace NUMINAMATH_CALUDE_expression_equalities_l3423_342395

theorem expression_equalities : 
  (1 / (Real.sqrt 2 - 1) + Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 6) + Real.sqrt 8 = 4) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_equalities_l3423_342395


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l3423_342381

theorem clothing_tax_rate 
  (total : ℝ) 
  (clothing_spend : ℝ) 
  (food_spend : ℝ) 
  (other_spend : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  clothing_spend = 0.6 * total →
  food_spend = 0.1 * total →
  other_spend = 0.3 * total →
  other_tax_rate = 0.08 →
  total_tax_rate = 0.048 →
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_spend + other_tax_rate * other_spend = total_tax_rate * total ∧
    clothing_tax_rate = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l3423_342381


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l3423_342308

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_cond : x + y = 10) 
  (diff_cond : |x - y| = 12) : 
  (∀ z : ℝ, (z - x) * (z - y) = 0 ↔ z^2 - 10*z - 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l3423_342308


namespace NUMINAMATH_CALUDE_flour_per_batch_correct_l3423_342362

/-- The number of cups of flour required for one batch of cookies. -/
def flour_per_batch : ℝ := 2

/-- The number of batches Gigi has baked. -/
def baked_batches : ℕ := 3

/-- The total amount of flour in Gigi's bag. -/
def total_flour : ℝ := 20

/-- The number of additional batches Gigi could make with remaining flour. -/
def future_batches : ℕ := 7

/-- Theorem stating that the amount of flour per batch is correct given the conditions. -/
theorem flour_per_batch_correct :
  flour_per_batch * (baked_batches + future_batches : ℝ) = total_flour :=
by sorry

end NUMINAMATH_CALUDE_flour_per_batch_correct_l3423_342362


namespace NUMINAMATH_CALUDE_max_carlson_jars_l3423_342347

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlsonWeights : List Nat  -- List of weights of Carlson's jars
  babyWeights : List Nat     -- List of weights of Baby's jars

/-- Checks if the given JamJars satisfies the initial condition -/
def satisfiesInitialCondition (jars : JamJars) : Prop :=
  jars.carlsonWeights.sum = 13 * jars.babyWeights.sum

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def satisfiesFinalCondition (jars : JamJars) : Prop :=
  let minWeight := jars.carlsonWeights.minimum?
  match minWeight with
  | some w => (jars.carlsonWeights.sum - w) = 8 * (jars.babyWeights.sum + w)
  | none => False

/-- Theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars :
  ∀ jars : JamJars,
    satisfiesInitialCondition jars →
    satisfiesFinalCondition jars →
    jars.carlsonWeights.length ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l3423_342347


namespace NUMINAMATH_CALUDE_system_solution_proof_l3423_342336

theorem system_solution_proof (x y : ℝ) : 
  (4 / (x^2 + y^2) + x^2 * y^2 = 5 ∧ x^4 + y^4 + 3 * x^2 * y^2 = 20) ↔ 
  ((x = Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = Real.sqrt 2 ∧ y = -Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l3423_342336


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3423_342331

theorem geometric_sequence_property (x : ℝ) (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
  a 1 = Real.sin x →
  a 2 = Real.cos x →
  a 3 = Real.tan x →
  a 8 = 1 + Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3423_342331


namespace NUMINAMATH_CALUDE_tree_prob_five_vertices_l3423_342307

/-- The number of vertices in the graph -/
def n : ℕ := 5

/-- The probability of drawing an edge between any two vertices -/
def edge_prob : ℚ := 1/2

/-- The number of labeled trees on n vertices -/
def num_labeled_trees (n : ℕ) : ℕ := n^(n-2)

/-- The total number of possible graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n.choose 2)

/-- The probability that a randomly generated graph is a tree -/
def tree_probability (n : ℕ) : ℚ := (num_labeled_trees n : ℚ) / (total_graphs n : ℚ)

theorem tree_prob_five_vertices :
  tree_probability n = 125 / 1024 :=
sorry

end NUMINAMATH_CALUDE_tree_prob_five_vertices_l3423_342307


namespace NUMINAMATH_CALUDE_roots_problem_l3423_342389

theorem roots_problem :
  (∀ x : ℝ, x > 0 → x^2 = 1/16 → x = 1/4) ∧
  (∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -8 → x = -2) := by
sorry

end NUMINAMATH_CALUDE_roots_problem_l3423_342389
