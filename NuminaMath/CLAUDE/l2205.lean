import Mathlib

namespace NUMINAMATH_CALUDE_cost_of_oil_l2205_220573

/-- The cost of oil given the total cost of groceries and the costs of beef and chicken -/
theorem cost_of_oil (total_cost beef_cost chicken_cost : ℝ) : 
  total_cost = 16 → beef_cost = 12 → chicken_cost = 3 → 
  total_cost - (beef_cost + chicken_cost) = 1 := by
sorry

end NUMINAMATH_CALUDE_cost_of_oil_l2205_220573


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l2205_220504

/-- The number of Popsicles Megan can finish in a given time -/
def popsicles_finished (popsicle_interval : ℕ) (total_time : ℕ) : ℕ :=
  total_time / popsicle_interval

theorem megan_popsicle_consumption :
  let popsicle_interval : ℕ := 15  -- minutes
  let hours : ℕ := 4
  let additional_minutes : ℕ := 30
  let total_time : ℕ := hours * 60 + additional_minutes
  popsicles_finished popsicle_interval total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l2205_220504


namespace NUMINAMATH_CALUDE_equation_solution_l2205_220568

theorem equation_solution (x : ℝ) :
  (x / 3) / 3 = 9 / (x / 3) → x = 3^(5/2) ∨ x = -(3^(5/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2205_220568


namespace NUMINAMATH_CALUDE_distance_B_C_is_250_l2205_220558

/-- Represents a city in a triangle of cities -/
structure City :=
  (name : String)

/-- Represents the distance between two cities -/
def distance (a b : City) : ℝ := sorry

/-- The theorem stating the distance between cities B and C -/
theorem distance_B_C_is_250 (A B C : City) 
  (h1 : distance A B = distance A C + distance B C - 200)
  (h2 : distance A C = distance A B + distance B C - 300) :
  distance B C = 250 := by sorry

end NUMINAMATH_CALUDE_distance_B_C_is_250_l2205_220558


namespace NUMINAMATH_CALUDE_system_solution_l2205_220565

/-- The system of equations:
    y^2 = (x+8)(x^2 + 2)
    y^2 - (8+4x)y + (16+16x-5x^2) = 0
    has solutions (0, ±4), (-2, ±6), (-5, ±9), and (19, ±99) -/
theorem system_solution :
  ∀ (x y : ℝ),
    (y^2 = (x+8)*(x^2 + 2) ∧
     y^2 - (8+4*x)*y + (16+16*x-5*x^2) = 0) ↔
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨
     (x = -2 ∧ (y = 6 ∨ y = -6)) ∨
     (x = -5 ∧ (y = 9 ∨ y = -9)) ∨
     (x = 19 ∧ (y = 99 ∨ y = -99))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2205_220565


namespace NUMINAMATH_CALUDE_ten_millions_count_hundred_thousands_count_l2205_220550

/-- Represents the progression rate between adjacent counting units -/
def progression_rate : ℕ := 10

/-- The number of ten millions in one hundred million -/
def ten_millions_in_hundred_million : ℕ := progression_rate

/-- The number of hundred thousands in one million -/
def hundred_thousands_in_million : ℕ := progression_rate

/-- Theorem stating the number of ten millions in one hundred million is 10 -/
theorem ten_millions_count : ten_millions_in_hundred_million = 10 := by sorry

/-- Theorem stating the number of hundred thousands in one million is 10 -/
theorem hundred_thousands_count : hundred_thousands_in_million = 10 := by sorry

end NUMINAMATH_CALUDE_ten_millions_count_hundred_thousands_count_l2205_220550


namespace NUMINAMATH_CALUDE_pallet_weight_l2205_220501

/-- Given a pallet with 3 boxes, where each box weighs 89 kilograms,
    the total weight of the pallet is 267 kilograms. -/
theorem pallet_weight (num_boxes : ℕ) (weight_per_box : ℕ) (total_weight : ℕ) : 
  num_boxes = 3 → weight_per_box = 89 → total_weight = num_boxes * weight_per_box → 
  total_weight = 267 := by
  sorry

end NUMINAMATH_CALUDE_pallet_weight_l2205_220501


namespace NUMINAMATH_CALUDE_square_plus_25_divisible_by_2_and_5_l2205_220502

/-- A positive integer with only prime divisors 2 and 5 -/
def HasOnly2And5AsDivisors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 5

theorem square_plus_25_divisible_by_2_and_5 :
  ∀ N : ℕ, N > 0 →
  HasOnly2And5AsDivisors N →
  (∃ M : ℕ, N + 25 = M^2) →
  N = 200 ∨ N = 2000 := by
sorry

end NUMINAMATH_CALUDE_square_plus_25_divisible_by_2_and_5_l2205_220502


namespace NUMINAMATH_CALUDE_equation_solution_set_l2205_220584

theorem equation_solution_set : 
  ∃ (S : Set ℝ), S = {x : ℝ | 16 * Real.sin (Real.pi * x) * Real.cos (Real.pi * x) = 16 * x + 1 / x} ∧ 
  S = {-(1/4), 1/4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l2205_220584


namespace NUMINAMATH_CALUDE_vacation_cost_l2205_220555

theorem vacation_cost (C : ℝ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2205_220555


namespace NUMINAMATH_CALUDE_max_whole_nine_one_number_l2205_220522

def is_nine_one_number (t : ℕ) : Prop :=
  t ≥ 1000 ∧ t ≤ 9999 ∧
  (t / 1000 + (t / 10) % 10 = 9) ∧
  ((t / 100) % 10 - t % 10 = 1)

def P (t : ℕ) : ℕ := 2 * (t / 1000) + (t % 10)

def Q (t : ℕ) : ℕ := 2 * ((t / 100) % 10) + ((t / 10) % 10)

def G (t : ℕ) : ℚ := 2 * (P t : ℚ) / (Q t : ℚ)

def is_whole_nine_one_number (t : ℕ) : Prop :=
  is_nine_one_number t ∧ (G t).isInt

theorem max_whole_nine_one_number :
  ∃ M : ℕ,
    is_whole_nine_one_number M ∧
    ∀ t : ℕ, is_whole_nine_one_number t → t ≤ M ∧
    M = 7524 :=
sorry

end NUMINAMATH_CALUDE_max_whole_nine_one_number_l2205_220522


namespace NUMINAMATH_CALUDE_union_of_sets_l2205_220517

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2205_220517


namespace NUMINAMATH_CALUDE_perpendicular_planes_perpendicular_lines_parallel_l2205_220543

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- Theorem 1: If a line is perpendicular to a plane and contained in another plane,
-- then the two planes are perpendicular
theorem perpendicular_planes
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contains β a) :
  planePerpendicular α β :=
sorry

-- Theorem 2: If two lines are perpendicular to the same plane,
-- then the lines are parallel
theorem perpendicular_lines_parallel
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b α) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_perpendicular_lines_parallel_l2205_220543


namespace NUMINAMATH_CALUDE_always_odd_expression_l2205_220579

theorem always_odd_expression (o n : ℕ) (ho : Odd o) (hn : n > 0) :
  Odd (o^3 + n^2 * o^2) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_expression_l2205_220579


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2205_220563

theorem number_exceeding_fraction : ∃ x : ℝ, x = (5 / 9) * x + 150 ∧ x = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2205_220563


namespace NUMINAMATH_CALUDE_parallel_line_through_point_A_l2205_220519

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the parallel line passing through point A
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

theorem parallel_line_through_point_A :
  (parallel_line point_A.1 point_A.2) ∧
  (∀ (x y : ℝ), parallel_line x y → given_line x y → x = y) ∧
  (∃ (m b : ℝ), ∀ (x y : ℝ), parallel_line x y ↔ y = m * x + b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_A_l2205_220519


namespace NUMINAMATH_CALUDE_square_ratio_problem_l2205_220596

theorem square_ratio_problem :
  ∀ (s1 s2 : ℝ),
  (s1^2 / s2^2 = 75 / 128) →
  ∃ (a b c : ℕ),
  (s1 / s2 = (a : ℝ) * Real.sqrt b / c) ∧
  a = 5 ∧ b = 6 ∧ c = 16 ∧
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l2205_220596


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l2205_220534

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 3 = 2024)
                                  (h2 : Nat.choose 24 4 = 10626)
                                  (h3 : Nat.choose 24 5 = 42504) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l2205_220534


namespace NUMINAMATH_CALUDE_complement_of_N_in_U_l2205_220553

def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_U :
  U \ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_in_U_l2205_220553


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l2205_220524

/-- Represents the remaining oil quantity in liters at time t in minutes -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
    Q t = initial_quantity - outflow_rate * t ∧
    Q 0 = initial_quantity ∧
    ∀ t₁ t₂ : ℝ, t₁ < t₂ → Q t₂ < Q t₁ := by
  sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l2205_220524


namespace NUMINAMATH_CALUDE_pyramid_edges_l2205_220561

/-- Represents a pyramid with a polygonal base. -/
structure Pyramid where
  base_sides : ℕ
  deriving Repr

/-- The number of faces in a pyramid. -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of vertices in a pyramid. -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid. -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

/-- Theorem: A pyramid with 16 faces and vertices combined has 14 edges. -/
theorem pyramid_edges (p : Pyramid) : 
  num_faces p + num_vertices p = 16 → num_edges p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_l2205_220561


namespace NUMINAMATH_CALUDE_carousel_horses_count_l2205_220542

theorem carousel_horses_count :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let gold_horses : ℕ := green_horses / 6
  blue_horses + purple_horses + green_horses + gold_horses = 33 :=
by sorry

end NUMINAMATH_CALUDE_carousel_horses_count_l2205_220542


namespace NUMINAMATH_CALUDE_det_A_equals_two_l2205_220521

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, -2; 1, d] →
  A + 2 * A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l2205_220521


namespace NUMINAMATH_CALUDE_equation_solutions_l2205_220518

def solution_set : Set (ℤ × ℤ) :=
  {(0, -4), (0, 8), (-2, 0), (-4, 8), (-6, 6), (0, 0), (-10, 4)}

def satisfies_equation (x y : ℤ) : Prop :=
  x + y ≠ 0 ∧ (x - y)^2 / (x + y) = x - y + 6

theorem equation_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2205_220518


namespace NUMINAMATH_CALUDE_percentage_problem_l2205_220526

theorem percentage_problem (x : ℝ) : 120 = 2.4 * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2205_220526


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_B_l2205_220512

/-- Proves that the percentage of ethanol in fuel B is 16% given the problem conditions -/
theorem ethanol_percentage_in_fuel_B (tank_capacity : ℝ) (ethanol_A : ℝ) (total_ethanol : ℝ) (fuel_A_volume : ℝ) : 
  tank_capacity = 200 →
  ethanol_A = 0.12 →
  total_ethanol = 28 →
  fuel_A_volume = 99.99999999999999 →
  (total_ethanol - ethanol_A * fuel_A_volume) / (tank_capacity - fuel_A_volume) = 0.16 := by
sorry

#eval (28 - 0.12 * 99.99999999999999) / (200 - 99.99999999999999)

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_B_l2205_220512


namespace NUMINAMATH_CALUDE_greatest_common_length_l2205_220569

theorem greatest_common_length (a b c : Nat) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l2205_220569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2205_220500

/-- The sum of an arithmetic sequence with first term 3, last term 153, and common difference 5,
    when divided by 24, has a remainder of 0. -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 → aₙ = 153 → d = 5 → aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2205_220500


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2205_220597

theorem complex_arithmetic_equality : (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + 2 * Complex.I * (3 - 5 * Complex.I) = 11 - 9 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2205_220597


namespace NUMINAMATH_CALUDE_exp_greater_than_power_e_l2205_220516

theorem exp_greater_than_power_e (x : ℝ) (h1 : x > 0) (h2 : x ≠ ℯ) : ℯ^x > x^ℯ := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_power_e_l2205_220516


namespace NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_l2205_220577

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_l2205_220577


namespace NUMINAMATH_CALUDE_like_terms_sum_l2205_220591

/-- Given that x^(n+1)y^3 and (1/3)x^3y^(m-1) are like terms, prove that m + n = 6 -/
theorem like_terms_sum (m n : ℤ) : 
  (∃ (x y : ℝ), x^(n+1) * y^3 = (1/3) * x^3 * y^(m-1)) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l2205_220591


namespace NUMINAMATH_CALUDE_heptagon_angle_sum_l2205_220537

/-- A polygon with vertices A, B, C, D, E, F, G -/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The sum of angles FAD, GBC, BCE, ADG, CEF, AFE, DGB -/
def angle_sum (h : Heptagon) : ℝ :=
  angle h.F h.A h.D +
  angle h.G h.B h.C +
  angle h.B h.C h.E +
  angle h.A h.D h.G +
  angle h.C h.E h.F +
  angle h.A h.F h.E +
  angle h.D h.G h.B

theorem heptagon_angle_sum (h : Heptagon) : angle_sum h = 540 := by sorry

end NUMINAMATH_CALUDE_heptagon_angle_sum_l2205_220537


namespace NUMINAMATH_CALUDE_partnership_profit_l2205_220509

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of investments (larger : smaller)
  time_ratio : ℕ        -- Ratio of investment periods (longer : shorter)
  smaller_profit : ℕ    -- Profit of the partner with smaller investment

/-- Calculates the total profit of a partnership -/
def total_profit (p : Partnership) : ℕ :=
  let profit_ratio := p.investment_ratio * p.time_ratio + 1
  profit_ratio * p.smaller_profit

/-- Theorem: For a partnership where one partner's investment is triple and 
    investment period is double that of the other, if the partner with 
    smaller investment receives 7000, the total profit is 49000 -/
theorem partnership_profit : 
  ∀ (p : Partnership), 
    p.investment_ratio = 3 → 
    p.time_ratio = 2 → 
    p.smaller_profit = 7000 → 
    total_profit p = 49000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l2205_220509


namespace NUMINAMATH_CALUDE_kolya_is_wrong_l2205_220531

/-- Represents a statement about the number of pencils. -/
structure PencilStatement where
  blue : ℕ
  green : ℕ

/-- The box of colored pencils. -/
def pencil_box : PencilStatement := sorry

/-- Vasya's statement -/
def vasya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4

/-- Kolya's statement -/
def kolya_statement (box : PencilStatement) : Prop :=
  box.green ≥ 5

/-- Petya's statement -/
def petya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

/-- Misha's statement -/
def misha_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

/-- Three statements are true and one is false -/
axiom three_true_one_false :
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ ¬kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ ¬misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ ¬petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (¬vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box)

theorem kolya_is_wrong :
  ¬kolya_statement pencil_box ∧
  vasya_statement pencil_box ∧
  petya_statement pencil_box ∧
  misha_statement pencil_box :=
by sorry

end NUMINAMATH_CALUDE_kolya_is_wrong_l2205_220531


namespace NUMINAMATH_CALUDE_first_class_size_l2205_220538

/-- The number of students in the second class -/
def second_class_students : ℕ := 48

/-- The average marks of the first class -/
def first_class_average : ℚ := 60

/-- The average marks of the second class -/
def second_class_average : ℚ := 58

/-- The average marks of all students -/
def total_average : ℚ := 59067961165048544 / 1000000000000000

/-- The number of students in the first class -/
def first_class_students : ℕ := 55

theorem first_class_size :
  (first_class_students * first_class_average + second_class_students * second_class_average) / 
  (first_class_students + second_class_students) = total_average :=
sorry

end NUMINAMATH_CALUDE_first_class_size_l2205_220538


namespace NUMINAMATH_CALUDE_wire_problem_l2205_220589

theorem wire_problem (total_length : ℝ) (num_parts : ℕ) (used_parts : ℕ) : 
  total_length = 50 ∧ 
  num_parts = 5 ∧ 
  used_parts = 3 → 
  total_length - (total_length / num_parts) * used_parts = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_problem_l2205_220589


namespace NUMINAMATH_CALUDE_binomial_20_10_l2205_220562

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824)
                       (h2 : Nat.choose 18 9 = 48620)
                       (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l2205_220562


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2205_220549

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 990) → 
  ((a + 2) + (b + 2) + (c + 2) = 36) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2205_220549


namespace NUMINAMATH_CALUDE_f_properties_l2205_220552

noncomputable def f (x : ℝ) : ℝ := (2^x) / (4^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≤ (1/2 : ℝ)) ∧
  (∃ x : ℝ, f x = (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2205_220552


namespace NUMINAMATH_CALUDE_range_of_m_l2205_220541

-- Define the set of real numbers between 1 and 2
def OpenInterval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def InequalityCondition (m : ℝ) : Prop :=
  ∀ x ∈ OpenInterval, x^2 + m*x + 2 ≥ 0

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (InequalityCondition m) ↔ m ≥ -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2205_220541


namespace NUMINAMATH_CALUDE_scientific_notation_of_1010659_l2205_220554

/-- The original number to be expressed in scientific notation -/
def original_number : ℕ := 1010659

/-- The number of significant figures to keep -/
def significant_figures : ℕ := 3

/-- Function to convert a natural number to scientific notation with given significant figures -/
noncomputable def to_scientific_notation (n : ℕ) (sig_figs : ℕ) : ℝ × ℤ := sorry

/-- Theorem stating that the scientific notation of 1,010,659 with three significant figures is 1.01 × 10^6 -/
theorem scientific_notation_of_1010659 :
  to_scientific_notation original_number significant_figures = (1.01, 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1010659_l2205_220554


namespace NUMINAMATH_CALUDE_exists_triangular_face_l2205_220507

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A face of a polyhedron is one of the flat polygonal surfaces that make up its boundary. -/
structure Face (P : ConvexPolyhedron) where
  -- Again, we just declare it exists without full definition
  dummy : Unit

/-- An edge of a polyhedron is a line segment where two faces meet. -/
structure Edge (P : ConvexPolyhedron) where
  dummy : Unit

/-- A vertex of a polyhedron is a point where three or more edges meet. -/
structure Vertex (P : ConvexPolyhedron) where
  dummy : Unit

/-- The number of edges meeting at a vertex. -/
def edgesAtVertex (P : ConvexPolyhedron) (v : Vertex P) : ℕ :=
  sorry -- Definition not provided, but assumed to exist

/-- Predicate to check if a face is triangular. -/
def isTriangular (P : ConvexPolyhedron) (f : Face P) : Prop :=
  sorry -- Definition not provided, but assumed to exist

/-- Theorem stating that if at least four edges meet at each vertex of a convex polyhedron,
    then at least one of its faces is a triangle. -/
theorem exists_triangular_face (P : ConvexPolyhedron)
  (h : ∀ (v : Vertex P), edgesAtVertex P v ≥ 4) :
  ∃ (f : Face P), isTriangular P f := by
  sorry

end NUMINAMATH_CALUDE_exists_triangular_face_l2205_220507


namespace NUMINAMATH_CALUDE_simplify_expression_l2205_220582

variable (a b : ℝ)

theorem simplify_expression (hb : b ≠ 0) :
  6 * a^5 * b^2 / (3 * a^3 * b^2) + (2 * a * b^3)^2 / (-b^2)^3 = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2205_220582


namespace NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2205_220533

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by a plane intersecting a cylinder -/
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ

/-- The theorem stating the relationship between the cylinder and the ellipse -/
theorem cylinder_ellipse_intersection
  (c : RightCircularCylinder)
  (e : Ellipse)
  (h1 : c.radius = 3)
  (h2 : e.minorAxis = 2 * c.radius)
  (h3 : e.majorAxis = e.minorAxis * 1.6)
  : e.majorAxis = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2205_220533


namespace NUMINAMATH_CALUDE_tetrahedron_similarity_counterexample_l2205_220527

/-- A tetrahedron with equilateral triangle base and three other sides --/
structure Tetrahedron :=
  (base : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Two triangular faces are similar --/
def similar_faces (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    ((t1.side1 = k * t2.side1 ∧ t1.side2 = k * t2.side2) ∨
     (t1.side1 = k * t2.side2 ∧ t1.side2 = k * t2.side3) ∨
     (t1.side1 = k * t2.side3 ∧ t1.side2 = k * t2.base) ∨
     (t1.side2 = k * t2.side1 ∧ t1.side3 = k * t2.side2) ∨
     (t1.side2 = k * t2.side2 ∧ t1.side3 = k * t2.side3) ∨
     (t1.side2 = k * t2.side3 ∧ t1.side3 = k * t2.base) ∨
     (t1.side3 = k * t2.side1 ∧ t1.base = k * t2.side2) ∨
     (t1.side3 = k * t2.side2 ∧ t1.base = k * t2.side3) ∨
     (t1.side3 = k * t2.side3 ∧ t1.base = k * t2.base))

/-- Two tetrahedrons are similar --/
def similar_tetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.base = k * t2.base ∧
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.side3 = k * t2.side3

/-- The main theorem --/
theorem tetrahedron_similarity_counterexample :
  ∃ (t1 t2 : Tetrahedron),
    (∀ (f1 f2 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → f2 t1 t1 → f1 = f2) ∧
      (f1 t2 t2 → f2 t2 t2 → f1 = f2)) ∧
    (∀ (f1 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → ∃ (f2 : Tetrahedron → Tetrahedron → Prop), f2 t2 t2 ∧ f1 = f2)) ∧
    ¬(similar_tetrahedrons t1 t2) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_similarity_counterexample_l2205_220527


namespace NUMINAMATH_CALUDE_fidos_yard_exploration_l2205_220593

theorem fidos_yard_exploration (s : ℝ) (s_pos : s > 0) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := π * s^2
  let ratio := circle_area / hexagon_area
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
    ratio = Real.sqrt a / b * π ∧
    a * b = 27 := by
  sorry

end NUMINAMATH_CALUDE_fidos_yard_exploration_l2205_220593


namespace NUMINAMATH_CALUDE_polynomial_roots_l2205_220590

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = 2 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2205_220590


namespace NUMINAMATH_CALUDE_flower_arrangement_problem_l2205_220580

/-- Represents the flower arrangement problem --/
theorem flower_arrangement_problem 
  (initial_roses : ℕ) 
  (initial_daisies : ℕ) 
  (thrown_roses : ℕ) 
  (thrown_daisies : ℕ) 
  (final_roses : ℕ) 
  (final_daisies : ℕ) 
  (time_constraint : ℕ) :
  initial_roses = 21 →
  initial_daisies = 17 →
  thrown_roses = 34 →
  thrown_daisies = 25 →
  final_roses = 15 →
  final_daisies = 10 →
  time_constraint = 2 →
  (thrown_roses + thrown_daisies) - 
  ((thrown_roses - initial_roses + final_roses) + 
   (thrown_daisies - initial_daisies + final_daisies)) = 13 :=
by sorry


end NUMINAMATH_CALUDE_flower_arrangement_problem_l2205_220580


namespace NUMINAMATH_CALUDE_prime_polynomial_R_value_l2205_220570

theorem prime_polynomial_R_value :
  ∀ (R Q : ℤ),
    R > 0 →
    (∃ p : ℕ+, Nat.Prime p ∧ (R^3 + 4*R^2 + (Q - 93)*R + 14*Q + 10 : ℤ) = p) →
    R = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_polynomial_R_value_l2205_220570


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2205_220575

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2 * |x - 1| ≤ a) → a ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2205_220575


namespace NUMINAMATH_CALUDE_gummy_bear_manufacturing_time_l2205_220572

/-- The time needed to manufacture gummy bears for a given number of packets -/
def manufacturingTime (bearsPerMinute : ℕ) (bearsPerPacket : ℕ) (numPackets : ℕ) : ℕ :=
  (numPackets * bearsPerPacket) / bearsPerMinute

theorem gummy_bear_manufacturing_time :
  manufacturingTime 300 50 240 = 40 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_manufacturing_time_l2205_220572


namespace NUMINAMATH_CALUDE_next_number_with_property_l2205_220574

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  has_property 1818 ∧
  (∀ m, 1818 < m ∧ m < 1832 → ¬ has_property m) ∧
  has_property 1832 := by sorry

end NUMINAMATH_CALUDE_next_number_with_property_l2205_220574


namespace NUMINAMATH_CALUDE_tv_price_change_l2205_220578

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.4) = P * 1.12 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l2205_220578


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2205_220532

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y = x*y) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9*x₀ + y₀ = x₀*y₀ ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2205_220532


namespace NUMINAMATH_CALUDE_complement_union_equals_ge_one_l2205_220588

open Set

def M : Set ℝ := {x | (x + 3) / (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

theorem complement_union_equals_ge_one : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_ge_one_l2205_220588


namespace NUMINAMATH_CALUDE_trig_identity_l2205_220595

theorem trig_identity : 
  (Real.tan (7.5 * π / 180) * Real.tan (15 * π / 180)) / 
    (Real.tan (15 * π / 180) - Real.tan (7.5 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (7.5 * π / 180)^2 - Real.cos (7.5 * π / 180)^2) = 
  -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l2205_220595


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_2x2x3_l2205_220545

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the volume of removed tetrahedra when corners are sliced to form regular hexagons -/
def volume_removed_tetrahedra (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The volume of removed tetrahedra for a 2x2x3 rectangular prism is (22 - 46√2) / 3 -/
theorem volume_removed_tetrahedra_2x2x3 :
  volume_removed_tetrahedra ⟨2, 2, 3⟩ = (22 - 46 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_2x2x3_l2205_220545


namespace NUMINAMATH_CALUDE_potato_bag_weights_l2205_220567

/-- Represents the weights of three bags of potatoes -/
structure BagWeights where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Calculates the new weights after adjustments -/
def adjustedWeights (w : BagWeights) : BagWeights :=
  { A := w.A - 0.1 * w.C
  , B := w.B + 0.15 * w.A
  , C := w.C }

/-- Theorem stating the result of the potato bag weight problem -/
theorem potato_bag_weights :
  ∀ w : BagWeights,
    w.A = 12 + 1/2 * w.B →
    w.B = 8 + 1/3 * w.C →
    w.C = 20 + 2 * w.A →
    let new_w := adjustedWeights w
    (new_w.A + new_w.B + new_w.C) = 139.55 := by
  sorry


end NUMINAMATH_CALUDE_potato_bag_weights_l2205_220567


namespace NUMINAMATH_CALUDE_rhombicosidodecahedron_symmetries_l2205_220583

/-- Represents a rhombicosidodecahedron -/
structure Rhombicosidodecahedron where
  triangular_faces : ℕ
  square_faces : ℕ
  pentagonal_faces : ℕ
  is_archimedean : Prop
  is_convex : Prop
  is_isogonal : Prop
  is_nonprismatic : Prop

/-- The number of rotational symmetries of a rhombicosidodecahedron -/
def rotational_symmetries (r : Rhombicosidodecahedron) : ℕ := 60

/-- Theorem stating that a rhombicosidodecahedron has 60 rotational symmetries -/
theorem rhombicosidodecahedron_symmetries (r : Rhombicosidodecahedron) 
  (h1 : r.triangular_faces = 20)
  (h2 : r.square_faces = 30)
  (h3 : r.pentagonal_faces = 12)
  (h4 : r.is_archimedean)
  (h5 : r.is_convex)
  (h6 : r.is_isogonal)
  (h7 : r.is_nonprismatic) :
  rotational_symmetries r = 60 := by
  sorry

end NUMINAMATH_CALUDE_rhombicosidodecahedron_symmetries_l2205_220583


namespace NUMINAMATH_CALUDE_ball_max_height_l2205_220556

/-- The height function of the ball -/
def f (t : ℝ) : ℝ := -16 * t^2 + 96 * t + 15

/-- Theorem stating that the maximum height of the ball is 159 feet -/
theorem ball_max_height :
  ∃ t_max : ℝ, ∀ t : ℝ, f t ≤ f t_max ∧ f t_max = 159 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2205_220556


namespace NUMINAMATH_CALUDE_BC_equals_2AB_l2205_220528

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (4, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

theorem BC_equals_2AB : vector_BC = (2 * vector_AB.1, 2 * vector_AB.2) := by
  sorry

end NUMINAMATH_CALUDE_BC_equals_2AB_l2205_220528


namespace NUMINAMATH_CALUDE_set_a_range_l2205_220581

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_set_a_range_l2205_220581


namespace NUMINAMATH_CALUDE_laura_debt_l2205_220520

/-- Calculates the total amount owed after one year given a principal amount,
    an annual interest rate, and assuming simple interest. -/
def totalAmountOwed (principal : ℝ) (interestRate : ℝ) : ℝ :=
  principal * (1 + interestRate)

/-- Proves that given a principal of $35 and an interest rate of 9%,
    the total amount owed after one year is $38.15. -/
theorem laura_debt : totalAmountOwed 35 0.09 = 38.15 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_l2205_220520


namespace NUMINAMATH_CALUDE_wage_increase_l2205_220587

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) : 
  new_wage = 90 ∧ 
  increase_percentage = 50 ∧ 
  new_wage = original_wage * (1 + increase_percentage / 100) → 
  original_wage = 60 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l2205_220587


namespace NUMINAMATH_CALUDE_larger_square_side_length_l2205_220540

theorem larger_square_side_length (smaller_side : ℝ) (larger_side : ℝ) : 
  smaller_side = 5 →
  larger_side = smaller_side + 5 →
  larger_side ^ 2 = 4 * smaller_side ^ 2 →
  larger_side = 10 := by
sorry

end NUMINAMATH_CALUDE_larger_square_side_length_l2205_220540


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l2205_220525

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem second_term_of_geometric_sequence
    (a : ℕ → ℕ)
    (is_geometric : IsGeometricSequence a)
    (first_term : a 1 = 5)
    (fifth_term : a 5 = 320) :
  a 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l2205_220525


namespace NUMINAMATH_CALUDE_locus_of_point_P_l2205_220529

/-- The locus of point P where a moving circle M with diameter PF₁ is tangent internally 
    to a fixed circle C -/
theorem locus_of_point_P (n m : ℝ) (h_positive : 0 < n ∧ n < m) :
  ∃ (locus : ℝ × ℝ → Prop),
    (∀ (P : ℝ × ℝ), locus P ↔ 
      (P.1^2 / m^2 + P.2^2 / (m^2 - n^2) = 1)) ∧
    (∀ (P : ℝ × ℝ), locus P → 
      ∃ (M : ℝ × ℝ),
        -- M is the center of the moving circle
        M = ((P.1 - (-n)) / 2, P.2 / 2) ∧
        -- M is internally tangent to the fixed circle C
        ((M.1^2 + M.2^2)^(1/2) + ((M.1 - (-n))^2 + M.2^2)^(1/2) = m) ∧
        -- PF₁ is a diameter of the moving circle
        (P.1 - (-n))^2 + P.2^2 = (2 * ((M.1 - (-n))^2 + M.2^2)^(1/2))^2) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l2205_220529


namespace NUMINAMATH_CALUDE_function_uniqueness_l2205_220564

theorem function_uniqueness (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) = f n + 1)
  (h2 : ∃ k, f k = 1)
  (h3 : ∀ m, ∃ n, f n ≤ m) :
  ∀ n, f n = n + 1 := by
sorry

end NUMINAMATH_CALUDE_function_uniqueness_l2205_220564


namespace NUMINAMATH_CALUDE_line_slope_and_angle_l2205_220592

/-- Theorem: For a line passing through points (-2,3) and (-1,2), its slope is -1
    and the angle it makes with the positive x-axis is 3π/4 -/
theorem line_slope_and_angle :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-1, 2)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let angle : ℝ := Real.arctan slope
  slope = -1 ∧ angle = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_angle_l2205_220592


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l2205_220511

/-- The number of balloons Tommy's mom gave him -/
def balloons_given : ℝ := 34.5

/-- The total number of balloons Tommy had after receiving more -/
def total_balloons : ℝ := 60.75

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℝ := total_balloons - balloons_given

theorem tommy_initial_balloons :
  initial_balloons = 26.25 := by sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l2205_220511


namespace NUMINAMATH_CALUDE_houses_in_block_l2205_220585

/-- Given a block where each house receives 32 pieces of junk mail and
    the entire block receives 640 pieces of junk mail, prove that
    there are 20 houses in the block. -/
theorem houses_in_block (mail_per_house : ℕ) (mail_per_block : ℕ)
    (h1 : mail_per_house = 32)
    (h2 : mail_per_block = 640) :
    mail_per_block / mail_per_house = 20 := by
  sorry

end NUMINAMATH_CALUDE_houses_in_block_l2205_220585


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2205_220535

/-- Represents a chess tournament -/
structure ChessTournament where
  /-- The number of participants in the tournament -/
  participants : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- Each participant plays exactly one game with each other participant -/
  one_game_each : total_games = participants * (participants - 1) / 2

/-- Theorem: A chess tournament with 190 games has 20 participants -/
theorem chess_tournament_participants (t : ChessTournament) 
    (h : t.total_games = 190) : t.participants = 20 := by
  sorry

#check chess_tournament_participants

end NUMINAMATH_CALUDE_chess_tournament_participants_l2205_220535


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_parts_l2205_220576

theorem quadratic_roots_integer_parts (n : ℕ) (h : n ≥ 1) :
  let original_eq := fun x : ℝ => x^2 + (2*n + 1)*x + 6*n - 5
  let result_eq := fun x : ℝ => x^2 + 2*(n + 1)*x + 8*(n - 1)
  ∃ (r₁ r₂ : ℝ), original_eq r₁ = 0 ∧ original_eq r₂ = 0 ∧
    result_eq (⌊r₁⌋) = 0 ∧ result_eq (⌊r₂⌋) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_parts_l2205_220576


namespace NUMINAMATH_CALUDE_area_ratio_for_specific_trapezoid_l2205_220530

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  PQ : ℝ  -- Length of base PQ
  RS : ℝ  -- Length of base RS
  -- Assume other necessary properties of a trapezoid

/-- The ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def area_ratio (t : ExtendedTrapezoid) : ℚ :=
  100 / 341

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_for_specific_trapezoid :
  ∃ t : ExtendedTrapezoid, t.PQ = 10 ∧ t.RS = 21 ∧ area_ratio t = 100 / 341 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_for_specific_trapezoid_l2205_220530


namespace NUMINAMATH_CALUDE_product_sum_equality_l2205_220510

/-- Given a base b, this function converts a number from base b to base 10 -/
def baseToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b -/
def decimalToBase (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: If (13)(15)(17) = 4652 in base b, then 13 + 15 + 17 = 51 in base b -/
theorem product_sum_equality (b : ℕ) (h : b > 1) :
  (baseToDecimal 13 b * baseToDecimal 15 b * baseToDecimal 17 b = baseToDecimal 4652 b) →
  (decimalToBase (baseToDecimal 13 b + baseToDecimal 15 b + baseToDecimal 17 b) b = 51) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_equality_l2205_220510


namespace NUMINAMATH_CALUDE_bob_orders_12_muffins_l2205_220514

/-- The number of muffins Bob orders per day -/
def muffins_per_day : ℕ := sorry

/-- The cost price of each muffin in cents -/
def cost_price : ℕ := 75

/-- The selling price of each muffin in cents -/
def selling_price : ℕ := 150

/-- The profit Bob makes per week in cents -/
def weekly_profit : ℕ := 6300

/-- Theorem stating that Bob orders 12 muffins per day -/
theorem bob_orders_12_muffins : muffins_per_day = 12 := by
  sorry

end NUMINAMATH_CALUDE_bob_orders_12_muffins_l2205_220514


namespace NUMINAMATH_CALUDE_quartic_sum_at_3_and_neg_3_l2205_220513

def quartic_polynomial (d a b c m : ℝ) (x : ℝ) : ℝ :=
  d * x^4 + a * x^3 + b * x^2 + c * x + m

theorem quartic_sum_at_3_and_neg_3 
  (d a b c m : ℝ) 
  (h1 : quartic_polynomial d a b c m 0 = m)
  (h2 : quartic_polynomial d a b c m 1 = 3 * m)
  (h3 : quartic_polynomial d a b c m (-1) = 4 * m) :
  quartic_polynomial d a b c m 3 + quartic_polynomial d a b c m (-3) = 144 * d + 47 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_at_3_and_neg_3_l2205_220513


namespace NUMINAMATH_CALUDE_jorge_goals_this_season_l2205_220586

/-- Given that Jorge scored 156 goals last season and the total number of goals he scored is 343,
    prove that the number of goals he scored this season is 187. -/
theorem jorge_goals_this_season (goals_last_season goals_total : ℕ)
    (h1 : goals_last_season = 156)
    (h2 : goals_total = 343) :
    goals_total - goals_last_season = 187 := by
  sorry

end NUMINAMATH_CALUDE_jorge_goals_this_season_l2205_220586


namespace NUMINAMATH_CALUDE_intersection_theorem_l2205_220548

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in parametric form -/
structure Line where
  x₀ : ℝ
  α : ℝ

/-- Represents a parabola -/
structure Parabola where
  p : ℝ

/-- Returns true if the given point lies on the line -/
def pointOnLine (point : Point) (line : Line) (t : ℝ) : Prop :=
  point.x = line.x₀ + t * Real.cos line.α ∧ point.y = t * Real.sin line.α

/-- Returns true if the given point lies on the parabola -/
def pointOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Main theorem -/
theorem intersection_theorem (line : Line) (parabola : Parabola) 
    (h_p : parabola.p > 0) :
  ∃ (A B : Point) (x₁ x₂ : ℝ),
    (∃ t₁, pointOnLine A line t₁ ∧ pointOnParabola A parabola) ∧
    (∃ t₂, pointOnLine B line t₂ ∧ pointOnParabola B parabola) ∧
    A.x = x₁ ∧ B.x = x₂ →
    (line.x₀^2 = x₁ * x₂) ∧
    (A.x * B.x + A.y * B.y = 0 → line.x₀ = 2 * parabola.p) := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2205_220548


namespace NUMINAMATH_CALUDE_max_player_salary_l2205_220557

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (max_team_salary : ℕ) :
  num_players = 23 →
  min_salary = 17000 →
  max_team_salary = 800000 →
  ∃ (max_single_salary : ℕ),
    max_single_salary = 426000 ∧
    (num_players - 1) * min_salary + max_single_salary = max_team_salary ∧
    ∀ (alternative_salary : ℕ),
      (num_players - 1) * min_salary + alternative_salary ≤ max_team_salary →
      alternative_salary ≤ max_single_salary :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l2205_220557


namespace NUMINAMATH_CALUDE_extended_volume_of_specific_box_l2205_220594

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The main theorem -/
theorem extended_volume_of_specific_box :
  let box : Box := { length := 2, width := 3, height := 6 }
  extendedVolume box = (324 + 37 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_volume_of_specific_box_l2205_220594


namespace NUMINAMATH_CALUDE_boat_journey_l2205_220551

-- Define the given constants
def total_time : ℝ := 19
def stream_velocity : ℝ := 4
def boat_speed : ℝ := 14

-- Define the distance between A and B
def distance_AB : ℝ := 122.14

-- Theorem statement
theorem boat_journey :
  let downstream_speed := boat_speed + stream_velocity
  let upstream_speed := boat_speed - stream_velocity
  total_time = distance_AB / downstream_speed + (distance_AB / 2) / upstream_speed :=
by
  sorry

#check boat_journey

end NUMINAMATH_CALUDE_boat_journey_l2205_220551


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2205_220599

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop := sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line3D) (p : Plane3D) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2205_220599


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l2205_220559

theorem magnitude_of_complex_number (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 1/w = s) : 
  Complex.abs w = 1 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l2205_220559


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2205_220503

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2205_220503


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2205_220508

/-- A rectangular prism with specific proportions -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 8

/-- Theorem stating that a rectangular prism with the given proportions has 8 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) : 
  parallel_edge_pairs prism = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2205_220508


namespace NUMINAMATH_CALUDE_race_length_proof_l2205_220598

/-- The length of the race in metres -/
def race_length : ℝ := 200

/-- The fraction of the race completed -/
def fraction_completed : ℝ := 0.25

/-- The distance run so far in metres -/
def distance_run : ℝ := 50

theorem race_length_proof : 
  fraction_completed * race_length = distance_run :=
by sorry

end NUMINAMATH_CALUDE_race_length_proof_l2205_220598


namespace NUMINAMATH_CALUDE_ladder_height_correct_l2205_220547

/-- The height of the ceiling in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light fixture below the ceiling in centimeters -/
def fixture_below_ceiling : ℝ := 15

/-- Bob's height in centimeters -/
def bob_height : ℝ := 170

/-- The distance Bob can reach above his head in centimeters -/
def bob_reach : ℝ := 52

/-- The height of the ladder in centimeters -/
def ladder_height : ℝ := 63

theorem ladder_height_correct :
  ceiling_height - fixture_below_ceiling = bob_height + bob_reach + ladder_height := by
  sorry

end NUMINAMATH_CALUDE_ladder_height_correct_l2205_220547


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l2205_220536

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  /-- Length of the first leg -/
  leg1 : ℝ
  /-- Length of the second leg -/
  leg2 : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1 ^ 2 + leg2 ^ 2 = hypotenuse ^ 2
  /-- All sides are positive -/
  leg1_pos : leg1 > 0
  leg2_pos : leg2 > 0
  hypotenuse_pos : hypotenuse > 0
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  inscribed : square_side > 0 ∧ square_side < leg1 ∧ square_side < leg2

/-- The side length of the inscribed square in the given right triangle is 12/5 -/
theorem inscribed_square_side_length (t : RightTriangleWithSquare) 
    (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : t.hypotenuse = 13) : 
    t.square_side = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l2205_220536


namespace NUMINAMATH_CALUDE_man_walking_speed_l2205_220571

/-- Calculates the speed of a man walking in the same direction as a train,
    given the train's length, speed, and time to cross the man. -/
theorem man_walking_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 600 →
  train_speed_kmh = 64 →
  crossing_time = 35.99712023038157 →
  ∃ (man_speed : ℝ), abs (man_speed - 1.10977777777778) < 0.00000000000001 :=
by sorry

end NUMINAMATH_CALUDE_man_walking_speed_l2205_220571


namespace NUMINAMATH_CALUDE_jeans_cost_l2205_220546

theorem jeans_cost (mary_sunglasses : ℕ) (mary_sunglasses_price : ℕ) (rose_shoes : ℕ) (rose_cards : ℕ) (rose_cards_price : ℕ) :
  mary_sunglasses = 2 →
  mary_sunglasses_price = 50 →
  rose_shoes = 150 →
  rose_cards = 2 →
  rose_cards_price = 25 →
  ∃ (jeans_cost : ℕ),
    mary_sunglasses * mary_sunglasses_price + jeans_cost =
    rose_shoes + rose_cards * rose_cards_price ∧
    jeans_cost = 100 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_l2205_220546


namespace NUMINAMATH_CALUDE_total_marbles_lost_l2205_220560

def initial_marbles : ℕ := 120

def marbles_lost_outside (total : ℕ) : ℕ :=
  total / 4

def marbles_given_away (remaining : ℕ) : ℕ :=
  remaining / 2

def marbles_lost_bag_tear : ℕ := 10

theorem total_marbles_lost : 
  let remaining_after_outside := initial_marbles - marbles_lost_outside initial_marbles
  let remaining_after_giving := remaining_after_outside - marbles_given_away remaining_after_outside
  let final_remaining := remaining_after_giving - marbles_lost_bag_tear
  initial_marbles - final_remaining = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_lost_l2205_220560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2205_220506

/-- An arithmetic sequence with a₂ = -5 and common difference d = 3 has a₁ = -8 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →  -- Definition of arithmetic sequence
  a 2 = -5 →                    -- Given: a₂ = -5
  d = 3 →                       -- Given: d = 3
  a 1 = -8 :=                   -- Prove: a₁ = -8
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2205_220506


namespace NUMINAMATH_CALUDE_carly_payment_l2205_220523

/-- The final amount Carly needs to pay after discount -/
def final_amount (wallet_cost purse_cost shoes_cost discount_rate : ℝ) : ℝ :=
  let total_cost := wallet_cost + purse_cost + shoes_cost
  total_cost * (1 - discount_rate)

/-- Theorem: Given the conditions, Carly needs to pay $198.90 after discount -/
theorem carly_payment : 
  ∀ (wallet_cost purse_cost shoes_cost : ℝ),
    wallet_cost = 22 →
    purse_cost = 4 * wallet_cost - 3 →
    shoes_cost = wallet_cost + purse_cost + 7 →
    final_amount wallet_cost purse_cost shoes_cost 0.1 = 198.90 :=
by
  sorry

#eval final_amount 22 85 114 0.1

end NUMINAMATH_CALUDE_carly_payment_l2205_220523


namespace NUMINAMATH_CALUDE_max_value_of_k_l2205_220515

theorem max_value_of_k : ∃ (k : ℝ), k = Real.sqrt 10 ∧ 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 7 → Real.sqrt (x - 2) + Real.sqrt (7 - x) ≤ k) ∧
  (∀ ε > 0, ∃ x : ℝ, 2 ≤ x ∧ x ≤ 7 ∧ Real.sqrt (x - 2) + Real.sqrt (7 - x) > k - ε) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_k_l2205_220515


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2205_220539

theorem fraction_decomposition (n : ℕ) (h1 : n ≥ 5) (h2 : Odd n) :
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2205_220539


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2205_220566

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2205_220566


namespace NUMINAMATH_CALUDE_population_growth_l2205_220544

/-- Given an initial population that increases by 10% annually for 2 years
    resulting in 14,520 people, prove that the initial population was 12,000. -/
theorem population_growth (P : ℝ) : 
  (P * (1 + 0.1)^2 = 14520) → P = 12000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l2205_220544


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2205_220505

/-- A line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x) -/
def isCommonTangent (k b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    (k * x₁ + b = Real.log (1 + x₁)) ∧
    (k * x₂ + b = 2 + Real.log x₂) ∧
    (k = 1 / (1 + x₁)) ∧
    (k = 1 / x₂)

/-- If a line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x), 
    then k + b = 3 - ln(2) -/
theorem common_tangent_sum (k b : ℝ) : 
  isCommonTangent k b → k + b = 3 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2205_220505
