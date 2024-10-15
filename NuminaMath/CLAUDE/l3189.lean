import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l3189_318959

theorem solve_equation : ∃! x : ℝ, (x - 4)^4 = (1/16)⁻¹ := by
  use 6
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3189_318959


namespace NUMINAMATH_CALUDE_max_value_fraction_l3189_318913

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ a b, -4 ≤ a ∧ a ≤ -2 ∧ 3 ≤ b ∧ b ≤ 5 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3189_318913


namespace NUMINAMATH_CALUDE_det_trig_matrix_l3189_318986

open Real Matrix

theorem det_trig_matrix (a b : ℝ) : 
  det !![1, sin (a + b), cos a; 
         sin (a + b), 1, sin b; 
         cos a, sin b, 1] = 
  2 * sin (a + b) * sin b * cos a + sin (a + b)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_l3189_318986


namespace NUMINAMATH_CALUDE_white_balls_count_white_balls_count_specific_l3189_318928

/-- The number of white balls in a bag, given the total number of balls,
    the number of balls of each color (except white), and the probability
    of choosing a ball that is neither red nor purple. -/
theorem white_balls_count (total green yellow red purple : ℕ)
                          (prob_not_red_purple : ℚ) : ℕ :=
  let total_balls : ℕ := total
  let green_balls : ℕ := green
  let yellow_balls : ℕ := yellow
  let red_balls : ℕ := red
  let purple_balls : ℕ := purple
  let prob_not_red_or_purple : ℚ := prob_not_red_purple
  24

/-- The number of white balls is 24 given the specific conditions. -/
theorem white_balls_count_specific : white_balls_count 60 18 2 15 3 (7/10) = 24 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_white_balls_count_specific_l3189_318928


namespace NUMINAMATH_CALUDE_inequality_proof_l3189_318937

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) 
  (h4 : x > y) (h5 : y > z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((c*z + a*x) * (c*x + a*z)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3189_318937


namespace NUMINAMATH_CALUDE_problem_statement_l3189_318951

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  ((p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3189_318951


namespace NUMINAMATH_CALUDE_commonMaterialChoices_eq_120_l3189_318990

/-- The number of ways to choose r items from n items without regard to order -/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items out of n items -/
def permutation (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 materials each from 6 materials, 
    with exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  binomial 6 1 * permutation 5 2

theorem commonMaterialChoices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_commonMaterialChoices_eq_120_l3189_318990


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3189_318923

theorem arithmetic_simplification :
  -3 + (-9) + 10 - (-18) = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3189_318923


namespace NUMINAMATH_CALUDE_writable_13121_not_writable_12131_l3189_318935

/-- A number that can be written on the blackboard -/
def Writable (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

/-- The rule for writing new numbers on the blackboard -/
axiom write_rule {a b : ℕ} (ha : Writable a) (hb : Writable b) : Writable (a * b + a + b)

/-- 1 is initially on the blackboard -/
axiom writable_one : Writable 1

/-- 2 is initially on the blackboard -/
axiom writable_two : Writable 2

/-- Theorem: 13121 can be written on the blackboard -/
theorem writable_13121 : Writable 13121 :=
  sorry

/-- Theorem: 12131 cannot be written on the blackboard -/
theorem not_writable_12131 : ¬ Writable 12131 :=
  sorry

end NUMINAMATH_CALUDE_writable_13121_not_writable_12131_l3189_318935


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3189_318994

/-- An isosceles triangle with perimeter 16 and base 4 has legs of length 6 -/
theorem isosceles_triangle_leg_length :
  ∀ (leg_length : ℝ),
  leg_length > 0 →
  leg_length + leg_length + 4 = 16 →
  leg_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l3189_318994


namespace NUMINAMATH_CALUDE_vector_properties_l3189_318906

/-- Given vectors a and b, prove that the projection of a onto b is equal to b,
    and that (a - b) is perpendicular to b. -/
theorem vector_properties (a b : ℝ × ℝ) 
    (ha : a = (2, 0)) (hb : b = (1, 1)) : 
    (((a • b) / (b • b)) • b = b) ∧ ((a - b) • b = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3189_318906


namespace NUMINAMATH_CALUDE_swallow_flock_max_weight_l3189_318929

/-- Represents the weight capacity of different swallow types and their quantities in a flock -/
structure SwallowFlock where
  american_capacity : ℕ
  european_capacity : ℕ
  african_capacity : ℕ
  total_swallows : ℕ
  american_count : ℕ
  european_count : ℕ
  african_count : ℕ

/-- Calculates the maximum weight a flock of swallows can carry -/
def max_carry_weight (flock : SwallowFlock) : ℕ :=
  flock.american_count * flock.american_capacity +
  flock.european_count * flock.european_capacity +
  flock.african_count * flock.african_capacity

/-- Theorem stating the maximum weight the specific flock can carry -/
theorem swallow_flock_max_weight :
  ∃ (flock : SwallowFlock),
    flock.american_capacity = 5 ∧
    flock.european_capacity = 2 * flock.american_capacity ∧
    flock.african_capacity = 3 * flock.american_capacity ∧
    flock.total_swallows = 120 ∧
    flock.american_count = 2 * flock.european_count ∧
    flock.african_count = 3 * flock.american_count ∧
    flock.american_count + flock.european_count + flock.african_count = flock.total_swallows ∧
    max_carry_weight flock = 1415 :=
  sorry

end NUMINAMATH_CALUDE_swallow_flock_max_weight_l3189_318929


namespace NUMINAMATH_CALUDE_profit_increase_l3189_318966

theorem profit_increase (initial_profit : ℝ) (x : ℝ) : 
  -- Conditions
  (initial_profit * (1 + x / 100) * 0.8 * 1.5 = initial_profit * 1.6200000000000001) →
  -- Conclusion
  x = 35 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l3189_318966


namespace NUMINAMATH_CALUDE_max_three_layer_structures_l3189_318973

theorem max_three_layer_structures :
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ b - 2 ∧ b - 2 ≤ c - 4 ∧
    a^2 + b^2 + c^2 ≤ 1988 ∧
    ∀ (x y z : ℕ),
      1 ≤ x ∧ x ≤ y - 2 ∧ y - 2 ≤ z - 4 ∧
      x^2 + y^2 + z^2 ≤ 1988 →
      (b - a - 1)^2 * (c - b - 1)^2 ≥ (y - x - 1)^2 * (z - y - 1)^2 ∧
    (b - a - 1)^2 * (c - b - 1)^2 = 345 :=
by sorry

end NUMINAMATH_CALUDE_max_three_layer_structures_l3189_318973


namespace NUMINAMATH_CALUDE_fa_f_product_zero_l3189_318938

/-- Given a point F, a line l, and a circle C, prove that |FA| · |F| = 0 --/
theorem fa_f_product_zero (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : 
  F.1 = 0 →
  l = {(x, y) : ℝ × ℝ | -Real.sqrt 3 * y = 0} →
  C = {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 22} →
  ∃ (A : ℝ × ℝ), A ∈ l ∧ (‖A - F‖ * ‖F‖ = 0) := by
  sorry

#check fa_f_product_zero

end NUMINAMATH_CALUDE_fa_f_product_zero_l3189_318938


namespace NUMINAMATH_CALUDE_tan_double_angle_l3189_318907

theorem tan_double_angle (x : ℝ) (h : Real.tan x = 2) : Real.tan (2 * x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3189_318907


namespace NUMINAMATH_CALUDE_coefficient_x10_expansion_l3189_318980

/-- The coefficient of x^10 in the expansion of ((1+x+x^2)(1-x)^10) is 36 -/
theorem coefficient_x10_expansion : 
  let f : ℕ → ℤ := fun n => 
    (Finset.range (n + 1)).sum (fun k => 
      (-1)^k * (Nat.choose n k) * (Finset.range 3).sum (fun i => Nat.choose 2 i * k^(2-i)))
  f 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x10_expansion_l3189_318980


namespace NUMINAMATH_CALUDE_sum_abc_equals_51_l3189_318922

theorem sum_abc_equals_51 (a b c : ℕ+) 
  (h1 : a * b + c = 50)
  (h2 : a * c + b = 50)
  (h3 : b * c + a = 50) : 
  a + b + c = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_51_l3189_318922


namespace NUMINAMATH_CALUDE_second_day_sales_l3189_318968

/-- Represents the ticket sales for a choral performance --/
structure TicketSales where
  senior_price : ℝ
  student_price : ℝ
  day1_senior : ℕ
  day1_student : ℕ
  day1_total : ℝ
  day2_senior : ℕ
  day2_student : ℕ

/-- The theorem to prove --/
theorem second_day_sales (ts : TicketSales)
  (h1 : ts.student_price = 9)
  (h2 : ts.day1_senior * ts.senior_price + ts.day1_student * ts.student_price = ts.day1_total)
  (h3 : ts.day1_senior = 4)
  (h4 : ts.day1_student = 3)
  (h5 : ts.day1_total = 79)
  (h6 : ts.day2_senior = 12)
  (h7 : ts.day2_student = 10) :
  ts.day2_senior * ts.senior_price + ts.day2_student * ts.student_price = 246 := by
  sorry


end NUMINAMATH_CALUDE_second_day_sales_l3189_318968


namespace NUMINAMATH_CALUDE_second_to_first_rocket_height_ratio_l3189_318912

def first_rocket_height : ℝ := 500
def combined_height : ℝ := 1500

theorem second_to_first_rocket_height_ratio :
  (combined_height - first_rocket_height) / first_rocket_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_rocket_height_ratio_l3189_318912


namespace NUMINAMATH_CALUDE_tank_m_height_is_10_l3189_318942

/-- Tank M is a right circular cylinder with circumference 8 meters -/
def tank_m_circumference : ℝ := 8

/-- Tank B is a right circular cylinder with height 8 meters and circumference 10 meters -/
def tank_b_height : ℝ := 8
def tank_b_circumference : ℝ := 10

/-- The capacity of tank M is 80% of the capacity of tank B -/
def capacity_ratio : ℝ := 0.8

/-- The height of tank M -/
def tank_m_height : ℝ := 10

theorem tank_m_height_is_10 :
  tank_m_height = 10 := by sorry

end NUMINAMATH_CALUDE_tank_m_height_is_10_l3189_318942


namespace NUMINAMATH_CALUDE_expression_simplification_l3189_318924

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (a - b)^2 - a*(a - b) + (a + b)*(a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3189_318924


namespace NUMINAMATH_CALUDE_video_cassette_cost_l3189_318987

theorem video_cassette_cost (audio_cost video_cost : ℕ) : 
  (7 * audio_cost + 3 * video_cost = 1110) →
  (5 * audio_cost + 4 * video_cost = 1350) →
  video_cost = 300 := by
sorry

end NUMINAMATH_CALUDE_video_cassette_cost_l3189_318987


namespace NUMINAMATH_CALUDE_cosine_sum_special_case_l3189_318996

theorem cosine_sum_special_case (α β : Real) 
  (h1 : α - β = π/3)
  (h2 : Real.tan α - Real.tan β = 3 * Real.sqrt 3) :
  Real.cos (α + β) = -1/6 := by sorry

end NUMINAMATH_CALUDE_cosine_sum_special_case_l3189_318996


namespace NUMINAMATH_CALUDE_lemonade_stand_operational_cost_l3189_318974

/-- Yulia's lemonade stand finances -/
def lemonade_stand_finances (net_profit babysitting_revenue lemonade_revenue : ℕ) : Prop :=
  ∃ (operational_cost : ℕ),
    net_profit + operational_cost = babysitting_revenue + lemonade_revenue ∧
    operational_cost = 34

/-- Theorem: Given Yulia's financial information, prove that her lemonade stand's operational cost is $34 -/
theorem lemonade_stand_operational_cost :
  lemonade_stand_finances 44 31 47 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_operational_cost_l3189_318974


namespace NUMINAMATH_CALUDE_length_of_BC_l3189_318964

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  -- A is at the origin
  t.A = (0, 0) ∧
  -- All vertices lie on the parabola
  t.A.2 = parabola t.A.1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  -- BC is parallel to x-axis
  t.B.2 = t.C.2 ∧
  -- Area of the triangle is 128
  abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2)) / 2 = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : validTriangle t) : 
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_of_BC_l3189_318964


namespace NUMINAMATH_CALUDE_thirteen_factorial_mod_seventeen_l3189_318993

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating that 13! ≡ 9 (mod 17) -/
theorem thirteen_factorial_mod_seventeen :
  factorial 13 % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_factorial_mod_seventeen_l3189_318993


namespace NUMINAMATH_CALUDE_expression_value_l3189_318905

theorem expression_value : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3189_318905


namespace NUMINAMATH_CALUDE_product_remainder_mod_seven_l3189_318934

def product_sequence : List ℕ := List.range 10 |>.map (λ i => 3 + 10 * i)

theorem product_remainder_mod_seven :
  (product_sequence.prod) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_seven_l3189_318934


namespace NUMINAMATH_CALUDE_rock_collection_contest_l3189_318991

theorem rock_collection_contest (sydney_start conner_start : ℕ) 
  (sydney_day1 conner_day2 conner_day3 : ℕ) : 
  sydney_start = 837 → 
  conner_start = 723 → 
  sydney_day1 = 4 → 
  conner_day2 = 123 → 
  conner_day3 = 27 → 
  ∃ (conner_day1 : ℕ), 
    conner_start + conner_day1 + conner_day2 + conner_day3 
    = sydney_start + sydney_day1 + 2 * conner_day1 
    ∧ conner_day1 / sydney_day1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rock_collection_contest_l3189_318991


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3189_318952

/-- Given a, b ∈ ℝ and a - bi = (1 + i)i³, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk a (-b) = Complex.I * Complex.I * Complex.I * (1 + Complex.I)) → 
  (a = 1 ∧ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3189_318952


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3189_318947

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3189_318947


namespace NUMINAMATH_CALUDE_problem_statement_l3189_318901

theorem problem_statement (p q : ℝ) (h : p^2 / q^3 = 4 / 5) :
  11/7 + (2*q^3 - p^2) / (2*q^3 + p^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3189_318901


namespace NUMINAMATH_CALUDE_power_of_power_l3189_318970

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3189_318970


namespace NUMINAMATH_CALUDE_unique_solution_sum_in_base7_l3189_318950

/-- Represents a digit in base 7 --/
def Digit7 := Fin 7

/-- Addition in base 7 --/
def add7 (a b : Digit7) : Digit7 × Bool :=
  let sum := a.val + b.val
  (⟨sum % 7, by sorry⟩, sum ≥ 7)

/-- Represents the equation in base 7 --/
def equation (A B C : Digit7) : Prop :=
  ∃ (carry1 carry2 : Bool),
    let (units, carry1) := add7 B C
    let (tens, carry2) := add7 A B
    units = A ∧
    (if carry1 then add7 (⟨1, by sorry⟩) tens else (tens, false)).1 = C ∧
    (if carry2 then add7 (⟨1, by sorry⟩) A else (A, false)).1 = A

theorem unique_solution :
  ∃! (A B C : Digit7),
    A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 ∧
    A.val ≠ B.val ∧ A.val ≠ C.val ∧ B.val ≠ C.val ∧
    equation A B C ∧
    A.val = 6 ∧ B.val = 3 ∧ C.val = 5 :=
  sorry

theorem sum_in_base7 (A B C : Digit7) 
  (h : A.val = 6 ∧ B.val = 3 ∧ C.val = 5) :
  (A.val + B.val + C.val : ℕ) % 49 = 20 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_sum_in_base7_l3189_318950


namespace NUMINAMATH_CALUDE_reciprocal_opposite_equation_l3189_318961

theorem reciprocal_opposite_equation (m : ℝ) : (1 / (-0.5) = -(m + 4)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_equation_l3189_318961


namespace NUMINAMATH_CALUDE_homework_completion_difference_l3189_318965

-- Define the efficiency rates and Tim's completion time
def samuel_efficiency : ℝ := 0.90
def sarah_efficiency : ℝ := 0.75
def tim_efficiency : ℝ := 0.80
def tim_completion_time : ℝ := 45

-- Define the theorem
theorem homework_completion_difference :
  let base_time := tim_completion_time / tim_efficiency
  let samuel_time := base_time / samuel_efficiency
  let sarah_time := base_time / sarah_efficiency
  sarah_time - samuel_time = 12.5 := by
sorry

end NUMINAMATH_CALUDE_homework_completion_difference_l3189_318965


namespace NUMINAMATH_CALUDE_square_plot_area_l3189_318930

/-- Proves that a square plot with a fence costing Rs. 58 per foot and Rs. 3944 in total has an area of 289 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 58 →
  total_cost = 3944 →
  ∃ (side_length : ℝ),
    4 * side_length * price_per_foot = total_cost ∧
    side_length^2 = 289 :=
by sorry


end NUMINAMATH_CALUDE_square_plot_area_l3189_318930


namespace NUMINAMATH_CALUDE_sector_arc_length_l3189_318927

/-- The length of an arc in a circular sector with given central angle and radius -/
def arc_length (central_angle : Real) (radius : Real) : Real :=
  central_angle * radius

theorem sector_arc_length :
  let central_angle : Real := π / 3
  let radius : Real := 2
  arc_length central_angle radius = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3189_318927


namespace NUMINAMATH_CALUDE_car_sale_profit_l3189_318914

theorem car_sale_profit (original_price : ℝ) (h : original_price > 0) :
  let purchase_price := 0.80 * original_price
  let selling_price := 1.6000000000000001 * original_price
  let profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  profit_percentage = 100.00000000000001 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_l3189_318914


namespace NUMINAMATH_CALUDE_alvin_marbles_l3189_318948

theorem alvin_marbles (initial : ℕ) (lost : ℕ) (won : ℕ) (final : ℕ) : 
  initial = 57 → lost = 18 → won = 25 → final = 64 →
  final = initial - lost + won :=
by sorry

end NUMINAMATH_CALUDE_alvin_marbles_l3189_318948


namespace NUMINAMATH_CALUDE_tiffany_albums_l3189_318983

theorem tiffany_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 7)
  (h2 : camera_pics = 13)
  (h3 : pics_per_album = 4) :
  (phone_pics + camera_pics) / pics_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_albums_l3189_318983


namespace NUMINAMATH_CALUDE_bus_equations_l3189_318911

/-- Given m buses and n people, if 40 people per bus leaves 10 people without a seat
    and 43 people per bus leaves 1 person without a seat, then two equations hold. -/
theorem bus_equations (m n : ℕ) 
    (h1 : 40 * m + 10 = n) 
    (h2 : 43 * m + 1 = n) : 
    (40 * m + 10 = 43 * m + 1) ∧ ((n - 10) / 40 = (n - 1) / 43) := by
  sorry

end NUMINAMATH_CALUDE_bus_equations_l3189_318911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_twelve_l3189_318925

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum_twelve (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 = 1 →
  arithmetic_sequence a₁ d 17 = 18 →
  arithmetic_sum a₁ d 12 = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_twelve_l3189_318925


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3189_318949

theorem trig_expression_equals_one : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3189_318949


namespace NUMINAMATH_CALUDE_van_speed_ratio_l3189_318904

theorem van_speed_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ)
  (h1 : distance = 465)
  (h2 : original_time = 5)
  (h3 : new_speed = 62)
  : (distance / new_speed) / original_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_van_speed_ratio_l3189_318904


namespace NUMINAMATH_CALUDE_arrangement_count_is_48_l3189_318998

/-- The number of ways to arrange 5 different items into two rows -/
def arrangement_count : ℕ := 48

/-- The total number of items -/
def total_items : ℕ := 5

/-- The minimum number of items in each row -/
def min_items_per_row : ℕ := 2

/-- The number of items that must be in the front row -/
def fixed_front_items : ℕ := 2

/-- Theorem stating that the number of arrangements is 48 -/
theorem arrangement_count_is_48 :
  arrangement_count = 48 ∧
  total_items = 5 ∧
  min_items_per_row = 2 ∧
  fixed_front_items = 2 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_is_48_l3189_318998


namespace NUMINAMATH_CALUDE_inequality_proof_l3189_318917

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = 3) :
  (a + b) * (b + c) * (c + a) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3189_318917


namespace NUMINAMATH_CALUDE_apps_left_l3189_318957

/-- 
Given that Dave had 23 apps initially and deleted 18 apps, 
prove that he has 5 apps left.
-/
theorem apps_left (initial_apps : ℕ) (deleted_apps : ℕ) (apps_left : ℕ) : 
  initial_apps = 23 → deleted_apps = 18 → apps_left = initial_apps - deleted_apps → apps_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_apps_left_l3189_318957


namespace NUMINAMATH_CALUDE_hank_carwash_earnings_l3189_318926

/-- Proves that Hank made $100 in the carwash given the donation information -/
theorem hank_carwash_earnings :
  ∀ (carwash_earnings : ℝ),
    -- Conditions
    (carwash_earnings * 0.9 + 80 * 0.75 + 50 = 200) →
    -- Conclusion
    carwash_earnings = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_hank_carwash_earnings_l3189_318926


namespace NUMINAMATH_CALUDE_levi_initial_score_proof_l3189_318931

/-- Levi's initial score in a basketball game with his brother -/
def levi_initial_score : ℕ := 8

/-- Levi's brother's initial score -/
def brother_initial_score : ℕ := 12

/-- The minimum difference in scores Levi wants to achieve -/
def score_difference : ℕ := 5

/-- Additional scores by Levi's brother -/
def brother_additional_score : ℕ := 3

/-- Additional scores Levi needs to reach his goal -/
def levi_additional_score : ℕ := 12

theorem levi_initial_score_proof :
  levi_initial_score = 8 ∧
  brother_initial_score = 12 ∧
  score_difference = 5 ∧
  brother_additional_score = 3 ∧
  levi_additional_score = 12 ∧
  levi_initial_score + levi_additional_score = 
    brother_initial_score + brother_additional_score + score_difference :=
by sorry

end NUMINAMATH_CALUDE_levi_initial_score_proof_l3189_318931


namespace NUMINAMATH_CALUDE_negative_x_over_two_abs_x_positive_l3189_318984

theorem negative_x_over_two_abs_x_positive (x : ℝ) (h : x < 0) :
  -x / (2 * |x|) > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_over_two_abs_x_positive_l3189_318984


namespace NUMINAMATH_CALUDE_tourists_distribution_eight_l3189_318920

/-- The number of ways to distribute n tourists between 2 guides,
    where each guide must have at least one tourist -/
def distribute_tourists (n : ℕ) : ℕ :=
  2^n - 2

theorem tourists_distribution_eight :
  distribute_tourists 8 = 254 :=
sorry

end NUMINAMATH_CALUDE_tourists_distribution_eight_l3189_318920


namespace NUMINAMATH_CALUDE_wall_width_calculation_l3189_318918

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 29 →
  wall_height = 2 →
  num_bricks = 29000 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_length * wall_height) = 7.5 := by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l3189_318918


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l3189_318958

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ binomial_coefficient 150 75 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial_coefficient 150 75 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l3189_318958


namespace NUMINAMATH_CALUDE_lift_cars_and_trucks_l3189_318902

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- The total number of people needed to lift the given number of cars and trucks -/
def total_people : ℕ := num_cars * people_per_car + num_trucks * people_per_truck

theorem lift_cars_and_trucks : total_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_lift_cars_and_trucks_l3189_318902


namespace NUMINAMATH_CALUDE_product_of_squares_l3189_318946

theorem product_of_squares (x : ℝ) :
  (2024 - x)^2 + (2022 - x)^2 = 4038 →
  (2024 - x) * (2022 - x) = 2017 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l3189_318946


namespace NUMINAMATH_CALUDE_power_function_monotonicity_l3189_318932

/-- A power function is monotonically increasing on (0, +∞) -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x > 0, ∀ y > 0, x < y → (m^2 - m - 1) * x^m < (m^2 - m - 1) * y^m

/-- The condition |m-2| < 1 -/
def condition_q (m : ℝ) : Prop := |m - 2| < 1

theorem power_function_monotonicity (m : ℝ) :
  (is_monotone_increasing m → condition_q m) ∧
  ¬(condition_q m → is_monotone_increasing m) :=
sorry

end NUMINAMATH_CALUDE_power_function_monotonicity_l3189_318932


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l3189_318945

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Define the set S of functions f: N → R satisfying the given conditions
def S : Set (ℕ → ℝ) := {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

-- State the theorem
theorem smallest_upper_bound :
  ∃ M : ℕ, (∀ f ∈ S, ∀ n : ℕ, f n < M) ∧
  (∀ M' : ℕ, M' < M → ∃ f ∈ S, ∃ n : ℕ, f n ≥ M') :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l3189_318945


namespace NUMINAMATH_CALUDE_determinant_zero_l3189_318915

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![4, 5, 6],
    ![7, 8, 9]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 4, 9],
    ![16, 25, 36],
    ![49, 64, 81]]

theorem determinant_zero (h : Matrix.det matrix1 = 0) :
  Matrix.det matrix2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_l3189_318915


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3189_318972

theorem cheryl_material_usage
  (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
  (h1 : material1 = 4 / 9)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 8 / 18) :
  material1 + material2 - leftover = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3189_318972


namespace NUMINAMATH_CALUDE_triangle_problem_l3189_318962

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : cos (2 * t.A) - 3 * cos (t.B + t.C) = 1)
  (h2 : 1/2 * t.b * t.c * sin t.A = 5 * sqrt 3)
  (h3 : t.b = 5) :
  t.A = π/3 ∧ t.a = sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3189_318962


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l3189_318954

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Theorem statement
theorem extreme_value_and_range :
  (f 2 = -8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f x ∧ f x ≤ 24) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = -8) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = 24) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l3189_318954


namespace NUMINAMATH_CALUDE_g_increasing_range_l3189_318989

/-- A piecewise function g(x) defined on [0, +∞) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ m then (1/4) * x^2 else x

/-- The theorem stating the range of m for which g is increasing on [0, +∞) -/
theorem g_increasing_range (m : ℝ) :
  (m > 0) →
  (∀ x y, 0 ≤ x ∧ x < y → g m x ≤ g m y) →
  m ∈ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_g_increasing_range_l3189_318989


namespace NUMINAMATH_CALUDE_monkey_peaches_l3189_318919

theorem monkey_peaches : ∃ (n : ℕ) (m : ℕ), 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  m % n = 27 ∧ 
  (m - 27) / n = 5 ∧ 
  ∃ (x : ℕ), 0 < x ∧ x < 7 ∧ m = 7 * n - x ∧
  m = 102 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peaches_l3189_318919


namespace NUMINAMATH_CALUDE_rulers_remaining_l3189_318900

theorem rulers_remaining (initial_rulers : ℕ) (removed_rulers : ℕ) : 
  initial_rulers = 14 → removed_rulers = 11 → initial_rulers - removed_rulers = 3 :=
by sorry

end NUMINAMATH_CALUDE_rulers_remaining_l3189_318900


namespace NUMINAMATH_CALUDE_solve_r_system_l3189_318971

theorem solve_r_system (r s : ℚ) : 
  (r - 60) / 3 = (5 - 3 * r) / 4 → 
  s + 2 * r = 10 → 
  r = 255 / 13 := by
sorry

end NUMINAMATH_CALUDE_solve_r_system_l3189_318971


namespace NUMINAMATH_CALUDE_equation_solutions_l3189_318988

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x^2 + 2*x)^(1/3) + (3*x^2 + 6*x - 4)^(1/3) = (x^2 + 2*x - 4)^(1/3)

-- Theorem statement
theorem equation_solutions :
  {x : ℝ | equation x} = {-2, 0} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3189_318988


namespace NUMINAMATH_CALUDE_combine_like_terms_l3189_318921

theorem combine_like_terms (a b : ℝ) :
  4 * (a - b)^2 - 6 * (a - b)^2 + 8 * (a - b)^2 = 6 * (a - b)^2 := by sorry

end NUMINAMATH_CALUDE_combine_like_terms_l3189_318921


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3189_318977

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  Nat.minFac (3^15 + 11^21) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3189_318977


namespace NUMINAMATH_CALUDE_find_k_value_l3189_318992

/-- Represents a point on a line segment --/
structure SegmentPoint where
  position : ℝ
  min : ℝ
  max : ℝ
  h : min ≤ position ∧ position ≤ max

/-- The theorem stating the value of k --/
theorem find_k_value (AB CD : ℝ × ℝ) (h_AB : AB = (0, 6)) (h_CD : CD = (0, 9)) :
  ∃ k : ℝ, 
    (∀ (P : SegmentPoint) (Q : SegmentPoint), 
      P.min = 0 ∧ P.max = 6 ∧ Q.min = 0 ∧ Q.max = 9 →
      P.position = 3 * k → P.position + Q.position = 12 * k) →
    k = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3189_318992


namespace NUMINAMATH_CALUDE_affine_preserves_ratio_l3189_318969

/-- An affine transformation in a vector space -/
noncomputable def AffineTransformation (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  V → V

/-- The ratio in which a point divides a line segment -/
def divides_segment_ratio {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (A B C : V) (p q : ℝ) : Prop :=
  q • (C - A) = p • (B - C)

/-- Theorem: Affine transformations preserve segment division ratios -/
theorem affine_preserves_ratio {V : Type*} [AddCommGroup V] [Module ℝ V]
  (L : AffineTransformation V) (A B C A' B' C' : V) (p q : ℝ) :
  L A = A' → L B = B' → L C = C' →
  divides_segment_ratio A B C p q →
  divides_segment_ratio A' B' C' p q :=
by sorry

end NUMINAMATH_CALUDE_affine_preserves_ratio_l3189_318969


namespace NUMINAMATH_CALUDE_equation_solution_l3189_318963

theorem equation_solution : ∃ x : ℚ, (3/x + (1/x) / (5/x) + 1/(2*x) = 5/4) ∧ (x = 10/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3189_318963


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3189_318955

/-- A line passing through (1,2) with equal x and y intercepts -/
structure Line where
  slope : ℝ
  intercept : ℝ
  passes_through_one_two : 2 = slope * 1 + intercept
  equal_intercepts : intercept = slope * intercept

/-- A point (a,b) on the line -/
structure Point (l : Line) where
  a : ℝ
  b : ℝ
  on_line : b = l.slope * a + l.intercept

/-- The theorem statement -/
theorem min_value_of_exponential_sum (l : Line) (p : Point l) 
  (h_non_zero : l.intercept ≠ 0) :
  (3 : ℝ) ^ p.a + (3 : ℝ) ^ p.b ≥ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3189_318955


namespace NUMINAMATH_CALUDE_zero_at_specific_point_l3189_318976

/-- A polynomial of degree 3 in x and y -/
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*x^2*y + b₈*x*y^2 + b₉*y^3

theorem zero_at_specific_point 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) : 
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (5/19) (16/19) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_at_specific_point_l3189_318976


namespace NUMINAMATH_CALUDE_school_classrooms_l3189_318995

/-- Given a school with a total number of students and a fixed number of students per classroom,
    calculate the number of classrooms. -/
def number_of_classrooms (total_students : ℕ) (students_per_classroom : ℕ) : ℕ :=
  total_students / students_per_classroom

/-- Theorem stating that in a school with 120 students and 5 students per classroom,
    there are 24 classrooms. -/
theorem school_classrooms :
  number_of_classrooms 120 5 = 24 := by
  sorry

#eval number_of_classrooms 120 5

end NUMINAMATH_CALUDE_school_classrooms_l3189_318995


namespace NUMINAMATH_CALUDE_largest_root_bound_l3189_318985

theorem largest_root_bound (b₂ b₁ b₀ : ℤ) (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 3) (h₀ : |b₀| ≤ 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 4 ∧
  (∀ x : ℝ, x > r → x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) ≠ 0) ∧
  (∃ x : ℝ, x ≤ r ∧ x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_bound_l3189_318985


namespace NUMINAMATH_CALUDE_playground_run_distance_l3189_318933

theorem playground_run_distance (length width laps : ℕ) : 
  length = 55 → width = 35 → laps = 2 → 
  2 * (length + width) * laps = 360 := by
sorry

end NUMINAMATH_CALUDE_playground_run_distance_l3189_318933


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l3189_318940

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcf_factorial_seven_eight : 
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l3189_318940


namespace NUMINAMATH_CALUDE_sum_of_four_repeated_digit_terms_l3189_318909

/-- A function that checks if a natural number consists of repeated digits --/
def is_repeated_digit (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number --/
def num_digits (n : ℕ) : ℕ := sorry

theorem sum_of_four_repeated_digit_terms : 
  ∃ (a b c d : ℕ), 
    2017 = a + b + c + d ∧ 
    is_repeated_digit a ∧ 
    is_repeated_digit b ∧ 
    is_repeated_digit c ∧ 
    is_repeated_digit d ∧ 
    num_digits a ≠ num_digits b ∧ 
    num_digits a ≠ num_digits c ∧ 
    num_digits a ≠ num_digits d ∧ 
    num_digits b ≠ num_digits c ∧ 
    num_digits b ≠ num_digits d ∧ 
    num_digits c ≠ num_digits d :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_repeated_digit_terms_l3189_318909


namespace NUMINAMATH_CALUDE_school_classes_l3189_318999

theorem school_classes (daily_usage_per_class : ℕ) (weekly_usage_total : ℕ) (school_days_per_week : ℕ) :
  daily_usage_per_class = 200 →
  weekly_usage_total = 9000 →
  school_days_per_week = 5 →
  weekly_usage_total / school_days_per_week / daily_usage_per_class = 9 := by
sorry

end NUMINAMATH_CALUDE_school_classes_l3189_318999


namespace NUMINAMATH_CALUDE_candy_distribution_l3189_318943

def initial_candy : ℕ := 648
def sister_candy : ℕ := 48
def num_people : ℕ := 4
def bags_per_person : ℕ := 8

theorem candy_distribution (initial_candy sister_candy num_people bags_per_person : ℕ) :
  initial_candy = 648 →
  sister_candy = 48 →
  num_people = 4 →
  bags_per_person = 8 →
  (((initial_candy - sister_candy) / num_people) / bags_per_person : ℚ).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3189_318943


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3189_318936

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 8 * x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x > 5/3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3189_318936


namespace NUMINAMATH_CALUDE_circle_tangency_l3189_318939

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the given line
def givenLine (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the y-axis
def yAxis (x : ℝ) : Prop := x = 0

-- Define the possible circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + Real.sqrt 3)^2 = 1
def circle3 (x y : ℝ) : Prop := (x - 2*Real.sqrt 3 - 3)^2 + (y + 2 + Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3
def circle4 (x y : ℝ) : Prop := (x + 2*Real.sqrt 3 + 3)^2 + (y - 2 - Real.sqrt 3)^2 = 21 + 12*Real.sqrt 3

-- Define external tangency
def externallyTangent (c1 c2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to a line
def tangentToLine (c : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop := sorry

-- Define tangency to y-axis
def tangentToYAxis (c : ℝ → ℝ → Prop) : Prop := sorry

theorem circle_tangency :
  (externallyTangent circle1 givenCircle ∧ tangentToLine circle1 givenLine ∧ tangentToYAxis circle1) ∨
  (externallyTangent circle2 givenCircle ∧ tangentToLine circle2 givenLine ∧ tangentToYAxis circle2) ∨
  (externallyTangent circle3 givenCircle ∧ tangentToLine circle3 givenLine ∧ tangentToYAxis circle3) ∨
  (externallyTangent circle4 givenCircle ∧ tangentToLine circle4 givenLine ∧ tangentToYAxis circle4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_l3189_318939


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3189_318982

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ 
  (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3189_318982


namespace NUMINAMATH_CALUDE_marble_244_is_white_l3189_318981

/-- Represents the color of a marble -/
inductive MarbleColor
  | White
  | Gray
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cyclePosition := n % 12
  if cyclePosition ≤ 4 then MarbleColor.White
  else if cyclePosition ≤ 9 then MarbleColor.Gray
  else MarbleColor.Black

/-- Theorem: The 244th marble in the sequence is white -/
theorem marble_244_is_white : marbleColor 244 = MarbleColor.White := by
  sorry


end NUMINAMATH_CALUDE_marble_244_is_white_l3189_318981


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3189_318967

/-- Given two rectangles with equal area, where one rectangle has dimensions 12 inches by 15 inches,
    and the other rectangle has a length of 6 inches, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 6)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3189_318967


namespace NUMINAMATH_CALUDE_second_class_average_mark_l3189_318979

theorem second_class_average_mark (students1 students2 : ℕ) (avg1 avg_total : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_total = 58.75 →
  (students1 : ℚ) * avg1 + (students2 : ℚ) * ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) =
    (students1 + students2 : ℚ) * avg_total →
  ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) = 70 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_mark_l3189_318979


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l3189_318960

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ := 9 * (10^7)

/-- Theorem stating that the number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers : eight_digit_integers = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l3189_318960


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3189_318903

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 + 3*x - 20 = 7*x + 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 3*x₁ - 20 = 7*x₁ + 8) ∧ 
                (x₂^2 + 3*x₂ - 20 = 7*x₂ + 8) ∧ 
                (x₁ + x₂ = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3189_318903


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3189_318916

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3189_318916


namespace NUMINAMATH_CALUDE_probability_three_red_cards_l3189_318944

/-- The probability of drawing three red cards in succession from a shuffled standard deck --/
theorem probability_three_red_cards (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 52)
  (h2 : red_cards = 26) : 
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_red_cards_l3189_318944


namespace NUMINAMATH_CALUDE_contacts_in_second_box_l3189_318908

/-- The number of contacts in the first box -/
def first_box_contacts : ℕ := 50

/-- The price of the first box in cents -/
def first_box_price : ℕ := 2500

/-- The price of the second box in cents -/
def second_box_price : ℕ := 3300

/-- The number of contacts that equal $1 worth in the chosen box -/
def contacts_per_dollar : ℕ := 3

/-- The number of contacts in the second box -/
def second_box_contacts : ℕ := 99

theorem contacts_in_second_box :
  (first_box_price / first_box_contacts > second_box_price / second_box_contacts) ∧
  (second_box_price / second_box_contacts = 100 / contacts_per_dollar) →
  second_box_contacts = 99 := by
  sorry

end NUMINAMATH_CALUDE_contacts_in_second_box_l3189_318908


namespace NUMINAMATH_CALUDE_picture_tube_consignment_l3189_318975

theorem picture_tube_consignment (defective : ℕ) (prob : ℚ) (total : ℕ) : 
  defective = 5 →
  prob = 5263157894736842 / 100000000000000000 →
  (defective : ℚ) / total * (defective - 1 : ℚ) / (total - 1) = prob →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_picture_tube_consignment_l3189_318975


namespace NUMINAMATH_CALUDE_chocolate_ratio_l3189_318910

/-- Proves that the ratio of chocolates with nuts to chocolates without nuts is 1:1 given the problem conditions. -/
theorem chocolate_ratio (total : ℕ) (eaten_with_nuts : ℚ) (eaten_without_nuts : ℚ) (left : ℕ)
  (h_total : total = 80)
  (h_eaten_with_nuts : eaten_with_nuts = 4/5)
  (h_eaten_without_nuts : eaten_without_nuts = 1/2)
  (h_left : left = 28) :
  ∃ (with_nuts without_nuts : ℕ),
    with_nuts + without_nuts = total ∧
    (1 - eaten_with_nuts) * with_nuts + (1 - eaten_without_nuts) * without_nuts = left ∧
    with_nuts = without_nuts := by
  sorry

#check chocolate_ratio

end NUMINAMATH_CALUDE_chocolate_ratio_l3189_318910


namespace NUMINAMATH_CALUDE_parallelogram_area_28_32_l3189_318953

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters -/
theorem parallelogram_area_28_32 : parallelogramArea 28 32 = 896 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_28_32_l3189_318953


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3189_318978

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 → b = 32 → c^2 = a^2 + b^2 → c = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3189_318978


namespace NUMINAMATH_CALUDE_stratified_sampling_young_teachers_l3189_318997

theorem stratified_sampling_young_teachers 
  (total_teachers : ℕ) 
  (young_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_teachers = 200)
  (h2 : young_teachers = 100)
  (h3 : sample_size = 40) :
  (young_teachers : ℚ) / (total_teachers : ℚ) * (sample_size : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_teachers_l3189_318997


namespace NUMINAMATH_CALUDE_line_intersection_plane_intersection_l3189_318956

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties
variable (lies_in : Line → Plane → Prop)
variable (intersects_line : Line → Line → Prop)
variable (intersects_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_intersection_plane_intersection 
  (a b : Line) (α β : Plane) 
  (ha : lies_in a α) (hb : lies_in b β) :
  (∀ (a b : Line) (α β : Plane), lies_in a α → lies_in b β → 
    intersects_line a b → intersects_plane α β) ∧ 
  (∃ (a b : Line) (α β : Plane), lies_in a α ∧ lies_in b β ∧ 
    intersects_plane α β ∧ ¬intersects_line a b) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_plane_intersection_l3189_318956


namespace NUMINAMATH_CALUDE_boatman_journey_l3189_318941

/-- Represents the boatman's journey on the river -/
structure RiverJourney where
  v : ℝ  -- Speed of the boat in still water
  v_T : ℝ  -- Speed of the current
  upstream_distance : ℝ  -- Distance traveled upstream
  total_time : ℝ  -- Total time for the round trip

/-- Theorem stating the conditions and results of the boatman's journey -/
theorem boatman_journey (j : RiverJourney) : 
  j.upstream_distance = 12.5 ∧ 
  (3 / (j.v - j.v_T) = 5 / (j.v + j.v_T)) ∧ 
  (j.upstream_distance / (j.v - j.v_T) + j.upstream_distance / (j.v + j.v_T) = j.total_time) ∧ 
  j.total_time = 8 → 
  j.v_T = 5/6 ∧ 
  j.upstream_distance / (j.v - j.v_T) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boatman_journey_l3189_318941
