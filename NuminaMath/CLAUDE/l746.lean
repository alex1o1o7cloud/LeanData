import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_equals_self_is_negative_one_l746_74620

theorem reciprocal_equals_self_is_negative_one (x : ℝ) :
  x < 0 ∧ 1 / x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_is_negative_one_l746_74620


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l746_74670

/-- A geometric sequence with given second and fifth terms -/
structure GeometricSequence where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- Properties of the geometric sequence -/
def GeometricSequence.properties (g : GeometricSequence) : Prop :=
  ∃ (b₁ r : ℝ),
    r > 0 ∧
    g.b₂ = b₁ * r ∧
    g.b₅ = b₁ * r^4 ∧
    let b₃ := b₁ * r^2
    let S₄ := b₁ * (r^4 - 1) / (r - 1)
    b₃ = 49 ∧ S₄ = 183.75

/-- Main theorem: The third term is 49 and the sum of first four terms is 183.75 -/
theorem geometric_sequence_theorem (g : GeometricSequence) :
  g.properties := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l746_74670


namespace NUMINAMATH_CALUDE_watermelon_sales_l746_74675

/-- Proves that the number of watermelons sold is 18, given the weight, price per pound, and total revenue -/
theorem watermelon_sales
  (weight : ℕ)
  (price_per_pound : ℕ)
  (total_revenue : ℕ)
  (h1 : weight = 23)
  (h2 : price_per_pound = 2)
  (h3 : total_revenue = 828) :
  total_revenue / (weight * price_per_pound) = 18 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_sales_l746_74675


namespace NUMINAMATH_CALUDE_treasure_trap_probability_l746_74604

/-- The number of islands --/
def num_islands : ℕ := 5

/-- The probability of an island having treasure and no traps --/
def p_treasure : ℚ := 1/5

/-- The probability of an island having traps but no treasure --/
def p_traps : ℚ := 1/5

/-- The probability of an island having neither traps nor treasure --/
def p_neither : ℚ := 3/5

/-- The number of islands with treasure --/
def treasure_islands : ℕ := 2

/-- The number of islands with traps --/
def trap_islands : ℕ := 2

/-- Theorem stating the probability of encountering exactly 2 islands with treasure and 2 with traps --/
theorem treasure_trap_probability : 
  (Nat.choose num_islands treasure_islands) * 
  (Nat.choose (num_islands - treasure_islands) trap_islands) * 
  (p_treasure ^ treasure_islands) * 
  (p_traps ^ trap_islands) * 
  (p_neither ^ (num_islands - treasure_islands - trap_islands)) = 18/625 := by
sorry

end NUMINAMATH_CALUDE_treasure_trap_probability_l746_74604


namespace NUMINAMATH_CALUDE_lottery_probability_maximum_l746_74616

/-- The probability of winning in one draw -/
def p₀ (n : ℕ) : ℚ := (10 * n) / ((n + 5) * (n + 4))

/-- The probability of exactly one win in three draws -/
def p (n : ℕ) : ℚ := 3 * p₀ n * (1 - p₀ n)^2

/-- The statement to prove -/
theorem lottery_probability_maximum (n : ℕ) (h : n > 1) :
  ∃ (max_n : ℕ) (max_p : ℚ),
    max_n > 1 ∧
    max_p = p max_n ∧
    ∀ m, m > 1 → p m ≤ max_p ∧
    max_n = 20 ∧
    max_p = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_maximum_l746_74616


namespace NUMINAMATH_CALUDE_plot_perimeter_is_180_l746_74664

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingRate : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  costRelation : totalFencingCost = fencingRate * (2 * (length + width))

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem: The perimeter of the specific plot is 180 meters -/
theorem plot_perimeter_is_180 (plot : RectangularPlot)
  (h1 : plot.fencingRate = 6.5)
  (h2 : plot.totalFencingCost = 1170) :
  perimeter plot = 180 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_is_180_l746_74664


namespace NUMINAMATH_CALUDE_coprime_20172019_l746_74683

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_20172019_l746_74683


namespace NUMINAMATH_CALUDE_equation_solution_l746_74618

theorem equation_solution : ∃ y : ℝ, 
  (y^2 - 3*y - 10)/(y + 2) + (4*y^2 + 17*y - 15)/(4*y - 1) = 5 ∧ y = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l746_74618


namespace NUMINAMATH_CALUDE_triangle_angle_C_l746_74679

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * Real.cos (θ / 2) ^ 2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_angle_C (θ A B C : Real) (a b c : Real) :
  0 < θ ∧ θ < Real.pi →
  f A θ = Real.sqrt 3 / 2 →
  a = 1 →
  b = Real.sqrt 2 →
  A + B + C = Real.pi →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  (C = 7 * Real.pi / 12 ∨ C = Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l746_74679


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l746_74655

theorem sqrt_product_equality (a : ℝ) (h : a ≥ 0) : Real.sqrt (2 * a) * Real.sqrt (3 * a) = a * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l746_74655


namespace NUMINAMATH_CALUDE_range_of_m_specific_m_value_l746_74638

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m - 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) (h : has_two_real_roots m) : m ≤ 2 := by sorry

-- Theorem for the specific value of m
theorem specific_m_value (m : ℝ) (h : has_two_real_roots m) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0 ∧ x₁^2 + x₂^2 = 6*x₁*x₂) →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_specific_m_value_l746_74638


namespace NUMINAMATH_CALUDE_gcd_735_1287_l746_74667

theorem gcd_735_1287 : Nat.gcd 735 1287 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_735_1287_l746_74667


namespace NUMINAMATH_CALUDE_total_apples_proof_l746_74626

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73
def benny_apples : ℕ := 48
def lucy_sales : ℕ := 15

theorem total_apples_proof :
  pinky_apples + danny_apples + benny_apples = 157 :=
by sorry

end NUMINAMATH_CALUDE_total_apples_proof_l746_74626


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l746_74641

/-- The perimeter of a semi-circle with radius 21.005164601010506 cm is 108.01915941002101 cm -/
theorem semicircle_perimeter : 
  let r : ℝ := 21.005164601010506
  (π * r + 2 * r) = 108.01915941002101 := by
sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l746_74641


namespace NUMINAMATH_CALUDE_fraction_and_percentage_l746_74614

theorem fraction_and_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_and_percentage_l746_74614


namespace NUMINAMATH_CALUDE_car_numbers_proof_l746_74639

theorem car_numbers_proof :
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    (∃ (a b c d : ℕ), x = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    (∃ (a b c d : ℕ), y = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    119 * x + 179 * y = 105080 ∧
    x = 337 ∧ y = 363 := by
  sorry

end NUMINAMATH_CALUDE_car_numbers_proof_l746_74639


namespace NUMINAMATH_CALUDE_triangle_existence_l746_74686

-- Define the basic types and structures
structure Point := (x y : ℝ)

def Angle := ℝ

-- Define the given elements
variable (F T : Point) -- F is midpoint of AB, T is foot of altitude
variable (α : Angle) -- angle at vertex A

-- Define the properties of the triangle
def is_midpoint (F A B : Point) : Prop := F = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

def is_altitude_foot (T A C : Point) : Prop := 
  (T.x - A.x) * (C.x - A.x) + (T.y - A.y) * (C.y - A.y) = 0

def angle_at_vertex (A B C : Point) (α : Angle) : Prop :=
  let v1 := Point.mk (B.x - A.x) (B.y - A.y)
  let v2 := Point.mk (C.x - A.x) (C.y - A.y)
  Real.cos α = (v1.x * v2.x + v1.y * v2.y) / 
    (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2))

-- The main theorem
theorem triangle_existence (F T : Point) (α : Angle) :
  ∃ (A B C : Point),
    is_midpoint F A B ∧
    is_altitude_foot T A C ∧
    angle_at_vertex A B C α ∧
    ¬(∀ (C' : Point), is_altitude_foot T A C' → C' = C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l746_74686


namespace NUMINAMATH_CALUDE_negation_of_every_constant_is_geometric_l746_74690

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal. -/
def IsConstant (s : Sequence) : Prop := ∀ n m : ℕ, s n = s m

/-- A sequence is geometric if the ratio between any two consecutive terms is constant and nonzero. -/
def IsGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

/-- The statement "Every constant sequence is a geometric sequence" -/
def EveryConstantIsGeometric : Prop :=
  ∀ s : Sequence, IsConstant s → IsGeometric s

/-- The negation of "Every constant sequence is a geometric sequence" -/
theorem negation_of_every_constant_is_geometric :
  ¬EveryConstantIsGeometric ↔ ∃ s : Sequence, IsConstant s ∧ ¬IsGeometric s :=
by
  sorry


end NUMINAMATH_CALUDE_negation_of_every_constant_is_geometric_l746_74690


namespace NUMINAMATH_CALUDE_section_b_average_weight_l746_74689

/-- Given a class with two sections, prove the average weight of section B --/
theorem section_b_average_weight
  (total_students : ℕ)
  (section_a_students : ℕ)
  (section_b_students : ℕ)
  (section_a_avg_weight : ℝ)
  (total_avg_weight : ℝ)
  (h1 : total_students = section_a_students + section_b_students)
  (h2 : section_a_students = 50)
  (h3 : section_b_students = 50)
  (h4 : section_a_avg_weight = 60)
  (h5 : total_avg_weight = 70)
  : (total_students * total_avg_weight - section_a_students * section_a_avg_weight) / section_b_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l746_74689


namespace NUMINAMATH_CALUDE_composite_cube_three_diff_squares_l746_74656

/-- A number is composite if it has a non-trivial factorization -/
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- The proposition that the cube of a composite number can be represented as the difference of two squares in at least three ways -/
theorem composite_cube_three_diff_squares (n : ℕ) (h : IsComposite n) : 
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    (n^3 = A₁^2 - B₁^2) ∧ 
    (n^3 = A₂^2 - B₂^2) ∧ 
    (n^3 = A₃^2 - B₃^2) ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) :=
sorry

end NUMINAMATH_CALUDE_composite_cube_three_diff_squares_l746_74656


namespace NUMINAMATH_CALUDE_tyrones_money_value_l746_74696

/-- Represents the total value of Tyrone's money in US dollars -/
def tyrones_money : ℚ :=
  let us_currency : ℚ :=
    4 * 1 +  -- $1 bills
    1 * 10 +  -- $10 bill
    2 * 5 +  -- $5 bills
    30 * (1/4) +  -- quarters
    5 * (1/2) +  -- half-dollar coins
    48 * (1/10) +  -- dimes
    12 * (1/20) +  -- nickels
    4 * 1 +  -- one-dollar coins
    64 * (1/100) +  -- pennies
    3 * 2 +  -- two-dollar bills
    5 * (1/2)  -- 50-cent coins

  let foreign_currency : ℚ :=
    20 * (11/10) +  -- Euro coins
    15 * (132/100) +  -- British Pound coins
    6 * (76/100)  -- Canadian Dollar coins

  us_currency + foreign_currency

/-- The theorem stating that Tyrone's money equals $98.90 -/
theorem tyrones_money_value : tyrones_money = 989/10 := by
  sorry

end NUMINAMATH_CALUDE_tyrones_money_value_l746_74696


namespace NUMINAMATH_CALUDE_r_equals_four_l746_74691

/-- Given pr = 360 and 6cr = 15, prove that r = 4 is a valid solution. -/
theorem r_equals_four (p c : ℚ) (h1 : p * 4 = 360) (h2 : 6 * c * 4 = 15) : 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_r_equals_four_l746_74691


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_is_correct_l746_74658

/-- Percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- Percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- Percentage of Type A defects that are reworked and repaired -/
def type_a_rework_rate : ℝ := 0.40

/-- Percentage of Type B defects that are reworked and repaired -/
def type_b_rework_rate : ℝ := 0.30

/-- Percentage of remaining Type A defects that are shipped -/
def type_a_ship_rate : ℝ := 0.03

/-- Percentage of remaining Type B defects that are shipped -/
def type_b_ship_rate : ℝ := 0.06

/-- The percentage of defective units (Type A or B) shipped for sale -/
def defective_shipped_percentage : ℝ :=
  type_a_defect_rate * (1 - type_a_rework_rate) * type_a_ship_rate +
  type_b_defect_rate * (1 - type_b_rework_rate) * type_b_ship_rate

theorem defective_shipped_percentage_is_correct :
  defective_shipped_percentage = 0.00462 := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_is_correct_l746_74658


namespace NUMINAMATH_CALUDE_hcl_moles_in_reaction_l746_74650

-- Define the reaction components
structure ReactionComponent where
  name : String
  moles : ℚ

-- Define the reaction
def reaction (hcl koh kcl h2o : ReactionComponent) : Prop :=
  hcl.name = "HCl" ∧ koh.name = "KOH" ∧ kcl.name = "KCl" ∧ h2o.name = "H2O" ∧
  hcl.moles = koh.moles ∧ hcl.moles = kcl.moles ∧ hcl.moles = h2o.moles

-- Theorem statement
theorem hcl_moles_in_reaction 
  (hcl koh kcl h2o : ReactionComponent)
  (h1 : reaction hcl koh kcl h2o)
  (h2 : koh.moles = 1)
  (h3 : kcl.moles = 1) :
  hcl.moles = 1 := by
  sorry


end NUMINAMATH_CALUDE_hcl_moles_in_reaction_l746_74650


namespace NUMINAMATH_CALUDE_division_remainder_problem_l746_74661

theorem division_remainder_problem :
  let dividend : ℕ := 171
  let divisor : ℕ := 21
  let quotient : ℕ := 8
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l746_74661


namespace NUMINAMATH_CALUDE_x_eq_4_is_linear_l746_74647

/-- A linear equation with one variable is of the form ax + b = 0, where a ≠ 0 and x is the variable. -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 4 represents the equation x = 4. -/
def f (x : ℝ) : ℝ := x - 4

theorem x_eq_4_is_linear :
  is_linear_equation_one_var f :=
sorry

end NUMINAMATH_CALUDE_x_eq_4_is_linear_l746_74647


namespace NUMINAMATH_CALUDE_ratio_and_equation_imply_c_value_l746_74633

theorem ratio_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_equation_imply_c_value_l746_74633


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l746_74625

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 12 ∧ m % k ≠ k - 1) ∧
  n = 27719 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l746_74625


namespace NUMINAMATH_CALUDE_divisibility_by_37_l746_74646

theorem divisibility_by_37 (a b c : ℕ) :
  let p := 100 * a + 10 * b + c
  let q := 100 * b + 10 * c + a
  let r := 100 * c + 10 * a + b
  37 ∣ p → (37 ∣ q ∧ 37 ∣ r) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l746_74646


namespace NUMINAMATH_CALUDE_expression_evaluation_l746_74643

theorem expression_evaluation :
  3 + 3 * Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (3 - Real.sqrt 3) = 4 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l746_74643


namespace NUMINAMATH_CALUDE_problem_solution_l746_74697

theorem problem_solution : ∃ m : ℚ, 15 + m * (25/3) = 6 * (25/3) - 10 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l746_74697


namespace NUMINAMATH_CALUDE_siblings_age_multiple_l746_74609

theorem siblings_age_multiple (kay_age : ℕ) (oldest_age : ℕ) (num_siblings : ℕ) : 
  kay_age = 32 →
  oldest_age = 44 →
  num_siblings = 14 →
  ∃ (youngest_age : ℕ), 
    youngest_age = kay_age / 2 - 5 ∧
    oldest_age / youngest_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_multiple_l746_74609


namespace NUMINAMATH_CALUDE_fifth_term_is_fifteen_l746_74640

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 4 = 16
  first_term : a 1 = 1

/-- The fifth term of the arithmetic sequence is 15 -/
theorem fifth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_fifteen_l746_74640


namespace NUMINAMATH_CALUDE_f_2014_equals_zero_l746_74681

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the periodicity property of f
def HasPeriodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x + f 2

-- Theorem statement
theorem f_2014_equals_zero 
  (h_even : IsEven f) 
  (h_periodicity : HasPeriodicity f) : 
  f 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2014_equals_zero_l746_74681


namespace NUMINAMATH_CALUDE_min_operation_result_l746_74694

def S : Finset Nat := {4, 6, 8, 12, 14, 18}

def operation (a b c : Nat) : Nat :=
  (a + b) * c - min a (min b c)

theorem min_operation_result :
  ∃ (result : Nat), result = 52 ∧
  ∀ (a b c : Nat), a ∈ S → b ∈ S → c ∈ S →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  operation a b c ≥ result :=
sorry

end NUMINAMATH_CALUDE_min_operation_result_l746_74694


namespace NUMINAMATH_CALUDE_xyz_value_l746_74629

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 192 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l746_74629


namespace NUMINAMATH_CALUDE_new_boarders_count_l746_74632

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ) (final_ratio_boarders : ℕ) (final_ratio_day : ℕ) :
  initial_boarders = 220 →
  initial_ratio_boarders = 5 →
  initial_ratio_day = 12 →
  final_ratio_boarders = 1 →
  final_ratio_day = 2 →
  ∃ (new_boarders : ℕ),
    new_boarders = 44 ∧
    (initial_boarders + new_boarders) * final_ratio_day = initial_boarders * initial_ratio_day * final_ratio_boarders :=
by sorry


end NUMINAMATH_CALUDE_new_boarders_count_l746_74632


namespace NUMINAMATH_CALUDE_isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l746_74698

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem isosceles_when_neg_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.b)^2 = 4 * (t.a + t.c) * (t.a - t.c) → t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) (h : t.a = t.b ∧ t.b = t.c) :
  ∃ x y, x = 0 ∧ y = -1 ∧ quadratic t x = 0 ∧ quadratic t y = 0 :=
sorry

end NUMINAMATH_CALUDE_isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l746_74698


namespace NUMINAMATH_CALUDE_conjugate_sum_product_l746_74630

theorem conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = -6) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_conjugate_sum_product_l746_74630


namespace NUMINAMATH_CALUDE_equation_holds_iff_l746_74666

theorem equation_holds_iff (a b c : ℝ) (ha : a ≠ 0) (hab : a + b ≠ 0) :
  (a + b + c) / a = (b + c) / (a + b) ↔ a = -(b + c) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_l746_74666


namespace NUMINAMATH_CALUDE_men_working_with_boys_l746_74619

-- Define the work done by one man per day
def work_man : ℚ := 1 / 48

-- Define the work done by one boy per day
def work_boy : ℚ := 5 / 96

-- Define the total work to be done
def total_work : ℚ := 1

theorem men_working_with_boys : ℕ :=
  let men_count : ℕ := 1
  have h1 : 2 * work_man + 4 * work_boy = total_work / 4 := by sorry
  have h2 : men_count * work_man + 6 * work_boy = total_work / 3 := by sorry
  have h3 : 2 * work_boy = 5 * work_man := by sorry
  men_count

end NUMINAMATH_CALUDE_men_working_with_boys_l746_74619


namespace NUMINAMATH_CALUDE_average_cat_weight_in_pounds_l746_74622

def cat_weights : List Real := [3.5, 7.2, 4.8, 6, 5.5, 9, 4, 7.5]
def kg_to_pounds : Real := 2.20462

theorem average_cat_weight_in_pounds :
  let total_weight_kg := cat_weights.sum
  let average_weight_kg := total_weight_kg / cat_weights.length
  let average_weight_pounds := average_weight_kg * kg_to_pounds
  average_weight_pounds = 13.0925 := by sorry

end NUMINAMATH_CALUDE_average_cat_weight_in_pounds_l746_74622


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l746_74663

theorem shopping_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  (h3 : remaining_money = (1 - spent_percentage) * original_amount) :
  original_amount = 320 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l746_74663


namespace NUMINAMATH_CALUDE_four_square_rectangle_exists_l746_74687

/-- Represents a color --/
structure Color : Type

/-- Represents a square on the grid --/
structure Square : Type :=
  (x : ℤ)
  (y : ℤ)
  (color : Color)

/-- Represents an infinite grid of colored squares --/
def InfiniteGrid : Type := ℤ → ℤ → Color

/-- Checks if four squares form a rectangle parallel to grid lines --/
def IsRectangle (s1 s2 s3 s4 : Square) : Prop :=
  (s1.x = s2.x ∧ s3.x = s4.x ∧ s1.y = s3.y ∧ s2.y = s4.y) ∨
  (s1.x = s3.x ∧ s2.x = s4.x ∧ s1.y = s2.y ∧ s3.y = s4.y)

/-- Main theorem: There always exist four squares of the same color forming a rectangle --/
theorem four_square_rectangle_exists (n : ℕ) (h : n ≥ 2) (grid : InfiniteGrid) :
  ∃ (s1 s2 s3 s4 : Square),
    s1.color = s2.color ∧ s2.color = s3.color ∧ s3.color = s4.color ∧
    IsRectangle s1 s2 s3 s4 := by
  sorry

end NUMINAMATH_CALUDE_four_square_rectangle_exists_l746_74687


namespace NUMINAMATH_CALUDE_watermelon_duration_example_l746_74654

/-- The number of weeks watermelons will last -/
def watermelon_duration (total : ℕ) (weekly_usage : ℕ) : ℕ :=
  total / weekly_usage

/-- Theorem: Given 30 watermelons and using 5 per week, they will last 6 weeks -/
theorem watermelon_duration_example : watermelon_duration 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_duration_example_l746_74654


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l746_74671

theorem quadratic_root_sum_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) ∧
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) ∧
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l746_74671


namespace NUMINAMATH_CALUDE_min_sum_given_product_l746_74601

theorem min_sum_given_product (a b c : ℕ+) : a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l746_74601


namespace NUMINAMATH_CALUDE_lucky_5n_is_52000_l746_74680

/-- A natural number is lucky if the sum of its digits is 7 -/
def isLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in increasing order -/
def luckySeq : ℕ → ℕ := sorry

/-- The nth element of the lucky number sequence is 2005 -/
axiom nth_lucky_is_2005 (n : ℕ) : luckySeq n = 2005

theorem lucky_5n_is_52000 (n : ℕ) : luckySeq (5 * n) = 52000 :=
  sorry

end NUMINAMATH_CALUDE_lucky_5n_is_52000_l746_74680


namespace NUMINAMATH_CALUDE_woodworker_solution_l746_74613

/-- Represents the number of furniture items made by a woodworker. -/
structure FurnitureCount where
  chairs : ℕ
  tables : ℕ
  cabinets : ℕ

/-- Calculates the total number of legs used for a given furniture count. -/
def totalLegs (f : FurnitureCount) : ℕ :=
  4 * f.chairs + 4 * f.tables + 2 * f.cabinets

/-- The woodworker's furniture count satisfies the given conditions. -/
def isSolution (f : FurnitureCount) : Prop :=
  f.chairs = 6 ∧ f.cabinets = 4 ∧ totalLegs f = 80

theorem woodworker_solution :
  ∃ f : FurnitureCount, isSolution f ∧ f.tables = 12 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_solution_l746_74613


namespace NUMINAMATH_CALUDE_a_n_properties_smallest_n_perfect_square_sum_l746_74693

/-- The largest n-digit number that is neither the sum nor the difference of two perfect squares -/
def a_n (n : ℕ) : ℕ := 10^n - 2

/-- The sum of squares of digits of a number -/
def sum_of_squares_of_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating the properties of a_n -/
theorem a_n_properties :
  ∀ (n : ℕ), n > 2 →
  (∀ (x y : ℕ), a_n n ≠ x^2 + y^2 ∧ a_n n ≠ x^2 - y^2) ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℕ), 10^m - 2 = x^2 + y^2 ∨ 10^m - 2 = x^2 - y^2) :=
sorry

/-- Theorem stating the smallest n for which the sum of squares of digits of a_n is a perfect square -/
theorem smallest_n_perfect_square_sum :
  ∃ (k : ℕ), sum_of_squares_of_digits (a_n 66) = k^2 ∧
  ∀ (n : ℕ), n < 66 → ¬∃ (k : ℕ), sum_of_squares_of_digits (a_n n) = k^2 :=
sorry

end NUMINAMATH_CALUDE_a_n_properties_smallest_n_perfect_square_sum_l746_74693


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l746_74676

theorem acute_triangle_properties (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_equation : Real.sqrt 3 * Real.sin ((B + C) / 2) - Real.cos A = 1) : 
  A = π / 3 ∧ ∀ x, x = Real.cos B + Real.cos C → Real.sqrt 3 / 2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l746_74676


namespace NUMINAMATH_CALUDE_no_integer_solutions_l746_74615

theorem no_integer_solutions :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l746_74615


namespace NUMINAMATH_CALUDE_existence_of_m_l746_74653

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 8) / (x n + 8)

theorem existence_of_m : ∃ m : ℕ, 
  123 ≤ m ∧ m ≤ 242 ∧ 
  x m ≤ 6 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 6 + 1 / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l746_74653


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l746_74688

theorem unique_number_with_gcd (n : ℕ) : 
  70 < n ∧ n < 80 ∧ Nat.gcd 15 n = 5 → n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l746_74688


namespace NUMINAMATH_CALUDE_integral_equals_half_l746_74600

open Real MeasureTheory Interval

/-- The definite integral of 1 / (1 + sin x - cos x)^2 from 2 arctan(1/2) to π/2 equals 1/2 -/
theorem integral_equals_half :
  ∫ x in 2 * arctan (1/2)..π/2, 1 / (1 + sin x - cos x)^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_half_l746_74600


namespace NUMINAMATH_CALUDE_one_alligator_per_week_l746_74610

/-- The number of Burmese pythons -/
def num_pythons : ℕ := 5

/-- The number of alligators eaten in the given time period -/
def num_alligators : ℕ := 15

/-- The number of weeks in the given time period -/
def num_weeks : ℕ := 3

/-- The number of alligators one Burmese python can eat per week -/
def alligators_per_python_per_week : ℚ := num_alligators / (num_pythons * num_weeks)

theorem one_alligator_per_week : 
  alligators_per_python_per_week = 1 :=
sorry

end NUMINAMATH_CALUDE_one_alligator_per_week_l746_74610


namespace NUMINAMATH_CALUDE_negation_or_implies_both_false_l746_74682

theorem negation_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_or_implies_both_false_l746_74682


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l746_74668

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and one focus at (-5, 0),
    prove that its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →
  (5^2 = 9 + m) →
  (∀ (x y : ℝ), x^2/9 - y^2/m = 1 → (y = (4/3)*x ∨ y = -(4/3)*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l746_74668


namespace NUMINAMATH_CALUDE_roadwork_problem_l746_74611

/-- Roadwork problem statement -/
theorem roadwork_problem (total_length pitch_day3 : ℝ) (h1 h2 h3 : ℕ) : 
  total_length = 16 ∧ 
  pitch_day3 = 6 ∧ 
  h1 = 2 ∧ 
  h2 = 5 ∧ 
  h3 = 3 → 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total_length ∧ 
    (2 * x - 1) > 0 ∧ 
    (2 * x - 1) < total_length ∧
    3 * x - 1 = total_length - (pitch_day3 / h2) / h3 ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roadwork_problem_l746_74611


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l746_74674

theorem greatest_three_digit_multiple_of_23 : 
  (∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 23 ∣ n → n ≤ 989) ∧ 
  989 ≤ 999 ∧ 989 ≥ 100 ∧ 23 ∣ 989 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l746_74674


namespace NUMINAMATH_CALUDE_square_and_ln_exp_are_geometric_l746_74645

/-- A function is geometric if it preserves geometric sequences -/
def IsGeometricFunction (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n : ℕ, a n ≠ 0) →
    (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
    (∀ n : ℕ, f (a (n + 1)) / f (a n) = f (a (n + 2)) / f (a (n + 1)))

theorem square_and_ln_exp_are_geometric :
  IsGeometricFunction (fun x ↦ x^2) ∧
  IsGeometricFunction (fun x ↦ Real.log (2^x)) :=
sorry

end NUMINAMATH_CALUDE_square_and_ln_exp_are_geometric_l746_74645


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l746_74648

/-- Represents a point in 2D Cartesian coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_across_x_axis :
  let P : Point2D := { x := 2, y := -3 }
  reflectAcrossXAxis P = { x := 2, y := 3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l746_74648


namespace NUMINAMATH_CALUDE_subtraction_proof_l746_74665

theorem subtraction_proof : 25.705 - 3.289 = 22.416 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_proof_l746_74665


namespace NUMINAMATH_CALUDE_sphere_volume_l746_74652

theorem sphere_volume (surface_area : Real) (volume : Real) : 
  surface_area = 100 * Real.pi → volume = (500 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l746_74652


namespace NUMINAMATH_CALUDE_problem_statement_l746_74636

theorem problem_statement : (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l746_74636


namespace NUMINAMATH_CALUDE_zoe_family_cost_l746_74637

/-- The total cost of soda and pizza for a group --/
def total_cost (num_people : ℕ) (soda_cost pizza_cost : ℚ) : ℚ :=
  num_people * (soda_cost + pizza_cost)

/-- Theorem: The total cost for Zoe and her family is $9 --/
theorem zoe_family_cost : 
  total_cost 6 (1/2) 1 = 9 := by sorry

end NUMINAMATH_CALUDE_zoe_family_cost_l746_74637


namespace NUMINAMATH_CALUDE_equation_solution_l746_74605

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (9 + Real.sqrt (15 + 5*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 5 + Real.sqrt 15) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l746_74605


namespace NUMINAMATH_CALUDE_puppies_given_to_friends_l746_74692

/-- The number of puppies Alyssa started with -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa has left -/
def remaining_puppies : ℕ := 5

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := initial_puppies - remaining_puppies

theorem puppies_given_to_friends : given_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_to_friends_l746_74692


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l746_74649

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -6*x^2 + 36*x - 30

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l746_74649


namespace NUMINAMATH_CALUDE_polynomial_calculation_l746_74621

/-- A polynomial of degree 4 with specific properties -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- Theorem stating the result of the calculation -/
theorem polynomial_calculation (a b c d : ℝ) 
  (h1 : P a b c d 1 = 1993)
  (h2 : P a b c d 2 = 3986)
  (h3 : P a b c d 3 = 5979) :
  (1/4) * (P a b c d 11 + P a b c d (-7)) = 4693 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_calculation_l746_74621


namespace NUMINAMATH_CALUDE_power_function_value_l746_74651

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through a point for a power function
def PassesThroughPoint (f : PowerFunction) (x y : ℝ) : Prop :=
  f x = y

-- State the theorem
theorem power_function_value (f : PowerFunction) :
  PassesThroughPoint f 9 (1/3) → f 25 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l746_74651


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l746_74657

/-- A cyclic quadrilateral with side lengths a, b, c, d, area Q, and circumradius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  Q : ℝ
  R : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ Q > 0 ∧ R > 0

/-- The main theorem about cyclic quadrilaterals -/
theorem cyclic_quadrilateral_theorem (ABCD : CyclicQuadrilateral) :
  ABCD.R^2 = ((ABCD.a * ABCD.b + ABCD.c * ABCD.d) * (ABCD.a * ABCD.c + ABCD.b * ABCD.d) * (ABCD.a * ABCD.d + ABCD.b * ABCD.c)) / (16 * ABCD.Q^2) ∧
  ABCD.R ≥ ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ∧
  (ABCD.R = ((ABCD.a * ABCD.b * ABCD.c * ABCD.d)^(3/4)) / (ABCD.Q * Real.sqrt 2) ↔ ABCD.a = ABCD.b ∧ ABCD.b = ABCD.c ∧ ABCD.c = ABCD.d) :=
by sorry


end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l746_74657


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l746_74635

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2*a + b)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (2*x + y)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l746_74635


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l746_74684

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  have h1 : 8251 = 6105 * 1 + 2146 := by sorry
  have h2 : 6105 = 2146 * 2 + 1813 := by sorry
  have h3 : 2146 = 1813 * 1 + 333 := by sorry
  have h4 : 333 = 148 * 2 + 37 := by sorry
  have h5 : 148 = 37 * 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l746_74684


namespace NUMINAMATH_CALUDE_parabola_properties_l746_74624

/-- Parabola passing through (-1, 0) -/
def parabola (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x - 3

theorem parabola_properties :
  ∃ (b : ℝ),
    (parabola b (-1) = 0) ∧
    (b = -4) ∧
    (∃ (h k : ℝ), h = -2 ∧ k = 1 ∧ ∀ x, parabola b x = -(x - h)^2 + k) ∧
    (∀ y₁ y₂ : ℝ, parabola b 1 = y₁ → parabola b (-1) = y₂ → y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l746_74624


namespace NUMINAMATH_CALUDE_toms_age_l746_74627

/-- Tom's age problem -/
theorem toms_age (s t : ℕ) : 
  t = 2 * s - 1 →  -- Tom's age is 1 year less than twice his sister's age
  t + s = 14 →     -- The sum of their ages is 14 years
  t = 9            -- Tom's age is 9 years
:= by sorry

end NUMINAMATH_CALUDE_toms_age_l746_74627


namespace NUMINAMATH_CALUDE_max_abs_z_value_l746_74644

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l746_74644


namespace NUMINAMATH_CALUDE_function_divisibility_property_l746_74678

def PositiveInt := {n : ℕ // n > 0}

theorem function_divisibility_property 
  (f : PositiveInt → PositiveInt) 
  (h : ∀ (m n : PositiveInt), (m.val^2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ (n : PositiveInt), (f n).val = n.val :=
sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l746_74678


namespace NUMINAMATH_CALUDE_points_eight_units_from_negative_three_l746_74685

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_eight_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 8 ↔ x = -11 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_eight_units_from_negative_three_l746_74685


namespace NUMINAMATH_CALUDE_point_y_value_l746_74603

/-- An angle with vertex at the origin and initial side on the non-negative x-axis -/
structure AngleAtOrigin where
  α : ℝ
  initial_side_on_x_axis : 0 ≤ α ∧ α < 2 * Real.pi

/-- A point on the terminal side of an angle -/
structure PointOnTerminalSide (angle : AngleAtOrigin) where
  x : ℝ
  y : ℝ
  on_terminal_side : x = 6 ∧ y = 6 * Real.tan angle.α

/-- Theorem: For an angle α with sin α = -4/5, if P(6, y) is on its terminal side, then y = -8 -/
theorem point_y_value (angle : AngleAtOrigin) 
  (h_sin : Real.sin angle.α = -4/5) 
  (point : PointOnTerminalSide angle) : 
  point.y = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_y_value_l746_74603


namespace NUMINAMATH_CALUDE_parking_lot_problem_l746_74699

theorem parking_lot_problem (medium_fee small_fee total_cars total_fee : ℕ)
  (h1 : medium_fee = 15)
  (h2 : small_fee = 8)
  (h3 : total_cars = 30)
  (h4 : total_fee = 324) :
  ∃ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = total_cars ∧
    medium_cars * medium_fee + small_cars * small_fee = total_fee ∧
    medium_cars = 12 ∧
    small_cars = 18 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l746_74699


namespace NUMINAMATH_CALUDE_intersection_triangle_area_l746_74608

-- Define the line L: x - 2y - 5 = 0
def L (x y : ℝ) : Prop := x - 2*y - 5 = 0

-- Define the circle C: x^2 + y^2 = 50
def C (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the intersection points
def A : ℝ × ℝ := (-5, -5)
def B : ℝ × ℝ := (7, 1)

-- Theorem statement
theorem intersection_triangle_area :
  L A.1 A.2 ∧ L B.1 B.2 ∧ C A.1 A.2 ∧ C B.1 B.2 →
  abs ((A.1 * B.2 - B.1 * A.2) / 2) = 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_triangle_area_l746_74608


namespace NUMINAMATH_CALUDE_chord_length_l746_74628

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 2 * y - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 2}
  let chord := line ∩ circle
  ∃ (a b : ℝ), (a, b) ∈ chord ∧ 
    ∃ (c d : ℝ), (c, d) ∈ chord ∧ 
      (a - c)^2 + (b - d)^2 = (4 * Real.sqrt 3 / 3)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l746_74628


namespace NUMINAMATH_CALUDE_fraction_power_four_l746_74673

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l746_74673


namespace NUMINAMATH_CALUDE_probability_two_fives_l746_74672

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_value : ℕ := 5
def target_count : ℕ := 2

theorem probability_two_fives (num_dice : ℕ) (num_sides : ℕ) (target_value : ℕ) (target_count : ℕ) :
  num_dice = 12 →
  num_sides = 6 →
  target_value = 5 →
  target_count = 2 →
  (Nat.choose num_dice target_count : ℚ) * (1 / num_sides : ℚ)^target_count * ((num_sides - 1) / num_sides : ℚ)^(num_dice - target_count) =
  (66 * 5^10 : ℚ) / 6^12 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_fives_l746_74672


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l746_74660

-- Define the complex number z
def z : ℂ := sorry

-- State the given condition
axiom z_condition : (1 - Complex.I) * z = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l746_74660


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l746_74659

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, 2 * x^2 + b * x + 16 = 0 → x.im ≠ 0) ↔ b ∈ Set.Ioo (-8 * Real.sqrt 2) (8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l746_74659


namespace NUMINAMATH_CALUDE_four_possible_ones_digits_l746_74634

-- Define a function to check if a number is divisible by 6
def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

-- Define a function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem four_possible_ones_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, ones_digit n ∈ s ∧ divisible_by_six n) ∧
    (∀ d, d ∈ s ↔ ∃ n, ones_digit n = d ∧ divisible_by_six n) ∧
    Finset.card s = 4 :=
sorry

end NUMINAMATH_CALUDE_four_possible_ones_digits_l746_74634


namespace NUMINAMATH_CALUDE_total_points_scored_l746_74677

/-- Given a player who plays 13 games and scores 7 points in each game,
    the total number of points scored is equal to 91. -/
theorem total_points_scored (games : ℕ) (points_per_game : ℕ) : 
  games = 13 → points_per_game = 7 → games * points_per_game = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_points_scored_l746_74677


namespace NUMINAMATH_CALUDE_original_number_l746_74607

theorem original_number : ∃ x : ℤ, 63 - 2 * x = 51 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l746_74607


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l746_74612

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l746_74612


namespace NUMINAMATH_CALUDE_log_inequality_l746_74606

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log b) / (a - 1) = (a + 1) / a) : 
  Real.log b / Real.log a > 2 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l746_74606


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l746_74669

noncomputable def f (x : ℝ) := x^3 - 3*x - 1

theorem f_monotonicity_and_intersection (x : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ ((x₁ < -1 ∧ x₂ < -1) ∨ (x₁ > 1 ∧ x₂ > 1)) → f x₁ < f x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔ -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l746_74669


namespace NUMINAMATH_CALUDE_math_score_calculation_math_score_is_75_l746_74662

theorem math_score_calculation (avg_four : ℝ) (drop : ℝ) : ℝ :=
  let total_four := 4 * avg_four
  let avg_five := avg_four - drop
  let total_five := 5 * avg_five
  total_five - total_four

theorem math_score_is_75 :
  math_score_calculation 90 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_math_score_is_75_l746_74662


namespace NUMINAMATH_CALUDE_student_count_problem_l746_74617

theorem student_count_problem :
  ∃! (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    22 * (a + b + c) = 2 * (100 * a + 10 * b + c) ∧
    100 * a + 10 * b + c = 198 :=
by sorry

end NUMINAMATH_CALUDE_student_count_problem_l746_74617


namespace NUMINAMATH_CALUDE_lucas_50th_term_mod_5_l746_74695

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_50th_term_mod_5 : lucas 49 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_50th_term_mod_5_l746_74695


namespace NUMINAMATH_CALUDE_machine_production_time_l746_74631

/-- The number of items the machine can produce in one hour -/
def items_per_hour : ℕ := 90

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The time it takes to produce one item in minutes -/
def time_per_item : ℚ := minutes_per_hour / items_per_hour

theorem machine_production_time : time_per_item = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_time_l746_74631


namespace NUMINAMATH_CALUDE_rain_probability_l746_74623

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l746_74623


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l746_74642

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000) → m % 5 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l746_74642


namespace NUMINAMATH_CALUDE_price_increase_and_discount_l746_74602

theorem price_increase_and_discount (original_price : ℝ) (increase_percentage : ℝ) :
  original_price * (1 + increase_percentage) * (1 - 0.2) = original_price →
  increase_percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_and_discount_l746_74602
