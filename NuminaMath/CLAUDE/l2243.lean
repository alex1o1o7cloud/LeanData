import Mathlib

namespace NUMINAMATH_CALUDE_max_assembly_and_impossibility_of_simultaneous_completion_l2243_224396

/-- Represents the number of wooden boards available -/
structure WoodenBoards :=
  (typeA : ℕ)
  (typeB : ℕ)

/-- Represents the requirements for assembling a desk and a chair -/
structure AssemblyRequirements :=
  (deskTypeA : ℕ)
  (deskTypeB : ℕ)
  (chairTypeA : ℕ)
  (chairTypeB : ℕ)

/-- Represents the assembly time for a desk and a chair -/
structure AssemblyTime :=
  (desk : ℕ)
  (chair : ℕ)

/-- Theorem stating the maximum number of desks and chairs that can be assembled
    and the impossibility of simultaneous completion -/
theorem max_assembly_and_impossibility_of_simultaneous_completion
  (boards : WoodenBoards)
  (requirements : AssemblyRequirements)
  (students : ℕ)
  (time : AssemblyTime)
  (h1 : boards.typeA = 400)
  (h2 : boards.typeB = 500)
  (h3 : requirements.deskTypeA = 2)
  (h4 : requirements.deskTypeB = 1)
  (h5 : requirements.chairTypeA = 1)
  (h6 : requirements.chairTypeB = 2)
  (h7 : students = 30)
  (h8 : time.desk = 10)
  (h9 : time.chair = 7) :
  (∃ (desks chairs : ℕ),
    desks = 100 ∧
    chairs = 200 ∧
    desks * requirements.deskTypeA + chairs * requirements.chairTypeA ≤ boards.typeA ∧
    desks * requirements.deskTypeB + chairs * requirements.chairTypeB ≤ boards.typeB ∧
    ∀ (desks' chairs' : ℕ),
      desks' > desks ∨ chairs' > chairs →
      desks' * requirements.deskTypeA + chairs' * requirements.chairTypeA > boards.typeA ∨
      desks' * requirements.deskTypeB + chairs' * requirements.chairTypeB > boards.typeB) ∧
  (∀ (group : ℕ),
    group ≤ students →
    (desks : ℚ) * time.desk / group ≠ (chairs : ℚ) * time.chair / (students - group)) :=
by sorry

end NUMINAMATH_CALUDE_max_assembly_and_impossibility_of_simultaneous_completion_l2243_224396


namespace NUMINAMATH_CALUDE_bee_population_theorem_bee_problem_solution_l2243_224377

/-- Represents the daily change in bee population -/
def daily_change (hatch_rate : ℕ) (loss_rate : ℕ) : ℤ :=
  hatch_rate - loss_rate

/-- Calculates the final bee population after a given number of days -/
def final_population (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_change hatch_rate loss_rate

/-- Theorem stating the relationship between initial population, hatch rate, loss rate, and final population -/
theorem bee_population_theorem (initial : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) (final : ℕ) :
  final_population initial hatch_rate loss_rate days = final ↔ loss_rate = 899 :=
by
  sorry

#eval final_population 12500 3000 899 7  -- Should evaluate to 27201

/-- Main theorem proving the specific case in the problem -/
theorem bee_problem_solution :
  final_population 12500 3000 899 7 = 27201 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_population_theorem_bee_problem_solution_l2243_224377


namespace NUMINAMATH_CALUDE_pascal_triangle_fifth_number_l2243_224313

theorem pascal_triangle_fifth_number : 
  let row := List.cons 1 (List.cons 15 (List.replicate 3 0))  -- represents the start of the row
  let fifth_number := Nat.choose 15 4  -- represents ₁₅C₄
  fifth_number = 1365 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_fifth_number_l2243_224313


namespace NUMINAMATH_CALUDE_partner_A_share_is_8160_l2243_224327

/-- Calculates the share of profit for partner A in a business partnership --/
def partner_A_share (total_profit : ℚ) (A_investment : ℚ) (B_investment : ℚ) (management_fee_percent : ℚ) : ℚ :=
  let management_fee := total_profit * management_fee_percent / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := A_investment + B_investment
  let A_proportion := A_investment / total_investment
  management_fee + (remaining_profit * A_proportion)

/-- Theorem stating that partner A's share is 8160 Rs under given conditions --/
theorem partner_A_share_is_8160 :
  partner_A_share 9600 5000 1000 10 = 8160 := by
  sorry

end NUMINAMATH_CALUDE_partner_A_share_is_8160_l2243_224327


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l2243_224309

theorem largest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    40 * n ≡ 140 [MOD 320] ∧
    ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 40 * m ≡ 140 [MOD 320]) → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l2243_224309


namespace NUMINAMATH_CALUDE_max_value_of_d_l2243_224394

theorem max_value_of_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ + (5 + Real.sqrt 105) / 2 = 10 ∧
                    a₀ * b₀ + a₀ * c₀ + a₀ * ((5 + Real.sqrt 105) / 2) + 
                    b₀ * c₀ + b₀ * ((5 + Real.sqrt 105) / 2) + 
                    c₀ * ((5 + Real.sqrt 105) / 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_d_l2243_224394


namespace NUMINAMATH_CALUDE_modular_difference_in_range_l2243_224365

def problem (a b : ℤ) : Prop :=
  a % 36 = 22 ∧ b % 36 = 85

def valid_range (n : ℤ) : Prop :=
  120 ≤ n ∧ n ≤ 161

theorem modular_difference_in_range (a b : ℤ) (h : problem a b) :
  ∃! n : ℤ, valid_range n ∧ (a - b) % 36 = n % 36 ∧ n = 153 := by sorry

end NUMINAMATH_CALUDE_modular_difference_in_range_l2243_224365


namespace NUMINAMATH_CALUDE_room_diagonal_l2243_224320

theorem room_diagonal (l h d : ℝ) (b : ℝ) : 
  l = 12 → h = 9 → d = 17 → d^2 = l^2 + b^2 + h^2 → b = 8 := by sorry

end NUMINAMATH_CALUDE_room_diagonal_l2243_224320


namespace NUMINAMATH_CALUDE_range_of_x_given_integer_part_l2243_224383

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the theorem
theorem range_of_x_given_integer_part (x : ℝ) :
  integerPart ((1 - 3*x) / 2) = -1 → 1/3 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_given_integer_part_l2243_224383


namespace NUMINAMATH_CALUDE_play_admission_receipts_l2243_224336

/-- Calculates the total admission receipts for a play -/
def totalAdmissionReceipts (totalPeople : ℕ) (adultPrice childPrice : ℕ) (children : ℕ) : ℕ :=
  let adults := totalPeople - children
  adults * adultPrice + children * childPrice

/-- Theorem: The total admission receipts for the play is $960 -/
theorem play_admission_receipts :
  totalAdmissionReceipts 610 2 1 260 = 960 := by
  sorry

end NUMINAMATH_CALUDE_play_admission_receipts_l2243_224336


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2243_224388

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x * (1 - 2*x)^4 = a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₃ + a₄ + a₅ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2243_224388


namespace NUMINAMATH_CALUDE_candy_necklace_packs_opened_l2243_224321

/-- Proves the number of candy necklace packs Emily opened for her classmates -/
theorem candy_necklace_packs_opened
  (total_packs : ℕ)
  (necklaces_per_pack : ℕ)
  (necklaces_left : ℕ)
  (h1 : total_packs = 9)
  (h2 : necklaces_per_pack = 8)
  (h3 : necklaces_left = 40) :
  (total_packs * necklaces_per_pack - necklaces_left) / necklaces_per_pack = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_necklace_packs_opened_l2243_224321


namespace NUMINAMATH_CALUDE_f_2015_equals_negative_5_l2243_224348

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2015_equals_negative_5
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : is_periodic f 4)
  (h3 : f 1 = 5) :
  f 2015 = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_equals_negative_5_l2243_224348


namespace NUMINAMATH_CALUDE_factor_implies_m_equals_one_l2243_224316

theorem factor_implies_m_equals_one (m : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 42 = (x + 6) * k) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_m_equals_one_l2243_224316


namespace NUMINAMATH_CALUDE_rebate_percentage_l2243_224393

theorem rebate_percentage (num_pairs : ℕ) (price_per_pair : ℚ) (total_rebate : ℚ) :
  num_pairs = 5 →
  price_per_pair = 28 →
  total_rebate = 14 →
  (total_rebate / (num_pairs * price_per_pair)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rebate_percentage_l2243_224393


namespace NUMINAMATH_CALUDE_multiply_63_57_l2243_224379

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l2243_224379


namespace NUMINAMATH_CALUDE_min_length_shared_side_l2243_224335

/-- Given two triangles ABC and DBC sharing side BC, with AB = 8, AC = 15, DC = 10, and BD = 25,
    the minimum possible integer length of BC is 15. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → 
  BC > AC - AB → BC > BD - DC → 
  BC ≥ 15 ∧ ∀ n : ℕ, n < 15 → ¬(BC = n) :=
by sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l2243_224335


namespace NUMINAMATH_CALUDE_unique_single_digit_cube_equation_l2243_224364

theorem unique_single_digit_cube_equation :
  ∃! (A : ℕ), A ∈ Finset.range 10 ∧ A ≠ 0 ∧ A^3 = 210 + A :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_single_digit_cube_equation_l2243_224364


namespace NUMINAMATH_CALUDE_inflection_point_and_concavity_l2243_224334

-- Define the function f(x) = x³ - 3x² + 5
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the first derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem inflection_point_and_concavity :
  -- The inflection point occurs at x = 1
  (∃ (ε : ℝ), ε > 0 ∧ 
    (∀ x ∈ Set.Ioo (1 - ε) 1, f'' x < 0) ∧
    (∀ x ∈ Set.Ioo 1 (1 + ε), f'' x > 0)) ∧
  -- f(1) = 3
  f 1 = 3 ∧
  -- The function is concave down for x < 1
  (∀ x < 1, f'' x < 0) ∧
  -- The function is concave up for x > 1
  (∀ x > 1, f'' x > 0) :=
by sorry

end NUMINAMATH_CALUDE_inflection_point_and_concavity_l2243_224334


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l2243_224300

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  focusA : Point
  focusB : Point
  passingPoint : Point

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

theorem ellipse_a_plus_k (e : Ellipse) :
  e.focusA = ⟨0, 1⟩ →
  e.focusB = ⟨0, -3⟩ →
  e.passingPoint = ⟨5, 0⟩ →
  e.a > 0 →
  e.b > 0 →
  satisfiesEllipseEquation e e.passingPoint →
  e.a + e.k = (Real.sqrt 26 + Real.sqrt 34 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l2243_224300


namespace NUMINAMATH_CALUDE_initial_visual_range_proof_l2243_224314

/-- The initial visual range without the telescope -/
def initial_range : ℝ := 50

/-- The visual range with the telescope -/
def telescope_range : ℝ := 150

/-- The percentage increase in visual range -/
def percentage_increase : ℝ := 200

theorem initial_visual_range_proof :
  initial_range = telescope_range / (1 + percentage_increase / 100) :=
by sorry

end NUMINAMATH_CALUDE_initial_visual_range_proof_l2243_224314


namespace NUMINAMATH_CALUDE_square_of_1017_l2243_224345

theorem square_of_1017 : (1017 : ℕ)^2 = 1034289 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1017_l2243_224345


namespace NUMINAMATH_CALUDE_problem_statement_l2243_224376

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) 
  (h2 : a ≤ 13) 
  (h3 : (51 ^ 2016 - a) % 13 = 0) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2243_224376


namespace NUMINAMATH_CALUDE_complex_addition_multiplication_l2243_224386

theorem complex_addition_multiplication : 
  let z₁ : ℂ := 2 + 6 * I
  let z₂ : ℂ := 5 - 3 * I
  3 * (z₁ + z₂) = 21 + 9 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_multiplication_l2243_224386


namespace NUMINAMATH_CALUDE_fraction_equality_l2243_224368

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 8 3) / (hash_op 8 3) = -15 / 133 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2243_224368


namespace NUMINAMATH_CALUDE_modular_exponentiation_difference_l2243_224370

theorem modular_exponentiation_difference (n : ℕ) :
  (45^2011 - 23^2011) % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_modular_exponentiation_difference_l2243_224370


namespace NUMINAMATH_CALUDE_jacks_recycling_l2243_224323

/-- Proves the number of cans Jack recycled given the deposit amounts and quantities of other items --/
theorem jacks_recycling
  (bottle_deposit : ℚ)
  (can_deposit : ℚ)
  (glass_deposit : ℚ)
  (num_bottles : ℕ)
  (num_glass : ℕ)
  (total_earnings : ℚ)
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : glass_deposit = 15 / 100)
  (h4 : num_bottles = 80)
  (h5 : num_glass = 50)
  (h6 : total_earnings = 25) :
  (total_earnings - (num_bottles * bottle_deposit + num_glass * glass_deposit)) / can_deposit = 190 := by
  sorry

end NUMINAMATH_CALUDE_jacks_recycling_l2243_224323


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2243_224317

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2243_224317


namespace NUMINAMATH_CALUDE_y_value_l2243_224369

theorem y_value (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2243_224369


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2243_224375

/-- Given that z = (a^2 - 1) + (a - 1)i is a purely imaginary number and a is real,
    prove that (a^2 + i) / (1 + ai) = i -/
theorem complex_fraction_equals_i (a : ℝ) (h : (a^2 - 1 : ℂ) + (a - 1)*I = (0 : ℂ) + I * ((a - 1 : ℝ) : ℂ)) :
  (a^2 + I) / (1 + a*I) = I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2243_224375


namespace NUMINAMATH_CALUDE_mickey_vs_twice_minnie_l2243_224360

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per week -/
def mickey_horses_per_week : ℕ := 98

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := mickey_horses_per_week / days_in_week

theorem mickey_vs_twice_minnie :
  2 * minnie_horses_per_day - mickey_horses_per_day = 6 :=
sorry

end NUMINAMATH_CALUDE_mickey_vs_twice_minnie_l2243_224360


namespace NUMINAMATH_CALUDE_arccos_sin_one_point_five_l2243_224382

theorem arccos_sin_one_point_five (π : Real) :
  π = 3.14159265358979323846 →
  Real.arccos (Real.sin 1.5) = 0.0708 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_point_five_l2243_224382


namespace NUMINAMATH_CALUDE_max_value_of_f_l2243_224304

-- Define the function f on [1, 4]
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem max_value_of_f :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Icc 1 4, f x = x^2 - 4*x + 5) →  -- f(x) = x^2 - 4x + 5 for x ∈ [1, 4]
  (∃ c ∈ Set.Icc (-4) (-1), ∀ x ∈ Set.Icc (-4) (-1), f x ≤ f c) →  -- maximum exists on [-4, -1]
  (∀ x ∈ Set.Icc (-4) (-1), f x ≤ -1) ∧  -- maximum value is at most -1
  (∃ x ∈ Set.Icc (-4) (-1), f x = -1)  -- maximum value -1 is achieved
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2243_224304


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l2243_224362

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- A circle can be inscribed within the quadrilateral -/
  has_inscribed_circle : Bool
  /-- A circle can be circumscribed around the quadrilateral -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  -- A square has all sides equal and all angles right angles
  sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, 
    and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral, q.perp_diagonals ∧ q.has_inscribed_circle ∧ q.has_circumscribed_circle ∧ ¬is_square q :=
by
  sorry


end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l2243_224362


namespace NUMINAMATH_CALUDE_profit_sharing_l2243_224343

/-- Profit sharing in a partnership --/
theorem profit_sharing
  (tom_investment jerry_investment : ℝ)
  (total_profit : ℝ)
  (tom_extra : ℝ)
  (h1 : tom_investment = 700)
  (h2 : jerry_investment = 300)
  (h3 : total_profit = 3000)
  (h4 : tom_extra = 800) :
  ∃ (equal_portion : ℝ),
    equal_portion = 1000 ∧
    (equal_portion / 2 + (tom_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion)) =
    (equal_portion / 2 + (jerry_investment / (tom_investment + jerry_investment)) * (total_profit - equal_portion) + tom_extra) :=
by sorry

end NUMINAMATH_CALUDE_profit_sharing_l2243_224343


namespace NUMINAMATH_CALUDE_largest_product_l2243_224319

def S : Finset Int := {-4, -3, -1, 5, 6, 7}

def isConsecutive (a b : Int) : Prop := b = a + 1 ∨ a = b + 1

def fourDistinctElements (a b c d : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def twoConsecutive (a b c d : Int) : Prop :=
  isConsecutive a b ∨ isConsecutive a c ∨ isConsecutive a d ∨
  isConsecutive b c ∨ isConsecutive b d ∨ isConsecutive c d

theorem largest_product :
  ∀ a b c d : Int,
    fourDistinctElements a b c d →
    twoConsecutive a b c d →
    a * b * c * d ≤ -210 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_l2243_224319


namespace NUMINAMATH_CALUDE_fourth_month_sales_l2243_224338

def sales_problem (m1 m2 m3 m5 m6 average : ℕ) : Prop :=
  ∃ m4 : ℕ, (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem fourth_month_sales :
  sales_problem 6435 6927 6855 6562 7391 6900 →
  ∃ m4 : ℕ, m4 = 7230 ∧ (6435 + 6927 + 6855 + m4 + 6562 + 7391) / 6 = 6900 :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l2243_224338


namespace NUMINAMATH_CALUDE_highest_numbered_street_l2243_224330

/-- Represents the length of Apple Street in meters -/
def street_length : ℕ := 15000

/-- Represents the distance between intersections in meters -/
def intersection_distance : ℕ := 500

/-- Calculates the number of numbered intersecting streets -/
def numbered_intersections : ℕ :=
  (street_length / intersection_distance) - 2

/-- Proves that the highest-numbered street is the 28th Street -/
theorem highest_numbered_street :
  numbered_intersections = 28 := by
  sorry

end NUMINAMATH_CALUDE_highest_numbered_street_l2243_224330


namespace NUMINAMATH_CALUDE_grocer_coffee_percentage_l2243_224378

/-- Calculates the percentage of decaffeinated coffee in a grocer's stock -/
theorem grocer_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60)
  : (initial_stock * initial_decaf_percent / 100 + additional_stock * additional_decaf_percent / 100) /
    (initial_stock + additional_stock) * 100 = 28 := by
  sorry

end NUMINAMATH_CALUDE_grocer_coffee_percentage_l2243_224378


namespace NUMINAMATH_CALUDE_sum_11_is_negative_11_l2243_224325

/-- An arithmetic sequence with its sum of terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  first_term : a 1 = -11
  sum_condition : S 10 / 10 - S 8 / 8 = 2

/-- The sum of the first 11 terms in the given arithmetic sequence is -11 -/
theorem sum_11_is_negative_11 (seq : ArithmeticSequence) : seq.S 11 = -11 := by
  sorry

end NUMINAMATH_CALUDE_sum_11_is_negative_11_l2243_224325


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l2243_224361

/-- Represents the speed of a canoe in different conditions -/
structure CanoeSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a canoe given its speed in still water and upstream -/
def downstreamSpeed (c : CanoeSpeed) : ℝ :=
  2 * c.stillWater - c.upstream

/-- Theorem stating that for a canoe with 12.5 km/hr speed in still water and 9 km/hr upstream speed, 
    the downstream speed is 16 km/hr -/
theorem canoe_downstream_speed :
  let c : CanoeSpeed := { stillWater := 12.5, upstream := 9 }
  downstreamSpeed c = 16 := by
  sorry


end NUMINAMATH_CALUDE_canoe_downstream_speed_l2243_224361


namespace NUMINAMATH_CALUDE_cube_volume_l2243_224354

/-- Given a cube with side perimeter 32 cm, its volume is 512 cubic cm. -/
theorem cube_volume (side_perimeter : ℝ) (h : side_perimeter = 32) : 
  (side_perimeter / 4)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l2243_224354


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2243_224374

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2243_224374


namespace NUMINAMATH_CALUDE_crayons_distribution_l2243_224308

/-- Given a total number of crayons and a number of boxes, 
    calculate the number of crayons per box -/
def crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) : ℕ :=
  total_crayons / num_boxes

/-- Theorem stating that given 80 crayons and 10 boxes, 
    the number of crayons per box is 8 -/
theorem crayons_distribution :
  crayons_per_box 80 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_distribution_l2243_224308


namespace NUMINAMATH_CALUDE_candy_sampling_theorem_l2243_224311

theorem candy_sampling_theorem (caught_percentage : Real) (total_sampling_percentage : Real)
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 24.444444444444443) :
  total_sampling_percentage - caught_percentage = 2.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_theorem_l2243_224311


namespace NUMINAMATH_CALUDE_leg_length_in_special_right_isosceles_triangle_l2243_224306

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0

/-- Theorem: In a 45-45-90 triangle with hypotenuse 12√2, the length of a leg is 12 -/
theorem leg_length_in_special_right_isosceles_triangle 
  (triangle : RightIsoscelesTriangle) 
  (h : triangle.hypotenuse = 12 * Real.sqrt 2) : 
  triangle.hypotenuse / Real.sqrt 2 = 12 := by
  sorry

#check leg_length_in_special_right_isosceles_triangle

end NUMINAMATH_CALUDE_leg_length_in_special_right_isosceles_triangle_l2243_224306


namespace NUMINAMATH_CALUDE_equation_system_solution_l2243_224324

theorem equation_system_solution (x z : ℝ) 
  (eq1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (eq2 : 3 * x + z + 4 = 0) :
  z^2 + 20 * z - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2243_224324


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2243_224351

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2243_224351


namespace NUMINAMATH_CALUDE_third_largest_three_digit_with_eight_ones_l2243_224367

/-- Given a list of digits, returns all three-digit numbers that can be formed using exactly three of those digits. -/
def threeDigitNumbers (digits : List Nat) : List Nat := sorry

/-- Checks if a number has 8 in the ones place. -/
def hasEightInOnes (n : Nat) : Bool := sorry

/-- The third largest element in a list of natural numbers. -/
def thirdLargest (numbers : List Nat) : Nat := sorry

theorem third_largest_three_digit_with_eight_ones : 
  let digits := [0, 1, 4, 8]
  let validNumbers := (threeDigitNumbers digits).filter hasEightInOnes
  thirdLargest validNumbers = 148 := by sorry

end NUMINAMATH_CALUDE_third_largest_three_digit_with_eight_ones_l2243_224367


namespace NUMINAMATH_CALUDE_right_to_left_equiv_standard_not_equiv_l2243_224347

/-- Evaluates an expression in a right-to-left order -/
noncomputable def evaluateRightToLeft (a b c d : ℝ) : ℝ :=
  a / (b - c - d)

/-- Standard algebraic evaluation -/
noncomputable def evaluateStandard (a b c d : ℝ) : ℝ :=
  a / b - c + d

/-- Theorem stating the equivalence of right-to-left evaluation and the correct standard algebraic form -/
theorem right_to_left_equiv (a b c d : ℝ) :
  evaluateRightToLeft a b c d = a / (b - c - d) :=
by sorry

/-- Theorem stating that the standard algebraic evaluation is not equivalent to the right-to-left evaluation -/
theorem standard_not_equiv (a b c d : ℝ) :
  evaluateStandard a b c d ≠ evaluateRightToLeft a b c d :=
by sorry

end NUMINAMATH_CALUDE_right_to_left_equiv_standard_not_equiv_l2243_224347


namespace NUMINAMATH_CALUDE_pauline_total_spend_l2243_224398

/-- The total amount Pauline will spend, including sales tax -/
def total_amount (pre_tax_amount : ℝ) (tax_rate : ℝ) : ℝ :=
  pre_tax_amount * (1 + tax_rate)

/-- Proof that Pauline will spend $162 on all items, including sales tax -/
theorem pauline_total_spend :
  total_amount 150 0.08 = 162 := by
  sorry

end NUMINAMATH_CALUDE_pauline_total_spend_l2243_224398


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l2243_224318

theorem distinct_primes_in_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (85 * 87 * 88 * 90) % p = 0 → p ∈ s) ∧ 
  Finset.card s = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l2243_224318


namespace NUMINAMATH_CALUDE_bmw_sales_l2243_224391

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (toyota_percent : ℚ) (nissan_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 10 / 100)
  (h_toyota : toyota_percent = 20 / 100)
  (h_nissan : nissan_percent = 30 / 100) :
  (total : ℚ) * (1 - (ford_percent + toyota_percent + nissan_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_l2243_224391


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2243_224337

/-- Represents a faulty meter with its weight and associated profit percentage -/
structure FaultyMeter where
  weight : ℕ
  profit_percentage : ℚ

/-- Calculates the weighted profit for a meter given its profit percentage and sales volume ratio -/
def weighted_profit (meter : FaultyMeter) (sales_ratio : ℚ) (total_ratio : ℚ) : ℚ :=
  meter.profit_percentage * (sales_ratio / total_ratio)

/-- Theorem stating that the overall profit percentage is 11.6% given the conditions -/
theorem shopkeeper_profit_percentage 
  (meter1 : FaultyMeter)
  (meter2 : FaultyMeter)
  (meter3 : FaultyMeter)
  (h1 : meter1.weight = 900)
  (h2 : meter2.weight = 850)
  (h3 : meter3.weight = 950)
  (h4 : meter1.profit_percentage = 1/10)
  (h5 : meter2.profit_percentage = 12/100)
  (h6 : meter3.profit_percentage = 15/100)
  (sales_ratio1 : ℚ)
  (sales_ratio2 : ℚ)
  (sales_ratio3 : ℚ)
  (h7 : sales_ratio1 = 5)
  (h8 : sales_ratio2 = 3)
  (h9 : sales_ratio3 = 2) :
  weighted_profit meter1 sales_ratio1 (sales_ratio1 + sales_ratio2 + sales_ratio3) +
  weighted_profit meter2 sales_ratio2 (sales_ratio1 + sales_ratio2 + sales_ratio3) +
  weighted_profit meter3 sales_ratio3 (sales_ratio1 + sales_ratio2 + sales_ratio3) =
  116/1000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2243_224337


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2243_224341

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 1
  y = 2*x^2 → y = shifted.a * (x + 4)^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2243_224341


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2243_224307

/-- Calculates the total surface area of a cube with holes cut through each face --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem stating that the total surface area of the given cube with holes is 222 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 5 2 = 222 := by
  sorry

#eval total_surface_area 5 2

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2243_224307


namespace NUMINAMATH_CALUDE_march_1900_rainfall_average_l2243_224372

/-- The average rainfall per minute given total rainfall and number of days -/
def average_rainfall_per_minute (total_rainfall : ℚ) (days : ℕ) : ℚ :=
  total_rainfall / (days * 24 * 60)

/-- Theorem stating that 620 inches of rainfall over 15 days results in an average of 31/1080 inches per minute -/
theorem march_1900_rainfall_average : 
  average_rainfall_per_minute 620 15 = 31 / 1080 := by
  sorry

end NUMINAMATH_CALUDE_march_1900_rainfall_average_l2243_224372


namespace NUMINAMATH_CALUDE_largest_possible_number_david_l2243_224356

/-- Represents a decimal number with up to two digits before and after the decimal point -/
structure DecimalNumber :=
  (beforeDecimal : Fin 100)
  (afterDecimal : Fin 100)

/-- Checks if a DecimalNumber has mutually different digits -/
def hasMutuallyDifferentDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Checks if a DecimalNumber has exactly two identical digits -/
def hasExactlyTwoIdenticalDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Converts a DecimalNumber to a rational number -/
def toRational (n : DecimalNumber) : ℚ :=
  sorry

/-- The sum of two DecimalNumbers -/
def sum (a b : DecimalNumber) : ℚ :=
  toRational a + toRational b

theorem largest_possible_number_david
  (jana david : DecimalNumber)
  (h_sum : sum jana david = 11.11)
  (h_david_digits : hasMutuallyDifferentDigits david)
  (h_jana_digits : hasExactlyTwoIdenticalDigits jana) :
  toRational david ≤ 0.9 :=
sorry

end NUMINAMATH_CALUDE_largest_possible_number_david_l2243_224356


namespace NUMINAMATH_CALUDE_students_taking_statistics_l2243_224371

theorem students_taking_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (history_or_statistics : ℕ) 
  (history_not_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 29) :
  ∃ (statistics : ℕ), statistics = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_statistics_l2243_224371


namespace NUMINAMATH_CALUDE_sugar_left_in_grams_l2243_224305

/-- The amount of sugar Pamela bought in ounces -/
def sugar_bought : ℝ := 9.8

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℝ := 28.35

/-- Theorem stating the amount of sugar Pamela has left in grams -/
theorem sugar_left_in_grams : 
  (sugar_bought - sugar_spilled) * oz_to_g = 130.41 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_in_grams_l2243_224305


namespace NUMINAMATH_CALUDE_total_cost_of_promotional_items_l2243_224366

/-- The cost of a calendar in dollars -/
def calendar_cost : ℚ := 3/4

/-- The cost of a date book in dollars -/
def date_book_cost : ℚ := 1/2

/-- The number of calendars ordered -/
def calendars_ordered : ℕ := 300

/-- The number of date books ordered -/
def date_books_ordered : ℕ := 200

/-- The total number of items ordered -/
def total_items : ℕ := 500

/-- Theorem stating the total cost of promotional items -/
theorem total_cost_of_promotional_items :
  calendars_ordered * calendar_cost + date_books_ordered * date_book_cost = 325/1 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_of_promotional_items_l2243_224366


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2243_224322

/-- A line given by parametric equations is tangent to a circle. -/
theorem line_tangent_to_circle (α : Real) (h1 : α > π / 2) :
  (∃ t : Real, ∀ φ : Real,
    let x_line := t * Real.cos α
    let y_line := t * Real.sin α
    let x_circle := 4 + 2 * Real.cos φ
    let y_circle := 2 * Real.sin φ
    (x_line - x_circle)^2 + (y_line - y_circle)^2 = 4) →
  α = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2243_224322


namespace NUMINAMATH_CALUDE_megan_songs_count_l2243_224353

/-- The number of songs Megan bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating the total number of songs Megan bought -/
theorem megan_songs_count :
  total_songs 2 8 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_songs_count_l2243_224353


namespace NUMINAMATH_CALUDE_fairview_soccer_contest_l2243_224329

/-- Calculates the number of penalty kicks in a soccer team contest --/
def penalty_kicks (total_players : ℕ) (initial_goalies : ℕ) (absent_players : ℕ) (absent_goalies : ℕ) : ℕ :=
  let remaining_players := total_players - absent_players
  let remaining_goalies := initial_goalies - absent_goalies
  remaining_goalies * (remaining_players - 1)

/-- Theorem stating the number of penalty kicks for the Fairview College Soccer Team contest --/
theorem fairview_soccer_contest : 
  penalty_kicks 25 4 2 1 = 66 := by
  sorry

end NUMINAMATH_CALUDE_fairview_soccer_contest_l2243_224329


namespace NUMINAMATH_CALUDE_inequality_bound_l2243_224312

theorem inequality_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) + 
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_bound_l2243_224312


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l2243_224355

theorem rachel_homework_difference :
  let math_pages : ℕ := 2
  let reading_pages : ℕ := 3
  let total_pages : ℕ := 15
  let biology_pages : ℕ := total_pages - (math_pages + reading_pages)
  biology_pages - reading_pages = 7 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l2243_224355


namespace NUMINAMATH_CALUDE_playground_girls_l2243_224373

theorem playground_girls (total_children : ℕ) (boys : ℕ) 
  (h1 : total_children = 62) 
  (h2 : boys = 27) : 
  total_children - boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l2243_224373


namespace NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l2243_224358

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ

/-- The distance between the apex and the center of the circumsphere of the frustum -/
noncomputable def apex_to_center_distance (p : CutPyramid) : ℝ :=
  sorry

/-- Theorem stating the relationship between the pyramid's properties and the apex-to-center distance -/
theorem apex_to_center_distance_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 6
  }
  apex_to_center_distance p = 5 * (36 ^ (1/3 : ℝ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l2243_224358


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l2243_224342

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l2243_224342


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2243_224326

open Set

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2243_224326


namespace NUMINAMATH_CALUDE_waste_recovery_analysis_l2243_224301

structure WasteData where
  m : ℕ
  a : ℝ
  freq1 : ℝ
  freq2 : ℝ
  freq5 : ℝ

def WasteAnalysis (data : WasteData) : Prop :=
  data.m > 0 ∧
  0.20 ≤ data.a ∧ data.a ≤ 0.30 ∧
  data.freq1 + data.freq2 + data.a + data.freq5 = 1 ∧
  data.freq1 = 0.05 ∧
  data.freq2 = 0.10 ∧
  data.freq5 = 0.15

theorem waste_recovery_analysis (data : WasteData) 
  (h : WasteAnalysis data) : 
  data.m = 20 ∧ 
  (∃ (median : ℝ), 4 ≤ median ∧ median < 5) ∧
  (∃ (avg : ℝ), avg ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_waste_recovery_analysis_l2243_224301


namespace NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l2243_224359

theorem goose_egg_hatch_fraction (eggs : ℕ) (hatched : ℕ) 
  (h1 : hatched ≤ eggs) 
  (h2 : (4 : ℚ) / 5 * ((2 : ℚ) / 5 * hatched) = 120) : 
  (hatched : ℚ) / eggs = 1 := by
  sorry

end NUMINAMATH_CALUDE_goose_egg_hatch_fraction_l2243_224359


namespace NUMINAMATH_CALUDE_domain_exclusion_sum_l2243_224385

theorem domain_exclusion_sum (A B : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ x = A ∨ x = B) → A + B = 3 := by
sorry

end NUMINAMATH_CALUDE_domain_exclusion_sum_l2243_224385


namespace NUMINAMATH_CALUDE_max_third_side_length_l2243_224346

theorem max_third_side_length (a b x : ℕ) (ha : a = 28) (hb : b = 47) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → x ≤ 74 :=
sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2243_224346


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2243_224339

noncomputable section

-- Define the * operation
def ast (x₁ x₂ : ℝ) : ℝ := (x₁ + x₂)^2 - (x₁ - x₂)^2

-- Define the point P
def P (x a : ℝ) : ℝ × ℝ := (x, Real.sqrt (ast x a))

-- Theorem statement
theorem trajectory_is_parabola (a : ℝ) (h₁ : a > 0) :
  ∃ k c : ℝ, ∀ x : ℝ, x ≥ 0 → (P x a).2^2 = k * (P x a).1 + c :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2243_224339


namespace NUMINAMATH_CALUDE_no_three_digit_sum_product_l2243_224392

theorem no_three_digit_sum_product : ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬(a + b + c = 100 * a + 10 * b + c - a * b * c) :=
by sorry


end NUMINAMATH_CALUDE_no_three_digit_sum_product_l2243_224392


namespace NUMINAMATH_CALUDE_iggy_running_time_l2243_224303

/-- Represents the daily running distances in miles -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Represents the pace in minutes per mile -/
def pace : Nat := 10

/-- Calculates the total running time in hours -/
def total_running_hours (miles : List Nat) (pace : Nat) : Nat :=
  (miles.sum * pace) / 60

/-- Theorem: Iggy's total running time from Monday to Friday is 4 hours -/
theorem iggy_running_time :
  total_running_hours daily_miles pace = 4 := by
  sorry

#eval total_running_hours daily_miles pace

end NUMINAMATH_CALUDE_iggy_running_time_l2243_224303


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l2243_224331

/-- The amount of grain spilled from a ship, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining on the ship, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 := by sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l2243_224331


namespace NUMINAMATH_CALUDE_fraction_calculation_l2243_224328

theorem fraction_calculation : 
  (8 / 4 * 9 / 3 * 20 / 5) / (10 / 5 * 12 / 4 * 15 / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2243_224328


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l2243_224340

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % (sumOfDigits n) = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (isMultipleOf11 132) ∧
  ¬(isLuckyInteger 132) ∧
  ∀ n : ℕ, n > 0 ∧ n < 132 ∧ (isMultipleOf11 n) → (isLuckyInteger n) := by
  sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l2243_224340


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l2243_224399

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverse digits of a two-digit integer -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem two_digit_reverse_sum (x y m : ℕ) : 
  TwoDigitInt x ∧ 
  TwoDigitInt y ∧
  y = reverseDigits x ∧
  x^2 - y^2 = 4 * m^2 ∧
  0 < m
  →
  x + y + m = 105 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l2243_224399


namespace NUMINAMATH_CALUDE_kaleb_shirts_removed_l2243_224390

/-- The number of shirts Kaleb got rid of -/
def shirts_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Kaleb got rid of 7 shirts -/
theorem kaleb_shirts_removed :
  let initial_shirts : ℕ := 17
  let remaining_shirts : ℕ := 10
  shirts_removed initial_shirts remaining_shirts = 7 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_shirts_removed_l2243_224390


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2243_224397

theorem cos_alpha_plus_pi_sixth (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : (Real.cos (2 * α)) / (1 + Real.tan α ^ 2) = 3 / 8) : 
  Real.cos (α + Real.pi / 6) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l2243_224397


namespace NUMINAMATH_CALUDE_consecutive_products_not_3000000_l2243_224350

theorem consecutive_products_not_3000000 :
  ∀ n : ℕ, (n - 1) * n + n * (n + 1) + (n - 1) * (n + 1) ≠ 3000000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_not_3000000_l2243_224350


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2243_224332

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_sum_reciprocals (a m n : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (hmn : m * n > 0) (h_line : -2 * m - n + 1 = 0) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2243_224332


namespace NUMINAMATH_CALUDE_barbecue_chicken_orders_l2243_224344

/-- Represents the number of pieces of chicken used in different dish types --/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for different dish types --/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- The total number of chicken pieces needed for all orders --/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : ℕ :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The theorem to prove --/
theorem barbecue_chicken_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.friedDinner = 2)
  (h5 : o.pasta = 6)
  (h6 : totalChickenPieces cp o = 37) :
  o.barbecue = 3 := by
  sorry

end NUMINAMATH_CALUDE_barbecue_chicken_orders_l2243_224344


namespace NUMINAMATH_CALUDE_valid_sequences_count_l2243_224315

/-- Represents a binary sequence with no consecutive 1s -/
inductive ValidSequence : Nat → Type
  | zero : ValidSequence 0
  | one : ValidSequence 1
  | appendZero : ValidSequence n → ValidSequence (n + 1)
  | appendOneZero : ValidSequence n → ValidSequence (n + 2)

/-- Counts the number of valid sequences of length n or less -/
def countValidSequences (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => countValidSequences (n+1) + countValidSequences n

theorem valid_sequences_count :
  countValidSequences 11 = 233 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l2243_224315


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2243_224384

theorem hot_dogs_remainder : 25197621 % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2243_224384


namespace NUMINAMATH_CALUDE_gaokao_probability_l2243_224363

/-- The probability of choosing both Physics and History in the Gaokao exam -/
theorem gaokao_probability (p_physics_not_history p_history_not_physics : ℝ) 
  (h1 : p_physics_not_history = 0.5)
  (h2 : p_history_not_physics = 0.3) :
  1 - p_physics_not_history - p_history_not_physics = 0.2 := by sorry

end NUMINAMATH_CALUDE_gaokao_probability_l2243_224363


namespace NUMINAMATH_CALUDE_searchlight_probability_l2243_224380

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- The number of degrees in a full circle -/
def degrees_in_circle : ℝ := 360

/-- The minimum number of seconds the man should stay in the dark -/
def min_dark_seconds : ℝ := 5

/-- The probability of a man staying in the dark for at least 5 seconds
    when a searchlight makes 2 revolutions per minute -/
theorem searchlight_probability : 
  (degrees_in_circle - (min_dark_seconds / (seconds_per_minute / revolutions_per_minute)) * degrees_in_circle) / degrees_in_circle = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_searchlight_probability_l2243_224380


namespace NUMINAMATH_CALUDE_area_of_LMNOPQ_l2243_224310

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- Represents the polygon LMNOPQ formed by two overlapping rectangles -/
structure PolygonLMNOPQ where
  lmno : Rectangle
  opqr : Rectangle
  lm : ℝ
  mn : ℝ
  no : ℝ
  -- Conditions
  h1 : lmno.a = lm
  h2 : lmno.b = mn
  h3 : opqr.a = mn
  h4 : opqr.b = lm
  h5 : lm = 8
  h6 : mn = 10
  h7 : no = 3

theorem area_of_LMNOPQ (p : PolygonLMNOPQ) : p.lmno.area = 80 := by
  sorry

#check area_of_LMNOPQ

end NUMINAMATH_CALUDE_area_of_LMNOPQ_l2243_224310


namespace NUMINAMATH_CALUDE_x_intercept_of_line_x_intercept_specific_line_l2243_224352

/-- Given two points on a line, calculate its x-intercept -/
theorem x_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = (m * x₁ - y₁) / m :=
by sorry

/-- The x-intercept of a line passing through (10, 3) and (-12, -8) is 4 -/
theorem x_intercept_specific_line :
  let x₁ : ℝ := 10
  let y₁ : ℝ := 3
  let x₂ : ℝ := -12
  let y₂ : ℝ := -8
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = 4 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_x_intercept_specific_line_l2243_224352


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l2243_224395

theorem cricket_innings_problem (initial_average : ℝ) (runs_next_inning : ℕ) (average_increase : ℝ) :
  initial_average = 15 ∧ runs_next_inning = 59 ∧ average_increase = 4 →
  ∃ n : ℕ, n = 10 ∧
    initial_average * n + runs_next_inning = (initial_average + average_increase) * (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l2243_224395


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l2243_224357

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l2243_224357


namespace NUMINAMATH_CALUDE_exists_valid_number_l2243_224302

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, (n / 10^i) % 10 ≠ 0)

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem exists_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ (n + reverse_number n) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_number_l2243_224302


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2243_224381

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate n 1 ++ [1 + 1 / n]
  (set.sum / set.length : ℚ) = 1 + 1 / (n * (n + 1)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2243_224381


namespace NUMINAMATH_CALUDE_schedule_theorem_l2243_224349

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of subjects to be scheduled -/
def num_subjects : ℕ := 4

/-- Calculates the number of ways to schedule subjects -/
def schedule_ways : ℕ := Nat.choose num_periods num_subjects * Nat.factorial num_subjects

/-- Theorem stating that the number of ways to schedule 4 subjects in 7 periods
    with no consecutive subjects is 840 -/
theorem schedule_theorem : schedule_ways = 840 := by
  sorry

end NUMINAMATH_CALUDE_schedule_theorem_l2243_224349


namespace NUMINAMATH_CALUDE_solution_value_l2243_224389

theorem solution_value (a b : ℝ) (h : a^2 + b^2 - 4*a - 6*b + 13 = 0) : 
  (a - b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_solution_value_l2243_224389


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2243_224333

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (a b : ℝ), a^2 + b^2 = 25 ∧ (x - a)^2 + (y - b)^2 = 4) →
  x^2 + y^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2243_224333


namespace NUMINAMATH_CALUDE_total_people_all_tribes_l2243_224387

/-- Represents a tribe with cannoneers, women, and men -/
structure Tribe where
  cannoneers : ℕ
  women : ℕ
  men : ℕ

/-- Calculates the total number of people in a tribe -/
def total_people (t : Tribe) : ℕ := t.cannoneers + t.women + t.men

/-- Represents the conditions for Tribe A -/
def tribe_a : Tribe :=
  { cannoneers := 63,
    women := 2 * 63,
    men := 2 * (2 * 63) }

/-- Represents the conditions for Tribe B -/
def tribe_b : Tribe :=
  { cannoneers := 45,
    women := 45 / 3,
    men := 3 * (45 / 3) }

/-- Represents the conditions for Tribe C -/
def tribe_c : Tribe :=
  { cannoneers := 108,
    women := 108 / 2,
    men := 108 / 2 }

theorem total_people_all_tribes : 
  total_people tribe_a + total_people tribe_b + total_people tribe_c = 834 := by
  sorry

end NUMINAMATH_CALUDE_total_people_all_tribes_l2243_224387
