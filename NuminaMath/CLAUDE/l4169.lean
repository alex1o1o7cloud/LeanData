import Mathlib

namespace NUMINAMATH_CALUDE_pies_sold_in_week_l4169_416972

/-- The number of pies sold daily by the restaurant -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_in_week_l4169_416972


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4169_416955

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (-1, 3)
  parallel a b → x = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4169_416955


namespace NUMINAMATH_CALUDE_labourer_savings_is_30_l4169_416947

/-- Calculates the savings of a labourer after clearing debt -/
def labourerSavings (monthlyIncome : ℕ) (initialExpenditure : ℕ) (initialMonths : ℕ)
  (reducedExpenditure : ℕ) (reducedMonths : ℕ) : ℕ :=
  let initialDebt := initialMonths * initialExpenditure - initialMonths * monthlyIncome
  let availableAmount := reducedMonths * monthlyIncome - reducedMonths * reducedExpenditure
  availableAmount - initialDebt

/-- The labourer's savings after clearing debt is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 78 85 6 60 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_labourer_savings_is_30_l4169_416947


namespace NUMINAMATH_CALUDE_divisibility_equation_solutions_l4169_416943

theorem divisibility_equation_solutions (n x y z t : ℕ+) :
  (n ^ x.val ∣ n ^ y.val + n ^ z.val) ∧ (n ^ y.val + n ^ z.val = n ^ t.val) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨
   (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equation_solutions_l4169_416943


namespace NUMINAMATH_CALUDE_other_number_proof_l4169_416970

theorem other_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 14) → 
  (Nat.lcm a b = 396) → 
  (a = 36) → 
  (b = 154) := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l4169_416970


namespace NUMINAMATH_CALUDE_watch_price_after_discounts_l4169_416950

/-- Calculates the final price of a watch after three consecutive discounts -/
def finalPrice (originalPrice : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that the final price of a 25000 rs watch after 15%, 20%, and 10% discounts is 15300 rs -/
theorem watch_price_after_discounts :
  finalPrice 25000 0.15 0.20 0.10 = 15300 := by
  sorry

end NUMINAMATH_CALUDE_watch_price_after_discounts_l4169_416950


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l4169_416933

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (p : Parabola) :
  let original := { a := -2, b := 0, c := 1 : Parabola }
  let shifted := shift_parabola original 1 2
  shifted = { a := -2, b := 4, c := 3 : Parabola } := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l4169_416933


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l4169_416916

theorem like_terms_exponent_relation (x y : ℤ) : 
  (∃ (m n : ℝ), -0.5 * m^x * n^3 = 5 * m^4 * n^y) → (y - x)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l4169_416916


namespace NUMINAMATH_CALUDE_proposition_a_is_true_l4169_416961

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end NUMINAMATH_CALUDE_proposition_a_is_true_l4169_416961


namespace NUMINAMATH_CALUDE_part_1_part_2_l4169_416953

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def satisfies_law_of_sines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

def is_geometric_sequence (a b c : Real) : Prop :=
  b * b = a * c

-- Theorem statements
theorem part_1 (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.A = 45) :
  t.a = Real.sqrt 2 := by sorry

theorem part_2 (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : is_geometric_sequence t.a t.b t.c) :
  t.A = 60 ∧ t.C = 60 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l4169_416953


namespace NUMINAMATH_CALUDE_triangle_problem_l4169_416976

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
  (h2 : a = 1)
  (h3 : Real.cos B + Real.cos C = 1) :
  Real.cos A = 1/3 ∧ c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4169_416976


namespace NUMINAMATH_CALUDE_craigs_age_l4169_416977

/-- Craig's age problem -/
theorem craigs_age (craig_age mother_age : ℕ) : 
  craig_age = mother_age - 24 →
  craig_age + mother_age = 56 →
  craig_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_craigs_age_l4169_416977


namespace NUMINAMATH_CALUDE_complex_number_modulus_l4169_416987

theorem complex_number_modulus : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l4169_416987


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4169_416902

theorem trigonometric_identity : 
  (Real.cos (28 * π / 180) * Real.cos (56 * π / 180)) / Real.sin (2 * π / 180) + 
  (Real.cos (2 * π / 180) * Real.cos (4 * π / 180)) / Real.sin (28 * π / 180) = 
  (Real.sqrt 3 * Real.sin (38 * π / 180)) / (4 * Real.sin (2 * π / 180) * Real.sin (28 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4169_416902


namespace NUMINAMATH_CALUDE_concert_tickets_theorem_l4169_416930

/-- Represents the ticket sales for a concert --/
structure ConcertTickets where
  regularPrice : ℕ
  discountGroup1Size : ℕ
  discountGroup1Percentage : ℕ
  discountGroup2Size : ℕ
  discountGroup2Percentage : ℕ
  totalRevenue : ℕ

/-- Calculates the total number of people who bought tickets --/
def totalPeople (ct : ConcertTickets) : ℕ :=
  ct.discountGroup1Size + ct.discountGroup2Size +
  ((ct.totalRevenue -
    (ct.discountGroup1Size * (ct.regularPrice * (100 - ct.discountGroup1Percentage) / 100)) -
    (ct.discountGroup2Size * (ct.regularPrice * (100 - ct.discountGroup2Percentage) / 100)))
   / ct.regularPrice)

/-- Theorem stating that given the concert conditions, 48 people bought tickets --/
theorem concert_tickets_theorem (ct : ConcertTickets)
  (h1 : ct.regularPrice = 20)
  (h2 : ct.discountGroup1Size = 10)
  (h3 : ct.discountGroup1Percentage = 40)
  (h4 : ct.discountGroup2Size = 20)
  (h5 : ct.discountGroup2Percentage = 15)
  (h6 : ct.totalRevenue = 820) :
  totalPeople ct = 48 := by
  sorry

#eval totalPeople {
  regularPrice := 20,
  discountGroup1Size := 10,
  discountGroup1Percentage := 40,
  discountGroup2Size := 20,
  discountGroup2Percentage := 15,
  totalRevenue := 820
}

end NUMINAMATH_CALUDE_concert_tickets_theorem_l4169_416930


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l4169_416914

/-- A line ax + by + c = 0 is parallel to the x-axis if and only if a = 0, b ≠ 0, and c ≠ 0 -/
def parallel_to_x_axis (a b c : ℝ) : Prop :=
  a = 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of the line in question -/
def line_equation (a x y : ℝ) : Prop :=
  (6 * a^2 - a - 2) * x + (3 * a^2 - 5 * a + 2) * y + a - 1 = 0

/-- The theorem to be proved -/
theorem line_parallel_to_x_axis (a : ℝ) :
  (∃ x y, line_equation a x y) ∧ 
  parallel_to_x_axis (6 * a^2 - a - 2) (3 * a^2 - 5 * a + 2) (a - 1) →
  a = -1/2 :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l4169_416914


namespace NUMINAMATH_CALUDE_inequality_solution_l4169_416974

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -3) ∨ (x > (-1 - Real.sqrt 61) / 6 ∧ x < (-1 + Real.sqrt 61) / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4169_416974


namespace NUMINAMATH_CALUDE_prime_and_even_under_10_composite_and_odd_under_10_l4169_416932

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k
def isOdd (n : ℕ) : Prop := ¬(isEven n)
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem prime_and_even_under_10 : ∃! n, n < 10 ∧ isPrime n ∧ isEven n :=
sorry

theorem composite_and_odd_under_10 : ∃! n, n < 10 ∧ isComposite n ∧ isOdd n :=
sorry

end NUMINAMATH_CALUDE_prime_and_even_under_10_composite_and_odd_under_10_l4169_416932


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4169_416922

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 1) ≥ 2

-- Define the solution set
def solution_set : Set ℝ := { x | 0 ≤ x ∧ x < 1 }

-- Theorem stating that the solution set is correct
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ∧ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4169_416922


namespace NUMINAMATH_CALUDE_trajectory_equation_1_trajectory_equation_1_converse_l4169_416980

/-- Given points A(3,0) and B(-3,0), and a point P(x,y) such that the product of slopes of AP and BP is -2,
    prove that the trajectory of P satisfies the equation x²/9 + y²/18 = 1 for x ≠ ±3 -/
theorem trajectory_equation_1 (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  (y / (x - 3)) * (y / (x + 3)) = -2 → x^2 / 9 + y^2 / 18 = 1 := by
sorry

/-- The converse: if a point P(x,y) satisfies x²/9 + y²/18 = 1 for x ≠ ±3,
    then the product of slopes of AP and BP is -2 -/
theorem trajectory_equation_1_converse (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  x^2 / 9 + y^2 / 18 = 1 → (y / (x - 3)) * (y / (x + 3)) = -2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_1_trajectory_equation_1_converse_l4169_416980


namespace NUMINAMATH_CALUDE_sum_difference_inequality_l4169_416928

theorem sum_difference_inequality 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (hb : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0) 
  (ha_sum : a₁*a₁ + a₁*a₂ + a₁*a₃ + a₂*a₂ + a₂*a₃ + a₃*a₃ ≤ 1) 
  (hb_sum : b₁*b₁ + b₁*b₂ + b₁*b₃ + b₂*b₂ + b₂*b₃ + b₃*b₃ ≤ 1) : 
  (a₁-b₁)*(a₁-b₁) + (a₁-b₁)*(a₂-b₂) + (a₁-b₁)*(a₃-b₃) + 
  (a₂-b₂)*(a₂-b₂) + (a₂-b₂)*(a₃-b₃) + (a₃-b₃)*(a₃-b₃) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_inequality_l4169_416928


namespace NUMINAMATH_CALUDE_square_sum_geq_two_l4169_416918

theorem square_sum_geq_two (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_two_l4169_416918


namespace NUMINAMATH_CALUDE_max_k_value_l4169_416927

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2/y^2 + y^2/x^2) + k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l4169_416927


namespace NUMINAMATH_CALUDE_intersection_length_circle_line_l4169_416989

/-- The intersection length of a circle and a line --/
theorem intersection_length_circle_line : 
  ∃ (A B : ℝ × ℝ),
    (A.1^2 + (A.2 - 1)^2 = 1) ∧ 
    (B.1^2 + (B.2 - 1)^2 = 1) ∧
    (A.1 - A.2 + 2 = 0) ∧ 
    (B.1 - B.2 + 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_circle_line_l4169_416989


namespace NUMINAMATH_CALUDE_june_math_book_price_l4169_416908

/-- The price of a math book that satisfies June's shopping constraints -/
def math_book_price : ℝ → Prop := λ x =>
  let total_budget : ℝ := 500
  let num_math_books : ℕ := 4
  let num_science_books : ℕ := num_math_books + 6
  let science_book_price : ℝ := 10
  let num_art_books : ℕ := 2 * num_math_books
  let art_book_price : ℝ := 20
  let music_books_cost : ℝ := 160
  (num_math_books : ℝ) * x + 
  (num_science_books : ℝ) * science_book_price + 
  (num_art_books : ℝ) * art_book_price + 
  music_books_cost = total_budget

theorem june_math_book_price : ∃ x : ℝ, math_book_price x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_june_math_book_price_l4169_416908


namespace NUMINAMATH_CALUDE_books_after_donation_l4169_416975

theorem books_after_donation (boris_initial : Nat) (cameron_initial : Nat)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30) :
  boris_initial - boris_initial / 4 + cameron_initial - cameron_initial / 3 = 38 := by
  sorry

#check books_after_donation

end NUMINAMATH_CALUDE_books_after_donation_l4169_416975


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l4169_416904

/-- Given a point P(2, -5), its symmetric point P' with respect to the x-axis has coordinates (2, 5) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (2, 5)
  (∀ (x y : ℝ), (x, y) = P → (x, -y) = P') :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l4169_416904


namespace NUMINAMATH_CALUDE_exists_valid_custom_division_l4169_416940

/-- A custom division type that allows introducing additional 7s in intermediate calculations -/
structure CustomDivision where
  dividend : Nat
  divisor : Nat
  quotient : Nat
  intermediate_sevens : List Nat

/-- Checks if a number contains at least one 7 -/
def containsSeven (n : Nat) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 7

/-- Theorem stating the existence of a valid custom division -/
theorem exists_valid_custom_division :
  ∃ (cd : CustomDivision),
    cd.dividend ≥ 1000000000 ∧ cd.dividend < 10000000000 ∧
    cd.divisor ≥ 100000 ∧ cd.divisor < 1000000 ∧
    cd.quotient ≥ 10000 ∧ cd.quotient < 100000 ∧
    containsSeven cd.dividend ∧
    containsSeven cd.divisor ∧
    cd.dividend = cd.divisor * cd.quotient :=
  sorry

#check exists_valid_custom_division

end NUMINAMATH_CALUDE_exists_valid_custom_division_l4169_416940


namespace NUMINAMATH_CALUDE_ratio_q_p_l4169_416946

def total_slips : ℕ := 60
def num_range : Set ℕ := Finset.range 10
def slips_per_num : ℕ := 6
def drawn_slips : ℕ := 4

def p : ℚ := (10 : ℚ) / Nat.choose total_slips drawn_slips
def q : ℚ := (5400 : ℚ) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 540 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l4169_416946


namespace NUMINAMATH_CALUDE_A_C_mutually_exclusive_not_complementary_l4169_416903

-- Define the event space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}
def C : Set Nat := {2, 4, 6}

-- Define mutually exclusive events
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary events
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω ∧ X ∩ Y = ∅

-- Theorem to prove
theorem A_C_mutually_exclusive_not_complementary :
  mutually_exclusive A C ∧ ¬complementary A C :=
sorry

end NUMINAMATH_CALUDE_A_C_mutually_exclusive_not_complementary_l4169_416903


namespace NUMINAMATH_CALUDE_irrigation_flux_theorem_l4169_416911

-- Define the irrigation system
structure IrrigationSystem where
  channels : List Char
  entry : Char
  exit : Char
  flux : Char → Char → ℝ

-- Define the properties of the irrigation system
def has_constant_flux_sum (sys : IrrigationSystem) : Prop :=
  ∀ (p q r : Char), p ∈ sys.channels → q ∈ sys.channels → r ∈ sys.channels →
    sys.flux p q + sys.flux q r = sys.flux p r

-- Define the theorem
theorem irrigation_flux_theorem (sys : IrrigationSystem) 
  (h_channels : sys.channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
  (h_entry : sys.entry = 'A')
  (h_exit : sys.exit = 'E')
  (h_constant_flux : has_constant_flux_sum sys)
  (h_flux_bc : sys.flux 'B' 'C' = q₀) :
  sys.flux 'A' 'B' = 2 * q₀ ∧ 
  sys.flux 'A' 'H' = 3/2 * q₀ ∧ 
  sys.flux 'A' 'B' + sys.flux 'A' 'H' = 7/2 * q₀ := by
  sorry

end NUMINAMATH_CALUDE_irrigation_flux_theorem_l4169_416911


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4169_416978

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin (π / 6 + x) * Real.sin (5 * π / 6 + x) / (Real.cos x)^2 + 2 * Real.tan x = 0) ∧ (Real.cos x ≠ 0) →
  (∃ k : ℤ, x = -Real.arctan (1 / 3) + k * π) ∨ (∃ n : ℤ, x = π / 4 + n * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4169_416978


namespace NUMINAMATH_CALUDE_cube_congruence_for_prime_l4169_416924

theorem cube_congruence_for_prime (p : ℕ) (k : ℕ) 
  (hp : Nat.Prime p) (hform : p = 3 * k + 1) : 
  ∃ a b : ℕ, 0 < a ∧ a < b ∧ b < Real.sqrt p ∧ a^3 ≡ b^3 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_cube_congruence_for_prime_l4169_416924


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l4169_416948

theorem trigonometric_sum_equals_three_halves :
  let α : Real := 5 * π / 24
  let β : Real := 11 * π / 24
  (Real.cos α) ^ 4 + (Real.cos β) ^ 4 + (Real.sin (π - α)) ^ 4 + (Real.sin (π - β)) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_three_halves_l4169_416948


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4169_416912

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 2*x + y = 1 → 1/x + 1/y ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4169_416912


namespace NUMINAMATH_CALUDE_circle_center_sum_l4169_416944

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 10*x - 12*y + 40) → 
  ((x - 5)^2 + (y + 6)^2 = 101) → 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4169_416944


namespace NUMINAMATH_CALUDE_angle_equivalence_l4169_416915

theorem angle_equivalence (α θ : Real) (h1 : α = 1690) (h2 : 0 < θ ∧ θ < 360) 
  (h3 : ∃ k : Int, α = k * 360 + θ) : θ = 250 := by
  sorry

end NUMINAMATH_CALUDE_angle_equivalence_l4169_416915


namespace NUMINAMATH_CALUDE_geometric_relations_l4169_416982

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometric_relations
  (l m : Line) (α β γ : Plane)
  (h1 : intersect β γ = l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m :=
by sorry

end NUMINAMATH_CALUDE_geometric_relations_l4169_416982


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l4169_416969

theorem cookie_boxes_problem (n : ℕ) : n = 12 ↔ 
  n > 0 ∧ 
  n - 11 ≥ 1 ∧ 
  n - 2 ≥ 1 ∧ 
  (n - 11) + (n - 2) < n ∧
  ∀ m : ℕ, m > n → ¬(m > 0 ∧ m - 11 ≥ 1 ∧ m - 2 ≥ 1 ∧ (m - 11) + (m - 2) < m) :=
by sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l4169_416969


namespace NUMINAMATH_CALUDE_factor_polynomial_l4169_416984

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l4169_416984


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l4169_416938

theorem nested_fraction_evaluation : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l4169_416938


namespace NUMINAMATH_CALUDE_nurses_survey_result_l4169_416939

def total_nurses : ℕ := 150
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 50
def both_conditions : ℕ := 30

theorem nurses_survey_result : 
  (total_nurses - (high_blood_pressure + heart_trouble - both_conditions)) / total_nurses * 100 = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nurses_survey_result_l4169_416939


namespace NUMINAMATH_CALUDE_rohan_conveyance_percentage_l4169_416999

def salary : ℝ := 5000
def food_percentage : ℝ := 40
def rent_percentage : ℝ := 20
def entertainment_percentage : ℝ := 10
def savings : ℝ := 1000

theorem rohan_conveyance_percentage :
  let total_accounted_percentage := food_percentage + rent_percentage + entertainment_percentage
  let remaining_after_savings := salary - savings
  let accounted_amount := (total_accounted_percentage / 100) * salary
  let conveyance_amount := remaining_after_savings - accounted_amount
  (conveyance_amount / salary) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rohan_conveyance_percentage_l4169_416999


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_50_l4169_416921

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ
  discount : ℕ → ℕ

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people - c.discount people

/-- First caterer's pricing model -/
def caterer1 : Caterer :=
  { base_fee := 120
  , per_person := 18
  , discount := λ _ => 0 }

/-- Second caterer's pricing model -/
def caterer2 : Caterer :=
  { base_fee := 250
  , per_person := 14
  , discount := λ n => if n ≥ 50 then 50 else 0 }

/-- Theorem stating that 50 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_50 :
  (total_cost caterer2 50 < total_cost caterer1 50) ∧
  (∀ n : ℕ, n < 50 → total_cost caterer1 n ≤ total_cost caterer2 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_50_l4169_416921


namespace NUMINAMATH_CALUDE_centroid_vector_sum_l4169_416923

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC with centroid G and an arbitrary point P,
    prove that PG = 1/3(PA + PB + PC) -/
theorem centroid_vector_sum (A B C P G : V) 
  (h : G = (1/3 : ℝ) • (A + B + C)) : 
  G - P = (1/3 : ℝ) • ((A - P) + (B - P) + (C - P)) := by
  sorry

end NUMINAMATH_CALUDE_centroid_vector_sum_l4169_416923


namespace NUMINAMATH_CALUDE_problem_statement_l4169_416981

theorem problem_statement :
  (∀ x : ℝ, 1 + 2 * x^4 ≥ 2 * x^3 + x^2) ∧
  (∀ x y z : ℝ, x + 2*y + 3*z = 6 →
    x^2 + y^2 + z^2 ≥ 18/7 ∧
    ∃ x y z : ℝ, x + 2*y + 3*z = 6 ∧ x^2 + y^2 + z^2 = 18/7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4169_416981


namespace NUMINAMATH_CALUDE_gcd_sum_characterization_l4169_416993

theorem gcd_sum_characterization (M : ℝ) (h_M : M ≥ 1) :
  ∀ n : ℕ, (∃ a b c : ℕ, a > M ∧ b > M ∧ c > M ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = Nat.gcd a b * Nat.gcd b c + Nat.gcd b c * Nat.gcd c a + Nat.gcd c a * Nat.gcd a b) ↔
  (Even (Nat.log 2 n) ∧ ¬∃ k : ℕ, n = 4^k) :=
by sorry

end NUMINAMATH_CALUDE_gcd_sum_characterization_l4169_416993


namespace NUMINAMATH_CALUDE_point_on_line_l4169_416990

/-- Given two points A and B in the Cartesian plane, if a point C satisfies the vector equation
    OC = s*OA + t*OB where s + t = 1, then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (s t : ℝ) :
  A = (2, 1) →
  B = (-1, -2) →
  C = s • A + t • B →
  s + t = 1 →
  C.1 - C.2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l4169_416990


namespace NUMINAMATH_CALUDE_rectangular_room_length_l4169_416935

theorem rectangular_room_length (area width : ℝ) (h1 : area = 215.6) (h2 : width = 14) :
  area / width = 15.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_room_length_l4169_416935


namespace NUMINAMATH_CALUDE_ball_prices_theorem_l4169_416973

/-- Represents the prices and quantities of soccer balls and volleyballs -/
structure BallPrices where
  soccer_price : ℝ
  volleyball_price : ℝ
  total_balls : ℕ
  max_cost : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (bp : BallPrices) : Prop :=
  bp.soccer_price = bp.volleyball_price + 15 ∧
  480 / bp.soccer_price = 390 / bp.volleyball_price ∧
  bp.total_balls = 100

/-- The theorem to be proven -/
theorem ball_prices_theorem (bp : BallPrices) 
  (h : satisfies_conditions bp) : 
  bp.soccer_price = 80 ∧ 
  bp.volleyball_price = 65 ∧ 
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             m * bp.soccer_price + (bp.total_balls - m) * bp.volleyball_price ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               n * bp.soccer_price + (bp.total_balls - n) * bp.volleyball_price > bp.max_cost :=
by
  sorry

end NUMINAMATH_CALUDE_ball_prices_theorem_l4169_416973


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l4169_416907

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.tan θ = 1/6) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l4169_416907


namespace NUMINAMATH_CALUDE_rectangle_area_l4169_416997

/-- The area of a rectangle with width 81/4 cm and height 148/9 cm is 333 cm². -/
theorem rectangle_area : 
  let width : ℚ := 81 / 4
  let height : ℚ := 148 / 9
  (width * height : ℚ) = 333 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l4169_416997


namespace NUMINAMATH_CALUDE_puzzle_sum_l4169_416937

theorem puzzle_sum (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 59) 
  (h4 : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  a + b + c = 69 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l4169_416937


namespace NUMINAMATH_CALUDE_hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l4169_416992

/-- The sum of the exterior angles of a hexagon is 360 degrees. -/
theorem hexagon_exterior_angle_sum : ℝ :=
  360

#check hexagon_exterior_angle_sum

/-- Proof of the theorem -/
theorem hexagon_exterior_angle_sum_proof :
  hexagon_exterior_angle_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l4169_416992


namespace NUMINAMATH_CALUDE_theater_attendance_l4169_416996

/-- Proves the number of children attending a theater given total attendance and revenue --/
theorem theater_attendance (adults children : ℕ) 
  (total_attendance : adults + children = 280)
  (total_revenue : 60 * adults + 25 * children = 14000) :
  children = 80 := by sorry

end NUMINAMATH_CALUDE_theater_attendance_l4169_416996


namespace NUMINAMATH_CALUDE_range_of_a_l4169_416951

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - m*x - 2 = 0

-- Define the inequality condition for a
def inequality_condition (a m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  a^2 - 5*a - 3 ≥ |x₁ - x₂|

-- Define the range for m
def m_range (m : ℝ) : Prop := -1 ≤ m ∧ m ≤ 1

-- Define the condition for the quadratic inequality having no solutions
def no_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 + 2*x - 1 ≤ 0

theorem range_of_a :
  ∀ m : ℝ, m_range m →
  ∀ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ ∧ x₁ ≠ x₂ →
  ∀ a : ℝ, (∀ m : ℝ, m_range m → inequality_condition a m x₁ x₂) ∧ no_solutions a →
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4169_416951


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4169_416913

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 9 ∧ (427398 - x) % 10 = 0 ∧ 
  ∀ (y : ℕ), y < x → (427398 - y) % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4169_416913


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l4169_416968

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ 
  (x : ℤ) ≡ 7 [ZMOD 9] ∧
  ∀ y : ℕ+, ((y : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ (y : ℤ) ≡ 7 [ZMOD 9]) → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l4169_416968


namespace NUMINAMATH_CALUDE_practice_paper_percentage_l4169_416952

theorem practice_paper_percentage (total_students : ℕ) 
  (passed_all : ℝ) (passed_none : ℝ) (passed_one : ℝ) (passed_four : ℝ) (passed_three : ℕ)
  (h1 : total_students = 2500)
  (h2 : passed_all = 0.1)
  (h3 : passed_none = 0.1)
  (h4 : passed_one = 0.2 * (1 - passed_all - passed_none))
  (h5 : passed_four = 0.24)
  (h6 : passed_three = 500) :
  let remaining := 1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students
  let passed_two := (1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students) * remaining
  ∃ (ε : ℝ), abs (passed_two - 0.5002) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_practice_paper_percentage_l4169_416952


namespace NUMINAMATH_CALUDE_roots_sum_minus_product_l4169_416985

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → 
  x₂^2 - 2*x₂ - 1 = 0 → 
  x₁ + x₂ - x₁*x₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_minus_product_l4169_416985


namespace NUMINAMATH_CALUDE_triangles_from_points_l4169_416910

/-- Represents a triangular paper with points -/
structure TriangularPaper where
  n : ℕ  -- number of points inside the triangle

/-- Condition that no three points are collinear -/
axiom not_collinear (paper : TriangularPaper) : True

/-- Function to calculate the number of smaller triangles -/
def num_triangles (paper : TriangularPaper) : ℕ :=
  2 * paper.n + 1

/-- Theorem stating the relationship between points and triangles -/
theorem triangles_from_points (paper : TriangularPaper) :
  num_triangles paper = 2 * paper.n + 1 :=
sorry

end NUMINAMATH_CALUDE_triangles_from_points_l4169_416910


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4169_416949

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^3 * a^6 = a^9 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4169_416949


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l4169_416906

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 11x + 2 -/
def quadratic_equation (x : ℝ) : ℝ := 5*x^2 - 11*x + 2

theorem discriminant_of_specific_quadratic :
  discriminant 5 (-11) 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l4169_416906


namespace NUMINAMATH_CALUDE_min_discount_rate_l4169_416965

/-- The minimum discount rate for a product with given cost and marked prices, ensuring a minimum profit percentage. -/
theorem min_discount_rate (cost : ℝ) (marked : ℝ) (min_profit_percent : ℝ) :
  cost = 1000 →
  marked = 1500 →
  min_profit_percent = 5 →
  ∃ x : ℝ, x = 0.7 ∧
    ∀ y : ℝ, (marked * y - cost ≥ cost * (min_profit_percent / 100) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_min_discount_rate_l4169_416965


namespace NUMINAMATH_CALUDE_problem_sum_value_l4169_416900

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series (3/4)^k from k=1 to 12 -/
def problem_sum : ℚ := geometric_sum (3/4) (3/4) 12

theorem problem_sum_value : problem_sum = 48738225 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_problem_sum_value_l4169_416900


namespace NUMINAMATH_CALUDE_ellipse_parallelogram_area_l4169_416988

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 1

-- Define the slope product condition
def slope_product (x1 y1 x2 y2 : ℝ) : Prop := (y1/x1) * (y2/x2) = -1/2

-- Define the area of the parallelogram
def parallelogram_area (x1 y1 x2 y2 : ℝ) : ℝ := 2 * |x1*y2 - x2*y1|

-- Theorem statement
theorem ellipse_parallelogram_area 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : on_ellipse x1 y1) 
  (h2 : on_ellipse x2 y2) 
  (h3 : slope_product x1 y1 x2 y2) : 
  parallelogram_area x1 y1 x2 y2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parallelogram_area_l4169_416988


namespace NUMINAMATH_CALUDE_sine_equation_solution_l4169_416934

theorem sine_equation_solution (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y = 0 →
  ∃ k n : ℤ, x = k * Real.pi ∧ y = n * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sine_equation_solution_l4169_416934


namespace NUMINAMATH_CALUDE_inverse_proportion_difference_positive_l4169_416967

theorem inverse_proportion_difference_positive 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ - y₂ > 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_difference_positive_l4169_416967


namespace NUMINAMATH_CALUDE_integer_fraction_values_l4169_416979

theorem integer_fraction_values (k : ℤ) : 
  (∃ n : ℤ, (2 * k^2 + k - 8) / (k - 1) = n) ↔ k ∈ ({6, 2, 0, -4} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_values_l4169_416979


namespace NUMINAMATH_CALUDE_angle_D_measure_l4169_416917

-- Define the angles in degrees
def angle_A : ℝ := 50
def angle_B : ℝ := 35
def angle_C : ℝ := 40

-- Define the configuration
structure TriangleConfiguration where
  -- Triangle 1
  internal_angle_A : ℝ
  internal_angle_B : ℝ
  -- External triangle
  external_angle_C : ℝ
  -- Constraints
  angle_A_eq : internal_angle_A = angle_A
  angle_B_eq : internal_angle_B = angle_B
  angle_C_eq : external_angle_C = angle_C

-- Theorem statement
theorem angle_D_measure (config : TriangleConfiguration) :
  ∃ (angle_D : ℝ), angle_D = 125 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l4169_416917


namespace NUMINAMATH_CALUDE_village_population_l4169_416929

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  partial_population = 36000 → percentage = 9/10 → 
  (percentage * total_population : ℚ) = partial_population → 
  total_population = 40000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l4169_416929


namespace NUMINAMATH_CALUDE_cube_volume_problem_l4169_416925

theorem cube_volume_problem (a : ℝ) : 
  (a - 2) * a * (a + 2) = a^3 - 8 → a^3 = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l4169_416925


namespace NUMINAMATH_CALUDE_johnny_runs_four_times_l4169_416901

theorem johnny_runs_four_times (block_length : ℝ) (average_distance : ℝ) :
  block_length = 200 →
  average_distance = 600 →
  ∃ (johnny_runs : ℕ),
    (average_distance = (block_length * johnny_runs + block_length * (johnny_runs / 2)) / 2) ∧
    johnny_runs = 4 :=
by sorry

end NUMINAMATH_CALUDE_johnny_runs_four_times_l4169_416901


namespace NUMINAMATH_CALUDE_extreme_value_condition_l4169_416971

/-- A function f: ℝ → ℝ has an extreme value at x₀ -/
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The statement that f'(x₀) = 0 is a necessary but not sufficient condition
    for f to have an extreme value at x₀ -/
theorem extreme_value_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, has_extreme_value f x₀ → (deriv f) x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, (deriv f) x₀ = 0 → has_extreme_value f x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l4169_416971


namespace NUMINAMATH_CALUDE_linear_function_decreasing_iff_k_lt_neg_two_l4169_416966

/-- A linear function y = mx + b where m = k + 2 and b = -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 2) * x - 1

/-- The property that y decreases as x increases -/
def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_function_decreasing_iff_k_lt_neg_two (k : ℝ) :
  decreasing_function (linear_function k) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_iff_k_lt_neg_two_l4169_416966


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4169_416962

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * (b^(1/2)) = k) →  -- The cube of a and square root of b vary inversely
  (3^3 * (64^(1/2)) = k) →        -- a = 3 when b = 64
  (a * b = 36) →                  -- Given condition ab = 36
  (b = 6) :=                      -- Prove that b = 6
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4169_416962


namespace NUMINAMATH_CALUDE_coloring_satisfies_conditions_l4169_416926

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function
def color (x y : Int) : Color :=
  if (x + y) % 2 = 0 then Color.Red
  else if x % 2 = 1 && y % 2 = 0 then Color.White
  else Color.Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define a property that a color appears infinitely many times on infinitely many horizontal lines
def infiniteOccurrence (c : Color) : Prop :=
  ∀ (n : Nat), ∃ (m : Int), ∀ (k : Int), ∃ (x : Int), 
    color x (m + k * n) = c

-- Define the parallelogram property
def parallelogramProperty : Prop :=
  ∀ (A B C : LatticePoint),
    color A.x A.y = Color.White →
    color B.x B.y = Color.Red →
    color C.x C.y = Color.Black →
    ∃ (D : LatticePoint),
      color D.x D.y = Color.Red ∧
      D.x - C.x = A.x - B.x ∧
      D.y - C.y = A.y - B.y

-- The main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, infiniteOccurrence c) ∧ parallelogramProperty :=
sorry

end NUMINAMATH_CALUDE_coloring_satisfies_conditions_l4169_416926


namespace NUMINAMATH_CALUDE_number_solution_l4169_416954

theorem number_solution : ∃ x : ℝ, (5020 - (502 / x) = 5015) ∧ x = 100.4 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l4169_416954


namespace NUMINAMATH_CALUDE_european_scientist_ratio_l4169_416941

theorem european_scientist_ratio (total : ℕ) (usa : ℕ) (canada : ℚ) : 
  total = 70 →
  usa = 21 →
  canada = 1/5 →
  (total - (canada * total).num - usa) / total = 1/2 := by
sorry

end NUMINAMATH_CALUDE_european_scientist_ratio_l4169_416941


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l4169_416959

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 21 * x * y = 7 - 3 * x - 4 * y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l4169_416959


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l4169_416945

/-- The volume of a sphere inscribed in a cube -/
theorem volume_of_inscribed_sphere (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 10 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (500 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l4169_416945


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4169_416995

theorem trigonometric_identity (θ φ : Real) 
  (h : (Real.sin θ)^4 / (Real.sin φ)^2 + (Real.cos θ)^4 / (Real.cos φ)^2 = 1) :
  (Real.cos φ)^4 / (Real.cos θ)^2 + (Real.sin φ)^4 / (Real.sin θ)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4169_416995


namespace NUMINAMATH_CALUDE_pamphlet_printing_speed_ratio_l4169_416936

theorem pamphlet_printing_speed_ratio : 
  ∀ (mike_speed : ℕ) (mike_hours_before_break : ℕ) (mike_hours_after_break : ℕ) 
    (leo_speed_multiplier : ℕ) (total_pamphlets : ℕ),
  mike_speed = 600 →
  mike_hours_before_break = 9 →
  mike_hours_after_break = 2 →
  total_pamphlets = 9400 →
  (mike_speed * mike_hours_before_break + 
   (mike_speed / 3) * mike_hours_after_break + 
   (leo_speed_multiplier * mike_speed) * (mike_hours_before_break / 3) = total_pamphlets) →
  leo_speed_multiplier = 2 := by
sorry

end NUMINAMATH_CALUDE_pamphlet_printing_speed_ratio_l4169_416936


namespace NUMINAMATH_CALUDE_overall_mean_score_l4169_416905

/-- Given the mean scores and ratios of students in three classes, prove the overall mean score --/
theorem overall_mean_score (m a e : ℕ) (M A E : ℝ) : 
  M = 78 → A = 68 → E = 82 →
  (m : ℝ) / a = 4 / 5 →
  ((m : ℝ) + a) / e = 9 / 2 →
  (M * m + A * a + E * e) / (m + a + e : ℝ) = 74.4 := by
  sorry

#check overall_mean_score

end NUMINAMATH_CALUDE_overall_mean_score_l4169_416905


namespace NUMINAMATH_CALUDE_emily_score_proof_l4169_416909

def emily_scores : List ℝ := [88, 92, 85, 90, 97]
def target_mean : ℝ := 91
def sixth_score : ℝ := 94

theorem emily_score_proof :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
  sorry

end NUMINAMATH_CALUDE_emily_score_proof_l4169_416909


namespace NUMINAMATH_CALUDE_b_nonnegative_l4169_416991

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem b_nonnegative 
  (a b c m₁ m₂ : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : f a b c 1 = 0) 
  (h4 : a^2 + (f a b c m₁ + f a b c m₂) * a + f a b c m₁ * f a b c m₂ = 0) :
  b ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_b_nonnegative_l4169_416991


namespace NUMINAMATH_CALUDE_louis_suit_cost_is_141_l4169_416920

/-- The cost of Louis's velvet suit materials -/
def louis_suit_cost (fabric_price_per_yard : ℝ) (pattern_price : ℝ) (thread_price_per_spool : ℝ) 
  (fabric_yards : ℝ) (thread_spools : ℕ) : ℝ :=
  fabric_price_per_yard * fabric_yards + pattern_price + thread_price_per_spool * thread_spools

/-- Theorem: The total cost of Louis's suit materials is $141 -/
theorem louis_suit_cost_is_141 : 
  louis_suit_cost 24 15 3 5 2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_louis_suit_cost_is_141_l4169_416920


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l4169_416957

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  ∃ x : ℝ, (x^2 + (a - b)*x + (b - c) = 0) ∨ 
            (x^2 + (b - c)*x + (c - a) = 0) ∨ 
            (x^2 + (c - a)*x + (a - b) = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l4169_416957


namespace NUMINAMATH_CALUDE_class_average_score_class_average_is_85_l4169_416919

theorem class_average_score : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_students, students_score_92, students_score_80, students_score_70, score_70 =>
    let total_score := students_score_92 * 92 + students_score_80 * 80 + students_score_70 * score_70
    total_score / total_students

theorem class_average_is_85 :
  class_average_score 10 5 4 1 70 = 85 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_class_average_is_85_l4169_416919


namespace NUMINAMATH_CALUDE_total_profit_is_36000_l4169_416960

/-- Represents the profit sharing problem of Tom and Jose's shop -/
def ProfitSharing (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : Prop :=
  let tom_total_investment := tom_investment * tom_months
  let jose_total_investment := jose_investment * jose_months
  let total_investment := tom_total_investment + jose_total_investment
  let profit_ratio := tom_total_investment / jose_total_investment
  let tom_profit := (profit_ratio * jose_profit) / (profit_ratio + 1)
  let total_profit := tom_profit + jose_profit
  total_profit = 36000

/-- The main theorem stating that given the investments and Jose's profit, the total profit is 36000 -/
theorem total_profit_is_36000 :
  ProfitSharing 30000 12 45000 10 20000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_36000_l4169_416960


namespace NUMINAMATH_CALUDE_complex_multiplication_l4169_416986

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4169_416986


namespace NUMINAMATH_CALUDE_viewer_ratio_l4169_416956

def voltaire_daily_viewers : ℕ := 50
def earnings_per_view : ℚ := 1/2
def leila_weekly_earnings : ℕ := 350
def days_per_week : ℕ := 7

theorem viewer_ratio : 
  let voltaire_weekly_viewers := voltaire_daily_viewers * days_per_week
  let leila_weekly_viewers := (leila_weekly_earnings : ℚ) / earnings_per_view
  (leila_weekly_viewers : ℚ) / (voltaire_weekly_viewers : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_viewer_ratio_l4169_416956


namespace NUMINAMATH_CALUDE_range_of_x_l4169_416998

def p (x : ℝ) := Real.log (x^2 - 2*x - 2) ≥ 0

def q (x : ℝ) := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hp : p x) (hq : ¬q x) : x ≥ 4 ∨ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l4169_416998


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4169_416963

/-- Represents a cube with holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculate the total surface area of a cube with holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let exposed_internal_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the specific cube with holes -/
theorem cube_with_holes_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4169_416963


namespace NUMINAMATH_CALUDE_sets_operations_l4169_416958

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | Real.exp (x - 1) ≥ 1}

-- Define the theorem
theorem sets_operations :
  (A ∪ B = {x | x > -3}) ∧
  ((Aᶜ) ∩ B = {x | x ≥ 2}) := by
  sorry

end NUMINAMATH_CALUDE_sets_operations_l4169_416958


namespace NUMINAMATH_CALUDE_library_books_count_l4169_416983

/-- Given a library with identical bookcases, prove the total number of books -/
theorem library_books_count (num_bookcases : ℕ) (shelves_per_bookcase : ℕ) (books_per_shelf : ℕ) :
  num_bookcases = 28 →
  shelves_per_bookcase = 6 →
  books_per_shelf = 19 →
  num_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l4169_416983


namespace NUMINAMATH_CALUDE_angle_sum_equality_l4169_416994

theorem angle_sum_equality (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧ 
  0 < β ∧ β < Real.pi/2 ∧ 
  Real.cos α = 7/Real.sqrt 50 ∧ 
  Real.tan β = 1/3 → 
  α + 2*β = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l4169_416994


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_equals_square_l4169_416964

theorem power_of_two_plus_one_equals_square (m n : ℕ) : 2^n + 1 = m^2 ↔ m = 3 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_equals_square_l4169_416964


namespace NUMINAMATH_CALUDE_special_triangle_solution_l4169_416931

/-- Represents a triangle with given properties -/
structure SpecialTriangle where
  a : ℝ
  r : ℝ
  ρ : ℝ
  h_a : a = 6
  h_r : r = 5
  h_ρ : ρ = 2

/-- The other two sides and area of the special triangle -/
def TriangleSolution (t : SpecialTriangle) : ℝ × ℝ × ℝ :=
  (8, 10, 24)

theorem special_triangle_solution (t : SpecialTriangle) :
  let (b, c, area) := TriangleSolution t
  b * c = 10 * area / 3 ∧
  b + c = area - t.a ∧
  area = t.ρ * (t.a + b + c) / 2 ∧
  area^2 = (t.a + b + c) / 2 * ((t.a + b + c) / 2 - t.a) * ((t.a + b + c) / 2 - b) * ((t.a + b + c) / 2 - c) ∧
  t.r = t.a * b * c / (4 * area) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_solution_l4169_416931


namespace NUMINAMATH_CALUDE_unique_odd_number_with_congruences_l4169_416942

theorem unique_odd_number_with_congruences : ∃! x : ℕ,
  500 < x ∧ x < 1000 ∧
  x % 25 = 6 ∧
  x % 9 = 7 ∧
  Odd x ∧
  x = 781 := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_congruences_l4169_416942
