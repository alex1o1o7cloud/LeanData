import Mathlib

namespace NUMINAMATH_CALUDE_marys_age_l2221_222152

theorem marys_age (mary_age rahul_age : ℕ) : 
  rahul_age = mary_age + 30 →
  rahul_age + 20 = 2 * (mary_age + 20) →
  mary_age = 10 := by
sorry

end NUMINAMATH_CALUDE_marys_age_l2221_222152


namespace NUMINAMATH_CALUDE_coat_price_calculation_l2221_222117

def calculate_final_price (initial_price : ℝ) (initial_tax_rate : ℝ) 
                          (discount_rate : ℝ) (additional_discount : ℝ) 
                          (final_tax_rate : ℝ) : ℝ :=
  let price_after_initial_tax := initial_price * (1 + initial_tax_rate)
  let price_after_discount := price_after_initial_tax * (1 - discount_rate)
  let price_after_additional_discount := price_after_discount - additional_discount
  price_after_additional_discount * (1 + final_tax_rate)

theorem coat_price_calculation :
  calculate_final_price 200 0.10 0.25 10 0.05 = 162.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l2221_222117


namespace NUMINAMATH_CALUDE_square_equation_solution_l2221_222179

theorem square_equation_solution :
  ∃! x : ℤ, (2020 + x)^2 = x^2 :=
by
  use -1010
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2221_222179


namespace NUMINAMATH_CALUDE_inequality_proof_l2221_222165

def f (a x : ℝ) : ℝ := |x - a| + 1

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a := 1 / m + 1 / n
  (∀ x, f a x ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2) →
  m + 2 * n ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2221_222165


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l2221_222111

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids / 2 = total_kids / 2) -- Half of kids go to soccer camp
  (h3 : (total_kids / 2) / 4 = (total_kids / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  : total_kids / 2 - (total_kids / 2) / 4 = 750 := by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l2221_222111


namespace NUMINAMATH_CALUDE_sqrt_19992000_floor_l2221_222164

theorem sqrt_19992000_floor : ⌊Real.sqrt 19992000⌋ = 4471 := by sorry

end NUMINAMATH_CALUDE_sqrt_19992000_floor_l2221_222164


namespace NUMINAMATH_CALUDE_root_ratio_sum_l2221_222116

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, k₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₁ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  (∃ a b : ℝ, k₂ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  k₁/k₂ + k₂/k₁ = 64/9 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_l2221_222116


namespace NUMINAMATH_CALUDE_tangent_points_and_circle_area_l2221_222147

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent points M and N
def M (x₁ : ℝ) : ℝ × ℝ := (x₁, parabola x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, parabola x₂)

-- State the theorem
theorem tangent_points_and_circle_area 
  (x₁ x₂ : ℝ) 
  (h_tangent : ∃ (k b : ℝ), ∀ x, k * x + b = parabola x → x = x₁ ∨ x = x₂)
  (h_order : x₁ < x₂) :
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x y : ℝ), (x - P.1)^2 + (y - P.2)^2 = r^2 ∧
      ∃ (k b : ℝ), k * x + b = y ∧ k * x₁ + b = parabola x₁ ∧ k * x₂ + b = parabola x₂) ∧
    π * r^2 = 16 * π / 5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_and_circle_area_l2221_222147


namespace NUMINAMATH_CALUDE_cheap_gym_cost_l2221_222127

/-- Represents the monthly cost of gym memberships and related calculations -/
def gym_costs (cheap_monthly : ℝ) : Prop :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_signup := 50
  let expensive_signup := 4 * expensive_monthly
  let cheap_yearly := cheap_signup + 12 * cheap_monthly
  let expensive_yearly := expensive_signup + 12 * expensive_monthly
  cheap_yearly + expensive_yearly = 650

theorem cheap_gym_cost : ∃ (cheap_monthly : ℝ), gym_costs cheap_monthly ∧ cheap_monthly = 10 := by
  sorry

end NUMINAMATH_CALUDE_cheap_gym_cost_l2221_222127


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2221_222132

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := V → Prop
def Plane (V : Type*) [NormedAddCommGroup V] := V → Prop

-- Define perpendicular relation between a line and a plane
def Perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallel relation between two lines
def Parallel (l1 l2 : Line V) : Prop := sorry

-- Theorem statement
theorem perpendicular_lines_parallel 
  (m n : Line V) (α : Plane V) 
  (hm : m ≠ n) 
  (h1 : Perpendicular m α) 
  (h2 : Perpendicular n α) : 
  Parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2221_222132


namespace NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l2221_222174

-- Define the property of being in the fourth quadrant
def is_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + 3 * Real.pi / 2 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi

-- Define the property of being in the second quadrant
def is_second_quadrant (θ : Real) : Prop :=
  ∃ k : Int, k * Real.pi + Real.pi / 2 ≤ θ ∧ θ ≤ k * Real.pi + Real.pi

-- State the theorem
theorem half_angle_in_second_quadrant (θ : Real) 
  (h1 : is_fourth_quadrant θ) 
  (h2 : |Real.cos (θ/2)| = -Real.cos (θ/2)) : 
  is_second_quadrant (θ/2) :=
sorry

end NUMINAMATH_CALUDE_half_angle_in_second_quadrant_l2221_222174


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2221_222196

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = -x -/
def symmetryLine (p : Point) : Prop := p.y = -p.x

/-- Definition of symmetry with respect to y = -x -/
def isSymmetric (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

theorem symmetry_of_point :
  let p1 : Point := ⟨1, 4⟩
  let p2 : Point := ⟨-4, -1⟩
  isSymmetric p1 p2 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2221_222196


namespace NUMINAMATH_CALUDE_equation_is_linear_one_variable_l2221_222160

/-- Represents a polynomial equation --/
structure PolynomialEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- Checks if a polynomial equation is linear with one variable --/
def is_linear_one_variable (eq : PolynomialEquation) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, eq.lhs x = a * x + b ∧ eq.rhs x = 0

/-- The equation y + 3 = 0 --/
def equation : PolynomialEquation :=
  { lhs := λ y => y + 3
    rhs := λ _ => 0 }

/-- Theorem stating that the equation y + 3 = 0 is a linear equation with one variable --/
theorem equation_is_linear_one_variable : is_linear_one_variable equation := by
  sorry

#check equation_is_linear_one_variable

end NUMINAMATH_CALUDE_equation_is_linear_one_variable_l2221_222160


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l2221_222131

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℝ)
  (group1 : ℕ)
  (avg1 : ℝ)
  (group2 : ℕ)
  (avg2 : ℝ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_group1 : group1 = 2)
  (h_avg1 : avg1 = 3.4)
  (h_group2 : group2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let remaining := total - group1 - group2
  let sum_all := total * avg_all
  let sum1 := group1 * avg1
  let sum2 := group2 * avg2
  let sum_remaining := sum_all - sum1 - sum2
  sum_remaining / remaining = 4.6 := by sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l2221_222131


namespace NUMINAMATH_CALUDE_distance_from_two_equals_three_l2221_222176

theorem distance_from_two_equals_three (x : ℝ) : 
  |x - 2| = 3 ↔ x = 5 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_distance_from_two_equals_three_l2221_222176


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l2221_222140

/-- A bag containing red and black balls -/
structure Bag where
  red : Nat
  black : Nat

/-- The outcome of drawing two balls -/
inductive DrawResult
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Event representing exactly one black ball drawn -/
def ExactlyOneBlack (result : DrawResult) : Prop :=
  result = DrawResult.OneRedOneBlack

/-- Event representing exactly two black balls drawn -/
def ExactlyTwoBlack (result : DrawResult) : Prop :=
  result = DrawResult.TwoBlack

/-- The sample space of all possible outcomes -/
def SampleSpace (bag : Bag) : Set DrawResult :=
  {DrawResult.TwoRed, DrawResult.OneRedOneBlack, DrawResult.TwoBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (E₁ E₂ : Set DrawResult) : Prop :=
  E₁ ∩ E₂ = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (E₁ E₂ : Set DrawResult) (S : Set DrawResult) : Prop :=
  E₁ ∪ E₂ = S

/-- Main theorem: ExactlyOneBlack and ExactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactly_one_two_black_mutually_exclusive_not_complementary (bag : Bag) :
  let S := SampleSpace bag
  let E₁ := {r : DrawResult | ExactlyOneBlack r}
  let E₂ := {r : DrawResult | ExactlyTwoBlack r}
  MutuallyExclusive E₁ E₂ ∧ ¬Complementary E₁ E₂ S :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l2221_222140


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2221_222173

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2221_222173


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2221_222151

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2221_222151


namespace NUMINAMATH_CALUDE_complex_modulus_l2221_222106

theorem complex_modulus (r : ℝ) (z : ℂ) (hr : |r| < 1) (hz : z - 1/z = r) :
  Complex.abs z = Real.sqrt (1 + r^2/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2221_222106


namespace NUMINAMATH_CALUDE_coat_price_l2221_222118

/-- The original price of a coat given a specific price reduction and percentage decrease. -/
theorem coat_price (price_reduction : ℝ) (percent_decrease : ℝ) (original_price : ℝ) : 
  price_reduction = 300 ∧ 
  percent_decrease = 0.60 ∧ 
  price_reduction = percent_decrease * original_price → 
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_coat_price_l2221_222118


namespace NUMINAMATH_CALUDE_total_arrangements_l2221_222155

def news_reports : ℕ := 5
def interviews : ℕ := 4
def total_programs : ℕ := 5
def min_news : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

def arrangements (news interviews total min_news : ℕ) : ℕ :=
  (choose news min_news * choose interviews (total - min_news) * permute total total) +
  (choose news (min_news + 1) * choose interviews (total - min_news - 1) * permute total total) +
  (choose news total * permute total total)

theorem total_arrangements :
  arrangements news_reports interviews total_programs min_news = 9720 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l2221_222155


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_square_sum_l2221_222141

theorem finite_solutions_factorial_square_sum (a : ℕ) :
  ∃ (n : ℕ), ∀ (x y : ℕ), x! = y^2 + a^2 → x ≤ n :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_square_sum_l2221_222141


namespace NUMINAMATH_CALUDE_sqrt_3_power_calculation_l2221_222184

theorem sqrt_3_power_calculation : 
  (Real.sqrt ((Real.sqrt 3) ^ 5)) ^ 6 = 2187 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_power_calculation_l2221_222184


namespace NUMINAMATH_CALUDE_root_equation_c_value_l2221_222108

theorem root_equation_c_value (c : ℝ) : 
  (1 : ℝ)^2 - 3*(1 : ℝ) + c = 0 → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_c_value_l2221_222108


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2221_222181

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2221_222181


namespace NUMINAMATH_CALUDE_frobenius_coin_problem_l2221_222130

/-- Two natural numbers are coprime -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set M of integers that can be expressed as ax + by for non-negative x and y -/
def M (a b : ℕ) : Set ℤ := {z : ℤ | ∃ x y : ℕ, z = a * x + b * y}

/-- The greatest integer not in M -/
def c (a b : ℕ) : ℤ := a * b - a - b

theorem frobenius_coin_problem (a b : ℕ) (h : Coprime a b) :
  (∀ z : ℤ, z > c a b → z ∈ M a b) ∧
  (c a b ∉ M a b) ∧
  (∀ n : ℤ, (n ∈ M a b ∧ (c a b - n) ∉ M a b) ∨ (n ∉ M a b ∧ (c a b - n) ∈ M a b)) :=
sorry

end NUMINAMATH_CALUDE_frobenius_coin_problem_l2221_222130


namespace NUMINAMATH_CALUDE_inequality_theorem_l2221_222114

theorem inequality_theorem (a b c m : ℝ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c : ℝ, a > b → b > c → (1 / (a - b) + 1 / (b - c) ≥ m / (a - c))) : 
  m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2221_222114


namespace NUMINAMATH_CALUDE_all_propositions_false_l2221_222188

theorem all_propositions_false : ∃ a b : ℝ,
  (a > b ∧ a^2 ≤ b^2) ∧
  (a^2 > b^2 ∧ a ≤ b) ∧
  (a > b ∧ b/a ≥ 1) ∧
  (a > b ∧ 1/a ≥ 1/b) := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2221_222188


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l2221_222150

theorem complex_product_real_imag_parts : ∃ (a b : ℝ), 
  (Complex.mk a b = (2 * Complex.I - 1) / Complex.I) ∧ (a * b = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l2221_222150


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_l2221_222123

/-- Represents an investment partner -/
structure Partner where
  investment : ℝ
  time : ℝ

/-- Theorem stating the relationship between profit ratio and investment ratio -/
theorem investment_ratio_from_profit_ratio
  (p q : Partner)
  (profit_ratio : ℝ × ℝ)
  (hp : p.time = 5)
  (hq : q.time = 12)
  (hprofit : profit_ratio = (7, 12)) :
  p.investment / q.investment = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_l2221_222123


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2221_222126

theorem quadratic_coefficient (a b c : ℝ) (h1 : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let Δ := b^2 - 4*a*c
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ (x - y)^2 = 1) →
  Δ = 1/4 →
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2221_222126


namespace NUMINAMATH_CALUDE_programmer_debug_time_l2221_222137

/-- Proves that given a 48-hour work week, where 1/4 of the time is spent on flow charts
    and 3/8 on coding, the remaining time spent on debugging is 18 hours. -/
theorem programmer_debug_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ) :
  total_hours = 48 →
  flow_chart_fraction = 1/4 →
  coding_fraction = 3/8 →
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end NUMINAMATH_CALUDE_programmer_debug_time_l2221_222137


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l2221_222161

theorem polynomial_division_proof (x : ℝ) :
  6 * x^3 + 12 * x^2 - 5 * x + 3 = 
  (3 * x + 4) * (2 * x^2 + (4/3) * x - 31/9) + 235/9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l2221_222161


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2221_222178

theorem complex_modulus_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2221_222178


namespace NUMINAMATH_CALUDE_max_electric_field_strength_l2221_222191

/-- The maximum electric field strength for two equal charges -/
theorem max_electric_field_strength 
  (Q : ℝ) -- charge
  (d : ℝ) -- half distance between charges
  (k : ℝ) -- Coulomb constant
  (h1 : Q > 0)
  (h2 : d > 0)
  (h3 : k > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
    ∀ (x : ℝ), x > 0 → 
      (2 * k * Q * x / (x^2 + d^2)^(3/2)) ≤ (4 * Real.sqrt 3 / 9) * (k * Q / d^2) :=
sorry

end NUMINAMATH_CALUDE_max_electric_field_strength_l2221_222191


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2221_222192

/-- The value of n for which two infinite geometric series satisfy given conditions -/
theorem geometric_series_relation (a₁ r₁ a₂ r₂ n : ℝ) : 
  a₁ = 12 →
  a₁ * r₁ = 3 →
  a₂ = 12 →
  a₂ * r₂ = 3 + n →
  (a₂ / (1 - r₂)) = 3 * (a₁ / (1 - r₁)) →
  n = 6 := by
sorry


end NUMINAMATH_CALUDE_geometric_series_relation_l2221_222192


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2221_222195

def chicken_soup_quantity : ℕ := 6
def chicken_soup_price : ℚ := 3/2

def tomato_soup_quantity : ℕ := 3
def tomato_soup_price : ℚ := 5/4

def vegetable_soup_quantity : ℕ := 4
def vegetable_soup_price : ℚ := 7/4

def clam_chowder_quantity : ℕ := 2
def clam_chowder_price : ℚ := 2

def french_onion_soup_quantity : ℕ := 1
def french_onion_soup_price : ℚ := 9/5

def minestrone_soup_quantity : ℕ := 5
def minestrone_soup_price : ℚ := 17/10

def total_cost : ℚ := 
  chicken_soup_quantity * chicken_soup_price +
  tomato_soup_quantity * tomato_soup_price +
  vegetable_soup_quantity * vegetable_soup_price +
  clam_chowder_quantity * clam_chowder_price +
  french_onion_soup_quantity * french_onion_soup_price +
  minestrone_soup_quantity * minestrone_soup_price

theorem total_cost_is_correct : total_cost = 3405/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2221_222195


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2221_222110

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 25) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 1160 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2221_222110


namespace NUMINAMATH_CALUDE_hemisphere_properties_l2221_222166

/-- Properties of a hemisphere with base area 144π -/
theorem hemisphere_properties :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  (2 * π * r^2 + π * r^2 = 432 * π) ∧
  ((2 / 3) * π * r^3 = 1152 * π) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_properties_l2221_222166


namespace NUMINAMATH_CALUDE_first_triangular_year_21st_century_l2221_222190

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- First triangular number year in the 21st century -/
theorem first_triangular_year_21st_century :
  ∃ n : ℕ, triangular n = 2016 ∧ 
  (∀ m : ℕ, triangular m ≥ 2000 → triangular n ≤ triangular m) := by
  sorry

end NUMINAMATH_CALUDE_first_triangular_year_21st_century_l2221_222190


namespace NUMINAMATH_CALUDE_bricks_per_row_l2221_222100

theorem bricks_per_row (total_bricks : ℕ) (rows_per_wall : ℕ) (num_walls : ℕ) 
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) :
  total_bricks / (rows_per_wall * num_walls) = 30 := by
sorry

end NUMINAMATH_CALUDE_bricks_per_row_l2221_222100


namespace NUMINAMATH_CALUDE_binary_conversion_theorem_l2221_222125

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_conversion_theorem :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let base7 : List ℕ := decimal_to_base7 decimal
  decimal = 45 ∧ base7 = [6, 3] := by sorry

end NUMINAMATH_CALUDE_binary_conversion_theorem_l2221_222125


namespace NUMINAMATH_CALUDE_cab_cost_for_week_long_event_l2221_222102

/-- Calculates the total cost of cab rides for a week-long event -/
def total_cab_cost (days : ℕ) (distance : ℝ) (cost_per_mile : ℝ) : ℝ :=
  2 * days * distance * cost_per_mile

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_week_long_event :
  total_cab_cost 7 200 2.5 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_cab_cost_for_week_long_event_l2221_222102


namespace NUMINAMATH_CALUDE_min_distance_point_l2221_222194

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The Fermat point of a triangle --/
def fermatPoint (t : Triangle) : ℝ × ℝ := sorry

/-- The sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (t : Triangle) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest angle in a triangle --/
def largestAngle (t : Triangle) : ℝ := sorry

/-- The vertex corresponding to the largest angle in a triangle --/
def largestAngleVertex (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The point that minimizes the sum of distances to the vertices of a triangle --/
theorem min_distance_point (t : Triangle) :
  ∃ (M : ℝ × ℝ), (∀ (p : ℝ × ℝ), sumOfDistances t M ≤ sumOfDistances t p) ∧
  ((largestAngle t < 2 * Real.pi / 3 ∧ M = fermatPoint t) ∨
   (largestAngle t ≥ 2 * Real.pi / 3 ∧ M = largestAngleVertex t)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_point_l2221_222194


namespace NUMINAMATH_CALUDE_even_decreasing_function_a_equals_two_l2221_222105

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if f(x) > f(y) for all 0 < x < y -/
def IsDecreasingOnPositives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem even_decreasing_function_a_equals_two (a : ℤ) :
  IsEven (fun x => x^(a^2 - 4*a)) →
  IsDecreasingOnPositives (fun x => x^(a^2 - 4*a)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_function_a_equals_two_l2221_222105


namespace NUMINAMATH_CALUDE_trip_distance_l2221_222149

/-- The total distance of a trip between three cities forming a right-angled triangle -/
theorem trip_distance (DE EF FD : ℝ) (h1 : DE = 4500) (h2 : FD = 4000) 
  (h3 : DE^2 = EF^2 + FD^2) : DE + EF + FD = 10562 := by
  sorry

end NUMINAMATH_CALUDE_trip_distance_l2221_222149


namespace NUMINAMATH_CALUDE_annalise_tissue_purchase_cost_l2221_222139

/-- Calculates the total cost of tissues given the number of boxes, packs per box, tissues per pack, and cost per tissue -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * costPerTissue

/-- Proves that the total cost for Annalise's purchase is $1,000 -/
theorem annalise_tissue_purchase_cost :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

#eval totalCost 10 20 100 (5 / 100)

end NUMINAMATH_CALUDE_annalise_tissue_purchase_cost_l2221_222139


namespace NUMINAMATH_CALUDE_harolds_marbles_distribution_l2221_222120

theorem harolds_marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) 
  (best_friends : ℕ) (marbles_per_best_friend : ℕ) (cousins : ℕ) (marbles_per_cousin : ℕ) 
  (school_friends : ℕ) :
  total_marbles = 5000 →
  kept_marbles = 250 →
  best_friends = 3 →
  marbles_per_best_friend = 100 →
  cousins = 5 →
  marbles_per_cousin = 75 →
  school_friends = 10 →
  (total_marbles - (kept_marbles + best_friends * marbles_per_best_friend + 
    cousins * marbles_per_cousin)) / school_friends = 407 := by
  sorry

#check harolds_marbles_distribution

end NUMINAMATH_CALUDE_harolds_marbles_distribution_l2221_222120


namespace NUMINAMATH_CALUDE_red_ball_probability_l2221_222182

/-- The probability of drawing a red ball from a pocket containing white, black, and red balls -/
theorem red_ball_probability (white black red : ℕ) (h : red = 1) :
  (red : ℚ) / (white + black + red : ℚ) = 1 / 9 :=
by
  sorry

#check red_ball_probability 3 5 1 rfl

end NUMINAMATH_CALUDE_red_ball_probability_l2221_222182


namespace NUMINAMATH_CALUDE_min_value_fraction_l2221_222112

theorem min_value_fraction (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 2011) 
  (hb : 1 ≤ b ∧ b ≤ 2011) 
  (hc : 1 ≤ c ∧ c ≤ 2011) : 
  (a * b + c : ℚ) / (a + b + c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2221_222112


namespace NUMINAMATH_CALUDE_square_difference_divisible_by_13_l2221_222138

theorem square_difference_divisible_by_13 (a b : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 1000) 
  (h2 : 1 ≤ b ∧ b ≤ 1000) 
  (h3 : a + b = 1001) : 
  13 ∣ (a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_square_difference_divisible_by_13_l2221_222138


namespace NUMINAMATH_CALUDE_purchase_costs_l2221_222159

def cost (x y : ℕ) : ℕ := x + 2 * y

theorem purchase_costs : 
  (cost 5 5 ≤ 18) ∧ 
  (cost 9 4 ≤ 18) ∧ 
  (cost 9 5 > 18) ∧ 
  (cost 2 6 ≤ 18) ∧ 
  (cost 16 0 ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_purchase_costs_l2221_222159


namespace NUMINAMATH_CALUDE_arithmetic_geometric_k4_l2221_222177

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  q_nonzero : q ≠ 0
  is_geometric : ∀ n, as.a (k (n + 1)) = q * as.a (k n)
  k1_not_1 : k 1 ≠ 1
  k2_not_2 : k 2 ≠ 2
  k3_not_6 : k 3 ≠ 6

/-- The main theorem -/
theorem arithmetic_geometric_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_k4_l2221_222177


namespace NUMINAMATH_CALUDE_boxwood_count_proof_l2221_222158

/-- The cost to trim up each boxwood -/
def trim_cost : ℚ := 5

/-- The cost to trim a boxwood into a fancy shape -/
def fancy_trim_cost : ℚ := 15

/-- The number of boxwoods to be shaped into spheres -/
def fancy_trim_count : ℕ := 4

/-- The total charge for the service -/
def total_charge : ℚ := 210

/-- The number of boxwood hedges the customer wants trimmed up -/
def boxwood_count : ℕ := 30

theorem boxwood_count_proof :
  trim_cost * boxwood_count + fancy_trim_cost * fancy_trim_count = total_charge :=
by sorry

end NUMINAMATH_CALUDE_boxwood_count_proof_l2221_222158


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l2221_222189

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem exists_192_with_gcd_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem no_greater_than_192 :
  ∀ m : ℕ, 192 < m → m < 200 → Nat.gcd m 18 ≠ 6 :=
by sorry

theorem solution_is_192 : 
  ∃! n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l2221_222189


namespace NUMINAMATH_CALUDE_train_journey_time_l2221_222180

theorem train_journey_time (S T D : ℝ) (h1 : D = S * T) (h2 : D = (S / 2) * (T + 4)) :
  T + 4 = 8 := by sorry

end NUMINAMATH_CALUDE_train_journey_time_l2221_222180


namespace NUMINAMATH_CALUDE_fish_given_by_ben_l2221_222183

theorem fish_given_by_ben (initial_fish : ℕ) (current_fish : ℕ) 
  (h1 : initial_fish = 31) (h2 : current_fish = 49) : 
  current_fish - initial_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_fish_given_by_ben_l2221_222183


namespace NUMINAMATH_CALUDE_five_power_sum_squares_l2221_222162

/-- A function that checks if a number is expressible as a sum of two squares -/
def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^2

/-- A function that checks if two numbers have the same parity -/
def same_parity (n m : ℕ) : Prop :=
  n % 2 = m % 2

theorem five_power_sum_squares (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  is_sum_of_two_squares (5^m + 5^n) ↔ same_parity n m :=
sorry

end NUMINAMATH_CALUDE_five_power_sum_squares_l2221_222162


namespace NUMINAMATH_CALUDE_triangle_solution_l2221_222134

theorem triangle_solution (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 →
  C = π - A - B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b = 2 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_solution_l2221_222134


namespace NUMINAMATH_CALUDE_sum_of_five_variables_l2221_222156

theorem sum_of_five_variables (a b c d e : ℝ) 
  (eq1 : a + b = 16)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 5)
  (eq5 : e + a = 7) :
  a + b + c + d + e = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_variables_l2221_222156


namespace NUMINAMATH_CALUDE_triangle_side_length_l2221_222143

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → b = 5 → C = 2 * π / 3 → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2221_222143


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l2221_222187

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular room -/
def perimeter (room : RoomDimensions) : ℝ :=
  2 * (room.length + room.width)

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem width_to_perimeter_ratio (room : RoomDimensions)
    (h1 : room.length = 25)
    (h2 : room.width = 15) :
    simplifyRatio (Nat.floor room.width) (Nat.floor (perimeter room)) = (3, 16) := by
  sorry

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l2221_222187


namespace NUMINAMATH_CALUDE_books_about_sports_l2221_222144

theorem books_about_sports (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 :=
sorry

end NUMINAMATH_CALUDE_books_about_sports_l2221_222144


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2221_222115

theorem sum_of_three_numbers : 85.9 + 5.31 + (43 / 2 : ℝ) = 112.71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2221_222115


namespace NUMINAMATH_CALUDE_arrange_five_classes_four_factories_l2221_222148

/-- The number of ways to arrange classes into factories -/
def arrange_classes (num_classes : ℕ) (num_factories : ℕ) : ℕ :=
  (num_classes.choose 2) * (num_factories.factorial)

/-- Theorem: The number of ways to arrange 5 classes into 4 factories is 240 -/
theorem arrange_five_classes_four_factories :
  arrange_classes 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_classes_four_factories_l2221_222148


namespace NUMINAMATH_CALUDE_operation_on_81_divided_by_3_l2221_222186

theorem operation_on_81_divided_by_3 : ∃ f : ℝ → ℝ, (f 81) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_operation_on_81_divided_by_3_l2221_222186


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2221_222157

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let pepperoni_slices : ℕ := total_slices / 3
  let plain_cost : ℚ := 12
  let pepperoni_cost : ℚ := 3
  let total_cost : ℚ := plain_cost + pepperoni_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let pepperoni_slice_cost : ℚ := cost_per_slice + pepperoni_cost / pepperoni_slices
  let plain_slice_cost : ℚ := cost_per_slice
  let mark_pepperoni_slices : ℕ := pepperoni_slices
  let mark_plain_slices : ℕ := 2
  let anne_slices : ℕ := total_slices - mark_pepperoni_slices - mark_plain_slices
  let mark_cost : ℚ := mark_pepperoni_slices * pepperoni_slice_cost + mark_plain_slices * plain_slice_cost
  let anne_cost : ℚ := anne_slices * plain_slice_cost
  mark_cost - anne_cost = 3 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2221_222157


namespace NUMINAMATH_CALUDE_expression_evaluation_l2221_222135

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d - 2)^d + d^2)^d = 1874164224 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2221_222135


namespace NUMINAMATH_CALUDE_crystal_cupcake_sales_l2221_222133

def crystal_sales (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction_factor : ℚ) (total_revenue : ℚ) (cookies_sold : ℕ) : Prop :=
  let reduced_cupcake_price := original_cupcake_price * price_reduction_factor
  let reduced_cookie_price := original_cookie_price * price_reduction_factor
  let cookie_revenue := reduced_cookie_price * cookies_sold
  let cupcake_revenue := total_revenue - cookie_revenue
  let cupcakes_sold := cupcake_revenue / reduced_cupcake_price
  cupcakes_sold = 16

theorem crystal_cupcake_sales :
  crystal_sales 3 2 (1/2) 32 8 := by sorry

end NUMINAMATH_CALUDE_crystal_cupcake_sales_l2221_222133


namespace NUMINAMATH_CALUDE_angle5_is_36_degrees_l2221_222121

-- Define the angles
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom parallel_lines : True  -- m ∥ n
axiom angle1_is_quarter_angle2 : angle1 = (1 / 4) * angle2
axiom alternate_interior_angles : angle5 = angle1
axiom straight_line : angle2 + angle5 = 180

-- Theorem to prove
theorem angle5_is_36_degrees : angle5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle5_is_36_degrees_l2221_222121


namespace NUMINAMATH_CALUDE_yellow_daisy_percentage_l2221_222170

/-- Represents the types of flowers in the collection -/
inductive FlowerType
| Tulip
| Daisy

/-- Represents the colors of flowers in the collection -/
inductive FlowerColor
| Red
| Yellow

/-- Represents the collection of flowers -/
structure FlowerCollection where
  total : ℕ
  tulips : ℕ
  redTulips : ℕ
  yellowDaisies : ℕ

/-- The theorem statement -/
theorem yellow_daisy_percentage
  (collection : FlowerCollection)
  (h_total : collection.total = 120)
  (h_tulips : collection.tulips = collection.total * 3 / 10)
  (h_redTulips : collection.redTulips = collection.tulips / 2)
  (h_yellowDaisies : collection.yellowDaisies = (collection.total - collection.tulips) * 3 / 5) :
  collection.yellowDaisies * 100 / collection.total = 40 := by
  sorry

end NUMINAMATH_CALUDE_yellow_daisy_percentage_l2221_222170


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l2221_222119

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0) →
  (q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0) →
  (r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0) →
  (s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l2221_222119


namespace NUMINAMATH_CALUDE_article_pricing_loss_l2221_222107

/-- Proves that for an article with a given cost price, selling at 216 results in a 20% profit,
    and selling at 153 results in a 15% loss. -/
theorem article_pricing_loss (CP : ℝ) : 
  CP * 1.2 = 216 → (CP - 153) / CP * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_loss_l2221_222107


namespace NUMINAMATH_CALUDE_no_real_solutions_l2221_222146

theorem no_real_solutions : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 6*y + 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2221_222146


namespace NUMINAMATH_CALUDE_purple_top_implies_violet_bottom_l2221_222167

/-- Represents the colors of the cube faces -/
inductive Color
  | R | P | O | Y | G | V

/-- Represents a cube with colored faces -/
structure Cube where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Represents the configuration of the six squares before folding -/
structure SquareConfiguration where
  square1 : Color
  square2 : Color
  square3 : Color
  square4 : Color
  square5 : Color
  square6 : Color

/-- Function to fold the squares into a cube -/
def foldIntoCube (config : SquareConfiguration) : Cube :=
  sorry

/-- Theorem stating that if P is on top, V is on the bottom -/
theorem purple_top_implies_violet_bottom (config : SquareConfiguration) :
  let cube := foldIntoCube config
  cube.top = Color.P → cube.bottom = Color.V :=
sorry

end NUMINAMATH_CALUDE_purple_top_implies_violet_bottom_l2221_222167


namespace NUMINAMATH_CALUDE_all_propositions_correct_l2221_222122

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

theorem all_propositions_correct :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  (double_factorial 2010 = 2^1005 * Nat.factorial 1005) ∧
  (double_factorial 2010 % 10 = 0) ∧
  (double_factorial 2011 % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_correct_l2221_222122


namespace NUMINAMATH_CALUDE_trapezoid_area_l2221_222193

-- Define the trapezoid ABCD and points X and Y
structure Trapezoid :=
  (A B C D X Y : ℝ × ℝ)

-- Define the properties of the trapezoid
def isIsosceles (t : Trapezoid) : Prop :=
  t.A.1 - t.B.1 = t.D.1 - t.C.1

def isParallel (t : Trapezoid) : Prop :=
  t.B.2 - t.C.2 = t.A.2 - t.D.2

def onDiagonal (t : Trapezoid) : Prop :=
  ∃ k₁ k₂ : ℝ, 0 ≤ k₁ ∧ k₁ ≤ k₂ ∧ k₂ ≤ 1 ∧
  t.X = (1 - k₁) • t.A + k₁ • t.C ∧
  t.Y = (1 - k₂) • t.A + k₂ • t.C

def angle (p q r : ℝ × ℝ) : ℝ := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

def area (t : Trapezoid) : ℝ := sorry

-- State the theorem
theorem trapezoid_area (t : Trapezoid) :
  isIsosceles t →
  isParallel t →
  onDiagonal t →
  angle t.A t.X t.D = π / 3 →
  angle t.B t.Y t.C = 2 * π / 3 →
  distance t.A t.X = 4 →
  distance t.X t.Y = 2 →
  distance t.Y t.C = 3 →
  area t = 21 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2221_222193


namespace NUMINAMATH_CALUDE_inequality_chain_l2221_222103

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ∧
  (a^2 + b^2)/(2*c) + (a^2 + c^2)/(2*b) + (b^2 + c^2)/(2*a) ≤ a^3/(b*c) + b^3/(a*c) + c^3/(a*b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l2221_222103


namespace NUMINAMATH_CALUDE_binary_sum_equals_141_l2221_222168

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number $1010101_2$ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number $111000_2$ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- Theorem stating that the sum of the two binary numbers is 141 in decimal -/
theorem binary_sum_equals_141 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_141_l2221_222168


namespace NUMINAMATH_CALUDE_dice_roll_circle_probability_l2221_222128

theorem dice_roll_circle_probability (r : ℕ) (h1 : 3 ≤ r) (h2 : r ≤ 18) :
  2 * Real.pi * r ≤ 2 * Real.pi * r^2 := by sorry

end NUMINAMATH_CALUDE_dice_roll_circle_probability_l2221_222128


namespace NUMINAMATH_CALUDE_goldfish_count_l2221_222136

/-- Represents the number of fish in each tank -/
structure FishTanks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the composition of the first tank -/
structure FirstTank where
  goldfish : ℕ
  beta : ℕ

/-- The problem statement -/
theorem goldfish_count (tanks : FishTanks) (first : FirstTank) : 
  tanks.first = first.goldfish + first.beta ∧
  tanks.second = 2 * tanks.first ∧
  tanks.third = tanks.second / 3 ∧
  tanks.third = 10 ∧
  first.beta = 8 →
  first.goldfish = 7 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l2221_222136


namespace NUMINAMATH_CALUDE_intersection_area_implies_m_values_l2221_222185

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (m x y : ℝ) : Prop := x - m*y + 2 = 0

def AreaABO (m : ℝ) : ℝ := 2

theorem intersection_area_implies_m_values (m : ℝ) :
  (∃ A B : ℝ × ℝ, 
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ 
    Line m A.1 A.2 ∧ Line m B.1 B.2 ∧
    AreaABO m = 2) →
  m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_implies_m_values_l2221_222185


namespace NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l2221_222198

/-- Given a cube with diagonal length a/2, prove that its total surface area is a^2/2 -/
theorem cube_surface_area_from_diagonal (a : ℝ) (h : a > 0) :
  let diagonal := a / 2
  let side := diagonal / Real.sqrt 3
  let surface_area := 6 * side ^ 2
  surface_area = a ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l2221_222198


namespace NUMINAMATH_CALUDE_mountain_bike_helmet_cost_l2221_222153

/-- Calculates the cost of a mountain bike helmet based on Alfonso's savings and earnings --/
theorem mountain_bike_helmet_cost
  (daily_earnings : ℕ)
  (current_savings : ℕ)
  (days_per_week : ℕ)
  (weeks_to_work : ℕ)
  (h1 : daily_earnings = 6)
  (h2 : current_savings = 40)
  (h3 : days_per_week = 5)
  (h4 : weeks_to_work = 10) :
  daily_earnings * days_per_week * weeks_to_work + current_savings = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_mountain_bike_helmet_cost_l2221_222153


namespace NUMINAMATH_CALUDE_weight_of_A_l2221_222101

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 77 kg -/
theorem weight_of_A (A B C D E : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  E = D + 5 →
  (B + C + D + E) / 4 = 79 →
  A = 77 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l2221_222101


namespace NUMINAMATH_CALUDE_stadium_entry_exit_options_l2221_222142

theorem stadium_entry_exit_options (south_gates north_gates : ℕ) 
  (h1 : south_gates = 4) 
  (h2 : north_gates = 3) : 
  (south_gates + north_gates) * (south_gates + north_gates) = 49 := by
  sorry

end NUMINAMATH_CALUDE_stadium_entry_exit_options_l2221_222142


namespace NUMINAMATH_CALUDE_contractor_job_problem_l2221_222163

/-- A contractor's job problem -/
theorem contractor_job_problem
  (total_days : ℕ) (initial_workers : ℕ) (first_period : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_workers = 10)
  (h3 : first_period = 20)
  (h4 : remaining_days = 75)
  (h5 : first_period * initial_workers = (total_days * initial_workers) / 4) :
  ∃ (fired : ℕ), 
    fired = 2 ∧
    remaining_days * (initial_workers - fired) = 
      (total_days * initial_workers) - (first_period * initial_workers) :=
by sorry

end NUMINAMATH_CALUDE_contractor_job_problem_l2221_222163


namespace NUMINAMATH_CALUDE_scientific_notation_of_86000_l2221_222104

theorem scientific_notation_of_86000 (average_price : ℝ) : 
  average_price = 86000 → average_price = 8.6 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_86000_l2221_222104


namespace NUMINAMATH_CALUDE_fair_spending_l2221_222169

def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

theorem fair_spending : money_at_arrival - money_at_departure = 71 := by
  sorry

end NUMINAMATH_CALUDE_fair_spending_l2221_222169


namespace NUMINAMATH_CALUDE_min_y_value_l2221_222113

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 30*x + 20*y) :
  ∃ (y_min : ℝ), y_min = 10 - 5 * Real.sqrt 13 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 30*x' + 20*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l2221_222113


namespace NUMINAMATH_CALUDE_rotated_line_equation_l2221_222145

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a line 90 degrees counterclockwise around a given point -/
def rotateLine90 (l : Line) (p : Point) : Line :=
  sorry

/-- The original line l₀ -/
def l₀ : Line :=
  { slope := 1, yIntercept := 1 }

/-- The point P around which the line is rotated -/
def P : Point :=
  { x := 3, y := 1 }

/-- The resulting line l after rotation -/
def l : Line :=
  rotateLine90 l₀ P

theorem rotated_line_equation :
  l.slope * P.x + l.yIntercept = P.y ∧ l.slope = -1 →
  ∀ x y, y + x - 4 = 0 ↔ y = l.slope * x + l.yIntercept :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l2221_222145


namespace NUMINAMATH_CALUDE_xyz_inequality_l2221_222172

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y*z + z*x + x*y - 2*x*y*z ∧ y*z + z*x + x*y - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2221_222172


namespace NUMINAMATH_CALUDE_singing_competition_result_l2221_222124

def singing_competition (total_contestants : ℕ) 
                        (female_solo_percent : ℚ) 
                        (male_solo_percent : ℚ) 
                        (group_percent : ℚ) 
                        (male_young_percent : ℚ) 
                        (female_young_percent : ℚ) : Prop :=
  let female_solo := ⌊(female_solo_percent * total_contestants : ℚ)⌋
  let male_solo := ⌊(male_solo_percent * total_contestants : ℚ)⌋
  let male_young := ⌊(male_young_percent * male_solo : ℚ)⌋
  let female_young := ⌊(female_young_percent * female_solo : ℚ)⌋
  total_contestants = 18 ∧
  female_solo_percent = 35/100 ∧
  male_solo_percent = 25/100 ∧
  group_percent = 40/100 ∧
  male_young_percent = 30/100 ∧
  female_young_percent = 20/100 ∧
  male_young = 1 ∧
  female_young = 1

theorem singing_competition_result : 
  singing_competition 18 (35/100) (25/100) (40/100) (30/100) (20/100) := by
  sorry

end NUMINAMATH_CALUDE_singing_competition_result_l2221_222124


namespace NUMINAMATH_CALUDE_bd_equals_twelve_l2221_222154

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- Theorem stating that BD = 12 is a valid solution for the given quadrilateral --/
theorem bd_equals_twelve (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 12)
  (h3 : q.CD = 6)
  (h4 : q.DA = 8) :
  q.BD = 12 → 
  (q.AB + q.BD > q.DA) ∧ 
  (q.BC + q.CD > q.BD) ∧ 
  (q.DA + q.BD > q.AB) ∧ 
  (q.BD + q.CD > q.BC) ∧ 
  (q.BD > 6) ∧ 
  (q.BD < 14) := by
  sorry

#check bd_equals_twelve

end NUMINAMATH_CALUDE_bd_equals_twelve_l2221_222154


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l2221_222199

theorem rock_collecting_contest (sydney_initial : ℕ) (conner_initial : ℕ) 
  (conner_day2 : ℕ) (conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  conner_day2 = 123 →
  conner_day3 = 27 →
  ∃ (sydney_day1 : ℕ),
    sydney_day1 ≤ 4 ∧
    sydney_day1 > 0 ∧
    conner_initial + 8 * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + 16 * sydney_day1 ∧
    ∀ (x : ℕ), x > sydney_day1 →
      conner_initial + 8 * x + conner_day2 + conner_day3 < 
      sydney_initial + x + 16 * x :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l2221_222199


namespace NUMINAMATH_CALUDE_tens_digit_of_3_pow_2016_l2221_222129

theorem tens_digit_of_3_pow_2016 :
  ∃ n : ℕ, 3^2016 = 100*n + 21 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_pow_2016_l2221_222129


namespace NUMINAMATH_CALUDE_checkerboard_black_squares_l2221_222109

theorem checkerboard_black_squares (n : ℕ) (h : n = 29) :
  let total_squares := n * n
  let black_squares := (n - 1) * (n - 1) / 2 + (n - 1)
  black_squares = 420 := by
sorry

end NUMINAMATH_CALUDE_checkerboard_black_squares_l2221_222109


namespace NUMINAMATH_CALUDE_stool_height_is_34cm_l2221_222197

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_height : ℝ) 
                 (light_bulb_distance_from_ceiling : ℝ)
                 (alice_height alice_reach : ℝ) : ℝ :=
  ceiling_height - floor_height - light_bulb_distance_from_ceiling - 
  (alice_height + alice_reach)

/-- Theorem stating the height of the stool Alice needs -/
theorem stool_height_is_34cm :
  let ceiling_height : ℝ := 2.4 * 100  -- Convert to cm
  let floor_height : ℝ := 0
  let light_bulb_distance_from_ceiling : ℝ := 10
  let alice_height : ℝ := 1.5 * 100  -- Convert to cm
  let alice_reach : ℝ := 46
  stool_height ceiling_height floor_height light_bulb_distance_from_ceiling
                alice_height alice_reach = 34 := by
  sorry

#eval stool_height (2.4 * 100) 0 10 (1.5 * 100) 46

end NUMINAMATH_CALUDE_stool_height_is_34cm_l2221_222197


namespace NUMINAMATH_CALUDE_a_profit_share_l2221_222175

-- Define the total investment and profit
def total_investment : ℕ := 90000
def total_profit : ℕ := 8640

-- Define the relationships between investments
def investment_relations (a b c : ℕ) : Prop :=
  a = b + 6000 ∧ b + 3000 = c ∧ a + b + c = total_investment

-- Define the profit sharing ratio
def profit_ratio (a b c : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ a / k = 11 ∧ b / k = 9 ∧ c / k = 10

-- Theorem statement
theorem a_profit_share (a b c : ℕ) :
  investment_relations a b c →
  profit_ratio a b c →
  (11 : ℚ) / 30 * total_profit = 3168 :=
by sorry

end NUMINAMATH_CALUDE_a_profit_share_l2221_222175


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2221_222171

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 * x + c = 0) →
  a + c = 17 →
  a > c →
  a = 15.375 ∧ c = 1.625 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2221_222171
