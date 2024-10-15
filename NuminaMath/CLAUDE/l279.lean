import Mathlib

namespace NUMINAMATH_CALUDE_min_xy_value_l279_27921

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x * y ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l279_27921


namespace NUMINAMATH_CALUDE_rectangle_path_ratio_l279_27940

/-- Represents a rectangle on a lattice grid --/
structure LatticeRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the number of shortest paths between opposite corners of a rectangle --/
def shortestPaths (rect : LatticeRectangle) : ℕ :=
  Nat.choose (rect.width + rect.height) rect.width

/-- Theorem: For a rectangle with height = k * width, the number of paths starting vertically
    is k times the number of paths starting horizontally --/
theorem rectangle_path_ratio {k : ℕ} (rect : LatticeRectangle) 
    (h : rect.height = k * rect.width) :
  shortestPaths ⟨rect.height, rect.width⟩ = k * shortestPaths ⟨rect.width, rect.height⟩ := by
  sorry

#check rectangle_path_ratio

end NUMINAMATH_CALUDE_rectangle_path_ratio_l279_27940


namespace NUMINAMATH_CALUDE_greater_number_is_84_l279_27938

theorem greater_number_is_84 (x y : ℝ) (h1 : x * y = 2688) (h2 : (x + y) - (x - y) = 64) : max x y = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_is_84_l279_27938


namespace NUMINAMATH_CALUDE_binomial_30_3_l279_27983

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l279_27983


namespace NUMINAMATH_CALUDE_noemi_blackjack_loss_l279_27963

/-- Calculates the amount lost on blackjack given initial amount, amount lost on roulette, and final amount -/
def blackjack_loss (initial : ℕ) (roulette_loss : ℕ) (final : ℕ) : ℕ :=
  initial - roulette_loss - final

/-- Proves that Noemi lost $500 on blackjack -/
theorem noemi_blackjack_loss :
  let initial := 1700
  let roulette_loss := 400
  let final := 800
  blackjack_loss initial roulette_loss final = 500 := by
  sorry

end NUMINAMATH_CALUDE_noemi_blackjack_loss_l279_27963


namespace NUMINAMATH_CALUDE_max_value_of_trig_expression_l279_27982

theorem max_value_of_trig_expression :
  ∀ α : Real, 0 ≤ α ∧ α ≤ π / 2 →
    (∀ β : Real, 0 ≤ β ∧ β ≤ π / 2 → 
      1 / (Real.sin β ^ 6 + Real.cos β ^ 6) ≤ 1 / (Real.sin α ^ 6 + Real.cos α ^ 6)) →
    1 / (Real.sin α ^ 6 + Real.cos α ^ 6) = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_expression_l279_27982


namespace NUMINAMATH_CALUDE_lemons_for_drinks_l279_27958

/-- The number of lemons needed to make a certain amount of lemonade and lemon tea -/
def lemons_needed (lemonade_ratio : ℚ) (tea_ratio : ℚ) (lemonade_gallons : ℚ) (tea_gallons : ℚ) : ℚ :=
  lemonade_ratio * lemonade_gallons + tea_ratio * tea_gallons

/-- Theorem stating the number of lemons needed for 6 gallons of lemonade and 5 gallons of lemon tea -/
theorem lemons_for_drinks : 
  let lemonade_ratio : ℚ := 36 / 48
  let tea_ratio : ℚ := 20 / 10
  lemons_needed lemonade_ratio tea_ratio 6 5 = 29/2 := by
  sorry

#eval (29 : ℚ) / 2  -- To verify that 29/2 is indeed equal to 14.5

end NUMINAMATH_CALUDE_lemons_for_drinks_l279_27958


namespace NUMINAMATH_CALUDE_tax_increase_l279_27980

/-- Calculates the tax amount based on income and tax rates -/
def calculate_tax (income : ℕ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  if income ≤ 500000 then
    (income : ℚ) * rate1
  else if income ≤ 1000000 then
    500000 * rate1 + ((income - 500000) : ℚ) * rate2
  else
    500000 * rate1 + 500000 * rate2 + ((income - 1000000) : ℚ) * rate2

/-- Represents the tax system change and income increase -/
theorem tax_increase :
  let old_tax := calculate_tax 1000000 (20/100) (25/100)
  let new_main_tax := calculate_tax 1500000 (30/100) (35/100)
  let rental_income : ℚ := 100000
  let rental_deduction : ℚ := 10/100
  let taxable_rental := rental_income * (1 - rental_deduction)
  let rental_tax := taxable_rental * (35/100)
  let new_total_tax := new_main_tax + rental_tax
  new_total_tax - old_tax = 306500 := by sorry

end NUMINAMATH_CALUDE_tax_increase_l279_27980


namespace NUMINAMATH_CALUDE_bd_length_is_ten_l279_27961

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the right angle at C
def isRightAngleAtC (t : Triangle) : Prop :=
  let (xa, ya) := t.A
  let (xb, yb) := t.B
  let (xc, yc) := t.C
  (xc - xa) * (xc - xb) + (yc - ya) * (yc - yb) = 0

-- Define the lengths of AC and BC
def AC_length (t : Triangle) : ℝ := 5
def BC_length (t : Triangle) : ℝ := 12

-- Define points D, E, F
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry
def F (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle at FED
def isRightAngleAtFED (t : Triangle) : Prop :=
  let (xd, yd) := D t
  let (xe, ye) := E t
  let (xf, yf) := F t
  (xf - xd) * (xe - xd) + (yf - yd) * (ye - yd) = 0

-- Define the lengths of DE and DF
def DE_length (t : Triangle) : ℝ := 5
def DF_length (t : Triangle) : ℝ := 3

-- Define the length of BD
def BD_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_ten (t : Triangle) :
  isRightAngleAtC t →
  isRightAngleAtFED t →
  BD_length t = 10 := by sorry

end NUMINAMATH_CALUDE_bd_length_is_ten_l279_27961


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l279_27941

theorem arithmetic_expression_equality : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l279_27941


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l279_27928

/-- Given an ellipse mx^2 + y^2 = 1 with eccentricity √3/2, its major axis length is either 2 or 4 -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ a^2 * m = b^2 ∧ (a^2 - b^2) / a^2 = 3/4) →  -- Eccentricity condition
  (∃ (l : ℝ), l = 2 ∨ l = 4 ∧ l = 2 * a) :=  -- Major axis length
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l279_27928


namespace NUMINAMATH_CALUDE_guitar_picks_l279_27974

theorem guitar_picks (total : ℕ) (red blue yellow : ℕ) : 
  2 * red = total →
  3 * blue = total →
  red + blue + yellow = total →
  blue = 12 →
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_guitar_picks_l279_27974


namespace NUMINAMATH_CALUDE_calculate_tax_rate_l279_27918

/-- Given a total purchase amount, percentage of total spent on sales tax,
    and the cost of tax-free items, calculate the tax rate on taxable purchases. -/
theorem calculate_tax_rate (total_purchase : ℝ) (tax_percentage : ℝ) (tax_free_cost : ℝ) :
  total_purchase = 40 →
  tax_percentage = 30 →
  tax_free_cost = 34.7 →
  ∃ (tax_rate : ℝ), abs (tax_rate - 226.42) < 0.01 ∧
    tax_rate = (tax_percentage / 100 * total_purchase) / (total_purchase - tax_free_cost) * 100 :=
by sorry

end NUMINAMATH_CALUDE_calculate_tax_rate_l279_27918


namespace NUMINAMATH_CALUDE_length_of_AB_is_8_l279_27989

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = -4*x ∧ x < 0

-- Define point P
def P : ℝ × ℝ := (-3, -2)

-- Define the line l passing through P
def l (x y : ℝ) : Prop := y + 2 = x + 3

-- Define the property of P being the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Main theorem
theorem length_of_AB_is_8 :
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 → C B.1 B.2 →
  l A.1 A.2 → l B.1 B.2 →
  is_midpoint A B →
  ‖A - B‖ = 8 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_is_8_l279_27989


namespace NUMINAMATH_CALUDE_polygon_exterior_interior_sum_equal_l279_27909

theorem polygon_exterior_interior_sum_equal (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_interior_sum_equal_l279_27909


namespace NUMINAMATH_CALUDE_hannah_dolls_multiplier_l279_27932

theorem hannah_dolls_multiplier (x : ℝ) : 
  x > 0 → -- Hannah has a positive number of times as many dolls
  8 * x + 8 = 48 → -- Total dolls equation
  x = 5 := by sorry

end NUMINAMATH_CALUDE_hannah_dolls_multiplier_l279_27932


namespace NUMINAMATH_CALUDE_parabola_rhombus_theorem_l279_27900

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the rhombus
def rhombus (O B F C : ℝ × ℝ) : Prop :=
  let (xo, yo) := O
  let (xb, yb) := B
  let (xf, yf) := F
  let (xc, yc) := C
  (xf - xo)^2 + (yf - yo)^2 = (xb - xc)^2 + (yb - yc)^2 ∧
  (xb - xo)^2 + (yb - yo)^2 = (xc - xo)^2 + (yc - yo)^2

-- Define the theorem
theorem parabola_rhombus_theorem (p : ℝ) (O B F C : ℝ × ℝ) :
  parabola p B.1 B.2 →
  parabola p C.1 C.2 →
  rhombus O B F C →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 →
  p = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parabola_rhombus_theorem_l279_27900


namespace NUMINAMATH_CALUDE_general_term_is_arithmetic_l279_27977

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := n^2 + 2*n

def a : ℕ → ℕ := fun n => S n - S (n-1)

-- Theorem 1: The general term of the sequence
theorem general_term : ∀ n : ℕ, n > 0 → a n = 2*n + 1 :=
sorry

-- Definition of arithmetic sequence
def is_arithmetic_sequence (f : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, n > 1 → f n - f (n-1) = d

-- Theorem 2: The sequence is arithmetic
theorem is_arithmetic : is_arithmetic_sequence a :=
sorry

end NUMINAMATH_CALUDE_general_term_is_arithmetic_l279_27977


namespace NUMINAMATH_CALUDE_function_equality_implies_a_range_l279_27997

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem function_equality_implies_a_range (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 5 ∨ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_range_l279_27997


namespace NUMINAMATH_CALUDE_average_of_quadratic_roots_l279_27953

theorem average_of_quadratic_roots (p q : ℝ) (h : p ≠ 0) :
  let f : ℝ → ℝ := λ x => 3 * p * x^2 - 6 * p * x + q
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_quadratic_roots_l279_27953


namespace NUMINAMATH_CALUDE_triangle_inequality_constant_l279_27995

theorem triangle_inequality_constant (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 ∧
  ∀ N : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < N) →
  2 ≤ N :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_constant_l279_27995


namespace NUMINAMATH_CALUDE_sandy_initial_money_l279_27987

def sandy_shopping (initial_money : ℝ) : Prop :=
  let watch_price : ℝ := 50
  let shirt_price : ℝ := 30
  let shoes_price : ℝ := 70
  let shirt_discount : ℝ := 0.1
  let shoes_discount : ℝ := 0.2
  let spent_percentage : ℝ := 0.3
  let money_left : ℝ := 210
  
  let total_cost : ℝ := watch_price + 
    shirt_price * (1 - shirt_discount) + 
    shoes_price * (1 - shoes_discount)
  
  (initial_money * spent_percentage = total_cost) ∧
  (initial_money * (1 - spent_percentage) = money_left)

theorem sandy_initial_money :
  sandy_shopping 300 := by sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l279_27987


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l279_27946

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  -a + a₁ - a₂ + a₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l279_27946


namespace NUMINAMATH_CALUDE_possible_numbers_correct_l279_27936

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
(hundreds : Digit)
(tens : Digit)
(ones : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.ones with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0

/-- The set of all possible original numbers -/
def possibleNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- Function to check if a number can be displayed as 351 with two malfunctioning segments -/
def canBeDisplayedAs351WithTwoMalfunctions (n : ThreeDigitNumber) : Prop :=
  ∃ (seg1 seg2 : Nat), seg1 ≠ seg2 ∧ seg1 < 7 ∧ seg2 < 7 ∧
    (n.toNat ∈ possibleNumbers)

/-- Theorem stating that the set of possible numbers is correct -/
theorem possible_numbers_correct :
  ∀ n : ThreeDigitNumber, canBeDisplayedAs351WithTwoMalfunctions n ↔ n.toNat ∈ possibleNumbers :=
sorry

end NUMINAMATH_CALUDE_possible_numbers_correct_l279_27936


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_four_l279_27957

theorem tan_alpha_two_implies_fraction_equals_four (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_four_l279_27957


namespace NUMINAMATH_CALUDE_solve_equation_l279_27962

theorem solve_equation : ∃ x : ℚ, (5*x + 8*x = 350 - 9*(x+8)) ∧ (x = 12 + 7/11) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l279_27962


namespace NUMINAMATH_CALUDE_jane_start_babysitting_age_l279_27975

/-- Represents the age at which Jane started babysitting -/
def start_age : ℕ := 8

/-- Jane's current age -/
def current_age : ℕ := 32

/-- Years since Jane stopped babysitting -/
def years_since_stopped : ℕ := 10

/-- Current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 24

/-- Theorem stating that Jane started babysitting at age 8 -/
theorem jane_start_babysitting_age :
  (start_age + years_since_stopped < current_age) ∧
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ start_age →
    jane_age ≤ current_age - years_since_stopped →
    child_age ≤ oldest_babysat_current_age - (current_age - jane_age) →
    child_age ≤ jane_age / 2) ∧
  (oldest_babysat_current_age = current_age - (start_age + 8)) :=
by sorry

#check jane_start_babysitting_age

end NUMINAMATH_CALUDE_jane_start_babysitting_age_l279_27975


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l279_27965

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  (∃ (c : ℝ), c ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = c • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l279_27965


namespace NUMINAMATH_CALUDE_negative_three_inverse_l279_27959

theorem negative_three_inverse : (-3 : ℚ)⁻¹ = -1/3 := by sorry

end NUMINAMATH_CALUDE_negative_three_inverse_l279_27959


namespace NUMINAMATH_CALUDE_probability_of_different_homes_l279_27935

def num_volunteers : ℕ := 5
def num_homes : ℕ := 2

def probability_different_homes : ℚ := 8/15

theorem probability_of_different_homes :
  let total_arrangements := (2^num_volunteers - 2)
  let arrangements_same_home := (2^(num_volunteers - 2) - 1) * 2
  (total_arrangements - arrangements_same_home : ℚ) / total_arrangements = probability_different_homes :=
sorry

end NUMINAMATH_CALUDE_probability_of_different_homes_l279_27935


namespace NUMINAMATH_CALUDE_circle_tangency_distance_ratio_l279_27990

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the four circles
variable (A₁ A₂ A₃ A₄ : Circle)

-- Define the points
variable (P T₁ T₂ T₃ T₄ : Point)

-- Define the tangency and intersection relations
variable (tangent : Circle → Circle → Point → Prop)
variable (intersect : Circle → Circle → Point → Prop)

-- State the theorem
theorem circle_tangency_distance_ratio
  (h1 : tangent A₁ A₃ P)
  (h2 : tangent A₂ A₄ P)
  (h3 : intersect A₁ A₂ T₁)
  (h4 : intersect A₂ A₃ T₂)
  (h5 : intersect A₃ A₄ T₃)
  (h6 : intersect A₄ A₁ T₄)
  (h7 : T₁ ≠ P ∧ T₂ ≠ P ∧ T₃ ≠ P ∧ T₄ ≠ P) :
  (dist T₁ T₂ * dist T₂ T₃) / (dist T₁ T₄ * dist T₃ T₄) = (dist P T₂)^2 / (dist P T₄)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_distance_ratio_l279_27990


namespace NUMINAMATH_CALUDE_complex_equation_solution_l279_27920

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) = (2 + Complex.I) / (1 + a * Complex.I) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l279_27920


namespace NUMINAMATH_CALUDE_string_length_problem_l279_27904

/-- Given three strings A, B, and C, where the length of A is 6 times the length of C
    and 5 times the length of B, and the length of B is 12 meters,
    prove that the length of C is 10 meters. -/
theorem string_length_problem (A B C : ℝ) 
    (h1 : A = 6 * C) 
    (h2 : A = 5 * B) 
    (h3 : B = 12) : 
  C = 10 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l279_27904


namespace NUMINAMATH_CALUDE_breakfast_consumption_l279_27969

/-- Represents the number of slices of bread each member consumes during breakfast -/
def breakfast_slices : ℕ := 3

/-- Represents the number of members in the household -/
def household_members : ℕ := 4

/-- Represents the number of slices each member consumes for snacks -/
def snack_slices : ℕ := 2

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves that last for 3 days -/
def loaves_for_three_days : ℕ := 5

/-- Represents the number of days the loaves last -/
def days_lasted : ℕ := 3

theorem breakfast_consumption :
  breakfast_slices = 3 ∧
  household_members * (breakfast_slices + snack_slices) * days_lasted = 
  loaves_for_three_days * slices_per_loaf := by
  sorry

#check breakfast_consumption

end NUMINAMATH_CALUDE_breakfast_consumption_l279_27969


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l279_27996

/-- Proves that given a mixture of x litres with 20% alcohol, when 5 litres of water are added 
    resulting in a new mixture with 15% alcohol, the value of x is 15 litres. -/
theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.20 * x = 0.15 * (x + 5)) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l279_27996


namespace NUMINAMATH_CALUDE_solve_nested_equation_l279_27910

theorem solve_nested_equation : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 58 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_nested_equation_l279_27910


namespace NUMINAMATH_CALUDE_combined_tax_rate_l279_27917

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate : ℚ) (mindy_rate : ℚ) (income_ratio : ℚ) : 
  mork_rate = 40 / 100 →
  mindy_rate = 30 / 100 →
  income_ratio = 3 →
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 325 / 1000 := by
  sorry

#eval (40 / 100 + 30 / 100 * 3) / (1 + 3)

end NUMINAMATH_CALUDE_combined_tax_rate_l279_27917


namespace NUMINAMATH_CALUDE_probability_three_consecutive_beliy_naliv_l279_27956

/-- The probability of selecting 3 "Beliy Naliv" bushes consecutively -/
def probability_three_consecutive (beliy_naliv : ℕ) (verlioka : ℕ) : ℚ :=
  (beliy_naliv / (beliy_naliv + verlioka)) *
  ((beliy_naliv - 1) / (beliy_naliv + verlioka - 1)) *
  ((beliy_naliv - 2) / (beliy_naliv + verlioka - 2))

/-- Theorem stating the probability of selecting 3 "Beliy Naliv" bushes consecutively is 1/8 -/
theorem probability_three_consecutive_beliy_naliv :
  probability_three_consecutive 9 7 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_consecutive_beliy_naliv_l279_27956


namespace NUMINAMATH_CALUDE_sum_equation_solution_l279_27933

theorem sum_equation_solution (x : ℤ) : 
  (1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_solution_l279_27933


namespace NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l279_27939

theorem cubic_root_sum_inverse_squares : 
  ∀ (a b c : ℝ), 
  (a^3 - 6*a^2 - a + 3 = 0) → 
  (b^3 - 6*b^2 - b + 3 = 0) → 
  (c^3 - 6*c^2 - c + 3 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 37/9) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l279_27939


namespace NUMINAMATH_CALUDE_prime_8p_plus_1_square_cube_l279_27970

theorem prime_8p_plus_1_square_cube (p : ℕ) : 
  Prime p → 
  ((∃ n : ℕ, 8 * p + 1 = n^2) ↔ p = 3) ∧ 
  (¬∃ n : ℕ, 8 * p + 1 = n^3) := by
sorry

end NUMINAMATH_CALUDE_prime_8p_plus_1_square_cube_l279_27970


namespace NUMINAMATH_CALUDE_solution_set_inequality_l279_27915

theorem solution_set_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l279_27915


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l279_27981

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_problem (a : ℕ → ℝ) 
    (h_seq : ArithGeomSeq a)
    (h_first : a 1 = 3)
    (h_sum : a 1 + a 3 + a 5 = 21) :
    a 2 * a 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l279_27981


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l279_27960

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48)
  (h2 : b + d = 6) : 
  a + c = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l279_27960


namespace NUMINAMATH_CALUDE_three_common_tangents_implies_a_equals_9_l279_27902

/-- Circle M with equation x^2 + y^2 - 4x + 3 = 0 -/
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- Another circle with equation x^2 + y^2 - 4x - 6y + a = 0 -/
def other_circle (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + a = 0

/-- Theorem stating that if circle M has exactly three common tangent lines
    with the other circle, then a = 9 -/
theorem three_common_tangents_implies_a_equals_9 :
  ∀ a : ℝ, (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, circle_M x y → (l₁ x y ∨ l₂ x y ∨ l₃ x y)) ∧
    (∀ x y, other_circle x y a → (l₁ x y ∨ l₂ x y ∨ l₃ x y))) →
  a = 9 :=
sorry

end NUMINAMATH_CALUDE_three_common_tangents_implies_a_equals_9_l279_27902


namespace NUMINAMATH_CALUDE_factorization_validity_l279_27972

theorem factorization_validity (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

#check factorization_validity

end NUMINAMATH_CALUDE_factorization_validity_l279_27972


namespace NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_proof_l279_27991

/-- The present age of person B given the conditions -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
    (a = b + 4) →              -- A is now 4 years older than B
    b = 34                     -- B's current age is 34

-- The proof is omitted
theorem present_age_of_B_proof : ∃ a b, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_proof_l279_27991


namespace NUMINAMATH_CALUDE_total_amount_proof_l279_27951

/-- The total amount of money shared by Debby, Maggie, and Alex -/
def total : ℝ := 22500

/-- Debby's share percentage -/
def debby_share : ℝ := 0.30

/-- Maggie's share percentage -/
def maggie_share : ℝ := 0.40

/-- Alex's share percentage -/
def alex_share : ℝ := 0.30

/-- Maggie's actual share amount -/
def maggie_amount : ℝ := 9000

theorem total_amount_proof :
  maggie_share * total = maggie_amount ∧
  debby_share + maggie_share + alex_share = 1 :=
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l279_27951


namespace NUMINAMATH_CALUDE_gold_cube_buying_price_l279_27929

/-- Proves that the buying price per gram of gold is $60 given the specified conditions --/
theorem gold_cube_buying_price (side_length : ℝ) (density : ℝ) (selling_factor : ℝ) (profit : ℝ) :
  side_length = 6 →
  density = 19 →
  selling_factor = 1.5 →
  profit = 123120 →
  let volume := side_length ^ 3
  let mass := density * volume
  let buying_price := profit / (selling_factor * mass - mass)
  buying_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_gold_cube_buying_price_l279_27929


namespace NUMINAMATH_CALUDE_total_ribbons_used_l279_27985

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used : 
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by sorry

end NUMINAMATH_CALUDE_total_ribbons_used_l279_27985


namespace NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l279_27927

theorem smallest_divisor_after_subtraction (n m k : ℕ) (h1 : n = 899830) (h2 : m = 6) (h3 : k = 8) :
  k > m ∧
  (n - m) % k = 0 ∧
  ∀ d : ℕ, m < d ∧ d < k → (n - m) % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l279_27927


namespace NUMINAMATH_CALUDE_roots_eighth_power_sum_l279_27913

theorem roots_eighth_power_sum (x y : ℝ) : 
  x^2 - 2*x*Real.sqrt 2 + 1 = 0 ∧ 
  y^2 - 2*y*Real.sqrt 2 + 1 = 0 ∧ 
  x ≠ y → 
  x^8 + y^8 = 1154 := by sorry

end NUMINAMATH_CALUDE_roots_eighth_power_sum_l279_27913


namespace NUMINAMATH_CALUDE_exists_m_divides_sum_powers_l279_27984

theorem exists_m_divides_sum_powers (n : ℕ+) :
  ∃ m : ℕ+, (7^n.val : ℤ) ∣ (3^m.val + 5^m.val - 1) := by sorry

end NUMINAMATH_CALUDE_exists_m_divides_sum_powers_l279_27984


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_t_in_range_l279_27968

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 4*x - 3 * Real.log x

def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                               (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

theorem f_not_monotonic_iff_t_in_range (t : ℝ) :
  not_monotonic f t (t + 1) ↔ t ∈ Ioo 0 1 ∪ Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_t_in_range_l279_27968


namespace NUMINAMATH_CALUDE_scientific_notation_correct_scientific_notation_format_l279_27988

/-- Represents the value in billions -/
def billion_value : ℝ := 57.44

/-- Represents the coefficient in scientific notation -/
def scientific_coefficient : ℝ := 5.744

/-- Represents the exponent in scientific notation -/
def scientific_exponent : ℤ := 9

/-- Asserts that the scientific notation is correct for the given value -/
theorem scientific_notation_correct :
  billion_value * 10^9 = scientific_coefficient * 10^scientific_exponent :=
by sorry

/-- Asserts that the coefficient in scientific notation is between 1 and 10 -/
theorem scientific_notation_format :
  1 ≤ scientific_coefficient ∧ scientific_coefficient < 10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_scientific_notation_format_l279_27988


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l279_27916

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 11*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  k + m = 52 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l279_27916


namespace NUMINAMATH_CALUDE_colonization_ways_l279_27926

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available units for colonization -/
def total_units : ℕ := 14

/-- Theorem stating that there are exactly 20 different ways to occupy the planets -/
theorem colonization_ways : 
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ earth_like_planets ∧ 
    p.2 ≤ mars_like_planets ∧ 
    p.1 * earth_like_units + p.2 * mars_like_units = total_units)
  (Finset.product (Finset.range (earth_like_planets + 1)) (Finset.range (mars_like_planets + 1)))).card = 20 :=
sorry

end NUMINAMATH_CALUDE_colonization_ways_l279_27926


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l279_27945

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10
  normalize : coefficient * (10 : ℝ) ^ exponent = number

/-- The number we want to represent in scientific notation -/
def number : ℕ := 37600

/-- The scientific notation of the number -/
def scientific_notation : ScientificNotation where
  coefficient := 3.76
  exponent := 4
  coeff_range := by sorry
  normalize := by sorry

/-- Theorem stating that the given scientific notation is correct for the number -/
theorem correct_scientific_notation :
  scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent = number := by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l279_27945


namespace NUMINAMATH_CALUDE_evolute_parabola_evolute_ellipse_l279_27908

-- Part 1: Parabola
theorem evolute_parabola (x y X Y : ℝ) :
  x^2 = 2 * (1 - y) →
  27 * X^2 = -8 * Y^3 :=
sorry

-- Part 2: Ellipse
theorem evolute_ellipse (a b c t X Y : ℝ) :
  c^2 = a^2 - b^2 →
  X = -(c^2 / a) * (Real.cos t)^3 ∧
  Y = -(c^2 / b) * (Real.sin t)^3 :=
sorry

end NUMINAMATH_CALUDE_evolute_parabola_evolute_ellipse_l279_27908


namespace NUMINAMATH_CALUDE_paper_tray_height_l279_27964

/-- The height of a paper tray formed from a square sheet -/
theorem paper_tray_height (side_length : ℝ) (cut_start : ℝ) : 
  side_length = 120 →
  cut_start = Real.sqrt 20 →
  2 * Real.sqrt 5 = 
    (Real.sqrt 2 * cut_start) / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l279_27964


namespace NUMINAMATH_CALUDE_sum_sequence_equality_l279_27914

theorem sum_sequence_equality (M : ℤ) : 
  1499 + 1497 + 1495 + 1493 + 1491 = 7500 - M → M = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_sequence_equality_l279_27914


namespace NUMINAMATH_CALUDE_periodic_decimal_difference_l279_27976

theorem periodic_decimal_difference : (4 : ℚ) / 11 - (9 : ℚ) / 25 = (1 : ℚ) / 275 := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_difference_l279_27976


namespace NUMINAMATH_CALUDE_vessel_volume_ratio_l279_27999

theorem vessel_volume_ratio : 
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_vessel_volume_ratio_l279_27999


namespace NUMINAMATH_CALUDE_base_n_representation_l279_27971

theorem base_n_representation (n : ℕ) (a b : ℤ) : 
  n > 8 → 
  n^2 - a*n + b = 0 → 
  a = n + 8 → 
  b = 8*n := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_l279_27971


namespace NUMINAMATH_CALUDE_sin_C_value_side_lengths_l279_27998

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 13 ∧ Real.cos t.A = 5/13

-- Theorem 1
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (ha : t.a = 36) :
  Real.sin t.C = 1/3 := by
  sorry

-- Theorem 2
theorem side_lengths (t : Triangle) (h : triangle_conditions t) (harea : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 4 * Real.sqrt 10 ∧ t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_side_lengths_l279_27998


namespace NUMINAMATH_CALUDE_circle_O_diameter_l279_27993

/-- The circle O with equation x^2 + y^2 - 2x + my - 4 = 0 -/
def circle_O (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

/-- The line with equation 2x + y = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

/-- Two points are symmetric about a line if the line is the perpendicular bisector of the segment connecting the points -/
def symmetric_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (midpoint : ℝ × ℝ), 
    line midpoint.1 midpoint.2 ∧ 
    (midpoint.1 = (M.1 + N.1) / 2) ∧ 
    (midpoint.2 = (M.2 + N.2) / 2) ∧
    ((N.1 - M.1) * 2 + (N.2 - M.2) = 0)

theorem circle_O_diameter : 
  ∃ (m : ℝ) (M N : ℝ × ℝ),
    circle_O m M.1 M.2 ∧ 
    circle_O m N.1 N.2 ∧ 
    symmetric_points M N symmetry_line →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = (1, -m/2) ∧ 
      radius = 3 ∧ 
      2 * radius = 6 :=
sorry

end NUMINAMATH_CALUDE_circle_O_diameter_l279_27993


namespace NUMINAMATH_CALUDE_fraction_doubled_l279_27907

theorem fraction_doubled (a b : ℝ) (h : a ≠ b) : 
  (2*a * 2*b) / (2*a - 2*b) = 2 * (a * b / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_doubled_l279_27907


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l279_27943

theorem vehicle_value_last_year 
  (value_this_year : ℝ)
  (ratio : ℝ)
  (h1 : value_this_year = 16000)
  (h2 : ratio = 0.8)
  (h3 : value_this_year = ratio * value_last_year) :
  value_last_year = 20000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l279_27943


namespace NUMINAMATH_CALUDE_sqrt_division_property_l279_27986

theorem sqrt_division_property (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x / Real.sqrt x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_property_l279_27986


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_l279_27931

theorem complex_fourth_quadrant (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 ↔ -6 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_l279_27931


namespace NUMINAMATH_CALUDE_cake_slices_kept_l279_27906

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (kept_slices : ℕ) : 
  total_slices = 12 →
  eaten_fraction = 1/4 →
  kept_slices = total_slices - (total_slices * (eaten_fraction.num / eaten_fraction.den).toNat) →
  kept_slices = 9 := by
  sorry

end NUMINAMATH_CALUDE_cake_slices_kept_l279_27906


namespace NUMINAMATH_CALUDE_large_doll_price_correct_l279_27947

def total_spending : ℝ := 350
def price_difference : ℝ := 2
def extra_dolls : ℕ := 20

def large_doll_price : ℝ := 7
def small_doll_price : ℝ := large_doll_price - price_difference

theorem large_doll_price_correct :
  (total_spending / small_doll_price = total_spending / large_doll_price + extra_dolls) ∧
  (large_doll_price > 0) ∧
  (small_doll_price > 0) := by
  sorry

end NUMINAMATH_CALUDE_large_doll_price_correct_l279_27947


namespace NUMINAMATH_CALUDE_g_sum_property_l279_27952

def g (x : ℝ) : ℝ := 3 * x^6 + 5 * x^4 - 6 * x^2 + 7

theorem g_sum_property : g 2 + g (-2) = 8 :=
by
  have h1 : g 2 = 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l279_27952


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l279_27992

/-- The price per pound of chicken given the conditions of Alice's grocery shopping --/
theorem chicken_price_per_pound (min_spend : ℝ) (amount_needed : ℝ)
  (chicken_weight : ℝ) (lettuce_price : ℝ) (tomatoes_price : ℝ)
  (sweet_potato_price : ℝ) (sweet_potato_count : ℕ)
  (broccoli_price : ℝ) (broccoli_count : ℕ)
  (brussels_sprouts_price : ℝ) :
  min_spend = 35 →
  amount_needed = 11 →
  chicken_weight = 1.5 →
  lettuce_price = 3 →
  tomatoes_price = 2.5 →
  sweet_potato_price = 0.75 →
  sweet_potato_count = 4 →
  broccoli_price = 2 →
  broccoli_count = 2 →
  brussels_sprouts_price = 2.5 →
  (min_spend - amount_needed - (lettuce_price + tomatoes_price +
    sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
    brussels_sprouts_price)) / chicken_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l279_27992


namespace NUMINAMATH_CALUDE_complex_number_forms_l279_27934

theorem complex_number_forms (z : ℂ) : 
  z = 4 * (Complex.cos (4 * Real.pi / 3) + Complex.I * Complex.sin (4 * Real.pi / 3)) →
  (z = -2 - 2 * Complex.I * Real.sqrt 3) ∧ 
  (z = 4 * Complex.exp (Complex.I * (4 * Real.pi / 3))) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_forms_l279_27934


namespace NUMINAMATH_CALUDE_circle_radius_is_5_l279_27942

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def TangentLine (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State the theorem
theorem circle_radius_is_5 :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C passes through point A
    A ∈ Circle center radius ∧
    -- Circle C is tangent to the line 3x-4y+7=0 at point B
    B ∈ Circle center radius ∧
    TangentLine B.1 B.2 ∧
    -- The radius of circle C is 5
    radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_5_l279_27942


namespace NUMINAMATH_CALUDE_special_pair_example_special_pair_with_three_special_pair_negation_l279_27901

/-- Definition of a special rational number pair -/
def is_special_pair (a b : ℚ) : Prop := a + b = a * b - 1

/-- Theorem 1: (5, 3/2) is a special rational number pair -/
theorem special_pair_example : is_special_pair 5 (3/2) := by sorry

/-- Theorem 2: If (a, 3) is a special rational number pair, then a = 2 -/
theorem special_pair_with_three (a : ℚ) : is_special_pair a 3 → a = 2 := by sorry

/-- Theorem 3: If (m, n) is a special rational number pair, then (-n, -m) is not a special rational number pair -/
theorem special_pair_negation (m n : ℚ) : is_special_pair m n → ¬ is_special_pair (-n) (-m) := by sorry

end NUMINAMATH_CALUDE_special_pair_example_special_pair_with_three_special_pair_negation_l279_27901


namespace NUMINAMATH_CALUDE_dog_weights_l279_27919

theorem dog_weights (y z : ℝ) : 
  let dog_weights : List ℝ := [25, 31, 35, 33, y, z]
  (dog_weights.take 4).sum / 4 = dog_weights.sum / 6 →
  y + z = 62 := by
sorry

end NUMINAMATH_CALUDE_dog_weights_l279_27919


namespace NUMINAMATH_CALUDE_max_demand_decrease_l279_27994

theorem max_demand_decrease (price_increase : ℝ) (revenue_increase : ℝ) : 
  price_increase = 0.20 →
  revenue_increase = 0.10 →
  (1 + price_increase) * (1 - (1 / 12 : ℝ)) ≥ 1 + revenue_increase :=
by sorry

end NUMINAMATH_CALUDE_max_demand_decrease_l279_27994


namespace NUMINAMATH_CALUDE_darryl_has_twenty_books_l279_27944

/-- Represents the number of books each friend has -/
structure BookCount where
  darryl : ℕ
  lamont : ℕ
  loris : ℕ

/-- The conditions of the problem -/
def BookProblem (bc : BookCount) : Prop :=
  bc.lamont = 2 * bc.darryl ∧
  bc.loris = bc.lamont - 3 ∧
  bc.darryl + bc.lamont + bc.loris = 97

/-- The theorem stating that Darryl has 20 books -/
theorem darryl_has_twenty_books :
  ∃ (bc : BookCount), BookProblem bc ∧ bc.darryl = 20 := by
  sorry

end NUMINAMATH_CALUDE_darryl_has_twenty_books_l279_27944


namespace NUMINAMATH_CALUDE_cubic_square_inequality_l279_27979

theorem cubic_square_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / (a^2 + b^2) ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_inequality_l279_27979


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l279_27911

def f (x : ℝ) : ℝ := 2 * x^3

theorem tangent_slope_at_point_one (x : ℝ) :
  HasDerivAt f 6 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l279_27911


namespace NUMINAMATH_CALUDE_lina_sticker_collection_l279_27937

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmeticSequenceSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lina's sticker collection problem -/
theorem lina_sticker_collection :
  arithmeticSequenceSum 3 2 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lina_sticker_collection_l279_27937


namespace NUMINAMATH_CALUDE_work_completion_proof_l279_27930

/-- The number of days it takes for a and b to finish the work together -/
def combined_days : ℝ := 30

/-- The number of days it takes for a to finish the work alone -/
def a_alone_days : ℝ := 60

/-- The number of days a worked alone after b left -/
def a_remaining_days : ℝ := 20

/-- The number of days a and b worked together before b left -/
def days_worked_together : ℝ := 20

theorem work_completion_proof :
  days_worked_together * (1 / combined_days) + a_remaining_days * (1 / a_alone_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l279_27930


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l279_27948

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (1 - Complex.I) * (2 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l279_27948


namespace NUMINAMATH_CALUDE_proposition_equivalence_l279_27922

theorem proposition_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l279_27922


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l279_27950

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3*m + 2)*(-1) - (2*m - 1)*1 + 5*m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l279_27950


namespace NUMINAMATH_CALUDE_trig_expression_equality_l279_27924

theorem trig_expression_equality : 
  1 / Real.sin (40 * π / 180) - Real.sqrt 2 / Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l279_27924


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_l279_27903

/-- A cubic polynomial with real coefficients and three non-zero real roots satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c. -/
theorem cubic_polynomial_inequality (a b c : ℝ) (P : ℝ → ℝ) (h1 : P = fun x ↦ x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ P x = 0 ∧ P y = 0 ∧ P z = 0) :
  6 * a^3 + 10 * (a^2 - 2*b)^(3/2) - 12 * a * b ≥ 27 * c := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_l279_27903


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l279_27955

theorem subtraction_of_decimals : 3.57 - 1.14 - 0.23 = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l279_27955


namespace NUMINAMATH_CALUDE_notebook_cost_proof_l279_27978

theorem notebook_cost_proof :
  ∀ (s c n : ℕ),
    s ≤ 36 →                     -- number of students who bought notebooks
    s > 36 / 2 →                 -- at least half of the students
    n > 2 →                      -- more than 2 notebooks per student
    c > n →                      -- cost in cents greater than number of notebooks
    s * c * n = 3969 →           -- total cost in cents
    c = 27 :=                    -- cost per notebook is 27 cents
by sorry

end NUMINAMATH_CALUDE_notebook_cost_proof_l279_27978


namespace NUMINAMATH_CALUDE_sum_of_special_system_l279_27949

theorem sum_of_special_system (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (a + c)) :
  a + b + c = 1128 / 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_system_l279_27949


namespace NUMINAMATH_CALUDE_jose_join_time_l279_27967

/-- Proves that Jose joined 2 months after Tom opened the shop given the investment and profit information -/
theorem jose_join_time (tom_investment : ℕ) (jose_investment : ℕ) (total_profit : ℕ) (jose_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  total_profit = 36000 →
  jose_profit = 20000 →
  ∃ x : ℕ, 
    (tom_investment * 12) / (jose_investment * (12 - x)) = (total_profit - jose_profit) / jose_profit ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_jose_join_time_l279_27967


namespace NUMINAMATH_CALUDE_largest_common_number_l279_27923

def is_in_first_sequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def is_in_second_sequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

def is_in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 150

theorem largest_common_number :
  (is_in_first_sequence 131) ∧
  (is_in_second_sequence 131) ∧
  (is_in_range 131) ∧
  (∀ y : ℕ, y > 131 →
    ¬(is_in_first_sequence y ∧ is_in_second_sequence y ∧ is_in_range y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l279_27923


namespace NUMINAMATH_CALUDE_stones_division_impossible_l279_27954

theorem stones_division_impossible (stones : List Nat) : 
  stones.length = 31 ∧ stones.sum = 660 → 
  ∃ (a b : Nat), a ∈ stones ∧ b ∈ stones ∧ a > 2 * b :=
by sorry

end NUMINAMATH_CALUDE_stones_division_impossible_l279_27954


namespace NUMINAMATH_CALUDE_bills_divisible_by_101_l279_27973

theorem bills_divisible_by_101 
  (a b : ℕ) 
  (h_not_cong : a % 101 ≠ b % 101) 
  (h_total : ℕ) 
  (h_total_eq : h_total = 100) :
  ∃ (subset : Finset ℕ), subset.card ≤ h_total ∧ 
    (∃ (k₁ k₂ : ℕ), k₁ + k₂ = subset.card ∧ (k₁ * a + k₂ * b) % 101 = 0) :=
sorry

end NUMINAMATH_CALUDE_bills_divisible_by_101_l279_27973


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l279_27925

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) :
  x / y = 37 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l279_27925


namespace NUMINAMATH_CALUDE_window_screen_sales_l279_27912

/-- Represents the monthly sales of window screens -/
structure MonthlySales where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ

/-- Calculates the total sales from January to April -/
def totalSales (sales : MonthlySales) : ℕ :=
  sales.january + sales.february + sales.march + sales.april

/-- Theorem stating the total sales given the conditions -/
theorem window_screen_sales : ∃ (sales : MonthlySales),
  sales.february = 2 * sales.january ∧
  sales.march = (5 / 4 : ℚ) * sales.february ∧
  sales.april = (9 / 10 : ℚ) * sales.march ∧
  sales.march = 12100 ∧
  totalSales sales = 37510 := by
  sorry


end NUMINAMATH_CALUDE_window_screen_sales_l279_27912


namespace NUMINAMATH_CALUDE_max_abs_z_is_one_l279_27905

theorem max_abs_z_is_one (a b c d z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs c = Complex.abs d)
  (h4 : Complex.abs a > 0)
  (h5 : a * z^3 + b * z^2 + c * z + d = 0) :
  Complex.abs z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_is_one_l279_27905


namespace NUMINAMATH_CALUDE_waynes_blocks_l279_27966

theorem waynes_blocks (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_waynes_blocks_l279_27966
