import Mathlib

namespace NUMINAMATH_CALUDE_fraction_multiplication_simplification_l2281_228162

theorem fraction_multiplication_simplification :
  (270 : ℚ) / 18 * (7 : ℚ) / 210 * (12 : ℚ) / 4 = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_simplification_l2281_228162


namespace NUMINAMATH_CALUDE_smallest_with_2023_divisors_l2281_228131

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n can be written as m * 6^k where 6 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * 6^k ∧ ¬(6 ∣ m)

theorem smallest_with_2023_divisors :
  ∃ (m k : ℕ),
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    has_form (m * 6^k) m k ∧
    num_divisors (m * 6^k) = 2023 ∧
    m + k = 59055 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_2023_divisors_l2281_228131


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2281_228100

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (a : ℕ), 5 * n = a ^ 2) ∧ 
   (∃ (b : ℕ), 3 * n = b ^ 3)) → 
  (∀ (m : ℕ), m > 0 → 
   (∃ (a : ℕ), 5 * m = a ^ 2) → 
   (∃ (b : ℕ), 3 * m = b ^ 3) → 
   m ≥ 1125) ∧
  (∃ (a b : ℕ), 5 * 1125 = a ^ 2 ∧ 3 * 1125 = b ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2281_228100


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2281_228166

theorem complex_equation_solution (z : ℂ) : (z * Complex.I = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2281_228166


namespace NUMINAMATH_CALUDE_julia_miles_driven_l2281_228173

theorem julia_miles_driven (darius_miles julia_miles total_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) : 
  julia_miles = 998 := by
  sorry

end NUMINAMATH_CALUDE_julia_miles_driven_l2281_228173


namespace NUMINAMATH_CALUDE_heath_planting_time_l2281_228171

/-- The number of hours Heath spent planting carrots -/
def planting_time (rows : ℕ) (plants_per_row : ℕ) (plants_per_hour : ℕ) : ℕ :=
  (rows * plants_per_row) / plants_per_hour

/-- Theorem stating that Heath spent 20 hours planting carrots -/
theorem heath_planting_time :
  planting_time 400 300 6000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_heath_planting_time_l2281_228171


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_eq_twice_side_l2281_228125

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- A point inside a square -/
structure PointInSquare (s : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 ≤ x ∧ x ≤ s
  y_bound : 0 ≤ y ∧ y ≤ s

/-- The sum of perpendiculars from a point to the sides of a square -/
def sumOfPerpendiculars (s : ℝ) (p : PointInSquare s) : ℝ :=
  p.x + (s - p.x) + p.y + (s - p.y)

/-- Theorem: The sum of perpendiculars from any point inside a square to its sides
    is equal to twice the side length of the square -/
theorem sum_of_perpendiculars_eq_twice_side {s : ℝ} (sq : Square s) (p : PointInSquare s) :
  sumOfPerpendiculars s p = 2 * s := by
  sorry


end NUMINAMATH_CALUDE_sum_of_perpendiculars_eq_twice_side_l2281_228125


namespace NUMINAMATH_CALUDE_inequality_solution_upper_bound_l2281_228101

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem inequality_solution (x : ℝ) : f x < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

-- Part II
theorem upper_bound (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x ≤ 5/6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_upper_bound_l2281_228101


namespace NUMINAMATH_CALUDE_johnny_earnings_l2281_228130

def calculate_earnings (hourly_wage : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) 
  (overtime_rate : ℝ) (tax_rate : ℝ) (insurance_rate : ℝ) : ℝ :=
  let regular_pay := hourly_wage * regular_hours
  let overtime_pay := hourly_wage * overtime_rate * overtime_hours
  let total_earnings := regular_pay + overtime_pay
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  total_earnings - tax_deduction - insurance_deduction

theorem johnny_earnings :
  let hourly_wage : ℝ := 8.25
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := 7
  let overtime_rate : ℝ := 1.5
  let tax_rate : ℝ := 0.08
  let insurance_rate : ℝ := 0.05
  abs (calculate_earnings hourly_wage regular_hours overtime_hours overtime_rate tax_rate insurance_rate - 362.47) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_johnny_earnings_l2281_228130


namespace NUMINAMATH_CALUDE_shift_hours_is_eight_l2281_228133

/-- Calculates the number of hours in each person's shift given the following conditions:
  * 20 people are hired
  * Each person makes on average 20 shirts per day
  * Employees are paid $12 an hour plus $5 per shirt
  * Shirts are sold for $35 each
  * Nonemployee expenses are $1000 a day
  * The company makes $9080 in profits per day
-/
def calculateShiftHours (
  numEmployees : ℕ)
  (shirtsPerPerson : ℕ)
  (hourlyWage : ℕ)
  (perShirtBonus : ℕ)
  (shirtPrice : ℕ)
  (nonEmployeeExpenses : ℕ)
  (dailyProfit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the shift hours calculated by the function is 8 -/
theorem shift_hours_is_eight :
  calculateShiftHours 20 20 12 5 35 1000 9080 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shift_hours_is_eight_l2281_228133


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2281_228187

theorem binomial_coefficient_equality (n : ℕ) (r : ℕ) : 
  (Nat.choose n (4*r - 1) = Nat.choose n (r + 1)) → 
  (n = 20 ∧ r = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2281_228187


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2281_228156

theorem cubic_root_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
   x^3 - 8*x^2 + a*x - b = 0 ∧
   y^3 - 8*y^2 + a*y - b = 0 ∧
   z^3 - 8*z^2 + a*z - b = 0) →
  a + b = 27 ∨ a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2281_228156


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l2281_228140

def original_budget : ℚ := 940
def new_budget : ℚ := 752

theorem magazine_budget_cut_percentage : 
  (original_budget - new_budget) / original_budget * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l2281_228140


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l2281_228192

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l2281_228192


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l2281_228185

theorem aunt_gift_amount (jade_initial julia_initial jack_initial total_after_gift : ℕ) : 
  jade_initial = 38 →
  julia_initial = jade_initial / 2 →
  jack_initial = 12 →
  total_after_gift = 132 →
  ∃ gift : ℕ, 
    jade_initial + julia_initial + jack_initial + 3 * gift = total_after_gift ∧
    gift = 21 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l2281_228185


namespace NUMINAMATH_CALUDE_equation_solution_l2281_228105

def solution_set : Set (ℤ × ℤ) := {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)}

theorem equation_solution (x y : ℤ) : 
  (x + y ≠ 0) → ((x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ solution_set) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2281_228105


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_legs_l2281_228135

theorem right_triangle_consecutive_even_legs (a b c : ℕ) : 
  -- a and b are the legs, c is the hypotenuse
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) →  -- consecutive even numbers
  (c = 34) →  -- hypotenuse is 34
  (a + b = 46) :=  -- sum of legs is 46
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_legs_l2281_228135


namespace NUMINAMATH_CALUDE_prob_odd_sum_given_even_product_l2281_228103

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 3 / 8

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 5 / 8

/-- The number of ways to get an odd sum with one odd die -/
def odd_sum_one_odd : ℕ := num_dice * 3 * 5^4

/-- The number of ways to get an odd sum with three odd dice -/
def odd_sum_three_odd : ℕ := (num_dice.choose 3) * 3^3 * 5^2

/-- The number of ways to get an odd sum with all odd dice -/
def odd_sum_all_odd : ℕ := 3^5

/-- The total number of favorable outcomes (odd sum) -/
def total_favorable : ℕ := odd_sum_one_odd + odd_sum_three_odd + odd_sum_all_odd

/-- The total number of possible outcomes where the product is even -/
def total_possible : ℕ := 8^5 - (3/8)^5 * 8^5

/-- The probability of getting an odd sum given that the product is even -/
theorem prob_odd_sum_given_even_product :
  (total_favorable : ℚ) / total_possible =
  (5 * 3 * 5^4 + 10 * 27 * 25 + 243) / (8^5 - (3/8)^5 * 8^5) :=
by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_given_even_product_l2281_228103


namespace NUMINAMATH_CALUDE_davids_age_l2281_228167

/-- Given that Yuan is 14 years old and twice David's age, prove that David is 7 years old. -/
theorem davids_age (yuan_age : ℕ) (david_age : ℕ) 
  (h1 : yuan_age = 14) 
  (h2 : yuan_age = 2 * david_age) : 
  david_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_davids_age_l2281_228167


namespace NUMINAMATH_CALUDE_ellipse_right_angle_triangle_area_l2281_228165

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {f : ℝ × ℝ | ∃ (x y : ℝ), f = (x, y) ∧ x^2 + y^2 = 1}

/-- Angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given by three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_angle_triangle_area 
  (p : ℝ × ℝ) (f₁ f₂ : ℝ × ℝ) 
  (h_p : p ∈ Ellipse) 
  (h_f : f₁ ∈ Foci ∧ f₂ ∈ Foci ∧ f₁ ≠ f₂) 
  (h_angle : angle (f₁.1 - p.1, f₁.2 - p.2) (f₂.1 - p.1, f₂.2 - p.2) = π / 2) :
  triangleArea f₁ p f₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_right_angle_triangle_area_l2281_228165


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2281_228128

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2281_228128


namespace NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_axes_l2281_228189

/-- A circle whose center lies on a parabola and is tangent to the parabola's axis and y-axis -/
theorem circle_on_parabola_tangent_to_axes :
  ∃ (x₀ y₀ r : ℝ),
    (x₀ < 0) ∧                             -- Center is on the left side of y-axis
    (y₀ = (1/2) * x₀^2) ∧                  -- Center lies on the parabola
    (∀ x y : ℝ,
      (x + 1)^2 + (y - 1/2)^2 = 1 ↔        -- Equation of the circle
      (x - x₀)^2 + (y - y₀)^2 = r^2) ∧     -- Standard form of circle equation
    (r = |x₀|) ∧                           -- Circle is tangent to y-axis
    (r = |y₀ - 1/2|)                       -- Circle is tangent to parabola's axis
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_axes_l2281_228189


namespace NUMINAMATH_CALUDE_police_coverage_l2281_228106

-- Define the type for intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the type for streets
inductive Street : Type
| ABCD | EFG | HIJK    -- Horizontal streets
| AEH | BFI | DGJ      -- Vertical streets
| HFC | CGK           -- Diagonal streets

-- Define a function to check if an intersection is on a street
def isOnStreet (i : Intersection) (s : Street) : Prop :=
  match s with
  | Street.ABCD => i = Intersection.A ∨ i = Intersection.B ∨ i = Intersection.C ∨ i = Intersection.D
  | Street.EFG => i = Intersection.E ∨ i = Intersection.F ∨ i = Intersection.G
  | Street.HIJK => i = Intersection.H ∨ i = Intersection.I ∨ i = Intersection.J ∨ i = Intersection.K
  | Street.AEH => i = Intersection.A ∨ i = Intersection.E ∨ i = Intersection.H
  | Street.BFI => i = Intersection.B ∨ i = Intersection.F ∨ i = Intersection.I
  | Street.DGJ => i = Intersection.D ∨ i = Intersection.G ∨ i = Intersection.J
  | Street.HFC => i = Intersection.H ∨ i = Intersection.F ∨ i = Intersection.C
  | Street.CGK => i = Intersection.C ∨ i = Intersection.G ∨ i = Intersection.K

-- Define a function to check if a street is covered by a set of intersections
def isCovered (s : Street) (intersections : Set Intersection) : Prop :=
  ∃ i ∈ intersections, isOnStreet i s

-- Theorem statement
theorem police_coverage :
  let policemen : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}
  ∀ s : Street, isCovered s policemen :=
by sorry

end NUMINAMATH_CALUDE_police_coverage_l2281_228106


namespace NUMINAMATH_CALUDE_last_erased_numbers_l2281_228110

-- Define a function to count prime factors
def count_prime_factors (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem last_erased_numbers :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 →
    (count_prime_factors n = 6 ↔ n = 64 ∨ n = 96) ∧
    (count_prime_factors n ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_last_erased_numbers_l2281_228110


namespace NUMINAMATH_CALUDE_fish_tank_problem_l2281_228138

theorem fish_tank_problem (initial_fish caught_fish : ℕ) : 
  caught_fish = initial_fish - 4 →
  initial_fish + caught_fish = 20 →
  caught_fish = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l2281_228138


namespace NUMINAMATH_CALUDE_sets_are_equal_l2281_228127

def M : Set ℝ := {y | ∃ x, y = x^2 + 3}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 3)}

theorem sets_are_equal : M = N := by sorry

end NUMINAMATH_CALUDE_sets_are_equal_l2281_228127


namespace NUMINAMATH_CALUDE_square_of_95_l2281_228121

theorem square_of_95 : (95 : ℤ)^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_95_l2281_228121


namespace NUMINAMATH_CALUDE_triangle_angle_relation_minimum_l2281_228152

theorem triangle_angle_relation_minimum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hSum : A + B + C = π) (hTriangle : 3 * (Real.cos (2 * A) - Real.cos (2 * C)) = 1 - Real.cos (2 * B)) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (Real.sin C / (Real.sin A * Real.sin B) + Real.cos C / Real.sin C) ≥ y → 
    y ≥ 2 * Real.sqrt 7 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_minimum_l2281_228152


namespace NUMINAMATH_CALUDE_line_y_intercept_l2281_228104

/-- A line in the xy-plane is defined by its slope and a point it passes through. 
    This theorem proves that for a line with slope 2 passing through (498, 998), 
    the y-intercept is 2. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 2 ∧ x = 498 ∧ y = 998 ∧ y = m * x + b → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2281_228104


namespace NUMINAMATH_CALUDE_highest_probability_prime_l2281_228196

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor_of_12 (n : ℕ) : Prop := 12 % n = 0

def total_outcomes : ℕ := 36

def prime_outcomes : ℕ := 15
def multiple_of_4_outcomes : ℕ := 9
def perfect_square_outcomes : ℕ := 7
def score_7_outcomes : ℕ := 6
def factor_of_12_outcomes : ℕ := 12

theorem highest_probability_prime :
  prime_outcomes > multiple_of_4_outcomes ∧
  prime_outcomes > perfect_square_outcomes ∧
  prime_outcomes > score_7_outcomes ∧
  prime_outcomes > factor_of_12_outcomes :=
sorry

end NUMINAMATH_CALUDE_highest_probability_prime_l2281_228196


namespace NUMINAMATH_CALUDE_vector_properties_l2281_228146

/-- Given vectors in R², prove perpendicularity implies tan(α + β) = 2
    and tan(α)tan(β) = 16 implies vectors are parallel -/
theorem vector_properties (α β : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (ha : a = (4 * Real.cos α, Real.sin α))
  (hb : b = (Real.sin β, 4 * Real.cos β))
  (hc : c = (Real.cos β, -4 * Real.sin β)) :
  (a.1 * (b.1 - 2*c.1) + a.2 * (b.2 - 2*c.2) = 0 → Real.tan (α + β) = 2) ∧
  (Real.tan α * Real.tan β = 16 → ∃ (k : ℝ), a = k • b) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2281_228146


namespace NUMINAMATH_CALUDE_base_h_equation_solution_l2281_228183

/-- Represents a number in base h --/
def BaseH (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The theorem statement --/
theorem base_h_equation_solution :
  ∃ (h : Nat), h > 1 ∧ 
    BaseH [8, 3, 7, 4] h + BaseH [6, 9, 2, 5] h = BaseH [1, 5, 3, 0, 9] h ∧
    h = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_h_equation_solution_l2281_228183


namespace NUMINAMATH_CALUDE_root_implies_sum_l2281_228195

theorem root_implies_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2)^3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_root_implies_sum_l2281_228195


namespace NUMINAMATH_CALUDE_exists_term_with_100_nines_l2281_228151

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number contains 100 consecutive nines if it can be written in the form
    k * 10^(100 + m) + (10^100 - 1) for some natural numbers k and m. -/
def Contains100ConsecutiveNines (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = k * (10^(100 + m)) + (10^100 - 1)

/-- In any infinite arithmetic progression of natural numbers,
    there exists a term that contains 100 consecutive nines. -/
theorem exists_term_with_100_nines (a : ℕ → ℕ) (h : ArithmeticProgression a) :
  ∃ n : ℕ, Contains100ConsecutiveNines (a n) := by
  sorry


end NUMINAMATH_CALUDE_exists_term_with_100_nines_l2281_228151


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l2281_228175

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x > 1 ∨ x < -6}

theorem set_intersection_and_union (a : ℝ) :
  (A a ∩ B = ∅ → a ∈ Set.Icc (-6) (-2)) ∧
  (A a ∪ B = B → a ∈ Set.Ioi 1 ∪ Set.Iio (-9)) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l2281_228175


namespace NUMINAMATH_CALUDE_team_a_two_projects_probability_l2281_228109

/-- The number of ways to distribute n identical objects into k distinct boxes,
    where each box must contain at least one object. -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The probability of team A contracting exactly two projects out of five projects
    distributed among four teams, where each team must contract at least one project. -/
theorem team_a_two_projects_probability :
  let total_distributions := stars_and_bars 5 4
  let favorable_distributions := stars_and_bars 3 3
  (favorable_distributions : ℚ) / total_distributions = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_team_a_two_projects_probability_l2281_228109


namespace NUMINAMATH_CALUDE_sum_PV_squared_l2281_228144

-- Define the triangle PQR
def PQR : Set (ℝ × ℝ) := sorry

-- Define the property of PQR being equilateral with side length 10
def is_equilateral_10 (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the four triangles PU₁V₁, PU₁V₂, PU₂V₃, and PU₂V₄
def PU1V1 : Set (ℝ × ℝ) := sorry
def PU1V2 : Set (ℝ × ℝ) := sorry
def PU2V3 : Set (ℝ × ℝ) := sorry
def PU2V4 : Set (ℝ × ℝ) := sorry

-- Define the property of a triangle being congruent to PQR
def is_congruent_to_PQR (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property QU₁ = QU₂ = 3
def QU1_QU2_eq_3 : Prop := sorry

-- Define the function to calculate PVₖ
def PV (k : ℕ) : ℝ := sorry

-- Theorem statement
theorem sum_PV_squared :
  is_equilateral_10 PQR ∧
  is_congruent_to_PQR PU1V1 ∧
  is_congruent_to_PQR PU1V2 ∧
  is_congruent_to_PQR PU2V3 ∧
  is_congruent_to_PQR PU2V4 ∧
  QU1_QU2_eq_3 →
  (PV 1)^2 + (PV 2)^2 + (PV 3)^2 + (PV 4)^2 = 800 := by sorry

end NUMINAMATH_CALUDE_sum_PV_squared_l2281_228144


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l2281_228163

theorem complex_exp_conversion (z : ℂ) :
  z = Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) →
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l2281_228163


namespace NUMINAMATH_CALUDE_sandra_coffee_cups_l2281_228111

/-- Given that Sandra and Marcie took a total of 8 cups of coffee, 
    and Marcie took 2 cups, prove that Sandra took 6 cups of coffee. -/
theorem sandra_coffee_cups (total : ℕ) (marcie : ℕ) (sandra : ℕ) 
  (h1 : total = 8) 
  (h2 : marcie = 2) 
  (h3 : sandra + marcie = total) : 
  sandra = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandra_coffee_cups_l2281_228111


namespace NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l2281_228199

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians + non_technicians = total_workers)
  (h2 : technicians = non_technicians)
  (h3 : permanent_technicians = technicians / 2)
  (h4 : permanent_non_technicians = non_technicians / 2)
  : (total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l2281_228199


namespace NUMINAMATH_CALUDE_inverse_relation_values_l2281_228113

/-- Represents the constant product of two inversely related quantities -/
def k : ℝ := 800 * 0.5

/-- Represents the relationship between inversely related quantities a and b -/
def inverse_relation (a b : ℝ) : Prop := a * b = k

theorem inverse_relation_values (a₁ a₂ : ℝ) (h₁ : inverse_relation 800 0.5) :
  (inverse_relation 1600 0.250) ∧ (inverse_relation 400 1.000) := by
  sorry

#check inverse_relation_values

end NUMINAMATH_CALUDE_inverse_relation_values_l2281_228113


namespace NUMINAMATH_CALUDE_perimeter_stones_count_l2281_228147

/-- Given a square arrangement of stones with 5 stones on each side,
    the number of stones on the perimeter is 16. -/
theorem perimeter_stones_count (side_length : ℕ) (h : side_length = 5) :
  4 * side_length - 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_stones_count_l2281_228147


namespace NUMINAMATH_CALUDE_mikes_video_game_earnings_l2281_228179

theorem mikes_video_game_earnings :
  let total_games : ℕ := 20
  let non_working_games : ℕ := 11
  let price_per_game : ℚ := 8
  let sales_tax_rate : ℚ := 12 / 100
  
  let working_games : ℕ := total_games - non_working_games
  let total_revenue : ℚ := working_games * price_per_game
  
  total_revenue = 72 :=
by sorry

end NUMINAMATH_CALUDE_mikes_video_game_earnings_l2281_228179


namespace NUMINAMATH_CALUDE_f_symmetry_and_increase_l2281_228143

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

def is_center_of_symmetry (c : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_symmetry_and_increase :
  (∀ k : ℤ, is_center_of_symmetry (k * Real.pi / 2 + Real.pi / 12, -1) f) ∧
  is_increasing_on f 0 (Real.pi / 3) ∧
  is_increasing_on f (5 * Real.pi / 6) Real.pi :=
sorry

end NUMINAMATH_CALUDE_f_symmetry_and_increase_l2281_228143


namespace NUMINAMATH_CALUDE_longest_collection_pages_l2281_228193

/-- Represents the number of pages per inch for a book collection -/
structure PagesPerInch where
  value : ℕ

/-- Represents the height of a book collection in inches -/
structure CollectionHeight where
  value : ℕ

/-- Calculates the total number of pages in a collection -/
def total_pages (ppi : PagesPerInch) (height : CollectionHeight) : ℕ :=
  ppi.value * height.value

/-- Represents Miles's book collection -/
def miles_collection : PagesPerInch × CollectionHeight :=
  ({ value := 5 }, { value := 240 })

/-- Represents Daphne's book collection -/
def daphne_collection : PagesPerInch × CollectionHeight :=
  ({ value := 50 }, { value := 25 })

theorem longest_collection_pages : 
  max (total_pages miles_collection.1 miles_collection.2)
      (total_pages daphne_collection.1 daphne_collection.2) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_longest_collection_pages_l2281_228193


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2281_228161

theorem simplify_polynomial (x : ℝ) : (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) = x^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2281_228161


namespace NUMINAMATH_CALUDE_keychain_manufacturing_cost_l2281_228184

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (initial_cost : ℝ) -- Initial manufacturing cost
  (initial_profit_percentage : ℝ) -- Initial profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (h1 : initial_cost = 65) -- Initial cost is $65
  (h2 : P - initial_cost = initial_profit_percentage * P) -- Initial profit equation
  (h3 : initial_profit_percentage = 0.35) -- Initial profit is 35%
  (h4 : new_profit_percentage = 0.50) -- New profit is 50%
  : ∃ C, P - C = new_profit_percentage * P ∧ C = 50 := by
sorry

end NUMINAMATH_CALUDE_keychain_manufacturing_cost_l2281_228184


namespace NUMINAMATH_CALUDE_rational_expression_evaluation_l2281_228198

theorem rational_expression_evaluation :
  let x : ℝ := 7
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 2410 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_evaluation_l2281_228198


namespace NUMINAMATH_CALUDE_train_speed_proof_l2281_228168

/-- Proves that the new train speed is 256 km/h given the problem conditions -/
theorem train_speed_proof (distance : ℝ) (speed_multiplier : ℝ) (time_reduction : ℝ) 
  (h1 : distance = 1280)
  (h2 : speed_multiplier = 3.2)
  (h3 : time_reduction = 11)
  (h4 : ∀ x : ℝ, distance / x - distance / (speed_multiplier * x) = time_reduction) :
  speed_multiplier * (distance / (distance / speed_multiplier + time_reduction)) = 256 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l2281_228168


namespace NUMINAMATH_CALUDE_sin_cos_value_l2281_228158

theorem sin_cos_value (x : ℝ) : 
  let a : ℝ × ℝ := (4 * Real.sin x, 1 - Real.cos x)
  let b : ℝ × ℝ := (1, -2)
  (a.1 * b.1 + a.2 * b.2 = -2) → (Real.sin x * Real.cos x = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l2281_228158


namespace NUMINAMATH_CALUDE_jessica_bank_account_l2281_228142

theorem jessica_bank_account (initial_balance : ℝ) 
  (withdrawal : ℝ) (final_balance : ℝ) (deposit_fraction : ℝ) :
  withdrawal = 200 ∧
  initial_balance - withdrawal = (3/5) * initial_balance ∧
  final_balance = 360 ∧
  final_balance = (initial_balance - withdrawal) + deposit_fraction * (initial_balance - withdrawal) →
  deposit_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_account_l2281_228142


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2281_228107

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 53 ∧ ∀ (s : ℝ), -3 * s^2 + 24 * s + 5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2281_228107


namespace NUMINAMATH_CALUDE_monthly_subscription_more_cost_effective_l2281_228194

/-- Represents the cost of internet access plans -/
def internet_cost (pay_per_minute_rate : ℚ) (monthly_fee : ℚ) (communication_fee : ℚ) (hours : ℚ) : ℚ × ℚ :=
  let minutes : ℚ := hours * 60
  let pay_per_minute_cost : ℚ := (pay_per_minute_rate + communication_fee) * minutes
  let monthly_subscription_cost : ℚ := monthly_fee + communication_fee * minutes
  (pay_per_minute_cost, monthly_subscription_cost)

theorem monthly_subscription_more_cost_effective :
  let (pay_per_minute_cost, monthly_subscription_cost) :=
    internet_cost (5 / 100) 50 (2 / 100) 20
  monthly_subscription_cost < pay_per_minute_cost :=
by sorry

end NUMINAMATH_CALUDE_monthly_subscription_more_cost_effective_l2281_228194


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2281_228181

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2281_228181


namespace NUMINAMATH_CALUDE_persons_count_l2281_228141

/-- The total number of persons in the group --/
def n : ℕ := sorry

/-- The total amount spent by the group in rupees --/
def total_spent : ℚ := 292.5

/-- The amount spent by each of the first 8 persons in rupees --/
def regular_spend : ℚ := 30

/-- The number of persons who spent the regular amount --/
def regular_count : ℕ := 8

/-- The extra amount spent by the last person compared to the average --/
def extra_spend : ℚ := 20

theorem persons_count :
  n = 9 ∧
  total_spent = regular_count * regular_spend + (total_spent / n + extra_spend) :=
sorry

end NUMINAMATH_CALUDE_persons_count_l2281_228141


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l2281_228114

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each science fiction book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages : total_pages = 3824 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l2281_228114


namespace NUMINAMATH_CALUDE_sine_area_theorem_l2281_228176

open Set
open MeasureTheory
open Interval

-- Define the sine function
noncomputable def f (x : ℝ) := Real.sin x

-- Define the interval
def I : Set ℝ := Icc (-Real.pi) (2 * Real.pi)

-- State the theorem
theorem sine_area_theorem :
  (∫ x in I, |f x| ∂volume) = 6 := by sorry

end NUMINAMATH_CALUDE_sine_area_theorem_l2281_228176


namespace NUMINAMATH_CALUDE_faye_candy_problem_l2281_228139

theorem faye_candy_problem (initial : ℕ) (received : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 47 → received = 40 → final = 62 → 
  initial - eaten + received = final → 
  eaten = 25 := by sorry

end NUMINAMATH_CALUDE_faye_candy_problem_l2281_228139


namespace NUMINAMATH_CALUDE_problem_solution_l2281_228120

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 8) (h2 : x = 2) : y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2281_228120


namespace NUMINAMATH_CALUDE_last_digit_product_l2281_228186

theorem last_digit_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_last_digit_product_l2281_228186


namespace NUMINAMATH_CALUDE_sqrt_300_simplified_l2281_228149

theorem sqrt_300_simplified : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplified_l2281_228149


namespace NUMINAMATH_CALUDE_executive_board_selection_l2281_228102

theorem executive_board_selection (n m : ℕ) (h1 : n = 12) (h2 : m = 5) :
  Nat.choose n m = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l2281_228102


namespace NUMINAMATH_CALUDE_triangle_angle_max_l2281_228129

theorem triangle_angle_max (c : ℝ) (X Y Z : ℝ) : 
  0 < X ∧ 0 < Y ∧ 0 < Z →  -- angles are positive
  X + Y + Z = 180 →  -- angle sum in a triangle
  Z ≤ Y ∧ Y ≤ X →  -- given order of angles
  c * X = 6 * Z →  -- given relation between X and Z
  Z ≤ 36 :=  -- maximum value of Z
by sorry

end NUMINAMATH_CALUDE_triangle_angle_max_l2281_228129


namespace NUMINAMATH_CALUDE_shaded_area_ratio_is_five_ninths_l2281_228188

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a shaded region in the grid -/
structure ShadedRegion :=
  (start_row : ℕ)
  (start_col : ℕ)
  (end_row : ℕ)
  (end_col : ℕ)

/-- Calculates the ratio of shaded area to total area -/
def shaded_area_ratio (g : Grid) (sr : ShadedRegion) : ℚ :=
  sorry

/-- Theorem stating the ratio of shaded area to total area for the given problem -/
theorem shaded_area_ratio_is_five_ninths :
  let g : Grid := ⟨9⟩
  let sr : ShadedRegion := ⟨2, 1, 5, 9⟩
  shaded_area_ratio g sr = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_is_five_ninths_l2281_228188


namespace NUMINAMATH_CALUDE_birdhouse_revenue_theorem_l2281_228137

/-- Calculates the total revenue from selling birdhouses with discount and tax --/
def birdhouse_revenue (
  extra_large_price : ℚ)
  (large_price : ℚ)
  (medium_price : ℚ)
  (small_price : ℚ)
  (extra_small_price : ℚ)
  (extra_large_qty : ℕ)
  (large_qty : ℕ)
  (medium_qty : ℕ)
  (small_qty : ℕ)
  (extra_small_qty : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  let total_before_discount :=
    extra_large_price * extra_large_qty +
    large_price * large_qty +
    medium_price * medium_qty +
    small_price * small_qty +
    extra_small_price * extra_small_qty
  let discounted_amount := total_before_discount * (1 - discount_rate)
  let final_amount := discounted_amount * (1 + tax_rate)
  final_amount

/-- Theorem stating the total revenue from selling birdhouses --/
theorem birdhouse_revenue_theorem :
  birdhouse_revenue 45 22 16 10 5 3 5 7 8 10 (1/10) (6/100) = 464.60 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_revenue_theorem_l2281_228137


namespace NUMINAMATH_CALUDE_pirate_treasure_l2281_228157

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l2281_228157


namespace NUMINAMATH_CALUDE_diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l2281_228118

structure Polygon where
  sides : ℕ
  interior_angle : ℝ
  diagonal_angle : ℝ

def is_equilateral_triangle (p : Polygon) : Prop :=
  p.sides = 3 ∧ p.interior_angle = 60

def is_regular_hexagon (p : Polygon) : Prop :=
  p.sides = 6 ∧ p.interior_angle = 120

theorem diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon (p : Polygon) :
  p.diagonal_angle = 60 → is_equilateral_triangle p ∨ is_regular_hexagon p :=
by sorry

end NUMINAMATH_CALUDE_diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l2281_228118


namespace NUMINAMATH_CALUDE_bottom_row_bricks_count_l2281_228119

/-- Represents a brick wall with a triangular pattern -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ
  h_rows : rows > 0
  h_pattern : total_bricks = (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- The specific brick wall in the problem -/
def problem_wall : BrickWall where
  rows := 5
  total_bricks := 200
  bottom_row_bricks := 42
  h_rows := by norm_num
  h_pattern := by norm_num

theorem bottom_row_bricks_count (wall : BrickWall) 
  (h_rows : wall.rows = 5) 
  (h_total : wall.total_bricks = 200) : 
  wall.bottom_row_bricks = 42 := by
  sorry

#check bottom_row_bricks_count

end NUMINAMATH_CALUDE_bottom_row_bricks_count_l2281_228119


namespace NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l2281_228112

/-- If the terminal side of angle α passes through point P(-5,-12), then sin(3π/2 + α) = 5/13 -/
theorem sin_three_pi_half_plus_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = -5 ∧ r * (Real.sin α) = -12) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l2281_228112


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2281_228164

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2281_228164


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2281_228124

def f (x : ℝ) := x^3 + 3*x - 3

theorem root_exists_in_interval : ∃ x ∈ Set.Icc 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2281_228124


namespace NUMINAMATH_CALUDE_not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l2281_228136

/-- A quadratic equation is a three equal root equation if one root is 1/3 of the other --/
def is_three_equal_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ = (1/3) * x₂

/-- The first part of the problem --/
theorem not_three_equal_root_equation : ¬ is_three_equal_root_equation 1 (-8) 11 := by
  sorry

/-- The second part of the problem --/
theorem three_equal_root_with_negative_one_root (b c : ℤ) :
  is_three_equal_root_equation 1 b c ∧ (∃ x : ℝ, x^2 + b*x + c = 0 ∧ x = -1) → b = 4 ∧ c = 3 := by
  sorry

/-- The third part of the problem --/
theorem three_equal_root_on_line (m n : ℝ) :
  n = 2*m + 1 ∧ is_three_equal_root_equation m n 2 → m = 3/2 ∨ m = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l2281_228136


namespace NUMINAMATH_CALUDE_xanadu_license_plates_l2281_228169

/-- The number of possible letters in each letter position of a Xanadu license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a Xanadu license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in Xanadu. -/
def total_license_plates : ℕ := num_letters^4 * num_digits^2

/-- Theorem stating the total number of valid license plates in Xanadu. -/
theorem xanadu_license_plates : total_license_plates = 45697600 := by
  sorry

end NUMINAMATH_CALUDE_xanadu_license_plates_l2281_228169


namespace NUMINAMATH_CALUDE_equalize_volume_l2281_228108

-- Define the volumes in milliliters
def transparent_volume : ℚ := 12400
def opaque_volume : ℚ := 7600

-- Define the conversion factor from milliliters to liters
def ml_to_l : ℚ := 1000

-- Define the function to calculate the volume to be transferred
def volume_to_transfer : ℚ :=
  (transparent_volume - opaque_volume) / 2

-- Theorem statement
theorem equalize_volume :
  volume_to_transfer = 2400 ∧
  volume_to_transfer / ml_to_l = (12 / 5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equalize_volume_l2281_228108


namespace NUMINAMATH_CALUDE_min_people_liking_both_l2281_228182

theorem min_people_liking_both (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h1 : total = 150)
  (h2 : mozart = 130)
  (h3 : beethoven = 110)
  (h4 : mozart ≤ total)
  (h5 : beethoven ≤ total) :
  mozart + beethoven - total ≤ (min mozart beethoven) ∧
  (min mozart beethoven) = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_people_liking_both_l2281_228182


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2281_228170

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2281_228170


namespace NUMINAMATH_CALUDE_concentric_circles_area_l2281_228172

theorem concentric_circles_area (r : Real) : 
  r > 0 → 
  (π * (3*r)^2 - π * (2*r)^2) + (π * (2*r)^2 - π * r^2) = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l2281_228172


namespace NUMINAMATH_CALUDE_double_sum_reciprocal_product_l2281_228116

/-- The double sum of 1/(mn(m+n+2)) from m=1 to infinity and n=1 to infinity equals -π²/6 -/
theorem double_sum_reciprocal_product : 
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = -π^2 / 6 := by sorry

end NUMINAMATH_CALUDE_double_sum_reciprocal_product_l2281_228116


namespace NUMINAMATH_CALUDE_cosine_identity_l2281_228160

theorem cosine_identity (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) (h2 : Real.cos (x - π/6) = -Real.sqrt 3 / 3) :
  Real.cos (x - π/3) = (-3 + Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l2281_228160


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2281_228117

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,3,4,5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {1,2,3,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2281_228117


namespace NUMINAMATH_CALUDE_cube_pyramid_plane_pairs_l2281_228122

/-- A solid formed by a cube and a pyramid --/
structure CubePyramidSolid where
  cube_edges : Finset (Fin 12)
  pyramid_edges : Finset (Fin 5)

/-- Function to count pairs of edges that determine a plane --/
def count_plane_determining_pairs (solid : CubePyramidSolid) : ℕ :=
  sorry

/-- Theorem stating the number of edge pairs determining a plane --/
theorem cube_pyramid_plane_pairs :
  ∀ (solid : CubePyramidSolid),
  count_plane_determining_pairs solid = 82 :=
sorry

end NUMINAMATH_CALUDE_cube_pyramid_plane_pairs_l2281_228122


namespace NUMINAMATH_CALUDE_soccer_team_wins_l2281_228115

/-- Given a soccer team that played 140 games and won 50 percent of them, 
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 140 → 
  win_percentage = 1/2 → 
  games_won = (total_games : ℚ) * win_percentage → 
  games_won = 70 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l2281_228115


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2281_228180

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + a*b + a*c + b^2 + b*c + c^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2281_228180


namespace NUMINAMATH_CALUDE_point_on_line_l2281_228123

/-- Given a line passing through (0,10) and (-8,0), this theorem proves that 
    the x-coordinate of a point on this line with y-coordinate -6 is -64/5 -/
theorem point_on_line (x : ℚ) : 
  (∀ t : ℚ, t * (-8) = x ∧ t * (-10) + 10 = -6) → x = -64/5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2281_228123


namespace NUMINAMATH_CALUDE_clothing_size_puzzle_l2281_228174

theorem clothing_size_puzzle (anna_size becky_size ginger_size subtracted_number : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - subtracted_number →
  ginger_size = 8 →
  subtracted_number = 4 := by
sorry

end NUMINAMATH_CALUDE_clothing_size_puzzle_l2281_228174


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l2281_228191

theorem factor_divisor_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧
  (∃ n : ℕ, 200 = 10 * n) ∧
  (¬ ∃ n : ℕ, 133 = 19 * n ∨ ∃ n : ℕ, 57 = 19 * n) ∧
  (∃ n : ℕ, 90 = 30 * n ∨ ∃ n : ℕ, 65 = 30 * n) ∧
  (¬ ∃ n : ℕ, 49 = 7 * n ∨ ∃ n : ℕ, 98 = 7 * n) :=
by sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l2281_228191


namespace NUMINAMATH_CALUDE_dance_result_l2281_228148

/-- Represents a sequence of dance steps, where positive numbers are forward steps
    and negative numbers are backward steps. -/
def dance_sequence : List Int := [-5, 10, -2, 2 * 2]

/-- Calculates the final position after performing a sequence of dance steps. -/
def final_position (steps : List Int) : Int :=
  steps.sum

/-- Proves that the given dance sequence results in a final position 7 steps forward. -/
theorem dance_result :
  final_position dance_sequence = 7 := by
  sorry

end NUMINAMATH_CALUDE_dance_result_l2281_228148


namespace NUMINAMATH_CALUDE_square_semicircle_diagonal_l2281_228145

-- Define the square and semicircle
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B.1 - A.1 = s ∧ B.2 - A.2 = 0 ∧
    C.1 - B.1 = 0 ∧ C.2 - B.2 = s ∧
    D.1 - C.1 = -s ∧ D.2 - C.2 = 0 ∧
    A.1 - D.1 = 0 ∧ A.2 - D.2 = -s

def Semicircle (O : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    B.1 - A.1 = 2 * r ∧ B.2 = A.2

-- Define the theorem
theorem square_semicircle_diagonal (A B C D M : ℝ × ℝ) :
  Square A B C D →
  Semicircle ((A.1 + B.1) / 2, A.2) A B →
  B.1 - A.1 = 8 →
  M.1 = (A.1 + B.1) / 2 ∧ M.2 - A.2 = 4 →
  (M.1 - D.1)^2 + (M.2 - D.2)^2 = 160 :=
sorry

end NUMINAMATH_CALUDE_square_semicircle_diagonal_l2281_228145


namespace NUMINAMATH_CALUDE_pine_saplings_sample_count_l2281_228154

/-- Calculates the number of pine saplings in a stratified sample -/
def pine_saplings_in_sample (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℕ :=
  (pine_saplings * sample_size) / total_saplings

/-- Theorem: The number of pine saplings in the stratified sample is 20 -/
theorem pine_saplings_sample_count :
  pine_saplings_in_sample 30000 4000 150 = 20 := by
  sorry

#eval pine_saplings_in_sample 30000 4000 150

end NUMINAMATH_CALUDE_pine_saplings_sample_count_l2281_228154


namespace NUMINAMATH_CALUDE_exam_exemption_logic_l2281_228177

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (score_above_90 : Student → Prop)
variable (exempted : Student → Prop)

-- State the theorem
theorem exam_exemption_logic (s : Student) 
  (h : ∀ x, score_above_90 x → exempted x) :
  ¬(exempted s) → ¬(score_above_90 s) := by
  sorry

end NUMINAMATH_CALUDE_exam_exemption_logic_l2281_228177


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_4410_l2281_228134

/-- Given that 4410 = 2 × 3² × 5 × 7², this function counts the number of positive integer factors of 4410 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization := [(2, 1), (3, 2), (5, 1), (7, 2)]
  sorry

/-- The theorem states that the number of positive integer factors of 4410 that are perfect squares is 4. -/
theorem perfect_square_factors_of_4410 : count_perfect_square_factors = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_4410_l2281_228134


namespace NUMINAMATH_CALUDE_percentage_markup_l2281_228190

theorem percentage_markup (cost_price selling_price : ℝ) : 
  cost_price = 7000 →
  selling_price = 8400 →
  (selling_price - cost_price) / cost_price * 100 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_markup_l2281_228190


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2281_228126

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 5.3
  , dozen := 36
  , twoDozen := 50 }

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 317 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2281_228126


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2281_228150

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m → m > -2 ∧ m < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2281_228150


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2281_228197

theorem expression_simplification_and_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -6
  let original_expression := 3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + 3/2 * x^2 * y)) + 2 * (3 * x * y^2 - x * y)
  let simplified_expression := 6 * x^2 * y
  original_expression = simplified_expression ∧ simplified_expression = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2281_228197


namespace NUMINAMATH_CALUDE_white_square_area_l2281_228153

-- Define the cube's properties
def cube_edge : ℝ := 8
def total_green_paint : ℝ := 192

-- Define the theorem
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let total_surface_area := 6 * face_area
  let green_area_per_face := total_green_paint / 6
  let white_area_per_face := face_area - green_area_per_face
  white_area_per_face = 32 := by sorry

end NUMINAMATH_CALUDE_white_square_area_l2281_228153


namespace NUMINAMATH_CALUDE_reciprocal_squares_sum_l2281_228155

theorem reciprocal_squares_sum (a b : ℕ) (h : a * b = 3) :
  (1 : ℚ) / a^2 + 1 / b^2 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_squares_sum_l2281_228155


namespace NUMINAMATH_CALUDE_min_sum_squares_l2281_228159

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : 0 < y₁) (pos₂ : 0 < y₂) (pos₃ : 0 < y₃)
  (sum_eq : y₁ + 3 * y₂ + 5 * y₃ = 120) :
  y₁^2 + y₂^2 + y₃^2 ≥ 43200 / 361 ∧
  ∃ y₁' y₂' y₃' : ℝ, 
    0 < y₁' ∧ 0 < y₂' ∧ 0 < y₃' ∧
    y₁' + 3 * y₂' + 5 * y₃' = 120 ∧
    y₁'^2 + y₂'^2 + y₃'^2 = 43200 / 361 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2281_228159


namespace NUMINAMATH_CALUDE_total_balloons_l2281_228178

def tom_balloons : ℕ := 9
def sara_balloons : ℕ := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l2281_228178


namespace NUMINAMATH_CALUDE_cycle_price_proof_l2281_228132

/-- Proves that a cycle sold at a 12% loss for Rs. 1232 had an original price of Rs. 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1232)
  (h2 : loss_percentage = 12) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l2281_228132
