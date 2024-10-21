import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_one_l219_21956

theorem fraction_power_product_one :
  (5 / 9 : ℚ) ^ 4 * (5 / 9 : ℚ) ^ (-4 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_one_l219_21956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l219_21949

def M : ℕ := 57^4 + 4 * 57^3 + 6 * 57^2 + 4 * 57 + 1

theorem number_of_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l219_21949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quick_speed_theorem_l219_21932

/-- The distance to Mr. Quick's workplace -/
noncomputable def distance : ℝ := sorry

/-- The ideal travel time in hours -/
noncomputable def ideal_time : ℝ := sorry

/-- The speed required to arrive exactly on time -/
noncomputable def required_speed : ℝ := distance / ideal_time

/-- Condition: Driving at 50 mph results in being 4 minutes late -/
axiom late_condition : distance = 50 * (ideal_time + 4/60)

/-- Condition: Driving at 70 mph results in being 4 minutes early -/
axiom early_condition : distance = 70 * (ideal_time - 4/60)

theorem quick_speed_theorem : required_speed = 58 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quick_speed_theorem_l219_21932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_evaluation_l219_21954

noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * Real.sin x ^ 2 - Real.cos x ^ 2 + 3

theorem triangle_function_evaluation 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : b / a = Real.sqrt 3) 
  (h2 : Real.sin (2 * A + C) / Real.sin A = 2 + 2 * Real.cos (A + C)) 
  (h3 : A + B + C = Real.pi) 
  (h4 : 0 < A ∧ A < Real.pi) 
  (h5 : 0 < B ∧ B < Real.pi) 
  (h6 : 0 < C ∧ C < Real.pi) : 
  f B = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_evaluation_l219_21954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_abc_l219_21997

-- Define the logarithms as noncomputable
noncomputable def a : ℝ := Real.log 3 / Real.log 15
noncomputable def b : ℝ := Real.log 5 / Real.log 25
noncomputable def c : ℝ := Real.log 7 / Real.log 35

-- Theorem statement
theorem descending_order_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_abc_l219_21997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_children_count_l219_21986

theorem camp_children_count (total : ℕ) (girls : ℕ) (new_total : ℕ) :
  (90 : ℚ) / 100 * total = total - girls →
  new_total = total + 50 →
  (5 : ℚ) / 100 * new_total = girls →
  total = 50 := by
  intros boys_percentage new_total_def new_girls_percentage
  sorry

#check camp_children_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_children_count_l219_21986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_work_time_l219_21929

/-- The time it takes Dan to complete the job alone -/
noncomputable def dans_time : ℝ := 12

/-- The time it takes Annie to complete the job alone -/
noncomputable def annies_time : ℝ := 9

/-- The portion of the job Dan completes in 8 hours -/
noncomputable def dans_portion : ℝ := 8 / dans_time

/-- The portion of the job Annie completes in 3 hours -/
noncomputable def annies_portion : ℝ := 3 / annies_time

theorem dans_work_time :
  dans_portion + annies_portion = 1 ∧ dans_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_work_time_l219_21929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_ages_theorem_l219_21983

/-- Represents the ages of children -/
def ChildrenAges : List ℕ := [2, 5, 8, 11, 14, 17, 20, 23, 26]

/-- The number of children -/
def NumChildren : ℕ := 9

/-- Checks if the list of ages represents equal intervals -/
def isEqualInterval (ages : List ℕ) : Prop :=
  ages.length > 1 →
  ∃ d : ℕ, ∀ i : Fin (ages.length - 1), ages[i.val + 1] - ages[i.val] = d

/-- Calculates the sum of squares of a list of numbers -/
def sumOfSquares (list : List ℕ) : ℕ :=
  list.foldl (fun acc x => acc + x * x) 0

/-- The main theorem to prove -/
theorem children_ages_theorem :
  (NumChildren = ChildrenAges.length) ∧
  (isEqualInterval ChildrenAges) ∧
  (∃ parentAge : ℕ, parentAge * parentAge = sumOfSquares ChildrenAges) →
  ChildrenAges = [2, 5, 8, 11, 14, 17, 20, 23, 26] := by
  sorry

#eval ChildrenAges
#eval NumChildren
#eval sumOfSquares ChildrenAges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_ages_theorem_l219_21983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_product_l219_21939

/-- Given that x, y, and z are different natural numbers, each with exactly four natural-number factors,
    prove that x^3 * y^4 * z^2 has 910 positive divisors. -/
theorem divisors_of_product (x y z : ℕ) (hx : (x.factors).length = 4)
    (hy : (y.factors).length = 4) (hz : (z.factors).length = 4)
    (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
    ((x^3 * y^4 * z^2).divisors).card = 910 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_product_l219_21939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2x_minus_1_l219_21905

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_2x_minus_1 :
  {x : ℝ | ∃ y, y = f (2*x - 1)} = Set.Icc 0 (5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2x_minus_1_l219_21905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l219_21941

/-- A rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- The length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  24 * Real.sqrt 58

/-- Theorem: In a rhombus with sides of length 61 units and a shorter diagonal of 110 units,
    the length of the longer diagonal is 24√58 units -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := { side_length := 61, shorter_diagonal := 110 }
  longer_diagonal r = 24 * Real.sqrt 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l219_21941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_count_properties_l219_21993

/-- Represents the number of paths for a frog jumping n times on a regular octagon -/
noncomputable def paths_count (n : ℕ) : ℝ := 
  if n % 2 = 1 then 0
  else (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n / 2 - 1) - (2 - Real.sqrt 2) ^ (n / 2 - 1))

/-- Theorem stating the properties of the paths_count function -/
theorem paths_count_properties :
  (∀ n : ℕ, n > 0 → paths_count (2 * n - 1) = 0) ∧
  (∀ n : ℕ, n > 0 → paths_count (2 * n) = 
    (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_count_properties_l219_21993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_F_E_l219_21946

/-- Represents a chess championship --/
structure Championship where
  participants : Finset String
  average_age : ℝ

/-- Represents the first championship in the problem --/
def first_championship : Championship :=
  { participants := {"A", "B", "C", "D", "E"},
    average_age := 28 }

/-- Represents the second championship in the problem --/
def second_championship : Championship :=
  { participants := {"A", "B", "C", "D", "F"},
    average_age := 30 }

/-- The time difference between championships in years --/
def time_difference : ℕ := 1

/-- Theorem stating the age difference between F and E --/
theorem age_difference_F_E :
  ∃ (age_A age_B age_C age_D age_E age_F : ℝ),
    (age_A + age_B + age_C + age_D + age_E) / 5 = first_championship.average_age ∧
    (age_A + 1 + age_B + 1 + age_C + 1 + age_D + 1 + age_F) / 5 = second_championship.average_age ∧
    age_F - (age_E + ↑time_difference) = 5 := by
  sorry

#check age_difference_F_E

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_F_E_l219_21946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_has_four_terms_l219_21927

/-- Represents a polynomial in x -/
def MyPolynomial (x : ℝ) := ℝ

/-- The original expression with * replaced by 2x -/
def expression (x : ℝ) : ℝ :=
  (x^3 - 2)^2 + (x^2 + 2*x)^2

/-- Counts the number of terms in a polynomial after simplification -/
noncomputable def count_terms (p : ℝ) : ℕ := sorry

/-- Theorem stating that the expression has exactly four terms -/
theorem expression_has_four_terms :
  ∀ x : ℝ, count_terms (expression x) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_has_four_terms_l219_21927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_seven_l219_21978

theorem three_digit_divisible_by_seven (n : ℕ) 
  (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 7 = 0)
  (h3 : (n / 10) % 10 = n % 10) :
  (n / 100 + (n % 100) / 10 + n % 10) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_seven_l219_21978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_two_l219_21999

/-- A string of digits satisfying the given conditions -/
structure ValidString where
  digits : List Nat
  first_is_two : digits.head? = some 2
  length_is_2500 : digits.length = 2500
  consecutive_divisible :
    ∀ i, i + 1 < digits.length →
      let n := digits[i]! * 10 + digits[i + 1]!
      n % 17 = 0 ∨ n % 23 = 0

/-- The largest possible last digit in a ValidString -/
def largest_last_digit (s : ValidString) : Nat :=
  (s.digits.getLast?.getD 0)

theorem largest_last_digit_is_two :
  ∀ s : ValidString, largest_last_digit s ≤ 2 :=
sorry

#check largest_last_digit_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_is_two_l219_21999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_f_and_x_axis_l219_21930

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then x + 1
  else if 0 ≤ x ∧ x ≤ Real.pi/2 then Real.cos x
  else 0

theorem area_enclosed_by_f_and_x_axis : ∫ x in (-1)..(Real.pi/2), |f x| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_f_and_x_axis_l219_21930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_l219_21908

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain D
def D : Set ℝ := Set.Icc 0 1

-- Condition 1: f is non-decreasing on [0,1]
axiom f_nondecreasing : ∀ {x y : ℝ}, x ∈ D → y ∈ D → x ≤ y → f x ≤ f y

-- Condition 2: f(0) = 0
axiom f_zero : f 0 = 0

-- Condition 3: f(x/3) = (1/2)f(x)
axiom f_third : ∀ x, f (x/3) = (1/2) * f x

-- Condition 4: f(1-x) = 1 - f(x)
axiom f_complement : ∀ x, f (1-x) = 1 - f x

-- Theorem to prove
theorem f_sum : f (5/12) + f (1/8) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_l219_21908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_arithmetic_progression_with_large_digit_sum_l219_21909

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem infinite_arithmetic_progression_with_large_digit_sum :
  ∀ M : ℝ, ∃ a : ℕ+,
    (¬ (10 ∣ a.val)) ∧
    (∀ k : ℕ+, sum_of_digits (k.val * a.val) > Int.floor M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_arithmetic_progression_with_large_digit_sum_l219_21909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_50_l219_21974

-- Define the function g(n)
noncomputable def g (n : ℝ) : ℝ := 
  (n^2 - 2*n + 1)^(1/3 : ℝ) + (n^2 - 1)^(1/3 : ℝ) + (n^2 + 2*n + 1)^(1/3 : ℝ)

-- Define the sequence of odd numbers from 1 to 999999
def oddSeq : List ℝ := List.range 500000 |>.map (fun i => 2 * (i : ℝ) + 1)

-- State the theorem
theorem sum_of_reciprocals_equals_50 : 
  (oddSeq.map (fun n => 1 / g n)).sum = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_50_l219_21974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_correct_l219_21938

def robot_price_1 : ℚ := 3
def robot_price_2 : ℚ := 9/2
def robot_price_3 : ℚ := 21/4
def price_multiplier : ℚ := 3
def sales_tax_rate : ℚ := 7/100

def total_payment : ℚ :=
  let tripled_prices := [robot_price_1, robot_price_2, robot_price_3].map (· * price_multiplier)
  let subtotal := tripled_prices.sum
  let sales_tax := (subtotal * sales_tax_rate).floor / 100
  (subtotal + sales_tax).floor / 100

theorem total_payment_is_correct :
  total_payment = 4093/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_correct_l219_21938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_approximately_four_percent_l219_21963

/-- Calculates the annual interest rate given present value, future value, and time period. -/
noncomputable def calculate_interest_rate (present_value : ℝ) (future_value : ℝ) (years : ℝ) : ℝ :=
  ((future_value / present_value) ^ (1 / years) - 1) * 100

/-- Theorem stating that the interest rate is approximately 4% given the problem conditions. -/
theorem interest_rate_is_approximately_four_percent 
  (present_value : ℝ) 
  (future_value : ℝ) 
  (years : ℝ) 
  (h1 : present_value = 156.25)
  (h2 : future_value = 169)
  (h3 : years = 2) :
  ∃ ε > 0, |calculate_interest_rate present_value future_value years - 4| < ε := by
  sorry

#eval Float.round ((((169 : Float) / 156.25) ^ (1 / 2) - 1) * 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_approximately_four_percent_l219_21963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l219_21945

noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

theorem v_domain : 
  {p : ℝ × ℝ | ∃ z, v p.1 p.2 = z} = {p : ℝ × ℝ | p.1 ≠ p.2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l219_21945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l219_21942

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 4 = (X - 3)^2 * q + (31*X - 56) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l219_21942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l219_21944

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (1 + 4 * a n + Real.sqrt (1 + 24 * a n)) / 16

theorem a_explicit_formula (n : ℕ) :
  a n = (1 / 3) * (1 + 1 / 2^(n - 1)) * (1 + 1 / 2^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l219_21944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_representation_sufficient_not_necessary_empty_intersection_negation_statement_l219_21901

-- Definition of odd numbers
def OddNumbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Statement A
theorem odd_numbers_representation : 
  OddNumbers = {x : ℤ | ∃ k : ℤ, x = 2 * k + 1} := by sorry

-- Statement B
theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 2 → x > 1) ∧ (∃ x : ℝ, x > 1 ∧ x ≤ 2) := by sorry

-- Statement C
theorem empty_intersection :
  ({x : ℝ | ∃ y : ℝ, y = x^2 + 1} : Set ℝ) ∩ 
  ({x : ℝ | ∃ y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.2 = p.1^2 + 1}} : Set ℝ) = ∅ := by sorry

-- Statement D (included for completeness, but marked as incorrect)
theorem negation_statement (incorrect : Bool) :
  ¬(∀ x : ℝ, x > 1 → x^2 - x > 0) ↔ (∃ x : ℝ, x > 1 ∧ x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_representation_sufficient_not_necessary_empty_intersection_negation_statement_l219_21901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_price_is_320_l219_21984

-- Define constants for the cylindrical jar
def cylinder_diameter : ℝ := 4
def cylinder_height : ℝ := 5
def cylinder_price : ℝ := 1.2

-- Define constants for the cone-shaped jar
def cone_diameter : ℝ := 8
def cone_height : ℝ := 10

-- Function to calculate volume of a cylinder
noncomputable def cylinder_volume (d h : ℝ) : ℝ := Real.pi * (d / 2) ^ 2 * h

-- Function to calculate volume of a cone
noncomputable def cone_volume (d h : ℝ) : ℝ := (1 / 3) * Real.pi * (d / 2) ^ 2 * h

-- Function to calculate price per unit volume
noncomputable def price_per_unit_volume (price volume : ℝ) : ℝ := price / volume

-- Theorem stating the price of the cone-shaped jar
theorem cone_price_is_320 :
  let cylinder_vol := cylinder_volume cylinder_diameter cylinder_height
  let cone_vol := cone_volume cone_diameter cone_height
  let price_rate := price_per_unit_volume cylinder_price cylinder_vol
  price_rate * cone_vol = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_price_is_320_l219_21984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_and_line_l219_21924

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 49/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the moving circle P
def moving_circle (P : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ), circle_M M.1 M.2 ∧ circle_N N.1 N.2 ∧
  (P.1 - M.1)^2 + (P.2 - M.2)^2 = (7/2 - r)^2 ∧
  (P.1 - N.1)^2 + (P.2 - N.2)^2 = (r + 1/2)^2

-- Define the trajectory of P
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -2

-- Theorem statement
theorem moving_circle_trajectory_and_line :
  ∀ (P : ℝ × ℝ) (r : ℝ),
  moving_circle P r →
  (∀ x y, trajectory x y ↔ ∃ r, moving_circle (x, y) r) ∧
  (∃ k, k^2 = 2 ∧
    ∀ A B : ℝ × ℝ,
    trajectory A.1 A.2 ∧ trajectory B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
    dot_product_condition A B →
    (k = Real.sqrt 2 ∨ k = -Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_and_line_l219_21924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_coefficient_sum_l219_21935

/-- Given two parallel lines with a distance of √5 between them, 
    prove that the sum of certain coefficients is either 0 or 30 -/
theorem parallel_lines_coefficient_sum (b c : ℝ) : 
  (∀ x y, x + 2*y + 3 = 0 ↔ 3*x + b*y + c = 0) → -- parallelism condition
  (abs (9 - c) / Real.sqrt 45 = Real.sqrt 5) →   -- distance condition
  b + c = 0 ∨ b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_coefficient_sum_l219_21935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_is_five_to_one_l219_21992

/-- A right triangular prism -/
structure RightTriangularPrism where
  AB : ℝ
  AA₁ : ℝ
  AB_eq_sqrt3_AA₁ : AB = Real.sqrt 3 * AA₁

/-- The ratio of surface areas of circumscribed to inscribed sphere -/
noncomputable def sphere_surface_area_ratio (prism : RightTriangularPrism) : ℝ :=
  (4 * Real.pi * 5) / (4 * Real.pi)

/-- Theorem stating the ratio of surface areas is 5:1 -/
theorem sphere_surface_area_ratio_is_five_to_one (prism : RightTriangularPrism) :
    sphere_surface_area_ratio prism = 5 := by
  unfold sphere_surface_area_ratio
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_ratio_is_five_to_one_l219_21992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l219_21988

/-- For a geometric sequence with first term a₁ and common ratio q, 
    S_n represents the sum of the first n terms. -/
noncomputable def S_n (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) (h₂ : q ≠ 0) (h₃ : q ≠ 1) :
  (∀ n : ℕ, 2 * S_n a₁ q n = S_n a₁ q (n + 1) + S_n a₁ q (n + 2)) →
  q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l219_21988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plus_count_is_fourteen_l219_21937

/-- A set of symbols consisting of plus and minus signs -/
structure SymbolSet where
  total : ℕ
  plusCount : ℕ
  minusCount : ℕ
  sum_eq : total = plusCount + minusCount
  plus_constraint : ∀ (subset : Finset ℕ), subset.card = 10 → (subset.filter (λ i => i < plusCount)).Nonempty
  minus_constraint : ∀ (subset : Finset ℕ), subset.card = 15 → (subset.filter (λ i => i ≥ plusCount ∧ i < total)).Nonempty

/-- The theorem stating that the number of plus signs is 14 -/
theorem plus_count_is_fourteen (s : SymbolSet) (h : s.total = 23) : s.plusCount = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plus_count_is_fourteen_l219_21937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_equals_light_l219_21962

/-- Represents a square on the grid -/
inductive Square
| Light
| Dark
deriving BEq, Repr

/-- Represents a row of alternating squares -/
def alternatingRow (startWithDark : Bool) : List Square :=
  List.replicate 8 (if startWithDark then Square.Dark else Square.Light)

/-- Represents the entire 8x8 grid -/
def chessboard : List (List Square) :=
  List.map (λ i => alternatingRow (i % 2 == 0)) (List.range 8)

/-- Counts the number of dark squares in a list of squares -/
def countDarkSquares (row : List Square) : Nat :=
  row.filter (· == Square.Dark) |>.length

theorem dark_equals_light :
  (chessboard.map countDarkSquares |>.sum) = 32 := by
  sorry

#eval chessboard.map countDarkSquares |>.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_equals_light_l219_21962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l219_21921

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) :
  (∀ x, f x > m^2 - m) → m ∈ Set.Icc (-1) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l219_21921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_plus_pi_fourth_l219_21982

theorem tan_double_angle_plus_pi_fourth (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.tan (2*α + π/4) = -17/31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_plus_pi_fourth_l219_21982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_l219_21973

-- Define the parabola P
noncomputable def P (x : ℝ) : ℝ := 4 * x^2

-- Define the vertex and focus of P
noncomputable def V₁ : ℝ × ℝ := (0, 0)
noncomputable def F₁ : ℝ × ℝ := (0, 1/16)

-- Define a function to check if a point is on P
def on_P (point : ℝ × ℝ) : Prop :=
  point.2 = P point.1

-- Define the condition for points A and B
def chord_condition (A B : ℝ × ℝ) : Prop :=
  on_P A ∧ on_P B ∧ (A.1 * B.1 = -1/16)

-- Define the midpoint of AB
noncomputable def midpoint_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the vertex and focus of Q
noncomputable def V₂ : ℝ × ℝ := (0, 1/4)
noncomputable def F₂ : ℝ × ℝ := (0, 9/32)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem parabola_ratio :
  distance F₁ F₂ / distance V₁ V₂ = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_l219_21973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_geometric_product_l219_21967

noncomputable section

/-- Geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n-1)

/-- Sum of first n terms of a geometric sequence -/
def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Product of first n terms of a geometric sequence -/
def geometric_product (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁^n * q^((n*(n-1))/2)

theorem max_geometric_product :
  ∃ (q : ℝ),
    let a := geometric_sequence 30 q
    let S := geometric_sum 30 q
    let T := geometric_product 30 q
    8 * S 6 = 9 * S 3 ∧
    ∀ n : ℕ, T n ≤ T 5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_geometric_product_l219_21967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l219_21943

/-- Represents a statement about sampling methods -/
inductive SamplingStatement
| SimpleRandomSmall
| SystematicRandom
| LotteryDrawing
| SystematicEqual

/-- Determines if a given sampling statement is correct -/
def is_correct (s : SamplingStatement) : Bool :=
  match s with
  | SamplingStatement.SimpleRandomSmall => true
  | SamplingStatement.SystematicRandom => false
  | SamplingStatement.LotteryDrawing => true
  | SamplingStatement.SystematicEqual => true

/-- The list of all sampling statements -/
def all_statements : List SamplingStatement :=
  [SamplingStatement.SimpleRandomSmall, SamplingStatement.SystematicRandom,
   SamplingStatement.LotteryDrawing, SamplingStatement.SystematicEqual]

/-- Counts the number of correct statements -/
def count_correct (statements : List SamplingStatement) : Nat :=
  statements.filter is_correct |>.length

theorem correct_statements_count :
  count_correct all_statements = 3 := by
  sorry

#eval count_correct all_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l219_21943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_cost_calculation_l219_21960

/-- Calculates the total cost for a group at a fair with different entrance fees and ride costs. -/
theorem fair_cost_calculation 
  (under_18_fee : ℝ)
  (over_18_fee_percentage : ℝ)
  (ride_cost : ℝ)
  (num_under_18 : ℕ)
  (num_over_18 : ℕ)
  (rides_per_person : ℕ)
  (h1 : under_18_fee = 5)
  (h2 : over_18_fee_percentage = 1.2)
  (h3 : ride_cost = 0.5)
  (h4 : num_under_18 = 2)
  (h5 : num_over_18 = 1)
  (h6 : rides_per_person = 3) :
  (num_under_18 : ℝ) * under_18_fee + 
  (num_over_18 : ℝ) * (under_18_fee * over_18_fee_percentage) + 
  ((num_under_18 + num_over_18) : ℝ) * (rides_per_person : ℝ) * ride_cost = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_cost_calculation_l219_21960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_trigonometric_formulas_l219_21940

-- Definitions for geometric shapes
structure Parallelogram
structure Prism (base : Type)
structure Parallelepiped extends Prism Parallelogram
structure RightParallelepiped extends Parallelepiped
structure Cuboid extends RightParallelepiped
structure RightPrism extends Cuboid
structure Cube extends RightPrism

-- Oblique projection area ratio
noncomputable def oblique_projection_area_ratio : ℝ := Real.sqrt 2 / 4

-- Surface area formulas
noncomputable def surface_area_cylinder (r l : ℝ) : ℝ := 2 * Real.pi * r * (r + l)
noncomputable def surface_area_cone (r l : ℝ) : ℝ := Real.pi * r * (r + l)
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Volume formulas
noncomputable def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def volume_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Trigonometric formulas
noncomputable def sin_sum (α β : ℝ) : ℝ := Real.sin α * Real.cos β + Real.cos α * Real.sin β
noncomputable def cos_sum (α β : ℝ) : ℝ := Real.cos α * Real.cos β - Real.sin α * Real.sin β
noncomputable def sin_double (α : ℝ) : ℝ := 2 * Real.sin α * Real.cos α
noncomputable def cos_double (α : ℝ) : ℝ := Real.cos α^2 - Real.sin α^2
noncomputable def cos_double_alt1 (α : ℝ) : ℝ := 2 * Real.cos α^2 - 1
noncomputable def cos_double_alt2 (α : ℝ) : ℝ := 1 - 2 * Real.sin α^2
noncomputable def sin_cos_sum (α : ℝ) : ℝ := Real.sqrt 2 * Real.sin (α + Real.pi/4)

theorem geometric_and_trigonometric_formulas :
  ∀ (r l h α β : ℝ),
  (surface_area_cylinder r l = 2 * Real.pi * r * (r + l)) ∧
  (surface_area_cone r l = Real.pi * r * (r + l)) ∧
  (surface_area_sphere r = 4 * Real.pi * r^2) ∧
  (volume_cylinder r h = Real.pi * r^2 * h) ∧
  (volume_cone r h = (1/3) * Real.pi * r^2 * h) ∧
  (volume_sphere r = (4/3) * Real.pi * r^3) ∧
  (sin_sum α β = Real.sin α * Real.cos β + Real.cos α * Real.sin β) ∧
  (cos_sum α β = Real.cos α * Real.cos β - Real.sin α * Real.sin β) ∧
  (sin_double α = 2 * Real.sin α * Real.cos α) ∧
  (cos_double α = Real.cos α^2 - Real.sin α^2) ∧
  (cos_double_alt1 α = 2 * Real.cos α^2 - 1) ∧
  (cos_double_alt2 α = 1 - 2 * Real.sin α^2) ∧
  (sin_cos_sum α = Real.sqrt 2 * Real.sin (α + Real.pi/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_trigonometric_formulas_l219_21940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l219_21998

/-- Given a triangle with side lengths a, b, and c, and m_c as the median to side c,
    prove that the median m_c satisfies the inequality (a+b-c)/2 < m_c < (a+b)/2. -/
theorem median_inequality (a b c m_c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_median : m_c^2 = (2 * (a^2 + b^2) - c^2) / 4) : 
  (a + b - c) / 2 < m_c ∧ m_c < (a + b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l219_21998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l219_21953

theorem cubic_polynomial_problem (a b c : ℝ) (Q : ℝ → ℝ) :
  (a^3 - 6*a^2 + 11*a - 6 = 0) →
  (b^3 - 6*b^2 + 11*b - 6 = 0) →
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) →
  (Q a = b + c) →
  (Q b = a + c) →
  (Q c = a + b) →
  (Q (a + b + c) = -27) →
  (∀ x, Q x = 27*x^3 - 162*x^2 + 297*x - 156) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l219_21953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l219_21912

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y
axiom f_zero_at_third : f (1/3) = 0

-- Define the logarithm base 1/8
noncomputable def log_eighth (x : ℝ) : ℝ := Real.log x / Real.log (1/8)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f (log_eighth x) > 0} = Set.union (Set.Ioo 0 (1/2)) (Set.Ioi 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l219_21912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cents_to_win_l219_21904

/-- Represents the state of the game -/
structure GameState where
  beans : ℕ
  cents : ℕ

/-- The rules of the game -/
def give_penny (state : GameState) : GameState :=
  { beans := state.beans * 5, cents := state.cents + 1 }

def give_nickel (state : GameState) : GameState :=
  { beans := state.beans + 1, cents := state.cents + 5 }

/-- Checks if the game is won -/
def is_won (state : GameState) : Prop :=
  state.beans > 2008 ∧ state.beans % 100 = 42

/-- The theorem to prove -/
theorem min_cents_to_win : 
  ∃ (final_state : GameState), 
    final_state.beans = 0 ∧ 
    (∃ (moves : List (GameState → GameState)), 
      (is_won (moves.foldl (λ s f => f s) final_state)) ∧
      ((moves.foldl (λ s f => f s) final_state).cents = 35) ∧
      ∀ (other_state : GameState), 
        is_won other_state → other_state.cents ≥ 35) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cents_to_win_l219_21904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_iff_a_in_open_zero_two_l219_21961

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2 + (a - 6) * x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a / x + 2 * x + (a - 6)

-- State the theorem
theorem not_monotonic_iff_a_in_open_zero_two :
  ∀ a : ℝ, (∃ x y : ℝ, 0 < x ∧ x < 3 ∧ 0 < y ∧ y < 3 ∧ 
    (f_deriv a x < 0 ∧ f_deriv a y > 0)) ↔ 0 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_iff_a_in_open_zero_two_l219_21961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_type_l219_21968

theorem quadratic_radical_type (a b c d : ℝ) : 
  a = Real.sqrt 12 ∧ b = Real.sqrt 18 ∧ c = Real.sqrt 30 ∧ d = Real.sqrt (2/3) →
  (∃ q : ℚ, a = q * Real.sqrt 3) ∧ 
  (∀ q : ℚ, b ≠ q * Real.sqrt 3) ∧ 
  (∀ q : ℚ, c ≠ q * Real.sqrt 3) ∧ 
  (∀ q : ℚ, d ≠ q * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_type_l219_21968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_parabola_l219_21952

/-- Given two points A(-1,1) and B(3,5), and a point C(x, 2x^2) moving on the curve y = 2x^2,
    the minimum value of the dot product AB · AC is -1/2. -/
theorem min_dot_product_on_parabola :
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (3, 5)
  let C : ℝ → ℝ × ℝ := fun x => (x, 2 * x^2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ → ℝ × ℝ := fun x => ((C x).1 - A.1, (C x).2 - A.2)
  let dot_product : ℝ → ℝ := fun x => AB.1 * (AC x).1 + AB.2 * (AC x).2
  ∃ x_min : ℝ, ∀ x : ℝ, dot_product x ≥ dot_product x_min ∧ dot_product x_min = -1/2
:= by
  sorry

#check min_dot_product_on_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_parabola_l219_21952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_from_1_to_2_l219_21934

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem integral_reciprocal_x_from_1_to_2 :
  ∫ x in (1:ℝ)..(2:ℝ), f x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_from_1_to_2_l219_21934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_existence_and_y_coordinate_l219_21919

noncomputable def A : ℝ × ℝ := (-3, 0)
noncomputable def B : ℝ × ℝ := (-2, 2)
noncomputable def C : ℝ × ℝ := (2, 2)
noncomputable def D : ℝ × ℝ := (3, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_P_existence_and_y_coordinate :
  ∃ P : ℝ × ℝ, 
    (distance P A + distance P D = 10) ∧ 
    (distance P B + distance P C = 10) ∧ 
    (P.2 = (-32 + 8 * Real.sqrt 21) / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_existence_and_y_coordinate_l219_21919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l219_21923

/-- Given a triangle ABC with the following properties:
    - b/c = 2√3/3
    - A + 3C = π
    - b = 3√3 (for part 3)
    Prove the following:
    1. cos C = √3/3
    2. sin B = 2√2/3
    3. The area of triangle ABC is 9√2/4 -/
theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  b / c = 2 * Real.sqrt 3 / 3 →
  A + 3 * C = Real.pi →
  b = 3 * Real.sqrt 3 →
  (Real.cos C = Real.sqrt 3 / 3) ∧
  (Real.sin B = 2 * Real.sqrt 2 / 3) ∧
  (1 / 2 * b * c * Real.sin A = 9 * Real.sqrt 2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l219_21923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_monotonicity_when_a_geq_1_range_of_a_when_increasing_l219_21951

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Theorem 1
theorem value_of_a (a : ℝ) (h : a > 0) :
  2 * f a 1 = f a (-1) → a = Real.sqrt 2 / 3 := by sorry

-- Theorem 2
theorem monotonicity_when_a_geq_1 (a : ℝ) (h1 : a ≥ 1) :
  StrictAntiOn (f a) (Set.Ici 0) := by sorry

-- Theorem 3
theorem range_of_a_when_increasing (a : ℝ) (h : a > 0) :
  StrictMonoOn (f a) (Set.Ici 1) →
  0 < a ∧ a ≤ Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_monotonicity_when_a_geq_1_range_of_a_when_increasing_l219_21951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l219_21914

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried." -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried." -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried." -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried." -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_sign : ℕ → Prop := fun n => n ≤ total_trees

/-- The theorem stating the minimum number of signs under which treasures can be buried -/
theorem min_treasure_signs : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ m : ℕ, m < n → ¬(
    (¬truthful_sign m → m = signs_15 ∨ m = signs_8 ∨ m = signs_4 ∨ m = signs_3) ∧
    (truthful_sign m → m ≠ signs_15 ∧ m ≠ signs_8 ∧ m ≠ signs_4 ∧ m ≠ signs_3)
  )) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l219_21914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_given_nonintersecting_lines_l219_21979

/-- Given two lines that do not intersect, prove that the area of the triangle formed by a specific line and the coordinate axes is 2 -/
theorem triangle_area_given_nonintersecting_lines (m : ℝ) : 
  ({(x, y) : ℝ × ℝ | (m + 3) * x + y = 3 * m - 4} ∩ {(x, y) : ℝ × ℝ | 7 * x + (5 - m) * y - 8 = 0} = ∅) →
  (((3 * m + 4) / (m + 3)) * (3 * m + 4) / 2 = 2) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_given_nonintersecting_lines_l219_21979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mcq_probabilities_l219_21969

structure MultipleChoiceQuestion where
  options : Finset Char
  correct_answer : Finset Char
  scoring : Finset Char → Nat

def random_selection (n : Nat) (options : Finset Char) : Finset (Finset Char) :=
  (Finset.powerset options).filter (λ s => s.card = n)

def MCQ : MultipleChoiceQuestion :=
  { options := {'A', 'B', 'C', 'D'},
    correct_answer := {'A', 'B', 'C'},
    scoring := λ s =>
      if s = {'A', 'B', 'C'} then 5
      else if s ⊆ {'A', 'B', 'C'} then 2
      else 0 }

def total_selections : Nat :=
  (Finset.powerset MCQ.options).card - 1

theorem mcq_probabilities :
  (((random_selection 1 MCQ.options).filter (λ s => MCQ.scoring s > 0)).card : ℚ) / (random_selection 1 MCQ.options).card = 3/4 ∧
  (((Finset.powerset MCQ.options).filter (λ s => s.card ≥ 2 ∧ MCQ.scoring s > 0)).card : ℚ) / ((Finset.powerset MCQ.options).filter (λ s => s.card ≥ 2)).card = 5/11 ∧
  (((random_selection 3 MCQ.options).filter (λ s => MCQ.scoring s > 0)).card : ℚ) / (random_selection 3 MCQ.options).card = 1/4 ∧
  (((Finset.powerset MCQ.options).filter (λ s => s.card > 0 ∧ MCQ.scoring s > 0)).card : ℚ) / total_selections = 7/15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mcq_probabilities_l219_21969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l219_21926

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x^3 - x + Real.sin x

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ,
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (Real.pi / 2) →
    f (Real.cos θ^2 + 2 * m * Real.sin θ) + f (-2 * m - 2) > 0) ↔
  m ∈ Set.Ici (-1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l219_21926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_primes_l219_21922

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number uses each digit from 1 to 7 exactly once -/
def usesDigits1To7Once (n : ℕ) : Prop :=
  ∃ (a b c d e f g : ℕ),
    a ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    b ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    c ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    d ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    e ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    f ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    g ∈ ({1,2,3,4,5,6,7} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g

theorem smallest_sum_of_three_primes :
  ∃ (p1 p2 p3 : ℕ),
    isPrime p1 ∧ isPrime p2 ∧ isPrime p3 ∧
    usesDigits1To7Once (p1 + p2 + p3) ∧
    p1 + p2 + p3 = 263 ∧
    ∀ (q1 q2 q3 : ℕ),
      isPrime q1 → isPrime q2 → isPrime q3 →
      usesDigits1To7Once (q1 + q2 + q3) →
      q1 + q2 + q3 ≥ 263 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_primes_l219_21922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_yellow_draw_l219_21931

/-- A bag containing colored balls -/
structure Bag where
  red : ℕ
  yellow : ℕ

/-- The probability of drawing at least one yellow ball when drawing two balls from a bag -/
noncomputable def prob_at_least_one_yellow (b : Bag) : ℝ :=
  1 - (b.red : ℝ) / (b.red + b.yellow) * ((b.red - 1) : ℝ) / (b.red + b.yellow - 1)

theorem certain_yellow_draw (b : Bag) (h1 : b.red = 1) (h2 : b.yellow = 3) :
  prob_at_least_one_yellow b = 1 := by
  sorry

#check certain_yellow_draw

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_yellow_draw_l219_21931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l219_21933

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line -/
def m : ℝ := 3

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := m * (x - point.1) + point.2

theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = 3*x - 1 :=
by
  intro x
  unfold tangent_line m point
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l219_21933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l219_21916

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x - 1

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

-- Theorem statement
theorem tangent_line_y_intercept (a : ℝ) :
  (f_derivative a 1 = (f a 1 - (-2)) / (1 - 0)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l219_21916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_36_l219_21970

/-- Calculates the upstream distance swum given downstream distance, time, and current speed -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (current_speed : ℝ) : ℝ :=
  let still_water_speed := downstream_distance / time - current_speed
  (still_water_speed - current_speed) * time

/-- Proves that the upstream distance is 36 km given the problem conditions -/
theorem upstream_distance_is_36 :
  upstream_distance 81 9 2.5 = 36 := by
  -- Unfold the definition of upstream_distance
  unfold upstream_distance
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_36_l219_21970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_properties_l219_21950

/-- The wedge product of two vectors in 3D space -/
def wedge (a b : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Properties of wedge product -/
theorem wedge_properties (l : ℝ) (a b c : ℝ × ℝ × ℝ) :
  (wedge (l • a) b = l * wedge a b) ∧
  (wedge a (b + c) = wedge a b + wedge a c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_properties_l219_21950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_is_rational_l219_21918

/-- Represents an infinite repeating decimal with a cycle of 1, 2, or 3 digits -/
structure RepeatingDecimal where
  intPart : ℤ
  nonRepeat : ℚ
  repeatPart : ℕ
  cycleLength : Fin 3

/-- Converts a RepeatingDecimal to a rational number -/
noncomputable def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- Theorem: Any infinite repeating decimal with a repeating cycle of 1, 2, or 3 digits
    can be expressed as a rational number -/
theorem repeating_decimal_is_rational (x : RepeatingDecimal) :
  ∃ (a b : ℤ), toRational x = a / b ∧ b ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_is_rational_l219_21918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_ending_75_l219_21977

theorem count_four_digit_integers_ending_75 : 
  (Finset.filter (fun n : ℕ => n ≥ 1000 ∧ n < 10000 ∧ n % 100 = 75) (Finset.range 10000)).card = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_ending_75_l219_21977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_P_l219_21906

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the point P
def point_P (a : ℝ) : ℝ × ℝ := (1, a)

-- Define the line containing the shortest chord
def shortest_chord_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Theorem statement
theorem shortest_chord_through_P (a : ℝ) :
  (∃ (x y : ℝ), my_circle x y ∧ x = 1 ∧ y = a) →  -- P(1,a) is inside the circle
  (∀ (x y : ℝ), shortest_chord_line x y → 
    ∃ (t : ℝ), x = 1 + t ∧ y = a + 2*t) →  -- The shortest chord passes through P
  a = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_P_l219_21906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_original_seat_l219_21980

-- Define the type for seats
inductive Seat : Type
| one : Seat
| two : Seat
| three : Seat
| four : Seat
| five : Seat
| six : Seat
| seven : Seat
deriving Repr, DecidableEq

-- Define the type for friends
inductive Friend : Type
| fiona : Friend
| greg : Friend
| hannah : Friend
| ian : Friend
| jane : Friend
| kayla : Friend
| lou : Friend
deriving Repr, DecidableEq

-- Define a function to represent the initial seating arrangement
noncomputable def initial_seating : Friend → Seat :=
  fun _ => Seat.one  -- Placeholder implementation

-- Define a function to represent the final seating arrangement
noncomputable def final_seating : Friend → Seat :=
  fun _ => Seat.one  -- Placeholder implementation

-- Define the movements
def greg_move (s : Seat) : Seat := s  -- Placeholder implementation
def hannah_move (s : Seat) : Seat := s  -- Placeholder implementation
def ian_jane_switch (s : Seat) : Seat := s  -- Placeholder implementation
def kayla_move (s : Seat) : Seat := s  -- Placeholder implementation
def lou_move (s : Seat) : Seat := s  -- Placeholder implementation

-- Theorem to prove
theorem fiona_original_seat :
  initial_seating Friend.fiona = Seat.one →
  (∃ (end_seat : Seat), end_seat = Seat.one ∨ end_seat = Seat.seven) →
  (∀ (f : Friend), f ≠ Friend.fiona →
    final_seating f = 
      lou_move (
        kayla_move (
          ian_jane_switch (
            hannah_move (
              greg_move (initial_seating f)
            )
          )
        )
      )
  ) →
  final_seating Friend.fiona = 
    (if initial_seating Friend.fiona = Seat.one then Seat.seven else Seat.one) :=
by
  intros h1 h2 h3
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_original_seat_l219_21980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l219_21990

theorem system_solution :
  ∀ x y z : ℝ,
  (x^2 + y*z = 1) ∧ 
  (y^2 - x*z = 0) ∧ 
  (z^2 + x*y = 1) →
  ((x = Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2 ∧ z = Real.sqrt 2 / 2) ∨ 
   (x = -Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2 ∧ z = -Real.sqrt 2 / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l219_21990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l219_21947

noncomputable def ellipse_equation (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def hyperbola_equation (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem eccentricity_relation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_eccentricity a b = Real.sqrt 2 / 2) :
  hyperbola_eccentricity a b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_relation_l219_21947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_l219_21991

/-- The area of a circular sector with radius r and central angle θ (in radians) -/
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

theorem sector_area_15_degrees :
  let r : ℝ := 6
  let θ : ℝ := 15 * Real.pi / 180
  sector_area r θ = (3/2) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_l219_21991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l219_21928

/-- Circle C in the Cartesian plane -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Line l passing through point P(1,1) with slope angle π/6 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

/-- Point P -/
def point_P : ℝ × ℝ := (1, 1)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The product of distances from P to intersection points of line_l and circle_C is 2 -/
theorem intersection_distance_product :
  ∃ t₁ t₂ : ℝ,
    t₁ ≠ t₂ ∧
    circle_C (line_l t₁).1 (line_l t₁).2 ∧
    circle_C (line_l t₂).1 (line_l t₂).2 ∧
    distance point_P (line_l t₁) * distance point_P (line_l t₂) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l219_21928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l219_21913

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : V) : Prop :=
  ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem collinear_vectors (e₁ e₂ : V) (k : ℝ) 
  (h_nonzero₁ : e₁ ≠ 0)
  (h_nonzero₂ : e₂ ≠ 0)
  (h_non_collinear : ¬ collinear e₁ e₂)
  (h_collinear : collinear (k • e₁ + e₂) (e₁ + k • e₂)) :
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l219_21913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sin_cos_identity_l219_21981

theorem triangle_angles_sin_cos_identity (A B C : ℝ) (n : ℤ)
  (h_triangle : A + B + C = π) :
  Real.sin ((2 * n + 1 : ℝ) * A) + Real.sin ((2 * n + 1 : ℝ) * B) + Real.sin ((2 * n + 1 : ℝ) * C) =
  (-1 : ℝ)^n * 4 * Real.cos ((2 * n + 1 : ℝ) * A / 2) * Real.cos ((2 * n + 1 : ℝ) * B / 2) * Real.cos ((2 * n + 1 : ℝ) * C / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_sin_cos_identity_l219_21981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_canonical_equations_l219_21907

/-- Given two planes in R³, prove that their line of intersection 
    can be represented by the given canonical equations. -/
theorem intersection_line_canonical_equations 
  (plane1 : ∀ x y z : ℝ, 3 * x + 3 * y + z - 1 = 0)
  (plane2 : ∀ x y z : ℝ, 2 * x - 3 * y - 2 * z + 6 = 0) :
  ∃ t : ℝ, ∀ x y z : ℝ, 
    (3 * x + 3 * y + z - 1 = 0 ∧ 2 * x - 3 * y - 2 * z + 6 = 0) ↔ 
    ((x + 1) / (-3) = t ∧ (y - 4/3) / 8 = t ∧ z / (-15) = t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_canonical_equations_l219_21907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_cost_theorem_l219_21925

/-- The total cost of making triangular signs -/
def total_cost (height base : ℚ) (price_per_sqm : ℚ) (num_signs : ℕ) : ℚ :=
  let area_sqm := (base * height / 2) * (1 / 100)  -- Convert from sq dm to sq m
  let cost_per_sign := area_sqm * price_per_sqm
  cost_per_sign * num_signs

/-- Theorem stating the total cost of making 100 triangular signs -/
theorem sign_cost_theorem :
  total_cost (78/10) 9 90 100 = 3159 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_cost_theorem_l219_21925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l219_21989

theorem sin_A_value (A : ℝ) 
  (h1 : Real.sin (A + π/4) = 7*Real.sqrt 2/10) 
  (h2 : A ∈ Set.Ioo (π/4) π) : 
  Real.sin A = 4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l219_21989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l219_21987

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ (r : ℚ), b = Int.floor (r * a) ∧ c = Int.floor (r * r * a)

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let digits := [n / 100, (n / 10) % 10, n % 10]
  List.Pairwise (· ≠ ·) digits ∧
  is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) ∧
  n / 100 < 9

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 568 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l219_21987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_m_n_l219_21920

/-- The minimum sum of m and n given the conditions -/
theorem min_sum_m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (m + n + 2*m*n = 1) → (m + n ≥ Real.sqrt 3 - 1 ∧ ∃ m₀ n₀, m₀ + n₀ = Real.sqrt 3 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_m_n_l219_21920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_at_500m_l219_21964

/-- Temperature decrease per 100 meters of elevation gain -/
noncomputable def temp_decrease_per_100m : ℝ := 0.7

/-- Temperature at the foot of the mountain in °C -/
noncomputable def temp_at_foot : ℝ := 28

/-- Elevation above the foot of the mountain in meters -/
noncomputable def elevation : ℝ := 500

/-- Calculate the temperature at a given elevation -/
noncomputable def temp_at_elevation (e : ℝ) : ℝ :=
  temp_at_foot - (e / 100) * temp_decrease_per_100m

theorem temp_at_500m :
  temp_at_elevation elevation = 24.5 := by
  -- Unfold the definitions
  unfold temp_at_elevation temp_at_foot temp_decrease_per_100m elevation
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_at_500m_l219_21964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_passes_through_points_l219_21994

def f (x : ℝ) : ℝ := x^4

theorem f_is_even_and_passes_through_points : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  f 0 = 0 ∧ 
  f 1 = 1 := by
  constructor
  · intro x
    simp [f]
    ring
  · constructor
    · simp [f]
    · simp [f]

#check f_is_even_and_passes_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_passes_through_points_l219_21994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l219_21917

/-- The decimal representation of the number we're working with -/
def repeating_decimal : ℚ := 71264 / 99900

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 79061333 / 999900

/-- Theorem stating that the repeating decimal equals the target fraction -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l219_21917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_in_first_interval_l219_21976

-- Define the function f(x) = x^3 - (1/2)^(x-2)
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

-- State the theorem
theorem intersection_point_in_first_interval 
  (x₀ : ℝ) 
  (n : ℕ+) 
  (h1 : f x₀ = 0) 
  (h2 : x₀ ∈ Set.Ioo (n : ℝ) ((n : ℝ) + 1)) : 
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_in_first_interval_l219_21976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_special_case_f_value_negative_1860_l219_21972

open Real

-- Define f(α) as given in the problem
noncomputable def f (α : ℝ) : ℝ := (sin (π - α) * cos (2*π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : ℝ) (h : π < α ∧ α < 3*π/2) : f α = cos α := by
  sorry

-- Theorem 2: Value of f(α) when cos(α - 3π/2) = 1/5
theorem f_value_special_case (α : ℝ) (h1 : π < α ∧ α < 3*π/2) (h2 : cos (α - 3*π/2) = 1/5) :
  f α = -2 * Real.sqrt 6 / 5 := by
  sorry

-- Theorem 3: Value of f(-1860°)
theorem f_value_negative_1860 : f (-1860 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_special_case_f_value_negative_1860_l219_21972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_rational_value_l219_21915

theorem cosine_sum_rational_value : 
  (Real.cos (3*π/16))^6 + (Real.cos (11*π/16))^6 + 3*Real.sqrt 2/16 = 5/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_rational_value_l219_21915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_random_sampling_properties_l219_21965

/-- Simple random sampling is a sampling technique where each sample has an equal probability of being chosen. -/
def SimpleRandomSampling : Type := Unit

/-- A population is a finite set of individuals. -/
structure Population where
  individuals : Finset Nat

/-- Sampling is a process of selecting individuals from a population. -/
structure Sampling where
  pop : Population
  selected : Finset Nat

/-- Equal probability sampling ensures each individual has the same chance of being selected. -/
def EqualProbabilitySampling (s : Sampling) : Prop :=
  ∀ i ∈ s.pop.individuals, ∃ p : ℝ, p > 0 ∧ p = p

theorem simple_random_sampling_properties (srs : SimpleRandomSampling) :
  -- Statement ①
  (∀ p : Population, True) ∧
  -- Statement ②
  (∀ s : Sampling, ∃ process : Unit, True) ∧
  -- Statement ③
  (∀ s : Sampling, s.selected ⊆ s.pop.individuals) ∧
  -- Statement ④
  (∀ s : Sampling, EqualProbabilitySampling s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_random_sampling_properties_l219_21965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l219_21971

theorem remainder_theorem (a b c : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 13)
  (hc : c % 15 = 14) :
  (a + b + c) % 15 = 8 ∧ (a + b + c) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l219_21971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_eq_15_l219_21975

/-- The coefficient of x^3 in the expansion of x(1 + x)^6 -/
def coefficient_x_cubed : ℕ :=
  (List.range 7).map (λ k => k * Nat.choose 6 (k - 1)) |>.sum

theorem coefficient_x_cubed_eq_15 : coefficient_x_cubed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_eq_15_l219_21975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_line_l219_21995

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

-- Define the foci
def focus1 : ℝ × ℝ := (-2, 0)
def focus2 : ℝ × ℝ := (2, 0)

-- Define the point Q
def Q : ℝ × ℝ := (0, 2)

-- Define the line passing through Q and a point (x, y)
def line_through_Q (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the area of triangle OEF
noncomputable def area_OEF (E F : ℝ × ℝ) : ℝ := 
  let (x1, y1) := E
  let (x2, y2) := F
  abs (x1 * y2 - x2 * y1) / 2

-- Theorem statement
theorem hyperbola_intersection_line :
  ∀ k : ℝ, ∀ E F : ℝ × ℝ,
  (∃ x y : ℝ, E = (x, y) ∧ hyperbola x y ∧ line_through_Q k x y) →
  (∃ x y : ℝ, F = (x, y) ∧ hyperbola x y ∧ line_through_Q k x y) →
  E ≠ F →
  area_OEF E F = 2 * Real.sqrt 2 →
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_line_l219_21995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_divisibility_condition_l219_21911

theorem binomial_divisibility_condition (k : ℕ) : k ≥ 2 → ∃ n : ℕ,
  n > 0 ∧ (Nat.choose n k) % n = 0 ∧ ∀ m : ℕ, 2 ≤ m → m < k → (Nat.choose n m) % n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_divisibility_condition_l219_21911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l219_21903

noncomputable def f (x : ℝ) : ℝ := Real.cos (2*x + Real.pi/6) + Real.cos (2*x - Real.pi/6) - Real.cos (2*x + Real.pi/2) + 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), x ∈ Set.Icc (k*Real.pi + Real.pi/12) (k*Real.pi + 7*Real.pi/12) → 
    ∀ (y : ℝ), y ∈ Set.Icc (k*Real.pi + Real.pi/12) (k*Real.pi + 7*Real.pi/12) → x < y → f y < f x) ∧
  (∃ (m : ℝ), m > 0 ∧ 
    (∀ (x : ℝ), f (x + m) = f (Real.pi/2 - x)) ∧
    (∀ (n : ℝ), n > 0 ∧ (∀ (x : ℝ), f (x + n) = f (Real.pi/2 - x)) → m ≤ n) ∧
    m = Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l219_21903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_distance_proof_l219_21948

/-- The distance between two cities given specific travel conditions. -/
def distance_between_cities (cara_speed dan_min_speed dan_delay : ℝ) : ℝ :=
  120 -- Distance between cities

/-- Theorem stating the conditions for the distance between cities. -/
theorem distance_theorem (cara_speed dan_min_speed dan_delay : ℝ) :
  let d := distance_between_cities cara_speed dan_min_speed dan_delay
  let cara_time := d / cara_speed
  let dan_time := d / dan_min_speed
  cara_time = dan_time + dan_delay ∧
  cara_speed = 30 ∧
  dan_min_speed = 48 ∧
  dan_delay = 1.5 →
  d = 120 := by
  sorry

/-- Proof of the distance between cities. -/
theorem distance_proof : distance_between_cities 30 48 1.5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_distance_proof_l219_21948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l219_21910

/-- A function f(x) = cos(2x + φ) with graph symmetric about (2π/3, 0) -/
noncomputable def f (φ : ℝ) : ℝ → ℝ := fun x ↦ Real.cos (2 * x + φ)

/-- The function g(x) obtained by translating f(x) to the right by m units -/
noncomputable def g (φ m : ℝ) : ℝ → ℝ := fun x ↦ f φ (x - m)

/-- Predicate for even functions -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- Theorem stating the minimum positive m for which g is even -/
theorem min_translation_for_even_function (φ : ℝ) :
  (∃ k : ℤ, φ = k * Real.pi - 5 * Real.pi / 6) →
  (∀ m : ℝ, m > 0 → IsEven (g φ m) → m ≥ Real.pi / 12) ∧
  IsEven (g φ (Real.pi / 12)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l219_21910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_edges_l219_21985

theorem rectangular_parallelepiped_edges 
  (a b c : ℝ) 
  (h1 : (a * b) / (b * c) = 16 / 21 ∧ (a * b) / (a * c) = 16 / 28) 
  (h2 : a^2 + b^2 + c^2 = 29^2) : 
  a = 16 ∧ b = 12 ∧ c = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_edges_l219_21985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l219_21900

theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l219_21900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l219_21958

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 61.11 →
  discount1 = 10 →
  final_price = list_price * (1 - discount1 / 100) * (1 - discount2 / 100) →
  discount2 = 3 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l219_21958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l219_21936

theorem sufficient_not_necessary :
  (∀ x : ℝ, |x - 2| < 1 → x^2 + x - 2 > 0) ∧
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ |x - 2| ≥ 1) :=
by
  constructor
  · intro x h
    -- Proof for the first part
    sorry
  · -- Proof for the second part
    use -3
    constructor
    · -- Show (-3)^2 + (-3) - 2 > 0
      norm_num
    · -- Show |(-3) - 2| ≥ 1
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l219_21936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angles_of_right_triangle_pyramid_l219_21966

/-- A pyramid with a right triangular base -/
structure RightTrianglePyramid where
  /-- One acute angle of the base triangle -/
  α : Real
  /-- Assumption that α is between 0 and π/2 -/
  α_pos : 0 < α
  α_lt_pi_div_2 : α < Real.pi / 2
  /-- All lateral edges are equally inclined to the base -/
  lateral_edges_equal_incline : True
  /-- The height of the pyramid is equal to the hypotenuse of the base triangle -/
  height_eq_hypotenuse : True

/-- The dihedral angles at the base of the pyramid -/
noncomputable def dihedral_angles (p : RightTrianglePyramid) : Fin 3 → Real
| 0 => Real.pi / 2
| 1 => Real.arctan (2 / Real.sin p.α)
| 2 => Real.arctan (2 / Real.cos p.α)

/-- Theorem stating the dihedral angles of the pyramid -/
theorem dihedral_angles_of_right_triangle_pyramid (p : RightTrianglePyramid) :
  dihedral_angles p = λ i => match i with
    | 0 => Real.pi / 2
    | 1 => Real.arctan (2 / Real.sin p.α)
    | 2 => Real.arctan (2 / Real.cos p.α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angles_of_right_triangle_pyramid_l219_21966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_solution_l219_21996

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 1) else -((-x) * ((-x) + 1))

-- State the theorem
theorem odd_function_solution (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  f a = -2 →                  -- f(a) = -2
  a = -2 :=                   -- Conclusion: a = -2
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_solution_l219_21996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l219_21957

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- The nth term of a geometric sequence -/
noncomputable def GeometricSequence.a_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * g.q ^ (n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSequence.S_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_properties :
  ∀ (g : GeometricSequence),
    (g.a_n 4 = 27 ∧ g.q = -3 → g.a_n 7 = 729 ∧ g.S_n 4 = -20) ∧
    (g.a_n 5 - g.a_n 1 = 15 ∧ g.a_n 4 - g.a_n 2 = 6 → g.a_n 3 = -4 ∨ g.a_n 3 = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l219_21957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_l219_21959

-- Define the points and circles
variable (A B C D E F : EuclideanPlane)
variable (ω1 ω2 : Circle EuclideanPlane)

-- Define the conditions
variable (h1 : Cyclic A B C D ω1)
variable (h2 : A ∈ ω2.points)
variable (h3 : B ∈ ω2.points)
variable (h4 : E ∈ ω2.points)
variable (h5 : E ∈ (Segment.mk D B).toLine)
variable (h6 : E ≠ B)
variable (h7 : F ∈ ω2.points)
variable (h8 : F ∈ (Segment.mk C A).toLine)
variable (h9 : F ≠ A)
variable (h10 : (ω1.tangentLine C).parallel (Line.throughPts A E))

-- State the theorem
theorem tangent_parallel :
  (ω2.tangentLine F).parallel (Line.throughPts A D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_l219_21959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_max_distance_l219_21955

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

-- Define the line l
def l (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x y a : ℝ) : ℝ :=
  abs (x + 4 * y - a - 4) / Real.sqrt 17

-- Theorem for intersection points
theorem intersection_points (a : ℝ) :
  a = -1 →
  ∃ θ₁ θ₂ : ℝ, C θ₁ = (3, 0) ∧ C θ₂ = (-21/25, 24/25) ∧
  ∃ t₁ t₂ : ℝ, l a t₁ = C θ₁ ∧ l a t₂ = C θ₂ :=
by sorry

-- Theorem for maximum distance
theorem max_distance (a : ℝ) :
  (∃ θ : ℝ, distance_point_to_line (C θ).1 (C θ).2 a = Real.sqrt 17) ∧
  (∀ θ : ℝ, distance_point_to_line (C θ).1 (C θ).2 a ≤ Real.sqrt 17) ↔
  a = -16 ∨ a = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_max_distance_l219_21955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_third_irrational_l219_21902

theorem pi_third_irrational (h1 : ∃ (a b : ℚ), -3/2 = a / b)
                            (h2 : ∃ (c d : ℚ), -Real.sqrt 4 = c / d)
                            (h3 : ∃ (e f : ℚ), (23 : ℚ) / 100 = e / f) :
  ¬ ∃ (x y : ℚ), Real.pi / 3 = x / y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_third_irrational_l219_21902
