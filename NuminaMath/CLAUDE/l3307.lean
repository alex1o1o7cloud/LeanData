import Mathlib

namespace NUMINAMATH_CALUDE_beth_cookie_price_l3307_330763

/-- Represents a cookie baker --/
structure Baker where
  name : String
  cookieShape : String
  cookieCount : ℕ
  cookieArea : ℝ
  cookiePrice : ℝ

/-- Given conditions of the problem --/
def alexBaker : Baker := {
  name := "Alex"
  cookieShape := "rectangle"
  cookieCount := 10
  cookieArea := 20
  cookiePrice := 0.50
}

def bethBaker : Baker := {
  name := "Beth"
  cookieShape := "circle"
  cookieCount := 16
  cookieArea := 12.5
  cookiePrice := 0  -- To be calculated
}

/-- The total dough used by each baker --/
def totalDough (b : Baker) : ℝ := b.cookieCount * b.cookieArea

/-- The total earnings of a baker --/
def totalEarnings (b : Baker) : ℝ := b.cookieCount * b.cookiePrice * 100  -- in cents

/-- The main theorem to prove --/
theorem beth_cookie_price :
  totalDough alexBaker = totalDough bethBaker →
  totalEarnings alexBaker = bethBaker.cookieCount * 31.25 := by
  sorry


end NUMINAMATH_CALUDE_beth_cookie_price_l3307_330763


namespace NUMINAMATH_CALUDE_base6_addition_l3307_330780

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base6_addition :
  base6ToDecimal [4, 5, 3, 5] + base6ToDecimal [2, 3, 2, 4, 3] =
  base6ToDecimal [3, 2, 2, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l3307_330780


namespace NUMINAMATH_CALUDE_square_area_given_circle_l3307_330774

-- Define the area of the circle
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_circle_l3307_330774


namespace NUMINAMATH_CALUDE_book_profit_rate_l3307_330756

/-- Calculate the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 90 is 80% -/
theorem book_profit_rate : rate_of_profit 50 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l3307_330756


namespace NUMINAMATH_CALUDE_julia_miles_driven_l3307_330716

theorem julia_miles_driven (darius_miles julia_miles total_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) : 
  julia_miles = 998 := by
  sorry

end NUMINAMATH_CALUDE_julia_miles_driven_l3307_330716


namespace NUMINAMATH_CALUDE_division_problem_l3307_330797

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 15 ∧ divisor = 3 ∧ remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3307_330797


namespace NUMINAMATH_CALUDE_student_desk_arrangement_impossibility_l3307_330726

theorem student_desk_arrangement_impossibility :
  ∀ (total_students total_desks : ℕ) 
    (girls boys : ℕ) 
    (girls_with_boys boys_with_girls : ℕ),
  total_students = 450 →
  total_desks = 225 →
  girls + boys = total_students →
  2 * total_desks = total_students →
  2 * girls_with_boys = girls →
  2 * boys_with_girls = boys →
  False :=
by sorry

end NUMINAMATH_CALUDE_student_desk_arrangement_impossibility_l3307_330726


namespace NUMINAMATH_CALUDE_all_sums_representable_l3307_330784

/-- Available coin denominations -/
def Coins : Set ℕ := {1, 2, 5, 10}

/-- A function that checks if a sum can be represented by both even and odd number of coins -/
def canRepresentEvenAndOdd (S : ℕ) : Prop :=
  ∃ (even_coins odd_coins : List ℕ),
    (∀ c ∈ even_coins, c ∈ Coins) ∧
    (∀ c ∈ odd_coins, c ∈ Coins) ∧
    (even_coins.sum = S) ∧
    (odd_coins.sum = S) ∧
    (even_coins.length % 2 = 0) ∧
    (odd_coins.length % 2 = 1)

/-- Theorem stating that any sum greater than 1 can be represented by both even and odd number of coins -/
theorem all_sums_representable (S : ℕ) (h : S > 1) :
  canRepresentEvenAndOdd S := by
  sorry

end NUMINAMATH_CALUDE_all_sums_representable_l3307_330784


namespace NUMINAMATH_CALUDE_marks_ratio_polly_willy_l3307_330769

/-- Given the ratios of marks between students, prove the ratio between Polly and Willy -/
theorem marks_ratio_polly_willy (p s w : ℝ) 
  (h1 : p / s = 4 / 5) 
  (h2 : s / w = 5 / 2) : 
  p / w = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_ratio_polly_willy_l3307_330769


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3307_330789

theorem min_value_of_expression (x y : ℝ) :
  (2 * x * y - 1)^2 + (x - y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (2 * a * b - 1)^2 + (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3307_330789


namespace NUMINAMATH_CALUDE_max_value_of_objective_function_l3307_330759

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  y ≥ x ∧ x + 3 * y ≤ 4 ∧ x ≥ -2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ :=
  |x - 3 * y|

-- Theorem statement
theorem max_value_of_objective_function :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (x y : ℝ), ConstraintSet x y →
  ObjectiveFunction x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_objective_function_l3307_330759


namespace NUMINAMATH_CALUDE_inequality_properties_l3307_330768

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3307_330768


namespace NUMINAMATH_CALUDE_function_characterization_l3307_330794

theorem function_characterization (f : ℝ → ℤ) 
  (h1 : ∀ x y : ℝ, f (x + y) < f x + f y)
  (h2 : ∀ x : ℝ, f (f x) = ⌊x⌋ + 2) :
  ∀ x : ℤ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_function_characterization_l3307_330794


namespace NUMINAMATH_CALUDE_stock_price_increase_l3307_330701

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l3307_330701


namespace NUMINAMATH_CALUDE_xy_value_l3307_330718

theorem xy_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3307_330718


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3307_330712

theorem sqrt_equation_solution (a b : ℕ) : 
  (Real.sqrt (8 + b / a) = 2 * Real.sqrt (b / a)) → (a = 63 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3307_330712


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3307_330770

/-- Given two circles X and Y where an arc of 60° on X has the same length as an arc of 40° on Y,
    the ratio of the area of circle X to the area of circle Y is 9/4. -/
theorem circle_area_ratio (R_X R_Y : ℝ) (h : R_X > 0 ∧ R_Y > 0) :
  (60 / 360 * (2 * Real.pi * R_X) = 40 / 360 * (2 * Real.pi * R_Y)) →
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3307_330770


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l3307_330799

/-- Represents the number of arrangements of books on a shelf. -/
def num_arrangements (n : ℕ) (chinese : ℕ) (math : ℕ) (physics : ℕ) : ℕ := sorry

/-- Theorem stating the number of arrangements for the given problem. -/
theorem book_arrangement_problem :
  num_arrangements 5 2 2 1 = 48 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l3307_330799


namespace NUMINAMATH_CALUDE_group_size_from_weight_change_l3307_330779

/-- The number of people in a group where replacing a 35 kg person with a 55 kg person
    increases the average weight by 2.5 kg is 8. -/
theorem group_size_from_weight_change (n : ℕ) : 
  (n : ℝ) * 2.5 = 55 - 35 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_weight_change_l3307_330779


namespace NUMINAMATH_CALUDE_max_value_after_operations_l3307_330766

def initial_numbers : List ℕ := [1, 2, 3]
def num_operations : ℕ := 9

def operation (numbers : List ℕ) : List ℕ :=
  let sum := numbers.sum
  let max := numbers.maximum?
  match max with
  | none => numbers
  | some m => (sum - m) :: (numbers.filter (· ≠ m))

def iterate_operation (n : ℕ) (numbers : List ℕ) : List ℕ :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_operation n (operation numbers)

theorem max_value_after_operations :
  (iterate_operation num_operations initial_numbers).maximum? = some 233 :=
sorry

end NUMINAMATH_CALUDE_max_value_after_operations_l3307_330766


namespace NUMINAMATH_CALUDE_classroom_size_is_81_l3307_330791

/-- Represents the number of students in a classroom with specific shirt and shorts conditions. -/
def classroom_size : ℕ → Prop := fun n =>
  ∃ (striped checkered shorts : ℕ),
    -- Total number of students
    n = striped + checkered
    -- Two-thirds wear striped shirts, one-third wear checkered shirts
    ∧ 3 * striped = 2 * n
    ∧ 3 * checkered = n
    -- Shorts condition
    ∧ shorts = checkered + 19
    -- Striped shirts condition
    ∧ striped = shorts + 8

/-- The number of students in the classroom satisfying the given conditions is 81. -/
theorem classroom_size_is_81 : classroom_size 81 := by
  sorry

end NUMINAMATH_CALUDE_classroom_size_is_81_l3307_330791


namespace NUMINAMATH_CALUDE_shoe_pairing_probability_l3307_330773

/-- A permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of permutations of n elements with all cycle lengths ≥ k -/
def numLongCyclePerms (n k : ℕ) : ℕ := sorry

/-- The probability of a random permutation of n elements having all cycle lengths ≥ k -/
def probLongCycles (n k : ℕ) : ℚ :=
  (numLongCyclePerms n k : ℚ) / (n.factorial : ℚ)

theorem shoe_pairing_probability :
  probLongCycles 8 5 = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_shoe_pairing_probability_l3307_330773


namespace NUMINAMATH_CALUDE_twelve_digit_numbers_with_consecutive_ones_l3307_330787

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_numbers (n : ℕ) : ℕ := 2^n

theorem twelve_digit_numbers_with_consecutive_ones : 
  (valid_numbers 12) - (fibonacci 11) = 3719 := by sorry

end NUMINAMATH_CALUDE_twelve_digit_numbers_with_consecutive_ones_l3307_330787


namespace NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l3307_330704

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l3307_330704


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3307_330723

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 6 * x + 3 = 0 ∧ 
   (k - 1) * y^2 + 6 * y + 3 = 0) ↔ 
  (k < 4 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3307_330723


namespace NUMINAMATH_CALUDE_quadratic_equation_set_equality_l3307_330750

theorem quadratic_equation_set_equality : 
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_set_equality_l3307_330750


namespace NUMINAMATH_CALUDE_trig_identities_l3307_330731

theorem trig_identities (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 ∧
  (5 * Real.cos α ^ 2 - 3 * Real.sin α ^ 2) / (1 + Real.sin α ^ 2) = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3307_330731


namespace NUMINAMATH_CALUDE_line_parallel_plane_transitivity_l3307_330745

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_transitivity 
  (a b : Line) (α : Plane) :
  parallel a α → subset b α → parallel a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_transitivity_l3307_330745


namespace NUMINAMATH_CALUDE_coldness_probability_l3307_330781

def word1 := "CART"
def word2 := "BLEND"
def word3 := "SHOW"
def target_word := "COLDNESS"

def select_letters (word : String) (n : Nat) : Nat := Nat.choose word.length n

theorem coldness_probability :
  let p1 := (1 : ℚ) / select_letters word1 2
  let p2 := (1 : ℚ) / select_letters word2 4
  let p3 := (1 : ℚ) / 2
  p1 * p2 * p3 = 1 / 60 := by sorry

end NUMINAMATH_CALUDE_coldness_probability_l3307_330781


namespace NUMINAMATH_CALUDE_clothing_size_puzzle_l3307_330717

theorem clothing_size_puzzle (anna_size becky_size ginger_size subtracted_number : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - subtracted_number →
  ginger_size = 8 →
  subtracted_number = 4 := by
sorry

end NUMINAMATH_CALUDE_clothing_size_puzzle_l3307_330717


namespace NUMINAMATH_CALUDE_min_sum_of_digits_of_sum_l3307_330760

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem min_sum_of_digits_of_sum (A B : ℕ) 
  (hA : sumOfDigits A = 59) 
  (hB : sumOfDigits B = 77) : 
  ∃ (C : ℕ), C = A + B ∧ sumOfDigits C = 1 ∧ 
  ∀ (D : ℕ), D = A + B → sumOfDigits D ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_of_sum_l3307_330760


namespace NUMINAMATH_CALUDE_extended_midpoint_theorem_l3307_330733

/-- Given two points in 2D space, find the coordinates of a point that is twice as far from their midpoint towards the second point. -/
theorem extended_midpoint_theorem (x₁ y₁ x₂ y₂ : ℚ) :
  let a := (x₁, y₁)
  let b := (x₂, y₂)
  let m := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let p := ((2 * x₂ + x₁) / 3, (2 * y₂ + y₁) / 3)
  (x₁ = 2 ∧ y₁ = 6 ∧ x₂ = 8 ∧ y₂ = 2) →
  p = (7, 8/3) :=
by sorry

end NUMINAMATH_CALUDE_extended_midpoint_theorem_l3307_330733


namespace NUMINAMATH_CALUDE_cubic_roots_property_l3307_330746

/-- 
Given a cubic polynomial x^3 + cx^2 + dx + 16c where c and d are nonzero integers,
if two of its roots coincide and all three roots are integers, then |cd| = 2560.
-/
theorem cubic_roots_property (c d : ℤ) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (∃ p q : ℤ, (∀ x : ℝ, x^3 + c*x^2 + d*x + 16*c = (x - p)^2 * (x - q))) →
  |c*d| = 2560 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_property_l3307_330746


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l3307_330764

def rental_problem (num_dvds : ℕ) (original_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_with_tax := discounted_price * (1 + tax_rate)
  let cost_per_dvd := total_with_tax / num_dvds
  ∃ (rounded_cost : ℚ), 
    rounded_cost = (cost_per_dvd * 100).floor / 100 ∧ 
    rounded_cost = 116 / 100

theorem dvd_rental_cost : 
  rental_problem 4 (480 / 100) (10 / 100) (7 / 100) :=
sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l3307_330764


namespace NUMINAMATH_CALUDE_x_values_when_two_in_M_l3307_330735

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4}

theorem x_values_when_two_in_M (x : ℝ) : 2 ∈ M x → x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_when_two_in_M_l3307_330735


namespace NUMINAMATH_CALUDE_proposition_implication_l3307_330775

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l3307_330775


namespace NUMINAMATH_CALUDE_min_operations_2_to_400_l3307_330719

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.MultiplyTwo => n * 2

/-- Checks if a sequence of operations transforms start into target --/
def transformsTo (start target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation start = target

/-- The minimum number of operations to transform 2 into 400 is 9 --/
theorem min_operations_2_to_400 :
  ∃ (ops : List Operation),
    transformsTo 2 400 ops ∧
    ops.length = 9 ∧
    (∀ (other_ops : List Operation),
      transformsTo 2 400 other_ops → other_ops.length ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_min_operations_2_to_400_l3307_330719


namespace NUMINAMATH_CALUDE_exponent_equality_l3307_330765

theorem exponent_equality (a b c d : ℝ) (x y z q : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(5*z)) 
  (h4 : c^(4*y) = d) : 
  2*x * 5*z = 3*q * 4*y := by
sorry

end NUMINAMATH_CALUDE_exponent_equality_l3307_330765


namespace NUMINAMATH_CALUDE_sherry_age_l3307_330798

theorem sherry_age (randolph_age sydney_age sherry_age : ℕ) : 
  randolph_age = 55 →
  randolph_age = sydney_age + 5 →
  sydney_age = 2 * sherry_age →
  sherry_age = 25 := by sorry

end NUMINAMATH_CALUDE_sherry_age_l3307_330798


namespace NUMINAMATH_CALUDE_value_of_a_l3307_330793

theorem value_of_a (a b c d : ℤ) 
  (eq1 : a = b + 3)
  (eq2 : b = c + 6)
  (eq3 : c = d + 15)
  (eq4 : d = 50) : 
  a = 74 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3307_330793


namespace NUMINAMATH_CALUDE_class_size_from_ratio_and_red_hair_count_l3307_330713

/-- Represents the number of children with each hair color in the ratio --/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Calculates the total parts in the ratio --/
def totalParts (ratio : HairColorRatio) : ℕ :=
  ratio.red + ratio.blonde + ratio.black

/-- Theorem: Given the hair color ratio and number of red-haired children, 
    prove the total number of children in the class --/
theorem class_size_from_ratio_and_red_hair_count 
  (ratio : HairColorRatio) 
  (red_hair_count : ℕ) 
  (h1 : ratio.red = 3) 
  (h2 : ratio.blonde = 6) 
  (h3 : ratio.black = 7) 
  (h4 : red_hair_count = 9) : 
  (red_hair_count * totalParts ratio) / ratio.red = 48 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_ratio_and_red_hair_count_l3307_330713


namespace NUMINAMATH_CALUDE_can_obtain_any_number_l3307_330755

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.append4 => n * 10 + 4
  | Operation.append0 => n * 10
  | Operation.divideBy2 => if n % 2 = 0 then n / 2 else n

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a natural number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- Theorem: Any natural number can be obtained from 4 using the allowed operations -/
theorem can_obtain_any_number : ∀ (n : ℕ), ∃ (seq : OperationSequence), applySequence 4 seq = n := by
  sorry

end NUMINAMATH_CALUDE_can_obtain_any_number_l3307_330755


namespace NUMINAMATH_CALUDE_triangle_side_length_l3307_330740

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.c = 1)           -- AC = 1
  (h2 : t.b = 3)           -- BC = 3
  (h3 : t.A + t.B = π / 3) -- A + B = 60° (in radians)
  : t.a = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3307_330740


namespace NUMINAMATH_CALUDE_range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l3307_330777

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) < 0}

-- Part 1: Range of a when 1 ∈ B
theorem range_of_a_when_one_in_B :
  ∀ a : ℝ, 1 ∈ B a ↔ 0 < a ∧ a < 1 := by sorry

-- Part 2: Range of a when B is a proper subset of A
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ (∃ y : ℝ, y ∈ A ∧ y ∉ B a) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l3307_330777


namespace NUMINAMATH_CALUDE_lassis_from_twenty_fruit_l3307_330732

/-- The number of lassis that can be made given a certain number of fruit units -/
def lassis_from_fruit (fruit_units : ℕ) : ℚ :=
  (9 : ℚ) / 4 * fruit_units

/-- Theorem stating that 45 lassis can be made from 20 fruit units -/
theorem lassis_from_twenty_fruit : lassis_from_fruit 20 = 45 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_twenty_fruit_l3307_330732


namespace NUMINAMATH_CALUDE_smallest_d_divisible_by_11_l3307_330776

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_d (d : ℕ) : ℕ :=
  457000 + d * 100 + 1

theorem smallest_d_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (number_with_d d) → d ≥ 5) ∧
    (is_divisible_by_11 (number_with_d 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_divisible_by_11_l3307_330776


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3307_330705

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3307_330705


namespace NUMINAMATH_CALUDE_point_between_l3307_330724

theorem point_between (a b c : ℚ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end NUMINAMATH_CALUDE_point_between_l3307_330724


namespace NUMINAMATH_CALUDE_expected_correct_guesses_eq_n_l3307_330708

/-- Represents the expected number of correct guesses when drawing balls from an urn -/
def expected_correct_guesses (n : ℕ) : ℝ :=
  n

/-- Theorem stating that the expected number of correct guesses is n -/
theorem expected_correct_guesses_eq_n (n : ℕ) :
  expected_correct_guesses n = n := by sorry

end NUMINAMATH_CALUDE_expected_correct_guesses_eq_n_l3307_330708


namespace NUMINAMATH_CALUDE_unique_number_equality_l3307_330786

theorem unique_number_equality : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_equality_l3307_330786


namespace NUMINAMATH_CALUDE_factorization_equality_l3307_330736

theorem factorization_equality (a b : ℝ) : a^2 * b - b = b * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3307_330736


namespace NUMINAMATH_CALUDE_at_least_one_positive_l3307_330782

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*x + π/2
  let b := y^2 - 2*y + π/3
  let c := z^2 - 2*z + π/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l3307_330782


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l3307_330744

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ

/-- The diagonal length of an isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal length of the specific isosceles trapezoid is 2√52 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 27, base2 := 11, leg := 12 }
  diagonal_length t = 2 * Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l3307_330744


namespace NUMINAMATH_CALUDE_bicentric_shapes_l3307_330785

-- Define the property of being bicentric
def IsBicentric (shape : Type) : Prop :=
  ∃ (circumscribed inscribed : Type), 
    (∀ (s : shape), ∃ (c : circumscribed), True) ∧ 
    (∀ (s : shape), ∃ (i : inscribed), True)

-- Define the shapes
def Square : Type := Unit
def Rectangle : Type := Unit
def RegularPentagon : Type := Unit
def Hexagon : Type := Unit

-- State the theorem
theorem bicentric_shapes :
  IsBicentric Square ∧
  IsBicentric RegularPentagon ∧
  ¬(∀ (r : Rectangle), IsBicentric Rectangle) ∧
  ¬(∀ (h : Hexagon), IsBicentric Hexagon) :=
sorry

end NUMINAMATH_CALUDE_bicentric_shapes_l3307_330785


namespace NUMINAMATH_CALUDE_brown_beads_count_l3307_330721

theorem brown_beads_count (green red taken_out left_in : ℕ) : 
  green = 1 → 
  red = 3 → 
  taken_out = 2 → 
  left_in = 4 → 
  green + red + (taken_out + left_in - (green + red)) = taken_out + left_in → 
  taken_out + left_in - (green + red) = 2 :=
by sorry

end NUMINAMATH_CALUDE_brown_beads_count_l3307_330721


namespace NUMINAMATH_CALUDE_cube_surface_area_l3307_330761

/-- The surface area of a cube with edge length 20 cm is 2400 cm². -/
theorem cube_surface_area : 
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length * edge_length
  surface_area = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3307_330761


namespace NUMINAMATH_CALUDE_opposite_sides_range_l3307_330767

theorem opposite_sides_range (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (2, m)
  let line (x y : ℝ) := x + 2*y - 4
  (line A.1 A.2) * (line B.1 B.2) < 0 → 0 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l3307_330767


namespace NUMINAMATH_CALUDE_fraction_simplification_l3307_330715

theorem fraction_simplification :
  1 / (1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4 + 1 / (1/3)^5) = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3307_330715


namespace NUMINAMATH_CALUDE_minimum_cost_green_plants_l3307_330741

/-- Represents the number of pots of green lily -/
def green_lily_pots : ℕ → Prop :=
  λ x => x ≥ 31 ∧ x ≤ 46

/-- Represents the number of pots of spider plant -/
def spider_plant_pots : ℕ → Prop :=
  λ y => y ≥ 0 ∧ y ≤ 15

/-- The total cost of purchasing the plants -/
def total_cost (x y : ℕ) : ℕ :=
  9 * x + 6 * y

theorem minimum_cost_green_plants :
  ∀ x y : ℕ,
    green_lily_pots x →
    spider_plant_pots y →
    x + y = 46 →
    x ≥ 2 * y →
    total_cost x y ≥ 369 :=
by
  sorry

#check minimum_cost_green_plants

end NUMINAMATH_CALUDE_minimum_cost_green_plants_l3307_330741


namespace NUMINAMATH_CALUDE_inverse_function_of_log_l3307_330788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_of_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a (2 : ℝ) = -1) :
  ∀ x, f⁻¹ a x = (1/2 : ℝ) ^ x :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_of_log_l3307_330788


namespace NUMINAMATH_CALUDE_kaleb_ferris_wheel_cost_l3307_330739

/-- The amount of money Kaleb spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Kaleb spent 27 dollars on the ferris wheel ride -/
theorem kaleb_ferris_wheel_cost :
  ferris_wheel_cost 6 3 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_ferris_wheel_cost_l3307_330739


namespace NUMINAMATH_CALUDE_correct_num_ways_to_purchase_l3307_330751

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of products Charlie and Delta buy together -/
def total_products : ℕ := 4

/-- Represents the purchasing behavior of Charlie and Delta -/
structure PurchasingBehavior where
  charlie_no_repeats : Bool  -- Charlie won't buy more than one of the same flavor
  delta_only_oreos : Bool    -- Delta is interested only in oreos
  delta_at_least_one : Bool  -- Delta must buy at least one oreo
  delta_allows_repeats : Bool -- Delta is open to repeats

/-- The actual purchasing behavior of Charlie and Delta -/
def actual_behavior : PurchasingBehavior := {
  charlie_no_repeats := true,
  delta_only_oreos := true,
  delta_at_least_one := true,
  delta_allows_repeats := true
}

/-- The function to calculate the number of ways to purchase products -/
def num_ways_to_purchase (behavior : PurchasingBehavior) : ℕ :=
  sorry -- The actual calculation would go here

/-- The theorem stating the correct number of ways to purchase -/
theorem correct_num_ways_to_purchase :
  num_ways_to_purchase actual_behavior = 2225 :=
sorry

end NUMINAMATH_CALUDE_correct_num_ways_to_purchase_l3307_330751


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3307_330738

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ :=
  let n : ℕ := 9  -- number of sides in a nonagon
  (n * (n - 3)) / 2

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals :
  num_diagonals_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3307_330738


namespace NUMINAMATH_CALUDE_candy_count_l3307_330707

/-- Given a jar with candy and secret eggs, calculate the number of candy pieces. -/
theorem candy_count (total_items secret_eggs : ℝ) (h1 : total_items = 3554) (h2 : secret_eggs = 145) :
  total_items - secret_eggs = 3409 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3307_330707


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l3307_330748

/-- The sum of the series Σ(n=0 to ∞) of 3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1)) is equal to 1 -/
theorem series_sum_equals_one :
  ∑' n : ℕ, (3 : ℝ) ^ n / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l3307_330748


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3307_330710

/-- Theorem: The quadratic equation x^2 - 6x + 9 = 0 has two equal real roots. -/
theorem quadratic_equal_roots :
  ∃ (x : ℝ), (x^2 - 6*x + 9 = 0) ∧ (∀ y : ℝ, y^2 - 6*y + 9 = 0 → y = x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3307_330710


namespace NUMINAMATH_CALUDE_apollonian_circle_apollonian_circle_specific_case_l3307_330790

/-- The locus of points with a constant ratio of distances to two fixed points is a circle. -/
theorem apollonian_circle (k : ℝ) (hk : k > 0 ∧ k ≠ 1) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = k ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

/-- The specific case where the ratio is 2 and the fixed points are A(-1,0) and B(2,0). -/
theorem apollonian_circle_specific_case :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = 2 ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_apollonian_circle_apollonian_circle_specific_case_l3307_330790


namespace NUMINAMATH_CALUDE_special_function_property_l3307_330706

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y)^2) = f x^2 + 2*x*(f y) + y^2

/-- The number of possible values of f(1) -/
def m (f : ℝ → ℝ) : ℕ := sorry

/-- The sum of all possible values of f(1) -/
def t (f : ℝ → ℝ) : ℝ := sorry

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  (m f : ℝ) * t f = 1 := by sorry

end NUMINAMATH_CALUDE_special_function_property_l3307_330706


namespace NUMINAMATH_CALUDE_cupcakes_eaten_later_is_22_l3307_330703

/-- Represents the cupcake business scenario --/
structure CupcakeBusiness where
  cost_per_cupcake : ℚ
  burnt_cupcakes : ℕ
  perfect_cupcakes : ℕ
  eaten_immediately : ℕ
  made_later : ℕ
  selling_price : ℚ
  net_profit : ℚ

/-- Calculates the number of cupcakes eaten later --/
def cupcakes_eaten_later (business : CupcakeBusiness) : ℚ :=
  let total_made := business.perfect_cupcakes + business.made_later
  let total_cost := (business.burnt_cupcakes + total_made) * business.cost_per_cupcake
  let available_for_sale := total_made - business.eaten_immediately
  ((available_for_sale * business.selling_price - total_cost - business.net_profit) / business.selling_price)

/-- Theorem stating the number of cupcakes eaten later --/
theorem cupcakes_eaten_later_is_22 (business : CupcakeBusiness)
  (h1 : business.cost_per_cupcake = 3/4)
  (h2 : business.burnt_cupcakes = 24)
  (h3 : business.perfect_cupcakes = 24)
  (h4 : business.eaten_immediately = 5)
  (h5 : business.made_later = 24)
  (h6 : business.selling_price = 2)
  (h7 : business.net_profit = 24) :
  cupcakes_eaten_later business = 22 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_later_is_22_l3307_330703


namespace NUMINAMATH_CALUDE_johns_and_brothers_age_sum_l3307_330796

/-- Given that John's age is four less than six times his brother's age,
    and his brother is 8 years old, prove that the sum of their ages is 52. -/
theorem johns_and_brothers_age_sum :
  ∀ (john_age brother_age : ℕ),
    brother_age = 8 →
    john_age = 6 * brother_age - 4 →
    john_age + brother_age = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_and_brothers_age_sum_l3307_330796


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3307_330747

/-- Given a quadratic function y = ax^2 + bx + 3 where a and b are constants and a ≠ 0,
    prove that if (-m,0) and (3m,0) lie on the graph of this function, then b^2 + 4a = 0. -/
theorem quadratic_function_property (a b m : ℝ) (h_a : a ≠ 0) :
  (a * (-m)^2 + b * (-m) + 3 = 0) →
  (a * (3*m)^2 + b * (3*m) + 3 = 0) →
  b^2 + 4*a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3307_330747


namespace NUMINAMATH_CALUDE_davids_running_speed_l3307_330709

/-- Given David's biathlon training conditions, prove the equation for his running speed --/
theorem davids_running_speed 
  (cycle_distance : ℝ) 
  (run_distance : ℝ) 
  (transition_time : ℝ) 
  (total_time : ℝ) 
  (h_cycle_distance : cycle_distance = 30) 
  (h_run_distance : run_distance = 7) 
  (h_transition_time : transition_time = 1/6) 
  (h_total_time : total_time = 170/60) : 
  ∃ x : ℝ, 24 * x^2 - (137/3) * x - 42 = 0 ∧ 
  cycle_distance / (3*x + 2) + run_distance / x = total_time - transition_time :=
by sorry

end NUMINAMATH_CALUDE_davids_running_speed_l3307_330709


namespace NUMINAMATH_CALUDE_caleb_caught_two_trouts_l3307_330727

/-- The number of trouts Caleb caught -/
def caleb_trouts : ℕ := 2

/-- The number of trouts Caleb's dad caught -/
def dad_trouts : ℕ := 3 * caleb_trouts

theorem caleb_caught_two_trouts :
  (dad_trouts = 3 * caleb_trouts) ∧
  (dad_trouts = caleb_trouts + 4) →
  caleb_trouts = 2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_caught_two_trouts_l3307_330727


namespace NUMINAMATH_CALUDE_functional_equation_problem_l3307_330725

theorem functional_equation_problem (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1) :
  ∀ x y : ℝ, g (x + f y) = -x + y - 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l3307_330725


namespace NUMINAMATH_CALUDE_fraction_power_product_specific_fraction_product_l3307_330758

theorem fraction_power_product (a b c d : ℚ) (j : ℕ) :
  (a / b) ^ j * (c / d) ^ j = ((a * c) / (b * d)) ^ j :=
sorry

theorem specific_fraction_product :
  (3 / 4 : ℚ) ^ 3 * (2 / 5 : ℚ) ^ 3 = 27 / 1000 :=
sorry

end NUMINAMATH_CALUDE_fraction_power_product_specific_fraction_product_l3307_330758


namespace NUMINAMATH_CALUDE_negation_of_fraction_inequality_l3307_330749

theorem negation_of_fraction_inequality :
  (¬ ∀ x : ℝ, 1 / (x - 2) < 0) ↔ (∃ x : ℝ, 1 / (x - 2) > 0 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_fraction_inequality_l3307_330749


namespace NUMINAMATH_CALUDE_pentagon_angle_C_l3307_330778

/-- Represents the angles of a pentagon in degrees -/
structure PentagonAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- Defines the properties of the pentagon's angles -/
def is_valid_pentagon (p : PentagonAngles) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧ p.C > 0 ∧ p.D > 0 ∧ p.E > 0 ∧
  p.A < p.B ∧ p.B < p.C ∧ p.C < p.D ∧ p.D < p.E ∧
  p.A + p.B + p.C + p.D + p.E = 540 ∧
  ∃ d : ℝ, d > 0 ∧ 
    p.B - p.A = d ∧
    p.C - p.B = d ∧
    p.D - p.C = d ∧
    p.E - p.D = d

theorem pentagon_angle_C (p : PentagonAngles) 
  (h : is_valid_pentagon p) : p.C = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_C_l3307_330778


namespace NUMINAMATH_CALUDE_number_of_tourists_l3307_330752

theorem number_of_tourists (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end NUMINAMATH_CALUDE_number_of_tourists_l3307_330752


namespace NUMINAMATH_CALUDE_circle_properties_l3307_330720

/-- Given a circle with equation x^2 - 8x - y^2 + 2y = 6, prove its properties. -/
theorem circle_properties :
  let E : Set (ℝ × ℝ) := {p | let (x, y) := p; x^2 - 8*x - y^2 + 2*y = 6}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ E ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c = 4 ∧
    d = 1 ∧
    s^2 = 11 ∧
    c + d + s = 5 + Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3307_330720


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3307_330700

/-- 
Given a parabola y = ax^2 + bx + c with vertex (h, h) and y-intercept (0, -2h), 
where h ≠ 0, the value of b is 6.
-/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x y, y = a * x^2 + b * x + c ↔ y - h = a * (x - h)^2) → 
  c = -2 * h → 
  b = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3307_330700


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3307_330702

theorem product_sum_theorem (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3307_330702


namespace NUMINAMATH_CALUDE_average_first_17_even_numbers_l3307_330737

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_first_17_even_numbers : 
  average (first_n_even_numbers 17) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_first_17_even_numbers_l3307_330737


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3307_330722

theorem quadratic_root_value (v : ℝ) : 
  (8 * ((-26 - Real.sqrt 450) / 10)^2 + 26 * ((-26 - Real.sqrt 450) / 10) + v = 0) → 
  v = 113 / 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3307_330722


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3307_330711

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- The smallest prime divisor of n -/
def s (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of a sequence satisfying the given conditions -/
theorem existence_of_sequence : ∃ (a : ℕ → ℕ), 
  (∀ k ∈ Finset.range 2022, a (k + 1) > a k + 1) ∧ 
  (∀ k ∈ Finset.range 2022, d (a (k + 1) - a k - 1) > 2023^k) ∧
  (∀ k ∈ Finset.range 2022, s (a (k + 1) - a k) > 2023^k) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3307_330711


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3307_330753

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x + 5)) ↔ x ≠ -5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3307_330753


namespace NUMINAMATH_CALUDE_root_of_equation_l3307_330757

theorem root_of_equation (x : ℝ) : 
  (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_of_equation_l3307_330757


namespace NUMINAMATH_CALUDE_work_completion_proof_l3307_330729

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of days it takes with fewer workers -/
def fewer_workers_days : ℕ := 20

/-- The reduction in the number of workers -/
def worker_reduction : ℕ := 10

/-- The original number of workers -/
def original_workers : ℕ := 20

theorem work_completion_proof :
  (original_workers * original_days = (original_workers - worker_reduction) * fewer_workers_days) ∧
  (original_workers > worker_reduction) :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3307_330729


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3307_330714

/-- Represents the probabilities of hitting different rings in a shooting training session -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  sum_to_one : ring10 + ring9 + ring8 + ring7 < 1
  non_negative : ring10 ≥ 0 ∧ ring9 ≥ 0 ∧ ring8 ≥ 0 ∧ ring7 ≥ 0

/-- The probability of hitting either the 10 or 9 ring -/
def prob_10_or_9 (p : ShootingProbabilities) : ℝ := p.ring10 + p.ring9

/-- The probability of scoring less than 7 rings -/
def prob_less_than_7 (p : ShootingProbabilities) : ℝ := 1 - (p.ring10 + p.ring9 + p.ring8 + p.ring7)

theorem shooting_probabilities (p : ShootingProbabilities) 
  (h1 : p.ring10 = 0.21) 
  (h2 : p.ring9 = 0.23) 
  (h3 : p.ring8 = 0.25) 
  (h4 : p.ring7 = 0.28) : 
  prob_10_or_9 p = 0.44 ∧ prob_less_than_7 p = 0.03 := by
  sorry

#eval prob_10_or_9 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩
#eval prob_less_than_7 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_shooting_probabilities_l3307_330714


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3307_330754

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 13
  let total_entries : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_entries : ℕ := odd_numbers * odd_numbers
  (odd_entries : ℚ) / total_entries = 36 / 169 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3307_330754


namespace NUMINAMATH_CALUDE_final_arrangement_decreasing_l3307_330783

/-- Represents a child with a unique height -/
structure Child :=
  (height : ℕ)

/-- Represents a row of children -/
def Row := List Child

/-- The operation of grouping and rearranging children -/
def groupAndRearrange (row : Row) : Row :=
  sorry

/-- Checks if a row is in decreasing order of height -/
def isDecreasingOrder (row : Row) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem final_arrangement_decreasing (n : ℕ) (initial_row : Row) :
  initial_row.length = n →
  (∀ i j, i ≠ j → (initial_row.get i).height ≠ (initial_row.get j).height) →
  isDecreasingOrder ((groupAndRearrange^[n-1]) initial_row) :=
sorry

end NUMINAMATH_CALUDE_final_arrangement_decreasing_l3307_330783


namespace NUMINAMATH_CALUDE_unsold_bag_weights_l3307_330772

def bag_weights : List Nat := [3, 7, 12, 15, 17, 28, 30]

def total_weight : Nat := bag_weights.sum

structure SalesDistribution where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  unsold : Nat

def is_valid_distribution (d : SalesDistribution) : Prop :=
  d.day1 + d.day2 + d.day3 + d.unsold = total_weight ∧
  d.day2 = 2 * d.day1 ∧
  d.day3 = 2 * d.day2 ∧
  d.unsold ∈ bag_weights

theorem unsold_bag_weights :
  ∀ d : SalesDistribution, is_valid_distribution d → d.unsold = 7 ∨ d.unsold = 28 :=
by sorry

end NUMINAMATH_CALUDE_unsold_bag_weights_l3307_330772


namespace NUMINAMATH_CALUDE_special_function_property_l3307_330743

/-- A function satisfying the given property for all real numbers -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, d^2 * g c = c^2 * g d

theorem special_function_property (g : ℝ → ℝ) (h : SpecialFunction g) (h3 : g 3 ≠ 0) :
  (g 6 + g 2) / g 3 = 40/9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l3307_330743


namespace NUMINAMATH_CALUDE_tournament_cycle_l3307_330730

def TournamentGraph := Fin 12 → Fin 12 → Bool

theorem tournament_cycle (g : TournamentGraph) : 
  (∀ i j : Fin 12, i ≠ j → (g i j ≠ g j i) ∧ (g i j ∨ g j i)) →
  (∀ i : Fin 12, ∃ j : Fin 12, g i j) →
  ∃ a b c : Fin 12, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ g a b ∧ g b c ∧ g c a :=
by sorry

end NUMINAMATH_CALUDE_tournament_cycle_l3307_330730


namespace NUMINAMATH_CALUDE_max_d_value_l3307_330792

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ k ∧ ∃ (m : ℕ), m > 0 ∧ d m = k :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3307_330792


namespace NUMINAMATH_CALUDE_right_triangle_with_acute_angle_greater_than_epsilon_l3307_330734

theorem right_triangle_with_acute_angle_greater_than_epsilon :
  ∀ ε : Real, 0 < ε → ε < π / 4 →
  ∃ a b c : ℕ, 
    a * a + b * b = c * c ∧ 
    Real.arctan (min (a / b) (b / a)) > ε :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_acute_angle_greater_than_epsilon_l3307_330734


namespace NUMINAMATH_CALUDE_polygon_diagonal_existence_and_count_l3307_330728

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields here

/-- A diagonal of a polygon -/
structure Diagonal (n : ℕ) (p : Polygon n) where
  -- Add necessary fields here

/-- Predicate to check if a diagonal is inside a polygon -/
def isInside (n : ℕ) (p : Polygon n) (d : Diagonal n p) : Prop :=
  sorry

theorem polygon_diagonal_existence_and_count (n : ℕ) (h : n ≥ 4) (p : Polygon n) :
  (∃ d : Diagonal n p, isInside n p d) ∧
  (∃ k : ℕ, k = n - 3 ∧ 
    (∀ m : ℕ, (∃ diagonals : Finset (Diagonal n p), 
      diagonals.card = m ∧ 
      (∀ d ∈ diagonals, isInside n p d)) → m ≥ k)) :=
sorry

end NUMINAMATH_CALUDE_polygon_diagonal_existence_and_count_l3307_330728


namespace NUMINAMATH_CALUDE_complex_sum_powers_l3307_330762

theorem complex_sum_powers (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l3307_330762


namespace NUMINAMATH_CALUDE_cousins_initial_money_l3307_330771

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_count : ℕ) (give_amount : ℕ) (cousin_initial : ℕ) : Prop :=
  let carmela_final := carmela_initial - (cousin_count * give_amount)
  let cousin_final := cousin_initial + give_amount
  carmela_final = cousin_final

/-- Proves that given the conditions, each cousin must have had $2 initially -/
theorem cousins_initial_money :
  money_distribution 7 4 1 2 := by
  sorry

end NUMINAMATH_CALUDE_cousins_initial_money_l3307_330771


namespace NUMINAMATH_CALUDE_age_problem_l3307_330795

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 27) : 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3307_330795


namespace NUMINAMATH_CALUDE_figure_area_l3307_330742

/-- The total area of a figure composed of four rectangles with given dimensions --/
def total_area (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) : ℕ :=
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width

/-- Theorem stating that the total area of the given figure is 89 square units --/
theorem figure_area : total_area 7 6 2 6 5 4 3 5 = 89 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l3307_330742
