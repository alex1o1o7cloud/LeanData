import Mathlib

namespace NUMINAMATH_CALUDE_black_region_area_l3186_318663

/-- The area of the black region in a square arrangement -/
theorem black_region_area (large_side : ℝ) (small_side1 : ℝ) (small_side2 : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side1 = 4)
  (h3 : small_side2 = 2) :
  large_side^2 - (small_side1^2 + small_side2^2) = 80 := by
  sorry

#check black_region_area

end NUMINAMATH_CALUDE_black_region_area_l3186_318663


namespace NUMINAMATH_CALUDE_jasmine_books_pages_l3186_318624

theorem jasmine_books_pages (books : Set ℕ) 
  (shortest longest middle : ℕ) 
  (h1 : shortest ∈ books) 
  (h2 : longest ∈ books) 
  (h3 : middle ∈ books)
  (h4 : shortest = longest / 4)
  (h5 : middle = 297)
  (h6 : middle = 3 * shortest)
  (h7 : ∀ b ∈ books, b ≤ longest) :
  longest = 396 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_books_pages_l3186_318624


namespace NUMINAMATH_CALUDE_golf_rounds_l3186_318647

theorem golf_rounds (n : ℕ) (average_score : ℚ) (new_score : ℚ) (drop : ℚ) : 
  average_score = 78 →
  new_score = 68 →
  drop = 2 →
  (n * average_score + new_score) / (n + 1) = average_score - drop →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_golf_rounds_l3186_318647


namespace NUMINAMATH_CALUDE_iphone_purchase_savings_l3186_318634

/-- The price of an iPhone X in dollars -/
def iphone_x_price : ℝ := 600

/-- The price of an iPhone Y in dollars -/
def iphone_y_price : ℝ := 800

/-- The discount rate for buying at least 2 smartphones of the same model -/
def same_model_discount : ℝ := 0.05

/-- The discount rate for mixed purchases of at least 3 smartphones -/
def mixed_purchase_discount : ℝ := 0.03

/-- The total cost of buying three iPhones individually -/
def individual_cost : ℝ := 2 * iphone_x_price + iphone_y_price

/-- The discounted price of two iPhone X models -/
def discounted_iphone_x : ℝ := 2 * (iphone_x_price * (1 - same_model_discount))

/-- The discounted price of one iPhone Y model -/
def discounted_iphone_y : ℝ := iphone_y_price * (1 - mixed_purchase_discount)

/-- The total cost of buying three iPhones together with discounts -/
def group_cost : ℝ := discounted_iphone_x + discounted_iphone_y

/-- The savings from buying three iPhones together vs. individually -/
def savings : ℝ := individual_cost - group_cost

theorem iphone_purchase_savings : savings = 84 := by sorry

end NUMINAMATH_CALUDE_iphone_purchase_savings_l3186_318634


namespace NUMINAMATH_CALUDE_work_completion_time_l3186_318670

/-- 
Given that:
- A does 20% less work than B per unit time
- A completes the work in 7.5 hours
Prove that B completes the same work in 6 hours
-/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) 
  (h1 : work_rate_A = 0.8 * work_rate_B) 
  (h2 : work_rate_A * 7.5 = work_rate_B * 6) : 
  work_rate_B * 6 = work_rate_A * 7.5 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3186_318670


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3186_318657

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3186_318657


namespace NUMINAMATH_CALUDE_total_short_trees_correct_park_short_trees_after_planting_l3186_318611

/-- Calculates the total number of short trees after planting -/
def total_short_trees (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Proves that the total number of short trees after planting is correct -/
theorem total_short_trees_correct (initial_short_trees planted_short_trees : ℕ) :
  total_short_trees initial_short_trees planted_short_trees = initial_short_trees + planted_short_trees :=
by sorry

/-- Proves that the specific case in the problem is correct -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 :=
by sorry

end NUMINAMATH_CALUDE_total_short_trees_correct_park_short_trees_after_planting_l3186_318611


namespace NUMINAMATH_CALUDE_min_max_sum_bound_l3186_318679

theorem min_max_sum_bound (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  ∃ (x : ℝ), x ≥ 1/3 ∧ 
    (∀ y, y = max (a+b+c) (max (b+c+d) (max (c+d+e) (max (d+e+f) (e+f+g)))) → y ≤ x) ∧
    (∃ (a' b' c' d' e' f' g' : ℝ),
      a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ e' ≥ 0 ∧ f' ≥ 0 ∧ g' ≥ 0 ∧
      a' + b' + c' + d' + e' + f' + g' = 1 ∧
      max (a'+b'+c') (max (b'+c'+d') (max (c'+d'+e') (max (d'+e'+f') (e'+f'+g')))) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_bound_l3186_318679


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3186_318682

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  property : a 2 + 4 * a 7 + a 12 = 96
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3186_318682


namespace NUMINAMATH_CALUDE_streaming_bill_fixed_fee_l3186_318625

/-- Represents the billing structure for a streaming service -/
structure StreamingBill where
  fixedFee : ℝ
  movieCharge : ℝ

/-- Calculates the total bill given number of movies watched -/
def StreamingBill.totalBill (bill : StreamingBill) (movies : ℝ) : ℝ :=
  bill.fixedFee + bill.movieCharge * movies

theorem streaming_bill_fixed_fee (bill : StreamingBill) :
  bill.totalBill 1 = 15.30 →
  bill.totalBill 1.5 = 20.55 →
  bill.fixedFee = 4.80 := by
  sorry

end NUMINAMATH_CALUDE_streaming_bill_fixed_fee_l3186_318625


namespace NUMINAMATH_CALUDE_triangle_properties_l3186_318673

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  2 * c * Real.cos B = 2 * a - b →
  -- Prove C = π/3
  C = π / 3 ∧
  -- When c = 3, prove a + b is in the range (3, 6]
  (c = 3 → 3 < a + b ∧ a + b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3186_318673


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l3186_318630

-- Define the function f(x) = x³ - 2x² + 2
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

-- Theorem statement
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) (-1/2 : ℝ), f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_f_has_root_in_interval_l3186_318630


namespace NUMINAMATH_CALUDE_divide_seven_friends_four_teams_l3186_318613

/-- The number of ways to divide n friends among k teams -/
def divideFriends (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Dividing 7 friends among 4 teams results in 16384 ways -/
theorem divide_seven_friends_four_teams : 
  divideFriends 7 4 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_divide_seven_friends_four_teams_l3186_318613


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3186_318615

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3186_318615


namespace NUMINAMATH_CALUDE_divisor_sum_360_l3186_318629

/-- Sum of positive divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem divisor_sum_360 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 360 → i + j + k = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_360_l3186_318629


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l3186_318699

theorem fruit_basket_problem (total_fruits : ℕ) 
  (basket_A basket_B : ℕ) 
  (apples_A pears_A apples_B pears_B : ℕ) :
  total_fruits = 82 →
  (basket_A + basket_B = total_fruits) →
  (basket_A ≥ basket_B → basket_A - basket_B < 10) →
  (basket_B > basket_A → basket_B - basket_A < 10) →
  (5 * apples_A = 2 * basket_A) →
  (7 * pears_B = 4 * basket_B) →
  (basket_A = apples_A + pears_A) →
  (basket_B = apples_B + pears_B) →
  (pears_A = 24 ∧ apples_B = 18) :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l3186_318699


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3186_318610

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3186_318610


namespace NUMINAMATH_CALUDE_student_contribution_l3186_318641

/-- Proves that if 30 students contribute equally every Friday for 2 months (8 Fridays)
    and collect a total of $480, then each student's contribution per Friday is $2. -/
theorem student_contribution
  (num_students : ℕ)
  (num_fridays : ℕ)
  (total_amount : ℕ)
  (h1 : num_students = 30)
  (h2 : num_fridays = 8)
  (h3 : total_amount = 480) :
  total_amount / (num_students * num_fridays) = 2 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l3186_318641


namespace NUMINAMATH_CALUDE_root_product_equals_27_l3186_318636

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l3186_318636


namespace NUMINAMATH_CALUDE_average_of_data_l3186_318617

def data : List ℝ := [2, 5, 5, 6, 7]

theorem average_of_data : (data.sum / data.length : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l3186_318617


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3186_318659

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3186_318659


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3186_318652

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3186_318652


namespace NUMINAMATH_CALUDE_decimal_rep_denominators_num_possible_denominators_l3186_318639

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the decimal representation
def DecimalRep (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℕ) / 999

-- Define the condition that not all digits are nine
def NotAllNine (a b c : Digit) : Prop :=
  ¬(a.val = 9 ∧ b.val = 9 ∧ c.val = 9)

-- Define the condition that not all digits are zero
def NotAllZero (a b c : Digit) : Prop :=
  ¬(a.val = 0 ∧ b.val = 0 ∧ c.val = 0)

-- Define the set of possible denominators
def PossibleDenominators : Finset ℕ :=
  {3, 9, 27, 37, 111, 333, 999}

-- The main theorem
theorem decimal_rep_denominators (a b c : Digit) 
  (h1 : NotAllNine a b c) (h2 : NotAllZero a b c) :
  (DecimalRep a b c).den ∈ PossibleDenominators := by
  sorry

-- The final result
theorem num_possible_denominators :
  Finset.card PossibleDenominators = 7 := by
  sorry

end NUMINAMATH_CALUDE_decimal_rep_denominators_num_possible_denominators_l3186_318639


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l3186_318651

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 - x + 1)
  (h2 : f 2 = 9) :
  f (-2) = 13 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l3186_318651


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l3186_318612

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, (x + 2) / (x - 1) = y) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l3186_318612


namespace NUMINAMATH_CALUDE_incorrect_elimination_process_l3186_318619

/-- Given a system of two linear equations in two variables, 
    prove that a specific elimination process is incorrect. -/
theorem incorrect_elimination_process 
  (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  ¬ (∃ (k : ℝ), 2 * a + b + 2 * (a - b) = 7 + 2 * k ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_incorrect_elimination_process_l3186_318619


namespace NUMINAMATH_CALUDE_second_digit_of_n_l3186_318696

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n ∧ 8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102 ∧ 81*n - 102 < 10^102) →
  ∃ k : ℕ, 12 * 10^97 ≤ n ∧ n < 13 * 10^97 ∧ n = 2 * 10^97 + k :=
by sorry

end NUMINAMATH_CALUDE_second_digit_of_n_l3186_318696


namespace NUMINAMATH_CALUDE_larger_number_proof_l3186_318697

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 40) (diff_eq : x - y = 10) :
  max x y = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3186_318697


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3186_318609

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3186_318609


namespace NUMINAMATH_CALUDE_max_product_of_two_different_numbers_exists_max_product_l3186_318620

def S : Set Int := {-9, -5, -3, 0, 4, 5, 8}

theorem max_product_of_two_different_numbers (a b : Int) :
  a ∈ S → b ∈ S → a ≠ b → a * b ≤ 45 := by
  sorry

theorem exists_max_product :
  ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_two_different_numbers_exists_max_product_l3186_318620


namespace NUMINAMATH_CALUDE_centroid_unique_point_l3186_318631

/-- Definition of a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of the centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- Definition of a point being inside or on the boundary of a triangle -/
def insideOrOnBoundary (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the centroid is the unique point satisfying the condition -/
theorem centroid_unique_point (t : Triangle) :
  ∃! M, insideOrOnBoundary M t ∧
    ∀ N, insideOrOnBoundary N t →
      ∃ P, insideOrOnBoundary P t ∧
        area (Triangle.mk M N P) ≥ (1/6 : ℝ) * area t :=
  sorry

end NUMINAMATH_CALUDE_centroid_unique_point_l3186_318631


namespace NUMINAMATH_CALUDE_cubic_double_root_value_l3186_318645

theorem cubic_double_root_value (a b : ℝ) (p q r : ℝ) : 
  (∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a)^2 * (x - b)) →
  p = -6 →
  q = 9 →
  r = 0 ∨ r = -4 :=
sorry

end NUMINAMATH_CALUDE_cubic_double_root_value_l3186_318645


namespace NUMINAMATH_CALUDE_max_ratio_squared_l3186_318656

theorem max_ratio_squared (c d x y : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c ≥ d)
  (heq : c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x)^2 + (d - y)^2)
  (hx : 0 ≤ x ∧ x < c) (hy : 0 ≤ y ∧ y < d) :
  (c / d)^2 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l3186_318656


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3186_318654

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, f x + (deriv f) x > 1) (h2 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3186_318654


namespace NUMINAMATH_CALUDE_tenth_term_is_144_l3186_318690

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem tenth_term_is_144 : fibonacci_like_sequence 9 = 144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_144_l3186_318690


namespace NUMINAMATH_CALUDE_power_multiplication_l3186_318695

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3186_318695


namespace NUMINAMATH_CALUDE_union_of_complements_is_certain_l3186_318650

-- Define the sample space
variable {Ω : Type}

-- Define events as sets of outcomes
variable (A B C D : Set Ω)

-- Define the properties of events
variable (h1 : A ∩ B = ∅)  -- A and B are mutually exclusive
variable (h2 : C = Aᶜ)     -- C is the complement of A
variable (h3 : D = Bᶜ)     -- D is the complement of B

-- Theorem statement
theorem union_of_complements_is_certain : C ∪ D = univ := by
  sorry

end NUMINAMATH_CALUDE_union_of_complements_is_certain_l3186_318650


namespace NUMINAMATH_CALUDE_sin_difference_quotient_zero_l3186_318687

theorem sin_difference_quotient_zero (x y : ℝ) 
  (hx : Real.tan x = x) 
  (hy : Real.tan y = y) 
  (hxy : |x| ≠ |y|) : 
  (Real.sin (x + y)) / (x + y) - (Real.sin (x - y)) / (x - y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_quotient_zero_l3186_318687


namespace NUMINAMATH_CALUDE_circular_sector_rotation_l3186_318607

/-- Given a circular sector rotated about one of its boundary radii, 
    if the spherical surface area of the resulting solid is equal to its conical surface area, 
    then the sine of the central angle of the circular sector is 4/5. -/
theorem circular_sector_rotation (α : Real) : 
  (∃ R : Real, R > 0 ∧ π * R^2 * Real.sin α = 2 * π * R^2 * (1 - Real.cos α)) → 
  Real.sin α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_circular_sector_rotation_l3186_318607


namespace NUMINAMATH_CALUDE_dot_product_problem_l3186_318646

theorem dot_product_problem (a b : ℝ × ℝ) : 
  a = (2, 1) → a - b = (-1, 2) → a • b = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l3186_318646


namespace NUMINAMATH_CALUDE_even_function_four_roots_sum_zero_l3186_318640

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four real roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d

/-- The theorem stating that for an even function with exactly four real roots, the sum of its roots is zero -/
theorem even_function_four_roots_sum_zero (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_four_roots : HasFourRealRoots f) :
    ∃ (a b c d : ℝ), HasFourRealRoots f ∧ a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_four_roots_sum_zero_l3186_318640


namespace NUMINAMATH_CALUDE_new_person_weight_is_106_l3186_318678

/-- The number of persons in the initial group -/
def initial_group_size : ℕ := 12

/-- The increase in average weight when the new person joins (in kg) -/
def average_weight_increase : ℝ := 4

/-- The weight of the person being replaced (in kg) -/
def replaced_person_weight : ℝ := 58

/-- The weight of the new person (in kg) -/
def new_person_weight : ℝ := 106

/-- Theorem stating that the weight of the new person is 106 kg -/
theorem new_person_weight_is_106 :
  new_person_weight = replaced_person_weight + initial_group_size * average_weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_is_106_l3186_318678


namespace NUMINAMATH_CALUDE_problem_3_l3186_318677

theorem problem_3 (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a^2 - 8 * a + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l3186_318677


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3186_318691

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3186_318691


namespace NUMINAMATH_CALUDE_seokjin_floors_to_bookstore_l3186_318621

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := 4

/-- Seokjin's current floor number -/
def current_floor : ℕ := 1

/-- The number of floors Seokjin must go up -/
def floors_to_go_up : ℕ := bookstore_floor - current_floor

/-- Theorem stating that Seokjin must go up 3 floors to reach the bookstore -/
theorem seokjin_floors_to_bookstore : floors_to_go_up = 3 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_floors_to_bookstore_l3186_318621


namespace NUMINAMATH_CALUDE_sophias_book_length_l3186_318693

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (total_pages / 2 : ℚ) + 45 →
  total_pages = 4556 :=
by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l3186_318693


namespace NUMINAMATH_CALUDE_nested_fraction_equation_l3186_318632

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225 / 73 → x = -647 / 177 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equation_l3186_318632


namespace NUMINAMATH_CALUDE_max_valid_rectangles_l3186_318635

/-- Represents a grid with dimensions and unit square size -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (unit_size : ℕ)

/-- Represents a coloring of the grid -/
def Coloring := Grid → Fin 2

/-- Represents a cutting of the grid into rectangles -/
def Cutting := Grid → List (ℕ × ℕ)

/-- Counts the number of rectangles with at most one black square -/
def count_valid_rectangles (g : Grid) (c : Coloring) (cut : Cutting) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of valid rectangles -/
theorem max_valid_rectangles (g : Grid) 
  (h1 : g.width = 2020)
  (h2 : g.height = 2020)
  (h3 : g.unit_size = 11)
  (h4 : g.width / g.unit_size * (g.height / g.unit_size) = 400) :
  ∃ (c : Coloring) (cut : Cutting), 
    ∀ (c' : Coloring) (cut' : Cutting), 
      count_valid_rectangles g c cut ≥ count_valid_rectangles g c' cut' ∧ 
      count_valid_rectangles g c cut = 20 :=
sorry

end NUMINAMATH_CALUDE_max_valid_rectangles_l3186_318635


namespace NUMINAMATH_CALUDE_motorbike_time_difference_l3186_318661

theorem motorbike_time_difference :
  let distance : ℝ := 960
  let speed_slow : ℝ := 60
  let speed_fast : ℝ := 64
  let time_slow : ℝ := distance / speed_slow
  let time_fast : ℝ := distance / speed_fast
  time_slow - time_fast = 1 := by
  sorry

end NUMINAMATH_CALUDE_motorbike_time_difference_l3186_318661


namespace NUMINAMATH_CALUDE_lunks_needed_for_dozen_apples_l3186_318648

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks : ℚ := 4 / 7

/-- Exchange rate between kunks and apples -/
def kunks_to_apples : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 12

/-- Theorem stating the number of lunks needed to buy 12 apples -/
theorem lunks_needed_for_dozen_apples :
  ⌈(apples_to_buy : ℚ) / kunks_to_apples / lunks_to_kunks⌉ = 14 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_dozen_apples_l3186_318648


namespace NUMINAMATH_CALUDE_square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l3186_318638

theorem square_of_arithmetic_mean_le_arithmetic_mean_of_squares
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b) / 2) ^ 2 ≤ (a ^ 2 + b ^ 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l3186_318638


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l3186_318662

/-- A function f(x) = 2ax² - x - 1 has exactly one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0

/-- The theorem stating that if f(x) = 2ax² - x - 1 has exactly one zero in (0, 1), 
    then a is in the interval (1, +∞) -/
theorem unique_zero_implies_a_range :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l3186_318662


namespace NUMINAMATH_CALUDE_f_symmetry_l3186_318606

/-- A function f(x) = ax^7 + bx - 2 where a and b are real numbers -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x - 2

/-- If f(2009) = 10, then f(-2009) = -14 -/
theorem f_symmetry (a b : ℝ) : f a b 2009 = 10 → f a b (-2009) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3186_318606


namespace NUMINAMATH_CALUDE_wilted_flowers_count_l3186_318666

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (remaining_bouquets : ℕ) : ℕ :=
  initial_flowers - (remaining_bouquets * flowers_per_bouquet)

theorem wilted_flowers_count :
  flower_problem 53 7 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wilted_flowers_count_l3186_318666


namespace NUMINAMATH_CALUDE_inequality_solution_l3186_318604

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 48 -/
theorem inequality_solution (a b c : ℝ) : 
  (∀ x, ((x - a) * (x - b)) / (x - c) ≥ 0 ↔ (x < -6 ∨ (20 ≤ x ∧ x ≤ 23))) →
  a < b →
  a + 2*b + 3*c = 48 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3186_318604


namespace NUMINAMATH_CALUDE_total_goals_is_fifteen_l3186_318649

def soccer_match_goals : ℕ := by
  -- Define the goals scored by The Kickers in the first period
  let kickers_first_period : ℕ := 2

  -- Define the goals scored by The Kickers in the second period
  let kickers_second_period : ℕ := 2 * kickers_first_period

  -- Define the goals scored by The Spiders in the first period
  let spiders_first_period : ℕ := kickers_first_period / 2

  -- Define the goals scored by The Spiders in the second period
  let spiders_second_period : ℕ := 2 * kickers_second_period

  -- Calculate the total goals
  let total_goals : ℕ := kickers_first_period + kickers_second_period + 
                         spiders_first_period + spiders_second_period

  -- Prove that the total goals equal 15
  have : total_goals = 15 := by sorry

  exact total_goals

-- Theorem stating that the total number of goals is 15
theorem total_goals_is_fifteen : soccer_match_goals = 15 := by sorry

end NUMINAMATH_CALUDE_total_goals_is_fifteen_l3186_318649


namespace NUMINAMATH_CALUDE_hobby_gender_independence_l3186_318660

/-- Represents the contingency table data -/
structure ContingencyTable where
  total : ℕ
  male_hobby : ℕ
  female_no_hobby : ℕ

/-- Calculates the chi-square value for the independence test -/
def chi_square (ct : ContingencyTable) : ℝ :=
  sorry

/-- Calculates the probability of selecting k males from those without a hobby -/
def prob_select_males (ct : ContingencyTable) (k : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of males selected -/
def expected_males (ct : ContingencyTable) : ℚ :=
  sorry

/-- Main theorem encompassing all parts of the problem -/
theorem hobby_gender_independence (ct : ContingencyTable) 
  (h1 : ct.total = 100) 
  (h2 : ct.male_hobby = 30) 
  (h3 : ct.female_no_hobby = 10) : 
  chi_square ct < 6.635 ∧ 
  prob_select_males ct 0 = 3/29 ∧ 
  prob_select_males ct 1 = 40/87 ∧ 
  prob_select_males ct 2 = 38/87 ∧
  expected_males ct = 4/3 :=
sorry

end NUMINAMATH_CALUDE_hobby_gender_independence_l3186_318660


namespace NUMINAMATH_CALUDE_express_w_in_terms_of_abc_l3186_318605

/-- Given distinct real numbers and a system of equations, prove the expression for w -/
theorem express_w_in_terms_of_abc (a b c w : ℝ) (x y z : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ w ≠ a ∧ w ≠ b ∧ w ≠ c ∧ 157 ≠ w ∧ 157 ≠ a ∧ 157 ≠ b ∧ 157 ≠ c)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  (a*b + a*c + b*c = 0 → w = -a*b/(a+b)) ∧ 
  (a*b + a*c + b*c ≠ 0 → w = -a*b*c/(a*b + a*c + b*c)) :=
sorry

end NUMINAMATH_CALUDE_express_w_in_terms_of_abc_l3186_318605


namespace NUMINAMATH_CALUDE_find_k_l3186_318684

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem find_k (k : ℤ) : 
  k % 2 = 1 → f (f (f k)) = 27 → k = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3186_318684


namespace NUMINAMATH_CALUDE_goods_train_length_goods_train_length_approx_280m_l3186_318694

/-- The length of a goods train passing a passenger train in opposite directions --/
theorem goods_train_length (v_passenger : ℝ) (v_goods : ℝ) (t_pass : ℝ) : ℝ :=
  let v_relative : ℝ := v_passenger + v_goods
  let v_relative_ms : ℝ := v_relative * 1000 / 3600
  let length : ℝ := v_relative_ms * t_pass
  by
    -- Proof goes here
    sorry

/-- The length of the goods train is approximately 280 meters --/
theorem goods_train_length_approx_280m :
  ∃ ε > 0, |goods_train_length 70 42 9 - 280| < ε :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_goods_train_length_goods_train_length_approx_280m_l3186_318694


namespace NUMINAMATH_CALUDE_function_inequality_l3186_318675

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x : ℝ, f x < (deriv^[2] f) x) : 
  (Real.exp 2019 * f (-2019) < f 0) ∧ (f 2019 > Real.exp 2019 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3186_318675


namespace NUMINAMATH_CALUDE_remaining_cents_l3186_318608

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of half a dollar in cents -/
def half_dollar : ℕ := 50

/-- The number of quarters Winston has initially -/
def initial_quarters : ℕ := 14

/-- Theorem: Given 14 quarters and spending half a dollar, the remaining amount in cents is 300 -/
theorem remaining_cents : 
  initial_quarters * quarter_value - half_dollar = 300 := by sorry

end NUMINAMATH_CALUDE_remaining_cents_l3186_318608


namespace NUMINAMATH_CALUDE_x_range_proof_l3186_318689

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem x_range_proof (f : ℝ → ℝ) (h_odd : is_odd f) (h_decreasing : is_decreasing f)
  (h_domain : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ∈ Set.Icc (-3 : ℝ) 3)
  (h_inequality : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f (x^2 - 2*x) + f (x - 2) < 0) :
  ∀ x, x ∈ Set.Ioc (2 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_x_range_proof_l3186_318689


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l3186_318692

/-- Represents a configuration of lit buttons on a 3 × 2 grid -/
def ButtonGrid := Fin 3 → Fin 2 → Bool

/-- Returns true if at least one button in the grid is lit -/
def atLeastOneLit (grid : ButtonGrid) : Prop :=
  ∃ i j, grid i j = true

/-- Two grids are equivalent if one can be obtained from the other by translation -/
def equivalentGrids (grid1 grid2 : ButtonGrid) : Prop :=
  sorry

/-- The number of distinct observable arrangements -/
def distinctArrangements : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct arrangements is 44 -/
theorem distinct_arrangements_count :
  distinctArrangements = 44 :=
sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l3186_318692


namespace NUMINAMATH_CALUDE_sin_difference_l3186_318667

theorem sin_difference (A B : Real) 
  (h1 : Real.tan A = 2 * Real.tan B) 
  (h2 : Real.sin (A + B) = 1/4) : 
  Real.sin (A - B) = 1/12 := by
sorry

end NUMINAMATH_CALUDE_sin_difference_l3186_318667


namespace NUMINAMATH_CALUDE_train_crossing_time_l3186_318618

-- Define the given values
def train_length : Real := 210  -- meters
def train_speed : Real := 25    -- km/h
def man_speed : Real := 2       -- km/h

-- Define the theorem
theorem train_crossing_time :
  let relative_speed : Real := train_speed + man_speed
  let relative_speed_mps : Real := relative_speed * 1000 / 3600
  let time : Real := train_length / relative_speed_mps
  time = 28 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l3186_318618


namespace NUMINAMATH_CALUDE_equation_solutions_l3186_318686

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 10*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 12*x - 8)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3186_318686


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3186_318683

theorem polynomial_factorization (x : ℤ) : 
  x^5 + x^4 + 1 = (x^3 - x + 1) * (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3186_318683


namespace NUMINAMATH_CALUDE_sine_transformation_l3186_318688

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * x + (2 * Real.pi) / 3) = Real.sin (2 * (x + Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sine_transformation_l3186_318688


namespace NUMINAMATH_CALUDE_vector_at_negative_seven_l3186_318681

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector at t = 0 -/
  a : ℝ × ℝ
  /-- The direction vector of the line -/
  d : ℝ × ℝ

/-- Get the vector on the line at a given t -/
def ParametricLine.vectorAt (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

/-- The main theorem -/
theorem vector_at_negative_seven
  (line : ParametricLine)
  (h1 : line.vectorAt 2 = (1, 4))
  (h2 : line.vectorAt 3 = (3, -4)) :
  line.vectorAt (-7) = (-17, 76) := by
  sorry


end NUMINAMATH_CALUDE_vector_at_negative_seven_l3186_318681


namespace NUMINAMATH_CALUDE_oldest_daughter_ages_l3186_318601

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 168

def has_ambiguous_sum (a b c : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_valid_triple x y z ∧ 
  x + y + z = a + b + c ∧ (x ≠ a ∨ y ≠ b ∨ z ≠ c)

theorem oldest_daughter_ages :
  ∀ (a b c : ℕ), is_valid_triple a b c → has_ambiguous_sum a b c →
  (max a (max b c) = 12 ∨ max a (max b c) = 14 ∨ max a (max b c) = 21) :=
by sorry

end NUMINAMATH_CALUDE_oldest_daughter_ages_l3186_318601


namespace NUMINAMATH_CALUDE_f_monotonicity_l3186_318669

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - 1 / x

theorem f_monotonicity (k : ℝ) :
  (∀ x > 0, HasDerivAt (f k) ((k * x + 1) / (x^2)) x) →
  (k ≥ 0 → ∀ x > 0, (k * x + 1) / (x^2) > 0) ∧
  (k < 0 → (∀ x, 0 < x ∧ x < -1/k → (k * x + 1) / (x^2) > 0) ∧
           (∀ x > -1/k, (k * x + 1) / (x^2) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_l3186_318669


namespace NUMINAMATH_CALUDE_representation_of_2015_l3186_318664

theorem representation_of_2015 : ∃ (a b c : ℤ),
  a + b + c = 2015 ∧
  Nat.Prime a.natAbs ∧
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
by sorry

end NUMINAMATH_CALUDE_representation_of_2015_l3186_318664


namespace NUMINAMATH_CALUDE_water_flow_restrictor_l3186_318600

theorem water_flow_restrictor (original_rate : ℝ) (reduced_rate : ℝ) : 
  original_rate = 5 →
  reduced_rate = 0.6 * original_rate - 1 →
  reduced_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_water_flow_restrictor_l3186_318600


namespace NUMINAMATH_CALUDE_inequality_transformation_l3186_318658

theorem inequality_transformation (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l3186_318658


namespace NUMINAMATH_CALUDE_class_size_calculation_l3186_318626

theorem class_size_calculation (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 13 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 26 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3186_318626


namespace NUMINAMATH_CALUDE_circle_equation_l3186_318637

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
theorem circle_equation (x y : ℝ) : 
  let h : ℝ := 1
  let k : ℝ := 2
  let r : ℝ := 5
  (x - h)^2 + (y - k)^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l3186_318637


namespace NUMINAMATH_CALUDE_dvd_price_percentage_l3186_318616

theorem dvd_price_percentage (srp : ℝ) (h1 : srp > 0) : 
  let marked_price := 0.6 * srp
  let bob_price := 0.4 * marked_price
  bob_price / srp = 0.24 := by
sorry

end NUMINAMATH_CALUDE_dvd_price_percentage_l3186_318616


namespace NUMINAMATH_CALUDE_perfect_linear_relationship_l3186_318602

-- Define a scatter plot as a list of points
def ScatterPlot := List (ℝ × ℝ)

-- Define a function to check if all points lie on a straight line
def allPointsOnLine (plot : ScatterPlot) : Prop := sorry

-- Define residuals
def residuals (plot : ScatterPlot) : List ℝ := sorry

-- Define sum of squares of residuals
def sumSquaresResiduals (plot : ScatterPlot) : ℝ := sorry

-- Define correlation coefficient
def correlationCoefficient (plot : ScatterPlot) : ℝ := sorry

-- Theorem statement
theorem perfect_linear_relationship (plot : ScatterPlot) :
  allPointsOnLine plot →
  (∀ r ∈ residuals plot, r = 0) ∧
  sumSquaresResiduals plot = 0 ∧
  |correlationCoefficient plot| = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linear_relationship_l3186_318602


namespace NUMINAMATH_CALUDE_fraction_product_equality_l3186_318674

theorem fraction_product_equality : (2/3)^4 * (1/5) * (3/4) = 4/135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l3186_318674


namespace NUMINAMATH_CALUDE_min_a_correct_l3186_318672

/-- The number of cards in the deck -/
def n : ℕ := 51

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a and a+7, and Dylan picks the other -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose (n - 2) 2

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 22

theorem min_a_correct :
  (∀ a : ℕ, 1 ≤ a ∧ a + 7 ≤ n → p a ≥ 1/2 → a ≥ min_a) ∧
  p min_a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_correct_l3186_318672


namespace NUMINAMATH_CALUDE_grandmothers_current_age_l3186_318622

/-- The age of Minyoung's grandmother this year, given that Minyoung is 7 years old this year
    and her grandmother turns 65 when Minyoung turns 10. -/
def grandmothers_age (minyoung_age : ℕ) (grandmother_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  grandmother_future_age - years_until_future

/-- Proof that Minyoung's grandmother is 62 years old this year. -/
theorem grandmothers_current_age :
  grandmothers_age 7 65 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_current_age_l3186_318622


namespace NUMINAMATH_CALUDE_binomial_probability_two_l3186_318603

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_two (X : ℕ → ℝ) :
  (∀ k, X k = binomial_pmf 6 (1/3) k) →
  X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_l3186_318603


namespace NUMINAMATH_CALUDE_horner_rule_v₃_l3186_318614

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℤ) : ℤ := (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (x : ℤ) : ℤ :=
  let v₀ := 1
  let v₁ := x - 5 * v₀
  let v₂ := x * v₁ + 6
  x * v₂ + 0

theorem horner_rule_v₃ :
  v₃ (-2) = -40 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v₃_l3186_318614


namespace NUMINAMATH_CALUDE_hexagram_arrangement_count_l3186_318642

/-- A hexagram is a regular six-pointed star with 12 points of intersection -/
structure Hexagram :=
  (points : Fin 12 → α)

/-- The number of symmetries of a hexagram (rotations and reflections) -/
def hexagram_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a hexagram,
    considering rotations and reflections as equivalent -/
def distinct_hexagram_arrangements : ℕ := Nat.factorial 12 / hexagram_symmetries

theorem hexagram_arrangement_count :
  distinct_hexagram_arrangements = 39916800 := by sorry

end NUMINAMATH_CALUDE_hexagram_arrangement_count_l3186_318642


namespace NUMINAMATH_CALUDE_monotonic_decreasing_domain_l3186_318685

/-- A monotonically decreasing function on (0, +∞) satisfying f(x) < f(2x - 2) has domain (1, 2) -/
theorem monotonic_decreasing_domain (f : ℝ → ℝ) :
  (∀ x y, 0 < x ∧ x < y → f y < f x) →  -- monotonically decreasing on (0, +∞)
  (∀ x, 0 < x → f x < f (2*x - 2)) →   -- condition f(x) < f(2x - 2)
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ∃ y, 0 < y ∧ f y < f (2*y - 2)} := by
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_domain_l3186_318685


namespace NUMINAMATH_CALUDE_not_p_sufficient_but_not_necessary_for_not_q_l3186_318633

-- Define the conditions p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for one condition to be sufficient but not necessary for another
def sufficient_but_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_p_sufficient_but_not_necessary_for_not_q :
  sufficient_but_not_necessary (¬∃ x, p x) (¬∃ x, q x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_but_not_necessary_for_not_q_l3186_318633


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3186_318628

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) ↔ (m ≤ -2 ∨ m ≥ -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3186_318628


namespace NUMINAMATH_CALUDE_percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l3186_318680

theorem percentage_chain (n : ℝ) : n * 0.5 * 0.3 * 0.1 = n * 0.015 := by sorry

theorem ten_percent_of_thirty_percent_of_fifty_percent_of_7000 :
  7000 * 0.5 * 0.3 * 0.1 = 105 := by sorry

end NUMINAMATH_CALUDE_percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l3186_318680


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3186_318653

theorem arithmetic_progression_sum (x y z d k : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  y * (z - x) - x * (y - z) = d ∧
  z * (x - y) - y * (z - x) = d ∧
  x * (y - z) + y * (z - x) + z * (x - y) = k
  → d = k / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3186_318653


namespace NUMINAMATH_CALUDE_shape_to_square_possible_l3186_318644

/-- Represents a shape on a graph paper --/
structure Shape :=
  (area : ℝ)

/-- Represents a triangle --/
structure Triangle :=
  (area : ℝ)

/-- Represents a square --/
structure Square :=
  (side_length : ℝ)

/-- Function to divide a shape into triangles --/
def divide_into_triangles (s : Shape) : List Triangle := sorry

/-- Function to assemble triangles into a square --/
def assemble_square (triangles : List Triangle) : Option Square := sorry

/-- Theorem stating that the shape can be divided into 5 triangles and assembled into a square --/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (triangles : List Triangle) (sq : Square), 
    divide_into_triangles s = triangles ∧ 
    triangles.length = 5 ∧ 
    assemble_square triangles = some sq :=
by sorry

end NUMINAMATH_CALUDE_shape_to_square_possible_l3186_318644


namespace NUMINAMATH_CALUDE_sufficient_condition_for_collinearity_l3186_318655

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem sufficient_condition_for_collinearity (x : ℝ) :
  let a : ℝ × ℝ := (1, 2 - x)
  let b : ℝ × ℝ := (2 + x, 3)
  b = (1, 3) → collinear a b :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_collinearity_l3186_318655


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3186_318676

theorem repeating_decimal_sum (a b c d : ℕ) : 
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  ((10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) →
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3186_318676


namespace NUMINAMATH_CALUDE_brother_age_twice_sister_l3186_318665

def brother_age_2005 : ℕ := 16
def sister_age_2005 : ℕ := 10
def reference_year : ℕ := 2005

theorem brother_age_twice_sister : 
  ∃ (year : ℕ), year = reference_year - (brother_age_2005 - 2 * sister_age_2005) ∧ year = 2001 :=
sorry

end NUMINAMATH_CALUDE_brother_age_twice_sister_l3186_318665


namespace NUMINAMATH_CALUDE_building_height_ratio_l3186_318627

/-- Given a flagpole and two buildings under similar shadow conditions, 
    this theorem proves that the ratio of the heights of Building A to Building B is 5:6. -/
theorem building_height_ratio 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_a_shadow : ℝ) 
  (building_b_shadow : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_a_shadow_pos : 0 < building_a_shadow)
  (building_b_shadow_pos : 0 < building_b_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_a_shadow : building_a_shadow = 60)
  (h_building_b_shadow : building_b_shadow = 72) :
  (flagpole_height / flagpole_shadow * building_a_shadow) / 
  (flagpole_height / flagpole_shadow * building_b_shadow) = 5 / 6 := by
  sorry

#check building_height_ratio

end NUMINAMATH_CALUDE_building_height_ratio_l3186_318627


namespace NUMINAMATH_CALUDE_total_net_amount_is_218_l3186_318671

/-- Represents a lottery ticket with its cost and number of winning numbers -/
structure LotteryTicket where
  cost : ℕ
  winningNumbers : ℕ

/-- Calculates the payout for a single ticket based on its winning numbers -/
def calculatePayout (ticket : LotteryTicket) : ℕ :=
  if ticket.winningNumbers ≤ 2 then
    ticket.winningNumbers * 15
  else
    30 + (ticket.winningNumbers - 2) * 20

/-- Calculates the net amount won for a single ticket -/
def calculateNetAmount (ticket : LotteryTicket) : ℤ :=
  (calculatePayout ticket : ℤ) - ticket.cost

/-- The set of lottery tickets Tony bought -/
def tonyTickets : List LotteryTicket := [
  ⟨5, 3⟩,
  ⟨7, 5⟩,
  ⟨4, 2⟩,
  ⟨6, 4⟩
]

/-- Theorem stating that the total net amount Tony won is $218 -/
theorem total_net_amount_is_218 :
  (tonyTickets.map calculateNetAmount).sum = 218 := by
  sorry

end NUMINAMATH_CALUDE_total_net_amount_is_218_l3186_318671


namespace NUMINAMATH_CALUDE_parabola_vertex_l3186_318698

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0 
    is [-7,1] ∪ [3,∞), prove that the vertex of the parabola is (-3, 16) -/
theorem parabola_vertex (c d : ℝ) 
  (h : Set.Icc (-7 : ℝ) 1 ∪ Set.Ici 3 = {x : ℝ | -x^2 + c*x + d ≤ 0}) : 
  let f := fun (x : ℝ) ↦ -x^2 + c*x + d
  ∃ (v : ℝ × ℝ), v = (-3, 16) ∧ ∀ (x : ℝ), f x ≤ f v.1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3186_318698


namespace NUMINAMATH_CALUDE_cubic_common_roots_l3186_318623

theorem cubic_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 12 = 0 ∧ 
    r^3 + b*r^2 + 23*r + 15 = 0 ∧
    s^3 + a*s^2 + 17*s + 12 = 0 ∧ 
    s^3 + b*s^2 + 23*s + 15 = 0) →
  a = -10 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_cubic_common_roots_l3186_318623


namespace NUMINAMATH_CALUDE_min_value_theorem_l3186_318668

/-- The minimum value of a function given specific conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hmn : m * n > 0) :
  let f := fun x => a^(x - 1) + 1
  let line := fun x y => 2 * m * x + n * y - 4 = 0
  ∃ (x y : ℝ), f x = y ∧ line x y →
  (4 / m + 2 / n : ℝ) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3186_318668


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3186_318643

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 35 = 0 ∧          -- multiple of 35
  (n / 100 + (n / 10) % 10 + n % 10 = 15) ∧  -- sum of digits is 15
  n = 735 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3186_318643
