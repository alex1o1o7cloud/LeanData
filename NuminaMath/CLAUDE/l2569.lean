import Mathlib

namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2569_256970

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2569_256970


namespace NUMINAMATH_CALUDE_hostel_provisions_l2569_256901

theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (remaining_days : ℕ) :
  initial_men = 250 →
  left_men = 50 →
  remaining_days = 45 →
  (initial_men : ℚ) * (initial_men - left_men : ℚ)⁻¹ * remaining_days = 36 :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l2569_256901


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2569_256943

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 2) % 12 = 0 ∧
  (n + 2) % 30 = 0 ∧
  (n + 2) % 48 = 0 ∧
  (n + 2) % 74 = 0 ∧
  (n + 2) % 100 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 44398 ∧
  ∀ m : ℕ, m < 44398 → ¬(is_divisible_by_all m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2569_256943


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2569_256981

theorem largest_whole_number_nine_times_less_than_150 :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2569_256981


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2569_256900

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle -/
def specificTriangle : Triangle :=
  { A := (4, 0),
    B := (6, 7),
    C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line AC and altitude from B to AB -/
theorem triangle_line_equations (t : Triangle) (t_eq : t = specificTriangle) :
  ∃ (lineAC altitudeB : LineEquation),
    lineAC = { a := 3, b := 4, c := -12 } ∧
    altitudeB = { a := 2, b := 7, c := -21 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2569_256900


namespace NUMINAMATH_CALUDE_remaining_area_calculation_l2569_256915

theorem remaining_area_calculation (large_square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : large_square_side = 9)
  (h2 : small_square1_side = 4)
  (h3 : small_square2_side = 2)
  (h4 : small_square1_side ^ 2 + small_square2_side ^ 2 ≤ large_square_side ^ 2) :
  large_square_side ^ 2 - (small_square1_side ^ 2 + small_square2_side ^ 2) = 61 := by
sorry


end NUMINAMATH_CALUDE_remaining_area_calculation_l2569_256915


namespace NUMINAMATH_CALUDE_apple_profit_calculation_l2569_256985

/-- Calculates the total percentage profit for a shopkeeper selling apples -/
theorem apple_profit_calculation (total_apples : ℝ) (first_portion : ℝ) (second_portion : ℝ)
  (first_profit_rate : ℝ) (second_profit_rate : ℝ) 
  (h1 : total_apples = 100)
  (h2 : first_portion = 0.5 * total_apples)
  (h3 : second_portion = 0.5 * total_apples)
  (h4 : first_profit_rate = 0.25)
  (h5 : second_profit_rate = 0.3) :
  (((first_portion * (1 + first_profit_rate) + second_portion * (1 + second_profit_rate)) - total_apples) / total_apples) * 100 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_apple_profit_calculation_l2569_256985


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l2569_256993

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), f r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l2569_256993


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2569_256974

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) : 
  total = 500 → music = 30 → art = 20 → both = 10 →
  total - (music + art - both) = 460 := by
sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2569_256974


namespace NUMINAMATH_CALUDE_distance_for_specific_triangle_l2569_256939

/-- A right-angled triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- The distance between the centers of the inscribed and circumscribed circles of a right triangle --/
def distance_between_centers (t : RightTriangle) : ℝ :=
  sorry

theorem distance_for_specific_triangle :
  let t : RightTriangle := ⟨8, 15, 17, by norm_num⟩
  distance_between_centers t = Real.sqrt 85 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_specific_triangle_l2569_256939


namespace NUMINAMATH_CALUDE_not_right_triangle_l2569_256989

theorem not_right_triangle (a b c : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 4) (hc : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l2569_256989


namespace NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l2569_256927

open Real

-- Define the function and its derivative
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the interval (0, 2)
def interval : Set ℝ := Set.Ioo 0 2

-- Define what it means for f' to have two zeros in the interval
def has_two_zeros (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0

-- Define what it means for f to have two extreme points in the interval
def has_two_extreme_points (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ ≠ x₂ ∧ 
    (∀ x ∈ I, g x ≤ g x₁) ∧ (∀ x ∈ I, g x ≤ g x₂)

-- Theorem stating that f' having two zeros is neither necessary nor sufficient for f having two extreme points
theorem two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f f', has_two_zeros f' interval → has_two_extreme_points f interval) ∧
  ¬(∀ f f', has_two_extreme_points f interval → has_two_zeros f' interval) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l2569_256927


namespace NUMINAMATH_CALUDE_divisible_by_nine_l2569_256928

theorem divisible_by_nine (k : ℕ+) : 
  ∃ n : ℤ, 3 * (2 + 7^(k.val)) = 9 * n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l2569_256928


namespace NUMINAMATH_CALUDE_sum_multiple_of_five_l2569_256908

theorem sum_multiple_of_five (a b : ℤ) (ha : ∃ m : ℤ, a = 5 * m) (hb : ∃ n : ℤ, b = 10 * n) :
  ∃ k : ℤ, a + b = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_of_five_l2569_256908


namespace NUMINAMATH_CALUDE_contradiction_assumption_correct_l2569_256929

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def isObtuse (angle : ℝ) : Prop := angle > 90

/-- The statement "A triangle has at most one obtuse angle". -/
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), (isObtuse a → ¬isObtuse b ∧ ¬isObtuse c) ∧
                 (isObtuse b → ¬isObtuse a ∧ ¬isObtuse c) ∧
                 (isObtuse c → ¬isObtuse a ∧ ¬isObtuse b)

/-- The correct assumption for the method of contradiction. -/
def correctAssumption (t : Triangle) : Prop :=
  ∃ (a b : ℝ), isObtuse a ∧ isObtuse b ∧ a ≠ b

theorem contradiction_assumption_correct (t : Triangle) :
  ¬atMostOneObtuseAngle t ↔ correctAssumption t :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_correct_l2569_256929


namespace NUMINAMATH_CALUDE_pond_length_l2569_256997

/-- Given a rectangular field and a square pond, prove the length of the pond -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 16 →
  field_length = 2 * field_width →
  pond_length ^ 2 = (field_length * field_width) / 2 →
  pond_length = 8 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l2569_256997


namespace NUMINAMATH_CALUDE_alternative_basis_l2569_256964

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} is a basis in space, prove that {c, a+b, a-b} is also a basis. -/
theorem alternative_basis
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![c, a+b, a-b] ∧
  Submodule.span ℝ {c, a+b, a-b} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_alternative_basis_l2569_256964


namespace NUMINAMATH_CALUDE_equation_solution_l2569_256942

theorem equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2569_256942


namespace NUMINAMATH_CALUDE_books_count_l2569_256949

theorem books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
  (h1 : benny_initial = 24)
  (h2 : given_to_sandy = 10)
  (h3 : tim_books = 33) :
  benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l2569_256949


namespace NUMINAMATH_CALUDE_base_subtraction_l2569_256912

-- Define a function to convert from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [1, 2, 3, 5, 4]
def base1 : Nat := 6

def num2 : List Nat := [1, 2, 3, 4]
def base2 : Nat := 7

-- State the theorem
theorem base_subtraction :
  to_base_10 num1 base1 - to_base_10 num2 base2 = 4851 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l2569_256912


namespace NUMINAMATH_CALUDE_worker_c_work_rate_l2569_256947

/-- Given workers A, B, and C, and their work rates, prove that C's work rate is 1/3 of the total work per hour. -/
theorem worker_c_work_rate
  (total_work : ℝ) -- Total work to be done
  (rate_a : ℝ) -- A's work rate
  (rate_b : ℝ) -- B's work rate
  (rate_c : ℝ) -- C's work rate
  (h1 : rate_a = total_work / 3) -- A can do the work in 3 hours
  (h2 : rate_b + rate_c = total_work / 2) -- B and C together can do the work in 2 hours
  (h3 : rate_a + rate_b = total_work / 2) -- A and B together can do the work in 2 hours
  : rate_c = total_work / 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_c_work_rate_l2569_256947


namespace NUMINAMATH_CALUDE_remainder_theorem_l2569_256940

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 160 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2569_256940


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2569_256918

theorem fraction_equivalence (x b : ℝ) : 
  (x + 2*b) / (x + 3*b) = 2/3 ↔ x = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2569_256918


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2569_256909

theorem arithmetic_expression_equality : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2569_256909


namespace NUMINAMATH_CALUDE_product_evaluation_l2569_256952

theorem product_evaluation (n : ℤ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) + 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2569_256952


namespace NUMINAMATH_CALUDE_january_salary_l2569_256934

/-- Prove that given the conditions, the salary for January is 3300 --/
theorem january_salary (jan feb mar apr may : ℕ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8800 →
  may = 6500 →
  jan = 3300 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l2569_256934


namespace NUMINAMATH_CALUDE_elias_bananas_l2569_256983

def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem elias_bananas : 
  let initial := 12
  let remaining := 11
  bananas_eaten initial remaining = 1 := by
sorry

end NUMINAMATH_CALUDE_elias_bananas_l2569_256983


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l2569_256922

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℤ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l2569_256922


namespace NUMINAMATH_CALUDE_james_bought_ten_shirts_l2569_256935

/-- Represents the number of shirts James bought -/
def num_shirts : ℕ := 10

/-- Represents the number of pants James bought -/
def num_pants : ℕ := num_shirts / 2

/-- Represents the cost of a single shirt in dollars -/
def shirt_cost : ℕ := 6

/-- Represents the cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 8

/-- Represents the total cost of the purchase in dollars -/
def total_cost : ℕ := 100

/-- Theorem stating that given the conditions, James bought 10 shirts -/
theorem james_bought_ten_shirts : 
  num_shirts * shirt_cost + num_pants * pants_cost = total_cost ∧ 
  num_shirts = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_bought_ten_shirts_l2569_256935


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l2569_256958

/-- The cost of pencils and notebooks given specific conditions --/
theorem pencil_notebook_cost :
  ∀ (p n : ℝ),
  -- Condition 1: 9 pencils and 10 notebooks cost $5.35
  9 * p + 10 * n = 5.35 →
  -- Condition 2: 6 pencils and 4 notebooks cost $2.50
  6 * p + 4 * n = 2.50 →
  -- The cost of 24 pencils (with 10% discount on packs of 4) and 15 notebooks is $9.24
  24 * (0.9 * p) + 15 * n = 9.24 :=
by
  sorry


end NUMINAMATH_CALUDE_pencil_notebook_cost_l2569_256958


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l2569_256920

/-- Represents the cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- The conditions of the problem -/
axiom luncheon_condition_1 : ∀ (c : LuncheonCost), 
  5 * c.sandwich + 8 * c.coffee + c.pie = 5.25

axiom luncheon_condition_2 : ∀ (c : LuncheonCost), 
  7 * c.sandwich + 12 * c.coffee + c.pie = 7.35

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (c : LuncheonCost) : 
  c.sandwich + c.coffee + c.pie = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l2569_256920


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2569_256963

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2569_256963


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2569_256955

/-- A parallelogram with opposite vertices (3, -4) and (13, 8) has its diagonals intersecting at (8, 2) -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (3, -4)
  let v2 : ℝ × ℝ := (13, 8)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 2) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2569_256955


namespace NUMINAMATH_CALUDE_existence_of_n_div_prime_count_l2569_256945

/-- π(x) denotes the number of prime numbers less than or equal to x -/
def prime_counting_function (x : ℕ) : ℕ := sorry

/-- For any integer m > 1, there exists an integer n > 1 such that n/π(n) = m -/
theorem existence_of_n_div_prime_count (m : ℕ) (h : m > 1) : 
  ∃ n : ℕ, n > 1 ∧ n = m * prime_counting_function n :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_div_prime_count_l2569_256945


namespace NUMINAMATH_CALUDE_target_container_marbles_l2569_256926

/-- A container type with volume and marble capacity -/
structure Container where
  volume : ℝ
  marbles : ℕ

/-- The ratio of marbles to volume is constant for all containers -/
axiom marble_volume_ratio (c1 c2 : Container) : 
  c1.marbles / c1.volume = c2.marbles / c2.volume

/-- Given container properties -/
def given_container : Container := { volume := 24, marbles := 30 }

/-- Container we want to find the marble count for -/
def target_container : Container := { volume := 72, marbles := 90 }

/-- Theorem: The target container can hold 90 marbles -/
theorem target_container_marbles : target_container.marbles = 90 := by
  sorry

end NUMINAMATH_CALUDE_target_container_marbles_l2569_256926


namespace NUMINAMATH_CALUDE_multiples_properties_l2569_256979

theorem multiples_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiples_properties_l2569_256979


namespace NUMINAMATH_CALUDE_a_range_l2569_256916

theorem a_range (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  a ∈ Set.Ici 4 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l2569_256916


namespace NUMINAMATH_CALUDE_equal_circles_in_quadrilateral_l2569_256967

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a convex quadrilateral with four circles inside -/
structure ConvexQuadrilateral where
  circle_a : Circle
  circle_b : Circle
  circle_c : Circle
  circle_d : Circle
  is_convex : Bool
  circles_touch_sides : Bool
  circles_touch_each_other : Bool
  has_inscribed_circle : Bool

/-- 
Given a convex quadrilateral with four circles inside, each touching two adjacent sides 
and two other circles externally, and given that a circle can be inscribed in the quadrilateral, 
at least two of the four circles have equal radii.
-/
theorem equal_circles_in_quadrilateral (q : ConvexQuadrilateral) 
  (h1 : q.is_convex = true) 
  (h2 : q.circles_touch_sides = true)
  (h3 : q.circles_touch_each_other = true)
  (h4 : q.has_inscribed_circle = true) : 
  (q.circle_a.radius = q.circle_b.radius) ∨ 
  (q.circle_a.radius = q.circle_c.radius) ∨ 
  (q.circle_a.radius = q.circle_d.radius) ∨ 
  (q.circle_b.radius = q.circle_c.radius) ∨ 
  (q.circle_b.radius = q.circle_d.radius) ∨ 
  (q.circle_c.radius = q.circle_d.radius) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_circles_in_quadrilateral_l2569_256967


namespace NUMINAMATH_CALUDE_polynomial_sum_l2569_256988

theorem polynomial_sum (x : ℝ) : (x^2 + 3*x - 4) + (-3*x + 1) = x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2569_256988


namespace NUMINAMATH_CALUDE_bundle_limit_points_l2569_256930

-- Define the types of bundles
inductive BundleType
  | Hyperbolic
  | Parabolic
  | Elliptic

-- Define a function that returns the number of limit points for a given bundle type
def limitPoints (b : BundleType) : Nat :=
  match b with
  | BundleType.Hyperbolic => 2
  | BundleType.Parabolic => 1
  | BundleType.Elliptic => 0

-- Theorem statement
theorem bundle_limit_points (b : BundleType) :
  (b = BundleType.Hyperbolic → limitPoints b = 2) ∧
  (b = BundleType.Parabolic → limitPoints b = 1) ∧
  (b = BundleType.Elliptic → limitPoints b = 0) :=
by sorry

end NUMINAMATH_CALUDE_bundle_limit_points_l2569_256930


namespace NUMINAMATH_CALUDE_kareems_son_age_l2569_256932

theorem kareems_son_age (kareem_age : ℕ) (son_age : ℕ) : 
  kareem_age = 42 →
  kareem_age = 3 * son_age →
  (kareem_age + 10) + (son_age + 10) = 76 →
  son_age = 14 := by
sorry

end NUMINAMATH_CALUDE_kareems_son_age_l2569_256932


namespace NUMINAMATH_CALUDE_prime_pair_sum_10_product_21_l2569_256936

theorem prime_pair_sum_10_product_21 : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_pair_sum_10_product_21_l2569_256936


namespace NUMINAMATH_CALUDE_minimum_value_theorem_equality_condition_l2569_256925

theorem minimum_value_theorem (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 2) : ∃ x, x > 2 ∧ x + 1 / (x - 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_equality_condition_l2569_256925


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2569_256994

/-- A geometric sequence with common ratio r satisfying a_n * a_(n+1) = 16^n has r = 4 --/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_prod : ∀ n, a n * a (n + 1) = 16^n) : 
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2569_256994


namespace NUMINAMATH_CALUDE_current_tariff_calculation_specific_case_calculation_l2569_256992

/-- Calculates the current actual tariff after two successive reductions -/
def current_tariff (S : ℝ) : ℝ := (1 - 0.4) * (1 - 0.3) * S

/-- Theorem stating the current actual tariff calculation -/
theorem current_tariff_calculation (S : ℝ) : 
  current_tariff S = (1 - 0.4) * (1 - 0.3) * S := by sorry

/-- Theorem for the specific case when S = 1000 -/
theorem specific_case_calculation : 
  current_tariff 1000 = 420 := by sorry

end NUMINAMATH_CALUDE_current_tariff_calculation_specific_case_calculation_l2569_256992


namespace NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2569_256961

theorem negation_of_square_positive_equals_zero (m : ℝ) :
  ¬(m > 0 ∧ m^2 = 0) ↔ (m ≤ 0 → m^2 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2569_256961


namespace NUMINAMATH_CALUDE_trajectory_of_P_l2569_256980

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point M on the ellipse C
def M : ℝ × ℝ → Prop := λ p => C p.1 p.2

-- Define the point N on the x-axis
def N (m : ℝ × ℝ) : ℝ × ℝ := (m.1, 0)

-- Define the vector NP
def NP (n p : ℝ × ℝ) : ℝ × ℝ := (p.1 - n.1, p.2 - n.2)

-- Define the vector NM
def NM (n m : ℝ × ℝ) : ℝ × ℝ := (m.1 - n.1, m.2 - n.2)

-- State the theorem
theorem trajectory_of_P (x y : ℝ) :
  (∃ m : ℝ × ℝ, M m ∧ 
   let n := N m
   NP n (x, y) = Real.sqrt 2 • NM n m) →
  x^2 + y^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2569_256980


namespace NUMINAMATH_CALUDE_faye_science_problems_l2569_256946

theorem faye_science_problems :
  ∀ (math_problems finished_problems remaining_problems : ℕ),
    math_problems = 46 →
    finished_problems = 40 →
    remaining_problems = 15 →
    math_problems + (finished_problems + remaining_problems - math_problems) = 
      finished_problems + remaining_problems :=
by
  sorry

end NUMINAMATH_CALUDE_faye_science_problems_l2569_256946


namespace NUMINAMATH_CALUDE_steaks_needed_l2569_256957

def family_members : ℕ := 5
def pounds_per_person : ℕ := 1
def ounces_per_steak : ℕ := 20
def ounces_per_pound : ℕ := 16

def total_ounces : ℕ := family_members * pounds_per_person * ounces_per_pound

theorem steaks_needed : (total_ounces / ounces_per_steak : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_steaks_needed_l2569_256957


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2569_256953

theorem basketball_lineup_combinations (total_players : ℕ) (quadruplets : ℕ) (lineup_size : ℕ) (quadruplets_in_lineup : ℕ) : 
  total_players = 16 → 
  quadruplets = 4 → 
  lineup_size = 7 → 
  quadruplets_in_lineup = 2 → 
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets + quadruplets_in_lineup) (lineup_size - quadruplets_in_lineup)) = 12012 := by
  sorry

#check basketball_lineup_combinations

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2569_256953


namespace NUMINAMATH_CALUDE_sequence_term_proof_l2569_256973

def sequence_sum (n : ℕ) := 3^n + 2

def sequence_term (n : ℕ) : ℝ :=
  if n = 1 then 5 else 2 * 3^(n-1)

theorem sequence_term_proof (n : ℕ) :
  sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l2569_256973


namespace NUMINAMATH_CALUDE_product_of_first_five_l2569_256937

def is_on_line (x y : ℝ) : Prop :=
  3 * x + y = 0

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → is_on_line (a (n+1)) (a n)

theorem product_of_first_five (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_five_l2569_256937


namespace NUMINAMATH_CALUDE_elena_car_rental_cost_l2569_256977

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def car_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that Elena's car rental cost is $215 given the specified conditions. -/
theorem elena_car_rental_cost :
  car_rental_cost 30 0.25 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_elena_car_rental_cost_l2569_256977


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2569_256948

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y a : ℝ) : Prop := (x+3)^2 + (y-a)^2 = 16

-- Define the tangency condition
def tangent (a : ℝ) : Prop := ∃ x y, circle1 x y ∧ circle2 x y a

-- Define the cube and sphere relationship
def cube_on_sphere (a : ℝ) : Prop := 
  ∃ r, r = a * Real.sqrt 3 / 2

-- Main theorem
theorem sphere_surface_area (a : ℝ) :
  a > 0 → tangent a → cube_on_sphere a → 4 * Real.pi * (a * Real.sqrt 3)^2 = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2569_256948


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2569_256931

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 7 / Real.sqrt 12) = 
  (35 * Real.sqrt 70) / 840 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2569_256931


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2569_256933

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 4*x + m ≥ 0) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2569_256933


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2569_256959

theorem quadratic_form_sum (x : ℝ) :
  ∃ (b c : ℝ), x^2 - 18*x + 45 = (x + b)^2 + c ∧ b + c = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2569_256959


namespace NUMINAMATH_CALUDE_x₀_value_l2569_256976

noncomputable section

variables (a b : ℝ) (x₀ : ℝ)

def f (x : ℝ) := a * x^2 + b

theorem x₀_value (ha : a ≠ 0) (hx₀ : x₀ > 0) 
  (h_integral : ∫ x in (0)..(2), f a b x = 2 * f a b x₀) : 
  x₀ = 2 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_x₀_value_l2569_256976


namespace NUMINAMATH_CALUDE_regular_price_is_80_l2569_256990

/-- The regular price of one tire -/
def regular_price : ℝ := 80

/-- The total cost for four tires -/
def total_cost : ℝ := 250

/-- Theorem: The regular price of one tire is 80 dollars -/
theorem regular_price_is_80 : regular_price = 80 :=
  by
    have h1 : total_cost = 3 * regular_price + 10 := by sorry
    have h2 : total_cost = 250 := by rfl
    sorry

#check regular_price_is_80

end NUMINAMATH_CALUDE_regular_price_is_80_l2569_256990


namespace NUMINAMATH_CALUDE_z_local_minimum_l2569_256911

-- Define the function
def z (x y : ℝ) : ℝ := x^3 + y^3 - 3*x*y

-- State the theorem
theorem z_local_minimum :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x y : ℝ),
    (x - 1)^2 + (y - 1)^2 < ε^2 → z x y ≥ z 1 1 ∧ z 1 1 = -1 :=
sorry

end NUMINAMATH_CALUDE_z_local_minimum_l2569_256911


namespace NUMINAMATH_CALUDE_farmer_extra_days_l2569_256991

/-- The number of extra days a farmer needs to work given initial and actual ploughing rates, total area, and remaining area. -/
theorem farmer_extra_days (initial_rate actual_rate total_area remaining_area : ℕ) : 
  initial_rate = 90 →
  actual_rate = 85 →
  total_area = 3780 →
  remaining_area = 40 →
  (total_area - remaining_area) % actual_rate = 0 →
  (remaining_area + actual_rate - 1) / actual_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_farmer_extra_days_l2569_256991


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l2569_256907

theorem rectangle_area_equals_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (a + b) →  -- area equals perimeter condition
  2 * (a + b) = 18 :=  -- conclusion: perimeter is 18
by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l2569_256907


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2569_256954

/-- The number of diagonals in an octagon -/
def diagonals_in_octagon : ℕ :=
  let vertices : ℕ := 8
  let sides : ℕ := 8
  (vertices.choose 2) - sides

/-- Theorem stating that the number of diagonals in an octagon is 20 -/
theorem octagon_diagonals :
  diagonals_in_octagon = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2569_256954


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l2569_256965

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_diagonal_theorem (ABCD : Quadrilateral) (O : Point) :
  is_convex ABCD →
  distance ABCD.A ABCD.B = 10 →
  distance ABCD.C ABCD.D = 15 →
  distance ABCD.A ABCD.C = 20 →
  O = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangle_area ABCD.A O ABCD.D = triangle_area ABCD.B O ABCD.C →
  distance ABCD.A O = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l2569_256965


namespace NUMINAMATH_CALUDE_expression_simplification_l2569_256923

theorem expression_simplification (x y : ℝ) 
  (h : |x + 1| + (2 * y - 4)^2 = 0) : 
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2569_256923


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_minus_two_l2569_256996

theorem power_of_two_greater_than_square_minus_two (n : ℕ) (h : n > 0) : 
  2^n > n^2 - 2 :=
by
  -- Assume the proposition holds for n = 1, n = 2, and n = 3
  have base_case_1 : 2^1 > 1^2 - 2 := by sorry
  have base_case_2 : 2^2 > 2^2 - 2 := by sorry
  have base_case_3 : 2^3 > 3^2 - 2 := by sorry

  -- Proof by induction
  induction n with
  | zero => contradiction
  | succ n ih =>
    -- Inductive step
    sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_minus_two_l2569_256996


namespace NUMINAMATH_CALUDE_cards_traded_is_35_l2569_256960

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ) 
  (padma_second_trade : ℕ) (robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + robert_first_trade + padma_second_trade + robert_second_trade

/-- Theorem stating the total number of cards traded is 35 -/
theorem cards_traded_is_35 : 
  total_cards_traded 75 88 2 10 15 8 = 35 := by
  sorry


end NUMINAMATH_CALUDE_cards_traded_is_35_l2569_256960


namespace NUMINAMATH_CALUDE_dessert_distribution_l2569_256917

/-- Proves that given 14 mini-cupcakes, 12 donut holes, and 13 students,
    if each student receives the same amount, then each student gets 2 desserts. -/
theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 := by
  sorry

end NUMINAMATH_CALUDE_dessert_distribution_l2569_256917


namespace NUMINAMATH_CALUDE_bankers_discount_l2569_256910

/-- Banker's discount calculation -/
theorem bankers_discount 
  (PV : ℝ) -- Present Value
  (BG : ℝ) -- Banker's Gain
  (n : ℕ) -- Total number of years
  (r1 : ℝ) -- Interest rate for first half of the period
  (r2 : ℝ) -- Interest rate for second half of the period
  (h : n = 8) -- The sum is due 8 years hence
  (h1 : r1 = 0.10) -- Interest rate is 10% for the first 4 years
  (h2 : r2 = 0.12) -- Interest rate is 12% for the remaining 4 years
  (h3 : BG = 900) -- The banker's gain is Rs. 900
  : ∃ (BD : ℝ), BD = BG + ((PV * (1 + r1) ^ (n / 2)) * (1 + r2) ^ (n / 2) - PV) :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_l2569_256910


namespace NUMINAMATH_CALUDE_smallest_triangle_perimeter_l2569_256941

theorem smallest_triangle_perimeter :
  ∀ a b c : ℕ,
  a ≥ 5 →
  b = a + 1 →
  c = b + 1 →
  a + b > c →
  a + c > b →
  b + c > a →
  ∀ x y z : ℕ,
  x ≥ 5 →
  y = x + 1 →
  z = y + 1 →
  x + y > z →
  x + z > y →
  y + z > x →
  a + b + c ≤ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_perimeter_l2569_256941


namespace NUMINAMATH_CALUDE_acid_dilution_l2569_256998

/-- Proves that adding 80/3 ounces of pure water to 40 ounces of a 25% acid solution
    results in a 15% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) : 
    initial_volume = 40 →
    initial_concentration = 0.25 →
    added_water = 80 / 3 →
    final_concentration = 0.15 →
    (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l2569_256998


namespace NUMINAMATH_CALUDE_intersection_M_N_l2569_256962

def M : Set ℝ := {1, 2, 3, 4, 5}
def N : Set ℝ := {x | Real.log x / Real.log 4 ≥ 1}

theorem intersection_M_N : M ∩ N = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2569_256962


namespace NUMINAMATH_CALUDE_cistern_fill_time_theorem_l2569_256971

/-- Represents the rate at which a pipe can fill or empty a cistern -/
structure PipeRate where
  fill : ℚ  -- Fraction of cistern filled or emptied
  time : ℚ  -- Time taken in minutes
  deriving Repr

/-- Calculates the rate of filling or emptying per minute -/
def rate_per_minute (p : PipeRate) : ℚ := p.fill / p.time

/-- Represents the problem of filling a cistern with multiple pipes -/
structure CisternProblem where
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_c : PipeRate
  target_fill : ℚ
  deriving Repr

/-- Calculates the time required to fill the target amount of the cistern -/
def fill_time (problem : CisternProblem) : ℚ :=
  let combined_rate := rate_per_minute problem.pipe_a + rate_per_minute problem.pipe_b - rate_per_minute problem.pipe_c
  problem.target_fill / combined_rate

/-- The main theorem stating the time required to fill half the cistern -/
theorem cistern_fill_time_theorem (problem : CisternProblem) 
  (h1 : problem.pipe_a = ⟨1/2, 10⟩)
  (h2 : problem.pipe_b = ⟨2/3, 15⟩)
  (h3 : problem.pipe_c = ⟨1/4, 20⟩)
  (h4 : problem.target_fill = 1/2) :
  fill_time problem = 720/118 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_theorem_l2569_256971


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_150_225_300_l2569_256982

theorem greatest_common_factor_of_150_225_300 : 
  Nat.gcd 150 (Nat.gcd 225 300) = 75 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_150_225_300_l2569_256982


namespace NUMINAMATH_CALUDE_modulo_equivalence_solution_l2569_256975

theorem modulo_equivalence_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_solution_l2569_256975


namespace NUMINAMATH_CALUDE_lcm_factor_not_unique_l2569_256999

/-- Given two positive integers with HCF 52 and larger number 624, 
    the other factor of their LCM cannot be uniquely determined. -/
theorem lcm_factor_not_unique (A B : ℕ+) : 
  (Nat.gcd A B = 52) → 
  (max A B = 624) → 
  ∃ (y : ℕ+), y ≠ 1 ∧ 
    ∃ (lcm : ℕ+), lcm = Nat.lcm A B ∧ lcm = 624 * y :=
by sorry

end NUMINAMATH_CALUDE_lcm_factor_not_unique_l2569_256999


namespace NUMINAMATH_CALUDE_jesus_squares_count_l2569_256968

/-- The number of squares Pedro has -/
def pedro_squares : ℕ := 200

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of extra squares Pedro has compared to Jesus and Linden combined -/
def extra_squares : ℕ := 65

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := pedro_squares - linden_squares - extra_squares

theorem jesus_squares_count : jesus_squares = 60 := by sorry

end NUMINAMATH_CALUDE_jesus_squares_count_l2569_256968


namespace NUMINAMATH_CALUDE_room_occupancy_l2569_256904

theorem room_occupancy (empty_chairs : ℕ) (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  empty_chairs = 14 →
  empty_chairs * 2 = total_chairs →
  seated_people = total_chairs - empty_chairs →
  seated_people = (2 : ℚ) / 3 * total_people →
  total_people = 21 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l2569_256904


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2569_256913

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 8) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 24 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2569_256913


namespace NUMINAMATH_CALUDE_ray_nickels_left_l2569_256987

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define Ray's initial amount in cents
def initial_amount : ℕ := 95

-- Define the amount given to Peter in cents
def amount_to_peter : ℕ := 25

-- Theorem stating that Ray will have 4 nickels left
theorem ray_nickels_left : 
  let amount_to_randi := 2 * amount_to_peter
  let total_given := amount_to_peter + amount_to_randi
  let remaining_cents := initial_amount - total_given
  remaining_cents / nickel_value = 4 := by
sorry

end NUMINAMATH_CALUDE_ray_nickels_left_l2569_256987


namespace NUMINAMATH_CALUDE_four_digit_number_divisible_by_twelve_l2569_256950

theorem four_digit_number_divisible_by_twelve (n : ℕ) (A : ℕ) : 
  n = 2000 + 10 * A + 2 →
  A < 10 →
  n % 12 = 0 →
  n = 2052 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_divisible_by_twelve_l2569_256950


namespace NUMINAMATH_CALUDE_problem_solution_l2569_256905

-- Definition of additive inverse
def additive_inverse (x y : ℝ) : Prop := x + y = 0

-- Definition of real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem problem_solution :
  -- Proposition 1
  (∀ x y : ℝ, additive_inverse x y → x + y = 0) ∧
  -- Proposition 3
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2569_256905


namespace NUMINAMATH_CALUDE_july_birth_percentage_l2569_256956

/-- The percentage of people born in July, given 15 out of 120 famous Americans were born in July -/
theorem july_birth_percentage :
  let total_people : ℕ := 120
  let july_births : ℕ := 15
  (july_births : ℚ) / total_people * 100 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l2569_256956


namespace NUMINAMATH_CALUDE_B_highest_score_l2569_256921

-- Define the structure for an applicant
structure Applicant where
  name : String
  knowledge : ℕ
  experience : ℕ
  language : ℕ

-- Define the weighting function
def weightedScore (a : Applicant) : ℚ :=
  (5 * a.knowledge + 2 * a.experience + 3 * a.language) / 10

-- Define the applicants
def A : Applicant := ⟨"A", 75, 80, 80⟩
def B : Applicant := ⟨"B", 85, 80, 70⟩
def C : Applicant := ⟨"C", 70, 78, 70⟩

-- Theorem stating that B has the highest weighted score
theorem B_highest_score :
  weightedScore B > weightedScore A ∧ weightedScore B > weightedScore C :=
by sorry

end NUMINAMATH_CALUDE_B_highest_score_l2569_256921


namespace NUMINAMATH_CALUDE_min_sum_on_circle_l2569_256906

theorem min_sum_on_circle (x y : ℝ) :
  Real.sqrt ((x - 2)^2 + (y - 1)^2) = 1 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (a b : ℝ), Real.sqrt ((a - 2)^2 + (b - 1)^2) = 1 → x + y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_on_circle_l2569_256906


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2569_256938

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -9 → 
      (2 * x + 4) / (x^2 + 2*x - 63) = A / (x - 7) + B / (x + 9)) ∧
    A = 9/8 ∧ B = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2569_256938


namespace NUMINAMATH_CALUDE_total_bowling_balls_l2569_256903

theorem total_bowling_balls (red_balls : ℕ) (green_extra : ℕ) : 
  red_balls = 30 → green_extra = 6 → red_balls + (red_balls + green_extra) = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l2569_256903


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2569_256902

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (x^2) - f x ^ 2 ≥ (1/4 : ℝ)) ∧ 
  Function.Injective f := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2569_256902


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2569_256944

/-- Given a triangle ABC with area 144√3 and satisfying the relation 
    (sin A * sin B * sin C) / (sin A + sin B + sin C) = 1/4, 
    prove that the smallest possible perimeter is achieved when the triangle is equilateral 
    with side length 24. -/
theorem min_perimeter_triangle (A B C : ℝ) (area : ℝ) (h_area : area = 144 * Real.sqrt 3) 
    (h_relation : (Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) :
  ∃ (s : ℝ), s = 24 ∧ 
    ∀ (a b c : ℝ), 
      (a * b * Real.sin C / 2 = area) → 
      ((Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) → 
      (a + b + c ≥ 3 * s) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2569_256944


namespace NUMINAMATH_CALUDE_linear_equation_with_solution_l2569_256914

/-- A linear equation with two variables that has a specific solution -/
theorem linear_equation_with_solution :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y = c ↔ x = -3 ∧ y = 1) ∧
    a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_with_solution_l2569_256914


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l2569_256984

theorem unique_prime_pair_solution : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l2569_256984


namespace NUMINAMATH_CALUDE_correct_packs_for_spoons_l2569_256919

/-- Calculates the number of packs needed to buy a specific number of spoons -/
def packs_needed (total_utensils_per_pack : ℕ) (spoons_wanted : ℕ) : ℕ :=
  let spoons_per_pack := total_utensils_per_pack / 3
  (spoons_wanted + spoons_per_pack - 1) / spoons_per_pack

theorem correct_packs_for_spoons :
  packs_needed 30 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_packs_for_spoons_l2569_256919


namespace NUMINAMATH_CALUDE_triangle_similarity_criterion_l2569_256951

theorem triangle_similarity_criterion (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
    Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_criterion_l2569_256951


namespace NUMINAMATH_CALUDE_product_of_fractions_l2569_256972

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2569_256972


namespace NUMINAMATH_CALUDE_prob_four_friends_same_group_l2569_256986

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- Represents the four friends -/
inductive Friend : Type
  | Al | Bob | Carol | Dan

/-- 
Theorem: The probability that four specific students (friends) are assigned 
to the same lunch group in a random assignment is 1/64.
-/
theorem prob_four_friends_same_group : 
  (prob_assigned_to_group ^ 3 : ℚ) = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_prob_four_friends_same_group_l2569_256986


namespace NUMINAMATH_CALUDE_triangle_reconstruction_theorem_l2569_256966

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the given points
variable (D E F : Point)

-- Define the properties of the given points
def is_altitude_median_intersection (D : Point) (T : Triangle) : Prop := sorry

def is_altitude_bisector_intersection (E : Point) (T : Triangle) : Prop := sorry

def is_median_bisector_intersection (F : Point) (T : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_reconstruction_theorem 
  (hD : ∃ T : Triangle, is_altitude_median_intersection D T)
  (hE : ∃ T : Triangle, is_altitude_bisector_intersection E T)
  (hF : ∃ T : Triangle, is_median_bisector_intersection F T) :
  ∃! T : Triangle, 
    is_altitude_median_intersection D T ∧ 
    is_altitude_bisector_intersection E T ∧ 
    is_median_bisector_intersection F T :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_theorem_l2569_256966


namespace NUMINAMATH_CALUDE_whale_first_hour_consumption_l2569_256995

def whale_feeding (first_hour : ℕ) : Prop :=
  let second_hour := first_hour + 3
  let third_hour := first_hour + 6
  let fourth_hour := first_hour + 9
  let fifth_hour := first_hour + 12
  (third_hour = 93) ∧ 
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450)

theorem whale_first_hour_consumption : 
  ∃ (x : ℕ), whale_feeding x ∧ x = 87 :=
sorry

end NUMINAMATH_CALUDE_whale_first_hour_consumption_l2569_256995


namespace NUMINAMATH_CALUDE_sum_of_x_values_l2569_256978

open Real

theorem sum_of_x_values (x : ℝ) : 
  (0 < x) → 
  (x < 180) → 
  (sin (2 * x * π / 180))^3 + (sin (6 * x * π / 180))^3 = 
    8 * (sin (3 * x * π / 180))^3 * (sin (x * π / 180))^3 → 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (0 < x₁) ∧ (x₁ < 180) ∧
    (0 < x₂) ∧ (x₂ < 180) ∧
    (0 < x₃) ∧ (x₃ < 180) ∧
    (sin (2 * x₁ * π / 180))^3 + (sin (6 * x₁ * π / 180))^3 = 
      8 * (sin (3 * x₁ * π / 180))^3 * (sin (x₁ * π / 180))^3 ∧
    (sin (2 * x₂ * π / 180))^3 + (sin (6 * x₂ * π / 180))^3 = 
      8 * (sin (3 * x₂ * π / 180))^3 * (sin (x₂ * π / 180))^3 ∧
    (sin (2 * x₃ * π / 180))^3 + (sin (6 * x₃ * π / 180))^3 = 
      8 * (sin (3 * x₃ * π / 180))^3 * (sin (x₃ * π / 180))^3 ∧
    x₁ + x₂ + x₃ = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l2569_256978


namespace NUMINAMATH_CALUDE_no_integer_solution_l2569_256969

theorem no_integer_solution : ¬∃ (x y : ℤ), (x + 2020) * (x + 2021) + (x + 2021) * (x + 2022) + (x + 2020) * (x + 2022) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2569_256969


namespace NUMINAMATH_CALUDE_inequality_proof_l2569_256924

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2569_256924
