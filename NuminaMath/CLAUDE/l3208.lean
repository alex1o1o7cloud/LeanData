import Mathlib

namespace NUMINAMATH_CALUDE_find_xy_l3208_320869

/-- Define the ⊕ operation for pairs of real numbers -/
def oplus (a b c d : ℝ) : ℝ × ℝ := (a + c, b * d)

/-- Theorem statement -/
theorem find_xy : ∃ (x y : ℝ), oplus x 1 2 y = (4, 2) ∧ (x, y) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_find_xy_l3208_320869


namespace NUMINAMATH_CALUDE_find_M_l3208_320811

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3208_320811


namespace NUMINAMATH_CALUDE_common_intersection_point_l3208_320861

-- Define the circle S
def S : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the tangent line L at A
def L : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the function for points X and Y on L
def X (p : ℝ) : ℝ × ℝ := (1, 2*p)
def Y (q : ℝ) : ℝ × ℝ := (1, -2*q)

-- Define the condition for X and Y
def XYCondition (p q c : ℝ) : Prop := p * q = c / 4

-- Define the theorem
theorem common_intersection_point (c : ℝ) (h : c > 0) :
  ∀ (p q : ℝ), p > 0 → q > 0 → XYCondition p q c →
  ∃ (R : ℝ × ℝ), R.1 = (4 - c) / (4 + c) ∧ R.2 = 0 ∧
  (∀ (P Q : ℝ × ℝ), P ∈ S → Q ∈ S →
   (∃ (t : ℝ), P = (1 - t) • B + t • X p) →
   (∃ (s : ℝ), Q = (1 - s) • B + s • Y q) →
   ∃ (k : ℝ), R = (1 - k) • P + k • Q) :=
sorry

end NUMINAMATH_CALUDE_common_intersection_point_l3208_320861


namespace NUMINAMATH_CALUDE_max_rectangles_6x6_grid_l3208_320829

/-- Counts the number of rectangles in a right triangle grid of size n x n -/
def count_rectangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- The maximum number of rectangles in a 6x6 right triangle grid is 126 -/
theorem max_rectangles_6x6_grid :
  count_rectangles 6 = 126 := by sorry

end NUMINAMATH_CALUDE_max_rectangles_6x6_grid_l3208_320829


namespace NUMINAMATH_CALUDE_distance_climbed_l3208_320897

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase_steps : ℕ := 24

/-- The number of steps in the second staircase -/
def second_staircase_steps : ℕ := 3 * first_staircase_steps

/-- The number of steps in the third staircase -/
def third_staircase_steps : ℕ := second_staircase_steps - 20

/-- The height of each step in feet -/
def step_height : ℚ := 6/10

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

/-- The total distance climbed in feet -/
def total_distance : ℚ := (total_steps : ℚ) * step_height

theorem distance_climbed : total_distance = 888/10 := by
  sorry

end NUMINAMATH_CALUDE_distance_climbed_l3208_320897


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3208_320887

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_slope_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3208_320887


namespace NUMINAMATH_CALUDE_max_area_PQR_max_area_incenters_l3208_320816

-- Define the equilateral triangle ABC with unit area
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the external equilateral triangles
def triangle_APB : Set (ℝ × ℝ) := sorry
def triangle_BQC : Set (ℝ × ℝ) := sorry
def triangle_CRA : Set (ℝ × ℝ) := sorry

-- Define the angles
def angle_APB : ℝ := 60
def angle_BQC : ℝ := 60
def angle_CRA : ℝ := 60

-- Define the points P, Q, R
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the incenters
def incenter_APB : ℝ × ℝ := sorry
def incenter_BQC : ℝ × ℝ := sorry
def incenter_CRA : ℝ × ℝ := sorry

-- Define the area function
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem for the maximum area of triangle PQR
theorem max_area_PQR :
  ∀ P Q R,
    P ∈ triangle_APB ∧ Q ∈ triangle_BQC ∧ R ∈ triangle_CRA →
    area {P, Q, R} ≤ 4 * Real.sqrt 3 :=
sorry

-- Theorem for the maximum area of triangle formed by incenters
theorem max_area_incenters :
  area {incenter_APB, incenter_BQC, incenter_CRA} ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_area_PQR_max_area_incenters_l3208_320816


namespace NUMINAMATH_CALUDE_number_problem_l3208_320839

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 58 ∧ x = 145 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3208_320839


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3208_320803

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3208_320803


namespace NUMINAMATH_CALUDE_vehicle_purchase_problem_l3208_320843

/-- Represents the purchase price and profit information for new energy vehicles -/
structure VehicleInfo where
  priceA : ℝ  -- Purchase price of type A vehicle in million yuan
  priceB : ℝ  -- Purchase price of type B vehicle in million yuan
  profitA : ℝ  -- Profit from selling one type A vehicle in million yuan
  profitB : ℝ  -- Profit from selling one type B vehicle in million yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  countA : ℕ  -- Number of type A vehicles
  countB : ℕ  -- Number of type B vehicles

/-- Calculates the total cost of a purchase plan given vehicle info -/
def totalCost (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.priceA * plan.countA + info.priceB * plan.countB

/-- Calculates the total profit of a purchase plan given vehicle info -/
def totalProfit (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.profitA * plan.countA + info.profitB * plan.countB

/-- Theorem stating the properties of the vehicle purchase problem -/
theorem vehicle_purchase_problem (info : VehicleInfo) :
  (totalCost ⟨3, 2⟩ info = 95) →
  (totalCost ⟨4, 1⟩ info = 110) →
  (info.profitA = 0.012) →
  (info.profitB = 0.008) →
  (∃ (plans : List PurchasePlan),
    (∀ plan ∈ plans, totalCost plan info = 250) ∧
    (plans.length = 4) ∧
    (∃ maxProfit : ℝ, maxProfit = 18.4 ∧
      ∀ plan ∈ plans, totalProfit plan info ≤ maxProfit)) :=
sorry


end NUMINAMATH_CALUDE_vehicle_purchase_problem_l3208_320843


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3208_320885

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 = 2 →
  a 4 * a 6 = 64 →
  (a 5 + a 6) / (a 1 + a 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3208_320885


namespace NUMINAMATH_CALUDE_plane_division_theorem_l3208_320850

/-- Represents the number of regions formed by lines in a plane -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- Checks if a pair (h, s) satisfies the problem conditions -/
def is_valid_pair (h s : ℕ) : Prop :=
  h > 0 ∧ s > 0 ∧ num_regions h s = 1992

theorem plane_division_theorem :
  ∀ h s : ℕ, is_valid_pair h s ↔ (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by sorry

end NUMINAMATH_CALUDE_plane_division_theorem_l3208_320850


namespace NUMINAMATH_CALUDE_range_of_m_l3208_320863

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 9*a + b = a*b)
  (h : ∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) :
  ∃ m₀ : ℝ, m₀ = 3 ∧ ∀ m : ℝ, (∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) → m ≥ m₀ :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3208_320863


namespace NUMINAMATH_CALUDE_nines_in_hundred_l3208_320849

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for all numbers from 1 to n -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of the digit 9 in all numbers from 1 to 100 (inclusive) is 20 -/
theorem nines_in_hundred : total_nines 100 = 20 := by sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l3208_320849


namespace NUMINAMATH_CALUDE_customer_payment_percentage_l3208_320809

theorem customer_payment_percentage (savings_percentage : ℝ) (payment_percentage : ℝ) :
  savings_percentage = 14.5 →
  payment_percentage = 100 - savings_percentage →
  payment_percentage = 85.5 :=
by sorry

end NUMINAMATH_CALUDE_customer_payment_percentage_l3208_320809


namespace NUMINAMATH_CALUDE_notebooks_given_to_tom_l3208_320842

def bernard_notebooks (red blue white remaining : ℕ) : Prop :=
  red + blue + white - remaining = 46

theorem notebooks_given_to_tom :
  bernard_notebooks 15 17 19 5 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_given_to_tom_l3208_320842


namespace NUMINAMATH_CALUDE_total_students_sum_l3208_320830

/-- The number of students in Varsity school -/
def varsity : ℕ := 1300

/-- The number of students in Northwest school -/
def northwest : ℕ := 1400

/-- The number of students in Central school -/
def central : ℕ := 1800

/-- The number of students in Greenbriar school -/
def greenbriar : ℕ := 1650

/-- The total number of students across all schools -/
def total_students : ℕ := varsity + northwest + central + greenbriar

theorem total_students_sum :
  total_students = 6150 := by sorry

end NUMINAMATH_CALUDE_total_students_sum_l3208_320830


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l3208_320886

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ Nᶜ = Set.univ :=
sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l3208_320886


namespace NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l3208_320896

theorem smallest_integer_solution (x : ℤ) : (7 - 5*x < 12) → x ≥ 0 :=
by
  sorry

theorem smallest_integer_solution_exists : ∃ x : ℤ, (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

theorem smallest_integer_solution_is_zero : 
  ∃ x : ℤ, x = 0 ∧ (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l3208_320896


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l3208_320814

theorem opposite_of_one_half : -(1 / 2 : ℚ) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l3208_320814


namespace NUMINAMATH_CALUDE_right_triangle_area_l3208_320858

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 8 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 24 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3208_320858


namespace NUMINAMATH_CALUDE_smallest_b_value_l3208_320860

/-- Given real numbers a and b where 2 < a < b, and no triangle with positive area
    has side lengths 2, a, and b or 1/b, 1/a, and 1/2, the smallest possible value of b is 6. -/
theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2))
  (h4 : ¬ (1/b + 1/a > 1/2 ∧ 1/b + 1/2 > 1/a ∧ 1/a + 1/2 > 1/b)) :
  6 ≤ b ∧ ∀ c, (2 < c → c < b → 
    ¬(2 + c > b ∧ 2 + b > c ∧ c + b > 2) → 
    ¬(1/b + 1/c > 1/2 ∧ 1/b + 1/2 > 1/c ∧ 1/c + 1/2 > 1/b) → 
    6 ≤ c) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3208_320860


namespace NUMINAMATH_CALUDE_digit_2003_is_4_l3208_320821

/-- Calculates the digit at a given position in the sequence of natural numbers written consecutively -/
def digitAtPosition (n : ℕ) : ℕ :=
  sorry

/-- The 2003rd digit in the sequence of natural numbers written consecutively is 4 -/
theorem digit_2003_is_4 : digitAtPosition 2003 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_2003_is_4_l3208_320821


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3208_320845

theorem absolute_value_inequality (x : ℝ) :
  |x + 2| + |x + 3| ≤ 2 ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3208_320845


namespace NUMINAMATH_CALUDE_max_value_of_z_l3208_320891

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + y ≤ 4 ∧ y - 2*x + 2 ≤ 0 ∧ y ≥ 0

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + 2*y

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), system x y ∧ z x y = 6 ∧
  ∀ (x' y' : ℝ), system x' y' → z x' y' ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l3208_320891


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l3208_320878

theorem quadratic_polynomial_from_sum_and_product (s r : ℝ) :
  ∃ (a b : ℝ), a + b = s ∧ a * b = r^3 →
  ∀ x : ℝ, (x - a) * (x - b) = x^2 - s*x + r^3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l3208_320878


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_l3208_320831

theorem x_positive_sufficient_not_necessary :
  (∀ x : ℝ, x > 0 → |x - 1| - |x| ≤ 1) ∧
  (∃ x : ℝ, x ≤ 0 ∧ |x - 1| - |x| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_l3208_320831


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3208_320808

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  (1025 = 23 * q + r) ∧ 
  (∀ (q' r' : ℕ+), 1025 = 23 * q' + r' → q' - r' ≤ q - r) ∧
  (q - r = 27) := by
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3208_320808


namespace NUMINAMATH_CALUDE_distinct_pair_count_l3208_320868

theorem distinct_pair_count :
  let S := Finset.range 15
  (S.card * (S.card - 1) : ℕ) = 210 := by sorry

end NUMINAMATH_CALUDE_distinct_pair_count_l3208_320868


namespace NUMINAMATH_CALUDE_common_chord_length_O1_O2_l3208_320826

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The length of the common chord between two circles -/
def common_chord_length (c1 c2 : Circle) : ℝ := sorry

/-- Circle O₁ with equation (x+1)²+(y-3)²=9 -/
def O1 : Circle :=
  { equation := λ x y ↦ (x + 1)^2 + (y - 3)^2 = 9 }

/-- Circle O₂ with equation x²+y²-4x+2y-11=0 -/
def O2 : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 4*x + 2*y - 11 = 0 }

/-- Theorem stating that the length of the common chord between O₁ and O₂ is 24/5 -/
theorem common_chord_length_O1_O2 :
  common_chord_length O1 O2 = 24/5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_O1_O2_l3208_320826


namespace NUMINAMATH_CALUDE_lattice_point_theorem_l3208_320806

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The set of all lattice points -/
def L : Set LatticePoint := Set.univ

/-- Check if a line segment between two lattice points contains no other lattice points -/
def noInteriorLatticePoints (a b : LatticePoint) : Prop :=
  ∀ p : LatticePoint, p ∈ L → p ≠ a → p ≠ b → ¬(∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y))

theorem lattice_point_theorem :
  (∀ a b c : LatticePoint, a ∈ L → b ∈ L → c ∈ L → a ≠ b → b ≠ c → a ≠ c →
    ∃ d : LatticePoint, d ∈ L ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
      noInteriorLatticePoints a d ∧ noInteriorLatticePoints b d ∧ noInteriorLatticePoints c d) ∧
  (∃ a b c d : LatticePoint, a ∈ L ∧ b ∈ L ∧ c ∈ L ∧ d ∈ L ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (¬noInteriorLatticePoints a b ∨ ¬noInteriorLatticePoints b c ∨
     ¬noInteriorLatticePoints c d ∨ ¬noInteriorLatticePoints d a)) :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_theorem_l3208_320806


namespace NUMINAMATH_CALUDE_weight_loss_problem_l3208_320823

theorem weight_loss_problem (total_loss weight_loss_2 weight_loss_3 weight_loss_4 : ℕ) 
  (h1 : total_loss = 103)
  (h2 : weight_loss_3 = 28)
  (h3 : weight_loss_4 = 28)
  (h4 : weight_loss_2 = weight_loss_3 + weight_loss_4 - 7) :
  ∃ (weight_loss_1 : ℕ), 
    weight_loss_1 + weight_loss_2 + weight_loss_3 + weight_loss_4 = total_loss ∧ 
    weight_loss_1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_problem_l3208_320823


namespace NUMINAMATH_CALUDE_four_inequalities_true_l3208_320818

theorem four_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x < a) (hyb : y < b) 
  (hxneg : x < 0) (hyneg : y < 0)
  (hapos : a > 0) (hbpos : b > 0) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_four_inequalities_true_l3208_320818


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3208_320856

theorem sum_of_quadratic_roots (x : ℝ) : 
  -48 * x^2 + 110 * x + 165 = 0 → 
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 55 / 24 ∧ 
             -48 * x₁^2 + 110 * x₁ + 165 = 0 ∧ 
             -48 * x₂^2 + 110 * x₂ + 165 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3208_320856


namespace NUMINAMATH_CALUDE_expansion_terms_product_l3208_320813

/-- The number of terms in the expansion of a product of two polynomials -/
def expansion_terms (n m : ℕ) : ℕ := n * m

theorem expansion_terms_product (n m : ℕ) (h1 : n = 3) (h2 : m = 5) :
  expansion_terms n m = 15 := by
  sorry

#check expansion_terms_product

end NUMINAMATH_CALUDE_expansion_terms_product_l3208_320813


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3208_320800

/-- Given a rhombus with area 80 and one diagonal of length 16, 
    prove that the other diagonal has length 10. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 80) 
  (h_d1 : d1 = 16) 
  (h_rhombus : area = d1 * d2 / 2) : 
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3208_320800


namespace NUMINAMATH_CALUDE_april_sales_calculation_l3208_320864

def january_sales : ℕ := 90
def february_sales : ℕ := 50
def march_sales : ℕ := 70
def average_sales : ℕ := 72
def total_months : ℕ := 5

theorem april_sales_calculation :
  ∃ (april_sales may_sales : ℕ),
    (january_sales + february_sales + march_sales + april_sales + may_sales) / total_months = average_sales ∧
    april_sales = 75 := by
  sorry

end NUMINAMATH_CALUDE_april_sales_calculation_l3208_320864


namespace NUMINAMATH_CALUDE_sequence_properties_l3208_320836

def sequence_a (n : ℕ+) : ℚ := (1/2) ^ (n.val - 2)

def sum_S (n : ℕ+) : ℚ := 4 * (1 - (1/2) ^ n.val)

theorem sequence_properties :
  ∀ (n : ℕ+),
  (∀ (m : ℕ+), sum_S (m + 1) = (1/2) * sum_S m + 2) →
  sequence_a 1 = 2 →
  sequence_a 2 = 1 →
  (∀ (k : ℕ+), sequence_a k = (1/2) ^ (k.val - 2)) ∧
  (∀ (t : ℕ+), (∀ (n : ℕ+), (sequence_a t * sum_S (n + 1) - 1) / (sequence_a t * sequence_a (n + 1) - 1) < 1/2) ↔ (t = 3 ∨ t = 4)) ∧
  (∀ (m n k : ℕ+), m ≠ n → n ≠ k → m ≠ k → sequence_a m + sequence_a n ≠ sequence_a k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3208_320836


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3208_320872

theorem inequality_equivalence (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔ 
  (x > 0 ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3208_320872


namespace NUMINAMATH_CALUDE_complement_A_union_B_l3208_320848

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B : Set ℝ := {y | ∃ x, y = |x|}

-- State the theorem
theorem complement_A_union_B :
  (Aᶜ ∪ B) = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l3208_320848


namespace NUMINAMATH_CALUDE_total_sleep_time_in_week_l3208_320895

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep_hours : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep_hours * days_in_week) + 
  ((cougar_sleep_hours + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_time_in_week_l3208_320895


namespace NUMINAMATH_CALUDE_no_solutions_exist_l3208_320899

theorem no_solutions_exist : ¬∃ (x y : ℕ+) (m : ℕ), 
  (x : ℝ)^2 + (y : ℝ)^2 = (x : ℝ)^5 ∧ x = m^6 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l3208_320899


namespace NUMINAMATH_CALUDE_september_percentage_l3208_320898

-- Define the total number of people surveyed
def total_people : ℕ := 150

-- Define the number of people born in September
def september_births : ℕ := 12

-- Define the percentage calculation function
def percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- State the theorem
theorem september_percentage : percentage september_births total_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_september_percentage_l3208_320898


namespace NUMINAMATH_CALUDE_dress_design_count_l3208_320820

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of fabric types available -/
def num_fabric_types : ℕ := 4

/-- The number of patterns available -/
def num_patterns : ℕ := 3

/-- Each dress design requires exactly one color, one fabric type, and one pattern -/
axiom dress_design_requirements : True

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_fabric_types * num_patterns

theorem dress_design_count : total_designs = 36 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l3208_320820


namespace NUMINAMATH_CALUDE_function_derivative_value_l3208_320874

/-- Given a function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem function_derivative_value (a : ℝ) : 
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by sorry

end NUMINAMATH_CALUDE_function_derivative_value_l3208_320874


namespace NUMINAMATH_CALUDE_tiktok_house_theorem_l3208_320854

/-- Represents a 3x3 grid of bloggers --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Represents a day's arrangement of bloggers --/
def DailyArrangement := Fin 9 → Fin 3 × Fin 3

/-- Represents the three days of arrangements --/
def ThreeDayArrangements := Fin 3 → DailyArrangement

/-- Checks if two positions in the grid are adjacent --/
def are_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Counts the number of unique pairs formed over three days --/
def count_unique_pairs (arrangements : ThreeDayArrangements) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem tiktok_house_theorem (arrangements : ThreeDayArrangements) :
  count_unique_pairs arrangements < (9 * 8) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tiktok_house_theorem_l3208_320854


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l3208_320859

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l3208_320859


namespace NUMINAMATH_CALUDE_prob_at_least_three_same_l3208_320880

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling a specific value on a single die -/
def probSingleDie : ℚ := 1 / numSides

/-- The probability that at least three out of four fair six-sided dice show the same value -/
def probAtLeastThreeSame : ℚ := 7 / 72

/-- Theorem stating that the probability of at least three out of four fair six-sided dice 
    showing the same value is 7/72 -/
theorem prob_at_least_three_same :
  probAtLeastThreeSame = 
    (1 * probSingleDie ^ 3) + -- Probability of all four dice showing same value
    (4 * probSingleDie ^ 2 * (1 - probSingleDie)) -- Probability of exactly three dice showing same value
  := by sorry

end NUMINAMATH_CALUDE_prob_at_least_three_same_l3208_320880


namespace NUMINAMATH_CALUDE_pumpkin_weight_problem_l3208_320866

/-- Given two pumpkins with a total weight of 12.7 pounds, 
    if one pumpkin weighs 4 pounds, then the other pumpkin weighs 8.7 pounds. -/
theorem pumpkin_weight_problem (total_weight : ℝ) (pumpkin1_weight : ℝ) (pumpkin2_weight : ℝ) :
  total_weight = 12.7 →
  pumpkin1_weight = 4 →
  total_weight = pumpkin1_weight + pumpkin2_weight →
  pumpkin2_weight = 8.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_problem_l3208_320866


namespace NUMINAMATH_CALUDE_expression_evaluation_l3208_320894

theorem expression_evaluation : (2^3 - 2^2) - (3^3 - 3^2) + (4^3 - 4^2) - (5^3 - 5^2) = -66 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3208_320894


namespace NUMINAMATH_CALUDE_linear_function_property_l3208_320852

theorem linear_function_property (k b : ℝ) : 
  (3 = k + b) → (2 = -k + b) → k^2 - b^2 = -6 := by sorry

end NUMINAMATH_CALUDE_linear_function_property_l3208_320852


namespace NUMINAMATH_CALUDE_cracked_to_broken_ratio_l3208_320893

/-- Represents the number of eggs in each category --/
structure EggCounts where
  total : ℕ
  broken : ℕ
  perfect : ℕ
  cracked : ℕ

/-- Theorem stating the ratio of cracked to broken eggs --/
theorem cracked_to_broken_ratio (e : EggCounts) : 
  e.total = 24 →
  e.broken = 3 →
  e.perfect - e.cracked = 9 →
  e.total = e.perfect + e.cracked + e.broken →
  (e.cracked : ℚ) / e.broken = 2 := by
  sorry

#check cracked_to_broken_ratio

end NUMINAMATH_CALUDE_cracked_to_broken_ratio_l3208_320893


namespace NUMINAMATH_CALUDE_quadratic_equality_existence_l3208_320832

theorem quadratic_equality_existence (P : ℝ → ℝ) (h : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c ∧ a ≠ 0) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equality_existence_l3208_320832


namespace NUMINAMATH_CALUDE_volleyball_tournament_probabilities_l3208_320877

theorem volleyball_tournament_probabilities :
  -- Definition of probability of student team winning a match
  let p_student_win : ℝ := 1/2
  -- Definition of probability of teacher team winning a match
  let p_teacher_win : ℝ := 3/5
  -- Total number of teams
  let total_teams : ℕ := 21
  -- Number of student teams
  let student_teams : ℕ := 20
  -- Number of teams advancing directly to quarterfinals
  let direct_advance : ℕ := 5
  -- Number of teams selected by drawing
  let drawn_teams : ℕ := 2

  -- 1. Probability of a student team winning two consecutive matches
  (p_student_win * p_student_win = 1/4) ∧

  -- 2. Probability distribution of number of rounds teacher team participates
  (1 - p_teacher_win = 2/5) ∧
  (p_teacher_win * (1 - p_teacher_win) = 6/25) ∧
  (p_teacher_win * p_teacher_win = 9/25) ∧

  -- 3. Expectation of number of rounds teacher team participates
  (1 * (1 - p_teacher_win) + 2 * (p_teacher_win * (1 - p_teacher_win)) + 3 * (p_teacher_win * p_teacher_win) = 49/25) :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_probabilities_l3208_320877


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3208_320881

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem shaded_area_theorem :
  (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l3208_320881


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3208_320855

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) 
  (h3 : a = 10) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3208_320855


namespace NUMINAMATH_CALUDE_medication_expiration_time_l3208_320801

-- Define the number of seconds in 8!
def medication_duration : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the number of seconds in a minute in this system
def seconds_per_minute : ℕ := 50

-- Define the release time
def release_time : String := "3 PM on February 14"

-- Define a function to calculate the expiration time
def calculate_expiration_time (duration : ℕ) (seconds_per_min : ℕ) (start_time : String) : String :=
  sorry

-- Theorem statement
theorem medication_expiration_time :
  calculate_expiration_time medication_duration seconds_per_minute release_time = "February 15, around 4 AM" :=
sorry

end NUMINAMATH_CALUDE_medication_expiration_time_l3208_320801


namespace NUMINAMATH_CALUDE_find_S_l3208_320853

def f : ℕ → ℕ
  | 0 => 0
  | n + 1 => f n + 3

theorem find_S (S : ℕ) (h : 2 * f S = 3996) : S = 666 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l3208_320853


namespace NUMINAMATH_CALUDE_correct_testing_schemes_l3208_320883

/-- The number of genuine products -/
def genuine_products : ℕ := 5

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The position at which the last defective product is detected -/
def last_defective_position : ℕ := 6

/-- The number of ways to arrange products such that the last defective product
    is at the specified position and all defective products are included -/
def testing_schemes : ℕ := defective_products * (genuine_products.choose 2) * (last_defective_position - 1).factorial

theorem correct_testing_schemes :
  testing_schemes = 4800 := by sorry

end NUMINAMATH_CALUDE_correct_testing_schemes_l3208_320883


namespace NUMINAMATH_CALUDE_smallest_x_value_l3208_320833

theorem smallest_x_value (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^3 + ((5*x - 20)/(4*x - 5))^2 - ((5*x - 20)/(4*x - 5)) - 15 = 0) → 
  (∀ y : ℝ, (((5*y - 20)/(4*y - 5))^3 + ((5*y - 20)/(4*y - 5))^2 - ((5*y - 20)/(4*y - 5)) - 15 = 0) → 
  x ≤ y) → 
  x = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3208_320833


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3208_320819

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.75 * P) : 
  M / N = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3208_320819


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3208_320888

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x = 2 * y ∧
  2 * (x + y) = 2 * (x * y) ∧
  x = 3 ∧ y = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3208_320888


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3208_320812

theorem least_integer_greater_than_sqrt_500 : 
  (∀ n : ℕ, n ≤ 22 → n ^ 2 ≤ 500) ∧ 23 ^ 2 > 500 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3208_320812


namespace NUMINAMATH_CALUDE_certain_number_power_l3208_320862

theorem certain_number_power (k : ℕ) (h : k = 11) :
  (1/2)^22 * (1/81)^k = (1/354294)^22 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_power_l3208_320862


namespace NUMINAMATH_CALUDE_triplet_solution_l3208_320846

def is_valid_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(24,30,23), (12,30,31), (18,40,9), (15,22,36), (12,30,31)}

theorem triplet_solution :
  ∀ (a b c : ℕ), is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_triplet_solution_l3208_320846


namespace NUMINAMATH_CALUDE_blanket_thickness_proof_l3208_320810

-- Define the initial thickness of the blanket
def initial_thickness : ℝ := 3

-- Define a function that calculates the thickness after n foldings
def thickness_after_foldings (n : ℕ) : ℝ :=
  initial_thickness * (2 ^ n)

-- Theorem statement
theorem blanket_thickness_proof :
  thickness_after_foldings 4 = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_blanket_thickness_proof_l3208_320810


namespace NUMINAMATH_CALUDE_five_dollar_neg_one_eq_zero_l3208_320882

-- Define the $ operation
def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

-- Theorem statement
theorem five_dollar_neg_one_eq_zero : dollar_op 5 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_neg_one_eq_zero_l3208_320882


namespace NUMINAMATH_CALUDE_at_least_one_black_certain_l3208_320865

/-- Represents the color of a ball -/
inductive BallColor
  | Black
  | White

/-- Represents the composition of balls in the bag -/
structure BagComposition where
  blackBalls : Nat
  whiteBalls : Nat

/-- Represents the result of drawing two balls -/
structure DrawResult where
  firstBall : BallColor
  secondBall : BallColor

/-- Defines the event of drawing at least one black ball -/
def AtLeastOneBlack (result : DrawResult) : Prop :=
  result.firstBall = BallColor.Black ∨ result.secondBall = BallColor.Black

/-- The theorem to be proved -/
theorem at_least_one_black_certain (bag : BagComposition) 
    (h1 : bag.blackBalls = 2) 
    (h2 : bag.whiteBalls = 1) : 
    ∀ (result : DrawResult), AtLeastOneBlack result :=
  sorry

end NUMINAMATH_CALUDE_at_least_one_black_certain_l3208_320865


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3208_320807

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 35 / 99 ∧ x = 431 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3208_320807


namespace NUMINAMATH_CALUDE_expression_equals_36_l3208_320867

theorem expression_equals_36 (k : ℚ) : k = 13 → k * (3 - 3 / k) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l3208_320867


namespace NUMINAMATH_CALUDE_cake_recipe_flour_flour_in_recipe_l3208_320875

theorem cake_recipe_flour (salt_cups : ℕ) (flour_added : ℕ) (flour_salt_diff : ℕ) : ℕ :=
  let total_flour := salt_cups + flour_salt_diff
  total_flour

theorem flour_in_recipe :
  cake_recipe_flour 7 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_flour_in_recipe_l3208_320875


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3208_320889

/-- The x-coordinate of the intersection point of two lines -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3208_320889


namespace NUMINAMATH_CALUDE_crew_size_proof_l3208_320815

/-- The number of laborers present on a certain day -/
def present_laborers : ℕ := 10

/-- The percentage of laborers that showed up for work (as a rational number) -/
def attendance_percentage : ℚ := 385 / 1000

/-- The total number of laborers in the crew -/
def total_laborers : ℕ := 26

theorem crew_size_proof :
  (present_laborers : ℚ) / attendance_percentage = total_laborers := by
  sorry

end NUMINAMATH_CALUDE_crew_size_proof_l3208_320815


namespace NUMINAMATH_CALUDE_only_45_increases_ninefold_l3208_320892

/-- A function that inserts a zero between the tens and units digits of a natural number -/
def insertZero (n : ℕ) : ℕ :=
  10 * (n / 10) * 10 + n % 10

/-- The property that a number increases ninefold when a zero is inserted between its digits -/
def increasesNinefold (n : ℕ) : Prop :=
  insertZero n = 9 * n

theorem only_45_increases_ninefold :
  ∀ n : ℕ, n ≠ 0 → (increasesNinefold n ↔ n = 45) :=
sorry

end NUMINAMATH_CALUDE_only_45_increases_ninefold_l3208_320892


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3208_320824

theorem arithmetic_problem : 
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1/2) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3208_320824


namespace NUMINAMATH_CALUDE_fourth_black_ball_probability_l3208_320834

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball on any draw -/
def prob_black_ball (box : ColoredBallBox) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The box described in the problem -/
def problem_box : ColoredBallBox :=
  { red_balls := 3, black_balls := 4 }

theorem fourth_black_ball_probability :
  prob_black_ball problem_box = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_black_ball_probability_l3208_320834


namespace NUMINAMATH_CALUDE_boy_age_proof_l3208_320851

/-- Given a group of boys with specific average ages, prove the age of the boy not in either subgroup -/
theorem boy_age_proof (total_boys : ℕ) (total_avg : ℚ) (first_six_avg : ℚ) (last_six_avg : ℚ) :
  total_boys = 13 ∧ 
  total_avg = 50 ∧ 
  first_six_avg = 49 ∧ 
  last_six_avg = 52 →
  ∃ (middle_boy_age : ℚ), middle_boy_age = 50 :=
by sorry


end NUMINAMATH_CALUDE_boy_age_proof_l3208_320851


namespace NUMINAMATH_CALUDE_inequality_proof_l3208_320841

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 1) (h2 : b ≥ 1) (h3 : c ≥ 1) (h4 : a + b + c = 9) :
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3208_320841


namespace NUMINAMATH_CALUDE_viewer_increase_l3208_320835

/-- The number of people who watched the second baseball game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first baseball game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third baseball game -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games last week -/
def last_week_viewers : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_viewers : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

theorem viewer_increase :
  this_week_viewers - last_week_viewers = 35 := by
  sorry

end NUMINAMATH_CALUDE_viewer_increase_l3208_320835


namespace NUMINAMATH_CALUDE_fraction_equality_l3208_320847

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3208_320847


namespace NUMINAMATH_CALUDE_original_trees_eq_sum_l3208_320802

/-- The number of trees Haley originally grew in her backyard -/
def original_trees : ℕ := 20

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 4

/-- The number of trees that died in the typhoon -/
def trees_died : ℕ := 16

/-- Theorem stating that the original number of trees equals the sum of trees left and trees that died -/
theorem original_trees_eq_sum : original_trees = trees_left + trees_died := by
  sorry

end NUMINAMATH_CALUDE_original_trees_eq_sum_l3208_320802


namespace NUMINAMATH_CALUDE_homothety_composition_l3208_320822

-- Define a homothety
structure Homothety (α : Type*) [AddCommGroup α] :=
  (center : α)
  (coefficient : ℝ)

-- Define a parallel translation
structure ParallelTranslation (α : Type*) [AddCommGroup α] :=
  (vector : α)

-- Define the composition of two homotheties
def compose_homotheties {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) : (ParallelTranslation α) ⊕ (Homothety α) :=
  sorry

-- Theorem statement
theorem homothety_composition {α : Type*} [AddCommGroup α] [Module ℝ α]
  (h1 h2 : Homothety α) :
  (∃ (t : ParallelTranslation α), compose_homotheties h1 h2 = Sum.inl t ∧
    ∃ (v : α), t.vector = v ∧ (∃ (c : ℝ), v = c • (h2.center - h1.center)) ∧
    h1.coefficient * h2.coefficient = 1) ∨
  (∃ (h : Homothety α), compose_homotheties h1 h2 = Sum.inr h ∧
    ∃ (c : ℝ), h.center = h1.center + c • (h2.center - h1.center) ∧
    h.coefficient = h1.coefficient * h2.coefficient ∧
    h1.coefficient * h2.coefficient ≠ 1) :=
  sorry

end NUMINAMATH_CALUDE_homothety_composition_l3208_320822


namespace NUMINAMATH_CALUDE_intersects_implies_a_in_range_l3208_320857

/-- A function f(x) that always intersects the x-axis -/
def f (m a x : ℝ) : ℝ := m * (x^2 - 1) + x - a

/-- The property that f(x) always intersects the x-axis for all m -/
def always_intersects (a : ℝ) : Prop :=
  ∀ m : ℝ, ∃ x : ℝ, f m a x = 0

/-- Theorem: If f(x) always intersects the x-axis for all m, then a is in [-1, 1] -/
theorem intersects_implies_a_in_range (a : ℝ) :
  always_intersects a → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersects_implies_a_in_range_l3208_320857


namespace NUMINAMATH_CALUDE_factorial_divisibility_l3208_320817

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

theorem factorial_divisibility (n : ℕ) (h : binary_ones n = 1995) :
  ∃ k : ℕ, n! = k * 2^(n - 1995) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l3208_320817


namespace NUMINAMATH_CALUDE_acute_triangle_count_l3208_320805

/-- Count of integers satisfying acute triangle conditions --/
theorem acute_triangle_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    18 + 36 > x ∧ 
    18 + x > 36 ∧ 
    36 + x > 18 ∧ 
    (x > 36 → x^2 < 18^2 + 36^2) ∧ 
    (x ≤ 36 → 36^2 < 18^2 + x^2))
    (Finset.range 55)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_count_l3208_320805


namespace NUMINAMATH_CALUDE_solution_value_l3208_320873

theorem solution_value (x a : ℝ) : 
  2 * (x - 6) = -16 →
  a * (x + 3) = (1/2) * a + x →
  a^2 - (a/2) + 1 = 19 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l3208_320873


namespace NUMINAMATH_CALUDE_next_number_with_property_l3208_320870

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  ∀ n : ℕ, n > 1818 →
  (∀ m : ℕ, 1818 < m ∧ m < n → ¬has_property m) →
  has_property n →
  n = 1832 :=
sorry

end NUMINAMATH_CALUDE_next_number_with_property_l3208_320870


namespace NUMINAMATH_CALUDE_no_solution_functional_equation_l3208_320844

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x * y) = f x * f y + 2 * x * y :=
by sorry

end NUMINAMATH_CALUDE_no_solution_functional_equation_l3208_320844


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l3208_320825

/-- Represents the number of items at each price point -/
structure ItemCounts where
  fiftyc : ℕ
  twofifty : ℕ
  four : ℕ

/-- Calculates the total number of items -/
def total_items (c : ItemCounts) : ℕ :=
  c.fiftyc + c.twofifty + c.four

/-- Calculates the total cost in cents -/
def total_cost (c : ItemCounts) : ℕ :=
  50 * c.fiftyc + 250 * c.twofifty + 400 * c.four

/-- The main theorem to prove -/
theorem fifty_cent_items_count (c : ItemCounts) :
  total_items c = 50 ∧ total_cost c = 5000 → c.fiftyc = 40 := by
  sorry

#check fifty_cent_items_count

end NUMINAMATH_CALUDE_fifty_cent_items_count_l3208_320825


namespace NUMINAMATH_CALUDE_train_vs_airplane_capacity_difference_l3208_320828

/-- The passenger capacity of a single train car -/
def train_car_capacity : ℕ := 60

/-- The passenger capacity of a 747 airplane -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes -/
def num_airplanes : ℕ := 2

/-- The theorem stating the difference in passenger capacity -/
theorem train_vs_airplane_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end NUMINAMATH_CALUDE_train_vs_airplane_capacity_difference_l3208_320828


namespace NUMINAMATH_CALUDE_graduating_class_size_l3208_320876

theorem graduating_class_size :
  let num_boys : ℕ := 138
  let girls_more_than_boys : ℕ := 69
  let num_girls : ℕ := num_boys + girls_more_than_boys
  let total_students : ℕ := num_boys + num_girls
  total_students = 345 := by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l3208_320876


namespace NUMINAMATH_CALUDE_transformation_theorem_l3208_320804

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the transformation g
def g : ℝ → ℝ := sorry

-- Theorem statement
theorem transformation_theorem :
  ∀ x : ℝ, g x = -f (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_transformation_theorem_l3208_320804


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3208_320837

theorem quadratic_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3208_320837


namespace NUMINAMATH_CALUDE_cricket_team_size_l3208_320840

theorem cricket_team_size :
  ∀ (n : ℕ),
  (n : ℝ) * 23 = (n - 2 : ℝ) * 22 + 55 →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3208_320840


namespace NUMINAMATH_CALUDE_cooldrink_mixture_l3208_320871

/-- Amount of Cool-drink B added to create a mixture with 10% jasmine water -/
theorem cooldrink_mixture (total_volume : ℝ) (cooldrink_a_volume : ℝ) (jasmine_water_added : ℝ) (fruit_juice_added : ℝ)
  (cooldrink_a_jasmine_percent : ℝ) (cooldrink_a_fruit_percent : ℝ)
  (cooldrink_b_jasmine_percent : ℝ) (cooldrink_b_fruit_percent : ℝ)
  (final_jasmine_percent : ℝ) :
  total_volume = 150 →
  cooldrink_a_volume = 80 →
  jasmine_water_added = 8 →
  fruit_juice_added = 20 →
  cooldrink_a_jasmine_percent = 0.12 →
  cooldrink_a_fruit_percent = 0.88 →
  cooldrink_b_jasmine_percent = 0.05 →
  cooldrink_b_fruit_percent = 0.95 →
  final_jasmine_percent = 0.10 →
  ∃ cooldrink_b_volume : ℝ,
    cooldrink_b_volume = 136 ∧
    (cooldrink_a_volume * cooldrink_a_jasmine_percent + cooldrink_b_volume * cooldrink_b_jasmine_percent + jasmine_water_added) / 
    (cooldrink_a_volume + cooldrink_b_volume + jasmine_water_added + fruit_juice_added) = final_jasmine_percent :=
by
  sorry

end NUMINAMATH_CALUDE_cooldrink_mixture_l3208_320871


namespace NUMINAMATH_CALUDE_student_composition_l3208_320838

/-- The number of ways to select participants from a group of students -/
def selectionWays (males females : ℕ) : ℕ :=
  males * (males - 1) * females

theorem student_composition :
  ∃ (males females : ℕ),
    males + females = 8 ∧
    selectionWays males females = 90 →
    males = 3 ∧ females = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_composition_l3208_320838


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3208_320879

theorem simplify_sqrt_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.sqrt (3 * a)) / (Real.sqrt (12 * a * b)) = (Real.sqrt b) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3208_320879


namespace NUMINAMATH_CALUDE_distribute_5_3_eq_31_l3208_320884

/-- The number of ways to distribute n different items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different items into 3 identical bags -/
def distribute_5_3 : ℕ := distribute 5 3

theorem distribute_5_3_eq_31 : distribute_5_3 = 31 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_eq_31_l3208_320884


namespace NUMINAMATH_CALUDE_events_related_95_percent_confidence_l3208_320827

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relation between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_95_percent_confidence (K : ℝ) :
  events_related K ↔ K^2 > critical_value :=
sorry

end NUMINAMATH_CALUDE_events_related_95_percent_confidence_l3208_320827


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_3_equals_1_l3208_320890

theorem sqrt_2x_minus_3_equals_1 (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_3_equals_1_l3208_320890
