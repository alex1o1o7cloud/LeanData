import Mathlib

namespace NUMINAMATH_CALUDE_square_area_error_l1081_108140

theorem square_area_error (S : ℝ) (h : S > 0) : 
  let measured_side := S * (1 + 0.06)
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1081_108140


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1081_108166

theorem triangle_sine_inequality (A B C : Real) :
  A + B + C = 180 →
  0 < A ∧ A ≤ 180 →
  0 < B ∧ B ≤ 180 →
  0 < C ∧ C ≤ 180 →
  Real.sin ((A - 30) * π / 180) + Real.sin ((B - 30) * π / 180) + Real.sin ((C - 30) * π / 180) ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1081_108166


namespace NUMINAMATH_CALUDE_linear_function_proof_l1081_108106

/-- A linear function passing through three given points -/
def linear_function (x : ℝ) : ℝ := 3 * x + 4

/-- Theorem stating that the linear function passes through the given points and f(40) = 124 -/
theorem linear_function_proof :
  (linear_function 2 = 10) ∧
  (linear_function 6 = 22) ∧
  (linear_function 10 = 34) ∧
  (linear_function 40 = 124) := by
  sorry

#check linear_function_proof

end NUMINAMATH_CALUDE_linear_function_proof_l1081_108106


namespace NUMINAMATH_CALUDE_candies_in_box_l1081_108145

def initial_candies : ℕ := 88
def diana_takes : ℕ := 6
def john_adds : ℕ := 12
def sara_takes : ℕ := 20

theorem candies_in_box : 
  initial_candies - diana_takes + john_adds - sara_takes = 74 :=
by sorry

end NUMINAMATH_CALUDE_candies_in_box_l1081_108145


namespace NUMINAMATH_CALUDE_circle_area_increase_l1081_108159

theorem circle_area_increase (π : ℝ) (h : π > 0) : 
  π * 5^2 - π * 2^2 = 21 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1081_108159


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l1081_108128

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : 
  total_runs = 120 →
  boundaries = 6 →
  sixes = 4 →
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l1081_108128


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1081_108116

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  (p.factorial - 1) % p = 0 ↔ Nat.Prime p := by sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1081_108116


namespace NUMINAMATH_CALUDE_special_matrix_product_l1081_108167

/-- A 5x5 matrix with special properties -/
structure SpecialMatrix where
  a : Fin 5 → Fin 5 → ℝ
  first_row_arithmetic : ∀ i j k : Fin 5, a 0 j - a 0 i = a 0 k - a 0 j
    → j - i = k - j
  columns_geometric : ∃ q : ℝ, ∀ i j : Fin 5, a (i+1) j = q * a i j
  a24_eq_4 : a 1 3 = 4
  a41_eq_neg2 : a 3 0 = -2
  a43_eq_10 : a 3 2 = 10

/-- The product of a₁₁ and a₅₅ is -11 -/
theorem special_matrix_product (m : SpecialMatrix) : m.a 0 0 * m.a 4 4 = -11 := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_product_l1081_108167


namespace NUMINAMATH_CALUDE_locus_of_centers_l1081_108184

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16

-- Define the property of being externally tangent to C1 and internally tangent to C2
def externally_internally_tangent (a b r : ℝ) : Prop :=
  (∃ x y : ℝ, C1 x y ∧ (a - x)^2 + (b - y)^2 = (r + 1)^2) ∧
  (∃ x y : ℝ, C2 x y ∧ (a - x)^2 + (b - y)^2 = (4 - r)^2)

-- State the theorem
theorem locus_of_centers :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_internally_tangent a b r) ↔
  84 * a^2 + 100 * b^2 - 168 * a - 441 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1081_108184


namespace NUMINAMATH_CALUDE_minimize_resistance_l1081_108178

/-- Represents the resistance of a component assembled using six resistors. -/
noncomputable def totalResistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (R₁ R₂ R₃ R₄ R₅ R₆ : ℝ) : ℝ :=
  sorry -- Definition of total resistance based on the given configuration

/-- Theorem stating the condition for minimizing the total resistance of the component. -/
theorem minimize_resistance
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄) (h₄ : a₄ > a₅) (h₅ : a₅ > a₆)
  (h₆ : a₁ > 0) (h₇ : a₂ > 0) (h₈ : a₃ > 0) (h₉ : a₄ > 0) (h₁₀ : a₅ > 0) (h₁₁ : a₆ > 0) :
  ∃ (R₁ R₂ : ℝ), 
    (R₁ = a₁ ∧ R₂ = a₂) ∨ (R₁ = a₂ ∧ R₂ = a₁) ∧
    ∀ (S₁ S₂ S₃ S₄ S₅ S₆ : ℝ),
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ R₁ R₂ a₃ a₄ a₅ a₆ ≤ 
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ S₁ S₂ S₃ S₄ S₅ S₆ :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_resistance_l1081_108178


namespace NUMINAMATH_CALUDE_f_13_equals_219_l1081_108112

def f (n : ℕ) : ℕ := n^2 + 3*n + 11

theorem f_13_equals_219 : f 13 = 219 := by sorry

end NUMINAMATH_CALUDE_f_13_equals_219_l1081_108112


namespace NUMINAMATH_CALUDE_tank_emptying_time_difference_l1081_108143

/-- Proves the time difference for emptying a tank with and without an inlet pipe. -/
theorem tank_emptying_time_difference 
  (tank_capacity : ℝ) 
  (outlet_rate : ℝ) 
  (inlet_rate : ℝ) 
  (h1 : tank_capacity = 21600) 
  (h2 : outlet_rate = 2160) 
  (h3 : inlet_rate = 960) : 
  (tank_capacity / outlet_rate) - (tank_capacity / (outlet_rate - inlet_rate)) = 8 := by
  sorry

#check tank_emptying_time_difference

end NUMINAMATH_CALUDE_tank_emptying_time_difference_l1081_108143


namespace NUMINAMATH_CALUDE_number_squared_sum_equals_100_l1081_108115

theorem number_squared_sum_equals_100 : ∃ x : ℝ, (7.5 * 7.5) + 37.5 + (x * x) = 100 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_squared_sum_equals_100_l1081_108115


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1081_108181

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1081_108181


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l1081_108124

/-- The smallest possible area of a right triangle with two sides measuring 6 and 8 units is 24 square units. -/
theorem smallest_area_right_triangle : ℝ := by
  -- Let a and b be the two given sides of the right triangle
  let a : ℝ := 6
  let b : ℝ := 8
  
  -- Define the function to calculate the area of a right triangle
  let area (x y : ℝ) : ℝ := (1 / 2) * x * y
  
  -- State that the smallest area is 24
  let smallest_area : ℝ := 24
  
  sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l1081_108124


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_three_l1081_108125

/-- Right triangle PQR with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Angle R is a right angle -/
  angle_r_is_right : True

/-- Calculate the radius of the inscribed circle in a right triangle -/
def inscribedCircleRadius (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle in the given right triangle is 3 -/
theorem inscribed_circle_radius_is_three :
  let t : RightTriangleWithInscribedCircle := ⟨15, 8, trivial⟩
  inscribedCircleRadius t = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_three_l1081_108125


namespace NUMINAMATH_CALUDE_rectangular_plot_area_breadth_ratio_l1081_108151

theorem rectangular_plot_area_breadth_ratio :
  let breadth : ℕ := 13
  let length : ℕ := breadth + 10
  let area : ℕ := length * breadth
  area / breadth = 23 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_breadth_ratio_l1081_108151


namespace NUMINAMATH_CALUDE_rose_bush_price_is_75_l1081_108169

-- Define the given conditions
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def aloe_count : ℕ := 2
def aloe_price : ℕ := 100
def total_spent_self : ℕ := 500

-- Define the function to calculate the price of each rose bush
def rose_bush_price : ℕ :=
  let self_rose_bushes := total_rose_bushes - friend_rose_bushes
  let aloe_total := aloe_count * aloe_price
  let rose_bushes_total := total_spent_self - aloe_total
  rose_bushes_total / self_rose_bushes

-- Theorem statement
theorem rose_bush_price_is_75 : rose_bush_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_price_is_75_l1081_108169


namespace NUMINAMATH_CALUDE_tank_filling_l1081_108186

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ new_buckets : ℕ, 
    new_buckets = ⌈(original_buckets : ℚ) / capacity_ratio⌉ ∧
    new_buckets = 63 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_l1081_108186


namespace NUMINAMATH_CALUDE_original_expenditure_l1081_108150

/-- Represents the hostel mess expenditure problem -/
structure HostelMess where
  initial_students : ℕ
  initial_expenditure : ℕ
  initial_avg_expenditure : ℕ

/-- Represents changes in the hostel mess -/
structure MessChange where
  day : ℕ
  student_change : ℤ
  expense_change : ℕ
  avg_expenditure_change : ℤ

/-- Theorem stating the original expenditure of the mess -/
theorem original_expenditure (mess : HostelMess) 
  (change1 : MessChange) (change2 : MessChange) (change3 : MessChange) : 
  mess.initial_students = 35 →
  change1.day = 10 → change1.student_change = 7 → change1.expense_change = 84 → change1.avg_expenditure_change = -1 →
  change2.day = 15 → change2.student_change = -5 → change2.expense_change = 40 → change2.avg_expenditure_change = 2 →
  change3.day = 25 → change3.student_change = 3 → change3.expense_change = 30 → change3.avg_expenditure_change = 0 →
  mess.initial_expenditure = 630 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_l1081_108150


namespace NUMINAMATH_CALUDE_marble_arrangement_l1081_108144

/-- Represents the number of green marbles -/
def green_marbles : Nat := 4

/-- Represents the number of red marbles -/
def red_marbles : Nat := 3

/-- Represents the maximum number of blue marbles that can be used to create a balanced arrangement -/
def m : Nat := 5

/-- Represents the total number of slots where blue marbles can be placed -/
def total_slots : Nat := green_marbles + red_marbles + 1

/-- Calculates the number of ways to arrange the marbles -/
def N : Nat := Nat.choose (m + total_slots - 1) m

/-- Theorem stating the properties of the marble arrangement -/
theorem marble_arrangement :
  (N % 1000 = 287) ∧
  (∀ k : Nat, k > m → Nat.choose (k + total_slots - 1) k % 1000 ≠ 287) := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_l1081_108144


namespace NUMINAMATH_CALUDE_cube_of_99999_l1081_108146

theorem cube_of_99999 : 
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_99999_l1081_108146


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l1081_108122

def batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let previous_average := (total_innings - 1) * (average_increase + (last_innings_score / total_innings))
  let new_total_score := previous_average + last_innings_score
  new_total_score / total_innings

theorem batsman_average_theorem :
  batsman_average 17 80 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l1081_108122


namespace NUMINAMATH_CALUDE_brick_height_l1081_108190

/-- Represents a rectangular solid brick made of unit cubes -/
structure RectangularBrick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular brick in unit cubes -/
def RectangularBrick.volume (brick : RectangularBrick) : ℕ :=
  brick.length * brick.width * brick.height

/-- The perimeter of the base of a rectangular brick -/
def RectangularBrick.basePerimeter (brick : RectangularBrick) : ℕ :=
  2 * (brick.length + brick.width)

theorem brick_height (brick : RectangularBrick) :
  brick.volume = 42 ∧
  brick.basePerimeter = 18 →
  brick.height = 3 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_l1081_108190


namespace NUMINAMATH_CALUDE_existence_of_large_subset_l1081_108126

/-- A family of 3-element subsets with at most one common element between any two subsets -/
def ValidFamily (I : Finset Nat) (A : Set (Finset Nat)) : Prop :=
  ∀ a ∈ A, a.card = 3 ∧ a ⊆ I ∧ ∀ b ∈ A, a ≠ b → (a ∩ b).card ≤ 1

/-- The theorem statement -/
theorem existence_of_large_subset (n : Nat) (I : Finset Nat) (hI : I.card = n) 
    (A : Set (Finset Nat)) (hA : ValidFamily I A) :
  ∃ X : Finset Nat, X ⊆ I ∧ 
    (∀ a ∈ A, ¬(a ⊆ X)) ∧ 
    X.card ≥ Nat.floor (Real.sqrt (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_subset_l1081_108126


namespace NUMINAMATH_CALUDE_rectangle_area_l1081_108130

theorem rectangle_area (x y : ℝ) 
  (h_perimeter : x + y = 5)
  (h_diagonal : x^2 + y^2 = 15) : 
  x * y = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1081_108130


namespace NUMINAMATH_CALUDE_jade_transactions_l1081_108100

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 15 →
  jade = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l1081_108100


namespace NUMINAMATH_CALUDE_four_wheeler_wheels_l1081_108127

theorem four_wheeler_wheels (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) : 
  num_four_wheelers = 17 → wheels_per_four_wheeler = 4 → num_four_wheelers * wheels_per_four_wheeler = 68 := by
  sorry

end NUMINAMATH_CALUDE_four_wheeler_wheels_l1081_108127


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1081_108132

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x ≤ 0) ↔ (0 ≤ x ∧ x ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1081_108132


namespace NUMINAMATH_CALUDE_f_extremum_f_two_zeros_harmonic_sum_bound_l1081_108170

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / x + a * Real.log (1 / x)

theorem f_extremum :
  let f₁ := f 1
  (∃ x₀ > 0, ∀ x > 0, f₁ x ≤ f₁ x₀) ∧
  f₁ 1 = 0 ∧
  (¬∃ x₀ > 0, ∀ x > 0, f₁ x ≥ f₁ x₀) := by sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, 1 / Real.exp 1 < x ∧ x < y ∧ y < Real.exp 1 ∧ f a x = 0 ∧ f a y = 0) ↔
  (Real.exp 1 / (Real.exp 1 + 1) < a ∧ a < 1) := by sorry

theorem harmonic_sum_bound (n : ℕ) (hn : n ≥ 3) :
  Real.log ((n + 1) / 3) < (Finset.range (n - 2)).sum (λ i => 1 / (i + 3 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_f_extremum_f_two_zeros_harmonic_sum_bound_l1081_108170


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1081_108135

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1081_108135


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1081_108180

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) < 12 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1081_108180


namespace NUMINAMATH_CALUDE_distance_between_points_l1081_108187

/-- The distance between two points (2, -7) and (-8, 4) is √221. -/
theorem distance_between_points : Real.sqrt 221 = Real.sqrt ((2 - (-8))^2 + ((-7) - 4)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1081_108187


namespace NUMINAMATH_CALUDE_total_money_l1081_108147

def cecil_money : ℕ := 600

def catherine_money : ℕ := 2 * cecil_money - 250

def carmela_money : ℕ := 2 * cecil_money + 50

theorem total_money : cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1081_108147


namespace NUMINAMATH_CALUDE_sin_two_phi_l1081_108174

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_two_phi_l1081_108174


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l1081_108138

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (P : ℝ × ℝ) 
  (h_P : P = (4, -2)) 
  (Q : ℝ × ℝ) 
  (h_Q : Q.1^2 + Q.2^2 = 4) 
  (M : ℝ × ℝ) 
  (h_M : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + (M.2 + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l1081_108138


namespace NUMINAMATH_CALUDE_triangular_number_representation_l1081_108173

theorem triangular_number_representation (n : ℕ+) :
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ l < k :=
by sorry

end NUMINAMATH_CALUDE_triangular_number_representation_l1081_108173


namespace NUMINAMATH_CALUDE_max_grandchildren_l1081_108107

/-- Calculates the number of grandchildren for a person with given conditions -/
def grandchildren_count (num_children : ℕ) (num_same_children : ℕ) (num_five_children : ℕ) (five_children : ℕ) : ℕ :=
  (num_same_children * num_children) + (num_five_children * five_children)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  let num_children := 8
  let num_same_children := 6
  let num_five_children := 2
  let five_children := 5
  grandchildren_count num_children num_same_children num_five_children five_children = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_grandchildren_l1081_108107


namespace NUMINAMATH_CALUDE_larger_number_theorem_l1081_108175

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem larger_number_theorem (a b : ℕ) 
  (h1 : Nat.gcd a b = 37)
  (h2 : is_prime 37)
  (h3 : ∃ (k : ℕ), Nat.lcm a b = k * 37 * 17 * 23 * 29 * 31) :
  max a b = 13007833 := by
sorry

end NUMINAMATH_CALUDE_larger_number_theorem_l1081_108175


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l1081_108189

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Right angle at F (we don't need to specify this explicitly in Lean)
  true

def angle_D_is_60_degrees (t : Triangle) : Prop :=
  -- Angle D is 60 degrees
  true

def DF_length (t : Triangle) : ℝ :=
  12

-- Define the incircle radius function
noncomputable def incircle_radius (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_D_is_60_degrees t)
  (h3 : DF_length t = 12) :
  incircle_radius t = 6 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l1081_108189


namespace NUMINAMATH_CALUDE_inequality_preservation_l1081_108113

theorem inequality_preservation (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1081_108113


namespace NUMINAMATH_CALUDE_juggling_balls_needed_l1081_108104

/-- The number of balls needed for a juggling spectacle -/
theorem juggling_balls_needed (num_jugglers : ℕ) (balls_per_juggler : ℕ) : 
  num_jugglers = 5000 → balls_per_juggler = 12 → num_jugglers * balls_per_juggler = 60000 := by
  sorry

end NUMINAMATH_CALUDE_juggling_balls_needed_l1081_108104


namespace NUMINAMATH_CALUDE_tank_circumference_l1081_108134

/-- Given two right circular cylindrical tanks C and B, prove that the circumference of tank B is 10 meters. -/
theorem tank_circumference (h_C h_B r_C r_B : ℝ) : 
  h_C = 10 →  -- Height of tank C
  h_B = 8 →   -- Height of tank B
  2 * Real.pi * r_C = 8 →  -- Circumference of tank C
  (Real.pi * r_C^2 * h_C) = 0.8 * (Real.pi * r_B^2 * h_B) →  -- Volume relation
  2 * Real.pi * r_B = 10  -- Circumference of tank B
:= by sorry

end NUMINAMATH_CALUDE_tank_circumference_l1081_108134


namespace NUMINAMATH_CALUDE_area_of_2015_l1081_108172

/-- Represents a grid composed of 1x1 squares -/
structure Grid where
  squares : Set (Int × Int)

/-- Represents a shaded region in the grid -/
inductive ShadedRegion
  | Horizontal : (Int × Int) → (Int × Int) → ShadedRegion
  | Vertical : (Int × Int) → (Int × Int) → ShadedRegion
  | Diagonal : (Int × Int) → (Int × Int) → ShadedRegion
  | Midpoint : (Int × Int) → (Int × Int) → ShadedRegion

/-- The set of shaded regions representing the number 2015 -/
def number2015 : Set ShadedRegion :=
  sorry

/-- Calculates the area of a set of shaded regions -/
def areaOfShadedRegions (regions : Set ShadedRegion) : ℚ :=
  sorry

/-- Theorem stating that the area of the shaded regions representing 2015 is 47½ -/
theorem area_of_2015 (g : Grid) :
  areaOfShadedRegions number2015 = 47 + (1/2) :=
sorry

end NUMINAMATH_CALUDE_area_of_2015_l1081_108172


namespace NUMINAMATH_CALUDE_shipwreck_year_conversion_l1081_108103

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal (octal : Nat) : Nat :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal year of the shipwreck -/
def shipwreck_year_octal : Nat := 536

theorem shipwreck_year_conversion :
  octal_to_decimal shipwreck_year_octal = 350 := by
  sorry

end NUMINAMATH_CALUDE_shipwreck_year_conversion_l1081_108103


namespace NUMINAMATH_CALUDE_parentheses_removal_equality_l1081_108102

theorem parentheses_removal_equality (a c : ℝ) : 3*a - (2*a - c) = 3*a - 2*a + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equality_l1081_108102


namespace NUMINAMATH_CALUDE_only_set2_forms_triangle_l1081_108199

-- Define a structure for a set of three line segments
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality check
def satisfiesTriangleInequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def set1 : LineSegmentSet := ⟨1, 2, 3⟩
def set2 : LineSegmentSet := ⟨2, 3, 4⟩
def set3 : LineSegmentSet := ⟨4, 4, 8⟩
def set4 : LineSegmentSet := ⟨5, 6, 12⟩

-- Theorem stating that only set2 satisfies the triangle inequality
theorem only_set2_forms_triangle :
  ¬(satisfiesTriangleInequality set1) ∧
  (satisfiesTriangleInequality set2) ∧
  ¬(satisfiesTriangleInequality set3) ∧
  ¬(satisfiesTriangleInequality set4) :=
sorry

end NUMINAMATH_CALUDE_only_set2_forms_triangle_l1081_108199


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l1081_108191

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_condition (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
  x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l1081_108191


namespace NUMINAMATH_CALUDE_square_perimeter_l1081_108136

theorem square_perimeter (a : ℝ) : 
  a > 0 → 
  let l := a * Real.sqrt 2
  let d := l / 2 + l / 4 + l / 8 + l / 16
  d = 15 * Real.sqrt 2 → 
  4 * a = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1081_108136


namespace NUMINAMATH_CALUDE_S_subset_T_l1081_108160

open Set Real

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ k : ℤ, p.1^2 - p.2^2 = 2*k + 1}

def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | sin (2*π*p.1^2) - sin (2*π*p.2^2) = cos (2*π*p.1^2) - cos (2*π*p.2^2)}

theorem S_subset_T : S ⊆ T := by
  sorry

end NUMINAMATH_CALUDE_S_subset_T_l1081_108160


namespace NUMINAMATH_CALUDE_tourist_ratio_l1081_108165

theorem tourist_ratio (initial_tourists : ℕ) (eaten_by_anaconda : ℕ) (final_tourists : ℕ) :
  initial_tourists = 30 →
  eaten_by_anaconda = 2 →
  final_tourists = 16 →
  ∃ (poisoned_tourists : ℕ),
    poisoned_tourists * 1 = (initial_tourists - eaten_by_anaconda - final_tourists) * 2 :=
by sorry

end NUMINAMATH_CALUDE_tourist_ratio_l1081_108165


namespace NUMINAMATH_CALUDE_free_throws_stats_l1081_108121

def free_throws : List ℝ := [20, 12, 22, 25, 10, 16, 15, 12, 30, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem free_throws_stats :
  median free_throws = 15.5 ∧ mean free_throws = 17.2 := by sorry

end NUMINAMATH_CALUDE_free_throws_stats_l1081_108121


namespace NUMINAMATH_CALUDE_square_roots_problem_l1081_108182

theorem square_roots_problem (a : ℝ) :
  (∃ x > 0, (2*a - 1)^2 = x ∧ (a - 2)^2 = x) → (2*a - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1081_108182


namespace NUMINAMATH_CALUDE_min_desks_for_arrangements_l1081_108149

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The number of different seating arrangements -/
def num_arrangements : ℕ := 2

/-- The minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

/-- A function that calculates the number of valid seating arrangements
    given the number of desks -/
def valid_arrangements (num_desks : ℕ) : ℕ := sorry

/-- Theorem stating that 5 is the minimum number of desks required -/
theorem min_desks_for_arrangements :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ m : ℕ, m < n → valid_arrangements m < num_arrangements) ∧
  valid_arrangements n = num_arrangements :=
sorry

end NUMINAMATH_CALUDE_min_desks_for_arrangements_l1081_108149


namespace NUMINAMATH_CALUDE_third_player_wins_probability_l1081_108154

/-- Represents a game where players take turns tossing a fair six-sided die. -/
structure DieTossingGame where
  num_players : ℕ
  target_player : ℕ
  prob_six : ℚ

/-- The probability that the target player is the first to toss a six. -/
noncomputable def probability_target_wins (game : DieTossingGame) : ℚ :=
  sorry

/-- Theorem stating the probability of the third player being the first to toss a six
    in a four-player game. -/
theorem third_player_wins_probability :
  let game := DieTossingGame.mk 4 3 (1/6)
  probability_target_wins game = 125/671 := by
  sorry

end NUMINAMATH_CALUDE_third_player_wins_probability_l1081_108154


namespace NUMINAMATH_CALUDE_proportion_problem_l1081_108101

theorem proportion_problem (x : ℝ) : (18 / 12 = x / (6 * 60)) → x = 540 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l1081_108101


namespace NUMINAMATH_CALUDE_bike_only_households_l1081_108152

theorem bike_only_households (total : ℕ) (neither : ℕ) (both : ℕ) (with_car : ℕ) :
  total = 90 →
  neither = 11 →
  both = 14 →
  with_car = 44 →
  total - neither - (with_car - both) - both = 35 :=
by sorry

end NUMINAMATH_CALUDE_bike_only_households_l1081_108152


namespace NUMINAMATH_CALUDE_total_homework_time_l1081_108161

-- Define the time left for each person
def jacob_time : ℕ := 18
def greg_time : ℕ := jacob_time - 6
def patrick_time : ℕ := 2 * greg_time - 4

-- Theorem to prove
theorem total_homework_time :
  jacob_time + greg_time + patrick_time = 50 :=
by sorry

end NUMINAMATH_CALUDE_total_homework_time_l1081_108161


namespace NUMINAMATH_CALUDE_triangle_area_l1081_108164

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the point L on EF
def L (t : Triangle) : ℝ × ℝ := sorry

-- State that DL is an altitude of triangle DEF
def is_altitude (t : Triangle) : Prop :=
  let (dx, dy) := t.D
  let (lx, ly) := L t
  (lx - dx) * (t.F.1 - t.E.1) + (ly - dy) * (t.F.2 - t.E.2) = 0

-- Define the lengths
def DE (t : Triangle) : ℝ := sorry
def EL (t : Triangle) : ℝ := sorry
def EF (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area (t : Triangle) 
  (h1 : is_altitude t)
  (h2 : DE t = 14)
  (h3 : EL t = 9)
  (h4 : EF t = 17) :
  let area := (EF t * Real.sqrt ((DE t)^2 - (EL t)^2)) / 2
  area = (17 * Real.sqrt 115) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l1081_108164


namespace NUMINAMATH_CALUDE_beth_initial_coins_l1081_108139

theorem beth_initial_coins (initial_coins : ℕ) : 
  (initial_coins + 35) / 2 = 80 → initial_coins = 125 := by
  sorry

end NUMINAMATH_CALUDE_beth_initial_coins_l1081_108139


namespace NUMINAMATH_CALUDE_max_garden_area_l1081_108197

/-- Represents a rectangular garden with one side bounded by a house. -/
structure Garden where
  width : ℝ
  length : ℝ

/-- The total fencing available -/
def total_fencing : ℝ := 500

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the amount of fencing used for three sides of the garden -/
def fencing_used (g : Garden) : ℝ := g.length + 2 * g.width

/-- Theorem stating the maximum area of the garden -/
theorem max_garden_area :
  ∃ (g : Garden), fencing_used g = total_fencing ∧
    ∀ (h : Garden), fencing_used h = total_fencing → garden_area h ≤ garden_area g ∧
    garden_area g = 31250 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l1081_108197


namespace NUMINAMATH_CALUDE_alcohol_percentage_x_is_correct_l1081_108176

/-- The percentage of alcohol by volume in solution x -/
def alcohol_percentage_x : ℝ := 0.10

/-- The percentage of alcohol by volume in solution y -/
def alcohol_percentage_y : ℝ := 0.30

/-- The volume of solution y in milliliters -/
def volume_y : ℝ := 600

/-- The volume of solution x in milliliters -/
def volume_x : ℝ := 200

/-- The percentage of alcohol by volume in the final mixture -/
def alcohol_percentage_final : ℝ := 0.25

theorem alcohol_percentage_x_is_correct :
  alcohol_percentage_x * volume_x + alcohol_percentage_y * volume_y =
  alcohol_percentage_final * (volume_x + volume_y) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_x_is_correct_l1081_108176


namespace NUMINAMATH_CALUDE_rectangle_area_l1081_108198

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 4 → w * l = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1081_108198


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l1081_108153

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) = (n + 1) * n / 2) → n! = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l1081_108153


namespace NUMINAMATH_CALUDE_shaded_area_is_thirty_l1081_108168

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_thirty (t : IsoscelesRightTriangle) : 
  (t.leg_length^2 / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_thirty_l1081_108168


namespace NUMINAMATH_CALUDE_premium_increases_after_accident_l1081_108110

/-- Represents a car insurance policy -/
structure CarInsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents the renewal of a car insurance policy -/
def renewPolicy (policy : CarInsurancePolicy) : CarInsurancePolicy :=
  { premium := if policy.hadAccident then policy.premium + 1 else policy.premium
    hadAccident := false }

/-- Theorem stating that the premium increases after an accident -/
theorem premium_increases_after_accident (policy : CarInsurancePolicy) :
  policy.hadAccident → (renewPolicy policy).premium > policy.premium :=
by sorry

#check premium_increases_after_accident

end NUMINAMATH_CALUDE_premium_increases_after_accident_l1081_108110


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l1081_108163

def S (n : ℕ) : ℤ := n^2 - 2*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem sum_of_specific_terms : a 3 + a 17 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l1081_108163


namespace NUMINAMATH_CALUDE_number_ratio_l1081_108183

theorem number_ratio (A B C : ℝ) 
  (h1 : (A + B + C) / (A + B - C) = 4/3)
  (h2 : (A + B) / (B + C) = 7/6) :
  ∃ (k : ℝ), k ≠ 0 ∧ A = 2*k ∧ B = 5*k ∧ C = k := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1081_108183


namespace NUMINAMATH_CALUDE_increasing_on_negative_reals_l1081_108137

theorem increasing_on_negative_reals (x₁ x₂ : ℝ) (h1 : x₁ < 0) (h2 : x₂ < 0) (h3 : x₁ < x₂) :
  2 * x₁ < 2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_increasing_on_negative_reals_l1081_108137


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1081_108188

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), Int.floor a = a → a^2 * b = 0) ↔
  (∃ (a b : ℝ), Int.floor a = a ∧ a^2 * b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1081_108188


namespace NUMINAMATH_CALUDE_egypt_promotion_free_tourists_l1081_108171

/-- Represents the number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- Represents the number of tourists who came on their own -/
def self_tourists : ℕ := 13

/-- Represents the number of tourists who brought no one -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_free_tourists :
  ∃ (total_tourists : ℕ),
    total_tourists = self_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists * 4 + self_tourists = free_tourists + no_referral_tourists :=
by sorry

end NUMINAMATH_CALUDE_egypt_promotion_free_tourists_l1081_108171


namespace NUMINAMATH_CALUDE_root_sum_product_l1081_108133

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 4*a^2 - a - 1 = 0) → 
  (b^4 - 4*b^2 - b - 1 = 0) → 
  (a + b) * (a * b + 1) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l1081_108133


namespace NUMINAMATH_CALUDE_slope_product_negative_one_l1081_108105

/-- Two lines with slopes that differ by 45° and are negative reciprocals have a slope product of -1 -/
theorem slope_product_negative_one (m n : ℝ) : 
  (∃ θ : ℝ, m = Real.tan (θ + π/4) ∧ n = Real.tan θ) →  -- L₁ makes 45° larger angle than L₂
  m = -1/n →                                           -- slopes are negative reciprocals
  m * n = -1 :=                                        -- product of slopes is -1
by sorry

end NUMINAMATH_CALUDE_slope_product_negative_one_l1081_108105


namespace NUMINAMATH_CALUDE_star_calculation_l1081_108196

-- Define the new operation
def star (m n : Int) : Int := m - n + 1

-- Theorem statement
theorem star_calculation : star (star 2 3) 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1081_108196


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1081_108192

theorem min_value_quadratic_sum :
  ∀ x y : ℝ, (2*x - y + 3)^2 + (x + 2*y - 1)^2 ≥ 295/72 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1081_108192


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1081_108157

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1081_108157


namespace NUMINAMATH_CALUDE_fraction_cube_three_fourths_l1081_108193

theorem fraction_cube_three_fourths : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_three_fourths_l1081_108193


namespace NUMINAMATH_CALUDE_expression_evaluation_l1081_108129

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x * (1 / x)) * (y / (1 / y)) = 1 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1081_108129


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_four_to_500_l1081_108117

theorem x_one_minus_f_equals_four_to_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_four_to_500_l1081_108117


namespace NUMINAMATH_CALUDE_max_value_theorem_l1081_108177

theorem max_value_theorem (u v : ℝ) 
  (h1 : 2 * u + 3 * v ≤ 10) 
  (h2 : 4 * u + v ≤ 9) : 
  u + 2 * v ≤ 6.1 ∧ ∃ (u₀ v₀ : ℝ), 2 * u₀ + 3 * v₀ ≤ 10 ∧ 4 * u₀ + v₀ ≤ 9 ∧ u₀ + 2 * v₀ = 6.1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1081_108177


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l1081_108131

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a * x + 1 > 0}

-- Theorem for part (I)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 6 ≤ x ∧ x < 10}) := by sorry

-- Theorem for part (II)
theorem range_of_a :
  ∀ a : ℝ, (A ∩ C a = A) → a ∈ Set.Ici (-1/6) := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l1081_108131


namespace NUMINAMATH_CALUDE_smallest_three_sum_of_two_squares_l1081_108194

/-- A function that returns the number of ways a positive integer can be expressed as the sum of two squares. -/
def countSumOfTwoSquares (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a positive integer can be expressed as the sum of two squares in exactly three distinct ways. -/
def hasExactlyThreeSumOfTwoSquares (n : ℕ) : Prop :=
  countSumOfTwoSquares n = 3

/-- Theorem stating that 325 is the smallest positive integer that can be expressed as the sum of two squares in exactly three distinct ways. -/
theorem smallest_three_sum_of_two_squares :
  hasExactlyThreeSumOfTwoSquares 325 ∧
  ∀ m : ℕ, m < 325 → ¬hasExactlyThreeSumOfTwoSquares m :=
sorry

end NUMINAMATH_CALUDE_smallest_three_sum_of_two_squares_l1081_108194


namespace NUMINAMATH_CALUDE_f_value_at_four_l1081_108156

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^a else |x - 2|

theorem f_value_at_four (a : ℝ) :
  (f a (-2) = f a 2) → f a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_four_l1081_108156


namespace NUMINAMATH_CALUDE_ratio_comparison_correct_l1081_108158

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_water_ratio : ℚ := 1 / 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 60

/-- The ratio of (flavoring to corn syrup in sport formulation) to (flavoring to corn syrup in standard formulation) -/
def ratio_comparison : ℚ := 3

/-- Theorem stating that the ratio comparison is correct given the problem conditions -/
theorem ratio_comparison_correct : 
  let standard_flavoring_to_corn := standard_ratio 0 / standard_ratio 1
  let sport_flavoring := sport_water * (sport_water_ratio * (standard_ratio 0 / standard_ratio 2))
  let sport_flavoring_to_corn := sport_flavoring / sport_corn_syrup
  (sport_flavoring_to_corn / standard_flavoring_to_corn) = ratio_comparison := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_correct_l1081_108158


namespace NUMINAMATH_CALUDE_fathers_age_l1081_108109

/-- 
Given:
- A man's current age is (2/5) of his father's age
- After 8 years, the man's age will be (1/2) of his father's age

Prove that the father's current age is 40 years.
-/
theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 : ℕ) * father_age / (5 : ℕ) →
  man_age + 8 = (father_age + 8) / (2 : ℕ) →
  father_age = 40 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1081_108109


namespace NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l1081_108119

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 48 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_pairs : (∀ m : ℕ, m > 0 ∧ m < 48 → g m ≠ 3) ∧ g 48 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l1081_108119


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1081_108118

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem isosceles_triangle (t : Triangle) 
  (h1 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h2 : t.A + t.B + t.C = Real.pi)
  (h3 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h4 : t.b / t.a = (1 - Real.cos t.B) / Real.cos t.A) :
  t.A = t.C :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1081_108118


namespace NUMINAMATH_CALUDE_total_triangles_is_28_l1081_108142

/-- Represents a triangular arrangement of equilateral triangles -/
structure TriangularArrangement where
  rows : ℕ
  -- Each row n contains n unit triangles
  unit_triangles_in_row : (n : ℕ) → n ≤ rows → ℕ
  unit_triangles_in_row_eq : ∀ n h, unit_triangles_in_row n h = n

/-- Counts the total number of equilateral triangles in the arrangement -/
def count_all_triangles (arrangement : TriangularArrangement) : ℕ :=
  sorry

/-- The main theorem: In a triangular arrangement with 6 rows, 
    the total number of equilateral triangles is 28 -/
theorem total_triangles_is_28 :
  ∀ (arrangement : TriangularArrangement),
  arrangement.rows = 6 →
  count_all_triangles arrangement = 28 :=
sorry

end NUMINAMATH_CALUDE_total_triangles_is_28_l1081_108142


namespace NUMINAMATH_CALUDE_purely_imaginary_z_implies_x_equals_one_l1081_108162

-- Define the complex number z as a function of x
def z (x : ℝ) : ℂ := (x^2 - 1 : ℝ) + (x + 1 : ℝ) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- Theorem statement
theorem purely_imaginary_z_implies_x_equals_one :
  ∀ x : ℝ, isPurelyImaginary (z x) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_implies_x_equals_one_l1081_108162


namespace NUMINAMATH_CALUDE_handbag_discount_proof_l1081_108123

theorem handbag_discount_proof (initial_price : ℝ) (regular_discount : ℝ) (monday_discount : ℝ) :
  initial_price = 250 →
  regular_discount = 0.4 →
  monday_discount = 0.1 →
  let price_after_regular_discount := initial_price * (1 - regular_discount)
  let final_price := price_after_regular_discount * (1 - monday_discount)
  final_price = 135 :=
by sorry

end NUMINAMATH_CALUDE_handbag_discount_proof_l1081_108123


namespace NUMINAMATH_CALUDE_cube_root_of_square_64_l1081_108148

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  x^(1/3) = 2 ∨ x^(1/3) = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_64_l1081_108148


namespace NUMINAMATH_CALUDE_sum_equals_220_l1081_108141

theorem sum_equals_220 : 145 + 33 + 29 + 13 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_220_l1081_108141


namespace NUMINAMATH_CALUDE_integer_quadruple_solution_l1081_108155

theorem integer_quadruple_solution :
  ∃! (S : Set (ℕ × ℕ × ℕ × ℕ)),
    S.Nonempty ∧
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔
      (1 < a ∧ a < b ∧ b < c ∧ c < d) ∧
      (∃ k : ℕ, a * b * c * d - 1 = k * ((a - 1) * (b - 1) * (c - 1) * (d - 1)))) ∧
    S = {(3, 5, 17, 255), (2, 4, 10, 80)} :=
by sorry

end NUMINAMATH_CALUDE_integer_quadruple_solution_l1081_108155


namespace NUMINAMATH_CALUDE_unicity_of_inverse_l1081_108111

variable {G : Type*} [Group G]

theorem unicity_of_inverse (x y z : G) (h1 : 1 = x * y) (h2 : 1 = z * x) :
  y = z ∧ y = x⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_unicity_of_inverse_l1081_108111


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1081_108185

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + x - 4 = 0 -/
def f (x : ℝ) : ℝ := x^2 + x - 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1081_108185


namespace NUMINAMATH_CALUDE_odd_function_theorem_l1081_108114

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) : 
  ∀ x, f x = x * (|x| - 2) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_theorem_l1081_108114


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l1081_108179

theorem discount_percentage_calculation (cost_price marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 95 →
  marked_price = 125 →
  profit_percentage = 25 →
  ∃ (discount_percentage : ℝ),
    discount_percentage = 5 ∧
    marked_price * (1 - discount_percentage / 100) = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l1081_108179


namespace NUMINAMATH_CALUDE_robin_ate_twelve_cupcakes_l1081_108120

/-- The number of cupcakes Robin ate with chocolate sauce -/
def chocolate_cupcakes : ℕ := 4

/-- The number of cupcakes Robin ate with buttercream frosting -/
def buttercream_cupcakes : ℕ := 2 * chocolate_cupcakes

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := chocolate_cupcakes + buttercream_cupcakes

theorem robin_ate_twelve_cupcakes : total_cupcakes = 12 := by
  sorry

end NUMINAMATH_CALUDE_robin_ate_twelve_cupcakes_l1081_108120


namespace NUMINAMATH_CALUDE_words_lost_proof_l1081_108195

/-- The number of letters in the language --/
def num_letters : ℕ := 69

/-- The index of the forbidden letter --/
def forbidden_letter_index : ℕ := 7

/-- The number of words lost due to prohibition --/
def words_lost : ℕ := 139

/-- Theorem stating the number of words lost due to prohibition --/
theorem words_lost_proof :
  (num_letters : ℕ) = 69 →
  (forbidden_letter_index : ℕ) = 7 →
  (words_lost : ℕ) = 139 :=
by
  sorry

#check words_lost_proof

end NUMINAMATH_CALUDE_words_lost_proof_l1081_108195


namespace NUMINAMATH_CALUDE_total_scoops_needed_l1081_108108

/-- Calculates the total number of scoops needed for baking ingredients --/
theorem total_scoops_needed
  (flour_cups : ℚ)
  (sugar_cups : ℚ)
  (milk_cups : ℚ)
  (flour_scoop : ℚ)
  (sugar_scoop : ℚ)
  (milk_scoop : ℚ)
  (h_flour : flour_cups = 4)
  (h_sugar : sugar_cups = 3)
  (h_milk : milk_cups = 2)
  (h_flour_scoop : flour_scoop = 1/4)
  (h_sugar_scoop : sugar_scoop = 1/3)
  (h_milk_scoop : milk_scoop = 1/2) :
  ⌈flour_cups / flour_scoop⌉ + ⌈sugar_cups / sugar_scoop⌉ + ⌈milk_cups / milk_scoop⌉ = 29 :=
by sorry

end NUMINAMATH_CALUDE_total_scoops_needed_l1081_108108
