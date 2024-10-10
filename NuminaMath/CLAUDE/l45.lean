import Mathlib

namespace polygon_area_bounds_l45_4512

-- Define the type for polygons
structure Polygon :=
  (vertices : List (Int × Int))
  (convex : Bool)
  (area : ℝ)

-- Define the theorem
theorem polygon_area_bounds :
  ∃ (a b c : ℝ) (α : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧
    (∀ n : ℕ, ∃ P : Polygon,
      P.convex = true ∧
      P.vertices.length = n ∧
      P.area < a * (n : ℝ)^3) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ b * (n : ℝ)^2) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ c * (n : ℝ)^(2 + α)) :=
sorry

end polygon_area_bounds_l45_4512


namespace triangle_condition_l45_4569

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_condition (k : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 3 → 
    f k a + f k b > f k c ∧ 
    f k b + f k c > f k a ∧ 
    f k c + f k a > f k b) ↔ 
  k > 2 :=
sorry

end triangle_condition_l45_4569


namespace missing_digit_is_three_l45_4502

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def six_digit_number (a b c d e f : ℕ) : ℕ := a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f

theorem missing_digit_is_three :
  ∃ (x : ℕ), x < 10 ∧ is_divisible_by_9 (six_digit_number 3 4 6 x 9 2) ∧ x = 3 := by
  sorry

end missing_digit_is_three_l45_4502


namespace range_of_function_l45_4507

theorem range_of_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ π/2) :
  ∃ y, -Real.sqrt 3 ≤ y ∧ y ≤ 2 ∧ y = Real.sqrt 3 * Real.sin x + Real.cos x :=
by sorry

end range_of_function_l45_4507


namespace remaining_amount_after_ten_months_l45_4593

/-- Represents a loan scenario where a person borrows money and pays it back in monthly installments. -/
structure LoanScenario where
  /-- The total amount borrowed -/
  borrowed_amount : ℝ
  /-- The fixed amount paid back each month -/
  monthly_payment : ℝ
  /-- Assumption that the borrowed amount is positive -/
  borrowed_positive : borrowed_amount > 0
  /-- Assumption that the monthly payment is positive -/
  payment_positive : monthly_payment > 0
  /-- After 6 months, half of the borrowed amount has been paid back -/
  half_paid_after_six_months : 6 * monthly_payment = borrowed_amount / 2

/-- Theorem stating that the remaining amount owed after 10 months is equal to
    the borrowed amount minus 10 times the monthly payment. -/
theorem remaining_amount_after_ten_months (scenario : LoanScenario) :
  scenario.borrowed_amount - 10 * scenario.monthly_payment =
  scenario.borrowed_amount - (6 * scenario.monthly_payment + 4 * scenario.monthly_payment) :=
by sorry

end remaining_amount_after_ten_months_l45_4593


namespace square_roots_values_l45_4588

theorem square_roots_values (m : ℝ) (a : ℝ) (h1 : a > 0) 
  (h2 : (3 * m - 1)^2 = a) (h3 : (-2 * m - 2)^2 = a) :
  a = 64 ∨ a = 64/25 := by
  sorry

end square_roots_values_l45_4588


namespace book_arrangement_ways_l45_4582

theorem book_arrangement_ways (n m : ℕ) (h : n + m = 9) (hn : n = 4) (hm : m = 5) :
  Nat.choose (n + m) n = 126 := by
  sorry

end book_arrangement_ways_l45_4582


namespace triangle_theorem_l45_4594

/-- Given a triangle ABC with sides a, b, c, inradius r, and exradii r₁, r₂, r₃ opposite vertices A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ

/-- Conditions for the triangle -/
def ValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a > t.r₁ ∧ t.b > t.r₂ ∧ t.c > t.r₃

/-- Definition of an acute triangle -/
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧ t.b^2 + t.c^2 > t.a^2 ∧ t.c^2 + t.a^2 > t.b^2

/-- The main theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : ValidTriangle t) :
  IsAcute t ∧ t.a + t.b + t.c > t.r + t.r₁ + t.r₂ + t.r₃ := by
  sorry

end triangle_theorem_l45_4594


namespace tunnel_length_l45_4553

/-- Calculates the length of a tunnel given the train's length, speed, and time to pass through. -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_seconds : ℝ) :
  train_length = 300 →
  train_speed_kmh = 54 →
  time_seconds = 100 →
  (train_speed_kmh * 1000 / 3600 * time_seconds) - train_length = 1200 := by
  sorry

#check tunnel_length

end tunnel_length_l45_4553


namespace attainable_tables_count_l45_4513

/-- Represents a table with signs -/
def Table (m n : ℕ) := Fin (2*m) → Fin (2*n) → Bool

/-- Determines if a table is attainable after one transformation -/
def IsAttainable (m n : ℕ) (t : Table m n) : Prop := sorry

/-- Counts the number of attainable tables -/
def CountAttainableTables (m n : ℕ) : ℕ := sorry

theorem attainable_tables_count (m n : ℕ) :
  CountAttainableTables m n = if m % 2 = 1 ∧ n % 2 = 1 then 2^(m+n-2) else 2^(m+n-1) := by sorry

end attainable_tables_count_l45_4513


namespace special_polynomial_property_l45_4591

/-- The polynomial type representing (1-z)^b₁ · (1-z²)^b₂ · (1-z³)^b₃ ··· (1-z³²)^b₃₂ -/
def SpecialPolynomial (b : Fin 32 → ℕ+) : Polynomial ℚ := sorry

/-- The property that after multiplying out and removing terms with degree > 32, 
    the polynomial equals 1 - 2z -/
def HasSpecialProperty (p : Polynomial ℚ) : Prop := sorry

theorem special_polynomial_property (b : Fin 32 → ℕ+) :
  HasSpecialProperty (SpecialPolynomial b) → b 31 = 2^27 - 2^11 := by sorry

end special_polynomial_property_l45_4591


namespace power_division_equality_l45_4548

theorem power_division_equality (a : ℝ) : a^11 / a^2 = a^9 := by
  sorry

end power_division_equality_l45_4548


namespace number_problem_l45_4530

theorem number_problem (x : ℝ) : (258/100 * x) / 6 = 543.95 → x = 1265 := by
  sorry

end number_problem_l45_4530


namespace tetrahedron_volume_relation_l45_4527

/-- A tetrahedron with volume V, face areas S_i, and distances H_i from an internal point to each face. -/
structure Tetrahedron where
  V : ℝ
  S : Fin 4 → ℝ
  H : Fin 4 → ℝ
  K : ℝ
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  H_positive : ∀ i, H i > 0
  K_positive : K > 0
  h_relation : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = K

theorem tetrahedron_volume_relation (t : Tetrahedron) :
  t.H 0 + 2 * t.H 1 + 3 * t.H 2 + 4 * t.H 3 = 3 * t.V / t.K := by
  sorry

end tetrahedron_volume_relation_l45_4527


namespace a_plus_b_value_l45_4511

theorem a_plus_b_value (a b : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 3) 
  (hab : |a-b| = -(a-b)) : 
  a + b = 5 ∨ a + b = 1 := by
sorry

end a_plus_b_value_l45_4511


namespace smallest_triangle_perimeter_l45_4598

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

end smallest_triangle_perimeter_l45_4598


namespace calculation_proof_l45_4525

theorem calculation_proof :
  (1 * (-8) - (-6) + (-3) = -5) ∧
  (5 / 13 - 3.7 + 8 / 13 + 1.7 = -1) := by
sorry

end calculation_proof_l45_4525


namespace sin_equality_necessary_not_sufficient_l45_4583

theorem sin_equality_necessary_not_sufficient :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) :=
by sorry

end sin_equality_necessary_not_sufficient_l45_4583


namespace intersection_point_circle_tangent_to_l₃_l45_4554

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

-- Define the intersection point C
def C : ℝ × ℝ := (-2, 4)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 4)^2 = 9

-- Theorem 1: Prove that C is the intersection of l₁ and l₂
theorem intersection_point : l₁ C.1 C.2 ∧ l₂ C.1 C.2 := by sorry

-- Theorem 2: Prove that the circle equation represents a circle with center C and tangent to l₃
theorem circle_tangent_to_l₃ : 
  ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
  (∃ (x y : ℝ), l₃ x y ∧ circle_equation x y) ∧
  (∀ (x y : ℝ), l₃ x y → (x - C.1)^2 + (y - C.2)^2 ≥ r^2) := by sorry

end intersection_point_circle_tangent_to_l₃_l45_4554


namespace impossible_all_black_l45_4575

/-- Represents a cell on the chessboard -/
structure Cell where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black

/-- Represents the chessboard -/
def Chessboard := Cell → Color

/-- Represents a valid inversion operation -/
inductive InversionOperation
  | Horizontal : Fin 8 → Fin 6 → InversionOperation
  | Vertical : Fin 6 → Fin 8 → InversionOperation

/-- Applies an inversion operation to the chessboard -/
def applyInversion (board : Chessboard) (op : InversionOperation) : Chessboard :=
  sorry

/-- Checks if the entire chessboard is black -/
def isAllBlack (board : Chessboard) : Prop :=
  ∀ cell, board cell = Color.Black

/-- Initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ => Color.White

/-- Theorem stating the impossibility of making the entire chessboard black -/
theorem impossible_all_black :
  ¬ ∃ (operations : List InversionOperation),
    isAllBlack (operations.foldl applyInversion initialBoard) :=
  sorry

end impossible_all_black_l45_4575


namespace books_per_shelf_l45_4579

def library1_total : ℕ := 24850
def library2_total : ℕ := 55300
def library1_leftover : ℕ := 154
def library2_leftover : ℕ := 175

theorem books_per_shelf :
  Int.gcd (library1_total - library1_leftover) (library2_total - library2_leftover) = 441 :=
by sorry

end books_per_shelf_l45_4579


namespace badminton_partitions_l45_4581

def number_of_partitions (n : ℕ) : ℕ := (n.choose 2) * ((n - 2).choose 2) / 2

theorem badminton_partitions :
  number_of_partitions 6 = 45 := by
  sorry

end badminton_partitions_l45_4581


namespace out_of_pocket_calculation_l45_4577

def out_of_pocket (initial_purchase : ℝ) (tv_return : ℝ) (bike_return : ℝ) (toaster_purchase : ℝ) : ℝ :=
  let total_return := tv_return + bike_return
  let sold_bike_cost := bike_return * 1.2
  let sold_bike_price := sold_bike_cost * 0.8
  initial_purchase - total_return - sold_bike_price + toaster_purchase

theorem out_of_pocket_calculation :
  out_of_pocket 3000 700 500 100 = 1420 := by
  sorry

end out_of_pocket_calculation_l45_4577


namespace equation_solution_l45_4599

theorem equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end equation_solution_l45_4599


namespace find_M_l45_4538

theorem find_M : ∃ M : ℕ, (1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M) → M = 75 := by
  sorry

end find_M_l45_4538


namespace smallest_n_is_five_l45_4526

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ n + 1 ∧ ¬is_divisible (n^2 - n + 1) m

theorem smallest_n_is_five :
  satisfies_condition 5 ∧
  ∀ n : ℕ, 0 < n ∧ n < 5 → ¬satisfies_condition n :=
sorry

end smallest_n_is_five_l45_4526


namespace binomial_coefficient_seven_three_l45_4585

theorem binomial_coefficient_seven_three : 
  Nat.choose 7 3 = 35 := by sorry

end binomial_coefficient_seven_three_l45_4585


namespace age_sum_l45_4549

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 18 years old
  Prove that the sum of their ages is 47 years. -/
theorem age_sum (a b c : ℕ) : 
  b = 18 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 47 := by sorry

end age_sum_l45_4549


namespace white_tree_count_l45_4536

/-- Represents the number of crepe myrtle trees of each color in the park -/
structure TreeCount where
  total : ℕ
  pink : ℕ
  red : ℕ
  white : ℕ

/-- The conditions of the park's tree distribution -/
def park_conditions (t : TreeCount) : Prop :=
  t.total = 42 ∧
  t.pink = t.total / 3 ∧
  t.red = 2 ∧
  t.white = t.total - t.pink - t.red ∧
  t.white > t.pink ∧ t.white > t.red

/-- Theorem stating that under the given conditions, the number of white trees is 26 -/
theorem white_tree_count (t : TreeCount) (h : park_conditions t) : t.white = 26 := by
  sorry

end white_tree_count_l45_4536


namespace quadratic_form_b_l45_4524

/-- Given a quadratic of the form x^2 + bx + 54 where b is positive,
    if it can be rewritten as (x+m)^2 + 18, then b = 12 -/
theorem quadratic_form_b (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 54 = (x+m)^2 + 18) → 
  b = 12 := by
sorry

end quadratic_form_b_l45_4524


namespace tangent_slope_at_x_4_l45_4501

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 8

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 5

-- Theorem statement
theorem tangent_slope_at_x_4 :
  f' 4 = 29 := by sorry

end tangent_slope_at_x_4_l45_4501


namespace angle_WYZ_measure_l45_4561

-- Define the angle measures
def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 30 := by sorry

end angle_WYZ_measure_l45_4561


namespace smallest_possible_a_l45_4509

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ) (h_a_pos : a > 0) 
  (h_poly : ∀ x : ℤ, ∃ k : ℤ, P x = k)
  (h_odd : P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a)
  (h_even : P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a) :
  945 ≤ a ∧ ∃ Q : ℤ → ℤ, 
    (∀ x : ℤ, ∃ k : ℤ, Q x = k) ∧
    (Q 2 = 126 ∧ Q 4 = -210 ∧ Q 6 = 126 ∧ Q 8 = -18 ∧ Q 10 = 126) ∧
    (∀ x : ℤ, P x - a = (x-1)*(x-3)*(x-5)*(x-7)*(Q x)) := by
  sorry

end smallest_possible_a_l45_4509


namespace absolute_difference_of_mn_l45_4532

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 2) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 7 := by
sorry

end absolute_difference_of_mn_l45_4532


namespace pencils_calculation_l45_4510

/-- Given a setup of pencils and crayons in rows, calculates the number of pencils per row. -/
def pencils_per_row (total_items : ℕ) (rows : ℕ) (crayons_per_row : ℕ) : ℕ :=
  (total_items - rows * crayons_per_row) / rows

theorem pencils_calculation :
  pencils_per_row 638 11 27 = 31 := by
  sorry

end pencils_calculation_l45_4510


namespace angle_measure_problem_l45_4551

theorem angle_measure_problem (C D : ℝ) 
  (h1 : C + D = 360)
  (h2 : C = 5 * D) : 
  C = 300 := by
sorry

end angle_measure_problem_l45_4551


namespace gcd_228_1995_l45_4533

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l45_4533


namespace flower_bouquet_carnations_percentage_l45_4521

theorem flower_bouquet_carnations_percentage 
  (total_flowers : ℕ) 
  (pink_flowers red_flowers pink_roses red_roses pink_carnations red_carnations : ℕ) :
  (pink_flowers = total_flowers / 2) →
  (red_flowers = total_flowers / 2) →
  (pink_roses = pink_flowers * 2 / 5) →
  (red_carnations = red_flowers * 2 / 3) →
  (pink_carnations = pink_flowers - pink_roses) →
  (red_roses = red_flowers - red_carnations) →
  (((pink_carnations + red_carnations : ℚ) / total_flowers) * 100 = 63) := by
  sorry

end flower_bouquet_carnations_percentage_l45_4521


namespace box_volume_correct_l45_4571

/-- The volume of an open box formed from a rectangular sheet -/
def boxVolume (x : ℝ) : ℝ := 4 * x^3 - 56 * x^2 + 192 * x

/-- The properties of the box construction -/
structure BoxProperties where
  sheet_length : ℝ
  sheet_width : ℝ
  corner_cut : ℝ
  max_height : ℝ
  h_length : sheet_length = 16
  h_width : sheet_width = 12
  h_max_height : max_height = 6
  h_corner_cut_range : 0 < corner_cut ∧ corner_cut ≤ max_height

/-- Theorem stating that the boxVolume function correctly calculates the volume of the box -/
theorem box_volume_correct (props : BoxProperties) (x : ℝ) 
    (h_x : 0 < x ∧ x ≤ props.max_height) : 
  boxVolume x = (props.sheet_length - 2*x) * (props.sheet_width - 2*x) * x := by
  sorry

#check box_volume_correct

end box_volume_correct_l45_4571


namespace difference_in_half_dollars_l45_4534

/-- The number of quarters Alice has -/
def alice_quarters (p : ℚ) : ℚ := 8 * p + 2

/-- The number of quarters Bob has -/
def bob_quarters (p : ℚ) : ℚ := 3 * p + 6

/-- Conversion factor from quarters to half-dollars -/
def quarter_to_half_dollar : ℚ := 1 / 2

theorem difference_in_half_dollars (p : ℚ) :
  (alice_quarters p - bob_quarters p) * quarter_to_half_dollar = 2.5 * p - 2 := by
  sorry

end difference_in_half_dollars_l45_4534


namespace max_triangle_chain_length_l45_4528

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into
  total_triangles : ℕ  -- total number of smaller triangles

/-- Represents a chain of triangles within a divided triangle -/
structure TriangleChain (dt : DividedTriangle) where
  length : ℕ  -- number of triangles in the chain

/-- The property that the total number of smaller triangles is n^2 -/
def total_triangles_prop (dt : DividedTriangle) : Prop :=
  dt.total_triangles = dt.n^2

/-- The theorem stating the maximum length of a triangle chain -/
theorem max_triangle_chain_length (dt : DividedTriangle) 
  (h : total_triangles_prop dt) : 
  ∃ (chain : TriangleChain dt), 
    ∀ (other_chain : TriangleChain dt), other_chain.length ≤ chain.length ∧ 
    chain.length = dt.n^2 - dt.n + 1 :=
sorry

end max_triangle_chain_length_l45_4528


namespace largest_n_l45_4559

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating the largest possible value of n -/
theorem largest_n : ∃ (x y : ℕ),
  x < 10 ∧ 
  y < 10 ∧ 
  x ≠ y ∧
  isPrime x ∧ 
  isPrime y ∧ 
  isPrime (10 * y + x) ∧
  1000 ≤ x * y * (10 * y + x) ∧ 
  x * y * (10 * y + x) < 10000 ∧
  ∀ (a b : ℕ), 
    a < 10 → 
    b < 10 → 
    a ≠ b →
    isPrime a → 
    isPrime b → 
    isPrime (10 * b + a) →
    1000 ≤ a * b * (10 * b + a) →
    a * b * (10 * b + a) < 10000 →
    a * b * (10 * b + a) ≤ x * y * (10 * y + x) :=
by sorry

end largest_n_l45_4559


namespace max_new_lines_theorem_l45_4586

/-- The maximum number of new lines formed by connecting intersection points 
    of n lines in a plane, where any two lines intersect and no three lines 
    pass through the same point. -/
def max_new_lines (n : ℕ) : ℚ :=
  (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem stating the maximum number of new lines formed by connecting 
    intersection points of n lines in a plane, where any two lines intersect 
    and no three lines pass through the same point. -/
theorem max_new_lines_theorem (n : ℕ) (h : n ≥ 3) :
  let original_lines := n
  let any_two_intersect := true
  let no_three_at_same_point := true
  max_new_lines n = (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3) :=
by sorry

end max_new_lines_theorem_l45_4586


namespace andrew_kept_130_stickers_l45_4576

def andrew_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def fred_stickers : ℕ := daniel_stickers + fred_extra_stickers
def shared_stickers : ℕ := daniel_stickers + fred_stickers
def andrew_kept_stickers : ℕ := andrew_stickers - shared_stickers

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end andrew_kept_130_stickers_l45_4576


namespace perfect_square_binomial_l45_4562

theorem perfect_square_binomial : ∃ (r s : ℝ), (r * x + s)^2 = 4 * x^2 + 20 * x + 25 := by sorry

end perfect_square_binomial_l45_4562


namespace equality_from_sum_of_squares_l45_4572

theorem equality_from_sum_of_squares (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c := by
  sorry

end equality_from_sum_of_squares_l45_4572


namespace equation_represents_point_l45_4508

theorem equation_represents_point (x y a b : ℝ) : 
  (x - a)^2 + (y + b)^2 = 0 ↔ x = a ∧ y = -b := by
sorry

end equation_represents_point_l45_4508


namespace twentieth_number_in_base6_l45_4592

-- Define a function to convert decimal to base 6
def decimalToBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem twentieth_number_in_base6 :
  decimalToBase6 20 = [3, 2] :=
sorry

end twentieth_number_in_base6_l45_4592


namespace sphere_volume_increase_l45_4596

theorem sphere_volume_increase (r₁ r₂ V₁ V₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) 
  (h₃ : V₁ = (4/3) * π * r₁^3) (h₄ : V₂ = (4/3) * π * r₂^3) : V₂ = 8 * V₁ := by
  sorry

end sphere_volume_increase_l45_4596


namespace x_plus_y_power_2023_l45_4563

theorem x_plus_y_power_2023 (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : 
  (x + y)^2023 = -1 := by
sorry

end x_plus_y_power_2023_l45_4563


namespace sharon_in_middle_l45_4573

-- Define the people
inductive Person : Type
| Aaron : Person
| Darren : Person
| Karen : Person
| Maren : Person
| Sharon : Person

-- Define the positions in the train
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position
| Fifth : Position

def is_behind (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.Second, Position.Third => True
  | Position.Third, Position.Fourth => True
  | Position.Fourth, Position.Fifth => True
  | _, _ => False

def is_in_front (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.First, Position.Second => True
  | Position.First, Position.Third => True
  | Position.First, Position.Fourth => True
  | Position.Second, Position.Third => True
  | Position.Second, Position.Fourth => True
  | Position.Third, Position.Fourth => True
  | _, _ => False

def at_least_one_between (p1 p2 p3 : Position) : Prop :=
  match p1, p2, p3 with
  | Position.First, Position.Third, Position.Fifth => True
  | Position.First, Position.Fourth, Position.Fifth => True
  | Position.First, Position.Third, Position.Fourth => True
  | Position.Second, Position.Fourth, Position.Fifth => True
  | _, _, _ => False

-- Define the seating arrangement
def seating_arrangement (seat : Person → Position) : Prop :=
  (seat Person.Maren = Position.Fifth) ∧
  (∃ p : Position, is_behind (seat Person.Aaron) p ∧ seat Person.Sharon = p) ∧
  (∃ p : Position, is_in_front (seat Person.Darren) (seat Person.Aaron)) ∧
  (at_least_one_between (seat Person.Karen) (seat Person.Darren) (seat Person.Karen) ∨
   at_least_one_between (seat Person.Darren) (seat Person.Karen) (seat Person.Darren))

theorem sharon_in_middle (seat : Person → Position) :
  seating_arrangement seat → seat Person.Sharon = Position.Third :=
sorry

end sharon_in_middle_l45_4573


namespace original_price_of_tv_l45_4555

/-- The original price of a television given a discount and total paid amount -/
theorem original_price_of_tv (discount_rate : ℚ) (total_paid : ℚ) : 
  discount_rate = 5 / 100 → 
  total_paid = 456 → 
  (1 - discount_rate) * 480 = total_paid :=
by sorry

end original_price_of_tv_l45_4555


namespace convention_handshakes_eq_990_l45_4514

/-- The number of handshakes at the Annual Mischief Convention -/
def convention_handshakes : ℕ :=
  let total_gremlins : ℕ := 30
  let total_imps : ℕ := 20
  let unfriendly_gremlins : ℕ := 10
  let friendly_gremlins : ℕ := total_gremlins - unfriendly_gremlins

  let gremlin_handshakes : ℕ := 
    (friendly_gremlins * (friendly_gremlins - 1)) / 2 + 
    unfriendly_gremlins * friendly_gremlins

  let imp_gremlin_handshakes : ℕ := total_imps * total_gremlins

  gremlin_handshakes + imp_gremlin_handshakes

theorem convention_handshakes_eq_990 : convention_handshakes = 990 := by
  sorry

end convention_handshakes_eq_990_l45_4514


namespace consecutive_even_numbers_sum_l45_4564

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    a + b + c + d = 52) → 
  n + 4 = 14 :=
by sorry

end consecutive_even_numbers_sum_l45_4564


namespace pythagoras_academy_olympiad_students_l45_4531

/-- The number of distinct students taking the Math Olympiad at Pythagoras Academy -/
def distinctStudents (eulerStudents gaussStudents fibonacciStudents doubleCountedStudents : ℕ) : ℕ :=
  eulerStudents + gaussStudents + fibonacciStudents - doubleCountedStudents

/-- Theorem stating the number of distinct students taking the Math Olympiad -/
theorem pythagoras_academy_olympiad_students :
  distinctStudents 15 10 12 3 = 34 := by
  sorry

end pythagoras_academy_olympiad_students_l45_4531


namespace symmetric_x_axis_coords_symmetric_y_axis_coords_l45_4590

/-- Given a point M with coordinates (x, y) in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to X-axis -/
def symmetricXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Definition of symmetry with respect to Y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the X-axis are (x, -y) -/
theorem symmetric_x_axis_coords (M : Point2D) :
  symmetricXAxis M = { x := M.x, y := -M.y } := by sorry

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the Y-axis are (-x, y) -/
theorem symmetric_y_axis_coords (M : Point2D) :
  symmetricYAxis M = { x := -M.x, y := M.y } := by sorry

end symmetric_x_axis_coords_symmetric_y_axis_coords_l45_4590


namespace divisors_of_2_pow_48_minus_1_l45_4580

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < b ∧ b < 70 ∧ (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 :=
by
  sorry

end divisors_of_2_pow_48_minus_1_l45_4580


namespace isabellas_hair_length_l45_4506

theorem isabellas_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 9)
  (h2 : cut_length = 9) :
  current_length + cut_length = 18 := by
  sorry

end isabellas_hair_length_l45_4506


namespace distinct_digits_base_eight_l45_4542

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are 250 three-digit numbers with distinct digits in base 8 -/
theorem distinct_digits_base_eight :
  distinctDigitNumbers 8 = 250 := by
  sorry

end distinct_digits_base_eight_l45_4542


namespace f_at_neg_one_equals_two_l45_4565

-- Define the function f(x) = -2x
def f (x : ℝ) : ℝ := -2 * x

-- Theorem stating that f(-1) = 2
theorem f_at_neg_one_equals_two : f (-1) = 2 := by
  sorry

end f_at_neg_one_equals_two_l45_4565


namespace find_m_value_l45_4543

/-- Given x and y values, prove that m = 3 when y is linearly related to x with equation y = 1.3x + 0.8 -/
theorem find_m_value (x : Fin 5 → ℝ) (y : Fin 5 → ℝ) (m : ℝ) : 
  x 0 = 1 ∧ x 1 = 3 ∧ x 2 = 4 ∧ x 3 = 5 ∧ x 4 = 7 ∧
  y 0 = 1 ∧ y 1 = m ∧ y 2 = 2*m+1 ∧ y 3 = 2*m+3 ∧ y 4 = 10 ∧
  (∀ i : Fin 5, y i = 1.3 * x i + 0.8) →
  m = 3 := by
  sorry

end find_m_value_l45_4543


namespace cos_2A_value_l45_4515

theorem cos_2A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 3 * Real.cos A - 8 * Real.tan A = 0) : 
  Real.cos (2 * A) = 7 / 9 := by
sorry

end cos_2A_value_l45_4515


namespace functional_equation_solution_l45_4537

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*(f x) + (f y)^2

/-- The main theorem stating that functions satisfying the equation are either
    the identity function or the identity function plus one. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x, f x = x) ∨ (∀ x, f x = x + 1) :=
sorry

end functional_equation_solution_l45_4537


namespace sum_of_squares_equals_one_l45_4595

theorem sum_of_squares_equals_one 
  (a b c p q r : ℝ) 
  (h1 : a * b = p) 
  (h2 : b * c = q) 
  (h3 : c * a = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  a^2 + b^2 + c^2 = 1 := by
sorry

end sum_of_squares_equals_one_l45_4595


namespace sum_of_i_powers_l45_4541

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  -- The proof goes here
  sorry

end sum_of_i_powers_l45_4541


namespace quadratic_rewrite_sum_l45_4557

theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (u v : ℝ), (9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 := by
  sorry

end quadratic_rewrite_sum_l45_4557


namespace min_c_value_l45_4568

/-- Given five consecutive positive integers a, b, c, d, e,
    if b + c + d is a perfect square and a + b + c + d + e is a perfect cube,
    then the minimum value of c is 675. -/
theorem min_c_value (a b c d e : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e → 
  ∃ m : ℕ, b + c + d = m^2 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ 675 :=
by sorry

end min_c_value_l45_4568


namespace solution_set_of_inequality_l45_4574

theorem solution_set_of_inequality (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 :=
sorry

end solution_set_of_inequality_l45_4574


namespace H2O_formation_l45_4503

-- Define the chemical reaction
def reaction_ratio : ℚ := 1

-- Define the given amounts of reactants
def KOH_moles : ℚ := 3
def NH4I_moles : ℚ := 3

-- Define the theorem
theorem H2O_formation (h : KOH_moles = NH4I_moles) :
  min KOH_moles NH4I_moles = 3 ∧ 
  reaction_ratio * min KOH_moles NH4I_moles = 3 :=
by sorry

end H2O_formation_l45_4503


namespace grants_score_l45_4584

/-- Given the scores of three students on a math test, prove Grant's score. -/
theorem grants_score (hunter_score john_score grant_score : ℕ) : 
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end grants_score_l45_4584


namespace p_less_than_q_l45_4529

/-- For all real x, if P = (x-2)(x-4) and Q = (x-3)^2, then P < Q. -/
theorem p_less_than_q (x : ℝ) : (x - 2) * (x - 4) < (x - 3)^2 := by
  sorry

end p_less_than_q_l45_4529


namespace age_difference_l45_4578

/-- Given three people A, B, and C, where C is 16 years younger than A,
    prove that the difference between the total age of A and B and
    the total age of B and C is 16 years. -/
theorem age_difference (A B C : ℕ) (h : C = A - 16) :
  (A + B) - (B + C) = 16 := by
  sorry

end age_difference_l45_4578


namespace range_of_m_l45_4570

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 4 = 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∧ ¬(q m)) → (2 ≤ m ∧ m < 3) :=
by sorry

end range_of_m_l45_4570


namespace f_2023_equals_2_l45_4550

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2023_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_interval : ∀ x ∈ Set.Icc 0 1, f x = 2^x) :
  f 2023 = 2 := by
  sorry

end f_2023_equals_2_l45_4550


namespace payment_calculation_l45_4567

theorem payment_calculation (payment_per_room : ℚ) (rooms_cleaned : ℚ) : 
  payment_per_room = 15 / 4 →
  rooms_cleaned = 9 / 5 →
  payment_per_room * rooms_cleaned = 27 / 4 := by
sorry

end payment_calculation_l45_4567


namespace sqrt_7_to_6th_power_l45_4500

theorem sqrt_7_to_6th_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_7_to_6th_power_l45_4500


namespace curve_C_and_perpendicular_lines_l45_4556

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.1^2 = P.2

-- Define the curve C
def curve_C (M : ℝ × ℝ) : Prop := M.1^2 = 4 * M.2

-- Define the relationship between P, D, and M
def point_relationship (P D M : ℝ × ℝ) : Prop :=
  D.1 = 0 ∧ D.2 = P.2 ∧ M.1 = 2 * P.1 ∧ M.2 = P.2

-- Define the line l
def line_l (y : ℝ) : Prop := y = -1

-- Define point F
def point_F : ℝ × ℝ := (0, 1)

-- Define perpendicular lines
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem curve_C_and_perpendicular_lines :
  ∀ (P D M A B A1 B1 : ℝ × ℝ),
    parabola P →
    point_relationship P D M →
    curve_C A ∧ curve_C B →
    line_l A1.2 ∧ line_l B1.2 →
    A1.1 = A.1 ∧ B1.1 = B.1 →
    perpendicular (A1.1 - point_F.1, A1.2 - point_F.2) (B1.1 - point_F.1, B1.2 - point_F.2) :=
by sorry

end curve_C_and_perpendicular_lines_l45_4556


namespace intersection_of_M_and_N_l45_4519

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l45_4519


namespace geometric_progression_solution_l45_4547

theorem geometric_progression_solution :
  ∀ (b₁ q : ℚ),
    b₁ + b₁ * q + b₁ * q^2 = 21 →
    b₁^2 + (b₁ * q)^2 + (b₁ * q^2)^2 = 189 →
    ((b₁ = 12 ∧ q = 1/2) ∨ (b₁ = 3 ∧ q = 2)) :=
by sorry

end geometric_progression_solution_l45_4547


namespace lizard_adoption_rate_l45_4539

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def cat_adoption_rate : ℚ := 1/4
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem lizard_adoption_rate : 
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let cats_adopted := (initial_cats : ℚ) * cat_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_cats := initial_cats - cats_adopted.floor
  let total_before_lizard_adoption := remaining_dogs + remaining_cats + initial_lizards + new_pets
  let lizards_adopted := total_before_lizard_adoption - total_pets_after_month
  lizards_adopted / initial_lizards = 1/5 := by sorry

end lizard_adoption_rate_l45_4539


namespace smallest_prime_triangle_perimeter_l45_4560

/-- A triangle with prime side lengths and prime perimeter -/
structure PrimeTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  ha : Nat.Prime a
  hb : Nat.Prime b
  hc : Nat.Prime c
  hab : a < b
  hbc : b < c
  hmin : 5 ≤ a
  htri1 : a + b > c
  htri2 : a + c > b
  htri3 : b + c > a
  hperi : Nat.Prime (a + b + c)

/-- The theorem stating the smallest perimeter of a PrimeTriangle is 23 -/
theorem smallest_prime_triangle_perimeter :
  ∀ t : PrimeTriangle, 23 ≤ t.a + t.b + t.c ∧
  ∃ t0 : PrimeTriangle, t0.a + t0.b + t0.c = 23 := by
  sorry

end smallest_prime_triangle_perimeter_l45_4560


namespace museum_artifact_distribution_l45_4589

theorem museum_artifact_distribution (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wing : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_ratio : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wing = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_ratio = 4 →
  (total_wings - painting_wings) * 
    ((large_painting_wing + small_painting_wings * paintings_per_small_wing) * artifact_ratio / (total_wings - painting_wings)) = 
  (total_wings - painting_wings) * 20 := by
sorry

end museum_artifact_distribution_l45_4589


namespace equation_solution_l45_4505

theorem equation_solution :
  ∃! r : ℚ, (r + 4) / (r - 3) = (r - 2) / (r + 2) ∧ r = -2/11 := by
  sorry

end equation_solution_l45_4505


namespace max_a_value_l45_4587

-- Define the function f(x) = |x-2| + |x-8|
def f (x : ℝ) : ℝ := |x - 2| + |x - 8|

-- State the theorem
theorem max_a_value : 
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), f x ≥ b) → b ≤ 6) :=
by sorry

end max_a_value_l45_4587


namespace valid_pairs_l45_4504

def is_valid_pair (square : Nat) (B : Nat) : Prop :=
  let num := 532900 + square * 10 + B
  (num % 6 = 0) ∧ 
  (square % 2 = 0) ∧ 
  (square ≤ 9) ∧ 
  (B ≤ 9)

theorem valid_pairs : 
  ∀ square B, is_valid_pair square B ↔ 
    ((square = 0 ∧ B = 3) ∨ 
     (square = 2 ∧ B = 1) ∨ 
     (square = 4 ∧ B = 2) ∨ 
     (square = 6 ∧ B = 0) ∨ 
     (square = 8 ∧ B = 1)) :=
by sorry

end valid_pairs_l45_4504


namespace train_crossing_time_l45_4545

/-- Given a train traveling at 72 kmph that passes a man on a 260-meter platform in 17 seconds,
    the time taken for the train to cross the entire platform is 30 seconds. -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) :
  train_speed_kmph = 72 →
  man_crossing_time = 17 →
  platform_length = 260 →
  (platform_length + train_speed_kmph * 1000 / 3600 * man_crossing_time) / (train_speed_kmph * 1000 / 3600) = 30 := by
  sorry

end train_crossing_time_l45_4545


namespace remainder_theorem_l45_4597

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 160 := by
  sorry

end remainder_theorem_l45_4597


namespace sqrt_12_same_type_as_sqrt_3_l45_4516

def is_same_type (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k * b

theorem sqrt_12_same_type_as_sqrt_3 :
  let options := [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18, Real.sqrt 6]
  ∃ (x : ℝ), x ∈ options ∧ is_same_type x (Real.sqrt 3) ∧
    ∀ (y : ℝ), y ∈ options → y ≠ x → ¬(is_same_type y (Real.sqrt 3)) :=
by
  sorry

end sqrt_12_same_type_as_sqrt_3_l45_4516


namespace unique_divisibility_condition_l45_4518

theorem unique_divisibility_condition : 
  ∃! A : ℕ, A < 10 ∧ 45 % A = 0 ∧ (273100 + A * 10 + 6) % 8 = 0 := by
  sorry

end unique_divisibility_condition_l45_4518


namespace range_of_a_l45_4552

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Set.univ \ B) ∪ A a = A a → a ≥ 2 :=
sorry

end range_of_a_l45_4552


namespace both_sports_fans_l45_4540

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 9

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- The number of students who like basketball or cricket or both -/
def total_fans : ℕ := 11

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := basketball_fans + cricket_fans - total_fans

theorem both_sports_fans : both_fans = 6 := by
  sorry

end both_sports_fans_l45_4540


namespace f_is_even_and_decreasing_l45_4520

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end f_is_even_and_decreasing_l45_4520


namespace number_equation_l45_4535

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by sorry

end number_equation_l45_4535


namespace largest_number_with_sum_16_l45_4523

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_16 :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 16 →
    n ≤ 4432 :=
by sorry

end largest_number_with_sum_16_l45_4523


namespace trajectory_line_passes_fixed_point_l45_4566

/-- The trajectory C is defined by the equation y^2 = 4x -/
def trajectory (x y : ℝ) : Prop := y^2 = 4*x

/-- A point P is on the trajectory if it satisfies the equation -/
def on_trajectory (P : ℝ × ℝ) : Prop :=
  trajectory P.1 P.2

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- A line passing through two points -/
def line_through (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

theorem trajectory_line_passes_fixed_point :
  ∀ A B : ℝ × ℝ,
  A ≠ (0, 0) → B ≠ (0, 0) → A ≠ B →
  on_trajectory A → on_trajectory B →
  dot_product A B = 0 →
  line_through A B (4, 0) := by
  sorry

end trajectory_line_passes_fixed_point_l45_4566


namespace polynomial_determination_l45_4544

/-- Given a polynomial Q(x) = x^4 - 2x^3 + 3x^2 + kx + m, where k and m are constants,
    prove that if Q(0) = 16 and Q(1) = 2, then Q(x) = x^4 - 2x^3 + 3x^2 - 16x + 16 -/
theorem polynomial_determination (k m : ℝ) : 
  let Q := fun (x : ℝ) => x^4 - 2*x^3 + 3*x^2 + k*x + m
  (Q 0 = 16) → (Q 1 = 2) → 
  (∀ x, Q x = x^4 - 2*x^3 + 3*x^2 - 16*x + 16) := by
  sorry

end polynomial_determination_l45_4544


namespace bicycle_trip_speed_l45_4558

/-- The speed of the second part of a bicycle trip satisfies an equation based on given conditions. -/
theorem bicycle_trip_speed (v : ℝ) : v > 0 → 0.7 + 10 / v = 17 / 7.99 := by
  sorry

end bicycle_trip_speed_l45_4558


namespace sock_order_ratio_l45_4522

theorem sock_order_ratio (black_pairs blue_pairs : ℕ) (price_blue : ℝ) :
  black_pairs = 4 →
  (4 * 2 * price_blue + blue_pairs * price_blue) * 1.5 = blue_pairs * 2 * price_blue + 4 * price_blue →
  blue_pairs = 16 :=
by sorry

end sock_order_ratio_l45_4522


namespace jackson_charity_collection_l45_4546

-- Define the working days in a week
def working_days : ℕ := 5

-- Define the amount collected on Monday and Tuesday
def monday_collection : ℕ := 300
def tuesday_collection : ℕ := 40

-- Define the average collection per 4 houses
def avg_collection_per_4_houses : ℕ := 10

-- Define the number of houses visited on each remaining day
def houses_per_day : ℕ := 88

-- Define the goal for the week
def weekly_goal : ℕ := 1000

-- Theorem statement
theorem jackson_charity_collection :
  monday_collection + tuesday_collection +
  (working_days - 2) * (houses_per_day / 4 * avg_collection_per_4_houses) =
  weekly_goal := by sorry

end jackson_charity_collection_l45_4546


namespace absolute_value_problem_l45_4517

theorem absolute_value_problem (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) :
  (a > b → a + b = 7 ∨ a + b = 3) ∧
  (|a + b| = |a| - |b| → (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2)) :=
by sorry

end absolute_value_problem_l45_4517
