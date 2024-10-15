import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_progression_five_digit_terms_l2967_296737

/-- 
Given an arithmetic progression with first term a₁ = -1 and common difference d = 19,
this theorem states that the terms consisting only of the digit 5 are given by the formula:
n = (5 * (10^(171k+1) + 35)) / 171, where k is a non-negative integer
-/
theorem arithmetic_progression_five_digit_terms 
  (k : ℕ) : 
  ∃ (n : ℕ), 
    ((-1 : ℤ) + (n - 1) * 19 = 5 * ((10 ^ (171 * k + 1) - 1) / 9)) ∧ 
    (n = (5 * (10 ^ (171 * k + 1) + 35)) / 171) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_five_digit_terms_l2967_296737


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2967_296714

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (ha : a = 9) (hb : b = 11) (hθ : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos θ ∧ c = Real.sqrt (202 + 99 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2967_296714


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l2967_296781

/-- A tetrahedron represented by four vertices in 3D space -/
structure Tetrahedron where
  v1 : Fin 3 → ℝ
  v2 : Fin 3 → ℝ
  v3 : Fin 3 → ℝ
  v4 : Fin 3 → ℝ

/-- A cube represented by its lower and upper bounds in 3D space -/
structure Cube where
  lower : Fin 3 → ℝ
  upper : Fin 3 → ℝ

/-- Function to calculate the volume of a tetrahedron -/
def volume_tetrahedron (t : Tetrahedron) : ℝ := sorry

/-- Function to calculate the volume of a cube -/
def volume_cube (c : Cube) : ℝ := sorry

/-- Function to check if a tetrahedron is inside a cube -/
def is_inside (t : Tetrahedron) (c : Cube) : Prop := sorry

/-- The main theorem: volume of tetrahedron inside unit cube is at most 1/3 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) (c : Cube) :
  is_inside t c →
  (∀ i, c.lower i = 0 ∧ c.upper i = 1) →
  volume_tetrahedron t ≤ (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l2967_296781


namespace NUMINAMATH_CALUDE_mrs_hilt_bug_count_l2967_296787

/-- The number of bugs Mrs. Hilt saw -/
def num_bugs : ℕ := 3

/-- The number of flowers each bug eats -/
def flowers_per_bug : ℕ := 2

/-- The total number of flowers eaten -/
def total_flowers : ℕ := 6

/-- Theorem: The number of bugs is correct given the conditions -/
theorem mrs_hilt_bug_count : 
  num_bugs * flowers_per_bug = total_flowers :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_bug_count_l2967_296787


namespace NUMINAMATH_CALUDE_xz_length_l2967_296721

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- ∠X = 90°
  t.X = 90

def has_hypotenuse_10 (t : Triangle) : Prop :=
  -- YZ = 10
  t.Y = 10

def satisfies_trig_relation (t : Triangle) : Prop :=
  -- tan Z = 3 sin Z
  Real.tan t.Z = 3 * Real.sin t.Z

-- Theorem statement
theorem xz_length (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : has_hypotenuse_10 t) 
  (h3 : satisfies_trig_relation t) : 
  -- XZ = 10/3
  t.Z = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_xz_length_l2967_296721


namespace NUMINAMATH_CALUDE_root_product_cubic_l2967_296789

theorem root_product_cubic (p q r : ℂ) : 
  (3 * p^3 - 8 * p^2 + p - 9 = 0) →
  (3 * q^3 - 8 * q^2 + q - 9 = 0) →
  (3 * r^3 - 8 * r^2 + r - 9 = 0) →
  p * q * r = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_product_cubic_l2967_296789


namespace NUMINAMATH_CALUDE_boy_age_multiple_l2967_296702

theorem boy_age_multiple : 
  let present_age : ℕ := 16
  let age_six_years_ago : ℕ := present_age - 6
  let age_four_years_hence : ℕ := present_age + 4
  (age_four_years_hence : ℚ) / (age_six_years_ago : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_boy_age_multiple_l2967_296702


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l2967_296730

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 5}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for A ∪ (ℝ \ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l2967_296730


namespace NUMINAMATH_CALUDE_paper_folding_l2967_296747

theorem paper_folding (paper_area : Real) (folded_point_distance : Real) : 
  paper_area = 18 →
  folded_point_distance = 2 * Real.sqrt 6 →
  ∃ (side_length : Real) (folded_leg : Real),
    side_length ^ 2 = paper_area ∧
    folded_leg ^ 2 = 12 ∧
    folded_point_distance ^ 2 = 2 * folded_leg ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_l2967_296747


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2967_296772

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := !![4/11, 3/11, 0; -1/11, 2/11, 0; 0, 0, 1/3]
  let A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -3, 0; 1, 4, 0; 0, 0, 3]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2967_296772


namespace NUMINAMATH_CALUDE_integer_divisibility_equivalence_l2967_296749

theorem integer_divisibility_equivalence (n : ℤ) : 
  (∃ a b : ℤ, 3 * n - 2 = 5 * a ∧ 2 * n + 1 = 7 * b) ↔ 
  (∃ k : ℤ, n = 35 * k + 24) := by
sorry

end NUMINAMATH_CALUDE_integer_divisibility_equivalence_l2967_296749


namespace NUMINAMATH_CALUDE_distance_to_reflection_distance_D_to_D_l2967_296768

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_reflection : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_distance_D_to_D_l2967_296768


namespace NUMINAMATH_CALUDE_expression_evaluation_l2967_296793

theorem expression_evaluation : ((69 + 7 * 8) / 3) * 12 = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2967_296793


namespace NUMINAMATH_CALUDE_evaluate_expression_l2967_296700

theorem evaluate_expression : (2 * 4 * 6) * (1/2 + 1/4 + 1/6) = 44 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2967_296700


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l2967_296799

-- Define quadrilaterals
structure Quadrilateral :=
  (is_rhombus : Bool)
  (is_parallelogram : Bool)

-- The given statement (not used in the proof, but included for completeness)
axiom rhombus_implies_parallelogram :
  ∀ q : Quadrilateral, q.is_rhombus → q.is_parallelogram

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ q : Quadrilateral, q.is_parallelogram ∧ ¬q.is_rhombus) ∧
  (∃ q : Quadrilateral, ¬q.is_rhombus ∧ q.is_parallelogram) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l2967_296799


namespace NUMINAMATH_CALUDE_find_missing_score_l2967_296753

def scores : List ℕ := [87, 88, 89, 0, 91, 92, 92, 93, 94]

theorem find_missing_score (x : ℕ) (h : x ∈ scores) :
  (List.sum (List.filter (λ y => y ≠ 87 ∧ y ≠ 94) (List.map (λ y => if y = 0 then x else y) scores))) / 7 = 91 →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_find_missing_score_l2967_296753


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l2967_296713

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves that the fraction of books sold is 2/3 under specific conditions. -/
theorem fraction_of_books_sold (total_books : ℕ) (sold_books : ℕ) : 
  (total_books > 50) →
  (sold_books = total_books - 50) →
  (sold_books * 5 = 500) →
  (sold_books : ℚ) / total_books = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l2967_296713


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2967_296746

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' : ℝ), b' ≠ b ∧ (8 : ℝ) * (2 : ℝ) = 9) →
  (∃ (c' : ℝ), c' ≠ c ∧ (-9 : ℝ) + (-1 : ℝ) = -b') →
  (b = -10 ∧ c = 9) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2967_296746


namespace NUMINAMATH_CALUDE_first_row_seats_theorem_l2967_296727

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  seatIncrement : ℕ
  totalSeats : ℕ

/-- Calculates the number of seats in the first row of the theater. -/
def firstRowSeats (t : Theater) : ℚ :=
  (t.totalSeats / 10 - 76) / 2

/-- Theorem stating the relationship between the total seats and the number of seats in the first row. -/
theorem first_row_seats_theorem (t : Theater) 
    (h1 : t.rows = 20)
    (h2 : t.seatIncrement = 4)
    (h3 : t.totalSeats = 10 * (firstRowSeats t * 2 + 76)) : 
  firstRowSeats t = (t.totalSeats / 10 - 76) / 2 := by
  sorry

end NUMINAMATH_CALUDE_first_row_seats_theorem_l2967_296727


namespace NUMINAMATH_CALUDE_minimum_sales_increase_l2967_296701

theorem minimum_sales_increase (x : ℝ) : 
  let jan_to_may : ℝ := 38.6
  let june : ℝ := 5
  let july : ℝ := june * (1 + x / 100)
  let august : ℝ := july * (1 + x / 100)
  let sep_oct : ℝ := july + august
  let total : ℝ := jan_to_may + june + july + august + sep_oct
  (total ≥ 70 ∧ ∀ y, y < x → (
    let july_y : ℝ := june * (1 + y / 100)
    let august_y : ℝ := july_y * (1 + y / 100)
    let sep_oct_y : ℝ := july_y + august_y
    let total_y : ℝ := jan_to_may + june + july_y + august_y + sep_oct_y
    total_y < 70
  )) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_minimum_sales_increase_l2967_296701


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2967_296761

theorem unknown_blanket_rate (blanket_count_1 blanket_count_2 blanket_count_3 : ℕ)
  (price_1 price_2 average_price : ℚ) (unknown_rate : ℚ) :
  blanket_count_1 = 4 →
  blanket_count_2 = 5 →
  blanket_count_3 = 2 →
  price_1 = 100 →
  price_2 = 150 →
  average_price = 150 →
  (blanket_count_1 * price_1 + blanket_count_2 * price_2 + blanket_count_3 * unknown_rate) / 
    (blanket_count_1 + blanket_count_2 + blanket_count_3) = average_price →
  unknown_rate = 250 := by
sorry


end NUMINAMATH_CALUDE_unknown_blanket_rate_l2967_296761


namespace NUMINAMATH_CALUDE_sandbox_cost_l2967_296766

/-- The cost to fill a rectangular sandbox with sand -/
theorem sandbox_cost (length width depth price_per_cubic_foot : ℝ) 
  (h_length : length = 4)
  (h_width : width = 3)
  (h_depth : depth = 1.5)
  (h_price : price_per_cubic_foot = 3) : 
  length * width * depth * price_per_cubic_foot = 54 := by
  sorry

#check sandbox_cost

end NUMINAMATH_CALUDE_sandbox_cost_l2967_296766


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l2967_296758

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_pos : ∀ x ≥ 0, f x = x^3 + x) :
  ∀ x < 0, f x = -x^3 - x :=
sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l2967_296758


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2967_296779

theorem triangle_angle_calculation (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.sin B ≠ 0) 
  (h7 : 3 / Real.sin A = Real.sqrt 3 / Real.sin B) (h8 : A = π/3) : B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2967_296779


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2967_296775

/-- 
Given a quadratic equation ax^2 + bx + c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) : 
  ∃ (r : ℝ), b = a*r ∧ c = b*r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2967_296775


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2967_296733

theorem product_of_three_numbers (x y z n : ℤ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y - 5 = n)
  (z_eq : z + 5 = n) :
  x * y * z = 372462 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2967_296733


namespace NUMINAMATH_CALUDE_distance_IP_equals_half_R_minus_r_l2967_296716

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the special points of the triangle
variable (I O G H P : EuclideanSpace ℝ (Fin 2))

-- Define the radii
variable (r R : ℝ)

-- Assumptions
variable (h_incenter : is_incenter I A B C)
variable (h_circumcenter : is_circumcenter O A B C)
variable (h_centroid : is_centroid G A B C)
variable (h_orthocenter : is_orthocenter H A B C)
variable (h_nine_point : is_nine_point_center P A B C)
variable (h_inradius : is_inradius r A B C)
variable (h_circumradius : is_circumradius R A B C)

-- Theorem statement
theorem distance_IP_equals_half_R_minus_r :
  dist I P = R / 2 - r :=
sorry

end NUMINAMATH_CALUDE_distance_IP_equals_half_R_minus_r_l2967_296716


namespace NUMINAMATH_CALUDE_optimal_rental_income_l2967_296707

/-- Represents a travel agency's room rental scenario -/
structure RentalScenario where
  initialRooms : ℕ
  initialRate : ℕ
  rateIncrement : ℕ
  occupancyDecrease : ℕ

/-- Calculates the total daily rental income for a given rate increase -/
def totalIncome (scenario : RentalScenario) (rateIncrease : ℕ) : ℕ :=
  let newRate := scenario.initialRate + rateIncrease
  let newOccupancy := scenario.initialRooms - (rateIncrease / scenario.rateIncrement) * scenario.occupancyDecrease
  newRate * newOccupancy

/-- Finds the optimal rate increase to maximize total daily rental income -/
def optimalRateIncrease (scenario : RentalScenario) : ℕ :=
  sorry

/-- Calculates the increase in total daily rental income -/
def incomeIncrease (scenario : RentalScenario) : ℕ :=
  totalIncome scenario (optimalRateIncrease scenario) - totalIncome scenario 0

/-- Theorem stating the optimal rate increase and income increase for the given scenario -/
theorem optimal_rental_income (scenario : RentalScenario) 
  (h1 : scenario.initialRooms = 120)
  (h2 : scenario.initialRate = 50)
  (h3 : scenario.rateIncrement = 5)
  (h4 : scenario.occupancyDecrease = 6) :
  optimalRateIncrease scenario = 25 ∧ incomeIncrease scenario = 750 := by
  sorry

end NUMINAMATH_CALUDE_optimal_rental_income_l2967_296707


namespace NUMINAMATH_CALUDE_binomial_10_3_l2967_296712

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2967_296712


namespace NUMINAMATH_CALUDE_shop_width_l2967_296771

/-- Given a rectangular shop with the following properties:
  * Length is 18 feet
  * Monthly rent is Rs. 3600
  * Annual rent per square foot is Rs. 120
  Prove that the width of the shop is 20 feet. -/
theorem shop_width (length : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) 
  (h1 : length = 18)
  (h2 : monthly_rent = 3600)
  (h3 : annual_rent_per_sqft = 120) :
  (monthly_rent * 12) / (length * annual_rent_per_sqft) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shop_width_l2967_296771


namespace NUMINAMATH_CALUDE_add_5_23_base6_l2967_296756

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 -/
def addBase6 (a b : ℕ) : ℕ := base10To6 (base6To10 a + base6To10 b)

theorem add_5_23_base6 : addBase6 5 23 = 32 := by sorry

end NUMINAMATH_CALUDE_add_5_23_base6_l2967_296756


namespace NUMINAMATH_CALUDE_triangle_least_perimeter_l2967_296780

theorem triangle_least_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + c ≤ a + b + x) →
  a + b + c = 75 :=
sorry

end NUMINAMATH_CALUDE_triangle_least_perimeter_l2967_296780


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_B_subset_A_range_l2967_296705

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a^2 - 1) < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B (Real.sqrt 2)) = {x | 1 ≤ x ∧ x ≤ Real.sqrt 2 ∨ 3 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem B_subset_A_range :
  ∀ a : ℝ, B a ⊆ A → 1 ≤ a ∧ a ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_B_subset_A_range_l2967_296705


namespace NUMINAMATH_CALUDE_tims_cards_l2967_296769

theorem tims_cards (ben_initial : ℕ) (ben_bought : ℕ) (tim : ℕ) : 
  ben_initial = 37 →
  ben_bought = 3 →
  ben_initial + ben_bought = 2 * tim →
  tim = 20 := by
sorry

end NUMINAMATH_CALUDE_tims_cards_l2967_296769


namespace NUMINAMATH_CALUDE_line_equation_correct_l2967_296731

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def point_on_line (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vector_parallel_to_line (v : Vector2D) (l : Line2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The line we're considering --/
def line_l : Line2D :=
  { a := 1, b := 2, c := -1 }

/-- The point A --/
def point_A : Point2D :=
  { x := 1, y := 0 }

/-- The direction vector of line l --/
def direction_vector : Vector2D :=
  { x := 2, y := -1 }

theorem line_equation_correct :
  point_on_line line_l point_A ∧
  vector_parallel_to_line direction_vector line_l :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2967_296731


namespace NUMINAMATH_CALUDE_monotonicity_condition_l2967_296734

/-- A function f is monotonically increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = kx^2 + (3k-2)x - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (3*k - 2) * x - 5

theorem monotonicity_condition (k : ℝ) :
  (MonotonicallyIncreasing (f k) 1) ↔ k ≥ 2/5 := by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l2967_296734


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_3x_plus_2_to_8_l2967_296767

theorem coefficient_x_cubed_3x_plus_2_to_8 : 
  (Finset.range 9).sum (λ k => Nat.choose 8 k * (3 ^ k) * (2 ^ (8 - k)) * if k = 3 then 1 else 0) = 48384 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_3x_plus_2_to_8_l2967_296767


namespace NUMINAMATH_CALUDE_pencils_remainder_l2967_296796

theorem pencils_remainder : Nat.mod 13254839 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remainder_l2967_296796


namespace NUMINAMATH_CALUDE_all_lines_pass_through_point_common_point_is_neg_two_two_l2967_296729

/-- A line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a, b, c form an arithmetic progression with common difference 3d -/
def is_ap (l : Line) (d : ℝ) : Prop :=
  l.b = l.a + 3 * d ∧ l.c = l.a + 6 * d

/-- Check if a point (x, y) lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Theorem stating that all lines satisfying the arithmetic progression condition pass through (-2, 2) -/
theorem all_lines_pass_through_point (l : Line) (d : ℝ) :
  is_ap l d → point_on_line l (-2) 2 := by
  sorry

/-- Main theorem proving the common point is (-2, 2) -/
theorem common_point_is_neg_two_two :
  ∃ (x y : ℝ), ∀ (l : Line) (d : ℝ), is_ap l d → point_on_line l x y ∧ x = -2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_point_common_point_is_neg_two_two_l2967_296729


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2967_296776

/-- Given a polynomial of the form 2a*x^4 + 5a*x^3 - 13x^2 - x^4 + 2021 + 2x + b*x^3 - b*x^4 - 13x^3,
    if it is a quadratic polynomial, then a^2 + b^2 = 13 -/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x, (2*a - 1 - b) * x^4 + (5*a + b - 13) * x^3 - 13*x^2 + 2*x + 2021 = 0 → 
        ∃ p q r : ℝ, ∀ x, p*x^2 + q*x + r = 0) →
  a^2 + b^2 = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2967_296776


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l2967_296718

theorem number_of_divisors_of_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l2967_296718


namespace NUMINAMATH_CALUDE_longest_tennis_match_duration_l2967_296742

theorem longest_tennis_match_duration (hours : ℕ) (minutes : ℕ) : 
  hours = 11 ∧ minutes = 5 → hours * 60 + minutes = 665 := by sorry

end NUMINAMATH_CALUDE_longest_tennis_match_duration_l2967_296742


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2967_296790

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2967_296790


namespace NUMINAMATH_CALUDE_min_moves_to_monochrome_l2967_296704

/-- A move on a checkerboard that inverts colors in a rectangle -/
structure Move where
  top_left : Nat × Nat
  bottom_right : Nat × Nat

/-- A checkerboard with m rows and n columns -/
structure Checkerboard (m n : Nat) where
  board : Matrix (Fin m) (Fin n) Bool

/-- The result of applying a move to a checkerboard -/
def apply_move (board : Checkerboard m n) (move : Move) : Checkerboard m n :=
  sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Check if a checkerboard is monochrome -/
def is_monochrome (board : Checkerboard m n) : Prop :=
  sorry

/-- The theorem stating the minimum number of moves required -/
theorem min_moves_to_monochrome (m n : Nat) :
  ∃ (moves : MoveSequence),
    (∀ (board : Checkerboard m n),
      is_monochrome (moves.foldl apply_move board)) ∧
    moves.length = Nat.floor (n / 2) + Nat.floor (m / 2) ∧
    (∀ (other_moves : MoveSequence),
      (∀ (board : Checkerboard m n),
        is_monochrome (other_moves.foldl apply_move board)) →
      other_moves.length ≥ moves.length) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_monochrome_l2967_296704


namespace NUMINAMATH_CALUDE_ice_cube_volume_l2967_296736

theorem ice_cube_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (initial_volume * (1/4) * (1/4) = 0.75) →
  initial_volume = 12 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l2967_296736


namespace NUMINAMATH_CALUDE_min_value_sum_l2967_296715

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b ≥ 9 / (2 * a) + 2 / b) : 
  a + b ≥ 5 * Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y ≥ 9 / (2 * x) + 2 / y → x + y ≥ a + b :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2967_296715


namespace NUMINAMATH_CALUDE_herring_cost_theorem_l2967_296794

def green_herring_price : ℝ := 2.50
def blue_herring_price : ℝ := 4.00
def green_herring_pounds : ℝ := 12
def blue_herring_pounds : ℝ := 7

theorem herring_cost_theorem :
  green_herring_price * green_herring_pounds + blue_herring_price * blue_herring_pounds = 58 :=
by sorry

end NUMINAMATH_CALUDE_herring_cost_theorem_l2967_296794


namespace NUMINAMATH_CALUDE_remaining_sessions_proof_l2967_296711

theorem remaining_sessions_proof (total_patients : Nat) (total_sessions : Nat) 
  (patient1_sessions : Nat) (extra_sessions : Nat) :
  total_patients = 4 →
  total_sessions = 25 →
  patient1_sessions = 6 →
  extra_sessions = 5 →
  total_sessions - (patient1_sessions + (patient1_sessions + extra_sessions)) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_sessions_proof_l2967_296711


namespace NUMINAMATH_CALUDE_total_money_calculation_l2967_296748

/-- Proves that the total amount of money is Rs 3000 given the specified conditions -/
theorem total_money_calculation (part1 part2 total interest_rate1 interest_rate2 total_interest : ℝ) :
  part1 = 300 →
  interest_rate1 = 0.03 →
  interest_rate2 = 0.05 →
  total_interest = 144 →
  total = part1 + part2 →
  part1 * interest_rate1 + part2 * interest_rate2 = total_interest →
  total = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_money_calculation_l2967_296748


namespace NUMINAMATH_CALUDE_min_legs_correct_l2967_296782

/-- The length of the circular track in meters -/
def track_length : ℕ := 660

/-- The length of each leg of the race in meters -/
def leg_length : ℕ := 150

/-- The minimum number of legs required for the relay race -/
def min_legs : ℕ := 22

/-- Theorem stating that the minimum number of legs is correct -/
theorem min_legs_correct :
  min_legs = Nat.lcm track_length leg_length / leg_length :=
by sorry

end NUMINAMATH_CALUDE_min_legs_correct_l2967_296782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2967_296708

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_ninth_term :
  ∀ n : ℕ, arithmeticSequence 1 (-2) n = -15 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2967_296708


namespace NUMINAMATH_CALUDE_words_with_at_least_two_consonants_l2967_296755

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- A function that counts the number of 5-letter words with at least two consonants -/
def countWordsWithAtLeastTwoConsonants : Nat := sorry

theorem words_with_at_least_two_consonants :
  countWordsWithAtLeastTwoConsonants = 7424 := by sorry

end NUMINAMATH_CALUDE_words_with_at_least_two_consonants_l2967_296755


namespace NUMINAMATH_CALUDE_carmen_candle_usage_l2967_296764

/-- Calculates the number of candles needed for a given number of nights and burning hours per night. -/
def candles_needed (total_nights : ℕ) (hours_per_night : ℕ) (nights_per_candle_at_one_hour : ℕ) : ℕ :=
  total_nights * hours_per_night / nights_per_candle_at_one_hour

theorem carmen_candle_usage :
  candles_needed 24 2 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carmen_candle_usage_l2967_296764


namespace NUMINAMATH_CALUDE_apartments_can_decrease_l2967_296788

/-- Represents a building configuration -/
structure Building where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a building -/
def total_apartments (b : Building) : ℕ :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Represents the modifications made to a building -/
structure Modification where
  entrances_removed : ℕ
  floors_added : ℕ

/-- Applies a modification to a building -/
def apply_modification (b : Building) (m : Modification) : Building :=
  { entrances := b.entrances - m.entrances_removed,
    floors := b.floors + m.floors_added,
    apartments_per_floor := b.apartments_per_floor }

/-- Theorem: It's possible for the number of apartments to decrease after modifications -/
theorem apartments_can_decrease (initial : Building) (mod1 mod2 : Modification) :
  ∃ (final : Building),
    final = apply_modification (apply_modification initial mod1) mod2 ∧
    total_apartments final < total_apartments initial :=
  sorry


end NUMINAMATH_CALUDE_apartments_can_decrease_l2967_296788


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2967_296722

def z : ℂ := (-2 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2967_296722


namespace NUMINAMATH_CALUDE_parabola_equation_l2967_296735

/-- Prove that for a parabola y^2 = 2px with p > 0, if there exists a point M(3, y) on the parabola
    such that the distance from M to the focus F(p/2, 0) is 5, then p = 4. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  (∃ y : ℝ, y^2 = 2*p*3 ∧ (3 - p/2)^2 + y^2 = 5^2) → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2967_296735


namespace NUMINAMATH_CALUDE_square_prism_properties_l2967_296706

/-- A right prism with a square base -/
structure SquarePrism where
  base_side : ℝ
  height : ℝ

/-- The lateral surface area of a square prism -/
def lateral_surface_area (p : SquarePrism) : ℝ := 4 * p.base_side * p.height

/-- The total surface area of a square prism -/
def surface_area (p : SquarePrism) : ℝ := 2 * p.base_side^2 + lateral_surface_area p

/-- The volume of a square prism -/
def volume (p : SquarePrism) : ℝ := p.base_side^2 * p.height

/-- Theorem about the surface area and volume of a specific square prism -/
theorem square_prism_properties :
  ∃ (p : SquarePrism), 
    lateral_surface_area p = 6^2 ∧ 
    surface_area p = 40.5 ∧ 
    volume p = 3.375 := by
  sorry


end NUMINAMATH_CALUDE_square_prism_properties_l2967_296706


namespace NUMINAMATH_CALUDE_arithmetic_mean_with_additional_number_l2967_296745

theorem arithmetic_mean_with_additional_number : 
  let numbers : List ℕ := [16, 24, 45, 63]
  let additional_number := 2 * numbers.head!
  let total_sum := numbers.sum + additional_number
  let count := numbers.length + 1
  (total_sum : ℚ) / count = 36 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_with_additional_number_l2967_296745


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l2967_296709

theorem expansion_coefficient_sum (a : ℤ) (n : ℕ) : 
  (2^n = 64) → 
  ((1 + a)^6 = 729) → 
  (a = -4 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l2967_296709


namespace NUMINAMATH_CALUDE_billys_age_l2967_296728

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 45) : 
  billy = 30 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l2967_296728


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2967_296757

theorem inequality_equivalence (x : ℝ) : -9 < 2*x - 1 ∧ 2*x - 1 ≤ 6 → -4 < x ∧ x ≤ 3.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2967_296757


namespace NUMINAMATH_CALUDE_seed_distribution_l2967_296732

theorem seed_distribution (total : ℕ) (a b c : ℕ) : 
  total = 100 →
  a = b + 10 →
  b = 30 →
  total = a + b + c →
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_seed_distribution_l2967_296732


namespace NUMINAMATH_CALUDE_remaining_score_is_40_l2967_296773

/-- Represents the score of a dodgeball player -/
structure PlayerScore where
  hitting : ℕ
  catching : ℕ
  eliminating : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : PlayerScore) : ℕ :=
  2 * score.hitting + 5 * score.catching + 10 * score.eliminating

/-- Represents the scores of all players in the game -/
structure GameScores where
  paige : PlayerScore
  brian : PlayerScore
  karen : PlayerScore
  jennifer : PlayerScore
  michael : PlayerScore

/-- The main theorem to prove -/
theorem remaining_score_is_40 (game : GameScores) : 
  totalScore game.paige = 21 →
  totalScore game.brian = 20 →
  game.karen.eliminating = 0 →
  game.jennifer.eliminating = 0 →
  game.michael.eliminating = 0 →
  totalScore game.paige + totalScore game.brian + 
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 81 →
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 40 := by
  sorry

#check remaining_score_is_40

end NUMINAMATH_CALUDE_remaining_score_is_40_l2967_296773


namespace NUMINAMATH_CALUDE_equation_solution_l2967_296750

theorem equation_solution (a b : ℝ) : 
  a^2 + b^2 + 2*a - 4*b + 5 = 0 → 2*a^2 + 4*b - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2967_296750


namespace NUMINAMATH_CALUDE_sum_largest_smallest_even_le_49_l2967_296774

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def largest_even_le_49 : ℕ := 48

def smallest_even_gt_0_le_49 : ℕ := 2

theorem sum_largest_smallest_even_le_49 :
  largest_even_le_49 + smallest_even_gt_0_le_49 = 50 ∧
  is_even largest_even_le_49 ∧
  is_even smallest_even_gt_0_le_49 ∧
  largest_even_le_49 ≤ 49 ∧
  smallest_even_gt_0_le_49 > 0 ∧
  smallest_even_gt_0_le_49 ≤ 49 ∧
  ∀ n, is_even n ∧ n > 0 ∧ n ≤ 49 → n ≤ largest_even_le_49 ∧ n ≥ smallest_even_gt_0_le_49 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_even_le_49_l2967_296774


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l2967_296754

-- Define the structure of the cube
structure Cube where
  edge_length : ℝ
  small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

-- Define the larger cube
def larger_cube : Cube :=
  { edge_length := 4
  , small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

-- Function to calculate the surface area of a cube
def surface_area (c : Cube) : ℝ :=
  6 * c.edge_length ^ 2

-- Function to calculate the number of exposed black faces
def exposed_black_faces (c : Cube) : ℕ :=
  24 + 4  -- 8 corners with 3 faces each, plus 4 along the top edge

-- Theorem stating the fraction of white surface area
theorem white_surface_area_fraction (c : Cube) :
  c = larger_cube →
  (surface_area c - exposed_black_faces c) / surface_area c = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l2967_296754


namespace NUMINAMATH_CALUDE_right_triangle_polyhedron_faces_even_l2967_296717

/-- A convex polyhedron with right-angled triangular faces -/
structure RightTrianglePolyhedron where
  faces : ℕ
  isConvex : Bool
  allFacesRightTriangle : Bool
  facesAtLeastFour : faces ≥ 4

/-- Theorem stating that the number of faces in a right-angled triangle polyhedron is even -/
theorem right_triangle_polyhedron_faces_even (p : RightTrianglePolyhedron) : 
  Even p.faces := by sorry

end NUMINAMATH_CALUDE_right_triangle_polyhedron_faces_even_l2967_296717


namespace NUMINAMATH_CALUDE_salary_fraction_on_food_l2967_296743

theorem salary_fraction_on_food
  (salary : ℝ)
  (rent_fraction : ℝ)
  (clothes_fraction : ℝ)
  (remaining : ℝ)
  (h1 : salary = 180000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 18000)
  (h5 : ∃ food_fraction : ℝ, 
    food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary) :
  ∃ food_fraction : ℝ, food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_fraction_on_food_l2967_296743


namespace NUMINAMATH_CALUDE_largest_common_term_l2967_296798

def is_in_first_sequence (x : ℕ) : Prop := ∃ n : ℕ, x = 2 + 5 * n

def is_in_second_sequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 8 * m

theorem largest_common_term : 
  (∀ x : ℕ, x ≤ 150 → is_in_first_sequence x → is_in_second_sequence x → x ≤ 117) ∧ 
  is_in_first_sequence 117 ∧ 
  is_in_second_sequence 117 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l2967_296798


namespace NUMINAMATH_CALUDE_sprained_vs_normal_time_difference_l2967_296710

/-- The time it takes Ann to frost a cake normally, in minutes -/
def normal_time : ℕ := 5

/-- The time it takes Ann to frost a cake with a sprained wrist, in minutes -/
def sprained_time : ℕ := 8

/-- The number of cakes Ann needs to frost -/
def num_cakes : ℕ := 10

/-- Theorem stating the difference in time to frost 10 cakes between sprained and normal conditions -/
theorem sprained_vs_normal_time_difference : 
  sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end NUMINAMATH_CALUDE_sprained_vs_normal_time_difference_l2967_296710


namespace NUMINAMATH_CALUDE_correct_divisor_proof_l2967_296791

theorem correct_divisor_proof (dividend : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : dividend % mistaken_divisor = 0)
  (h2 : dividend / mistaken_divisor = 63)
  (h3 : mistaken_divisor = 12)
  (h4 : dividend % correct_quotient = 0)
  (h5 : dividend / correct_quotient = 36) :
  dividend / 36 = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_proof_l2967_296791


namespace NUMINAMATH_CALUDE_adiabatic_compression_work_l2967_296759

theorem adiabatic_compression_work
  (k : ℝ) (p₁ V₁ V₂ : ℝ) (h_k : k > 1) (h_V : V₂ > 0) :
  let W := (p₁ * V₁) / (k - 1) * (1 - (V₁ / V₂) ^ (k - 1))
  let c := p₁ * V₁^k
  ∀ (p v : ℝ), p * v^k = c →
  W = -(∫ (x : ℝ) in V₁..V₂, c / x^k) :=
sorry

end NUMINAMATH_CALUDE_adiabatic_compression_work_l2967_296759


namespace NUMINAMATH_CALUDE_degree_of_product_l2967_296797

/-- The degree of a polynomial resulting from the multiplication of three given expressions -/
theorem degree_of_product : ℕ :=
  let expr1 := (fun x : ℝ => x^5)
  let expr2 := (fun x : ℝ => x + 1/x)
  let expr3 := (fun x : ℝ => 1 + 3/x + 4/x^2 + 5/x^3)
  let product := (fun x : ℝ => expr1 x * expr2 x * expr3 x)
  6

#check degree_of_product

end NUMINAMATH_CALUDE_degree_of_product_l2967_296797


namespace NUMINAMATH_CALUDE_eight_valid_arrangements_l2967_296720

/-- A type representing the possible positions of a card -/
inductive Position
  | Original
  | Left
  | Right

/-- A type representing a card arrangement -/
def Arrangement := Fin 5 → Position

/-- A function to check if an arrangement is valid -/
def is_valid (arr : Arrangement) : Prop :=
  ∀ i : Fin 5, arr i = Position.Original ∨ arr i = Position.Left ∨ arr i = Position.Right

/-- The number of valid arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 8 valid arrangements -/
theorem eight_valid_arrangements : num_valid_arrangements = 8 := by sorry

end NUMINAMATH_CALUDE_eight_valid_arrangements_l2967_296720


namespace NUMINAMATH_CALUDE_geometric_proof_l2967_296778

/-- The problem setup for the geometric proof -/
structure GeometricSetup where
  -- Line l equation
  l : ℝ → ℝ → Prop
  l_def : ∀ x y, l x y ↔ x + 2 * y - 1 = 0

  -- Circle C equations
  C : ℝ → ℝ → Prop
  C_def : ∀ x y, C x y ↔ ∃ φ, x = 3 + 3 * Real.cos φ ∧ y = 3 * Real.sin φ

  -- Ray OM
  α : ℝ
  α_range : 0 < α ∧ α < Real.pi / 2

  -- Function to convert Cartesian to polar coordinates
  to_polar : ℝ × ℝ → ℝ × ℝ

  -- Function to get the length of OP
  OP_length : ℝ

  -- Function to get the length of OQ
  OQ_length : ℝ

/-- The main theorem to be proved -/
theorem geometric_proof (setup : GeometricSetup) : 
  setup.OP_length * setup.OQ_length = 6 → setup.α = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_proof_l2967_296778


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_norm_l2967_296783

/-- Two vectors in ℝ² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x + 1, 2]
def b : Fin 2 → ℝ := ![1, -1]

/-- Parallel vectors have proportional components -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * w i

theorem parallel_vectors_sum_norm (x : ℝ) :
  parallel (a x) b → ‖(a x) + b‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_norm_l2967_296783


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2967_296784

theorem arithmetic_sequence_middle_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
    a 0 = 3^2 →                                           -- first term is 3^2
    a 2 = 3^3 →                                           -- third term is 3^3
    a 1 = 18 :=                                           -- second term is 18
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2967_296784


namespace NUMINAMATH_CALUDE_wire_attachment_point_existence_l2967_296762

theorem wire_attachment_point_existence :
  ∃! x : ℝ, 0 < x ∧ x < 5 ∧ Real.sqrt (x^2 + 3.6^2) + Real.sqrt ((x + 5)^2 + 3.6^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_wire_attachment_point_existence_l2967_296762


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_24_degrees_l2967_296752

/-- Theorem: For a regular polygon with exterior angles measuring 24 degrees each,
    the number of sides is 15 and the sum of interior angles is 2340 degrees. -/
theorem regular_polygon_exterior_24_degrees :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  n = 15 ∧ (n - 2) * 180 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_24_degrees_l2967_296752


namespace NUMINAMATH_CALUDE_arrange_plants_under_lamps_count_l2967_296724

/-- Represents the number of ways to arrange plants under lamps -/
def arrange_plants_under_lamps : ℕ :=
  let num_plants : ℕ := 4
  let num_plant_types : ℕ := 3
  let num_lamps : ℕ := 4
  let num_lamp_colors : ℕ := 2
  
  -- All plants under same color lamp
  let all_under_one_color : ℕ := num_lamp_colors
  let three_under_one_color : ℕ := num_plants * num_lamp_colors
  
  -- Plants under different colored lamps
  let two_types_each_color : ℕ := (Nat.choose num_plant_types 2) * num_lamp_colors
  let one_type_alone : ℕ := num_plant_types * num_lamp_colors
  
  all_under_one_color + three_under_one_color + two_types_each_color + one_type_alone

/-- Theorem stating the correct number of ways to arrange plants under lamps -/
theorem arrange_plants_under_lamps_count :
  arrange_plants_under_lamps = 22 := by sorry

end NUMINAMATH_CALUDE_arrange_plants_under_lamps_count_l2967_296724


namespace NUMINAMATH_CALUDE_square_roots_problem_l2967_296739

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (x + 2)^2 = a) (h3 : (2*x - 5)^2 = a) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2967_296739


namespace NUMINAMATH_CALUDE_tan_half_product_l2967_296719

theorem tan_half_product (a b : Real) :
  7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 5 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l2967_296719


namespace NUMINAMATH_CALUDE_instantaneous_velocity_zero_at_two_l2967_296777

-- Define the motion law
def motion_law (t : ℝ) : ℝ := t^2 - 4*t + 5

-- Define the instantaneous velocity (derivative of motion law)
def instantaneous_velocity (t : ℝ) : ℝ := 2*t - 4

-- Theorem statement
theorem instantaneous_velocity_zero_at_two :
  ∃ (t : ℝ), instantaneous_velocity t = 0 ∧ t = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_zero_at_two_l2967_296777


namespace NUMINAMATH_CALUDE_height_relation_l2967_296786

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  equal_volume : r1^2 * h1 = r2^2 * h2  -- cylinders have equal volume
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l2967_296786


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2967_296785

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 :=
by
  sorry

theorem problem_solution :
  let n := 13294
  let d := 97
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 ∧ k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2967_296785


namespace NUMINAMATH_CALUDE_tree_height_reaches_29_feet_in_15_years_l2967_296703

/-- Calculates the height of the tree after a given number of years -/
def tree_height (years : ℕ) : ℕ :=
  let initial_height := 4
  let first_year_growth := 5
  let second_year_growth := 4
  let min_growth := 1
  let rec height_after (n : ℕ) (current_height : ℕ) (current_growth : ℕ) : ℕ :=
    if n = 0 then
      current_height
    else if n = 1 then
      height_after (n - 1) (current_height + first_year_growth) second_year_growth
    else if current_growth > min_growth then
      height_after (n - 1) (current_height + current_growth) (current_growth - 1)
    else
      height_after (n - 1) (current_height + min_growth) min_growth
  height_after years initial_height first_year_growth

/-- Theorem stating that it takes 15 years for the tree to reach or exceed 29 feet -/
theorem tree_height_reaches_29_feet_in_15_years :
  tree_height 15 ≥ 29 ∧ ∀ y : ℕ, y < 15 → tree_height y < 29 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_reaches_29_feet_in_15_years_l2967_296703


namespace NUMINAMATH_CALUDE_item_list_price_l2967_296792

theorem item_list_price (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_item_list_price_l2967_296792


namespace NUMINAMATH_CALUDE_integer_solution_of_quadratic_equation_l2967_296760

theorem integer_solution_of_quadratic_equation (x y : ℤ) :
  x^2 + y^2 = 3*x*y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_quadratic_equation_l2967_296760


namespace NUMINAMATH_CALUDE_tangent_slope_implies_tan_value_l2967_296740

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - (1/4) * sin x - (Real.sqrt 3 / 4) * cos x

theorem tangent_slope_implies_tan_value (x₀ : ℝ) :
  (deriv f x₀ = 1) → tan x₀ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_tan_value_l2967_296740


namespace NUMINAMATH_CALUDE_sammy_janine_bottle_cap_difference_l2967_296723

/-- Proof that Sammy has 2 more bottle caps than Janine -/
theorem sammy_janine_bottle_cap_difference :
  ∀ (sammy janine billie : ℕ),
    sammy > janine →
    janine = 3 * billie →
    billie = 2 →
    sammy = 8 →
    sammy - janine = 2 := by
  sorry

end NUMINAMATH_CALUDE_sammy_janine_bottle_cap_difference_l2967_296723


namespace NUMINAMATH_CALUDE_more_pockets_than_dollars_per_wallet_l2967_296795

/-- Represents the distribution of dollars, wallets, and pockets -/
structure Distribution where
  total_dollars : ℕ
  num_wallets : ℕ
  num_pockets : ℕ
  dollars_per_pocket : ℕ → ℕ
  dollars_per_wallet : ℕ → ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Distribution) : Prop :=
  d.total_dollars = 2003 ∧
  d.num_wallets > 0 ∧
  d.num_pockets > 0 ∧
  (∀ p, p < d.num_pockets → d.dollars_per_pocket p < d.num_wallets) ∧
  (∀ w, w < d.num_wallets → d.dollars_per_wallet w ≤ d.total_dollars / d.num_wallets)

/-- The theorem to be proved -/
theorem more_pockets_than_dollars_per_wallet (d : Distribution) 
  (h : problem_conditions d) : 
  ∀ w, w < d.num_wallets → d.num_pockets > d.dollars_per_wallet w :=
sorry

end NUMINAMATH_CALUDE_more_pockets_than_dollars_per_wallet_l2967_296795


namespace NUMINAMATH_CALUDE_integer_solution_zero_l2967_296770

theorem integer_solution_zero (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_solution_zero_l2967_296770


namespace NUMINAMATH_CALUDE_inequality_proof_l2967_296738

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2967_296738


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2967_296744

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = -5/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2967_296744


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2967_296751

theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ → ℝ := fun θ r ↦ r
  let circle_equation : ℝ → ℝ → Prop := fun θ r ↦ ρ θ r = Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  let is_center : ℝ → ℝ → Prop := fun r θ ↦ ∀ θ' r', circle_equation θ' r' → 
    (r * Real.cos θ - r' * Real.cos θ')^2 + (r * Real.sin θ - r' * Real.sin θ')^2 = r^2
  is_center 1 (Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2967_296751


namespace NUMINAMATH_CALUDE_lana_muffin_sales_l2967_296741

/-- Lana's muffin sales problem -/
theorem lana_muffin_sales (goal : ℕ) (morning_sales : ℕ) (afternoon_sales : ℕ)
  (h1 : goal = 20)
  (h2 : morning_sales = 12)
  (h3 : afternoon_sales = 4) :
  goal - morning_sales - afternoon_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_lana_muffin_sales_l2967_296741


namespace NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l2967_296763

/-- Converts a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to base 5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l2967_296763


namespace NUMINAMATH_CALUDE_equation_with_parentheses_is_true_l2967_296726

theorem equation_with_parentheses_is_true : 7 * 9 + 12 / (3 - 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_equation_with_parentheses_is_true_l2967_296726


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l2967_296725

/-- Given a parabola and an ellipse, prove that the parameter m of the parabola
    has a specific value when the focus of the parabola coincides with the left
    focus of the ellipse. -/
theorem parabola_ellipse_focus_coincidence (m : ℝ) : 
  (∀ x y : ℝ, y^2 = (4/m)*x → x^2/7 + y^2/3 = 1) →
  (∃ x y : ℝ, y^2 = (4/m)*x ∧ x^2/7 + y^2/3 = 1 ∧ x = -2) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l2967_296725


namespace NUMINAMATH_CALUDE_range_of_m_range_of_a_l2967_296765

-- Define the propositions
def p (m : ℝ) : Prop := |m - 2| < 1
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*Real.sqrt 2*x + m = 0
def r (m a : ℝ) : Prop := a - 2 < m ∧ m < a + 1

-- Theorem 1
theorem range_of_m (m : ℝ) : p m ∧ ¬(q m) → 2 < m ∧ m < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : 
  (∀ m : ℝ, p m → r m a) ∧ ¬(∀ m : ℝ, r m a → p m) → 
  2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_a_l2967_296765
