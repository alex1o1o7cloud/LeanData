import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_squared_l1149_114969

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of the sequence equals 27. -/
def SumFirstFiveIs27 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 27

/-- The sum of the reciprocals of the first five terms of the sequence equals 3. -/
def SumReciprocalFirstFiveIs3 (a : ℕ → ℝ) : Prop :=
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = 3

theorem geometric_sequence_third_term_squared
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_sum : SumFirstFiveIs27 a)
  (h_sum_reciprocal : SumReciprocalFirstFiveIs3 a) :
  (a 3) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_squared_l1149_114969


namespace NUMINAMATH_CALUDE_sin_product_seventh_pi_l1149_114984

theorem sin_product_seventh_pi : 
  Real.sin (π / 7) * Real.sin (2 * π / 7) * Real.sin (3 * π / 7) = Real.sqrt 13 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_seventh_pi_l1149_114984


namespace NUMINAMATH_CALUDE_square_plus_one_greater_than_one_l1149_114993

theorem square_plus_one_greater_than_one (a : ℝ) (h : a ≠ 0) : a^2 + 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_greater_than_one_l1149_114993


namespace NUMINAMATH_CALUDE_total_cost_of_shed_is_818_25_l1149_114995

/-- Calculate the total cost of constructing a shed given the following conditions:
  * 1000 bricks are needed
  * 30% of bricks are at 50% discount off $0.50 each
  * 40% of bricks are at 20% discount off $0.50 each
  * 30% of bricks are at full price of $0.50 each
  * 5% tax on total cost of bricks
  * Additional building materials cost $200
  * 7% tax on additional building materials
  * Labor fees are $20 per hour for 10 hours
-/
def total_cost_of_shed : ℝ :=
  let total_bricks : ℝ := 1000
  let brick_full_price : ℝ := 0.50
  let discounted_bricks_1 : ℝ := 0.30 * total_bricks
  let discounted_bricks_2 : ℝ := 0.40 * total_bricks
  let full_price_bricks : ℝ := 0.30 * total_bricks
  let discount_1 : ℝ := 0.50
  let discount_2 : ℝ := 0.20
  let brick_tax_rate : ℝ := 0.05
  let additional_materials_cost : ℝ := 200
  let materials_tax_rate : ℝ := 0.07
  let labor_rate : ℝ := 20
  let labor_hours : ℝ := 10

  let discounted_price_1 : ℝ := brick_full_price * (1 - discount_1)
  let discounted_price_2 : ℝ := brick_full_price * (1 - discount_2)
  
  let brick_cost : ℝ := 
    discounted_bricks_1 * discounted_price_1 +
    discounted_bricks_2 * discounted_price_2 +
    full_price_bricks * brick_full_price
  
  let brick_tax : ℝ := brick_cost * brick_tax_rate
  let materials_tax : ℝ := additional_materials_cost * materials_tax_rate
  let labor_cost : ℝ := labor_rate * labor_hours

  brick_cost + brick_tax + additional_materials_cost + materials_tax + labor_cost

theorem total_cost_of_shed_is_818_25 : 
  total_cost_of_shed = 818.25 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_shed_is_818_25_l1149_114995


namespace NUMINAMATH_CALUDE_product_distribution_l1149_114998

theorem product_distribution (n : ℕ) (h : n = 6) :
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) =
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) :=
by sorry

end NUMINAMATH_CALUDE_product_distribution_l1149_114998


namespace NUMINAMATH_CALUDE_divisibility_by_100_l1149_114963

theorem divisibility_by_100 (a : ℕ) (h : ¬(5 ∣ a)) : 100 ∣ (a^8 + 3*a^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_100_l1149_114963


namespace NUMINAMATH_CALUDE_equation_always_has_real_root_l1149_114959

theorem equation_always_has_real_root :
  ∀ (q : ℝ), ∃ (x : ℝ), x^6 + q*x^4 + q^2*x^2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_always_has_real_root_l1149_114959


namespace NUMINAMATH_CALUDE_value_of_shares_theorem_l1149_114991

/-- Represents the value of shares bought by an investor -/
def value_of_shares (N : ℝ) : ℝ := 0.5 * N * 25

/-- Theorem stating the relationship between the value of shares and the number of shares -/
theorem value_of_shares_theorem (N : ℝ) (dividend_rate : ℝ) (return_rate : ℝ) (share_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : return_rate = 0.25)
  (h3 : share_price = 25) :
  value_of_shares N = return_rate * (value_of_shares N) / dividend_rate := by
sorry

end NUMINAMATH_CALUDE_value_of_shares_theorem_l1149_114991


namespace NUMINAMATH_CALUDE_final_staff_count_l1149_114980

/- Define the initial number of staff in each category -/
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def initial_assistants : ℕ := 9
def initial_interns : ℕ := 6

/- Define the number of staff who quit or are transferred -/
def doctors_quit : ℕ := 5
def nurses_quit : ℕ := 2
def assistants_quit : ℕ := 3
def nurses_transferred : ℕ := 2
def interns_transferred : ℕ := 4

/- Define the number of staff on leave -/
def doctors_on_leave : ℕ := 4
def nurses_on_leave : ℕ := 3

/- Define the number of new staff joining -/
def new_doctors : ℕ := 3
def new_nurses : ℕ := 5

/- Theorem to prove the final staff count -/
theorem final_staff_count :
  (initial_doctors - doctors_quit - doctors_on_leave + new_doctors) +
  (initial_nurses - nurses_quit - nurses_transferred - nurses_on_leave + new_nurses) +
  (initial_assistants - assistants_quit) +
  (initial_interns - interns_transferred) = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_staff_count_l1149_114980


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l1149_114926

def is_valid_function (f : ℤ → ℤ) : Prop :=
  f 1 ≠ f (-1) ∧ ∀ m n : ℤ, (f (m + n))^2 ∣ (f m - f n)

theorem characterize_valid_functions :
  ∀ f : ℤ → ℤ, is_valid_function f →
    (∀ x : ℤ, f x = 1 ∨ f x = -1) ∨
    (∀ x : ℤ, f x = 2 ∨ f x = -2) ∧ f 1 = -f (-1) :=
by sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l1149_114926


namespace NUMINAMATH_CALUDE_function_existence_condition_l1149_114934

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n) = n + a) ↔ (a = 0 ∨ a > 0) ∧ k ∣ a :=
by sorry

end NUMINAMATH_CALUDE_function_existence_condition_l1149_114934


namespace NUMINAMATH_CALUDE_christine_savings_l1149_114966

/-- Calculates the amount saved given a commission rate, total sales, and personal needs allocation. -/
def amount_saved (commission_rate : ℝ) (total_sales : ℝ) (personal_needs_rate : ℝ) : ℝ :=
  let commission_earned := commission_rate * total_sales
  let savings_rate := 1 - personal_needs_rate
  savings_rate * commission_earned

/-- Proves that given a 12% commission rate on $24000 worth of sales, 
    and allocating 60% of earnings to personal needs, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved 0.12 24000 0.60 = 1152 := by
sorry

end NUMINAMATH_CALUDE_christine_savings_l1149_114966


namespace NUMINAMATH_CALUDE_exponential_linear_inequalities_l1149_114983

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A linear function with slope k -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem exponential_linear_inequalities (k : ℝ) :
  (∃ (y : ℝ), ∀ (x : ℝ), f x - (x + 1) ≥ y ∧ ∃ (x : ℝ), f x - (x + 1) = y) ∧
  (k > 1 → ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < x₀ → f x < g k x) ∧
  (∃ (m : ℝ), m > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < m → |f x - g k x| > x) ↔ (k ≤ 0 ∨ k > 2) := by
  sorry

end NUMINAMATH_CALUDE_exponential_linear_inequalities_l1149_114983


namespace NUMINAMATH_CALUDE_product_of_first_three_terms_l1149_114902

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_of_first_three_terms 
  (a₁ : ℝ) -- first term
  (d : ℝ) -- common difference
  (h1 : arithmetic_sequence a₁ d 7 = 20) -- seventh term is 20
  (h2 : d = 2) -- common difference is 2
  : a₁ * (a₁ + d) * (a₁ + 2 * d) = 960 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_three_terms_l1149_114902


namespace NUMINAMATH_CALUDE_at_least_one_not_divisible_l1149_114942

theorem at_least_one_not_divisible (a b c d : ℕ) (h : a * d - b * c > 1) :
  ¬(a * d - b * c ∣ a) ∨ ¬(a * d - b * c ∣ b) ∨ ¬(a * d - b * c ∣ c) ∨ ¬(a * d - b * c ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_divisible_l1149_114942


namespace NUMINAMATH_CALUDE_ice_cream_cost_l1149_114957

theorem ice_cream_cost (price : ℚ) (discount : ℚ) : 
  price = 99/100 ∧ discount = 1/10 → 
  price + price * (1 - discount) = 1881/1000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l1149_114957


namespace NUMINAMATH_CALUDE_dark_light_difference_l1149_114956

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := Vector Square 9

/-- Represents the entire 9x9 grid -/
def Grid := Vector Row 9

/-- Creates an alternating row starting with the given square color -/
def alternatingRow (start : Square) : Row := sorry

/-- Creates the 9x9 grid with alternating pattern -/
def createGrid : Grid :=
  Vector.ofFn (λ i => alternatingRow (if i % 2 = 0 then Square.Dark else Square.Light))

/-- Counts the number of dark squares in the grid -/
def countDarkSquares (grid : Grid) : Nat := sorry

/-- Counts the number of light squares in the grid -/
def countLightSquares (grid : Grid) : Nat := sorry

/-- The main theorem stating the difference between dark and light squares -/
theorem dark_light_difference :
  let grid := createGrid
  countDarkSquares grid = countLightSquares grid + 1 := by sorry

end NUMINAMATH_CALUDE_dark_light_difference_l1149_114956


namespace NUMINAMATH_CALUDE_millionth_digit_of_three_forty_first_l1149_114927

def fraction : ℚ := 3 / 41

def decimal_expansion (q : ℚ) : ℕ → ℕ := sorry

def nth_digit_after_decimal_point (q : ℚ) (n : ℕ) : ℕ :=
  decimal_expansion q n

theorem millionth_digit_of_three_forty_first (n : ℕ) (h : n = 1000000) :
  nth_digit_after_decimal_point fraction n = 7 := by sorry

end NUMINAMATH_CALUDE_millionth_digit_of_three_forty_first_l1149_114927


namespace NUMINAMATH_CALUDE_at_least_one_inequality_holds_l1149_114908

-- Define a triangle in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_inequality_holds (t : Triangle) (M N : Point) :
  isInside t M →
  isInside t N →
  M ≠ N →
  (distance t.A N > distance t.A M) ∨
  (distance t.B N > distance t.B M) ∨
  (distance t.C N > distance t.C M) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_inequality_holds_l1149_114908


namespace NUMINAMATH_CALUDE_perpendicular_a_parallel_distance_l1149_114920

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := 2 * a * x + y - 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 1) * y + 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 
  ∃ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ (x₁ - x₂) * (y₁ - y₂) = -1

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := 
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 2 * a * (a - 1) = a

-- Theorem for perpendicular case
theorem perpendicular_a : ∀ a : ℝ, perpendicular a → a = -1 ∨ a = 1/2 :=
sorry

-- Theorem for parallel case
theorem parallel_distance : ∀ a : ℝ, parallel a → a ≠ 1 → 
  ∃ d : ℝ, d = (3 * Real.sqrt 10) / 10 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = d^2)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_a_parallel_distance_l1149_114920


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l1149_114907

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l1149_114907


namespace NUMINAMATH_CALUDE_parallelogram_inscribed_circles_l1149_114931

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram ABCD -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if a circle is inscribed in a triangle -/
def is_inscribed (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def on_segment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_inscribed_circles 
  (ABCD : Parallelogram) 
  (P : Point) 
  (c_ABC : Circle) 
  (c_DAP : Circle) 
  (c_DCP : Circle) :
  on_segment P ABCD.A ABCD.C →
  is_inscribed c_ABC ABCD.A ABCD.B ABCD.C →
  is_inscribed c_DAP ABCD.D ABCD.A P →
  is_inscribed c_DCP ABCD.D ABCD.C P →
  distance ABCD.D ABCD.A + distance ABCD.D ABCD.C = 3 * distance ABCD.A ABCD.C →
  distance ABCD.D ABCD.A = distance ABCD.D P →
  (distance ABCD.D ABCD.A + distance ABCD.A P = distance ABCD.D ABCD.C + distance ABCD.C P) ∧
  (c_DAP.radius / c_DCP.radius = distance ABCD.A P / distance P ABCD.C) ∧
  (c_DAP.radius / c_DCP.radius = 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_inscribed_circles_l1149_114931


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l1149_114973

/-- The actual distance traveled by a person, given two walking speeds and additional distance information. -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (additional_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 10)
  (h3 : additional_distance = 20)
  (h4 : ∀ d : ℝ, d / slow_speed = (d + additional_distance) / fast_speed) :
  ∃ d : ℝ, d = 20 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l1149_114973


namespace NUMINAMATH_CALUDE_minimum_parents_needed_minimum_parents_for_tour_l1149_114922

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let drivers_needed := (total_people + car_capacity - 1) / car_capacity
  drivers_needed

theorem minimum_parents_for_tour :
  minimum_parents_needed 50 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_parents_needed_minimum_parents_for_tour_l1149_114922


namespace NUMINAMATH_CALUDE_inequality_proof_l1149_114932

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let P := Real.sqrt ((a^2 + b^2)/2) - (a + b)/2
  let Q := (a + b)/2 - Real.sqrt (a*b)
  let R := Real.sqrt (a*b) - (2*a*b)/(a + b)
  Q ≥ P ∧ P ≥ R := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1149_114932


namespace NUMINAMATH_CALUDE_tan_2018pi_minus_alpha_l1149_114917

theorem tan_2018pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (3 * π / 2)) 
  (h2 : Real.cos (3 * π / 2 - α) = Real.sqrt 3 / 2) : 
  Real.tan (2018 * π - α) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2018pi_minus_alpha_l1149_114917


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l1149_114929

/-- The number of triangles formed by selecting three non-collinear points from a set of points on a triangle -/
def num_triangles (a b c : ℕ) : ℕ :=
  let total_points := 3 + a + b + c
  let total_combinations := (total_points.choose 3)
  let collinear_combinations := (a + 2).choose 3 + (b + 2).choose 3 + (c + 2).choose 3
  total_combinations - collinear_combinations

/-- Theorem stating that the number of triangles formed in the given configuration is 357 -/
theorem triangle_count_theorem : num_triangles 2 3 7 = 357 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_theorem_l1149_114929


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1149_114962

def f (x : ℝ) : ℝ := 4*x^4 + 17*x^3 - 37*x^2 + 6*x

theorem roots_of_polynomial :
  ∃ (a b c d : ℝ),
    (a = 0) ∧
    (b = 1/2) ∧
    (c = (-9 + Real.sqrt 129) / 4) ∧
    (d = (-9 - Real.sqrt 129) / 4) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1149_114962


namespace NUMINAMATH_CALUDE_linear_system_integer_solution_l1149_114958

theorem linear_system_integer_solution :
  ∃ (x y : ℤ), x + y = 5 ∧ 2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_integer_solution_l1149_114958


namespace NUMINAMATH_CALUDE_muffins_per_box_l1149_114964

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) 
  (h1 : total_muffins = 96) (h2 : num_boxes = 8) :
  total_muffins / num_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_box_l1149_114964


namespace NUMINAMATH_CALUDE_quadrilateral_vector_equality_l1149_114987

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b c : V)

-- Define the points
variable (O A B C M N : V)

-- State the conditions
variable (h1 : M - O = (1/3) • (A - O))
variable (h2 : N - B = (1/2) • (C - B))
variable (h3 : A - O = a)
variable (h4 : B - O = b)
variable (h5 : C - O = c)

-- State the theorem
theorem quadrilateral_vector_equality : 
  M - N = -(2/3) • a + (1/2) • b + (1/2) • c := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_equality_l1149_114987


namespace NUMINAMATH_CALUDE_mice_elimination_time_l1149_114921

/-- Represents the rate at which cats hunt mice -/
def hunting_rate : ℝ := 0.1

/-- Represents the total amount of work to eliminate all mice -/
def total_work : ℝ := 1

/-- Represents the number of days taken by initial cats -/
def initial_days : ℕ := 5

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem mice_elimination_time :
  let initial_work := hunting_rate * initial_cats * initial_days
  let remaining_work := total_work - initial_work
  let final_rate := hunting_rate * final_cats
  initial_days + (remaining_work / final_rate) = 7 := by sorry

end NUMINAMATH_CALUDE_mice_elimination_time_l1149_114921


namespace NUMINAMATH_CALUDE_concentric_squares_ratio_l1149_114911

/-- Given two concentric squares ABCD (outer) and EFGH (inner) with side lengths a and b
    respectively, if the area of the shaded region between them is p% of the area of ABCD,
    then a/b = 1/sqrt(1-p/100). -/
theorem concentric_squares_ratio (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : 0 < p ∧ p < 100) :
  (a^2 - b^2) / a^2 = p / 100 → a / b = 1 / Real.sqrt (1 - p / 100) := by
  sorry

end NUMINAMATH_CALUDE_concentric_squares_ratio_l1149_114911


namespace NUMINAMATH_CALUDE_sum_abs_zero_implies_a_minus_abs_2a_l1149_114946

theorem sum_abs_zero_implies_a_minus_abs_2a (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_zero_implies_a_minus_abs_2a_l1149_114946


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1149_114971

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 m ∧ line_l B.1 B.2 m

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem circle_intersection_theorem (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points A B m →
  ((angle_ACB A B = 2 * π / 3) ∨ (distance A B = 2 * sqrt 3)) →
  (m = sqrt 2 - 1 ∨ m = -sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1149_114971


namespace NUMINAMATH_CALUDE_xyz_value_l1149_114953

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 8/3) :
  x * y * z = (17 + Real.sqrt 285) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1149_114953


namespace NUMINAMATH_CALUDE_rectangle_area_puzzle_l1149_114939

/-- Given a rectangle divided into six smaller rectangles, if five of the rectangles
    have areas 126, 63, 161, 20, and 40, then the area of the remaining rectangle is 101. -/
theorem rectangle_area_puzzle (A B C D E F : ℝ) :
  A = 126 →
  B = 63 →
  C = 161 →
  D = 20 →
  E = 40 →
  A + B + C + D + E + F = (A + B) + C →
  F = 101 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_puzzle_l1149_114939


namespace NUMINAMATH_CALUDE_negative_intervals_l1149_114912

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 2) * (x - 3)

-- Define the set of x for which f(x) is negative
def S : Set ℝ := {x | f x < 0}

-- State the theorem
theorem negative_intervals : S = Set.Iio (-2) ∪ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_negative_intervals_l1149_114912


namespace NUMINAMATH_CALUDE_michael_and_brothers_ages_l1149_114955

/-- The ages of Michael and his brothers satisfy the given conditions and their combined age is 28. -/
theorem michael_and_brothers_ages :
  ∀ (michael_age older_brother_age younger_brother_age : ℕ),
    younger_brother_age = 5 →
    older_brother_age = 3 * younger_brother_age →
    older_brother_age = 1 + 2 * (michael_age - 1) →
    michael_age + older_brother_age + younger_brother_age = 28 :=
by
  sorry


end NUMINAMATH_CALUDE_michael_and_brothers_ages_l1149_114955


namespace NUMINAMATH_CALUDE_certain_number_proof_l1149_114928

theorem certain_number_proof (x : ℕ) (certain_number : ℕ) : 
  (certain_number = 3 * x + 36) → (x = 4) → (certain_number = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1149_114928


namespace NUMINAMATH_CALUDE_sqrt3_expression_equals_zero_l1149_114906

theorem sqrt3_expression_equals_zero :
  Real.sqrt 3 * (1 - Real.sqrt 3) - |-(Real.sqrt 3)| + (27 : ℝ) ^ (1/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_expression_equals_zero_l1149_114906


namespace NUMINAMATH_CALUDE_option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l1149_114944

-- Option A
theorem option_a_same_function (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option B
theorem option_b_different_function : ∃ x : ℤ, 2*x + 1 ≠ 2*x - 1 := by sorry

-- Option C
def domain_f (x : ℝ) : Prop := x^2 ≥ 9
def domain_g (x : ℝ) : Prop := x ≥ 3

theorem option_c_different_domain : domain_f ≠ domain_g := by sorry

-- Option D
theorem option_d_same_function (x t : ℝ) (h : x = t) : x^2 - 2*x - 1 = t^2 - 2*t - 1 := by sorry

end NUMINAMATH_CALUDE_option_a_same_function_option_b_different_function_option_c_different_domain_option_d_same_function_l1149_114944


namespace NUMINAMATH_CALUDE_attic_junk_items_l1149_114935

theorem attic_junk_items (total : ℕ) (useful : ℕ) (junk_percent : ℚ) :
  useful = (20 : ℚ) / 100 * total →
  junk_percent = 70 / 100 →
  useful = 8 →
  ⌊junk_percent * total⌋ = 28 := by
sorry

end NUMINAMATH_CALUDE_attic_junk_items_l1149_114935


namespace NUMINAMATH_CALUDE_fourth_month_sale_problem_l1149_114913

/-- Calculates the sale in the fourth month given the sales of other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating the sale in the fourth month given the problem conditions --/
theorem fourth_month_sale_problem :
  fourthMonthSale 5420 5660 6200 6500 8270 6400 = 6350 := by
  sorry

#eval fourthMonthSale 5420 5660 6200 6500 8270 6400

end NUMINAMATH_CALUDE_fourth_month_sale_problem_l1149_114913


namespace NUMINAMATH_CALUDE_triangle_theorem_l1149_114918

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC where c sin B = √3 b cos C and a + b = 6,
    the angle C is π/3 and the minimum value of c is 3 -/
theorem triangle_theorem (t : Triangle) 
    (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C) 
    (h2 : t.a + t.b = 6) : 
    t.C = π / 3 ∧ t.c ≥ 3 ∧ ∃ (t' : Triangle), t'.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1149_114918


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1149_114976

-- Define the hyperbola
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2^2 / b^2) - (p.1^2 / a^2) = 1}

-- Define the asymptotes
def Asymptotes (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 ∨ p.2 = -m * p.1}

theorem hyperbola_equation 
  (h : (3, 2 * Real.sqrt 2) ∈ Hyperbola 3 2) 
  (a : Asymptotes (2/3) = Asymptotes (2/3)) :
  Hyperbola 3 2 = {p : ℝ × ℝ | (p.2^2 / 4) - (p.1^2 / 9) = 1} :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1149_114976


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1149_114997

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1149_114997


namespace NUMINAMATH_CALUDE_second_group_has_ten_students_l1149_114915

/-- The number of students in the second kindergartner group -/
def second_group_size : ℕ := 10

/-- The number of students in the first kindergartner group -/
def first_group_size : ℕ := 9

/-- The number of students in the third kindergartner group -/
def third_group_size : ℕ := 11

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the second group has 10 students -/
theorem second_group_has_ten_students :
  second_group_size = 10 :=
by
  sorry

#check second_group_has_ten_students

end NUMINAMATH_CALUDE_second_group_has_ten_students_l1149_114915


namespace NUMINAMATH_CALUDE_house_sale_buyback_loss_l1149_114979

/-- Represents the financial outcome of a house sale and buyback transaction -/
def houseSaleBuybackOutcome (initialValue : ℝ) (profitPercentage : ℝ) (lossPercentage : ℝ) : ℝ :=
  let salePrice := initialValue * (1 + profitPercentage)
  let buybackPrice := salePrice * (1 - lossPercentage)
  buybackPrice - initialValue

/-- Theorem stating that the financial outcome for the given scenario results in a $240 loss -/
theorem house_sale_buyback_loss :
  houseSaleBuybackOutcome 12000 0.2 0.15 = -240 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_buyback_loss_l1149_114979


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1149_114986

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 96 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1149_114986


namespace NUMINAMATH_CALUDE_computer_lab_setup_l1149_114941

-- Define the cost of computers and investment range
def standard_teacher_cost : ℕ := 8000
def standard_student_cost : ℕ := 3500
def advanced_teacher_cost : ℕ := 11500
def advanced_student_cost : ℕ := 7000
def min_investment : ℕ := 200000
def max_investment : ℕ := 210000

-- Define the number of student computers in each lab
def standard_students : ℕ := 55
def advanced_students : ℕ := 27

-- Theorem stating the problem
theorem computer_lab_setup :
  (standard_teacher_cost + standard_student_cost * standard_students = 
   advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (min_investment < standard_teacher_cost + standard_student_cost * standard_students) ∧
  (standard_teacher_cost + standard_student_cost * standard_students < max_investment) ∧
  (min_investment < advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (advanced_teacher_cost + advanced_student_cost * advanced_students < max_investment) := by
  sorry

end NUMINAMATH_CALUDE_computer_lab_setup_l1149_114941


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l1149_114948

/-- The ratio of the area of an equilateral triangle to the area of a pentagon formed by
    placing the triangle atop a square (where the triangle's base equals the square's side) -/
theorem triangle_to_pentagon_area_ratio :
  let s : ℝ := 1  -- Assume unit length for simplicity
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_area := square_area + triangle_area
  triangle_area / pentagon_area = (4 * Real.sqrt 3 - 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l1149_114948


namespace NUMINAMATH_CALUDE_intersection_M_N_l1149_114975

def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1149_114975


namespace NUMINAMATH_CALUDE_boys_joined_school_l1149_114930

theorem boys_joined_school (initial_boys final_boys : ℕ) 
  (h1 : initial_boys = 214)
  (h2 : final_boys = 1124) :
  final_boys - initial_boys = 910 := by
  sorry

end NUMINAMATH_CALUDE_boys_joined_school_l1149_114930


namespace NUMINAMATH_CALUDE_seven_to_hundred_l1149_114974

theorem seven_to_hundred : (777 / 7) - (77 / 7) = 100 := by
  sorry

end NUMINAMATH_CALUDE_seven_to_hundred_l1149_114974


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1149_114933

/-- Given that z₁ = -1 + i and z₁z₂ = -2, prove that |z₂ + 2i| = √10 -/
theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + Complex.I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1149_114933


namespace NUMINAMATH_CALUDE_pythagorean_pattern_solution_for_eleven_l1149_114943

theorem pythagorean_pattern (n : ℕ) : 
  (2*n + 1)^2 + (2*n^2 + 2*n)^2 = (2*n^2 + 2*n + 1)^2 := by sorry

theorem solution_for_eleven : 
  let n : ℕ := 5
  (2*n^2 + 2*n + 1) = 61 := by sorry

end NUMINAMATH_CALUDE_pythagorean_pattern_solution_for_eleven_l1149_114943


namespace NUMINAMATH_CALUDE_snow_probability_l1149_114909

theorem snow_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1/2) 
  (h2 : p2 = 3/4) 
  (h3 : p3 = 2/3) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 23/24 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1149_114909


namespace NUMINAMATH_CALUDE_team_size_is_eight_l1149_114904

/-- The number of players in the basketball team -/
def n : ℕ := sorry

/-- The initial average height of the team in centimeters -/
def initial_average : ℝ := 190

/-- The height of the player leaving the team in centimeters -/
def height_leaving : ℝ := 197

/-- The height of the player joining the team in centimeters -/
def height_joining : ℝ := 181

/-- The new average height of the team after the player change in centimeters -/
def new_average : ℝ := 188

/-- Theorem stating that the number of players in the team is 8 -/
theorem team_size_is_eight :
  (n : ℝ) * initial_average - (height_leaving - height_joining) = n * new_average ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_team_size_is_eight_l1149_114904


namespace NUMINAMATH_CALUDE_statue_carving_l1149_114947

theorem statue_carving (initial_weight : ℝ) (first_week_cut : ℝ) (second_week_cut : ℝ) (final_weight : ℝ) :
  initial_weight = 250 →
  first_week_cut = 0.3 →
  second_week_cut = 0.2 →
  final_weight = 105 →
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut)
  let third_week_cut := (weight_after_second_week - final_weight) / weight_after_second_week
  third_week_cut = 0.25 := by
sorry

end NUMINAMATH_CALUDE_statue_carving_l1149_114947


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1149_114967

/-- Represents the number of students selected from each year in a stratified sample. -/
structure StratifiedSample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample given the total number of students and sample size. -/
def calculate_stratified_sample (total_students : ℕ) (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) (sample_size : ℕ) : StratifiedSample :=
  { first_year := (first_year * sample_size) / total_students,
    second_year := (second_year * sample_size) / total_students,
    third_year := (third_year * sample_size) / total_students }

theorem stratified_sample_theorem :
  let total_students : ℕ := 900
  let first_year : ℕ := 300
  let second_year : ℕ := 200
  let third_year : ℕ := 400
  let sample_size : ℕ := 45
  let result := calculate_stratified_sample total_students first_year second_year third_year sample_size
  result.first_year = 15 ∧ result.second_year = 10 ∧ result.third_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1149_114967


namespace NUMINAMATH_CALUDE_bigger_part_of_54_l1149_114951

theorem bigger_part_of_54 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) (h3 : x > 0) (h4 : y > 0) :
  max x y = 34 := by
sorry

end NUMINAMATH_CALUDE_bigger_part_of_54_l1149_114951


namespace NUMINAMATH_CALUDE_max_sum_is_four_l1149_114945

-- Define the system of inequalities and conditions
def system (x y : ℕ) : Prop :=
  5 * x + 10 * y ≤ 30 ∧ 2 * x - y ≤ 3

-- Theorem statement
theorem max_sum_is_four :
  ∃ (x y : ℕ), system x y ∧ x + y = 4 ∧
  ∀ (a b : ℕ), system a b → a + b ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_four_l1149_114945


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1149_114903

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -b / a) := by sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 3 * x^3 + 7 * x^2 - 12 * x - 4
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -7 / 3) := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1149_114903


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l1149_114970

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44 / 100) 
  (h3 : red_percent = 28 / 100) 
  (h4 : green_percent = 10 / 100) : 
  ℕ := by
  
  sorry

#check students_wearing_other_colors

end NUMINAMATH_CALUDE_students_wearing_other_colors_l1149_114970


namespace NUMINAMATH_CALUDE_complex_problem_l1149_114978

def complex_operation (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_problem (x y : ℂ) : 
  x = (1 - I) / (1 + I) →
  y = complex_operation (4 * I) (1 + I) (3 - x * I) (x + I) →
  y = -2 - 2 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_problem_l1149_114978


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_proof_l1149_114992

/-- The speed of a boat in still water, given stream speed and downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  25

/-- Proof that the boat's speed in still water is 25 km/hr -/
theorem boat_speed_proof 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_proof_l1149_114992


namespace NUMINAMATH_CALUDE_power_division_nineteen_l1149_114990

theorem power_division_nineteen : (19 : ℕ)^11 / (19 : ℕ)^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l1149_114990


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_minimum_l1149_114985

theorem hyperbola_focal_length_minimum (a b c : ℝ) : 
  a > 0 → b > 0 → c^2 = a^2 + b^2 → a + b - c = 2 → 2*c ≥ 4 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_minimum_l1149_114985


namespace NUMINAMATH_CALUDE_sin_squared_minus_2sin_range_l1149_114950

theorem sin_squared_minus_2sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_2sin_range_l1149_114950


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l1149_114910

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) 
  (h1 : square_area = 1600)
  (h2 : rectangle_area = 160)
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l1149_114910


namespace NUMINAMATH_CALUDE_digit_sum_reduction_count_l1149_114900

def digitSumReduction (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

def countDigits (d : ℕ) : ℕ := 
  (999999999 / 9 : ℕ) + (if d = 1 then 1 else 0)

theorem digit_sum_reduction_count :
  countDigits 1 = countDigits 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_reduction_count_l1149_114900


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l1149_114949

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n % (sumOfDigits n) = 0

def isMultipleOf7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 : 
  (isMultipleOf7 14) ∧ 
  ¬(isLucky 14) ∧ 
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ (isMultipleOf7 n) → (isLucky n) := by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l1149_114949


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l1149_114954

theorem chocolate_bars_distribution (total_bars : ℕ) (num_small_boxes : ℕ) 
  (h1 : total_bars = 504) (h2 : num_small_boxes = 18) :
  total_bars / num_small_boxes = 28 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l1149_114954


namespace NUMINAMATH_CALUDE_sqrt_two_plus_three_times_sqrt_two_minus_three_l1149_114914

theorem sqrt_two_plus_three_times_sqrt_two_minus_three : (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_three_times_sqrt_two_minus_three_l1149_114914


namespace NUMINAMATH_CALUDE_browser_tabs_l1149_114982

theorem browser_tabs (T : ℚ) : 
  (9 / 40 : ℚ) * T = 90 → T = 400 := by
  sorry

end NUMINAMATH_CALUDE_browser_tabs_l1149_114982


namespace NUMINAMATH_CALUDE_inequality_solution_l1149_114981

theorem inequality_solution (x : ℝ) : 2 * (2 * x - 1) > 3 * x - 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1149_114981


namespace NUMINAMATH_CALUDE_other_communities_count_l1149_114919

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 850)
  (h_muslim : muslim_percent = 44 / 100)
  (h_hindu : hindu_percent = 32 / 100)
  (h_sikh : sikh_percent = 10 / 100) :
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 119 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l1149_114919


namespace NUMINAMATH_CALUDE_shelby_total_stars_l1149_114937

/-- The number of gold stars Shelby earned yesterday -/
def yesterday_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def today_stars : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := yesterday_stars + today_stars

theorem shelby_total_stars : total_stars = 7 := by
  sorry

end NUMINAMATH_CALUDE_shelby_total_stars_l1149_114937


namespace NUMINAMATH_CALUDE_product_squared_l1149_114936

theorem product_squared (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by sorry

end NUMINAMATH_CALUDE_product_squared_l1149_114936


namespace NUMINAMATH_CALUDE_smallest_n_squares_average_is_square_l1149_114905

theorem smallest_n_squares_average_is_square : 
  (∀ k : ℕ, k > 1 ∧ k < 337 → ¬ (∃ m : ℕ, (k + 1) * (2 * k + 1) / 6 = m^2)) ∧ 
  (∃ m : ℕ, (337 + 1) * (2 * 337 + 1) / 6 = m^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_squares_average_is_square_l1149_114905


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l1149_114965

theorem min_value_quadratic_roots (k : ℝ) (α β : ℝ) : 
  (α ^ 2 - 2 * k * α + k + 20 = 0) →
  (β ^ 2 - 2 * k * β + k + 20 = 0) →
  (k ≤ -4 ∨ k ≥ 5) →
  (∀ k', k' ≤ -4 ∨ k' ≥ 5 → (α + 1) ^ 2 + (β + 1) ^ 2 ≥ 18) ∧
  ((α + 1) ^ 2 + (β + 1) ^ 2 = 18 ↔ k = -4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l1149_114965


namespace NUMINAMATH_CALUDE_dans_age_l1149_114996

theorem dans_age (dan_age ben_age : ℕ) : 
  ben_age = dan_age - 3 →
  ben_age + dan_age = 53 →
  dan_age = 28 := by
sorry

end NUMINAMATH_CALUDE_dans_age_l1149_114996


namespace NUMINAMATH_CALUDE_min_value_of_y_l1149_114925

-- Define a function that calculates the sum of squares of 11 consecutive integers
def sumOfSquares (x : ℤ) : ℤ := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2

-- Theorem statement
theorem min_value_of_y (y : ℤ) : (∃ x : ℤ, y^2 = sumOfSquares x) → y ≥ -11 ∧ (∃ x : ℤ, (-11)^2 = sumOfSquares x) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l1149_114925


namespace NUMINAMATH_CALUDE_complement_union_intersection_equivalence_l1149_114972

-- Define the sets U, M, and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_union_intersection_equivalence :
  ∀ x : ℝ, x ∈ (U \ M) ∪ (M ∩ N) ↔ x < 5 := by sorry

end NUMINAMATH_CALUDE_complement_union_intersection_equivalence_l1149_114972


namespace NUMINAMATH_CALUDE_dans_remaining_money_l1149_114960

def dans_money_left (initial_amount spent_on_candy spent_on_chocolate : ℕ) : ℕ :=
  initial_amount - (spent_on_candy + spent_on_chocolate)

theorem dans_remaining_money :
  dans_money_left 7 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l1149_114960


namespace NUMINAMATH_CALUDE_min_balloons_required_l1149_114938

/-- Represents a balloon color -/
inductive Color
| A | B | C | D | E

/-- Represents a row of balloons -/
def BalloonRow := List Color

/-- Checks if two colors are adjacent in a balloon row -/
def areAdjacent (row : BalloonRow) (c1 c2 : Color) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a balloon row -/
def allPairsAdjacent (row : BalloonRow) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areAdjacent row c1 c2

/-- The main theorem: minimum number of balloons required is 11 -/
theorem min_balloons_required :
  ∀ row : BalloonRow,
    allPairsAdjacent row →
    row.length ≥ 11 ∧
    (∃ row' : BalloonRow, allPairsAdjacent row' ∧ row'.length = 11) :=
by sorry

end NUMINAMATH_CALUDE_min_balloons_required_l1149_114938


namespace NUMINAMATH_CALUDE_fifth_result_proof_l1149_114916

theorem fifth_result_proof (total_average : ℚ) (first_five_average : ℚ) (last_seven_average : ℚ) 
  (h1 : total_average = 42)
  (h2 : first_five_average = 49)
  (h3 : last_seven_average = 52) :
  ∃ (fifth_result : ℚ), fifth_result = 147 ∧ 
    (5 * first_five_average + 7 * last_seven_average - fifth_result) / 11 = total_average := by
  sorry

end NUMINAMATH_CALUDE_fifth_result_proof_l1149_114916


namespace NUMINAMATH_CALUDE_clock_hands_90_degree_times_l1149_114940

/-- The angle (in degrees) that the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- The angle (in degrees) that the hour hand moves per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The relative speed (in degrees per minute) at which the minute hand moves compared to the hour hand -/
def relative_speed : ℚ := minute_hand_speed - hour_hand_speed

/-- The time (in minutes) when the clock hands first form a 90° angle after 12:00 -/
def first_90_degree_time : ℚ := 90 / relative_speed

/-- The time (in minutes) when the clock hands form a 90° angle for the second time after 12:00 -/
def second_90_degree_time : ℚ := 270 / relative_speed

theorem clock_hands_90_degree_times :
  (first_90_degree_time = 180/11) ∧ 
  (second_90_degree_time = 540/11) := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_90_degree_times_l1149_114940


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1149_114924

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1149_114924


namespace NUMINAMATH_CALUDE_power_function_properties_l1149_114952

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

theorem power_function_properties :
  ∃ (m : ℝ), ∀ (x : ℝ), f m x = x^2 ∧
  ∀ (k : ℝ),
    (∀ (x : ℝ), x < 2 ∨ x > k → f m x > (k + 2) * x - 2 * k) ∧
    (k = 2 → ∀ (x : ℝ), x ≠ 2 → f m x > (k + 2) * x - 2 * k) ∧
    (k < 2 → ∀ (x : ℝ), x < k ∨ x > 2 → f m x > (k + 2) * x - 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l1149_114952


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l1149_114923

/-- Given 5 consecutive points on a line, prove the length of the last segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Points represented as real numbers
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensures points are consecutive
  (h_bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  (h_ae : e - a = 22) -- ae = 22
  : e - d = 8 := by sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l1149_114923


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l1149_114961

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) →  -- x is the arithmetic mean of a and b
  (2 * x^2 = a^2 - b^2) →  -- x² is the arithmetic mean of a² and -b²
  (a = -b ∨ a = 3*b) :=  -- The relationship between a and b
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l1149_114961


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l1149_114989

def p (x : ℝ) : Prop := |x + 1| ≤ 4

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l1149_114989


namespace NUMINAMATH_CALUDE_prob_one_common_is_two_thirds_l1149_114988

/-- The number of elective courses available -/
def num_courses : ℕ := 4

/-- The number of courses each student selects -/
def courses_per_student : ℕ := 2

/-- The total number of ways two students can select their courses -/
def total_selections : ℕ := (num_courses.choose courses_per_student) ^ 2

/-- The number of ways two students can select courses with exactly one in common -/
def one_common_selection : ℕ := num_courses * (num_courses - 1) * (num_courses - 2)

/-- The probability of two students sharing exactly one course in common -/
def prob_one_common : ℚ := one_common_selection / total_selections

theorem prob_one_common_is_two_thirds : prob_one_common = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_common_is_two_thirds_l1149_114988


namespace NUMINAMATH_CALUDE_conical_cylinder_volume_l1149_114901

/-- The volume of a conical cylinder with base radius 3 cm and slant height 5 cm is 12π cm³ -/
theorem conical_cylinder_volume : 
  ∀ (r h s : ℝ), 
  r = 3 → s = 5 → h^2 + r^2 = s^2 →
  (1/3) * π * r^2 * h = 12 * π := by
sorry

end NUMINAMATH_CALUDE_conical_cylinder_volume_l1149_114901


namespace NUMINAMATH_CALUDE_liam_and_sisters_ages_l1149_114977

theorem liam_and_sisters_ages (a b : ℕ+) (h1 : a < b) (h2 : a * b * b = 72) : 
  a + b + b = 14 := by
sorry

end NUMINAMATH_CALUDE_liam_and_sisters_ages_l1149_114977


namespace NUMINAMATH_CALUDE_order_of_numbers_l1149_114994

theorem order_of_numbers : 70.3 > 0.37 ∧ 0.37 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1149_114994


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l1149_114968

/-- Represents a student with their skills -/
structure Student where
  hasExcellentEnglish : Bool
  hasStrongComputer : Bool

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of students with excellent English scores -/
def excellentEnglishCount : Nat := 2

/-- The number of students with strong computer skills -/
def strongComputerCount : Nat := 3

/-- The number of students to be allocated to each company -/
def studentsPerCompany : Nat := 4

/-- Calculates the number of valid allocation schemes -/
def countAllocationSchemes (students : List Student) : Nat :=
  sorry

/-- Theorem stating the number of valid allocation schemes -/
theorem allocation_schemes_count :
  ∀ (students : List Student),
    students.length = totalStudents →
    (students.filter (·.hasExcellentEnglish)).length = excellentEnglishCount →
    (students.filter (·.hasStrongComputer)).length = strongComputerCount →
    countAllocationSchemes students = 36 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l1149_114968


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l1149_114999

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l1149_114999
