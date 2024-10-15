import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3029_302989

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 3*x^2 + 1)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (15^2) + (5^2) = 375 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3029_302989


namespace NUMINAMATH_CALUDE_squares_ending_in_nine_l3029_302977

theorem squares_ending_in_nine (x : ℤ) :
  (x ^ 2) % 10 = 9 ↔ ∃ a : ℤ, (x = 10 * a + 3 ∨ x = 10 * a + 7) :=
by sorry

end NUMINAMATH_CALUDE_squares_ending_in_nine_l3029_302977


namespace NUMINAMATH_CALUDE_x_intercept_of_line_A_l3029_302995

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem: The x-intercept of line A is 2 -/
theorem x_intercept_of_line_A (lineA lineB : Line) (intersection : IntersectionPoint) :
  lineA.slope = -1 →
  lineB.slope = 5 →
  lineB.yIntercept = -10 →
  intersection.x + intersection.y = 2 →
  lineA.yIntercept - lineA.slope * intersection.x = lineB.slope * intersection.x + lineB.yIntercept →
  lineA.yIntercept = 2 →
  -lineA.slope * 2 + lineA.yIntercept = 0 := by
  sorry

#check x_intercept_of_line_A

end NUMINAMATH_CALUDE_x_intercept_of_line_A_l3029_302995


namespace NUMINAMATH_CALUDE_quarter_percent_of_200_l3029_302965

theorem quarter_percent_of_200 : (1 / 4 : ℚ) / 100 * 200 = (1 / 2 : ℚ) := by sorry

#eval (1 / 4 : ℚ) / 100 * 200

end NUMINAMATH_CALUDE_quarter_percent_of_200_l3029_302965


namespace NUMINAMATH_CALUDE_mary_remaining_stickers_l3029_302908

/-- Calculates the number of remaining stickers after Mary uses some on her journal. -/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (per_other_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * per_other_page)

/-- Proves that Mary has 44 stickers remaining after using some on her journal. -/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_stickers_l3029_302908


namespace NUMINAMATH_CALUDE_inequality_solution_set_sqrt_sum_inequality_l3029_302966

-- Part I
theorem inequality_solution_set (x : ℝ) :
  (|x - 5| - |2*x + 3| ≥ 1) ↔ (-7 ≤ x ∧ x ≤ 1/3) := by sorry

-- Part II
theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1/2) :
  Real.sqrt a + Real.sqrt b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_sqrt_sum_inequality_l3029_302966


namespace NUMINAMATH_CALUDE_smallest_prime_with_composite_reverse_l3029_302909

/-- A function that reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem -/
theorem smallest_prime_with_composite_reverse :
  ∃ (p : ℕ),
    isPrime p ∧
    p ≥ 10 ∧
    p < 100 ∧
    p / 10 = 3 ∧
    isComposite (reverseDigits p) ∧
    (∀ q : ℕ, isPrime q → q ≥ 10 → q < 100 → q / 10 = 3 →
      isComposite (reverseDigits q) → p ≤ q) ∧
    p = 23 :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_composite_reverse_l3029_302909


namespace NUMINAMATH_CALUDE_expression_value_l3029_302978

theorem expression_value (a b c : ℤ) (h1 : a = 12) (h2 : b = 2) (h3 : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3029_302978


namespace NUMINAMATH_CALUDE_ramanujan_hardy_game_l3029_302913

theorem ramanujan_hardy_game (h r : ℂ) : 
  h * r = 32 - 8 * I ∧ h = 5 + 3 * I → r = 4 - 4 * I := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_game_l3029_302913


namespace NUMINAMATH_CALUDE_sector_central_angle_l3029_302987

theorem sector_central_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  l + 2 * r = 6 →
  (1 / 2) * l * r = 2 →
  α = l / r →
  α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3029_302987


namespace NUMINAMATH_CALUDE_division_problem_l3029_302985

theorem division_problem (a b c : ℚ) : 
  a = (2 : ℚ) / 3 * (b + c) →
  b = (6 : ℚ) / 9 * (a + c) →
  a = 200 →
  a + b + c = 500 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3029_302985


namespace NUMINAMATH_CALUDE_vector_colinearity_l3029_302979

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 0)
def c : ℝ × ℝ := (2, 1)

theorem vector_colinearity (k : ℝ) :
  (∃ t : ℝ, t ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = (t * c.1, t * c.2)) →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_colinearity_l3029_302979


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l3029_302947

-- Define the circles
def circle1_center : ℝ × ℝ := (1, 3)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (10, 6)
def circle2_radius : ℝ := 7

-- Define the tangent line equation
def tangent_line (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- State the theorem
theorem external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧
  (∀ (x : ℝ), tangent_line m b x = m * x + b) ∧
  b = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l3029_302947


namespace NUMINAMATH_CALUDE_simplify_expressions_l3029_302971

variable (x y a : ℝ)

theorem simplify_expressions :
  (5 * x - 3 * (2 * x - 3 * y) + x = 9 * y) ∧
  (3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3029_302971


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_prove_area_of_triangle_PQR_l3029_302938

/-- Given two lines intersecting at point P(2,8), where one line has a slope of 3
    and the other has a slope of -1, and Q and R are the x-intercepts of these lines respectively,
    the area of triangle PQR is 128/3. -/
theorem area_of_triangle_PQR : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (2, 8)
    let slope1 : ℝ := 3
    let slope2 : ℝ := -1
    let line1 := fun x => slope1 * (x - P.1) + P.2
    let line2 := fun x => slope2 * (x - P.1) + P.2
    let Q : ℝ × ℝ := (-(line1 0) / slope1, 0)
    let R : ℝ × ℝ := (-(line2 0) / slope2, 0)
    area = 128 / 3 ∧
    area = (1 / 2) * (R.1 - Q.1) * P.2

/-- Proof of the theorem -/
theorem prove_area_of_triangle_PQR : area_of_triangle_PQR (128 / 3) := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_prove_area_of_triangle_PQR_l3029_302938


namespace NUMINAMATH_CALUDE_inequality_solution_l3029_302968

theorem inequality_solution (x : ℕ+) : 
  (12 * x + 5 < 10 * x + 15) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3029_302968


namespace NUMINAMATH_CALUDE_pats_stick_is_30_inches_l3029_302944

/-- The length of Pat's stick in inches -/
def pats_stick_length : ℝ := 30

/-- The length of the portion of Pat's stick covered in dirt, in inches -/
def covered_portion : ℝ := 7

/-- The length of Sarah's stick in inches -/
def sarahs_stick_length : ℝ := 46

/-- The length of Jane's stick in inches -/
def janes_stick_length : ℝ := 22

/-- Proves that Pat's stick is 30 inches long given the conditions -/
theorem pats_stick_is_30_inches :
  (pats_stick_length = covered_portion + (sarahs_stick_length / 2)) ∧
  (janes_stick_length = sarahs_stick_length - 24) ∧
  (janes_stick_length = 22) →
  pats_stick_length = 30 := by sorry

end NUMINAMATH_CALUDE_pats_stick_is_30_inches_l3029_302944


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l3029_302902

-- Define complex numbers z1 and z2
def z1 : ℂ := 2 + Complex.I
def z2 : ℂ := 1 - Complex.I

-- Define the product z
def z : ℂ := z1 * z2

-- Theorem statement
theorem product_in_fourth_quadrant :
  z.re > 0 ∧ z.im < 0 :=
sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l3029_302902


namespace NUMINAMATH_CALUDE_batsman_average_l3029_302915

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 16 * previous_average ∧
  (previous_total + 65 : ℚ) / 17 = previous_average + 3 →
  (previous_total + 65 : ℚ) / 17 = 17 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l3029_302915


namespace NUMINAMATH_CALUDE_handshake_theorem_l3029_302936

/-- Represents a gathering of people where each person shakes hands with a fixed number of others. -/
structure Gathering where
  num_people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering. -/
def total_handshakes (g : Gathering) : ℕ :=
  g.num_people * g.handshakes_per_person / 2

/-- Theorem stating that in a gathering of 30 people where each person shakes hands with 3 others,
    the total number of handshakes is 45. -/
theorem handshake_theorem (g : Gathering) (h1 : g.num_people = 30) (h2 : g.handshakes_per_person = 3) :
  total_handshakes g = 45 := by
  sorry

#eval total_handshakes ⟨30, 3⟩

end NUMINAMATH_CALUDE_handshake_theorem_l3029_302936


namespace NUMINAMATH_CALUDE_fabian_walking_speed_l3029_302950

theorem fabian_walking_speed (initial_hours : ℕ) (additional_hours : ℕ) (total_distance : ℕ) :
  initial_hours = 3 →
  additional_hours = 3 →
  total_distance = 30 →
  (initial_hours + additional_hours) * (total_distance / (initial_hours + additional_hours)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_fabian_walking_speed_l3029_302950


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3029_302961

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3029_302961


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l3029_302949

/-- A special municipal payroll tax system -/
structure PayrollTaxSystem where
  threshold : ℝ
  taxRate : ℝ

/-- A company subject to the payroll tax system -/
structure Company where
  payroll : ℝ
  taxPaid : ℝ

/-- Theorem: Given the specific conditions, prove the tax rate is 0.2% -/
theorem tax_rate_calculation (system : PayrollTaxSystem) (company : Company) :
  system.threshold = 200000 ∧
  company.payroll = 400000 ∧
  company.taxPaid = 400 →
  system.taxRate = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l3029_302949


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l3029_302910

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x : ℝ, x^2 - x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l3029_302910


namespace NUMINAMATH_CALUDE_tangent_line_theorem_intersecting_line_theorem_l3029_302959

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 6
def tangent_line_2 (x y : ℝ) : Prop := 5*x + 12*y - 78 = 0

-- Define the intersecting line equation
def intersecting_line (x y : ℝ) : Prop := 
  ∃ k, y - 4 = k*(x - 6) ∧ (k = (4 + Real.sqrt 17)/3 ∨ k = (4 - Real.sqrt 17)/3)

theorem tangent_line_theorem :
  ∀ x y : ℝ, 
  (∃ l : ℝ → ℝ → Prop, (l x y ↔ tangent_line_1 x ∨ tangent_line_2 x y) ∧ 
    (l (point_P.1) (point_P.2) ∧ 
     ∀ a b : ℝ, circle_equation a b → (l a b → a = (point_P.1) ∧ b = (point_P.2)))) :=
sorry

theorem intersecting_line_theorem :
  ∀ x y : ℝ,
  (intersecting_line x y →
    (x = point_P.1 ∧ y = point_P.2 ∨
     (∃ a b : ℝ, circle_equation a b ∧ intersecting_line a b ∧
      ∃ c d : ℝ, circle_equation c d ∧ intersecting_line c d ∧
      (a - c)^2 + (b - d)^2 = 18))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_intersecting_line_theorem_l3029_302959


namespace NUMINAMATH_CALUDE_total_highlighters_l3029_302939

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 3)
  (h2 : yellow = 7)
  (h3 : blue = 5) :
  pink + yellow + blue = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l3029_302939


namespace NUMINAMATH_CALUDE_ABD_collinear_l3029_302935

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (m n : V)
variable (A B C D : V)

axiom m_n_not_collinear : ¬ ∃ (k : ℝ), m = k • n

axiom AB_def : B - A = m + 5 • n
axiom BC_def : C - B = -2 • m + 8 • n
axiom CD_def : D - C = 4 • m + 2 • n

theorem ABD_collinear : ∃ (k : ℝ), D - A = k • (B - A) := by sorry

end NUMINAMATH_CALUDE_ABD_collinear_l3029_302935


namespace NUMINAMATH_CALUDE_house_square_footage_l3029_302905

def house_problem (smaller_house_original : ℝ) : Prop :=
  let larger_house : ℝ := 7300
  let expansion : ℝ := 3500
  let total_after_expansion : ℝ := 16000
  (smaller_house_original + expansion + larger_house = total_after_expansion) ∧
  (smaller_house_original = 5200)

theorem house_square_footage : ∃ (x : ℝ), house_problem x :=
  sorry

end NUMINAMATH_CALUDE_house_square_footage_l3029_302905


namespace NUMINAMATH_CALUDE_x_plus_y_values_l3029_302903

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l3029_302903


namespace NUMINAMATH_CALUDE_donut_selection_count_l3029_302998

theorem donut_selection_count :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 3  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 21 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_count_l3029_302998


namespace NUMINAMATH_CALUDE_paper_cutting_impossibility_l3029_302981

theorem paper_cutting_impossibility : ¬ ∃ m : ℕ, 1 + 3 * m = 50 := by
  sorry

end NUMINAMATH_CALUDE_paper_cutting_impossibility_l3029_302981


namespace NUMINAMATH_CALUDE_rajan_share_is_2400_l3029_302916

/-- Calculates the share of profit for a partner in a business based on investments and durations. -/
def calculate_share (rajan_investment : ℕ) (rajan_duration : ℕ) 
                    (rakesh_investment : ℕ) (rakesh_duration : ℕ)
                    (mukesh_investment : ℕ) (mukesh_duration : ℕ)
                    (total_profit : ℕ) : ℕ :=
  let rajan_product := rajan_investment * rajan_duration
  let rakesh_product := rakesh_investment * rakesh_duration
  let mukesh_product := mukesh_investment * mukesh_duration
  let total_product := rajan_product + rakesh_product + mukesh_product
  (rajan_product * total_profit) / total_product

/-- Theorem stating that Rajan's share of the profit is 2400 given the specified investments and durations. -/
theorem rajan_share_is_2400 :
  calculate_share 20000 12 25000 4 15000 8 4600 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_rajan_share_is_2400_l3029_302916


namespace NUMINAMATH_CALUDE_expand_expression_l3029_302994

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) = 60 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3029_302994


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3029_302931

/-- The coefficient of x^5y^2 in the expansion of (x^2 + x + y)^5 -/
def coefficient_x5y2 : ℕ :=
  -- We don't define the actual calculation here, just the type
  30

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5y2 = binomial 5 2 * binomial 3 1 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3029_302931


namespace NUMINAMATH_CALUDE_pencil_ratio_l3029_302927

theorem pencil_ratio (jeanine_initial : ℕ) (clare : ℕ) : 
  jeanine_initial = 18 →
  clare = jeanine_initial * 2 / 3 - 3 →
  clare.gcd jeanine_initial = clare →
  clare / (clare.gcd jeanine_initial) = 1 ∧ 
  jeanine_initial / (clare.gcd jeanine_initial) = 2 := by
sorry

end NUMINAMATH_CALUDE_pencil_ratio_l3029_302927


namespace NUMINAMATH_CALUDE_car_overtake_time_l3029_302923

/-- The time it takes for a car to overtake a motorcyclist by 36 km -/
theorem car_overtake_time (v_motorcycle : ℝ) (v_car : ℝ) (head_start : ℝ) (overtake_distance : ℝ) :
  v_motorcycle = 45 →
  v_car = 60 →
  head_start = 2/3 →
  overtake_distance = 36 →
  ∃ t : ℝ, t = 4.4 ∧ 
    v_car * t = v_motorcycle * (t + head_start) + overtake_distance :=
by sorry

end NUMINAMATH_CALUDE_car_overtake_time_l3029_302923


namespace NUMINAMATH_CALUDE_charge_account_interest_l3029_302934

/-- Calculate the amount owed after one year with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given a charge of $54 with 5% simple annual interest, 
    the amount owed after one year is $56.70 -/
theorem charge_account_interest : 
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  amountOwed principal rate time = 56.70 := by
  sorry


end NUMINAMATH_CALUDE_charge_account_interest_l3029_302934


namespace NUMINAMATH_CALUDE_faster_train_speed_l3029_302945

theorem faster_train_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (slower_speed : ℝ) 
  (h1 : distance = 536) 
  (h2 : time = 4) 
  (h3 : slower_speed = 60) :
  ∃ faster_speed : ℝ, 
    faster_speed = distance / time - slower_speed ∧ 
    faster_speed = 74 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3029_302945


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l3029_302992

/-- Proves that adding 5 liters of yellow tint to a 30-liter mixture
    with 30% yellow tint results in a new mixture with 40% yellow tint -/
theorem yellow_tint_percentage
  (original_volume : ℝ)
  (original_yellow_percent : ℝ)
  (added_yellow : ℝ)
  (h1 : original_volume = 30)
  (h2 : original_yellow_percent = 30)
  (h3 : added_yellow = 5) :
  let original_yellow := original_volume * (original_yellow_percent / 100)
  let new_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  new_yellow / new_volume * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l3029_302992


namespace NUMINAMATH_CALUDE_train_speed_l3029_302984

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 400) (h2 : time = 10) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3029_302984


namespace NUMINAMATH_CALUDE_modulus_of_z_l3029_302958

theorem modulus_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3029_302958


namespace NUMINAMATH_CALUDE_sqrt_increasing_l3029_302924

/-- The square root function is increasing on the non-negative real numbers. -/
theorem sqrt_increasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → Real.sqrt x₁ < Real.sqrt x₂ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_increasing_l3029_302924


namespace NUMINAMATH_CALUDE_fifth_root_monotone_l3029_302982

theorem fifth_root_monotone (x y : ℝ) (h : x < y) : (x^(1/5) : ℝ) < (y^(1/5) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_monotone_l3029_302982


namespace NUMINAMATH_CALUDE_even_function_property_l3029_302948

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_negative : ∀ x < 0, f x = 1 + 2*x) : 
  ∀ x > 0, f x = 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l3029_302948


namespace NUMINAMATH_CALUDE_total_matches_played_l3029_302933

theorem total_matches_played (average_all : ℝ) (average_first_six : ℝ) (average_last_four : ℝ)
  (h1 : average_all = 38.9)
  (h2 : average_first_six = 41)
  (h3 : average_last_four = 35.75) :
  ∃ n : ℕ, n = 10 ∧ average_all * n = average_first_six * 6 + average_last_four * 4 :=
by sorry

end NUMINAMATH_CALUDE_total_matches_played_l3029_302933


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l3029_302932

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(2)x^2 with Q(-1) = 3, prove Q(x) = 3(1 + x + x^2) -/
theorem polynomial_uniqueness (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) 
  (h2 : Q (-1) = 3) : ∀ x, Q x = 3 * (1 + x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l3029_302932


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l3029_302911

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 360)
  (h_half_lipstick : ∃ lipstick_wearers : ℕ, 2 * lipstick_wearers = total_students)
  (h_red : ∃ red_wearers : ℕ, 4 * red_wearers = lipstick_wearers)
  (h_pink : ∃ pink_wearers : ℕ, 3 * pink_wearers = lipstick_wearers)
  (h_purple : ∃ purple_wearers : ℕ, 6 * purple_wearers = lipstick_wearers)
  (h_green : ∃ green_wearers : ℕ, 12 * green_wearers = lipstick_wearers)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers + green_wearers)) :
  blue_wearers = 30 := by
  sorry


end NUMINAMATH_CALUDE_blue_lipstick_count_l3029_302911


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3029_302974

theorem diophantine_equation_solution (a b c : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∃ (x y z : ℤ), (x, y, z) ≠ (0, 0, 0) ∧ a * x^2 + b * y^2 + c * z^2 = 0) :
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3029_302974


namespace NUMINAMATH_CALUDE_gcd_876543_765432_l3029_302991

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_876543_765432_l3029_302991


namespace NUMINAMATH_CALUDE_g_has_four_roots_l3029_302907

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then |2^x - 1| else 3 / (x - 1)

-- Define the composition function g
noncomputable def g (x : ℝ) : ℝ := f (f x) - 2

-- Theorem statement
theorem g_has_four_roots :
  ∃ (a b c d : ℝ), (∀ x : ℝ, g x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
sorry

end NUMINAMATH_CALUDE_g_has_four_roots_l3029_302907


namespace NUMINAMATH_CALUDE_lcm_24_150_l3029_302990

theorem lcm_24_150 : Nat.lcm 24 150 = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_150_l3029_302990


namespace NUMINAMATH_CALUDE_largest_x_value_l3029_302926

theorem largest_x_value (x y : ℤ) : 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 2/3 ∧ x + y = 10 →
  x ≤ 4 ∧ (∃ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 2/3 ∧ z + (10 - z) = 10 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3029_302926


namespace NUMINAMATH_CALUDE_sum_of_first_49_odd_numbers_l3029_302906

theorem sum_of_first_49_odd_numbers : 
  (Finset.range 49).sum (fun i => 2 * i + 1) = 2401 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_49_odd_numbers_l3029_302906


namespace NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3029_302937

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 8}

-- Define the region where x + y ≤ 5
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 + p.2 ≤ 5}

-- Define the measure (area) of the rectangle
noncomputable def rectangleArea : ℝ := 32

-- Define the measure (area) of the region
noncomputable def regionArea : ℝ := 12

-- Theorem statement
theorem probability_x_plus_y_le_5 :
  (regionArea / rectangleArea : ℝ) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_le_5_l3029_302937


namespace NUMINAMATH_CALUDE_special_function_at_three_l3029_302963

/-- A function satisfying f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1) ∧ f 0 = 2

/-- The value of f(3) for a special function f -/
theorem special_function_at_three (f : ℝ → ℝ) (h : special_function f) : f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_three_l3029_302963


namespace NUMINAMATH_CALUDE_equation_solutions_l3029_302904

theorem equation_solutions : 
  {x : ℝ | x^4 + (3 - x)^4 = 130} = {0, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3029_302904


namespace NUMINAMATH_CALUDE_min_value_expression_l3029_302920

theorem min_value_expression (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  (6 * w) / (3 * u + 2 * v) + (6 * u) / (2 * v + 3 * w) + (2 * v) / (u + w) ≥ 2.5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3029_302920


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3029_302925

open Real

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k : ℝ)^2 / 3^k

theorem infinite_series_sum : series_sum = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3029_302925


namespace NUMINAMATH_CALUDE_johns_allowance_spent_l3029_302955

theorem johns_allowance_spent (allowance : ℚ) (arcade_fraction : ℚ) (candy_spent : ℚ) 
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_spent = 0.9) :
  let remaining := allowance - arcade_fraction * allowance
  let toy_spent := remaining - candy_spent
  toy_spent / remaining = 1/3 := by sorry

end NUMINAMATH_CALUDE_johns_allowance_spent_l3029_302955


namespace NUMINAMATH_CALUDE_expression_factorization_l3029_302922

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5) = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3029_302922


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3029_302954

theorem absolute_value_sum (a : ℝ) (h : 1 < a ∧ a < 2) : |a - 2| + |1 - a| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3029_302954


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3029_302941

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) --/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 1) → endpoint1 = (7, -3) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 5) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3029_302941


namespace NUMINAMATH_CALUDE_secret_number_probability_l3029_302976

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  (units_digit n) % 3 = 0 ∧
  n > 75

theorem secret_number_probability :
  ∃! (valid_numbers : Finset ℕ),
    (∀ n, n ∈ valid_numbers ↔ satisfies_conditions n) ∧
    valid_numbers.card = 3 :=
sorry

end NUMINAMATH_CALUDE_secret_number_probability_l3029_302976


namespace NUMINAMATH_CALUDE_distance_to_grandma_is_100_l3029_302918

/-- Represents the efficiency of a car in miles per gallon -/
def car_efficiency : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons -/
def gas_needed : ℝ := 5

/-- Calculates the distance to Grandma's house based on car efficiency and gas needed -/
def distance_to_grandma : ℝ := car_efficiency * gas_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles -/
theorem distance_to_grandma_is_100 : distance_to_grandma = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_grandma_is_100_l3029_302918


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3029_302900

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  b = 50 * Real.sqrt 6 →
  c = 150 →
  C = π / 3 →
  b / Real.sin B = c / Real.sin C →
  B < C →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3029_302900


namespace NUMINAMATH_CALUDE_peytons_children_l3029_302999

/-- The number of juice boxes each child uses per week -/
def juice_boxes_per_week : ℕ := 5

/-- The number of weeks in the school year -/
def school_year_weeks : ℕ := 25

/-- The total number of juice boxes needed for all children for the entire school year -/
def total_juice_boxes : ℕ := 375

/-- Peyton's number of children -/
def num_children : ℕ := total_juice_boxes / (juice_boxes_per_week * school_year_weeks)

theorem peytons_children :
  num_children = 3 :=
sorry

end NUMINAMATH_CALUDE_peytons_children_l3029_302999


namespace NUMINAMATH_CALUDE_a_plus_b_values_l3029_302960

theorem a_plus_b_values (a b : ℝ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l3029_302960


namespace NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l3029_302993

theorem harmonic_mean_of_2_3_6 (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  3 / (1/a + 1/b + 1/c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l3029_302993


namespace NUMINAMATH_CALUDE_fountain_area_l3029_302969

-- Define the fountain
structure Fountain :=
  (ab : ℝ)  -- Length of AB
  (dc : ℝ)  -- Length of DC
  (h_ab_positive : ab > 0)
  (h_dc_positive : dc > 0)
  (h_d_midpoint : True)  -- Represents that D is the midpoint of AB
  (h_c_center : True)    -- Represents that C is the center of the fountain

-- Define the theorem
theorem fountain_area (f : Fountain) (h_ab : f.ab = 20) (h_dc : f.dc = 12) : 
  (π * (f.ab / 2) ^ 2 + π * f.dc ^ 2) = 244 * π := by
  sorry


end NUMINAMATH_CALUDE_fountain_area_l3029_302969


namespace NUMINAMATH_CALUDE_triangle_area_l3029_302951

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 ∧
  -- Angle Q = 60°
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ∧
  -- Angle R = 30°
  4 * ((P.1 - R.1)^2 + (P.2 - R.2)^2) = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 ∧
  -- QR = 12
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 144

-- Theorem statement
theorem triangle_area (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let area := abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2
  area = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3029_302951


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3029_302983

theorem committee_selection_ways : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3029_302983


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l3029_302986

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ+), 
  ∃ (k : ℕ+), 
  15 * k = 15 * (x + 7 * c) ∧ 
  ∀ (d : ℕ+), (∀ (y z : ℕ+), ∃ (m : ℕ+), d * m = 15 * (y + 7 * z)) → d ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l3029_302986


namespace NUMINAMATH_CALUDE_tangent_angle_cosine_at_e_l3029_302997

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_angle_cosine_at_e :
  let θ := Real.arctan (deriv f e)
  Real.cos θ = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_tangent_angle_cosine_at_e_l3029_302997


namespace NUMINAMATH_CALUDE_rectangle_diagonal_shortcut_l3029_302952

theorem rectangle_diagonal_shortcut (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≤ y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x/y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_shortcut_l3029_302952


namespace NUMINAMATH_CALUDE_expression_value_l3029_302970

theorem expression_value (a b c : ℤ) : 
  (-a = 2) → (abs b = 6) → (-c + b = -10) → (8 - a + b - c = 0) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3029_302970


namespace NUMINAMATH_CALUDE_eulers_formula_l3029_302967

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  V : Type u  -- Vertex type
  E : Type v  -- Edge type
  F : Type w  -- Face type
  vertex_count : Nat
  edge_count : Nat
  face_count : Nat
  is_connected : Bool

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) :
  G.is_connected → G.vertex_count - G.edge_count + G.face_count = 2 := by
  sorry

#check eulers_formula

end NUMINAMATH_CALUDE_eulers_formula_l3029_302967


namespace NUMINAMATH_CALUDE_wang_heng_birth_date_l3029_302972

theorem wang_heng_birth_date :
  ∃! (year month : ℕ),
    1901 ≤ year ∧ year ≤ 2000 ∧
    1 ≤ month ∧ month ≤ 12 ∧
    (month * 2 + 5) * 50 + year - 250 = 2088 ∧
    year = 1988 ∧
    month = 1 := by
  sorry

end NUMINAMATH_CALUDE_wang_heng_birth_date_l3029_302972


namespace NUMINAMATH_CALUDE_triangle_area_is_36_sqrt_21_l3029_302930

/-- Triangle with an incircle that trisects a median -/
structure TriangleWithTrisectingIncircle where
  /-- Side length QR -/
  qr : ℝ
  /-- Radius of the incircle -/
  r : ℝ
  /-- Length of the median PS -/
  ps : ℝ
  /-- The incircle evenly trisects the median PS -/
  trisects_median : ps = 3 * r
  /-- QR equals 30 -/
  qr_length : qr = 30

/-- The area of a triangle with a trisecting incircle -/
def triangle_area (t : TriangleWithTrisectingIncircle) : ℝ := sorry

/-- Theorem stating the area of the specific triangle -/
theorem triangle_area_is_36_sqrt_21 (t : TriangleWithTrisectingIncircle) :
  triangle_area t = 36 * Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_sqrt_21_l3029_302930


namespace NUMINAMATH_CALUDE_four_weavers_four_days_eight_mats_l3029_302957

/-- The rate at which mat-weavers work, in mats per weaver per day -/
def weaving_rate (mats : ℕ) (weavers : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / (weavers * days)

/-- The number of mats that can be woven given a number of weavers, days, and a weaving rate -/
def mats_woven (weavers : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (weavers : ℚ) * days * rate

theorem four_weavers_four_days_eight_mats 
  (h : weaving_rate 16 8 8 = weaving_rate 8 4 4) : 
  mats_woven 4 4 (weaving_rate 16 8 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_weavers_four_days_eight_mats_l3029_302957


namespace NUMINAMATH_CALUDE_triangle_squares_area_l3029_302919

theorem triangle_squares_area (y : ℝ) : 
  y > 0 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * (3 * y) * (6 * y)) = 1000 →
  y = 10 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_squares_area_l3029_302919


namespace NUMINAMATH_CALUDE_work_completion_time_l3029_302980

theorem work_completion_time (x_days y_days combined_days : ℝ) 
  (hx : x_days = 15)
  (hc : combined_days = 11.25)
  (h_combined : 1 / x_days + 1 / y_days = 1 / combined_days) :
  y_days = 45 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3029_302980


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3029_302975

/-- Given a sample divided into groups, this theorem proves that when one group
    has a frequency of 36 and a rate of 0.25, the total sample size is 144. -/
theorem sample_size_calculation (n : ℕ) (f : ℕ) (r : ℚ)
  (h1 : f = 36)
  (h2 : r = 1/4)
  (h3 : r = f / n) :
  n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l3029_302975


namespace NUMINAMATH_CALUDE_range_of_a_l3029_302953

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ∀ x, (¬ p x ↔ ¬ q x a)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3029_302953


namespace NUMINAMATH_CALUDE_equation_proof_l3029_302917

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = Real.sqrt 256 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3029_302917


namespace NUMINAMATH_CALUDE_solution_set_inequality_range_of_m_l3029_302914

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 :=
sorry

-- Theorem for part 2
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - f (-x) ≤ 4 / a + 1 / b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_range_of_m_l3029_302914


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3029_302946

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  ellipse_C (Real.sqrt 2) 1 a b →
  (∃ (x : ℝ), x > 0 ∧ ellipse_C x 0 a b ∧ x^2 = 2) →
  (∀ (k : ℝ), k > 0 →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      x₂ - 1 = -x₁ ∧
      y₂ = -k - y₁) →
    k = Real.sqrt 2 / 2 ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 42 / 2) →
  a^2 = 4 ∧ b^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3029_302946


namespace NUMINAMATH_CALUDE_correct_sum_after_mistake_l3029_302929

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem correct_sum_after_mistake (original : ℕ) (mistaken : ℕ) :
  is_three_digit original →
  original % 10 = 9 →
  mistaken = original - 3 →
  mistaken + 57 = 823 →
  original + 57 = 826 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_mistake_l3029_302929


namespace NUMINAMATH_CALUDE_logarithm_system_solution_l3029_302940

theorem logarithm_system_solution :
  ∃ (x y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0) ∧
    (Real.log z / Real.log (2 * x) = 3) ∧
    (Real.log z / Real.log (5 * y) = 6) ∧
    (Real.log z / Real.log (x * y) = 2/3) ∧
    (x = 1 / (2 * Real.rpow 10 (1/3))) ∧
    (y = 1 / (5 * Real.rpow 10 (1/6))) ∧
    (z = 1/10) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_system_solution_l3029_302940


namespace NUMINAMATH_CALUDE_problem_statement_l3029_302956

theorem problem_statement :
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧
  (∀ (a : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a/2 ≥ x + 2*y + 2*z) ↔ (a ≤ 0 ∨ a ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3029_302956


namespace NUMINAMATH_CALUDE_no_common_points_range_and_max_m_l3029_302962

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * x
noncomputable def h (x : ℝ) := exp x / x

theorem no_common_points_range_and_max_m :
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → f x ≠ g a x) ∧
  (∃ m : ℝ, ∀ x : ℝ, x > 1/2 → f x + m / x < h x) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 1/2 → f x + m / x < h x) → m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_no_common_points_range_and_max_m_l3029_302962


namespace NUMINAMATH_CALUDE_root_sum_cube_theorem_l3029_302973

theorem root_sum_cube_theorem (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) →
  (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) →
  (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) →
  ((x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0) →
  (a = 9) := by
sorry

end NUMINAMATH_CALUDE_root_sum_cube_theorem_l3029_302973


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3029_302912

theorem reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3029_302912


namespace NUMINAMATH_CALUDE_square_grid_perimeter_l3029_302964

theorem square_grid_perimeter (total_area : ℝ) (h_area : total_area = 144) :
  let side_length := Real.sqrt (total_area / 4)
  let perimeter := 4 * (2 * side_length)
  perimeter = 48 := by
sorry

end NUMINAMATH_CALUDE_square_grid_perimeter_l3029_302964


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l3029_302921

/-- Proves that Anthony handled 10% more transactions than Mabel -/
theorem anthony_transaction_percentage (mabel cal jade anthony : ℕ) : 
  mabel = 90 →
  cal = (2 : ℚ) / 3 * anthony →
  jade = cal + 17 →
  jade = 83 →
  (anthony - mabel : ℚ) / mabel * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l3029_302921


namespace NUMINAMATH_CALUDE_prob_three_primes_six_dice_l3029_302988

/-- The probability of rolling a prime number on a 10-sided die -/
def prob_prime_10 : ℚ := 2 / 5

/-- The probability of not rolling a prime number on a 10-sided die -/
def prob_not_prime_10 : ℚ := 3 / 5

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

theorem prob_three_primes_six_dice : 
  (choose_3_from_6 : ℚ) * prob_prime_10^3 * prob_not_prime_10^3 = 4320 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_primes_six_dice_l3029_302988


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3029_302942

theorem quadratic_factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + d)*(x + e)) ∧
  (∀ x : ℝ, x^2 - 15*x + 54 = (x - e)*(x - f)) →
  d + e + f = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3029_302942


namespace NUMINAMATH_CALUDE_function_passes_through_point_two_two_l3029_302901

theorem function_passes_through_point_two_two 
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := fun (x : ℝ) => a^(x - 2) + 1
  f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_function_passes_through_point_two_two_l3029_302901


namespace NUMINAMATH_CALUDE_whitney_bought_two_posters_l3029_302943

/-- Represents the purchase at the school book fair -/
structure BookFairPurchase where
  initialAmount : ℕ
  posterCost : ℕ
  notebookCost : ℕ
  bookmarkCost : ℕ
  numNotebooks : ℕ
  numBookmarks : ℕ
  amountLeft : ℕ

/-- Theorem stating that Whitney bought 2 posters -/
theorem whitney_bought_two_posters (purchase : BookFairPurchase)
  (h1 : purchase.initialAmount = 40)
  (h2 : purchase.posterCost = 5)
  (h3 : purchase.notebookCost = 4)
  (h4 : purchase.bookmarkCost = 2)
  (h5 : purchase.numNotebooks = 3)
  (h6 : purchase.numBookmarks = 2)
  (h7 : purchase.amountLeft = 14) :
  ∃ (numPosters : ℕ), numPosters = 2 ∧
    purchase.initialAmount = 
      numPosters * purchase.posterCost +
      purchase.numNotebooks * purchase.notebookCost +
      purchase.numBookmarks * purchase.bookmarkCost +
      purchase.amountLeft :=
by sorry

end NUMINAMATH_CALUDE_whitney_bought_two_posters_l3029_302943


namespace NUMINAMATH_CALUDE_unique_number_exists_l3029_302928

theorem unique_number_exists : ∃! x : ℝ, x > 0 ∧ 100000 * x = 5 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3029_302928


namespace NUMINAMATH_CALUDE_triangle_proof_l3029_302996

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    and vectors m and n, prove that C = π/3 and if a^2 = 2b^2 + c^2, then tan(A) = -3√3 --/
theorem triangle_proof (a b c A B C : Real) (m n : Real × Real) :
  let m_x := 2 * Real.cos (C / 2)
  let m_y := -Real.sin C
  let n_x := Real.cos (C / 2)
  let n_y := 2 * Real.sin C
  m = (m_x, m_y) →
  n = (n_x, n_y) →
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  (C = Real.pi / 3 ∧ (a^2 = 2*b^2 + c^2 → Real.tan A = -3 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l3029_302996
