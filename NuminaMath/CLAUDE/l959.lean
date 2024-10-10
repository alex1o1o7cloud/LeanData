import Mathlib

namespace min_odd_in_A_P_l959_95981

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- Statement: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P) : 
  ∃ x : ℤ, x % 2 = 1 ∧ (x : ℝ) ∈ A_P P :=
sorry

end min_odd_in_A_P_l959_95981


namespace claire_balloons_given_away_l959_95965

/-- The number of balloons Claire gave away during the fair -/
def balloons_given_away (initial_balloons : ℕ) (floated_away : ℕ) (grabbed_from_coworker : ℕ) (final_balloons : ℕ) : ℕ :=
  initial_balloons - floated_away + grabbed_from_coworker - final_balloons

/-- Theorem stating the number of balloons Claire gave away during the fair -/
theorem claire_balloons_given_away :
  balloons_given_away 50 12 11 39 = 10 := by
  sorry

#eval balloons_given_away 50 12 11 39

end claire_balloons_given_away_l959_95965


namespace a_divides_next_squared_plus_next_plus_one_l959_95924

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 5 * a (n + 1) - a n - 1

theorem a_divides_next_squared_plus_next_plus_one :
  ∀ n : ℕ, (a n) ∣ ((a (n + 1))^2 + a (n + 1) + 1) :=
by sorry

end a_divides_next_squared_plus_next_plus_one_l959_95924


namespace doubled_to_original_ratio_l959_95985

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 51) : 
  (2 * x) / x = 2 :=
by sorry

end doubled_to_original_ratio_l959_95985


namespace vasims_share_l959_95976

/-- Represents the distribution of money among three people -/
structure Distribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ

/-- Checks if the distribution follows the given ratio -/
def is_valid_ratio (d : Distribution) : Prop :=
  11 * d.faruk = 3 * d.ranjith ∧ 5 * d.faruk = 3 * d.vasim

/-- The main theorem to prove -/
theorem vasims_share (d : Distribution) :
  is_valid_ratio d → d.ranjith - d.faruk = 2400 → d.vasim = 1500 := by
  sorry


end vasims_share_l959_95976


namespace quadratic_roots_range_l959_95938

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ < 1 →
  x₂ > 1 →
  a < -2 := by
sorry

end quadratic_roots_range_l959_95938


namespace profit_percentage_is_50_percent_l959_95930

/-- Calculates the profit percentage given the costs and selling price -/
def profit_percentage (purchase_price repair_cost transport_cost selling_price : ℕ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := selling_price - total_cost
  (profit : ℚ) / (total_cost : ℚ) * 100

/-- Theorem stating that the profit percentage for the given scenario is 50% -/
theorem profit_percentage_is_50_percent :
  profit_percentage 11000 5000 1000 25500 = 50 := by
  sorry

end profit_percentage_is_50_percent_l959_95930


namespace range_of_a_l959_95929

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end range_of_a_l959_95929


namespace diagonals_perpendicular_l959_95902

/-- Given four points A, B, C, and D in a 2D plane, prove that the diagonals of the quadrilateral ABCD are perpendicular. -/
theorem diagonals_perpendicular (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 3))
  (hB : B = (2, 6))
  (hC : C = (6, -1))
  (hD : D = (-3, -4)) : 
  (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0 := by
  sorry

#check diagonals_perpendicular

end diagonals_perpendicular_l959_95902


namespace gecko_infertile_eggs_percentage_l959_95983

theorem gecko_infertile_eggs_percentage 
  (total_eggs : ℕ) 
  (hatched_eggs : ℕ) 
  (calcification_rate : ℚ) :
  total_eggs = 30 →
  hatched_eggs = 16 →
  calcification_rate = 1/3 →
  ∃ (infertile_percentage : ℚ),
    infertile_percentage = 20/100 ∧
    hatched_eggs = (total_eggs : ℚ) * (1 - infertile_percentage) * (1 - calcification_rate) :=
by sorry

end gecko_infertile_eggs_percentage_l959_95983


namespace unique_n_divisible_by_11_l959_95943

theorem unique_n_divisible_by_11 : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end unique_n_divisible_by_11_l959_95943


namespace prove_annes_cleaning_time_l959_95952

/-- Represents the time it takes Anne to clean the house alone -/
def annes_cleaning_time : ℝ := 12

/-- Represents Bruce's cleaning rate (houses per hour) -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (houses per hour) -/
noncomputable def anne_rate : ℝ := sorry

theorem prove_annes_cleaning_time :
  -- Bruce and Anne can clean the house in 4 hours together
  (bruce_rate + anne_rate) * 4 = 1 →
  -- If Anne's speed is doubled, they can clean the house in 3 hours
  (bruce_rate + 2 * anne_rate) * 3 = 1 →
  -- Then Anne's individual cleaning time is 12 hours
  annes_cleaning_time = 1 / anne_rate :=
by sorry

end prove_annes_cleaning_time_l959_95952


namespace perfect_square_binomial_l959_95955

theorem perfect_square_binomial (c : ℚ) : 
  (∃ t u : ℚ, ∀ x : ℚ, c * x^2 + (45/2) * x + 1 = (t * x + u)^2) → 
  c = 2025/16 := by
sorry

end perfect_square_binomial_l959_95955


namespace even_quadratic_sum_l959_95948

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

/-- The main theorem -/
theorem even_quadratic_sum (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  let interval := Set.Icc (-2 * a - 5) 1
  IsEvenOn f (-2 * a - 5) 1 → a + 2 * b = -2 := by
sorry

end even_quadratic_sum_l959_95948


namespace inequality_theorem_l959_95994

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end inequality_theorem_l959_95994


namespace system_solution_ratio_l959_95928

/-- Given a system of linear equations with non-zero solutions x, y, and z,
    prove that xz/y^2 = 175 -/
theorem system_solution_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (eq1 : x + (95/3)*y + 4*z = 0)
  (eq2 : 4*x + (95/3)*y - 3*z = 0)
  (eq3 : 3*x + 5*y - 4*z = 0) :
  x*z/y^2 = 175 := by
sorry


end system_solution_ratio_l959_95928


namespace smallest_integer_with_remainders_l959_95973

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 8 = 2 → n ≤ m) ∧
  n = 170 ∧
  131 ≤ n ∧ n ≤ 170 :=
by sorry

end smallest_integer_with_remainders_l959_95973


namespace min_value_expression_min_value_achievable_l959_95968

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a ≥ 6 * 12^(1/6) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a = 6 * 12^(1/6) :=
by sorry

end min_value_expression_min_value_achievable_l959_95968


namespace ratio_fraction_equality_l959_95951

theorem ratio_fraction_equality (a b c : ℝ) (h : a ≠ 0) :
  (a : ℝ) / 2 = b / 3 ∧ a / 2 = c / 4 →
  (a - b + c) / b = 1 := by
sorry

end ratio_fraction_equality_l959_95951


namespace right_triangle_area_l959_95906

/-- The area of a right triangle with hypotenuse 12 inches and one angle 30° is 18√3 square inches -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30°
  area = h * h * Real.sin θ * Real.cos θ / 2 →  -- area formula for right triangle
  area = 18 * Real.sqrt 3 :=
by sorry

end right_triangle_area_l959_95906


namespace max_divisible_arrangement_l959_95980

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i : ℕ, i < arr.length - 1 → 
    is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i+1, by sorry⟩) ∨ 
    is_divisible (arr.get ⟨i+1, by sorry⟩) (arr.get ⟨i, by sorry⟩)

def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem max_divisible_arrangement :
  (∃ (arr : List ℕ), arr.length = 8 ∧ 
    (∀ x ∈ arr, x ∈ cards) ∧ 
    valid_arrangement arr) ∧
  (∀ (arr : List ℕ), arr.length > 8 → 
    (∀ x ∈ arr, x ∈ cards) → 
    ¬valid_arrangement arr) := by sorry

end max_divisible_arrangement_l959_95980


namespace smallest_n_for_inequality_l959_95964

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (k : ℕ), k < n → (10 : ℝ) ^ (2 ^ (k + 1)) < 1000) ∧
  (10 : ℝ) ^ (2 ^ (n + 1)) ≥ 1000 := by
  sorry

end smallest_n_for_inequality_l959_95964


namespace variance_of_white_balls_l959_95931

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := 7

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The probability of drawing a white ball -/
def p : ℚ := white_balls / total_balls

/-- X is the random variable representing the number of white balls drawn -/
def X : Type := Unit

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls :
  binomial_variance num_trials p = 12 / 7 :=
sorry

end variance_of_white_balls_l959_95931


namespace jake_watching_show_l959_95925

theorem jake_watching_show (total_show_length : ℝ) (friday_watch_time : ℝ)
  (monday_fraction : ℝ) (tuesday_watch_time : ℝ) (thursday_fraction : ℝ) :
  total_show_length = 52 →
  friday_watch_time = 19 →
  monday_fraction = 1/2 →
  tuesday_watch_time = 4 →
  thursday_fraction = 1/2 →
  ∃ (wednesday_fraction : ℝ),
    wednesday_fraction = 1/4 ∧
    total_show_length = 
      (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24 +
       thursday_fraction * (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24)) +
      friday_watch_time :=
by
  sorry

#check jake_watching_show

end jake_watching_show_l959_95925


namespace carissa_street_crossing_l959_95907

/-- Carissa's street crossing problem -/
theorem carissa_street_crossing 
  (walking_speed : ℝ) 
  (street_width : ℝ) 
  (total_time : ℝ) 
  (n : ℝ) 
  (h1 : walking_speed = 2) 
  (h2 : street_width = 260) 
  (h3 : total_time = 30) 
  (h4 : n > 0) :
  let running_speed := n * walking_speed
  let walking_time := total_time / (1 + n)
  let running_time := n * walking_time
  walking_speed * walking_time + running_speed * running_time = street_width →
  running_speed = 10 := by sorry

end carissa_street_crossing_l959_95907


namespace hannahs_remaining_money_l959_95939

/-- The problem of calculating Hannah's remaining money after selling cookies and cupcakes and buying measuring spoons. -/
theorem hannahs_remaining_money :
  let cookie_count : ℕ := 40
  let cookie_price : ℚ := 4/5  -- $0.8 expressed as a rational number
  let cupcake_count : ℕ := 30
  let cupcake_price : ℚ := 2
  let spoon_set_count : ℕ := 2
  let spoon_set_price : ℚ := 13/2  -- $6.5 expressed as a rational number
  
  let total_sales := cookie_count * cookie_price + cupcake_count * cupcake_price
  let total_spent := spoon_set_count * spoon_set_price
  let remaining_money := total_sales - total_spent

  remaining_money = 79 := by
  sorry

end hannahs_remaining_money_l959_95939


namespace vector_magnitude_l959_95971

/-- Given two vectors a and b in a 2D space, if the angle between them is 120°,
    |a| = 2, and |a + b| = √7, then |b| = 3. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  let θ := Real.arccos (-1/2)  -- 120° in radians
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos θ →
  Real.sqrt (a.1^2 + a.2^2) = 2 →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 →
  Real.sqrt (b.1^2 + b.2^2) = 3 := by
sorry

end vector_magnitude_l959_95971


namespace count_three_digit_divisible_by_nine_l959_95913

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_three_digit_divisible_by_nine :
  let min_num : ℕ := 108
  let max_num : ℕ := 999
  let common_diff : ℕ := 9
  (∀ n, is_three_digit n ∧ n % 9 = 0 → min_num ≤ n ∧ n ≤ max_num) →
  (∀ n, min_num ≤ n ∧ n ≤ max_num ∧ n % 9 = 0 → is_three_digit n) →
  (∀ n m, min_num ≤ n ∧ n < m ∧ m ≤ max_num ∧ n % 9 = 0 ∧ m % 9 = 0 → m - n = common_diff) →
  (Finset.filter (λ n => n % 9 = 0) (Finset.range (max_num - min_num + 1))).card + 1 = 100 :=
by sorry

end count_three_digit_divisible_by_nine_l959_95913


namespace smallest_n_with_shared_digit_arrangement_l959_95986

/-- A function that checks if two natural numbers share a digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (d ∈ a.digits 10) ∧ (d ∈ b.digits 10)

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def valid_arrangement (lst : List ℕ) : Prop :=
  ∀ i : ℕ, i < lst.length → share_digit (lst.get! i) (lst.get! ((i + 1) % lst.length))

/-- The main theorem stating that 29 is the smallest N satisfying the conditions -/
theorem smallest_n_with_shared_digit_arrangement :
  ∀ N : ℕ, N ≥ 2 →
  (∃ lst : List ℕ, lst.length = N ∧ lst.toFinset = Finset.range N.succ ∧ valid_arrangement lst) →
  N ≥ 29 :=
sorry

end smallest_n_with_shared_digit_arrangement_l959_95986


namespace remainder_of_valid_polynomials_l959_95914

/-- The number of elements in the tuple -/
def tuple_size : ℕ := 2011

/-- The upper bound for each element in the tuple -/
def upper_bound : ℕ := 2011^2

/-- The degree of the polynomial -/
def poly_degree : ℕ := 4019

/-- The modulus for the divisibility conditions -/
def modulus : ℕ := 2011^2

/-- The final modulus for the remainder -/
def final_modulus : ℕ := 1000

/-- The expected remainder -/
def expected_remainder : ℕ := 281

/-- A function representing the conditions on the polynomial -/
def valid_polynomial (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, ∃ k : ℤ, f n = k) ∧
  (∀ i : ℕ, i ≤ tuple_size → ∃ k : ℤ, f i - k = modulus * (f i / modulus)) ∧
  (∀ n : ℤ, ∃ k : ℤ, f (n + tuple_size) - f n = modulus * k)

/-- The main theorem -/
theorem remainder_of_valid_polynomials :
  (upper_bound ^ (poly_degree + 1)) % final_modulus = expected_remainder := by
  sorry

end remainder_of_valid_polynomials_l959_95914


namespace range_of_a_l959_95997

/-- The set A defined by the equation x^2 + 4x = 0 -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B defined by the equation x^2 + ax + a = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ 0 ≤ a ∧ a < 4 := by sorry

end range_of_a_l959_95997


namespace range_of_m_l959_95940

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 4 < 0 → (x - m)*(x - m - 3) > 0) ∧ 
  (∃ x : ℝ, (x - m)*(x - m - 3) > 0 ∧ x^2 + 3*x - 4 ≥ 0) →
  m ≤ -7 ∨ m ≥ 1 :=
sorry

end range_of_m_l959_95940


namespace gathering_attendance_l959_95947

theorem gathering_attendance (wine soda both : ℕ) 
  (h1 : wine = 26) 
  (h2 : soda = 22) 
  (h3 : both = 17) : 
  wine + soda - both = 31 := by
  sorry

end gathering_attendance_l959_95947


namespace total_sum_is_2743_l959_95979

/-- The total sum lent, given the conditions of the problem -/
def total_sum_lent : ℕ := 2743

/-- The second part of the sum -/
def second_part : ℕ := 1688

/-- Calculates the interest for a given principal, rate, and time -/
def interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

/-- Theorem stating that the total sum lent is 2743 -/
theorem total_sum_is_2743 :
  ∃ (first_part : ℕ),
    interest first_part (3/100) 8 = interest second_part (5/100) 3 ∧
    first_part + second_part = total_sum_lent :=
by sorry

end total_sum_is_2743_l959_95979


namespace parallel_line_theorem_perpendicular_line_theorem_l959_95942

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + 5 = 0}

-- Define the point that line l passes through
def point : ℝ × ℝ := (1, -4)

-- Define parallel line
def parallel_line (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + m = 0}

-- Define perpendicular line
def perpendicular_line (n : ℝ) : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - n = 0}

-- Theorem for parallel case
theorem parallel_line_theorem :
  ∃ m : ℝ, point ∈ parallel_line m ∧ m = 10 :=
sorry

-- Theorem for perpendicular case
theorem perpendicular_line_theorem :
  ∃ n : ℝ, point ∈ perpendicular_line n ∧ n = 11 :=
sorry

end parallel_line_theorem_perpendicular_line_theorem_l959_95942


namespace unique_tangent_implies_radius_l959_95996

/-- A circle in the x-y plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the x-y plane -/
def Point := ℝ × ℝ

/-- The number of tangent lines from a point to a circle -/
def numTangentLines (c : Circle) (p : Point) : ℕ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem unique_tangent_implies_radius (c : Circle) (p : Point) :
  c.center = (3, -1) →
  p = (-2, 1) →
  numTangentLines c p = 1 →
  c.radius = Real.sqrt 29 := by sorry

end unique_tangent_implies_radius_l959_95996


namespace crazy_silly_school_books_left_l959_95959

/-- Given a series with a total number of books and a number of books read,
    calculate the number of books left to read. -/
def booksLeftToRead (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series with 19 books, if 4 books have been read,
    then the number of books left to read is 15. -/
theorem crazy_silly_school_books_left :
  booksLeftToRead 19 4 = 15 := by
  sorry

end crazy_silly_school_books_left_l959_95959


namespace sphere_volume_circumscribing_cube_l959_95912

/-- The volume of a sphere that circumscribes a cube with side length 2 is 4√3π. -/
theorem sphere_volume_circumscribing_cube (cube_side : ℝ) (sphere_volume : ℝ) : 
  cube_side = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3 * cube_side / 2)^3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end sphere_volume_circumscribing_cube_l959_95912


namespace qr_distance_l959_95905

/-- Right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  right_angle : DE^2 + EF^2 = DF^2

/-- Circle centered at Q tangent to DE at D and passing through F -/
structure CircleQ where
  Q : ℝ × ℝ
  tangent_DE : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- Circle centered at R tangent to EF at E and passing through F -/
structure CircleR where
  R : ℝ × ℝ
  tangent_EF : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- The main theorem statement -/
theorem qr_distance (t : RightTriangle) (cq : CircleQ) (cr : CircleR) 
  (h1 : t.DE = 5) (h2 : t.EF = 12) (h3 : t.DF = 13) :
  Real.sqrt ((cq.Q.1 - cr.R.1)^2 + (cq.Q.2 - cr.R.2)^2) = 13.54 := by
  sorry

end qr_distance_l959_95905


namespace sector_arc_length_l959_95970

/-- Given a sector with circumference 8 and central angle 2 radians, 
    the length of its arc is 4 -/
theorem sector_arc_length (c : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) : 
  c = 8 →  -- circumference of the sector
  θ = 2 →  -- central angle in radians
  c = l + 2 * r →  -- circumference formula for a sector
  l = r * θ →  -- arc length formula
  l = 4 := by
  sorry

end sector_arc_length_l959_95970


namespace max_value_of_sqrt_sum_l959_95982

theorem max_value_of_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end max_value_of_sqrt_sum_l959_95982


namespace second_derivative_of_cosine_at_pi_third_l959_95927

open Real

theorem second_derivative_of_cosine_at_pi_third :
  let f : ℝ → ℝ := fun x ↦ cos x
  (deriv (deriv f)) (π / 3) = -1 / 2 := by
  sorry

end second_derivative_of_cosine_at_pi_third_l959_95927


namespace sufficient_condition_for_inequality_l959_95934

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by
  sorry

end sufficient_condition_for_inequality_l959_95934


namespace card_ratio_l959_95961

/-- Proves the ratio of cards eaten by the dog to the total cards before the incident -/
theorem card_ratio (new_cards : ℕ) (remaining_cards : ℕ) : 
  new_cards = 4 → remaining_cards = 34 → 
  (new_cards + remaining_cards - remaining_cards) / (new_cards + remaining_cards) = 2 / 19 := by
  sorry

end card_ratio_l959_95961


namespace point_P_coordinates_l959_95915

def M : ℝ × ℝ := (2, 2)
def N : ℝ × ℝ := (5, -2)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem point_P_coordinates :
  ∀ x : ℝ,
    let P : ℝ × ℝ := (x, 0)
    is_right_angle M P N → x = 1 ∨ x = 6 := by
  sorry

end point_P_coordinates_l959_95915


namespace fuel_tank_capacity_l959_95998

/-- Given a fuel tank partially filled with fuel A and then filled to capacity with fuel B,
    prove that the capacity of the tank is 162.5 gallons. -/
theorem fuel_tank_capacity
  (capacity : ℝ)
  (fuel_a_volume : ℝ)
  (fuel_a_ethanol_percent : ℝ)
  (fuel_b_ethanol_percent : ℝ)
  (total_ethanol : ℝ)
  (h1 : fuel_a_volume = 49.99999999999999)
  (h2 : fuel_a_ethanol_percent = 0.12)
  (h3 : fuel_b_ethanol_percent = 0.16)
  (h4 : total_ethanol = 30)
  (h5 : fuel_a_ethanol_percent * fuel_a_volume +
        fuel_b_ethanol_percent * (capacity - fuel_a_volume) = total_ethanol) :
  capacity = 162.5 := by
  sorry

end fuel_tank_capacity_l959_95998


namespace camel_cost_l959_95920

/-- Proves that the cost of one camel is 5600 given the specified conditions --/
theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 140000) : 
  camel 1 = 5600 := by
  sorry

end camel_cost_l959_95920


namespace uv_length_in_triangle_l959_95919

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real)
  (xz_length : Real)
  (yz_length : Real)

-- Define the angle bisector points S and T
structure AngleBisectorPoints :=
  (S : ℝ × ℝ)
  (T : ℝ × ℝ)

-- Define the perpendicular feet U and V
structure PerpendicularFeet :=
  (U : ℝ × ℝ)
  (V : ℝ × ℝ)

-- Define the theorem
theorem uv_length_in_triangle (t : Triangle) (ab : AngleBisectorPoints) (pf : PerpendicularFeet) :
  t.xy_length = 140 ∧ t.xz_length = 130 ∧ t.yz_length = 150 →
  -- S is on the angle bisector of angle X and YZ
  -- T is on the angle bisector of angle Y and XZ
  -- U is the foot of the perpendicular from Z to YT
  -- V is the foot of the perpendicular from Z to XS
  Real.sqrt ((pf.U.1 - pf.V.1)^2 + (pf.U.2 - pf.V.2)^2) = 70 := by
  sorry

end uv_length_in_triangle_l959_95919


namespace systematic_sample_sum_l959_95918

/-- Systematic sampling function that returns the nth element in a sample of size k from a population of size n -/
def systematicSample (n k : ℕ) (i : ℕ) : ℕ :=
  i * (n / k) + 1

theorem systematic_sample_sum (a b : ℕ) :
  systematicSample 60 5 0 = 4 ∧
  systematicSample 60 5 1 = a ∧
  systematicSample 60 5 2 = 28 ∧
  systematicSample 60 5 3 = b ∧
  systematicSample 60 5 4 = 52 →
  a + b = 56 := by
sorry

end systematic_sample_sum_l959_95918


namespace common_chord_equation_l959_95936

/-- The equation of the common chord of two circles -/
def common_chord (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => c1 p ∧ c2 p

/-- First circle equation -/
def circle1 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 + 2*x = 0

/-- Second circle equation -/
def circle2 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 - 4*y = 0

/-- The proposed common chord equation -/
def proposed_chord : ℝ × ℝ → Prop :=
  fun (x, y) => x + 2*y = 0

theorem common_chord_equation :
  common_chord circle1 circle2 = proposed_chord := by
  sorry

end common_chord_equation_l959_95936


namespace heights_sum_l959_95922

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem heights_sum (john lena rebeca : ℕ) 
  (h1 : john = 152)
  (h2 : john = lena + 15)
  (h3 : rebeca = john + 6) :
  lena + rebeca = 295 := by
  sorry

end heights_sum_l959_95922


namespace no_odd_white_columns_exists_odd_black_columns_l959_95966

/-- Represents a 3x3x3 cube composed of white and black unit cubes -/
structure Cube :=
  (white_count : Nat)
  (black_count : Nat)
  (total_count : Nat)
  (is_valid : white_count + black_count = total_count ∧ total_count = 27)

/-- Represents a column in the cube -/
structure Column :=
  (white_count : Nat)
  (black_count : Nat)
  (is_valid : white_count + black_count = 3)

/-- Checks if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Theorem: It is impossible for each column to contain an odd number of white cubes -/
theorem no_odd_white_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ¬ (∀ col : Column, is_odd col.white_count) :=
sorry

/-- Theorem: It is possible for each column to contain an odd number of black cubes -/
theorem exists_odd_black_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ∃ (arrangement : List Column), (∀ col ∈ arrangement, is_odd col.black_count) ∧ 
    arrangement.length = 27 ∧ (arrangement.map Column.black_count).sum = 13 :=
sorry

end no_odd_white_columns_exists_odd_black_columns_l959_95966


namespace reciprocal_sum_of_roots_l959_95935

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 6 * r^2 - 11 * r + 7 = 0 ∧ 
              6 * s^2 - 11 * s + 7 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = 11 / 7 := by
sorry

end reciprocal_sum_of_roots_l959_95935


namespace increase_by_percentage_l959_95990

theorem increase_by_percentage (initial : ℕ) (percentage : ℕ) (result : ℕ) : 
  initial = 150 → percentage = 40 → result = initial + (initial * percentage) / 100 → result = 210 := by
  sorry

end increase_by_percentage_l959_95990


namespace coin_container_total_l959_95958

theorem coin_container_total : ∃ (x : ℕ),
  (x * 1 + x * 3 * 10 + x * 3 * 5 * 25) = 63000 :=
by
  sorry

end coin_container_total_l959_95958


namespace systematic_sampling_smallest_number_l959_95989

theorem systematic_sampling_smallest_number 
  (total_classes : Nat) 
  (selected_classes : Nat) 
  (sum_of_selected : Nat) : 
  total_classes = 30 → 
  selected_classes = 6 → 
  sum_of_selected = 87 → 
  (total_classes / selected_classes : Nat) = 5 → 
  ∃ x : Nat, 
    x + (x + 5) + (x + 10) + (x + 15) + (x + 20) + (x + 25) = sum_of_selected ∧ 
    x = 2 ∧ 
    (∀ y : Nat, y + (y + 5) + (y + 10) + (y + 15) + (y + 20) + (y + 25) = sum_of_selected → y ≥ x) :=
by sorry

end systematic_sampling_smallest_number_l959_95989


namespace units_digit_G_3_l959_95978

-- Define G_n
def G (n : ℕ) : ℕ := 2^(2^(2^n)) + 1

-- Theorem statement
theorem units_digit_G_3 : G 3 % 10 = 7 := by
  sorry

end units_digit_G_3_l959_95978


namespace renata_final_amount_l959_95932

/-- Represents Renata's financial transactions --/
def renataTransactions : List Int :=
  [10, -4, 90, -50, -10, -5, -1, -1, 65]

/-- Calculates the final amount Renata has --/
def finalAmount (transactions : List Int) : Int :=
  transactions.sum

/-- Theorem stating that Renata ends up with $94 --/
theorem renata_final_amount :
  finalAmount renataTransactions = 94 := by
  sorry

#eval finalAmount renataTransactions

end renata_final_amount_l959_95932


namespace total_vehicles_calculation_l959_95977

theorem total_vehicles_calculation (lanes : ℕ) (trucks_per_lane : ℕ) : 
  lanes = 4 →
  trucks_per_lane = 60 →
  (lanes * trucks_per_lane + lanes * (2 * lanes * trucks_per_lane)) = 2160 := by
  sorry

end total_vehicles_calculation_l959_95977


namespace g_is_even_l959_95916

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := x^2 * f x

/-- Theorem stating that g is an even function -/
theorem g_is_even (f : ℝ → ℝ) (h : FunctionalEq f) :
  ∀ x : ℝ, g f (-x) = g f x :=
by sorry

end g_is_even_l959_95916


namespace train_speed_l959_95910

/-- The speed of two trains crossing each other -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 16) :
  let relative_speed := 2 * train_length / crossing_time
  let train_speed := relative_speed / 2
  let train_speed_kmh := train_speed * 3.6
  train_speed_kmh = 27 := by sorry

end train_speed_l959_95910


namespace dodge_trucks_count_l959_95963

/-- Represents the number of vehicles of each type in the parking lot -/
structure ParkingLot where
  dodge : ℚ
  ford : ℚ
  toyota : ℚ
  nissan : ℚ
  volkswagen : ℚ
  honda : ℚ
  mazda : ℚ
  chevrolet : ℚ
  subaru : ℚ
  fiat : ℚ

/-- The conditions of the parking lot -/
def validParkingLot (p : ParkingLot) : Prop :=
  p.ford = (1/3) * p.dodge ∧
  p.ford = 2 * p.toyota ∧
  p.toyota = (7/9) * p.nissan ∧
  p.volkswagen = (1/2) * p.toyota ∧
  p.honda = (3/4) * p.ford ∧
  p.mazda = (2/5) * p.nissan ∧
  p.chevrolet = (2/3) * p.honda ∧
  p.subaru = 4 * p.dodge ∧
  p.fiat = (1/2) * p.mazda ∧
  p.volkswagen = 5

theorem dodge_trucks_count (p : ParkingLot) (h : validParkingLot p) : p.dodge = 60 := by
  sorry

end dodge_trucks_count_l959_95963


namespace pure_imaginary_implies_a_eq_three_l959_95900

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (a^2 - 9) + (a + 3)i is a pure imaginary number, prove that a = 3. -/
theorem pure_imaginary_implies_a_eq_three (a : ℝ) 
    (h : IsPureImaginary ((a^2 - 9) + (a + 3)*I)) : a = 3 := by
  sorry

end pure_imaginary_implies_a_eq_three_l959_95900


namespace prism_volume_l959_95909

/-- A right rectangular prism with face areas 18, 32, and 48 square inches has a volume of 288 cubic inches. -/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 18) 
  (area2 : w * h = 32) 
  (area3 : l * h = 48) : 
  l * w * h = 288 := by
  sorry

end prism_volume_l959_95909


namespace triangle_side_angle_relation_l959_95926

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the theorem
theorem triangle_side_angle_relation (t : Triangle) 
  (h1 : t.α > 0 ∧ t.β > 0 ∧ t.γ > 0)
  (h2 : t.α + t.β + t.γ = Real.pi)
  (h3 : 3 * t.α + 2 * t.β = Real.pi) :
  t.a^2 + t.b * t.c - t.c^2 = 0 := by
  sorry

end triangle_side_angle_relation_l959_95926


namespace polynomial_identity_l959_95911

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x, P x - 3 * x = 5 * x^2 - 3 * x - 5) → 
  (∀ x, P x = 5 * x^2 - 5) := by
sorry

end polynomial_identity_l959_95911


namespace largest_number_l959_95941

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -|(-4)| ∧ b = 0 ∧ c = 1 ∧ d = -(-3) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c :=
by sorry

end largest_number_l959_95941


namespace triangle_side_ratio_l959_95903

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 9sin²B = 4sin²A and cosC = 1/4, then c/a = √(10)/3 -/
theorem triangle_side_ratio (a b c A B C : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 9 * (sin B)^2 = 4 * (sin A)^2) (h5 : cos C = 1/4) :
  c / a = sqrt 10 / 3 := by
  sorry

end triangle_side_ratio_l959_95903


namespace quadratic_roots_property_l959_95944

theorem quadratic_roots_property (r s : ℝ) : 
  (3 * r^2 - 5 * r - 7 = 0) → 
  (3 * s^2 - 5 * s - 7 = 0) → 
  (r ≠ s) →
  (4 * r^2 - 4 * s^2) / (r - s) = 20 / 3 := by
sorry

end quadratic_roots_property_l959_95944


namespace first_girl_productivity_higher_l959_95960

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ :=
  k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

/-- Theorem: The first girl's productivity is 5% higher than the second girl's -/
theorem first_girl_productivity_higher (girl1 girl2 : Knitter)
    (h1 : girl1.workTime = 5)
    (h2 : girl2.workTime = 7)
    (h3 : girl1.breakTime = 1)
    (h4 : girl2.breakTime = 1)
    (h5 : ∃ t : ℕ, totalWorkTime girl1 t = totalWorkTime girl2 t ∧ t > 0) :
    (21 : ℚ) / 20 = girl2.workTime / girl1.workTime := by
  sorry

end first_girl_productivity_higher_l959_95960


namespace cookie_box_calories_l959_95946

theorem cookie_box_calories (bags_per_box : ℕ) (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) 
  (h1 : bags_per_box = 6)
  (h2 : cookies_per_bag = 25)
  (h3 : calories_per_cookie = 18) :
  bags_per_box * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end cookie_box_calories_l959_95946


namespace dunk_a_clown_tickets_l959_95999

def total_tickets : ℕ := 40
def num_rides : ℕ := 3
def tickets_per_ride : ℕ := 4

theorem dunk_a_clown_tickets : 
  total_tickets - (num_rides * tickets_per_ride) = 28 := by
  sorry

end dunk_a_clown_tickets_l959_95999


namespace twice_x_greater_than_five_l959_95937

theorem twice_x_greater_than_five (x : ℝ) : (2 * x > 5) ↔ (2 * x > 5) := by sorry

end twice_x_greater_than_five_l959_95937


namespace smallest_prime_with_digit_sum_23_l959_95991

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 757 :=
by sorry

end smallest_prime_with_digit_sum_23_l959_95991


namespace crayons_in_box_l959_95995

def blue_crayons : ℕ := 3

def red_crayons : ℕ := 4 * blue_crayons

def total_crayons : ℕ := red_crayons + blue_crayons

theorem crayons_in_box : total_crayons = 15 := by
  sorry

end crayons_in_box_l959_95995


namespace not_all_right_triangles_are_isosceles_l959_95921

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  sum_angles : angleA + angleB + angleC = 180
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define a right triangle
def IsRight (t : Triangle) : Prop :=
  t.angleA = 90 ∨ t.angleB = 90 ∨ t.angleC = 90

-- The theorem to prove
theorem not_all_right_triangles_are_isosceles :
  ¬ (∀ t : Triangle, IsRight t → IsIsosceles t) :=
sorry

end not_all_right_triangles_are_isosceles_l959_95921


namespace star_two_one_l959_95993

-- Define the ∗ operation for real numbers
def star (x y : ℝ) : ℝ := x - y + x * y

-- State the theorem
theorem star_two_one : star 2 1 = 3 := by
  sorry

end star_two_one_l959_95993


namespace bisecting_line_sum_l959_95974

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → 
    (x^2 + y^2 + 2*x + 2*y - 1 = 0 → 
      ∃ (p q : ℝ), p^2 + q^2 + 2*p + 2*q - 1 = 0 ∧ 
        a * p + b * q + 1 = 0 ∧ (p, q) ≠ (x, y))

/-- Theorem: If a line ax + by + 1 = 0 bisects the circumference of 
    the circle x^2 + y^2 + 2x + 2y - 1 = 0, then a + b = 1 -/
theorem bisecting_line_sum (l : BisectingLine) : l.a + l.b = 1 := by
  sorry

end bisecting_line_sum_l959_95974


namespace unique_solution_condition_l959_95962

/-- The equation (x+4)(x+1) = m + 2x has exactly one real solution if and only if m = 7/4 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 4) * (x + 1) = m + 2 * x) ↔ m = 7 / 4 := by
  sorry

end unique_solution_condition_l959_95962


namespace sqrt_equation_condition_l959_95987

theorem sqrt_equation_condition (a b : ℝ) (k : ℕ+) :
  (Real.sqrt (a^2 + (k.val * b)^2) = a + k.val * b) ↔ (a * k.val * b = 0 ∧ a + k.val * b ≥ 0) :=
sorry

end sqrt_equation_condition_l959_95987


namespace yolanda_scoring_l959_95954

/-- Yolanda's basketball scoring problem -/
theorem yolanda_scoring (total_points : ℕ) (num_games : ℕ) (avg_free_throws : ℕ) (avg_two_pointers : ℕ) 
  (h1 : total_points = 345)
  (h2 : num_games = 15)
  (h3 : avg_free_throws = 4)
  (h4 : avg_two_pointers = 5) :
  (total_points / num_games - (avg_free_throws * 1 + avg_two_pointers * 2)) / 3 = 3 := by
  sorry

end yolanda_scoring_l959_95954


namespace sufficient_but_not_necessary_l959_95953

theorem sufficient_but_not_necessary :
  (∃ x : ℝ, (|x - 1| < 4 ∧ ¬(x * (x - 5) < 0))) ∧
  (∀ x : ℝ, (x * (x - 5) < 0) → |x - 1| < 4) :=
by sorry

end sufficient_but_not_necessary_l959_95953


namespace perp_condition_for_parallel_l959_95969

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the given lines and planes
variable (a b : Line) (α β : Plane)

-- State the theorem
theorem perp_condition_for_parallel 
  (h1 : perp a α) 
  (h2 : subset b β) :
  (∀ α β, parallel α β → perpLine a b) ∧ 
  (∃ α β, perpLine a b ∧ ¬parallel α β) :=
sorry

end perp_condition_for_parallel_l959_95969


namespace binary_subtraction_equiv_l959_95949

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem binary_subtraction_equiv :
  let a := [true, true, false, true, true]  -- 11011 in binary
  let b := [true, false, true]              -- 101 in binary
  let result := [true, true, false, true]   -- 1011 in binary (11 in decimal)
  binary_to_decimal a - binary_to_decimal b = binary_to_decimal result :=
by
  sorry

#eval binary_to_decimal [true, true, false, true, true]  -- Should output 27
#eval binary_to_decimal [true, false, true]              -- Should output 5
#eval binary_to_decimal [true, true, false, true]        -- Should output 11

end binary_subtraction_equiv_l959_95949


namespace line_equation_l959_95975

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- A line passing through point (1, 1) -/
def line_through_M (m : ℝ) (x y : ℝ) : Prop := x = m * (y - 1) + 1

/-- The line intersects the ellipse at two points -/
def intersects_twice (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂

/-- M is the midpoint of the line segment AB -/
def M_is_midpoint (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂ ∧
    (y₁ + y₂) / 2 = 1

/-- The main theorem -/
theorem line_equation :
  ∃ m : ℝ, ellipse M.1 M.2 ∧
    intersects_twice m ∧
    M_is_midpoint m ∧
    ∀ x y : ℝ, line_through_M m x y ↔ 3 * x + 4 * y - 7 = 0 := by
  sorry

end line_equation_l959_95975


namespace class_representatives_count_l959_95984

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_boys num_girls num_subjects : ℕ) : ℕ :=
  (num_boys.choose 1) * (num_girls.choose 2) * (num_subjects.factorial)

/-- Theorem: The number of ways to select 2 girls from 3 girls, 1 boy from 3 boys,
    and arrange them as representatives for 3 subjects is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 3 = 54 := by
  sorry

end class_representatives_count_l959_95984


namespace perpendicular_tangents_intersection_l959_95904

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x₀, 
    prove that x₀ = - ∛(36) / 6 -/
theorem perpendicular_tangents_intersection (x₀ : ℝ) : 
  (∀ x, (2 * x) * (3 * x^2) = -1 → x = x₀) →
  x₀ = - (36 : ℝ)^(1/3) / 6 :=
by sorry

end perpendicular_tangents_intersection_l959_95904


namespace girls_not_participating_count_l959_95901

/-- Represents the number of students in an extracurricular activity -/
structure Activity where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- Represents the school's student body and activities -/
structure School where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  soccer : Activity
  basketball : Activity
  chess : Activity
  math : Activity
  glee : Activity
  absent_boys : ℕ
  absent_girls : ℕ

/-- The number of girls not participating in any extracurricular activities -/
def girls_not_participating (s : School) : ℕ :=
  s.total_girls - s.soccer.girls - s.basketball.girls - s.chess.girls - s.math.girls - s.glee.girls - s.absent_girls

theorem girls_not_participating_count (s : School) :
  s.total_students = 800 ∧
  s.total_boys = 420 ∧
  s.total_girls = 380 ∧
  s.soccer.total = 320 ∧
  s.soccer.boys = 224 ∧
  s.basketball.total = 280 ∧
  s.basketball.girls = 182 ∧
  s.chess.total = 70 ∧
  s.chess.boys = 56 ∧
  s.math.total = 50 ∧
  s.math.boys = 25 ∧
  s.math.girls = 25 ∧
  s.absent_boys = 21 ∧
  s.absent_girls = 30 →
  girls_not_participating s = 33 := by
  sorry


end girls_not_participating_count_l959_95901


namespace hyperbola_equation_correct_l959_95967

/-- A hyperbola with foci on the X-axis, distance between vertices of 6, and asymptote equations y = ± 3/2 x -/
structure Hyperbola where
  foci_on_x_axis : Bool
  vertex_distance : ℝ
  asymptote_slope : ℝ

/-- The equation of a hyperbola in the form (x²/a² - y²/b² = 1) -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 9 - 4 * y^2 / 81 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_correct (h : Hyperbola) 
  (h_foci : h.foci_on_x_axis = true)
  (h_vertex : h.vertex_distance = 6)
  (h_asymptote : h.asymptote_slope = 3/2) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - 4 * y^2 / 81 = 1 := by
  sorry

end hyperbola_equation_correct_l959_95967


namespace min_value_expression_l959_95992

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 17/2 := by
  sorry

end min_value_expression_l959_95992


namespace base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l959_95956

/- Define a function to convert a list of digits to a number in base 3 -/
def toBase3 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/- Define a function to check if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/- Define a function to sum the digits of a list -/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

/- Theorem for base 3 even numbers -/
theorem base3_even_iff_sum_even (digits : List Nat) :
  isEven (toBase3 digits) ↔ isEven (sumDigits digits) := by
  sorry

/- Define a function to convert a list of digits to a number in base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/- Define a function to check if a number is divisible by 7 -/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/- Define a function to compute the sum of digits multiplied by powers of 10 mod 7 -/
def sumDigitsPowersOf10Mod7 (digits : List Nat) : Nat :=
  (List.range digits.length).zip digits
  |> List.foldl (fun acc (i, d) => (acc + d * (10^i % 7)) % 7) 0

/- Theorem for base 10 multiples of 7 -/
theorem base10_multiple_of_7_iff_sum_congruent (digits : List Nat) :
  isDivisibleBy7 (toBase10 digits) ↔ sumDigitsPowersOf10Mod7 digits = 0 := by
  sorry

end base3_even_iff_sum_even_base10_multiple_of_7_iff_sum_congruent_l959_95956


namespace alice_gives_no_stickers_to_charlie_l959_95923

/-- Represents the sticker distribution problem --/
def sticker_distribution (c : ℕ) : Prop :=
  let alice_initial := 12 * c
  let bob_initial := 3 * c
  let charlie_initial := c
  let dave_initial := c
  let alice_final := alice_initial - (2 * c - bob_initial) - (3 * c - dave_initial)
  let bob_final := 2 * c
  let charlie_final := c
  let dave_final := 3 * c
  (alice_final - alice_initial) / alice_initial = 0

/-- Theorem stating that Alice gives 0 fraction of her stickers to Charlie --/
theorem alice_gives_no_stickers_to_charlie (c : ℕ) (hc : c > 0) :
  sticker_distribution c :=
sorry

end alice_gives_no_stickers_to_charlie_l959_95923


namespace son_age_is_eighteen_l959_95972

/-- Represents the ages of a father and son -/
structure FatherSonAges where
  fatherAge : ℕ
  sonAge : ℕ

/-- The condition that the father is 20 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 20

/-- The condition that in two years, the father's age will be twice the son's age -/
def futureAgeRelation (ages : FatherSonAges) : Prop :=
  ages.fatherAge + 2 = 2 * (ages.sonAge + 2)

/-- Theorem stating that given the conditions, the son's present age is 18 -/
theorem son_age_is_eighteen (ages : FatherSonAges) 
  (h1 : ageDifference ages) (h2 : futureAgeRelation ages) : ages.sonAge = 18 := by
  sorry

end son_age_is_eighteen_l959_95972


namespace function_increasing_iff_a_nonpositive_l959_95945

/-- The function f(x) = (1/3)x^3 - ax is increasing on ℝ if and only if a ≤ 0 -/
theorem function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => (1/3) * x^3 - a * x) (x^2 - a) x) →
  (∀ x y : ℝ, x < y → ((1/3) * x^3 - a * x) < ((1/3) * y^3 - a * y)) ↔
  a ≤ 0 := by sorry

end function_increasing_iff_a_nonpositive_l959_95945


namespace evaluate_expression_l959_95917

theorem evaluate_expression : 9^6 * 3^3 / 3^15 = 1 := by
  sorry

end evaluate_expression_l959_95917


namespace circle_center_proof_l959_95957

/-- A line in the 2D plane represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

/-- Check if a circle is tangent to a line --/
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.1 + l.b * c.center.2 - l.c) = c.radius * (l.a^2 + l.b^2).sqrt

theorem circle_center_proof :
  let line1 : Line := { a := 5, b := -2, c := 40 }
  let line2 : Line := { a := 5, b := -2, c := 10 }
  let line3 : Line := { a := 3, b := -4, c := 0 }
  let center : ℝ × ℝ := (50/7, 75/14)
  ∃ (r : ℝ), 
    let c : Circle := { center := center, radius := r }
    circleTangentToLine c line1 ∧ 
    circleTangentToLine c line2 ∧ 
    pointOnLine center line3 := by
  sorry


end circle_center_proof_l959_95957


namespace repair_cost_is_13000_l959_95933

/-- Calculates the repair cost given the purchase price, selling price, and profit percentage --/
def calculate_repair_cost (purchase_price selling_price profit_percent : ℚ) : ℚ :=
  let total_cost := selling_price / (1 + profit_percent / 100)
  total_cost - purchase_price

/-- Theorem stating that the repair cost is 13000 given the problem conditions --/
theorem repair_cost_is_13000 :
  let purchase_price : ℚ := 42000
  let selling_price : ℚ := 60900
  let profit_percent : ℚ := 10.727272727272727
  calculate_repair_cost purchase_price selling_price profit_percent = 13000 := by
  sorry


end repair_cost_is_13000_l959_95933


namespace a_squared_coefficient_zero_l959_95988

theorem a_squared_coefficient_zero (p : ℚ) : 
  (∀ a : ℚ, (a^2 - p*a + 6) * (2*a - 1) = (-p*a + 6) * (2*a - 1)) → p = -1/2 :=
by sorry

end a_squared_coefficient_zero_l959_95988


namespace sqrt_2_simplest_l959_95908

-- Define a function to represent the simplicity of a square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ 1 → x ≠ y * y * (x / (y * y))

-- State the theorem
theorem sqrt_2_simplest : 
  is_simplest_sqrt (Real.sqrt 2) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 20) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt (1/2)) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 0.2) :=
sorry

end sqrt_2_simplest_l959_95908


namespace hyperbola_parameters_l959_95950

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The shortest distance from a point on the hyperbola to one of its foci -/
def shortest_focal_distance (h : Hyperbola) : ℝ := 2

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y / p.x = h.b / h.a ∨ p.y / p.x = -h.b / h.a

/-- The given point P -/
def P : Point := ⟨3, 4⟩

theorem hyperbola_parameters (h : Hyperbola) 
  (h_focal : shortest_focal_distance h = 2)
  (h_asymptote : on_asymptote h P) :
  h.a = 3 ∧ h.b = 4 := by sorry

end hyperbola_parameters_l959_95950
