import Mathlib

namespace intersection_segment_length_l899_89909

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
by sorry

end intersection_segment_length_l899_89909


namespace tim_necklace_profit_l899_89980

/-- Represents the properties of a necklace type -/
structure NecklaceType where
  charms : ℕ
  charmCost : ℕ
  sellingPrice : ℕ

/-- Calculates the profit for a single necklace -/
def profit (n : NecklaceType) : ℕ :=
  n.sellingPrice - n.charms * n.charmCost

/-- Represents the sales information -/
structure Sales where
  typeA : NecklaceType
  typeB : NecklaceType
  soldA : ℕ
  soldB : ℕ

/-- Calculates the total profit from all sales -/
def totalProfit (s : Sales) : ℕ :=
  s.soldA * profit s.typeA + s.soldB * profit s.typeB

/-- Tim's necklace business theorem -/
theorem tim_necklace_profit :
  let s : Sales := {
    typeA := { charms := 8, charmCost := 10, sellingPrice := 125 },
    typeB := { charms := 12, charmCost := 18, sellingPrice := 280 },
    soldA := 45,
    soldB := 35
  }
  totalProfit s = 4265 := by sorry

end tim_necklace_profit_l899_89980


namespace fraction_equality_l899_89914

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end fraction_equality_l899_89914


namespace holiday_savings_l899_89972

theorem holiday_savings (sam_savings : ℕ) (total_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1000 →
  total_savings = 1900 →
  victory_savings < sam_savings →
  victory_savings = total_savings - sam_savings →
  sam_savings - victory_savings = 100 :=
by
  sorry

end holiday_savings_l899_89972


namespace binary_three_is_three_l899_89993

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the number 3 -/
def binary_three : List Bool := [true, true]

theorem binary_three_is_three :
  binary_to_decimal binary_three = 3 := by
  sorry

end binary_three_is_three_l899_89993


namespace find_x_l899_89904

-- Define the binary operation
def binary_op (n : ℤ) (x : ℤ) : ℤ := n - (n * x)

-- State the theorem
theorem find_x : ∃ x : ℤ, 
  (∀ n : ℕ, n > 2 → binary_op n x ≥ 10) ∧ 
  (binary_op 2 x < 10) ∧
  x = -3 := by
  sorry

end find_x_l899_89904


namespace vector_equality_implies_x_value_l899_89953

/-- Given vectors a and b in R², if the magnitude of their sum equals the magnitude of their difference, then the second component of b is 3. -/
theorem vector_equality_implies_x_value (a b : ℝ × ℝ) :
  a = (2, -4) →
  b.1 = 6 →
  ‖a + b‖ = ‖a - b‖ →
  b.2 = 3 := by
  sorry


end vector_equality_implies_x_value_l899_89953


namespace caterer_order_total_price_l899_89987

theorem caterer_order_total_price :
  let ice_cream_bars := 125
  let sundaes := 125
  let ice_cream_bar_price := 0.60
  let sundae_price := 1.2
  let total_price := ice_cream_bars * ice_cream_bar_price + sundaes * sundae_price
  total_price = 225 := by sorry

end caterer_order_total_price_l899_89987


namespace average_weight_increase_l899_89966

theorem average_weight_increase (initial_weight : ℝ) : 
  let initial_average := (initial_weight + 65) / 2
  let new_average := (initial_weight + 74) / 2
  new_average - initial_average = 4.5 := by
sorry


end average_weight_increase_l899_89966


namespace tattoo_ratio_l899_89941

def jason_arm_tattoos : ℕ := 2
def jason_leg_tattoos : ℕ := 3
def jason_arms : ℕ := 2
def jason_legs : ℕ := 2
def adam_tattoos : ℕ := 23

def jason_total_tattoos : ℕ :=
  jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

theorem tattoo_ratio :
  ∃ (m : ℕ), adam_tattoos = m * jason_total_tattoos + 3 ∧
  adam_tattoos.gcd jason_total_tattoos = 1 := by
  sorry

end tattoo_ratio_l899_89941


namespace factorization_equality_l899_89982

theorem factorization_equality (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := by
  sorry

end factorization_equality_l899_89982


namespace log_division_simplification_l899_89918

theorem log_division_simplification :
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end log_division_simplification_l899_89918


namespace three_digit_congruence_count_l899_89959

theorem three_digit_congruence_count : 
  let count := Finset.filter (fun y => 100 ≤ y ∧ y ≤ 999 ∧ (4325 * y + 692) % 17 = 1403 % 17) (Finset.range 1000)
  ↑count.card = 53 := by sorry

end three_digit_congruence_count_l899_89959


namespace polynomial_factorization_l899_89969

theorem polynomial_factorization (x y : ℝ) : x^3 * y - 4 * x * y^3 = x * y * (x + 2 * y) * (x - 2 * y) := by
  sorry

end polynomial_factorization_l899_89969


namespace power_difference_equality_l899_89956

theorem power_difference_equality (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^1023 - 1/x^1023 = 5 * Real.sqrt 2 := by
  sorry

end power_difference_equality_l899_89956


namespace order_of_powers_l899_89975

theorem order_of_powers : 4^9 < 6^7 ∧ 6^7 < 3^13 := by
  sorry

end order_of_powers_l899_89975


namespace f_diff_max_min_eq_one_l899_89911

/-- The function f(x) = x^2 - 2bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x - 1

/-- The closed interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

/-- The statement that the difference between the maximum and minimum values of f(x) on [0, 1] is 1 -/
def diffMaxMin (b : ℝ) : Prop :=
  ∃ (max min : ℝ), (∀ x ∈ I, f b x ≤ max) ∧
                   (∀ x ∈ I, min ≤ f b x) ∧
                   (max - min = 1)

/-- The main theorem -/
theorem f_diff_max_min_eq_one :
  ∀ b : ℝ, diffMaxMin b ↔ (b = 0 ∨ b = 1) :=
sorry

end f_diff_max_min_eq_one_l899_89911


namespace sum_min_max_x_l899_89900

theorem sum_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 8) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 8 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 4 :=
sorry

end sum_min_max_x_l899_89900


namespace factor_sum_l899_89948

theorem factor_sum (a b : ℝ) : 
  (∃ m n : ℝ, ∀ x : ℝ, x^4 + a*x^2 + b = (x^2 + 2*x + 5) * (x^2 + m*x + n)) →
  a + b = 31 := by
sorry

end factor_sum_l899_89948


namespace f_divisible_by_36_l899_89923

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem f_divisible_by_36 : ∀ n : ℕ, 36 ∣ f n := by sorry

end f_divisible_by_36_l899_89923


namespace cos_squared_alpha_minus_pi_fourth_l899_89997

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) :
  Real.cos (α - Real.pi / 4) ^ 2 = 2 / 3 := by
  sorry

end cos_squared_alpha_minus_pi_fourth_l899_89997


namespace water_tank_capacity_water_tank_capacity_proof_l899_89936

/-- Proves that a cylindrical water tank holds 75 liters when full -/
theorem water_tank_capacity : ℝ → Prop :=
  fun c => 
    (∃ w : ℝ, w / c = 1 / 3 ∧ (w + 5) / c = 2 / 5) → c = 75

/-- The proof of the water tank capacity theorem -/
theorem water_tank_capacity_proof : water_tank_capacity 75 := by
  sorry

end water_tank_capacity_water_tank_capacity_proof_l899_89936


namespace line_perpendicular_to_plane_and_line_l899_89977

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line (a b : Line) (α : Plane) :
  (perpendicularToPlane a α ∧ perpendicular a b) → parallelToPlane b α := by
  sorry

end line_perpendicular_to_plane_and_line_l899_89977


namespace diophantine_equation_solutions_l899_89958

theorem diophantine_equation_solutions (k : ℤ) :
  (k > 7 → ∃ (x y : ℕ), 5 * x + 3 * y = k) ∧
  (k > 15 → ∃ (x y : ℕ+), 5 * x + 3 * y = k) ∧
  (∀ N : ℤ, (∀ k > N, ∃ (x y : ℕ+), 5 * x + 3 * y = k) → N ≥ 15) :=
by sorry

end diophantine_equation_solutions_l899_89958


namespace imaginary_part_of_complex_fraction_l899_89968

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by sorry

end imaginary_part_of_complex_fraction_l899_89968


namespace bertha_initial_balls_l899_89944

def tennis_balls (initial_balls : ℕ) : Prop :=
  let worn_out := 20 / 10
  let lost := 20 / 5
  let bought := (20 / 4) * 3
  initial_balls - worn_out - lost + bought - 1 = 10

theorem bertha_initial_balls :
  ∃ (initial_balls : ℕ), tennis_balls initial_balls ∧ initial_balls = 2 :=
sorry

end bertha_initial_balls_l899_89944


namespace gcd_18_30_is_6_and_even_l899_89907

theorem gcd_18_30_is_6_and_even : 
  Nat.gcd 18 30 = 6 ∧ Even 6 := by
  sorry

end gcd_18_30_is_6_and_even_l899_89907


namespace pythagorean_triple_3_4_5_l899_89995

theorem pythagorean_triple_3_4_5 : 
  ∃ (x : ℕ), x > 0 ∧ 3^2 + 4^2 = x^2 :=
by
  use 5
  sorry

#check pythagorean_triple_3_4_5

end pythagorean_triple_3_4_5_l899_89995


namespace hyperbola_asymptote_circle_intersection_chord_length_l899_89938

/-- The length of the chord formed by the intersection of an asymptote of a hyperbola with a specific circle -/
theorem hyperbola_asymptote_circle_intersection_chord_length 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 5  -- eccentricity
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 3)^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x ∨ y = -(b / a) * x}
  ∀ (A B : ℝ × ℝ), A ∈ circle → B ∈ circle → A ∈ asymptote → B ∈ asymptote →
  e^2 = 1 + b^2 / a^2 →
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 :=
by sorry

end hyperbola_asymptote_circle_intersection_chord_length_l899_89938


namespace smallest_integer_greater_than_half_ninths_l899_89929

theorem smallest_integer_greater_than_half_ninths : ∀ n : ℤ, (1/2 : ℚ) < (n : ℚ)/9 ↔ n ≥ 5 :=
by sorry

end smallest_integer_greater_than_half_ninths_l899_89929


namespace cubic_sum_greater_than_mixed_product_l899_89950

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : 
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end cubic_sum_greater_than_mixed_product_l899_89950


namespace equilateral_triangle_from_sequences_l899_89940

/-- Given a triangle ABC where:
    - The angles A, B, C form an arithmetic sequence
    - The sides a, b, c (opposite to angles A, B, C respectively) form a geometric sequence
    Prove that the triangle is equilateral -/
theorem equilateral_triangle_from_sequences (A B C a b c : ℝ) : 
  (∃ d : ℝ, B - A = d ∧ C - B = d) →  -- Angles form arithmetic sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Sides form geometric sequence
  A + B + C = π →                     -- Sum of angles in a triangle
  A > 0 ∧ B > 0 ∧ C > 0 →             -- Positive angles
  a > 0 ∧ b > 0 ∧ c > 0 →             -- Positive side lengths
  (A = π/3 ∧ B = π/3 ∧ C = π/3) :=    -- Triangle is equilateral
by sorry

end equilateral_triangle_from_sequences_l899_89940


namespace cross_product_example_l899_89924

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example :
  let u : ℝ × ℝ × ℝ := (3, 2, 4)
  let v : ℝ × ℝ × ℝ := (4, 3, -1)
  cross_product u v = (-14, 19, 1) := by
  sorry

end cross_product_example_l899_89924


namespace square_root_81_l899_89962

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end square_root_81_l899_89962


namespace disjunction_true_l899_89925

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end disjunction_true_l899_89925


namespace unit_vectors_collinear_with_vector_l899_89999

def vector : ℝ × ℝ × ℝ := (-3, -4, 5)

theorem unit_vectors_collinear_with_vector :
  let norm := Real.sqrt ((-3)^2 + (-4)^2 + 5^2)
  let unit_vector₁ : ℝ × ℝ × ℝ := (3 * Real.sqrt 2 / 10, 2 * Real.sqrt 2 / 5, -Real.sqrt 2 / 2)
  let unit_vector₂ : ℝ × ℝ × ℝ := (-3 * Real.sqrt 2 / 10, -2 * Real.sqrt 2 / 5, Real.sqrt 2 / 2)
  (∃ (k : ℝ), vector = (k • unit_vector₁)) ∧
  (∃ (k : ℝ), vector = (k • unit_vector₂)) ∧
  (norm * norm = (-3)^2 + (-4)^2 + 5^2) ∧
  (Real.sqrt 2 * Real.sqrt 2 = 2) ∧
  (∀ (v : ℝ × ℝ × ℝ), (∃ (k : ℝ), vector = (k • v)) → (v = unit_vector₁ ∨ v = unit_vector₂)) :=
by sorry

end unit_vectors_collinear_with_vector_l899_89999


namespace lcm_gcd_product_24_60_l899_89902

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end lcm_gcd_product_24_60_l899_89902


namespace quadratic_vertex_l899_89981

/-- The quadratic function f(x) = 3x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by sorry

end quadratic_vertex_l899_89981


namespace difference_of_squares_and_product_l899_89974

theorem difference_of_squares_and_product (a b : ℝ) 
  (h1 : a^2 + b^2 = 150) 
  (h2 : a * b = 25) : 
  |a - b| = 10 := by
sorry

end difference_of_squares_and_product_l899_89974


namespace weight_of_B_l899_89939

theorem weight_of_B (A B C : ℝ) :
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 41 →
  (B + C) / 2 = 43 →
  B = 33 := by
sorry

end weight_of_B_l899_89939


namespace journey_speed_proof_l899_89930

/-- Proves that given a journey of approximately 3 km divided into three equal parts,
    where the first part is traveled at 3 km/hr, the second at 4 km/hr,
    and the total journey takes 47 minutes, the speed of the third part must be 5 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3.000000000000001)
  (h2 : total_time = 47 / 60) -- Convert 47 minutes to hours
  (h3 : ∃ (d : ℝ), d > 0 ∧ 3 * d = total_distance) -- Equal distances for each part
  (h4 : ∃ (v : ℝ), v > 0 ∧ 1 / 3 + 1 / 4 + 1 / v = total_time) -- Time equation
  : ∃ (v : ℝ), v = 5 := by
  sorry

end journey_speed_proof_l899_89930


namespace S_is_line_l899_89979

-- Define the complex number (2+5i)
def a : ℂ := 2 + 5 * Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), a * z = r}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (m b : ℝ), S = {z : ℂ | z.im = m * z.re + b} :=
sorry

end S_is_line_l899_89979


namespace work_completion_time_l899_89988

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1 / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1 / 18)         -- b alone finishes in 18 days
  (h3 : c = 1 / 9)          -- c alone finishes in 9 days
  : a = 1 / 12 :=           -- a alone finishes in 12 days
by sorry

end work_completion_time_l899_89988


namespace graph_quadrants_l899_89961

/-- Given a > 1 and b < -1, the graph of f(x) = a^x + b intersects Quadrants I, III, and IV, but not Quadrant II -/
theorem graph_quadrants (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  let f : ℝ → ℝ := λ x ↦ a^x + b
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Quadrant I
  (∃ x y, x < 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant III
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant IV
  (∀ x y, ¬(x < 0 ∧ y > 0 ∧ f x = y))  -- Not in Quadrant II
  := by sorry

end graph_quadrants_l899_89961


namespace m_values_l899_89955

-- Define the sets A and B
def A : Set ℝ := {x | x^2 ≠ 1}
def B (m : ℝ) : Set ℝ := {x | m * x = 1}

-- State the theorem
theorem m_values (h : ∀ m : ℝ, A ∪ B m = A) :
  {m : ℝ | ∃ x, x ∈ B m} = {-1, 0, 1} := by sorry

end m_values_l899_89955


namespace second_smallest_divisor_l899_89990

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem second_smallest_divisor (n : ℕ) : 
  (is_divisible (n + 3) 12 ∧ 
   is_divisible (n + 3) 35 ∧ 
   is_divisible (n + 3) 40) →
  (∀ m : ℕ, m < n → ¬(is_divisible (m + 3) 12 ∧ 
                      is_divisible (m + 3) 35 ∧ 
                      is_divisible (m + 3) 40)) →
  (∃ d : ℕ, d ≠ 1 ∧ is_divisible (n + 3) d ∧ 
   d ≠ 12 ∧ d ≠ 35 ∧ d ≠ 40 ∧
   (∀ k : ℕ, 1 < k → k < d → ¬is_divisible (n + 3) k)) →
  is_divisible (n + 3) 3 :=
by sorry

end second_smallest_divisor_l899_89990


namespace product_of_integers_l899_89976

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (diff_squares_eq : x^2 - y^2 = 36) :
  x * y = 80 := by
  sorry

end product_of_integers_l899_89976


namespace car_speed_problem_l899_89937

/-- Proves that the speed of the first car is 60 mph given the problem conditions -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  2.5 * v + 2.5 * 64 = 310 → 
  v = 60 := by
sorry

end car_speed_problem_l899_89937


namespace min_value_and_exponential_sum_l899_89991

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - 2*a| + |x + b|

-- State the theorem
theorem min_value_and_exponential_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 2) 
  (hmin_exists : ∃ x, f x a b = 2) : 
  (2*a + b = 2) ∧ (∀ a' b', a' > 0 → b' > 0 → 2*a' + b' = 2 → 9^a' + 3^b' ≥ 6) ∧ 
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ 2*a' + b' = 2 ∧ 9^a' + 3^b' = 6) :=
sorry

end min_value_and_exponential_sum_l899_89991


namespace set_relationship_l899_89919

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0 ∧ x ≠ 1}
def N : Set ℝ := {x | |x + 1| ≤ 2}
def P : Set ℝ := {x | (1/2 : ℝ)^(x^2 + 2*x - 3) ≥ 1}

-- State the theorem
theorem set_relationship : M ⊆ N ∧ N = P := by sorry

end set_relationship_l899_89919


namespace line_circle_intersection_l899_89912

/-- The line y = x + 1 intersects the circle x² + y² = 1 at two distinct points, 
    and neither of these points is the center of the circle (0, 0). -/
theorem line_circle_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ = x₁ + 1) ∧ (x₁^2 + y₁^2 = 1) ∧
    (y₂ = x₂ + 1) ∧ (x₂^2 + y₂^2 = 1) ∧
    (x₁ ≠ x₂) ∧ (y₁ ≠ y₂) ∧
    (x₁ ≠ 0 ∨ y₁ ≠ 0) ∧ (x₂ ≠ 0 ∨ y₂ ≠ 0) :=
by sorry

end line_circle_intersection_l899_89912


namespace rain_on_tuesday_l899_89934

theorem rain_on_tuesday (rain_monday : ℝ) (no_rain : ℝ) (rain_both : ℝ)
  (h1 : rain_monday = 0.62)
  (h2 : no_rain = 0.28)
  (h3 : rain_both = 0.44) :
  rain_monday + (1 - no_rain) - rain_both = 0.54 := by
sorry

end rain_on_tuesday_l899_89934


namespace discount_price_l899_89931

theorem discount_price (a : ℝ) :
  let discounted_price := a
  let discount_rate := 0.3
  let original_price := discounted_price / (1 - discount_rate)
  original_price = 10 / 7 * a :=
by sorry

end discount_price_l899_89931


namespace smallest_n_congruence_l899_89913

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 19 * m ≡ 1453 [MOD 8] → n ≤ m) ∧ 19 * n ≡ 1453 [MOD 8] := by
  sorry

end smallest_n_congruence_l899_89913


namespace domain_of_composition_l899_89971

def f : Set ℝ → Prop := λ S => ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1

theorem domain_of_composition (f : Set ℝ → Prop) (h : f (Set.Icc 0 1)) :
  f (Set.Icc 0 (1/2)) :=
sorry

end domain_of_composition_l899_89971


namespace blue_corduroy_glasses_count_l899_89910

theorem blue_corduroy_glasses_count (total_students : ℕ) 
  (blue_shirt_percent : ℚ) (corduroy_percent : ℚ) (glasses_percent : ℚ) :
  total_students = 1500 →
  blue_shirt_percent = 35 / 100 →
  corduroy_percent = 20 / 100 →
  glasses_percent = 15 / 100 →
  ⌊total_students * blue_shirt_percent * corduroy_percent * glasses_percent⌋ = 15 := by
sorry

end blue_corduroy_glasses_count_l899_89910


namespace solve_for_k_l899_89927

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end solve_for_k_l899_89927


namespace company_kw_price_percentage_l899_89985

/-- The price of Company KW as a percentage of the combined assets of Companies A and B -/
theorem company_kw_price_percentage (P A B : ℝ) 
  (h1 : P = 1.30 * A) 
  (h2 : P = 2.00 * B) : 
  ∃ (ε : ℝ), abs (P / (A + B) - 0.7879) < ε ∧ ε > 0 := by
  sorry

end company_kw_price_percentage_l899_89985


namespace units_digit_of_27_times_46_l899_89921

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end units_digit_of_27_times_46_l899_89921


namespace factorization_problem_1_l899_89963

theorem factorization_problem_1 (x y : ℝ) :
  9 - x^2 + 12*x*y - 36*y^2 = (3 + x - 6*y) * (3 - x + 6*y) := by sorry

end factorization_problem_1_l899_89963


namespace samantha_born_in_1975_l899_89928

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1983

-- Define Samantha's age when she took the seventh AMC 8
def samantha_age_seventh_amc8 : ℕ := 14

-- Define the number of years between first and seventh AMC 8
def years_between_first_and_seventh : ℕ := 6

-- Define the year Samantha took the seventh AMC 8
def samantha_seventh_amc8_year : ℕ := first_amc8_year + years_between_first_and_seventh

-- Define Samantha's birth year
def samantha_birth_year : ℕ := samantha_seventh_amc8_year - samantha_age_seventh_amc8

-- Theorem to prove
theorem samantha_born_in_1975 : samantha_birth_year = 1975 := by
  sorry

end samantha_born_in_1975_l899_89928


namespace unique_paths_equal_binomial_coefficient_l899_89901

/-- The number of rows in the grid -/
def n : ℕ := 6

/-- The number of columns in the grid -/
def m : ℕ := 6

/-- The total number of steps required to reach the destination -/
def total_steps : ℕ := n + m

/-- The number of ways to choose n right moves out of total_steps moves -/
def num_paths : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of unique paths from A to B is equal to C(12,6) -/
theorem unique_paths_equal_binomial_coefficient : 
  num_paths = 924 := by sorry

end unique_paths_equal_binomial_coefficient_l899_89901


namespace josh_marbles_l899_89986

/-- The number of marbles Josh has after losing some and giving away half of the remainder --/
def final_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  let remaining := initial - lost
  remaining - (remaining / 2)

/-- Theorem stating that Josh ends up with 103 marbles --/
theorem josh_marbles : final_marbles 320 115 = 103 := by
  sorry

end josh_marbles_l899_89986


namespace sum_of_exponential_equality_l899_89994

theorem sum_of_exponential_equality (a b : ℝ) (h : (2 : ℝ) ^ b = (2 : ℝ) ^ (6 - a)) : a + b = 6 := by
  sorry

end sum_of_exponential_equality_l899_89994


namespace hazel_lemonade_cups_l899_89970

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  total_cups = 56 ∧
  (total_cups / 2 : ℕ) + cups_sold_to_kids + (cups_sold_to_kids / 2 : ℕ) + 1 = total_cups :=
by sorry


end hazel_lemonade_cups_l899_89970


namespace total_discount_is_65_percent_l899_89960

/-- Represents the discount percentage as a real number between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The half-price sale discount -/
def half_price_discount : Discount := ⟨0.5, by norm_num⟩

/-- The additional coupon discount -/
def coupon_discount : Discount := ⟨0.3, by norm_num⟩

/-- Calculates the final price after applying two successive discounts -/
def apply_discounts (d1 d2 : Discount) : ℝ := (1 - d1.val) * (1 - d2.val)

/-- The theorem to be proved -/
theorem total_discount_is_65_percent :
  apply_discounts half_price_discount coupon_discount = 0.35 := by
  sorry

end total_discount_is_65_percent_l899_89960


namespace cubic_equation_roots_range_l899_89932

theorem cubic_equation_roots_range (k : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x = k ∧ y^3 - 3*y = k ∧ z^3 - 3*z = k) → 
  -2 < k ∧ k < 2 := by
sorry

end cubic_equation_roots_range_l899_89932


namespace parallel_line_through_point_l899_89973

/-- Given a point A(2,1) and a line 2x-y+3=0, prove that 2x-y-3=0 is the equation of the line
    passing through A and parallel to 2x-y+3=0 -/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 3 = 0) →  -- Given line equation
  (2 * 2 - 1 = y) →      -- Point A(2,1) satisfies the new line equation
  (2 * x - y - 3 = 0) →  -- New line equation
  (∃ k : ℝ, k ≠ 0 ∧ (2 : ℝ) / 1 = (2 : ℝ) / 1) -- Parallel lines have equal slopes
  := by sorry

end parallel_line_through_point_l899_89973


namespace lab_budget_remaining_l899_89951

/-- Given a laboratory budget and expenses, calculate the remaining budget. -/
theorem lab_budget_remaining (budget : ℚ) (flask_cost : ℚ) : 
  budget = 325 →
  flask_cost = 150 →
  let test_tube_cost := (2 / 3) * flask_cost
  let safety_gear_cost := (1 / 2) * test_tube_cost
  let total_expense := flask_cost + test_tube_cost + safety_gear_cost
  budget - total_expense = 25 := by sorry

end lab_budget_remaining_l899_89951


namespace train_speed_without_stoppages_l899_89916

/-- The average speed of a train without stoppages, given certain conditions. -/
theorem train_speed_without_stoppages (distance : ℝ) (time_with_stops : ℝ) (time_without_stops : ℝ)
  (h1 : time_without_stops = time_with_stops / 2)
  (h2 : distance / time_with_stops = 125) :
  distance / time_without_stops = 250 := by
  sorry

end train_speed_without_stoppages_l899_89916


namespace sum_squares_five_consecutive_integers_l899_89942

theorem sum_squares_five_consecutive_integers (n : ℤ) :
  ∃ k : ℤ, (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k :=
by sorry

end sum_squares_five_consecutive_integers_l899_89942


namespace third_term_is_nine_l899_89954

/-- A geometric sequence where the first term is 4, the second term is 6, and the third term is x -/
def geometric_sequence (x : ℝ) : ℕ → ℝ
| 0 => 4
| 1 => 6
| 2 => x
| (n + 3) => sorry

/-- Theorem: In the given geometric sequence, the third term x is equal to 9 -/
theorem third_term_is_nine :
  ∃ x : ℝ, (∀ n : ℕ, geometric_sequence x (n + 1) = (geometric_sequence x n) * (geometric_sequence x 1 / geometric_sequence x 0)) → x = 9 :=
sorry

end third_term_is_nine_l899_89954


namespace megan_removed_two_albums_l899_89957

/-- Calculates the number of albums removed from a shopping cart. -/
def albums_removed (initial_albums : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ) : ℕ :=
  initial_albums - (total_songs_bought / songs_per_album)

/-- Proves that Megan removed 2 albums from her shopping cart. -/
theorem megan_removed_two_albums :
  albums_removed 8 7 42 = 2 := by
  sorry

end megan_removed_two_albums_l899_89957


namespace intersection_implies_a_value_l899_89926

def A (a : ℝ) : Set ℝ := {4, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {1} → a = -1 := by
  sorry

end intersection_implies_a_value_l899_89926


namespace min_value_theorem_l899_89949

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) : 
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ b, b + 3 / a ≥ m :=
sorry

end min_value_theorem_l899_89949


namespace floor_length_is_20_l899_89945

/-- Proves that the length of a rectangular floor is 20 meters given the specified conditions -/
theorem floor_length_is_20 (breadth : ℝ) (length : ℝ) (area : ℝ) (total_cost : ℝ) (rate : ℝ) : 
  length = breadth + 2 * breadth →  -- length is 200% more than breadth
  area = length * breadth →         -- area formula
  area = total_cost / rate →        -- area from cost and rate
  total_cost = 400 →                -- given total cost
  rate = 3 →                        -- given rate per square meter
  length = 20 := by
sorry

end floor_length_is_20_l899_89945


namespace luke_money_in_january_l899_89967

/-- The amount of money Luke had in January -/
def initial_amount : ℕ := sorry

/-- The amount Luke spent -/
def spent : ℕ := 11

/-- The amount Luke received from his mom -/
def received : ℕ := 21

/-- The amount Luke has now -/
def current_amount : ℕ := 58

theorem luke_money_in_january :
  initial_amount = 48 :=
by
  have h : initial_amount - spent + received = current_amount := by sorry
  sorry

end luke_money_in_january_l899_89967


namespace steves_return_speed_l899_89996

/-- Proves that given a round trip with specified conditions, the return speed is 10 km/h -/
theorem steves_return_speed (total_distance : ℝ) (total_time : ℝ) (outbound_distance : ℝ) :
  total_distance = 40 →
  total_time = 6 →
  outbound_distance = 20 →
  let outbound_speed := outbound_distance / (total_time / 2)
  let return_speed := 2 * outbound_speed
  return_speed = 10 := by
  sorry

end steves_return_speed_l899_89996


namespace unit_conversions_l899_89998

-- Define conversion factors
def cm_to_dm : ℚ := 10
def cm_to_m : ℚ := 100
def kg_to_ton : ℚ := 1000
def g_to_kg : ℚ := 1000
def min_to_hour : ℚ := 60

-- Define the theorem
theorem unit_conversions :
  (4800 / cm_to_dm = 480 ∧ 4800 / cm_to_m = 48) ∧
  (5080 / kg_to_ton = 5 ∧ 5080 % kg_to_ton = 80) ∧
  (8 * g_to_kg + 60 = 8060) ∧
  (3 * min_to_hour + 20 = 200) := by
  sorry


end unit_conversions_l899_89998


namespace sqrt_360000_l899_89917

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_l899_89917


namespace stating_count_five_digit_divisible_by_12_is_72000_l899_89922

/-- 
A function that counts the number of positive five-digit integers divisible by 12.
-/
def count_five_digit_divisible_by_12 : ℕ :=
  sorry

/-- 
Theorem stating that the count of positive five-digit integers divisible by 12 is 72000.
-/
theorem count_five_digit_divisible_by_12_is_72000 : 
  count_five_digit_divisible_by_12 = 72000 :=
by sorry

end stating_count_five_digit_divisible_by_12_is_72000_l899_89922


namespace solution_set_is_open_interval_l899_89903

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def increasing_f := ∀ x y, x < y → f x < f y
def f_zero_is_neg_one := f 0 = -1
def f_three_is_one := f 3 = 1

-- Define the solution set
def solution_set (f : ℝ → ℝ) := {x : ℝ | |f x| < 1}

-- State the theorem
theorem solution_set_is_open_interval
  (h_increasing : increasing_f f)
  (h_zero : f_zero_is_neg_one f)
  (h_three : f_three_is_one f) :
  solution_set f = Set.Ioo 0 3 :=
sorry

end solution_set_is_open_interval_l899_89903


namespace collinear_with_a_l899_89992

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

/-- Given vector a = (1, 2), prove that (k, 2k) is collinear with a for any non-zero real k -/
theorem collinear_with_a (k : ℝ) (hk : k ≠ 0) : 
  collinear (1, 2) (k, 2*k) := by
sorry

end collinear_with_a_l899_89992


namespace min_distance_to_line_l899_89908

theorem min_distance_to_line (x y : ℝ) (h : 2 * x - y - 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' - y' - 5 = 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
sorry

end min_distance_to_line_l899_89908


namespace zsigmondy_prime_l899_89920

theorem zsigmondy_prime (n : ℕ+) (p : ℕ) (k : ℕ) :
  3^(n : ℕ) - 2^(n : ℕ) = p^k → Nat.Prime p → Nat.Prime n := by
  sorry

end zsigmondy_prime_l899_89920


namespace only_D_is_certain_l899_89952

structure Event where
  name : String
  is_certain : Bool

def A : Event := { name := "Moonlight in front of the bed", is_certain := false }
def B : Event := { name := "Lonely smoke in the desert", is_certain := false }
def C : Event := { name := "Reach for the stars with your hand", is_certain := false }
def D : Event := { name := "Yellow River flows into the sea", is_certain := true }

def events : List Event := [A, B, C, D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by
  sorry

end only_D_is_certain_l899_89952


namespace expression_value_l899_89984

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end expression_value_l899_89984


namespace largest_divisor_of_consecutive_odd_product_l899_89983

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 105 → ¬(∀ (m : ℕ), Even m → m > 0 → 
    k ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) ∧ 
  (∀ (m : ℕ), Even m → m > 0 → 
    105 ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) :=
sorry

end largest_divisor_of_consecutive_odd_product_l899_89983


namespace cannot_form_triangle_l899_89989

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (2, 2, 6) cannot form a triangle -/
theorem cannot_form_triangle : ¬ can_form_triangle 2 2 6 := by
  sorry


end cannot_form_triangle_l899_89989


namespace expected_teachers_with_masters_l899_89905

def total_teachers : ℕ := 320
def masters_degree_ratio : ℚ := 1 / 4

theorem expected_teachers_with_masters :
  (total_teachers : ℚ) * masters_degree_ratio = 80 := by
  sorry

end expected_teachers_with_masters_l899_89905


namespace perpendicular_parallel_transitive_l899_89964

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Two lines are perpendicular if their direction vectors are orthogonal
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if their direction vectors are scalar multiples of each other
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Theorem statement
theorem perpendicular_parallel_transitive (l1 l2 l3 : Line3D) :
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end perpendicular_parallel_transitive_l899_89964


namespace circle_tangent_to_line_l899_89946

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define the symmetric point of (-2, 0) with respect to line l
def symmetric_point (a b : ℝ) : Prop :=
  (b - 0) / (a + 2) = -1 ∧ (a - (-2)) / 2 + b / 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

-- Define the tangency condition
def is_tangent (x y : ℝ) : Prop := line_l x y ∧ circle_equation x y

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), is_tangent x y ∧
  ∃ (a b : ℝ), symmetric_point a b ∧
  (x - a)^2 + (y - b)^2 = ((a + b) / Real.sqrt 2)^2 :=
sorry

end circle_tangent_to_line_l899_89946


namespace inequality_proof_l899_89906

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2) :
  (abs (a + b + c - a * b * c) ≤ 2) ∧
  (abs (a^3 + b^3 + c^3 - 3 * a * b * c) ≤ 2 * Real.sqrt 2) := by
  sorry

end inequality_proof_l899_89906


namespace problem_statement_l899_89933

theorem problem_statement (a b : ℝ) : 
  |a + b - 1| + Real.sqrt (2 * a + b - 2) = 0 → (b - a)^2023 = -1 := by
  sorry

end problem_statement_l899_89933


namespace train_length_l899_89978

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 21 → ∃ length : ℝ, abs (length - 350.07) < 0.01 := by
  sorry

end train_length_l899_89978


namespace total_amount_is_3200_l899_89965

/-- Proves that the total amount of money divided into two parts is 3200, given the problem conditions. -/
theorem total_amount_is_3200 
  (total : ℝ) -- Total amount of money
  (part1 : ℝ) -- First part of money (invested at 3%)
  (part2 : ℝ) -- Second part of money (invested at 5%)
  (h1 : part1 = 800) -- First part is Rs 800
  (h2 : part2 = total - part1) -- Second part is the remainder
  (h3 : 0.03 * part1 + 0.05 * part2 = 144) -- Total interest is Rs 144
  : total = 3200 :=
by sorry

end total_amount_is_3200_l899_89965


namespace smallest_cube_square_l899_89915

theorem smallest_cube_square (x : ℕ) (M : ℤ) : x = 11025 ↔ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(∃ N : ℤ, 2520 * y = N^3 ∧ ∃ z : ℕ, y = z^2)) ∧ 
  (∃ N : ℤ, 2520 * x = N^3) ∧ 
  (∃ z : ℕ, x = z^2) :=
sorry

end smallest_cube_square_l899_89915


namespace average_first_21_multiples_of_5_l899_89935

theorem average_first_21_multiples_of_5 : 
  let multiples := (fun i => 5 * i) 
  let sum := (List.range 21).map multiples |>.sum
  sum / 21 = 55 := by
sorry


end average_first_21_multiples_of_5_l899_89935


namespace square_roots_of_25_l899_89943

theorem square_roots_of_25 : Set ℝ := by
  -- Define the set of square roots of 25
  let roots : Set ℝ := {x : ℝ | x^2 = 25}
  
  -- Prove that this set is equal to {-5, 5}
  have h : roots = {-5, 5} := by sorry
  
  -- Return the set of square roots
  exact roots

end square_roots_of_25_l899_89943


namespace ice_cream_consumption_l899_89947

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25)
  (h2 : saturday_amount = 0.25) :
  friday_amount + saturday_amount = 3.50 := by
sorry

end ice_cream_consumption_l899_89947
