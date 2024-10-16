import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2729_272934

theorem simplify_expression : (81 ^ (1/4) - (33/4) ^ (1/2)) ^ 2 = (69 - 12 * 33 ^ (1/2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2729_272934


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2729_272967

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2729_272967


namespace NUMINAMATH_CALUDE_no_solution_implies_non_positive_product_l2729_272970

theorem no_solution_implies_non_positive_product (a b : ℝ) : 
  (∀ x : ℝ, (3*a + 8*b)*x + 7 ≠ 0) → a*b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_non_positive_product_l2729_272970


namespace NUMINAMATH_CALUDE_base_five_3214_equals_434_l2729_272959

def base_five_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base_five_3214_equals_434 :
  base_five_to_ten [4, 1, 2, 3] = 434 := by
  sorry

end NUMINAMATH_CALUDE_base_five_3214_equals_434_l2729_272959


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l2729_272918

/-- The speed difference of a bullet fired from a moving horse with wind assistance -/
theorem bullet_speed_difference
  (horse_speed : ℝ) 
  (bullet_speed : ℝ)
  (wind_speed : ℝ)
  (h1 : horse_speed = 20)
  (h2 : bullet_speed = 400)
  (h3 : wind_speed = 10) :
  (bullet_speed + horse_speed + wind_speed) - (bullet_speed - horse_speed - wind_speed) = 60 := by
  sorry


end NUMINAMATH_CALUDE_bullet_speed_difference_l2729_272918


namespace NUMINAMATH_CALUDE_watermelons_left_after_sales_l2729_272932

def initial_watermelons : ℕ := 10 * 12

def yesterday_sale_percentage : ℚ := 40 / 100

def today_sale_fraction : ℚ := 1 / 4

def tomorrow_sale_multiplier : ℚ := 3 / 2

def discount_threshold : ℕ := 10

theorem watermelons_left_after_sales : 
  let yesterday_sale := initial_watermelons * yesterday_sale_percentage
  let after_yesterday := initial_watermelons - yesterday_sale
  let today_sale := after_yesterday * today_sale_fraction
  let after_today := after_yesterday - today_sale
  let tomorrow_sale := today_sale * tomorrow_sale_multiplier
  after_today - tomorrow_sale = 27 := by sorry

end NUMINAMATH_CALUDE_watermelons_left_after_sales_l2729_272932


namespace NUMINAMATH_CALUDE_line_l_general_form_l2729_272975

/-- A line passing through point A(-2, 2) with the same y-intercept as y = x + 6 -/
def line_l (x y : ℝ) : Prop :=
  ∃ (m b : ℝ), y = m * x + b ∧ 2 = m * (-2) + b ∧ b = 6

/-- The general form equation of line l -/
def general_form (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- Theorem stating that the general form equation of line l is 2x - y + 6 = 0 -/
theorem line_l_general_form : 
  ∀ x y : ℝ, line_l x y ↔ general_form x y :=
sorry

end NUMINAMATH_CALUDE_line_l_general_form_l2729_272975


namespace NUMINAMATH_CALUDE_missed_number_sum_l2729_272994

theorem missed_number_sum (n : ℕ) (missing : ℕ) : 
  n = 63 → 
  missing ≤ n →
  (n * (n + 1)) / 2 - missing = 1991 →
  missing = 25 := by
sorry

end NUMINAMATH_CALUDE_missed_number_sum_l2729_272994


namespace NUMINAMATH_CALUDE_min_sum_squares_l2729_272941

/-- B-neighborhood of A is defined as the solution set of |x-A| < B -/
def neighborhood (A B : ℝ) := {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) : 
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 → 
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2729_272941


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_specific_ellipse_eccentricity_l2729_272942

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 is √(1 - b²/a²) -/
theorem ellipse_eccentricity (a b : ℝ) (h : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) :=
sorry

/-- The eccentricity of the ellipse x²/9 + y² = 1 is 2√2/3 -/
theorem specific_ellipse_eccentricity :
  let e := Real.sqrt (1 - 1^2 / 3^2)
  (∀ x y : ℝ, x^2 / 9 + y^2 = 1) →
  e = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_specific_ellipse_eccentricity_l2729_272942


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l2729_272988

/-- Given an ellipse and a hyperbola with coinciding foci, prove that d^2 = 215/16 -/
theorem ellipse_hyperbola_foci (d : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/d^2 = 1 ↔ x^2/169 - y^2/64 = 1/16) →
  d^2 = 215/16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l2729_272988


namespace NUMINAMATH_CALUDE_middle_frequency_is_32_l2729_272912

/-- Represents a frequency distribution histogram -/
structure Histogram where
  n : ℕ  -- number of rectangles
  middle_area : ℕ  -- area of the middle rectangle
  total_area : ℕ  -- total area of the histogram
  h_area_sum : middle_area + (n - 1) * middle_area = total_area  -- area sum condition
  h_total_area : total_area = 160  -- total area is 160

/-- The frequency of the middle group in the histogram is 32 -/
theorem middle_frequency_is_32 (h : Histogram) : h.middle_area = 32 := by
  sorry

end NUMINAMATH_CALUDE_middle_frequency_is_32_l2729_272912


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l2729_272978

theorem least_subtrahend_for_divisibility (n m : ℕ) (hn : n = 13602) (hm : m = 87) :
  ∃ (k : ℕ), k = 30 ∧ 
  (∀ (x : ℕ), x < k → ¬(∃ (q : ℕ), n - x = m * q)) ∧
  (∃ (q : ℕ), n - k = m * q) :=
sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l2729_272978


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l2729_272997

theorem not_divisible_by_nine (t : ℤ) (k : ℤ) (h : k = 9 * t + 8) :
  ¬ (9 ∣ (5 * (9 * t + 8) * (9 * 25 * t + 222))) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l2729_272997


namespace NUMINAMATH_CALUDE_length_CD_l2729_272951

theorem length_CD (AB D C : ℝ) : 
  AB = 48 →                 -- Length of AB is 48
  D = AB / 3 →              -- AD is 1/3 of AB
  C = AB / 2 →              -- C is the midpoint of AB
  C - D = 8 :=              -- Length of CD is 8
by
  sorry


end NUMINAMATH_CALUDE_length_CD_l2729_272951


namespace NUMINAMATH_CALUDE_trapezoid_bases_l2729_272972

theorem trapezoid_bases (d : ℝ) (l : ℝ) (h : d = 15 ∧ l = 17) :
  ∃ (b₁ b₂ : ℝ),
    b₁ = 9 ∧
    b₂ = 25 ∧
    b₁ + b₂ = 2 * l ∧
    b₂ - b₁ = 2 * Real.sqrt (l^2 - d^2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l2729_272972


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2729_272928

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 2 + p * (2 + Complex.I) + q = 0 →
  (2 - Complex.I : ℂ) ^ 2 + p * (2 - Complex.I) + q = 0 →
  p = -4 ∧ q = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2729_272928


namespace NUMINAMATH_CALUDE_seventh_triangular_number_l2729_272948

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_number : triangular_number 7 = 28 := by sorry

end NUMINAMATH_CALUDE_seventh_triangular_number_l2729_272948


namespace NUMINAMATH_CALUDE_remaining_tickets_proof_l2729_272939

def tickets_to_be_sold (total_tickets jude_tickets : ℕ) : ℕ :=
  let andrea_tickets := 6 * jude_tickets
  let sandra_tickets := 3 * jude_tickets + 10
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets)

theorem remaining_tickets_proof (total_tickets jude_tickets : ℕ) 
  (h1 : total_tickets = 300) 
  (h2 : jude_tickets = 24) : 
  tickets_to_be_sold total_tickets jude_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_proof_l2729_272939


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2729_272992

/-- The vertex coordinates of a parabola in the form y = (x - h)^2 + k are (h, k) -/
theorem parabola_vertex_coordinates (x y : ℝ) :
  y = (x - 3)^2 → (3, 0) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2729_272992


namespace NUMINAMATH_CALUDE_focus_coordinates_l2729_272925

/-- The parabola defined by the equation y = (1/8)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The focus of the parabola y = (1/8)x^2 -/
def focus_of_parabola : Focus := { x := 0, y := 2 }

/-- Theorem: The focus of the parabola y = (1/8)x^2 is (0, 2) -/
theorem focus_coordinates :
  focus_of_parabola.x = 0 ∧ focus_of_parabola.y = 2 :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l2729_272925


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2729_272937

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (b : ℝ) :
  isPureImaginary ((1 + b * Complex.I) * (2 - Complex.I)) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2729_272937


namespace NUMINAMATH_CALUDE_hall_volume_l2729_272949

theorem hall_volume (length width height : ℝ) : 
  length = 15 ∧ 
  width = 12 ∧ 
  2 * (length * width) = 2 * (length * height) + 2 * (width * height) → 
  length * width * height = 8004 :=
by sorry

end NUMINAMATH_CALUDE_hall_volume_l2729_272949


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l2729_272979

theorem smallest_denominator_between_fractions : 
  ∃ (p q : ℕ), 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    q = 4027 ∧
    (∀ (p' q' : ℕ), (1 : ℚ) / 2014 < (p' : ℚ) / q' → (p' : ℚ) / q' < (1 : ℚ) / 2013 → q ≤ q') :=
by sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l2729_272979


namespace NUMINAMATH_CALUDE_arrangements_count_l2729_272914

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 4

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of arrangements for the given scenario -/
def num_arrangements : ℕ := 
  (choose num_teachers teachers_on_monday) * (Nat.factorial (num_teachers - teachers_on_monday))

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l2729_272914


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l2729_272913

theorem triangle_circumcircle_intersection (PQ QR RP : ℝ) (h1 : PQ = 39) (h2 : QR = 15) (h3 : RP = 50) : 
  ∃ (PS : ℝ), PS = 5 * Real.sqrt 61 ∧ 
  ⌊5 + Real.sqrt 61⌋ = 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l2729_272913


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2729_272990

/-- Calculates the profit percentage given cost price, marked price, and discount rate. -/
def profit_percentage (cost_price marked_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 25% for the given conditions. -/
theorem profit_percentage_is_25_percent :
  profit_percentage (47.50 : ℚ) (62.5 : ℚ) (0.05 : ℚ) = 25 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2729_272990


namespace NUMINAMATH_CALUDE_fraction_equality_l2729_272999

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2729_272999


namespace NUMINAMATH_CALUDE_super_ball_distance_l2729_272905

def bounce_height (initial_height : ℝ) (bounce_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ n)

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (num_bounces : ℕ) : ℝ :=
  let descents := initial_height + (Finset.sum (Finset.range num_bounces) (λ i => bounce_height initial_height bounce_factor i))
  let ascents := Finset.sum (Finset.range (num_bounces + 1)) (λ i => bounce_height initial_height bounce_factor i)
  descents + ascents

theorem super_ball_distance :
  total_distance 20 0.6 4 = 69.632 := by
  sorry

end NUMINAMATH_CALUDE_super_ball_distance_l2729_272905


namespace NUMINAMATH_CALUDE_birthday_product_difference_l2729_272908

theorem birthday_product_difference (n : ℕ) (h : n = 7) : (n + 1)^2 - n^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_product_difference_l2729_272908


namespace NUMINAMATH_CALUDE_point_above_line_l2729_272922

theorem point_above_line (a : ℝ) : 
  (2*a - (-1) + 1 < 0) ↔ (a < -1) := by sorry

end NUMINAMATH_CALUDE_point_above_line_l2729_272922


namespace NUMINAMATH_CALUDE_purely_imaginary_product_l2729_272931

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I * (x^4 + 2*x^3 + x^2 + 2*x) = (x + Complex.I) * ((x^2 + 1) + Complex.I) * ((x + 2) + Complex.I)) ↔ 
  (x = 0 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_product_l2729_272931


namespace NUMINAMATH_CALUDE_product_in_N_not_in_M_l2729_272957

def M : Set ℤ := {x | ∃ m : ℤ, x = 3*m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 2}

theorem product_in_N_not_in_M (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) :
  (x * y) ∈ N ∧ (x * y) ∉ M := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_not_in_M_l2729_272957


namespace NUMINAMATH_CALUDE_simplify_fraction_l2729_272969

theorem simplify_fraction : (90 : ℚ) / 8100 = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2729_272969


namespace NUMINAMATH_CALUDE_vector_orthogonality_l2729_272910

theorem vector_orthogonality (a b c : ℝ × ℝ) (x : ℝ) : 
  a = (1, 2) → b = (1, 0) → c = (3, 4) → 
  (b.1 + x * a.1, b.2 + x * a.2) • c = 0 → 
  x = -3/11 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l2729_272910


namespace NUMINAMATH_CALUDE_fraction_inequality_l2729_272936

theorem fraction_inequality (x : ℝ) : 
  -4 ≤ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3) / (2*x^2 + 2*x + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2729_272936


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l2729_272911

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l2729_272911


namespace NUMINAMATH_CALUDE_hexagon_properties_l2729_272986

/-- A regular hexagon with diagonals -/
structure RegularHexagonWithDiagonals where
  /-- The area of the regular hexagon -/
  area : ℝ
  /-- The hexagon is regular -/
  is_regular : Bool
  /-- All diagonals are drawn -/
  diagonals_drawn : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : RegularHexagonWithDiagonals) : ℕ := sorry

/-- The area of the new regular hexagon formed by combining all quadrilateral parts -/
def new_hexagon_area (h : RegularHexagonWithDiagonals) : ℝ := sorry

/-- Theorem about the properties of a regular hexagon with diagonals -/
theorem hexagon_properties (h : RegularHexagonWithDiagonals) 
  (h_area : h.area = 144)
  (h_regular : h.is_regular = true)
  (h_diagonals : h.diagonals_drawn = true) :
  num_parts h = 24 ∧ new_hexagon_area h = 48 := by sorry

end NUMINAMATH_CALUDE_hexagon_properties_l2729_272986


namespace NUMINAMATH_CALUDE_fibonacci_type_repeated_values_l2729_272993

def FibonacciType (x : ℕ → ℝ) (A B : ℝ) :=
  x 0 = A ∧ x 1 = B ∧ ∀ n, x (n + 2) = x (n + 1) + x n

def RepeatedValue (x : ℕ → ℝ) (C : ℝ) :=
  ∃ t s, t ≠ s ∧ x t = C ∧ x s = C

theorem fibonacci_type_repeated_values :
  (∀ k : ℕ, ∃ A B : ℝ, ∃ x : ℕ → ℝ,
    FibonacciType x A B ∧ (∃ C : ℝ, ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ n ∈ S, x n = C)) ∧
  (¬ ∃ A B : ℝ, ∃ x : ℕ → ℝ,
    FibonacciType x A B ∧ (∃ C : ℝ, ∀ k : ℕ, ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ n ∈ S, x n = C)) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_type_repeated_values_l2729_272993


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2729_272996

/-- An isosceles triangle with sides of 5 and 10 has a perimeter of 25. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∧ b = 10 ∧ c = 10) →  -- Isosceles triangle with sides 5 and 10
    (a + b + c = 25)             -- Perimeter is 25

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 5 10 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2729_272996


namespace NUMINAMATH_CALUDE_factorization_sum_l2729_272906

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 20*x + 75 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 22*x + 120 = (x - e)*(x - f)) →
  d + e + f = 37 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l2729_272906


namespace NUMINAMATH_CALUDE_uncovered_area_l2729_272920

/-- The area not covered by a smaller square and a right triangle within a larger square -/
theorem uncovered_area (larger_side small_side triangle_base triangle_height : ℝ) 
  (h1 : larger_side = 10)
  (h2 : small_side = 4)
  (h3 : triangle_base = 3)
  (h4 : triangle_height = 3)
  : larger_side ^ 2 - (small_side ^ 2 + (triangle_base * triangle_height) / 2) = 79.5 := by
  sorry

#check uncovered_area

end NUMINAMATH_CALUDE_uncovered_area_l2729_272920


namespace NUMINAMATH_CALUDE_sets_properties_l2729_272954

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 < a ∧ a < 8)) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l2729_272954


namespace NUMINAMATH_CALUDE_parabola_vertex_l2729_272938

/-- The parabola defined by y = -x^2 + 2x + 3 has its vertex at the point (1, 4). -/
theorem parabola_vertex (x y : ℝ) : 
  y = -x^2 + 2*x + 3 → (1, 4) = (x, y) ∨ ∃ t : ℝ, y < -t^2 + 2*t + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2729_272938


namespace NUMINAMATH_CALUDE_inequality_contradiction_l2729_272904

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l2729_272904


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2729_272998

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3) ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = Real.sqrt (Real.sqrt 3) ∧ b = Real.sqrt (Real.sqrt 3) ∧ c = Real.sqrt (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2729_272998


namespace NUMINAMATH_CALUDE_ribbon_pieces_l2729_272977

theorem ribbon_pieces (original_length : ℝ) (piece_length : ℝ) (remaining_length : ℝ) : 
  original_length = 51 →
  piece_length = 0.15 →
  remaining_length = 36 →
  (original_length - remaining_length) / piece_length = 100 := by
sorry

end NUMINAMATH_CALUDE_ribbon_pieces_l2729_272977


namespace NUMINAMATH_CALUDE_sqrt_problems_l2729_272940

-- Define the arithmetic square root
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define the square root function that returns a set
def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

theorem sqrt_problems :
  (∀ x : ℝ, x > 0 → arithmeticSqrt x ≥ 0) ∧
  (squareRoot 81 = {9, -9}) ∧
  (|2 - Real.sqrt 5| = Real.sqrt 5 - 2) ∧
  (Real.sqrt (4/121) = 2/11) ∧
  (2 * Real.sqrt 3 - 5 * Real.sqrt 3 = -3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_problems_l2729_272940


namespace NUMINAMATH_CALUDE_correct_calculation_l2729_272983

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2729_272983


namespace NUMINAMATH_CALUDE_work_increase_with_absence_l2729_272971

theorem work_increase_with_absence (p : ℕ) (W : ℝ) (h : p > 0) :
  let initial_work_per_person := W / p
  let remaining_persons := (2 : ℝ) / 3 * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - initial_work_per_person = W / (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absence_l2729_272971


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2729_272995

/-- Given an ellipse with equation 25x^2 - 100x + 4y^2 + 8y + 4 = 0, 
    the distance between its foci is 2√21. -/
theorem ellipse_foci_distance (x y : ℝ) : 
  25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 4 = 0 → 
  ∃ (c : ℝ), c = 2 * Real.sqrt 21 ∧ 
  c = (distance_between_foci : ℝ → ℝ) (25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 4) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_distance_l2729_272995


namespace NUMINAMATH_CALUDE_equation_solution_exists_and_unique_l2729_272929

/-- Represents a digit assignment for the equation TIXO + TIGR = SPIT -/
structure DigitAssignment where
  T : Nat
  I : Nat
  X : Nat
  O : Nat
  G : Nat
  R : Nat
  S : Nat
  P : Nat
  all_digits : List Nat := [T, I, X, O, G, R, S, P]

/-- Checks if all digits in the assignment are unique and non-zero -/
def valid_assignment (a : DigitAssignment) : Prop :=
  a.all_digits.all (λ d => d > 0 ∧ d ≤ 9) ∧ a.all_digits.Nodup

/-- Converts a four-digit number represented by individual digits to its numerical value -/
def to_number (a b c d : Nat) : Nat := 1000 * a + 100 * b + 10 * c + d

/-- Theorem stating the existence and uniqueness of the solution -/
theorem equation_solution_exists_and_unique :
  ∃! (a : DigitAssignment),
    valid_assignment a ∧
    to_number a.T a.I a.G a.R = 1345 ∧
    to_number a.T a.I a.X a.O = 1386 ∧
    to_number a.S a.P a.I a.T = 2731 :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_and_unique_l2729_272929


namespace NUMINAMATH_CALUDE_figure_b_cannot_be_formed_l2729_272973

/-- A piece is represented by its width and height -/
structure Piece where
  width : ℕ
  height : ℕ

/-- A figure is represented by its width and height -/
structure Figure where
  width : ℕ
  height : ℕ

/-- The set of available pieces -/
def pieces : Finset Piece := sorry

/-- The set of figures to be formed -/
def figures : Finset Figure := sorry

/-- Function to check if a figure can be formed from the given pieces -/
def canFormFigure (p : Finset Piece) (f : Figure) : Prop := sorry

/-- Theorem stating that Figure B cannot be formed while others can -/
theorem figure_b_cannot_be_formed :
  ∃ (b : Figure),
    b ∈ figures ∧
    ¬(canFormFigure pieces b) ∧
    ∀ (f : Figure), f ∈ figures ∧ f ≠ b → canFormFigure pieces f :=
sorry

end NUMINAMATH_CALUDE_figure_b_cannot_be_formed_l2729_272973


namespace NUMINAMATH_CALUDE_partnership_theorem_l2729_272968

def partnership (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) : Prop :=
  let b_share : ℝ := (1 / 4) * total_capital
  let c_share : ℝ := (1 / 5) * total_capital
  let d_share : ℝ := total_capital - (b_share + c_share + (83 / 249) * total_capital)
  let a_share : ℝ := (83 / 249) * total_capital
  (a_profit / total_profit = 83 / 249) ∧
  (b_share + c_share + d_share + a_share = total_capital) ∧
  (d_share ≥ 0)

theorem partnership_theorem (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) 
  (h1 : total_capital > 0)
  (h2 : total_profit = 2490)
  (h3 : a_profit = 830) :
  partnership total_capital total_profit a_profit :=
by
  sorry

end NUMINAMATH_CALUDE_partnership_theorem_l2729_272968


namespace NUMINAMATH_CALUDE_min_value_expression1_min_value_expression2_min_value_expression3_l2729_272909

/-- The minimum value of x^2 + y^2 + xy + x + y for real x and y is -1/3 -/
theorem min_value_expression1 :
  ∃ (m : ℝ), m = -1/3 ∧ ∀ (x y : ℝ), x^2 + y^2 + x*y + x + y ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + xy + yz + zx + x + y + z for real x, y, and z is -3/8 -/
theorem min_value_expression2 :
  ∃ (m : ℝ), m = -3/8 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + x*y + y*z + z*x + x + y + z ≥ m :=
sorry

/-- The minimum value of x^2 + y^2 + z^2 + r^2 + xy + xz + xr + yz + yr + zr + x + y + z + r for real x, y, z, and r is -2/5 -/
theorem min_value_expression3 :
  ∃ (m : ℝ), m = -2/5 ∧ ∀ (x y z r : ℝ),
    x^2 + y^2 + z^2 + r^2 + x*y + x*z + x*r + y*z + y*r + z*r + x + y + z + r ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression1_min_value_expression2_min_value_expression3_l2729_272909


namespace NUMINAMATH_CALUDE_inequality_holds_l2729_272974

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

-- State the theorem
theorem inequality_holds (a b : ℝ) (ha : a > 2) (hb : b > 0) :
  ∀ x, |x + 1| < b → |2 * f x - 4| < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2729_272974


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_eight_l2729_272987

theorem reciprocal_of_negative_eight :
  ∃ x : ℚ, x * (-8) = 1 ∧ x = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_eight_l2729_272987


namespace NUMINAMATH_CALUDE_flag_raising_arrangements_l2729_272923

/-- The number of classes in the first year of high school -/
def first_year_classes : ℕ := 8

/-- The number of classes in the second year of high school -/
def second_year_classes : ℕ := 6

/-- The total number of possible arrangements for selecting one class for flag-raising duty -/
def total_arrangements : ℕ := first_year_classes + second_year_classes

/-- Theorem stating that the total number of possible arrangements is 14 -/
theorem flag_raising_arrangements :
  total_arrangements = 14 := by sorry

end NUMINAMATH_CALUDE_flag_raising_arrangements_l2729_272923


namespace NUMINAMATH_CALUDE_cube_condition_l2729_272956

theorem cube_condition (n : ℤ) : 
  (∃ k : ℤ, 6 * n + 2 = k ^ 3) ↔ 
  (∃ m : ℤ, n = 36 * m ^ 3 + 36 * m ^ 2 + 12 * m + 1) := by
sorry

end NUMINAMATH_CALUDE_cube_condition_l2729_272956


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l2729_272991

/-- Represents the number of vehicles sold -/
structure VehicleSales where
  sports_cars : ℕ
  sedans : ℕ
  trucks : ℕ

/-- The ratio of vehicles sold -/
def sales_ratio : VehicleSales := ⟨3, 5, 4⟩

/-- The anticipated number of sports cars to be sold -/
def anticipated_sports_cars : ℕ := 36

/-- Calculates the number of sedans sold based on the ratio and sports cars sold -/
def calculate_sedans (ratio : VehicleSales) (sports_cars : ℕ) : ℕ :=
  (ratio.sedans * sports_cars) / ratio.sports_cars

/-- Calculates the number of trucks sold based on the ratio and sports cars sold -/
def calculate_trucks (ratio : VehicleSales) (sports_cars : ℕ) : ℕ :=
  (ratio.trucks * sports_cars) / ratio.sports_cars

theorem dealership_sales_prediction :
  calculate_sedans sales_ratio anticipated_sports_cars = 60 ∧
  calculate_trucks sales_ratio anticipated_sports_cars = 48 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l2729_272991


namespace NUMINAMATH_CALUDE_money_calculation_l2729_272921

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 : ℕ) (n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that given 72 total notes with 57 being 50 rupee notes, the total amount is 10350 rupees -/
theorem money_calculation : total_amount 57 (72 - 57) = 10350 := by
  sorry

end NUMINAMATH_CALUDE_money_calculation_l2729_272921


namespace NUMINAMATH_CALUDE_units_digit_of_3542_to_876_l2729_272946

theorem units_digit_of_3542_to_876 : ∃ n : ℕ, 3542^876 ≡ 6 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_3542_to_876_l2729_272946


namespace NUMINAMATH_CALUDE_weight_of_scaled_object_l2729_272953

/-- Given two similar three-dimensional objects where one has all dimensions 3 times
    larger than the other, if the smaller object weighs 10 grams, 
    then the larger object weighs 270 grams. -/
theorem weight_of_scaled_object (weight_small : ℝ) (scale_factor : ℝ) :
  weight_small = 10 →
  scale_factor = 3 →
  weight_small * scale_factor^3 = 270 := by
sorry


end NUMINAMATH_CALUDE_weight_of_scaled_object_l2729_272953


namespace NUMINAMATH_CALUDE_equation_study_path_and_future_l2729_272943

/-- Represents the steps in the study path of equations -/
inductive StudyPath
  | Definition
  | Solution
  | Solving
  | Application

/-- Represents types of equations that may be studied in the future -/
inductive FutureEquation
  | LinearQuadratic
  | LinearCubic
  | SystemQuadratic

/-- Represents an example of a future equation -/
def futureEquationExample : ℝ → ℝ := fun x => x^3 + 2*x + 1

/-- Theorem stating the study path of equations and future equations to be studied -/
theorem equation_study_path_and_future :
  (∃ (path : List StudyPath), path = [StudyPath.Definition, StudyPath.Solution, StudyPath.Solving, StudyPath.Application]) ∧
  (∃ (future : List FutureEquation), future = [FutureEquation.LinearQuadratic, FutureEquation.LinearCubic, FutureEquation.SystemQuadratic]) ∧
  (∃ (x : ℝ), futureEquationExample x = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_study_path_and_future_l2729_272943


namespace NUMINAMATH_CALUDE_frac_5_13_150th_digit_l2729_272930

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit_after_decimal (n d : ℕ) (k : ℕ) : ℕ := sorry

theorem frac_5_13_150th_digit :
  nth_digit_after_decimal 5 13 150 = 5 := by sorry

end NUMINAMATH_CALUDE_frac_5_13_150th_digit_l2729_272930


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l2729_272915

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 15 → initial_participants = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l2729_272915


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l2729_272958

theorem ones_digit_of_large_power (n : ℕ) : 
  (35^(35*(17^17)) : ℕ) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l2729_272958


namespace NUMINAMATH_CALUDE_ali_baba_cave_entry_l2729_272947

theorem ali_baba_cave_entry (n : ℕ) (h : n = 28) :
  ∃ (m : ℕ), m ≤ 11 ∧
  ∀ (counters : Finset ℕ),
    counters.card = n →
    (∀ x ∈ counters, 1 ≤ x ∧ x ≤ 2017) →
    ∃ (moves : List (ℕ × Finset ℕ)),
      moves.length = m ∧
      (∀ (counter : ℕ), counter ∈ counters →
        counter - (moves.map (λ (move : ℕ × Finset ℕ) => if counter ∈ move.2 then move.1 else 0)).sum = 0) :=
by sorry

#check ali_baba_cave_entry

end NUMINAMATH_CALUDE_ali_baba_cave_entry_l2729_272947


namespace NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l2729_272926

theorem sin_cos_difference_36_degrees : 
  Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_36_degrees_l2729_272926


namespace NUMINAMATH_CALUDE_second_largest_of_five_consecutive_sum_90_l2729_272989

theorem second_largest_of_five_consecutive_sum_90 (a b c d e : ℕ) : 
  (a + b + c + d + e = 90) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  d = 19 := by
sorry

end NUMINAMATH_CALUDE_second_largest_of_five_consecutive_sum_90_l2729_272989


namespace NUMINAMATH_CALUDE_exponent_equality_l2729_272981

theorem exponent_equality (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 27) : 
  x = 28 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2729_272981


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l2729_272963

/-- The number of ways to get at least 6 consecutive heads in 9 coin flips -/
def consecutiveHeadsCount : ℕ := 49

/-- The total number of possible outcomes when flipping a coin 9 times -/
def totalOutcomes : ℕ := 512

/-- A fair coin is flipped 9 times. This theorem states that the probability
    of getting at least 6 consecutive heads is 49/512. -/
theorem probability_six_consecutive_heads :
  (consecutiveHeadsCount : ℚ) / totalOutcomes = 49 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l2729_272963


namespace NUMINAMATH_CALUDE_sam_initial_nickels_l2729_272927

/-- Given information about Sam's nickels --/
structure SamNickels where
  initial : ℕ  -- Initial number of nickels
  given : ℕ    -- Number of nickels given by dad
  final : ℕ    -- Final number of nickels

/-- Theorem stating the initial number of nickels Sam had --/
theorem sam_initial_nickels (s : SamNickels) (h : s.final = s.initial + s.given) 
  (h_final : s.final = 63) (h_given : s.given = 39) : s.initial = 24 := by
  sorry

#check sam_initial_nickels

end NUMINAMATH_CALUDE_sam_initial_nickels_l2729_272927


namespace NUMINAMATH_CALUDE_time_per_check_is_two_minutes_l2729_272933

/-- The time per check for lice checks at an elementary school -/
def time_per_check : ℕ :=
  let kindergarteners : ℕ := 26
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let total_students : ℕ := kindergarteners + first_graders + second_graders + third_graders
  let total_time_hours : ℕ := 3
  let total_time_minutes : ℕ := total_time_hours * 60
  total_time_minutes / total_students

/-- Theorem stating that the time per check is 2 minutes -/
theorem time_per_check_is_two_minutes : time_per_check = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_per_check_is_two_minutes_l2729_272933


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2729_272962

theorem min_value_quadratic (x y : ℝ) (h : x + y = 5) :
  x^2 - x*y + y^2 ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ x₀^2 - x₀*y₀ + y₀^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2729_272962


namespace NUMINAMATH_CALUDE_natural_number_pairs_l2729_272907

theorem natural_number_pairs (a b : ℕ) :
  (∃ k : ℕ, b - 1 = k * (a + 1)) →
  (∃ m : ℕ, a^2 + a + 2 = m * b) →
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * k^2 + 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l2729_272907


namespace NUMINAMATH_CALUDE_kathleen_bottle_caps_l2729_272916

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := 5

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := store_visits * bottle_caps_per_visit

theorem kathleen_bottle_caps : total_bottle_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_bottle_caps_l2729_272916


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2729_272900

theorem floor_ceil_sum : ⌊(-3.276 : ℝ)⌋ + ⌈(-17.845 : ℝ)⌉ = -21 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2729_272900


namespace NUMINAMATH_CALUDE_fill_large_bottle_l2729_272901

/-- The volume of shampoo in milliliters that a medium-sized bottle holds -/
def medium_bottle_volume : ℕ := 150

/-- The volume of shampoo in milliliters that a large bottle holds -/
def large_bottle_volume : ℕ := 1200

/-- The number of medium-sized bottles needed to fill a large bottle -/
def bottles_needed : ℕ := large_bottle_volume / medium_bottle_volume

theorem fill_large_bottle : bottles_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_fill_large_bottle_l2729_272901


namespace NUMINAMATH_CALUDE_purple_chip_value_l2729_272944

def blue_value : ℕ := 1
def green_value : ℕ := 5
def red_value : ℕ := 11

def is_valid_product (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), blue_value^a * green_value^b * x^c * red_value^d = 28160

theorem purple_chip_value :
  ∀ x : ℕ,
  green_value < x →
  x < red_value →
  is_valid_product x →
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_purple_chip_value_l2729_272944


namespace NUMINAMATH_CALUDE_mikes_cards_l2729_272902

theorem mikes_cards (x : ℕ) : x + 18 = 82 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_mikes_cards_l2729_272902


namespace NUMINAMATH_CALUDE_expression_evaluation_l2729_272950

theorem expression_evaluation (a b : ℤ) (ha : a = 1) (hb : b = -1) :
  5 * a * b^2 - (3 * a * b + 2 * (-2 * a * b^2 + a * b)) = 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2729_272950


namespace NUMINAMATH_CALUDE_total_fish_cost_l2729_272966

def fish_cost : ℕ := 4
def dog_fish : ℕ := 40
def cat_fish : ℕ := dog_fish / 2

theorem total_fish_cost : dog_fish * fish_cost + cat_fish * fish_cost = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_cost_l2729_272966


namespace NUMINAMATH_CALUDE_problem_statement_l2729_272955

theorem problem_statement (x y : ℝ) 
  (hx : x * (Real.exp x + Real.log x + x) = 1)
  (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  0 < x ∧ x < 1 ∧ y - x > 1 ∧ y - x < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2729_272955


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2729_272903

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2729_272903


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2729_272984

-- Define the polynomial
def polynomial (a b c d : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the root
def root : ℂ := 2 + Complex.I

-- Theorem statement
theorem sum_of_coefficients (a b c d : ℝ) : 
  polynomial a b c d root = 0 → a + b + c + d = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2729_272984


namespace NUMINAMATH_CALUDE_max_value_fraction_l2729_272924

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (∀ a b : ℝ, -4 ≤ a ∧ a ≤ -2 → 2 ≤ b ∧ b ≤ 4 → (x + y) / x ≥ (a + b) / a) →
  (x + y) / x = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2729_272924


namespace NUMINAMATH_CALUDE_student_number_problem_l2729_272917

theorem student_number_problem : ∃ x : ℤ, 2 * x - 138 = 110 ∧ x = 124 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2729_272917


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l2729_272982

def inverse_proportion (x y : ℝ) : Prop := y = -4 / x

theorem inverse_proportion_points :
  inverse_proportion (-2) 2 ∧
  ¬ inverse_proportion 1 4 ∧
  ¬ inverse_proportion (-2) (-2) ∧
  ¬ inverse_proportion (-4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l2729_272982


namespace NUMINAMATH_CALUDE_certain_number_proof_l2729_272960

theorem certain_number_proof : ∃ x : ℝ, 45 * 12 = 0.60 * x ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2729_272960


namespace NUMINAMATH_CALUDE_polynomial_value_at_2_l2729_272964

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 8

-- Theorem statement
theorem polynomial_value_at_2 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_2_l2729_272964


namespace NUMINAMATH_CALUDE_workshop_workers_l2729_272985

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (other_salary : ℕ) :
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 20000 →
  other_salary = 6000 →
  ∃ (total_workers : ℕ),
    total_workers = technician_count + (technician_count * technician_salary + (total_workers - technician_count) * other_salary) / average_salary ∧
    total_workers = 49 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2729_272985


namespace NUMINAMATH_CALUDE_min_m_for_log_triangle_l2729_272919

/-- The minimum value of M such that for any right-angled triangle with sides a, b, c > M,
    the logarithms of these sides also form a triangle. -/
theorem min_m_for_log_triangle : ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ (a b c : ℝ), a > M → b > M → c > M →
    a^2 + b^2 = c^2 →
    Real.log a + Real.log b > Real.log c) ∧
  (∀ (M' : ℝ), M' < M →
    ∃ (a b c : ℝ), a > M' ∧ b > M' ∧ c > M' ∧
      a^2 + b^2 = c^2 ∧
      Real.log a + Real.log b ≤ Real.log c) :=
by sorry

end NUMINAMATH_CALUDE_min_m_for_log_triangle_l2729_272919


namespace NUMINAMATH_CALUDE_water_bottle_calculation_l2729_272935

/-- Given an initial number of bottles, calculate the final number after removing some and adding more. -/
def final_bottles (initial remove add : ℕ) : ℕ :=
  initial - remove + add

/-- Theorem: Given 14 initial bottles, removing 8 and adding 45 results in 51 bottles. -/
theorem water_bottle_calculation :
  final_bottles 14 8 45 = 51 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_calculation_l2729_272935


namespace NUMINAMATH_CALUDE_no_solution_for_seven_power_plus_cube_divisible_by_nine_l2729_272961

theorem no_solution_for_seven_power_plus_cube_divisible_by_nine :
  ∀ n : ℕ, n ≥ 1 → ¬(9 ∣ 7^n + n^3) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_seven_power_plus_cube_divisible_by_nine_l2729_272961


namespace NUMINAMATH_CALUDE_stating_orthogonal_parallelepiped_angle_properties_l2729_272980

/-- 
Represents the angles formed between the diagonal and the edges of an orthogonal parallelepiped.
-/
structure ParallelepipedAngles where
  α₁ : ℝ
  α₂ : ℝ
  α₃ : ℝ

/-- 
Theorem stating the properties of angles in an orthogonal parallelepiped.
-/
theorem orthogonal_parallelepiped_angle_properties (angles : ParallelepipedAngles) :
  Real.sin angles.α₁ ^ 2 + Real.sin angles.α₂ ^ 2 + Real.sin angles.α₃ ^ 2 = 1 ∧
  Real.cos angles.α₁ ^ 2 + Real.cos angles.α₂ ^ 2 + Real.cos angles.α₃ ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_orthogonal_parallelepiped_angle_properties_l2729_272980


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2729_272952

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_is_term (a : ℕ → ℕ) : Prop :=
  ∀ p s, ∃ t, a p + a s = a t

theorem arithmetic_sequence_property (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d →
  a 1 = 12 →
  d > 0 →
  sum_is_term a →
  d = 6 ∨ d = 3 ∨ d = 2 ∨ d = 1 →
  d = 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2729_272952


namespace NUMINAMATH_CALUDE_faster_train_speed_l2729_272965

theorem faster_train_speed 
  (train_length : ℝ) 
  (speed_difference : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : speed_difference = 36) 
  (h3 : passing_time = 54) : 
  ∃ (faster_speed : ℝ), faster_speed = 46 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2729_272965


namespace NUMINAMATH_CALUDE_total_people_in_program_l2729_272976

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l2729_272976


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2729_272945

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 3*y = 1) :
  (1/x + 1/(3*y)) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2729_272945
