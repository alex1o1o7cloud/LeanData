import Mathlib

namespace NUMINAMATH_CALUDE_even_factors_count_l3060_306087

def n : ℕ := 2^3 * 3^2 * 7^3 * 5^1

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count : num_even_factors n = 72 := by sorry

end NUMINAMATH_CALUDE_even_factors_count_l3060_306087


namespace NUMINAMATH_CALUDE_milk_tea_sales_l3060_306067

theorem milk_tea_sales (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total + (3 : ℚ) / 10 * total + 15 = total) 
  (h2 : (3 : ℚ) / 10 * total = 15) : total = 50 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_l3060_306067


namespace NUMINAMATH_CALUDE_bakery_chairs_l3060_306047

/-- The number of chairs in a bakery -/
def total_chairs (indoor_tables outdoor_tables chairs_per_indoor_table chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Proof that the total number of chairs in the bakery is 60 -/
theorem bakery_chairs :
  total_chairs 8 12 3 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bakery_chairs_l3060_306047


namespace NUMINAMATH_CALUDE_coin_problem_l3060_306026

def is_valid_amount (n : ℕ) : Prop :=
  ∃ (x : ℕ), 
    n = 5 * x ∧ 
    n ≤ 100000 ∧ 
    x % 12 = 3 ∧ 
    x % 18 = 3 ∧ 
    x % 45 = 3 ∧ 
    x % 11 = 0

def valid_amounts : Set ℕ :=
  {1815, 11715, 21615, 31515, 41415, 51315, 61215, 71115, 81015, 90915}

theorem coin_problem : 
  ∀ n : ℕ, is_valid_amount n ↔ n ∈ valid_amounts :=
sorry

end NUMINAMATH_CALUDE_coin_problem_l3060_306026


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3060_306098

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3060_306098


namespace NUMINAMATH_CALUDE_cube_root_difference_l3060_306090

theorem cube_root_difference : (8 : ℝ) ^ (1/3) - (343 : ℝ) ^ (1/3) = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_difference_l3060_306090


namespace NUMINAMATH_CALUDE_half_number_plus_seven_l3060_306094

theorem half_number_plus_seven (n : ℝ) : n = 20 → (n / 2) + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_half_number_plus_seven_l3060_306094


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l3060_306016

/-- Given two fixed points in a metric space, the set of points whose sum of distances to these fixed points equals the distance between the fixed points is equal to the set containing only the fixed points. -/
theorem trajectory_is_line_segment {α : Type*} [MetricSpace α] (F₁ F₂ : α) (h : dist F₁ F₂ = 8) :
  {M : α | dist M F₁ + dist M F₂ = 8} = {F₁, F₂} := by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l3060_306016


namespace NUMINAMATH_CALUDE_symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l3060_306085

/-- A function f: ℝ → ℝ is symmetric about the origin if f(-x) = -f(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem symmetric_implies_abs_even (f : ℝ → ℝ) :
  SymmetricAboutOrigin f → EvenFunction (fun x ↦ |f x|) :=
by sorry

theorem abs_even_not_sufficient_for_symmetric :
  ∃ f : ℝ → ℝ, EvenFunction (fun x ↦ |f x|) ∧ ¬SymmetricAboutOrigin f :=
by sorry

end NUMINAMATH_CALUDE_symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l3060_306085


namespace NUMINAMATH_CALUDE_length_EM_is_sqrt_6_l3060_306034

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the parabola y² = 4x -/
def parabola : Parabola :=
  { p := 1
  , focus := (1, 0) }

/-- Definition of the line l passing through F and intersecting the parabola -/
def line_l : Line :=
  sorry

/-- Definition of points A and B where line l intersects the parabola -/
def point_A : Point :=
  sorry

def point_B : Point :=
  sorry

/-- Definition of point E (foot of the perpendicular) -/
def point_E : Point :=
  sorry

/-- Definition of point M (intersection of perpendicular bisector with x-axis) -/
def point_M : Point :=
  sorry

/-- Statement: The length of EM is √6 -/
theorem length_EM_is_sqrt_6 (h : Real.sqrt ((point_E.x - point_M.x)^2 + (point_E.y - point_M.y)^2) = Real.sqrt 6) :
  Real.sqrt ((point_E.x - point_M.x)^2 + (point_E.y - point_M.y)^2) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_length_EM_is_sqrt_6_l3060_306034


namespace NUMINAMATH_CALUDE_line_equation_proof_l3060_306068

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  x₀ : ℝ
  y₀ : ℝ
  passes_through : a * x₀ + b * y₀ + c = 0

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equation_proof :
  ∃ (l : Line),
    l.x₀ = 1 ∧
    l.y₀ = 2 ∧
    l.a = 1 ∧
    l.b = 2 ∧
    l.c = -5 ∧
    perpendicular l ⟨2, -1, 1, 0, 0, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3060_306068


namespace NUMINAMATH_CALUDE_power_of_product_l3060_306036

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3060_306036


namespace NUMINAMATH_CALUDE_mario_haircut_price_l3060_306001

/-- The price of a haircut on a weekday -/
def weekday_price : ℝ := 18

/-- The price of a haircut on a weekend -/
def weekend_price : ℝ := 27

/-- The weekend price is 50% more than the weekday price -/
axiom weekend_price_relation : weekend_price = weekday_price * 1.5

theorem mario_haircut_price : weekday_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_mario_haircut_price_l3060_306001


namespace NUMINAMATH_CALUDE_johnny_take_home_pay_is_67_32_l3060_306041

/-- Calculates Johnny's take-home pay after taxes based on his work hours and pay rates. -/
def johnny_take_home_pay (task_a_rate : ℝ) (task_b_rate : ℝ) (total_hours : ℝ) (task_a_hours : ℝ) (tax_rate : ℝ) : ℝ :=
  let task_b_hours := total_hours - task_a_hours
  let total_earnings := task_a_rate * task_a_hours + task_b_rate * task_b_hours
  let tax := tax_rate * total_earnings
  total_earnings - tax

/-- Proves that Johnny's take-home pay after taxes is $67.32 given the specified conditions. -/
theorem johnny_take_home_pay_is_67_32 :
  johnny_take_home_pay 6.75 8.25 10 4 0.12 = 67.32 := by
  sorry

end NUMINAMATH_CALUDE_johnny_take_home_pay_is_67_32_l3060_306041


namespace NUMINAMATH_CALUDE_common_difference_is_negative_three_l3060_306038

def arithmetic_sequence (n : ℕ) : ℤ := 2 - 3 * n

theorem common_difference_is_negative_three :
  ∃ d : ℤ, ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = d ∧ d = -3 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_three_l3060_306038


namespace NUMINAMATH_CALUDE_four_people_three_rooms_l3060_306037

/-- The number of ways to distribute n people into k non-empty rooms -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem four_people_three_rooms :
  distribute 4 3 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_four_people_three_rooms_l3060_306037


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_progressions_l3060_306024

/-- The sum of the first 40 terms of an arithmetic progression -/
def S (p : ℕ) : ℕ :=
  let a := p  -- first term
  let d := 2 * p + 2  -- common difference
  let n := 40  -- number of terms
  n * (2 * a + (n - 1) * d) / 2

/-- The sum of S_p for p from 1 to 10 -/
def total_sum : ℕ :=
  (Finset.range 10).sum (fun i => S (i + 1))

theorem sum_of_arithmetic_progressions : total_sum = 103600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_progressions_l3060_306024


namespace NUMINAMATH_CALUDE_pythagorean_numbers_l3060_306082

theorem pythagorean_numbers (m : ℕ) (a b c : ℝ) : 
  m % 2 = 1 → 
  m > 1 → 
  a = (1/2 : ℝ) * m^2 - (1/2 : ℝ) → 
  c = (1/2 : ℝ) * m^2 + (1/2 : ℝ) → 
  a < c → 
  b < c → 
  a^2 + b^2 = c^2 → 
  b = m := by sorry

end NUMINAMATH_CALUDE_pythagorean_numbers_l3060_306082


namespace NUMINAMATH_CALUDE_base_76_congruence_l3060_306058

theorem base_76_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 18) 
  (h3 : (276935824 : ℤ) ≡ b [ZMOD 17]) : b = 0 ∨ b = 17 := by
  sorry

#check base_76_congruence

end NUMINAMATH_CALUDE_base_76_congruence_l3060_306058


namespace NUMINAMATH_CALUDE_bus_full_after_twelve_stops_l3060_306063

/-- The number of seats in the bus -/
def bus_seats : ℕ := 78

/-- The function representing the total number of passengers after n stops -/
def total_passengers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that 12 is the smallest positive integer that fills the bus -/
theorem bus_full_after_twelve_stops :
  (∀ k : ℕ, k > 0 → k < 12 → total_passengers k < bus_seats) ∧
  total_passengers 12 = bus_seats :=
sorry

end NUMINAMATH_CALUDE_bus_full_after_twelve_stops_l3060_306063


namespace NUMINAMATH_CALUDE_total_rainfall_is_25_l3060_306027

def sunday_rainfall : ℝ := 4

def monday_rainfall : ℝ := sunday_rainfall + 3

def tuesday_rainfall : ℝ := 2 * monday_rainfall

def total_rainfall : ℝ := sunday_rainfall + monday_rainfall + tuesday_rainfall

theorem total_rainfall_is_25 : total_rainfall = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_is_25_l3060_306027


namespace NUMINAMATH_CALUDE_three_same_one_different_probability_l3060_306051

def number_of_dice : ℕ := 4
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ number_of_dice

def successful_outcomes : ℕ := faces_per_die * (number_of_dice.choose 3) * (faces_per_die - 1)

def probability_three_same_one_different : ℚ := successful_outcomes / total_outcomes

theorem three_same_one_different_probability :
  probability_three_same_one_different = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_three_same_one_different_probability_l3060_306051


namespace NUMINAMATH_CALUDE_min_value_expression_l3060_306070

theorem min_value_expression (a b m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) 
  (h5 : a + b = 1) (h6 : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3060_306070


namespace NUMINAMATH_CALUDE_find_y_l3060_306064

theorem find_y (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3060_306064


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3060_306011

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3060_306011


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l3060_306004

/-- Represents a parabola in the form y^2 = 4px --/
structure Parabola where
  p : ℝ

/-- The directrix of a parabola --/
def directrix (para : Parabola) : ℝ := -para.p

theorem parabola_directrix_equation :
  let para : Parabola := ⟨1⟩
  directrix para = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l3060_306004


namespace NUMINAMATH_CALUDE_calculation_proof_l3060_306074

theorem calculation_proof : 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + 0.40 + (0.5 : ℝ)^2 = 0.9666875 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3060_306074


namespace NUMINAMATH_CALUDE_sarahs_pastry_flour_l3060_306006

def rye_flour : ℕ := 5
def wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def total_flour : ℕ := 20

def pastry_flour : ℕ := total_flour - (rye_flour + wheat_bread_flour + chickpea_flour)

theorem sarahs_pastry_flour : pastry_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_pastry_flour_l3060_306006


namespace NUMINAMATH_CALUDE_product_of_sums_l3060_306077

theorem product_of_sums (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b + a + b = 35) (hbc : b * c + b + c = 35) (hca : c * a + c + a = 35) :
  (a + 1) * (b + 1) * (c + 1) = 216 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l3060_306077


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3060_306055

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of the specific geometric sequence -/
theorem tenth_term_of_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3060_306055


namespace NUMINAMATH_CALUDE_xyz_value_l3060_306076

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 18)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3060_306076


namespace NUMINAMATH_CALUDE_sum_of_f_symmetric_points_sum_of_roots_l3060_306054

-- Define the cubic function f
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

-- Theorem 1
theorem sum_of_f_symmetric_points (x₁ x₂ : ℝ) (h : x₁ + x₂ = 0) : 
  f x₁ + f x₂ = -2 := by sorry

-- Theorem 2
theorem sum_of_roots (m n : ℝ) 
  (hm : m^3 - 3*m^2 + 5*m - 4 = 0) 
  (hn : n^3 - 3*n^2 + 5*n - 2 = 0) : 
  m + n = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_f_symmetric_points_sum_of_roots_l3060_306054


namespace NUMINAMATH_CALUDE_triangle_inequality_l3060_306060

theorem triangle_inequality (a b c m : ℝ) (γ : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < m) (h5 : 0 < γ) (h6 : γ < π) : 
  a + b + m ≤ ((2 + Real.cos (γ / 2)) / (2 * Real.sin (γ / 2))) * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3060_306060


namespace NUMINAMATH_CALUDE_first_day_revenue_l3060_306012

/-- Calculates the revenue from ticket sales given the number of senior and student tickets sold and their prices -/
def revenue (senior_count : ℕ) (student_count : ℕ) (senior_price : ℚ) (student_price : ℚ) : ℚ :=
  senior_count * senior_price + student_count * student_price

/-- Represents the ticket sales scenario -/
structure TicketSales where
  day1_senior : ℕ
  day1_student : ℕ
  day2_senior : ℕ
  day2_student : ℕ
  day2_revenue : ℚ
  student_price : ℚ

/-- Theorem stating that the first day's revenue is $79 -/
theorem first_day_revenue (ts : TicketSales) 
  (h1 : ts.day1_senior = 4)
  (h2 : ts.day1_student = 3)
  (h3 : ts.day2_senior = 12)
  (h4 : ts.day2_student = 10)
  (h5 : ts.day2_revenue = 246)
  (h6 : ts.student_price = 9)
  : ∃ (senior_price : ℚ), revenue ts.day1_senior ts.day1_student senior_price ts.student_price = 79 := by
  sorry

end NUMINAMATH_CALUDE_first_day_revenue_l3060_306012


namespace NUMINAMATH_CALUDE_sqrt_4_4_times_9_2_l3060_306028

theorem sqrt_4_4_times_9_2 : Real.sqrt (4^4 * 9^2) = 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_4_times_9_2_l3060_306028


namespace NUMINAMATH_CALUDE_dress_price_problem_l3060_306022

/-- 
Given a dress with an original price x, if Barb buys it for (x/2 - 10) dollars 
and saves 80 dollars, then x = 140.
-/
theorem dress_price_problem (x : ℝ) 
  (h1 : x - (x / 2 - 10) = 80) : x = 140 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_problem_l3060_306022


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3060_306080

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (d h w : ℝ) -- d: diameter, h: height, w: stripe width
  (n : ℕ) -- number of revolutions
  (h_d : d = 40)
  (h_h : h = 120)
  (h_w : w = 4)
  (h_n : n = 3) :
  w * n * (h / n) * Real.sqrt ((π * d)^2 + (h / n)^2) = 480 * Real.sqrt (π^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l3060_306080


namespace NUMINAMATH_CALUDE_number_problem_l3060_306083

theorem number_problem : 
  ∃ x : ℚ, (30 / 100 : ℚ) * x = (25 / 100 : ℚ) * 40 ∧ x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3060_306083


namespace NUMINAMATH_CALUDE_four_people_seven_chairs_two_occupied_l3060_306002

/-- The number of ways to arrange n distinct objects in r positions --/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways four people can sit in a row of seven chairs
    where two specific chairs are always occupied --/
theorem four_people_seven_chairs_two_occupied : 
  permutation 5 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_four_people_seven_chairs_two_occupied_l3060_306002


namespace NUMINAMATH_CALUDE_sqrt_equation_root_l3060_306059

theorem sqrt_equation_root :
  ∃ x : ℝ, (Real.sqrt x + Real.sqrt (x + 2) = 12) ∧ (x = 5041 / 144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_root_l3060_306059


namespace NUMINAMATH_CALUDE_orthic_triangle_similarity_l3060_306000

/-- A triangle with angles A, B, and C in degrees -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle where
  A := 180 - 2 * t.A
  B := 180 - 2 * t.B
  C := 180 - 2 * t.C
  sum_180 := sorry
  positive := sorry

/-- Two triangles are similar if their corresponding angles are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

theorem orthic_triangle_similarity (t : Triangle) 
  (h_not_right : t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90) :
  similar t (orthicTriangle t) ↔ t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_similarity_l3060_306000


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l3060_306099

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 1

-- Define the perpendicular line (our answer)
def perp_line (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- State the theorem
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    curve x₀ y₀ ∧
    -- The point (x₀, y₀) is on the perpendicular line
    perp_line x₀ y₀ ∧
    -- The perpendicular line is indeed perpendicular to the given line
    (3 : ℝ) * (1 / 3 : ℝ) = -1 ∧
    -- The slope of the curve at (x₀, y₀) equals the slope of the perpendicular line
    (3 * x₀^2 + 6 * x₀) = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_and_tangent_l3060_306099


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3060_306035

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3060_306035


namespace NUMINAMATH_CALUDE_sequence_sum_equals_5923_l3060_306042

def arithmetic_sum (a1 l1 d : ℤ) : ℤ :=
  let n := (l1 - a1) / d + 1
  n * (a1 + l1) / 2

def sequence_sum : ℤ :=
  3 * (arithmetic_sum 45 93 2) + 2 * (arithmetic_sum (-4) 38 2)

theorem sequence_sum_equals_5923 : sequence_sum = 5923 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_5923_l3060_306042


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3060_306057

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a * (a - 1) + a * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3060_306057


namespace NUMINAMATH_CALUDE_factorization_equality_l3060_306033

theorem factorization_equality (a x y : ℝ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3060_306033


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3060_306091

/-- The coordinates of a point with respect to the origin are the same as its Cartesian coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let p : ℝ × ℝ := (x, y)
  p = p := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3060_306091


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35__1015_divisible_by_35__1015_is_four_digit_smallest_four_digit_divisible_by_35_is_1015_l3060_306061

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 1015 → ¬(35 ∣ n) :=
by
  sorry

theorem _1015_divisible_by_35 : 35 ∣ 1015 :=
by
  sorry

theorem _1015_is_four_digit : 1000 ≤ 1015 ∧ 1015 ≤ 9999 :=
by
  sorry

theorem smallest_four_digit_divisible_by_35_is_1015 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (35 ∣ n) → 1015 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35__1015_divisible_by_35__1015_is_four_digit_smallest_four_digit_divisible_by_35_is_1015_l3060_306061


namespace NUMINAMATH_CALUDE_sphere_in_cube_l3060_306021

theorem sphere_in_cube (edge : ℝ) (radius : ℝ) : 
  edge = 8 →
  (4 / 3) * Real.pi * radius^3 = (1 / 2) * edge^3 →
  radius = (192 / Real.pi)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cube_l3060_306021


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3060_306008

theorem cubic_equation_roots (m : ℝ) : 
  (m = 3 ∨ m = -2) → 
  ∃ (z₁ z₂ z₃ : ℝ), 
    z₁ = -1 ∧ z₂ = -3 ∧ z₃ = 4 ∧
    ∀ (z : ℝ), z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6) = 0 ↔ 
      (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3060_306008


namespace NUMINAMATH_CALUDE_compute_expression_l3060_306078

theorem compute_expression : 20 * (180 / 3 + 40 / 5 + 16 / 32 + 2) = 1410 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3060_306078


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3060_306050

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The focal length of a hyperbola -/
def focal_length (h : Hyperbola a b) : ℝ := sorry

/-- A point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The distance between two points -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- The origin (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- Left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (P : PointOnHyperbola h) :
  distance P.x P.y (origin.1) (origin.2) = focal_length h / 2 →
  distance P.x P.y (left_focus h).1 (left_focus h).2 + 
    distance P.x P.y (right_focus h).1 (right_focus h).2 = 4 * a →
  eccentricity h = Real.sqrt 10 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3060_306050


namespace NUMINAMATH_CALUDE_value_of_expression_l3060_306045

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) :
  4 * a + 8 * b - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3060_306045


namespace NUMINAMATH_CALUDE_mat_coverage_fraction_l3060_306065

/-- The fraction of a square tabletop covered by a circular mat -/
theorem mat_coverage_fraction (r : ℝ) (s : ℝ) (hr : r = 10) (hs : s = 24) :
  (π * r^2) / (s^2) = 100 * π / 576 := by
  sorry

end NUMINAMATH_CALUDE_mat_coverage_fraction_l3060_306065


namespace NUMINAMATH_CALUDE_shortest_line_on_square_pyramid_l3060_306089

/-- The shortest line on the lateral faces of a square pyramid -/
theorem shortest_line_on_square_pyramid (a m : ℝ) (ha : a > 0) (hm : m > 0) (h_eq : a = m) :
  let x := Real.sqrt (2 * a^2)
  let m₁ := Real.sqrt (x^2 - (a/2)^2)
  2 * a * m₁ / x = 80 * Real.sqrt (5/6) :=
by sorry

end NUMINAMATH_CALUDE_shortest_line_on_square_pyramid_l3060_306089


namespace NUMINAMATH_CALUDE_total_balls_is_six_l3060_306025

/-- The number of balls in each box -/
def balls_per_box : ℕ := 3

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The total number of balls -/
def total_balls : ℕ := balls_per_box * num_boxes

theorem total_balls_is_six : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_is_six_l3060_306025


namespace NUMINAMATH_CALUDE_horse_grain_consumption_l3060_306052

/-- Calculates the amount of grain each horse eats per day -/
theorem horse_grain_consumption
  (num_horses : ℕ)
  (oats_per_meal : ℕ)
  (oats_meals_per_day : ℕ)
  (total_days : ℕ)
  (total_food : ℕ)
  (h1 : num_horses = 4)
  (h2 : oats_per_meal = 4)
  (h3 : oats_meals_per_day = 2)
  (h4 : total_days = 3)
  (h5 : total_food = 132) :
  (total_food - num_horses * oats_per_meal * oats_meals_per_day * total_days) / (num_horses * total_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_horse_grain_consumption_l3060_306052


namespace NUMINAMATH_CALUDE_eccentricity_is_sqrt_three_l3060_306029

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A circle centered on a hyperbola and tangent to the x-axis at a focus -/
structure TangentCircle (h : Hyperbola) where
  center : ℝ × ℝ
  h_on_hyperbola : center.1^2 / h.a^2 - center.2^2 / h.b^2 = 1
  h_tangent_at_focus : center.1 = h.a * (h.a^2 + h.b^2).sqrt / h.a

/-- The property that the circle intersects the y-axis forming an equilateral triangle -/
def forms_equilateral_triangle (h : Hyperbola) (c : TangentCircle h) : Prop :=
  ∃ (y₁ y₂ : ℝ), 
    y₁ < c.center.2 ∧ c.center.2 < y₂ ∧
    (c.center.1^2 + (y₁ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (c.center.1^2 + (y₂ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (y₂ - y₁ = h.b^2 / h.a * Real.sqrt 3)

/-- The theorem stating that the eccentricity of the hyperbola is √3 -/
theorem eccentricity_is_sqrt_three (h : Hyperbola) (c : TangentCircle h)
  (h_equilateral : forms_equilateral_triangle h c) :
  (h.a^2 + h.b^2).sqrt / h.a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_is_sqrt_three_l3060_306029


namespace NUMINAMATH_CALUDE_cosine_equality_l3060_306086

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 92 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l3060_306086


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3060_306079

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perp_line_plane m α → 
  perp_line_plane n β → 
  perp_plane α β → 
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3060_306079


namespace NUMINAMATH_CALUDE_constant_pace_running_time_l3060_306032

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunningPace where
  distance : ℝ
  time : ℝ

/-- Theorem: If it takes 24 minutes to run 3 miles at a constant pace, 
    then it will take 16 minutes to run 2 miles at the same pace -/
theorem constant_pace_running_time 
  (park : RunningPace) 
  (library : RunningPace) 
  (h1 : park.distance = 3) 
  (h2 : park.time = 24) 
  (h3 : library.distance = 2) 
  (h4 : park.time / park.distance = library.time / library.distance) : 
  library.time = 16 := by
sorry

end NUMINAMATH_CALUDE_constant_pace_running_time_l3060_306032


namespace NUMINAMATH_CALUDE_soccer_ball_contribution_l3060_306023

theorem soccer_ball_contribution (k l m : ℝ) : 
  k ≥ 0 → l ≥ 0 → m ≥ 0 →
  k + l + m = 6 →
  2 * k ≤ l + m →
  2 * l ≤ k + m →
  2 * m ≤ k + l →
  k = 2 ∧ l = 2 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_contribution_l3060_306023


namespace NUMINAMATH_CALUDE_pen_problem_l3060_306071

theorem pen_problem (marked_price : ℝ) (num_pens : ℕ) : 
  marked_price > 0 →
  num_pens * marked_price = 46 * marked_price →
  (num_pens * marked_price * 0.99 - 46 * marked_price) / (46 * marked_price) * 100 = 29.130434782608695 →
  num_pens = 60 := by
sorry

end NUMINAMATH_CALUDE_pen_problem_l3060_306071


namespace NUMINAMATH_CALUDE_fibonacci_special_sequence_l3060_306044

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_special_sequence (a b c : ℕ) :
  (fib c = 2 * fib b - fib a) →
  (fib c - fib a = fib a) →
  (a + c = 1700) →
  a = 849 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_special_sequence_l3060_306044


namespace NUMINAMATH_CALUDE_function_zero_range_l3060_306066

open Real

theorem function_zero_range (f : ℝ → ℝ) (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 π, f x = 0) →
  (∀ x, f x = 2 * sin (x + π / 4) + m) →
  m ∈ Set.Icc (-2) (Real.sqrt 2) ∧ m ≠ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_function_zero_range_l3060_306066


namespace NUMINAMATH_CALUDE_xiaoming_pe_grade_l3060_306014

/-- Calculates the semester physical education grade based on given scores and weights -/
def calculate_semester_grade (extracurricular_score midterm_score final_score : ℚ) 
  (extracurricular_weight midterm_weight final_weight : ℕ) : ℚ :=
  (extracurricular_score * extracurricular_weight + 
   midterm_score * midterm_weight + 
   final_score * final_weight) / 
  (extracurricular_weight + midterm_weight + final_weight)

/-- Xiaoming's physical education grade theorem -/
theorem xiaoming_pe_grade :
  let max_score : ℚ := 100
  let extracurricular_score : ℚ := 95
  let midterm_score : ℚ := 90
  let final_score : ℚ := 85
  let extracurricular_weight : ℕ := 2
  let midterm_weight : ℕ := 4
  let final_weight : ℕ := 4
  calculate_semester_grade extracurricular_score midterm_score final_score
    extracurricular_weight midterm_weight final_weight = 89 := by
  sorry


end NUMINAMATH_CALUDE_xiaoming_pe_grade_l3060_306014


namespace NUMINAMATH_CALUDE_inequality_proof_l3060_306019

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3060_306019


namespace NUMINAMATH_CALUDE_sum_greater_than_double_smaller_l3060_306013

theorem sum_greater_than_double_smaller (a b c : ℝ) 
  (h1 : a > c) (h2 : b > c) : a + b > 2 * c := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_double_smaller_l3060_306013


namespace NUMINAMATH_CALUDE_distance_blown_westward_is_200km_l3060_306005

/-- Represents the journey of a ship -/
structure ShipJourney where
  speed : ℝ
  travelTime : ℝ
  totalDistance : ℝ
  finalPosition : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  journey.speed * journey.travelTime - journey.finalPosition

/-- Theorem stating the distance blown westward is 200 km -/
theorem distance_blown_westward_is_200km (journey : ShipJourney) 
  (h1 : journey.speed = 30)
  (h2 : journey.travelTime = 20)
  (h3 : journey.speed * journey.travelTime = journey.totalDistance / 2)
  (h4 : journey.finalPosition = journey.totalDistance / 3) :
  distanceBlownWestward journey = 200 := by
  sorry

#check distance_blown_westward_is_200km

end NUMINAMATH_CALUDE_distance_blown_westward_is_200km_l3060_306005


namespace NUMINAMATH_CALUDE_games_left_to_play_l3060_306043

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  total_points : Nat
  lowest_score : Nat
  top_two_equal : Bool

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total points that will be distributed in the tournament --/
def total_tournament_points (n : Nat) : Nat :=
  2 * total_matches n

/-- Theorem: In a round-robin tournament with 9 teams, where the total points
    scored is 44, the lowest-scoring team has 1 point, and the top two teams
    have equal points, there are 14 games left to play. --/
theorem games_left_to_play (t : Tournament)
  (h1 : t.num_teams = 9)
  (h2 : t.total_points = 44)
  (h3 : t.lowest_score = 1)
  (h4 : t.top_two_equal = true) :
  total_matches t.num_teams - t.total_points / 2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_games_left_to_play_l3060_306043


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3060_306040

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3060_306040


namespace NUMINAMATH_CALUDE_farm_animals_l3060_306046

theorem farm_animals (horses cows : ℕ) : 
  horses = 6 * cows →  -- Initial ratio of horses to cows is 6:1
  (horses - 15) = 3 * (cows + 15) →  -- New ratio after transaction is 3:1
  (horses - 15) - (cows + 15) = 70 := by  -- Difference after transaction is 70
sorry

end NUMINAMATH_CALUDE_farm_animals_l3060_306046


namespace NUMINAMATH_CALUDE_banana_pear_difference_l3060_306049

/-- Represents a bowl of fruit with apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Properties of the fruit bowl -/
def is_valid_fruit_bowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas > bowl.pears ∧
  bowl.apples + bowl.pears + bowl.bananas = 19 ∧
  bowl.bananas = 9

theorem banana_pear_difference (bowl : FruitBowl) 
  (h : is_valid_fruit_bowl bowl) : 
  bowl.bananas - bowl.pears = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_difference_l3060_306049


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l3060_306097

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180 degrees clockwise around the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating that rotating the given points 180 degrees clockwise
    results in the expected transformed points -/
theorem rotation_180_maps_points :
  let C : Point := { x := 3, y := -2 }
  let D : Point := { x := 2, y := -5 }
  let C' : Point := { x := -3, y := 2 }
  let D' : Point := { x := -2, y := 5 }
  rotate180 C = C' ∧ rotate180 D = D' := by
  sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l3060_306097


namespace NUMINAMATH_CALUDE_pages_per_day_l3060_306062

/-- Given a book with 612 pages read over 6 days, prove that the number of pages read per day is 102. -/
theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 612) (h2 : days = 6) :
  total_pages / days = 102 := by
  sorry

#check pages_per_day

end NUMINAMATH_CALUDE_pages_per_day_l3060_306062


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3060_306081

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2 ∧ x = 2) →
  (∃ y : ℝ, 3 * y^2 + m * y = -2 ∧ y = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3060_306081


namespace NUMINAMATH_CALUDE_grasshopper_distance_l3060_306018

def grasshopper_jumps : List ℝ := [2, -3, 8, -1]

theorem grasshopper_distance : 
  let distances := List.zipWith (fun x y => |x - y|) grasshopper_jumps grasshopper_jumps.tail
  List.sum distances = 25 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_distance_l3060_306018


namespace NUMINAMATH_CALUDE_triangle_angle_120_degrees_l3060_306073

theorem triangle_angle_120_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 - b^2 = 3*b*c)
  (h3 : Real.sin C = 2 * Real.sin B)
  (h4 : A + B + C = π)
  (h5 : a / Real.sin A = b / Real.sin B)
  (h6 : b / Real.sin B = c / Real.sin C)
  : A = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_120_degrees_l3060_306073


namespace NUMINAMATH_CALUDE_min_sum_positive_integers_l3060_306092

theorem min_sum_positive_integers (x y z w : ℕ+) 
  (h : (2 : ℕ) * x ^ 2 = (5 : ℕ) * y ^ 3 ∧ 
       (5 : ℕ) * y ^ 3 = (8 : ℕ) * z ^ 4 ∧ 
       (8 : ℕ) * z ^ 4 = (3 : ℕ) * w) : 
  x + y + z + w ≥ 54 := by
sorry

end NUMINAMATH_CALUDE_min_sum_positive_integers_l3060_306092


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3060_306010

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3060_306010


namespace NUMINAMATH_CALUDE_vans_needed_for_field_trip_l3060_306056

theorem vans_needed_for_field_trip (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) :
  van_capacity = 5 → num_students = 25 → num_adults = 5 →
  (num_students + num_adults) / van_capacity = 6 :=
by sorry

end NUMINAMATH_CALUDE_vans_needed_for_field_trip_l3060_306056


namespace NUMINAMATH_CALUDE_trajectory_of_point_l3060_306053

/-- The trajectory of a point M(x, y) satisfying the given distance condition -/
theorem trajectory_of_point (x y : ℝ) :
  (((x - 2)^2 + y^2).sqrt = |x + 3| - 1) →
  (y^2 = 8 * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_l3060_306053


namespace NUMINAMATH_CALUDE_max_k_value_l3060_306096

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (Real.sqrt 7 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l3060_306096


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l3060_306048

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 20 → lost_shoes = 9 → max_pairs = 11 →
  max_pairs = initial_pairs - lost_shoes ∧ 
  max_pairs * 2 + lost_shoes ≤ initial_pairs * 2 :=
by sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l3060_306048


namespace NUMINAMATH_CALUDE_first_digit_base_nine_is_three_l3060_306039

def base_three_representation : List Nat := [2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1]

def y : Nat := base_three_representation.enum.foldl (fun acc (i, digit) => acc + digit * (3 ^ (base_three_representation.length - 1 - i))) 0

theorem first_digit_base_nine_is_three :
  ∃ (rest : Nat), y = 3 * (9 ^ (Nat.log 9 y)) + rest ∧ rest < 9 ^ (Nat.log 9 y) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_base_nine_is_three_l3060_306039


namespace NUMINAMATH_CALUDE_remy_water_usage_l3060_306095

/-- Proves that Remy used 25 gallons of water given the conditions of the problem. -/
theorem remy_water_usage (roman : ℕ) (remy : ℕ) : 
  remy = 3 * roman + 1 →  -- Condition 1
  roman + remy = 33 →     -- Condition 2
  remy = 25 := by
sorry

end NUMINAMATH_CALUDE_remy_water_usage_l3060_306095


namespace NUMINAMATH_CALUDE_min_value_fraction_l3060_306015

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 2) :
  (2 * x + y) / (x * y) ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3060_306015


namespace NUMINAMATH_CALUDE_range_of_a_l3060_306069

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -2 * a * x - 8

-- Define proposition q
def q (a : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), 
  x^2 + y^2 - 4*x + a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

-- Main theorem
theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  {a : ℝ | a < 0 ∨ (a ≥ 4 ∧ a < 8)} = {a : ℝ | p a ∨ q a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3060_306069


namespace NUMINAMATH_CALUDE_average_height_is_64_inches_l3060_306030

/-- Given the heights of Parker, Daisy, and Reese, prove their average height is 64 inches. -/
theorem average_height_is_64_inches 
  (reese_height : ℕ)
  (daisy_height : ℕ)
  (parker_height : ℕ)
  (h1 : reese_height = 60)
  (h2 : daisy_height = reese_height + 8)
  (h3 : parker_height = daisy_height - 4) :
  (reese_height + daisy_height + parker_height) / 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_average_height_is_64_inches_l3060_306030


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3060_306072

theorem algebraic_expression_value :
  let a : ℝ := 1 + Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  a^2 + b^2 - 2*a + 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3060_306072


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_values_l3060_306031

theorem arithmetic_geometric_sequence_values :
  ∀ (a b c : ℝ),
  (∃ (d : ℝ), b = (a + c) / 2 ∧ c - b = b - a) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (∃ (r : ℝ), (b + 2) ^ 2 = (a + 2) * (c + 5)) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_values_l3060_306031


namespace NUMINAMATH_CALUDE_black_balls_count_l3060_306017

theorem black_balls_count (orange white : ℕ) (prob : ℚ) (black : ℕ) :
  orange = 8 →
  white = 6 →
  prob = 8038095238095238093 / 21100000000000000000 →
  (black : ℚ) / (orange + white + black : ℚ) = prob →
  black = 9 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_count_l3060_306017


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3060_306075

/-- Pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of the pentagon -/
def pentagon_area (p : CornerCutPentagon) : ℕ :=
  1421

/-- Theorem stating that the area of the specified pentagon is 1421 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : 
  pentagon_area p = 1421 := by sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l3060_306075


namespace NUMINAMATH_CALUDE_problem_statement_l3060_306088

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3060_306088


namespace NUMINAMATH_CALUDE_parallel_vectors_l3060_306084

def a : ℝ × ℝ := (1, -1)
def b : ℝ → ℝ × ℝ := λ x => (x, 1)

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, b x = k • a) → x = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3060_306084


namespace NUMINAMATH_CALUDE_share_distribution_l3060_306003

theorem share_distribution (total : ℕ) (a b c : ℚ) 
  (h1 : total = 880)
  (h2 : a + b + c = total)
  (h3 : 4 * a = 5 * b)
  (h4 : 5 * b = 10 * c) :
  c = 160 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l3060_306003


namespace NUMINAMATH_CALUDE_stock_percentage_value_l3060_306020

/-- Calculates the percentage value of a stock given its yield and price. -/
def percentageValue (yield : ℝ) (price : ℝ) : ℝ :=
  yield * 100

theorem stock_percentage_value :
  let yield : ℝ := 0.10
  let price : ℝ := 80
  percentageValue yield price = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_value_l3060_306020


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l3060_306093

theorem set_of_positive_rationals (S : Set ℚ) 
  (closure : ∀ a b : ℚ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S)
  (trichotomy : ∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)) :
  S = {r : ℚ | 0 < r} := by
sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l3060_306093


namespace NUMINAMATH_CALUDE_exponential_decreasing_zero_two_l3060_306007

theorem exponential_decreasing_zero_two (m n : ℝ) : m > n → (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_zero_two_l3060_306007


namespace NUMINAMATH_CALUDE_union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l3060_306009

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m - 1) * (x - m - 7) > 0}

-- Theorem 1
theorem union_A_complement_B_when_m_neg_two :
  A ∪ (Set.univ \ B (-2)) = {x : ℝ | -2 < x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem A_implies_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ↔ m ∈ Set.Iic (-9) ∪ Set.Ici 1 := by sorry

-- Theorem 3
theorem A_not_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∉ B m) ↔ m ∈ Set.Icc (-5) (-3) := by sorry

-- Theorem 4
theorem A_subset_complement_B_iff :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m ∈ Set.Ioo (-5) (-3) := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l3060_306009
