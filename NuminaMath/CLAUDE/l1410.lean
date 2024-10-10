import Mathlib

namespace geometric_sequence_sum_l1410_141043

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 :=
by sorry

end geometric_sequence_sum_l1410_141043


namespace geometric_sequence_a7_l1410_141025

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 11 = 27 → a 7 = 9 := by
  sorry

end geometric_sequence_a7_l1410_141025


namespace m_range_theorem_l1410_141075

def f (x : ℝ) : ℝ := x^2 - 2*x

def g (m : ℝ) (x : ℝ) : ℝ := m*x + 2

theorem m_range_theorem (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g m x₁ = f x₀) →
  m ∈ Set.Icc (-1 : ℝ) (1/2) :=
by sorry

end m_range_theorem_l1410_141075


namespace seventh_root_unity_product_l1410_141051

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 7 := by
  sorry

end seventh_root_unity_product_l1410_141051


namespace f_properties_l1410_141034

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f satisfies the given functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x + f 1

-- f is monotonically increasing on [0, 1]
def monotone_increasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- The graph of f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (1 - x)

-- f is periodic
def periodic (f : ℝ → ℝ) : Prop := ∃ p > 0, ∀ x, f (x + p) = f x

-- f has local minima at even x-coordinates
def local_minima_at_even (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ ε > 0, ∀ y, |y - x| < ε → f x ≤ f y

theorem f_properties (heven : even_function f)
                     (heq : satisfies_equation f)
                     (hmon : monotone_increasing_on_unit_interval f) :
  symmetric_about_one f ∧ periodic f ∧ local_minima_at_even f := by sorry

end

end f_properties_l1410_141034


namespace antihomologous_properties_l1410_141000

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Homothety center -/
def S : Point := sorry

/-- Given two circles satisfying the problem conditions -/
def circle1 : Circle := sorry
def circle2 : Circle := sorry

/-- Antihomologous points -/
def isAntihomologous (p q : Point) : Prop := sorry

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- A circle is tangent to another circle -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Main theorem -/
theorem antihomologous_properties 
  (h1 : circle1.radius > circle2.radius)
  (h2 : isTangent circle1 circle2 ∨ 
        (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 
        > (circle1.radius + circle2.radius)^2) :
  (∀ (c : Circle) (p1 p2 p3 p4 : Point),
    isAntihomologous p1 p2 →
    onCircle p1 c ∧ onCircle p2 c →
    onCircle p3 circle1 ∧ onCircle p4 circle2 ∧ onCircle p3 c ∧ onCircle p4 c →
    isAntihomologous p3 p4) ∧
  (∀ (c : Circle),
    isTangent c circle1 ∧ isTangent c circle2 →
    ∃ (p1 p2 : Point),
      onCircle p1 circle1 ∧ onCircle p2 circle2 ∧
      onCircle p1 c ∧ onCircle p2 c ∧
      isAntihomologous p1 p2) :=
by sorry

end antihomologous_properties_l1410_141000


namespace remainder_3211_103_l1410_141096

theorem remainder_3211_103 : 3211 % 103 = 18 := by
  sorry

end remainder_3211_103_l1410_141096


namespace shortest_distance_between_inscribed_circles_shortest_distance_proof_l1410_141071

/-- The shortest distance between two circles inscribed in two of nine identical squares 
    (each with side length 1) that form a larger square -/
theorem shortest_distance_between_inscribed_circles : ℝ :=
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let num_small_squares : ℕ := 9
  let circle_radius : ℝ := small_square_side / 2
  2 * Real.sqrt 2 - 1

/-- Proof of the shortest distance between the inscribed circles -/
theorem shortest_distance_proof :
  shortest_distance_between_inscribed_circles = 2 * Real.sqrt 2 - 1 := by
  sorry

end shortest_distance_between_inscribed_circles_shortest_distance_proof_l1410_141071


namespace factor_3x_squared_minus_75_l1410_141077

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_3x_squared_minus_75_l1410_141077


namespace first_operation_result_l1410_141056

theorem first_operation_result (x : ℝ) : (x - 24) / 10 = 3 → (x - 5) / 7 = 7 := by
  sorry

end first_operation_result_l1410_141056


namespace min_sum_squares_l1410_141048

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 7}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 2 :=
sorry

end min_sum_squares_l1410_141048


namespace prob_two_heads_and_three_l1410_141054

-- Define the probability of getting heads on a fair coin
def prob_heads : ℚ := 1/2

-- Define the probability of rolling a 3 on a fair six-sided die
def prob_three : ℚ := 1/6

-- State the theorem
theorem prob_two_heads_and_three (h1 : prob_heads = 1/2) (h2 : prob_three = 1/6) : 
  prob_heads * prob_heads * prob_three = 1/24 := by
  sorry


end prob_two_heads_and_three_l1410_141054


namespace cosine_sum_simplification_l1410_141065

theorem cosine_sum_simplification :
  Real.cos ((2 * Real.pi) / 17) + Real.cos ((6 * Real.pi) / 17) + Real.cos ((8 * Real.pi) / 17) = (Real.sqrt 13 - 1) / 4 :=
by sorry

end cosine_sum_simplification_l1410_141065


namespace sector_central_angle_l1410_141010

/-- Given a sector with radius 10 cm and perimeter 45 cm, its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (α : ℝ) : 
  r = 10 → p = 45 → α = (p - 2 * r) / r → α = 2.5 := by
  sorry

end sector_central_angle_l1410_141010


namespace max_volume_right_prism_l1410_141009

/-- Given a right prism with rectangular base (sides a and b) and height h, 
    where the sum of areas of two lateral faces and one base is 40,
    the maximum volume of the prism is 80√30/9 -/
theorem max_volume_right_prism (a b h : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : h > 0)
  (h₄ : a * h + b * h + a * b = 40) : 
  a * b * h ≤ 80 * Real.sqrt 30 / 9 := by
  sorry

#check max_volume_right_prism

end max_volume_right_prism_l1410_141009


namespace regular_tetrahedron_has_four_faces_l1410_141068

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: A regular tetrahedron has 4 faces -/
theorem regular_tetrahedron_has_four_faces (t : RegularTetrahedron) : num_faces t = 4 := by
  sorry

end regular_tetrahedron_has_four_faces_l1410_141068


namespace arithmetic_sign_change_geometric_sign_alternation_l1410_141078

-- Define an arithmetic progression
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define a geometric progression
def geometric_progression (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem for arithmetic progression sign change
theorem arithmetic_sign_change (a₁ : ℝ) (d : ℝ) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → (arithmetic_progression a₁ d n > 0) ∧
                     ∀ m : ℕ, m > k → (arithmetic_progression a₁ d m < 0) :=
sorry

-- Theorem for geometric progression sign alternation
theorem geometric_sign_alternation (a₁ : ℝ) (r : ℝ) (h : r < 0) :
  ∀ n : ℕ, (geometric_progression a₁ r (2*n) > 0) ∧ 
           (geometric_progression a₁ r (2*n + 1) < 0) :=
sorry

end arithmetic_sign_change_geometric_sign_alternation_l1410_141078


namespace franks_reading_rate_l1410_141003

/-- Represents a book with its properties --/
structure Book where
  pages : ℕ
  chapters : ℕ
  days_to_read : ℕ

/-- Calculates the number of chapters read per day --/
def chapters_per_day (b : Book) : ℚ :=
  (b.chapters : ℚ) / (b.days_to_read : ℚ)

/-- Theorem stating the number of chapters read per day for Frank's book --/
theorem franks_reading_rate (b : Book) 
    (h1 : b.pages = 193)
    (h2 : b.chapters = 15)
    (h3 : b.days_to_read = 660) :
    chapters_per_day b = 15 / 660 := by
  sorry

end franks_reading_rate_l1410_141003


namespace unique_solution_for_quadratic_difference_l1410_141062

theorem unique_solution_for_quadratic_difference (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ x : ℝ, (x + m)^2 - (x + n)^2 = (m - n)^2 → x = a * m + b * n ∧ a = 0 ∧ b ≠ 0 :=
by sorry

end unique_solution_for_quadratic_difference_l1410_141062


namespace trapezoid_bases_l1410_141055

/-- Given a trapezoid with midline 6 and difference between bases 4, prove the bases are 4 and 8 -/
theorem trapezoid_bases (a b : ℝ) : 
  (a + b) / 2 = 6 → -- midline is 6
  a - b = 4 →       -- difference between bases is 4
  (a = 8 ∧ b = 4) := by
sorry

end trapezoid_bases_l1410_141055


namespace smallest_cube_sum_solution_l1410_141036

/-- The smallest positive integer solution for the equation u³ + v³ + w³ = x³ -/
theorem smallest_cube_sum_solution :
  let P : ℕ → ℕ → ℕ → ℕ → Prop :=
    fun u v w x => u^3 + v^3 + w^3 = x^3 ∧ 
                   u < v ∧ v < w ∧ w < x ∧
                   v = u + 1 ∧ w = v + 1 ∧ x = w + 1
  ∀ u v w x, P u v w x → x ≥ 6 :=
by sorry

end smallest_cube_sum_solution_l1410_141036


namespace smallest_b_value_l1410_141023

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^4 + b.val^4) / (a.val + b.val)) (a.val * b.val) = 16) :
  b.val ≥ 4 ∧ ∃ (a₀ b₀ : ℕ+), b₀.val = 4 ∧ a₀.val - b₀.val = 8 ∧ 
    Nat.gcd ((a₀.val^4 + b₀.val^4) / (a₀.val + b₀.val)) (a₀.val * b₀.val) = 16 :=
by sorry

end smallest_b_value_l1410_141023


namespace equal_numbers_product_l1410_141045

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end equal_numbers_product_l1410_141045


namespace lcm_18_30_l1410_141080

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l1410_141080


namespace cleaning_times_l1410_141063

/-- Proves the cleaning times for Bob and Carol given Alice's cleaning time -/
theorem cleaning_times (alice_time : ℕ) (bob_time carol_time : ℕ) : 
  alice_time = 40 →
  bob_time = alice_time / 4 →
  carol_time = 2 * bob_time →
  (bob_time = 10 ∧ carol_time = 20) := by
  sorry

end cleaning_times_l1410_141063


namespace number_of_colors_l1410_141085

/-- A crayon factory produces crayons of different colors. -/
structure CrayonFactory where
  /-- Number of crayons of each color in a box -/
  crayons_per_color_per_box : ℕ
  /-- Number of boxes filled per hour -/
  boxes_per_hour : ℕ
  /-- Total number of crayons produced in 4 hours -/
  total_crayons_4hours : ℕ

/-- Theorem stating the number of colors produced by the factory -/
theorem number_of_colors (factory : CrayonFactory)
    (h1 : factory.crayons_per_color_per_box = 2)
    (h2 : factory.boxes_per_hour = 5)
    (h3 : factory.total_crayons_4hours = 160) :
    (factory.total_crayons_4hours / (4 * factory.boxes_per_hour * factory.crayons_per_color_per_box) : ℕ) = 4 := by
  sorry

end number_of_colors_l1410_141085


namespace triangle_pqr_properties_l1410_141052

/-- Triangle PQR with vertices P(-2,3), Q(4,5), and R(1,-4), and point S(p,q) inside the triangle such that triangles PQS, QRS, and RPS have equal areas -/
structure TrianglePQR where
  P : ℝ × ℝ := (-2, 3)
  Q : ℝ × ℝ := (4, 5)
  R : ℝ × ℝ := (1, -4)
  S : ℝ × ℝ
  equal_areas : True  -- Placeholder for the equal areas condition

/-- The coordinates of point S -/
def point_S (t : TrianglePQR) : ℝ × ℝ := t.S

/-- The perimeter of triangle PQR -/
noncomputable def perimeter (t : TrianglePQR) : ℝ :=
  Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58

/-- Main theorem about the triangle PQR and point S -/
theorem triangle_pqr_properties (t : TrianglePQR) :
  point_S t = (1, 4/3) ∧
  10 * (point_S t).1 + (point_S t).2 = 34/3 ∧
  perimeter t = Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58 ∧
  34/3 < perimeter t :=
by sorry

end triangle_pqr_properties_l1410_141052


namespace triangle_area_problem_l1410_141059

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end triangle_area_problem_l1410_141059


namespace unseen_area_30_40_l1410_141002

/-- Represents a rectangular room with guards in opposite corners. -/
structure GuardedRoom where
  length : ℝ
  width : ℝ
  guard1_pos : ℝ × ℝ
  guard2_pos : ℝ × ℝ

/-- Calculates the area of the room that neither guard can see. -/
def unseen_area (room : GuardedRoom) : ℝ :=
  sorry

/-- Theorem stating that for a room of 30m x 40m with guards in opposite corners,
    the unseen area is 225 m². -/
theorem unseen_area_30_40 :
  let room : GuardedRoom := {
    length := 30,
    width := 40,
    guard1_pos := (0, 0),
    guard2_pos := (30, 40)
  }
  unseen_area room = 225 := by sorry

end unseen_area_30_40_l1410_141002


namespace reciprocal_of_negative_2021_l1410_141073

theorem reciprocal_of_negative_2021 :
  let reciprocal (x : ℚ) := 1 / x
  reciprocal (-2021) = -1 / 2021 := by
  sorry

end reciprocal_of_negative_2021_l1410_141073


namespace consecutive_integers_average_l1410_141033

theorem consecutive_integers_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 →
  (a + b + c + d + e) / 5 = 8 →
  e - a = 4 →
  (b + d) / 2 = 8 := by
sorry

end consecutive_integers_average_l1410_141033


namespace fruit_sales_theorem_l1410_141038

/-- The standard weight of a batch of fruits in kilograms -/
def standard_weight : ℕ := 30

/-- The weight deviations from the standard weight -/
def weight_deviations : List ℤ := [9, -10, -5, 6, -7, -6, 7, 10]

/-- The price per kilogram on the first day in yuan -/
def price_per_kg : ℕ := 10

/-- The discount rate for the second day as a rational number -/
def discount_rate : ℚ := 1/10

theorem fruit_sales_theorem :
  let total_weight := (List.sum weight_deviations + standard_weight * weight_deviations.length : ℤ)
  let first_day_sales := (price_per_kg * (total_weight / 2) : ℚ)
  let second_day_sales := (price_per_kg * (1 - discount_rate) * (total_weight - total_weight / 2) : ℚ)
  total_weight = 244 ∧ (first_day_sales + second_day_sales : ℚ) = 2318 := by
  sorry

end fruit_sales_theorem_l1410_141038


namespace cloud_counting_l1410_141090

theorem cloud_counting (carson_clouds : ℕ) (brother_multiplier : ℕ) : 
  carson_clouds = 6 → 
  brother_multiplier = 3 → 
  carson_clouds + carson_clouds * brother_multiplier = 24 :=
by sorry

end cloud_counting_l1410_141090


namespace remaining_payment_theorem_l1410_141005

def calculate_remaining_payment (deposit : ℚ) (percentage : ℚ) : ℚ :=
  deposit / percentage - deposit

def total_remaining_payment (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ) : ℚ :=
  calculate_remaining_payment deposit1 percentage1 +
  calculate_remaining_payment deposit2 percentage2 +
  calculate_remaining_payment deposit3 percentage3

theorem remaining_payment_theorem (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ)
  (h1 : deposit1 = 105)
  (h2 : deposit2 = 180)
  (h3 : deposit3 = 300)
  (h4 : percentage1 = 1/10)
  (h5 : percentage2 = 15/100)
  (h6 : percentage3 = 1/5) :
  total_remaining_payment deposit1 deposit2 deposit3 percentage1 percentage2 percentage3 = 3165 := by
  sorry

#eval total_remaining_payment 105 180 300 (1/10) (15/100) (1/5)

end remaining_payment_theorem_l1410_141005


namespace first_number_is_202_l1410_141042

def numbers : List ℕ := [202, 204, 205, 206, 209, 209, 210, 212]

theorem first_number_is_202 (x : ℕ) 
  (h : (numbers.sum + x) / 9 = 207) : 
  numbers.head? = some 202 := by
  sorry

end first_number_is_202_l1410_141042


namespace entrance_fee_is_five_l1410_141021

/-- The entrance fee per person for a concert, given the following conditions:
  * Tickets cost $50.00 each
  * There's a 15% processing fee for tickets
  * There's a $10.00 parking fee
  * The total cost for two people is $135.00
-/
def entrance_fee : ℝ := by
  sorry

theorem entrance_fee_is_five : entrance_fee = 5 := by
  sorry

end entrance_fee_is_five_l1410_141021


namespace vins_bike_distance_l1410_141088

/-- Calculates the total distance ridden in a week given daily distances and number of days -/
def total_distance (to_school : ℕ) (from_school : ℕ) (days : ℕ) : ℕ :=
  (to_school + from_school) * days

/-- Proves that given the specific distances and number of days, the total distance is 65 miles -/
theorem vins_bike_distance : total_distance 6 7 5 = 65 := by
  sorry

end vins_bike_distance_l1410_141088


namespace min_slope_tangent_line_l1410_141039

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f' x ≥ f' (-1)) ∧
    (f' (-1) = 3) ∧
    (f (-1) = -5) ∧
    (a * x + b * y + c = 0) ∧
    (a / b = f' (-1)) ∧
    ((-1) * a + (-5) * b + c = 0) ∧
    (a = 3 ∧ b = -1 ∧ c = -2) :=
by sorry

end min_slope_tangent_line_l1410_141039


namespace twenty_five_binary_l1410_141027

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twenty_five_binary :
  toBinary 25 = [true, false, false, true, true] :=
by sorry

end twenty_five_binary_l1410_141027


namespace dodecahedron_edge_probability_l1410_141041

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  valid_edge : ∀ e ∈ edges, e.1 ≠ e.2
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_probability (d : Dodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge
    in a regular dodecahedron with 20 vertices is 3/19 -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry


end dodecahedron_edge_probability_l1410_141041


namespace no_real_roots_l1410_141091

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (3 * x + 9) + 8 / Real.sqrt (3 * x + 9) = 4 := by
  sorry

end no_real_roots_l1410_141091


namespace suitcase_profit_l1410_141076

/-- Calculates the total profit and profit per suitcase for a store selling suitcases. -/
theorem suitcase_profit (num_suitcases : ℕ) (purchase_price : ℕ) (total_revenue : ℕ) :
  num_suitcases = 60 →
  purchase_price = 100 →
  total_revenue = 8100 →
  (total_revenue - num_suitcases * purchase_price = 2100) ∧
  ((total_revenue - num_suitcases * purchase_price) / num_suitcases = 35) := by
  sorry

#check suitcase_profit

end suitcase_profit_l1410_141076


namespace optimal_launch_angle_l1410_141098

/-- 
Given a target at horizontal distance A and height B, 
the angle α that minimizes the initial speed of a projectile to hit the target 
is given by α = arctan((B + √(A² + B²))/A).
-/
theorem optimal_launch_angle (A B : ℝ) (hA : A > 0) (hB : B ≥ 0) :
  let C := Real.sqrt (A^2 + B^2)
  let α := Real.arctan ((B + C) / A)
  ∀ θ : ℝ, 
    0 < θ ∧ θ < π / 2 → 
    (Real.sin θ)^2 * (A^2 + B^2) ≤ (Real.sin (2*α)) * (A^2 + B^2) / 2 :=
by sorry

end optimal_launch_angle_l1410_141098


namespace range_of_c_l1410_141014

/-- Given c > 0, if the function y = c^x is decreasing on ℝ and the minimum value of f(x) = x^2 - c^2 
    is no greater than -1/16, then 1/4 ≤ c < 1 -/
theorem range_of_c (c : ℝ) (hc : c > 0) 
  (hp : ∀ (x y : ℝ), x < y → c^x > c^y) 
  (hq : ∃ (k : ℝ), ∀ (x : ℝ), x^2 - c^2 ≥ k ∧ k ≤ -1/16) : 
  1/4 ≤ c ∧ c < 1 := by
sorry

end range_of_c_l1410_141014


namespace y_derivative_l1410_141093

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) - Real.sqrt 2 * Real.log (1 + x)

theorem y_derivative (x : ℝ) (h : x ≠ -1) : 
  deriv y x = (1 - x) / Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) := by
  sorry

end y_derivative_l1410_141093


namespace ellipse_dot_product_bound_l1410_141013

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the function to be bounded
def f (P Q : ℝ × ℝ) : ℝ :=
  dot_product (P.1, P.2) (Q.1, Q.2) + 
  dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

-- Theorem statement
theorem ellipse_dot_product_bound :
  ∀ P Q : ℝ × ℝ, C P.1 P.2 → C Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ f P Q ∧ f P Q ≤ -52/3 :=
sorry

end ellipse_dot_product_bound_l1410_141013


namespace substitution_result_l1410_141064

theorem substitution_result (x y : ℝ) :
  (y = x - 1) ∧ (x - 2*y = 7) → (x - 2*x + 2 = 7) := by sorry

end substitution_result_l1410_141064


namespace identity_proof_special_case_proof_l1410_141011

-- Define the sequence f_n = a^n + b^n
def f (a b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => a^(n+1) + b^(n+1)

theorem identity_proof (a b : ℝ) (n : ℕ) :
  f a b (n + 1) = (a + b) * (f a b n) - a * b * (f a b (n - 1)) :=
by sorry

theorem special_case_proof (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  f a b 10 = 123 :=
by sorry

end identity_proof_special_case_proof_l1410_141011


namespace room_area_in_sqm_l1410_141087

-- Define the room dimensions
def room_length : Real := 18
def room_width : Real := 9

-- Define the conversion factor
def sqft_to_sqm : Real := 10.7639

-- Theorem statement
theorem room_area_in_sqm :
  let area_sqft := room_length * room_width
  let area_sqm := area_sqft / sqft_to_sqm
  ⌊area_sqm⌋ = 15 := by sorry

end room_area_in_sqm_l1410_141087


namespace window_purchase_savings_l1410_141018

/-- Represents the store's window sale offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyCount : ℕ
  freeCount : ℕ

/-- Calculates the cost of purchasing a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyCount + offer.freeCount)
  let remainder := windowsNeeded % (offer.buyCount + offer.freeCount)
  let windowsPaidFor := fullSets * offer.buyCount + min remainder offer.buyCount
  windowsPaidFor * offer.regularPrice

/-- The main theorem stating the savings when Dave and Doug purchase windows together -/
theorem window_purchase_savings : 
  let offer : WindowOffer := ⟨100, 3, 1⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  let totalWindows : ℕ := davesWindows + dougsWindows
  let separateCost : ℕ := calculateCost offer davesWindows + calculateCost offer dougsWindows
  let combinedCost : ℕ := calculateCost offer totalWindows
  let savings : ℕ := separateCost - combinedCost
  savings = 600 := by sorry

end window_purchase_savings_l1410_141018


namespace max_value_of_f_l1410_141060

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end max_value_of_f_l1410_141060


namespace hannah_practice_hours_l1410_141058

/-- Hannah's weekend practice hours -/
def weekend_hours : ℕ := sorry

theorem hannah_practice_hours : 
  (weekend_hours + (weekend_hours + 17) = 33) → 
  weekend_hours = 8 := by sorry

end hannah_practice_hours_l1410_141058


namespace equation_solution_l1410_141040

def solution_set : Set ℝ := {0, -6}

theorem equation_solution :
  ∀ x : ℝ, (2 * |x + 3| - 4 = 2) ↔ x ∈ solution_set := by
  sorry

#check equation_solution

end equation_solution_l1410_141040


namespace magic_square_sum_l1410_141026

/-- Represents a 3x3 magic square --/
structure MagicSquare :=
  (a b c d e : ℕ)
  (row1_sum : 30 + d + 24 = 32 + e + b)
  (row2_sum : 20 + e + b = 32 + e + b)
  (row3_sum : c + 32 + a = 32 + e + b)
  (col1_sum : 30 + 20 + c = 32 + e + b)
  (col2_sum : d + e + 32 = 32 + e + b)
  (col3_sum : 24 + b + a = 32 + e + b)
  (diag1_sum : 30 + e + a = 32 + e + b)
  (diag2_sum : 24 + e + c = 32 + e + b)

/-- The sum of d and e in the magic square is 54 --/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end magic_square_sum_l1410_141026


namespace three_digit_number_problem_l1410_141083

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def reverse_number (h t u : ℕ) : ℕ := u * 100 + t * 10 + h

theorem three_digit_number_problem (h t u : ℕ) :
  is_single_digit h ∧
  is_single_digit t ∧
  u = h + 6 ∧
  u + h = 16 ∧
  (h * 100 + t * 10 + u + reverse_number h t u) % 10 = 6 ∧
  ((h * 100 + t * 10 + u + reverse_number h t u) / 10) % 10 = 9 →
  h = 5 ∧ t = 5 ∧ u = 11 := by
sorry

end three_digit_number_problem_l1410_141083


namespace weight_of_a_l1410_141035

/-- Given the weights of 5 people A, B, C, D, and E, prove that A weighs 64 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  (b + c + d + e) / 4 = 79 →
  a = 64 := by
sorry

end weight_of_a_l1410_141035


namespace arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l1410_141029

theorem arithmetic_geometric_inequality_two_vars (a b : ℝ) (h : a ≤ b) :
  a^2 + b^2 ≥ 2 * a * b := by sorry

theorem arithmetic_geometric_inequality_three_vars (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by sorry

end arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l1410_141029


namespace fifth_set_fraction_approx_three_fourths_l1410_141097

-- Define the duration of the whole match in minutes
def whole_match_duration : ℕ := 665

-- Define the duration of the fifth set in minutes
def fifth_set_duration : ℕ := 491

-- Define a function to calculate the fraction
def match_fraction : ℚ := fifth_set_duration / whole_match_duration

-- Define what we consider as "approximately equal" (e.g., within 0.02)
def approximately_equal (x y : ℚ) : Prop := abs (x - y) < 1/50

-- Theorem statement
theorem fifth_set_fraction_approx_three_fourths :
  approximately_equal match_fraction (3/4) :=
sorry

end fifth_set_fraction_approx_three_fourths_l1410_141097


namespace emmas_speed_last_segment_l1410_141019

def total_distance : ℝ := 150
def total_time : ℝ := 2
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 75
def num_segments : ℕ := 3

theorem emmas_speed_last_segment (speed_segment3 : ℝ) : 
  (speed_segment1 + speed_segment2 + speed_segment3) / num_segments = total_distance / total_time →
  speed_segment3 = 100 := by
sorry

end emmas_speed_last_segment_l1410_141019


namespace median_length_right_triangle_l1410_141024

theorem median_length_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (1 / 2 : ℝ) * c
  median = 5 := by
sorry

end median_length_right_triangle_l1410_141024


namespace fraction_identity_l1410_141067

theorem fraction_identity (a b c : ℝ) 
  (h1 : a + b + c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0)
  (h5 : (a + b + c)⁻¹ = a⁻¹ + b⁻¹ + c⁻¹) :
  (a^5 + b^5 + c^5)⁻¹ = a⁻¹^5 + b⁻¹^5 + c⁻¹^5 := by
  sorry

end fraction_identity_l1410_141067


namespace fraction_simplification_l1410_141084

theorem fraction_simplification : (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end fraction_simplification_l1410_141084


namespace defeat_monster_time_l1410_141007

/-- The time required to defeat a monster given the attack rates of two Ultramen and the monster's durability. -/
theorem defeat_monster_time 
  (monster_durability : ℕ) 
  (ultraman1_rate : ℕ) 
  (ultraman2_rate : ℕ) 
  (h1 : monster_durability = 100)
  (h2 : ultraman1_rate = 12)
  (h3 : ultraman2_rate = 8) : 
  (monster_durability : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 := by
  sorry

end defeat_monster_time_l1410_141007


namespace f_derivative_l1410_141020

noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.cos x

theorem f_derivative (x : ℝ) (h : x > 0) : 
  deriv f x = 1 / x - Real.sin x := by sorry

end f_derivative_l1410_141020


namespace sum_of_bases_equals_999_l1410_141046

/-- Converts a number from base 11 to base 10 -/
def base11To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 12 to base 10 -/
def base12To10 (n : ℕ) : ℕ := sorry

/-- Represents the digit A in base 12 -/
def A : ℕ := 10

theorem sum_of_bases_equals_999 :
  base11To10 379 + base12To10 (3 * 12^2 + A * 12 + 9) = 999 := by sorry

end sum_of_bases_equals_999_l1410_141046


namespace solution_value_l1410_141066

theorem solution_value (x m : ℝ) : x = 3 ∧ (11 - 2*x = m*x - 1) → m = 2 := by
  sorry

end solution_value_l1410_141066


namespace right_triangle_set_l1410_141074

theorem right_triangle_set (a b c : ℝ) : 
  (a = 1.5 ∧ b = 2 ∧ c = 2.5) → 
  a^2 + b^2 = c^2 ∧
  ¬(4^2 + 5^2 = 6^2) ∧
  ¬(1^2 + (Real.sqrt 2)^2 = 2.5^2) ∧
  ¬(2^2 + 3^2 = 4^2) :=
by sorry

end right_triangle_set_l1410_141074


namespace valid_placements_count_l1410_141061

/-- Represents a ball -/
inductive Ball : Type
| A : Ball
| B : Ball
| C : Ball
| D : Ball

/-- Represents a box -/
inductive Box : Type
| one : Box
| two : Box
| three : Box

/-- A placement of balls into boxes -/
def Placement := Ball → Box

/-- Checks if a placement is valid -/
def isValidPlacement (p : Placement) : Prop :=
  (∀ b : Box, ∃ ball : Ball, p ball = b) ∧ 
  (p Ball.A ≠ p Ball.B)

/-- The number of valid placements -/
def numValidPlacements : ℕ := sorry

theorem valid_placements_count : numValidPlacements = 30 := by sorry

end valid_placements_count_l1410_141061


namespace min_days_to_triple_debt_l1410_141053

/-- The borrowed amount in dollars -/
def borrowed_amount : ℝ := 15

/-- The daily interest rate as a decimal -/
def daily_interest_rate : ℝ := 0.1

/-- Calculate the amount owed after a given number of days -/
def amount_owed (days : ℝ) : ℝ :=
  borrowed_amount * (1 + daily_interest_rate * days)

/-- The minimum number of days needed to owe at least triple the borrowed amount -/
def min_days : ℕ := 20

theorem min_days_to_triple_debt :
  (∀ d : ℕ, d < min_days → amount_owed d < 3 * borrowed_amount) ∧
  amount_owed min_days ≥ 3 * borrowed_amount :=
sorry

end min_days_to_triple_debt_l1410_141053


namespace craig_total_distance_l1410_141017

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := 0.2

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_home : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := distance_school_to_david + distance_david_to_home

/-- Theorem stating that the total distance Craig walked is 0.9 miles -/
theorem craig_total_distance : total_distance = 0.9 := by
  sorry

end craig_total_distance_l1410_141017


namespace inscribed_quadrilateral_perimeter_bound_l1410_141050

/-- A rectangle in 2D space -/
structure Rectangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  is_rectangle : sorry

/-- A quadrilateral inscribed in a rectangle -/
structure InscribedQuadrilateral (rect : Rectangle) where
  k : ℝ × ℝ
  l : ℝ × ℝ
  m : ℝ × ℝ
  n : ℝ × ℝ
  on_sides : sorry

/-- Calculate the perimeter of a quadrilateral -/
def perimeter (q : InscribedQuadrilateral rect) : ℝ := sorry

/-- Calculate the length of the diagonal of a rectangle -/
def diagonal_length (rect : Rectangle) : ℝ := sorry

/-- Theorem: The perimeter of an inscribed quadrilateral is at least twice the diagonal of the rectangle -/
theorem inscribed_quadrilateral_perimeter_bound (rect : Rectangle) (q : InscribedQuadrilateral rect) :
  perimeter q ≥ 2 * diagonal_length rect := by sorry

end inscribed_quadrilateral_perimeter_bound_l1410_141050


namespace pablo_works_seven_hours_l1410_141037

/-- Represents the puzzle-solving scenario for Pablo --/
structure PuzzleScenario where
  pieces_per_hour : ℕ
  small_puzzles : ℕ
  small_puzzle_pieces : ℕ
  large_puzzles : ℕ
  large_puzzle_pieces : ℕ
  days_to_complete : ℕ

/-- Calculates the hours Pablo works on puzzles each day --/
def hours_per_day (scenario : PuzzleScenario) : ℚ :=
  let total_pieces := scenario.small_puzzles * scenario.small_puzzle_pieces +
                      scenario.large_puzzles * scenario.large_puzzle_pieces
  let total_hours := total_pieces / scenario.pieces_per_hour
  total_hours / scenario.days_to_complete

/-- Theorem stating that Pablo works 7 hours per day on puzzles --/
theorem pablo_works_seven_hours (scenario : PuzzleScenario) 
  (h1 : scenario.pieces_per_hour = 100)
  (h2 : scenario.small_puzzles = 8)
  (h3 : scenario.small_puzzle_pieces = 300)
  (h4 : scenario.large_puzzles = 5)
  (h5 : scenario.large_puzzle_pieces = 500)
  (h6 : scenario.days_to_complete = 7) :
  hours_per_day scenario = 7 := by
  sorry

end pablo_works_seven_hours_l1410_141037


namespace salary_increase_after_two_years_l1410_141006

-- Define the raise percentage
def raise_percentage : ℝ := 0.05

-- Define the number of six-month periods in two years
def periods : ℕ := 4

-- Theorem stating the salary increase after two years
theorem salary_increase_after_two_years :
  let final_multiplier := (1 + raise_percentage) ^ periods
  abs (final_multiplier - 1 - 0.2155) < 0.0001 := by
  sorry

end salary_increase_after_two_years_l1410_141006


namespace expression_evaluation_l1410_141079

theorem expression_evaluation :
  let a : ℤ := -4
  (4 * a^2 - 3*a) - (2 * a^2 + a - 1) + (2 - a^2 + 4*a) = 19 := by
  sorry

end expression_evaluation_l1410_141079


namespace cubic_sum_of_quadratic_roots_l1410_141094

theorem cubic_sum_of_quadratic_roots :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + 4*x₁ + 2 = 0) →
  (x₂^2 + 4*x₂ + 2 = 0) →
  (x₁ ≠ x₂) →
  x₁^3 + 14*x₂ + 55 = 7 :=
by sorry

end cubic_sum_of_quadratic_roots_l1410_141094


namespace average_stamps_is_25_l1410_141070

/-- Calculates the average number of stamps collected per day -/
def average_stamps_collected (days : ℕ) (initial_stamps : ℕ) (daily_increase : ℕ) : ℚ :=
  let total_stamps := (days : ℚ) / 2 * (2 * initial_stamps + (days - 1) * daily_increase)
  total_stamps / days

/-- Proves that the average number of stamps collected per day is 25 -/
theorem average_stamps_is_25 :
  average_stamps_collected 6 10 6 = 25 := by
sorry

end average_stamps_is_25_l1410_141070


namespace smallest_difference_in_triangle_l1410_141072

theorem smallest_difference_in_triangle (XZ XY YZ : ℕ) : 
  XZ + XY + YZ = 3030 →
  XZ < XY →
  XY ≤ YZ →
  ∃ k : ℕ, XY = 5 * k →
  ∀ XZ' XY' YZ' : ℕ, 
    XZ' + XY' + YZ' = 3030 →
    XZ' < XY' →
    XY' ≤ YZ' →
    (∃ k' : ℕ, XY' = 5 * k') →
    XY - XZ ≤ XY' - XZ' :=
by sorry

end smallest_difference_in_triangle_l1410_141072


namespace original_group_size_is_correct_l1410_141030

/-- Represents the number of men in the original group -/
def original_group_size : ℕ := 22

/-- Represents the number of days the original group planned to work -/
def original_days : ℕ := 20

/-- Represents the number of men who became absent -/
def absent_men : ℕ := 2

/-- Represents the number of days the remaining group worked -/
def actual_days : ℕ := 22

/-- Theorem stating that the original group size is correct given the conditions -/
theorem original_group_size_is_correct :
  (original_group_size : ℚ) * (actual_days : ℚ) * ((original_group_size - absent_men) : ℚ) = 
  (original_group_size : ℚ) * (original_group_size : ℚ) * (original_days : ℚ) :=
by sorry

end original_group_size_is_correct_l1410_141030


namespace one_third_of_seven_point_two_l1410_141044

theorem one_third_of_seven_point_two :
  (7.2 : ℚ) / 3 = 2 + 2 / 5 := by sorry

end one_third_of_seven_point_two_l1410_141044


namespace positive_sum_inequality_sqrt_difference_inequality_l1410_141028

-- Problem 1
theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by sorry

-- Problem 2
theorem sqrt_difference_inequality (n : ℕ+) :
  Real.sqrt (n + 1) - Real.sqrt n > Real.sqrt (n + 2) - Real.sqrt (n + 1) := by sorry

end positive_sum_inequality_sqrt_difference_inequality_l1410_141028


namespace hyperbola_equation_l1410_141031

/-- Given a hyperbola with eccentricity e = √6/2 and the area of rectangle OMPN equal to √2,
    which is also equal to (1/2)ab, prove that the equation of the hyperbola is x^2/4 - y^2/2 = 1. -/
theorem hyperbola_equation (e a b : ℝ) (h1 : e = Real.sqrt 6 / 2) 
    (h2 : (1/2) * a * b = Real.sqrt 2) : 
    ∀ (x y : ℝ), x^2/4 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
by sorry

end hyperbola_equation_l1410_141031


namespace cost_per_metre_l1410_141069

theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) (h1 : total_length = 9.25) (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end cost_per_metre_l1410_141069


namespace eggs_per_year_is_3380_l1410_141086

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let weeks_per_year : ℕ := 52
  
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  let eggs_per_week : ℕ := eggs_per_day * days_per_week
  
  eggs_per_week * weeks_per_year

theorem eggs_per_year_is_3380 : eggs_per_year = 3380 := by
  sorry

end eggs_per_year_is_3380_l1410_141086


namespace decimal_to_binary_nineteen_l1410_141015

theorem decimal_to_binary_nineteen : 
  (1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 19 := by
  sorry

end decimal_to_binary_nineteen_l1410_141015


namespace line_point_ratio_l1410_141001

/-- Given four points A, B, C, D on a directed line such that AC/CB + AD/DB = 0,
    prove that 1/AC + 1/AD = 2/AB -/
theorem line_point_ratio (A B C D : ℝ) (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) :
  1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end line_point_ratio_l1410_141001


namespace square_minus_product_equals_one_l1410_141016

theorem square_minus_product_equals_one : 1999^2 - 2000 * 1998 = 1 := by
  sorry

end square_minus_product_equals_one_l1410_141016


namespace parabola_vertex_l1410_141004

/-- The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 1 → 
  ∃ (vertex_x vertex_y : ℝ), 
    vertex_x = 1 ∧ 
    vertex_y = 4 ∧ 
    ∀ (x' : ℝ), -3 * x'^2 + 6 * x' + 1 ≤ vertex_y :=
by sorry

end parabola_vertex_l1410_141004


namespace valid_placement_iff_even_l1410_141092

/-- Represents a chessboard with one corner cut off -/
structure Chessboard (n : ℕ) :=
  (size : ℕ := 2*n + 1)
  (corner_cut : Bool := true)

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (n : ℕ) :=
  (board : Chessboard n)
  (total_dominos : ℕ)
  (horizontal_dominos : ℕ)

/-- Checks if a domino placement is valid -/
def is_valid_placement (n : ℕ) (placement : DominoPlacement n) : Prop :=
  placement.total_dominos * 2 = placement.board.size^2 - 1 ∧
  placement.horizontal_dominos * 2 = placement.total_dominos

/-- The main theorem stating the condition for valid placement -/
theorem valid_placement_iff_even (n : ℕ) :
  (∃ (placement : DominoPlacement n), is_valid_placement n placement) ↔ Even n :=
sorry

end valid_placement_iff_even_l1410_141092


namespace husband_age_is_54_l1410_141049

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens ≤ 9)
  (h2 : ones ≤ 9)

/-- Converts an Age to its numerical value -/
def Age.toNat (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h2, a.h1⟩

theorem husband_age_is_54 (wife : Age) (husband : Age) :
  husband = wife.reverse →
  husband.toNat > wife.toNat →
  husband.toNat - wife.toNat = (husband.toNat + wife.toNat) / 11 →
  husband.toNat = 54 := by
  sorry

end husband_age_is_54_l1410_141049


namespace cos_thirty_degrees_l1410_141022

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l1410_141022


namespace number_equation_solution_l1410_141089

theorem number_equation_solution : 
  ∃ x : ℝ, (10 * x = 2 * x - 36) ∧ (x = -4.5) := by
  sorry

end number_equation_solution_l1410_141089


namespace ellipse_slope_product_l1410_141057

/-- The ellipse C with semi-major axis a and semi-minor axis b -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop :=
  Real.sqrt 7 * x - Real.sqrt 5 * y + 12 = 0

/-- The point A -/
def A : ℝ × ℝ := (-4, 0)

/-- The point R -/
def R : ℝ × ℝ := (3, 0)

/-- The vertical line that M and N lie on -/
def vertical_line (x : ℝ) : Prop :=
  x = 16/3

theorem ellipse_slope_product (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/4) 
  (h4 : ∃ (x y : ℝ), ellipse b b x y ∧ tangent_line x y) :
  ∃ (P Q M N : ℝ × ℝ) (k1 k2 : ℝ),
    ellipse a b P.1 P.2 ∧
    ellipse a b Q.1 Q.2 ∧
    vertical_line M.1 ∧
    vertical_line N.1 ∧
    k1 * k2 = -12/7 :=
  sorry

end ellipse_slope_product_l1410_141057


namespace simplify_sqrt_18_l1410_141012

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_18_l1410_141012


namespace fixed_point_of_linear_function_l1410_141008

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem fixed_point_of_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  linear_function k b (-3) = -2 := by
  sorry

end fixed_point_of_linear_function_l1410_141008


namespace train_length_calculation_l1410_141095

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time, calculate the length of the train. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 160 := by
  sorry

#check train_length_calculation

end train_length_calculation_l1410_141095


namespace clock_rotation_proof_l1410_141081

/-- The number of large divisions on a clock face -/
def clock_divisions : ℕ := 12

/-- The number of degrees in one large division -/
def degrees_per_division : ℝ := 30

/-- The number of hours between 3 o'clock and 6 o'clock -/
def hours_elapsed : ℕ := 3

/-- The degree of rotation of the hour hand from 3 o'clock to 6 o'clock -/
def hour_hand_rotation : ℝ := hours_elapsed * degrees_per_division

theorem clock_rotation_proof :
  hour_hand_rotation = 90 :=
by sorry

end clock_rotation_proof_l1410_141081


namespace smallest_angle_in_special_triangle_l1410_141047

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a + b + c = 180 →
  c = 5 * a →
  b = 3 * a →
  a = 20 :=
by
  sorry

end smallest_angle_in_special_triangle_l1410_141047


namespace magnitude_relationship_l1410_141032

-- Define the equations for a, b, and c
def equation_a (x : ℝ) : Prop := 2^x + x = 1
def equation_b (x : ℝ) : Prop := 2^x + x = 2
def equation_c (x : ℝ) : Prop := 3^x + x = 2

-- State the theorem
theorem magnitude_relationship (a b c : ℝ) 
  (ha : equation_a a) (hb : equation_b b) (hc : equation_c c) : 
  a < c ∧ c < b := by
  sorry

end magnitude_relationship_l1410_141032


namespace no_winning_strategy_strategy_independent_no_strategy_better_than_half_l1410_141099

/-- Represents a deck of cards with red and black suits -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- A strategy is a function that decides whether to stop based on the current deck state -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state -/
def winProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

/-- Theorem stating that no strategy can achieve a winning probability greater than 0.5 -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d ≤ 1/2 := by
  sorry

/-- Theorem stating that the winning probability is independent of the strategy -/
theorem strategy_independent (d : Deck) (s₁ s₂ : Strategy) :
  winProbability d = winProbability d := by
  sorry

/-- Main theorem: No strategy exists that guarantees a winning probability greater than 0.5 -/
theorem no_strategy_better_than_half (d : Deck) :
  d.red = d.black → ∀ s : Strategy, winProbability d ≤ 1/2 := by
  sorry

end no_winning_strategy_strategy_independent_no_strategy_better_than_half_l1410_141099


namespace oil_for_rest_of_bike_l1410_141082

/-- Proves the amount of oil needed for the rest of the bike --/
theorem oil_for_rest_of_bike 
  (oil_per_wheel : ℝ) 
  (num_wheels : ℕ) 
  (total_oil : ℝ) 
  (h1 : oil_per_wheel = 10)
  (h2 : num_wheels = 2)
  (h3 : total_oil = 25) :
  total_oil - (oil_per_wheel * num_wheels) = 5 := by
sorry

end oil_for_rest_of_bike_l1410_141082
