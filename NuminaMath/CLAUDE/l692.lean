import Mathlib

namespace percentage_of_liars_l692_69255

theorem percentage_of_liars (truth_speakers : ℝ) (both_speakers : ℝ) (truth_or_lie_prob : ℝ) :
  truth_speakers = 0.3 →
  both_speakers = 0.1 →
  truth_or_lie_prob = 0.4 →
  ∃ (lie_speakers : ℝ), lie_speakers = 0.2 ∧ 
    truth_or_lie_prob = truth_speakers + lie_speakers - both_speakers :=
by sorry

end percentage_of_liars_l692_69255


namespace negation_of_implication_l692_69287

theorem negation_of_implication :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x = 3 ∧ x^2 - 2*x - 3 ≠ 0) :=
by sorry

end negation_of_implication_l692_69287


namespace tangent_line_b_value_l692_69208

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_b_value :
  ∀ a k b : ℝ,
  f a 2 = 3 →                        -- The curve passes through (2, 3)
  f' a 2 = k →                       -- The slope of the tangent line at x = 2
  3 = k * 2 + b →                    -- The tangent line passes through (2, 3)
  b = -15 := by sorry

end tangent_line_b_value_l692_69208


namespace floor_sqrt_50_squared_l692_69225

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l692_69225


namespace T_is_three_rays_l692_69201

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set T as described in the problem -/
def T : Set Point :=
  {p : Point | (4 = p.x + 1 ∧ p.y - 5 ≤ 4) ∨
               (4 = p.y - 5 ∧ p.x + 1 ≤ 4) ∨
               (p.x + 1 = p.y - 5 ∧ 4 ≤ p.x + 1)}

/-- A ray starting from a point in a given direction -/
structure Ray where
  start : Point
  direction : ℝ × ℝ

/-- The three rays that should describe T -/
def threeRays : List Ray :=
  [{ start := ⟨3, 9⟩, direction := (0, -1) },   -- Vertically downward
   { start := ⟨3, 9⟩, direction := (-1, 0) },   -- Horizontally leftward
   { start := ⟨3, 9⟩, direction := (1, 1) }]    -- Diagonally upward

/-- Theorem stating that T is equivalent to three rays with a common point -/
theorem T_is_three_rays : 
  ∀ p : Point, p ∈ T ↔ ∃ r ∈ threeRays, ∃ t : ℝ, t ≥ 0 ∧ 
    p.x = r.start.x + t * r.direction.1 ∧ 
    p.y = r.start.y + t * r.direction.2 :=
sorry

end T_is_three_rays_l692_69201


namespace checkerboard_triangle_area_theorem_l692_69227

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle on a 2D grid -/
structure GridTriangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The area of a triangle -/
def triangleArea (t : GridTriangle) : ℝ :=
  sorry

/-- Whether two triangles are similar -/
def areSimilar (t1 t2 : GridTriangle) : Prop :=
  sorry

/-- The area of the white part of a triangle -/
def whiteArea (t : GridTriangle) : ℝ :=
  sorry

/-- The area of the black part of a triangle -/
def blackArea (t : GridTriangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem checkerboard_triangle_area_theorem (X : GridTriangle) (S : ℝ) 
  (h : triangleArea X = S) : 
  ∃ Y : GridTriangle, areSimilar X Y ∧ whiteArea Y = S ∧ blackArea Y = S :=
sorry

end checkerboard_triangle_area_theorem_l692_69227


namespace last_digit_to_appear_is_six_l692_69297

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppearedBy (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Main theorem
theorem last_digit_to_appear_is_six :
  ∃ n : ℕ, (∀ d : ℕ, d < 10 → digitAppearedBy d n) ∧
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d < 10 → digitAppearedBy d m)) ∧
  unitsDigit (fib n) = 6 :=
sorry

end last_digit_to_appear_is_six_l692_69297


namespace green_ball_probability_l692_69290

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨8, 2⟩
def Z : Container := ⟨5, 5⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting a green ball -/
def probabilityGreenBall : ℚ :=
  (List.sum (containers.map greenProbability)) / containers.length

theorem green_ball_probability :
  probabilityGreenBall = 7 / 15 := by
  sorry

end green_ball_probability_l692_69290


namespace bus_row_capacity_l692_69272

/-- Represents a bus with a given number of rows and total capacity -/
structure Bus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of children each row can accommodate -/
def childrenPerRow (bus : Bus) : ℕ := bus.capacity / bus.rows

/-- Theorem: Given a bus with 9 rows and a capacity of 36 children,
    prove that each row can accommodate 4 children -/
theorem bus_row_capacity (bus : Bus) 
    (h_rows : bus.rows = 9) 
    (h_capacity : bus.capacity = 36) : 
    childrenPerRow bus = 4 := by
  sorry

end bus_row_capacity_l692_69272


namespace third_quadrant_condition_l692_69258

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 1) (a + 1)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- Theorem statement
theorem third_quadrant_condition (a : ℝ) :
  in_third_quadrant (z a) ↔ a < -1 := by
  sorry

end third_quadrant_condition_l692_69258


namespace smallest_two_digit_prime_reverse_composite_l692_69281

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  if n < 10 then n else
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_reverse_composite :
  ∃ p : ℕ, 
    p ≥ 10 ∧ p < 100 ∧  -- two-digit number
    is_prime p ∧
    is_composite (reverse_digits p) ∧
    p / 10 ≤ 3 ∧  -- starts with a digit less than or equal to 3
    (∀ q : ℕ, q ≥ 10 ∧ q < p ∧ is_prime q ∧ q / 10 ≤ 3 → ¬(is_composite (reverse_digits q))) ∧
    p = 23 :=
by sorry

end smallest_two_digit_prime_reverse_composite_l692_69281


namespace tan_difference_l692_69261

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end tan_difference_l692_69261


namespace negation_of_existence_negation_of_cubic_equation_l692_69213

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ ∀ x : ℝ, x^3 - 2*x + 1 ≠ 0 := by sorry

end negation_of_existence_negation_of_cubic_equation_l692_69213


namespace line_perp_plane_criterion_l692_69211

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_criterion 
  (α β γ : Plane) (m n l : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end line_perp_plane_criterion_l692_69211


namespace quadratic_equation_with_given_roots_l692_69228

theorem quadratic_equation_with_given_roots :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = 0 ↔ x = -5 ∨ x = 7) →
  (∀ x, f x = (x + 5) * (x - 7)) :=
by
  sorry

end quadratic_equation_with_given_roots_l692_69228


namespace descending_order_l692_69219

-- Define the numbers in their respective bases
def a : ℕ := 3 * 16 + 14
def b : ℕ := 2 * 6^2 + 1 * 6 + 0
def c : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4 + 0
def d : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1

-- Theorem statement
theorem descending_order : b > c ∧ c > a ∧ a > d := by
  sorry

end descending_order_l692_69219


namespace tangent_square_area_l692_69223

/-- Given a 6 by 6 square with semicircles on its sides, prove the area of the tangent square ABCD -/
theorem tangent_square_area :
  -- Original square side length
  let original_side : ℝ := 6
  -- Radius of semicircles (half of original side)
  let semicircle_radius : ℝ := original_side / 2
  -- Side length of square ABCD (original side + 2 * radius)
  let abcd_side : ℝ := original_side + 2 * semicircle_radius
  -- Area of square ABCD
  let abcd_area : ℝ := abcd_side ^ 2
  -- The area of square ABCD is 144
  abcd_area = 144 := by sorry

end tangent_square_area_l692_69223


namespace rect_to_spherical_conversion_l692_69291

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical_conversion
  (x y z : ℝ)
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h_x : x = 0)
  (h_y : y = -3 * Real.sqrt 3)
  (h_z : z = 3)
  (h_ρ_val : ρ = 6)
  (h_θ_val : θ = 3 * Real.pi / 2)
  (h_φ_val : φ = Real.pi / 3) :
  x = ρ * Real.sin φ * Real.cos θ ∧
  y = ρ * Real.sin φ * Real.sin θ ∧
  z = ρ * Real.cos φ :=
by
  sorry

#check rect_to_spherical_conversion

end rect_to_spherical_conversion_l692_69291


namespace extra_fruits_l692_69270

def red_apples_ordered : ℕ := 60
def green_apples_ordered : ℕ := 34
def bananas_ordered : ℕ := 25
def oranges_ordered : ℕ := 45

def red_apple_students : ℕ := 3
def green_apple_students : ℕ := 2
def banana_students : ℕ := 5
def orange_students : ℕ := 10

def red_apples_per_student : ℕ := 2
def green_apples_per_student : ℕ := 2
def bananas_per_student : ℕ := 2
def oranges_per_student : ℕ := 1

theorem extra_fruits :
  red_apples_ordered - red_apple_students * red_apples_per_student +
  green_apples_ordered - green_apple_students * green_apples_per_student +
  bananas_ordered - banana_students * bananas_per_student +
  oranges_ordered - orange_students * oranges_per_student = 134 := by
  sorry

end extra_fruits_l692_69270


namespace simplify_and_evaluate_l692_69263

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  1 / (x - 1) - 2 / (x^2 - 1) = -1 := by sorry

end simplify_and_evaluate_l692_69263


namespace boat_round_trip_equation_l692_69257

/-- Represents the equation for a boat's round trip between two points -/
def boat_equation (distance : ℝ) (flow_speed : ℝ) (boat_speed : ℝ) (total_time : ℝ) : Prop :=
  (distance / (boat_speed + flow_speed)) + (distance / (boat_speed - flow_speed)) = total_time

/-- Theorem stating that the given equation correctly represents the boat's round trip -/
theorem boat_round_trip_equation : 
  ∀ (x : ℝ), x > 5 → boat_equation 60 5 x 8 :=
by sorry

end boat_round_trip_equation_l692_69257


namespace max_three_match_winners_200_l692_69267

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament --/
def total_matches (t : Tournament) : ℕ :=
  t.participants - 1

/-- Calculates the maximum number of participants who can win at least 3 matches --/
def max_participants_with_three_wins (t : Tournament) : ℕ :=
  (total_matches t) / 3

/-- Theorem stating the maximum number of participants who can win at least 3 matches
    in a tournament with 200 participants --/
theorem max_three_match_winners_200 :
  ∃ (t : Tournament), t.participants = 200 ∧ max_participants_with_three_wins t = 66 :=
by
  sorry


end max_three_match_winners_200_l692_69267


namespace triangle_angle_ratio_l692_69209

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- All angles are positive
  a + b + c = 180 →        -- Sum of angles is 180 degrees
  a = 20 →                 -- Smallest angle is 20 degrees
  c = 5 * a →              -- Largest angle is 5 times the smallest
  a ≤ b ∧ b ≤ c →          -- a is smallest, c is largest
  b / a = 3 :=             -- Ratio of middle to smallest is 3:1
by sorry

end triangle_angle_ratio_l692_69209


namespace select_cubes_eq_31_l692_69246

/-- The number of ways to select 10 cubes from a set of 7 red cubes, 3 blue cubes, and 9 green cubes -/
def select_cubes : ℕ :=
  let red_cubes := 7
  let blue_cubes := 3
  let green_cubes := 9
  let total_selected := 10
  (Finset.range (red_cubes + 1)).sum (λ r => 
    (Finset.range (blue_cubes + 1)).sum (λ b => 
      let g := total_selected - r - b
      if g ≥ 0 ∧ g ≤ green_cubes then 1 else 0
    )
  )

theorem select_cubes_eq_31 : select_cubes = 31 := by sorry

end select_cubes_eq_31_l692_69246


namespace lower_limit_of_g_l692_69204

-- Define the function f(n)
def f (n : ℕ) : ℕ := Finset.prod (Finset.range (n^2 - 3)) (λ i => i + 4)

-- Define the function g(n) with a parameter m for the lower limit
def g (n m : ℕ) : ℕ := Finset.prod (Finset.range (n - m + 1)) (λ i => (i + m)^2)

-- State the theorem
theorem lower_limit_of_g : ∃ m : ℕ, 
  m = 2 ∧ 
  (∀ n : ℕ, n ≥ m → g n m ≠ 0) ∧
  (∃ k : ℕ, (f 3 / g 3 m).factorization 2 = 4) :=
sorry

end lower_limit_of_g_l692_69204


namespace p_range_q_range_p_or_q_false_range_l692_69244

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2/(1-2*m) + y^2/(m+2) = 1 ∧ (1-2*m)*(m+2) < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m ≤ -2 ∨ m ≥ 1 :=
sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -2 ∨ m > 1/2 :=
sorry

-- Theorem for the range of m when "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -2 < m ∧ m ≤ 1/2 :=
sorry

end p_range_q_range_p_or_q_false_range_l692_69244


namespace train_distance_theorem_l692_69269

/-- Calculates the distance a train can travel given its coal efficiency and remaining coal. -/
def train_distance (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) : ℚ :=
  (coal_remaining / pounds_per_unit) * miles_per_unit

/-- Proves that a train with given efficiency and coal amount can travel the calculated distance. -/
theorem train_distance_theorem (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) :
  miles_per_unit = 5 → pounds_per_unit = 2 → coal_remaining = 160 →
  train_distance miles_per_unit pounds_per_unit coal_remaining = 400 := by
  sorry

#check train_distance_theorem

end train_distance_theorem_l692_69269


namespace melany_candy_l692_69293

theorem melany_candy (hugh tommy melany : ℕ) (total_after : ℕ) :
  hugh = 8 →
  tommy = 6 →
  total_after = 7 * 3 →
  hugh + tommy + melany = total_after →
  melany = 7 :=
by sorry

end melany_candy_l692_69293


namespace relationship_abc_l692_69284

theorem relationship_abc :
  let a := Real.tan (135 * π / 180)
  let b := Real.cos (Real.cos 0)
  let c := (fun x : ℝ => (x^2 + 1/2)^0) 0
  c > b ∧ b > a := by sorry

end relationship_abc_l692_69284


namespace intersection_of_M_and_N_l692_69276

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l692_69276


namespace expression_factorization_l692_69268

theorem expression_factorization (y : ℝ) : 
  5 * y * (y - 2) + 10 * (y - 2) - 15 * (y - 2) = 5 * (y - 2) * (y - 1) := by
  sorry

end expression_factorization_l692_69268


namespace trapezoid_perimeter_l692_69238

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (EF : ℝ)
  (GH : ℝ)
  (EG : ℝ)
  (FH : ℝ)
  (is_trapezoid : EF ≠ GH)

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + t.EG + t.FH

/-- Theorem: The perimeter of the given trapezoid is 38 units -/
theorem trapezoid_perimeter :
  ∃ (t : Trapezoid), t.EF = 10 ∧ t.GH = 14 ∧ t.EG = 7 ∧ t.FH = 7 ∧ perimeter t = 38 :=
by
  sorry

end trapezoid_perimeter_l692_69238


namespace solution_set_inequality_l692_69294

theorem solution_set_inequality (a b : ℝ) :
  ({x : ℝ | a * x^2 - 5 * x + b > 0} = {x : ℝ | -3 < x ∧ x < 2}) →
  ({x : ℝ | b * x^2 - 5 * x + a > 0} = {x : ℝ | x < -3 ∨ x > 2}) :=
by sorry

end solution_set_inequality_l692_69294


namespace max_absolute_value_complex_l692_69234

theorem max_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - 8 * Complex.I) = 20) :
  Complex.abs z ≤ Real.sqrt 222 :=
sorry

end max_absolute_value_complex_l692_69234


namespace inequality_proof_l692_69215

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a + 1 / b > b + 1 / a := by sorry

end inequality_proof_l692_69215


namespace triangle_is_right_angled_l692_69245

/-- A triangle is right-angled if the square of its longest side equals the sum of squares of the other two sides. -/
def IsRightAngled (a b c : ℝ) : Prop :=
  (a ≥ b ∧ a ≥ c ∧ a^2 = b^2 + c^2) ∨
  (b ≥ a ∧ b ≥ c ∧ b^2 = a^2 + c^2) ∨
  (c ≥ a ∧ c ≥ b ∧ c^2 = a^2 + b^2)

/-- Given three real numbers a, b, and c that satisfy the equation
    a^2 + b^2 + c^2 - 12a - 16b - 20c + 200 = 0,
    prove that they form a right-angled triangle. -/
theorem triangle_is_right_angled (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) :
  IsRightAngled a b c :=
sorry

end triangle_is_right_angled_l692_69245


namespace product_ratio_theorem_l692_69285

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 3/4 := by
  sorry

end product_ratio_theorem_l692_69285


namespace baseball_gear_cost_l692_69279

def initial_amount : ℕ := 67
def amount_left : ℕ := 33

theorem baseball_gear_cost :
  initial_amount - amount_left = 34 :=
by sorry

end baseball_gear_cost_l692_69279


namespace closest_root_is_point_four_l692_69266

/-- Quadratic function f(x) = 3x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 6 * x + c

/-- The constant c in the quadratic function -/
def c : ℝ := 2.24  -- f(0) = 2.24, so c = 2.24

theorem closest_root_is_point_four :
  let options : List ℝ := [0.2, 0.4, 0.6, 0.8]
  ∃ (root : ℝ), f c root = 0 ∧
    ∀ (x : ℝ), x ∈ options → |x - root| ≥ |0.4 - root| :=
by sorry

end closest_root_is_point_four_l692_69266


namespace tan_45_degrees_l692_69222

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l692_69222


namespace expected_red_pairs_50_cards_l692_69286

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h_total : total = red + black)
  (h_equal : red = black)

/-- The expected number of adjacent red pairs in a circular arrangement -/
def expected_red_pairs (d : Deck) : ℚ :=
  (d.red : ℚ) * ((d.red - 1) / (d.total - 1))

theorem expected_red_pairs_50_cards :
  ∃ d : Deck, d.total = 50 ∧ expected_red_pairs d = 600 / 49 := by
  sorry

end expected_red_pairs_50_cards_l692_69286


namespace minimal_adjective_f_25_l692_69241

/-- A function g: ℤ → ℤ is adjective if g(m) + g(n) > max(m², n²) for any integers m and n -/
def Adjective (g : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, g m + g n > max (m ^ 2) (n ^ 2)

/-- The sum of f(1) to f(30) -/
def SumF (f : ℤ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f (i + 1))

/-- f is an adjective function that minimizes SumF -/
def IsMinimalAdjective (f : ℤ → ℤ) : Prop :=
  Adjective f ∧ ∀ g : ℤ → ℤ, Adjective g → SumF f ≤ SumF g

theorem minimal_adjective_f_25 (f : ℤ → ℤ) (hf : IsMinimalAdjective f) : f 25 ≥ 498 := by
  sorry

end minimal_adjective_f_25_l692_69241


namespace first_interest_rate_is_five_percent_l692_69242

-- Define the total amount, amounts lent at each rate, and the known interest rate
def total_amount : ℝ := 2500
def amount_first_rate : ℝ := 2000
def amount_second_rate : ℝ := total_amount - amount_first_rate
def second_rate : ℝ := 6

-- Define the total yearly annual income
def total_income : ℝ := 130

-- Define the first interest rate as a variable
variable (first_rate : ℝ)

-- Theorem statement
theorem first_interest_rate_is_five_percent :
  (amount_first_rate * first_rate / 100 + amount_second_rate * second_rate / 100 = total_income) →
  first_rate = 5 := by
sorry

end first_interest_rate_is_five_percent_l692_69242


namespace smallest_angle_satisfying_equation_l692_69206

theorem smallest_angle_satisfying_equation :
  let y := Real.pi / 18
  (∀ z : Real, 0 < z ∧ z < y → Real.sin (4 * z) * Real.sin (5 * z) ≠ Real.cos (4 * z) * Real.cos (5 * z)) ∧
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) := by
  sorry

end smallest_angle_satisfying_equation_l692_69206


namespace angles_with_same_terminal_side_l692_69236

theorem angles_with_same_terminal_side (θ : Real) :
  θ = 150 * Real.pi / 180 →
  {β : Real | ∃ k : ℤ, β = 5 * Real.pi / 6 + 2 * k * Real.pi} =
  {β : Real | ∃ k : ℤ, β = θ + 2 * k * Real.pi} :=
by sorry

end angles_with_same_terminal_side_l692_69236


namespace simplify_trig_expression_l692_69251

theorem simplify_trig_expression (x : ℝ) (h : 1 + Real.sin x + Real.cos x ≠ 0) :
  (1 + Real.sin x - Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.tan (x / 2) := by
  sorry

end simplify_trig_expression_l692_69251


namespace quadratic_roots_imaginary_l692_69224

theorem quadratic_roots_imaginary (a b c a₁ b₁ c₁ : ℝ) : 
  let discriminant := 4 * ((a * a₁ + b * b₁ + c * c₁)^2 - (a^2 + b^2 + c^2) * (a₁^2 + b₁^2 + c₁^2))
  discriminant ≤ 0 ∧ 
  (discriminant = 0 ↔ ∃ (k : ℝ), k ≠ 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) := by
  sorry


end quadratic_roots_imaginary_l692_69224


namespace smallest_fraction_l692_69232

theorem smallest_fraction : 
  let a := 7 / 15
  let b := 5 / 11
  let c := 16 / 33
  let d := 49 / 101
  let e := 89 / 183
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e :=
by sorry

end smallest_fraction_l692_69232


namespace card_partition_theorem_l692_69271

/-- Represents a card with a number written on it -/
structure Card where
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- The sum of numbers on a stack of cards -/
def stackSum (s : Stack) : Nat :=
  s.map (λ c => c.number) |>.sum

theorem card_partition_theorem (n k : Nat) (cards : List Card) :
  (∀ c ∈ cards, c.number ≤ n) →
  (cards.map (λ c => c.number)).sum = k * n.factorial →
  ∃ (partition : List Stack),
    partition.length = k ∧
    partition.all (λ s => stackSum s = n.factorial) ∧
    partition.join = cards :=
  sorry

end card_partition_theorem_l692_69271


namespace birdhouse_distance_l692_69226

/-- Proves that the birdhouse distance is 1200 feet given the problem conditions --/
theorem birdhouse_distance (car_distance : ℝ) (car_speed_mph : ℝ) 
  (lawn_chair_distance_multiplier : ℝ) (lawn_chair_time_multiplier : ℝ)
  (birdhouse_distance_multiplier : ℝ) (birdhouse_speed_percentage : ℝ) :
  car_distance = 200 →
  car_speed_mph = 80 →
  lawn_chair_distance_multiplier = 2 →
  lawn_chair_time_multiplier = 1.5 →
  birdhouse_distance_multiplier = 3 →
  birdhouse_speed_percentage = 0.6 →
  (birdhouse_distance_multiplier * lawn_chair_distance_multiplier * car_distance) = 1200 := by
  sorry

#check birdhouse_distance

end birdhouse_distance_l692_69226


namespace ratio_problem_l692_69221

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 := by
  sorry

end ratio_problem_l692_69221


namespace log_c_27_is_0_75_implies_c_is_81_l692_69252

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_c_27_is_0_75_implies_c_is_81 :
  ∀ c : ℝ, c > 0 → log c 27 = 0.75 → c = 81 := by
  sorry

end log_c_27_is_0_75_implies_c_is_81_l692_69252


namespace fraction_subtraction_l692_69233

theorem fraction_subtraction (x : ℝ) (hx : x ≠ 0) : 1 / x - 2 / (3 * x) = 1 / (3 * x) := by
  sorry

end fraction_subtraction_l692_69233


namespace cupcakes_frosted_l692_69220

def cagney_rate : ℚ := 1 / 24
def lacey_rate : ℚ := 1 / 30
def casey_rate : ℚ := 1 / 40
def working_time : ℕ := 6 * 60  -- 6 minutes in seconds

theorem cupcakes_frosted :
  (cagney_rate + lacey_rate + casey_rate) * working_time = 36 := by
  sorry

end cupcakes_frosted_l692_69220


namespace yoongi_calculation_l692_69229

theorem yoongi_calculation (x : ℕ) : 
  (x ≥ 10 ∧ x < 100) → (x - 35 = 27) → (x - 53 = 9) := by
  sorry

end yoongi_calculation_l692_69229


namespace smallest_angle_tangent_equation_l692_69275

theorem smallest_angle_tangent_equation (x : Real) : 
  (x > 0) →
  (Real.tan (6 * x * Real.pi / 180) = 
    (Real.cos (2 * x * Real.pi / 180) - Real.sin (2 * x * Real.pi / 180)) / 
    (Real.cos (2 * x * Real.pi / 180) + Real.sin (2 * x * Real.pi / 180))) →
  x = 5.625 := by
sorry

end smallest_angle_tangent_equation_l692_69275


namespace hydrangea_spend_1989_to_2021_l692_69260

/-- The amount spent on hydrangeas from a start year to an end year -/
def hydrangeaSpend (startYear endYear : ℕ) (pricePerPlant : ℚ) : ℚ :=
  (endYear - startYear + 1 : ℕ) * pricePerPlant

/-- Theorem stating the total spend on hydrangeas from 1989 to 2021 -/
theorem hydrangea_spend_1989_to_2021 :
  hydrangeaSpend 1989 2021 20 = 640 := by
  sorry

end hydrangea_spend_1989_to_2021_l692_69260


namespace jadens_estimate_l692_69254

theorem jadens_estimate (p q δ γ : ℝ) 
  (h1 : p > q) 
  (h2 : q > 0) 
  (h3 : δ > γ) 
  (h4 : γ > 0) : 
  (p + δ) - (q - γ) > p - q := by
  sorry

end jadens_estimate_l692_69254


namespace highest_number_on_paper_l692_69231

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.010416666666666666 → n = 96 := by
  sorry

end highest_number_on_paper_l692_69231


namespace optimal_price_reduction_maximizes_profit_l692_69296

/-- Profit function for shirt sales based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x + 20) * (40 - x)

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 15

theorem optimal_price_reduction_maximizes_profit :
  ∀ x : ℝ, 0 ≤ x → x ≤ 40 → profit x ≤ profit optimal_reduction := by
  sorry

#check optimal_price_reduction_maximizes_profit

end optimal_price_reduction_maximizes_profit_l692_69296


namespace class_mean_score_l692_69292

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = 50 →
  first_day_students = 40 →
  second_day_students = 10 →
  first_day_mean = 80 / 100 →
  second_day_mean = 90 / 100 →
  let overall_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  overall_mean = 82 / 100 := by
sorry

end class_mean_score_l692_69292


namespace distance_and_intersection_l692_69239

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 3)

theorem distance_and_intersection :
  let distance := Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2)
  let slope := (point2.2 - point1.2) / (point2.1 - point1.1)
  let y_intercept := point1.2 - slope * point1.1
  let line := fun x => slope * x + y_intercept
  (distance = 4 * Real.sqrt 5) ∧
  (line (-1) ≠ 0) :=
by sorry

end distance_and_intersection_l692_69239


namespace specific_figure_perimeter_l692_69205

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  rectangleWidth : ℕ
  rectangleHeight : ℕ
  lShapeOutward : ℕ
  lShapeDownward : ℕ

/-- Calculate the perimeter of the SquareFigure -/
def calculatePerimeter (figure : SquareFigure) : ℝ :=
  let bottomLength := figure.rectangleWidth * figure.squareSideLength
  let topLength := (figure.rectangleWidth + figure.lShapeOutward) * figure.squareSideLength
  let leftHeight := figure.rectangleHeight * figure.squareSideLength
  let rightHeight := (figure.rectangleHeight + figure.lShapeDownward) * figure.squareSideLength
  bottomLength + topLength + leftHeight + rightHeight

/-- Theorem stating that the perimeter of the specific figure is 26 units -/
theorem specific_figure_perimeter :
  let figure : SquareFigure := {
    squareSideLength := 2
    rectangleWidth := 3
    rectangleHeight := 2
    lShapeOutward := 2
    lShapeDownward := 1
  }
  calculatePerimeter figure = 26 := by
  sorry

end specific_figure_perimeter_l692_69205


namespace square_of_difference_l692_69218

theorem square_of_difference (x : ℝ) : (8 - Real.sqrt (x^2 + 64))^2 = x^2 + 128 - 16 * Real.sqrt (x^2 + 64) := by
  sorry

end square_of_difference_l692_69218


namespace lucy_calculation_mistake_l692_69282

theorem lucy_calculation_mistake (a b c : ℝ) 
  (h1 : a / (b * c) = 4)
  (h2 : (a / b) / c = 12)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a / b = 4 * Real.sqrt 3 := by
  sorry

end lucy_calculation_mistake_l692_69282


namespace sum_of_fourth_powers_l692_69288

theorem sum_of_fourth_powers (x y : ℕ+) : x^4 + y^4 = 4721 → x + y = 13 := by
  sorry

end sum_of_fourth_powers_l692_69288


namespace perpendicular_vectors_k_value_l692_69278

/-- Given vectors a, b, and c in ℝ², prove that if k*a + 2*b is perpendicular to c,
    then k = -17/3 -/
theorem perpendicular_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (-1, 5))
    (h3 : c = (2, -3))
    (h4 : (k * a.1 + 2 * b.1, k * a.2 + 2 * b.2) • c = 0) :
  k = -17/3 := by
  sorry

end perpendicular_vectors_k_value_l692_69278


namespace museum_visitors_l692_69274

theorem museum_visitors (V : ℕ) : 
  (∃ E : ℕ, 
    (E + 150 = V) ∧ 
    (E = (3 * V) / 4)) → 
  V = 600 := by
sorry

end museum_visitors_l692_69274


namespace necessary_condition_k_l692_69277

theorem necessary_condition_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) ∧
  (∃ x : ℝ, (x < k ∨ x > k + 2) ∧ ¬(-4 < x ∧ x < 1)) ↔
  k ≤ -6 ∨ k ≥ 1 :=
sorry

end necessary_condition_k_l692_69277


namespace ranas_speed_l692_69265

/-- Proves that Rana's speed is 5 kmph given the problem conditions -/
theorem ranas_speed (circumference : ℝ) (ajith_speed : ℝ) (meeting_time : ℝ) 
  (h1 : circumference = 115)
  (h2 : ajith_speed = 4)
  (h3 : meeting_time = 115) :
  ∃ v : ℝ, v = 5 ∧ 
    (v * meeting_time - ajith_speed * meeting_time) / circumference = 1 :=
by sorry

end ranas_speed_l692_69265


namespace route_down_is_24_miles_l692_69289

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_up

/-- Theorem: The length of the route down the mountain is 24 miles -/
theorem route_down_is_24_miles (trip : HikingTrip)
  (h1 : trip.rate_up = 8)
  (h2 : trip.time_up = 2)
  (h3 : trip.rate_down_factor = 1.5) :
  route_down_length trip = 24 := by
  sorry

end route_down_is_24_miles_l692_69289


namespace trigonometric_identity_l692_69217

theorem trigonometric_identity : 
  Real.sin (135 * π / 180) * Real.cos (-15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end trigonometric_identity_l692_69217


namespace grandpa_mingming_age_ratio_l692_69253

theorem grandpa_mingming_age_ratio :
  let grandpa_age : ℕ := 65
  let mingming_age : ℕ := 5
  let next_year_ratio : ℕ := (grandpa_age + 1) / (mingming_age + 1)
  next_year_ratio = 11 := by
  sorry

end grandpa_mingming_age_ratio_l692_69253


namespace triangle_ABC_is_obtuse_l692_69243

theorem triangle_ABC_is_obtuse (A B C : Real) (hA : A = 10) (hB : B = 60) 
  (hsum : A + B + C = 180) : C > 90 := by
  sorry

end triangle_ABC_is_obtuse_l692_69243


namespace simplify_expression_1_simplify_expression_2_l692_69264

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 0) :
  (4 * x^2) / (x^2 - y^2) / (x / (x + y)) = 4 * x / (x - y) := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) - 1 = 1 / (m - 1) := by sorry

end simplify_expression_1_simplify_expression_2_l692_69264


namespace longest_tape_l692_69295

theorem longest_tape (minji seungyeon hyesu : ℝ) 
  (h_minji : minji = 0.74)
  (h_seungyeon : seungyeon = 13/20)
  (h_hyesu : hyesu = 4/5) :
  hyesu > minji ∧ hyesu > seungyeon :=
by sorry

end longest_tape_l692_69295


namespace interview_probability_l692_69240

def total_students : ℕ := 30
def french_students : ℕ := 20
def spanish_students : ℕ := 24

theorem interview_probability :
  let both_classes := french_students + spanish_students - total_students
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let unfavorable_combinations := only_french.choose 2 + only_spanish.choose 2
  (total_combinations - unfavorable_combinations : ℚ) / total_combinations = 25 / 29 := by
  sorry

end interview_probability_l692_69240


namespace shift_left_one_unit_l692_69273

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

theorem shift_left_one_unit (p : Parabola) :
  p.a = 2 ∧ p.b = 0 ∧ p.c = -1 →
  let p_shifted := shift_horizontal p 1
  p_shifted.a = 2 ∧ p_shifted.b = 4 ∧ p_shifted.c = 1 :=
by sorry

end shift_left_one_unit_l692_69273


namespace polygonal_chain_existence_l692_69283

-- Define a type for points in a plane
def Point := ℝ × ℝ

-- Define a type for lines in a plane
def Line := Point → Point → Prop

-- Define a type for a polygonal chain
def PolygonalChain (n : ℕ) := Fin (n + 1) → Point

-- Define the property of n lines in a plane
def LinesInPlane (n : ℕ) (lines : Fin n → Line) : Prop :=
  -- No two lines are parallel
  ∀ i j, i ≠ j → ¬ (∀ p q, lines i p q ↔ lines j p q) ∧
  -- No three lines intersect at a single point
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ ∃ p, (lines i p p ∧ lines j p p ∧ lines k p p)

-- Define the property of a non-self-intersecting polygonal chain
def NonSelfIntersecting (chain : PolygonalChain n) : Prop :=
  ∀ i j k l, i < j → j < k → k < l → 
    ¬ (∃ p, (chain i = p ∧ chain j = p) ∨ (chain k = p ∧ chain l = p))

-- Define the property that each line contains exactly one segment of the chain
def EachLineOneSegment (n : ℕ) (lines : Fin n → Line) (chain : PolygonalChain n) : Prop :=
  ∀ i, ∃! j, lines i (chain j) (chain (j + 1))

-- The main theorem
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h : LinesInPlane n lines) :
  ∃ chain : PolygonalChain n, NonSelfIntersecting chain ∧ EachLineOneSegment n lines chain :=
sorry

end polygonal_chain_existence_l692_69283


namespace dividend_calculation_l692_69202

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 16)
  (h_quotient : quotient = 9)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 149 := by
  sorry

end dividend_calculation_l692_69202


namespace complex_exponential_sum_l692_69262

theorem complex_exponential_sum (α β γ : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) + Complex.exp (Complex.I * γ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) + Complex.exp (-Complex.I * γ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end complex_exponential_sum_l692_69262


namespace abs_sum_equals_five_l692_69203

theorem abs_sum_equals_five (a b c : ℝ) 
  (h1 : a^2 - b*c = 14)
  (h2 : b^2 - c*a = 14)
  (h3 : c^2 - a*b = -3) :
  |a + b + c| = 5 := by
  sorry

end abs_sum_equals_five_l692_69203


namespace thickness_after_four_folds_l692_69248

def blanket_thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

theorem thickness_after_four_folds :
  blanket_thickness 3 4 = 48 := by
  sorry

end thickness_after_four_folds_l692_69248


namespace intersection_of_sets_l692_69249

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3, 4}
  let N : Set ℕ := {0, 1, 2, 3}
  M ∩ N = {1, 2, 3} := by sorry

end intersection_of_sets_l692_69249


namespace complex_equality_l692_69256

theorem complex_equality (a b : ℝ) (h : Complex.I * (a + Complex.I) = b - Complex.I) : a - b = 0 := by
  sorry

end complex_equality_l692_69256


namespace geometric_sequence_tangent_l692_69212

/-- Given a geometric sequence {a_n} where a_2 * a_3 * a_4 = -a_7^2 = -64,
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tangent (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                            -- given condition
  a 7^2 = 64 →                                          -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end geometric_sequence_tangent_l692_69212


namespace sum_of_odd_integers_13_to_41_l692_69298

theorem sum_of_odd_integers_13_to_41 :
  let first_term : ℕ := 13
  let last_term : ℕ := 41
  let common_difference : ℕ := 2
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℝ) / 2 * (first_term + last_term) = 405 :=
by sorry

end sum_of_odd_integers_13_to_41_l692_69298


namespace max_value_theorem_l692_69210

/-- Given a line ax + 2by - 1 = 0 intercepting a chord of length 2√3 on the circle x^2 + y^2 = 4,
    the maximum value of 3a + 2b is √10. -/
theorem max_value_theorem (a b : ℝ) : 
  (∃ x y : ℝ, a * x + 2 * b * y - 1 = 0 ∧ x^2 + y^2 = 4) →  -- Line intersects circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, a * x₁ + 2 * b * y₁ - 1 = 0 ∧ 
                     a * x₂ + 2 * b * y₂ - 1 = 0 ∧
                     x₁^2 + y₁^2 = 4 ∧
                     x₂^2 + y₂^2 = 4 ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →  -- Chord length is 2√3
  (a^2 + 4 * b^2 = 1) →  -- Distance from center to line is 1
  (∀ c : ℝ, 3 * a + 2 * b ≤ c → c ≥ Real.sqrt 10) ∧ 
  (∃ a₀ b₀ : ℝ, 3 * a₀ + 2 * b₀ = Real.sqrt 10) :=
by sorry

end max_value_theorem_l692_69210


namespace inequality_system_solution_l692_69230

theorem inequality_system_solution (x : ℝ) :
  (x - 4 > (3/2) * x - 3) ∧
  ((2 + x) / 3 - 1 ≤ (1 + x) / 2) →
  -5 ≤ x ∧ x < -2 := by
  sorry

end inequality_system_solution_l692_69230


namespace greatest_marble_difference_is_six_l692_69237

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two natural numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The greatest difference between marble counts in any basket is 6 -/
theorem greatest_marble_difference_is_six :
  let basketA : Basket := { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 }
  let basketB : Basket := { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 }
  let basketC : Basket := { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }
  let diffA := absDiff basketA.count1 basketA.count2
  let diffB := absDiff basketB.count1 basketB.count2
  let diffC := absDiff basketC.count1 basketC.count2
  (max diffA (max diffB diffC)) = 6 := by
  sorry

end greatest_marble_difference_is_six_l692_69237


namespace shortest_distance_dasha_vasya_l692_69200

-- Define the friends as vertices in a graph
inductive Friend : Type
| Asya : Friend
| Galia : Friend
| Borya : Friend
| Dasha : Friend
| Vasya : Friend

-- Define the distance function between friends
def distance : Friend → Friend → ℕ
| Friend.Asya, Friend.Galia => 12
| Friend.Galia, Friend.Asya => 12
| Friend.Galia, Friend.Borya => 10
| Friend.Borya, Friend.Galia => 10
| Friend.Asya, Friend.Borya => 8
| Friend.Borya, Friend.Asya => 8
| Friend.Dasha, Friend.Galia => 15
| Friend.Galia, Friend.Dasha => 15
| Friend.Vasya, Friend.Galia => 17
| Friend.Galia, Friend.Vasya => 17
| _, _ => 0  -- Default case for undefined distances

-- Define the shortest path function
def shortest_path (a b : Friend) : ℕ := sorry

-- Theorem statement
theorem shortest_distance_dasha_vasya :
  shortest_path Friend.Dasha Friend.Vasya = 18 := by sorry

end shortest_distance_dasha_vasya_l692_69200


namespace pure_imaginary_complex_number_l692_69250

theorem pure_imaginary_complex_number (b : ℝ) : 
  let z : ℂ := (1 + b * Complex.I) * (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I ∧ y ≠ 0) → b = 2 := by
  sorry

end pure_imaginary_complex_number_l692_69250


namespace bowl_glass_pairings_l692_69299

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

/-- Theorem: The number of possible pairings of bowls and glasses is 25 -/
theorem bowl_glass_pairings :
  total_pairings = 25 := by sorry

end bowl_glass_pairings_l692_69299


namespace impossible_11_difference_l692_69280

/-- Represents an L-shaped piece -/
structure LPiece where
  cells : ℕ
  odd_cells : Odd cells

/-- Represents a partition of a square into L-shaped pieces -/
structure Partition where
  pieces : List LPiece
  total_cells : (pieces.map LPiece.cells).sum = 120 * 120

theorem impossible_11_difference (p1 p2 : Partition) : 
  p2.pieces.length ≠ p1.pieces.length + 11 := by
  sorry

end impossible_11_difference_l692_69280


namespace db_length_determined_l692_69247

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CD to AB
def altitudeCD (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xD, yD) := D
  (xD - xA) * (xB - xA) + (yD - yA) * (yB - yA) = 0 ∧
  (xC - xD) * (xB - xA) + (yC - yD) * (yB - yA) = 0

-- Define the altitude AE to BC
def altitudeAE (t : Triangle) (E : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xE, yE) := E
  (xE - xB) * (xC - xB) + (yE - yB) * (yC - yB) = 0 ∧
  (xA - xE) * (xC - xB) + (yA - yE) * (yC - yB) = 0

-- Define the lengths of AB, CD, and AE
def lengthAB (t : Triangle) : ℝ := sorry
def lengthCD (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry
def lengthAE (t : Triangle) (E : ℝ × ℝ) : ℝ := sorry

-- Define the length of DB
def lengthDB (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem db_length_determined (t : Triangle) (D E : ℝ × ℝ) :
  altitudeCD t D → altitudeAE t E →
  ∃! db : ℝ, db = lengthDB t D := by sorry

end db_length_determined_l692_69247


namespace smithtown_handedness_ratio_l692_69259

-- Define the population of Smithtown
structure Population where
  total : ℝ
  men : ℝ
  women : ℝ
  rightHanded : ℝ
  leftHanded : ℝ

-- Define the conditions
def smithtown_conditions (p : Population) : Prop :=
  p.men / p.women = 3 / 2 ∧
  p.men = p.rightHanded ∧
  p.leftHanded / p.total = 0.2500000000000001

-- Theorem statement
theorem smithtown_handedness_ratio (p : Population) :
  smithtown_conditions p →
  p.rightHanded / p.leftHanded = 3 / 1 :=
by sorry

end smithtown_handedness_ratio_l692_69259


namespace two_integers_sum_l692_69216

theorem two_integers_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ x - y = 4 ∧ x * y = 156 → x + y = 28 := by
  sorry

end two_integers_sum_l692_69216


namespace all_shaded_areas_different_l692_69235

/-- Represents a square with its division and shaded area -/
structure Square where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares in the problem -/
def square_I : Square := { total_divisions := 8, shaded_divisions := 3 }
def square_II : Square := { total_divisions := 9, shaded_divisions := 3 }
def square_III : Square := { total_divisions := 8, shaded_divisions := 4 }

/-- Calculate the shaded fraction of a square -/
def shaded_fraction (s : Square) : ℚ :=
  (s.shaded_divisions : ℚ) / (s.total_divisions : ℚ)

/-- Theorem stating that the shaded areas of all three squares are different -/
theorem all_shaded_areas_different :
  shaded_fraction square_I ≠ shaded_fraction square_II ∧
  shaded_fraction square_I ≠ shaded_fraction square_III ∧
  shaded_fraction square_II ≠ shaded_fraction square_III :=
sorry

end all_shaded_areas_different_l692_69235


namespace min_value_a_plus_2b_l692_69207

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  ∀ x y, x > 0 → y > 0 → (x + y) / (x * y) = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

end min_value_a_plus_2b_l692_69207


namespace salary_increase_l692_69214

/-- If a salary increases by 33.33% to $80, prove that the original salary was $60 -/
theorem salary_increase (original : ℝ) (increase_percent : ℝ) (new_salary : ℝ) :
  increase_percent = 33.33 ∧ new_salary = 80 ∧ new_salary = original * (1 + increase_percent / 100) →
  original = 60 := by
  sorry

end salary_increase_l692_69214
