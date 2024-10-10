import Mathlib

namespace triangle_inequality_l1711_171193

/-- Triangle inequality for sides and area -/
theorem triangle_inequality (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : S = Real.sqrt (((a + b + c) / 2) * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
  sorry


end triangle_inequality_l1711_171193


namespace unique_angle_l1711_171197

def is_valid_angle (a b c d e f : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  10 * a + b < 90 ∧
  10 * c + d < 60 ∧
  10 * e + f < 60

def is_complement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  (10 * a + b) + (10 * a1 + b1) = 89 ∧
  (10 * c + d) + (10 * c1 + d1) = 59 ∧
  (10 * e + f) + (10 * e1 + f1) = 60

def is_rearrangement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  ∃ (n m : ℕ), n + m = 6 ∧ n ≤ m ∧
  (10^n + 1) * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) +
  (10^m + 1) * (100000 * a1 + 10000 * b1 + 1000 * c1 + 100 * d1 + 10 * e1 + f1) = 895960

theorem unique_angle :
  ∀ (a b c d e f : ℕ),
    is_valid_angle a b c d e f →
    (∃ (a1 b1 c1 d1 e1 f1 : ℕ),
      is_valid_angle a1 b1 c1 d1 e1 f1 ∧
      is_complement a b c d e f a1 b1 c1 d1 e1 f1 ∧
      is_rearrangement a b c d e f a1 b1 c1 d1 e1 f1) →
    a = 4 ∧ b = 5 ∧ c = 4 ∧ d = 4 ∧ e = 1 ∧ f = 5 :=
by sorry

end unique_angle_l1711_171197


namespace right_triangle_side_length_l1711_171130

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (Q.1 - R.1) / Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5/13) 
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 13) : 
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 12 := by
sorry

end right_triangle_side_length_l1711_171130


namespace joans_kittens_l1711_171116

/-- Given that Joan initially had 8 kittens and received 2 more from her friends,
    prove that she now has 10 kittens in total. -/
theorem joans_kittens (initial : Nat) (received : Nat) (total : Nat) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
sorry

end joans_kittens_l1711_171116


namespace intersection_point_is_unique_l1711_171137

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-1, -2)

/-- First line equation: 2x + 3y + 8 = 0 -/
def line1 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + 3 * p.2 + 8 = 0

/-- Second line equation: x - y - 1 = 0 -/
def line2 (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique point that does so -/
theorem intersection_point_is_unique :
  line1 intersection_point ∧ 
  line2 intersection_point ∧ 
  ∀ p : ℝ × ℝ, line1 p ∧ line2 p → p = intersection_point :=
sorry

end intersection_point_is_unique_l1711_171137


namespace sum_greater_than_product_l1711_171134

theorem sum_greater_than_product (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end sum_greater_than_product_l1711_171134


namespace side_significant_digits_l1711_171166

/-- The area of the square in square meters -/
def area : ℝ := 2.7509

/-- The precision of the area measurement in square meters -/
def precision : ℝ := 0.0001

/-- The number of significant digits in the measurement of the side of the square -/
def significant_digits : ℕ := 5

/-- Theorem stating that the number of significant digits in the measurement of the side of the square is 5 -/
theorem side_significant_digits : 
  ∀ (side : ℝ), side^2 = area → significant_digits = 5 := by
  sorry

end side_significant_digits_l1711_171166


namespace number_problem_l1711_171115

theorem number_problem (x : ℝ) : (x / 6) * 12 = 15 → x = 7.5 := by
  sorry

end number_problem_l1711_171115


namespace range_of_f_l1711_171196

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 2)

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y < 21 ∨ y > 21} :=
by sorry

end range_of_f_l1711_171196


namespace complex_multiplication_opposites_l1711_171133

theorem complex_multiplication_opposites (a : ℝ) (i : ℂ) (h1 : a > 0) (h2 : i * i = -1) :
  (Complex.re (a * i * (a + i)) = -Complex.im (a * i * (a + i))) → a = 1 := by
  sorry

end complex_multiplication_opposites_l1711_171133


namespace sun_division_l1711_171124

theorem sun_division (x y z : ℝ) (total : ℝ) : 
  (y = 0.45 * x) →
  (z = 0.50 * x) →
  (y = 63) →
  (total = x + y + z) →
  total = 273 := by
sorry

end sun_division_l1711_171124


namespace pass_rate_two_steps_l1711_171186

/-- The pass rate of a product going through two independent processing steps -/
def product_pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent
    processing steps with defect rates a and b is (1-a) * (1-b) -/
theorem pass_rate_two_steps (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  product_pass_rate a b = (1 - a) * (1 - b) := by
  sorry

#check pass_rate_two_steps

end pass_rate_two_steps_l1711_171186


namespace set_intersection_and_union_l1711_171135

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem set_intersection_and_union (a : ℝ) :
  (A a) ∩ (B a) = {-3} →
  a = -1 ∧ (A a) ∪ (B a) = {-4, -3, 0, 1, 2} := by sorry

end set_intersection_and_union_l1711_171135


namespace intersection_complement_theorem_l1711_171187

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end intersection_complement_theorem_l1711_171187


namespace plane_equation_through_point_perpendicular_to_vector_l1711_171194

/-- A plane passing through a point and perpendicular to a non-zero vector -/
theorem plane_equation_through_point_perpendicular_to_vector
  (x₀ y₀ z₀ : ℝ) (a b c : ℝ) (h : (a, b, c) ≠ (0, 0, 0)) :
  ∀ x y z : ℝ,
  (a * (x - x₀) + b * (y - y₀) + c * (z - z₀) = 0) ↔
  ((x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀, y₀, z₀) + t • (a, b, c)}ᶜ) :=
by sorry


end plane_equation_through_point_perpendicular_to_vector_l1711_171194


namespace min_distance_squared_l1711_171145

/-- The minimum squared distance between a curve and a line -/
theorem min_distance_squared (a b m n : ℝ) : 
  a > 0 → 
  b = -1/2 * a^2 + 3 * Real.log a → 
  n = 2 * m + 1/2 → 
  ∃ (min_dist : ℝ), 
    (∀ (x y : ℝ), y = -1/2 * x^2 + 3 * Real.log x → 
      (x - m)^2 + (y - n)^2 ≥ min_dist) ∧
    min_dist = 9/5 := by
  sorry

end min_distance_squared_l1711_171145


namespace ellipse_min_sum_l1711_171172

/-- Given an ellipse that passes through a point (a, b), prove the minimum value of m + n -/
theorem ellipse_min_sum (a b m n : ℝ) : 
  m > 0 → n > 0 → m > n → a ≠ 0 → b ≠ 0 → abs a ≠ abs b →
  (a^2 / m^2) + (b^2 / n^2) = 1 →
  ∀ m' n', m' > 0 → n' > 0 → m' > n' → (a^2 / m'^2) + (b^2 / n'^2) = 1 →
  m + n ≤ m' + n' →
  m + n = (a^(2/3) + b^(2/3))^(3/2) :=
by sorry

end ellipse_min_sum_l1711_171172


namespace double_angle_formulas_l1711_171192

open Real

theorem double_angle_formulas (α p q : ℝ) (h : tan α = p / q) :
  sin (2 * α) = (2 * p * q) / (p^2 + q^2) ∧
  cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  tan (2 * α) = (2 * p * q) / (q^2 - p^2) := by
  sorry

end double_angle_formulas_l1711_171192


namespace complex_absolute_value_product_l1711_171179

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 12 * Real.sqrt 35 := by
  sorry

end complex_absolute_value_product_l1711_171179


namespace petya_win_probability_is_1_256_l1711_171159

/-- The "Heap of Stones" game --/
structure HeapOfStones where
  initial_stones : Nat
  max_stones_per_turn : Nat

/-- A player in the game --/
inductive Player
  | Petya
  | Computer

/-- The game state --/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- The result of a game --/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The strategy for the computer player --/
def computer_strategy : GameState → Nat := sorry

/-- The random strategy for Petya --/
def petya_random_strategy : GameState → Nat := sorry

/-- Play a single game --/
def play_game (game : HeapOfStones) : GameResult := sorry

/-- Calculate the probability of Petya winning --/
def petya_win_probability (game : HeapOfStones) : ℚ := sorry

/-- The main theorem --/
theorem petya_win_probability_is_1_256 :
  let game : HeapOfStones := ⟨16, 4⟩
  petya_win_probability game = 1 / 256 := by sorry

end petya_win_probability_is_1_256_l1711_171159


namespace hexagon_same_length_probability_l1711_171188

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements (sides and diagonals) in a regular hexagon -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length from a regular hexagon -/
def prob_same_length : ℚ := 17/35

theorem hexagon_same_length_probability :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length := by
  sorry

end hexagon_same_length_probability_l1711_171188


namespace min_value_vector_expr_l1711_171112

/-- Given plane vectors a, b, and c satisfying certain conditions, 
    the minimum value of a specific vector expression is 1/2. -/
theorem min_value_vector_expr 
  (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖c‖ = 2) 
  (h4 : a + b + c = (0, 0)) :
  ∃ (min : ℝ), min = 1/2 ∧ 
  ∀ (x y : ℝ), 0 ≤ x → x ≤ 1/2 → 1/2 ≤ y → y ≤ 1 →
  ‖x • (a - c) + y • (b - c) + c‖ ≥ min :=
by sorry

end min_value_vector_expr_l1711_171112


namespace complex_fraction_calculation_l1711_171189

theorem complex_fraction_calculation : 
  (2 + 2/3 : ℚ) * ((1/3 - 1/11) / (1/11 + 1/5)) / (8/27) = 7 + 1/2 := by
  sorry

end complex_fraction_calculation_l1711_171189


namespace complex_equation_solution_l1711_171146

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l1711_171146


namespace division_multiplication_result_l1711_171163

theorem division_multiplication_result : 
  let number := 5
  let intermediate := number / 6
  let result := intermediate * 12
  result = 10 := by sorry

end division_multiplication_result_l1711_171163


namespace extreme_values_and_interval_extrema_l1711_171180

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3/2}

theorem extreme_values_and_interval_extrema :
  -- Global maximum
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧
  -- Global minimum
  (∃ (x : ℝ), f x = -2 ∧ ∀ (y : ℝ), f y ≥ f x) ∧
  -- Maximum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  -- Minimum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = -18 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) :=
by sorry


end extreme_values_and_interval_extrema_l1711_171180


namespace quadratic_function_properties_l1711_171122

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (∀ x, f (x + 2) = f (x + 1) + 2*x + 1) ∧
  (∀ m, (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-1) 3) → m ∈ Set.Icc 1 3) :=
by sorry


end quadratic_function_properties_l1711_171122


namespace quadratic_function_theorem_l1711_171181

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is even and has a range of (-∞, 4] -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := fun x ↦ (x + a) * (b * x + 2 * a)

/-- The property of being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of a function -/
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop := ∀ y, y ∈ S ↔ ∃ x, f x = y

theorem quadratic_function_theorem (a b : ℝ) :
  is_even (quadratic_function a b) ∧ 
  has_range (quadratic_function a b) {y | y ≤ 4} →
  quadratic_function a b = fun x ↦ -2 * x^2 + 4 := by
  sorry

end quadratic_function_theorem_l1711_171181


namespace absolute_value_inequality_solution_set_l1711_171175

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 2 * |x - 1| - 1 < 0} = {x : ℝ | 1/2 < x ∧ x < 3/2} := by
  sorry

end absolute_value_inequality_solution_set_l1711_171175


namespace sum_of_integers_l1711_171128

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + 2*r = 10)
  (eq2 : q - r + s = 9)
  (eq3 : r - 2*s + p = 6)
  (eq4 : s - p + q = 7) :
  p + q + r + s = 32 := by sorry

end sum_of_integers_l1711_171128


namespace candy_distribution_properties_l1711_171174

-- Define the people
inductive Person : Type
| Chun : Person
| Tian : Person
| Zhen : Person
| Mei : Person
| Li : Person

-- Define the order of taking candies
def Order := Fin 5 → Person

-- Define the number of candies taken by each person
def CandiesTaken := Person → ℕ

-- Define the properties of the candy distribution
structure CandyDistribution where
  order : Order
  candiesTaken : CandiesTaken
  initialCandies : ℕ
  allDifferent : ∀ (p q : Person), p ≠ q → candiesTaken p ≠ candiesTaken q
  tianHalf : candiesTaken Person.Tian = (initialCandies - candiesTaken Person.Chun) / 2
  zhenTwoThirds : candiesTaken Person.Zhen = 2 * (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Li) / 3
  meiAll : candiesTaken Person.Mei = initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Zhen - candiesTaken Person.Li
  liHalf : candiesTaken Person.Li = (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian) / 2

-- Theorem statement
theorem candy_distribution_properties (d : CandyDistribution) :
  (∃ i : Fin 5, d.order i = Person.Zhen ∧ i = 3) ∧
  d.initialCandies ≥ 16 := by
  sorry

end candy_distribution_properties_l1711_171174


namespace triangle_perimeter_triangle_perimeter_proof_l1711_171118

/-- Given a triangle with sides of lengths 15 cm, 6 cm, and 12 cm, its perimeter is 33 cm. -/
theorem triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 15 ∧ b = 6 ∧ c = 12 ∧
      perimeter = a + b + c ∧
      perimeter = 33

-- The proof is omitted
theorem triangle_perimeter_proof : triangle_perimeter 33 := by sorry

end triangle_perimeter_triangle_perimeter_proof_l1711_171118


namespace negative_x_to_negative_k_is_positive_l1711_171161

theorem negative_x_to_negative_k_is_positive
  (x : ℝ) (k : ℤ) (hx : x < 0) (hk : k > 0) :
  -x^(-k) > 0 :=
by sorry

end negative_x_to_negative_k_is_positive_l1711_171161


namespace max_xy_value_l1711_171143

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3*x*y = 2015) : 
  ∀ a b : ℝ, a^2 + b^2 + 3*a*b = 2015 → x*y ≤ 403 ∧ ∃ c d : ℝ, c^2 + d^2 + 3*c*d = 2015 ∧ c*d = 403 := by
  sorry

end max_xy_value_l1711_171143


namespace equal_elevation_angles_l1711_171131

/-- Given two flagpoles of heights h and k, separated by 2a units on a horizontal plane,
    this theorem characterizes the set of points where the angles of elevation to the tops
    of the poles are equal. -/
theorem equal_elevation_angles
  (h k a : ℝ) (h_pos : h > 0) (k_pos : k > 0) (a_pos : a > 0) :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 
    h / Real.sqrt ((x + a)^2 + y^2) = k / Real.sqrt ((x - a)^2 + y^2)
  (h = k → ∀ y, P (0, y)) ∧ 
  (h ≠ k → ∃ c r, ∀ x y, P (x, y) ↔ (x - c)^2 + y^2 = r^2) :=
by sorry

end equal_elevation_angles_l1711_171131


namespace father_son_speed_ratio_l1711_171171

/-- 
Given a hallway of length 16 meters where a father and son start walking from opposite ends 
at the same time and meet at a point 12 meters from the father's end, 
the ratio of the father's walking speed to the son's walking speed is 3:1.
-/
theorem father_son_speed_ratio 
  (hallway_length : ℝ) 
  (meeting_point : ℝ) 
  (father_speed : ℝ) 
  (son_speed : ℝ) 
  (h1 : hallway_length = 16)
  (h2 : meeting_point = 12)
  (h3 : father_speed > 0)
  (h4 : son_speed > 0)
  (h5 : meeting_point / father_speed = (hallway_length - meeting_point) / son_speed) :
  father_speed / son_speed = 3 := by
  sorry

end father_son_speed_ratio_l1711_171171


namespace right_triangles_on_circle_l1711_171152

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (t : ℕ), t = 1000 ∧ t = (n / 2) * (n - 2)) :=
by
  sorry

end right_triangles_on_circle_l1711_171152


namespace iterated_forward_difference_of_exponential_l1711_171155

def f (n : ℕ) : ℕ := 3^n

def forwardDifference (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

def iteratedForwardDifference (g : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => g
  | k + 1 => forwardDifference (iteratedForwardDifference g k)

theorem iterated_forward_difference_of_exponential (k : ℕ) (h : k ≥ 1) :
  ∀ n, iteratedForwardDifference f k n = 2^k * 3^n := by
  sorry

end iterated_forward_difference_of_exponential_l1711_171155


namespace litter_patrol_aluminum_cans_l1711_171101

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := 18

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := total_litter - glass_bottles

theorem litter_patrol_aluminum_cans : aluminum_cans = 8 := by
  sorry

end litter_patrol_aluminum_cans_l1711_171101


namespace game_a_vs_game_b_l1711_171176

def p_heads : ℚ := 2/3
def p_tails : ℚ := 1/3

def p_win_game_a : ℚ := p_heads^3 + p_tails^3

def p_same_pair : ℚ := p_heads^2 + p_tails^2
def p_win_game_b : ℚ := p_same_pair^2

theorem game_a_vs_game_b : p_win_game_a - p_win_game_b = 2/81 := by
  sorry

end game_a_vs_game_b_l1711_171176


namespace triangle_problem_l1711_171105

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2*b - c) * Real.cos A - a * Real.cos C = 0 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end triangle_problem_l1711_171105


namespace mans_speed_against_current_l1711_171190

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem: Given the specified conditions, the man's speed against the current is 9.4 km/hr -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.8 = 9.4 := by
  sorry

#eval speedAgainstCurrent 15 2.8

end mans_speed_against_current_l1711_171190


namespace exponent_division_l1711_171158

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^3 = a^3 := by
  sorry

end exponent_division_l1711_171158


namespace quadratic_non_real_roots_l1711_171168

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end quadratic_non_real_roots_l1711_171168


namespace max_value_x_plus_inverse_l1711_171139

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end max_value_x_plus_inverse_l1711_171139


namespace fraction_problem_l1711_171160

theorem fraction_problem :
  ∃ x : ℚ, x * 180 = 18 ∧ x < 0.15 → x = 1/10 := by
  sorry

end fraction_problem_l1711_171160


namespace manuscript_revision_cost_l1711_171123

/-- The cost per page for manuscript revision --/
def revision_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (initial_cost_per_page : ℚ) (total_cost : ℚ) : ℚ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_cost_per_page
  let revision_pages := pages_revised_once + 2 * pages_revised_twice
  (total_cost - initial_typing_cost) / revision_pages

theorem manuscript_revision_cost :
  revision_cost 100 20 30 10 1400 = 5 := by
  sorry

end manuscript_revision_cost_l1711_171123


namespace inscribed_square_side_length_squared_l1711_171114

/-- A square inscribed in an ellipse with specific properties -/
structure InscribedSquare where
  /-- The ellipse equation: x^2 + 3y^2 = 3 -/
  ellipse : ∀ (x y : ℝ), x^2 + 3 * y^2 = 3 → True
  /-- One vertex of the square is at (0, 1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the square lies along the y-axis -/
  diagonal_on_y_axis : ∃ (v1 v2 : ℝ × ℝ), v1.1 = 0 ∧ v2.1 = 0 ∧ v1 ≠ v2

/-- The theorem stating the square of the side length of the inscribed square -/
theorem inscribed_square_side_length_squared (s : InscribedSquare) :
  ∃ (side_length : ℝ), side_length^2 = 5/3 - 2 * Real.sqrt (2/3) :=
sorry

end inscribed_square_side_length_squared_l1711_171114


namespace polynomial_factorization_l1711_171120

theorem polynomial_factorization :
  ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + x^3 + x^2 + x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end polynomial_factorization_l1711_171120


namespace intersection_A_B_l1711_171142

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - 2^x)}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end intersection_A_B_l1711_171142


namespace bills_piggy_bank_l1711_171151

theorem bills_piggy_bank (x : ℕ) : 
  (∀ week : ℕ, week ≥ 1 ∧ week ≤ 8 → x + 2 * week = 3 * x) →
  x + 2 * 8 = 24 :=
by sorry

end bills_piggy_bank_l1711_171151


namespace complex_magnitude_equals_five_l1711_171162

theorem complex_magnitude_equals_five (t : ℝ) :
  Complex.abs (1 + 2 * t * Complex.I) = 5 ↔ t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end complex_magnitude_equals_five_l1711_171162


namespace condition_relationship_l1711_171178

theorem condition_relationship : ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ x + y > 3) :=
by
  sorry

end condition_relationship_l1711_171178


namespace product_closest_to_1200_l1711_171150

def product : ℝ := 0.000315 * 3928500

def options : List ℝ := [1100, 1200, 1300, 1400]

theorem product_closest_to_1200 : 
  1200 ∈ options ∧ ∀ x ∈ options, |product - 1200| ≤ |product - x| :=
by sorry

end product_closest_to_1200_l1711_171150


namespace max_value_squared_sum_l1711_171111

/-- Given a point P(x,y) satisfying certain conditions, 
    the maximum value of x^2 + y^2 is 18. -/
theorem max_value_squared_sum (x y : ℝ) 
  (h1 : x ≥ 1) 
  (h2 : y ≥ x) 
  (h3 : x - 2*y + 3 ≥ 0) : 
  ∃ (max : ℝ), max = 18 ∧ x^2 + y^2 ≤ max :=
by sorry

end max_value_squared_sum_l1711_171111


namespace solution_sets_union_l1711_171106

-- Define the solution sets A and B
def A (p q : ℝ) : Set ℝ := {x | x^2 - (p-1)*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 + (q-1)*x + p = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (p q : ℝ), A p q ∩ B p q = {-2}) →
  (∃ (p q : ℝ), A p q ∪ B p q = {-2, -1, 1}) :=
by sorry

end solution_sets_union_l1711_171106


namespace percentage_difference_l1711_171169

theorem percentage_difference : 
  (60 * 80 / 100) - (25 * 4 / 5) = 28 := by
  sorry

end percentage_difference_l1711_171169


namespace complex_absolute_value_l1711_171195

theorem complex_absolute_value (z : ℂ) : z = 7 + 3*I → Complex.abs (z^2 + 8*z + 65) = Real.sqrt 30277 := by
  sorry

end complex_absolute_value_l1711_171195


namespace boat_breadth_l1711_171110

/-- Given a boat with the following properties:
  - length of 7 meters
  - sinks by 1 cm when a man gets on it
  - the man's mass is 210 kg
  - the density of water is 1000 kg/m³
  - the acceleration due to gravity is 9.81 m/s²
  Prove that the breadth of the boat is 3 meters. -/
theorem boat_breadth (length : ℝ) (sink_depth : ℝ) (man_mass : ℝ) (water_density : ℝ) (gravity : ℝ) :
  length = 7 →
  sink_depth = 0.01 →
  man_mass = 210 →
  water_density = 1000 →
  gravity = 9.81 →
  ∃ (breadth : ℝ), breadth = 3 ∧ man_mass = (length * breadth * sink_depth) * water_density :=
by sorry

end boat_breadth_l1711_171110


namespace analysis_method_sufficient_conditions_l1711_171199

/-- The analysis method in mathematical proofs -/
structure AnalysisMethod where
  /-- The method starts from the conclusion to be proved -/
  starts_from_conclusion : Bool
  /-- The method progressively searches for conditions -/
  progressive_search : Bool
  /-- The type of conditions the method searches for -/
  condition_type : Type

/-- Definition of sufficient conditions -/
def SufficientCondition : Type := Unit

/-- Theorem: The analysis method searches for sufficient conditions -/
theorem analysis_method_sufficient_conditions (am : AnalysisMethod) :
  am.starts_from_conclusion ∧ am.progressive_search →
  am.condition_type = SufficientCondition := by
  sorry

end analysis_method_sufficient_conditions_l1711_171199


namespace larger_number_of_product_18_sum_15_l1711_171127

theorem larger_number_of_product_18_sum_15 (x y : ℝ) : 
  x * y = 18 → x + y = 15 → max x y = 12 := by
sorry

end larger_number_of_product_18_sum_15_l1711_171127


namespace average_difference_approx_l1711_171138

def total_students : ℕ := 180
def total_teachers : ℕ := 6
def class_enrollments : List ℕ := [80, 40, 40, 10, 5, 5]

def teacher_average (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.sum : ℚ) / teachers

def student_average (students : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.map (λ n => n * n)).sum / students

theorem average_difference_approx (ε : ℚ) (hε : ε > 0) :
  ∃ δ : ℚ, δ > 0 ∧ 
    |teacher_average total_students total_teachers class_enrollments - 
     student_average total_students class_enrollments + 24.17| < δ ∧ δ < ε :=
by sorry

end average_difference_approx_l1711_171138


namespace least_multiple_of_smallest_primes_gt_5_l1711_171156

def smallest_primes_gt_5 : List Nat := [7, 11, 13]

theorem least_multiple_of_smallest_primes_gt_5 :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ smallest_primes_gt_5, p ∣ n) → n ≥ 1001) ∧
  (∀ p ∈ smallest_primes_gt_5, p ∣ 1001) :=
sorry

end least_multiple_of_smallest_primes_gt_5_l1711_171156


namespace simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l1711_171136

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := 2*x^2 - 3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_3B_specific (x y : ℝ) (h1 : x + y = 6/7) (h2 : x*y = -1) :
  2 * A x y - 3 * B x y = 17 :=
sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_3B_independent :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * A x y - 3 * B x y = 49/11 :=
sorry

end simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l1711_171136


namespace circle_point_range_l1711_171154

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (0, a)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := M
  circle_C x y ∧ (x^2 + (y - a)^2 = 2 * (x^2 + y^2))

-- Theorem statement
theorem circle_point_range (a : ℝ) :
  a > 0 →
  (∃ M : ℝ × ℝ, condition_M M a) →
  Real.sqrt 3 ≤ a ∧ a ≤ 4 + Real.sqrt 19 :=
by sorry

end circle_point_range_l1711_171154


namespace turban_price_l1711_171167

theorem turban_price (annual_salary : ℝ) (turban_price : ℝ) (work_fraction : ℝ) (partial_payment : ℝ) :
  annual_salary = 90 ∧ 
  work_fraction = 3/4 ∧ 
  work_fraction * (annual_salary + turban_price) = partial_payment + turban_price ∧
  partial_payment = 45 →
  turban_price = 90 := by sorry

end turban_price_l1711_171167


namespace total_bees_after_changes_l1711_171144

/-- Represents a bee hive with initial bees and changes in population --/
structure BeeHive where
  initial : ℕ
  fly_in : ℕ
  fly_out : ℕ

/-- Calculates the final number of bees in a hive after changes --/
def final_bees (hive : BeeHive) : ℕ :=
  hive.initial + hive.fly_in - hive.fly_out

/-- Represents the bee colony --/
def BeeColony : List BeeHive := [
  { initial := 45, fly_in := 12, fly_out := 8 },
  { initial := 60, fly_in := 15, fly_out := 20 },
  { initial := 75, fly_in := 10, fly_out := 5 }
]

/-- Theorem stating the total number of bees after changes --/
theorem total_bees_after_changes :
  (BeeColony.map final_bees).sum = 184 := by
  sorry

end total_bees_after_changes_l1711_171144


namespace complex_number_properties_l1711_171198

theorem complex_number_properties (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  (Complex.abs z = Real.sqrt 2) ∧ (z^2 - 2*z + 2 = 0) := by
  sorry

end complex_number_properties_l1711_171198


namespace original_phone_number_proof_l1711_171109

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d1 := n / 100000
  let rest := n % 100000
  d1 * 1000000 + 800000 + rest

def second_upgrade (n : ℕ) : ℕ :=
  2000000 + n

theorem original_phone_number_proof :
  ∃! n : ℕ, is_valid_phone_number n ∧
    second_upgrade (first_upgrade n) = 81 * n ∧
    n = 282500 :=
sorry

end original_phone_number_proof_l1711_171109


namespace existence_of_m_l1711_171107

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by
  sorry

end existence_of_m_l1711_171107


namespace quadratic_equation_completion_square_l1711_171141

theorem quadratic_equation_completion_square (x : ℝ) :
  (16 * x^2 - 32 * x - 512 = 0) →
  ∃ (k m : ℝ), ((x + k)^2 = m) ∧ (m = 65) :=
by sorry

end quadratic_equation_completion_square_l1711_171141


namespace problem_solution_l1711_171129

theorem problem_solution (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end problem_solution_l1711_171129


namespace present_age_of_B_l1711_171165

-- Define the ages of A and B as natural numbers
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 7

-- Theorem statement
theorem present_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 := by
  sorry

end present_age_of_B_l1711_171165


namespace line_inclination_through_origin_and_negative_one_l1711_171157

/-- The angle of inclination of a line passing through (0, 0) and (-1, -1) is 45°. -/
theorem line_inclination_through_origin_and_negative_one : ∃ (α : ℝ), 
  (∀ (x y : ℝ), y = x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) →
  α * (π / 180) = π / 4 := by
  sorry

end line_inclination_through_origin_and_negative_one_l1711_171157


namespace train_problem_solution_l1711_171185

/-- Represents the train problem scenario -/
structure TrainProblem where
  totalDistance : ℝ
  trainBTime : ℝ
  meetingPointA : ℝ
  trainATime : ℝ

/-- The solution to the train problem -/
def solveTrain (p : TrainProblem) : Prop :=
  p.totalDistance = 125 ∧
  p.trainBTime = 8 ∧
  p.meetingPointA = 50 ∧
  p.trainATime = 12

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem train_problem_solution :
  ∀ (p : TrainProblem),
    p.totalDistance = 125 ∧
    p.trainBTime = 8 ∧
    p.meetingPointA = 50 →
    solveTrain p :=
by
  sorry

#check train_problem_solution

end train_problem_solution_l1711_171185


namespace sin_240_degrees_l1711_171103

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l1711_171103


namespace sum_greater_than_six_random_event_l1711_171126

def numbers : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def sumGreaterThanSix (a b c : ℕ) : Prop := a + b + c > 6

theorem sum_greater_than_six_random_event :
  ∃ (a b c : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ sumGreaterThanSix a b c ∧
  ∃ (x y z : ℕ), x ∈ numbers ∧ y ∈ numbers ∧ z ∈ numbers ∧ ¬sumGreaterThanSix x y z :=
sorry

end sum_greater_than_six_random_event_l1711_171126


namespace students_present_l1711_171113

/-- Given a class of 100 students with 14% absent, prove that the number of students present is 86. -/
theorem students_present (total_students : ℕ) (absent_percentage : ℚ) : 
  total_students = 100 → 
  absent_percentage = 14/100 → 
  (total_students : ℚ) * (1 - absent_percentage) = 86 :=
by sorry

end students_present_l1711_171113


namespace cod_fish_sold_l1711_171100

theorem cod_fish_sold (total : ℕ) (haddock_percent : ℚ) (halibut_percent : ℚ) 
  (h1 : total = 220)
  (h2 : haddock_percent = 40 / 100)
  (h3 : halibut_percent = 40 / 100) :
  (total : ℚ) * (1 - haddock_percent - halibut_percent) = 44 := by sorry

end cod_fish_sold_l1711_171100


namespace quadratic_root_property_l1711_171132

theorem quadratic_root_property (n : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + n = 0) → 
  (x₂^2 - 3*x₂ + n = 0) → 
  (x₁ + x₂ - 2 = x₁ * x₂) → 
  n = 1 := by
sorry

end quadratic_root_property_l1711_171132


namespace chi_square_test_win_probability_not_C_given_not_win_l1711_171147

-- Define the data from the problem
def flavor1_C : ℕ := 20
def flavor1_nonC : ℕ := 75
def flavor2_C : ℕ := 10
def flavor2_nonC : ℕ := 45
def total_samples : ℕ := 150

-- Define the chi-square test statistic function
def chi_square (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the proportions of card types
def prob_A : ℚ := 2 / 5
def prob_B : ℚ := 2 / 5
def prob_C : ℚ := 1 / 5

-- Theorem statements
theorem chi_square_test :
  chi_square flavor1_C flavor1_nonC flavor2_C flavor2_nonC total_samples < 6635 / 1000 :=
sorry

theorem win_probability :
  (3 * prob_A * prob_B * prob_C : ℚ) = 24 / 125 :=
sorry

theorem not_C_given_not_win :
  ((1 - prob_C)^3 : ℚ) / (1 - 3 * prob_A * prob_B * prob_C) = 64 / 101 :=
sorry

end chi_square_test_win_probability_not_C_given_not_win_l1711_171147


namespace football_team_progress_l1711_171177

/-- The progress of a football team after a series of gains and losses -/
theorem football_team_progress 
  (L1 G1 L2 G2 G3 : ℤ) 
  (hL1 : L1 = 17)
  (hG1 : G1 = 35)
  (hL2 : L2 = 22)
  (hG2 : G2 = 8) :
  (G1 + G2 - (L1 + L2)) + G3 = 4 + G3 :=
by sorry

end football_team_progress_l1711_171177


namespace three_digit_divisible_by_square_sum_digits_l1711_171153

def isValidNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (let digits := [n / 100, (n / 10) % 10, n % 10]
   digits.toFinset.card = 3 ∧
   n % ((digits.sum)^2) = 0)

theorem three_digit_divisible_by_square_sum_digits :
  {n : ℕ | isValidNumber n} =
  {162, 243, 324, 405, 512, 648, 729, 810, 972} :=
by sorry

end three_digit_divisible_by_square_sum_digits_l1711_171153


namespace determinant_evaluation_l1711_171104

theorem determinant_evaluation (x : ℝ) : 
  Matrix.det !![x + 2, x - 1, x; x - 1, x + 2, x; x, x, x + 3] = 14 * x + 9 := by
  sorry

end determinant_evaluation_l1711_171104


namespace smallest_x_is_correct_l1711_171191

/-- The smallest positive integer x such that 2520x is a perfect cube -/
def smallest_x : ℕ := 3675

/-- 2520 as a natural number -/
def given_number : ℕ := 2520

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ M : ℕ, given_number * y = M^3) ∧
  ∃ M : ℕ, given_number * smallest_x = M^3 :=
sorry

end smallest_x_is_correct_l1711_171191


namespace polynomial_evaluation_l1711_171102

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^3 - x^2 + x - 1 = 51 := by
  sorry

end polynomial_evaluation_l1711_171102


namespace inverse_of_10_mod_1729_l1711_171117

theorem inverse_of_10_mod_1729 : ∃ x : ℕ, x ≤ 1728 ∧ (10 * x) % 1729 = 1 :=
by
  use 1537
  sorry

end inverse_of_10_mod_1729_l1711_171117


namespace count_factorizable_pairs_eq_325_l1711_171170

/-- Counts the number of ordered pairs (a,b) satisfying the factorization condition -/
def count_factorizable_pairs : ℕ :=
  (Finset.range 50).sum (λ a => (a + 1) / 2)

/-- The main theorem stating that the count of factorizable pairs is 325 -/
theorem count_factorizable_pairs_eq_325 : count_factorizable_pairs = 325 := by
  sorry


end count_factorizable_pairs_eq_325_l1711_171170


namespace find_T_l1711_171183

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 120 ∧ T = 96 := by
  sorry

end find_T_l1711_171183


namespace equation_solution_l1711_171125

theorem equation_solution : 
  ∃ x : ℚ, (1 : ℚ) / 3 + 1 / x = 7 / 12 ∧ x = 4 := by
  sorry

end equation_solution_l1711_171125


namespace max_odd_numbers_in_even_product_l1711_171140

theorem max_odd_numbers_in_even_product (numbers : Finset ℕ) :
  numbers.card = 7 →
  (numbers.prod (fun x ↦ x)) % 2 = 0 →
  (numbers.filter (fun x ↦ x % 2 = 1)).card ≤ 6 :=
by sorry

end max_odd_numbers_in_even_product_l1711_171140


namespace quadratic_roots_relationship_l1711_171121

theorem quadratic_roots_relationship (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, m₁ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, m₂ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₃ ∨ x = x₄) →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end quadratic_roots_relationship_l1711_171121


namespace animals_remaining_l1711_171164

theorem animals_remaining (cows dogs : ℕ) : 
  cows = 2 * dogs →
  cows = 184 →
  (184 - 184 / 4) + (dogs - 3 * dogs / 4) = 161 := by
sorry

end animals_remaining_l1711_171164


namespace trig_inequality_l1711_171149

theorem trig_inequality : ∃ (a b c : ℝ), 
  a = Real.cos 1 ∧ 
  b = Real.sin 1 ∧ 
  c = Real.tan 1 ∧ 
  a < b ∧ b < c :=
by sorry

end trig_inequality_l1711_171149


namespace sin_plus_cos_range_l1711_171119

theorem sin_plus_cos_range (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b^2 = a * c →            -- Given condition
  ∃ (x : Real), 1 < x ∧ x ≤ Real.sqrt 2 ∧ x = Real.sin B + Real.cos B :=
by sorry

end sin_plus_cos_range_l1711_171119


namespace percentage_of_students_taking_music_l1711_171148

/-- Calculates the percentage of students taking music in a school -/
theorem percentage_of_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (drama_students : ℕ)
  (h1 : total_students = 2000)
  (h2 : dance_students = 450)
  (h3 : art_students = 680)
  (h4 : drama_students = 370) :
  (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 := by
  sorry

end percentage_of_students_taking_music_l1711_171148


namespace right_triangle_hypotenuse_l1711_171182

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 140 → 
  b = 210 → 
  c^2 = a^2 + b^2 → 
  c = 70 * Real.sqrt 13 := by
sorry

end right_triangle_hypotenuse_l1711_171182


namespace sum_of_solutions_g_l1711_171108

def f (x : ℝ) : ℝ := -x^2 + 10*x - 20

def g : ℝ → ℝ := (f^[2010])

theorem sum_of_solutions_g (h : ∃ (S : Finset ℝ), S.card = 2^2010 ∧ ∀ x ∈ S, g x = 2) :
  ∃ (S : Finset ℝ), S.card = 2^2010 ∧ (∀ x ∈ S, g x = 2) ∧ (S.sum id = 5 * 2^2010) := by
  sorry

end sum_of_solutions_g_l1711_171108


namespace projectile_max_height_l1711_171184

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 45

/-- Theorem: The maximum height reached by the projectile is 153 feet -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 153 :=
sorry

end projectile_max_height_l1711_171184


namespace truck_rental_miles_driven_l1711_171173

theorem truck_rental_miles_driven 
  (rental_fee : ℝ) 
  (charge_per_mile : ℝ) 
  (total_paid : ℝ) 
  (h1 : rental_fee = 20.99)
  (h2 : charge_per_mile = 0.25)
  (h3 : total_paid = 95.74) : 
  ⌊(total_paid - rental_fee) / charge_per_mile⌋ = 299 := by
sorry


end truck_rental_miles_driven_l1711_171173
