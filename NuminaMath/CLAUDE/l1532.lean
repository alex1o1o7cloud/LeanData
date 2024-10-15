import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_of_distances_l1532_153270

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is isosceles with AB = AC = b and BC = a -/
def isIsoscelesTriangle (t : Triangle) (a b : ℝ) : Prop :=
  let d (p q : Point) := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)
  d t.A t.B = b ∧ d t.A t.C = b ∧ d t.B t.C = a

/-- Checks if a point is inside a triangle -/
def isInside (P : Point) (t : Triangle) : Prop := sorry

/-- Calculates the sum of distances from a point to each side of a triangle -/
def sumOfDistances (P : Point) (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem max_sum_of_distances (t : Triangle) (a b : ℝ) (P : Point) :
  a ≤ b →
  isIsoscelesTriangle t a b →
  isInside P t →
  sumOfDistances P t < 2 * b + a := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_l1532_153270


namespace NUMINAMATH_CALUDE_set_intersections_and_union_l1532_153207

def A : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 0}
def B : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 = 0}
def C : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 3}

theorem set_intersections_and_union :
  (A ∩ B = {(0, 0)}) ∧
  (A ∩ C = ∅) ∧
  ((A ∩ B) ∪ (B ∩ C) = {(0, 0), (3/5, -9/5)}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersections_and_union_l1532_153207


namespace NUMINAMATH_CALUDE_g_of_2_l1532_153286

def g (x : ℝ) : ℝ := x^2 - 3*x + 1

theorem g_of_2 : g 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_l1532_153286


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l1532_153240

theorem quadratic_completion_of_square (b : ℝ) (p : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/6 = (x+p)^2 + 1/18) → 
  b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l1532_153240


namespace NUMINAMATH_CALUDE_largest_cubic_root_bound_l1532_153217

theorem largest_cubic_root_bound (a₂ a₁ a₀ : ℝ) 
  (h₂ : |a₂| ≤ 2) (h₁ : |a₁| ≤ 2) (h₀ : |a₀| ≤ 2) :
  ∃ r : ℝ, (r > 0) ∧ (5/2 ≤ r) ∧ (r < 3) ∧
  (∀ x : ℝ, x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r) :=
sorry

end NUMINAMATH_CALUDE_largest_cubic_root_bound_l1532_153217


namespace NUMINAMATH_CALUDE_cube_ratios_l1532_153282

/-- Given two cubes with edge lengths 4 inches and 24 inches respectively,
    prove that the ratio of their volumes is 1/216 and
    the ratio of their surface areas is 1/36 -/
theorem cube_ratios (edge1 edge2 : ℝ) (h1 : edge1 = 4) (h2 : edge2 = 24) :
  (edge1^3 / edge2^3 = 1 / 216) ∧ ((6 * edge1^2) / (6 * edge2^2) = 1 / 36) := by
  sorry

end NUMINAMATH_CALUDE_cube_ratios_l1532_153282


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l1532_153269

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℕ) (time : ℕ) : ℚ :=
  (amount - principal : ℚ) * 100 / (principal * time)

/-- Theorem stating that the simple interest rate is 12% given the problem conditions -/
theorem simple_interest_rate_problem :
  simple_interest_rate 750 1200 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l1532_153269


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1532_153231

theorem system_solution_ratio (x y z a b : ℝ) : 
  (4 * x - 2 * y + z = a) →
  (6 * y - 12 * x - 3 * z = b) →
  (b ≠ 0) →
  (a / b = -1 / 3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1532_153231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1532_153283

-- Define an arithmetic sequence of 8 terms
def arithmetic_sequence (a d : ℤ) : Fin 8 → ℤ :=
  fun i => a + i.val * d

theorem arithmetic_sequence_proof :
  ∀ a d : ℤ,
  (arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 41) →
  (arithmetic_sequence a d 0 * arithmetic_sequence a d 7 = 114) →
  ((∀ i : Fin 8, arithmetic_sequence a d i = arithmetic_sequence 3 5 i) ∨
   (∀ i : Fin 8, arithmetic_sequence a d i = arithmetic_sequence 38 (-5) i)) :=
by
  sorry

#check arithmetic_sequence_proof

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1532_153283


namespace NUMINAMATH_CALUDE_cylinder_volume_scaling_l1532_153220

/-- Proves that doubling both the radius and height of a cylindrical container increases its volume by a factor of 8 -/
theorem cylinder_volume_scaling (r h V : ℝ) (h1 : V = Real.pi * r^2 * h) :
  Real.pi * (2*r)^2 * (2*h) = 8 * V := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_scaling_l1532_153220


namespace NUMINAMATH_CALUDE_perfect_squares_l1532_153253

theorem perfect_squares (m n a : ℝ) (h : a = m * n) : 
  ((m - n) / 2)^2 + a = ((m + n) / 2)^2 ∧ 
  ((m + n) / 2)^2 - a = ((m - n) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_l1532_153253


namespace NUMINAMATH_CALUDE_sum_of_squares_l1532_153226

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1532_153226


namespace NUMINAMATH_CALUDE_average_of_xyz_l1532_153203

theorem average_of_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) :
  (x + y + z) / 3 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1532_153203


namespace NUMINAMATH_CALUDE_simplified_expression_l1532_153229

theorem simplified_expression (a : ℤ) (h : a = 2022) : 
  (a + 1 : ℚ) / a - 2 * (a : ℚ) / (a + 1) = (-a^2 + 2*a + 1 : ℚ) / (a * (a + 1)) ∧
  -a^2 + 2*a + 1 = -2022^2 + 4045 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_l1532_153229


namespace NUMINAMATH_CALUDE_sum_of_integers_l1532_153291

theorem sum_of_integers (a b c d : ℕ+) 
  (eq1 : a * b + c * d = 38)
  (eq2 : a * c + b * d = 34)
  (eq3 : a * d + b * c = 43) :
  a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1532_153291


namespace NUMINAMATH_CALUDE_center_cell_value_l1532_153267

/-- Represents a 5x5 square with arithmetic progressions in rows and columns -/
def ArithmeticSquare : Type := Fin 5 → Fin 5 → ℤ

/-- The property that a row forms an arithmetic progression -/
def row_is_arithmetic_progression (s : ArithmeticSquare) (i : Fin 5) : Prop :=
  ∀ j k : Fin 5, s i k - s i j = (k - j) * (s i 1 - s i 0)

/-- The property that a column forms an arithmetic progression -/
def col_is_arithmetic_progression (s : ArithmeticSquare) (j : Fin 5) : Prop :=
  ∀ i k : Fin 5, s k j - s i j = (k - i) * (s 1 j - s 0 j)

/-- The main theorem -/
theorem center_cell_value (s : ArithmeticSquare) 
  (corner_values : s 0 0 = 1 ∧ s 0 4 = 25 ∧ s 4 0 = 81 ∧ s 4 4 = 17)
  (rows_arithmetic : ∀ i : Fin 5, row_is_arithmetic_progression s i)
  (cols_arithmetic : ∀ j : Fin 5, col_is_arithmetic_progression s j) :
  s 2 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_center_cell_value_l1532_153267


namespace NUMINAMATH_CALUDE_female_attendees_on_time_l1532_153244

/-- Proves that the fraction of female attendees who arrived on time is 0.9 -/
theorem female_attendees_on_time 
  (total_attendees : ℝ) 
  (male_ratio : ℝ) 
  (male_on_time_ratio : ℝ) 
  (not_on_time_ratio : ℝ) 
  (h1 : male_ratio = 3/5) 
  (h2 : male_on_time_ratio = 7/8) 
  (h3 : not_on_time_ratio = 0.115) : 
  let female_ratio := 1 - male_ratio
  let female_on_time_ratio := 
    (1 - not_on_time_ratio - male_ratio * male_on_time_ratio) / female_ratio
  female_on_time_ratio = 0.9 := by
sorry

end NUMINAMATH_CALUDE_female_attendees_on_time_l1532_153244


namespace NUMINAMATH_CALUDE_sin_2x_plus_one_l1532_153295

theorem sin_2x_plus_one (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : 
  Real.sin (2 * x) + 1 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_plus_one_l1532_153295


namespace NUMINAMATH_CALUDE_wheel_of_fortune_probability_l1532_153273

def wheel_divisions : Finset ℕ := {0, 1000, 300, 5000, 400, 600, 800}

def num_spins : ℕ := 4

def total_outcomes : ℕ := 7^num_spins

def valid_combinations : List (List ℕ) := [
  [1000, 600, 600, 600],
  [1000, 1000, 400, 400],
  [800, 800, 600, 600],
  [800, 800, 800, 400]
]

def count_permutations (combination : List ℕ) : ℕ :=
  Nat.factorial num_spins

def total_valid_outcomes : ℕ :=
  (valid_combinations.map count_permutations).sum

theorem wheel_of_fortune_probability :
  (total_valid_outcomes : ℚ) / total_outcomes = 96 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_wheel_of_fortune_probability_l1532_153273


namespace NUMINAMATH_CALUDE_picks_theorem_lattice_points_in_triangle_l1532_153239

/-- A point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  O : Point

/-- Counts the number of lattice points on a line segment -/
def countLatticePointsOnSegment (p1 p2 : Point) : ℕ :=
  sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℚ :=
  sorry

/-- Counts the number of lattice points inside a triangle -/
def countLatticePointsInside (t : Triangle) : ℕ :=
  sorry

/-- Pick's theorem: S = N + L/2 - 1, where S is the area, N is the number of interior lattice points, and L is the number of boundary lattice points -/
theorem picks_theorem (t : Triangle) (S : ℚ) (N L : ℕ) :
  S = N + L / 2 - 1 →
  S = triangleArea t →
  L = countLatticePointsOnSegment t.A t.B + countLatticePointsOnSegment t.B t.O + countLatticePointsOnSegment t.O t.A →
  N = countLatticePointsInside t :=
  sorry

theorem lattice_points_in_triangle :
  let t : Triangle := { A := { x := 0, y := 30 }, B := { x := 20, y := 10 }, O := { x := 0, y := 0 } }
  countLatticePointsInside t = 271 :=
by sorry

end NUMINAMATH_CALUDE_picks_theorem_lattice_points_in_triangle_l1532_153239


namespace NUMINAMATH_CALUDE_laura_circle_arrangements_l1532_153216

def numbers : List ℕ := [2, 3, 5, 6, 11]

def is_valid_arrangement (num1 num2 num3 denom : ℕ) : Prop :=
  num1 ∈ numbers ∧ num2 ∈ numbers ∧ num3 ∈ numbers ∧ denom ∈ numbers ∧
  num1 ≠ num2 ∧ num1 ≠ num3 ∧ num1 ≠ denom ∧
  num2 ≠ num3 ∧ num2 ≠ denom ∧
  num3 ≠ denom ∧
  (num1 + num2 + num3) % denom = 0

theorem laura_circle_arrangements :
  ∃! (arrangements : List (ℕ × ℕ × ℕ × ℕ)),
    (∀ arr ∈ arrangements, is_valid_arrangement arr.1 arr.2.1 arr.2.2.1 arr.2.2.2) ∧
    arrangements.length = 4 :=
by sorry

end NUMINAMATH_CALUDE_laura_circle_arrangements_l1532_153216


namespace NUMINAMATH_CALUDE_farm_ploughing_rate_l1532_153200

/-- Given a farm field and ploughing conditions, calculate the required daily ploughing rate to finish on time -/
theorem farm_ploughing_rate (total_area planned_rate actual_rate extra_days left_area : ℕ) : 
  total_area = 720 ∧ 
  actual_rate = 85 ∧ 
  extra_days = 2 ∧ 
  left_area = 40 →
  (total_area : ℚ) / ((total_area - left_area) / actual_rate - extra_days) = 120 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_rate_l1532_153200


namespace NUMINAMATH_CALUDE_rohan_house_rent_percentage_l1532_153237

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  savings : ℝ
  house_rent_percent : ℝ

/-- Theorem stating that Rohan spends 20% of his salary on house rent -/
theorem rohan_house_rent_percentage 
  (rf : RohanFinances)
  (h_salary : rf.salary = 7500)
  (h_food : rf.food_percent = 40)
  (h_entertainment : rf.entertainment_percent = 10)
  (h_conveyance : rf.conveyance_percent = 10)
  (h_savings : rf.savings = 1500)
  (h_total : rf.food_percent + rf.entertainment_percent + rf.conveyance_percent + 
             rf.house_rent_percent + (rf.savings / rf.salary * 100) = 100) :
  rf.house_rent_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_rohan_house_rent_percentage_l1532_153237


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l1532_153297

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Axiom: m and n are different lines
axiom different_lines : m ≠ n

-- Axiom: α and β are different planes
axiom different_planes : α ≠ β

-- Theorem 1 (Proposition ②)
theorem prop_2 : 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  perpendicular m n → 
  perpendicular_plane α β :=
sorry

-- Theorem 2 (Proposition ④)
theorem prop_4 : 
  perpendicular_line_plane m α → 
  parallel_line_plane n β → 
  parallel_plane α β → 
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l1532_153297


namespace NUMINAMATH_CALUDE_dave_clothes_tickets_l1532_153292

def dave_tickets : ℕ := 13
def toys_tickets : ℕ := 8
def clothes_toys_difference : ℕ := 10

theorem dave_clothes_tickets :
  dave_tickets ≥ toys_tickets + clothes_toys_difference →
  toys_tickets + clothes_toys_difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_clothes_tickets_l1532_153292


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l1532_153234

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_single_digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d ≤ 9

theorem seven_digit_divisible_by_11 (m n : ℕ) :
  is_single_digit m →
  is_single_digit n →
  is_divisible_by_11 (742 * 10000 + m * 1000 + 83 * 10 + n) →
  m + n = 1 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l1532_153234


namespace NUMINAMATH_CALUDE_even_function_sum_l1532_153212

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

theorem even_function_sum (a b : ℝ) : 
  (∀ x ∈ Set.Icc (a^2 - 2) a, f a b x = f a b (-x)) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l1532_153212


namespace NUMINAMATH_CALUDE_square_diagonal_double_area_l1532_153294

theorem square_diagonal_double_area (d₁ : ℝ) (d₂ : ℝ) : 
  d₁ = 4 * Real.sqrt 2 → 
  d₂ * d₂ = 2 * (d₁ * d₁ / 2) → 
  d₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_double_area_l1532_153294


namespace NUMINAMATH_CALUDE_power_of_64_l1532_153221

theorem power_of_64 : (64 : ℝ) ^ (5/6 : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l1532_153221


namespace NUMINAMATH_CALUDE_selling_price_is_80_percent_l1532_153247

/-- Represents the money distribution and orange selling scenario -/
structure OrangeSelling where
  cara_ratio : ℚ
  janet_ratio : ℚ
  jerry_ratio : ℚ
  total_money : ℚ
  loss : ℚ

/-- Calculates the selling price as a percentage of the buying price -/
def sellingPricePercentage (scenario : OrangeSelling) : ℚ :=
  let x := scenario.total_money / (scenario.cara_ratio + scenario.janet_ratio + scenario.jerry_ratio)
  let cara_money := scenario.cara_ratio * x
  let janet_money := scenario.janet_ratio * x
  let buying_price := cara_money + janet_money
  let selling_price := buying_price - scenario.loss
  (selling_price / buying_price) * 100

/-- Theorem stating that the selling price is 80% of the buying price -/
theorem selling_price_is_80_percent (scenario : OrangeSelling) 
  (h1 : scenario.cara_ratio = 4)
  (h2 : scenario.janet_ratio = 5)
  (h3 : scenario.jerry_ratio = 6)
  (h4 : scenario.total_money = 75)
  (h5 : scenario.loss = 9) :
  sellingPricePercentage scenario = 80 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_80_percent_l1532_153247


namespace NUMINAMATH_CALUDE_soy_sauce_bottles_l1532_153243

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for recipe 1 in cups -/
def Recipe1Cups : ℕ := 2

/-- Represents the amount of soy sauce needed for recipe 2 in cups -/
def Recipe2Cups : ℕ := 1

/-- Represents the amount of soy sauce needed for recipe 3 in cups -/
def Recipe3Cups : ℕ := 3

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := Recipe1Cups + Recipe2Cups + Recipe3Cups

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by
  sorry

end NUMINAMATH_CALUDE_soy_sauce_bottles_l1532_153243


namespace NUMINAMATH_CALUDE_perfect_power_relation_l1532_153218

theorem perfect_power_relation (x y : ℕ+) (k : ℕ+) :
  (x * y^433 = k^2016) → ∃ m : ℕ+, x^433 * y = m^2016 := by
  sorry

end NUMINAMATH_CALUDE_perfect_power_relation_l1532_153218


namespace NUMINAMATH_CALUDE_triangle_side_length_l1532_153204

theorem triangle_side_length (a : ℝ) : 
  (6 + 2 > a ∧ 6 + a > 2 ∧ a + 2 > 6) → a = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1532_153204


namespace NUMINAMATH_CALUDE_parabola_through_circle_center_l1532_153274

/-- Represents a circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Represents a parabola in the 2D plane --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ → Prop

/-- The given circle --/
def given_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x + 6*y + 9 = 0 }

/-- Theorem stating the properties of the parabola passing through the circle's center --/
theorem parabola_through_circle_center (p : Parabola) :
  p.vertex = (0, 0) →
  (p.axis_of_symmetry = λ x y => x = 0 ∨ y = 0) →
  (∃ x y, given_circle.equation x y ∧ p.equation x y) →
  (∀ x y, p.equation x y ↔ (y = -3*x^2 ∨ y^2 = 9*x)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_circle_center_l1532_153274


namespace NUMINAMATH_CALUDE_gcd_of_polynomials_l1532_153250

theorem gcd_of_polynomials (a : ℤ) (h : ∃ k : ℤ, a = 720 * k) :
  Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomials_l1532_153250


namespace NUMINAMATH_CALUDE_exceeded_goal_l1532_153284

/-- Represents the school band's car wash fundraiser --/
def car_wash_fundraiser (goal : ℕ) (basic_price deluxe_price premium_price cookie_price : ℕ)
  (basic_count deluxe_count premium_count cookie_count : ℕ) : Prop :=
  let total_earnings := basic_price * basic_count + deluxe_price * deluxe_count +
                        premium_price * premium_count + cookie_price * cookie_count
  total_earnings - goal = 32

/-- Theorem stating that the school band has exceeded their fundraising goal by $32 --/
theorem exceeded_goal : car_wash_fundraiser 150 5 8 12 2 10 6 2 30 := by
  sorry

end NUMINAMATH_CALUDE_exceeded_goal_l1532_153284


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1532_153290

theorem at_least_one_real_root (p q : ℕ) (h_distinct : p ≠ q) (h_positive_p : p > 0) (h_positive_q : q > 0) :
  (p^2 : ℝ) - 4*q ≥ 0 ∨ (q^2 : ℝ) - 4*p ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1532_153290


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l1532_153255

/-- Given that Shannon has 48 heart-shaped stones and wants to make 6 bracelets
    with an equal number of stones in each, prove that the number of
    heart-shaped stones per bracelet is 8. -/
theorem stones_per_bracelet (total_stones : ℕ) (num_bracelets : ℕ) 
  (h1 : total_stones = 48) (h2 : num_bracelets = 6) :
  total_stones / num_bracelets = 8 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l1532_153255


namespace NUMINAMATH_CALUDE_total_players_on_ground_l1532_153208

theorem total_players_on_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 12)
  (h2 : hockey_players = 17)
  (h3 : football_players = 11)
  (h4 : softball_players = 10) :
  cricket_players + hockey_players + football_players + softball_players = 50 := by
sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l1532_153208


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l1532_153227

/-- Theorem: One cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  ∀ (m cm : ℝ), m = 100 * cm → m^3 = 1000000 * cm^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l1532_153227


namespace NUMINAMATH_CALUDE_tangent_circle_center_l1532_153268

/-- A circle with radius 1 and center in the first quadrant, tangent to 4x - 3y = 0 and x-axis --/
structure TangentCircle where
  a : ℝ
  b : ℝ
  first_quadrant : 0 < a ∧ 0 < b
  tangent_to_line : |4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2) = 1
  tangent_to_x_axis : b = 1

/-- The center of the tangent circle is at (2, 1) --/
theorem tangent_circle_center (c : TangentCircle) : c.a = 2 ∧ c.b = 1 := by
  sorry

#check tangent_circle_center

end NUMINAMATH_CALUDE_tangent_circle_center_l1532_153268


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1532_153202

theorem purely_imaginary_condition (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x + 1)*I).re = 0 ∧ (((x^2 - 1) : ℂ) + (x + 1)*I).im ≠ 0 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1532_153202


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l1532_153288

-- Define the custom multiplication operation
def custom_mult (a b : ℝ) : ℝ := a^2 + a*b - b^2

-- State the theorem
theorem custom_mult_four_three : custom_mult 4 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l1532_153288


namespace NUMINAMATH_CALUDE_principal_is_10000_l1532_153236

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- Theorem stating that given the conditions, the principal is 10000 -/
theorem principal_is_10000 (loan : SimpleLoan) 
  (h_rate : loan.rate = 12)
  (h_time : loan.time = 3)
  (h_interest : loan.interest = 3600)
  (h_simple_interest : loan.interest = loan.principal * loan.rate * loan.time / 100) :
  loan.principal = 10000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_10000_l1532_153236


namespace NUMINAMATH_CALUDE_triangle_theorem_l1532_153289

/-- Given a triangle ABC with side lengths a, b, and c, and angles A, B, and C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given conditions -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a = 2 * Real.sqrt 3)
  (h2 : t.b + t.c = 4) :
  t.A = 2 * π / 3 ∧ 
  t.b * t.c = 4 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1532_153289


namespace NUMINAMATH_CALUDE_merchant_tea_cups_l1532_153205

theorem merchant_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11) 
  (h2 : b + c = 15) 
  (h3 : a + c = 14) : 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_tea_cups_l1532_153205


namespace NUMINAMATH_CALUDE_constant_term_in_system_of_equations_l1532_153299

theorem constant_term_in_system_of_equations :
  ∀ (x y k : ℝ),
  (7 * x + y = 19) →
  (x + 3 * y = k) →
  (2 * x + y = 5) →
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_constant_term_in_system_of_equations_l1532_153299


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l1532_153245

theorem trigonometric_calculations :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l1532_153245


namespace NUMINAMATH_CALUDE_ellipse_sum_l1532_153241

-- Define the foci
def F₁ : ℝ × ℝ := (0, 1)
def F₂ : ℝ × ℝ := (6, 1)

-- Define the distance sum constant
def distance_sum : ℝ := 8

-- Define the ellipse properties
def ellipse_properties (h k a b : ℝ) : Prop :=
  ∀ (x y : ℝ),
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔
    Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
    Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = distance_sum

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  ellipse_properties h k a b →
  h + k + a + b = 8 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l1532_153241


namespace NUMINAMATH_CALUDE_alex_not_reading_probability_l1532_153214

theorem alex_not_reading_probability :
  let p_reading : ℚ := 5/9
  let p_not_reading : ℚ := 1 - p_reading
  p_not_reading = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_alex_not_reading_probability_l1532_153214


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1532_153275

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a = 4 * b →    -- ratio of angles is 4:1
  b = 36 :=      -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1532_153275


namespace NUMINAMATH_CALUDE_tims_toads_l1532_153233

theorem tims_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ) 
  (h1 : jim_toads = tim_toads + 20)
  (h2 : sarah_toads = 2 * jim_toads)
  (h3 : sarah_toads = 100) : 
  tim_toads = 30 := by
sorry

end NUMINAMATH_CALUDE_tims_toads_l1532_153233


namespace NUMINAMATH_CALUDE_factorial_not_divisible_by_square_l1532_153215

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem factorial_not_divisible_by_square (n : ℕ) :
  ¬((n - 1).factorial % (n^2) = 0) ↔ 
  (n = 8 ∨ n = 9 ∨ 
   (∃ p : ℕ, is_prime p ∧ (n = p ∨ n = 2*p))) :=
sorry

end NUMINAMATH_CALUDE_factorial_not_divisible_by_square_l1532_153215


namespace NUMINAMATH_CALUDE_equation_three_real_roots_l1532_153256

theorem equation_three_real_roots (k : ℂ) : 
  (∃! (r₁ r₂ r₃ : ℝ), ∀ (x : ℝ), 
    (x / (x + 3) + x / (x - 3) = k * x) ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ↔ 
  (k = Complex.I / 3 ∨ k = -Complex.I / 3) :=
sorry

end NUMINAMATH_CALUDE_equation_three_real_roots_l1532_153256


namespace NUMINAMATH_CALUDE_race_distance_l1532_153213

/-- The race problem -/
theorem race_distance (d : ℝ) (a b c : ℝ) : 
  d > 0 ∧ a > b ∧ b > c ∧ a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive distances and speeds
  d / a = (d - 15) / b →  -- A beats B by 15 meters
  d / b = (d - 30) / c →  -- B beats C by 30 meters
  d / a = (d - 40) / c →  -- A beats C by 40 meters
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l1532_153213


namespace NUMINAMATH_CALUDE_equation_solutions_l1532_153225

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2))
  ∀ x : ℝ, f x = 54 ↔ x = 9 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1532_153225


namespace NUMINAMATH_CALUDE_sector_area_l1532_153278

/-- The area of a sector with given arc length and diameter -/
theorem sector_area (arc_length diameter : ℝ) (h1 : arc_length = 30) (h2 : diameter = 16) :
  (1 / 2) * (diameter / 2) * arc_length = 120 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l1532_153278


namespace NUMINAMATH_CALUDE_most_accurate_reading_l1532_153262

def scale_start : ℝ := 10.25
def scale_end : ℝ := 10.5
def arrow_position : ℝ := 10.3  -- Approximate position based on the problem description

def options : List ℝ := [10.05, 10.15, 10.25, 10.3, 10.6]

theorem most_accurate_reading :
  scale_start < arrow_position ∧ 
  arrow_position < scale_end ∧
  |arrow_position - 10.3| < |arrow_position - ((scale_start + scale_end) / 2)| →
  (options.filter (λ x => x ≥ scale_start ∧ x ≤ scale_end)).argmin (λ x => |x - arrow_position|) = some 10.3 := by
  sorry

end NUMINAMATH_CALUDE_most_accurate_reading_l1532_153262


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l1532_153280

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_intersection_equals_specific_set :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l1532_153280


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1532_153277

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 4

-- State the theorem
theorem f_decreasing_interval :
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  (∃ a : ℝ, ∀ x y : ℝ, a ≤ x ∧ x < y → f y < f x) ∧
  (∀ a : ℝ, (∀ x y : ℝ, a ≤ x ∧ x < y → f y < f x) → a ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1532_153277


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_to_9000_l1532_153252

theorem last_three_digits_of_2_to_9000 (h : 2^300 ≡ 1 [ZMOD 1000]) :
  2^9000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_to_9000_l1532_153252


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l1532_153259

/-- Represents the problem of determining the planned daily ploughing area for a farmer --/
theorem farmer_ploughing_problem 
  (total_area : ℝ) 
  (actual_daily_area : ℝ) 
  (extra_days : ℕ) 
  (area_left : ℝ) 
  (h1 : total_area = 448) 
  (h2 : actual_daily_area = 85) 
  (h3 : extra_days = 2) 
  (h4 : area_left = 40) : 
  ∃ planned_daily_area : ℝ, 
    planned_daily_area = 188.5 ∧ 
    (total_area / planned_daily_area + extra_days) * actual_daily_area = total_area - area_left :=
sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l1532_153259


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1532_153223

theorem quadratic_roots_difference (p q : ℝ) : 
  p^2 - 7*p + 12 = 0 → q^2 - 7*q + 12 = 0 → |p - q| = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1532_153223


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l1532_153261

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10) (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l1532_153261


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1532_153238

theorem fraction_equation_solution (n : ℚ) : 
  2 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4 → n = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1532_153238


namespace NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l1532_153296

/-- Given a cylinder with a square side-unfolding, the ratio of its total surface area
    to its side surface area is (1 + 2π) / (2π). -/
theorem cylinder_surface_area_ratio (r : ℝ) (h : r > 0) :
  let height := 2 * π * r
  let side_area := 2 * π * r * height
  let total_area := side_area + 2 * π * r^2
  total_area / side_area = (1 + 2 * π) / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l1532_153296


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1532_153285

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 1 * a 3 + 2 * a 2 * a 3 + a 1 * a 5 = 16 →
  a 2 + a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1532_153285


namespace NUMINAMATH_CALUDE_monomial_exponents_l1532_153246

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℤ) : Prop :=
  ∀ x y, ∃ k₁ k₂ : ℤ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ a x y = k₁ ∧ b x y = k₂

theorem monomial_exponents (m n : ℕ) :
  like_terms (fun x y => 6 * x^5 * y^(2*n)) (fun x y => -2 * x^m * y^4) →
  m + 2*n = 9 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_l1532_153246


namespace NUMINAMATH_CALUDE_donation_to_third_orphanage_l1532_153276

theorem donation_to_third_orphanage 
  (total_donation : ℝ)
  (first_orphanage : ℝ)
  (second_orphanage : ℝ)
  (h1 : total_donation = 650)
  (h2 : first_orphanage = 175)
  (h3 : second_orphanage = 225) :
  total_donation - (first_orphanage + second_orphanage) = 250 := by
  sorry

end NUMINAMATH_CALUDE_donation_to_third_orphanage_l1532_153276


namespace NUMINAMATH_CALUDE_salty_sweet_difference_l1532_153222

/-- Represents the number of cookies Paco had and ate -/
structure CookieCount where
  initialSweet : ℕ
  initialSalty : ℕ
  eatenSweet : ℕ
  eatenSalty : ℕ

/-- Theorem stating the difference between salty and sweet cookies eaten -/
theorem salty_sweet_difference (c : CookieCount)
  (h1 : c.initialSweet = 40)
  (h2 : c.initialSalty = 25)
  (h3 : c.eatenSweet = 15)
  (h4 : c.eatenSalty = 28) :
  c.eatenSalty - c.eatenSweet = 13 := by
  sorry

end NUMINAMATH_CALUDE_salty_sweet_difference_l1532_153222


namespace NUMINAMATH_CALUDE_shortest_chord_through_M_l1532_153249

/-- The equation of the circle O -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 10 = 0

/-- The coordinates of point M -/
def point_M : ℝ × ℝ := (3, 0)

/-- The equation of the line containing the shortest chord through point M -/
def shortest_chord_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem stating that the given line equation is indeed the shortest chord through M -/
theorem shortest_chord_through_M :
  ∀ (x y : ℝ), circle_equation x y →
  shortest_chord_line x y ↔ 
  (∀ (l : ℝ → ℝ → Prop), 
    (l point_M.1 point_M.2) → 
    (∃ (p q : ℝ), circle_equation p q ∧ l p q) →
    (∀ (a b : ℝ), circle_equation a b ∧ l a b → 
      (a - point_M.1)^2 + (b - point_M.2)^2 ≥ (x - point_M.1)^2 + (y - point_M.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_through_M_l1532_153249


namespace NUMINAMATH_CALUDE_min_value_inequality_l1532_153251

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) * ((a + b + d)⁻¹ + (a + c + d)⁻¹ + (b + c + d)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1532_153251


namespace NUMINAMATH_CALUDE_snackles_remainder_l1532_153254

theorem snackles_remainder (m : ℕ) (h : m % 11 = 4) : (3 * m) % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_snackles_remainder_l1532_153254


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1532_153260

theorem quadratic_equation_properties (k : ℝ) (a b : ℝ) :
  (∀ x, x^2 + 2*x - k = 0 ↔ x = a ∨ x = b) →
  a ≠ b →
  (k > -1) ∧ (a / (a + 1) - 1 / (b + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1532_153260


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l1532_153287

theorem inequality_implies_lower_bound (x y : ℝ) 
  (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) : 
  x ≥ -1/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l1532_153287


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1532_153209

theorem ten_thousandths_place_of_5_32 :
  (5 : ℚ) / 32 = 0.15625 := by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1532_153209


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1532_153258

theorem sum_of_three_numbers : 1.48 + 2.32 + 8.45 = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1532_153258


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1532_153235

def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 3 then {x | x < -3/a ∨ x > -1}
  else if a = 3 then {x | x ≠ -1}
  else if a > 3 then {x | x < -1 ∨ x > -3/a}
  else if a = 0 then {x | x > -1}
  else if a < 0 then {x | -1 < x ∧ x < -3/a}
  else ∅

theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  a * x^2 + 3 * x + 2 > -a * x - 1 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1532_153235


namespace NUMINAMATH_CALUDE_animal_pairing_theorem_l1532_153210

/-- Represents the number of dogs in the problem -/
def num_dogs : ℕ := 5

/-- Represents the number of cats in the problem -/
def num_cats : ℕ := 4

/-- Represents the number of bowls of milk in the problem -/
def num_bowls : ℕ := 7

/-- Represents the total number of animals (dogs and cats) -/
def total_animals : ℕ := num_dogs + num_cats

/-- Represents the number of ways to pair dogs and cats -/
def num_pairings : ℕ := num_dogs * num_cats

theorem animal_pairing_theorem :
  num_pairings = 20 ∧
  total_animals = num_bowls + 2 :=
sorry

end NUMINAMATH_CALUDE_animal_pairing_theorem_l1532_153210


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1532_153279

/-- Given two vectors a and b in ℝ², prove that if k*a + b is parallel to a - 3*b, 
    then k = -1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (-3, 2)) 
    (h_parallel : ∃ (c : ℝ), c • (k • a + b) = a - 3 • b) : 
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1532_153279


namespace NUMINAMATH_CALUDE_three_n_squared_plus_nine_composite_l1532_153265

theorem three_n_squared_plus_nine_composite (n : ℕ) : ∃ (k : ℕ), k > 1 ∧ k ∣ (3 * n^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_three_n_squared_plus_nine_composite_l1532_153265


namespace NUMINAMATH_CALUDE_ethan_works_five_days_per_week_l1532_153271

/-- Calculates the number of days Ethan works per week given his hourly rate, daily hours, total earnings, and number of weeks worked. -/
def days_worked_per_week (hourly_rate : ℚ) (hours_per_day : ℚ) (total_earnings : ℚ) (num_weeks : ℚ) : ℚ :=
  total_earnings / num_weeks / (hourly_rate * hours_per_day)

/-- Proves that Ethan works 5 days per week given the problem conditions. -/
theorem ethan_works_five_days_per_week :
  days_worked_per_week 18 8 3600 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ethan_works_five_days_per_week_l1532_153271


namespace NUMINAMATH_CALUDE_aziz_is_36_l1532_153266

/-- Calculates Aziz's age in 2021 given the year his parents moved to America and the number of years they lived there before he was born -/
def aziz_age (parents_move_year : ℕ) (years_before_birth : ℕ) : ℕ :=
  2021 - (parents_move_year + years_before_birth)

/-- Theorem stating that Aziz's age in 2021 is 36 years -/
theorem aziz_is_36 : aziz_age 1982 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_aziz_is_36_l1532_153266


namespace NUMINAMATH_CALUDE_largest_c_for_g_range_two_l1532_153298

/-- The quadratic function g(x) = x^2 - 6x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: The largest value of c for which 2 is in the range of g(x) is 11 -/
theorem largest_c_for_g_range_two :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 2) ↔ c ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_g_range_two_l1532_153298


namespace NUMINAMATH_CALUDE_initial_bales_count_l1532_153263

theorem initial_bales_count (added_bales current_total : ℕ) 
  (h1 : added_bales = 26)
  (h2 : current_total = 54)
  : current_total - added_bales = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_bales_count_l1532_153263


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1532_153264

theorem part_to_whole_ratio 
  (N P : ℚ) 
  (h1 : (1/4) * (1/3) * P = 17) 
  (h2 : (2/5) * N = 204) : 
  P/N = 2/5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1532_153264


namespace NUMINAMATH_CALUDE_coloring_books_distribution_l1532_153206

def books_per_shelf (initial_stock : ℕ) (books_sold : ℕ) (num_shelves : ℕ) : ℕ :=
  (initial_stock - books_sold) / num_shelves

theorem coloring_books_distribution 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (num_shelves : ℕ) 
  (h1 : initial_stock = 40) 
  (h2 : books_sold = 20) 
  (h3 : num_shelves = 5) :
  books_per_shelf initial_stock books_sold num_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_distribution_l1532_153206


namespace NUMINAMATH_CALUDE_poetry_society_arrangement_l1532_153248

-- Define the number of people in the group
def total_people : ℕ := 8

-- Define the number of people who must not be adjacent
def special_people : ℕ := 3

-- Define the number of remaining people
def remaining_people : ℕ := total_people - special_people

-- Define the number of spaces available for special people
def available_spaces : ℕ := remaining_people + 1

-- Theorem statement
theorem poetry_society_arrangement :
  (remaining_people.factorial) * (available_spaces.factorial / (available_spaces - special_people).factorial) = 14400 :=
sorry

end NUMINAMATH_CALUDE_poetry_society_arrangement_l1532_153248


namespace NUMINAMATH_CALUDE_b_plus_c_positive_l1532_153201

theorem b_plus_c_positive (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_positive_l1532_153201


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l1532_153232

theorem negative_one_to_zero_power : (-1 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l1532_153232


namespace NUMINAMATH_CALUDE_tangent_product_equality_l1532_153293

theorem tangent_product_equality : 
  (1 + Real.tan (17 * π / 180)) * 
  (1 + Real.tan (28 * π / 180)) * 
  (1 + Real.tan (27 * π / 180)) * 
  (1 + Real.tan (18 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equality_l1532_153293


namespace NUMINAMATH_CALUDE_management_subcommittee_count_l1532_153228

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers managers subcommitteeSize minManagers : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize -
  (choose (totalMembers - managers) subcommitteeSize +
   choose managers 1 * choose (totalMembers - managers) (subcommitteeSize - 1))

theorem management_subcommittee_count :
  validSubcommittees 12 5 5 2 = 596 := by sorry

end NUMINAMATH_CALUDE_management_subcommittee_count_l1532_153228


namespace NUMINAMATH_CALUDE_parabola_equation_l1532_153281

/-- Given a parabola C: y²=2px (p>0) with focus F, and a point A on C such that the midpoint of AF is (2,2), prove that the equation of C is y² = 8x. -/
theorem parabola_equation (p : ℝ) (F A : ℝ × ℝ) (h1 : p > 0) (h2 : F = (p/2, 0)) 
  (h3 : A.1^2 = 2*p*A.2) (h4 : ((F.1 + A.1)/2, (F.2 + A.2)/2) = (2, 2)) :
  ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*p*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1532_153281


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l1532_153224

/-- Converts a list of binary digits to its decimal representation. -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 19 as a list of booleans. -/
def binary_19 : List Bool := [true, true, false, false, true]

/-- Theorem stating that the binary number 10011 is equal to the decimal number 19. -/
theorem binary_10011_equals_19 : binary_to_decimal binary_19 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l1532_153224


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1532_153230

/-- Represents a two-digit number with units digit x -/
def two_digit_number (x : ℕ) : ℕ := 10 * (x + 4) + x

/-- The equation that needs to be proven -/
def equation_holds (x : ℕ) : Prop :=
  x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4

/-- The sum of squares of digits is 4 less than the number -/
def sum_squares_property (x : ℕ) : Prop :=
  x^2 + (x + 4)^2 = two_digit_number x - 4

theorem two_digit_number_property (x : ℕ) :
  equation_holds x ↔ sum_squares_property x :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1532_153230


namespace NUMINAMATH_CALUDE_triangle_height_decrease_l1532_153219

theorem triangle_height_decrease (b h : ℝ) (b_new h_new : ℝ) :
  b > 0 → h > 0 →
  b_new = 1.1 * b →
  (1/2) * b_new * h_new = 1.045 * ((1/2) * b * h) →
  h_new = 0.5 * h := by
sorry

end NUMINAMATH_CALUDE_triangle_height_decrease_l1532_153219


namespace NUMINAMATH_CALUDE_farm_tree_count_l1532_153257

/-- Represents the state of trees on the farm --/
structure FarmTrees where
  mahogany : ℕ
  narra : ℕ

/-- Calculates the total number of trees --/
def total_trees (ft : FarmTrees) : ℕ := ft.mahogany + ft.narra

/-- Represents the number of fallen trees --/
structure FallenTrees where
  mahogany : ℕ
  narra : ℕ

/-- Represents the farm's tree management process --/
def farm_process (initial : FarmTrees) (fallen : FallenTrees) : ℕ :=
  let remaining := total_trees initial - (fallen.mahogany + fallen.narra)
  let new_mahogany := 3 * fallen.mahogany
  let new_narra := 2 * fallen.narra
  remaining + new_mahogany + new_narra

/-- Theorem stating the final number of trees on the farm --/
theorem farm_tree_count : 
  ∀ (initial : FarmTrees) (fallen : FallenTrees),
  initial.mahogany = 50 → 
  initial.narra = 30 → 
  fallen.mahogany + fallen.narra = 5 →
  fallen.mahogany = fallen.narra + 1 →
  farm_process initial fallen = 88 := by
  sorry


end NUMINAMATH_CALUDE_farm_tree_count_l1532_153257


namespace NUMINAMATH_CALUDE_circle_radius_in_triangle_l1532_153242

/-- Triangle DEF with specified side lengths -/
structure Triangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  h_de : de = 64
  h_df : df = 64
  h_ef : ef = 72

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Tangency relation between circle and line segment -/
def IsTangent (c : Circle) (a b : ℝ × ℝ) : Prop := sorry

/-- External tangency relation between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- A circle is inside a triangle -/
def IsInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) :
  t.de = 64 →
  t.df = 64 →
  t.ef = 72 →
  r.radius = 20 →
  IsTangent r (0, 0) (t.df, 0) →
  IsTangent r (t.ef, 0) (0, 0) →
  IsExternallyTangent s r →
  IsTangent s (0, 0) (t.de, 0) →
  IsTangent s (t.ef, 0) (0, 0) →
  IsInside s t →
  s.radius = 52 - 4 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_triangle_l1532_153242


namespace NUMINAMATH_CALUDE_first_chapter_has_48_pages_l1532_153211

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  second_chapter_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def first_chapter_pages (b : Book) : ℕ :=
  b.total_pages - b.second_chapter_pages

/-- Theorem stating that for a book with 94 total pages and 46 pages in the second chapter,
    the first chapter has 48 pages -/
theorem first_chapter_has_48_pages (b : Book)
    (h1 : b.total_pages = 94)
    (h2 : b.second_chapter_pages = 46) :
    first_chapter_pages b = 48 := by
  sorry


end NUMINAMATH_CALUDE_first_chapter_has_48_pages_l1532_153211


namespace NUMINAMATH_CALUDE_total_milk_volume_l1532_153272

-- Define the conversion factor from milliliters to liters
def ml_to_l : ℚ := 1 / 1000

-- Define the volumes of milk in liters
def volume1 : ℚ := 2
def volume2 : ℚ := 750 * ml_to_l
def volume3 : ℚ := 250 * ml_to_l

-- State the theorem
theorem total_milk_volume :
  volume1 + volume2 + volume3 = 3 := by sorry

end NUMINAMATH_CALUDE_total_milk_volume_l1532_153272
