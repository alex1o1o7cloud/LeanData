import Mathlib

namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1209_120975

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the configuration of a room with a sloped ceiling -/
structure Room where
  p : Point3D -- Point where walls and ceiling meet
  slope : ℝ -- Slope of the ceiling (rise / run)

/-- Represents the position of a fly in the room -/
structure FlyPosition where
  distWall1 : ℝ -- Distance from first wall
  distWall2 : ℝ -- Distance from second wall
  distFromP : ℝ -- Distance from point P

/-- Calculates the distance of a fly from the sloped ceiling in a room -/
def distanceFromCeiling (r : Room) (f : FlyPosition) : ℝ :=
  sorry

/-- Theorem stating that the fly's distance from the ceiling is (3√60 - 8)/3 -/
theorem fly_distance_from_ceiling (r : Room) (f : FlyPosition) :
  r.p = Point3D.mk 0 0 0 →
  r.slope = 1/3 →
  f.distWall1 = 2 →
  f.distWall2 = 6 →
  f.distFromP = 10 →
  distanceFromCeiling r f = (3 * Real.sqrt 60 - 8) / 3 :=
sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1209_120975


namespace NUMINAMATH_CALUDE_ABD_collinear_l1209_120979

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b : V)

-- Define the points
variable (A B C D : V)

-- Define the vector relationships
axiom AB_def : B - A = a + 2 • b
axiom BC_def : C - B = -5 • a + 6 • b
axiom CD_def : D - C = 7 • a - 2 • b

-- Theorem to prove
theorem ABD_collinear : ∃ (t : ℝ), D - A = t • (B - A) := by
  sorry

end NUMINAMATH_CALUDE_ABD_collinear_l1209_120979


namespace NUMINAMATH_CALUDE_problem_statement_l1209_120969

theorem problem_statement (x y : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : x * y = -15) :
  4 * x^2 + 4 * y^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1209_120969


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1209_120943

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ ∃ x, f x < m :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1209_120943


namespace NUMINAMATH_CALUDE_problem_1_l1209_120904

theorem problem_1 : (-20) + 3 - (-5) - 7 = -19 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1209_120904


namespace NUMINAMATH_CALUDE_dinos_remaining_balance_l1209_120999

/-- Represents a gig with hours worked per month and hourly rate -/
structure Gig where
  hours : ℕ
  rate : ℕ

/-- Calculates the monthly earnings from a gig -/
def monthlyEarnings (g : Gig) : ℕ := g.hours * g.rate

/-- Represents Dino's gigs -/
def dinos_gigs : List Gig := [
  ⟨20, 10⟩,
  ⟨30, 20⟩,
  ⟨5, 40⟩,
  ⟨15, 25⟩,
  ⟨10, 30⟩
]

/-- Dino's monthly expenses for each month -/
def monthly_expenses : List ℕ := [500, 550, 520, 480]

/-- The number of months -/
def num_months : ℕ := 4

theorem dinos_remaining_balance :
  (dinos_gigs.map monthlyEarnings).sum * num_months -
  monthly_expenses.sum = 4650 := by sorry

end NUMINAMATH_CALUDE_dinos_remaining_balance_l1209_120999


namespace NUMINAMATH_CALUDE_factorization_equality_l1209_120937

theorem factorization_equality (x y : ℝ) : 4 * x^2 - 8 * x * y + 4 * y^2 = 4 * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1209_120937


namespace NUMINAMATH_CALUDE_joan_seashells_l1209_120938

theorem joan_seashells (total : ℝ) (percentage : ℝ) (remaining : ℝ) : 
  total = 79.5 → 
  percentage = 45 → 
  remaining = total - (percentage / 100) * total → 
  remaining = 43.725 := by
sorry

end NUMINAMATH_CALUDE_joan_seashells_l1209_120938


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l1209_120994

theorem hyperbola_intersection_line (θ : Real) : 
  let ρ := λ θ : Real => 3 / (1 - 2 * Real.cos θ)
  let A := (ρ θ, θ)
  let B := (ρ (θ + π), θ + π)
  let distance := |ρ θ + ρ (θ + π)|
  distance = 6 → 
    θ = π/2 ∨ θ = π/4 ∨ θ = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l1209_120994


namespace NUMINAMATH_CALUDE_max_expenditure_max_expected_expenditure_l1209_120934

-- Define the linear regression model
def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

-- State the theorem
theorem max_expenditure (x : ℝ) (e : ℝ) :
  x = 10 →
  0.8 * x + 2 + e ≤ 10.5 :=
by
  sorry

-- Define the constraint on e
def e_constraint (e : ℝ) : Prop := abs e ≤ 0.5

-- State the main theorem
theorem max_expected_expenditure (x : ℝ) :
  x = 10 →
  ∀ e, e_constraint e →
  linear_regression x 0.8 2 e ≤ 10.5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expenditure_max_expected_expenditure_l1209_120934


namespace NUMINAMATH_CALUDE_frame_cells_l1209_120924

theorem frame_cells (n : ℕ) (h : n = 254) : 
  n^2 - (n - 2)^2 = 2016 :=
by sorry

end NUMINAMATH_CALUDE_frame_cells_l1209_120924


namespace NUMINAMATH_CALUDE_square_divided_into_rectangles_l1209_120900

theorem square_divided_into_rectangles (square_perimeter : ℝ) 
  (h1 : square_perimeter = 200) : 
  let side_length := square_perimeter / 4
  let rectangle_length := side_length
  let rectangle_width := side_length / 2
  2 * (rectangle_length + rectangle_width) = 150 := by
sorry

end NUMINAMATH_CALUDE_square_divided_into_rectangles_l1209_120900


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l1209_120957

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r ∧ 
    (∀ s : ℕ, Nat.Prime s ∧ s ∣ x → s = p ∨ s = q ∨ s = r)) →
  7 ∣ x →
  x = 728 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l1209_120957


namespace NUMINAMATH_CALUDE_parabola_equation_l1209_120976

/-- Given a parabola in the form x² = 2py where p > 0, with axis of symmetry y = -1/2,
    prove that its equation is x² = 2y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : -p/2 = -1/2) :
  ∀ x y : ℝ, x^2 = 2*p*y ↔ x^2 = 2*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1209_120976


namespace NUMINAMATH_CALUDE_doughnut_cost_l1209_120939

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℝ := sorry

/-- The cost of two dozen doughnuts -/
def cost_two_dozen : ℝ := 14

theorem doughnut_cost : cost_one_dozen = 7 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_cost_l1209_120939


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l1209_120995

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ (∀ x y : ℝ, mx + y - 3 = 0 → 2*x + m*(m-1)*y + 2 = 0 → m*2 + 1*m*(m-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l1209_120995


namespace NUMINAMATH_CALUDE_two_cars_meeting_l1209_120909

/-- Two cars meeting on a highway problem -/
theorem two_cars_meeting (highway_length : ℝ) (car1_speed : ℝ) (meeting_time : ℝ) :
  highway_length = 45 →
  car1_speed = 14 →
  meeting_time = 1.5 →
  ∃ car2_speed : ℝ,
    car2_speed = 16 ∧
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length :=
by sorry

end NUMINAMATH_CALUDE_two_cars_meeting_l1209_120909


namespace NUMINAMATH_CALUDE_profit_percentage_l1209_120961

theorem profit_percentage (selling_price cost_price profit : ℝ) : 
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 100/3 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_l1209_120961


namespace NUMINAMATH_CALUDE_valid_table_iff_odd_l1209_120911

/-- A square table of size n × n -/
def SquareTable (n : ℕ) := Fin n → Fin n → ℚ

/-- The sum of numbers on a diagonal of a square table -/
def diagonalSum (table : SquareTable n) (d : ℕ) : ℚ :=
  sorry

/-- A square table is valid if the sum of numbers on each diagonal is 1 -/
def isValidTable (table : SquareTable n) : Prop :=
  ∀ d, d < 4*n - 2 → diagonalSum table d = 1

/-- There exists a valid square table of size n × n if and only if n is odd -/
theorem valid_table_iff_odd (n : ℕ) :
  (∃ (table : SquareTable n), isValidTable table) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_table_iff_odd_l1209_120911


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1209_120903

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: If x^2 - kx + 64 is a perfect square trinomial, then k = 16 or k = -16 -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 64 → k = 16 ∨ k = -16 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1209_120903


namespace NUMINAMATH_CALUDE_lace_cost_per_meter_l1209_120926

-- Define the lengths in centimeters
def cuff_length : ℝ := 50
def hem_length : ℝ := 300
def ruffle_length : ℝ := 20
def total_cost : ℝ := 36

-- Define the number of cuffs and ruffles
def num_cuffs : ℕ := 2
def num_ruffles : ℕ := 5

-- Define the conversion factor from cm to m
def cm_to_m : ℝ := 100

-- Theorem to prove
theorem lace_cost_per_meter :
  let total_length := num_cuffs * cuff_length + hem_length + (hem_length / 3) + num_ruffles * ruffle_length
  let total_length_m := total_length / cm_to_m
  total_cost / total_length_m = 6 := by
  sorry

end NUMINAMATH_CALUDE_lace_cost_per_meter_l1209_120926


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1209_120933

theorem triangle_side_difference (y : ℕ) : 
  (y > 0 ∧ y + 7 > 9 ∧ y + 9 > 7 ∧ 7 + 9 > y) →
  (∃ (max min : ℕ), 
    (∀ z : ℕ, (z > 0 ∧ z + 7 > 9 ∧ z + 9 > 7 ∧ 7 + 9 > z) → z ≤ max ∧ z ≥ min) ∧
    max - min = 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1209_120933


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_a_minus_b_eq_one_l1209_120915

/-- The polynomial in question -/
def P (a b x y : ℝ) : ℝ := x^2 + a*x*y + b*y^2 - 5*x + y + 6

/-- The factor of the polynomial -/
def F (x y : ℝ) : ℝ := x + y - 2

theorem polynomial_factor_implies_a_minus_b_eq_one (a b : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, P a b x y = F x y * k) →
  a - b = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_a_minus_b_eq_one_l1209_120915


namespace NUMINAMATH_CALUDE_trajectory_equation_l1209_120945

/-- Given a fixed point A(1, 2) and a moving point P(x, y) in a Cartesian coordinate system,
    if OP · OA = 4, then the equation of the trajectory of P is x + 2y - 4 = 0. -/
theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let O : ℝ × ℝ := (0, 0)
  (P.1 - O.1) * (A.1 - O.1) + (P.2 - O.2) * (A.2 - O.2) = 4 →
  x + 2 * y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1209_120945


namespace NUMINAMATH_CALUDE_john_participation_count_l1209_120977

/-- Represents the possible point values in the archery competition -/
inductive ArcheryPoints
  | first : ArcheryPoints
  | second : ArcheryPoints
  | third : ArcheryPoints
  | fourth : ArcheryPoints

/-- Returns the point value for a given place -/
def pointValue (p : ArcheryPoints) : Nat :=
  match p with
  | ArcheryPoints.first => 11
  | ArcheryPoints.second => 7
  | ArcheryPoints.third => 5
  | ArcheryPoints.fourth => 2

/-- Represents John's participation in the archery competition -/
def JohnParticipation := List ArcheryPoints

/-- Calculates the product of points for a given participation list -/
def productOfPoints (participation : JohnParticipation) : Nat :=
  participation.foldl (fun acc p => acc * pointValue p) 1

/-- Theorem: John participated 7 times given the conditions -/
theorem john_participation_count :
  ∃ (participation : JohnParticipation),
    productOfPoints participation = 38500 ∧ participation.length = 7 :=
by sorry

end NUMINAMATH_CALUDE_john_participation_count_l1209_120977


namespace NUMINAMATH_CALUDE_debt_doubling_time_l1209_120908

def interest_rate : ℝ := 0.07

theorem debt_doubling_time : 
  ∀ t : ℕ, t < 10 → (1 + interest_rate) ^ t ≤ 2 ∧ 
  (1 + interest_rate) ^ 10 > 2 := by sorry

end NUMINAMATH_CALUDE_debt_doubling_time_l1209_120908


namespace NUMINAMATH_CALUDE_find_y_value_l1209_120921

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1209_120921


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1209_120920

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 :
  ∃ m : ℕ+, (3^100 * m.val + (3^100 - 1)) % 2005 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l1209_120920


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1209_120913

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define point P on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b 2 3

-- Define the condition for slope of MA being 1 and MF = AF
def slope_and_distance_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola a b x y ∧ (y - (-a)) / (x - (-a)) = 1 ∧
  (x - 2*a)^2 + y^2 = (3*a)^2

-- Define the perpendicularity condition
def perpendicular_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  point_on_hyperbola a b →
  slope_and_distance_condition a b →
  perpendicular_condition a b →
  (a = 1 ∧ b = Real.sqrt 3) ∧
  (∀ (k t : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola 1 (Real.sqrt 3) x₁ y₁ ∧
    hyperbola 1 (Real.sqrt 3) x₂ y₂ ∧
    y₁ = k * x₁ + t ∧
    y₂ = k * x₂ + t) →
  |t| / Real.sqrt (1 + k^2) = Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1209_120913


namespace NUMINAMATH_CALUDE_max_b_cubic_function_max_b_value_l1209_120983

/-- A cubic function f(x) = ax³ + bx + c -/
def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

/-- The maximum possible value of b in a cubic function f(x) = ax³ + bx + c
    where 0 ≤ f(x) ≤ 1 for all x in [0, 1] -/
theorem max_b_cubic_function :
  ∃ (b_max : ℝ),
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

/-- The maximum possible value of b is 3√3/2 -/
theorem max_b_value : 
  ∃ (b_max : ℝ),
    b_max = 3 * Real.sqrt 3 / 2 ∧
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_max_b_cubic_function_max_b_value_l1209_120983


namespace NUMINAMATH_CALUDE_data_value_proof_l1209_120941

theorem data_value_proof (a b c : ℝ) 
  (h1 : a + b = c)
  (h2 : b = 3 * a)
  (h3 : a + b + c = 96) :
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_data_value_proof_l1209_120941


namespace NUMINAMATH_CALUDE_triangle_inequality_l1209_120972

/-- Given a triangle with side lengths a, b, and c, 
    the inequality a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) ≥ 0 holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1209_120972


namespace NUMINAMATH_CALUDE_pudding_distribution_l1209_120919

theorem pudding_distribution (total_cups : ℕ) (additional_cups : ℕ) 
  (h1 : total_cups = 315)
  (h2 : additional_cups = 121)
  (h3 : ∀ (students : ℕ), students > 0 → 
    (total_cups + additional_cups) % students = 0 → 
    total_cups < students * ((total_cups + additional_cups) / students)) :
  ∃ (students : ℕ), students = 4 ∧ 
    (total_cups + additional_cups) % students = 0 ∧
    total_cups < students * ((total_cups + additional_cups) / students) :=
by sorry

end NUMINAMATH_CALUDE_pudding_distribution_l1209_120919


namespace NUMINAMATH_CALUDE_mrs_hilt_friends_l1209_120950

/-- Mrs. Hilt's friends problem -/
theorem mrs_hilt_friends (friends_can_go : ℕ) (friends_cant_go : ℕ) 
  (h1 : friends_can_go = 8) (h2 : friends_cant_go = 7) : 
  friends_can_go + friends_cant_go = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_friends_l1209_120950


namespace NUMINAMATH_CALUDE_jack_sugar_today_l1209_120914

/-- The amount of sugar Jack has today -/
def S : ℕ := by sorry

/-- Theorem: Jack has 65 pounds of sugar today -/
theorem jack_sugar_today : S = 65 := by
  have h1 : S - 18 + 50 = 97 := by sorry
  sorry


end NUMINAMATH_CALUDE_jack_sugar_today_l1209_120914


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l1209_120936

theorem triangle_side_sum_max (a b c : ℝ) (C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  a > 0 →
  b > 0 →
  c > 0 →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  a + b ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l1209_120936


namespace NUMINAMATH_CALUDE_suraya_caleb_difference_l1209_120912

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := kayla_apples - 5

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- Theorem stating the difference between Suraya's and Caleb's apple count -/
theorem suraya_caleb_difference : suraya_apples - caleb_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_suraya_caleb_difference_l1209_120912


namespace NUMINAMATH_CALUDE_longest_segment_l1209_120956

-- Define the triangle ABD
structure TriangleABD where
  angleABD : ℝ
  angleADB : ℝ
  hab : angleABD = 30
  had : angleADB = 70

-- Define the triangle BCD
structure TriangleBCD where
  angleCBD : ℝ
  angleBDC : ℝ
  hcb : angleCBD = 45
  hbd : angleBDC = 60

-- Define the lengths of the segments
variables {AB AD BD BC CD : ℝ}

-- State the theorem
theorem longest_segment (abd : TriangleABD) (bcd : TriangleBCD) :
  CD > BC ∧ BC > BD ∧ BD > AB ∧ AB > AD :=
sorry

end NUMINAMATH_CALUDE_longest_segment_l1209_120956


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1209_120964

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 6
  let θ : ℝ := (7 * π) / 4
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1209_120964


namespace NUMINAMATH_CALUDE_annie_hamburger_cost_l1209_120927

/-- Calculates the cost of a single hamburger given the initial amount,
    cost of a milkshake, number of hamburgers and milkshakes bought,
    and the remaining amount after purchase. -/
def hamburger_cost (initial_amount : ℕ) (milkshake_cost : ℕ) 
                   (hamburgers_bought : ℕ) (milkshakes_bought : ℕ) 
                   (remaining_amount : ℕ) : ℕ :=
  (initial_amount - remaining_amount - milkshake_cost * milkshakes_bought) / hamburgers_bought

/-- Theorem stating that given Annie's purchases and finances, 
    each hamburger costs $4. -/
theorem annie_hamburger_cost : 
  hamburger_cost 132 5 8 6 70 = 4 := by
  sorry

end NUMINAMATH_CALUDE_annie_hamburger_cost_l1209_120927


namespace NUMINAMATH_CALUDE_valid_systematic_sampling_l1209_120965

/-- Represents a systematic sampling selection -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the set of selected numbers for a systematic sampling -/
def generateSelection (s : SystematicSampling) : Finset Nat :=
  let interval := s.totalStudents / s.sampleSize
  Finset.image (fun i => s.startingNumber + i * interval) (Finset.range s.sampleSize)

/-- Theorem stating that {3, 13, 23, 33, 43} is a valid systematic sampling selection -/
theorem valid_systematic_sampling :
  ∃ (s : SystematicSampling),
    s.totalStudents = 50 ∧
    s.sampleSize = 5 ∧
    1 ≤ s.startingNumber ∧
    s.startingNumber ≤ s.totalStudents ∧
    generateSelection s = {3, 13, 23, 33, 43} :=
sorry

end NUMINAMATH_CALUDE_valid_systematic_sampling_l1209_120965


namespace NUMINAMATH_CALUDE_fruit_problem_solution_l1209_120963

/-- Represents the solution to the fruit buying problem -/
def FruitSolution : Type := ℕ × ℕ × ℕ

/-- The total number of fruits bought -/
def total_fruits : ℕ := 100

/-- The total cost in copper coins -/
def total_cost : ℕ := 100

/-- The cost of a single peach in copper coins -/
def peach_cost : ℕ := 3

/-- The cost of a single plum in copper coins -/
def plum_cost : ℕ := 4

/-- The number of olives that can be bought for 1 copper coin -/
def olives_per_coin : ℕ := 7

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (solution : FruitSolution) : Prop :=
  let (peaches, plums, olives) := solution
  peaches + plums + olives = total_fruits ∧
  peach_cost * peaches + plum_cost * plums + (olives / olives_per_coin) = total_cost

/-- The correct solution to the problem -/
def correct_solution : FruitSolution := (3, 20, 77)

/-- Theorem stating that the correct_solution is the unique valid solution -/
theorem fruit_problem_solution :
  is_valid_solution correct_solution ∧
  ∀ (other : FruitSolution), is_valid_solution other → other = correct_solution :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_solution_l1209_120963


namespace NUMINAMATH_CALUDE_smallest_fraction_l1209_120990

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) ((x^2+1)/8)))) = 8/(x+2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1209_120990


namespace NUMINAMATH_CALUDE_divisibility_problem_l1209_120922

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (∃ k : ℕ, abc - 1 = k * ((a - 1) * (b - 1) * (c - 1))) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1209_120922


namespace NUMINAMATH_CALUDE_sine_matrix_det_zero_l1209_120902

open Real Matrix

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![sin 1, sin 2, sin 3; 
                                       sin 4, sin 5, sin 6; 
                                       sin 7, sin 8, sin 9]
  det A = 0 := by
sorry

end NUMINAMATH_CALUDE_sine_matrix_det_zero_l1209_120902


namespace NUMINAMATH_CALUDE_jodi_walking_days_l1209_120970

def weekly_distance (days_per_week : ℕ) : ℕ := 
  1 * days_per_week + 2 * days_per_week + 3 * days_per_week + 4 * days_per_week

theorem jodi_walking_days : 
  ∃ (days_per_week : ℕ), weekly_distance days_per_week = 60 ∧ days_per_week = 6 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walking_days_l1209_120970


namespace NUMINAMATH_CALUDE_remainder_problem_l1209_120997

theorem remainder_problem (N : ℤ) : N % 221 = 43 → N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1209_120997


namespace NUMINAMATH_CALUDE_rick_ironing_theorem_l1209_120981

/-- Represents the number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Calculates the total number of pieces of clothing Rick has ironed -/
def total_clothes_ironed : ℕ :=
  shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem :
  total_clothes_ironed = 27 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_theorem_l1209_120981


namespace NUMINAMATH_CALUDE_division_problem_l1209_120984

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1209_120984


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l1209_120954

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l1209_120954


namespace NUMINAMATH_CALUDE_earth_sun_distance_calculation_l1209_120962

/-- The speed of light in vacuum (in m/s) -/
def speed_of_light : ℝ := 3 * 10^8

/-- The time taken for sunlight to reach Earth (in s) -/
def time_to_earth : ℝ := 5 * 10^2

/-- The distance between the Earth and the Sun (in m) -/
def earth_sun_distance : ℝ := 1.5 * 10^11

/-- Theorem stating that the distance between the Earth and the Sun
    is equal to the product of the speed of light and the time taken
    for sunlight to reach Earth -/
theorem earth_sun_distance_calculation :
  earth_sun_distance = speed_of_light * time_to_earth := by
  sorry

end NUMINAMATH_CALUDE_earth_sun_distance_calculation_l1209_120962


namespace NUMINAMATH_CALUDE_equation_solutions_l1209_120940

open Complex

-- Define the set of solutions
def solutions : Set ℂ :=
  {2, -2, 1 + Complex.I * Real.sqrt 3, 1 - Complex.I * Real.sqrt 3,
   -1 + Complex.I * Real.sqrt 3, -1 - Complex.I * Real.sqrt 3}

-- State the theorem
theorem equation_solutions :
  {x : ℂ | x^6 - 64 = 0} = solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1209_120940


namespace NUMINAMATH_CALUDE_function_properties_l1209_120982

-- Define the properties of function f
def additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

def positive_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x > 0

-- State the theorem
theorem function_properties (f : ℝ → ℝ) 
  (h_add : additive f) (h_pos : positive_for_positive f) : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1209_120982


namespace NUMINAMATH_CALUDE_fourth_difference_zero_third_nonzero_l1209_120992

def u (n : ℕ) : ℤ := n^3 + n

def Δ' (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | k + 1 => Δ' (Δ k u)

theorem fourth_difference_zero_third_nonzero :
  (∀ n, Δ 4 u n = 0) ∧ (∃ n, Δ 3 u n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_fourth_difference_zero_third_nonzero_l1209_120992


namespace NUMINAMATH_CALUDE_complex_simplification_l1209_120930

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex fraction in the problem -/
noncomputable def z : ℂ := (2 + 3*i) / (2 - 3*i)

/-- The main theorem -/
theorem complex_simplification : z^8 * 3 = 3 := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1209_120930


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1209_120985

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (3 * x + 6 > x + 8 ∧ x / 4 ≥ (x - 1) / 3) ↔ x ∈ ({2, 3, 4} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1209_120985


namespace NUMINAMATH_CALUDE_base12_2413_mod_9_l1209_120906

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2413
def base12_2413 : List Nat := [2, 4, 1, 3]

-- Theorem statement
theorem base12_2413_mod_9 :
  (base12ToDecimal base12_2413) % 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_base12_2413_mod_9_l1209_120906


namespace NUMINAMATH_CALUDE_harriet_trip_time_l1209_120966

theorem harriet_trip_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 90)
  (h3 : return_speed = 160) :
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed) / outbound_speed * 60 = 192 := by
  sorry

end NUMINAMATH_CALUDE_harriet_trip_time_l1209_120966


namespace NUMINAMATH_CALUDE_store_pricing_strategy_l1209_120978

/-- Calculates the marked price as a percentage of the list price given the purchase discount,
    selling discount, and desired profit percentage. -/
def markedPricePercentage (purchaseDiscount sellingDiscount profitPercentage : ℚ) : ℚ :=
  let costPrice := 1 - purchaseDiscount
  let markupFactor := (1 + profitPercentage) / (1 - sellingDiscount)
  costPrice * markupFactor * 100

/-- Theorem stating that under the given conditions, the marked price should be 121.⅓% of the list price -/
theorem store_pricing_strategy :
  markedPricePercentage (30/100) (25/100) (30/100) = 121 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l1209_120978


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_denominator_l1209_120946

theorem simplify_and_rationalize_denominator :
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_denominator_l1209_120946


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1209_120925

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The length of the major axis of the ellipse is 10.5 -/
theorem ellipse_major_axis_length :
  major_axis_length 3 0.75 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1209_120925


namespace NUMINAMATH_CALUDE_b_le_c_for_geometric_l1209_120993

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Definition of b_n -/
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (n + 1) + a (n + 2)

/-- Definition of c_n -/
def c (a : ℕ → ℝ) (n : ℕ) : ℝ := a n + a (n + 3)

/-- Theorem: For a geometric sequence a, b_n ≤ c_n for all n -/
theorem b_le_c_for_geometric (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, b a n ≤ c a n := by
  sorry

end NUMINAMATH_CALUDE_b_le_c_for_geometric_l1209_120993


namespace NUMINAMATH_CALUDE_scientific_notation_of_10500_l1209_120968

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_10500 :
  toScientificNotation 10500 = ScientificNotation.mk 1.05 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_10500_l1209_120968


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1209_120918

/-- Given a line with equation 2x + 3y = 6, the slope of a perpendicular line is 3/2 -/
theorem perpendicular_slope (x y : ℝ) :
  (2 * x + 3 * y = 6) →
  ∃ m : ℝ, m = 3 / 2 ∧ m * (-2 / 3) = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1209_120918


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1209_120971

/-- Given a quadratic inequality with solution set (x₁, x₂), prove certain properties of the roots -/
theorem quadratic_inequality_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h_sol : ∀ x, a * (x - 1) * (x + 3) + 2 > 0 ↔ x ∈ Set.Ioo x₁ x₂) 
  (h_order : x₁ < x₂) :
  x₁ + x₂ + 2 = 0 ∧ |x₁ - x₂| > 4 ∧ x₁ * x₂ + 3 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1209_120971


namespace NUMINAMATH_CALUDE_points_on_line_l1209_120947

/-- Given a line with equation x = 2y + 5, prove that for any real number n,
    the points (m, n) and (m + 1, n + 0.5) lie on this line, where m = 2n + 5. -/
theorem points_on_line (n : ℝ) : 
  let m : ℝ := 2 * n + 5
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 1, n + 0.5)
  (point1.1 = 2 * point1.2 + 5) ∧ (point2.1 = 2 * point2.2 + 5) :=
by
  sorry


end NUMINAMATH_CALUDE_points_on_line_l1209_120947


namespace NUMINAMATH_CALUDE_simplify_expression_l1209_120916

theorem simplify_expression (x y : ℝ) : 3*x + 4*x + 5*y + 2*y = 7*x + 7*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1209_120916


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l1209_120907

/-- Represents a cube with circular holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_diameter : ℝ

/-- Calculates the total surface area of a cube with circular holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  sorry

/-- Theorem stating the total surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_diameter := 2 }
  total_surface_area cube = 96 + 42 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_surface_area_l1209_120907


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l1209_120955

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ (M : ℝ), M = 10 ∧ ∀ (a b : ℝ), a^2 + b^2 - 2*a + 4*b = 0 → a - 2*b ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l1209_120955


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1209_120951

theorem max_value_sqrt_sum (x : ℝ) (h : -25 ≤ x ∧ x ≤ 25) :
  Real.sqrt (25 + x) + Real.sqrt (25 - x) ≤ 10 ∧
  (Real.sqrt (25 + x) + Real.sqrt (25 - x) = 10 ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1209_120951


namespace NUMINAMATH_CALUDE_board_meeting_arrangement_l1209_120952

/-- The number of ways to arrange 3 indistinguishable objects among 8 positions -/
def arrangement_count : ℕ := 56

/-- The total number of seats -/
def total_seats : ℕ := 10

/-- The number of stools (men) -/
def stool_count : ℕ := 5

/-- The number of rocking chairs (women) -/
def chair_count : ℕ := 5

/-- The number of positions to fill after fixing first and last seats -/
def remaining_positions : ℕ := total_seats - 2

/-- The number of remaining stools to place after fixing first and last seats -/
def remaining_stools : ℕ := stool_count - 2

theorem board_meeting_arrangement :
  arrangement_count = Nat.choose remaining_positions remaining_stools := by
  sorry

end NUMINAMATH_CALUDE_board_meeting_arrangement_l1209_120952


namespace NUMINAMATH_CALUDE_wang_liang_is_president_l1209_120991

-- Define the students and positions
inductive Student : Type
| ZhangQiang : Student
| LiMing : Student
| WangLiang : Student

inductive Position : Type
| President : Position
| LifeDelegate : Position
| StudyDelegate : Position

-- Define the council as a function from Position to Student
def Council := Position → Student

-- Define the predictions
def PredictionA (c : Council) : Prop :=
  c Position.President = Student.ZhangQiang ∧ c Position.LifeDelegate = Student.LiMing

def PredictionB (c : Council) : Prop :=
  c Position.President = Student.WangLiang ∧ c Position.LifeDelegate = Student.ZhangQiang

def PredictionC (c : Council) : Prop :=
  c Position.President = Student.LiMing ∧ c Position.StudyDelegate = Student.ZhangQiang

-- Define the condition that each prediction is half correct
def HalfCorrectPredictions (c : Council) : Prop :=
  (PredictionA c = true) = (PredictionA c = false) ∧
  (PredictionB c = true) = (PredictionB c = false) ∧
  (PredictionC c = true) = (PredictionC c = false)

-- Theorem statement
theorem wang_liang_is_president :
  ∀ c : Council, HalfCorrectPredictions c → c Position.President = Student.WangLiang :=
by
  sorry

end NUMINAMATH_CALUDE_wang_liang_is_president_l1209_120991


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l1209_120923

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles --/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Checks if the triangle satisfies the triangle inequality --/
def Triangle.satisfiesInequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem isosceles_triangle_proof (rope_length : ℝ) 
  (h1 : rope_length = 18) 
  (h2 : ∃ t : Triangle, t.isIsosceles ∧ t.a + t.b + t.c = rope_length ∧ t.a = t.b ∧ t.a = 2 * t.c) :
  ∃ t : Triangle, t.isIsosceles ∧ t.satisfiesInequality ∧ t.a = 36/5 ∧ t.b = 36/5 ∧ t.c = 18/5 ∧
  ∃ t2 : Triangle, t2.isIsosceles ∧ t2.satisfiesInequality ∧ t2.a = 4 ∧ t2.b = 7 ∧ t2.c = 7 ∧
  t2.a + t2.b + t2.c = rope_length :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_proof_l1209_120923


namespace NUMINAMATH_CALUDE_finishing_order_equals_starting_order_l1209_120942

/-- Represents an athlete in the race -/
inductive Athlete : Type
  | Grisha : Athlete
  | Sasha : Athlete
  | Lena : Athlete

/-- Represents the order of athletes -/
def AthleteOrder := List Athlete

/-- The starting order of the race -/
def startingOrder : AthleteOrder := [Athlete.Grisha, Athlete.Sasha, Athlete.Lena]

/-- The number of overtakes by each athlete -/
def overtakes : Athlete → Nat
  | Athlete.Grisha => 10
  | Athlete.Sasha => 4
  | Athlete.Lena => 6

/-- No three athletes were at the same position simultaneously -/
axiom no_triple_overtake : True

/-- All athletes finished at different times -/
axiom different_finish_times : True

/-- The finishing order of the race -/
def finishingOrder : AthleteOrder := sorry

/-- Theorem stating that the finishing order is the same as the starting order -/
theorem finishing_order_equals_starting_order : 
  finishingOrder = startingOrder := by sorry

end NUMINAMATH_CALUDE_finishing_order_equals_starting_order_l1209_120942


namespace NUMINAMATH_CALUDE_josies_calculation_l1209_120953

theorem josies_calculation (a b c d e : ℤ) : 
  a = 2 → b = 1 → c = -1 → d = 3 → 
  (a - b + c^2 - d + e = a - (b - (c^2 - (d + e)))) → e = 0 := by
sorry

end NUMINAMATH_CALUDE_josies_calculation_l1209_120953


namespace NUMINAMATH_CALUDE_triangle_area_from_square_areas_l1209_120974

theorem triangle_area_from_square_areas (a b c : ℝ) (h1 : a^2 = 36) (h2 : b^2 = 64) (h3 : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_areas_l1209_120974


namespace NUMINAMATH_CALUDE_read_book_series_l1209_120996

/-- The number of weeks needed to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - (first_week + second_week)
  2 + (remaining_books + subsequent_weeks - 1) / subsequent_weeks

/-- Proof that it takes 7 weeks to read the book series -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_series_l1209_120996


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1209_120980

theorem opposite_of_negative_fraction :
  -(-(4/5 : ℚ)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1209_120980


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l1209_120998

theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^2 + a*b + c^2 - b*c = 2*a*c) : 
  a = c ∨ a = b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l1209_120998


namespace NUMINAMATH_CALUDE_truncated_cone_height_l1209_120973

/-- The height of a circular truncated cone with given top and bottom surface areas and volume. -/
theorem truncated_cone_height (S₁ S₂ V : ℝ) (h : ℝ) 
    (hS₁ : S₁ = 4 * Real.pi)
    (hS₂ : S₂ = 9 * Real.pi)
    (hV : V = 19 * Real.pi)
    (h_def : V = (1/3) * h * (S₁ + Real.sqrt (S₁ * S₂) + S₂)) :
  h = 3 := by
  sorry

#check truncated_cone_height

end NUMINAMATH_CALUDE_truncated_cone_height_l1209_120973


namespace NUMINAMATH_CALUDE_pentagon_to_squares_area_ratio_l1209_120987

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  side : ℝ

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Calculate the area of a pentagon given its vertices -/
def pentagonArea (a b c d e : Point) : ℝ := sorry

/-- Main theorem: The ratio of the pentagon area to the sum of square areas is 5/12 -/
theorem pentagon_to_squares_area_ratio 
  (squareABCD squareEFGH squareKLMO : Square)
  (a b c d e f g h k l m o : Point) :
  squareABCD.side = 1 →
  squareEFGH.side = 2 →
  squareKLMO.side = 1 →
  b.x = h.x ∧ b.y = e.y → -- AB aligns with HE
  g.x = o.x ∧ m.y = k.y → -- GM aligns with OK
  d.x = (h.x + e.x) / 2 ∧ d.y = h.y → -- D is midpoint of HE
  c.x = h.x + (2/3) * (g.x - h.x) ∧ c.y = h.y → -- C is one-third along HG from H
  (pentagonArea a m k c b) / (squareArea squareABCD + squareArea squareEFGH + squareArea squareKLMO) = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_to_squares_area_ratio_l1209_120987


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1209_120901

/-- 
Given a man's speed with the current and the speed of the current,
this theorem proves the man's speed against the current.
-/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 3) : 
  speed_with_current - 2 * current_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1209_120901


namespace NUMINAMATH_CALUDE_triangle_area_l1209_120944

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area : 
  let area := (1/2) * |a.1 * b.2 - a.2 * b.1|
  area = 9/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1209_120944


namespace NUMINAMATH_CALUDE_bicycle_wheels_l1209_120989

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  num_tricycles = 7 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l1209_120989


namespace NUMINAMATH_CALUDE_button_comparison_l1209_120931

theorem button_comparison (mari_buttons sue_buttons : ℕ) 
  (h1 : mari_buttons = 8)
  (h2 : sue_buttons = 22)
  (h3 : ∃ kendra_buttons : ℕ, sue_buttons = kendra_buttons / 2)
  (h4 : ∃ kendra_buttons : ℕ, kendra_buttons > 5 * mari_buttons) :
  ∃ kendra_buttons : ℕ, kendra_buttons - (5 * mari_buttons) = 4 :=
by sorry

end NUMINAMATH_CALUDE_button_comparison_l1209_120931


namespace NUMINAMATH_CALUDE_min_width_rectangle_l1209_120932

theorem min_width_rectangle (w : ℝ) : w > 0 →
  w * (w + 20) ≥ 150 →
  ∀ x > 0, x * (x + 20) ≥ 150 → w ≤ x →
  w = 10 := by
sorry

end NUMINAMATH_CALUDE_min_width_rectangle_l1209_120932


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l1209_120960

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l1209_120960


namespace NUMINAMATH_CALUDE_sqrt_point_zero_nine_equals_point_three_l1209_120929

theorem sqrt_point_zero_nine_equals_point_three :
  Real.sqrt 0.09 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_zero_nine_equals_point_three_l1209_120929


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1209_120986

def a : ℝ × ℝ := (1, 1)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h : (4 * a.1, 4 * a.2) + b = (4, 2)) : 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1209_120986


namespace NUMINAMATH_CALUDE_total_marks_for_exam_l1209_120958

/-- Calculates the total marks given the number of candidates and average score -/
def totalMarks (numCandidates : ℕ) (averageScore : ℚ) : ℚ :=
  numCandidates * averageScore

/-- Proves that for 250 candidates with an average score of 42, the total marks is 10500 -/
theorem total_marks_for_exam : totalMarks 250 42 = 10500 := by
  sorry

#eval totalMarks 250 42

end NUMINAMATH_CALUDE_total_marks_for_exam_l1209_120958


namespace NUMINAMATH_CALUDE_product_equals_fraction_l1209_120959

/-- The decimal representation of the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := repeating_decimal * 11

/-- Theorem stating that the product of 0.456̄ and 11 is equal to 1672/333 -/
theorem product_equals_fraction : product = 1672 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l1209_120959


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1209_120935

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 30 ≤ 0 ∧ 
  (∀ (m : ℤ), m^2 - 11*m + 30 ≤ 0 → m ≤ n) ∧
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1209_120935


namespace NUMINAMATH_CALUDE_total_hours_worked_l1209_120928

theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) :
  hours_per_day = 3 →
  days_worked = 5 →
  total_hours = hours_per_day * days_worked →
  total_hours = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_hours_worked_l1209_120928


namespace NUMINAMATH_CALUDE_charlottes_age_l1209_120949

theorem charlottes_age (B E C : ℚ) 
  (h1 : B = 4 * C)
  (h2 : E = C + 5)
  (h3 : B = E) :
  C = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_charlottes_age_l1209_120949


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_problem_l1209_120917

theorem consecutive_even_numbers_problem :
  ∀ (x y z : ℕ),
  (y = x + 2) →
  (z = y + 2) →
  (3 * x = 2 * z + 14) →
  z = 26 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_problem_l1209_120917


namespace NUMINAMATH_CALUDE_a_less_than_b_l1209_120988

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l1209_120988


namespace NUMINAMATH_CALUDE_sqrt_sum_max_value_l1209_120948

theorem sqrt_sum_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ max :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_max_value_l1209_120948


namespace NUMINAMATH_CALUDE_energy_after_moving_charge_l1209_120910

/-- The energy stored between two point charges is inversely proportional to their distance -/
axiom energy_inverse_distance {d₁ d₂ : ℝ} {E₁ E₂ : ℝ} (h : d₁ > 0 ∧ d₂ > 0) :
  E₁ / E₂ = d₂ / d₁

/-- The total energy of four point charges at the corners of a square -/
def initial_energy : ℝ := 20

/-- The number of energy pairs in the initial square configuration -/
def initial_pairs : ℕ := 6

theorem energy_after_moving_charge (d : ℝ) (h : d > 0) :
  let initial_pair_energy := initial_energy / initial_pairs
  let center_to_corner_distance := d / Real.sqrt 2
  let new_center_pair_energy := initial_pair_energy * d / center_to_corner_distance
  3 * new_center_pair_energy + 3 * initial_pair_energy = 10 * Real.sqrt 2 + 10 := by
sorry

end NUMINAMATH_CALUDE_energy_after_moving_charge_l1209_120910


namespace NUMINAMATH_CALUDE_range_of_a_for_circle_condition_l1209_120967

/-- The range of 'a' for which there exists a point M on the circle (x-a)^2 + (y-a+2)^2 = 1
    such that |MA| = 2|MO|, where A is (0, -3) and O is the origin. -/
theorem range_of_a_for_circle_condition (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ 
    (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) ↔ 
  0 ≤ a ∧ a ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_for_circle_condition_l1209_120967


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_147_l1209_120905

theorem greatest_prime_factor_of_147 : ∃ p : ℕ, p = 7 ∧ Nat.Prime p ∧ p ∣ 147 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 147 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_147_l1209_120905
