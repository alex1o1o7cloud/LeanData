import Mathlib

namespace NUMINAMATH_CALUDE_water_volume_in_first_solution_l3987_398708

/-- The cost per liter of a spirit-water solution is directly proportional to the fraction of spirit by volume. -/
axiom cost_proportional_to_spirit_fraction (cost spirit_vol total_vol : ℝ) : 
  cost = (spirit_vol / total_vol) * (cost * total_vol / spirit_vol)

/-- The cost of the first solution with 1 liter of spirit and an unknown amount of water -/
def first_solution_cost : ℝ := 0.50

/-- The cost of the second solution with 1 liter of spirit and 2 liters of water -/
def second_solution_cost : ℝ := 0.50

/-- The volume of spirit in both solutions -/
def spirit_volume : ℝ := 1

/-- The volume of water in the second solution -/
def second_solution_water_volume : ℝ := 2

/-- The volume of water in the first solution -/
def first_solution_water_volume : ℝ := 2

theorem water_volume_in_first_solution : 
  first_solution_water_volume = 2 := by sorry

end NUMINAMATH_CALUDE_water_volume_in_first_solution_l3987_398708


namespace NUMINAMATH_CALUDE_hallie_paintings_sold_l3987_398767

/-- The number of paintings Hallie sold -/
def paintings_sold (prize : ℕ) (painting_price : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - prize) / painting_price

theorem hallie_paintings_sold :
  paintings_sold 150 50 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hallie_paintings_sold_l3987_398767


namespace NUMINAMATH_CALUDE_contradiction_assumption_l3987_398716

theorem contradiction_assumption (a b c d : ℝ) :
  (¬ (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0)) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l3987_398716


namespace NUMINAMATH_CALUDE_no_valid_replacements_l3987_398707

theorem no_valid_replacements :
  ∀ z : ℕ, z < 10 → ¬(35000 + 100 * z + 45) % 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_valid_replacements_l3987_398707


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3987_398723

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 4 / y) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3987_398723


namespace NUMINAMATH_CALUDE_seven_people_round_table_l3987_398794

/-- The number of distinct arrangements of n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: There are 720 distinct ways to arrange 7 people around a round table. -/
theorem seven_people_round_table : roundTableArrangements 7 = 720 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l3987_398794


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3987_398755

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of the coefficients is 1 -/
def coefficientsAreCoprime (coeff : PlaneCoefficients) : Prop :=
  Nat.gcd (Int.natAbs coeff.A) (Nat.gcd (Int.natAbs coeff.B) (Nat.gcd (Int.natAbs coeff.C) (Int.natAbs coeff.D))) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -1, 3⟩ →
  p2 = ⟨0, -1, 5⟩ →
  p3 = ⟨-1, -3, 4⟩ →
  coeff = ⟨1, 2, -1, 3⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  coefficientsAreCoprime coeff :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3987_398755


namespace NUMINAMATH_CALUDE_function_identity_l3987_398750

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3987_398750


namespace NUMINAMATH_CALUDE_probability_at_least_one_pen_l3987_398781

theorem probability_at_least_one_pen
  (p_ball : ℝ)
  (p_ink : ℝ)
  (h_ball : p_ball = 3 / 5)
  (h_ink : p_ink = 2 / 3)
  (h_nonneg_ball : 0 ≤ p_ball)
  (h_nonneg_ink : 0 ≤ p_ink)
  (h_le_one_ball : p_ball ≤ 1)
  (h_le_one_ink : p_ink ≤ 1)
  (h_independent : True)  -- Assumption of independence
  : p_ball + p_ink - p_ball * p_ink = 13 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_pen_l3987_398781


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3987_398735

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the shorter leg -/
  a : ℝ
  /-- The length of the longer leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Ensure a is the shorter leg -/
  h_a_le_b : a ≤ b
  /-- Pythagorean theorem -/
  h_pythagorean : a^2 + b^2 = c^2
  /-- Formula for the radius of the inscribed circle -/
  h_radius : r = (a + b - c) / 2
  /-- Positivity conditions -/
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_c_pos : c > 0
  h_r_pos : r > 0

/-- The main theorem: the radius of the inscribed circle is less than one-third of the longer leg -/
theorem inscribed_circle_radius_bound (t : RightTriangleWithInscribedCircle) : t.r < t.b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3987_398735


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l3987_398727

/-- A regular star polygon with n points -/
structure RegularStarPolygon where
  n : ℕ
  edges : Fin (2 * n) → ℝ
  angles_A : Fin n → ℝ
  angles_B : Fin n → ℝ
  edges_equal : ∀ i j, edges i = edges j
  angles_A_equal : ∀ i j, angles_A i = angles_A j
  angles_B_equal : ∀ i j, angles_B i = angles_B j
  angle_difference : ∀ i, angles_B i - angles_A i = 15

/-- The theorem stating that for a regular star polygon with the given conditions, n must be 24 -/
theorem regular_star_polygon_points (star : RegularStarPolygon) :
  (∀ i, star.angles_B i - star.angles_A i = 15) → star.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l3987_398727


namespace NUMINAMATH_CALUDE_expression_simplification_l3987_398703

theorem expression_simplification : 
  let f (x : ℤ) := x^4 + 324
  ((f 12) * (f 26) * (f 38) * (f 50) * (f 62)) / 
  ((f 6) * (f 18) * (f 30) * (f 42) * (f 54)) = 3968 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3987_398703


namespace NUMINAMATH_CALUDE_base_five_representation_of_156_l3987_398709

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Checks if a list of digits represents a valid base 5 number --/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5)

theorem base_five_representation_of_156 :
  let base5Repr := toBaseFive 156
  isValidBaseFive base5Repr ∧ base5Repr = [1, 1, 1, 1] := by
  sorry

#eval toBaseFive 156  -- Should output [1, 1, 1, 1]

end NUMINAMATH_CALUDE_base_five_representation_of_156_l3987_398709


namespace NUMINAMATH_CALUDE_factor_expression_l3987_398705

theorem factor_expression (x : ℝ) : 
  (4 * x^4 + 128 * x^3 - 9) - (-6 * x^4 + 2 * x^3 - 9) = 2 * x^3 * (5 * x + 63) := by
sorry

end NUMINAMATH_CALUDE_factor_expression_l3987_398705


namespace NUMINAMATH_CALUDE_intersection_of_AB_and_CD_l3987_398732

def A : ℝ × ℝ × ℝ := (2, -1, 2)
def B : ℝ × ℝ × ℝ := (12, -11, 7)
def C : ℝ × ℝ × ℝ := (1, 4, -7)
def D : ℝ × ℝ × ℝ := (4, -2, 13)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_of_AB_and_CD :
  line_intersection A B C D = (8/3, -7/3, 7/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_AB_and_CD_l3987_398732


namespace NUMINAMATH_CALUDE_simplify_expression_l3987_398737

theorem simplify_expression (a b c : ℝ) 
  (h : Real.sqrt (a - 5) + (b - 3)^2 = Real.sqrt (c - 4) + Real.sqrt (4 - c)) :
  Real.sqrt c / (Real.sqrt a - Real.sqrt b) = Real.sqrt 5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3987_398737


namespace NUMINAMATH_CALUDE_nell_initial_cards_l3987_398738

/-- The number of cards Nell gave away -/
def cards_given_away : ℕ := 276

/-- The number of cards Nell has left -/
def cards_left : ℕ := 252

/-- Nell's initial number of cards -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem nell_initial_cards : initial_cards = 528 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l3987_398738


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3987_398785

theorem no_integer_solutions : ¬∃ (a b : ℤ), a^3 + 3*a^2 + 2*a = 125*b^3 + 75*b^2 + 15*b + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3987_398785


namespace NUMINAMATH_CALUDE_juanita_dessert_cost_l3987_398740

/-- Calculates the cost of Juanita's dessert given the prices of individual items --/
def dessert_cost (brownie_price ice_cream_price syrup_price nuts_price : ℚ) : ℚ :=
  brownie_price + 2 * ice_cream_price + 2 * syrup_price + nuts_price

/-- Proves that Juanita's dessert costs $7.00 given the prices of individual items --/
theorem juanita_dessert_cost :
  dessert_cost 2.5 1 0.5 1.5 = 7 := by
  sorry

#eval dessert_cost 2.5 1 0.5 1.5

end NUMINAMATH_CALUDE_juanita_dessert_cost_l3987_398740


namespace NUMINAMATH_CALUDE_johns_initial_speed_johns_initial_speed_proof_l3987_398792

theorem johns_initial_speed 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) : ℝ :=
  let final_time := initial_time * (1 + time_increase_percent / 100)
  let initial_speed := (final_distance / final_time) - speed_increase
  initial_speed

#check johns_initial_speed 8 75 4 168 = 8

theorem johns_initial_speed_proof 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) :
  johns_initial_speed initial_time time_increase_percent speed_increase final_distance = 8 :=
by sorry

end NUMINAMATH_CALUDE_johns_initial_speed_johns_initial_speed_proof_l3987_398792


namespace NUMINAMATH_CALUDE_problem_statement_l3987_398736

theorem problem_statement (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = -8) :
  y * Real.sqrt (x / y) + x * Real.sqrt (y / x) = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3987_398736


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3987_398770

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 := by
sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3987_398770


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3987_398733

def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 48 * x - y^2 + 6 * y + 50 = 0

def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_equation = 2 * Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3987_398733


namespace NUMINAMATH_CALUDE_exponent_sum_l3987_398793

theorem exponent_sum (a x y : ℝ) (hx : a^x = 2) (hy : a^y = 3) : a^(x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3987_398793


namespace NUMINAMATH_CALUDE_hiker_distance_l3987_398730

theorem hiker_distance (s t : ℝ) 
  (h1 : (s + 1) * (2/3 * t) = s * t) 
  (h2 : (s - 1) * (t + 3) = s * t) : 
  s * t = 6 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l3987_398730


namespace NUMINAMATH_CALUDE_triangle_side_length_bound_l3987_398783

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area S = √3/4 * (a² + c² - b²) and b = √3,
    prove that (√3 - 1)a + 2c is bounded by (3 - √3, 2√6]. -/
theorem triangle_side_length_bound (a c : ℝ) (h_positive : a > 0 ∧ c > 0) :
  let b := Real.sqrt 3
  let S := Real.sqrt 3 / 4 * (a^2 + c^2 - b^2)
  3 - Real.sqrt 3 < (Real.sqrt 3 - 1) * a + 2 * c ∧
  (Real.sqrt 3 - 1) * a + 2 * c ≤ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_bound_l3987_398783


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l3987_398789

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 48)
  (h5 : x*y + y*z + z*x = 26) : 
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l3987_398789


namespace NUMINAMATH_CALUDE_three_of_a_kind_probability_l3987_398729

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a hand -/
def HandSize : ℕ := 5

/-- Represents the number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a "three of a kind" hand with two other cards of different ranks -/
def probThreeOfAKind : ℚ :=
  let totalHands := Nat.choose StandardDeck HandSize
  let threeOfAKindHands := NumRanks * Nat.choose CardsPerRank 3 * (NumRanks - 1) * CardsPerRank * (NumRanks - 2) * CardsPerRank
  threeOfAKindHands / totalHands

theorem three_of_a_kind_probability : probThreeOfAKind = 1719 / 40921 := by
  sorry

end NUMINAMATH_CALUDE_three_of_a_kind_probability_l3987_398729


namespace NUMINAMATH_CALUDE_constant_sum_property_l3987_398706

/-- Represents a triangle with numbers at its vertices -/
structure NumberedTriangle where
  a : ℝ  -- Number at vertex A
  b : ℝ  -- Number at vertex B
  c : ℝ  -- Number at vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.a + (t.b + t.c) = t.b + (t.c + t.a) ∧
  t.b + (t.c + t.a) = t.c + (t.a + t.b) ∧
  t.c + (t.a + t.b) = t.a + t.b + t.c :=
sorry

end NUMINAMATH_CALUDE_constant_sum_property_l3987_398706


namespace NUMINAMATH_CALUDE_find_c_l3987_398720

theorem find_c (a b : ℕ) (h_prime_a : Nat.Prime a) (h_prime_b : Nat.Prime b) :
  (∃ k : ℕ, (10^k ≤ a) ∧ (a < 10^(k+1)) ∧ (10^k ≤ b) ∧ (b < 10^(k+1))) →
  (∃ c : ℕ, c = 10^k * a + b ∧ c - a * b = 154) →
  (∃ c : ℕ, c = 1997) :=
sorry

end NUMINAMATH_CALUDE_find_c_l3987_398720


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3987_398746

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence : 
  let a₁ := 8
  let a₂ := 5
  let a₃ := 2
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = -79 := by
sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3987_398746


namespace NUMINAMATH_CALUDE_minuend_is_zero_l3987_398754

theorem minuend_is_zero (x y : ℝ) (h : x - y = -y) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_minuend_is_zero_l3987_398754


namespace NUMINAMATH_CALUDE_x_greater_than_e_l3987_398712

theorem x_greater_than_e (x : ℝ) (h1 : Real.log x > 0) (h2 : x > 1) : x > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_e_l3987_398712


namespace NUMINAMATH_CALUDE_square_sum_implies_product_zero_l3987_398780

theorem square_sum_implies_product_zero (n : ℝ) :
  (n - 2022)^2 + (2023 - n)^2 = 1 → (2022 - n) * (n - 2023) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_zero_l3987_398780


namespace NUMINAMATH_CALUDE_min_orchard_space_l3987_398779

/-- The space required for planting trees in an orchard. -/
def orchard_space (apple apricot plum : ℕ) : ℕ :=
  apple^2 + 5*apricot + plum^3

/-- The minimum space required for planting 10 trees, including at least one of each type. -/
theorem min_orchard_space :
  ∃ (apple apricot plum : ℕ),
    apple + apricot + plum = 10 ∧
    apple ≥ 1 ∧ apricot ≥ 1 ∧ plum ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 10 →
      a ≥ 1 → b ≥ 1 → c ≥ 1 →
      orchard_space apple apricot plum ≤ orchard_space a b c ∧
      orchard_space apple apricot plum = 37 :=
by sorry

end NUMINAMATH_CALUDE_min_orchard_space_l3987_398779


namespace NUMINAMATH_CALUDE_Bob_is_shortest_l3987_398742

-- Define a type for the friends
inductive Friend
| Amy
| Bob
| Carla
| Dan
| Eric

-- Define a relation for "taller than"
def taller_than : Friend → Friend → Prop :=
  sorry

-- State the theorem
theorem Bob_is_shortest (h1 : taller_than Friend.Amy Friend.Carla)
                        (h2 : taller_than Friend.Eric Friend.Dan)
                        (h3 : taller_than Friend.Dan Friend.Bob)
                        (h4 : taller_than Friend.Carla Friend.Eric) :
  ∀ f : Friend, f ≠ Friend.Bob → taller_than f Friend.Bob :=
by sorry

end NUMINAMATH_CALUDE_Bob_is_shortest_l3987_398742


namespace NUMINAMATH_CALUDE_mrs_hilt_total_chapters_l3987_398722

/-- The total number of chapters Mrs. Hilt has read -/
def total_chapters_read : ℕ :=
  let last_month_17ch := 4 * 17
  let last_month_25ch := 3 * 25
  let last_month_30ch := 2 * 30
  let this_month_book1 := 18
  let this_month_book2 := 24
  last_month_17ch + last_month_25ch + last_month_30ch + this_month_book1 + this_month_book2

/-- Theorem stating that Mrs. Hilt has read 245 chapters in total -/
theorem mrs_hilt_total_chapters : total_chapters_read = 245 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_chapters_l3987_398722


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3987_398775

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 15 = 0 ↔ x = (a : ℂ) + b * I ∨ x = (a : ℂ) - b * I) →
  a + b^2 = 79/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3987_398775


namespace NUMINAMATH_CALUDE_line_point_k_value_l3987_398796

/-- A line contains the points (3, 5), (-3, k), and (-9, -2). The value of k is 3/2. -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 5 = m * 3 + b ∧ k = m * (-3) + b ∧ -2 = m * (-9) + b) → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3987_398796


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l3987_398798

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 30 →
  side = 10 →
  median = 9 →
  area = (1/2) * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l3987_398798


namespace NUMINAMATH_CALUDE_f_zero_range_l3987_398758

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_range (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_range_l3987_398758


namespace NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l3987_398749

structure Container where
  red_balls : ℕ
  green_balls : ℕ

def total_balls (c : Container) : ℕ := c.red_balls + c.green_balls

def prob_green (c : Container) : ℚ :=
  c.green_balls / (total_balls c)

def containers : List Container := [
  ⟨8, 4⟩,  -- Container I
  ⟨2, 4⟩,  -- Container II
  ⟨2, 4⟩   -- Container III
]

theorem prob_green_ball_is_five_ninths :
  (containers.map prob_green).sum / containers.length = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l3987_398749


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l3987_398782

theorem common_factor_of_polynomial (m a b : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3*m*a^2 - 6*m*a*b = m * (k₁*a^2 + k₂*a*b) :=
sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l3987_398782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3987_398719

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3987_398719


namespace NUMINAMATH_CALUDE_minimum_additional_games_l3987_398728

def initial_games : ℕ := 3
def initial_wins : ℕ := 2
def target_percentage : ℚ := 9/10

def winning_percentage (additional_games : ℕ) : ℚ :=
  (initial_wins + additional_games) / (initial_games + additional_games)

theorem minimum_additional_games :
  ∃ N : ℕ, (∀ n : ℕ, n < N → winning_percentage n < target_percentage) ∧
            winning_percentage N ≥ target_percentage ∧
            N = 7 :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_games_l3987_398728


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l3987_398721

theorem infinitely_many_pairs_divisibility :
  ∀ k : ℕ, ∃ n m : ℕ, (n + m)^2 / (n + 7) = k :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l3987_398721


namespace NUMINAMATH_CALUDE_triangle_shape_l3987_398777

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_shape (t : Triangle) (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  (t.A = t.B) ∨ (t.A + t.B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3987_398777


namespace NUMINAMATH_CALUDE_triangle_max_area_l3987_398757

/-- Given a triangle ABC where c = 2 and b = √2 * a, 
    the maximum area of the triangle is 2√2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : c = 2) (h2 : b = Real.sqrt 2 * a) :
  ∃ (S : ℝ), S = (Real.sqrt 2 : ℝ) * 2 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2)/(2*a*b))) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3987_398757


namespace NUMINAMATH_CALUDE_james_weight_vest_savings_l3987_398788

/-- The amount James saves by assembling his own weight vest -/
theorem james_weight_vest_savings : 
  let weight_vest_cost : ℝ := 250
  let weight_plates_pounds : ℝ := 200
  let weight_plates_cost_per_pound : ℝ := 1.2
  let ready_made_vest_cost : ℝ := 700
  let ready_made_vest_discount : ℝ := 100
  
  let james_vest_cost := weight_vest_cost + weight_plates_pounds * weight_plates_cost_per_pound
  let discounted_ready_made_vest_cost := ready_made_vest_cost - ready_made_vest_discount
  
  discounted_ready_made_vest_cost - james_vest_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_savings_l3987_398788


namespace NUMINAMATH_CALUDE_chessboard_ratio_sum_l3987_398710

/-- The number of rectangles formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_rectangles : ℕ := 1296

/-- The number of squares formed on an 8x8 chessboard with 9 horizontal and 9 vertical lines -/
def total_squares : ℕ := 204

/-- The ratio of squares to rectangles as a simplified fraction -/
def square_rectangle_ratio : ℚ := total_squares / total_rectangles

theorem chessboard_ratio_sum :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ square_rectangle_ratio = m / n ∧ m + n = 125 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_ratio_sum_l3987_398710


namespace NUMINAMATH_CALUDE_league_teams_count_l3987_398771

theorem league_teams_count (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_l3987_398771


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3987_398799

theorem max_value_quadratic (r : ℝ) : 
  -3 * r^2 + 30 * r + 8 ≤ 83 ∧ ∃ r : ℝ, -3 * r^2 + 30 * r + 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3987_398799


namespace NUMINAMATH_CALUDE_special_point_is_zero_l3987_398726

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 14 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

/-- The theorem stating that (5/19, 16/19) is a zero of all polynomials p satisfying the given conditions -/
theorem special_point_is_zero (b : Fin 14 → ℝ) : 
  (p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ p b 0 1 = 0 ∧ 
   p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ p b (-1) (-1) = 0 ∧ 
   p b 2 2 = 0 ∧ p b 2 (-2) = 0 ∧ p b (-2) 2 = 0) → 
  p b (5/19) (16/19) = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_point_is_zero_l3987_398726


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3987_398718

theorem rectangle_dimensions (w : ℝ) (h : w > 0) :
  let l := 2 * w
  let area := w * l
  let perimeter := 2 * (w + l)
  area = 2 * perimeter → w = 6 ∧ l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3987_398718


namespace NUMINAMATH_CALUDE_product_and_closest_value_l3987_398747

def calculate_product : ℝ := 2.5 * (53.6 - 0.4)

def options : List ℝ := [120, 130, 133, 140, 150]

theorem product_and_closest_value :
  calculate_product = 133 ∧
  ∀ x ∈ options, |calculate_product - 133| ≤ |calculate_product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_and_closest_value_l3987_398747


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3987_398786

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ ∀ y : ℝ, y ∈ Set.Ioo (1/2) 1 → (deriv (f 1)) t = 0 ∧ (deriv (f 1)) y = 0 → y = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3987_398786


namespace NUMINAMATH_CALUDE_lines_skew_and_parallel_l3987_398762

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_skew_and_parallel (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : skew c b := by
  sorry

end NUMINAMATH_CALUDE_lines_skew_and_parallel_l3987_398762


namespace NUMINAMATH_CALUDE_l_plaque_four_equal_parts_l3987_398769

/-- An L-shaped plaque -/
structure LPlaque where
  width : ℝ
  height : ℝ
  thickness : ℝ

/-- A straight cut on the plaque -/
inductive Cut
  | Vertical (x : ℝ)
  | Horizontal (y : ℝ)

/-- The result of applying cuts to an L-shaped plaque -/
def applyCuts (p : LPlaque) (cuts : List Cut) : List (Set (ℝ × ℝ)) :=
  sorry

/-- Check if all pieces have equal area -/
def equalAreas (pieces : List (Set (ℝ × ℝ))) : Prop :=
  sorry

/-- Main theorem: An L-shaped plaque can be divided into four equal parts using straight cuts -/
theorem l_plaque_four_equal_parts (p : LPlaque) :
  ∃ (cuts : List Cut), (applyCuts p cuts).length = 4 ∧ equalAreas (applyCuts p cuts) :=
sorry

end NUMINAMATH_CALUDE_l_plaque_four_equal_parts_l3987_398769


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3987_398714

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 1) = 0 ↔ y = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3987_398714


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l3987_398763

theorem unique_solution_quadratic_system (x : ℚ) :
  (6 * x^2 + 19 * x - 7 = 0) ∧ (18 * x^2 + 47 * x - 21 = 0) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l3987_398763


namespace NUMINAMATH_CALUDE_final_tile_difference_l3987_398764

/-- Represents the number of tiles in the figure -/
structure TileCount where
  blue : ℕ
  red : ℕ

/-- Calculates the difference between red and blue tiles after adding a border -/
def tileDifference (initial : TileCount) (borderTiles : ℕ) : ℤ :=
  (initial.red + borderTiles : ℤ) - initial.blue

/-- Theorem stating the difference between red and blue tiles after adding the border -/
theorem final_tile_difference (initial : TileCount) (borderTiles : ℕ) :
    initial.blue = 17 → initial.red = 8 → borderTiles = 24 →
    tileDifference initial borderTiles = 15 := by
  sorry

end NUMINAMATH_CALUDE_final_tile_difference_l3987_398764


namespace NUMINAMATH_CALUDE_loan_duration_to_c_l3987_398773

/-- Proves that the number of years A lent money to C is 4, given the specified conditions. -/
theorem loan_duration_to_c (principal_b principal_c total_interest : ℚ) 
  (duration_b : ℚ) (rate : ℚ) : 
  principal_b = 5000 →
  principal_c = 3000 →
  duration_b = 2 →
  rate = 7.000000000000001 / 100 →
  total_interest = 1540 →
  total_interest = principal_b * rate * duration_b + principal_c * rate * (4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_to_c_l3987_398773


namespace NUMINAMATH_CALUDE_vector_angle_difference_l3987_398715

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (h5 : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h6 : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_difference_l3987_398715


namespace NUMINAMATH_CALUDE_opposite_roots_n_value_l3987_398725

/-- Given a rational function equal to (n-2)/(n+2) with roots of opposite signs, prove n = 2b + 2 -/
theorem opposite_roots_n_value (b d p q n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - b*x + d) / (p*x - q) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  p = b + 1 →
  n = 2*b + 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_n_value_l3987_398725


namespace NUMINAMATH_CALUDE_odd_function_value_l3987_398731

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = x * (x - 1)) :
  f (-3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3987_398731


namespace NUMINAMATH_CALUDE_twenty_one_in_fibonacci_l3987_398791

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem twenty_one_in_fibonacci : ∃ n : ℕ, fibonacci n = 21 := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_in_fibonacci_l3987_398791


namespace NUMINAMATH_CALUDE_largest_green_socks_l3987_398756

theorem largest_green_socks (g y : ℕ) :
  let t := g + y
  (t ≤ 2023) →
  ((g * (g - 1) + y * (y - 1)) / (t * (t - 1)) = 1/3) →
  g ≤ 990 ∧ ∃ (g' y' : ℕ), g' = 990 ∧ y' + g' ≤ 2023 ∧
    ((g' * (g' - 1) + y' * (y' - 1)) / ((g' + y') * (g' + y' - 1)) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_largest_green_socks_l3987_398756


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l3987_398704

/-- The number of ways to arrange 4 passengers in 10 seats with exactly 5 consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- The number of seats in the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial num_passengers) * 
    (Nat.factorial 5 / (Nat.factorial 3)) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l3987_398704


namespace NUMINAMATH_CALUDE_loss_of_30_notation_l3987_398724

def profit_notation (amount : ℤ) : ℤ := amount

def loss_notation (amount : ℤ) : ℤ := -amount

theorem loss_of_30_notation :
  profit_notation 20 = 20 →
  loss_notation 30 = -30 :=
by
  sorry

end NUMINAMATH_CALUDE_loss_of_30_notation_l3987_398724


namespace NUMINAMATH_CALUDE_point_above_line_t_range_l3987_398795

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define what it means for a point to be above the line
def above_line (x y : ℝ) : Prop := x - 2*y + 4 < 0

-- Theorem statement
theorem point_above_line_t_range :
  ∀ t : ℝ, above_line (-2) t → t > 1 :=
by sorry

end NUMINAMATH_CALUDE_point_above_line_t_range_l3987_398795


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3987_398768

theorem inscribed_circle_area_ratio (a : ℝ) (ha : a > 0) :
  let square_area := a^2
  let circle_radius := a / 2
  let circle_area := π * circle_radius^2
  circle_area / square_area = π / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3987_398768


namespace NUMINAMATH_CALUDE_prime_power_congruence_l3987_398797

theorem prime_power_congruence (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  (p^(p+2) + (p+2)^p) % (2*p+2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_congruence_l3987_398797


namespace NUMINAMATH_CALUDE_triangles_in_regular_decagon_l3987_398766

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_in_decagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

theorem triangles_in_regular_decagon : 
  triangles_in_decagon = Nat.choose decagon_vertices triangle_vertices := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_regular_decagon_l3987_398766


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l3987_398784

theorem divisible_by_nine_sequence (start : ℕ) (h1 : start ≥ 32) (h2 : start % 9 = 0) : 
  let sequence := List.range 7
  let last_number := start + 9 * 6
  last_number = 90 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l3987_398784


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l3987_398748

/-- A line passing through two points (8, 2) and (4, 6) intersects the x-axis at (10, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (8, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = 10
:= by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l3987_398748


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3987_398744

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculate the area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  ∃ ε > 0, |trapezoid_area t - 1242.425| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3987_398744


namespace NUMINAMATH_CALUDE_tourists_eq_scientific_l3987_398711

/-- Represents the number of domestic tourists during the "May Day" holiday in 2023 (in millions) -/
def tourists : ℝ := 274

/-- Represents the scientific notation of the number of tourists -/
def tourists_scientific : ℝ := 2.74 * (10 ^ 8)

/-- Theorem stating that the number of tourists in millions is equal to its scientific notation representation -/
theorem tourists_eq_scientific : tourists * (10 ^ 6) = tourists_scientific := by sorry

end NUMINAMATH_CALUDE_tourists_eq_scientific_l3987_398711


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_optimal_l3987_398741

-- Define the quadratic function
def f (x : ℝ) := x^2 - 4*x

-- State the theorem
theorem max_m_value :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ m) → m ≤ -3 :=
by sorry

-- Define the maximum value of m
def max_m : ℝ := -3

-- Prove that this is indeed the maximum value
theorem max_m_is_optimal :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ max_m) ∧
  ∀ ε > 0, ∃ x ∈ Set.Ioo 0 1, f x < max_m + ε :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_optimal_l3987_398741


namespace NUMINAMATH_CALUDE_cottage_configuration_exists_l3987_398774

/-- A configuration of points on a circle -/
def Configuration := List ℕ

/-- The sum of elements in a list -/
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

/-- Check if all elements in a list are unique -/
def all_unique (l : List ℕ) : Prop := l.Nodup

/-- Generate all distances between points in a circular configuration -/
def generate_distances (config : Configuration) : List ℕ :=
  let n := config.length
  let total := list_sum config
  List.range n >>= fun i =>
    List.range n >>= fun j =>
      if i < j then
        let dist := (list_sum (config.take j) - list_sum (config.take i) + total) % total
        [min dist (total - dist)]
      else
        []

/-- The main theorem statement -/
theorem cottage_configuration_exists : ∃ (config : Configuration),
  (config.length = 6) ∧
  (list_sum config = 27) ∧
  (all_unique (generate_distances config)) ∧
  (∀ d, d ∈ generate_distances config → d ≥ 1 ∧ d ≤ 26) :=
sorry

end NUMINAMATH_CALUDE_cottage_configuration_exists_l3987_398774


namespace NUMINAMATH_CALUDE_negation_equivalence_l3987_398713

universe u

-- Define the universe of discourse
variable {Person : Type u}

-- Define predicates
variable (Teacher : Person → Prop)
variable (ExcellentInMath : Person → Prop)
variable (PoorInMath : Person → Prop)

-- Define the theorem
theorem negation_equivalence :
  (∃ x, Teacher x ∧ PoorInMath x) ↔ ¬(∀ x, Teacher x → ExcellentInMath x) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3987_398713


namespace NUMINAMATH_CALUDE_theatre_distance_is_340_l3987_398761

/-- Represents the problem of Julia's drive to the theatre. -/
structure JuliaDrive where
  initial_speed : ℝ
  speed_increase : ℝ
  initial_time : ℝ
  late_time : ℝ
  early_time : ℝ

/-- Calculates the total distance to the theatre based on the given conditions. -/
def calculate_distance (drive : JuliaDrive) : ℝ :=
  let total_time := drive.initial_time + (drive.late_time + drive.early_time)
  let remaining_time := total_time - drive.initial_time
  let remaining_distance := (drive.initial_speed + drive.speed_increase) * remaining_time
  drive.initial_speed * drive.initial_time + remaining_distance

/-- Theorem stating that the distance to the theatre is 340 miles. -/
theorem theatre_distance_is_340 (drive : JuliaDrive)
  (h1 : drive.initial_speed = 40)
  (h2 : drive.speed_increase = 20)
  (h3 : drive.initial_time = 1)
  (h4 : drive.late_time = 1.5)
  (h5 : drive.early_time = 1) :
  calculate_distance drive = 340 := by
  sorry

end NUMINAMATH_CALUDE_theatre_distance_is_340_l3987_398761


namespace NUMINAMATH_CALUDE_boat_speed_l3987_398753

/-- Proves that the speed of a boat in still water is 30 kmph given specific conditions -/
theorem boat_speed (x : ℝ) (h1 : x > 0) : 
  (∃ t : ℝ, t > 0 ∧ 80 = (x + 10) * t ∧ 40 = (x - 10) * t) → x = 30 := by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l3987_398753


namespace NUMINAMATH_CALUDE_max_segment_for_quadrilateral_l3987_398760

theorem max_segment_for_quadrilateral
  (a b c d : ℝ)
  (total_length : a + b + c + d = 2)
  (ordered_segments : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (∃ (x : ℝ), x < 1 ∧
    (∀ (y : ℝ), y < x →
      (a + b > y ∧ a + c > y ∧ a + d > y ∧
       b + c > y ∧ b + d > y ∧ c + d > y))) ∧
  (∀ (z : ℝ), z ≥ 1 →
    ¬(a + b > z ∧ a + c > z ∧ a + d > z ∧
      b + c > z ∧ b + d > z ∧ c + d > z)) :=
by sorry

end NUMINAMATH_CALUDE_max_segment_for_quadrilateral_l3987_398760


namespace NUMINAMATH_CALUDE_function_equality_l3987_398700

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3987_398700


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l3987_398776

theorem modulo_eleven_residue : (178 + 4 * 28 + 8 * 62 + 3 * 21) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l3987_398776


namespace NUMINAMATH_CALUDE_book_price_percentage_l3987_398752

theorem book_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
sorry

end NUMINAMATH_CALUDE_book_price_percentage_l3987_398752


namespace NUMINAMATH_CALUDE_adult_average_age_l3987_398765

theorem adult_average_age
  (total_members : ℕ)
  (total_average_age : ℚ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (girls_average_age : ℚ)
  (boys_average_age : ℚ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_adults = 5)
  (h6 : girls_average_age = 16)
  (h7 : boys_average_age = 17)
  (h8 : total_members = num_girls + num_boys + num_adults) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_adults = 32 := by
  sorry

end NUMINAMATH_CALUDE_adult_average_age_l3987_398765


namespace NUMINAMATH_CALUDE_copper_weights_problem_l3987_398790

theorem copper_weights_problem :
  ∃ (x y z u : ℕ+),
    (x : ℤ) + y + z + u = 40 ∧
    ∀ W : ℤ, 1 ≤ W ∧ W ≤ 40 →
      ∃ (a b c d : ℤ),
        (a = -1 ∨ a = 0 ∨ a = 1) ∧
        (b = -1 ∨ b = 0 ∨ b = 1) ∧
        (c = -1 ∨ c = 0 ∨ c = 1) ∧
        (d = -1 ∨ d = 0 ∨ d = 1) ∧
        W = a * x + b * y + c * z + d * u :=
by sorry

end NUMINAMATH_CALUDE_copper_weights_problem_l3987_398790


namespace NUMINAMATH_CALUDE_quadrilateral_circumscribed_circle_l3987_398701

/-- The quadrilateral formed by four lines has a circumscribed circle -/
theorem quadrilateral_circumscribed_circle 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l₄ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x + 3*y - 15 = 0)
  (h₂ : ∀ x y k, l₂ x y k ↔ k*x - y - 6 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ x + 5*y = 0)
  (h₄ : ∀ x y, l₄ x y ↔ y = 0) :
  ∃ (k : Real) (circle : Real → Real → Prop),
    k = -8/15 ∧
    (∀ x y, circle x y ↔ x^2 + y^2 - 15*x - 159*y = 0) ∧
    (∀ x y, (l₁ x y ∨ l₂ x y k ∨ l₃ x y ∨ l₄ x y) → circle x y) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscribed_circle_l3987_398701


namespace NUMINAMATH_CALUDE_oranges_in_bowl_l3987_398734

def bowl_of_fruit (num_bananas : ℕ) (num_apples : ℕ) (num_oranges : ℕ) : Prop :=
  num_apples = 2 * num_bananas ∧
  num_bananas + num_apples + num_oranges = 12

theorem oranges_in_bowl :
  ∃ (num_oranges : ℕ), bowl_of_fruit 2 (2 * 2) num_oranges ∧ num_oranges = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_oranges_in_bowl_l3987_398734


namespace NUMINAMATH_CALUDE_packs_needed_for_360_days_l3987_398772

/-- The number of dog walks per day -/
def walks_per_day : ℕ := 2

/-- The number of wipes used per walk -/
def wipes_per_walk : ℕ := 1

/-- The number of wipes in a pack -/
def wipes_per_pack : ℕ := 120

/-- The number of days we need to cover -/
def days_to_cover : ℕ := 360

/-- The number of packs needed for the given number of days -/
def packs_needed : ℕ := 
  (days_to_cover * walks_per_day * wipes_per_walk + wipes_per_pack - 1) / wipes_per_pack

theorem packs_needed_for_360_days : packs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_packs_needed_for_360_days_l3987_398772


namespace NUMINAMATH_CALUDE_furniture_purchase_proof_l3987_398743

/-- Calculates the number of furniture pieces purchased given the total payment, reimbursement, and cost per piece. -/
def furniture_pieces (total_payment : ℕ) (reimbursement : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_payment - reimbursement) / cost_per_piece

/-- Proves that given the specific values in the problem, the number of furniture pieces is 150. -/
theorem furniture_purchase_proof :
  furniture_pieces 20700 600 134 = 150 := by
  sorry

#eval furniture_pieces 20700 600 134

end NUMINAMATH_CALUDE_furniture_purchase_proof_l3987_398743


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3987_398717

theorem algebraic_expression_value (k p : ℝ) :
  (∀ x : ℝ, (6 * x + 2) * (3 - x) = -6 * x^2 + k * x + p) →
  (k - p)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3987_398717


namespace NUMINAMATH_CALUDE_monomial_evaluation_l3987_398751

theorem monomial_evaluation : 0.007 * (-5)^7 * 2^9 = -280000 := by
  sorry

end NUMINAMATH_CALUDE_monomial_evaluation_l3987_398751


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3987_398787

def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 > 0

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 1
  else if 0 < a ∧ a < 1 then Set.Iio 1 ∪ Set.Ioi (1/a)
  else if a = 1 then Set.Iio 1 ∪ Set.Ioi 1
  else Set.Iio (1/a) ∪ Set.Ioi 1

theorem inequality_solution_set (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | inequality a x} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3987_398787


namespace NUMINAMATH_CALUDE_parabola_vertex_l3987_398759

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by the equation y^2 - 8y + 4x = 12 -/
def Parabola := {p : Point | p.y^2 - 8*p.y + 4*p.x = 12}

/-- The vertex of a parabola -/
def vertex : Point := ⟨7, 4⟩

/-- Theorem stating that the vertex of the parabola is (7, 4) -/
theorem parabola_vertex : vertex ∈ Parabola ∧ ∀ p ∈ Parabola, p.x ≥ vertex.x := by
  sorry

#check parabola_vertex

end NUMINAMATH_CALUDE_parabola_vertex_l3987_398759


namespace NUMINAMATH_CALUDE_carol_blocks_l3987_398778

/-- Given that Carol starts with 42 blocks and loses 25 blocks, 
    prove that she ends with 17 blocks. -/
theorem carol_blocks : 
  let initial_blocks : ℕ := 42
  let lost_blocks : ℕ := 25
  initial_blocks - lost_blocks = 17 := by
  sorry

end NUMINAMATH_CALUDE_carol_blocks_l3987_398778


namespace NUMINAMATH_CALUDE_christina_account_balance_l3987_398702

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem christina_account_balance :
  initial_balance - transferred_amount = remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_christina_account_balance_l3987_398702


namespace NUMINAMATH_CALUDE_gcd_lcm_3869_6497_l3987_398745

theorem gcd_lcm_3869_6497 :
  (Nat.gcd 3869 6497 = 73) ∧
  (Nat.lcm 3869 6497 = 344341) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_3869_6497_l3987_398745


namespace NUMINAMATH_CALUDE_pattern_theorem_l3987_398739

/-- Function to create a number with the first n digits of 123456... -/
def firstNDigits (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (firstNDigits (n-1)) * 10 + n

/-- Function to create a number with n ones -/
def nOnes (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (nOnes (n-1)) * 10 + 1

/-- Theorem stating the pattern observed in the problem -/
theorem pattern_theorem (n : ℕ) : 
  (firstNDigits n) * 9 + (n + 1) = nOnes (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_pattern_theorem_l3987_398739
