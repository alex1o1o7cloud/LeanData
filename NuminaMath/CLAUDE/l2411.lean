import Mathlib

namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2411_241148

/-- Custom binary operation ◇ -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 4 ◇ y = 50, then y = 58/7 -/
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2411_241148


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_l2411_241135

theorem negation_of_universal_positive_cubic (x : ℝ) :
  (¬ ∀ x ≥ 0, x^3 + x > 0) ↔ (∃ x ≥ 0, x^3 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_l2411_241135


namespace NUMINAMATH_CALUDE_bob_candy_count_l2411_241137

/-- Calculates Bob's share of candies given a total amount and a ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total / (samRatio + bobRatio)) * bobRatio

/-- The total number of candies Bob received --/
def bobTotalCandies : ℕ :=
  bobShare 45 2 3 + bobShare 60 3 1 + (45 / 2)

theorem bob_candy_count : bobTotalCandies = 64 := by
  sorry

#eval bobTotalCandies

end NUMINAMATH_CALUDE_bob_candy_count_l2411_241137


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_l2411_241155

theorem smallest_three_digit_divisible : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (6 ∣ n) ∧ (5 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (6 ∣ m) ∧ (5 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → n ≤ m) ∧
  n = 360 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_l2411_241155


namespace NUMINAMATH_CALUDE_sequence_general_term_l2411_241176

theorem sequence_general_term (n : ℕ+) : 
  let a : ℕ+ → ℝ := fun i => Real.sqrt i
  a n = Real.sqrt n := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2411_241176


namespace NUMINAMATH_CALUDE_freezing_temp_theorem_l2411_241142

def freezing_water : Int := 0
def freezing_alcohol : Int := -117
def freezing_mercury : Int := -39

def highest_freezing_temp : Int := max freezing_water (max freezing_alcohol freezing_mercury)
def lowest_freezing_temp : Int := min freezing_water (min freezing_alcohol freezing_mercury)

theorem freezing_temp_theorem :
  highest_freezing_temp = freezing_water ∧ lowest_freezing_temp = freezing_alcohol := by
  sorry

end NUMINAMATH_CALUDE_freezing_temp_theorem_l2411_241142


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2411_241107

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ), m > 0 → b > 0 →
  (5 * m + 4 * b) * 3 = 3 * m + 20 * b →
  m / b = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2411_241107


namespace NUMINAMATH_CALUDE_mbmt_equation_solution_l2411_241159

theorem mbmt_equation_solution :
  ∃ (T H E M B : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ M ∧ T ≠ B ∧
    H ≠ E ∧ H ≠ M ∧ H ≠ B ∧
    E ≠ M ∧ E ≠ B ∧
    M ≠ B ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ M < 10 ∧ B < 10 ∧
    B = 4 ∧ E = 2 ∧ T = 6 ∧
    (100 * T + 10 * H + E) + (1000 * M + 100 * B + 10 * M + T) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_mbmt_equation_solution_l2411_241159


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2411_241182

/-- Proves that the manufacturing cost of a shoe is 200, given the transportation cost,
    selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) 
  (profit_margin : ℚ) (h1 : transportation_cost = 500 / 100)
  (h2 : selling_price = 246) (h3 : profit_margin = 20 / 100) :
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2411_241182


namespace NUMINAMATH_CALUDE_special_arrangement_count_l2411_241174

/-- The number of ways to arrange n people in a row --/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    with the elderly people next to each other but not at the ends --/
def specialArrangement : ℕ :=
  choose 5 2 * linearArrangements 4 * 2

theorem special_arrangement_count : specialArrangement = 960 := by
  sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l2411_241174


namespace NUMINAMATH_CALUDE_large_cube_single_color_face_l2411_241191

/-- Represents a small cube with colored faces -/
structure SmallCube :=
  (white_faces : Fin 2)
  (blue_faces : Fin 2)
  (red_faces : Fin 2)

/-- Represents the large cube assembled from small cubes -/
def LargeCube := Fin 10 → Fin 10 → Fin 10 → SmallCube

/-- Predicate to check if two adjacent small cubes have matching colors -/
def matching_colors (c1 c2 : SmallCube) : Prop := sorry

/-- Predicate to check if a face of the large cube is a single color -/
def single_color_face (cube : LargeCube) : Prop := sorry

/-- Main theorem: The large cube has at least one face that is a single color -/
theorem large_cube_single_color_face 
  (cube : LargeCube)
  (h_matching : ∀ i j k i' j' k', 
    (i = i' ∧ j = j' ∧ (k + 1 = k' ∨ k = k' + 1)) ∨
    (i = i' ∧ (j + 1 = j' ∨ j = j' + 1) ∧ k = k') ∨
    ((i + 1 = i' ∨ i = i' + 1) ∧ j = j' ∧ k = k') →
    matching_colors (cube i j k) (cube i' j' k')) :
  single_color_face cube :=
sorry

end NUMINAMATH_CALUDE_large_cube_single_color_face_l2411_241191


namespace NUMINAMATH_CALUDE_foci_of_given_hyperbola_l2411_241168

/-- A hyperbola is defined by its equation and foci coordinates -/
structure Hyperbola where
  a_squared : ℝ
  b_squared : ℝ
  equation : (x y : ℝ) → Prop := λ x y => x^2 / a_squared - y^2 / b_squared = 1

/-- The foci of a hyperbola are the two fixed points used in its geometric definition -/
def foci (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = h.a_squared + h.b_squared ∧ p.2 = 0}

/-- The given hyperbola from the problem -/
def given_hyperbola : Hyperbola :=
  { a_squared := 7
    b_squared := 3 }

/-- The theorem states that the foci of the given hyperbola are (√10, 0) and (-√10, 0) -/
theorem foci_of_given_hyperbola :
  foci given_hyperbola = {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)} := by
  sorry

end NUMINAMATH_CALUDE_foci_of_given_hyperbola_l2411_241168


namespace NUMINAMATH_CALUDE_consecutive_product_theorem_l2411_241130

theorem consecutive_product_theorem (n : ℕ+) : 
  (∃ k : ℕ, k > 1 ∧ (n.val^6 + 5*n.val^3 + 4*n.val + 116 = (k * (k + 1)) ∨
                     n.val^6 + 5*n.val^3 + 4*n.val + 116 = ((k - 1) * k * (k + 1)) ∨
                     n.val^6 + 5*n.val^3 + 4*n.val + 116 = (k * (k + 1) * (k + 2) * (k + 3)))) ↔ 
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_theorem_l2411_241130


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l2411_241180

/-- The hyperbola equation -/
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Point on asymptote condition -/
def point_on_asymptote (a b : ℝ) : Prop :=
  4 / 3 = b / a

/-- Perpendicular foci condition -/
def perpendicular_foci (c : ℝ) : Prop :=
  4 / (3 + c) * (4 / (3 - c)) = -1

/-- Relationship between a, b, and c -/
def foci_distance (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Main theorem -/
theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : point_on_asymptote a b)
  (h_foci : ∃ c, perpendicular_foci c ∧ foci_distance a b c) :
  hyperbola_equation 3 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l2411_241180


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l2411_241104

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Translation of line segment AB to A'B' -/
theorem translation_of_line_segment (A B A' : Point) (t : Translation) :
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 0 ∧ B.y = 3 ∧
  A'.x = 2 ∧ A'.y = 1 ∧
  A' = applyTranslation A t →
  applyTranslation B t = { x := 4, y := 4 } :=
by sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l2411_241104


namespace NUMINAMATH_CALUDE_positive_X_value_l2411_241197

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 338) : X = 17 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l2411_241197


namespace NUMINAMATH_CALUDE_probability_of_one_each_l2411_241105

def drawer_contents : ℕ := 7

def total_items : ℕ := 4 * drawer_contents

def ways_to_select_one_of_each : ℕ := drawer_contents^4

def total_selections : ℕ := (total_items.choose 4)

theorem probability_of_one_each : 
  (ways_to_select_one_of_each : ℚ) / total_selections = 2401 / 20475 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_one_each_l2411_241105


namespace NUMINAMATH_CALUDE_barbara_shopping_cost_l2411_241146

/-- The amount Barbara spent on goods other than tuna and water -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid - (tuna_packs : ℚ) * tuna_price - (water_bottles : ℚ) * water_price

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping_cost :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_cost_l2411_241146


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2411_241190

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2411_241190


namespace NUMINAMATH_CALUDE_max_ratio_on_circle_l2411_241157

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define a function to check if a point is on the circle x^2 + y^2 = 16
def onCircle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 16

-- Define a function to calculate the squared distance between two points
def squaredDistance (p1 p2 : IntPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Theorem statement
theorem max_ratio_on_circle (A B C D : IntPoint) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  onCircle A ∧ onCircle B ∧ onCircle C ∧ onCircle D →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance A B ∧ n > 0 →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance C D ∧ n > 0 →
  ∀ r : ℚ, r * (squaredDistance C D : ℚ) ≤ (squaredDistance A B : ℚ) → r ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_on_circle_l2411_241157


namespace NUMINAMATH_CALUDE_triangle_properties_l2411_241161

/-- An acute triangle with sides a, b, c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0
  cosine_law : b^2 = a^2 + c^2 - a*c

/-- The perimeter of a triangle -/
def perimeter (t : AcuteTriangle) : ℝ := t.a + t.b + t.c

/-- The area of a triangle -/
def area (t : AcuteTriangle) : ℝ := sorry

theorem triangle_properties (t : AcuteTriangle) :
  (∃ angleA : ℝ, angleA = 60 * (π / 180) ∧ t.c = 2 → t.a = 2) ∧
  (area t = 2 * Real.sqrt 3 →
    6 * Real.sqrt 2 ≤ perimeter t ∧ perimeter t < 6 + 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2411_241161


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2411_241195

theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let AB := 2
  let BC := Real.sqrt 5
  let CD := Real.sqrt 3
  let DE := 1
  let AC := Real.sqrt ((AB ^ 2) + (BC ^ 2))
  let AD := Real.sqrt ((AC ^ 2) + (CD ^ 2))
  let AE := Real.sqrt ((AD ^ 2) + (DE ^ 2))
  AB + BC + CD + DE + AE = 3 + Real.sqrt 5 + Real.sqrt 3 + 1 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l2411_241195


namespace NUMINAMATH_CALUDE_car_rental_budget_is_75_l2411_241199

/-- Calculates the budget for a car rental given the daily rate, per-mile rate, and miles driven. -/
def carRentalBudget (dailyRate : ℝ) (perMileRate : ℝ) (milesDriven : ℝ) : ℝ :=
  dailyRate + perMileRate * milesDriven

/-- Theorem: The budget for a car rental with specific rates and mileage is $75.00. -/
theorem car_rental_budget_is_75 :
  let dailyRate : ℝ := 30
  let perMileRate : ℝ := 0.18
  let milesDriven : ℝ := 250.0
  carRentalBudget dailyRate perMileRate milesDriven = 75 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_budget_is_75_l2411_241199


namespace NUMINAMATH_CALUDE_factorial_ratio_52_50_l2411_241112

theorem factorial_ratio_52_50 : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_52_50_l2411_241112


namespace NUMINAMATH_CALUDE_katherines_bananas_l2411_241144

/-- Given Katherine's fruit inventory, calculate the number of bananas -/
theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = apples + pears + bananas →
  total = 21 →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_bananas_l2411_241144


namespace NUMINAMATH_CALUDE_cosine_equality_l2411_241189

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (865 * π / 180) → n = 35 ∨ n = 145 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2411_241189


namespace NUMINAMATH_CALUDE_faster_train_speed_l2411_241129

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed 
  (train_length : ℝ) 
  (speed_difference : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 65)
  (h2 : speed_difference = 36)
  (h3 : passing_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2411_241129


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2411_241136

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2411_241136


namespace NUMINAMATH_CALUDE_shaded_area_of_circles_l2411_241126

theorem shaded_area_of_circles (R : ℝ) (h : R = 8) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 32 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_circles_l2411_241126


namespace NUMINAMATH_CALUDE_valid_pairs_l2411_241162

def is_valid_pair (m n : ℕ) : Prop :=
  Nat.Prime m ∧ Nat.Prime n ∧ m < n ∧ n < 5 * m ∧ Nat.Prime (m + 3 * n)

theorem valid_pairs :
  ∀ m n : ℕ, is_valid_pair m n ↔ (m = 2 ∧ n = 3) ∨ (m = 2 ∧ n = 5) ∨ (m = 2 ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2411_241162


namespace NUMINAMATH_CALUDE_existence_of_irreducible_fractions_l2411_241125

theorem existence_of_irreducible_fractions : ∃ (a b : ℕ), 
  (Nat.gcd a b = 1) ∧ 
  (Nat.gcd (a + 1) b = 1) ∧ 
  (Nat.gcd (a + 1) (b + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irreducible_fractions_l2411_241125


namespace NUMINAMATH_CALUDE_team_average_score_l2411_241140

theorem team_average_score (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) :
  player1_score = 20 →
  player2_score = player1_score / 2 →
  player3_score = 6 * player2_score →
  (player1_score + player2_score + player3_score) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l2411_241140


namespace NUMINAMATH_CALUDE_tan_identity_implies_cos_squared_l2411_241165

theorem tan_identity_implies_cos_squared (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.cos (θ + π/4)^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tan_identity_implies_cos_squared_l2411_241165


namespace NUMINAMATH_CALUDE_transport_tax_calculation_l2411_241131

def calculate_transport_tax (engine_power : ℕ) (tax_rate : ℕ) (ownership_months : ℕ) : ℕ :=
  (engine_power * tax_rate * ownership_months) / 12

theorem transport_tax_calculation :
  calculate_transport_tax 250 75 2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_transport_tax_calculation_l2411_241131


namespace NUMINAMATH_CALUDE_jeremy_stroll_time_l2411_241149

/-- Proves that Jeremy's strolling time is 10 hours given his distance and speed -/
theorem jeremy_stroll_time (distance : ℝ) (speed : ℝ) (h1 : distance = 20) (h2 : speed = 2) :
  distance / speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_stroll_time_l2411_241149


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_constant_value_l2411_241179

theorem polynomial_factor_implies_constant_value (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_constant_value_l2411_241179


namespace NUMINAMATH_CALUDE_lampshire_parade_group_size_l2411_241175

theorem lampshire_parade_group_size (n : ℕ) : 
  (∃ k : ℕ, n = 30 * k) →
  (30 * n) % 31 = 7 →
  (30 * n) % 17 = 0 →
  30 * n < 1500 →
  (∀ m : ℕ, 
    (∃ j : ℕ, m = 30 * j) →
    (30 * m) % 31 = 7 →
    (30 * m) % 17 = 0 →
    30 * m < 1500 →
    30 * m ≤ 30 * n) →
  30 * n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_lampshire_parade_group_size_l2411_241175


namespace NUMINAMATH_CALUDE_solve_equation_l2411_241143

theorem solve_equation (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2411_241143


namespace NUMINAMATH_CALUDE_problem_statement_l2411_241115

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  (a < 1/2 ∧ 1/2 < b) ∧ (a < a^2 + b^2 ∧ a^2 + b^2 < b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2411_241115


namespace NUMINAMATH_CALUDE_laptop_gifting_l2411_241117

theorem laptop_gifting (n m : ℕ) (hn : n = 15) (hm : m = 3) :
  (n.factorial / (n - m).factorial) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_laptop_gifting_l2411_241117


namespace NUMINAMATH_CALUDE_least_even_integer_for_300p_perfect_square_l2411_241109

theorem least_even_integer_for_300p_perfect_square :
  ∀ p : ℕ,
    p % 2 = 0 →
    (∃ n : ℕ, 300 * p = n^2) →
    p ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_least_even_integer_for_300p_perfect_square_l2411_241109


namespace NUMINAMATH_CALUDE_plane_representations_l2411_241194

/-- Given a plane with equation 2x - 2y + z - 20 = 0, prove its representations in intercept and normal forms -/
theorem plane_representations (x y z : ℝ) :
  (2*x - 2*y + z - 20 = 0) →
  (x/10 + y/(-10) + z/20 = 1) ∧
  (-2/3*x + 2/3*y - 1/3*z + 20/3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_representations_l2411_241194


namespace NUMINAMATH_CALUDE_hours_worked_per_day_l2411_241152

/-- Given a person who worked for 5 days and a total of 15 hours, 
    prove that the number of hours worked each day is equal to 3. -/
theorem hours_worked_per_day 
  (days_worked : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_worked = 5) 
  (h2 : total_hours = 15) : 
  total_hours / days_worked = 3 := by
  sorry

end NUMINAMATH_CALUDE_hours_worked_per_day_l2411_241152


namespace NUMINAMATH_CALUDE_tank_length_is_25_l2411_241192

/-- Given a tank with specific dimensions and plastering costs, prove its length is 25 meters -/
theorem tank_length_is_25 (width : ℝ) (depth : ℝ) (plaster_cost_per_sqm : ℝ) (total_plaster_cost : ℝ) :
  width = 12 →
  depth = 6 →
  plaster_cost_per_sqm = 0.45 →
  total_plaster_cost = 334.8 →
  (∃ (length : ℝ), 
    total_plaster_cost / plaster_cost_per_sqm = 2 * (length * depth) + 2 * (width * depth) + (length * width) ∧
    length = 25) := by
  sorry

end NUMINAMATH_CALUDE_tank_length_is_25_l2411_241192


namespace NUMINAMATH_CALUDE_unique_polynomial_function_l2411_241158

/-- A polynomial function over ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- Predicate to check if a function is a polynomial of degree ≥ 1 -/
def IsPolynomialDegreeGEOne (f : PolynomialFunction) : Prop := sorry

/-- The conditions that the polynomial function must satisfy -/
def SatisfiesConditions (f : PolynomialFunction) : Prop :=
  IsPolynomialDegreeGEOne f ∧
  (∀ x : ℝ, f (x^2) = (f x)^3) ∧
  (∀ x : ℝ, f (f x) = f x)

/-- Theorem stating that there exists exactly one polynomial function satisfying the conditions -/
theorem unique_polynomial_function :
  ∃! f : PolynomialFunction, SatisfiesConditions f := by sorry

end NUMINAMATH_CALUDE_unique_polynomial_function_l2411_241158


namespace NUMINAMATH_CALUDE_tan_sin_equation_l2411_241122

theorem tan_sin_equation (m : ℝ) : 
  Real.tan (20 * π / 180) + m * Real.sin (20 * π / 180) = Real.sqrt 3 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_equation_l2411_241122


namespace NUMINAMATH_CALUDE_toms_profit_l2411_241196

/-- Calculates Tom's profit from lawn mowing and weed pulling -/
def calculate_profit (lawns_mowed : ℕ) (price_per_lawn : ℕ) (gas_expense : ℕ) (weed_pulling_income : ℕ) : ℕ :=
  lawns_mowed * price_per_lawn + weed_pulling_income - gas_expense

/-- Theorem: Tom's profit last month was $29 -/
theorem toms_profit :
  calculate_profit 3 12 17 10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_toms_profit_l2411_241196


namespace NUMINAMATH_CALUDE_rectangle_area_l2411_241163

/-- A rectangle with perimeter 40 and length twice its width has area 800/9 -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2411_241163


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2411_241164

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2023)) ↔ x ≠ 2023 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2411_241164


namespace NUMINAMATH_CALUDE_polygon_triangulation_l2411_241120

/-- A color type with three possible values -/
inductive Color
  | one
  | two
  | three

/-- A vertex of a polygon -/
structure Vertex where
  color : Color

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : List Vertex
  convex : Bool
  all_colors_present : Bool
  no_adjacent_same_color : Bool

/-- A triangle with three vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- A triangulation of a polygon -/
structure Triangulation where
  triangles : List Triangle

/-- The main theorem -/
theorem polygon_triangulation (p : ConvexPolygon) :
  p.convex ∧ p.all_colors_present ∧ p.no_adjacent_same_color →
  ∃ (t : Triangulation), ∀ (triangle : Triangle), triangle ∈ t.triangles →
    triangle.v1.color ≠ triangle.v2.color ∧
    triangle.v2.color ≠ triangle.v3.color ∧
    triangle.v3.color ≠ triangle.v1.color :=
sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l2411_241120


namespace NUMINAMATH_CALUDE_martin_purchase_cost_l2411_241102

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (prices : StorePrices) : Prop :=
  prices.notebook + prices.eraser = 85 ∧
  prices.pencil + prices.eraser = 45 ∧
  3 * prices.pencil + 3 * prices.notebook + 3 * prices.eraser = 315

/-- The theorem stating that Martin's purchase costs 80 cents -/
theorem martin_purchase_cost (prices : StorePrices) 
  (h : store_conditions prices) : 
  prices.pencil + prices.notebook = 80 := by
  sorry

end NUMINAMATH_CALUDE_martin_purchase_cost_l2411_241102


namespace NUMINAMATH_CALUDE_bridget_fruits_count_bridget_fruits_proof_l2411_241100

theorem bridget_fruits_count : ℕ → ℕ → Prop :=
  fun apples oranges =>
    apples / oranges = 2 →
    apples / 2 - 3 = 4 →
    oranges - 3 = 5 →
    apples + oranges = 21

theorem bridget_fruits_proof : ∃ a o : ℕ, bridget_fruits_count a o := by
  sorry

end NUMINAMATH_CALUDE_bridget_fruits_count_bridget_fruits_proof_l2411_241100


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l2411_241184

theorem prime_pair_divisibility (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ k₁ k₂ : ℤ, ((2 * p^2 - 1)^q + 1 : ℤ) = k₁ * (p + q) ∧ 
                ((2 * q^2 - 1)^p + 1 : ℤ) = k₂ * (p + q)) ↔ 
  p = q := by sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l2411_241184


namespace NUMINAMATH_CALUDE_sum_of_factors_48_l2411_241150

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_48 : sum_of_factors 48 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_48_l2411_241150


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2411_241187

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 2 * x + y = 7) 
  (eq2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2411_241187


namespace NUMINAMATH_CALUDE_employee_transfer_solution_l2411_241181

/-- Represents the company's employee transfer problem -/
def EmployeeTransfer (a : ℝ) (x : ℕ) : Prop :=
  let total_employees : ℕ := 100
  let manufacturing_before : ℝ := a * total_employees
  let manufacturing_after : ℝ := a * 1.2 * (total_employees - x)
  let service_output : ℝ := 3.5 * a * x
  (manufacturing_after ≥ manufacturing_before) ∧ 
  (service_output ≥ 0.5 * manufacturing_before) ∧
  (x ≤ total_employees)

/-- Theorem stating the solution to the employee transfer problem -/
theorem employee_transfer_solution (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, EmployeeTransfer a x ∧ x ≥ 15 ∧ x ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_employee_transfer_solution_l2411_241181


namespace NUMINAMATH_CALUDE_simplify_expression_l2411_241153

theorem simplify_expression : (5 + 7 + 3) / 3 - 2 / 3 - 1 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2411_241153


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l2411_241108

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem set_operations_and_inclusion (a : ℝ) :
  (a = 7/2 → M ∪ N a = {x | -2 ≤ x ∧ x ≤ 6} ∧
             (Set.univ \ M) ∩ N a = {x | 5 < x ∧ x ≤ 6}) ∧
  (M ⊇ N a ↔ a ∈ Set.Iic 3) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l2411_241108


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_inequality_l2411_241127

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_cubic_inequality : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_inequality_l2411_241127


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l2411_241124

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + a^2 > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a+1)*x + a - 1 = 0 ∧ y^2 + (a+1)*y + a - 1 = 0

def r (a m : ℝ) : Prop := a^2 - 2*a + 1 - m^2 ≥ 0 ∧ m > 0

-- Theorem 1
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → (-2 ≤ a ∧ a < 1) ∨ a > 2 :=
sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : (∀ a : ℝ, ¬(r a m) → ¬(p a)) ∧ (∃ a : ℝ, ¬(r a m) ∧ p a) → m > 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l2411_241124


namespace NUMINAMATH_CALUDE_melanie_plums_l2411_241147

/-- The number of plums Melanie has after giving some away -/
def plums_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Melanie has 4 plums after initially picking 7 and giving 3 away -/
theorem melanie_plums : plums_remaining 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l2411_241147


namespace NUMINAMATH_CALUDE_gcd_equation_solution_l2411_241106

theorem gcd_equation_solution (b d : ℕ) : 
  Nat.gcd b 175 = d → 
  176 * (b - 11 * d + 1) = 5 * d + 1 → 
  b = 385 := by
sorry

end NUMINAMATH_CALUDE_gcd_equation_solution_l2411_241106


namespace NUMINAMATH_CALUDE_train_length_l2411_241138

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 64 * (5 / 18) → time = 9 → speed * time = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2411_241138


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2411_241132

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (List.sum digits)

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2411_241132


namespace NUMINAMATH_CALUDE_baseball_team_groups_l2411_241111

theorem baseball_team_groups (new_players : ℕ) (returning_players : ℕ) (players_per_group : ℕ) :
  new_players = 48 →
  returning_players = 6 →
  players_per_group = 6 →
  (new_players + returning_players) / players_per_group = 9 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l2411_241111


namespace NUMINAMATH_CALUDE_teal_color_perception_l2411_241121

theorem teal_color_perception (total : ℕ) (greenish : ℕ) (both : ℕ) (neither : ℕ) :
  total = 120 →
  greenish = 80 →
  both = 35 →
  neither = 20 →
  ∃ bluish : ℕ, bluish = 55 ∧ bluish = total - (greenish - both) - both - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l2411_241121


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2411_241173

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 17 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 17 = 0 → n ≤ m) ∧
  n = 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2411_241173


namespace NUMINAMATH_CALUDE_david_pushups_count_l2411_241119

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The difference between David's and Zachary's push-ups -/
def david_extra_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + david_extra_pushups

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l2411_241119


namespace NUMINAMATH_CALUDE_min_area_MAB_l2411_241172

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the area of triangle MAB
def area_MAB (k : ℝ) : ℝ := 4*(1 + k^2)^(3/2)

-- State the theorem
theorem min_area_MAB :
  ∃ (min_area : ℝ), min_area = 4 ∧
  ∀ (k : ℝ), area_MAB k ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_area_MAB_l2411_241172


namespace NUMINAMATH_CALUDE_min_sum_of_product_100_l2411_241166

theorem min_sum_of_product_100 (a b : ℤ) (h : a * b = 100) :
  ∀ (x y : ℤ), x * y = 100 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 100 ∧ a₀ + b₀ = -101 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_100_l2411_241166


namespace NUMINAMATH_CALUDE_terminal_side_symmetry_ratio_l2411_241169

theorem terminal_side_symmetry_ratio (θ : Real) (x y : Real) :
  θ ∈ Set.Ioo 0 360 →
  -- Terminal side of θ is symmetric to terminal side of 660° w.r.t. x-axis
  (∃ k : ℤ, θ + 660 = 360 * (2 * k + 1)) →
  x ≠ 0 ∨ y ≠ 0 →  -- P(x, y) is not the origin
  y / x = Real.tan θ →  -- P(x, y) is on the terminal side of θ
  x * y / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_symmetry_ratio_l2411_241169


namespace NUMINAMATH_CALUDE_function_properties_l2411_241114

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
variable (h : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)

-- Theorem statement
theorem function_properties :
  (f 0 = 0) ∧ (f (-1) = 0) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2411_241114


namespace NUMINAMATH_CALUDE_solve_equation_l2411_241110

-- Define the ⊗ operation
def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

-- State the theorem
theorem solve_equation (x : ℝ) : 
  otimes (x + 1) (x - 2) = 5 → x = 0 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2411_241110


namespace NUMINAMATH_CALUDE_function_properties_l2411_241160

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : functional_equation f) : 
  (f 0 = 2) ∧ 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x : ℝ, f (x + 6) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2411_241160


namespace NUMINAMATH_CALUDE_max_value_determines_parameter_l2411_241128

/-- Given a system of linear inequalities and an objective function,
    prove that the maximum value of the objective function
    determines the value of a parameter. -/
theorem max_value_determines_parameter
  (x y z a : ℝ)
  (h1 : x - 3 ≤ 0)
  (h2 : y - a ≤ 0)
  (h3 : x + y ≥ 0)
  (h4 : z = 2*x + y)
  (h5 : ∀ x' y' z', x' - 3 ≤ 0 → y' - a ≤ 0 → x' + y' ≥ 0 → z' = 2*x' + y' → z' ≤ 10)
  (h6 : ∃ x' y' z', x' - 3 ≤ 0 ∧ y' - a ≤ 0 ∧ x' + y' ≥ 0 ∧ z' = 2*x' + y' ∧ z' = 10) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_determines_parameter_l2411_241128


namespace NUMINAMATH_CALUDE_absolute_difference_21st_terms_l2411_241183

-- Define arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

-- Define the sequences C and D
def C (n : ℕ) : ℤ := arithmeticSequence 50 12 n
def D (n : ℕ) : ℤ := arithmeticSequence 50 (-14) n

-- State the theorem
theorem absolute_difference_21st_terms :
  |C 21 - D 21| = 520 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_21st_terms_l2411_241183


namespace NUMINAMATH_CALUDE_acute_angles_equation_solution_l2411_241170

theorem acute_angles_equation_solution (A B : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  Real.sin A * Real.cos B + Real.sqrt (2 * Real.sin A) * Real.sin B = (3 * Real.sin A + 1) / Real.sqrt 5 →
  A = π/6 ∧ B = π/2 - Real.arcsin (Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_equation_solution_l2411_241170


namespace NUMINAMATH_CALUDE_dentist_age_problem_l2411_241123

/-- Given a dentist's current age and the relationship between his past and future ages,
    calculate how many years ago his age was being considered. -/
theorem dentist_age_problem (current_age : ℕ) (h : current_age = 32) : 
  ∃ (x : ℕ), (1 / 6 : ℚ) * (current_age - x) = (1 / 10 : ℚ) * (current_age + 8) ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_dentist_age_problem_l2411_241123


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l2411_241116

theorem favorite_numbers_exist : ∃ x y : ℕ, x > y ∧ x ≠ y ∧ (x + y) + (x - y) + x * y + (x / y) = 98 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l2411_241116


namespace NUMINAMATH_CALUDE_golden_raisins_fraction_of_total_cost_l2411_241156

/-- Represents the cost of ingredients relative to golden raisins -/
structure IngredientCost where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

/-- Represents the weight of ingredients in pounds -/
structure IngredientWeight where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

def mixtureCost (cost : IngredientCost) (weight : IngredientWeight) : ℚ :=
  cost.goldenRaisins * weight.goldenRaisins +
  cost.almonds * weight.almonds +
  cost.cashews * weight.cashews +
  cost.walnuts * weight.walnuts

theorem golden_raisins_fraction_of_total_cost 
  (cost : IngredientCost)
  (weight : IngredientWeight)
  (h1 : cost.goldenRaisins = 1)
  (h2 : cost.almonds = 2 * cost.goldenRaisins)
  (h3 : cost.cashews = 3 * cost.goldenRaisins)
  (h4 : cost.walnuts = 4 * cost.goldenRaisins)
  (h5 : weight.goldenRaisins = 4)
  (h6 : weight.almonds = 2)
  (h7 : weight.cashews = 1)
  (h8 : weight.walnuts = 1) :
  (cost.goldenRaisins * weight.goldenRaisins) / mixtureCost cost weight = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_golden_raisins_fraction_of_total_cost_l2411_241156


namespace NUMINAMATH_CALUDE_bc_length_fraction_l2411_241103

/-- Given a line segment AD with points B and C on it, prove that BC = 5/36 * AD -/
theorem bc_length_fraction (A B C D : ℝ) : 
  (B - A) = 3 * (D - B) →  -- AB = 3 * BD
  (C - A) = 8 * (D - C) →  -- AC = 8 * CD
  (C - B) = (5 / 36) * (D - A) := by sorry

end NUMINAMATH_CALUDE_bc_length_fraction_l2411_241103


namespace NUMINAMATH_CALUDE_average_hours_worked_l2411_241185

/-- Represents the number of hours worked on a given day type in a month -/
structure MonthlyHours where
  weekday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Represents the work schedule for a month -/
structure MonthSchedule where
  days : ℕ
  weekdays : ℕ
  saturdays : ℕ
  sundays : ℕ
  hours : MonthlyHours
  vacation_days : ℕ

def april : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 6, saturday := 4, sunday := 0 }
    vacation_days := 5 }

def june : MonthSchedule :=
  { days := 30
    weekdays := 30
    saturdays := 0
    sundays := 0
    hours := { weekday := 5, saturday := 5, sunday := 5 }
    vacation_days := 4 }

def september : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 8, saturday := 0, sunday := 0 }
    vacation_days := 0 }

def calculate_hours (m : MonthSchedule) : ℕ :=
  (m.weekdays - m.vacation_days) * m.hours.weekday +
  m.saturdays * m.hours.saturday +
  m.sundays * m.hours.sunday

theorem average_hours_worked :
  (calculate_hours april + calculate_hours june + calculate_hours september) / 3 = 141 :=
sorry

end NUMINAMATH_CALUDE_average_hours_worked_l2411_241185


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2411_241139

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.10 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 345 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2411_241139


namespace NUMINAMATH_CALUDE_johns_change_l2411_241154

/-- The change John receives when buying barbells -/
def change_received (num_barbells : ℕ) (barbell_cost : ℕ) (money_given : ℕ) : ℕ :=
  money_given - (num_barbells * barbell_cost)

/-- Theorem: John's change when buying 3 barbells at $270 each and giving $850 is $40 -/
theorem johns_change :
  change_received 3 270 850 = 40 := by
  sorry

end NUMINAMATH_CALUDE_johns_change_l2411_241154


namespace NUMINAMATH_CALUDE_merchant_profit_l2411_241167

/-- Calculates the profit percentage given the ratio of cost price to selling price -/
def profit_percentage (cost_price : ℚ) (selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Proves that if the cost price of 19 articles is equal to the selling price of 16 articles,
    then the merchant makes a profit of 18.75% -/
theorem merchant_profit :
  ∀ (cost_price selling_price : ℚ),
  19 * cost_price = 16 * selling_price →
  profit_percentage cost_price selling_price = 18.75 := by
sorry

#eval profit_percentage 16 19 -- Should evaluate to 18.75

end NUMINAMATH_CALUDE_merchant_profit_l2411_241167


namespace NUMINAMATH_CALUDE_tan_product_ninth_pi_l2411_241198

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_ninth_pi_l2411_241198


namespace NUMINAMATH_CALUDE_square_division_l2411_241193

/-- A square can be divided into n smaller squares for any natural number n ≥ 6 -/
theorem square_division (n : ℕ) (h : n ≥ 6) : 
  ∃ (partition : List (ℕ × ℕ)), 
    (partition.length = n) ∧ 
    (∀ (x y : ℕ × ℕ), x ∈ partition → y ∈ partition → x ≠ y → 
      (x.1 < y.1 ∨ x.2 < y.2 ∨ y.1 < x.1 ∨ y.2 < x.2)) ∧
    (∃ (side : ℕ), ∀ (square : ℕ × ℕ), square ∈ partition → 
      square.1 ≤ side ∧ square.2 ≤ side) := by
  sorry

end NUMINAMATH_CALUDE_square_division_l2411_241193


namespace NUMINAMATH_CALUDE_unique_triangle_number_three_identical_digits_l2411_241177

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number composed of identical digits -/
def is_three_identical_digits (n : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ d < 10 ∧ n = d * 111

theorem unique_triangle_number_three_identical_digits :
  ∃! (n : ℕ), n > 0 ∧ is_three_identical_digits (triangle_number n) :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_number_three_identical_digits_l2411_241177


namespace NUMINAMATH_CALUDE_rectangle_area_l2411_241118

/-- Given a rectangle divided into four smaller rectangles by two lines parallel to its sides,
    where one of these smaller rectangles is a square, and the perimeters of the rectangles
    adjacent to the square are 20 cm and 16 cm, prove that the area of the original rectangle
    is 80 cm². -/
theorem rectangle_area (a b x : ℝ) (h1 : a + x = 10) (h2 : b + x = 8) : a * b = 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2411_241118


namespace NUMINAMATH_CALUDE_billy_score_is_13_l2411_241133

/-- Represents a contestant's performance on the AMC 8 contest -/
structure AMC8Performance where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered : Nat
  correct_point_value : Nat
  incorrect_point_value : Nat
  unanswered_point_value : Nat

/-- Calculates the score for an AMC 8 performance -/
def calculate_score (performance : AMC8Performance) : Nat :=
  performance.correct_answers * performance.correct_point_value +
  performance.incorrect_answers * performance.incorrect_point_value +
  performance.unanswered * performance.unanswered_point_value

/-- Billy's performance on the AMC 8 contest -/
def billy_performance : AMC8Performance := {
  total_questions := 25,
  correct_answers := 13,
  incorrect_answers := 7,
  unanswered := 5,
  correct_point_value := 1,
  incorrect_point_value := 0,
  unanswered_point_value := 0
}

theorem billy_score_is_13 : calculate_score billy_performance = 13 := by
  sorry

end NUMINAMATH_CALUDE_billy_score_is_13_l2411_241133


namespace NUMINAMATH_CALUDE_barbara_winning_condition_l2411_241151

/-- The game rules for Alberto and Barbara --/
structure GameRules where
  alberto_choice : ℕ → ℕ
  barbara_choice : ℕ → ℕ → ℕ
  alberto_move : ℕ → ℕ
  max_moves : ℕ

/-- Barbara's winning condition --/
def barbara_wins (n : ℕ) (rules : GameRules) : Prop :=
  ∃ (strategy : ℕ → ℕ), ∀ (alberto_plays : ℕ → ℕ),
    ∃ (m : ℕ), m ≤ rules.max_moves ∧
    (strategy (alberto_plays m)) = n

/-- The main theorem --/
theorem barbara_winning_condition (n : ℕ) (h : n > 1) :
  (∃ (rules : GameRules), barbara_wins n rules) ↔ (∃ (k : ℕ), n = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_barbara_winning_condition_l2411_241151


namespace NUMINAMATH_CALUDE_matrix_power_2023_l2411_241145

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 6069, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l2411_241145


namespace NUMINAMATH_CALUDE_fiftieth_islander_statement_l2411_241171

/-- Represents the type of islander: Knight (always tells the truth) or Liar (always lies) -/
inductive IslanderType
| Knight
| Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
| Knight
| Liar

/-- A function that determines what an islander at a given position says about their right neighbor -/
def whatTheySay (position : Nat) : Statement :=
  if position % 2 = 1 then Statement.Knight else Statement.Liar

/-- The main theorem to prove -/
theorem fiftieth_islander_statement :
  ∀ (islanders : Fin 50 → IslanderType),
  (∀ (i : Fin 50), 
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Knight) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Knight)) →
  whatTheySay 50 = Statement.Knight :=
sorry

end NUMINAMATH_CALUDE_fiftieth_islander_statement_l2411_241171


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2411_241134

/-- The value of m for which the line x + y + m = 0 is tangent to the circle x² + y² = m -/
theorem tangent_line_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 ≠ m) ∧ 
  (∃ x y : ℝ, x + y + m = 0 ∧ x^2 + y^2 = m) → 
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2411_241134


namespace NUMINAMATH_CALUDE_angle_A_is_30_degrees_l2411_241101

/-- In a triangle ABC, given that the side opposite to angle B is twice the length of the side opposite to angle A, and angle B is 60° greater than angle A, prove that angle A measures 30°. -/
theorem angle_A_is_30_degrees (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 * a ∧  -- Given condition
  B = A + π / 3 →  -- B = A + 60° (in radians)
  A = π / 6 :=  -- A = 30° (in radians)
by sorry

end NUMINAMATH_CALUDE_angle_A_is_30_degrees_l2411_241101


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2411_241186

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2411_241186


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2411_241178

/-- The equation of the line passing through the tangency points of two tangent lines drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (5, 3) →
  r = 3 →
  ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    ((A.1 - P.1)^2 + (A.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    ((B.1 - P.1)^2 + (B.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    (∀ x y : ℝ, 5*x + 3*y - 9 = 0 ↔ (x - A.1)*(B.2 - A.2) = (y - A.2)*(B.1 - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2411_241178


namespace NUMINAMATH_CALUDE_floor_length_calculation_l2411_241113

theorem floor_length_calculation (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 80 →
  length = 3 * Real.sqrt (80 / 3) := by
sorry

end NUMINAMATH_CALUDE_floor_length_calculation_l2411_241113


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l2411_241188

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - 7/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l2411_241188


namespace NUMINAMATH_CALUDE_m_less_than_2_necessary_not_sufficient_l2411_241141

-- Define the quadratic function
def f (m x : ℝ) := x^2 + m*x + 1

-- Define the condition for the solution set to be ℝ
def solution_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- Define the necessary and sufficient condition
def necessary_and_sufficient (m : ℝ) : Prop :=
  m^2 - 4 < 0

-- Theorem: m < 2 is a necessary but not sufficient condition
theorem m_less_than_2_necessary_not_sufficient :
  (∀ m, solution_is_real m → m < 2) ∧
  ¬(∀ m, m < 2 → solution_is_real m) :=
sorry

end NUMINAMATH_CALUDE_m_less_than_2_necessary_not_sufficient_l2411_241141
