import Mathlib

namespace NUMINAMATH_CALUDE_fraction_difference_l3582_358203

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l3582_358203


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3582_358264

theorem smallest_multiple_of_6_and_15 :
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 6 ∣ x ∧ 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3582_358264


namespace NUMINAMATH_CALUDE_profit_maximization_l3582_358249

variable (x : ℝ)

def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
def sales_revenue (x : ℝ) : ℝ := 18*x
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_maximization (h : x > 0) :
  profit x = -x^3 + 24*x^2 - 45*x - 10 ∧
  ∃ (max_x : ℝ), max_x = 15 ∧
    ∀ (y : ℝ), y > 0 → profit y ≤ profit max_x ∧
    profit max_x = 1340 :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l3582_358249


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3582_358223

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3582_358223


namespace NUMINAMATH_CALUDE_total_siblings_weight_l3582_358231

def antonio_weight : ℕ := 50
def sister_weight_diff : ℕ := 12
def antonio_backpack : ℕ := 5
def sister_backpack : ℕ := 3
def marco_weight : ℕ := 30
def stuffed_animal : ℕ := 2

theorem total_siblings_weight :
  (antonio_weight + (antonio_weight - sister_weight_diff) + marco_weight) +
  (antonio_backpack + sister_backpack + stuffed_animal) = 128 := by
  sorry

end NUMINAMATH_CALUDE_total_siblings_weight_l3582_358231


namespace NUMINAMATH_CALUDE_insertPluses_l3582_358204

/-- The number of ones in the original number -/
def n : ℕ := 15

/-- The number of plus signs to be inserted -/
def k : ℕ := 9

/-- The number of spaces between the ones where plus signs can be inserted -/
def spaces : ℕ := n - 1

-- Statement of the theorem
theorem insertPluses : 
  (Nat.choose spaces k : ℕ) = (2002 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_insertPluses_l3582_358204


namespace NUMINAMATH_CALUDE_simplify_expression_l3582_358225

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3582_358225


namespace NUMINAMATH_CALUDE_area_triangle_DEF_area_triangle_DEF_is_six_l3582_358221

/-- Triangle DEF with vertices D, E, and F, where F lies on the line x + y = 6 -/
structure TriangleDEF where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_D : D = (2, 1)
  h_E : E = (1, 4)
  h_F : F.1 + F.2 = 6

/-- The area of triangle DEF is 6 -/
theorem area_triangle_DEF (t : TriangleDEF) : ℝ :=
  6

/-- The area of triangle DEF is indeed 6 -/
theorem area_triangle_DEF_is_six (t : TriangleDEF) :
  area_triangle_DEF t = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_area_triangle_DEF_is_six_l3582_358221


namespace NUMINAMATH_CALUDE_hoseok_flowers_left_l3582_358208

/-- Calculates the number of flowers Hoseok has left after giving some away. -/
def flowers_left (initial : ℕ) (to_minyoung : ℕ) (to_yoojeong : ℕ) : ℕ :=
  initial - (to_minyoung + to_yoojeong)

/-- Theorem stating that Hoseok has 7 flowers left after giving some away. -/
theorem hoseok_flowers_left :
  flowers_left 18 5 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_flowers_left_l3582_358208


namespace NUMINAMATH_CALUDE_root_product_theorem_l3582_358226

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) : 
  (x₁^6 - x₁^3 + 1 = 0) → 
  (x₂^6 - x₂^3 + 1 = 0) → 
  (x₃^6 - x₃^3 + 1 = 0) → 
  (x₄^6 - x₄^3 + 1 = 0) → 
  (x₅^6 - x₅^3 + 1 = 0) → 
  (x₆^6 - x₆^3 + 1 = 0) → 
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) * (x₆^2 - 3) = 757 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3582_358226


namespace NUMINAMATH_CALUDE_thomas_daniel_equation_l3582_358252

theorem thomas_daniel_equation (b c : ℝ) : 
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) → 
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_thomas_daniel_equation_l3582_358252


namespace NUMINAMATH_CALUDE_money_sharing_l3582_358288

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 60 →
  total = 260 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3582_358288


namespace NUMINAMATH_CALUDE_f_1_equals_5_l3582_358266

-- Define the quadratic polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
axiom quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
axiom f_2_3 : f 2 = 2 ∧ f 3 = 2
axiom g_2_3 : g 2 = 2 ∧ g 3 = 2
axiom g_1 : g 1 = 2
axiom f_5 : f 5 = 7
axiom g_5 : g 5 = 2

-- State the theorem
theorem f_1_equals_5 : f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_f_1_equals_5_l3582_358266


namespace NUMINAMATH_CALUDE_larry_dog_time_l3582_358292

/-- The number of minutes in half an hour -/
def half_hour : ℕ := 30

/-- The number of minutes spent feeding the dog daily -/
def feeding_time : ℕ := 12

/-- The total number of minutes Larry spends on his dog daily -/
def total_time : ℕ := 72

/-- The number of sessions Larry spends walking and playing with his dog daily -/
def walking_playing_sessions : ℕ := 2

theorem larry_dog_time :
  half_hour * walking_playing_sessions + feeding_time = total_time :=
sorry

end NUMINAMATH_CALUDE_larry_dog_time_l3582_358292


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3582_358205

theorem ancient_chinese_math_problem (a₁ : ℝ) : 
  (a₁ * (1 - (1/2)^6) / (1 - 1/2) = 378) →
  (a₁ * (1/2)^4 = 12) :=
by sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3582_358205


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l3582_358201

/-- If the simplest quadratic radical 2√(4m-1) is of the same type as √(2+3m), then m = 3. -/
theorem quadratic_radical_equality (m : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ k * (4 * m - 1) = 2 + 3 * m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l3582_358201


namespace NUMINAMATH_CALUDE_leakage_empty_time_l3582_358239

/-- Given a pipe that fills a tank in 'a' hours without leakage,
    and takes 7 times longer with leakage, the time taken by the leakage
    alone to empty the tank is (7a/6) hours. -/
theorem leakage_empty_time (a : ℝ) (h : a > 0) :
  let fill_time_with_leakage := 7 * a
  let fill_rate := 1 / a
  let combined_fill_rate := 1 / fill_time_with_leakage
  let leakage_rate := fill_rate - combined_fill_rate
  leakage_rate⁻¹ = 7 * a / 6 :=
by sorry

end NUMINAMATH_CALUDE_leakage_empty_time_l3582_358239


namespace NUMINAMATH_CALUDE_complex_division_result_l3582_358256

theorem complex_division_result : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l3582_358256


namespace NUMINAMATH_CALUDE_simplify_expression_l3582_358238

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3582_358238


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3582_358214

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 1 ∧ 3*x - 2*y = 7 ∧ x = 2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3582_358214


namespace NUMINAMATH_CALUDE_matrix_determinant_l3582_358291

theorem matrix_determinant (x y : ℝ) : 
  Matrix.det ![![x, x, y], ![x, y, x], ![y, x, x]] = 3 * x^2 * y - 2 * x^3 - y^3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3582_358291


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l3582_358253

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 1

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x : ℝ, (hyperbola m x (parabola x)) ∧
    (∀ x' : ℝ, x' ≠ x → ¬(hyperbola m x' (parabola x')))

theorem parabola_hyperbola_tangent :
  are_tangent 1 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l3582_358253


namespace NUMINAMATH_CALUDE_joanne_coins_l3582_358241

def coins_problem (first_hour : ℕ) (next_two_hours : ℕ) (fourth_hour : ℕ) (total_after : ℕ) : Prop :=
  let total_collected := first_hour + 2 * next_two_hours + fourth_hour
  total_collected - total_after = 15

theorem joanne_coins : coins_problem 15 35 50 120 := by
  sorry

end NUMINAMATH_CALUDE_joanne_coins_l3582_358241


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3582_358242

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -4/15 ∧ Q = -11/6 ∧ R = 31/10) ∧
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 5) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3582_358242


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3582_358299

/-- Given three real numbers forming an arithmetic sequence with sum 12,
    and their translations forming a geometric sequence,
    prove that the only solutions are (1, 4, 7) and (10, 4, -2) -/
theorem arithmetic_geometric_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (a + b + c = 12) →                  -- sum condition
  (∃ r : ℝ, (b + 2) = (a + 2) * r ∧ (c + 5) = (b + 2) * r) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3582_358299


namespace NUMINAMATH_CALUDE_jim_purchase_cost_l3582_358285

/-- The cost of a lamp in dollars -/
def lamp_cost : ℝ := 7

/-- The cost difference between a lamp and a bulb in dollars -/
def cost_difference : ℝ := 4

/-- The number of lamps bought -/
def num_lamps : ℕ := 2

/-- The number of bulbs bought -/
def num_bulbs : ℕ := 6

/-- The total cost of Jim's purchase -/
def total_cost : ℝ := num_lamps * lamp_cost + num_bulbs * (lamp_cost - cost_difference)

theorem jim_purchase_cost :
  total_cost = 32 := by sorry

end NUMINAMATH_CALUDE_jim_purchase_cost_l3582_358285


namespace NUMINAMATH_CALUDE_production_calculation_l3582_358251

-- Define the production rate for 6 machines
def production_rate_6 : ℕ := 300

-- Define the number of machines in the original setup
def original_machines : ℕ := 6

-- Define the number of machines in the new setup
def new_machines : ℕ := 10

-- Define the duration in minutes
def duration : ℕ := 4

-- Theorem to prove
theorem production_calculation :
  (new_machines * duration * production_rate_6) / original_machines = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_production_calculation_l3582_358251


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_perpendicular_l3582_358222

/-- Ellipse M with focal length 2√3 and eccentricity √3/2 -/
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Circle N with radius r -/
def circle_N (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- Tangent line l with slope k -/
def line_l (x y k m : ℝ) : Prop :=
  y = k * x + m

/-- P and Q are intersection points of line l and ellipse M -/
def intersection_points (P Q : ℝ × ℝ) (k m : ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  ellipse_M x₁ y₁ ∧ ellipse_M x₂ y₂ ∧
  line_l x₁ y₁ k m ∧ line_l x₂ y₂ k m

/-- OP and OQ are perpendicular -/
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_circle_tangent_perpendicular (k m r : ℝ) (P Q : ℝ × ℝ) :
  m^2 = r^2 * (k^2 + 1) →
  intersection_points P Q k m →
  (perpendicular P Q ↔ r = 2 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_perpendicular_l3582_358222


namespace NUMINAMATH_CALUDE_pm2_5_diameter_scientific_notation_l3582_358215

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The diameter of PM2.5 particulate matter in meters -/
def pm2_5_diameter : ℝ := 0.0000025

/-- The scientific notation representation of the PM2.5 diameter -/
def pm2_5_scientific : ScientificNotation :=
  { coefficient := 2.5
    exponent := -6
    valid := by sorry }

theorem pm2_5_diameter_scientific_notation :
  pm2_5_diameter = pm2_5_scientific.coefficient * (10 : ℝ) ^ pm2_5_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_pm2_5_diameter_scientific_notation_l3582_358215


namespace NUMINAMATH_CALUDE_max_min_sum_equals_22_5_l3582_358261

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the maximum value of xy + xz + yz plus 5 times the minimum value of xy + xz + yz equals 22.5 -/
theorem max_min_sum_equals_22_5 :
  ∃ (N n : ℝ),
    (∀ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 →
      x * y + x * z + y * z ≤ N ∧
      n ≤ x * y + x * z + y * z) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = N) ∧
    (∃ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 ∧ x * y + x * z + y * z = n) ∧
    N + 5 * n = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_equals_22_5_l3582_358261


namespace NUMINAMATH_CALUDE_reimu_win_probability_l3582_358224

/-- Represents the result of a single coin toss -/
inductive CoinSide
| Red
| Green

/-- Represents the state of a single coin -/
structure Coin :=
  (side1 : CoinSide)
  (side2 : CoinSide)

/-- Represents the game state -/
structure GameState :=
  (coins : Finset Coin)

/-- The number of coins in the game -/
def numCoins : Nat := 4

/-- A game is valid if it has the correct number of coins -/
def validGame (g : GameState) : Prop :=
  g.coins.card = numCoins

/-- The probability of Reimu winning the game -/
def reimuWinProbability (g : GameState) : ℚ :=
  sorry

/-- The main theorem: probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (g : GameState) (h : validGame g) : 
  reimuWinProbability g = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_reimu_win_probability_l3582_358224


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l3582_358230

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (para_line : Line → Line → Prop)
variable (para_line_plane : Line → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem only_one_correct_proposition :
  (¬(∀ (a b c : Line) (α : Plane), 
    subset a α → subset b α → perp c a → perp c b → perp_line_plane c α)) ∧
  (¬(∀ (a b : Line) (α : Plane),
    subset b α → para_line a b → para_line_plane a α)) ∧
  (¬(∀ (a b : Line) (α β : Plane),
    para_line_plane a α → intersect α β b → para_line a b)) ∧
  (∀ (a b : Line) (α : Plane),
    perp_line_plane a α → perp_line_plane b α → para_line a b) ∧
  (¬(∀ (a b c : Line) (α β : Plane),
    ((subset a α → subset b α → perp c a → perp c b → perp_line_plane c α) ∨
     (subset b α → para_line a b → para_line_plane a α) ∨
     (para_line_plane a α → intersect α β b → para_line a b)) ∧
    (perp_line_plane a α → perp_line_plane b α → para_line a b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l3582_358230


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3582_358211

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3582_358211


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l3582_358206

/-- A complex number z is pure imaginary if its real part is 0 and its imaginary part is not 0 -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_implies_x_equals_one :
  ∀ x : ℝ, IsPureImaginary ((x^2 - 1) + (x^2 + 3*x + 2)*I) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l3582_358206


namespace NUMINAMATH_CALUDE_blue_hat_cost_l3582_358269

theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 548 →
  green_hats = 38 →
  (total_price - green_hats * green_hat_cost) / (total_hats - green_hats) = 6 :=
by sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l3582_358269


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l3582_358240

def cube_side_length : ℕ := 13

def red_face_area : ℕ := 6 * cube_side_length^2

def total_face_area : ℕ := 6 * cube_side_length^3

def blue_face_area : ℕ := total_face_area - red_face_area

theorem blue_to_red_ratio :
  blue_face_area / red_face_area = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l3582_358240


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l3582_358255

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_first_eight : 
  geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l3582_358255


namespace NUMINAMATH_CALUDE_proportion_sum_l3582_358286

theorem proportion_sum (a b c d : ℚ) 
  (h1 : a/b = 3/2) 
  (h2 : c/d = 3/2) 
  (h3 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_proportion_sum_l3582_358286


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3582_358263

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3582_358263


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_m_part_II_l3582_358216

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Theorem for part (I)
theorem solution_set_part_I :
  {x : ℝ | f x ≥ 3 - |x - 2|} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} :=
sorry

-- Theorem for part (II)
theorem range_of_m_part_II :
  ∀ m : ℝ, (∃ x : ℝ, f x ≤ 2*m - |x + 4|) → m ≥ 7/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_m_part_II_l3582_358216


namespace NUMINAMATH_CALUDE_staircase_arrangement_count_l3582_358207

/-- The number of ways to arrange 3 people on a 7-step staircase --/
def arrangement_count : ℕ := 336

/-- The number of steps on the staircase --/
def num_steps : ℕ := 7

/-- The maximum number of people that can stand on a single step --/
def max_per_step : ℕ := 2

/-- The number of people to be arranged on the staircase --/
def num_people : ℕ := 3

/-- Theorem stating that the number of arrangements is 336 --/
theorem staircase_arrangement_count :
  arrangement_count = 336 :=
by sorry

end NUMINAMATH_CALUDE_staircase_arrangement_count_l3582_358207


namespace NUMINAMATH_CALUDE_paraboloid_surface_area_l3582_358296

/-- The paraboloid of revolution --/
def paraboloid (x y z : ℝ) : Prop := 3 * y = x^2 + z^2

/-- The bounding plane --/
def bounding_plane (y : ℝ) : Prop := y = 6

/-- The first octant --/
def first_octant (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

/-- The surface area of the part of the paraboloid --/
noncomputable def surface_area : ℝ := sorry

/-- The theorem stating the surface area of the specified part of the paraboloid --/
theorem paraboloid_surface_area :
  surface_area = 39 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_paraboloid_surface_area_l3582_358296


namespace NUMINAMATH_CALUDE_tony_additional_degrees_l3582_358200

/-- Represents the number of years Tony spent in school for various degrees -/
structure TonySchoolYears where
  science : ℕ
  physics : ℕ
  additional : ℕ
  total : ℕ

/-- Calculates the number of additional degrees Tony got -/
def additional_degrees (years : TonySchoolYears) : ℕ :=
  (years.total - years.science - years.physics) / years.science

/-- Theorem stating that Tony got 2 additional degrees -/
theorem tony_additional_degrees :
  ∀ (years : TonySchoolYears),
    years.science = 4 →
    years.physics = 2 →
    years.total = 14 →
    additional_degrees years = 2 := by
  sorry

#check tony_additional_degrees

end NUMINAMATH_CALUDE_tony_additional_degrees_l3582_358200


namespace NUMINAMATH_CALUDE_combination_equation_solution_l3582_358279

theorem combination_equation_solution (n : ℕ) : 
  Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l3582_358279


namespace NUMINAMATH_CALUDE_sandys_remaining_nickels_l3582_358244

/-- Given an initial number of nickels and a number of borrowed nickels,
    calculate the remaining nickels. -/
def remaining_nickels (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Sandy's remaining nickels is 11 -/
theorem sandys_remaining_nickels :
  remaining_nickels 31 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandys_remaining_nickels_l3582_358244


namespace NUMINAMATH_CALUDE_unique_perfect_square_l3582_358228

theorem unique_perfect_square (n : ℕ+) : n^2 - 19*n - 99 = m^2 ↔ n = 199 :=
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_l3582_358228


namespace NUMINAMATH_CALUDE_negative_two_x_plus_two_positive_l3582_358246

theorem negative_two_x_plus_two_positive (x : ℝ) : x < 1 → -2*x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_plus_two_positive_l3582_358246


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l3582_358297

/-- A square pyramid is a polyhedron with a square base and four triangular faces. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat

/-- The sum of faces, edges, and vertices for a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l3582_358297


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3582_358237

theorem more_girls_than_boys :
  ∀ (girls boys : ℕ),
    girls > boys →
    girls + boys = 41 →
    girls = 22 →
    girls - boys = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3582_358237


namespace NUMINAMATH_CALUDE_bat_ball_cost_difference_l3582_358217

theorem bat_ball_cost_difference (bat_cost ball_cost : ℕ) : 
  (2 * bat_cost + 3 * ball_cost = 1300) →
  (3 * bat_cost + 2 * ball_cost = 1200) →
  (ball_cost - bat_cost = 100) := by
sorry

end NUMINAMATH_CALUDE_bat_ball_cost_difference_l3582_358217


namespace NUMINAMATH_CALUDE_total_commute_time_is_19_point_1_l3582_358280

/-- Represents the commute schedule for a week --/
structure CommuteSchedule where
  normalWalkTime : ℝ
  normalBikeTime : ℝ
  wednesdayExtraTime : ℝ
  fridayExtraTime : ℝ
  rainIncreaseFactor : ℝ
  mondayIsWalking : Bool
  tuesdayIsBiking : Bool
  wednesdayIsWalking : Bool
  thursdayIsWalking : Bool
  fridayIsBiking : Bool
  mondayIsRainy : Bool
  thursdayIsRainy : Bool

/-- Calculates the total commute time for a week given a schedule --/
def totalCommuteTime (schedule : CommuteSchedule) : ℝ :=
  let mondayTime := if schedule.mondayIsWalking then
    (if schedule.mondayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let tuesdayTime := if schedule.tuesdayIsBiking then schedule.normalBikeTime * 2
  else schedule.normalWalkTime * 2

  let wednesdayTime := if schedule.wednesdayIsWalking then (schedule.normalWalkTime + schedule.wednesdayExtraTime) * 2
  else schedule.normalBikeTime * 2

  let thursdayTime := if schedule.thursdayIsWalking then
    (if schedule.thursdayIsRainy then schedule.normalWalkTime * (1 + schedule.rainIncreaseFactor) else schedule.normalWalkTime) * 2
  else schedule.normalBikeTime * 2

  let fridayTime := if schedule.fridayIsBiking then (schedule.normalBikeTime + schedule.fridayExtraTime) * 2
  else schedule.normalWalkTime * 2

  mondayTime + tuesdayTime + wednesdayTime + thursdayTime + fridayTime

/-- The main theorem stating that given the specific schedule, the total commute time is 19.1 hours --/
theorem total_commute_time_is_19_point_1 :
  let schedule : CommuteSchedule := {
    normalWalkTime := 2
    normalBikeTime := 1
    wednesdayExtraTime := 0.5
    fridayExtraTime := 0.25
    rainIncreaseFactor := 0.2
    mondayIsWalking := true
    tuesdayIsBiking := true
    wednesdayIsWalking := true
    thursdayIsWalking := true
    fridayIsBiking := true
    mondayIsRainy := true
    thursdayIsRainy := true
  }
  totalCommuteTime schedule = 19.1 := by sorry

end NUMINAMATH_CALUDE_total_commute_time_is_19_point_1_l3582_358280


namespace NUMINAMATH_CALUDE_different_orders_eq_120_l3582_358283

/-- The number of ways to arrange n elements. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of students who won awards. -/
def total_students : ℕ := 6

/-- The number of students whose order is fixed. -/
def fixed_order_students : ℕ := 3

/-- The number of different orders for all students to go on stage. -/
def different_orders : ℕ := permutations total_students / permutations fixed_order_students

theorem different_orders_eq_120 : different_orders = 120 := by
  sorry

end NUMINAMATH_CALUDE_different_orders_eq_120_l3582_358283


namespace NUMINAMATH_CALUDE_point_to_line_distance_l3582_358290

theorem point_to_line_distance (a : ℝ) : 
  (∃ d : ℝ, d = 4 ∧ d = |3*a - 4*6 - 2| / Real.sqrt (3^2 + (-4)^2)) →
  (a = 2 ∨ a = 46/3) :=
sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l3582_358290


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l3582_358272

def a : ℝ × ℝ := (-2, -1)

theorem vector_b_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10)
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l3582_358272


namespace NUMINAMATH_CALUDE_inequality_multiplication_l3582_358227

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l3582_358227


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3582_358293

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 136) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l3582_358293


namespace NUMINAMATH_CALUDE_max_product_sum_l3582_358289

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({5, 7, 9, 11} : Set ℕ) →
  g ∈ ({5, 7, 9, 11} : Set ℕ) →
  h ∈ ({5, 7, 9, 11} : Set ℕ) →
  j ∈ ({5, 7, 9, 11} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j : ℕ) ≤ 240 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l3582_358289


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3582_358248

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 + (a - 1) * x + 3

-- State the theorem
theorem even_function_implies_a_equals_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3582_358248


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l3582_358235

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) (n : ℝ) : Prop := x^2/3 - y^2/n = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define a predicate for a point being a focus of the hyperbola
def is_hyperbola_focus (x y : ℝ) (n : ℝ) : Prop :=
  hyperbola x y n ∧ x^2 - y^2 = 3 + n

-- State the theorem
theorem parabola_hyperbola_focus (n : ℝ) :
  (∃ x y, is_hyperbola_focus x y n ∧ (x, y) = parabola_focus) → n = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l3582_358235


namespace NUMINAMATH_CALUDE_alyssa_kittens_l3582_358298

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave away -/
def kittens_given_away : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - kittens_given_away

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_kittens_l3582_358298


namespace NUMINAMATH_CALUDE_sixth_term_is_twelve_l3582_358274

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 10
  fifth_term : a + 4*d = 10

/-- The sixth term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a + 5*seq.d = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_twelve_l3582_358274


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l3582_358259

theorem point_movement_on_number_line :
  let start : ℤ := 0
  let move_right : ℤ := 2
  let move_left : ℤ := 8
  let final_position : ℤ := start + move_right - move_left
  final_position = -6 := by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l3582_358259


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l3582_358220

theorem smallest_divisible_by_10_and_24 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l3582_358220


namespace NUMINAMATH_CALUDE_oak_trees_cut_down_problem_l3582_358229

/-- The number of oak trees cut down in a park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given 9 initial oak trees and 7 remaining after cutting, 2 oak trees were cut down --/
theorem oak_trees_cut_down_problem : oak_trees_cut_down 9 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_cut_down_problem_l3582_358229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3582_358247

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement of the problem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 + 3 * (a 3) - 18 = 0 →
  (a 8)^2 + 3 * (a 8) - 18 = 0 →
  a 5 + a 6 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3582_358247


namespace NUMINAMATH_CALUDE_A_subset_B_iff_a_geq_2_plus_sqrt5_l3582_358250

/-- Set A defined as a circle with center (2,1) and radius 1 -/
def A : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- Set B defined by the condition 2|x-1| + |y-1| ≤ a -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 2 * |p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating that A is a subset of B if and only if a ≥ 2 + √5 -/
theorem A_subset_B_iff_a_geq_2_plus_sqrt5 :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_iff_a_geq_2_plus_sqrt5_l3582_358250


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l3582_358284

def g (x : ℝ) : ℝ := x^2 - x

theorem g_of_x_plus_3 : g (x + 3) = x^2 + 5*x + 6 := by sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l3582_358284


namespace NUMINAMATH_CALUDE_diagonal_length_count_l3582_358271

/-- Represents a quadrilateral ABCD with given side lengths and diagonal AC --/
structure Quadrilateral where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  ad : ℕ
  ac : ℕ

/-- Checks if the triangle inequality holds for a triangle with given side lengths --/
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the number of possible integer lengths for the diagonal --/
theorem diagonal_length_count (q : Quadrilateral) : 
  q.ab = 9 → q.bc = 11 → q.cd = 18 → q.ad = 14 →
  (∀ x : ℕ, 5 ≤ x → x ≤ 19 → 
    (q.ac = x → 
      triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (∀ x : ℕ, x < 5 ∨ x > 19 → 
    ¬(triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (Finset.range 15).card = 15 := by
  sorry

#check diagonal_length_count

end NUMINAMATH_CALUDE_diagonal_length_count_l3582_358271


namespace NUMINAMATH_CALUDE_base12_addition_l3582_358275

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its decimal (base 10) value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal (base 10) value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- The main theorem to prove --/
theorem base12_addition :
  base12ToDecimal [Digit12.D3, Digit12.C, Digit12.D5] +
  base12ToDecimal [Digit12.D2, Digit12.A, Digit12.B] =
  base12ToDecimal [Digit12.D6, Digit12.D3, Digit12.D4] := by
  sorry


end NUMINAMATH_CALUDE_base12_addition_l3582_358275


namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l3582_358218

/-- For any triangle, the sum of reciprocals of altitudes equals the sum of reciprocals of exradii, which equals the reciprocal of the inradius. -/
theorem triangle_reciprocal_sum (a b c h_a h_b h_c r_a r_b r_c r A p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ A > 0 ∧ p > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area : A = p * r)
  (h_altitude_a : h_a = 2 * A / a)
  (h_altitude_b : h_b = 2 * A / b)
  (h_altitude_c : h_c = 2 * A / c)
  (h_exradius_a : r_a = A / (p - a))
  (h_exradius_b : r_b = A / (p - b))
  (h_exradius_c : r_c = A / (p - c)) :
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r_a + 1 / r_b + 1 / r_c ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r := by sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l3582_358218


namespace NUMINAMATH_CALUDE_product_price_interval_l3582_358236

theorem product_price_interval (price : ℝ) 
  (h1 : price < 2000)
  (h2 : price > 1000)
  (h3 : price < 1500)
  (h4 : price > 1250)
  (h5 : price > 1375) :
  price ∈ Set.Ioo 1375 1500 := by
sorry

end NUMINAMATH_CALUDE_product_price_interval_l3582_358236


namespace NUMINAMATH_CALUDE_parade_tricycles_l3582_358245

theorem parade_tricycles :
  ∀ (w b t : ℕ),
    w + b + t = 10 →
    2 * b + 3 * t = 25 →
    t = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_parade_tricycles_l3582_358245


namespace NUMINAMATH_CALUDE_eggs_in_jar_l3582_358234

/-- The number of eggs left in a jar after some are removed -/
def eggs_left (original : ℕ) (removed : ℕ) : ℕ := original - removed

/-- Theorem: Given 27 original eggs and 7 removed eggs, 20 eggs are left -/
theorem eggs_in_jar : eggs_left 27 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_jar_l3582_358234


namespace NUMINAMATH_CALUDE_sin_pi_sixth_plus_tan_pi_third_l3582_358295

theorem sin_pi_sixth_plus_tan_pi_third :
  Real.sin (π / 6) + Real.tan (π / 3) = 1 / 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_plus_tan_pi_third_l3582_358295


namespace NUMINAMATH_CALUDE_base8_digit_product_8654_l3582_358268

/-- Convert a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8654₁₀ is 0 --/
theorem base8_digit_product_8654 :
  listProduct (toBase8 8654) = 0 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_8654_l3582_358268


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l3582_358257

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 8 minutes later than normal, is 32 minutes. -/
theorem bus_stop_walking_time : ∃ (T : ℝ), 
  (T > 0) ∧ 
  (4/5 * T + 8 = T) ∧ 
  (T = 32) := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l3582_358257


namespace NUMINAMATH_CALUDE_meal_cost_theorem_l3582_358270

theorem meal_cost_theorem (initial_people : ℕ) (additional_people : ℕ) (share_decrease : ℚ) :
  initial_people = 5 →
  additional_people = 3 →
  share_decrease = 15 →
  let total_people := initial_people + additional_people
  let total_cost := (initial_people * share_decrease * total_people) / (total_people - initial_people)
  total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_theorem_l3582_358270


namespace NUMINAMATH_CALUDE_experiment_success_probability_l3582_358219

-- Define the experiment setup
structure ExperimentSetup where
  box1_total : ℕ := 10
  box1_a : ℕ := 7
  box1_b : ℕ := 3
  box2_total : ℕ := 10
  box2_red : ℕ := 5
  box3_total : ℕ := 10
  box3_red : ℕ := 8

-- Define the probability of success
def probability_of_success (setup : ExperimentSetup) : ℚ :=
  let p1 := (setup.box1_a : ℚ) / setup.box1_total * setup.box2_red / setup.box2_total
  let p2 := (setup.box1_b : ℚ) / setup.box1_total * setup.box3_red / setup.box3_total
  p1 + p2

-- Theorem statement
theorem experiment_success_probability (setup : ExperimentSetup) :
  probability_of_success setup = 59 / 100 := by
  sorry

end NUMINAMATH_CALUDE_experiment_success_probability_l3582_358219


namespace NUMINAMATH_CALUDE_sum_of_powers_eight_l3582_358262

theorem sum_of_powers_eight (x : ℝ) : x^5 + x^5 + x^5 + x^5 = x^(17/3) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_eight_l3582_358262


namespace NUMINAMATH_CALUDE_sum_of_number_and_square_l3582_358212

theorem sum_of_number_and_square : 11 + 11^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_square_l3582_358212


namespace NUMINAMATH_CALUDE_square_root_of_3_plus_4i_l3582_358273

theorem square_root_of_3_plus_4i :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I ∧
  (-2 - Complex.I) ^ 2 = (3 : ℂ) + 4 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_of_3_plus_4i_l3582_358273


namespace NUMINAMATH_CALUDE_same_solution_k_value_l3582_358213

theorem same_solution_k_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 * k = 21 ↔ 5 * x + 3 = 0) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l3582_358213


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3582_358277

theorem complex_modulus_product : Complex.abs (4 - 3 * Complex.I) * Complex.abs (4 + 3 * Complex.I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3582_358277


namespace NUMINAMATH_CALUDE_abc_inequality_l3582_358282

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (hsum : a^2 + b^2 + c^2 = 9) : a * b * c + 1 > 3 * a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3582_358282


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l3582_358210

theorem binomial_expansion_theorem (y b : ℚ) (m : ℕ) : 
  (Nat.choose m 4 : ℚ) * y^(m-4) * b^4 = 210 →
  (Nat.choose m 5 : ℚ) * y^(m-5) * b^5 = 462 →
  (Nat.choose m 6 : ℚ) * y^(m-6) * b^6 = 792 →
  m = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l3582_358210


namespace NUMINAMATH_CALUDE_range_of_a_l3582_358243

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (h : ∀ x, x ∈ A → x ∈ B a) 
                   (h_not_nec : ∃ x, x ∈ B a ∧ x ∉ A) : 
  a > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3582_358243


namespace NUMINAMATH_CALUDE_bulbs_per_pack_l3582_358232

/-- The number of bulbs Sean needs to replace in each room --/
def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4

/-- The total number of bulbs Sean needs to replace in the rooms --/
def room_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

/-- The number of bulbs Sean needs to replace in the garage --/
def garage_bulbs : ℕ := room_bulbs / 2

/-- The total number of bulbs Sean needs to replace --/
def total_bulbs : ℕ := room_bulbs + garage_bulbs

/-- The number of packs Sean will buy --/
def num_packs : ℕ := 6

/-- Theorem: The number of bulbs in each pack is 2 --/
theorem bulbs_per_pack : total_bulbs / num_packs = 2 := by
  sorry

end NUMINAMATH_CALUDE_bulbs_per_pack_l3582_358232


namespace NUMINAMATH_CALUDE_horse_purchase_problem_l3582_358209

theorem horse_purchase_problem (cuirassier_total : ℝ) (dragoon_total : ℝ) (dragoon_extra : ℕ) (price_diff : ℝ) :
  cuirassier_total = 11250 ∧ 
  dragoon_total = 16000 ∧ 
  dragoon_extra = 15 ∧ 
  price_diff = 50 →
  ∃ (cuirassier_count dragoon_count : ℕ) (cuirassier_price dragoon_price : ℝ),
    cuirassier_count = 25 ∧
    dragoon_count = 40 ∧
    cuirassier_price = 450 ∧
    dragoon_price = 400 ∧
    cuirassier_count * cuirassier_price = cuirassier_total ∧
    dragoon_count * dragoon_price = dragoon_total ∧
    dragoon_count = cuirassier_count + dragoon_extra ∧
    cuirassier_price = dragoon_price + price_diff :=
by sorry

end NUMINAMATH_CALUDE_horse_purchase_problem_l3582_358209


namespace NUMINAMATH_CALUDE_relationship_abc_l3582_358287

theorem relationship_abc (a b c : ℝ) (ha : a = (0.4 : ℝ)^2) (hb : b = 2^(0.4 : ℝ)) (hc : c = Real.log 2 / Real.log 0.4) :
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3582_358287


namespace NUMINAMATH_CALUDE_value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l3582_358281

/-- Represents a non-negative integer as a list of its digits in reverse order -/
def Digits := List Nat

/-- Computes the value of a number from its digits -/
def value (d : Digits) : Nat :=
  d.enum.foldl (fun acc (i, digit) => acc + digit * 10^i) 0

/-- Computes the sum of digits -/
def digitSum (d : Digits) : Nat :=
  d.sum

/-- States that for any number, its value is congruent to its digit sum modulo 9 -/
theorem value_congr_digitSum_mod_nine (d : Digits) :
  value d ≡ digitSum d [MOD 9] := by
  sorry

/-- The main theorem: a number is divisible by 9 iff its digit sum is divisible by 9 -/
theorem divisible_by_nine_iff_digitSum_divisible_by_nine (d : Digits) :
  9 ∣ value d ↔ 9 ∣ digitSum d := by
  sorry

end NUMINAMATH_CALUDE_value_congr_digitSum_mod_nine_divisible_by_nine_iff_digitSum_divisible_by_nine_l3582_358281


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l3582_358258

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_equality_iff_a_in_range :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l3582_358258


namespace NUMINAMATH_CALUDE_morning_earnings_l3582_358294

/-- Represents the types of vehicles William washes --/
inductive VehicleType
  | NormalCar
  | BigSUV
  | Minivan

/-- Represents a customer's order --/
structure Order where
  vehicles : List VehicleType
  multipleVehicles : Bool

def basePrice (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 15
  | VehicleType.BigSUV => 25
  | VehicleType.Minivan => 20

def washTime (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 1
  | VehicleType.BigSUV => 2
  | VehicleType.Minivan => 1.5

def applyDiscount (price : ℚ) : ℚ :=
  price * (1 - 0.1)

def calculateOrderPrice (o : Order) : ℚ :=
  let baseTotal := (o.vehicles.map basePrice).sum
  if o.multipleVehicles then applyDiscount baseTotal else baseTotal

def morningOrders : List Order :=
  [
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.NormalCar,
                   VehicleType.BigSUV, VehicleType.BigSUV, VehicleType.Minivan],
      multipleVehicles := false },
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.BigSUV],
      multipleVehicles := true }
  ]

theorem morning_earnings :
  (morningOrders.map calculateOrderPrice).sum = 164.5 := by sorry

end NUMINAMATH_CALUDE_morning_earnings_l3582_358294


namespace NUMINAMATH_CALUDE_line_through_origin_and_point_l3582_358276

/-- A line passing through two points (0,0) and (-4,3) has the function expression y = -3/4 * x -/
theorem line_through_origin_and_point (x y : ℝ) : 
  (0 : ℝ) = 0 * x + y ∧ 3 = -4 * (-3/4) + y → y = -3/4 * x := by
sorry

end NUMINAMATH_CALUDE_line_through_origin_and_point_l3582_358276


namespace NUMINAMATH_CALUDE_janet_clarinet_lessons_l3582_358254

/-- Proves that Janet takes 3 hours of clarinet lessons per week -/
theorem janet_clarinet_lessons :
  let clarinet_hourly_rate : ℕ := 40
  let piano_hourly_rate : ℕ := 28
  let piano_hours_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let annual_cost_difference : ℕ := 1040
  ∃ (clarinet_hours_per_week : ℕ),
    clarinet_hours_per_week = 3 ∧
    weeks_per_year * piano_hourly_rate * piano_hours_per_week - 
    weeks_per_year * clarinet_hourly_rate * clarinet_hours_per_week = 
    annual_cost_difference :=
by
  sorry

end NUMINAMATH_CALUDE_janet_clarinet_lessons_l3582_358254


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3582_358233

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → 
  b = (1, -2) → 
  m • a + n • b = (9, -8) → 
  m - n = -3 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3582_358233


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_four_sqrt_five_l3582_358202

theorem sqrt_difference_equals_negative_four_sqrt_five :
  Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5) = -4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_four_sqrt_five_l3582_358202


namespace NUMINAMATH_CALUDE_kira_breakfast_time_l3582_358278

/-- Represents the time taken to cook a single item -/
def cook_time (quantity : ℕ) (time_per_item : ℕ) : ℕ := quantity * time_per_item

/-- Represents Kira's breakfast preparation -/
def kira_breakfast : Prop :=
  let sausage_time := cook_time 3 5
  let egg_time := cook_time 6 4
  let bread_time := cook_time 4 3
  let hash_brown_time := cook_time 2 7
  let bacon_time := cook_time 4 6
  sausage_time + egg_time + bread_time + hash_brown_time + bacon_time = 89

theorem kira_breakfast_time : kira_breakfast := by
  sorry

end NUMINAMATH_CALUDE_kira_breakfast_time_l3582_358278


namespace NUMINAMATH_CALUDE_fraction_simplification_l3582_358265

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3582_358265


namespace NUMINAMATH_CALUDE_square_of_999999999_has_8_zeros_l3582_358267

theorem square_of_999999999_has_8_zeros :
  let n : ℕ := 999999999
  ∃ m : ℕ, n^2 = m * 10^8 ∧ m % 10 ≠ 0 ∧ m ≥ 10^9 ∧ m < 10^10 :=
by sorry

end NUMINAMATH_CALUDE_square_of_999999999_has_8_zeros_l3582_358267


namespace NUMINAMATH_CALUDE_exponential_inequality_l3582_358260

theorem exponential_inequality (a x : ℝ) : 
  a > Real.log 2 - 1 → x > 0 → Real.exp x > x^2 - 2*a*x + 1 := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3582_358260
