import Mathlib

namespace NUMINAMATH_CALUDE_product_sum_squares_l1073_107394

theorem product_sum_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_l1073_107394


namespace NUMINAMATH_CALUDE_percentage_problem_l1073_107380

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 36 → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1073_107380


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1073_107305

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the length function
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem quadrilateral_perimeter (ABCD : Quadrilateral) :
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  perpendicular ABCD.D ABCD.C ABCD.B ABCD.C →
  length ABCD.A ABCD.B = 7 →
  length ABCD.D ABCD.C = 3 →
  length ABCD.B ABCD.C = 10 →
  length ABCD.A ABCD.C = 15 →
  perimeter ABCD = 20 + 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1073_107305


namespace NUMINAMATH_CALUDE_product_evaluation_l1073_107310

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1073_107310


namespace NUMINAMATH_CALUDE_solve_simple_interest_l1073_107354

def simple_interest_problem (principal : ℝ) (interest_paid : ℝ) : Prop :=
  ∃ (rate : ℝ),
    principal = 900 ∧
    interest_paid = 729 ∧
    rate > 0 ∧
    rate < 100 ∧
    interest_paid = (principal * rate * rate) / 100 ∧
    rate = 9

theorem solve_simple_interest :
  ∀ (principal interest_paid : ℝ),
    simple_interest_problem principal interest_paid :=
  sorry

end NUMINAMATH_CALUDE_solve_simple_interest_l1073_107354


namespace NUMINAMATH_CALUDE_g_2000_divisors_l1073_107332

/-- g(n) is the smallest power of 5 such that 1/g(n) has exactly n digits after the decimal point -/
def g (n : ℕ) : ℕ := 5^n

/-- The number of positive integer divisors of x -/
def num_divisors (x : ℕ) : ℕ := sorry

theorem g_2000_divisors : num_divisors (g 2000) = 2001 := by sorry

end NUMINAMATH_CALUDE_g_2000_divisors_l1073_107332


namespace NUMINAMATH_CALUDE_max_rectangle_area_garden_max_area_l1073_107379

/-- The maximum area of a rectangle with a fixed perimeter -/
theorem max_rectangle_area (p : ℝ) (h : p > 0) : 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ 
    ∀ l' w' : ℝ, l' > 0 → w' > 0 → 2 * (l' + w') = p → l * w ≥ l' * w') →
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ l * w = (p / 4) ^ 2 :=
by sorry

/-- The maximum area of a rectangle with perimeter 400 feet is 10000 square feet -/
theorem garden_max_area : 
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = 400 ∧ l * w = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_garden_max_area_l1073_107379


namespace NUMINAMATH_CALUDE_common_term_value_l1073_107381

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℝ  -- First term
  a₂ : ℝ  -- Second term

/-- Represents a geometric progression -/
structure GeometricProgression where
  g₁ : ℝ  -- First term
  g₂ : ℝ  -- Second term

/-- Given arithmetic and geometric progressions, if there exists a common term, it is 37/3 -/
theorem common_term_value (x : ℝ) (ap : ArithmeticProgression) (gp : GeometricProgression) 
  (h_ap : ap.a₁ = 2*x - 3 ∧ ap.a₂ = 5*x - 11)
  (h_gp : gp.g₁ = x + 1 ∧ gp.g₂ = 2*x + 3)
  (h_common : ∃ t : ℝ, (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
                       (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1))) :
  ∃ t : ℝ, t = 37/3 ∧ (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
               (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_common_term_value_l1073_107381


namespace NUMINAMATH_CALUDE_y_value_l1073_107387

theorem y_value : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1073_107387


namespace NUMINAMATH_CALUDE_binomial_product_theorem_l1073_107314

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the smallest prime number greater than 10
def smallest_prime_gt_10 : ℕ := 11

-- Theorem statement
theorem binomial_product_theorem :
  binomial 18 6 * smallest_prime_gt_10 = 80080 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_theorem_l1073_107314


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l1073_107398

theorem negation_of_forall_exp_gt_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l1073_107398


namespace NUMINAMATH_CALUDE_birds_joined_l1073_107385

theorem birds_joined (initial_birds : ℕ) (final_birds : ℕ) (initial_storks : ℕ) :
  let birds_joined := final_birds - initial_birds
  birds_joined = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_joined_l1073_107385


namespace NUMINAMATH_CALUDE_dans_potatoes_l1073_107322

/-- The number of potatoes Dan has after rabbits eat some -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

theorem dans_potatoes : remaining_potatoes 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_potatoes_l1073_107322


namespace NUMINAMATH_CALUDE_hollow_cube_5x5x5_l1073_107318

/-- The number of cubes needed for a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ :=
  6 * (n^2 - (n-2)^2) - 12 * (n-2)

/-- Theorem: A hollow cube with outer dimensions 5 * 5 * 5 requires 60 cubes -/
theorem hollow_cube_5x5x5 : hollow_cube_cubes 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hollow_cube_5x5x5_l1073_107318


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l1073_107325

theorem max_leftover_grapes (n : ℕ) : ∃ k : ℕ, n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l1073_107325


namespace NUMINAMATH_CALUDE_swan_population_l1073_107384

/-- The number of swans doubles every 2 years -/
def doubles_every_two_years (S : ℕ → ℕ) : Prop :=
  ∀ n, S (n + 2) = 2 * S n

/-- In 10 years, there will be 480 swans -/
def swans_in_ten_years (S : ℕ → ℕ) : Prop :=
  S 10 = 480

/-- The current number of swans -/
def current_swans : ℕ := 15

theorem swan_population (S : ℕ → ℕ) 
  (h1 : doubles_every_two_years S) 
  (h2 : swans_in_ten_years S) : 
  S 0 = current_swans := by
  sorry

end NUMINAMATH_CALUDE_swan_population_l1073_107384


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l1073_107331

theorem absolute_value_inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l1073_107331


namespace NUMINAMATH_CALUDE_xyz_sum_product_sqrt_l1073_107392

theorem xyz_sum_product_sqrt (x y z : ℝ) 
  (h1 : y + z = 16)
  (h2 : z + x = 17)
  (h3 : x + y = 18) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 1831.78125 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_sqrt_l1073_107392


namespace NUMINAMATH_CALUDE_probability_of_three_in_three_eighths_l1073_107324

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem probability_of_three_in_three_eighths :
  let decimal := decimal_representation 3 8
  let total_digits := decimal.length
  let count_threes := (decimal.filter (· = 3)).length
  (count_threes : ℚ) / total_digits = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_three_in_three_eighths_l1073_107324


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l1073_107396

theorem quadratic_root_implies_u (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 469) / 12)^2 + 25 * ((-25 - Real.sqrt 469) / 12) + u = 0) → 
  u = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l1073_107396


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_l1073_107321

/-- A right-angled triangle with sides in arithmetic progression and area 486 dm² has sides 27 dm, 36 dm, and 45 dm. -/
theorem right_triangle_arithmetic_sides (a b c : ℝ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (b - a = c - b) →  -- Sides in arithmetic progression
  (a * b / 2 = 486) →  -- Area of the triangle
  (a = 27 ∧ b = 36 ∧ c = 45) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_l1073_107321


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1073_107377

theorem perpendicular_vectors (m : ℚ) : 
  let a : ℚ × ℚ := (-2, m)
  let b : ℚ × ℚ := (-1, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → m = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1073_107377


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_two_l1073_107372

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent line at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem tangent_slope_at_point_two :
  tangent_slope 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_two_l1073_107372


namespace NUMINAMATH_CALUDE_total_combinations_eq_nine_l1073_107344

/-- The number of characters available to choose from. -/
def num_characters : ℕ := 3

/-- The number of cars available to choose from. -/
def num_cars : ℕ := 3

/-- The total number of possible combinations when choosing one character and one car. -/
def total_combinations : ℕ := num_characters * num_cars

/-- Theorem stating that the total number of combinations is 9. -/
theorem total_combinations_eq_nine : total_combinations = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_nine_l1073_107344


namespace NUMINAMATH_CALUDE_maruti_car_price_increase_l1073_107308

theorem maruti_car_price_increase (P S : ℝ) (x : ℝ) 
  (h1 : P > 0) (h2 : S > 0) : 
  (P * (1 + x / 100) * (S * 0.8) = P * S * 1.04) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_maruti_car_price_increase_l1073_107308


namespace NUMINAMATH_CALUDE_smallest_product_l1073_107304

def S : Finset Int := {-9, -7, -4, 2, 5, 7}

theorem smallest_product (a b : Int) :
  a ∈ S → b ∈ S → a * b ≥ -63 ∧ ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = -63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l1073_107304


namespace NUMINAMATH_CALUDE_tunnel_length_proof_l1073_107337

/-- Represents the scale of a map -/
structure MapScale where
  ratio : ℚ

/-- Represents a length on a map -/
structure MapLength where
  length : ℚ
  unit : String

/-- Represents an actual length in reality -/
structure ActualLength where
  length : ℚ
  unit : String

/-- Converts a MapLength to an ActualLength based on a given MapScale -/
def convertMapLengthToActual (scale : MapScale) (mapLength : MapLength) : ActualLength :=
  { length := mapLength.length * scale.ratio
    unit := "cm" }

/-- Converts centimeters to kilometers -/
def cmToKm (cm : ℚ) : ℚ :=
  cm / 100000

theorem tunnel_length_proof (scale : MapScale) (mapLength : MapLength) :
  scale.ratio = 38000 →
  mapLength.length = 7 →
  mapLength.unit = "cm" →
  let actualLength := convertMapLengthToActual scale mapLength
  cmToKm actualLength.length = 2.66 := by
    sorry

end NUMINAMATH_CALUDE_tunnel_length_proof_l1073_107337


namespace NUMINAMATH_CALUDE_investment_gain_percentage_l1073_107373

-- Define the initial investment
def initial_investment : ℝ := 100

-- Define the first year loss percentage
def first_year_loss_percent : ℝ := 10

-- Define the second year gain percentage
def second_year_gain_percent : ℝ := 25

-- Theorem to prove the overall gain percentage
theorem investment_gain_percentage :
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let second_year_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  let overall_gain_percent := (second_year_amount - initial_investment) / initial_investment * 100
  overall_gain_percent = 12.5 := by
sorry

end NUMINAMATH_CALUDE_investment_gain_percentage_l1073_107373


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1073_107390

/-- Given a geometric sequence {a_n} with first three terms a-1, a+1, a+2, 
    prove that its general formula is a_n = -1/(2^(n-3)) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 →
  a_n 2 = a + 1 →
  a_n 3 = a + 2 →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  ∀ n : ℕ, n ≥ 1 → a_n n = -1 / (2^(n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1073_107390


namespace NUMINAMATH_CALUDE_sum_of_squares_l1073_107389

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + a * c = 72) (h2 : a + b + c = 14) :
  a^2 + b^2 + c^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1073_107389


namespace NUMINAMATH_CALUDE_trig_identities_l1073_107397

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α - Real.pi/4) = 1/3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 80/37) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1073_107397


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l1073_107307

/-- The number of blocks in the neighborhood -/
def num_blocks : ℕ := 16

/-- The number of junk mail pieces given to each house -/
def mail_per_house : ℕ := 4

/-- The total number of junk mail pieces given out -/
def total_mail : ℕ := 1088

/-- The number of houses in each block -/
def houses_per_block : ℕ := 17

theorem junk_mail_distribution :
  houses_per_block * num_blocks * mail_per_house = total_mail :=
by sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l1073_107307


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_factorials_mod_100_l1073_107369

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials (n : ℕ) : ℕ := (List.range n).foldl (fun acc i => acc * factorial (i + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).reverse.takeWhile (· = 0) |>.length

theorem trailing_zeros_of_product_factorials_mod_100 :
  trailing_zeros (product_of_factorials 50) % 100 = 12 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_factorials_mod_100_l1073_107369


namespace NUMINAMATH_CALUDE_no_mem_is_veen_l1073_107351

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for Mem, En, and Veen
variable (Mem En Veen : U → Prop)

-- Theorem statement
theorem no_mem_is_veen 
  (h1 : ∀ x, Mem x → En x)  -- All Mems are Ens
  (h2 : ∀ x, En x → ¬Veen x)  -- No Ens are Veens
  : ∀ x, Mem x → ¬Veen x :=  -- No Mem is a Veen
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_no_mem_is_veen_l1073_107351


namespace NUMINAMATH_CALUDE_yard_area_l1073_107334

/-- The area of a rectangular yard with a square cut-out --/
theorem yard_area (length width cut_side : ℕ) 
  (h1 : length = 20) 
  (h2 : width = 16) 
  (h3 : cut_side = 4) : 
  length * width - cut_side * cut_side = 304 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l1073_107334


namespace NUMINAMATH_CALUDE_three_isosceles_triangles_l1073_107365

-- Define a point on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define a triangle on the grid
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d12 := (t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2
  let d23 := (t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2
  let d31 := (t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

-- Define the five triangles
def triangle1 : Triangle := { v1 := {x := 0, y := 7}, v2 := {x := 2, y := 7}, v3 := {x := 1, y := 5} }
def triangle2 : Triangle := { v1 := {x := 4, y := 3}, v2 := {x := 4, y := 5}, v3 := {x := 6, y := 3} }
def triangle3 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 3, y := 3}, v3 := {x := 6, y := 2} }
def triangle4 : Triangle := { v1 := {x := 1, y := 1}, v2 := {x := 0, y := 3}, v3 := {x := 3, y := 1} }
def triangle5 : Triangle := { v1 := {x := 3, y := 6}, v2 := {x := 4, y := 4}, v3 := {x := 5, y := 7} }

-- Theorem statement
theorem three_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  ¬(isIsosceles triangle4) ∧
  ¬(isIsosceles triangle5) := by
  sorry

end NUMINAMATH_CALUDE_three_isosceles_triangles_l1073_107365


namespace NUMINAMATH_CALUDE_money_market_investment_ratio_l1073_107378

def initial_amount : ℚ := 25
def amount_to_mom : ℚ := 8
def num_items : ℕ := 5
def item_cost : ℚ := 1/2
def final_amount : ℚ := 6

theorem money_market_investment_ratio :
  let remaining_after_mom := initial_amount - amount_to_mom
  let spent_on_items := num_items * item_cost
  let before_investment := remaining_after_mom - spent_on_items
  let invested := before_investment - final_amount
  (invested : ℚ) / remaining_after_mom = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_money_market_investment_ratio_l1073_107378


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1073_107300

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → 2^x + x - 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x + x - 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1073_107300


namespace NUMINAMATH_CALUDE_z_purely_imaginary_and_fourth_quadrant_l1073_107399

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem z_purely_imaginary_and_fourth_quadrant :
  (∃! m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ∧ m = 3) ∧
  (¬∃ m : ℝ, (z m).re > 0 ∧ (z m).im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_and_fourth_quadrant_l1073_107399


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1073_107327

-- Define the condition for factoring quadratic trinomials
def is_factorizable (p q m n : ℤ) : Prop :=
  q = m * n ∧ p = m + n

-- Theorem 1
theorem factorization_1 : ∀ x : ℤ, x^2 - 7*x + 12 = (x - 3) * (x - 4) :=
  sorry

-- Theorem 2
theorem factorization_2 : ∀ x y : ℤ, (x - y)^2 + 4*(x - y) + 3 = (x - y + 1) * (x - y + 3) :=
  sorry

-- Theorem 3
theorem factorization_3 : ∀ a b : ℤ, (a + b) * (a + b - 2) - 3 = (a + b - 3) * (a + b + 1) :=
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_l1073_107327


namespace NUMINAMATH_CALUDE_quadratic_bound_l1073_107343

theorem quadratic_bound (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) : 
  ∀ x : ℝ, |x| ≤ 1 → |2*a*x + b| ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_bound_l1073_107343


namespace NUMINAMATH_CALUDE_arithmetic_sum_formula_main_theorem_l1073_107345

-- Define the sum of an arithmetic sequence from 1 to n
def arithmeticSum (n : ℕ) : ℕ := n * (1 + n) / 2

-- Define the sum of the odd numbers from 1 to 69
def oddSum : ℕ := (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21 + 23 + 25 + 27 + 29 + 31 + 33 + 35 + 37 + 39 + 41 + 43 + 45 + 47 + 49 + 51 + 53 + 55 + 57 + 59 + 61 + 63 + 65 + 67 + 69)

-- Theorem stating the correctness of the arithmetic sum formula
theorem arithmetic_sum_formula (n : ℕ) : 
  (List.range n).sum = arithmeticSum n :=
by sorry

-- Given condition
axiom odd_sum_condition : 3 * oddSum = 3675

-- Main theorem to prove
theorem main_theorem (n : ℕ) :
  (List.range n).sum = n * (1 + n) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sum_formula_main_theorem_l1073_107345


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1073_107330

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1073_107330


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l1073_107391

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: If g(3i) = 0 and g(1+2i) = 0, then p + q + r + s = 39 -/
theorem polynomial_root_sum (p q r s : ℝ) : 
  g p q r s (3*I) = 0 → g p q r s (1 + 2*I) = 0 → p + q + r + s = 39 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l1073_107391


namespace NUMINAMATH_CALUDE_trapezoid_construction_l1073_107371

/-- Represents a trapezoid with sides a, b, c where a ∥ c -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_parallel_c : True  -- Represents the condition a ∥ c

/-- The condition that angle γ is twice as large as angle α -/
def angle_condition (t : Trapezoid) : Prop :=
  ∃ (α : ℝ), t.b * Real.sin (2 * α) = t.a - t.c

theorem trapezoid_construction (t : Trapezoid) 
  (h : angle_condition t) : 
  (t.b ≠ t.a - t.c → False) ∧
  (t.b = t.a - t.c → ∀ (ε : ℝ), ∃ (t' : Trapezoid), 
    t'.a = t.a ∧ t'.b = t.b ∧ t'.c = t.c ∧ 
    angle_condition t' ∧ t' ≠ t) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_construction_l1073_107371


namespace NUMINAMATH_CALUDE_maria_car_trip_l1073_107362

theorem maria_car_trip (D : ℝ) : 
  (D / 2 + (D / 2) / 4 + 150 = D) → D = 400 := by sorry

end NUMINAMATH_CALUDE_maria_car_trip_l1073_107362


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l1073_107313

theorem least_n_for_inequality : 
  (∀ n : ℕ, n > 0 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 → n ≥ 4) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l1073_107313


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l1073_107329

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (n^2 : ℝ)) ∧ 
  (∀ (n : ℕ), n > 10 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 1 / (n^2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l1073_107329


namespace NUMINAMATH_CALUDE_coffee_mixture_price_l1073_107347

/-- The price of the second type of coffee bean -/
def second_coffee_price : ℝ := 36

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 11.25

/-- The price of the first type of coffee bean per pound -/
def first_coffee_price : ℝ := 9

/-- The weight of each type of coffee bean in the mixture -/
def each_coffee_weight : ℝ := 25

theorem coffee_mixture_price :
  second_coffee_price * each_coffee_weight +
  first_coffee_price * each_coffee_weight =
  mixture_price * total_mixture_weight :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_price_l1073_107347


namespace NUMINAMATH_CALUDE_compound_interest_existence_l1073_107341

/-- Proves the existence of a principal amount and interest rate satisfying the compound interest conditions --/
theorem compound_interest_existence : ∃ (P r : ℝ), 
  P * (1 + r)^2 = 8840 ∧ P * (1 + r)^3 = 9261 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_existence_l1073_107341


namespace NUMINAMATH_CALUDE_two_distinct_roots_l1073_107311

/-- The equation has exactly two distinct real roots for x when p is in the specified range -/
theorem two_distinct_roots (p : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    Real.sqrt (2*p + 1 - x₁^2) + Real.sqrt (3*x₁ + p + 4) = Real.sqrt (x₁^2 + 9*x₁ + 3*p + 9) ∧
    Real.sqrt (2*p + 1 - x₂^2) + Real.sqrt (3*x₂ + p + 4) = Real.sqrt (x₂^2 + 9*x₂ + 3*p + 9)) ↔
  (-1/4 < p ∧ p ≤ 0) ∨ p ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l1073_107311


namespace NUMINAMATH_CALUDE_triplet_equality_l1073_107326

theorem triplet_equality (a b c : ℝ) :
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + a * c) →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triplet_equality_l1073_107326


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1073_107302

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1073_107302


namespace NUMINAMATH_CALUDE_pirate_catch_caravel_l1073_107306

/-- Represents the velocity of a ship in nautical miles per hour -/
structure Velocity where
  speed : ℝ
  angle : ℝ

/-- Represents the position of a ship -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculate the minimum speed required for the pirate ship to catch the caravel -/
def min_pirate_speed (initial_distance : ℝ) (caravel_velocity : Velocity) : ℝ :=
  sorry

theorem pirate_catch_caravel (initial_distance : ℝ) (caravel_velocity : Velocity) :
  initial_distance = 10 ∧
  caravel_velocity.speed = 12 ∧
  caravel_velocity.angle = -5 * π / 6 →
  min_pirate_speed initial_distance caravel_velocity = 6 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_pirate_catch_caravel_l1073_107306


namespace NUMINAMATH_CALUDE_coupon_discount_proof_l1073_107335

/-- Calculates the discount given the costs and final amount paid -/
def calculate_discount (magazine_cost pencil_cost final_amount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - final_amount

theorem coupon_discount_proof :
  let magazine_cost : ℚ := 85/100
  let pencil_cost : ℚ := 1/2
  let final_amount : ℚ := 1
  calculate_discount magazine_cost pencil_cost final_amount = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_coupon_discount_proof_l1073_107335


namespace NUMINAMATH_CALUDE_train_length_l1073_107339

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 → time_s = 20 → length_m = 1000 → 
  length_m = (speed_kmh * (5/18)) * time_s := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1073_107339


namespace NUMINAMATH_CALUDE_parabola_equation_l1073_107363

/-- A parabola with vertex at the origin -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = kx -/
  k : ℝ
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The condition that the focus is on the x-axis -/
def focus_on_x_axis (p : Parabola) : Prop :=
  p.focus.2 = 0

/-- The condition that a perpendicular line from the origin to a line passing through the focus has its foot at (2, 1) -/
def perpendicular_foot_condition (p : Parabola) : Prop :=
  ∃ (m : ℝ), m * p.focus.1 = p.focus.2 ∧ 2 * m = 1

/-- The theorem stating that if the two conditions are met, the parabola's equation is y^2 = 10x -/
theorem parabola_equation (p : Parabola) :
  focus_on_x_axis p → perpendicular_foot_condition p → p.k = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1073_107363


namespace NUMINAMATH_CALUDE_probability_AC_less_than_11_l1073_107356

-- Define the given lengths
def AB : ℝ := 10
def BC : ℝ := 6

-- Define the maximum length of AC
def AC_max : ℝ := 11

-- Define the angle α
def α : ℝ → Prop := λ x => 0 < x ∧ x < Real.pi / 2

-- Define the probability function
noncomputable def P : ℝ := (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63))

-- State the theorem
theorem probability_AC_less_than_11 :
  ∀ x, α x → P = (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63)) :=
by sorry

end NUMINAMATH_CALUDE_probability_AC_less_than_11_l1073_107356


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1073_107340

/-- Proves that (8-15i)/(3+4i) = -36/25 - 77/25*i -/
theorem complex_fraction_simplification :
  (8 - 15 * Complex.I) / (3 + 4 * Complex.I) = -36/25 - 77/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1073_107340


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1073_107352

/-- The area of the upper triangle formed by a diagonal line in a 20cm x 15cm rectangle, 
    where the diagonal starts from the corner of a 5cm x 5cm square within the rectangle. -/
theorem shaded_area_theorem (total_width total_height small_square_side : ℝ) 
  (hw : total_width = 20)
  (hh : total_height = 15)
  (hs : small_square_side = 5) : 
  let large_width := total_width - small_square_side
  let large_height := total_height
  let diagonal_slope := large_height / total_width
  let intersection_y := diagonal_slope * small_square_side
  let triangle_base := large_width
  let triangle_height := large_height - intersection_y
  triangle_base * triangle_height / 2 = 84.375 := by
  sorry

#eval (20 - 5) * (15 - 15 / 20 * 5) / 2

end NUMINAMATH_CALUDE_shaded_area_theorem_l1073_107352


namespace NUMINAMATH_CALUDE_friend_payment_ratio_l1073_107348

def james_meal : ℚ := 16
def friend_meal : ℚ := 14
def tip_percentage : ℚ := 20 / 100
def james_total_paid : ℚ := 21

def total_bill : ℚ := james_meal + friend_meal
def tip : ℚ := total_bill * tip_percentage
def total_bill_with_tip : ℚ := total_bill + tip
def james_share : ℚ := james_total_paid - tip
def friend_payment : ℚ := total_bill - james_share

theorem friend_payment_ratio :
  friend_payment / total_bill_with_tip = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_friend_payment_ratio_l1073_107348


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_260_l1073_107342

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n can be expressed as a sum of at least three distinct powers of 2 -/
def hasAtLeastThreeDistinctPowers (n : ℕ) : Prop := sorry

theorem least_sum_of_exponents_for_260 :
  ∀ k : ℕ, (hasAtLeastThreeDistinctPowers 260 ∧ sumOfExponents 260 = k) → k ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_260_l1073_107342


namespace NUMINAMATH_CALUDE_sum_less_than_addends_l1073_107319

theorem sum_less_than_addends : ∃ a b : ℝ, a + b < a ∧ a + b < b := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_l1073_107319


namespace NUMINAMATH_CALUDE_sanity_indeterminable_likely_vampire_l1073_107374

-- Define the types of beings
inductive Being
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define the claim
def claimsLostMind (b : Being) : Prop := true

-- Theorem 1: It's impossible to determine sanity from the claim
theorem sanity_indeterminable (b : Being) (claim : claimsLostMind b) :
  ¬ ∃ (state : MentalState), (b = Being.Human → state = MentalState.Sane) ∧
                             (b = Being.Vampire → state = MentalState.Sane) :=
sorry

-- Theorem 2: The being is most likely a vampire
theorem likely_vampire (b : Being) (claim : claimsLostMind b) :
  b = Being.Vampire :=
sorry

end NUMINAMATH_CALUDE_sanity_indeterminable_likely_vampire_l1073_107374


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1073_107338

theorem gcd_polynomial_and_multiple (y : ℤ) : 
  18090 ∣ y → 
  Int.gcd ((3*y + 5)*(6*y + 7)*(10*y + 3)*(5*y + 11)*(y + 7)) y = 8085 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1073_107338


namespace NUMINAMATH_CALUDE_function_equality_implies_a_value_l1073_107336

/-- The function f(x) = x -/
def f (x : ℝ) : ℝ := x

/-- The function g(x) = ax^2 - x, parameterized by a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

/-- The theorem stating that under given conditions, a = 3/2 -/
theorem function_equality_implies_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ * f x₂ = g a x₁ * g a x₂) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_value_l1073_107336


namespace NUMINAMATH_CALUDE_rectangle_area_l1073_107360

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1073_107360


namespace NUMINAMATH_CALUDE_multiple_exists_l1073_107349

theorem multiple_exists (n : ℕ) (S : Finset ℕ) : 
  S ⊆ Finset.range (2 * n + 1) →
  S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
sorry

end NUMINAMATH_CALUDE_multiple_exists_l1073_107349


namespace NUMINAMATH_CALUDE_cloth_profit_proof_l1073_107383

def cloth_problem (selling_price total_meters cost_price_per_meter : ℕ) : Prop :=
  let total_cost := total_meters * cost_price_per_meter
  let total_profit := selling_price - total_cost
  let profit_per_meter := total_profit / total_meters
  profit_per_meter = 5

theorem cloth_profit_proof :
  cloth_problem 8925 85 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_proof_l1073_107383


namespace NUMINAMATH_CALUDE_triple_sharp_of_30_l1073_107367

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem triple_sharp_of_30 : sharp (sharp (sharp 30)) = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_of_30_l1073_107367


namespace NUMINAMATH_CALUDE_complex_perpendicular_l1073_107309

theorem complex_perpendicular (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) :
  Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂) → z₁.re * z₂.re + z₁.im * z₂.im = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_perpendicular_l1073_107309


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l1073_107388

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift operation
def verticalShift (g : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ g x - c

-- Theorem statement
theorem graph_vertical_shift (x : ℝ) : 
  (verticalShift f 2) x = f x - 2 := by sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l1073_107388


namespace NUMINAMATH_CALUDE_kolya_is_wrong_l1073_107375

/-- Represents a statement about the number of pencils. -/
structure PencilStatement where
  blue : ℕ
  green : ℕ

/-- The box of colored pencils. -/
def pencil_box : PencilStatement := sorry

/-- Vasya's statement -/
def vasya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4

/-- Kolya's statement -/
def kolya_statement (box : PencilStatement) : Prop :=
  box.green ≥ 5

/-- Petya's statement -/
def petya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

/-- Misha's statement -/
def misha_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

/-- Three statements are true and one is false -/
axiom three_true_one_false :
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ ¬kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ ¬misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ ¬petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (¬vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box)

theorem kolya_is_wrong :
  ¬kolya_statement pencil_box ∧
  vasya_statement pencil_box ∧
  petya_statement pencil_box ∧
  misha_statement pencil_box :=
by sorry

end NUMINAMATH_CALUDE_kolya_is_wrong_l1073_107375


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l1073_107301

/-- The number of integer solutions to the equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃ (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧
  (∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l1073_107301


namespace NUMINAMATH_CALUDE_abs_x_plus_y_equals_three_l1073_107312

theorem abs_x_plus_y_equals_three (x y : ℝ) 
  (eq1 : |x| + 2*y = 2) 
  (eq2 : 2*|x| + y = 7) : 
  |x| + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_plus_y_equals_three_l1073_107312


namespace NUMINAMATH_CALUDE_add_twice_equals_thrice_l1073_107346

theorem add_twice_equals_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_add_twice_equals_thrice_l1073_107346


namespace NUMINAMATH_CALUDE_new_average_score_l1073_107316

theorem new_average_score (n : ℕ) (initial_avg : ℚ) (new_score : ℚ) :
  n = 9 →
  initial_avg = 80 →
  new_score = 100 →
  (n * initial_avg + new_score) / (n + 1) = 82 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_l1073_107316


namespace NUMINAMATH_CALUDE_annes_bowling_score_l1073_107395

theorem annes_bowling_score (annes_score bob_score : ℕ) : 
  annes_score = bob_score + 50 →
  (annes_score + bob_score) / 2 = 150 →
  annes_score = 175 := by
sorry

end NUMINAMATH_CALUDE_annes_bowling_score_l1073_107395


namespace NUMINAMATH_CALUDE_ranch_cows_l1073_107328

/-- The number of cows owned by We the People -/
def wtp_cows : ℕ := 17

/-- The number of cows owned by Happy Good Healthy Family -/
def hghf_cows : ℕ := 3 * wtp_cows + 2

/-- The total number of cows in the ranch -/
def total_cows : ℕ := wtp_cows + hghf_cows

theorem ranch_cows : total_cows = 70 := by
  sorry

end NUMINAMATH_CALUDE_ranch_cows_l1073_107328


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1073_107323

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_combinations : choose 13 4 = 715 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1073_107323


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l1073_107386

/-- If the function f(x) = x^2 - 2x + m has a minimum value of -2 on the interval [2, +∞),
    then m = -2. -/
theorem minimum_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^2 - 2*x + m) →
  (∀ x ≥ 2, f x ≥ -2) →
  (∃ x ≥ 2, f x = -2) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l1073_107386


namespace NUMINAMATH_CALUDE_race_problem_l1073_107359

/-- The race problem -/
theorem race_problem (john_speed steve_speed : ℝ) (duration : ℝ) (final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : duration = 28)
  (h4 : final_distance = 2) :
  john_speed * duration - steve_speed * duration - final_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l1073_107359


namespace NUMINAMATH_CALUDE_pizza_slices_l1073_107303

theorem pizza_slices (x : ℚ) 
  (half_eaten : x / 2 = x - x / 2)
  (third_of_remaining_eaten : x / 2 - (x / 2) / 3 = x / 2 - x / 6)
  (four_slices_left : x / 2 - x / 6 = 4) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1073_107303


namespace NUMINAMATH_CALUDE_peach_pies_l1073_107358

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 1 →
  blueberry_ratio = 4 →
  cherry_ratio = 3 →
  peach_ratio = 2 →
  (peach_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio) = 8 := by
  sorry

#check peach_pies

end NUMINAMATH_CALUDE_peach_pies_l1073_107358


namespace NUMINAMATH_CALUDE_sqrt_2023_between_40_and_45_l1073_107364

theorem sqrt_2023_between_40_and_45 : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_between_40_and_45_l1073_107364


namespace NUMINAMATH_CALUDE_two_color_cubes_count_l1073_107376

/-- Represents a cube with painted stripes -/
structure StripedCube where
  edge_length : ℕ
  stripe_count : ℕ

/-- Counts the number of smaller cubes with exactly two faces painted with different colors -/
def count_two_color_cubes (cube : StripedCube) : ℕ :=
  sorry

/-- Theorem stating the correct number of two-color cubes for a 6x6x6 cube with three stripes -/
theorem two_color_cubes_count (cube : StripedCube) :
  cube.edge_length = 6 ∧ cube.stripe_count = 3 →
  count_two_color_cubes cube = 12 :=
by sorry

end NUMINAMATH_CALUDE_two_color_cubes_count_l1073_107376


namespace NUMINAMATH_CALUDE_jeans_cost_l1073_107366

theorem jeans_cost (mary_sunglasses : ℕ) (mary_sunglasses_price : ℕ) (rose_shoes : ℕ) (rose_cards : ℕ) (rose_cards_price : ℕ) :
  mary_sunglasses = 2 →
  mary_sunglasses_price = 50 →
  rose_shoes = 150 →
  rose_cards = 2 →
  rose_cards_price = 25 →
  ∃ (jeans_cost : ℕ),
    mary_sunglasses * mary_sunglasses_price + jeans_cost =
    rose_shoes + rose_cards * rose_cards_price ∧
    jeans_cost = 100 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_l1073_107366


namespace NUMINAMATH_CALUDE_rhinos_count_l1073_107382

/-- The number of animals Erica saw during her safari --/
def total_animals : ℕ := 20

/-- The number of lions seen on Saturday --/
def lions : ℕ := 3

/-- The number of elephants seen on Saturday --/
def elephants : ℕ := 2

/-- The number of buffaloes seen on Sunday --/
def buffaloes : ℕ := 2

/-- The number of leopards seen on Sunday --/
def leopards : ℕ := 5

/-- The number of warthogs seen on Monday --/
def warthogs : ℕ := 3

/-- The number of rhinos seen on Monday --/
def rhinos : ℕ := total_animals - (lions + elephants + buffaloes + leopards + warthogs)

theorem rhinos_count : rhinos = 5 := by
  sorry

end NUMINAMATH_CALUDE_rhinos_count_l1073_107382


namespace NUMINAMATH_CALUDE_deposit_calculation_l1073_107333

theorem deposit_calculation (initial_deposit : ℚ) : 
  (initial_deposit - initial_deposit / 4 - (initial_deposit - initial_deposit / 4) * 4 / 9 - 640) = 3 / 20 * initial_deposit →
  initial_deposit = 2400 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1073_107333


namespace NUMINAMATH_CALUDE_problem_solution_l1073_107317

theorem problem_solution (x y : ℝ) : 
  x = 201 → x^3 * y - 2 * x^2 * y + x * y = 804000 → y = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1073_107317


namespace NUMINAMATH_CALUDE_set_membership_problem_l1073_107315

theorem set_membership_problem (n : ℕ) (x y z w : ℕ) 
  (hn : n ≥ 4)
  (hx : x ∈ Finset.range n)
  (hy : y ∈ Finset.range n)
  (hz : z ∈ Finset.range n)
  (hw : w ∈ Finset.range n)
  (hS : Set.Mem (x, y, z) S ∧ Set.Mem (z, w, x) S) :
  Set.Mem (y, z, w) S ∧ Set.Mem (x, y, w) S :=
by
  sorry
where
  X : Finset ℕ := Finset.range n
  S : Set (ℕ × ℕ × ℕ) := 
    {p | p.1 ∈ X ∧ p.2.1 ∈ X ∧ p.2.2 ∈ X ∧
      ((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∨
       (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∨
       (p.2.2 < p.1 ∧ p.1 < p.2.1)) ∧
      ¬((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∧
        (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∧
        (p.2.2 < p.1 ∧ p.1 < p.2.1))}

end NUMINAMATH_CALUDE_set_membership_problem_l1073_107315


namespace NUMINAMATH_CALUDE_min_value_triangle_sides_l1073_107370

/-- 
Given a triangle with side lengths x+10, x+5, and 4x, where the angle opposite to side 4x
is the largest angle, the minimum value of 4x - (x+5) is 5.
-/
theorem min_value_triangle_sides (x : ℝ) : 
  (x + 5 + 4*x > x + 10) ∧ 
  (x + 5 + x + 10 > 4*x) ∧ 
  (4*x + x + 10 > x + 5) ∧
  (4*x > x + 5) ∧ 
  (4*x > x + 10) →
  ∃ (y : ℝ), y ≥ x ∧ ∀ (z : ℝ), z ≥ x → 4*z - (z + 5) ≥ 4*y - (y + 5) ∧ 4*y - (y + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_triangle_sides_l1073_107370


namespace NUMINAMATH_CALUDE_part1_part2_l1073_107350

-- Define the function f(x) = x / (e^x)
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the function g(x) = f(x) - m
noncomputable def g (x m : ℝ) : ℝ := f x - m

-- Theorem for part 1
theorem part1 (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧
   ∀ x, x > 0 → g x m = 0 → x = x₁ ∨ x = x₂) →
  0 < m ∧ m < 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) :
  (∃! n : ℤ, (f n)^2 - a * f n > 0 ∧ ∀ x : ℝ, x > 0 → (f x)^2 - a * f x > 0 → ⌊x⌋ = n) →
  2 / Real.exp 2 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1073_107350


namespace NUMINAMATH_CALUDE_max_trio_sum_l1073_107357

/-- A trio is a set of three distinct integers where two are divisors or multiples of the third -/
def is_trio (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ((a ∣ c ∧ b ∣ c) ∨ (a ∣ b ∧ c ∣ b) ∨ (b ∣ a ∧ c ∣ a))

/-- The set of integers from 1 to 2002 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2002}

theorem max_trio_sum :
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → is_trio a b c →
    a + b + c ≤ 4004 ∧
    (a + b + c = 4004 ↔ c = 2002 ∧ a ∣ 2002 ∧ b = 2002 - a) :=
sorry

end NUMINAMATH_CALUDE_max_trio_sum_l1073_107357


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l1073_107353

theorem smallest_solution_quadratic (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) → y ≥ -10 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l1073_107353


namespace NUMINAMATH_CALUDE_polynomial_has_non_real_root_l1073_107393

def is_valid_polynomial (P : Polynomial ℝ) : Prop :=
  (P.degree ≥ 4) ∧
  (∀ i, P.coeff i ∈ ({-1, 0, 1} : Set ℝ)) ∧
  (P.eval 0 ≠ 0)

theorem polynomial_has_non_real_root (P : Polynomial ℝ) 
  (h : is_valid_polynomial P) : 
  ∃ z : ℂ, z.im ≠ 0 ∧ P.eval (z.re : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_non_real_root_l1073_107393


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l1073_107368

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l1073_107368


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l1073_107361

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (m k : ℕ),
    m < 10000 ∧ k < 100 ∧
    n = 100000 * (n / 100000) + 1000 * (n / 1000 % 100) + (n % 1000) ∧
    4 * n = k * 10000 + m ∧
    n = m * 100 + k

theorem six_digit_number_theorem :
  {n : ℕ | is_valid_number n} = {142857, 190476, 238095} :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l1073_107361


namespace NUMINAMATH_CALUDE_angle_equality_l1073_107320

theorem angle_equality (α : Real) : 
  0 ≤ α ∧ α < 2 * Real.pi ∧ 
  (Real.sin α = Real.sin (215 * Real.pi / 180)) ∧ 
  (Real.cos α = Real.cos (215 * Real.pi / 180)) →
  α = 235 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l1073_107320


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1073_107355

/-- Calculates the speed of a train given its length, time to cross a bridge, and total length of bridge and train. -/
theorem train_speed_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : crossing_time = 30) 
  (h3 : total_length = 245) : 
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1073_107355
