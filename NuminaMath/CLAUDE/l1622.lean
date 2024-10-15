import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_of_square_root_7_minus_3_squared_l1622_162215

theorem sqrt_of_square_root_7_minus_3_squared (x : ℝ) :
  Real.sqrt ((Real.sqrt 7 - 3) ^ 2) = 3 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_root_7_minus_3_squared_l1622_162215


namespace NUMINAMATH_CALUDE_add_five_sixteen_base7_l1622_162200

/-- Converts a base 7 number to decimal --/
def toDecimal (b₇ : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (d : ℕ) : ℕ := sorry

/-- Addition in base 7 --/
def addBase7 (a₇ b₇ : ℕ) : ℕ := 
  toBase7 (toDecimal a₇ + toDecimal b₇)

theorem add_five_sixteen_base7 : 
  addBase7 5 16 = 24 := by sorry

end NUMINAMATH_CALUDE_add_five_sixteen_base7_l1622_162200


namespace NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l1622_162263

theorem unique_solution_to_diophantine_equation :
  ∃! (x y z n : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ n ≥ 2 ∧
    z ≤ 5 * 2^(2*n) ∧
    x^(2*n + 1) - y^(2*n + 1) = x * y * z + 2^(2*n + 1) ∧
    x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l1622_162263


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_half_l1622_162297

/-- Given a positive real number a, if the equation x² - 2ax - 2a ln x = 0
    has a unique solution in the interval (0, +∞), then a = 1/2. -/
theorem unique_solution_implies_a_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, x > 0 ∧ x^2 - 2*a*x - 2*a*(Real.log x) = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_half_l1622_162297


namespace NUMINAMATH_CALUDE_sixth_result_proof_l1622_162286

theorem sixth_result_proof (total_results : ℕ) (first_group : ℕ) (last_group : ℕ)
  (total_average : ℚ) (first_average : ℚ) (last_average : ℚ)
  (h1 : total_results = 11)
  (h2 : first_group = 6)
  (h3 : last_group = 6)
  (h4 : total_average = 60)
  (h5 : first_average = 58)
  (h6 : last_average = 63) :
  ∃ (sixth_result : ℚ), sixth_result = 66 := by
sorry

end NUMINAMATH_CALUDE_sixth_result_proof_l1622_162286


namespace NUMINAMATH_CALUDE_inequality_proof_l1622_162223

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = 14) : a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1622_162223


namespace NUMINAMATH_CALUDE_jake_has_fewer_than_19_peaches_l1622_162245

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jill : ℕ
  jake : ℕ

/-- The given conditions -/
def peach_conditions (p : PeachCount) : Prop :=
  p.steven = 19 ∧
  p.jill = 6 ∧
  p.steven = p.jill + 13 ∧
  p.jake < p.steven

/-- Theorem: Jake has fewer than 19 peaches -/
theorem jake_has_fewer_than_19_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.jake < 19 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_fewer_than_19_peaches_l1622_162245


namespace NUMINAMATH_CALUDE_monkeys_on_different_ladders_l1622_162288

/-- Represents a ladder in the system -/
structure Ladder where
  id : Nat

/-- Represents a monkey in the system -/
structure Monkey where
  id : Nat
  currentLadder : Ladder

/-- Represents a rope connecting two ladders -/
structure Rope where
  ladder1 : Ladder
  ladder2 : Ladder
  height1 : Nat
  height2 : Nat

/-- Represents the state of the system -/
structure MonkeyLadderSystem where
  n : Nat
  ladders : List Ladder
  monkeys : List Monkey
  ropes : List Rope

/-- Predicate to check if all monkeys are on different ladders -/
def allMonkeysOnDifferentLadders (system : MonkeyLadderSystem) : Prop :=
  ∀ m1 m2 : Monkey, m1 ∈ system.monkeys → m2 ∈ system.monkeys → m1 ≠ m2 →
    m1.currentLadder ≠ m2.currentLadder

/-- The main theorem stating that all monkeys end up on different ladders -/
theorem monkeys_on_different_ladders (system : MonkeyLadderSystem) 
    (h1 : system.n > 0)
    (h2 : system.ladders.length = system.n)
    (h3 : system.monkeys.length = system.n)
    (h4 : ∀ m : Monkey, m ∈ system.monkeys → m.currentLadder ∈ system.ladders)
    (h5 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ∈ system.ladders ∧ r.ladder2 ∈ system.ladders)
    (h6 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ≠ r.ladder2)
    (h7 : ∀ r1 r2 : Rope, r1 ∈ system.ropes → r2 ∈ system.ropes → r1 ≠ r2 →
      (r1.ladder1 = r2.ladder1 → r1.height1 ≠ r2.height1) ∧
      (r1.ladder2 = r2.ladder2 → r1.height2 ≠ r2.height2))
    : allMonkeysOnDifferentLadders system :=
  sorry

end NUMINAMATH_CALUDE_monkeys_on_different_ladders_l1622_162288


namespace NUMINAMATH_CALUDE_solution_set_f_leq_15_max_a_for_inequality_l1622_162248

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the solution set of f(x) ≤ 15
theorem solution_set_f_leq_15 :
  {x : ℝ | f x ≤ 15} = Set.Icc (-8 : ℝ) 7 := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, -x^2 + a ≤ f x) ↔ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_15_max_a_for_inequality_l1622_162248


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1622_162208

theorem modular_inverse_of_5_mod_23 : ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  constructor
  · norm_num
  · norm_num

#eval (5 * 14) % 23  -- This should output 1

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l1622_162208


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9911_l1622_162261

theorem largest_prime_factor_of_9911 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 9911 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9911 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9911_l1622_162261


namespace NUMINAMATH_CALUDE_simple_interest_time_l1622_162290

/-- Proves that the time for simple interest is 3 years under given conditions -/
theorem simple_interest_time (P : ℝ) (r : ℝ) (compound_principal : ℝ) (compound_time : ℝ) : 
  P = 1400.0000000000014 →
  r = 0.10 →
  compound_principal = 4000 →
  compound_time = 2 →
  P * r * 3 = (compound_principal * ((1 + r) ^ compound_time - 1)) / 2 →
  3 = (((compound_principal * ((1 + r) ^ compound_time - 1)) / 2) / (P * r)) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_l1622_162290


namespace NUMINAMATH_CALUDE_store_revenue_l1622_162292

def shirt_price : ℝ := 10
def jeans_price : ℝ := 2 * shirt_price
def jacket_price : ℝ := 3 * jeans_price
def discount_rate : ℝ := 0.1

def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def num_jackets : ℕ := 15

def total_revenue : ℝ :=
  (num_shirts : ℝ) * shirt_price +
  (num_jeans : ℝ) * jeans_price +
  (num_jackets : ℝ) * jacket_price * (1 - discount_rate)

theorem store_revenue :
  total_revenue = 1210 := by sorry

end NUMINAMATH_CALUDE_store_revenue_l1622_162292


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1622_162204

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1622_162204


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1622_162285

theorem quadratic_factorization (m n : ℝ) : 
  (∃ (x : ℝ), x^2 - m*x + n = 0) ∧ 
  (3 : ℝ)^2 - m*(3 : ℝ) + n = 0 ∧ 
  (-4 : ℝ)^2 - m*(-4 : ℝ) + n = 0 →
  ∀ (x : ℝ), x^2 - m*x + n = (x - 3)*(x + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1622_162285


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l1622_162229

/-- If P(a,b) is in the second quadrant, then Q(-b,a-3) is in the third quadrant -/
theorem point_quadrant_relation (a b : ℝ) : 
  (a < 0 ∧ b > 0) → (-b < 0 ∧ a - 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l1622_162229


namespace NUMINAMATH_CALUDE_cubic_roots_proof_l1622_162236

theorem cubic_roots_proof (k : ℝ) (p q r : ℝ) : 
  (2 * p^3 + k * p^2 - 6 * p - 3 = 0) →
  (p = 3) →
  (p + q + r = 5) →
  (p * q * r = -6) →
  ({q, r} : Set ℝ) = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_proof_l1622_162236


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1622_162267

/-- The number of values of 'a' for which the line y = x + a passes through
    the vertex of the parabola y = x^2 + a^2 is exactly 2. -/
theorem line_passes_through_parabola_vertex :
  let line := λ (x a : ℝ) => x + a
  let parabola := λ (x a : ℝ) => x^2 + a^2
  let vertex := λ (a : ℝ) => (0, a^2)
  ∃! (s : Finset ℝ), (∀ a ∈ s, line 0 a = (vertex a).2) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1622_162267


namespace NUMINAMATH_CALUDE_tan_half_sum_l1622_162230

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_l1622_162230


namespace NUMINAMATH_CALUDE_profit_in_scientific_notation_l1622_162235

theorem profit_in_scientific_notation :
  (74.5 : ℝ) * 1000000000 = 7.45 * (10 : ℝ)^9 :=
by sorry

end NUMINAMATH_CALUDE_profit_in_scientific_notation_l1622_162235


namespace NUMINAMATH_CALUDE_doctor_lawyer_ratio_l1622_162239

theorem doctor_lawyer_ratio (total : ℕ) (avg_all avg_doc avg_law : ℚ) 
  (h_total : total = 50)
  (h_avg_all : avg_all = 50)
  (h_avg_doc : avg_doc = 45)
  (h_avg_law : avg_law = 60) :
  ∃ (num_doc num_law : ℕ),
    num_doc + num_law = total ∧
    (avg_doc * num_doc + avg_law * num_law : ℚ) / total = avg_all ∧
    2 * num_law = num_doc :=
by sorry

end NUMINAMATH_CALUDE_doctor_lawyer_ratio_l1622_162239


namespace NUMINAMATH_CALUDE_range_of_x_l1622_162243

theorem range_of_x (x : ℝ) : 
  (0 ≤ x ∧ x < 2 * Real.pi) → 
  (Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) →
  (x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1622_162243


namespace NUMINAMATH_CALUDE_f_min_value_h_unique_zero_l1622_162219

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x
noncomputable def h (x : ℝ) : ℝ := g x - f (-1) x

-- Theorem for part 1
theorem f_min_value (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ IsLocalMin (f a) x ∧ f a x = a - a * Real.log a :=
sorry

-- Theorem for part 2
theorem h_unique_zero :
  ∃! (x : ℝ), x ∈ Set.Ioo 0 1 ∧ h x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_h_unique_zero_l1622_162219


namespace NUMINAMATH_CALUDE_computer_price_is_150_l1622_162210

/-- The price per computer in a factory with given production and earnings -/
def price_per_computer (daily_production : ℕ) (weekly_earnings : ℕ) : ℚ :=
  weekly_earnings / (daily_production * 7)

/-- Theorem stating that the price per computer is $150 -/
theorem computer_price_is_150 :
  price_per_computer 1500 1575000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_is_150_l1622_162210


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1622_162217

theorem pythagorean_triple_divisibility (x y z : ℕ+) (h : x^2 + y^2 = z^2) :
  (3 ∣ x ∨ 3 ∣ y) ∧ (5 ∣ x ∨ 5 ∣ y ∨ 5 ∣ z) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1622_162217


namespace NUMINAMATH_CALUDE_abs_c_value_l1622_162247

def polynomial (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) : 
  polynomial a b c (3 - Complex.I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 129 :=
sorry

end NUMINAMATH_CALUDE_abs_c_value_l1622_162247


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achieved_l1622_162249

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 14400/29 := by
  sorry

theorem min_sum_squares_achieved (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  2*x₁ + 3*x₂ + 4*x₃ = 120 ∧ 
  x₁^2 + x₂^2 + x₃^2 < 14400/29 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achieved_l1622_162249


namespace NUMINAMATH_CALUDE_prob_not_blue_twelve_sided_die_l1622_162262

-- Define the die
structure Die :=
  (sides : ℕ)
  (red_faces : ℕ)
  (yellow_faces : ℕ)
  (green_faces : ℕ)
  (blue_faces : ℕ)

-- Define the specific die from the problem
def twelve_sided_die : Die :=
  { sides := 12
  , red_faces := 5
  , yellow_faces := 4
  , green_faces := 2
  , blue_faces := 1 }

-- Define the probability of not rolling a blue face
def prob_not_blue (d : Die) : ℚ :=
  (d.sides - d.blue_faces) / d.sides

-- Theorem statement
theorem prob_not_blue_twelve_sided_die :
  prob_not_blue twelve_sided_die = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_twelve_sided_die_l1622_162262


namespace NUMINAMATH_CALUDE_sabrina_profit_is_35_l1622_162213

def sabrina_profit (total_loaves : ℕ) (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) (production_cost : ℚ) : ℚ :=
  let morning_loaves : ℕ := (2 * total_loaves) / 3
  let morning_revenue : ℚ := morning_loaves * morning_price
  let afternoon_loaves : ℕ := (total_loaves - morning_loaves) / 2
  let afternoon_revenue : ℚ := afternoon_loaves * (afternoon_price_ratio * morning_price)
  let evening_loaves : ℕ := total_loaves - morning_loaves - afternoon_loaves
  let evening_revenue : ℚ := evening_loaves * evening_price
  let total_revenue : ℚ := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost : ℚ := total_loaves * production_cost
  total_revenue - total_cost

theorem sabrina_profit_is_35 :
  sabrina_profit 60 2 (1/4) 1 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_profit_is_35_l1622_162213


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_on_parabola_l1622_162221

/-- Given points A and B on the parabola y = -2x^2 forming an isosceles right triangle ABO 
    with O at the origin, prove that the length of OA (equal to OB) is √5 when a = 1. -/
theorem isosceles_right_triangle_on_parabola :
  ∀ (a : ℝ), 
  let A : ℝ × ℝ := (a, -2 * a^2)
  let B : ℝ × ℝ := (-a, -2 * a^2)
  let O : ℝ × ℝ := (0, 0)
  -- A and B are on the parabola y = -2x^2
  (A.2 = -2 * A.1^2 ∧ B.2 = -2 * B.1^2) →
  -- ABO is an isosceles right triangle with right angle at O
  (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)) →
  (A.1 - O.1)^2 + (A.2 - O.2)^2 + (B.1 - O.1)^2 + (B.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  -- When a = 1, the length of OA (equal to OB) is √5
  a = 1 → Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_on_parabola_l1622_162221


namespace NUMINAMATH_CALUDE_fraction_of_fraction_l1622_162274

theorem fraction_of_fraction : (5 / 12) / (3 / 4) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_l1622_162274


namespace NUMINAMATH_CALUDE_square_difference_minus_difference_l1622_162279

theorem square_difference_minus_difference (a b : ℤ) : 
  ((a + b)^2 - (a - b)^2) - (a - b) = 4*a*b - (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_minus_difference_l1622_162279


namespace NUMINAMATH_CALUDE_factor_theorem_quadratic_l1622_162268

theorem factor_theorem_quadratic (k : ℚ) : 
  (∀ m : ℚ, (m - 8) ∣ (m^2 - k*m - 24)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_quadratic_l1622_162268


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1622_162238

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1980) → (180 * ((n + 4) - 2) = 2700) := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1622_162238


namespace NUMINAMATH_CALUDE_line_equations_l1622_162214

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equations (l₁ : Line) (p : Point) :
  l₁.a = 2 ∧ l₁.b = 4 ∧ l₁.c = -1 ∧ p.x = 1 ∧ p.y = -2 →
  (∃ l₂ : Line, l₂.parallel l₁ ∧ l₂.contains p ∧ l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 3) ∧
  (∃ l₂ : Line, l₂.perpendicular l₁ ∧ l₂.contains p ∧ l₂.a = 2 ∧ l₂.b = -1 ∧ l₂.c = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l1622_162214


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_to_fraction_l1622_162295

theorem sqrt_mixed_number_to_fraction :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_to_fraction_l1622_162295


namespace NUMINAMATH_CALUDE_spherical_segment_height_l1622_162278

/-- The height of a spherical segment given a right-angled triangle inscribed in its base -/
theorem spherical_segment_height
  (S : ℝ) -- Area of the inscribed right-angled triangle
  (α : ℝ) -- Acute angle of the inscribed right-angled triangle
  (β : ℝ) -- Central angle of the segment's arc in axial section
  (h_S_pos : S > 0)
  (h_α_pos : 0 < α)
  (h_α_lt_pi_2 : α < π / 2)
  (h_β_pos : 0 < β)
  (h_β_lt_pi : β < π) :
  ∃ (height : ℝ), height = Real.sqrt (S / Real.sin (2 * α)) * Real.tan (β / 4) :=
sorry

end NUMINAMATH_CALUDE_spherical_segment_height_l1622_162278


namespace NUMINAMATH_CALUDE_sheila_mwf_hours_l1622_162253

/-- Represents Sheila's work schedule and earnings --/
structure SheilaWork where
  mwf_hours : ℕ  -- Hours worked on Monday, Wednesday, Friday
  tt_hours : ℕ   -- Hours worked on Tuesday, Thursday
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday --/
theorem sheila_mwf_hours (s : SheilaWork) 
  (h1 : s.tt_hours = 6)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : s.weekly_earnings = s.hourly_rate * (3 * s.mwf_hours + 2 * s.tt_hours)) :
  s.mwf_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_sheila_mwf_hours_l1622_162253


namespace NUMINAMATH_CALUDE_sparrows_to_cardinals_ratio_l1622_162252

/-- The number of cardinals Camille saw -/
def cardinals : ℕ := 3

/-- The number of robins Camille saw -/
def robins : ℕ := 4 * cardinals

/-- The number of blue jays Camille saw -/
def blue_jays : ℕ := 2 * cardinals

/-- The total number of birds Camille saw -/
def total_birds : ℕ := 31

/-- The number of sparrows Camille saw -/
def sparrows : ℕ := total_birds - (cardinals + robins + blue_jays)

theorem sparrows_to_cardinals_ratio :
  (sparrows : ℚ) / cardinals = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_sparrows_to_cardinals_ratio_l1622_162252


namespace NUMINAMATH_CALUDE_investment_value_after_one_year_l1622_162254

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_a_multiplier : ℝ := 2
def stock_b_multiplier : ℝ := 2
def stock_c_multiplier : ℝ := 0.5

theorem investment_value_after_one_year :
  let investment_per_stock := initial_investment / num_stocks
  let stock_a_value := investment_per_stock * stock_a_multiplier
  let stock_b_value := investment_per_stock * stock_b_multiplier
  let stock_c_value := investment_per_stock * stock_c_multiplier
  stock_a_value + stock_b_value + stock_c_value = 1350 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_after_one_year_l1622_162254


namespace NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l1622_162222

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h_r : r = 15) (h_θ : θ = 45 * π / 180) :
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := r^2 * Real.sin θ / 2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l1622_162222


namespace NUMINAMATH_CALUDE_max_excellent_videos_l1622_162277

/-- A micro-video with likes and expert score -/
structure MicroVideo where
  likes : ℕ
  expertScore : ℕ

/-- Determines if one video is not inferior to another -/
def notInferior (a b : MicroVideo) : Prop :=
  a.likes ≥ b.likes ∨ a.expertScore ≥ b.expertScore

/-- Determines if a video is excellent among a list of videos -/
def isExcellent (v : MicroVideo) (videos : List MicroVideo) : Prop :=
  ∀ u ∈ videos, notInferior v u

/-- The main theorem to prove -/
theorem max_excellent_videos (videos : List MicroVideo) 
  (h : videos.length = 5) :
  ∃ (excellentVideos : List MicroVideo), 
    excellentVideos.length ≤ 5 ∧ 
    ∀ v ∈ excellentVideos, isExcellent v videos ∧
    ∀ v ∈ videos, isExcellent v videos → v ∈ excellentVideos :=
  sorry

end NUMINAMATH_CALUDE_max_excellent_videos_l1622_162277


namespace NUMINAMATH_CALUDE_tobys_money_l1622_162209

/-- 
Proves that if Toby gives 1/7 of his money to each of his two brothers 
and is left with $245, then the initial amount of money he received was $343.
-/
theorem tobys_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 2 * (1 / 7)) = 245) → initial_amount = 343 := by
  sorry

end NUMINAMATH_CALUDE_tobys_money_l1622_162209


namespace NUMINAMATH_CALUDE_worker_efficiency_l1622_162234

/-- Given two workers A and B, where A is twice as efficient as B, and they complete a work together in 12 days, prove that A can complete the work alone in 18 days. -/
theorem worker_efficiency (work_rate_A work_rate_B : ℝ) (total_time : ℝ) :
  work_rate_A = 2 * work_rate_B →
  work_rate_A + work_rate_B = 1 / total_time →
  total_time = 12 →
  1 / work_rate_A = 18 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l1622_162234


namespace NUMINAMATH_CALUDE_sine_inequality_l1622_162272

theorem sine_inequality (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) ↔
  (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_l1622_162272


namespace NUMINAMATH_CALUDE_parabola_point_value_l1622_162216

theorem parabola_point_value (a b : ℝ) : 
  (a * (-2)^2 + b * (-2) + 5 = 9) → (2*a - b + 6 = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l1622_162216


namespace NUMINAMATH_CALUDE_marys_animal_count_l1622_162283

/-- The number of animals Mary thought were in the petting zoo -/
def marys_count (actual_count : ℕ) (double_counted : ℕ) (forgotten : ℕ) : ℕ :=
  actual_count + double_counted - forgotten

/-- Theorem stating that Mary thought there were 60 animals in the petting zoo -/
theorem marys_animal_count :
  marys_count 56 7 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_marys_animal_count_l1622_162283


namespace NUMINAMATH_CALUDE_line_no_dot_count_l1622_162201

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters containing both a dot and a straight line -/
def dot_and_line : ℕ := 13

/-- Represents the number of letters containing a dot but not a straight line -/
def dot_no_line : ℕ := 3

/-- Theorem stating that the number of letters containing a straight line but not a dot is 24 -/
theorem line_no_dot_count : 
  total_letters - (dot_and_line + dot_no_line) = 24 := by sorry

end NUMINAMATH_CALUDE_line_no_dot_count_l1622_162201


namespace NUMINAMATH_CALUDE_no_valid_placement_l1622_162241

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the piece types
inductive Piece
| Rook
| Knight
| Bishop

-- Define the placement function type
def Placement := Chessboard → Piece

-- Define the attack relations
def rook_attacks (a b : Chessboard) : Prop :=
  (a.1 = b.1 ∨ a.2 = b.2) ∧ a ≠ b

def knight_attacks (a b : Chessboard) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ abs (a.2 - b.2) = 2) ∨
  (abs (a.1 - b.1) = 2 ∧ abs (a.2 - b.2) = 1)

def bishop_attacks (a b : Chessboard) : Prop :=
  abs (a.1 - b.1) = abs (a.2 - b.2) ∧ a ≠ b

-- Define the validity of a placement
def valid_placement (p : Placement) : Prop :=
  ∀ a b : Chessboard,
    (p a = Piece.Rook ∧ rook_attacks a b → p b = Piece.Knight) ∧
    (p a = Piece.Knight ∧ knight_attacks a b → p b = Piece.Bishop) ∧
    (p a = Piece.Bishop ∧ bishop_attacks a b → p b = Piece.Rook)

-- Theorem statement
theorem no_valid_placement : ¬∃ p : Placement, valid_placement p :=
  sorry

end NUMINAMATH_CALUDE_no_valid_placement_l1622_162241


namespace NUMINAMATH_CALUDE_two_roots_condition_l1622_162232

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 - 2*x + a = 0

-- Define the condition for having two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Statement of the theorem
theorem two_roots_condition (a : ℝ) :
  has_two_distinct_roots a ↔ a < 1 :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l1622_162232


namespace NUMINAMATH_CALUDE_rebecca_hours_less_than_toby_l1622_162211

theorem rebecca_hours_less_than_toby (x : ℕ) : 
  x + (2 * x - 10) + 56 = 157 → 64 - 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_hours_less_than_toby_l1622_162211


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_bounds_l1622_162259

-- Define the area function S
noncomputable def S (α : Real) : Real :=
  let β := α / 2
  let r := Real.sqrt (1 / (2 * Real.tan β))
  let a := r * (1 + Real.sin β) / Real.cos β
  let b := if β ≤ Real.pi / 4 then r * Real.sin (2 * β) else r
  (b / a) ^ 2

-- State the theorem
theorem isosceles_triangle_area_bounds :
  ∀ α : Real, Real.pi / 3 ≤ α ∧ α ≤ 2 * Real.pi / 3 →
    (1 / 4 : Real) ≥ S α ∧ S α ≥ 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_bounds_l1622_162259


namespace NUMINAMATH_CALUDE_least_distinct_values_in_list_l1622_162282

/-- Given a list of 2520 positive integers with a unique mode occurring exactly 12 times,
    the least number of distinct values in the list is 229. -/
theorem least_distinct_values_in_list :
  ∀ (L : List ℕ+) (mode : ℕ+),
    L.length = 2520 →
    (∃! x, x ∈ L ∧ L.count x = 12) →
    (∃ x, x ∈ L ∧ L.count x = 12) →
    (∀ x, x ∈ L → L.count x ≤ 12) →
    L.toFinset.card ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_in_list_l1622_162282


namespace NUMINAMATH_CALUDE_expression_simplification_l1622_162287

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1622_162287


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1622_162242

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ 
  (∃ (n : ℕ), 3 * b + 4 = n^2) ∧
  (∀ (k : ℕ), k > 4 ∧ k < b → ¬∃ (m : ℕ), 3 * k + 4 = m^2) ∧
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1622_162242


namespace NUMINAMATH_CALUDE_only_set_b_is_right_triangle_l1622_162270

-- Define the sets of numbers
def set_a : List ℕ := [2, 3, 4]
def set_b : List ℕ := [3, 4, 5]
def set_c : List ℕ := [5, 6, 7]
def set_d : List ℕ := [7, 8, 9]

-- Define a function to check if a set of three numbers satisfies the Pythagorean theorem
def is_right_triangle (sides : List ℕ) : Prop :=
  match sides with
  | [a, b, c] => a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2
  | _ => False

-- Theorem statement
theorem only_set_b_is_right_triangle :
  ¬(is_right_triangle set_a) ∧
  (is_right_triangle set_b) ∧
  ¬(is_right_triangle set_c) ∧
  ¬(is_right_triangle set_d) :=
by sorry

end NUMINAMATH_CALUDE_only_set_b_is_right_triangle_l1622_162270


namespace NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l1622_162289

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_diameter_scientific_notation :
  toScientificNotation 0.00000012 = ScientificNotation.mk 1.2 (-7) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l1622_162289


namespace NUMINAMATH_CALUDE_sweets_expenditure_l1622_162203

theorem sweets_expenditure (initial_amount : ℝ) (amount_per_friend : ℝ) (num_friends : ℕ) :
  initial_amount = 10.50 →
  amount_per_friend = 3.40 →
  num_friends = 2 →
  initial_amount - (amount_per_friend * num_friends) = 3.70 :=
by sorry

end NUMINAMATH_CALUDE_sweets_expenditure_l1622_162203


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1622_162256

theorem abs_inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1622_162256


namespace NUMINAMATH_CALUDE_expression_simplification_l1622_162275

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  (x^2 + 2 / x^2) * (y^2 + 2 / y^2) + (x^2 - 2 / y^2) * (y^2 - 2 / x^2) = 2 + 8 / (x^2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1622_162275


namespace NUMINAMATH_CALUDE_frustum_area_relation_l1622_162293

/-- A frustum with base areas S₁ and S₂, and midsection area S₀ -/
structure Frustum where
  S₁ : ℝ
  S₂ : ℝ
  S₀ : ℝ
  h_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₀ > 0

theorem frustum_area_relation (f : Frustum) : 2 * Real.sqrt f.S₀ = Real.sqrt f.S₁ + Real.sqrt f.S₂ := by
  sorry

end NUMINAMATH_CALUDE_frustum_area_relation_l1622_162293


namespace NUMINAMATH_CALUDE_percentage_of_indian_women_l1622_162269

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (indian_men_percentage : ℚ) (indian_children_percentage : ℚ) (non_indian_percentage : ℚ)
  (h_total_men : total_men = 700)
  (h_total_women : total_women = 500)
  (h_total_children : total_children = 800)
  (h_indian_men : indian_men_percentage = 20 / 100)
  (h_indian_children : indian_children_percentage = 10 / 100)
  (h_non_indian : non_indian_percentage = 79 / 100) :
  (((1 - non_indian_percentage) * (total_men + total_women + total_children)
    - indian_men_percentage * total_men
    - indian_children_percentage * total_children)
   / total_women) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_women_l1622_162269


namespace NUMINAMATH_CALUDE_car_price_theorem_l1622_162233

def asking_price_proof (P : ℝ) : Prop :=
  let first_offer := (9/10) * P
  let second_offer := P - 320
  (first_offer - second_offer = 200) → (P = 1200)

theorem car_price_theorem :
  ∀ P : ℝ, asking_price_proof P :=
sorry

end NUMINAMATH_CALUDE_car_price_theorem_l1622_162233


namespace NUMINAMATH_CALUDE_double_inverse_g_10_l1622_162260

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := Real.sqrt y - 1

-- Theorem statement
theorem double_inverse_g_10 :
  g_inv (g_inv 10) = Real.sqrt (Real.sqrt 10 - 1) - 1 :=
by sorry

end NUMINAMATH_CALUDE_double_inverse_g_10_l1622_162260


namespace NUMINAMATH_CALUDE_target_hit_probability_l1622_162250

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1622_162250


namespace NUMINAMATH_CALUDE_new_person_weight_l1622_162231

/-- Given a group of 8 persons where replacing one person weighing 35 kg
    with a new person increases the average weight by 5 kg,
    prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1622_162231


namespace NUMINAMATH_CALUDE_nested_radical_value_l1622_162255

/-- The value of the infinite nested radical √(15 + √(15 + √(15 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt 15))))

/-- Theorem stating that the nested radical equals 5 -/
theorem nested_radical_value : nestedRadical = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1622_162255


namespace NUMINAMATH_CALUDE_tiffany_homework_problems_l1622_162228

/-- The total number of problems Tiffany had to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
                   (math_problems_per_page reading_problems_per_page science_problems_per_page history_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems_per_page

/-- Theorem stating that the total number of problems is 46 -/
theorem tiffany_homework_problems :
  total_problems 6 4 3 2 3 3 4 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_homework_problems_l1622_162228


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l1622_162212

/-- Given a triangle with vertices (2,3), (0,0), and (0,y) where y > 0,
    if the area of the triangle is 36 square units, then y = 39 -/
theorem third_vertex_coordinates (y : ℝ) (h1 : y > 0) : 
  (1/2 : ℝ) * |2 * (3 - y)| = 36 → y = 39 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l1622_162212


namespace NUMINAMATH_CALUDE_division_problem_l1622_162220

theorem division_problem (divisor quotient remainder dividend : ℕ) 
  (h1 : divisor = 21)
  (h2 : remainder = 7)
  (h3 : dividend = 301)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 14 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1622_162220


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l1622_162271

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l1622_162271


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1622_162226

theorem abs_sum_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1622_162226


namespace NUMINAMATH_CALUDE_part1_part2_l1622_162246

-- Definition of arithmetic sequence sum
def S (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * d

-- Part 1
theorem part1 : ∃! k : ℕ+, S (3/2) 1 (k^2) = (S (3/2) 1 k)^2 := by sorry

-- Part 2
theorem part2 : ∀ a1 d : ℚ, 
  (∀ k : ℕ+, S a1 d (k^2) = (S a1 d k)^2) ↔ 
  ((a1 = 0 ∧ d = 0) ∨ (a1 = 1 ∧ d = 0) ∨ (a1 = 1 ∧ d = 2)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1622_162246


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l1622_162257

/-- The ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to corn syrup in the sport formulation is three times that of standard -/
def sport_ratio_multiplier : ℚ := 3

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
theorem sport_to_standard_ratio : 
  (sport_corn_syrup / sport_ratio_multiplier / sport_water) / 
  (standard_ratio 0 / standard_ratio 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l1622_162257


namespace NUMINAMATH_CALUDE_xiaomas_calculation_l1622_162225

theorem xiaomas_calculation (square : ℤ) (h : 40 + square = 35) : 40 / square = -8 := by
  sorry

end NUMINAMATH_CALUDE_xiaomas_calculation_l1622_162225


namespace NUMINAMATH_CALUDE_largest_binomial_term_l1622_162237

theorem largest_binomial_term (n : ℕ) (x : ℝ) (h1 : n = 500) (h2 : x = 0.3) :
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k = 125 ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_term_l1622_162237


namespace NUMINAMATH_CALUDE_nancy_total_games_l1622_162294

/-- The number of games Nancy attended this month -/
def games_this_month : ℕ := 9

/-- The number of games Nancy attended last month -/
def games_last_month : ℕ := 8

/-- The number of games Nancy plans to attend next month -/
def games_next_month : ℕ := 7

/-- The total number of games Nancy would attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem nancy_total_games : total_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l1622_162294


namespace NUMINAMATH_CALUDE_photos_per_remaining_page_l1622_162299

theorem photos_per_remaining_page (total_photos : ℕ) (total_pages : ℕ) 
  (first_15_photos : ℕ) (next_15_photos : ℕ) (following_10_photos : ℕ) :
  total_photos = 500 →
  total_pages = 60 →
  first_15_photos = 3 →
  next_15_photos = 4 →
  following_10_photos = 5 →
  (17 : ℕ) = (total_photos - (15 * first_15_photos + 15 * next_15_photos + 10 * following_10_photos)) / (total_pages - 40) :=
by sorry

end NUMINAMATH_CALUDE_photos_per_remaining_page_l1622_162299


namespace NUMINAMATH_CALUDE_area_gray_quadrilateral_l1622_162276

/-- The Stomachion puzzle square --/
def stomachion_square : ℝ := 12

/-- The length of side AB in the gray quadrilateral --/
def side_AB : ℝ := 6

/-- The height of triangle ABD --/
def height_ABD : ℝ := 3

/-- The length of side BC in the gray quadrilateral --/
def side_BC : ℝ := 3

/-- The height of triangle BCD --/
def height_BCD : ℝ := 2

/-- The area of the gray quadrilateral ABCD in the Stomachion puzzle --/
theorem area_gray_quadrilateral : 
  (1/2 * side_AB * height_ABD) + (1/2 * side_BC * height_BCD) = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_gray_quadrilateral_l1622_162276


namespace NUMINAMATH_CALUDE_mean_median_difference_l1622_162264

theorem mean_median_difference (x : ℕ) : 
  let s := [x, x + 2, x + 4, x + 7, x + 27]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  let median := x + 4
  mean = median + 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1622_162264


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_14_4_l1622_162240

theorem floor_plus_self_eq_14_4 :
  ∃! r : ℝ, ⌊r⌋ + r = 14.4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_14_4_l1622_162240


namespace NUMINAMATH_CALUDE_other_sales_percentage_l1622_162273

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were not notebooks or markers is 32% -/
theorem other_sales_percentage :
  total_sales - (notebook_sales + marker_sales) = 32 := by sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l1622_162273


namespace NUMINAMATH_CALUDE_grid_paths_count_l1622_162296

/-- The number of paths from (0, 0) to (n, n) on an n × n grid,
    moving only 1 up or 1 right at a time -/
def gridPaths (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

/-- Theorem stating that the number of paths on an n × n grid
    from (0, 0) to (n, n), moving only 1 up or 1 right at a time,
    is equal to (2n choose n) -/
theorem grid_paths_count (n : ℕ) :
  gridPaths n = Nat.choose (2 * n) n := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_count_l1622_162296


namespace NUMINAMATH_CALUDE_line_only_count_l1622_162202

/-- Represents the alphabet with its properties -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  dot_only : ℕ
  has_dot_or_line : total = dot_and_line + dot_only + (total - (dot_and_line + dot_only))

/-- The specific alphabet from the problem -/
def problem_alphabet : Alphabet := {
  total := 50
  dot_and_line := 16
  dot_only := 4
  has_dot_or_line := by sorry
}

/-- The number of letters with a straight line but no dot -/
def line_only (a : Alphabet) : ℕ := a.total - (a.dot_and_line + a.dot_only)

/-- Theorem stating the result for the problem alphabet -/
theorem line_only_count : line_only problem_alphabet = 30 := by sorry

end NUMINAMATH_CALUDE_line_only_count_l1622_162202


namespace NUMINAMATH_CALUDE_twentyFourthDigitOfSum_l1622_162280

-- Define the decimal representation of a rational number
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

-- Define the sum of two decimal representations
def sumDecimalRepresentations (f g : ℕ → ℕ) : ℕ → ℕ := sorry

-- The main theorem
theorem twentyFourthDigitOfSum :
  let f := decimalRepresentation (1/9 : ℚ)
  let g := decimalRepresentation (1/4 : ℚ)
  let sum := sumDecimalRepresentations f g
  sum 24 = 1 := by sorry

end NUMINAMATH_CALUDE_twentyFourthDigitOfSum_l1622_162280


namespace NUMINAMATH_CALUDE_exponential_strictly_increasing_l1622_162251

theorem exponential_strictly_increasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → (2 : ℝ) ^ x₁ < (2 : ℝ) ^ x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_strictly_increasing_l1622_162251


namespace NUMINAMATH_CALUDE_contrapositive_relation_l1622_162227

theorem contrapositive_relation (p r s : Prop) :
  (¬p ↔ r) → (r → s) → (s ↔ (¬p → p)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_relation_l1622_162227


namespace NUMINAMATH_CALUDE_morks_tax_rate_l1622_162224

/-- Given the tax rates and income ratio of Mork and Mindy, prove Mork's tax rate --/
theorem morks_tax_rate (r : ℝ) : 
  (r * 1 + 0.3 * 4) / 5 = 0.32 → r = 0.4 := by sorry

end NUMINAMATH_CALUDE_morks_tax_rate_l1622_162224


namespace NUMINAMATH_CALUDE_homework_time_ratio_l1622_162258

theorem homework_time_ratio :
  ∀ (geog_time : ℝ) (sci_time : ℝ),
    geog_time > 0 →
    sci_time = (60 + geog_time) / 2 →
    60 + geog_time + sci_time = 135 →
    geog_time / 60 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_ratio_l1622_162258


namespace NUMINAMATH_CALUDE_compounded_ratio_is_two_to_one_l1622_162281

/-- The compounded ratio of three given ratios -/
def compounded_ratio (r1 r2 r3 : Rat × Rat) : Rat × Rat :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

/-- The given ratios -/
def ratio1 : Rat × Rat := (2, 3)
def ratio2 : Rat × Rat := (6, 11)
def ratio3 : Rat × Rat := (11, 2)

/-- The theorem stating that the compounded ratio of the given ratios is 2:1 -/
theorem compounded_ratio_is_two_to_one :
  compounded_ratio ratio1 ratio2 ratio3 = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_compounded_ratio_is_two_to_one_l1622_162281


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1622_162298

theorem polynomial_factorization (a b c : ℂ) :
  let ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)
  a^3 + b^3 + c^3 - 3*a*b*c = (a + b + c) * (a + ω*b + ω^2*c) * (a + ω^2*b + ω*c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1622_162298


namespace NUMINAMATH_CALUDE_discount_tax_equivalence_l1622_162291

theorem discount_tax_equivalence (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  let discounted_price := original_price * (1 - discount_rate)
  let taxed_price := original_price * (1 + tax_rate)
  discounted_price * (1 + tax_rate) = taxed_price * (1 - discount_rate) :=
by sorry

#check discount_tax_equivalence 90 0.2 0.06

end NUMINAMATH_CALUDE_discount_tax_equivalence_l1622_162291


namespace NUMINAMATH_CALUDE_line_mb_less_than_neg_one_l1622_162205

/-- A line passing through two points (0, 3) and (2, -1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point1 : m * 0 + b = 3
  point2 : m * 2 + b = -1

/-- Theorem stating that for a line passing through (0, 3) and (2, -1), mb < -1 -/
theorem line_mb_less_than_neg_one (l : Line) : l.m * l.b < -1 := by
  sorry


end NUMINAMATH_CALUDE_line_mb_less_than_neg_one_l1622_162205


namespace NUMINAMATH_CALUDE_house_distance_proof_l1622_162244

/-- Represents the position of a house on a straight street -/
structure HousePosition :=
  (position : ℝ)

/-- The distance between two houses -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

theorem house_distance_proof
  (A B V G : HousePosition)
  (h1 : distance A B = 600)
  (h2 : distance V G = 600)
  (h3 : distance A G = 3 * distance B V) :
  distance A G = 900 ∨ distance A G = 1800 := by
  sorry


end NUMINAMATH_CALUDE_house_distance_proof_l1622_162244


namespace NUMINAMATH_CALUDE_divisibility_by_480_l1622_162284

theorem divisibility_by_480 (a : ℤ) 
  (h1 : ¬ (4 ∣ a)) 
  (h2 : a % 10 = 4) : 
  480 ∣ (a * (a^2 - 1) * (a^2 - 4)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_480_l1622_162284


namespace NUMINAMATH_CALUDE_evie_shells_left_l1622_162207

/-- The number of shells Evie has left after collecting for 6 days and giving some to her brother -/
def shells_left (days : ℕ) (shells_per_day : ℕ) (shells_given : ℕ) : ℕ :=
  days * shells_per_day - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left 6 10 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_evie_shells_left_l1622_162207


namespace NUMINAMATH_CALUDE_xy_value_l1622_162206

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 40) : x * y = 85 / 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1622_162206


namespace NUMINAMATH_CALUDE_square_sum_value_l1622_162218

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1622_162218


namespace NUMINAMATH_CALUDE_line_point_sum_l1622_162266

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ 
  s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 = 
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- Theorem statement -/
theorem line_point_sum (r s : ℝ) : 
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_line_point_sum_l1622_162266


namespace NUMINAMATH_CALUDE_log_intersection_and_exponential_inequality_l1622_162265

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the inverse function of f (exponential function)
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem log_intersection_and_exponential_inequality :
  (∃! x : ℝ, f x = x - 1) ∧
  (∀ m n : ℝ, m < n → (g n - g m) / (n - m) > g ((m + n) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_intersection_and_exponential_inequality_l1622_162265
