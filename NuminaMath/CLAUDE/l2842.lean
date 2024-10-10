import Mathlib

namespace expression_simplification_l2842_284233

theorem expression_simplification (x : ℝ) : 
  ((7*x + 3) - 3*x*2)*5 + (5 - 2/2)*(8*x - 5) = 37*x - 5 := by
  sorry

end expression_simplification_l2842_284233


namespace clock_strike_duration_clock_strike_six_duration_l2842_284248

-- Define the clock striking behavior
def ClockStrike (strikes : ℕ) (duration : ℝ) : Prop :=
  strikes > 0 ∧ duration > 0 ∧ (strikes - 1) * (duration / (strikes - 1)) = duration

-- Theorem statement
theorem clock_strike_duration (strikes₁ strikes₂ : ℕ) (duration₁ : ℝ) :
  ClockStrike strikes₁ duration₁ →
  strikes₂ > strikes₁ →
  ClockStrike strikes₂ ((strikes₂ - 1) * (duration₁ / (strikes₁ - 1))) :=
by
  sorry

-- The specific problem instance
theorem clock_strike_six_duration :
  ClockStrike 3 12 → ClockStrike 6 30 :=
by
  sorry

end clock_strike_duration_clock_strike_six_duration_l2842_284248


namespace birth_rate_calculation_l2842_284226

/-- The average birth rate in the city (people per 2 seconds) -/
def average_birth_rate : ℝ := sorry

/-- The death rate in the city (people per 2 seconds) -/
def death_rate : ℝ := 2

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 172800

/-- The number of 2-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation :
  average_birth_rate = 6 :=
sorry

end birth_rate_calculation_l2842_284226


namespace cow_distribution_theorem_l2842_284292

/-- Represents the distribution of cows among four sons -/
structure CowDistribution where
  total : ℕ
  first_son : ℚ
  second_son : ℚ
  third_son : ℚ
  fourth_son : ℕ

/-- Theorem stating the total number of cows given the distribution -/
theorem cow_distribution_theorem (d : CowDistribution) :
  d.first_son = 1/3 ∧ 
  d.second_son = 1/5 ∧ 
  d.third_son = 1/6 ∧ 
  d.fourth_son = 12 ∧
  d.first_son + d.second_son + d.third_son + (d.fourth_son : ℚ) / d.total = 1 →
  d.total = 40 := by
  sorry

end cow_distribution_theorem_l2842_284292


namespace number_of_children_l2842_284264

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  total_pencils / pencils_per_child = 15 := by
  sorry

end number_of_children_l2842_284264


namespace willam_land_percentage_l2842_284261

/-- Given that farm tax is levied on 40% of cultivated land, prove that Mr. Willam's
    taxable land is 12.5% of the village's total taxable land. -/
theorem willam_land_percentage (total_tax : ℝ) (willam_tax : ℝ)
    (h1 : total_tax = 3840)
    (h2 : willam_tax = 480) :
    willam_tax / total_tax * 100 = 12.5 := by
  sorry

end willam_land_percentage_l2842_284261


namespace root_of_cubic_equation_l2842_284209

theorem root_of_cubic_equation :
  let x : ℝ := Real.sin (π / 14)
  (0 < x ∧ x < Real.pi / 13) ∧
  8 * x^3 - 4 * x^2 - 4 * x + 1 = 0 :=
by sorry

end root_of_cubic_equation_l2842_284209


namespace seventeen_in_sample_l2842_284266

/-- Represents a systematic sampling with equal intervals -/
structure SystematicSampling where
  start : ℕ
  interval : ℕ
  size : ℕ

/-- Checks if a number is included in the systematic sampling -/
def isInSample (s : SystematicSampling) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.size ∧ n = s.start + k * s.interval

/-- Theorem: Given a systematic sampling that includes 5, 23, and 29, it also includes 17 -/
theorem seventeen_in_sample (s : SystematicSampling) 
  (h5 : isInSample s 5) 
  (h23 : isInSample s 23) 
  (h29 : isInSample s 29) : 
  isInSample s 17 := by
  sorry

end seventeen_in_sample_l2842_284266


namespace binomial_expansion_properties_l2842_284257

/-- 
Given a binomial expansion $(ax^m + bx^n)^{12}$ with specific conditions,
this theorem proves properties about the constant term and the range of $\frac{a}{b}$.
-/
theorem binomial_expansion_properties 
  (a b : ℝ) (m n : ℤ) 
  (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : 2*m + n = 0) :
  (∃ (r : ℕ), r = 4 ∧ m*(12 - r) + n*r = 0) ∧ 
  (8/5 ≤ a/b ∧ a/b ≤ 9/4) := by
  sorry

end binomial_expansion_properties_l2842_284257


namespace rower_upstream_speed_l2842_284240

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that given a man's speed in still water is 33 kmph and his downstream speed is 41 kmph, 
    his upstream speed is 25 kmph -/
theorem rower_upstream_speed :
  let still_water_speed := (33 : ℝ)
  let downstream_speed := (41 : ℝ)
  upstream_speed still_water_speed downstream_speed = 25 := by
sorry

#eval upstream_speed 33 41

end rower_upstream_speed_l2842_284240


namespace fraction_simplification_l2842_284243

theorem fraction_simplification : (3 : ℚ) / 462 + (17 : ℚ) / 42 = (95 : ℚ) / 231 := by
  sorry

end fraction_simplification_l2842_284243


namespace largest_value_l2842_284256

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 1 - 0.1)
  (hb : b = 1 - 0.01)
  (hc : c = 1 - 0.001)
  (hd : d = 1 - 0.0001)
  (he : e = 1 - 0.00001) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by sorry

end largest_value_l2842_284256


namespace tan_difference_special_angle_l2842_284294

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end tan_difference_special_angle_l2842_284294


namespace fourth_term_of_geometric_sequence_l2842_284201

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n, a (n + 1) = a n * r

/-- The fourth term of a geometric sequence with first term 3 and third term 75 is 375. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℕ) (h : IsGeometricSequence a) 
    (h1 : a 1 = 3) (h3 : a 3 = 75) : a 4 = 375 := by
  sorry

end fourth_term_of_geometric_sequence_l2842_284201


namespace solution_set_theorem_inequality_theorem_l2842_284289

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_theorem :
  {x : ℝ | f (x - 1) + f (x + 3) ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for part II
theorem inequality_theorem (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end solution_set_theorem_inequality_theorem_l2842_284289


namespace tangent_triangle_area_l2842_284249

/-- The area of the triangle formed by the tangent line to y = log₂ x at (1, 0) and the axes -/
theorem tangent_triangle_area : 
  let f (x : ℝ) := Real.log x / Real.log 2
  let tangent_line (x : ℝ) := (1 / Real.log 2) * (x - 1)
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1 / Real.log 2
  let triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)
  triangle_area = 1 / (2 * Real.log 2) := by sorry

end tangent_triangle_area_l2842_284249


namespace x_fourth_minus_reciprocal_l2842_284295

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end x_fourth_minus_reciprocal_l2842_284295


namespace projection_vector_l2842_284218

/-- Given two plane vectors a and b, prove that the projection of b onto a is (-1, 2) -/
theorem projection_vector (a b : ℝ × ℝ) : 
  a = (1, -2) → b = (3, 4) → 
  (((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a) = (-1, 2) := by
  sorry

end projection_vector_l2842_284218


namespace mutually_inscribed_pentagons_exist_l2842_284238

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pentagon in a 2D plane -/
structure Pentagon where
  vertices : Fin 5 → Point

/-- Checks if a point lies on a line segment or its extension -/
def pointOnLineSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if two pentagons are mutually inscribed -/
def areMutuallyInscribed (p1 p2 : Pentagon) : Prop :=
  ∀ (i : Fin 5), 
    (pointOnLineSegment (p1.vertices i) (p2.vertices i) (p2.vertices ((i + 1) % 5))) ∧
    (pointOnLineSegment (p2.vertices i) (p1.vertices i) (p1.vertices ((i + 1) % 5)))

/-- Theorem: For any given pentagon, there exists another pentagon mutually inscribed with it -/
theorem mutually_inscribed_pentagons_exist (p : Pentagon) : 
  ∃ (q : Pentagon), areMutuallyInscribed p q := by sorry

end mutually_inscribed_pentagons_exist_l2842_284238


namespace max_cuboid_path_length_l2842_284278

noncomputable def max_path_length (a b c : ℝ) : ℝ :=
  4 * Real.sqrt (a^2 + b^2 + c^2) + 
  4 * Real.sqrt (max a b * max b c) + 
  min a (min b c) + 
  max a (max b c)

theorem max_cuboid_path_length :
  max_path_length 2 2 1 = 12 + 8 * Real.sqrt 2 + 3 := by
  sorry

end max_cuboid_path_length_l2842_284278


namespace distribution_difference_l2842_284293

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
sorry

end distribution_difference_l2842_284293


namespace complex_equation_sum_l2842_284223

theorem complex_equation_sum (a b : ℝ) : 
  (3 * b : ℂ) + (2 * a - 2) * I = 1 - I → a + b = 5/6 := by
  sorry

end complex_equation_sum_l2842_284223


namespace odometer_seven_count_l2842_284287

/-- A function that counts the number of sevens in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is six-digit -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem odometer_seven_count (n : ℕ) (h1 : is_six_digit n) (h2 : count_sevens n = 4) :
  count_sevens (n + 900) ≠ 1 := by sorry

end odometer_seven_count_l2842_284287


namespace total_filled_boxes_is_16_l2842_284251

/-- Represents the types of trading cards --/
inductive CardType
  | Magic
  | Rare
  | Common

/-- Represents the types of boxes --/
inductive BoxType
  | Small
  | Large

/-- Defines the capacity of each box type for each card type --/
def boxCapacity (b : BoxType) (c : CardType) : ℕ :=
  match b, c with
  | BoxType.Small, CardType.Magic => 5
  | BoxType.Small, CardType.Rare => 5
  | BoxType.Small, CardType.Common => 6
  | BoxType.Large, CardType.Magic => 10
  | BoxType.Large, CardType.Rare => 10
  | BoxType.Large, CardType.Common => 15

/-- Calculates the number of fully filled boxes of a given type for a specific card type --/
def filledBoxes (cardCount : ℕ) (b : BoxType) (c : CardType) : ℕ :=
  cardCount / boxCapacity b c

/-- The main theorem stating that the total number of fully filled boxes is 16 --/
theorem total_filled_boxes_is_16 :
  let magicCards := 33
  let rareCards := 28
  let commonCards := 33
  let smallBoxesMagic := filledBoxes magicCards BoxType.Small CardType.Magic
  let smallBoxesRare := filledBoxes rareCards BoxType.Small CardType.Rare
  let smallBoxesCommon := filledBoxes commonCards BoxType.Small CardType.Common
  smallBoxesMagic + smallBoxesRare + smallBoxesCommon = 16 :=
by
  sorry


end total_filled_boxes_is_16_l2842_284251


namespace gross_revenue_increase_l2842_284206

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.5)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 20) := by
  sorry

end gross_revenue_increase_l2842_284206


namespace yellow_crane_tower_visitor_l2842_284234

structure Person :=
  (name : String)
  (visited : Bool)
  (statement : Bool)

def A : Person := { name := "A", visited := false, statement := false }
def B : Person := { name := "B", visited := false, statement := false }
def C : Person := { name := "C", visited := false, statement := false }

def people : List Person := [A, B, C]

theorem yellow_crane_tower_visitor :
  (∃! p : Person, p.visited = true) →
  (∃! p : Person, p.statement = false) →
  (A.statement = (¬C.visited)) →
  (B.statement = B.visited) →
  (C.statement = A.statement) →
  A.visited = true :=
by sorry

end yellow_crane_tower_visitor_l2842_284234


namespace problem_solution_l2842_284212

theorem problem_solution (x y z : ℝ) 
  (h1 : (x + y)^2 + (y + z)^2 + (x + z)^2 = 94)
  (h2 : (x - y)^2 + (y - z)^2 + (x - z)^2 = 26) :
  (x * y + y * z + x * z = 17) ∧
  ((x + 2*y + 3*z)^2 + (y + 2*z + 3*x)^2 + (z + 2*x + 3*y)^2 = 794) := by
  sorry

end problem_solution_l2842_284212


namespace determinant_zero_l2842_284210

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by sorry

end determinant_zero_l2842_284210


namespace q_satisfies_conditions_l2842_284288

/-- The quadratic polynomial q(x) -/
def q (x : ℚ) : ℚ := (20/7) * x^2 - (60/7) * x - 360/7

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q (-1) = -40 := by sorry

end q_satisfies_conditions_l2842_284288


namespace sequence_sum_100_l2842_284283

/-- Sequence sum type -/
def SequenceSum (a : ℕ+ → ℝ) : ℕ+ → ℝ 
  | n => (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem -/
theorem sequence_sum_100 (a : ℕ+ → ℝ) (t : ℝ) : 
  (∀ n : ℕ+, a n > 0) → 
  a 1 = 1 → 
  (∀ n : ℕ+, 2 * SequenceSum a n = a n * (a n + t)) → 
  SequenceSum a 100 = 5050 := by
  sorry

end sequence_sum_100_l2842_284283


namespace quadratic_root_transformation_l2842_284285

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((p^2 * u + q)^2 - q * (p^2 * u + q) + p * r = 0) ∧
  ((p^2 * v + q)^2 - q * (p^2 * v + q) + p * r = 0) :=
by sorry

end quadratic_root_transformation_l2842_284285


namespace product_remainder_main_theorem_l2842_284263

theorem product_remainder (a b : Nat) : (a * b) % 9 = ((a % 9) * (b % 9)) % 9 := by sorry

theorem main_theorem : (98 * 102) % 9 = 3 := by
  -- The proof would go here, but we're only providing the statement
  sorry

end product_remainder_main_theorem_l2842_284263


namespace carlas_order_cost_l2842_284282

/-- The original cost of Carla's order at McDonald's -/
def original_cost : ℝ := 7.50

/-- The coupon value -/
def coupon_value : ℝ := 2.50

/-- The senior discount percentage -/
def senior_discount : ℝ := 0.20

/-- The final amount Carla pays -/
def final_payment : ℝ := 4.00

/-- Theorem stating that the original cost is correct given the conditions -/
theorem carlas_order_cost :
  (original_cost - coupon_value) * (1 - senior_discount) = final_payment :=
by sorry

end carlas_order_cost_l2842_284282


namespace smallest_divisible_by_12_and_60_l2842_284286

theorem smallest_divisible_by_12_and_60 : Nat.lcm 12 60 = 60 := by
  sorry

end smallest_divisible_by_12_and_60_l2842_284286


namespace complex_square_equals_negative_100_minus_64i_l2842_284279

theorem complex_square_equals_negative_100_minus_64i :
  ∀ z : ℂ, z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end complex_square_equals_negative_100_minus_64i_l2842_284279


namespace sqrt_221_between_14_and_15_l2842_284219

theorem sqrt_221_between_14_and_15 : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_between_14_and_15_l2842_284219


namespace complex_number_quadrant_l2842_284203

theorem complex_number_quadrant : ∃ (z : ℂ), z = (25 / (3 - 4*I)) * I ∧ (z.re < 0 ∧ z.im > 0) := by
  sorry

end complex_number_quadrant_l2842_284203


namespace unique_sequence_existence_l2842_284277

theorem unique_sequence_existence :
  ∃! (a : ℕ → ℕ), 
    a 1 = 1 ∧
    a 2 > 1 ∧
    ∀ n : ℕ, n ≥ 1 → 
      (a (n + 1) * (a (n + 1) - 1) : ℚ) = 
        (a n * a (n + 2) : ℚ) / ((a n * a (n + 2) - 1 : ℚ) ^ (1/3) + 1) - 1 := by
  sorry

end unique_sequence_existence_l2842_284277


namespace sin_2theta_value_l2842_284291

theorem sin_2theta_value (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = 2*Real.sqrt 2/3 := by
  sorry

end sin_2theta_value_l2842_284291


namespace sum_of_roots_cubic_l2842_284273

theorem sum_of_roots_cubic (x : ℝ) : 
  (x + 2)^2 * (x - 3) = 40 → 
  ∃ (r₁ r₂ r₃ : ℝ), r₁ + r₂ + r₃ = -1 ∧ 
    ((x = r₁) ∨ (x = r₂) ∨ (x = r₃)) :=
by sorry

end sum_of_roots_cubic_l2842_284273


namespace function_properties_l2842_284221

-- Define the function f(x) = k - 1/x
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k - 1/x

-- Theorem statement
theorem function_properties (k : ℝ) :
  -- 1. Domain of f
  (∀ x : ℝ, x ≠ 0 → f k x ∈ Set.univ) ∧
  -- 2. f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f k x < f k y) ∧
  -- 3. If f is odd, then k = 0
  ((∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) → k = 0) :=
by sorry

end function_properties_l2842_284221


namespace boy_travel_time_l2842_284281

/-- Proves that given the conditions of the problem, the boy arrives 8 minutes early on the second day -/
theorem boy_travel_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 →
  speed_day1 = 5 →
  speed_day2 = 10 →
  late_time = 7 / 60 →
  let time_day1 : ℝ := distance / speed_day1
  let on_time : ℝ := time_day1 - late_time
  let time_day2 : ℝ := distance / speed_day2
  (on_time - time_day2) * 60 = 8 := by sorry

end boy_travel_time_l2842_284281


namespace chessboard_square_selection_l2842_284207

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents the number of ways to choose squares from a chessboard -/
def choose_squares (board : Chessboard) (num_squares : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose 60 squares from an 11x11 chessboard
    with no adjacent squares is 62 -/
theorem chessboard_square_selection :
  let board : Chessboard := ⟨11⟩
  choose_squares board 60 = 62 := by sorry

end chessboard_square_selection_l2842_284207


namespace complex_modulus_problem_l2842_284271

theorem complex_modulus_problem (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end complex_modulus_problem_l2842_284271


namespace max_tiles_on_floor_l2842_284222

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.length * d.width

/-- Represents the floor and tile dimensions -/
def floor : Dimensions := { length := 400, width := 600 }
def tile : Dimensions := { length := 20, width := 30 }

/-- Theorem stating the maximum number of tiles that can fit on the floor -/
theorem max_tiles_on_floor :
  (area floor / area tile : ℝ) = 400 := by sorry

end max_tiles_on_floor_l2842_284222


namespace sin_cos_relation_l2842_284205

theorem sin_cos_relation (x : Real) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := by
sorry

end sin_cos_relation_l2842_284205


namespace base_8_4531_equals_2393_l2842_284220

def base_8_to_10 (a b c d : ℕ) : ℕ :=
  a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0

theorem base_8_4531_equals_2393 :
  base_8_to_10 4 5 3 1 = 2393 := by
  sorry

end base_8_4531_equals_2393_l2842_284220


namespace equation_solutions_l2842_284272

theorem equation_solutions :
  (∃ x : ℝ, x - 2 * (5 + x) = -4 ∧ x = -6) ∧
  (∃ x : ℝ, (2 * x - 1) / 2 = 1 - (3 - x) / 4 ∧ x = 1) := by
  sorry

end equation_solutions_l2842_284272


namespace max_cables_cut_theorem_l2842_284215

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_computers - network.num_clusters)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end max_cables_cut_theorem_l2842_284215


namespace max_value_theorem_l2842_284267

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := 2 * x * Real.log (2 * x)

theorem max_value_theorem (x₁ x₂ t : ℝ) (h₁ : f x₁ = t) (h₂ : g x₂ = t) (h₃ : t > 0) :
  ∃ (m : ℝ), m = (2 : ℝ) / Real.exp 1 ∧ 
  ∀ (y : ℝ), y = (Real.log t) / (x₁ * x₂) → y ≤ m :=
sorry

end max_value_theorem_l2842_284267


namespace recurrence_sequence_general_term_l2842_284274

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  a 1 = 1 ∧
  ∀ n, (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

/-- The theorem stating that the sequence satisfying the recurrence relation
    has the general term a_n = 1/(2^(n-1)) -/
theorem recurrence_sequence_general_term (a : ℕ → ℝ) 
    (h : RecurrenceSequence a) : 
    ∀ n, a n = 1 / (2 ^ (n - 1)) := by
  sorry

end recurrence_sequence_general_term_l2842_284274


namespace equation_not_equivalent_l2842_284229

theorem equation_not_equivalent (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ 
  ¬((3*x + 2*y = x*y) ∨ 
    (y = 3*x/(5 - y)) ∨ 
    (x/3 + y/2 = 3) ∨ 
    (3*y/(y - 5) = x)) := by
  sorry

end equation_not_equivalent_l2842_284229


namespace stone_pile_division_l2842_284227

/-- Two natural numbers are similar if they differ by no more than twice -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A sequence of operations to combine piles -/
inductive CombineSeq : ℕ → ℕ → Type
  | single : (n : ℕ) → CombineSeq n 1
  | combine : {n m k : ℕ} → (s : CombineSeq n m) → (t : CombineSeq n k) → 
              similar m k → CombineSeq n (m + k)

/-- Any pile of stones can be divided into piles of single stones -/
theorem stone_pile_division (n : ℕ) : CombineSeq n n := by sorry

end stone_pile_division_l2842_284227


namespace shepherd_boys_sticks_l2842_284242

theorem shepherd_boys_sticks (x : ℕ) : 6 * x + 14 = 8 * x - 2 := by
  sorry

end shepherd_boys_sticks_l2842_284242


namespace ada_original_seat_l2842_284276

-- Define the type for seats
inductive Seat
| one | two | three | four | five | six

-- Define the type for friends
inductive Friend
| ada | bea | ceci | dee | edie | fana

-- Define the initial seating arrangement
def initial_seating : Friend → Seat := sorry

-- Define the movement function
def move (s : Seat) (n : Int) : Seat := sorry

-- Define the final seating arrangement after movements
def final_seating : Friend → Seat := sorry

-- Theorem to prove
theorem ada_original_seat :
  (∀ f : Friend, f ≠ Friend.ada → final_seating f ≠ initial_seating f) →
  (final_seating Friend.ada = Seat.one ∨ final_seating Friend.ada = Seat.six) →
  initial_seating Friend.ada = Seat.two :=
sorry

end ada_original_seat_l2842_284276


namespace rectangle_area_integer_l2842_284216

theorem rectangle_area_integer (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (n : ℕ), (a + b) * Real.sqrt (a * b) = n) ↔ (a = 9 ∧ b = 4) := by
  sorry

end rectangle_area_integer_l2842_284216


namespace paths_in_7x8_grid_l2842_284228

/-- The number of paths in a grid moving only up or right -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid moving only up or right is 6435 -/
theorem paths_in_7x8_grid : grid_paths 7 8 = 6435 := by
  sorry

end paths_in_7x8_grid_l2842_284228


namespace ratio_seconds_minutes_l2842_284231

theorem ratio_seconds_minutes : ∃ x : ℝ, (12 / x = 6 / (4 * 60)) ∧ x = 480 := by
  sorry

end ratio_seconds_minutes_l2842_284231


namespace negation_of_existence_negation_of_specific_proposition_l2842_284235

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x₀ > 0, p x₀) ↔ ∀ x > 0, ¬(p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x₀ > 0, 2^x₀ ≥ 3) ↔ ∀ x > 0, 2^x < 3 := by sorry

end negation_of_existence_negation_of_specific_proposition_l2842_284235


namespace parallel_lines_m_values_l2842_284224

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (b1 ≠ 0 ∧ b2 ≠ 0 ∧ a1 / b1 = a2 / b2) ∨ (b1 = 0 ∧ b2 = 0)

/-- The main theorem -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel 1 (1+m) (m-2) m 2 6 → m = -2 ∨ m = 1 := by
  sorry


end parallel_lines_m_values_l2842_284224


namespace root_shift_polynomial_l2842_284225

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x, x^3 - 3*x^2 + 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 - 12*x^2 + 49*x - 67 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end root_shift_polynomial_l2842_284225


namespace intersection_line_canonical_equations_l2842_284202

/-- The canonical equations of the line formed by the intersection of two planes -/
theorem intersection_line_canonical_equations 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop) 
  (canonical_eq : ℝ → ℝ → ℝ → Prop) : 
  (∀ x y z, plane1 x y z ↔ 3*x + 3*y - 2*z - 1 = 0) →
  (∀ x y z, plane2 x y z ↔ 2*x - 3*y + z + 6 = 0) →
  (∀ x y z, canonical_eq x y z ↔ (x + 1)/(-3) = (y - 4/3)/(-7) ∧ (y - 4/3)/(-7) = z/(-15)) →
  ∀ x y z, (plane1 x y z ∧ plane2 x y z) ↔ canonical_eq x y z :=
sorry

end intersection_line_canonical_equations_l2842_284202


namespace system_of_equations_solution_l2842_284213

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 3 * y = -14) ∧ (5 * x + 3 * y = -12) ∧ (x = -26/9) ∧ (y = -22/27) := by
  sorry

end system_of_equations_solution_l2842_284213


namespace construction_materials_cost_l2842_284297

/-- The total cost of materials bought by the construction company -/
def total_cost (gravel_tons sand_tons cement_tons : ℝ) 
               (gravel_price sand_price cement_price : ℝ) : ℝ :=
  gravel_tons * gravel_price + sand_tons * sand_price + cement_tons * cement_price

/-- Theorem stating the total cost of materials -/
theorem construction_materials_cost : 
  total_cost 5.91 8.11 4.35 30.50 40.50 55.60 = 750.57 := by
  sorry

end construction_materials_cost_l2842_284297


namespace book_page_digits_l2842_284252

/-- Calculate the total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigit := min n 9
  let doubleDigit := max 0 (min n 99 - 9)
  let tripleDigit := max 0 (n - 99)
  singleDigit + 2 * doubleDigit + 3 * tripleDigit

/-- The total number of digits used in numbering the pages of a book with 360 pages is 972 -/
theorem book_page_digits :
  totalDigits 360 = 972 := by
  sorry

end book_page_digits_l2842_284252


namespace g_magnitude_l2842_284253

/-- A quadratic function that is even on a specific interval -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a

/-- The function g defined as a transformation of f -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relative magnitudes of g at specific points -/
theorem g_magnitude (a : ℝ) (h : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) :
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end g_magnitude_l2842_284253


namespace area_of_region_R_l2842_284280

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R within the square -/
def region_R (s : Square) := {p : ℝ × ℝ | 
  p.1 ≥ 0 ∧ p.1 ≤ s.side_length ∧ 
  p.2 ≥ 0 ∧ p.2 ≤ s.side_length ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + p.2^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + (p.2 - s.side_length)^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < (p.1 - s.side_length)^2 + (p.2 - s.side_length)^2
}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of region R in a square with side length 3 is 9/4 -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9/4 := by sorry

end area_of_region_R_l2842_284280


namespace class_composition_l2842_284208

theorem class_composition (total students : ℕ) (girls boys : ℕ) : 
  students = girls + boys →
  (girls : ℚ) / (students : ℚ) = 60 / 100 →
  ((girls - 1 : ℚ) / ((students - 3) : ℚ)) = 125 / 200 →
  girls = 21 ∧ boys = 14 :=
by sorry

end class_composition_l2842_284208


namespace similar_right_triangles_leg_l2842_284230

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs y and 6, prove that y = 8 -/
theorem similar_right_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by sorry

end similar_right_triangles_leg_l2842_284230


namespace estimate_greater_than_exact_l2842_284250

theorem estimate_greater_than_exact (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (a' b' c' d' : ℕ) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≥ c) (hd' : d' ≥ d) : 
  (a' : ℚ) / b' + c' - d' > (a : ℚ) / b + c - d :=
by sorry

end estimate_greater_than_exact_l2842_284250


namespace arithmetic_sequence_minimum_value_l2842_284270

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_special : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 4 / n = 3 / 2) :=
sorry

end arithmetic_sequence_minimum_value_l2842_284270


namespace shadow_height_calculation_l2842_284258

/-- Given a lamppost and a person casting shadows under the same light source,
    calculate the person's height using the ratio method. -/
theorem shadow_height_calculation
  (lamppost_height : ℝ)
  (lamppost_shadow : ℝ)
  (michael_shadow : ℝ)
  (h_lamppost_height : lamppost_height = 50)
  (h_lamppost_shadow : lamppost_shadow = 25)
  (h_michael_shadow : michael_shadow = 20 / 12)  -- Convert 20 inches to feet
  : ∃ (michael_height : ℝ),
    michael_height = (lamppost_height / lamppost_shadow) * michael_shadow ∧
    michael_height * 12 = 40 := by
  sorry

end shadow_height_calculation_l2842_284258


namespace twenty_four_game_4888_l2842_284247

/-- The "24 points" game with cards 4, 8, 8, 8 -/
theorem twenty_four_game_4888 :
  let a : ℕ := 4
  let b : ℕ := 8
  let c : ℕ := 8
  let d : ℕ := 8
  (a - (c / d)) * b = 24 :=
by sorry

end twenty_four_game_4888_l2842_284247


namespace train_speed_calculation_l2842_284284

/-- Proves that given the conditions of a jogger and a train, the train's speed is 36 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (distance_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  distance_ahead = 240 →
  train_length = 130 →
  passing_time = 37 →
  ∃ (train_speed : ℝ), train_speed = 36 :=
by
  sorry


end train_speed_calculation_l2842_284284


namespace bus_profit_at_2600_passengers_l2842_284217

/-- Represents the monthly profit of a minibus based on the number of passengers -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * (passengers : ℤ) - 5000

/-- Theorem stating that the bus makes a profit with 2600 passengers -/
theorem bus_profit_at_2600_passengers :
  monthly_profit 2600 > 0 := by
  sorry

end bus_profit_at_2600_passengers_l2842_284217


namespace lines_exist_iff_angle_geq_60_l2842_284237

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the given point and planes
variable (P : Point3D) -- Given point
variable (givenPlane : Plane) -- Given plane
variable (firstProjectionPlane : Plane) -- First projection plane

-- Define the angle between two planes
def angleBetweenPlanes (p1 p2 : Plane) : ℝ := sorry

-- Define a line passing through a point
structure Line where
  point : Point3D
  direction : Point3D

-- Define the angle between a line and a plane
def angleLinePlane (l : Line) (p : Plane) : ℝ := sorry

-- Define the distance between a point and a plane
def distancePointPlane (point : Point3D) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem lines_exist_iff_angle_geq_60 :
  (∃ (l1 l2 : Line),
    l1.point = P ∧
    l2.point = P ∧
    angleLinePlane l1 firstProjectionPlane = 60 ∧
    angleLinePlane l2 firstProjectionPlane = 60 ∧
    distancePointPlane l1.point givenPlane = distancePointPlane l2.point givenPlane) ↔
  angleBetweenPlanes givenPlane firstProjectionPlane ≥ 60 :=
sorry

end lines_exist_iff_angle_geq_60_l2842_284237


namespace cookies_per_pack_l2842_284236

theorem cookies_per_pack (num_trays : ℕ) (cookies_per_tray : ℕ) (num_packs : ℕ) :
  num_trays = 4 →
  cookies_per_tray = 24 →
  num_packs = 8 →
  (num_trays * cookies_per_tray) / num_packs = 12 :=
by sorry

end cookies_per_pack_l2842_284236


namespace unique_vector_b_l2842_284241

def a : ℝ × ℝ := (-4, 3)
def c : ℝ × ℝ := (1, 1)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def acute_angle (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 > 0

theorem unique_vector_b :
  ∃! b : ℝ × ℝ,
    collinear b a ∧
    ‖b‖ = 10 ∧
    acute_angle b c ∧
    b = (8, -6) := by sorry

end unique_vector_b_l2842_284241


namespace min_abs_phi_l2842_284259

/-- Given a function y = 2sin(x + φ), prove that if the abscissa is shortened to 1/3
    and the graph is shifted right by π/4, resulting in symmetry about (π/3, 0),
    then the minimum value of |φ| is π/4. -/
theorem min_abs_phi (φ : Real) : 
  (∀ x, 2 * Real.sin (3 * x + φ - 3 * Real.pi / 4) = 
        2 * Real.sin (3 * (2 * Real.pi / 3 - x) + φ - 3 * Real.pi / 4)) → 
  (∃ k : ℤ, φ = Real.pi / 4 + k * Real.pi) ∧ 
  (∀ ψ : Real, (∃ k : ℤ, ψ = Real.pi / 4 + k * Real.pi) → |ψ| ≥ |φ|) := by
  sorry

end min_abs_phi_l2842_284259


namespace choose_three_from_thirteen_l2842_284239

theorem choose_three_from_thirteen : Nat.choose 13 3 = 286 := by sorry

end choose_three_from_thirteen_l2842_284239


namespace coloring_book_shelves_l2842_284296

/-- Given a store's coloring book inventory and sales, calculate the number of shelves needed to display the remaining books. -/
theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) :
  initial_stock = 86 →
  books_sold = 37 →
  books_per_shelf = 7 →
  (initial_stock - books_sold) / books_per_shelf = 7 :=
by
  sorry

end coloring_book_shelves_l2842_284296


namespace valid_configuration_iff_n_eq_4_l2842_284200

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  values : Fin n → ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- The condition that no three points are collinear -/
def noThreeCollinear (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) ≠ 0

/-- The condition that the area of any triangle equals the sum of corresponding values -/
def areaEqualsSumOfValues (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.values i + config.values j + config.values k

/-- The main theorem stating that a valid configuration exists if and only if n = 4 -/
theorem valid_configuration_iff_n_eq_4 :
  (∃ (config : PointConfiguration n), n > 3 ∧ noThreeCollinear config ∧ areaEqualsSumOfValues config) ↔
  n = 4 := by
  sorry

end valid_configuration_iff_n_eq_4_l2842_284200


namespace cos_four_pi_thirds_l2842_284262

theorem cos_four_pi_thirds : Real.cos (4 * Real.pi / 3) = -1 / 2 := by
  sorry

end cos_four_pi_thirds_l2842_284262


namespace sum_of_two_primes_10003_l2842_284260

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end sum_of_two_primes_10003_l2842_284260


namespace seventy_fifth_term_of_sequence_l2842_284214

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem seventy_fifth_term_of_sequence :
  arithmetic_sequence 2 4 75 = 298 := by sorry

end seventy_fifth_term_of_sequence_l2842_284214


namespace potato_price_proof_l2842_284244

/-- The original price of a bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earning := bags_bought * original_price * andrey_increase
  let boris_first_earning := boris_first_sale * original_price * boris_first_increase
  let boris_second_earning := boris_second_sale * original_price * boris_first_increase * boris_second_increase
  boris_first_earning + boris_second_earning - andrey_earning = earnings_difference :=
by sorry

end potato_price_proof_l2842_284244


namespace sum_base5_digits_2010_l2842_284298

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 5) :: aux (m / 5)
  aux n |>.reverse

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- The sum of digits in the base-5 representation of 2010 equals 6 -/
theorem sum_base5_digits_2010 : sumDigits (toBase5 2010) = 6 := by
  sorry

end sum_base5_digits_2010_l2842_284298


namespace tree_height_average_l2842_284265

def tree_heights : List ℕ → Prop
  | [h1, h2, h3, h4, h5] =>
    h2 = 6 ∧
    (h1 = 2 * h2 ∨ 2 * h1 = h2) ∧
    (h2 = 2 * h3 ∨ 2 * h2 = h3) ∧
    (h3 = 2 * h4 ∨ 2 * h3 = h4) ∧
    (h4 = 2 * h5 ∨ 2 * h4 = h5)
  | _ => False

theorem tree_height_average :
  ∀ (heights : List ℕ),
    tree_heights heights →
    (heights.sum : ℚ) / heights.length = 66 / 5 := by
  sorry

end tree_height_average_l2842_284265


namespace unique_k_value_l2842_284299

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - (k + 2) * x + 6

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  (k + 2)^2 - 4 * 3 * 6 ≥ 0

-- Define the condition that 3 is a root
def three_is_root (k : ℝ) : Prop :=
  quadratic k 3 = 0

-- The main theorem
theorem unique_k_value :
  ∃! k : ℝ, has_real_roots k ∧ three_is_root k :=
sorry

end unique_k_value_l2842_284299


namespace cuboid_volume_problem_l2842_284232

theorem cuboid_volume_problem (x y : ℕ) : 
  (x > 0) → 
  (y > 0) → 
  (x < 4) → 
  (y < 15) → 
  (15 * 5 * 4 - x * 5 * y = 120) → 
  (x + y = 15) := by
sorry

end cuboid_volume_problem_l2842_284232


namespace divisibility_of_sum_of_powers_l2842_284254

theorem divisibility_of_sum_of_powers (a b : ℤ) (n : ℕ) :
  (a + b) ∣ (a^(2*n + 1) + b^(2*n + 1)) := by sorry

end divisibility_of_sum_of_powers_l2842_284254


namespace arithmetic_geometric_progression_l2842_284245

-- Arithmetic Progression
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (a + (n - 1) * d / 2)

-- Geometric Progression
def geometric_product (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n * (n - 1) / 2)

theorem arithmetic_geometric_progression :
  (arithmetic_sum 0 (1/3) 15 = 35) ∧
  (geometric_product 1 (10^(1/3)) 15 = 10^35) := by
  sorry

end arithmetic_geometric_progression_l2842_284245


namespace trailing_zeros_50_factorial_l2842_284269

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry

end trailing_zeros_50_factorial_l2842_284269


namespace rectangle_area_change_l2842_284204

theorem rectangle_area_change 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := l * 1.6
  let new_width := w * 0.4
  let initial_area := l * w
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = -0.36 := by
  sorry

end rectangle_area_change_l2842_284204


namespace boat_transport_two_days_l2842_284255

/-- The number of people a boat can transport in multiple days -/
def boat_transport (capacity : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  capacity * trips_per_day * days

/-- Theorem: A boat with capacity 12 making 4 trips per day can transport 96 people in 2 days -/
theorem boat_transport_two_days :
  boat_transport 12 4 2 = 96 := by
  sorry

end boat_transport_two_days_l2842_284255


namespace james_sticker_collection_l2842_284246

theorem james_sticker_collection (initial : ℕ) (gift : ℕ) (given_away : ℕ) 
  (h1 : initial = 478) 
  (h2 : gift = 182) 
  (h3 : given_away = 276) : 
  initial + gift - given_away = 384 := by
  sorry

end james_sticker_collection_l2842_284246


namespace largest_number_l2842_284275

theorem largest_number (a b c d : ℝ) : 
  a = 1 → b = 0 → c = |-2| → d = -3 → 
  max a (max b (max c d)) = |-2| := by
sorry

end largest_number_l2842_284275


namespace marks_age_difference_l2842_284290

theorem marks_age_difference (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age + 4 = 2 * (aaron_current_age + 4) + 2 →
  (mark_current_age - 3) - 3 * (aaron_current_age - 3) = 1 := by
  sorry

end marks_age_difference_l2842_284290


namespace append_five_to_two_digit_number_l2842_284268

/-- Given a two-digit number with tens' digit t and units' digit u,
    when the digit 5 is placed after this number,
    the resulting number is equal to 100t + 10u + 5. -/
theorem append_five_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 5 = 100 * t + 10 * u + 5 := by
  sorry

end append_five_to_two_digit_number_l2842_284268


namespace simplify_sqrt_difference_l2842_284211

theorem simplify_sqrt_difference : 
  (Real.sqrt 800 / Real.sqrt 50) - (Real.sqrt 288 / Real.sqrt 72) = 2 := by
  sorry

end simplify_sqrt_difference_l2842_284211
