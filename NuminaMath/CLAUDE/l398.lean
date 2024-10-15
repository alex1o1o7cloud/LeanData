import Mathlib

namespace NUMINAMATH_CALUDE_problem_statements_l398_39823

theorem problem_statements :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l398_39823


namespace NUMINAMATH_CALUDE_parabola_properties_l398_39874

/-- Definition of the parabola -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Definition of the directrix -/
def directrix (x : ℝ) : Prop := x = -4

/-- Definition of a line passing through (-4, 0) with slope m -/
def line (m x y : ℝ) : Prop := y = m * (x + 4)

/-- Theorem stating the properties of the parabola and its intersecting lines -/
theorem parabola_properties :
  (∃ (x y : ℝ), directrix x ∧ y = 0) ∧ 
  (∀ (m : ℝ), (∃ (x y : ℝ), parabola x y ∧ line m x y) ↔ m ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l398_39874


namespace NUMINAMATH_CALUDE_cubic_function_properties_l398_39881

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem stating the properties of the cubic function f -/
theorem cubic_function_properties :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧ 
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t : ℝ), 
      (t = 0 ∧ 2 * 0 + f a b 0 - 1 = 0) ∨
      (t = 1/4 ∧ 33 * (1/4) + 16 * (f a b (1/4)) - 16 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l398_39881


namespace NUMINAMATH_CALUDE_max_x_value_l398_39838

theorem max_x_value (x y : ℝ) (h1 : x + y ≤ 1) (h2 : y + x + y ≤ 1) :
  x ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ ≤ 1 ∧ y₀ + x₀ + y₀ ≤ 1 ∧ x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l398_39838


namespace NUMINAMATH_CALUDE_x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l398_39890

theorem x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero 
  (x y : ℝ) : x^3 * y^2 - y^2 * x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l398_39890


namespace NUMINAMATH_CALUDE_class_average_score_l398_39824

theorem class_average_score
  (num_boys : ℕ)
  (num_girls : ℕ)
  (avg_score_boys : ℚ)
  (avg_score_girls : ℚ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 4)
  (h3 : avg_score_boys = 84)
  (h4 : avg_score_girls = 92) :
  (num_boys * avg_score_boys + num_girls * avg_score_girls) / (num_boys + num_girls) = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l398_39824


namespace NUMINAMATH_CALUDE_travel_time_proof_l398_39873

/-- Given a person traveling at 20 km/hr for a distance of 160 km,
    prove that the time taken is 8 hours. -/
theorem travel_time_proof (speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : speed = 20)
    (h2 : distance = 160)
    (h3 : time * speed = distance) :
  time = 8 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_proof_l398_39873


namespace NUMINAMATH_CALUDE_limit_expected_sides_l398_39804

/-- The expected number of sides of a polygon after k cuts -/
def expected_sides (n : ℕ) (k : ℕ) : ℚ :=
  (n + 4 * k : ℚ) / (k + 1 : ℚ)

/-- Theorem: The limit of expected sides approaches 4 as k approaches infinity -/
theorem limit_expected_sides (n : ℕ) :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides n k - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_expected_sides_l398_39804


namespace NUMINAMATH_CALUDE_expression_simplification_l398_39842

theorem expression_simplification (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - 
  (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 = 
  (2*(a + b + c - d))^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l398_39842


namespace NUMINAMATH_CALUDE_car_wash_earnings_per_car_l398_39839

/-- Proves that a car wash company making $2000 in 5 days while cleaning 80 cars per day earns $5 per car --/
theorem car_wash_earnings_per_car 
  (cars_per_day : ℕ) 
  (total_days : ℕ) 
  (total_earnings : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : total_days = 5) 
  (h3 : total_earnings = 2000) : 
  total_earnings / (cars_per_day * total_days) = 5 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_per_car_l398_39839


namespace NUMINAMATH_CALUDE_expression_evaluation_l398_39819

theorem expression_evaluation : -30 + 12 * (8 / 4)^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l398_39819


namespace NUMINAMATH_CALUDE_largest_cards_per_page_l398_39833

theorem largest_cards_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 1080) 
  (h2 : album2 = 1620) 
  (h3 : album3 = 540) : 
  Nat.gcd album1 (Nat.gcd album2 album3) = 540 := by
  sorry

end NUMINAMATH_CALUDE_largest_cards_per_page_l398_39833


namespace NUMINAMATH_CALUDE_systematic_sample_first_product_l398_39863

/-- Represents a systematic sample from a range of numbered products. -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  sample_interval : ℕ
  first_product : ℕ

/-- Creates a systematic sample given the total number of products and sample size. -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products,
    sample_size := sample_size,
    sample_interval := total_products / sample_size,
    first_product := 1 }

/-- Checks if a given product number is in the systematic sample. -/
def is_in_sample (s : SystematicSample) (product_number : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < s.sample_size ∧ product_number = s.first_product + k * s.sample_interval

/-- Theorem: In a systematic sample of size 5 from 80 products, 
    if product 42 is in the sample, then the first product's number is 10. -/
theorem systematic_sample_first_product :
  let s := create_systematic_sample 80 5
  is_in_sample s 42 → s.first_product = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_first_product_l398_39863


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt33_l398_39893

theorem consecutive_integers_around_sqrt33 (a b : ℤ) :
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 33) →  -- a < √33
  (Real.sqrt 33 < b) →  -- √33 < b
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt33_l398_39893


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l398_39852

/-- A quadrilateral with vertices at (1,2), (4,6), (6,5), and (4,1) -/
def Quadrilateral : Set (ℝ × ℝ) :=
  {(1, 2), (4, 6), (6, 5), (4, 1)}

/-- The perimeter of the quadrilateral -/
noncomputable def perimeter : ℝ :=
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist (1, 2) (4, 6) + dist (4, 6) (6, 5) + dist (6, 5) (4, 1) + dist (4, 1) (1, 2)

/-- The theorem to be proved -/
theorem quadrilateral_perimeter_sum (c d : ℤ) :
  perimeter = c * Real.sqrt 5 + d * Real.sqrt 13 → c + d = 9 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_sum_l398_39852


namespace NUMINAMATH_CALUDE_smallest_alpha_is_eight_l398_39810

/-- A quadratic polynomial P(x) = ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The derivative of a quadratic polynomial at x = 0 -/
def QuadraticPolynomial.deriv_at_zero (P : QuadraticPolynomial) : ℝ :=
  P.b

/-- The property that |P(x)| ≤ 1 for x ∈ [0,1] -/
def bounded_on_unit_interval (P : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |P.eval x| ≤ 1

theorem smallest_alpha_is_eight :
  (∃ α : ℝ, ∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ α) ∧
  (∀ β : ℝ, (∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ β) → 8 ≤ β) :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_is_eight_l398_39810


namespace NUMINAMATH_CALUDE_circle_angle_distance_sum_l398_39803

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the angle
def Angle : Type := Point → Point → Point → Prop

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the line segment
def LineSegment (p q : Point) : Point → Prop := sorry

-- State the theorem
theorem circle_angle_distance_sum
  (circle : Circle)
  (angle : Angle)
  (A B C : Point)
  (h1 : circle A ∧ circle B ∧ circle C)
  (h2 : angle A B C)
  (h3 : ∀ p, LineSegment A B p → distance C p = 8)
  (h4 : ∃ (d1 d2 : ℝ), d1 = d2 + 30 ∧
        (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2))) :
  ∃ (d1 d2 : ℝ), d1 + d2 = 34 ∧
    (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2)) :=
sorry

end NUMINAMATH_CALUDE_circle_angle_distance_sum_l398_39803


namespace NUMINAMATH_CALUDE_cattle_land_is_40_l398_39871

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  crop_production : ℕ

/-- Calculates the land dedicated to rearing cattle -/
def cattle_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.crop_production)

/-- Theorem stating that the land dedicated to rearing cattle is 40 hectares -/
theorem cattle_land_is_40 (farm : FarmLand) 
    (h1 : farm.total = 150)
    (h2 : farm.house_and_machinery = 25)
    (h3 : farm.future_expansion = 15)
    (h4 : farm.crop_production = 70) : 
  cattle_land farm = 40 := by
  sorry

#eval cattle_land { total := 150, house_and_machinery := 25, future_expansion := 15, crop_production := 70 }

end NUMINAMATH_CALUDE_cattle_land_is_40_l398_39871


namespace NUMINAMATH_CALUDE_brick_length_calculation_l398_39896

/-- Calculates the length of a brick given wall dimensions, partial brick dimensions, and number of bricks --/
theorem brick_length_calculation (wall_length wall_width wall_height brick_width brick_height num_bricks : ℝ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_width = 11.25 →
  brick_height = 6 →
  num_bricks = 3200 →
  (wall_length * wall_width * wall_height) / (num_bricks * brick_width * brick_height) = 50 := by
  sorry

#check brick_length_calculation

end NUMINAMATH_CALUDE_brick_length_calculation_l398_39896


namespace NUMINAMATH_CALUDE_max_surrounding_squares_l398_39861

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Predicate to check if two squares are non-overlapping -/
def non_overlapping (s1 s2 : Square) : Prop :=
  sorry

/-- Function to count the number of non-overlapping squares around a central square -/
def count_surrounding_squares (central : Square) (surrounding : List Square) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of non-overlapping squares 
    that can be placed around a given square is 8 -/
theorem max_surrounding_squares (central : Square) (surrounding : List Square) :
  count_surrounding_squares central surrounding ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_surrounding_squares_l398_39861


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l398_39897

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 14

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 28

/-- Theorem stating that one bowling ball weighs 14 pounds given the conditions -/
theorem bowling_ball_weight_is_14 :
  (8 * bowling_ball_weight = 4 * kayak_weight) ∧
  (3 * kayak_weight = 84) →
  bowling_ball_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l398_39897


namespace NUMINAMATH_CALUDE_distribute_10_4_l398_39892

/-- The number of ways to distribute n identical objects among k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 10 identical objects among 4 distinct containers,
    where each container must have at least one object, results in 34 unique arrangements. -/
theorem distribute_10_4 : distribute 10 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_4_l398_39892


namespace NUMINAMATH_CALUDE_min_value_of_S_l398_39817

def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S :
  ∃ (min : ℝ), min = 112.5 ∧ ∀ (x : ℝ), S x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_S_l398_39817


namespace NUMINAMATH_CALUDE_remainder_x_squared_mod_30_l398_39826

theorem remainder_x_squared_mod_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  x^2 ≡ 21 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_remainder_x_squared_mod_30_l398_39826


namespace NUMINAMATH_CALUDE_female_officers_count_l398_39813

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 210 →
  female_on_duty_ratio = 2/3 →
  female_ratio = 24/100 →
  ∃ (total_female : ℕ), total_female = 583 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_ratio : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l398_39813


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l398_39806

theorem complex_equation_modulus : ∃ (x y : ℝ), 
  (Complex.I + 1) * x + Complex.I * y = (Complex.I + 3 * Complex.I) * Complex.I ∧ 
  Complex.abs (x + Complex.I * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l398_39806


namespace NUMINAMATH_CALUDE_min_square_sum_min_square_sum_achievable_l398_39832

theorem min_square_sum (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 2 * x₃ = 50) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 1250 / 7 := by
  sorry

theorem min_square_sum_achievable : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  x₁ + 3 * x₂ + 2 * x₃ = 50 ∧ 
  x₁^2 + x₂^2 + x₃^2 = 1250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_min_square_sum_min_square_sum_achievable_l398_39832


namespace NUMINAMATH_CALUDE_anthony_lunch_money_l398_39858

theorem anthony_lunch_money (juice_cost cupcake_cost remaining_amount : ℕ) 
  (h1 : juice_cost = 27)
  (h2 : cupcake_cost = 40)
  (h3 : remaining_amount = 8) :
  juice_cost + cupcake_cost + remaining_amount = 75 := by
  sorry

end NUMINAMATH_CALUDE_anthony_lunch_money_l398_39858


namespace NUMINAMATH_CALUDE_max_value_expression_l398_39818

theorem max_value_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l398_39818


namespace NUMINAMATH_CALUDE_representatives_count_l398_39853

/-- The number of ways to select representatives from a group with females and males -/
def select_representatives (num_females num_males num_representatives : ℕ) : ℕ :=
  Nat.choose num_females 1 * Nat.choose num_males 2 + 
  Nat.choose num_females 2 * Nat.choose num_males 1

/-- Theorem stating that selecting 3 representatives from 3 females and 4 males, 
    with at least one of each, results in 30 different ways -/
theorem representatives_count : select_representatives 3 4 3 = 30 := by
  sorry

#eval select_representatives 3 4 3

end NUMINAMATH_CALUDE_representatives_count_l398_39853


namespace NUMINAMATH_CALUDE_smallest_block_size_l398_39800

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are visible. -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Checks if the given dimensions satisfy the problem conditions. -/
def isValidBlock (d : BlockDimensions) : Prop :=
  invisibleCubes d = 300 ∧ d.length > 1 ∧ d.width > 1 ∧ d.height > 1

/-- Theorem stating that the smallest possible number of cubes is 462. -/
theorem smallest_block_size :
  ∃ (d : BlockDimensions), isValidBlock d ∧ totalCubes d = 462 ∧
  (∀ (d' : BlockDimensions), isValidBlock d' → totalCubes d' ≥ 462) :=
sorry

end NUMINAMATH_CALUDE_smallest_block_size_l398_39800


namespace NUMINAMATH_CALUDE_solve_equation_l398_39872

theorem solve_equation : ∃ X : ℝ, 
  (((4 - 3.5 * (2 + 1/7 - (1 + 1/5))) / 0.16) / X = 
  ((3 + 2/7 - (3/14 / (1/6))) / (41 + 23/84 - (40 + 49/60))) ∧ X = 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l398_39872


namespace NUMINAMATH_CALUDE_divisibility_problem_l398_39894

theorem divisibility_problem (m n : ℕ) (h : m * n ∣ m + n) :
  (Nat.Prime n → n ∣ m) ∧
  (∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q → ¬(n ∣ m)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l398_39894


namespace NUMINAMATH_CALUDE_polynomial_expansion_l398_39860

theorem polynomial_expansion (x : ℝ) : 
  (5 * x - 3) * (2 * x^3 + 7 * x - 1) = 10 * x^4 - 6 * x^3 + 35 * x^2 - 26 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l398_39860


namespace NUMINAMATH_CALUDE_right_triangle_area_l398_39844

/-- The area of a right triangle with a 30-inch leg and a 34-inch hypotenuse is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l398_39844


namespace NUMINAMATH_CALUDE_lcm_gcd_product_9_12_l398_39831

theorem lcm_gcd_product_9_12 : Nat.lcm 9 12 * Nat.gcd 9 12 = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_9_12_l398_39831


namespace NUMINAMATH_CALUDE_problem_solution_l398_39884

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_solution : 
  avg3 (avg3 2 3 (-1)) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l398_39884


namespace NUMINAMATH_CALUDE_group_size_from_shoes_l398_39864

/-- Given a group of people where the total number of shoes is 20 and each person has 2 shoes,
    prove that the number of people in the group is 10. -/
theorem group_size_from_shoes (total_shoes : ℕ) (shoes_per_person : ℕ) 
    (h1 : total_shoes = 20) (h2 : shoes_per_person = 2) : 
    total_shoes / shoes_per_person = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_shoes_l398_39864


namespace NUMINAMATH_CALUDE_boys_girls_ratio_l398_39805

/-- Given a classroom with boys and girls, prove that the initial ratio of boys to girls is 1:1 --/
theorem boys_girls_ratio (total : ℕ) (left : ℕ) (B G : ℕ) : 
  total = 32 →  -- Total number of boys and girls initially
  left = 8 →  -- Number of girls who left
  B = 2 * (G - left) →  -- After girls left, there are twice as many boys as girls
  B + G = total →  -- Total is the sum of boys and girls
  B = G  -- The number of boys equals the number of girls, implying a 1:1 ratio
  := by sorry

end NUMINAMATH_CALUDE_boys_girls_ratio_l398_39805


namespace NUMINAMATH_CALUDE_medium_mall_sample_l398_39841

def stratified_sample (total_sample : ℕ) (ratio : List ℕ) : List ℕ :=
  let total_ratio := ratio.sum
  ratio.map (λ r => (total_sample * r) / total_ratio)

theorem medium_mall_sample :
  let ratio := [2, 4, 9]
  let sample := stratified_sample 45 ratio
  sample[1] = 12 := by sorry

end NUMINAMATH_CALUDE_medium_mall_sample_l398_39841


namespace NUMINAMATH_CALUDE_helper_sequences_count_l398_39837

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of student helpers possible in a week -/
def helper_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating that the number of different sequences of student helpers in a week is 3375 -/
theorem helper_sequences_count : helper_sequences = 3375 := by
  sorry

end NUMINAMATH_CALUDE_helper_sequences_count_l398_39837


namespace NUMINAMATH_CALUDE_decimal_expansion_eight_elevenths_repeating_block_size_l398_39834

/-- The smallest repeating block in the decimal expansion of 8/11 contains 2 digits. -/
theorem decimal_expansion_eight_elevenths_repeating_block_size :
  ∃ (a b : ℕ) (h : b ≠ 0),
    (8 : ℚ) / 11 = (a : ℚ) / (10^b - 1) ∧
    ∀ (c d : ℕ) (h' : d ≠ 0), (8 : ℚ) / 11 = (c : ℚ) / (10^d - 1) → b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_decimal_expansion_eight_elevenths_repeating_block_size_l398_39834


namespace NUMINAMATH_CALUDE_tan_690_degrees_l398_39816

theorem tan_690_degrees : Real.tan (690 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_690_degrees_l398_39816


namespace NUMINAMATH_CALUDE_prime_power_difference_l398_39812

theorem prime_power_difference (n : ℕ+) (p : ℕ) (k : ℕ) :
  (3 : ℕ) ^ n.val - (2 : ℕ) ^ n.val = p ^ k ∧ Nat.Prime p → Nat.Prime n.val :=
by sorry

end NUMINAMATH_CALUDE_prime_power_difference_l398_39812


namespace NUMINAMATH_CALUDE_league_games_l398_39846

theorem league_games (n : ℕ) (m : ℕ) (h1 : n = 20) (h2 : m = 4) :
  (n * (n - 1) / 2) * m = 760 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l398_39846


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l398_39836

def num_maple_trees : ℕ := 4
def num_oak_trees : ℕ := 5
def num_birch_trees : ℕ := 6

def total_trees : ℕ := num_maple_trees + num_oak_trees + num_birch_trees

def favorable_arrangements : ℕ := (Nat.choose 10 6) * (Nat.choose 9 4)
def total_arrangements : ℕ := Nat.factorial total_trees

theorem birch_tree_arrangement_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l398_39836


namespace NUMINAMATH_CALUDE_function_properties_l398_39847

-- Define the function f(x)
def f (d : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + d

-- State the theorem
theorem function_properties :
  ∃ (d : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≥ -4) ∧ 
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = -4) →
    d = 1 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≤ 23) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = 23) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l398_39847


namespace NUMINAMATH_CALUDE_sin_29pi_over_6_l398_39883

theorem sin_29pi_over_6 : Real.sin (29 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_29pi_over_6_l398_39883


namespace NUMINAMATH_CALUDE_triangle_equilateral_from_cos_product_l398_39811

/-- A triangle is equilateral if all its angles are equal -/
def IsEquilateral (A B C : ℝ) : Prop := A = B ∧ B = C

/-- Given a triangle ABC, if cos(A-B)cos(B-C)cos(C-A)=1, then the triangle is equilateral -/
theorem triangle_equilateral_from_cos_product (A B C : ℝ) 
  (h : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  IsEquilateral A B C := by
  sorry


end NUMINAMATH_CALUDE_triangle_equilateral_from_cos_product_l398_39811


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l398_39888

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l398_39888


namespace NUMINAMATH_CALUDE_total_commission_is_42000_l398_39802

def sale_price : ℝ := 10000
def commission_rate_first_100 : ℝ := 0.03
def commission_rate_after_100 : ℝ := 0.04
def total_machines_sold : ℕ := 130

def commission_first_100 : ℝ := 100 * sale_price * commission_rate_first_100
def commission_after_100 : ℝ := (total_machines_sold - 100) * sale_price * commission_rate_after_100

theorem total_commission_is_42000 :
  commission_first_100 + commission_after_100 = 42000 := by
  sorry

end NUMINAMATH_CALUDE_total_commission_is_42000_l398_39802


namespace NUMINAMATH_CALUDE_expression_simplification_l398_39820

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) :
  (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = (a - 1) / (a + 1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l398_39820


namespace NUMINAMATH_CALUDE_cos_2theta_value_l398_39808

theorem cos_2theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l398_39808


namespace NUMINAMATH_CALUDE_S_formula_l398_39862

def N (n : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem S_formula (n : ℕ) : S n = (4^n + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_S_formula_l398_39862


namespace NUMINAMATH_CALUDE_inheritance_problem_l398_39876

/-- Proves the unique solution for the inheritance problem --/
theorem inheritance_problem (A B C : ℕ) : 
  (A + B + C = 30000) →  -- Total inheritance
  (A - B = B - C) →      -- B's relationship to A and C
  (A = B + C) →          -- A's relationship to B and C
  (A = 15000 ∧ B = 10000 ∧ C = 5000) := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l398_39876


namespace NUMINAMATH_CALUDE_f_is_linear_l398_39869

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x - 2 = 1/3 -/
def f (x : ℝ) : ℝ := x - 2

theorem f_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_linear_l398_39869


namespace NUMINAMATH_CALUDE_inverse_function_derivative_l398_39830

-- Define the function f and its inverse g
variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (x₀ : ℝ) (y₀ : ℝ)

-- State the conditions
variable (hf : Differentiable ℝ f)
variable (hg : Differentiable ℝ g)
variable (hinverse : Function.LeftInverse g f ∧ Function.RightInverse g f)
variable (hderiv : (deriv f x₀) ≠ 0)
variable (hy : y₀ = f x₀)

-- State the theorem
theorem inverse_function_derivative :
  (deriv g y₀) = 1 / (deriv f x₀) := by sorry

end NUMINAMATH_CALUDE_inverse_function_derivative_l398_39830


namespace NUMINAMATH_CALUDE_collision_count_is_25_l398_39891

/-- Represents a set of identical balls moving in one direction -/
structure BallSet :=
  (count : Nat)
  (direction : Bool)  -- True for left to right, False for right to left

/-- Calculates the total number of collisions between two sets of balls -/
def totalCollisions (set1 set2 : BallSet) : Nat :=
  set1.count * set2.count

/-- Theorem stating that the total number of collisions is 25 -/
theorem collision_count_is_25 :
  ∀ (left right : BallSet),
    left.count = 5 ∧ 
    right.count = 5 ∧ 
    left.direction ≠ right.direction →
    totalCollisions left right = 25 := by
  sorry

#eval totalCollisions ⟨5, true⟩ ⟨5, false⟩

end NUMINAMATH_CALUDE_collision_count_is_25_l398_39891


namespace NUMINAMATH_CALUDE_sqrt_greater_than_3x_iff_less_than_one_ninth_l398_39899

theorem sqrt_greater_than_3x_iff_less_than_one_ninth 
  (x : ℝ) (hx : x > 0) : 
  Real.sqrt x > 3 * x ↔ x < 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_3x_iff_less_than_one_ninth_l398_39899


namespace NUMINAMATH_CALUDE_fraction_equality_l398_39885

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l398_39885


namespace NUMINAMATH_CALUDE_order_of_numbers_l398_39848

theorem order_of_numbers : 7^(3/10) > 0.3^7 ∧ 0.3^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l398_39848


namespace NUMINAMATH_CALUDE_sector_arc_length_l398_39898

/-- Given a circular sector with area 4 and central angle 2 radians, prove that the length of the arc is 4. -/
theorem sector_arc_length (S : ℝ) (θ : ℝ) (l : ℝ) : 
  S = 4 → θ = 2 → l = S * θ / 2 → l = 4 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l398_39898


namespace NUMINAMATH_CALUDE_yellow_stamp_price_is_two_l398_39843

/-- Calculates the price per yellow stamp needed to reach a total sale amount --/
def price_per_yellow_stamp (red_count : ℕ) (red_price : ℚ) (blue_count : ℕ) (blue_price : ℚ) 
  (yellow_count : ℕ) (total_sale : ℚ) : ℚ :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining := total_sale - (red_earnings + blue_earnings)
  remaining / yellow_count

/-- Theorem stating that the price per yellow stamp is $2 given the problem conditions --/
theorem yellow_stamp_price_is_two :
  price_per_yellow_stamp 20 1.1 80 0.8 7 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_stamp_price_is_two_l398_39843


namespace NUMINAMATH_CALUDE_exists_removable_factorial_for_perfect_square_l398_39878

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def product_of_factorials (n : ℕ) : ℕ := (Finset.range n).prod (λ i => factorial (i + 1))

theorem exists_removable_factorial_for_perfect_square :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ 
  ∃ m : ℕ, product_of_factorials 100 / factorial k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_removable_factorial_for_perfect_square_l398_39878


namespace NUMINAMATH_CALUDE_nth_group_sum_correct_l398_39875

/-- The sum of the n-th group in the sequence of positive integers grouped as 1, 2+3, 4+5+6, 7+8+9+10, ... -/
def nth_group_sum (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- The first element of the n-th group -/
def first_element (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

/-- The last element of the n-th group -/
def last_element (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem nth_group_sum_correct (n : ℕ) (h : n > 0) :
  nth_group_sum n = (n * (first_element n + last_element n)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_nth_group_sum_correct_l398_39875


namespace NUMINAMATH_CALUDE_square_of_digit_sum_sum_of_cube_digits_l398_39821

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Part a
theorem square_of_digit_sum (N : ℕ) : N = (sumOfDigits N)^2 ↔ N = 1 ∨ N = 81 := by sorry

-- Part b
theorem sum_of_cube_digits (N : ℕ) : N = sumOfDigits (N^3) ↔ N = 1 ∨ N = 8 ∨ N = 17 ∨ N = 18 ∨ N = 26 ∨ N = 27 := by sorry

end NUMINAMATH_CALUDE_square_of_digit_sum_sum_of_cube_digits_l398_39821


namespace NUMINAMATH_CALUDE_parabola_vertex_l398_39859

/-- The parabola defined by y = -3(x-1)^2 - 2 has its vertex at (1, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * (x - 1)^2 - 2 → (1, -2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l398_39859


namespace NUMINAMATH_CALUDE_rain_probability_l398_39815

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h_saturday : p_saturday = 0.6)
  (h_sunday : p_sunday = 0.4)
  (h_independent : True) -- Assumption of independence
  : p_saturday * (1 - p_sunday) + (1 - p_saturday) * p_sunday = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l398_39815


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l398_39850

/-- Given a line passing through points (1, -7) and (k, 19) that is parallel to 3x + 4y = 12, prove that k = -101/3 -/
theorem parallel_line_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (m * 1 + b = -7) ∧ (m * k + b = 19) ∧ (m = -3/4)) → 
  k = -101/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l398_39850


namespace NUMINAMATH_CALUDE_exponential_function_inequality_range_l398_39854

/-- The range of k for which the given inequality holds for all real x₁ and x₂ -/
theorem exponential_function_inequality_range :
  ∀ (k : ℝ), 
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → 
    |((Real.exp x₁) - (Real.exp x₂)) / (x₁ - x₂)| < |k| * ((Real.exp x₁) + (Real.exp x₂))) 
  ↔ 
  (k ≤ -1/2 ∨ k ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_inequality_range_l398_39854


namespace NUMINAMATH_CALUDE_prob_all_same_color_is_correct_l398_39827

def yellow_marbles : ℕ := 3
def green_marbles : ℕ := 7
def purple_marbles : ℕ := 5
def total_marbles : ℕ := yellow_marbles + green_marbles + purple_marbles
def drawn_marbles : ℕ := 4

def prob_all_same_color : ℚ :=
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (purple_marbles * (purple_marbles - 1) * (purple_marbles - 2) * (purple_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem prob_all_same_color_is_correct : prob_all_same_color = 532 / 4095 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_same_color_is_correct_l398_39827


namespace NUMINAMATH_CALUDE_product_max_for_square_l398_39865

/-- A quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The product of sums of opposite sides pairs -/
def productOfSums (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Theorem: The product of sums is maximum when the quadrilateral is a square -/
theorem product_max_for_square (q : CyclicQuadrilateral) :
  productOfSums q ≤ productOfSums { a := (q.a + q.b + q.c + q.d) / 4,
                                    b := (q.a + q.b + q.c + q.d) / 4,
                                    c := (q.a + q.b + q.c + q.d) / 4,
                                    d := (q.a + q.b + q.c + q.d) / 4,
                                    inscribed := sorry } := by
  sorry

end NUMINAMATH_CALUDE_product_max_for_square_l398_39865


namespace NUMINAMATH_CALUDE_harmonic_table_sum_remainder_l398_39829

theorem harmonic_table_sum_remainder : ∃ k : ℕ, (2^2007 - 1) / 2007 ≡ 1 [MOD 2008] := by
  sorry

end NUMINAMATH_CALUDE_harmonic_table_sum_remainder_l398_39829


namespace NUMINAMATH_CALUDE_negative_sum_reciprocal_l398_39868

theorem negative_sum_reciprocal (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  min (a + 1/b) (min (b + 1/c) (c + 1/a)) ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_reciprocal_l398_39868


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l398_39845

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

-- Define symmetry about the y-axis
def symmetricAboutYAxis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ, symmetricAboutYAxis (translate f 1) exp → f = λ x => exp (-x - 1) := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_and_translation_l398_39845


namespace NUMINAMATH_CALUDE_bounds_of_W_l398_39877

/-- Given conditions on x, y, and z, prove the bounds of W = 2x + 6y + 4z -/
theorem bounds_of_W (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (ineq_one : 3 * y + z ≥ 2)
  (x_bounds : 0 ≤ x ∧ x ≤ 1)
  (y_bounds : 0 ≤ y ∧ y ≤ 2) :
  let W := 2 * x + 6 * y + 4 * z
  ∃ (W_min W_max : ℝ), W_min = 4 ∧ W_max = 6 ∧ W_min ≤ W ∧ W ≤ W_max :=
by sorry

end NUMINAMATH_CALUDE_bounds_of_W_l398_39877


namespace NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l398_39882

theorem continuous_function_zero_on_interval
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) :
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l398_39882


namespace NUMINAMATH_CALUDE_vector_magnitude_relation_l398_39835

variables {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem vector_magnitude_relation (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : a = -3 • b) :
  ‖a‖ = 3 * ‖b‖ :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_relation_l398_39835


namespace NUMINAMATH_CALUDE_exactly_two_true_l398_39856

-- Define the propositions
def proposition1 : Prop :=
  (∀ x, x^2 - 3*x + 2 = 0 → x = 2 ∨ x = 1) →
  (∀ x, x^2 - 3*x + 2 ≠ 0 → x ≠ 2 ∨ x ≠ 1)

def proposition2 : Prop :=
  (∀ x > 1, x^2 - 1 > 0) →
  (∃ x > 1, x^2 - 1 ≤ 0)

def proposition3 (p q : Prop) : Prop :=
  (¬p ∧ ¬q → ¬(p ∨ q)) ∧ ¬(¬(p ∨ q) → ¬p ∧ ¬q)

-- Theorem stating that exactly two propositions are true
theorem exactly_two_true :
  (¬proposition1 ∧ proposition2 ∧ proposition3 True False) ∨
  (¬proposition1 ∧ proposition2 ∧ proposition3 False True) ∨
  (proposition2 ∧ proposition3 True False ∧ proposition3 False True) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_l398_39856


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l398_39807

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l398_39807


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l398_39857

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 1}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 4)}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 1 4 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l398_39857


namespace NUMINAMATH_CALUDE_handshake_arrangement_remainder_l398_39825

/-- The number of people in the group -/
def n : ℕ := 10

/-- The number of handshakes each person makes -/
def k : ℕ := 3

/-- Two arrangements are considered different if at least two people who shake hands
    in one arrangement don't in the other -/
def different_arrangement : Prop := sorry

/-- The total number of possible handshaking arrangements -/
def M : ℕ := sorry

/-- The theorem stating the main result -/
theorem handshake_arrangement_remainder :
  M % 500 = 84 := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_remainder_l398_39825


namespace NUMINAMATH_CALUDE_seashells_solution_l398_39809

/-- The number of seashells found by Sam, Mary, John, and Emily -/
def seashells_problem (sam mary john emily : ℕ) : Prop :=
  sam = 18 ∧ mary = 47 ∧ john = 32 ∧ emily = 26 →
  sam + mary + john + emily = 123

/-- Theorem stating the solution to the seashells problem -/
theorem seashells_solution : seashells_problem 18 47 32 26 := by
  sorry

end NUMINAMATH_CALUDE_seashells_solution_l398_39809


namespace NUMINAMATH_CALUDE_circle_to_ellipse_transformation_l398_39855

/-- A circle in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- An ellipse in the x'y'-plane -/
def Ellipse (x' y' : ℝ) : Prop := x'^2/16 + y'^2/4 = 1

/-- The scaling transformation -/
def ScalingTransformation (x' y' : ℝ) : ℝ × ℝ := (4*x', y')

theorem circle_to_ellipse_transformation :
  ∀ (x' y' : ℝ), 
  let (x, y) := ScalingTransformation x' y'
  Circle x y ↔ Ellipse x' y' := by
sorry

end NUMINAMATH_CALUDE_circle_to_ellipse_transformation_l398_39855


namespace NUMINAMATH_CALUDE_f_sixteen_value_l398_39801

theorem f_sixteen_value (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x) = -2 * f x) 
  (h2 : f 1 = -3) : 
  f 16 = -48 := by
sorry

end NUMINAMATH_CALUDE_f_sixteen_value_l398_39801


namespace NUMINAMATH_CALUDE_calculation_proof_l398_39895

theorem calculation_proof :
  ((-48) * 0.125 + 48 * (11/8) + (-48) * (5/4) = 0) ∧
  (|(-7/9)| / ((2/3) - (1/5)) - (1/3) * ((-4)^2) = -11/3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l398_39895


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l398_39840

-- Define the line equation
def line_equation (a x : ℝ) : ℝ := a * x - 3 * a + 2

-- Theorem stating that the line passes through (3, 2) for any real number a
theorem fixed_point_theorem (a : ℝ) : line_equation a 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l398_39840


namespace NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l398_39814

variables {f g : ℝ → ℝ} {a b x : ℝ}

theorem product_decreasing_implies_inequality
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l398_39814


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l398_39870

theorem simplify_and_evaluate (a b : ℝ) (h : |2 - a + b| + (a * b + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l398_39870


namespace NUMINAMATH_CALUDE_pond_water_after_50_days_l398_39879

/-- Calculates the remaining water in a pond after a given number of days, 
    given an initial amount and daily evaporation rate. -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given specific initial conditions, 
    the amount of water remaining after 50 days is 200 gallons. -/
theorem pond_water_after_50_days 
  (initial_amount : ℝ) 
  (evaporation_rate : ℝ) 
  (h1 : initial_amount = 250)
  (h2 : evaporation_rate = 1) :
  remaining_water initial_amount evaporation_rate 50 = 200 :=
by
  sorry

#eval remaining_water 250 1 50

end NUMINAMATH_CALUDE_pond_water_after_50_days_l398_39879


namespace NUMINAMATH_CALUDE_house_height_difference_l398_39866

/-- Given three houses with heights 80 feet, 70 feet, and 99 feet,
    prove that the difference between the average height and 80 feet is 3 feet. -/
theorem house_height_difference (h₁ h₂ h₃ : ℝ) 
  (h₁_height : h₁ = 80)
  (h₂_height : h₂ = 70)
  (h₃_height : h₃ = 99) :
  (h₁ + h₂ + h₃) / 3 - h₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_house_height_difference_l398_39866


namespace NUMINAMATH_CALUDE_multiplication_problem_l398_39889

theorem multiplication_problem : ∃ x : ℕ, 72517 * x = 724807415 ∧ x = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l398_39889


namespace NUMINAMATH_CALUDE_smallest_satisfying_both_properties_l398_39849

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)

theorem smallest_satisfying_both_properties : 
  ∀ n : ℕ, n < 2019 → ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_both_properties_l398_39849


namespace NUMINAMATH_CALUDE_bridge_length_proof_l398_39851

/-- Given a train of length 110 meters, traveling at 45 km/hr, that crosses a bridge in 30 seconds,
    prove that the length of the bridge is 265 meters. -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 265 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l398_39851


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_range_of_a_l398_39867

-- Part I
theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_range_of_a_l398_39867


namespace NUMINAMATH_CALUDE_part_one_part_two_l398_39886

-- Define the conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one : ∀ x : ℝ, (p 1 x ∨ q x) → (1 < x ∧ x < 3) := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, q x → p a x) ∧ 
  (∃ x : ℝ, p a x ∧ ¬q x) ∧ 
  (a > 0) → 
  (1 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l398_39886


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l398_39880

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of faces in a rectangular prism. -/
def faces_count : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges_count : ℕ := 12

/-- The number of diagonals in a rectangular prism. -/
def diagonals_count (rp : RectangularPrism) : ℕ := 16

/-- Theorem: A rectangular prism with different length, width, and height has exactly 16 diagonals. -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : 
  diagonals_count rp = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l398_39880


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l398_39822

/-- Given plane vectors a and b, where the angle between them is 60°,
    a = (2,0), and |b| = 1, then |a+b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (a = (2, 0)) →
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) →
  (Real.cos (60 * Real.pi / 180) = a.1 * b.1 + a.2 * b.2) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l398_39822


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l398_39887

def sequence_v (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients :
  (∃ a b c : ℚ, ∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) →
  (sequence_v 1 = 7) →
  (∀ n : ℕ, sequence_v (n + 1) - sequence_v n = 5 + 6 * (n - 1)) →
  (∃ a b c : ℚ, (∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) ∧ a + b + c = 7) :=
by sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l398_39887


namespace NUMINAMATH_CALUDE_hypotenuse_length_l398_39828

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- Hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The perimeter of the triangle is 40 -/
  perimeter : a + b + c = 40
  /-- The area of the triangle is 30 -/
  area : a * b / 2 = 30
  /-- The ratio of the legs is 3:4 -/
  leg_ratio : 3 * a = 4 * b

theorem hypotenuse_length (t : RightTriangle) : t.c = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l398_39828
