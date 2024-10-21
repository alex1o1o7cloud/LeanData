import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_competition_result_l157_15774

/-- Represents a rower in the competition -/
structure Rower where
  speedWithStream : ℝ
  speedAgainstStream : ℝ

/-- Calculates the effective downstream speed of a rower given the wind speed -/
noncomputable def effectiveDownstreamSpeed (r : Rower) (windSpeed : ℝ) : ℝ :=
  r.speedWithStream + windSpeed

/-- Calculates the time taken to cover a distance at a given speed -/
noncomputable def timeTaken (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem rowing_competition_result 
  (man1 : Rower) 
  (man2 : Rower) 
  (windSpeed : ℝ) 
  (distance : ℝ) 
  (h1 : man1.speedWithStream = 26)
  (h2 : man1.speedAgainstStream = 14)
  (h3 : man2.speedWithStream = 22)
  (h4 : man2.speedAgainstStream = 18)
  (h5 : windSpeed = 3)
  (h6 : distance = 40) :
  let time1 := timeTaken distance (effectiveDownstreamSpeed man1 windSpeed)
  let time2 := timeTaken distance (effectiveDownstreamSpeed man2 windSpeed)
  time1 < time2 ∧ 13.24 ≤ (time2 - time1) * 60 ∧ (time2 - time1) * 60 < 13.25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_competition_result_l157_15774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coprime_with_prime_l157_15798

-- Define a function to represent the coefficients of a number in base p
def baseP_coeff (n : ℕ) (p : ℕ) (i : ℕ) : ℕ :=
  (n / p^i) % p

-- Define the property that a's coefficients are greater than or equal to b's
def coeff_geq (a b p : ℕ) : Prop :=
  ∀ i, baseP_coeff a p i ≥ baseP_coeff b p i

theorem binomial_coprime_with_prime (p a b : ℕ) (hp : Nat.Prime p) (hab : a ≥ b) 
  (h_coeff : coeff_geq a b p) : Nat.Coprime (Nat.choose a b) p := by
  sorry

#check binomial_coprime_with_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coprime_with_prime_l157_15798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l157_15732

theorem sin_cos_difference (θ₁ θ₂ : ℝ) :
  Real.sin (θ₁ * π / 180) * Real.cos (θ₂ * π / 180) - 
  Real.sin ((90 - θ₁) * π / 180) * Real.cos ((90 - θ₂) * π / 180) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l157_15732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_condition_impossible_product_condition_possible_l157_15726

-- Define the type for positive rational numbers
def PositiveRational := {q : ℚ // q > 0}

-- Define the coloring function
def Coloring := PositiveRational → Bool

-- Helper function for addition of PositiveRational
def addPositiveRational (x y : PositiveRational) : PositiveRational :=
  ⟨x.val + y.val, by
    have h1 : x.val > 0 := x.property
    have h2 : y.val > 0 := y.property
    exact add_pos h1 h2⟩

-- Helper function for multiplication of PositiveRational
def mulPositiveRational (x y : PositiveRational) : PositiveRational :=
  ⟨x.val * y.val, by
    have h1 : x.val > 0 := x.property
    have h2 : y.val > 0 := y.property
    exact mul_pos h1 h2⟩

-- Theorem for the sum condition
theorem sum_condition_impossible :
  ¬∃(c : Coloring), 
    (∃(r : PositiveRational), c r = true) ∧ 
    (∃(b : PositiveRational), c b = false) ∧
    (∀(x y : PositiveRational), c x = c y → c (addPositiveRational x y) = c x) :=
sorry

-- Theorem for the product condition
theorem product_condition_possible :
  ∃(c : Coloring), 
    (∃(r : PositiveRational), c r = true) ∧ 
    (∃(b : PositiveRational), c b = false) ∧
    (∀(x y : PositiveRational), c x = c y → c (mulPositiveRational x y) = c x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_condition_impossible_product_condition_possible_l157_15726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_three_from_ten_l157_15731

theorem choose_three_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 3 → n.factorial / (n - k).factorial = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_three_from_ten_l157_15731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l157_15794

/-- The number of bounces needed for a ball to reach a height less than 2 feet -/
theorem ball_bounce_count : ∃ k : ℕ, 
  (∀ n : ℕ, n < k → 20 * (3/4 : ℝ)^n ≥ 2) ∧ 
  20 * (3/4 : ℝ)^k < 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l157_15794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l157_15730

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x^2
  else if 1 < x ∧ x < 2 then 2
  else if x ≥ 2 then 3
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc 0 2 ∪ {3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l157_15730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l157_15710

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | (n + 3) => (1 + sequence_a (n + 1) * sequence_a (n + 2)) / sequence_a n

theorem sequence_a_is_integer (n : ℕ) : ∃ (k : ℤ), sequence_a n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_is_integer_l157_15710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l157_15790

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_side_length 
  (B : ℝ) 
  (hB : f B = -1/18) 
  (AC BC AB : ℝ)
  (hAC : AC = 2 * Real.sqrt 5) 
  (hBC : BC = 6) 
  : AB = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l157_15790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_negative_l157_15708

theorem product_sum_negative (a b c d e f : ℤ) 
  (h_max_neg : (if a < 0 then 1 else 0) + 
               (if b < 0 then 1 else 0) + 
               (if c < 0 then 1 else 0) + 
               (if d < 0 then 1 else 0) + 
               (if e < 0 then 1 else 0) + 
               (if f < 0 then 1 else 0) ≤ 5) :
  a * b + c * d * e * f < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_negative_l157_15708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_min_on_interval_f_max_on_interval_f_min_value_f_max_value_l157_15737

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

-- Theorem for the monotonicity of f
theorem f_increasing_on_interval : 
  ∀ x y : ℝ, 1 ≤ x → x < y → f x < f y :=
by sorry

-- Theorem for the minimum value of f on [2, 4]
theorem f_min_on_interval : 
  ∀ x : ℝ, 2 ≤ x → x ≤ 4 → f 2 ≤ f x :=
by sorry

-- Theorem for the maximum value of f on [2, 4]
theorem f_max_on_interval : 
  ∀ x : ℝ, 2 ≤ x → x ≤ 4 → f x ≤ f 4 :=
by sorry

-- Theorem for the specific minimum value
theorem f_min_value : f 2 = 5 / 3 :=
by sorry

-- Theorem for the specific maximum value
theorem f_max_value : f 4 = 9 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_f_min_on_interval_f_max_on_interval_f_min_value_f_max_value_l157_15737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_purchase_theorem_l157_15735

/-- Calculates the total purchase amount given the tax rate, tax percentage of total, and tax-free amount -/
noncomputable def total_purchase (tax_rate : ℝ) (tax_percent_of_total : ℝ) (tax_free_amount : ℝ) : ℝ :=
  let taxable_amount := tax_free_amount * tax_percent_of_total / (tax_rate - tax_percent_of_total)
  taxable_amount + tax_free_amount

/-- Theorem stating that under the given conditions, the total purchase amount is 89.325 -/
theorem total_purchase_theorem :
  total_purchase 0.06 0.30 39.7 = 89.325 := by
  -- Unfold the definition of total_purchase
  unfold total_purchase
  -- Simplify the expression
  simp
  -- The proof is completed with sorry as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_purchase_theorem_l157_15735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_l157_15795

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ 
  a 5 = 1/3 ∧ 
  ∀ n : ℕ, n ≥ 3 → a n = (a (n-1)) / (a (n-2))

theorem sequence_2016 (a : ℕ → ℚ) (h : my_sequence a) : a 2016 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_l157_15795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_nine_l157_15797

/-- The maximum number of items (sandwiches and drinks) that can be purchased with a given amount of money, prioritizing sandwiches over drinks. -/
def max_items (total_money : ℚ) (sandwich_cost : ℚ) (drink_cost : ℚ) : ℕ :=
  let max_sandwiches := (total_money / sandwich_cost).floor.toNat
  let remaining_money := total_money - max_sandwiches * sandwich_cost
  let max_drinks := (remaining_money / drink_cost).floor.toNat
  max_sandwiches + max_drinks

/-- Theorem stating that with 30 yuan, sandwiches costing 4.5 yuan each, and drinks costing 1 yuan each, the maximum number of items that can be purchased is 9. -/
theorem max_items_is_nine :
  max_items 30 4.5 1 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_nine_l157_15797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l157_15729

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem integer_part_of_expression :
  integerPart (1 / Real.sqrt (16 - 6 * Real.sqrt 7)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_expression_l157_15729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l157_15760

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cosine_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

/-- Theorem: The cosine of the angle between vectors (1, 1) and (-1, 2) is √10/10 -/
theorem cosine_angle_specific_vectors :
  cosine_angle (1, 1) (-1, 2) = Real.sqrt 10 / 10 := by
  -- Unfold the definition of cosine_angle
  unfold cosine_angle
  -- Simplify the numerator and denominator
  simp [Real.sqrt_mul, Real.sqrt_sq]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l157_15760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l157_15701

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = -9

-- Define the area of the region
noncomputable def region_area : ℝ := 16 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l157_15701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l157_15785

open Real

theorem max_value_of_expression (x y : ℝ) (hx : x ∈ Set.Ioo 0 (π/2)) (hy : y ∈ Set.Ioo 0 (π/2)) :
  let A := (sqrt (cos x * cos y)) / (sqrt (tan x⁻¹) + sqrt (tan y⁻¹))
  A ≤ sqrt 2 / 4 ∧ ∃ x y, x ∈ Set.Ioo 0 (π/2) ∧ y ∈ Set.Ioo 0 (π/2) ∧ A = sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l157_15785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_A_pass_prob_total_assessments_three_prob_l157_15771

/-- Represents a teacher in the assessment system -/
structure Teacher where
  pass_prob : ℚ  -- Probability of passing each assessment
  deriving Repr

/-- The assessment system for creating a civilized city -/
structure AssessmentSystem where
  A : Teacher
  B : Teacher

-- Define Teacher A
def teacher_A : Teacher := { pass_prob := 2/3 }

-- Define Teacher B
def teacher_B : Teacher := { pass_prob := 1/2 }

/-- Theorem: Probability of Teacher A passing the assessment -/
theorem teacher_A_pass_prob (system : AssessmentSystem) (h : system.A = teacher_A) :
  (system.A.pass_prob + (1 - system.A.pass_prob) * system.A.pass_prob) = 8/9 := by
  sorry

/-- Theorem: Probability that the total number of assessments for A and B is 3 -/
theorem total_assessments_three_prob (system : AssessmentSystem) 
  (h1 : system.A = teacher_A) (h2 : system.B = teacher_B) :
  (system.A.pass_prob * (1 - system.B.pass_prob) + 
   (1 - system.A.pass_prob) * system.B.pass_prob) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_A_pass_prob_total_assessments_three_prob_l157_15771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_serves_eight_l157_15728

/-- Calculate the number of people served by a pie given its ingredients' costs and cost per serving -/
def people_served (apple_pounds : ℚ) (apple_price_per_pound : ℚ) (crust_price : ℚ) 
                  (lemon_price : ℚ) (butter_price : ℚ) (price_per_serving : ℚ) : ℕ :=
  let total_cost := apple_pounds * apple_price_per_pound + crust_price + lemon_price + butter_price
  (total_cost / price_per_serving).floor.toNat

/-- Theorem stating that the pie serves 8 people given the specified costs -/
theorem pie_serves_eight :
  people_served 2 2 2 (1/2) (3/2) 1 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_serves_eight_l157_15728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladybug_journey_l157_15705

def ladybug_movements : List ℚ := [5, -6, 10, -5, -6, 12, -10]
def ladybug_speed : ℚ := 1/2

theorem ladybug_journey :
  (ladybug_movements.sum = 0) ∧
  (ladybug_movements.map abs).sum / ladybug_speed = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladybug_journey_l157_15705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_attempt_second_attempt_race_times_l157_15776

/-- Two runners A and B compete in a 5000-meter race -/
noncomputable def race_distance : ℝ := 5000

/-- A's speed in meters per minute -/
noncomputable def speed_A : ℝ := race_distance / 15

/-- B's speed in meters per minute -/
noncomputable def speed_B : ℝ := race_distance / 20

/-- First attempt: A gives B a 1 km head start and finishes 1 minute earlier -/
theorem first_attempt :
  (1000 / speed_B) = (race_distance / speed_A) + 1 := by
  sorry

/-- Second attempt: A gives B an 8-minute head start and is 1 km from finish when B finishes -/
theorem second_attempt :
  (race_distance / speed_B) = ((race_distance - 1000) / speed_A) + 8 := by
  sorry

/-- Prove that A completes 5000 meters in 15 minutes and B in 20 minutes -/
theorem race_times : (race_distance / speed_A = 15) ∧ (race_distance / speed_B = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_attempt_second_attempt_race_times_l157_15776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_existence_condition_chord_length_when_m_is_one_l157_15702

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + m - 3 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- Theorem for the range of m
theorem circle_existence_condition (m : ℝ) :
  (∃ x y : ℝ, circle_equation x y m) → m < 7 := by
  sorry

-- Theorem for the chord length when m = 1
theorem chord_length_when_m_is_one :
  let chord_length := 2 * Real.sqrt 2
  ∃ x1 y1 x2 y2 : ℝ,
    circle_equation x1 y1 1 ∧
    circle_equation x2 y2 1 ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_existence_condition_chord_length_when_m_is_one_l157_15702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_increase_l157_15722

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The increase in area when each side of an equilateral triangle is increased by d -/
noncomputable def areaIncrease (s d : ℝ) : ℝ :=
  equilateralTriangleArea (s + d) - equilateralTriangleArea s

theorem equilateral_triangle_area_increase :
  ∃ (s : ℝ), s > 0 ∧ equilateralTriangleArea s = 36 * Real.sqrt 3 ∧
  areaIncrease s 3 = 20.25 * Real.sqrt 3 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_increase_l157_15722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_4_l157_15752

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 18 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

-- State the theorem
theorem f_of_g_of_4 : f (g 4) = (57 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_4_l157_15752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cutting_theorem_l157_15717

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a configuration of smaller cubes within a larger cube -/
structure CubeConfiguration where
  large_cube : Cube
  small_cubes : List Cube

/-- Checks if a configuration is valid according to the problem conditions -/
def is_valid_configuration (config : CubeConfiguration) : Prop :=
  -- The large cube has edge length 4
  config.large_cube.edge = 4 ∧
  -- All small cubes have whole number edge lengths
  (∀ c, c ∈ config.small_cubes → c.edge > 0) ∧
  -- Not all small cubes are the same size
  (∃ c1 c2, c1 ∈ config.small_cubes ∧ c2 ∈ config.small_cubes ∧ c1.edge ≠ c2.edge) ∧
  -- The total volume of small cubes equals the volume of the large cube
  (config.small_cubes.map (λ c => c.edge ^ 3)).sum = config.large_cube.edge ^ 3

/-- The main theorem to be proved -/
theorem cube_cutting_theorem :
  ∃! config : CubeConfiguration,
    is_valid_configuration config ∧
    config.small_cubes.length = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cutting_theorem_l157_15717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumradius_l157_15749

/-- The radius of the circumscribed sphere of a tetrahedron -/
noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (Real.sqrt (4 * c^4 - a^2 * b^2)) / (2 * Real.sqrt (4 * c^2 - a^2 - b^2))

/-- Theorem: The radius of the circumscribed sphere of a tetrahedron
    with one edge a, its opposite edge b, and remaining edges c,
    is given by (√(4c^4 - a^2b^2)) / (2√(4c^2 - a^2 - b^2)) -/
theorem tetrahedron_circumradius (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ R : ℝ, R = circumradius a b c ∧ R > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumradius_l157_15749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_proof_l157_15770

/-- Given a right triangle ABC with AB = 15 and BC = 20, and a square BDEF with
    EH (height of triangle EMN) = 2, prove that the area of square BDEF is 100. -/
theorem square_area_proof (A B C D E F H M N : ℝ × ℝ) : 
  -- Right triangle ABC
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AB = 15
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 →
  -- BC = 20
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 20 →
  -- BDEF is a square
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) ∧
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) ∧
  Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) →
  -- EH (height of triangle EMN) = 2
  Real.sqrt ((H.1 - E.1)^2 + (H.2 - E.2)^2) = 2 →
  -- Area of square BDEF is 100
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_proof_l157_15770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l157_15715

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n : ℕ, a (n + 1) = a n * q
  h2 : ∀ n : ℕ, a n > 0

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 1
  else g.a 1 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_problem (g : GeometricSequence)
    (h1 : g.a 1 = 3)
    (h2 : g.a 2 * g.a 4 = 144) :
    g.q = 2 ∧ geometricSum g 10 = 3069 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l157_15715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l157_15727

/-- Predicate to represent that a given equation in x and y is a hyperbola -/
def IsHyperbola (x y k : ℝ) : Prop := 
  x^2 / (k + 2) - y^2 / (5 - k) = 1

/-- 
Given that the equation (x^2)/(k+2) - (y^2)/(5-k) = 1 represents a hyperbola,
prove that k must satisfy -2 < k < 5
-/
theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, IsHyperbola x y k) → 
  -2 < k ∧ k < 5 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_k_range_l157_15727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l157_15789

/-- The volume of a regular tetrahedron with base edge length 2 and height 1 is √3/3 -/
theorem regular_tetrahedron_volume (base_edge : ℝ) (height : ℝ) (h1 : base_edge = 2) (h2 : height = 1) :
  (1 / 3 : ℝ) * (Real.sqrt 3 / 4) * base_edge^2 * height = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l157_15789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_points_l157_15709

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem line_equidistant_from_points (P A B : Point) :
  P.x = 1 ∧ P.y = 2 ∧ A.x = 2 ∧ A.y = 3 ∧ B.x = 0 ∧ B.y = -5 →
  ∃ l : Line,
    P.on_line l ∧
    distance_point_to_line A l = distance_point_to_line B l ∧
    ((l.a = 4 ∧ l.b = -1 ∧ l.c = -2) ∨ (l.a = 1 ∧ l.b = 0 ∧ l.c = -1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_points_l157_15709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_AB_l157_15741

noncomputable def complex_to_vector (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def vector_subtraction (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_AB (OA OB : ℂ) (h1 : OA = 2 + Complex.I) (h2 : OB = 4 - 3*Complex.I) :
  vector_magnitude (vector_subtraction (complex_to_vector OB) (complex_to_vector OA)) = 2 * Real.sqrt 5 := by
  sorry

#check magnitude_of_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_AB_l157_15741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_PA_l157_15724

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 2 + t ∧ y = 2 - 2*t

-- Define a point P on curve C
noncomputable def point_P (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 3 * Real.sin θ)

-- Define the distance between a point and line l
noncomputable def distance_to_l (x y : ℝ) : ℝ := 
  (Real.sqrt 5 / 5) * |4 * x + 3 * y - 6|

-- Define |PA| where A is on line l and angle between PA and l is 30°
noncomputable def length_PA (θ : ℝ) : ℝ := 
  (2 * Real.sqrt 5 / 5) * |5 * Real.sin (θ + Real.arcsin (1/2)) - 6|

-- Theorem statement
theorem max_min_PA : 
  (∀ θ : ℝ, length_PA θ ≤ 2 * Real.sqrt 5 / 5) ∧ 
  (∃ θ : ℝ, length_PA θ = 2 * Real.sqrt 5 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_PA_l157_15724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_dividing_N_l157_15721

def consecutive_digits (start : Nat) (stop : Nat) : Nat :=
  sorry

def N : Nat := consecutive_digits 21 99

theorem highest_power_of_three_dividing_N :
  ∃ (m : Nat), N = 3^2 * m ∧ ¬(∃ (n : Nat), N = 3^3 * n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_dividing_N_l157_15721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_implies_transformed_mean_l157_15765

noncomputable def sample_variance (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  (1 / 5) * (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 - 80)

noncomputable def sample_mean (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  (a₁ + a₂ + a₃ + a₄ + a₅) / 5

noncomputable def transformed_mean (a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  ((2 * a₁ + 1) + (2 * a₂ + 1) + (2 * a₃ + 1) + (2 * a₄ + 1) + (2 * a₅ + 1)) / 5

theorem variance_implies_transformed_mean (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  sample_variance a₁ a₂ a₃ a₄ a₅ = (1 / 5) * (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 - 80) →
  transformed_mean a₁ a₂ a₃ a₄ a₅ = 9 ∨ transformed_mean a₁ a₂ a₃ a₄ a₅ = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_implies_transformed_mean_l157_15765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l157_15756

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the point Q
noncomputable def Q : ℝ × ℝ := (6 * Real.sqrt 5 / 5, 0)

-- Define the constant value
def constant_value : ℝ := 10

-- Theorem statement
theorem ellipse_intersection_property :
  ∃ (Q : ℝ × ℝ), Q.2 = 0 ∧ 
    (Q.1 = 6 * Real.sqrt 5 / 5 ∨ Q.1 = -6 * Real.sqrt 5 / 5) ∧
    ∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 → ellipse B.1 B.2 →
      (∃ (t : ℝ), A = Q + t • (1, 1) ∧ B = Q + t • (1, -1)) →
      1 / (Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2))^2 +
      1 / (Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2))^2 = constant_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l157_15756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l157_15703

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Define the trains
structure Train where
  length : ℝ
  speed : ℝ

-- Define the problem setup
noncomputable def train_A : Train := { length := 1500, speed := 60 * km_hr_to_m_s }
noncomputable def train_B : Train := { length := 1200, speed := 45 * km_hr_to_m_s }
noncomputable def train_C : Train := { length := 900, speed := 30 * km_hr_to_m_s }

-- Define the theorem
theorem train_passing_time :
  let combined_length := train_A.length + train_B.length
  let time_taken := combined_length / train_C.speed
  ∃ ε > 0, |time_taken - 324.07| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l157_15703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l157_15784

/-- A line that passes through a point and intersects positive x and y axes -/
structure IntersectingLine where
  -- Slope and y-intercept of the line
  m : ℝ
  b : ℝ
  -- The line passes through (1,2)
  passes_through_point : m + b = 2
  -- The line intersects positive x and y axes
  intersects_axes : m < 0 ∧ b > 0

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ :=
  (l.b * (-l.b / l.m)) / 2

/-- The theorem stating that 2x + y - 4 = 0 minimizes the triangle area -/
theorem min_triangle_area :
  ∀ l : IntersectingLine,
    triangle_area l ≥ 4 ∧
    (triangle_area l = 4 ↔ l.m = -1/2 ∧ l.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l157_15784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l157_15725

/-- Calculates the distance a truck can travel given the amount of diesel and efficiency parameters. -/
noncomputable def truck_distance (base_distance : ℝ) (base_fuel : ℝ) (fuel : ℝ) (efficiency_threshold : ℝ) (efficiency_decrease : ℝ) : ℝ :=
  let base_efficiency := base_distance / base_fuel
  let full_efficiency_distance := min fuel efficiency_threshold * base_efficiency
  let reduced_efficiency_distance := max (fuel - efficiency_threshold) 0 * (base_efficiency * (1 - efficiency_decrease))
  full_efficiency_distance + reduced_efficiency_distance

/-- Proves that the truck can travel 441 miles with 15 gallons of diesel under the given conditions. -/
theorem truck_travel_distance :
  truck_distance 300 10 15 12 0.1 = 441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l157_15725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l157_15733

noncomputable section

-- Define the functions f and g
def f (p q x : ℝ) : ℝ := x^2 + p*x + q
def g (x : ℝ) : ℝ := x + 1/x^2

-- Define the interval [1, 2]
def I : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_value_of_f (p q : ℝ) :
  (∃ x ∈ I, ∀ y ∈ I, f p q x ≤ f p q y ∧ g x ≤ g y) →
  (∃ x ∈ I, f p q x = g x) →
  (∃ x ∈ I, ∀ y ∈ I, f p q y ≤ f p q x) →
  (∃ x ∈ I, f p q x = 4 - (5/2)*Real.rpow 2 (1/3) + Real.rpow 4 (1/3)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l157_15733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l157_15761

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt ((v.1 ^ 2) + (v.2 ^ 2))

theorem magnitude_of_b (a b : ℝ × ℝ) : 
  a = (3, -2) → a + b = (0, 2) → vector_magnitude b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l157_15761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l157_15754

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  point1 : Point
  point2 : Point

-- Define helper functions
def isBetween (A B C : Point) : Prop :=
  sorry -- Define the condition for B being between A and C

def circumcircle (A B C : Point) : Circle :=
  sorry -- Define the circumcircle of triangle ABC

def circle_with_diameter (A B : Point) : Circle :=
  sorry -- Define the circle with diameter AB

-- Define the problem setup
def problem_setup (O A B E F : Point) (R R' : ℝ) : Prop :=
  let k : Circle := ⟨O, R⟩
  let k' : Circle := ⟨O, R'⟩
  let line1 : Line := ⟨A, B⟩
  let line2 : Line := ⟨E, F⟩
  R < R' ∧
  (O.x - A.x) * (B.x - A.x) + (O.y - A.y) * (B.y - A.y) = 0 ∧
  (O.x - E.x) * (F.x - E.x) + (O.y - E.y) * (F.y - E.y) = 0 ∧
  (A.x - O.x)^2 + (A.y - O.y)^2 = R^2 ∧
  (B.x - O.x)^2 + (B.y - O.y)^2 = R'^2 ∧
  (E.x - O.x)^2 + (E.y - O.y)^2 = R^2 ∧
  (F.x - O.x)^2 + (F.y - O.y)^2 = R'^2 ∧
  isBetween O A B ∧
  isBetween E O F

-- Define the conclusion
def conclusion (O A B E F : Point) : Prop :=
  ∃ M : Point,
    M ∈ { P | (P.x - (circumcircle O A E).center.x)^2 + (P.y - (circumcircle O A E).center.y)^2 = (circumcircle O A E).radius^2 } ∧
    M ∈ { P | (P.x - (circumcircle O B F).center.x)^2 + (P.y - (circumcircle O B F).center.y)^2 = (circumcircle O B F).radius^2 } ∧
    M ∈ { P | (P.x - (circle_with_diameter E F).center.x)^2 + (P.y - (circle_with_diameter E F).center.y)^2 = (circle_with_diameter E F).radius^2 } ∧
    M ∈ { P | (P.x - (circle_with_diameter A B).center.x)^2 + (P.y - (circle_with_diameter A B).center.y)^2 = (circle_with_diameter A B).radius^2 }

-- State the theorem
theorem circles_concurrency (O A B E F : Point) (R R' : ℝ) :
  problem_setup O A B E F R R' → conclusion O A B E F :=
by
  sorry -- The proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l157_15754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l157_15755

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The foci of the hyperbola -/
structure Foci (a c : ℝ) where
  left : ℝ × ℝ
  right : ℝ × ℝ
  distance : left.1 - right.1 = 2 * c

/-- A point on the hyperbola -/
structure Point (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity (a b c : ℝ) (h : Hyperbola a b) (f : Foci a c) 
  (P Q : Point a b) (hP : P.x > 0) (hQ : Q.x > 0) :
  let F₁ := f.left
  let F₂ := f.right
  (P.x - F₂.1)^2 + (P.y - F₂.2)^2 = 4 * ((Q.x - F₂.1)^2 + (Q.y - F₂.2)^2) →
  ((F₁.1 - Q.x) * (P.x - Q.x) + (F₁.2 - Q.y) * (P.y - Q.y) = 0) →
  eccentricity a c = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l157_15755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l157_15769

-- Define the set of solutions
noncomputable def solution_set : Set ℝ := 
  {x : ℝ | (0 < x ∧ x ≤ Real.rpow 5 (-Real.sqrt (Real.log 3 / Real.log 5))) ∨ 
           x = 1 ∨ 
           (Real.rpow 5 (Real.sqrt (Real.log 3 / Real.log 5)) ≤ x)}

-- State the theorem
theorem inequality_equivalence (x : ℝ) (h : 0 < x) : 
  Real.rpow (Real.rpow 125 (1/10)) ((2 * Real.log x / Real.log 5) ^ 2) + 3 ≥ 
  Real.rpow x (Real.log x / Real.log 5) + 3 * Real.rpow (Real.rpow x (1/5)) (Real.log x / Real.log 5) ↔ 
  x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l157_15769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_price_is_15_l157_15745

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := sorry

/-- The regular price of Pony jeans in dollars -/
def pony_price : ℝ := 20

/-- The discount rate for Fox jeans as a decimal -/
def fox_discount : ℝ := sorry

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.18000000000000014

/-- The total savings from buying 3 pairs of Fox jeans and 2 pairs of Pony jeans -/
def total_savings : ℝ := 9

/-- The sum of the two discount rates as a decimal -/
def total_discount_rate : ℝ := 0.22

theorem fox_price_is_15 :
  fox_discount + pony_discount = total_discount_rate →
  3 * (fox_price * fox_discount) + 2 * (pony_price * pony_discount) = total_savings →
  fox_price = 15 := by
  sorry

#check fox_price_is_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_price_is_15_l157_15745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l157_15762

def sequence_a : ℕ → ℕ
  | 0 => 1989^1989
  | n+1 => (Nat.digits 10 (sequence_a n)).sum

theorem a_5_equals_9 : sequence_a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l157_15762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l157_15714

/-- Given two bases R₁ and R₂, and two numbers F₁ and F₂ with the following properties:
    - In base R₁, F₁ = 0.373737... and F₂ = 0.737373...
    - In base R₂, F₁ = 0.252525... and F₂ = 0.525252...
    Prove that R₁ + R₂ = 19 -/
theorem base_sum_theorem (R₁ R₂ F₁ F₂ : ℚ) : 
  (F₁ = (3 * R₁ + 7) / (R₁^2 - 1)) → 
  (F₂ = (7 * R₁ + 3) / (R₁^2 - 1)) → 
  (F₁ = (2 * R₂ + 5) / (R₂^2 - 1)) → 
  (F₂ = (5 * R₂ + 2) / (R₂^2 - 1)) → 
  R₁ + R₂ = 19 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check base_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l157_15714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_l157_15712

theorem cos_60_minus_alpha (α : ℝ) : 
  Real.cos (α - 30 * (π / 180)) + Real.sin α = (3 / 5) * Real.sqrt 3 → 
  Real.cos (60 * (π / 180) - α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_l157_15712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_squared_l157_15738

theorem largest_n_squared :
  ∃ n : ℕ+, n = 8 ∧
  ∀ k : ℕ+, (∃ x y z : ℕ+, k^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6) →
  k ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_squared_l157_15738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_condition_necessary_not_sufficient_l157_15777

theorem hyperbola_condition (a b : ℝ) : 
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1) ↔ a * b < 0 := by sorry

theorem necessary_not_sufficient :
  (∀ a b : ℝ, b < 0 ∧ 0 < a → a * b < 0) ∧
  (∃ a b : ℝ, a * b < 0 ∧ ¬(b < 0 ∧ 0 < a)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_condition_necessary_not_sufficient_l157_15777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_and_k_range_l157_15781

noncomputable section

open Real

/-- The original function f --/
def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

/-- The function y derived from f --/
def y (ω x : ℝ) : ℝ := f (ω * x - π / 12)

/-- The function g derived from f --/
def g (x : ℝ) : ℝ := f (π / 2 - x)

/-- The function h derived from g --/
def h (x : ℝ) : ℝ := g (2 * (x - π / 12))

theorem omega_range_and_k_range :
  (∀ ω : ℝ, 0 < ω ∧ (∀ x ∈ Set.Ioo (π / 2) π, (deriv (y ω)) x < 0) ↔ ω ∈ Set.Icc (1 / 2) (5 / 4)) ∧
  (∀ k : ℝ, (∃ x ∈ Set.Icc (-π / 12) (5 * π / 12), (1 / 2) * h x - k * (sin x + cos x) = 0)
    ↔ k ∈ Set.Icc (-sqrt 2 / 2) (sqrt 2 / 2)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_and_k_range_l157_15781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_example_l157_15779

/-- The surface area of a frustum of a cone. -/
noncomputable def frustumSurfaceArea (r₁ r₂ h : ℝ) : ℝ :=
  Real.pi * (r₁^2 + r₂^2 + (r₁ + r₂) * Real.sqrt (h^2 + (r₂ - r₁)^2))

/-- Theorem: The surface area of a frustum of a cone with top radius 1, bottom radius 2, and height √3 is equal to 11π. -/
theorem frustum_surface_area_example : 
  frustumSurfaceArea 1 2 (Real.sqrt 3) = 11 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_example_l157_15779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_combined_area_l157_15782

/-- Represents the properties of a rectangular sandbox -/
structure Sandbox where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular sandbox -/
def perimeter (s : Sandbox) : ℝ := 2 * (s.width + s.length)

/-- Calculates the area of a rectangular sandbox -/
def area (s : Sandbox) : ℝ := s.width * s.length

/-- Calculates the diagonal of a rectangular sandbox using the Pythagorean theorem -/
noncomputable def diagonal (s : Sandbox) : ℝ := Real.sqrt (s.width ^ 2 + s.length ^ 2)

/-- The main theorem stating the combined area of the two sandboxes -/
theorem sandbox_combined_area :
  ∃ (s1 s2 : Sandbox),
    perimeter s1 = 30 ∧
    s1.length = 2 * s1.width ∧
    s2.length = 3 * s2.width ∧
    diagonal s2 = 15 ∧
    abs ((area s1 + area s2) - 117.42) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_combined_area_l157_15782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_additional_payment_proof_l157_15787

/-- Prove that the husband needs to pay $48 more to split expenses equally --/
def husband_additional_payment (
  salary : ℕ
) (medical_cost : ℕ
) (husband_paid : ℕ
) : ℕ :=
  let couple_contribution := medical_cost / 2
  let total_expenses := salary + couple_contribution
  let each_share := total_expenses / 2
  each_share - husband_paid

#eval husband_additional_payment 160 128 64 -- Expected output: 48

/-- The theorem statement --/
theorem husband_additional_payment_proof (
  salary : ℕ
) (medical_cost : ℕ
) (husband_paid : ℕ
) : husband_additional_payment salary medical_cost husband_paid = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_husband_additional_payment_proof_l157_15787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l157_15783

-- Define f as an odd function on ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Define g as an even function on ℝ
noncomputable def g : ℝ → ℝ := sorry

-- Axiom: f is odd
axiom f_odd (x : ℝ) : f (-x) = -f x

-- Axiom: g is even
axiom g_even (x : ℝ) : g (-x) = g x

-- Axiom: For x < 0, f'(x)g(x) + f(x)g'(x) > 0
axiom derivative_condition (x : ℝ) (h : x < 0) : 
  (deriv f x) * g x + f x * (deriv g x) > 0

-- Axiom: g(-3) = 0
axiom g_neg_three : g (-3) = 0

-- Theorem to prove
theorem solution_set : 
  {x : ℝ | f x * g x < 0} = Set.Ioi (-3) ∩ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l157_15783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_is_one_l157_15753

/-- Regular octagon with side length 2 -/
structure RegularOctagon :=
  (sideLength : ℝ)
  (isSideLength2 : sideLength = 2)

/-- Circle externally placed on the octagon -/
structure ExternalCircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Line passing through two points -/
structure Line :=
  (point1 : ℝ × ℝ)
  (point2 : ℝ × ℝ)

/-- Helper function to calculate circle area -/
noncomputable def circle_area (circle : ExternalCircle) : ℝ :=
  Real.pi * circle.radius ^ 2

/-- Predicate to check if a circle is tangent to side AB of the octagon -/
def circle_tangent_to_AB (octagon : RegularOctagon) (circle : ExternalCircle) : Prop :=
  sorry

/-- Predicate to check if a circle is tangent to side EF of the octagon -/
def circle_tangent_to_EF (octagon : RegularOctagon) (circle : ExternalCircle) : Prop :=
  sorry

/-- Predicate to check if two circles are tangent to a line -/
def circles_tangent_to_line (circle1 : ExternalCircle) (circle2 : ExternalCircle) (line : Line) : Prop :=
  sorry

/-- Predicate to check if a line passes through points C and G of the octagon -/
def line_passes_through_C_and_G (octagon : RegularOctagon) (line : Line) : Prop :=
  sorry

/-- Theorem stating the ratio of areas of two circles tangent to the octagon -/
theorem circle_area_ratio_is_one 
  (octagon : RegularOctagon)
  (circle1 : ExternalCircle)
  (circle2 : ExternalCircle)
  (tangentLine : Line)
  (h1 : circle_tangent_to_AB octagon circle1)
  (h2 : circle_tangent_to_EF octagon circle2)
  (h3 : circles_tangent_to_line circle1 circle2 tangentLine)
  (h4 : line_passes_through_C_and_G octagon tangentLine) :
  (circle_area circle1) / (circle_area circle2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_is_one_l157_15753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_condition_l157_15736

noncomputable section

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def line_l (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Statement 1: Cartesian equation of l
theorem cartesian_equation_of_l (x y m : ℝ) :
  (∃ ρ θ, line_l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  Real.sqrt 3 * x + y + 2 * m = 0 := by
  sorry

-- Statement 2: Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ t, ∃ x y, curve_C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_condition_l157_15736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_condition_l157_15763

/-- The distance from a point (x,y) to a line ax + y + 1 = 0 -/
noncomputable def distanceToLine (x y a : ℝ) : ℝ := 
  abs (a * x + y + 1) / Real.sqrt (a^2 + 1)

/-- Theorem stating that the distances from A(-3,-4) and B(6,3) to the line ax + y + 1 = 0 
    are equal if and only if a = -7/9 or a = -1/3 -/
theorem equal_distance_condition (a : ℝ) : 
  distanceToLine (-3) (-4) a = distanceToLine 6 3 a ↔ a = -7/9 ∨ a = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_condition_l157_15763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l157_15758

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) 
                       (candidate_a_percent : ℚ) (candidate_b_percent : ℚ) (candidate_c_percent : ℚ) :
  total_votes = 800000 →
  invalid_percent = 1/5 →
  candidate_a_percent = 45/100 →
  candidate_b_percent = 30/100 →
  candidate_c_percent = 25/100 →
  ∃ (valid_votes : ℕ) (votes_a : ℕ) (votes_b : ℕ) (votes_c : ℕ),
    valid_votes = total_votes - (invalid_percent * ↑total_votes).floor ∧
    votes_a = (candidate_a_percent * ↑valid_votes).floor ∧
    votes_b = (candidate_b_percent * ↑valid_votes).floor ∧
    votes_c = (candidate_c_percent * ↑valid_votes).floor ∧
    votes_a = 288000 ∧
    votes_b = 192000 ∧
    votes_c = 160000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l157_15758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l157_15704

theorem junior_score (total_students : ℕ) (junior_ratio senior_ratio : ℚ) 
  (combined_avg senior_avg : ℚ) (h1 : junior_ratio = 1/5)
  (h2 : senior_ratio = 4/5) (h3 : junior_ratio + senior_ratio = 1)
  (h4 : combined_avg = 80) (h5 : senior_avg = 78) : ℚ := by
  
  let junior_score := 
    (combined_avg * total_students - senior_avg * (senior_ratio * total_students)) / 
    (junior_ratio * total_students)
  
  have : junior_score = 88 := by sorry
  
  exact junior_score


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l157_15704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_l157_15711

theorem max_value_sin_cos (α β γ : ℝ) (h1 : α ∈ Set.Icc 0 (2 * Real.pi))
  (h2 : β ∈ Set.Icc 0 (2 * Real.pi)) (h3 : γ ∈ Set.Icc 0 (2 * Real.pi))
  (h4 : Real.sin (α - β) = 1/4) :
  ∃ (x : ℝ), x = Real.sqrt 10 / 2 ∧
  ∀ (y : ℝ), y = Real.sin (α - γ) + Real.cos (β - γ) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_l157_15711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_percentage_l157_15700

/-- Represents the duration of Makarla's work day in hours -/
noncomputable def work_day_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
noncomputable def first_meeting_minutes : ℝ := 60

/-- Calculates the total time spent in meetings in minutes -/
noncomputable def total_meeting_time : ℝ :=
  first_meeting_minutes + 2 * first_meeting_minutes + (2 * first_meeting_minutes) / 2

/-- Calculates the percentage of work day spent in meetings -/
noncomputable def meeting_percentage : ℝ :=
  (total_meeting_time / (work_day_hours * 60)) * 100

theorem meeting_time_percentage :
  meeting_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_percentage_l157_15700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l157_15746

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 2 else 2^(-x) - 2

-- State the theorem
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x ≥ 0, f x = 2^x - 2) :
  {x : ℝ | f (x - 1) ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l157_15746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l157_15792

/-- Rectangle ABCD with given dimensions -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- Point of intersection of diagonals -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Pyramid formed from the rectangle -/
structure Pyramid where
  base : Rectangle
  apex : IntersectionPoint

/-- All faces of the pyramid are isosceles triangles -/
def isIsoscelesPyramid (p : Pyramid) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- Volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := sorry

theorem pyramid_volume (p : Pyramid) 
  (h1 : p.base.AB = 10 * Real.sqrt 5)
  (h2 : p.base.BC = 15 * Real.sqrt 5)
  (h3 : isIsoscelesPyramid p) :
  ∃ (v : ℝ), v = 18750 / Real.sqrt 475 ∧ v = volume p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l157_15792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_double_four_l157_15740

theorem probability_double_four : ℝ := by
  let prob_single_four : ℝ := 1 / 6
  let prob_double_four : ℝ := prob_single_four * prob_single_four
  have : prob_double_four = 1 / 36 := by
    -- Proof steps would go here
    sorry
  exact prob_double_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_double_four_l157_15740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l157_15757

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := ![4, m]
def b : Fin 2 → ℝ := ![1, -2]

-- Define the perpendicularity condition
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

-- Theorem statement
theorem vector_sum_magnitude (m : ℝ) :
  perpendicular (a m) b →
  magnitude (λ i => (a m) i + b i) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l157_15757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_elephants_l157_15786

/-- The number of elephants in Utopia National Park --/
def elephants_in_park : ℕ → ℕ := sorry

/-- The constant rate of elephants leaving the park during the exodus --/
def exodus_rate : ℕ := 2880

/-- The duration of the exodus in hours --/
def exodus_duration : ℕ := 4

/-- The constant rate of elephants entering the park after the exodus --/
def entry_rate : ℕ := 1500

/-- The duration of the entry period in hours --/
def entry_duration : ℕ := 7

/-- The final number of elephants in the park --/
def final_elephants : ℕ := 28980

/-- The theorem stating the initial number of elephants in the park --/
theorem initial_elephants :
  elephants_in_park 0 = 27960 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_elephants_l157_15786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_property_l157_15743

/-- A point in 2D space with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 : ℝ)

/-- Area of a triangle formed by three points -/
def triangleArea (p q r : Point) : ℚ :=
  (1/2 : ℚ) * ((q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)).natAbs

/-- Generate n points with coordinates (i, i²) -/
def generatePoints (n : ℕ) : List Point :=
  List.range n |>.map (fun i => ⟨i + 1, (i + 1)^2⟩)

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

theorem points_property (n : ℕ) (h : n ≥ 3) :
  let points := generatePoints n
  (∀ (p q r : Point), p ∈ points → q ∈ points → r ∈ points → ¬collinear p q r) ∧
  (∀ (p q : Point), p ∈ points → q ∈ points → p ≠ q → Irrational (distance p q)) ∧
  (∀ (p q r : Point), p ∈ points → q ∈ points → r ∈ points → 
    p ≠ q ∧ q ≠ r ∧ p ≠ r → (triangleArea p q r).den ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_property_l157_15743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cheburashkas_is_eleven_l157_15775

/-- Represents the number of Cheburashkas in a row -/
def cheburashkas_per_row : ℕ → ℕ := sorry

/-- Represents the total number of characters in a row before erasure -/
def characters_per_row (n : ℕ) : ℕ := 
  2 * cheburashkas_per_row n - 1 + cheburashkas_per_row n

/-- The total number of Krakozyabras after erasure -/
def total_krakozyabras : ℕ := 29

/-- Theorem stating that the total number of Cheburashkas is 11 -/
theorem total_cheburashkas_is_eleven :
  ∃ n : ℕ, 
    n > 0 ∧ 
    cheburashkas_per_row n > 0 ∧
    cheburashkas_per_row (n + 1) > 0 ∧
    2 * (characters_per_row n + characters_per_row (n + 1)) - 2 = total_krakozyabras ∧
    cheburashkas_per_row n + cheburashkas_per_row (n + 1) = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cheburashkas_is_eleven_l157_15775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_solution_set_l157_15720

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 2^(x + b)

-- Theorem 1: Find the value of b
theorem find_b : ∃ b : ℝ, f b 2 = 8 ∧ b = 1 := by sorry

-- Theorem 2: Find the solution set of the inequality
theorem solution_set (x : ℝ) : f 1 x > 332 ↔ x > 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_solution_set_l157_15720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l157_15751

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | 1 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 4}

theorem M_intersect_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l157_15751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_seven_and_half_hours_l157_15719

/-- Represents the travel scenario with Alice, Bob, and Clara -/
structure TravelScenario where
  total_distance : ℚ
  car_speed : ℚ
  bob_cycle_speed : ℚ
  clara_cycle_speed : ℚ
  switch_distance : ℚ

/-- Calculates the total travel time for the given scenario -/
def total_travel_time (scenario : TravelScenario) : ℚ :=
  scenario.switch_distance / scenario.car_speed +
  (scenario.total_distance - scenario.switch_distance) / scenario.clara_cycle_speed

/-- Theorem stating that the total travel time is 7.5 hours -/
theorem travel_time_is_seven_and_half_hours : 
  ∀ (scenario : TravelScenario),
  scenario.total_distance = 150 ∧ 
  scenario.car_speed = 30 ∧ 
  scenario.bob_cycle_speed = 10 ∧
  scenario.clara_cycle_speed = 15 ∧
  scenario.switch_distance / scenario.bob_cycle_speed + 
    (scenario.total_distance - scenario.switch_distance) / scenario.car_speed = 
      total_travel_time scenario →
  total_travel_time scenario = 15/2 := by
  sorry

#eval (15 : ℚ) / 2  -- To verify that 15/2 equals 7.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_is_seven_and_half_hours_l157_15719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_constraint_l157_15716

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: Given points A(1,2,3) and B(4,2,a), if the distance between A and B is √10, then a = 2 or a = 4 -/
theorem distance_constraint (a : ℝ) : 
  distance 1 2 3 4 2 a = Real.sqrt 10 → a = 2 ∨ a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_constraint_l157_15716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l157_15759

-- Define the set of given numbers
noncomputable def givenNumbers : List ℝ := [-5, Real.pi, -1/3, 22/7, Real.sqrt 9, -0.2, Real.sqrt 5, 0, -1.1010010001]

-- Define predicates for each category
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = ↑n
def isNegativeFraction (x : ℝ) : Prop := x < 0 ∧ ∃ p q : ℤ, q ≠ 0 ∧ x = ↑p / ↑q
def isIrrational (x : ℝ) : Prop := ¬(∃ p q : ℤ, q ≠ 0 ∧ x = ↑p / ↑q)

-- Theorem statement
theorem correct_categorization :
  (∀ x ∈ givenNumbers, isInteger x ↔ x ∈ ({-5, Real.sqrt 9, 0} : Set ℝ)) ∧
  (∀ x ∈ givenNumbers, isNegativeFraction x ↔ x ∈ ({-1/3, -0.2} : Set ℝ)) ∧
  (∀ x ∈ givenNumbers, isIrrational x ↔ x ∈ ({Real.pi, Real.sqrt 5, -1.1010010001} : Set ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l157_15759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_relatively_prime_dates_in_september_l157_15744

/-- The number of days in September --/
def september_days : ℕ := 30

/-- The month number of September --/
def september_month : ℕ := 9

/-- A day is relatively prime to the month if their GCD is 1 --/
def is_relatively_prime_date (day : ℕ) (month : ℕ) : Prop :=
  Nat.gcd day month = 1

/-- Count of relatively prime dates in September --/
def count_relatively_prime_dates : ℕ :=
  (Finset.range september_days).filter (λ day => Nat.gcd (day + 1) september_month = 1) |>.card

/-- Theorem: There are 20 relatively prime dates in September --/
theorem twenty_relatively_prime_dates_in_september :
  count_relatively_prime_dates = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_relatively_prime_dates_in_september_l157_15744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l157_15791

theorem tan_ratio_from_sin_sum_diff (x y : ℝ) 
  (h1 : Real.sin (x + y) = 5/8) 
  (h2 : Real.sin (x - y) = 1/4) : 
  Real.tan x / Real.tan y = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l157_15791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_sum_of_primes_l157_15734

theorem not_perfect_square_sum_of_primes (p q : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_perfect_square : ∃ (a : ℕ), p + q^2 = a^2) :
  ¬ ∃ (b : ℕ), p^2 + q^n = b^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_sum_of_primes_l157_15734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_count_l157_15747

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]

def threshold : ℚ := 11/10

def is_selected (x : ℚ) : Bool := x ≥ threshold

theorem selected_count :
  (numbers.filter is_selected).length = 3 := by
  -- Proof goes here
  sorry

#eval numbers.filter is_selected
#eval (numbers.filter is_selected).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_count_l157_15747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_theorem_l157_15773

/-- Represents a bicycle wheel -/
structure Wheel where
  radius : ℝ

/-- Represents a bicycle -/
structure Bicycle where
  front_wheel : Wheel
  back_wheel : Wheel

/-- Calculates the number of revolutions made by the back wheel -/
noncomputable def back_wheel_revolutions (b : Bicycle) (front_revolutions : ℝ) : ℝ :=
  (b.front_wheel.radius * front_revolutions) / b.back_wheel.radius

theorem back_wheel_revolutions_theorem (b : Bicycle) (front_revolutions : ℝ)
    (h1 : b.front_wheel.radius = 3)
    (h2 : b.back_wheel.radius = 0.5)
    (h3 : front_revolutions = 150) :
    back_wheel_revolutions b front_revolutions = 900 := by
  sorry

#check back_wheel_revolutions_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_theorem_l157_15773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_properties_q_properties_l157_15778

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the periodicity of f
axiom f_periodic (x : ℝ) : f (x + 2 * Real.pi) = f x

-- Define functions g and h
noncomputable def g (x : ℝ) : ℝ := (f x + f (-x)) / 2
noncomputable def h (x : ℝ) : ℝ := (f x - f (-x)) / 2

-- Define function p
noncomputable def p (x : ℝ) : ℝ :=
  if x ≠ Real.pi * (Int.floor (x / Real.pi) + 1/2) then
    (g f x - g f (x + Real.pi)) / (2 * Real.cos x)
  else
    0

-- Define function q
noncomputable def q (x : ℝ) : ℝ :=
  if x ≠ Real.pi * Int.floor (2*x / Real.pi) / 2 then
    (h f x + h f (x + Real.pi)) / (2 * Real.sin (2*x))
  else
    0

-- Theorem stating that p is even and π-periodic
theorem p_properties (f : ℝ → ℝ) : 
  (∀ x, p f x = p f (-x)) ∧ (∀ x, p f (x + Real.pi) = p f x) :=
sorry

-- Theorem stating that q is even and π-periodic
theorem q_properties (f : ℝ → ℝ) : 
  (∀ x, q f x = q f (-x)) ∧ (∀ x, q f (x + Real.pi) = q f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_properties_q_properties_l157_15778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_constraint_implies_m_range_l157_15764

noncomputable section

/-- Curve C in Cartesian coordinates -/
def curve_C (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos α + 1, Real.sqrt 2 * Real.sin α + 1)

/-- Line l in polar coordinates -/
def line_l (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi/4) = m

/-- Distance from a point to a line in Cartesian coordinates -/
def distance_point_to_line (x y m : ℝ) : ℝ :=
  |x + y - Real.sqrt 2 * m| / Real.sqrt 2

/-- Theorem: If there exists a point P on curve C such that the distance from P to line l is √2/2,
    then -√2/2 ≤ m ≤ 5√2/2 -/
theorem distance_constraint_implies_m_range :
  ∃ α : ℝ, distance_point_to_line (curve_C α).1 (curve_C α).2 m = Real.sqrt 2 / 2 →
  -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 5 * Real.sqrt 2 / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_constraint_implies_m_range_l157_15764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l157_15793

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem f_and_g_properties :
  ∀ (A ω φ : ℝ),
    A > 0 → ω > 0 → 0 < φ → φ < π →
    (∀ x : ℝ, f A ω φ x ≤ 2) →
    f A ω φ (π / 3) = 2 →
    (∃ k : ℤ, ∀ x : ℝ, f A ω φ x = 0 → f A ω φ (x + π) = 0) →
    (∃ g : ℝ → ℝ, ∀ x : ℝ, g x = f A ω φ x * Real.cos x - 1) →
    (∃ B : Set ℝ, B = Set.Ioo 0 (π / 2)) →
    (∃ C : Set ℝ, C = Set.Ioc (-1) (1 / 2)) →
    (∀ x : ℝ, f A ω φ x = 2 * Real.sin (x + π / 6)) ∧
    (∀ g : ℝ → ℝ, (∀ x : ℝ, g x = f A ω φ x * Real.cos x - 1) →
      ∀ x ∈ Set.Ioo 0 (π / 2), g x ∈ Set.Ioc (-1) (1 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l157_15793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l157_15718

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2 / x - 1
  else if x < 0 then -2 / x - 1
  else 0  -- Defining f(0) as 0 to make it total

theorem f_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x : ℝ, x < 0 → f x = -2 / x - 1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l157_15718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l157_15767

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N = ![![20/9, 5/9], ![7/9, 22/9]] ∧
  N.mulVec (![2, 1] : Fin 2 → ℚ) = ![5, 4] ∧
  N.mulVec (![1, -4] : Fin 2 → ℚ) = ![0, -9] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l157_15767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_less_than_v_main_result_l157_15799

def u : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | n + 2 => 2 / (2 + u (n + 1))

def v : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | n + 2 => 3 / (3 + v (n + 1))

theorem u_less_than_v (n : ℕ) : u n < v n := by
  sorry

theorem main_result : u 2022 < v 2022 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_less_than_v_main_result_l157_15799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l157_15742

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := ((u.1 * a.1 + u.2 * a.2) / (a.1 * a.1 + a.2 * a.2))
  (scalar * a.1, scalar * a.2)

theorem vector_satisfies_projections :
  let u : ℝ × ℝ := (-6/5, 93/5)
  proj (3, 1) u = (45/10, 15/10) ∧ proj (1, 2) u = (36/5, 72/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l157_15742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stabilization_implies_zeros_l157_15766

/-- Given three real numbers x, y, and z, this function computes the next iteration
    of absolute differences. -/
def nextIteration (xyz : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := xyz
  (|x - y|, |y - z|, |z - x|)

/-- A sequence stabilizes if after some number of iterations, it returns to its initial state. -/
def stabilizes (xyz : ℝ × ℝ × ℝ) : Prop :=
  ∃ n : ℕ, (Nat.iterate nextIteration n xyz) = xyz

/-- If x = 1 and the sequence of absolute differences stabilizes, then y = 0 and z = 0. -/
theorem stabilization_implies_zeros (x y z : ℝ) (hx : x = 1) (hstab : stabilizes (x, y, z)) :
  y = 0 ∧ z = 0 := by
  sorry

#check stabilization_implies_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stabilization_implies_zeros_l157_15766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l157_15768

def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (1, 3)

theorem projection_of_a_onto_b :
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj = (1/2, 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_l157_15768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_return_to_start_l157_15750

-- Define the type for a point in the plane
def Point := ℝ × ℝ

-- Define the allowed moves
def move (p : Point) : Set Point :=
  let (x, y) := p
  {(x, y + 2*x), (x, y - 2*x), (x + 2*y, y), (x - 2*y, y)}

-- Define the starting point
noncomputable def start : Point := (1, Real.sqrt 2)

-- Define the game
def game : ℕ → Set Point
  | 0 => {start}
  | n + 1 => ⋃ p ∈ game n, move p

-- Theorem statement
theorem cannot_return_to_start :
  ∀ n : ℕ, start ∉ (game n \ {start}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_return_to_start_l157_15750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l157_15780

noncomputable section

/-- Parabola function y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Triangle ABC with vertices on parabola y = 2x^2 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h₁ : A.2 = parabola A.1
  h₂ : B.2 = parabola B.1
  h₃ : C.2 = parabola C.1

/-- The area of a triangle given its base and height -/
def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_side_length 
  (t : Triangle)
  (h₁ : t.A = (0, 0))
  (h₂ : t.B.2 = t.C.2)  -- BC is parallel to x-axis
  (h₃ : triangleArea (t.C.1 - t.B.1) t.B.2 = 72) :
  t.C.1 - t.B.1 = 2 * Real.rpow 36 (1/3) := by
  sorry

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l157_15780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_log_curve_l157_15772

-- Define the curve and line functions
noncomputable def curve (x : ℝ) : ℝ := Real.log x
noncomputable def line (b : ℝ) (x : ℝ) : ℝ := (1/2) * x + b

-- State the theorem
theorem tangent_line_to_log_curve (b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 
    curve x = line b x ∧ 
    (∀ y : ℝ, y > 0 → curve y ≤ line b y)) →
  b = Real.log 2 - 1 := by
  sorry

#check tangent_line_to_log_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_log_curve_l157_15772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l157_15723

/-- The probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℝ := 3/4

/-- The number of coin tosses -/
def n_tosses : ℕ := 60

/-- The probability of getting an odd number of heads after n tosses -/
noncomputable def P (n : ℕ) : ℝ :=
  1/2 - 1/2 * (-1/2)^n

/-- Theorem: The probability of getting an odd number of heads
    after 60 tosses of an unfair coin with 3/4 probability of heads
    is equal to 1/2(1 - 1/2^60) -/
theorem odd_heads_probability :
  P n_tosses = 1/2 * (1 - 1/2^60) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l157_15723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_given_diagonals_l157_15748

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- The area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ := (r.diagonal1 * r.diagonal2) / 2

/-- Theorem: The area of a rhombus with diagonals 20 cm and 25 cm is 250 square centimeters -/
theorem rhombus_area_with_given_diagonals :
  ∃ (r : Rhombus), r.diagonal1 = 20 ∧ r.diagonal2 = 25 ∧ area r = 250 := by
  -- Construct a rhombus with the given diagonal lengths
  let r : Rhombus := ⟨20, 25⟩
  -- Show that this rhombus satisfies the conditions
  existsi r
  constructor
  · rfl  -- r.diagonal1 = 20
  constructor
  · rfl  -- r.diagonal2 = 25
  -- Calculate the area
  unfold area
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_given_diagonals_l157_15748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l157_15707

/-- The eccentricity of an ellipse -/
def eccentricity_ellipse : ℝ → ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity_hyperbola : ℝ → ℝ := sorry

/-- The angle formed by the foci and their intersection point -/
def foci_intersection_angle : ℝ → ℝ → ℝ → ℝ := sorry

theorem ellipse_hyperbola_eccentricity 
  (e₁ : ℝ) -- eccentricity of the ellipse
  (e₂ : ℝ) -- eccentricity of the hyperbola
  (θ : ℝ)  -- angle formed by the foci and their intersection point
  (h₁ : 0 < e₁ ∧ e₁ < 1) -- condition for ellipse eccentricity
  (h₂ : e₂ > 1) -- condition for hyperbola eccentricity
  (h₃ : e₂ = 2 * e₁) -- given condition
  (h₄ : Real.cos θ = 3/5) -- given condition
  : e₁ = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l157_15707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l157_15713

/-- The set of allowed values for a, b, and c -/
noncomputable def AllowedValues : Set ℝ := {1, 2, 4}

/-- The expression we want to maximize -/
noncomputable def expression (a b c : ℝ) : ℝ := (a / 2) / (b / c)

/-- Theorem stating that the maximum value of the expression is 4 -/
theorem max_expression_value :
  ∃ (a b c : ℝ),
    a ∈ AllowedValues ∧
    b ∈ AllowedValues ∧
    c ∈ AllowedValues ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    expression a b c = 4 ∧
    ∀ (x y z : ℝ),
      x ∈ AllowedValues →
      y ∈ AllowedValues →
      z ∈ AllowedValues →
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      expression x y z ≤ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l157_15713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_squared_l157_15739

/-- A cylindrical barrel with water -/
structure Barrel where
  radius : ℝ
  height : ℝ

/-- A cubic object -/
structure Cube where
  side : ℝ

noncomputable def Barrel.volume (b : Barrel) : ℝ := Real.pi * b.radius^2 * b.height

def Cube.volume (c : Cube) : ℝ := c.side^3

noncomputable def water_displaced (b : Barrel) (c : Cube) : ℝ :=
  min (Cube.volume c) (Barrel.volume b)

/-- The volume of water displaced by a cube in a barrel -/
theorem water_displaced_squared (b : Barrel) (c : Cube) 
  (h1 : b.radius = 3)
  (h2 : b.height = 8)
  (h3 : c.side = 6)
  (h4 : b.height ≥ c.side) :
  (water_displaced b c)^2 = 46656 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_squared_l157_15739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l157_15706

noncomputable def data : List ℝ := [6, 7, 7, 8, 7]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data : variance data = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l157_15706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_is_94_point_5_l157_15788

def carl_cube_volume : ℝ := 3^3
def kate_cube_volume : ℝ := 1.5^3

def total_volume : ℝ :=
  3 * carl_cube_volume + 4 * kate_cube_volume

theorem total_volume_is_94_point_5 :
  total_volume = 94.5 := by
  unfold total_volume carl_cube_volume kate_cube_volume
  norm_num

#eval total_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_is_94_point_5_l157_15788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l157_15796

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem trigonometric_function_properties :
  let f₁ := λ (x : ℝ) => Real.sin (4 * x)
  let f₂ := λ (x : ℝ) => Real.cos (2 * x)
  let f₃ := λ (x : ℝ) => Real.tan (2 * x)
  let f₄ := λ (x : ℝ) => Real.sin (π / 2 - 4 * x)
  (is_even f₄ ∧ has_period f₄ (π / 2)) ∧
  (¬(is_even f₁ ∧ has_period f₁ (π / 2))) ∧
  (¬(is_even f₂ ∧ has_period f₂ (π / 2))) ∧
  (¬(is_even f₃ ∧ has_period f₃ (π / 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l157_15796
