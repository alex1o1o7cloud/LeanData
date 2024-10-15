import Mathlib

namespace NUMINAMATH_CALUDE_investment_increase_l813_81368

/-- Represents the broker's investments over three years -/
def investment_change (S R : ℝ) : ℝ := by
  -- Define the changes for each year
  let year1_stock := S * 1.5
  let year1_real_estate := R * 1.2
  
  let year2_stock := year1_stock * 0.7
  let year2_real_estate := year1_real_estate * 1.1
  
  let year3_stock_initial := year2_stock + 0.5 * S
  let year3_real_estate_initial := year2_real_estate - 0.2 * R
  
  let year3_stock_final := year3_stock_initial * 1.25
  let year3_real_estate_final := year3_real_estate_initial * 0.95
  
  -- Calculate the net change
  let net_change := (year3_stock_final + year3_real_estate_final) - (S + R)
  
  exact net_change

/-- Theorem stating the net increase in investment wealth -/
theorem investment_increase (S R : ℝ) : 
  investment_change S R = 0.9375 * S + 0.064 * R := by
  sorry

end NUMINAMATH_CALUDE_investment_increase_l813_81368


namespace NUMINAMATH_CALUDE_cube_split_2015_l813_81357

/-- The number of odd numbers in the "split" of n^3, for n ≥ 2 -/
def split_count (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number, starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_2015 (m : ℕ) (hm : m > 0) :
  (∃ k, k > 0 ∧ k ≤ split_count m ∧ nth_odd k = 2015) ↔ m = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_2015_l813_81357


namespace NUMINAMATH_CALUDE_equality_of_fractions_implies_equality_of_products_l813_81360

theorem equality_of_fractions_implies_equality_of_products 
  (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : 
  x * (z + t + y) = z * (x + y + t) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_implies_equality_of_products_l813_81360


namespace NUMINAMATH_CALUDE_no_integer_solutions_l813_81345

theorem no_integer_solutions : ¬∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l813_81345


namespace NUMINAMATH_CALUDE_magnitude_of_c_l813_81367

/-- Given vectors a and b, if there exists a vector c satisfying certain conditions, then the magnitude of c is 2√5. -/
theorem magnitude_of_c (a b c : ℝ × ℝ) : 
  a = (-1, 2) →
  b = (3, -6) →
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -1/2 →
  c.1 * (-1) + c.2 * 8 = 5 →
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_c_l813_81367


namespace NUMINAMATH_CALUDE_min_chord_length_l813_81394

/-- The minimum chord length of a circle intersected by a line passing through a fixed point -/
theorem min_chord_length (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : 
  O = (2, 3) → r = 3 → P = (1, 1) → 
  let d := Real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2)
  ∃ (A B : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧ 
                   (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
                   (∀ (X Y : ℝ × ℝ), 
                     (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2 → 
                     (Y.1 - O.1)^2 + (Y.2 - O.2)^2 = r^2 →
                     Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) ≥ 
                     Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
                   Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_min_chord_length_l813_81394


namespace NUMINAMATH_CALUDE_rogers_candy_problem_l813_81332

/-- Roger's candy problem -/
theorem rogers_candy_problem (initial_candies given_candies remaining_candies : ℕ) :
  given_candies = 3 →
  remaining_candies = 92 →
  initial_candies = remaining_candies + given_candies →
  initial_candies = 95 :=
by sorry

end NUMINAMATH_CALUDE_rogers_candy_problem_l813_81332


namespace NUMINAMATH_CALUDE_lucia_dance_class_cost_l813_81352

/-- Represents the cost calculation for Lucia's dance classes over a six-week period. -/
def dance_class_cost (hip_hop_cost ballet_cost jazz_cost salsa_cost contemporary_cost : ℚ)
  (hip_hop_freq ballet_freq jazz_freq salsa_freq contemporary_freq : ℚ)
  (extra_salsa_cost : ℚ) : ℚ :=
  hip_hop_cost * hip_hop_freq * 6 +
  ballet_cost * ballet_freq * 6 +
  jazz_cost * jazz_freq * 6 +
  salsa_cost * (6 / salsa_freq) +
  contemporary_cost * (6 / contemporary_freq) +
  extra_salsa_cost

/-- Proves that the total cost of Lucia's dance classes for a six-week period is $465.50. -/
theorem lucia_dance_class_cost :
  dance_class_cost 10.50 12.25 8.75 15 10 3 2 1 2 3 12 = 465.50 := by
  sorry

end NUMINAMATH_CALUDE_lucia_dance_class_cost_l813_81352


namespace NUMINAMATH_CALUDE_money_division_l813_81383

/-- Given a sum of money divided among three people a, b, and c, with the following conditions:
  1. a gets one-third of what b and c together get
  2. b gets two-sevenths of what a and c together get
  3. a receives $20 more than b
  Prove that the total amount shared is $720 -/
theorem money_division (a b c : ℚ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 20 →
  a + b + c = 720 := by
  sorry


end NUMINAMATH_CALUDE_money_division_l813_81383


namespace NUMINAMATH_CALUDE_ellipse_properties_l813_81330

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance 2√3 -/
def Ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 - b^2 = 3

/-- The equation of the ellipse -/
def EllipseEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

/-- Line l₁ with slope k intersecting the ellipse at two points -/
def Line1 (k : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y = k * x ∧ k ≠ 0

/-- Line l₂ with slope k/4 passing through a point on the ellipse -/
def Line2 (k : ℝ) (x₀ y₀ : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y - y₀ = (k/4) * (x - x₀)

theorem ellipse_properties (a b : ℝ) (h : Ellipse a b) :
  ∃ (k x₀ y₀ x₁ y₁ : ℝ),
    EllipseEquation a b (x₀, y₀) ∧
    EllipseEquation a b (x₁, y₁) ∧
    Line1 k (x₀, y₀) ∧
    Line2 k x₀ y₀ (x₁, y₁) ∧
    (y₁ - y₀) * (x₁ - x₀) = -1/k ∧
    (∀ (x y : ℝ), EllipseEquation a b (x, y) ↔ x^2/4 + y^2 = 1) ∧
    (∃ (M N : ℝ),
      Line2 k x₀ y₀ (M, 0) ∧
      Line2 k x₀ y₀ (0, N) ∧
      ∀ (M' N' : ℝ),
        Line2 k x₀ y₀ (M', 0) ∧
        Line2 k x₀ y₀ (0, N') →
        abs (M * N) / 2 ≤ 9/8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l813_81330


namespace NUMINAMATH_CALUDE_square_diagonal_length_l813_81373

/-- The length of the diagonal of a square with area 72 and perimeter 33.94112549695428 is 12 -/
theorem square_diagonal_length (area : ℝ) (perimeter : ℝ) (h_area : area = 72) (h_perimeter : perimeter = 33.94112549695428) :
  let side := (perimeter / 4 : ℝ)
  Real.sqrt (2 * side ^ 2) = 12 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l813_81373


namespace NUMINAMATH_CALUDE_intersection_sum_l813_81388

/-- Two circles with centers on the line x + y = 0 intersect at points M(m, 1) and N(-1, n) -/
def circles_intersection (m n : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 = 0) ∧ 
    (c₂.1 + c₂.2 = 0) ∧ 
    ((m - c₁.1)^2 + (1 - c₁.2)^2 = (-1 - c₁.1)^2 + (n - c₁.2)^2) ∧
    ((m - c₂.1)^2 + (1 - c₂.2)^2 = (-1 - c₂.1)^2 + (n - c₂.2)^2)

/-- The theorem to be proved -/
theorem intersection_sum (m n : ℝ) (h : circles_intersection m n) : m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l813_81388


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l813_81341

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : GeometricSequence) : ℝ :=
  Classical.choose (seq.is_geometric 1 (by norm_num))

theorem geometric_sequence_ratio 
  (seq : GeometricSequence) 
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  common_ratio seq = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l813_81341


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l813_81376

/-- The perimeter of a semicircle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) : 
  let P := r * Real.pi + 2 * r
  P = r * Real.pi + 2 * r :=
by sorry

#check semicircle_perimeter

end NUMINAMATH_CALUDE_semicircle_perimeter_l813_81376


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l813_81315

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a > 0) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l813_81315


namespace NUMINAMATH_CALUDE_quadrilateral_trapezoid_or_parallelogram_l813_81370

/-- A quadrilateral with areas of triangles formed by diagonals -/
structure Quadrilateral where
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  area_positive : s₁ > 0 ∧ s₂ > 0 ∧ s₃ > 0 ∧ s₄ > 0

/-- Definition of a trapezoid or parallelogram based on triangle areas -/
def is_trapezoid_or_parallelogram (q : Quadrilateral) : Prop :=
  q.s₁ = q.s₃ ∨ q.s₂ = q.s₄

/-- The main theorem -/
theorem quadrilateral_trapezoid_or_parallelogram (q : Quadrilateral) :
  (q.s₁ + q.s₂) * (q.s₃ + q.s₄) = (q.s₁ + q.s₄) * (q.s₂ + q.s₃) →
  is_trapezoid_or_parallelogram q :=
by sorry


end NUMINAMATH_CALUDE_quadrilateral_trapezoid_or_parallelogram_l813_81370


namespace NUMINAMATH_CALUDE_natural_roots_equation_l813_81327

theorem natural_roots_equation :
  ∃ (x y z t : ℕ),
    17 * (x * y * z * t + x * y + x * t + z * t + 1) - 54 * (y * z * t + y + t) = 0 ∧
    x = 3 ∧ y = 5 ∧ z = 1 ∧ t = 2 :=
by sorry

end NUMINAMATH_CALUDE_natural_roots_equation_l813_81327


namespace NUMINAMATH_CALUDE_cosine_of_angle_through_point_l813_81395

/-- If the terminal side of angle α passes through point P (-1, -√2), then cos α = -√3/3 -/
theorem cosine_of_angle_through_point :
  ∀ α : Real,
  let P : Real × Real := (-1, -Real.sqrt 2)
  (∃ t : Real, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_through_point_l813_81395


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l813_81336

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) : 
  ((-1 : ℝ)^2 - k * (-1) + 1 = 0) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l813_81336


namespace NUMINAMATH_CALUDE_no_right_triangle_with_perimeter_5_times_inradius_l813_81387

theorem no_right_triangle_with_perimeter_5_times_inradius :
  ¬∃ (a b c : ℕ+), 
    (a.val^2 + b.val^2 = c.val^2) ∧  -- right triangle condition
    ((a.val + b.val + c.val : ℚ) = 5 * (a.val * b.val : ℚ) / (a.val + b.val + c.val : ℚ)) 
    -- perimeter = 5 * in-radius condition
  := by sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_perimeter_5_times_inradius_l813_81387


namespace NUMINAMATH_CALUDE_cork_price_calculation_l813_81349

/-- The price of a bottle of wine with a cork -/
def bottle_with_cork : ℚ := 2.10

/-- The additional cost of a bottle without a cork compared to the cork price -/
def additional_cost : ℚ := 2.00

/-- The price of the cork -/
def cork_price : ℚ := 0.05

theorem cork_price_calculation :
  cork_price + (cork_price + additional_cost) = bottle_with_cork :=
by sorry

end NUMINAMATH_CALUDE_cork_price_calculation_l813_81349


namespace NUMINAMATH_CALUDE_binomial_expansion_102_l813_81379

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104040401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_102_l813_81379


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l813_81317

/-- The side length of a smaller cube inscribed between a sphere and one face of a larger cube inscribed in the sphere. -/
theorem smaller_cube_side_length (R : ℝ) : 
  R = Real.sqrt 3 →  -- Radius of the sphere
  ∃ (x : ℝ), 
    x > 0 ∧  -- Side length of smaller cube is positive
    x < 2 ∧  -- Side length of smaller cube is less than that of larger cube
    (1 + x + x * Real.sqrt 2 / 2)^2 = 3 ∧  -- Equation derived from geometric relationships
    x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l813_81317


namespace NUMINAMATH_CALUDE_train_length_l813_81384

/-- The length of a train given its speed, time to cross a platform, and the platform's length. -/
theorem train_length (train_speed : ℝ) (cross_time : ℝ) (platform_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  cross_time = 25 → 
  platform_length = 300.04 →
  (train_speed * cross_time - platform_length) = 199.96 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l813_81384


namespace NUMINAMATH_CALUDE_distance_A_to_C_l813_81361

/-- Prove the distance between cities A and C given travel conditions -/
theorem distance_A_to_C (time_E time_F : ℝ) (distance_AB : ℝ) (speed_ratio : ℝ) :
  time_E = 3 →
  time_F = 4 →
  distance_AB = 900 →
  speed_ratio = 4 →
  let speed_E := distance_AB / time_E
  let speed_F := speed_E / speed_ratio
  distance_AB / time_E = 4 * (distance_AB / time_E / speed_ratio) →
  speed_F * time_F = 300 :=
by sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l813_81361


namespace NUMINAMATH_CALUDE_cyclist_journey_time_l813_81301

theorem cyclist_journey_time (a v : ℝ) (h1 : a > 0) (h2 : v > 0) (h3 : a / v = 5) :
  (a / (2 * v)) + (a / (2 * (1.25 * v))) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_time_l813_81301


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l813_81343

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

/-- A right triangle -/
structure RightTriangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point
  is_right_angle : (vertex1.1 - vertex2.1) * (vertex1.1 - vertex3.1) + 
                   (vertex1.2 - vertex2.2) * (vertex1.2 - vertex3.2) = 0

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (c : Circle) (t : RightTriangle) : Prop :=
  let (x1, y1) := t.vertex1
  let (x2, y2) := t.vertex2
  let (x3, y3) := t.vertex3
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x3 - cx)^2 + (y3 - cy)^2 = c.radius^2

/-- Check if a line passes through a point -/
def passesThrough (p1 : Point) (p2 : Point) (p : Point) : Prop :=
  (p.1 - p1.1) * (p2.2 - p1.2) = (p.2 - p1.2) * (p2.1 - p1.1)

theorem inscribed_right_triangle_exists (c : Circle) (A B : Point) 
  (h1 : isInside c A) (h2 : isInside c B) :
  ∃ (t : RightTriangle), isInscribed c t ∧ 
    (passesThrough t.vertex1 t.vertex2 A ∨ passesThrough t.vertex1 t.vertex3 A) ∧
    (passesThrough t.vertex1 t.vertex2 B ∨ passesThrough t.vertex1 t.vertex3 B) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l813_81343


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l813_81304

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l813_81304


namespace NUMINAMATH_CALUDE_thirteenth_number_l813_81375

theorem thirteenth_number (results : Vector ℝ 25) 
  (h1 : results.toList.sum / 25 = 18)
  (h2 : (results.take 12).toList.sum / 12 = 14)
  (h3 : (results.drop 13).toList.sum / 12 = 17) :
  results[12] = 78 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_number_l813_81375


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l813_81319

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 1
def line2 (x y : ℝ) : Prop := 5 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 99 / 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l813_81319


namespace NUMINAMATH_CALUDE_trig_equality_l813_81369

theorem trig_equality : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l813_81369


namespace NUMINAMATH_CALUDE_angies_age_l813_81390

theorem angies_age :
  ∀ (A : ℕ), (2 * A + 4 = 20) → A = 8 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_l813_81390


namespace NUMINAMATH_CALUDE_two_pair_probability_l813_81346

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Number of cards in a poker hand -/
def HandSize : ℕ := 5

/-- Number of ways to choose 5 cards from 52 -/
def TotalOutcomes : ℕ := Nat.choose StandardDeck HandSize

/-- Number of ways to form a two pair -/
def TwoPairOutcomes : ℕ := NumRanks * (Nat.choose CardsPerRank 2) * (NumRanks - 1) * (Nat.choose CardsPerRank 2) * (NumRanks - 2) * CardsPerRank

/-- Probability of forming a two pair -/
def TwoPairProbability : ℚ := TwoPairOutcomes / TotalOutcomes

theorem two_pair_probability : TwoPairProbability = 108 / 1005 := by
  sorry

end NUMINAMATH_CALUDE_two_pair_probability_l813_81346


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l813_81378

def f (x : ℝ) := x^2 + |x - 2|

theorem f_of_one_eq_two : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l813_81378


namespace NUMINAMATH_CALUDE_sum_of_xyz_l813_81385

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l813_81385


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l813_81321

def microphotonics : ℝ := 13
def home_electronics : ℝ := 24
def food_additives : ℝ := 15
def genetically_modified_microorganisms : ℝ := 29
def industrial_lubricants : ℝ := 8
def total_circle_degrees : ℝ := 360

def other_sectors_sum : ℝ := 
  microphotonics + home_electronics + food_additives + 
  genetically_modified_microorganisms + industrial_lubricants

def basic_astrophysics_percentage : ℝ := 100 - other_sectors_sum

theorem basic_astrophysics_degrees : 
  (basic_astrophysics_percentage / 100) * total_circle_degrees = 39.6 := by
  sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l813_81321


namespace NUMINAMATH_CALUDE_union_of_sets_l813_81318

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l813_81318


namespace NUMINAMATH_CALUDE_ramsey_33_l813_81359

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- A function type representing a coloring of line segments -/
def Coloring := Fin 9 → Fin 9 → Color

/-- Predicate to check if four points are coplanar -/
def are_coplanar (p₁ p₂ p₃ p₄ : Point3D) : Prop := sorry

/-- Predicate to check if a set of points forms a monochromatic triangle under a given coloring -/
def has_monochromatic_triangle (points : Fin 9 → Point3D) (coloring : Coloring) : Prop := sorry

theorem ramsey_33 (points : Fin 9 → Point3D) 
  (h_not_coplanar : ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l → 
    ¬are_coplanar (points i) (points j) (points k) (points l)) :
  ∀ coloring : Coloring, has_monochromatic_triangle points coloring := by
  sorry

end NUMINAMATH_CALUDE_ramsey_33_l813_81359


namespace NUMINAMATH_CALUDE_cereal_box_capacity_l813_81326

theorem cereal_box_capacity (cups_per_serving : ℕ) (total_servings : ℕ) : 
  cups_per_serving = 2 → total_servings = 9 → cups_per_serving * total_servings = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_capacity_l813_81326


namespace NUMINAMATH_CALUDE_fish_pond_population_l813_81391

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged : ℕ := 80

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 80

/-- The number of tagged fish found in the second catch -/
def tagged_in_second : ℕ := 2

/-- The approximate total number of fish in the pond -/
def total_fish : ℕ := 3200

/-- Theorem stating that the given conditions lead to the approximate number of fish in the pond -/
theorem fish_pond_population :
  (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l813_81391


namespace NUMINAMATH_CALUDE_student_a_test_questions_l813_81339

/-- Represents the grading system and test results for Student A -/
structure TestResults where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses

/-- The total number of questions on the test -/
def total_questions (t : TestResults) : ℕ :=
  t.correct_responses + t.incorrect_responses

/-- Theorem stating that the total number of questions on Student A's test is 100 -/
theorem student_a_test_questions :
  ∃ t : TestResults, t.correct_responses = 90 ∧ t.score = 70 ∧ total_questions t = 100 := by
  sorry


end NUMINAMATH_CALUDE_student_a_test_questions_l813_81339


namespace NUMINAMATH_CALUDE_latest_start_time_l813_81356

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Represents a turkey roasting scenario -/
structure TurkeyRoast where
  num_turkeys : ℕ
  turkey_weight : ℕ
  roast_time_per_pound : ℕ
  dinner_time : Time

def total_roast_time (tr : TurkeyRoast) : ℕ :=
  tr.num_turkeys * tr.turkey_weight * tr.roast_time_per_pound

def subtract_hours (t : Time) (h : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes - h * 60
  ⟨total_minutes / 60, total_minutes % 60, by sorry⟩

theorem latest_start_time (tr : TurkeyRoast) 
  (h_num : tr.num_turkeys = 2)
  (h_weight : tr.turkey_weight = 16)
  (h_roast_time : tr.roast_time_per_pound = 15)
  (h_dinner : tr.dinner_time = ⟨18, 0, by sorry⟩) :
  subtract_hours tr.dinner_time (total_roast_time tr / 60) = ⟨10, 0, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_latest_start_time_l813_81356


namespace NUMINAMATH_CALUDE_aj_has_370_stamps_l813_81333

/-- The number of stamps each person has -/
structure StampCollection where
  aj : ℕ  -- AJ's stamps
  kj : ℕ  -- KJ's stamps
  cj : ℕ  -- CJ's stamps

/-- The conditions of the stamp collection problem -/
def StampProblemConditions (s : StampCollection) : Prop :=
  (s.cj = 2 * s.kj + 5) ∧  -- CJ has 5 more than twice KJ's stamps
  (s.kj = s.aj / 2) ∧      -- KJ has half as many as AJ
  (s.aj + s.kj + s.cj = 930)  -- Total stamps is 930

/-- The theorem stating that AJ has 370 stamps given the conditions -/
theorem aj_has_370_stamps :
  ∀ s : StampCollection, StampProblemConditions s → s.aj = 370 := by
  sorry

end NUMINAMATH_CALUDE_aj_has_370_stamps_l813_81333


namespace NUMINAMATH_CALUDE_toms_living_room_length_l813_81335

def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250
def boxes_needed : ℕ := 7

theorem toms_living_room_length : 
  (flooring_laid + boxes_needed * flooring_per_box) / room_width = 16 := by
  sorry

end NUMINAMATH_CALUDE_toms_living_room_length_l813_81335


namespace NUMINAMATH_CALUDE_novel_writing_speed_l813_81397

/-- Calculates the average writing speed given the total number of words and hours spent writing. -/
def average_writing_speed (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  total_words / total_hours

/-- Theorem stating that for a novel with 60,000 words completed in 120 hours, 
    the average writing speed is 500 words per hour. -/
theorem novel_writing_speed :
  average_writing_speed 60000 120 = 500 := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_speed_l813_81397


namespace NUMINAMATH_CALUDE_college_selection_ways_l813_81362

theorem college_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 2 →
  (m * (n - m).choose (k - 1)) + ((n - m).choose k) = 16 := by sorry

end NUMINAMATH_CALUDE_college_selection_ways_l813_81362


namespace NUMINAMATH_CALUDE_unique_function_existence_l813_81381

/-- Given positive real numbers a and b, and X being the set of non-negative real numbers,
    there exists a unique function f: X → X such that f(f(x)) = b(a + b)x - af(x) for all x ∈ X,
    and this function is f(x) = bx. -/
theorem unique_function_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃! f : {x : ℝ | 0 ≤ x} → {x : ℝ | 0 ≤ x},
    (∀ x, f (f x) = b * (a + b) * x - a * f x) ∧
    (∀ x, f x = b * x) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l813_81381


namespace NUMINAMATH_CALUDE_jellybean_probability_l813_81355

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def jellybeans_picked : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + white_jellybeans) 1) /
  Nat.choose total_jellybeans jellybeans_picked = 14 / 99 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l813_81355


namespace NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l813_81386

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : d > 0) :
  let frac := (72 : ℚ) / ((2^4 * 3^6) : ℚ)
  ∃ (a b c : ℕ) (r : ℚ),
    frac = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + r ∧
    0 < a ∧ a < 10 ∧
    0 < b ∧ b < 10 ∧
    0 < c ∧ c < 10 ∧
    0 ≤ r ∧ r < 1/1000 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l813_81386


namespace NUMINAMATH_CALUDE_no_common_root_l813_81329

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, x₀^2 + b*x₀ + c = 0 ∧ x₀^2 + a*x₀ + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l813_81329


namespace NUMINAMATH_CALUDE_gcf_of_60_and_75_l813_81325

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_75_l813_81325


namespace NUMINAMATH_CALUDE_sqrt_six_minus_one_over_two_lt_one_l813_81371

theorem sqrt_six_minus_one_over_two_lt_one : (Real.sqrt 6 - 1) / 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_minus_one_over_two_lt_one_l813_81371


namespace NUMINAMATH_CALUDE_subtraction_reciprocal_l813_81365

theorem subtraction_reciprocal (x y : ℝ) (h : x - y = 3 * x * y) :
  1 / x - 1 / y = -3 :=
by sorry

end NUMINAMATH_CALUDE_subtraction_reciprocal_l813_81365


namespace NUMINAMATH_CALUDE_find_x_l813_81348

theorem find_x : ∃ x : ℚ, (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l813_81348


namespace NUMINAMATH_CALUDE_smallest_possible_value_l813_81340

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (a = 56) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, (Nat.gcd 56 c = y + 7) ∧ (Nat.lcm 56 c = y * (y + 7)))) →
  b = 294 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l813_81340


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l813_81389

theorem max_slope_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m : ℚ) (x y : ℤ), 1/3 < m → m < a → 1 ≤ x → x ≤ 50 →
    y = m * x + 3 → ¬(∃ (x' y' : ℤ), x' = x ∧ y' = y)) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ) (x y : ℤ), 1/3 < m → m < a' → 1 ≤ x → x ≤ 50 →
      y = m * x + 3 ∧ (∃ (x' y' : ℤ), x' = x ∧ y' = y)) :=
by sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l813_81389


namespace NUMINAMATH_CALUDE_find_a20_l813_81337

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem find_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 20 = -30 := by
  sorry

end NUMINAMATH_CALUDE_find_a20_l813_81337


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l813_81347

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c

def condition2 (t : Triangle) : Prop :=
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.A = Real.pi / 3

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l813_81347


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l813_81328

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) : 
  c / d = -11 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l813_81328


namespace NUMINAMATH_CALUDE_max_value_of_f_l813_81338

open Real

noncomputable def f (x : ℝ) := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l813_81338


namespace NUMINAMATH_CALUDE_quadratic_prime_values_l813_81392

theorem quadratic_prime_values (p : ℕ) (hp : p > 1) :
  ∀ x : ℕ, 0 ≤ x ∧ x < p →
    (Nat.Prime (x^2 - x + p) ↔ (x = 0 ∨ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_prime_values_l813_81392


namespace NUMINAMATH_CALUDE_employed_males_percentage_l813_81344

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 75)
  : (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100 = 15) := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l813_81344


namespace NUMINAMATH_CALUDE_quadratic_factorization_l813_81364

theorem quadratic_factorization :
  ∀ x : ℝ, 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l813_81364


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l813_81380

theorem no_solution_implies_m_equals_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 * x) / (x - 1) - 1 ≠ m / (x - 1)) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l813_81380


namespace NUMINAMATH_CALUDE_triangle_inequality_l813_81303

/-- Given a triangle with side lengths a, b, and c, 
    prove the inequality and its equality condition --/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0) ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l813_81303


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l813_81308

theorem sum_of_cubes_divisibility (a b c : ℤ) : 
  (3 ∣ (a + b + c)) → (3 ∣ (a^3 + b^3 + c^3)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l813_81308


namespace NUMINAMATH_CALUDE_positive_expressions_l813_81363

theorem positive_expressions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + x^2 * z ∧ 
  0 < y + x^2 ∧ 
  0 < y + y^2 ∧ 
  0 < y + 2 * z := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l813_81363


namespace NUMINAMATH_CALUDE_power_sixteen_seven_fourths_l813_81398

theorem power_sixteen_seven_fourths : (16 : ℝ) ^ (7/4) = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_seven_fourths_l813_81398


namespace NUMINAMATH_CALUDE_additional_cards_l813_81307

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) : 
  total_cards = 160 ∧ complete_decks = 3 ∧ cards_per_deck = 52 →
  total_cards - (complete_decks * cards_per_deck) = 4 := by
sorry

end NUMINAMATH_CALUDE_additional_cards_l813_81307


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l813_81399

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  ∃ (speed_B : ℝ), speed_B = 12 ∧ 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference := by
  sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l813_81399


namespace NUMINAMATH_CALUDE_sequence_on_line_geometric_l813_81372

/-- Given a sequence {a_n} where (n, a_n) is on the line 2x - y + 1 = 0,
    if a_1, a_4, and a_m form a geometric sequence, then m = 13. -/
theorem sequence_on_line_geometric (a : ℕ → ℝ) :
  (∀ n, 2 * n - a n + 1 = 0) →
  (∃ r, a 4 = a 1 * r ∧ a m = a 4 * r) →
  m = 13 :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_geometric_l813_81372


namespace NUMINAMATH_CALUDE_olivia_wallet_problem_l813_81393

theorem olivia_wallet_problem (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by sorry

end NUMINAMATH_CALUDE_olivia_wallet_problem_l813_81393


namespace NUMINAMATH_CALUDE_toy_pricing_and_profit_l813_81306

/-- Represents the order quantity and price for toys -/
structure ToyOrder where
  quantity : ℕ
  price : ℚ

/-- Calculates the factory price based on order quantity -/
def factoryPrice (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else if x < 600 then max (62 - x / 50) 50
  else 50

/-- Calculates the profit for a given order quantity -/
def profit (x : ℕ) : ℚ := (factoryPrice x - 40) * x

theorem toy_pricing_and_profit :
  (∃ x : ℕ, x > 100 ∧ factoryPrice x = 50 → x = 600) ∧
  (∀ x : ℕ, x > 0 → factoryPrice x = 
    if x ≤ 100 then 60
    else if x < 600 then 62 - x / 50
    else 50) ∧
  profit 500 = 6000 := by
  sorry


end NUMINAMATH_CALUDE_toy_pricing_and_profit_l813_81306


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l813_81331

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/5 : ℝ) * total_weight +     -- Weight of sand
  (3/4 : ℝ) * total_weight +     -- Weight of water
  6 = total_weight →             -- Weight of gravel
  total_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l813_81331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l813_81310

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_2 + a_7 + a_15 = 12,
    prove that a_8 = 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 + a 15 = 12) : a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l813_81310


namespace NUMINAMATH_CALUDE_store_pricing_strategy_l813_81314

theorem store_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := 0.7 * list_price
  let marked_price := 1.07 * list_price
  let selling_price := 0.85 * marked_price
  selling_price = 1.3 * purchase_price :=
by sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l813_81314


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l813_81350

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 894 / 1000) :
  (total_oranges * (1 - rotten_oranges_percentage) + total_bananas * (1 - (4 / 100 : ℚ))) / (total_oranges + total_bananas) = good_fruits_percentage :=
by sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l813_81350


namespace NUMINAMATH_CALUDE_symmetry_of_graphs_l813_81320

theorem symmetry_of_graphs (f : ℝ → ℝ) (a : ℝ) :
  ∀ x y : ℝ, f (a - x) = y ↔ f (x - a) = y :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_graphs_l813_81320


namespace NUMINAMATH_CALUDE_average_age_parents_and_children_l813_81305

theorem average_age_parents_and_children (num_children : ℕ) (num_parents : ℕ) 
  (avg_age_children : ℝ) (avg_age_parents : ℝ) :
  num_children = 40 →
  num_parents = 60 →
  avg_age_children = 12 →
  avg_age_parents = 35 →
  (num_children * avg_age_children + num_parents * avg_age_parents) / (num_children + num_parents) = 25.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_parents_and_children_l813_81305


namespace NUMINAMATH_CALUDE_rational_function_value_l813_81358

/-- A rational function with specific properties -/
structure RationalFunction where
  r : ℝ → ℝ
  s : ℝ → ℝ
  r_linear : ∃ a b : ℝ, ∀ x, r x = a * x + b
  s_quadratic : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  asymptote_neg_two : s (-2) = 0
  asymptote_three : s 3 = 0
  passes_origin : r 0 = 0 ∧ s 0 ≠ 0
  passes_one_neg_two : r 1 / s 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.r 2 / f.s 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l813_81358


namespace NUMINAMATH_CALUDE_initial_cars_count_l813_81309

/-- The initial number of cars on the lot -/
def initial_cars : ℕ := sorry

/-- The percentage of initial cars that are silver -/
def initial_silver_percent : ℚ := 1/5

/-- The number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- The percentage of new cars that are silver -/
def new_silver_percent : ℚ := 1/2

/-- The percentage of total cars that are silver after the new shipment -/
def total_silver_percent : ℚ := 2/5

theorem initial_cars_count : initial_cars = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_cars_count_l813_81309


namespace NUMINAMATH_CALUDE_frustum_cross_section_area_l813_81353

theorem frustum_cross_section_area 
  (S' S Q : ℝ) 
  (n m : ℝ) 
  (h1 : S' > 0) 
  (h2 : S > 0) 
  (h3 : Q > 0) 
  (h4 : n > 0) 
  (h5 : m > 0) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) := by
sorry

end NUMINAMATH_CALUDE_frustum_cross_section_area_l813_81353


namespace NUMINAMATH_CALUDE_product_sum_difference_l813_81302

theorem product_sum_difference (a b N : ℤ) : b = 7 → b - a = 2 → a * b = 2 * (a + b) + N → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_l813_81302


namespace NUMINAMATH_CALUDE_ninth_triangle_shaded_fraction_l813_81396

/- Define the sequence of shaded triangles -/
def shaded_triangles (n : ℕ) : ℕ := 2 * n - 1

/- Define the sequence of total triangles -/
def total_triangles (n : ℕ) : ℕ := 4^(n - 1)

/- Theorem statement -/
theorem ninth_triangle_shaded_fraction :
  shaded_triangles 9 / total_triangles 9 = 17 / 65536 := by
  sorry

end NUMINAMATH_CALUDE_ninth_triangle_shaded_fraction_l813_81396


namespace NUMINAMATH_CALUDE_fraction_decimal_places_l813_81316

/-- The number of decimal places when converting the fraction 123456789 / (2^26 * 5^4) to a decimal -/
def decimal_places : ℕ :=
  let numerator : ℕ := 123456789
  let denominator : ℕ := 2^26 * 5^4
  26

theorem fraction_decimal_places :
  decimal_places = 26 :=
sorry

end NUMINAMATH_CALUDE_fraction_decimal_places_l813_81316


namespace NUMINAMATH_CALUDE_total_temp_remaining_days_l813_81311

/-- Calculates the total temperature of the remaining days in a week given specific conditions. -/
theorem total_temp_remaining_days 
  (avg_temp : ℝ) 
  (days_in_week : ℕ) 
  (temp_first_three : ℝ) 
  (days_first_three : ℕ) 
  (temp_thur_fri : ℝ) 
  (days_thur_fri : ℕ) :
  avg_temp = 60 ∧ 
  days_in_week = 7 ∧ 
  temp_first_three = 40 ∧ 
  days_first_three = 3 ∧ 
  temp_thur_fri = 80 ∧ 
  days_thur_fri = 2 →
  avg_temp * days_in_week - (temp_first_three * days_first_three + temp_thur_fri * days_thur_fri) = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_temp_remaining_days_l813_81311


namespace NUMINAMATH_CALUDE_instant_noodle_change_l813_81366

theorem instant_noodle_change (total_change : ℕ) (total_notes : ℕ) (x : ℕ) (y : ℕ) : 
  total_change = 95 →
  total_notes = 16 →
  x + y = total_notes →
  10 * x + 5 * y = total_change →
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_instant_noodle_change_l813_81366


namespace NUMINAMATH_CALUDE_bedroom_painting_area_l813_81382

/-- The total area of walls to be painted in multiple bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Theorem: The total area of walls to be painted in 4 bedrooms is 1520 square feet -/
theorem bedroom_painting_area : 
  total_paintable_area 4 14 11 9 70 = 1520 := by
  sorry


end NUMINAMATH_CALUDE_bedroom_painting_area_l813_81382


namespace NUMINAMATH_CALUDE_isosceles_base_length_l813_81322

/-- The length of the base of an isosceles triangle, given specific conditions -/
theorem isosceles_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h1 : equilateral_perimeter = 45) 
  (h2 : isosceles_perimeter = 40) : ℝ :=
by
  -- The length of the base of the isosceles triangle is 10
  sorry

#check isosceles_base_length

end NUMINAMATH_CALUDE_isosceles_base_length_l813_81322


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l813_81300

def total_dogs : ℕ := 264
def running_dogs : ℕ := 40
def playing_dogs : ℕ := 66
def barking_dogs : ℕ := 44
def digging_dogs : ℕ := 26
def agility_dogs : ℕ := 12

theorem dogs_not_doing_anything : 
  total_dogs - (running_dogs + playing_dogs + barking_dogs + digging_dogs + agility_dogs) = 76 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l813_81300


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_l813_81334

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetricPointXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Theorem: The symmetric point of M(m,n,p) with respect to xOy plane is (m,n,-p) -/
theorem symmetric_point_xoy (m n p : ℝ) :
  let M : Point3D := { x := m, y := n, z := p }
  symmetricPointXOY M = { x := m, y := n, z := -p } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_l813_81334


namespace NUMINAMATH_CALUDE_base7_even_digits_528_l813_81354

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base7_even_digits_528 :
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end NUMINAMATH_CALUDE_base7_even_digits_528_l813_81354


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l813_81377

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define set A
def A : Set ℂ := {i, i^2, i^3, i^4}

-- Define set B
def B : Set ℂ := {1, -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1, -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l813_81377


namespace NUMINAMATH_CALUDE_sum_of_simplified_fraction_75_135_l813_81342

def simplify_fraction (n d : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd n d
  (n / g, d / g)

theorem sum_of_simplified_fraction_75_135 :
  let (n, d) := simplify_fraction 75 135
  n + d = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_simplified_fraction_75_135_l813_81342


namespace NUMINAMATH_CALUDE_perpendicular_sum_l813_81324

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, then the second component of b is -4. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 3 ∧ b.2 = -2) :
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → b.1 = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_l813_81324


namespace NUMINAMATH_CALUDE_carrot_bundle_price_is_two_dollars_l813_81374

/-- Represents the farmer's harvest and sales data -/
structure FarmerData where
  potatoes : ℕ
  carrots : ℕ
  potatoesPerBundle : ℕ
  carrotsPerBundle : ℕ
  potatoBundlePrice : ℚ
  totalRevenue : ℚ

/-- Calculates the price of each carrot bundle -/
def carrotBundlePrice (data : FarmerData) : ℚ :=
  let potatoBundles := data.potatoes / data.potatoesPerBundle
  let potatoRevenue := potatoBundles * data.potatoBundlePrice
  let carrotRevenue := data.totalRevenue - potatoRevenue
  let carrotBundles := data.carrots / data.carrotsPerBundle
  carrotRevenue / carrotBundles

/-- Theorem stating that the carrot bundle price is $2.00 -/
theorem carrot_bundle_price_is_two_dollars 
  (data : FarmerData) 
  (h1 : data.potatoes = 250)
  (h2 : data.carrots = 320)
  (h3 : data.potatoesPerBundle = 25)
  (h4 : data.carrotsPerBundle = 20)
  (h5 : data.potatoBundlePrice = 19/10)
  (h6 : data.totalRevenue = 51) :
  carrotBundlePrice data = 2 := by
  sorry

#eval carrotBundlePrice {
  potatoes := 250,
  carrots := 320,
  potatoesPerBundle := 25,
  carrotsPerBundle := 20,
  potatoBundlePrice := 19/10,
  totalRevenue := 51
}

end NUMINAMATH_CALUDE_carrot_bundle_price_is_two_dollars_l813_81374


namespace NUMINAMATH_CALUDE_mary_next_birthday_age_l813_81351

theorem mary_next_birthday_age 
  (mary_age sally_age danielle_age : ℝ)
  (h1 : mary_age = 1.25 * sally_age)
  (h2 : sally_age = 0.7 * danielle_age)
  (h3 : mary_age + sally_age + danielle_age = 36) :
  ⌊mary_age⌋ + 1 = 13 :=
by sorry

end NUMINAMATH_CALUDE_mary_next_birthday_age_l813_81351


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l813_81313

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 3 → x * y = 154 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l813_81313


namespace NUMINAMATH_CALUDE_two_fixed_points_l813_81323

/-- A function satisfying the given property -/
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + x * y + 1

/-- The main theorem -/
theorem two_fixed_points
  (f : ℝ → ℝ)
  (h1 : satisfies_property f)
  (h2 : f (-2) = -2) :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ a : ℤ, a ∈ s ↔ f a = a :=
sorry

end NUMINAMATH_CALUDE_two_fixed_points_l813_81323


namespace NUMINAMATH_CALUDE_ellipse_condition_l813_81312

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  9 * x^2 + y^2 - 18 * x - 2 * y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ (a ≠ b ∨ c ≠ 0 ∨ d ≠ 0) ∧
    ∀ x y : ℝ, curve_equation x y k ↔ a * (x - c)^2 + b * (y - d)^2 = e

/-- The main theorem -/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l813_81312
