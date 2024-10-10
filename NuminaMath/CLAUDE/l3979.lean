import Mathlib

namespace smallest_solution_quadratic_equation_l3979_397904

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 6 * (8 * x^2 + 7 * x + 11) - x * (8 * x - 45)
  ∃ x : ℝ, (f x = 0) ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ (x = -11/8) := by
  sorry

end smallest_solution_quadratic_equation_l3979_397904


namespace milk_water_ratio_after_mixing_l3979_397935

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Represents the result of mixing two mixtures -/
def mix (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- The ratio of milk to water in a mixture -/
def milkWaterRatio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_after_mixing :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  milkWaterRatio (mix m1 m2) = 5 := by
  sorry

end milk_water_ratio_after_mixing_l3979_397935


namespace triangle_segment_equality_l3979_397955

theorem triangle_segment_equality (AB AC : ℝ) (m n : ℕ+) :
  AB = 33 →
  AC = 21 →
  ∃ (BC : ℝ), BC = m →
  ∃ (D E : ℝ × ℝ),
    D.1 + D.2 = AB ∧
    E.1 + E.2 = AC ∧
    D.1 = n ∧
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = n ∧
    E.2 = n →
  n = 11 ∨ n = 21 := by
sorry


end triangle_segment_equality_l3979_397955


namespace factoring_quadratic_l3979_397942

theorem factoring_quadratic (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) := by
  sorry

end factoring_quadratic_l3979_397942


namespace number_problem_l3979_397973

theorem number_problem (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 40 → x = 160 / 7 := by
  sorry

end number_problem_l3979_397973


namespace inverse_count_mod_eleven_l3979_397919

theorem inverse_count_mod_eleven : 
  ∃ (S : Finset ℕ), 
    S.card = 10 ∧ 
    (∀ a ∈ S, a ≤ 10) ∧
    (∀ a ∈ S, ∃ b : ℕ, (a * b) % 11 = 1) ∧
    (∀ a : ℕ, a ≤ 10 → (∃ b : ℕ, (a * b) % 11 = 1) → a ∈ S) :=
by sorry

end inverse_count_mod_eleven_l3979_397919


namespace river_flow_rate_l3979_397976

/-- Given a river with specified dimensions and flow rate, calculate its velocity --/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_rate : ℝ) (velocity : ℝ) : 
  depth = 8 →
  width = 25 →
  flow_rate = 26666.666666666668 →
  velocity = flow_rate / (depth * width) →
  velocity = 133.33333333333334 := by
  sorry

#check river_flow_rate

end river_flow_rate_l3979_397976


namespace smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l3979_397921

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (450 * x) → x ≥ 25 := by
  sorry

theorem twenty_five_satisfies : 625 ∣ (450 * 25) := by
  sorry

theorem smallest_x_is_25 : ∃ x : ℕ, x > 0 ∧ 625 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 625 ∣ (450 * y)) → x ≤ y := by
  sorry

end smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l3979_397921


namespace rhombus_perimeter_l3979_397928

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (AC BD : ℝ) (h1 : AC = 8) (h2 : BD = 6) :
  let side_length := Real.sqrt ((AC / 2) ^ 2 + (BD / 2) ^ 2)
  4 * side_length = 20 := by sorry

end rhombus_perimeter_l3979_397928


namespace expression_value_l3979_397922

theorem expression_value : (2.502 + 0.064)^2 - (2.502 - 0.064)^2 / (2.502 * 0.064) = 4.002 := by
  sorry

end expression_value_l3979_397922


namespace total_blocks_adolfos_blocks_l3979_397968

/-- Given an initial number of blocks and a number of blocks added, 
    the total number of blocks is equal to the sum of the initial blocks and added blocks. -/
theorem total_blocks (initial_blocks added_blocks : ℕ) :
  initial_blocks + added_blocks = initial_blocks + added_blocks := by
  sorry

/-- Adolfo's block problem -/
theorem adolfos_blocks : 
  let initial_blocks : ℕ := 35
  let added_blocks : ℕ := 30
  initial_blocks + added_blocks = 65 := by
  sorry

end total_blocks_adolfos_blocks_l3979_397968


namespace xiaoliang_step_count_l3979_397994

/-- Represents the number of steps a person climbs to reach their floor -/
structure StepCount where
  floor : ℕ
  steps : ℕ

/-- Represents the building with information about Xiaoping and Xiaoliang -/
structure Building where
  xiaoping : StepCount
  xiaoliang : StepCount

/-- The theorem stating the number of steps Xiaoliang climbs -/
theorem xiaoliang_step_count (b : Building) 
  (h1 : b.xiaoping.floor = 5)
  (h2 : b.xiaoliang.floor = 4)
  (h3 : b.xiaoping.steps = 80) :
  b.xiaoliang.steps = 60 := by
  sorry

end xiaoliang_step_count_l3979_397994


namespace worker_productivity_ratio_l3979_397915

theorem worker_productivity_ratio :
  ∀ (x y : ℝ),
  (x > 0) →
  (y > 0) →
  (2 * (x + y) = 1) →
  (x / 3 + 3 * y = 1) →
  (y / x = 5 / 3) :=
by
  sorry

end worker_productivity_ratio_l3979_397915


namespace minimal_area_parallelepiped_l3979_397980

/-- A right parallelepiped with integer side lengths -/
structure RightParallelepiped where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- The volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℕ :=
  p.a * p.b * p.c

/-- The surface area of a right parallelepiped -/
def surfaceArea (p : RightParallelepiped) : ℕ :=
  2 * (p.a * p.b + p.b * p.c + p.c * p.a)

/-- The set of all right parallelepipeds with volume > 1000 -/
def validParallelepipeds : Set RightParallelepiped :=
  {p : RightParallelepiped | volume p > 1000}

theorem minimal_area_parallelepiped :
  ∃ (p : RightParallelepiped),
    p ∈ validParallelepipeds ∧
    p.a = 7 ∧ p.b = 12 ∧ p.c = 12 ∧
    ∀ (q : RightParallelepiped),
      q ∈ validParallelepipeds →
      surfaceArea p ≤ surfaceArea q :=
sorry

end minimal_area_parallelepiped_l3979_397980


namespace set_formation_criterion_l3979_397979

-- Define a type for objects
variable {α : Type}

-- Define a predicate for well-defined and specific objects
variable (is_well_defined : α → Prop)

-- Define a predicate for collections that can form sets
def can_form_set (collection : Set α) : Prop :=
  ∀ x ∈ collection, is_well_defined x

-- Theorem statement
theorem set_formation_criterion (collection : Set α) :
  can_form_set is_well_defined collection ↔ ∀ x ∈ collection, is_well_defined x :=
sorry

end set_formation_criterion_l3979_397979


namespace quadratic_solution_range_l3979_397910

theorem quadratic_solution_range (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  (f 3.24 = -0.02) → (f 3.25 = 0.01) → (f 3.26 = 0.03) →
  ∃ x, f x = 0 ∧ 3.24 < x ∧ x < 3.25 :=
by sorry

end quadratic_solution_range_l3979_397910


namespace parallelepiped_volume_theorem_l3979_397960

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  -- Side lengths of the base parallelogram
  side1 : ℝ
  side2 : ℝ
  -- Acute angle of the base parallelogram in radians
  angle : ℝ
  -- Length of the longest diagonal
  diagonal : ℝ

/-- Calculate the volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  -- This function will be defined in the proof
  sorry

/-- The main theorem to prove -/
theorem parallelepiped_volume_theorem (p : RightParallelepiped) 
  (h1 : p.side1 = 1)
  (h2 : p.side2 = 4)
  (h3 : p.angle = π / 3)  -- 60 degrees in radians
  (h4 : p.diagonal = 5) :
  volume p = 4 * Real.sqrt 3 := by
  sorry

end parallelepiped_volume_theorem_l3979_397960


namespace solution_implies_a_zero_l3979_397944

/-- Given a system of linear equations and an additional equation with parameter a,
    prove that a must be zero if the solution of the system satisfies the additional equation. -/
theorem solution_implies_a_zero (x y a : ℝ) : 
  2 * x + 7 * y = 11 →
  5 * x - 4 * y = 6 →
  3 * x - 6 * y + 2 * a = 0 →
  a = 0 := by
  sorry

end solution_implies_a_zero_l3979_397944


namespace difference_average_median_l3979_397956

theorem difference_average_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((1 + (a + 1) + (2*a + b) + (a + b + 1)) / 4) - ((a + 1 + (a + b + 1)) / 2)| = 1/4 := by
sorry

end difference_average_median_l3979_397956


namespace complex_product_one_plus_i_one_minus_i_l3979_397986

theorem complex_product_one_plus_i_one_minus_i : 
  (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end complex_product_one_plus_i_one_minus_i_l3979_397986


namespace custom_mul_five_three_l3979_397950

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 - a*b + b^2

/-- Theorem stating that 5*3 = 19 under the custom multiplication -/
theorem custom_mul_five_three : custom_mul 5 3 = 19 := by
  sorry

end custom_mul_five_three_l3979_397950


namespace leos_garden_tulips_leos_garden_tulips_proof_l3979_397949

/-- Calculates the number of tulips in Leo's garden after additions -/
theorem leos_garden_tulips (initial_carnations : ℕ) (added_carnations : ℕ) 
  (tulip_ratio : ℕ) (carnation_ratio : ℕ) : ℕ :=
  let total_carnations := initial_carnations + added_carnations
  let tulips := (total_carnations / carnation_ratio) * tulip_ratio
  tulips

/-- Proves that Leo will have 36 tulips after the additions -/
theorem leos_garden_tulips_proof :
  leos_garden_tulips 49 35 3 7 = 36 := by
  sorry

end leos_garden_tulips_leos_garden_tulips_proof_l3979_397949


namespace sin_triple_angle_l3979_397993

theorem sin_triple_angle (θ : ℝ) :
  Real.sin (3 * θ) = 4 * Real.sin θ * Real.sin (π / 3 + θ) * Real.sin (2 * π / 3 + θ) := by
  sorry

end sin_triple_angle_l3979_397993


namespace bones_fraction_in_beef_l3979_397908

/-- The price of beef with bones in rubles per kilogram -/
def price_beef_with_bones : ℝ := 78

/-- The price of boneless beef in rubles per kilogram -/
def price_boneless_beef : ℝ := 90

/-- The price of bones in rubles per kilogram -/
def price_bones : ℝ := 15

/-- The fraction of bones in a kilogram of beef -/
def fraction_bones : ℝ := 0.16

theorem bones_fraction_in_beef :
  price_bones * fraction_bones + price_boneless_beef * (1 - fraction_bones) = price_beef_with_bones :=
sorry

end bones_fraction_in_beef_l3979_397908


namespace marked_box_second_row_l3979_397901

/-- Represents the number of cakes in each box of the pyramid -/
structure CakePyramid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ

/-- The condition that each box in a higher row contains the sum of cakes in the two adjacent boxes below -/
def valid_pyramid (p : CakePyramid) : Prop :=
  p.e = p.a + p.b ∧
  p.f = p.b + p.c ∧
  p.g = p.c + p.d ∧
  p.h = p.e + p.f

/-- The condition that three boxes contain 3, 5, and 6 cakes -/
def marked_boxes (p : CakePyramid) : Prop :=
  (p.a = 3 ∨ p.a = 5 ∨ p.a = 6) ∧
  (p.d = 3 ∨ p.d = 5 ∨ p.d = 6) ∧
  (p.f = 3 ∨ p.f = 5 ∨ p.f = 6) ∧
  p.a ≠ p.d ∧ p.a ≠ p.f ∧ p.d ≠ p.f

/-- The total number of cakes in the pyramid -/
def total_cakes (p : CakePyramid) : ℕ :=
  p.a + p.b + p.c + p.d + p.e + p.f + p.g + p.h

/-- The theorem stating that the marked box in the second row from the bottom contains 3 cakes -/
theorem marked_box_second_row (p : CakePyramid) :
  valid_pyramid p → marked_boxes p → (∀ q : CakePyramid, valid_pyramid q → marked_boxes q → total_cakes q ≥ total_cakes p) → p.f = 3 := by
  sorry


end marked_box_second_row_l3979_397901


namespace divisibility_of_sum_of_powers_l3979_397943

theorem divisibility_of_sum_of_powers (n : ℕ) : 13 ∣ (3^1974 + 2^1974) := by
  sorry

end divisibility_of_sum_of_powers_l3979_397943


namespace min_prime_angle_sum_90_l3979_397969

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_prime_angle_sum_90 :
  ∀ x y : ℕ,
    isPrime x →
    isPrime y →
    x + y = 90 →
    y ≥ 7 :=
by sorry

end min_prime_angle_sum_90_l3979_397969


namespace percentage_increase_decrease_l3979_397997

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hq_bound : q < 100) :
  M * (1 + p / 100) * (1 - q / 100) = 1.1 * M ↔ 
  p = (10 + 100 * q) / (100 - q) :=
by sorry

end percentage_increase_decrease_l3979_397997


namespace eighteen_twelve_over_fiftyfour_six_l3979_397957

theorem eighteen_twelve_over_fiftyfour_six : (18 ^ 12) / (54 ^ 6) = 46656 := by
  sorry

end eighteen_twelve_over_fiftyfour_six_l3979_397957


namespace geometric_sum_property_l3979_397995

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_property (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q = 2 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sum_property_l3979_397995


namespace most_efficient_numbering_system_l3979_397983

/-- Represents a numbering system for a population --/
inductive NumberingSystem
  | OneToN
  | ZeroToNMinusOne
  | TwoDigitZeroToNMinusOne
  | ThreeDigitZeroToNMinusOne

/-- Determines if a numbering system is most efficient for random number table sampling --/
def is_most_efficient (n : NumberingSystem) (population_size : ℕ) (sample_size : ℕ) : Prop :=
  n = NumberingSystem.ThreeDigitZeroToNMinusOne ∧ 
  population_size = 106 ∧ 
  sample_size = 10

/-- Theorem stating the most efficient numbering system for the given conditions --/
theorem most_efficient_numbering_system :
  ∃ (n : NumberingSystem), is_most_efficient n 106 10 :=
sorry

end most_efficient_numbering_system_l3979_397983


namespace complex_expression_simplification_l3979_397934

theorem complex_expression_simplification :
  (8 - 5*Complex.I) + 3*(2 - 4*Complex.I) = 14 - 17*Complex.I :=
by sorry

end complex_expression_simplification_l3979_397934


namespace locus_is_circle_l3979_397923

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the locus of points
def Locus (c : Circle) (B C : PointOnCircle c) : Set (ℝ × ℝ) :=
  { M | ∃ (A : PointOnCircle c),
    let K := ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2)
    M ∈ { P | (P.1 - A.point.1) * (C.point.1 - A.point.1) + (P.2 - A.point.2) * (C.point.2 - A.point.2) = 0 } ∧
    (K.1 - M.1) * (C.point.1 - A.point.1) + (K.2 - M.2) * (C.point.2 - A.point.2) = 0 }

-- Theorem statement
theorem locus_is_circle (c : Circle) (B C : PointOnCircle c) :
  ∃ (c' : Circle), Locus c B C = { P | (P.1 - c'.center.1)^2 + (P.2 - c'.center.2)^2 = c'.radius^2 } ∧
  B.point ∈ Locus c B C ∧ C.point ∈ Locus c B C :=
sorry

end locus_is_circle_l3979_397923


namespace slope_product_negative_half_exists_line_equal_distances_l3979_397929

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and Q
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (-2, 0)

-- Theorem for part (I)
theorem slope_product_negative_half (x y : ℝ) :
  C x y → x ≠ 0 → (y - A.2) / (x - A.1) * (y - B.2) / (x - B.1) = -1/2 := by sorry

-- Theorem for part (II)
theorem exists_line_equal_distances :
  ∃ (M N : ℝ × ℝ), 
    C M.1 M.2 ∧ C N.1 N.2 ∧ 
    M ≠ N ∧
    M.2 = 0 ∧ N.2 = 0 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (N.1 - B.1)^2 + (N.2 - B.2)^2 := by sorry

end slope_product_negative_half_exists_line_equal_distances_l3979_397929


namespace john_scores_42_points_l3979_397964

/-- The number of points John scores in a game, given the scoring pattern and game duration -/
def johnTotalPoints (pointsPer4Min : ℕ) (intervalsPer12Min : ℕ) (numPeriods : ℕ) : ℕ :=
  pointsPer4Min * intervalsPer12Min * numPeriods

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points :
  let pointsPer4Min := 2 * 2 + 1 * 3  -- 2 shots worth 2 points and 1 shot worth 3 points
  let intervalsPer12Min := 12 / 4     -- Each period is 12 minutes, divided into 4-minute intervals
  let numPeriods := 2                 -- He plays for 2 periods
  johnTotalPoints pointsPer4Min intervalsPer12Min numPeriods = 42 := by
  sorry

#eval johnTotalPoints (2 * 2 + 1 * 3) (12 / 4) 2

end john_scores_42_points_l3979_397964


namespace expression_value_l3979_397975

theorem expression_value : (3^4 * 5^2 * 7^3 * 11) / (7 * 11^2) = 9025 := by
  sorry

end expression_value_l3979_397975


namespace simplify_power_expression_l3979_397936

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by sorry

end simplify_power_expression_l3979_397936


namespace representation_bound_l3979_397988

def f (n : ℕ) : ℕ := sorry

theorem representation_bound (n : ℕ) (h : n ≥ 3) :
  (2 : ℝ) ^ (n^2/4) < (f (2^n) : ℝ) ∧ (f (2^n) : ℝ) < (2 : ℝ) ^ (n^2/2) := by
  sorry

end representation_bound_l3979_397988


namespace goose_egg_count_l3979_397951

/-- The number of goose eggs laid at a pond -/
def num_eggs : ℕ := 650

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 2/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 130

theorem goose_egg_count :
  (↑num_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction) : ℚ) = survived_year :=
sorry

end goose_egg_count_l3979_397951


namespace f_at_negative_one_l3979_397952

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 16

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 5*x^3 + b*x^2 + 150*x + c

theorem f_at_negative_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c (-1) = -1347 := by sorry

end f_at_negative_one_l3979_397952


namespace polynomial_value_theorem_l3979_397940

def g (x : ℝ) : ℝ := -x^3 + x^2 - x + 1

theorem polynomial_value_theorem :
  g 3 = 1 ∧ 12 * (-1) - 6 * 1 + 3 * (-1) - 1 = -22 := by
  sorry

end polynomial_value_theorem_l3979_397940


namespace factorization_equality_simplification_equality_system_of_inequalities_l3979_397998

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  x^2 * (x - 3) + y^2 * (3 - x) = (x - 3) * (x + y) * (x - y) := by sorry

-- Problem 2
theorem simplification_equality (x : ℝ) (h1 : x ≠ 3/5) (h2 : x ≠ -3/5) :
  (2*x / (5*x - 3)) / (3 / (25*x^2 - 9)) * (x / (5*x + 3)) = 2/3 * x^2 := by sorry

-- Problem 3
theorem system_of_inequalities (x : ℝ) :
  ((x - 3) / 2 + 3 ≥ x + 1) ∧ (1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end factorization_equality_simplification_equality_system_of_inequalities_l3979_397998


namespace number_puzzle_l3979_397927

theorem number_puzzle : ∃ x : ℝ, ((2 * x - 37 + 25) / 8 = 5) ∧ x = 26 := by
  sorry

end number_puzzle_l3979_397927


namespace wood_square_weight_l3979_397967

/-- Represents a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Calculates the area of a square -/
def square_area (s : ℝ) : ℝ := s * s

/-- Theorem: Given two square pieces of wood with uniform density, 
    where the first piece has a side length of 3 inches and weighs 15 ounces, 
    and the second piece has a side length of 6 inches, 
    the weight of the second piece is 60 ounces. -/
theorem wood_square_weight 
  (first : WoodSquare) 
  (second : WoodSquare) 
  (h1 : first.side_length = 3) 
  (h2 : first.weight = 15) 
  (h3 : second.side_length = 6) : 
  second.weight = 60 := by
  sorry


end wood_square_weight_l3979_397967


namespace smallest_repeating_block_8_13_l3979_397917

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def repeating_block (l : List ℕ) : List ℕ := sorry

theorem smallest_repeating_block_8_13 :
  (repeating_block (decimal_expansion 8 13)).length = 6 := by sorry

end smallest_repeating_block_8_13_l3979_397917


namespace exist_four_numbers_squares_l3979_397996

theorem exist_four_numbers_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end exist_four_numbers_squares_l3979_397996


namespace sin_240_degrees_l3979_397965

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l3979_397965


namespace min_sum_of_squares_l3979_397961

theorem min_sum_of_squares (x y z k : ℝ) 
  (h1 : (x + 8) * (y - 8) = 0) 
  (h2 : x + y + z = k) : 
  x^2 + y^2 + z^2 ≥ 64 + k^2/2 - 4*k + 32 :=
sorry

end min_sum_of_squares_l3979_397961


namespace rectangle_area_l3979_397933

theorem rectangle_area (length width : ℝ) (h1 : length = 5) (h2 : width = 17/20) :
  length * width = 4.25 := by
  sorry

end rectangle_area_l3979_397933


namespace quadratic_inequality_solution_implies_b_zero_l3979_397925

theorem quadratic_inequality_solution_implies_b_zero
  (a b m : ℝ)
  (h : ∀ x, ax^2 - a*x + b < 0 ↔ m < x ∧ x < m + 1) :
  b = 0 :=
sorry

end quadratic_inequality_solution_implies_b_zero_l3979_397925


namespace inscribed_circle_radius_l3979_397958

/-- An isosceles trapezoid with specific dimensions and inscribed circles --/
structure IsoscelesTrapezoidWithCircles where
  -- The length of side AB
  ab : ℝ
  -- The length of sides BC and DA
  bc : ℝ
  -- The length of side CD
  cd : ℝ
  -- The radius of circles centered at A and B
  r_ab : ℝ
  -- The radius of circles centered at C and D
  r_cd : ℝ

/-- The theorem stating the radius of the inscribed circle tangent to all four circles --/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles)
  (h_ab : t.ab = 10)
  (h_bc : t.bc = 7)
  (h_cd : t.cd = 6)
  (h_r_ab : t.r_ab = 4)
  (h_r_cd : t.r_cd = 3) :
  ∃ r : ℝ, r = (-81 + 57 * Real.sqrt 5) / 23 ∧
    (∃ O : ℝ × ℝ, ∃ A B C D : ℝ × ℝ,
      -- O is the center of the inscribed circle
      -- A, B, C, D are the centers of the given circles
      -- The inscribed circle is tangent to all four given circles
      True) := by
  sorry

end inscribed_circle_radius_l3979_397958


namespace myopia_functional_relationship_l3979_397932

/-- The functional relationship between the degree of myopia glasses and focal length of lenses -/
def myopia_relationship (y x : ℝ) : Prop :=
  y = 100 / x

/-- y and x are inversely proportional -/
def inverse_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / x

theorem myopia_functional_relationship :
  ∀ y x : ℝ, 
  inverse_proportional y x → 
  (y = 400 ∧ x = 0.25) → 
  myopia_relationship y x :=
sorry

end myopia_functional_relationship_l3979_397932


namespace final_center_coordinates_l3979_397963

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (6, -5)

-- Define the reflection about y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the reflection over y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the composition of the two reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_y_eq_x p)

-- Theorem statement
theorem final_center_coordinates :
  double_reflection initial_center = (5, 6) := by sorry

end final_center_coordinates_l3979_397963


namespace correct_algebraic_expression_l3979_397909

-- Define the set of possible expressions
inductive AlgebraicExpression
  | MixedNumber : AlgebraicExpression  -- represents 1½a
  | ExplicitMultiplication : AlgebraicExpression  -- represents a × b
  | DivisionSign : AlgebraicExpression  -- represents a ÷ b
  | ImplicitMultiplication : AlgebraicExpression  -- represents 2a

-- Define the property of being correctly written
def isCorrectlyWritten (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.ImplicitMultiplication => True
  | _ => False

-- Theorem statement
theorem correct_algebraic_expression :
  isCorrectlyWritten AlgebraicExpression.ImplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.MixedNumber ∧
  ¬isCorrectlyWritten AlgebraicExpression.ExplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.DivisionSign :=
sorry

end correct_algebraic_expression_l3979_397909


namespace clock_angle_4_to_545_l3979_397902

-- Define the clock structure
structure Clock :=
  (numbers : Nat)
  (angle_between_numbers : Real)
  (divisions_between_numbers : Nat)
  (angle_between_divisions : Real)

-- Define the function to calculate the angle turned by the hour hand
def angle_turned (c : Clock) (start_hour : Nat) (end_hour : Nat) (end_minute : Nat) : Real :=
  sorry

-- Theorem statement
theorem clock_angle_4_to_545 (c : Clock) :
  c.numbers = 12 →
  c.angle_between_numbers = 30 →
  c.divisions_between_numbers = 5 →
  c.angle_between_divisions = 6 →
  angle_turned c 4 5 45 = 52.5 :=
sorry

end clock_angle_4_to_545_l3979_397902


namespace complementary_angles_of_same_angle_are_equal_l3979_397970

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- An angle is the complement of another if together they form 90 degrees -/
def IsComplement (α β : ℝ) : Prop := Complementary α β

theorem complementary_angles_of_same_angle_are_equal 
  (θ α β : ℝ) (h1 : IsComplement θ α) (h2 : IsComplement θ β) : α = β := by
  sorry

#check complementary_angles_of_same_angle_are_equal

end complementary_angles_of_same_angle_are_equal_l3979_397970


namespace polynomial_root_implies_coefficients_l3979_397984

theorem polynomial_root_implies_coefficients : ∀ (c d : ℝ),
  (∃ (x : ℂ), x^3 + c*x^2 + 2*x + d = 0 ∧ x = Complex.mk 2 (-3)) →
  c = 5/4 ∧ d = -143/4 := by
sorry

end polynomial_root_implies_coefficients_l3979_397984


namespace smallest_n_for_integer_root_l3979_397991

theorem smallest_n_for_integer_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬ ∃ (k : ℕ), k^2 = 2019 - m) ∧
  ∃ (k : ℕ), k^2 = 2019 - n :=
by sorry

end smallest_n_for_integer_root_l3979_397991


namespace line_intersects_circle_l3979_397913

/-- The line l with equation x - ky - 1 = 0 intersects the circle C with equation x^2 + y^2 = 2 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), (x - k*y - 1 = 0) ∧ (x^2 + y^2 = 2) := by
  sorry

end line_intersects_circle_l3979_397913


namespace circle_area_from_diameter_endpoints_l3979_397945

/-- The area of a circle with diameter endpoints C(-2,3) and D(6,9) is 25π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (-2, 3)
  let d : ℝ × ℝ := (6, 9)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius := Real.sqrt diameter_squared / 2
  let area := π * radius^2
  area = 25 * π :=
by
  sorry

end circle_area_from_diameter_endpoints_l3979_397945


namespace scout_weekend_earnings_l3979_397930

/-- Scout's weekend earnings calculation --/
theorem scout_weekend_earnings 
  (base_pay : ℝ) 
  (tip_per_customer : ℝ)
  (saturday_hours : ℝ) 
  (saturday_customers : ℕ)
  (sunday_hours : ℝ) 
  (sunday_customers : ℕ)
  (h1 : base_pay = 10)
  (h2 : tip_per_customer = 5)
  (h3 : saturday_hours = 4)
  (h4 : saturday_customers = 5)
  (h5 : sunday_hours = 5)
  (h6 : sunday_customers = 8) :
  base_pay * (saturday_hours + sunday_hours) + 
  tip_per_customer * (saturday_customers + sunday_customers) = 155 := by
sorry

end scout_weekend_earnings_l3979_397930


namespace exists_unreachable_grid_l3979_397999

/-- Represents an 8x8 grid of natural numbers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Represents a subgrid selection, either 3x3 or 4x4 -/
inductive Subgrid
| three : Fin 6 → Fin 6 → Subgrid
| four : Fin 5 → Fin 5 → Subgrid

/-- Applies the increment operation to a subgrid -/
def applyOperation (g : Grid) (s : Subgrid) : Grid :=
  sorry

/-- Checks if all numbers in the grid are divisible by 10 -/
def allDivisibleBy10 (g : Grid) : Prop :=
  ∀ i j, (g i j) % 10 = 0

/-- The main theorem statement -/
theorem exists_unreachable_grid :
  ∃ (initial : Grid), ¬∃ (ops : List Subgrid), allDivisibleBy10 (ops.foldl applyOperation initial) :=
sorry

end exists_unreachable_grid_l3979_397999


namespace circle_equation_through_pole_l3979_397911

/-- A circle in a polar coordinate system --/
structure PolarCircle where
  center : (ℝ × ℝ)
  passes_through_pole : Bool

/-- The polar coordinate equation of a circle --/
def polar_equation (c : PolarCircle) : ℝ → ℝ := sorry

theorem circle_equation_through_pole (c : PolarCircle) 
  (h1 : c.center = (Real.sqrt 2, Real.pi))
  (h2 : c.passes_through_pole = true) :
  polar_equation c = λ θ => -2 * Real.sqrt 2 * Real.cos θ := by
  sorry

end circle_equation_through_pole_l3979_397911


namespace random_events_identification_l3979_397938

-- Define the type for events
inductive Event
| CoinToss : Event
| ChargeAttraction : Event
| WaterFreeze : Event
| DieRoll : Event

-- Define a predicate for random events
def isRandomEvent : Event → Prop
| Event.CoinToss => true
| Event.ChargeAttraction => false
| Event.WaterFreeze => false
| Event.DieRoll => true

-- Theorem statement
theorem random_events_identification :
  (isRandomEvent Event.CoinToss ∧ isRandomEvent Event.DieRoll) ∧
  (¬isRandomEvent Event.ChargeAttraction ∧ ¬isRandomEvent Event.WaterFreeze) :=
by sorry

end random_events_identification_l3979_397938


namespace carol_wins_probability_l3979_397946

/-- The probability of getting a six on a single die toss -/
def prob_six : ℚ := 1 / 6

/-- The probability of not getting a six on a single die toss -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Carol in the sequence -/
def players_before_carol : ℕ := 2

/-- The total number of players in the sequence -/
def total_players : ℕ := 4

/-- The probability that Carol wins on her first turn in any cycle -/
def prob_carol_wins_first_turn : ℚ := prob_not_six ^ players_before_carol * prob_six

/-- The probability that no one wins in a full cycle -/
def prob_no_win_cycle : ℚ := prob_not_six ^ total_players

/-- Theorem: The probability that Carol is the first to toss a six is 25/91 -/
theorem carol_wins_probability :
  prob_carol_wins_first_turn / (1 - prob_no_win_cycle) = 25 / 91 := by
  sorry

end carol_wins_probability_l3979_397946


namespace tangent_line_and_minimum_value_l3979_397962

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_and_minimum_value (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 - (a + 2) * x + Real.log x) →
  (a = 1 → ∀ y, y = -2 ↔ y = f 1 1 ∧ (∀ h, h ≠ 0 → (f 1 (1 + h) - f 1 1) / h = 0)) ∧
  (a > 0 → (∀ x, x ∈ Set.Icc 1 (Real.exp 1) → f a x ≥ -2) ∧ 
           (∃ x, x ∈ Set.Icc 1 (Real.exp 1) ∧ f a x = -2) →
           a ≥ 1) :=
sorry

end tangent_line_and_minimum_value_l3979_397962


namespace quadratic_roots_irrational_l3979_397990

theorem quadratic_roots_irrational (k : ℝ) (h1 : k^2 = 16/3) (h2 : ∀ x, x^2 - 5*k*x + 3*k^2 = 0 → ∃ y, x^2 - 5*k*x + 3*k^2 = 0 ∧ x * y = 16) :
  ∃ x y : ℝ, x^2 - 5*k*x + 3*k^2 = 0 ∧ y^2 - 5*k*y + 3*k^2 = 0 ∧ x * y = 16 ∧ (¬ ∃ m n : ℤ, x = m / n ∨ y = m / n) :=
by
  sorry

end quadratic_roots_irrational_l3979_397990


namespace area_of_similar_rectangle_l3979_397954

-- Define the properties of rectangle R1
def side_R1 : ℝ := 3
def area_R1 : ℝ := 18

-- Define the diagonal of rectangle R2
def diagonal_R2 : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle (side_R1 area_R1 diagonal_R2 : ℝ) 
  (h1 : side_R1 > 0)
  (h2 : area_R1 > 0)
  (h3 : diagonal_R2 > 0) :
  let other_side_R1 := area_R1 / side_R1
  let ratio := other_side_R1 / side_R1
  let side_R2 := (diagonal_R2^2 / (1 + ratio^2))^(1/2)
  side_R2 * (ratio * side_R2) = 160 := by
sorry

end area_of_similar_rectangle_l3979_397954


namespace min_time_less_than_3_9_l3979_397992

/-- The walking speed of each person in km/h -/
def walking_speed : ℝ := 6

/-- The speed of the motorcycle in km/h -/
def motorcycle_speed : ℝ := 90

/-- The total distance to be covered in km -/
def total_distance : ℝ := 135

/-- The maximum number of people the motorcycle can carry -/
def max_motorcycle_capacity : ℕ := 2

/-- The number of people travelling -/
def num_people : ℕ := 3

/-- The minimum time required for all people to reach the destination -/
noncomputable def min_time : ℝ := 
  (23 * total_distance) / (9 * motorcycle_speed)

theorem min_time_less_than_3_9 : min_time < 3.9 := by
  sorry

end min_time_less_than_3_9_l3979_397992


namespace sixteen_greater_than_thirtytwo_l3979_397972

/-- Represents a domino placement on a board -/
structure DominoPlacement (n : ℕ) where
  placements : Fin n → Fin 8 × Fin 8 × Bool
  no_overlap : ∀ i j, i ≠ j → placements i ≠ placements j

/-- The number of ways to place n dominoes on an 8x8 board -/
def num_placements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of 16-domino placements is greater than 32-domino placements -/
theorem sixteen_greater_than_thirtytwo :
  num_placements 16 > num_placements 32 := by sorry

end sixteen_greater_than_thirtytwo_l3979_397972


namespace garden_volume_l3979_397920

/-- Calculates the volume of a rectangular prism -/
def rectangularPrismVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a rectangular prism garden with dimensions 12 m, 5 m, and 3 m is 180 cubic meters -/
theorem garden_volume :
  rectangularPrismVolume 12 5 3 = 180 := by
  sorry

end garden_volume_l3979_397920


namespace range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l3979_397948

-- Part 1
theorem range_sin_plus_cos :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x) = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
sorry

-- Part 2
theorem range_sin_plus_cos_minus_sin_2x :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x - Real.sin (2 * x)) = Set.Icc (-1 - Real.sqrt 2) (5/4) := by
sorry

end range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l3979_397948


namespace video_game_theorem_l3979_397914

def video_game_problem (x : ℝ) (n : ℕ) (y : ℝ) : Prop :=
  x > 0 ∧ n > 0 ∧ y > 0 ∧
  (1/4 : ℝ) * x = (1/2 : ℝ) * n * y ∧
  (1/3 : ℝ) * x = x - ((1/2 : ℝ) * x + (1/6 : ℝ) * x)

theorem video_game_theorem (x : ℝ) (n : ℕ) (y : ℝ) 
  (h : video_game_problem x n y) : True :=
by
  sorry

end video_game_theorem_l3979_397914


namespace test_failure_rate_l3979_397953

/-- The percentage of students who failed a test, given the number of boys and girls
    and their respective pass rates. -/
def percentageFailed (numBoys numGirls : ℕ) (boyPassRate girlPassRate : ℚ) : ℚ :=
  let totalStudents := numBoys + numGirls
  let failedStudents := numBoys * (1 - boyPassRate) + numGirls * (1 - girlPassRate)
  failedStudents / totalStudents

/-- Theorem stating that given 50 boys and 100 girls, with 50% of boys passing
    and 40% of girls passing, the percentage of total students who failed is 56.67%. -/
theorem test_failure_rate : 
  percentageFailed 50 100 (1/2) (2/5) = 8500/15000 := by
  sorry

end test_failure_rate_l3979_397953


namespace f_satisfies_conditions_l3979_397974

-- Define the function
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_satisfies_conditions :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through first quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through second quadrant
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂)  -- Increasing in first quadrant
  := by sorry

end f_satisfies_conditions_l3979_397974


namespace parabola_chord_dot_product_l3979_397924

/-- The parabola y^2 = 4x with focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A chord passing through the focus -/
def Chord (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ ∃ t : ℝ, (1 - t) • A + t • B = Focus

theorem parabola_chord_dot_product (A B : ℝ × ℝ) (h : Chord A B) :
    (A.1 * B.1 + A.2 * B.2 : ℝ) = -3 := by
  sorry

end parabola_chord_dot_product_l3979_397924


namespace circle_tangency_l3979_397931

structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def touches_internally (c1 c2 : Circle) (p : Point) : Prop := sorry

def center_on_circle (c1 c2 : Circle) : Prop := sorry

def common_chord_intersects (c1 c2 c3 : Circle) (a b : Point) : Prop := sorry

def line_intersects_circle (p1 p2 : Point) (c : Circle) (q : Point) : Prop := sorry

def is_tangent (c : Circle) (p1 p2 : Point) : Prop := sorry

theorem circle_tangency 
  (Ω Ω₁ Ω₂ : Circle) 
  (M N A B C D : Point) :
  touches_internally Ω₁ Ω M →
  touches_internally Ω₂ Ω N →
  center_on_circle Ω₂ Ω₁ →
  common_chord_intersects Ω₁ Ω₂ Ω A B →
  line_intersects_circle M A Ω₁ C →
  line_intersects_circle M B Ω₁ D →
  is_tangent Ω₂ C D := by
    sorry

end circle_tangency_l3979_397931


namespace sqrt_sum_fractions_equals_sqrt_181_over_12_l3979_397939

theorem sqrt_sum_fractions_equals_sqrt_181_over_12 :
  Real.sqrt (9/16 + 25/36) = Real.sqrt 181 / 12 := by
  sorry

end sqrt_sum_fractions_equals_sqrt_181_over_12_l3979_397939


namespace equation_proof_l3979_397971

theorem equation_proof : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end equation_proof_l3979_397971


namespace village_distance_l3979_397903

def round_trip_time : ℝ := 4
def uphill_speed : ℝ := 15
def downhill_speed : ℝ := 30

theorem village_distance (d : ℝ) (h : d > 0) :
  d / uphill_speed + d / downhill_speed = round_trip_time →
  d = 40 := by
sorry

end village_distance_l3979_397903


namespace square_difference_l3979_397905

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) :
  (x - y)^2 = 4 := by
  sorry

end square_difference_l3979_397905


namespace heath_carrot_planting_l3979_397916

/-- Calculates the number of carrots planted per hour given the number of rows, plants per row, and total planting time. -/
def carrots_per_hour (rows : ℕ) (plants_per_row : ℕ) (total_hours : ℕ) : ℕ :=
  (rows * plants_per_row) / total_hours

/-- Proves that given 400 rows of carrots, 300 plants per row, and 20 hours of planting time, the number of carrots planted per hour is 6,000. -/
theorem heath_carrot_planting :
  carrots_per_hour 400 300 20 = 6000 := by
  sorry

end heath_carrot_planting_l3979_397916


namespace modulus_of_complex_l3979_397959

theorem modulus_of_complex (i : ℂ) : i * i = -1 → Complex.abs (2 * i - 5 / (2 - i)) = Real.sqrt 5 := by
  sorry

end modulus_of_complex_l3979_397959


namespace shopping_trip_remainder_l3979_397906

/-- Calculates the remaining amount after a shopping trip --/
theorem shopping_trip_remainder
  (initial_amount : ℝ)
  (peach_price peach_quantity : ℝ)
  (cherry_price cherry_quantity : ℝ)
  (baguette_price baguette_quantity : ℝ)
  (discount_threshold discount_rate : ℝ)
  (tax_rate : ℝ)
  (h1 : initial_amount = 20)
  (h2 : peach_price = 2)
  (h3 : peach_quantity = 3)
  (h4 : cherry_price = 3.5)
  (h5 : cherry_quantity = 2)
  (h6 : baguette_price = 1.25)
  (h7 : baguette_quantity = 4)
  (h8 : discount_threshold = 10)
  (h9 : discount_rate = 0.1)
  (h10 : tax_rate = 0.05) :
  let subtotal := peach_price * peach_quantity + cherry_price * cherry_quantity + baguette_price * baguette_quantity
  let discounted_total := if subtotal > discount_threshold then subtotal * (1 - discount_rate) else subtotal
  let final_total := discounted_total * (1 + tax_rate)
  let remainder := initial_amount - final_total
  remainder = 2.99 := by sorry

end shopping_trip_remainder_l3979_397906


namespace square_sum_geq_product_sum_l3979_397947

theorem square_sum_geq_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end square_sum_geq_product_sum_l3979_397947


namespace ellipse_parabola_intersection_l3979_397989

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y p : ℝ) : Prop := x^2 = 2 * p * y

/-- The directrix l of C₂ -/
def l (y : ℝ) : Prop := y = -2

/-- Intersection point of l and C₁ -/
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 2 ∧ y = -2

/-- Common focus condition -/
def common_focus (a b p : ℝ) : Prop := sorry

theorem ellipse_parabola_intersection (a b p : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : p > 0)
  (h4 : common_focus a b p)
  (h5 : ∃ x y, C₁ x y a b ∧ l y ∧ intersection_point x y) :
  (∀ x y, C₁ x y a b ↔ y^2 / 8 + x^2 / 4 = 1) ∧
  (∀ x y, C₂ x y p ↔ x^2 = 8 * y) ∧
  (∃ min max : ℝ, min = -8 ∧ max = 2 ∧
    ∀ t : ℝ, ∃ x₃ y₃ x₄ y₄ : ℝ,
      C₁ x₃ y₃ a b ∧ C₁ x₄ y₄ a b ∧
      min < x₃ * x₄ + y₃ * y₄ ∧ x₃ * x₄ + y₃ * y₄ ≤ max) :=
sorry

end ellipse_parabola_intersection_l3979_397989


namespace smallest_solution_congruence_l3979_397907

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y := by
  sorry

end smallest_solution_congruence_l3979_397907


namespace stating_min_pieces_for_equal_division_l3979_397985

/-- Represents the number of pieces a pie is cut into -/
def NumPieces : ℕ := 11

/-- Represents the first group size -/
def GroupSize1 : ℕ := 5

/-- Represents the second group size -/
def GroupSize2 : ℕ := 7

/-- 
Theorem stating that NumPieces is the minimum number of pieces 
that allows equal division among GroupSize1 or GroupSize2 people 
-/
theorem min_pieces_for_equal_division :
  (∃ (k : ℕ), k * GroupSize1 = NumPieces) ∧ 
  (∃ (k : ℕ), k * GroupSize2 = NumPieces) ∧
  (∀ (n : ℕ), n < NumPieces → 
    (¬∃ (k : ℕ), k * GroupSize1 = n) ∨ 
    (¬∃ (k : ℕ), k * GroupSize2 = n)) :=
sorry

end stating_min_pieces_for_equal_division_l3979_397985


namespace trig_identity_l3979_397966

theorem trig_identity : 
  Real.sin (46 * π / 180) * Real.cos (16 * π / 180) - 
  Real.cos (314 * π / 180) * Real.sin (16 * π / 180) = 1/2 := by
  sorry

end trig_identity_l3979_397966


namespace polynomial_coefficient_sum_l3979_397978

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℚ, 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
  sorry

end polynomial_coefficient_sum_l3979_397978


namespace slope_zero_sufficient_not_necessary_l3979_397987

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a line passing through (-1, 1)
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 - 1 = m * (p.1 + 1)}

-- Define tangency
def IsTangent (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∩ Circle ∧ ∀ q : ℝ × ℝ, q ∈ l ∩ Circle → q = p

-- Theorem statement
theorem slope_zero_sufficient_not_necessary :
  (∃ l : Set (ℝ × ℝ), l = Line 0 ∧ IsTangent l) ∧
  (∃ l : Set (ℝ × ℝ), IsTangent l ∧ l ≠ Line 0) :=
sorry

end slope_zero_sufficient_not_necessary_l3979_397987


namespace smallest_concatenated_multiple_of_2016_l3979_397926

def concatenate_twice (n : ℕ) : ℕ :=
  n * 1000 + n

theorem smallest_concatenated_multiple_of_2016 :
  ∀ A : ℕ, A > 0 →
    (∃ k : ℕ, concatenate_twice A = 2016 * k) →
    A ≥ 288 :=
sorry

end smallest_concatenated_multiple_of_2016_l3979_397926


namespace both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l3979_397981

-- Define the propositions
variable (p q : Prop)

-- Define the shooting scenarios
def both_unsuccessful : Prop := ¬p ∧ ¬q
def both_successful : Prop := p ∧ q
def exactly_one_successful : Prop := (¬p ∧ q) ∨ (p ∧ ¬q)
def at_least_one_successful : Prop := p ∨ q
def at_most_one_successful : Prop := ¬(p ∧ q)

-- Theorem statements
theorem both_unsuccessful_correct (p q : Prop) : 
  both_unsuccessful p q ↔ ¬p ∧ ¬q := by sorry

theorem both_successful_correct (p q : Prop) : 
  both_successful p q ↔ p ∧ q := by sorry

theorem exactly_one_successful_correct (p q : Prop) : 
  exactly_one_successful p q ↔ (¬p ∧ q) ∨ (p ∧ ¬q) := by sorry

theorem at_least_one_successful_correct (p q : Prop) : 
  at_least_one_successful p q ↔ p ∨ q := by sorry

theorem at_most_one_successful_correct (p q : Prop) : 
  at_most_one_successful p q ↔ ¬(p ∧ q) := by sorry

end both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l3979_397981


namespace geometric_sequence_max_product_l3979_397982

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^((n * (n - 1)) / 2))

theorem geometric_sequence_max_product :
  ∃ (q : ℝ) (n : ℕ),
    geometric_sequence (-6) q 4 = (-3/4) ∧
    q = (1/2) ∧
    n = 4 ∧
    ∀ (m : ℕ), m ≠ 0 → product_of_terms (-6) q m ≤ product_of_terms (-6) q n :=
by sorry

end geometric_sequence_max_product_l3979_397982


namespace max_value_sin_cos_l3979_397977

theorem max_value_sin_cos (a b : ℝ) (h : a^2 + b^2 ≥ 1) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → a * Real.sin θ + b * Real.cos θ ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ a * Real.sin θ + b * Real.cos θ = Real.sqrt (a^2 + b^2)) :=
by sorry

end max_value_sin_cos_l3979_397977


namespace savings_calculation_l3979_397918

theorem savings_calculation (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 10 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 11400 := by
sorry

end savings_calculation_l3979_397918


namespace stratified_sampling_male_count_l3979_397937

theorem stratified_sampling_male_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) :
  (sample_size * total_male) / (total_male + total_female) = 12 := by
  sorry

end stratified_sampling_male_count_l3979_397937


namespace optimal_chair_removal_l3979_397912

def chairs_per_row : ℕ := 13
def initial_chairs : ℕ := 169
def expected_attendees : ℕ := 95
def max_removable_chairs : ℕ := 26

theorem optimal_chair_removal :
  ∀ n : ℕ,
  n ≤ max_removable_chairs →
  (initial_chairs - n) % chairs_per_row = 0 →
  initial_chairs - max_removable_chairs ≤ initial_chairs - n →
  (initial_chairs - n) - expected_attendees ≥
    (initial_chairs - max_removable_chairs) - expected_attendees :=
by sorry

end optimal_chair_removal_l3979_397912


namespace rowing_round_trip_time_l3979_397900

/-- The time taken for a round trip rowing journey given the rowing speed, current velocity, and distance. -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_velocity : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_velocity = 2)
  (h3 : distance = 72)
  : (distance / (rowing_speed - current_velocity) + distance / (rowing_speed + current_velocity)) = 15 :=
by sorry

end rowing_round_trip_time_l3979_397900


namespace flour_already_added_l3979_397941

theorem flour_already_added (total_flour : ℕ) (flour_needed : ℕ) (flour_already_added : ℕ) : 
  total_flour = 9 → flour_needed = 6 → flour_already_added = total_flour - flour_needed → 
  flour_already_added = 3 := by
sorry

end flour_already_added_l3979_397941
