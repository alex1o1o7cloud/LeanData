import Mathlib

namespace NUMINAMATH_CALUDE_real_roots_imply_real_roots_l502_50200

/-- Given a quadratic equation x^2 + px + q = 0 with real roots, 
    prove that related equations also have real roots. -/
theorem real_roots_imply_real_roots 
  (p q k x₁ x₂ : ℝ) 
  (hk : k ≠ 0) 
  (hx : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  ∃ (y₁ y₂ z₁ z₂ z₁' z₂' : ℝ), 
    (y₁^2 + (k + 1/k)*p*y₁ + p^2 + q*(k - 1/k)^2 = 0 ∧ 
     y₂^2 + (k + 1/k)*p*y₂ + p^2 + q*(k - 1/k)^2 = 0) ∧
    (z₁^2 - y₁*z₁ + q = 0 ∧ z₂^2 - y₁*z₂ + q = 0) ∧
    (z₁'^2 - y₂*z₁' + q = 0 ∧ z₂'^2 - y₂*z₂' + q = 0) ∧
    y₁ = k*x₁ + (1/k)*x₂ ∧ 
    y₂ = k*x₂ + (1/k)*x₁ ∧
    z₁ = k*x₁ ∧ 
    z₂ = (1/k)*x₂ ∧ 
    z₁' = k*x₂ ∧ 
    z₂' = (1/k)*x₁ := by
  sorry

end NUMINAMATH_CALUDE_real_roots_imply_real_roots_l502_50200


namespace NUMINAMATH_CALUDE_division_problem_solution_l502_50274

theorem division_problem_solution :
  ∀ (D d q r : ℕ),
    D + d + q + r = 205 →
    q = d →
    D = d * q + r →
    D = 174 ∧ d = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_solution_l502_50274


namespace NUMINAMATH_CALUDE_original_game_points_l502_50218

/-- The number of points in the original game -/
def P : ℕ := 60

/-- X can give Y 20 points in a game of P points -/
def X_gives_Y (p : ℕ) : Prop := p - 20 > 0

/-- X can give Z 30 points in a game of P points -/
def X_gives_Z (p : ℕ) : Prop := p - 30 > 0

/-- In a game of 120 points, Y can give Z 30 points -/
def Y_gives_Z_120 : Prop := 120 - 30 > 0

/-- The ratio of scores when Y and Z play against X is equal to their ratio in a 120-point game -/
def score_ratio (p : ℕ) : Prop := (p - 20) * 90 = (p - 30) * 120

theorem original_game_points :
  X_gives_Y P ∧ X_gives_Z P ∧ Y_gives_Z_120 ∧ score_ratio P → P = 60 :=
by sorry

end NUMINAMATH_CALUDE_original_game_points_l502_50218


namespace NUMINAMATH_CALUDE_count_fourth_powers_between_10_and_10000_l502_50213

theorem count_fourth_powers_between_10_and_10000 : 
  (Finset.filter (fun n : ℕ => 10 ≤ n^4 ∧ n^4 ≤ 10000) (Finset.range (10000 + 1))).card = 19 :=
by sorry

end NUMINAMATH_CALUDE_count_fourth_powers_between_10_and_10000_l502_50213


namespace NUMINAMATH_CALUDE_total_ndfl_is_11050_l502_50267

/-- Calculates the total NDFL (personal income tax) on income from securities --/
def calculate_ndfl (dividend_income : ℝ) (ofz_coupon_income : ℝ) (corporate_coupon_income : ℝ) 
  (shares_sold : ℕ) (sale_price_per_share : ℝ) (purchase_price_per_share : ℝ) 
  (dividend_tax_rate : ℝ) (corporate_coupon_tax_rate : ℝ) (capital_gains_tax_rate : ℝ) : ℝ :=
  let capital_gains := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let dividend_tax := dividend_income * dividend_tax_rate
  let corporate_coupon_tax := corporate_coupon_income * corporate_coupon_tax_rate
  let capital_gains_tax := capital_gains * capital_gains_tax_rate
  dividend_tax + corporate_coupon_tax + capital_gains_tax

/-- Theorem stating that the total NDFL on income from securities is 11,050 rubles --/
theorem total_ndfl_is_11050 :
  calculate_ndfl 50000 40000 30000 100 200 150 0.13 0.13 0.13 = 11050 := by
  sorry

end NUMINAMATH_CALUDE_total_ndfl_is_11050_l502_50267


namespace NUMINAMATH_CALUDE_total_students_is_540_l502_50206

/-- Represents the student population of a high school. -/
structure StudentPopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the student population problem. -/
def studentPopulationProblem (p : StudentPopulation) : Prop :=
  p.sophomores = 144 ∧
  p.freshmen = (125 * p.juniors) / 100 ∧
  p.sophomores = (90 * p.freshmen) / 100 ∧
  p.seniors = (20 * (p.freshmen + p.sophomores + p.juniors + p.seniors)) / 100

/-- The theorem stating that the total number of students is 540. -/
theorem total_students_is_540 (p : StudentPopulation) 
  (h : studentPopulationProblem p) : 
  p.freshmen + p.sophomores + p.juniors + p.seniors = 540 := by
  sorry


end NUMINAMATH_CALUDE_total_students_is_540_l502_50206


namespace NUMINAMATH_CALUDE_gcd_of_factorial_8_and_10_l502_50223

theorem gcd_of_factorial_8_and_10 :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_factorial_8_and_10_l502_50223


namespace NUMINAMATH_CALUDE_sine_double_angle_l502_50234

theorem sine_double_angle (A : ℝ) (h : Real.cos (π/4 + A) = 5/13) : 
  Real.sin (2 * A) = 119/169 := by
  sorry

end NUMINAMATH_CALUDE_sine_double_angle_l502_50234


namespace NUMINAMATH_CALUDE_max_russian_score_l502_50209

/-- Represents a chess player -/
structure Player where
  country : String
  score : ℚ

/-- Represents a chess tournament -/
structure Tournament where
  players : Finset Player
  russianPlayers : Finset Player
  winner : Player
  runnerUp : Player

/-- The scoring system for the tournament -/
def scoringSystem : ℚ × ℚ × ℚ := (1, 1/2, 0)

/-- Theorem statement for the maximum score of Russian players -/
theorem max_russian_score (t : Tournament) : 
  t.players.card = 20 ∧ 
  t.russianPlayers.card = 6 ∧
  t.winner.country = "Russia" ∧
  t.runnerUp.country = "Armenia" ∧
  t.winner.score > t.runnerUp.score ∧
  (∀ p ∈ t.players, p ≠ t.winner → p ≠ t.runnerUp → t.runnerUp.score > p.score) →
  (t.russianPlayers.sum (λ p => p.score)) ≤ 96 := by
  sorry

end NUMINAMATH_CALUDE_max_russian_score_l502_50209


namespace NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l502_50247

/-- Represents a chromosome in a human cell -/
structure Chromosome where
  parent : Bool  -- true for paternal, false for maternal

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Represents a human cell -/
structure HumanCell where
  chromosomePairs : List HomologousPair

/-- Axiom: Humans reproduce sexually -/
axiom human_sexual_reproduction : True

/-- Axiom: Fertilization involves fusion of sperm and egg cells -/
axiom fertilization_fusion : True

/-- Axiom: Meiosis occurs in formation of reproductive cells -/
axiom meiosis_in_reproduction : True

/-- Axiom: Zygote chromosome count is restored to somatic cell count -/
axiom zygote_chromosome_restoration : True

/-- Axiom: Half of zygote chromosomes from sperm, half from egg -/
axiom zygote_chromosome_origin : True

/-- Theorem: Each pair of homologous chromosomes is provided by both parents -/
theorem homologous_pair_from_both_parents (cell : HumanCell) : 
  ∀ pair ∈ cell.chromosomePairs, pair.chromosome1.parent ≠ pair.chromosome2.parent := by
  sorry


end NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l502_50247


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l502_50291

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l502_50291


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l502_50219

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l502_50219


namespace NUMINAMATH_CALUDE_parabola_vertex_l502_50239

/-- The equation of a parabola is y^2 - 4y + 2x + 8 = 0. 
    This theorem proves that the vertex of the parabola is (-2, 2). -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 - 4*y + 2*x + 8 = 0 → (x, y) = (-2, 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l502_50239


namespace NUMINAMATH_CALUDE_complex_equation_implies_product_l502_50292

theorem complex_equation_implies_product (x y : ℝ) : 
  (x + Complex.I) * (3 + y * Complex.I) = (2 : ℂ) + 4 * Complex.I → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_product_l502_50292


namespace NUMINAMATH_CALUDE_highest_backing_is_5000_l502_50233

/-- Represents the financial backing levels for a crowdfunding campaign -/
structure FinancialBacking where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  backers_lowest : ℕ
  backers_second : ℕ
  backers_highest : ℕ
  total_raised : ℕ

/-- The financial backing levels satisfy the given conditions -/
def ValidFinancialBacking (fb : FinancialBacking) : Prop :=
  fb.second_level = 10 * fb.lowest_level ∧
  fb.highest_level = 10 * fb.second_level ∧
  fb.backers_lowest = 10 ∧
  fb.backers_second = 3 ∧
  fb.backers_highest = 2 ∧
  fb.total_raised = 12000 ∧
  fb.total_raised = fb.backers_lowest * fb.lowest_level + 
                    fb.backers_second * fb.second_level + 
                    fb.backers_highest * fb.highest_level

/-- Theorem: The highest level of financial backing is $5000 -/
theorem highest_backing_is_5000 (fb : FinancialBacking) 
  (h : ValidFinancialBacking fb) : fb.highest_level = 5000 := by
  sorry

end NUMINAMATH_CALUDE_highest_backing_is_5000_l502_50233


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l502_50249

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  ∃ (M : ℝ), M = 27 ∧ x^3 / y^4 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 
    3 ≤ x₀ * y₀^2 ∧ x₀ * y₀^2 ≤ 8 ∧ 
    4 ≤ x₀^2 / y₀ ∧ x₀^2 / y₀ ≤ 9 ∧ 
    x₀^3 / y₀^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l502_50249


namespace NUMINAMATH_CALUDE_power_equality_l502_50243

theorem power_equality : (32 : ℕ)^4 * 4^5 = 2^30 := by sorry

end NUMINAMATH_CALUDE_power_equality_l502_50243


namespace NUMINAMATH_CALUDE_set_operations_l502_50246

def A : Set ℝ := {x | x < -2 ∨ x > 5}
def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem set_operations :
  (Aᶜ : Set ℝ) = {x | -2 ≤ x ∧ x ≤ 5} ∧
  (Bᶜ : Set ℝ) = {x | x < 4 ∨ x > 6} ∧
  (A ∩ B : Set ℝ) = {x | 5 < x ∧ x ≤ 6} ∧
  ((A ∪ B)ᶜ : Set ℝ) = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l502_50246


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l502_50261

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * ((-31 - Real.sqrt 621) / 10)^2 + 31 * ((-31 - Real.sqrt 621) / 10) + u = 0) → 
  u = 85 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l502_50261


namespace NUMINAMATH_CALUDE_range_m_prop_p_range_m_prop_p_not_q_l502_50299

/-- Proposition p: For all real x, x²-2mx-3m > 0 -/
def prop_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x - 3*m > 0

/-- Proposition q: There exists a real x such that x²+4mx+1 < 0 -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*m*x + 1 < 0

/-- The range of m for which proposition p is true -/
theorem range_m_prop_p : 
  {m : ℝ | prop_p m} = Set.Ioo (-3) 0 :=
sorry

/-- The range of m for which proposition p is true and proposition q is false -/
theorem range_m_prop_p_not_q : 
  {m : ℝ | prop_p m ∧ ¬(prop_q m)} = Set.Ico (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_m_prop_p_range_m_prop_p_not_q_l502_50299


namespace NUMINAMATH_CALUDE_hexagon_side_length_l502_50248

/-- A regular hexagon with a point inside it -/
structure RegularHexagonWithPoint where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- The point inside the hexagon -/
  point : ℝ × ℝ
  /-- First vertex of the hexagon -/
  vertex1 : ℝ × ℝ
  /-- Second vertex of the hexagon -/
  vertex2 : ℝ × ℝ
  /-- Third vertex of the hexagon -/
  vertex3 : ℝ × ℝ
  /-- The hexagon is regular -/
  regular : side_length > 0
  /-- The distance between the point and the first vertex is 1 -/
  dist1 : Real.sqrt ((point.1 - vertex1.1)^2 + (point.2 - vertex1.2)^2) = 1
  /-- The distance between the point and the second vertex is 1 -/
  dist2 : Real.sqrt ((point.1 - vertex2.1)^2 + (point.2 - vertex2.2)^2) = 1
  /-- The distance between the point and the third vertex is 2 -/
  dist3 : Real.sqrt ((point.1 - vertex3.1)^2 + (point.2 - vertex3.2)^2) = 2
  /-- The vertices are consecutive -/
  consecutive : Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = side_length ∧
                Real.sqrt ((vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2) = side_length

/-- The theorem stating that the side length of the hexagon is √3 -/
theorem hexagon_side_length (h : RegularHexagonWithPoint) : h.side_length = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l502_50248


namespace NUMINAMATH_CALUDE_museum_trip_total_l502_50221

theorem museum_trip_total (first_bus second_bus third_bus fourth_bus : ℕ) : 
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  fourth_bus = first_bus + 9 →
  first_bus + second_bus + third_bus + fourth_bus = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l502_50221


namespace NUMINAMATH_CALUDE_smallest_area_three_interior_points_l502_50201

/-- A square with diagonals aligned with coordinate axes -/
structure AlignedSquare where
  side : ℝ
  center : ℝ × ℝ

/-- Count of interior lattice points in a square -/
def interiorLatticePoints (s : AlignedSquare) : ℕ := sorry

/-- The area of an AlignedSquare -/
def area (s : AlignedSquare) : ℝ := s.side * s.side

/-- Theorem: Smallest area of an AlignedSquare with exactly three interior lattice points is 8 -/
theorem smallest_area_three_interior_points :
  ∃ (s : AlignedSquare), 
    interiorLatticePoints s = 3 ∧ 
    area s = 8 ∧
    ∀ (t : AlignedSquare), interiorLatticePoints t = 3 → area t ≥ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_area_three_interior_points_l502_50201


namespace NUMINAMATH_CALUDE_race_time_difference_l502_50297

/-- 
Given a 1000-meter race where runner A completes the race in 192 seconds and 
is 40 meters ahead of runner B at the finish line, prove that A beats B by 7.68 seconds.
-/
theorem race_time_difference (race_distance : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_distance = 1000 →
  a_time = 192 →
  distance_difference = 40 →
  (race_distance / a_time) * (distance_difference / race_distance) * a_time = 7.68 :=
by sorry

end NUMINAMATH_CALUDE_race_time_difference_l502_50297


namespace NUMINAMATH_CALUDE_expression_comparison_l502_50256

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2) ∧
  (∃ a b : ℝ, (a + 1/a) * (b + 1/b) > ((a + b)/2 + 2/(a + b))^2) ∧
  (∃ a b : ℝ, ((a + b)/2 + 2/(a + b))^2 > (a + 1/a) * (b + 1/b)) :=
by sorry

end NUMINAMATH_CALUDE_expression_comparison_l502_50256


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l502_50227

theorem cyclic_sum_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_3 : x + y + z = 3) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < 3 + x*y + y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l502_50227


namespace NUMINAMATH_CALUDE_bagel_cost_proof_l502_50241

/-- The cost of a dozen bagels when bought together -/
def dozen_cost : ℝ := 24

/-- The amount saved per bagel when buying a dozen -/
def savings_per_bagel : ℝ := 0.25

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The individual cost of a bagel -/
def individual_cost : ℝ := 2.25

theorem bagel_cost_proof :
  individual_cost = (dozen_cost + dozen * savings_per_bagel) / dozen :=
by sorry

end NUMINAMATH_CALUDE_bagel_cost_proof_l502_50241


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l502_50285

theorem binomial_coefficient_seven_four : (7 : ℕ).choose 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l502_50285


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_p_iff_p_minus_one_l502_50203

theorem smallest_n_divisible_by_p_iff_p_minus_one : ∃ (n : ℕ), n = 1806 ∧
  (∀ (p : ℕ), Nat.Prime p → (p ∣ n ↔ (p - 1) ∣ n)) ∧
  (∀ (m : ℕ), m < n → ∃ (q : ℕ), Nat.Prime q ∧ ((q ∣ m ↔ (q - 1) ∣ m) → False)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_p_iff_p_minus_one_l502_50203


namespace NUMINAMATH_CALUDE_pepsi_volume_l502_50269

theorem pepsi_volume (maaza : ℕ) (sprite : ℕ) (total_cans : ℕ) (pepsi : ℕ) : 
  maaza = 40 →
  sprite = 368 →
  total_cans = 69 →
  (maaza + sprite + pepsi) % total_cans = 0 →
  pepsi = 75 :=
by sorry

end NUMINAMATH_CALUDE_pepsi_volume_l502_50269


namespace NUMINAMATH_CALUDE_double_base_cost_increase_l502_50266

/-- The cost function for a given base value -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that doubling the base value results in a cost that is 1600% of the original -/
theorem double_base_cost_increase (t : ℝ) (b : ℝ) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

end NUMINAMATH_CALUDE_double_base_cost_increase_l502_50266


namespace NUMINAMATH_CALUDE_distinct_products_between_squares_l502_50211

theorem distinct_products_between_squares (n a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 →
  a * d ≠ b * c :=
by sorry

end NUMINAMATH_CALUDE_distinct_products_between_squares_l502_50211


namespace NUMINAMATH_CALUDE_average_height_problem_l502_50289

/-- Proves that in a class of 50 students, if the average height of 10 students is 167 cm
    and the average height of the whole class is 168.6 cm, then the average height of
    the remaining 40 students is 169 cm. -/
theorem average_height_problem (total_students : ℕ) (group1_students : ℕ) 
  (group2_height : ℝ) (class_avg_height : ℝ) :
  total_students = 50 →
  group1_students = 40 →
  group2_height = 167 →
  class_avg_height = 168.6 →
  ∃ (group1_height : ℝ),
    group1_height = 169 ∧
    (group1_students : ℝ) * group1_height + (total_students - group1_students : ℝ) * group2_height =
      (total_students : ℝ) * class_avg_height :=
by sorry

end NUMINAMATH_CALUDE_average_height_problem_l502_50289


namespace NUMINAMATH_CALUDE_section_area_regular_triangular_pyramid_l502_50236

/-- The area of a section in a regular triangular pyramid -/
theorem section_area_regular_triangular_pyramid
  (a h : ℝ)
  (ha : a > 0)
  (hh : h > (a * Real.sqrt 6) / 6) :
  let area := (3 * a^2 * h) / (4 * Real.sqrt (a^2 + 3 * h^2))
  ∃ (S : ℝ), S = area ∧ S > 0 :=
by sorry

end NUMINAMATH_CALUDE_section_area_regular_triangular_pyramid_l502_50236


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l502_50240

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l502_50240


namespace NUMINAMATH_CALUDE_garden_fencing_l502_50202

/-- Calculates the perimeter of a rectangular garden with given length and width ratio --/
theorem garden_fencing (length : ℝ) (h1 : length = 80) : 
  2 * (length + length / 2) = 240 := by
  sorry


end NUMINAMATH_CALUDE_garden_fencing_l502_50202


namespace NUMINAMATH_CALUDE_equivalent_expression_proof_l502_50237

theorem equivalent_expression_proof (n : ℕ) (hn : n > 1) :
  ∃ (p q : ℕ → ℕ),
    (∀ m : ℕ, m > 1 → 16^m + 4^m + 1 = (2^(p m) - 1) / (2^(q m) - 1)) ∧
    (∃ k : ℚ, ∀ m : ℕ, m > 1 → p m / q m = k) ∧
    p 2006 - q 2006 = 8024 :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_proof_l502_50237


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_l502_50278

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_four_digit_divisible (p : ℕ) 
  (h1 : is_four_digit p)
  (h2 : is_four_digit (reverse_digits p))
  (h3 : p % 63 = 0)
  (h4 : (reverse_digits p) % 63 = 0)
  (h5 : p % 19 = 0) :
  p ≤ 5985 ∧ (∀ q : ℕ, 
    is_four_digit q → 
    is_four_digit (reverse_digits q) → 
    q % 63 = 0 → 
    (reverse_digits q) % 63 = 0 → 
    q % 19 = 0 → 
    q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_l502_50278


namespace NUMINAMATH_CALUDE_theater_line_up_ways_l502_50224

theorem theater_line_up_ways : 
  let number_of_windows : ℕ := 2
  let number_of_people : ℕ := 6
  number_of_windows ^ number_of_people * Nat.factorial number_of_people = 46080 :=
by sorry

end NUMINAMATH_CALUDE_theater_line_up_ways_l502_50224


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_append_l502_50273

/-- A function that appends two numbers -/
def append (a b : ℕ) : ℕ := a * (10 ^ (Nat.digits 10 b).length) + b

/-- Predicate to check if a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ), append n (n + 1) = m ^ 2

/-- The smallest three-digit number satisfying the condition -/
def smallest_satisfying_number : ℕ := 183

theorem smallest_three_digit_square_append :
  (smallest_satisfying_number ≥ 100) ∧
  (smallest_satisfying_number < 1000) ∧
  satisfies_condition smallest_satisfying_number ∧
  ∀ n, n ≥ 100 ∧ n < smallest_satisfying_number → ¬(satisfies_condition n) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_append_l502_50273


namespace NUMINAMATH_CALUDE_emily_necklaces_l502_50270

/-- Given that Emily used a total of 18 beads and each necklace requires 3 beads,
    prove that the number of necklaces she made is 6. -/
theorem emily_necklaces :
  let total_beads : ℕ := 18
  let beads_per_necklace : ℕ := 3
  let necklaces_made : ℕ := total_beads / beads_per_necklace
  necklaces_made = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l502_50270


namespace NUMINAMATH_CALUDE_hyperbola_equation_l502_50208

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ  -- Half-length of the transverse axis
  b : ℝ  -- Half-length of the conjugate axis
  c : ℝ  -- Focal distance

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- Theorem: Given a hyperbola with specific properties, its standard equation is y²/4 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
  (vertex_condition : h.a = 2)
  (axis_sum_condition : 2 * h.a + 2 * h.b = Real.sqrt 2 * 2 * h.c) :
  standardEquation h x y ↔ y^2 / 4 - x^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l502_50208


namespace NUMINAMATH_CALUDE_fraction_addition_l502_50226

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l502_50226


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l502_50250

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l502_50250


namespace NUMINAMATH_CALUDE_janet_video_game_lives_l502_50225

theorem janet_video_game_lives : ∀ initial_lives : ℕ,
  initial_lives - 23 + 46 = 70 → initial_lives = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_janet_video_game_lives_l502_50225


namespace NUMINAMATH_CALUDE_max_value_of_expression_l502_50259

theorem max_value_of_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 2 ∧
    (a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c')) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l502_50259


namespace NUMINAMATH_CALUDE_max_value_expression_l502_50257

theorem max_value_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / abs x + abs y / y - (x * y) / abs (x * y) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l502_50257


namespace NUMINAMATH_CALUDE_carol_peanuts_count_l502_50279

def initial_peanuts : ℕ := 2
def additional_peanuts : ℕ := 5

theorem carol_peanuts_count : initial_peanuts + additional_peanuts = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_count_l502_50279


namespace NUMINAMATH_CALUDE_complex_equation_solution_l502_50282

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l502_50282


namespace NUMINAMATH_CALUDE_product_of_factorials_plus_one_l502_50281

theorem product_of_factorials_plus_one : 
  (1 + 1 / 1) * 
  (1 + 1 / 2) * 
  (1 + 1 / 6) * 
  (1 + 1 / 24) * 
  (1 + 1 / 120) * 
  (1 + 1 / 720) * 
  (1 + 1 / 5040) = 5041 / 5040 := by sorry

end NUMINAMATH_CALUDE_product_of_factorials_plus_one_l502_50281


namespace NUMINAMATH_CALUDE_gold_award_middle_sum_l502_50205

/-- Represents the sequence of gold awards --/
def gold_sequence (n : ℕ) : ℚ := sorry

theorem gold_award_middle_sum :
  (∀ i j : ℕ, i < j → i < 10 → j < 10 → gold_sequence j - gold_sequence i = (j - i) * (gold_sequence 1 - gold_sequence 0)) →
  gold_sequence 7 + gold_sequence 8 + gold_sequence 9 = 12 →
  gold_sequence 0 + gold_sequence 1 + gold_sequence 2 + gold_sequence 3 = 12 →
  gold_sequence 4 + gold_sequence 5 + gold_sequence 6 = 83/26 := by
  sorry

end NUMINAMATH_CALUDE_gold_award_middle_sum_l502_50205


namespace NUMINAMATH_CALUDE_product_remainder_l502_50215

theorem product_remainder (k : ℕ) : ∃ n : ℕ, 
  n = 5 * k + 1 ∧ (14452 * 15652 * n) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l502_50215


namespace NUMINAMATH_CALUDE_triangle_inradius_l502_50263

/-- Given a triangle with perimeter 28 cm and area 35 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) : 
  P = 28 → A = 35 → A = r * (P / 2) → r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l502_50263


namespace NUMINAMATH_CALUDE_two_person_subcommittees_with_male_l502_50254

theorem two_person_subcommittees_with_male (total : Nat) (men : Nat) (women : Nat) :
  total = 8 →
  men = 5 →
  women = 3 →
  Nat.choose total 2 - Nat.choose women 2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_with_male_l502_50254


namespace NUMINAMATH_CALUDE_percent_both_correct_l502_50220

theorem percent_both_correct
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_neither : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 25)
  (h3 : percent_neither = 20)
  : ℝ :=
by
  -- Define the percentage of students who answered both questions correctly
  let percent_both : ℝ := percent_first + percent_second - (100 - percent_neither)
  
  -- Prove that percent_both equals 20
  have : percent_both = 20 := by sorry
  
  -- Return the result
  exact percent_both

end NUMINAMATH_CALUDE_percent_both_correct_l502_50220


namespace NUMINAMATH_CALUDE_sqrt_square_not_always_equal_l502_50275

theorem sqrt_square_not_always_equal (a : ℝ) : ¬(∀ a, Real.sqrt (a^2) = a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_not_always_equal_l502_50275


namespace NUMINAMATH_CALUDE_cupcakes_left_l502_50217

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (5 * dozen) / 2

/-- The initial number of people in the class -/
def initial_class_size : ℕ := 27 + 1 + 1

/-- The number of students absent -/
def absent_students : ℕ := 3

/-- The actual number of people present in the class -/
def class_size : ℕ := initial_class_size - absent_students

theorem cupcakes_left : cupcakes_brought - class_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_l502_50217


namespace NUMINAMATH_CALUDE_compute_expression_l502_50253

theorem compute_expression : 3 * 3^4 + 9^60 / 9^58 = 324 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l502_50253


namespace NUMINAMATH_CALUDE_solution_set_inequality_l502_50216

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 3) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l502_50216


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l502_50286

theorem opposite_of_negative_seven :
  ∀ x : ℤ, x + (-7) = 0 → x = 7 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l502_50286


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l502_50298

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let S := {x : ℝ | x^2 + (a + 2) * x + 2 * a < 0}
  (a < 2 → S = {x : ℝ | -2 < x ∧ x < -a}) ∧
  (a = 2 → S = ∅) ∧
  (a > 2 → S = {x : ℝ | -a < x ∧ x < -2}) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l502_50298


namespace NUMINAMATH_CALUDE_rod_length_proof_l502_50262

/-- Given that a 6 meter rod weighs 6.1 kg, prove that a 12.2 kg rod is 12 meters long -/
theorem rod_length_proof (weight : ℝ) (length : ℝ) : 
  (weight = 12.2) → 
  (6.1 / 6 = weight / length) → 
  length = 12 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_proof_l502_50262


namespace NUMINAMATH_CALUDE_rebecca_groups_l502_50264

/-- The number of eggs Rebecca has -/
def total_eggs : ℕ := 20

/-- The number of marbles Rebecca has -/
def total_marbles : ℕ := 6

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 5

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The maximum number of groups that can be created -/
def max_groups : ℕ := min (total_eggs / eggs_per_group) (total_marbles / marbles_per_group)

theorem rebecca_groups : max_groups = 3 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_groups_l502_50264


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l502_50207

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l502_50207


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l502_50214

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism. -/
theorem cone_prism_volume_ratio
  (a b h_c h_p : ℝ)
  (h_ab : b < a)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_pos_h_c : h_c > 0)
  (h_pos_h_p : h_p > 0) :
  (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) = π * b * h_c / (12 * a * h_p) :=
by sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l502_50214


namespace NUMINAMATH_CALUDE_smallest_x_value_l502_50251

theorem smallest_x_value (x : ℝ) : 
  (5 * x^2 + 7 * x + 3 = 6) → x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l502_50251


namespace NUMINAMATH_CALUDE_unique_number_property_l502_50276

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l502_50276


namespace NUMINAMATH_CALUDE_circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l502_50229

-- Define the basic types
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the tangency and passing through relations
def tangent_to_line (c : Circle) (l : Line) : Prop := sorry

def passes_through (c : Circle) (p : Point) : Prop := sorry

def tangent_to_circle (c1 : Circle) (c2 : Circle) : Prop := sorry

-- Part a
theorem circle_tangent_to_two_lines_through_point 
  (l1 l2 : Line) (A : Point) : 
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ passes_through S A := by
  sorry

-- Part b
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) :
  ∃ (S : Circle), passes_through S A ∧ passes_through S B ∧ tangent_to_line S l := by
  sorry

-- Part c
theorem circle_tangent_to_two_lines_and_circle 
  (l1 l2 : Line) (S_bar : Circle) :
  ∃ (S : Circle), tangent_to_line S l1 ∧ tangent_to_line S l2 ∧ tangent_to_circle S S_bar := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_two_lines_through_point_circle_through_two_points_tangent_to_line_circle_tangent_to_two_lines_and_circle_l502_50229


namespace NUMINAMATH_CALUDE_max_garden_area_l502_50222

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 400
  length_ge : length ≥ 100
  width_ge : width ≥ 50

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a garden with given constraints -/
theorem max_garden_area :
  ∀ g : Garden, g.area ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l502_50222


namespace NUMINAMATH_CALUDE_additional_male_workers_hired_l502_50296

theorem additional_male_workers_hired (
  initial_female_percentage : ℚ)
  (final_female_percentage : ℚ)
  (final_total_employees : ℕ)
  (h1 : initial_female_percentage = 3/5)
  (h2 : final_female_percentage = 11/20)
  (h3 : final_total_employees = 240) :
  (final_total_employees : ℚ) - (final_female_percentage * final_total_employees) / initial_female_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_additional_male_workers_hired_l502_50296


namespace NUMINAMATH_CALUDE_juniper_bones_l502_50277

/-- Calculates the final number of bones Juniper has after transactions --/
def final_bones (initial : ℕ) : ℕ :=
  let additional := (initial * 50) / 100
  let total := initial + additional
  let stolen := (total * 25) / 100
  total - stolen

/-- Theorem stating that Juniper ends up with 5 bones --/
theorem juniper_bones : final_bones 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_l502_50277


namespace NUMINAMATH_CALUDE_two_functions_satisfy_equation_l502_50260

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

/-- The main theorem stating that there are exactly two functions satisfying the equation -/
theorem two_functions_satisfy_equation :
  ∃! (s : Set (ℝ → ℝ)), 
    (∀ f ∈ s, SatisfiesFunctionalEquation f) ∧ 
    s = {ZeroFunction, SquareFunction} :=
  sorry

end NUMINAMATH_CALUDE_two_functions_satisfy_equation_l502_50260


namespace NUMINAMATH_CALUDE_jim_can_bake_two_loaves_l502_50293

/-- The amount of flour Jim has in the cupboard (in grams) -/
def flour_cupboard : ℕ := 200

/-- The amount of flour Jim has on the kitchen counter (in grams) -/
def flour_counter : ℕ := 100

/-- The amount of flour Jim has in the pantry (in grams) -/
def flour_pantry : ℕ := 100

/-- The amount of flour required for one loaf of bread (in grams) -/
def flour_per_loaf : ℕ := 200

/-- The total amount of flour Jim has (in grams) -/
def total_flour : ℕ := flour_cupboard + flour_counter + flour_pantry

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := total_flour / flour_per_loaf

theorem jim_can_bake_two_loaves : loaves_baked = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_can_bake_two_loaves_l502_50293


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l502_50212

theorem min_value_sum_squares (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b - 3*b^2 = 1 → a^2 + b^2 ≥ min) ∧
  min = (Real.sqrt 5 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l502_50212


namespace NUMINAMATH_CALUDE_recess_time_calculation_l502_50210

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (base_time : ℕ) (a_count b_count c_count d_count : ℕ) : ℕ :=
  base_time + 2 * a_count + b_count - d_count

/-- Theorem stating that given the specific grade distribution, the total recess time is 47 minutes -/
theorem recess_time_calculation :
  let base_time : ℕ := 20
  let a_count : ℕ := 10
  let b_count : ℕ := 12
  let c_count : ℕ := 14
  let d_count : ℕ := 5
  total_recess_time base_time a_count b_count c_count d_count = 47 := by
  sorry

#eval total_recess_time 20 10 12 14 5

end NUMINAMATH_CALUDE_recess_time_calculation_l502_50210


namespace NUMINAMATH_CALUDE_min_S_19_l502_50287

/-- Given an arithmetic sequence {a_n} where S_8 ≤ 6 and S_11 ≥ 27, 
    the minimum value of S_19 is 133. -/
theorem min_S_19 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 8 ≤ 6 →                                 -- Given condition
  S 11 ≥ 27 →                               -- Given condition
  ∀ S_19 : ℝ, (S_19 = S 19 → S_19 ≥ 133) :=
by sorry

end NUMINAMATH_CALUDE_min_S_19_l502_50287


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_proof_l502_50238

/-- The probability of selecting two adjacent vertices when choosing 2 distinct vertices at random from a decagon -/
def prob_adjacent_vertices_decagon : ℚ := 2 / 9

/-- The number of vertices in a decagon -/
def decagon_vertices : ℕ := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def adjacent_vertices_per_vertex : ℕ := 2

theorem prob_adjacent_vertices_decagon_proof :
  prob_adjacent_vertices_decagon = 
    (adjacent_vertices_per_vertex : ℚ) / ((decagon_vertices - 1) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_proof_l502_50238


namespace NUMINAMATH_CALUDE_school_pet_ownership_l502_50295

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (rabbit_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : rabbit_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (rabbit_owners : ℚ) / total_students * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_pet_ownership_l502_50295


namespace NUMINAMATH_CALUDE_garden_flowers_l502_50231

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  rows_front : ℕ  -- Number of rows in front of the rose
  rows_back : ℕ   -- Number of rows behind the rose
  cols_right : ℕ  -- Number of columns to the right of the rose
  cols_left : ℕ   -- Number of columns to the left of the rose

/-- Calculates the total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ :=
  (g.rows_front + g.rows_back + 1) * (g.cols_right + g.cols_left + 1)

/-- Theorem stating that a garden with the given properties has 462 flowers. -/
theorem garden_flowers :
  ∀ (g : Garden),
    g.rows_front = 6 ∧
    g.rows_back = 15 ∧
    g.cols_right = 12 ∧
    g.cols_left = 8 →
    total_flowers g = 462 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_l502_50231


namespace NUMINAMATH_CALUDE_marts_income_percentage_l502_50284

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.3)) :
  mart / juan = 0.78 := by
sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l502_50284


namespace NUMINAMATH_CALUDE_train_length_calculation_l502_50280

theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 27) : ∃ L : ℝ,
  L = 37.5 ∧ 
  2 * L = (v_fast - v_slow) * (5 / 18) * t :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l502_50280


namespace NUMINAMATH_CALUDE_sqrt_144000_l502_50230

theorem sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144000_l502_50230


namespace NUMINAMATH_CALUDE_fourth_corner_rectangle_area_l502_50290

/-- Given a large rectangle divided into 9 smaller rectangles, where three corner rectangles
    have areas 9, 15, and 12, and the area ratios are the same between adjacent small rectangles,
    the area of the fourth corner rectangle is 20. -/
theorem fourth_corner_rectangle_area :
  ∀ (A B C D : ℝ),
    A = 9 →
    B = 15 →
    C = 12 →
    A / C = B / D →
    D = 20 := by
  sorry

end NUMINAMATH_CALUDE_fourth_corner_rectangle_area_l502_50290


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l502_50268

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (x / (x - 3) = 2 + m / (x - 3)) →
  (x > 0) →
  (m < 6 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l502_50268


namespace NUMINAMATH_CALUDE_system_solution_l502_50245

theorem system_solution (x y k : ℝ) : 
  x + 2*y = 2*k ∧ 
  2*x + y = 4*k ∧ 
  x + y = 4 → 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_l502_50245


namespace NUMINAMATH_CALUDE_charlies_share_l502_50271

theorem charlies_share (total : ℚ) (a b c : ℚ) : 
  total = 10000 →
  a = (1/3) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 6000 := by
sorry

end NUMINAMATH_CALUDE_charlies_share_l502_50271


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l502_50228

/-- Converts a two-digit number in base b to base 10 -/
def to_base_10 (digit : Nat) (base : Nat) : Nat :=
  base * digit + digit

/-- Checks if a digit is valid in the given base -/
def is_valid_digit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    is_valid_digit C 6 ∧
    is_valid_digit D 8 ∧
    to_base_10 C 6 = to_base_10 D 8 ∧
    to_base_10 C 6 = 63 ∧
    (∀ (C' D' : Nat),
      is_valid_digit C' 6 →
      is_valid_digit D' 8 →
      to_base_10 C' 6 = to_base_10 D' 8 →
      to_base_10 C' 6 ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l502_50228


namespace NUMINAMATH_CALUDE_backyard_area_l502_50288

theorem backyard_area (length width : ℝ) 
  (h1 : length * 50 = 2000)
  (h2 : (2 * length + 2 * width) * 20 = 2000) :
  length * width = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l502_50288


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l502_50244

-- Define the number of each type of clothing
def num_hats : ℕ := 3
def num_shirts : ℕ := 4
def num_shorts : ℕ := 5
def num_socks : ℕ := 6

-- Define the total number of articles
def total_articles : ℕ := num_hats + num_shirts + num_shorts + num_socks

-- Define the number of articles to be drawn
def draw_count : ℕ := 4

-- Theorem statement
theorem probability_of_specific_draw :
  (num_hats.choose 1 * num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1) /
  (total_articles.choose draw_count) = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l502_50244


namespace NUMINAMATH_CALUDE_free_flowers_per_dozen_l502_50204

def flowers_per_dozen : ℕ := 12

theorem free_flowers_per_dozen 
  (bought_dozens : ℕ) 
  (total_flowers : ℕ) 
  (h1 : bought_dozens = 3) 
  (h2 : total_flowers = 42) : ℕ := by
  sorry

#check free_flowers_per_dozen

end NUMINAMATH_CALUDE_free_flowers_per_dozen_l502_50204


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l502_50235

theorem polynomial_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 2) * (2 * x^3 - 7) = a * x^5 + k * x^2 + n) →
  a - n + k = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l502_50235


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l502_50232

/-- Given a cubic polynomial x^3 + px + q = 0 where p and q are rational,
    if 3 - √5 is a root and the polynomial has an integer root,
    then this integer root must be -6. -/
theorem cubic_polynomial_integer_root
  (p q : ℚ)
  (h1 : ∃ (x : ℝ), x^3 + p*x + q = 0)
  (h2 : (3 - Real.sqrt 5)^3 + p*(3 - Real.sqrt 5) + q = 0)
  (h3 : ∃ (r : ℤ), r^3 + p*r + q = 0) :
  ∃ (r : ℤ), r^3 + p*r + q = 0 ∧ r = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l502_50232


namespace NUMINAMATH_CALUDE_perimeter_semicircular_pentagon_l502_50272

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a regular pentagon --/
theorem perimeter_semicircular_pentagon (side_length : ℝ) : 
  side_length = 5 / π → 
  (5 : ℝ) * (π * side_length / 2) = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_pentagon_l502_50272


namespace NUMINAMATH_CALUDE_three_square_games_l502_50258

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 4

/-- The number of games two specific players play together -/
def games_together : ℕ := 45

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem three_square_games (p1 p2 : Fin total_players) : 
  p1 ≠ p2 → (total_combinations / total_players) * (players_per_game - 1) / (total_players - 1) = games_together :=
sorry

end NUMINAMATH_CALUDE_three_square_games_l502_50258


namespace NUMINAMATH_CALUDE_student_boat_problem_l502_50294

theorem student_boat_problem (students boats : ℕ) : 
  (7 * boats + 5 = students) → 
  (8 * boats = students + 2) → 
  (students = 54 ∧ boats = 7) :=
by sorry

end NUMINAMATH_CALUDE_student_boat_problem_l502_50294


namespace NUMINAMATH_CALUDE_milk_division_l502_50252

theorem milk_division (total_milk : ℚ) (num_kids : ℕ) (milk_per_kid : ℚ) : 
  total_milk = 3 → 
  num_kids = 5 → 
  milk_per_kid = total_milk / num_kids → 
  milk_per_kid = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_division_l502_50252


namespace NUMINAMATH_CALUDE_pat_to_kate_ratio_l502_50242

-- Define the variables
def total_hours : ℕ := 117
def mark_extra_hours : ℕ := 65

-- Define the hours charged by each person as real numbers
variable (pat_hours kate_hours mark_hours : ℝ)

-- Define the conditions
axiom total_hours_sum : pat_hours + kate_hours + mark_hours = total_hours
axiom pat_to_mark_ratio : pat_hours = (1/3) * mark_hours
axiom mark_to_kate_diff : mark_hours = kate_hours + mark_extra_hours

-- Define the theorem
theorem pat_to_kate_ratio :
  (∃ r : ℝ, pat_hours = r * kate_hours) →
  pat_hours / kate_hours = 2 := by sorry

end NUMINAMATH_CALUDE_pat_to_kate_ratio_l502_50242


namespace NUMINAMATH_CALUDE_larger_integer_proof_l502_50255

theorem larger_integer_proof (x y : ℕ+) : 
  (y = x + 8) → (x * y = 272) → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l502_50255


namespace NUMINAMATH_CALUDE_brick_length_calculation_l502_50283

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℝ := 2500  -- 25 meters = 2500 cm
def courtyard_width : ℝ := 1600   -- 16 meters = 1600 cm

-- Define the brick properties
def brick_width : ℝ := 10         -- 10 cm
def total_bricks : ℕ := 20000

-- Define the theorem
theorem brick_length_calculation :
  ∃ (brick_length : ℝ),
    brick_length > 0 ∧
    brick_length * brick_width * total_bricks = courtyard_length * courtyard_width ∧
    brick_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l502_50283


namespace NUMINAMATH_CALUDE_equation_solution_l502_50265

theorem equation_solution :
  ∀ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l502_50265
