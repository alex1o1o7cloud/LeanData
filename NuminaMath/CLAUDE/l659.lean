import Mathlib

namespace circle_equation_from_parabola_focus_l659_65925

/-- Given a parabola y^2 = 4x and a circle with its center at the focus of the parabola
    passing through the origin, the equation of the circle is x^2 + y^2 - 2x = 0 -/
theorem circle_equation_from_parabola_focus (x y : ℝ) :
  (y^2 = 4*x) →  -- Parabola equation
  (∃ (h k r : ℝ), (h = 1 ∧ k = 0) ∧  -- Focus at (1, 0)
    ((0 - h)^2 + (0 - k)^2 = r^2) ∧  -- Circle passes through origin
    ((x - h)^2 + (y - k)^2 = r^2)) →  -- General circle equation
  x^2 + y^2 - 2*x = 0 :=  -- Resulting circle equation
by sorry

end circle_equation_from_parabola_focus_l659_65925


namespace piggy_bank_savings_l659_65960

theorem piggy_bank_savings (x y : ℕ) : 
  x + y = 290 →  -- Total number of coins
  2 * (y / 4) = x / 3 →  -- Relationship between coin values
  2 * y + x = 406  -- Total amount saved
  := by sorry

end piggy_bank_savings_l659_65960


namespace gravitational_force_at_distance_l659_65920

/-- Gravitational force function -/
noncomputable def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / d^2

theorem gravitational_force_at_distance
  (k : ℝ)
  (h1 : gravitational_force k 5000 = 500)
  (h2 : k > 0) :
  gravitational_force k 25000 = 1/5 := by
  sorry

#check gravitational_force_at_distance

end gravitational_force_at_distance_l659_65920


namespace hotel_tax_calculation_l659_65937

/-- Calculates the business tax paid given revenue and tax rate -/
def business_tax (revenue : ℕ) (tax_rate : ℚ) : ℚ :=
  (revenue : ℚ) * tax_rate

theorem hotel_tax_calculation :
  let revenue : ℕ := 10000000  -- 10 million yuan
  let tax_rate : ℚ := 5 / 100   -- 5%
  business_tax revenue tax_rate = 500 := by sorry

end hotel_tax_calculation_l659_65937


namespace ball_collection_theorem_l659_65959

theorem ball_collection_theorem (r b y : ℕ) : 
  b + y = 9 →
  r + y = 5 →
  r + b = 6 →
  r + b + y = 10 := by
sorry

end ball_collection_theorem_l659_65959


namespace isosceles_triangle_perimeter_l659_65931

theorem isosceles_triangle_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 9*x₁ + 18 = 0 →
  x₂^2 - 9*x₂ + 18 = 0 →
  x₁ ≠ x₂ →
  (x₁ + x₂ + max x₁ x₂ = 15) :=
sorry

end isosceles_triangle_perimeter_l659_65931


namespace more_birds_than_nests_l659_65913

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end more_birds_than_nests_l659_65913


namespace sqrt_10_factorial_div_210_l659_65911

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sqrt_10_factorial_div_210 : 
  Real.sqrt (factorial 10 / 210) = 72 * Real.sqrt 5 := by sorry

end sqrt_10_factorial_div_210_l659_65911


namespace twelve_digit_divisibility_l659_65962

theorem twelve_digit_divisibility (n : ℕ) (h : 100000 ≤ n ∧ n < 1000000) :
  ∃ k : ℕ, 1000001 * n + n = 1000001 * k := by
  sorry

end twelve_digit_divisibility_l659_65962


namespace intersection_condition_union_condition_l659_65963

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a^2 - 12 = 0}

-- Part 1
theorem intersection_condition (a : ℝ) : A ∩ B a = A → a = -2 := by sorry

-- Part 2
theorem union_condition (a : ℝ) : A ∪ B a = A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end intersection_condition_union_condition_l659_65963


namespace relationship_abc_l659_65904

theorem relationship_abc : 
  2022^0 > 8^2022 * (-0.125)^2023 ∧ 8^2022 * (-0.125)^2023 > 2021 * 2023 - 2022^2 := by
  sorry

end relationship_abc_l659_65904


namespace well_volume_l659_65946

/-- The volume of a circular cylinder with diameter 2 meters and height 14 meters is 14π cubic meters. -/
theorem well_volume :
  let diameter : ℝ := 2
  let height : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  volume = 14 * π :=
by sorry

end well_volume_l659_65946


namespace implication_equivalence_l659_65932

theorem implication_equivalence (P Q : Prop) : 
  (P → Q) ↔ (¬Q → ¬P) :=
sorry

end implication_equivalence_l659_65932


namespace isosceles_triangle_50_largest_angle_l659_65991

/-- An isosceles triangle with one angle opposite an equal side measuring 50 degrees -/
structure IsoscelesTriangle50 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- Assertion that the angle opposite an equal side is 50 degrees -/
  h_angle_50 : angle_opposite_equal_side = 50

/-- 
Theorem: In an isosceles triangle where one of the angles opposite an equal side 
measures 50°, the largest angle measures 80°.
-/
theorem isosceles_triangle_50_largest_angle 
  (t : IsoscelesTriangle50) : t.largest_angle = 80 := by
  sorry

end isosceles_triangle_50_largest_angle_l659_65991


namespace root_sum_product_l659_65914

theorem root_sum_product (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (-3 + 2 * Complex.I)^2 + p * (-3 + 2 * Complex.I) + q = 0 →
  p + q = 38 := by sorry

end root_sum_product_l659_65914


namespace searchlight_revolutions_per_minute_l659_65966

/-- 
Given a searchlight that completes one revolution in a time period where 
half of that period is 10 seconds of darkness, prove that the number of 
revolutions per minute is 3.
-/
theorem searchlight_revolutions_per_minute : 
  ∀ (r : ℝ), 
  (r > 0) →  -- r is positive (revolutions per minute)
  (60 / r / 2 = 10) →  -- half the period of one revolution is 10 seconds
  r = 3 := by sorry

end searchlight_revolutions_per_minute_l659_65966


namespace road_travel_cost_l659_65957

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width travel_cost_per_sqm : ℕ) : 
  lawn_length = 90 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  travel_cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * travel_cost_per_sqm = 4200 :=
by sorry

end road_travel_cost_l659_65957


namespace f_properties_l659_65950

noncomputable def f (x : ℝ) := Real.log ((1 + x) / (1 - x))

theorem f_properties :
  ∃ (k : ℝ),
    (∀ x ∈ Set.Ioo 0 1, f x > k * (x + x^3 / 3)) ∧
    (∀ k' > k, ∃ x ∈ Set.Ioo 0 1, f x ≤ k' * (x + x^3 / 3)) ∧
    k = 2 ∧
    (∀ x ∈ Set.Ioo 0 1, f x > 2 * (x + x^3 / 3)) ∧
    (∀ h ∈ Set.Ioo 0 1, (f h - f 0) / h = 2) :=
by sorry

end f_properties_l659_65950


namespace range_of_m_l659_65910

def p (m : ℝ) : Prop := 0 ≤ m ∧ m ≤ 3

def q (m : ℝ) : Prop := (m - 2) * (m - 4) ≤ 0

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ∈ Set.Icc 0 2 ∪ Set.Ioc 3 4) :=
by sorry

end range_of_m_l659_65910


namespace contrapositive_diagonals_parallelogram_l659_65979

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for diagonals to bisect each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := (q.vertices 0 + q.vertices 2) / 2
  let mid2 := (q.vertices 1 + q.vertices 3) / 2
  mid1 = mid2

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 0 - q.vertices 1) = (q.vertices 3 - q.vertices 2) ∧
  (q.vertices 0 - q.vertices 3) = (q.vertices 1 - q.vertices 2)

-- The theorem to prove
theorem contrapositive_diagonals_parallelogram :
  ∀ q : Quadrilateral, ¬(is_parallelogram q) → ¬(diagonals_bisect q) :=
by sorry

end contrapositive_diagonals_parallelogram_l659_65979


namespace polynomial_identity_sum_of_squares_l659_65999

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ y, 729 * y^3 + 64 = (p * y^2 + q * y + r) * (s * y^2 + t * y + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 543106 := by
sorry

end polynomial_identity_sum_of_squares_l659_65999


namespace translation_proof_l659_65915

-- Define the original linear function
def original_function (x : ℝ) : ℝ := 3 * x - 1

-- Define the translation
def translation : ℝ := 3

-- Define the resulting function after translation
def translated_function (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem translation_proof :
  ∀ x : ℝ, translated_function x = original_function x + translation :=
by
  sorry

end translation_proof_l659_65915


namespace expected_successes_eq_38_l659_65945

/-- The probability of getting at least one 5 or 6 when throwing 3 dice -/
def p : ℚ := 19 / 27

/-- The number of experiments -/
def n : ℕ := 54

/-- A trial is successful if at least one 5 or 6 appears when throwing 3 dice -/
axiom success_definition : True

/-- The number of successful trials follows a binomial distribution -/
axiom binomial_distribution : True

/-- The expected number of successful trials in 54 experiments -/
def expected_successes : ℚ := n * p

theorem expected_successes_eq_38 : expected_successes = 38 := by
  sorry

end expected_successes_eq_38_l659_65945


namespace train_speed_l659_65902

/-- Given a train of length 350 meters that crosses a pole in 21 seconds, its speed is 60 km/hr. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 350) (h2 : crossing_time = 21) :
  (train_length / 1000) / (crossing_time / 3600) = 60 :=
sorry

end train_speed_l659_65902


namespace smallest_fraction_greater_than_31_17_l659_65987

theorem smallest_fraction_greater_than_31_17 :
  ∀ a b : ℤ, b < 17 → (a : ℚ) / b > 31 / 17 → 11 / 6 ≤ (a : ℚ) / b :=
by
  sorry

end smallest_fraction_greater_than_31_17_l659_65987


namespace factorization_identity_l659_65973

theorem factorization_identity (m : ℝ) : m^2 + 3*m = m*(m + 3) := by
  sorry

end factorization_identity_l659_65973


namespace arithmetic_geometric_sequence_ratio_l659_65948

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℝ),
    is_arithmetic_sequence (-1) a₁ a₂ 8 →
    is_geometric_sequence (-1) b₁ b₂ b₃ (-4) →
    (a₁ * a₂) / b₂ = -5 := by
  sorry

end arithmetic_geometric_sequence_ratio_l659_65948


namespace problem_statement_l659_65907

theorem problem_statement : 3 * 3^4 - 9^35 / 9^33 = 162 := by
  sorry

end problem_statement_l659_65907


namespace town_population_proof_l659_65982

/-- The annual decrease rate of the town's population -/
def annual_decrease_rate : ℝ := 0.2

/-- The population after 2 years -/
def population_after_2_years : ℝ := 19200

/-- The initial population of the town -/
def initial_population : ℝ := 30000

theorem town_population_proof :
  let remaining_rate := 1 - annual_decrease_rate
  (remaining_rate ^ 2) * initial_population = population_after_2_years :=
by sorry

end town_population_proof_l659_65982


namespace circle_center_transformation_l659_65980

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x initial_center
  let final_position := translate_right reflected 5
  final_position = (3, -6) := by sorry

end circle_center_transformation_l659_65980


namespace root_equation_l659_65903

noncomputable def f (x : ℝ) : ℝ := if x < 0 then -2*x else x^2 - 1

theorem root_equation (a : ℝ) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x : ℝ, f x + 2 * Real.sqrt (1 - x^2) + |f x - 2 * Real.sqrt (1 - x^2)| - 2*a*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  x₃ - x₂ = 2*(x₂ - x₁) →
  a = (Real.sqrt 17 - 3) / 2 :=
sorry

end root_equation_l659_65903


namespace probability_closer_to_center_l659_65984

theorem probability_closer_to_center (r : ℝ) (h : r > 0) :
  let outer_circle_area := π * r^2
  let inner_circle_area := π * r
  let probability := inner_circle_area / outer_circle_area
  probability = 1/4 := by
sorry

end probability_closer_to_center_l659_65984


namespace remaining_segments_theorem_l659_65947

/-- Represents the spiral pattern described in the problem -/
def spiral_pattern (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2) + n + 1

/-- The total length of the spiral in centimeters -/
def total_length : ℕ := 400

/-- The number of segments already drawn -/
def segments_drawn : ℕ := 7

/-- Calculates the total number of segments in the spiral -/
def total_segments (n : ℕ) : ℕ := 2 * n + 1

theorem remaining_segments_theorem :
  ∃ n : ℕ, 
    spiral_pattern n = total_length ∧ 
    total_segments n - segments_drawn = 32 :=
sorry

end remaining_segments_theorem_l659_65947


namespace largest_n_divisibility_l659_65968

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1098 → ¬(n + 11 ∣ n^3 + 101) ∧ (1098 + 11 ∣ 1098^3 + 101) :=
by sorry

end largest_n_divisibility_l659_65968


namespace geometric_progression_p_l659_65909

theorem geometric_progression_p (p : ℝ) : 
  p > 0 ∧ 
  (3 * Real.sqrt p) ^ 2 = (-p - 8) * (p - 7) ↔ 
  p = 4 := by
sorry

end geometric_progression_p_l659_65909


namespace subtracted_number_l659_65989

theorem subtracted_number (x : ℝ) : x = 7 → 4 * 5.0 - x = 13 := by
  sorry

end subtracted_number_l659_65989


namespace tourism_revenue_scientific_notation_l659_65978

/-- Represents the tourism revenue in yuan -/
def tourism_revenue : ℝ := 12.41e9

/-- Represents the scientific notation of the tourism revenue -/
def scientific_notation : ℝ := 1.241e9

/-- Theorem stating that the tourism revenue is equal to its scientific notation representation -/
theorem tourism_revenue_scientific_notation : tourism_revenue = scientific_notation := by
  sorry

end tourism_revenue_scientific_notation_l659_65978


namespace hyperbola_asymptotes_l659_65924

/-- A hyperbola with equation x^2/4 - y^2/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/16 = 1

/-- Asymptotic lines with equations y = ±2x -/
def asymptotic_lines (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola equation implies the asymptotic lines,
    but the asymptotic lines do not necessarily imply the specific hyperbola equation -/
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptotic_lines x y) ∧
  ¬(∀ x y : ℝ, asymptotic_lines x y → hyperbola x y) :=
sorry

end hyperbola_asymptotes_l659_65924


namespace well_depth_is_784_l659_65944

/-- The depth of the well in feet -/
def well_depth : ℝ := 784

/-- The total time for the stone to fall and the sound to return, in seconds -/
def total_time : ℝ := 7.7

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1120

/-- The function describing the distance fallen by the stone in t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

/-- Theorem stating that the well depth is 784 feet given the conditions -/
theorem well_depth_is_784 :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧ 
    stone_fall t_fall = well_depth ∧
    t_fall + well_depth / sound_velocity = total_time :=
sorry

end well_depth_is_784_l659_65944


namespace hyperbola_line_intersection_l659_65942

/-- Hyperbola equation: x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- Line equation: y = k(x - 1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The line intersects the hyperbola at two points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  k ∈ Set.Ioo (-(2 * Real.sqrt 3 / 3)) (-1) ∪ 
      Set.Ioo (-1) 1 ∪ 
      Set.Ioo 1 (2 * Real.sqrt 3 / 3)

/-- The line intersects the hyperbola at exactly one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -(2 * Real.sqrt 3 / 3)

theorem hyperbola_line_intersection :
  (∀ k : ℝ, intersects_at_two_points k ↔ 
    ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
      hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂) ∧
  (∀ k : ℝ, intersects_at_one_point k ↔ 
    (∃ x y : ℝ, hyperbola x y ∧ line k x y) ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end hyperbola_line_intersection_l659_65942


namespace diaz_future_age_l659_65994

/-- Proves Diaz's age 20 years from now given the conditions in the problem -/
theorem diaz_future_age (sierra_age : ℕ) (diaz_age : ℕ) : 
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 := by
  sorry

end diaz_future_age_l659_65994


namespace fraction_equality_l659_65936

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end fraction_equality_l659_65936


namespace circus_performance_time_l659_65958

/-- Represents the time each entertainer stands on their back legs -/
structure CircusTime where
  pulsar : ℝ
  polly : ℝ
  petra : ℝ
  penny : ℝ
  parker : ℝ

/-- Calculates the total time all entertainers stand on their back legs -/
def totalTime (ct : CircusTime) : ℝ :=
  ct.pulsar + ct.polly + ct.petra + ct.penny + ct.parker

/-- Theorem stating the conditions and the result to be proved -/
theorem circus_performance_time :
  ∀ (ct : CircusTime),
    ct.pulsar = 10 →
    ct.polly = 3 * ct.pulsar →
    ct.petra = ct.polly / 6 →
    ct.penny = 2 * (ct.pulsar + ct.polly + ct.petra) →
    ct.parker = (ct.pulsar + ct.polly + ct.petra + ct.penny) / 4 →
    totalTime ct = 168.75 := by
  sorry


end circus_performance_time_l659_65958


namespace olympic_medal_theorem_l659_65929

/-- Represents the number of ways to award medals in the Olympic 100-meter finals -/
def olympic_medal_ways (total_sprinters : ℕ) (british_sprinters : ℕ) (medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_theorem :
  let total_sprinters := 10
  let british_sprinters := 4
  let medals := 3
  olympic_medal_ways total_sprinters british_sprinters medals = 912 :=
by
  sorry

end olympic_medal_theorem_l659_65929


namespace option_b_more_favorable_example_option_b_more_favorable_l659_65993

/-- Represents the financial data for a business --/
structure FinancialData where
  planned_revenue : ℕ
  advances_received : ℕ
  monthly_expenses : ℕ

/-- Calculates the tax payable under option (a) --/
def tax_option_a (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let tax := total_income * 6 / 100
  let insurance_contributions := data.monthly_expenses * 12
  let deduction := min (tax / 2) insurance_contributions
  tax - deduction

/-- Calculates the tax payable under option (b) --/
def tax_option_b (data : FinancialData) : ℕ :=
  let total_income := data.planned_revenue + data.advances_received
  let annual_expenses := data.monthly_expenses * 12
  let tax_base := max 0 (total_income - annual_expenses)
  let tax := max (total_income / 100) (tax_base * 15 / 100)
  tax

/-- Theorem stating that option (b) results in lower tax --/
theorem option_b_more_favorable (data : FinancialData) :
  tax_option_b data < tax_option_a data :=
by sorry

/-- Example financial data --/
def example_data : FinancialData :=
  { planned_revenue := 120000000
  , advances_received := 30000000
  , monthly_expenses := 11790000 }

/-- Proof that option (b) is more favorable for the example data --/
theorem example_option_b_more_favorable :
  tax_option_b example_data < tax_option_a example_data :=
by sorry

end option_b_more_favorable_example_option_b_more_favorable_l659_65993


namespace no_counterexamples_l659_65967

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

theorem no_counterexamples :
  ¬ ∃ N : ℕ, 
    (sum_of_digits N = 5) ∧ 
    (has_no_zero_digit N) ∧ 
    (Nat.Prime N) ∧ 
    (N % 5 = 0) := by
  sorry

end no_counterexamples_l659_65967


namespace ellipse_area_theorem_l659_65928

/-- Represents an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  a_gt_b : a > b
  b_pos : b > 0
  vertex_y : b = 1
  eccentricity : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2

/-- Represents a line passing through the right focus of the ellipse -/
structure FocusLine (e : Ellipse) where
  k : ℝ  -- Slope of the line

/-- Represents two points on the ellipse intersected by the focus line -/
structure IntersectionPoints (e : Ellipse) (l : FocusLine e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_ellipse_A : A.1^2 / e.a^2 + A.2^2 / e.b^2 = 1
  on_ellipse_B : B.1^2 / e.a^2 + B.2^2 / e.b^2 = 1
  on_line_A : A.2 = l.k * (A.1 - Real.sqrt (e.a^2 - e.b^2))
  on_line_B : B.2 = l.k * (B.1 - Real.sqrt (e.a^2 - e.b^2))
  perpendicular : A.1 * B.1 + A.2 * B.2 = 0  -- OA ⊥ OB condition

/-- Main theorem statement -/
theorem ellipse_area_theorem (e : Ellipse) (l : FocusLine e) (p : IntersectionPoints e l) :
  e.a^2 = 2 ∧ 
  (abs (p.A.1 - p.B.1) * abs (p.A.2 - p.B.2) / 2 = 2 * Real.sqrt 3 / 5) := by
  sorry

end ellipse_area_theorem_l659_65928


namespace sufficient_but_not_necessary_l659_65908

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
  sorry

end sufficient_but_not_necessary_l659_65908


namespace fourth_root_of_207360000_l659_65992

theorem fourth_root_of_207360000 : (207360000 : ℝ) ^ (1/4 : ℝ) = 120 := by
  sorry

end fourth_root_of_207360000_l659_65992


namespace complement_A_in_U_l659_65918

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end complement_A_in_U_l659_65918


namespace age_problem_l659_65955

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end age_problem_l659_65955


namespace multiples_properties_l659_65965

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 4 * m) : 
  (∃ n : ℤ, b = 2 * n) ∧ (∃ p : ℤ, a - b = 5 * p) := by
  sorry

end multiples_properties_l659_65965


namespace mike_shortfall_l659_65970

def max_marks : ℕ := 800
def pass_percentage : ℚ := 30 / 100
def mike_score : ℕ := 212

theorem mike_shortfall :
  (↑max_marks * pass_percentage).floor - mike_score = 28 :=
sorry

end mike_shortfall_l659_65970


namespace forgotten_angle_measure_l659_65927

/-- The sum of interior angles of a polygon with n sides --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The sum of all but one interior angle of the polygon --/
def partial_sum : ℝ := 2017

/-- The measure of the forgotten angle --/
def forgotten_angle : ℝ := 143

theorem forgotten_angle_measure :
  ∃ (n : ℕ), n > 3 ∧ sum_interior_angles n = partial_sum + forgotten_angle :=
sorry

end forgotten_angle_measure_l659_65927


namespace fraction_power_cube_l659_65935

theorem fraction_power_cube : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by sorry

end fraction_power_cube_l659_65935


namespace max_value_on_circle_l659_65977

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 25 →
  ∃ (t_max : ℝ), t_max = 6 * Real.sqrt 10 ∧
  ∀ t, t = Real.sqrt (18 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) →
  t ≤ t_max :=
by sorry

end max_value_on_circle_l659_65977


namespace vaishali_saree_stripes_l659_65981

theorem vaishali_saree_stripes :
  ∀ (brown gold blue : ℕ),
    gold = 3 * brown →
    blue = 5 * gold →
    blue = 60 →
    brown = 4 :=
by
  sorry

end vaishali_saree_stripes_l659_65981


namespace average_cost_before_gratuity_l659_65986

theorem average_cost_before_gratuity
  (num_individuals : ℕ)
  (total_bill_with_gratuity : ℚ)
  (gratuity_rate : ℚ)
  (h1 : num_individuals = 9)
  (h2 : total_bill_with_gratuity = 756)
  (h3 : gratuity_rate = 1/5) :
  (total_bill_with_gratuity / (1 + gratuity_rate)) / num_individuals = 70 := by
sorry

end average_cost_before_gratuity_l659_65986


namespace min_xy_value_l659_65906

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1)⁻¹ + (y + 1)⁻¹ = (1 : ℝ) / 2) : 
  ∀ z, z = x * y → z ≥ 9 :=
by sorry

end min_xy_value_l659_65906


namespace sqrt_221_between_15_and_16_l659_65996

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end sqrt_221_between_15_and_16_l659_65996


namespace zero_point_of_odd_function_l659_65953

/-- A function f is odd if f(-x) = -f(x) for all x. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem zero_point_of_odd_function (f : ℝ → ℝ) (x₀ : ℝ) :
  IsOdd f →
  f x₀ + Real.exp x₀ = 0 →
  Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end zero_point_of_odd_function_l659_65953


namespace square_land_side_length_l659_65939

theorem square_land_side_length (area : ℝ) (is_square : Bool) : 
  area = 400 ∧ is_square = true → ∃ (side : ℝ), side * side = area ∧ side = 20 := by
  sorry

end square_land_side_length_l659_65939


namespace min_unsuccessful_placements_l659_65974

/-- A board is represented as a function from (Fin 8 × Fin 8) to Int -/
def Board := Fin 8 → Fin 8 → Int

/-- A tetromino is represented as a list of four pairs of coordinates -/
def Tetromino := List (Fin 8 × Fin 8)

/-- A valid board has only 1 and -1 as values -/
def validBoard (b : Board) : Prop :=
  ∀ i j, b i j = 1 ∨ b i j = -1

/-- A valid tetromino has four distinct cells within the board -/
def validTetromino (t : Tetromino) : Prop :=
  t.length = 4 ∧ t.Nodup

/-- The sum of a tetromino's cells on a board -/
def tetrominoSum (b : Board) (t : Tetromino) : Int :=
  t.foldl (fun sum (i, j) => sum + b i j) 0

/-- An unsuccessful placement has a non-zero sum -/
def unsuccessfulPlacement (b : Board) (t : Tetromino) : Prop :=
  tetrominoSum b t ≠ 0

/-- The main theorem -/
theorem min_unsuccessful_placements (b : Board) (h : validBoard b) :
  ∃ (unsuccessfulPlacements : List Tetromino),
    unsuccessfulPlacements.length ≥ 36 ∧
    ∀ t ∈ unsuccessfulPlacements, validTetromino t ∧ unsuccessfulPlacement b t :=
  sorry

end min_unsuccessful_placements_l659_65974


namespace petyas_class_girls_count_l659_65934

theorem petyas_class_girls_count :
  ∀ (x y : ℕ),
  x + y ≤ 40 →
  (2 : ℚ) / 3 * x + (1 : ℚ) / 7 * y = (1 : ℚ) / 3 * (x + y) →
  x = 12 :=
λ x y h1 h2 => by
  sorry

end petyas_class_girls_count_l659_65934


namespace hay_consumption_time_l659_65949

/-- The number of weeks it takes for a group of animals to eat a given amount of hay -/
def time_to_eat_hay (goat_rate sheep_rate cow_rate : ℚ) (num_goats num_sheep num_cows : ℕ) (total_hay : ℚ) : ℚ :=
  total_hay / (goat_rate * num_goats + sheep_rate * num_sheep + cow_rate * num_cows)

/-- Theorem: Given the rates of hay consumption and number of animals, it takes 16 weeks to eat 30 cartloads of hay -/
theorem hay_consumption_time :
  let goat_rate : ℚ := 1 / 6
  let sheep_rate : ℚ := 1 / 8
  let cow_rate : ℚ := 1 / 3
  let num_goats : ℕ := 5
  let num_sheep : ℕ := 3
  let num_cows : ℕ := 2
  let total_hay : ℚ := 30
  time_to_eat_hay goat_rate sheep_rate cow_rate num_goats num_sheep num_cows total_hay = 16 := by
  sorry


end hay_consumption_time_l659_65949


namespace cube_sum_of_three_numbers_l659_65995

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x * y + x * z + y * z = 1)
  (prod_eq : x * y * z = 1) :
  x^3 + y^3 + z^3 = 1 := by
  sorry

end cube_sum_of_three_numbers_l659_65995


namespace x₄_x₁_diff_l659_65988

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = -f (200 - x)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- The x-intercepts are in increasing order
axiom x_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

-- The difference between x₃ and x₂
axiom x₃_x₂_diff : x₃ - x₂ = 200

-- The vertex of g is on the graph of f
axiom vertex_on_f : ∃ x, g x = f x ∧ ∀ y, g y ≤ g x

-- Theorem to prove
theorem x₄_x₁_diff : x₄ - x₁ = 1000 + 800 * Real.sqrt 3 := by sorry

end x₄_x₁_diff_l659_65988


namespace pencil_price_solution_l659_65922

def pencil_price_problem (pencil_price notebook_price : ℕ) : Prop :=
  (pencil_price + notebook_price = 950) ∧ 
  (notebook_price = pencil_price + 150)

theorem pencil_price_solution : 
  ∃ (pencil_price notebook_price : ℕ), 
    pencil_price_problem pencil_price notebook_price ∧ 
    pencil_price = 400 := by
  sorry

end pencil_price_solution_l659_65922


namespace select_three_from_seven_eq_210_l659_65951

/-- The number of ways to select 3 distinct individuals from a group of 7 people to fill 3 distinct positions. -/
def select_three_from_seven : ℕ :=
  7 * 6 * 5

/-- Theorem stating that selecting 3 distinct individuals from a group of 7 people to fill 3 distinct positions can be done in 210 ways. -/
theorem select_three_from_seven_eq_210 :
  select_three_from_seven = 210 := by
  sorry

end select_three_from_seven_eq_210_l659_65951


namespace triangle_table_height_l659_65921

theorem triangle_table_height (a b c : ℝ) (h_a : a = 25) (h_b : b = 31) (h_c : c = 34) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_max := 2 * area / (a + b + c)
  h_max = 4 * Real.sqrt 231 / 3 := by sorry

end triangle_table_height_l659_65921


namespace expr_is_monomial_l659_65964

-- Define what a monomial is
def is_monomial (expr : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (n : ℕ), ∀ x, expr x = a * x^n

-- Define the expression y/2023
def expr (y : ℚ) : ℚ := y / 2023

-- Theorem statement
theorem expr_is_monomial : is_monomial expr :=
sorry

end expr_is_monomial_l659_65964


namespace rationalize_denominator_l659_65923

theorem rationalize_denominator : (5 : ℝ) / Real.sqrt 125 = Real.sqrt 5 / 5 := by sorry

end rationalize_denominator_l659_65923


namespace second_prize_proportion_l659_65998

theorem second_prize_proportion (total winners : ℕ) 
  (first second third : ℕ) 
  (h1 : first + second + third = winners)
  (h2 : (first + second : ℚ) / winners = 3 / 4)
  (h3 : (second + third : ℚ) / winners = 2 / 3) :
  (second : ℚ) / winners = 5 / 12 := by
  sorry

end second_prize_proportion_l659_65998


namespace simplify_trig_expression_l659_65916

theorem simplify_trig_expression :
  let θ : Real := 160 * π / 180  -- Convert 160° to radians
  (θ > π / 2) ∧ (θ < π) →  -- 160° is in the second quadrant
  1 / Real.sqrt (1 + Real.tan θ ^ 2) = -Real.cos θ := by
  sorry

end simplify_trig_expression_l659_65916


namespace bisection_method_root_location_l659_65954

def f (x : ℝ) := x^3 - 6*x^2 + 4

theorem bisection_method_root_location :
  (∃ r ∈ Set.Ioo 0 1, f r = 0) →
  (f 0 > 0) →
  (f 1 < 0) →
  (f (1/2) > 0) →
  ∃ r ∈ Set.Ioo (1/2) 1, f r = 0 := by sorry

end bisection_method_root_location_l659_65954


namespace tangent_line_parallel_l659_65952

/-- Given a function f(x) = ln x - ax, if its derivative at x = 1 is -2, then a = 3 -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x - a * x
  (deriv f 1 = -2) → a = 3 := by
sorry

end tangent_line_parallel_l659_65952


namespace inequality_proof_l659_65941

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb1 : 1 > b) (hb2 : b > -1) :
  a > b^2 := by
  sorry

end inequality_proof_l659_65941


namespace xy_squared_minus_x_squared_y_l659_65943

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 3) : 
  x * y^2 - x^2 * y = -6 := by
  sorry

end xy_squared_minus_x_squared_y_l659_65943


namespace sqrt_expression_equals_sqrt_three_l659_65971

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end sqrt_expression_equals_sqrt_three_l659_65971


namespace solve_water_problem_l659_65969

def water_problem (initial_water evaporated_water rain_duration rain_rate final_water : ℝ) : Prop :=
  let water_after_evaporation := initial_water - evaporated_water
  let rainwater_added := (rain_duration / 10) * rain_rate
  let water_after_rain := water_after_evaporation + rainwater_added
  let water_drained := water_after_rain - final_water
  water_drained = 3500

theorem solve_water_problem :
  water_problem 6000 2000 30 350 1550 := by
  sorry

end solve_water_problem_l659_65969


namespace linear_function_not_in_fourth_quadrant_l659_65972

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  sorry

/-- The specific linear function y = 2x + 1 -/
def f : LinearFunction :=
  { slope := 2, yIntercept := 1 }

theorem linear_function_not_in_fourth_quadrant :
  ¬ passesThrough f Quadrant.fourth :=
sorry

end linear_function_not_in_fourth_quadrant_l659_65972


namespace least_n_with_zero_in_factorization_l659_65983

/-- A function that checks if a positive integer contains the digit 0 -/
def containsZero (n : ℕ+) : Prop :=
  ∃ (k : ℕ), n.val = 10 * k ∨ n.val % 10 = 0

/-- A function that checks if all factorizations of 10^n contain a zero -/
def allFactorizationsContainZero (n : ℕ) : Prop :=
  ∀ (a b : ℕ+), a * b = 10^n → (containsZero a ∨ containsZero b)

/-- The main theorem stating that 8 is the least positive integer satisfying the condition -/
theorem least_n_with_zero_in_factorization :
  (allFactorizationsContainZero 8) ∧
  (∀ m : ℕ, m < 8 → ¬(allFactorizationsContainZero m)) :=
sorry

end least_n_with_zero_in_factorization_l659_65983


namespace additional_students_score_l659_65919

/-- Given a class with the following properties:
  * There are 17 students in total
  * The average grade of 15 students is 85
  * After including two more students, the new average becomes 87
  This theorem proves that the combined score of the two additional students is 204. -/
theorem additional_students_score (total_students : ℕ) (initial_students : ℕ) 
  (initial_average : ℝ) (final_average : ℝ) : 
  total_students = 17 → 
  initial_students = 15 → 
  initial_average = 85 → 
  final_average = 87 → 
  (total_students * final_average - initial_students * initial_average : ℝ) = 204 := by
  sorry

#check additional_students_score

end additional_students_score_l659_65919


namespace negation_of_proposition_l659_65990

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by
  sorry

end negation_of_proposition_l659_65990


namespace cubes_fill_box_completely_l659_65930

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  (cubesAlongDimension d.length cubeSize) *
  (cubesAlongDimension d.width cubeSize) *
  (cubesAlongDimension d.height cubeSize)

/-- Calculates the volume occupied by the cubes -/
def cubesVolume (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  totalCubes d cubeSize * (cubeSize ^ 3)

/-- Theorem: The volume occupied by 4-inch cubes in the given box is 100% of the box's volume -/
theorem cubes_fill_box_completely (d : BoxDimensions) (h1 : d.length = 16) (h2 : d.width = 12) (h3 : d.height = 8) :
  cubesVolume d 4 = boxVolume d := by
  sorry

#eval cubesVolume ⟨16, 12, 8⟩ 4
#eval boxVolume ⟨16, 12, 8⟩

end cubes_fill_box_completely_l659_65930


namespace fill_cistern_time_cistern_filling_problem_l659_65933

/-- The time taken for two pipes to fill a cistern together -/
theorem fill_cistern_time (time_A time_B : ℝ) (h1 : time_A > 0) (h2 : time_B > 0) :
  let combined_rate := 1 / time_A + 1 / time_B
  combined_rate⁻¹ = (time_A * time_B) / (time_A + time_B) := by sorry

/-- Proof of the cistern filling problem -/
theorem cistern_filling_problem :
  let time_A : ℝ := 36  -- Time for Pipe A to fill the entire cistern
  let time_B : ℝ := 24  -- Time for Pipe B to fill the entire cistern
  let combined_time := (time_A * time_B) / (time_A + time_B)
  combined_time = 14.4 := by sorry

end fill_cistern_time_cistern_filling_problem_l659_65933


namespace calculate_expression_l659_65938

theorem calculate_expression : 5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end calculate_expression_l659_65938


namespace total_loaves_is_nine_l659_65905

/-- The number of bags of bread -/
def num_bags : ℕ := 3

/-- The number of loaves in each bag -/
def loaves_per_bag : ℕ := 3

/-- The total number of loaves of bread -/
def total_loaves : ℕ := num_bags * loaves_per_bag

theorem total_loaves_is_nine : total_loaves = 9 := by
  sorry

end total_loaves_is_nine_l659_65905


namespace square_root_problem_l659_65985

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21) / (Real.sqrt x) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 3.0892857142857144 →
  x = 0.64 := by
  sorry

end square_root_problem_l659_65985


namespace product_unit_digit_l659_65900

def unit_digit (n : ℕ) : ℕ := n % 10

theorem product_unit_digit : 
  unit_digit (624 * 708 * 913 * 463) = 8 := by
  sorry

end product_unit_digit_l659_65900


namespace triangle_properties_l659_65926

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 6 ∧ t.A = 2 * Real.pi / 3 ∧
  ((t.B = Real.pi / 4) ∨ (t.a = 3))

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) :
  t.c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 ∧
  (1 / 2 * t.b * t.c * Real.sin t.A) = (9 - 3 * Real.sqrt 3) / 4 := by
  sorry

end triangle_properties_l659_65926


namespace clock_hands_right_angle_period_l659_65975

/-- The number of times clock hands are at right angles in 12 hours -/
def right_angles_per_12_hours : ℕ := 22

/-- The number of times clock hands are at right angles in the given period -/
def given_right_angles : ℕ := 88

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

theorem clock_hands_right_angle_period :
  (given_right_angles / right_angles_per_12_hours) * 12 = hours_per_day :=
sorry

end clock_hands_right_angle_period_l659_65975


namespace equation_solutions_l659_65912

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, y₁ = 3 + 2 * Real.sqrt 2 ∧ y₂ = 3 - 2 * Real.sqrt 2 ∧
    ∀ y : ℝ, y^2 - 6*y + 1 = 0 ↔ (y = y₁ ∨ y = y₂)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 12 ∧
    ∀ x : ℝ, 2*(x-4)^2 = x^2 - 16 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solutions_l659_65912


namespace peter_money_carried_l659_65956

/-- The amount of money Peter carried to the market -/
def money_carried : ℝ := sorry

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

/-- The quantity of potatoes bought in kilos -/
def potato_quantity : ℝ := 6

/-- The price of tomatoes per kilo -/
def tomato_price : ℝ := 3

/-- The quantity of tomatoes bought in kilos -/
def tomato_quantity : ℝ := 9

/-- The price of cucumbers per kilo -/
def cucumber_price : ℝ := 4

/-- The quantity of cucumbers bought in kilos -/
def cucumber_quantity : ℝ := 5

/-- The price of bananas per kilo -/
def banana_price : ℝ := 5

/-- The quantity of bananas bought in kilos -/
def banana_quantity : ℝ := 3

/-- The amount of money Peter has remaining after buying all items -/
def money_remaining : ℝ := 426

theorem peter_money_carried :
  money_carried = 
    potato_price * potato_quantity +
    tomato_price * tomato_quantity +
    cucumber_price * cucumber_quantity +
    banana_price * banana_quantity +
    money_remaining :=
by sorry

end peter_money_carried_l659_65956


namespace max_marks_proof_l659_65997

/-- Given a passing threshold, actual score, and shortfall, calculates the maximum possible marks -/
def calculate_max_marks (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) : ℚ :=
  (actual_score + shortfall : ℚ) / passing_threshold

/-- Proves that the maximum marks is 617.5 given the problem conditions -/
theorem max_marks_proof (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) 
    (h1 : passing_threshold = 0.4)
    (h2 : actual_score = 212)
    (h3 : shortfall = 35) :
  calculate_max_marks passing_threshold actual_score shortfall = 617.5 := by
  sorry

#eval calculate_max_marks 0.4 212 35

end max_marks_proof_l659_65997


namespace ellipse_right_triangle_distance_to_x_axis_l659_65901

/-- An ellipse with semi-major axis 4 and semi-minor axis 3 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2 / 16 + y^2 / 9 = 1)

/-- The foci of the ellipse -/
def foci (e : Ellipse) : ℝ × ℝ := sorry

/-- A point P on the ellipse forms a right triangle with the foci -/
def right_triangle_with_foci (e : Ellipse) (p : ℝ × ℝ) : Prop := sorry

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_triangle_distance_to_x_axis (e : Ellipse) (p : ℝ × ℝ) :
  p.1^2 / 16 + p.2^2 / 9 = 1 →
  right_triangle_with_foci e p →
  distance_to_x_axis p = 9/4 := by sorry

end ellipse_right_triangle_distance_to_x_axis_l659_65901


namespace product_of_difference_and_sum_is_zero_l659_65917

theorem product_of_difference_and_sum_is_zero (a : ℝ) (x y : ℝ) 
  (h1 : x = a + 5)
  (h2 : a = 20)
  (h3 : y = 25) :
  (x - y) * (x + y) = 0 := by
sorry

end product_of_difference_and_sum_is_zero_l659_65917


namespace function_decomposition_into_symmetric_parts_l659_65961

/-- A function is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- A function is symmetric about the vertical line x = a if f(x) = f(2a - x) for all x ∈ ℝ -/
def SymmetricAboutVerticalLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * a - x)

/-- Main theorem: Any function on ℝ can be represented as the sum of two symmetric functions -/
theorem function_decomposition_into_symmetric_parts (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ),
    (∀ x : ℝ, f x = f₁ x + f₂ x) ∧
    SymmetricAboutYAxis f₁ ∧
    a > 0 ∧
    SymmetricAboutVerticalLine f₂ a :=
  sorry

end function_decomposition_into_symmetric_parts_l659_65961


namespace manager_chef_wage_difference_l659_65940

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 6.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.20

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : validSteakhouseWages w) : w.manager - w.chef = 2.60 := by
  sorry

end manager_chef_wage_difference_l659_65940


namespace square_area_ratio_l659_65976

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((4*y)^2) = 1/16 := by
sorry

end square_area_ratio_l659_65976
