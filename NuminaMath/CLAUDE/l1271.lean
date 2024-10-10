import Mathlib

namespace midsphere_radius_is_geometric_mean_l1271_127121

/-- A regular tetrahedron with its associated spheres -/
structure RegularTetrahedron where
  /-- The radius of the insphere (inscribed sphere) -/
  r_in : ℝ
  /-- The radius of the circumsphere (circumscribed sphere) -/
  r_out : ℝ
  /-- The radius of the midsphere (edge-touching sphere) -/
  r_mid : ℝ
  /-- The radii are positive -/
  h_positive : r_in > 0 ∧ r_out > 0 ∧ r_mid > 0

/-- The radius of the midsphere is the geometric mean of the radii of the insphere and circumsphere -/
theorem midsphere_radius_is_geometric_mean (t : RegularTetrahedron) :
  t.r_mid ^ 2 = t.r_in * t.r_out := by
  sorry

end midsphere_radius_is_geometric_mean_l1271_127121


namespace gcf_lcm_sum_l1271_127102

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 9 (Nat.gcd 15 27)) →
  (B = Nat.lcm 9 (Nat.lcm 15 27)) →
  A + B = 138 := by
sorry

end gcf_lcm_sum_l1271_127102


namespace set_equality_l1271_127171

def set_a : Set ℕ := {x : ℕ | 2 * x + 3 ≥ 3 * x}
def set_b : Set ℕ := {0, 1, 2, 3}

theorem set_equality : set_a = set_b := by
  sorry

end set_equality_l1271_127171


namespace yellow_square_ratio_l1271_127143

/-- Represents a square banner with a symmetric cross -/
structure Banner where
  side : ℝ
  cross_area_ratio : ℝ
  yellow_area_ratio : ℝ

/-- The banner satisfies the problem conditions -/
def valid_banner (b : Banner) : Prop :=
  b.side > 0 ∧
  b.cross_area_ratio = 0.25 ∧
  b.yellow_area_ratio > 0 ∧
  b.yellow_area_ratio < b.cross_area_ratio

theorem yellow_square_ratio (b : Banner) (h : valid_banner b) :
  b.yellow_area_ratio = 0.01 := by
  sorry

end yellow_square_ratio_l1271_127143


namespace modular_congruence_l1271_127198

theorem modular_congruence (x : ℤ) :
  (5 * x + 9) % 16 = 3 → (3 * x + 8) % 16 = 14 := by
sorry

end modular_congruence_l1271_127198


namespace same_row_exists_l1271_127131

/-- Represents a seating arrangement for a class session -/
def SeatingArrangement := Fin 50 → Fin 7

theorem same_row_exists (morning afternoon : SeatingArrangement) : 
  ∃ (s1 s2 : Fin 50), s1 ≠ s2 ∧ morning s1 = morning s2 ∧ afternoon s1 = afternoon s2 := by
  sorry

end same_row_exists_l1271_127131


namespace may_red_yarns_l1271_127150

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

theorem may_red_yarns : 
  scarves_per_yarn * (blue_yarns + yellow_yarns + red_yarns) = total_scarves := by
  sorry

end may_red_yarns_l1271_127150


namespace david_pushups_count_l1271_127110

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 3 * zachary_pushups

theorem david_pushups_count : david_pushups = 57 := by
  sorry

end david_pushups_count_l1271_127110


namespace drinks_preparation_l1271_127192

/-- Given a number of pitchers and the capacity of each pitcher in glasses,
    calculate the total number of glasses that can be filled. -/
def total_glasses (num_pitchers : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  num_pitchers * glasses_per_pitcher

/-- Theorem stating that 9 pitchers, each filling 6 glasses, results in 54 glasses total. -/
theorem drinks_preparation :
  total_glasses 9 6 = 54 := by
  sorry

end drinks_preparation_l1271_127192


namespace scaling_transformation_result_l1271_127167

/-- A scaling transformation in a 2D plane -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Apply a scaling transformation to a point -/
def applyTransformation (t : ScalingTransformation) (p : Point) : Point :=
  { x := t.x_scale * p.x,
    y := t.y_scale * p.y }

theorem scaling_transformation_result :
  let A : Point := { x := 1/3, y := -2 }
  let φ : ScalingTransformation := { x_scale := 3, y_scale := 1/2 }
  let A' : Point := applyTransformation φ A
  A'.x = 1 ∧ A'.y = -1 := by sorry

end scaling_transformation_result_l1271_127167


namespace min_value_of_expression_l1271_127123

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_zeros : x₁^2 - 4*a*x₁ + a^2 = 0 ∧ x₂^2 - 4*a*x₂ + a^2 = 0) :
  x₁ + x₂ + a / (x₁ * x₂) ≥ 4 := by
  sorry

end min_value_of_expression_l1271_127123


namespace only_valid_quadruples_l1271_127138

/-- A quadruple of non-negative integers satisfying the given conditions -/
structure ValidQuadruple where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  eq : a * b = 2 * (1 + c * d)
  triangle : (a - c) + (b - d) > c + d ∧ 
             (a - c) + (c + d) > b - d ∧ 
             (b - d) + (c + d) > a - c

/-- The theorem stating that only two specific quadruples satisfy the conditions -/
theorem only_valid_quadruples : 
  ∀ q : ValidQuadruple, (q.a = 1 ∧ q.b = 2 ∧ q.c = 0 ∧ q.d = 1) ∨ 
                        (q.a = 2 ∧ q.b = 1 ∧ q.c = 1 ∧ q.d = 0) := by
  sorry

end only_valid_quadruples_l1271_127138


namespace total_cost_calculation_l1271_127106

/-- The total cost of buying jerseys and basketballs -/
def total_cost (m n : ℝ) : ℝ := 8 * m + 5 * n

/-- Theorem: The total cost of buying 8 jerseys at m yuan each and 5 basketballs at n yuan each is 8m + 5n yuan -/
theorem total_cost_calculation (m n : ℝ) : 
  total_cost m n = 8 * m + 5 * n := by sorry

end total_cost_calculation_l1271_127106


namespace quadratic_as_binomial_square_l1271_127169

theorem quadratic_as_binomial_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end quadratic_as_binomial_square_l1271_127169


namespace solve_motel_problem_l1271_127105

def motel_problem (higher_rate : ℕ) : Prop :=
  ∃ (num_higher_rate : ℕ) (num_lower_rate : ℕ),
    -- There are two types of room rates: $40 and a higher amount
    higher_rate > 40 ∧
    -- The actual total rent charged was $1000
    num_higher_rate * higher_rate + num_lower_rate * 40 = 1000 ∧
    -- If 10 rooms at the higher rate were rented for $40 instead, the total rent would be reduced by 20%
    (num_higher_rate - 10) * higher_rate + (num_lower_rate + 10) * 40 = 800

theorem solve_motel_problem : 
  ∃ (higher_rate : ℕ), motel_problem higher_rate ∧ higher_rate = 60 :=
sorry

end solve_motel_problem_l1271_127105


namespace computer_price_ratio_l1271_127127

theorem computer_price_ratio (x : ℝ) (h : 1.3 * x = 351) :
  (x + 1.3 * x) / x = 2.3 := by
  sorry

end computer_price_ratio_l1271_127127


namespace arithmetic_calculations_l1271_127179

theorem arithmetic_calculations :
  (5 + (-6) + 3 - (-4) = 6) ∧
  (-1^2024 - (2 - (-2)^3) / (-2/5) * 5/2 = 123/2) := by
  sorry

end arithmetic_calculations_l1271_127179


namespace tens_digit_of_five_pow_2023_l1271_127174

theorem tens_digit_of_five_pow_2023 : (5^2023 / 10) % 10 = 2 := by
  sorry

end tens_digit_of_five_pow_2023_l1271_127174


namespace perpendicular_vectors_m_value_l1271_127145

/-- Given two vectors OA and OB in R², prove that if they are perpendicular
    and OA = (-1, 2) and OB = (3, m), then m = 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) (m : ℝ) :
  OA = (-1, 2) →
  OB = (3, m) →
  OA.1 * OB.1 + OA.2 * OB.2 = 0 →
  m = 3/2 := by
sorry

end perpendicular_vectors_m_value_l1271_127145


namespace jeremy_cannot_be_sure_l1271_127160

theorem jeremy_cannot_be_sure (n : ℕ) : ∃ (remaining_permutations : ℝ), 
  remaining_permutations > 1 ∧ 
  remaining_permutations = (2^n).factorial / 2^(n * 2^(n-1)) := by
  sorry

#check jeremy_cannot_be_sure

end jeremy_cannot_be_sure_l1271_127160


namespace hyperbola_eccentricity_l1271_127129

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (F P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x < -a ∨ x > a)) →
  (∀ (x y : ℝ), x^2 + y^2 = b^2 / 4 → ((P.1 - x) * (F.1 - P.1) + (P.2 - y) * (F.2 - P.2) = 0)) →
  Q.1^2 + Q.2^2 = b^2 / 4 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = ((P.1 - F.1)^2 + (P.2 - F.2)^2) / 4 →
  c^2 / a^2 = 5 :=
by sorry

end hyperbola_eccentricity_l1271_127129


namespace eliana_steps_l1271_127196

/-- The number of steps Eliana walked on the first day before adding more steps -/
def initial_steps : ℕ := 200

/-- The number of additional steps Eliana walked on the first day -/
def additional_steps : ℕ := 300

/-- The number of extra steps Eliana walked on the third day compared to the second day -/
def extra_steps : ℕ := 100

/-- The total number of steps Eliana walked over the three days -/
def total_steps : ℕ := 2600

theorem eliana_steps :
  let first_day := initial_steps + additional_steps
  let second_day := 2 * first_day
  let third_day := second_day + extra_steps
  first_day + second_day + third_day = total_steps := by sorry

end eliana_steps_l1271_127196


namespace new_tires_cost_l1271_127136

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def total_spent : ℝ := 387.85

theorem new_tires_cost (new_tires_cost : ℝ) : 
  new_tires_cost = total_spent - (speakers_cost + cd_player_cost) :=
by sorry

end new_tires_cost_l1271_127136


namespace extended_line_segment_l1271_127185

/-- Given a line segment AB extended to points P and Q, prove the expressions for P and Q -/
theorem extended_line_segment (A B P Q : ℝ × ℝ) : 
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ 
    7 * (P.1 - B.1) = 2 * (P.1 - A.1) ∧
    7 * (P.2 - B.2) = 2 * (P.2 - A.2) ∧
    5 * (Q.1 - B.1) = (Q.1 - A.1) ∧
    5 * (Q.2 - B.2) = (Q.2 - A.2)) →
  (P = (-2/5 : ℝ) • A + (7/5 : ℝ) • B ∧
   Q = (-1/4 : ℝ) • A + (5/4 : ℝ) • B) := by
sorry

end extended_line_segment_l1271_127185


namespace job_duration_l1271_127120

theorem job_duration (daily_wage : ℕ) (daily_fine : ℕ) (total_earnings : ℕ) (absent_days : ℕ) :
  daily_wage = 10 →
  daily_fine = 2 →
  total_earnings = 216 →
  absent_days = 7 →
  ∃ (work_days : ℕ), work_days * daily_wage - absent_days * daily_fine = total_earnings ∧ work_days = 23 :=
by sorry

end job_duration_l1271_127120


namespace lansing_elementary_students_l1271_127149

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

/-- Theorem stating the total number of elementary students in Lansing -/
theorem lansing_elementary_students : total_students = 6175 := by
  sorry

end lansing_elementary_students_l1271_127149


namespace moon_radius_scientific_notation_l1271_127126

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is equal to its scientific notation representation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end moon_radius_scientific_notation_l1271_127126


namespace quadratic_transform_l1271_127140

theorem quadratic_transform (p q r : ℝ) :
  (∃ m l : ℝ, ∀ x : ℝ, px^2 + qx + r = 5*(x - 3)^2 + 15 ∧ 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) →
  (∃ m l : ℝ, ∀ x : ℝ, 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) :=
by sorry

end quadratic_transform_l1271_127140


namespace partial_fraction_decomposition_product_l1271_127125

theorem partial_fraction_decomposition_product (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 →
    (x^2 - 13) / ((x - 2) * (x + 2) * (x - 3)) =
    A / (x - 2) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 81 / 100 := by
sorry

end partial_fraction_decomposition_product_l1271_127125


namespace total_steps_rachel_l1271_127172

theorem total_steps_rachel (steps_up steps_down : ℕ) 
  (h1 : steps_up = 567) 
  (h2 : steps_down = 325) : 
  steps_up + steps_down = 892 := by
sorry

end total_steps_rachel_l1271_127172


namespace unique_solution_for_prime_power_equation_l1271_127152

theorem unique_solution_for_prime_power_equation :
  ∀ m p x : ℕ,
    Prime p →
    2^m * p^2 + 27 = x^3 →
    m = 1 ∧ p = 7 :=
by sorry

end unique_solution_for_prime_power_equation_l1271_127152


namespace quadratic_equations_common_root_l1271_127147

theorem quadratic_equations_common_root (a b c x : ℝ) 
  (h1 : a * c ≠ 0) (h2 : a ≠ c) 
  (hM : a * x^2 + b * x + c = 0) 
  (hN : c * x^2 + b * x + a = 0) : 
  x = 1 ∨ x = -1 :=
sorry

end quadratic_equations_common_root_l1271_127147


namespace consecutive_non_prime_powers_l1271_127104

theorem consecutive_non_prime_powers (r : ℕ) (hr : r > 0) :
  ∃ x : ℕ, ∀ i ∈ Finset.range r, ¬ ∃ (p : ℕ) (n : ℕ), Prime p ∧ x + i + 1 = p ^ n :=
sorry

end consecutive_non_prime_powers_l1271_127104


namespace chemical_reaction_results_l1271_127153

/-- Represents the chemical reaction between CaCO3 and HCl -/
structure ChemicalReaction where
  temperature : ℝ
  pressure : ℝ
  hcl_moles : ℝ
  cacl2_moles : ℝ
  co2_moles : ℝ
  h2o_moles : ℝ
  std_enthalpy_change : ℝ

/-- Calculates the amount of CaCO3 required and the change in enthalpy -/
def calculate_reaction_results (reaction : ChemicalReaction) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct results of the chemical reaction -/
theorem chemical_reaction_results :
  let reaction := ChemicalReaction.mk 25 1 4 2 2 2 (-178)
  let (caco3_grams, enthalpy_change) := calculate_reaction_results reaction
  caco3_grams = 200.18 ∧ enthalpy_change = -356 := by
  sorry

end chemical_reaction_results_l1271_127153


namespace twentieth_is_thursday_l1271_127117

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with the given condition -/
structure Month where
  dates : List Date
  threeSundaysOnEvenDates : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Sunday ∧ d2.dayOfWeek = DayOfWeek.Sunday ∧ d3.dayOfWeek = DayOfWeek.Sunday ∧
    d1.day % 2 = 0 ∧ d2.day % 2 = 0 ∧ d3.day % 2 = 0

/-- Theorem stating that the 20th is a Thursday in a month with three Sundays on even dates -/
theorem twentieth_is_thursday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Thursday :=
sorry

end twentieth_is_thursday_l1271_127117


namespace euro_problem_l1271_127184

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem euro_problem (a : ℝ) :
  euro a (euro 4 5) = 640 → a = 8 := by
  sorry

end euro_problem_l1271_127184


namespace cosine_rationality_l1271_127130

theorem cosine_rationality (x : ℝ) 
  (h1 : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (h2 : ∃ r : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑r) :
  ∃ (a b : ℚ), (Real.cos (64 * x) = ↑a ∧ Real.cos (65 * x) = ↑b) :=
sorry

end cosine_rationality_l1271_127130


namespace unique_m_for_inequality_l1271_127164

/-- The approximate value of log_10(2) -/
def log10_2 : ℝ := 0.3010

/-- The theorem stating that 155 is the unique positive integer m satisfying the inequality -/
theorem unique_m_for_inequality : ∃! (m : ℕ), m > 0 ∧ (10 : ℝ)^(m - 1) < 2^512 ∧ 2^512 < 10^m :=
by
  -- The proof would go here
  sorry

end unique_m_for_inequality_l1271_127164


namespace power_function_property_l1271_127141

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 / f 2 = 3) : 
  f (1/2) = 1/3 := by
sorry

end power_function_property_l1271_127141


namespace sons_age_l1271_127135

/-- Given a man and his son, where the man is 28 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 26 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 28 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end sons_age_l1271_127135


namespace circle_center_and_radius_l1271_127116

/-- Given a circle C with equation x^2 + y^2 + y = 0, its center is (0, -1/2) and its radius is 1/2 -/
theorem circle_center_and_radius (x y : ℝ) :
  x^2 + y^2 + y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, -1/2) ∧
    radius = 1/2 ∧
    ∀ (point : ℝ × ℝ), point.1^2 + point.2^2 + point.2 = 0 ↔
      (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l1271_127116


namespace circle_radius_in_square_configuration_l1271_127119

/-- A configuration of five congruent circles packed inside a unit square,
    where one circle is centered at the center of the square and
    the other four are tangent to the central circle and two adjacent sides of the square. -/
structure CircleConfiguration where
  radius : ℝ
  is_unit_square : ℝ
  circle_count : ℕ
  central_circle_exists : Bool
  external_circles_tangent : Bool

/-- The radius of each circle in the described configuration is √2 / (4 + 2√2) -/
theorem circle_radius_in_square_configuration (config : CircleConfiguration) 
  (h1 : config.is_unit_square = 1)
  (h2 : config.circle_count = 5)
  (h3 : config.central_circle_exists = true)
  (h4 : config.external_circles_tangent = true) :
  config.radius = Real.sqrt 2 / (4 + 2 * Real.sqrt 2) := by
  sorry

end circle_radius_in_square_configuration_l1271_127119


namespace right_triangle_area_perimeter_l1271_127155

theorem right_triangle_area_perimeter 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  (1/2 * a * b = 30) ∧ (a + b + c = 30) := by
  sorry

end right_triangle_area_perimeter_l1271_127155


namespace log_equality_l1271_127199

theorem log_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) - 2 * Real.log 2 = 0.5 * (Real.log x + Real.log y) := by
  sorry

end log_equality_l1271_127199


namespace arcsin_zero_l1271_127118

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end arcsin_zero_l1271_127118


namespace parking_fines_count_l1271_127159

/-- Represents the number of citations issued for each category -/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  parking : ℕ

/-- Theorem stating that given the conditions, the number of parking fines is 16 -/
theorem parking_fines_count (c : Citations) : 
  c.littering = 4 ∧ 
  c.littering = c.offLeash ∧ 
  c.littering + c.offLeash + c.parking = 24 → 
  c.parking = 16 := by
sorry

end parking_fines_count_l1271_127159


namespace hexagon_side_length_l1271_127197

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (d : ℝ) (h : d = 10) : 
  let s := d * 2 / (3 : ℝ).sqrt
  s = 40 / 3 := by sorry

#check hexagon_side_length

end hexagon_side_length_l1271_127197


namespace power_sum_equality_l1271_127168

theorem power_sum_equality : (-1)^53 + 2^(5^3 - 2^3 + 3^2) = 2^126 - 1 := by
  sorry

end power_sum_equality_l1271_127168


namespace journey_speed_calculation_l1271_127100

/-- Proves that given a journey of 540 miles, where the last 120 miles are traveled at 40 mph,
    and the average speed for the entire journey is 54 mph, the speed for the first 420 miles
    must be 60 mph. -/
theorem journey_speed_calculation (v : ℝ) : 
  v > 0 →                           -- Assume positive speed
  540 / (420 / v + 120 / 40) = 54 → -- Average speed equation
  v = 60 :=                         -- Conclusion: speed for first part is 60 mph
by sorry

end journey_speed_calculation_l1271_127100


namespace total_rainfall_l1271_127190

theorem total_rainfall (monday tuesday wednesday : ℚ) 
  (h1 : monday = 0.16666666666666666)
  (h2 : tuesday = 0.4166666666666667)
  (h3 : wednesday = 0.08333333333333333) :
  monday + tuesday + wednesday = 0.6666666666666667 := by
  sorry

end total_rainfall_l1271_127190


namespace sum_of_prime_divisors_2018_l1271_127176

theorem sum_of_prime_divisors_2018 : ∃ p q : Nat, 
  p.Prime ∧ q.Prime ∧ 
  p ≠ q ∧
  p * q = 2018 ∧
  (∀ r : Nat, r.Prime → r ∣ 2018 → r = p ∨ r = q) ∧
  p + q = 1011 := by
  sorry

end sum_of_prime_divisors_2018_l1271_127176


namespace cricket_team_captain_age_l1271_127187

theorem cricket_team_captain_age
  (team_size : ℕ)
  (captain_age : ℕ)
  (wicket_keeper_age : ℕ)
  (team_average_age : ℕ)
  (h1 : team_size = 11)
  (h2 : wicket_keeper_age = captain_age + 3)
  (h3 : (team_size - 2) * (team_average_age - 1) = team_size * team_average_age - captain_age - wicket_keeper_age)
  (h4 : team_average_age = 23) :
  captain_age = 26 := by
sorry

end cricket_team_captain_age_l1271_127187


namespace segment_translation_l1271_127195

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate_left (p : Point) (units : ℝ) : Point :=
  (p.1 - units, p.2)

-- Define the problem statement
theorem segment_translation :
  let A : Point := (-1, 4)
  let B : Point := (-4, 1)
  let A₁ : Point := translate_left A 4
  let B₁ : Point := translate_left B 4
  A₁ = (-5, 4) ∧ B₁ = (-8, 1) := by sorry

end segment_translation_l1271_127195


namespace lcm_of_12_25_45_60_l1271_127108

theorem lcm_of_12_25_45_60 : Nat.lcm 12 (Nat.lcm 25 (Nat.lcm 45 60)) = 900 := by
  sorry

end lcm_of_12_25_45_60_l1271_127108


namespace max_districts_in_park_l1271_127161

theorem max_districts_in_park (park_side : ℝ) (district_length : ℝ) (district_width : ℝ)
  (h_park_side : park_side = 14)
  (h_district_length : district_length = 8)
  (h_district_width : district_width = 2) :
  ⌊(park_side^2) / (district_length * district_width)⌋ = 12 := by
sorry

end max_districts_in_park_l1271_127161


namespace expression_sign_negative_l1271_127186

theorem expression_sign_negative :
  0 < 1 ∧ 1 < Real.pi / 2 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y) →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) →
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 :=
by sorry

end expression_sign_negative_l1271_127186


namespace ordered_pair_solution_l1271_127142

theorem ordered_pair_solution (a b : ℤ) :
  Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * (1 / Real.sin (50 * π / 180)) →
  a = 3 ∧ b = -1 := by
sorry

end ordered_pair_solution_l1271_127142


namespace projection_theorem_l1271_127144

def vector1 : ℝ × ℝ := (-4, 2)
def vector2 : ℝ × ℝ := (3, 5)

theorem projection_theorem (v : ℝ × ℝ) :
  ∃ (p : ℝ × ℝ), 
    (∃ (k1 : ℝ), p = Prod.mk (k1 * v.1) (k1 * v.2) ∧ 
      (p.1 - vector1.1) * v.1 + (p.2 - vector1.2) * v.2 = 0) ∧
    (∃ (k2 : ℝ), p = Prod.mk (k2 * v.1) (k2 * v.2) ∧ 
      (p.1 - vector2.1) * v.1 + (p.2 - vector2.2) * v.2 = 0) →
    p = (-39/29, 91/29) := by
  sorry

end projection_theorem_l1271_127144


namespace fraction_comparison_l1271_127156

theorem fraction_comparison (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5/3 → 
  (8*x - 3 > 5 - 3*x ↔ (8/11 < x ∧ x < 5/3) ∨ (5/3 < x ∧ x ≤ 3)) :=
by sorry

end fraction_comparison_l1271_127156


namespace power_fraction_simplification_l1271_127124

theorem power_fraction_simplification : (8^15) / (16^7) = 8 := by
  sorry

end power_fraction_simplification_l1271_127124


namespace square_root_equation_l1271_127146

theorem square_root_equation (x : ℝ) :
  Real.sqrt (3 * x + 4) = 12 → x = 140 / 3 := by
  sorry

end square_root_equation_l1271_127146


namespace project_completion_days_l1271_127188

/-- Calculates the number of days required to complete a project given normal work hours, 
    extra work hours, and total project hours. -/
theorem project_completion_days 
  (normal_hours : ℕ) 
  (extra_hours : ℕ) 
  (total_project_hours : ℕ) 
  (h1 : normal_hours = 10)
  (h2 : extra_hours = 5)
  (h3 : total_project_hours = 1500) : 
  total_project_hours / (normal_hours + extra_hours) = 100 := by
  sorry

end project_completion_days_l1271_127188


namespace largest_four_digit_number_l1271_127107

def digits : Finset Nat := {5, 1, 6, 2, 4}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

theorem largest_four_digit_number :
  ∀ n : Nat, is_valid_number n → n ≤ 6542 :=
sorry

end largest_four_digit_number_l1271_127107


namespace mixture_ratio_change_l1271_127151

def initial_ratio : ℚ := 3 / 2
def initial_total : ℚ := 20
def added_water : ℚ := 10

def milk : ℚ := initial_total * (initial_ratio / (1 + initial_ratio))
def water : ℚ := initial_total * (1 / (1 + initial_ratio))

def new_water : ℚ := water + added_water
def new_ratio : ℚ := milk / new_water

theorem mixture_ratio_change :
  new_ratio = 2 / 3 := by sorry

end mixture_ratio_change_l1271_127151


namespace boys_girls_percentage_difference_l1271_127115

theorem boys_girls_percentage_difference : ¬ (∀ (girls boys : ℝ), 
  boys = girls * (1 + 0.25) → girls = boys * (1 - 0.25)) := by
  sorry

end boys_girls_percentage_difference_l1271_127115


namespace employee_count_l1271_127128

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := sorry

/-- The average monthly salary of employees (excluding manager) in Rs. -/
def avg_salary : ℕ := 2000

/-- The increase in average salary when manager's salary is added, in Rs. -/
def salary_increase : ℕ := 200

/-- The manager's monthly salary in Rs. -/
def manager_salary : ℕ := 5800

theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + salary_increase ∧
  num_employees = 18 := by sorry

end employee_count_l1271_127128


namespace thirteen_rectangles_l1271_127194

/-- A rectangle with integer side lengths. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Checks if a rectangle meets the given criteria. -/
def meetsConditions (rect : Rectangle) : Prop :=
  rect.width > 0 ∧ rect.height > 0 ∧
  2 * (rect.width + rect.height) = 80 ∧
  ∃ k : ℕ, rect.width = 3 * k

/-- Two rectangles are considered congruent if they have the same dimensions (ignoring orientation). -/
def areCongruent (rect1 rect2 : Rectangle) : Prop :=
  (rect1.width = rect2.width ∧ rect1.height = rect2.height) ∨
  (rect1.width = rect2.height ∧ rect1.height = rect2.width)

/-- The main theorem stating that there are exactly 13 non-congruent rectangles meeting the conditions. -/
theorem thirteen_rectangles :
  ∃ (rectangles : Finset Rectangle),
    rectangles.card = 13 ∧
    (∀ rect ∈ rectangles, meetsConditions rect) ∧
    (∀ rect, meetsConditions rect → ∃ unique_rect ∈ rectangles, areCongruent rect unique_rect) :=
  sorry

end thirteen_rectangles_l1271_127194


namespace alex_painting_time_l1271_127178

/-- Given Jose's painting rate and the combined painting rate of Jose and Alex,
    calculate Alex's individual painting rate. -/
theorem alex_painting_time (jose_time : ℝ) (combined_time : ℝ) (alex_time : ℝ) : 
  jose_time = 7 → combined_time = 7 / 3 → alex_time = 7 / 2 := by
  sorry

#check alex_painting_time

end alex_painting_time_l1271_127178


namespace car_count_l1271_127111

/-- The total number of cars in a rectangular arrangement -/
def total_cars (front_to_back : ℕ) (left_to_right : ℕ) : ℕ :=
  front_to_back * left_to_right

/-- Theorem stating the total number of cars given the position of red cars -/
theorem car_count (red_from_front red_from_left red_from_back red_from_right : ℕ) 
    (h1 : red_from_front + red_from_back = 25)
    (h2 : red_from_left + red_from_right = 35) :
    total_cars (red_from_front + red_from_back - 1) (red_from_left + red_from_right - 1) = 816 := by
  sorry

#eval total_cars 24 34  -- Should output 816

end car_count_l1271_127111


namespace race_result_l1271_127109

-- Define the participants
inductive Participant
| Hare
| Fox
| Moose

-- Define the possible positions
inductive Position
| First
| Second

-- Define the statements made by the squirrels
def squirrel1_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Hare ∧ second = Participant.Fox

def squirrel2_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Moose ∧ second = Participant.Hare

-- Define the owl's statement
def owl_statement (s1 : Prop) (s2 : Prop) : Prop :=
  (s1 ∧ ¬s2) ∨ (¬s1 ∧ s2)

-- The main theorem
theorem race_result :
  ∃ (winner second : Participant),
    owl_statement (squirrel1_statement winner second) (squirrel2_statement winner second) →
    winner = Participant.Moose ∧ second = Participant.Fox :=
by sorry

end race_result_l1271_127109


namespace ball_distribution_theorem_l1271_127139

/-- Represents the distribution of painted balls among different colors. -/
structure BallDistribution where
  totalBalls : ℕ
  numColors : ℕ
  equalColorCount : ℕ
  doubleColorCount : ℕ
  ballsPerEqualColor : ℕ
  ballsPerDoubleColor : ℕ

/-- Theorem stating the correct distribution of balls among colors. -/
theorem ball_distribution_theorem (d : BallDistribution) : 
  d.totalBalls = 600 ∧ 
  d.numColors = 15 ∧ 
  d.equalColorCount = 10 ∧ 
  d.doubleColorCount = 5 ∧ 
  d.ballsPerDoubleColor = 2 * d.ballsPerEqualColor →
  d.ballsPerEqualColor = 30 ∧ 
  d.ballsPerDoubleColor = 60 ∧
  d.totalBalls = d.equalColorCount * d.ballsPerEqualColor + d.doubleColorCount * d.ballsPerDoubleColor :=
by sorry

end ball_distribution_theorem_l1271_127139


namespace coin_problem_l1271_127157

theorem coin_problem (x : ℕ) : 
  (x + (x + 3) + (20 - 2*x) = 23) →  -- Total coins
  (5*x + 10*(x + 3) + 25*(20 - 2*x) = 320) →  -- Total value
  (20 - 2*x) - x = 2  -- Difference between 25-cent and 5-cent coins
  := by sorry

end coin_problem_l1271_127157


namespace table_price_is_84_l1271_127181

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
def price_ratio_condition : Prop :=
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $96 -/
def total_price_condition : Prop :=
  chair_price + table_price = 96

theorem table_price_is_84 
  (h1 : price_ratio_condition) 
  (h2 : total_price_condition) : 
  table_price = 84 := by sorry

end table_price_is_84_l1271_127181


namespace trains_crossing_time_l1271_127133

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 →
  train_speed_kmh = 54 →
  (2 * train_length) / (2 * (train_speed_kmh * 1000 / 3600)) = 8 := by
  sorry

end trains_crossing_time_l1271_127133


namespace range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l1271_127122

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := ∃ m : ℝ, 1 < m ∧ m < 2 ∧ x = (1/2)^(m-1)

-- Part I
theorem range_of_x_when_a_is_quarter :
  ∀ x : ℝ, (p x (1/4) ∧ q x) ↔ (1/2 < x ∧ x < 3/4) :=
sorry

-- Part II
theorem range_of_a_when_q_sufficient_not_necessary :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ↔ 
  (∀ a : ℝ, (1/3 ≤ a ∧ a ≤ 1/2)) :=
sorry

end range_of_x_when_a_is_quarter_range_of_a_when_q_sufficient_not_necessary_l1271_127122


namespace division_problem_l1271_127134

theorem division_problem (L S q : ℕ) : 
  L - S = 1000 → 
  L = 1100 → 
  L = S * q + 10 → 
  q = 10 := by sorry

end division_problem_l1271_127134


namespace power_of_two_problem_l1271_127189

theorem power_of_two_problem (k : ℕ) (N : ℕ) :
  2^k = N → 2^(2*k + 2) = 64 → N = 4 := by
  sorry

end power_of_two_problem_l1271_127189


namespace hawks_score_l1271_127175

theorem hawks_score (total_points : ℕ) (first_day_margin : ℕ) (second_day_margin : ℕ)
  (h_total : total_points = 130)
  (h_first_margin : first_day_margin = 10)
  (h_second_margin : second_day_margin = 20)
  (h_equal_total : ∃ (eagles_total hawks_total : ℕ),
    eagles_total + hawks_total = total_points ∧ eagles_total = hawks_total) :
  ∃ (hawks_score : ℕ), hawks_score = 65 ∧ hawks_score * 2 = total_points :=
sorry

end hawks_score_l1271_127175


namespace fraction_simplification_l1271_127113

theorem fraction_simplification (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) :
  (5 * a + 3 * b) / (a^2 - b^2) - (2 * a) / (a^2 - b^2) = 3 / (a - b) := by
  sorry

end fraction_simplification_l1271_127113


namespace kia_vehicles_count_l1271_127182

/-- The number of Kia vehicles on the lot -/
def num_kia (total vehicles : ℕ) (num_dodge num_hyundai : ℕ) : ℕ :=
  total - num_dodge - num_hyundai

/-- Theorem stating the number of Kia vehicles on the lot -/
theorem kia_vehicles_count :
  let total := 400
  let num_dodge := total / 2
  let num_hyundai := num_dodge / 2
  num_kia total num_dodge num_hyundai = 100 := by
sorry

end kia_vehicles_count_l1271_127182


namespace sculpture_and_base_height_l1271_127165

/-- The total height of a sculpture and its base -/
def total_height (sculpture_height_m : ℝ) (base_height_cm : ℝ) : ℝ :=
  sculpture_height_m * 100 + base_height_cm

/-- Theorem stating that a 0.88m sculpture on a 20cm base is 108cm tall -/
theorem sculpture_and_base_height : 
  total_height 0.88 20 = 108 := by sorry

end sculpture_and_base_height_l1271_127165


namespace inequality_proof_l1271_127137

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end inequality_proof_l1271_127137


namespace accessories_cost_l1271_127163

theorem accessories_cost (computer_cost : ℝ) (playstation_worth : ℝ) (pocket_payment : ℝ)
  (h1 : computer_cost = 700)
  (h2 : playstation_worth = 400)
  (h3 : pocket_payment = 580) :
  let playstation_sold := playstation_worth * 0.8
  let total_available := playstation_sold + pocket_payment
  let accessories_cost := total_available - computer_cost
  accessories_cost = 200 := by sorry

end accessories_cost_l1271_127163


namespace probability_above_parabola_l1271_127162

def is_single_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def above_parabola (a b : ℕ) : Prop := ∀ x : ℚ, b > a * x^2 + b * x

def count_valid_pairs : ℕ := 69

def total_pairs : ℕ := 81

theorem probability_above_parabola :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 23 / 27 := by sorry

end probability_above_parabola_l1271_127162


namespace octagon_diagonal_intersection_probability_l1271_127173

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def octagon_diagonals : ℕ := octagon_vertices * (octagon_vertices - 3) / 2

/-- The number of ways to select two distinct diagonals from a regular octagon -/
def ways_to_select_two_diagonals : ℕ := 
  Nat.choose octagon_diagonals 2

/-- The number of ways to select four vertices from a regular octagon -/
def ways_to_select_four_vertices : ℕ := 
  Nat.choose octagon_vertices 4

/-- The probability that two randomly selected distinct diagonals 
    in a regular octagon intersect at a point strictly within the octagon -/
theorem octagon_diagonal_intersection_probability : 
  (ways_to_select_four_vertices : ℚ) / ways_to_select_two_diagonals = 7 / 19 := by
  sorry

end octagon_diagonal_intersection_probability_l1271_127173


namespace train_length_l1271_127114

/-- The length of a train given its speed, bridge length, and time to pass the bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 40 →
  (train_speed * passing_time) - bridge_length = 360 :=
by sorry

end train_length_l1271_127114


namespace max_food_per_guest_l1271_127103

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 325) 
  (h2 : min_guests = 163) : 
  ∃ (max_food : ℝ), max_food ≤ 2 ∧ max_food > total_food / min_guests :=
by
  sorry

end max_food_per_guest_l1271_127103


namespace equivalence_of_propositions_l1271_127101

theorem equivalence_of_propositions (a b c : ℝ) :
  (a < b → a + c < b + c) ∧
  (a + c < b + c → a < b) ∧
  (a ≥ b → a + c ≥ b + c) ∧
  (a + c ≥ b + c → a ≥ b) := by
  sorry

end equivalence_of_propositions_l1271_127101


namespace inequality_proof_l1271_127132

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l1271_127132


namespace max_value_constraint_l1271_127183

theorem max_value_constraint (p q r : ℝ) (h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) :
  5 * p + 3 * q + 10 * r ≤ 10 * Real.sqrt 13 / 3 :=
by sorry

end max_value_constraint_l1271_127183


namespace units_digit_of_m_squared_plus_three_to_m_l1271_127158

def m : ℕ := 2021^2 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^2 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 := by sorry

end units_digit_of_m_squared_plus_three_to_m_l1271_127158


namespace work_completion_days_l1271_127112

/-- Calculates the initial number of days planned to complete a work given the total number of men,
    number of absent men, and the number of days taken by the remaining men. -/
def initialDays (totalMen : ℕ) (absentMen : ℕ) (daysWithAbsent : ℕ) : ℕ :=
  (totalMen - absentMen) * daysWithAbsent / totalMen

/-- Proves that given 20 men where 10 become absent and the remaining 10 complete the work in 40 days,
    the original plan was to complete the work in 20 days. -/
theorem work_completion_days :
  initialDays 20 10 40 = 20 := by
  sorry

#eval initialDays 20 10 40

end work_completion_days_l1271_127112


namespace star_three_neg_two_l1271_127180

/-- Definition of the ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := b^3 - abs (b - a)

/-- Theorem stating that 3☆(-2) = -13 -/
theorem star_three_neg_two : star 3 (-2) = -13 := by sorry

end star_three_neg_two_l1271_127180


namespace parallel_plane_count_l1271_127166

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Enum representing the possible number of parallel planes -/
inductive ParallelPlaneCount
  | Zero
  | One
  | Infinite

/-- Function to determine the number of parallel planes -/
def countParallelPlanes (l1 l2 : Line3D) : ParallelPlaneCount :=
  sorry

/-- Theorem stating that the number of parallel planes is either zero, one, or infinite -/
theorem parallel_plane_count (l1 l2 : Line3D) :
  ∃ (count : ParallelPlaneCount), countParallelPlanes l1 l2 = count :=
sorry

end parallel_plane_count_l1271_127166


namespace tammy_climbing_speed_l1271_127177

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * ((total_time + time_difference) / 2) + 
    (v + speed_difference) * ((total_time - time_difference) / 2) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end tammy_climbing_speed_l1271_127177


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1271_127148

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l1271_127148


namespace vacation_payment_difference_is_zero_l1271_127170

/-- Represents the vacation expenses and payments of four friends -/
structure VacationExpenses where
  alice_paid : ℝ
  bob_paid : ℝ
  charlie_paid : ℝ
  donna_paid : ℝ
  alice_to_charlie : ℝ
  bob_to_donna : ℝ

/-- Theorem stating that the difference between Alice's payment to Charlie
    and Bob's payment to Donna is zero, given the vacation expenses -/
theorem vacation_payment_difference_is_zero
  (expenses : VacationExpenses)
  (h1 : expenses.alice_paid = 90)
  (h2 : expenses.bob_paid = 150)
  (h3 : expenses.charlie_paid = 120)
  (h4 : expenses.donna_paid = 240)
  (h5 : expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid = 600)
  (h6 : (expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid) / 4 = 150)
  (h7 : expenses.alice_to_charlie = 150 - expenses.alice_paid)
  (h8 : expenses.bob_to_donna = 150 - expenses.bob_paid)
  : expenses.alice_to_charlie - expenses.bob_to_donna = 0 := by
  sorry

#check vacation_payment_difference_is_zero

end vacation_payment_difference_is_zero_l1271_127170


namespace ratio_problem_l1271_127193

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end ratio_problem_l1271_127193


namespace complex_product_real_imag_parts_l1271_127191

theorem complex_product_real_imag_parts : 
  let Z : ℂ := (1 + Complex.I) * (2 - Complex.I)
  let m : ℝ := Z.re
  let n : ℝ := Z.im
  m * n = 3 := by sorry

end complex_product_real_imag_parts_l1271_127191


namespace courtyard_paving_l1271_127154

/-- The length of the courtyard in meters -/
def courtyard_length : ℝ := 25

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 20

/-- The length of a brick in meters -/
def brick_length : ℝ := 0.15

/-- The width of a brick in meters -/
def brick_width : ℝ := 0.08

/-- The total number of bricks required to cover the courtyard -/
def total_bricks : ℕ := 41667

theorem courtyard_paving :
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = total_bricks := by
  sorry

end courtyard_paving_l1271_127154
