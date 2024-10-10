import Mathlib

namespace probability_not_snowing_l2735_273511

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2/7) : 
  1 - p_snow = 5/7 := by
  sorry

end probability_not_snowing_l2735_273511


namespace max_value_trigonometric_expression_l2735_273500

theorem max_value_trigonometric_expression :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 3 * Real.cos x + 4 * Real.sin x ≤ M :=
sorry

end max_value_trigonometric_expression_l2735_273500


namespace go_game_probability_l2735_273516

theorem go_game_probability (P : ℝ) (h1 : P > 1/2) 
  (h2 : P^2 + (1-P)^2 = 5/8) : P = 3/4 := by
  sorry

end go_game_probability_l2735_273516


namespace range_of_a_l2735_273514

/-- A function that is monotonically increasing on an interval -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

/-- The definition of a hyperbola -/
def IsHyperbola (a : ℝ) : Prop :=
  2 * a^2 - 3 * a - 2 < 0

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (MonotonicallyIncreasing f 1 2) ∧
  (IsHyperbola a) →
  -1/2 < a ∧ a ≤ Real.sqrt 2 :=
by sorry

end range_of_a_l2735_273514


namespace repeating_decimal_568_l2735_273505

/-- The repeating decimal 0.568568568... is equal to the fraction 568/999 -/
theorem repeating_decimal_568 : 
  (∑' n : ℕ, (568 : ℚ) / 1000^(n+1)) = 568 / 999 := by sorry

end repeating_decimal_568_l2735_273505


namespace complex_magnitude_proof_l2735_273513

theorem complex_magnitude_proof : Complex.abs (3/4 + 3*I) = (Real.sqrt 153)/4 := by
  sorry

end complex_magnitude_proof_l2735_273513


namespace race_finish_time_difference_l2735_273535

theorem race_finish_time_difference :
  ∀ (total_runners : ℕ) 
    (fast_runners : ℕ) 
    (slow_runners : ℕ) 
    (fast_time : ℝ) 
    (total_time : ℝ),
  total_runners = fast_runners + slow_runners →
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  total_time = 70 →
  ∃ (slow_time : ℝ),
    total_time = fast_runners * fast_time + slow_runners * slow_time ∧
    slow_time - fast_time = 2 :=
by sorry

end race_finish_time_difference_l2735_273535


namespace triangle_problem_l2735_273538

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : Real :=
  v.x * w.x + v.y * w.y

variable (ABC : Triangle)

/-- Vector m as defined in the problem -/
def m : Vector2D :=
  { x := Real.cos (ABC.A - ABC.B),
    y := Real.sin (ABC.A - ABC.B) }

/-- Vector n as defined in the problem -/
def n : Vector2D :=
  { x := Real.cos ABC.B,
    y := -Real.sin ABC.B }

/-- Main theorem capturing the problem statement and its solution -/
theorem triangle_problem (h1 : dot_product (m ABC) (n ABC) = -3/5)
                         (h2 : ABC.a = 4 * Real.sqrt 2)
                         (h3 : ABC.b = 5) :
  Real.sin ABC.A = 4/5 ∧
  ABC.B = π/4 ∧
  -(ABC.c * Real.cos ABC.B) = -Real.sqrt 2 / 2 :=
sorry

end

end triangle_problem_l2735_273538


namespace complex_fraction_equality_l2735_273523

/-- The complex number -2i/(1+i) is equal to -1-i -/
theorem complex_fraction_equality : ((-2 * Complex.I) / (1 + Complex.I)) = (-1 - Complex.I) := by
  sorry

end complex_fraction_equality_l2735_273523


namespace intersection_point_l2735_273547

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*Real.cos θ + 3 = 0 ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi

/-- Curve C2 in parametric form -/
def C2 (x y t : ℝ) : Prop := x = t * Real.cos (Real.pi/6) ∧ y = t * Real.sin (Real.pi/6)

/-- The intersection point of C1 and C2 has polar coordinates (√3, π/6) -/
theorem intersection_point : 
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ (∃ (x y t : ℝ), C2 x y t ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
  ρ = Real.sqrt 3 ∧ θ = Real.pi/6 := by sorry

end intersection_point_l2735_273547


namespace quadratic_root_in_interval_l2735_273589

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_ineq : |a*b - a*c| > |b^2 - a*c| + |a*b - c^2|)
  : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_root_in_interval_l2735_273589


namespace cone_generatrix_length_l2735_273542

-- Define the cone's properties
def base_radius : ℝ := 6

-- Define the theorem
theorem cone_generatrix_length :
  ∀ (generatrix : ℝ),
  (2 * Real.pi * base_radius = Real.pi * generatrix) →
  generatrix = 12 := by
sorry

end cone_generatrix_length_l2735_273542


namespace sin_315_degrees_l2735_273501

theorem sin_315_degrees :
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l2735_273501


namespace power_of_two_equality_l2735_273574

theorem power_of_two_equality (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ y → y = 37 := by
  sorry

end power_of_two_equality_l2735_273574


namespace max_handshakes_specific_gathering_l2735_273506

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : Nat
  restricted_people : Nat
  max_handshakes_per_person : Nat
  max_handshakes_for_restricted : Nat

/-- Calculates the maximum number of handshakes in a gathering. -/
def max_handshakes (g : Gathering) : Nat :=
  sorry

/-- Theorem stating the maximum number of handshakes for the specific gathering. -/
theorem max_handshakes_specific_gathering :
  let g : Gathering := {
    people := 30,
    restricted_people := 5,
    max_handshakes_per_person := 29,
    max_handshakes_for_restricted := 10
  }
  max_handshakes g = 325 := by
  sorry

end max_handshakes_specific_gathering_l2735_273506


namespace cricket_run_rate_problem_l2735_273530

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_problem :
  required_run_rate 50 10 (32/10) 272 = 6 := by
  sorry

end cricket_run_rate_problem_l2735_273530


namespace sphere_radius_from_cross_sections_l2735_273594

theorem sphere_radius_from_cross_sections (r : ℝ) (h₁ h₂ : ℝ) : 
  h₁ > h₂ →
  h₁ - h₂ = 1 →
  π * (r^2 - h₁^2) = 5 * π →
  π * (r^2 - h₂^2) = 8 * π →
  r = 3 := by
sorry

end sphere_radius_from_cross_sections_l2735_273594


namespace sqrt_x_minus_one_meaningful_l2735_273586

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l2735_273586


namespace apple_probability_l2735_273561

theorem apple_probability (p_less_200 p_not_less_350 : ℝ) 
  (h1 : p_less_200 = 0.25) 
  (h2 : p_not_less_350 = 0.22) : 
  1 - p_less_200 - p_not_less_350 = 0.53 := by
  sorry

end apple_probability_l2735_273561


namespace collinear_points_m_value_l2735_273584

/-- Given vectors AB, BC, and AD, prove that if A, C, and D are collinear, then m = -2/3 -/
theorem collinear_points_m_value 
  (AB BC AD : ℝ × ℝ)
  (h1 : AB = (7, 6))
  (h2 : BC = (-3, m))
  (h3 : AD = (-1, 2*m))
  (h4 : ∃ k : ℝ, k ≠ 0 ∧ AB + BC = k • AD) :
  m = -2/3 :=
sorry

end collinear_points_m_value_l2735_273584


namespace zit_difference_l2735_273557

def swanson_avg : ℕ := 5
def swanson_kids : ℕ := 25
def jones_avg : ℕ := 6
def jones_kids : ℕ := 32

theorem zit_difference : 
  jones_avg * jones_kids - swanson_avg * swanson_kids = 67 := by
  sorry

end zit_difference_l2735_273557


namespace onion_basket_change_l2735_273577

/-- The net change in the number of onions in Sara's basket -/
def net_change (sara_added : ℤ) (sally_removed : ℤ) (fred_added : ℤ) : ℤ :=
  sara_added - sally_removed + fred_added

/-- Theorem stating that the net change in onions is 8 -/
theorem onion_basket_change :
  net_change 4 5 9 = 8 := by
  sorry

end onion_basket_change_l2735_273577


namespace min_value_reciprocal_sum_l2735_273560

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation ax - by + 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 2 = 0

/-- The chord length is 4 -/
def chord_length (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧ line_equation a b x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  chord_length a b → (1/a + 1/b ≥ 3/2 + Real.sqrt 2) ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ chord_length a₀ b₀ ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2) :=
sorry

end min_value_reciprocal_sum_l2735_273560


namespace problem_statement_l2735_273524

theorem problem_statement (x y : ℝ) : 
  (|x - y| > x) → (x + y > 0) → (x > 0 ∧ y > 0) := by sorry

end problem_statement_l2735_273524


namespace roe_savings_l2735_273534

def savings_problem (x : ℝ) : Prop :=
  let jan_to_jul := 7 * x
  let aug_to_nov := 4 * 15
  let december := 20
  jan_to_jul + aug_to_nov + december = 150

theorem roe_savings : ∃ x : ℝ, savings_problem x ∧ x = 10 :=
  sorry

end roe_savings_l2735_273534


namespace lenny_pens_percentage_l2735_273533

theorem lenny_pens_percentage (total_boxes : ℕ) (pens_per_box : ℕ) (remaining_pens : ℕ) : 
  total_boxes = 20 →
  pens_per_box = 5 →
  remaining_pens = 45 →
  ∃ (percentage : ℚ),
    percentage = 40 ∧
    (3/4 : ℚ) * ((total_boxes * pens_per_box : ℚ) - percentage) / 100 * (total_boxes * pens_per_box) = remaining_pens :=
by sorry

end lenny_pens_percentage_l2735_273533


namespace inequality_solution_implies_a_range_l2735_273518

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (1 - a) * x > 1 - a ↔ x < 1) → a > 1 := by sorry

end inequality_solution_implies_a_range_l2735_273518


namespace opposite_of_negative_two_l2735_273528

theorem opposite_of_negative_two :
  -((-2 : ℤ)) = (2 : ℤ) := by sorry

end opposite_of_negative_two_l2735_273528


namespace marching_band_weight_l2735_273555

/-- The total weight carried by the Oprah Winfrey High School marching band -/
def total_weight : ℕ :=
  let trumpet_weight := 5 + 3
  let clarinet_weight := 5 + 3
  let trombone_weight := 10 + 4
  let tuba_weight := 20 + 5
  let drummer_weight := 15 + 6
  let percussionist_weight := 8 + 3
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drummer_count := 2
  let percussionist_count := 4
  trumpet_count * trumpet_weight +
  clarinet_count * clarinet_weight +
  trombone_count * trombone_weight +
  tuba_count * tuba_weight +
  drummer_count * drummer_weight +
  percussionist_count * percussionist_weight

theorem marching_band_weight : total_weight = 393 := by
  sorry

end marching_band_weight_l2735_273555


namespace positive_integer_pairs_satisfying_equation_l2735_273503

theorem positive_integer_pairs_satisfying_equation :
  ∀ a b : ℕ+, 
    (a.val * b.val - a.val - b.val = 12) ↔ ((a = 2 ∧ b = 14) ∨ (a = 14 ∧ b = 2)) :=
by sorry

end positive_integer_pairs_satisfying_equation_l2735_273503


namespace three_digit_sum_24_count_l2735_273546

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_24_count :
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_three_digit n ∧ digit_sum n = 24) ∧ s.card = 4 ∧
    ∀ n, is_three_digit n → digit_sum n = 24 → n ∈ s) :=
by sorry

end three_digit_sum_24_count_l2735_273546


namespace coefficient_of_x_cubed_term_l2735_273563

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := 9 - (3 / 2) * r

theorem coefficient_of_x_cubed_term :
  ∃ (r : ℕ), exponent r = 3 ∧ binomial 9 r = 126 := by sorry

end coefficient_of_x_cubed_term_l2735_273563


namespace proportion_problem_l2735_273572

theorem proportion_problem (hours_per_day : ℕ) (h : hours_per_day = 24) :
  ∃ x : ℕ, (36 : ℚ) / 3 = x / (24 * hours_per_day) ∧ x = 6912 := by
  sorry

end proportion_problem_l2735_273572


namespace birthday_1200th_day_l2735_273576

/-- Given a person born on a Monday, their 1200th day of life will fall on a Thursday. -/
theorem birthday_1200th_day : 
  ∀ (birth_day : Nat), 
  birth_day % 7 = 1 →  -- Monday is represented as 1 (1-based indexing for days of week)
  (birth_day + 1200) % 7 = 5  -- Thursday is represented as 5
  := by sorry

end birthday_1200th_day_l2735_273576


namespace club_size_l2735_273532

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := glove_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 3

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (glove_cost + helmet_cost) + cap_cost

/-- The total expenditure for all players' equipment in dollars -/
def total_expenditure : ℕ := 2968

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size : num_players = 72 := by
  sorry

end club_size_l2735_273532


namespace special_line_unique_l2735_273551

/-- A line that satisfies the given conditions -/
structure SpecialLine where
  m : ℝ
  b : ℝ
  b_nonzero : b ≠ 0
  passes_through : m * 2 + b = 7

/-- The condition for the intersection points -/
def intersectionCondition (l : SpecialLine) (k : ℝ) : Prop :=
  |k^2 + 4*k + 3 - (l.m * k + l.b)| = 4

/-- The main theorem -/
theorem special_line_unique (l : SpecialLine) :
  (∃! k, intersectionCondition l k) → l.m = 10 ∧ l.b = -13 := by
  sorry

end special_line_unique_l2735_273551


namespace spherical_distance_60N_l2735_273517

/-- The spherical distance between two points on a latitude circle --/
def spherical_distance (R : ℝ) (latitude : ℝ) (arc_length : ℝ) : ℝ :=
  sorry

/-- Theorem: Spherical distance between two points on 60°N latitude --/
theorem spherical_distance_60N (R : ℝ) (h : R > 0) :
  spherical_distance R (π / 3) (π * R / 2) = π * R / 3 :=
sorry

end spherical_distance_60N_l2735_273517


namespace complex_fraction_simplification_l2735_273508

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 :=
by sorry

end complex_fraction_simplification_l2735_273508


namespace tuna_cost_theorem_l2735_273582

/-- Calculates the cost of a single can of tuna in cents -/
def tuna_cost_cents (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
                    (amount_paid : ℕ) (change_received : ℕ) : ℕ :=
  let total_paid := amount_paid - change_received
  let coupon_discount := num_coupons * coupon_value
  let total_cost := total_paid * 100 + coupon_discount
  total_cost / num_cans

theorem tuna_cost_theorem : 
  tuna_cost_cents 9 5 25 2000 550 = 175 := by
  sorry

end tuna_cost_theorem_l2735_273582


namespace marys_average_speed_l2735_273509

/-- Mary's round trip walk problem -/
theorem marys_average_speed (uphill_distance : ℝ) (downhill_distance : ℝ)
  (uphill_time : ℝ) (downhill_time : ℝ)
  (h1 : uphill_distance = 1.5)
  (h2 : downhill_distance = 1.5)
  (h3 : uphill_time = 45 / 60)  -- Convert 45 minutes to hours
  (h4 : downhill_time = 15 / 60)  -- Convert 15 minutes to hours
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end marys_average_speed_l2735_273509


namespace marys_remaining_money_l2735_273556

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  50 - (4 * p + 2.5 * p + 2 * 4 * p)

/-- Theorem stating that Mary's remaining money is 50 - 14.5p dollars -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 14.5 * p := by
  sorry

end marys_remaining_money_l2735_273556


namespace function_divisibility_property_l2735_273587

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_divisibility_property 
  (f : ℤ → ℕ) 
  (h : ∀ (m n : ℤ), is_divisible (f (m - n)) (f m - f n)) :
  ∀ (m n : ℤ), is_divisible (f m) (f n) → is_divisible (f m) (f n) :=
sorry

end function_divisibility_property_l2735_273587


namespace f_properties_l2735_273554

noncomputable def f (x : Real) : Real := Real.sqrt 3 * (Real.sin x) ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  let a := π / 2
  let b := π
  ∃ (max_value min_value : Real),
    (∀ x ∈ Set.Icc a b, f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc a b, f x ≥ min_value) ∧
    (f (5 * π / 6) = 0) ∧
    (f π = 0) ∧
    (max_value = Real.sqrt 3) ∧
    (f (π / 2) = max_value) ∧
    (min_value = -1 + Real.sqrt 3 / 2) ∧
    (f (11 * π / 12) = min_value) :=
by sorry

end f_properties_l2735_273554


namespace widget_earnings_proof_l2735_273580

/-- Calculates the earnings per widget given the hourly wage, required widgets per week,
    work hours per week, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (widgets_per_week : ℕ) (hours_per_week : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - (hourly_wage * hours_per_week)) / widgets_per_week

/-- Proves that the earnings per widget is $0.16 given the specific conditions. -/
theorem widget_earnings_proof :
  let hourly_wage : ℚ := 25/2  -- $12.50
  let widgets_per_week : ℕ := 1250
  let hours_per_week : ℕ := 40
  let total_earnings : ℚ := 700
  earnings_per_widget hourly_wage widgets_per_week hours_per_week total_earnings = 4/25  -- $0.16
  := by sorry

end widget_earnings_proof_l2735_273580


namespace solution_l2735_273548

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- Define the set M
def M : Set ℝ := {x | f x = 0}

-- Theorem statement
theorem solution : {1, 3} ∪ {2, 3} = M := by sorry

end solution_l2735_273548


namespace set_operations_and_subset_condition_l2735_273596

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the set M parameterized by k
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset_condition :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by
  sorry

end set_operations_and_subset_condition_l2735_273596


namespace min_value_at_three_l2735_273522

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 28 + Real.sqrt (9 - x^2)

theorem min_value_at_three :
  ∀ x : ℝ, 9 - x^2 ≥ 0 → f 3 ≤ f x :=
by sorry

end min_value_at_three_l2735_273522


namespace max_z_value_l2735_273593

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 9) (prod_eq : x*y + y*z + z*x = 24) :
  z ≤ 5 := by
sorry

end max_z_value_l2735_273593


namespace star_polygon_forms_pyramid_net_iff_l2735_273581

/-- A structure representing two concentric circles with a star-shaped polygon construction -/
structure ConcentricCirclesWithStarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  (r_positive : r > 0)
  (R_greater : R > r)

/-- The condition for the star-shaped polygon to form the net of a pyramid -/
def forms_pyramid_net (c : ConcentricCirclesWithStarPolygon) : Prop :=
  c.R > 2 * c.r

/-- Theorem stating the necessary and sufficient condition for the star-shaped polygon
    to form the net of a pyramid -/
theorem star_polygon_forms_pyramid_net_iff (c : ConcentricCirclesWithStarPolygon) :
  forms_pyramid_net c ↔ c.R > 2 * c.r :=
sorry

end star_polygon_forms_pyramid_net_iff_l2735_273581


namespace climb_out_of_well_l2735_273531

/-- The number of days it takes for a man to climb out of a well -/
def daysToClimbWell (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let dailyProgress := climbUp - slipDown
  let daysForMostOfWell := (wellDepth - 1) / dailyProgress
  let remainingDistance := (wellDepth - 1) % dailyProgress
  if remainingDistance = 0 then
    daysForMostOfWell + 1
  else
    daysForMostOfWell + 2

/-- Theorem stating that it takes 30 days to climb out of a 30-meter well 
    when climbing 4 meters up and slipping 3 meters down each day -/
theorem climb_out_of_well : daysToClimbWell 30 4 3 = 30 := by
  sorry

end climb_out_of_well_l2735_273531


namespace constant_ratio_theorem_l2735_273502

theorem constant_ratio_theorem (x₁ x₂ : ℚ) (y₁ y₂ : ℚ) (k : ℚ) :
  (2 * x₁ - 5) / (y₁ + 10) = k →
  (2 * x₂ - 5) / (y₂ + 10) = k →
  x₁ = 5 →
  y₁ = 4 →
  y₂ = 8 →
  x₂ = 40 / 7 := by
sorry

end constant_ratio_theorem_l2735_273502


namespace highest_validity_rate_is_91_percent_l2735_273539

/-- Represents the voting results for three candidates -/
structure VotingResult where
  total_ballots : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Calculates the highest possible validity rate for a given voting result -/
def highest_validity_rate (result : VotingResult) : ℚ :=
  let total_votes := result.votes_a + result.votes_b + result.votes_c
  let invalid_votes := total_votes - 2 * result.total_ballots
  (result.total_ballots - invalid_votes : ℚ) / result.total_ballots

/-- The main theorem stating the highest possible validity rate -/
theorem highest_validity_rate_is_91_percent (result : VotingResult) :
  result.total_ballots = 100 ∧
  result.votes_a = 88 ∧
  result.votes_b = 75 ∧
  result.votes_c = 46 →
  highest_validity_rate result = 91 / 100 :=
by sorry

#eval highest_validity_rate ⟨100, 88, 75, 46⟩

end highest_validity_rate_is_91_percent_l2735_273539


namespace knight_2008_winner_condition_l2735_273543

/-- Represents the game where n knights sit at a round table, count 1, 2, 3 clockwise,
    and those who say 2 or 3 are eliminated. -/
def KnightGame (n : ℕ) := True

/-- Predicate indicating whether a knight wins the game -/
def IsWinner (game : KnightGame n) (knight : ℕ) : Prop := True

theorem knight_2008_winner_condition (n : ℕ) :
  (∃ (game : KnightGame n), IsWinner game 2008) ↔
  (∃ (k : ℕ), k ≥ 6 ∧ (n = 1338 + 3^k ∨ n = 1338 + 2 * 3^k)) :=
sorry

end knight_2008_winner_condition_l2735_273543


namespace sqrt_588_simplification_l2735_273525

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end sqrt_588_simplification_l2735_273525


namespace fraction_equality_l2735_273510

theorem fraction_equality : (1 : ℚ) / 4 - (1 : ℚ) / 6 + (1 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end fraction_equality_l2735_273510


namespace weight_of_second_person_l2735_273512

/-- Proves that the weight of the second person who joined the group is 78 kg -/
theorem weight_of_second_person
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_first_person : ℝ)
  (h_initial_average : initial_average = 48)
  (h_final_average : final_average = 51)
  (h_initial_members : initial_members = 23)
  (h_weight_first_person : weight_first_person = 93)
  : ∃ (weight_second_person : ℝ),
    weight_second_person = 78 ∧
    (initial_members : ℝ) * final_average =
      initial_members * initial_average + weight_first_person + weight_second_person :=
by sorry

end weight_of_second_person_l2735_273512


namespace evening_ticket_price_is_seven_l2735_273515

/-- Represents the earnings of a movie theater on a single day. -/
structure TheaterEarnings where
  matineePrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  openingNightCustomers : ℕ
  totalEarnings : ℕ

/-- Calculates the evening ticket price based on the theater's earnings. -/
def eveningTicketPrice (e : TheaterEarnings) : ℕ :=
  let totalCustomers := e.matineeCustomers + e.eveningCustomers + e.openingNightCustomers
  let popcornEarnings := (totalCustomers / 2) * e.popcornPrice
  let knownEarnings := e.matineeCustomers * e.matineePrice + 
                       e.openingNightCustomers * e.openingNightPrice + 
                       popcornEarnings
  (e.totalEarnings - knownEarnings) / e.eveningCustomers

/-- Theorem stating that the evening ticket price is 7 dollars given the specific conditions. -/
theorem evening_ticket_price_is_seven :
  let e : TheaterEarnings := {
    matineePrice := 5,
    openingNightPrice := 10,
    popcornPrice := 10,
    matineeCustomers := 32,
    eveningCustomers := 40,
    openingNightCustomers := 58,
    totalEarnings := 1670
  }
  eveningTicketPrice e = 7 := by sorry

end evening_ticket_price_is_seven_l2735_273515


namespace geometric_sequence_ratio_l2735_273520

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a → ArithmeticCondition a →
  ∀ n : ℕ, (a (n + 3) + a (n + 2)) / (a (n + 1) + a n) = 9 := by sorry

end geometric_sequence_ratio_l2735_273520


namespace yellow_pill_cost_l2735_273550

theorem yellow_pill_cost (weeks : ℕ) (daily_yellow : ℕ) (daily_blue : ℕ) 
  (yellow_blue_diff : ℚ) (total_cost : ℚ) :
  weeks = 3 →
  daily_yellow = 1 →
  daily_blue = 1 →
  yellow_blue_diff = 2 →
  total_cost = 903 →
  ∃ (yellow_cost : ℚ), 
    yellow_cost = 22.5 ∧ 
    (weeks * 7 * (yellow_cost + (yellow_cost - yellow_blue_diff)) = total_cost) :=
by sorry

end yellow_pill_cost_l2735_273550


namespace video_game_price_l2735_273571

theorem video_game_price (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  total_earnings = 56 → 
  total_earnings / (total_games - non_working_games) = 7 := by
sorry

end video_game_price_l2735_273571


namespace coronene_bond_arrangements_l2735_273544

/-- Represents a molecule with carbon and hydrogen atoms -/
structure Molecule where
  carbon_count : ℕ
  hydrogen_count : ℕ

/-- Represents the bonding requirements for atoms -/
structure BondRequirement where
  carbon_bonds : ℕ
  hydrogen_bonds : ℕ

/-- Counts the number of valid bond arrangements for a given molecule -/
def count_bond_arrangements (m : Molecule) (req : BondRequirement) : ℕ :=
  sorry

/-- Coronene molecule -/
def coronene : Molecule :=
  { carbon_count := 24, hydrogen_count := 12 }

/-- Standard bonding requirement -/
def standard_requirement : BondRequirement :=
  { carbon_bonds := 4, hydrogen_bonds := 1 }

theorem coronene_bond_arrangements :
  count_bond_arrangements coronene standard_requirement = 20 :=
by sorry

end coronene_bond_arrangements_l2735_273544


namespace deepak_current_age_l2735_273537

-- Define the ratio of Rahul to Deepak's age
def age_ratio : ℚ := 4 / 3

-- Define Rahul's age after 4 years
def rahul_future_age : ℕ := 32

-- Define the number of years in the future for Rahul's age
def years_in_future : ℕ := 4

-- Theorem to prove Deepak's current age
theorem deepak_current_age :
  ∃ (rahul_age deepak_age : ℕ),
    (rahul_age : ℚ) / deepak_age = age_ratio ∧
    rahul_age + years_in_future = rahul_future_age ∧
    deepak_age = 21 := by
  sorry

end deepak_current_age_l2735_273537


namespace line_passes_through_fixed_point_l2735_273549

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), 2 * (-1/2) - m * (-3) + 1 - 3*m = 0 := by
  sorry

end line_passes_through_fixed_point_l2735_273549


namespace trajectory_equation_l2735_273590

/-- The trajectory of a point M(x,y) such that its distance to the line x = 4 
    is twice its distance to the point (1,0) -/
def trajectory (x y : ℝ) : Prop :=
  (x - 4)^2 = ((x - 1)^2 + y^2) / 4

/-- The equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  trajectory x y ↔ 3 * x^2 + 30 * x - y^2 - 63 = 0 := by
  sorry

end trajectory_equation_l2735_273590


namespace car_cleaning_time_l2735_273540

theorem car_cleaning_time (outside_time : ℕ) (inside_time : ℕ) : 
  outside_time = 80 →
  inside_time = outside_time / 4 →
  outside_time + inside_time = 100 :=
by sorry

end car_cleaning_time_l2735_273540


namespace cube_volume_from_surface_area_l2735_273559

theorem cube_volume_from_surface_area :
  ∀ (surface_area : ℝ) (volume : ℝ),
    surface_area = 384 →
    volume = (surface_area / 6) ^ (3/2) →
    volume = 512 := by
  sorry

end cube_volume_from_surface_area_l2735_273559


namespace parabolas_intersection_l2735_273569

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 11

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola1 x = parabola2 x}

-- Theorem statement
theorem parabolas_intersection :
  ∃ (x1 x2 : ℝ), x1 ∈ intersection_points ∧ x2 ∈ intersection_points ∧
  x1 = (3 + Real.sqrt 61) / 2 ∧ x2 = (3 - Real.sqrt 61) / 2 :=
sorry

end parabolas_intersection_l2735_273569


namespace combination_equality_implies_n_18_l2735_273573

theorem combination_equality_implies_n_18 (n : ℕ) :
  (Nat.choose n 14 = Nat.choose n 4) → n = 18 := by
  sorry

end combination_equality_implies_n_18_l2735_273573


namespace ball_fall_time_l2735_273558

/-- The time for a ball to fall from 60 meters to 30 meters under gravity -/
theorem ball_fall_time (g : Real) (h₀ h₁ : Real) (t : Real) :
  g = 9.8 →
  h₀ = 60 →
  h₁ = 30 →
  h₁ = h₀ - (1/2) * g * t^2 →
  t = Real.sqrt (2 * (h₀ - h₁) / g) :=
by sorry

end ball_fall_time_l2735_273558


namespace line_parameterization_l2735_273527

/-- Given a line y = -2x + 7 parameterized by (x, y) = (p, 3) + t(6, l),
    prove that p = 2 and l = -12 -/
theorem line_parameterization (x y p l t : ℝ) : 
  (y = -2 * x + 7) →
  (x = p + 6 * t ∧ y = 3 + l * t) →
  (p = 2 ∧ l = -12) := by
  sorry

end line_parameterization_l2735_273527


namespace tangent_line_equation_l2735_273541

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point through which the line passes
def P : ℝ × ℝ := (2, 0)

-- Define the possible tangent lines
def line1 (x y : ℝ) : Prop := y = 0
def line2 (x y : ℝ) : Prop := 27*x - y - 54 = 0

theorem tangent_line_equation :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, y = m*(x - P.1) + P.2 → 
      (∃ t : ℝ, x = t ∧ y = f t ∧ 
        (∀ s : ℝ, s ≠ t → y - f t < m*(s - t)))) ↔ 
    (line1 x y ∨ line2 x y) :=
sorry

end tangent_line_equation_l2735_273541


namespace top_books_sold_l2735_273575

/-- The number of "TOP" books sold last week -/
def top_books : ℕ := 13

/-- The price of a "TOP" book in dollars -/
def top_price : ℕ := 8

/-- The price of an "ABC" book in dollars -/
def abc_price : ℕ := 23

/-- The number of "ABC" books sold last week -/
def abc_books : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books in dollars -/
def earnings_difference : ℕ := 12

theorem top_books_sold : 
  top_books * top_price - abc_books * abc_price = earnings_difference := by
  sorry

end top_books_sold_l2735_273575


namespace smallest_d_is_one_l2735_273536

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def are_distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts three digits to a three-digit number -/
def to_number (a b c : Digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

/-- Converts four digits to a four-digit number -/
def to_four_digit (d c b d' : Digit) : ℕ :=
  1000 * d.val + 100 * c.val + 10 * b.val + d'.val

theorem smallest_d_is_one :
  ∃ (a b c d : Digit),
    are_distinct a b c d ∧
    to_number a b c * b.val = to_four_digit d c b d ∧
    ∀ (a' b' c' d' : Digit),
      are_distinct a' b' c' d' →
      to_number a' b' c' * b'.val = to_four_digit d' c' b' d' →
      d.val ≤ d'.val :=
sorry

end smallest_d_is_one_l2735_273536


namespace cos_36_degrees_l2735_273519

theorem cos_36_degrees : Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end cos_36_degrees_l2735_273519


namespace two_numbers_with_given_means_l2735_273585

theorem two_numbers_with_given_means : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧
  b = (5 - Real.sqrt 5) / 2 := by
  sorry

end two_numbers_with_given_means_l2735_273585


namespace expected_digits_is_one_point_six_l2735_273578

/-- A fair 20-sided die -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob (n : ℕ) : ℚ := if n ∈ icosahedralDie then 1 / 20 else 0

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun n => prob n * numDigits n)

theorem expected_digits_is_one_point_six :
  expectedDigits = 8/5 := by sorry

end expected_digits_is_one_point_six_l2735_273578


namespace seashell_count_after_six_weeks_l2735_273567

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := 50 + 20 * n

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := 30 * (2 ^ n)

/-- The total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  totalShells 6 = 1110 := by
  sorry

end seashell_count_after_six_weeks_l2735_273567


namespace odd_prime_property_l2735_273521

/-- P(x) is the smallest prime factor of x^2 + 1 -/
noncomputable def P (x : ℕ) : ℕ := Nat.minFac (x^2 + 1)

theorem odd_prime_property (p a : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (ha : a < p) (ha_cong : a^2 + 1 ≡ 0 [MOD p]) : 
  a ≠ p - a ∧ P a = p ∧ P (p - a) = p :=
sorry

end odd_prime_property_l2735_273521


namespace condition_a_geq_4_l2735_273597

theorem condition_a_geq_4 (a : ℝ) :
  (a ≥ 4 → ∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0) ∧
  ¬(∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - 2*x + 4 - a ≤ 0 → a ≥ 4) :=
by sorry

end condition_a_geq_4_l2735_273597


namespace hyperbola_eccentricity_range_l2735_273599

/-- The eccentricity of a hyperbola with given properties is between 1 and 2√3/3 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * x = a * y ∨ b * x = -a * y}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ (p : ℝ × ℝ), p ∈ asymptotes ∩ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 := by
sorry

end hyperbola_eccentricity_range_l2735_273599


namespace crayon_distribution_sum_l2735_273591

def arithmeticSequenceSum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem crayon_distribution_sum :
  arithmeticSequenceSum 18 12 2 = 522 := by
  sorry

end crayon_distribution_sum_l2735_273591


namespace prairie_total_area_l2735_273595

/-- The total area of a prairie given the dusted and untouched areas -/
theorem prairie_total_area (dusted_area untouched_area : ℕ) 
  (h1 : dusted_area = 64535)
  (h2 : untouched_area = 522) :
  dusted_area + untouched_area = 65057 := by
  sorry

#check prairie_total_area

end prairie_total_area_l2735_273595


namespace triangle_side_length_l2735_273592

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 9 →
  b = 2 * Real.sqrt 3 →
  C = 150 * π / 180 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l2735_273592


namespace prob_two_red_balls_l2735_273579

/-- The probability of picking two red balls from a bag -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12) (h2 : red_balls = 5) : 
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 5 / 33 := by
  sorry

end prob_two_red_balls_l2735_273579


namespace triangle_inequality_fraction_l2735_273565

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : (a + b) / (a + b + c) > 1 / 2 := by
  sorry

end triangle_inequality_fraction_l2735_273565


namespace perpendicular_slope_l2735_273553

/-- Given a line with equation 5x - 4y = 20, the slope of the perpendicular line is -4/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 4 * y = 20) → 
  (∃ m : ℝ, m = -4/5 ∧ m * (5/4) = -1) :=
by sorry

end perpendicular_slope_l2735_273553


namespace expression_evaluation_l2735_273507

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℤ := -4
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11 := by
sorry

end expression_evaluation_l2735_273507


namespace banana_profit_calculation_grocer_profit_is_eight_dollars_l2735_273504

/-- Calculates the profit for a grocer selling bananas -/
theorem banana_profit_calculation (purchase_price : ℚ) (purchase_weight : ℚ) 
  (sale_price : ℚ) (sale_weight : ℚ) (total_weight : ℚ) : ℚ :=
  let cost_per_pound := purchase_price / purchase_weight
  let revenue_per_pound := sale_price / sale_weight
  let total_cost := cost_per_pound * total_weight
  let total_revenue := revenue_per_pound * total_weight
  let profit := total_revenue - total_cost
  profit

/-- Proves that the grocer's profit is $8.00 given the specified conditions -/
theorem grocer_profit_is_eight_dollars : 
  banana_profit_calculation (1/2) 3 1 4 96 = 8 := by
  sorry

end banana_profit_calculation_grocer_profit_is_eight_dollars_l2735_273504


namespace gift_purchase_cost_l2735_273570

def total_cost (items : List (ℕ × ℚ)) (sales_tax_rate : ℚ) (credit_card_rebate : ℚ)
  (book_discount_rate : ℚ) (sneaker_discount_rate : ℚ) : ℚ :=
  sorry

theorem gift_purchase_cost :
  let items : List (ℕ × ℚ) := [
    (3, 26), (2, 83), (1, 90), (4, 7), (3, 15), (2, 22), (5, 8), (1, 65)
  ]
  let sales_tax_rate : ℚ := 6.5 / 100
  let credit_card_rebate : ℚ := 12
  let book_discount_rate : ℚ := 10 / 100
  let sneaker_discount_rate : ℚ := 15 / 100
  total_cost items sales_tax_rate credit_card_rebate book_discount_rate sneaker_discount_rate = 564.96 :=
by sorry

end gift_purchase_cost_l2735_273570


namespace shoe_size_for_given_length_xiao_gang_shoe_size_l2735_273564

/-- A linear function representing the relationship between shoe size and foot length. -/
def shoe_size_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Theorem stating the shoe size for a given foot length based on the given conditions. -/
theorem shoe_size_for_given_length (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  shoe_size_function k b 24.5 = 39 := by
  sorry

/-- Corollary: Xiao Gang's shoe size -/
theorem xiao_gang_shoe_size (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  ∃ y : ℝ, y = shoe_size_function k b 24.5 ∧ y = 39 := by
  sorry

end shoe_size_for_given_length_xiao_gang_shoe_size_l2735_273564


namespace cos_double_angle_when_tan_is_one_l2735_273529

theorem cos_double_angle_when_tan_is_one (θ : Real) (h : Real.tan θ = 1) : 
  Real.cos (2 * θ) = 0 := by
  sorry

end cos_double_angle_when_tan_is_one_l2735_273529


namespace leap_year_1996_not_others_l2735_273552

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996_not_others : 
  is_leap_year 1996 ∧ 
  ¬is_leap_year 1998 ∧ 
  ¬is_leap_year 2010 ∧ 
  ¬is_leap_year 2100 :=
by sorry

end leap_year_1996_not_others_l2735_273552


namespace georges_calculation_l2735_273583

theorem georges_calculation (y : ℝ) : y / 7 = 30 → y + 70 = 280 := by
  sorry

end georges_calculation_l2735_273583


namespace quadratic_with_irrational_root_l2735_273568

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
  sorry

end quadratic_with_irrational_root_l2735_273568


namespace unique_triangle_with_perimeter_8_l2735_273562

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle has perimeter 8 -/
def has_perimeter_8 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 8

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The main theorem to be proved -/
theorem unique_triangle_with_perimeter_8 :
  ∃! t : IntTriangle, has_perimeter_8 t ∧
  (∀ t' : IntTriangle, has_perimeter_8 t' → are_congruent t t') :=
sorry

end unique_triangle_with_perimeter_8_l2735_273562


namespace addends_satisfy_conditions_l2735_273566

/-- Represents the correct sum of two addends -/
def correct_sum : Nat := 982

/-- Represents the incorrect sum when one addend is missing a 0 in the units place -/
def incorrect_sum : Nat := 577

/-- Represents the first addend -/
def addend1 : Nat := 450

/-- Represents the second addend -/
def addend2 : Nat := 532

/-- Theorem stating that the two addends satisfy the problem conditions -/
theorem addends_satisfy_conditions : 
  (addend1 + addend2 = correct_sum) ∧ 
  (addend1 + (addend2 - 50) = incorrect_sum) := by
  sorry

#check addends_satisfy_conditions

end addends_satisfy_conditions_l2735_273566


namespace ellipse_eccentricity_l2735_273545

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) 
    that shares a focus with the parabola y^2 = 12x is √3/2 -/
theorem ellipse_eccentricity (k : ℝ) (hk : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let ellipse_focus : ℝ × ℝ := (3, 0)  -- Focus of the parabola
  ellipse_focus ∈ ellipse → -- Assuming the focus is on the ellipse
  let a := Real.sqrt (3*k)  -- Semi-major axis
  let c := 3  -- Distance from center to focus
  let e := c / a  -- Eccentricity
  e = Real.sqrt 3 / 2 := by
sorry


end ellipse_eccentricity_l2735_273545


namespace teams_bc_work_time_l2735_273588

-- Define the workload of projects
def project_a_workload : ℝ := 1
def project_b_workload : ℝ := 1.25

-- Define the time it takes for each team to complete Project A
def team_a_time : ℝ := 20
def team_b_time : ℝ := 24
def team_c_time : ℝ := 30

-- Define variables for the unknown times
def time_bc_together : ℝ := 15
def time_c_with_a : ℝ := 20 -- This is not given, but we need it for the theorem

theorem teams_bc_work_time :
  (time_bc_together / team_b_time + time_bc_together / team_c_time + time_c_with_a / team_b_time = project_b_workload) ∧
  (time_bc_together / team_a_time + time_c_with_a / team_c_time + time_c_with_a / team_a_time = project_a_workload) :=
by sorry

end teams_bc_work_time_l2735_273588


namespace average_pen_price_is_correct_l2735_273526

/-- The average price of a pen before discount given the following conditions:
  * 30 pens and 75 pencils were purchased
  * The total cost after a 10% discount is $510
  * The average price of a pencil before discount is $2.00
-/
def averagePenPrice (numPens : ℕ) (numPencils : ℕ) (totalCostAfterDiscount : ℚ) 
  (pencilPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount : ℚ := totalCostAfterDiscount / (1 - discountRate)
  let totalPencilCost : ℚ := numPencils * pencilPrice
  let totalPenCost : ℚ := totalCostBeforeDiscount - totalPencilCost
  totalPenCost / numPens

theorem average_pen_price_is_correct : 
  averagePenPrice 30 75 510 2 (1/10) = 13.89 := by
  sorry

end average_pen_price_is_correct_l2735_273526


namespace simple_interest_problem_l2735_273598

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 →
  P = 300 := by
sorry

end simple_interest_problem_l2735_273598
