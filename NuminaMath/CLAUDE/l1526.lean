import Mathlib

namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l1526_152681

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 840 →
  absent_children = 420 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l1526_152681


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1526_152660

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line2D) 
  (point : Point2D) 
  (h1 : given_line.a = 1 ∧ given_line.b = 2 ∧ given_line.c = 1) 
  (h2 : point.x = 1 ∧ point.y = 1) : 
  ∃ (l : Line2D), 
    pointOnLine point l ∧ 
    perpendicularLines l given_line ∧ 
    l.a = 2 ∧ l.b = -1 ∧ l.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1526_152660


namespace NUMINAMATH_CALUDE_square_side_equals_pi_l1526_152642

theorem square_side_equals_pi : ∃ x : ℝ, 
  4 * x = 2 * π * 2 ∧ x = π := by
  sorry

end NUMINAMATH_CALUDE_square_side_equals_pi_l1526_152642


namespace NUMINAMATH_CALUDE_complex_roots_circle_l1526_152658

theorem complex_roots_circle (z : ℂ) : 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (-2/3)) = 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_circle_l1526_152658


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1526_152621

theorem polar_to_cartesian_circle (x y ρ : ℝ) :
  ρ = 2 ↔ x^2 + y^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1526_152621


namespace NUMINAMATH_CALUDE_batch_size_calculation_l1526_152643

theorem batch_size_calculation (sample_size : ℕ) (probability : ℚ) (total : ℕ) : 
  sample_size = 30 →
  probability = 1/4 →
  (sample_size : ℚ) / probability = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_batch_size_calculation_l1526_152643


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1526_152686

theorem trig_equation_solution (x : Real) :
  0 < x ∧ x < 180 →
  Real.tan ((150 - x) * Real.pi / 180) = 
    (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) / 
    (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
  x = 100 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1526_152686


namespace NUMINAMATH_CALUDE_billys_age_l1526_152608

theorem billys_age :
  ∀ (billy_age joe_age : ℚ),
    billy_age + 5 = 2 * joe_age →
    billy_age + joe_age = 60 →
    billy_age = 115 / 3 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l1526_152608


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l1526_152664

theorem distance_between_circle_centers (a b c : ℝ) (h_a : a = 17) (h_b : b = 15) (h_c : c = 10) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let AI := Real.sqrt (16 + (K / s) ^ 2)
  20 * AI = 20 * Real.sqrt (16 + 5544 / 441) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l1526_152664


namespace NUMINAMATH_CALUDE_total_apples_is_100_l1526_152667

-- Define the types of apples
inductive AppleType
| Sweet
| Sour

-- Define the price function for apples
def applePrice : AppleType → ℚ
| AppleType.Sweet => 1/2
| AppleType.Sour => 1/10

-- Define the proportion of sweet apples
def sweetProportion : ℚ := 3/4

-- Define the total earnings
def totalEarnings : ℚ := 40

-- Theorem statement
theorem total_apples_is_100 :
  ∃ (n : ℕ), n = 100 ∧
  n * (sweetProportion * applePrice AppleType.Sweet +
       (1 - sweetProportion) * applePrice AppleType.Sour) = totalEarnings :=
by
  sorry


end NUMINAMATH_CALUDE_total_apples_is_100_l1526_152667


namespace NUMINAMATH_CALUDE_exponent_of_five_in_thirty_factorial_l1526_152624

theorem exponent_of_five_in_thirty_factorial :
  ∃ n : ℕ, (30 : ℕ).factorial = 5^7 * n ∧ ¬(5 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_thirty_factorial_l1526_152624


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1526_152698

variable (a b : ℝ × ℝ)

theorem vector_magnitude_proof 
  (h1 : ‖a - 2 • b‖ = 1) 
  (h2 : a • b = 1) : 
  ‖a + 2 • b‖ = 3 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1526_152698


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_l1526_152673

theorem square_minus_a_nonpositive (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_l1526_152673


namespace NUMINAMATH_CALUDE_cylinder_volume_square_cross_section_l1526_152692

/-- The volume of a cylinder with a square cross-section of area 4 is 2π. -/
theorem cylinder_volume_square_cross_section (a : ℝ) (h : a = 4) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π * 2 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_square_cross_section_l1526_152692


namespace NUMINAMATH_CALUDE_pythagorean_side_divisible_by_five_l1526_152623

theorem pythagorean_side_divisible_by_five (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := by
sorry

end NUMINAMATH_CALUDE_pythagorean_side_divisible_by_five_l1526_152623


namespace NUMINAMATH_CALUDE_weight_within_range_l1526_152666

/-- The labeled weight of the flour in kilograms -/
def labeled_weight : ℝ := 25

/-- The tolerance range for the flour weight in kilograms -/
def tolerance : ℝ := 0.2

/-- The actual weight of the flour in kilograms -/
def actual_weight : ℝ := 25.1

/-- Theorem stating that the actual weight is within the acceptable range -/
theorem weight_within_range : 
  labeled_weight - tolerance ≤ actual_weight ∧ actual_weight ≤ labeled_weight + tolerance :=
by sorry

end NUMINAMATH_CALUDE_weight_within_range_l1526_152666


namespace NUMINAMATH_CALUDE_anna_ate_three_cupcakes_l1526_152657

def total_cupcakes : ℕ := 60
def fraction_given_away : ℚ := 4/5
def cupcakes_left : ℕ := 9

theorem anna_ate_three_cupcakes :
  total_cupcakes - (fraction_given_away * total_cupcakes).floor - cupcakes_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_anna_ate_three_cupcakes_l1526_152657


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1526_152661

/-- 
Given that the equation (x^2)/(5-m) + (y^2)/(m+3) = 1 represents an ellipse,
prove that the range of values for m is (-3, 1) ∪ (1, 5).
-/
theorem ellipse_m_range (x y m : ℝ) : 
  (∃ x y, x^2 / (5 - m) + y^2 / (m + 3) = 1 ∧ 5 - m ≠ m + 3) → 
  m ∈ Set.Ioo (-3 : ℝ) 1 ∪ Set.Ioo 1 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1526_152661


namespace NUMINAMATH_CALUDE_identical_differences_l1526_152631

theorem identical_differences (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_bound : ∀ i, a i < 70) : 
  ∃ (d : ℕ) (i j k l : Fin 19), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    a (i.succ) - a i = d ∧ 
    a (j.succ) - a j = d ∧ 
    a (k.succ) - a k = d ∧ 
    a (l.succ) - a l = d :=
sorry

end NUMINAMATH_CALUDE_identical_differences_l1526_152631


namespace NUMINAMATH_CALUDE_average_milk_production_per_cow_l1526_152645

theorem average_milk_production_per_cow (num_cows : ℕ) (total_milk : ℕ) (num_days : ℕ) 
  (h_cows : num_cows = 40)
  (h_milk : total_milk = 12000)
  (h_days : num_days = 30) :
  total_milk / num_cows / num_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_milk_production_per_cow_l1526_152645


namespace NUMINAMATH_CALUDE_smaller_number_between_5_and_8_l1526_152605

theorem smaller_number_between_5_and_8 :
  (5 ≤ 8) ∧ (∀ x : ℝ, 5 ≤ x ∧ x ≤ 8 → 5 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_smaller_number_between_5_and_8_l1526_152605


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_achieved_l1526_152685

theorem max_value_of_sum_of_roots (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  (Real.sqrt (x + 20) + Real.sqrt (20 - x) + Real.sqrt (2 * x) + Real.sqrt (30 - x)) ≤ Real.sqrt 630 :=
by sorry

theorem max_value_achieved (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  ∃ y, 0 ≤ y ∧ y ≤ 20 ∧
    (Real.sqrt (y + 20) + Real.sqrt (20 - y) + Real.sqrt (2 * y) + Real.sqrt (30 - y)) = Real.sqrt 630 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_achieved_l1526_152685


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1526_152634

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 150 →
  throwers = 60 →
  (total_players - throwers) % 2 = 0 →
  105 = throwers + (total_players - throwers) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1526_152634


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1526_152627

/-- Given a principal sum and a time period of 2 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 10%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 2 * 10 / 100 = P / 5) → 10 = (P / 5) / P * 100 / 2 := by
  sorry

#check interest_rate_proof

end NUMINAMATH_CALUDE_interest_rate_proof_l1526_152627


namespace NUMINAMATH_CALUDE_exists_line_not_through_lattice_points_l1526_152640

-- Define a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a lattice point (grid point)
def isLatticePoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem statement
theorem exists_line_not_through_lattice_points :
  ∃ (l : Line), ∀ (p : Point), isLatticePoint p → ¬ pointOnLine p l :=
sorry

end NUMINAMATH_CALUDE_exists_line_not_through_lattice_points_l1526_152640


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1526_152625

theorem profit_percentage_calculation
  (purchase_price : ℕ)
  (repair_cost : ℕ)
  (transportation_charges : ℕ)
  (selling_price : ℕ)
  (h1 : purchase_price = 12000)
  (h2 : repair_cost = 5000)
  (h3 : transportation_charges = 1000)
  (h4 : selling_price = 27000) :
  (selling_price - (purchase_price + repair_cost + transportation_charges)) * 100 /
  (purchase_price + repair_cost + transportation_charges) = 50 :=
by
  sorry

#check profit_percentage_calculation

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1526_152625


namespace NUMINAMATH_CALUDE_max_sock_pairs_john_sock_problem_l1526_152697

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) : ℕ :=
  let total_socks := 2 * initial_pairs
  let remaining_socks := total_socks - lost_socks
  let guaranteed_pairs := initial_pairs - lost_socks
  let possible_new_pairs := (remaining_socks - 2 * guaranteed_pairs) / 2
  guaranteed_pairs + possible_new_pairs

theorem john_sock_problem :
  max_sock_pairs 10 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sock_pairs_john_sock_problem_l1526_152697


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1526_152693

/-- Given a right triangle ABC with vertices A(45,0), B(20,0), and C(0,30),
    and an inscribed rectangle DEFG where the area of triangle CGF is 351,
    prove that the area of rectangle DEFG is 468. -/
theorem inscribed_rectangle_area (A B C D E F G : ℝ × ℝ) : 
  A = (45, 0) →
  B = (20, 0) →
  C = (0, 30) →
  (D.1 ≥ 0 ∧ D.1 ≤ 45 ∧ D.2 = 0) →
  (E.1 = D.1 ∧ E.2 > 0 ∧ E.2 < 30) →
  (F.1 = 20 ∧ F.2 = E.2) →
  (G.1 = 0 ∧ G.2 = E.2) →
  (C.2 - E.2) * F.1 / 2 = 351 →
  (F.1 - D.1) * E.2 = 468 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1526_152693


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1526_152669

theorem rectangular_field_area (perimeter width length : ℝ) : 
  perimeter = 100 → 
  2 * (length + width) = perimeter → 
  length = 3 * width → 
  length * width = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1526_152669


namespace NUMINAMATH_CALUDE_subway_security_comprehensive_l1526_152668

-- Define the type for survey options
inductive SurveyOption
| TouristSatisfaction
| SubwaySecurity
| YellowRiverFish
| LightBulbLifespan

-- Define what it means for a survey to be comprehensive
def is_comprehensive (survey : SurveyOption) : Prop :=
  match survey with
  | SurveyOption.SubwaySecurity => true
  | _ => false

-- Theorem statement
theorem subway_security_comprehensive :
  ∀ (survey : SurveyOption),
    is_comprehensive survey ↔ survey = SurveyOption.SubwaySecurity :=
by sorry

end NUMINAMATH_CALUDE_subway_security_comprehensive_l1526_152668


namespace NUMINAMATH_CALUDE_circumscribed_circle_twice_inscribed_l1526_152651

/-- Given a square, the area of its circumscribed circle is twice the area of its inscribed circle -/
theorem circumscribed_circle_twice_inscribed (a : ℝ) (ha : a > 0) :
  let square_side := 2 * a
  let inscribed_radius := a
  let circumscribed_radius := a * Real.sqrt 2
  (π * circumscribed_radius ^ 2) = 2 * (π * inscribed_radius ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_twice_inscribed_l1526_152651


namespace NUMINAMATH_CALUDE_calculate_expression_l1526_152604

theorem calculate_expression : (-3)^2 - (1/5)⁻¹ - Real.sqrt 8 * Real.sqrt 2 + (-2)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1526_152604


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1526_152677

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 3 ∧ x - y = 1 → x = 2 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ 3*x + 2*y = 10 → x = 3 ∧ y = 1/2 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1526_152677


namespace NUMINAMATH_CALUDE_notebook_duration_example_l1526_152663

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l1526_152663


namespace NUMINAMATH_CALUDE_student_weight_l1526_152695

/-- Given two people, a student and his sister, prove that the student's weight is 60 kg
    under the following conditions:
    1. If the student loses 5 kg, he will weigh 25% more than his sister.
    2. Together, they now weigh 104 kg. -/
theorem student_weight (student_weight sister_weight : ℝ) : 
  (student_weight - 5 = 1.25 * sister_weight) →
  (student_weight + sister_weight = 104) →
  student_weight = 60 := by
  sorry

#check student_weight

end NUMINAMATH_CALUDE_student_weight_l1526_152695


namespace NUMINAMATH_CALUDE_exponent_fraction_simplification_l1526_152690

theorem exponent_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_simplification_l1526_152690


namespace NUMINAMATH_CALUDE_triangle_side_values_l1526_152620

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 8 11 (x.val ^ 2)) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l1526_152620


namespace NUMINAMATH_CALUDE_a_lower_bound_l1526_152679

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * x + 8 * x^3 + a^2 * Real.exp (2 * x) < 4 * x^2 + a * Real.exp x + a^3 * Real.exp (3 * x)

-- State the theorem
theorem a_lower_bound (a : ℝ) (h : inequality_condition a) : a > 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_a_lower_bound_l1526_152679


namespace NUMINAMATH_CALUDE_carrot_count_l1526_152665

theorem carrot_count (initial picked_later thrown_out : ℕ) :
  initial ≥ thrown_out →
  initial - thrown_out + picked_later = initial + picked_later - thrown_out :=
by sorry

end NUMINAMATH_CALUDE_carrot_count_l1526_152665


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_is_one_l1526_152633

theorem zeroth_power_of_nonzero_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_is_one_l1526_152633


namespace NUMINAMATH_CALUDE_power_of_two_multiple_one_two_l1526_152656

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every power of 2, there exists a multiple of it that only contains digits 1 and 2 -/
theorem power_of_two_multiple_one_two :
  ∀ k : ℕ, ∃ n : ℕ, 2^k ∣ n ∧ onlyOneAndTwo n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_multiple_one_two_l1526_152656


namespace NUMINAMATH_CALUDE_no_prime_solution_l1526_152636

theorem no_prime_solution :
  ∀ p : ℕ, Prime p → 2 * p^3 - 5 * p + 14 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1526_152636


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1526_152615

theorem perfect_square_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 - 6*x + a^2 = y^2) → (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1526_152615


namespace NUMINAMATH_CALUDE_toms_average_increase_l1526_152601

/-- Calculates the increase in average score given four exam scores -/
def increase_in_average (score1 score2 score3 score4 : ℚ) : ℚ :=
  let initial_average := (score1 + score2 + score3) / 3
  let new_average := (score1 + score2 + score3 + score4) / 4
  new_average - initial_average

/-- Theorem: The increase in Tom's average score is 3.25 -/
theorem toms_average_increase :
  increase_in_average 72 78 81 90 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_toms_average_increase_l1526_152601


namespace NUMINAMATH_CALUDE_average_difference_l1526_152611

theorem average_difference (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1526_152611


namespace NUMINAMATH_CALUDE_ivan_petrovich_savings_l1526_152688

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  let principal : ℝ := 750000
  let rate : ℝ := 0.08
  let time : ℝ := 12
  simple_interest principal rate time = 1470000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_petrovich_savings_l1526_152688


namespace NUMINAMATH_CALUDE_company_uniforms_l1526_152635

theorem company_uniforms (num_stores : ℕ) (uniforms_per_store : ℕ) 
  (h1 : num_stores = 32) (h2 : uniforms_per_store = 4) : 
  num_stores * uniforms_per_store = 128 := by
  sorry

end NUMINAMATH_CALUDE_company_uniforms_l1526_152635


namespace NUMINAMATH_CALUDE_speed_increase_problem_l1526_152653

/-- The speed increase problem -/
theorem speed_increase_problem 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : initial_speed = 2) 
  (h2 : distance = 2) 
  (h3 : late_time = 1/6) 
  (h4 : early_time = 1/6) : 
  ∃ (speed_increase : ℝ), 
    speed_increase = 
      (distance / (distance / initial_speed - late_time - early_time)) - initial_speed ∧ 
    speed_increase = 1 := by
  sorry

#check speed_increase_problem

end NUMINAMATH_CALUDE_speed_increase_problem_l1526_152653


namespace NUMINAMATH_CALUDE_arrangement_schemes_eq_twelve_l1526_152602

/-- The number of ways to divide 2 teachers and 4 students into 2 groups -/
def arrangement_schemes : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 4 2)

/-- Theorem stating that the number of arrangement schemes is 12 -/
theorem arrangement_schemes_eq_twelve :
  arrangement_schemes = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_schemes_eq_twelve_l1526_152602


namespace NUMINAMATH_CALUDE_jericho_debt_ratio_l1526_152603

theorem jericho_debt_ratio :
  ∀ (jericho_money annika_debt manny_debt : ℚ),
    2 * jericho_money = 60 →
    annika_debt = 14 →
    jericho_money - annika_debt - manny_debt = 9 →
    manny_debt / annika_debt = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jericho_debt_ratio_l1526_152603


namespace NUMINAMATH_CALUDE_wall_width_calculation_l1526_152682

/-- Calculates the width of a wall given its other dimensions and the number and size of bricks used. -/
theorem wall_width_calculation 
  (wall_length wall_height : ℝ) 
  (brick_length brick_width brick_height : ℝ)
  (num_bricks : ℕ) : 
  wall_length = 800 ∧ 
  wall_height = 600 ∧
  brick_length = 125 ∧ 
  brick_width = 11.25 ∧ 
  brick_height = 6 ∧
  num_bricks = 1280 →
  ∃ (wall_width : ℝ), 
    wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
      num_bricks * (brick_length * brick_width * brick_height) := by
  sorry


end NUMINAMATH_CALUDE_wall_width_calculation_l1526_152682


namespace NUMINAMATH_CALUDE_jake_paid_forty_l1526_152641

/-- Calculates the amount paid before working given initial debt, hourly rate, hours worked, and that the remaining debt was paid off by working. -/
def amount_paid_before_working (initial_debt : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  initial_debt - (hourly_rate * hours_worked)

/-- Proves that Jake paid $40 before working, given the problem conditions. -/
theorem jake_paid_forty :
  amount_paid_before_working 100 15 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jake_paid_forty_l1526_152641


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1526_152655

/-- Given a circle and a line, find the equations of lines tangent to the circle and parallel to the given line -/
theorem tangent_line_to_circle (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 + 2*y = 0}
  let l2 := {(x, y) | 3*x + 4*y - 6 = 0}
  ∃ (b : ℝ), (b = -1 ∨ b = 9) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ {(x, y) | 3*x + 4*y + b = 0} → 
      (∃ (x0 y0 : ℝ), (x0, y0) ∈ circle ∧ 
        ((x - x0)^2 + (y - y0)^2 = 1 ∧
         ∀ (x1 y1 : ℝ), (x1, y1) ∈ circle → (x1 - x0)^2 + (y1 - y0)^2 ≤ 1)))
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1526_152655


namespace NUMINAMATH_CALUDE_intersection_M_N_l1526_152606

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : ∀ x : ℝ, x ∈ (M ∩ N) ↔ 2 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1526_152606


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l1526_152689

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (2 : ℝ) -- Slope of the tangent line
  let b := P.2 - m * P.1 -- y-intercept of the tangent line
  b = 10 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l1526_152689


namespace NUMINAMATH_CALUDE_third_term_is_35_l1526_152699

/-- An arithmetic sequence with 6 terms -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (n : ℕ)
  (h_arithmetic : ∀ i j, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j)
  (h_length : n = 6)
  (h_first : a 0 = 23)
  (h_last : a 5 = 47)

/-- The third term of the arithmetic sequence is 35 -/
theorem third_term_is_35 (seq : ArithmeticSequence) : seq.a 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_35_l1526_152699


namespace NUMINAMATH_CALUDE_unique_solution_l1526_152684

theorem unique_solution (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
  a = 7 ∧ b = 5 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1526_152684


namespace NUMINAMATH_CALUDE_farmer_turkeys_l1526_152626

theorem farmer_turkeys (total_cost : ℝ) (kept_turkeys : ℕ) (sale_revenue : ℝ) (profit_per_bird : ℝ) :
  total_cost = 60 ∧
  kept_turkeys = 15 ∧
  sale_revenue = 54 ∧
  profit_per_bird = 0.1 →
  ∃ n : ℕ,
    n * (total_cost / n) = total_cost ∧
    ((total_cost / n) + profit_per_bird) * (n - kept_turkeys) = sale_revenue ∧
    n = 75 :=
by sorry

end NUMINAMATH_CALUDE_farmer_turkeys_l1526_152626


namespace NUMINAMATH_CALUDE_probability_one_defective_six_two_two_l1526_152662

/-- The probability of selecting exactly one defective product from a set of items. -/
def probability_one_defective (total_items defective_items items_selected : ℕ) : ℚ :=
  let favorable_outcomes := (total_items - defective_items).choose (items_selected - 1) * defective_items.choose 1
  let total_outcomes := total_items.choose items_selected
  favorable_outcomes / total_outcomes

/-- Theorem: The probability of selecting exactly one defective product when taking 2 items at random from 6 items with 2 defective products is 8/15. -/
theorem probability_one_defective_six_two_two :
  probability_one_defective 6 2 2 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_six_two_two_l1526_152662


namespace NUMINAMATH_CALUDE_discarded_fruit_percentages_l1526_152691

/-- Represents the percentages of fruit sold and discarded over two days -/
structure FruitPercentages where
  pear_sold_day1 : ℝ
  pear_discarded_day1 : ℝ
  pear_sold_day2 : ℝ
  pear_discarded_day2 : ℝ
  apple_sold_day1 : ℝ
  apple_discarded_day1 : ℝ
  apple_sold_day2 : ℝ
  apple_discarded_day2 : ℝ
  orange_sold_day1 : ℝ
  orange_discarded_day1 : ℝ
  orange_sold_day2 : ℝ
  orange_discarded_day2 : ℝ

/-- Calculates the total percentage of fruit discarded over two days -/
def totalDiscardedPercentage (fp : FruitPercentages) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct percentages of discarded fruit -/
theorem discarded_fruit_percentages (fp : FruitPercentages) 
  (h1 : fp.pear_sold_day1 = 20)
  (h2 : fp.pear_discarded_day1 = 30)
  (h3 : fp.pear_sold_day2 = 10)
  (h4 : fp.pear_discarded_day2 = 20)
  (h5 : fp.apple_sold_day1 = 25)
  (h6 : fp.apple_discarded_day1 = 15)
  (h7 : fp.apple_sold_day2 = 15)
  (h8 : fp.apple_discarded_day2 = 10)
  (h9 : fp.orange_sold_day1 = 30)
  (h10 : fp.orange_discarded_day1 = 35)
  (h11 : fp.orange_sold_day2 = 20)
  (h12 : fp.orange_discarded_day2 = 30) :
  totalDiscardedPercentage fp = (34.08, 16.66875, 35.42) := by
  sorry

end NUMINAMATH_CALUDE_discarded_fruit_percentages_l1526_152691


namespace NUMINAMATH_CALUDE_z_in_terms_of_a_b_s_l1526_152648

theorem z_in_terms_of_a_b_s 
  (z a b s : ℝ) 
  (hz : z ≠ 0) 
  (heq : z = a^3 * b^2 + 6*z*s - 9*s^2) :
  z = (a^3 * b^2 - 9*s^2) / (1 - 6*s) :=
by sorry

end NUMINAMATH_CALUDE_z_in_terms_of_a_b_s_l1526_152648


namespace NUMINAMATH_CALUDE_relationship_xyz_l1526_152622

theorem relationship_xyz (a : ℝ) (x y z : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (hx : x = a^a) (hy : y = a) (hz : z = Real.log a / Real.log a) : 
  z > x ∧ x > y := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l1526_152622


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l1526_152619

theorem choose_three_from_nine (n : ℕ) (r : ℕ) (h1 : n = 9) (h2 : r = 3) :
  Nat.choose n r = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l1526_152619


namespace NUMINAMATH_CALUDE_sqrt_x_plus_5_real_l1526_152632

theorem sqrt_x_plus_5_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 5) ↔ x ≥ -5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_5_real_l1526_152632


namespace NUMINAMATH_CALUDE_each_score_is_individual_l1526_152672

/-- Represents a student in the study -/
structure Student where
  id : Nat
  score : ℝ

/-- Represents the statistical study -/
structure CivilizationKnowledgeStudy where
  population : Finset Student
  sample : Finset Student
  pop_size : Nat
  sample_size : Nat

/-- Properties of the study -/
def valid_study (study : CivilizationKnowledgeStudy) : Prop :=
  study.pop_size = 1200 ∧
  study.sample_size = 100 ∧
  study.sample ⊆ study.population ∧
  study.population.card = study.pop_size ∧
  study.sample.card = study.sample_size

/-- Theorem stating that each student's score is an individual observation -/
theorem each_score_is_individual (study : CivilizationKnowledgeStudy) 
  (h : valid_study study) : 
  ∀ s ∈ study.population, ∃! x : ℝ, x = s.score :=
sorry

end NUMINAMATH_CALUDE_each_score_is_individual_l1526_152672


namespace NUMINAMATH_CALUDE_broken_line_length_bound_l1526_152687

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a broken line on a chessboard -/
structure BrokenLine :=
  (board : Chessboard)
  (isClosed : Bool)
  (noSelfIntersections : Bool)
  (joinsAdjacentCells : Bool)
  (isSymmetricToDiagonal : Bool)

/-- Calculates the length of a broken line -/
def brokenLineLength (line : BrokenLine) : ℝ :=
  sorry

/-- Theorem: The length of a specific broken line on a 15x15 chessboard is at most 200 -/
theorem broken_line_length_bound (line : BrokenLine) :
  line.board.size = 15 →
  line.isClosed = true →
  line.noSelfIntersections = true →
  line.joinsAdjacentCells = true →
  line.isSymmetricToDiagonal = true →
  brokenLineLength line ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_broken_line_length_bound_l1526_152687


namespace NUMINAMATH_CALUDE_profit_percentage_per_item_l1526_152609

theorem profit_percentage_per_item (total_cost : ℝ) (num_bought num_sold : ℕ) 
  (h1 : num_bought = 30)
  (h2 : num_sold = 20)
  (h3 : total_cost > 0)
  (h4 : num_bought > num_sold)
  (h5 : num_sold * (total_cost / num_bought) = total_cost) :
  (((total_cost / num_sold) - (total_cost / num_bought)) / (total_cost / num_bought)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_per_item_l1526_152609


namespace NUMINAMATH_CALUDE_twenty_fifth_is_monday_l1526_152650

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

/-- Checks if a given number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 == 0

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  threeEvenSaturdays : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Saturday ∧
    d2.dayOfWeek = DayOfWeek.Saturday ∧
    d3.dayOfWeek = DayOfWeek.Saturday ∧
    isEven d1.day ∧ isEven d2.day ∧ isEven d3.day ∧
    d1.day ≠ d2.day ∧ d2.day ≠ d3.day ∧ d1.day ≠ d3.day

/-- Theorem: In a month where three Saturdays fall on even dates, 
    the 25th day of that month is a Monday -/
theorem twenty_fifth_is_monday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 25 ∧ d.dayOfWeek = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_is_monday_l1526_152650


namespace NUMINAMATH_CALUDE_triangle_problem_l1526_152671

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.sin C = b * Real.sin A ∧ b * Real.sin C = c * Real.sin B ∧ c * Real.sin A = a * Real.sin B →
  (a * (Real.sin C - Real.sin A)) / (Real.sin C + Real.sin B) = c - b →
  Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4 →
  B = π / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1526_152671


namespace NUMINAMATH_CALUDE_custom_mul_equality_l1526_152618

/-- Custom multiplication operation for real numbers -/
def custom_mul (x y : ℝ) : ℝ := (x - y)^2

/-- Theorem stating the equality for the given expression using custom multiplication -/
theorem custom_mul_equality (x y z : ℝ) : 
  custom_mul (x - y) (y - z) = (x - 2*y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_equality_l1526_152618


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l1526_152617

/-- Given a point A with coordinates (1, 2), its symmetric point A' with respect to the origin has coordinates (-1, -2) -/
theorem symmetric_point_wrt_origin :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := (-A.1, -A.2)
  A' = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l1526_152617


namespace NUMINAMATH_CALUDE_inequality_proof_l1526_152670

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 + 1/x) * (1 + 1/y) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1526_152670


namespace NUMINAMATH_CALUDE_point_on_graph_l1526_152600

def is_on_graph (x y k : ℝ) : Prop := y = k * x - 2

theorem point_on_graph (k : ℝ) :
  is_on_graph 2 4 k → is_on_graph 1 1 k :=
by sorry

end NUMINAMATH_CALUDE_point_on_graph_l1526_152600


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l1526_152629

/-- Given a flagpole and a building under similar conditions, 
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 22)
  (h3 : building_shadow = 55)
  (h4 : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l1526_152629


namespace NUMINAMATH_CALUDE_at_least_one_equal_to_a_l1526_152659

theorem at_least_one_equal_to_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equal_to_a_l1526_152659


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1526_152639

theorem batsman_average_increase 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (new_average : ℚ) :
  total_innings = 17 →
  last_innings_score = 85 →
  new_average = 37 →
  (total_innings * new_average - last_innings_score) / (total_innings - 1) + 3 = new_average :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1526_152639


namespace NUMINAMATH_CALUDE_min_value_3a_minus_2ab_l1526_152680

theorem min_value_3a_minus_2ab :
  ∀ a b : ℕ+, a < 8 → b < 8 → (3 * a - 2 * a * b : ℤ) ≥ -77 ∧
  ∃ a₀ b₀ : ℕ+, a₀ < 8 ∧ b₀ < 8 ∧ (3 * a₀ - 2 * a₀ * b₀ : ℤ) = -77 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_minus_2ab_l1526_152680


namespace NUMINAMATH_CALUDE_appetizers_per_guest_is_six_l1526_152646

def number_of_guests : ℕ := 30

def prepared_appetizers : ℕ := 3 * 12 + 2 * 12 + 2 * 12

def additional_appetizers : ℕ := 8 * 12

def total_appetizers : ℕ := prepared_appetizers + additional_appetizers

def appetizers_per_guest : ℚ := total_appetizers / number_of_guests

theorem appetizers_per_guest_is_six :
  appetizers_per_guest = 6 := by
  sorry

end NUMINAMATH_CALUDE_appetizers_per_guest_is_six_l1526_152646


namespace NUMINAMATH_CALUDE_A_power_100_eq_A_l1526_152675

/-- The matrix A -/
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]]

/-- Theorem stating that A^100 = A -/
theorem A_power_100_eq_A : A ^ 100 = A := by sorry

end NUMINAMATH_CALUDE_A_power_100_eq_A_l1526_152675


namespace NUMINAMATH_CALUDE_trees_on_rectangular_plot_l1526_152630

/-- The number of trees planted on a rectangular plot -/
def num_trees (length width spacing : ℕ) : ℕ :=
  ((length / spacing) + 1) * ((width / spacing) + 1)

/-- Theorem: The number of trees planted at a five-foot distance from each other
    on a rectangular plot of land with sides 120 feet and 70 feet is 375 -/
theorem trees_on_rectangular_plot :
  num_trees 120 70 5 = 375 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_rectangular_plot_l1526_152630


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1526_152674

/-- Given a quadratic equation x^2 - (3+m)x + 3m = 0 with real roots x1 and x2
    satisfying 2x1 - x1x2 + 2x2 = 12, prove that x1 = -6 and x2 = 3 -/
theorem quadratic_equation_roots (m : ℝ) (x1 x2 : ℝ) :
  x1^2 - (3+m)*x1 + 3*m = 0 →
  x2^2 - (3+m)*x2 + 3*m = 0 →
  2*x1 - x1*x2 + 2*x2 = 12 →
  (x1 = -6 ∧ x2 = 3) ∨ (x1 = 3 ∧ x2 = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1526_152674


namespace NUMINAMATH_CALUDE_valid_polynomials_eq_target_polynomials_l1526_152696

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  b : ℤ
  c : ℤ

/-- The set of all valid quadratic polynomials satisfying the conditions -/
def ValidPolynomials : Set QuadraticPolynomial :=
  { p | ∃ (r₁ r₂ : ℤ), 
    (r₁ * r₂ = p.c) ∧ 
    (r₁ + r₂ = -p.b) ∧ 
    (1 + p.b + p.c = 10) }

/-- The set of specific polynomials we want to prove are the only valid ones -/
def TargetPolynomials : Set QuadraticPolynomial :=
  { ⟨-13, 22⟩, ⟨-9, 18⟩, ⟨9, 0⟩, ⟨5, 4⟩ }

/-- The main theorem: ValidPolynomials equals TargetPolynomials -/
theorem valid_polynomials_eq_target_polynomials : 
  ValidPolynomials = TargetPolynomials := by
  sorry


end NUMINAMATH_CALUDE_valid_polynomials_eq_target_polynomials_l1526_152696


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l1526_152644

/-- A function f(x) = x^3 - 3x^2 + ax - 5 is monotonically increasing on ℝ if and only if a ≥ 3 -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - 3*x^2 + a*x - 5)) ↔ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l1526_152644


namespace NUMINAMATH_CALUDE_find_S_l1526_152612

-- Define the relationship between R, S, and T
def relationship (c R S T : ℝ) : Prop :=
  R = c * (S / T)

-- Define the theorem
theorem find_S (c : ℝ) :
  relationship c (4/3) (3/7) (9/14) →
  relationship c (Real.sqrt 98) S (Real.sqrt 32) →
  S = 28 := by
  sorry


end NUMINAMATH_CALUDE_find_S_l1526_152612


namespace NUMINAMATH_CALUDE_incorrect_trigonometric_statement_l1526_152683

theorem incorrect_trigonometric_statement :
  ∃ (α : Real) (k : Real), k ≠ 0 ∧ 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = 3 * k ∧ t * Real.sin α = 4 * k) ∧
  Real.sin α ≠ 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_trigonometric_statement_l1526_152683


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1526_152613

/-- Given that ten identical bowling balls weigh the same as four identical kayaks,
    and one kayak weighs 35 pounds, prove that one bowling ball weighs 14 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowling_ball_weight kayak_weight : ℝ),
    kayak_weight = 35 →
    10 * bowling_ball_weight = 4 * kayak_weight →
    bowling_ball_weight = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1526_152613


namespace NUMINAMATH_CALUDE_illustration_project_time_l1526_152637

/-- Calculates the total time spent on an illustration project with three phases -/
def total_illustration_time (
  landscape_count : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_reduction : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_count : ℕ)
  (portrait_draw_time : ℝ)
  (portrait_color_reduction : ℝ)
  (portrait_enhance_time : ℝ)
  (abstract_count : ℕ)
  (abstract_draw_time : ℝ)
  (abstract_color_reduction : ℝ)
  (abstract_enhance_time : ℝ) : ℝ :=
  let landscape_time := landscape_count * (landscape_draw_time + landscape_draw_time * (1 - landscape_color_reduction) + landscape_enhance_time)
  let portrait_time := portrait_count * (portrait_draw_time + portrait_draw_time * (1 - portrait_color_reduction) + portrait_enhance_time)
  let abstract_time := abstract_count * (abstract_draw_time + abstract_draw_time * (1 - abstract_color_reduction) + abstract_enhance_time)
  landscape_time + portrait_time + abstract_time

theorem illustration_project_time :
  total_illustration_time 10 2 0.3 0.75 15 3 0.25 1 20 1.5 0.4 0.5 = 193.25 := by
  sorry

end NUMINAMATH_CALUDE_illustration_project_time_l1526_152637


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l1526_152694

/-- The probability of drawing exactly one white ball from a bag -/
def prob_one_white (red : ℕ) (white : ℕ) : ℚ :=
  white / (red + white)

/-- The probability of drawing exactly one red ball from a bag -/
def prob_one_red (red : ℕ) (white : ℕ) : ℚ :=
  red / (red + white)

theorem ball_probability_theorem (n : ℕ) :
  prob_one_white 5 3 = 3/8 ∧
  (prob_one_red 5 (3 + n) = 1/2 → n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l1526_152694


namespace NUMINAMATH_CALUDE_nico_wednesday_pages_l1526_152614

/-- Represents the number of pages read on each day --/
structure PagesRead where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Represents Nico's reading activity over three days --/
def nico_reading : PagesRead where
  monday := 20
  tuesday := 12
  wednesday := 51 - (20 + 12)

theorem nico_wednesday_pages :
  nico_reading.wednesday = 19 := by
  sorry

end NUMINAMATH_CALUDE_nico_wednesday_pages_l1526_152614


namespace NUMINAMATH_CALUDE_yoongi_behind_count_l1526_152610

/-- Given a line of students, calculate the number of students behind a specific position. -/
def studentsBehindinLine (totalStudents : ℕ) (position : ℕ) : ℕ :=
  totalStudents - position

theorem yoongi_behind_count :
  let totalStudents : ℕ := 20
  let jungkookPosition : ℕ := 3
  let yoongiPosition : ℕ := jungkookPosition + 1
  studentsBehindinLine totalStudents yoongiPosition = 16 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_behind_count_l1526_152610


namespace NUMINAMATH_CALUDE_books_combination_l1526_152676

/- Given conditions -/
def totalBooks : ℕ := 13
def booksToSelect : ℕ := 3

/- Theorem to prove -/
theorem books_combination : Nat.choose totalBooks booksToSelect = 286 := by
  sorry

end NUMINAMATH_CALUDE_books_combination_l1526_152676


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1526_152652

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_picks : ℕ := 7
def num_blue_picked : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_picks num_blue_picked) * 
  (probability_blue ^ num_blue_picked) * 
  (probability_red ^ (num_picks - num_blue_picked)) = 862 / 3417 :=
sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1526_152652


namespace NUMINAMATH_CALUDE_patio_layout_l1526_152607

theorem patio_layout (r c : ℕ) : 
  r * c = 160 ∧ 
  (r + 4) * (c - 2) = 160 ∧ 
  r % 5 = 0 ∧ 
  c % 5 = 0 → 
  r = 16 := by sorry

end NUMINAMATH_CALUDE_patio_layout_l1526_152607


namespace NUMINAMATH_CALUDE_min_side_b_in_special_triangle_l1526_152649

/-- 
Given a triangle ABC where:
- Angles A, B, and C form an arithmetic sequence
- Sides opposite to angles A, B, and C are a, b, and c respectively
- 3ac + b² = 25
This theorem states that the minimum value of side b is 5/2
-/
theorem min_side_b_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → c > 0 →  -- Ensuring positive side lengths
  2 * B = A + C →  -- Arithmetic sequence condition
  A + B + C = π →  -- Sum of angles in a triangle
  3 * a * c + b^2 = 25 →  -- Given condition
  b ≥ 5/2 ∧ ∃ (a₀ c₀ : ℝ), a₀ > 0 ∧ c₀ > 0 ∧ 3 * a₀ * c₀ + (5/2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_side_b_in_special_triangle_l1526_152649


namespace NUMINAMATH_CALUDE_problem_statement_l1526_152647

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 3 - x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1526_152647


namespace NUMINAMATH_CALUDE_sequence_a_integer_sequence_a_recurrence_l1526_152678

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) * (sequence_a (n + 1) + 1) / sequence_a n

theorem sequence_a_integer (n : ℕ) : ∃ k : ℤ, sequence_a n = k := by
  sorry

theorem sequence_a_recurrence (n : ℕ) : n ≥ 1 →
  sequence_a (n + 1) * sequence_a (n - 1) = sequence_a n * (sequence_a n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_integer_sequence_a_recurrence_l1526_152678


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1526_152616

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1526_152616


namespace NUMINAMATH_CALUDE_jogger_train_distance_l1526_152638

/-- Proves the distance a jogger is ahead of a train engine given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time = train_length + 230 := by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l1526_152638


namespace NUMINAMATH_CALUDE_evaluate_expression_l1526_152628

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1526_152628


namespace NUMINAMATH_CALUDE_six_hamburgers_left_over_l1526_152654

/-- Given a restaurant that made hamburgers and served some, calculate the number left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ := made - served

/-- Prove that when 9 hamburgers are made and 3 are served, 6 are left over. -/
theorem six_hamburgers_left_over :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_hamburgers_left_over_l1526_152654
