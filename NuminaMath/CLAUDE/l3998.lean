import Mathlib

namespace inequality_proof_l3998_399804

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l3998_399804


namespace sequence_a1_value_l3998_399883

theorem sequence_a1_value (p q : ℝ) (a : ℕ → ℝ) 
  (hp : p > 0) (hq : q > 0)
  (ha_pos : ∀ n, a n > 0)
  (ha_0 : a 0 = 1)
  (ha_rec : ∀ n, a (n + 2) = p * a n - q * a (n + 1)) :
  a 1 = (-q + Real.sqrt (q^2 + 4*p)) / 2 :=
sorry

end sequence_a1_value_l3998_399883


namespace boat_speed_l3998_399824

theorem boat_speed (t : ℝ) (h : t > 0) : 
  let v_s : ℝ := 21
  let upstream_time : ℝ := 2 * t
  let downstream_time : ℝ := t
  let v_b : ℝ := (v_s * (upstream_time + downstream_time)) / (upstream_time - downstream_time)
  v_b = 63 := by sorry

end boat_speed_l3998_399824


namespace squared_binomial_subtraction_difference_of_squares_l3998_399816

-- Problem 1
theorem squared_binomial_subtraction (a b : ℝ) :
  a^2 * b - (-2 * a * b^2)^2 = a^2 * b - 4 * a^2 * b^4 := by sorry

-- Problem 2
theorem difference_of_squares (x y : ℝ) :
  (3 * x - 2 * y) * (3 * x + 2 * y) = 9 * x^2 - 4 * y^2 := by sorry

end squared_binomial_subtraction_difference_of_squares_l3998_399816


namespace population_and_sample_properties_l3998_399885

/-- Represents a student in the seventh grade -/
structure Student where
  id : Nat

/-- Represents a population of students -/
structure Population where
  students : Finset Student
  size : Nat
  h_size : students.card = size

/-- Represents a sample of students -/
structure Sample where
  students : Finset Student
  population : Population
  h_subset : students ⊆ population.students

/-- The main theorem stating properties of the population and sample -/
theorem population_and_sample_properties
  (total_students : Finset Student)
  (h_total : total_students.card = 800)
  (sample_students : Finset Student)
  (h_sample : sample_students ⊆ total_students)
  (h_sample_size : sample_students.card = 50) :
  let pop : Population := ⟨total_students, 800, h_total⟩
  let samp : Sample := ⟨sample_students, pop, h_sample⟩
  (pop.size = 800) ∧
  (samp.students ⊆ pop.students) ∧
  (samp.students.card = 50) := by
  sorry


end population_and_sample_properties_l3998_399885


namespace arithmetic_simplification_l3998_399831

theorem arithmetic_simplification : 4 * (8 - 3) - 6 / 3 = 18 := by
  sorry

end arithmetic_simplification_l3998_399831


namespace x_value_proof_l3998_399875

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 16 * x * y = 2 * x^3 + 4 * x^2 * y) : x = 4 := by
  sorry

end x_value_proof_l3998_399875


namespace complement_of_A_l3998_399871

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := Set.Ioc (-2) 1

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Iic (-2) ∪ Set.Ioi 1 := by sorry

end complement_of_A_l3998_399871


namespace base_conversion_problem_l3998_399891

theorem base_conversion_problem :
  ∀ c d : ℕ,
  c < 10 → d < 10 →
  (5 * 6^2 + 2 * 6^1 + 4 * 6^0 = 2 * 10^2 + c * 10^1 + d * 10^0) →
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end base_conversion_problem_l3998_399891


namespace population_decreases_below_threshold_l3998_399820

/-- The annual decrease rate of the population -/
def decrease_rate : ℝ := 0.5

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.05

/-- The number of years it takes for the population to decrease below the threshold -/
def years_to_threshold : ℕ := 5

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (decrease_rate ^ years)

theorem population_decreases_below_threshold :
  ∀ initial_population : ℝ,
  initial_population > 0 →
  population_after_years initial_population years_to_threshold < threshold * initial_population ∧
  population_after_years initial_population (years_to_threshold - 1) ≥ threshold * initial_population :=
by sorry

end population_decreases_below_threshold_l3998_399820


namespace paperclip_excess_day_l3998_399809

def paperclip_sequence (k : ℕ) : ℕ := 4 * 3^k

theorem paperclip_excess_day :
  (∀ j : ℕ, j < 6 → paperclip_sequence j ≤ 2000) ∧
  paperclip_sequence 6 > 2000 :=
sorry

end paperclip_excess_day_l3998_399809


namespace sin_150_degrees_l3998_399897

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l3998_399897


namespace expenditure_recording_l3998_399839

-- Define a type for financial transactions
inductive Transaction
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

-- Define a function to record transactions
def record_transaction (t : Transaction) : ℤ :=
  match t with
  | Transaction.Income a => a
  | Transaction.Expenditure a => -a

-- Theorem statement
theorem expenditure_recording (income_amount expenditure_amount : ℤ) 
  (h1 : income_amount > 0) (h2 : expenditure_amount > 0) :
  record_transaction (Transaction.Income income_amount) = income_amount ∧
  record_transaction (Transaction.Expenditure expenditure_amount) = -expenditure_amount :=
by sorry

end expenditure_recording_l3998_399839


namespace backpack_cost_l3998_399892

/-- The cost of backpacks with discount and monogramming -/
theorem backpack_cost (original_price : ℝ) (discount_percent : ℝ) (monogram_fee : ℝ) (quantity : ℕ) :
  original_price = 20 →
  discount_percent = 20 →
  monogram_fee = 12 →
  quantity = 5 →
  quantity * (original_price * (1 - discount_percent / 100) + monogram_fee) = 140 := by
  sorry

end backpack_cost_l3998_399892


namespace pants_cost_is_6_l3998_399868

/-- The cost of one pair of pants -/
def pants_cost : ℚ := 6

/-- The cost of one shirt -/
def shirt_cost : ℚ := 10

/-- Theorem stating the cost of one pair of pants is $6 -/
theorem pants_cost_is_6 :
  (2 * pants_cost + 5 * shirt_cost = 62) →
  (2 * shirt_cost = 20) →
  pants_cost = 6 := by
  sorry

end pants_cost_is_6_l3998_399868


namespace semicircle_radius_in_specific_triangle_l3998_399807

/-- An isosceles triangle with a semicircle inscribed on its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base
  /-- The radius plus the height of the triangle equals the length of the equal sides -/
  radius_plus_height_eq_side : radius + height = Real.sqrt ((base / 2) ^ 2 + height ^ 2)

/-- The radius of the inscribed semicircle in the specific isosceles triangle -/
theorem semicircle_radius_in_specific_triangle :
  ∃ (t : IsoscelesTriangleWithSemicircle), t.base = 20 ∧ t.height = 12 ∧ t.radius = 12 := by
  sorry

end semicircle_radius_in_specific_triangle_l3998_399807


namespace line_through_point_l3998_399800

theorem line_through_point (a : ℚ) : 
  (3 * a * 2 + (2 * a + 3) * (-5) = 4 * a + 6) → a = -21 / 8 := by
  sorry

end line_through_point_l3998_399800


namespace exponent_multiplication_l3998_399887

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end exponent_multiplication_l3998_399887


namespace range_of_m_l3998_399855

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) →
  ((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0) = False) →
  ((m + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0) = True) →
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
by sorry

end range_of_m_l3998_399855


namespace min_distance_is_8_l3998_399841

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  (a - 2 * Real.exp a) / b = (1 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1

-- Define the distance function
def distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

-- Theorem statement
theorem min_distance_is_8 :
  ∀ a b c d : ℝ, condition a b c d → 
  ∀ x y z w : ℝ, condition x y z w →
  distance a b c d ≥ 8 ∧ (∃ a₀ b₀ c₀ d₀ : ℝ, condition a₀ b₀ c₀ d₀ ∧ distance a₀ b₀ c₀ d₀ = 8) :=
sorry

end min_distance_is_8_l3998_399841


namespace total_combinations_eq_twelve_l3998_399893

/-- The number of paint colors available. -/
def num_colors : ℕ := 4

/-- The number of painting methods available. -/
def num_methods : ℕ := 3

/-- The total number of combinations of paint color and painting method. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 12. -/
theorem total_combinations_eq_twelve : total_combinations = 12 := by
  sorry

end total_combinations_eq_twelve_l3998_399893


namespace new_savings_is_200_l3998_399808

/-- Calculates the new monthly savings after an increase in expenses -/
def new_monthly_savings (salary : ℚ) (initial_savings_rate : ℚ) (expense_increase_rate : ℚ) : ℚ :=
  let initial_expenses := salary * (1 - initial_savings_rate)
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

/-- Proves that the new monthly savings is 200 given the specified conditions -/
theorem new_savings_is_200 :
  new_monthly_savings 5000 (20 / 100) (20 / 100) = 200 := by
  sorry

end new_savings_is_200_l3998_399808


namespace arc_RS_range_l3998_399818

/-- An isosceles triangle with a rolling circle -/
structure RollingCircleTriangle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The radius of the rolling circle -/
  radius : ℝ
  /-- The position of the tangent point P along the base (0 ≤ p ≤ base) -/
  p : ℝ
  /-- The triangle is isosceles -/
  isosceles : altitude = base / 2
  /-- The altitude is twice the radius -/
  altitude_radius : altitude = 2 * radius
  /-- The tangent point is on the base -/
  p_on_base : 0 ≤ p ∧ p ≤ base

/-- The arc RS of the rolling circle -/
def arc_RS (t : RollingCircleTriangle) : ℝ := sorry

/-- Theorem: The arc RS varies from 90° to 180° -/
theorem arc_RS_range (t : RollingCircleTriangle) : 
  90 ≤ arc_RS t ∧ arc_RS t ≤ 180 := by sorry

end arc_RS_range_l3998_399818


namespace equation_has_integer_solution_l3998_399882

theorem equation_has_integer_solution (a b : ℤ) : ∃ x : ℤ, (x - a) * (x - b) * (x - 3) + 1 = 0 := by
  sorry

end equation_has_integer_solution_l3998_399882


namespace preimage_of_one_two_l3998_399810

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - 3 * p.2)

theorem preimage_of_one_two :
  f (1, 0) = (1, 2) := by
  sorry

end preimage_of_one_two_l3998_399810


namespace fixed_point_and_bisecting_line_l3998_399817

/-- The line equation ax - y + 2 + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

/-- The line equation 4x + y + 3 = 0 -/
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0

/-- The line equation 3x - 5y - 5 = 0 -/
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + 1 = 0 -/
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    (∃ t : ℝ, line_l1 (point_P.1 - t) (point_P.2 - t) ∧
              line_l2 (point_P.1 + t) (point_P.2 + t))) :=
by sorry

end fixed_point_and_bisecting_line_l3998_399817


namespace function_inequality_l3998_399835

/-- Given a function f(x) = x^2 - (a + 1/a)x + 1, if for any x in (1, 3),
    f(x) + (1/a)x > -3 always holds, then a < 4. -/
theorem function_inequality (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Ioo 1 3, x^2 - (a + 1/a)*x + 1 + (1/a)*x > -3) → a < 4 := by
  sorry

end function_inequality_l3998_399835


namespace max_value_of_a_l3998_399877

theorem max_value_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 6)
  (prod_sum_eq : a * b + a * c + b * c = 11) :
  a ≤ 2 + 2 * Real.sqrt 3 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 6 ∧ 
                    a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 11 ∧ 
                    a₀ = 2 + 2 * Real.sqrt 3 / 3 :=
by sorry

end max_value_of_a_l3998_399877


namespace sorting_abc_l3998_399838

theorem sorting_abc (a b c : Real)
  (ha : 0 ≤ a ∧ a ≤ Real.pi / 2)
  (hb : 0 ≤ b ∧ b ≤ Real.pi / 2)
  (hc : 0 ≤ c ∧ c ≤ Real.pi / 2)
  (ca : Real.cos a = a)
  (sb : Real.sin (Real.cos b) = b)
  (cs : Real.cos (Real.sin c) = c) :
  b < a ∧ a < c := by
sorry

end sorting_abc_l3998_399838


namespace mary_has_ten_more_than_marco_l3998_399864

/-- Calculates the difference in money between Mary and Marco after transactions. -/
def moneyDifference (marco_initial : ℕ) (mary_initial : ℕ) (mary_spent : ℕ) : ℕ :=
  let marco_gives := marco_initial / 2
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spent
  mary_final - marco_final

/-- Proves that Mary has $10 more than Marco after the described transactions. -/
theorem mary_has_ten_more_than_marco :
  moneyDifference 24 15 5 = 10 := by
  sorry

end mary_has_ten_more_than_marco_l3998_399864


namespace dvds_per_season_l3998_399890

theorem dvds_per_season (total_dvds : ℕ) (num_seasons : ℕ) 
  (h1 : total_dvds = 40) (h2 : num_seasons = 5) : 
  total_dvds / num_seasons = 8 := by
  sorry

end dvds_per_season_l3998_399890


namespace partial_fraction_decomposition_product_l3998_399895

theorem partial_fraction_decomposition_product (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (35 * x - 29) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -246 := by
sorry

end partial_fraction_decomposition_product_l3998_399895


namespace quadratic_equation_solution_l3998_399852

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -10 + 10 * Real.sqrt 2) ∧ 
              (x₂ = -10 - 10 * Real.sqrt 2) ∧ 
              (∀ x : ℝ, (10 - x)^2 = 2*x^2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_solution_l3998_399852


namespace inscribed_quadrilateral_relation_l3998_399823

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- Side length a -/
  a : ℝ
  /-- Side length b -/
  b : ℝ
  /-- Side length c -/
  c : ℝ
  /-- Side length d, which is also the diameter of the semicircle -/
  d : ℝ
  /-- All side lengths are positive -/
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  /-- The quadrilateral is inscribed in a semicircle with diameter d -/
  inscribed : True

/-- The main theorem about the relationship between side lengths of an inscribed quadrilateral -/
theorem inscribed_quadrilateral_relation (q : InscribedQuadrilateral) :
  q.d^3 - (q.a^2 + q.b^2 + q.c^2) * q.d - 2 * q.a * q.b * q.c = 0 := by
  sorry

end inscribed_quadrilateral_relation_l3998_399823


namespace larger_integer_value_l3998_399859

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * (b : ℕ) = 360) :
  max a b = 30 := by
  sorry

end larger_integer_value_l3998_399859


namespace helen_made_56_pies_l3998_399886

/-- The number of pies Helen made -/
def helen_pies (pinky_pies total_pies : ℕ) : ℕ := total_pies - pinky_pies

/-- Proof that Helen made 56 pies -/
theorem helen_made_56_pies : helen_pies 147 203 = 56 := by
  sorry

end helen_made_56_pies_l3998_399886


namespace parabola_shift_theorem_l3998_399806

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a - right,
    b := p.b - down }

theorem parabola_shift_theorem (p : Parabola) :
  shift_parabola { a := 2, b := 3 } 3 2 = { a := -1, b := 1 } :=
by sorry

end parabola_shift_theorem_l3998_399806


namespace solution_implies_m_value_l3998_399876

theorem solution_implies_m_value (m : ℚ) :
  (∀ x : ℚ, (m - 2) * x = 5 * (x + 1) → x = 2) →
  m = 19 / 2 := by
sorry

end solution_implies_m_value_l3998_399876


namespace simplify_expression_l3998_399867

theorem simplify_expression : 
  (Real.sqrt 6 - Real.sqrt 18) * Real.sqrt (1/3) + 2 * Real.sqrt 6 = Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end simplify_expression_l3998_399867


namespace gear_diameter_relation_l3998_399860

/-- Represents a circular gear with a diameter and revolutions per minute. -/
structure Gear where
  diameter : ℝ
  rpm : ℝ

/-- Represents a system of two interconnected gears. -/
structure GearSystem where
  gearA : Gear
  gearB : Gear
  /-- The gears travel at the same circumferential rate -/
  same_rate : gearA.diameter * gearA.rpm = gearB.diameter * gearB.rpm

/-- Theorem stating the relationship between gear diameters given their rpm ratio -/
theorem gear_diameter_relation (sys : GearSystem) 
  (h1 : sys.gearB.diameter = 50)
  (h2 : sys.gearA.rpm = 5 * sys.gearB.rpm) :
  sys.gearA.diameter = 10 := by
  sorry

end gear_diameter_relation_l3998_399860


namespace profit_ratio_from_investment_l3998_399847

/-- The profit ratio of two partners given their investment ratio and investment durations -/
theorem profit_ratio_from_investment 
  (p_investment q_investment : ℕ) 
  (p_duration q_duration : ℚ) 
  (h_investment_ratio : p_investment * 5 = q_investment * 7)
  (h_p_duration : p_duration = 5)
  (h_q_duration : q_duration = 11) :
  p_investment * p_duration * 11 = q_investment * q_duration * 7 :=
by sorry

end profit_ratio_from_investment_l3998_399847


namespace classes_taught_total_l3998_399828

/-- The number of classes Eduardo taught -/
def eduardo_classes : ℕ := 3

/-- The number of classes Frankie taught -/
def frankie_classes : ℕ := 2 * eduardo_classes

/-- The total number of classes taught by Eduardo and Frankie -/
def total_classes : ℕ := eduardo_classes + frankie_classes

theorem classes_taught_total : total_classes = 9 := by
  sorry

end classes_taught_total_l3998_399828


namespace sum_of_coordinates_of_symmetric_points_l3998_399842

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- Given that point P(x,1) is symmetric to point Q(-3,y) with respect to the origin, prove that x + y = 2 -/
theorem sum_of_coordinates_of_symmetric_points :
  ∀ x y : ℝ, symmetric_wrt_origin (x, 1) (-3, y) → x + y = 2 :=
by sorry

end sum_of_coordinates_of_symmetric_points_l3998_399842


namespace chairs_in_clubroom_l3998_399813

/-- Represents the number of chairs in the clubroom -/
def num_chairs : ℕ := 17

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs the table has -/
def table_legs : ℕ := 3

/-- Represents the number of unoccupied chairs -/
def unoccupied_chairs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 101

/-- Proves that the number of chairs in the clubroom is correct given the conditions -/
theorem chairs_in_clubroom :
  num_chairs * legs_per_chair + table_legs = total_legs + 2 * (num_chairs - unoccupied_chairs) :=
by sorry

end chairs_in_clubroom_l3998_399813


namespace circle_fixed_points_l3998_399856

theorem circle_fixed_points (m : ℝ) :
  let circle := λ (x y : ℝ) => x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2
  circle 1 1 = 0 ∧ circle (1/5) (7/5) = 0 := by
  sorry

end circle_fixed_points_l3998_399856


namespace product_minus_third_lower_bound_l3998_399870

theorem product_minus_third_lower_bound 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (a : ℝ) 
  (h1 : x * y - z = a) 
  (h2 : y * z - x = a) 
  (h3 : z * x - y = a) : 
  a ≥ -1/4 := by
sorry

end product_minus_third_lower_bound_l3998_399870


namespace front_axle_wheels_l3998_399866

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) : ℚ :=
  3.5 + 0.5 * (x - 2)

/-- The number of axles for an 18-wheel truck with f wheels on the front axle -/
def num_axles (f : ℕ) : ℕ :=
  1 + (18 - f) / 4

theorem front_axle_wheels :
  ∃ (f : ℕ), f > 0 ∧ f < 18 ∧ 
  toll (num_axles f) = 5 ∧
  f = 2 := by
  sorry

end front_axle_wheels_l3998_399866


namespace part1_part2_l3998_399862

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) (-1) ↔ f x a - |x - a| ≤ 2) →
  a = 2 := by sorry

-- Part 2
theorem part2 (a m : ℝ) :
  (∃ x₀, f x₀ a < 4 * m + m^2) →
  m ∈ Set.Ioi 1 ∪ Set.Iio (-5) := by sorry

end part1_part2_l3998_399862


namespace time_spent_calculation_susan_time_allocation_l3998_399834

/-- Given a ratio of activities and time spent on one activity, calculate the time spent on another activity -/
theorem time_spent_calculation (reading_ratio : ℕ) (hangout_ratio : ℕ) (reading_hours : ℕ) 
  (h1 : reading_ratio > 0)
  (h2 : hangout_ratio > 0)
  (h3 : reading_hours > 0) :
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = hangout_ratio * reading_hours :=
by sorry

/-- Susan's time allocation problem -/
theorem susan_time_allocation :
  let reading_ratio : ℕ := 4
  let hangout_ratio : ℕ := 10
  let reading_hours : ℕ := 8
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = 20 :=
by sorry

end time_spent_calculation_susan_time_allocation_l3998_399834


namespace total_teaching_time_l3998_399805

/-- Represents a teacher's class schedule -/
structure Schedule where
  math_classes : ℕ
  science_classes : ℕ
  history_classes : ℕ
  math_duration : ℝ
  science_duration : ℝ
  history_duration : ℝ

/-- Calculates the total teaching time for a given schedule -/
def total_time (s : Schedule) : ℝ :=
  s.math_classes * s.math_duration +
  s.science_classes * s.science_duration +
  s.history_classes * s.history_duration

/-- Eduardo's teaching schedule -/
def eduardo : Schedule :=
  { math_classes := 3
    science_classes := 4
    history_classes := 2
    math_duration := 1
    science_duration := 1.5
    history_duration := 2 }

/-- Frankie's teaching schedule (double of Eduardo's) -/
def frankie : Schedule :=
  { math_classes := 2 * eduardo.math_classes
    science_classes := 2 * eduardo.science_classes
    history_classes := 2 * eduardo.history_classes
    math_duration := eduardo.math_duration
    science_duration := eduardo.science_duration
    history_duration := eduardo.history_duration }

/-- Theorem: The total teaching time for Eduardo and Frankie is 39 hours -/
theorem total_teaching_time : total_time eduardo + total_time frankie = 39 := by
  sorry


end total_teaching_time_l3998_399805


namespace defect_rate_two_procedures_l3998_399879

/-- The defect rate of a product after two independent procedures -/
def overall_defect_rate (a b : ℝ) : ℝ := 1 - (1 - a) * (1 - b)

/-- Theorem: The overall defect rate of a product after two independent procedures
    with defect rates a and b is 1 - (1-a)(1-b) -/
theorem defect_rate_two_procedures
  (a b : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  : overall_defect_rate a b = 1 - (1 - a) * (1 - b) :=
by sorry

end defect_rate_two_procedures_l3998_399879


namespace ellipse_parameter_range_l3998_399874

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) + y^2 / (1 - m) = 1

/-- Conditions for the equation to represent an ellipse with foci on the x-axis -/
def is_valid_ellipse (m : ℝ) : Prop :=
  2 + m > 0 ∧ 1 - m > 0 ∧ 2 + m > 1 - m

/-- The range of m for which the equation represents a valid ellipse -/
theorem ellipse_parameter_range :
  ∀ m : ℝ, is_valid_ellipse m ↔ -1/2 < m ∧ m < 1 :=
sorry

end ellipse_parameter_range_l3998_399874


namespace distance_from_two_is_six_l3998_399863

theorem distance_from_two_is_six (x : ℝ) : |x - 2| = 6 → x = 8 ∨ x = -4 := by
  sorry

end distance_from_two_is_six_l3998_399863


namespace algebraic_expression_equality_l3998_399836

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + x + 3 = 7 → 3*x^2 + 3*x + 7 = 19 := by
  sorry

end algebraic_expression_equality_l3998_399836


namespace x_minus_p_equals_two_l3998_399845

theorem x_minus_p_equals_two (x p : ℝ) (h1 : |x - 2| = p) (h2 : x > 2) : x - p = 2 := by
  sorry

end x_minus_p_equals_two_l3998_399845


namespace six_throws_total_skips_l3998_399803

def stone_skips (n : ℕ) : ℕ := n^2 + n

def total_skips (num_throws : ℕ) : ℕ :=
  (List.range num_throws).map stone_skips |>.sum

theorem six_throws_total_skips :
  total_skips 5 + 2 * stone_skips 6 = 154 := by
  sorry

end six_throws_total_skips_l3998_399803


namespace larger_number_proof_l3998_399848

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
sorry

end larger_number_proof_l3998_399848


namespace long_distance_call_cost_decrease_l3998_399826

/-- The percent decrease in cost of a long-distance call --/
def percent_decrease (initial_cost final_cost : ℚ) : ℚ :=
  (initial_cost - final_cost) / initial_cost * 100

/-- Theorem: The percent decrease from 35 cents to 5 cents is approximately 86% --/
theorem long_distance_call_cost_decrease :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |percent_decrease (35/100) (5/100) - 86| < ε :=
sorry

end long_distance_call_cost_decrease_l3998_399826


namespace max_value_theorem_l3998_399888

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := -(Real.log x) / x

theorem max_value_theorem (x₁ x₂ t : ℝ) (h1 : f x₁ = t) (h2 : g x₂ = t) (h3 : t > 0) :
  (∀ y₁ y₂ s : ℝ, f y₁ = s → g y₂ = s → s > 0 → y₁ / (y₂ * Real.exp s) ≤ 1 / Real.exp 1) ∧
  (∃ z₁ z₂ r : ℝ, f z₁ = r ∧ g z₂ = r ∧ r > 0 ∧ z₁ / (z₂ * Real.exp r) = 1 / Real.exp 1) :=
sorry

end max_value_theorem_l3998_399888


namespace mrs_hilt_apples_per_hour_l3998_399821

/-- Given a total number of apples and hours, calculate the apples eaten per hour -/
def apples_per_hour (total_apples : ℕ) (total_hours : ℕ) : ℚ :=
  total_apples / total_hours

/-- Theorem: Mrs. Hilt ate 5 apples per hour -/
theorem mrs_hilt_apples_per_hour :
  apples_per_hour 15 3 = 5 := by
  sorry

end mrs_hilt_apples_per_hour_l3998_399821


namespace square_perimeter_when_area_equals_diagonal_l3998_399815

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → 
  s^2 = s * Real.sqrt 2 → 
  4 * s = 4 * Real.sqrt 2 :=
by
  sorry

end square_perimeter_when_area_equals_diagonal_l3998_399815


namespace l₁_passes_through_neg_one_neg_one_perpendicular_condition_l3998_399878

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def l₂ (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

-- Theorem 1: l₁ passes through (-1, -1) for all a
theorem l₁_passes_through_neg_one_neg_one (a : ℝ) : l₁ a (-1) (-1) := by sorry

-- Theorem 2: If l₁ ⊥ l₂, then a = 0 or a = -4
theorem perpendicular_condition (a : ℝ) : 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) → 
  a = 0 ∨ a = -4 := by sorry

end l₁_passes_through_neg_one_neg_one_perpendicular_condition_l3998_399878


namespace f_has_unique_zero_and_g_max_a_l3998_399846

noncomputable def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

noncomputable def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_has_unique_zero_and_g_max_a :
  (∃! x : ℝ, x ≥ 1 ∧ f x = 0) ∧
  (∀ a : ℝ, a > 6 → ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ g a x₂ < g a x₁) ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → g 6 x₁ ≤ g 6 x₂) :=
by sorry

end f_has_unique_zero_and_g_max_a_l3998_399846


namespace gcd_of_product_form_l3998_399851

def product_form (a b c d : ℤ) : ℤ :=
  (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b)

theorem gcd_of_product_form :
  ∃ (g : ℤ), g > 0 ∧ 
  (∀ (a b c d : ℤ), g ∣ product_form a b c d) ∧
  (∀ (h : ℤ), h > 0 → (∀ (a b c d : ℤ), h ∣ product_form a b c d) → h ∣ g) ∧
  g = 12 := by
  sorry

end gcd_of_product_form_l3998_399851


namespace intersection_condition_l3998_399812

theorem intersection_condition (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  (∃! x, x ∈ A ∩ B) → 0 ≤ a ∧ a < 1 :=
by
  sorry

end intersection_condition_l3998_399812


namespace two_white_balls_probability_l3998_399853

/-- The probability of drawing two white balls consecutively without replacement -/
theorem two_white_balls_probability 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) : 
  (white_balls : ℚ) / total_balls * ((white_balls - 1) : ℚ) / (total_balls - 1) = 5 / 14 := by
  sorry

end two_white_balls_probability_l3998_399853


namespace triangle_max_perimeter_l3998_399894

/-- Given a triangle ABC where angle A is 60° and side a is 4, 
    the maximum perimeter of the triangle is 12. -/
theorem triangle_max_perimeter (b c : ℝ) : 
  let A : ℝ := 60 * π / 180  -- Convert 60° to radians
  let a : ℝ := 4
  b > 0 → c > 0 →   -- Ensure positive side lengths
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine theorem
  a + b + c ≤ 12 :=
by sorry

end triangle_max_perimeter_l3998_399894


namespace combinations_equal_fifteen_l3998_399814

/-- The number of window treatment types available. -/
def num_treatments : ℕ := 3

/-- The number of colors available. -/
def num_colors : ℕ := 5

/-- The total number of combinations of window treatment type and color. -/
def total_combinations : ℕ := num_treatments * num_colors

/-- Theorem stating that the total number of combinations is 15. -/
theorem combinations_equal_fifteen : total_combinations = 15 := by
  sorry

end combinations_equal_fifteen_l3998_399814


namespace field_division_l3998_399801

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 := by
sorry

end field_division_l3998_399801


namespace unique_positive_number_l3998_399861

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 * (1 / x) := by
  sorry

end unique_positive_number_l3998_399861


namespace min_sum_of_equal_multiples_l3998_399837

theorem min_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (m : ℕ+), ∀ (a b c : ℕ+), ((4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val) →
    m.val ≤ a.val + b.val + c.val ∧ m.val = x.val + y.val + z.val ∧ m.val = 37 :=
sorry

end min_sum_of_equal_multiples_l3998_399837


namespace percent_swap_l3998_399850

theorem percent_swap (x : ℝ) (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 := by
  sorry

end percent_swap_l3998_399850


namespace sqrt_identity_in_range_l3998_399819

theorem sqrt_identity_in_range (θ : Real) (h : θ ∈ Set.Ioo (7 * Real.pi / 4) (2 * Real.pi)) :
  Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) = Real.cos θ - Real.sin θ := by
  sorry

end sqrt_identity_in_range_l3998_399819


namespace hyperbola_equation_l3998_399873

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

-- Define the axis of the parabola
def parabola_axis (p : ℝ) (x : ℝ) : Prop :=
  x = -p / 2

-- Theorem statement
theorem hyperbola_equation (a b p : ℝ) :
  a > 0 ∧ b > 0 ∧ p > 0 ∧
  (∃ x₀ y₀, asymptote a b x₀ y₀ ∧ parabola_axis p x₀ ∧ x₀ = -2 ∧ y₀ = -4) ∧
  (∃ x₁ y₁ x₂ y₂, hyperbola a b x₁ y₁ ∧ x₁ = -a ∧ y₁ = 0 ∧
                  parabola p x₂ y₂ ∧ x₂ = p ∧ y₂ = 0 ∧
                  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  a = 2 ∧ b = 1 :=
by sorry

end hyperbola_equation_l3998_399873


namespace total_students_l3998_399843

/-- Represents the age groups in the school -/
inductive AgeGroup
  | Below8
  | Exactly8
  | Between9And10
  | Above10

/-- Represents the school with its student distribution -/
structure School where
  totalStudents : ℕ
  ageDistribution : AgeGroup → ℚ
  exactly8Count : ℕ

/-- The conditions of the problem -/
def schoolConditions (s : School) : Prop :=
  s.ageDistribution AgeGroup.Below8 = 1/5 ∧
  s.ageDistribution AgeGroup.Exactly8 = 1/4 ∧
  s.ageDistribution AgeGroup.Between9And10 = 7/20 ∧
  s.ageDistribution AgeGroup.Above10 = 1/5 ∧
  s.exactly8Count = 15

/-- The theorem to prove -/
theorem total_students (s : School) (h : schoolConditions s) : s.totalStudents = 60 := by
  sorry

end total_students_l3998_399843


namespace largest_pot_cost_is_correct_l3998_399884

/-- The cost of the largest pot given 6 pots with increasing prices -/
def largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_difference : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_difference * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_difference * (num_pots - 1)

/-- Theorem stating the cost of the largest pot -/
theorem largest_pot_cost_is_correct : 
  largest_pot_cost 6 (39/5) (1/4) = 77/40 := by
  sorry

#eval largest_pot_cost 6 (39/5) (1/4)

end largest_pot_cost_is_correct_l3998_399884


namespace lindsay_dolls_theorem_l3998_399896

theorem lindsay_dolls_theorem (blonde : ℕ) (brown : ℕ) (black : ℕ) : 
  blonde = 4 →
  brown = 4 * blonde →
  black = brown - 2 →
  brown + black - blonde = 26 := by
  sorry

end lindsay_dolls_theorem_l3998_399896


namespace normal_distribution_symmetry_l3998_399881

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The cumulative distribution function (CDF) for a normal distribution -/
noncomputable def normalCDF (nd : NormalDistribution) (x : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (nd : NormalDistribution) 
  (h_mean : nd.μ = 85) 
  (h_cdf : normalCDF nd 122 = 0.96) :
  normalCDF nd 48 = 0.04 :=
sorry

end normal_distribution_symmetry_l3998_399881


namespace cos_75_degrees_l3998_399899

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l3998_399899


namespace path_count_theorem_l3998_399854

def grid_path (right up : ℕ) : ℕ := Nat.choose (right + up) up

theorem path_count_theorem :
  let right : ℕ := 6
  let up : ℕ := 4
  let total_path_length : ℕ := right + up
  grid_path right up = 210 := by
  sorry

end path_count_theorem_l3998_399854


namespace square_area_from_perimeter_l3998_399802

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4) ^ 2 → area = 100 := by
sorry

end square_area_from_perimeter_l3998_399802


namespace digital_root_of_1999_factorial_l3998_399869

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The digital root function -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 1) % 9

/-- Theorem: The digital root of 1999! is 9 -/
theorem digital_root_of_1999_factorial :
  digitalRoot (factorial 1999) = 9 := by
  sorry

end digital_root_of_1999_factorial_l3998_399869


namespace arithmetic_sequence_ratio_l3998_399833

theorem arithmetic_sequence_ratio (a d : ℝ) : 
  (a + d) + (a + 3*d) = 6*a ∧ 
  a + 2*d = 10 →
  a / (a + 3*d) = 1/4 := by sorry

end arithmetic_sequence_ratio_l3998_399833


namespace necessary_but_not_sufficient_l3998_399811

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  ¬(∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) :=
by sorry

end necessary_but_not_sufficient_l3998_399811


namespace no_positive_integer_solutions_l3998_399858

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ+), x^2 + y^2 = x^4 := by sorry

end no_positive_integer_solutions_l3998_399858


namespace sum_of_factors_of_30_l3998_399829

def factors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by sorry

end sum_of_factors_of_30_l3998_399829


namespace eric_apples_l3998_399830

theorem eric_apples (r y g : ℕ) : 
  r = y →                           -- Red apples = Yellow apples (in first box)
  r = (1/3 : ℚ) * (r + g : ℚ) →     -- Red apples are 1/3 of second box after moving
  r + y + g = 28 →                  -- Total number of apples
  r = 7 := by sorry

end eric_apples_l3998_399830


namespace not_square_n5_plus_7_l3998_399898

theorem not_square_n5_plus_7 (n : ℤ) (h : n > 1) : ¬ ∃ k : ℤ, n^5 + 7 = k^2 := by
  sorry

end not_square_n5_plus_7_l3998_399898


namespace distance_circle_center_to_line_l3998_399857

/-- Given a circle with polar equation ρ = 4sin(θ) and a line with parametric equation x = √3t, y = t,
    the distance from the center of the circle to the line is √3. -/
theorem distance_circle_center_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4*y}
  let line := {(x, y) : ℝ × ℝ | ∃ t : ℝ, x = Real.sqrt 3 * t ∧ y = t}
  let circle_center := (0, 2)
  ∃ p ∈ line, Real.sqrt ((circle_center.1 - p.1)^2 + (circle_center.2 - p.2)^2) = Real.sqrt 3 :=
by sorry

end distance_circle_center_to_line_l3998_399857


namespace largest_angle_in_special_triangle_l3998_399889

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a + b + 2c = a² and a + b - 2c = -1, then the largest angle is 120°. -/
theorem largest_angle_in_special_triangle (a b c : ℝ) (h1 : a + b + 2*c = a^2) (h2 : a + b - 2*c = -1) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
    A + B + C = Real.pi ∧    -- Sum of angles in a triangle
    max A (max B C) = 2*Real.pi/3 :=  -- Largest angle is 120°
by sorry

end largest_angle_in_special_triangle_l3998_399889


namespace inverse_between_zero_and_one_l3998_399880

theorem inverse_between_zero_and_one (x : ℝ) : 0 < (1 : ℝ) / x ∧ (1 : ℝ) / x < 1 ↔ x > 1 := by
  sorry

end inverse_between_zero_and_one_l3998_399880


namespace sequence_may_or_may_not_be_arithmetic_l3998_399865

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- The first five terms of the sequence are 1, 2, 3, 4, 5. -/
def first_five_terms (s : Sequence) : Prop :=
  s 0 = 1 ∧ s 1 = 2 ∧ s 2 = 3 ∧ s 3 = 4 ∧ s 4 = 5

theorem sequence_may_or_may_not_be_arithmetic :
  ∃ s₁ s₂ : Sequence, first_five_terms s₁ ∧ first_five_terms s₂ ∧
    is_arithmetic s₁ ∧ ¬is_arithmetic s₂ := by
  sorry

end sequence_may_or_may_not_be_arithmetic_l3998_399865


namespace base_equivalence_l3998_399849

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (x y : Nat) : Nat :=
  x * 6 + y

/-- Converts a number in base b to base 10 -/
def baseBToBase10 (b x y z : Nat) : Nat :=
  x * b^2 + y * b + z

theorem base_equivalence :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 5 3 = baseBToBase10 b 1 1 3 :=
by sorry

end base_equivalence_l3998_399849


namespace michaels_matchsticks_l3998_399872

/-- The number of matchsticks Michael had originally -/
def original_matchsticks : ℕ := 1700

/-- The number of houses Michael created -/
def houses : ℕ := 30

/-- The number of towers Michael created -/
def towers : ℕ := 20

/-- The number of bridges Michael created -/
def bridges : ℕ := 10

/-- The number of matchsticks used for each house -/
def matchsticks_per_house : ℕ := 10

/-- The number of matchsticks used for each tower -/
def matchsticks_per_tower : ℕ := 15

/-- The number of matchsticks used for each bridge -/
def matchsticks_per_bridge : ℕ := 25

/-- Theorem stating that Michael's original pile of matchsticks was 1700 -/
theorem michaels_matchsticks :
  original_matchsticks = 2 * (houses * matchsticks_per_house +
                              towers * matchsticks_per_tower +
                              bridges * matchsticks_per_bridge) :=
by sorry

end michaels_matchsticks_l3998_399872


namespace count_integers_eq_880_l3998_399840

/-- Fibonacci sequence with F₁ = 2 and F₂ = 3 -/
def F : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => F (n + 1) + F n

/-- The number of 10-digit integers with digits 1 or 2 and two consecutive 1's -/
def count_integers : ℕ := 2^10 - F 9

theorem count_integers_eq_880 : count_integers = 880 := by
  sorry

#eval count_integers  -- Should output 880

end count_integers_eq_880_l3998_399840


namespace similar_triangles_leg_length_l3998_399825

theorem similar_triangles_leg_length (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = 12 → b = 9 → c = 7.5 →
  a / c = b / d →
  d = 5.625 := by
sorry

end similar_triangles_leg_length_l3998_399825


namespace unique_solution_l3998_399822

structure Grid :=
  (a b c : ℕ)
  (row_sum : ℕ)
  (col_sum : ℕ)

def is_valid_grid (g : Grid) : Prop :=
  g.row_sum = 9 ∧
  g.col_sum = 12 ∧
  g.a + g.b + g.c = g.row_sum ∧
  4 + g.a + 1 + g.b = g.col_sum ∧
  g.a + 2 + 6 = g.col_sum ∧
  3 + 1 + 6 + g.c = g.col_sum ∧
  g.b + 2 + g.c = g.row_sum

theorem unique_solution :
  ∃! g : Grid, is_valid_grid g ∧ g.a = 6 ∧ g.b = 5 ∧ g.c = 2 :=
sorry

end unique_solution_l3998_399822


namespace one_quarter_of_seven_point_two_l3998_399827

theorem one_quarter_of_seven_point_two : 
  (7.2 / 4 : ℚ) = 9 / 5 := by sorry

end one_quarter_of_seven_point_two_l3998_399827


namespace used_cd_cost_correct_l3998_399844

/-- The cost of Lakota's purchase -/
def lakota_cost : ℝ := 127.92

/-- The cost of Mackenzie's purchase -/
def mackenzie_cost : ℝ := 133.89

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The cost of a single used CD -/
def used_cd_cost : ℝ := 9.99

theorem used_cd_cost_correct :
  ∃ (new_cd_cost : ℝ),
    lakota_new * new_cd_cost + lakota_used * used_cd_cost = lakota_cost ∧
    mackenzie_new * new_cd_cost + mackenzie_used * used_cd_cost = mackenzie_cost :=
by sorry

end used_cd_cost_correct_l3998_399844


namespace prob_at_least_one_even_is_five_ninths_l3998_399832

/-- A set of cards labeled 1, 2, and 3 -/
def cards : Finset ℕ := {1, 2, 3}

/-- The event of drawing an even number -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- The sample space of two draws with replacement -/
def sample_space : Finset (ℕ × ℕ) :=
  (cards.product cards)

/-- The favorable outcomes (at least one even number) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p => is_even p.1 ∨ is_even p.2)

/-- The probability of drawing at least one even number in two draws -/
def prob_at_least_one_even : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem prob_at_least_one_even_is_five_ninths :
  prob_at_least_one_even = 5 / 9 := by sorry

end prob_at_least_one_even_is_five_ninths_l3998_399832
