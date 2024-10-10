import Mathlib

namespace min_selling_price_theorem_l128_12819

/-- Represents the fruit shop scenario with two batches of fruits. -/
structure FruitShop where
  batch1_price : ℝ  -- Price per kg of first batch
  batch1_quantity : ℝ  -- Quantity of first batch in kg
  batch2_price : ℝ  -- Price per kg of second batch
  batch2_quantity : ℝ  -- Quantity of second batch in kg

/-- Calculates the minimum selling price for remaining fruits to achieve the target profit. -/
def min_selling_price (shop : FruitShop) (target_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum selling price for the given scenario. -/
theorem min_selling_price_theorem (shop : FruitShop) :
  shop.batch1_price = 50 ∧
  shop.batch2_price = 55 ∧
  shop.batch1_quantity * shop.batch1_price = 1100 ∧
  shop.batch2_quantity * shop.batch2_price = 1100 ∧
  shop.batch1_quantity = shop.batch2_quantity + 2 ∧
  shop.batch2_price = 1.1 * shop.batch1_price →
  min_selling_price shop 1000 = 60 :=
by sorry

end min_selling_price_theorem_l128_12819


namespace complex_sum_example_l128_12809

theorem complex_sum_example (z₁ z₂ : ℂ) : 
  z₁ = 2 + 3*I ∧ z₂ = -4 - 5*I → z₁ + z₂ = -2 - 2*I := by
  sorry

end complex_sum_example_l128_12809


namespace circle_tangent_to_line_l128_12852

/-- A circle with center (1, 0) and radius √m is tangent to the line x + y = 1 if and only if m = 1/2 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = m ∧ x + y = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = m → x' + y' ≥ 1) ↔ 
  m = 1/2 :=
sorry

end circle_tangent_to_line_l128_12852


namespace power_of_negative_product_l128_12834

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end power_of_negative_product_l128_12834


namespace line_through_point_l128_12830

theorem line_through_point (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = 5 * x + a → (x = a → y = a^2)) → a = 6 := by
  sorry

end line_through_point_l128_12830


namespace student_marks_l128_12876

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (marks_obtained : ℕ) : 
  total_marks = 800 →
  passing_percentage = 33 / 100 →
  failed_by = 89 →
  marks_obtained = total_marks * passing_percentage - failed_by →
  marks_obtained = 175 := by
sorry

#eval (800 : ℕ) * (33 : ℚ) / 100 - 89  -- Expected output: 175

end student_marks_l128_12876


namespace extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l128_12887

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- Part I
theorem extreme_values_when_a_is_4 :
  let f := f 4
  (∃ x, ∀ y, f y ≤ f x) ∧ (∃ x, ∀ y, f y ≥ f x) ∧
  (∀ x, f x ≤ 1) ∧ (∀ x, f x ≥ -1) :=
sorry

-- Part II
theorem a_range_when_f_geq_4_on_interval :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) → a ≥ 7 :=
sorry

end extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l128_12887


namespace tickets_found_is_zero_l128_12804

/-- The number of carnival games --/
def num_games : ℕ := 5

/-- The value of each ticket in dollars --/
def ticket_value : ℕ := 3

/-- The total value of all tickets in dollars --/
def total_value : ℕ := 30

/-- The number of tickets won from each game --/
def tickets_per_game : ℕ := total_value / (num_games * ticket_value)

/-- The number of tickets found on the floor --/
def tickets_found : ℕ := total_value - (num_games * tickets_per_game * ticket_value)

theorem tickets_found_is_zero : tickets_found = 0 := by
  sorry

end tickets_found_is_zero_l128_12804


namespace same_terminal_side_angle_l128_12884

theorem same_terminal_side_angle : ∃ (θ : ℝ), 
  θ ∈ Set.Icc (-2 * Real.pi) 0 ∧ 
  ∃ (k : ℤ), θ = (52 / 7 : ℝ) * Real.pi + 2 * k * Real.pi ∧
  θ = -(4 / 7 : ℝ) * Real.pi := by
  sorry

end same_terminal_side_angle_l128_12884


namespace tax_percentage_proof_l128_12801

/-- 
Given:
- total_income: The total annual income
- after_tax_income: The income left after paying taxes

Prove that the percentage of income paid in taxes is 18%
-/
theorem tax_percentage_proof (total_income after_tax_income : ℝ) 
  (h1 : total_income = 60000)
  (h2 : after_tax_income = 49200) :
  (total_income - after_tax_income) / total_income * 100 = 18 := by
  sorry


end tax_percentage_proof_l128_12801


namespace quadratic_equation_result_l128_12869

theorem quadratic_equation_result (m : ℝ) (h : m^2 + 2*m = 3) : 4*m^2 + 8*m - 1 = 11 := by
  sorry

end quadratic_equation_result_l128_12869


namespace amoeba_count_after_ten_days_l128_12853

/-- The number of amoebas in the puddle after n days -/
def amoeba_count (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba growth process continues -/
def days : ℕ := 10

/-- Theorem: The number of amoebas after 10 days is 59049 -/
theorem amoeba_count_after_ten_days :
  amoeba_count days = 59049 := by
  sorry

end amoeba_count_after_ten_days_l128_12853


namespace intersecting_lines_determine_plane_l128_12821

-- Define the concepts of point, line, and plane
variable (Point Line Plane : Type)

-- Define the concept of intersection for lines
variable (intersect : Line → Line → Prop)

-- Define the concept of a line lying on a plane
variable (lieOn : Line → Plane → Prop)

-- Define the concept of a plane containing two lines
variable (contains : Plane → Line → Line → Prop)

-- Theorem: Two intersecting lines determine a unique plane
theorem intersecting_lines_determine_plane
  (l1 l2 : Line)
  (h_intersect : intersect l1 l2)
  : ∃! p : Plane, contains p l1 l2 :=
sorry

end intersecting_lines_determine_plane_l128_12821


namespace inequality_proof_equality_condition_l128_12873

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end inequality_proof_equality_condition_l128_12873


namespace trigonometric_identities_l128_12883

theorem trigonometric_identities (α : ℝ) 
  (h : (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7) :
  (Real.tan (π / 2 - α) = 1 / 2) ∧
  (3 * Real.cos α * Real.sin (α + π) + 2 * (Real.cos (α + π / 2))^2 = 2 / 5) := by
  sorry

end trigonometric_identities_l128_12883


namespace range_of_x₀_l128_12835

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
def point_M (x₀ : ℝ) : ℝ × ℝ := (x₀, 2 - x₀)

-- Define the angle OMN
def angle_OMN (O M N : ℝ × ℝ) : ℝ := sorry

-- Define the existence of point N on circle O
def exists_N (x₀ : ℝ) : Prop :=
  ∃ N : ℝ × ℝ, circle_O N.1 N.2 ∧ angle_OMN (0, 0) (point_M x₀) N = 30

-- Theorem statement
theorem range_of_x₀ (x₀ : ℝ) :
  exists_N x₀ → 0 ≤ x₀ ∧ x₀ ≤ 2 :=
sorry

end range_of_x₀_l128_12835


namespace isosceles_triangle_perimeter_l128_12818

-- Define an isosceles triangle with side lengths 3 and 5
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 5 ∧ (a = c ∨ b = c)) ∨ (a = 5 ∧ b = 3 ∧ (a = c ∨ b = c))

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → (Perimeter a b c = 11 ∨ Perimeter a b c = 13) :=
by sorry

end isosceles_triangle_perimeter_l128_12818


namespace sin_225_degrees_l128_12877

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l128_12877


namespace taxi_fare_calculation_l128_12841

/-- A taxi fare system with a fixed starting fee and a proportional amount per mile -/
structure TaxiFare where
  startingFee : ℝ
  costPerMile : ℝ

/-- Calculate the total fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.costPerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : calculateFare tf 60 = 150)
  : calculateFare tf 80 = 193.33 := by
  sorry

end taxi_fare_calculation_l128_12841


namespace coin_distribution_l128_12802

theorem coin_distribution (x y : ℕ) : 
  x + y = 16 → 
  x^2 - y^2 = 16 * (x - y) → 
  x = 8 ∧ y = 8 := by
  sorry

end coin_distribution_l128_12802


namespace larger_number_of_sum_and_difference_l128_12845

theorem larger_number_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 45) 
  (diff_eq : x - y = 7) : 
  max x y = 26 := by
sorry

end larger_number_of_sum_and_difference_l128_12845


namespace smallest_n_for_sqrt_diff_smallest_positive_integer_l128_12803

theorem smallest_n_for_sqrt_diff (n : ℕ) : n ≥ 10001 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.005 := by
  sorry

theorem smallest_positive_integer : ∀ m : ℕ, m < 10001 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.005 := by
  sorry

end smallest_n_for_sqrt_diff_smallest_positive_integer_l128_12803


namespace gcf_252_96_l128_12882

theorem gcf_252_96 : Nat.gcd 252 96 = 12 := by
  sorry

end gcf_252_96_l128_12882


namespace group_size_proof_l128_12864

theorem group_size_proof (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total = (28 : ℚ) / 100 * total + 96) 
  (h2 : (28 : ℚ) / 100 * total = total - ((2 : ℚ) / 5 * total - 96)) : 
  total = 800 := by
sorry

end group_size_proof_l128_12864


namespace sector_radius_l128_12865

/-- Given a sector with a central angle of 150° and an arc length of 5π/2 cm, its radius is 3 cm. -/
theorem sector_radius (θ : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  θ = 150 → 
  arc_length = (5/2) * Real.pi → 
  arc_length = (θ / 360) * 2 * Real.pi * radius → 
  radius = 3 := by
sorry

end sector_radius_l128_12865


namespace number_of_workers_l128_12815

theorem number_of_workers (
  avg_salary_with_first_supervisor : ℝ)
  (first_supervisor_salary : ℝ)
  (avg_salary_with_new_supervisor : ℝ)
  (new_supervisor_salary : ℝ)
  (h1 : avg_salary_with_first_supervisor = 430)
  (h2 : first_supervisor_salary = 870)
  (h3 : avg_salary_with_new_supervisor = 440)
  (h4 : new_supervisor_salary = 960) :
  ∃ (w : ℕ), w = 8 ∧
  (w + 1) * avg_salary_with_first_supervisor - first_supervisor_salary =
  9 * avg_salary_with_new_supervisor - new_supervisor_salary :=
by sorry

end number_of_workers_l128_12815


namespace angle_trig_sum_l128_12860

theorem angle_trig_sum (a : ℝ) (ha : a ≠ 0) :
  let α := Real.arctan (3*a / (-4*a))
  if a > 0 then
    Real.sin α + Real.cos α - Real.tan α = 11/20
  else
    Real.sin α + Real.cos α - Real.tan α = 19/20 := by
  sorry

end angle_trig_sum_l128_12860


namespace increasing_symmetric_function_inequality_l128_12832

theorem increasing_symmetric_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x ≥ 2 → y ≥ 2 → x < y → f x < f y) 
  (h_symmetric : ∀ x, f (2 + x) = f (2 - x)) 
  (h_inequality : f (1 - 2 * x^2) < f (1 + 2*x - x^2)) :
  -2 < x ∧ x < 0 :=
by sorry

end increasing_symmetric_function_inequality_l128_12832


namespace derivative_f_at_pi_over_2_l128_12838

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_over_2 :
  deriv f (π / 2) = 1 := by sorry

end derivative_f_at_pi_over_2_l128_12838


namespace ipod_problem_l128_12890

def problem (emmy_initial : ℕ) (emmy_lost : ℕ) (rosa_given_away : ℕ) : Prop :=
  let emmy_current := emmy_initial - emmy_lost
  let rosa_current := emmy_current / 3
  let rosa_initial := rosa_current + rosa_given_away
  emmy_current + rosa_current = 21

theorem ipod_problem : problem 25 9 4 := by
  sorry

end ipod_problem_l128_12890


namespace complex_modulus_problem_l128_12856

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I * z) = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l128_12856


namespace ships_initial_distance_l128_12874

/-- The initial distance between two ships moving towards a port -/
def initial_distance : ℝ := 240

/-- The distance traveled by the second ship when a right triangle is formed -/
def right_triangle_distance : ℝ := 80

/-- The remaining distance for the second ship when the first ship reaches the port -/
def remaining_distance : ℝ := 120

theorem ships_initial_distance :
  ∃ (v₁ v₂ : ℝ), v₁ > 0 ∧ v₂ > 0 ∧
  (initial_distance - v₁ * (right_triangle_distance / v₂))^2 + right_triangle_distance^2 = initial_distance^2 ∧
  (initial_distance / v₁) * v₂ = initial_distance - remaining_distance :=
by sorry

#check ships_initial_distance

end ships_initial_distance_l128_12874


namespace function_value_order_l128_12810

/-- A quadratic function with symmetry about x = 5 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetric : ∀ x, a * (5 - x)^2 + b * (5 - x) + c = a * (5 + x)^2 + b * (5 + x) + c

/-- The quadratic function -/
def f (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating the order of function values -/
theorem function_value_order (q : SymmetricQuadratic) :
  f q (2 * Real.pi) < f q (Real.sqrt 40) ∧ f q (Real.sqrt 40) < f q (5 * Real.sin (π / 4)) := by
  sorry

end function_value_order_l128_12810


namespace inequality_not_always_true_l128_12816

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hz : z ≠ 0) :
  ¬ ∀ (x y z : ℝ), x > 0 → y > 0 → x^2 > y^2 → z ≠ 0 → x * z^3 > y * z^3 :=
by sorry

end inequality_not_always_true_l128_12816


namespace cards_after_exchange_and_giveaway_l128_12813

/-- Represents the number of cards in a box for each sport --/
structure CardCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Represents the number of boxes for each sport --/
structure BoxCounts where
  basketball : ℕ
  baseball : ℕ
  football : ℕ
  hockey : ℕ
  soccer : ℕ

/-- Calculate the total number of cards --/
def totalCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  cards.basketball * boxes.basketball +
  cards.baseball * boxes.baseball +
  cards.football * boxes.football +
  cards.hockey * boxes.hockey +
  cards.soccer * boxes.soccer

/-- The number of cards exchanged between Ben and Alex --/
def exchangedCards (cards : CardCounts) (boxes : BoxCounts) : ℕ :=
  (cards.basketball / 2) * boxes.basketball +
  (cards.baseball / 2) * boxes.baseball

theorem cards_after_exchange_and_giveaway 
  (ben_cards : CardCounts)
  (ben_boxes : BoxCounts)
  (alex_cards : CardCounts)
  (alex_boxes : BoxCounts)
  (h1 : ben_cards.basketball = 20)
  (h2 : ben_cards.baseball = 15)
  (h3 : ben_cards.football = 12)
  (h4 : ben_boxes.basketball = 8)
  (h5 : ben_boxes.baseball = 10)
  (h6 : ben_boxes.football = 12)
  (h7 : alex_cards.hockey = 15)
  (h8 : alex_cards.soccer = 18)
  (h9 : alex_boxes.hockey = 6)
  (h10 : alex_boxes.soccer = 9)
  (cards_given_away : ℕ)
  (h11 : cards_given_away = 175) :
  totalCards ben_cards ben_boxes + totalCards alex_cards alex_boxes - cards_given_away = 531 := by
  sorry


end cards_after_exchange_and_giveaway_l128_12813


namespace o2_moles_combined_l128_12895

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (C2H6_ratio : ℚ)
  (O2_ratio : ℚ)
  (C2H4O_ratio : ℚ)
  (H2O_ratio : ℚ)

-- Define the balanced reaction
def balanced_reaction : Reaction :=
  { C2H6_ratio := 1
  , O2_ratio := 1/2
  , C2H4O_ratio := 1
  , H2O_ratio := 1 }

-- Theorem statement
theorem o2_moles_combined 
  (r : Reaction) 
  (h1 : r.C2H6_ratio = 1) 
  (h2 : r.C2H4O_ratio = 1) 
  (h3 : r = balanced_reaction) : 
  r.O2_ratio = 1/2 := by
  sorry

end o2_moles_combined_l128_12895


namespace a_value_l128_12827

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

/-- The value of b -/
def b : ℕ := 2120

/-- The ratio q -/
def q : ℚ := 1/2

/-- The value of a -/
def a : ℕ := 1060

/-- Theorem stating that a = 1060 given the conditions -/
theorem a_value : a = 1060 :=
  sorry

end a_value_l128_12827


namespace unique_solution_for_digit_equation_l128_12875

theorem unique_solution_for_digit_equation :
  ∃! (A B D E : ℕ),
    (A < 10 ∧ B < 10 ∧ D < 10 ∧ E < 10) ∧  -- Base 10 digits
    (A ≠ B ∧ A ≠ D ∧ A ≠ E ∧ B ≠ D ∧ B ≠ E ∧ D ≠ E) ∧  -- Different digits
    (A^(10*A + A) + 10*A + A = 
      B * 10^15 + B * 10^14 + 9 * 10^13 +
      D * 10^12 + E * 10^11 + D * 10^10 +
      B * 10^9 + E * 10^8 + E * 10^7 +
      B * 10^6 + B * 10^5 + B * 10^4 +
      B * 10^3 + B * 10^2 + E * 10^1 + E * 10^0) ∧
    (A = 3 ∧ B = 5 ∧ D = 0 ∧ E = 6) :=
by sorry

end unique_solution_for_digit_equation_l128_12875


namespace problem_solution_l128_12889

theorem problem_solution (x : ℝ) :
  x - Real.sqrt (x^2 + 1) + 1 / (x + Real.sqrt (x^2 + 1)) = 28 →
  x^2 - Real.sqrt (x^4 + 1) + 1 / (x^2 - Real.sqrt (x^4 + 1)) = -2 * Real.sqrt 38026 := by
  sorry

end problem_solution_l128_12889


namespace roots_of_quadratic_equation_l128_12861

theorem roots_of_quadratic_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end roots_of_quadratic_equation_l128_12861


namespace intersection_equality_l128_12814

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = -1 ∨ a = 1/3) :=
sorry

end intersection_equality_l128_12814


namespace dennis_teaching_years_l128_12844

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) 
  (total : v + a + d = 93)
  (v_more_than_a : v = a + 9)
  (v_less_than_d : v = d - 9) :
  d = 40 := by
  sorry

end dennis_teaching_years_l128_12844


namespace article_cost_l128_12831

theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (gain_percentage : ℝ) :
  sell_price_high = 350 →
  sell_price_low = 340 →
  gain_percentage = 0.04 →
  ∃ (cost : ℝ),
    sell_price_high - cost = (1 + gain_percentage) * (sell_price_low - cost) ∧
    cost = 90 := by
  sorry

end article_cost_l128_12831


namespace work_completion_time_l128_12871

/-- Given a work that can be completed by A in 14 days and by A and B together in 10 days,
    prove that B can complete the work alone in 35 days. -/
theorem work_completion_time (work : ℝ) (A B : ℝ → ℝ) : 
  (A work = work / 14) →
  (A work + B work = work / 10) →
  B work = work / 35 := by
  sorry

end work_completion_time_l128_12871


namespace birds_in_tree_l128_12822

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end birds_in_tree_l128_12822


namespace sum_of_segments_equals_radius_l128_12854

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (vertices : Fin (4*k+2) → ℝ × ℝ)

/-- The segments cut by angle A₍ₖ₎OA₍ₖ₊₁₎ on the lines A₁A₍₂ₖ₎, A₂A₍₂ₖ₋₁₎, ..., A₍ₖ₎A₍ₖ₊₁₎ -/
def cut_segments (p : RegularPolygon k) : List (ℝ × ℝ) :=
  sorry

/-- The sum of the lengths of the cut segments -/
def sum_of_segment_lengths (segments : List (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The sum of the lengths of the cut segments is equal to the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segment_lengths (cut_segments p) = p.radius :=
sorry

end sum_of_segments_equals_radius_l128_12854


namespace max_profit_multimedia_devices_l128_12863

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the number of devices -/
def device_constraint (x : ℝ) : Prop := 10 ≤ x ∧ x ≤ 50

theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), device_constraint x ∧
    (∀ y, device_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 19 ∧
    x = 10 := by sorry

end max_profit_multimedia_devices_l128_12863


namespace more_larger_boxes_l128_12828

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

/-- Represents the total number of glasses -/
def total_glasses : ℕ := 480

theorem more_larger_boxes (s l : ℕ) : 
  s * small_box + l * large_box = total_glasses →
  (s + l : ℚ) = (total_glasses : ℚ) / average_glasses →
  l > s →
  l - s = 16 := by
  sorry

end more_larger_boxes_l128_12828


namespace complex_absolute_value_l128_12878

theorem complex_absolute_value : Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 2)) ^ 6 = 576 := by
  sorry

end complex_absolute_value_l128_12878


namespace unique_solution_for_A_l128_12824

/-- Given an equation 1A + 4B3 = 469, where A and B are single digits and 4B3 is a three-digit number,
    prove that A = 6 is the unique solution for A. -/
theorem unique_solution_for_A : ∃! (A : ℕ), ∃ (B : ℕ),
  (A < 10) ∧ (B < 10) ∧ (400 ≤ 4 * 10 * B + 3) ∧ (4 * 10 * B + 3 < 1000) ∧
  (10 * A + 4 * 10 * B + 3 = 469) ∧ A = 6 :=
sorry

end unique_solution_for_A_l128_12824


namespace geometric_sequence_sixth_term_l128_12880

/-- Given a geometric sequence of positive numbers where the fourth term is 16
    and the ninth term is 8, the sixth term is equal to 16 * (4^(1/5)) -/
theorem geometric_sequence_sixth_term
  (a : ℝ → ℝ)  -- The sequence
  (r : ℝ)      -- Common ratio
  (h_positive : ∀ n, a n > 0)  -- All terms are positive
  (h_geometric : ∀ n, a (n + 1) = a n * r)  -- It's a geometric sequence
  (h_fourth : a 4 = 16)  -- The fourth term is 16
  (h_ninth : a 9 = 8)  -- The ninth term is 8
  : a 6 = 16 * (4^(1/5)) := by
sorry

end geometric_sequence_sixth_term_l128_12880


namespace highway_distance_theorem_l128_12898

/-- The distance between two points A and B on a highway -/
def distance_AB : ℝ := 198

/-- The speed of vehicles traveling from A to B -/
def speed_AB : ℝ := 50

/-- The speed of vehicles traveling from B to A -/
def speed_BA : ℝ := 60

/-- The distance from point B where car X breaks down -/
def breakdown_distance : ℝ := 30

/-- The delay in the second meeting due to the breakdown -/
def delay_time : ℝ := 1.2

theorem highway_distance_theorem :
  distance_AB = 198 :=
sorry

end highway_distance_theorem_l128_12898


namespace sum_of_digits_divisible_by_11_l128_12843

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 sequential natural numbers, there's always one with sum of digits divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (N + k) % 11 = 0) := by sorry

end sum_of_digits_divisible_by_11_l128_12843


namespace initial_pencils_count_l128_12870

/-- The number of pencils Sara added to the drawer -/
def pencils_added : ℕ := 100

/-- The total number of pencils in the drawer after Sara's addition -/
def total_pencils : ℕ := 215

/-- The initial number of pencils in the drawer -/
def initial_pencils : ℕ := total_pencils - pencils_added

theorem initial_pencils_count : initial_pencils = 115 := by
  sorry

end initial_pencils_count_l128_12870


namespace parabola_sum_l128_12800

/-- Represents a parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 7 → p.x_coord (-4) = 5 → p.a + p.b + p.c = -42 := by
  sorry

end parabola_sum_l128_12800


namespace necessary_implies_sufficient_l128_12817

-- Define what it means for q to be a necessary condition for p
def necessary_condition (p q : Prop) : Prop :=
  p → q

-- Define what it means for p to be a sufficient condition for q
def sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem necessary_implies_sufficient (p q : Prop) 
  (h : necessary_condition p q) : sufficient_condition p q :=
by
  sorry


end necessary_implies_sufficient_l128_12817


namespace shortest_side_length_l128_12867

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radius of inscribed circle
  r : ℝ
  -- Segments of side 'a' divided by tangent point
  a1 : ℝ
  a2 : ℝ
  -- Conditions
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < a1 ∧ 0 < a2
  tangent_point : a = a1 + a2
  radius : r = 5
  side_sum : b + c = 36
  segments : a1 = 7 ∧ a2 = 9

/-- The length of the shortest side in the triangle is 14 units -/
theorem shortest_side_length (t : TriangleWithInscribedCircle) : 
  min t.a (min t.b t.c) = 14 := by
  sorry

end shortest_side_length_l128_12867


namespace product_of_roots_cubic_l128_12858

theorem product_of_roots_cubic (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 4 * a - 12 = 0) ∧
  (3 * b^3 - 9 * b^2 + 4 * b - 12 = 0) ∧
  (3 * c^3 - 9 * c^2 + 4 * c - 12 = 0) →
  a * b * c = 4 := by
sorry

end product_of_roots_cubic_l128_12858


namespace two_digit_congruent_to_three_mod_four_count_l128_12836

theorem two_digit_congruent_to_three_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 3) (Finset.range 100)).card = 23 := by
  sorry

end two_digit_congruent_to_three_mod_four_count_l128_12836


namespace sin_sum_alpha_beta_l128_12879

theorem sin_sum_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = 0) : 
  Real.sin (α + β) = -1/2 := by
  sorry

end sin_sum_alpha_beta_l128_12879


namespace largest_x_sqrt_3x_eq_5x_l128_12866

theorem largest_x_sqrt_3x_eq_5x :
  ∃ (x_max : ℚ), x_max = 3/25 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 5 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 5 * x_max := by
  sorry

end largest_x_sqrt_3x_eq_5x_l128_12866


namespace multiplication_division_equality_l128_12847

theorem multiplication_division_equality : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 800 := by
  sorry

end multiplication_division_equality_l128_12847


namespace mark_collection_l128_12885

/-- The amount Mark collects for the homeless -/
theorem mark_collection (households_per_day : ℕ) (days : ℕ) (giving_ratio : ℚ) (donation_amount : ℕ) : 
  households_per_day = 20 →
  days = 5 →
  giving_ratio = 1/2 →
  donation_amount = 40 →
  (households_per_day * days : ℚ) * giving_ratio * donation_amount = 2000 := by
  sorry

#check mark_collection

end mark_collection_l128_12885


namespace square_difference_l128_12848

theorem square_difference (a b : ℝ) :
  let A : ℝ := (5*a + 3*b)^2 - (5*a - 3*b)^2
  A = 60*a*b := by sorry

end square_difference_l128_12848


namespace unique_valid_sequence_l128_12839

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ i j k, a i + a j ≠ a k) ∧
  (∀ m, ∃ k > m, a k = 2 * k - 1)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, is_valid_sequence a ↔ (∀ n, a n = 2 * n - 1) :=
sorry

end unique_valid_sequence_l128_12839


namespace right_triangle_side_length_l128_12840

theorem right_triangle_side_length 
  (A B C : ℝ) (AB BC AC : ℝ) :
  -- Triangle ABC is right-angled at A
  A + B + C = π / 2 →
  -- BC = 10
  BC = 10 →
  -- tan C = 3cos B
  Real.tan C = 3 * Real.cos B →
  -- AB² + AC² = BC²
  AB^2 + AC^2 = BC^2 →
  -- AB = (20√2)/3
  AB = 20 * Real.sqrt 2 / 3 := by
sorry

end right_triangle_side_length_l128_12840


namespace cuboid_volume_l128_12811

/-- Given a cuboid with three side faces sharing a common vertex having areas 3, 5, and 15,
    prove that its volume is 15. -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3)
  (h2 : b * c = 5)
  (h3 : a * c = 15) : 
  a * b * c = 15 := by
  sorry

end cuboid_volume_l128_12811


namespace millionaire_hat_sale_l128_12896

/-- Proves that the fraction of hats sold is 2/3 given the conditions of the problem -/
theorem millionaire_hat_sale (H : ℝ) (h1 : H > 0) : 
  let brown_hats := (1/4 : ℝ) * H
  let sold_brown_hats := (4/5 : ℝ) * brown_hats
  let remaining_hats := H - sold_brown_hats - ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)
  let remaining_brown_hats := brown_hats - sold_brown_hats
  (remaining_brown_hats / remaining_hats) = (15/100 : ℝ) →
  (sold_brown_hats + ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)) / H = (2/3 : ℝ) := by
sorry

end millionaire_hat_sale_l128_12896


namespace magnitude_of_z_l128_12891

-- Define the complex number
def z : ℂ := 7 - 24 * Complex.I

-- State the theorem
theorem magnitude_of_z : Complex.abs z = 25 := by
  sorry

end magnitude_of_z_l128_12891


namespace sequence_inequality_l128_12829

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) = 2 * b n

theorem sequence_inequality (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h1 : a 1 + b 1 > 0)
  (h2 : a 2 + b 2 < 0) :
  let m := a 4 + b 3
  m < 0 := by
  sorry

end sequence_inequality_l128_12829


namespace solution_set_of_inequality_l128_12820

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
  sorry

end solution_set_of_inequality_l128_12820


namespace paper_distribution_l128_12892

theorem paper_distribution (x y : ℕ+) : 
  x * y = 221 ↔ (x = 1 ∧ y = 221) ∨ (x = 221 ∧ y = 1) ∨ (x = 13 ∧ y = 17) ∨ (x = 17 ∧ y = 13) := by
  sorry

end paper_distribution_l128_12892


namespace well_volume_approximation_l128_12857

/-- The volume of a cylinder with diameter 6 meters and height 24 meters is approximately 678.58464 cubic meters. -/
theorem well_volume_approximation :
  let diameter : ℝ := 6
  let height : ℝ := 24
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 678.58464| < ε :=
by sorry

end well_volume_approximation_l128_12857


namespace call_center_theorem_l128_12806

/-- Represents the ratio of team A's size to team B's size -/
def team_size_ratio : ℚ := 5/8

/-- Represents the fraction of total calls processed by team B -/
def team_b_call_fraction : ℚ := 4/5

/-- Represents the ratio of calls processed by each member of team A to each member of team B -/
def member_call_ratio : ℚ := 2/5

theorem call_center_theorem :
  let total_calls : ℚ := 1
  let team_a_call_fraction : ℚ := total_calls - team_b_call_fraction
  team_size_ratio * (team_a_call_fraction / team_b_call_fraction) = member_call_ratio := by
  sorry

end call_center_theorem_l128_12806


namespace a_range_l128_12855

theorem a_range (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 4) 
  (order : a > b ∧ b > c) : 
  2/3 < a ∧ a < 2 := by
sorry

end a_range_l128_12855


namespace cubes_in_figure_100_l128_12807

/-- Represents the number of cubes in a figure at position n -/
def num_cubes (n : ℕ) : ℕ := 2 * n^3 + n^2 + 3 * n + 1

/-- The sequence of cubes follows the given pattern for the first four figures -/
axiom pattern_holds : num_cubes 0 = 1 ∧ num_cubes 1 = 7 ∧ num_cubes 2 = 25 ∧ num_cubes 3 = 63

/-- The number of cubes in figure 100 is 2010301 -/
theorem cubes_in_figure_100 : num_cubes 100 = 2010301 := by
  sorry

end cubes_in_figure_100_l128_12807


namespace discount_order_matters_l128_12850

/-- Proves that applying a percentage discount followed by a fixed discount
    results in a lower final price than the reverse order. -/
theorem discount_order_matters (initial_price percent_off fixed_off : ℝ) 
  (h_initial : initial_price = 50)
  (h_percent : percent_off = 0.15)
  (h_fixed : fixed_off = 6) : 
  (1 - percent_off) * initial_price - fixed_off < 
  (1 - percent_off) * (initial_price - fixed_off) := by
  sorry

end discount_order_matters_l128_12850


namespace trig_expression_equals_negative_two_l128_12808

theorem trig_expression_equals_negative_two :
  5 * Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -2 := by
  sorry

end trig_expression_equals_negative_two_l128_12808


namespace real_sum_greater_than_two_l128_12805

theorem real_sum_greater_than_two (x y : ℝ) : x + y > 2 → x > 1 ∨ y > 1 := by
  sorry

end real_sum_greater_than_two_l128_12805


namespace remainder_9387_div_11_l128_12894

theorem remainder_9387_div_11 : 9387 % 11 = 7 := by
  sorry

end remainder_9387_div_11_l128_12894


namespace toms_out_of_pocket_cost_l128_12868

theorem toms_out_of_pocket_cost 
  (visit_cost : ℝ) 
  (cast_cost : ℝ) 
  (insurance_coverage_percentage : ℝ) 
  (h1 : visit_cost = 300)
  (h2 : cast_cost = 200)
  (h3 : insurance_coverage_percentage = 60) :
  let total_cost := visit_cost + cast_cost
  let insurance_coverage := (insurance_coverage_percentage / 100) * total_cost
  let out_of_pocket_cost := total_cost - insurance_coverage
  out_of_pocket_cost = 200 := by
sorry

end toms_out_of_pocket_cost_l128_12868


namespace simplify_and_evaluate_l128_12846

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  x + (1/3) * y^2 - 2 * (x - (1/3) * y^2) = 3 := by
  sorry

end simplify_and_evaluate_l128_12846


namespace mass_of_man_on_boat_l128_12897

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man in the given scenario. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 4
  let boat_breadth : ℝ := 3
  let boat_sinking : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000 -- kg/m³
  mass_of_man boat_length boat_breadth boat_sinking water_density = 120 := by
  sorry

end mass_of_man_on_boat_l128_12897


namespace calculator_game_sum_l128_12823

def iterate_calculator (n : ℕ) (initial : ℤ) (f : ℤ → ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_calculator m initial f)

theorem calculator_game_sum (n : ℕ) : 
  iterate_calculator n 1 (λ x => x^3) + 
  iterate_calculator n 0 (λ x => x^2) + 
  iterate_calculator n (-1) (λ x => -x) = 0 :=
by sorry

end calculator_game_sum_l128_12823


namespace tan_beta_value_l128_12881

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end tan_beta_value_l128_12881


namespace existence_of_coprime_sum_l128_12826

theorem existence_of_coprime_sum (n k : ℕ+) 
  (h : n.val % 2 = 1 ∨ (n.val % 2 = 0 ∧ k.val % 2 = 0)) :
  ∃ a b : ℤ, Nat.gcd a.natAbs n.val = 1 ∧ 
             Nat.gcd b.natAbs n.val = 1 ∧ 
             k.val = a + b := by
  sorry

end existence_of_coprime_sum_l128_12826


namespace opposite_def_opposite_of_four_l128_12837

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 4 is -4 -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end opposite_def_opposite_of_four_l128_12837


namespace quadratic_inequality_problem_l128_12872

-- Define the given quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the given inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a) →
  (a = -2 ∧
   ∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end quadratic_inequality_problem_l128_12872


namespace distance_at_least_diameter_time_l128_12859

/-- Represents a circular track -/
structure Track where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a car on a track -/
structure Car where
  track : Track
  clockwise : Bool
  position : ℝ → ℝ × ℝ

/-- The setup of the problem -/
def problem_setup : ℝ × Track × Track × Car × Car := sorry

/-- The time during which the distance between the cars is at least the diameter of each track -/
def time_at_least_diameter (setup : ℝ × Track × Track × Car × Car) : ℝ := sorry

/-- The main theorem stating that the time during which the distance between the cars 
    is at least the diameter of each track is 1/2 hour -/
theorem distance_at_least_diameter_time 
  (setup : ℝ × Track × Track × Car × Car) 
  (h_setup : setup = problem_setup) : 
  time_at_least_diameter setup = 1/2 := by sorry

end distance_at_least_diameter_time_l128_12859


namespace c_highest_prob_exactly_two_passing_l128_12833

-- Define the probabilities of passing each exam for A, B, and C
def probATheory : ℚ := 4/5
def probAPractical : ℚ := 1/2
def probBTheory : ℚ := 3/4
def probBPractical : ℚ := 2/3
def probCTheory : ℚ := 2/3
def probCPractical : ℚ := 5/6

-- Define the probabilities of obtaining the "certificate of passing" for A, B, and C
def probAPassing : ℚ := probATheory * probAPractical
def probBPassing : ℚ := probBTheory * probBPractical
def probCPassing : ℚ := probCTheory * probCPractical

-- Theorem 1: C has the highest probability of obtaining the "certificate of passing"
theorem c_highest_prob : 
  probCPassing > probAPassing ∧ probCPassing > probBPassing :=
sorry

-- Theorem 2: The probability that exactly two out of A, B, and C obtain the "certificate of passing" is 11/30
theorem exactly_two_passing :
  probAPassing * probBPassing * (1 - probCPassing) +
  probAPassing * (1 - probBPassing) * probCPassing +
  (1 - probAPassing) * probBPassing * probCPassing = 11/30 :=
sorry

end c_highest_prob_exactly_two_passing_l128_12833


namespace impossible_to_reach_in_time_l128_12849

/-- Represents the problem of traveling to the train station -/
structure TravelProblem where
  totalTime : ℝ
  totalDistance : ℝ
  firstKilometerSpeed : ℝ

/-- Defines the given travel problem -/
def givenProblem : TravelProblem where
  totalTime := 2  -- 2 minutes
  totalDistance := 2  -- 2 km
  firstKilometerSpeed := 30  -- 30 km/h

/-- Theorem stating that it's impossible to reach the destination in time -/
theorem impossible_to_reach_in_time (p : TravelProblem) 
  (h1 : p.totalTime = 2)
  (h2 : p.totalDistance = 2)
  (h3 : p.firstKilometerSpeed = 30) : 
  ¬ ∃ (secondKilometerSpeed : ℝ), 
    (1 / (p.firstKilometerSpeed / 60)) + (1 / secondKilometerSpeed) ≤ p.totalTime :=
by sorry

#check impossible_to_reach_in_time givenProblem rfl rfl rfl

end impossible_to_reach_in_time_l128_12849


namespace circle_C_tangent_line_l_line_AB_passes_through_intersection_l128_12862

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 3

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define line AB
def line_AB (x y : ℝ) : Prop := 2*x + y - 4 = 0

-- Theorem 1: Circle C is tangent to line l
theorem circle_C_tangent_line_l : ∃ (x y : ℝ), circle_C x y ∧ line_l x := by sorry

-- Theorem 2: Line AB passes through the intersection of circles C and O
theorem line_AB_passes_through_intersection :
  ∀ (x y : ℝ), (circle_C x y ∧ circle_O x y) → line_AB x y := by sorry

end circle_C_tangent_line_l_line_AB_passes_through_intersection_l128_12862


namespace average_of_DEF_l128_12842

theorem average_of_DEF (D E F : ℚ) 
  (eq1 : 2003 * F - 4006 * D = 8012)
  (eq2 : 2003 * E + 6009 * D = 10010) :
  (D + E + F) / 3 = 3 := by
sorry

end average_of_DEF_l128_12842


namespace part_one_part_two_l128_12825

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := 1
  f x a b > 8 ↔ (x < -1 ∨ x > 1.5) :=
sorry

-- Part II
theorem part_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) →
  (1/a + 1/b ≥ (3 + 2*Real.sqrt 2) / 2) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) ∧ 1/a + 1/b = (3 + 2*Real.sqrt 2) / 2) :=
sorry

end part_one_part_two_l128_12825


namespace bank_line_theorem_l128_12851

/-- Represents a bank line with fast and slow customers. -/
structure BankLine where
  total_customers : Nat
  fast_customers : Nat
  slow_customers : Nat
  fast_operation_time : Nat
  slow_operation_time : Nat

/-- Calculates the minimum total wasted person-minutes. -/
def minimum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the maximum total wasted person-minutes. -/
def maximum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the expected number of wasted person-minutes. -/
def expected_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Theorem stating the results for the specific bank line scenario. -/
theorem bank_line_theorem (line : BankLine) 
    (h1 : line.total_customers = 8)
    (h2 : line.fast_customers = 5)
    (h3 : line.slow_customers = 3)
    (h4 : line.fast_operation_time = 1)
    (h5 : line.slow_operation_time = 5) :
  minimum_wasted_time line = 40 ∧
  maximum_wasted_time line = 100 ∧
  expected_wasted_time line = 70 :=
  sorry

end bank_line_theorem_l128_12851


namespace factory_workers_count_l128_12812

/-- Proves the number of workers in a factory given certain salary information --/
theorem factory_workers_count :
  let initial_average : ℚ := 430
  let initial_supervisor_salary : ℚ := 870
  let new_average : ℚ := 390
  let new_supervisor_salary : ℚ := 510
  let total_people : ℕ := 9
  ∃ (workers : ℕ),
    (workers : ℚ) + 1 = (total_people : ℚ) ∧
    (workers + 1) * initial_average = workers * initial_average + initial_supervisor_salary ∧
    total_people * new_average = workers * initial_average + new_supervisor_salary ∧
    workers = 8 := by
  sorry

end factory_workers_count_l128_12812


namespace probability_two_defective_shipment_l128_12888

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  |probability_two_defective 240 84 - 1216/10000| < ε :=
sorry

end probability_two_defective_shipment_l128_12888


namespace origin_on_circle_l128_12886

theorem origin_on_circle (center_x center_y radius : ℝ) 
  (h1 : center_x = 5)
  (h2 : center_y = 12)
  (h3 : radius = 13) :
  (center_x^2 + center_y^2).sqrt = radius :=
sorry

end origin_on_circle_l128_12886


namespace exist_integers_product_minus_third_l128_12899

theorem exist_integers_product_minus_third : ∃ (a b c : ℤ), 
  (a * b - c = 2018) ∧ (b * c - a = 2018) ∧ (c * a - b = 2018) := by
sorry

end exist_integers_product_minus_third_l128_12899


namespace tetrakaidecagon_area_approx_l128_12893

/-- A tetrakaidecagon inscribed in a square -/
structure InscribedTetrakaidecagon where
  /-- The side length of the square -/
  square_side : ℝ
  /-- The number of segments each side of the square is divided into -/
  num_segments : ℕ
  /-- The perimeter of the square -/
  square_perimeter : ℝ
  /-- The perimeter of the square is 56 meters -/
  perimeter_constraint : square_perimeter = 56
  /-- Each side of the square is divided into equal segments -/
  side_division : square_side = square_perimeter / 4
  /-- The number of segments is 7 -/
  segment_count : num_segments = 7

/-- The area of the inscribed tetrakaidecagon -/
noncomputable def tetrakaidecagon_area (t : InscribedTetrakaidecagon) : ℝ :=
  t.square_side ^ 2 - 16 * (1 / 2 * (t.square_side / t.num_segments) ^ 2)

/-- Theorem stating the area of the inscribed tetrakaidecagon -/
theorem tetrakaidecagon_area_approx (t : InscribedTetrakaidecagon) :
  abs (tetrakaidecagon_area t - 21.92) < 0.01 := by
  sorry

end tetrakaidecagon_area_approx_l128_12893
