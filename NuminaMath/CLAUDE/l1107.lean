import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1107_110704

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 8)
  (h_b_a : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1107_110704


namespace NUMINAMATH_CALUDE_students_taking_both_music_and_art_l1107_110727

theorem students_taking_both_music_and_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (neither : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : neither = 440) : 
  total - neither - (music + art - (total - neither)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_music_and_art_l1107_110727


namespace NUMINAMATH_CALUDE_car_journey_cost_l1107_110759

/-- Calculates the total cost of a car journey given various expenses -/
theorem car_journey_cost
  (rental_cost : ℝ)
  (rental_discount_percent : ℝ)
  (gas_cost_per_gallon : ℝ)
  (gas_gallons : ℝ)
  (driving_cost_per_mile : ℝ)
  (miles_driven : ℝ)
  (toll_fees : ℝ)
  (parking_cost_per_day : ℝ)
  (parking_days : ℝ)
  (h1 : rental_cost = 150)
  (h2 : rental_discount_percent = 15)
  (h3 : gas_cost_per_gallon = 3.5)
  (h4 : gas_gallons = 8)
  (h5 : driving_cost_per_mile = 0.5)
  (h6 : miles_driven = 320)
  (h7 : toll_fees = 15)
  (h8 : parking_cost_per_day = 20)
  (h9 : parking_days = 3) :
  rental_cost * (1 - rental_discount_percent / 100) +
  gas_cost_per_gallon * gas_gallons +
  driving_cost_per_mile * miles_driven +
  toll_fees +
  parking_cost_per_day * parking_days = 390.5 := by
  sorry


end NUMINAMATH_CALUDE_car_journey_cost_l1107_110759


namespace NUMINAMATH_CALUDE_ticket_sales_proof_l1107_110731

theorem ticket_sales_proof (total_tickets : ℕ) (reduced_price_tickets : ℕ) (full_price_ratio : ℕ) :
  total_tickets = 25200 →
  reduced_price_tickets = 5400 →
  full_price_ratio = 5 →
  reduced_price_tickets + full_price_ratio * reduced_price_tickets = total_tickets →
  full_price_ratio * reduced_price_tickets = 21000 :=
by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_proof_l1107_110731


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1107_110743

theorem more_girls_than_boys (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1107_110743


namespace NUMINAMATH_CALUDE_tan_five_pi_fourths_l1107_110753

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourths_l1107_110753


namespace NUMINAMATH_CALUDE_tom_build_time_l1107_110772

theorem tom_build_time (avery_time : ℝ) (joint_work_time : ℝ) (tom_finish_time : ℝ) :
  avery_time = 3 →
  joint_work_time = 1 →
  tom_finish_time = 39.99999999999999 / 60 →
  ∃ (tom_solo_time : ℝ),
    (1 / avery_time + 1 / tom_solo_time) * joint_work_time + 
    (1 / tom_solo_time) * tom_finish_time = 1 ∧
    tom_solo_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_tom_build_time_l1107_110772


namespace NUMINAMATH_CALUDE_weight_identification_unbiased_weight_identification_biased_l1107_110744

/-- Represents a weight with a mass in grams -/
structure Weight where
  mass : ℕ

/-- Represents a balance scale -/
inductive BalanceResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing operation -/
def weighing (left right : List Weight) (bias : ℕ := 0) : BalanceResult :=
  sorry

/-- Represents the process of identifying weights -/
def identifyWeights (weights : List Weight) (numWeighings : ℕ) (bias : ℕ := 0) : Bool :=
  sorry

/-- The set of weights Tanya has -/
def tanyasWeights : List Weight :=
  [⟨1000⟩, ⟨1002⟩, ⟨1004⟩, ⟨1005⟩]

theorem weight_identification_unbiased :
  ¬ (identifyWeights tanyasWeights 4 0) :=
sorry

theorem weight_identification_biased :
  identifyWeights tanyasWeights 4 1 :=
sorry

end NUMINAMATH_CALUDE_weight_identification_unbiased_weight_identification_biased_l1107_110744


namespace NUMINAMATH_CALUDE_expression_simplification_l1107_110784

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) :
  (x - 2) / (6 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1107_110784


namespace NUMINAMATH_CALUDE_age_sum_problem_l1107_110764

theorem age_sum_problem :
  ∀ (y k : ℕ+),
    y * (2 * y) * k = 72 →
    y + (2 * y) + k = 13 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1107_110764


namespace NUMINAMATH_CALUDE_distance_to_plane_value_l1107_110723

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 20
def QR : ℝ := 21
def RP : ℝ := 29

-- Define the distance from S to the plane of triangle PQR
def distance_to_plane : ℝ := sorry

-- State the theorem
theorem distance_to_plane_value : distance_to_plane = (266 : ℝ) * Real.sqrt 154 / 14 := by sorry

end NUMINAMATH_CALUDE_distance_to_plane_value_l1107_110723


namespace NUMINAMATH_CALUDE_gold_hoard_problem_l1107_110797

theorem gold_hoard_problem (total_per_brother : ℝ) (eldest_gold : ℝ) (eldest_silver_fraction : ℝ)
  (total_silver : ℝ) (h1 : total_per_brother = 100)
  (h2 : eldest_gold = 30)
  (h3 : eldest_silver_fraction = 1/5)
  (h4 : total_silver = 350) :
  eldest_gold + (total_silver - eldest_silver_fraction * total_silver) = 50 := by
  sorry


end NUMINAMATH_CALUDE_gold_hoard_problem_l1107_110797


namespace NUMINAMATH_CALUDE_intersection_when_a_10_subset_condition_l1107_110783

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Theorem for part 1
theorem intersection_when_a_10 :
  A 10 ∩ B = {x | 21 ≤ x ∧ x ≤ 22} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ a : ℝ, A a ⊆ B ↔ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_10_subset_condition_l1107_110783


namespace NUMINAMATH_CALUDE_solve_equation_l1107_110725

theorem solve_equation (x : ℚ) : (2 * x + 7) / 5 = 22 → x = 103 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1107_110725


namespace NUMINAMATH_CALUDE_direction_vector_b_l1107_110776

/-- Given a line passing through points (-4, 6) and (3, -3), 
    prove that the direction vector of the form (b, 1) has b = -7/9 -/
theorem direction_vector_b (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-4, 6) → p2 = (3, -3) → 
  ∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (b, 1) → 
  b = -7/9 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_l1107_110776


namespace NUMINAMATH_CALUDE_winning_strategy_l1107_110755

/-- Represents the winner of the game -/
inductive Winner
  | FirstPlayer
  | SecondPlayer

/-- Determines the winner of the game based on board dimensions -/
def gameWinner (n k : ℕ) : Winner :=
  if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer

/-- Theorem stating the winning condition for the game -/
theorem winning_strategy (n k : ℕ) (h1 : n > 0) (h2 : k > 1) :
  gameWinner n k = if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer :=
by sorry

end NUMINAMATH_CALUDE_winning_strategy_l1107_110755


namespace NUMINAMATH_CALUDE_rachel_class_selection_l1107_110726

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem rachel_class_selection :
  let total_classes : ℕ := 10
  let mandatory_classes : ℕ := 2
  let classes_to_choose : ℕ := 5
  let remaining_classes := total_classes - mandatory_classes
  let additional_classes := classes_to_choose - mandatory_classes
  choose remaining_classes additional_classes = 56 := by sorry

end NUMINAMATH_CALUDE_rachel_class_selection_l1107_110726


namespace NUMINAMATH_CALUDE_total_cards_proof_l1107_110795

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 6

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_cards_proof : total_cards = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_proof_l1107_110795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1107_110766

/-- Given arithmetic sequences a and b satisfying certain conditions, prove a₁b₁ = 4 -/
theorem arithmetic_sequence_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- a is arithmetic
  (∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- b is arithmetic
  a 2 * b 2 = 4 →
  a 3 * b 3 = 8 →
  a 4 * b 4 = 16 →
  a 1 * b 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1107_110766


namespace NUMINAMATH_CALUDE_burger_cost_is_13_l1107_110705

/-- The cost of a single burger given the conditions of Alice's burger purchases in June. -/
def burger_cost (burgers_per_day : ℕ) (days_in_june : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (burgers_per_day * days_in_june)

/-- Theorem stating that the cost of each burger is 13 dollars under the given conditions. -/
theorem burger_cost_is_13 :
  burger_cost 4 30 1560 = 13 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_13_l1107_110705


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1107_110778

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 - 5*x - 6 = 0) ∧ 
  ¬(x^2 - 5*x - 6 = 0 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1107_110778


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1107_110735

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 10) :
  (a + b + x = 73) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1107_110735


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1107_110732

-- Problem 1
theorem equation_one_solutions (x : ℝ) :
  4 * x^2 - 16 = 0 ↔ x = 2 ∨ x = -2 :=
sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) :
  (2*x - 1)^3 + 64 = 0 ↔ x = -3/2 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1107_110732


namespace NUMINAMATH_CALUDE_m_geq_n_l1107_110720

theorem m_geq_n (a x : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ 4 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_m_geq_n_l1107_110720


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1107_110746

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 8) :
  a 3 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1107_110746


namespace NUMINAMATH_CALUDE_owen_sleep_time_l1107_110752

theorem owen_sleep_time (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_owen_sleep_time_l1107_110752


namespace NUMINAMATH_CALUDE_evaluate_expression_l1107_110748

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1107_110748


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1107_110771

-- Define the point (x, y) in the Cartesian coordinate system
def point (a : ℝ) : ℝ × ℝ := (-2, a^2 + 1)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (a : ℝ) :
  in_second_quadrant (point a) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1107_110771


namespace NUMINAMATH_CALUDE_vector_relations_l1107_110789

-- Define the plane vector type
structure PlaneVector where
  x : ℝ
  y : ℝ

-- Define the "›" relation
def vecGreater (a b : PlaneVector) : Prop :=
  a.x > b.x ∨ (a.x = b.x ∧ a.y > b.y)

-- Define vector addition
def vecAdd (a b : PlaneVector) : PlaneVector :=
  ⟨a.x + b.x, a.y + b.y⟩

-- Define dot product
def vecDot (a b : PlaneVector) : ℝ :=
  a.x * b.x + a.y * b.y

-- Theorem statement
theorem vector_relations :
  let e₁ : PlaneVector := ⟨1, 0⟩
  let e₂ : PlaneVector := ⟨0, 1⟩
  let zero : PlaneVector := ⟨0, 0⟩
  
  -- Proposition 1
  (vecGreater e₁ e₂ ∧ vecGreater e₂ zero) ∧
  
  -- Proposition 2
  (∀ a₁ a₂ a₃ : PlaneVector, vecGreater a₁ a₂ → vecGreater a₂ a₃ → vecGreater a₁ a₃) ∧
  
  -- Proposition 3
  (∀ a₁ a₂ a : PlaneVector, vecGreater a₁ a₂ → vecGreater (vecAdd a₁ a) (vecAdd a₂ a)) ∧
  
  -- Proposition 4 (negation)
  ¬(∀ a a₁ a₂ : PlaneVector, vecGreater a zero → vecGreater a₁ a₂ → vecGreater ⟨vecDot a a₁, 0⟩ ⟨vecDot a a₂, 0⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l1107_110789


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l1107_110728

/-- Given a triangle with sides a, b, and x, where a > b, 
    prove that the perimeter m satisfies 2a < m < 2(a+b) -/
theorem triangle_perimeter_range 
  (a b x : ℝ) 
  (h1 : a > b) 
  (h2 : a - b < x) 
  (h3 : x < a + b) : 
  2 * a < a + b + x ∧ a + b + x < 2 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l1107_110728


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1107_110707

/-- The ellipse E -/
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l -/
def l (k m x y : ℝ) : Prop := y = k*x + m

/-- Predicate to check if a point is on the ellipse E -/
def on_ellipse (x y : ℝ) : Prop := E x y

/-- Predicate to check if a point is on the line l -/
def on_line (k m x y : ℝ) : Prop := l k m x y

/-- The right vertex of the ellipse -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Predicate to check if two points are different -/
def different (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem ellipse_line_intersection (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    on_ellipse M.1 M.2 ∧
    on_ellipse N.1 N.2 ∧
    on_line k m M.1 M.2 ∧
    on_line k m N.1 N.2 ∧
    different M right_vertex ∧
    different N right_vertex ∧
    different M N →
    on_line k m (2/7) 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1107_110707


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1107_110733

theorem floor_equation_solutions (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) :
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, ⌊a * x⌋ = x)) ↔ 
  (1 + 1 / n : ℝ) ≤ a ∧ a < 1 + 1 / (n - 1) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1107_110733


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1107_110745

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) :
  Real.sqrt ((x * y + x + y) / z) + Real.sqrt ((y * z + y + z) / x) + Real.sqrt ((z * x + z + x) / y) ≥ 
  3 * Real.sqrt (3 * (x + 2) * (y + 2) * (z + 2) / ((2 * x + 1) * (2 * y + 1) * (2 * z + 1))) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1107_110745


namespace NUMINAMATH_CALUDE_nancy_problem_rate_l1107_110711

/-- Given Nancy's homework details, prove she can finish 8 problems per hour -/
theorem nancy_problem_rate :
  let math_problems : ℝ := 17.0
  let spelling_problems : ℝ := 15.0
  let total_hours : ℝ := 4.0
  let total_problems := math_problems + spelling_problems
  let problems_per_hour := total_problems / total_hours
  problems_per_hour = 8 := by sorry

end NUMINAMATH_CALUDE_nancy_problem_rate_l1107_110711


namespace NUMINAMATH_CALUDE_order_of_abc_l1107_110774

theorem order_of_abc : ∀ (a b c : ℝ), 
  a = 2^(1/10) → 
  b = Real.log (1/2) → 
  c = (2/3)^Real.pi → 
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1107_110774


namespace NUMINAMATH_CALUDE_copper_price_calculation_l1107_110767

/-- The price of copper per pound in cents -/
def copper_price : ℚ := 65

/-- The price of zinc per pound in cents -/
def zinc_price : ℚ := 30

/-- The weight of brass in pounds -/
def brass_weight : ℚ := 70

/-- The price of brass per pound in cents -/
def brass_price : ℚ := 45

/-- The weight of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The weight of zinc used in pounds -/
def zinc_weight : ℚ := 40

theorem copper_price_calculation : 
  copper_price * copper_weight + zinc_price * zinc_weight = brass_price * brass_weight :=
sorry

end NUMINAMATH_CALUDE_copper_price_calculation_l1107_110767


namespace NUMINAMATH_CALUDE_boris_candy_problem_l1107_110749

theorem boris_candy_problem (initial_candy : ℕ) : 
  let daughter_eats : ℕ := 8
  let num_bowls : ℕ := 4
  let boris_takes_per_bowl : ℕ := 3
  let candy_left_in_one_bowl : ℕ := 20
  (initial_candy - daughter_eats) / num_bowls - boris_takes_per_bowl = candy_left_in_one_bowl →
  initial_candy = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l1107_110749


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l1107_110782

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l1107_110782


namespace NUMINAMATH_CALUDE_min_value_expression_l1107_110799

theorem min_value_expression (x : ℝ) (h : x > 1) :
  x + 9 / x - 2 ≥ 4 ∧ ∃ y > 1, y + 9 / y - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1107_110799


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1107_110757

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int), (∀ n : Int, (n - 3) * (n + 5) * (n - 1) < 0 ↔ n ∈ S) ∧ Finset.card S = 6 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1107_110757


namespace NUMINAMATH_CALUDE_sedans_sold_prediction_l1107_110717

/-- The ratio of sports cars to sedans -/
def car_ratio : ℚ := 3 / 5

/-- The number of sports cars predicted to be sold -/
def sports_cars_sold : ℕ := 36

/-- The number of sedans expected to be sold -/
def sedans_sold : ℕ := 60

/-- Theorem stating the relationship between sports cars and sedans sold -/
theorem sedans_sold_prediction :
  (car_ratio * sports_cars_sold : ℚ) = sedans_sold := by sorry

end NUMINAMATH_CALUDE_sedans_sold_prediction_l1107_110717


namespace NUMINAMATH_CALUDE_count_permutable_divisible_by_11_l1107_110779

/-- A function that counts the number of integers with k digits (including leading zeros)
    whose digits can be permuted to form a number divisible by 11 -/
def f (k : ℕ) : ℕ := sorry

/-- A predicate that checks if an integer's digits can be permuted to form a number divisible by 11 -/
def can_permute_to_divisible_by_11 (n : ℕ) : Prop := sorry

theorem count_permutable_divisible_by_11 (m : ℕ+) :
  f (2 * m) = 10 * f (2 * m - 1) :=
by sorry

end NUMINAMATH_CALUDE_count_permutable_divisible_by_11_l1107_110779


namespace NUMINAMATH_CALUDE_hundred_million_scientific_notation_l1107_110730

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem hundred_million_scientific_notation :
  toScientificNotation 100000000 = ScientificNotation.mk 1 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_hundred_million_scientific_notation_l1107_110730


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1107_110742

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1107_110742


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1107_110773

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1107_110773


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1107_110790

noncomputable def m : ℝ → ℝ → ℝ → ℝ := sorry

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B) →
  (let C : ℝ × ℝ := (1, 0);
   ∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B ∧
    abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2 = 8/5) →
  m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1107_110790


namespace NUMINAMATH_CALUDE_student_grade_problem_l1107_110729

theorem student_grade_problem (grade2 grade3 average : ℚ) 
  (h1 : grade2 = 80/100)
  (h2 : grade3 = 85/100)
  (h3 : average = 75/100)
  (h4 : (grade1 + grade2 + grade3) / 3 = average) :
  grade1 = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_problem_l1107_110729


namespace NUMINAMATH_CALUDE_unique_A_for_multiple_of_9_l1107_110702

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (A : ℕ) : ℕ := 2 + A + 3 + A

def four_digit_number (A : ℕ) : ℕ := 2000 + 100 * A + 30 + A

theorem unique_A_for_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_number A) ∧ A = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_A_for_multiple_of_9_l1107_110702


namespace NUMINAMATH_CALUDE_matrix_cube_sum_l1107_110798

/-- Given a 3x3 complex matrix N of the form [d e f; e f d; f d e] where N^2 = I and def = -1,
    the possible values of d^3 + e^3 + f^3 are 2 and 4. -/
theorem matrix_cube_sum (d e f : ℂ) : 
  let N : Matrix (Fin 3) (Fin 3) ℂ := !![d, e, f; e, f, d; f, d, e]
  (N ^ 2 = 1 ∧ d * e * f = -1) →
  (d^3 + e^3 + f^3 = 2 ∨ d^3 + e^3 + f^3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_matrix_cube_sum_l1107_110798


namespace NUMINAMATH_CALUDE_x_2007_equals_2_l1107_110769

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + x (n + 1)) / x n

theorem x_2007_equals_2 : x 2007 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_2007_equals_2_l1107_110769


namespace NUMINAMATH_CALUDE_max_servings_is_16_l1107_110700

/-- Represents the number of servings that can be made from a given ingredient --/
def servings_from_ingredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Represents the available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (min (servings_from_ingredient available.bananas recipe.bananas)
         (servings_from_ingredient available.yogurt recipe.yogurt))
    (min (servings_from_ingredient available.honey recipe.honey)
         (servings_from_ingredient available.strawberries recipe.strawberries))

theorem max_servings_is_16 (recipe : Recipe) (available : Available) :
  recipe.bananas = 3 ∧ recipe.yogurt = 1 ∧ recipe.honey = 2 ∧ recipe.strawberries = 2 ∧
  available.bananas = 12 ∧ available.yogurt = 6 ∧ available.honey = 16 ∧ available.strawberries = 8 →
  max_servings recipe available = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_16_l1107_110700


namespace NUMINAMATH_CALUDE_minimum_savings_for_contribution_l1107_110781

def savings_september : ℕ := 50
def savings_october : ℕ := 37
def savings_november : ℕ := 11
def mom_contribution : ℕ := 25
def video_game_cost : ℕ := 87
def amount_left : ℕ := 36

def total_savings : ℕ := savings_september + savings_october + savings_november

theorem minimum_savings_for_contribution :
  total_savings = (amount_left + video_game_cost) - mom_contribution :=
by sorry

end NUMINAMATH_CALUDE_minimum_savings_for_contribution_l1107_110781


namespace NUMINAMATH_CALUDE_shaded_area_equals_sixteen_twentyseventh_l1107_110716

/-- Represents the fraction of shaded area in each iteration -/
def shaded_fraction : ℕ → ℚ
  | 0 => 4/9
  | n + 1 => shaded_fraction n + (4/9) * (1/4)^(n+1)

/-- The limit of the shaded fraction as the number of iterations approaches infinity -/
def shaded_limit : ℚ := 16/27

theorem shaded_area_equals_sixteen_twentyseventh :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |shaded_fraction n - shaded_limit| < ε :=
sorry

end NUMINAMATH_CALUDE_shaded_area_equals_sixteen_twentyseventh_l1107_110716


namespace NUMINAMATH_CALUDE_multiple_of_x_l1107_110715

theorem multiple_of_x (x y m : ℤ) : 
  (4 * x + y = 34) →
  (m * x - y = 20) →
  (y^2 = 4) →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_of_x_l1107_110715


namespace NUMINAMATH_CALUDE_ale_age_l1107_110791

/-- Represents a year-month combination -/
structure YearMonth where
  year : ℕ
  month : ℕ
  h_month_valid : month ≥ 1 ∧ month ≤ 12

/-- Calculates the age in years between two YearMonth dates -/
def ageInYears (birth death : YearMonth) : ℕ :=
  death.year - birth.year

theorem ale_age :
  let birth := YearMonth.mk 1859 1 (by simp)
  let death := YearMonth.mk 2014 8 (by simp)
  ageInYears birth death = 155 := by
  sorry

#check ale_age

end NUMINAMATH_CALUDE_ale_age_l1107_110791


namespace NUMINAMATH_CALUDE_speed_ratio_l1107_110780

/-- Represents the scenario of Xiaoqing and Xiaoqiang's journey --/
structure Journey where
  distance : ℝ
  walking_speed : ℝ
  motorcycle_speed : ℝ
  (walking_speed_pos : walking_speed > 0)
  (motorcycle_speed_pos : motorcycle_speed > 0)
  (distance_pos : distance > 0)

/-- The time taken for the entire journey is 2.5 times the direct trip --/
def journey_time_constraint (j : Journey) : Prop :=
  (j.distance / j.motorcycle_speed) * 2.5 = 
    (j.distance / j.motorcycle_speed) + 
    (j.distance / j.motorcycle_speed - j.distance / j.walking_speed)

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio (j : Journey) 
  (h : journey_time_constraint j) : 
  j.motorcycle_speed / j.walking_speed = 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l1107_110780


namespace NUMINAMATH_CALUDE_three_in_all_curriculums_l1107_110724

/-- Represents the number of people in each curriculum or combination of curriculums -/
structure CurriculumParticipants where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ
  allCurriculums : ℕ

/-- Theorem stating that given the conditions, 3 people participate in all curriculums -/
theorem three_in_all_curriculums (p : CurriculumParticipants) 
  (h1 : p.yoga = 25)
  (h2 : p.cooking = 15)
  (h3 : p.weaving = 8)
  (h4 : p.cookingOnly = 2)
  (h5 : p.cookingAndYoga = 7)
  (h6 : p.cookingAndWeaving = 3)
  (h7 : p.cooking = p.cookingOnly + p.cookingAndYoga + p.cookingAndWeaving + p.allCurriculums) :
  p.allCurriculums = 3 := by
  sorry


end NUMINAMATH_CALUDE_three_in_all_curriculums_l1107_110724


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_relation_l1107_110741

theorem circle_radius_from_area_circumference_relation : 
  ∀ r : ℝ, r > 0 → (3 * (2 * Real.pi * r) = Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_relation_l1107_110741


namespace NUMINAMATH_CALUDE_height_difference_l1107_110768

theorem height_difference (parker daisy reese : ℕ) 
  (h1 : daisy = reese + 8)
  (h2 : reese = 60)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  daisy - parker = 4 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l1107_110768


namespace NUMINAMATH_CALUDE_correct_num_pigs_l1107_110785

/-- The number of pigs Randy has -/
def num_pigs : ℕ := 2

/-- The amount of feed per pig per day in pounds -/
def feed_per_pig_per_day : ℕ := 10

/-- The total amount of feed for all pigs per week in pounds -/
def total_feed_per_week : ℕ := 140

/-- Theorem stating that the number of pigs is correct given the feeding conditions -/
theorem correct_num_pigs : 
  num_pigs * feed_per_pig_per_day * 7 = total_feed_per_week := by
  sorry


end NUMINAMATH_CALUDE_correct_num_pigs_l1107_110785


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1107_110710

/-- The speed of a boat in still water, given that:
    1. It takes 90 minutes less to travel 36 miles downstream than upstream.
    2. The speed of the stream is 2 mph. -/
theorem boat_speed_in_still_water : ∃ (b : ℝ),
  (36 / (b - 2) - 36 / (b + 2) = 1.5) ∧ b = 10 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1107_110710


namespace NUMINAMATH_CALUDE_honor_roll_fraction_l1107_110761

theorem honor_roll_fraction (female_honor : Rat) (male_honor : Rat) (female_ratio : Rat) : 
  female_honor = 7/12 →
  male_honor = 11/15 →
  female_ratio = 13/27 →
  (female_ratio * female_honor) + ((1 - female_ratio) * male_honor) = 1071/1620 := by
sorry

end NUMINAMATH_CALUDE_honor_roll_fraction_l1107_110761


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1107_110751

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1107_110751


namespace NUMINAMATH_CALUDE_largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l1107_110703

theorem largest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem largest_integer_value : ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) :=
  sorry

theorem negative_four_satisfies : (5 - 4*(-4) > 17) :=
  sorry

theorem largest_integer_is_negative_four : 
  ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) ∧ (-4 ≤ x) → x = -4 :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l1107_110703


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1107_110775

theorem inequality_equivalence (x : ℝ) : (x - 2)^2 < 9 ↔ -1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1107_110775


namespace NUMINAMATH_CALUDE_sqrt_one_third_equals_sqrt_three_over_three_l1107_110760

theorem sqrt_one_third_equals_sqrt_three_over_three :
  Real.sqrt (1 / 3) = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_equals_sqrt_three_over_three_l1107_110760


namespace NUMINAMATH_CALUDE_impossible_transformation_number_54_impossible_l1107_110740

/-- Represents the allowed operations on the number -/
inductive Operation
  | Multiply2
  | Multiply3
  | Divide2
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply2 => n * 2
  | Operation.Multiply3 => n * 3
  | Operation.Divide2 => n / 2
  | Operation.Divide3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Returns the sum of exponents in the prime factorization of a number -/
def sumOfExponents (n : ℕ) : ℕ :=
  (Nat.factorization n).sum (fun _ e => e)

/-- Theorem stating that it's impossible to transform 12 into 54 with exactly 60 operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation), ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry

/-- Corollary: The number 54 cannot appear on the screen after exactly one minute -/
theorem number_54_impossible : ∃ (ops : List Operation), ops.length = 60 ∧ applyOperations 12 ops = 54 → False := by
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_number_54_impossible_l1107_110740


namespace NUMINAMATH_CALUDE_abc_product_l1107_110792

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24) (hac : a * c = 40) (hbc : b * c = 60) :
  a * b * c = 240 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1107_110792


namespace NUMINAMATH_CALUDE_a_subset_M_l1107_110722

noncomputable def a : ℝ := Real.sqrt 3

def M : Set ℝ := {x | x ≤ 3}

theorem a_subset_M : {a} ⊆ M := by sorry

end NUMINAMATH_CALUDE_a_subset_M_l1107_110722


namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1107_110708

/-- The solution to the system of equations 4x - 6y = -3 and 9x + 3y = 6.3 -/
def solution_pair : ℝ × ℝ := (0.436, 0.792)

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop := 4 * x - 6 * y = -3

/-- The second equation of the system -/
def equation2 (x y : ℝ) : Prop := 9 * x + 3 * y = 6.3

/-- Theorem stating that the solution_pair satisfies both equations -/
theorem solution_satisfies_system : 
  let (x, y) := solution_pair
  equation1 x y ∧ equation2 x y :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = solution_pair :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1107_110708


namespace NUMINAMATH_CALUDE_kitten_growth_theorem_l1107_110750

/-- Represents the length of a kitten at different stages of growth -/
structure KittenGrowth where
  initial_length : ℝ
  first_double : ℝ
  second_double : ℝ

/-- Theorem stating that if a kitten's length doubles twice and ends at 16 inches, its initial length was 4 inches -/
theorem kitten_growth_theorem (k : KittenGrowth) :
  k.second_double = 16 ∧ 
  k.first_double = 2 * k.initial_length ∧ 
  k.second_double = 2 * k.first_double →
  k.initial_length = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_kitten_growth_theorem_l1107_110750


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l1107_110765

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) : 
  ∃ (c : ℝ), c = r * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l1107_110765


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l1107_110777

theorem total_weight_of_balls (blue_weight brown_weight green_weight : ℝ) 
  (h1 : blue_weight = 6)
  (h2 : brown_weight = 3.12)
  (h3 : green_weight = 4.5) :
  blue_weight + brown_weight + green_weight = 13.62 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l1107_110777


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1107_110718

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                   -- One side is 10 feet
  2 * W + L = 146 →          -- Total fencing is 146 feet
  L * W = 680 :=             -- Area of the field is 680 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1107_110718


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1107_110754

theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (2, -1) →
  (a.1 + x * b.1, a.2 + x * b.2) • b = 0 →
  x = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1107_110754


namespace NUMINAMATH_CALUDE_select_with_defective_test_methods_l1107_110706

-- Define the total number of products and the number of defective products
def total_products : ℕ := 10
def defective_products : ℕ := 3

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem for the first question
theorem select_with_defective :
  binomial total_products 3 - binomial (total_products - defective_products) 3 = 85 :=
sorry

-- Theorem for the second question
theorem test_methods :
  binomial defective_products 1 * (binomial (total_products - defective_products) 2 * binomial 2 2) * binomial 4 4 = 1512 :=
sorry

end NUMINAMATH_CALUDE_select_with_defective_test_methods_l1107_110706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1107_110796

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 5 < seq.S 6 ∧ seq.S 6 = seq.S 7 ∧ seq.S 7 > seq.S 8

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ d, ∀ n, seq.a (n + 1) - seq.a n = d ∧ d < 0) ∧ 
  seq.S 9 < seq.S 5 ∧
  seq.a 7 = 0 ∧
  (∀ n, seq.S n ≤ seq.S 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1107_110796


namespace NUMINAMATH_CALUDE_second_class_size_l1107_110763

/-- Given two classes of students, where:
    - The first class has 24 students with an average mark of 40
    - The second class has an unknown number of students with an average mark of 60
    - The average mark of all students combined is 53.513513513513516
    This theorem proves that the number of students in the second class is 50. -/
theorem second_class_size (n : ℕ) :
  let first_class_size : ℕ := 24
  let first_class_avg : ℝ := 40
  let second_class_avg : ℝ := 60
  let total_avg : ℝ := 53.513513513513516
  let total_size : ℕ := first_class_size + n
  (first_class_size * first_class_avg + n * second_class_avg) / total_size = total_avg →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_second_class_size_l1107_110763


namespace NUMINAMATH_CALUDE_remainder_problem_l1107_110712

theorem remainder_problem (k : ℕ+) (h : 80 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1107_110712


namespace NUMINAMATH_CALUDE_winning_configurations_l1107_110737

/-- Represents a wall configuration in the brick removal game -/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the second player -/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The list of all possible starting configurations -/
def startingConfigs : List WallConfig :=
  [⟨[7, 3, 2]⟩, ⟨[7, 4, 1]⟩, ⟨[8, 3, 1]⟩, ⟨[7, 2, 2]⟩, ⟨[7, 3, 3]⟩]

/-- The main theorem to be proved -/
theorem winning_configurations :
  (∀ c ∈ startingConfigs, isWinningForSecondPlayer c ↔ (c = ⟨[7, 3, 2]⟩ ∨ c = ⟨[8, 3, 1]⟩)) :=
  sorry

end NUMINAMATH_CALUDE_winning_configurations_l1107_110737


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l1107_110713

theorem zero_exponent_rule (x : ℚ) (h : x ≠ 0) : x ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l1107_110713


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_4x_l1107_110758

theorem factorization_xy_squared_minus_4x (x y : ℝ) : 
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_4x_l1107_110758


namespace NUMINAMATH_CALUDE_final_daisy_count_l1107_110793

/-- Represents the number of flowers in Laura's garden -/
structure GardenFlowers where
  daisies : ℕ
  tulips : ℕ

/-- Represents the ratio of daisies to tulips -/
structure FlowerRatio where
  daisy : ℕ
  tulip : ℕ

/-- Theorem stating the final number of daisies after adding tulips while maintaining the ratio -/
theorem final_daisy_count 
  (initial : GardenFlowers) 
  (ratio : FlowerRatio) 
  (added_tulips : ℕ) : 
  (ratio.daisy : ℚ) / (ratio.tulip : ℚ) = (initial.daisies : ℚ) / (initial.tulips : ℚ) →
  initial.tulips = 32 →
  added_tulips = 24 →
  ratio.daisy = 3 →
  ratio.tulip = 4 →
  let final_tulips := initial.tulips + added_tulips
  let final_daisies := (ratio.daisy : ℚ) / (ratio.tulip : ℚ) * final_tulips
  final_daisies = 42 := by
  sorry


end NUMINAMATH_CALUDE_final_daisy_count_l1107_110793


namespace NUMINAMATH_CALUDE_fourth_side_length_l1107_110738

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- The quadrilateral is inscribed in the circle -/
  inscribed : True
  /-- The quadrilateral is not a rectangle -/
  not_rectangle : True

/-- Theorem: In a quadrilateral inscribed in a circle with radius 150√2,
    if three sides have length 150, then the fourth side has length 300√2 -/
theorem fourth_side_length (q : InscribedQuadrilateral)
    (h_radius : q.radius = 150 * Real.sqrt 2)
    (h_side1 : q.side1 = 150)
    (h_side2 : q.side2 = 150)
    (h_side3 : q.side3 = 150) :
    q.side4 = 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1107_110738


namespace NUMINAMATH_CALUDE_combined_efficiency_approx_38_l1107_110734

-- Define the fuel efficiencies and distance
def jane_efficiency : ℚ := 30
def mike_efficiency : ℚ := 15
def carl_efficiency : ℚ := 20
def distance : ℚ := 100

-- Define the combined fuel efficiency function
def combined_efficiency (e1 e2 e3 d : ℚ) : ℚ :=
  (3 * d) / (d / e1 + d / e2 + d / e3)

-- State the theorem
theorem combined_efficiency_approx_38 :
  ∃ ε > 0, abs (combined_efficiency jane_efficiency mike_efficiency carl_efficiency distance - 38) < ε :=
by sorry

end NUMINAMATH_CALUDE_combined_efficiency_approx_38_l1107_110734


namespace NUMINAMATH_CALUDE_symmetry_axis_phi_l1107_110719

/-- The value of φ when f(x) and g(x) have the same axis of symmetry --/
theorem symmetry_axis_phi : ∀ (ω : ℝ), ω > 0 →
  (∀ (φ : ℝ), |φ| < π/2 →
    (∀ (x : ℝ), 3 * Real.sin (ω * x - π/3) = 3 * Real.sin (ω * x + φ + π/2)) →
    φ = π/6) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_phi_l1107_110719


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l1107_110739

/-- Represents the number of chairs in a complete row -/
def chairs_per_row : ℕ := 11

/-- Represents the initial total number of chairs -/
def initial_chairs : ℕ := 110

/-- Represents the number of students attending the assembly -/
def students : ℕ := 70

/-- Represents the number of chairs to be removed -/
def chairs_to_remove : ℕ := 33

/-- Proves that removing 33 chairs results in the optimal arrangement -/
theorem optimal_chair_removal :
  let remaining_chairs := initial_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧
  (remaining_chairs ≥ students) ∧
  (∀ n : ℕ, n < chairs_to_remove →
    ((initial_chairs - n) % chairs_per_row = 0) →
    (initial_chairs - n < students ∨ initial_chairs - n > remaining_chairs)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l1107_110739


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l1107_110786

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ m : ℤ, 5 * (n^2 + 2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l1107_110786


namespace NUMINAMATH_CALUDE_andy_max_demerits_l1107_110787

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def maxDemerits : ℕ := by sorry

/-- The number of demerits Andy gets per instance of showing up late -/
def demeritsPerLateInstance : ℕ := 2

/-- The number of times Andy showed up late -/
def lateInstances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demeritsForJoke : ℕ := 15

/-- The number of additional demerits Andy can get before getting fired -/
def remainingDemerits : ℕ := 23

theorem andy_max_demerits :
  maxDemerits = demeritsPerLateInstance * lateInstances + demeritsForJoke + remainingDemerits := by
  sorry

end NUMINAMATH_CALUDE_andy_max_demerits_l1107_110787


namespace NUMINAMATH_CALUDE_basketball_lineup_selection_l1107_110770

theorem basketball_lineup_selection (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 1) :
  n * Nat.choose (n - k) (m - k) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_selection_l1107_110770


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1107_110794

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + 2*x^2 + b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1107_110794


namespace NUMINAMATH_CALUDE_odd_even_digit_difference_l1107_110736

/-- The upper bound of the range of integers we're considering -/
def upper_bound : ℕ := 8 * 10^20

/-- Counts the number of integers up to n (inclusive) that contain only odd digits -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- Counts the number of integers up to n (inclusive) that contain only even digits -/
def count_even_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating the difference between odd-digit-only and even-digit-only numbers -/
theorem odd_even_digit_difference :
  count_odd_digits upper_bound - count_even_digits upper_bound = (5^21 - 1) / 4 := by sorry

end NUMINAMATH_CALUDE_odd_even_digit_difference_l1107_110736


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1107_110788

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 / ((2*x + y) * (2*x + z)) + y^2 / ((2*y + x) * (2*y + z)) + z^2 / ((2*z + x) * (2*z + y)) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1107_110788


namespace NUMINAMATH_CALUDE_candy_distribution_l1107_110762

theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ) :
  initial_candies = 20 →
  friends = 6 →
  additional_candies = 4 →
  (initial_candies + additional_candies) / friends = 4 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1107_110762


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_l1107_110714

theorem ball_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 24 →
  hole_depth = 8 →
  (hole_diameter / 2) ^ 2 + (ball_radius - hole_depth) ^ 2 = ball_radius ^ 2 →
  ball_radius = 13 := by
sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_l1107_110714


namespace NUMINAMATH_CALUDE_area_of_efgh_is_72_l1107_110721

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two opposite corners -/
structure Rectangle where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The configuration of circles and rectangle in the problem -/
structure CircleConfiguration where
  efgh : Rectangle
  circleA : Circle
  circleB : Circle
  circleC : Circle
  circleD : Circle

/-- Checks if two circles are congruent -/
def areCongruentCircles (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a circle is tangent to two adjacent sides of a rectangle -/
def isTangentToAdjacentSides (c : Circle) (r : Rectangle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if the centers of four circles form a rectangle -/
def centersFormRectangle (c1 c2 c3 c4 : Circle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry -- Definition omitted for brevity

theorem area_of_efgh_is_72 (config : CircleConfiguration) :
  (areCongruentCircles config.circleA config.circleB) →
  (areCongruentCircles config.circleA config.circleC) →
  (areCongruentCircles config.circleA config.circleD) →
  (config.circleB.radius = 3) →
  (isTangentToAdjacentSides config.circleB config.efgh) →
  (centersFormRectangle config.circleA config.circleB config.circleC config.circleD) →
  rectangleArea config.efgh = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_efgh_is_72_l1107_110721


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_x_squared_l1107_110747

theorem integral_sqrt_minus_x_squared :
  ∫ x in (0 : ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - x^2) = π / 4 - 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_x_squared_l1107_110747


namespace NUMINAMATH_CALUDE_eleventh_inning_score_l1107_110701

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ

/-- Calculates the average score of a batsman -/
def average (stats : BatsmanStats) : ℚ :=
  stats.totalScore / stats.innings

/-- Theorem: Given the conditions, the score in the 11th inning is 110 runs -/
theorem eleventh_inning_score
  (stats10 : BatsmanStats)
  (stats11 : BatsmanStats)
  (h1 : stats10.innings = 10)
  (h2 : stats11.innings = 11)
  (h3 : average stats11 = 60)
  (h4 : average stats11 - average stats10 = 5) :
  stats11.totalScore - stats10.totalScore = 110 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_inning_score_l1107_110701


namespace NUMINAMATH_CALUDE_area_square_with_semicircles_l1107_110756

/-- The area of a shape formed by a square with semicircles on each side -/
theorem area_square_with_semicircles (π : ℝ) : 
  let square_side : ℝ := 2 * π
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := 1 / 2 * π * semicircle_radius ^ 2
  let total_semicircle_area : ℝ := 4 * semicircle_area
  let total_area : ℝ := square_area + total_semicircle_area
  total_area = 2 * π^2 * (π + 2) :=
by sorry

end NUMINAMATH_CALUDE_area_square_with_semicircles_l1107_110756


namespace NUMINAMATH_CALUDE_simplify_expression_l1107_110709

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3 - 2) / (3 * x^3))^2) =
  (Real.sqrt (x^12 - 2*x^9 + 6*x^6 - 2*x^3 + 4)) / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1107_110709
