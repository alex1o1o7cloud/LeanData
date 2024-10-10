import Mathlib

namespace complex_equality_l615_61540

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 :=
sorry

end complex_equality_l615_61540


namespace vendor_profit_calculation_l615_61509

/-- Calculates the profit for a vendor selling apples and oranges --/
def vendor_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                  (orange_buy_price : ℚ) (orange_sell_price : ℚ) 
                  (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := apple_sell_price - apple_buy_price
  let orange_profit := orange_sell_price - orange_buy_price
  apple_profit * apples_sold + orange_profit * oranges_sold

theorem vendor_profit_calculation :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1    -- $1 each
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  vendor_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry


end vendor_profit_calculation_l615_61509


namespace intersection_on_diagonal_l615_61590

-- Define the basic geometric objects
variable (A B C D K L M P Q : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define points K, L, M on sides or their extensions
def on_side_or_extension (X Y Z : EuclideanPlane) : Prop := sorry

-- Define the intersection of two lines
def intersect (W X Y Z : EuclideanPlane) : EuclideanPlane := sorry

-- Define a point lying on a line
def lies_on (X Y Z : EuclideanPlane) : Prop := sorry

-- Theorem statement
theorem intersection_on_diagonal 
  (h_quad : is_quadrilateral A B C D)
  (h_K : on_side_or_extension K A B)
  (h_L : on_side_or_extension L B C)
  (h_M : on_side_or_extension M C D)
  (h_P : P = intersect K L A C)
  (h_Q : Q = intersect L M B D) :
  lies_on (intersect K Q M P) A D := by sorry

end intersection_on_diagonal_l615_61590


namespace roll_five_dice_probability_l615_61500

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to roll an equal number of 1's and 6's -/
def equal_ones_and_sixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's -/
def prob_more_ones_than_sixes : ℚ := 167 / 486

theorem roll_five_dice_probability :
  prob_more_ones_than_sixes = 1 / 2 * (1 - equal_ones_and_sixes / total_outcomes) :=
sorry

end roll_five_dice_probability_l615_61500


namespace function_value_at_half_l615_61516

theorem function_value_at_half (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f (Real.sin x) = x) →
  f (1 / 2) = π / 6 := by
  sorry

end function_value_at_half_l615_61516


namespace city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l615_61554

/-- Represents a survey scenario -/
inductive SurveyScenario
  | class_myopia
  | grade_morning_exercise
  | class_body_temperature
  | city_extracurricular_reading

/-- Determines if a survey scenario is suitable for sampling -/
def suitable_for_sampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.city_extracurricular_reading => True
  | _ => False

/-- Theorem stating that the city-wide extracurricular reading survey is suitable for sampling -/
theorem city_reading_survey_suitable :
  suitable_for_sampling SurveyScenario.city_extracurricular_reading :=
by
  sorry

/-- Theorem stating that the class myopia survey is not suitable for sampling -/
theorem class_myopia_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_myopia :=
by
  sorry

/-- Theorem stating that the grade morning exercise survey is not suitable for sampling -/
theorem grade_exercise_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.grade_morning_exercise :=
by
  sorry

/-- Theorem stating that the class body temperature survey is not suitable for sampling -/
theorem class_temperature_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_body_temperature :=
by
  sorry

end city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l615_61554


namespace tony_total_cost_l615_61524

/-- Represents the total cost of Tony's purchases at the toy store -/
def total_cost (lego_price toy_sword_price play_dough_price : ℝ)
               (lego_sets toy_swords play_doughs : ℕ)
               (first_day_discount second_day_discount sales_tax : ℝ) : ℝ :=
  let first_day_cost := (2 * lego_price + 3 * toy_sword_price) * (1 - first_day_discount) * (1 + sales_tax)
  let second_day_cost := ((lego_sets - 2) * lego_price + (toy_swords - 3) * toy_sword_price + play_doughs * play_dough_price) * (1 - second_day_discount) * (1 + sales_tax)
  first_day_cost + second_day_cost

/-- Theorem stating that Tony's total cost matches the calculated amount -/
theorem tony_total_cost :
  total_cost 250 120 35 3 5 10 0.2 0.1 0.05 = 1516.20 := by
  sorry

end tony_total_cost_l615_61524


namespace two_digit_minus_reverse_63_l615_61505

/-- Reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_minus_reverse_63 (n : ℕ) :
  is_two_digit n ∧ n - reverse_two_digit n = 63 → n = 81 ∨ n = 92 :=
by sorry

end two_digit_minus_reverse_63_l615_61505


namespace non_working_games_l615_61530

theorem non_working_games (total : ℕ) (working : ℕ) (h1 : total = 30) (h2 : working = 17) :
  total - working = 13 := by
  sorry

end non_working_games_l615_61530


namespace inverse_function_property_l615_61534

-- Define the function f and its inverse g
variable (f g : ℝ → ℝ)

-- Define the property that g is the inverse of f
variable (h₁ : ∀ x, g (f x) = x)
variable (h₂ : ∀ x, f (g x) = x)

-- Define the given property of f
variable (h₃ : ∀ a b, f (a * b) = f a + f b)

-- Theorem to prove
theorem inverse_function_property :
  ∀ a b, g (a + b) = g a * g b :=
sorry

end inverse_function_property_l615_61534


namespace initial_ratio_of_men_to_women_l615_61519

theorem initial_ratio_of_men_to_women 
  (initial_men : ℕ) 
  (initial_women : ℕ) 
  (final_men : ℕ) 
  (final_women : ℕ) 
  (h1 : final_men = initial_men + 2)
  (h2 : final_women = 2 * (initial_women - 3))
  (h3 : final_men = 14)
  (h4 : final_women = 24) :
  initial_men / initial_women = 4 / 5 := by
sorry

end initial_ratio_of_men_to_women_l615_61519


namespace sin_2x_eq_cos_x_div_2_solutions_l615_61567

theorem sin_2x_eq_cos_x_div_2_solutions (x : ℝ) : 
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ s, Real.sin (2 * x) = Real.cos (x / 2)) ∧
    (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ Real.sin (2 * y) = Real.cos (y / 2) → y ∈ s) :=
sorry

end sin_2x_eq_cos_x_div_2_solutions_l615_61567


namespace area_of_inner_triangle_l615_61582

/-- Given a triangle and points dividing its sides in a 1:2 ratio, 
    the area of the new triangle formed by these points is 1/9 of the original triangle's area. -/
theorem area_of_inner_triangle (T : ℝ) (h : T > 0) :
  ∃ (A : ℝ), A = T / 9 ∧ A > 0 := by
  sorry

end area_of_inner_triangle_l615_61582


namespace composite_sequences_exist_l615_61508

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def consecutive_composites (start : ℕ) (len : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range len → is_composite (start + i)

theorem composite_sequences_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ consecutive_composites start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ consecutive_composites start 11) :=
sorry

end composite_sequences_exist_l615_61508


namespace polynomial_factorization_l615_61550

theorem polynomial_factorization (y : ℝ) : 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) := by
  sorry

end polynomial_factorization_l615_61550


namespace milk_measurement_problem_l615_61576

/-- Represents a container for milk -/
structure Container :=
  (capacity : ℕ)
  (content : ℕ)

/-- Represents the state of all containers -/
structure State :=
  (can1 : Container)
  (can2 : Container)
  (jug5 : Container)
  (jug4 : Container)

/-- Represents a pouring operation -/
inductive Operation
  | CanToJug : Container → Container → Operation
  | JugToJug : Container → Container → Operation
  | JugToCan : Container → Container → Operation

/-- The result of applying an operation to a state -/
def applyOperation (s : State) (op : Operation) : State :=
  sorry

/-- The initial state with two full 80-liter cans and empty jugs -/
def initialState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 80⟩
  , jug5 := ⟨5, 0⟩
  , jug4 := ⟨4, 0⟩ }

/-- The goal state with exactly 2 liters in each jug -/
def goalState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 76⟩
  , jug5 := ⟨5, 2⟩
  , jug4 := ⟨4, 2⟩ }

/-- Theorem stating that the goal state can be reached in exactly 9 operations -/
theorem milk_measurement_problem :
  ∃ (ops : List Operation),
    ops.length = 9 ∧
    (ops.foldl applyOperation initialState) = goalState :=
  sorry

end milk_measurement_problem_l615_61576


namespace remainder_23_pow_2003_mod_7_l615_61512

theorem remainder_23_pow_2003_mod_7 : 23^2003 % 7 = 4 := by
  sorry

end remainder_23_pow_2003_mod_7_l615_61512


namespace shooting_scores_theorem_l615_61533

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

theorem shooting_scores_theorem :
  let avg_A := (scores_A.sum) / (scores_A.length : ℝ)
  let avg_B := (scores_B.sum) / (scores_B.length : ℝ)
  let avg_total := ((scores_A ++ scores_B).sum) / ((scores_A ++ scores_B).length : ℝ)
  (avg_A < avg_B) ∧ (avg_total = 6.6) := by
  sorry

end shooting_scores_theorem_l615_61533


namespace cube_edge_length_l615_61563

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 150) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 5 := by
  sorry

end cube_edge_length_l615_61563


namespace trajectory_equation_tangent_relation_constant_triangle_area_l615_61583

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) + Real.sqrt ((x - Real.sqrt 3)^2 + y^2) = 4

theorem trajectory_equation (x y : ℝ) :
  trajectory x y → x^2/4 + y^2 = 1 := by sorry

theorem tangent_relation (x y k m : ℝ) :
  trajectory x y → y = k*x + m → m^2 = 1 + 4*k^2 := by sorry

theorem constant_triangle_area (x y k m : ℝ) (A B : ℝ × ℝ) :
  trajectory x y →
  y = k*x + m →
  A.1^2/16 + A.2^2/4 = 1 →
  B.1^2/16 + B.2^2/4 = 1 →
  A.2 = k*A.1 + m →
  B.2 = k*B.1 + m →
  (1/2) * |m| * |A.1 - B.1| = 2 * Real.sqrt 3 := by sorry

end trajectory_equation_tangent_relation_constant_triangle_area_l615_61583


namespace silver_cost_per_ounce_l615_61561

theorem silver_cost_per_ounce
  (silver_amount : ℝ)
  (gold_amount : ℝ)
  (gold_silver_price_ratio : ℝ)
  (total_spent : ℝ)
  (h1 : silver_amount = 1.5)
  (h2 : gold_amount = 2 * silver_amount)
  (h3 : gold_silver_price_ratio = 50)
  (h4 : total_spent = 3030)
  (h5 : silver_amount * silver_cost + gold_amount * (gold_silver_price_ratio * silver_cost) = total_spent) :
  silver_cost = 20 :=
by sorry

end silver_cost_per_ounce_l615_61561


namespace midpoint_product_l615_61587

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end midpoint_product_l615_61587


namespace cannot_afford_both_phones_l615_61569

/-- Represents the financial situation of a couple --/
structure FinancialSituation where
  income : ℕ
  expenses : ℕ
  phoneACost : ℕ
  phoneBCost : ℕ

/-- Determines if a couple can afford to buy both phones --/
def canAffordBothPhones (situation : FinancialSituation) : Prop :=
  situation.income - situation.expenses ≥ situation.phoneACost + situation.phoneBCost

/-- The specific financial situation of Alexander and Natalia --/
def alexanderAndNatalia : FinancialSituation :=
  { income := 186000
    expenses := 119000
    phoneACost := 57000
    phoneBCost := 37000 }

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones :
  ¬(canAffordBothPhones alexanderAndNatalia) := by
  sorry


end cannot_afford_both_phones_l615_61569


namespace polynomial_factorization_l615_61568

theorem polynomial_factorization :
  ∀ x : ℝ, x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end polynomial_factorization_l615_61568


namespace pie_eating_contest_l615_61541

theorem pie_eating_contest :
  let student1 : ℚ := 8 / 9
  let student2 : ℚ := 5 / 6
  let student3 : ℚ := 2 / 3
  student1 + student2 + student3 = 43 / 18 :=
by sorry

end pie_eating_contest_l615_61541


namespace pipe_a_fills_in_12_hours_l615_61532

/-- Represents the time (in hours) taken by pipe A to fill the cistern -/
def pipe_a_time : ℝ := 12

/-- Represents the time (in hours) taken by pipe B to leak out the cistern -/
def pipe_b_time : ℝ := 18

/-- Represents the time (in hours) taken to fill the cistern when both pipes are open -/
def both_pipes_time : ℝ := 36

/-- Proves that pipe A fills the cistern in 12 hours given the conditions -/
theorem pipe_a_fills_in_12_hours :
  (1 / pipe_a_time) - (1 / pipe_b_time) = (1 / both_pipes_time) :=
by sorry

end pipe_a_fills_in_12_hours_l615_61532


namespace sale_discount_proof_l615_61557

theorem sale_discount_proof (original_price : ℝ) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let final_price := (1 - coupon_discount) * sale_price
  final_price = 0.4 * original_price :=
by sorry

end sale_discount_proof_l615_61557


namespace quadrilateral_angle_proof_l615_61549

theorem quadrilateral_angle_proof (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 85 →
  B + C + D = 180 →
  D = 42.5 := by
sorry

end quadrilateral_angle_proof_l615_61549


namespace intersection_and_complement_l615_61555

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}

theorem intersection_and_complement :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (Set.compl (A ∩ B) = {x | x < -3 ∨ x > -2}) := by sorry

end intersection_and_complement_l615_61555


namespace tan_sum_identity_l615_61597

theorem tan_sum_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  Real.tan (α + π / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end tan_sum_identity_l615_61597


namespace series_convergence_power_l615_61526

theorem series_convergence_power (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_conv : Summable a) :
  Summable (fun n => (a n) ^ (n / (n + 1))) := by
sorry

end series_convergence_power_l615_61526


namespace parallel_vectors_x_value_l615_61577

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end parallel_vectors_x_value_l615_61577


namespace martha_cards_l615_61566

theorem martha_cards (x : ℝ) : x - 3.0 = 73 → x = 76 := by
  sorry

end martha_cards_l615_61566


namespace cos_sin_sum_l615_61592

theorem cos_sin_sum (x : ℝ) (h : Real.cos (x - π/3) = 1/3) :
  Real.cos (2*x - 5*π/3) + Real.sin (π/3 - x)^2 = 5/3 := by
  sorry

end cos_sin_sum_l615_61592


namespace dice_roll_probability_l615_61547

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a 2 on the first die -/
def prob_first_die_2 : ℚ := 1 / num_sides

/-- The probability of rolling a 5 or 6 on the second die -/
def prob_second_die_5_or_6 : ℚ := 2 / num_sides

/-- The probability of the combined event -/
def prob_combined : ℚ := prob_first_die_2 * prob_second_die_5_or_6

theorem dice_roll_probability : prob_combined = 1 / 18 := by
  sorry

end dice_roll_probability_l615_61547


namespace ellipse_major_axis_length_l615_61596

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point2D
  focus2 : Point2D
  majorAxis : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if the line intersects the ellipse at exactly one point -/
def intersectsAtOnePoint (e : Ellipse) (l : Line2D) : Prop :=
  sorry

theorem ellipse_major_axis_length :
  ∀ (e : Ellipse) (l : Line2D),
    e.focus1 = Point2D.mk (-2) 0 →
    e.focus2 = Point2D.mk 2 0 →
    l.a = 1 →
    l.b = Real.sqrt 3 →
    l.c = 4 →
    intersectsAtOnePoint e l →
    e.majorAxis = 2 * Real.sqrt 7 :=
  sorry

end ellipse_major_axis_length_l615_61596


namespace intersection_of_A_and_B_l615_61585

def set_A : Set ℝ := {x | 2 * x - 1 ≤ 0}
def set_B : Set ℝ := {x | 1 / x > 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1/2} := by
  sorry

end intersection_of_A_and_B_l615_61585


namespace tangent_line_circle_parabola_l615_61572

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the xy-plane -/
structure Parabola where
  vertex : ℝ × ℝ
  a : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle at a given point -/
def isTangentToCircle (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Checks if a line is tangent to a parabola at a given point -/
def isTangentToParabola (l : Line) (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_parabola (c : Circle) (p : Parabola) (l : Line) (point : ℝ × ℝ) :
  c.center = (1, 2) →
  c.radius^2 = 1^2 + 2^2 + a →
  p.vertex = (0, 0) →
  p.a = 1/4 →
  isTangentToCircle l c point →
  isTangentToParabola l p point →
  a = 3 := by sorry

end tangent_line_circle_parabola_l615_61572


namespace product_abcd_l615_61535

/-- Given positive real numbers a, b, c, and d satisfying the specified conditions,
    prove that their product equals 14400. -/
theorem product_abcd (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 762)
  (h_sum_ab_cd : a * b + c * d = 260)
  (h_sum_ac_bd : a * c + b * d = 365)
  (h_sum_ad_bc : a * d + b * c = 244) :
  a * b * c * d = 14400 := by
  sorry

end product_abcd_l615_61535


namespace f_has_max_and_min_l615_61562

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has both a maximum and a minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) ↔ 
  (a < -1 ∨ a > 2) :=
sorry

end f_has_max_and_min_l615_61562


namespace arithmetic_calculation_l615_61527

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 2 + 3^2 = 65 := by
  sorry

end arithmetic_calculation_l615_61527


namespace park_creatures_l615_61522

/-- The number of dogs at the park -/
def num_dogs : ℕ := 60

/-- The number of people at the park -/
def num_people : ℕ := num_dogs / 2

/-- The number of snakes at the park -/
def num_snakes : ℕ := num_people / 2

/-- The total number of eyes and legs of all creatures at the park -/
def total_eyes_and_legs : ℕ := 510

theorem park_creatures :
  (num_dogs = 2 * num_people) ∧
  (num_people = 2 * num_snakes) ∧
  (4 * num_dogs + 4 * num_people + 2 * num_snakes = total_eyes_and_legs) :=
by sorry

#check park_creatures

end park_creatures_l615_61522


namespace triangle_area_cosine_sum_maximum_l615_61501

theorem triangle_area_cosine_sum_maximum (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = Real.sqrt 3 →
  a^2 = b^2 + c^2 + b*c →
  S = (1/2) * a * b * Real.sin C →
  (∃ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k) →
  (∀ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k → k ≥ Real.sqrt 3) :=
by sorry

#check triangle_area_cosine_sum_maximum

end triangle_area_cosine_sum_maximum_l615_61501


namespace day_of_week_theorem_l615_61528

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : ℤ) (dayNumber : ℕ) : DayOfWeek := sorry

theorem day_of_week_theorem (M : ℤ) :
  dayOfWeek M 200 = DayOfWeek.Monday →
  dayOfWeek (M + 2) 300 = DayOfWeek.Monday →
  dayOfWeek (M - 1) 100 = DayOfWeek.Tuesday :=
by sorry

end day_of_week_theorem_l615_61528


namespace exists_integer_solution_l615_61518

theorem exists_integer_solution : ∃ x : ℤ, 2 * x^2 - 3 * x + 1 = 0 := by
  sorry

end exists_integer_solution_l615_61518


namespace constant_width_max_length_l615_61556

/-- A convex curve in a 2D plane. -/
structure ConvexCurve where
  -- Add necessary fields and conditions to define a convex curve
  is_convex : Bool
  diameter : ℝ
  length : ℝ

/-- A curve of constant width. -/
structure ConstantWidthCurve extends ConvexCurve where
  constant_width : ℝ
  is_constant_width : Bool

/-- The theorem stating that curves of constant width 1 have the greatest length among all convex curves of diameter 1. -/
theorem constant_width_max_length :
  ∀ (K : ConvexCurve),
    K.diameter = 1 →
    ∀ (C : ConstantWidthCurve),
      C.diameter = 1 →
      C.constant_width = 1 →
      C.is_constant_width →
      K.length ≤ C.length :=
sorry


end constant_width_max_length_l615_61556


namespace binary_product_example_l615_61586

/-- Given two binary numbers represented as natural numbers, 
    this function computes their product in binary representation -/
def binary_multiply (a b : ℕ) : ℕ := 
  (a.digits 2).foldl (λ acc d => acc * 2 + d) 0 * 
  (b.digits 2).foldl (λ acc d => acc * 2 + d) 0

/-- Theorem stating that the product of 1101₂ and 1011₂ is 10011011₂ -/
theorem binary_product_example : binary_multiply 13 11 = 155 := by
  sorry

#eval binary_multiply 13 11

end binary_product_example_l615_61586


namespace total_amount_in_euros_l615_61548

/-- Represents the distribution of shares among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the exchange rate from dollars to euros -/
def exchange_rate : ℝ := 0.85

/-- Defines the share ratios relative to w -/
def share_ratios : ShareDistribution := {
  w := 1,
  x := 0.75,
  y := 0.5,
  z := 0.25
}

/-- Theorem stating the total amount in euros given the conditions -/
theorem total_amount_in_euros : 
  ∀ (shares : ShareDistribution),
  shares.w * exchange_rate = 15 →
  (shares.w + shares.x + shares.y + shares.z) * exchange_rate = 37.5 :=
by
  sorry

#check total_amount_in_euros

end total_amount_in_euros_l615_61548


namespace platform_length_calculation_l615_61564

/-- Calculates the length of a platform given train and crossing information -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 470 →
  train_speed_kmh = 55 →
  crossing_time = 64.79481641468682 →
  ∃ (platform_length : ℝ), abs (platform_length - 520) < 0.1 :=
by
  sorry

end platform_length_calculation_l615_61564


namespace diagram3_illustrates_inflation_l615_61591

/-- Represents a diagram showing economic data over time -/
structure EconomicDiagram where
  prices : ℕ → ℝ
  time : ℕ

/-- Definition of inflation -/
def is_inflation (d : EconomicDiagram) : Prop :=
  ∀ t₁ t₂, t₁ < t₂ → d.prices t₁ < d.prices t₂

/-- Diagram №3 from the problem -/
def diagram3 : EconomicDiagram :=
  sorry

/-- Theorem stating that Diagram №3 illustrates inflation -/
theorem diagram3_illustrates_inflation : is_inflation diagram3 := by
  sorry

end diagram3_illustrates_inflation_l615_61591


namespace function_property_l615_61542

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 13/4 := by
  sorry

end function_property_l615_61542


namespace collinear_points_sum_l615_61553

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points in 3D space -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, t ≠ s ∧ 
    q.x = (1 - t) * p.x + t * r.x ∧
    q.y = (1 - t) * p.y + t * r.y ∧
    q.z = (1 - t) * p.z + t * r.z ∧
    q.x = (1 - s) * p.x + s * r.x ∧
    q.y = (1 - s) * p.y + s * r.y ∧
    q.z = (1 - s) * p.z + s * r.z

theorem collinear_points_sum (x y z : ℝ) :
  collinear (Point3D.mk x 1 z) (Point3D.mk 2 y z) (Point3D.mk x y 3) →
  x + y = 3 := by
  sorry

end collinear_points_sum_l615_61553


namespace email_difference_l615_61581

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 12

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := 44

/-- The difference between the number of emails Jack received in the morning and afternoon -/
theorem email_difference : morning_emails - afternoon_emails = 7 := by
  sorry

end email_difference_l615_61581


namespace quadratic_linear_system_solution_l615_61599

theorem quadratic_linear_system_solution :
  ∀ x y : ℝ,
  (x^2 - 6*x + 8 = 0) ∧ (y + 2*x = 12) →
  ((x = 4 ∧ y = 4) ∨ (x = 2 ∧ y = 8)) :=
by sorry

end quadratic_linear_system_solution_l615_61599


namespace quadratic_inequality_solution_set_l615_61546

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
sorry

end quadratic_inequality_solution_set_l615_61546


namespace money_game_total_determinable_l615_61565

/-- Represents the money redistribution game among four friends -/
structure MoneyGame where
  -- Initial amounts
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  ben_initial : ℝ
  -- Final amounts
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ
  ben_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  -- Amy's turn
  let amy_after := game.amy_initial - game.jan_initial - game.toy_initial - game.ben_initial
  let jan_after1 := 2 * game.jan_initial
  let toy_after1 := 2 * game.toy_initial
  let ben_after1 := 2 * game.ben_initial
  -- Jan's turn
  let amy_after2 := 2 * amy_after
  let toy_after2 := 2 * toy_after1
  let ben_after2 := 2 * ben_after1
  let jan_after2 := jan_after1 - (amy_after + toy_after1 + ben_after1)
  -- Toy's turn
  let amy_after3 := 2 * amy_after2
  let jan_after3 := 2 * jan_after2
  let ben_after3 := 2 * ben_after2
  let toy_after3 := toy_after2 - (amy_after2 + jan_after2 + ben_after2)
  -- Ben's turn
  game.amy_final = 2 * amy_after3 ∧
  game.jan_final = 2 * jan_after3 ∧
  game.toy_final = 2 * toy_after3 ∧
  game.ben_final = ben_after3 - (amy_after3 + jan_after3 + toy_after3)

/-- The theorem statement -/
theorem money_game_total_determinable (game : MoneyGame) :
  game.toy_initial = 24 ∧ 
  game.toy_final = 96 ∧ 
  redistribute game → 
  ∃ total : ℝ, total = game.amy_final + game.jan_final + game.toy_final + game.ben_final :=
by sorry


end money_game_total_determinable_l615_61565


namespace c1_minus_c4_equals_9_l615_61506

def f (c1 c2 c3 c4 x : ℕ) : ℕ := 
  (x^2 - 8*x + c1) * (x^2 - 8*x + c2) * (x^2 - 8*x + c3) * (x^2 - 8*x + c4)

theorem c1_minus_c4_equals_9 
  (c1 c2 c3 c4 : ℕ) 
  (h1 : c1 ≥ c2) 
  (h2 : c2 ≥ c3) 
  (h3 : c3 ≥ c4)
  (h4 : ∃ (M : Finset ℕ), M.card = 7 ∧ ∀ x ∈ M, f c1 c2 c3 c4 x = 0) :
  c1 - c4 = 9 := by
sorry

end c1_minus_c4_equals_9_l615_61506


namespace parabola_inequality_l615_61589

/-- Prove that for a parabola y = ax^2 + bx + c with a < 0, passing through points (-1, 0) and (m, 0) where 3 < m < 4, the inequality 3a + c > 0 holds. -/
theorem parabola_inequality (a b c m : ℝ) : 
  a < 0 → 
  3 < m → 
  m < 4 → 
  a * (-1)^2 + b * (-1) + c = 0 → 
  a * m^2 + b * m + c = 0 → 
  3 * a + c > 0 :=
by sorry

end parabola_inequality_l615_61589


namespace triangle_properties_l615_61502

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 2)
  (h2 : abc.c = 1)
  (h3 : Real.tan abc.A + Real.tan abc.B = -(Real.tan abc.A * Real.tan abc.B)) :
  (Real.tan (abc.A + abc.B) = 1) ∧ 
  (((2 : Real) - Real.sqrt 2) / 2 = 1/2 * abc.a * abc.b * Real.sin abc.C) := by
  sorry

end triangle_properties_l615_61502


namespace clock_angle_at_eight_thirty_l615_61544

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_eight_thirty :
  let hour_angle : ℝ := (8 * 30 + 30 / 2)
  let minute_angle : ℝ := 180
  let angle_diff : ℝ := |hour_angle - minute_angle|
  angle_diff = 75 := by
  sorry

end clock_angle_at_eight_thirty_l615_61544


namespace quadratic_roots_product_l615_61517

theorem quadratic_roots_product (a b : ℝ) : 
  (a^2 + 2012*a + 1 = 0) → 
  (b^2 + 2012*b + 1 = 0) → 
  (2 + 2013*a + a^2) * (2 + 2013*b + b^2) = -2010 := by
sorry

end quadratic_roots_product_l615_61517


namespace cookie_jar_problem_l615_61570

theorem cookie_jar_problem :
  ∃ (n c : ℕ),
    12 ≤ n ∧ n ≤ 36 ∧
    (n - 1) * c + (c + 1) = 1000 ∧
    n + (c + 1) = 65 :=
by sorry

end cookie_jar_problem_l615_61570


namespace expression_evaluation_l615_61521

theorem expression_evaluation (a b c : ℝ) 
  (h1 : a = 2)
  (h2 : b = a + 4)
  (h3 : c = b - 20)
  (h4 : a^2 + a ≠ 0)
  (h5 : b^2 - 6*b + 8 ≠ 0)
  (h6 : c^2 + 12*c + 36 ≠ 0) :
  (a^2 + 2*a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6*b + 8) * (c^2 + 16*c + 64) / (c^2 + 12*c + 36) = 3/4 := by
  sorry

end expression_evaluation_l615_61521


namespace sally_cracker_sales_l615_61575

theorem sally_cracker_sales (saturday_sales : ℕ) (sunday_increase_percent : ℕ) : 
  saturday_sales = 60 → 
  sunday_increase_percent = 50 → 
  saturday_sales + (saturday_sales + sunday_increase_percent * saturday_sales / 100) = 150 := by
sorry

end sally_cracker_sales_l615_61575


namespace right_triangle_inscribed_shapes_l615_61598

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square with vertex C inside the triangle
    2. The dimensions of the largest rectangle with vertex C inside the triangle -/
theorem right_triangle_inscribed_shapes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rect_width := a / 2
  let rect_height := b / 2
  (∀ s : ℝ, s > 0 ∧ s ≤ a ∧ s ≤ b → s ≤ square_side) ∧
  (∀ w h : ℝ, w > 0 ∧ h > 0 ∧ w ≤ a ∧ h ≤ b ∧ w / a + h / b ≤ 1 → w * h ≤ rect_width * rect_height) :=
by sorry

end right_triangle_inscribed_shapes_l615_61598


namespace smallest_factor_product_l615_61588

theorem smallest_factor_product (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 1764 → ¬(2^5 ∣ 936 * m ∧ 3^3 ∣ 936 * m ∧ 14^2 ∣ 936 * m)) ∧
  (2^5 ∣ 936 * 1764 ∧ 3^3 ∣ 936 * 1764 ∧ 14^2 ∣ 936 * 1764) :=
by sorry

end smallest_factor_product_l615_61588


namespace diagonal_cut_color_distribution_l615_61594

/-- Represents the color distribution of a scarf --/
structure ColorDistribution where
  white : ℚ
  grey : ℚ
  black : ℚ

/-- Represents a square scarf --/
structure SquareScarf where
  side_length : ℚ
  black_area : ℚ
  grey_area : ℚ

/-- Represents a triangular scarf obtained by cutting a square scarf diagonally --/
structure TriangularScarf where
  color_distribution : ColorDistribution

def diagonal_cut (s : SquareScarf) : (TriangularScarf × TriangularScarf) :=
  sorry

theorem diagonal_cut_color_distribution 
  (s : SquareScarf) 
  (h1 : s.black_area = 1/6) 
  (h2 : s.grey_area = 1/3) :
  let (t1, t2) := diagonal_cut s
  t1.color_distribution = { white := 3/4, grey := 2/9, black := 1/36 } ∧
  t2.color_distribution = { white := 1/4, grey := 4/9, black := 11/36 } :=
sorry

end diagonal_cut_color_distribution_l615_61594


namespace egg_problem_l615_61520

theorem egg_problem (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 100 →
  5 * x + 6 * y + 9 * z = 600 →
  (x = y ∨ y = z ∨ x = z) →
  x = 60 ∧ y = 20 := by
  sorry

end egg_problem_l615_61520


namespace imaginary_part_of_z_l615_61574

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + 2*i) / i
  Complex.im z = -1 := by sorry

end imaginary_part_of_z_l615_61574


namespace box_height_rounding_equivalence_l615_61507

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem box_height_rounding_equivalence :
  let height1 : ℕ := 53
  let height2 : ℕ := 78
  let correct_sum := height1 + height2
  let alice_sum := height1 + round_to_nearest_ten height2
  round_to_nearest_ten correct_sum = round_to_nearest_ten alice_sum :=
by
  sorry

end box_height_rounding_equivalence_l615_61507


namespace line_intersects_circle_r_range_l615_61536

/-- The range of r for a line intersecting a circle -/
theorem line_intersects_circle_r_range (α : Real) (r : Real) :
  (∃ x y : Real, x * Real.cos α + y * Real.sin α = 1 ∧ x^2 + y^2 = r^2) →
  r > 0 →
  r > 1 := by
  sorry

end line_intersects_circle_r_range_l615_61536


namespace stratified_sampling_example_l615_61543

/-- The number of ways to select students using stratified sampling -/
def stratified_sampling_selection (female_count male_count total_selected : ℕ) : ℕ :=
  let female_selected := (female_count * total_selected) / (female_count + male_count)
  let male_selected := total_selected - female_selected
  (Nat.choose female_count female_selected) * (Nat.choose male_count male_selected)

/-- Theorem: The number of ways to select 3 students from 8 female and 4 male students
    using stratified sampling by gender ratio is 112 -/
theorem stratified_sampling_example : stratified_sampling_selection 8 4 3 = 112 := by
  sorry

end stratified_sampling_example_l615_61543


namespace jellybean_problem_l615_61529

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  let total_initial := initial_bags * initial_average
  let new_average := initial_average + average_increase
  let total_new := (initial_bags + 1) * new_average
  total_new - total_initial = 362 :=
by sorry

end jellybean_problem_l615_61529


namespace crayon_selection_theorem_l615_61510

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def selected_crayons : ℕ := 5

theorem crayon_selection_theorem :
  (Nat.choose total_crayons selected_crayons - 
   Nat.choose (total_crayons - red_crayons) selected_crayons) = 2211 := by
  sorry

end crayon_selection_theorem_l615_61510


namespace intersection_of_A_and_B_l615_61511

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l615_61511


namespace max_a_value_l615_61580

/-- Given integers a and b satisfying the conditions, the maximum value of a is 23 -/
theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a ≤ 23 ∧ ∃ (a₀ b₀ : ℤ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀ + b₀ + a₀ * b₀ = 143 ∧ a₀ = 23 := by
  sorry

end max_a_value_l615_61580


namespace isosceles_triangle_base_l615_61525

theorem isosceles_triangle_base (t α : ℝ) (h_t : t > 0) (h_α : 0 < α ∧ α < π) :
  ∃ a : ℝ, a > 0 ∧ a = 2 * Real.sqrt (t * Real.tan (α / 2)) ∧
    ∃ b : ℝ, b > 0 ∧
      let m := b * Real.cos (α / 2)
      t = (1 / 2) * a * m ∧
      α = 2 * Real.arccos (m / b) :=
by
  sorry

end isosceles_triangle_base_l615_61525


namespace total_leaked_equals_1958_l615_61537

/-- Represents the data for an oil pipe leak -/
structure PipeLeak where
  name : String
  leakRate : ℕ  -- gallons per hour
  fixTime : ℕ  -- hours

/-- Calculates the total amount of oil leaked from a pipe during repair -/
def totalLeakedDuringRepair (pipe : PipeLeak) : ℕ :=
  pipe.leakRate * pipe.fixTime

/-- The set of all pipe leaks -/
def pipeLeaks : List PipeLeak := [
  { name := "A", leakRate := 25, fixTime := 10 },
  { name := "B", leakRate := 37, fixTime := 7 },
  { name := "C", leakRate := 55, fixTime := 12 },
  { name := "D", leakRate := 41, fixTime := 9 },
  { name := "E", leakRate := 30, fixTime := 14 }
]

/-- Calculates the total amount of oil leaked from all pipes during repair -/
def totalLeaked : ℕ :=
  (pipeLeaks.map totalLeakedDuringRepair).sum

theorem total_leaked_equals_1958 : totalLeaked = 1958 := by
  sorry

#eval totalLeaked  -- This will print the result

end total_leaked_equals_1958_l615_61537


namespace perpendicular_bisector_of_intersection_l615_61593

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (a1 a2 b1 b2 : ℝ),
    circle1 a1 a2 ∧ circle1 b1 b2 ∧
    circle2 a1 a2 ∧ circle2 b1 b2 ∧
    (∀ x y : ℝ, perp_bisector x y ↔ 
      ((x - a1)^2 + (y - a2)^2 = (x - b1)^2 + (y - b2)^2)) :=
sorry

end perpendicular_bisector_of_intersection_l615_61593


namespace smaller_number_in_ratio_l615_61503

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Two positive numbers
  b / a = 11 / 7 ∧  -- In the ratio 7:11
  b - a = 16  -- Larger number exceeds smaller by 16
  → a = 28 := by  -- The smaller number is 28
sorry

end smaller_number_in_ratio_l615_61503


namespace power_of_64_l615_61560

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end power_of_64_l615_61560


namespace quadratic_factorization_l615_61513

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l615_61513


namespace binary_111011001001_equals_3785_l615_61545

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011001001_equals_3785 :
  binary_to_decimal [true, false, false, true, false, false, true, true, false, true, true, true] = 3785 := by
  sorry

end binary_111011001001_equals_3785_l615_61545


namespace volleyball_scores_l615_61523

/-- Volleyball competition scores -/
theorem volleyball_scores (lizzie_score : ℕ) (nathalie_score : ℕ) (aimee_score : ℕ) (team_score : ℕ) :
  lizzie_score = 4 →
  nathalie_score = lizzie_score + 3 →
  aimee_score = 2 * (lizzie_score + nathalie_score) →
  team_score = 50 →
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
sorry


end volleyball_scores_l615_61523


namespace parabola_symmetric_points_l615_61514

/-- The parabola defined by y^2 = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line y = (1/4)x - 2 -/
def SymmetryLine (x y : ℝ) : Prop := y = (1/4)*x - 2

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def SymmetricPoints (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  SymmetryLine ((x₁ + x₂)/2) ((y₁ + y₂)/2)

/-- The equation of line AB: 4x + y - 15 = 0 -/
def LineAB (x y : ℝ) : Prop := 4*x + y - 15 = 0

theorem parabola_symmetric_points (x₁ y₁ x₂ y₂ : ℝ) :
  Parabola x₁ y₁ → Parabola x₂ y₂ → SymmetricPoints x₁ y₁ x₂ y₂ →
  ∀ x y, LineAB x y ↔ (y - y₁)/(x - x₁) = (y₂ - y)/(x₂ - x) :=
sorry

end parabola_symmetric_points_l615_61514


namespace circle_center_is_neg_two_three_l615_61595

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y + 1 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle with the given equation is (-2, 3) -/
theorem circle_center_is_neg_two_three :
  ∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 12 :=
sorry

end circle_center_is_neg_two_three_l615_61595


namespace total_bottles_l615_61578

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 9) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 17 := by
  sorry

end total_bottles_l615_61578


namespace cat_food_insufficiency_l615_61559

theorem cat_food_insufficiency (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end cat_food_insufficiency_l615_61559


namespace triangle_inequality_l615_61558

theorem triangle_inequality (a b c : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h4 : a * b + b * c + c * a = 18) : 
  1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3 > 1 / (a + b + c - 3) := by
  sorry

end triangle_inequality_l615_61558


namespace merchant_discount_theorem_l615_61573

/-- Proves that a merchant offering a 10% discount on a 20% markup results in an 8% profit -/
theorem merchant_discount_theorem (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) 
  (discount_percentage : ℝ) (h1 : markup_percentage = 20) (h2 : profit_percentage = 8) :
  discount_percentage = 10 ↔ 
    cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 
    cost_price * (1 + profit_percentage / 100) :=
by sorry

end merchant_discount_theorem_l615_61573


namespace reverse_digits_problem_l615_61504

/-- Given two two-digit numbers where the second is the reverse of the first,
    if their quotient is 1.75 and the product of the first with its tens digit
    is 3.5 times the second, then the numbers are 21 and 12. -/
theorem reverse_digits_problem (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧  -- x is a two-digit number
  10 ≤ y ∧ y < 100 ∧  -- y is a two-digit number
  y = (x % 10) * 10 + (x / 10) ∧  -- y is the reverse of x
  (x : ℚ) / y = 1.75 ∧  -- their quotient is 1.75
  x * (x / 10) = (7 * y) / 2  -- product of x and its tens digit is 3.5 times y
  → x = 21 ∧ y = 12 := by
sorry

end reverse_digits_problem_l615_61504


namespace inequality_solution_l615_61552

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 4) > 5 / x + 21 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end inequality_solution_l615_61552


namespace caroline_score_l615_61539

structure Player where
  name : String
  score : ℕ

def winning_score : ℕ := 21

theorem caroline_score (caroline anthony leo : Player)
  (h1 : anthony.score = 19)
  (h2 : leo.score = 28)
  (h3 : ∃ p : Player, p ∈ [caroline, anthony, leo] ∧ p.score = winning_score) :
  caroline.score = winning_score :=
sorry

end caroline_score_l615_61539


namespace harvey_sam_race_l615_61515

theorem harvey_sam_race (sam_miles harvey_miles : ℕ) : 
  sam_miles = 12 → 
  harvey_miles > sam_miles → 
  sam_miles + harvey_miles = 32 → 
  harvey_miles - sam_miles = 8 := by
sorry

end harvey_sam_race_l615_61515


namespace smallest_perfect_square_sum_24_consecutive_l615_61584

/-- The sum of 24 consecutive positive integers starting from n -/
def sum_24_consecutive (n : ℕ) : ℕ := 12 * (2 * n + 23)

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum_24_consecutive :
  (∃ n : ℕ, is_perfect_square (sum_24_consecutive n)) ∧
  (∀ n : ℕ, is_perfect_square (sum_24_consecutive n) → sum_24_consecutive n ≥ 300) :=
sorry

end smallest_perfect_square_sum_24_consecutive_l615_61584


namespace acute_triangle_inequality_l615_61538

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  R : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c
  inradius_positive : 0 < r
  circumradius_positive : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- State the theorem
theorem acute_triangle_inequality (t : AcuteTriangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * (t.R + t.r)^2 := by
  sorry

end acute_triangle_inequality_l615_61538


namespace sum_x_y_equals_nine_l615_61571

theorem sum_x_y_equals_nine (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 4) : 
  x + y = 9 := by
  sorry

end sum_x_y_equals_nine_l615_61571


namespace intersection_of_P_and_Q_l615_61531

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x ≥ 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end intersection_of_P_and_Q_l615_61531


namespace intersection_of_M_and_N_l615_61579

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end intersection_of_M_and_N_l615_61579


namespace quadratic_inequality_solution_set_l615_61551

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x < 0} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end quadratic_inequality_solution_set_l615_61551
