import Mathlib

namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2049_204910

/-- A parabola is defined by its vertex, directrix, and focus. -/
structure Parabola where
  vertex : ℝ × ℝ
  directrix : ℝ
  focus : ℝ × ℝ

/-- Given a parabola with vertex at (2,0) and directrix x = -1, its focus is at (5,0). -/
theorem parabola_focus_coordinates :
  ∀ p : Parabola, p.vertex = (2, 0) ∧ p.directrix = -1 → p.focus = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2049_204910


namespace NUMINAMATH_CALUDE_at_most_one_negative_l2049_204956

theorem at_most_one_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b - c ≤ 0 → b + c - a > 0 ∧ c + a - b > 0) ∧
  (b + c - a ≤ 0 → a + b - c > 0 ∧ c + a - b > 0) ∧
  (c + a - b ≤ 0 → a + b - c > 0 ∧ b + c - a > 0) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_negative_l2049_204956


namespace NUMINAMATH_CALUDE_dalton_has_excess_money_l2049_204944

def jump_rope_cost : ℝ := 7
def board_game_cost : ℝ := 12
def ball_cost : ℝ := 4
def jump_rope_discount : ℝ := 2
def ball_discount : ℝ := 1
def jump_rope_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def ball_quantity : ℕ := 4
def allowance_savings : ℝ := 30
def uncle_money : ℝ := 25
def grandma_money : ℝ := 10
def sales_tax_rate : ℝ := 0.08

def total_cost_before_discounts : ℝ :=
  jump_rope_cost * jump_rope_quantity +
  board_game_cost * board_game_quantity +
  ball_cost * ball_quantity

def total_discounts : ℝ :=
  jump_rope_discount * jump_rope_quantity +
  ball_discount * ball_quantity

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discounts

def sales_tax : ℝ :=
  total_cost_after_discounts * sales_tax_rate

def final_total_cost : ℝ :=
  total_cost_after_discounts + sales_tax

def total_money_dalton_has : ℝ :=
  allowance_savings + uncle_money + grandma_money

theorem dalton_has_excess_money :
  total_money_dalton_has - final_total_cost = 9.92 := by sorry

end NUMINAMATH_CALUDE_dalton_has_excess_money_l2049_204944


namespace NUMINAMATH_CALUDE_sum_base4_equals_l2049_204935

/-- Converts a base 4 number represented as a list of digits to a natural number -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem sum_base4_equals :
  base4ToNat [2, 1, 2] + base4ToNat [1, 0, 3] + base4ToNat [3, 2, 1] =
  base4ToNat [1, 0, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_l2049_204935


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l2049_204974

theorem equal_ratios_sum (P Q : ℚ) :
  (4 : ℚ) / 9 = P / 63 ∧ (4 : ℚ) / 9 = 108 / Q → P + Q = 271 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l2049_204974


namespace NUMINAMATH_CALUDE_infinitely_many_polynomials_l2049_204983

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that x, y, and z must satisfy -/
def SphereCondition (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 + 2*x*y*z = 1

/-- The condition that the polynomial P must satisfy -/
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ x y z : ℝ, SphereCondition x y z →
    P x^2 + P y^2 + P z^2 + 2*(P x)*(P y)*(P z) = 1

/-- The main theorem stating that there are infinitely many polynomials satisfying the condition -/
theorem infinitely_many_polynomials :
  ∃ (S : Set RealPolynomial), (Set.Infinite S) ∧ (∀ P ∈ S, PolynomialCondition P) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_polynomials_l2049_204983


namespace NUMINAMATH_CALUDE_aziz_parents_years_in_america_before_birth_l2049_204932

theorem aziz_parents_years_in_america_before_birth :
  let current_year : ℕ := 2021
  let aziz_age : ℕ := 36
  let parents_move_year : ℕ := 1982
  let aziz_birth_year : ℕ := current_year - aziz_age
  let years_before_birth : ℕ := aziz_birth_year - parents_move_year
  years_before_birth = 3 :=
by sorry

end NUMINAMATH_CALUDE_aziz_parents_years_in_america_before_birth_l2049_204932


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l2049_204920

def total_stuffed_animals (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  after_mom + (dad_multiplier * after_mom)

theorem willy_stuffed_animals :
  total_stuffed_animals 10 2 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l2049_204920


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2049_204941

/-- Given an article with a certain selling price, prove that the profit percent is 42.5%
    when selling at 2/3 of that price would result in a loss of 5%. -/
theorem profit_percent_calculation (P : ℝ) (C : ℝ) (h : (2/3) * P = 0.95 * C) :
  (P - C) / C * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l2049_204941


namespace NUMINAMATH_CALUDE_scientific_notation_4212000_l2049_204902

theorem scientific_notation_4212000 :
  4212000 = 4.212 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4212000_l2049_204902


namespace NUMINAMATH_CALUDE_grocer_bananas_purchase_l2049_204962

/-- The number of pounds of bananas purchased by a grocer -/
def bananas_purchased (buy_price : ℚ) (sell_price : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_price / 4 - buy_price / 3)

/-- Theorem stating that the grocer purchased 72 pounds of bananas -/
theorem grocer_bananas_purchase :
  bananas_purchased (1/2) (1/1) 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_grocer_bananas_purchase_l2049_204962


namespace NUMINAMATH_CALUDE_investment_return_percentage_l2049_204921

/-- Calculates the return percentage for a two-venture investment --/
def calculate_return_percentage (total_investment : ℚ) (investment1 : ℚ) (investment2 : ℚ) 
  (profit_percentage1 : ℚ) (loss_percentage2 : ℚ) : ℚ :=
  let profit1 := investment1 * profit_percentage1
  let loss2 := investment2 * loss_percentage2
  let net_income := profit1 - loss2
  (net_income / total_investment) * 100

/-- Theorem stating that the return percentage is 6.5% for the given investment scenario --/
theorem investment_return_percentage : 
  calculate_return_percentage 25000 16250 16250 (15/100) (5/100) = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_investment_return_percentage_l2049_204921


namespace NUMINAMATH_CALUDE_five_dice_not_same_l2049_204975

theorem five_dice_not_same (n : ℕ) (h : n = 8) :
  (1 - (n : ℚ) / n^5) = 4095 / 4096 :=
sorry

end NUMINAMATH_CALUDE_five_dice_not_same_l2049_204975


namespace NUMINAMATH_CALUDE_videotape_boxes_needed_l2049_204985

/-- Represents the duration of a program -/
structure Duration :=
  (value : ℝ)

/-- Represents a box of videotape -/
structure Box :=
  (capacity : ℝ)

/-- Represents the content to be recorded -/
structure Content :=
  (tvEpisodes : ℕ)
  (skits : ℕ)
  (songs : ℕ)

def Box.canRecord (b : Box) (d1 d2 : Duration) (n1 n2 : ℕ) : Prop :=
  n1 * d1.value + n2 * d2.value ≤ b.capacity

theorem videotape_boxes_needed 
  (tvDuration skitDuration songDuration : Duration)
  (box : Box)
  (content : Content)
  (h1 : box.canRecord tvDuration skitDuration 2 1)
  (h2 : box.canRecord skitDuration songDuration 2 3)
  (h3 : skitDuration.value > songDuration.value)
  (h4 : content.tvEpisodes = 7 ∧ content.skits = 11 ∧ content.songs = 20) :
  (∃ n : ℕ, n = 8 ∨ n = 9) ∧ 
  (∀ m : ℕ, m < 8 → 
    m * box.capacity < 
      content.tvEpisodes * tvDuration.value + 
      content.skits * skitDuration.value + 
      content.songs * songDuration.value) :=
sorry

end NUMINAMATH_CALUDE_videotape_boxes_needed_l2049_204985


namespace NUMINAMATH_CALUDE_solution_exists_l2049_204905

/-- The system of equations has at least one solution if and only if a is in the specified set -/
theorem solution_exists (a : ℝ) : 
  (∃ x y : ℝ, x - 1 = a * (y^3 - 1) ∧ 
              2 * x / (|y^3| + y^3) = Real.sqrt x ∧ 
              y > 0 ∧ x ≥ 0) ↔ 
  a < 0 ∨ (0 ≤ a ∧ a < 1) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l2049_204905


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2049_204950

theorem max_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := fun x => 10 * x - 2 * x^2
  ∃ (max_val : ℝ), max_val = 12.5 ∧ ∀ y : ℝ, f y ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2049_204950


namespace NUMINAMATH_CALUDE_problem_solution_l2049_204917

theorem problem_solution (a b c x y z : ℝ) 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5)
  (h2 : c / x + a / y + b / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2049_204917


namespace NUMINAMATH_CALUDE_river_road_cars_l2049_204924

theorem river_road_cars (B C : ℕ) 
  (h1 : B * 13 = C)  -- ratio of buses to cars is 1:13
  (h2 : B = C - 60)  -- there are 60 fewer buses than cars
  : C = 65 := by sorry

end NUMINAMATH_CALUDE_river_road_cars_l2049_204924


namespace NUMINAMATH_CALUDE_function_properties_l2049_204998

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a^(2*x) - 2*a^x + 1

theorem function_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, y < 1 → ∃ x : ℝ, f a x = y) ∧
  (∀ x : ℝ, f a x < 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≥ -7) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2049_204998


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2049_204958

def polynomial_remainder_problem (p : ℝ → ℝ) (r : ℝ → ℝ) : Prop :=
  (p (-1) = 2) ∧ 
  (p 3 = -2) ∧ 
  (p (-4) = 5) ∧ 
  (∃ q : ℝ → ℝ, ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * q x + r x) ∧
  (r (-5) = 6)

theorem polynomial_remainder_theorem :
  ∃ p r : ℝ → ℝ, polynomial_remainder_problem p r :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2049_204958


namespace NUMINAMATH_CALUDE_prob_not_blue_marble_l2049_204988

/-- Given odds ratio for an event --/
structure OddsRatio :=
  (for_event : ℕ)
  (against_event : ℕ)

/-- Calculates the probability of an event not occurring given its odds ratio --/
def probability_of_not_occurring (odds : OddsRatio) : ℚ :=
  odds.against_event / (odds.for_event + odds.against_event)

/-- Theorem: The probability of not pulling a blue marble is 6/11 given odds of 5:6 --/
theorem prob_not_blue_marble (odds : OddsRatio) 
  (h : odds = OddsRatio.mk 5 6) : 
  probability_of_not_occurring odds = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_marble_l2049_204988


namespace NUMINAMATH_CALUDE_forty_second_card_l2049_204933

def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

theorem forty_second_card :
  card_sequence 42 = 3 := by
  sorry

end NUMINAMATH_CALUDE_forty_second_card_l2049_204933


namespace NUMINAMATH_CALUDE_park_attendance_solution_l2049_204940

/-- Represents the number of people at Minewaska State Park --/
structure ParkAttendance where
  hikers : ℕ
  bikers : ℕ
  kayakers : ℕ

/-- The conditions of the park attendance problem --/
def parkProblem (p : ParkAttendance) : Prop :=
  p.hikers = p.bikers + 178 ∧
  p.kayakers * 2 = p.bikers ∧
  p.hikers + p.bikers + p.kayakers = 920

/-- The theorem stating the solution to the park attendance problem --/
theorem park_attendance_solution :
  ∃ p : ParkAttendance, parkProblem p ∧ p.hikers = 474 := by
  sorry

end NUMINAMATH_CALUDE_park_attendance_solution_l2049_204940


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2049_204978

theorem complex_equation_solution (m A B : ℝ) : 
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = Complex.mk A B →
  A + B = 0 →
  m = -2/3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2049_204978


namespace NUMINAMATH_CALUDE_largest_quantity_l2049_204966

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2049_204966


namespace NUMINAMATH_CALUDE_gumball_distribution_l2049_204959

theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat) : 
  joanna_initial = 40 →
  jacques_initial = 60 →
  let joanna_final := joanna_initial + 5 * joanna_initial
  let jacques_final := jacques_initial + 3 * jacques_initial
  let total := joanna_final + jacques_final
  let shared := total / 2
  shared = 240 :=
by sorry

end NUMINAMATH_CALUDE_gumball_distribution_l2049_204959


namespace NUMINAMATH_CALUDE_inequality_solution_l2049_204926

theorem inequality_solution (x : ℝ) : x / (x^2 + 3*x + 2) ≥ 0 ↔ x ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2049_204926


namespace NUMINAMATH_CALUDE_perimeter_of_square_III_is_four_l2049_204984

/-- Given three squares I, II, and III, prove that the perimeter of square III is 4 -/
theorem perimeter_of_square_III_is_four :
  ∀ (side_I side_II side_III : ℝ),
  side_I * 4 = 20 →
  side_II * 4 = 16 →
  side_III = side_I - side_II →
  side_III * 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_III_is_four_l2049_204984


namespace NUMINAMATH_CALUDE_milkshake_ice_cream_difference_l2049_204916

/-- Given the number of milkshakes and ice cream cones sold, prove the difference -/
theorem milkshake_ice_cream_difference (milkshakes ice_cream_cones : ℕ) 
  (h1 : milkshakes = 82) (h2 : ice_cream_cones = 67) : 
  milkshakes - ice_cream_cones = 15 := by
  sorry

end NUMINAMATH_CALUDE_milkshake_ice_cream_difference_l2049_204916


namespace NUMINAMATH_CALUDE_bills_toilet_paper_supply_l2049_204968

/-- Theorem: Bill's Toilet Paper Supply

Given:
- Bill uses the bathroom 3 times a day
- Bill uses 5 squares of toilet paper each time
- Each roll has 300 squares of toilet paper
- Bill's toilet paper supply will last for 20000 days

Prove that Bill has 1000 rolls of toilet paper.
-/
theorem bills_toilet_paper_supply 
  (bathroom_visits_per_day : ℕ) 
  (squares_per_visit : ℕ) 
  (squares_per_roll : ℕ) 
  (supply_duration_days : ℕ) 
  (h1 : bathroom_visits_per_day = 3)
  (h2 : squares_per_visit = 5)
  (h3 : squares_per_roll = 300)
  (h4 : supply_duration_days = 20000) :
  (bathroom_visits_per_day * squares_per_visit * supply_duration_days) / squares_per_roll = 1000 := by
  sorry

#check bills_toilet_paper_supply

end NUMINAMATH_CALUDE_bills_toilet_paper_supply_l2049_204968


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2049_204945

/-- For a parabola y² = 2px, if the distance from (4, 0) to the focus (p/2, 0) is 5, then p = 8 -/
theorem parabola_focus_distance (p : ℝ) : 
  (∀ y : ℝ, y^2 = 2*p*4) → -- point (4, y) is on the parabola
  ((4 - p/2)^2 + 0^2)^(1/2) = 5 → -- distance from (4, 0) to focus (p/2, 0) is 5
  p = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2049_204945


namespace NUMINAMATH_CALUDE_total_crayons_for_six_children_l2049_204972

/-- Calculates the total number of crayons given the number of children and crayons per child -/
def total_crayons (num_children : ℕ) (crayons_per_child : ℕ) : ℕ :=
  num_children * crayons_per_child

/-- Theorem: Given 6 children with 3 crayons each, the total number of crayons is 18 -/
theorem total_crayons_for_six_children :
  total_crayons 6 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_for_six_children_l2049_204972


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_implies_a_range_l2049_204925

theorem tangent_line_y_intercept_implies_a_range (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * x^2
  ∀ m : ℝ, m > 1 →
    let f' : ℝ → ℝ := λ x ↦ Real.exp x + 2 * a * x
    let tangent_slope : ℝ := f' m
    let tangent_y_intercept : ℝ := f m - tangent_slope * m
    tangent_y_intercept < 1 →
    a ∈ Set.Ici (-1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_implies_a_range_l2049_204925


namespace NUMINAMATH_CALUDE_sum_of_integers_l2049_204931

theorem sum_of_integers : (-25) + 34 + 156 + (-65) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2049_204931


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2049_204976

theorem least_subtraction_for_divisibility : ∃! n : ℕ, n ≤ 12 ∧ (427398 - n) % 13 = 0 ∧ ∀ m : ℕ, m < n → (427398 - m) % 13 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2049_204976


namespace NUMINAMATH_CALUDE_dividing_line_equation_l2049_204991

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the region S formed by the union of nine unit circles -/
def region_S : Set (ℝ × ℝ) :=
  sorry

/-- The line with slope 4 that divides region S into two equal areas -/
def dividing_line : ℝ → ℝ :=
  sorry

/-- Theorem stating that the dividing line has the equation 4x - y = 3 -/
theorem dividing_line_equation :
  ∀ x y, dividing_line y = x ↔ 4 * x - y = 3 :=
sorry

end NUMINAMATH_CALUDE_dividing_line_equation_l2049_204991


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l2049_204979

theorem lcd_of_fractions (a b c d e f : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) (hd : d = 6) (he : e = 8) (hf : f = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e f)))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l2049_204979


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l2049_204982

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : (y - 3)^2 = 8 * (x - 2)

/-- A circle tangent to the y-axis -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  tangent_to_y_axis : radius = center.x

theorem circle_passes_through_fixed_point (P : ParabolaPoint) (C : TangentCircle) 
  (h : C.center = P) : 
  (C.center.x - 4)^2 + (C.center.y - 3)^2 = C.radius^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l2049_204982


namespace NUMINAMATH_CALUDE_parabola_max_triangle_area_l2049_204911

/-- Given a parabola y = ax^2 + bx + c with a ≠ 0, intersecting the x-axis at A and B
    and the y-axis at C, with its vertex on y = -1, and ABC forming a right triangle,
    prove that the maximum area of triangle ABC is 1. -/
theorem parabola_max_triangle_area (a b c : ℝ) (ha : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let vertex_y := -1
  let A := {x : ℝ | f x = 0 ∧ x < 0}
  let B := {x : ℝ | f x = 0 ∧ x > 0}
  let C := (0, c)
  (∃ x, f x = vertex_y) →
  (∃ x₁ ∈ A, ∃ x₂ ∈ B, c^2 = (-x₁) * x₂) →
  (∀ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| → S ≤ 1) ∧ 
  (∃ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| ∧ S = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_triangle_area_l2049_204911


namespace NUMINAMATH_CALUDE_even_function_shift_l2049_204936

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval (a,b) if
    for all x, y in (a,b), x < y implies f(x) < f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- A function has a symmetry axis at x = k if f(k + x) = f(k - x) for all x -/
def HasSymmetryAxis (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (k + x) = f (k - x)

theorem even_function_shift (f : ℝ → ℝ) :
    IsEven f →
    MonoIncOn f 3 5 →
    HasSymmetryAxis (fun x ↦ f (x - 1)) 1 ∧
    MonoIncOn (fun x ↦ f (x - 1)) 4 6 := by
  sorry

end NUMINAMATH_CALUDE_even_function_shift_l2049_204936


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l2049_204949

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l2049_204949


namespace NUMINAMATH_CALUDE_hyperbola_proof_l2049_204912

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the new hyperbola
def new_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define a function to check if two hyperbolas have the same asymptotes
def same_asymptotes (h1 h2 : (ℝ → ℝ → Prop)) : Prop := sorry

-- Theorem statement
theorem hyperbola_proof :
  same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l2049_204912


namespace NUMINAMATH_CALUDE_lock_settings_count_l2049_204922

/-- The number of digits on each dial of the lock -/
def num_digits : ℕ := 8

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of different settings possible for the lock -/
def num_settings : ℕ := 1680

/-- Theorem stating that the number of different settings for the lock
    with the given conditions is equal to 1680 -/
theorem lock_settings_count :
  (num_digits.factorial) / ((num_digits - num_dials).factorial) = num_settings :=
sorry

end NUMINAMATH_CALUDE_lock_settings_count_l2049_204922


namespace NUMINAMATH_CALUDE_dealer_profit_selling_price_percentage_l2049_204977

theorem dealer_profit (list_price : ℝ) (purchase_price selling_price : ℝ) : 
  purchase_price = 3/4 * list_price →
  selling_price = 2 * purchase_price →
  selling_price = 3/2 * list_price :=
by sorry

theorem selling_price_percentage (list_price : ℝ) (selling_price : ℝ) :
  selling_price = 3/2 * list_price →
  (selling_price - list_price) / list_price = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_dealer_profit_selling_price_percentage_l2049_204977


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l2049_204904

theorem polygon_sides_from_exterior_angle (exterior_angle : ℝ) (n : ℕ) :
  exterior_angle = 36 →
  (360 : ℝ) / exterior_angle = n →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l2049_204904


namespace NUMINAMATH_CALUDE_website_earnings_l2049_204918

/-- John's website earnings problem -/
theorem website_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ)
  (h1 : visits_per_month = 30000)
  (h2 : days_per_month = 30)
  (h3 : earnings_per_visit = 1 / 100) :
  (visits_per_month : ℚ) * earnings_per_visit / days_per_month = 10 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_l2049_204918


namespace NUMINAMATH_CALUDE_abcd_sum_l2049_204965

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -31 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_l2049_204965


namespace NUMINAMATH_CALUDE_symmetry_implies_p_plus_r_zero_l2049_204934

/-- Represents a curve of the form y = (px + 2q) / (rx + 2s) -/
structure Curve where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- The property of y = 2x being an axis of symmetry for the curve -/
def is_axis_of_symmetry (c : Curve) : Prop :=
  ∀ x y : ℝ, y = (c.p * x + 2 * c.q) / (c.r * x + 2 * c.s) →
    y = (c.p * (y / 2) + 2 * c.q) / (c.r * (y / 2) + 2 * c.s)

/-- The main theorem stating that if y = 2x is an axis of symmetry, then p + r = 0 -/
theorem symmetry_implies_p_plus_r_zero (c : Curve) :
  is_axis_of_symmetry c → c.p + c.r = 0 := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_p_plus_r_zero_l2049_204934


namespace NUMINAMATH_CALUDE_first_player_wins_l2049_204928

/-- A proper divisor of n is a positive integer that divides n and is less than n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d > 0 ∧ d < n ∧ n % d = 0

/-- The game state, representing the number of tokens in the bowl. -/
structure GameState where
  tokens : ℕ

/-- A valid move in the game. -/
def ValidMove (s : GameState) (m : ℕ) : Prop :=
  ProperDivisor m s.tokens

/-- The game ends when the number of tokens exceeds 2024. -/
def GameOver (s : GameState) : Prop :=
  s.tokens > 2024

/-- The theorem stating that the first player has a winning strategy. -/
theorem first_player_wins :
  ∃ (strategy : GameState → ℕ),
    (∀ s : GameState, ¬GameOver s → ValidMove s (strategy s)) ∧
    (∀ (play : ℕ → GameState),
      play 0 = ⟨2⟩ →
      (∀ n : ℕ, ¬GameOver (play n) →
        play (n + 1) = ⟨(play n).tokens + strategy (play n)⟩ ∨
        (∃ m : ℕ, ValidMove (play n) m ∧
          play (n + 1) = ⟨(play n).tokens + m⟩)) →
      ∃ k : ℕ, GameOver (play k) ∧ k % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2049_204928


namespace NUMINAMATH_CALUDE_somu_age_problem_l2049_204969

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 6 = (father_age - 6) / 5 →
  somu_age = 12 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l2049_204969


namespace NUMINAMATH_CALUDE_ellis_card_difference_l2049_204973

/-- Represents the number of cards each player has -/
structure CardDistribution where
  ellis : ℕ
  orion : ℕ

/-- Calculates the card distribution based on the total cards and ratio -/
def distribute_cards (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ) : CardDistribution :=
  let part_value := total / (ellis_ratio + orion_ratio)
  { ellis := ellis_ratio * part_value,
    orion := orion_ratio * part_value }

/-- Theorem stating that Ellis has 332 more cards than Orion -/
theorem ellis_card_difference (total : ℕ) (ellis_ratio : ℕ) (orion_ratio : ℕ)
  (h_total : total = 2500)
  (h_ellis_ratio : ellis_ratio = 17)
  (h_orion_ratio : orion_ratio = 13) :
  let distribution := distribute_cards total ellis_ratio orion_ratio
  distribution.ellis - distribution.orion = 332 := by
  sorry


end NUMINAMATH_CALUDE_ellis_card_difference_l2049_204973


namespace NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l2049_204900

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l2049_204900


namespace NUMINAMATH_CALUDE_tim_balloon_count_l2049_204930

/-- Given that Dan has 58.0 violet balloons and 10.0 times more violet balloons than Tim,
    prove that Tim has 5.8 violet balloons. -/
theorem tim_balloon_count : 
  ∀ (dan_balloons tim_balloons : ℝ),
    dan_balloons = 58.0 →
    dan_balloons = 10.0 * tim_balloons →
    tim_balloons = 5.8 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l2049_204930


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2049_204927

theorem triangle_angle_problem (A B C : ℕ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A < B →
  B < C →
  4 * C = 7 * A →
  B = 59 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2049_204927


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2049_204907

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 660 →
  crossing_time = 54.995600351971845 →
  ∃ (speed : ℝ), abs (speed - 54.0036) < 0.0001 ∧ 
  speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2049_204907


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2049_204929

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a₁ + a₂ + a₃ + a₄ = -80) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2049_204929


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_three_l2049_204914

/-- Given a real number m, the system of inequalities {x - m > 2, x - 2m < -1} has no solution if and only if m ≤ 3. -/
theorem no_solution_iff_m_leq_three (m : ℝ) : 
  (∀ x : ℝ, ¬(x - m > 2 ∧ x - 2*m < -1)) ↔ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_three_l2049_204914


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_800_l2049_204992

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

theorem least_sum_of_exponents_for_800 :
  ∀ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 800 powers →
    powers.length ≥ 3 →
    powers.sum ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_800_l2049_204992


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2049_204954

theorem tan_alpha_plus_pi_fourth (α : Real) (m : Real) (h : m ≠ 0) :
  let P : Real × Real := (m, -2*m)
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2049_204954


namespace NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l2049_204995

theorem multiplicative_inverse_mod_million : ∃ N : ℕ, 
  (N > 0) ∧ 
  (N < 1000000) ∧ 
  ((123456 * 769230 * N) % 1000000 = 1) ∧ 
  (N = 1053) := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l2049_204995


namespace NUMINAMATH_CALUDE_cone_volume_l2049_204964

/-- Given a cone with slant height 5 and lateral surface area 20π, prove its volume is 16π -/
theorem cone_volume (s : ℝ) (l : ℝ) (v : ℝ) : 
  s = 5 → l = 20 * Real.pi → v = (16 : ℝ) * Real.pi → 
  (s^2 * Real.pi / l = s / 4) ∧ 
  (v = (1/3) * (l/s)^2 * (s^2 - (l/(Real.pi * s))^2)) := by
  sorry

#check cone_volume

end NUMINAMATH_CALUDE_cone_volume_l2049_204964


namespace NUMINAMATH_CALUDE_smallest_among_four_rationals_l2049_204939

theorem smallest_among_four_rationals :
  let a : ℚ := -2/3
  let b : ℚ := -1
  let c : ℚ := 0
  let d : ℚ := 1
  b < a ∧ b < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_smallest_among_four_rationals_l2049_204939


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2049_204947

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2049_204947


namespace NUMINAMATH_CALUDE_apples_per_basket_l2049_204971

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) (h1 : total_baskets = 37) (h2 : total_apples = 629) :
  total_apples / total_baskets = 17 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_basket_l2049_204971


namespace NUMINAMATH_CALUDE_remainder_problem_l2049_204993

theorem remainder_problem (k : ℕ) :
  k > 0 ∧ k < 100 ∧
  k % 5 = 2 ∧
  k % 6 = 3 ∧
  k % 8 = 7 →
  k % 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2049_204993


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2049_204923

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, |a - b^2| + |b - a^2| ≤ 1 → (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧
  (∃ a b : ℝ, (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2 ∧ |a - b^2| + |b - a^2| > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2049_204923


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l2049_204951

theorem greatest_integer_radius (A : ℝ) (h1 : 50 * Real.pi < A) (h2 : A < 75 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi = A ∧ r ≤ 8 ∧ ∀ (s : ℕ), s * s * Real.pi = A → s ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l2049_204951


namespace NUMINAMATH_CALUDE_remainder_twelve_pow_2012_mod_5_l2049_204901

theorem remainder_twelve_pow_2012_mod_5 : 12^2012 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_twelve_pow_2012_mod_5_l2049_204901


namespace NUMINAMATH_CALUDE_wheat_bags_theorem_l2049_204999

/-- Represents the deviation of each bag from the standard weight -/
def deviations : List Int := [-6, -3, -1, 7, 3, 4, -3, -2, -2, 1]

/-- The number of bags -/
def num_bags : Nat := 10

/-- The standard weight per bag in kg -/
def standard_weight : Int := 150

/-- The sum of all deviations -/
def total_deviation : Int := deviations.sum

/-- The average weight per bag -/
noncomputable def average_weight : ℚ := 
  (num_bags * standard_weight + total_deviation) / num_bags

theorem wheat_bags_theorem : 
  total_deviation = -2 ∧ average_weight = 149.8 := by sorry

end NUMINAMATH_CALUDE_wheat_bags_theorem_l2049_204999


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l2049_204948

-- Define the constants and variables
variable (a b c k : ℝ)
variable (y₁ y₂ y₃ : ℝ)

-- State the theorem
theorem inverse_proportion_order (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) 
  (h4 : k > 0)
  (h5 : y₁ = k / (a - b))
  (h6 : y₂ = k / (a - c))
  (h7 : y₃ = k / (c - a)) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l2049_204948


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l2049_204952

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting at 2, ending at 2014, 
    with a common difference of 3, has 671 terms. -/
theorem arithmetic_sequence_2_to_2014 :
  arithmeticSequenceLength 2 2014 3 = 671 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l2049_204952


namespace NUMINAMATH_CALUDE_total_viewing_time_l2049_204943

/-- The viewing times for the original animal types -/
def original_times : List Nat := [4, 6, 7, 5, 9]

/-- The viewing times for the new animal types -/
def new_times : List Nat := [3, 7, 8, 10]

/-- The total number of animal types -/
def total_types : Nat := original_times.length + new_times.length

theorem total_viewing_time :
  (List.sum original_times) + (List.sum new_times) = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_l2049_204943


namespace NUMINAMATH_CALUDE_prob_at_least_one_odd_prob_outside_or_on_circle_l2049_204994

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the sample space for two die rolls
def TwoRolls : Type := Die × Die

-- Define the probability measure
def P : Set TwoRolls → ℚ := sorry

-- Define the event of at least one odd number
def AtLeastOneOdd : Set TwoRolls := sorry

-- Define the event of the point lying outside or on the circle
def OutsideOrOnCircle : Set TwoRolls := sorry

-- Theorem for the first probability
theorem prob_at_least_one_odd : P AtLeastOneOdd = 3/4 := sorry

-- Theorem for the second probability
theorem prob_outside_or_on_circle : P OutsideOrOnCircle = 7/9 := sorry

end NUMINAMATH_CALUDE_prob_at_least_one_odd_prob_outside_or_on_circle_l2049_204994


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_m_eq_6_l2049_204960

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

/-- The function f(x) = x^3 - mx^2 + 2m^2 - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - m*x^2 + 2*m^2 - 5

theorem monotone_decreasing_implies_m_eq_6 :
  ∀ m : ℝ, MonotonicallyDecreasing (f m) (-9) 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_m_eq_6_l2049_204960


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2049_204919

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2049_204919


namespace NUMINAMATH_CALUDE_max_value_of_S_l2049_204990

theorem max_value_of_S (a b : ℝ) :
  3 * a^2 + 5 * abs b = 7 →
  let S := 2 * a^2 - 3 * abs b
  ∀ x y : ℝ, 3 * x^2 + 5 * abs y = 7 → 2 * x^2 - 3 * abs y ≤ |14| / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_S_l2049_204990


namespace NUMINAMATH_CALUDE_complex_symmetric_modulus_l2049_204953

theorem complex_symmetric_modulus (z₁ z₂ : ℂ) :
  (z₁.re = z₂.im ∧ z₁.im = z₂.re) →  -- symmetry about y=x
  z₁ * z₂ = Complex.I * 9 →          -- z₁z₂ = 9i
  Complex.abs z₁ = 3 := by           -- |z₁| = 3
sorry

end NUMINAMATH_CALUDE_complex_symmetric_modulus_l2049_204953


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l2049_204987

/-- The number of possible fruit baskets given the constraints -/
def fruitBaskets (totalApples totalOranges : ℕ) (minApples minOranges : ℕ) : ℕ :=
  (totalApples - minApples + 1) * (totalOranges - minOranges + 1)

/-- Theorem stating the number of possible fruit baskets under given conditions -/
theorem fruit_basket_combinations :
  fruitBaskets 6 12 1 2 = 66 := by
  sorry

#eval fruitBaskets 6 12 1 2

end NUMINAMATH_CALUDE_fruit_basket_combinations_l2049_204987


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l2049_204970

/-- The original price of merchandise satisfying given conditions -/
def original_price : ℝ := 175

/-- The loss when sold at 60% of the original price -/
def loss_at_60_percent : ℝ := 20

/-- The gain when sold at 80% of the original price -/
def gain_at_80_percent : ℝ := 15

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  (0.6 * original_price + loss_at_60_percent = 0.8 * original_price - gain_at_80_percent) := by
  sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l2049_204970


namespace NUMINAMATH_CALUDE_school_population_l2049_204913

theorem school_population (b g t : ℕ) : 
  b = 6 * g → g = 5 * t → b + g + t = 36 * t :=
by sorry

end NUMINAMATH_CALUDE_school_population_l2049_204913


namespace NUMINAMATH_CALUDE_geometry_propositions_l2049_204996

-- Define the type for planes
variable (Plane : Type)

-- Define the type for lines
variable (Line : Type)

-- Define the relation for two planes being distinct
variable (distinct : Plane → Plane → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (within : Line → Plane → Prop)

-- Define the relation for two lines being parallel
variable (parallel_lines : Line → Line → Prop)

-- Define the relation for two planes being parallel
variable (parallel_planes : Plane → Plane → Prop)

-- Define the relation for a line being perpendicular to a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation for a line being perpendicular to another line
variable (perp_lines : Line → Line → Prop)

-- Define the relation for two planes intersecting along a line
variable (intersect_along : Plane → Plane → Line → Prop)

-- Define the relation for two planes being perpendicular
variable (perp_planes : Plane → Plane → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

theorem geometry_propositions 
  (α β : Plane) 
  (h_distinct : distinct α β) :
  (∀ (l1 l2 m1 m2 : Line), 
    intersect l1 l2 ∧ within l1 α ∧ within l2 α ∧ 
    within m1 β ∧ within m2 β ∧ 
    parallel_lines l1 m1 ∧ parallel_lines l2 m2 → 
    parallel_planes α β) ∧ 
  (∃ (l : Line) (m1 m2 : Line), 
    perp_line_plane l α ∧ 
    within m1 α ∧ within m2 α ∧ intersect m1 m2 ∧ 
    perp_lines l m1 ∧ perp_lines l m2 ∧ 
    ¬(∀ (n : Line), within n α ∧ perp_lines l n → perp_line_plane l α)) ∧
  (∃ (l m : Line), 
    intersect_along α β l ∧ within m α ∧ perp_lines m l ∧ ¬perp_planes α β) ∧
  (∀ (l m : Line), 
    outside l α ∧ within m α ∧ parallel_lines l m → 
    ∀ (n : Line), within n α → ¬intersect l n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2049_204996


namespace NUMINAMATH_CALUDE_largest_number_l2049_204903

/-- Converts a number from base b to decimal (base 10) --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: 11 in base 3 is greater than 3 in base 10, 11 in base 2, and 3 in base 8 --/
theorem largest_number :
  (to_decimal 11 3 > to_decimal 3 10) ∧
  (to_decimal 11 3 > to_decimal 11 2) ∧
  (to_decimal 11 3 > to_decimal 3 8) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2049_204903


namespace NUMINAMATH_CALUDE_intersection_M_N_l2049_204908

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2049_204908


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2049_204946

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2049_204946


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_surveys_l2049_204963

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasDistinctGroups : Bool
  hasSmallDifferences : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimalSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasDistinctGroups then SamplingMethod.Stratified
  else if s.hasSmallDifferences && s.sampleSize < 10 then SamplingMethod.SimpleRandom
  else SamplingMethod.Systematic

/-- The main theorem stating the optimal sampling methods for the two surveys -/
theorem optimal_sampling_methods_for_surveys :
  let survey1 : Survey := {
    totalPopulation := 500,
    sampleSize := 100,
    hasDistinctGroups := true,
    hasSmallDifferences := false
  }
  let survey2 : Survey := {
    totalPopulation := 15,
    sampleSize := 3,
    hasDistinctGroups := false,
    hasSmallDifferences := true
  }
  (optimalSamplingMethod survey1 = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod survey2 = SamplingMethod.SimpleRandom) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_for_surveys_l2049_204963


namespace NUMINAMATH_CALUDE_expired_milk_probability_l2049_204981

theorem expired_milk_probability (total_bags : ℕ) (expired_bags : ℕ) 
  (h1 : total_bags = 25) (h2 : expired_bags = 4) :
  (expired_bags : ℚ) / total_bags = 4 / 25 :=
by sorry

end NUMINAMATH_CALUDE_expired_milk_probability_l2049_204981


namespace NUMINAMATH_CALUDE_reassignment_count_l2049_204906

/-- The number of people and jobs -/
def n : ℕ := 5

/-- The number of ways to reassign n jobs to n people such that at least 2 people change jobs -/
def reassignments (n : ℕ) : ℕ := n.factorial - 1

/-- Theorem: The number of ways to reassign 5 jobs to 5 people, 
    such that at least 2 people change jobs from their initial assignment, is 5! - 1 -/
theorem reassignment_count : reassignments n = 119 := by
  sorry

end NUMINAMATH_CALUDE_reassignment_count_l2049_204906


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2049_204915

/-- Calculates the rate of mixed oil per litre given volumes and prices of different oils. -/
theorem mixed_oil_rate (v1 v2 v3 v4 : ℚ) (p1 p2 p3 p4 : ℚ) :
  v1 = 10 ∧ v2 = 5 ∧ v3 = 3 ∧ v4 = 2 ∧
  p1 = 50 ∧ p2 = 66 ∧ p3 = 75 ∧ p4 = 85 →
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4) = 61.25 := by
  sorry

#eval (10 * 50 + 5 * 66 + 3 * 75 + 2 * 85) / (10 + 5 + 3 + 2)

end NUMINAMATH_CALUDE_mixed_oil_rate_l2049_204915


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_area_difference_l2049_204938

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the vertices of the ellipse
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define a line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the slope ratio condition
def slope_ratio (k₁ k₂ : ℝ) : Prop := k₁ / k₂ = 1 / 9

-- Define the intersection points
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂

-- Define the triangles' areas
def area_diff (S₁ S₂ : ℝ) : Prop := S₁ - S₂ ≤ 15

-- Theorem 1: The line passes through (4, 0)
theorem line_passes_through_fixed_point (k m : ℝ) :
  intersection_points k m →
  (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂) →
  line k m 4 0 :=
sorry

-- Theorem 2: Maximum value of S₁ - S₂
theorem max_area_difference :
  ∀ S₁ S₂ : ℝ,
  (∃ k m : ℝ, intersection_points k m ∧ 
   (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂)) →
  area_diff S₁ S₂ ∧ 
  (∀ S₁' S₂' : ℝ, area_diff S₁' S₂' → S₁ - S₂ ≥ S₁' - S₂') :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_area_difference_l2049_204938


namespace NUMINAMATH_CALUDE_paige_remaining_stickers_l2049_204942

/-- The number of space stickers Paige has -/
def space_stickers : ℕ := 100

/-- The number of cat stickers Paige has -/
def cat_stickers : ℕ := 50

/-- The number of friends Paige is sharing with -/
def num_friends : ℕ := 3

/-- The function to calculate the number of remaining stickers -/
def remaining_stickers (space : ℕ) (cat : ℕ) (friends : ℕ) : ℕ :=
  (space % friends) + (cat % friends)

/-- Theorem stating that Paige will have 3 stickers left -/
theorem paige_remaining_stickers :
  remaining_stickers space_stickers cat_stickers num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_paige_remaining_stickers_l2049_204942


namespace NUMINAMATH_CALUDE_partition_large_rectangle_l2049_204986

/-- Definition of a "good" rectangle -/
inductive GoodRectangle
  | square : GoodRectangle
  | rectangle : GoodRectangle

/-- Predicate to check if a rectangle can be partitioned into good rectangles -/
def can_partition (a b : ℕ) : Prop :=
  ∃ (num_squares num_rectangles : ℕ),
    2 * 2 * num_squares + 1 * 11 * num_rectangles = a * b

/-- Theorem: Any rectangle with integer sides greater than 100 can be partitioned into good rectangles -/
theorem partition_large_rectangle (a b : ℕ) (ha : a > 100) (hb : b > 100) :
  can_partition a b := by
  sorry


end NUMINAMATH_CALUDE_partition_large_rectangle_l2049_204986


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l2049_204955

theorem equilateral_triangle_division : 
  ∃ (k m : ℕ), 2007 = 9 + 3 * k ∧ 2008 = 4 + 3 * m :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l2049_204955


namespace NUMINAMATH_CALUDE_darren_tshirts_l2049_204937

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end NUMINAMATH_CALUDE_darren_tshirts_l2049_204937


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2049_204961

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (a ≠ 5 ∧ b ≠ -5) ↔ (a + b ≠ 0) → False :=
by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2049_204961


namespace NUMINAMATH_CALUDE_margarets_mean_score_l2049_204997

def scores : List ℝ := [88, 90, 94, 95, 96, 99]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores : List ℝ) (margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 2 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 97 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l2049_204997


namespace NUMINAMATH_CALUDE_two_lines_theorem_l2049_204909

/-- Two lines in the plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ
  l₂ : ℝ → ℝ → ℝ
  a : ℝ
  b : ℝ
  h₁ : ∀ x y, l₁ x y = a * x - b * y + 4
  h₂ : ∀ x y, l₂ x y = (a - 1) * x + y + 2

/-- Scenario 1: l₁ passes through (-3,-1) and is perpendicular to l₂ -/
def scenario1 (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0 ∧ 
  (lines.a / lines.b) * (1 - lines.a) = -1

/-- Scenario 2: l₁ is parallel to l₂ and has y-intercept -3 -/
def scenario2 (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a ∧
  4 / lines.b = -3

theorem two_lines_theorem (lines : TwoLines) :
  (scenario1 lines → lines.a = 2 ∧ lines.b = 2) ∧
  (scenario2 lines → lines.a = 4 ∧ lines.b = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_two_lines_theorem_l2049_204909


namespace NUMINAMATH_CALUDE_ajay_work_days_l2049_204957

/-- The number of days it takes Vijay to complete the work alone -/
def vijay_days : ℝ := 24

/-- The number of days it takes Ajay and Vijay to complete the work together -/
def together_days : ℝ := 6

/-- The number of days it takes Ajay to complete the work alone -/
noncomputable def ajay_days : ℝ := 
  (vijay_days * together_days) / (vijay_days - together_days)

theorem ajay_work_days : ajay_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_ajay_work_days_l2049_204957


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2049_204980

theorem sphere_surface_area (r : ℝ) (R : ℝ) :
  r > 0 → R > 0 →
  r^2 + 1^2 = R^2 →
  π * r^2 = π →
  4 * π * R^2 = 8 * π := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2049_204980


namespace NUMINAMATH_CALUDE_max_y_value_l2049_204989

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : 
  ∃ (max_y : ℤ), (∀ (z : ℤ), ∃ (w : ℤ), w * z + 3 * w + 2 * z = -6 → z ≤ max_y) ∧ max_y = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2049_204989


namespace NUMINAMATH_CALUDE_pineapples_cost_theorem_l2049_204967

/-- The cost relationship between bananas, apples, and pineapples -/
structure FruitCosts where
  banana_to_apple : ℚ    -- 5 bananas = 3 apples
  apple_to_pineapple : ℚ  -- 9 apples = 6 pineapples

/-- The number of pineapples that cost the same as 30 bananas -/
def pineapples_equal_to_30_bananas (costs : FruitCosts) : ℚ :=
  30 * (costs.apple_to_pineapple / 9) * (3 / 5)

theorem pineapples_cost_theorem (costs : FruitCosts) 
  (h1 : costs.banana_to_apple = 3 / 5)
  (h2 : costs.apple_to_pineapple = 6 / 9) :
  pineapples_equal_to_30_bananas costs = 12 := by
  sorry

end NUMINAMATH_CALUDE_pineapples_cost_theorem_l2049_204967
