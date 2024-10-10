import Mathlib

namespace q_polynomial_l2121_212171

theorem q_polynomial (x : ℝ) (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2*x^6 + 4*x^4 - 5*x^3 + 2*x) = (3*x^4 + x^3 - 11*x^2 + 6*x + 3)) :
  q x = -2*x^6 - x^4 + 6*x^3 - 11*x^2 + 4*x + 3 := by
sorry

end q_polynomial_l2121_212171


namespace sqrt_five_approximation_l2121_212126

theorem sqrt_five_approximation :
  (2^2 < 5 ∧ 5 < 3^2) →
  (2.2^2 < 5 ∧ 5 < 2.3^2) →
  (2.23^2 < 5 ∧ 5 < 2.24^2) →
  (2.236^2 < 5 ∧ 5 < 2.237^2) →
  ∃ (x : ℝ), x^2 = 5 ∧ |x - 2.24| < 0.005 :=
by sorry

end sqrt_five_approximation_l2121_212126


namespace circle_center_range_l2121_212144

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the circle C
def circle_C (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_center_range :
  ∀ center_x center_y : ℝ,
  line_l center_x center_y →
  (∃ x : ℝ, circle_C center_x center_y x 0 ∧ circle_C center_x center_y (-x) 0 ∧ x^2 = 3/4) →
  (∃ mx my : ℝ, circle_C center_x center_y mx my ∧
    (mx - point_A.1)^2 + (my - point_A.2)^2 = 4 * ((mx - center_x)^2 + (my - center_y)^2)) →
  0 ≤ center_x ∧ center_x ≤ 12/5 :=
by sorry

end circle_center_range_l2121_212144


namespace fraction_sum_difference_l2121_212165

theorem fraction_sum_difference : (3 / 50 + 2 / 25 - 5 / 1000 : ℚ) = 0.135 := by
  sorry

end fraction_sum_difference_l2121_212165


namespace turtle_problem_l2121_212176

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 25) :
  let additional_turtles := 5 * initial_turtles - 4
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles - (total_turtles / 3)
  remaining_turtles = 98 := by
sorry

end turtle_problem_l2121_212176


namespace arithmetic_mean_of_fractions_l2121_212187

theorem arithmetic_mean_of_fractions : 
  (5 : ℚ) / 6 = ((9 : ℚ) / 12 + (11 : ℚ) / 12) / 2 := by
sorry

end arithmetic_mean_of_fractions_l2121_212187


namespace total_crayons_l2121_212148

theorem total_crayons (boxes : ℕ) (crayons_per_box : ℕ) (h1 : boxes = 8) (h2 : crayons_per_box = 7) :
  boxes * crayons_per_box = 56 := by
  sorry

end total_crayons_l2121_212148


namespace no_distinct_unit_fraction_sum_l2121_212145

theorem no_distinct_unit_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ¬∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ (p - 1 : ℚ) / p = 1 / a + 1 / b :=
by sorry

end no_distinct_unit_fraction_sum_l2121_212145


namespace max_sum_of_squares_l2121_212125

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 170 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
by sorry

end max_sum_of_squares_l2121_212125


namespace mario_age_l2121_212136

/-- Mario and Maria's ages problem -/
theorem mario_age (mario_age maria_age : ℕ) : 
  mario_age + maria_age = 7 → 
  mario_age = maria_age + 1 → 
  mario_age = 4 := by
sorry

end mario_age_l2121_212136


namespace gcd_lcm_product_75_90_l2121_212139

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end gcd_lcm_product_75_90_l2121_212139


namespace book_ratio_problem_l2121_212194

theorem book_ratio_problem (lit sci : ℕ) (h : lit * 5 = sci * 8) : 
  (lit - sci : ℚ) / sci = 3 / 5 ∧ (lit - sci : ℚ) / lit = 3 / 8 := by
  sorry

end book_ratio_problem_l2121_212194


namespace correct_arrangement_count_l2121_212156

/-- The number of ways to arrange 2 boys and 3 girls in a row with specific conditions -/
def arrangementCount : ℕ :=
  let totalPeople : ℕ := 5
  let boys : ℕ := 2
  let girls : ℕ := 3
  let boyA : ℕ := 1
  48

/-- Theorem stating that the number of arrangements satisfying the given conditions is 48 -/
theorem correct_arrangement_count :
  arrangementCount = 48 :=
by sorry

end correct_arrangement_count_l2121_212156


namespace josh_book_purchase_l2121_212122

/-- The number of books Josh bought -/
def num_books : ℕ := sorry

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def film_cost : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 4

/-- The cost of each CD in dollars -/
def cd_cost : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

theorem josh_book_purchase : 
  num_books * book_cost + num_films * film_cost + num_cds * cd_cost = total_spent ∧ 
  num_books = 4 := by sorry

end josh_book_purchase_l2121_212122


namespace parabola_sum_l2121_212182

/-- A parabola with equation y = ax^2 + bx + c, vertex (-3, 4), vertical axis of symmetry, 
    and passing through (4, -2) has a + b + c = 100/49 -/
theorem parabola_sum (a b c : ℚ) : 
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ 
    (x = -3 ∧ y = 4) ∨ 
    (x = 4 ∧ y = -2) ∨ 
    (∃ k : ℚ, y - 4 = k * (x + 3)^2)) →
  a + b + c = 100/49 := by
sorry

end parabola_sum_l2121_212182


namespace square_area_increase_l2121_212162

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.4 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.96 := by
sorry

end square_area_increase_l2121_212162


namespace wedge_volume_approximation_l2121_212129

/-- The volume of a wedge cut from a cylinder --/
theorem wedge_volume_approximation (r h : ℝ) (h_r : r = 6) (h_h : h = 6) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := cylinder_volume / 2
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |wedge_volume - 339.12| < ε :=
by sorry

end wedge_volume_approximation_l2121_212129


namespace exactly_one_divisible_by_five_l2121_212106

theorem exactly_one_divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
  (5 ∣ (a^2 - 1)) ≠ (5 ∣ (a^2 + 1)) :=
by sorry

end exactly_one_divisible_by_five_l2121_212106


namespace horizon_fantasy_meetup_handshakes_l2121_212101

/-- Calculates the number of handshakes in a group where everyone shakes hands with everyone else once -/
def handshakesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of handshakes between two groups where everyone in one group shakes hands with everyone in the other group once -/
def handshakesBetweenGroups (n m : ℕ) : ℕ := n * m

theorem horizon_fantasy_meetup_handshakes :
  let gremlins : ℕ := 25
  let imps : ℕ := 20
  let sprites : ℕ := 10
  let gremlinHandshakes := handshakesInGroup gremlins
  let gremlinImpHandshakes := handshakesBetweenGroups gremlins imps
  let spriteHandshakes := handshakesInGroup sprites
  let gremlinSpriteHandshakes := handshakesBetweenGroups gremlins sprites
  gremlinHandshakes + gremlinImpHandshakes + spriteHandshakes + gremlinSpriteHandshakes = 1095 := by
  sorry

#eval handshakesInGroup 25 + handshakesBetweenGroups 25 20 + handshakesInGroup 10 + handshakesBetweenGroups 25 10

end horizon_fantasy_meetup_handshakes_l2121_212101


namespace three_cubic_yards_to_cubic_feet_l2121_212168

-- Define the conversion factor
def yard_to_foot : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 3

-- Theorem to prove
theorem three_cubic_yards_to_cubic_feet :
  cubic_yards * (yard_to_foot ^ 3) = 81 := by
  sorry

end three_cubic_yards_to_cubic_feet_l2121_212168


namespace max_tickets_for_hockey_l2121_212179

def max_tickets (ticket_price : ℕ) (budget : ℕ) : ℕ :=
  (budget / ticket_price : ℕ)

theorem max_tickets_for_hockey (ticket_price : ℕ) (budget : ℕ) 
  (h1 : ticket_price = 20) (h2 : budget = 150) : 
  max_tickets ticket_price budget = 7 := by
  sorry

end max_tickets_for_hockey_l2121_212179


namespace perpendicular_vectors_imply_m_value_l2121_212146

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 4]
def b (m : ℝ) : Fin 2 → ℝ := ![-1, 2*m]
def c (m : ℝ) : Fin 2 → ℝ := ![m, -4]

/-- The sum of vectors a and b -/
def a_plus_b (m : ℝ) : Fin 2 → ℝ := ![a 0 + b m 0, a 1 + b m 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

/-- Theorem stating that if c is perpendicular to (a + b), then m = -8/3 -/
theorem perpendicular_vectors_imply_m_value :
  ∀ m : ℝ, dot_product (c m) (a_plus_b m) = 0 → m = -8/3 := by
sorry

end perpendicular_vectors_imply_m_value_l2121_212146


namespace difference_of_roots_quadratic_l2121_212172

theorem difference_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁ - r₂ = 1 :=
by
  sorry

end difference_of_roots_quadratic_l2121_212172


namespace f_composition_fixed_points_l2121_212197

def f (x : ℝ) := x^3 - 3*x^2

theorem f_composition_fixed_points :
  ∃ (x : ℝ), f (f x) = f x ∧ (x = 0 ∨ x = 3) :=
sorry

end f_composition_fixed_points_l2121_212197


namespace eleventh_term_ratio_l2121_212104

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.firstTerm + (n - 1) * seq.commonDiff) / 2

/-- nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem eleventh_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, sumOfTerms seq1 n / sumOfTerms seq2 n = (7 * n + 1) / (4 * n + 27)) :
  nthTerm seq1 11 / nthTerm seq2 11 = 4 / 3 := by
  sorry

end eleventh_term_ratio_l2121_212104


namespace number_of_factors_46464_l2121_212116

theorem number_of_factors_46464 : Nat.card (Nat.divisors 46464) = 36 := by
  sorry

end number_of_factors_46464_l2121_212116


namespace lcm_gcf_problem_l2121_212198

theorem lcm_gcf_problem (n m : ℕ+) : 
  Nat.lcm n m = 54 → Nat.gcd n m = 8 → n = 36 → m = 12 := by
  sorry

end lcm_gcf_problem_l2121_212198


namespace sum_of_medians_is_64_l2121_212152

def median (scores : List ℕ) : ℚ :=
  sorry

theorem sum_of_medians_is_64 (scores_A scores_B : List ℕ) : 
  median scores_A + median scores_B = 64 :=
sorry

end sum_of_medians_is_64_l2121_212152


namespace car_speed_graph_comparison_l2121_212149

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The graph representation of a car's travel -/
structure GraphLine where
  height : ℝ
  length : ℝ

/-- Theorem: If Car N travels at three times the speed of Car M for the same distance,
    then on a speed-time graph, Car N's line will be thrice as high and one-third the length of Car M's line. -/
theorem car_speed_graph_comparison (m n : Car) (gm gn : GraphLine) :
  n.speed = 3 * m.speed →
  m.distance = n.distance →
  gm.height = m.speed →
  gm.length = m.time →
  gn.height = n.speed →
  gn.length = n.time →
  gn.height = 3 * gm.height ∧ gn.length = gm.length / 3 := by
  sorry


end car_speed_graph_comparison_l2121_212149


namespace calories_burned_jogging_l2121_212117

/-- Calculate calories burned by jogging -/
theorem calories_burned_jogging (laps_per_night : ℕ) (feet_per_lap : ℕ) (feet_per_calorie : ℕ) (days : ℕ) : 
  laps_per_night = 5 →
  feet_per_lap = 100 →
  feet_per_calorie = 25 →
  days = 5 →
  (laps_per_night * feet_per_lap * days) / feet_per_calorie = 100 := by
  sorry

end calories_burned_jogging_l2121_212117


namespace shifted_line_equation_l2121_212115

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem shifted_line_equation (x y : ℝ) :
  let original_line := Line.mk (-2) 0
  let shifted_line := shift_line original_line 3
  y = shifted_line.slope * x + shifted_line.intercept ↔ y = -2 * x + 3 := by
  sorry

end shifted_line_equation_l2121_212115


namespace sweep_probability_l2121_212175

/-- Represents a clock with four equally spaced points -/
structure Clock :=
  (points : Fin 4 → ℕ)
  (equally_spaced : ∀ i : Fin 4, points i = i.val * 3)

/-- Represents a 20-minute period on the clock -/
def Period : ℕ := 20

/-- Calculates the number of favorable intervals in a 60-minute period -/
def favorable_intervals (c : Clock) (p : ℕ) : ℕ :=
  4 * 5  -- 4 intervals of 5 minutes each

/-- The probability of sweeping exactly two points in the given period -/
def probability (c : Clock) (p : ℕ) : ℚ :=
  (favorable_intervals c p : ℚ) / 60

/-- The main theorem stating the probability is 1/3 -/
theorem sweep_probability (c : Clock) :
  probability c Period = 1 / 3 := by
  sorry

end sweep_probability_l2121_212175


namespace xy_range_l2121_212192

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 3*y + 2/x + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end xy_range_l2121_212192


namespace distance_on_line_l2121_212159

/-- The distance between two points (p, q) and (r, s) on the line y = 2x + 3, where s = 2r + 6 -/
theorem distance_on_line (p r : ℝ) : 
  let q := 2 * p + 3
  let s := 2 * r + 6
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) := by
  sorry

end distance_on_line_l2121_212159


namespace discount_profit_calculation_l2121_212112

/-- Calculates the profit percentage with a given discount, based on the no-discount profit percentage. -/
def profit_with_discount (no_discount_profit : ℝ) (discount : ℝ) : ℝ :=
  ((1 + no_discount_profit) * (1 - discount) - 1) * 100

/-- Theorem stating that with a 5% discount and a 150% no-discount profit, the profit is 137.5% -/
theorem discount_profit_calculation :
  profit_with_discount 1.5 0.05 = 137.5 := by
  sorry

#eval profit_with_discount 1.5 0.05

end discount_profit_calculation_l2121_212112


namespace polynomial_division_theorem_l2121_212110

theorem polynomial_division_theorem (x : ℝ) :
  (x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32) * (x + 2) + 76 = x^6 + 12 := by
  sorry

end polynomial_division_theorem_l2121_212110


namespace cube_root_unity_sum_l2121_212178

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^4 + (2 + ω - ω^2)^4 = 512 := by
  sorry

end cube_root_unity_sum_l2121_212178


namespace weight_difference_l2121_212127

/-- Given Heather's and Emily's weights, prove the weight difference between them. -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h_heather : heather_weight = 87)
  (h_emily : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end weight_difference_l2121_212127


namespace library_items_count_l2121_212147

theorem library_items_count (notebooks : ℕ) (pens : ℕ) : 
  notebooks = 30 →
  pens = notebooks + 50 →
  notebooks + pens = 110 := by
  sorry

end library_items_count_l2121_212147


namespace nancy_coffee_spend_l2121_212102

/-- The amount Nancy spends on coffee over a given number of days -/
def coffee_expenditure (days : ℕ) (espresso_price iced_price : ℚ) : ℚ :=
  days * (espresso_price + iced_price)

/-- Theorem: Nancy spends $110.00 on coffee over 20 days -/
theorem nancy_coffee_spend :
  coffee_expenditure 20 3 2.5 = 110 := by
sorry

end nancy_coffee_spend_l2121_212102


namespace day_of_week_problem_l2121_212135

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek :=
  sorry

theorem day_of_week_problem (N : Year) :
  dayOfWeek N 250 = DayOfWeek.Sunday →
  dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Sunday →
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Monday :=
sorry

end day_of_week_problem_l2121_212135


namespace trig_expression_equals_negative_four_l2121_212164

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 * Real.sin (10 * π / 180) - Real.cos (10 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.sin (10 * π / 180)) = -4 := by
  sorry

end trig_expression_equals_negative_four_l2121_212164


namespace pencil_price_decrease_l2121_212161

/-- The original price for a set of pencils -/
def original_set_price : ℚ := 4

/-- The number of pencils in the original set -/
def original_set_count : ℕ := 3

/-- The promotional price for a set of pencils -/
def promo_set_price : ℚ := 3

/-- The number of pencils in the promotional set -/
def promo_set_count : ℕ := 4

/-- Calculate the price per pencil given the set price and count -/
def price_per_pencil (set_price : ℚ) (set_count : ℕ) : ℚ :=
  set_price / set_count

/-- Calculate the percent decrease between two prices -/
def percent_decrease (old_price : ℚ) (new_price : ℚ) : ℚ :=
  (old_price - new_price) / old_price * 100

/-- The theorem stating the percent decrease in pencil price -/
theorem pencil_price_decrease :
  let original_price := price_per_pencil original_set_price original_set_count
  let promo_price := price_per_pencil promo_set_price promo_set_count
  let decrease := percent_decrease original_price promo_price
  ∃ (ε : ℚ), abs (decrease - 43.6) < ε ∧ ε < 0.1 :=
sorry

end pencil_price_decrease_l2121_212161


namespace tan_equality_l2121_212189

theorem tan_equality : 
  3.439 * Real.tan (110 * π / 180) + Real.tan (50 * π / 180) + Real.tan (20 * π / 180) = 
  Real.tan (110 * π / 180) * Real.tan (50 * π / 180) * Real.tan (20 * π / 180) := by
  sorry

end tan_equality_l2121_212189


namespace alice_monthly_increase_l2121_212108

/-- Represents Alice's savings pattern over three months -/
def aliceSavings (initialSavings : ℝ) (monthlyIncrease : ℝ) : ℝ :=
  initialSavings + (initialSavings + monthlyIncrease) + (initialSavings + 2 * monthlyIncrease)

/-- Theorem stating Alice's monthly savings increase -/
theorem alice_monthly_increase (initialSavings totalSavings : ℝ) 
  (h1 : initialSavings = 10)
  (h2 : totalSavings = 70)
  (h3 : ∃ x : ℝ, aliceSavings initialSavings x = totalSavings) :
  ∃ x : ℝ, x = 40 / 3 ∧ aliceSavings initialSavings x = totalSavings :=
sorry

end alice_monthly_increase_l2121_212108


namespace race_result_l2121_212114

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- The race setup -/
def Race (sasha lesha kolya : Runner) : Prop :=
  sasha.speed > 0 ∧ lesha.speed > 0 ∧ kolya.speed > 0 ∧
  sasha.speed ≠ lesha.speed ∧ sasha.speed ≠ kolya.speed ∧ lesha.speed ≠ kolya.speed ∧
  sasha.distance = 100 ∧
  lesha.distance = 90 ∧
  kolya.distance = 81

theorem race_result (sasha lesha kolya : Runner) 
  (h : Race sasha lesha kolya) : 
  sasha.distance - kolya.distance = 19 := by
  sorry

end race_result_l2121_212114


namespace trisection_intersection_l2121_212199

/-- Given two points on the natural logarithm curve, prove that the x-coordinate of the 
    intersection point between a horizontal line through the first trisection point and 
    the curve is 2^(7/3). -/
theorem trisection_intersection (A B C : ℝ × ℝ) : 
  A.1 = 2 → 
  A.2 = Real.log 2 →
  B.1 = 32 → 
  B.2 = Real.log 32 →
  C.2 = (2 / 3) * A.2 + (1 / 3) * B.2 →
  ∃ (x : ℝ), x > 0 ∧ Real.log x = C.2 →
  x = 2^(7/3) := by
sorry

end trisection_intersection_l2121_212199


namespace g_zero_at_three_l2121_212183

-- Define the polynomial g(x)
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

-- Theorem statement
theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -867 := by sorry

end g_zero_at_three_l2121_212183


namespace equation_solution_l2121_212141

theorem equation_solution (x : ℝ) : 
  (x^2 - 2*x - 8 = -(x + 4)*(x - 1)) ↔ (x = -2 ∨ x = 3) :=
by sorry

end equation_solution_l2121_212141


namespace direction_vector_coefficient_l2121_212177

/-- Given a line passing through points (-2, 5) and (1, 0), prove that its direction vector of the form (a, -1) has a = 3/5 -/
theorem direction_vector_coefficient (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-2, 5) → p2 = (1, 0) → 
  (p2.1 - p1.1, p2.2 - p1.2) = (3 * a, -3 * a) → 
  a = 3/5 := by sorry

end direction_vector_coefficient_l2121_212177


namespace belyNaliv_triple_l2121_212133

/-- Represents the number of apples of each variety -/
structure AppleCount where
  antonovka : ℝ
  grushovka : ℝ
  belyNaliv : ℝ

/-- The total number of apples -/
def totalApples (count : AppleCount) : ℝ :=
  count.antonovka + count.grushovka + count.belyNaliv

/-- Condition: Tripling Antonovka apples increases the total by 70% -/
axiom antonovka_triple (count : AppleCount) :
  2 * count.antonovka = 0.7 * totalApples count

/-- Condition: Tripling Grushovka apples increases the total by 50% -/
axiom grushovka_triple (count : AppleCount) :
  2 * count.grushovka = 0.5 * totalApples count

/-- Theorem: Tripling Bely Naliv apples increases the total by 80% -/
theorem belyNaliv_triple (count : AppleCount) :
  2 * count.belyNaliv = 0.8 * totalApples count := by
  sorry

end belyNaliv_triple_l2121_212133


namespace total_distance_mercedes_davonte_l2121_212169

/-- 
Given:
- Jonathan ran 7.5 kilometers
- Mercedes ran twice the distance of Jonathan
- Davonte ran 2 kilometers farther than Mercedes

Prove that the total distance run by Mercedes and Davonte is 32 kilometers
-/
theorem total_distance_mercedes_davonte (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ)
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
  sorry

end total_distance_mercedes_davonte_l2121_212169


namespace range_of_m_l2121_212163

theorem range_of_m (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end range_of_m_l2121_212163


namespace solve_linear_equation_l2121_212153

theorem solve_linear_equation :
  ∃ x : ℚ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) ∧ x = -1/11 := by
  sorry

end solve_linear_equation_l2121_212153


namespace cos_585_degrees_l2121_212131

theorem cos_585_degrees :
  Real.cos (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_585_degrees_l2121_212131


namespace triangle_side_length_l2121_212181

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 →
  a^2 + c^2 = 3*a*c →
  b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l2121_212181


namespace triangle_at_most_one_obtuse_angle_l2121_212138

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse_angle (t : Triangle) : 
  ¬(∃ (i j : Fin 3), i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end triangle_at_most_one_obtuse_angle_l2121_212138


namespace xy_equation_implications_l2121_212193

theorem xy_equation_implications (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end xy_equation_implications_l2121_212193


namespace base_9_to_base_10_conversion_l2121_212180

-- Define the base-9 number
def base_9_number : ℕ := 5126

-- Define the conversion function from base 9 to base 10
def base_9_to_base_10 (n : ℕ) : ℕ :=
  (n % 10) +
  ((n / 10) % 10) * 9 +
  ((n / 100) % 10) * 9^2 +
  ((n / 1000) % 10) * 9^3

-- Theorem statement
theorem base_9_to_base_10_conversion :
  base_9_to_base_10 base_9_number = 3750 := by
  sorry

end base_9_to_base_10_conversion_l2121_212180


namespace negative_fraction_comparison_l2121_212143

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end negative_fraction_comparison_l2121_212143


namespace triangle_cut_theorem_l2121_212103

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Represents the line PQ that cuts the triangle -/
structure CuttingLine where
  length : ℝ

/-- Theorem statement for the triangle problem -/
theorem triangle_cut_theorem 
  (triangle : IsoscelesTriangle) 
  (cutting_line : CuttingLine) : 
  triangle.height = 30 ∧ 
  triangle.base * triangle.height / 2 = 180 ∧
  triangle.base * triangle.height / 2 - 135 = 
    (triangle.base * triangle.height / 2) / 4 →
  cutting_line.length = 6 := by
  sorry

end triangle_cut_theorem_l2121_212103


namespace line_intersects_circle_l2121_212105

/-- The line l with equation x - y + √3 = 0 intersects the circle C with equation x² + (y - √2)² = 2 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), (x - y + Real.sqrt 3 = 0) ∧ (x^2 + (y - Real.sqrt 2)^2 = 2) := by
  sorry

end line_intersects_circle_l2121_212105


namespace strawberry_milk_probability_l2121_212121

theorem strawberry_milk_probability :
  let n : ℕ := 6  -- Total number of days
  let k : ℕ := 5  -- Number of successful days
  let p : ℚ := 3/4  -- Probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 729/2048 := by
  sorry

end strawberry_milk_probability_l2121_212121


namespace no_singleton_set_with_conditions_l2121_212109

theorem no_singleton_set_with_conditions :
  ¬ ∃ (A : Set ℝ), (∃ (a : ℝ), A = {a}) ∧
    (∀ a : ℝ, a ∈ A → (1 / (1 - a)) ∈ A) ∧
    (1 ∈ A) := by
  sorry

end no_singleton_set_with_conditions_l2121_212109


namespace building_height_is_100_l2121_212107

/-- The height of a building with an elevator --/
def building_height (acceleration : ℝ) (constant_velocity : ℝ) (constant_time : ℝ) (acc_time : ℝ) : ℝ :=
  -- Distance during acceleration and deceleration
  2 * (0.5 * acceleration * acc_time^2) +
  -- Distance during constant velocity
  constant_velocity * constant_time

/-- Theorem stating the height of the building --/
theorem building_height_is_100 :
  building_height 2.5 5 18 2 = 100 := by
  sorry

#eval building_height 2.5 5 18 2

end building_height_is_100_l2121_212107


namespace sam_lee_rates_sum_of_squares_l2121_212157

theorem sam_lee_rates_sum_of_squares : 
  ∃ (r c k : ℕ+), 
    (4 * r + 5 * c + 3 * k = 120) ∧ 
    (5 * r + 3 * c + 4 * k = 138) ∧ 
    (r ^ 2 + c ^ 2 + k ^ 2 = 436) := by
  sorry

end sam_lee_rates_sum_of_squares_l2121_212157


namespace estimated_percentage_is_5_7_l2121_212160

/-- Represents the data from the household survey -/
structure SurveyData where
  total_households : ℕ
  ordinary_families : ℕ
  high_income_families : ℕ
  ordinary_sample_size : ℕ
  high_income_sample_size : ℕ
  ordinary_with_3plus_houses : ℕ
  high_income_with_3plus_houses : ℕ

/-- Calculates the estimated percentage of families with 3 or more houses -/
def estimatePercentage (data : SurveyData) : ℚ :=
  let ordinary_estimate := (data.ordinary_families : ℚ) * (data.ordinary_with_3plus_houses : ℚ) / (data.ordinary_sample_size : ℚ)
  let high_income_estimate := (data.high_income_families : ℚ) * (data.high_income_with_3plus_houses : ℚ) / (data.high_income_sample_size : ℚ)
  let total_estimate := ordinary_estimate + high_income_estimate
  (total_estimate / (data.total_households : ℚ)) * 100

/-- The survey data for the household study -/
def surveyData : SurveyData := {
  total_households := 100000,
  ordinary_families := 99000,
  high_income_families := 1000,
  ordinary_sample_size := 990,
  high_income_sample_size := 100,
  ordinary_with_3plus_houses := 50,
  high_income_with_3plus_houses := 70
}

/-- Theorem stating that the estimated percentage of families with 3 or more houses is 5.7% -/
theorem estimated_percentage_is_5_7 :
  estimatePercentage surveyData = 57/10 := by
  sorry


end estimated_percentage_is_5_7_l2121_212160


namespace two_cos_45_equals_sqrt_2_l2121_212113

theorem two_cos_45_equals_sqrt_2 : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_equals_sqrt_2_l2121_212113


namespace solution_set_for_a_eq_1_minimum_value_range_l2121_212167

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem minimum_value_range :
  {a : ℝ | ∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y} = {a : ℝ | -3 ≤ a ∧ a ≤ 3} := by sorry

end solution_set_for_a_eq_1_minimum_value_range_l2121_212167


namespace kilometer_to_leaps_l2121_212120

/-- Conversion between units of length -/
theorem kilometer_to_leaps 
  (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (h1 : p * (1 : ℝ) = q * (1 : ℝ))  -- p strides = q leaps
  (h2 : r * (1 : ℝ) = s * (1 : ℝ))  -- r bounds = s strides
  (h3 : t * (1 : ℝ) = u * (1 : ℝ))  -- t bounds = u kilometers
  : (1 : ℝ) * (1 : ℝ) = (t * s * q) / (u * r * p) * (1 : ℝ) := by
  sorry


end kilometer_to_leaps_l2121_212120


namespace defective_bulb_probability_l2121_212154

/-- The probability of selecting at least one defective bulb when choosing two bulbs at random from a box containing 24 bulbs, of which 4 are defective, is 43/138. -/
theorem defective_bulb_probability (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 24) (h2 : defective_bulbs = 4) :
  let non_defective : ℕ := total_bulbs - defective_bulbs
  let prob_both_non_defective : ℚ := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 43 / 138 := by
  sorry

end defective_bulb_probability_l2121_212154


namespace cupboard_cost_price_proof_l2121_212124

/-- The cost price of a cupboard satisfying given conditions -/
def cupboard_cost_price : ℝ := 6250

/-- The selling price of the cupboard -/
def selling_price (cost : ℝ) : ℝ := cost * (1 - 0.12)

/-- The selling price that would result in a 12% profit -/
def profit_selling_price (cost : ℝ) : ℝ := cost * (1 + 0.12)

theorem cupboard_cost_price_proof :
  selling_price cupboard_cost_price + 1500 = profit_selling_price cupboard_cost_price :=
sorry

end cupboard_cost_price_proof_l2121_212124


namespace evaluate_expression_l2121_212130

theorem evaluate_expression : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end evaluate_expression_l2121_212130


namespace range_of_positive_integers_in_list_K_l2121_212188

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list_K :
  let K := consecutive_integers (-5) 12
  range (positive_integers K) = 5 := by
  sorry

end range_of_positive_integers_in_list_K_l2121_212188


namespace weekend_rain_probability_l2121_212119

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.72 :=
by sorry

end weekend_rain_probability_l2121_212119


namespace increase_mode_effect_l2121_212174

def shoe_sizes : List ℕ := [35, 36, 37, 38, 39]
def sales_quantities : List ℕ := [2, 8, 10, 6, 2]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem increase_mode_effect 
  (most_common : ℕ) 
  (h1 : most_common ∈ shoe_sizes) 
  (h2 : ∀ x ∈ shoe_sizes, (sales_quantities.count most_common) ≥ (sales_quantities.count x)) :
  ∃ n : ℕ, 
    (mode (sales_quantities.map (λ x => if x = most_common then x + n else x)) = mode sales_quantities) ∧
    (mean (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ mean sales_quantities ∨
     median (sales_quantities.map (λ x => if x = most_common then x + n else x)) = median sales_quantities ∨
     variance (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ variance sales_quantities) :=
by sorry

end increase_mode_effect_l2121_212174


namespace tom_run_distance_l2121_212191

theorem tom_run_distance (total_distance : ℝ) (walk_speed : ℝ) (run_speed : ℝ) 
  (friend_time : ℝ) (max_total_time : ℝ) :
  total_distance = 2800 →
  walk_speed = 75 →
  run_speed = 225 →
  friend_time = 5 →
  max_total_time = 30 →
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_distance ∧
    (x / walk_speed + (total_distance - x) / run_speed + friend_time ≤ max_total_time) ∧
    (total_distance - x ≤ 1387.5) :=
by sorry

end tom_run_distance_l2121_212191


namespace coopers_age_l2121_212140

theorem coopers_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (maria_older : maria = dante + 1) :
  cooper = 6 := by
  sorry

end coopers_age_l2121_212140


namespace bill_division_l2121_212137

/-- The total bill amount when three people divide it evenly -/
def total_bill (individual_payment : ℕ) : ℕ := 3 * individual_payment

/-- Theorem: If three people divide a bill evenly and each pays $33, then the total bill is $99 -/
theorem bill_division (individual_payment : ℕ) 
  (h : individual_payment = 33) : 
  total_bill individual_payment = 99 := by
  sorry

end bill_division_l2121_212137


namespace electronic_dogs_distance_l2121_212128

/-- Represents a vertex of a cube --/
inductive Vertex
| A | B | C | D | A1 | B1 | C1 | D1

/-- Represents the position of an electronic dog on the cube --/
structure DogPosition where
  vertex : Vertex
  segments_completed : Nat

/-- The cube with edge length 1 --/
def unitCube : Set Vertex := {Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.A1, Vertex.B1, Vertex.C1, Vertex.D1}

/-- The distance between two vertices of the unit cube --/
def distance (v1 v2 : Vertex) : Real := sorry

/-- The movement rule for the dogs --/
def validMove (v1 v2 v3 : Vertex) : Prop := sorry

/-- The final position of the black dog after 2008 segments --/
def blackDogFinalPosition : DogPosition := ⟨Vertex.A, 2008⟩

/-- The final position of the yellow dog after 2009 segments --/
def yellowDogFinalPosition : DogPosition := ⟨Vertex.A1, 2009⟩

theorem electronic_dogs_distance :
  distance blackDogFinalPosition.vertex yellowDogFinalPosition.vertex = 1 := by sorry

end electronic_dogs_distance_l2121_212128


namespace election_votes_total_l2121_212142

/-- Proves that the total number of votes in an election is 180, given that Emma received 4/15 of the total votes and 48 votes in total. -/
theorem election_votes_total (emma_fraction : Rat) (emma_votes : ℕ) (total_votes : ℕ) 
  (h1 : emma_fraction = 4 / 15)
  (h2 : emma_votes = 48)
  (h3 : emma_fraction * total_votes = emma_votes) :
  total_votes = 180 := by
  sorry

end election_votes_total_l2121_212142


namespace bob_has_22_pennies_l2121_212185

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- Condition 1: If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have twice as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 2 * (alex_pennies + 2)

/-- Theorem: Bob currently has 22 pennies -/
theorem bob_has_22_pennies : bob_pennies = 22 := by sorry

end bob_has_22_pennies_l2121_212185


namespace root_in_interval_l2121_212111

def f (x : ℝ) := 3*x^2 + 3*x - 8

theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (f 1 < 0) →
  (f 1.5 > 0) →
  (f 1.25 < 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end root_in_interval_l2121_212111


namespace arithmetic_sequence_problem_l2121_212151

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 3 + a 13 - a 8 = 24 := by
  sorry

end arithmetic_sequence_problem_l2121_212151


namespace discount_store_purchase_l2121_212158

/-- Represents the number of items of each type bought by one person -/
structure ItemCounts where
  typeA : ℕ
  typeB : ℕ

/-- Represents the prices of items -/
structure Prices where
  typeA : ℕ
  typeB : ℕ

def total_spent (counts : ItemCounts) (prices : Prices) : ℕ :=
  counts.typeA * prices.typeA + counts.typeB * prices.typeB

theorem discount_store_purchase : ∃ (counts : ItemCounts),
  let prices : Prices := ⟨8, 9⟩
  total_spent counts prices = 172 ∧
  counts.typeA + counts.typeB = counts.typeA + counts.typeB ∧
  counts.typeA = 4 ∧
  counts.typeB = 6 := by
  sorry

end discount_store_purchase_l2121_212158


namespace neighbor_purchase_theorem_l2121_212123

/-- Proves that given the conditions of the problem, the total amount spent is 168 shillings -/
theorem neighbor_purchase_theorem (x : ℝ) 
  (h1 : x > 0)  -- Quantity purchased is positive
  (h2 : 2*x + 1.5*x = 3.5*x)  -- Total cost equation
  (h3 : (3.5*x/2)/2 + (3.5*x/2)/1.5 = 2*x + 2)  -- Equal division condition
  : 3.5*x = 168 := by
  sorry

end neighbor_purchase_theorem_l2121_212123


namespace slope_angle_range_l2121_212118

-- Define the slope k and the angle θ
variable (k : ℝ) (θ : ℝ)

-- Define the condition that the lines intersect in the first quadrant
def intersect_in_first_quadrant (k : ℝ) : Prop :=
  (3 + Real.sqrt 3) / (1 + k) > 0 ∧ (3 * k - Real.sqrt 3) / (1 + k) > 0

-- Define the relationship between k and θ
def slope_angle_relation (k θ : ℝ) : Prop :=
  k = Real.tan θ

-- State the theorem
theorem slope_angle_range (h1 : intersect_in_first_quadrant k) 
  (h2 : slope_angle_relation k θ) : 
  θ > Real.pi / 6 ∧ θ < Real.pi / 2 :=
sorry

end slope_angle_range_l2121_212118


namespace quadratic_two_roots_l2121_212196

theorem quadratic_two_roots (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (4 * a)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end quadratic_two_roots_l2121_212196


namespace simplify_and_ratio_l2121_212184

theorem simplify_and_ratio (m : ℚ) : 
  let expr := (6 * m + 18) / 6
  let simplified := m + 3
  expr = simplified ∧ 
  (∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 3) := by
sorry

end simplify_and_ratio_l2121_212184


namespace triangle_inradius_l2121_212190

/-- Given a triangle with perimeter 32 cm and area 56 cm², its inradius is 3.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : p = 32) 
    (h2 : A = 56) 
    (h3 : A = r * p / 2) : r = 3.5 := by
  sorry

end triangle_inradius_l2121_212190


namespace vasya_drove_two_fifths_l2121_212186

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton + d.vasya + d.sasha + d.dima = 1 ∧  -- Total distance is 1
  d.anton = d.vasya / 2 ∧                     -- Anton drove half of Vasya's distance
  d.sasha = d.anton + d.dima ∧                -- Sasha drove as long as Anton and Dima combined
  d.dima = 1 / 10                             -- Dima drove one-tenth of the distance

/-- Theorem: Under the given conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths (d : DistanceFractions) 
  (h : driving_conditions d) : d.vasya = 2 / 5 := by
  sorry

end vasya_drove_two_fifths_l2121_212186


namespace least_common_multiple_first_ten_l2121_212132

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
  sorry

end least_common_multiple_first_ten_l2121_212132


namespace part_one_part_two_l2121_212170

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_arithmetic_sequence (B A C : ℝ) : Prop :=
  ∃ d : ℝ, A - B = C - A ∧ A - B = d

-- Theorem 1
theorem part_one (t : Triangle) (m : ℝ) 
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a^2 - t.c^2 = t.b^2 - m*t.b*t.c) : 
  m = 1 := by sorry

-- Theorem 2
theorem part_two (t : Triangle)
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b + t.c = 3) :
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by sorry

end part_one_part_two_l2121_212170


namespace bella_roses_from_parents_l2121_212150

/-- The number of dancer friends Bella has -/
def num_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := 44

/-- The number of roses Bella received from her parents -/
def roses_from_parents : ℕ := total_roses - (num_friends * roses_per_friend)

theorem bella_roses_from_parents :
  roses_from_parents = 24 :=
sorry

end bella_roses_from_parents_l2121_212150


namespace unique_solution_l2121_212100

/-- Define the function f as specified in the problem -/
def f (x y z : ℕ+) : ℤ :=
  (((x + y - 2) * (x + y - 1)) / 2) - z

/-- Theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (a b c d : ℕ+), f a b c = 1993 ∧ f c d a = 1993 ∧ a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42 :=
sorry

end unique_solution_l2121_212100


namespace parabola_translation_l2121_212173

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-2) 0 0
  let translated := translate original 1 (-3)
  translated = Parabola.mk (-2) 4 (-3) := by sorry

end parabola_translation_l2121_212173


namespace max_value_trig_expression_l2121_212134

theorem max_value_trig_expression (α β : Real) (h1 : 0 ≤ α ∧ α ≤ π/4) (h2 : 0 ≤ β ∧ β ≤ π/4) :
  ∃ (M : Real), M = Real.sqrt 5 ∧ ∀ (x y : Real), 0 ≤ x ∧ x ≤ π/4 → 0 ≤ y ∧ y ≤ π/4 →
    Real.sin (x - y) + 2 * Real.sin (x + y) ≤ M :=
by sorry

end max_value_trig_expression_l2121_212134


namespace town_population_problem_l2121_212155

theorem town_population_problem :
  ∃ n : ℕ, 
    (∃ a b : ℕ, 
      n * (n + 1) / 2 + 121 = a^2 ∧
      n * (n + 1) / 2 + 121 + 144 = b^2) ∧
    n * (n + 1) / 2 = 2280 := by
  sorry

end town_population_problem_l2121_212155


namespace f_greater_than_one_exists_max_a_l2121_212166

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Statement 1
theorem f_greater_than_one (x : ℝ) (h : x > 0) : f 2 x > 1 := by
  sorry

-- Statement 2
theorem exists_max_a :
  ∃ (a : ℕ), (∀ (x : ℝ), x > 0 → f_deriv a x ≥ x^2 * Real.log x) ∧
  (∀ (b : ℕ), (∀ (x : ℝ), x > 0 → f_deriv b x ≥ x^2 * Real.log x) → b ≤ a) := by
  sorry

end

end f_greater_than_one_exists_max_a_l2121_212166


namespace fifth_term_value_l2121_212195

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem fifth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end fifth_term_value_l2121_212195
