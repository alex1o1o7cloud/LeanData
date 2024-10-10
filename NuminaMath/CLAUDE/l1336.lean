import Mathlib

namespace bing_dwen_dwen_prices_l1336_133689

theorem bing_dwen_dwen_prices 
  (total_budget : ℝ) 
  (budget_A : ℝ) 
  (price_difference : ℝ) 
  (quantity_ratio : ℝ) :
  total_budget = 1700 →
  budget_A = 800 →
  price_difference = 25 →
  quantity_ratio = 3 →
  ∃ (price_B : ℝ) (price_A : ℝ),
    price_B = 15 ∧
    price_A = 40 ∧
    price_A = price_B + price_difference ∧
    (total_budget - budget_A) / price_B = quantity_ratio * (budget_A / price_A) := by
  sorry

#check bing_dwen_dwen_prices

end bing_dwen_dwen_prices_l1336_133689


namespace road_trip_gas_usage_l1336_133687

/-- Calculates the total gallons of gas used on a road trip --/
theorem road_trip_gas_usage
  (highway_miles : ℝ)
  (highway_efficiency : ℝ)
  (city_miles : ℝ)
  (city_efficiency : ℝ)
  (h1 : highway_miles = 210)
  (h2 : highway_efficiency = 35)
  (h3 : city_miles = 54)
  (h4 : city_efficiency = 18) :
  highway_miles / highway_efficiency + city_miles / city_efficiency = 9 :=
by
  sorry

#check road_trip_gas_usage

end road_trip_gas_usage_l1336_133687


namespace polynomial_factorization_l1336_133618

theorem polynomial_factorization (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end polynomial_factorization_l1336_133618


namespace binomial_probability_ge_two_l1336_133670

/-- A random variable following a Binomial distribution B(10, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function of ξ -/
def pmf (k : ℕ) : ℝ := sorry

/-- The cumulative distribution function of ξ -/
def cdf (k : ℕ) : ℝ := sorry

theorem binomial_probability_ge_two :
  (1 - cdf 1) = 1013 / 1024 :=
sorry

end binomial_probability_ge_two_l1336_133670


namespace extension_point_coordinates_l1336_133620

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance from P₁ to P is twice the distance from P to P₂,
    prove that P has the specified coordinates. -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t ∉ [0, 1] ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by sorry

end extension_point_coordinates_l1336_133620


namespace total_sales_theorem_l1336_133697

/-- Calculate total sales from lettuce and tomatoes -/
def total_sales (customers : ℕ) (lettuce_per_customer : ℕ) (lettuce_price : ℚ) 
  (tomatoes_per_customer : ℕ) (tomato_price : ℚ) : ℚ :=
  (customers * lettuce_per_customer * lettuce_price) + 
  (customers * tomatoes_per_customer * tomato_price)

/-- Theorem: Total sales from lettuce and tomatoes is $2000 per month -/
theorem total_sales_theorem : 
  total_sales 500 2 1 4 (1/2) = 2000 := by
  sorry

end total_sales_theorem_l1336_133697


namespace xia_initial_stickers_l1336_133633

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Theorem: Xia had 150 stickers at the beginning -/
theorem xia_initial_stickers :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 := by
  sorry

end xia_initial_stickers_l1336_133633


namespace shark_observation_l1336_133639

theorem shark_observation (p_truth : ℝ) (p_shark : ℝ) (n : ℕ) :
  p_truth = 1/6 →
  p_shark = 0.027777777777777773 →
  p_shark = p_truth * (1 / n) →
  n = 6 := by sorry

end shark_observation_l1336_133639


namespace vector_sum_magnitude_l1336_133616

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, and the angle between them is 120°, then |a - b| = √13 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3) 
  (h2 : ‖b‖ = 4) 
  (h3 : a.1 * b.1 + a.2 * b.2 = -6) : ‖a - b‖ = Real.sqrt 13 := by
  sorry

end vector_sum_magnitude_l1336_133616


namespace keith_total_expenses_l1336_133648

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

theorem keith_total_expenses : 
  speakers_cost + cd_player_cost + tires_cost = 387.85 := by
  sorry

end keith_total_expenses_l1336_133648


namespace parabola_circle_theorem_l1336_133638

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define line l
def line_l (y : ℝ) : ℝ := 2 * y + 2

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Origin
def O : ℝ × ℝ := (0, 0)

theorem parabola_circle_theorem :
  parabola E.1 E.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  A.1 = line_l A.2 →
  B.1 = line_l B.2 →
  A ≠ E →
  B ≠ E →
  M.2 = -2 →
  N.2 = -2 →
  (∃ t : ℝ, M = (1 - t) • E + t • A) →
  (∃ s : ℝ, N = (1 - s) • E + s • B) →
  (O.1 - M.1) * (O.1 - N.1) + (O.2 - M.2) * (O.2 - N.2) = 0 :=
sorry

end parabola_circle_theorem_l1336_133638


namespace blackjack_payout_ratio_l1336_133604

/-- Represents the payout ratio for a blackjack in a casino game -/
structure BlackjackPayout where
  original_bet : ℚ
  total_payout : ℚ

/-- Calculates the payout ratio for a blackjack given the original bet and total payout -/
def payout_ratio (bp : BlackjackPayout) : ℚ × ℚ :=
  let winnings := bp.total_payout - bp.original_bet
  (winnings, bp.original_bet)

/-- Theorem stating that for the given conditions, the payout ratio is 1:2 -/
theorem blackjack_payout_ratio :
  let bp := BlackjackPayout.mk 40 60
  payout_ratio bp = (1, 2) := by
  sorry

end blackjack_payout_ratio_l1336_133604


namespace least_positive_integer_t_l1336_133650

theorem least_positive_integer_t : ∃ (t : ℕ+), 
  (∀ (x y : ℕ+), (x^2 + y^2)^2 + 2*t*x*(x^2 + y^2) = t^2*y^2 → t ≥ 25) ∧ 
  (∃ (x y : ℕ+), (x^2 + y^2)^2 + 2*25*x*(x^2 + y^2) = 25^2*y^2) := by
  sorry

end least_positive_integer_t_l1336_133650


namespace intersection_on_y_axis_l1336_133623

/-- Given two lines l₁ and l₂, prove that if their intersection is on the y-axis, then C = -4 -/
theorem intersection_on_y_axis (A : ℝ) :
  ∃ (x y : ℝ),
    (A * x + 3 * y + C = 0) ∧
    (2 * x - 3 * y + 4 = 0) ∧
    (x = 0) →
    C = -4 :=
by sorry

end intersection_on_y_axis_l1336_133623


namespace a_range_l1336_133690

theorem a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |Real.sin x| > a)
  (h3 : ∀ x : ℝ, x ∈ [π/4, 3*π/4] → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ici (Real.sqrt 2 / 2) ∩ Set.Iio 1 :=
by sorry

end a_range_l1336_133690


namespace candy_distribution_l1336_133666

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) (num_students : ℕ) : 
  total_candy = 18 → candy_per_student = 2 → total_candy = candy_per_student * num_students → num_students = 9 := by
  sorry

end candy_distribution_l1336_133666


namespace max_product_constraint_l1336_133683

theorem max_product_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 40) :
  x * y ≤ 400 := by
  sorry

end max_product_constraint_l1336_133683


namespace max_students_social_practice_l1336_133676

theorem max_students_social_practice (max_fund car_rental per_student_cost : ℕ) 
  (h1 : max_fund = 800)
  (h2 : car_rental = 300)
  (h3 : per_student_cost = 15) :
  ∃ (max_students : ℕ), 
    max_students = 33 ∧ 
    max_students * per_student_cost + car_rental ≤ max_fund ∧
    ∀ (n : ℕ), n * per_student_cost + car_rental ≤ max_fund → n ≤ max_students :=
sorry

end max_students_social_practice_l1336_133676


namespace translation_result_l1336_133699

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y + dy)

theorem translation_result :
  let P : ℝ × ℝ := (-2, 1)
  let dx : ℝ := 3
  let dy : ℝ := 4
  let P' : ℝ × ℝ := translate_point P.1 P.2 dx dy
  P' = (1, 5) := by sorry

end translation_result_l1336_133699


namespace salt_solution_replacement_l1336_133630

theorem salt_solution_replacement (original_salt_percentage : Real) 
  (replaced_fraction : Real) (final_salt_percentage : Real) 
  (replacing_salt_percentage : Real) : 
  original_salt_percentage = 13 →
  replaced_fraction = 1/4 →
  final_salt_percentage = 16 →
  (1 - replaced_fraction) * original_salt_percentage + 
    replaced_fraction * replacing_salt_percentage = final_salt_percentage →
  replacing_salt_percentage = 25 := by
sorry

end salt_solution_replacement_l1336_133630


namespace accommodation_arrangements_theorem_l1336_133631

/-- The number of ways to arrange 5 people in 3 rooms with constraints -/
def accommodationArrangements (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 5 people in 3 rooms with A and B not sharing -/
def accommodationArrangementsWithConstraint (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

theorem accommodation_arrangements_theorem :
  accommodationArrangementsWithConstraint 5 3 2 = 72 :=
sorry

end accommodation_arrangements_theorem_l1336_133631


namespace unique_triplet_l1336_133693

theorem unique_triplet : ∃! (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b * c + 1) % a = 0 ∧
  (a * c + 1) % b = 0 ∧
  (a * b + 1) % c = 0 ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end unique_triplet_l1336_133693


namespace quotient_problem_l1336_133626

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by sorry

end quotient_problem_l1336_133626


namespace percentage_of_120_to_40_l1336_133617

theorem percentage_of_120_to_40 : ∀ (x y : ℝ), x = 120 ∧ y = 40 → (x / y) * 100 = 300 := by
  sorry

end percentage_of_120_to_40_l1336_133617


namespace cans_difference_l1336_133647

/-- The number of cans collected by Sarah yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of cans collected by Lara yesterday -/
def lara_yesterday : ℕ := sarah_yesterday + 30

/-- The number of cans collected by Alex yesterday -/
def alex_yesterday : ℕ := 90

/-- The number of cans collected by Sarah today -/
def sarah_today : ℕ := 40

/-- The number of cans collected by Lara today -/
def lara_today : ℕ := 70

/-- The number of cans collected by Alex today -/
def alex_today : ℕ := 55

/-- The total number of cans collected yesterday -/
def total_yesterday : ℕ := sarah_yesterday + lara_yesterday + alex_yesterday

/-- The total number of cans collected today -/
def total_today : ℕ := sarah_today + lara_today + alex_today

theorem cans_difference : total_yesterday - total_today = 55 := by
  sorry

end cans_difference_l1336_133647


namespace binomial_sum_first_six_l1336_133675

theorem binomial_sum_first_six (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1024 := by
sorry

end binomial_sum_first_six_l1336_133675


namespace forest_trees_l1336_133612

/-- Calculates the total number of trees in a forest given the conditions --/
theorem forest_trees (street_side : ℝ) (forest_area_multiplier : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_area_multiplier = 3 →
  trees_per_sqm = 4 →
  (forest_area_multiplier * street_side^2 * trees_per_sqm : ℝ) = 120000 := by
  sorry

#check forest_trees

end forest_trees_l1336_133612


namespace rectangle_area_l1336_133669

/-- Rectangle ABCD with point E on AB and point F on AC -/
structure RectangleConfig where
  /-- Length of side AD -/
  a : ℝ
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2
  /-- AB = 2 × AD -/
  ab_twice_ad : B.1 - A.1 = 2 * a
  /-- E is the midpoint of AB -/
  e_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- F is on AC -/
  f_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  /-- F is on DE -/
  f_on_de : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (D.1 + s * (E.1 - D.1), D.2 + s * (E.2 - D.2))
  /-- Area of quadrilateral BFED is 50 -/
  area_bfed : abs ((B.1 - F.1) * (E.2 - D.2) - (B.2 - F.2) * (E.1 - D.1)) / 2 = 50

/-- The area of rectangle ABCD is 300 -/
theorem rectangle_area (config : RectangleConfig) : (config.B.1 - config.A.1) * (config.B.2 - config.D.2) = 300 := by
  sorry

end rectangle_area_l1336_133669


namespace A_intersect_B_l1336_133679

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by sorry

end A_intersect_B_l1336_133679


namespace horizontal_distance_P_Q_l1336_133625

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem stating the horizontal distance between P and Q -/
theorem horizontal_distance_P_Q : 
  ∀ (xp xq : ℝ), 
  f xp = 8 → 
  f xq = -1 → 
  (∀ x : ℝ, f x = -1 → |x - xp| ≥ |xq - xp|) → 
  |xq - xp| = 3 * Real.sqrt 3 := by
  sorry

end horizontal_distance_P_Q_l1336_133625


namespace car_travel_distance_l1336_133601

def ring_travel (d1 d2 d4 total : ℕ) : Prop :=
  ∃ d3 : ℕ, d1 + d2 + d3 + d4 = total

theorem car_travel_distance :
  ∀ (d1 d2 d4 total : ℕ),
    d1 = 5 →
    d2 = 8 →
    d4 = 0 →
    total = 23 →
    ring_travel d1 d2 d4 total →
    ∃ d3 : ℕ, d3 = 10 :=
by
  sorry

end car_travel_distance_l1336_133601


namespace expression_equality_l1336_133610

theorem expression_equality : (2 + Real.sqrt 6) * (2 - Real.sqrt 6) - (Real.sqrt 3 + 1)^2 = -6 - 2 * Real.sqrt 3 := by
  sorry

end expression_equality_l1336_133610


namespace only_cylinder_has_quadrilateral_cross_section_l1336_133696

-- Define the types of solids
inductive Solid
| Cone
| Cylinder
| Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def has_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => True
  | Solid.Sphere => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, has_quadrilateral_cross_section s ↔ s = Solid.Cylinder :=
by sorry

end only_cylinder_has_quadrilateral_cross_section_l1336_133696


namespace matching_socks_probability_l1336_133694

/-- The number of gray-bottomed socks -/
def gray_socks : ℕ := 12

/-- The number of white-bottomed socks -/
def white_socks : ℕ := 10

/-- The total number of socks -/
def total_socks : ℕ := gray_socks + white_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of selecting a matching pair of socks -/
theorem matching_socks_probability :
  (choose_two gray_socks + choose_two white_socks : ℚ) / choose_two total_socks = 111 / 231 := by
  sorry

end matching_socks_probability_l1336_133694


namespace angle_CED_measure_l1336_133622

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the circles
def circle1 : Set (ℝ × ℝ) := sorry
def circle2 : Set (ℝ × ℝ) := sorry

-- State the conditions
axiom circles_congruent : circle1 = circle2
axiom A_center_circle1 : A ∈ circle1
axiom B_center_circle2 : B ∈ circle2
axiom B_on_circle1 : B ∈ circle1
axiom A_on_circle2 : A ∈ circle2
axiom C_on_line_AB : sorry
axiom D_on_line_AB : sorry
axiom E_intersection : E ∈ circle1 ∩ circle2

-- Define the angle CED
def angle_CED : ℝ := sorry

-- Theorem to prove
theorem angle_CED_measure : angle_CED = 120 := by sorry

end angle_CED_measure_l1336_133622


namespace prime_triple_divisibility_l1336_133678

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end prime_triple_divisibility_l1336_133678


namespace sum_of_roots_l1336_133674

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end sum_of_roots_l1336_133674


namespace min_socks_for_twenty_pairs_l1336_133632

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (requiredPairs : Nat) : Nat :=
  5 + 2 * (requiredPairs - 1)

/-- Theorem stating the minimum number of socks needed for 20 pairs -/
theorem min_socks_for_twenty_pairs (drawer : SockDrawer) 
  (h1 : drawer.red = 120)
  (h2 : drawer.green = 100)
  (h3 : drawer.blue = 80)
  (h4 : drawer.black = 50) :
  minSocksForPairs drawer 20 = 43 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 80, black := 50 } 20

end min_socks_for_twenty_pairs_l1336_133632


namespace max_value_of_vector_expression_l1336_133600

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_expression (a b c : V) 
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) :
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 + 2 * ‖a + b - c‖^2 ≤ 290 := by
  sorry

end max_value_of_vector_expression_l1336_133600


namespace incorrect_expression_l1336_133686

theorem incorrect_expression (x y : ℚ) (h : x / y = 4 / 5) : 
  (x + 2 * y) / y = 14 / 5 ∧ 
  y / (2 * x - y) = 5 / 3 ∧ 
  (4 * x - y) / y = 11 / 5 ∧ 
  x / (3 * y) = 4 / 15 ∧ 
  (2 * x - y) / x ≠ 7 / 5 := by
  sorry

end incorrect_expression_l1336_133686


namespace sets_properties_l1336_133640

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) :=
by sorry

end sets_properties_l1336_133640


namespace paving_rate_calculation_l1336_133671

/-- Given a rectangular room with length and width, and the total cost of paving,
    calculate the rate of paving per square meter. -/
theorem paving_rate_calculation 
  (length width total_cost : ℝ) 
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) : 
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_calculation_l1336_133671


namespace max_value_F_l1336_133605

/-- The function f(x) = ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The function g(x) = cx² + bx + a -/
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

/-- The function F(x) = |f(x) · g(x)| -/
def F (a b c x : ℝ) : ℝ := |f a b c x * g a b c x|

theorem max_value_F (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  ∃ M, M = 2 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, F a b c x ≤ M :=
sorry

end max_value_F_l1336_133605


namespace soda_cost_l1336_133688

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℚ
  soda : ℚ
  fries : ℚ

/-- The total cost of Keegan's purchase in cents -/
def keegan_total (c : Cost) : ℚ := 3 * c.burger + 2 * c.soda + c.fries

/-- The total cost of Alex's purchase in cents -/
def alex_total (c : Cost) : ℚ := 2 * c.burger + 3 * c.soda + c.fries

theorem soda_cost :
  ∃ (c : Cost),
    keegan_total c = 975 ∧
    alex_total c = 900 ∧
    c.soda = 18.75 := by
  sorry

end soda_cost_l1336_133688


namespace tank_b_height_is_six_l1336_133664

/-- The height of a cylindrical tank B, given another tank A and their properties --/
def heightOfTankB (heightA circumferenceA circumferenceB : ℝ) : ℝ :=
  let radiusA := circumferenceA / (2 * Real.pi)
  let radiusB := circumferenceB / (2 * Real.pi)
  let capacityA := Real.pi * radiusA^2 * heightA
  6

/-- Theorem stating that the height of tank B is 6 meters under given conditions --/
theorem tank_b_height_is_six (heightA circumferenceA circumferenceB : ℝ)
  (h_heightA : heightA = 10)
  (h_circumferenceA : circumferenceA = 6)
  (h_circumferenceB : circumferenceB = 10)
  (h_capacity_ratio : Real.pi * (circumferenceA / (2 * Real.pi))^2 * heightA = 
                      0.6 * (Real.pi * (circumferenceB / (2 * Real.pi))^2 * heightOfTankB heightA circumferenceA circumferenceB)) :
  heightOfTankB heightA circumferenceA circumferenceB = 6 := by
  sorry

#check tank_b_height_is_six

end tank_b_height_is_six_l1336_133664


namespace min_mushrooms_collected_l1336_133691

/-- Represents the number of mushrooms collected by Vasya and Masha over two days -/
structure MushroomCollection where
  vasya_day1 : ℕ
  vasya_day2 : ℕ

/-- Calculates the total number of mushrooms collected by both Vasya and Masha -/
def total_mushrooms (c : MushroomCollection) : ℚ :=
  (c.vasya_day1 + c.vasya_day2 : ℚ) + 
  ((3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2)

/-- Checks if the collection satisfies the given conditions -/
def is_valid_collection (c : MushroomCollection) : Prop :=
  (3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2 = 
  (11/10 : ℚ) * (c.vasya_day1 + c.vasya_day2)

/-- The main theorem stating the minimum number of mushrooms collected -/
theorem min_mushrooms_collected :
  ∃ (c : MushroomCollection), 
    is_valid_collection c ∧ 
    (∀ (c' : MushroomCollection), is_valid_collection c' → 
      total_mushrooms c ≤ total_mushrooms c') ∧
    ⌈total_mushrooms c⌉ = 19 := by
  sorry


end min_mushrooms_collected_l1336_133691


namespace ceiling_sqrt_225_l1336_133692

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end ceiling_sqrt_225_l1336_133692


namespace sum_2011_is_29_l1336_133682

/-- Given a sequence of 2011 consecutive five-digit numbers, this function
    returns the sum of digits for the nth number in the sequence. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence starts with a five-digit number. -/
axiom start_five_digit : ∃ k : ℕ, 10000 ≤ k ∧ k < 100000 ∧ ∀ i, 1 ≤ i ∧ i ≤ 2011 → k + i - 1 < 100000

/-- The sum of digits of the 21st number is 37. -/
axiom sum_21 : sumOfDigits 21 = 37

/-- The sum of digits of the 54th number is 7. -/
axiom sum_54 : sumOfDigits 54 = 7

/-- The main theorem: the sum of digits of the 2011th number is 29. -/
theorem sum_2011_is_29 : sumOfDigits 2011 = 29 := sorry

end sum_2011_is_29_l1336_133682


namespace x_complements_c_l1336_133667

/-- Represents a date in a month --/
structure Date :=
  (value : ℕ)
  (h : value > 0 ∧ value ≤ 31)

/-- Represents letters on the calendar --/
inductive Letter
| A | B | C | X

/-- A calendar is a function that assigns a date to each letter --/
def Calendar := Letter → Date

/-- The condition that B is two weeks after A --/
def twoWeeksAfter (cal : Calendar) : Prop :=
  (cal Letter.B).value = (cal Letter.A).value + 14

/-- The condition that the sum of dates behind C and X equals the sum of dates behind A and B --/
def sumEqual (cal : Calendar) : Prop :=
  (cal Letter.C).value + (cal Letter.X).value = (cal Letter.A).value + (cal Letter.B).value

/-- The main theorem --/
theorem x_complements_c (cal : Calendar) 
  (h1 : twoWeeksAfter cal) 
  (h2 : sumEqual cal) : 
  (cal Letter.X).value = (cal Letter.C).value + 18 :=
sorry

end x_complements_c_l1336_133667


namespace equal_principal_repayment_formula_l1336_133654

/-- Repayment amount for the nth month -/
def repayment_amount (n : ℕ) : ℚ :=
  3928 - 8 * n

/-- Properties of the loan -/
def loan_amount : ℚ := 480000
def repayment_years : ℕ := 20
def monthly_interest_rate : ℚ := 4 / 1000

theorem equal_principal_repayment_formula :
  ∀ n : ℕ, n > 0 → n ≤ repayment_years * 12 →
  repayment_amount n =
    loan_amount / (repayment_years * 12) +
    (loan_amount - (n - 1) * (loan_amount / (repayment_years * 12))) * monthly_interest_rate :=
by sorry

end equal_principal_repayment_formula_l1336_133654


namespace function_property_l1336_133661

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x + 2

theorem function_property (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ ∈ Set.Icc 1 (exp 1), ∃ x₂ ∈ Set.Icc 1 (exp 1), f a x₁ + f a x₂ = 4) →
  a = exp 1 + 1 := by
  sorry

end function_property_l1336_133661


namespace point_P_coordinates_l1336_133665

-- Define the coordinate system and points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- State the theorem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    vec A P = (2 : ℝ) • vec P B ∧ 
    P = (7/3, 1/3) := by
  sorry

end point_P_coordinates_l1336_133665


namespace gigi_additional_batches_l1336_133607

/-- Represents the number of cups of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches Gigi has already baked -/
def batches_baked : ℕ := 3

/-- Represents the total amount of flour in Gigi's bag -/
def total_flour : ℕ := 20

/-- Calculates the number of additional batches Gigi can make with the remaining flour -/
def additional_batches : ℕ := (total_flour - batches_baked * flour_per_batch) / flour_per_batch

/-- Proves that Gigi can make 7 more batches of cookies with the remaining flour -/
theorem gigi_additional_batches : additional_batches = 7 := by
  sorry

end gigi_additional_batches_l1336_133607


namespace empty_set_proof_l1336_133621

theorem empty_set_proof : {x : ℝ | x > 6 ∧ x < 1} = ∅ := by
  sorry

end empty_set_proof_l1336_133621


namespace divisible_by_two_l1336_133614

theorem divisible_by_two (a : ℤ) (h : 2 ∣ a^2) : 2 ∣ a := by
  sorry

end divisible_by_two_l1336_133614


namespace smallest_3digit_prime_factor_of_binom_300_150_l1336_133646

theorem smallest_3digit_prime_factor_of_binom_300_150 :
  let n := Nat.choose 300 150
  ∃ (p : Nat), Prime p ∧ 100 ≤ p ∧ p < 1000 ∧ p ∣ n ∧
    ∀ (q : Nat), Prime q → 100 ≤ q → q < p → ¬(q ∣ n) :=
by
  sorry

end smallest_3digit_prime_factor_of_binom_300_150_l1336_133646


namespace sum_squared_equals_129_l1336_133636

theorem sum_squared_equals_129 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 + a*b + b^2 = 25)
  (h2 : b^2 + b*c + c^2 = 49)
  (h3 : c^2 + c*a + a^2 = 64) :
  (a + b + c)^2 = 129 := by
  sorry

end sum_squared_equals_129_l1336_133636


namespace connor_date_cost_l1336_133628

/-- The cost of Connor's movie date -/
def movie_date_cost (ticket_price : ℚ) (ticket_quantity : ℕ) (combo_meal_price : ℚ) (candy_price : ℚ) (candy_quantity : ℕ) : ℚ :=
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem: Connor's movie date costs $36.00 -/
theorem connor_date_cost :
  movie_date_cost 10 2 11 (5/2) 2 = 36 :=
sorry

end connor_date_cost_l1336_133628


namespace conic_sections_eccentricity_l1336_133608

theorem conic_sections_eccentricity (x : ℝ) : 
  (2 * x^2 - 5 * x + 2 = 0) →
  (x = 2 ∨ x = 1/2) ∧ 
  ((0 < x ∧ x < 1) ∨ x > 1) :=
sorry

end conic_sections_eccentricity_l1336_133608


namespace functional_equation_solutions_l1336_133627

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = x - 1) ∨ (∀ x, f x = 1 - x) := by
  sorry

end functional_equation_solutions_l1336_133627


namespace game_price_is_correct_l1336_133611

/-- The price of each game Zachary sold -/
def game_price : ℝ := 5

/-- The number of games Zachary sold -/
def zachary_games : ℕ := 40

/-- The amount of money Zachary received -/
def zachary_amount : ℝ := game_price * zachary_games

/-- The amount of money Jason received -/
def jason_amount : ℝ := zachary_amount * 1.3

/-- The amount of money Ryan received -/
def ryan_amount : ℝ := jason_amount + 50

/-- The total amount received by all three friends -/
def total_amount : ℝ := 770

theorem game_price_is_correct : 
  zachary_amount + jason_amount + ryan_amount = total_amount := by sorry

end game_price_is_correct_l1336_133611


namespace product_calculation_l1336_133615

theorem product_calculation : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end product_calculation_l1336_133615


namespace perpendicular_line_exists_l1336_133651

-- Define a line in a 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to not be on a line
def Point.notOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c ≠ 0

-- Define what it means for a line to be perpendicular to another line
def Line.perpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The theorem statement
theorem perpendicular_line_exists (P : Point) (L : Line) (h : P.notOnLine L) :
  ∃ (M : Line), M.perpendicularTo L ∧ P.notOnLine M := by sorry

end perpendicular_line_exists_l1336_133651


namespace two_integers_sum_l1336_133644

theorem two_integers_sum (x y : ℕ+) : 
  (x : ℤ) - (y : ℤ) = 5 ∧ 
  (x : ℕ) * y = 180 → 
  (x : ℕ) + y = 25 := by
sorry

end two_integers_sum_l1336_133644


namespace remainder_puzzle_l1336_133629

theorem remainder_puzzle : (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 := by
  sorry

end remainder_puzzle_l1336_133629


namespace certain_number_proof_l1336_133659

theorem certain_number_proof (N : ℝ) : 
  (2 / 5 : ℝ) * N - (3 / 5 : ℝ) * 125 = 45 → N = 300 :=
by sorry

end certain_number_proof_l1336_133659


namespace monday_profit_ratio_l1336_133663

def total_profit : ℚ := 1200
def wednesday_profit : ℚ := 500
def tuesday_profit : ℚ := (1 / 4) * total_profit

def monday_profit : ℚ := total_profit - tuesday_profit - wednesday_profit

theorem monday_profit_ratio : 
  monday_profit / total_profit = 1 / 3 := by sorry

end monday_profit_ratio_l1336_133663


namespace negative_a_power_five_l1336_133643

theorem negative_a_power_five (a : ℝ) : (-a)^3 * (-a)^2 = -a^5 := by
  sorry

end negative_a_power_five_l1336_133643


namespace smallest_number_divisible_by_63_with_digit_sum_63_l1336_133698

def digit_sum (n : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := sorry

theorem smallest_number_divisible_by_63_with_digit_sum_63 :
  ∃ (n : ℕ),
    is_divisible_by n 63 ∧
    digit_sum n = 63 ∧
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 63 ∧ digit_sum m = 63)) ∧
    n = 63999999 :=
  sorry

end smallest_number_divisible_by_63_with_digit_sum_63_l1336_133698


namespace original_triangle_area_l1336_133695

theorem original_triangle_area
  (original : Real)  -- Area of the original triangle
  (new : Real)       -- Area of the new triangle
  (h1 : new = 256)   -- The area of the new triangle is 256 square feet
  (h2 : new = 16 * original)  -- The new triangle's area is 16 times the original
  : original = 16 :=
by sorry

end original_triangle_area_l1336_133695


namespace w_to_twelve_power_l1336_133672

theorem w_to_twelve_power (w : ℂ) (h : w = (-Real.sqrt 3 + Complex.I) / 3) :
  w^12 = 400 / 531441 := by
  sorry

end w_to_twelve_power_l1336_133672


namespace x_minus_p_in_terms_of_p_l1336_133656

theorem x_minus_p_in_terms_of_p (x p : ℝ) : 
  (|x - 3| = p + 1) → (x < 3) → (x - p = 2 - 2*p) := by
  sorry

end x_minus_p_in_terms_of_p_l1336_133656


namespace wrong_mark_value_l1336_133685

/-- Proves that the wrongly entered mark is 73 given the conditions of the problem -/
theorem wrong_mark_value (correct_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : correct_mark = 63)
  (h2 : class_size = 20)
  (h3 : average_increase = 1/2) :
  ∃ x : ℕ, x = 73 ∧ (x : ℚ) - correct_mark = class_size * average_increase := by
  sorry


end wrong_mark_value_l1336_133685


namespace tax_free_items_cost_l1336_133645

def total_cost : ℝ := 120

def first_bracket_percentage : ℝ := 0.4
def second_bracket_percentage : ℝ := 0.3
def tax_free_percentage : ℝ := 1 - first_bracket_percentage - second_bracket_percentage

def first_bracket_tax_rate : ℝ := 0.06
def second_bracket_tax_rate : ℝ := 0.08
def second_bracket_discount : ℝ := 0.05

def first_bracket_cost : ℝ := total_cost * first_bracket_percentage
def second_bracket_cost : ℝ := total_cost * second_bracket_percentage
def tax_free_cost : ℝ := total_cost * tax_free_percentage

theorem tax_free_items_cost :
  tax_free_cost = 36 := by sorry

end tax_free_items_cost_l1336_133645


namespace opposite_of_neg_three_l1336_133603

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end opposite_of_neg_three_l1336_133603


namespace tangent_parallel_line_a_value_l1336_133660

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x + 2

-- Define the point where the tangent touches the curve
def point : ℝ × ℝ := (1, 5)

-- Define the general form of the line parallel to the tangent
def parallel_line (a x y : ℝ) : Prop := 2 * a * x - y - 6 = 0

-- Theorem statement
theorem tangent_parallel_line_a_value :
  ∃ (a : ℝ), 
    (f point.1 = point.2) ∧ 
    (f' point.1 = 2 * a) ∧ 
    (parallel_line a point.1 point.2) ∧
    (a = 4) := by sorry

end tangent_parallel_line_a_value_l1336_133660


namespace custom_operation_result_l1336_133609

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a^2 - a*b

-- State the theorem
theorem custom_operation_result : star (star (-1) 2) 3 = 0 := by sorry

end custom_operation_result_l1336_133609


namespace sum_of_squares_of_roots_l1336_133602

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 11 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 11 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 157 / 36 := by
sorry

end sum_of_squares_of_roots_l1336_133602


namespace compound_molecular_weight_l1336_133655

-- Define atomic weights
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_C : ℝ := 12.011

-- Define number of atoms for each element
def num_H : ℕ := 4
def num_Cr : ℕ := 2
def num_O : ℕ := 4
def num_N : ℕ := 3
def num_C : ℕ := 5

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_Cr : ℝ) * atomic_weight_Cr +
  (num_O : ℝ) * atomic_weight_O +
  (num_N : ℝ) * atomic_weight_N +
  (num_C : ℝ) * atomic_weight_C

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight = 274.096 := by sorry

end compound_molecular_weight_l1336_133655


namespace cube_split_theorem_l1336_133680

def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1 ∧ 
  ∃ (start : ℕ), (Finset.range m).sum (λ i => 2 * (start + i) + 1) = m^3 ∧
  ∃ (i : Fin m), n = 2 * (start + i) + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : is_split_number m 333) : m = 18 := by
  sorry

end cube_split_theorem_l1336_133680


namespace number_of_signups_l1336_133681

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of sports competitions --/
def num_competitions : ℕ := 3

/-- Theorem: The number of ways for students to sign up for competitions --/
theorem number_of_signups : (num_competitions ^ num_students) = 243 := by
  sorry

end number_of_signups_l1336_133681


namespace part_one_solution_set_part_two_minimum_value_l1336_133624

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one_solution_set (x : ℝ) : 
  (f 1 x ≥ 4 - |x - 3|) ↔ (x ≤ 0 ∨ x ≥ 4) := by sorry

-- Part II
theorem part_two_minimum_value (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (Set.Icc 0 2 = {x | f a x ≤ 1}) → 
  (1 / m + 1 / (2 * n) = a) → 
  (∀ k l, k > 0 → l > 0 → 1 / k + 1 / (2 * l) = a → m * n ≤ k * l) →
  m * n = 2 := by sorry

end part_one_solution_set_part_two_minimum_value_l1336_133624


namespace reciprocal_problem_l1336_133619

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 50 * (1 / x) = 80 := by
  sorry

end reciprocal_problem_l1336_133619


namespace arithmetic_sequence_common_difference_l1336_133641

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S₃ = 3S₂ + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l1336_133641


namespace count_inequalities_l1336_133677

def is_inequality (e : String) : Bool :=
  match e with
  | "x - y" => false
  | "x ≤ y" => true
  | "x + y" => false
  | "x^2 - 3y" => false
  | "x ≥ 0" => true
  | "1/2x ≠ 3" => true
  | _ => false

def expressions : List String := [
  "x - y",
  "x ≤ y",
  "x + y",
  "x^2 - 3y",
  "x ≥ 0",
  "1/2x ≠ 3"
]

theorem count_inequalities :
  (expressions.filter is_inequality).length = 3 := by
  sorry

end count_inequalities_l1336_133677


namespace prob_two_even_balls_l1336_133653

/-- The probability of drawing two even-numbered balls without replacement from 16 balls numbered 1 to 16 is 7/30. -/
theorem prob_two_even_balls (n : ℕ) (h : n = 16) :
  (Nat.card {i : Fin n | i.val % 2 = 0} : ℚ) / n *
  ((Nat.card {i : Fin n | i.val % 2 = 0} - 1) : ℚ) / (n - 1) = 7 / 30 := by
  sorry

end prob_two_even_balls_l1336_133653


namespace arrangement_count_is_288_l1336_133657

/-- The number of ways to arrange 4 mathematics books and 4 history books with constraints -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 4
  let block_arrangements : ℕ := 2  -- Math block and history block
  let math_internal_arrangements : ℕ := Nat.factorial (math_books - 1)  -- Excluding M1
  let history_internal_arrangements : ℕ := Nat.factorial history_books
  block_arrangements * math_internal_arrangements * history_internal_arrangements

/-- Theorem stating that the number of valid arrangements is 288 -/
theorem arrangement_count_is_288 : arrangement_count = 288 := by
  sorry

end arrangement_count_is_288_l1336_133657


namespace train_length_calculation_l1336_133634

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 240 →
  passing_time = 37 →
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length = 130 := by
sorry

end train_length_calculation_l1336_133634


namespace visits_needed_is_eleven_l1336_133642

/-- The cost of headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference in cost between pool-only and sauna-only visits in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the number of pool-only visits needed to save for headphones -/
def visits_needed : ℕ :=
  let sauna_cost := (combined_cost - pool_sauna_diff) / 2
  let pool_only_cost := sauna_cost + pool_sauna_diff
  let savings_per_visit := combined_cost - pool_only_cost
  (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem visits_needed_is_eleven : visits_needed = 11 := by
  sorry

end visits_needed_is_eleven_l1336_133642


namespace inverse_proportion_example_l1336_133668

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_proportion_example :
  ∀ p q : ℝ → ℝ,
  InverselyProportional p q →
  p 6 = 25 →
  p 15 = 10 := by
sorry

end inverse_proportion_example_l1336_133668


namespace square_difference_l1336_133652

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
sorry

end square_difference_l1336_133652


namespace square_in_base_b_l1336_133649

/-- Represents a number in base b with digits d₂d₁d₀ --/
def base_b_number (b : ℕ) (d₂ d₁ d₀ : ℕ) : ℕ := d₂ * b^2 + d₁ * b + d₀

/-- The number 144 in base b --/
def number_144_b (b : ℕ) : ℕ := base_b_number b 1 4 4

theorem square_in_base_b (b : ℕ) (h : b > 4) :
  ∃ (n : ℕ), number_144_b b = n^2 := by
  sorry

end square_in_base_b_l1336_133649


namespace total_pens_l1336_133635

def red_pens : ℕ := 65
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

theorem total_pens : red_pens + blue_pens + black_pens = 168 := by
  sorry

end total_pens_l1336_133635


namespace soccer_team_goalies_l1336_133637

theorem soccer_team_goalies :
  ∀ (goalies defenders midfielders strikers : ℕ),
    defenders = 10 →
    midfielders = 2 * defenders →
    strikers = 7 →
    goalies + defenders + midfielders + strikers = 40 →
    goalies = 3 := by
  sorry

end soccer_team_goalies_l1336_133637


namespace ratio_chain_l1336_133662

theorem ratio_chain (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13) :
  a / d = 5 / 36 := by
  sorry

end ratio_chain_l1336_133662


namespace consecutive_sum_inequality_l1336_133673

theorem consecutive_sum_inequality (nums : Fin 100 → ℝ) 
  (h_distinct : ∀ i j : Fin 100, i ≠ j → nums i ≠ nums j) :
  ∃ i : Fin 100, nums i + nums ((i + 3) % 100) > nums ((i + 1) % 100) + nums ((i + 2) % 100) :=
sorry

end consecutive_sum_inequality_l1336_133673


namespace square_sum_given_sum_and_product_l1336_133658

theorem square_sum_given_sum_and_product (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 3) :
  2 * a^2 + 2 * b^2 = 38 := by
  sorry

end square_sum_given_sum_and_product_l1336_133658


namespace share_ratio_a_to_b_l1336_133606

/-- Prove that the ratio of A's share to B's share is 4:1 -/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 578 →
  b_share = c_share / 4 →
  a_share = 408 →
  b_share = 102 →
  c_share = 68 →
  a_share + b_share + c_share = amount →
  a_share / b_share = 4 := by
  sorry

end share_ratio_a_to_b_l1336_133606


namespace simplify_expression_l1336_133613

theorem simplify_expression (y : ℝ) : 4*y + 9*y^2 + 6 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 3 := by
  sorry

end simplify_expression_l1336_133613


namespace difference_of_squares_l1336_133684

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end difference_of_squares_l1336_133684
