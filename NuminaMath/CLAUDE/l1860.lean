import Mathlib

namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_ratio_l1860_186092

-- Define the Triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the similarity ratio between triangles
def similarityRatio (t1 t2 : Triangle) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem similar_triangles_perimeter_ratio 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) : 
  perimeter ABC / perimeter DEF = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_ratio_l1860_186092


namespace NUMINAMATH_CALUDE_jen_bryan_score_difference_l1860_186072

theorem jen_bryan_score_difference (bryan_score : ℕ) (total_points : ℕ) (sammy_mistakes : ℕ) :
  bryan_score = 20 →
  total_points = 35 →
  sammy_mistakes = 7 →
  ∃ (jen_score : ℕ) (sammy_score : ℕ),
    sammy_score = total_points - sammy_mistakes ∧
    jen_score = sammy_score + 2 ∧
    jen_score - bryan_score = 10 :=
by sorry

end NUMINAMATH_CALUDE_jen_bryan_score_difference_l1860_186072


namespace NUMINAMATH_CALUDE_pizza_slices_l1860_186068

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) : 
  total_pizzas = 21 → total_slices = 168 → slices_per_pizza * total_pizzas = total_slices → slices_per_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1860_186068


namespace NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l1860_186037

theorem pascal_triangle_row_15_fifth_number :
  let row := List.map (fun k => Nat.choose 15 k) (List.range 16)
  row[0] = 1 ∧ row[1] = 15 →
  row[4] = Nat.choose 15 4 ∧ Nat.choose 15 4 = 1365 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l1860_186037


namespace NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l1860_186040

/-- The minimum distance from the midpoint of two points on parallel lines x-y-5=0 and x-y-15=0 to the origin is 5√2. -/
theorem min_distance_midpoint_to_origin : ℝ → Prop := 
  fun d => ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 5) →
    (x₂ - y₂ = 15) →
    let midpoint_x := (x₁ + x₂) / 2
    let midpoint_y := (y₁ + y₂) / 2
    d = Real.sqrt 50

-- The proof is omitted
theorem min_distance_midpoint_to_origin_is_5sqrt2 : 
  min_distance_midpoint_to_origin (5 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l1860_186040


namespace NUMINAMATH_CALUDE_pants_cost_l1860_186061

theorem pants_cost (initial_amount : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  remaining_amount = 20 →
  initial_amount - (num_shirts * shirt_cost) - remaining_amount = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l1860_186061


namespace NUMINAMATH_CALUDE_triangle_properties_l1860_186054

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (∃ k : ℝ, t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k → Real.cos t.C < 0) ∧
  (Real.sin t.A > Real.sin t.B → t.A > t.B) ∧
  (t.C = π/3 ∧ t.b = 10 ∧ t.c = 9 → ∃ t1 t2 : Triangle, t1 ≠ t2 ∧ 
    t1.b = t.b ∧ t1.c = t.c ∧ t1.C = t.C ∧
    t2.b = t.b ∧ t2.c = t.c ∧ t2.C = t.C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1860_186054


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_l1860_186046

theorem cos_sum_seventh_roots : Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (8 * Real.pi / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_l1860_186046


namespace NUMINAMATH_CALUDE_substitution_result_l1860_186051

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem substitution_result (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 3 * x^2 ≠ 0) :
  F ((3 * x - x^3) / (1 + 3 * x^2)) = 3 * F x :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l1860_186051


namespace NUMINAMATH_CALUDE_hyperbola_center_l1860_186044

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola_equation x y → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

theorem hyperbola_center : is_center 2 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1860_186044


namespace NUMINAMATH_CALUDE_second_month_bill_l1860_186014

/-- Represents Elvin's monthly telephone bill -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill -/
def TelephoneBill.total (bill : TelephoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem second_month_bill
  (firstMonth secondMonth : TelephoneBill)
  (h1 : firstMonth.total = 46)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  secondMonth.total = 76 := by
  sorry

#check second_month_bill

end NUMINAMATH_CALUDE_second_month_bill_l1860_186014


namespace NUMINAMATH_CALUDE_small_cup_volume_l1860_186070

theorem small_cup_volume (small_cup : ℝ) (large_container : ℝ) : 
  (8 * small_cup + 5400 = large_container) →
  (12 * 530 = large_container) →
  small_cup = 120 := by
sorry

end NUMINAMATH_CALUDE_small_cup_volume_l1860_186070


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1860_186071

def math_club_selection (boys girls : ℕ) (team_size : ℕ) (team_boys team_girls : ℕ) : ℕ :=
  Nat.choose boys team_boys * Nat.choose girls team_girls

theorem math_club_team_selection :
  math_club_selection 7 9 6 4 2 = 1260 :=
by sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1860_186071


namespace NUMINAMATH_CALUDE_xiao_li_score_l1860_186055

/-- Calculates the comprehensive score based on content and culture scores -/
def comprehensive_score (content_score culture_score : ℝ) : ℝ :=
  0.4 * content_score + 0.6 * culture_score

/-- Theorem stating that Xiao Li's comprehensive score is 86 points -/
theorem xiao_li_score : comprehensive_score 80 90 = 86 := by
  sorry

end NUMINAMATH_CALUDE_xiao_li_score_l1860_186055


namespace NUMINAMATH_CALUDE_certain_number_problem_l1860_186058

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1860_186058


namespace NUMINAMATH_CALUDE_equation_solutions_l1860_186000

theorem equation_solutions :
  (∀ x : ℝ, (2*x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1860_186000


namespace NUMINAMATH_CALUDE_magic_square_y_zero_l1860_186033

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  g : ℤ
  h : ℤ
  i : ℤ
  is_magic : 
    a + b + c = d + e + f ∧
    a + b + c = g + h + i ∧
    a + b + c = a + d + g ∧
    a + b + c = b + e + h ∧
    a + b + c = c + f + i ∧
    a + b + c = a + e + i ∧
    a + b + c = c + e + g

/-- The theorem stating that y must be 0 in the given magic square configuration -/
theorem magic_square_y_zero (ms : MagicSquare) 
  (h1 : ms.a = y)
  (h2 : ms.b = 17)
  (h3 : ms.c = 124)
  (h4 : ms.d = 9) :
  y = 0 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_y_zero_l1860_186033


namespace NUMINAMATH_CALUDE_team_a_builds_30m_per_day_l1860_186041

/-- Represents the daily road-building rate of Team A in meters -/
def team_a_rate : ℝ := 30

/-- Represents the daily road-building rate of Team B in meters -/
def team_b_rate : ℝ := team_a_rate + 10

/-- Represents the total length of road built by Team A in meters -/
def team_a_total : ℝ := 120

/-- Represents the total length of road built by Team B in meters -/
def team_b_total : ℝ := 160

/-- Theorem stating that Team A's daily rate is 30m, given the problem conditions -/
theorem team_a_builds_30m_per_day :
  (team_a_total / team_a_rate = team_b_total / team_b_rate) ∧
  (team_b_rate = team_a_rate + 10) ∧
  (team_a_rate = 30) := by sorry

end NUMINAMATH_CALUDE_team_a_builds_30m_per_day_l1860_186041


namespace NUMINAMATH_CALUDE_smallest_m_for_cube_sum_inequality_l1860_186006

theorem smallest_m_for_cube_sum_inequality :
  ∃ (m : ℝ), m = 27 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧
  (∀ (m' : ℝ), m' < m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_cube_sum_inequality_l1860_186006


namespace NUMINAMATH_CALUDE_inequality_solution_l1860_186089

def inequality (a x : ℝ) : Prop :=
  (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

theorem inequality_solution :
  (∀ a : ℝ, inequality a x) ↔ x = -2 ∨ x = 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1860_186089


namespace NUMINAMATH_CALUDE_probability_at_least_one_black_l1860_186008

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black :
  let total_ways := Nat.choose total_balls selected_balls
  let all_red_ways := Nat.choose red_balls selected_balls
  let at_least_one_black_ways := total_ways - all_red_ways
  (at_least_one_black_ways : ℚ) / total_ways = 13 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_black_l1860_186008


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l1860_186031

/-- The area enclosed by the line y=4x and the curve y=x^3 is 8 -/
theorem area_between_line_and_curve : 
  let f (x : ℝ) := 4 * x
  let g (x : ℝ) := x^3
  ∫ x in (-2)..2, |f x - g x| = 8 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l1860_186031


namespace NUMINAMATH_CALUDE_initial_loss_percentage_l1860_186084

/-- Proves that for an article with a cost price of $400, if increasing the selling price
    by $100 results in a 5% gain, then the initial loss percentage is 20%. -/
theorem initial_loss_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price + 100 = 1.05 * cost_price) :
  (cost_price - selling_price) / cost_price * 100 = 20 := by
  sorry

#check initial_loss_percentage

end NUMINAMATH_CALUDE_initial_loss_percentage_l1860_186084


namespace NUMINAMATH_CALUDE_gregs_mom_cookies_l1860_186017

theorem gregs_mom_cookies (greg_halves brad_halves left_halves : ℕ) 
  (h1 : greg_halves = 4)
  (h2 : brad_halves = 6)
  (h3 : left_halves = 18) :
  (greg_halves + brad_halves + left_halves) / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gregs_mom_cookies_l1860_186017


namespace NUMINAMATH_CALUDE_movie_ticket_price_is_30_l1860_186007

/-- The price of a movie ticket -/
def movie_ticket_price : ℝ := sorry

/-- The price of a football game ticket -/
def football_ticket_price : ℝ := sorry

/-- Eight movie tickets cost 2 times as much as one football game ticket -/
axiom ticket_price_relation : 8 * movie_ticket_price = 2 * football_ticket_price

/-- The total amount paid for 8 movie tickets and 5 football game tickets is $840 -/
axiom total_cost : 8 * movie_ticket_price + 5 * football_ticket_price = 840

theorem movie_ticket_price_is_30 : movie_ticket_price = 30 := by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_is_30_l1860_186007


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l1860_186056

/-- Represents the distance traveled by a boat in one hour -/
structure BoatTravel where
  speedStillWater : ℝ
  distanceAgainstStream : ℝ
  timeTravel : ℝ

/-- Calculates the distance traveled along the stream -/
def distanceAlongStream (bt : BoatTravel) : ℝ :=
  let streamSpeed := bt.speedStillWater - bt.distanceAgainstStream
  (bt.speedStillWater + streamSpeed) * bt.timeTravel

/-- Theorem: Given the conditions, the boat travels 13 km along the stream -/
theorem boat_distance_along_stream :
  ∀ (bt : BoatTravel),
    bt.speedStillWater = 11 ∧
    bt.distanceAgainstStream = 9 ∧
    bt.timeTravel = 1 →
    distanceAlongStream bt = 13 := by
  sorry


end NUMINAMATH_CALUDE_boat_distance_along_stream_l1860_186056


namespace NUMINAMATH_CALUDE_symmetric_points_product_l1860_186039

/-- Given two points A and B symmetric about the origin, prove their coordinates' product -/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l1860_186039


namespace NUMINAMATH_CALUDE_ratio_equality_l1860_186026

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  a / b = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l1860_186026


namespace NUMINAMATH_CALUDE_horner_rule_f_3_l1860_186095

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_rule_f_3 :
  f 3 = horner_eval f_coeffs 3 ∧ horner_eval f_coeffs 3 = 1642 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_f_3_l1860_186095


namespace NUMINAMATH_CALUDE_fraction_problem_l1860_186065

theorem fraction_problem (n d : ℕ) (h1 : d = 2*n - 1) (h2 : (n + 1) * 5 = (d + 1) * 3) : n = 5 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1860_186065


namespace NUMINAMATH_CALUDE_smallest_integer_in_sequence_l1860_186066

theorem smallest_integer_in_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧ c < 90 →
  (a + b + c + 90) / 4 = 72 →
  a ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_sequence_l1860_186066


namespace NUMINAMATH_CALUDE_both_charts_rough_determination_l1860_186018

/-- Represents a chart type -/
inductive ChartType
  | ThreeD_Column
  | TwoD_Bar

/-- Represents the ability to determine relationships between categorical variables -/
inductive RelationshipDetermination
  | Accurate
  | Rough
  | Unable

/-- Function that determines the relationship determination capability of a chart type -/
def chart_relationship_determination : ChartType → RelationshipDetermination
  | ChartType.ThreeD_Column => RelationshipDetermination.Rough
  | ChartType.TwoD_Bar => RelationshipDetermination.Rough

/-- Theorem stating that both 3D column charts and 2D bar charts can roughly determine relationships -/
theorem both_charts_rough_determination :
  (chart_relationship_determination ChartType.ThreeD_Column = RelationshipDetermination.Rough) ∧
  (chart_relationship_determination ChartType.TwoD_Bar = RelationshipDetermination.Rough) :=
by
  sorry

#check both_charts_rough_determination

end NUMINAMATH_CALUDE_both_charts_rough_determination_l1860_186018


namespace NUMINAMATH_CALUDE_yoongi_calculation_l1860_186083

theorem yoongi_calculation (x : ℝ) : 5 * x = 30 → x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l1860_186083


namespace NUMINAMATH_CALUDE_bear_census_l1860_186029

def total_bears (black_a : ℕ) : ℕ :=
  let black_b := 3 * black_a
  let black_c := 2 * black_b
  let white_a := black_a / 2
  let white_b := black_b / 2
  let white_c := black_c / 2
  let brown_a := black_a + 40
  let brown_b := black_b + 40
  let brown_c := black_c + 40
  black_a + black_b + black_c +
  white_a + white_b + white_c +
  brown_a + brown_b + brown_c

theorem bear_census (black_a : ℕ) (h1 : black_a = 60) :
  total_bears black_a = 1620 := by
  sorry

end NUMINAMATH_CALUDE_bear_census_l1860_186029


namespace NUMINAMATH_CALUDE_ollie_caught_five_fish_l1860_186004

/-- The number of fish caught by Ollie given the fishing results of Patrick and Angus -/
def ollies_fish (patrick_fish : ℕ) (angus_more_than_patrick : ℕ) (ollie_fewer_than_angus : ℕ) : ℕ :=
  patrick_fish + angus_more_than_patrick - ollie_fewer_than_angus

/-- Theorem stating that Ollie caught 5 fish given the problem conditions -/
theorem ollie_caught_five_fish :
  ollies_fish 8 4 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ollie_caught_five_fish_l1860_186004


namespace NUMINAMATH_CALUDE_no_persistent_numbers_l1860_186021

/-- A number is persistent if, when multiplied by any positive integer, 
    the result always contains all ten digits 0,1,...,9. -/
def IsPersistent (n : ℕ) : Prop :=
  ∀ k : ℕ+, ∀ d : Fin 10, ∃ m : ℕ, (n * k : ℕ) / 10^m % 10 = d

/-- There are no persistent numbers. -/
theorem no_persistent_numbers : ¬∃ n : ℕ, IsPersistent n := by
  sorry


end NUMINAMATH_CALUDE_no_persistent_numbers_l1860_186021


namespace NUMINAMATH_CALUDE_boat_speed_distance_relationship_l1860_186009

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Represents distances traveled by the boat -/
structure BoatDistance where
  downstream : ℝ
  upstream : ℝ

/-- Theorem stating the relationship between boat speed, current speed, and distances traveled -/
theorem boat_speed_distance_relationship 
  (speed : BoatSpeed) 
  (distance : BoatDistance) 
  (currentSpeed : ℝ) :
  speed.stillWater = 12 →
  speed.downstream = speed.stillWater + currentSpeed →
  speed.upstream = speed.stillWater - currentSpeed →
  distance.downstream = speed.downstream * 3 →
  distance.upstream = speed.upstream * 15 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_distance_relationship_l1860_186009


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1860_186030

theorem fraction_equation_solution :
  ∀ y : ℚ, (2 / 5 : ℚ) - (1 / 3 : ℚ) = 4 / y → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1860_186030


namespace NUMINAMATH_CALUDE_smallest_abs_z_l1860_186096

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 3*I) = 20) :
  ∃ (min_abs : ℝ), min_abs = 2.25 ∧ ∀ w : ℂ, Complex.abs (w - 15) + Complex.abs (w + 3*I) = 20 → Complex.abs w ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l1860_186096


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1860_186069

/-- Given that the coefficient of x^-3 in the expansion of (2x - a/x)^7 is 84, prove that a = -1 -/
theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 7 5 : ℝ) * 2^2 * (-a)^5 = 84 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1860_186069


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1860_186082

/-- Represents a camp in the summer program -/
structure Camp where
  size : ℕ
  deriving Repr

/-- Represents the summer program -/
structure SummerProgram where
  totalStudents : ℕ
  camps : List Camp
  sampleSize : ℕ
  deriving Repr

/-- Calculates the number of students to be sampled from each camp -/
def stratifiedSample (program : SummerProgram) : List ℕ :=
  program.camps.map (fun camp => 
    (camp.size * program.sampleSize) / program.totalStudents)

/-- Theorem stating the correct stratified sampling for the given summer program -/
theorem correct_stratified_sample :
  let program : SummerProgram := {
    totalStudents := 500,
    camps := [{ size := 200 }, { size := 150 }, { size := 150 }],
    sampleSize := 50
  }
  stratifiedSample program = [20, 15, 15] := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1860_186082


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_l1860_186090

theorem mean_equality_implies_y (y : ℝ) : 
  (7 + 9 + 14 + 23) / 4 = (18 + y) / 2 → y = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_l1860_186090


namespace NUMINAMATH_CALUDE_monotonic_square_exists_l1860_186064

/-- A function that returns the number of digits of a positive integer in base 10 -/
def numDigits (x : ℕ+) : ℕ := sorry

/-- A function that checks if a positive integer is monotonic in base 10 -/
def isMonotonic (x : ℕ+) : Prop := sorry

/-- For every positive integer n, there exists an n-digit monotonic number which is a perfect square -/
theorem monotonic_square_exists (n : ℕ+) : ∃ x : ℕ+, 
  (numDigits x = n) ∧ 
  isMonotonic x ∧ 
  ∃ y : ℕ+, x = y * y := by
  sorry

end NUMINAMATH_CALUDE_monotonic_square_exists_l1860_186064


namespace NUMINAMATH_CALUDE_reflection_set_bounded_l1860_186067

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- The set of points generated by the reflection process -/
def ReflectionSet (A B C : Point) : Set Point :=
  sorry

/-- A line in the plane -/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is on one side of a line -/
def onOneSideOfLine (p : Point) (l : Line) : Prop :=
  sorry

theorem reflection_set_bounded (A B C : Point) (hDistinct : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  ∃ (l : Line), ∀ (p : Point), p ∈ ReflectionSet A B C → onOneSideOfLine p l :=
sorry

end NUMINAMATH_CALUDE_reflection_set_bounded_l1860_186067


namespace NUMINAMATH_CALUDE_school_bus_seats_l1860_186074

/-- Given a school with students and buses, calculate the number of seats per bus. -/
def seats_per_bus (total_students : ℕ) (num_buses : ℕ) : ℕ :=
  total_students / num_buses

/-- Theorem stating that for a school with 11210 students and 95 buses, each bus has 118 seats. -/
theorem school_bus_seats :
  seats_per_bus 11210 95 = 118 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_seats_l1860_186074


namespace NUMINAMATH_CALUDE_triangle_problem_l1860_186042

theorem triangle_problem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - 2 * a = 0 ∧
  b = Real.sqrt 7 ∧
  1/2 * a * b * Real.sin C = Real.sqrt 3 / 2 →
  B = 2 * π / 3 ∧ a + b + c = 3 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1860_186042


namespace NUMINAMATH_CALUDE_complex_modulus_product_range_l1860_186024

theorem complex_modulus_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 4)
  (h₂ : Complex.abs (z₁ - z₂) = 3) :
  7/4 ≤ Complex.abs (z₁ * z₂) ∧ Complex.abs (z₁ * z₂) ≤ 25/4 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_product_range_l1860_186024


namespace NUMINAMATH_CALUDE_stating_systematic_sampling_theorem_l1860_186023

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- 
  Given a systematic sampling scheme and a group number,
  returns the number drawn from that group
-/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * (s.population_size / s.sample_size)

/-- 
  Theorem stating that if the number drawn from the 13th group is 101
  in a systematic sampling of 20 from 160, then the number drawn from
  the 3rd group is 21
-/
theorem systematic_sampling_theorem :
  ∀ (s : SystematicSampling),
    s.population_size = 160 →
    s.sample_size = 20 →
    number_in_group s 13 = 101 →
    number_in_group s 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_stating_systematic_sampling_theorem_l1860_186023


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1860_186053

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1860_186053


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1860_186034

theorem polynomial_divisibility (a b c α β γ p : ℤ) (hp : Prime p)
  (h_div_α : p ∣ (a * α^2 + b * α + c))
  (h_div_β : p ∣ (a * β^2 + b * β + c))
  (h_div_γ : p ∣ (a * γ^2 + b * γ + c))
  (h_diff_αβ : ¬(p ∣ (α - β)))
  (h_diff_βγ : ¬(p ∣ (β - γ)))
  (h_diff_γα : ¬(p ∣ (γ - α))) :
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) ∧ (∀ x : ℤ, p ∣ (a * x^2 + b * x + c)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1860_186034


namespace NUMINAMATH_CALUDE_min_distance_theorem_l1860_186073

-- Define the conditions
def condition1 (a b : ℝ) : Prop := Real.log (b + 1) + a - 3 * b = 0

def condition2 (c d : ℝ) : Prop := 2 * d - c + Real.sqrt 5 = 0

-- Define the distance function
def distance_squared (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- State the theorem
theorem min_distance_theorem (a b c d : ℝ) 
  (h1 : condition1 a b) (h2 : condition2 c d) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y z w : ℝ), condition1 x y → condition2 z w → 
    distance_squared x y z w ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l1860_186073


namespace NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l1860_186010

def is_prime (n : ℕ) : Prop := sorry

def sum_of_primes_less_than_20 : ℕ := sorry

theorem sum_of_primes_less_than_20_is_77 : 
  sum_of_primes_less_than_20 = 77 := by sorry

end NUMINAMATH_CALUDE_sum_of_primes_less_than_20_is_77_l1860_186010


namespace NUMINAMATH_CALUDE_expression_value_l1860_186049

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1860_186049


namespace NUMINAMATH_CALUDE_max_value_product_l1860_186020

theorem max_value_product (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 + x*y + y^2) * (x^2 + x*z + z^2) * (y^2 + y*z + z^2) ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1860_186020


namespace NUMINAMATH_CALUDE_base_subtraction_l1860_186001

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The result of subtracting the base 10 representation of 243 in base 6
    from the base 10 representation of 325 in base 9 is 167 -/
theorem base_subtraction :
  to_base_10 [5, 2, 3] 9 - to_base_10 [3, 4, 2] 6 = 167 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l1860_186001


namespace NUMINAMATH_CALUDE_waiter_tips_theorem_l1860_186003

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips_theorem :
  total_tips 10 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_theorem_l1860_186003


namespace NUMINAMATH_CALUDE_correct_sample_l1860_186052

def random_number_table : List (List Nat) := [
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43, 84, 26, 34, 91, 64],
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54],
  [57, 60, 86, 32, 44, 09, 47, 27, 96, 54, 49, 17, 46, 09, 62, 90, 52, 84, 77, 27, 08, 02, 73, 43, 28]
]

def start_row : Nat := 5
def start_col : Nat := 4
def total_bottles : Nat := 80
def sample_size : Nat := 6

def is_valid_bottle (n : Nat) : Bool :=
  n < total_bottles

def select_sample (table : List (List Nat)) (row : Nat) (col : Nat) : List Nat :=
  sorry

theorem correct_sample :
  select_sample random_number_table start_row start_col = [77, 39, 49, 54, 43, 17] :=
by sorry

end NUMINAMATH_CALUDE_correct_sample_l1860_186052


namespace NUMINAMATH_CALUDE_probability_three_primes_equals_target_l1860_186016

/-- The number of sides on each die -/
def sides : ℕ := 12

/-- The number of dice rolled -/
def numDice : ℕ := 8

/-- The number of prime numbers on a 12-sided die -/
def numPrimes : ℕ := 5

/-- The number of dice that should show a prime number -/
def targetPrimes : ℕ := 3

/-- The probability of rolling exactly three prime numbers when rolling 8 fair 12-sided dice -/
def probabilityThreePrimes : ℚ :=
  (Nat.choose numDice targetPrimes : ℚ) *
  (numPrimes / sides) ^ targetPrimes *
  ((sides - numPrimes) / sides) ^ (numDice - targetPrimes)

theorem probability_three_primes_equals_target :
  probabilityThreePrimes = 448 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_equals_target_l1860_186016


namespace NUMINAMATH_CALUDE_range_of_a_l1860_186080

-- Define the conditions
def p (x : ℝ) : Prop := x^2 + 2*x > 3
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x a, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, (∃ x : ℝ, q x a) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1860_186080


namespace NUMINAMATH_CALUDE_david_work_rate_l1860_186077

/-- The number of days it takes John to complete the work -/
def john_days : ℝ := 9

/-- The number of days it takes David and John together to complete the work -/
def combined_days : ℝ := 3.2142857142857144

/-- The number of days it takes David to complete the work alone -/
def david_days : ℝ := 5

/-- Theorem stating that given John's work rate and the combined work rate of David and John,
    David's individual work rate can be determined -/
theorem david_work_rate (ε : ℝ) (h_ε : ε > 0) :
  ∃ (d : ℝ), abs (d - david_days) < ε ∧
  1 / d + 1 / john_days = 1 / combined_days :=
sorry

end NUMINAMATH_CALUDE_david_work_rate_l1860_186077


namespace NUMINAMATH_CALUDE_chocolate_distribution_chocolate_squares_per_student_l1860_186002

theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_multiplier : Nat) (num_students : Nat) : Nat :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

-- The main theorem
theorem chocolate_squares_per_student :
  chocolate_distribution 7 8 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_chocolate_squares_per_student_l1860_186002


namespace NUMINAMATH_CALUDE_stream_speed_l1860_186050

/-- Proves that given a man's swimming speed in still water and the relationship
    between upstream and downstream swimming times, the speed of the stream is 0.5 km/h. -/
theorem stream_speed (swimming_speed : ℝ) (upstream_time_ratio : ℝ) :
  swimming_speed = 1.5 →
  upstream_time_ratio = 2 →
  ∃ (stream_speed : ℝ),
    (swimming_speed + stream_speed) * 1 = (swimming_speed - stream_speed) * upstream_time_ratio ∧
    stream_speed = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1860_186050


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1860_186028

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_sum1 : a 2 + a 3 = 2) 
  (h_sum2 : a 4 + a 5 = 32) : 
  q = 4 ∨ q = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1860_186028


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1860_186097

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧
  (∃ x y : ℝ, y / x + x / y ≥ 2 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1860_186097


namespace NUMINAMATH_CALUDE_complex_solutions_count_l1860_186047

open Complex

theorem complex_solutions_count : 
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 + z^2 - 3*z - 3)
  ∃ (S : Finset ℂ), (∀ z ∈ S, f z = 0) ∧ (∀ z ∉ S, f z ≠ 0) ∧ Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l1860_186047


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l1860_186085

theorem product_remainder_by_ten : 
  (2468 * 7531 * 92045) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l1860_186085


namespace NUMINAMATH_CALUDE_steel_rod_length_l1860_186091

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  /-- The weight of the rod in kilograms -/
  weight : ℝ
  /-- The length of the rod in meters -/
  length : ℝ
  /-- The rod is uniform, so weight per unit length is constant -/
  uniform : weight / length = 19 / 5

/-- Theorem stating that a steel rod weighing 42.75 kg has a length of 11.25 meters -/
theorem steel_rod_length (rod : SteelRod) (h : rod.weight = 42.75) : rod.length = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_steel_rod_length_l1860_186091


namespace NUMINAMATH_CALUDE_all_blue_figures_are_small_l1860_186057

-- Define the universe of shapes
inductive Shape
| Square
| Triangle

-- Define colors
inductive Color
| Blue
| Red

-- Define sizes
inductive Size
| Large
| Small

-- Define a figure as a combination of shape, color, and size
structure Figure where
  shape : Shape
  color : Color
  size : Size

-- State the conditions
axiom large_is_square : 
  ∀ (f : Figure), f.size = Size.Large → f.shape = Shape.Square

axiom blue_is_triangle : 
  ∀ (f : Figure), f.color = Color.Blue → f.shape = Shape.Triangle

-- Theorem to prove
theorem all_blue_figures_are_small : 
  ∀ (f : Figure), f.color = Color.Blue → f.size = Size.Small :=
sorry

end NUMINAMATH_CALUDE_all_blue_figures_are_small_l1860_186057


namespace NUMINAMATH_CALUDE_school_boys_count_l1860_186099

theorem school_boys_count :
  ∀ (total_students : ℕ) (boys : ℕ),
    total_students = 400 →
    boys + (boys * total_students / 100) = total_students →
    boys = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1860_186099


namespace NUMINAMATH_CALUDE_jerry_butterflies_left_l1860_186011

/-- Given that Jerry had 93 butterflies initially and released 11 butterflies,
    prove that he now has 82 butterflies left. -/
theorem jerry_butterflies_left (initial : ℕ) (released : ℕ) (left : ℕ) : 
  initial = 93 → released = 11 → left = initial - released → left = 82 := by
  sorry

end NUMINAMATH_CALUDE_jerry_butterflies_left_l1860_186011


namespace NUMINAMATH_CALUDE_approx_cube_root_25_correct_l1860_186059

/-- Approximate value of the cube root of 25 -/
def approx_cube_root_25 : ℝ := 2.926

/-- Generalized binomial theorem approximation for small x -/
def binomial_approx (α x : ℝ) : ℝ := 1 + α * x

/-- Cube root of 27 -/
def cube_root_27 : ℝ := 3

theorem approx_cube_root_25_correct :
  let x := -2/27
  let α := 1/3
  approx_cube_root_25 = cube_root_27 * binomial_approx α x := by sorry

end NUMINAMATH_CALUDE_approx_cube_root_25_correct_l1860_186059


namespace NUMINAMATH_CALUDE_ten_integer_segments_l1860_186048

/-- Represents a right triangle ABC with integer side lengths -/
structure RightTriangle where
  ab : ℕ
  bc : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex B to the hypotenuse AC in a right triangle -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with sides 18 and 24,
    there are exactly 10 distinct integer lengths of segments from B to AC -/
theorem ten_integer_segments :
  ∃ (t : RightTriangle), t.ab = 18 ∧ t.bc = 24 ∧ count_integer_segments t = 10 :=
sorry

end NUMINAMATH_CALUDE_ten_integer_segments_l1860_186048


namespace NUMINAMATH_CALUDE_stone_exit_and_return_velocity_range_l1860_186094

/-- 
Theorem: Stone Exit and Return Velocity Range

For a stone thrown upwards in a well with the following properties:
- Well depth: h = 10 meters
- Cover cycle: opens for 1 second, closes for 1 second
- Stone thrown 0.5 seconds before cover opens
- Acceleration due to gravity: g = 10 m/s²

The initial velocities V for which the stone will exit the well and fall back onto the cover
are in the range (85/6, 33/2) ∪ (285/14, 45/2).
-/
theorem stone_exit_and_return_velocity_range (h g τ : ℝ) (V : ℝ) : 
  h = 10 → 
  g = 10 → 
  τ = 1 → 
  (V ∈ Set.Ioo (85/6) (33/2) ∪ Set.Ioo (285/14) (45/2)) ↔ 
  (∃ t : ℝ, 
    t > 0 ∧ 
    V * t - (1/2) * g * t^2 ≥ h ∧
    ∃ t' : ℝ, t' > t ∧ V * t' - (1/2) * g * t'^2 = 0 ∧
    (∃ n : ℕ, t' = (2*n + 3/2) * τ ∨ t' = (2*n + 7/2) * τ)) :=
by sorry

end NUMINAMATH_CALUDE_stone_exit_and_return_velocity_range_l1860_186094


namespace NUMINAMATH_CALUDE_sara_apples_l1860_186012

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) : 
  total = 240 →
  ali_factor = 7 →
  total = sara_apples + ali_factor * sara_apples →
  sara_apples = 30 := by
sorry

end NUMINAMATH_CALUDE_sara_apples_l1860_186012


namespace NUMINAMATH_CALUDE_system_solution_correct_l1860_186036

theorem system_solution_correct (x y : ℝ) : 
  x = 3 ∧ y = 1 → (2 * x - 3 * y = 3 ∧ x + 2 * y = 5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_correct_l1860_186036


namespace NUMINAMATH_CALUDE_pebble_collection_l1860_186093

theorem pebble_collection (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 15 → a = 1 → d = 1 → (n * (2 * a + (n - 1) * d)) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_l1860_186093


namespace NUMINAMATH_CALUDE_magnitude_BC_is_sqrt_29_l1860_186015

-- Define the points and vector
def A : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (0, 2)
def AB : ℝ × ℝ := (3, 5)

-- Theorem statement
theorem magnitude_BC_is_sqrt_29 :
  let B : ℝ × ℝ := (A.1 + AB.1, A.2 + AB.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  Real.sqrt (BC.1^2 + BC.2^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_BC_is_sqrt_29_l1860_186015


namespace NUMINAMATH_CALUDE_coffee_shop_spending_l1860_186022

theorem coffee_shop_spending (A B : ℝ) : 
  B = 0.5 * A → A = B + 15 → A + B = 45 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_l1860_186022


namespace NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l1860_186078

/-- Represents the different grade levels in the study -/
inductive GradeLevel
  | Three
  | Six
  | Nine

/-- Represents different sampling methods -/
inductive SamplingMethod
  | LotDrawing
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents the study of visual acuity across different grade levels -/
structure VisualAcuityStudy where
  gradeLevels : List GradeLevel
  sampleProportion : ℝ
  samplingMethod : SamplingMethod

/-- Checks if a sampling method is the most reasonable for a given study -/
def isMostReasonable (study : VisualAcuityStudy) (method : SamplingMethod) : Prop :=
  method = study.samplingMethod ∧
  ∀ otherMethod : SamplingMethod, otherMethod ≠ method → 
    (study.samplingMethod = otherMethod → False)

/-- The main theorem stating that stratified sampling is the most reasonable method for the visual acuity study -/
theorem stratified_sampling_most_reasonable (study : VisualAcuityStudy) :
  study.gradeLevels = [GradeLevel.Three, GradeLevel.Six, GradeLevel.Nine] →
  0 < study.sampleProportion ∧ study.sampleProportion ≤ 1 →
  isMostReasonable study SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l1860_186078


namespace NUMINAMATH_CALUDE_max_value_theorem_l1860_186087

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ 3/2 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1860_186087


namespace NUMINAMATH_CALUDE_mary_has_ten_marbles_l1860_186032

/-- The number of blue marbles Dan has -/
def dan_marbles : ℕ := 5

/-- The factor by which Mary has more marbles than Dan -/
def mary_factor : ℕ := 2

/-- The number of blue marbles Mary has -/
def mary_marbles : ℕ := mary_factor * dan_marbles

theorem mary_has_ten_marbles : mary_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_marbles_l1860_186032


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1860_186086

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P -/
def P (m : ℝ) : Point :=
  { x := 3 * m - 6, y := m + 1 }

/-- Definition of point A -/
def A : Point :=
  { x := 1, y := -2 }

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def lies_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- Two points form a line parallel to the x-axis if they have the same y-coordinate -/
def parallel_to_x_axis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

theorem point_P_coordinates :
  (∃ m : ℝ, lies_on_y_axis (P m) ∧ P m = { x := 0, y := 3 }) ∧
  (∃ m : ℝ, parallel_to_x_axis (P m) A ∧ P m = { x := -15, y := -2 }) :=
sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1860_186086


namespace NUMINAMATH_CALUDE_exactly_ten_naas_l1860_186005

-- Define the set S
variable (S : Type)

-- Define gib and naa as elements of S
variable (gib naa : S)

-- Define the collection relation
variable (is_collection_of : S → S → Prop)

-- Define the belonging relation
variable (belongs_to : S → S → Prop)

-- P1: Every gib is a collection of naas
axiom P1 : ∀ g : S, (g = gib) → ∃ n : S, (n = naa) ∧ is_collection_of g n

-- P2: Any two distinct gibs have two and only two naas in common
axiom P2 : ∀ g1 g2 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g1 ≠ g2) →
  ∃! n1 n2 : S, (n1 = naa) ∧ (n2 = naa) ∧ (n1 ≠ n2) ∧
  is_collection_of g1 n1 ∧ is_collection_of g1 n2 ∧
  is_collection_of g2 n1 ∧ is_collection_of g2 n2

-- P3: Every naa belongs to three and only three gibs
axiom P3 : ∀ n : S, (n = naa) →
  ∃! g1 g2 g3 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧
  (g1 ≠ g2) ∧ (g2 ≠ g3) ∧ (g1 ≠ g3) ∧
  belongs_to n g1 ∧ belongs_to n g2 ∧ belongs_to n g3

-- P4: There are exactly five gibs
axiom P4 : ∃! g1 g2 g3 g4 g5 : S,
  (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧ (g4 = gib) ∧ (g5 = gib) ∧
  (g1 ≠ g2) ∧ (g1 ≠ g3) ∧ (g1 ≠ g4) ∧ (g1 ≠ g5) ∧
  (g2 ≠ g3) ∧ (g2 ≠ g4) ∧ (g2 ≠ g5) ∧
  (g3 ≠ g4) ∧ (g3 ≠ g5) ∧
  (g4 ≠ g5)

-- Theorem: There are exactly ten naas
theorem exactly_ten_naas : ∃! n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 : S,
  (n1 = naa) ∧ (n2 = naa) ∧ (n3 = naa) ∧ (n4 = naa) ∧ (n5 = naa) ∧
  (n6 = naa) ∧ (n7 = naa) ∧ (n8 = naa) ∧ (n9 = naa) ∧ (n10 = naa) ∧
  (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n1 ≠ n6) ∧ (n1 ≠ n7) ∧ (n1 ≠ n8) ∧ (n1 ≠ n9) ∧ (n1 ≠ n10) ∧
  (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n2 ≠ n6) ∧ (n2 ≠ n7) ∧ (n2 ≠ n8) ∧ (n2 ≠ n9) ∧ (n2 ≠ n10) ∧
  (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n3 ≠ n6) ∧ (n3 ≠ n7) ∧ (n3 ≠ n8) ∧ (n3 ≠ n9) ∧ (n3 ≠ n10) ∧
  (n4 ≠ n5) ∧ (n4 ≠ n6) ∧ (n4 ≠ n7) ∧ (n4 ≠ n8) ∧ (n4 ≠ n9) ∧ (n4 ≠ n10) ∧
  (n5 ≠ n6) ∧ (n5 ≠ n7) ∧ (n5 ≠ n8) ∧ (n5 ≠ n9) ∧ (n5 ≠ n10) ∧
  (n6 ≠ n7) ∧ (n6 ≠ n8) ∧ (n6 ≠ n9) ∧ (n6 ≠ n10) ∧
  (n7 ≠ n8) ∧ (n7 ≠ n9) ∧ (n7 ≠ n10) ∧
  (n8 ≠ n9) ∧ (n8 ≠ n10) ∧
  (n9 ≠ n10) :=
sorry

end NUMINAMATH_CALUDE_exactly_ten_naas_l1860_186005


namespace NUMINAMATH_CALUDE_exists_invariant_point_l1860_186045

/-- A set of non-constant functions with specific properties -/
def FunctionSet (G : Set (ℝ → ℝ)) : Prop :=
  ∀ f ∈ G, ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧
  (∀ g ∈ G, (g ∘ f) ∈ G) ∧
  (Function.Bijective f → Function.invFun f ∈ G) ∧
  (∃ xₑ : ℝ, f xₑ = xₑ)

/-- The main theorem -/
theorem exists_invariant_point {G : Set (ℝ → ℝ)} (hG : FunctionSet G) :
  ∃ k : ℝ, ∀ f ∈ G, f k = k := by sorry

end NUMINAMATH_CALUDE_exists_invariant_point_l1860_186045


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l1860_186075

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^12345 + i^12346 + i^12347 + i^12348 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l1860_186075


namespace NUMINAMATH_CALUDE_proposition_truth_values_l1860_186098

theorem proposition_truth_values (p q : Prop) (hp : p) (hq : ¬q) :
  (p ∨ q) ∧ ¬(¬p) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l1860_186098


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l1860_186063

/-- Calculates the final price of a bicycle after two consecutive sales with given profit percentages -/
def final_price (initial_cost : ℚ) (profit1 : ℚ) (profit2 : ℚ) : ℚ :=
  let price1 := initial_cost * (1 + profit1)
  price1 * (1 + profit2)

/-- Theorem stating that for a bicycle with initial cost 150, sold twice with profits of 20% and 25%, the final price is 225 -/
theorem bicycle_price_calculation :
  final_price 150 (20/100) (25/100) = 225 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l1860_186063


namespace NUMINAMATH_CALUDE_andy_lateness_l1860_186043

structure TravelDelay where
  normalTime : Nat
  redLights : Nat
  redLightDelay : Nat
  constructionDelay : Nat
  detourDelay : Nat
  storeDelay : Nat
  trafficDelay : Nat
  departureTime : Nat
  schoolStartTime : Nat

def calculateLateness (delay : TravelDelay) : Nat :=
  let totalDelay := delay.normalTime +
                    delay.redLights * delay.redLightDelay +
                    delay.constructionDelay +
                    delay.detourDelay +
                    delay.storeDelay +
                    delay.trafficDelay
  let arrivalTime := delay.departureTime + totalDelay
  if arrivalTime > delay.schoolStartTime then
    arrivalTime - delay.schoolStartTime
  else
    0

theorem andy_lateness (delay : TravelDelay)
  (h1 : delay.normalTime = 30)
  (h2 : delay.redLights = 4)
  (h3 : delay.redLightDelay = 3)
  (h4 : delay.constructionDelay = 10)
  (h5 : delay.detourDelay = 7)
  (h6 : delay.storeDelay = 5)
  (h7 : delay.trafficDelay = 15)
  (h8 : delay.departureTime = 435)  -- 7:15 AM in minutes since midnight
  (h9 : delay.schoolStartTime = 480)  -- 8:00 AM in minutes since midnight
  : calculateLateness delay = 34 := by
  sorry


end NUMINAMATH_CALUDE_andy_lateness_l1860_186043


namespace NUMINAMATH_CALUDE_parallel_lines_slope_parallel_line_k_value_l1860_186088

/-- A line through two points is parallel to another line if and only if their slopes are equal -/
theorem parallel_lines_slope (x1 y1 x2 y2 a b c : ℝ) :
  (∀ x y, a * x + b * y = c → y = (-a/b) * x + c/b) →
  (y2 - y1) / (x2 - x1) = -a/b ↔ 
  (∀ x y, y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1) → a * x + b * y = c) :=
sorry

/-- The value of k for which the line through (4, 3) and (k, -5) is parallel to 3x - 2y = 6 -/
theorem parallel_line_k_value : 
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) ∧
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) → k = -4/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_parallel_line_k_value_l1860_186088


namespace NUMINAMATH_CALUDE_sixteen_percent_of_forty_percent_of_93_75_l1860_186081

theorem sixteen_percent_of_forty_percent_of_93_75 : 
  (0.16 * (0.4 * 93.75)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_percent_of_forty_percent_of_93_75_l1860_186081


namespace NUMINAMATH_CALUDE_jim_journey_distance_l1860_186035

/-- The total distance of Jim's journey -/
def total_distance (miles_driven : ℕ) (miles_remaining : ℕ) : ℕ :=
  miles_driven + miles_remaining

/-- Theorem: The total distance of Jim's journey is 1200 miles -/
theorem jim_journey_distance :
  total_distance 768 432 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_distance_l1860_186035


namespace NUMINAMATH_CALUDE_bobby_candy_theorem_l1860_186013

def candy_problem (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - first_eaten - second_eaten

theorem bobby_candy_theorem :
  candy_problem 21 5 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_theorem_l1860_186013


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l1860_186079

theorem definite_integral_sin_plus_one (f : ℝ → ℝ) (h : ∀ x, f x = 1 + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l1860_186079


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1860_186060

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term
  (a : ℝ)  -- Third term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 2*d) = 12)  -- Sum of third and fifth terms is 12
  : a + d = 6 :=  -- Fourth term is 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1860_186060


namespace NUMINAMATH_CALUDE_integer_root_prime_coefficients_l1860_186019

/-- A polynomial of degree 4 with prime coefficients p and q that has an integer root -/
def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 - (p : ℤ) * x^3 + (q : ℤ) = 0

/-- The main theorem stating that if x^4 - px^3 + q = 0 has an integer root,
    and p and q are prime numbers, then p = 3 and q = 2 -/
theorem integer_root_prime_coefficients :
  ∀ p q : ℕ, Prime p → Prime q → has_integer_root p q → p = 3 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_prime_coefficients_l1860_186019


namespace NUMINAMATH_CALUDE_x15x_divisible_by_18_l1860_186038

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def x15x (x : ℕ) : ℕ := x * 1000 + 100 + 50 + x

theorem x15x_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_four_digit (x15x x) ∧ (x15x x) % 18 = 0 ∧ x = 6 := by
sorry

end NUMINAMATH_CALUDE_x15x_divisible_by_18_l1860_186038


namespace NUMINAMATH_CALUDE_blocks_and_colors_l1860_186025

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → 
  blocks_per_color = 7 → 
  total_blocks = blocks_per_color * colors_used → 
  colors_used = 7 := by
sorry

end NUMINAMATH_CALUDE_blocks_and_colors_l1860_186025


namespace NUMINAMATH_CALUDE_cube_surface_area_l1860_186076

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1860_186076


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1860_186027

/-- A rectangle with perimeter 40 meters has a maximum area of 100 square meters. -/
theorem rectangle_max_area :
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ 2 * (w + h) = 40 ∧
  (∀ (w' h' : ℝ), w' > 0 → h' > 0 → 2 * (w' + h') = 40 → w' * h' ≤ w * h) ∧
  w * h = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1860_186027


namespace NUMINAMATH_CALUDE_sales_count_l1860_186062

theorem sales_count (big_sale_commission : ℝ) (average_increase : ℝ) (new_average : ℝ) :
  big_sale_commission = 1300 →
  average_increase = 150 →
  new_average = 400 →
  ∃ (n : ℕ), (n : ℝ) * (new_average - average_increase) + big_sale_commission = new_average * ((n : ℝ) + 1) ∧
              n + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sales_count_l1860_186062
