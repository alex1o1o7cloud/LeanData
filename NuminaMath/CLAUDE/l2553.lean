import Mathlib

namespace apple_pyramid_count_l2553_255323

/-- Represents the number of apples in a layer of the pyramid -/
def layer_count (length width : ℕ) : ℕ := length * width

/-- Represents the pyramid-like stack of apples -/
def apple_pyramid : ℕ :=
  let base := layer_count 4 6
  let second := layer_count 3 5
  let third := layer_count 2 4
  let top := layer_count 2 3  -- double row on top
  base + second + third + top

/-- Theorem stating that the apple pyramid contains exactly 53 apples -/
theorem apple_pyramid_count : apple_pyramid = 53 := by
  sorry

end apple_pyramid_count_l2553_255323


namespace two_part_trip_first_part_length_l2553_255315

/-- Proves that in a two-part trip with given conditions, the first part is 30 km long -/
theorem two_part_trip_first_part_length 
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_first_part = 60)
  (h3 : speed_second_part = 30)
  (h4 : average_speed = 40) :
  ∃ (first_part_distance : ℝ),
    first_part_distance = 30 ∧
    first_part_distance / speed_first_part + (total_distance - first_part_distance) / speed_second_part = total_distance / average_speed :=
by sorry

end two_part_trip_first_part_length_l2553_255315


namespace rectangle_area_l2553_255374

/-- The area of a rectangle bounded by y = a, y = -b, x = -c, and x = 2d, 
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c := by
  sorry

#check rectangle_area

end rectangle_area_l2553_255374


namespace equal_money_in_40_days_l2553_255390

/-- The number of days it takes for Taehyung and Minwoo to have the same amount of money -/
def days_to_equal_money (taehyung_initial : ℕ) (minwoo_initial : ℕ) 
  (taehyung_daily : ℕ) (minwoo_daily : ℕ) : ℕ :=
  (taehyung_initial - minwoo_initial) / (minwoo_daily - taehyung_daily)

/-- Theorem stating that it takes 40 days for Taehyung and Minwoo to have the same amount of money -/
theorem equal_money_in_40_days :
  days_to_equal_money 12000 4000 300 500 = 40 := by
  sorry

end equal_money_in_40_days_l2553_255390


namespace circle_area_with_constraints_fountain_base_area_l2553_255341

/-- The area of a circle with specific constraints -/
theorem circle_area_with_constraints (d : ℝ) (r : ℝ) :
  d = 20 →  -- diameter is 20 feet
  r ^ 2 = 10 ^ 2 + 15 ^ 2 →  -- radius squared equals 10^2 + 15^2 (from Pythagorean theorem)
  π * r ^ 2 = 325 * π := by
  sorry

/-- The main theorem proving the area of the circular base -/
theorem fountain_base_area : ∃ (A : ℝ), A = 325 * π := by
  sorry

end circle_area_with_constraints_fountain_base_area_l2553_255341


namespace sum_of_doubles_l2553_255301

theorem sum_of_doubles (a b c d e f : ℚ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 := by
sorry

end sum_of_doubles_l2553_255301


namespace cousins_distribution_eq_52_l2553_255347

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_52 : cousins_distribution = 52 := by sorry

end cousins_distribution_eq_52_l2553_255347


namespace rectangle_formation_count_l2553_255376

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem rectangle_formation_count : 
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 6
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 150 := by
  sorry

end rectangle_formation_count_l2553_255376


namespace test_score_difference_l2553_255307

theorem test_score_difference (score_60 score_75 score_85 score_95 : ℝ)
  (percent_60 percent_75 percent_85 percent_95 : ℝ) :
  score_60 = 60 ∧ 
  score_75 = 75 ∧ 
  score_85 = 85 ∧ 
  score_95 = 95 ∧
  percent_60 = 0.2 ∧
  percent_75 = 0.4 ∧
  percent_85 = 0.25 ∧
  percent_95 = 0.15 ∧
  percent_60 + percent_75 + percent_85 + percent_95 = 1 →
  let mean := percent_60 * score_60 + percent_75 * score_75 + 
              percent_85 * score_85 + percent_95 * score_95
  let median := score_75
  abs (mean - median) = 2.5 := by
sorry

end test_score_difference_l2553_255307


namespace division_equality_l2553_255349

theorem division_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (3 * a * b)) / (b / (3 * a)) = 1 / (b ^ 2) := by
sorry

end division_equality_l2553_255349


namespace doubled_average_l2553_255368

theorem doubled_average (n : ℕ) (initial_avg : ℝ) (h1 : n = 12) (h2 : initial_avg = 36) :
  let total := n * initial_avg
  let new_total := 2 * total
  let new_avg := new_total / n
  new_avg = 72 := by sorry

end doubled_average_l2553_255368


namespace brownies_count_l2553_255387

/-- Given a box that can hold 7 brownies and 49 full boxes of brownies,
    prove that the total number of brownies is 343. -/
theorem brownies_count (brownies_per_box : ℕ) (full_boxes : ℕ) 
  (h1 : brownies_per_box = 7)
  (h2 : full_boxes = 49) : 
  brownies_per_box * full_boxes = 343 := by
  sorry

end brownies_count_l2553_255387


namespace inscribed_circle_probability_l2553_255308

/-- The probability of a point randomly chosen within a right-angled triangle
    with legs 8 and 15 lying inside its inscribed circle is 3π/20. -/
theorem inscribed_circle_probability : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let r : ℝ := (a * b) / (a + b + c)
  let triangle_area : ℝ := (1/2) * a * b
  let circle_area : ℝ := Real.pi * r^2
  (circle_area / triangle_area) = (3 * Real.pi) / 20 := by
sorry

end inscribed_circle_probability_l2553_255308


namespace acrobats_count_correct_l2553_255362

/-- Represents the number of acrobats in the parade. -/
def num_acrobats : ℕ := 4

/-- Represents the number of elephants in the parade. -/
def num_elephants : ℕ := 8

/-- Represents the number of horses in the parade. -/
def num_horses : ℕ := 8

/-- The total number of legs in the parade. -/
def total_legs : ℕ := 72

/-- The total number of heads in the parade. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  num_acrobats * 2 + num_elephants * 4 + num_horses * 4 = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads :=
by sorry

end acrobats_count_correct_l2553_255362


namespace tank_capacity_proof_l2553_255310

/-- The total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 112.5

/-- Theorem stating that the tank capacity is correct given the problem conditions. -/
theorem tank_capacity_proof :
  tank_capacity = 112.5 ∧
  (0.5 * tank_capacity = 0.9 * tank_capacity - 45) :=
by sorry

end tank_capacity_proof_l2553_255310


namespace apartment_cost_splitting_l2553_255399

/-- The number of people splitting the cost of the new apartment -/
def num_people_splitting_cost : ℕ := 3

/-- John's two brothers -/
def num_brothers : ℕ := 2

/-- The total number of people splitting the cost is John plus his brothers -/
theorem apartment_cost_splitting :
  num_people_splitting_cost = 1 + num_brothers := by
  sorry

end apartment_cost_splitting_l2553_255399


namespace cactus_path_problem_l2553_255352

theorem cactus_path_problem (num_plants : ℕ) (camel_steps : ℕ) (kangaroo_jumps : ℕ) (total_distance : ℝ) :
  num_plants = 51 →
  camel_steps = 56 →
  kangaroo_jumps = 14 →
  total_distance = 7920 →
  let num_gaps := num_plants - 1
  let total_camel_steps := num_gaps * camel_steps
  let total_kangaroo_jumps := num_gaps * kangaroo_jumps
  let camel_step_length := total_distance / total_camel_steps
  let kangaroo_jump_length := total_distance / total_kangaroo_jumps
  kangaroo_jump_length - camel_step_length = 8.5 := by
  sorry

end cactus_path_problem_l2553_255352


namespace system_of_equations_result_l2553_255325

theorem system_of_equations_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
  sorry

end system_of_equations_result_l2553_255325


namespace ice_cream_flavors_l2553_255398

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end ice_cream_flavors_l2553_255398


namespace compound_interest_calculation_l2553_255367

/-- Given an annual interest rate and time period, calculates the compound interest
    if the simple interest is known. -/
theorem compound_interest_calculation
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Annual interest rate (as a percentage)
  (T : ℝ) -- Time period in years
  (h1 : R = 20)
  (h2 : T = 2)
  (h3 : P * R * T / 100 = 400) -- Simple interest formula
  : P * (1 + R/100)^T - P = 440 := by
  sorry

#check compound_interest_calculation

end compound_interest_calculation_l2553_255367


namespace vector_problems_l2553_255309

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, -3)

theorem vector_problems :
  -- Part I: Dot product
  (a.1 * b.1 + a.2 * b.2 = -2) ∧
  -- Part II: Parallel vector with given magnitude
  (∃ (c : ℝ × ℝ), (c.2 / c.1 = a.2 / a.1) ∧ 
                  (c.1^2 + c.2^2 = 20) ∧ 
                  ((c = (-2, -4)) ∨ (c = (2, 4)))) ∧
  -- Part III: Perpendicular vectors condition
  (∃ (k : ℝ), ((b.1 + k * a.1)^2 + (b.2 + k * a.2)^2 = 
               (b.1 - k * a.1)^2 + (b.2 - k * a.2)^2) ∧
              (k^2 = 5)) :=
by sorry


end vector_problems_l2553_255309


namespace system_equations_properties_l2553_255302

theorem system_equations_properties (a : ℝ) (x y : ℝ) 
  (h1 : x + 3*y = 4 - a) 
  (h2 : x - y = 3*a) 
  (h3 : -3 ≤ a ∧ a ≤ 1) :
  (a = -2 → x = -y) ∧ 
  (a = 1 → x + y = 3) ∧ 
  (x ≤ 1 → 1 ≤ y ∧ y ≤ 4) := by
sorry

end system_equations_properties_l2553_255302


namespace reciprocal_of_negative_three_l2553_255336

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_three :
  reciprocal (-3) = -1/3 := by
  sorry

end reciprocal_of_negative_three_l2553_255336


namespace distribution_theorem_l2553_255380

/-- The number of ways to distribute 4 men and 5 women into three groups of three people each,
    with at least one man and one woman in each group. -/
def distribution_ways : ℕ := 360

/-- The number of men -/
def num_men : ℕ := 4

/-- The number of women -/
def num_women : ℕ := 5

/-- The size of each group -/
def group_size : ℕ := 3

/-- The total number of groups -/
def num_groups : ℕ := 3

theorem distribution_theorem :
  (∀ (group : Fin num_groups), ∃ (m w : ℕ), m ≥ 1 ∧ w ≥ 1 ∧ m + w = group_size) →
  (num_men + num_women = num_groups * group_size) →
  distribution_ways = 360 := by
  sorry

end distribution_theorem_l2553_255380


namespace geometric_sequence_ratio_l2553_255366

/-- A geometric sequence with a_1 = 1 and a_3 = 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 2 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 5 + a 10) / (a 1 + a 6) = 4 := by
  sorry

end geometric_sequence_ratio_l2553_255366


namespace solution_difference_l2553_255313

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

/-- Definition of p and q as solutions to the equation -/
def p_and_q_are_solutions (p q : ℝ) : Prop :=
  equation p ∧ equation q ∧ p ≠ q

theorem solution_difference (p q : ℝ) :
  p_and_q_are_solutions p q → p > q → p - q = 10 := by
  sorry

end solution_difference_l2553_255313


namespace expression_simplification_and_value_l2553_255350

theorem expression_simplification_and_value (a : ℤ) 
  (h1 : 0 < a) (h2 : a < Int.floor (Real.sqrt 5)) : 
  (((a^2 - 1) / (a^2 + 2*a)) / ((a - 1) / a) - a / (a + 2) : ℚ) = 1 / (a + 2) ∧
  (((1^2 - 1) / (1^2 + 2*1)) / ((1 - 1) / 1) - 1 / (1 + 2) : ℚ) = 1 / 3 := by
  sorry

end expression_simplification_and_value_l2553_255350


namespace apartment_complexes_count_l2553_255379

/-- The maximum number of apartment complexes that can be built on a rectangular land -/
def max_apartment_complexes (land_width land_length complex_side : ℕ) : ℕ :=
  (land_width / complex_side) * (land_length / complex_side)

/-- Theorem stating the maximum number of apartment complexes that can be built -/
theorem apartment_complexes_count :
  max_apartment_complexes 262 185 18 = 140 := by
  sorry

end apartment_complexes_count_l2553_255379


namespace parallelogram_cut_slope_sum_l2553_255397

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Defines the specific parallelogram from the problem -/
def problemParallelogram : Parallelogram :=
  { v1 := { x := 15, y := 70 }
  , v2 := { x := 15, y := 210 }
  , v3 := { x := 45, y := 280 }
  , v4 := { x := 45, y := 140 }
  }

/-- A line through the origin with slope m/n -/
structure Line where
  m : ℕ
  n : ℕ
  coprime : Nat.Coprime m n

/-- Predicate to check if a line cuts the parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (l : Line) (p : Parallelogram) : Prop :=
  sorry -- Definition omitted for brevity

theorem parallelogram_cut_slope_sum :
  ∃ (l : Line), cutsIntoCongruentPolygons l problemParallelogram ∧ l.m + l.n = 41 :=
sorry

end parallelogram_cut_slope_sum_l2553_255397


namespace cube_root_problem_l2553_255359

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end cube_root_problem_l2553_255359


namespace complex_division_simplification_l2553_255372

theorem complex_division_simplification (z : ℂ) : 
  z = (4 + 3*I) / (1 + 2*I) → z = 2 - I :=
by sorry

end complex_division_simplification_l2553_255372


namespace sqrt_difference_squared_l2553_255339

theorem sqrt_difference_squared : (Real.sqrt 25 - Real.sqrt 9)^2 = 4 := by
  sorry

end sqrt_difference_squared_l2553_255339


namespace sum_of_roots_quadratic_l2553_255319

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → x₁ + x₂ = 2 := by
sorry

end sum_of_roots_quadratic_l2553_255319


namespace opponent_total_score_l2553_255373

def team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def games_lost_by_one (scores : List ℕ) : ℕ := 6

def score_ratio_in_other_games : ℕ := 3

theorem opponent_total_score :
  let opponent_scores := team_scores.map (λ score =>
    if score % 2 = 1 then score + 1
    else score / score_ratio_in_other_games)
  opponent_scores.sum = 60 := by sorry

end opponent_total_score_l2553_255373


namespace luna_makes_seven_per_hour_l2553_255378

/-- The number of milkshakes Augustus can make per hour -/
def augustus_rate : ℕ := 3

/-- The number of hours Augustus and Luna work -/
def work_hours : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := 80

/-- The number of milkshakes Luna can make per hour -/
def luna_rate : ℕ := (total_milkshakes - augustus_rate * work_hours) / work_hours

theorem luna_makes_seven_per_hour : luna_rate = 7 := by
  sorry

end luna_makes_seven_per_hour_l2553_255378


namespace equation_solution_l2553_255311

theorem equation_solution :
  ∃ x : ℝ, (3 / (x + 2) - 1 / x = 0) ∧ x = 1 :=
by
  sorry

end equation_solution_l2553_255311


namespace eight_mile_taxi_cost_l2553_255393

/-- Calculates the cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℚ) (cost_per_mile : ℚ) (distance : ℚ) : ℚ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem: The cost of an 8-mile taxi ride with a $2.00 fixed cost and $0.30 per mile is $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2 (3/10) 8 = 44/10 := by
  sorry

end eight_mile_taxi_cost_l2553_255393


namespace min_markers_to_sell_is_1200_l2553_255383

/-- Represents the number of markers bought -/
def markers_bought : ℕ := 2000

/-- Represents the cost price of each marker in cents -/
def cost_price : ℕ := 20

/-- Represents the selling price of each marker in cents -/
def selling_price : ℕ := 50

/-- Represents the minimum profit desired in cents -/
def min_profit : ℕ := 20000

/-- Calculates the minimum number of markers that must be sold to achieve the desired profit -/
def min_markers_to_sell : ℕ :=
  (markers_bought * cost_price + min_profit) / (selling_price - cost_price)

/-- Theorem stating that the minimum number of markers to sell is 1200 -/
theorem min_markers_to_sell_is_1200 : min_markers_to_sell = 1200 := by
  sorry

#eval min_markers_to_sell

end min_markers_to_sell_is_1200_l2553_255383


namespace correct_calculation_l2553_255321

theorem correct_calculation (x : ℤ) (h : x - 59 = 43) : x - 46 = 56 := by
  sorry

end correct_calculation_l2553_255321


namespace like_terms_imply_exponents_l2553_255328

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, m1 x y ≠ 0 ∧ m2 x y ≠ 0 → (x = x ∧ y = y)

/-- The first monomial 2x^3y^4 -/
def m1 (x y : ℕ) : ℚ := 2 * (x^3 * y^4)

/-- The second monomial -2x^ay^(2b) -/
def m2 (a b x y : ℕ) : ℚ := -2 * (x^a * y^(2*b))

theorem like_terms_imply_exponents (a b : ℕ) :
  are_like_terms m1 (m2 a b) → a = 3 ∧ b = 2 := by
  sorry

end like_terms_imply_exponents_l2553_255328


namespace smallest_n_for_geometric_sums_l2553_255343

def is_geometric_sum (x : ℕ) : Prop :=
  ∃ (a r : ℕ), r > 1 ∧ x = a + a*r + a*r^2

theorem smallest_n_for_geometric_sums : 
  (∀ n : ℕ, n < 6 → ¬(is_geometric_sum (7*n + 1) ∧ is_geometric_sum (8*n + 1))) ∧
  (is_geometric_sum (7*6 + 1) ∧ is_geometric_sum (8*6 + 1)) :=
sorry

end smallest_n_for_geometric_sums_l2553_255343


namespace partial_fraction_decomposition_l2553_255345

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 2/3 ∧ Q = 8/9 ∧ R = -5/9) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) =
      P / (x - 1) + Q / (x - 4) + R / (x + 2) :=
by
  sorry

end partial_fraction_decomposition_l2553_255345


namespace remainder_theorem_l2553_255384

theorem remainder_theorem (A B : ℕ) (h1 : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_theorem_l2553_255384


namespace three_number_difference_l2553_255340

theorem three_number_difference (x y z : ℝ) : 
  x = 2 * y ∧ x = 3 * z ∧ (x + y + z) / 3 = 88 → x - z = 96 := by
  sorry

end three_number_difference_l2553_255340


namespace coin_arrangements_l2553_255356

/-- Represents the number of gold coins -/
def num_gold_coins : Nat := 4

/-- Represents the number of silver coins -/
def num_silver_coins : Nat := 4

/-- Represents the total number of coins -/
def total_coins : Nat := num_gold_coins + num_silver_coins

/-- Calculates the number of ways to arrange gold and silver coins -/
def color_arrangements : Nat := Nat.choose total_coins num_gold_coins

/-- Calculates the number of valid orientations (face up or down) -/
def orientation_arrangements : Nat := total_coins + 1

/-- Theorem: The number of distinguishable arrangements of 8 coins (4 gold and 4 silver)
    stacked so that no two adjacent coins are face to face is 630 -/
theorem coin_arrangements :
  color_arrangements * orientation_arrangements = 630 := by
  sorry

end coin_arrangements_l2553_255356


namespace school_supplies_ratio_l2553_255363

/-- Proves the ratio of school supplies spending to remaining money after textbooks is 1:4 --/
theorem school_supplies_ratio (total : ℕ) (remaining : ℕ) : 
  total = 960 →
  remaining = 360 →
  (total - total / 2 - remaining) / (total / 2) = 1 / 4 := by
  sorry

end school_supplies_ratio_l2553_255363


namespace min_dot_product_l2553_255389

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus
def line_through_focus (x y : ℝ) : Prop := y = x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x - 1

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Statement of the theorem
theorem min_dot_product :
  ∃ (M N : ℝ × ℝ),
    parabola M.1 M.2 ∧
    parabola N.1 N.2 ∧
    line_through_focus M.1 M.2 ∧
    line_through_focus N.1 N.2 ∧
    (∀ (P : ℝ × ℝ), tangent_line P.1 P.2 →
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) ≥ -14) ∧
    (∃ (P : ℝ × ℝ), tangent_line P.1 P.2 ∧
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) = -14) :=
by
  sorry

end min_dot_product_l2553_255389


namespace derivative_f_at_zero_l2553_255335

-- Define the function f(x) = (2x + 1)³
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end derivative_f_at_zero_l2553_255335


namespace zero_point_location_l2553_255306

-- Define the function f
variable {f : ℝ → ℝ}

-- Define the property of having exactly one zero point in an interval
def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_point_location (h1 : has_unique_zero f 0 16)
                            (h2 : has_unique_zero f 0 8)
                            (h3 : has_unique_zero f 0 4)
                            (h4 : has_unique_zero f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
sorry

end zero_point_location_l2553_255306


namespace expression_simplification_l2553_255382

theorem expression_simplification :
  Real.sqrt 5 * (5 ^ (1/2 : ℝ)) + 20 / 4 * 3 - 9 ^ (3/2 : ℝ) = -7 := by
  sorry

end expression_simplification_l2553_255382


namespace f_properties_l2553_255365

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin x * Real.cos x + (1 + Real.tan x ^ 2) * Real.cos x ^ 2

theorem f_properties : 
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), f x ≤ 3/2) ∧ 
  (∃ (x : ℝ), f x = 3/2) := by
  sorry

end f_properties_l2553_255365


namespace one_fourth_of_eight_x_plus_two_l2553_255395

theorem one_fourth_of_eight_x_plus_two (x : ℝ) : (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 := by
  sorry

end one_fourth_of_eight_x_plus_two_l2553_255395


namespace symmetric_points_difference_l2553_255370

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_wrt_origin a 1 5 b → a - b = -4 := by
  sorry

end symmetric_points_difference_l2553_255370


namespace chessboard_coverage_l2553_255351

/-- Represents a subregion of a chessboard -/
structure Subregion where
  rows : Finset Nat
  cols : Finset Nat

/-- The chessboard and its subregions -/
structure Chessboard where
  n : Nat
  subregions : Finset Subregion

/-- The semi-perimeter of a subregion -/
def semiPerimeter (s : Subregion) : Nat :=
  s.rows.card + s.cols.card

/-- Whether a subregion covers a cell -/
def covers (s : Subregion) (i j : Nat) : Prop :=
  i ∈ s.rows ∧ j ∈ s.cols

/-- The main diagonal of the chessboard -/
def mainDiagonal (n : Nat) : Set (Nat × Nat) :=
  {p | p.1 = p.2 ∧ p.1 < n}

/-- The theorem to be proved -/
theorem chessboard_coverage (cb : Chessboard) : 
  (∀ s ∈ cb.subregions, semiPerimeter s ≥ cb.n) →
  (∀ p ∈ mainDiagonal cb.n, ∃ s ∈ cb.subregions, covers s p.1 p.2) →
  (cb.subregions.sum (λ s => (s.rows.card * s.cols.card)) ≥ cb.n^2 / 2) :=
sorry

end chessboard_coverage_l2553_255351


namespace reflection_of_M_across_x_axis_l2553_255318

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M with coordinates (1, 2) -/
def M : ℝ × ℝ := (1, 2)

theorem reflection_of_M_across_x_axis :
  reflect_x M = (1, -2) := by sorry

end reflection_of_M_across_x_axis_l2553_255318


namespace system_solution_l2553_255348

theorem system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry

end system_solution_l2553_255348


namespace proportional_segment_length_l2553_255371

/-- Triangle ABC with sides a, b, c, and an interior point P creating parallel segments of length d -/
structure ProportionalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The condition that the parallel segments split the sides proportionally -/
def is_proportional (t : ProportionalTriangle) : Prop :=
  t.d / t.c * t.b + t.d / t.a * t.b = t.b

/-- The theorem stating that for the given triangle, the proportional segments have length 28.25 -/
theorem proportional_segment_length 
  (t : ProportionalTriangle) 
  (h1 : t.a = 500) 
  (h2 : t.b = 550) 
  (h3 : t.c = 650) 
  (h4 : is_proportional t) : 
  t.d = 28.25 := by
  sorry

end proportional_segment_length_l2553_255371


namespace total_cost_is_53_l2553_255334

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 10
def discount_threshold : ℕ := 15
def discount_amount : ℕ := 5

def total_items : ℕ := sandwich_quantity + soda_quantity

def total_cost : ℕ :=
  sandwich_cost * sandwich_quantity + soda_cost * soda_quantity - 
  if total_items > discount_threshold then discount_amount else 0

theorem total_cost_is_53 : total_cost = 53 := by
  sorry

end total_cost_is_53_l2553_255334


namespace unit_conversions_l2553_255330

theorem unit_conversions :
  (∀ (cm : ℝ), cm * (1 / 100) = cm / 100) ∧
  (∀ (hectares : ℝ), hectares * 10000 = hectares * 10000) ∧
  (∀ (kg g : ℝ), kg + g / 1000 = kg + g * (1 / 1000)) ∧
  (∀ (m cm : ℝ), m + cm / 100 = m + cm * (1 / 100)) →
  (120 : ℝ) * (1 / 100) = 1.2 ∧
  (0.3 : ℝ) * 10000 = 3000 ∧
  10 + 10 / 1000 = 10.01 ∧
  1 + 3 / 100 = 1.03 := by
  sorry

end unit_conversions_l2553_255330


namespace original_room_length_l2553_255391

theorem original_room_length :
  ∀ (x : ℝ),
  (4 * ((x + 2) * 20) + 2 * ((x + 2) * 20) = 1800) →
  x = 13 :=
by
  sorry

end original_room_length_l2553_255391


namespace smallest_n_with_properties_l2553_255377

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def contains_distinct_digits (n : ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ contains_digit n d₁ ∧ contains_digit n d₂

theorem smallest_n_with_properties : 
  (∀ m : ℕ, m < 128 → ¬(is_terminating_decimal m ∧ contains_digit m 9 ∧ contains_distinct_digits m)) ∧
  (is_terminating_decimal 128 ∧ contains_digit 128 9 ∧ contains_distinct_digits 128) :=
sorry

end smallest_n_with_properties_l2553_255377


namespace base_seven_addition_l2553_255300

/-- Given an addition problem in base 7: 5XY₇ + 62₇ = 64X₇, prove that X + Y = 8 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (6 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 8 := by
sorry

end base_seven_addition_l2553_255300


namespace m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l2553_255346

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m : ℝ) : Prop :=
  (m - 1) / m = (m - 1) / 2

/-- m = 2 is a sufficient condition for the lines to be parallel -/
theorem m_2_sufficient : are_parallel 2 := by sorry

/-- m = 2 is not a necessary condition for the lines to be parallel -/
theorem m_2_not_necessary : ∃ m : ℝ, m ≠ 2 ∧ are_parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem m_2_sufficient_but_not_necessary : 
  (are_parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ are_parallel m) := by sorry

end m_2_sufficient_m_2_not_necessary_m_2_sufficient_but_not_necessary_l2553_255346


namespace log_ratio_theorem_l2553_255329

theorem log_ratio_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Real.log (a^3)) / (Real.log (a^2)) = 3/2 := by
  sorry

end log_ratio_theorem_l2553_255329


namespace grid_and_unshaded_area_sum_l2553_255361

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents an unshaded square -/
structure UnshadedSquare :=
  (side_length : ℝ)

/-- Calculates the total area of a grid -/
def grid_area (g : Grid) : ℝ :=
  (g.size * g.square_size) ^ 2

/-- Calculates the area of an unshaded square -/
def unshaded_square_area (u : UnshadedSquare) : ℝ :=
  u.side_length ^ 2

/-- The main theorem to prove -/
theorem grid_and_unshaded_area_sum :
  let g : Grid := { size := 6, square_size := 3 }
  let u : UnshadedSquare := { side_length := 1.5 }
  let num_unshaded : ℕ := 5
  grid_area g + (num_unshaded * unshaded_square_area u) = 335.25 := by
  sorry


end grid_and_unshaded_area_sum_l2553_255361


namespace expression_evaluation_l2553_255332

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end expression_evaluation_l2553_255332


namespace computer_sales_total_l2553_255303

theorem computer_sales_total (total : ℕ) : 
  (total / 2 : ℕ) + (total / 3 : ℕ) + 12 = total → total = 72 := by
  sorry

end computer_sales_total_l2553_255303


namespace platform_length_l2553_255314

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) :
  train_length = 360 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 43.2 →
  (train_speed_kmh * (1000 / 3600) * time_to_pass) - train_length = 180 := by
  sorry

end platform_length_l2553_255314


namespace min_cost_rose_garden_l2553_255320

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : Float
  tulip : Float
  orchid : Float
  rose : Float
  peony : Float

/-- Represents the dimensions of each region in the flower bed -/
structure FlowerBedRegions where
  bottom_left : Nat × Nat
  top_left : Nat × Nat
  bottom_right : Nat × Nat
  middle_right : Nat × Nat
  top_right : Nat × Nat

/-- Calculates the minimum cost for Rose's garden -/
def calculateMinCost (costs : FlowerCost) (regions : FlowerBedRegions) : Float :=
  sorry

/-- Theorem stating that the minimum cost for Rose's garden is $173.75 -/
theorem min_cost_rose_garden (costs : FlowerCost) (regions : FlowerBedRegions) :
  costs.sunflower = 0.75 ∧
  costs.tulip = 1.25 ∧
  costs.orchid = 1.75 ∧
  costs.rose = 2 ∧
  costs.peony = 2.5 ∧
  regions.bottom_left = (7, 2) ∧
  regions.top_left = (5, 5) ∧
  regions.bottom_right = (6, 4) ∧
  regions.middle_right = (8, 3) ∧
  regions.top_right = (8, 3) →
  calculateMinCost costs regions = 173.75 :=
by sorry

end min_cost_rose_garden_l2553_255320


namespace system_of_inequalities_solution_l2553_255381

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) ↔ x < -2 := by
  sorry

end system_of_inequalities_solution_l2553_255381


namespace simplified_fraction_ratio_l2553_255337

theorem simplified_fraction_ratio (m : ℝ) (c d : ℤ) :
  (∃ (k : ℝ), (5 * m + 15) / 5 = k ∧ k = c * m + d) →
  d / c = 3 :=
by sorry

end simplified_fraction_ratio_l2553_255337


namespace line_condition_perpendicular_condition_equal_intercepts_condition_l2553_255305

/-- The equation of a line with parameter m -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 5 - 2*m = 0

/-- The condition for the equation to represent a line -/
theorem line_condition (m : ℝ) : 
  (∃ x y, line_equation m x y) ↔ m ≠ -1 :=
sorry

/-- The condition for the line to be perpendicular to the x-axis -/
theorem perpendicular_condition (m : ℝ) :
  (m^2 - 2*m - 3 = 0 ∧ 2*m^2 + m - 1 ≠ 0) ↔ m = 1/2 :=
sorry

/-- The condition for the line to have equal intercepts on both axes -/
theorem equal_intercepts_condition (m : ℝ) :
  (∃ a ≠ 0, line_equation m a 0 ∧ line_equation m 0 (-a)) ↔ m = -2 :=
sorry

end line_condition_perpendicular_condition_equal_intercepts_condition_l2553_255305


namespace job_completion_multiple_l2553_255354

/-- Given workers A and B, and their work rates, calculate the multiple of the original job they complete when working together for a given number of days. -/
theorem job_completion_multiple 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days : ℝ) 
  (h1 : days_A > 0) 
  (h2 : days_B > 0) 
  (h3 : work_days > 0) : 
  work_days * (1 / days_A + 1 / days_B) = 4 := by
  sorry

#check job_completion_multiple 45 30 72

end job_completion_multiple_l2553_255354


namespace felicity_gasoline_usage_l2553_255317

/-- Represents the fuel consumption and distance data for a road trip. -/
structure RoadTripData where
  felicity_mpg : ℝ
  adhira_mpg : ℝ
  benjamin_ethanol_mpg : ℝ
  benjamin_biodiesel_mpg : ℝ
  total_distance : ℝ
  adhira_felicity_diff : ℝ
  felicity_benjamin_diff : ℝ
  ethanol_ratio : ℝ
  biodiesel_ratio : ℝ
  felicity_adhira_fuel_ratio : ℝ
  benjamin_adhira_fuel_diff : ℝ

/-- Calculates the amount of gasoline used by Felicity given the road trip data. -/
def calculate_felicity_gasoline (data : RoadTripData) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Felicity used 56 gallons of gasoline on her trip. -/
theorem felicity_gasoline_usage (data : RoadTripData) 
    (h1 : data.felicity_mpg = 35)
    (h2 : data.adhira_mpg = 25)
    (h3 : data.benjamin_ethanol_mpg = 30)
    (h4 : data.benjamin_biodiesel_mpg = 20)
    (h5 : data.total_distance = 1750)
    (h6 : data.adhira_felicity_diff = 150)
    (h7 : data.felicity_benjamin_diff = 50)
    (h8 : data.ethanol_ratio = 0.35)
    (h9 : data.biodiesel_ratio = 0.65)
    (h10 : data.felicity_adhira_fuel_ratio = 2)
    (h11 : data.benjamin_adhira_fuel_diff = 5) :
  calculate_felicity_gasoline data = 56 := by
  sorry

end felicity_gasoline_usage_l2553_255317


namespace job_selection_probability_l2553_255360

theorem job_selection_probability (carol_prob bernie_prob : ℚ) 
  (h_carol : carol_prob = 4/5)
  (h_bernie : bernie_prob = 3/5) : 
  carol_prob * bernie_prob = 12/25 := by
sorry

end job_selection_probability_l2553_255360


namespace road_trip_driving_hours_l2553_255358

/-- Proves that in a 3-day road trip where one person drives 6 hours each day
    and the total driving time is 42 hours, the other person drives 8 hours each day. -/
theorem road_trip_driving_hours (total_days : ℕ) (krista_hours_per_day : ℕ) (total_hours : ℕ)
    (h1 : total_days = 3)
    (h2 : krista_hours_per_day = 6)
    (h3 : total_hours = 42) :
    (total_hours - krista_hours_per_day * total_days) / total_days = 8 := by
  sorry

end road_trip_driving_hours_l2553_255358


namespace branches_per_tree_is_100_l2553_255316

/-- Represents the farm with trees and their branches -/
structure Farm where
  num_trees : ℕ
  leaves_per_subbranch : ℕ
  subbranches_per_branch : ℕ
  total_leaves : ℕ

/-- Calculates the number of branches per tree on the farm -/
def branches_per_tree (f : Farm) : ℕ :=
  f.total_leaves / (f.num_trees * f.subbranches_per_branch * f.leaves_per_subbranch)

/-- Theorem stating that the number of branches per tree is 100 -/
theorem branches_per_tree_is_100 (f : Farm) 
  (h1 : f.num_trees = 4)
  (h2 : f.leaves_per_subbranch = 60)
  (h3 : f.subbranches_per_branch = 40)
  (h4 : f.total_leaves = 96000) : 
  branches_per_tree f = 100 := by
  sorry

#eval branches_per_tree { num_trees := 4, leaves_per_subbranch := 60, subbranches_per_branch := 40, total_leaves := 96000 }

end branches_per_tree_is_100_l2553_255316


namespace angle_B_magnitude_triangle_area_l2553_255344

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a + 2 * t.c = 2 * t.b * Real.cos t.A

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3

def satisfiesCondition3 (t : Triangle) : Prop :=
  t.a + t.c = 4

-- Theorem 1
theorem angle_B_magnitude (t : Triangle) (h : satisfiesCondition1 t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfiesCondition1 t) 
  (h2 : satisfiesCondition2 t) 
  (h3 : satisfiesCondition3 t) : 
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 := by
  sorry

end angle_B_magnitude_triangle_area_l2553_255344


namespace page_lines_increase_l2553_255312

theorem page_lines_increase (L : ℕ) (h : L + 60 = 240) : 
  (60 : ℝ) / L = 1 / 3 := by
  sorry

end page_lines_increase_l2553_255312


namespace committee_selection_l2553_255369

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 35) : Nat.choose n 4 = 35 := by
  sorry

end committee_selection_l2553_255369


namespace gcd_299_621_l2553_255331

theorem gcd_299_621 : Nat.gcd 299 621 = 23 := by
  sorry

end gcd_299_621_l2553_255331


namespace fathers_age_l2553_255326

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 10 = (1 / 2) * (father_age + 10) → 
  father_age = 50 := by
sorry

end fathers_age_l2553_255326


namespace test_scores_l2553_255364

/-- Given a test with 50 questions, each worth 2 marks, prove that the total combined score
    for three students (Meghan, Jose, and Alisson) is 210 marks, given the following conditions:
    - Meghan scored 20 marks less than Jose
    - Jose scored 40 more marks than Alisson
    - Jose got 5 questions wrong -/
theorem test_scores (total_questions : Nat) (marks_per_question : Nat)
    (meghan_jose_diff : Nat) (jose_alisson_diff : Nat) (jose_wrong : Nat) :
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_jose_diff = 20 →
  jose_alisson_diff = 40 →
  jose_wrong = 5 →
  ∃ (meghan_score jose_score alisson_score : Nat),
    meghan_score + jose_score + alisson_score = 210 :=
by sorry


end test_scores_l2553_255364


namespace gcd_437_323_l2553_255355

theorem gcd_437_323 : Nat.gcd 437 323 = 19 := by
  sorry

end gcd_437_323_l2553_255355


namespace purchase_cost_l2553_255304

/-- The cost of a single can of soda in dollars -/
def soda_cost : ℝ := 1

/-- The number of soda cans purchased -/
def num_sodas : ℕ := 3

/-- The number of soups purchased -/
def num_soups : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 1

/-- The cost of a single soup in dollars -/
def soup_cost : ℝ := num_sodas * soda_cost

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := num_sodas * soda_cost + num_soups * soup_cost + num_sandwiches * sandwich_cost

theorem purchase_cost : total_cost = 18 := by
  sorry

end purchase_cost_l2553_255304


namespace isosceles_triangle_perimeter_l2553_255375

/-- An isosceles triangle with sides a, b, and c, where b = c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, t.a = 3 → t.b = 6 → perimeter t = 15 := by
  sorry

end isosceles_triangle_perimeter_l2553_255375


namespace binary_147_ones_zeros_difference_l2553_255386

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

theorem binary_147_ones_zeros_difference :
  let bin_147 := binary_representation 147
  let ones := count_ones bin_147
  let zeros := count_zeros bin_147
  ones - zeros = 0 := by sorry

end binary_147_ones_zeros_difference_l2553_255386


namespace vowels_on_board_l2553_255353

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end vowels_on_board_l2553_255353


namespace conditional_probability_A_given_B_l2553_255357

def group_A : List Nat := [76, 90, 84, 86, 81, 87, 86, 82, 85, 83]
def group_B : List Nat := [82, 84, 85, 89, 79, 80, 91, 89, 79, 74]

def total_students : Nat := group_A.length + group_B.length

def students_A_above_85 : Nat := (group_A.filter (λ x => x ≥ 85)).length
def students_B_above_85 : Nat := (group_B.filter (λ x => x ≥ 85)).length
def total_above_85 : Nat := students_A_above_85 + students_B_above_85

def P_B : Rat := total_above_85 / total_students
def P_AB : Rat := students_A_above_85 / total_students

theorem conditional_probability_A_given_B :
  P_AB / P_B = 5 / 9 := by sorry

end conditional_probability_A_given_B_l2553_255357


namespace student_count_last_year_l2553_255394

theorem student_count_last_year 
  (increase_rate : Real) 
  (current_count : Nat) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ∃ (last_year_count : Nat), 
    (last_year_count : Real) * (1 + increase_rate) = current_count ∧ 
    last_year_count = 800 := by
  sorry

end student_count_last_year_l2553_255394


namespace score_statistics_l2553_255338

def scores : List ℕ := [42, 43, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 47, 47]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem score_statistics :
  mode scores = 46 ∧ median scores = 45 := by sorry

end score_statistics_l2553_255338


namespace mrs_hilt_initial_money_l2553_255396

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_initial_money :
  ∀ (initial_money toy_truck_cost pencil_case_cost money_left : ℕ),
  toy_truck_cost = 3 →
  pencil_case_cost = 2 →
  money_left = 5 →
  initial_money = toy_truck_cost + pencil_case_cost + money_left →
  initial_money = 10 :=
by
  sorry


end mrs_hilt_initial_money_l2553_255396


namespace negation_of_existence_circle_negation_l2553_255333

theorem negation_of_existence (P : ℝ × ℝ → Prop) :
  (¬ ∃ p, P p) ↔ (∀ p, ¬ P p) := by sorry

theorem circle_negation :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) := by sorry

end negation_of_existence_circle_negation_l2553_255333


namespace kelly_apple_count_l2553_255388

/-- 
Theorem: Given Kelly's initial apple count and the number of additional apples picked,
prove that the total number of apples is the sum of these two quantities.
-/
theorem kelly_apple_count (initial_apples additional_apples : ℕ) :
  initial_apples = 56 →
  additional_apples = 49 →
  initial_apples + additional_apples = 105 := by
  sorry

end kelly_apple_count_l2553_255388


namespace car_rental_cost_per_mile_l2553_255385

theorem car_rental_cost_per_mile 
  (base_cost : ℝ) 
  (total_miles : ℝ) 
  (total_cost : ℝ) 
  (h1 : base_cost = 150)
  (h2 : total_miles = 1364)
  (h3 : total_cost = 832) :
  (total_cost - base_cost) / total_miles = 0.50 := by
sorry

end car_rental_cost_per_mile_l2553_255385


namespace prob_different_topics_correct_l2553_255342

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5 / 6

/-- Theorem stating that the probability of two students selecting different topics
    from num_topics options is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end prob_different_topics_correct_l2553_255342


namespace museum_artifact_count_l2553_255392

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_painting_count : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing --/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_painting_count + (m.painting_wings - 1) * m.small_paintings_per_wing
  let total_artifacts := ((m.artifact_multiplier * total_paintings) / 8) * 8
  let artifact_wings := m.total_wings - m.painting_wings
  total_artifacts / artifact_wings

/-- Theorem: In the given museum setup, each artifact wing contains 34 artifacts --/
theorem museum_artifact_count (m : Museum) 
  (h1 : m.total_wings = 12)
  (h2 : m.painting_wings = 4)
  (h3 : m.large_painting_count = 1)
  (h4 : m.small_paintings_per_wing = 15)
  (h5 : m.artifact_multiplier = 6) :
  artifacts_per_wing m = 34 := by
  sorry

end museum_artifact_count_l2553_255392


namespace fourth_student_id_l2553_255327

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_id : ℕ

/-- Checks if a given ID is in the systematic sample. -/
def SystematicSample.contains (s : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.first_id + k * s.interval

/-- The theorem to be proved. -/
theorem fourth_student_id
  (s : SystematicSample)
  (h_class_size : s.class_size = 52)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_29 : s.contains 29)
  (h_contains_42 : s.contains 42) :
  s.contains 16 :=
sorry

end fourth_student_id_l2553_255327


namespace inequality_system_product_l2553_255322

theorem inequality_system_product (x y : ℤ) : 
  (x^3 + y^2 - 3*y + 1 < 0 ∧ 3*x^3 - y^2 + 3*y > 0) → 
  (∃ (y1 y2 : ℤ), y1 ≠ y2 ∧ 
    (x^3 + y1^2 - 3*y1 + 1 < 0 ∧ 3*x^3 - y1^2 + 3*y1 > 0) ∧
    (x^3 + y2^2 - 3*y2 + 1 < 0 ∧ 3*x^3 - y2^2 + 3*y2 > 0) ∧
    y1 * y2 = 2) :=
by sorry

end inequality_system_product_l2553_255322


namespace inequality_implication_l2553_255324

theorem inequality_implication (x y : ℝ) : 5 * x > -5 * y → x + y > 0 := by
  sorry

end inequality_implication_l2553_255324
