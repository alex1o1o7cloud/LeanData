import Mathlib

namespace drama_club_subjects_l3299_329931

/-- Given a group of students in a drama club, prove the number of students
    taking neither mathematics nor physics. -/
theorem drama_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
    (h1 : total = 60)
    (h2 : math = 40)
    (h3 : physics = 35)
    (h4 : both = 25) :
    total - (math + physics - both) = 10 := by
  sorry

end drama_club_subjects_l3299_329931


namespace equation_solution_l3299_329935

theorem equation_solution : ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 3*x + 5) / (x + 3) = x + 6 := by
  sorry

end equation_solution_l3299_329935


namespace max_value_of_function_max_value_achievable_l3299_329968

theorem max_value_of_function (x : ℝ) : 
  x^6 / (x^8 + 2*x^7 - 4*x^6 + 8*x^5 + 16*x^4) ≤ 1/12 :=
sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x^6 / (x^8 + 2*x^7 - 4*x^6 + 8*x^5 + 16*x^4) = 1/12 :=
sorry

end max_value_of_function_max_value_achievable_l3299_329968


namespace product_of_points_on_line_l3299_329962

/-- A line in the coordinate plane passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem product_of_points_on_line :
  ∀ x y : ℝ,
  line_k x 6 →
  line_k 10 y →
  x * y = 60 := by
sorry

end product_of_points_on_line_l3299_329962


namespace train_speeds_l3299_329980

/-- Represents the speeds and lengths of two trains meeting on parallel tracks -/
structure TrainMeeting where
  speed1 : ℝ  -- Speed of first train in m/s
  speed2 : ℝ  -- Speed of second train in m/s
  length1 : ℝ  -- Length of first train in meters
  length2 : ℝ  -- Length of second train in meters
  initialDistance : ℝ  -- Initial distance between trains in meters
  meetingTime : ℝ  -- Time taken for trains to meet in seconds
  timeDifference : ℝ  -- Difference in time taken to pass a signal

/-- The theorem stating the speeds of the trains given the conditions -/
theorem train_speeds (tm : TrainMeeting) 
  (h1 : tm.length1 = 490)
  (h2 : tm.length2 = 210)
  (h3 : tm.initialDistance = 700)
  (h4 : tm.meetingTime = 28)
  (h5 : tm.timeDifference = 35)
  (h6 : tm.initialDistance = tm.meetingTime * (tm.speed1 + tm.speed2))
  (h7 : tm.length1 / tm.speed1 - tm.length2 / tm.speed2 = tm.timeDifference) :
  tm.speed1 = 10 ∧ tm.speed2 = 15 := by
  sorry


end train_speeds_l3299_329980


namespace expression_evaluation_l3299_329997

theorem expression_evaluation :
  let a : ℚ := -1/2
  (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := by sorry

end expression_evaluation_l3299_329997


namespace no_integer_roots_for_odd_coefficients_l3299_329932

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end no_integer_roots_for_odd_coefficients_l3299_329932


namespace candy_store_revenue_calculation_l3299_329963

/-- Calculates the total revenue of a candy store based on their sales of fudge, chocolate truffles, and chocolate-covered pretzels. -/
def candy_store_revenue (fudge_pounds : ℝ) (fudge_price : ℝ) 
                        (truffle_dozens : ℝ) (truffle_price : ℝ)
                        (pretzel_dozens : ℝ) (pretzel_price : ℝ) : ℝ :=
  fudge_pounds * fudge_price +
  truffle_dozens * 12 * truffle_price +
  pretzel_dozens * 12 * pretzel_price

/-- The candy store's revenue from selling fudge, chocolate truffles, and chocolate-covered pretzels is $212.00. -/
theorem candy_store_revenue_calculation :
  candy_store_revenue 20 2.50 5 1.50 3 2.00 = 212.00 := by
  sorry

end candy_store_revenue_calculation_l3299_329963


namespace toy_car_cost_l3299_329964

-- Define the given values
def initial_amount : ℚ := 17.80
def num_cars : ℕ := 4
def race_track_cost : ℚ := 6.00
def remaining_amount : ℚ := 8.00

-- Define the theorem
theorem toy_car_cost :
  (initial_amount - remaining_amount - race_track_cost) / num_cars = 0.95 := by
  sorry

end toy_car_cost_l3299_329964


namespace sphere_volume_from_intersection_l3299_329904

/-- Given a sphere intersected by a plane at distance 1 from its center,
    creating a cross-sectional area of π, prove that its volume is (8√2π)/3. -/
theorem sphere_volume_from_intersection (r : ℝ) : 
  (r^2 - 1^2 = 1^2) →   -- Pythagorean theorem for the right triangle
  (π * 1^2 = π) →       -- Cross-sectional area is π
  ((4/3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3) := by
  sorry

end sphere_volume_from_intersection_l3299_329904


namespace geometric_sequence_206th_term_l3299_329956

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_206th_term :
  let a₁ := 4
  let a₂ := -12
  let r := a₂ / a₁
  geometric_sequence a₁ r 206 = -4 * 3^204 :=
by sorry

end geometric_sequence_206th_term_l3299_329956


namespace clock_problem_l3299_329911

/-- Represents a clock with special striking properties -/
structure StrikingClock where
  /-- Time for each stroke and interval between strokes (in seconds) -/
  stroke_time : ℝ
  /-- Calculates the total time lapse for striking a given hour -/
  time_lapse : ℕ → ℝ
  /-- The time lapse is (2n - 1) * stroke_time, where n is the hour -/
  time_lapse_eq : ∀ (hour : ℕ), time_lapse hour = (2 * hour - 1) * stroke_time

/-- The theorem representing our clock problem -/
theorem clock_problem (clock : StrikingClock) 
    (h1 : clock.time_lapse 7 = 26) 
    (h2 : ∃ (hour : ℕ), clock.time_lapse hour = 22) : 
  ∃ (hour : ℕ), hour = 6 ∧ clock.time_lapse hour = 22 :=
sorry

end clock_problem_l3299_329911


namespace queen_probability_l3299_329947

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_queens : ℕ)

/-- The probability of drawing a specific card from a deck -/
def draw_probability (d : Deck) (num_specific_cards : ℕ) : ℚ :=
  num_specific_cards / d.total_cards

/-- Our specific deck -/
def modified_deck : Deck :=
  { total_cards := 54,
    num_queens := 4 }

theorem queen_probability :
  draw_probability modified_deck modified_deck.num_queens = 2 / 27 := by
  sorry

end queen_probability_l3299_329947


namespace remainder_13_pow_1033_mod_50_l3299_329927

theorem remainder_13_pow_1033_mod_50 : 13^1033 % 50 = 3 := by sorry

end remainder_13_pow_1033_mod_50_l3299_329927


namespace parabola_line_intersection_sum_l3299_329941

/-- Parabola P with equation y = x^2 + 4 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 4

/-- Point Q -/
def Q : ℝ × ℝ := (10, 5)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- The line through Q with slope m does not intersect P -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

theorem parabola_line_intersection_sum (r s : ℝ) 
  (h : ∀ m, no_intersection m ↔ r < m ∧ m < s) : 
  r + s = 40 := by sorry

end parabola_line_intersection_sum_l3299_329941


namespace exp_function_inequality_l3299_329920

/-- Given an exponential function f(x) = a^x where 0 < a < 1, 
    prove that f(3) * f(2) < f(2) -/
theorem exp_function_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let f := fun (x : ℝ) => a^x
  f 3 * f 2 < f 2 := by
  sorry

end exp_function_inequality_l3299_329920


namespace regular_polygon_sides_l3299_329957

theorem regular_polygon_sides (n : ℕ) (interior_angle exterior_angle : ℝ) : 
  n > 2 →
  interior_angle / exterior_angle = 5 →
  interior_angle + exterior_angle = 180 →
  n * exterior_angle = 360 →
  n = 12 := by
sorry

end regular_polygon_sides_l3299_329957


namespace expression_simplification_l3299_329979

theorem expression_simplification :
  (4 * 7) / (12 * 14) * ((9 * 12 * 14) / (4 * 7 * 9))^2 = 1 := by
  sorry

end expression_simplification_l3299_329979


namespace number_of_pumice_rocks_l3299_329909

/-- The number of slate rocks -/
def slate_rocks : ℕ := 10

/-- The number of granite rocks -/
def granite_rocks : ℕ := 4

/-- The probability of choosing 2 slate rocks at random without replacement -/
def prob_two_slate : ℚ := 15/100

/-- The number of pumice rocks -/
def pumice_rocks : ℕ := 11

theorem number_of_pumice_rocks :
  (slate_rocks : ℚ) * (slate_rocks - 1) / 
  ((slate_rocks + pumice_rocks + granite_rocks) * (slate_rocks + pumice_rocks + granite_rocks - 1)) = 
  prob_two_slate := by sorry

end number_of_pumice_rocks_l3299_329909


namespace uncovered_side_length_l3299_329922

/-- Represents a rectangular field with three sides fenced --/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field --/
def uncovered_side (field : FencedField) : ℝ :=
  field.length

theorem uncovered_side_length
  (field : FencedField)
  (h_area : field.area = 50)
  (h_fencing : field.fencing = 25)
  (h_rect : field.area = field.length * field.width)
  (h_fence : field.fencing = 2 * field.width + field.length) :
  uncovered_side field = 20 := by
  sorry

end uncovered_side_length_l3299_329922


namespace plum_problem_l3299_329967

theorem plum_problem (x : ℕ) : 
  (4 * x / 5 : ℚ) = (5 * x / 6 : ℚ) - 1 → 2 * x = 60 :=
by
  sorry

end plum_problem_l3299_329967


namespace quadratic_function_properties_l3299_329995

/-- A quadratic function passing through two given points -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 4

theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (QuadraticFunction a b (-1) = 3) ∧
    (QuadraticFunction a b 2 = 18) ∧
    (a = 2 ∧ b = 3) ∧
    (let vertex_x := -b / (2 * a)
     let vertex_y := QuadraticFunction a b vertex_x
     vertex_x = -3/4 ∧ vertex_y = 23/8) := by
  sorry

end quadratic_function_properties_l3299_329995


namespace unique_prime_factors_count_l3299_329906

theorem unique_prime_factors_count (n : ℕ+) (h : Nat.card (Nat.divisors n) = 12320) :
  Finset.card (Nat.factors n).toFinset = 4 := by
  sorry

end unique_prime_factors_count_l3299_329906


namespace right_triangle_30_perpendicular_segment_l3299_329940

/-- In a right triangle with one angle of 30°, the segment of the perpendicular
    from the hypotenuse midpoint to the longer leg is one-third of the longer leg. -/
theorem right_triangle_30_perpendicular_segment (A B C : ℝ × ℝ) 
  (h_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_30deg : Real.cos (Real.arccos ((C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) = Real.sqrt 3 / 2)
  (M : ℝ × ℝ)
  (h_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (K : ℝ × ℝ)
  (h_perpendicular : (K.1 - M.1) * (B.1 - A.1) + (K.2 - M.2) * (B.2 - A.2) = 0)
  (h_on_leg : ∃ t : ℝ, K = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))) :
  Real.sqrt ((K.1 - M.1)^2 + (K.2 - M.2)^2) = 
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / 3 :=
by sorry


end right_triangle_30_perpendicular_segment_l3299_329940


namespace lcm_gcd_problem_l3299_329989

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 8820)
  (h2 : Nat.gcd a b = 36)
  (h3 : a = 360) :
  b = 882 := by
  sorry

end lcm_gcd_problem_l3299_329989


namespace basketball_game_scores_l3299_329913

/-- Represents the scores of a team in four quarters -/
structure Scores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the given scores form an increasing geometric sequence -/
def isGeometric (s : Scores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if the given scores form an increasing arithmetic sequence -/
def isArithmetic (s : Scores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : Scores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the halftime score for a team -/
def halftimeScore (s : Scores) : ℕ := s.q1 + s.q2

theorem basketball_game_scores (eagles lions : Scores) 
  (h1 : isGeometric eagles)
  (h2 : isArithmetic lions)
  (h3 : halftimeScore eagles = halftimeScore lions)
  (h4 : totalScore eagles = totalScore lions) :
  halftimeScore eagles + halftimeScore lions = 8 := by
  sorry

end basketball_game_scores_l3299_329913


namespace water_bill_ratio_l3299_329988

def electricity_bill : ℚ := 60
def gas_bill : ℚ := 40
def water_bill : ℚ := 40
def internet_bill : ℚ := 25

def gas_bill_paid : ℚ := (3/4) * gas_bill + 5
def internet_bill_paid : ℚ := 4 * 5

def total_remaining : ℚ := 30

def water_bill_paid : ℚ := water_bill - (total_remaining - (gas_bill - gas_bill_paid) - (internet_bill - internet_bill_paid))

theorem water_bill_ratio : water_bill_paid / water_bill = 1 / 2 := by
  sorry

end water_bill_ratio_l3299_329988


namespace inequality_proof_l3299_329908

theorem inequality_proof (a b c x : ℝ) :
  (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + b^2) ≤ 
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ∧
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ≤ 
  (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + b^2) :=
by sorry

end inequality_proof_l3299_329908


namespace lollipop_challenge_l3299_329954

def joann_lollipops (n : ℕ) : ℕ := 8 + 2 * n

def tom_lollipops (n : ℕ) : ℕ := 5 * 2^(n - 1)

def total_lollipops : ℕ := 
  (Finset.range 7).sum joann_lollipops + (Finset.range 7).sum tom_lollipops

theorem lollipop_challenge : total_lollipops = 747 := by
  sorry

end lollipop_challenge_l3299_329954


namespace inner_outer_hexagon_area_ratio_is_three_fourths_l3299_329930

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The ratio of the area of the inner hexagon to the area of the outer hexagon -/
def inner_outer_hexagon_area_ratio (h : RegularHexagon) : ℚ :=
  3 / 4

/-- The theorem stating that the ratio of the areas is 3/4 -/
theorem inner_outer_hexagon_area_ratio_is_three_fourths (h : RegularHexagon) :
  inner_outer_hexagon_area_ratio h = 3 / 4 := by
  sorry

end inner_outer_hexagon_area_ratio_is_three_fourths_l3299_329930


namespace remaining_water_l3299_329990

/-- The amount of distilled water remaining after two experiments -/
theorem remaining_water (initial : ℚ) (used1 : ℚ) (used2 : ℚ) :
  initial = 3 →
  used1 = 5/4 →
  used2 = 1/3 →
  initial - (used1 + used2) = 17/12 := by
sorry

end remaining_water_l3299_329990


namespace line_parallel_perpendicular_l3299_329945

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α :=
sorry

end line_parallel_perpendicular_l3299_329945


namespace triangle_angle_sixty_degrees_l3299_329986

theorem triangle_angle_sixty_degrees (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  -- a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Given condition
  (2 * b - c) * Real.cos A = a * Real.cos C →
  -- Conclusion
  A = Real.pi / 3 := by
sorry

end triangle_angle_sixty_degrees_l3299_329986


namespace sum_of_positive_reals_l3299_329925

theorem sum_of_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end sum_of_positive_reals_l3299_329925


namespace original_number_is_eight_l3299_329974

theorem original_number_is_eight (N : ℤ) : (N + 1) % 9 = 0 → N = 8 := by
  sorry

end original_number_is_eight_l3299_329974


namespace function_value_at_two_l3299_329943

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end function_value_at_two_l3299_329943


namespace smallest_possible_M_l3299_329998

theorem smallest_possible_M (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2010) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  ∀ M', (∃ a' b' c' d' e' : ℕ+, a' + b' + c' + d' + e' = 2010 ∧
    M' = max (a' + b') (max (b' + c') (max (c' + d') (d' + e')))) →
  M' ≥ 671 :=
sorry

end smallest_possible_M_l3299_329998


namespace cake_division_l3299_329919

theorem cake_division (cake_weight : ℝ) (pierre_ate : ℝ) : 
  cake_weight = 400 ∧ pierre_ate = 100 → 
  ∃ (n : ℕ), n = 8 ∧ cake_weight / n = pierre_ate / 2 := by
sorry

end cake_division_l3299_329919


namespace candy_bar_count_l3299_329907

theorem candy_bar_count (bags : ℕ) (candy_per_bag : ℕ) (h1 : bags = 5) (h2 : candy_per_bag = 3) :
  bags * candy_per_bag = 15 := by
  sorry

end candy_bar_count_l3299_329907


namespace right_angled_triangle_exists_l3299_329936

theorem right_angled_triangle_exists (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  a ^ 2 + b ^ 2 = c ^ 2 :=
by sorry

end right_angled_triangle_exists_l3299_329936


namespace optimal_rate_maximizes_income_l3299_329903

/-- Represents the hotel's room pricing and occupancy model -/
structure HotelModel where
  totalRooms : ℕ
  baseRate : ℕ
  occupancyDecrease : ℕ
  rateIncrease : ℕ

/-- Calculates the number of occupied rooms based on the new rate -/
def occupiedRooms (model : HotelModel) (newRate : ℕ) : ℤ :=
  model.totalRooms - (newRate - model.baseRate) / model.rateIncrease * model.occupancyDecrease

/-- Calculates the total daily income based on the new rate -/
def dailyIncome (model : HotelModel) (newRate : ℕ) : ℕ :=
  newRate * (occupiedRooms model newRate).toNat

/-- The optimal rate that maximizes daily income -/
def optimalRate (model : HotelModel) : ℕ := model.baseRate + model.rateIncrease * (model.totalRooms / model.occupancyDecrease) / 2

/-- Theorem stating that the optimal rate maximizes daily income -/
theorem optimal_rate_maximizes_income (model : HotelModel) :
  model.totalRooms = 300 →
  model.baseRate = 200 →
  model.occupancyDecrease = 10 →
  model.rateIncrease = 20 →
  ∀ rate, dailyIncome model (optimalRate model) ≥ dailyIncome model rate := by
  sorry

#eval optimalRate { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 }
#eval dailyIncome { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 } 400

end optimal_rate_maximizes_income_l3299_329903


namespace polynomial_value_at_minus_one_l3299_329944

/-- Given real numbers a, b, and c, define polynomials g and f -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 5
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 50*x + c

/-- The main theorem to prove -/
theorem polynomial_value_at_minus_one 
  (a b c : ℝ) 
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    g a r1 = 0 ∧ g a r2 = 0 ∧ g a r3 = 0)
  (h2 : ∀ x : ℝ, g a x = 0 → f b c x = 0) :
  f b c (-1) = -1804 := by
  sorry

#check polynomial_value_at_minus_one

end polynomial_value_at_minus_one_l3299_329944


namespace ruiStateSurvey2016_sampleSize_l3299_329912

/-- Represents a survey about student heights -/
structure HeightSurvey where
  city : String
  year : Nat
  sampleCount : Nat

/-- Definition of sample size for a height survey -/
def sampleSize (survey : HeightSurvey) : Nat := survey.sampleCount

/-- The specific survey conducted in Rui State City in 2016 -/
def ruiStateSurvey2016 : HeightSurvey := {
  city := "Rui State City"
  year := 2016
  sampleCount := 200
}

/-- Theorem stating that the sample size of the Rui State City survey in 2016 is 200 -/
theorem ruiStateSurvey2016_sampleSize :
  sampleSize ruiStateSurvey2016 = 200 := by
  sorry


end ruiStateSurvey2016_sampleSize_l3299_329912


namespace five_digit_divisible_by_18_l3299_329991

theorem five_digit_divisible_by_18 (n : ℕ) : 
  n < 10 ∧ 
  73420 ≤ 7342 * 10 + n ∧ 
  7342 * 10 + n < 73430 ∧
  (7342 * 10 + n) % 18 = 0 
  ↔ n = 2 := by sorry

end five_digit_divisible_by_18_l3299_329991


namespace maciek_purchase_cost_l3299_329993

/-- The cost of Maciek's purchases -/
def total_cost (pretzel_cost : ℝ) (chip_cost_percentage : ℝ) : ℝ :=
  let chip_cost := pretzel_cost * (1 + chip_cost_percentage)
  2 * pretzel_cost + 2 * chip_cost

/-- Proof that Maciek's purchases cost $22 -/
theorem maciek_purchase_cost :
  total_cost 4 0.75 = 22 := by
  sorry

end maciek_purchase_cost_l3299_329993


namespace m_zero_sufficient_not_necessary_l3299_329952

/-- Determines if the equation x^2+y^2-4x+2y+m=0 represents a circle -/
def is_circle (m : ℝ) : Prop := (-4)^2 + 2^2 - 4*m > 0

/-- The condition m=0 is sufficient but not necessary for the equation to represent a circle -/
theorem m_zero_sufficient_not_necessary : 
  (is_circle 0) ∧ (∃ m : ℝ, m ≠ 0 ∧ is_circle m) :=
sorry

end m_zero_sufficient_not_necessary_l3299_329952


namespace waiter_tables_l3299_329966

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : 
  initial_customers = 44 → left_customers = 12 → people_per_table = 8 → 
  (initial_customers - left_customers) / people_per_table = 4 := by
sorry

end waiter_tables_l3299_329966


namespace library_problem_l3299_329961

/-- Represents the number of students that can be helped on the fourth day given the initial number of books and the number of students helped in the first three days. -/
def students_helped_fourth_day (total_books : ℕ) (books_per_student : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  (total_books - (day1 + day2 + day3) * books_per_student) / books_per_student

/-- Theorem stating that given the specific conditions of the library problem, 9 students can be helped on the fourth day. -/
theorem library_problem :
  students_helped_fourth_day 120 5 4 5 6 = 9 := by
  sorry

end library_problem_l3299_329961


namespace marble_theorem_l3299_329996

/-- Represents a jar containing marbles -/
structure Jar where
  red : ℕ
  yellow : ℕ

/-- The problem setup -/
def marble_problem : Prop :=
  ∃ (jar1 jar2 : Jar),
    -- Ratio conditions
    jar1.red * 2 = jar1.yellow * 7 ∧
    jar2.red * 3 = jar2.yellow * 5 ∧
    -- Total yellow marbles
    jar1.yellow + jar2.yellow = 50 ∧
    -- Total marbles in Jar 2 is 20 more than Jar 1
    jar2.red + jar2.yellow = jar1.red + jar1.yellow + 20 ∧
    -- The conclusion we want to prove
    jar1.red = jar2.red + 2

theorem marble_theorem : marble_problem := by
  sorry

end marble_theorem_l3299_329996


namespace sum_mod_seven_l3299_329934

theorem sum_mod_seven : (5432 + 5433 + 5434 + 5435) % 7 = 2 := by
  sorry

end sum_mod_seven_l3299_329934


namespace linear_equation_solution_l3299_329926

theorem linear_equation_solution (a : ℝ) : 
  (∃ (x y : ℝ), a * x - 2 * y = 2 ∧ x = 4 ∧ y = 5) → a = 3 := by
  sorry

end linear_equation_solution_l3299_329926


namespace power_negative_cube_fourth_l3299_329938

theorem power_negative_cube_fourth (a : ℝ) : (-a^3)^4 = a^12 := by
  sorry

end power_negative_cube_fourth_l3299_329938


namespace pencil_count_l3299_329928

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 33 initial pencils and 27 added pencils, the total is 60 -/
theorem pencil_count : total_pencils 33 27 = 60 := by
  sorry

end pencil_count_l3299_329928


namespace parabola_vertex_l3299_329929

/-- The vertex of the parabola y = 4x^2 + 16x + 20 is (-2, 4) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => 4 * x^2 + 16 * x + 20
  ∃! (m n : ℝ), (∀ x, f x ≥ f m) ∧ f m = n ∧ m = -2 ∧ n = 4 :=
sorry

end parabola_vertex_l3299_329929


namespace class_average_theorem_l3299_329971

theorem class_average_theorem :
  let boy_percentage : ℝ := 0.4
  let girl_percentage : ℝ := 1 - boy_percentage
  let boy_score : ℝ := 80
  let girl_score : ℝ := 90
  let class_average : ℝ := boy_percentage * boy_score + girl_percentage * girl_score
  class_average = 86 := by
sorry

end class_average_theorem_l3299_329971


namespace power_of_power_three_cubed_squared_l3299_329951

theorem power_of_power_three_cubed_squared : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_cubed_squared_l3299_329951


namespace ball_probabilities_l3299_329982

/-- Represents the contents of a box with black and white balls -/
structure Box where
  total : ℕ
  black : ℕ
  white : ℕ
  black_ratio : ℚ
  white_ratio : ℚ
  ratio_sum_one : black_ratio + white_ratio = 1
  contents_match : black + white = total

/-- The setup of the three boxes as per the problem -/
def box_setup : (Box × Box × Box) := sorry

/-- The probability of selecting all black balls when choosing one from each box -/
def prob_all_black (boxes : Box × Box × Box) : ℚ := sorry

/-- The probability of selecting a white ball from all boxes combined -/
def prob_white_combined (boxes : Box × Box × Box) : ℚ := sorry

/-- Main theorem stating the probabilities as per the problem -/
theorem ball_probabilities :
  let boxes := box_setup
  prob_all_black boxes = 1/20 ∧ prob_white_combined boxes = 3/5 := by
  sorry

end ball_probabilities_l3299_329982


namespace provisions_last_20_days_l3299_329975

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def days_after_reinforcement (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (reinforcement : ℕ) : ℕ :=
  let initial_man_days := initial_men * initial_days
  let used_man_days := initial_men * days_before_reinforcement
  let remaining_man_days := initial_man_days - used_man_days
  let total_men_after_reinforcement := initial_men + reinforcement
  remaining_man_days / total_men_after_reinforcement

/-- Theorem stating that given the problem conditions, the provisions will last 20 more days after reinforcement -/
theorem provisions_last_20_days :
  days_after_reinforcement 2000 65 15 3000 = 20 := by
  sorry

end provisions_last_20_days_l3299_329975


namespace factorize_2x_squared_minus_18_l3299_329910

theorem factorize_2x_squared_minus_18 (x : ℝ) :
  2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by sorry

end factorize_2x_squared_minus_18_l3299_329910


namespace troy_computer_worth_l3299_329953

/-- The worth of Troy's new computer -/
def new_computer_worth (initial_savings selling_price additional_needed : ℕ) : ℕ :=
  initial_savings + selling_price + additional_needed

/-- Theorem: The worth of Troy's new computer is $80 -/
theorem troy_computer_worth :
  new_computer_worth 50 20 10 = 80 := by
  sorry

end troy_computer_worth_l3299_329953


namespace james_coat_cost_l3299_329917

def total_cost : ℕ := 110
def shoe_cost : ℕ := 30
def jeans_cost : ℕ := 20

def james_items : ℕ := 3  -- 1 coat and 2 pairs of jeans
def jamie_items : ℕ := 1  -- 1 pair of shoes

theorem james_coat_cost : 
  total_cost - (shoe_cost + 2 * jeans_cost) = 40 := by
  sorry

end james_coat_cost_l3299_329917


namespace contractor_problem_l3299_329970

/-- Represents the number of days originally planned to complete the work -/
def original_days : ℕ := 9

/-- Represents the number of absent laborers -/
def absent_laborers : ℕ := 10

/-- Represents the number of days taken by the remaining laborers to complete the work -/
def actual_days : ℕ := 18

/-- Represents the total number of laborers originally employed -/
def total_laborers : ℕ := 11

theorem contractor_problem :
  (original_days : ℚ) * (total_laborers - absent_laborers) = actual_days * total_laborers :=
by sorry

end contractor_problem_l3299_329970


namespace canoe_weight_with_dog_l3299_329984

/-- Calculates the total weight carried by Penny's canoe with her dog -/
theorem canoe_weight_with_dog (normal_capacity : ℕ) (person_weight : ℝ) : 
  normal_capacity = 6 →
  person_weight = 140 →
  (2 : ℝ) / 3 * normal_capacity * person_weight + 1 / 4 * person_weight = 595 :=
by
  sorry

#check canoe_weight_with_dog

end canoe_weight_with_dog_l3299_329984


namespace f_composition_result_l3299_329924

noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

theorem f_composition_result : f (f (f (f (2 + I)))) = 164833 + 354816 * I := by
  sorry

end f_composition_result_l3299_329924


namespace weight_differences_l3299_329978

/-- Given the weights of four individuals, prove the weight differences between one individual and the other three. -/
theorem weight_differences (H E1 E2 E3 : ℕ) 
  (h_H : H = 87)
  (h_E1 : E1 = 58)
  (h_E2 : E2 = 56)
  (h_E3 : E3 = 64) :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) := by
  sorry

end weight_differences_l3299_329978


namespace probability_of_seven_in_three_elevenths_l3299_329905

-- Define the fraction
def fraction : ℚ := 3 / 11

-- Define the decimal representation as a sequence of digits
def decimal_representation : ℕ → ℕ
  | 0 => 0  -- The digit before the decimal point
  | (n + 1) => if n % 2 = 0 then 2 else 7  -- The repeating pattern 27

-- Define the probability of selecting a 7
def probability_of_seven : ℚ := 1 / 2

-- Theorem statement
theorem probability_of_seven_in_three_elevenths :
  (∃ (n : ℕ), decimal_representation n = 7) ∧ 
  (∀ (m : ℕ), m ≠ 0 → decimal_representation m = decimal_representation (m + 2)) →
  probability_of_seven = 1 / 2 :=
sorry

end probability_of_seven_in_three_elevenths_l3299_329905


namespace symmetric_difference_A_B_l3299_329960

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- Define the set difference operation
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetric_difference (M N : Set ℝ) : Set ℝ := (set_difference M N) ∪ (set_difference N M)

-- State the theorem
theorem symmetric_difference_A_B :
  symmetric_difference A B = {x | x ≥ 0 ∨ x < -9/4} := by sorry

end symmetric_difference_A_B_l3299_329960


namespace solution_set_of_inequality_l3299_329923

open Set
open Function
open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x > f x) :
  {x : ℝ | f x / Real.exp x > f 1 / Real.exp 1} = Ioi 1 := by sorry

end solution_set_of_inequality_l3299_329923


namespace range_of_2a_plus_3b_l3299_329959

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 2*a + 3*b ≤ x → x ≥ 3) ∧ 
  (∀ y, 2*a + 3*b ≥ y → y ≤ -3) :=
by sorry

#check range_of_2a_plus_3b

end range_of_2a_plus_3b_l3299_329959


namespace pool_depth_is_10_feet_l3299_329958

-- Define the pool parameters
def drainRate : ℝ := 60
def poolWidth : ℝ := 80
def poolLength : ℝ := 150
def drainTime : ℝ := 2000

-- Theorem statement
theorem pool_depth_is_10_feet :
  let totalVolume := drainRate * drainTime
  let poolArea := poolWidth * poolLength
  totalVolume / poolArea = 10 := by
  sorry

end pool_depth_is_10_feet_l3299_329958


namespace problem_solution_l3299_329987

theorem problem_solution : 
  (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / N → N = 1991 := by
  sorry

end problem_solution_l3299_329987


namespace michael_has_fifteen_robots_l3299_329950

/-- Calculates the number of flying robots Michael has given Tom's count and the multiplier. -/
def michaels_robots (toms_robots : ℕ) (multiplier : ℕ) : ℕ :=
  toms_robots * multiplier

/-- Proves that Michael has 15 flying robots given the conditions. -/
theorem michael_has_fifteen_robots :
  let toms_robots : ℕ := 3
  let multiplier : ℕ := 4
  michaels_robots toms_robots multiplier = 15 := by
  sorry

#eval michaels_robots 3 4  -- This should output 15

end michael_has_fifteen_robots_l3299_329950


namespace sum_exterior_angles_regular_hexagon_l3299_329981

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 angles of equal measure. -/
def RegularHexagon : Type := Unit

/-- The sum of the exterior angles of a polygon. -/
def SumExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a regular hexagon is 360°. -/
theorem sum_exterior_angles_regular_hexagon :
  SumExteriorAngles RegularHexagon = 360 := by sorry

end sum_exterior_angles_regular_hexagon_l3299_329981


namespace post_office_packages_l3299_329999

theorem post_office_packages (letters_per_day : ℕ) (total_mail : ℕ) (days_per_month : ℕ) (months : ℕ) :
  letters_per_day = 60 →
  total_mail = 14400 →
  days_per_month = 30 →
  months = 6 →
  (total_mail - letters_per_day * days_per_month * months) / (days_per_month * months) = 20 :=
by sorry

end post_office_packages_l3299_329999


namespace largest_three_digit_congruence_l3299_329900

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 998 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (70 * n) % 350 = 210 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ (70 * m) % 350 = 210 → m ≤ n :=
by sorry

end largest_three_digit_congruence_l3299_329900


namespace max_sum_of_complex_numbers_l3299_329992

theorem max_sum_of_complex_numbers (a b : ℂ) : 
  a^2 + b^2 = 5 → 
  a^3 + b^3 = 7 → 
  (a + b).re ≤ (-1 + Real.sqrt 57) / 2 :=
by sorry

end max_sum_of_complex_numbers_l3299_329992


namespace portion_to_whole_cup_ratio_l3299_329915

theorem portion_to_whole_cup_ratio : 
  ∀ (grains_per_cup : ℕ) 
    (tablespoons_per_portion : ℕ) 
    (teaspoons_per_tablespoon : ℕ) 
    (grains_per_teaspoon : ℕ),
  grains_per_cup = 480 →
  tablespoons_per_portion = 8 →
  teaspoons_per_tablespoon = 3 →
  grains_per_teaspoon = 10 →
  (tablespoons_per_portion * teaspoons_per_tablespoon * grains_per_teaspoon) * 2 = grains_per_cup :=
by
  sorry

end portion_to_whole_cup_ratio_l3299_329915


namespace cd_length_problem_l3299_329965

theorem cd_length_problem (x : ℝ) : x > 0 ∧ x + x + 2*x = 6 → x = 1.5 := by
  sorry

end cd_length_problem_l3299_329965


namespace spider_position_after_2055_jumps_l3299_329994

/-- Represents the possible positions on the circle -/
inductive Position : Type
  | one | two | three | four | five | six | seven

/-- Defines the next position after a hop based on the current position -/
def nextPosition (p : Position) : Position :=
  match p with
  | Position.one => Position.two
  | Position.two => Position.five
  | Position.three => Position.four
  | Position.four => Position.seven
  | Position.five => Position.six
  | Position.six => Position.two
  | Position.seven => Position.one

/-- Calculates the position after n hops -/
def positionAfterNHops (start : Position) (n : ℕ) : Position :=
  match n with
  | 0 => start
  | n + 1 => nextPosition (positionAfterNHops start n)

theorem spider_position_after_2055_jumps :
  positionAfterNHops Position.six 2055 = Position.two :=
sorry

end spider_position_after_2055_jumps_l3299_329994


namespace right_triangle_cos_z_l3299_329942

theorem right_triangle_cos_z (X Y Z : ℝ) : 
  -- Triangle XYZ is right-angled at X
  X + Y + Z = π →
  X = π / 2 →
  -- sin Y = 3/5
  Real.sin Y = 3 / 5 →
  -- Prove: cos Z = 3/5
  Real.cos Z = 3 / 5 := by
sorry

end right_triangle_cos_z_l3299_329942


namespace circle_set_equivalence_l3299_329937

-- Define the circle C
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point D
def D : ℝ × ℝ := sorry

-- Define the circle C
def C : Circle := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the set of points A
def A : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem circle_set_equivalence :
  (∀ (p : ℝ × ℝ), p ∈ A ↔ 
    distance p D < C.radius ∧ 
    (∀ (q : ℝ × ℝ), distance q C.center = C.radius → distance p D ≤ distance p q)) ↔
  (∀ (p : ℝ × ℝ), p ∈ A ↔ 
    distance p D < C.radius) :=
sorry

end circle_set_equivalence_l3299_329937


namespace complex_magnitude_constraint_l3299_329985

theorem complex_magnitude_constraint (a : ℝ) :
  let z : ℂ := 1 + a * I
  (Complex.abs z < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
  sorry

end complex_magnitude_constraint_l3299_329985


namespace box_volume_increase_l3299_329949

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4320)
  (surface_area : 2 * (l * w + w * h + h * l) = 1704)
  (edge_sum : 4 * (l + w + h) = 208) :
  (l + 1) * (w + 1) * (h + 1) = 5225 := by
  sorry

end box_volume_increase_l3299_329949


namespace f_min_value_is_4_l3299_329976

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem f_min_value_is_4 : ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 := by sorry

end f_min_value_is_4_l3299_329976


namespace min_value_of_f_l3299_329973

/-- The function representing the expression |x-1|+3 --/
def f (x : ℝ) : ℝ := |x - 1| + 3

/-- Theorem stating that the minimum value of |x-1|+3 is 3 and occurs at x = 1 --/
theorem min_value_of_f :
  (∀ x : ℝ, f x ≥ 3) ∧ (f 1 = 3) := by
  sorry

end min_value_of_f_l3299_329973


namespace steves_remaining_oranges_l3299_329977

/-- Given that Steve has 46 oranges initially, shares 4 with Patrick and 7 with Samantha,
    prove that he will have 35 oranges left. -/
theorem steves_remaining_oranges :
  ∀ (initial shared_patrick shared_samantha : ℕ),
    initial = 46 →
    shared_patrick = 4 →
    shared_samantha = 7 →
    initial - (shared_patrick + shared_samantha) = 35 := by
  sorry

end steves_remaining_oranges_l3299_329977


namespace bucket_capacity_l3299_329955

theorem bucket_capacity (x : ℝ) 
  (h1 : 12 * x = 84 * 7) : x = 49 := by
  sorry

end bucket_capacity_l3299_329955


namespace no_function_with_double_application_plus_2019_l3299_329916

-- Statement of the theorem
theorem no_function_with_double_application_plus_2019 :
  ¬ (∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2019) := by
  sorry

end no_function_with_double_application_plus_2019_l3299_329916


namespace solution_set_f_gt_x_range_of_a_when_f_plus_3_nonneg_l3299_329914

noncomputable section

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - (2*a + 2)

-- Part 1: Solution set of f(x) > x
theorem solution_set_f_gt_x (a : ℝ) :
  (∀ x, f a x > x ↔ 
    (a > -3/2 ∧ (x > 2*a + 2 ∨ x < -1)) ∨
    (a = -3/2 ∧ x ≠ -1) ∨
    (a < -3/2 ∧ (x > -1 ∨ x < 2*a + 2))) :=
sorry

-- Part 2: Range of a when f(x) + 3 ≥ 0 for x ∈ (-1, +∞)
theorem range_of_a_when_f_plus_3_nonneg :
  (∀ x, x > -1 → f a x + 3 ≥ 0) ↔ a ≤ Real.sqrt 2 - 1 :=
sorry

end solution_set_f_gt_x_range_of_a_when_f_plus_3_nonneg_l3299_329914


namespace cos_double_angle_special_case_l3299_329918

theorem cos_double_angle_special_case (θ : Real) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 → Real.cos (2 * θ) = 4 / 5 := by
  sorry

end cos_double_angle_special_case_l3299_329918


namespace clark_bought_seven_parts_l3299_329901

/-- The number of parts Clark bought -/
def n : ℕ := sorry

/-- The original price of each part in dollars -/
def original_price : ℕ := 80

/-- The total amount Clark paid in dollars -/
def total_paid : ℕ := 439

/-- The total discount in dollars -/
def total_discount : ℕ := 121

theorem clark_bought_seven_parts : n = 7 := by
  sorry

end clark_bought_seven_parts_l3299_329901


namespace expression_evaluation_l3299_329969

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 2
  (7 * a^2 * b + (-4 * a^2 * b + 5 * a * b^2) - (2 * a^2 * b - 3 * a * b^2)) = -30 :=
by sorry

end expression_evaluation_l3299_329969


namespace distance_traveled_by_A_is_60km_l3299_329948

/-- Calculates the distance traveled by A until meeting B given initial conditions and speed doubling rule -/
def distanceTraveledByA (initialDistance : ℝ) (initialSpeedA : ℝ) (initialSpeedB : ℝ) : ℝ :=
  let firstHourDistance := initialSpeedA
  let secondHourDistance := 2 * initialSpeedA
  let thirdHourDistance := 4 * initialSpeedA * 0.75
  firstHourDistance + secondHourDistance + thirdHourDistance

/-- Theorem stating that A travels 60 km until meeting B -/
theorem distance_traveled_by_A_is_60km :
  distanceTraveledByA 90 10 5 = 60 := by
  sorry

#eval distanceTraveledByA 90 10 5

end distance_traveled_by_A_is_60km_l3299_329948


namespace wage_payment_days_l3299_329921

/-- Given a sum of money that can pay y's wages for 45 days and both x and y's wages for 20 days,
    prove that it can pay x's wages for 36 days. -/
theorem wage_payment_days (S : ℝ) (Wx Wy : ℝ) (S_positive : S > 0) (Wx_positive : Wx > 0) (Wy_positive : Wy > 0) :
  S = 45 * Wy ∧ S = 20 * (Wx + Wy) → S = 36 * Wx := by
  sorry

#check wage_payment_days

end wage_payment_days_l3299_329921


namespace sum_odd_positions_arithmetic_sequence_l3299_329983

/-- Represents an arithmetic sequence with the given properties -/
def ArithmeticSequence (n : ℕ) (d : ℕ) (total_sum : ℕ) :=
  {seq : ℕ → ℕ | 
    (∀ i, i > 0 → i < n → seq (i + 1) = seq i + d) ∧
    (Finset.sum (Finset.range n) seq = total_sum)}

/-- Sum of terms at odd positions in the sequence -/
def SumOddPositions (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ i => i % 2 = 1) (Finset.range n)) seq

theorem sum_odd_positions_arithmetic_sequence :
  ∀ (seq : ℕ → ℕ),
    seq ∈ ArithmeticSequence 1500 2 7500 →
    SumOddPositions seq 1500 = 3000 := by
  sorry

end sum_odd_positions_arithmetic_sequence_l3299_329983


namespace factor_expression_l3299_329933

theorem factor_expression (x : ℝ) : x * (x + 4) + 3 * (x + 4) = (x + 4) * (x + 3) := by
  sorry

end factor_expression_l3299_329933


namespace condition_relationship_l3299_329972

theorem condition_relationship (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (p → q) ∧ ¬(q → p) := by
  sorry

end condition_relationship_l3299_329972


namespace repeating_decimal_56_l3299_329946

def repeating_decimal (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem repeating_decimal_56 :
  repeating_decimal 56 = 56 / 99 := by sorry

end repeating_decimal_56_l3299_329946


namespace betty_strawberries_l3299_329902

/-- Proves that Betty picked 16 strawberries given the conditions of the problem -/
theorem betty_strawberries : ∃ (B N : ℕ),
  let M := B + 20
  let total_strawberries := B + M + N
  let jars := 40 / 4
  let strawberries_per_jar := 7
  B + 20 = 2 * N ∧
  total_strawberries = jars * strawberries_per_jar ∧
  B = 16 := by
  sorry


end betty_strawberries_l3299_329902


namespace triangle_inequality_l3299_329939

/-- Given a triangle ABC with sides a and b, and a point E on side AB such that AE:EB = n:m, 
    prove that CE < (ma + mb) / (m + n). -/
theorem triangle_inequality (A B C E : ℝ × ℝ) (a b : ℝ) (m n : ℝ) :
  let AB := dist A B
  let BC := dist B C
  let CA := dist C A
  let AE := dist A E
  let EB := dist E B
  let CE := dist C E
  (AB = a) →
  (BC = b) →
  (E.1 - A.1) / (B.1 - E.1) = n / m →
  (E.2 - A.2) / (B.2 - E.2) = n / m →
  CE < (m * a + m * b) / (m + n) :=
by sorry

end triangle_inequality_l3299_329939
