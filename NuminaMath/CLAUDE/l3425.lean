import Mathlib

namespace solve_equation_l3425_342547

theorem solve_equation : 
  ∃ x : ℝ, 3 + 2 * (8 - x) = 24.16 ∧ x = -2.58 := by
  sorry

end solve_equation_l3425_342547


namespace reflection_of_line_l3425_342533

/-- Given a line with equation 2x + 3y - 5 = 0, its reflection about the line y = x
    is the line with equation 3x + 2y - 5 = 0 -/
theorem reflection_of_line :
  let original_line : ℝ → ℝ → Prop := λ x y ↦ 2*x + 3*y - 5 = 0
  let reflection_axis : ℝ → ℝ → Prop := λ x y ↦ y = x
  let reflected_line : ℝ → ℝ → Prop := λ x y ↦ 3*x + 2*y - 5 = 0
  ∀ (x y : ℝ), original_line x y ↔ reflected_line y x :=
by sorry

end reflection_of_line_l3425_342533


namespace min_value_inequality_l3425_342527

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  ∃ (x : ℝ), ∀ (y : ℝ), (b - 2*c) / a ≥ x ∧ (b - 2*c) / a = x ↔ b / a = 4/5 ∧ c / a = 11/5 :=
by sorry

end min_value_inequality_l3425_342527


namespace complex_fraction_simplification_l3425_342531

theorem complex_fraction_simplification : 
  (((12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484)) : ℚ) /
  ((6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484)) = 181 := by
  sorry

end complex_fraction_simplification_l3425_342531


namespace equal_intercept_line_perpendicular_line_l3425_342516

-- Define the point (2, 3)
def point : ℝ × ℝ := (2, 3)

-- Define the lines given in the problem
def line1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y - 2 = 0
def line3 (x y : ℝ) : Prop := 7*x + 5*y + 1 = 0

-- Define the concept of a line having equal intercepts
def has_equal_intercepts (a b c : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c/a = c/b

-- Define perpendicularity of lines
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Statement for the first part of the problem
theorem equal_intercept_line :
  ∃ (a b c : ℝ), (a * point.1 + b * point.2 + c = 0) ∧
  has_equal_intercepts a b c ∧
  ((a = 3 ∧ b = -2 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -5)) := by sorry

-- Statement for the second part of the problem
theorem perpendicular_line :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
  perpendicular a b 7 5 ∧
  a = 5 ∧ b = -7 ∧ c = -3 := by sorry

end equal_intercept_line_perpendicular_line_l3425_342516


namespace fib_100_mod_9_l3425_342553

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 100th Fibonacci number is congruent to 3 modulo 9 -/
theorem fib_100_mod_9 : fib 100 % 9 = 3 := by
  sorry

end fib_100_mod_9_l3425_342553


namespace negation_of_universal_proposition_l3425_342599

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_universal_proposition_l3425_342599


namespace fraction_addition_l3425_342534

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end fraction_addition_l3425_342534


namespace problem_solution_l3425_342594

theorem problem_solution : (1 / (Real.sqrt 2 + 1) - Real.sqrt 8 + (Real.sqrt 3 + 1) ^ 0) = -Real.sqrt 2 := by
  sorry

end problem_solution_l3425_342594


namespace min_distance_between_sets_l3425_342542

/-- The minimum distance between a point on the set defined by y² - 3x² - 2xy - 9 - 12x = 0
    and a point on the set defined by x² - 8y + 23 + 6x + y² = 0 -/
theorem min_distance_between_sets :
  let set1 := {(x, y) : ℝ × ℝ | y^2 - 3*x^2 - 2*x*y - 9 - 12*x = 0}
  let set2 := {(x, y) : ℝ × ℝ | x^2 - 8*y + 23 + 6*x + y^2 = 0}
  ∃ (min_dist : ℝ), min_dist = (7 * Real.sqrt 10) / 10 - Real.sqrt 2 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ set1 → b ∈ set2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
by sorry

end min_distance_between_sets_l3425_342542


namespace travel_time_calculation_l3425_342503

theorem travel_time_calculation (total_time subway_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : subway_time = 10)
  (h3 : total_time = subway_time + 2 * subway_time + (total_time - subway_time - 2 * subway_time)) :
  total_time - subway_time - 2 * subway_time = 8 :=
by sorry

end travel_time_calculation_l3425_342503


namespace goldies_earnings_l3425_342589

/-- Calculates the total earnings for pet-sitting over two weeks -/
def total_earnings (hourly_rate : ℕ) (hours_week1 : ℕ) (hours_week2 : ℕ) : ℕ :=
  hourly_rate * hours_week1 + hourly_rate * hours_week2

/-- Proves that Goldie's total earnings for two weeks of pet-sitting is $250 -/
theorem goldies_earnings : total_earnings 5 20 30 = 250 := by
  sorry

end goldies_earnings_l3425_342589


namespace llama_accessible_area_l3425_342592

/-- Represents a rectangular shed -/
structure Shed :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area accessible to a llama tied to the corner of a shed -/
def accessible_area (s : Shed) (leash_length : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the accessible area for a llama tied to a 2m by 4m shed with a 4m leash -/
theorem llama_accessible_area :
  let s : Shed := ⟨4, 2⟩
  let leash_length : ℝ := 4
  accessible_area s leash_length = 13 * Real.pi := by
  sorry

end llama_accessible_area_l3425_342592


namespace expression_evaluation_l3425_342520

theorem expression_evaluation : -20 + 12 * ((5 + 15) / 4) = 40 := by
  sorry

end expression_evaluation_l3425_342520


namespace equation_solution_l3425_342511

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end equation_solution_l3425_342511


namespace lawn_mowing_earnings_l3425_342513

/-- Edward's lawn mowing earnings problem -/
theorem lawn_mowing_earnings 
  (total_earnings summer_earnings spring_earnings supplies_cost end_amount : ℕ)
  (h1 : total_earnings = summer_earnings + spring_earnings)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5)
  (h4 : end_amount = 24)
  (h5 : total_earnings = end_amount + supplies_cost) :
  spring_earnings = 2 := by
  sorry

end lawn_mowing_earnings_l3425_342513


namespace colored_triangle_existence_l3425_342570

-- Define the number of colors
def num_colors : ℕ := 1992

-- Define a type for colors
def Color := Fin num_colors

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for triangles
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a point is on a side of a triangle (excluding vertices)
def on_side (p : Point) (t : Triangle) : Prop := sorry

-- Define a function to check if a side of a triangle contains a point of a given color
def side_has_color (t : Triangle) (c : Color) : Prop := sorry

-- State the theorem
theorem colored_triangle_existence :
  (∀ c : Color, ∃ p : Point, coloring p = c) →
  ∀ T : Triangle, ∃ T' : Triangle,
    congruent T T' ∧
    ∃ c1 c2 c3 : Color,
      side_has_color T' c1 ∧
      side_has_color T' c2 ∧
      side_has_color T' c3 :=
by sorry

end colored_triangle_existence_l3425_342570


namespace red_in_B_equals_black_in_C_l3425_342558

-- Define the types for balls and boxes
inductive Color : Type
| Red : Color
| Black : Color

structure Box :=
  (red : Nat)
  (black : Nat)

-- Define the initial state
def initial_state (n : Nat) : Box × Box × Box :=
  ⟨⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩

-- Define the process of distributing balls
def distribute_balls (n : Nat) : Box × Box × Box :=
  sorry

-- Theorem statement
theorem red_in_B_equals_black_in_C (n : Nat) (h : Even n) :
  let ⟨boxA, boxB, boxC⟩ := distribute_balls n
  boxB.red = boxC.black := by sorry

end red_in_B_equals_black_in_C_l3425_342558


namespace carrot_consumption_theorem_l3425_342559

theorem carrot_consumption_theorem :
  ∃ (x y z : ℕ), x + y + z = 15 ∧ z % 2 = 1 := by
  sorry

end carrot_consumption_theorem_l3425_342559


namespace derivative_x_ln_x_l3425_342572

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * Real.log x) x = Real.log x + 1 := by
  sorry

end derivative_x_ln_x_l3425_342572


namespace parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l3425_342597

/-- The slope of a line parallel to the line containing the points (2, -3) and (-4, 5) is -4/3 -/
theorem parallel_line_slope : ℝ → ℝ → Prop :=
  fun x y =>
    let point1 : ℝ × ℝ := (2, -3)
    let point2 : ℝ × ℝ := (-4, 5)
    let slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
    slope = -4/3

/-- The theorem statement -/
theorem parallel_line_slope_is_negative_four_thirds :
  ∃ (x y : ℝ), parallel_line_slope x y :=
by
  sorry

end parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l3425_342597


namespace x_squared_congruence_l3425_342573

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 15 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
  sorry

end x_squared_congruence_l3425_342573


namespace student_average_score_l3425_342560

/-- Given a student's scores in physics, chemistry, and mathematics, prove that the average of all three subjects is 60. -/
theorem student_average_score (P C M : ℝ) : 
  P = 140 →                -- Physics score
  (P + M) / 2 = 90 →       -- Average of physics and mathematics
  (P + C) / 2 = 70 →       -- Average of physics and chemistry
  (P + C + M) / 3 = 60 :=  -- Average of all three subjects
by
  sorry


end student_average_score_l3425_342560


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l3425_342500

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- State the theorem
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l3425_342500


namespace inequality_of_reciprocals_l3425_342550

theorem inequality_of_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end inequality_of_reciprocals_l3425_342550


namespace problem_solution_l3425_342557

theorem problem_solution (x y z : ℤ) : 
  x = 12 → y = 18 → z = x - y → z * (x + y) = -180 := by
  sorry

end problem_solution_l3425_342557


namespace green_area_growth_rate_l3425_342585

theorem green_area_growth_rate :
  ∀ x : ℝ, (1 + x)^2 = 1.44 → x = 0.2 :=
by
  sorry

end green_area_growth_rate_l3425_342585


namespace correct_num_footballs_l3425_342568

/-- The number of footballs bought by the school gym -/
def num_footballs : ℕ := 22

/-- The number of basketballs bought by the school gym -/
def num_basketballs : ℕ := 6

/-- Theorem stating that the number of footballs is correct given the conditions -/
theorem correct_num_footballs : 
  (num_footballs = 3 * num_basketballs + 4) ∧ 
  (num_footballs = 4 * num_basketballs - 2) := by
  sorry

#check correct_num_footballs

end correct_num_footballs_l3425_342568


namespace twelve_people_circular_arrangements_l3425_342567

/-- The number of distinct circular arrangements of n people, considering rotational symmetry -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct circular arrangements of 12 people, considering rotational symmetry, is equal to 11! -/
theorem twelve_people_circular_arrangements : 
  circularArrangements 12 = Nat.factorial 11 := by
  sorry

end twelve_people_circular_arrangements_l3425_342567


namespace triangle_altitude_equals_twice_base_l3425_342502

/-- Given a square with side length x and a triangle with base x, 
    if their areas are equal, then the altitude of the triangle is 2x. -/
theorem triangle_altitude_equals_twice_base (x : ℝ) (h : x > 0) : 
  x^2 = (1/2) * x * (2*x) := by sorry

end triangle_altitude_equals_twice_base_l3425_342502


namespace rectangle_area_theorem_l3425_342576

theorem rectangle_area_theorem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 2) = x * y ∧ 
  x * y = 36 := by
  sorry

end rectangle_area_theorem_l3425_342576


namespace school_classes_count_l3425_342515

/-- Proves that the number of classes in a school is 1, given the conditions of the reading program -/
theorem school_classes_count (s : ℕ) (h1 : s > 0) : ∃ c : ℕ,
  (c * s = 1) ∧
  (6 * 12 * (c * s) = 72) :=
by
  sorry

#check school_classes_count

end school_classes_count_l3425_342515


namespace assignment_count_correct_l3425_342509

/-- The number of ways to assign 5 students to 5 universities with exactly 2 visiting Peking University -/
def assignment_count : ℕ := 640

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of universities -/
def num_universities : ℕ := 5

/-- The number of students who should visit Peking University -/
def peking_visitors : ℕ := 2

theorem assignment_count_correct : 
  assignment_count = (num_students.choose peking_visitors) * 
    (num_universities - 1) ^ (num_students - peking_visitors) := by
  sorry

end assignment_count_correct_l3425_342509


namespace unique_prime_solution_l3425_342555

theorem unique_prime_solution :
  ∀ p q : ℕ,
    Prime p → Prime q →
    p^3 - q^5 = (p + q)^2 →
    p = 7 ∧ q = 3 :=
by sorry

end unique_prime_solution_l3425_342555


namespace product_three_consecutive_odds_divisible_by_three_l3425_342523

theorem product_three_consecutive_odds_divisible_by_three (n : ℤ) (h : n > 0) :
  ∃ k : ℤ, (2*n + 1) * (2*n + 3) * (2*n + 5) = 3 * k :=
by
  sorry

end product_three_consecutive_odds_divisible_by_three_l3425_342523


namespace tri_divisible_iff_l3425_342586

/-- A polynomial is tri-divisible if 3 divides f(k) for any integer k -/
def TriDivisible (f : Polynomial ℤ) : Prop :=
  ∀ k : ℤ, (3 : ℤ) ∣ (f.eval k)

/-- The necessary and sufficient condition for a polynomial to be tri-divisible -/
theorem tri_divisible_iff (f : Polynomial ℤ) :
  TriDivisible f ↔ ∃ (Q : Polynomial ℤ) (a b c : ℤ),
    f = (X - 1) * (X - 2) * X * Q + 3 * (a * X^2 + b * X + c) :=
sorry

end tri_divisible_iff_l3425_342586


namespace driver_net_pay_driver_net_pay_result_l3425_342522

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) (gas_price : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let gas_used := total_distance / fuel_efficiency
  let total_earnings := earnings_per_mile * total_distance
  let gas_cost := gas_price * gas_used
  let net_earnings := total_earnings - gas_cost
  let net_rate := net_earnings / travel_time
  net_rate

/-- The driver's net rate of pay is $39.75 per hour --/
theorem driver_net_pay_result : 
  driver_net_pay 3 75 25 0.65 3 = 39.75 := by
  sorry

end driver_net_pay_driver_net_pay_result_l3425_342522


namespace parallelogram_area_parallelogram_area_proof_l3425_342508

/-- The area of a parallelogram with base 12 cm and height 10 cm is 120 cm². -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 12 ∧ height = 10 → area = base * height → area = 120

#check parallelogram_area

-- Proof
theorem parallelogram_area_proof : parallelogram_area 12 10 120 := by
  sorry

end parallelogram_area_parallelogram_area_proof_l3425_342508


namespace car_speed_calculation_l3425_342507

theorem car_speed_calculation (D : ℝ) (h_D_pos : D > 0) : ∃ v : ℝ,
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end car_speed_calculation_l3425_342507


namespace increasing_sequence_condition_l3425_342519

theorem increasing_sequence_condition (a : ℝ) :
  (∀ n : ℕ+, (n : ℝ) - a < ((n + 1) : ℝ) - a) ↔ a < (3 / 2) :=
by sorry

end increasing_sequence_condition_l3425_342519


namespace first_18_even_numbers_average_l3425_342540

/-- The sequence of even numbers -/
def evenSequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => evenSequence n + 2

/-- The sum of the first n terms in the even number sequence -/
def evenSum (n : ℕ) : ℕ :=
  (List.range n).map evenSequence |>.sum

/-- The average of the first n terms in the even number sequence -/
def evenAverage (n : ℕ) : ℚ :=
  evenSum n / n

theorem first_18_even_numbers_average :
  evenAverage 18 = 19 := by
  sorry

end first_18_even_numbers_average_l3425_342540


namespace herd_size_l3425_342582

theorem herd_size (herd : ℕ) : 
  (1 / 3 : ℚ) * herd + (1 / 6 : ℚ) * herd + (1 / 7 : ℚ) * herd + 15 = herd →
  herd = 42 := by
sorry

end herd_size_l3425_342582


namespace petya_vasya_game_l3425_342587

theorem petya_vasya_game (k : ℚ) : ∃ (a b c : ℚ), 
  ∃ (x y : ℂ), x ≠ y ∧ 
  (x^3 + a*x^2 + b*x + c = 0) ∧ 
  (y^3 + a*y^2 + b*y + c = 0) ∧ 
  (x - y = 2014 ∨ y - x = 2014) :=
sorry

end petya_vasya_game_l3425_342587


namespace x_intercept_of_line_l3425_342537

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l3425_342537


namespace cube_volume_ratio_l3425_342584

theorem cube_volume_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 9 / 25) : 
  b^3 / a^3 = 125 / 27 := by
  sorry

end cube_volume_ratio_l3425_342584


namespace ben_peas_count_l3425_342506

/-- The number of sugar snap peas Ben wants to pick initially -/
def total_peas : ℕ := 56

/-- The time it takes Ben to pick all the peas (in minutes) -/
def total_time : ℕ := 7

/-- The number of peas Ben can pick in 9 minutes -/
def peas_in_9_min : ℕ := 72

/-- The time it takes Ben to pick 72 peas (in minutes) -/
def time_for_72_peas : ℕ := 9

/-- Theorem stating that the number of sugar snap peas Ben wants to pick initially is 56 -/
theorem ben_peas_count : 
  (total_peas : ℚ) / total_time = (peas_in_9_min : ℚ) / time_for_72_peas ∧
  total_peas = 56 := by
  sorry


end ben_peas_count_l3425_342506


namespace james_daily_trips_l3425_342541

/-- The number of bags James can carry per trip -/
def bags_per_trip : ℕ := 10

/-- The total number of bags James delivers in 5 days -/
def total_bags : ℕ := 1000

/-- The number of days James works -/
def total_days : ℕ := 5

/-- The number of trips James takes each day -/
def trips_per_day : ℕ := total_bags / (bags_per_trip * total_days)

theorem james_daily_trips : trips_per_day = 20 := by
  sorry

end james_daily_trips_l3425_342541


namespace empty_chests_count_l3425_342574

/-- Represents a nested chest system -/
structure ChestSystem where
  total_chests : ℕ
  non_empty_chests : ℕ
  hNonEmpty : non_empty_chests = 2006
  hTotal : total_chests = 10 * non_empty_chests + 1

/-- The number of empty chests in the system -/
def empty_chests (cs : ChestSystem) : ℕ :=
  cs.total_chests - (cs.non_empty_chests + 1)

/-- Theorem stating the number of empty chests in the given system -/
theorem empty_chests_count (cs : ChestSystem) : empty_chests cs = 18054 := by
  sorry

end empty_chests_count_l3425_342574


namespace triangle_area_ratio_l3425_342563

-- Define the triangle ABC inscribed in a unit circle
def Triangle (A B C : ℝ) := True

-- Define the area of a triangle
def area (a b c : ℝ) : ℝ := sorry

-- Define the sine function
noncomputable def sin (θ : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_ratio 
  (A B C : ℝ) 
  (h : Triangle A B C) :
  area (sin A) (sin B) (sin C) = (1/4 : ℝ) * area (2 * sin A) (2 * sin B) (2 * sin C) :=
sorry

end triangle_area_ratio_l3425_342563


namespace triangle_abc_properties_l3425_342588

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (Real.sin (B - C) + Real.sin A = Real.sin (2 * C)) →
  (A = Real.pi / 3) ∧
  ((1/2 * a * b * Real.sin (Real.pi / 3) = 2 * Real.sqrt 3 / 3) ∨
   (1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3)) :=
by sorry

end triangle_abc_properties_l3425_342588


namespace arithmetic_calculation_l3425_342596

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 35 / 5 = 64 := by
  sorry

end arithmetic_calculation_l3425_342596


namespace system_solution_correct_l3425_342581

theorem system_solution_correct (x y : ℝ) : 
  x = 2 ∧ y = 0 → (x - 2*y = 2 ∧ 2*x + y = 4) :=
by sorry

end system_solution_correct_l3425_342581


namespace henry_tic_tac_toe_wins_l3425_342591

theorem henry_tic_tac_toe_wins 
  (total_games : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) 
  (h1 : total_games = 14) 
  (h2 : losses = 2) 
  (h3 : draws = 10) : 
  total_games - losses - draws = 2 := by
  sorry

end henry_tic_tac_toe_wins_l3425_342591


namespace simplify_expression_l3425_342595

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : 
  8 * x^3 * y / (2 * x)^2 = 2 * x * y :=
by sorry

end simplify_expression_l3425_342595


namespace nine_rings_five_classes_l3425_342575

/-- Represents the number of classes in a school day based on bell rings --/
def number_of_classes (total_rings : ℕ) : ℕ :=
  let completed_classes := (total_rings - 1) / 2
  completed_classes + 1

/-- Theorem stating that 9 total bell rings corresponds to 5 classes --/
theorem nine_rings_five_classes : number_of_classes 9 = 5 := by
  sorry

end nine_rings_five_classes_l3425_342575


namespace arithmetic_square_root_of_sqrt_16_l3425_342580

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l3425_342580


namespace modulus_of_2_minus_i_l3425_342517

theorem modulus_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_2_minus_i_l3425_342517


namespace unique_natural_solution_l3425_342535

theorem unique_natural_solution : 
  ∃! (n : ℕ), n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 :=
by
  sorry

end unique_natural_solution_l3425_342535


namespace dozens_of_eggs_l3425_342510

def eggs_bought : ℕ := 72
def eggs_per_dozen : ℕ := 12

theorem dozens_of_eggs : eggs_bought / eggs_per_dozen = 6 := by
  sorry

end dozens_of_eggs_l3425_342510


namespace max_min_quadratic_function_l3425_342571

theorem max_min_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 2
  let interval : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
  (∀ x ∈ interval, f x ≤ -2) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ x ∈ interval, f x ≥ -6) ∧
  (∃ x ∈ interval, f x = -6) :=
by sorry

end max_min_quadratic_function_l3425_342571


namespace chess_tournament_games_l3425_342530

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 19 players, where each player plays twice against every other player, the total number of games played is 684. -/
theorem chess_tournament_games :
  tournament_games 19 = 342 ∧ 2 * tournament_games 19 = 684 := by
  sorry

#eval 2 * tournament_games 19

end chess_tournament_games_l3425_342530


namespace cubic_polynomials_common_roots_l3425_342566

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 10*r + 3 = 0 ∧
    r^3 + b*r^2 + 21*r + 12 = 0 ∧
    s^3 + a*s^2 + 10*s + 3 = 0 ∧
    s^3 + b*s^2 + 21*s + 12 = 0) →
  a = 9 ∧ b = 10 := by
sorry

end cubic_polynomials_common_roots_l3425_342566


namespace fourth_degree_polynomial_composable_l3425_342528

/-- A fourth-degree polynomial -/
structure FourthDegreePolynomial where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  A_nonzero : A ≠ 0

/-- Condition for a fourth-degree polynomial to be expressible as a composition of two quadratic polynomials -/
def is_composable (f : FourthDegreePolynomial) : Prop :=
  f.D = (f.B * f.C) / (2 * f.A) - (f.B^3) / (8 * f.A^2)

/-- Theorem stating the necessary and sufficient condition for a fourth-degree polynomial 
    to be expressible as a composition of two quadratic polynomials -/
theorem fourth_degree_polynomial_composable (f : FourthDegreePolynomial) :
  (∃ (p q : ℝ → ℝ), (∀ x, f.A * x^4 + f.B * x^3 + f.C * x^2 + f.D * x + f.E = p (q x)) ∧
                     (∃ a b c r s t, p x = a * x^2 + b * x + c ∧
                                     q x = r * x^2 + s * x + t)) ↔
  is_composable f :=
sorry

end fourth_degree_polynomial_composable_l3425_342528


namespace union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3425_342548

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: A ∩ C ≠ ∅ if and only if a < 8
theorem intersection_A_C_nonempty (a : ℝ) : A ∩ C a ≠ ∅ ↔ a < 8 := by sorry

end union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3425_342548


namespace plaid_shirts_count_l3425_342504

/-- Prove that the number of plaid shirts is 3 -/
theorem plaid_shirts_count (total_shirts : ℕ) (total_pants : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ) : ℕ :=
  by
  have total_items : ℕ := total_shirts + total_pants
  have plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
  have plaid_shirts : ℕ := plaid_or_purple - purple_pants
  exact plaid_shirts

#check plaid_shirts_count 5 24 5 21

end plaid_shirts_count_l3425_342504


namespace post_office_mail_handling_l3425_342501

/-- Represents the number of months required for a post office to handle a given amount of mail --/
def months_to_handle_mail (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / ((letters_per_day + packages_per_day) * days_per_month)

/-- Theorem stating that it takes 6 months to handle 14400 pieces of mail given the specified conditions --/
theorem post_office_mail_handling :
  months_to_handle_mail 60 20 30 14400 = 6 := by
  sorry

end post_office_mail_handling_l3425_342501


namespace faucet_filling_time_faucet_filling_time_is_135_l3425_342579

/-- If three faucets can fill a 100-gallon tub in 6 minutes, 
    then four faucets will fill a 50-gallon tub in 135 seconds. -/
theorem faucet_filling_time : ℝ → Prop :=
  fun time_seconds =>
    let three_faucet_volume : ℝ := 100  -- gallons
    let three_faucet_time : ℝ := 6    -- minutes
    let four_faucet_volume : ℝ := 50   -- gallons
    
    let one_faucet_rate : ℝ := three_faucet_volume / (3 * three_faucet_time)
    let four_faucet_rate : ℝ := 4 * one_faucet_rate
    
    time_seconds = (four_faucet_volume / four_faucet_rate) * 60

theorem faucet_filling_time_is_135 : faucet_filling_time 135 := by sorry

end faucet_filling_time_faucet_filling_time_is_135_l3425_342579


namespace exactly_five_false_propositions_l3425_342549

-- Define the geometric objects
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry
def Angle : Type := sorry

-- Define geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 l3 : Line) : Prop := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def commonPoint (pl1 pl2 : Plane) (p : Point) : Prop := sorry
def sidesParallel (a1 a2 : Angle) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 → coplanar l1 l2 l3
def prop3 : Prop := ∀ p1 p2 p3 p4 : Point, (∃ pl : Plane, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl)) → ¬collinear p1 p2 p3 ∧ ¬collinear p1 p2 p4 ∧ ¬collinear p1 p3 p4 ∧ ¬collinear p2 p3 p4
def prop4 : Prop := ∀ pl1 pl2 : Plane, (∃ p1 p2 p3 : Point, commonPoint pl1 pl2 p1 ∧ commonPoint pl1 pl2 p2 ∧ commonPoint pl1 pl2 p3) → pl1 = pl2
def prop5 : Prop := ∃ α β : Plane, ∃! p : Point, commonPoint α β p
def prop6 : Prop := ∀ a1 a2 : Angle, sidesParallel a1 a2 → a1 = a2

-- Theorem statement
theorem exactly_five_false_propositions : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 ∧ ¬prop5 ∧ ¬prop6 := by sorry

end exactly_five_false_propositions_l3425_342549


namespace green_shirt_pairs_l3425_342514

theorem green_shirt_pairs (total_students : ℕ) (red_shirts : ℕ) (green_shirts : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 180 →
  red_shirts = 83 →
  green_shirts = 97 →
  total_pairs = 90 →
  red_red_pairs = 35 →
  red_shirts + green_shirts = total_students →
  2 * total_pairs = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 42 ∧ 
    green_green_pairs + red_red_pairs + (green_shirts - 2 * green_green_pairs) = total_pairs :=
by sorry

end green_shirt_pairs_l3425_342514


namespace bug_triangle_probability_l3425_342505

/-- Probability of the bug being at the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => (1 - P n) / 2

/-- The bug's movement on an equilateral triangle -/
theorem bug_triangle_probability :
  P 12 = 683 / 2048 :=
by sorry

end bug_triangle_probability_l3425_342505


namespace square_difference_divided_by_nine_l3425_342536

theorem square_difference_divided_by_nine : (104^2 - 95^2) / 9 = 199 := by
  sorry

end square_difference_divided_by_nine_l3425_342536


namespace income_percentage_l3425_342583

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.6 * juan) : 
  mart = 0.64 * juan := by
sorry

end income_percentage_l3425_342583


namespace taxi_overtakes_bus_l3425_342545

theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_delay : ℝ) (speed_difference : ℝ)
  (h1 : taxi_speed = 60)
  (h2 : bus_delay = 3)
  (h3 : speed_difference = 30) :
  let bus_speed := taxi_speed - speed_difference
  let overtake_time := (bus_speed * bus_delay) / (taxi_speed - bus_speed)
  overtake_time = 3 := by
sorry

end taxi_overtakes_bus_l3425_342545


namespace binomial_18_4_l3425_342539

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end binomial_18_4_l3425_342539


namespace distance_calculation_l3425_342565

theorem distance_calculation (train_speed ship_speed : ℝ) (time_difference : ℝ) (distance : ℝ) : 
  train_speed = 48 →
  ship_speed = 60 →
  time_difference = 2 →
  distance / train_speed = distance / ship_speed + time_difference →
  distance = 480 := by
sorry

end distance_calculation_l3425_342565


namespace sqrt_expression_equality_l3425_342543

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt 2 * (Real.sqrt 8 - 3 * Real.sqrt (1/2)) = 2 * Real.sqrt 3 - 1 := by
  sorry

end sqrt_expression_equality_l3425_342543


namespace max_train_collection_l3425_342561

/-- The number of trains Max receives each year --/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains --/
def collection_years : ℕ := 5

/-- The total number of trains Max has after the collection period --/
def initial_trains : ℕ := trains_per_year * collection_years

/-- The factor by which Max's train collection is multiplied at the end --/
def doubling_factor : ℕ := 2

/-- The final number of trains Max has --/
def final_trains : ℕ := initial_trains * doubling_factor

theorem max_train_collection :
  final_trains = 30 :=
sorry

end max_train_collection_l3425_342561


namespace not_all_zero_equiv_one_nonzero_l3425_342551

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  (¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end not_all_zero_equiv_one_nonzero_l3425_342551


namespace initial_money_calculation_l3425_342525

theorem initial_money_calculation (initial_amount : ℚ) : 
  (2/5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

#check initial_money_calculation

end initial_money_calculation_l3425_342525


namespace price_reduction_equation_l3425_342556

/-- Represents the price reduction scenario for a medicine -/
structure PriceReduction where
  original_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- Theorem stating the relationship between original price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.original_price = 25)
  (h2 : pr.final_price = 16)
  : pr.original_price * (1 - pr.reduction_percentage)^2 = pr.final_price := by
  sorry

end price_reduction_equation_l3425_342556


namespace min_value_expression_min_value_achievable_l3425_342529

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) ≥ 10 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^3 + 27 * b^3 + 125 * c^3 + 1 / (a * b * c) = 10 * Real.sqrt 6 :=
by sorry

end min_value_expression_min_value_achievable_l3425_342529


namespace shopping_tax_percentage_l3425_342577

theorem shopping_tax_percentage (total : ℝ) (h_total_pos : total > 0) : 
  let clothing_percent : ℝ := 0.50
  let food_percent : ℝ := 0.20
  let other_percent : ℝ := 0.30
  let clothing_tax_rate : ℝ := 0.04
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.08
  let clothing_amount := clothing_percent * total
  let food_amount := food_percent * total
  let other_amount := other_percent * total
  let clothing_tax := clothing_tax_rate * clothing_amount
  let food_tax := food_tax_rate * food_amount
  let other_tax := other_tax_rate * other_amount
  let total_tax := clothing_tax + food_tax + other_tax
  total_tax / total = 0.0440 :=
by sorry

end shopping_tax_percentage_l3425_342577


namespace triangle_inequality_max_l3425_342562

theorem triangle_inequality_max (a b c x y z : ℝ) 
  (triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (positive : 0 < x ∧ 0 < y ∧ 0 < z) 
  (sum_one : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ 
    (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) := by
  sorry

end triangle_inequality_max_l3425_342562


namespace point_position_on_line_l3425_342552

/-- Given five points on a line and a point P satisfying certain conditions, prove the position of P -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P, b ≤ P ∧ P ≤ c →
  (A - P) / (P - D) = (B - P) / (P - C) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end point_position_on_line_l3425_342552


namespace probability_neither_red_nor_purple_l3425_342593

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 15) (h3 : purple = 3) : 
  (total - (red + purple)) / total = 7 / 10 := by
  sorry

end probability_neither_red_nor_purple_l3425_342593


namespace replaced_person_weight_l3425_342590

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (9/2) 101 = 65 := by
  sorry

end replaced_person_weight_l3425_342590


namespace problem_statement_l3425_342546

theorem problem_statement (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 := by sorry

end problem_statement_l3425_342546


namespace fraction_sum_equals_61_30_l3425_342569

theorem fraction_sum_equals_61_30 :
  (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end fraction_sum_equals_61_30_l3425_342569


namespace trapezoid_bases_l3425_342538

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  -- The lengths of the two bases
  base1 : ℝ
  base2 : ℝ
  -- The length of the side
  side : ℝ
  -- The ratio of areas divided by the midline
  areaRatio : ℝ
  -- Conditions
  side_length : side = 3 ∨ side = 5
  inscribable : base1 + base2 = 2 * side
  area_ratio : areaRatio = 5 / 11

/-- The theorem stating the lengths of the bases -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) : 
  (t.base1 = 1 ∧ t.base2 = 7) ∨ (t.base1 = 7 ∧ t.base2 = 1) := by
  sorry

end trapezoid_bases_l3425_342538


namespace book_sale_problem_l3425_342554

theorem book_sale_problem (cost_loss book_loss_price book_gain_price : ℝ) :
  cost_loss = 175 →
  book_loss_price = book_gain_price →
  book_loss_price = 0.85 * cost_loss →
  ∃ cost_gain : ℝ,
    book_gain_price = 1.19 * cost_gain ∧
    cost_loss + cost_gain = 300 :=
by sorry

end book_sale_problem_l3425_342554


namespace complex_expression_1_complex_expression_2_l3425_342578

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_squared : i^2 = -1
axiom i_fourth : i^4 = 1

-- Theorem for the first expression
theorem complex_expression_1 :
  (4 - i^5) * (6 + 2*i^7) + (7 + i^11) * (4 - 3*i) = 57 - 39*i :=
by sorry

-- Theorem for the second expression
theorem complex_expression_2 :
  (5 * (4 + i)^2) / (i * (2 + i)) = -47 - 98*i :=
by sorry

end complex_expression_1_complex_expression_2_l3425_342578


namespace no_real_sqrt_negative_quadratic_l3425_342521

theorem no_real_sqrt_negative_quadratic :
  ¬ ∃ x : ℝ, ∃ y : ℝ, y^2 = -(x^2 + 2*x + 4) := by
  sorry

end no_real_sqrt_negative_quadratic_l3425_342521


namespace percentage_problem_l3425_342532

theorem percentage_problem (P : ℝ) (x : ℝ) : 
  x = 840 → P * x = 0.15 * 1500 - 15 → P = 0.25 := by
sorry

end percentage_problem_l3425_342532


namespace crayon_ratio_l3425_342544

def initial_crayons : ℕ := 18
def new_crayons : ℕ := 20
def total_crayons : ℕ := 29

theorem crayon_ratio :
  (initial_crayons - (total_crayons - new_crayons)) * 2 = initial_crayons :=
sorry

end crayon_ratio_l3425_342544


namespace circle_center_point_is_center_l3425_342598

/-- The center of a circle given by the equation x^2 - 6x + y^2 + 8y - 16 = 0 is (3, -4) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 6*x + y^2 + 8*y - 16 = 0) ↔ ((x - 3)^2 + (y + 4)^2 = 9) :=
by sorry

/-- The point (3, -4) is the center of the circle -/
theorem point_is_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), x^2 - 6*x + y^2 + 8*y - 16 = 0 ↔ (x - 3)^2 + (y + 4)^2 = r^2 :=
by sorry

end circle_center_point_is_center_l3425_342598


namespace negation_of_proposition_l3425_342512

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end negation_of_proposition_l3425_342512


namespace nines_in_sixty_houses_l3425_342564

def count_nines (n : ℕ) : ℕ :=
  (n + 10) / 10

theorem nines_in_sixty_houses :
  count_nines 60 = 6 := by
  sorry

end nines_in_sixty_houses_l3425_342564


namespace trigonometric_roots_problem_l3425_342524

open Real

theorem trigonometric_roots_problem (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ (¬ ∃ (x : ℝ), tan (2*(α + β)) = x) :=
by sorry

end trigonometric_roots_problem_l3425_342524


namespace foot_to_total_distance_ratio_l3425_342526

/-- Proves that the ratio of distance traveled by foot to total distance is 1:4 -/
theorem foot_to_total_distance_ratio :
  let total_distance : ℝ := 40
  let bus_distance : ℝ := total_distance / 2
  let car_distance : ℝ := 10
  let foot_distance : ℝ := total_distance - bus_distance - car_distance
  foot_distance / total_distance = 1 / 4 := by
  sorry

end foot_to_total_distance_ratio_l3425_342526


namespace lightest_box_weight_l3425_342518

/-- Given three boxes with pairwise sums of weights 83 kg, 85 kg, and 86 kg,
    the weight of the lightest box is 41 kg. -/
theorem lightest_box_weight (s m l : ℝ) : 
  s ≤ m ∧ m ≤ l ∧ 
  m + s = 83 ∧ 
  l + s = 85 ∧ 
  l + m = 86 → 
  s = 41 := by
sorry

end lightest_box_weight_l3425_342518
