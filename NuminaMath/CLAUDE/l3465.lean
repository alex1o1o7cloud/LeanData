import Mathlib

namespace min_value_reciprocal_sum_l3465_346581

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 ∧ (1/m + 1/n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end min_value_reciprocal_sum_l3465_346581


namespace complex_fraction_sum_l3465_346556

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) / (1 + Complex.I) = (a : ℂ) + (b : ℂ) * Complex.I → a + b = 1 := by
sorry

end complex_fraction_sum_l3465_346556


namespace f_increasing_neg_f_max_neg_l3465_346554

/-- An odd function that is increasing on [3, 7] with minimum value 5 -/
def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f is increasing on [3, 7] -/
axiom f_increasing_pos : ∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 7 → f x < f y

/-- The minimum value of f on [3, 7] is 5 -/
axiom f_min_pos : ∃ x₀, 3 ≤ x₀ ∧ x₀ ≤ 7 ∧ f x₀ = 5 ∧ ∀ x, 3 ≤ x ∧ x ≤ 7 → f x₀ ≤ f x

/-- f is increasing on [-7, -3] -/
theorem f_increasing_neg : ∀ x y, -7 ≤ x ∧ x < y ∧ y ≤ -3 → f x < f y :=
sorry

/-- The maximum value of f on [-7, -3] is -5 -/
theorem f_max_neg : ∃ x₀, -7 ≤ x₀ ∧ x₀ ≤ -3 ∧ f x₀ = -5 ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ f x₀ :=
sorry

end f_increasing_neg_f_max_neg_l3465_346554


namespace pig_farmer_scenario_profit_l3465_346565

/-- Represents the profit calculation for a pig farmer --/
def pig_farmer_profit (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feed_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feed_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- Theorem stating the profit for the given scenario --/
theorem pig_farmer_scenario_profit :
  pig_farmer_profit 6 300 10 12 16 = 960 :=
sorry

end pig_farmer_scenario_profit_l3465_346565


namespace power_sum_problem_l3465_346579

theorem power_sum_problem : ∃ (x y n : ℕ+), 
  (x * y = 6) ∧ 
  (x ^ n.val + y ^ n.val = 35) ∧ 
  (∀ m : ℕ+, m < n → x ^ m.val + y ^ m.val ≠ 35) := by
  sorry

end power_sum_problem_l3465_346579


namespace min_balls_to_guarantee_color_l3465_346525

theorem min_balls_to_guarantee_color (red green yellow blue white black : ℕ) 
  (h_red : red = 30) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 12) (h_black : black = 10) : 
  ∃ (n : ℕ), n = 95 ∧ 
  (∀ m : ℕ, m < n → 
    ∃ (r g y b w k : ℕ), r < 20 ∧ g < 20 ∧ y < 20 ∧ b < 20 ∧ w < 20 ∧ k < 20 ∧
    r + g + y + b + w + k = m ∧
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black) ∧
  (∀ (r g y b w k : ℕ), r + g + y + b + w + k = n →
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
    r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20 ∨ k ≥ 20) :=
by sorry

end min_balls_to_guarantee_color_l3465_346525


namespace square_circumscribed_circle_radius_l3465_346553

/-- Given a square with perimeter x and circumscribed circle radius y, prove that y = (√2 / 8) * x -/
theorem square_circumscribed_circle_radius (x y : ℝ) 
  (h_perimeter : x > 0) -- Ensure positive perimeter
  (h_square : ∃ (s : ℝ), s > 0 ∧ 4 * s = x) -- Existence of square side length
  (h_circumscribed : y > 0) -- Ensure positive radius
  : y = (Real.sqrt 2 / 8) * x := by
  sorry

end square_circumscribed_circle_radius_l3465_346553


namespace range_of_a_min_value_expression_equality_condition_l3465_346576

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| + |x - 20|

-- Define the property that the solution set is not empty
def solution_set_nonempty (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < 10 * a + 10

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : solution_set_nonempty a → a > 0 := by sorry

-- Theorem for the minimum value of a + 4/a^2
theorem min_value_expression (a : ℝ) (h : a > 0) :
  a + 4 / a^2 ≥ 3 := by sorry

-- Theorem for the equality condition
theorem equality_condition (a : ℝ) (h : a > 0) :
  a + 4 / a^2 = 3 ↔ a = 2 := by sorry

end range_of_a_min_value_expression_equality_condition_l3465_346576


namespace problem_solution_l3465_346575

theorem problem_solution (x : ℝ) (hx_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 := by
  sorry

end problem_solution_l3465_346575


namespace car_travel_time_l3465_346562

/-- Given two cars A and B with specific speeds and distance ratios, 
    prove that Car A takes 6 hours to reach its destination. -/
theorem car_travel_time :
  ∀ (speed_A speed_B time_B distance_A distance_B : ℝ),
  speed_A = 50 →
  speed_B = 100 →
  time_B = 1 →
  distance_A / distance_B = 3 →
  distance_B = speed_B * time_B →
  distance_A = speed_A * (distance_A / speed_A) →
  distance_A / speed_A = 6 := by
sorry

end car_travel_time_l3465_346562


namespace shared_foci_hyperbola_ellipse_l3465_346587

theorem shared_foci_hyperbola_ellipse (a : ℝ) : 
  (∀ x y : ℝ, x^2 / (a + 1) - y^2 = 1 ↔ x^2 / 4 + y^2 / a^2 = 1) →
  a + 1 > 0 →
  4 > a^2 →
  a = 1 :=
by sorry

end shared_foci_hyperbola_ellipse_l3465_346587


namespace salary_increase_l3465_346521

theorem salary_increase (starting_salary current_salary : ℝ) 
  (h1 : starting_salary = 80000)
  (h2 : current_salary = 134400)
  (h3 : current_salary = 1.2 * (starting_salary * 1.4)) :
  starting_salary * 1.4 = starting_salary + 0.4 * starting_salary :=
by sorry

end salary_increase_l3465_346521


namespace new_person_weight_l3465_346586

/-- Given two people, where one weighs 65 kg, if replacing that person with a new person
    increases the average weight by 4.5 kg, then the new person weighs 74 kg. -/
theorem new_person_weight (initial_weight : ℝ) : 
  let total_initial_weight := initial_weight + 65
  let new_average_weight := (total_initial_weight / 2) + 4.5
  let new_total_weight := new_average_weight * 2
  new_total_weight - initial_weight = 74 := by
sorry


end new_person_weight_l3465_346586


namespace dream_car_gas_consumption_l3465_346542

/-- Represents the gas consumption problem for Dream's car -/
theorem dream_car_gas_consumption 
  (gas_per_mile : ℝ) 
  (miles_today : ℝ) 
  (miles_tomorrow : ℝ) 
  (total_gas : ℝ) :
  miles_today = 400 →
  miles_tomorrow = miles_today + 200 →
  total_gas = 4000 →
  gas_per_mile * miles_today + gas_per_mile * miles_tomorrow = total_gas →
  gas_per_mile = 4 := by
sorry

end dream_car_gas_consumption_l3465_346542


namespace solution_set_f_solution_set_g_l3465_346561

-- Define the quadratic functions
def f (x : ℝ) := x^2 - x - 6
def g (x : ℝ) := -2*x^2 + x + 1

-- Theorem for the first inequality
theorem solution_set_f : 
  {x : ℝ | f x > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Theorem for the second inequality
theorem solution_set_g :
  {x : ℝ | g x < 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end solution_set_f_solution_set_g_l3465_346561


namespace car_speed_comparison_l3465_346545

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (2 * u + v) / 3
  x ≤ y := by
sorry

end car_speed_comparison_l3465_346545


namespace pyramid_with_base_six_has_56_apples_l3465_346597

/-- Calculates the number of apples in a triangular layer -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a triangular pyramid stack of apples -/
structure ApplePyramid where
  base_side : ℕ
  inv_base_side_pos : 0 < base_side

/-- Calculates the total number of apples in the pyramid -/
def total_apples (pyramid : ApplePyramid) : ℕ :=
  (List.range pyramid.base_side).map triangular_number |>.sum

/-- The theorem stating that a pyramid with base side 6 contains 56 apples -/
theorem pyramid_with_base_six_has_56_apples :
  ∃ (pyramid : ApplePyramid), pyramid.base_side = 6 ∧ total_apples pyramid = 56 := by
  sorry

end pyramid_with_base_six_has_56_apples_l3465_346597


namespace f_abs_x_is_even_l3465_346514

theorem f_abs_x_is_even (f : ℝ → ℝ) : 
  let g := fun (x : ℝ) ↦ f (|x|)
  ∀ x, g (-x) = g x := by sorry

end f_abs_x_is_even_l3465_346514


namespace negation_equivalence_l3465_346557

/-- The original proposition p -/
def p : Prop := ∀ x : ℝ, x^2 + x - 6 ≤ 0

/-- The proposed negation of p -/
def q : Prop := ∃ x : ℝ, x^2 + x - 6 > 0

/-- Theorem stating that q is the negation of p -/
theorem negation_equivalence : ¬p ↔ q := by
  sorry

end negation_equivalence_l3465_346557


namespace village_new_average_age_l3465_346530

/-- Represents the population data of a village --/
structure VillagePopulation where
  men_ratio : ℚ
  women_ratio : ℚ
  men_increase : ℚ
  men_avg_age : ℚ
  women_avg_age : ℚ

/-- Calculates the new average age of the population after men's population increase --/
def new_average_age (v : VillagePopulation) : ℚ :=
  let new_men_ratio := v.men_ratio * (1 + v.men_increase)
  let total_population := new_men_ratio + v.women_ratio
  let total_age := new_men_ratio * v.men_avg_age + v.women_ratio * v.women_avg_age
  total_age / total_population

/-- Theorem stating that the new average age is approximately 37.3 years --/
theorem village_new_average_age :
  let v : VillagePopulation := {
    men_ratio := 3,
    women_ratio := 4,
    men_increase := 1/10,
    men_avg_age := 40,
    women_avg_age := 35
  }
  ∃ ε > 0, |new_average_age v - 37.3| < ε :=
sorry

end village_new_average_age_l3465_346530


namespace percentage_theorem_l3465_346568

theorem percentage_theorem (y x z : ℝ) (h : y * x^2 + 3 * z - 6 > 0) :
  ((2 * (y * x^2 + 3 * z - 6)) / 5 + (3 * (y * x^2 + 3 * z - 6)) / 10) / (y * x^2 + 3 * z - 6) * 100 = 70 := by
  sorry

end percentage_theorem_l3465_346568


namespace inequality_solution_set_l3465_346560

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 2| + |2*x + 4| < 10) ↔ (x > -4 ∧ x < 2) := by
sorry

end inequality_solution_set_l3465_346560


namespace complex_equation_solution_l3465_346516

theorem complex_equation_solution : ∃ (x : ℂ), 5 + 2 * Complex.I * x = -3 - 6 * Complex.I * x ∧ x = Complex.I := by
  sorry

end complex_equation_solution_l3465_346516


namespace function_satisfies_equation_l3465_346522

def f (n : ℤ) : ℤ := -2 * n + 3

theorem function_satisfies_equation :
  ∀ a b : ℤ, f (a + b) + f (a^2 + b^2) = f a * f b + 2 :=
by
  sorry

end function_satisfies_equation_l3465_346522


namespace smallest_number_in_sequence_l3465_346529

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 29 →                 -- Median is 29
  c = b + 7 →              -- Largest number is 7 more than median
  a < b ∧ b < c →          -- Ensuring order: a < b < c
  a = 25 :=                -- Smallest number is 25
by sorry

end smallest_number_in_sequence_l3465_346529


namespace complex_expression_equals_nine_l3465_346523

theorem complex_expression_equals_nine :
  (Real.sqrt 2 - 3) ^ (0 : ℝ) - Real.sqrt 9 + |(-2 : ℝ)| + (-1/3 : ℝ) ^ (-2 : ℝ) = 9 := by
  sorry

end complex_expression_equals_nine_l3465_346523


namespace triangle_perimeter_l3465_346598

/-- Given a right-angled triangle PQR with R at the right angle, 
    PR = 4000, and PQ = 5000, prove that PQ + QR + PR = 16500 -/
theorem triangle_perimeter (PR PQ QR : ℝ) : 
  PR = 4000 → 
  PQ = 5000 → 
  QR^2 = PQ^2 - PR^2 → 
  PQ + QR + PR = 16500 := by
  sorry

end triangle_perimeter_l3465_346598


namespace compare_fractions_l3465_346599

theorem compare_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / (1 + x + y) < x / (1 + x) + y / (1 + y) := by
  sorry

end compare_fractions_l3465_346599


namespace f_increasing_iff_a_nonpositive_l3465_346548

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem f_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 := by sorry

end f_increasing_iff_a_nonpositive_l3465_346548


namespace at_least_three_lines_intersect_l3465_346519

/-- A line that divides a square into two quadrilaterals -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_point : Point

/-- A square with dividing lines -/
structure DividedSquare where
  side_length : ℝ
  dividing_lines : List DividingLine

/-- The theorem statement -/
theorem at_least_three_lines_intersect (square : DividedSquare) :
  square.side_length > 0 ∧
  square.dividing_lines.length = 9 ∧
  (∀ l ∈ square.dividing_lines, l.divides_square ∧ l.area_ratio = 2 / 3) →
  ∃ p : Point, (square.dividing_lines.filter (λ l => l.intersects_point = p)).length ≥ 3 :=
sorry

end at_least_three_lines_intersect_l3465_346519


namespace point_movement_l3465_346535

/-- Given a point P at (-1, 2), moving it 2 units left and 1 unit up results in point M at (-3, 3) -/
theorem point_movement :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (P.1 - 2, P.2 + 1)
  M = (-3, 3) := by sorry

end point_movement_l3465_346535


namespace full_price_revenue_l3465_346532

/-- Represents a concert ticket sale scenario -/
structure ConcertSale where
  fullPrice : ℕ  -- Number of full-price tickets
  discountPrice : ℕ  -- Number of discount-price tickets
  price : ℕ  -- Price of a full-price ticket in dollars

/-- Conditions for a valid concert sale -/
def isValidSale (sale : ConcertSale) : Prop :=
  sale.fullPrice + sale.discountPrice = 200 ∧
  sale.fullPrice * sale.price + sale.discountPrice * (sale.price / 3) = 3000

/-- Theorem stating the revenue from full-price tickets -/
theorem full_price_revenue (sale : ConcertSale) 
  (h : isValidSale sale) : sale.fullPrice * sale.price = 1500 := by
  sorry


end full_price_revenue_l3465_346532


namespace age_ratio_after_five_years_l3465_346513

/-- Theorem: Ratio of parent's age to son's age after 5 years -/
theorem age_ratio_after_five_years
  (parent_age : ℕ)
  (son_age : ℕ)
  (h1 : parent_age = 45)
  (h2 : son_age = 15) :
  (parent_age + 5) / (son_age + 5) = 5 / 2 := by
  sorry

end age_ratio_after_five_years_l3465_346513


namespace unique_prime_between_squares_l3465_346566

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 8 = m^2 ∧ 
  m = n + 1 := by
  sorry

end unique_prime_between_squares_l3465_346566


namespace opposite_of_two_l3465_346591

theorem opposite_of_two : -(2 : ℝ) = -2 := by sorry

end opposite_of_two_l3465_346591


namespace jogging_duration_sum_l3465_346507

/-- The duration in minutes between 5 p.m. and 6 p.m. -/
def total_duration : ℕ := 60

/-- The probability of one friend arriving while the other is jogging -/
def meeting_probability : ℚ := 1/2

/-- Represents the duration each friend stays for jogging -/
structure JoggingDuration where
  x : ℕ
  y : ℕ
  z : ℕ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  z_not_perfect_square : ∀ (p : ℕ), Prime p → ¬(p^2 ∣ z)
  duration_eq : (x : ℚ) - y * Real.sqrt z = total_duration - total_duration * Real.sqrt 2

theorem jogging_duration_sum (d : JoggingDuration) : d.x + d.y + d.z = 92 := by
  sorry

end jogging_duration_sum_l3465_346507


namespace ellipse_intersection_relation_l3465_346524

/-- Theorem: Relationship between y-coordinates of intersection points on an ellipse --/
theorem ellipse_intersection_relation (a b m : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  a > b ∧ b > 0 ∧ m > a →  -- Conditions on a, b, and m
  (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧  -- A is on the ellipse
  (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧  -- B is on the ellipse
  (∃ k : ℝ, x₁ = k * y₁ + m ∧ x₂ = k * y₂ + m) →  -- A and B are on a line through M
  x₃ = a^2 / m ∧ x₄ = a^2 / m →  -- P and Q are on the line x = a^2/m
  (y₃ * (x₁ + a) = y₁ * (x₃ + a)) ∧  -- P is on line A₁A
  (y₄ * (x₂ + a) = y₂ * (x₄ + a)) →  -- Q is on line A₁B
  1 / y₁ + 1 / y₂ = 1 / y₃ + 1 / y₄ :=
by sorry


end ellipse_intersection_relation_l3465_346524


namespace exists_valid_division_l3465_346541

/-- A tiling of a 6x6 board with 2x1 dominos -/
def Tiling := Fin 6 → Fin 6 → Fin 18

/-- A division of the board into two rectangles -/
structure Division where
  horizontal : Bool
  position : Fin 6

/-- Checks if a domino crosses the dividing line -/
def crossesDivision (t : Tiling) (d : Division) : Prop :=
  ∃ (i j : Fin 6), 
    (d.horizontal ∧ i = d.position ∧ t i j = t (i + 1) j) ∨
    (¬d.horizontal ∧ j = d.position ∧ t i j = t i (j + 1))

/-- The main theorem -/
theorem exists_valid_division (t : Tiling) : 
  ∃ (d : Division), ¬crossesDivision t d := by
  sorry

end exists_valid_division_l3465_346541


namespace square_binomial_k_l3465_346573

theorem square_binomial_k (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) → k = 100 := by
sorry

end square_binomial_k_l3465_346573


namespace birds_remaining_after_week_l3465_346585

/-- Calculates the number of birds remaining after a week given initial counts and daily losses. -/
def birdsRemaining (initialChickens initialTurkeys initialGuineaFowls : ℕ)
  (oddDayLossChickens oddDayLossTurkeys oddDayLossGuineaFowls : ℕ)
  (evenDayLossChickens evenDayLossTurkeys evenDayLossGuineaFowls : ℕ) : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let remainingChickens := initialChickens - (oddDays * oddDayLossChickens + evenDays * evenDayLossChickens)
  let remainingTurkeys := initialTurkeys - (oddDays * oddDayLossTurkeys + evenDays * evenDayLossTurkeys)
  let remainingGuineaFowls := initialGuineaFowls - (oddDays * oddDayLossGuineaFowls + evenDays * evenDayLossGuineaFowls)
  remainingChickens + remainingTurkeys + remainingGuineaFowls

/-- Theorem stating that given the initial bird counts and daily losses, 379 birds remain after a week. -/
theorem birds_remaining_after_week :
  birdsRemaining 300 200 80 20 8 5 15 5 3 = 379 := by sorry

end birds_remaining_after_week_l3465_346585


namespace cubic_equation_solutions_l3465_346574

theorem cubic_equation_solutions :
  ∀ x y : ℤ, x^3 = y^3 + 2*y^2 + 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -2 ∧ y = -3) :=
by sorry

end cubic_equation_solutions_l3465_346574


namespace f_has_minimum_at_6_l3465_346571

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 32

/-- Theorem stating that f has a minimum at x = 6 -/
theorem f_has_minimum_at_6 : 
  ∀ x : ℝ, f x ≥ f 6 := by sorry

end f_has_minimum_at_6_l3465_346571


namespace units_digit_of_n_l3465_346501

/-- Given two natural numbers m and n, returns true if m has units digit 9 -/
def has_units_digit_9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given two natural numbers m and n, returns true if their product equals 31^6 -/
def product_equals_31_pow_6 (m n : ℕ) : Prop :=
  m * n = 31^6

/-- Theorem stating that if m has units digit 9 and m * n = 31^6, then n has units digit 9 -/
theorem units_digit_of_n (m n : ℕ) 
  (h1 : has_units_digit_9 m) 
  (h2 : product_equals_31_pow_6 m n) : 
  n % 10 = 9 := by
  sorry

end units_digit_of_n_l3465_346501


namespace mary_card_count_l3465_346594

/-- The number of Pokemon cards Mary has after receiving gifts from Sam and Alex -/
def final_card_count (initial_cards torn_cards sam_gift alex_gift : ℕ) : ℕ :=
  initial_cards - torn_cards + sam_gift + alex_gift

/-- Theorem stating that Mary has 196 Pokemon cards after the events described -/
theorem mary_card_count : 
  final_card_count 123 18 56 35 = 196 := by
  sorry

end mary_card_count_l3465_346594


namespace complement_M_intersect_N_l3465_346536

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N : (U \ M) ∩ N = {0, 4} := by
  sorry

end complement_M_intersect_N_l3465_346536


namespace triangle_ratio_l3465_346531

theorem triangle_ratio (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end triangle_ratio_l3465_346531


namespace intersection_locus_l3465_346503

-- Define the two lines as functions of t
def line1 (x y t : ℝ) : Prop := 2 * x + 3 * y = t
def line2 (x y t : ℝ) : Prop := 5 * x - 7 * y = t

-- Define the locus line
def locusLine (x y : ℝ) : Prop := y = 0.3 * x

-- Theorem statement
theorem intersection_locus :
  ∀ (t : ℝ), ∃ (x y : ℝ), line1 x y t ∧ line2 x y t → locusLine x y :=
by sorry

end intersection_locus_l3465_346503


namespace three_numbers_product_l3465_346592

theorem three_numbers_product (x y z : ℤ) : 
  x + y + z = 165 ∧ 
  7 * x = y - 9 ∧ 
  7 * x = z + 9 → 
  x * y * z = 64328 := by
  sorry

end three_numbers_product_l3465_346592


namespace arrangement_problem_l3465_346564

def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem arrangement_problem (n_boys n_girls : ℕ) (h_boys : n_boys = 6) (h_girls : n_girls = 4) :
  -- (I) Girls standing together
  (A n_girls n_girls * A (n_boys + 1) (n_boys + 1) = A 4 4 * A 7 7) ∧
  -- (II) No two girls adjacent
  (A n_boys n_boys * A (n_boys + 1) n_girls = A 6 6 * A 7 4) ∧
  -- (III) Boys A, B, C in alphabetical order
  (A (n_boys + n_girls) (n_boys + n_girls - 3) = A 10 7) :=
sorry

end arrangement_problem_l3465_346564


namespace power_of_i_2023_l3465_346546

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_2023 : i^2023 = -i := by sorry

end power_of_i_2023_l3465_346546


namespace min_value_implies_a_inequality_implies_m_range_l3465_346528

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 1 ∨ a = -1 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-1) 1, m^2 - |m| - f x a < 0) → -2 < m ∧ m < 2 :=
sorry

end min_value_implies_a_inequality_implies_m_range_l3465_346528


namespace base8_to_base5_conversion_l3465_346517

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- The number 427 in base 8 -/
def num_base8 : ℕ := 427

/-- The number 2104 in base 5 -/
def num_base5 : ℕ := 2104

theorem base8_to_base5_conversion :
  base10ToBase5 (base8ToBase10 num_base8) = num_base5 := by sorry

end base8_to_base5_conversion_l3465_346517


namespace main_project_time_l3465_346539

def total_days : ℕ := 4
def hours_per_day : ℝ := 8
def time_on_smaller_tasks : ℝ := 9
def time_on_naps : ℝ := 13.5

theorem main_project_time :
  total_days * hours_per_day - time_on_smaller_tasks - time_on_naps = 9.5 := by
sorry

end main_project_time_l3465_346539


namespace octal_74532_to_decimal_l3465_346549

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_74532_to_decimal :
  octal_to_decimal [2, 3, 5, 4, 7] = 31066 := by
  sorry

end octal_74532_to_decimal_l3465_346549


namespace equation_solution_l3465_346577

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l3465_346577


namespace local_minimum_implies_c_eq_two_l3465_346505

/-- The function f(x) -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x * (x - c)^2 + 3

/-- Theorem: If f(x) has a local minimum at x = 2, then c = 2 -/
theorem local_minimum_implies_c_eq_two (c : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≥ f c 2) →
  c = 2 :=
by sorry

end local_minimum_implies_c_eq_two_l3465_346505


namespace opposite_of_negative_half_l3465_346512

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end opposite_of_negative_half_l3465_346512


namespace sum_inequality_l3465_346583

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 + 2*b^2 + 3) + 1 / (b^2 + 2*c^2 + 3) + 1 / (c^2 + 2*a^2 + 3) ≤ 1/2 := by
  sorry

end sum_inequality_l3465_346583


namespace function_properties_l3465_346518

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - 1) - a

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -(abs (x + m))

/-- The statement that g(x) > -1 has exactly one integer solution, which is -3 -/
def has_unique_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), g m (n : ℝ) > -1 ∧ n = -3

theorem function_properties (a m : ℝ) 
  (h_unique : has_unique_integer_solution m) :
  m = 3 ∧ (∀ x, f a x > g m x) → a < 4 := by sorry

end function_properties_l3465_346518


namespace perfect_square_base9_l3465_346526

/-- Represents a number in base 9 of the form ac7b -/
structure Base9Number where
  a : ℕ
  c : ℕ
  b : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.c + 63 + n.b

/-- Theorem stating that if a Base9Number is a perfect square, then b must be 0 -/
theorem perfect_square_base9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.b = 0 := by
  sorry

end perfect_square_base9_l3465_346526


namespace sum_squares_ge_product_sum_l3465_346550

theorem sum_squares_ge_product_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end sum_squares_ge_product_sum_l3465_346550


namespace distinct_lines_theorem_l3465_346547

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to determine if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Function to create a line from two points -/
def line_from_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.y * q.x - p.x * q.y }

/-- Function to check if two lines are distinct -/
def distinct_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a ∨ l1.a * l2.c ≠ l1.c * l2.a ∨ l1.b * l2.c ≠ l1.c * l2.b

/-- Theorem: For n points on a plane, not all collinear, there are at least n distinct lines -/
theorem distinct_lines_theorem (n : ℕ) (points : Fin n → Point) 
  (h : ∃ i j k : Fin n, ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line), ∀ i j : Fin n, i ≠ j → distinct_lines (lines i) (lines j) :=
sorry

end distinct_lines_theorem_l3465_346547


namespace family_composition_l3465_346504

theorem family_composition :
  ∀ (boys girls : ℕ),
  (boys > 0 ∧ girls > 0) →
  (boys - 1 = girls) →
  (boys = 2 * (girls - 1)) →
  (boys = 4 ∧ girls = 3) :=
by sorry

end family_composition_l3465_346504


namespace ratio_HC_JE_l3465_346569

-- Define the points
variable (A B C D E F G H J K : ℝ × ℝ)

-- Define the conditions
axiom points_on_line : ∃ (t : ℝ), A = (0, 0) ∧ B = (1, 0) ∧ C = (3, 0) ∧ D = (4, 0) ∧ E = (5, 0) ∧ F = (7, 0)
axiom G_off_line : G.2 ≠ 0
axiom H_on_GD : ∃ (t : ℝ), H = G + t • (D - G)
axiom J_on_GF : ∃ (t : ℝ), J = G + t • (F - G)
axiom K_on_GB : ∃ (t : ℝ), K = G + t • (B - G)
axiom parallel_lines : ∃ (k : ℝ), 
  H - C = k • (G - A) ∧ 
  J - E = k • (G - A) ∧ 
  K - B = k • (G - A)

-- Define the theorem
theorem ratio_HC_JE : 
  (H.1 - C.1) / (J.1 - E.1) = 7/8 :=
sorry

end ratio_HC_JE_l3465_346569


namespace sequence_proof_l3465_346508

def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def geometric_sequence (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f (n + 1) / f n = f 2 / f 1

def sum_sequence (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_sequence f n + f (n + 1)

theorem sequence_proof 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : arithmetic_sequence a b c)
  (h_sum : a + b + c = 15)
  (b_n : ℕ → ℝ)
  (h_geometric : geometric_sequence (λ n => b_n (n + 2)))
  (h_relation : b_n 3 = a + 2 ∧ b_n 4 = b + 5 ∧ b_n 5 = c + 13) :
  (∀ n : ℕ, b_n n = (5/4) * 2^(n-1)) ∧
  (geometric_sequence (λ n => sum_sequence b_n n + 5/4) ∧
   (sum_sequence b_n 1 + 5/4 = 5/2) ∧
   (∀ n : ℕ, (sum_sequence b_n (n+1) + 5/4) / (sum_sequence b_n n + 5/4) = 2)) :=
sorry

end sequence_proof_l3465_346508


namespace M_subset_N_l3465_346590

/-- Set M definition -/
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

/-- Set N definition -/
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

/-- Theorem stating that M is a subset of N -/
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l3465_346590


namespace upstream_speed_calculation_mans_upstream_speed_is_twelve_l3465_346580

/-- Calculates the speed of a man rowing upstream given his speed in still water and his speed downstream. -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a given man's speed in still water and downstream, 
    his upstream speed is equal to twice his still water speed minus his downstream speed. -/
theorem upstream_speed_calculation 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0)
  (h2 : speed_downstream > speed_still) :
  speed_upstream speed_still speed_downstream = 2 * speed_still - speed_downstream :=
by sorry

/-- The speed of the man rowing upstream in the given problem. -/
def mans_upstream_speed : ℝ := speed_upstream 25 38

/-- Theorem proving that the man's upstream speed in the given problem is 12 km/h. -/
theorem mans_upstream_speed_is_twelve : 
  mans_upstream_speed = 12 :=
by sorry

end upstream_speed_calculation_mans_upstream_speed_is_twelve_l3465_346580


namespace quadratic_inequality_solution_l3465_346555

/-- Given that the solution set of ax² + bx + 1 > 0 is {x | -1 < x < 1/3}, prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 := by
sorry

end quadratic_inequality_solution_l3465_346555


namespace walnut_trees_planted_l3465_346588

theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 22)
  (h2 : final_trees = 55) :
  final_trees - initial_trees = 33 := by
  sorry

end walnut_trees_planted_l3465_346588


namespace stack_height_problem_l3465_346559

/-- Calculates the total height of a stack of discs with a cylindrical item on top -/
def total_height (top_diameter : ℕ) (bottom_diameter : ℕ) (disc_thickness : ℕ) (cylinder_height : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  let discs_height := num_discs * disc_thickness
  discs_height + cylinder_height

/-- The problem statement -/
theorem stack_height_problem :
  let top_diameter := 15
  let bottom_diameter := 1
  let disc_thickness := 2
  let cylinder_height := 10
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = 26 := by
  sorry

end stack_height_problem_l3465_346559


namespace rectangular_solid_diagonal_l3465_346509

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24)
  (h2 : 4 * (a + b + c) = 28) :
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end rectangular_solid_diagonal_l3465_346509


namespace triangle_count_in_specific_rectangle_l3465_346584

/-- Represents a rectangle divided by vertical and horizontal lines -/
structure DividedRectangle where
  vertical_divisions : ℕ
  horizontal_divisions : ℕ

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  let small_rectangles := r.vertical_divisions * r.horizontal_divisions
  let smallest_triangles := small_rectangles * 4
  let isosceles_by_width := small_rectangles
  let large_right_triangles := small_rectangles * 2
  let largest_isosceles := r.horizontal_divisions
  smallest_triangles + isosceles_by_width + large_right_triangles + largest_isosceles

/-- Theorem stating that a rectangle divided by 3 vertical and 2 horizontal lines contains 50 triangles -/
theorem triangle_count_in_specific_rectangle :
  let r : DividedRectangle := ⟨3, 2⟩
  count_triangles r = 50 := by
  sorry

end triangle_count_in_specific_rectangle_l3465_346584


namespace seating_chart_interpretation_l3465_346506

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpretSeatingChart (pair : ℕ × ℕ) : SeatingChart :=
  ⟨pair.1, pair.2⟩

theorem seating_chart_interpretation :
  let chart := interpretSeatingChart (5, 4)
  chart.columns = 5 ∧ chart.rows = 4 := by
  sorry

end seating_chart_interpretation_l3465_346506


namespace sum_of_digits_power_product_l3465_346520

def power_product (a b c : ℕ) : ℕ := a^2010 * b^2012 * c

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits (power_product 2 5 7) = 13 := by sorry

end sum_of_digits_power_product_l3465_346520


namespace pattern_equality_l3465_346563

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The theorem stating the equality of the observed pattern -/
theorem pattern_equality (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

#check pattern_equality

end pattern_equality_l3465_346563


namespace power_sum_inequality_l3465_346510

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_pyth : a^2 + b^2 = c^2) (hn : n > 2) : a^n + b^n < c^n := by
  sorry

end power_sum_inequality_l3465_346510


namespace l_companion_properties_l3465_346533

/-- Definition of an l-companion function -/
def is_l_companion (f : ℝ → ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ Continuous f ∧ ∀ x : ℝ, f (x + l) + l * f x = 0

theorem l_companion_properties (f : ℝ → ℝ) (l : ℝ) (h : is_l_companion f l) :
  (∀ c : ℝ, is_l_companion (λ _ => c) l → c = 0) ∧
  ¬ is_l_companion (λ x => x) l ∧
  ¬ is_l_companion (λ x => x^2) l ∧
  ∃ x : ℝ, f x = 0 :=
by sorry

end l_companion_properties_l3465_346533


namespace sum_range_l3465_346558

theorem sum_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) 
  (h : a^2 - a + b^2 - b + a*b = 0) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end sum_range_l3465_346558


namespace west_east_correspondence_l3465_346582

-- Define a type for directions
inductive Direction
| East
| West

-- Define a function to represent distance with direction
def distance_with_direction (d : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.East => d
  | Direction.West => -d

-- State the theorem
theorem west_east_correspondence :
  (distance_with_direction 2023 Direction.West = -2023) →
  (distance_with_direction 2023 Direction.East = 2023) :=
by
  sorry

end west_east_correspondence_l3465_346582


namespace sum_of_divisors_l3465_346502

def isPrime (n : ℕ) : Prop := sorry

def numDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors (p q r : ℕ) (hp : isPrime p) (hq : isPrime q) (hr : isPrime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  let a := p^4
  let b := q * r
  let k := a^5
  let m := b^2
  numDivisors k + numDivisors m = 30 := by sorry

end sum_of_divisors_l3465_346502


namespace cake_recipe_proof_l3465_346551

def baking_problem (total_flour sugar_needed flour_added : ℕ) : Prop :=
  total_flour - flour_added - sugar_needed = 5

theorem cake_recipe_proof :
  baking_problem 10 3 2 := by sorry

end cake_recipe_proof_l3465_346551


namespace min_value_quadratic_sum_l3465_346511

theorem min_value_quadratic_sum (a b c d : ℝ) (h : a * d + b * c = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = a^2 + b^2 + c^2 + d^2 + (a + c)^2 + (b - d)^2 → u ≥ min_val :=
by sorry

end min_value_quadratic_sum_l3465_346511


namespace binary_110101_equals_53_l3465_346552

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

/-- Theorem stating that 110101₂ equals 53 in decimal -/
theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end binary_110101_equals_53_l3465_346552


namespace inverse_proportion_point_value_l3465_346572

/-- Prove that for an inverse proportion function y = k/x (k ≠ 0),
    if points A(2,m) and B(m,n) lie on its graph, then n = 2. -/
theorem inverse_proportion_point_value (k m n : ℝ) : 
  k ≠ 0 → m = k / 2 → n = k / m → n = 2 := by
  sorry

end inverse_proportion_point_value_l3465_346572


namespace least_bench_sections_l3465_346570

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k < M → ¬(120 ∣ 8*k ∧ 120 ∣ 12*k ∧ 120 ∣ 10*k)) ∧ 
  (120 ∣ 8*M ∧ 120 ∣ 12*M ∧ 120 ∣ 10*M) → M = 15 := by
  sorry

end least_bench_sections_l3465_346570


namespace license_plate_theorem_l3465_346500

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels (including Y) -/
def vowel_count : ℕ := 6

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible three-character license plates with two consonants followed by a vowel -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count

theorem license_plate_theorem : license_plate_count = 2400 := by
  sorry

end license_plate_theorem_l3465_346500


namespace intersection_A_B_l3465_346543

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end intersection_A_B_l3465_346543


namespace inequality_solution_set_quadratic_inequality_range_l3465_346593

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  9 / (x + 4) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici (1/2) :=
sorry

-- Part 2
theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → 
  k > Real.sqrt 2 ∨ k < -Real.sqrt 2 :=
sorry

end inequality_solution_set_quadratic_inequality_range_l3465_346593


namespace floor_product_equals_17_l3465_346578

def solution_set : Set ℝ := Set.Ici 4.25 ∩ Set.Iio 4.5

theorem floor_product_equals_17 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 17 ↔ x ∈ solution_set := by sorry

end floor_product_equals_17_l3465_346578


namespace highway_vehicles_l3465_346567

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end highway_vehicles_l3465_346567


namespace remainder_444_power_444_mod_13_l3465_346537

theorem remainder_444_power_444_mod_13 : 444^444 ≡ 1 [MOD 13] := by
  sorry

end remainder_444_power_444_mod_13_l3465_346537


namespace constant_in_toll_formula_l3465_346595

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 1.50 * (x - 2)

/-- The number of axles on an 18-wheel truck with 2 wheels on the front axle and 4 wheels on each other axle -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for the 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 6

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 1.50 := by
  sorry

end constant_in_toll_formula_l3465_346595


namespace milk_drinking_l3465_346544

theorem milk_drinking (total_milk : ℚ) (drunk_fraction : ℚ) : 
  total_milk = 1/4 → drunk_fraction = 3/4 → drunk_fraction * total_milk = 3/16 := by
  sorry

end milk_drinking_l3465_346544


namespace sum_equals_three_or_seven_l3465_346527

theorem sum_equals_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  x + y + z = 3 ∨ x + y + z = 7 := by
sorry

end sum_equals_three_or_seven_l3465_346527


namespace friends_with_oranges_l3465_346540

theorem friends_with_oranges (total_friends : ℕ) (friends_with_pears : ℕ) : 
  total_friends = 15 → friends_with_pears = 9 → total_friends - friends_with_pears = 6 := by
  sorry

end friends_with_oranges_l3465_346540


namespace prime_factors_count_l3465_346515

theorem prime_factors_count (p q r : ℕ) (h1 : p = 4) (h2 : q = 7) (h3 : r = 11) 
  (h4 : p = 2^2) (h5 : Nat.Prime q) (h6 : Nat.Prime r) : 
  (Nat.factors (p^11 * q^7 * r^2)).length = 31 := by
sorry

end prime_factors_count_l3465_346515


namespace find_divisor_l3465_346534

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 269 * D) (h2 : N % 67 = 1) : D = 1 := by
  sorry

end find_divisor_l3465_346534


namespace twenty_fifth_digit_sum_l3465_346589

/-- The decimal representation of 1/9 -/
def decimal_1_9 : ℚ := 1/9

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/9 and 1/11 -/
def sum_decimals : ℚ := decimal_1_9 + decimal_1_11

/-- The 25th digit after the decimal point in a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem twenty_fifth_digit_sum :
  nth_digit_after_decimal sum_decimals 25 = 2 := by sorry

end twenty_fifth_digit_sum_l3465_346589


namespace min_distance_between_curves_l3465_346538

theorem min_distance_between_curves (a b c d : ℝ) :
  (a - 2*Real.exp a)/b = (1 - c)/(d - 1) ∧ (a - 2*Real.exp a)/b = 1 →
  (∀ x y z w : ℝ, (x - 2*Real.exp x)/y = (1 - z)/(w - 1) ∧ (x - 2*Real.exp x)/y = 1 →
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 :=
by sorry

end min_distance_between_curves_l3465_346538


namespace marathon_remainder_yards_l3465_346596

/-- Represents the length of a marathon in miles and yards -/
structure Marathon :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a distance in miles and yards -/
structure Distance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_length : Marathon :=
  { miles := 30, yards := 520 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 8

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), y < yards_per_mile ∧
    Distance.yards (
      { miles := m
      , yards := y
      } : Distance
    ) = 640 ∧
    Distance.miles (
      { miles := m
      , yards := y
      } : Distance
    ) * yards_per_mile + y =
    num_marathons * (marathon_length.miles * yards_per_mile + marathon_length.yards) :=
by sorry

end marathon_remainder_yards_l3465_346596
