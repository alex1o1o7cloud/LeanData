import Mathlib

namespace sons_age_is_fourteen_l1957_195752

/-- Proves that given the conditions, the son's present age is 14 years -/
theorem sons_age_is_fourteen (son_age father_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
  sorry

end sons_age_is_fourteen_l1957_195752


namespace sphere_volume_circumscribing_cube_l1957_195756

/-- The volume of a sphere circumscribing a cube with edge length 2 cm -/
theorem sphere_volume_circumscribing_cube : 
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end sphere_volume_circumscribing_cube_l1957_195756


namespace decimal_to_fraction_l1957_195740

theorem decimal_to_fraction : 
  (3.36 : ℚ) = 84 / 25 := by sorry

end decimal_to_fraction_l1957_195740


namespace student_calculation_error_l1957_195715

theorem student_calculation_error (x y : ℝ) : 
  (5/4 : ℝ) * x = (4/5 : ℝ) * x + 36 ∧ 
  (7/3 : ℝ) * y = (3/7 : ℝ) * y + 28 → 
  x = 80 ∧ y = 14.7 := by
sorry

end student_calculation_error_l1957_195715


namespace rationalize_denominator_l1957_195767

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
by sorry

end rationalize_denominator_l1957_195767


namespace distance_to_origin_l1957_195793

/-- Given that point A has coordinates (√3, 2, 5) and its projection on the x-axis is (√3, 0, 0),
    prove that the distance from A to the origin is 4√2. -/
theorem distance_to_origin (A : ℝ × ℝ × ℝ) (h : A = (Real.sqrt 3, 2, 5)) :
  Real.sqrt ((Real.sqrt 3)^2 + 2^2 + 5^2) = 4 * Real.sqrt 2 := by
sorry

end distance_to_origin_l1957_195793


namespace roots_have_different_signs_l1957_195725

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
def quadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_have_different_signs (a b c : ℝ) (ha : a ≠ 0) :
  (quadraticPolynomial a b c (1/a)) * (quadraticPolynomial a b c c) < 0 →
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ ∀ x, quadraticPolynomial a b c x = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

end roots_have_different_signs_l1957_195725


namespace steak_weight_for_tommy_family_l1957_195717

/-- Given a family where each member wants one pound of steak, 
    this function calculates the weight of each steak needed to be purchased. -/
def steak_weight (family_size : ℕ) (num_steaks : ℕ) : ℚ :=
  (family_size : ℚ) / (num_steaks : ℚ)

/-- Proves that for a family of 5 members, each wanting one pound of steak,
    and needing to buy 4 steaks, the weight of each steak is 1.25 pounds. -/
theorem steak_weight_for_tommy_family : 
  steak_weight 5 4 = 5/4 := by sorry

end steak_weight_for_tommy_family_l1957_195717


namespace uphill_distance_is_six_l1957_195718

/-- Represents the travel problem with given conditions -/
structure TravelProblem where
  total_time : ℚ
  total_distance : ℕ
  speed_uphill : ℕ
  speed_flat : ℕ
  speed_downhill : ℕ

/-- Checks if a solution satisfies the problem conditions -/
def is_valid_solution (problem : TravelProblem) (uphill_distance : ℕ) (flat_distance : ℕ) : Prop :=
  let downhill_distance := problem.total_distance - uphill_distance - flat_distance
  uphill_distance + flat_distance ≤ problem.total_distance ∧
  (uphill_distance : ℚ) / problem.speed_uphill +
  (flat_distance : ℚ) / problem.speed_flat +
  (downhill_distance : ℚ) / problem.speed_downhill = problem.total_time

/-- The main theorem stating that 6 km is the correct uphill distance -/
theorem uphill_distance_is_six (problem : TravelProblem) 
  (h1 : problem.total_time = 67 / 30)
  (h2 : problem.total_distance = 10)
  (h3 : problem.speed_uphill = 4)
  (h4 : problem.speed_flat = 5)
  (h5 : problem.speed_downhill = 6) :
  ∃ (flat_distance : ℕ), is_valid_solution problem 6 flat_distance ∧
  ∀ (other_uphill : ℕ) (other_flat : ℕ),
    other_uphill ≠ 6 → ¬ is_valid_solution problem other_uphill other_flat :=
by sorry


end uphill_distance_is_six_l1957_195718


namespace range_of_a_l1957_195779

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > a - 1
def solution_set (x : ℝ) : Prop := x < 1

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality a x ↔ solution_set x) → a < 1 :=
by sorry

end range_of_a_l1957_195779


namespace vector_collinearity_angle_l1957_195795

theorem vector_collinearity_angle (θ : Real) :
  let a : Fin 2 → Real := ![2 * Real.cos θ, 2 * Real.sin θ]
  let b : Fin 2 → Real := ![3, Real.sqrt 3]
  (∃ (k : Real), a = k • b) →
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 := by
sorry

end vector_collinearity_angle_l1957_195795


namespace museum_entrance_cost_l1957_195761

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost : total_cost 20 3 5 = 115 := by
  sorry

end museum_entrance_cost_l1957_195761


namespace smallest_six_digit_divisible_by_111_l1957_195785

theorem smallest_six_digit_divisible_by_111 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 → n ≥ 100011 :=
by sorry

end smallest_six_digit_divisible_by_111_l1957_195785


namespace reciprocal_of_negative_four_l1957_195741

theorem reciprocal_of_negative_four :
  (1 : ℚ) / (-4 : ℚ) = -1/4 := by sorry

end reciprocal_of_negative_four_l1957_195741


namespace arithmetic_sequence_sum_l1957_195758

/-- The sum of an arithmetic sequence with first term 5, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 5  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum : arithmetic_sum = 390 := by
  sorry

end arithmetic_sequence_sum_l1957_195758


namespace negative_three_squared_times_negative_one_third_cubed_l1957_195783

theorem negative_three_squared_times_negative_one_third_cubed :
  -3^2 * (-1/3)^3 = 1/3 := by
  sorry

end negative_three_squared_times_negative_one_third_cubed_l1957_195783


namespace haleys_trees_l1957_195745

theorem haleys_trees (initial_trees : ℕ) : 
  (initial_trees - 4 + 5 = 10) → initial_trees = 9 := by
  sorry

end haleys_trees_l1957_195745


namespace marble_problem_l1957_195700

theorem marble_problem (g j : ℕ) 
  (hg : g % 8 = 5) 
  (hj : j % 8 = 6) : 
  (g + 5 + j) % 8 = 0 := by
sorry

end marble_problem_l1957_195700


namespace product_trailing_zeros_l1957_195720

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  trailing_zeros (45 * 320 * 125) = 5 := by sorry

end product_trailing_zeros_l1957_195720


namespace dart_partitions_l1957_195713

def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem dart_partitions :
  partition_count 5 3 = 5 := by
  sorry

end dart_partitions_l1957_195713


namespace rent_increase_percentage_l1957_195716

/-- Proves that the rent increase percentage is 25% given the specified conditions -/
theorem rent_increase_percentage 
  (num_friends : ℕ) 
  (initial_avg_rent : ℝ) 
  (new_avg_rent : ℝ) 
  (original_rent : ℝ) : 
  num_friends = 4 → 
  initial_avg_rent = 800 → 
  new_avg_rent = 850 → 
  original_rent = 800 → 
  (new_avg_rent * num_friends - initial_avg_rent * num_friends) / original_rent * 100 = 25 := by
  sorry

end rent_increase_percentage_l1957_195716


namespace henry_trays_capacity_l1957_195724

/-- The number of trays Henry picked up from the first table -/
def trays_table1 : ℕ := 29

/-- The number of trays Henry picked up from the second table -/
def trays_table2 : ℕ := 52

/-- The total number of trips Henry made -/
def total_trips : ℕ := 9

/-- The number of trays Henry could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem henry_trays_capacity : trays_per_trip = 9 := by
  sorry

end henry_trays_capacity_l1957_195724


namespace product_mod_seventeen_l1957_195780

theorem product_mod_seventeen :
  (5007 * 5008 * 5009 * 5010 * 5011) % 17 = 0 := by
  sorry

end product_mod_seventeen_l1957_195780


namespace M_has_three_elements_l1957_195749

def M : Set ℝ :=
  {m | ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    m = x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|}

theorem M_has_three_elements :
  ∃ a b c : ℝ, M = {a, b, c} :=
sorry

end M_has_three_elements_l1957_195749


namespace initial_concentration_proof_l1957_195765

/-- Proves that the initial concentration of an acidic liquid is 40% given the problem conditions --/
theorem initial_concentration_proof (initial_volume : ℝ) (water_removed : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  water_removed = 4 →
  final_concentration = 60 →
  (initial_volume - water_removed) * final_concentration / 100 = initial_volume * 40 / 100 := by
  sorry

#check initial_concentration_proof

end initial_concentration_proof_l1957_195765


namespace magic_square_sum_l1957_195768

theorem magic_square_sum (a b c d e f g : ℕ) : 
  (a + 13 + 12 + 1 = 34) →
  (g + 13 + 2 + 16 = 34) →
  (f + 16 + 9 + 4 = 34) →
  (c + 1 + 15 + 4 = 34) →
  (b + 12 + 7 + 9 = 34) →
  (d + 15 + 6 + 3 = 34) →
  (e + 2 + 7 + 14 = 34) →
  a - b - c + d + e + f - g = 11 := by
  sorry

end magic_square_sum_l1957_195768


namespace farmer_profit_is_960_l1957_195708

/-- Represents the farmer's pig business -/
structure PigBusiness where
  num_piglets : ℕ
  sale_price : ℕ
  min_growth_months : ℕ
  feed_cost_per_month : ℕ
  pigs_sold_12_months : ℕ
  pigs_sold_16_months : ℕ

/-- Calculates the total profit for the pig business -/
def calculate_profit (business : PigBusiness) : ℕ :=
  let revenue := business.sale_price * (business.pigs_sold_12_months + business.pigs_sold_16_months)
  let feed_cost_12_months := business.feed_cost_per_month * business.min_growth_months * business.pigs_sold_12_months
  let feed_cost_16_months := business.feed_cost_per_month * 16 * business.pigs_sold_16_months
  let total_feed_cost := feed_cost_12_months + feed_cost_16_months
  revenue - total_feed_cost

/-- The farmer's profit is $960 -/
theorem farmer_profit_is_960 (business : PigBusiness) 
    (h1 : business.num_piglets = 6)
    (h2 : business.sale_price = 300)
    (h3 : business.min_growth_months = 12)
    (h4 : business.feed_cost_per_month = 10)
    (h5 : business.pigs_sold_12_months = 3)
    (h6 : business.pigs_sold_16_months = 3) :
    calculate_profit business = 960 := by
  sorry

end farmer_profit_is_960_l1957_195708


namespace area_of_sliced_quadrilateral_l1957_195794

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral formed by slicing a rectangular prism -/
structure SlicedQuadrilateral where
  prism : RectangularPrism
  A : Point3D -- vertex
  B : Point3D -- midpoint on length edge
  C : Point3D -- midpoint on width edge
  D : Point3D -- midpoint on height edge

/-- Calculate the area of the sliced quadrilateral -/
def areaOfSlicedQuadrilateral (quad : SlicedQuadrilateral) : ℝ :=
  sorry -- Placeholder for the actual calculation

/-- Theorem: The area of the sliced quadrilateral is 1.5 square units -/
theorem area_of_sliced_quadrilateral :
  let prism := RectangularPrism.mk 2 3 4
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 1 0 0
  let C := Point3D.mk 0 1.5 0
  let D := Point3D.mk 0 0 2
  let quad := SlicedQuadrilateral.mk prism A B C D
  areaOfSlicedQuadrilateral quad = 1.5 := by
  sorry


end area_of_sliced_quadrilateral_l1957_195794


namespace ivanov_exaggerating_l1957_195771

-- Define the probabilities of machine breakdowns
def p1 : ℝ := 0.4
def p2 : ℝ := 0.3
def p3 : ℝ := 0

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the expected number of breakdowns per day
def expected_breakdowns_per_day : ℝ := p1 + p2 + p3

-- Define the expected number of breakdowns per week
def expected_breakdowns_per_week : ℝ := expected_breakdowns_per_day * days_in_week

-- Theorem statement
theorem ivanov_exaggerating : expected_breakdowns_per_week < 12 := by
  sorry

end ivanov_exaggerating_l1957_195771


namespace circular_film_radius_l1957_195711

/-- The radius of a circular film formed by pouring a liquid from a rectangular box into water -/
theorem circular_film_radius (box_length box_width box_height film_thickness : ℝ) 
  (h1 : box_length = 8)
  (h2 : box_width = 4)
  (h3 : box_height = 15)
  (h4 : film_thickness = 0.2)
  : ∃ (r : ℝ), r = Real.sqrt (2400 / Real.pi) ∧ 
    π * r^2 * film_thickness = box_length * box_width * box_height :=
by sorry

end circular_film_radius_l1957_195711


namespace equation_solution_l1957_195712

theorem equation_solution : ∃! x : ℝ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9 := by sorry

end equation_solution_l1957_195712


namespace prime_iff_k_t_greater_n_div_4_l1957_195731

theorem prime_iff_k_t_greater_n_div_4 (n : ℕ) (k t : ℕ) : 
  Odd n → n > 3 →
  (∀ k' < k, ¬ ∃ m : ℕ, k' * n + 1 = m * m) →
  (∀ t' < t, ¬ ∃ m : ℕ, t' * n = m * m) →
  (∃ m : ℕ, k * n + 1 = m * m) →
  (∃ m : ℕ, t * n = m * m) →
  (Nat.Prime n ↔ (k > n / 4 ∧ t > n / 4)) :=
by sorry

end prime_iff_k_t_greater_n_div_4_l1957_195731


namespace appointment_ways_l1957_195704

def dedicated_fitters : ℕ := 5
def dedicated_turners : ℕ := 4
def versatile_workers : ℕ := 2
def total_workers : ℕ := dedicated_fitters + dedicated_turners + versatile_workers
def required_fitters : ℕ := 4
def required_turners : ℕ := 4

theorem appointment_ways : 
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners required_turners * Nat.choose versatile_workers 1) +
  (Nat.choose dedicated_fitters required_fitters * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 1) * Nat.choose versatile_workers 2) +
  (Nat.choose dedicated_fitters (required_fitters - 1) * Nat.choose dedicated_turners (required_turners - 2) * Nat.choose versatile_workers 2) = 190 := by
  sorry

end appointment_ways_l1957_195704


namespace geometric_sequence_value_l1957_195755

theorem geometric_sequence_value (b : ℝ) (h1 : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 15/4 = b * r) → b = 15 * Real.sqrt 2 / 2 := by
  sorry

end geometric_sequence_value_l1957_195755


namespace basketball_court_length_l1957_195750

theorem basketball_court_length :
  ∀ (width length : ℝ),
  length = width + 14 →
  2 * length + 2 * width = 96 →
  length = 31 :=
by
  sorry

end basketball_court_length_l1957_195750


namespace geometric_sum_first_six_l1957_195781

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end geometric_sum_first_six_l1957_195781


namespace binomial_prob_example_l1957_195705

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For X ~ B(4, 1/3), P(X = 1) = 32/81 -/
theorem binomial_prob_example :
  let n : ℕ := 4
  let p : ℝ := 1/3
  let k : ℕ := 1
  binomial_pmf n p k = 32/81 := by
sorry

end binomial_prob_example_l1957_195705


namespace inequality_proof_l1957_195701

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end inequality_proof_l1957_195701


namespace log_simplification_l1957_195714

theorem log_simplification (a b c d x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) :
  Real.log (a^2 / b) + Real.log (b / c) + Real.log (c / d^2) - Real.log ((a^2 * y) / (d^2 * x)) = Real.log (x / y) := by
  sorry

end log_simplification_l1957_195714


namespace multiply_and_simplify_l1957_195775

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end multiply_and_simplify_l1957_195775


namespace coefficient_x4_proof_l1957_195757

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 2 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -5

theorem coefficient_x4_proof : ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 := by
  sorry

end coefficient_x4_proof_l1957_195757


namespace average_height_students_count_l1957_195729

/-- Represents the number of students in different height categories --/
structure HeightDistribution where
  total : ℕ
  short : ℕ
  tall : ℕ
  extremelyTall : ℕ

/-- Calculates the number of students with average height --/
def averageHeightStudents (h : HeightDistribution) : ℕ :=
  h.total - (h.short + h.tall + h.extremelyTall)

/-- Theorem: The number of students with average height in the given class is 110 --/
theorem average_height_students_count (h : HeightDistribution) 
  (h_total : h.total = 400)
  (h_short : h.short = 2 * h.total / 5)
  (h_extremelyTall : h.extremelyTall = h.total / 10)
  (h_tall : h.tall = 90) :
  averageHeightStudents h = 110 := by
  sorry

#eval averageHeightStudents ⟨400, 160, 90, 40⟩

end average_height_students_count_l1957_195729


namespace hallway_tiling_l1957_195799

theorem hallway_tiling (hallway_length hallway_width : ℕ) 
  (border_tile_size interior_tile_size : ℕ) : 
  hallway_length = 20 → 
  hallway_width = 14 → 
  border_tile_size = 2 → 
  interior_tile_size = 3 → 
  (2 * (hallway_length - 2 * border_tile_size) / border_tile_size + 
   2 * (hallway_width - 2 * border_tile_size) / border_tile_size + 4) + 
  ((hallway_length - 2 * border_tile_size) * 
   (hallway_width - 2 * border_tile_size)) / (interior_tile_size^2) = 48 := by
  sorry

end hallway_tiling_l1957_195799


namespace no_prime_for_expression_l1957_195760

theorem no_prime_for_expression (p : ℕ) (hp : Nat.Prime p) : ¬ Nat.Prime (22 * p^2 + 23) := by
  sorry

end no_prime_for_expression_l1957_195760


namespace move_point_theorem_l1957_195738

/-- A point in 2D space represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point left by a given distance. -/
def moveLeft (p : Point) (distance : ℝ) : Point :=
  { x := p.x - distance, y := p.y }

/-- Moves a point up by a given distance. -/
def moveUp (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

/-- Theorem stating that moving point P(0, 3) left by 2 units and up by 1 unit results in P₁(-2, 4). -/
theorem move_point_theorem : 
  let P : Point := { x := 0, y := 3 }
  let P₁ : Point := moveUp (moveLeft P 2) 1
  P₁.x = -2 ∧ P₁.y = 4 := by
  sorry

end move_point_theorem_l1957_195738


namespace fence_cost_l1957_195769

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 :=
by sorry

end fence_cost_l1957_195769


namespace line_through_point_l1957_195746

theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * (-3) = 5 * 1) → k = 1/3 := by sorry

end line_through_point_l1957_195746


namespace rectangle_ratio_is_two_l1957_195791

/-- Represents the configuration of rectangles around a square -/
structure SquareWithRectangles where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (config : SquareWithRectangles) : Prop :=
  -- The area of the outer square is 9 times that of the inner square
  (config.inner_square_side + 2 * config.rectangle_short_side) ^ 2 = 9 * config.inner_square_side ^ 2 ∧
  -- The outer square's side length is composed of the inner square and two short sides of rectangles
  config.inner_square_side + 2 * config.rectangle_short_side = 
    config.rectangle_long_side + config.rectangle_short_side

/-- The theorem to prove -/
theorem rectangle_ratio_is_two (config : SquareWithRectangles) 
  (h : problem_conditions config) : 
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end rectangle_ratio_is_two_l1957_195791


namespace hexagon_walk_distance_l1957_195774

theorem hexagon_walk_distance (side_length : ℝ) (walk_distance : ℝ) (end_distance : ℝ) : 
  side_length = 3 →
  walk_distance = 11 →
  end_distance = 2 * Real.sqrt 3 →
  ∃ (x y : ℝ), x^2 + y^2 = end_distance^2 :=
by sorry

end hexagon_walk_distance_l1957_195774


namespace arithmetic_sequence_problem_l1957_195742

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) :
  a 8 = 8 := by
  sorry

end arithmetic_sequence_problem_l1957_195742


namespace geometric_sequence_sum_l1957_195727

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem to be proved -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 5 + a 6 = 4) →
  (a 15 + a 16 = 16) →
  (a 25 + a 26 = 64) :=
by sorry

end geometric_sequence_sum_l1957_195727


namespace perpendicular_line_equation_l1957_195736

/-- The equation of a line perpendicular to x - y = 0 and passing through (1, 0) -/
theorem perpendicular_line_equation :
  let l₁ : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}  -- The line x - y = 0
  let p : ℝ × ℝ := (1, 0)  -- The point (1, 0)
  let l₂ : Set (ℝ × ℝ) := {q | q.1 + q.2 - 1 = 0}  -- The line we want to prove
  (∀ x y, (x, y) ∈ l₂ ↔ x + y - 1 = 0) ∧  -- l₂ is indeed x + y - 1 = 0
  p ∈ l₂ ∧  -- l₂ passes through (1, 0)
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₁ ∧ x₁ ≠ x₂ →
    (x₁ - x₂) * ((1 - x) / (0 - y)) = -1)  -- l₂ is perpendicular to l₁
  := by sorry

end perpendicular_line_equation_l1957_195736


namespace acute_angle_trig_inequality_l1957_195766

theorem acute_angle_trig_inequality (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  1/2 < Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ∧
  Real.sqrt 3 / 2 * Real.sin α + 1/2 * Real.cos α ≤ 1 := by
  sorry

end acute_angle_trig_inequality_l1957_195766


namespace value_of_expression_l1957_195739

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end value_of_expression_l1957_195739


namespace cube_sum_equals_diff_implies_square_sum_less_than_one_l1957_195762

theorem cube_sum_equals_diff_implies_square_sum_less_than_one 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end cube_sum_equals_diff_implies_square_sum_less_than_one_l1957_195762


namespace candy_distribution_theorem_l1957_195721

/-- The number of ways to distribute candy among children with restrictions -/
def distribute_candy (total_candy : ℕ) (num_children : ℕ) (min_candy : ℕ) (max_candy : ℕ) : ℕ :=
  sorry

/-- Theorem stating the specific case of candy distribution -/
theorem candy_distribution_theorem :
  distribute_candy 40 3 2 19 = 171 :=
by sorry

end candy_distribution_theorem_l1957_195721


namespace caras_cat_catch_proof_l1957_195722

/-- The number of animals Cara's cat catches given Martha's cat's catch -/
def caras_cat_catch (marthas_rats : ℕ) (marthas_birds : ℕ) : ℕ :=
  5 * (marthas_rats + marthas_birds) - 3

theorem caras_cat_catch_proof :
  caras_cat_catch 3 7 = 47 := by
  sorry

end caras_cat_catch_proof_l1957_195722


namespace imaginary_part_of_complex_fraction_l1957_195753

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1957_195753


namespace emily_productivity_l1957_195702

/-- Emily's work productivity over two days -/
theorem emily_productivity (p h : ℕ) : 
  p = 3 * h →                           -- Condition: p = 3h
  (p - 3) * (h + 3) - p * h = 6 * h - 9 -- Prove: difference in pages is 6h - 9
  := by sorry

end emily_productivity_l1957_195702


namespace no_five_digit_flippy_divisible_by_11_l1957_195719

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 
  (∃ (d1 d2 d3 d4 d5 : ℕ), 
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    ((d1 = a ∧ d2 = b ∧ d3 = a ∧ d4 = b ∧ d5 = a) ∨
     (d1 = b ∧ d2 = a ∧ d3 = b ∧ d4 = a ∧ d5 = b)))

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 : 
  ¬∃ (n : ℕ), is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end no_five_digit_flippy_divisible_by_11_l1957_195719


namespace additional_interest_percentage_l1957_195709

theorem additional_interest_percentage
  (initial_deposit : ℝ)
  (amount_after_3_years : ℝ)
  (target_amount : ℝ)
  (time_period : ℝ)
  (h1 : initial_deposit = 8000)
  (h2 : amount_after_3_years = 11200)
  (h3 : target_amount = 11680)
  (h4 : time_period = 3) :
  let original_interest := amount_after_3_years - initial_deposit
  let target_interest := target_amount - initial_deposit
  let additional_interest := target_interest - original_interest
  let additional_rate := (additional_interest * 100) / (initial_deposit * time_period)
  additional_rate = 2 := by
sorry

end additional_interest_percentage_l1957_195709


namespace share_distribution_l1957_195797

theorem share_distribution (a b c : ℕ) : 
  a + b + c = 1010 →
  (a - 25) * 2 = (b - 10) * 3 →
  (a - 25) * 5 = (c - 15) * 3 →
  c = 495 := by
sorry

end share_distribution_l1957_195797


namespace opposite_of_2023_l1957_195763

theorem opposite_of_2023 : 
  ∀ (x : ℤ), (x + 2023 = 0) → x = -2023 := by
  sorry

end opposite_of_2023_l1957_195763


namespace whole_number_between_36_and_40_l1957_195734

theorem whole_number_between_36_and_40 (M : ℤ) :
  (9 < M / 4 ∧ M / 4 < 10) → (M = 37 ∨ M = 38 ∨ M = 39) := by
  sorry

end whole_number_between_36_and_40_l1957_195734


namespace speed_difference_l1957_195796

/-- The speed difference between a cyclist and a car -/
theorem speed_difference (cyclist_distance car_distance : ℝ) (time : ℝ) 
  (h_cyclist : cyclist_distance = 88)
  (h_car : car_distance = 48)
  (h_time : time = 8)
  (h_time_pos : time > 0) :
  cyclist_distance / time - car_distance / time = 5 := by
sorry

end speed_difference_l1957_195796


namespace triangle_abc_properties_l1957_195759

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Given condition
  (2 * b * Real.cos C = 2 * a + c) →
  -- Additional condition for part 2
  (2 * Real.sqrt 3 * Real.sin (A / 2 + π / 6) * Real.cos (A / 2 + π / 6) - 
   2 * Real.sin (A / 2 + π / 6) ^ 2 = 11 / 13) →
  -- Conclusions to prove
  (B = 2 * π / 3 ∧ 
   Real.cos C = (12 + 5 * Real.sqrt 3) / 26) := by
sorry

end triangle_abc_properties_l1957_195759


namespace inverse_g_sum_l1957_195710

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_g_sum : ∃ (a b : ℝ), g a = 9 ∧ g b = -64 ∧ a + b = -5 := by
  sorry

end inverse_g_sum_l1957_195710


namespace arithmetic_equation_l1957_195789

theorem arithmetic_equation : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end arithmetic_equation_l1957_195789


namespace circle_area_increase_l1957_195798

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.12 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  new_area = 1.2544 * original_area := by
sorry

end circle_area_increase_l1957_195798


namespace tea_mixture_price_l1957_195733

/-- Given two types of tea mixed in equal proportions, this theorem proves
    the price of the second tea given the price of the first tea and the mixture. -/
theorem tea_mixture_price
  (price_tea1 : ℝ)
  (price_mixture : ℝ)
  (h1 : price_tea1 = 64)
  (h2 : price_mixture = 69) :
  ∃ (price_tea2 : ℝ),
    price_tea2 = 74 ∧
    (price_tea1 + price_tea2) / 2 = price_mixture :=
by
  sorry

end tea_mixture_price_l1957_195733


namespace total_cleaning_time_is_136_l1957_195744

/-- The time in minutes Richard takes to clean his room once -/
def richard_time : ℕ := 22

/-- The time in minutes Cory takes to clean her room once -/
def cory_time : ℕ := richard_time + 3

/-- The time in minutes Blake takes to clean his room once -/
def blake_time : ℕ := cory_time - 4

/-- The number of times they clean their rooms per week -/
def cleanings_per_week : ℕ := 2

/-- The total time spent cleaning rooms by all three people in a week -/
def total_cleaning_time : ℕ := (richard_time + cory_time + blake_time) * cleanings_per_week

theorem total_cleaning_time_is_136 : total_cleaning_time = 136 := by
  sorry

end total_cleaning_time_is_136_l1957_195744


namespace power_division_equals_729_l1957_195790

theorem power_division_equals_729 : (3 ^ 12) / (27 ^ 2) = 729 := by sorry

end power_division_equals_729_l1957_195790


namespace total_legs_calculation_l1957_195747

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a centipede has -/
def centipede_legs : ℕ := 100

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of centipedes in the room -/
def num_centipedes : ℕ := 3

/-- The total number of legs for all spiders and centipedes -/
def total_legs : ℕ := num_spiders * spider_legs + num_centipedes * centipede_legs

theorem total_legs_calculation :
  total_legs = 332 := by sorry

end total_legs_calculation_l1957_195747


namespace fraction_inequality_l1957_195703

theorem fraction_inequality (x : ℝ) :
  -3 ≤ x ∧ x ≤ 3 →
  (8 * x - 3 < 9 + 5 * x ↔ -3 ≤ x ∧ x < 3) :=
by sorry

end fraction_inequality_l1957_195703


namespace point_P_in_quadrant_III_l1957_195723

def point_in_quadrant_III (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_quadrant_III :
  point_in_quadrant_III (-1 : ℝ) (-2 : ℝ) :=
by sorry

end point_P_in_quadrant_III_l1957_195723


namespace f_is_even_and_increasing_l1957_195778

def f (x : ℝ) := x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l1957_195778


namespace license_plate_count_l1957_195782

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits (0-9) --/
def num_digits : ℕ := 10

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The number of positions for digits --/
def num_digit_positions : ℕ := 3

/-- The number of valid license plates --/
def num_valid_plates : ℕ := 6591000

theorem license_plate_count :
  num_letters ^ 3 * (num_digit_positions * num_odd_digits * num_even_digits ^ 2) = num_valid_plates :=
sorry

end license_plate_count_l1957_195782


namespace max_cards_no_sum_l1957_195706

/-- Given a positive integer k, prove that from 2k+1 cards numbered 1 to 2k+1,
    the maximum number of cards that can be selected such that no selected number
    is the sum of two other selected numbers is k+1. -/
theorem max_cards_no_sum (k : ℕ) : ∃ (S : Finset ℕ),
  S.card = k + 1 ∧
  S.toSet ⊆ Finset.range (2*k + 2) ∧
  (∀ x ∈ S, ∀ y ∈ S, ∀ z ∈ S, x + y ≠ z) ∧
  (∀ T : Finset ℕ, T.toSet ⊆ Finset.range (2*k + 2) →
    (∀ x ∈ T, ∀ y ∈ T, ∀ z ∈ T, x + y ≠ z) →
    T.card ≤ k + 1) :=
sorry

end max_cards_no_sum_l1957_195706


namespace subtract_fifteen_from_number_l1957_195777

theorem subtract_fifteen_from_number (x : ℝ) : x / 10 = 6 → x - 15 = 45 := by
  sorry

end subtract_fifteen_from_number_l1957_195777


namespace eight_digit_repeating_divisible_by_10001_l1957_195792

/-- An 8-digit positive integer whose first four digits are the same as its last four digits -/
def EightDigitRepeating (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem eight_digit_repeating_divisible_by_10001 (n : ℕ) (h : EightDigitRepeating n) :
  10001 ∣ n := by
  sorry

end eight_digit_repeating_divisible_by_10001_l1957_195792


namespace geometry_propositions_l1957_195776

-- Define the concepts
def Plane : Type := sorry
def Line : Type := sorry
def perpendicular (a b : Plane) : Prop := sorry
def parallel (a b : Plane) : Prop := sorry
def passes_through (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line (p : Plane) : Line := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection_line (p q : Plane) : Line := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ (p q : Plane) (l : Line),
    passes_through l q ∧ l = perpendicular_line p → perpendicular p q

def proposition_2 : Prop :=
  ∀ (p q : Plane) (l m : Line),
    in_plane l p ∧ in_plane m p ∧ parallel l q ∧ parallel m q → parallel p q

def proposition_3 : Prop :=
  ∀ (p q : Plane) (l : Line),
    perpendicular p q ∧ in_plane l p ∧ ¬perpendicular l (intersection_line p q) →
    ¬perpendicular l q

def proposition_4 : Prop :=
  ∀ (p : Plane) (l m : Line),
    parallel l p ∧ parallel m p → parallel_lines l m

-- State the theorem
theorem geometry_propositions :
  proposition_1 ∧ proposition_3 ∧ ¬proposition_2 ∧ ¬proposition_4 := by
  sorry

end geometry_propositions_l1957_195776


namespace number_ordering_l1957_195772

theorem number_ordering : 
  (-1.1 : ℝ) < -0.75 ∧ 
  -0.75 < -2/3 ∧ 
  -2/3 < 1/200 ∧ 
  1/200 = (0.005 : ℝ) ∧ 
  0.005 < 4/6 ∧ 
  4/6 < 5/7 ∧ 
  5/7 < 11/15 ∧ 
  11/15 < 1 := by sorry

end number_ordering_l1957_195772


namespace initial_birds_count_l1957_195707

theorem initial_birds_count (initial_storks : ℕ) (additional_birds : ℕ) (total_after : ℕ) :
  initial_storks = 2 →
  additional_birds = 5 →
  total_after = 10 →
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_birds = total_after ∧ initial_birds = 3 :=
by
  sorry

end initial_birds_count_l1957_195707


namespace largest_number_with_conditions_l1957_195764

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.get i ≠ digits.get j) ∧
  (0 ∉ digits) ∧
  (digits.sum = 18)

theorem largest_number_with_conditions :
  ∀ n : ℕ, is_valid_number n → n ≤ 6543 :=
by sorry

end largest_number_with_conditions_l1957_195764


namespace xiaopang_problem_l1957_195770

theorem xiaopang_problem (a : ℕ) (d : ℕ) (n : ℕ) : 
  a = 1 → d = 2 → n = 8 → (n / 2) * (2 * a + (n - 1) * d) = 64 := by
  sorry

end xiaopang_problem_l1957_195770


namespace f_deriv_at_one_l1957_195754

-- Define a differentiable function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the condition f(x) = 2xf'(1) + ln x
axiom f_condition (x : ℝ) : f x = 2 * x * (deriv f 1) + Real.log x

-- Theorem statement
theorem f_deriv_at_one : deriv f 1 = -1 := by sorry

end f_deriv_at_one_l1957_195754


namespace horse_speed_problem_l1957_195735

/-- A problem from "Nine Chapters on the Mathematical Art" about horse speeds and travel times. -/
theorem horse_speed_problem (x : ℝ) (h_x : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  2 * slow_horse_speed = fast_horse_speed :=
by sorry

end horse_speed_problem_l1957_195735


namespace cut_cube_edge_count_l1957_195787

/-- A cube with corners cut off -/
structure CutCube where
  /-- The number of vertices in the original cube -/
  original_vertices : Nat
  /-- The number of edges in the original cube -/
  original_edges : Nat
  /-- The number of new edges created by each vertex cut -/
  new_edges_per_vertex : Nat

/-- The theorem stating that a cube with corners cut off has 56 edges -/
theorem cut_cube_edge_count (c : CutCube) 
  (h1 : c.original_vertices = 8)
  (h2 : c.original_edges = 12)
  (h3 : c.new_edges_per_vertex = 4) :
  c.new_edges_per_vertex * c.original_vertices + 2 * c.original_edges = 56 := by
  sorry

#check cut_cube_edge_count

end cut_cube_edge_count_l1957_195787


namespace problem_solution_l1957_195773

theorem problem_solution : 2^2 + (-3)^2 - 1^2 + 4*2*(-3) = -12 := by
  sorry

end problem_solution_l1957_195773


namespace max_digit_sum_l1957_195732

theorem max_digit_sum (a b c x y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (1000 * (1 : ℚ) / (100 * a + 10 * b + c) = y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 50) →  -- y is an integer and 0 < y ≤ 50
  (1000 * (1 : ℚ) / (100 * y + 10 * y + y) = x) →  -- 0.yyy = 1/x
  (0 < x ∧ x ≤ 9) →  -- x is an integer and 0 < x ≤ 9
  (∀ a' b' c' : ℕ, 
    (a' < 10 ∧ b' < 10 ∧ c' < 10) →
    (∃ x' y' : ℕ, 
      (1000 * (1 : ℚ) / (100 * a' + 10 * b' + c') = y') ∧
      (0 < y' ∧ y' ≤ 50) ∧
      (1000 * (1 : ℚ) / (100 * y' + 10 * y' + y') = x') ∧
      (0 < x' ∧ x' ≤ 9)) →
    (a + b + c ≥ a' + b' + c')) →
  a + b + c = 8 := by
sorry

end max_digit_sum_l1957_195732


namespace least_n_mod_1000_l1957_195728

/-- Sum of digits in base 4 representation -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ :=
  sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ :=
  sorry

theorem least_n_mod_1000 : N % 1000 = 151 := by
  sorry

end least_n_mod_1000_l1957_195728


namespace second_drive_speed_l1957_195730

def same_distance_drives (v : ℝ) : Prop :=
  let d := 180 / 3  -- distance for each drive
  (d / 4 + d / v + d / 6 = 37) ∧ (d / 4 + d / v + d / 6 > 0)

theorem second_drive_speed : ∃ v : ℝ, same_distance_drives v ∧ v = 5 := by
  sorry

end second_drive_speed_l1957_195730


namespace abc_remainder_mod_7_l1957_195748

theorem abc_remainder_mod_7 (a b c : ℕ) 
  (h_a : a < 7) (h_b : b < 7) (h_c : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 2)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 0 := by
sorry

end abc_remainder_mod_7_l1957_195748


namespace variance_mean_preserved_l1957_195784

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  ((xs.map (λ x => ((x : ℚ) - m) ^ 2)).sum) / xs.length

def replacement_set1 : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, -1, 5]
def replacement_set2 : List Int := [-5, 1, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean replacement_set1 ∧
   variance initial_set = variance replacement_set1) ∨
  (mean initial_set = mean replacement_set2 ∧
   variance initial_set = variance replacement_set2) :=
by sorry

end variance_mean_preserved_l1957_195784


namespace oil_distance_theorem_l1957_195751

/-- Represents the relationship between remaining oil and distance traveled --/
def oil_distance_relation (x : ℝ) : ℝ := 62 - 0.12 * x

theorem oil_distance_theorem :
  let initial_oil : ℝ := 62
  let data_points : List (ℝ × ℝ) := [(100, 50), (200, 38), (300, 26), (400, 14)]
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = oil_distance_relation x :=
by sorry

end oil_distance_theorem_l1957_195751


namespace shaded_area_percentage_l1957_195786

theorem shaded_area_percentage (side_length : ℝ) (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ) :
  side_length = 6 ∧
  rect1_width = 2 ∧ rect1_height = 2 ∧
  rect2_width = 4 ∧ rect2_height = 1 ∧
  rect3_width = 6 ∧ rect3_height = 1 →
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) / (side_length * side_length) = 22 / 36 :=
by sorry

end shaded_area_percentage_l1957_195786


namespace complex_symmetry_quotient_l1957_195788

theorem complex_symmetry_quotient (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → z₁ = 1 + I → z₁ / z₂ = I := by
  sorry

end complex_symmetry_quotient_l1957_195788


namespace min_value_sum_l1957_195743

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its minimum value
def a : ℝ := sorry

-- Define b as the minimum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem min_value_sum :
  ∀ x : ℝ, f x ≥ b ∧ a + b = -3 :=
sorry

end min_value_sum_l1957_195743


namespace fish_cost_theorem_l1957_195737

theorem fish_cost_theorem (dog_fish : ℕ) (fish_price : ℕ) :
  dog_fish = 40 →
  fish_price = 4 →
  (dog_fish + dog_fish / 2) * fish_price = 240 :=
by
  sorry

end fish_cost_theorem_l1957_195737


namespace certain_number_exists_l1957_195726

theorem certain_number_exists : ∃ x : ℝ, 
  5.4 * x - (0.6 * 10) / 1.2 = 31.000000000000004 ∧ 
  abs (x - 6.666666666666667) < 1e-15 := by
  sorry

end certain_number_exists_l1957_195726
