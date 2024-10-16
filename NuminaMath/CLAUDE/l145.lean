import Mathlib

namespace NUMINAMATH_CALUDE_total_distance_run_l145_14540

theorem total_distance_run (num_students : ℕ) (avg_distance : ℕ) (h1 : num_students = 18) (h2 : avg_distance = 106) :
  num_students * avg_distance = 1908 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_run_l145_14540


namespace NUMINAMATH_CALUDE_max_a_value_l145_14519

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a x₂) →
  ∀ a' : ℝ, (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a' x₂) →
  a' ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l145_14519


namespace NUMINAMATH_CALUDE_car_production_total_l145_14502

/-- The number of cars produced in North America -/
def north_america_cars : ℕ := 3884

/-- The number of cars produced in Europe -/
def europe_cars : ℕ := 2871

/-- The total number of cars produced -/
def total_cars : ℕ := north_america_cars + europe_cars

theorem car_production_total : total_cars = 6755 := by
  sorry

end NUMINAMATH_CALUDE_car_production_total_l145_14502


namespace NUMINAMATH_CALUDE_parallelogram_condition_l145_14553

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_a_gt_b : a > b

/-- Checks if a point lies on the unit circle -/
def onUnitCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 1

/-- Checks if a point lies on the given ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The main theorem stating the condition for the parallelogram property -/
theorem parallelogram_condition (e : Ellipse) :
  (∀ p : Point, onEllipse p e → 
    ∃ q r s : Point, 
      onEllipse q e ∧ onEllipse r e ∧ onEllipse s e ∧
      onUnitCircle q ∧ onUnitCircle s ∧
      -- Additional conditions for parallelogram property would be defined here
      True) → 
  1 / e.a^2 + 1 / e.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_condition_l145_14553


namespace NUMINAMATH_CALUDE_andreas_living_room_area_andreas_living_room_area_is_48_l145_14583

/-- The area of Andrea's living room floor given a carpet covering 75% of it -/
theorem andreas_living_room_area (carpet_width : ℝ) (carpet_length : ℝ) 
  (carpet_coverage_percentage : ℝ) : ℝ :=
  let carpet_area := carpet_width * carpet_length
  let floor_area := carpet_area / carpet_coverage_percentage
  floor_area

/-- Proof of Andrea's living room floor area -/
theorem andreas_living_room_area_is_48 :
  andreas_living_room_area 4 9 0.75 = 48 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_andreas_living_room_area_is_48_l145_14583


namespace NUMINAMATH_CALUDE_tom_speed_proof_l145_14536

/-- Represents the speed from B to C in miles per hour -/
def speed_B_to_C : ℝ := 64.8

/-- Represents the distance between B and C in miles -/
def distance_B_to_C : ℝ := 1  -- We use 1 as a variable to represent this distance

theorem tom_speed_proof :
  let distance_W_to_B : ℝ := 2 * distance_B_to_C
  let speed_W_to_B : ℝ := 60
  let average_speed : ℝ := 36
  let total_distance : ℝ := distance_W_to_B + distance_B_to_C
  let time_W_to_B : ℝ := distance_W_to_B / speed_W_to_B
  let time_B_to_C : ℝ := distance_B_to_C / speed_B_to_C
  let total_time : ℝ := time_W_to_B + time_B_to_C
  average_speed = total_distance / total_time →
  speed_B_to_C = 64.8 := by
  sorry

#check tom_speed_proof

end NUMINAMATH_CALUDE_tom_speed_proof_l145_14536


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l145_14561

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun x : ℝ => a * Real.log x + x^2
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l145_14561


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l145_14592

def total_handshakes (na nb : ℕ) : ℕ :=
  (na + nb) * (na + nb - 1) / 2 + na + nb

def is_valid_configuration (na nb : ℕ) : Prop :=
  na < nb ∧ total_handshakes na nb = 465

theorem min_coach_handshakes :
  ∃ (na nb : ℕ), is_valid_configuration na nb ∧
  ∀ (ma mb : ℕ), is_valid_configuration ma mb → na ≤ ma :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l145_14592


namespace NUMINAMATH_CALUDE_least_d_value_l145_14547

theorem least_d_value (c d : ℕ+) 
  (hc_factors : (Nat.divisors c.val).card = 4)
  (hd_factors : (Nat.divisors d.val).card = c.val)
  (hd_div_c : c.val ∣ d.val) :
  72 ≤ d.val :=
sorry

end NUMINAMATH_CALUDE_least_d_value_l145_14547


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l145_14511

theorem cos_squared_minus_sin_squared_15_deg : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l145_14511


namespace NUMINAMATH_CALUDE_combination_sum_identity_l145_14564

theorem combination_sum_identity (k m n : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (fun i => Nat.choose n i * Nat.choose n (m - i)) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_identity_l145_14564


namespace NUMINAMATH_CALUDE_equation_solution_l145_14574

theorem equation_solution (m : ℕ+) : 
  (∃ x : ℕ+, x ≠ 8 ∧ (m * x : ℚ) / (x - 8 : ℚ) = ((4 * m + x) : ℚ) / (x - 8 : ℚ)) ↔ 
  m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l145_14574


namespace NUMINAMATH_CALUDE_value_added_to_number_l145_14541

theorem value_added_to_number (sum number value : ℕ) : 
  sum = number + value → number = 81 → sum = 96 → value = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l145_14541


namespace NUMINAMATH_CALUDE_three_integers_problem_l145_14523

theorem three_integers_problem :
  ∃ (a b c : ℕ) (k : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    k > 0 ∧
    a + b + c = 93 ∧
    a * b * c = 3375 ∧
    b = k * a ∧
    c = k^2 * a ∧
    a = 3 ∧ b = 15 ∧ c = 75 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_problem_l145_14523


namespace NUMINAMATH_CALUDE_tram_length_l145_14565

/-- The length of a tram given observation times and tunnel length -/
theorem tram_length (pass_time : ℝ) (tunnel_length : ℝ) (tunnel_time : ℝ) :
  pass_time = 4 →
  tunnel_length = 64 →
  tunnel_time = 12 →
  ∃ (tram_length : ℝ),
    tram_length > 0 ∧
    tram_length / pass_time = (tunnel_length + tram_length) / tunnel_time ∧
    tram_length = 32 :=
by sorry


end NUMINAMATH_CALUDE_tram_length_l145_14565


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l145_14566

/-- The trajectory of point Q given a line l and the relation between Q and a point P on l -/
theorem trajectory_of_Q (x y m n : ℝ) : 
  (2 * m + 4 * n + 3 = 0) →  -- P(m, n) is on line l
  (m = 3 * x ∧ n = 3 * y) →  -- P = 3Q, derived from 2⃗OQ = ⃗QP
  (2 * x + 4 * y + 1 = 0) :=  -- Trajectory equation of Q
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l145_14566


namespace NUMINAMATH_CALUDE_f_properties_l145_14501

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x^2

theorem f_properties :
  (∃ x : ℝ, x ≠ 0 ∧ f x = 0 ↔ x = 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ∀ y : ℝ, y ≠ 0 → f y ≤ f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x ≤ f 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l145_14501


namespace NUMINAMATH_CALUDE_liar_count_l145_14549

/-- Represents a candidate's statement about the number of lies told before their turn. -/
structure CandidateStatement where
  position : Nat
  claimed_lies : Nat
  is_truthful : Bool

/-- The debate scenario with 12 candidates. -/
def debate_scenario (statements : Vector CandidateStatement 12) : Prop :=
  (∀ i : Fin 12, (statements.get i).position = i.val + 1) ∧
  (∀ i : Fin 12, (statements.get i).claimed_lies = i.val + 1) ∧
  (∃ i : Fin 12, (statements.get i).is_truthful)

/-- The theorem to be proved. -/
theorem liar_count (statements : Vector CandidateStatement 12) 
  (h : debate_scenario statements) : 
  (statements.toList.filter (fun s => !s.is_truthful)).length = 11 := by
  sorry


end NUMINAMATH_CALUDE_liar_count_l145_14549


namespace NUMINAMATH_CALUDE_pond_width_pond_width_is_10_l145_14578

/-- The width of a rectangular pond, given its length, depth, and volume of soil extracted. -/
theorem pond_width (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume = length * depth * (volume / (length * depth)) :=
by sorry

/-- The width of the pond is 10 meters. -/
theorem pond_width_is_10 (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume / (length * depth) = 10 :=
by sorry

end NUMINAMATH_CALUDE_pond_width_pond_width_is_10_l145_14578


namespace NUMINAMATH_CALUDE_area_union_rotated_triangle_l145_14546

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The centroid of a triangle -/
def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Rotation of a point around another point by 180 degrees -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The union of two regions -/
def unionArea (area1 : ℝ) (area2 : ℝ) : ℝ := sorry

theorem area_union_rotated_triangle (t : Triangle) :
  let m := t.centroid
  let t' := Triangle.mk t.a t.b t.c t.h_positive
  unionArea t.area t'.area = t.area := by sorry

end NUMINAMATH_CALUDE_area_union_rotated_triangle_l145_14546


namespace NUMINAMATH_CALUDE_price_change_theorem_l145_14581

theorem price_change_theorem (initial_price : ℝ) (x : ℝ) : 
  initial_price > 0 →
  let price1 := initial_price * (1 + 0.3)
  let price2 := price1 * (1 - 0.15)
  let price3 := price2 * (1 + 0.1)
  let price4 := price3 * (1 - x / 100)
  price4 = initial_price →
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l145_14581


namespace NUMINAMATH_CALUDE_propositions_analysis_l145_14543

-- Proposition 1
def has_real_roots (q : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + q = 0

-- Proposition 2
def both_zero (x y : ℝ) : Prop := x = 0 ∧ y = 0

theorem propositions_analysis :
  -- Proposition 1
  (¬ (∀ q : ℝ, has_real_roots q → q < 1)) ∧  -- Converse is false
  (∀ q : ℝ, ¬(has_real_roots q) → q ≥ 1) ∧  -- Contrapositive is true
  -- Proposition 2
  (∀ x y : ℝ, both_zero x y → x^2 + y^2 = 0) ∧  -- Converse is true
  (∀ x y : ℝ, ¬(both_zero x y) → x^2 + y^2 ≠ 0)  -- Contrapositive is true
  := by sorry

end NUMINAMATH_CALUDE_propositions_analysis_l145_14543


namespace NUMINAMATH_CALUDE_jennys_money_l145_14554

theorem jennys_money (initial_money : ℚ) : 
  (initial_money * (1 - 3/7) = 24) → 
  (initial_money / 2 = 21) := by
sorry

end NUMINAMATH_CALUDE_jennys_money_l145_14554


namespace NUMINAMATH_CALUDE_solve_product_equation_l145_14506

theorem solve_product_equation : 
  ∀ x : ℝ, 6 * (x - 3) * (x + 5) = 0 ↔ x = -5 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_product_equation_l145_14506


namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_a_gt_one_l145_14545

theorem solution_set_nonempty_iff_a_gt_one :
  ∀ a : ℝ, (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_a_gt_one_l145_14545


namespace NUMINAMATH_CALUDE_group_size_l145_14575

/-- The number of people in a group, given certain weight changes. -/
theorem group_size (avg_increase : ℝ) (old_weight new_weight : ℝ) (h1 : avg_increase = 1.5)
    (h2 : new_weight - old_weight = 6) : ℤ :=
  4

#check group_size

end NUMINAMATH_CALUDE_group_size_l145_14575


namespace NUMINAMATH_CALUDE_bananas_permutations_count_l145_14504

/-- The number of unique permutations of the letters in "BANANAS" -/
def bananas_permutations : ℕ := 420

/-- The total number of letters in "BANANAS" -/
def total_letters : ℕ := 7

/-- The number of occurrences of 'A' in "BANANAS" -/
def a_count : ℕ := 3

/-- The number of occurrences of 'N' in "BANANAS" -/
def n_count : ℕ := 2

/-- Theorem stating that the number of unique permutations of the letters in "BANANAS"
    is equal to 420, given the total number of letters and the counts of repeated letters. -/
theorem bananas_permutations_count : 
  bananas_permutations = (Nat.factorial total_letters) / 
    ((Nat.factorial a_count) * (Nat.factorial n_count)) := by
  sorry

end NUMINAMATH_CALUDE_bananas_permutations_count_l145_14504


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l145_14597

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangePeopleInChairs (n : ℕ) (m : ℕ) : ℕ :=
  (m - n + 1).factorial

theorem five_people_six_chairs :
  arrangePeopleInChairs 5 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l145_14597


namespace NUMINAMATH_CALUDE_triangle_area_l145_14526

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) :
  (1/2 : ℝ) * a * b = 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l145_14526


namespace NUMINAMATH_CALUDE_regression_prediction_l145_14520

/-- Represents the regression equation y = mx + b -/
structure RegressionLine where
  m : ℝ
  b : ℝ

/-- Calculates the y-value for a given x using the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.m * x + line.b

theorem regression_prediction 
  (line : RegressionLine)
  (h1 : line.m = 9.4)
  (h2 : line.predict 3.5 = 42)
  : line.predict 6 = 65.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_prediction_l145_14520


namespace NUMINAMATH_CALUDE_largest_equal_division_l145_14595

theorem largest_equal_division (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_equal_division_l145_14595


namespace NUMINAMATH_CALUDE_point_coordinates_l145_14529

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the fourth quadrant with distances 3 and 5 to x-axis and y-axis respectively has coordinates (5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 3) 
  (h3 : distance_to_y_axis p = 5) : 
  p = Point.mk 5 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l145_14529


namespace NUMINAMATH_CALUDE_fraction_equality_l145_14500

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + y) / (x - 5 * y) = -3) : 
  (x + 5 * y) / (5 * x - y) = 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l145_14500


namespace NUMINAMATH_CALUDE_solution_to_equation_l145_14588

theorem solution_to_equation : 
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^3 + 7*x) + Real.sqrt (x + 7) = 50 - x^2) ∧ 
            (x = (29/12)^2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l145_14588


namespace NUMINAMATH_CALUDE_function_characterization_l145_14569

theorem function_characterization
  (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f m + f n = max (f (m + n)) (f (m - n))) :
  ∃ k : ℕ, ∀ x : ℤ, f x = k * |x| :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l145_14569


namespace NUMINAMATH_CALUDE_population_after_two_years_l145_14594

-- Define the initial population
def initial_population : ℕ := 415600

-- Define the first year increase rate
def first_year_increase : ℚ := 25 / 100

-- Define the second year decrease rate
def second_year_decrease : ℚ := 30 / 100

-- Theorem statement
theorem population_after_two_years :
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let final_population := population_after_first_year * (1 - second_year_decrease)
  final_population = 363650 := by
  sorry

end NUMINAMATH_CALUDE_population_after_two_years_l145_14594


namespace NUMINAMATH_CALUDE_min_transportation_fee_l145_14585

/-- Represents the transportation problem with given parameters -/
structure TransportProblem where
  total_goods : ℕ
  large_truck_capacity : ℕ
  large_truck_cost : ℕ
  small_truck_capacity : ℕ
  small_truck_cost : ℕ

/-- Calculates the transportation cost for a given number of large and small trucks -/
def transportation_cost (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : ℕ :=
  large_trucks * p.large_truck_cost + small_trucks * p.small_truck_cost

/-- Checks if a combination of trucks can transport all goods -/
def can_transport_all (p : TransportProblem) (large_trucks : ℕ) (small_trucks : ℕ) : Prop :=
  large_trucks * p.large_truck_capacity + small_trucks * p.small_truck_capacity ≥ p.total_goods

/-- Theorem stating that the minimum transportation fee is 1800 yuan -/
theorem min_transportation_fee (p : TransportProblem) 
    (h1 : p.total_goods = 20)
    (h2 : p.large_truck_capacity = 7)
    (h3 : p.large_truck_cost = 600)
    (h4 : p.small_truck_capacity = 4)
    (h5 : p.small_truck_cost = 400) :
    (∀ large_trucks small_trucks, can_transport_all p large_trucks small_trucks →
      transportation_cost p large_trucks small_trucks ≥ 1800) ∧
    (∃ large_trucks small_trucks, can_transport_all p large_trucks small_trucks ∧
      transportation_cost p large_trucks small_trucks = 1800) :=
  sorry


end NUMINAMATH_CALUDE_min_transportation_fee_l145_14585


namespace NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l145_14548

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) :
  (10*a + 3*b + 5*c) ≤ Real.sqrt 134 :=
by sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, 9*a^2 + 4*b^2 + 25*c^2 = 1 ∧ 
    Real.sqrt 134 - ε < 10*a + 3*b + 5*c :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l145_14548


namespace NUMINAMATH_CALUDE_coin_problem_l145_14544

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →
  2 * x + 5 * y = 125 →
  y = 15 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l145_14544


namespace NUMINAMATH_CALUDE_circles_and_line_properties_l145_14568

-- Define Circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define Circle D
def circle_D (x y : ℝ) : Prop := (x - 5)^2 + (y - 4)^2 = 4

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := x = 5 ∨ 7*x - 24*y + 61 = 0

-- Theorem statement
theorem circles_and_line_properties :
  -- Part 1: Circles C and D are externally tangent
  (∃ (x y : ℝ), circle_C x y ∧ circle_D x y) ∧
  -- The distance between centers is equal to the sum of radii
  ((2 - 5)^2 + (0 - 4)^2 : ℝ) = (3 + 2)^2 ∧
  -- Part 2: Line l is tangent to Circle C and passes through (5,4)
  (∀ (x y : ℝ), line_l x y → 
    -- Line passes through (5,4)
    (x = 5 ∧ y = 4 ∨ 7*5 - 24*4 + 61 = 0) ∧
    -- Line is tangent to Circle C (distance from center to line is equal to radius)
    ((2*7 + 0*(-24) - 61)^2 / (7^2 + (-24)^2) : ℝ) = 3^2) :=
sorry

end NUMINAMATH_CALUDE_circles_and_line_properties_l145_14568


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l145_14570

def n : ℕ := 10
def k : ℕ := 4
def p : ℚ := 1/3

theorem unfair_coin_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 13440/59049 := by sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l145_14570


namespace NUMINAMATH_CALUDE_marks_deck_cost_l145_14571

/-- Calculates the total cost of constructing and sealing a rectangular deck. -/
def deck_total_cost (length width construction_cost_per_sqft sealant_cost_per_sqft : ℝ) : ℝ :=
  let area := length * width
  let construction_cost := area * construction_cost_per_sqft
  let sealant_cost := area * sealant_cost_per_sqft
  construction_cost + sealant_cost

/-- Theorem stating that the total cost of Mark's deck is $4800. -/
theorem marks_deck_cost : 
  deck_total_cost 30 40 3 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_marks_deck_cost_l145_14571


namespace NUMINAMATH_CALUDE_equal_elements_from_inequalities_l145_14562

theorem equal_elements_from_inequalities (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end NUMINAMATH_CALUDE_equal_elements_from_inequalities_l145_14562


namespace NUMINAMATH_CALUDE_leak_emptying_time_l145_14559

theorem leak_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 6048 →
  inlet_rate = 6 →
  emptying_time_with_inlet = 12 →
  (tank_capacity / (tank_capacity / emptying_time_with_inlet + inlet_rate * 60)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l145_14559


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l145_14537

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three : f (f (-3)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l145_14537


namespace NUMINAMATH_CALUDE_abs_neg_2023_l145_14573

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l145_14573


namespace NUMINAMATH_CALUDE_correct_operation_l145_14589

theorem correct_operation (a : ℝ) : (-a + 2) * (-a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l145_14589


namespace NUMINAMATH_CALUDE_sequence_convergence_l145_14539

def converges (a : ℕ → ℝ) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem sequence_convergence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n ≥ 2, a (n + 1) ≤ (a n * (a (n - 1))^2)^(1/3)) :
  converges a :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l145_14539


namespace NUMINAMATH_CALUDE_quadratic_roots_l145_14556

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 4*x - 21 = 0) ↔ (x = 3 ∨ x = -7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l145_14556


namespace NUMINAMATH_CALUDE_sum_of_medians_l145_14534

def player_A_median : ℝ := 36
def player_B_median : ℝ := 27

theorem sum_of_medians : player_A_median + player_B_median = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_medians_l145_14534


namespace NUMINAMATH_CALUDE_no_valid_A_l145_14532

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 75 % A = 0 ∧ (5361000 + 100 * A + 4) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l145_14532


namespace NUMINAMATH_CALUDE_work_completion_proof_l145_14530

/-- The number of days it takes p to complete the work alone -/
def p_days : ℕ := 80

/-- The number of days it takes q to complete the work alone -/
def q_days : ℕ := 48

/-- The total number of days the work lasted -/
def total_days : ℕ := 35

/-- The number of days after which q joined p -/
def q_join_day : ℕ := 8

/-- The work rate of p per day -/
def p_rate : ℚ := 1 / p_days

/-- The work rate of q per day -/
def q_rate : ℚ := 1 / q_days

/-- The total work completed is 1 (representing 100%) -/
def total_work : ℚ := 1

theorem work_completion_proof :
  p_rate * q_join_day + (p_rate + q_rate) * (total_days - q_join_day) = total_work :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l145_14530


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l145_14591

theorem ceiling_floor_difference : 
  ⌈(15 / 8 : ℝ) * (-34 / 4 : ℝ)⌉ - ⌊(15 / 8 : ℝ) * ⌊-34 / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l145_14591


namespace NUMINAMATH_CALUDE_f_two_zeros_l145_14577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

theorem f_two_zeros (a : ℝ) :
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (2 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_l145_14577


namespace NUMINAMATH_CALUDE_mac_running_rate_l145_14584

/-- The running rate of Apple in miles per hour -/
def apple_rate : ℝ := 3

/-- The race distance in miles -/
def race_distance : ℝ := 24

/-- The time difference between Apple and Mac in minutes -/
def time_difference : ℝ := 120

/-- Mac's running rate in miles per hour -/
def mac_rate : ℝ := 4

/-- Theorem stating that given the conditions, Mac's running rate is 4 miles per hour -/
theorem mac_running_rate : 
  let apple_time := race_distance / apple_rate * 60  -- Apple's time in minutes
  let mac_time := apple_time - time_difference       -- Mac's time in minutes
  mac_rate = race_distance / (mac_time / 60) := by
sorry

end NUMINAMATH_CALUDE_mac_running_rate_l145_14584


namespace NUMINAMATH_CALUDE_congruence_problem_l145_14538

theorem congruence_problem (a b n : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 85 [ZMOD 60] →
  150 ≤ n →
  n ≤ 241 →
  (a - b ≡ n [ZMOD 60]) ↔ (n = 180 ∨ n = 240) :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l145_14538


namespace NUMINAMATH_CALUDE_max_y_value_l145_14510

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  y ≤ 27 + Real.sqrt 829 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20*x₀ + 54*y₀ ∧ y₀ = 27 + Real.sqrt 829 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l145_14510


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l145_14563

theorem fraction_to_decimal : 
  ∃ (n : ℕ), (58 : ℚ) / 160 = (3625 : ℚ) / (10^n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l145_14563


namespace NUMINAMATH_CALUDE_min_value_of_a_l145_14552

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) → 
  a ≥ 4 ∧ ∀ b : ℝ, b > 0 → (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + b / y) ≥ 9) → b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l145_14552


namespace NUMINAMATH_CALUDE_f_derivative_l145_14587

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  (deriv f) x = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l145_14587


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l145_14567

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 7 * k) →
  (∃ k₁ k₂ k₃ k₄ k₅ : ℤ, a = 7 * k₁ ∧ b = 7 * k₂ ∧ c = 7 * k₃ ∧ d = 7 * k₄ ∧ e = 7 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l145_14567


namespace NUMINAMATH_CALUDE_x_range_and_max_y_over_x_l145_14596

/-- Circle C with center (4,3) and radius 3 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 - 3)^2 = 9}

/-- A point P on circle C -/
def P : ℝ × ℝ := sorry

/-- P is on circle C -/
axiom hP : P ∈ C

theorem x_range_and_max_y_over_x :
  (1 ≤ P.1 ∧ P.1 ≤ 7) ∧
  ∀ Q ∈ C, Q.2 / Q.1 ≤ 24 / 7 := by sorry

end NUMINAMATH_CALUDE_x_range_and_max_y_over_x_l145_14596


namespace NUMINAMATH_CALUDE_inequality_implication_l145_14590

theorem inequality_implication (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l145_14590


namespace NUMINAMATH_CALUDE_inequality_proof_l145_14579

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l145_14579


namespace NUMINAMATH_CALUDE_books_read_during_trip_l145_14550

-- Define the travel distance in miles
def travel_distance : ℕ := 6760

-- Define the reading rate in miles per book
def miles_per_book : ℕ := 450

-- Theorem to prove
theorem books_read_during_trip : travel_distance / miles_per_book = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_during_trip_l145_14550


namespace NUMINAMATH_CALUDE_function_and_range_proof_l145_14555

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B (f : ℝ → ℝ) : Set ℝ := {x | 1 < f x ∧ f x < 3}

-- Define the theorem
theorem function_and_range_proof 
  (a b : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_f : ∀ x, f x = a * x + b)
  (h_f_condition : ∀ x, f (2 * x + 1) = 4 * x + 1)
  (h_subset : B f ⊆ A a) :
  (∀ x, f x = 2 * x - 1) ∧ (1/2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_proof_l145_14555


namespace NUMINAMATH_CALUDE_person_age_puzzle_l145_14514

theorem person_age_puzzle (x : ℤ) : 3 * (x + 5) - 3 * (x - 5) = x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l145_14514


namespace NUMINAMATH_CALUDE_gym_attendance_l145_14535

theorem gym_attendance (initial_lifters : ℕ) : 
  initial_lifters + 5 - 2 = 19 → initial_lifters = 16 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_l145_14535


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l145_14533

open Complex

theorem imaginary_part_of_z : ∃ (z : ℂ), z = (1 + I)^2 + I^2010 ∧ z.im = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l145_14533


namespace NUMINAMATH_CALUDE_negation_of_existence_l145_14576

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x y : ℤ), x^2 + y^2 = 2015) ↔ (∀ (x y : ℤ), x^2 + y^2 ≠ 2015) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l145_14576


namespace NUMINAMATH_CALUDE_fruit_store_inventory_l145_14580

theorem fruit_store_inventory (initial_amount : ℚ) : 
  initial_amount - 3/10 + 2/5 = 19/20 → initial_amount = 17/20 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_inventory_l145_14580


namespace NUMINAMATH_CALUDE_opposite_of_2023_l145_14598

theorem opposite_of_2023 :
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l145_14598


namespace NUMINAMATH_CALUDE_point_on_h_graph_coordinate_sum_l145_14551

theorem point_on_h_graph_coordinate_sum : 
  ∀ (g h : ℝ → ℝ),
  g 4 = -5 →
  (∀ x, h x = (g x)^2 + 3) →
  4 + h 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_point_on_h_graph_coordinate_sum_l145_14551


namespace NUMINAMATH_CALUDE_negation_of_implication_l145_14521

theorem negation_of_implication :
  (¬(∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0)) ↔
  (∀ x : ℝ, x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l145_14521


namespace NUMINAMATH_CALUDE_margarita_run_distance_l145_14503

/-- Proves that Margarita ran 18 feet given the conditions of the long jump event -/
theorem margarita_run_distance (ricciana_run : ℝ) (ricciana_jump : ℝ) (margarita_total : ℝ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_total = ricciana_run + ricciana_jump + 1 →
  ∃ (margarita_run : ℝ) (margarita_jump : ℝ),
    margarita_jump = 2 * ricciana_jump - 1 ∧
    margarita_total = margarita_run + margarita_jump ∧
    margarita_run = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_margarita_run_distance_l145_14503


namespace NUMINAMATH_CALUDE_triple_layer_area_is_six_l145_14582

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallSize : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given arrangement -/
def tripleLayerArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_layer_area_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallSize = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleLayerArea arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_layer_area_is_six_l145_14582


namespace NUMINAMATH_CALUDE_inserted_square_side_length_l145_14516

/-- An isosceles triangle with a square inserted -/
structure TriangleWithSquare where
  /-- Length of the lateral sides of the isosceles triangle -/
  lateral_side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Side length of the inserted square -/
  square_side : ℝ

/-- Theorem: In an isosceles triangle with lateral sides of 10 and base of 12, 
    the side length of an inserted square is 24/5 -/
theorem inserted_square_side_length 
  (t : TriangleWithSquare) 
  (h1 : t.lateral_side = 10) 
  (h2 : t.base = 12) : 
  t.square_side = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_inserted_square_side_length_l145_14516


namespace NUMINAMATH_CALUDE_chord_equation_l145_14509

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by y² = 6x -/
def Parabola := {p : Point | p.y^2 = 6 * p.x}

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point bisects a chord of the parabola -/
def bisectsChord (p : Point) (l : Line) : Prop :=
  p ∈ Parabola ∧ 
  ∃ a b : Point, a ≠ b ∧ 
    a ∈ Parabola ∧ b ∈ Parabola ∧
    a.onLine l ∧ b.onLine l ∧
    p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2

/-- The main theorem to be proved -/
theorem chord_equation : 
  let p := Point.mk 4 1
  let l := Line.mk 3 (-1) (-11)
  p ∈ Parabola ∧ bisectsChord p l := by sorry

end NUMINAMATH_CALUDE_chord_equation_l145_14509


namespace NUMINAMATH_CALUDE_simplify_fraction_l145_14542

theorem simplify_fraction : 24 * (8 / 15) * (5 / 18) = 32 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l145_14542


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_230_l145_14586

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  AD_longer_than_BC : BC < AP + BC + DQ

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem stating that the perimeter of the given trapezoid is 230 -/
theorem trapezoid_perimeter_is_230 (t : Trapezoid) 
    (h1 : t.BC = 60)
    (h2 : t.AP = 24)
    (h3 : t.DQ = 11)
    (h4 : t.AB = 40)
    (h5 : t.CD = 35) : 
  perimeter t = 230 := by
  sorry

#check trapezoid_perimeter_is_230

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_230_l145_14586


namespace NUMINAMATH_CALUDE_max_value_of_f_l145_14572

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 3/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l145_14572


namespace NUMINAMATH_CALUDE_max_intersections_pentagon_circle_l145_14599

/-- A regular pentagon -/
structure RegularPentagon where
  -- We don't need to define the structure, just declare it exists
  
/-- A circle -/
structure Circle where
  -- We don't need to define the structure, just declare it exists

/-- The maximum number of intersections between a line segment and a circle is 2 -/
axiom max_intersections_line_circle : ℕ

/-- The number of sides in a regular pentagon -/
def pentagon_sides : ℕ := 5

/-- Theorem: The maximum number of intersections between a regular pentagon and a circle is 10 -/
theorem max_intersections_pentagon_circle (p : RegularPentagon) (c : Circle) :
  (max_intersections_line_circle * pentagon_sides : ℕ) = 10 := by
  sorry

#check max_intersections_pentagon_circle

end NUMINAMATH_CALUDE_max_intersections_pentagon_circle_l145_14599


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l145_14525

theorem binomial_coefficient_ratio (n : ℕ) : 
  (∃ r : ℕ, r + 2 ≤ n ∧ 
    (n.choose r : ℚ) / (n.choose (r + 1)) = 1 / 2 ∧
    (n.choose (r + 1) : ℚ) / (n.choose (r + 2)) = 2 / 3) → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l145_14525


namespace NUMINAMATH_CALUDE_market_fruit_count_l145_14507

/-- Calculates the total number of apples and oranges in a market -/
def total_fruits (num_apples : ℕ) (apple_orange_diff : ℕ) : ℕ :=
  num_apples + (num_apples - apple_orange_diff)

/-- Theorem: Given a market with 164 apples and 27 more apples than oranges,
    the total number of apples and oranges is 301 -/
theorem market_fruit_count : total_fruits 164 27 = 301 := by
  sorry

end NUMINAMATH_CALUDE_market_fruit_count_l145_14507


namespace NUMINAMATH_CALUDE_f_is_power_function_l145_14593

-- Define what a power function is
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the function we want to prove is a power function
def f (x : ℝ) : ℝ := x ^ (1/2)

-- Theorem statement
theorem f_is_power_function : is_power_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_power_function_l145_14593


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l145_14522

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year with a starting day -/
structure Year where
  startDay : DayOfWeek

/-- Counts the number of Sundays in the first n days of a year -/
def countSundays (y : Year) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year is 7 -/
theorem max_sundays_in_45_days :
  ∀ y : Year, countSundays y 45 ≤ 7 ∧ ∃ y' : Year, countSundays y' 45 = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l145_14522


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l145_14517

theorem average_of_eleven_numbers 
  (n : ℕ) 
  (first_six_avg : ℚ) 
  (last_six_avg : ℚ) 
  (sixth_number : ℚ) 
  (h1 : n = 11) 
  (h2 : first_six_avg = 98) 
  (h3 : last_six_avg = 65) 
  (h4 : sixth_number = 318) : 
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l145_14517


namespace NUMINAMATH_CALUDE_line_in_plane_equivalence_l145_14508

-- Define a type for geometric objects
inductive GeometricObject
| Line : GeometricObject
| Plane : GeometricObject

-- Define a predicate for "is in"
def isIn (a b : GeometricObject) : Prop := sorry

-- Define the subset relation
def subset (a b : GeometricObject) : Prop := sorry

-- Theorem statement
theorem line_in_plane_equivalence (l : GeometricObject) (α : GeometricObject) :
  (l = GeometricObject.Line ∧ α = GeometricObject.Plane ∧ isIn l α) ↔ subset l α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_equivalence_l145_14508


namespace NUMINAMATH_CALUDE_system_solution_l145_14558

theorem system_solution (x y : ℝ) : 
  (2 * x^2 - 7 * x * y - 4 * y^2 + 9 * x - 18 * y + 10 = 0 ∧ x^2 + 2 * y^2 = 6) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) ∨ (x = -22/9 ∧ y = -1/9)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l145_14558


namespace NUMINAMATH_CALUDE_vector_on_line_l145_14528

/-- Given distinct vectors a and b in a vector space V over ℝ,
    prove that the vector (1/2)a + (1/2)b lies on the line passing through a and b. -/
theorem vector_on_line {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_l145_14528


namespace NUMINAMATH_CALUDE_product_positive_not_imply_both_positive_l145_14557

theorem product_positive_not_imply_both_positive : ∃ (a b : ℝ), a * b > 0 ∧ ¬(a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_product_positive_not_imply_both_positive_l145_14557


namespace NUMINAMATH_CALUDE_triangle_properties_l145_14524

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (R : ℝ) :
  R = Real.sqrt 3 →
  (2 * Real.sin A - Real.sin C) / Real.sin B = Real.cos C / Real.cos B →
  (∀ x y z, x + y + z = Real.pi → Real.sin x / a = Real.sin y / b) →
  (b = 2 * R * Real.sin B) →
  (b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) →
  (∃ (S : ℝ), S = 1/2 * a * c * Real.sin B) →
  (B = Real.pi / 3 ∧ 
   b = 3 ∧ 
   (∃ (S_max : ℝ), S_max = 9 * Real.sqrt 3 / 4 ∧ 
     (∀ S, S ≤ S_max) ∧ 
     (S = S_max ↔ a = c ∧ a = 3))) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l145_14524


namespace NUMINAMATH_CALUDE_not_enough_money_l145_14505

theorem not_enough_money (book1_price book2_price available_money : ℝ) 
  (h1 : book1_price = 21.8)
  (h2 : book2_price = 19.5)
  (h3 : available_money = 40) :
  book1_price + book2_price > available_money := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l145_14505


namespace NUMINAMATH_CALUDE_jerry_money_duration_l145_14560

/-- The number of weeks Jerry's money will last given his earnings and weekly spending -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Jerry's money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_duration_l145_14560


namespace NUMINAMATH_CALUDE_value_of_S_l145_14527

/-- Given S = 6 × 10000 + 5 × 1000 + 4 × 10 + 3 × 1, prove that S = 65043 -/
theorem value_of_S : 
  let S := 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1
  S = 65043 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_l145_14527


namespace NUMINAMATH_CALUDE_leftover_slices_is_ten_l145_14513

/-- Calculates the number of leftover pizza slices --/
def leftover_slices (small_pizza_slices : ℕ) (large_pizza_slices : ℕ) 
  (small_pizzas : ℕ) (large_pizzas : ℕ) (george_slices : ℕ) 
  (bill_fred_mark_slices : ℕ) : ℕ :=
  let total_slices := small_pizza_slices * small_pizzas + large_pizza_slices * large_pizzas
  let bob_slices := george_slices + 1
  let susie_slices := bob_slices / 2
  let eaten_slices := george_slices + bob_slices + susie_slices + 3 * bill_fred_mark_slices
  total_slices - eaten_slices

/-- Theorem stating that the number of leftover slices is 10 --/
theorem leftover_slices_is_ten : 
  leftover_slices 4 8 3 2 3 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_leftover_slices_is_ten_l145_14513


namespace NUMINAMATH_CALUDE_tangent_circles_radii_relation_l145_14512

/-- Three circles with centers O₁, O₂, and O₃, tangent to each other and a line -/
structure TangentCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  R₃ : ℝ
  tangent_to_line : Bool
  tangent_to_each_other : Bool

/-- The theorem stating the relationship between the radii of three tangent circles -/
theorem tangent_circles_radii_relation (c : TangentCircles) :
  c.tangent_to_line ∧ c.tangent_to_each_other →
  1 / Real.sqrt c.R₂ = 1 / Real.sqrt c.R₁ + 1 / Real.sqrt c.R₃ := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_relation_l145_14512


namespace NUMINAMATH_CALUDE_second_pedal_triangle_rotation_l145_14515

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_180 : angle_a + angle_b + angle_c = 180

/-- Computes the angles of the first pedal triangle -/
def first_pedal_triangle (t : Triangle) : Triangle :=
  { angle_a := 2 * t.angle_a,
    angle_b := 2 * t.angle_b,
    angle_c := 2 * t.angle_c - 180,
    sum_180 := by sorry }

/-- Computes the angles of the second pedal triangle -/
def second_pedal_triangle (t : Triangle) : Triangle :=
  let pt := first_pedal_triangle t
  { angle_a := 180 - 2 * pt.angle_a,
    angle_b := 180 - 2 * pt.angle_b,
    angle_c := 180 - 2 * pt.angle_c,
    sum_180 := by sorry }

/-- Computes the rotation angle between two triangles -/
def rotation_angle (t1 t2 : Triangle) : ℝ :=
  (180 - t1.angle_c) + t2.angle_b

/-- Theorem statement -/
theorem second_pedal_triangle_rotation (t : Triangle)
  (h1 : t.angle_a = 12)
  (h2 : t.angle_b = 36)
  (h3 : t.angle_c = 132) :
  rotation_angle t (second_pedal_triangle t) = 120 := by sorry

end NUMINAMATH_CALUDE_second_pedal_triangle_rotation_l145_14515


namespace NUMINAMATH_CALUDE_total_is_255_l145_14518

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a b c : ℕ)

/-- Calculates the total amount of money given a ratio and the first person's share -/
def totalAmount (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  let multiplier := firstShare / ratio.a
  multiplier * (ratio.a + ratio.b + ratio.c)

/-- Theorem stating that for the given ratio and first share, the total amount is 255 -/
theorem total_is_255 (ratio : MoneyRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 5) (h3 : ratio.c = 9) 
    (h4 : totalAmount ratio 45 = 255) : totalAmount ratio 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_is_255_l145_14518


namespace NUMINAMATH_CALUDE_vector_sum_equality_l145_14531

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- For any three points A, B, and C in a vector space, 
    the sum of vectors AB, BC, and BA equals vector BC. -/
theorem vector_sum_equality (A B C : V) : 
  (B - A) + (C - B) + (A - B) = C - B := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l145_14531
