import Mathlib

namespace coeff_x_cube_p_l2454_245404

-- Define the polynomial
def p (x : ℝ) := x^2 - 3*x + 3

-- Define the coefficient of x in the expansion of p(x)^3
def coeff_x (p : ℝ → ℝ) : ℝ :=
  (p 1 - p 0) - (p (-1) - p 0)

-- Theorem statement
theorem coeff_x_cube_p : coeff_x (fun x ↦ (p x)^3) = -81 := by
  sorry

end coeff_x_cube_p_l2454_245404


namespace lucy_fish_count_l2454_245414

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end lucy_fish_count_l2454_245414


namespace arithmetic_sequence_properties_l2454_245424

/-- An arithmetic sequence and its partial sums with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2
  S5_lt_S6 : S 5 < S 6
  S6_eq_S7 : S 6 = S 7
  S7_gt_S8 : S 7 > S 8

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 6 > 0 ∧ seq.a 7 = 0 ∧ seq.a 8 < 0 ∧ common_difference seq < 0 :=
sorry

end arithmetic_sequence_properties_l2454_245424


namespace value_of_x_l2454_245445

theorem value_of_x : (2011^3 - 2011^2) / 2011 = 2011 * 2010 := by
  sorry

end value_of_x_l2454_245445


namespace roots_equation_l2454_245444

theorem roots_equation (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → 
  β^2 - 3*β + 1 = 0 → 
  7 * α^5 + 8 * β^4 = 1448 := by
  sorry

end roots_equation_l2454_245444


namespace wrong_observation_value_l2454_245418

theorem wrong_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : correct_value = 48) :
  let wrong_value := n * original_mean - (n * corrected_mean - correct_value)
  wrong_value = 23 := by sorry

end wrong_observation_value_l2454_245418


namespace crayons_per_child_l2454_245437

/-- Given that there are 6 children and a total of 18 crayons,
    prove that each child has 3 crayons. -/
theorem crayons_per_child :
  ∀ (total_crayons : ℕ) (num_children : ℕ),
    total_crayons = 18 →
    num_children = 6 →
    total_crayons / num_children = 3 :=
by sorry

end crayons_per_child_l2454_245437


namespace sons_age_l2454_245440

/-- Given a man and his son, where the man is 18 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 16 years. -/
theorem sons_age (man_age son_age : ℕ) : 
  man_age = son_age + 18 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
  sorry

end sons_age_l2454_245440


namespace unique_triple_existence_l2454_245469

theorem unique_triple_existence (p : ℕ) (hp : Prime p) 
  (h_prime : ∀ n : ℕ, 0 < n → n < p → Prime (n^2 - n + p)) :
  ∃! (a b c : ℤ), 
    b^2 - 4*a*c = 1 - 4*p ∧ 
    0 < a ∧ a ≤ c ∧ 
    -a ≤ b ∧ b < a ∧
    a = 1 ∧ b = -1 ∧ c = p := by
  sorry

end unique_triple_existence_l2454_245469


namespace exactly_one_true_proposition_l2454_245419

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relationships
def skew (l1 l2 : Line) : Prop := sorry
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def oblique_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ (p1 p2 : Plane) (l1 l2 : Line), 
  p1 ≠ p2 → in_plane l1 p1 → in_plane l2 p2 → skew l1 l2

def prop2 : Prop := ∀ (p1 p2 : Plane) (l : Line), 
  oblique_to_plane l p1 → 
  (perpendicular p1 p2 ∧ in_plane l p2) → 
  ∀ (p3 : Plane), perpendicular p1 p3 ∧ in_plane l p3 → p2 = p3

def prop3 : Prop := ∀ (p1 p2 p3 : Plane), 
  perpendicular p1 p2 → perpendicular p1 p3 → parallel p2 p3

-- Theorem statement
theorem exactly_one_true_proposition : 
  (prop1 = False ∧ prop2 = True ∧ prop3 = False) := by sorry

end exactly_one_true_proposition_l2454_245419


namespace triangle_properties_l2454_245494

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the incircle of a triangle -/
def incircle_radius (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Theorem: In triangle ABC, if a = 8 and the incircle radius is √3, then A = π/3 and the area is 11√3 -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 8) 
  (h2 : incircle_radius t = Real.sqrt 3) : 
  t.A = π/3 ∧ triangle_area t = 11 * Real.sqrt 3 := by sorry

end triangle_properties_l2454_245494


namespace third_day_breath_holding_l2454_245432

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem third_day_breath_holding :
  let seq := breath_holding_sequence
  seq 1 = 10 ∧ 
  seq 2 = 20 ∧ 
  seq 6 = 90 →
  seq 3 = 30 := by
  sorry

end third_day_breath_holding_l2454_245432


namespace negation_of_squared_plus_one_geq_one_l2454_245423

theorem negation_of_squared_plus_one_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end negation_of_squared_plus_one_geq_one_l2454_245423


namespace prime_sum_divisible_by_six_l2454_245488

theorem prime_sum_divisible_by_six (p q r : Nat) : 
  Prime p → Prime q → Prime r → p > 3 → q > 3 → r > 3 → Prime (p + q + r) → 
  (6 ∣ p + q) ∨ (6 ∣ p + r) ∨ (6 ∣ q + r) := by
  sorry

end prime_sum_divisible_by_six_l2454_245488


namespace fish_population_estimate_l2454_245479

/-- Approximate number of fish in a pond given tagging and recapture data -/
theorem fish_population_estimate (tagged_initial : ℕ) (second_catch : ℕ) (tagged_second : ℕ) :
  tagged_initial = 70 →
  second_catch = 50 →
  tagged_second = 2 →
  (tagged_second : ℚ) / second_catch = tagged_initial / (tagged_initial + 1680) :=
by
  sorry

#check fish_population_estimate

end fish_population_estimate_l2454_245479


namespace product_of_large_numbers_l2454_245436

theorem product_of_large_numbers : (4 * 10^6) * (8 * 10^6) = 3.2 * 10^13 := by
  sorry

end product_of_large_numbers_l2454_245436


namespace gym_cost_is_650_l2454_245451

/-- Calculates the total cost of two gym memberships for a year -/
def total_gym_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) : ℕ :=
  let expensive_monthly_fee := 3 * cheap_monthly_fee
  let expensive_signup_fee := 4 * expensive_monthly_fee
  let cheap_yearly_cost := 12 * cheap_monthly_fee + cheap_signup_fee
  let expensive_yearly_cost := 12 * expensive_monthly_fee + expensive_signup_fee
  cheap_yearly_cost + expensive_yearly_cost

/-- Proves that the total cost of two gym memberships for a year is $650 -/
theorem gym_cost_is_650 : total_gym_cost 10 50 = 650 := by
  sorry

end gym_cost_is_650_l2454_245451


namespace even_odd_sum_difference_l2454_245461

/-- Sum of first n positive even integers -/
def sum_even (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of first n positive odd integers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 30 positive even integers
    and the sum of the first 30 positive odd integers is 30 -/
theorem even_odd_sum_difference : sum_even 30 - sum_odd 30 = 30 := by
  sorry

end even_odd_sum_difference_l2454_245461


namespace expression_equality_l2454_245495

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 / y) :
  (x - 2 / x) * (y + 2 / y) = x^2 - y^2 := by
  sorry

end expression_equality_l2454_245495


namespace assembly_line_increased_rate_l2454_245400

/-- Represents the production rate of an assembly line -/
structure AssemblyLine where
  initial_rate : ℝ
  increased_rate : ℝ
  initial_order : ℝ
  second_order : ℝ
  average_output : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem assembly_line_increased_rate (a : AssemblyLine) 
  (h1 : a.initial_rate = 30)
  (h2 : a.initial_order = 60)
  (h3 : a.second_order = 60)
  (h4 : a.average_output = 40)
  (h5 : (a.initial_order + a.second_order) / 
        (a.initial_order / a.initial_rate + a.second_order / a.increased_rate) = a.average_output) :
  a.increased_rate = 60 := by
  sorry


end assembly_line_increased_rate_l2454_245400


namespace double_side_halves_energy_l2454_245473

/-- Represents the energy stored between two point charges -/
structure EnergyBetweenCharges where
  distance : ℝ
  charge1 : ℝ
  charge2 : ℝ
  energy : ℝ

/-- Represents a configuration of three point charges in an equilateral triangle -/
structure TriangleConfiguration where
  sideLength : ℝ
  charge : ℝ
  totalEnergy : ℝ

/-- The relation between energy, distance, and charges -/
axiom energy_proportionality 
  (e1 e2 : EnergyBetweenCharges) : 
  e1.charge1 = e2.charge1 → e1.charge2 = e2.charge2 → 
  e1.energy * e1.distance = e2.energy * e2.distance

/-- The total energy in a triangle configuration is the sum of energies between pairs -/
axiom triangle_energy 
  (tc : TriangleConfiguration) (e : EnergyBetweenCharges) :
  e.distance = tc.sideLength → e.charge1 = tc.charge → e.charge2 = tc.charge →
  tc.totalEnergy = 3 * e.energy

/-- Theorem: Doubling the side length of the triangle halves the total energy -/
theorem double_side_halves_energy 
  (tc1 tc2 : TriangleConfiguration) :
  tc1.charge = tc2.charge →
  tc2.sideLength = 2 * tc1.sideLength →
  tc2.totalEnergy = tc1.totalEnergy / 2 := by
  sorry

end double_side_halves_energy_l2454_245473


namespace decimal_difference_proof_l2454_245405

/-- The repeating decimal 0.727272... --/
def repeating_decimal : ℚ := 8 / 11

/-- The terminating decimal 0.72 --/
def terminating_decimal : ℚ := 72 / 100

/-- The difference between the repeating decimal and the terminating decimal --/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_proof : decimal_difference = 2 / 275 := by
  sorry

end decimal_difference_proof_l2454_245405


namespace prime_roots_sum_reciprocals_l2454_245415

theorem prime_roots_sum_reciprocals (p q m : ℕ) : 
  Prime p → Prime q → 
  (p : ℝ)^2 - 99*p + m = 0 → 
  (q : ℝ)^2 - 99*q + m = 0 → 
  (p : ℝ)/q + (q : ℝ)/p = 9413/194 := by
  sorry

end prime_roots_sum_reciprocals_l2454_245415


namespace tablet_cash_price_l2454_245477

/-- Represents the installment plan for a tablet purchase -/
structure InstallmentPlan where
  downPayment : ℕ
  firstFourMonths : ℕ
  middleFourMonths : ℕ
  lastFourMonths : ℕ
  savings : ℕ

/-- Calculates the cash price of the tablet given the installment plan -/
def cashPrice (plan : InstallmentPlan) : ℕ :=
  plan.downPayment +
  4 * plan.firstFourMonths +
  4 * plan.middleFourMonths +
  4 * plan.lastFourMonths -
  plan.savings

/-- Theorem stating that the cash price of the tablet is 450 -/
theorem tablet_cash_price :
  let plan := InstallmentPlan.mk 100 40 35 30 70
  cashPrice plan = 450 := by
  sorry

end tablet_cash_price_l2454_245477


namespace complex_coordinates_of_i_times_one_minus_i_l2454_245487

theorem complex_coordinates_of_i_times_one_minus_i :
  let i : ℂ := Complex.I
  (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by sorry

end complex_coordinates_of_i_times_one_minus_i_l2454_245487


namespace inverse_variation_problem_l2454_245410

-- Define the relationship between x and y
def inverse_variation (x y : ℝ) (k : ℝ) : Prop := x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ k : ℝ) 
  (h1 : inverse_variation x₁ y₁ k)
  (h2 : x₁ = 8)
  (h3 : y₁ = 1)
  (h4 : y₂ = 2)
  (h5 : inverse_variation x₂ y₂ k) :
  x₂ = 1 := by
sorry

end inverse_variation_problem_l2454_245410


namespace rhombus_perimeter_l2454_245459

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l2454_245459


namespace function_extrema_l2454_245425

/-- Given constants a and b, if the function f(x) = ax^3 + b*ln(x + sqrt(1+x^2)) + 3
    has a maximum value of 10 on the interval (-∞, 0),
    then the minimum value of f(x) on the interval (0, +∞) is -4. -/
theorem function_extrema (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^3 + b * Real.log (x + Real.sqrt (1 + x^2)) + 3
  (∃ (M : ℝ), M = 10 ∧ ∀ (x : ℝ), x < 0 → f x ≤ M) →
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
by sorry


end function_extrema_l2454_245425


namespace chord_intersection_segments_l2454_245441

theorem chord_intersection_segments (r : ℝ) (chord_length : ℝ) 
  (hr : r = 7) (hchord : chord_length = 10) : 
  ∃ (ak kb : ℝ), 
    ak = r - 2 * Real.sqrt 6 ∧ 
    kb = r + 2 * Real.sqrt 6 ∧ 
    ak + kb = 2 * r ∧
    ak * kb = (chord_length / 2) ^ 2 := by
  sorry

end chord_intersection_segments_l2454_245441


namespace initial_dogs_count_l2454_245475

/-- Proves that the initial number of dogs in a pet center is 36, given the conditions of the problem. -/
theorem initial_dogs_count (initial_cats : ℕ) (adopted_dogs : ℕ) (added_cats : ℕ) (final_total : ℕ) 
  (h1 : initial_cats = 29)
  (h2 : adopted_dogs = 20)
  (h3 : added_cats = 12)
  (h4 : final_total = 57)
  (h5 : final_total = initial_cats + added_cats + (initial_dogs - adopted_dogs)) :
  initial_dogs = 36 := by
  sorry

end initial_dogs_count_l2454_245475


namespace larger_number_is_448_l2454_245480

/-- Given two positive integers with specific HCF and LCM properties, prove the larger number is 448 -/
theorem larger_number_is_448 (a b : ℕ+) : 
  (Nat.gcd a b = 32) →
  (∃ (x y : ℕ+), x = 13 ∧ y = 14 ∧ Nat.lcm a b = 32 * x * y) →
  max a b = 448 := by
sorry

end larger_number_is_448_l2454_245480


namespace function_passes_through_point_l2454_245491

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by
  sorry

end function_passes_through_point_l2454_245491


namespace bike_ride_problem_l2454_245443

theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (speed_good : ℝ) (speed_tired : ℝ) :
  total_distance = 122 →
  total_time = 8 →
  speed_good = 20 →
  speed_tired = 12 →
  ∃ time_feeling_good : ℝ,
    time_feeling_good * speed_good + (total_time - time_feeling_good) * speed_tired = total_distance ∧
    time_feeling_good = 13 / 4 :=
by sorry

end bike_ride_problem_l2454_245443


namespace prime_between_50_60_mod_7_l2454_245456

theorem prime_between_50_60_mod_7 :
  ∀ n : ℕ,
  (Prime n) →
  (50 < n) →
  (n < 60) →
  (n % 7 = 4) →
  n = 53 :=
by sorry

end prime_between_50_60_mod_7_l2454_245456


namespace abs_a_plus_b_equals_three_sqrt_three_l2454_245474

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 3 * x * y + 1

/-- The theorem statement -/
theorem abs_a_plus_b_equals_three_sqrt_three
  (a b : ℝ)
  (h1 : f a b + 1 = 42)
  (h2 : f b a = 42) :
  |a + b| = 3 * Real.sqrt 3 := by
  sorry

end abs_a_plus_b_equals_three_sqrt_three_l2454_245474


namespace function_composition_equality_l2454_245406

/-- Given functions f, g, and h, proves that A = 3B / (1 + C) -/
theorem function_composition_equality (A B C : ℝ) : 
  let f := fun x => A * x - 3 * B^2
  let g := fun x => B * x
  let h := fun x => x + C
  f (g (h 1)) = 0 → A = 3 * B / (1 + C) := by
  sorry

end function_composition_equality_l2454_245406


namespace boxes_per_hand_for_ten_people_l2454_245453

/-- Given a group of people and the total number of boxes they can hold,
    calculate the number of boxes a single person can hold in each hand. -/
def boxes_per_hand (group_size : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / group_size) / 2

/-- Theorem stating that for a group of 10 people holding 20 boxes in total,
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_ten_people :
  boxes_per_hand 10 20 = 1 := by
  sorry

end boxes_per_hand_for_ten_people_l2454_245453


namespace max_m_value_l2454_245442

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end max_m_value_l2454_245442


namespace neighborhood_total_l2454_245454

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  car : ℕ
  bikeOnly : ℕ

/-- Calculates the total number of households in the neighborhood -/
def totalHouseholds (n : Neighborhood) : ℕ :=
  n.neither + (n.car - n.both) + n.bikeOnly + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 16)
  (h3 : n.car = 44)
  (h4 : n.bikeOnly = 35) :
  totalHouseholds n = 90 := by
  sorry

#eval totalHouseholds { neither := 11, both := 16, car := 44, bikeOnly := 35 }

end neighborhood_total_l2454_245454


namespace investment_return_calculation_l2454_245481

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.085 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  investment_1 + investment_2 = total_investment →
  investment_1 * return_rate_1 + investment_2 * ((total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2) = total_investment * combined_return_rate →
  (total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2 = 0.09 :=
by sorry

end investment_return_calculation_l2454_245481


namespace circumcircle_equation_l2454_245476

/-- The circumcircle of a triangle ABC with vertices A(-√3, 0), B(√3, 0), and C(0, 3) 
    has the equation x² + (y - 1)² = 4 -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-Real.sqrt 3, 0)
  let B : ℝ × ℝ := (Real.sqrt 3, 0)
  let C : ℝ × ℝ := (0, 3)
  x^2 + (y - 1)^2 = 4 ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
by sorry


end circumcircle_equation_l2454_245476


namespace fuel_capacity_ratio_l2454_245489

theorem fuel_capacity_ratio (original_cost : ℝ) (price_increase : ℝ) (new_cost : ℝ) :
  original_cost = 200 →
  price_increase = 0.2 →
  new_cost = 480 →
  (new_cost / (original_cost * (1 + price_increase))) = 2 :=
by sorry

end fuel_capacity_ratio_l2454_245489


namespace no_rational_solution_l2454_245482

theorem no_rational_solution : ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014 := by
  sorry

end no_rational_solution_l2454_245482


namespace no_real_solutions_for_abs_equation_l2454_245422

theorem no_real_solutions_for_abs_equation : 
  ¬ ∃ x : ℝ, |x^2 - 3| = 2*x + 6 := by
sorry

end no_real_solutions_for_abs_equation_l2454_245422


namespace probability_prime_sum_digits_l2454_245447

def ball_numbers : List Nat := [10, 11, 13, 14, 17, 19, 21, 23]

def sum_of_digits (n : Nat) : Nat :=
  n.repr.foldl (fun acc d => acc + d.toNat - 48) 0

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

theorem probability_prime_sum_digits :
  let favorable_outcomes := (ball_numbers.map sum_of_digits).filter is_prime |>.length
  let total_outcomes := ball_numbers.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end probability_prime_sum_digits_l2454_245447


namespace pyramid_sphere_radii_relation_main_theorem_l2454_245412

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  h : ℝ  -- height of the pyramid
  a : ℝ  -- half of the base edge length

/-- The theorem stating the relationship between R and r for a regular quadrilateral pyramid -/
theorem pyramid_sphere_radii_relation (p : RegularQuadrilateralPyramid) :
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∀ p : RegularQuadrilateralPyramid, p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  intro p
  exact pyramid_sphere_radii_relation p

end pyramid_sphere_radii_relation_main_theorem_l2454_245412


namespace remi_water_spill_l2454_245458

/-- Represents the amount of water Remi spilled the first time -/
def first_spill : ℕ := sorry

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The number of days Remi drinks water -/
def days : ℕ := 7

/-- The amount of water Remi spilled the second time -/
def second_spill : ℕ := 8

/-- The total amount of water Remi actually drank in ounces -/
def total_drunk : ℕ := 407

theorem remi_water_spill :
  first_spill = 5 ∧
  bottle_capacity * refills_per_day * days - first_spill - second_spill = total_drunk :=
by sorry

end remi_water_spill_l2454_245458


namespace largest_remainder_269_l2454_245408

theorem largest_remainder_269 (n : ℕ) (h : n < 150) :
  ∃ (q r : ℕ), 269 = n * q + r ∧ r < n ∧ r ≤ 133 ∧
  (∀ (q' r' : ℕ), 269 = n * q' + r' ∧ r' < n → r' ≤ r) :=
sorry

end largest_remainder_269_l2454_245408


namespace exists_arrangement_for_23_l2454_245457

/-- Recursive definition of the sequence F_i -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧
  F 12 % 23 = 0 :=
sorry

end exists_arrangement_for_23_l2454_245457


namespace triangle_problem_l2454_245426

theorem triangle_problem (a b c A B C : ℝ) (h1 : 2 * a * Real.cos B + b = 2 * c)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π / 3 ∧ Real.sin B + Real.sin C = Real.sqrt 6 / 2 := by sorry

end triangle_problem_l2454_245426


namespace total_toys_is_160_l2454_245446

/-- The number of toys Kamari has -/
def kamari_toys : ℕ := 65

/-- The number of additional toys Anais has compared to Kamari -/
def anais_extra_toys : ℕ := 30

/-- The total number of toys Anais and Kamari have together -/
def total_toys : ℕ := kamari_toys + (kamari_toys + anais_extra_toys)

/-- Theorem stating that the total number of toys is 160 -/
theorem total_toys_is_160 : total_toys = 160 := by
  sorry

end total_toys_is_160_l2454_245446


namespace spherical_to_rectangular_transformation_l2454_245434

/-- Given a point with rectangular coordinates (3, -4, 2) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ + π, φ) has rectangular coordinates (-3, 4, 2). -/
theorem spherical_to_rectangular_transformation (ρ θ φ : Real) :
  (ρ * Real.sin φ * Real.cos θ = 3 ∧
   ρ * Real.sin φ * Real.sin θ = -4 ∧
   ρ * Real.cos φ = 2) →
  (ρ * Real.sin φ * Real.cos (θ + Real.pi) = -3 ∧
   ρ * Real.sin φ * Real.sin (θ + Real.pi) = 4 ∧
   ρ * Real.cos φ = 2) :=
by sorry


end spherical_to_rectangular_transformation_l2454_245434


namespace complement_union_eq_result_l2454_245471

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 5}

-- Define sets P and Q
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {2, 4}

-- Theorem statement
theorem complement_union_eq_result : (U \ P) ∪ Q = {2, 4} := by
  sorry

end complement_union_eq_result_l2454_245471


namespace polynomial_property_implies_P0_values_l2454_245421

/-- A polynomial P with real coefficients satisfying the given property -/
def SatisfiesProperty (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (|y^2 - P x| ≤ 2 * |x|) ↔ (|x^2 - P y| ≤ 2 * |y|)

/-- The theorem stating the possible values of P(0) -/
theorem polynomial_property_implies_P0_values (P : ℝ → ℝ) (h : SatisfiesProperty P) :
  P 0 < 0 ∨ P 0 = 1 :=
sorry

end polynomial_property_implies_P0_values_l2454_245421


namespace max_fraction_value_l2454_245467

theorem max_fraction_value : 
  ∃ (a b c d e f : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    f ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a / b + c / d) / (e / f) = 14 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      y ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      z ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      w ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      u ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      v ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧
      w ≠ u ∧ w ≠ v ∧
      u ≠ v →
      (x / y + z / w) / (u / v) ≤ 14 := by
  sorry

end max_fraction_value_l2454_245467


namespace trigonometric_inequality_l2454_245411

theorem trigonometric_inequality (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end trigonometric_inequality_l2454_245411


namespace rectangle_ratio_sum_l2454_245490

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Definition of the specific rectangle in the problem -/
def problemRectangle : Rectangle :=
  { A := ⟨0, 0⟩
  , B := ⟨6, 0⟩
  , C := ⟨6, 3⟩
  , D := ⟨0, 3⟩ }

/-- Point E on BC -/
def E : Point :=
  ⟨6, 1⟩

/-- Point F on CE -/
def F : Point :=
  ⟨6, 2⟩

/-- Theorem statement -/
theorem rectangle_ratio_sum (r s t : ℕ) :
  (r > 0 ∧ s > 0 ∧ t > 0) →
  (Nat.gcd r (Nat.gcd s t) = 1) →
  (∃ (P Q : Point),
    P.x = Q.x ∧ 
    P.y < Q.y ∧
    Q.y < problemRectangle.D.y ∧
    P.x > problemRectangle.A.x ∧
    P.x < problemRectangle.B.x ∧
    (P.x - problemRectangle.A.x) / (Q.x - P.x) = r / s ∧
    (Q.x - P.x) / (problemRectangle.B.x - Q.x) = s / t) →
  r + s + t = 20 := by
    sorry

end rectangle_ratio_sum_l2454_245490


namespace x_y_z_relation_l2454_245486

theorem x_y_z_relation (x y z : ℝ) : 
  x = 100.48 → 
  y = 100.48 → 
  x * z = y^2 → 
  z = 1 :=
by sorry

end x_y_z_relation_l2454_245486


namespace intersection_equality_l2454_245493

theorem intersection_equality (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end intersection_equality_l2454_245493


namespace trapezoid_diagonals_l2454_245485

/-- A trapezoid with bases a and c, legs b and d, and diagonals e and f. -/
structure Trapezoid (a c b d e f : ℝ) : Prop where
  positive_a : 0 < a
  positive_c : 0 < c
  positive_b : 0 < b
  positive_d : 0 < d
  positive_e : 0 < e
  positive_f : 0 < f
  a_greater_c : a > c

/-- The diagonals of a trapezoid can be expressed in terms of its sides. -/
theorem trapezoid_diagonals (a c b d e f : ℝ) (trap : Trapezoid a c b d e f) :
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := by
  sorry


end trapezoid_diagonals_l2454_245485


namespace chemical_mixture_problem_l2454_245470

/-- Proves that adding 20 liters of chemical x to 80 liters of a mixture that is 30% chemical x
    results in a new mixture that is 44% chemical x. -/
theorem chemical_mixture_problem :
  let initial_volume : ℝ := 80
  let initial_concentration : ℝ := 0.30
  let added_volume : ℝ := 20
  let final_concentration : ℝ := 0.44
  (initial_volume * initial_concentration + added_volume) / (initial_volume + added_volume) = final_concentration :=
by sorry

end chemical_mixture_problem_l2454_245470


namespace provisions_problem_l2454_245460

/-- The number of days the provisions last for the initial group -/
def initial_days : ℝ := 12

/-- The number of additional men joining the group -/
def additional_men : ℕ := 300

/-- The number of days the provisions last after the additional men join -/
def new_days : ℝ := 9.662337662337663

/-- The initial number of men in the group -/
def initial_men : ℕ := 1240

theorem provisions_problem :
  ∃ (M : ℝ), 
    (M ≥ 0) ∧ 
    (abs (M - initial_men) < 1) ∧
    (M * initial_days = (M + additional_men) * new_days) :=
by sorry

end provisions_problem_l2454_245460


namespace target_sectors_degrees_l2454_245448

def circle_degrees : ℝ := 360

def microphotonics_percent : ℝ := 12
def home_electronics_percent : ℝ := 17
def food_additives_percent : ℝ := 9
def genetically_modified_microorganisms_percent : ℝ := 22
def industrial_lubricants_percent : ℝ := 6
def artificial_intelligence_percent : ℝ := 4
def nanotechnology_percent : ℝ := 5

def basic_astrophysics_percent : ℝ :=
  100 - (microphotonics_percent + home_electronics_percent + food_additives_percent +
         genetically_modified_microorganisms_percent + industrial_lubricants_percent +
         artificial_intelligence_percent + nanotechnology_percent)

def target_sectors_percent : ℝ :=
  basic_astrophysics_percent + artificial_intelligence_percent + nanotechnology_percent

theorem target_sectors_degrees :
  target_sectors_percent * (circle_degrees / 100) = 122.4 := by
  sorry

end target_sectors_degrees_l2454_245448


namespace no_rin_is_bin_l2454_245452

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bin, Fin, and Rin
variable (Bin Fin Rin : U → Prop)

-- Premise I: All Bins are Fins
axiom all_bins_are_fins : ∀ x, Bin x → Fin x

-- Premise II: Some Rins are not Fins
axiom some_rins_not_fins : ∃ x, Rin x ∧ ¬Fin x

-- Theorem to prove
theorem no_rin_is_bin : (∀ x, Bin x → Fin x) → (∃ x, Rin x ∧ ¬Fin x) → (∀ x, Rin x → ¬Bin x) :=
sorry

end no_rin_is_bin_l2454_245452


namespace x_gt_4_sufficient_not_necessary_for_inequality_l2454_245407

theorem x_gt_4_sufficient_not_necessary_for_inequality :
  (∀ x : ℝ, x > 4 → x^2 - 4*x > 0) ∧
  (∃ x : ℝ, x^2 - 4*x > 0 ∧ ¬(x > 4)) := by
  sorry

end x_gt_4_sufficient_not_necessary_for_inequality_l2454_245407


namespace not_A_implies_not_all_right_l2454_245464

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State Ms. Carroll's promise
variable (carroll_promise : ∀ s : Student, got_all_right s → received_A s)

-- Theorem to prove
theorem not_A_implies_not_all_right :
  ∀ s : Student, ¬(received_A s) → ¬(got_all_right s) :=
sorry

end not_A_implies_not_all_right_l2454_245464


namespace car_speed_conversion_l2454_245463

/-- Conversion factor from m/s to km/h -/
def conversion_factor : ℝ := 3.6

/-- Speed of the car in m/s -/
def speed_ms : ℝ := 10

/-- Speed of the car in km/h -/
def speed_kmh : ℝ := speed_ms * conversion_factor

theorem car_speed_conversion :
  speed_kmh = 36 := by sorry

end car_speed_conversion_l2454_245463


namespace shopping_cost_calculation_l2454_245468

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let subtotal := tshirt_price * tshirt_quantity + 
                  sweater_price * sweater_quantity + 
                  jacket_price * jacket_quantity * (1 - jacket_discount)
  let total_with_tax := subtotal * (1 + sales_tax)
  total_with_tax = 504 := by sorry


end shopping_cost_calculation_l2454_245468


namespace min_calls_proof_l2454_245462

/-- Represents the minimum number of calls per month -/
def min_calls : ℕ := 66

/-- Represents the monthly rental fee in yuan -/
def rental_fee : ℚ := 12

/-- Represents the cost per call in yuan -/
def cost_per_call : ℚ := (1/5 : ℚ)

/-- Represents the minimum monthly phone bill in yuan -/
def min_monthly_bill : ℚ := 25

theorem min_calls_proof :
  (min_calls : ℚ) * cost_per_call + rental_fee > min_monthly_bill ∧
  ∀ n : ℕ, n < min_calls → (n : ℚ) * cost_per_call + rental_fee ≤ min_monthly_bill :=
by sorry

end min_calls_proof_l2454_245462


namespace dice_probability_l2454_245497

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The total number of possible outcomes when rolling seven dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to get exactly one pair with the other five dice all different -/
def one_pair_outcomes : ℕ := num_sides * (num_dice.choose 2) * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4) * (num_sides - 5)

/-- The number of ways to get exactly two pairs with the other three dice all different -/
def two_pairs_outcomes : ℕ := (num_sides.choose 2) * (num_dice.choose 2) * ((num_dice - 2).choose 2) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_pair_outcomes + two_pairs_outcomes

/-- The probability of getting at least one pair but no three of a kind when rolling seven standard six-sided dice -/
theorem dice_probability : (favorable_outcomes : ℚ) / total_outcomes = 315 / 972 := by
  sorry

end dice_probability_l2454_245497


namespace kaleb_books_l2454_245409

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end kaleb_books_l2454_245409


namespace quadratic_inequality_range_l2454_245420

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
sorry

end quadratic_inequality_range_l2454_245420


namespace min_value_theorem_l2454_245413

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 →
    2 * x + y + z ≥ 2 * Real.sqrt 3 + 2) ∧
  (∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 ∧
    2 * x + y + z = 2 * Real.sqrt 3 + 2) :=
by sorry

end min_value_theorem_l2454_245413


namespace dog_weight_multiple_l2454_245416

/-- Given the weights of three dogs (chihuahua, pitbull, and great dane), 
    prove that the great dane's weight is 3 times the pitbull's weight plus 10 pounds. -/
theorem dog_weight_multiple (c p g : ℝ) 
  (h1 : c + p + g = 439)  -- Total weight
  (h2 : p = 3 * c)        -- Pitbull's weight relation to chihuahua
  (h3 : g = 307)          -- Great dane's weight
  : ∃ m : ℝ, g = m * p + 10 ∧ m = 3 := by
  sorry

#check dog_weight_multiple

end dog_weight_multiple_l2454_245416


namespace similar_triangles_leg_l2454_245455

/-- Two similar right triangles with legs 10 and 8 in the first triangle,
    and x and 5 in the second triangle. -/
structure SimilarRightTriangles where
  x : ℝ
  similarity : (10 : ℝ) / x = 8 / 5

theorem similar_triangles_leg (t : SimilarRightTriangles) : t.x = 6.25 := by
  sorry

end similar_triangles_leg_l2454_245455


namespace surviving_positions_32_l2454_245483

/-- Represents the selection process for an international exchange event. -/
def SelectionProcess (n : ℕ) : Prop :=
  n > 0 ∧ ∃ k, 2^k = n

/-- Represents a valid initial position in the selection process. -/
def ValidPosition (n : ℕ) (p : ℕ) : Prop :=
  1 ≤ p ∧ p ≤ n

/-- Represents a position that survives all elimination rounds. -/
def SurvivingPosition (n : ℕ) (p : ℕ) : Prop :=
  ValidPosition n p ∧ ∃ k, 2^k = p

/-- The main theorem stating that positions 16 and 32 are the only surviving positions in a 32-student selection process. -/
theorem surviving_positions_32 :
  SelectionProcess 32 →
  ∀ p, SurvivingPosition 32 p ↔ (p = 16 ∨ p = 32) :=
by sorry

end surviving_positions_32_l2454_245483


namespace arithmetic_progression_condition_l2454_245478

theorem arithmetic_progression_condition 
  (a b c : ℝ) (p n k : ℕ+) : 
  (∃ (d : ℝ) (a₁ : ℝ), a = a₁ + (p - 1) * d ∧ b = a₁ + (n - 1) * d ∧ c = a₁ + (k - 1) * d) ↔ 
  (a * (n - k) + b * (k - p) + c * (p - n) = 0) ∧ 
  ((b - a) / (c - b) = (n - p : ℝ) / (k - n : ℝ)) := by
  sorry

end arithmetic_progression_condition_l2454_245478


namespace cube_minus_reciprocal_cube_l2454_245435

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 4) : x^3 - 1/x^3 = 76 := by
  sorry

end cube_minus_reciprocal_cube_l2454_245435


namespace product_of_sums_l2454_245429

theorem product_of_sums (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b + a + b = 35) (hbc : b * c + b + c = 35) (hca : c * a + c + a = 35) :
  (a + 1) * (b + 1) * (c + 1) = 216 := by
sorry

end product_of_sums_l2454_245429


namespace book_selection_probability_l2454_245403

theorem book_selection_probability :
  let n : ℕ := 12  -- Total number of books
  let k : ℕ := 6   -- Number of books each student selects
  let m : ℕ := 3   -- Number of books in common

  -- Probability of selecting exactly m books in common
  (Nat.choose n m * Nat.choose (n - m) (k - m) * Nat.choose (n - k) (k - m) : ℚ) /
  (Nat.choose n k * Nat.choose n k : ℚ) = 100 / 231 := by
sorry

end book_selection_probability_l2454_245403


namespace arrangement_count_l2454_245466

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers in each group -/
def teachers_per_group : ℕ := 1

/-- The number of students in each group -/
def students_per_group : ℕ := 2

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := 12

theorem arrangement_count :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) = total_arrangements :=
sorry

end arrangement_count_l2454_245466


namespace canoe_oar_probability_l2454_245433

theorem canoe_oar_probability (p : ℝ) :
  p ≥ 0 ∧ p ≤ 1 →
  2 * p - p^2 = 0.84 →
  p = 0.6 := by
sorry

end canoe_oar_probability_l2454_245433


namespace point_in_fourth_quadrant_l2454_245439

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If P(m, 1) is in the second quadrant, then B(-m+1, -1) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (m : ℝ) :
  isInSecondQuadrant (Point.mk m 1) → isInFourthQuadrant (Point.mk (-m + 1) (-1)) :=
by sorry

end point_in_fourth_quadrant_l2454_245439


namespace smallest_k_with_remainders_l2454_245499

theorem smallest_k_with_remainders (k : ℕ) : 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 8 = 1 ∧ 
  k % 4 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m) →
  k = 105 := by
sorry

end smallest_k_with_remainders_l2454_245499


namespace parabola_ellipse_tangent_property_l2454_245417

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola E
def parabola (x y : ℝ) : Prop := y^2 = 6*x + 15

-- Define the focus F
def F : ℝ × ℝ := (-1, 0)

-- Define a point on the parabola
def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define tangent points on the ellipse
def tangent_points (M N : ℝ × ℝ) : Prop := ellipse M.1 M.2 ∧ ellipse N.1 N.2

-- Theorem statement
theorem parabola_ellipse_tangent_property
  (A M N : ℝ × ℝ)
  (h_A : on_parabola A)
  (h_MN : tangent_points M N) :
  (∃ (t : ℝ), F.1 + t * (A.1 - F.1) = (M.1 + N.1) / 2 ∧
              F.2 + t * (A.2 - F.2) = (M.2 + N.2) / 2) ∧
  (∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
    Real.cos θ = ((M.1 + 1) * (N.1 + 1) + M.2 * N.2) /
                 (Real.sqrt ((M.1 + 1)^2 + M.2^2) * Real.sqrt ((N.1 + 1)^2 + N.2^2))) :=
sorry

end parabola_ellipse_tangent_property_l2454_245417


namespace binomial_sum_problem_l2454_245472

theorem binomial_sum_problem (a : ℚ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℚ) :
  (∀ x, (a*x + 1)^5 * (x + 2)^4 = a₀*(x + 2)^9 + a₁*(x + 2)^8 + a₂*(x + 2)^7 + 
                                   a₃*(x + 2)^6 + a₄*(x + 2)^5 + a₅*(x + 2)^4 + 
                                   a₆*(x + 2)^3 + a₇*(x + 2)^2 + a₈*(x + 2) + a₉) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 1024 →
  a₀ + a₂ + a₄ + a₆ + a₈ = (2^10 - 14^5) / 2 := by
sorry

end binomial_sum_problem_l2454_245472


namespace marcus_has_more_cards_l2454_245450

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The number of baseball cards Carter has -/
def carter_cards : ℕ := 152

/-- The difference in baseball cards between Marcus and Carter -/
def card_difference : ℕ := marcus_cards - carter_cards

theorem marcus_has_more_cards : card_difference = 58 := by
  sorry

end marcus_has_more_cards_l2454_245450


namespace fraction_equality_l2454_245428

theorem fraction_equality (a b : ℚ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end fraction_equality_l2454_245428


namespace unoccupied_area_l2454_245402

/-- The area of the region not occupied by a smaller square inside a larger square -/
theorem unoccupied_area (large_side small_side : ℝ) (h1 : large_side = 10) (h2 : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end unoccupied_area_l2454_245402


namespace power_two_divides_power_odd_minus_one_l2454_245438

theorem power_two_divides_power_odd_minus_one (k n : ℕ) (h_k_odd : Odd k) (h_n_ge_one : n ≥ 1) :
  ∃ m : ℤ, k^(2^n) - 1 = 2^(n+2) * m :=
sorry

end power_two_divides_power_odd_minus_one_l2454_245438


namespace lines_parallel_if_one_in_plane_one_parallel_to_plane_l2454_245449

-- Define the plane and lines
variable (α : Plane) (m n : Line)

-- Define the property of lines being coplanar
def coplanar (l₁ l₂ : Line) : Prop := sorry

-- Define the property of a line being contained in a plane
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- Define the property of a line being parallel to a plane
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the property of two lines being parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem lines_parallel_if_one_in_plane_one_parallel_to_plane
  (h_coplanar : coplanar m n)
  (h_m_in_α : contained_in m α)
  (h_n_parallel_α : parallel_to_plane n α) :
  parallel m n :=
sorry

end lines_parallel_if_one_in_plane_one_parallel_to_plane_l2454_245449


namespace three_positions_from_six_people_l2454_245498

def number_of_people : ℕ := 6
def number_of_positions : ℕ := 3

theorem three_positions_from_six_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 120 :=
by sorry

end three_positions_from_six_people_l2454_245498


namespace jackets_sold_after_noon_l2454_245484

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts :=
by
  sorry

#check jackets_sold_after_noon

end jackets_sold_after_noon_l2454_245484


namespace prob_heart_then_king_is_one_fiftytwo_l2454_245492

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of kings in a standard deck
def kings_in_deck : ℕ := 4

-- Define the probability of drawing a heart first and a king second
def prob_heart_then_king : ℚ := hearts_in_deck / standard_deck * kings_in_deck / (standard_deck - 1)

-- Theorem statement
theorem prob_heart_then_king_is_one_fiftytwo :
  prob_heart_then_king = 1 / standard_deck :=
by sorry

end prob_heart_then_king_is_one_fiftytwo_l2454_245492


namespace cards_remaining_l2454_245431

def initial_cards : Nat := 13
def cards_given_away : Nat := 9

theorem cards_remaining (initial : Nat) (given_away : Nat) : 
  initial = initial_cards → given_away = cards_given_away → 
  initial - given_away = 4 := by
  sorry

end cards_remaining_l2454_245431


namespace total_bill_correct_l2454_245427

/-- Represents the group composition and meal prices -/
structure GroupInfo where
  adults : Nat
  teenagers : Nat
  children : Nat
  adultMealPrice : ℚ
  teenagerMealPrice : ℚ
  childrenMealPrice : ℚ
  adultSodaPrice : ℚ
  childrenSodaPrice : ℚ
  appetizerPrice : ℚ
  dessertPrice : ℚ

/-- Represents the number of additional items ordered -/
structure AdditionalItems where
  appetizers : Nat
  desserts : Nat

/-- Represents the discount conditions -/
structure DiscountConditions where
  adultMealDiscount : ℚ
  childrenMealSodaDiscount : ℚ
  totalBillDiscount : ℚ
  minChildrenForDiscount : Nat
  teenagersPerFreeDessert : Nat
  minTotalForExtraDiscount : ℚ

/-- Calculates the total bill after all applicable discounts and special offers -/
def calculateTotalBill (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) : ℚ :=
  sorry

/-- Theorem stating that the calculated total bill matches the expected result -/
theorem total_bill_correct (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) :
  let expectedBill : ℚ := 230.70
  calculateTotalBill group items discounts = expectedBill :=
by
  sorry

end total_bill_correct_l2454_245427


namespace max_distance_from_origin_l2454_245430

def center : ℝ × ℝ := (5, -2)
def rope_length : ℝ := 12

theorem max_distance_from_origin :
  let max_dist := rope_length + Real.sqrt ((center.1 ^ 2) + (center.2 ^ 2))
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2) ≤ rope_length →
    Real.sqrt (p.1 ^ 2 + p.2 ^ 2) ≤ max_dist :=
by sorry

end max_distance_from_origin_l2454_245430


namespace quadratic_equal_roots_l2454_245496

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end quadratic_equal_roots_l2454_245496


namespace rain_hours_calculation_l2454_245465

theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
sorry

end rain_hours_calculation_l2454_245465


namespace isosceles_triangle_perimeter_l2454_245401

/-- An isosceles triangle with sides a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : (a = b ∧ a > 0) ∨ (b = c ∧ b > 0) ∨ (a = c ∧ a > 0)

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: An isosceles triangle with one side of length 6 and another of length 5 
    has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 6 ∧ (t.b = 5 ∨ t.c = 5)) ∨ (t.b = 6 ∧ (t.a = 5 ∨ t.c = 5)) ∨ (t.c = 6 ∧ (t.a = 5 ∨ t.b = 5))) →
  (perimeter t = 16 ∨ perimeter t = 17) :=
by sorry


end isosceles_triangle_perimeter_l2454_245401
