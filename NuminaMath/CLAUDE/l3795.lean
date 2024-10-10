import Mathlib

namespace angle_measure_l3795_379576

theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 90) ∧ (90 - x = 2 * x + 15) ∧ (x = 25) :=
by sorry

end angle_measure_l3795_379576


namespace boat_downstream_time_l3795_379597

theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (current_rate : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 18) 
  (h2 : current_rate = 4) 
  (h3 : distance = 5.133333333333334) : 
  (distance / (boat_speed + current_rate)) * 60 = 14 := by
sorry

end boat_downstream_time_l3795_379597


namespace lineArrangements_eq_36_l3795_379519

/-- The number of ways to arrange 3 students (who must stand together) and 2 teachers in a line -/
def lineArrangements : ℕ :=
  let studentsCount : ℕ := 3
  let teachersCount : ℕ := 2
  let unitsCount : ℕ := teachersCount + 1  -- Students count as one unit
  (Nat.factorial unitsCount) * (Nat.factorial studentsCount)

theorem lineArrangements_eq_36 : lineArrangements = 36 := by
  sorry

end lineArrangements_eq_36_l3795_379519


namespace rectangle_ratio_l3795_379577

theorem rectangle_ratio (w : ℝ) : 
  w > 0 → 
  2 * w + 2 * 10 = 30 → 
  w / 10 = 1 / 2 := by
sorry

end rectangle_ratio_l3795_379577


namespace two_digit_multiplication_swap_l3795_379520

theorem two_digit_multiplication_swap (a b c d : Nat) : 
  (a ≥ 1 ∧ a ≤ 9) →
  (b ≥ 0 ∧ b ≤ 9) →
  (c ≥ 1 ∧ c ≤ 9) →
  (d ≥ 0 ∧ d ≤ 9) →
  ((10 * a + b) * (10 * c + d) - (10 * b + a) * (10 * c + d) = 4248) →
  ((10 * a + b) * (10 * c + d) = 5369 ∨ (10 * a + b) * (10 * c + d) = 4720) :=
by sorry

end two_digit_multiplication_swap_l3795_379520


namespace sqrt_equation_solution_l3795_379595

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end sqrt_equation_solution_l3795_379595


namespace distance_from_bogula_to_bolifoyn_l3795_379566

/-- The distance from Bogula to Bolifoyn in miles -/
def total_distance : ℝ := 10

/-- The time in hours at which they approach Pigtown -/
def time_to_pigtown : ℝ := 1

/-- The additional distance traveled after Pigtown in miles -/
def additional_distance : ℝ := 5

/-- The total travel time in hours -/
def total_time : ℝ := 4

theorem distance_from_bogula_to_bolifoyn :
  ∃ (distance_to_pigtown : ℝ),
    /- After 20 minutes (1/3 hour), half of the remaining distance to Pigtown is covered -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 2 ∧
    /- The distance covered is twice less than the remaining distance to Pigtown -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 3 ∧
    /- They took another 5 miles after approaching Pigtown -/
    total_distance = distance_to_pigtown + additional_distance ∧
    /- The total travel time is 4 hours -/
    total_time * (total_distance / total_time) = total_distance := by
  sorry


end distance_from_bogula_to_bolifoyn_l3795_379566


namespace fractional_equation_solution_l3795_379538

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x ≠ 0 ∧ x + 1 ≠ 0) ∧ (1 / x = 2 / (x + 1)) ∧ x = 1 :=
by sorry

end fractional_equation_solution_l3795_379538


namespace basketball_team_starters_l3795_379543

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  (quadruplets * choose (total_players - quadruplets) (starters - 1)) = 1320 := by
  sorry

end basketball_team_starters_l3795_379543


namespace grouping_theorem_l3795_379541

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 4 men and 3 women into a group of five 
    (with at least two men and two women) and a group of two -/
def groupingWays : ℕ :=
  choose 4 2 * choose 3 2 * choose 3 1

theorem grouping_theorem : groupingWays = 54 := by sorry

end grouping_theorem_l3795_379541


namespace fourth_root_cubed_l3795_379500

theorem fourth_root_cubed (x : ℝ) : (x^(1/4))^3 = 729 → x = 6561 := by
  sorry

end fourth_root_cubed_l3795_379500


namespace largest_rhombus_diagonal_l3795_379510

/-- The diagonal of the largest rhombus inscribed in a circle with radius 10 cm is 20 cm. -/
theorem largest_rhombus_diagonal (r : ℝ) (h : r = 10) : 
  2 * r = 20 := by sorry

end largest_rhombus_diagonal_l3795_379510


namespace dan_picked_more_apples_l3795_379505

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more_apples : dan_apples - benny_apples = 7 := by
  sorry

end dan_picked_more_apples_l3795_379505


namespace food_rent_ratio_l3795_379531

/-- Esperanza's monthly financial situation -/
structure EsperanzaFinances where
  rent : ℝ
  food : ℝ
  mortgage : ℝ
  savings : ℝ
  taxes : ℝ
  salary : ℝ

/-- Conditions for Esperanza's finances -/
def validFinances (e : EsperanzaFinances) : Prop :=
  e.rent = 600 ∧
  e.mortgage = 3 * e.food ∧
  e.savings = 2000 ∧
  e.taxes = 2/5 * e.savings ∧
  e.salary = 4840 ∧
  e.salary = e.rent + e.food + e.mortgage + e.savings + e.taxes

/-- The theorem to prove -/
theorem food_rent_ratio (e : EsperanzaFinances) 
  (h : validFinances e) : e.food / e.rent = 3 / 5 := by
  sorry


end food_rent_ratio_l3795_379531


namespace hilary_kernels_to_shuck_l3795_379506

/-- Calculates the total number of kernels Hilary has to shuck --/
def total_kernels (ears_per_stalk : ℕ) (num_stalks : ℕ) (kernels_first_half : ℕ) (additional_kernels_second_half : ℕ) : ℕ :=
  let total_ears := ears_per_stalk * num_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels_second_half
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

/-- Theorem stating that Hilary has 237,600 kernels to shuck --/
theorem hilary_kernels_to_shuck :
  total_kernels 4 108 500 100 = 237600 := by
  sorry

end hilary_kernels_to_shuck_l3795_379506


namespace prob_even_sum_is_11_20_l3795_379590

/-- Represents a wheel with a certain number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  valid : total = even + odd

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- The two wheels in the game -/
def wheel1 : Wheel := ⟨5, 2, 3, rfl⟩
def wheel2 : Wheel := ⟨4, 1, 3, rfl⟩

/-- The theorem to be proved -/
theorem prob_even_sum_is_11_20 :
  prob_even wheel1 * prob_even wheel2 + prob_odd wheel1 * prob_odd wheel2 = 11 / 20 := by
  sorry

end prob_even_sum_is_11_20_l3795_379590


namespace sum_of_cubes_plus_linear_positive_l3795_379553

theorem sum_of_cubes_plus_linear_positive
  (a b c : ℝ)
  (hab : a + b > 0)
  (hac : a + c > 0)
  (hbc : b + c > 0) :
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 :=
by sorry

end sum_of_cubes_plus_linear_positive_l3795_379553


namespace remainder_theorem_l3795_379535

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 4*x^6 - 9*x^4 + 3*x^3 - 5*x^2 + 8

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + polynomial (2 : ℝ) ∧
    polynomial (2 : ℝ) = 1020 := by
  sorry

end remainder_theorem_l3795_379535


namespace probability_same_color_l3795_379598

def num_balls : ℕ := 6
def num_colors : ℕ := 3
def balls_per_color : ℕ := 2

def same_color_combinations : ℕ := num_colors

def total_combinations : ℕ := num_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 1 / 5 := by sorry

end probability_same_color_l3795_379598


namespace angle_theorem_l3795_379571

theorem angle_theorem (α β : Real) (P : Real × Real) :
  P = (3, 4) → -- Point P is (3,4)
  (∃ r : Real, r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) → -- P is on terminal side of α
  Real.cos β = 5/13 → -- cos β = 5/13
  β ∈ Set.Icc 0 (Real.pi / 2) → -- β ∈ [0, π/2]
  Real.sin α = 4/5 ∧ Real.cos (α - β) = 63/65 := by
  sorry

end angle_theorem_l3795_379571


namespace monotonic_cubic_implies_a_geq_one_l3795_379575

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = (1/3)x³ + x² + ax - 5 is monotonic for all real x, then a ≥ 1 -/
theorem monotonic_cubic_implies_a_geq_one (a : ℝ) :
  Monotonic (fun x => (1/3) * x^3 + x^2 + a*x - 5) → a ≥ 1 := by
  sorry


end monotonic_cubic_implies_a_geq_one_l3795_379575


namespace translation_result_l3795_379516

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point left by a given amount -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point up by a given amount -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The initial point A -/
def A : Point :=
  { x := 2, y := 3 }

/-- The final point after translation -/
def finalPoint : Point :=
  translateUp (translateLeft A 3) 2

theorem translation_result :
  finalPoint = { x := -1, y := 5 } := by sorry

end translation_result_l3795_379516


namespace alicia_tax_deduction_l3795_379507

/-- Calculates the local tax deduction in cents per hour given an hourly wage in dollars and a tax rate percentage. -/
def local_tax_deduction (hourly_wage : ℚ) (tax_rate_percent : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate_percent / 100)

/-- Theorem: Given Alicia's hourly wage of $25 and a local tax rate of 2%, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem alicia_tax_deduction :
  local_tax_deduction 25 2 = 50 := by
  sorry

#eval local_tax_deduction 25 2

end alicia_tax_deduction_l3795_379507


namespace tim_speed_proof_l3795_379581

/-- Represents the initial distance between Tim and Élan in miles -/
def initial_distance : ℝ := 180

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 120

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 40

theorem tim_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    t + 2*t = tim_travel_distance ∧
    t = tim_initial_speed :=
by sorry

end tim_speed_proof_l3795_379581


namespace fast_food_cost_l3795_379591

/-- The cost of items at a fast food restaurant -/
theorem fast_food_cost (H M F : ℝ) : 
  (3 * H + 5 * M + F = 23.50) → 
  (5 * H + 9 * M + F = 39.50) → 
  (2 * H + 2 * M + 2 * F = 15.00) :=
by sorry

end fast_food_cost_l3795_379591


namespace inequality_with_gcd_l3795_379589

theorem inequality_with_gcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) :
  (a + 1) / (b + 1 : ℝ) ≤ Nat.gcd a b + 1 := by
  sorry

end inequality_with_gcd_l3795_379589


namespace max_area_triangle_OPQ_l3795_379564

/-- The maximum area of triangle OPQ given the specified conditions -/
theorem max_area_triangle_OPQ :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let O : ℝ × ℝ := (0, 0)
  ∃ (M P Q : ℝ × ℝ),
    (M.1 ≠ -2 ∧ M.1 ≠ 2) →  -- M is not on the same vertical line as A or B
    (M.2 / (M.1 + 2)) * (M.2 / (M.1 - 2)) = -3/4 →  -- Product of slopes AM and BM
    (P.2 - Q.2) / (P.1 - Q.1) = 1 →  -- PQ has slope 1
    (M.1^2 / 4 + M.2^2 / 3 = 1) →  -- M is on the locus
    (P.1^2 / 4 + P.2^2 / 3 = 1) →  -- P is on the locus
    (Q.1^2 / 4 + Q.2^2 / 3 = 1) →  -- Q is on the locus
    (∀ R : ℝ × ℝ, R.1^2 / 4 + R.2^2 / 3 = 1 →  -- For all points R on the locus
      abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 ≥
      abs ((P.1 - O.1) * (R.2 - O.2) - (R.1 - O.1) * (P.2 - O.2)) / 2) →
    abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 = Real.sqrt 3 :=
by sorry

end max_area_triangle_OPQ_l3795_379564


namespace total_water_needed_is_112_l3795_379555

/-- Calculates the total gallons of water needed for Nicole's fish tanks in four weeks -/
def water_needed_in_four_weeks : ℕ :=
  let num_tanks : ℕ := 4
  let first_tank_gallons : ℕ := 8
  let num_first_type_tanks : ℕ := 2
  let num_second_type_tanks : ℕ := num_tanks - num_first_type_tanks
  let second_tank_gallons : ℕ := first_tank_gallons - 2
  let weeks : ℕ := 4
  
  let weekly_total : ℕ := 
    first_tank_gallons * num_first_type_tanks + 
    second_tank_gallons * num_second_type_tanks
  
  weekly_total * weeks

/-- Theorem stating that the total gallons of water needed in four weeks is 112 -/
theorem total_water_needed_is_112 : water_needed_in_four_weeks = 112 := by
  sorry

end total_water_needed_is_112_l3795_379555


namespace geometric_sequence_ratio_geometric_sequence_sum_ratio_l3795_379569

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) :
  geometric_sequence a₁ q (n + 1) = q * geometric_sequence a₁ q n := by sorry

theorem geometric_sequence_sum_ratio (a₁ : ℝ) :
  let q := -1/3
  (geometric_sequence a₁ q 1 + geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7) /
  (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 + geometric_sequence a₁ q 6 + geometric_sequence a₁ q 8) = -3 := by sorry

end geometric_sequence_ratio_geometric_sequence_sum_ratio_l3795_379569


namespace expression_value_l3795_379515

theorem expression_value : 
  (45 + (23 / 89) * Real.sin (π / 6)) * (4 * (3 ^ 2) - 7 * ((-2) ^ 3)) = 4186 := by
sorry

end expression_value_l3795_379515


namespace car_trip_mpg_l3795_379502

/-- Calculates the average miles per gallon for a trip given odometer readings and gas fill-ups --/
def average_mpg (initial_reading : ℕ) (final_reading : ℕ) (gas_used : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / gas_used

theorem car_trip_mpg : 
  let initial_reading : ℕ := 48500
  let second_reading : ℕ := 48800
  let final_reading : ℕ := 49350
  let first_fillup : ℕ := 8
  let second_fillup : ℕ := 10
  let third_fillup : ℕ := 15
  let total_gas_used : ℕ := second_fillup + third_fillup
  average_mpg initial_reading final_reading total_gas_used = 34 := by
  sorry

#eval average_mpg 48500 49350 25

end car_trip_mpg_l3795_379502


namespace club_officer_selection_l3795_379554

theorem club_officer_selection (total_members : Nat) (boys : Nat) (girls : Nat)
  (h1 : total_members = boys + girls)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : boys > 0)
  (h5 : girls > 0) :
  boys * girls = 216 := by
  sorry

end club_officer_selection_l3795_379554


namespace rectangular_prism_volume_l3795_379527

theorem rectangular_prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 15 → w * h = 10 → l * h = 6 →
  l * w * h = 30 := by
sorry

end rectangular_prism_volume_l3795_379527


namespace custom_operation_properties_l3795_379579

-- Define the custom operation *
noncomputable def customMul (x y : ℝ) : ℝ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem statement
theorem custom_operation_properties :
  (∀ a : ℝ, customMul (-15) (customMul 3 0) = -18) ∧
  (∀ a : ℝ, 
    (a < 0 → customMul 3 a + a = 2 * a - 3) ∧
    (a = 0 → customMul 3 a + a = 3) ∧
    (a > 0 → customMul 3 a + a = 2 * a + 3)) :=
by sorry

end custom_operation_properties_l3795_379579


namespace fraction_not_going_on_trip_l3795_379540

theorem fraction_not_going_on_trip :
  ∀ (S : ℝ) (J : ℝ),
    S > 0 →
    J = (2/3) * S →
    ((3/4) * J + (1/3) * S) / (J + S) = 5/6 := by
  sorry

end fraction_not_going_on_trip_l3795_379540


namespace sample_capacity_l3795_379593

/-- Given a sample divided into groups, prove that the sample capacity is 160
    when a certain group has a frequency of 20 and a frequency rate of 0.125. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ)
  (h1 : frequency = 20)
  (h2 : frequency_rate = 1/8)
  (h3 : (frequency : ℚ) / n = frequency_rate) :
  n = 160 := by
  sorry

end sample_capacity_l3795_379593


namespace lcm_gcf_problem_l3795_379546

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end lcm_gcf_problem_l3795_379546


namespace imaginary_sum_equals_two_l3795_379503

def i : ℂ := Complex.I

theorem imaginary_sum_equals_two :
  i^15 + i^20 + i^25 + i^30 + i^35 + i^40 = (2 : ℂ) :=
by sorry

end imaginary_sum_equals_two_l3795_379503


namespace parallelogram_distance_l3795_379511

/-- Given a parallelogram with base 10 feet, height 30 feet, and side length 60 feet,
    prove that the distance between the 60-foot sides is 5 feet. -/
theorem parallelogram_distance (base height side : ℝ) 
  (h_base : base = 10)
  (h_height : height = 30)
  (h_side : side = 60) :
  (base * height) / side = 5 := by
  sorry

end parallelogram_distance_l3795_379511


namespace stock_price_increase_l3795_379570

/-- Calculate the percent increase in stock price -/
theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end stock_price_increase_l3795_379570


namespace employee_distribution_percentage_difference_l3795_379544

theorem employee_distribution_percentage_difference :
  let total_degrees : ℝ := 360
  let manufacturing_degrees : ℝ := 162
  let sales_degrees : ℝ := 108
  let research_degrees : ℝ := 54
  let admin_degrees : ℝ := 36
  let manufacturing_percent := (manufacturing_degrees / total_degrees) * 100
  let sales_percent := (sales_degrees / total_degrees) * 100
  let research_percent := (research_degrees / total_degrees) * 100
  let admin_percent := (admin_degrees / total_degrees) * 100
  let max_percent := max manufacturing_percent (max sales_percent (max research_percent admin_percent))
  let min_percent := min manufacturing_percent (min sales_percent (min research_percent admin_percent))
  (max_percent - min_percent) = 35 := by
  sorry

end employee_distribution_percentage_difference_l3795_379544


namespace gcd_108_45_l3795_379518

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by sorry

end gcd_108_45_l3795_379518


namespace range_of_m_l3795_379526

/-- Given sets A and B, where B is a subset of A, prove that m ≤ 0 -/
theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 3}
  B ⊆ A → m ≤ 0 := by sorry

end range_of_m_l3795_379526


namespace third_wall_length_l3795_379514

/-- Calculates the length of the third wall in a hall of mirrors. -/
theorem third_wall_length
  (total_glass : ℝ)
  (wall1_length wall1_height : ℝ)
  (wall2_length wall2_height : ℝ)
  (wall3_height : ℝ)
  (h1 : total_glass = 960)
  (h2 : wall1_length = 30 ∧ wall1_height = 12)
  (h3 : wall2_length = 30 ∧ wall2_height = 12)
  (h4 : wall3_height = 12)
  : ∃ (wall3_length : ℝ),
    total_glass = wall1_length * wall1_height + wall2_length * wall2_height + wall3_length * wall3_height
    ∧ wall3_length = 20 :=
by
  sorry

end third_wall_length_l3795_379514


namespace x_plus_two_equals_seven_implies_x_equals_five_l3795_379524

theorem x_plus_two_equals_seven_implies_x_equals_five :
  ∀ x : ℝ, x + 2 = 7 → x = 5 := by
  sorry

end x_plus_two_equals_seven_implies_x_equals_five_l3795_379524


namespace amys_flash_drive_storage_l3795_379586

/-- Calculates the total storage space used on Amy's flash drive -/
def total_storage_space (music_files : ℝ) (music_size : ℝ) (video_files : ℝ) (video_size : ℝ) (picture_files : ℝ) (picture_size : ℝ) : ℝ :=
  music_files * music_size + video_files * video_size + picture_files * picture_size

/-- Theorem: The total storage space used on Amy's flash drive is 1116 MB -/
theorem amys_flash_drive_storage :
  total_storage_space 4 5 21 50 23 2 = 1116 := by
  sorry

end amys_flash_drive_storage_l3795_379586


namespace find_number_l3795_379557

theorem find_number : ∃ x : ℝ, (x / 23 - 67) * 2 = 102 ∧ x = 2714 := by
  sorry

end find_number_l3795_379557


namespace N_is_composite_l3795_379588

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end N_is_composite_l3795_379588


namespace expression_simplification_and_evaluation_l3795_379525

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x^2 - 3*x - 4 = 0 → x ≠ -1 →
  (2 - (x - 1) / (x + 1)) / ((x^2 + 6*x + 9) / (x^2 - 1)) = (x - 1) / (x + 3) ∧
  (x - 1) / (x + 3) = 3 / 7 :=
by
  sorry

end expression_simplification_and_evaluation_l3795_379525


namespace total_glows_is_569_l3795_379592

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def time_duration : ℕ := 4969

/-- The interval at which Light A glows, in seconds -/
def light_a_interval : ℕ := 16

/-- The interval at which Light B glows, in seconds -/
def light_b_interval : ℕ := 35

/-- The interval at which Light C glows, in seconds -/
def light_c_interval : ℕ := 42

/-- The number of times Light A glows -/
def light_a_glows : ℕ := time_duration / light_a_interval

/-- The number of times Light B glows -/
def light_b_glows : ℕ := time_duration / light_b_interval

/-- The number of times Light C glows -/
def light_c_glows : ℕ := time_duration / light_c_interval

/-- The total number of glows for all light sources combined -/
def total_glows : ℕ := light_a_glows + light_b_glows + light_c_glows

theorem total_glows_is_569 : total_glows = 569 := by
  sorry

end total_glows_is_569_l3795_379592


namespace function_negative_on_interval_l3795_379522

/-- The function f(x) = x^2 + mx - 1 is negative on [m, m+1] iff m is in (-√2/2, 0) -/
theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-(Real.sqrt 2)/2) 0 :=
sorry

end function_negative_on_interval_l3795_379522


namespace parallel_planes_lines_l3795_379580

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_lines 
  (α β : Plane) (m n : Line) :
  parallel α β →
  line_parallel_plane m α →
  line_parallel_line n m →
  ¬ line_in_plane n β →
  line_parallel_plane n β :=
by sorry

end parallel_planes_lines_l3795_379580


namespace impossible_all_tails_l3795_379508

/-- Represents a 4x4 grid of binary values -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents the possible flip operations -/
inductive FlipOperation
| Row : Fin 4 → FlipOperation
| Column : Fin 4 → FlipOperation
| Diagonal : Bool → Fin 4 → FlipOperation

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  Matrix.of (fun i j => if i = 0 ∧ j < 2 then true else false)

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if all values in the grid are false (tails) -/
def allTails (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Main theorem: It's impossible to reach all tails from the initial configuration -/
theorem impossible_all_tails :
  ¬∃ (ops : List FlipOperation), allTails (ops.foldl applyFlip initialGrid) :=
  sorry

end impossible_all_tails_l3795_379508


namespace conical_flask_height_l3795_379558

/-- The height of a conical flask given water depths in two positions -/
theorem conical_flask_height (h : ℝ) : 
  (h > 0) →  -- The height is positive
  (h^3 - (h-1)^3 = 8) →  -- Volume equation derived from the two water depths
  h = 1/2 + Real.sqrt 93 / 6 := by
sorry

end conical_flask_height_l3795_379558


namespace marks_animals_legs_l3795_379561

/-- The number of legs of all animals owned by Mark -/
def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

/-- Theorem stating the total number of legs of Mark's animals -/
theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
  sorry

#check marks_animals_legs

end marks_animals_legs_l3795_379561


namespace reflection_result_l3795_379560

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 2, p.1 - 2)

/-- The final position of point C after two reflections -/
def C_double_prime : ℝ × ℝ :=
  reflect_line (reflect_y_axis (5, 3))

theorem reflection_result :
  C_double_prime = (5, -7) :=
by sorry

end reflection_result_l3795_379560


namespace adams_age_problem_l3795_379585

theorem adams_age_problem :
  ∃! x : ℕ,
    x > 0 ∧
    ∃ m : ℕ, x - 2 = m ^ 2 ∧
    ∃ n : ℕ, x + 2 = n ^ 3 ∧
    x = 6 := by
  sorry

end adams_age_problem_l3795_379585


namespace fraction_decomposition_l3795_379556

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 4/3) :
  (7 * x - 13) / (3 * x^2 + 2 * x - 8) = 27 / (10 * (x + 2)) - 11 / (10 * (3 * x - 4)) := by
  sorry

end fraction_decomposition_l3795_379556


namespace apple_weight_l3795_379504

theorem apple_weight (total_weight orange_weight grape_weight strawberry_weight : ℝ) 
  (h1 : total_weight = 10)
  (h2 : orange_weight = 1)
  (h3 : grape_weight = 3)
  (h4 : strawberry_weight = 3) :
  total_weight - (orange_weight + grape_weight + strawberry_weight) = 3 :=
by sorry

end apple_weight_l3795_379504


namespace cubic_equation_proof_l3795_379572

theorem cubic_equation_proof :
  let f : ℝ → ℝ := fun x ↦ x^3 - 5*x - 2
  ∃ (x₁ x₂ x₃ : ℝ),
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁ * x₂ * x₃ = x₁ + x₂ + x₃ + 2) ∧
    (x₁^2 + x₂^2 + x₃^2 = 10) ∧
    (x₁^3 + x₂^3 + x₃^3 = 6) :=
by sorry

#check cubic_equation_proof

end cubic_equation_proof_l3795_379572


namespace simplify_expression_l3795_379529

theorem simplify_expression (r : ℝ) : 150 * r - 70 * r + 25 = 80 * r + 25 := by
  sorry

end simplify_expression_l3795_379529


namespace max_volume_parallelepiped_l3795_379552

/-- The volume of a rectangular parallelepiped with square base of side length x
    and lateral faces with perimeter 6 -/
def volume (x : ℝ) : ℝ := x^2 * (3 - x)

/-- The maximum volume of a rectangular parallelepiped with square base
    and lateral faces with perimeter 6 is 4 -/
theorem max_volume_parallelepiped :
  ∃ (x : ℝ), x > 0 ∧ x < 3 ∧
  (∀ (y : ℝ), y > 0 → y < 3 → volume y ≤ volume x) ∧
  volume x = 4 := by sorry

end max_volume_parallelepiped_l3795_379552


namespace sqrt_eight_equals_two_sqrt_two_l3795_379583

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_equals_two_sqrt_two_l3795_379583


namespace guessing_game_scores_l3795_379547

-- Define the players and their scores
def Hajar : ℕ := 42
def Farah : ℕ := Hajar + 24
def Sami : ℕ := Farah + 18

-- Theorem statement
theorem guessing_game_scores :
  Hajar = 42 ∧ Farah = 66 ∧ Sami = 84 ∧
  Farah - Hajar = 24 ∧ Sami - Farah = 18 ∧
  Farah > Hajar ∧ Sami > Hajar :=
by
  sorry


end guessing_game_scores_l3795_379547


namespace cos_4theta_from_exp_l3795_379573

theorem cos_4theta_from_exp (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 8) / 3 → 
  Real.cos (4 * θ) = 17 / 81 := by
sorry

end cos_4theta_from_exp_l3795_379573


namespace house_expansion_l3795_379517

/-- Given two houses with areas 5200 and 7300 square feet, if their total area
    after expanding the smaller house is 16000 square feet, then the expansion
    size is 3500 square feet. -/
theorem house_expansion (small_house large_house expanded_total : ℕ)
    (h1 : small_house = 5200)
    (h2 : large_house = 7300)
    (h3 : expanded_total = 16000)
    (h4 : expanded_total = small_house + large_house + expansion_size) :
    expansion_size = 3500 := by
  sorry

end house_expansion_l3795_379517


namespace team_a_win_probability_l3795_379533

theorem team_a_win_probability (p : ℝ) (h : p = 2/3) : 
  p^2 + p^2 * (1 - p) = 20/27 := by
  sorry

end team_a_win_probability_l3795_379533


namespace triangle_base_length_l3795_379501

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) :
  height = 10 →
  area = 50 →
  area = (base * height) / 2 →
  base = 10 :=
by
  sorry

end triangle_base_length_l3795_379501


namespace existence_of_k_with_n_prime_factors_l3795_379549

theorem existence_of_k_with_n_prime_factors 
  (m n : ℕ+) : 
  ∃ k : ℕ+, ∃ p : Finset ℕ, 
    (∀ x ∈ p, Nat.Prime x) ∧ 
    (Finset.card p ≥ n) ∧ 
    (∀ x ∈ p, x ∣ (2^(k:ℕ) - m)) :=
by
  sorry

end existence_of_k_with_n_prime_factors_l3795_379549


namespace f_unique_solution_l3795_379534

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

theorem f_unique_solution :
  ∃! x, f x = 1/4 ∧ x ∈ Set.univ := by sorry

end f_unique_solution_l3795_379534


namespace quadratic_inequality_solution_l3795_379574

-- Define the quadratic function
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x - 6

-- Define the solution set
def solution_set (p : ℝ) : Set ℝ := {x : ℝ | f p x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (p : ℝ) :
  solution_set p = {x : ℝ | -3 < x ∧ x < 2} → p = -2 :=
by sorry

end quadratic_inequality_solution_l3795_379574


namespace rectangle_area_relationship_l3795_379594

/-- Theorem: For a rectangle with area 4 and side lengths x and y, y = 4/x where x > 0 -/
theorem rectangle_area_relationship (x y : ℝ) (h1 : x > 0) (h2 : x * y = 4) : y = 4 / x := by
  sorry

end rectangle_area_relationship_l3795_379594


namespace jake_weight_loss_l3795_379562

theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) (weight_loss : ℕ) : 
  total_weight = 290 → 
  jake_weight = 196 → 
  jake_weight - weight_loss = 2 * (total_weight - jake_weight) → 
  weight_loss = 8 := by
  sorry

end jake_weight_loss_l3795_379562


namespace hyperbola_equilateral_focus_l3795_379551

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the origin to the right focus is 6 and 
    the asymptote forms an equilateral triangle with the origin and the right focus,
    then a = 3 and b = 3√3 -/
theorem hyperbola_equilateral_focus (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := 6  -- distance from origin to right focus
  let slope := b / a  -- slope of asymptote
  c^2 = a^2 + b^2 →  -- focus property of hyperbola
  slope = Real.sqrt 3 →  -- equilateral triangle condition
  (a = 3 ∧ b = 3 * Real.sqrt 3) := by
sorry

end hyperbola_equilateral_focus_l3795_379551


namespace integer_roots_cubic_l3795_379521

theorem integer_roots_cubic (b : ℤ) : 
  (∃ x : ℤ, x^3 - 2*x^2 + b*x + 6 = 0) ↔ b ∈ ({-25, -7, -5, 3, 13, 47} : Set ℤ) := by
  sorry

end integer_roots_cubic_l3795_379521


namespace investment_percentage_l3795_379568

/-- Proves that given the investment conditions, the percentage of the other investment is 7% -/
theorem investment_percentage (total_investment : ℝ) (investment_at_8_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 22000)
  (h2 : investment_at_8_percent = 17000)
  (h3 : total_interest = 1710) :
  (total_interest - investment_at_8_percent * 0.08) / (total_investment - investment_at_8_percent) = 0.07 := by
  sorry


end investment_percentage_l3795_379568


namespace total_books_eq_read_plus_unread_l3795_379596

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 20

/-- The number of books yet to be read -/
def unread_books : ℕ := 5

/-- The number of books already read -/
def read_books : ℕ := 15

/-- Theorem stating that the total number of books is the sum of read and unread books -/
theorem total_books_eq_read_plus_unread : 
  total_books = read_books + unread_books := by
  sorry

end total_books_eq_read_plus_unread_l3795_379596


namespace dot_product_is_two_l3795_379578

/-- A rhombus with side length 2 and angle BAC of 60° -/
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (is_rhombus : sorry)
  (side_length : sorry)
  (angle_BAC : sorry)

/-- The dot product of vectors BC and AC in the given rhombus -/
def dot_product_BC_AC (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The dot product of vectors BC and AC in the given rhombus is 2 -/
theorem dot_product_is_two (r : Rhombus) : dot_product_BC_AC r = 2 :=
  sorry

end dot_product_is_two_l3795_379578


namespace total_goals_in_five_matches_l3795_379584

/-- A football player's goal scoring record over 5 matches -/
structure FootballPlayer where
  /-- The average number of goals per match before the fifth match -/
  initial_average : ℝ
  /-- The number of goals scored in the fifth match -/
  fifth_match_goals : ℕ
  /-- The increase in average goals after the fifth match -/
  average_increase : ℝ

/-- Theorem stating the total number of goals scored over 5 matches -/
theorem total_goals_in_five_matches (player : FootballPlayer)
    (h1 : player.fifth_match_goals = 2)
    (h2 : player.average_increase = 0.3) :
    (player.initial_average * 4 + player.fifth_match_goals : ℝ) = 4 := by
  sorry

end total_goals_in_five_matches_l3795_379584


namespace unique_solution_quadratic_linear_l3795_379537

theorem unique_solution_quadratic_linear (m : ℝ) :
  (∃! x : ℝ, x^2 = 4*x + m) ↔ m = -4 :=
sorry

end unique_solution_quadratic_linear_l3795_379537


namespace sequence_next_terms_l3795_379567

def sequence1 : List ℕ := [7, 11, 19, 35]
def sequence2 : List ℕ := [1, 4, 9, 16, 25]

def next_in_sequence1 (seq : List ℕ) : ℕ :=
  let diffs := List.zipWith (·-·) (seq.tail) seq
  let last_diff := diffs.getLast!
  seq.getLast! + (2 * last_diff)

def next_in_sequence2 (seq : List ℕ) : ℕ :=
  (seq.length + 1) ^ 2

theorem sequence_next_terms :
  next_in_sequence1 sequence1 = 67 ∧ next_in_sequence2 sequence2 = 36 := by
  sorry

end sequence_next_terms_l3795_379567


namespace largest_divisor_of_difference_of_squares_l3795_379536

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Even a → Even b → b < a → 
    k ∣ (a^2 - b^2) ∧ k = 16 ∧ ∀ (l : ℕ), (∀ (x y : ℕ), Even x → Even y → y < x → l ∣ (x^2 - y^2)) → l ≤ k) :=
sorry

end largest_divisor_of_difference_of_squares_l3795_379536


namespace fixed_point_on_line_l3795_379532

theorem fixed_point_on_line (m : ℝ) : 
  let x : ℝ := -1
  let y : ℝ := -1/2
  (m^2 + 6*m + 3) * x - (2*m^2 + 18*m + 2) * y - 3*m + 2 = 0 := by
  sorry

end fixed_point_on_line_l3795_379532


namespace student_distribution_l3795_379582

theorem student_distribution (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) :
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = (Nat.choose n m) * (Nat.choose (n - m) m) * 1 :=
sorry

end student_distribution_l3795_379582


namespace tuesday_attendance_proof_l3795_379542

/-- The number of people who attended class on Tuesday -/
def tuesday_attendance : ℕ := sorry

/-- The number of people who attended class on Monday -/
def monday_attendance : ℕ := 10

/-- The number of people who attended class on Wednesday, Thursday, and Friday -/
def wednesday_to_friday_attendance : ℕ := 10

/-- The total number of days -/
def total_days : ℕ := 5

/-- The average attendance over all days -/
def average_attendance : ℕ := 11

theorem tuesday_attendance_proof :
  tuesday_attendance = 15 :=
by
  have h1 : monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance = average_attendance * total_days :=
    sorry
  sorry

end tuesday_attendance_proof_l3795_379542


namespace scores_with_two_ways_exist_l3795_379548

/-- Represents a scoring configuration for a test -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents a possible answer combination -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + 
  answers.unanswered * config.unanswered_points +
  answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given configuration -/
def is_valid_combination (config : ScoringConfig) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = config.total_questions

/-- Defines the existence of scores with exactly two ways to achieve them -/
def exists_scores_with_two_ways (config : ScoringConfig) : Prop :=
  ∃ S : ℕ, 
    0 ≤ S ∧ S ≤ 175 ∧
    (∃ (a b : AnswerCombination),
      a ≠ b ∧
      is_valid_combination config a ∧
      is_valid_combination config b ∧
      calculate_score config a = S ∧
      calculate_score config b = S ∧
      ∀ c : AnswerCombination, 
        is_valid_combination config c ∧ calculate_score config c = S → (c = a ∨ c = b))

/-- The main theorem to prove -/
theorem scores_with_two_ways_exist : 
  let config : ScoringConfig := {
    total_questions := 25,
    correct_points := 7,
    unanswered_points := 3,
    incorrect_points := 0
  }
  exists_scores_with_two_ways config := by
  sorry

end scores_with_two_ways_exist_l3795_379548


namespace cos_plus_sin_implies_cos_double_angle_l3795_379550

theorem cos_plus_sin_implies_cos_double_angle 
  (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.cos (2 * θ) = -527/625 := by
  sorry

end cos_plus_sin_implies_cos_double_angle_l3795_379550


namespace product_of_three_numbers_l3795_379565

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 3 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 176 := by sorry

end product_of_three_numbers_l3795_379565


namespace min_value_sum_product_l3795_379530

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 := by
  sorry

end min_value_sum_product_l3795_379530


namespace carol_first_six_prob_l3795_379563

/-- The probability of rolling a number other than 6 on a fair six-sided die. -/
def prob_not_six : ℚ := 5/6

/-- The probability of rolling a 6 on a fair six-sided die. -/
def prob_six : ℚ := 1/6

/-- The number of players before Carol. -/
def players_before_carol : ℕ := 2

/-- The total number of players. -/
def total_players : ℕ := 4

/-- The probability that Carol is the first to roll a six in the dice game. -/
theorem carol_first_six_prob : 
  (prob_not_six^players_before_carol * prob_six) / (1 - prob_not_six^total_players) = 125/671 := by
  sorry

end carol_first_six_prob_l3795_379563


namespace fraction_inequality_l3795_379512

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : c < d) (h4 : d < 0) : 
  d / a < c / a :=
by sorry

end fraction_inequality_l3795_379512


namespace matrix_equation_proof_l3795_379599

theorem matrix_equation_proof : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^3 - 3 • M^2 + 4 • M = !![7, 14; 3.5, 7] := by sorry

end matrix_equation_proof_l3795_379599


namespace min_real_roots_2010_l3795_379545

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := { p : Polynomial ℝ // p.degree = 2010 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The theorem statement -/
theorem min_real_roots_2010 (p : RealPolynomial2010) 
  (h : distinctAbsValues p = 1010) : 
  realRootCount p ≥ 10 := sorry

end min_real_roots_2010_l3795_379545


namespace ways_to_top_teaching_building_l3795_379528

/-- A building with multiple floors and staircases -/
structure Building where
  floors : ℕ
  staircases_per_floor : ℕ

/-- The number of ways to go from the bottom floor to the top floor -/
def ways_to_top (b : Building) : ℕ :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The specific building in the problem -/
def teaching_building : Building :=
  { floors := 5, staircases_per_floor := 2 }

theorem ways_to_top_teaching_building :
  ways_to_top teaching_building = 2^4 := by
  sorry

#eval ways_to_top teaching_building

end ways_to_top_teaching_building_l3795_379528


namespace young_in_specific_sample_l3795_379523

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  young_population : ℕ
  sample_size : ℕ

/-- Calculates the number of young people in a stratified sample -/
def young_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.young_population) / s.total_population

/-- Theorem stating the number of young people in the specific stratified sample -/
theorem young_in_specific_sample :
  let s : StratifiedSample := {
    total_population := 108,
    young_population := 51,
    sample_size := 36
  }
  young_in_sample s = 17 := by
  sorry

end young_in_specific_sample_l3795_379523


namespace sum_234_142_in_base4_l3795_379539

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits represents a valid base 4 number -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem sum_234_142_in_base4 :
  let sum := 234 + 142
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 3, 0] := by
  sorry

end sum_234_142_in_base4_l3795_379539


namespace mango_rate_calculation_l3795_379513

def grape_weight : ℝ := 7
def grape_rate : ℝ := 68
def mango_weight : ℝ := 9
def total_paid : ℝ := 908

theorem mango_rate_calculation :
  (total_paid - grape_weight * grape_rate) / mango_weight = 48 := by
  sorry

end mango_rate_calculation_l3795_379513


namespace factorization_difference_l3795_379559

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
sorry

end factorization_difference_l3795_379559


namespace students_taking_neither_music_nor_art_l3795_379587

theorem students_taking_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 20) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 470 := by
  sorry

end students_taking_neither_music_nor_art_l3795_379587


namespace ratio_of_sum_and_difference_l3795_379509

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_sum_diff : a + b = 4 * (a - b)) : a / b = 5 / 3 := by
  sorry

end ratio_of_sum_and_difference_l3795_379509
