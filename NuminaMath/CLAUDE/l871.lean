import Mathlib

namespace percentage_difference_l871_87163

theorem percentage_difference : 
  (0.80 * 170 : ℝ) - (0.35 * 300 : ℝ) = 31 := by sorry

end percentage_difference_l871_87163


namespace tangent_line_y_intercept_l871_87179

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle

/-- The y-intercept of a line -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  let line : TangentLine := { circle1 := c1, circle2 := c2 }
  yIntercept line = 2 * Real.sqrt 82 := by sorry

end tangent_line_y_intercept_l871_87179


namespace michelle_savings_l871_87100

/-- Represents the number of $100 bills Michelle has after exchanging her savings -/
def number_of_bills : ℕ := 8

/-- Represents the value of each bill in dollars -/
def bill_value : ℕ := 100

/-- Theorem stating that Michelle's total savings equal $800 -/
theorem michelle_savings : number_of_bills * bill_value = 800 := by
  sorry

end michelle_savings_l871_87100


namespace computer_literate_female_employees_l871_87192

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℝ)
  (male_computer_literate_percentage : ℝ)
  (total_computer_literate_percentage : ℝ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 0.6)
  (h_male_cl : male_computer_literate_percentage = 0.5)
  (h_total_cl : total_computer_literate_percentage = 0.62) :
  ⌊female_percentage * total_employees - 
   (1 - female_percentage) * male_computer_literate_percentage * total_employees⌋ = 504 :=
by sorry

end computer_literate_female_employees_l871_87192


namespace johns_car_repair_cost_l871_87106

/-- Calculates the total cost of car repairs including sales tax -/
def total_repair_cost (engine_labor_rate : ℝ) (engine_labor_hours : ℝ) (engine_part_cost : ℝ)
                      (brake_labor_rate : ℝ) (brake_labor_hours : ℝ) (brake_part_cost : ℝ)
                      (tire_labor_rate : ℝ) (tire_labor_hours : ℝ) (tire_cost : ℝ)
                      (sales_tax_rate : ℝ) : ℝ :=
  let engine_cost := engine_labor_rate * engine_labor_hours + engine_part_cost
  let brake_cost := brake_labor_rate * brake_labor_hours + brake_part_cost
  let tire_cost := tire_labor_rate * tire_labor_hours + tire_cost
  let total_before_tax := engine_cost + brake_cost + tire_cost
  let tax_amount := sales_tax_rate * total_before_tax
  total_before_tax + tax_amount

/-- Theorem stating that the total repair cost for John's car is $5238 -/
theorem johns_car_repair_cost :
  total_repair_cost 75 16 1200 85 10 800 50 4 600 0.08 = 5238 := by
  sorry

end johns_car_repair_cost_l871_87106


namespace divisibility_of_10_pow_6_minus_1_l871_87164

theorem divisibility_of_10_pow_6_minus_1 :
  ∃ (a b c d : ℕ), 10^6 - 1 = 7 * a ∧ 10^6 - 1 = 13 * b ∧ 10^6 - 1 = 91 * c ∧ 10^6 - 1 = 819 * d :=
by sorry

end divisibility_of_10_pow_6_minus_1_l871_87164


namespace completing_square_transformation_l871_87171

theorem completing_square_transformation (x : ℝ) : 
  (x^2 + 6*x - 4 = 0) ↔ ((x + 3)^2 = 13) :=
by sorry

end completing_square_transformation_l871_87171


namespace door_cost_ratio_l871_87181

theorem door_cost_ratio (bedroom_doors : ℕ) (outside_doors : ℕ) 
  (outside_door_cost : ℚ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  outside_door_cost = 20 →
  total_cost = 70 →
  ∃ (bedroom_door_cost : ℚ),
    bedroom_doors * bedroom_door_cost + outside_doors * outside_door_cost = total_cost ∧
    bedroom_door_cost / outside_door_cost = 1 / 2 := by
  sorry

end door_cost_ratio_l871_87181


namespace intersection_condition_l871_87158

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end intersection_condition_l871_87158


namespace coal_pile_remaining_l871_87113

theorem coal_pile_remaining (total : ℝ) (used : ℝ) (remaining : ℝ) : 
  used = (4 : ℝ) / 10 * total → remaining = (6 : ℝ) / 10 * total :=
by
  sorry

end coal_pile_remaining_l871_87113


namespace polynomial_integer_coefficients_l871_87129

theorem polynomial_integer_coefficients (a b c : ℚ) : 
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) → 
  (∃ (a' b' c' : ℤ), a = a' ∧ b = b' ∧ c = c') := by
  sorry

end polynomial_integer_coefficients_l871_87129


namespace shelves_filled_with_carvings_l871_87120

def wood_carvings_per_shelf : ℕ := 8
def total_wood_carvings : ℕ := 56

theorem shelves_filled_with_carvings :
  total_wood_carvings / wood_carvings_per_shelf = 7 := by
  sorry

end shelves_filled_with_carvings_l871_87120


namespace circle_sum_inequality_l871_87156

theorem circle_sum_inequality (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (i : Fin 100), (nums i + nums ((i + 1) % 100)) < (nums ((i + 2) % 100) + nums ((i + 3) % 100)) :=
sorry

end circle_sum_inequality_l871_87156


namespace closest_root_l871_87168

def options : List ℤ := [2, 3, 4, 5]

theorem closest_root (x : ℝ) (h : x^3 - 9 = 16) : 
  3 = (options.argmin (λ y => |y - x|)).get sorry :=
sorry

end closest_root_l871_87168


namespace quadratic_sequence_problem_l871_87155

theorem quadratic_sequence_problem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 2)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 52) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 65 := by
  sorry

end quadratic_sequence_problem_l871_87155


namespace sin_315_degrees_l871_87185

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end sin_315_degrees_l871_87185


namespace min_rubber_bands_specific_l871_87196

/-- Calculates the minimum number of rubber bands needed to tie matches and cotton swabs into bundles. -/
def min_rubber_bands (total_matches : ℕ) (total_swabs : ℕ) (matches_per_bundle : ℕ) (swabs_per_bundle : ℕ) (bands_per_bundle : ℕ) : ℕ :=
  let match_bundles := total_matches / matches_per_bundle
  let swab_bundles := total_swabs / swabs_per_bundle
  (match_bundles + swab_bundles) * bands_per_bundle

/-- Theorem stating that given the specific conditions, the minimum number of rubber bands needed is 14. -/
theorem min_rubber_bands_specific : 
  min_rubber_bands 40 34 8 12 2 = 14 := by
  sorry

end min_rubber_bands_specific_l871_87196


namespace arithmetic_sequence_sum_l871_87199

/-- Given an arithmetic sequence {a_n} where a_4 + a_8 = 16, prove that a_2 + a_6 + a_10 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 4 + a 8 = 16) →                               -- given condition
  (a 2 + a 6 + a 10 = 24) :=                       -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l871_87199


namespace parallel_condition_l871_87152

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The condition ab = 1 -/
def condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b = 1

theorem parallel_condition (l1 l2 : Line) :
  (parallel l1 l2 → condition l1 l2) ∧
  ¬(condition l1 l2 → parallel l1 l2) :=
sorry

end parallel_condition_l871_87152


namespace number_division_problem_l871_87132

theorem number_division_problem :
  let sum := 3927 + 2873
  let diff := 3927 - 2873
  let quotient := 3 * diff
  ∀ (N r : ℕ), 
    N / sum = quotient ∧ 
    N % sum = r ∧ 
    r < sum →
    N = 21481600 + r :=
by sorry

end number_division_problem_l871_87132


namespace radius_circle_q_is_ten_l871_87109

/-- A triangle ABC with two equal sides and a circle P tangent to two sides -/
structure IsoscelesTriangleWithTangentCircle where
  /-- The length of the equal sides AB and AC -/
  side_length : ℝ
  /-- The length of the base BC -/
  base_length : ℝ
  /-- The radius of the circle P tangent to AC and BC -/
  circle_p_radius : ℝ

/-- The radius of circle Q, which is externally tangent to P and tangent to AB and BC -/
def radius_circle_q (t : IsoscelesTriangleWithTangentCircle) : ℝ := sorry

/-- The main theorem: In the given configuration, the radius of circle Q is 10 -/
theorem radius_circle_q_is_ten
  (t : IsoscelesTriangleWithTangentCircle)
  (h1 : t.side_length = 120)
  (h2 : t.base_length = 90)
  (h3 : t.circle_p_radius = 30) :
  radius_circle_q t = 10 := by sorry

end radius_circle_q_is_ten_l871_87109


namespace toby_friends_percentage_l871_87194

def toby_boy_friends : ℕ := 33
def toby_girl_friends : ℕ := 27

theorem toby_friends_percentage :
  (toby_boy_friends : ℚ) / (toby_boy_friends + toby_girl_friends : ℚ) * 100 = 55 := by
  sorry

end toby_friends_percentage_l871_87194


namespace cube_diagonal_l871_87133

theorem cube_diagonal (V : ℝ) (A : ℝ) (s : ℝ) (d : ℝ) : 
  V = 384 → A = 384 → V = s^3 → A = 6 * s^2 → d = s * Real.sqrt 3 → d = 8 * Real.sqrt 3 := by
  sorry

end cube_diagonal_l871_87133


namespace zero_exponent_eq_one_l871_87103

theorem zero_exponent_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_eq_one_l871_87103


namespace debt_payment_problem_l871_87121

/-- Proves that the amount of each of the first 20 payments is $410 given the problem conditions. -/
theorem debt_payment_problem (total_payments : ℕ) (first_payments : ℕ) (payment_increase : ℕ) (average_payment : ℕ) :
  total_payments = 65 →
  first_payments = 20 →
  payment_increase = 65 →
  average_payment = 455 →
  ∃ (x : ℕ),
    x * first_payments + (x + payment_increase) * (total_payments - first_payments) = average_payment * total_payments ∧
    x = 410 :=
by sorry

end debt_payment_problem_l871_87121


namespace cos_two_pi_seventh_inequality_l871_87122

theorem cos_two_pi_seventh_inequality (a : ℝ) :
  a = Real.cos ((2 * Real.pi) / 7) →
  0 < (1 : ℝ) / 2 ∧ (1 : ℝ) / 2 < a ∧ a < Real.sqrt 2 / 2 ∧ Real.sqrt 2 / 2 < 1 →
  2^(a - 1/2) < 2 * a := by
  sorry

end cos_two_pi_seventh_inequality_l871_87122


namespace limit_f_at_zero_l871_87159

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 2) - Real.sqrt 2) / Real.sin (3 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds ((Real.sqrt 2) / 24)) :=
sorry

end limit_f_at_zero_l871_87159


namespace other_amount_theorem_l871_87165

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem other_amount_theorem :
  let initial_principal : ℝ := 200
  let initial_rate : ℝ := 0.1
  let initial_time : ℝ := 12
  let other_rate : ℝ := 0.12
  let other_time : ℝ := 2
  let other_principal : ℝ := 1000
  simple_interest initial_principal initial_rate initial_time =
    simple_interest other_principal other_rate other_time := by
  sorry

end other_amount_theorem_l871_87165


namespace two_by_two_squares_count_l871_87137

theorem two_by_two_squares_count (grid_size : ℕ) (cuts : ℕ) (figures : ℕ) 
  (h1 : grid_size = 100)
  (h2 : cuts = 10000)
  (h3 : figures = 2500) : 
  ∃ (x : ℕ), x = 2300 ∧ 
  (8 * x + 10 * (figures - x) = 4 * grid_size + 2 * cuts) := by
  sorry

#check two_by_two_squares_count

end two_by_two_squares_count_l871_87137


namespace fraction_division_simplify_fraction_division_l871_87118

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by sorry

end fraction_division_simplify_fraction_division_l871_87118


namespace complex_equation_solution_l871_87176

theorem complex_equation_solution :
  ∃ (z : ℂ), 5 + 2 * I * z = 3 - 5 * I * z ∧ z = (2 * I) / 7 := by
  sorry

end complex_equation_solution_l871_87176


namespace length_of_QR_l871_87173

-- Define the right triangle PQR
def right_triangle_PQR (QR : ℝ) : Prop :=
  ∃ (P Q R : ℝ × ℝ),
    P = (0, 0) ∧  -- P is at the origin
    Q.1 = 12 ∧ Q.2 = 0 ∧  -- Q is on the horizontal axis, 12 units from P
    R.2 ≠ 0 ∧  -- R is not on the horizontal axis (to ensure a right triangle)
    (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = QR^2 ∧  -- Pythagorean theorem
    (R.1 - P.1)^2 + (R.2 - P.2)^2 = QR^2  -- Pythagorean theorem

-- State the theorem
theorem length_of_QR :
  ∀ QR : ℝ, right_triangle_PQR QR → Real.cos (Real.arccos 0.3) = 12 / QR → QR = 40 :=
by
  sorry


end length_of_QR_l871_87173


namespace person_age_puzzle_l871_87191

theorem person_age_puzzle (x : ℝ) : 4 * (x + 3) - 4 * (x - 3) = x ↔ x = 24 := by
  sorry

end person_age_puzzle_l871_87191


namespace well_depth_l871_87102

/-- Proves that a circular well with diameter 4 meters and volume 301.59289474462014 cubic meters has a depth of 24 meters. -/
theorem well_depth (diameter : Real) (volume : Real) (depth : Real) :
  diameter = 4 →
  volume = 301.59289474462014 →
  depth = volume / (π * (diameter / 2)^2) →
  depth = 24 := by
  sorry

end well_depth_l871_87102


namespace digit_zero_equality_l871_87104

-- Define a function to count digits in a number
def countDigits (n : ℕ) : ℕ := sorry

-- Define a function to count zeros in a number
def countZeros (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of digits in a sequence
def sumDigits (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of zeros in a sequence
def sumZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_zero_equality : sumDigits (10^8) = sumZeros (10^9) := by sorry

end digit_zero_equality_l871_87104


namespace min_sum_of_product_l871_87119

theorem min_sum_of_product (a b : ℤ) (h : a * b = 72) : 
  ∀ (x y : ℤ), x * y = 72 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end min_sum_of_product_l871_87119


namespace marcus_bird_count_l871_87175

theorem marcus_bird_count (humphrey_count darrel_count average_count : ℕ) 
  (h1 : humphrey_count = 11)
  (h2 : darrel_count = 9)
  (h3 : average_count = 9)
  (h4 : (humphrey_count + darrel_count + marcus_count) / 3 = average_count) :
  marcus_count = 7 :=
by
  sorry

end marcus_bird_count_l871_87175


namespace relationship_abc_l871_87136

-- Define the constants
noncomputable def a : ℝ := Real.pi ^ (1/3)
noncomputable def b : ℝ := (Real.log 3) / (Real.log Real.pi)
noncomputable def c : ℝ := Real.log (Real.sqrt 3 - 1)

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end relationship_abc_l871_87136


namespace determine_friendship_graph_l871_87187

/-- Represents the friendship graph among apprentices -/
def FriendshipGraph := Fin 10 → Fin 10 → Prop

/-- Represents a duty assignment for a single day -/
def DutyAssignment := Fin 10 → Bool

/-- Calculates the number of missing pastries for a given duty assignment and friendship graph -/
def missingPastries (duty : DutyAssignment) (friends : FriendshipGraph) : ℕ :=
  sorry

/-- Theorem: The chef can determine the friendship graph after 45 days -/
theorem determine_friendship_graph 
  (friends : FriendshipGraph) :
  ∃ (assignments : Fin 45 → DutyAssignment),
    ∀ (other_friends : FriendshipGraph),
      (∀ (day : Fin 45), missingPastries (assignments day) friends = 
                          missingPastries (assignments day) other_friends) →
      friends = other_friends :=
sorry

end determine_friendship_graph_l871_87187


namespace cars_produced_in_europe_l871_87147

theorem cars_produced_in_europe (total_cars : ℕ) (north_america_cars : ℕ) (europe_cars : ℕ) :
  total_cars = 6755 →
  north_america_cars = 3884 →
  total_cars = north_america_cars + europe_cars →
  europe_cars = 2871 :=
by sorry

end cars_produced_in_europe_l871_87147


namespace mans_speed_against_current_l871_87146

-- Define the given speeds
def speed_with_current : ℝ := 15
def current_speed : ℝ := 2.8

-- Define the speed against the current
def speed_against_current : ℝ := speed_with_current - 2 * current_speed

-- Theorem statement
theorem mans_speed_against_current :
  speed_against_current = 9.4 :=
by sorry

end mans_speed_against_current_l871_87146


namespace linear_function_identification_l871_87195

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_identification :
  let f₁ : ℝ → ℝ := λ x ↦ x^3
  let f₂ : ℝ → ℝ := λ x ↦ -2*x + 1
  let f₃ : ℝ → ℝ := λ x ↦ 2/x
  let f₄ : ℝ → ℝ := λ x ↦ 2*x^2 + 1
  is_linear f₂ ∧ ¬is_linear f₁ ∧ ¬is_linear f₃ ∧ ¬is_linear f₄ :=
by sorry

end linear_function_identification_l871_87195


namespace child_ticket_cost_l871_87184

/-- Given information about ticket sales for a baseball game, prove the cost of a child ticket. -/
theorem child_ticket_cost
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_ticket_price = 5)
  (h2 : total_tickets = 85)
  (h3 : total_revenue = 275)
  (h4 : adult_tickets = 35) :
  (total_revenue - adult_tickets * adult_ticket_price) / (total_tickets - adult_tickets) = 2 :=
by sorry

end child_ticket_cost_l871_87184


namespace sector_angle_l871_87157

/-- Given an arc length of 4 cm and a radius of 2 cm, the central angle of the sector in radians is 2. -/
theorem sector_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 4) (h2 : radius = 2) :
  arc_length / radius = 2 := by
  sorry

end sector_angle_l871_87157


namespace sock_ratio_is_two_elevenths_l871_87115

/-- Represents the sock order problem -/
structure SockOrder where
  blackPairs : ℕ
  bluePairs : ℕ
  blackPrice : ℝ
  bluePrice : ℝ

/-- The original sock order -/
def originalOrder : SockOrder :=
  { blackPairs := 6,
    bluePairs := 0,  -- This will be determined
    blackPrice := 0, -- This will be determined
    bluePrice := 0   -- This will be determined
  }

/-- The interchanged sock order -/
def interchangedOrder (o : SockOrder) : SockOrder :=
  { blackPairs := o.bluePairs,
    bluePairs := o.blackPairs,
    blackPrice := o.blackPrice,
    bluePrice := o.bluePrice
  }

/-- Calculate the total cost of a sock order -/
def totalCost (o : SockOrder) : ℝ :=
  o.blackPairs * o.blackPrice + o.bluePairs * o.bluePrice

/-- The theorem stating the ratio of black to blue socks -/
theorem sock_ratio_is_two_elevenths :
  ∃ (o : SockOrder),
    o.blackPairs = 6 ∧
    o.blackPrice = 2 * o.bluePrice ∧
    totalCost (interchangedOrder o) = 1.6 * totalCost o ∧
    o.blackPairs / o.bluePairs = 2 / 11 :=
  sorry

end sock_ratio_is_two_elevenths_l871_87115


namespace rower_downstream_speed_l871_87189

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 48 kmph -/
theorem rower_downstream_speed :
  downstream_speed 32 40 = 48 := by
  sorry

end rower_downstream_speed_l871_87189


namespace nancy_folders_l871_87108

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → deleted_files = 31 → files_per_folder = 7 → 
  (initial_files - deleted_files) / files_per_folder = 7 := by
  sorry

end nancy_folders_l871_87108


namespace chips_sold_in_month_l871_87105

theorem chips_sold_in_month (week1 : ℕ) (week2 : ℕ) (week3 : ℕ) (week4 : ℕ) 
  (h1 : week1 = 15)
  (h2 : week2 = 3 * week1)
  (h3 : week3 = 20)
  (h4 : week4 = 20) :
  week1 + week2 + week3 + week4 = 100 := by
  sorry

end chips_sold_in_month_l871_87105


namespace expected_games_is_correct_l871_87111

/-- Represents the state of the game --/
inductive GameState
| Ongoing : ℕ → ℕ → GameState  -- Number of wins for player A and B
| Finished : GameState

/-- The probability of player A winning in an odd-numbered game --/
def prob_A_odd : ℚ := 3/5

/-- The probability of player B winning in an even-numbered game --/
def prob_B_even : ℚ := 3/5

/-- Determines if the game is finished based on the number of wins --/
def is_finished (wins_A wins_B : ℕ) : Bool :=
  (wins_A ≥ wins_B + 2) ∨ (wins_B ≥ wins_A + 2)

/-- Calculates the expected number of games until the match ends --/
noncomputable def expected_games : ℚ :=
  25/6

/-- Theorem stating that the expected number of games is 25/6 --/
theorem expected_games_is_correct : expected_games = 25/6 := by
  sorry

end expected_games_is_correct_l871_87111


namespace surface_area_unchanged_l871_87141

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  side : ℝ

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged 
  (original : CubeDimensions) 
  (corner : CornerCubeDimensions) 
  (h1 : original.length = original.width ∧ original.width = original.height)
  (h2 : original.length = 5)
  (h3 : corner.side = 2) : 
  surfaceArea original = surfaceArea original := by sorry

end surface_area_unchanged_l871_87141


namespace square_count_3x3_and_5x5_l871_87138

/-- Represents a square grid with uniform distance between consecutive dots -/
structure UniformSquareGrid (n : ℕ) :=
  (size : ℕ)
  (uniform_distance : Bool)

/-- Counts the number of squares with all 4 vertices on the dots in a grid -/
def count_squares (grid : UniformSquareGrid n) : ℕ :=
  sorry

theorem square_count_3x3_and_5x5 :
  ∀ (grid3 : UniformSquareGrid 3) (grid5 : UniformSquareGrid 5),
    grid3.size = 3 ∧ grid3.uniform_distance = true →
    grid5.size = 5 ∧ grid5.uniform_distance = true →
    count_squares grid3 = 4 ∧ count_squares grid5 = 50 :=
by sorry

end square_count_3x3_and_5x5_l871_87138


namespace least_common_multiple_first_ten_l871_87167

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ (n : ℕ), n > 0 ∧ 
  (∀ i ∈ first_ten_integers, i.succ ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i.succ ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end least_common_multiple_first_ten_l871_87167


namespace intersection_point_of_f_and_inverse_l871_87145

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 9*x^2 + 24*x + 36

-- State the theorem
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

end intersection_point_of_f_and_inverse_l871_87145


namespace sum_of_digits_8_pow_2010_l871_87193

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2010 is 1. -/
theorem sum_of_digits_8_pow_2010 : ∃ n : ℕ, 8^2010 = 100 * n + 1 := by sorry

end sum_of_digits_8_pow_2010_l871_87193


namespace sphere_hemisphere_volume_ratio_l871_87124

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 2 / 27 := by
  sorry

end sphere_hemisphere_volume_ratio_l871_87124


namespace triangle_side_range_l871_87143

theorem triangle_side_range (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 4 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (∃ (A' : ℝ), A' ≠ A ∧ 0 < A' ∧ A' < π ∧ a / Real.sin A' = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end triangle_side_range_l871_87143


namespace range_of_m_l871_87131

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≤ 0 ∨ m ≥ 4) := by
sorry

end range_of_m_l871_87131


namespace solution_set_equivalence_l871_87151

theorem solution_set_equivalence (x : ℝ) :
  (1 - |x|) * (1 + x) > 0 ↔ x < 1 ∧ x ≠ -1 := by sorry

end solution_set_equivalence_l871_87151


namespace least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l871_87153

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := ∃ k, n = m * k

theorem least_multiple_with_digit_product_multiple : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n ≥ 315 := by sorry

theorem three_one_five_satisfies :
  isMultipleOf 315 15 ∧ isMultipleOf (digitProduct 315) 15 := by sorry

theorem least_multiple_with_digit_product_multiple_is_315 : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n = 315 ∨ n > 315 := by sorry

end least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l871_87153


namespace inequalities_from_sum_of_reciprocal_squares_l871_87190

theorem inequalities_from_sum_of_reciprocal_squares
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : 1 / a^2 + 1 / b^2 + 1 / c^2 = 1) :
  (1 / a + 1 / b + 1 / c ≤ Real.sqrt 3) ∧
  (a^2 / b^4 + b^2 / c^4 + c^2 / a^4 ≥ 1) := by
  sorry

end inequalities_from_sum_of_reciprocal_squares_l871_87190


namespace equation_has_four_solutions_l871_87107

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  9 * x^2 - 27 * (floor x) + 22 = 0

/-- The theorem stating that the equation has exactly 4 real solutions -/
theorem equation_has_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, equation x ∧
  ∀ y : ℝ, equation y → y ∈ s :=
sorry

end equation_has_four_solutions_l871_87107


namespace orthogonal_projection_locus_l871_87160

/-- Given a line (x/a) + (y/b) = 1 where (1/a^2) + (1/b^2) = 1/c^2 (c constant),
    the orthogonal projection of the origin on this line always lies on the circle x^2 + y^2 = c^2 -/
theorem orthogonal_projection_locus (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c > 0) :
  (1 / a^2 + 1 / b^2 = 1 / c^2) →
  ∃ (x y : ℝ), (x / a + y / b = 1) ∧ 
               (y = (a / b) * x) ∧
               (x^2 + y^2 = c^2) :=
by sorry

end orthogonal_projection_locus_l871_87160


namespace area_at_stage_8_l871_87174

/-- The area of a rectangle formed by adding squares -/
def rectangleArea (numSquares : ℕ) (squareSide : ℕ) : ℕ :=
  numSquares * (squareSide * squareSide)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end area_at_stage_8_l871_87174


namespace barbara_typing_time_l871_87130

/-- Calculates the time needed to type a document given the original typing speed,
    speed decrease, and document length. -/
def typing_time (original_speed : ℕ) (speed_decrease : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_decrease)

/-- Proves that given the specific conditions, the typing time is 20 minutes. -/
theorem barbara_typing_time :
  typing_time 212 40 3440 = 20 := by
  sorry

end barbara_typing_time_l871_87130


namespace median_to_hypotenuse_length_l871_87114

theorem median_to_hypotenuse_length (a b c m : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → m = c / 2 → m = 2.5 := by
sorry

end median_to_hypotenuse_length_l871_87114


namespace rectangular_parallelepiped_volume_l871_87162

/-- The volume of a rectangular parallelepiped with given conditions -/
theorem rectangular_parallelepiped_volume :
  ∀ (length width height : ℝ),
  length > 0 →
  width > 0 →
  height > 0 →
  length = width →
  2 * (length + width) = 32 →
  height = 9 →
  length * width * height = 576 := by
  sorry

end rectangular_parallelepiped_volume_l871_87162


namespace jules_starting_fee_is_two_l871_87142

/-- Calculates the starting fee per walk for Jules' dog walking service -/
def starting_fee_per_walk (total_vacation_cost : ℚ) (family_members : ℕ) 
  (price_per_block : ℚ) (dogs_walked : ℕ) (total_blocks : ℕ) : ℚ :=
  let individual_contribution := total_vacation_cost / family_members
  let earnings_from_blocks := price_per_block * total_blocks
  let total_starting_fees := individual_contribution - earnings_from_blocks
  total_starting_fees / dogs_walked

/-- Proves that Jules' starting fee per walk is $2 given the problem conditions -/
theorem jules_starting_fee_is_two :
  starting_fee_per_walk 1000 5 (5/4) 20 128 = 2 := by
  sorry

end jules_starting_fee_is_two_l871_87142


namespace max_sum_abs_coords_ellipse_l871_87177

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

theorem max_sum_abs_coords_ellipse :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x y : ℝ, ellipse x y → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, ellipse x y ∧ |x| + |y| = M) :=
sorry

end max_sum_abs_coords_ellipse_l871_87177


namespace simplify_expression_1_simplify_expression_2_l871_87180

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (2*x - y)^2 - 4*(x - y)*(x + 2*y) = -8*x*y + 9*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b c : ℝ) :
  (a - 2*b - 3*c)*(a - 2*b + 3*c) = a^2 + 4*b^2 - 4*a*b - 9*c^2 := by sorry

end simplify_expression_1_simplify_expression_2_l871_87180


namespace recipe_batches_for_competition_l871_87149

/-- Calculates the number of full recipe batches needed for a math competition --/
def recipe_batches_needed (total_students : ℕ) (attendance_drop : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let attending_students := (total_students : ℚ) * (1 - attendance_drop)
  let total_cookies_needed := (attending_students * cookies_per_student : ℚ).ceil
  let batches_needed := (total_cookies_needed / cookies_per_batch : ℚ).ceil
  batches_needed.toNat

/-- Proves that 17 full recipe batches are needed for the math competition --/
theorem recipe_batches_for_competition : 
  recipe_batches_needed 144 (30/100) 3 18 = 17 := by
  sorry

end recipe_batches_for_competition_l871_87149


namespace error_clock_correct_time_fraction_l871_87154

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The clock displays '5' instead of '2' -/
  display_error : ℕ → ℕ
  display_error_def : ∀ n, display_error n = if n = 2 then 5 else n

/-- The fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  5/8

theorem error_clock_correct_time_fraction (clock : ErrorClock) :
  correct_time_fraction clock = 5/8 := by
  sorry

end error_clock_correct_time_fraction_l871_87154


namespace det_A_l871_87126

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, -6, 6; 0, 6, -2; 3, -1, 2]

theorem det_A : Matrix.det A = -52 := by
  sorry

end det_A_l871_87126


namespace coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l871_87150

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  (Nat.choose 8 5) = 56 :=
by sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l871_87150


namespace boys_to_total_ratio_l871_87125

theorem boys_to_total_ratio (boys girls : ℕ) (h1 : boys > 0) (h2 : girls > 0) : 
  let total := boys + girls
  let prob_boy := boys / total
  let prob_girl := girls / total
  prob_boy = (1 / 4 : ℚ) * prob_girl →
  (boys : ℚ) / total = 1 / 5 := by
sorry

end boys_to_total_ratio_l871_87125


namespace line_parameterization_l871_87127

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 4 * x - 9

-- Define the parameterization
def parameterization (x y s p t : ℝ) : Prop :=
  x = s + 5 * t ∧ y = 3 + p * t

-- Theorem statement
theorem line_parameterization (s p : ℝ) :
  (∀ x y t : ℝ, line_equation x y ∧ parameterization x y s p t) →
  s = 3 ∧ p = 20 := by sorry

end line_parameterization_l871_87127


namespace sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l871_87112

-- Problem 1
theorem sqrt_expression_equals_zero :
  Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- Problem 2
theorem sqrt_product_division_equals_three_sqrt_two_over_two :
  Real.sqrt 12 * (Real.sqrt 3 / 2) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 := by sorry

end sqrt_expression_equals_zero_sqrt_product_division_equals_three_sqrt_two_over_two_l871_87112


namespace function_inequality_implies_parameter_bound_l871_87140

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end function_inequality_implies_parameter_bound_l871_87140


namespace polygon_angle_sum_l871_87144

theorem polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 + (180 - ((n - 2) * 180) / n) = 1350 → n = 9 := by
  sorry

end polygon_angle_sum_l871_87144


namespace clothes_cost_calculation_l871_87198

def savings_june : ℕ := 21
def savings_july : ℕ := 46
def savings_august : ℕ := 45
def school_supplies_cost : ℕ := 12
def remaining_balance : ℕ := 46

def total_savings : ℕ := savings_june + savings_july + savings_august

def clothes_cost : ℕ := total_savings - school_supplies_cost - remaining_balance

theorem clothes_cost_calculation :
  clothes_cost = 54 :=
by sorry

end clothes_cost_calculation_l871_87198


namespace repaved_total_correct_l871_87197

/-- The total inches of road repaved by a construction company -/
def total_repaved (before_today : ℕ) (today : ℕ) : ℕ :=
  before_today + today

/-- Theorem stating that the total inches repaved is 4938 -/
theorem repaved_total_correct : total_repaved 4133 805 = 4938 := by
  sorry

end repaved_total_correct_l871_87197


namespace hyperbola_sum_l871_87183

/-- Given a hyperbola with center (-3, 1), one focus at (2, 1), and one vertex at (-1, 1),
    prove that h + k + a + b = 0 + √21, where (h, k) is the center, a is the distance from
    the center to the vertex, and b^2 = c^2 - a^2 with c being the distance from the center
    to the focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  (2 : ℝ) - h = c →
  (-1 : ℝ) - h = a →
  b^2 = c^2 - a^2 →
  h + k + a + b = 0 + Real.sqrt 21 := by
  sorry


end hyperbola_sum_l871_87183


namespace perpendicular_vectors_m_value_l871_87134

theorem perpendicular_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, m - 4]
  (∀ i, i < 2 → a i * b i = 0) → m = 4 := by
  sorry

end perpendicular_vectors_m_value_l871_87134


namespace point_q_coordinates_l871_87116

/-- Given two points P and Q in a 2D Cartesian coordinate system, prove that Q has coordinates (1, -3) -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ) -- P and Q are points in 2D space
  (h_P : P = (1, 2)) -- P has coordinates (1, 2)
  (h_Q_below : Q.2 < 0) -- Q is below the x-axis
  (h_parallel : P.1 = Q.1) -- PQ is parallel to the y-axis
  (h_distance : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5) -- PQ = 5
  : Q = (1, -3) := by
  sorry

end point_q_coordinates_l871_87116


namespace smallest_number_l871_87172

theorem smallest_number (S : Set ℕ) (h : S = {5, 8, 3, 2, 6}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 2 := by
  sorry

end smallest_number_l871_87172


namespace opposite_numbers_sum_zero_l871_87188

/-- Two real numbers are opposite if their sum is zero. -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- If a and b are opposite numbers, then their sum is zero. -/
theorem opposite_numbers_sum_zero (a b : ℝ) (h : are_opposite a b) : a + b = 0 := by
  sorry

end opposite_numbers_sum_zero_l871_87188


namespace equation_substitution_l871_87169

theorem equation_substitution :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 2
  let eq2 : ℝ → ℝ := λ y => 2 * y - 1
  ∀ y : ℝ, eq1 (eq2 y) y = 3 * (2 * y - 1) - 4 * y - 2 := by
  sorry

end equation_substitution_l871_87169


namespace earth_inhabitable_fraction_l871_87178

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  inhabitable_land_fraction * land_fraction * earth_surface = (2 : ℚ) / 9 :=
by sorry

end earth_inhabitable_fraction_l871_87178


namespace number_equality_l871_87170

theorem number_equality (x : ℝ) (h : 0.15 * x = 0.25 * 16 + 2) : x = 40 := by
  sorry

end number_equality_l871_87170


namespace power_simplification_l871_87139

theorem power_simplification :
  (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3) = 10^1.7 := by
  sorry

end power_simplification_l871_87139


namespace remainder_theorem_l871_87148

def polynomial (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 6*x^4 - x^3 + x^2 - 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * q x + polynomial 3 := by sorry

end remainder_theorem_l871_87148


namespace complement_A_intersect_B_l871_87135

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | 0 < x ∧ x ≤ 2} :=
by sorry

end complement_A_intersect_B_l871_87135


namespace inscribed_triangle_area_ratio_l871_87110

theorem inscribed_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_base := s
  let triangle_height := s / 2
  let triangle_area := (triangle_base * triangle_height) / 2
  triangle_area / square_area = 1 / 4 := by
sorry

end inscribed_triangle_area_ratio_l871_87110


namespace apartment_cost_ratio_l871_87166

/-- Proves that the ratio of room costs on the third floor to the first floor is 4/3 --/
theorem apartment_cost_ratio :
  ∀ (cost_floor1 cost_floor2 rooms_per_floor total_earnings : ℕ),
    cost_floor1 = 15 →
    cost_floor2 = 20 →
    rooms_per_floor = 3 →
    total_earnings = 165 →
    (total_earnings - (cost_floor1 * rooms_per_floor + cost_floor2 * rooms_per_floor)) / rooms_per_floor / cost_floor1 = 4/3 := by
  sorry

end apartment_cost_ratio_l871_87166


namespace max_days_same_shift_l871_87101

/-- The number of nurses in the ward -/
def num_nurses : ℕ := 15

/-- The number of shifts per day -/
def shifts_per_day : ℕ := 3

/-- Calculates the number of possible nurse pair combinations -/
def nurse_pair_combinations (n : ℕ) : ℕ := n.choose 2

/-- Theorem: Maximum days for two specific nurses to work the same shift again -/
theorem max_days_same_shift : 
  nurse_pair_combinations num_nurses / shifts_per_day = 35 := by
  sorry

end max_days_same_shift_l871_87101


namespace f_inequality_part1_f_inequality_part2_l871_87186

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

theorem f_inequality_part1 :
  ∀ x : ℝ, f 1 x > 1 ↔ x > 1/2 := by sorry

theorem f_inequality_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end f_inequality_part1_f_inequality_part2_l871_87186


namespace total_students_l871_87123

-- Define the score groups
inductive ScoreGroup
| Low : ScoreGroup    -- [20, 40)
| Medium : ScoreGroup -- [40, 60)
| High : ScoreGroup   -- [60, 80)
| VeryHigh : ScoreGroup -- [80, 100]

-- Define the frequency distribution
def FrequencyDistribution := ScoreGroup → ℕ

-- Theorem statement
theorem total_students (freq : FrequencyDistribution) 
  (below_60 : freq ScoreGroup.Low + freq ScoreGroup.Medium = 15) :
  freq ScoreGroup.Low + freq ScoreGroup.Medium + 
  freq ScoreGroup.High + freq ScoreGroup.VeryHigh = 50 :=
by
  sorry

end total_students_l871_87123


namespace simplify_sqrt_m3n2_l871_87128

theorem simplify_sqrt_m3n2 (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  Real.sqrt (m^3 * n^2) = -m * n * Real.sqrt m := by sorry

end simplify_sqrt_m3n2_l871_87128


namespace total_time_conversion_l871_87182

/-- Given 3450 minutes and 7523 seconds, prove that the total time is 59 hours, 35 minutes, and 23 seconds. -/
theorem total_time_conversion (minutes : ℕ) (seconds : ℕ) : 
  minutes = 3450 ∧ seconds = 7523 → 
  ∃ (hours : ℕ) (remaining_minutes : ℕ) (remaining_seconds : ℕ),
    hours = 59 ∧ 
    remaining_minutes = 35 ∧ 
    remaining_seconds = 23 ∧
    minutes * 60 + seconds = hours * 3600 + remaining_minutes * 60 + remaining_seconds :=
by sorry

end total_time_conversion_l871_87182


namespace sequence_perfect_squares_l871_87161

theorem sequence_perfect_squares (a b : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∃ A : ℕ → ℤ, ∀ n : ℕ, a n = (A n)^2 := by
sorry

end sequence_perfect_squares_l871_87161


namespace some_multiplier_value_l871_87117

theorem some_multiplier_value : ∃ (some_multiplier : ℤ), 
  |5 - some_multiplier * (3 - 12)| - |5 - 11| = 71 ∧ some_multiplier = 8 := by
  sorry

end some_multiplier_value_l871_87117
