import Mathlib

namespace trip_length_proof_l2217_221760

/-- Represents the total length of the trip in miles -/
def total_distance : ℝ := 95

/-- Represents the distance traveled on battery -/
def battery_distance : ℝ := 30

/-- Represents the distance traveled on first gasoline mode -/
def first_gas_distance : ℝ := 70

/-- Represents the rate of gasoline consumption in the first gasoline mode -/
def first_gas_rate : ℝ := 0.03

/-- Represents the rate of gasoline consumption in the second gasoline mode -/
def second_gas_rate : ℝ := 0.04

/-- Represents the overall average miles per gallon -/
def average_mpg : ℝ := 50

theorem trip_length_proof :
  total_distance = battery_distance + first_gas_distance +
    (total_distance - battery_distance - first_gas_distance) ∧
  (first_gas_rate * first_gas_distance +
   second_gas_rate * (total_distance - battery_distance - first_gas_distance)) *
    average_mpg = total_distance :=
by sorry

end trip_length_proof_l2217_221760


namespace min_sum_abc_l2217_221717

theorem min_sum_abc (a b c : ℕ+) (h : a.val * b.val * c.val + b.val * c.val + c.val = 2014) :
  ∃ (a' b' c' : ℕ+), 
    a'.val * b'.val * c'.val + b'.val * c'.val + c'.val = 2014 ∧
    a'.val + b'.val + c'.val = 40 ∧
    ∀ (x y z : ℕ+), x.val * y.val * z.val + y.val * z.val + z.val = 2014 → 
      x.val + y.val + z.val ≥ 40 :=
by sorry

end min_sum_abc_l2217_221717


namespace outfit_count_l2217_221749

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 3

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The number of tie options (including no tie). -/
def tie_options : ℕ := num_ties + 1

/-- The number of belt options (including no belt). -/
def belt_options : ℕ := num_belts + 1

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * belt_options

theorem outfit_count : total_outfits = 432 := by
  sorry

end outfit_count_l2217_221749


namespace lg_expression_equals_one_l2217_221735

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 ^ 2 * lg 250 + lg 5 ^ 2 * lg 40 = 1 := by sorry

end lg_expression_equals_one_l2217_221735


namespace carol_cupcakes_l2217_221784

/-- The number of cupcakes Carol has after initially making some, selling some, and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (made_more : ℕ) : ℕ :=
  initial - sold + made_more

/-- Theorem stating that Carol has 49 cupcakes in total -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end carol_cupcakes_l2217_221784


namespace fraction_of_shaded_hexagons_l2217_221707

/-- Given a set of hexagons, some of which are shaded, prove that the fraction of shaded hexagons is correct. -/
theorem fraction_of_shaded_hexagons 
  (total : ℕ) 
  (shaded : ℕ) 
  (h1 : total = 9) 
  (h2 : shaded = 5) : 
  (shaded : ℚ) / total = 5 / 9 := by
  sorry

end fraction_of_shaded_hexagons_l2217_221707


namespace expression_simplification_l2217_221736

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3 * a / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = a / (a - 2) := by
  sorry

end expression_simplification_l2217_221736


namespace m_range_l2217_221785

/-- The function f(x) defined as x^2 + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- The theorem stating the range of m given the conditions --/
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
sorry

end m_range_l2217_221785


namespace largest_solution_of_equation_l2217_221770

theorem largest_solution_of_equation :
  let f (x : ℝ) := x / 7 + 3 / (7 * x)
  ∃ (max_x : ℝ), f max_x = 1 ∧ max_x = (7 + Real.sqrt 37) / 2 ∧
    ∀ (y : ℝ), f y = 1 → y ≤ max_x :=
by sorry

end largest_solution_of_equation_l2217_221770


namespace ratio_of_special_means_l2217_221778

theorem ratio_of_special_means (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h5 : a + b = 36) :
  a / b = 4 := by
  sorry

end ratio_of_special_means_l2217_221778


namespace johns_running_time_l2217_221740

theorem johns_running_time (H : ℝ) : 
  H > 0 →
  (12 : ℝ) * (1.75 * H) = 168 →
  H = 8 :=
by
  sorry

end johns_running_time_l2217_221740


namespace smallest_number_with_remainders_l2217_221768

theorem smallest_number_with_remainders : ∃ (x : ℕ), 
  (x % 3 = 2) ∧ (x % 5 = 3) ∧ (x % 7 = 4) ∧
  (∀ y : ℕ, y < x → ¬((y % 3 = 2) ∧ (y % 5 = 3) ∧ (y % 7 = 4))) ∧
  x = 23 := by
  sorry

end smallest_number_with_remainders_l2217_221768


namespace colin_speed_l2217_221791

/-- Proves that Colin's speed is 4 mph given the relationships between speeds of Bruce, Tony, Brandon, and Colin -/
theorem colin_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = tony_speed / 3 →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end colin_speed_l2217_221791


namespace solve_equation_l2217_221716

theorem solve_equation (x : ℝ) : (3 * x + 36 = 48) → x = 4 := by
  sorry

end solve_equation_l2217_221716


namespace line_intersects_circle_l2217_221787

/-- Theorem: If a point (x₀, y₀) is outside a circle with radius r centered at the origin,
    then the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

/-- Definition: A point (x₀, y₀) is outside the circle x² + y² = r² -/
def point_outside_circle (x₀ y₀ r : ℝ) : Prop :=
  x₀^2 + y₀^2 > r^2

/-- Definition: The line equation x₀x + y₀y = r² -/
def line_equation (x₀ y₀ r x y : ℝ) : Prop :=
  x₀*x + y₀*y = r^2

/-- Definition: A point (x, y) is on the circle x² + y² = r² -/
def point_on_circle (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

end line_intersects_circle_l2217_221787


namespace remainder_problem_l2217_221751

theorem remainder_problem (p : Nat) (h : Prime p) (h1 : p = 13) :
  (7 * 12^24 + 2^24) % p = 8 := by
  sorry

end remainder_problem_l2217_221751


namespace unique_solution_is_x_minus_one_l2217_221727

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem unique_solution_is_x_minus_one (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∀ x : ℝ, f x = x - 1 := by
  sorry

end unique_solution_is_x_minus_one_l2217_221727


namespace monotone_increasing_range_l2217_221799

/-- The function f(x) = lg(x^2 + ax - a - 1) is monotonically increasing in [2, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → 2 ≤ y → x ≤ y →
    Real.log (x^2 + a*x - a - 1) ≤ Real.log (y^2 + a*y - a - 1)

/-- The theorem stating the range of a for which f(x) is monotonically increasing -/
theorem monotone_increasing_range :
  {a : ℝ | is_monotone_increasing a} = Set.Ioi (-3) :=
sorry

end monotone_increasing_range_l2217_221799


namespace divisibility_by_7_and_11_l2217_221729

theorem divisibility_by_7_and_11 (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 3^(2*n+1) + 2^(n+2) = 7*k) ∧
  (∃ m : ℤ, 3^(2*n+2) + 2^(6*n+1) = 11*m) := by
  sorry

end divisibility_by_7_and_11_l2217_221729


namespace expression_simplification_l2217_221796

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a + 1)) = Real.sqrt 3 := by
  sorry

end expression_simplification_l2217_221796


namespace tan_product_undefined_l2217_221759

theorem tan_product_undefined : 
  ¬∃ (x : ℝ), Real.tan (π / 6) * Real.tan (π / 3) * Real.tan (π / 2) = x :=
by sorry

end tan_product_undefined_l2217_221759


namespace percentage_of_300_l2217_221732

/-- Calculates the percentage of a given amount -/
def percentage (percent : ℚ) (amount : ℚ) : ℚ :=
  (percent / 100) * amount

/-- Proves that 25% of Rs. 300 is equal to Rs. 75 -/
theorem percentage_of_300 : percentage 25 300 = 75 := by
  sorry

end percentage_of_300_l2217_221732


namespace binomial_expansion_probability_l2217_221797

/-- The number of terms in the binomial expansion -/
def num_terms : ℕ := 9

/-- The exponent of the binomial -/
def n : ℕ := num_terms - 1

/-- The number of rational terms in the expansion -/
def num_rational_terms : ℕ := 3

/-- The number of irrational terms in the expansion -/
def num_irrational_terms : ℕ := num_terms - num_rational_terms

/-- The total number of permutations of all terms -/
def total_permutations : ℕ := (Nat.factorial num_terms)

/-- The number of favorable permutations where rational terms are not adjacent -/
def favorable_permutations : ℕ := 
  (Nat.factorial num_irrational_terms) * (Nat.choose (num_irrational_terms + 1) num_rational_terms)

/-- The probability that all rational terms are not adjacent when rearranged -/
def probability : ℚ := (favorable_permutations : ℚ) / (total_permutations : ℚ)

theorem binomial_expansion_probability : probability = 5 / 12 := by sorry

end binomial_expansion_probability_l2217_221797


namespace x_squared_geq_one_necessary_not_sufficient_l2217_221739

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
  sorry

end x_squared_geq_one_necessary_not_sufficient_l2217_221739


namespace time_after_12345_seconds_l2217_221747

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  valid : hours < 24 ∧ minutes < 60 ∧ seconds < 60

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem time_after_12345_seconds : 
  addSeconds ⟨18, 15, 0, sorry⟩ 12345 = ⟨21, 40, 45, sorry⟩ := by
  sorry

end time_after_12345_seconds_l2217_221747


namespace total_shoes_l2217_221728

def shoe_store_problem (brown_shoes : ℕ) (black_shoes : ℕ) : Prop :=
  black_shoes = 2 * brown_shoes ∧
  brown_shoes = 22 ∧
  black_shoes + brown_shoes = 66

theorem total_shoes : ∃ (brown_shoes black_shoes : ℕ), shoe_store_problem brown_shoes black_shoes := by
  sorry

end total_shoes_l2217_221728


namespace brand_a_soap_users_l2217_221746

theorem brand_a_soap_users (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 260 →
  neither = 80 →
  both = 30 →
  (total - neither) = (3 * both) + both + (total - neither - 3 * both - both) →
  (total - neither - 3 * both - both) = 60 :=
by sorry

end brand_a_soap_users_l2217_221746


namespace brody_battery_usage_l2217_221742

/-- Represents the battery life of Brody's calculator -/
def BatteryLife : Type := ℚ

/-- The total battery life of the calculator when fully charged (in hours) -/
def full_battery : ℚ := 60

/-- The duration of Brody's exam (in hours) -/
def exam_duration : ℚ := 2

/-- The remaining battery life after the exam (in hours) -/
def remaining_battery : ℚ := 13

/-- The fraction of battery Brody has used up -/
def battery_used_fraction : ℚ := 3/4

/-- Theorem stating that the fraction of battery Brody has used up is 3/4 -/
theorem brody_battery_usage :
  (full_battery - (remaining_battery + exam_duration)) / full_battery = battery_used_fraction := by
  sorry

end brody_battery_usage_l2217_221742


namespace refrigerator_is_right_prism_other_objects_not_right_prisms_l2217_221711

-- Define the properties of a right prism
structure RightPrism :=
  (has_congruent_polygonal_bases : Bool)
  (has_rectangular_lateral_faces : Bool)

-- Define the properties of a refrigerator
structure Refrigerator :=
  (shape : RightPrism)

-- Theorem stating that a refrigerator can be modeled as a right prism
theorem refrigerator_is_right_prism (r : Refrigerator) : 
  r.shape.has_congruent_polygonal_bases ∧ r.shape.has_rectangular_lateral_faces := by
  sorry

-- Define other objects for comparison
structure Basketball :=
  (is_spherical : Bool)

structure Shuttlecock :=
  (has_conical_shape : Bool)

structure Thermos :=
  (is_cylindrical : Bool)

-- Theorem stating that other objects are not right prisms
theorem other_objects_not_right_prisms : 
  ∀ (b : Basketball) (s : Shuttlecock) (t : Thermos),
  ¬(∃ (rp : RightPrism), rp.has_congruent_polygonal_bases ∧ rp.has_rectangular_lateral_faces) := by
  sorry

end refrigerator_is_right_prism_other_objects_not_right_prisms_l2217_221711


namespace linear_function_b_values_l2217_221715

theorem linear_function_b_values (k b : ℝ) :
  (∀ x, -3 ≤ x ∧ x ≤ 1 → -1 ≤ k * x + b ∧ k * x + b ≤ 8) →
  b = 5/4 ∨ b = 23/4 := by
sorry

end linear_function_b_values_l2217_221715


namespace diane_honey_harvest_l2217_221750

/-- The total amount of honey harvested over three years -/
def total_honey_harvest (year1 : ℕ) (increase_year2 : ℕ) (increase_year3 : ℕ) : ℕ :=
  year1 + (year1 + increase_year2) + (year1 + increase_year2 + increase_year3)

/-- Theorem stating the total honey harvest over three years -/
theorem diane_honey_harvest :
  total_honey_harvest 2479 6085 7890 = 27497 := by
  sorry

end diane_honey_harvest_l2217_221750


namespace students_liking_sports_l2217_221745

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 10) → 
  (C.card = 8) → 
  ((B ∩ C).card = 4) → 
  ((B ∪ C).card = 14) := by
  sorry

end students_liking_sports_l2217_221745


namespace complex_modulus_product_l2217_221703

/-- Given a complex number with modulus √2018, prove that its product with its conjugate is 2018 -/
theorem complex_modulus_product (a b : ℝ) : 
  (Complex.abs (Complex.mk a b))^2 = 2018 → (a + Complex.I * b) * (a - Complex.I * b) = 2018 := by
  sorry

end complex_modulus_product_l2217_221703


namespace greatest_three_digit_multiple_of_17_l2217_221790

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n :=
by sorry

end greatest_three_digit_multiple_of_17_l2217_221790


namespace percentage_of_80_equal_to_12_l2217_221731

theorem percentage_of_80_equal_to_12 (p : ℝ) : 
  (p / 100) * 80 = 12 → p = 15 := by sorry

end percentage_of_80_equal_to_12_l2217_221731


namespace circle_tangent_trajectory_l2217_221763

-- Define the circle M
def CircleM (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 10

-- Define the line on which the center of M lies
def CenterLine (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define points A, B, C, and D
def PointA : ℝ × ℝ := (-5, 0)
def PointB : ℝ × ℝ := (1, 0)
def PointC : ℝ × ℝ := (1, 2)
def PointD : ℝ × ℝ := (-3, 4)

-- Define the tangent line through C
def TangentLineC (x y : ℝ) : Prop := 3*x + y - 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 10

theorem circle_tangent_trajectory :
  -- The center of M is on the given line
  (∃ x y, CircleM x y ∧ CenterLine x y) ∧
  -- M passes through A and B
  (CircleM PointA.1 PointA.2 ∧ CircleM PointB.1 PointB.2) →
  -- 1. Equation of circle M is correct
  (∀ x y, CircleM x y ↔ (x + 2)^2 + (y - 1)^2 = 10) ∧
  -- 2. Equation of tangent line through C is correct
  (∀ x y, TangentLineC x y ↔ 3*x + y - 5 = 0) ∧
  -- 3. Trajectory equation of Q is correct
  (∀ x y, (TrajectoryQ x y ∧ ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4))) ↔
    (∃ x₀ y₀, CircleM x₀ y₀ ∧ 
      x = (-5 + x₀ + 3)/2 ∧ 
      y = (y₀ + 4)/2 ∧
      ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4)))) :=
by sorry

end circle_tangent_trajectory_l2217_221763


namespace expression_equivalence_l2217_221762

theorem expression_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := by
  sorry

end expression_equivalence_l2217_221762


namespace pure_imaginary_complex_number_l2217_221721

theorem pure_imaginary_complex_number (m : ℝ) :
  (((m^2 + 2*m - 3) : ℂ) + (m - 1)*I = (0 : ℂ) + ((m - 1)*I : ℂ)) → m = -3 :=
by sorry

end pure_imaginary_complex_number_l2217_221721


namespace journey_time_ratio_l2217_221767

/-- Proves the ratio of new time to original time for a given journey -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 ∧ original_time = 6 ∧ new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end journey_time_ratio_l2217_221767


namespace faucet_fill_time_l2217_221700

-- Define the constants from the problem
def tub_size_1 : ℝ := 200  -- Size of the first tub in gallons
def tub_size_2 : ℝ := 50   -- Size of the second tub in gallons
def faucets_1 : ℝ := 4     -- Number of faucets for the first tub
def faucets_2 : ℝ := 8     -- Number of faucets for the second tub
def time_1 : ℝ := 12       -- Time to fill the first tub in minutes

-- Define the theorem
theorem faucet_fill_time :
  ∃ (rate : ℝ),
    (rate * faucets_1 * time_1 = tub_size_1) ∧
    (rate * faucets_2 * (90 / 60) = tub_size_2) :=
by sorry

end faucet_fill_time_l2217_221700


namespace tournament_handshakes_l2217_221752

theorem tournament_handshakes (n : ℕ) (m : ℕ) (h : n = 4 ∧ m = 2) :
  let total_players := n * m
  let handshakes_per_player := total_players - m
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  sorry

end tournament_handshakes_l2217_221752


namespace grocery_store_price_l2217_221743

/-- The price of a bulk warehouse deal for sparkling water -/
def bulk_price : ℚ := 12

/-- The number of cans in the bulk warehouse deal -/
def bulk_cans : ℕ := 48

/-- The additional cost per can at the grocery store compared to the bulk warehouse -/
def additional_cost : ℚ := 1/4

/-- The number of cans in the grocery store deal -/
def grocery_cans : ℕ := 12

/-- The price of the grocery store deal for sparkling water -/
def grocery_price : ℚ := 6

theorem grocery_store_price :
  grocery_price = (bulk_price / bulk_cans + additional_cost) * grocery_cans :=
by sorry

end grocery_store_price_l2217_221743


namespace polynomial_product_expansion_l2217_221724

theorem polynomial_product_expansion (x : ℝ) :
  (2 * x^3 - 3 * x^2 + 4) * (3 * x^2 + x + 1) =
  6 * x^5 - 7 * x^4 - x^3 + 9 * x^2 + 4 * x + 4 := by
  sorry

end polynomial_product_expansion_l2217_221724


namespace quadratic_root_relation_l2217_221771

theorem quadratic_root_relation (a b : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 3 * r - 8 = 0) ∧ 
              (2 * s^2 - 3 * s - 8 = 0) ∧ 
              ((r + 3)^2 + a * (r + 3) + b = 0) ∧ 
              ((s + 3)^2 + a * (s + 3) + b = 0)) →
  b = 9.5 := by
sorry

end quadratic_root_relation_l2217_221771


namespace stream_speed_l2217_221776

theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 250 →
  downstream_time = 7 →
  upstream_distance = 150 →
  upstream_time = 21 →
  ∃ s : ℝ, abs (s - 14.28) < 0.01 ∧ 
  (∃ b : ℝ, downstream_distance / downstream_time = b + s ∧
            upstream_distance / upstream_time = b - s) :=
by sorry

end stream_speed_l2217_221776


namespace factorization_problem_1_factorization_problem_2_l2217_221722

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -3 * x^2 + 6 * x * y - 3 * y^2 = -3 * (x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) :
  8 * m^2 * (m + n) - 2 * (m + n) = 2 * (m + n) * (2 * m + 1) * (2 * m - 1) := by sorry

end factorization_problem_1_factorization_problem_2_l2217_221722


namespace min_value_g_l2217_221738

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ := sorry

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Function g(X) as defined in the problem -/
def g (tetra : Tetrahedron) (X : Point3D) : ℝ :=
  distance tetra.A X + distance tetra.B X + distance tetra.C X + distance tetra.D X

/-- Theorem stating the minimum value of g(X) for the given tetrahedron -/
theorem min_value_g (tetra : Tetrahedron) 
  (h1 : distance tetra.A tetra.D = 30)
  (h2 : distance tetra.B tetra.C = 30)
  (h3 : distance tetra.A tetra.C = 46)
  (h4 : distance tetra.B tetra.D = 46)
  (h5 : distance tetra.A tetra.B = 50)
  (h6 : distance tetra.C tetra.D = 50) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 628 ∧ ∀ (X : Point3D), g tetra X ≥ min_val :=
sorry

end min_value_g_l2217_221738


namespace problem_solution_l2217_221795

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (x y a : ℕ) : 
  x > 0 → y > 0 → a > 0 → 
  x * y = 32 → 
  sum_of_digits ((10 ^ x) ^ a - 64) = 279 → 
  a = 1 := by sorry

end problem_solution_l2217_221795


namespace rational_inequality_solution_l2217_221705

theorem rational_inequality_solution (x : ℝ) : 
  (1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔ 
  (x < -3 ∨ (-1 < x ∧ x < 0)) :=
sorry

end rational_inequality_solution_l2217_221705


namespace train_length_train_length_proof_l2217_221794

/-- The length of a train given specific conditions --/
theorem train_length : ℝ :=
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds

  100 -- meters

/-- Proof that the train length is correct given the conditions --/
theorem train_length_proof :
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds
  train_length = 100 := by
  sorry

end train_length_train_length_proof_l2217_221794


namespace diagonal_grid_4x3_triangles_l2217_221723

/-- Represents a rectangular grid with diagonals -/
structure DiagonalGrid :=
  (columns : Nat)
  (rows : Nat)

/-- Calculates the number of triangles in a diagonal grid -/
def count_triangles (grid : DiagonalGrid) : Nat :=
  let small_triangles := 2 * grid.columns * grid.rows
  let larger_rectangles := (grid.columns - 1) * (grid.rows - 1)
  let larger_triangles := 8 * larger_rectangles
  let additional_triangles := 6  -- Simplified count for larger configurations
  small_triangles + larger_triangles + additional_triangles

/-- Theorem stating that a 4x3 diagonal grid contains 78 triangles -/
theorem diagonal_grid_4x3_triangles :
  ∃ (grid : DiagonalGrid), grid.columns = 4 ∧ grid.rows = 3 ∧ count_triangles grid = 78 :=
by
  sorry

#eval count_triangles ⟨4, 3⟩

end diagonal_grid_4x3_triangles_l2217_221723


namespace expression_simplification_l2217_221782

theorem expression_simplification (y : ℝ) : 
  3*y - 5*y^2 + 2 + (8 - 5*y + 2*y^2) = -3*y^2 - 2*y + 10 := by
  sorry

end expression_simplification_l2217_221782


namespace angle_point_cosine_l2217_221709

/-- Given an angle α in the first quadrant and a point P(a, √5) on its terminal side,
    if cos α = (√2/4)a, then a = √3 -/
theorem angle_point_cosine (α : Real) (a : Real) :
  0 < α ∧ α < π / 2 →  -- α is in the first quadrant
  (∃ (P : ℝ × ℝ), P = (a, Real.sqrt 5) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = Real.cos α) →  -- P(a, √5) is on the terminal side
  Real.cos α = (Real.sqrt 2 / 4) * a →  -- given condition
  a = Real.sqrt 3 :=
by
  sorry

end angle_point_cosine_l2217_221709


namespace intersection_M_N_l2217_221793

-- Define set M
def M : Set ℝ := {x | x < 2016}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_M_N_l2217_221793


namespace charity_savings_interest_l2217_221769

/-- Represents the initial savings amount in dollars -/
def P : ℝ := 2181

/-- Represents the first interest rate (8% per annum) -/
def r1 : ℝ := 0.08

/-- Represents the second interest rate (4% per annum) -/
def r2 : ℝ := 0.04

/-- Represents the time period for each interest rate (3 months = 0.25 years) -/
def t : ℝ := 0.25

/-- Represents the final amount after applying both interest rates -/
def A : ℝ := 2247.50

/-- Theorem stating that the initial amount P results in the final amount A 
    after applying the given interest rates for the specified time periods -/
theorem charity_savings_interest : 
  P * (1 + r1 * t) * (1 + r2 * t) = A := by sorry

end charity_savings_interest_l2217_221769


namespace no_solution_for_sqrt_equation_l2217_221779

theorem no_solution_for_sqrt_equation :
  ¬ ∃ x : ℝ, x > 9 ∧ Real.sqrt (x - 9) + 3 = Real.sqrt (x + 9) - 3 := by
  sorry


end no_solution_for_sqrt_equation_l2217_221779


namespace smallest_sum_B_plus_c_l2217_221737

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ), 
  (0 ≤ B ∧ B ≤ 4) ∧ 
  (c > 6) ∧
  (31 * B = 4 * c + 4) ∧
  (∀ (B' c' : ℕ), (0 ≤ B' ∧ B' ≤ 4) ∧ (c' > 6) ∧ (31 * B' = 4 * c' + 4) → B + c ≤ B' + c') ∧
  B + c = 34 := by
  sorry

end smallest_sum_B_plus_c_l2217_221737


namespace subtraction_of_integers_l2217_221733

theorem subtraction_of_integers : 2 - 3 = -1 := by sorry

end subtraction_of_integers_l2217_221733


namespace complement_of_union_l2217_221748

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end complement_of_union_l2217_221748


namespace carrot_sticks_before_dinner_l2217_221798

theorem carrot_sticks_before_dinner 
  (before : ℕ) 
  (after : ℕ) 
  (total : ℕ) 
  (h1 : after = 15) 
  (h2 : total = 37) 
  (h3 : before + after = total) : 
  before = 22 := by
sorry

end carrot_sticks_before_dinner_l2217_221798


namespace total_shells_l2217_221761

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : afternoon_shells = 324) : 
  morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_l2217_221761


namespace sin_2theta_value_l2217_221712

theorem sin_2theta_value (θ : ℝ) 
  (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end sin_2theta_value_l2217_221712


namespace prob_answer_within_four_rings_l2217_221777

/-- The probability of answering a phone call at a specific ring. -/
def prob_answer_at_ring : Fin 4 → ℝ
  | 0 => 0.1  -- First ring
  | 1 => 0.3  -- Second ring
  | 2 => 0.4  -- Third ring
  | 3 => 0.1  -- Fourth ring

/-- Theorem: The probability of answering the phone within the first four rings is 0.9. -/
theorem prob_answer_within_four_rings :
  (Finset.sum Finset.univ prob_answer_at_ring) = 0.9 := by
  sorry

end prob_answer_within_four_rings_l2217_221777


namespace arithmetic_operation_l2217_221726

theorem arithmetic_operation : 5 + 4 - 3 + 2 - 1 = 7 := by
  sorry

end arithmetic_operation_l2217_221726


namespace pond_to_field_area_ratio_l2217_221772

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 28 →
    pond_side = 7 →
    (pond_side^2) / (field_length * field_width) = 1 / 8 := by
  sorry

end pond_to_field_area_ratio_l2217_221772


namespace pizza_toppings_l2217_221719

/-- Given a pizza with 24 slices, where every slice has at least one topping,
    if exactly 15 slices have ham and exactly 17 slices have cheese,
    then the number of slices with both ham and cheese is 8. -/
theorem pizza_toppings (total : Nat) (ham : Nat) (cheese : Nat) (both : Nat) :
  total = 24 →
  ham = 15 →
  cheese = 17 →
  both + (ham - both) + (cheese - both) = total →
  both = 8 := by
sorry

end pizza_toppings_l2217_221719


namespace tax_free_limit_correct_l2217_221704

/-- The tax-free total value limit for imported goods in country X. -/
def tax_free_limit : ℝ := 500

/-- The tax rate applied to the value exceeding the tax-free limit. -/
def tax_rate : ℝ := 0.08

/-- The total value of goods imported by a specific tourist. -/
def total_value : ℝ := 730

/-- The tax paid by the tourist. -/
def tax_paid : ℝ := 18.40

/-- Theorem stating that the tax-free limit is correct given the problem conditions. -/
theorem tax_free_limit_correct : 
  tax_rate * (total_value - tax_free_limit) = tax_paid :=
by sorry

end tax_free_limit_correct_l2217_221704


namespace hyperbola_asymptotes_l2217_221701

/-- Given a hyperbola with equation x²/64 - y²/36 = 1, 
    its asymptotes have equations y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 
  (x^2 / 64 - y^2 / 36 = 1) →
  (∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x)) := by
  sorry

end hyperbola_asymptotes_l2217_221701


namespace gcd_problem_l2217_221766

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1116 * k) :
  Int.gcd (b^2 + 11*b + 36) (b + 6) = 6 := by
  sorry

end gcd_problem_l2217_221766


namespace equal_distribution_of_treats_l2217_221773

def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

theorem equal_distribution_of_treats :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end equal_distribution_of_treats_l2217_221773


namespace hyperbola_min_distance_hyperbola_min_distance_achieved_l2217_221764

theorem hyperbola_min_distance (x y : ℝ) : 
  (x^2 / 8) - (y^2 / 4) = 1 → |x - y| ≥ 2 :=
by sorry

theorem hyperbola_min_distance_achieved : 
  ∃ (x y : ℝ), (x^2 / 8) - (y^2 / 4) = 1 ∧ |x - y| = 2 :=
by sorry

end hyperbola_min_distance_hyperbola_min_distance_achieved_l2217_221764


namespace secretary_work_hours_l2217_221734

theorem secretary_work_hours (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / b = 2 / 3 →
  b / c = 3 / 5 →
  c = 40 →
  a + b + c = 80 :=
by sorry

end secretary_work_hours_l2217_221734


namespace polynomial_property_l2217_221718

def Q (x d e f : ℝ) : ℝ := 3 * x^4 + d * x^3 + e * x^2 + f * x - 27

theorem polynomial_property (d e f : ℝ) :
  (∀ x₁ x₂ x₃ x₄ : ℝ, Q x₁ d e f = 0 ∧ Q x₂ d e f = 0 ∧ Q x₃ d e f = 0 ∧ Q x₄ d e f = 0 →
    x₁ + x₂ + x₃ + x₄ = x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄) ∧
  (x₁ + x₂ + x₃ + x₄ = 3 + d + e + f - 27) ∧
  e = 0 →
  f = -12 := by sorry

end polynomial_property_l2217_221718


namespace total_age_problem_l2217_221714

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 12 →
  a + b + c = 32 := by
sorry

end total_age_problem_l2217_221714


namespace term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l2217_221783

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℕ :=
  let start := 1 + n * (n - 1) / 2
  (n * (2 * start + (n - 1))) / 2

/-- Theorem stating that the 2023rd term of the sequence is 2023 -/
theorem term_2023_equals_2023 : sequenceTerm 64 = 2023 := by
  sorry

/-- Theorem stating that the 2023rd term is in the 64th group -/
theorem term_2023_in_group_64 :
  (63 * 64) / 2 < 2023 ∧ 2023 ≤ (64 * 65) / 2 := by
  sorry

/-- Theorem stating that the 2023rd term is the 7th term in its group -/
theorem term_2023_is_7th_in_group :
  2023 - (63 * 64) / 2 = 7 := by
  sorry

end term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l2217_221783


namespace quadratic_distinct_roots_range_l2217_221789

/-- The range of k for which the quadratic equation kx^2 - 4x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 4 * x₁ - 2 = 0 ∧ k * x₂^2 - 4 * x₂ - 2 = 0) ↔ 
  (k > -2 ∧ k ≠ 0) :=
sorry

end quadratic_distinct_roots_range_l2217_221789


namespace complex_equation_solution_l2217_221765

theorem complex_equation_solution (z : ℂ) :
  (2 - Complex.I) * z = Complex.I ^ 2022 →
  z = -2/5 - (1/5) * Complex.I :=
by sorry

end complex_equation_solution_l2217_221765


namespace line_equation_l2217_221786

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through a point
def line_through_point (k m : ℝ) (x y : ℝ) : Prop := y = k*(x - m)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- Main theorem
theorem line_equation (P M N Q : ℝ × ℝ) :
  let (xp, yp) := P
  let (xm, ym) := M
  let (xn, yn) := N
  let (xq, yq) := Q
  (∃ k : ℝ, 
    (∀ x y, line_through_point k 1 x y → (circle_F x y ∨ parabola_C x y)) ∧
    line_through_point k 1 xp yp ∧
    line_through_point k 1 xm ym ∧
    line_through_point k 1 xn yn ∧
    line_through_point k 1 xq yq ∧
    arithmetic_sequence (Real.sqrt ((xp-1)^2 + yp^2)) (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) ∧
    arithmetic_sequence (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) (Real.sqrt ((xq-1)^2 + yq^2))) →
  (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
sorry

end line_equation_l2217_221786


namespace average_sleep_time_l2217_221788

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time :
  (sleep_times.sum / sleep_times.length : ℝ) = 9 := by sorry

end average_sleep_time_l2217_221788


namespace student_b_visited_a_l2217_221755

structure Student :=
  (visitedA : Bool)
  (visitedB : Bool)
  (visitedC : Bool)

def citiesVisited (s : Student) : Nat :=
  (if s.visitedA then 1 else 0) +
  (if s.visitedB then 1 else 0) +
  (if s.visitedC then 1 else 0)

theorem student_b_visited_a (studentA studentB studentC : Student) :
  citiesVisited studentA > citiesVisited studentB →
  studentA.visitedB = false →
  studentB.visitedC = false →
  (studentA.visitedA = true ∧ studentB.visitedA = true ∧ studentC.visitedA = true) ∨
  (studentA.visitedB = true ∧ studentB.visitedB = true ∧ studentC.visitedB = true) ∨
  (studentA.visitedC = true ∧ studentB.visitedC = true ∧ studentC.visitedC = true) →
  studentB.visitedA = true :=
by
  sorry

end student_b_visited_a_l2217_221755


namespace function_transformation_l2217_221756

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = -1) : f (2 - 1) - 1 = -2 := by
  sorry

end function_transformation_l2217_221756


namespace inequality_solution_set_minimum_value_no_positive_solution_l2217_221744

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value :
  ∃ (k : ℝ), k = 1 ∧ ∀ (x : ℝ), f x ≥ k := by sorry

-- Theorem for non-existence of positive a and b
theorem no_positive_solution :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 := by sorry

end inequality_solution_set_minimum_value_no_positive_solution_l2217_221744


namespace henry_birthday_money_l2217_221720

theorem henry_birthday_money (initial_amount spent_amount final_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 10)
  (h3 : final_amount = 19) :
  final_amount + spent_amount - initial_amount = 18 := by
  sorry

end henry_birthday_money_l2217_221720


namespace converse_zero_product_l2217_221710

theorem converse_zero_product (a b : ℝ) : 
  (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) := by sorry

end converse_zero_product_l2217_221710


namespace quinn_free_donuts_l2217_221753

/-- The number of books required to earn one donut coupon -/
def books_per_coupon : ℕ := 5

/-- The number of books Quinn reads per week -/
def books_per_week : ℕ := 2

/-- The number of weeks Quinn reads -/
def weeks_read : ℕ := 10

/-- The total number of books Quinn reads -/
def total_books : ℕ := books_per_week * weeks_read

/-- The number of free donuts Quinn is eligible for -/
def free_donuts : ℕ := total_books / books_per_coupon

theorem quinn_free_donuts : free_donuts = 4 := by
  sorry

end quinn_free_donuts_l2217_221753


namespace inequality_proof_l2217_221775

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + y - 1)^2 / z + (y + z - 1)^2 / x + (z + x - 1)^2 / y ≥ x + y + z := by
  sorry

end inequality_proof_l2217_221775


namespace chess_team_arrangements_l2217_221730

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 4

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the requirement that two specific girls must sit together -/
def specific_girls_together : Prop := True

/-- Represents the requirement that a boy must sit at each end -/
def boy_at_each_end : Prop := True

/-- The number of possible arrangements of the chess team -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements is 72 -/
theorem chess_team_arrangements :
  num_boys = 3 →
  num_girls = 4 →
  specific_girls_together →
  boy_at_each_end →
  num_arrangements = 72 := by sorry

end chess_team_arrangements_l2217_221730


namespace fraction_transformation_l2217_221741

theorem fraction_transformation (a b : ℝ) (h : b ≠ 0) :
  a / b = (a + 2 * a) / (b + 2 * b) := by
  sorry

end fraction_transformation_l2217_221741


namespace fraction_simplification_l2217_221757

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hden : y^2 - 1/x^2 ≠ 0) : 
  (x^2 - 1/y^2) / (y^2 - 1/x^2) = x^2 / y^2 := by
  sorry

end fraction_simplification_l2217_221757


namespace first_term_of_geometric_sequence_l2217_221780

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : geometric_sequence a r 4 = 24) 
  (h2 : geometric_sequence a r 5 = 48) : 
  a = 3 := by
sorry

end first_term_of_geometric_sequence_l2217_221780


namespace largest_geometric_three_digit_number_l2217_221702

/-- Checks if three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three digits are distinct -/
def are_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a three-digit number -/
def three_digit_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem largest_geometric_three_digit_number :
  ∀ n : ℕ,
  (∃ a b c : ℕ, n = three_digit_number a b c ∧
                a ≤ 8 ∧
                a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
                a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
                is_geometric_sequence a b c ∧
                are_distinct a b c) →
  n ≤ 842 :=
sorry

end largest_geometric_three_digit_number_l2217_221702


namespace tetrahedron_volume_with_inscribed_sphere_l2217_221708

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume_with_inscribed_sphere
  (R : ℝ)  -- Radius of the inscribed sphere
  (S₁ S₂ S₃ S₄ : ℝ)  -- Areas of the four faces of the tetrahedron
  (h₁ : R > 0)  -- Radius is positive
  (h₂ : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)  -- Face areas are positive
  : ∃ V : ℝ, V = R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 :=
by sorry

end tetrahedron_volume_with_inscribed_sphere_l2217_221708


namespace line_contains_point_l2217_221758

theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (1/3) = -2 * 4) ↔ (k = 9) := by sorry

end line_contains_point_l2217_221758


namespace balloon_count_l2217_221713

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 := by
  sorry

end balloon_count_l2217_221713


namespace set_equality_l2217_221725

def A : Set ℝ := {x : ℝ | |x| < 3}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem set_equality : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end set_equality_l2217_221725


namespace smallest_a_in_special_progression_l2217_221792

theorem smallest_a_in_special_progression (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  (∀ a' b' c' : ℤ, a' < b' → b' < c' → (2 * b' = a' + c') → (c' * c' = a' * b') → a ≤ a') →
  a = -4 := by
sorry

end smallest_a_in_special_progression_l2217_221792


namespace water_problem_solution_l2217_221754

def water_problem (total_water : ℕ) (car_water : ℕ) (num_cars : ℕ) (plant_water_diff : ℕ) : ℕ :=
  let car_total := car_water * num_cars
  let plant_water := car_total - plant_water_diff
  let used_water := car_total + plant_water
  let remaining_water := total_water - used_water
  remaining_water / 2

theorem water_problem_solution :
  water_problem 65 7 2 11 = 24 := by
  sorry

end water_problem_solution_l2217_221754


namespace train_length_calculation_l2217_221706

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 46.5) 
  (h2 : man_speed = 2.5) 
  (h3 : passing_time = 62.994960403167745) : 
  ∃ (length : ℝ), abs (length - 770) < 0.1 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2217_221706


namespace range_of_f_l2217_221781

def f (x : ℤ) : ℤ := (x - 1)^2 + 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end range_of_f_l2217_221781


namespace unique_solution_condition_l2217_221774

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x) ↔ 
  (k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5) :=
sorry

end unique_solution_condition_l2217_221774
