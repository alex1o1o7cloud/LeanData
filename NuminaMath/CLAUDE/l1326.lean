import Mathlib

namespace NUMINAMATH_CALUDE_euler_children_mean_age_l1326_132676

def euler_children_ages : List ℕ := [7, 7, 7, 12, 12, 14, 15]

theorem euler_children_mean_age :
  (euler_children_ages.sum : ℚ) / euler_children_ages.length = 74 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_children_mean_age_l1326_132676


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1326_132645

theorem quadratic_root_difference : ∃ (x₁ x₂ : ℝ),
  (5 + 3 * Real.sqrt 2) * x₁^2 - (1 - Real.sqrt 2) * x₁ - 1 = 0 ∧
  (5 + 3 * Real.sqrt 2) * x₂^2 - (1 - Real.sqrt 2) * x₂ - 1 = 0 ∧
  x₁ ≠ x₂ ∧
  max x₁ x₂ - min x₁ x₂ = 2 * Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1326_132645


namespace NUMINAMATH_CALUDE_comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l1326_132657

/-- Represents a technology company -/
structure TechCompany where
  name : String
  profit : ℝ
  employeeRetention : ℝ
  productivity : ℝ
  workLifeIntegration : ℝ

/-- Represents an office environment -/
structure OfficeEnvironment where
  hasWorkSpaces : Bool
  hasLeisureSpaces : Bool
  hasLivingSpaces : Bool

/-- Function to determine if an office environment is comprehensive -/
def isComprehensiveOffice (office : OfficeEnvironment) : Bool :=
  office.hasWorkSpaces ∧ office.hasLeisureSpaces ∧ office.hasLivingSpaces

/-- Function to calculate the impact of office environment on company metrics -/
def officeImpact (company : TechCompany) (office : OfficeEnvironment) : TechCompany :=
  if isComprehensiveOffice office then
    { company with
      employeeRetention := company.employeeRetention * 1.1
      productivity := company.productivity * 1.15
      workLifeIntegration := company.workLifeIntegration * 1.2
    }
  else
    company

/-- Theorem stating that comprehensive offices increase profit -/
theorem comprehensive_office_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (officeImpact company office).profit > company.profit :=
by sorry

/-- Main theorem proving that building comprehensive offices increases profit through specific factors -/
theorem building_comprehensive_offices_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (∃ newCompany : TechCompany,
    newCompany = officeImpact company office ∧
    newCompany.profit > company.profit ∧
    newCompany.employeeRetention > company.employeeRetention ∧
    newCompany.productivity > company.productivity ∧
    newCompany.workLifeIntegration > company.workLifeIntegration) :=
by sorry

end NUMINAMATH_CALUDE_comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l1326_132657


namespace NUMINAMATH_CALUDE_fractional_inequality_l1326_132637

theorem fractional_inequality (x : ℝ) : (2*x - 1) / (x + 1) < 0 ↔ -1 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_l1326_132637


namespace NUMINAMATH_CALUDE_odd_function_property_l1326_132681

-- Define an odd function on an interval
def odd_function_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ (a ≤ b)

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a b : ℝ) 
  (h : odd_function_on_interval f a b) : f (2 * (a + b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1326_132681


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l1326_132608

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists z in T such that z^n = 1 -/
def HasNthRoot (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ T ∧ z^n = 1

/-- 13 is the smallest positive integer satisfying the HasNthRoot property -/
theorem smallest_m_is_13 :
  HasNthRoot 13 ∧ ∀ m : ℕ, m > 0 → m < 13 → ¬HasNthRoot m :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l1326_132608


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1326_132636

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 12 = 36) :
  w / 12 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1326_132636


namespace NUMINAMATH_CALUDE_greenhouse_path_area_l1326_132633

/-- Calculates the total area of paths in Joanna's greenhouse --/
theorem greenhouse_path_area :
  let num_rows : ℕ := 5
  let beds_per_row : ℕ := 3
  let bed_width : ℕ := 4
  let bed_height : ℕ := 3
  let path_width : ℕ := 2
  
  let total_width : ℕ := beds_per_row * bed_width + (beds_per_row + 1) * path_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * path_width
  
  let total_area : ℕ := total_width * total_height
  let bed_area : ℕ := num_rows * beds_per_row * bed_width * bed_height
  
  total_area - bed_area = 360 :=
by sorry


end NUMINAMATH_CALUDE_greenhouse_path_area_l1326_132633


namespace NUMINAMATH_CALUDE_train_speed_l1326_132677

/-- The speed of a train given specific conditions -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 10 →
  t_stationary = 30 →
  l_stationary = 600 →
  ∃ v : ℝ, v * t_pole = v * t_stationary - l_stationary ∧ v * 3.6 = 108 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1326_132677


namespace NUMINAMATH_CALUDE_welders_count_l1326_132620

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 72

/-- The number of days needed to complete the order with all welders -/
def initial_days : ℕ := 5

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed to complete the order after some welders leave -/
def additional_days : ℕ := 6

/-- The theorem stating that the initial number of welders is 72 -/
theorem welders_count :
  initial_welders = 72 ∧
  (1 : ℚ) / (initial_days * initial_welders) = 
  (1 : ℚ) / (additional_days * (initial_welders - leaving_welders)) :=
by sorry

end NUMINAMATH_CALUDE_welders_count_l1326_132620


namespace NUMINAMATH_CALUDE_integral_value_l1326_132623

theorem integral_value : ∫ x in (2 : ℝ)..4, (x^3 - 3*x^2 + 5) / x^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l1326_132623


namespace NUMINAMATH_CALUDE_min_dividing_segment_length_l1326_132666

/-- A trapezoid with midsegment length 4 and a line parallel to the bases dividing its area in half -/
structure DividedTrapezoid where
  /-- The length of the midsegment -/
  midsegment_length : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The length of the segment created by the dividing line -/
  dividing_segment : ℝ
  /-- The midsegment length is 4 -/
  midsegment_eq : midsegment_length = 4
  /-- The dividing line splits the area in half -/
  area_split : (lower_base + dividing_segment) * (height / 2) = (upper_base + dividing_segment) * (height / 2)
  /-- The sum of the bases is twice the midsegment length -/
  bases_sum : lower_base + upper_base = 2 * midsegment_length

/-- The minimum possible length of the dividing segment is 4 -/
theorem min_dividing_segment_length (t : DividedTrapezoid) : 
  ∃ (min_length : ℝ), min_length = 4 ∧ ∀ (x : ℝ), t.dividing_segment ≥ min_length :=
by sorry

end NUMINAMATH_CALUDE_min_dividing_segment_length_l1326_132666


namespace NUMINAMATH_CALUDE_adam_lawn_mowing_earnings_l1326_132642

/-- Adam's lawn mowing earnings problem -/
theorem adam_lawn_mowing_earnings 
  (dollars_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8)
  : (total_lawns - forgotten_lawns) * dollars_per_lawn = 36 := by
  sorry

end NUMINAMATH_CALUDE_adam_lawn_mowing_earnings_l1326_132642


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l1326_132615

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The equation of one asymptote -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote has equation y = 2x + 1 -/
  asymptote1_eq : ∀ x, asymptote1 x = 2 * x + 1
  /-- Condition that the foci have x-coordinate 4 -/
  foci_x_eq : foci_x = 4

/-- The theorem stating the equation of the other asymptote -/
theorem other_asymptote_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = -2 * x + 17) ∧ 
  (∀ x y, y = f x ↔ y = h.asymptote1 x ∨ y = -2 * x + 17) :=
sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l1326_132615


namespace NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l1326_132696

/-- A line in a coordinate plane. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The product of a line's slope and y-intercept. -/
def slopeInterceptProduct (l : Line) : ℝ := l.slope * l.yIntercept

/-- Theorem: For a line with y-intercept -2 and slope 3, the product of its slope and y-intercept is -6. -/
theorem slope_intercept_product_specific_line :
  ∃ l : Line, l.yIntercept = -2 ∧ l.slope = 3 ∧ slopeInterceptProduct l = -6 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l1326_132696


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1326_132626

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ (1^2 + 2*1 - m > 0)) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1326_132626


namespace NUMINAMATH_CALUDE_prob_three_different_suits_value_l1326_132694

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def prob_three_different_suits : ℚ :=
  (cards_per_suit * (number_of_suits - 1) : ℚ) / (standard_deck_size - 1) *
  (cards_per_suit * (number_of_suits - 2) : ℚ) / (standard_deck_size - 2)

theorem prob_three_different_suits_value : 
  prob_three_different_suits = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_different_suits_value_l1326_132694


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1326_132614

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1326_132614


namespace NUMINAMATH_CALUDE_root_product_l1326_132640

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end NUMINAMATH_CALUDE_root_product_l1326_132640


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1326_132644

theorem polygon_interior_angles (n : ℕ) (h : n > 2) :
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 2400 + x = (n - 2) * 180) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1326_132644


namespace NUMINAMATH_CALUDE_pig_count_l1326_132684

/-- Given a group of pigs and hens, if the total number of legs is 22 more than twice 
    the total number of heads, then the number of pigs is 11. -/
theorem pig_count (pigs hens : ℕ) : 
  4 * pigs + 2 * hens = 2 * (pigs + hens) + 22 → pigs = 11 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l1326_132684


namespace NUMINAMATH_CALUDE_tree_survival_probability_l1326_132641

/-- Probability that at least one tree survives after transplantation -/
theorem tree_survival_probability :
  let survival_rate_A : ℚ := 5/6
  let survival_rate_B : ℚ := 4/5
  let num_trees_A : ℕ := 2
  let num_trees_B : ℕ := 2
  -- Probability that at least one tree survives
  1 - (1 - survival_rate_A) ^ num_trees_A * (1 - survival_rate_B) ^ num_trees_B = 899/900 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_survival_probability_l1326_132641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1326_132607

theorem arithmetic_sequence_terms (a₁ : ℤ) (d : ℤ) (last : ℤ) (n : ℕ) :
  a₁ = -6 →
  d = 4 →
  last ≤ 50 →
  last = a₁ + (n - 1) * d →
  n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1326_132607


namespace NUMINAMATH_CALUDE_function_composition_ratio_l1326_132649

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l1326_132649


namespace NUMINAMATH_CALUDE_app_cost_calculation_l1326_132699

/-- Calculates the total cost of an app with online access -/
def total_cost (app_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  app_price + monthly_fee * months

/-- Theorem: The total cost for an app with an initial price of $5 and 
    a monthly online access fee of $8, used for 2 months, is $21 -/
theorem app_cost_calculation : total_cost 5 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_app_cost_calculation_l1326_132699


namespace NUMINAMATH_CALUDE_seven_bus_routes_l1326_132609

/-- Represents a bus stop in the network -/
structure BusStop :=
  (id : ℕ)

/-- Represents a bus route in the network -/
structure BusRoute :=
  (id : ℕ)
  (stops : Finset BusStop)

/-- Represents the entire bus network -/
structure BusNetwork :=
  (stops : Finset BusStop)
  (routes : Finset BusRoute)

/-- Every stop is reachable from any other stop without transfer -/
def all_stops_reachable (network : BusNetwork) : Prop :=
  ∀ s₁ s₂ : BusStop, s₁ ∈ network.stops → s₂ ∈ network.stops → 
    ∃ r : BusRoute, r ∈ network.routes ∧ s₁ ∈ r.stops ∧ s₂ ∈ r.stops

/-- Each pair of routes intersects at exactly one unique stop -/
def unique_intersection (network : BusNetwork) : Prop :=
  ∀ r₁ r₂ : BusRoute, r₁ ∈ network.routes → r₂ ∈ network.routes → r₁ ≠ r₂ →
    ∃! s : BusStop, s ∈ r₁.stops ∧ s ∈ r₂.stops

/-- Each route has exactly three stops -/
def three_stops_per_route (network : BusNetwork) : Prop :=
  ∀ r : BusRoute, r ∈ network.routes → Finset.card r.stops = 3

/-- There is more than one route -/
def multiple_routes (network : BusNetwork) : Prop :=
  ∃ r₁ r₂ : BusRoute, r₁ ∈ network.routes ∧ r₂ ∈ network.routes ∧ r₁ ≠ r₂

/-- The main theorem: Given the conditions, prove that there are 7 bus routes -/
theorem seven_bus_routes (network : BusNetwork) 
  (h1 : all_stops_reachable network)
  (h2 : unique_intersection network)
  (h3 : three_stops_per_route network)
  (h4 : multiple_routes network) :
  Finset.card network.routes = 7 :=
sorry

end NUMINAMATH_CALUDE_seven_bus_routes_l1326_132609


namespace NUMINAMATH_CALUDE_triangle_similarity_condition_l1326_132611

theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_condition_l1326_132611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_problem_l1326_132604

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Define the theorem
theorem arithmetic_sequence_log_problem 
  (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_arithmetic : arithmetic_sequence (λ n => 
    if n = 0 then log (a^5 * b^4)
    else if n = 1 then log (a^7 * b^9)
    else if n = 2 then log (a^10 * b^13)
    else if n = 9 then log (b^72)
    else 0)) : 
  ∃ n : ℕ, log (b^n) = (λ k => 
    if k = 0 then log (a^5 * b^4)
    else if k = 1 then log (a^7 * b^9)
    else if k = 2 then log (a^10 * b^13)
    else if k = 9 then log (b^72)
    else 0) 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_problem_l1326_132604


namespace NUMINAMATH_CALUDE_douglas_county_z_votes_l1326_132675

theorem douglas_county_z_votes
  (total_percent : ℝ)
  (county_x_percent : ℝ)
  (county_y_percent : ℝ)
  (county_x_voters : ℝ)
  (county_y_voters : ℝ)
  (county_z_voters : ℝ)
  (h1 : total_percent = 63)
  (h2 : county_x_percent = 74)
  (h3 : county_y_percent = 67)
  (h4 : county_x_voters = 3 * county_z_voters)
  (h5 : county_y_voters = 2 * county_z_voters)
  : ∃ county_z_percent : ℝ,
    county_z_percent = 22 ∧
    total_percent / 100 * (county_x_voters + county_y_voters + county_z_voters) =
    county_x_percent / 100 * county_x_voters +
    county_y_percent / 100 * county_y_voters +
    county_z_percent / 100 * county_z_voters :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_z_votes_l1326_132675


namespace NUMINAMATH_CALUDE_unique_a_value_l1326_132697

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℝ, 3 ∈ A a ∧ a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1326_132697


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1326_132682

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- Theorem: For an arithmetic sequence satisfying given conditions, a₈ = -26 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 8 = -26 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1326_132682


namespace NUMINAMATH_CALUDE_sea_glass_collection_l1326_132668

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : dorothy_total = 57) :
  ∃ (rose_blue : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    rose_blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l1326_132668


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1326_132679

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (-2 + 3 * Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1326_132679


namespace NUMINAMATH_CALUDE_factorization_problem_l1326_132667

theorem factorization_problem (x y : ℝ) : (x - y)^2 - 2*(x - y) + 1 = (x - y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l1326_132667


namespace NUMINAMATH_CALUDE_base9_multiplication_l1326_132621

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let digits := n.digits 9
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  n.digits 9 |> List.reverse |> List.foldl (fun acc d => acc * 10 + d) 0

/-- Multiplication in base-9 --/
def multBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem base9_multiplication :
  multBase9 327 6 = 2226 := by sorry

end NUMINAMATH_CALUDE_base9_multiplication_l1326_132621


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1326_132610

/-- 
Theorem: If an amount increases by 1/8th of itself each year for two years 
and results in 81000, then the initial amount was 64000.
-/
theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 81000) → P = 64000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1326_132610


namespace NUMINAMATH_CALUDE_factorization_proof_l1326_132629

variable (x y a b : ℝ)

theorem factorization_proof :
  (9*(3*x - 2*y)^2 - (x - y)^2 = (10*x - 7*y)*(8*x - 5*y)) ∧
  (a^2*b^2 + 4*a*b + 4 - b^2 = (a*b + 2 + b)*(a*b + 2 - b)) := by
  sorry


end NUMINAMATH_CALUDE_factorization_proof_l1326_132629


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1326_132674

/-- The number of ways to arrange n boys and m girls in a row with girls standing together -/
def arrange_girls_together (n m : ℕ) : ℕ := sorry

/-- The number of ways to arrange n boys and m girls in a row with no two boys next to each other -/
def arrange_boys_apart (n m : ℕ) : ℕ := sorry

theorem arrangement_theorem :
  (arrange_girls_together 3 4 = 576) ∧ 
  (arrange_boys_apart 3 4 = 1440) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1326_132674


namespace NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l1326_132686

theorem derivative_inequality_implies_function_inequality
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x < 2 * f x) : 
  Real.exp 2 * f 0 > f 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l1326_132686


namespace NUMINAMATH_CALUDE_binomial_product_l1326_132648

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1326_132648


namespace NUMINAMATH_CALUDE_greenfield_basketball_league_players_l1326_132660

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 7

/-- The total expenditure on socks and T-shirts for all players in dollars -/
def total_expenditure : ℕ := 4092

/-- The number of pairs of socks required per player -/
def socks_per_player : ℕ := 2

/-- The number of T-shirts required per player -/
def tshirts_per_player : ℕ := 2

/-- The cost of equipment (socks and T-shirts) for one player in dollars -/
def cost_per_player : ℕ := 
  socks_per_player * sock_cost + tshirts_per_player * (sock_cost + tshirt_additional_cost)

/-- The number of players in the Greenfield Basketball League -/
def num_players : ℕ := total_expenditure / cost_per_player

theorem greenfield_basketball_league_players : num_players = 108 := by
  sorry

end NUMINAMATH_CALUDE_greenfield_basketball_league_players_l1326_132660


namespace NUMINAMATH_CALUDE_vector_collinearity_l1326_132698

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem vector_collinearity (x : ℝ) :
  (∃ (k : ℝ), a - b x = k • b x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1326_132698


namespace NUMINAMATH_CALUDE_calculate_expression_l1326_132653

theorem calculate_expression : 2 * (8 ^ (1/3) - Real.sqrt 2) - (27 ^ (1/3) - Real.sqrt 2) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1326_132653


namespace NUMINAMATH_CALUDE_fourth_root_784_times_cube_root_512_l1326_132646

theorem fourth_root_784_times_cube_root_512 : 
  (784 : ℝ) ^ (1/4) * (512 : ℝ) ^ (1/3) = 16 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_784_times_cube_root_512_l1326_132646


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l1326_132661

/-- A regular nonagon -/
structure RegularNonagon where
  /-- Side length of the nonagon -/
  side_length : ℝ
  /-- Length of the shortest diagonal -/
  shortest_diagonal : ℝ
  /-- Length of the longest diagonal -/
  longest_diagonal : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length

/-- 
In a regular nonagon, the length of the longest diagonal 
is equal to the sum of the side length and the shortest diagonal 
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.side_length + n.shortest_diagonal = n.longest_diagonal := by
  sorry


end NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l1326_132661


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1326_132603

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p * q)  -- p, a, q form a geometric sequence
  (harith : b + c = p + q)  -- p, b, c, q form an arithmetic sequence
  : ∀ x : ℝ, b * x^2 - 2 * a * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1326_132603


namespace NUMINAMATH_CALUDE_max_triangle_area_l1326_132680

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 1/a^2 + 3/(4*b^2) = 1

/-- Line l intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*x + m

/-- Perpendicular bisector condition -/
def perp_bisector_condition (E : Ellipse) (l : IntersectingLine E) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/E.a^2 + y₁^2/E.b^2 = 1 ∧
    x₂^2/E.a^2 + y₂^2/E.b^2 = 1 ∧
    y₁ = l.k*x₁ + l.m ∧
    y₂ = l.k*x₂ + l.m ∧
    (x₁ + x₂)/2 = 0 ∧
    (y₁ + y₂)/2 = 1/2

/-- Area of triangle AOB -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ :=
  sorry

/-- Theorem: Maximum area of triangle AOB is 1 -/
theorem max_triangle_area (E : Ellipse) :
  ∃ (l : IntersectingLine E),
    perp_bisector_condition E l ∧
    triangle_area E l = 1 ∧
    ∀ (l' : IntersectingLine E),
      perp_bisector_condition E l' →
      triangle_area E l' ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1326_132680


namespace NUMINAMATH_CALUDE_no_common_points_l1326_132670

theorem no_common_points (m : ℝ) : 
  (∀ x y : ℝ, x + m^2 * y + 6 = 0 ∧ (m - 2) * x + 3 * m * y + 2 * m = 0 → False) ↔ 
  (m = 0 ∨ m = -1) :=
sorry

end NUMINAMATH_CALUDE_no_common_points_l1326_132670


namespace NUMINAMATH_CALUDE_bus_dispatch_interval_l1326_132665

/-- Represents the speed of a vehicle or person -/
structure Speed : Type :=
  (value : ℝ)

/-- Represents a time interval -/
structure TimeInterval : Type :=
  (minutes : ℝ)

/-- Represents a distance -/
structure Distance : Type :=
  (value : ℝ)

/-- The speed of Xiao Nan -/
def xiao_nan_speed : Speed := ⟨1⟩

/-- The speed of Xiao Yu -/
def xiao_yu_speed : Speed := ⟨3 * xiao_nan_speed.value⟩

/-- The time interval at which Xiao Nan encounters buses -/
def xiao_nan_encounter_interval : TimeInterval := ⟨10⟩

/-- The time interval at which Xiao Yu encounters buses -/
def xiao_yu_encounter_interval : TimeInterval := ⟨5⟩

/-- The speed of the bus -/
def bus_speed : Speed := ⟨5 * xiao_nan_speed.value⟩

/-- The distance between two consecutive buses -/
def bus_distance (s : Speed) (t : TimeInterval) : Distance :=
  ⟨s.value * t.minutes⟩

/-- The theorem stating that the interval between bus dispatches is 8 minutes -/
theorem bus_dispatch_interval :
  ∃ (t : TimeInterval),
    t.minutes = 8 ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value - xiao_nan_speed.value)) xiao_nan_encounter_interval ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value + xiao_yu_speed.value)) xiao_yu_encounter_interval :=
sorry

end NUMINAMATH_CALUDE_bus_dispatch_interval_l1326_132665


namespace NUMINAMATH_CALUDE_fraction_simplification_l1326_132690

theorem fraction_simplification : (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1326_132690


namespace NUMINAMATH_CALUDE_power_multiplication_l1326_132673

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1326_132673


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1326_132613

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1326_132613


namespace NUMINAMATH_CALUDE_surface_area_of_seven_solid_arrangement_l1326_132628

/-- Represents a 1 × 1 × 2 solid -/
structure Solid :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents the arrangement of solids as shown in the diagram -/
def Arrangement := List Solid

/-- Calculates the surface area of the arrangement -/
def surfaceArea (arr : Arrangement) : ℝ :=
  sorry

/-- The specific arrangement of seven solids as shown in the diagram -/
def sevenSolidArrangement : Arrangement :=
  List.replicate 7 { length := 1, width := 1, height := 2 }

theorem surface_area_of_seven_solid_arrangement :
  surfaceArea sevenSolidArrangement = 42 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_seven_solid_arrangement_l1326_132628


namespace NUMINAMATH_CALUDE_shoes_cost_calculation_l1326_132654

def shopping_problem (initial_amount sweater_cost tshirt_cost amount_left : ℕ) : Prop :=
  let total_spent := initial_amount - amount_left
  let other_items_cost := sweater_cost + tshirt_cost
  let shoes_cost := total_spent - other_items_cost
  shoes_cost = 11

theorem shoes_cost_calculation :
  shopping_problem 91 24 6 50 := by sorry

end NUMINAMATH_CALUDE_shoes_cost_calculation_l1326_132654


namespace NUMINAMATH_CALUDE_parabola_line_intersection_trajectory_l1326_132652

-- Define the parabola Ω
def Ω : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 5)^2 + p.2^2 = 16}

-- Define the line l
def l : Set (ℝ × ℝ) → Prop := λ L => ∃ (m b : ℝ), L = {p : ℝ × ℝ | p.1 = m * p.2 + b} ∨ L = {p : ℝ × ℝ | p.1 = 1} ∨ L = {p : ℝ × ℝ | p.1 = 9}

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem parabola_line_intersection_trajectory
  (L : Set (ℝ × ℝ))
  (A B M Q : ℝ × ℝ)
  (hΩ : Ω.Nonempty)
  (hC : C.Nonempty)
  (hl : l L)
  (hAB : A ∈ L ∧ B ∈ L ∧ A ∈ Ω ∧ B ∈ Ω ∧ A ≠ B)
  (hM : M ∈ L ∧ M ∈ C)
  (hMmid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hOAOB : (A.1 * B.1 + A.2 * B.2) = 0)
  (hQ : Q ∈ L ∧ (Q.1 - O.1) * (B.1 - A.1) + (Q.2 - O.2) * (B.2 - A.2) = 0) :
  Q.1^2 - 4 * Q.1 + Q.2^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_trajectory_l1326_132652


namespace NUMINAMATH_CALUDE_decimal_sum_theorem_l1326_132664

theorem decimal_sum_theorem : 0.03 + 0.004 + 0.009 + 0.0001 = 0.0431 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_theorem_l1326_132664


namespace NUMINAMATH_CALUDE_jellybean_count_l1326_132634

/-- The number of jellybeans in a dozen -/
def jellybeans_per_dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * jellybeans_per_dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have together -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1326_132634


namespace NUMINAMATH_CALUDE_touchdown_points_l1326_132658

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l1326_132658


namespace NUMINAMATH_CALUDE_simplify_fraction_l1326_132683

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1326_132683


namespace NUMINAMATH_CALUDE_urn_operations_theorem_l1326_132650

/-- Represents the state of the urn with white and black marbles -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the three possible operations on the urn -/
inductive Operation
  | removeBlackAddWhite
  | removeWhiteAddBoth
  | removeBothAddBoth

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeBlackAddWhite => 
      ⟨state.white + 3, state.black - 2⟩
  | Operation.removeWhiteAddBoth => 
      ⟨state.white - 2, state.black + 1⟩
  | Operation.removeBothAddBoth => 
      ⟨state.white - 2, state.black⟩

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial state -/
def applySequence (init : UrnState) (seq : OperationSequence) : UrnState :=
  seq.foldl applyOperation init

/-- The theorem to be proved -/
theorem urn_operations_theorem : 
  ∃ (seq : OperationSequence), 
    applySequence ⟨150, 50⟩ seq = ⟨148, 2⟩ :=
sorry

end NUMINAMATH_CALUDE_urn_operations_theorem_l1326_132650


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1326_132630

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1326_132630


namespace NUMINAMATH_CALUDE_snack_machine_cost_l1326_132643

/-- The total cost for students buying candy bars and chips -/
def total_cost (num_students : ℕ) (candy_price chip_price : ℚ) (candy_per_student chip_per_student : ℕ) : ℚ :=
  num_students * (candy_price * candy_per_student + chip_price * chip_per_student)

/-- Theorem: The total cost for 5 students to each get 1 candy bar at $2 and 2 bags of chips at $0.50 per bag is $15 -/
theorem snack_machine_cost : total_cost 5 2 (1/2) 1 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_snack_machine_cost_l1326_132643


namespace NUMINAMATH_CALUDE_min_value_of_f_l1326_132687

def f (x : ℝ) : ℝ := x^3 - 3*x + 9

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1326_132687


namespace NUMINAMATH_CALUDE_gcd_1729_1323_l1326_132600

theorem gcd_1729_1323 : Nat.gcd 1729 1323 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1323_l1326_132600


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l1326_132601

theorem least_reducible_fraction (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 34 → ¬(Nat.gcd (m - 29) (3 * m + 8) > 1)) ∧ 
  (34 > 0) ∧ 
  (Nat.gcd (34 - 29) (3 * 34 + 8) > 1) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l1326_132601


namespace NUMINAMATH_CALUDE_catering_pies_l1326_132622

theorem catering_pies (total_pies : ℕ) (num_teams : ℕ) (first_team_pies : ℕ) (third_team_pies : ℕ) 
  (h1 : total_pies = 750)
  (h2 : num_teams = 3)
  (h3 : first_team_pies = 235)
  (h4 : third_team_pies = 240) :
  total_pies - first_team_pies - third_team_pies = 275 := by
  sorry

end NUMINAMATH_CALUDE_catering_pies_l1326_132622


namespace NUMINAMATH_CALUDE_corner_spheres_sum_diameter_l1326_132612

-- Define a sphere in a corner
structure CornerSphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

-- Define the condition for a point on the sphere
def satisfiesCondition (s : CornerSphere) : Prop :=
  ∃ (x y z : ℝ), 
    (x - s.radius)^2 + (y - s.radius)^2 + (z - s.radius)^2 = s.radius^2 ∧
    x = 5 ∧ y = 5 ∧ z = 10

theorem corner_spheres_sum_diameter :
  ∀ (s1 s2 : CornerSphere),
    satisfiesCondition s1 → satisfiesCondition s2 →
    s1.center = (s1.radius, s1.radius, s1.radius) →
    s2.center = (s2.radius, s2.radius, s2.radius) →
    2 * (s1.radius + s2.radius) = 40 := by
  sorry

end NUMINAMATH_CALUDE_corner_spheres_sum_diameter_l1326_132612


namespace NUMINAMATH_CALUDE_proposition_is_false_l1326_132662

theorem proposition_is_false : ¬(∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l1326_132662


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1326_132695

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 4) :
  a 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1326_132695


namespace NUMINAMATH_CALUDE_sum_of_squares_2005_squared_l1326_132671

theorem sum_of_squares_2005_squared :
  ∃ (a b c d e f g h : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
    2005^2 = a^2 + b^2 ∧
    2005^2 = c^2 + d^2 ∧
    2005^2 = e^2 + f^2 ∧
    2005^2 = g^2 + h^2 ∧
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (a, b) ≠ (g, h) ∧
    (c, d) ≠ (e, f) ∧ (c, d) ≠ (g, h) ∧
    (e, f) ≠ (g, h) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_2005_squared_l1326_132671


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_relation_l1326_132619

/-- A function f has a derivative at x₀ -/
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - f' * (x - x₀)| ≤ ε * |x - x₀|

/-- x₀ is an extremum point of f -/
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The derivative of f at x₀ is 0 -/
def derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), has_derivative_at f x₀ ∧ f' = 0

theorem extremum_derivative_zero_relation (f : ℝ → ℝ) (x₀ : ℝ) :
  (has_derivative_at f x₀ →
    (is_extremum_point f x₀ → derivative_zero f x₀) ∧
    ¬(derivative_zero f x₀ → is_extremum_point f x₀)) :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_relation_l1326_132619


namespace NUMINAMATH_CALUDE_qiqi_problem_solving_l1326_132618

/-- Represents the number of problems completed in a given time -/
structure ProblemRate where
  problems : ℕ
  minutes : ℕ

/-- Calculates the number of problems that can be completed in a given time,
    given a known problem rate -/
def calculateProblems (rate : ProblemRate) (time : ℕ) : ℕ :=
  (rate.problems * time) / rate.minutes

theorem qiqi_problem_solving :
  let initialRate : ProblemRate := ⟨15, 5⟩
  calculateProblems initialRate 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_qiqi_problem_solving_l1326_132618


namespace NUMINAMATH_CALUDE_percentage_increase_lines_l1326_132693

theorem percentage_increase_lines (initial : ℕ) (final : ℕ) (increase_percent : ℚ) : 
  initial = 500 →
  final = 800 →
  increase_percent = 60 →
  (final - initial : ℚ) / initial * 100 = increase_percent :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_lines_l1326_132693


namespace NUMINAMATH_CALUDE_equation_transformation_l1326_132616

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1326_132616


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1326_132689

-- Define the function f(x) = 2x
def f (x : ℝ) : ℝ := 2 * x

-- Theorem stating that f is both odd and increasing
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1326_132689


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1326_132692

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 ≥ 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1326_132692


namespace NUMINAMATH_CALUDE_problem_1_l1326_132627

theorem problem_1 (x : ℝ) (hx : x ≠ 0) :
  (-2 * x^5 + 3 * x^3 - (1/2) * x^2) / ((-1/2 * x)^2) = -8 * x^3 + 12 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1326_132627


namespace NUMINAMATH_CALUDE_rope_cutting_l1326_132606

theorem rope_cutting (total_length : ℝ) (piece_length : ℝ) 
  (h1 : total_length = 20) 
  (h2 : piece_length = 3.8) : 
  (∃ (num_pieces : ℕ) (remaining : ℝ), 
    num_pieces = 5 ∧ 
    remaining = 1 ∧ 
    (↑num_pieces : ℝ) * piece_length + remaining = total_length ∧ 
    remaining < piece_length) := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l1326_132606


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l1326_132678

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school -/
def num_teachers (s : School) : ℕ :=
  let total_classes := s.num_students * s.classes_per_student
  let unique_classes := (total_classes + s.students_per_class - 1) / s.students_per_class
  (unique_classes + s.classes_per_teacher - 1) / s.classes_per_teacher

/-- King Middle School -/
def king_middle_school : School :=
  { num_students := 1200
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 35 }

theorem king_middle_school_teachers :
  num_teachers king_middle_school = 42 := by
  sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l1326_132678


namespace NUMINAMATH_CALUDE_product_equals_584638125_l1326_132625

theorem product_equals_584638125 : 625 * 935421 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_584638125_l1326_132625


namespace NUMINAMATH_CALUDE_sum_abcd_equals_neg_ten_thirds_l1326_132663

theorem sum_abcd_equals_neg_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_neg_ten_thirds_l1326_132663


namespace NUMINAMATH_CALUDE_shoes_sold_this_week_l1326_132672

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def still_needed : ℕ := 41

theorem shoes_sold_this_week :
  monthly_goal - sold_last_week - still_needed = 12 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_this_week_l1326_132672


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1326_132639

/-- The quadratic equation (r-4)x^2 - 2(r-3)x + r = 0 has two distinct roots, both greater than -1,
    if and only if 3.5 < r < 4.5 -/
theorem quadratic_roots_condition (r : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧
    (r - 4) * x₁^2 - 2*(r - 3) * x₁ + r = 0 ∧
    (r - 4) * x₂^2 - 2*(r - 3) * x₂ + r = 0) ↔
  (3.5 < r ∧ r < 4.5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1326_132639


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l1326_132656

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) → n ≤ 7 :=
by sorry

theorem exists_largest_n_binomial_sum : 
  ∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ n) :=
by sorry

theorem largest_n_is_seven : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 7) ∧
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l1326_132656


namespace NUMINAMATH_CALUDE_gcd_12m_18n_min_l1326_132635

/-- For positive integers m and n with gcd(m, n) = 10, the smallest possible value of gcd(12m, 18n) is 60 -/
theorem gcd_12m_18n_min (m n : ℕ+) (h : Nat.gcd m.val n.val = 10) :
  ∃ (k : ℕ+), (∀ (a b : ℕ+), Nat.gcd (12 * a.val) (18 * b.val) ≥ k.val) ∧
              (Nat.gcd (12 * m.val) (18 * n.val) = k.val) ∧
              k = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12m_18n_min_l1326_132635


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l1326_132647

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 12, width 14, and height 7 is 700 -/
theorem cuboid_surface_area_example : cuboid_surface_area 12 14 7 = 700 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l1326_132647


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1326_132638

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 15 = 0 ↔ x = 3 ∨ x = 5) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1326_132638


namespace NUMINAMATH_CALUDE_black_hole_convergence_l1326_132631

/-- Counts the number of even digits in a natural number -/
def countEvenDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- Counts the total number of digits in a natural number -/
def countTotalDigits (n : ℕ) : ℕ := sorry

/-- Applies the transformation rule to a natural number -/
def transform (n : ℕ) : ℕ :=
  100 * (countEvenDigits n) + 10 * (countOddDigits n) + (countTotalDigits n)

/-- The black hole number -/
def blackHoleNumber : ℕ := 123

/-- Theorem stating that repeated application of the transformation 
    will always result in the black hole number -/
theorem black_hole_convergence (n : ℕ) : 
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (transform^[m] n = blackHoleNumber) :=
sorry

end NUMINAMATH_CALUDE_black_hole_convergence_l1326_132631


namespace NUMINAMATH_CALUDE_car_stopping_distance_l1326_132605

/-- The distance function representing the car's motion during emergency braking -/
def s (t : ℝ) : ℝ := 30 * t - 5 * t^2

/-- The maximum distance traveled by the car before stopping -/
def max_distance : ℝ := 45

/-- Theorem stating that the maximum value of s(t) is 45 -/
theorem car_stopping_distance :
  ∃ t₀ : ℝ, ∀ t : ℝ, s t ≤ s t₀ ∧ s t₀ = max_distance :=
sorry

end NUMINAMATH_CALUDE_car_stopping_distance_l1326_132605


namespace NUMINAMATH_CALUDE_four_digit_sum_divisible_by_nine_l1326_132685

theorem four_digit_sum_divisible_by_nine 
  (a b c d e f g h i j : Nat) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
              f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
              g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
              h ≠ i ∧ h ≠ j ∧
              i ≠ j)
  (digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10)
  (sum_equality : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  (1000 * g + 100 * h + 10 * i + j) % 9 = 0 := by
  sorry


end NUMINAMATH_CALUDE_four_digit_sum_divisible_by_nine_l1326_132685


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_is_empty_l1326_132602

-- Define the sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x | 10^2 * 2 = 10^2}

-- State the theorem
theorem intersection_A_complement_B_is_empty :
  A ∩ Bᶜ = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_is_empty_l1326_132602


namespace NUMINAMATH_CALUDE_proposition_is_false_l1326_132655

theorem proposition_is_false : ∃ m n : ℤ, m > n ∧ m^2 ≤ n^2 := by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l1326_132655


namespace NUMINAMATH_CALUDE_rectangle_area_l1326_132617

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1326_132617


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l1326_132688

/-- The number of ways to arrange students from 5 grades visiting 5 museums,
    with exactly 2 grades visiting the Jia Science Museum -/
def arrangement_count : ℕ :=
  Nat.choose 5 2 * (4^3)

/-- Theorem stating that the number of arrangements is correct -/
theorem arrangement_count_correct :
  arrangement_count = Nat.choose 5 2 * (4^3) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l1326_132688


namespace NUMINAMATH_CALUDE_infinite_occurrence_in_sequence_l1326_132624

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Computes the sum of digits of a natural number in decimal system -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- Computes a_n for a given polynomial and natural number n -/
def computeA (P : IntPolynomial) (n : Nat) : Nat :=
  sumOfDigits (sorry)  -- Evaluate P(n) and compute sum of digits

/-- The main theorem -/
theorem infinite_occurrence_in_sequence (P : IntPolynomial) :
  ∃ (k : Nat), Set.Infinite {n : Nat | computeA P n = k} :=
sorry

end NUMINAMATH_CALUDE_infinite_occurrence_in_sequence_l1326_132624


namespace NUMINAMATH_CALUDE_inverse_sixteen_mod_97_l1326_132691

theorem inverse_sixteen_mod_97 (h : (8⁻¹ : ZMod 97) = 85) : (16⁻¹ : ZMod 97) = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sixteen_mod_97_l1326_132691


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1326_132632

/-- The number of cakes the baker still has after selling some -/
def remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) : ℕ :=
  cakes_made - cakes_sold

/-- Theorem stating that the baker has 139 cakes remaining -/
theorem baker_remaining_cakes :
  remaining_cakes 149 10 = 139 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1326_132632


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1326_132659

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence where the third term is equal to twice 
the sum of the first two terms plus 1, and the fourth term is equal 
to twice the sum of the first three terms plus 1, prove that the 
common ratio is 3.
-/
theorem geometric_sequence_common_ratio 
  (seq : GeometricSequence) 
  (h₁ : seq.a 3 = 2 * (seq.a 1 + seq.a 2) + 1)
  (h₂ : seq.a 4 = 2 * (seq.a 1 + seq.a 2 + seq.a 3) + 1) : 
  seq.q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1326_132659


namespace NUMINAMATH_CALUDE_lucas_100_mod_9_l1326_132669

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- Lucas sequence modulo 9 has a period of 12 -/
axiom lucas_mod_9_period (n : ℕ) : lucas (n + 12) % 9 = lucas n % 9

/-- The 4th term of Lucas sequence modulo 9 is 7 -/
axiom lucas_4_mod_9 : lucas 3 % 9 = 7

theorem lucas_100_mod_9 : lucas 99 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_100_mod_9_l1326_132669


namespace NUMINAMATH_CALUDE_additional_workers_needed_l1326_132651

/-- Represents the problem of determining the number of additional workers needed to complete a construction project on time. -/
theorem additional_workers_needed
  (total_days : ℕ)
  (initial_workers : ℕ)
  (days_passed : ℕ)
  (work_completed_percentage : ℚ)
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed_percentage = 2/5)
  : ∃ (additional_workers : ℕ),
    (initial_workers + additional_workers) * (total_days - days_passed) =
    (1 - work_completed_percentage) * (initial_workers * total_days) ∧
    additional_workers = 4 := by
  sorry

end NUMINAMATH_CALUDE_additional_workers_needed_l1326_132651
