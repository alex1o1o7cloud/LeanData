import Mathlib

namespace set_operations_l125_12516

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x < 5}

-- State the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (Aᶜ ∩ B = {x | x < 2}) := by sorry

end set_operations_l125_12516


namespace overtime_hours_l125_12561

theorem overtime_hours (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 198 →
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 13 := by sorry

end overtime_hours_l125_12561


namespace min_value_theorem_l125_12549

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end min_value_theorem_l125_12549


namespace b_over_a_is_real_l125_12547

variable (a b x y : ℂ)

theorem b_over_a_is_real
  (h1 : a * b ≠ 0)
  (h2 : Complex.abs x = Complex.abs y)
  (h3 : x + y = a)
  (h4 : x * y = b) :
  ∃ (r : ℝ), b / a = r :=
sorry

end b_over_a_is_real_l125_12547


namespace function_behavior_l125_12587

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end function_behavior_l125_12587


namespace death_rate_calculation_l125_12544

/-- Given a birth rate, net growth rate, and initial population, 
    calculate the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) 
                          (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Theorem stating that under the given conditions, 
    the death rate is 16 per certain number of people. -/
theorem death_rate_calculation :
  let birth_rate : ℝ := 52
  let net_growth_rate : ℝ := 0.012
  let initial_population : ℝ := 3000
  calculate_death_rate birth_rate net_growth_rate initial_population = 16 := by
  sorry

#eval calculate_death_rate 52 0.012 3000

end death_rate_calculation_l125_12544


namespace range_of_a_given_p_and_q_l125_12530

-- Define the propositions
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - Real.log x - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_of_a : Set ℝ :=
  Set.Ici (-4) ∪ Set.Icc (-2) 1

-- State the theorem
theorem range_of_a_given_p_and_q :
  ∀ a : ℝ, prop_p a ∧ prop_q a ↔ a ∈ range_of_a :=
sorry

end range_of_a_given_p_and_q_l125_12530


namespace thousandth_digit_is_zero_l125_12534

def factorial (n : ℕ) : ℕ := Nat.factorial n

def expression : ℚ := (factorial 13 * factorial 23 + factorial 15 * factorial 17) / 7

theorem thousandth_digit_is_zero :
  ∃ (n : ℕ), n ≥ 1000 ∧ (expression * 10^n).floor % 10 = 0 :=
sorry

end thousandth_digit_is_zero_l125_12534


namespace smaller_number_in_ratio_l125_12529

theorem smaller_number_in_ratio (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / b → x * y = d → 
  x = Real.sqrt ((a * d) / b) ∧ x < y := by sorry

end smaller_number_in_ratio_l125_12529


namespace remainder_problem_l125_12585

theorem remainder_problem (k : ℕ) :
  k > 0 ∧ k < 100 ∧
  k % 5 = 2 ∧
  k % 6 = 3 ∧
  k % 8 = 7 →
  k % 9 = 6 := by
sorry

end remainder_problem_l125_12585


namespace second_grade_sample_l125_12506

/-- Calculates the number of students to be drawn from the second grade in a stratified sample -/
def students_from_second_grade (total_sample : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) : ℕ :=
  (total_sample * ratio_second) / (ratio_first + ratio_second + ratio_third)

/-- Theorem: Given the conditions, the number of students to be drawn from the second grade is 15 -/
theorem second_grade_sample :
  students_from_second_grade 50 3 3 4 = 15 := by
  sorry

#eval students_from_second_grade 50 3 3 4

end second_grade_sample_l125_12506


namespace product_sum_theorem_l125_12507

theorem product_sum_theorem (p q r s t : ℤ) :
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 25 := by
sorry

end product_sum_theorem_l125_12507


namespace animal_arrangement_count_l125_12517

-- Define the number of each type of animal
def num_parrots : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 4

-- Define the total number of animals
def total_animals : ℕ := num_parrots + num_dogs + num_cats

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  2 * (Nat.factorial num_parrots) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

-- Theorem statement
theorem animal_arrangement_count :
  num_arrangements = 34560 := by sorry

end animal_arrangement_count_l125_12517


namespace tan_alpha_value_l125_12500

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/3 := by
sorry

end tan_alpha_value_l125_12500


namespace smallest_gcd_multiple_l125_12525

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 10) :
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 10 ∧ Nat.gcd (8 * m') (12 * n') = 40) ∧
  (∀ (m'' n'' : ℕ+), Nat.gcd m'' n'' = 10 → Nat.gcd (8 * m'') (12 * n'') ≥ 40) :=
sorry

end smallest_gcd_multiple_l125_12525


namespace sphere_radius_from_depression_l125_12532

/-- The radius of a sphere that creates a circular depression with given diameter and depth when partially submerged. -/
def sphere_radius (depression_diameter : ℝ) (depression_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a sphere with radius 13cm creates a circular depression
    with diameter 24cm and depth 8cm when partially submerged. -/
theorem sphere_radius_from_depression :
  sphere_radius 24 8 = 13 := by
  sorry

end sphere_radius_from_depression_l125_12532


namespace no_additional_cocoa_needed_l125_12501

/-- Represents the chocolate cake recipe and baking scenario. -/
structure ChocolateCakeScenario where
  recipe_ratio : Real  -- Amount of cocoa powder per pound of cake batter
  cake_weight : Real   -- Total weight of the cake to be made
  given_cocoa : Real   -- Amount of cocoa powder already provided

/-- Calculates if additional cocoa powder is needed for the chocolate cake. -/
def additional_cocoa_needed (scenario : ChocolateCakeScenario) : Real :=
  scenario.recipe_ratio * scenario.cake_weight - scenario.given_cocoa

/-- Proves that no additional cocoa powder is needed in the given scenario. -/
theorem no_additional_cocoa_needed (scenario : ChocolateCakeScenario) 
  (h1 : scenario.recipe_ratio = 0.4)
  (h2 : scenario.cake_weight = 450)
  (h3 : scenario.given_cocoa = 259) : 
  additional_cocoa_needed scenario ≤ 0 := by
  sorry

#eval additional_cocoa_needed { recipe_ratio := 0.4, cake_weight := 450, given_cocoa := 259 }

end no_additional_cocoa_needed_l125_12501


namespace min_value_sin_cos_l125_12546

theorem min_value_sin_cos (p q : ℝ) : 
  (∀ θ : ℝ, p * Real.sin θ - q * Real.cos θ ≥ -Real.sqrt (p^2 + q^2)) ∧ 
  (∃ θ : ℝ, p * Real.sin θ - q * Real.cos θ = -Real.sqrt (p^2 + q^2)) := by
sorry

end min_value_sin_cos_l125_12546


namespace twenty_customers_without_fish_l125_12541

/-- Represents the fish market scenario -/
structure FishMarket where
  total_customers : ℕ
  num_tuna : ℕ
  tuna_weight : ℕ
  customer_request : ℕ

/-- Calculates the number of customers who will go home without fish -/
def customers_without_fish (market : FishMarket) : ℕ :=
  market.total_customers - (market.num_tuna * market.tuna_weight / market.customer_request)

/-- Theorem stating that in the given scenario, 20 customers will go home without fish -/
theorem twenty_customers_without_fish :
  let market : FishMarket := {
    total_customers := 100,
    num_tuna := 10,
    tuna_weight := 200,
    customer_request := 25
  }
  customers_without_fish market = 20 := by sorry

end twenty_customers_without_fish_l125_12541


namespace lcm_from_product_and_gcd_l125_12558

theorem lcm_from_product_and_gcd (a b c : ℕ+) :
  a * b * c = 1354808 ∧ Nat.gcd a (Nat.gcd b c) = 11 →
  Nat.lcm a (Nat.lcm b c) = 123164 := by
  sorry

end lcm_from_product_and_gcd_l125_12558


namespace sum_of_x_and_y_l125_12595

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 := by
  sorry

end sum_of_x_and_y_l125_12595


namespace binary_ternary_equality_l125_12545

/-- Represents a digit in base 2 (binary) -/
def BinaryDigit : Type := {n : ℕ // n < 2}

/-- Represents a digit in base 3 (ternary) -/
def TernaryDigit : Type := {n : ℕ // n < 3}

/-- Converts a binary number to decimal -/
def binaryToDecimal (d₂ : BinaryDigit) (d₁ : BinaryDigit) (d₀ : BinaryDigit) : ℕ :=
  d₂.val * 2^2 + d₁.val * 2^1 + d₀.val * 2^0

/-- Converts a ternary number to decimal -/
def ternaryToDecimal (d₂ : TernaryDigit) (d₁ : TernaryDigit) (d₀ : TernaryDigit) : ℕ :=
  d₂.val * 3^2 + d₁.val * 3^1 + d₀.val * 3^0

theorem binary_ternary_equality :
  ∀ (x : TernaryDigit) (y : BinaryDigit),
    binaryToDecimal ⟨1, by norm_num⟩ y ⟨1, by norm_num⟩ = ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ →
    x.val = 1 ∧ y.val = 1 ∧ ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ = 11 :=
by sorry

#check binary_ternary_equality

end binary_ternary_equality_l125_12545


namespace conic_not_parabola_l125_12596

/-- A conic section represented by the equation x^2 + ky^2 = 1 -/
def conic (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- Definition of a parabola -/
def is_parabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ S ↔ a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0

/-- Theorem: The conic section x^2 + ky^2 = 1 is not a parabola for any real k -/
theorem conic_not_parabola : ∀ (k : ℝ), ¬(is_parabola (conic k)) := by
  sorry

end conic_not_parabola_l125_12596


namespace astronomical_unit_scientific_notation_l125_12551

/-- One astronomical unit in kilometers -/
def astronomical_unit : ℝ := 1.496e9

/-- Scientific notation representation of one astronomical unit -/
def astronomical_unit_scientific : ℝ := 1.496 * 10^8

/-- Theorem stating that the astronomical unit can be expressed in scientific notation -/
theorem astronomical_unit_scientific_notation :
  astronomical_unit = astronomical_unit_scientific := by
  sorry

end astronomical_unit_scientific_notation_l125_12551


namespace cost_per_square_meter_is_two_l125_12527

-- Define the lawn dimensions
def lawn_length : ℝ := 80
def lawn_width : ℝ := 60

-- Define the road width
def road_width : ℝ := 10

-- Define the total cost of traveling both roads
def total_cost : ℝ := 2600

-- Theorem to prove
theorem cost_per_square_meter_is_two :
  let road_area := (lawn_length * road_width + lawn_width * road_width) - road_width * road_width
  total_cost / road_area = 2 := by
  sorry

end cost_per_square_meter_is_two_l125_12527


namespace vector_problems_l125_12502

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_problems :
  (∃ k : ℝ, (a.1 + k * c.1, a.2 + k * c.2) • (2 * b.1 - a.1, 2 * b.2 - a.2) = 0 → k = -11/18) ∧
  (∃ d : ℝ × ℝ, ∃ t : ℝ, d = (t * c.1, t * c.2) ∧ d.1^2 + d.2^2 = 34 → 
    d = (4 * Real.sqrt 2, Real.sqrt 2) ∨ d = (-4 * Real.sqrt 2, -Real.sqrt 2)) :=
by sorry

#check vector_problems

end vector_problems_l125_12502


namespace sum_with_radical_conjugate_l125_12538

theorem sum_with_radical_conjugate : 
  let a : ℝ := 15
  let b : ℝ := Real.sqrt 500
  (a - b) + (a + b) = 30 := by sorry

end sum_with_radical_conjugate_l125_12538


namespace decoration_problem_l125_12505

theorem decoration_problem (total_decorations : ℕ) (nails_used : ℕ) 
  (h1 : total_decorations = (3 * nails_used) / 2)
  (h2 : nails_used = 50) : 
  total_decorations - nails_used - (2 * (total_decorations - nails_used)) / 5 = 15 := by
  sorry

end decoration_problem_l125_12505


namespace complex_number_location_l125_12583

theorem complex_number_location :
  ∀ z : ℂ, z * (2 + Complex.I) = 1 + 3 * Complex.I →
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_location_l125_12583


namespace sams_work_hours_sams_september_february_hours_l125_12554

/-- Calculates the number of hours Sam worked from September to February -/
theorem sams_work_hours (earnings_mar_aug : ℝ) (hours_mar_aug : ℝ) (console_cost : ℝ) (car_repair_cost : ℝ) (remaining_hours : ℝ) : ℝ :=
  let hourly_rate := earnings_mar_aug / hours_mar_aug
  let remaining_earnings := console_cost - (earnings_mar_aug - car_repair_cost)
  let total_hours_needed := remaining_earnings / hourly_rate
  total_hours_needed - remaining_hours

/-- Proves that Sam worked 8 hours from September to February -/
theorem sams_september_february_hours : sams_work_hours 460 23 600 340 16 = 8 := by
  sorry

end sams_work_hours_sams_september_february_hours_l125_12554


namespace magnitude_of_vector_difference_l125_12563

/-- Given plane vectors a and b with specified properties, prove that |2a-b| = √91 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (Real.cos (5 * Real.pi / 6) = a.1 * b.1 + a.2 * b.2) →  -- angle between a and b is 5π/6
  (a.1^2 + a.2^2 = 16) →  -- |a| = 4
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 91) :=  -- |2a-b| = √91
by sorry

end magnitude_of_vector_difference_l125_12563


namespace six_digit_numbers_with_zero_l125_12581

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (six_digit_no_zero : ℕ) :
  total_six_digit = 9 * 10^5 →
  six_digit_no_zero = 9^6 →
  total_six_digit - six_digit_no_zero = 368559 := by
sorry

end six_digit_numbers_with_zero_l125_12581


namespace inverse_variation_problem_l125_12513

/-- Given that p and q vary inversely, prove that when q = 2.8 for p = 500, 
    then q = 1.12 when p = 1250 -/
theorem inverse_variation_problem (p q : ℝ) (h : p * q = 500 * 2.8) :
  p = 1250 → q = 1.12 := by
  sorry

end inverse_variation_problem_l125_12513


namespace fish_speed_problem_l125_12512

/-- Calculates the downstream speed of a fish given its upstream and still water speeds. -/
def fish_downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: A fish with an upstream speed of 35 kmph and a still water speed of 45 kmph 
    has a downstream speed of 55 kmph. -/
theorem fish_speed_problem :
  fish_downstream_speed 35 45 = 55 := by
  sorry

#eval fish_downstream_speed 35 45

end fish_speed_problem_l125_12512


namespace right_triangle_cos_B_l125_12584

theorem right_triangle_cos_B (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c^2 = a^2 + b^2) : 
  let cos_B := b / c
  cos_B = 15 / 17 := by
sorry

end right_triangle_cos_B_l125_12584


namespace single_colony_days_l125_12586

/-- Represents the growth of bacteria colonies -/
def BacteriaGrowth : Type :=
  { n : ℕ // n > 0 }

/-- The number of days it takes for two colonies to reach the habitat limit -/
def two_colony_days : BacteriaGrowth := ⟨15, by norm_num⟩

/-- Calculates the size of a colony after n days, given its initial size -/
def colony_size (initial : ℕ) (days : ℕ) : ℕ :=
  initial * 2^days

/-- Theorem stating that a single colony takes 16 days to reach the habitat limit -/
theorem single_colony_days :
  ∃ (limit : ℕ), limit > 0 ∧
    colony_size 1 (two_colony_days.val + 1) = limit ∧
    colony_size 2 two_colony_days.val = limit := by
  sorry

end single_colony_days_l125_12586


namespace airline_capacity_l125_12573

/-- Calculates the number of passengers an airline can accommodate daily --/
theorem airline_capacity
  (num_airplanes : ℕ)
  (rows_per_airplane : ℕ)
  (seats_per_row : ℕ)
  (flights_per_day : ℕ)
  (h1 : num_airplanes = 5)
  (h2 : rows_per_airplane = 20)
  (h3 : seats_per_row = 7)
  (h4 : flights_per_day = 2) :
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day = 7000 :=
by sorry

end airline_capacity_l125_12573


namespace min_value_expression_l125_12535

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (x + 2) * (2 * y + 1) / (x * y) ≥ 19 + 4 * Real.sqrt 15 :=
by sorry

end min_value_expression_l125_12535


namespace abc_value_l125_12510

theorem abc_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) 
  (h3 : a + b + c = 10) : 
  a * b * c = -150 + 15 * Real.sqrt 69 := by
  sorry

end abc_value_l125_12510


namespace quadratic_value_l125_12539

/-- A quadratic function with axis of symmetry at x = 3.5 and p(-6) = 0 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_value (d e f : ℝ) :
  (∀ x : ℝ, p d e f x = p d e f (7 - x)) →  -- Axis of symmetry at x = 3.5
  p d e f (-6) = 0 →                        -- p(-6) = 0
  ∃ n : ℤ, p d e f 13 = n →                 -- p(13) is an integer
  p d e f 13 = 0 :=                         -- Conclusion: p(13) = 0
by
  sorry

end quadratic_value_l125_12539


namespace sin_arccos_three_fifths_l125_12521

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end sin_arccos_three_fifths_l125_12521


namespace existence_of_x_y_for_power_of_two_l125_12598

theorem existence_of_x_y_for_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ x y : ℕ+, 2^n = 7 * x^2 + y^2 := by
  sorry

end existence_of_x_y_for_power_of_two_l125_12598


namespace larger_number_of_pair_l125_12553

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 50) : max x y = 29 := by
  sorry

end larger_number_of_pair_l125_12553


namespace martin_martina_ages_l125_12540

/-- Martin's age -/
def martin_age : ℕ := 33

/-- Martina's age -/
def martina_age : ℕ := 22

/-- The condition from Martin's statement -/
def martin_condition (x y : ℕ) : Prop :=
  x = 3 * (y - (x - y))

/-- The condition from Martina's statement -/
def martina_condition (x y : ℕ) : Prop :=
  x + (x + (x - y)) = 77

theorem martin_martina_ages :
  martin_condition martin_age martina_age ∧
  martina_condition martin_age martina_age :=
by sorry

end martin_martina_ages_l125_12540


namespace quadratic_cubic_relation_l125_12562

theorem quadratic_cubic_relation (x : ℝ) : x^2 + x - 1 = 0 → 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end quadratic_cubic_relation_l125_12562


namespace triangle_area_from_squares_l125_12543

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 225) (hc : c^2 = 324) :
  (1/2) * a * b = 45 :=
sorry

end triangle_area_from_squares_l125_12543


namespace exactly_two_sunny_days_probability_l125_12576

theorem exactly_two_sunny_days_probability 
  (num_days : ℕ) 
  (rain_prob : ℝ) 
  (sunny_prob : ℝ) :
  num_days = 3 →
  rain_prob = 0.6 →
  sunny_prob = 1 - rain_prob →
  (num_days.choose 2 : ℝ) * sunny_prob^2 * rain_prob = 54/125 :=
by sorry

end exactly_two_sunny_days_probability_l125_12576


namespace quadratic_inequality_l125_12590

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - (k - 4)*x - k + 7 > 0) ↔ (k > 4 ∧ k < 6) :=
sorry

end quadratic_inequality_l125_12590


namespace mod_63_calculation_l125_12528

theorem mod_63_calculation : ∃ (a b : ℤ), 
  (7 * a) % 63 = 1 ∧ 
  (13 * b) % 63 = 1 ∧ 
  (3 * a + 9 * b) % 63 = 48 := by
  sorry

end mod_63_calculation_l125_12528


namespace tangent_slope_at_one_l125_12565

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 3 := by sorry

end tangent_slope_at_one_l125_12565


namespace quadratic_factorization_l125_12597

theorem quadratic_factorization (x : ℝ) : x^2 - 7*x + 10 = (x - 2) * (x - 5) := by
  sorry

end quadratic_factorization_l125_12597


namespace only_f4_decreasing_l125_12557

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem stating that only f4 has a negative derivative for all real x
theorem only_f4_decreasing :
  (∀ x : ℝ, deriv f1 x > 0) ∧
  (∃ x : ℝ, deriv f2 x ≥ 0) ∧
  (∀ x : ℝ, deriv f3 x > 0) ∧
  (∀ x : ℝ, deriv f4 x < 0) :=
by sorry

end only_f4_decreasing_l125_12557


namespace absolute_value_z_l125_12518

theorem absolute_value_z (w z : ℂ) : 
  w * z = 20 - 21 * I → Complex.abs w = Real.sqrt 29 → Complex.abs z = Real.sqrt 29 := by
  sorry

end absolute_value_z_l125_12518


namespace inequality_solution_sets_l125_12560

open Set

theorem inequality_solution_sets (a b : ℝ) :
  {x : ℝ | a * x - b < 0} = Ioi 1 →
  {x : ℝ | (a * x + b) * (x - 3) > 0} = Ioo (-1) 3 := by
  sorry

end inequality_solution_sets_l125_12560


namespace sum_two_smallest_angles_l125_12504

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles of the quadrilateral
def angle (P Q R : Point) : ℝ := sorry

-- Define the conditions
axiom quad_angles_arithmetic : ∃ (a d : ℝ), 
  angle B A D = a ∧
  angle A B C = a + d ∧
  angle B C D = a + 2*d ∧
  angle C D A = a + 3*d

axiom angle_equality1 : angle A B D = angle D B C
axiom angle_equality2 : angle A D B = angle B D C

axiom triangle_ABD_arithmetic : ∃ (x y : ℝ),
  angle B A D = x ∧
  angle A B D = x + y ∧
  angle A D B = x + 2*y

axiom triangle_DCB_arithmetic : ∃ (x y : ℝ),
  angle D C B = x ∧
  angle C D B = x + y ∧
  angle C B D = x + 2*y

axiom smallest_angle : angle B A D = 10
axiom second_angle : angle A B C = 70

-- Theorem to prove
theorem sum_two_smallest_angles :
  angle B A D + angle A B C = 80 := by sorry

end sum_two_smallest_angles_l125_12504


namespace division_remainder_l125_12531

theorem division_remainder : 
  let a := 555
  let b := 445
  let number := 220030
  let sum := a + b
  let diff := a - b
  let quotient := 2 * diff
  number % sum = 30 := by
sorry

end division_remainder_l125_12531


namespace train_length_l125_12579

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 150 := by
  sorry

end train_length_l125_12579


namespace manager_percentage_after_leaving_l125_12515

/-- Calculates the new percentage of managers after some leave the room -/
def new_manager_percentage (initial_employees : ℕ) (initial_manager_percentage : ℚ) 
  (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_manager_percentage / 100) * initial_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  let remaining_employees : ℚ := initial_employees - managers_leaving
  (remaining_managers / remaining_employees) * 100

/-- Theorem stating that given the initial conditions and managers leaving, 
    the new percentage of managers is 98% -/
theorem manager_percentage_after_leaving :
  new_manager_percentage 200 99 99.99999999999991 = 98 := by
  sorry

end manager_percentage_after_leaving_l125_12515


namespace shaded_area_approx_l125_12589

/-- The area of the shaded region formed by two circles with radii 3 and 6 -/
def shaded_area (π : ℝ) : ℝ :=
  let small_radius : ℝ := 3
  let large_radius : ℝ := 6
  let left_rectangle : ℝ := small_radius * (2 * small_radius)
  let right_rectangle : ℝ := large_radius * (2 * large_radius)
  let small_semicircle : ℝ := 0.5 * π * small_radius ^ 2
  let large_semicircle : ℝ := 0.5 * π * large_radius ^ 2
  (left_rectangle + right_rectangle) - (small_semicircle + large_semicircle)

theorem shaded_area_approx :
  ∃ (π : ℝ), abs (shaded_area π - 19.3) < 0.05 :=
sorry

end shaded_area_approx_l125_12589


namespace least_prime_factor_of_5_6_minus_5_4_l125_12574

theorem least_prime_factor_of_5_6_minus_5_4 :
  Nat.minFac (5^6 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5_6_minus_5_4_l125_12574


namespace measure_six_with_special_ruler_l125_12591

/-- A ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Definition of a ruler with marks at 0, 2, and 5 -/
def specialRuler : Ruler :=
  { marks := [0, 2, 5] }

/-- A function to check if a length can be measured using a given ruler -/
def canMeasure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (a b : ℝ), a ∈ r.marks ∧ b ∈ r.marks ∧ (b - a = length ∨ a - b = length)

/-- Theorem stating that the special ruler can measure a segment of length 6 -/
theorem measure_six_with_special_ruler :
  canMeasure specialRuler 6 := by
  sorry


end measure_six_with_special_ruler_l125_12591


namespace equation_solution_l125_12578

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ -4 ∧ (-x^2 = (4*x + 2) / (x + 4)) ↔ (x = -1 ∨ x = -2) :=
by sorry

end equation_solution_l125_12578


namespace solution_set_implies_a_equals_one_l125_12552

/-- The solution set of the inequality (x+a)/(x^2+4x+3) > 0 --/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x | (x + a) / (x^2 + 4*x + 3) > 0}

/-- The theorem stating that if the solution set is {x | x > -3, x ≠ -1}, then a = 1 --/
theorem solution_set_implies_a_equals_one :
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1}) →
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1} ∧ a = 1) :=
by sorry

end solution_set_implies_a_equals_one_l125_12552


namespace solution_set_when_m_neg_one_m_range_for_subset_condition_l125_12550

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem m_range_for_subset_condition :
  {m : ℝ | ∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2 * x + 1|} = Set.Icc (-11/4 : ℝ) 0 := by sorry

end solution_set_when_m_neg_one_m_range_for_subset_condition_l125_12550


namespace dave_ice_cubes_l125_12570

theorem dave_ice_cubes (original : ℕ) (new : ℕ) (total : ℕ) : 
  original = 2 → new = 7 → total = original + new → total = 9 := by
  sorry

end dave_ice_cubes_l125_12570


namespace min_value_of_sum_squares_l125_12593

theorem min_value_of_sum_squares (x y z : ℝ) 
  (eq1 : x + 2*y - 5*z = 3)
  (eq2 : x - 2*y - z = -5) :
  ∃ (min : ℝ), min = 54/11 ∧ ∀ (x' y' z' : ℝ), 
    x' + 2*y' - 5*z' = 3 → x' - 2*y' - z' = -5 → 
    x'^2 + y'^2 + z'^2 ≥ min :=
by sorry

end min_value_of_sum_squares_l125_12593


namespace derivative_of_f_l125_12599

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) :
  deriv f x = (-x * Real.sin x - Real.cos x) / (x^2) := by
  sorry

end derivative_of_f_l125_12599


namespace sqrt_three_seven_plus_four_sqrt_three_l125_12524

theorem sqrt_three_seven_plus_four_sqrt_three :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := by
  sorry

end sqrt_three_seven_plus_four_sqrt_three_l125_12524


namespace x_intercept_of_line_l125_12508

/-- The x-intercept of a line is a point where the line crosses the x-axis (y = 0). -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is represented as ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 4 7 28 = (7, 0) ∧
  line_equation 4 7 28 (x_intercept 4 7 28).1 (x_intercept 4 7 28).2 :=
by sorry

end x_intercept_of_line_l125_12508


namespace percentage_calculations_l125_12567

theorem percentage_calculations (M N : ℝ) (h : M < N) :
  (100 * (N - M) / M = (N - M) / M * 100) ∧
  (100 * M / N = M / N * 100) :=
by sorry

end percentage_calculations_l125_12567


namespace scale_division_theorem_l125_12533

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of equal parts the scale is divided into -/
def num_parts : ℕ := 5

/-- The length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division_theorem :
  inches_to_feet_and_inches part_length = (2, 1) := by
  sorry

end scale_division_theorem_l125_12533


namespace ed_doug_marble_difference_l125_12568

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug : ℕ) (ed_lost : ℕ) (ed_doug_diff : ℕ) :
  ed_initial > doug →
  ed_initial = 91 →
  ed_lost = 21 →
  ed_initial - ed_lost - doug = ed_doug_diff →
  ed_doug_diff = 9 →
  ed_initial - doug = 30 :=
by
  sorry

end ed_doug_marble_difference_l125_12568


namespace exp_properties_l125_12577

-- Define the exponential function
noncomputable def Exp : ℝ → ℝ := Real.exp

-- Theorem statement
theorem exp_properties :
  (∀ (a b x : ℝ), Exp ((a + b) * x) = Exp (a * x) * Exp (b * x)) ∧
  (∀ (x : ℝ) (k : ℕ), Exp (k * x) = (Exp x) ^ k) := by
  sorry

end exp_properties_l125_12577


namespace valid_numbers_l125_12526

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 100) % 10 = n % 10) ∧
  (n ^ 2) % ((n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)) = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1111, 1212, 1515, 2424, 3636} :=
by sorry

end valid_numbers_l125_12526


namespace cube_sum_of_sum_and_product_l125_12537

theorem cube_sum_of_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 11) : x^3 + y^3 = 670 := by
  sorry

end cube_sum_of_sum_and_product_l125_12537


namespace least_positive_multiple_least_positive_multiple_when_x_24_l125_12592

theorem least_positive_multiple (x y : ℤ) : ∃ (k : ℤ), k > 0 ∧ k * (x + 16 * y) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (x + 16 * y) = n * 8) → m ≥ k :=
  by sorry

theorem least_positive_multiple_when_x_24 : ∃ (k : ℤ), k > 0 ∧ k * (24 + 16 * (-1)) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (24 + 16 * (-1)) = n * 8) → m ≥ k :=
  by sorry

end least_positive_multiple_least_positive_multiple_when_x_24_l125_12592


namespace greatest_multiple_of_5_and_6_less_than_1000_l125_12514

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ n : ℕ, n = 990 ∧
    5 ∣ n ∧
    6 ∣ n ∧
    n < 1000 ∧
    ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l125_12514


namespace certain_amount_proof_l125_12582

theorem certain_amount_proof : 
  let x : ℝ := 10
  let percentage_of_500 : ℝ := 0.05 * 500
  let percentage_of_x : ℝ := 0.5 * x
  percentage_of_500 - percentage_of_x = 20 := by
sorry

end certain_amount_proof_l125_12582


namespace greatest_n_for_perfect_square_product_l125_12572

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_n_for_perfect_square_product (n : ℕ) : 
  n ≤ 2010 →
  (∀ k : ℕ, k > n → k ≤ 2010 → ¬is_perfect_square ((sum_squares k) * (sum_squares (2*k) - sum_squares k))) →
  is_perfect_square ((sum_squares n) * (sum_squares (2*n) - sum_squares n)) →
  n = 1935 := by sorry

end greatest_n_for_perfect_square_product_l125_12572


namespace ac_negative_l125_12542

theorem ac_negative (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : a / b + c / d = (a + c) / (b + d)) : a * c < 0 := by
  sorry

end ac_negative_l125_12542


namespace system_of_equations_solution_l125_12509

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  -- Proof goes here
  sorry

end system_of_equations_solution_l125_12509


namespace tangent_line_curve_equivalence_l125_12536

theorem tangent_line_curve_equivalence 
  (α β m n : ℝ) 
  (h_pos_α : α > 0) 
  (h_pos_β : β > 0) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_relation : 1 / α + 1 / β = 1) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    (m * x + n * y = 1) ∧ 
    (x ^ α + y ^ α = 1) ∧
    (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' ^ α + y' ^ α = 1 → m * x' + n * y' ≥ 1))
  ↔ 
  (m ^ β + n ^ β = 1) :=
sorry

end tangent_line_curve_equivalence_l125_12536


namespace crayons_per_child_l125_12559

/-- Given a group of children with crayons, prove that each child has 12 crayons. -/
theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 18) 
  (h2 : total_crayons = 216) : 
  total_crayons / total_children = 12 := by
  sorry

#check crayons_per_child

end crayons_per_child_l125_12559


namespace three_y_squared_l125_12569

theorem three_y_squared (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end three_y_squared_l125_12569


namespace tan_theta_minus_pi_fourth_l125_12511

theorem tan_theta_minus_pi_fourth (θ : Real) : 
  (∃ (x y : Real), x = 2 ∧ y = 3 ∧ Real.tan θ = y / x) → 
  Real.tan (θ - Real.pi / 4) = 1 / 5 := by
sorry

end tan_theta_minus_pi_fourth_l125_12511


namespace domino_placement_theorem_l125_12503

/-- Represents a 6x6 chessboard -/
def Chessboard : Type := Fin 6 × Fin 6

/-- Represents a domino placement on the chessboard -/
def Domino : Type := Chessboard × Chessboard

/-- Check if two squares are adjacent -/
def adjacent (s1 s2 : Chessboard) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Check if a domino placement is valid -/
def valid_domino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- The main theorem -/
theorem domino_placement_theorem
  (dominos : Finset Domino)
  (h1 : dominos.card = 11)
  (h2 : ∀ d ∈ dominos, valid_domino d)
  (h3 : ∀ s1 s2 : Chessboard, s1 ≠ s2 →
        (∃ d ∈ dominos, d.1 = s1 ∨ d.2 = s1) →
        (∃ d ∈ dominos, d.1 = s2 ∨ d.2 = s2) →
        s1 ≠ s2) :
  ∃ s1 s2 : Chessboard, adjacent s1 s2 ∧
    (∀ d ∈ dominos, d.1 ≠ s1 ∧ d.2 ≠ s1 ∧ d.1 ≠ s2 ∧ d.2 ≠ s2) :=
by sorry

end domino_placement_theorem_l125_12503


namespace owlHootsPerMinute_l125_12575

/-- The number of hoot sounds one barnyard owl makes per minute, given that 3 owls together make 5 less than 20 hoots per minute. -/
def owlHoots : ℕ :=
  let totalHoots : ℕ := 20 - 5
  let numOwls : ℕ := 3
  totalHoots / numOwls

/-- Theorem stating that one barnyard owl makes 5 hoot sounds per minute under the given conditions. -/
theorem owlHootsPerMinute : owlHoots = 5 := by
  sorry

end owlHootsPerMinute_l125_12575


namespace equation_solutions_l125_12564

/-- The equation has solutions when the parameter a is greater than 1 -/
def has_solution (a : ℝ) : Prop :=
  a > 1

/-- The solutions of the equation for a given parameter a -/
def solutions (a : ℝ) : Set ℝ :=
  if a > 2 then { (1 - a) / a, -1, 1 - a }
  else if a = 2 then { -1, -1/2 }
  else if 1 < a ∧ a < 2 then { (1 - a) / a, -1, 1 - a }
  else ∅

/-- The main theorem stating that the equation has solutions for a > 1 
    and providing these solutions -/
theorem equation_solutions (a : ℝ) :
  has_solution a →
  ∃ x : ℝ, x ∈ solutions a ∧
    (2 - 2 * a * (x + 1)) / (|x| - x) = Real.sqrt (1 - a - a * x) :=
by
  sorry


end equation_solutions_l125_12564


namespace thomas_salary_l125_12520

/-- Given the average salaries of two groups, prove Thomas's salary --/
theorem thomas_salary (raj roshan thomas : ℕ) : 
  (raj + roshan) / 2 = 4000 →
  (raj + roshan + thomas) / 3 = 5000 →
  thomas = 7000 := by
sorry

end thomas_salary_l125_12520


namespace trailing_zeros_of_999999999996_squared_l125_12523

/-- The number of trailing zeros in 999,999,999,996^2 is 11 -/
theorem trailing_zeros_of_999999999996_squared : 
  (999999999996 : ℕ)^2 % 10^12 = 16 := by sorry

end trailing_zeros_of_999999999996_squared_l125_12523


namespace perpendicular_bisector_c_value_l125_12548

/-- Given that the line x + y = c is the perpendicular bisector of the line segment
    from (2, 4) to (6, 8), prove that c = 10. -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ 
    ((x - 4)^2 + (y - 6)^2 = 8) ∧ 
    ((x - 2) * (8 - 4) = (y - 4) * (6 - 2))) →
  c = 10 := by
sorry

end perpendicular_bisector_c_value_l125_12548


namespace sum_value_l125_12580

theorem sum_value (a b : ℝ) (h1 : |a| = 1) (h2 : b = -2) : 
  a + b = -3 ∨ a + b = -1 := by
sorry

end sum_value_l125_12580


namespace jules_dog_walking_rate_l125_12571

/-- Proves that Jules charges $1.25 per block for dog walking -/
theorem jules_dog_walking_rate :
  let vacation_cost : ℚ := 1000
  let family_members : ℕ := 5
  let start_fee : ℚ := 2
  let dogs_walked : ℕ := 20
  let total_blocks : ℕ := 128
  let individual_contribution := vacation_cost / family_members
  let total_start_fees := start_fee * dogs_walked
  let remaining_to_earn := individual_contribution - total_start_fees
  let rate_per_block := remaining_to_earn / total_blocks
  rate_per_block = 1.25 := by
sorry


end jules_dog_walking_rate_l125_12571


namespace fourth_root_of_256000000_l125_12588

theorem fourth_root_of_256000000 : Real.sqrt (Real.sqrt 256000000) = 40 := by
  sorry

end fourth_root_of_256000000_l125_12588


namespace paul_min_correct_answers_l125_12594

def min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (unanswered_points : ℕ) (attempted : ℕ) (min_score : ℕ) : ℕ :=
  let unanswered := total_questions - attempted
  let unanswered_score := unanswered * unanswered_points
  let required_attempted_score := min_score - unanswered_score
  ((required_attempted_score + incorrect_points * attempted - 1) / (correct_points + incorrect_points)) + 1

theorem paul_min_correct_answers :
  min_correct_answers 25 7 2 2 18 90 = 13 := by
  sorry

end paul_min_correct_answers_l125_12594


namespace prob_heart_or_king_is_31_52_l125_12522

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (hearts : ℕ := 13)
  (kings : ℕ := 4)
  (king_of_hearts : ℕ := 1)

/-- The probability of drawing at least one heart or king when drawing two cards without replacement -/
def prob_heart_or_king (d : Deck) : ℚ :=
  1 - (d.total_cards - (d.hearts + d.kings - d.king_of_hearts)) * (d.total_cards - 1 - (d.hearts + d.kings - d.king_of_hearts)) /
      (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing at least one heart or king is 31/52 -/
theorem prob_heart_or_king_is_31_52 (d : Deck) :
  prob_heart_or_king d = 31 / 52 := by
  sorry

end prob_heart_or_king_is_31_52_l125_12522


namespace train_distance_problem_l125_12556

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 40)
  (h3 : distance_diff = 100) :
  let time := distance_diff / (speed1 - speed2)
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let total_distance := distance1 + distance2
  total_distance = 900 := by sorry

end train_distance_problem_l125_12556


namespace y_intercept_of_l_l125_12566

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- The line l is defined by the equation y = 3x - 2 -/
def l (x : ℝ) : ℝ := 3 * x - 2

theorem y_intercept_of_l :
  y_intercept 3 (-2) = -2 := by sorry

end y_intercept_of_l_l125_12566


namespace curve_C_bound_expression_l125_12555

theorem curve_C_bound_expression (x y : ℝ) :
  4 * x^2 + y^2 = 16 →
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end curve_C_bound_expression_l125_12555


namespace min_value_reciprocal_sum_l125_12519

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end min_value_reciprocal_sum_l125_12519
