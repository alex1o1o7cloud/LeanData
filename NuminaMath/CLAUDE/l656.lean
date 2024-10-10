import Mathlib

namespace quadratic_exponent_implies_m_eq_two_l656_65657

/-- A function is quadratic if it can be expressed as ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem: If y = (m+2)x^(m²-2) is quadratic, then m = 2 -/
theorem quadratic_exponent_implies_m_eq_two (m : ℝ) :
  IsQuadratic (fun x ↦ (m + 2) * x^(m^2 - 2)) → m = 2 := by
  sorry

end quadratic_exponent_implies_m_eq_two_l656_65657


namespace honor_roll_fraction_l656_65626

theorem honor_roll_fraction (total_students : ℝ) (female_students : ℝ) (male_students : ℝ) 
  (female_honor : ℝ) (male_honor : ℝ) :
  female_students = (2 / 5) * total_students →
  male_students = (3 / 5) * total_students →
  female_honor = (5 / 6) * female_students →
  male_honor = (2 / 3) * male_students →
  (female_honor + male_honor) / total_students = 11 / 15 := by
sorry

end honor_roll_fraction_l656_65626


namespace smallest_n_for_183_div_11_l656_65643

theorem smallest_n_for_183_div_11 :
  ∃! n : ℕ, (183 + n) % 11 = 0 ∧ ∀ m : ℕ, m < n → (183 + m) % 11 ≠ 0 :=
by
  -- The proof goes here
  sorry

end smallest_n_for_183_div_11_l656_65643


namespace inequality_solution_l656_65620

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + 
   (Real.sqrt (Real.sqrt (3 * x^2 + 6 * x + 7) + 
               Real.sqrt (5 * x^2 + 10 * x + 14) + 
               x^2 + 2 * x - 4))^(1/4) ≤ 18 - 8 * x) ↔ 
  x = -1 := by
sorry

end inequality_solution_l656_65620


namespace even_function_implies_a_equals_one_l656_65630

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The specific function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + (a - 1) * x + a

theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
sorry

end even_function_implies_a_equals_one_l656_65630


namespace negation_of_proposition_negation_of_square_le_power_two_l656_65618

theorem negation_of_proposition (p : ℕ → Prop) :
  (¬∀ n : ℕ, p n) ↔ (∃ n : ℕ, ¬p n) := by sorry

theorem negation_of_square_le_power_two :
  (¬∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by sorry

end negation_of_proposition_negation_of_square_le_power_two_l656_65618


namespace increasing_quadratic_iff_l656_65634

/-- A function f is increasing on an interval [x0, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ x y, x0 ≤ x → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem increasing_quadratic_iff (a : ℝ) :
  IncreasingOn (f a) 4 ↔ a ≥ -3 :=
sorry

end increasing_quadratic_iff_l656_65634


namespace power_multiplication_l656_65615

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l656_65615


namespace parallel_vectors_characterization_l656_65696

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ - x₂ * y₁ = 0

/-- The proposed condition for parallel vectors -/
def proposed_condition (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ = x₂ * y₁

theorem parallel_vectors_characterization (a b : ℝ × ℝ) :
  (are_parallel a b ↔ proposed_condition a b) ∧
  (∃ a b : ℝ × ℝ, are_parallel a b ≠ proposed_condition a b) :=
sorry

end parallel_vectors_characterization_l656_65696


namespace orangeade_pricing_l656_65635

/-- Represents the amount of orange juice used each day -/
def orange_juice : ℝ := sorry

/-- Represents the amount of water used on the first day -/
def water : ℝ := sorry

/-- The price per glass on the first day -/
def price_day1 : ℝ := 0.60

/-- The price per glass on the third day -/
def price_day3 : ℝ := sorry

/-- The volume of orangeade on the first day -/
def volume_day1 : ℝ := orange_juice + water

/-- The volume of orangeade on the second day -/
def volume_day2 : ℝ := orange_juice + 2 * water

/-- The volume of orangeade on the third day -/
def volume_day3 : ℝ := orange_juice + 3 * water

theorem orangeade_pricing :
  (orange_juice > 0) →
  (water > 0) →
  (orange_juice = water) →
  (price_day1 * volume_day1 = price_day3 * volume_day3) →
  (price_day3 = price_day1 / 2) := by
  sorry

end orangeade_pricing_l656_65635


namespace pentagon_regular_if_equal_altitudes_and_medians_l656_65613

/-- A pentagon is a polygon with five vertices and five edges. -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- An altitude of a pentagon is the perpendicular drop from a vertex to the opposite side. -/
def altitude (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A median of a pentagon is the line joining a vertex to the midpoint of the opposite side. -/
def median (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A pentagon is regular if all its sides are equal and all its interior angles are equal. -/
def is_regular (p : Pentagon) : Prop := sorry

/-- Theorem: If all altitudes and all medians of a pentagon have the same length, then the pentagon is regular. -/
theorem pentagon_regular_if_equal_altitudes_and_medians (p : Pentagon) 
  (h1 : ∀ i j : Fin 5, altitude p i = altitude p j) 
  (h2 : ∀ i j : Fin 5, median p i = median p j) : 
  is_regular p := by sorry

end pentagon_regular_if_equal_altitudes_and_medians_l656_65613


namespace f_at_negative_two_l656_65616

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_at_negative_two : f (-2) = -75 := by
  sorry

end f_at_negative_two_l656_65616


namespace right_triangle_hypotenuse_l656_65612

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 2500 →  -- Given condition
  c = 25 * Real.sqrt 2 := by
sorry

end right_triangle_hypotenuse_l656_65612


namespace sqrt_sum_zero_implies_y_minus_x_l656_65672

theorem sqrt_sum_zero_implies_y_minus_x (x y : ℝ) :
  Real.sqrt (2 * x + y) + Real.sqrt (x^2 - 9) = 0 →
  (y - x = -9 ∨ y - x = 9) :=
by sorry

end sqrt_sum_zero_implies_y_minus_x_l656_65672


namespace triangle_angle_sum_l656_65679

theorem triangle_angle_sum (a b c : ℝ) : 
  b = 2 * a →
  c = a - 40 →
  a + b + c = 180 →
  a + c = 70 := by
sorry

end triangle_angle_sum_l656_65679


namespace cube_edge_length_l656_65638

/-- Given three cubes with edge lengths 6, 10, and x, when melted together to form a new cube
    with edge length 12, prove that x = 8 -/
theorem cube_edge_length (x : ℝ) : x > 0 → 6^3 + 10^3 + x^3 = 12^3 → x = 8 := by sorry

end cube_edge_length_l656_65638


namespace equation_has_real_root_l656_65689

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 2) * (x - 3) := by
  sorry

end equation_has_real_root_l656_65689


namespace remainder_divisibility_l656_65691

theorem remainder_divisibility (x y z p : ℕ) : 
  0 < x → 0 < y → 0 < z →  -- x, y, z are positive integers
  Nat.Prime p →            -- p is prime
  x < y → y < z → z < p →  -- x < y < z < p
  x^3 % p = y^3 % p →      -- x^3 and y^3 have the same remainder mod p
  y^3 % p = z^3 % p →      -- y^3 and z^3 have the same remainder mod p
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
  sorry

end remainder_divisibility_l656_65691


namespace rush_delivery_percentage_l656_65611

theorem rush_delivery_percentage (original_cost : ℝ) (rush_cost_per_type : ℝ) (num_types : ℕ) :
  original_cost = 40 →
  rush_cost_per_type = 13 →
  num_types = 4 →
  (rush_cost_per_type * num_types - original_cost) / original_cost * 100 = 30 := by
  sorry

end rush_delivery_percentage_l656_65611


namespace expression_evaluation_l656_65650

/-- Given a = -2 and b = -1/2, prove that 2(3a^2 - 4ab) - [a^2 - 3(2a + 3ab)] evaluates to 9 -/
theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := -1/2
  2 * (3 * a^2 - 4 * a * b) - (a^2 - 3 * (2 * a + 3 * a * b)) = 9 := by
sorry


end expression_evaluation_l656_65650


namespace sine_zeros_range_l656_65699

open Real

theorem sine_zeros_range (ω : ℝ) : 
  (ω > 0) → 
  (∃! (z₁ z₂ : ℝ), 0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ ≤ π/4 ∧ 
    sin (2*ω*z₁ - π/6) = 0 ∧ sin (2*ω*z₂ - π/6) = 0 ∧
    ∀ z, 0 ≤ z ∧ z ≤ π/4 ∧ sin (2*ω*z - π/6) = 0 → z = z₁ ∨ z = z₂) ↔ 
  (7/3 ≤ ω ∧ ω < 13/3) :=
by sorry

end sine_zeros_range_l656_65699


namespace power_product_squared_l656_65675

theorem power_product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end power_product_squared_l656_65675


namespace stacy_height_proof_l656_65610

def stacy_height_problem (last_year_height : ℕ) (brother_growth : ℕ) (growth_difference : ℕ) : Prop :=
  let stacy_growth : ℕ := brother_growth + growth_difference
  let current_height : ℕ := last_year_height + stacy_growth
  current_height = 57

theorem stacy_height_proof :
  stacy_height_problem 50 1 6 := by
  sorry

end stacy_height_proof_l656_65610


namespace root_product_expression_l656_65655

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 3 = 0) →
  (β^2 - 2*p*β + 3 = 0) →
  (γ^2 - 3*q*γ + 4 = 0) →
  (δ^2 - 3*q*δ + 4 = 0) →
  (α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2*p - 3*q)^2 := by sorry

end root_product_expression_l656_65655


namespace expression_simplification_l656_65693

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)) = 1 := by
  sorry

end expression_simplification_l656_65693


namespace right_triangle_area_l656_65641

/-- The area of a right triangle with hypotenuse 5√2 and one leg 5 is 12.5 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) 
  (h2 : c = 5 * Real.sqrt 2) (h3 : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end right_triangle_area_l656_65641


namespace smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l656_65614

theorem smallest_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → n < 11 → ¬(Nat.gcd n 2100 = 1) :=
sorry

theorem eleven_has_inverse_mod_2100 : Nat.gcd 11 2100 = 1 :=
sorry

theorem eleven_is_smallest_with_inverse_mod_2100 : 
  ∀ n : ℕ, n > 1 → Nat.gcd n 2100 = 1 → n ≥ 11 :=
sorry

end smallest_inverse_mod_2100_eleven_has_inverse_mod_2100_eleven_is_smallest_with_inverse_mod_2100_l656_65614


namespace championship_outcomes_l656_65658

theorem championship_outcomes (num_students : ℕ) (num_events : ℕ) : 
  num_students = 4 → num_events = 3 → (num_students ^ num_events : ℕ) = 64 := by
  sorry

#check championship_outcomes

end championship_outcomes_l656_65658


namespace oranges_in_box_l656_65649

/-- Given an initial number of oranges in a box and a number of oranges added,
    the final number of oranges in the box is equal to the sum of the initial number and the added number. -/
theorem oranges_in_box (initial : ℝ) (added : ℝ) :
  initial + added = 90 :=
by sorry

end oranges_in_box_l656_65649


namespace limit_exists_l656_65624

/-- Prove the existence of δ(ε) for the limit of (5x^2 - 24x - 5) / (x - 5) as x approaches 5 -/
theorem limit_exists (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ 5 → |x - 5| < δ →
    |(5 * x^2 - 24 * x - 5) / (x - 5) - 26| < ε := by
  sorry

end limit_exists_l656_65624


namespace angle_through_point_l656_65601

theorem angle_through_point (α : Real) : 
  0 ≤ α → α < 2 * Real.pi → 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
sorry

end angle_through_point_l656_65601


namespace probability_ray_in_angle_l656_65682

/-- The probability of a randomly drawn ray falling within a 60-degree angle in a circular region is 1/6. -/
theorem probability_ray_in_angle (angle : ℝ) (total_angle : ℝ) : 
  angle = 60 → total_angle = 360 → angle / total_angle = 1 / 6 := by
  sorry

end probability_ray_in_angle_l656_65682


namespace cubic_roots_sum_l656_65608

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 27 * p - 6 = 0) →
  (3 * q^3 - 9 * q^2 + 27 * q - 6 = 0) →
  (3 * r^3 - 9 * r^2 + 27 * r - 6 = 0) →
  (p + q + r = 3) →
  (p * q + q * r + r * p = 9) →
  (p * q * r = 2) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := by
  sorry

end cubic_roots_sum_l656_65608


namespace system_one_solution_system_two_solution_l656_65609

-- System 1
theorem system_one_solution (x y : ℝ) : 
  (4 * x - 2 * y = 14) ∧ (3 * x + 2 * y = 7) → x = 3 ∧ y = -1 :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (y = x + 1) ∧ (2 * x + y = 10) → x = 3 ∧ y = 4 :=
by sorry

end system_one_solution_system_two_solution_l656_65609


namespace fence_cost_l656_65600

-- Define the side lengths of the pentagon
def side1 : ℕ := 10
def side2 : ℕ := 14
def side3 : ℕ := 12
def side4 : ℕ := 8
def side5 : ℕ := 6

-- Define the prices per foot for each group of sides
def price1 : ℕ := 45  -- Price for first two sides
def price2 : ℕ := 55  -- Price for third and fourth sides
def price3 : ℕ := 60  -- Price for last side

-- Define the total cost function
def totalCost : ℕ := 
  side1 * price1 + side2 * price1 + 
  side3 * price2 + side4 * price2 + 
  side5 * price3

-- Theorem stating that the total cost is 2540
theorem fence_cost : totalCost = 2540 := by
  sorry

end fence_cost_l656_65600


namespace inequality_proof_l656_65652

theorem inequality_proof (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt (a * x / (a * x + 8)) ≤ 2 := by
  sorry

end inequality_proof_l656_65652


namespace light_bulbs_configuration_equals_59_l656_65663

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the light bulb configuration -/
def light_bulbs : List Bool := [true, true, true, false, true, true]

theorem light_bulbs_configuration_equals_59 :
  binary_to_decimal light_bulbs = 59 := by
  sorry

end light_bulbs_configuration_equals_59_l656_65663


namespace total_pools_l656_65628

def arkAndAthleticPools : ℕ := 200
def poolSupplyMultiplier : ℕ := 3

theorem total_pools : arkAndAthleticPools + poolSupplyMultiplier * arkAndAthleticPools = 800 := by
  sorry

end total_pools_l656_65628


namespace stratified_sample_size_l656_65677

/-- Represents the proportion of a population group -/
structure PopulationProportion where
  value : ℚ
  nonneg : 0 ≤ value

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_size : ℕ
  middle_aged_size : ℕ
  middle_aged_size_le_total : middle_aged_size ≤ total_size

/-- Given population proportions and a stratified sample, proves the total sample size -/
theorem stratified_sample_size 
  (elderly : PopulationProportion)
  (middle_aged : PopulationProportion)
  (young : PopulationProportion)
  (sample : StratifiedSample)
  (h1 : elderly.value + middle_aged.value + young.value = 1)
  (h2 : elderly.value = 2 / 10)
  (h3 : middle_aged.value = 3 / 10)
  (h4 : young.value = 5 / 10)
  (h5 : sample.middle_aged_size = 12) :
  sample.total_size = 40 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l656_65677


namespace quadratic_equation_solutions_l656_65647

theorem quadratic_equation_solutions (b c x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁ ≠ x₂) →
  (|x₁ - x₂| = 1) →
  (|b - c| = 1) →
  ((b = -1 ∧ c = 0) ∨ (b = 5 ∧ c = 6) ∨ (b = 1 ∧ c = 0) ∨ (b = 3 ∧ c = 2)) :=
by sorry

end quadratic_equation_solutions_l656_65647


namespace sun_radius_scientific_notation_l656_65631

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The radius of the Sun in meters -/
def sun_radius : ℝ := 696000000

/-- Converts a real number to scientific notation -/
noncomputable def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem sun_radius_scientific_notation :
  to_scientific_notation sun_radius = ScientificNotation.mk 6.96 8 sorry := by
  sorry

end sun_radius_scientific_notation_l656_65631


namespace dumplings_eaten_l656_65632

theorem dumplings_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 14 → remaining = 7 → eaten = initial - remaining :=
by sorry

end dumplings_eaten_l656_65632


namespace floral_shop_sales_theorem_l656_65690

/-- Represents the sales and prices of bouquets for a floral shop over three days -/
structure FloralShopSales where
  /-- Number of rose bouquets sold on Monday -/
  rose_monday : ℕ
  /-- Number of lily bouquets sold on Monday -/
  lily_monday : ℕ
  /-- Number of orchid bouquets sold on Monday -/
  orchid_monday : ℕ
  /-- Price of rose bouquets on Monday -/
  rose_price_monday : ℕ
  /-- Price of lily bouquets on Monday -/
  lily_price_monday : ℕ
  /-- Price of orchid bouquets on Monday -/
  orchid_price_monday : ℕ
  /-- Price of rose bouquets on Tuesday -/
  rose_price_tuesday : ℕ
  /-- Price of lily bouquets on Tuesday -/
  lily_price_tuesday : ℕ
  /-- Price of orchid bouquets on Tuesday -/
  orchid_price_tuesday : ℕ
  /-- Price of rose bouquets on Wednesday -/
  rose_price_wednesday : ℕ
  /-- Price of lily bouquets on Wednesday -/
  lily_price_wednesday : ℕ
  /-- Price of orchid bouquets on Wednesday -/
  orchid_price_wednesday : ℕ

/-- Calculates the total number and value of bouquets sold over three days -/
def calculate_total_sales (sales : FloralShopSales) : ℕ × ℕ :=
  let rose_tuesday := 3 * sales.rose_monday
  let lily_tuesday := 2 * sales.lily_monday
  let orchid_tuesday := sales.orchid_monday / 2
  let rose_wednesday := rose_tuesday / 3
  let lily_wednesday := lily_tuesday / 4
  let orchid_wednesday := (2 * orchid_tuesday) / 3
  
  let total_roses := sales.rose_monday + rose_tuesday + rose_wednesday
  let total_lilies := sales.lily_monday + lily_tuesday + lily_wednesday
  let total_orchids := sales.orchid_monday + orchid_tuesday + orchid_wednesday
  
  let total_bouquets := total_roses + total_lilies + total_orchids
  
  let rose_value := sales.rose_monday * sales.rose_price_monday + 
                    rose_tuesday * sales.rose_price_tuesday + 
                    rose_wednesday * sales.rose_price_wednesday
  let lily_value := sales.lily_monday * sales.lily_price_monday + 
                    lily_tuesday * sales.lily_price_tuesday + 
                    lily_wednesday * sales.lily_price_wednesday
  let orchid_value := sales.orchid_monday * sales.orchid_price_monday + 
                      orchid_tuesday * sales.orchid_price_tuesday + 
                      orchid_wednesday * sales.orchid_price_wednesday
  
  let total_value := rose_value + lily_value + orchid_value
  
  (total_bouquets, total_value)

theorem floral_shop_sales_theorem (sales : FloralShopSales) 
  (h1 : sales.rose_monday = 12)
  (h2 : sales.lily_monday = 8)
  (h3 : sales.orchid_monday = 6)
  (h4 : sales.rose_price_monday = 10)
  (h5 : sales.lily_price_monday = 15)
  (h6 : sales.orchid_price_monday = 20)
  (h7 : sales.rose_price_tuesday = 12)
  (h8 : sales.lily_price_tuesday = 18)
  (h9 : sales.orchid_price_tuesday = 22)
  (h10 : sales.rose_price_wednesday = 8)
  (h11 : sales.lily_price_wednesday = 12)
  (h12 : sales.orchid_price_wednesday = 16) :
  calculate_total_sales sales = (99, 1322) := by
  sorry


end floral_shop_sales_theorem_l656_65690


namespace ContrapositiveDual_l656_65625

-- Define what it means for a number to be even
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k

-- The original proposition
def OriginalProposition : Prop :=
  ∀ a b : Int, IsEven a ∧ IsEven b → IsEven (a + b)

-- The contrapositive we want to prove
def Contrapositive : Prop :=
  ∀ a b : Int, ¬IsEven (a + b) → ¬(IsEven a ∧ IsEven b)

-- The theorem stating that the contrapositive is correct
theorem ContrapositiveDual : OriginalProposition ↔ Contrapositive := by
  sorry

end ContrapositiveDual_l656_65625


namespace solution_of_inequality1_solution_of_inequality2_l656_65684

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x > -1 ∧ x < 1}

-- Define the solution set for the second inequality
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_of_inequality1 :
  {x : ℝ | (2 * x) / (x + 1) < 1} = solutionSet1 := by sorry

-- Theorem for the second inequality
theorem solution_of_inequality2 (a : ℝ) :
  {x : ℝ | x^2 + (2 - a) * x - 2 * a ≥ 0} = solutionSet2 a := by sorry

end solution_of_inequality1_solution_of_inequality2_l656_65684


namespace vector_problem_l656_65659

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := ![m, 2]
def b : Fin 2 → ℝ := ![2, -3]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_problem (m : ℝ) :
  are_parallel (a m + b) (a m - b) → m = -4/3 := by
  sorry

end vector_problem_l656_65659


namespace password_letters_count_l656_65687

theorem password_letters_count : ∃ (n : ℕ), 
  (n ^ 4 : ℕ) - n * (n - 1) * (n - 2) * (n - 3) = 936 ∧ n = 6 := by
  sorry

end password_letters_count_l656_65687


namespace min_value_of_f_l656_65670

-- Define the function f(x)
def f (x : ℝ) : ℝ := 27 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 2 ∧
  (∀ y ∈ Set.Icc (-4 : ℝ) 2, f y ≥ f x) ∧
  f x = -54 := by
  sorry

end min_value_of_f_l656_65670


namespace no_allowable_formations_l656_65665

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem no_allowable_formations :
  ∀ s t : ℕ,
    s * t = 240 →
    is_prime s →
    8 ≤ t →
    t ≤ 30 →
    ¬∃ (s t : ℕ), s * t = 240 ∧ is_prime s ∧ 8 ≤ t ∧ t ≤ 30 :=
by
  sorry

#check no_allowable_formations

end no_allowable_formations_l656_65665


namespace inequality_proof_l656_65621

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end inequality_proof_l656_65621


namespace bernardo_wins_l656_65688

theorem bernardo_wins (N : ℕ) : N = 78 ↔ 
  N ∈ Finset.range 1000 ∧ 
  (∀ m : ℕ, m < N → m ∉ Finset.range 1000 ∨ 
    3 * m ≥ 1000 ∨ 
    3 * m + 75 ≥ 1000 ∨ 
    9 * m + 225 ≥ 1000 ∨ 
    9 * m + 300 < 1000) ∧
  3 * N < 1000 ∧
  3 * N + 75 < 1000 ∧
  9 * N + 225 < 1000 ∧
  9 * N + 300 ≥ 1000 := by
sorry

#eval (78 / 10) + (78 % 10)  -- Sum of digits of 78

end bernardo_wins_l656_65688


namespace quadratic_roots_to_coeff_difference_l656_65680

theorem quadratic_roots_to_coeff_difference (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ (x = -1/2 ∨ x = 1/3)) → 
  a - b = -10 := by
sorry

end quadratic_roots_to_coeff_difference_l656_65680


namespace second_boy_speed_l656_65697

/-- Given two boys walking in the same direction for 16 hours, with one boy walking at 5.5 kmph
    and ending up 32 km apart, prove that the speed of the second boy is 7.5 kmph. -/
theorem second_boy_speed (first_speed : ℝ) (time : ℝ) (distance : ℝ) (second_speed : ℝ) :
  first_speed = 5.5 →
  time = 16 →
  distance = 32 →
  distance = (second_speed - first_speed) * time →
  second_speed = 7.5 := by
  sorry

end second_boy_speed_l656_65697


namespace remainder_seventeen_power_sixtythree_mod_seven_l656_65681

theorem remainder_seventeen_power_sixtythree_mod_seven :
  17^63 % 7 = 6 := by
sorry

end remainder_seventeen_power_sixtythree_mod_seven_l656_65681


namespace parallel_angles_theorem_l656_65644

/-- Two angles with parallel sides --/
structure ParallelAngles where
  α : ℝ
  β : ℝ
  x : ℝ
  parallel : Bool
  α_eq : α = 2 * x + 10
  β_eq : β = 3 * x - 20

/-- The possible values for α in the parallel angles scenario --/
def possible_α_values (angles : ParallelAngles) : Set ℝ :=
  {70, 86}

/-- Theorem stating that the possible values for α are 70° or 86° --/
theorem parallel_angles_theorem (angles : ParallelAngles) :
  angles.α ∈ possible_α_values angles :=
by
  sorry


end parallel_angles_theorem_l656_65644


namespace jane_usable_a4_sheets_l656_65623

/-- Represents the different types of paper sheets -/
inductive SheetType
  | BrownA4
  | YellowA4
  | YellowA3
  | PinkA2

/-- Calculates the number of usable sheets given the total and damaged counts -/
def usableSheets (total : ℕ) (damaged : ℕ) : ℕ :=
  total - damaged + (damaged / 2)

/-- Theorem: Jane has 40 total usable A4 sheets for sketching -/
theorem jane_usable_a4_sheets :
  let brown_a4_total := 28
  let yellow_a4_total := 18
  let yellow_a3_total := 9
  let pink_a2_total := 10
  let brown_a4_damaged := 3
  let yellow_a4_damaged := 5
  let yellow_a3_damaged := 2
  let pink_a2_damaged := 2
  let brown_a4_usable := usableSheets brown_a4_total brown_a4_damaged
  let yellow_a4_usable := usableSheets yellow_a4_total yellow_a4_damaged
  brown_a4_usable + yellow_a4_usable = 40 := by
    sorry


end jane_usable_a4_sheets_l656_65623


namespace painting_time_l656_65651

theorem painting_time (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : 
  total_rooms = 10 → time_per_room = 8 → painted_rooms = 8 → 
  (total_rooms - painted_rooms) * time_per_room = 16 := by
sorry

end painting_time_l656_65651


namespace crabapple_sequences_l656_65695

/-- The number of students in Mrs. Crabapple's British Literature class -/
def num_students : ℕ := 13

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of possible sequences of crabapple recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 2197 :=
sorry

end crabapple_sequences_l656_65695


namespace log_equation_solution_l656_65673

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * Real.log t / Real.log 3 = Real.log (4 * t) / Real.log 3 + 2 → t = 6 :=
by
  sorry

end log_equation_solution_l656_65673


namespace jacob_insects_compared_to_dean_l656_65661

theorem jacob_insects_compared_to_dean :
  ∀ (angela_insects jacob_insects dean_insects : ℕ),
    angela_insects = 75 →
    dean_insects = 30 →
    angela_insects * 2 = jacob_insects →
    jacob_insects / dean_insects = 5 :=
by
  sorry

end jacob_insects_compared_to_dean_l656_65661


namespace work_completion_time_proportional_aartis_triple_work_time_l656_65666

/-- If a person can complete a piece of work in a certain number of days,
    then the time to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional
  (days_for_single_work : ℕ) (work_multiple : ℕ) :
  let days_for_multiple_work := days_for_single_work * work_multiple
  days_for_multiple_work = days_for_single_work * work_multiple :=
by sorry

/-- Aarti's work completion time for triple work -/
theorem aartis_triple_work_time :
  let days_for_single_work := 9
  let work_multiple := 3
  let days_for_triple_work := days_for_single_work * work_multiple
  days_for_triple_work = 27 :=
by sorry

end work_completion_time_proportional_aartis_triple_work_time_l656_65666


namespace trivia_team_distribution_l656_65640

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) :
  total_students = 65 →
  not_picked = 17 →
  num_groups = 8 →
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end trivia_team_distribution_l656_65640


namespace negation_of_universal_proposition_l656_65629

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end negation_of_universal_proposition_l656_65629


namespace dhoni_leftover_earnings_l656_65633

theorem dhoni_leftover_earnings (total_earnings rent_percentage dishwasher_discount : ℝ) :
  rent_percentage = 40 →
  dishwasher_discount = 20 →
  let dishwasher_percentage := rent_percentage - (dishwasher_discount / 100) * rent_percentage
  let total_spent_percentage := rent_percentage + dishwasher_percentage
  let leftover_percentage := 100 - total_spent_percentage
  leftover_percentage = 28 :=
by sorry

end dhoni_leftover_earnings_l656_65633


namespace rectangular_plot_length_difference_l656_65674

/-- Proves that for a rectangular plot with given conditions, the length is 20 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 60 ∧ 
  length > breadth ∧ 
  2 * (length + breadth) * 26.5 = 5300 → 
  length - breadth = 20 := by
  sorry

end rectangular_plot_length_difference_l656_65674


namespace power_of_two_mod_nine_periodic_l656_65627

/-- The sequence of remainders when powers of 2 are divided by 9 is periodic with period 6 -/
theorem power_of_two_mod_nine_periodic :
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), (2^(n + p) : ℕ) % 9 = (2^n : ℕ) % 9 ∧ p = 6 := by
  sorry

end power_of_two_mod_nine_periodic_l656_65627


namespace days_missed_by_mike_and_sarah_l656_65639

/-- Given the number of days missed by Vanessa, Mike, and Sarah, prove that Mike and Sarah missed 12 days together. -/
theorem days_missed_by_mike_and_sarah
  (total_days : ℕ)
  (vanessa_mike_days : ℕ)
  (vanessa_days : ℕ)
  (h1 : total_days = 17)
  (h2 : vanessa_mike_days = 14)
  (h3 : vanessa_days = 5)
  : ∃ (mike_days sarah_days : ℕ),
    mike_days + sarah_days = 12 ∧
    vanessa_days + mike_days + sarah_days = total_days ∧
    vanessa_days + mike_days = vanessa_mike_days :=
by
  sorry


end days_missed_by_mike_and_sarah_l656_65639


namespace king_queen_prob_l656_65604

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Queens in a standard deck -/
def numQueens : ℕ := 4

/-- Calculates the probability of drawing a King followed by a Queen from a standard deck -/
def probKingQueen (deck : StandardDeck) : ℚ :=
  (numKings * numQueens : ℚ) / (deckSize * (deckSize - 1))

/-- Theorem stating that the probability of drawing a King followed by a Queen is 4/663 -/
theorem king_queen_prob : 
  ∀ (deck : StandardDeck), probKingQueen deck = 4 / 663 := by
  sorry

end king_queen_prob_l656_65604


namespace not_all_perfect_squares_l656_65606

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by
  sorry

end not_all_perfect_squares_l656_65606


namespace arithmetic_operations_l656_65636

theorem arithmetic_operations : 
  (12 - (-18) + (-7) - 20 = 3) ∧ 
  (-4 / (1/2) * 8 = -64) := by
sorry

end arithmetic_operations_l656_65636


namespace greatest_common_length_l656_65671

theorem greatest_common_length (a b c : ℕ) (ha : a = 48) (hb : b = 64) (hc : c = 80) :
  Nat.gcd a (Nat.gcd b c) = 16 := by
  sorry

end greatest_common_length_l656_65671


namespace park_visitors_l656_65602

theorem park_visitors (bike_riders : ℕ) (hikers : ℕ) : 
  bike_riders = 249 →
  hikers = bike_riders + 178 →
  bike_riders + hikers = 676 :=
by sorry

end park_visitors_l656_65602


namespace point_parameters_l656_65683

/-- Parametric equation of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The given line -/
def givenLine : ParametricLine :=
  { x := λ t => 1 + 2 * t,
    y := λ t => 2 - 3 * t }

/-- Point A -/
def pointA : Point :=
  { x := 1,
    y := 2 }

/-- Point B -/
def pointB : Point :=
  { x := -1,
    y := 5 }

/-- Theorem stating that the parameters for points A and B are 0 and -1 respectively -/
theorem point_parameters : 
  (∃ t : ℝ, givenLine.x t = pointA.x ∧ givenLine.y t = pointA.y ∧ t = 0) ∧
  (∃ t : ℝ, givenLine.x t = pointB.x ∧ givenLine.y t = pointB.y ∧ t = -1) :=
by sorry

end point_parameters_l656_65683


namespace computation_proof_l656_65694

theorem computation_proof : 45 * (28 + 72) + 55 * 45 = 6975 := by
  sorry

end computation_proof_l656_65694


namespace smallest_integer_with_divisibility_condition_l656_65669

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_integer_with_divisibility_condition : 
  ∀ n : ℕ, n > 0 →
  (∀ i ∈ Finset.range 31, i ≠ 23 ∧ i ≠ 24 → is_divisible n i) →
  ¬(is_divisible n 23) →
  ¬(is_divisible n 24) →
  n ≥ 2230928700 :=
sorry

end smallest_integer_with_divisibility_condition_l656_65669


namespace sum_of_first_fifteen_multiples_of_eight_l656_65642

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n positive multiples of m -/
def sum_of_multiples (m n : ℕ) : ℕ := m * sum_of_naturals n

theorem sum_of_first_fifteen_multiples_of_eight :
  sum_of_multiples 8 15 = 960 := by
  sorry

end sum_of_first_fifteen_multiples_of_eight_l656_65642


namespace cube_volume_from_paper_l656_65653

theorem cube_volume_from_paper (paper_length paper_width : ℝ) 
  (h1 : paper_length = 48)
  (h2 : paper_width = 72)
  (h3 : 1 = 12) : -- 1 foot = 12 inches
  let paper_area := paper_length * paper_width
  let cube_face_area := paper_area / 6
  let cube_side_length := Real.sqrt cube_face_area
  let cube_side_length_feet := cube_side_length / 12
  cube_side_length_feet ^ 3 = 8 := by
sorry

end cube_volume_from_paper_l656_65653


namespace first_candle_triple_second_at_correct_time_l656_65646

/-- The time (in hours) when the first candle is three times the height of the second candle -/
def time_when_first_is_triple_second : ℚ := 40 / 11

/-- The initial height of both candles -/
def initial_height : ℚ := 1

/-- The time (in hours) it takes for the first candle to burn out completely -/
def first_candle_burnout_time : ℚ := 5

/-- The time (in hours) it takes for the second candle to burn out completely -/
def second_candle_burnout_time : ℚ := 4

/-- The height of the first candle at time t -/
def first_candle_height (t : ℚ) : ℚ := initial_height - (t / first_candle_burnout_time)

/-- The height of the second candle at time t -/
def second_candle_height (t : ℚ) : ℚ := initial_height - (t / second_candle_burnout_time)

theorem first_candle_triple_second_at_correct_time :
  first_candle_height time_when_first_is_triple_second = 
  3 * second_candle_height time_when_first_is_triple_second :=
sorry

end first_candle_triple_second_at_correct_time_l656_65646


namespace total_cds_l656_65637

theorem total_cds (a b : ℕ) : 
  (b + 6 = 2 * (a - 6)) →
  (a + 6 = b - 6) →
  a + b = 72 := by
sorry

end total_cds_l656_65637


namespace remaining_income_percentage_l656_65645

-- Define the percentages as fractions
def food_percent : ℚ := 35 / 100
def education_percent : ℚ := 25 / 100
def transportation_percent : ℚ := 15 / 100
def medical_percent : ℚ := 10 / 100
def rent_percent_of_remaining : ℚ := 80 / 100

-- Theorem statement
theorem remaining_income_percentage :
  let initial_expenses := food_percent + education_percent + transportation_percent + medical_percent
  let remaining_after_initial := 1 - initial_expenses
  let rent_expense := rent_percent_of_remaining * remaining_after_initial
  1 - (initial_expenses + rent_expense) = 3 / 100 := by
  sorry

end remaining_income_percentage_l656_65645


namespace smallest_k_inequality_l656_65664

theorem smallest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end smallest_k_inequality_l656_65664


namespace pentagon_perimeter_l656_65678

/-- The perimeter of pentagon ABCDE with given side lengths -/
theorem pentagon_perimeter (AB BC CD DE AE : ℝ) : 
  AB = 1 → BC = Real.sqrt 3 → CD = 2 → DE = Real.sqrt 5 → AE = Real.sqrt 13 →
  AB + BC + CD + DE + AE = 3 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 13 := by
  sorry

end pentagon_perimeter_l656_65678


namespace divisibility_condition_l656_65654

def M (n : ℤ) : Finset ℤ := {n, n + 1, n + 2, n + 3, n + 4}

def S (n : ℤ) : ℤ := (M n).sum (fun x => x^2)

def P (n : ℤ) : ℤ := (M n).prod (fun x => x^2)

theorem divisibility_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := by
  sorry

end divisibility_condition_l656_65654


namespace tan_product_equals_15_l656_65686

theorem tan_product_equals_15 : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end tan_product_equals_15_l656_65686


namespace harvard_attendance_percentage_l656_65685

theorem harvard_attendance_percentage 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ)
  (other_schools_rate : ℚ)
  (attending_students : ℕ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  other_schools_rate = 1 / 10 →
  attending_students = 900 →
  (attending_students : ℚ) / (total_applicants * acceptance_rate) = 9 / 10 := by
  sorry

end harvard_attendance_percentage_l656_65685


namespace largest_root_of_quadratic_l656_65607

theorem largest_root_of_quadratic (y : ℝ) :
  (6 * y ^ 2 - 31 * y + 35 = 0) → y ≤ (5 / 2 : ℝ) :=
by
  sorry

end largest_root_of_quadratic_l656_65607


namespace vasya_no_purchase_days_l656_65656

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
    x + y + z + w = 15 →  -- Total school days
    9 * x + 4 * z = 30 →  -- Total marshmallows bought
    2 * y + z = 9 →       -- Total meat pies bought
    w = 7 :=              -- Days with no purchase
by
  sorry

end vasya_no_purchase_days_l656_65656


namespace total_fruits_is_213_l656_65660

/-- Represents a fruit grower -/
structure FruitGrower where
  watermelons : ℕ
  pineapples : ℕ

/-- Calculates the total fruits grown by a single grower -/
def totalFruits (grower : FruitGrower) : ℕ :=
  grower.watermelons + grower.pineapples

/-- Represents the group of fruit growers -/
def fruitGrowers : List FruitGrower :=
  [{ watermelons := 37, pineapples := 56 },  -- Jason
   { watermelons := 68, pineapples := 27 },  -- Mark
   { watermelons := 11, pineapples := 14 }]  -- Sandy

/-- Theorem: The total number of fruits grown by the group is 213 -/
theorem total_fruits_is_213 : 
  (fruitGrowers.map totalFruits).sum = 213 := by
  sorry

end total_fruits_is_213_l656_65660


namespace cube_difference_equality_l656_65648

theorem cube_difference_equality : 
  - (666 : ℤ)^3 + (555 : ℤ)^3 = ((666 : ℤ)^2 - 666 * 555 + (555 : ℤ)^2) * (-124072470) := by
  sorry

end cube_difference_equality_l656_65648


namespace colored_cells_count_l656_65676

theorem colored_cells_count (k l : ℕ) : 
  k * l = 74 → 
  (∃ (rows cols : ℕ), 
    rows = 2 * k + 1 ∧ 
    cols = 2 * l + 1 ∧ 
    (rows * cols - 74 = 301 ∨ rows * cols - 74 = 373)) := by
  sorry

end colored_cells_count_l656_65676


namespace range_of_m_l656_65662

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^3 + b^3 + 1 = m * (a + b + 1)^3) :
  (3 * Real.sqrt 2 - 4) / 2 ≤ m ∧ m < 1/4 := by
  sorry

end range_of_m_l656_65662


namespace min_value_of_f_l656_65622

def f (x : ℝ) := 27 * x - x^3

theorem min_value_of_f :
  ∃ (min : ℝ), min = -54 ∧
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end min_value_of_f_l656_65622


namespace tea_leaf_problem_l656_65605

theorem tea_leaf_problem (num_plants : ℕ) (remaining_fraction : ℚ) (total_remaining : ℕ) :
  num_plants = 3 →
  remaining_fraction = 2/3 →
  total_remaining = 36 →
  ∃ initial_per_plant : ℕ,
    initial_per_plant * num_plants * remaining_fraction = total_remaining ∧
    initial_per_plant = 18 :=
by sorry

end tea_leaf_problem_l656_65605


namespace age_problem_l656_65668

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  b = 2 * c →  -- b is twice as old as c
  a + b + c = 17 →  -- The total of the ages of a, b, and c is 17
  b = 6 :=  -- Prove that b is 6 years old
by
  sorry

end age_problem_l656_65668


namespace max_intersection_difference_l656_65692

/-- The first function in the problem -/
def f (x : ℝ) : ℝ := 4 - x^2 + x^3

/-- The second function in the problem -/
def g (x : ℝ) : ℝ := 2 + 2*x^2 + x^3

/-- The difference between the y-coordinates of the intersection points -/
def intersection_difference (x : ℝ) : ℝ := |f x - g x|

/-- The theorem stating the maximum difference between y-coordinates of intersection points -/
theorem max_intersection_difference : 
  ∃ (x : ℝ), f x = g x ∧ 
  ∀ (y : ℝ), f y = g y → intersection_difference x ≥ intersection_difference y ∧
  intersection_difference x = 2 * (2/3)^(3/2) :=
sorry

end max_intersection_difference_l656_65692


namespace expectation_decreases_variance_increases_l656_65617

def boxA : ℕ := 1
def boxB : ℕ := 6
def redInB : ℕ := 3

def E (n : ℕ) : ℚ := (n / 2 + 1) / (n + 1)

def D (n : ℕ) : ℚ := E n * (1 - E n)

theorem expectation_decreases_variance_increases :
  ∀ n m : ℕ, 1 ≤ n → n < m → m ≤ 6 →
    (E n > E m) ∧ (D n < D m) := by
  sorry

end expectation_decreases_variance_increases_l656_65617


namespace chinese_character_equation_l656_65603

theorem chinese_character_equation :
  ∃! (a b c d : Nat),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    d + d + d + d = 48 ∧
    1000 * a + 100 * b + 10 * c + d = 1468 :=
by
  sorry

end chinese_character_equation_l656_65603


namespace auction_bid_ratio_l656_65698

/-- Auction problem statement -/
theorem auction_bid_ratio :
  -- Auction starts at $300
  let start_price : ℕ := 300
  -- Harry's first bid adds $200 to the starting value
  let harry_first_bid : ℕ := start_price + 200
  -- A third bidder adds three times Harry's bid
  let third_bid : ℕ := harry_first_bid + 3 * harry_first_bid
  -- Harry's final bid is $4,000
  let harry_final_bid : ℕ := 4000
  -- Harry's final bid exceeded the third bidder's bid by $1500
  let third_bid_final : ℕ := harry_final_bid - 1500
  -- Calculate the second bidder's bid
  let second_bid : ℕ := third_bid_final - 3 * harry_first_bid
  -- The ratio of the second bidder's bid to Harry's first bid is 2:1
  second_bid / harry_first_bid = 2 := by
  sorry

end auction_bid_ratio_l656_65698


namespace geometric_progression_problem_l656_65619

def is_geometric_progression (a : ℝ) (r : ℝ) : List ℝ → Prop
  | [x₁, x₂, x₃, x₄, x₅] => x₁ = a ∧ x₂ = a * r ∧ x₃ = a * r^2 ∧ x₄ = a * r^3 ∧ x₅ = a * r^4
  | _ => False

theorem geometric_progression_problem (a r : ℝ) (h₁ : a + a * r^2 + a * r^4 = 63) (h₂ : a * r + a * r^3 = 30) :
  is_geometric_progression a r [3, 6, 12, 24, 48] ∨ is_geometric_progression a r [48, 24, 12, 6, 3] := by
  sorry

end geometric_progression_problem_l656_65619


namespace min_distance_ellipse_line_l656_65667

def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y

def point_A (y : ℝ) : ℝ × ℝ := (4, y)

def point_B (m n : ℝ) : ℝ × ℝ := (m, n)

def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem min_distance_ellipse_line :
  ∃ (y m n : ℝ),
    point_on_ellipse 0 (Real.sqrt 5) ∧
    point_on_ellipse m n ∧
    perpendicular (point_A y) (point_B m n) ∧
    ∀ (y' m' n' : ℝ),
      point_on_ellipse m' n' →
      perpendicular (point_A y') (point_B m' n') →
      (m - 4)^2 + (n - y)^2 ≤ (m' - 4)^2 + (n' - y')^2 ∧
      (m - 4)^2 + (n - y)^2 = 21 :=
by sorry

end min_distance_ellipse_line_l656_65667
