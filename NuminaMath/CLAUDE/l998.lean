import Mathlib

namespace bucket_radius_l998_99804

/-- Proves that a cylindrical bucket with height 36 cm, when emptied to form a conical heap
    of height 12 cm and base radius 63 cm, has a radius of 21 cm. -/
theorem bucket_radius (h_cylinder h_cone r_cone : ℝ) 
    (h_cylinder_val : h_cylinder = 36)
    (h_cone_val : h_cone = 12)
    (r_cone_val : r_cone = 63)
    (volume_eq : π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :
  r_cylinder = 21 :=
sorry

end bucket_radius_l998_99804


namespace accessory_production_equation_l998_99898

theorem accessory_production_equation 
  (initial_production : ℕ) 
  (total_production : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 600000) 
  (h2 : total_production = 2180000) :
  (600 : ℝ) + 600 * (1 + x) + 600 * (1 + x)^2 = 2180 :=
by sorry

end accessory_production_equation_l998_99898


namespace total_harvest_l998_99854

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem: The total number of sacks of oranges harvested after 6 days is 498 -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end total_harvest_l998_99854


namespace fraction_equals_zero_l998_99802

theorem fraction_equals_zero (x : ℝ) : (x - 1) / (3 * x + 1) = 0 → x = 1 := by
  sorry

end fraction_equals_zero_l998_99802


namespace geometric_sequence_properties_l998_99815

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_properties (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3 → increasing_sequence a) ∧
  (increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  (a 1 ≥ a 2 ∧ a 2 ≥ a 3 → ¬increasing_sequence a) ∧
  (¬increasing_sequence a → a 1 ≥ a 2 ∧ a 2 ≥ a 3) :=
sorry

end geometric_sequence_properties_l998_99815


namespace tan_two_pi_thirds_l998_99891

theorem tan_two_pi_thirds : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end tan_two_pi_thirds_l998_99891


namespace circle_polar_equation_l998_99860

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  a : ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a given polar equation represents the specified circle -/
def is_correct_polar_equation (circle : PolarCircle) (equation : ℝ → ℝ → Prop) : Prop :=
  circle.center = (circle.a / 2, Real.pi / 2) ∧
  circle.radius = circle.a / 2 ∧
  ∀ θ ρ, equation ρ θ ↔ ρ = circle.a * Real.sin θ

theorem circle_polar_equation (a : ℝ) (h : a > 0) :
  let circle : PolarCircle := ⟨a, (a / 2, Real.pi / 2), a / 2⟩
  is_correct_polar_equation circle (fun ρ θ ↦ ρ = a * Real.sin θ) := by
  sorry

end circle_polar_equation_l998_99860


namespace equation_solution_l998_99842

theorem equation_solution : 
  ∀ x : ℝ, (x - 1)^2 = 64 ↔ x = 9 ∨ x = -7 := by sorry

end equation_solution_l998_99842


namespace age_problem_l998_99890

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 37 →  -- Total age is 37
  b = 14 :=  -- B's age is 14
by
  sorry

end age_problem_l998_99890


namespace same_terminal_side_l998_99828

theorem same_terminal_side (a b : Real) : 
  a = -7 * π / 9 → b = 11 * π / 9 → ∃ k : Int, b - a = 2 * π * k := by
  sorry

#check same_terminal_side

end same_terminal_side_l998_99828


namespace cow_chicken_goat_problem_l998_99861

theorem cow_chicken_goat_problem (cows chickens goats : ℕ) : 
  cows + chickens + goats = 12 →
  4 * cows + 2 * chickens + 4 * goats = 18 + 2 * (cows + chickens + goats) →
  cows + goats = 9 := by
  sorry

end cow_chicken_goat_problem_l998_99861


namespace f_monotonicity_and_max_value_l998_99825

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

-- State the theorem
theorem f_monotonicity_and_max_value :
  -- Part 1: Monotonicity
  (∀ x y : ℝ, x < y ∧ y < 3/4 → f x > f y) ∧
  (∀ x y : ℝ, 3/4 < x ∧ x < y → f x < f y) ∧
  -- Part 2: Maximum value on [2, 4]
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → f x ≤ 42) ∧
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = 42) :=
by sorry


end f_monotonicity_and_max_value_l998_99825


namespace road_trip_cost_l998_99865

/-- Represents a city with its distance from the starting point and gas price -/
structure City where
  distance : ℝ
  gasPrice : ℝ

/-- Calculates the total cost of a road trip given the car's specifications and cities visited -/
def totalTripCost (fuelEfficiency : ℝ) (tankCapacity : ℝ) (cities : List City) : ℝ :=
  cities.foldl (fun acc city => acc + tankCapacity * city.gasPrice) 0

/-- Theorem: The total cost of the road trip is $192.00 -/
theorem road_trip_cost :
  let fuelEfficiency : ℝ := 30
  let tankCapacity : ℝ := 20
  let cities : List City := [
    { distance := 290, gasPrice := 3.10 },
    { distance := 450, gasPrice := 3.30 },
    { distance := 620, gasPrice := 3.20 }
  ]
  totalTripCost fuelEfficiency tankCapacity cities = 192 :=
by
  sorry

#eval totalTripCost 30 20 [
  { distance := 290, gasPrice := 3.10 },
  { distance := 450, gasPrice := 3.30 },
  { distance := 620, gasPrice := 3.20 }
]

end road_trip_cost_l998_99865


namespace trip_distance_l998_99893

/-- Proves that the total distance of a trip is 350 km given specific conditions -/
theorem trip_distance (first_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (avg_speed : ℝ) :
  first_distance = 200 →
  first_speed = 20 →
  second_speed = 15 →
  avg_speed = 17.5 →
  ∃ (total_distance : ℝ),
    total_distance = first_distance + (avg_speed * (first_distance / first_speed + (total_distance - first_distance) / second_speed) - first_distance) ∧
    total_distance = 350 :=
by sorry

end trip_distance_l998_99893


namespace degree_of_divisor_l998_99811

/-- Given a polynomial f of degree 15 and another polynomial d, 
    if f divided by d results in a quotient of degree 7 and a remainder of degree 4, 
    then the degree of d is 8. -/
theorem degree_of_divisor (f d : Polynomial ℝ) (q : Polynomial ℝ) (r : Polynomial ℝ) :
  Polynomial.degree f = 15 →
  f = d * q + r →
  Polynomial.degree q = 7 →
  Polynomial.degree r = 4 →
  Polynomial.degree d = 8 := by
  sorry


end degree_of_divisor_l998_99811


namespace alex_mean_score_l998_99841

def scores : List ℝ := [86, 88, 90, 91, 95, 99]

def jane_score_count : ℕ := 2
def alex_score_count : ℕ := 4
def jane_mean_score : ℝ := 93

theorem alex_mean_score : 
  (scores.sum - jane_score_count * jane_mean_score) / alex_score_count = 90.75 := by
  sorry

end alex_mean_score_l998_99841


namespace power_of_fraction_to_decimal_l998_99875

theorem power_of_fraction_to_decimal :
  (4 / 5 : ℚ) ^ 3 = 512 / 1000 := by sorry

end power_of_fraction_to_decimal_l998_99875


namespace range_of_m_for_propositions_l998_99873

theorem range_of_m_for_propositions (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∨
  (∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0) →
  m < -1 := by sorry

end range_of_m_for_propositions_l998_99873


namespace muffin_division_l998_99871

theorem muffin_division (total_muffins : ℕ) (total_people : ℕ) (muffins_per_person : ℕ) : 
  total_muffins = 20 →
  total_people = 5 →
  total_muffins = total_people * muffins_per_person →
  muffins_per_person = 4 :=
by sorry

end muffin_division_l998_99871


namespace no_primes_satisfying_equation_l998_99874

theorem no_primes_satisfying_equation : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c :=
by sorry

end no_primes_satisfying_equation_l998_99874


namespace dividend_calculation_l998_99830

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 35.8)
  (h2 : quotient = 21.65)
  (h3 : remainder = 11.3) :
  divisor * quotient + remainder = 786.47 :=
by sorry

end dividend_calculation_l998_99830


namespace positive_real_inequalities_l998_99816

theorem positive_real_inequalities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (|a^2 - b^2| = 1 → |a - b| < 1) := by
  sorry

end positive_real_inequalities_l998_99816


namespace sum_of_multiples_of_4_between_63_and_151_l998_99856

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let first := (lower + 3) / 4 * 4
  let last := upper / 4 * 4
  let n := (last - first) / 4 + 1
  n * (first + last) / 2

theorem sum_of_multiples_of_4_between_63_and_151 :
  sumOfMultiplesOf4 63 151 = 2332 := by
  sorry

end sum_of_multiples_of_4_between_63_and_151_l998_99856


namespace sqrt_72_equals_6_sqrt_2_l998_99866

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_equals_6_sqrt_2_l998_99866


namespace carpet_length_is_two_l998_99834

/-- Represents a rectangular carpet with three concentric regions -/
structure Carpet where
  central_length : ℝ
  central_width : ℝ
  mid_width : ℝ
  outer_width : ℝ

/-- Calculates the area of the central region -/
def central_area (c : Carpet) : ℝ := c.central_length * c.central_width

/-- Calculates the area of the middle region -/
def middle_area (c : Carpet) : ℝ :=
  (c.central_length + 2 * c.mid_width) * (c.central_width + 2 * c.mid_width) - c.central_length * c.central_width

/-- Calculates the area of the outer region -/
def outer_area (c : Carpet) : ℝ :=
  (c.central_length + 2 * c.mid_width + 2 * c.outer_width) * (c.central_width + 2 * c.mid_width + 2 * c.outer_width) -
  (c.central_length + 2 * c.mid_width) * (c.central_width + 2 * c.mid_width)

/-- Checks if three areas form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℝ) : Prop := 2 * b = a + c

theorem carpet_length_is_two (c : Carpet) 
  (h1 : c.central_width = 1)
  (h2 : c.mid_width = 1)
  (h3 : c.outer_width = 1)
  (h4 : is_arithmetic_progression (central_area c) (middle_area c) (outer_area c)) :
  c.central_length = 2 := by
  sorry

end carpet_length_is_two_l998_99834


namespace walter_hushpuppies_cooking_time_l998_99835

/-- Calculates the time required to cook hushpuppies for a given number of guests -/
def cookingTime (guests : ℕ) (hushpuppiesPerGuest : ℕ) (hushpuppiesPerBatch : ℕ) (minutesPerBatch : ℕ) : ℕ :=
  let totalHushpuppies := guests * hushpuppiesPerGuest
  let batches := (totalHushpuppies + hushpuppiesPerBatch - 1) / hushpuppiesPerBatch
  batches * minutesPerBatch

/-- Proves that the cooking time for Walter's hushpuppies is 80 minutes -/
theorem walter_hushpuppies_cooking_time :
  cookingTime 20 5 10 8 = 80 := by
  sorry

end walter_hushpuppies_cooking_time_l998_99835


namespace unique_positive_solution_l998_99877

/-- The polynomial function f(x) = x^10 + 5x^9 + 28x^8 + 145x^7 - 1897x^6 -/
def f (x : ℝ) : ℝ := x^10 + 5*x^9 + 28*x^8 + 145*x^7 - 1897*x^6

/-- Theorem: The equation f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by sorry

end unique_positive_solution_l998_99877


namespace angle_bisector_intersection_ratio_l998_99837

/-- Given a triangle PQR with points M on PQ and N on PR such that
    PM:MQ = 2:6 and PN:NR = 3:9, if PS is the angle bisector of angle P
    intersecting MN at L, then PL:PS = 1:4 -/
theorem angle_bisector_intersection_ratio (P Q R M N S L : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 2 * t = 6 * (1 - t)) →
  (∃ u : ℝ, N = (1 - u) • P + u • R ∧ 3 * u = 9 * (1 - u)) →
  (∃ v : ℝ, S = (1 - v) • P + v • Q ∧ 
            ∃ w : ℝ, S = (1 - w) • P + w • R ∧
            v / (1 - v) = w / (1 - w)) →
  (∃ k : ℝ, L = (1 - k) • M + k • N) →
  (∃ r : ℝ, L = (1 - r) • P + r • S ∧ r = 1/4) :=
by sorry

end angle_bisector_intersection_ratio_l998_99837


namespace sum_of_coefficients_l998_99855

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ = -14 := by
sorry

end sum_of_coefficients_l998_99855


namespace square_to_octagon_triangle_to_icosagon_l998_99897

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a triangle
structure Triangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define an octagon
structure Octagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a 20-sided polygon (icosagon)
structure Icosagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Function to cut a square into two parts
def cut_square (s : Square) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an octagon from two parts
def form_octagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Octagon := sorry

-- Function to cut a triangle into two parts
def cut_triangle (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an icosagon from two parts
def form_icosagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Icosagon := sorry

-- Theorem stating that a square can be cut into two parts to form an octagon
theorem square_to_octagon (s : Square) :
  ∃ (o : Octagon), form_octagon (cut_square s) = o := sorry

-- Theorem stating that a triangle can be cut into two parts to form an icosagon
theorem triangle_to_icosagon (t : Triangle) :
  ∃ (i : Icosagon), form_icosagon (cut_triangle t) = i := sorry

end square_to_octagon_triangle_to_icosagon_l998_99897


namespace geometric_sequence_common_ratio_l998_99859

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 0 + a 1 + a 2 = 3 * a 0) 
  (h_nonzero : a 0 ≠ 0) : 
  q = -2 ∨ q = 1 := by
sorry

end geometric_sequence_common_ratio_l998_99859


namespace arithmetic_sequence_length_l998_99853

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) = s n + d

/-- The last term of a finite arithmetic sequence. -/
def last_term (s : ℕ → ℤ) (n : ℕ) : ℤ := s (n - 1)

theorem arithmetic_sequence_length :
  ∀ s : ℕ → ℤ,
  is_arithmetic_sequence s →
  s 0 = -3 →
  last_term s 13 = 45 →
  ∃ n : ℕ, n = 13 ∧ last_term s n = 45 :=
by sorry

end arithmetic_sequence_length_l998_99853


namespace inequality_solution_sets_l998_99814

theorem inequality_solution_sets (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 3) →
  (∀ x : ℝ, (b*x^2 + a) / (x + 1) > 0 ↔ x < -1 ∨ (-1/3 < x ∧ x < 0)) :=
by sorry

end inequality_solution_sets_l998_99814


namespace spurs_basketball_count_l998_99806

/-- The number of players on the Spurs basketball team -/
def num_players : ℕ := 22

/-- The number of basketballs each player has -/
def balls_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := num_players * balls_per_player

theorem spurs_basketball_count : total_basketballs = 242 := by
  sorry

end spurs_basketball_count_l998_99806


namespace complex_modulus_problem_l998_99822

theorem complex_modulus_problem (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) - i) / (a + i)).im ≠ 0 ∧ (((1 : ℂ) - i) / (a + i)).re = 0 →
  Complex.abs ((2 * a + 1 : ℂ) + Complex.I * Real.sqrt 2) = Real.sqrt 11 :=
by sorry

end complex_modulus_problem_l998_99822


namespace number_in_scientific_notation_l998_99847

/-- Definition of scientific notation -/
def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

/-- The number to be expressed in scientific notation -/
def number : ℝ := 123000

/-- Theorem stating that 123000 can be expressed as 1.23 × 10^5 in scientific notation -/
theorem number_in_scientific_notation :
  scientific_notation number 1.23 5 :=
sorry

end number_in_scientific_notation_l998_99847


namespace total_coins_is_twelve_l998_99807

def coins_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x / 2)

theorem total_coins_is_twelve :
  ∃ x : ℕ, 
    x > 0 ∧ 
    let (pete_coins, paul_coins) := coins_distribution x
    pete_coins = 5 * paul_coins ∧
    pete_coins + paul_coins = 12 := by
  sorry

end total_coins_is_twelve_l998_99807


namespace new_average_weight_l998_99868

/-- Given 29 students with an average weight of 28 kg, after admitting a new student weighing 1 kg,
    the new average weight of all 30 students is 27.1 kg. -/
theorem new_average_weight (initial_count : ℕ) (initial_avg : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_student_weight = 1 →
  let total_weight := initial_count * initial_avg + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.1 :=
by sorry

end new_average_weight_l998_99868


namespace restaurant_tip_percentage_l998_99881

/-- Calculates the tip percentage given the cost of an appetizer, number of entrees,
    cost per entree, and total amount spent at a restaurant. -/
theorem restaurant_tip_percentage
  (appetizer_cost : ℚ)
  (num_entrees : ℕ)
  (entree_cost : ℚ)
  (total_spent : ℚ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : total_spent = 108) :
  (total_spent - (appetizer_cost + num_entrees * entree_cost)) / (appetizer_cost + num_entrees * entree_cost) = 1/5 := by
  sorry

end restaurant_tip_percentage_l998_99881


namespace cubic_root_sum_cubes_l998_99899

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 502 * a + 1004 = 0) →
  (4 * b^3 + 502 * b + 1004 = 0) →
  (4 * c^3 + 502 * c + 1004 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 := by
sorry

end cubic_root_sum_cubes_l998_99899


namespace orangeade_price_theorem_l998_99800

/-- Represents the price of orangeade per glass -/
@[ext] structure OrangeadePrice where
  price : ℚ

/-- Represents the composition of orangeade -/
@[ext] structure OrangeadeComposition where
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade -/
def total_volume (c : OrangeadeComposition) : ℚ :=
  c.orange_juice + c.water

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : ℚ) : ℚ :=
  price.price * volume

theorem orangeade_price_theorem 
  (day1_comp : OrangeadeComposition)
  (day2_comp : OrangeadeComposition)
  (day2_price : OrangeadePrice)
  (h1 : day1_comp.orange_juice = day1_comp.water)
  (h2 : day2_comp.orange_juice = day1_comp.orange_juice)
  (h3 : day2_comp.water = 2 * day2_comp.orange_juice)
  (h4 : day2_price.price = 32/100)
  (h5 : ∃ (day1_price : OrangeadePrice), 
        revenue day1_price (total_volume day1_comp) = 
        revenue day2_price (total_volume day2_comp)) :
  ∃ (day1_price : OrangeadePrice), day1_price.price = 48/100 := by
sorry


end orangeade_price_theorem_l998_99800


namespace min_tablets_to_extract_l998_99882

/-- Represents the number of tablets for each medicine type in the box -/
structure TabletCount where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least two of each type -/
def minTablets (count : TabletCount) : Nat :=
  (count.a - 1) + (count.b - 1) + 2

/-- Theorem stating the minimum number of tablets to extract for the given problem -/
theorem min_tablets_to_extract (box : TabletCount) 
  (ha : box.a = 25) (hb : box.b = 30) (hc : box.c = 20) : 
  minTablets box = 55 := by
  sorry

#eval minTablets { a := 25, b := 30, c := 20 }

end min_tablets_to_extract_l998_99882


namespace trigonometric_equation_solution_l998_99821

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.cos (9 * x) - Real.cos (5 * x) - Real.sqrt 2 * Real.cos (4 * x) + Real.sin (9 * x) + Real.sin (5 * x) = 0) →
  (∃ k : ℤ, x = π / 8 + π * k / 2 ∨ x = π / 20 + 2 * π * k / 5 ∨ x = π / 12 + 2 * π * k / 9) :=
by sorry

end trigonometric_equation_solution_l998_99821


namespace point_on_line_l998_99845

theorem point_on_line (m : ℝ) : (5 : ℝ) = 2 * m + 1 → m = 2 := by
  sorry

end point_on_line_l998_99845


namespace clothing_store_problem_l998_99894

/-- The clothing store problem -/
theorem clothing_store_problem 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ)
  (h1 : cost = 50)
  (h2 : initial_price = 60)
  (h3 : initial_volume = 800)
  (h4 : price_increase = 5)
  (h5 : volume_decrease = 100) :
  let sales_volume (x : ℝ) := initial_volume - (volume_decrease / price_increase) * (x - initial_price)
  let profit (x : ℝ) := (x - cost) * sales_volume x
  ∃ (max_price : ℝ) (max_profit : ℝ),
    -- 1. Sales volume at 70 yuan
    sales_volume 70 = 600 ∧
    -- 2. Profit at 70 yuan
    profit 70 = 12000 ∧
    -- 3. Profit function
    (∀ x, profit x = -20 * x^2 + 3000 * x - 100000) ∧
    -- 4. Maximum profit
    (∀ x, profit x ≤ max_profit) ∧ max_price = 75 ∧ max_profit = 12500 ∧
    -- 5. Selling prices for 12000 yuan profit
    profit 70 = 12000 ∧ profit 80 = 12000 ∧
    (∀ x, profit x = 12000 → (x = 70 ∨ x = 80)) := by
  sorry

end clothing_store_problem_l998_99894


namespace octagon_perimeter_l998_99829

/-- The perimeter of an octagon with alternating side lengths -/
theorem octagon_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 2) :
  4 * a + 4 * b = 12 + 8 * Real.sqrt 2 := by
  sorry

#check octagon_perimeter

end octagon_perimeter_l998_99829


namespace complex_equation_proof_l998_99886

theorem complex_equation_proof (a : ℝ) : 
  ((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0 → a = 1 := by
  sorry

end complex_equation_proof_l998_99886


namespace range_of_a_l998_99831

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end range_of_a_l998_99831


namespace alison_large_tubs_l998_99867

/-- The number of large tubs Alison bought -/
def num_large_tubs : ℕ := 3

/-- The number of small tubs Alison bought -/
def num_small_tubs : ℕ := 6

/-- The cost of each large tub in dollars -/
def cost_large_tub : ℕ := 6

/-- The cost of each small tub in dollars -/
def cost_small_tub : ℕ := 5

/-- The total cost of all tubs in dollars -/
def total_cost : ℕ := 48

theorem alison_large_tubs : 
  num_large_tubs * cost_large_tub + num_small_tubs * cost_small_tub = total_cost := by
  sorry

end alison_large_tubs_l998_99867


namespace jennas_tanning_schedule_l998_99878

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule 
  (total_time : ℕ) 
  (daily_time : ℕ) 
  (last_two_weeks_time : ℕ) 
  (h1 : total_time = 200)
  (h2 : daily_time = 30)
  (h3 : last_two_weeks_time = 80) :
  (total_time - last_two_weeks_time) / (2 * daily_time) = 2 := by
  sorry

end jennas_tanning_schedule_l998_99878


namespace tire_repair_cost_l998_99872

/-- Calculates the final cost of tire repairs -/
def final_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The final cost for repairing 4 tires is $30 -/
theorem tire_repair_cost : final_cost 7 0.5 4 = 30 := by
  sorry

end tire_repair_cost_l998_99872


namespace car_speed_relationship_l998_99864

/-- Represents the relationship between the speeds and travel times of two cars -/
theorem car_speed_relationship (x : ℝ) : x > 0 →
  (80 / x - 2 = 80 / (3 * x) + 2 / 3) ↔
  (80 / x = 80 / (3 * x) + 2 + 2 / 3 ∧
   80 = x * (80 / (3 * x) + 2 + 2 / 3) ∧
   80 = 3 * x * (80 / (3 * x) + 2 / 3)) := by
  sorry

#check car_speed_relationship

end car_speed_relationship_l998_99864


namespace not_all_primes_from_cards_l998_99838

/-- A card with two digits -/
structure Card :=
  (front : Nat)
  (back : Nat)

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Generate all two-digit numbers from two cards -/
def twoDigitNumbers (card1 card2 : Card) : List Nat :=
  [
    10 * card1.front + card2.front,
    10 * card1.front + card2.back,
    10 * card1.back + card2.front,
    10 * card1.back + card2.back,
    10 * card2.front + card1.front,
    10 * card2.front + card1.back,
    10 * card2.back + card1.front,
    10 * card2.back + card1.back
  ]

/-- Main theorem -/
theorem not_all_primes_from_cards :
  ∀ (card1 card2 : Card),
    card1.front ≠ card1.back ∧
    card2.front ≠ card2.back ∧
    card1.front ≠ card2.front ∧
    card1.front ≠ card2.back ∧
    card1.back ≠ card2.front ∧
    card1.back ≠ card2.back ∧
    card1.front < 10 ∧ card1.back < 10 ∧ card2.front < 10 ∧ card2.back < 10 →
    ∃ (n : Nat), n ∈ twoDigitNumbers card1 card2 ∧ ¬isPrime n :=
by sorry

end not_all_primes_from_cards_l998_99838


namespace five_fourths_of_twelve_fifths_l998_99805

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 5/4 * (12/5) → x = 3 := by
  sorry

end five_fourths_of_twelve_fifths_l998_99805


namespace distinct_colorings_l998_99809

/-- Represents the symmetry group of a regular decagon -/
def DecagonSymmetryGroup : Type := Unit

/-- The order of the decagon symmetry group -/
def decagon_symmetry_order : ℕ := 10

/-- The number of disks in the decagon -/
def total_disks : ℕ := 10

/-- The number of disks to be colored -/
def colored_disks : ℕ := 8

/-- The number of blue disks -/
def blue_disks : ℕ := 4

/-- The number of red disks -/
def red_disks : ℕ := 3

/-- The number of green disks -/
def green_disks : ℕ := 2

/-- The number of yellow disks -/
def yellow_disks : ℕ := 1

/-- The total number of colorings without considering symmetry -/
def total_colorings : ℕ := (total_disks.choose blue_disks) * 
                           ((total_disks - blue_disks).choose red_disks) * 
                           ((total_disks - blue_disks - red_disks).choose green_disks) * 
                           ((total_disks - blue_disks - red_disks - green_disks).choose yellow_disks)

/-- The number of distinct colorings considering symmetry -/
theorem distinct_colorings : 
  (total_colorings / decagon_symmetry_order : ℚ) = 1260 := by sorry

end distinct_colorings_l998_99809


namespace dimitri_calories_l998_99883

/-- Calculates the total calories consumed by Dimitri over two days -/
def calories_two_days (burgers_per_day : ℕ) (calories_per_burger : ℕ) : ℕ :=
  2 * burgers_per_day * calories_per_burger

/-- Proves that Dimitri consumes 120 calories over two days -/
theorem dimitri_calories : calories_two_days 3 20 = 120 := by
  sorry

end dimitri_calories_l998_99883


namespace solution_set_l998_99818

/-- A function that checks if three positive real numbers can form a non-degenerate triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- The property that n must satisfy -/
def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ∃ (l j k : ℕ), is_triangle (a * n ^ k) (b * n ^ j) (c * n ^ l)

/-- The main theorem stating that only 2, 3, and 4 satisfy the condition -/
theorem solution_set : {n : ℕ | satisfies_condition n} = {2, 3, 4} := by sorry

end solution_set_l998_99818


namespace new_average_is_34_l998_99848

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverage (performance : BatsmanPerformance) : ℕ :=
  performance.lastInningScore + (performance.innings - 1) * (performance.lastInningScore / performance.innings + performance.averageIncrease - 3)

/-- Theorem stating that the new average is 34 for the given conditions -/
theorem new_average_is_34 (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningScore = 82)
  (h3 : performance.averageIncrease = 3) :
  newAverage performance = 34 := by
  sorry

end new_average_is_34_l998_99848


namespace two_true_propositions_l998_99812

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : Nat, n = 2 ∧ 
    (((a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      (a * c^2 ≤ b * c^2 → a ≤ b)) → n = 4) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 2) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      ¬(a * c^2 > b * c^2 → a > b) ∧ 
      ¬(a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 0)) :=
sorry

end two_true_propositions_l998_99812


namespace not_always_same_direction_for_parallel_vectors_l998_99843

-- Define a vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define parallel vectors
def parallel (u v : V) : Prop :=
  ∃ k : ℝ, v = k • u

-- Theorem statement
theorem not_always_same_direction_for_parallel_vectors :
  ¬ ∀ (u v : V), parallel u v → (∃ k : ℝ, k > 0 ∧ v = k • u) :=
sorry

end not_always_same_direction_for_parallel_vectors_l998_99843


namespace quadratic_roots_relation_l998_99850

theorem quadratic_roots_relation (k n p : ℝ) (hk : k ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -p ∧ s₁ * s₂ = k) ∧
               (3*s₁ + 3*s₂ = -k ∧ 9*s₁*s₂ = n)) →
  n / p = 27 := by
  sorry

end quadratic_roots_relation_l998_99850


namespace sheep_sheepdog_distance_l998_99863

/-- The initial distance between a sheep and a sheepdog -/
def initial_distance (sheep_speed sheepdog_speed : ℝ) (catch_time : ℝ) : ℝ :=
  sheepdog_speed * catch_time - sheep_speed * catch_time

/-- Theorem stating the initial distance between the sheep and sheepdog -/
theorem sheep_sheepdog_distance :
  initial_distance 12 20 20 = 160 := by
  sorry

end sheep_sheepdog_distance_l998_99863


namespace strawberry_yield_per_row_l998_99820

theorem strawberry_yield_per_row :
  let total_rows : ℕ := 7
  let total_yield : ℕ := 1876
  let yield_per_row : ℕ := total_yield / total_rows
  yield_per_row = 268 := by sorry

end strawberry_yield_per_row_l998_99820


namespace unique_number_l998_99879

theorem unique_number : ∃! x : ℕ, 
  x > 0 ∧ 
  (∃ k : ℕ, 10 * x + 4 = k * (x + 4)) ∧
  (10 * x + 4) / (x + 4) = x + 4 - 27 ∧
  x = 32 := by
sorry

end unique_number_l998_99879


namespace exists_irrational_less_than_three_l998_99810

theorem exists_irrational_less_than_three : ∃ x : ℝ, Irrational x ∧ |x| < 3 := by
  sorry

end exists_irrational_less_than_three_l998_99810


namespace N_subset_M_l998_99823

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem N_subset_M : N ⊆ M := by sorry

end N_subset_M_l998_99823


namespace opposite_roots_quadratic_l998_99801

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end opposite_roots_quadratic_l998_99801


namespace goat_redistribution_impossibility_l998_99836

theorem goat_redistribution_impossibility :
  ¬ ∃ (n m : ℕ), n + 7 * m = 150 ∧ 7 * n + m = 150 :=
by sorry

end goat_redistribution_impossibility_l998_99836


namespace crow_count_proof_l998_99888

/-- The number of crows in the first group -/
def first_group_count : ℕ := 3

/-- The number of worms eaten by the first group in one hour -/
def first_group_worms : ℕ := 30

/-- The number of crows in the second group -/
def second_group_count : ℕ := 5

/-- The number of worms eaten by the second group in two hours -/
def second_group_worms : ℕ := 100

/-- The number of hours the second group took to eat their worms -/
def second_group_hours : ℕ := 2

theorem crow_count_proof : first_group_count = 3 := by
  sorry

end crow_count_proof_l998_99888


namespace car_speed_theorem_l998_99870

def car_speed_problem (first_hour_speed average_speed : ℝ) : Prop :=
  let total_time : ℝ := 2
  let second_hour_speed : ℝ := 2 * average_speed - first_hour_speed
  second_hour_speed = 50

theorem car_speed_theorem :
  car_speed_problem 90 70 := by sorry

end car_speed_theorem_l998_99870


namespace second_duck_bread_pieces_l998_99857

theorem second_duck_bread_pieces : 
  ∀ (total_bread pieces_left first_duck_fraction last_duck_pieces : ℕ),
  total_bread = 100 →
  pieces_left = 30 →
  first_duck_fraction = 2 →  -- Represents 1/2
  last_duck_pieces = 7 →
  ∃ (second_duck_pieces : ℕ),
    second_duck_pieces = total_bread - pieces_left - (total_bread / first_duck_fraction) - last_duck_pieces ∧
    second_duck_pieces = 13 := by
  sorry

end second_duck_bread_pieces_l998_99857


namespace rectangular_plot_ratio_l998_99803

theorem rectangular_plot_ratio (length breadth area : ℝ) : 
  breadth = 14 →
  area = 588 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end rectangular_plot_ratio_l998_99803


namespace unique_integer_solution_l998_99824

theorem unique_integer_solution : 
  ∀ n : ℤ, (⌊(n^2 / 4 : ℚ) + n⌋ - ⌊n / 2⌋^2 = 5) ↔ n = 10 := by
  sorry

end unique_integer_solution_l998_99824


namespace soldier_average_score_l998_99808

theorem soldier_average_score : 
  let shots : List ℕ := List.replicate 6 10 ++ [9] ++ List.replicate 3 8
  (shots.sum : ℚ) / shots.length = 93/10 := by
  sorry

end soldier_average_score_l998_99808


namespace max_value_cos_sin_l998_99887

theorem max_value_cos_sin (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * Real.cos x + Real.sin x
  f x ≤ Real.sqrt 5 ∧ ∃ y, f y = Real.sqrt 5 := by
  sorry

end max_value_cos_sin_l998_99887


namespace probability_sum_10_three_dice_l998_99817

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 27

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 8 := by
  sorry

end probability_sum_10_three_dice_l998_99817


namespace sum_of_product_sequence_l998_99846

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ 2 * (a 3) = a 2

def arithmetic_sequence (b : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  b 1 = 1 ∧ S 3 = b 2 + 4

theorem sum_of_product_sequence
  (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (T : ℕ → ℚ) :
  geometric_sequence a →
  arithmetic_sequence b S →
  (∀ n : ℕ, T n = (a n) * (b n)) →
  ∀ n : ℕ, T n = 2 - (n + 2) * (1/2)^n :=
by sorry

end sum_of_product_sequence_l998_99846


namespace hyperbola_asymptote_through_point_implies_a_l998_99892

/-- A hyperbola with equation x²/a² - y²/4 = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  a_pos : a > 0

/-- The asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = (2/h.a) * x ∨ y = -(2/h.a) * x}

/-- Theorem stating that if one asymptote passes through (2, 1), then a = 4 -/
theorem hyperbola_asymptote_through_point_implies_a
  (h : Hyperbola)
  (asymptote_through_point : (2, 1) ∈ asymptotes h) :
  h.a = 4 := by
  sorry

end hyperbola_asymptote_through_point_implies_a_l998_99892


namespace inequality_solution_l998_99885

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem inequality_solution (x : ℕ) :
  x > 0 → (choose 5 x + permute x 3 < 30 ↔ x = 3 ∨ x = 4) :=
by sorry

end inequality_solution_l998_99885


namespace jaden_toy_cars_l998_99819

theorem jaden_toy_cars (initial_cars birthday_cars sister_cars friend_cars final_cars : ℕ) :
  initial_cars = 14 →
  birthday_cars = 12 →
  sister_cars = 8 →
  friend_cars = 3 →
  final_cars = 43 →
  ∃ (bought_cars : ℕ), 
    initial_cars + birthday_cars + bought_cars - sister_cars - friend_cars = final_cars ∧
    bought_cars = 28 :=
by sorry

end jaden_toy_cars_l998_99819


namespace nonzero_term_count_correct_l998_99880

/-- The number of nonzero terms in the expanded and simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzeroTermCount : ℕ := 1010025

/-- The degree of the polynomial expression -/
def degree : ℕ := 2008

theorem nonzero_term_count_correct :
  nonzeroTermCount = (degree / 2 + 1)^2 :=
sorry

end nonzero_term_count_correct_l998_99880


namespace no_integer_cube_equal_3n2_plus_3n_plus_7_l998_99852

theorem no_integer_cube_equal_3n2_plus_3n_plus_7 :
  ¬ ∃ (n m : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
  sorry

end no_integer_cube_equal_3n2_plus_3n_plus_7_l998_99852


namespace max_balloons_proof_l998_99839

/-- Represents the maximum number of balloons that can be purchased given a budget and pricing scheme. -/
def max_balloons (budget : ℕ) (regular_price : ℕ) (set_price : ℕ) : ℕ :=
  (budget / set_price) * 3

/-- Proves that given $120 to spend, with balloons priced at $4 each, and a special sale where every set of 3 balloons costs $7, the maximum number of balloons that can be purchased is 51. -/
theorem max_balloons_proof :
  max_balloons 120 4 7 = 51 := by
  sorry

end max_balloons_proof_l998_99839


namespace escalator_steps_l998_99840

/-- The number of steps Xiaolong takes to go down the escalator -/
def steps_down : ℕ := 30

/-- The number of steps Xiaolong takes to go up the escalator -/
def steps_up : ℕ := 90

/-- The ratio of Xiaolong's speed going up compared to going down -/
def speed_ratio : ℕ := 3

/-- The total number of visible steps on the escalator -/
def total_steps : ℕ := 60

theorem escalator_steps :
  ∃ (x : ℚ),
    (steps_down : ℚ) + (steps_down : ℚ) * x = (steps_up : ℚ) - (steps_up : ℚ) / speed_ratio * x ∧
    x = 1 ∧
    total_steps = steps_down + steps_down := by sorry

end escalator_steps_l998_99840


namespace equation_represents_hyperbola_l998_99832

/-- The equation x^2 - 18y^2 - 6x + 4y + 9 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), x^2 - 18*y^2 - 6*x + 4*y + 9 = 0 ↔
  ((x - c) / a)^2 - ((y - d) / b)^2 = 1 ∧
  e = 1 := by sorry

end equation_represents_hyperbola_l998_99832


namespace circle_equation_l998_99844

/-- A circle C with center (0, a) -/
structure Circle (a : ℝ) where
  center : ℝ × ℝ := (0, a)

/-- The equation of a circle with center (0, a) and radius r -/
def circleEquation (c : Circle a) (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = r^2

/-- The circle passes through the point (1, 0) -/
def passesThrough (c : Circle a) (r : ℝ) : Prop :=
  circleEquation c r 1 0

/-- The circle is divided by the x-axis into two arcs with length ratio 1:2 -/
def arcRatio (c : Circle a) : Prop :=
  abs (a / 1) = Real.sqrt 3

theorem circle_equation (a : ℝ) (c : Circle a) (h1 : passesThrough c (Real.sqrt (4/3)))
    (h2 : arcRatio c) :
    ∀ x y : ℝ, circleEquation c (Real.sqrt (4/3)) x y ↔ 
      x^2 + (y - Real.sqrt 3 / 3)^2 = 4/3 ∨ x^2 + (y + Real.sqrt 3 / 3)^2 = 4/3 :=
  sorry

end circle_equation_l998_99844


namespace equation_solution_l998_99862

theorem equation_solution : ∃ x : ℚ, 25 - 8 = 3 * x + 1 ∧ x = 16/3 := by
  sorry

end equation_solution_l998_99862


namespace simplest_fraction_C_l998_99826

def is_simplest_fraction (num : ℚ → ℚ) (denom : ℚ → ℚ) : Prop :=
  ∀ a : ℚ, ∀ k : ℚ, k ≠ 0 → num a / denom a = (k * num a) / (k * denom a) → k = 1 ∨ k = -1

theorem simplest_fraction_C :
  is_simplest_fraction (λ a => 2 * a) (λ a => 2 - a) :=
sorry

end simplest_fraction_C_l998_99826


namespace triangle_theorem_l998_99849

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (a, b + c)
  let n : ℝ × ℝ := (1, Real.cos C + Real.sqrt 3 * Real.sin C)
  (∀ k : ℝ, m = k • n) ∧  -- m is parallel to n
  3 * b * c = 16 - a^2 ∧
  A = Real.pi / 3 ∧
  (∀ S : ℝ, S = 1/2 * b * c * Real.sin A → S ≤ Real.sqrt 3)

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  triangle_problem a b c A B C → 
    A = Real.pi / 3 ∧
    (∃ S : ℝ, S = 1/2 * b * c * Real.sin A ∧ S = Real.sqrt 3) :=
by sorry

end triangle_theorem_l998_99849


namespace min_value_reciprocal_sum_l998_99851

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  1/m + 1/n ≥ 4 := by
  sorry

end min_value_reciprocal_sum_l998_99851


namespace power_mod_eleven_l998_99858

theorem power_mod_eleven : 6^305 % 11 = 10 := by sorry

end power_mod_eleven_l998_99858


namespace division_problem_solution_l998_99895

/-- Represents the division problem with given conditions -/
structure DivisionProblem where
  D : ℕ  -- dividend
  d : ℕ  -- divisor
  q : ℕ  -- quotient
  r : ℕ  -- remainder
  P : ℕ  -- prime number
  h1 : D = d * q + r
  h2 : r = 6
  h3 : d = 5 * q
  h4 : d = 3 * r + 2
  h5 : ∃ k : ℕ, D = P * k
  h6 : ∃ n : ℕ, q = n * n
  h7 : Nat.Prime P

theorem division_problem_solution (prob : DivisionProblem) : prob.D = 86 ∧ ∃ k : ℕ, prob.D = prob.P * k := by
  sorry

#check division_problem_solution

end division_problem_solution_l998_99895


namespace charles_watercolor_pictures_after_work_l998_99813

/-- Represents the number of pictures drawn on a specific type of paper -/
structure PictureCount where
  regular : ℕ
  watercolor : ℕ

/-- Represents the initial paper count and pictures drawn on different occasions -/
structure DrawingData where
  initialRegular : ℕ
  initialWatercolor : ℕ
  todayPictures : PictureCount
  yesterdayBeforeWork : ℕ
  remainingRegular : ℕ

/-- Calculates the number of watercolor pictures drawn after work yesterday -/
def watercolorPicturesAfterWork (data : DrawingData) : ℕ :=
  data.initialWatercolor - data.todayPictures.watercolor -
  (data.yesterdayBeforeWork - (data.initialRegular - data.todayPictures.regular - data.remainingRegular))

/-- Theorem stating that Charles drew 6 watercolor pictures after work yesterday -/
theorem charles_watercolor_pictures_after_work :
  let data : DrawingData := {
    initialRegular := 10,
    initialWatercolor := 10,
    todayPictures := { regular := 4, watercolor := 2 },
    yesterdayBeforeWork := 6,
    remainingRegular := 2
  }
  watercolorPicturesAfterWork data = 6 := by sorry

end charles_watercolor_pictures_after_work_l998_99813


namespace line_perpendicular_to_plane_l998_99896

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (L m : Line) (α : Plane) 
  (h1 : parallel m L) 
  (h2 : perpendicular m α) : 
  perpendicular L α :=
sorry

end line_perpendicular_to_plane_l998_99896


namespace f_8_equals_8_65_l998_99833

/-- A function that takes a natural number and returns a rational number -/
def f (n : ℕ) : ℚ := n / (n^2 + 1)

/-- Theorem stating that f(8) equals 8/65 -/
theorem f_8_equals_8_65 : f 8 = 8 / 65 := by
  sorry

end f_8_equals_8_65_l998_99833


namespace least_possible_area_l998_99827

/-- The least possible length of a side when measured as 4 cm to the nearest centimeter -/
def min_side_length : ℝ := 3.5

/-- The measured length of the square's side to the nearest centimeter -/
def measured_side_length : ℕ := 4

/-- The least possible area of the square -/
def min_area : ℝ := min_side_length ^ 2

theorem least_possible_area :
  min_area = 12.25 := by sorry

end least_possible_area_l998_99827


namespace param_line_point_l998_99876

/-- A parameterized line in 2D space -/
structure ParamLine where
  /-- The vector on the line at parameter t -/
  vector : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with known points, we can determine another point -/
theorem param_line_point (l : ParamLine)
  (h1 : l.vector 5 = (2, 1))
  (h2 : l.vector 6 = (5, -7)) :
  l.vector 1 = (-40, 113) := by
  sorry

end param_line_point_l998_99876


namespace average_salary_all_employees_l998_99889

/-- Calculate the average salary of all employees in an office --/
theorem average_salary_all_employees 
  (avg_salary_officers : ℝ) 
  (avg_salary_non_officers : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : avg_salary_officers = 440)
  (h2 : avg_salary_non_officers = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 480) :
  (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

end average_salary_all_employees_l998_99889


namespace factorization_equality_l998_99869

theorem factorization_equality (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^3 + b^3 + c^3 - 3*a*b*c) :=
by sorry

end factorization_equality_l998_99869


namespace more_girls_than_boys_l998_99884

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42)
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    4 * boys = 3 * girls ∧ 
    girls - boys = 6 := by
sorry

end more_girls_than_boys_l998_99884
