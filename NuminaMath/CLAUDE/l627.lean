import Mathlib

namespace expected_urns_with_one_marble_value_l627_62770

/-- The number of urns -/
def n : ℕ := 7

/-- The number of marbles -/
def m : ℕ := 5

/-- The probability that a specific urn has exactly one marble -/
def p : ℚ := m * (n - 1)^(m - 1) / n^m

/-- The expected number of urns with exactly one marble -/
def expected_urns_with_one_marble : ℚ := n * p

theorem expected_urns_with_one_marble_value : 
  expected_urns_with_one_marble = 6480 / 2401 := by sorry

end expected_urns_with_one_marble_value_l627_62770


namespace max_four_digit_binary_is_15_l627_62745

/-- The maximum value of a four-digit binary number in decimal -/
def max_four_digit_binary : ℕ := 15

/-- A function to convert a four-digit binary number to decimal -/
def binary_to_decimal (b₃ b₂ b₁ b₀ : Bool) : ℕ :=
  (if b₃ then 8 else 0) + (if b₂ then 4 else 0) + (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- Theorem stating that the maximum value of a four-digit binary number is 15 -/
theorem max_four_digit_binary_is_15 :
  ∀ b₃ b₂ b₁ b₀ : Bool, binary_to_decimal b₃ b₂ b₁ b₀ ≤ max_four_digit_binary :=
by sorry

end max_four_digit_binary_is_15_l627_62745


namespace sum_of_1006th_row_is_20112_l627_62762

/-- Calculates the sum of numbers in the nth row of the pattern -/
def row_sum (n : ℕ) : ℕ := n * (3 * n - 1) / 2

/-- The theorem states that the sum of numbers in the 1006th row equals 20112 -/
theorem sum_of_1006th_row_is_20112 : row_sum 1006 = 20112 := by
  sorry

end sum_of_1006th_row_is_20112_l627_62762


namespace dow_jones_decrease_l627_62750

theorem dow_jones_decrease (initial_value end_value : ℝ) : 
  (end_value = initial_value * 0.98) → 
  (end_value = 8722) → 
  (initial_value = 8900) := by
sorry

end dow_jones_decrease_l627_62750


namespace cupcakes_sold_correct_l627_62776

/-- Represents the number of cupcakes Katie sold at the bake sale -/
def cupcakes_sold (initial : ℕ) (additional : ℕ) (remaining : ℕ) : ℕ :=
  initial + additional - remaining

/-- Proves that the number of cupcakes sold is correct given the initial,
    additional, and remaining cupcakes -/
theorem cupcakes_sold_correct (initial : ℕ) (additional : ℕ) (remaining : ℕ) :
  cupcakes_sold initial additional remaining = initial + additional - remaining :=
by sorry

end cupcakes_sold_correct_l627_62776


namespace ariels_female_fish_l627_62703

/-- Given that Ariel has 45 fish in total and 2/3 of the fish are male,
    prove that the number of female fish is 15. -/
theorem ariels_female_fish :
  ∀ (total_fish : ℕ) (male_fraction : ℚ),
    total_fish = 45 →
    male_fraction = 2/3 →
    (total_fish : ℚ) * (1 - male_fraction) = 15 :=
by sorry

end ariels_female_fish_l627_62703


namespace smallest_candy_count_l627_62723

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m ≤ 999 ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0 → m ≥ n) ∧
  n = 128 :=
by sorry

end smallest_candy_count_l627_62723


namespace sum_of_ages_l627_62714

/-- Given that in 5 years Nacho will be three times older than Divya, 
    and Divya is currently 5 years old, prove that the sum of their 
    current ages is 30 years. -/
theorem sum_of_ages (nacho_age divya_age : ℕ) : 
  divya_age = 5 → 
  nacho_age + 5 = 3 * (divya_age + 5) → 
  nacho_age + divya_age = 30 := by
sorry

end sum_of_ages_l627_62714


namespace martha_router_time_l627_62726

theorem martha_router_time (x : ℝ) 
  (router_time : x > 0)
  (hold_time : ℝ)
  (hold_time_def : hold_time = 6 * x)
  (yelling_time : ℝ)
  (yelling_time_def : yelling_time = 3 * x)
  (total_time : x + hold_time + yelling_time = 100) :
  x = 10 := by
sorry

end martha_router_time_l627_62726


namespace marble_distribution_theorem_l627_62759

/-- The number of ways to distribute marbles to students under specific conditions -/
def marbleDistributionWays : ℕ := 3150

/-- The total number of marbles -/
def totalMarbles : ℕ := 12

/-- The number of red marbles -/
def redMarbles : ℕ := 3

/-- The number of blue marbles -/
def blueMarbles : ℕ := 4

/-- The number of green marbles -/
def greenMarbles : ℕ := 5

/-- The total number of students -/
def totalStudents : ℕ := 12

theorem marble_distribution_theorem :
  marbleDistributionWays = 3150 ∧
  totalMarbles = redMarbles + blueMarbles + greenMarbles ∧
  totalStudents = totalMarbles ∧
  ∃ (distribution : Fin totalStudents → Fin 3),
    (∃ (i j : Fin totalStudents), i ≠ j ∧ distribution i = distribution j) ∧
    (∃ (k : Fin totalStudents), distribution k = 2) :=
by sorry

end marble_distribution_theorem_l627_62759


namespace cubic_equation_has_real_root_l627_62715

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, a * x^3 + a * x + b = 0 := by sorry

end cubic_equation_has_real_root_l627_62715


namespace combined_body_is_pentahedron_l627_62778

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  edge_length : ℝ

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- Represents the new geometric body formed by combining a regular quadrangular pyramid and a regular tetrahedron -/
structure CombinedBody where
  pyramid : RegularQuadrangularPyramid
  tetrahedron : RegularTetrahedron

/-- Defines the property of being a pentahedron -/
def is_pentahedron (body : CombinedBody) : Prop := sorry

theorem combined_body_is_pentahedron 
  (pyramid : RegularQuadrangularPyramid) 
  (tetrahedron : RegularTetrahedron) 
  (h : pyramid.edge_length = tetrahedron.edge_length) : 
  is_pentahedron (CombinedBody.mk pyramid tetrahedron) :=
sorry

end combined_body_is_pentahedron_l627_62778


namespace fraction_problem_l627_62795

theorem fraction_problem (x : ℚ) : 150 * x = 37 + 1/2 → x = 1/4 := by
  sorry

end fraction_problem_l627_62795


namespace soccer_team_lineups_l627_62768

/-- The number of possible lineups for a soccer team -/
def number_of_lineups (total_players : ℕ) (goalkeeper : ℕ) (defenders : ℕ) (others : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) defenders) * (Nat.choose (total_players - 1 - defenders) others)

/-- Theorem: The number of possible lineups for a soccer team of 18 players,
    with 1 goalkeeper, 4 defenders, and 4 other players is 30,544,200 -/
theorem soccer_team_lineups :
  number_of_lineups 18 1 4 4 = 30544200 := by
  sorry

end soccer_team_lineups_l627_62768


namespace bus_departure_theorem_l627_62793

/-- Represents the rules for bus departure and current occupancy -/
structure BusOccupancy where
  min_departure : Nat
  max_departure : Nat
  current_occupancy : Nat
  departure_rule : min_departure > 15 ∧ max_departure ≤ 30
  occupancy_valid : current_occupancy < min_departure

/-- Calculates the number of additional people needed for the bus to depart -/
def additional_people_needed (bus : BusOccupancy) : Nat :=
  bus.min_departure - bus.current_occupancy

/-- Theorem stating that for a bus with specific occupancy rules and current state,
    the number of additional people needed is 7 -/
theorem bus_departure_theorem (bus : BusOccupancy)
    (h1 : bus.min_departure = 16)
    (h2 : bus.current_occupancy = 9) :
    additional_people_needed bus = 7 := by
  sorry

#eval additional_people_needed ⟨16, 30, 9, by simp, by simp⟩

end bus_departure_theorem_l627_62793


namespace largest_possible_a_l627_62789

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c + 1)
  (h3 : c < 5 * d - 2)
  (h4 : d ≤ 50)
  (h5 : ∃ k : ℕ, d = 5 * k) :
  a ≤ 1481 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1481 ∧
    a' < 3 * b' ∧
    b' < 2 * c' + 1 ∧
    c' < 5 * d' - 2 ∧
    d' ≤ 50 ∧
    ∃ k : ℕ, d' = 5 * k :=
by sorry

end largest_possible_a_l627_62789


namespace definite_integral_equals_ln3_minus_ln2_plus_1_l627_62733

theorem definite_integral_equals_ln3_minus_ln2_plus_1 :
  let a : ℝ := 2 * Real.arctan (1 / 3)
  let b : ℝ := 2 * Real.arctan (1 / 2)
  let f (x : ℝ) := 1 / (Real.sin x * (1 - Real.sin x))
  ∫ x in a..b, f x = Real.log 3 - Real.log 2 + 1 := by sorry

end definite_integral_equals_ln3_minus_ln2_plus_1_l627_62733


namespace fish_catch_problem_l627_62798

theorem fish_catch_problem (total_fish : ℕ) 
  (first_fisherman_carp_ratio : ℚ) (second_fisherman_perch_ratio : ℚ) :
  total_fish = 70 ∧ 
  first_fisherman_carp_ratio = 5 / 9 ∧ 
  second_fisherman_perch_ratio = 7 / 17 →
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch * first_fisherman_carp_ratio = 
      second_catch * second_fisherman_perch_ratio ∧
    first_catch = 36 ∧ 
    second_catch = 34 := by
  sorry

#check fish_catch_problem

end fish_catch_problem_l627_62798


namespace difference_not_arithmetic_for_k_ge_4_l627_62711

/-- Two geometric sequences -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The difference sequence -/
def difference_sequence (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n - b n

/-- Arithmetic sequence with non-zero common difference -/
def is_arithmetic_with_nonzero_diff (c : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, c (n + 1) - c n = d

theorem difference_not_arithmetic_for_k_ge_4 (a b : ℕ → ℝ) (k : ℕ) 
  (h1 : geometric_sequence a)
  (h2 : geometric_sequence b)
  (h3 : k ≥ 4) :
  ¬ is_arithmetic_with_nonzero_diff (difference_sequence a b) :=
sorry

end difference_not_arithmetic_for_k_ge_4_l627_62711


namespace smallest_n_without_quadratic_number_l627_62790

def isQuadraticNumber (x : ℝ) : Prop :=
  ∃ (a b c : ℤ), a ≠ 0 ∧ 
  (|a| ≤ 10 ∧ |a| ≥ 1) ∧ 
  (|b| ≤ 10 ∧ |b| ≥ 1) ∧ 
  (|c| ≤ 10 ∧ |c| ≥ 1) ∧ 
  a * x^2 + b * x + c = 0

def hasQuadraticNumber (l r : ℝ) : Prop :=
  ∃ x, l < x ∧ x < r ∧ isQuadraticNumber x

def noQuadraticNumber (n : ℕ) : Prop :=
  ¬(hasQuadraticNumber (n - 1/3) n) ∨ ¬(hasQuadraticNumber n (n + 1/3))

theorem smallest_n_without_quadratic_number :
  (∀ m : ℕ, m < 11 → ¬(noQuadraticNumber m)) ∧ noQuadraticNumber 11 :=
sorry

end smallest_n_without_quadratic_number_l627_62790


namespace tram_route_difference_l627_62782

/-- Represents a point on the circular tram line -/
inductive TramStop
| Circus
| Park
| Zoo

/-- Represents the distance between two points on the tram line -/
def distance (a b : TramStop) : ℝ := sorry

/-- The total circumference of the tram line -/
def circumference : ℝ := sorry

theorem tram_route_difference :
  let park_to_zoo := distance TramStop.Park TramStop.Zoo
  let park_to_circus_via_zoo := distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus
  let park_to_circus_direct := distance TramStop.Park TramStop.Circus
  
  -- The distance from Park to Zoo via Circus is three times longer than the direct route
  distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus + distance TramStop.Circus TramStop.Park = 3 * park_to_zoo →
  
  -- The distance from Circus to Zoo via Park is half as long as the direct route
  distance TramStop.Circus TramStop.Park + park_to_zoo = (1/2) * distance TramStop.Circus TramStop.Zoo →
  
  -- The difference between the longer and shorter routes from Park to Circus is 1/12 of the total circumference
  park_to_circus_via_zoo - park_to_circus_direct = (1/12) * circumference :=
by sorry

end tram_route_difference_l627_62782


namespace simplify_product_of_square_roots_l627_62705

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (5 * 2 * x) * Real.sqrt (x^3 * 5^3) = 25 * x^2 * Real.sqrt 2 := by
  sorry

end simplify_product_of_square_roots_l627_62705


namespace geometry_theorem_l627_62739

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms
axiom different_lines {l1 l2 : Line} : l1 ≠ l2
axiom different_planes {p1 p2 : Plane} : p1 ≠ p2

-- Theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) :
  (perpendicular m n ∧ perpendicularLP m α ∧ ¬subset n α → parallel n α) ∧
  (perpendicular m n ∧ perpendicularLP m α ∧ perpendicularLP n β → perpendicularPP α β) :=
by sorry

end geometry_theorem_l627_62739


namespace prob_e_value_l627_62753

-- Define the probability measure
variable (p : Set α → ℝ)

-- Define events e and f
variable (e f : Set α)

-- Define the conditions
variable (h1 : p f = 75)
variable (h2 : p (e ∩ f) = 75)
variable (h3 : p f / p e = 3)

-- Theorem statement
theorem prob_e_value : p e = 25 := by
  sorry

end prob_e_value_l627_62753


namespace intersection_parallel_line_equation_l627_62708

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line. -/
theorem intersection_parallel_line_equation 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop)
  (l_parallel : ℝ → ℝ → Prop)
  (result_line : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x - 4 * y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ 4 * x - 2 * y + 7 = 0)
  (h_result : ∀ x y, result_line x y ↔ 2 * x - y - 18 = 0) :
  ∃ (x₀ y₀ : ℝ), 
    (l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧ 
    (∃ (k : ℝ), ∀ x y, result_line x y ↔ l_parallel (x - x₀) (y - y₀)) ∧
    result_line x₀ y₀ := by
  sorry

end intersection_parallel_line_equation_l627_62708


namespace fibonacci_identities_l627_62734

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_identities (n : ℕ) : 
  (fib (2*n + 1) * fib (2*n - 1) = fib (2*n)^2 + 1) ∧ 
  (fib (2*n + 1)^2 + fib (2*n - 1)^2 + 1 = 3 * fib (2*n + 1) * fib (2*n - 1)) := by
  sorry

end fibonacci_identities_l627_62734


namespace sin_cos_sixth_power_sum_l627_62736

theorem sin_cos_sixth_power_sum (α : ℝ) (h : Real.sin (2 * α) = 1/2) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 = 5/8 := by
  sorry

end sin_cos_sixth_power_sum_l627_62736


namespace simplify_expression_l627_62721

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := by
  sorry

end simplify_expression_l627_62721


namespace smallest_number_in_sequence_l627_62702

theorem smallest_number_in_sequence (x y z t : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  t = (y + z) / 3 →
  (x + y + z + t) / 4 = 220 →
  x = 2640 / 43 := by
sorry

end smallest_number_in_sequence_l627_62702


namespace circle_center_radius_sum_l627_62780

/-- Given a circle C with equation x^2 + 6x + 36 = -y^2 - 8y + 45,
    prove that its center coordinates (a, b) and radius r satisfy a + b + r = -7 + √34 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 6*x + 36 = -y^2 - 8*y + 45) →
  (∀ x y, (x + 3)^2 + (y + 4)^2 = 34) →
  (a = -3 ∧ b = -4) →
  r = Real.sqrt 34 →
  a + b + r = -7 + Real.sqrt 34 :=
by sorry

end circle_center_radius_sum_l627_62780


namespace painting_time_with_break_l627_62709

/-- The time it takes Doug and Dave to paint a room together, including a break -/
theorem painting_time_with_break (doug_time dave_time break_time : ℝ) 
  (h_doug : doug_time = 4)
  (h_dave : dave_time = 6)
  (h_break : break_time = 2) : 
  ∃ s : ℝ, s = 22 / 5 ∧ 
  (1 / doug_time + 1 / dave_time) * (s - break_time) = 1 := by
  sorry

end painting_time_with_break_l627_62709


namespace sum_of_squares_of_roots_l627_62710

theorem sum_of_squares_of_roots (q r s : ℝ) : 
  (3 * q^3 - 4 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 4 * r^2 + 6 * r + 15 = 0) →
  (3 * s^3 - 4 * s^2 + 6 * s + 15 = 0) →
  q^2 + r^2 + s^2 = -20/9 := by
sorry

end sum_of_squares_of_roots_l627_62710


namespace product_xy_equals_sqrt_30_6_l627_62707

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

/-- The product of x and y in the parallelogram EFGH -/
def product_xy (p : Parallelogram) : ℝ → ℝ → ℝ := fun x y => x * y

/-- Theorem: The product of x and y in the given parallelogram is √(30.6) -/
theorem product_xy_equals_sqrt_30_6 (p : Parallelogram) 
  (h1 : p.EF = 54)
  (h2 : ∀ x, p.FG x = 8 * x^2 + 2)
  (h3 : ∀ y, p.GH y = 5 * y^2 + 20)
  (h4 : p.HE = 38) :
  ∃ x y, product_xy p x y = Real.sqrt 30.6 := by
  sorry

#check product_xy_equals_sqrt_30_6

end product_xy_equals_sqrt_30_6_l627_62707


namespace roof_dimension_difference_l627_62783

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  area : ℝ
  length_width_ratio : length = 4 * width
  area_equation : area = length * width

/-- The difference between the length and width of the roof -/
def length_width_difference (roof : RoofDimensions) : ℝ :=
  roof.length - roof.width

/-- Theorem stating the approximate difference between length and width -/
theorem roof_dimension_difference : 
  ∃ (roof : RoofDimensions), 
    roof.area = 675 ∧ 
    (abs (length_width_difference roof - 38.97) < 0.01) := by
  sorry

end roof_dimension_difference_l627_62783


namespace quadratic_root_problem_l627_62706

theorem quadratic_root_problem (a : ℝ) : 
  (2^2 + 2 - a = 0) → 
  (∃ x : ℝ, x ≠ 2 ∧ x^2 + x - a = 0) → 
  ((-3)^2 + (-3) - a = 0) :=
by sorry

end quadratic_root_problem_l627_62706


namespace tangent_line_parallel_points_l627_62764

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y → (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end tangent_line_parallel_points_l627_62764


namespace flight_savings_l627_62712

/-- Calculates the savings by choosing the cheaper flight between two airlines with given prices and discounts -/
theorem flight_savings (delta_price united_price : ℝ) (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  let delta_final := delta_price * (1 - delta_discount)
  let united_final := united_price * (1 - united_discount)
  min delta_final united_final = delta_final →
  united_final - delta_final = 90 := by
sorry


end flight_savings_l627_62712


namespace range_of_f_l627_62724

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f : {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by sorry

end range_of_f_l627_62724


namespace smallest_interesting_number_l627_62781

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ k m : ℕ, 2 * n = k^2 ∧ 15 * n = m^3

/-- 1800 is the smallest interesting natural number. -/
theorem smallest_interesting_number : 
  IsInteresting 1800 ∧ ∀ n < 1800, ¬IsInteresting n := by
  sorry

#check smallest_interesting_number

end smallest_interesting_number_l627_62781


namespace inequality_proof_l627_62747

theorem inequality_proof (x y : ℝ) (n : ℕ+) (hx : x > 0) (hy : y > 0) :
  (x^n.val / (1 + x^2)) + (y^n.val / (1 + y^2)) ≤ (x^n.val + y^n.val) / (1 + x*y) := by
  sorry

end inequality_proof_l627_62747


namespace pages_read_difference_l627_62741

theorem pages_read_difference (beatrix_pages : ℕ) (cristobal_pages : ℕ) : 
  beatrix_pages = 704 → 
  cristobal_pages = 15 + 3 * beatrix_pages → 
  cristobal_pages - beatrix_pages = 1423 := by
  sorry

end pages_read_difference_l627_62741


namespace line_segment_ratios_l627_62763

/-- Given points A, B, C on a straight line with AC : BC = m : n,
    prove the ratios AC : AB and BC : AB -/
theorem line_segment_ratios
  (A B C : ℝ) -- Points on a real line
  (m n : ℕ) -- Natural numbers for the ratio
  (h_line : (A ≤ B ∧ B ≤ C) ∨ (A ≤ C ∧ C ≤ B) ∨ (B ≤ A ∧ A ≤ C)) -- Points are on a line
  (h_ratio : |C - A| / |C - B| = m / n) : -- Given ratio
  (∃ (r₁ r₂ : ℚ),
    (r₁ = m / (m + n) ∧ r₂ = n / (m + n)) ∨
    (r₁ = m / (n - m) ∧ r₂ = n / (n - m)) ∨
    (m = n ∧ r₁ = 1 / 2 ∧ r₂ = 1 / 2)) ∧
  (|A - C| / |A - B| = r₁ ∧ |B - C| / |A - B| = r₂) :=
sorry

end line_segment_ratios_l627_62763


namespace sin_45_degrees_l627_62718

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end sin_45_degrees_l627_62718


namespace sin_sum_inverse_trig_functions_l627_62788

theorem sin_sum_inverse_trig_functions :
  Real.sin (Real.arcsin (4/5) + Real.arctan (3/2) + Real.arccos (1/3)) = (17 - 12 * Real.sqrt 2) / (15 * Real.sqrt 13) := by
  sorry

end sin_sum_inverse_trig_functions_l627_62788


namespace help_sign_white_area_l627_62732

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a letter painted with 1-unit wide strokes -/
def letterArea (letter : Char) : ℕ :=
  match letter with
  | 'H' => 13
  | 'E' => 9
  | 'L' => 8
  | 'P' => 10
  | _ => 0

/-- Calculates the total area of a word painted with 1-unit wide strokes -/
def wordArea (word : String) : ℕ :=
  word.toList.map letterArea |> List.sum

/-- Theorem: The white area of the sign with "HELP" painted is 35 square units -/
theorem help_sign_white_area (sign : SignDimensions) 
  (h1 : sign.width = 15) 
  (h2 : sign.height = 5) : 
  sign.width * sign.height - wordArea "HELP" = 35 := by
  sorry

end help_sign_white_area_l627_62732


namespace john_learning_time_l627_62761

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days John needs to learn all vowels -/
def total_days : ℕ := 15

/-- The number of days John needs to learn one alphabet (vowel) -/
def days_per_alphabet : ℚ := total_days / num_vowels

theorem john_learning_time : days_per_alphabet = 3 := by
  sorry

end john_learning_time_l627_62761


namespace basketball_team_size_l627_62758

theorem basketball_team_size (total_points : ℕ) (min_score : ℕ) (max_score : ℕ) :
  total_points = 100 →
  min_score = 7 →
  max_score = 23 →
  ∃ (team_size : ℕ) (scores : List ℕ),
    team_size = 12 ∧
    scores.length = team_size ∧
    scores.sum = total_points ∧
    (∀ s ∈ scores, min_score ≤ s ∧ s ≤ max_score) :=
by sorry

end basketball_team_size_l627_62758


namespace test_question_count_l627_62756

/-- Given a test with the following properties:
  * The test is worth 100 points
  * There are 2-point and 4-point questions
  * There are 30 two-point questions
  Prove that the total number of questions is 40 -/
theorem test_question_count (total_points : ℕ) (two_point_count : ℕ) :
  total_points = 100 →
  two_point_count = 30 →
  ∃ (four_point_count : ℕ),
    total_points = 2 * two_point_count + 4 * four_point_count ∧
    two_point_count + four_point_count = 40 :=
by sorry

end test_question_count_l627_62756


namespace average_battery_lifespan_l627_62746

def battery_lifespans : List ℝ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

theorem average_battery_lifespan :
  (List.sum battery_lifespans) / (List.length battery_lifespans) = 28 := by
  sorry

end average_battery_lifespan_l627_62746


namespace circle_center_sum_l627_62737

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 15 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 8*y - 15)) ∧ 
               h + k = 7 :=
by sorry

end circle_center_sum_l627_62737


namespace pet_store_cats_count_l627_62785

theorem pet_store_cats_count (siamese : Float) (house : Float) (added : Float) :
  siamese = 13.0 → house = 5.0 → added = 10.0 →
  siamese + house + added = 28.0 := by
  sorry

end pet_store_cats_count_l627_62785


namespace hotel_supplies_theorem_l627_62773

/-- The greatest number of bathrooms that can be stocked identically with given supplies -/
def max_bathrooms (toilet_paper soap towels shower_gel : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (Nat.gcd toilet_paper soap) towels) shower_gel

/-- Theorem stating that the maximum number of bathrooms that can be stocked
    with the given supplies is 6 -/
theorem hotel_supplies_theorem :
  max_bathrooms 36 18 24 12 = 6 := by
  sorry

end hotel_supplies_theorem_l627_62773


namespace complex_magnitude_of_special_z_l627_62786

theorem complex_magnitude_of_special_z : 
  let i : ℂ := Complex.I
  let z : ℂ := -i^2022 + i
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_magnitude_of_special_z_l627_62786


namespace work_completion_theorem_l627_62719

theorem work_completion_theorem 
  (total_work : ℕ) 
  (days_group1 : ℕ) 
  (men_group1 : ℕ) 
  (days_group2 : ℕ) : 
  men_group1 * days_group1 = total_work → 
  total_work = days_group2 * (total_work / days_group2) → 
  men_group1 = 10 → 
  days_group1 = 35 → 
  days_group2 = 50 → 
  total_work / days_group2 = 7 :=
by
  sorry

#check work_completion_theorem

end work_completion_theorem_l627_62719


namespace divisible_by_six_and_inductive_step_l627_62749

theorem divisible_by_six_and_inductive_step (n : ℕ) :
  6 ∣ (n * (n + 1) * (2 * n + 1)) ∧
  (∀ k : ℕ, (k + 1) * ((k + 1) + 1) * (2 * (k + 1) + 1) = k * (k + 1) * (2 * k + 1) + 6 * (k + 1)^2) :=
by sorry

end divisible_by_six_and_inductive_step_l627_62749


namespace scooter_initial_price_l627_62744

/-- The initial purchase price of a scooter, given the repair cost, selling price, and gain percentage. -/
theorem scooter_initial_price (repair_cost selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : repair_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ (initial_price : ℝ), 
    selling_price = (1 + gain_percent / 100) * (initial_price + repair_cost) ∧ 
    initial_price = 800 := by
  sorry

end scooter_initial_price_l627_62744


namespace safari_count_l627_62792

/-- The total number of animals counted during the safari --/
def total_animals (antelopes rabbits hyenas wild_dogs leopards : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards

/-- Theorem stating the total number of animals counted during the safari --/
theorem safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards = 605 := by
  sorry


end safari_count_l627_62792


namespace equation_root_range_l627_62799

theorem equation_root_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + a^2 - 3 = 0) → 
  -Real.sqrt 3 < a ∧ a ≤ 2 := by
sorry

end equation_root_range_l627_62799


namespace correct_number_increase_l627_62731

theorem correct_number_increase : 
  ∀ (a b c d : ℕ), 
    (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) →
    (a + (b + 1) * c - d = 36) ∧
    (¬(a + 1 + b * c - d = 36)) ∧
    (¬(a + b * (c + 1) - d = 36)) ∧
    (¬(a + b * c - (d + 1) = 36)) :=
by sorry

end correct_number_increase_l627_62731


namespace negative_64_to_two_thirds_power_l627_62755

theorem negative_64_to_two_thirds_power (x : ℝ) : x = (-64)^(2/3) → x = 16 := by
  sorry

end negative_64_to_two_thirds_power_l627_62755


namespace tara_dad_attendance_l627_62771

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The number of games Tara's dad attended in the second year -/
def games_attended_second_year : ℕ := 14

/-- The difference in games attended between the first and second year -/
def games_difference : ℕ := 4

/-- The percentage of games Tara's dad attended in the first year -/
def attendance_percentage : ℚ := 90

theorem tara_dad_attendance :
  (games_attended_second_year + games_difference) / games_per_year * 100 = attendance_percentage := by
  sorry

end tara_dad_attendance_l627_62771


namespace work_completion_time_l627_62735

theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 1 / a + 1 / b = 0.5 / 10) : b = 60 := by
  sorry

end work_completion_time_l627_62735


namespace simplify_polynomial_l627_62791

theorem simplify_polynomial (p : ℝ) : 
  (3 * p^3 - 5*p + 6) + (4 - 6*p^2 + 2*p) = 3*p^3 - 6*p^2 - 3*p + 10 := by
  sorry

end simplify_polynomial_l627_62791


namespace inequality_solution_l627_62772

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4) ↔ (x ∈ Set.Ioo (-2) 2 ∪ Set.Ici 3) :=
sorry

end inequality_solution_l627_62772


namespace eighth_term_value_l627_62704

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem eighth_term_value 
  (a : ℕ → ℕ) 
  (h_seq : RecurrenceSequence a) 
  (h_seventh : a 7 = 120) : 
  a 8 = 194 := by
sorry

end eighth_term_value_l627_62704


namespace line_constant_value_l627_62729

/-- Given a line passing through points (m, n) and (m + 2, n + 0.5) with equation x = k * y + 5, prove that k = 4 -/
theorem line_constant_value (m n k : ℝ) : 
  (m = k * n + 5) ∧ (m + 2 = k * (n + 0.5) + 5) → k = 4 := by
  sorry

end line_constant_value_l627_62729


namespace find_y_value_l627_62751

theorem find_y_value : ∃ y : ℚ, 
  (1/4 : ℚ) * ((y + 8) + (7*y + 4) + (3*y + 9) + (4*y + 5)) = 6*y - 10 → y = 22/3 := by
  sorry

end find_y_value_l627_62751


namespace expression_equality_l627_62769

theorem expression_equality : 7^2 - 2*(5) + 4^2 / 2 = 47 := by
  sorry

end expression_equality_l627_62769


namespace no_intersection_in_S_l627_62774

-- Define the set S inductively
inductive S : (Real → Real) → Prop
  | base : S (fun x ↦ x)
  | sub {f} : S f → S (fun x ↦ x - f x)
  | add {f} : S f → S (fun x ↦ x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : Real → Real), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end no_intersection_in_S_l627_62774


namespace simplify_expression_l627_62727

theorem simplify_expression : 
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by
sorry

end simplify_expression_l627_62727


namespace basketball_shot_probability_l627_62700

theorem basketball_shot_probability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < 1) (hbb : b < 1) (hcb : c < 1) (sum_prob : a + b + c = 1) (expected_value : 3*a + 2*b = 2) :
  (2/a + 1/(3*b)) ≥ 16/3 := by
sorry

end basketball_shot_probability_l627_62700


namespace geometric_sequence_sum_l627_62716

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 9 →
  a 5 + a 6 = 81 := by
sorry

end geometric_sequence_sum_l627_62716


namespace no_twelve_consecutive_primes_in_ap_l627_62765

theorem no_twelve_consecutive_primes_in_ap (a d : ℕ) (h_d : d < 2000) :
  ¬ ∀ k : Fin 12, Nat.Prime (a + k.val * d) := by
  sorry

end no_twelve_consecutive_primes_in_ap_l627_62765


namespace min_value_triangle_sides_l627_62787

theorem min_value_triangle_sides (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 9) 
  (htri : x + y > z ∧ y + z > x ∧ z + x > y) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end min_value_triangle_sides_l627_62787


namespace book_sales_ratio_l627_62796

theorem book_sales_ratio : 
  ∀ (wednesday thursday friday : ℕ),
  wednesday = 15 →
  thursday = 3 * wednesday →
  wednesday + thursday + friday = 69 →
  friday * 5 = thursday :=
λ wednesday thursday friday hw ht htot =>
  sorry

end book_sales_ratio_l627_62796


namespace sum_of_smaller_radii_eq_original_radius_l627_62757

/-- A triangle with an inscribed circle and three smaller triangles formed by tangents -/
structure InscribedCircleTriangle where
  /-- The radius of the circle inscribed in the original triangle -/
  r : ℝ
  /-- The radius of the circle inscribed in the first smaller triangle -/
  r₁ : ℝ
  /-- The radius of the circle inscribed in the second smaller triangle -/
  r₂ : ℝ
  /-- The radius of the circle inscribed in the third smaller triangle -/
  r₃ : ℝ
  /-- Ensure all radii are positive -/
  r_pos : r > 0
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0
  r₃_pos : r₃ > 0

/-- The sum of the radii of the inscribed circles in the smaller triangles
    equals the radius of the inscribed circle in the original triangle -/
theorem sum_of_smaller_radii_eq_original_radius (t : InscribedCircleTriangle) :
  t.r₁ + t.r₂ + t.r₃ = t.r := by
  sorry

end sum_of_smaller_radii_eq_original_radius_l627_62757


namespace colin_average_time_l627_62730

/-- Represents Colin's running times for each mile -/
def colinTimes : List ℕ := [6, 5, 5, 4]

/-- The number of miles Colin ran -/
def totalMiles : ℕ := colinTimes.length

/-- Calculates the average time per mile -/
def averageTime : ℚ := (colinTimes.sum : ℚ) / totalMiles

theorem colin_average_time :
  averageTime = 5 := by sorry

end colin_average_time_l627_62730


namespace trajectory_of_midpoint_l627_62760

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the midpoint relationship
def is_midpoint (mx my px py : ℝ) : Prop :=
  mx = (px + 4) / 2 ∧ my = py / 2

-- Theorem statement
theorem trajectory_of_midpoint :
  ∀ (x y : ℝ), 
    (∃ (x1 y1 : ℝ), on_ellipse x1 y1 ∧ is_midpoint x y x1 y1) →
    (x - 2)^2 + 4*y^2 = 1 :=
by sorry

end trajectory_of_midpoint_l627_62760


namespace xy_sum_zero_l627_62713

theorem xy_sum_zero (x y : ℝ) :
  (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1 →
  x + y = 0 ∧ ∀ z, ((x + Real.sqrt (x^2 + 1)) * (z + Real.sqrt (z^2 + 1)) = 1 → x + z = 0) :=
by sorry

end xy_sum_zero_l627_62713


namespace range_of_a_range_of_t_l627_62752

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Statement for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a ≥ f x} = {a : ℝ | a ≥ -5/2} := by sorry

-- Statement for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ -t^2 - 5/2*t - 1} = 
  {t : ℝ | t ≥ 1/2 ∨ t ≤ -3} := by sorry

end range_of_a_range_of_t_l627_62752


namespace cube_volume_after_increase_l627_62754

theorem cube_volume_after_increase (surface_area : ℝ) (increase_factor : ℝ) : 
  surface_area = 864 → increase_factor = 1.5 → 
  (increase_factor * (surface_area / 6).sqrt) ^ 3 = 5832 := by
  sorry

end cube_volume_after_increase_l627_62754


namespace flight_speed_l627_62738

/-- Given a flight distance of 256 miles and a flight time of 8 hours,
    prove that the speed is 32 miles per hour. -/
theorem flight_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 256) 
    (h2 : time = 8) 
    (h3 : speed = distance / time) : speed = 32 := by
  sorry

end flight_speed_l627_62738


namespace binomial_7_choose_4_l627_62728

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_choose_4_l627_62728


namespace jeff_total_hours_l627_62766

/-- Represents Jeff's weekly schedule --/
structure JeffSchedule where
  facebook_hours_per_day : ℕ
  weekend_work_ratio : ℕ
  twitter_hours_per_weekend_day : ℕ
  instagram_hours_per_weekday : ℕ
  weekday_work_ratio : ℕ

/-- Calculates Jeff's total hours spent on work, Twitter, and Instagram in a week --/
def total_hours (schedule : JeffSchedule) : ℕ :=
  let weekend_work_hours := 2 * (schedule.facebook_hours_per_day / schedule.weekend_work_ratio)
  let weekday_work_hours := 5 * (4 * (schedule.facebook_hours_per_day + schedule.instagram_hours_per_weekday))
  let twitter_hours := 2 * schedule.twitter_hours_per_weekend_day
  let instagram_hours := 5 * schedule.instagram_hours_per_weekday
  weekend_work_hours + weekday_work_hours + twitter_hours + instagram_hours

/-- Theorem stating Jeff's total hours in a week --/
theorem jeff_total_hours : 
  ∀ (schedule : JeffSchedule),
    schedule.facebook_hours_per_day = 3 ∧
    schedule.weekend_work_ratio = 3 ∧
    schedule.twitter_hours_per_weekend_day = 2 ∧
    schedule.instagram_hours_per_weekday = 1 ∧
    schedule.weekday_work_ratio = 4 →
    total_hours schedule = 91 := by
  sorry

end jeff_total_hours_l627_62766


namespace expression_equals_one_l627_62748

theorem expression_equals_one (x : ℝ) 
  (h1 : x^3 + 2*x + 1 ≠ 0) 
  (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ((((x+2)^2 * (x^2-x+2)^2) / (x^3+2*x+1)^2)^3 * 
   (((x-2)^2 * (x^2+x+2)^2) / (x^3-2*x-1)^2)^3) = 1 := by
  sorry

end expression_equals_one_l627_62748


namespace ellipse_slope_theorem_l627_62742

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define points A and B as endpoints of minor axis
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define line l passing through (0,1)
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * x

-- Define points C and D on the ellipse and line l
def C (k : ℝ) : ℝ × ℝ := sorry
def D (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes k₁ and k₂
def k₁ (k : ℝ) : ℝ := sorry
def k₂ (k : ℝ) : ℝ := sorry

theorem ellipse_slope_theorem (k : ℝ) :
  (∀ x y, ellipse x y → line_l k x y → (x, y) = C k ∨ (x, y) = D k) →
  k₁ k / k₂ k = 2 →
  k = 3 := by sorry

end ellipse_slope_theorem_l627_62742


namespace total_legos_after_winning_l627_62779

def initial_legos : ℕ := 2080
def won_legos : ℕ := 17

theorem total_legos_after_winning :
  initial_legos + won_legos = 2097 := by sorry

end total_legos_after_winning_l627_62779


namespace circle_parallel_lines_distance_l627_62784

-- Define the circle
variable (r : ℝ) -- radius of the circle

-- Define the chords
def chord1 : ℝ := 45
def chord2 : ℝ := 49
def chord3 : ℝ := 49
def chord4 : ℝ := 45

-- Define the distance between adjacent parallel lines
def d : ℝ := 2.8

-- State the theorem
theorem circle_parallel_lines_distance :
  ∃ (r : ℝ), 
    r > 0 ∧
    chord1 = 45 ∧
    chord2 = 49 ∧
    chord3 = 49 ∧
    chord4 = 45 ∧
    d = 2.8 ∧
    r^2 = 506.25 + (1/4) * d^2 ∧
    r^2 = 600.25 + (49/4) * d^2 :=
by sorry

end circle_parallel_lines_distance_l627_62784


namespace max_yellow_apples_max_total_apples_l627_62701

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure ApplesTaken :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stopCondition (taken : ApplesTaken) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 11, yellow := 14, red := 19 }

/-- Theorem stating the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples (taken : ApplesTaken) 
  (h : taken.yellow ≤ initialBasket.yellow) 
  (h_stop : ¬stopCondition taken) : 
  taken.yellow ≤ 14 :=
sorry

/-- Theorem stating the maximum total number of apples that can be taken -/
theorem max_total_apples (taken : ApplesTaken) 
  (h_green : taken.green ≤ initialBasket.green)
  (h_yellow : taken.yellow ≤ initialBasket.yellow)
  (h_red : taken.red ≤ initialBasket.red)
  (h_stop : ¬stopCondition taken) :
  taken.green + taken.yellow + taken.red ≤ 42 :=
sorry

end max_yellow_apples_max_total_apples_l627_62701


namespace sin_140_cos_50_plus_sin_130_cos_40_eq_1_l627_62717

theorem sin_140_cos_50_plus_sin_130_cos_40_eq_1 :
  Real.sin (140 * π / 180) * Real.cos (50 * π / 180) +
  Real.sin (130 * π / 180) * Real.cos (40 * π / 180) = 1 := by
  sorry

end sin_140_cos_50_plus_sin_130_cos_40_eq_1_l627_62717


namespace min_value_quadratic_l627_62725

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 10 := by
  sorry

end min_value_quadratic_l627_62725


namespace function_periodic_l627_62720

/-- A function satisfying certain symmetry properties is periodic -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 3) = f (3 - x))
  (h2 : ∀ x : ℝ, f (x + 11) = f (11 - x)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
sorry

end function_periodic_l627_62720


namespace f_even_and_increasing_l627_62777

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end f_even_and_increasing_l627_62777


namespace first_day_over_500_l627_62722

/-- Represents the number of markers Liam has on a given day -/
def markers (day : ℕ) : ℕ :=
  if day = 1 then 5
  else if day = 2 then 10
  else 5 * 3^(day - 2)

/-- The day of the week as a number from 1 to 7 -/
def dayOfWeek (day : ℕ) : ℕ :=
  (day - 1) % 7 + 1

theorem first_day_over_500 :
  ∃ d : ℕ, markers d > 500 ∧ 
    ∀ k < d, markers k ≤ 500 ∧
    dayOfWeek d = 6 :=
  sorry

end first_day_over_500_l627_62722


namespace kabadi_players_count_l627_62740

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 25

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 35

theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_players :=
by
  sorry

#check kabadi_players_count

end kabadi_players_count_l627_62740


namespace hot_drink_price_range_l627_62775

/-- Represents the price increase in yuan -/
def price_increase : ℝ → ℝ := λ x => x

/-- Represents the new price of a hot drink in yuan -/
def new_price : ℝ → ℝ := λ x => 1.5 + price_increase x

/-- Represents the daily sales volume as a function of price increase -/
def daily_sales : ℝ → ℝ := λ x => 800 - 20 * (10 * price_increase x)

/-- Represents the daily profit as a function of price increase -/
def daily_profit : ℝ → ℝ := λ x => (new_price x - 0.9) * daily_sales x

theorem hot_drink_price_range :
  ∃ (lower upper : ℝ), lower = 1.9 ∧ upper = 4.5 ∧
  ∀ x, daily_profit x ≥ 720 ↔ new_price x ∈ Set.Icc lower upper :=
by sorry

end hot_drink_price_range_l627_62775


namespace unique_perfect_square_sum_l627_62797

theorem unique_perfect_square_sum (p : Nat) (hp : p.Prime ∧ p > 2) :
  ∃! n : Nat, n > 0 ∧ ∃ k : Nat, n^2 + n*p = k^2 :=
by
  use ((p - 1)^2) / 4
  sorry

end unique_perfect_square_sum_l627_62797


namespace machine_doesnt_require_repair_l627_62794

/-- Represents a weighing machine for food portions -/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair based on its measurements -/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.unreadable_deviation_bound ≥ m.max_deviation

theorem machine_doesnt_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) :=
sorry

#check machine_doesnt_require_repair

end machine_doesnt_require_repair_l627_62794


namespace extremum_implies_not_monotonic_l627_62743

open Set
open Function

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define differentiability
variable (h_diff : Differentiable ℝ f)

-- Define the existence of an extremum
variable (h_extremum : ∃ x₀ : ℝ, IsLocalExtremum ℝ f x₀)

-- Theorem statement
theorem extremum_implies_not_monotonic :
  ¬(StrictMono f ∨ StrictAnti f) :=
sorry

end extremum_implies_not_monotonic_l627_62743


namespace oil_price_reduction_l627_62767

/-- Represents the price reduction problem for oil -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_price * original_quantity = 684) →
  (0.8 * original_price * (original_quantity + 4) = 684) →
  (0.8 * original_price = 34.20) :=
by sorry

end oil_price_reduction_l627_62767
