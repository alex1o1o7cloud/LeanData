import Mathlib

namespace eighth_power_sum_l316_31670

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^8 + b^8 = 47 := by
  sorry

end eighth_power_sum_l316_31670


namespace stair_climbing_time_l316_31612

theorem stair_climbing_time (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 15 → d = 10 → n = 4 →
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 120 := by
  sorry

end stair_climbing_time_l316_31612


namespace polynomial_identity_l316_31685

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x : ℝ, P (x^3 - 2) = P x^3 - 2) ↔ (∀ x : ℝ, P x = x) :=
by sorry

end polynomial_identity_l316_31685


namespace count_possible_sums_l316_31601

def bag_A : Finset ℕ := {0, 1, 3, 5}
def bag_B : Finset ℕ := {0, 2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem count_possible_sums : possible_sums.card = 10 := by
  sorry

end count_possible_sums_l316_31601


namespace b1f_hex_to_dec_l316_31613

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

theorem b1f_hex_to_dec :
  hex_string_to_dec "B1F" = 2847 := by
  sorry


end b1f_hex_to_dec_l316_31613


namespace quadratic_equation_roots_l316_31675

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 - (a^2 - 2*a - 15)*x + (a - 1) = 0 ∧ 
               y^2 - (a^2 - 2*a - 15)*y + (a - 1) = 0 ∧ 
               x = -y) → 
  a = -3 := by
sorry

end quadratic_equation_roots_l316_31675


namespace susan_remaining_moves_l316_31665

/-- Represents the board game with 100 spaces -/
def BoardGame := 100

/-- Susan's movements over 7 turns -/
def susanMoves : List ℤ := [15, 2, 20, 0, 2, 0, 12]

/-- The total distance Susan has moved -/
def totalDistance : ℤ := susanMoves.sum

/-- Theorem: Susan needs to move 49 more spaces to reach the end -/
theorem susan_remaining_moves : BoardGame - totalDistance = 49 := by
  sorry

end susan_remaining_moves_l316_31665


namespace polynomial_factorization_l316_31629

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end polynomial_factorization_l316_31629


namespace probability_two_odd_numbers_l316_31692

/-- A fair eight-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The set of odd numbers on an eight-sided die -/
def OddNumbers : Finset ℕ := Finset.filter (fun x => x % 2 = 1) EightSidedDie

/-- The probability of an event occurring when rolling two fair eight-sided dice -/
def probability (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / (EightSidedDie.card * EightSidedDie.card)

/-- The event of rolling two odd numbers -/
def TwoOddNumbers : Finset (ℕ × ℕ) :=
  Finset.product OddNumbers OddNumbers

theorem probability_two_odd_numbers : probability TwoOddNumbers = 1 / 4 := by
  sorry

end probability_two_odd_numbers_l316_31692


namespace pinedale_bus_distance_l316_31607

theorem pinedale_bus_distance (average_speed : ℝ) (stop_interval : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60) 
  (h2 : stop_interval = 5 / 60) 
  (h3 : num_stops = 8) : 
  average_speed * (stop_interval * num_stops) = 40 := by
  sorry

end pinedale_bus_distance_l316_31607


namespace hcf_problem_l316_31639

theorem hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b % 11 = 0) 
  (h2 : Nat.lcm a b % 12 = 0) (h3 : max a b = 480) : Nat.gcd a b = 40 := by
  sorry

end hcf_problem_l316_31639


namespace midpoint_fraction_l316_31672

theorem midpoint_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (3 : ℚ) / 4 / 2 + (5 : ℚ) / 6 / 2 ∧ n = 19 ∧ d = 24 := by
  sorry

end midpoint_fraction_l316_31672


namespace josh_pencils_calculation_l316_31697

/-- The number of pencils Josh had initially -/
def initial_pencils : ℕ := 142

/-- The number of pencils Josh gave away -/
def pencils_given_away : ℕ := 31

/-- The number of pencils Josh is left with -/
def remaining_pencils : ℕ := initial_pencils - pencils_given_away

theorem josh_pencils_calculation : remaining_pencils = 111 := by
  sorry

end josh_pencils_calculation_l316_31697


namespace three_digit_sum_property_l316_31610

theorem three_digit_sum_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧
  (∃ x y z : ℕ, 
    n = 100 * x + 10 * y + z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = y + x^2 + z^3) ∧
  n = 357 :=
by sorry

end three_digit_sum_property_l316_31610


namespace complex_number_quadrant_l316_31693

theorem complex_number_quadrant (z : ℂ) : z * (1 - Complex.I) = (1 + 2 * Complex.I) * Complex.I →
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
by sorry

end complex_number_quadrant_l316_31693


namespace adoption_time_l316_31634

def initial_puppies : ℕ := 3
def new_puppies : ℕ := 3
def adoption_rate : ℕ := 3

theorem adoption_time :
  (initial_puppies + new_puppies) / adoption_rate = 2 :=
sorry

end adoption_time_l316_31634


namespace percent_equation_l316_31678

theorem percent_equation (x y : ℝ) (P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 15 := by
  sorry

end percent_equation_l316_31678


namespace progression_check_l316_31606

theorem progression_check (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 6) (hc : c = 4.5) :
  (∃ (r m : ℝ), (b / a) ^ r = (c / a) ^ m) ∧
  ¬(∃ (d : ℝ), b - a = c - b) :=
by sorry

end progression_check_l316_31606


namespace parallel_iff_m_eq_four_l316_31691

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (-((3 * m - 4) / 4) = -(m / 2))

/-- The condition for parallelism is equivalent to m = 4 -/
theorem parallel_iff_m_eq_four :
  ∀ m : ℝ, parallel_lines m ↔ m = 4 := by sorry

end parallel_iff_m_eq_four_l316_31691


namespace faucet_filling_time_l316_31631

/-- Given that four faucets can fill a 120-gallon tub in 5 minutes,
    prove that two faucets can fill a 60-gallon tub in 5 minutes. -/
theorem faucet_filling_time 
  (tub_capacity : ℝ) 
  (filling_time : ℝ) 
  (faucet_count : ℕ) :
  tub_capacity = 120 ∧ 
  filling_time = 5 ∧ 
  faucet_count = 4 →
  ∃ (new_tub_capacity : ℝ) (new_faucet_count : ℕ),
    new_tub_capacity = 60 ∧
    new_faucet_count = 2 ∧
    (new_tub_capacity / new_faucet_count) / (tub_capacity / faucet_count) * filling_time = 5 :=
by sorry

end faucet_filling_time_l316_31631


namespace constant_function_value_l316_31659

theorem constant_function_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end constant_function_value_l316_31659


namespace multiples_properties_l316_31683

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) := by
sorry

end multiples_properties_l316_31683


namespace infinite_geometric_series_ratio_l316_31640

theorem infinite_geometric_series_ratio (a S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end infinite_geometric_series_ratio_l316_31640


namespace complement_union_theorem_l316_31694

def I : Set Nat := {1, 2, 3, 4}
def S : Set Nat := {1, 3}
def T : Set Nat := {4}

theorem complement_union_theorem :
  (I \ S) ∪ T = {2, 4} := by sorry

end complement_union_theorem_l316_31694


namespace circle_diameter_ratio_l316_31682

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end circle_diameter_ratio_l316_31682


namespace pyramid_height_equals_cube_volume_l316_31652

theorem pyramid_height_equals_cube_volume (cube_edge : Real) (pyramid_base : Real) (pyramid_height : Real) : 
  cube_edge = 6 →
  pyramid_base = 12 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 4.5 := by
sorry

end pyramid_height_equals_cube_volume_l316_31652


namespace four_digit_number_with_zero_removal_l316_31687

/-- Represents a four-digit number with one digit being zero -/
structure FourDigitNumberWithZero where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10
  has_one_zero : (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0)

/-- The value of the four-digit number -/
def value (n : FourDigitNumberWithZero) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the number after removing the zero -/
def valueWithoutZero (n : FourDigitNumberWithZero) : Nat :=
  if n.a = 0 then 100 * n.b + 10 * n.c + n.d
  else if n.b = 0 then 100 * n.a + 10 * n.c + n.d
  else if n.c = 0 then 100 * n.a + 10 * n.b + n.d
  else 100 * n.a + 10 * n.b + n.c

theorem four_digit_number_with_zero_removal (n : FourDigitNumberWithZero) :
  (value n = 9 * valueWithoutZero n) → (value n = 2025 ∨ value n = 6075) := by
  sorry


end four_digit_number_with_zero_removal_l316_31687


namespace surface_is_cone_l316_31674

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of the surface --/
def surfaceEquation (p : Point3D) (a b c d θ : ℝ) : Prop :=
  (p.x - a)^2 + (p.y - b)^2 + (p.z - c)^2 = (d * Real.cos θ)^2

/-- The set of points satisfying the equation --/
def surfaceSet (a b c d : ℝ) : Set Point3D :=
  {p : Point3D | ∃ θ, surfaceEquation p a b c d θ}

/-- Definition of a cone --/
def isCone (S : Set Point3D) : Prop :=
  ∃ v : Point3D, ∃ axis : Point3D → Point3D → Prop,
    ∀ p ∈ S, ∃ r θ : ℝ, r ≥ 0 ∧ 
      p = Point3D.mk (v.x + r * Real.cos θ) (v.y + r * Real.sin θ) (v.z + r)

theorem surface_is_cone (d : ℝ) (h : d > 0) :
  isCone (surfaceSet 0 0 0 d) := by
  sorry

end surface_is_cone_l316_31674


namespace map_distance_conversion_l316_31669

/-- Given a map scale where 1 inch represents 500 meters, 
    this theorem proves that a line segment of 7.25 inches 
    on the map represents 3625 meters in reality. -/
theorem map_distance_conversion 
  (scale : ℝ) 
  (map_length : ℝ) 
  (h1 : scale = 500) 
  (h2 : map_length = 7.25) : 
  map_length * scale = 3625 := by
sorry

end map_distance_conversion_l316_31669


namespace last_digit_of_tower_of_power_l316_31661

def tower_of_power (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n + 1 => 2^(tower_of_power n)

theorem last_digit_of_tower_of_power :
  tower_of_power 2007 % 10 = 6 :=
by sorry

end last_digit_of_tower_of_power_l316_31661


namespace carla_counting_theorem_l316_31666

theorem carla_counting_theorem (ceiling_tiles : ℕ) (books : ℕ) 
  (h1 : ceiling_tiles = 38) 
  (h2 : books = 75) : 
  ceiling_tiles * 2 + books * 3 = 301 := by
  sorry

end carla_counting_theorem_l316_31666


namespace raccoon_stall_time_l316_31690

/-- Proves that the time both locks together stall the raccoons is 60 minutes -/
theorem raccoon_stall_time : ∀ (t1 t2 t_both : ℕ),
  t1 = 5 →
  t2 = 3 * t1 - 3 →
  t_both = 5 * t2 →
  t_both = 60 := by
  sorry

end raccoon_stall_time_l316_31690


namespace mica_pasta_purchase_l316_31651

theorem mica_pasta_purchase (pasta_price : ℝ) (beef_price : ℝ) (sauce_price : ℝ) 
  (quesadilla_price : ℝ) (total_budget : ℝ) :
  pasta_price = 1.5 →
  beef_price = 8 →
  sauce_price = 2 →
  quesadilla_price = 6 →
  total_budget = 15 →
  (total_budget - (beef_price * 0.25 + sauce_price * 2 + quesadilla_price)) / pasta_price = 2 :=
by
  sorry

#check mica_pasta_purchase

end mica_pasta_purchase_l316_31651


namespace product_remainder_one_mod_three_l316_31605

theorem product_remainder_one_mod_three (a b : ℕ) :
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
  sorry

end product_remainder_one_mod_three_l316_31605


namespace farmer_land_area_l316_31649

theorem farmer_land_area : ∃ (total : ℚ),
  total > 0 ∧
  total / 3 + total / 4 + total / 5 + 26 = total ∧
  total = 120 := by
  sorry

end farmer_land_area_l316_31649


namespace total_jokes_over_two_saturdays_l316_31638

/-- 
Given that Jessy told 11 jokes and Alan told 7 jokes on the first Saturday,
and they both double their jokes on the second Saturday, prove that the
total number of jokes told over both Saturdays is 54.
-/
theorem total_jokes_over_two_saturdays 
  (jessy_first : ℕ) 
  (alan_first : ℕ) 
  (h1 : jessy_first = 11) 
  (h2 : alan_first = 7) : 
  jessy_first + alan_first + 2 * jessy_first + 2 * alan_first = 54 := by
  sorry

end total_jokes_over_two_saturdays_l316_31638


namespace problem_statement_l316_31621

theorem problem_statement (m n : ℤ) : 
  |m - 2023| + (n + 2024)^2 = 0 → (m + n)^2023 = -1 := by
sorry

end problem_statement_l316_31621


namespace max_value_sqrt_sum_l316_31611

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -2/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -2) :
  ∃ (max : ℝ), max = Real.sqrt 57 ∧ 
    ∀ a b c : ℝ, a + b + c = 2 → a ≥ -2/3 → b ≥ -1 → c ≥ -2 →
      Real.sqrt (3*a + 2) + Real.sqrt (3*b + 4) + Real.sqrt (3*c + 7) ≤ max :=
by
  sorry

end max_value_sqrt_sum_l316_31611


namespace no_distinct_cube_sum_equality_l316_31620

theorem no_distinct_cube_sum_equality (a b c d : ℕ) :
  a^3 + b^3 = c^3 + d^3 → a + b = c + d → ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end no_distinct_cube_sum_equality_l316_31620


namespace share_difference_after_tax_l316_31650

/-- Represents the share ratios for p, q, r, and s respectively -/
def shareRatios : Fin 4 → ℕ
  | 0 => 3
  | 1 => 7
  | 2 => 12
  | 3 => 5

/-- Represents the tax rates for p, q, r, and s respectively -/
def taxRates : Fin 4 → ℚ
  | 0 => 1/10
  | 1 => 15/100
  | 2 => 1/5
  | 3 => 1/4

/-- The difference between p and q's shares after tax deduction -/
def differenceAfterTax : ℚ := 2400

theorem share_difference_after_tax :
  let x : ℚ := differenceAfterTax / (shareRatios 1 * (1 - taxRates 1) - shareRatios 0 * (1 - taxRates 0))
  let qShare : ℚ := shareRatios 1 * x * (1 - taxRates 1)
  let rShare : ℚ := shareRatios 2 * x * (1 - taxRates 2)
  abs (rShare - qShare - 2695.38) < 0.01 := by
  sorry

end share_difference_after_tax_l316_31650


namespace third_vertex_y_coordinate_l316_31616

/-- An equilateral triangle with two vertices at (3, 4) and (13, 4), and the third vertex in the first quadrant -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  h1 : v1 = (3, 4)
  h2 : v2 = (13, 4)
  h3 : v3.1 > 0 ∧ v3.2 > 0  -- First quadrant condition
  h4 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v1.1 - v3.1)^2 + (v1.2 - v3.2)^2
  h5 : (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (v2.1 - v3.1)^2 + (v2.2 - v3.2)^2

/-- The y-coordinate of the third vertex is 4 + 5√3 -/
theorem third_vertex_y_coordinate (t : EquilateralTriangle) : t.v3.2 = 4 + 5 * Real.sqrt 3 := by
  sorry

end third_vertex_y_coordinate_l316_31616


namespace line_direction_vector_l316_31645

def point := ℝ × ℝ

-- Define the two points on the line
def p1 : point := (-3, 4)
def p2 : point := (2, -1)

-- Define the direction vector type
def direction_vector := ℝ × ℝ

-- Function to calculate the direction vector between two points
def calc_direction_vector (p q : point) : direction_vector :=
  (q.1 - p.1, q.2 - p.2)

-- Function to scale a vector
def scale_vector (v : direction_vector) (s : ℝ) : direction_vector :=
  (s * v.1, s * v.2)

-- Theorem statement
theorem line_direction_vector : 
  ∃ (a : ℝ), calc_direction_vector p1 p2 = scale_vector (a, 2) (-5/2) ∧ a = -2 := by
  sorry

end line_direction_vector_l316_31645


namespace bee_travel_distance_l316_31608

theorem bee_travel_distance (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (speed_bee : ℝ)
  (h1 : initial_distance = 120)
  (h2 : speed_A = 30)
  (h3 : speed_B = 10)
  (h4 : speed_bee = 60) :
  let relative_speed := speed_A + speed_B
  let meeting_time := initial_distance / relative_speed
  speed_bee * meeting_time = 180 := by
sorry

end bee_travel_distance_l316_31608


namespace division_problem_l316_31602

theorem division_problem :
  ∃! x : ℕ, x < 50 ∧ ∃ m : ℕ, 100 = m * x + 6 :=
by
  -- The proof goes here
  sorry

end division_problem_l316_31602


namespace group_size_from_circular_arrangements_l316_31647

/-- The number of ways to arrange k people from a group of n people around a circular table. -/
def circularArrangements (n k : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: If there are 144 ways to seat 5 people around a circular table from a group of n people, then n = 7. -/
theorem group_size_from_circular_arrangements (n : ℕ) 
  (h : circularArrangements n 5 = 144) : n = 7 := by
  sorry

end group_size_from_circular_arrangements_l316_31647


namespace simplify_fraction_product_l316_31615

theorem simplify_fraction_product : 
  10 * (15 / 8) * (-28 / 45) * (3 / 5) = -7 / 4 := by
  sorry

end simplify_fraction_product_l316_31615


namespace twenty_men_handshakes_l316_31696

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 men, the maximum number of handshakes without cyclic handshakes is 190 -/
theorem twenty_men_handshakes :
  max_handshakes 20 = 190 := by
  sorry

#eval max_handshakes 20  -- This will evaluate to 190

end twenty_men_handshakes_l316_31696


namespace point_on_line_line_slope_is_one_line_equation_correct_l316_31698

/-- A line passing through the point (1, 3) with slope 1 -/
def line (x y : ℝ) : Prop := x - y + 2 = 0

/-- The point (1, 3) lies on the line -/
theorem point_on_line : line 1 3 := by sorry

/-- The slope of the line is 1 -/
theorem line_slope_is_one :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → line x₁ y₁ → line x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 1 := by sorry

/-- The equation x - y + 2 = 0 represents the unique line passing through (1, 3) with slope 1 -/
theorem line_equation_correct :
  ∀ (x y : ℝ), (x - y + 2 = 0) ↔ (∃ (m b : ℝ), m = 1 ∧ y = m * (x - 1) + 3) := by sorry

end point_on_line_line_slope_is_one_line_equation_correct_l316_31698


namespace division_problem_l316_31635

theorem division_problem (A : ℕ) (h : 23 = A * 3 + 2) : A = 7 := by
  sorry

end division_problem_l316_31635


namespace may_has_greatest_percentage_difference_l316_31644

/-- Represents the sales data for a single month --/
structure MonthSales where
  drummers : ℕ
  bugles : ℕ
  flutes : ℕ

/-- Calculates the percentage difference for a given month's sales --/
def percentageDifference (sales : MonthSales) : ℚ :=
  let max := max sales.drummers (max sales.bugles sales.flutes)
  let min := min sales.drummers (min sales.bugles sales.flutes)
  (max - min : ℚ) / min * 100

/-- Sales data for each month --/
def januarySales : MonthSales := ⟨5, 4, 6⟩
def februarySales : MonthSales := ⟨6, 5, 6⟩
def marchSales : MonthSales := ⟨6, 6, 6⟩
def aprilSales : MonthSales := ⟨7, 5, 8⟩
def maySales : MonthSales := ⟨3, 5, 4⟩

/-- Theorem: May has the greatest percentage difference in sales --/
theorem may_has_greatest_percentage_difference :
  percentageDifference maySales > percentageDifference januarySales ∧
  percentageDifference maySales > percentageDifference februarySales ∧
  percentageDifference maySales > percentageDifference marchSales ∧
  percentageDifference maySales > percentageDifference aprilSales :=
by sorry


end may_has_greatest_percentage_difference_l316_31644


namespace function_inequality_implies_unique_a_l316_31627

theorem function_inequality_implies_unique_a :
  ∀ (a : ℝ),
  (∀ (x : ℝ), Real.exp x + a * (x^2 - x) - Real.cos x ≥ 0) →
  a = 1 := by
sorry

end function_inequality_implies_unique_a_l316_31627


namespace x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l316_31630

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l316_31630


namespace total_volume_is_716_l316_31667

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The number of cubes Carl has -/
def carl_cubes : ℕ := 8

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of cubes Kate has -/
def kate_cubes : ℕ := 4

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 5

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_716 : total_volume = 716 := by
  sorry

end total_volume_is_716_l316_31667


namespace olivia_checking_time_l316_31624

def time_spent_checking (num_problems : ℕ) (time_per_problem : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (num_problems * time_per_problem)

theorem olivia_checking_time :
  time_spent_checking 7 4 31 = 3 := by sorry

end olivia_checking_time_l316_31624


namespace stating_external_diagonals_inequality_invalid_external_diagonals_l316_31632

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- 
Theorem stating that for a valid set of external diagonal lengths of a right regular prism,
the sum of squares of the two smaller lengths is greater than or equal to 
the square of the largest length.
-/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a ^ 2 + d.b ^ 2 ≥ d.c ^ 2 := by
  sorry

/-- 
Proves that {6, 8, 11} cannot be the lengths of external diagonals of a right regular prism
-/
theorem invalid_external_diagonals : 
  ¬∃ (d : ExternalDiagonals), d.a = 6 ∧ d.b = 8 ∧ d.c = 11 := by
  sorry

end stating_external_diagonals_inequality_invalid_external_diagonals_l316_31632


namespace nabla_neg_five_neg_seven_l316_31657

def nabla (a b : ℝ) : ℝ := a * b + a - b

theorem nabla_neg_five_neg_seven : nabla (-5) (-7) = 37 := by
  sorry

end nabla_neg_five_neg_seven_l316_31657


namespace a_eq_4_neither_sufficient_nor_necessary_l316_31671

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line l₁: ax + 8y - 8 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := a, b := 8, c := -8 }

/-- The second line l₂: 2x + ay - a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := 2, b := a, c := -a }

/-- The main theorem stating that a = 4 is neither sufficient nor necessary for parallelism -/
theorem a_eq_4_neither_sufficient_nor_necessary :
  ∃ a : ℝ, a ≠ 4 ∧ parallel (l1 a) (l2 a) ∧
  ∃ b : ℝ, b = 4 ∧ ¬parallel (l1 b) (l2 b) :=
sorry

end a_eq_4_neither_sufficient_nor_necessary_l316_31671


namespace divisible_by_three_l316_31676

def five_digit_number (n : Nat) : Nat :=
  52000 + n * 100 + 48

theorem divisible_by_three (n : Nat) : 
  n < 10 → (five_digit_number n % 3 = 0 ↔ n = 2) := by
  sorry

end divisible_by_three_l316_31676


namespace pascal_row10_sums_l316_31668

/-- Represents a row in Pascal's Triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The 10th row of Pascal's Triangle -/
def row10 : PascalRow 10 := sorry

/-- Sum of elements in a Pascal's Triangle row -/
def row_sum (n : ℕ) (row : PascalRow n) : ℕ := sorry

/-- Sum of squares of elements in a Pascal's Triangle row -/
def row_sum_of_squares (n : ℕ) (row : PascalRow n) : ℕ := sorry

theorem pascal_row10_sums :
  (row_sum 10 row10 = 2^10) ∧
  (row_sum_of_squares 10 row10 = 183756) := by sorry

end pascal_row10_sums_l316_31668


namespace garden_flowers_l316_31636

theorem garden_flowers (roses tulips : ℕ) (percent_not_roses : ℚ) (total daisies : ℕ) : 
  roses = 25 →
  tulips = 40 →
  percent_not_roses = 3/4 →
  total = roses + tulips + daisies →
  (total : ℚ) * (1 - percent_not_roses) = roses →
  daisies = 35 :=
by sorry

end garden_flowers_l316_31636


namespace arithmetic_sequence_fifth_term_l316_31625

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_21st : a 21 = 26)
  (h_22nd : a 22 = 30) :
  a 5 = -38 :=
sorry

end arithmetic_sequence_fifth_term_l316_31625


namespace student_selection_probability_l316_31643

theorem student_selection_probability (b g o : ℝ) : 
  b + g + o = 1 →  -- total probability
  b > 0 ∧ g > 0 ∧ o > 0 →  -- probabilities are positive
  b = (1/2) * o →  -- boy probability is half of other
  g = o - b →  -- girl probability is difference between other and boy
  b = 1/4 :=  -- ratio of boys to total is 1/4
by sorry

end student_selection_probability_l316_31643


namespace negative_abs_equal_l316_31654

theorem negative_abs_equal : -|5| = -|-5| := by
  sorry

end negative_abs_equal_l316_31654


namespace bicycle_trip_speed_l316_31646

/-- Proves that given a total distance of 400 km, with the first 100 km traveled at 20 km/h
    and an average speed of 16 km/h for the entire trip, the speed for the remainder of the trip is 15 km/h. -/
theorem bicycle_trip_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
  (h1 : total_distance = 400)
  (h2 : first_part_distance = 100)
  (h3 : first_part_speed = 20)
  (h4 : average_speed = 16) :
  let remainder_distance := total_distance - first_part_distance
  let total_time := total_distance / average_speed
  let first_part_time := first_part_distance / first_part_speed
  let remainder_time := total_time - first_part_time
  remainder_distance / remainder_time = 15 :=
by sorry

end bicycle_trip_speed_l316_31646


namespace max_value_of_quadratic_l316_31656

/-- The quadratic function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The range of x values we're considering -/
def range : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }

theorem max_value_of_quadratic :
  ∃ (m : ℝ), m = 36 ∧ ∀ x ∈ range, f x ≤ m :=
sorry

end max_value_of_quadratic_l316_31656


namespace perimeter_is_120_inches_l316_31664

/-- The perimeter of a figure formed by cutting an equilateral triangle from a square and rotating it -/
def rotated_triangle_perimeter (square_side : ℝ) (triangle_side : ℝ) : ℝ :=
  3 * square_side + 3 * triangle_side

/-- Theorem: The perimeter of the new figure is 120 inches -/
theorem perimeter_is_120_inches :
  let square_side := (20 : ℝ)
  let triangle_side := (20 : ℝ)
  rotated_triangle_perimeter square_side triangle_side = 120 := by
  sorry

#eval rotated_triangle_perimeter 20 20

end perimeter_is_120_inches_l316_31664


namespace jelly_cost_l316_31653

theorem jelly_cost (N B J : ℕ) (h1 : N = 15) 
  (h2 : 6 * B * N + 7 * J * N = 315) 
  (h3 : B > 0) (h4 : J > 0) : 
  7 * J * N / 100 = 315 / 100 := by
  sorry

end jelly_cost_l316_31653


namespace x_value_proof_l316_31695

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (144 : ℝ)^(1/7) :=
by sorry

end x_value_proof_l316_31695


namespace neighborhood_households_l316_31663

theorem neighborhood_households (no_car_no_bike : ℕ) (car_and_bike : ℕ) (total_with_car : ℕ) (bike_only : ℕ) :
  no_car_no_bike = 11 →
  car_and_bike = 18 →
  total_with_car = 44 →
  bike_only = 35 →
  no_car_no_bike + car_and_bike + (total_with_car - car_and_bike) + bike_only = 90 :=
by sorry

end neighborhood_households_l316_31663


namespace milk_container_percentage_difference_l316_31686

/-- Given a scenario where milk is transferred between containers, this theorem proves
    the percentage difference between the quantity in one container and the original capacity. -/
theorem milk_container_percentage_difference
  (total_milk : ℝ)
  (transfer_amount : ℝ)
  (h_total : total_milk = 1216)
  (h_transfer : transfer_amount = 152)
  (h_equal_after_transfer : ∃ (b c : ℝ), b + c = total_milk ∧ b + transfer_amount = c - transfer_amount) :
  ∃ (b : ℝ), (total_milk - b) / total_milk * 100 = 56.25 := by
  sorry

#eval (1216 - 532) / 1216 * 100  -- Should output approximately 56.25

end milk_container_percentage_difference_l316_31686


namespace union_of_A_and_B_l316_31660

def A : Set ℤ := {0, 1}
def B : Set ℤ := {0, -1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end union_of_A_and_B_l316_31660


namespace shapes_can_form_both_rectangles_l316_31614

/-- Represents a pentagon -/
structure Pentagon where
  area : ℝ

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a set of shapes consisting of two pentagons and a triangle -/
structure ShapeSet where
  pentagon1 : Pentagon
  pentagon2 : Pentagon
  triangle : Triangle

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a set of shapes can form a given rectangle -/
def can_form_rectangle (shapes : ShapeSet) (rect : Rectangle) : Prop :=
  shapes.pentagon1.area + shapes.pentagon2.area + shapes.triangle.area = rect.width * rect.height

/-- The main theorem stating that it's possible to have a set of shapes
    that can form both a 4x6 and a 3x8 rectangle -/
theorem shapes_can_form_both_rectangles :
  ∃ (shapes : ShapeSet),
    can_form_rectangle shapes (Rectangle.mk 4 6) ∧
    can_form_rectangle shapes (Rectangle.mk 3 8) := by
  sorry

end shapes_can_form_both_rectangles_l316_31614


namespace no_k_for_always_negative_quadratic_l316_31684

theorem no_k_for_always_negative_quadratic :
  ¬ ∃ k : ℝ, ∀ x : ℝ, x^2 - (k + 4) * x + k - 3 < 0 := by
  sorry

end no_k_for_always_negative_quadratic_l316_31684


namespace diamond_four_three_l316_31642

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^3

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end diamond_four_three_l316_31642


namespace complement_of_intersection_in_S_l316_31603

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}

theorem complement_of_intersection_in_S :
  (S \ (S ∩ T)) = {2} := by sorry

end complement_of_intersection_in_S_l316_31603


namespace remainder_not_always_same_l316_31626

theorem remainder_not_always_same (a b : ℕ) :
  (3 * a + b) % 10 = (3 * b + a) % 10 →
  ¬(a % 10 = b % 10) :=
by sorry

end remainder_not_always_same_l316_31626


namespace christmas_tree_lights_l316_31617

theorem christmas_tree_lights (red : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 26)
  (h_yellow : yellow = 37)
  (h_blue : blue = 32) :
  red + yellow + blue = 95 := by
  sorry

end christmas_tree_lights_l316_31617


namespace eight_team_tournament_l316_31600

/-- The number of matches in a single-elimination tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 8 teams requires 7 matches -/
theorem eight_team_tournament : num_matches 8 = 7 := by
  sorry

end eight_team_tournament_l316_31600


namespace profit_per_meter_l316_31679

/-- Given a cloth sale scenario, calculate the profit per meter. -/
theorem profit_per_meter
  (meters_sold : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : meters_sold = 60)
  (h2 : total_selling_price = 8400)
  (h3 : cost_price_per_meter = 128) :
  (total_selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 := by
sorry


end profit_per_meter_l316_31679


namespace opposite_of_negative_one_half_l316_31637

theorem opposite_of_negative_one_half : 
  ∃ (x : ℚ), x + (-1/2) = 0 ∧ x = 1/2 := by
  sorry

end opposite_of_negative_one_half_l316_31637


namespace solution_set_implies_a_range_l316_31662

theorem solution_set_implies_a_range (a : ℝ) : 
  (∃ P : Set ℝ, (∀ x ∈ P, (x + 1) / (x + a) < 2) ∧ 1 ∉ P) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end solution_set_implies_a_range_l316_31662


namespace pedros_daughters_l316_31623

/-- The number of ice cream flavors available -/
def num_flavors : ℕ := 12

/-- The number of scoops in each child's combo -/
def scoops_per_combo : ℕ := 3

/-- The total number of scoops ordered for each flavor -/
def scoops_per_flavor : ℕ := 2

structure Family where
  num_boys : ℕ
  num_girls : ℕ

/-- Pedro's family satisfies the given conditions -/
def is_valid_family (f : Family) : Prop :=
  f.num_boys > 0 ∧ 
  f.num_girls > f.num_boys ∧
  (f.num_boys + f.num_girls) * scoops_per_combo = num_flavors * scoops_per_flavor ∧
  ∃ (boys_flavors girls_flavors : Finset ℕ), 
    boys_flavors.card = (3 * f.num_boys) / 2 ∧
    girls_flavors.card = (3 * f.num_girls) / 2 ∧
    boys_flavors ∩ girls_flavors = ∅ ∧
    boys_flavors ∪ girls_flavors = Finset.range num_flavors

theorem pedros_daughters (f : Family) (h : is_valid_family f) : f.num_girls = 6 :=
sorry

end pedros_daughters_l316_31623


namespace buoy_radius_l316_31641

/-- The radius of a buoy given the dimensions of the hole it leaves -/
theorem buoy_radius (hole_width : ℝ) (hole_depth : ℝ) : 
  hole_width = 30 → hole_depth = 10 → ∃ r : ℝ, r = 16.25 ∧ 
  ∃ x : ℝ, x^2 + (hole_width/2)^2 = (x + hole_depth)^2 ∧ r = x + hole_depth :=
by sorry

end buoy_radius_l316_31641


namespace circles_externally_tangent_l316_31673

def circle1_center : ℝ × ℝ := (-32, 42)
def circle2_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 52
def circle2_radius : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)
  d = circle1_radius + circle2_radius := by sorry

end circles_externally_tangent_l316_31673


namespace probability_is_one_third_l316_31622

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x > 2y --/
def probability_x_gt_2y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 1
  h_x := by norm_num
  h_y := by norm_num
}

theorem probability_is_one_third :
  probability_x_gt_2y problem_rectangle = 1/3 := by
  sorry

end probability_is_one_third_l316_31622


namespace max_value_of_f_l316_31688

def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end max_value_of_f_l316_31688


namespace technicians_avg_salary_is_900_l316_31633

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary_all : ℕ
  num_technicians : ℕ
  avg_salary_non_tech : ℕ

/-- Calculates the average salary of technicians given workshop data -/
def avg_salary_technicians (w : Workshop) : ℕ :=
  let total_salary := w.total_workers * w.avg_salary_all
  let non_tech_workers := w.total_workers - w.num_technicians
  let non_tech_salary := non_tech_workers * w.avg_salary_non_tech
  let tech_salary := total_salary - non_tech_salary
  tech_salary / w.num_technicians

/-- Theorem stating that the average salary of technicians is 900 given the workshop conditions -/
theorem technicians_avg_salary_is_900 (w : Workshop) 
  (h1 : w.total_workers = 20)
  (h2 : w.avg_salary_all = 750)
  (h3 : w.num_technicians = 5)
  (h4 : w.avg_salary_non_tech = 700) :
  avg_salary_technicians w = 900 := by
  sorry

#eval avg_salary_technicians ⟨20, 750, 5, 700⟩

end technicians_avg_salary_is_900_l316_31633


namespace amount_per_bulb_is_fifty_cents_l316_31680

/-- The amount Jane earned for planting flower bulbs -/
def total_earned : ℚ := 75

/-- The number of tulip bulbs Jane planted -/
def tulip_bulbs : ℕ := 20

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

/-- The number of iris bulbs Jane planted -/
def iris_bulbs : ℕ := tulip_bulbs / 2

/-- The number of crocus bulbs Jane planted -/
def crocus_bulbs : ℕ := 3 * daffodil_bulbs

/-- The total number of bulbs Jane planted -/
def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

/-- The amount paid per bulb -/
def amount_per_bulb : ℚ := total_earned / total_bulbs

theorem amount_per_bulb_is_fifty_cents : amount_per_bulb = 1/2 := by
  sorry

end amount_per_bulb_is_fifty_cents_l316_31680


namespace c_is_winner_l316_31677

-- Define the candidates
inductive Candidate
| A
| B
| C

-- Define the election result
structure ElectionResult where
  total_voters : Nat
  votes : Candidate → Nat
  vote_count_valid : votes Candidate.A + votes Candidate.B + votes Candidate.C = total_voters

-- Define the winner selection rule
def is_winner (result : ElectionResult) (c : Candidate) : Prop :=
  ∀ other : Candidate, result.votes c ≥ result.votes other

-- Theorem statement
theorem c_is_winner (result : ElectionResult) 
  (h_total : result.total_voters = 30)
  (h_a : result.votes Candidate.A = 12)
  (h_b : result.votes Candidate.B = 3)
  (h_c : result.votes Candidate.C = 15) :
  is_winner result Candidate.C :=
by sorry

end c_is_winner_l316_31677


namespace arithmetic_sequence_with_difference_two_l316_31689

def a (n : ℕ) : ℝ := 2 * (n + 1) + 3

theorem arithmetic_sequence_with_difference_two :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end arithmetic_sequence_with_difference_two_l316_31689


namespace complex_equal_modulus_unequal_square_exists_l316_31681

theorem complex_equal_modulus_unequal_square_exists : 
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end complex_equal_modulus_unequal_square_exists_l316_31681


namespace fourth_pill_time_l316_31604

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

/-- The time interval between pills in minutes -/
def pillInterval : Nat := 75

/-- The starting time when the first pill is taken -/
def startTime : Time := ⟨11, 5, by sorry, by sorry⟩

/-- The number of pills taken -/
def pillCount : Nat := 4

theorem fourth_pill_time :
  addMinutes startTime ((pillCount - 1) * pillInterval) = ⟨14, 50, by sorry, by sorry⟩ :=
sorry

end fourth_pill_time_l316_31604


namespace intersection_inequality_solution_l316_31699

/-- Given two lines y = 3x + a and y = -2x + b that intersect at a point with x-coordinate -5,
    the solution set of the inequality 3x + a < -2x + b is {x ∈ ℝ | x < -5}. -/
theorem intersection_inequality_solution (a b : ℝ) :
  (∃ y, 3 * (-5) + a = y ∧ -2 * (-5) + b = y) →
  (∀ x, 3 * x + a < -2 * x + b ↔ x < -5) :=
by sorry

end intersection_inequality_solution_l316_31699


namespace complex_absolute_value_l316_31609

open Complex

theorem complex_absolute_value : ∀ (i : ℂ), i * i = -1 → Complex.abs (2 * i * (1 - 2 * i)) = 2 * Real.sqrt 5 := by
  sorry

end complex_absolute_value_l316_31609


namespace total_games_is_32_l316_31658

/-- The number of games won by Jerry -/
def jerry_games : ℕ := 7

/-- The number of games won by Dave -/
def dave_games : ℕ := jerry_games + 3

/-- The number of games won by Ken -/
def ken_games : ℕ := dave_games + 5

/-- The total number of games played -/
def total_games : ℕ := ken_games + dave_games + jerry_games

theorem total_games_is_32 : total_games = 32 := by
  sorry

end total_games_is_32_l316_31658


namespace day_relationship_l316_31628

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the weekday of a given YearDay -/
def weekday_of_yearday : YearDay → Weekday := sorry

/-- Theorem stating the relationship between the given days and their weekdays -/
theorem day_relationship (N : Int) :
  (weekday_of_yearday ⟨N, 250⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N + 1, 150⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N - 1, 50⟩ = Weekday.Saturday) :=
by sorry

end day_relationship_l316_31628


namespace stevens_peaches_l316_31619

-- Define the number of peaches Jake has
def jakes_peaches : ℕ := 7

-- Define the difference in peaches between Steven and Jake
def peach_difference : ℕ := 6

-- Theorem stating that Steven has 13 peaches
theorem stevens_peaches : 
  jakes_peaches + peach_difference = 13 := by sorry

end stevens_peaches_l316_31619


namespace average_rounds_is_four_l316_31618

/-- Represents the distribution of golfers and rounds played --/
structure GolfData :=
  (rounds : Fin 6 → ℕ)
  (golfers : Fin 6 → ℕ)

/-- Calculates the average number of rounds played, rounded to the nearest whole number --/
def averageRoundsRounded (data : GolfData) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => (data.rounds i.succ) * (data.golfers i.succ))
  let totalGolfers := (Finset.range 6).sum (λ i => data.golfers i.succ)
  (totalRounds + totalGolfers / 2) / totalGolfers

/-- The given golf data --/
def givenData : GolfData :=
  { rounds := λ i => i,
    golfers := λ i => match i with
      | 1 => 6
      | 2 => 3
      | 3 => 2
      | 4 => 4
      | 5 => 6
      | 6 => 4 }

theorem average_rounds_is_four :
  averageRoundsRounded givenData = 4 := by
  sorry

end average_rounds_is_four_l316_31618


namespace hyperbola_asymptotes_l316_31655

theorem hyperbola_asymptotes (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (2 * Real.sqrt (-1/m) = 4) →
  (∀ x y : ℝ, y = 2*x ∨ y = -2*x) :=
by sorry

end hyperbola_asymptotes_l316_31655


namespace range_of_m_l316_31648

theorem range_of_m (p : ℝ → Prop) (m : ℝ) 
  (h1 : ∀ x, p x ↔ x^2 + 2*x - m > 0)
  (h2 : ¬ p 1)
  (h3 : p 2) :
  3 ≤ m ∧ m < 8 := by
sorry

end range_of_m_l316_31648
